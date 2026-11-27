//! NVIDIA Nemotron-Labs-Image-30B-A3B: autoregressive text to image on a Nemotron-H language model
//! (the 30B-A3B hybrid Mamba-2 / attention / MoE of Nemotron 3 Nano, [`crate::nemotron`]) whose
//! vocabulary is extended by 196,560 visual tokens, rendered to pixels by the bundled L24SQ ViT
//! tokenizer ([`vit`]).
//!
//! The model card's protocol, followed here:
//!
//! - The prompt is the chat template with thinking off: the card's system prompt, then a user turn
//!   `Generate a {W}x{H} image.\n\n{prompt}`, then `<|im_start|>assistant\n<think></think>`.
//! - The reply is a line break and one image block: `<|img_start|>`, a size header in text (`height=H width=W`),
//!   then for each of the `H / 16` rows `<|vis_row|>` followed by `W / 16` visual tokens
//!   (`<|vis_i|>`, id `131075 + i`, `i` an index of the tokenizer's codebook), then
//!   `<|img_end|>`. A 1024x1024 image is 4,176 tokens.
//! - Sampling at temperature 0.9 with top-p 1 and no top-k; no classifier-free guidance.
//!
//! By default the block's structure is enforced ([`Structure::Grid`]): the header and the row
//! markers are written for the size asked for, and the visual tokens are sampled among the
//! visual vocabulary only, so every generation yields an image of that size (the reference
//! samples everything and rejects a generation whose size or grid is off).
//! [`Structure::Free`] samples every token as the reference does and decodes the grid the model
//! wrote, at the size its header announced.
//!
//! Source images (the model's text-and-image-to-image training, which NVIDIA reports as trained
//! but not evaluated) are encoded by the tokenizer's encoder into the same blocks and placed in
//! the user turn before the instruction.
//!
//! Checkpoints: nvidia/Nemotron-Labs-Image-30B-A3B (float32 safetensors, 130 GB, converted at
//! load to bf16 / `--quant q8` / `q4`) and its llama.cpp GGUF conversions (`nemotron_h_moe` with
//! the 327,680-row vocabulary), whose tokenizer folder is fetched from NVIDIA's repository.

pub mod vit;

use crate::clock::Instant;
use crate::generate::Flow;
use crate::model::Quant;
use crate::nemotron::{NemotronConfig, NemotronModel};
use crate::platform::DevicePref;
use crate::safetensors::ModelFiles;
use crate::sampler::Sampler;
use crate::tokenizer::Tokenizer;
use crate::video::RgbImage;
use anyhow::{Context, Result, anyhow, bail};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use vit::{Precision, Vit, VitConfig};

pub use crate::sd::Image;

/// NVIDIA's repository, which holds the tokenizer folder GGUF conversions leave out.
pub const REPO: &str = "nvidia/Nemotron-Labs-Image-30B-A3B";
/// The tokenizer's folder in [`REPO`].
pub const TOKENIZER_DIR: &str = "Nemotron-Labs-Image-Tokenizer";

/// The model card's system prompt.
pub const SYSTEM_PROMPT: &str = "Generate an image matching the requested resolution by detailing the color, quantity, text, shape, size, texture, spatial relationships of the objects and background.";

/// The recommended sizes of the model card (width x height).
pub const RECOMMENDED_SIZES: [(usize, usize); 4] = [(1024, 1024), (1184, 880), (1248, 832), (880, 1184)];

/// Pixels per visual token along each side.
pub const PATCH: usize = 16;

/// The model's largest trained image area (1024 x 1024 pixels).
pub const MAX_AREA: usize = 1024 * 1024;

/// Ids of the visual vocabulary (`unitron_extension_manifest.json`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Vocab {
    pub img_start: u32,
    pub img_end: u32,
    pub vis_row: u32,
    /// `<|vis_0|>`.
    pub vis_start: u32,
    /// One past the last visual token.
    pub vis_end: u32,
    /// `<|im_end|>`: the end of a reply.
    pub im_end: u32,
}

impl Vocab {
    /// The release's layout: the three framing tokens after the 131,072 text tokens, then the
    /// codebook.
    pub const fn release(im_end: u32) -> Self {
        Self {
            img_start: 131_072,
            img_end: 131_073,
            vis_row: 131_074,
            vis_start: 131_075,
            vis_end: 131_075 + 196_560,
            im_end,
        }
    }

    /// From `unitron_extension_manifest.json` when the checkpoint has one.
    fn from_manifest(dir: &Path, im_end: u32) -> Result<Self> {
        let p = dir.join("unitron_extension_manifest.json");
        if !crate::fsx::exists(&p) {
            return Ok(Self::release(im_end));
        }
        let m: serde_json::Value = serde_json::from_slice(&crate::fsx::read(&p)?)?;
        let r = &m["unitron_token_ranges"];
        let id = |v: &serde_json::Value, what: &str| -> Result<u32> {
            v.as_u64()
                .map(|x| x as u32)
                .with_context(|| format!("unitron_extension_manifest.json: no {what}"))
        };
        Ok(Self {
            img_start: id(&r["framing"]["<|img_start|>"], "<|img_start|>")?,
            img_end: id(&r["framing"]["<|img_end|>"], "<|img_end|>")?,
            vis_row: id(&r["framing"]["<|vis_row|>"], "<|vis_row|>")?,
            vis_start: id(&r["vision"]["start_id"], "vision start_id")?,
            vis_end: id(&r["vision"]["end_id"], "vision end_id")?,
            im_end,
        })
    }

    pub fn is_visual(&self, id: u32) -> bool {
        (self.vis_start..self.vis_end).contains(&id)
    }

    pub fn codebook_size(&self) -> usize {
        (self.vis_end - self.vis_start) as usize
    }
}

/// Whether `path` is a Nemotron-Labs-Image checkpoint: a `nemotron_h` folder with the visual
/// vocabulary (its manifest, or the tokenizer folder beside a 327,680-row vocabulary), or a
/// GGUF file of a Nemotron-H model whose vocabulary has `<|img_start|>` at id 131,072.
pub fn is_nemotron_image(path: &Path) -> bool {
    if crate::gguf::is_gguf(path) {
        return crate::gguf::header(path).is_ok_and(|h| {
            let arch = h
                .get("general.architecture")
                .and_then(crate::gguf::Value::as_str)
                .unwrap_or_default();
            arch.starts_with("nemotron_h")
                && h.get("tokenizer.ggml.tokens")
                    .and_then(crate::gguf::Value::as_array)
                    .is_some_and(|t| t.len() > 131_074 && t[131_072].as_str() == Some("<|img_start|>"))
        });
    }
    let Ok(b) = crate::fsx::read(&path.join("config.json")) else {
        return false;
    };
    let Ok(c) = serde_json::from_slice::<serde_json::Value>(&b) else {
        return false;
    };
    c["model_type"] == "nemotron_h"
        && (crate::fsx::exists(&path.join("unitron_extension_manifest.json"))
            || (c["vocab_size"].as_u64().is_some_and(|v| v >= 327_635)
                && crate::fsx::exists(&path.join(TOKENIZER_DIR).join("config.json"))))
}

/// Loading options.
#[derive(Clone, Debug)]
pub struct LoadOptions {
    /// Precision of the language model's matrices for a float checkpoint (GGUF files keep their
    /// own); default bf16.
    pub quant: Quant,
    /// Map the weights from disk instead of reading them into memory.
    pub mmap: bool,
    /// Compute device (process-wide; set by the first model loaded).
    pub device: DevicePref,
    /// Where repo ids are downloaded (default [`crate::hf::cache_root`]).
    pub cache_dir: Option<PathBuf>,
    /// The image tokenizer: a folder (`Nemotron-Labs-Image-Tokenizer/`) or a repo id holding
    /// that folder. None: the checkpoint's own folder, else [`REPO`]'s, fetched on first use.
    pub tokenizer: Option<String>,
    /// Keep the tokenizer's float32 weights (default: rounded to bf16, as the reference runs it).
    pub vit_f32: bool,
    /// Load the tokenizer's encoder, for source images (default true).
    pub encoder: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            quant: Quant::Bf16,
            mmap: false,
            device: DevicePref::Auto,
            cache_dir: None,
            tokenizer: None,
            vit_f32: false,
            encoder: true,
        }
    }
}

fn flag(key: &str, value: &str) -> Result<bool> {
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(anyhow!("{key}: expected a boolean (1/0, true/false), got {value:?}")),
    }
}

impl LoadOptions {
    /// Sets an option by name (the C API's and the server's `key=value` options): `quant`
    /// (bf16, q8, q4), `mmap`, `device` (auto, cpu, metal, cuda, vulkan), `cache_dir`,
    /// `tokenizer`, `vit_precision` (bf16, f32), `encoder`.
    pub fn set(&mut self, key: &str, value: &str) -> Result<()> {
        match key {
            "quant" => self.quant = Quant::parse(value).with_context(|| format!("quant: unknown quantization {value:?} (bf16, q8, q4)"))?,
            "mmap" => self.mmap = flag(key, value)?,
            "device" => {
                self.device =
                    DevicePref::parse(value).with_context(|| format!("device: unknown device {value:?} (auto|cpu|cuda|metal|vulkan)"))?
            }
            "cache_dir" => self.cache_dir = (!value.is_empty()).then(|| PathBuf::from(value)),
            "tokenizer" => self.tokenizer = (!value.is_empty()).then(|| value.to_string()),
            "vit_precision" => {
                self.vit_f32 = match value {
                    "bf16" => false,
                    "f32" => true,
                    _ => bail!("vit_precision: expected bf16 or f32, got {value:?}"),
                }
            }
            "encoder" => self.encoder = flag(key, value)?,
            _ => bail!("unknown option {key:?} (quant, mmap, device, cache_dir, tokenizer, vit_precision, encoder)"),
        }
        Ok(())
    }
}

/// How the image block is generated.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Structure {
    /// The header and row markers written for the size asked for; visual tokens sampled among
    /// the visual vocabulary only.
    #[default]
    Grid,
    /// Every token sampled over the whole vocabulary (the reference); the image has the size
    /// the model's header announces.
    Free,
}

impl Structure {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "grid" => Ok(Self::Grid),
            "free" => Ok(Self::Free),
            _ => bail!("structure: expected grid or free, got {s:?}"),
        }
    }
}

/// Text-to-image options; the defaults are the model card's. Deserializes from JSON with every
/// key optional.
#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GenerateOptions {
    /// Image size in pixels: positive multiples of 16 (default 1024 x 1024; see
    /// [`RECOMMENDED_SIZES`]).
    pub width: usize,
    pub height: usize,
    /// Sampling temperature (0.9); 0 picks the most likely token.
    pub temperature: f32,
    /// Nucleus sampling threshold (1: off).
    pub top_p: f32,
    /// Keep only the `top_k` most likely tokens (0: all).
    pub top_k: usize,
    pub seed: u64,
    /// Images generated from the prompt, with seeds `seed`, `seed + 1`, ... (default 1).
    pub num_images: usize,
    /// The system prompt; None for [`SYSTEM_PROMPT`], an empty string for none.
    pub system_prompt: Option<String>,
    pub structure: Structure,
    /// [`Structure::Free`]: the most tokens generated (0: the grid of the size asked for plus
    /// some slack, at least the card's 4,352).
    pub max_tokens: usize,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            width: 1024,
            height: 1024,
            temperature: 0.9,
            top_p: 1.0,
            top_k: 0,
            seed: 42,
            num_images: 1,
            system_prompt: None,
            structure: Structure::Grid,
            max_tokens: 0,
        }
    }
}

impl GenerateOptions {
    /// Sets an option by name (`width`, `height`, `size` as `WxH`, `temperature`, `top_p`,
    /// `top_k`, `seed`, `num_images`, `system_prompt`, `structure` (grid, free), `max_tokens`).
    pub fn set(&mut self, key: &str, value: &str) -> Result<()> {
        let num = |v: &str| -> Result<usize> { v.trim().parse().with_context(|| format!("{key}: not a number: {v:?}")) };
        match key {
            "width" => self.width = num(value)?,
            "height" => self.height = num(value)?,
            "size" => {
                let (w, h) = value
                    .split_once(['x', 'X'])
                    .with_context(|| format!("size: expected WxH, got {value:?}"))?;
                (self.width, self.height) = (num(w)?, num(h)?);
            }
            "temperature" => self.temperature = value.parse().with_context(|| format!("temperature: {value:?}"))?,
            "top_p" => self.top_p = value.parse().with_context(|| format!("top_p: {value:?}"))?,
            "top_k" => self.top_k = num(value)?,
            "seed" => self.seed = value.parse().with_context(|| format!("seed: {value:?}"))?,
            "num_images" | "n" => self.num_images = num(value)?,
            "system_prompt" | "system" => self.system_prompt = Some(value.to_string()),
            "structure" => self.structure = Structure::parse(value)?,
            "max_tokens" => self.max_tokens = num(value)?,
            _ => bail!(
                "unknown option {key:?} (width, height, size, temperature, top_p, top_k, seed, num_images, system_prompt, structure, max_tokens)"
            ),
        }
        Ok(())
    }

    fn check(&self) -> Result<()> {
        let (w, h) = (self.width, self.height);
        if w == 0 || h == 0 || !w.is_multiple_of(PATCH) || !h.is_multiple_of(PATCH) {
            bail!("image size {w}x{h}: width and height must be positive multiples of {PATCH}");
        }
        if w * h > 4 * MAX_AREA {
            bail!("image size {w}x{h}: more than four times the model's largest trained area (1024x1024)");
        }
        if !(self.temperature >= 0.0 && self.top_p > 0.0 && self.top_p <= 1.0) {
            bail!("temperature must be >= 0 and top_p in (0, 1]");
        }
        Ok(())
    }
}

/// Where a generation is, for the progress callback.
#[derive(Clone, Copy, Debug)]
pub struct Progress {
    /// The image being generated (from 0) and how many there are.
    pub image: usize,
    pub images: usize,
    /// Visual tokens sampled so far for this image, and how many its grid has.
    pub step: usize,
    pub steps: usize,
}

/// Timings of a generation, in seconds (summed over the images).
#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub prompt_tokens: usize,
    /// The prompt through the language model.
    pub prefill: f64,
    /// Tokens generated after the prompt (visual tokens and markers).
    pub tokens: usize,
    pub token_secs: f64,
    /// The tokenizer's decoder.
    pub decode: f64,
}

/// One generated image with its tokens.
#[derive(Clone)]
pub struct Generated {
    pub image: Image,
    /// Codebook indices, row by row.
    pub codes: Vec<u32>,
    /// `(rows, columns)` of visual tokens.
    pub grid: (usize, usize),
    /// The ids the model generated after the prompt.
    pub tokens: Vec<u32>,
}

/// Stops a running generation from another thread.
#[derive(Clone, Debug)]
pub struct CancelHandle(Arc<AtomicBool>);

impl CancelHandle {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

/// A loaded Nemotron-Labs-Image model.
pub struct NemotronImage {
    // Declared before the files they borrow: dropped first.
    lm: NemotronModel<'static>,
    vit: Vit<'static>,
    tok: Tokenizer,
    pub vocab: Vocab,
    cancel: Arc<AtomicBool>,
    pub path: PathBuf,
    /// The tokenizer's folder.
    pub tokenizer_path: PathBuf,
    pub load_secs: f64,
    _vit_files: Box<ModelFiles>,
    _files: Box<ModelFiles>,
}

/// The tokenizer folder for checkpoint `dir` (a folder or a GGUF file): `opt` (a folder, or a
/// repo id holding [`TOKENIZER_DIR`]), the checkpoint's own folder, a copy beside a GGUF file,
/// or [`REPO`]'s, fetched.
fn tokenizer_dir(dir: &Path, opt: Option<&str>, cache_dir: Option<&Path>) -> Result<PathBuf> {
    let has = |d: &Path| crate::fsx::exists(&d.join("config.json")) && crate::fsx::exists(&d.join("model.safetensors"));
    if let Some(o) = opt {
        let p = Path::new(o);
        for d in [p.to_path_buf(), p.join(TOKENIZER_DIR)] {
            if has(&d) {
                return Ok(d);
            }
        }
        return fetch_tokenizer(o, cache_dir);
    }
    let base = if crate::gguf::is_gguf(dir) {
        dir.parent().unwrap_or(Path::new("."))
    } else {
        dir
    };
    let own = base.join(TOKENIZER_DIR);
    if has(&own) {
        return Ok(own);
    }
    fetch_tokenizer(REPO, cache_dir)
}

#[cfg(feature = "hf")]
fn fetch_tokenizer(repo: &str, cache_dir: Option<&Path>) -> Result<PathBuf> {
    let root = crate::hf::resolve_prefix(repo, TOKENIZER_DIR, cache_dir)?;
    Ok(root.join(TOKENIZER_DIR))
}

#[cfg(not(feature = "hf"))]
fn fetch_tokenizer(repo: &str, _cache_dir: Option<&Path>) -> Result<PathBuf> {
    bail!("the image tokenizer ({TOKENIZER_DIR}/ of {repo}) is not on disk, and this build cannot fetch it (no `hf` feature)")
}

/// The size header of an image block (`height=H width=W`).
pub fn size_header(width: usize, height: usize) -> String {
    format!("height={height} width={width}")
}

/// `(width, height)` from a size header.
pub fn parse_size_header(s: &str) -> Option<(usize, usize)> {
    let h = s.find("height=")?;
    let rest = &s[h + 7..];
    let hd: String = rest.chars().take_while(char::is_ascii_digit).collect();
    let w = rest.find("width=")?;
    let wd: String = rest[w + 6..].chars().take_while(char::is_ascii_digit).collect();
    Some((wd.parse().ok()?, hd.parse().ok()?))
}

/// Samples from `logits` (a slice of the vocabulary): plain softmax sampling at the sampler's
/// temperature unless it truncates (top-k / top-p) or is greedy.
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

impl NemotronImage {
    /// Loads a checkpoint: a folder, a GGUF file, or a repo id (`nvidia/Nemotron-Labs-Image-30B-A3B`,
    /// `org/name:QUANT` for a GGUF repository), fetched into the cache on first use; and the image
    /// tokenizer ([`LoadOptions::tokenizer`]).
    pub fn load(path_or_repo: &str, opts: &LoadOptions) -> Result<NemotronImage> {
        let t0 = Instant::now();
        let dir = crate::engine::resolve_model(path_or_repo, opts.cache_dir.as_deref())?;
        if !is_nemotron_image(&dir) {
            bail!(
                "{}: not a Nemotron-Labs-Image checkpoint (a nemotron_h model with the visual vocabulary)",
                dir.display()
            );
        }
        crate::gpu::ensure(opts.device)?;
        let tdir = tokenizer_dir(&dir, opts.tokenizer.as_deref(), opts.cache_dir.as_deref())?;
        let cfg = NemotronConfig::load(&dir)?;
        let files = Box::new(ModelFiles::open(&dir, !opts.mmap)?);
        // SAFETY: the box's heap block does not move, and `NemotronImage` drops `lm` (every
        // borrow of it) before `_files`.
        let fs: &'static ModelFiles = unsafe { &*(files.as_ref() as *const ModelFiles) };
        let tok = crate::engine::load_tokenizer(&dir)?;
        let im_end = tok.token_to_id("<|im_end|>").context("tokenizer: no <|im_end|>")?;
        let vocab = if crate::gguf::is_gguf(&dir) {
            Vocab::release(im_end)
        } else {
            Vocab::from_manifest(&dir, im_end)?
        };
        if (vocab.vis_end as usize) > cfg.vocab_size {
            bail!(
                "the visual vocabulary (ids up to {}) is past the model's {} rows",
                vocab.vis_end,
                cfg.vocab_size
            );
        }
        let vcfg = VitConfig::from_json(&serde_json::from_slice(&crate::fsx::read(&tdir.join("config.json"))?)?)?;
        if vcfg.n_embed != vocab.codebook_size() {
            bail!(
                "the tokenizer's codebook ({}) is not the model's visual vocabulary ({})",
                vcfg.n_embed,
                vocab.codebook_size()
            );
        }
        let vfiles = Box::new(ModelFiles::open(&tdir, !opts.mmap)?);
        // SAFETY: as above, `vit` is dropped before `_vit_files`.
        let vfs: &'static ModelFiles = unsafe { &*(vfiles.as_ref() as *const ModelFiles) };
        let prec = if opts.vit_f32 { Precision::F32 } else { Precision::Bf16 };
        let vit = Vit::load(vcfg, vfs, prec, opts.encoder, true)?;
        let mut lm = NemotronModel::load(cfg, fs, opts.quant)?;
        // Inside an image only the visual tokens' logits are read: their rows of the head.
        lm.load_head_rows(fs, vocab.vis_start as usize..vocab.vis_end as usize)?;
        Ok(NemotronImage {
            lm,
            vit,
            tok,
            vocab,
            cancel: Arc::new(AtomicBool::new(false)),
            path: dir,
            tokenizer_path: tdir,
            load_secs: t0.elapsed().as_secs_f64(),
            _vit_files: vfiles,
            _files: files,
        })
    }

    pub fn config(&self) -> &NemotronConfig {
        &self.lm.cfg
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }

    /// The language model.
    pub fn language_model(&self) -> &NemotronModel<'static> {
        &self.lm
    }

    /// The image tokenizer.
    pub fn image_tokenizer(&self) -> &Vit<'static> {
        &self.vit
    }

    pub fn cancel_handle(&self) -> CancelHandle {
        CancelHandle(self.cancel.clone())
    }

    fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.tok.encode(text, false).map_err(|e| anyhow!("encode: {e}"))?.get_ids().to_vec())
    }

    /// The token ids of an image block for a grid of codes (`rows x columns`, the image
    /// `16 columns` wide).
    pub fn image_block(&self, codes: &[u32], grid: (usize, usize)) -> Result<Vec<u32>> {
        let (gh, gw) = grid;
        if codes.len() != gh * gw {
            bail!("{} codes for a {gh}x{gw} grid", codes.len());
        }
        let v = &self.vocab;
        let mut ids = vec![v.img_start];
        ids.extend(self.encode_text(&size_header(gw * PATCH, gh * PATCH))?);
        for row in codes.chunks(gw) {
            ids.push(v.vis_row);
            ids.extend(row.iter().map(|&c| v.vis_start + c));
        }
        ids.push(v.img_end);
        Ok(ids)
    }

    /// An image as the model's tokens: resized (Lanczos) to fit [`MAX_AREA`] when larger, its
    /// sides rounded down to multiples of 16, encoded by the tokenizer. Returns the codes and
    /// the grid.
    pub fn encode_image(&self, img: &RgbImage) -> Result<(Vec<u32>, (usize, usize))> {
        if !self.vit.has_encoder() {
            bail!("source images need the tokenizer's encoder (loaded with encoder=false)");
        }
        let (w, h) = (img.width, img.height);
        if w < PATCH || h < PATCH {
            bail!("source image {w}x{h} is smaller than one {PATCH}x{PATCH} patch");
        }
        let s = ((MAX_AREA as f64) / (w * h) as f64).sqrt().min(1.0);
        let tw = (((w as f64 * s) as usize) / PATCH * PATCH).max(PATCH);
        let th = (((h as f64 * s) as usize) / PATCH * PATCH).max(PATCH);
        let rgb = if (tw, th) == (w, h) {
            img.data.clone()
        } else {
            crate::resample::resize_lanczos_rgb8(&img.data, w, h, tw, th)
        };
        self.vit.encode(&vit::from_rgb8(&rgb, th, tw), th, tw)
    }

    /// Renders a grid of codes to pixels with the tokenizer's decoder.
    pub fn decode_codes(&self, codes: &[u32], grid: (usize, usize)) -> Result<Image> {
        let px = self.vit.decode(codes, grid)?;
        let (h, w) = (grid.0 * PATCH, grid.1 * PATCH);
        Ok(Image {
            width: w,
            height: h,
            rgb: vit::to_rgb8(&px, h, w),
        })
    }

    /// The prompt's token ids: the chat template with thinking off, the source images' blocks
    /// then `Generate a {W}x{H} image.\n\n{prompt}` in the user turn.
    pub fn prompt_ids(&self, prompt: &str, sources: &[(Vec<u32>, (usize, usize))], o: &GenerateOptions) -> Result<Vec<u32>> {
        let system = o.system_prompt.as_deref().unwrap_or(SYSTEM_PROMPT);
        let mut ids = self.encode_text(&format!("<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n"))?;
        for (codes, grid) in sources {
            ids.extend(self.image_block(codes, *grid)?);
            ids.extend(self.encode_text("\n")?);
        }
        ids.extend(self.encode_text(&format!(
            "Generate a {}x{} image.\n\n{}<|im_end|>\n<|im_start|>assistant\n<think></think>",
            o.width, o.height, prompt
        ))?);
        Ok(ids)
    }

    /// Tokens through the language model: the logits of the visual tokens after the last
    /// (`Vocab::vis_start..vis_end`).
    fn step(&self, sess: &mut crate::nemotron::NemotronSession, tokens: &[u32]) -> Vec<f32> {
        // One token at a time on the GPU decode plan when there is one (only the last one's
        // logits computed).
        if tokens.len() <= 2
            && let Some((first, last)) = tokens.split_last().map(|(l, f)| (f, *l))
            && first.iter().all(|&t| self.lm.gpu_step(sess, t, false).is_some())
        {
            return self.lm.forward_rows(sess, &[last]);
        }
        self.lm.forward_rows(sess, tokens)
    }

    /// Generates images from `prompt` (and `sources`, images the instruction refers to),
    /// calling `progress` after every visual token; [`Flow::Stop`] (or [`CancelHandle::cancel`])
    /// ends the call with an error.
    pub fn generate(
        &self,
        prompt: &str,
        sources: &[RgbImage],
        o: &GenerateOptions,
        progress: &mut dyn FnMut(Progress) -> Flow,
    ) -> Result<(Vec<Generated>, Stats)> {
        o.check()?;
        self.cancel.store(false, Ordering::Relaxed);
        let src: Vec<_> = sources.iter().map(|s| self.encode_image(s)).collect::<Result<_>>()?;
        let ids = self.prompt_ids(prompt, &src, o)?;
        let mut st = Stats {
            prompt_tokens: ids.len(),
            ..Default::default()
        };
        let n = o.num_images.max(1);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let seed = o.seed.wrapping_add(i as u64);
            let g = self.generate_one(&ids, o, seed, &mut st, &mut |step, steps| {
                if self.cancel.load(Ordering::Relaxed) {
                    return Flow::Stop;
                }
                progress(Progress {
                    image: i,
                    images: n,
                    step,
                    steps,
                })
            })?;
            out.push(g);
        }
        Ok((out, st))
    }

    /// One image from the prompt `ids`.
    fn generate_one(
        &self,
        ids: &[u32],
        o: &GenerateOptions,
        seed: u64,
        st: &mut Stats,
        progress: &mut dyn FnMut(usize, usize) -> Flow,
    ) -> Result<Generated> {
        let v = self.vocab;
        let mut sampler = Sampler::new(o.temperature, o.top_k, o.top_p, seed);
        let mut sess = self.lm.new_session();
        let (gw, gh) = (o.width / PATCH, o.height / PATCH);
        let stopped = || anyhow!("generation stopped");
        match o.structure {
            Structure::Grid => {
                // The prompt, the opening marker, the header and the first row marker in one pass.
                let mut pre = ids.to_vec();
                // The model opens its reply with a line break before the block.
                let mut out_ids = self.encode_text("\n")?;
                out_ids.push(v.img_start);
                out_ids.extend(self.encode_text(&size_header(o.width, o.height))?);
                out_ids.push(v.vis_row);
                pre.extend(&out_ids);
                let t = Instant::now();
                let mut logits = self.step(&mut sess, &pre);
                st.prefill += t.elapsed().as_secs_f64();
                let t = Instant::now();
                let steps = gh * gw;
                let mut codes = Vec::with_capacity(steps);
                for r in 0..gh {
                    for c in 0..gw {
                        let code = sample(&mut sampler, &logits);
                        codes.push(code);
                        let tk = v.vis_start + code;
                        out_ids.push(tk);
                        if progress(codes.len(), steps) == Flow::Stop {
                            return Err(stopped());
                        }
                        if c + 1 < gw {
                            logits = self.step(&mut sess, &[tk]);
                        } else if r + 1 < gh {
                            logits = self.step(&mut sess, &[tk, v.vis_row]);
                            out_ids.push(v.vis_row);
                        } else {
                            out_ids.push(v.img_end);
                        }
                    }
                }
                st.token_secs += t.elapsed().as_secs_f64();
                st.tokens += out_ids.len();
                let t = Instant::now();
                let image = self.decode_codes(&codes, (gh, gw))?;
                st.decode += t.elapsed().as_secs_f64();
                Ok(Generated {
                    image,
                    codes,
                    grid: (gh, gw),
                    tokens: out_ids,
                })
            }
            Structure::Free => {
                let max = if o.max_tokens > 0 {
                    o.max_tokens
                } else {
                    (gh * (gw + 1) + 192).max(4352)
                };
                let t = Instant::now();
                let mut logits = self.lm.forward(&mut sess, ids);
                st.prefill += t.elapsed().as_secs_f64();
                let t = Instant::now();
                let mut out_ids = Vec::new();
                let steps = gh * gw;
                let mut seen = 0;
                while out_ids.len() < max {
                    let tk = sample(&mut sampler, &logits);
                    out_ids.push(tk);
                    if v.is_visual(tk) {
                        seen += 1;
                        if progress(seen.min(steps), steps) == Flow::Stop {
                            return Err(stopped());
                        }
                    }
                    if tk == v.img_end || tk == v.im_end {
                        break;
                    }
                    logits = self.lm.forward(&mut sess, &[tk]);
                }
                st.token_secs += t.elapsed().as_secs_f64();
                st.tokens += out_ids.len();
                let (codes, grid) = self.parse_block(&out_ids)?;
                let t = Instant::now();
                let image = self.decode_codes(&codes, grid)?;
                st.decode += t.elapsed().as_secs_f64();
                Ok(Generated {
                    image,
                    codes,
                    grid,
                    tokens: out_ids,
                })
            }
        }
    }

    /// The grid of an image block among generated ids (see [`parse_block`]).
    pub fn parse_block(&self, ids: &[u32]) -> Result<(Vec<u32>, (usize, usize))> {
        parse_block(&self.vocab, ids, &|t| self.tok.decode(t, false).unwrap_or_default())
    }
}

/// The grid of an image block among generated ids: its header's size (rounded up to whole
/// patches) and its rows, each cut or padded (with its last code, or code 0 for a missing
/// row) to the header's width; missing rows are repeats of the last.
pub fn parse_block(v: &Vocab, ids: &[u32], text: &dyn Fn(&[u32]) -> String) -> Result<(Vec<u32>, (usize, usize))> {
    let start = ids
        .iter()
        .position(|&t| t == v.img_start)
        .context("the generation has no <|img_start|>")?;
    let body = &ids[start + 1..];
    let end = body.iter().position(|&t| t == v.img_end).unwrap_or(body.len());
    let body = &body[..end];
    let first_row = body.iter().position(|&t| t == v.vis_row).unwrap_or(body.len());
    let header = text(&body[..first_row]);
    let mut rows: Vec<Vec<u32>> = Vec::new();
    for &t in &body[first_row..] {
        if t == v.vis_row {
            rows.push(Vec::new());
        } else if v.is_visual(t)
            && let Some(r) = rows.last_mut()
        {
            r.push(t - v.vis_start);
        }
    }
    let (gw, gh) = match parse_size_header(&header) {
        Some((w, h)) if w > 0 && h > 0 => (w.div_ceil(PATCH), h.div_ceil(PATCH)),
        _ => {
            let w = rows.iter().map(Vec::len).max().unwrap_or(0);
            (w, rows.len())
        }
    };
    if gw == 0 || gh == 0 || rows.is_empty() {
        bail!("the generation has no image grid (header {header:?})");
    }
    let mut codes = Vec::with_capacity(gw * gh);
    for r in 0..gh {
        let row = rows.get(r).or(rows.last()).unwrap();
        for c in 0..gw {
            codes.push(row.get(c).or(row.last()).copied().unwrap_or(0));
        }
    }
    Ok((codes, (gh, gw)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_header_round_trip() {
        assert_eq!(size_header(1184, 880), "height=880 width=1184");
        assert_eq!(parse_size_header("height=880 width=1184"), Some((1184, 880)));
        assert_eq!(parse_size_header(" height=1024\nwidth=512 "), Some((512, 1024)));
        assert_eq!(parse_size_header("width=512"), None);
    }

    #[test]
    fn block_parsing() {
        let v = Vocab::release(11);
        // Text tokens 1000.. stand for the header's text.
        let text = |t: &[u32]| {
            if t.is_empty() {
                String::new()
            } else {
                "height=48 width=32".to_string()
            }
        };
        let vis = |c: u32| v.vis_start + c;
        let mut ids = vec![5, v.img_start, 1000, 1001, v.vis_row, vis(1), vis(2), v.vis_row, vis(3)];
        ids.extend([vis(4), vis(9), v.img_end, 77]);
        // 2 columns, 3 rows: the long row is cut, the missing one repeats the last.
        let (codes, grid) = parse_block(&v, &ids, &text).unwrap();
        assert_eq!(grid, (3, 2));
        assert_eq!(codes, vec![1, 2, 3, 4, 3, 4]);
        // No header: the grid as written.
        let bare = [v.img_start, v.vis_row, vis(7), vis(8), v.vis_row, vis(5), vis(6)];
        assert_eq!(parse_block(&v, &bare, &|_| String::new()).unwrap(), (vec![7, 8, 5, 6], (2, 2)));
        assert!(parse_block(&v, &[1, 2, 3], &text).is_err());
        assert_eq!(v.codebook_size(), 196_560);
        assert!(v.is_visual(131_075) && !v.is_visual(131_074) && !v.is_visual(327_635));
    }

    #[test]
    fn options_by_name() {
        let mut o = GenerateOptions::default();
        o.set("size", "1184x880").unwrap();
        o.set("structure", "free").unwrap();
        o.set("seed", "7").unwrap();
        assert_eq!((o.width, o.height, o.structure, o.seed), (1184, 880, Structure::Free, 7));
        assert!(o.set("size", "12").is_err());
        let j: GenerateOptions = serde_json::from_str(r#"{"width": 880, "height": 1184, "structure": "grid"}"#).unwrap();
        assert_eq!((j.width, j.height, j.temperature), (880, 1184, 0.9));
        let bad = GenerateOptions {
            width: 1000,
            ..Default::default()
        };
        assert!(bad.check().is_err());
        let mut l = LoadOptions::default();
        l.set("vit_precision", "f32").unwrap();
        l.set("quant", "q4").unwrap();
        assert!(l.vit_f32 && l.quant == Quant::Q4);
    }
}
