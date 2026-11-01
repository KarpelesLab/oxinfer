//! Step1X-Edit: instruction-based image editing from a diffusers folder
//! (`stepfun-ai/Step1X-Edit-v1p2`, `-v1p2-preview`, `-v1p1-diffusers`), following the
//! `Step1XEditPipelineV1P2` / `Step1XEditPipeline` of StepFun's diffusers fork:
//!
//! - Qwen2.5-VL-7B (`crate::qwen_vl`; v1.2 ships a fine-tuned copy) reads the instruction and the
//!   input image (resized to ~1024x1024, then by the processor to 256..400 vision tokens), with a
//!   fixed editing prefix whose first tokens are dropped. Characters inside quotes are tokenized
//!   one by one (v1.2 also feeds their raw token embeddings to the DiT, for text rendering);
//! - the DiT ([`transformer`]): a connector refines the hidden states into the FLUX text stream,
//!   then FLUX.1's 19 double + 38 single blocks run over `[text; target; reference]`. v1.2 places
//!   the image's vision tokens in the text stream at the RoPE coordinates of the latent patches
//!   they cover, and the reference latents at index 1 on the same grid as the target;
//! - flow matching with Euler steps and FLUX's resolution-dependent shift; true classifier-free
//!   guidance against an empty instruction (same image), with the guidance difference divided by
//!   `|cond - uncond|^0.4` per token where that norm exceeds 1;
//! - the FLUX VAE ([`crate::flux::vae`]); the output is resized back to the input's size.
//!
//! The output keeps the input's aspect ratio at about `size_level`^2 pixels (default 1024). The
//! optional *thinking* mode (v1.2) first has the VL model rewrite the instruction from the image.
//! Components load on first use; the 16 GB text encoder is dropped after encoding unless asked
//! to stay; the transformer can be quantized at load (q8 / q4), mapped, or streamed.

pub mod transformer;

use crate::clock::Instant;
use crate::flux::rope::Rope;
use crate::flux::sched::{FlowMatchEuler, flux_sigmas};
use crate::flux::vae::{self, Vae};
use crate::model::Quant;
use crate::qwen_vl::{QwenVl, QwenVlOptions, VlImage};
pub use crate::sd::Image;
use crate::video::RgbImage;
use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use transformer::{Cond, Config, Transformer};

/// v1.2's instruction prefix (`EDIT_PREFIX`).
const PREFIX_V12: &str = "Given a reference image and a user editing instruction.\nYou need to understand the information of the reference image and the content of the editing instruction,\nand then generate a detailed description of the edited target image based on the reference image and the editing instruction. \nThe description should include the requirements in the editing instruction and try to closely match the content of the target image.\nThe editing instruction is:";
/// Tokens of the v1.2 template dropped from the hidden states.
const DROP_V12: usize = 90;
/// Most text tokens kept (v1.2).
const MAX_TOKENS_V12: usize = 2048;

/// v1.0 / v1.1's prefix (`QWEN25VL_PREFIX` of `Step1XEditPipeline`).
const PREFIX_V11: &str = "Given a user prompt, generate an \"Enhanced prompt\" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:\n- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.\n- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.\n\nHere are examples of how to transform or refine prompts:\n- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.\n- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.\n\nPlease generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:\nUser Prompt:";
const DROP_V11: usize = 217;
/// v1.1 pads (or cuts) the text stream to this many tokens.
const MAX_TOKENS_V11: usize = 640;

/// The default negative prompt of text-to-image runs (no input image).
const T2I_NEGATIVE: &str = "worst quality, wrong limbs, unreasonable limbs, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting";

/// Which prompt encoding a checkpoint was trained with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Version {
    /// v1.0 / v1.1 (`Step1XEditPipeline`): 640 padded text tokens, all at RoPE position 0.
    V1p1,
    /// v1.2 and v1.2-preview (`Step1XEditPipelineV1P2`; the transformer has
    /// `text_token_mapping`).
    V1p2,
}

/// Whether `dir` is a Step1X-Edit diffusers folder.
pub fn is_step1x(dir: &Path) -> bool {
    let read = |p: PathBuf| -> Option<serde_json::Value> { serde_json::from_slice(&crate::fsx::read(&p).ok()?).ok() };
    read(dir.join("model_index.json")).is_some_and(|v| {
        v["_class_name"].as_str().is_some_and(|c| c.starts_with("Step1XEdit"))
            || v["transformer"][1].as_str() == Some("Step1XEditTransformer2DModel")
    })
}

/// How to load a [`Pipeline`].
#[derive(Clone, Debug)]
pub struct Options {
    /// Precision of the transformer's matmul weights.
    pub quant: Quant,
    /// Precision of the Qwen2.5-VL text encoder's language model.
    pub text_quant: Quant,
    /// Map weights from disk instead of reading them into memory.
    pub mmap: bool,
    /// The transformer from elsewhere: a diffusers `transformer/` folder or a safetensors / GGUF
    /// file with the same tensor names.
    pub transformer: Option<PathBuf>,
    /// Stream the transformer's blocks from storage (weights mapped, read ahead per block).
    pub stream: Option<crate::stream::StreamOptions>,
    /// Keep the text encoder loaded between calls (otherwise dropped after each encoding).
    pub keep_text_encoder: bool,
    /// Attention in f32 on the GPU (default: the half-precision matrix-unit kernels, 28% faster
    /// per pass; the first velocity is as close to the f32 reference either way, 3e-4).
    pub exact_attention: bool,
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
            stream: None,
            keep_text_encoder: false,
            exact_attention: false,
            device: crate::platform::DevicePref::Auto,
        }
    }
}

impl Options {
    /// Sets an option by name (the C API's `key=value` options): `quant`, `text_quant` (bf16,
    /// q8, q4), `mmap`, `transformer`, `keep_text_encoder`, `stream`, `exact_attention`, `device`.
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
            "keep_text_encoder" => self.keep_text_encoder = flag()?,
            "stream" => self.stream = flag()?.then(crate::stream::StreamOptions::default),
            "exact_attention" => self.exact_attention = flag()?,
            "device" => {
                self.device = crate::platform::DevicePref::parse(value)
                    .with_context(|| format!("device: unknown device {value:?} (auto|cpu|cuda|metal|vulkan)"))?
            }
            _ => bail!("unknown option {key:?} (quant, text_quant, mmap, transformer, keep_text_encoder, stream, exact_attention, device)"),
        }
        Ok(())
    }
}

/// One edit's settings (the pipeline's `__call__` arguments).
#[derive(Clone, Debug)]
pub struct Params {
    pub prompt: String,
    /// Negative prompt (default: empty for edits, a list of defects for text to image).
    pub negative: Option<String>,
    /// Denoising steps (diffusers' default 28; StepFun's examples use 50).
    pub steps: usize,
    /// True classifier-free guidance scale (none at 1 or below).
    pub true_cfg: f32,
    pub seed: u64,
    /// The working resolution: about `size_level`^2 pixels at the input's aspect ratio
    /// (multiples of 16). Default 1024.
    pub size_level: usize,
    /// The returned image's size (default: the input's; text to image: `size_level` square).
    pub width: Option<usize>,
    pub height: Option<usize>,
    /// v1.2 thinking mode: the VL model first rewrites the instruction from the image (sampled,
    /// temperature 0.7, top-p 0.9, up to 128 tokens).
    pub think: bool,
    /// Above this timestep (0..1000) the guidance difference is divided by its norm^power.
    pub timesteps_truncate: f32,
    pub norm_power: f32,
}

impl Params {
    pub fn new(prompt: &str) -> Self {
        Self {
            prompt: prompt.to_string(),
            negative: None,
            steps: 28,
            true_cfg: 6.0,
            seed: 42,
            size_level: 1024,
            width: None,
            height: None,
            think: false,
            timesteps_truncate: 0.93,
            norm_power: 0.4,
        }
    }
}

/// A request as JSON (the C API's `request_json`; every key but `prompt` optional).
#[derive(Clone, Debug, Default, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Request {
    pub prompt: String,
    pub negative: Option<String>,
    pub steps: Option<usize>,
    pub true_cfg: Option<f32>,
    pub seed: Option<u64>,
    pub size_level: Option<usize>,
    pub width: Option<usize>,
    pub height: Option<usize>,
    pub think: Option<bool>,
    /// The input image file (decoded with the `media` feature); none: text to image.
    pub images: Vec<String>,
}

impl Request {
    pub fn params(&self) -> Params {
        let mut p = Params::new(&self.prompt);
        p.negative = self.negative.clone();
        if let Some(s) = self.steps {
            p.steps = s;
        }
        if let Some(g) = self.true_cfg {
            p.true_cfg = g;
        }
        if let Some(s) = self.seed {
            p.seed = s;
        }
        if let Some(s) = self.size_level {
            p.size_level = s;
        }
        p.width = self.width;
        p.height = self.height;
        p.think = self.think.unwrap_or(false);
        p
    }
}

/// Timings of one call, in seconds.
#[derive(Default, Debug, Clone)]
pub struct Timings {
    pub load_text: f64,
    pub think: f64,
    pub text: f64,
    pub load_transformer: f64,
    pub vae_encode: f64,
    pub denoise: f64,
    pub decode: f64,
    pub transformer_calls: usize,
    /// Tokens per transformer call (text + target + reference).
    pub tokens: usize,
    /// The working resolution.
    pub size: (usize, usize),
    /// The instruction the thinking mode produced.
    pub rewritten: Option<String>,
}

struct LoadedTransformer {
    // Drop order: the transformer (holding a stream hook), then the streamer.
    t: Transformer,
    _stream: Option<Arc<crate::stream::Streamer>>,
}

/// A Step1X-Edit pipeline (see the module docs).
pub struct Pipeline {
    pub dir: PathBuf,
    pub version: Version,
    pub opts: Options,
    pub sched: FlowMatchEuler,
    pub tcfg: Config,
    text: Option<QwenVl>,
    tr: Option<LoadedTransformer>,
    vae: Vae,
    cancel: Arc<AtomicBool>,
}

fn dump_dir() -> Option<PathBuf> {
    std::env::var_os("OXINFER_STEP1X_DUMP").map(PathBuf::from)
}

fn dump(name: &str, v: &[f32]) {
    if let Some(d) = dump_dir() {
        let _ = std::fs::write(d.join(name), v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
    }
}

/// [`dump`] unless the file exists (the first call's values).
pub(crate) fn dump_once(name: &str, v: &[f32]) {
    if let Some(d) = dump_dir()
        && !d.join(name).exists()
    {
        dump(name, v);
    }
}

/// `name` from the dump folder, if there (reference checks: the reference's inputs instead of
/// ours).
fn read_override(name: &str, len: Option<usize>) -> Result<Option<Vec<f32>>> {
    let Some(f) = dump_dir().map(|d| d.join(name)).filter(|f| f.exists()) else {
        return Ok(None);
    };
    let b = std::fs::read(&f).with_context(|| format!("read {}", f.display()))?;
    let v: Vec<f32> = b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
    if let Some(n) = len
        && v.len() != n
    {
        bail!("{} has {} values, expected {n}", f.display(), v.len());
    }
    eprintln!("[using {}]", f.display());
    Ok(Some(v))
}

/// The working size of `encode_image`: about `level`^2 pixels at `w:h`, multiples of 16.
pub fn working_size(w: usize, h: usize, level: usize) -> (usize, usize) {
    let ar = w as f64 / h as f64;
    let l2 = (level * level) as f64;
    let (wn, hn) = if w > h {
        let wn = (l2 * ar).sqrt().ceil();
        (wn, (wn / ar).ceil())
    } else {
        let hn = (l2 / ar).sqrt().ceil();
        ((hn * ar).ceil(), hn)
    };
    (wn as usize / 16 * 16, hn as usize / 16 * 16)
}

/// `torch.nn.functional.interpolate(mode="bilinear", antialias=True)` of an interleaved RGB
/// image in f32 (separable triangle filter widened by the scale when shrinking).
pub fn resize_bilinear_aa(x: &[f32], w: usize, h: usize, tw: usize, th: usize) -> Vec<f32> {
    fn coeffs(n_in: usize, n_out: usize) -> Vec<(usize, Vec<f32>)> {
        let scale = n_in as f64 / n_out as f64;
        let support = if scale >= 1.0 { scale } else { 1.0 };
        let inv = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
        (0..n_out)
            .map(|i| {
                let center = scale * (i as f64 + 0.5);
                let xmin = ((center - support + 0.5) as i64).max(0) as usize;
                let xmax = ((center + support + 0.5) as i64).min(n_in as i64) as usize;
                let mut ws: Vec<f64> = (xmin..xmax)
                    .map(|j| {
                        let t = ((j as f64 - center + 0.5) * inv).abs();
                        if t < 1.0 { 1.0 - t } else { 0.0 }
                    })
                    .collect();
                let s: f64 = ws.iter().sum();
                if s != 0.0 {
                    ws.iter_mut().for_each(|v| *v /= s);
                }
                (xmin, ws.into_iter().map(|v| v as f32).collect())
            })
            .collect()
    }
    let (cx, cy) = (coeffs(w, tw), coeffs(h, th));
    // Horizontal pass, then vertical.
    let mut tmp = vec![0f32; h * tw * 3];
    crate::kernels::for_chunks(&mut tmp, tw * 3, h > 16, |y, row| {
        let src = &x[y * w * 3..(y + 1) * w * 3];
        for (ox, (x0, ws)) in cx.iter().enumerate() {
            for c in 0..3 {
                row[ox * 3 + c] = ws.iter().enumerate().map(|(k, &wt)| wt * src[(x0 + k) * 3 + c]).sum();
            }
        }
    });
    let mut out = vec![0f32; th * tw * 3];
    crate::kernels::for_chunks(&mut out, tw * 3, th > 16, |oy, row| {
        let (y0, ws) = &cy[oy];
        for (k, &wt) in ws.iter().enumerate() {
            let src = &tmp[(y0 + k) * tw * 3..(y0 + k + 1) * tw * 3];
            for (o, &s) in row.iter_mut().zip(src) {
                *o += wt * s;
            }
        }
    });
    out
}

/// v1.1's resize of the vision tower's image before the processor: sides rounded to multiples
/// of 28 within 4..16384 tokens of 28x28.
pub fn v11_vision_size(w: usize, h: usize) -> (usize, usize) {
    let (minp, maxp) = (4.0 * 784.0, 16384.0 * 784.0);
    let (wf, hf) = (w as f64, h as f64);
    let mut hb = (28.0f64).max((hf / 28.0).round_ties_even() * 28.0);
    let mut wb = (28.0f64).max((wf / 28.0).round_ties_even() * 28.0);
    if hb * wb > maxp {
        let beta = (hf * wf / maxp).sqrt();
        hb = (hf / beta / 28.0).floor() * 28.0;
        wb = (wf / beta / 28.0).floor() * 28.0;
    } else if hb * wb < minp {
        let beta = (minp / (hf * wf)).sqrt();
        hb = (hf * beta / 28.0).ceil() * 28.0;
        wb = (wf * beta / 28.0).ceil() * 28.0;
    }
    (wb as usize, hb as usize)
}

/// `split_string_in_quotation`: pieces of `s` (from character `prefix_len` on) with whether each
/// is one character inside quotes ("..." or the Chinese quotes); quoted text becomes one piece
/// per character. Indices are in characters, as Python's.
pub fn split_quotes(s: &str, prefix_len: usize) -> Vec<(String, bool)> {
    #[derive(PartialEq)]
    enum St {
        Out,
        En,
        Zh,
    }
    let mut res: Vec<(String, bool)> = Vec::new();
    let mut st = St::Out;
    let mut buf = String::new();
    let mut quote: Vec<char> = Vec::new();
    for (i, ch) in s.chars().enumerate() {
        if i < prefix_len {
            buf.push(ch);
            continue;
        }
        match st {
            St::Out => match ch {
                '"' => {
                    buf.push('"');
                    res.push((std::mem::take(&mut buf), false));
                    st = St::En;
                }
                '\u{201c}' => {
                    if !buf.is_empty() {
                        buf.push('\u{201c}');
                        res.push((std::mem::take(&mut buf), false));
                    }
                    buf.clear();
                    st = St::Zh;
                }
                c => buf.push(c),
            },
            St::En | St::Zh => {
                let close = if st == St::En { '"' } else { '\u{201d}' };
                if ch == close {
                    res.extend(quote.drain(..).map(|c| (c.to_string(), true)));
                    st = St::Out;
                    buf = close.to_string();
                } else {
                    quote.push(ch);
                }
            }
        }
    }
    if st != St::Out {
        let malformed: String = quote.iter().collect::<String>() + &buf;
        match res.last_mut() {
            Some(last) if !last.1 => last.0.push_str(&malformed),
            _ => res.push((malformed, false)),
        }
    } else if !buf.is_empty() {
        res.push((buf, false));
    }
    res
}

/// `split_string_in_quotation_and_special_tokens`: `s` split at the special tokens (longest
/// first), the other parts split at quotes ([`split_quotes`], the prefix measured from the
/// start of `s`).
pub fn split_prompt(s: &str, prefix_len: usize, specials: &[String]) -> Vec<(String, bool)> {
    let mut sp: Vec<&str> = specials.iter().map(String::as_str).filter(|t| !t.is_empty()).collect();
    sp.sort_by_key(|t| std::cmp::Reverse(t.len()));
    sp.dedup();
    // Parts: (text, is_special, start in characters).
    let mut parts: Vec<(&str, bool, usize)> = Vec::new();
    let (mut start, mut i, mut chars) = (0usize, 0usize, 0usize);
    let mut start_chars = 0usize;
    while i < s.len() {
        if let Some(t) = sp.iter().find(|t| s[i..].starts_with(**t)) {
            if start < i {
                parts.push((&s[start..i], false, start_chars));
            }
            parts.push((t, true, chars));
            i += t.len();
            chars += t.chars().count();
            start = i;
            start_chars = chars;
        } else {
            let c = s[i..].chars().next().unwrap();
            i += c.len_utf8();
            chars += 1;
        }
    }
    if start < s.len() {
        parts.push((&s[start..], false, start_chars));
    }
    let mut out = Vec::new();
    for (p, special, at) in parts {
        if special {
            out.push((p.to_string(), false));
        } else {
            out.extend(split_quotes(p, prefix_len.saturating_sub(at)));
        }
    }
    out
}

/// Qwen2.5-VL `get_rope_index` positions of one sequence with one image (`grid` merged tokens
/// starting at the first `image_token`): tokens with id 0 count as padding (the pipeline's
/// `attention_mask = input_ids > 0`), keep position 1 and are skipped.
pub fn rope_positions(ids: &[u32], image_token: u32, grid: (usize, usize)) -> Vec<[usize; 3]> {
    let (gh, gw) = grid;
    let mut out = vec![[1usize; 3]; ids.len()];
    let (mut next, mut k, mut base) = (0usize, 0usize, 0usize);
    for (i, &id) in ids.iter().enumerate() {
        if id == 0 {
            continue;
        }
        if id == image_token && k < gh * gw {
            if k == 0 {
                base = next;
            }
            out[i] = [base, base + k / gw, base + k % gw];
            k += 1;
            if k == gh * gw {
                next = base + gh.max(gw);
            }
            continue;
        }
        out[i] = [next; 3];
        next += 1;
    }
    out
}

/// A prompt encoded for the transformer.
struct Encoded {
    /// `[n, 3584]` last hidden states (after the dropped prefix).
    hidden: Vec<f32>,
    /// `[n, 3584]` raw token embeddings where `mask` (quoted characters), zero elsewhere.
    embeds: Vec<f32>,
    mask: Vec<bool>,
    /// `[n, 3]` RoPE ids of the text tokens.
    txt_ids: Vec<f32>,
    n: usize,
    /// Tokens before the padding.
    valid: usize,
}

impl Pipeline {
    /// Opens a Step1X-Edit diffusers folder (or a Hugging Face repo id, fetched into the cache):
    /// configs and the VAE now, the text encoder and transformer on first use.
    pub fn load(path_or_repo: &str, opts: Options) -> Result<Self> {
        let dir = crate::engine::resolve_model(path_or_repo, None)?;
        if !is_step1x(&dir) {
            bail!("{}: not a Step1X-Edit diffusers pipeline", dir.display());
        }
        if !crate::gpu::initialized() {
            crate::gpu::init(&crate::platform::detect(), opts.device);
        }
        let tdir = match &opts.transformer {
            Some(p) if p.is_dir() => p.clone(),
            _ => dir.join("transformer"),
        };
        let tcfg = Config::load(&tdir).with_context(|| format!("{}/config.json", tdir.display()))?;
        let version = if tcfg.text_token_mapping { Version::V1p2 } else { Version::V1p1 };
        let sched = FlowMatchEuler::load(&dir.join("scheduler").join("scheduler_config.json"))?;
        let vae = Vae::load(&dir.join("vae"), true)?;
        if vae.latent_channels * 4 != tcfg.in_channels {
            bail!(
                "VAE latent channels {} do not match the transformer input {}",
                vae.latent_channels,
                tcfg.in_channels
            );
        }
        Ok(Self {
            dir,
            version,
            opts,
            sched,
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

    /// Loads the text encoder (if not loaded).
    pub fn load_text_encoder(&mut self) -> Result<f64> {
        if self.text.is_some() {
            return Ok(0.0);
        }
        let t0 = Instant::now();
        let pdir = self.dir.join("processor");
        let has = |f: &str| crate::fsx::exists(&pdir.join(f));
        let vo = QwenVlOptions {
            quant: self.opts.text_quant,
            mmap: self.opts.mmap,
            tokenizer_dir: has("tokenizer.json").then(|| pdir.clone()),
            processor_dir: has("preprocessor_config.json").then(|| pdir.clone()),
            vision: true,
        };
        let vl = QwenVl::load(&self.dir.join("text_encoder"), &vo)?;
        if vl.hidden_size() != self.tcfg.connector_in {
            bail!(
                "text encoder width {} does not match the connector's {}",
                vl.hidden_size(),
                self.tcfg.connector_in
            );
        }
        self.text = Some(vl);
        Ok(t0.elapsed().as_secs_f64())
    }

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
        let in_memory = !(self.opts.mmap || self.opts.stream.is_some());
        let mut t = Transformer::load(&src, self.tcfg.clone(), self.opts.quant, in_memory)?;
        let stream = match &self.opts.stream {
            Some(so) => {
                let s = crate::stream::Streamer::new(&[t.files()], t.stream_units(), so.clone())?;
                let streamed: Vec<bool> = s.plan().units.iter().map(|u| !u.2).collect();
                t.mark_streamed(&streamed);
                t.set_stream(Some(crate::stream::StreamHook::new(s.clone(), 0)));
                Some(s)
            }
            None => None,
        };
        self.tr = Some(LoadedTransformer { t, _stream: stream });
        Ok(t0.elapsed().as_secs_f64())
    }

    pub fn unload_transformer(&mut self) {
        self.tr = None;
    }

    /// The transformer, once loaded.
    pub fn transformer(&self) -> Option<&Transformer> {
        self.tr.as_ref().map(|t| &t.t)
    }

    /// The text encoder, while loaded.
    pub fn text_encoder(&self) -> Option<&QwenVl> {
        self.text.as_ref()
    }

    pub fn vae(&self) -> &Vae {
        &self.vae
    }

    /// The added (special) tokens of the tokenizer, as the pipeline's `_special_token_strings`.
    fn specials(vl: &QwenVl) -> Vec<String> {
        vl.tokenizer
            .get_added_tokens_decoder()
            .values()
            .map(|t| t.content.clone())
            .collect()
    }

    /// Token ids of the split prompt: each piece tokenized on its own, the image placeholder
    /// expanded. Returns the ids and the quoted-character mask.
    fn tokenize_pieces(vl: &QwenVl, pieces: &[(String, bool)], img: &VlImage) -> Result<(Vec<u32>, Vec<bool>)> {
        let (mut ids, mut mask) = (Vec::new(), Vec::new());
        for (p, quoted) in pieces {
            let t = if p.contains("<|image_pad|>") {
                vl.tokenize(&p.replace("<|image_pad|>", &"<|image_pad|>".repeat(img.n)))?
            } else {
                vl.tokenize(p)?
            };
            mask.extend(std::iter::repeat_n(*quoted, t.len()));
            ids.extend(t);
        }
        Ok((ids, mask))
    }

    /// v1.2's `_get_qwenvl_embeds` + `encode_prompt` for one instruction. `src` is the size of
    /// the image given to the vision tower (before the processor's resize).
    fn encode_v12(&self, vl: &QwenVl, prompt: &str, img: &VlImage, src: (usize, usize), name: &str) -> Result<Encoded> {
        let mm = vl.mm.as_ref().context("the text encoder has no vision config")?;
        let text = format!(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{PREFIX_V12} {prompt}Picture 1: <|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n"
        );
        let pieces = split_prompt(&text, PREFIX_V12.chars().count(), &Self::specials(vl));
        let (ids, qmask) = Self::tokenize_pieces(vl, &pieces, img)?;
        let pos = rope_positions(&ids, mm.image_token_id, img.grid);
        if dump_dir().is_some() {
            dump(&format!("{name}_ids.bin"), &ids.iter().map(|&i| i as f32).collect::<Vec<_>>());
            dump(
                &format!("{name}_qmask.bin"),
                &qmask.iter().map(|&m| m as u8 as f32).collect::<Vec<_>>(),
            );
        }
        let l = ids.len();
        if l <= DROP_V12 {
            bail!("the prompt template is shorter than the {DROP_V12} tokens it drops");
        }
        let h = vl.hidden_states_tokens(ids.clone(), std::slice::from_ref(img), Some(pos.clone()), &[])?;
        let d = h.hidden;
        let n = (l - DROP_V12).min(MAX_TOKENS_V12);
        let keep = DROP_V12..DROP_V12 + n;
        let hidden = h.last[keep.start * d..keep.end * d].to_vec();
        let mask: Vec<bool> = qmask[keep.clone()].to_vec();
        let mut embeds = vec![0f32; n * d];
        for (r, (&m, &id)) in mask.iter().zip(&ids[keep.clone()]).enumerate() {
            if m {
                vl.model().embed.row(id as usize, &mut embeds[r * d..(r + 1) * d]);
            }
        }
        // RoPE ids: the time axis of the VL positions shifted to end at -1; vision tokens get the
        // latent-patch coordinates of the pixels they cover.
        let p0: Vec<f64> = pos[keep.clone()].iter().map(|p| p[0] as f64).collect();
        let top = p0.iter().cloned().fold(f64::MIN, f64::max) + 1.0;
        let (vh, vw) = img.grid;
        let (rw, rh) = img.size;
        let (sw, sh) = src;
        let sx = (rw as f64 - 1.0) / (sw as f64 - 1.0);
        let sy = (rh as f64 - 1.0) / (sh as f64 - 1.0);
        let mut txt_ids = vec![0f32; n * 3];
        let mut k = 0usize;
        for (r, &id) in ids[keep.clone()].iter().enumerate() {
            txt_ids[r * 3] = (p0[r] - top) as f32;
            if id == mm.image_token_id && k < vh * vw {
                let (y, x) = ((k / vw) as f64, (k % vw) as f64);
                txt_ids[r * 3 + 1] = (((28.0 * y + 13.5) / sy) / 16.0 - 11.0 / 16.0) as f32;
                txt_ids[r * 3 + 2] = (((28.0 * x + 13.5) / sx) / 16.0 - 11.0 / 16.0) as f32;
                k += 1;
            }
        }
        Ok(Encoded {
            hidden,
            embeds,
            mask,
            txt_ids,
            n,
            valid: n,
        })
    }

    /// v1.1's encoding: prefix, image and instruction; 217 tokens dropped, then padded to 640
    /// with zeros (the padding is masked in the connector, not in the DiT); RoPE ids all 0.
    fn encode_v11(&self, vl: &QwenVl, prompt: &str, img: &VlImage, name: &str) -> Result<Encoded> {
        let text = format!(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{PREFIX_V11}<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n"
        );
        let ids = Self::v11_ids(vl, &text, img)?;
        if dump_dir().is_some() {
            dump(&format!("{name}_ids.bin"), &ids.iter().map(|&i| i as f32).collect::<Vec<_>>());
        }
        let l = ids.len();
        if l <= DROP_V11 {
            bail!("the prompt template is shorter than the {DROP_V11} tokens it drops");
        }
        let last = Self::masked_hidden(vl, &ids, img)?;
        let d = vl.hidden_size();
        let valid = (l - DROP_V11).min(MAX_TOKENS_V11);
        let mut hidden = vec![0f32; MAX_TOKENS_V11 * d];
        hidden[..valid * d].copy_from_slice(&last[DROP_V11 * d..(DROP_V11 + valid) * d]);
        Ok(Encoded {
            hidden,
            embeds: Vec::new(),
            mask: vec![false; MAX_TOKENS_V11],
            txt_ids: vec![0f32; MAX_TOKENS_V11 * 3],
            n: MAX_TOKENS_V11,
            valid,
        })
    }

    /// v1.1's text encoder call: `attention_mask = input_ids > 0`, so tokens with id 0 ("!")
    /// are padding: no other token sees them, they keep position 1 and the rest close the gap.
    /// The other tokens' states are exact (the sequence runs without them); a padding token's
    /// own state comes from a pass that sees the tokens before it and itself (the reference's
    /// row also leaves itself out).
    fn masked_hidden(vl: &QwenVl, ids: &[u32], img: &VlImage) -> Result<Vec<f32>> {
        if !ids.contains(&0) {
            return Ok(vl.hidden_states_tokens(ids.to_vec(), std::slice::from_ref(img), None, &[])?.last);
        }
        let mm = vl.mm.as_ref().context("the text encoder has no vision config")?;
        let d = vl.hidden_size();
        let model = vl.model();
        let full = rope_positions(ids, mm.image_token_id, img.grid);
        let cache_pos: Vec<[usize; 3]> = ids.iter().zip(&full).filter(|(i, _)| **i != 0).map(|(_, p)| *p).collect();
        let feats = vec![crate::mm::MmFeature {
            token_id: mm.image_token_id,
            emb: img.emb.clone(),
            n: img.n,
            grids: vec![img.grid],
        }];
        let mut sess = model.new_session();
        let mut out = vec![0f32; ids.len() * d];
        let mut i = 0;
        while i < ids.len() {
            if ids[i] == 0 {
                let at = sess.pos;
                let mut p = cache_pos.clone();
                p.insert(at, [1, 1, 1]);
                sess.positions = Some(p);
                let h = model.forward_hidden_rec(&mut sess, &[0]);
                out[i * d..(i + 1) * d].copy_from_slice(&h);
                sess.rewind(at)?;
                i += 1;
                continue;
            }
            let j = ids[i..].iter().position(|&t| t == 0).map_or(ids.len(), |k| i + k);
            let run = &ids[i..j];
            sess.positions = Some(cache_pos.clone());
            let used = ids[..i].iter().filter(|&&t| t == mm.image_token_id).count();
            let ov = if run.contains(&mm.image_token_id) {
                if used > 0 {
                    bail!("an image split by a padding token");
                }
                crate::mm::overrides(run, &feats, d)?
            } else {
                Vec::new()
            };
            let refs: Vec<(usize, &[f32])> = ov.iter().map(|(k, v)| (*k, v.as_slice())).collect();
            let h = model.forward_hidden_mm(&mut sess, run, &refs);
            out[i * d..j * d].copy_from_slice(&h);
            i = j;
        }
        Ok(out)
    }

    /// v1.1's token ids: the template tokenized whole up to `<|vision_end|>`, then the rest
    /// re-tokenized in pieces with quoted characters one by one (`_split_string`).
    fn v11_ids(vl: &QwenVl, text: &str, img: &VlImage) -> Result<Vec<u32>> {
        const VISION_END: u32 = 151653;
        let full = vl.tokenize(&text.replace("<|image_pad|>", &"<|image_pad|>".repeat(img.n)))?;
        let s: String = text.replace('\'', "\"").replace(['\u{201c}', '\u{201d}'], "\"");
        let mut pieces: Vec<String> = Vec::new();
        let (mut inq, mut tmp) = (false, String::new());
        for (i, ch) in s.chars().enumerate() {
            if ch == '"' && i > 155 {
                tmp.push(ch);
                if !inq {
                    pieces.push(std::mem::take(&mut tmp));
                }
                inq = !inq;
                continue;
            }
            if inq {
                if ch.is_whitespace() {
                    // the reference keeps whitespace inside quotes as its own token
                }
                pieces.push(format!("\u{201c}{ch}\u{201d}"));
            } else {
                tmp.push(ch);
            }
        }
        if !tmp.is_empty() {
            pieces.push(tmp);
        }
        let mut new = Vec::new();
        for p in &pieces {
            let mut t = vl.tokenize(p)?;
            if t.len() >= 2 && t[0] == 2073 && t[t.len() - 1] == 854 {
                t = t[1..t.len() - 1].to_vec();
            }
            new.extend(t);
        }
        let i1 = full.iter().position(|&t| t == VISION_END).context("no <|vision_end|>")?;
        let i2 = new.iter().position(|&t| t == VISION_END).context("no <|vision_end|>")?;
        let mut ids = full[..i1].to_vec();
        ids.extend_from_slice(&new[i2..]);
        Ok(ids)
    }

    /// Thinking mode: the VL model rewrites `instruction` as a clear edit instruction from the
    /// image (`ImageEditEvaluator_EN.prompt_reformat`; sampled as the reference).
    fn think(&self, vl: &QwenVl, image: &RgbImage, instruction: &str, seed: u64) -> Result<String> {
        let mm = vl.mm.as_ref().context("the text encoder has no vision config")?;
        // image_to_data_url: at most 1024 on the long side.
        let (mut w, mut h) = (image.width, image.height);
        let rgb = if w.max(h) > 1024 {
            let s = 1024.0 / w.max(h) as f64;
            let (tw, th) = (((w as f64 * s).round() as usize).max(1), ((h as f64 * s).round() as usize).max(1));
            let r = crate::resample::resize_bicubic_rgb8(&image.data, w, h, tw, th);
            (w, h) = (tw, th);
            r
        } else {
            image.data.clone()
        };
        let img = vl.encode_image(&rgb, w, h, 802_816)?;
        let zh = instruction.chars().filter(|c| ('\u{4e00}'..='\u{9fff}').contains(c)).count() > 1;
        let lang = if zh { "中文" } else { "英文" };
        let system = "You are an Edit Task Instruction Rewriter. Your role is to maintain the original meaning while transforming vague, colloquial, or imprecise instructions into clear, standardized, and actionable ones. If the instructions are already sufficiently clear, they should be returned without modification.";
        let user = format!(
            "You are an 'editing task instruction rewriter'. Your task is:\nWhile keeping the meaning completely unchanged, convert the user's vague, colloquial, or informal editing instruction into a clear, standard, and actionable instruction.\nWhen the user's input contains complex meanings, break it down into two to three simple, clear editing instructions. Ensure these instructions do not conflict with each other and retain only the meaningful ones. Output only the rewritten content.\nThe final editing instruction should preferably contain only one editing action. If the meaning is too complex to be described with a single action, the total number of actions must not exceed three.\nIf the input includes explicit text modifications, enclose the text to be changed in double quotes.\nEditing types mainly include: object addition, object replacement, object deletion, color modification, background replacement, style change, material change, and text replacement.\nWhere possible, the final instruction should include the editing location, the action (see the editing types), and the object being edited. Follow natural-language conventions; numbering is not allowed\u{2014}use natural-language sequencing instead.\nIf the user's input requires understanding and reasoning, think about where the final image will change based on the user's description, rather than returning a result prior to such reasoning.\nRewrite in the style of the following examples.\nExample 1: User input: 'Symptoms of potassium deficiency in leaves' \u{2192} Rewritten: 'Leaves turn yellow, and the tips dry out.'\nExample 2: User input: 'Make the bread look over-fried' \u{2192} Rewritten: 'Change the bread\u{2019}s color to a dark, toasted brown.'\nExample 3: User input: 'Photoshop me a boyfriend; make it a two-person photo' \u{2192} Rewritten: 'Add a man next to the woman.'\nExample 4: User input: 'Improve this image, remove the distant power lines, keep a realistic style' \u{2192} Rewritten: 'Remove the power lines.'\nExample 5: User input: 'Change \u{201c}TOFEL\u{201d} in the image to \u{201c}IELTS\u{201d}' \u{2192} Rewritten: 'Replace the text \"TOFEL\" with \"IELTS\".'\nExample 6: User input: 'Make it an ID photo.' \u{2192} Rewritten: 'Replace the background with white.'\nExample 7: User input: 'Change the tie to black.' \u{2192} Rewritten: 'Change the tie color to black.'\nExample 8: User input: 'Remove the watermark in the bottom-right corner.' \u{2192} Rewritten: 'Remove the bottom-right watermark.'\nExample 9: User input: 'Increase hair volume. Change the hairstyle to long hair. Make the person look gentler.' \u{2192} Rewritten: 'Increase the amount of hair to make the person look gentler.'\nUser input instruction:{instruction}\nYour returned instruction must be in {lang}. Your returned instruction must be in {lang}. Your returned instruction must be in {lang}.\nThe returned instruction is:"
        );
        let text = format!(
            "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{}{user}<|im_end|>\n<|im_start|>assistant\n",
            img.placeholder()
        );
        let ids = vl.tokenize(&text)?;
        let model = vl.model();
        let feats = vec![crate::mm::MmFeature {
            token_id: mm.image_token_id,
            emb: img.emb.clone(),
            n: img.n,
            grids: vec![img.grid],
        }];
        let mut sess = model.new_session();
        for (p, gh, gw) in crate::mm::vision_spans(&ids, &feats)? {
            sess.add_vision_span(p, gh, gw);
        }
        let ov = crate::mm::overrides(&ids, &feats, vl.hidden_size())?;
        let refs: Vec<(usize, &[f32])> = ov.iter().map(|(i, v)| (*i, v.as_slice())).collect();
        let hs = model.forward_hidden_mm(&mut sess, &ids, &refs);
        let d = vl.hidden_size();
        let mut logits = model.logits(&hs[(ids.len() - 1) * d..], 1);
        let mut sampler = crate::sampler::Sampler::new(0.7, 0, 0.9, seed);
        let eos = vl.tokenizer.token_to_id("<|im_end|>").unwrap_or(151645);
        let mut out = Vec::new();
        for _ in 0..128 {
            let t = sampler.sample(&logits);
            if t == eos || t == 151643 {
                break;
            }
            out.push(t);
            logits = model.forward(&mut sess, &[t]);
        }
        let s = vl.tokenizer.decode(&out, true).map_err(|e| anyhow::anyhow!("decode: {e}"))?;
        Ok(s.trim().to_string())
    }

    /// Edits `image` following the instruction in `p.prompt`.
    pub fn edit(&mut self, image: &RgbImage, p: &Params, progress: &mut dyn FnMut(usize, usize)) -> Result<(Image, Timings)> {
        if image.width < 16 || image.height < 16 || image.data.len() != image.width * image.height * 3 {
            bail!("the input image must be RGB8 and at least 16x16");
        }
        self.run(Some(image), p, progress)
    }

    /// Text to image (the pipeline without an input image: the vision tower sees a black
    /// square).
    pub fn generate(&mut self, p: &Params, progress: &mut dyn FnMut(usize, usize)) -> Result<(Image, Timings)> {
        self.run(None, p, progress)
    }

    /// [`Pipeline::edit`] on an oxideav picture.
    #[cfg(feature = "oxideav")]
    pub fn edit_picture(
        &mut self,
        picture: &crate::av::Picture,
        p: &Params,
        progress: &mut dyn FnMut(usize, usize),
    ) -> Result<(Image, Timings)> {
        self.edit(&picture.to_rgb()?, p, progress)
    }

    fn run(&mut self, image: Option<&RgbImage>, p: &Params, progress: &mut dyn FnMut(usize, usize)) -> Result<(Image, Timings)> {
        self.cancel.store(false, Ordering::Relaxed);
        let mut tm = Timings::default();
        let level = p.size_level.max(16);
        // Working size, the vision tower's image and the VAE's.
        let (w, h, vl_rgb, vl_size, vae_px, out_size) = match image {
            Some(img) => {
                let (w, h) = working_size(img.width, img.height, level);
                if w < 16 || h < 16 {
                    bail!("working size {w}x{h} is too small");
                }
                let vl_rgb = crate::resample::resize_lanczos_rgb8(&img.data, img.width, img.height, w, h);
                // The VAE's input: v1.2 resizes the original with torch's antialiased bilinear,
                // v1.1 takes the Lanczos-resized image the vision tower gets.
                let px: Vec<f32> = match self.version {
                    Version::V1p2 => {
                        let unit: Vec<f32> = img.data.iter().map(|&v| v as f32 / 255.0).collect();
                        resize_bilinear_aa(&unit, img.width, img.height, w, h)
                            .iter()
                            .map(|&v| v * 2.0 - 1.0)
                            .collect()
                    }
                    Version::V1p1 => vae::rgb8_to_unit(&vl_rgb),
                };
                (
                    w,
                    h,
                    vl_rgb,
                    (w, h),
                    Some(px),
                    (p.width.unwrap_or(img.width), p.height.unwrap_or(img.height)),
                )
            }
            None => {
                let (w, h) = (p.width.unwrap_or(level) / 16 * 16, p.height.unwrap_or(level) / 16 * 16);
                if w < 16 || h < 16 {
                    bail!("output size {w}x{h} is too small");
                }
                (w, h, vec![0u8; level * level * 3], (level, level), None, (w, h))
            }
        };
        tm.size = (w, h);
        let negative = p
            .negative
            .clone()
            .unwrap_or_else(|| if image.is_some() { String::new() } else { T2I_NEGATIVE.to_string() });
        let guided = p.true_cfg > 1.0;

        // Prompts (exact GPU matmuls: the VL's hidden states have large outlier channels).
        tm.load_text = self.load_text_encoder()?;
        let was_exact = crate::gpu::exact();
        crate::gpu::set_exact(true);
        let t_enc = Instant::now();
        let enc = (|| -> Result<(Encoded, Option<Encoded>, Option<String>, f64)> {
            let vl = self.text.as_ref().unwrap();
            let t0 = Instant::now();
            let mut prompt = p.prompt.clone();
            let mut rewritten = None;
            if p.think
                && let Some(img) = image
            {
                prompt = self.think(vl, img, &p.prompt, p.seed)?;
                rewritten = Some(prompt.clone());
            }
            let think_s = t0.elapsed().as_secs_f64();
            let vimg = match self.version {
                Version::V1p2 => vl.encode_image(&vl_rgb, vl_size.0, vl_size.1, 0)?,
                Version::V1p1 => {
                    // The pipeline's own resize to multiples of 28 (PIL bicubic) first.
                    let (bw, bh) = v11_vision_size(vl_size.0, vl_size.1);
                    let r = crate::resample::resize_bicubic_rgb8(&vl_rgb, vl_size.0, vl_size.1, bw, bh);
                    vl.encode_image(&r, bw, bh, 0)?
                }
            };
            dump("vl_image.bin", &vimg.emb);
            let enc = |s: &str, name: &str| -> Result<Encoded> {
                match self.version {
                    Version::V1p2 => self.encode_v12(vl, s, &vimg, vl_size, name),
                    Version::V1p1 => self.encode_v11(vl, s, &vimg, name),
                }
            };
            let pos = enc(&prompt, "prompt")?;
            let neg = if guided { Some(enc(&negative, "negative")?) } else { None };
            Ok((pos, neg, rewritten, think_s))
        })();
        crate::gpu::set_exact(was_exact);
        let (mut pos, mut neg, rewritten, think_s) = enc?;
        tm.think = think_s;
        tm.rewritten = rewritten;
        if let Some(v) = read_override("prompt_hidden_in.bin", Some(pos.hidden.len()))? {
            pos.hidden = v;
        }
        if let (Some(n), Some(v)) = (neg.as_mut(), read_override("negative_hidden_in.bin", None)?) {
            n.hidden = v;
        }
        dump("prompt_hidden.bin", &pos.hidden);
        dump("prompt_txt_ids.bin", &pos.txt_ids);
        if let Some(n) = &neg {
            dump("negative_hidden.bin", &n.hidden);
            dump("negative_txt_ids.bin", &n.txt_ids);
        }
        tm.text = t_enc.elapsed().as_secs_f64() - think_s;
        if !self.opts.keep_text_encoder {
            self.unload_text_encoder();
        }
        self.check_cancel()?;

        // Reference latents.
        let t1 = Instant::now();
        let c = self.vae.latent_channels;
        let (lh, lw) = (2 * (h / 16), 2 * (w / 16));
        let ref_tok = match &vae_px {
            Some(px) => {
                dump("vae_in.bin", px);
                let z = match read_override("ref_latents_in.bin", Some(c * lh * lw))? {
                    Some(v) => vae::chw_to_hwc(&v, c, lh, lw),
                    None => {
                        // Exact GPU matmuls: the encoder's latents are sensitive to rounding
                        // (half-precision activations move them by 8e-4, bf16 weights by 1.5%).
                        let was = crate::gpu::exact();
                        crate::gpu::set_exact(true);
                        let r = self.vae.encode(px, h, w);
                        crate::gpu::set_exact(was);
                        r?.0
                    }
                };
                dump("ref_latents.bin", &vae::hwc_to_chw(&z, c, lh, lw));
                let t = vae::pack(&z, lh, lw, c);
                dump("ref_tokens.bin", &t);
                t
            }
            None => Vec::new(),
        };
        tm.vae_encode = t1.elapsed().as_secs_f64();

        tm.load_transformer = self.load_transformer()?;
        let t2 = Instant::now();
        let tr = &self.tr.as_ref().unwrap().t;
        let cin = self.tcfg.in_channels;
        let n_out = (lh / 2) * (lw / 2);
        let n_img = n_out + ref_tok.len() / cin;
        let noise = match read_override("latents_in.bin", Some(c * lh * lw))? {
            Some(v) => v,
            None => crate::sd::ops::Rng::new(p.seed).normals(c * lh * lw),
        };
        let mut x = vae::pack(&vae::chw_to_hwc(&noise, c, lh, lw), lh, lw, c);
        let rope_for = |e: &Encoded| {
            let mut ids = e.txt_ids.clone();
            // (index, offset, rows, columns): the target, then the reference at index 1 (v1.1
            // also shifts it by one row and column).
            let mut grids = vec![(0.0f32, 0.0f32, lh / 2, lw / 2)];
            if !ref_tok.is_empty() {
                let off = if self.version == Version::V1p1 { 1.0 } else { 0.0 };
                grids.push((1.0, off, lh / 2, lw / 2));
            }
            for (idx, off, gh, gw) in grids {
                for y in 0..gh {
                    for xx in 0..gw {
                        ids.extend_from_slice(&[idx, y as f32 + off, xx as f32 + off]);
                    }
                }
            }
            Rope::new(&ids, &self.tcfg.axes, self.tcfg.theta)
        };
        let prep = |e: &Encoded| tr.prepare(&e.hidden, e.n, e.valid, &e.embeds, &e.mask);
        let (cpos, rpos): (Cond, Rope) = (prep(&pos), rope_for(&pos));
        let cneg: Option<(Cond, Rope)> = neg.as_ref().map(|e| (prep(e), rope_for(e)));
        tm.tokens = pos.n + n_img;
        drop((pos, neg));
        let plan = self.sched.plan(&flux_sigmas(p.steps.max(1)), Some(self.sched.mu(n_out)))?;
        let steps = plan.timesteps.len();
        struct AttnGuard(bool);
        impl Drop for AttnGuard {
            fn drop(&mut self) {
                crate::gpu::set_exact_attention(self.0);
            }
        }
        let _attn = AttnGuard(crate::gpu::set_exact_attention(self.opts.exact_attention));
        let mut input = vec![0f32; n_img * cin];
        input[n_out * cin..].copy_from_slice(&ref_tok);
        for i in 0..steps {
            self.check_cancel()?;
            input[..n_out * cin].copy_from_slice(&x);
            let (t, sigma) = (plan.timesteps[i], plan.sigmas[i]);
            let mut v = tr.forward(&input, n_img, &cpos, sigma, &rpos);
            tm.transformer_calls += 1;
            v.truncate(n_out * cin);
            if let Some((cn, rn)) = &cneg {
                let mut u = tr.forward(&input, n_img, cn, sigma, rn);
                tm.transformer_calls += 1;
                u.truncate(n_out * cin);
                if i == 0 {
                    dump("v0_cond.bin", &v);
                    dump("v0_uncond.bin", &u);
                }
                let g = p.true_cfg;
                let rescale = t > p.timesteps_truncate;
                crate::kernels::for_chunks(&mut v, cin, true, |r, row| {
                    let un = &u[r * cin..(r + 1) * cin];
                    let mut f = 1.0f32;
                    if rescale {
                        let nrm = row.iter().zip(un).map(|(&a, &b)| (a - b) * (a - b)).sum::<f32>().sqrt();
                        if nrm > 1.0 {
                            f = nrm.powf(p.norm_power);
                        }
                    }
                    for (a, &b) in row.iter_mut().zip(un) {
                        *a = b + g * (*a - b) / f;
                    }
                });
            }
            if i == 0 {
                dump("v0.bin", &v);
            }
            FlowMatchEuler::step(&plan, i, &mut x, &v);
            progress(i + 1, steps);
        }
        tm.denoise = t2.elapsed().as_secs_f64();

        let t3 = Instant::now();
        let z = vae::unpack(&x, lh, lw, c);
        dump("latents_out.bin", &vae::hwc_to_chw(&z, c, lh, lw));
        let (px, ph, pw) = self.vae.decode(&z, lh, lw);
        dump("image.bin", &vae::hwc_to_chw(&px, 3, ph, pw));
        let mut rgb = vae::unit_to_rgb8(&px);
        let (ow, oh) = out_size;
        let (ow, oh) = (ow.max(1), oh.max(1));
        if (ow, oh) != (pw, ph) {
            rgb = crate::resample::resize_bicubic_rgb8(&rgb, pw, ph, ow, oh);
        }
        tm.decode = t3.elapsed().as_secs_f64();
        Ok((
            Image {
                width: ow,
                height: oh,
                rgb,
            },
            tm,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn working_sizes() {
        // encode_image(size_level=1024): 4:3 -> ceil(sqrt(1024^2 * 4/3)) = 1183 -> 1168 x 880.
        assert_eq!(working_size(800, 600, 1024), (1168, 880));
        assert_eq!(working_size(600, 800, 1024), (880, 1168));
        assert_eq!(working_size(512, 512, 1024), (1024, 1024));
        assert_eq!(working_size(1000, 500, 512), (720, 352));
    }

    #[test]
    fn quote_split() {
        let s = |v: &[(&str, bool)]| v.iter().map(|(a, b)| (a.to_string(), *b)).collect::<Vec<_>>();
        assert_eq!(split_quotes("ab", 0), s(&[("ab", false)]));
        assert_eq!(
            split_quotes("say \"hi\" now", 0),
            s(&[("say \"", false), ("h", true), ("i", true), ("\" now", false)])
        );
        // Inside the prefix nothing splits; an unterminated quote is glued back.
        assert_eq!(split_quotes("\"x\" \"y", 3), s(&[("\"x\" \"y", false)]));
        assert_eq!(
            split_quotes("\u{201c}ab\u{201d}c", 0),
            s(&[("a", true), ("b", true), ("\u{201d}c", false)])
        );
        let sp = vec!["<|im_start|>".to_string(), "<|im_end|>".to_string()];
        assert_eq!(
            split_prompt("<|im_start|>u \"a\"<|im_end|>", 0, &sp),
            s(&[
                ("<|im_start|>", false),
                ("u \"", false),
                ("a", true),
                ("\"", false),
                ("<|im_end|>", false)
            ])
        );
    }

    #[test]
    fn rope_index() {
        // text, text, [image 2x3], text: the image at 2, rows / columns offset; text continues at
        // 2 + max(2, 3) = 5.
        let ids = [5, 6, 9, 9, 9, 9, 9, 9, 7];
        let p = rope_positions(&ids, 9, (2, 3));
        assert_eq!(p[0], [0, 0, 0]);
        assert_eq!(p[2], [2, 2, 2]);
        assert_eq!(p[7], [2, 3, 4]);
        assert_eq!(p[8], [5, 5, 5]);
        // A token 0 is skipped (position 1) and the rest closes the gap.
        let p = rope_positions(&[5, 0, 6], 9, (1, 1));
        assert_eq!(p, vec![[0, 0, 0], [1, 1, 1], [1, 1, 1]]);
    }

    #[test]
    fn bilinear_aa() {
        // torch: interpolate([0, 1, 2, 3], 2, bilinear, antialias) = [0.7143, 2.2857]; constant
        // images stay constant.
        let x: Vec<f32> = [0.0f32, 1.0, 2.0, 3.0].iter().flat_map(|&v| [v, v, v]).collect();
        let y = resize_bilinear_aa(&x, 4, 1, 2, 1);
        assert!((y[0] - 0.714_285_7).abs() < 1e-6 && (y[3] - 2.285_714_3).abs() < 1e-6, "{y:?}");
        let c = vec![0.5f32; 7 * 5 * 3];
        assert!(resize_bilinear_aa(&c, 7, 5, 3, 9).iter().all(|&v| (v - 0.5).abs() < 1e-6));
    }
}
