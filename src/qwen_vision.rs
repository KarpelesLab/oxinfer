//! Qwen3.5 / 3.6 / 3.8 vision (`Qwen3_5ForConditionalGeneration`'s `model.visual`): the image and
//! video processors (`Qwen2VLImageProcessor` / `Qwen3VLVideoProcessor`: smart resize to multiples
//! of 32 pixels within a pixel budget, uint8 bicubic, normalization to [-1, 1], 16x16 patches
//! with 2 temporal frames each, emitted in 2x2 spatial-merge blocks), and the ViT: a Conv3d patch
//! embedding, learned position embeddings bilinearly resampled to the patch grid, axial 2D RoPE,
//! pre-LayerNorm blocks with biased projections and bidirectional attention within each frame,
//! and the merger that folds every 2x2 block into one language-model token.
//!
//! The language model places the tokens with M-RoPE (see `QwenSession::add_vision_span`).
//! DeepStack (Qwen3-VL's `deepstack_visual_indexes`: the tower's hidden states after a few of
//! its layers, each through its own merger, added to the image tokens after the language
//! model's first layers) is computed for conditioning encoders ([`Qwen35Tower::encode_deep`],
//! `crate::qwen_vl`); the Qwen3.5-family checkpoints have none, and the chat engine does not
//! run it.

use crate::bert::{LayerNorm, gelu_erf};
use crate::kernels::*;
use crate::model::{Lin, Quant, linear, load_lin};
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::path::{Path, PathBuf};

fn d3() -> usize {
    3
}
fn d16() -> usize {
    16
}
fn d2() -> usize {
    2
}
fn d2304() -> usize {
    2304
}

/// `vision_config` of a Qwen3.5-family checkpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct QwenVisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    #[serde(default)]
    pub hidden_act: Option<String>,
    pub intermediate_size: usize,
    pub num_heads: usize,
    #[serde(default = "d3")]
    pub in_channels: usize,
    #[serde(default = "d16")]
    pub patch_size: usize,
    #[serde(default = "d2")]
    pub spatial_merge_size: usize,
    #[serde(default = "d2")]
    pub temporal_patch_size: usize,
    pub out_hidden_size: usize,
    #[serde(default = "d2304")]
    pub num_position_embeddings: usize,
    #[serde(default)]
    pub deepstack_visual_indexes: Vec<usize>,
    #[serde(default)]
    pub rope_parameters: Option<serde_json::Value>,
    /// Qwen2.5-VL: window attention over `window_size` pixels in every block but
    /// `fullatt_block_indexes` (see `crate::qwen25_vision`). Absent in Qwen3.5 towers.
    #[serde(default)]
    pub window_size: Option<usize>,
    #[serde(default)]
    pub fullatt_block_indexes: Vec<usize>,
    /// Qwen2.5-VL video: M-RoPE time steps per second.
    #[serde(default)]
    pub tokens_per_second: Option<f64>,
    /// MiniMax M3's CLIP-style tower with 3D RoPE (`crate::minimax_vision`).
    #[serde(default)]
    pub minimax_m3: bool,
    /// MiniMax M3: the projector's width, and the tower's LayerNorm epsilon.
    #[serde(default)]
    pub projector_hidden_size: Option<usize>,
    #[serde(default)]
    pub layer_norm_eps: Option<f64>,
}

impl QwenVisionConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }
    /// A Qwen2-VL / Qwen2.5-VL tower (window attention, RMSNorm, SwiGLU; `crate::qwen25_vision`).
    pub fn is_qwen25(&self) -> bool {
        self.window_size.is_some()
    }
    pub(crate) fn rope_theta(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .and_then(|r| r.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .unwrap_or(10000.0)
    }
}

/// Image or video processor settings (`preprocessor_config.json`,
/// `video_preprocessor_config.json`, or the `image_processor` / `video_processor` entries of
/// `processor_config.json`).
#[derive(Debug, Clone)]
pub struct ProcessorConfig {
    /// `size.shortest_edge` / `size.longest_edge`: the pixel budget (all frames of a video
    /// together for the video processor).
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
    /// Video only: frames sampled per second, and the frame-count bounds.
    pub fps: f64,
    pub min_frames: usize,
    pub max_frames: usize,
}

impl ProcessorConfig {
    /// The image processor settings every published Qwen3.5 / 3.6 checkpoint ships (used when a
    /// checkpoint lacks the file: the class defaults would be Qwen2-VL's 14-pixel patches).
    fn image_default() -> Self {
        Self {
            min_pixels: 65536,
            max_pixels: 16777216,
            mean: [0.5; 3],
            std: [0.5; 3],
            patch_size: 16,
            temporal_patch_size: 2,
            merge_size: 2,
            fps: 2.0,
            min_frames: 4,
            max_frames: 768,
        }
    }
    /// The video processor settings of the published checkpoints (`Qwen3VLVideoProcessor`'s
    /// sampling defaults).
    fn video_default() -> Self {
        Self {
            min_pixels: 4096,
            max_pixels: 25165824,
            ..Self::image_default()
        }
    }
    fn apply(&mut self, v: &serde_json::Value) {
        let num = |k: &str| v.get(k).and_then(|x| x.as_f64());
        if let Some(s) = v.get("size") {
            if let Some(x) = s.get("shortest_edge").and_then(|x| x.as_u64()) {
                self.min_pixels = x as usize;
            }
            if let Some(x) = s.get("longest_edge").and_then(|x| x.as_u64()) {
                self.max_pixels = x as usize;
            }
        }
        if let Some(x) = num("min_pixels") {
            self.min_pixels = x as usize;
        }
        if let Some(x) = num("max_pixels") {
            self.max_pixels = x as usize;
        }
        let triple = |k: &str| -> Option<[f32; 3]> {
            let a = v.get(k)?.as_array()?;
            (a.len() == 3).then(|| [0, 1, 2].map(|i| a[i].as_f64().unwrap_or(0.0) as f32))
        };
        if let Some(m) = triple("image_mean") {
            self.mean = m;
        }
        if let Some(s) = triple("image_std") {
            self.std = s;
        }
        let int = |k: &str| v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize);
        self.patch_size = int("patch_size").unwrap_or(self.patch_size);
        self.temporal_patch_size = int("temporal_patch_size").unwrap_or(self.temporal_patch_size);
        self.merge_size = int("merge_size").unwrap_or(self.merge_size);
        self.min_frames = int("min_frames").unwrap_or(self.min_frames);
        self.max_frames = int("max_frames").unwrap_or(self.max_frames);
        if let Some(f) = num("fps") {
            self.fps = f;
        }
    }

    /// Loads the image (`video` false) or video processor settings of a checkpoint, falling
    /// back the way transformers does: `processor_config.json` entry, then the dedicated file,
    /// then (video) `preprocessor_config.json`, then the published defaults.
    pub fn load(dir: &Path, video: bool) -> Self {
        let read = |name: &str| -> Option<serde_json::Value> {
            let b = crate::fsx::read(&dir.join(name)).ok()?;
            serde_json::from_slice(&b).ok()
        };
        let mut c = if video { Self::video_default() } else { Self::image_default() };
        let key = if video { "video_processor" } else { "image_processor" };
        if let Some(v) = read("processor_config.json").and_then(|p| p.get(key).cloned()) {
            c.apply(&v);
            return c;
        }
        let files: &[&str] = if video {
            &["video_preprocessor_config.json", "preprocessor_config.json"]
        } else {
            &["preprocessor_config.json"]
        };
        if let Some(v) = files.iter().find_map(|f| read(f)) {
            c.apply(&v);
        }
        c
    }
}

/// Multimodal settings of a Qwen3.5-family checkpoint.
#[derive(Debug, Clone)]
pub struct QwenMmConfig {
    pub vision: QwenVisionConfig,
    pub image: ProcessorConfig,
    pub video: ProcessorConfig,
    pub image_token_id: u32,
    pub video_token_id: u32,
    /// What surrounds and fills an image's placeholder in the prompt: start, one per token, end
    /// (`<|vision_start|>`, `<|image_pad|>`, `<|vision_end|>` for Qwen).
    pub markers: [&'static str; 3],
    /// The llama.cpp `mmproj` GGUF file the tower comes from (None: the checkpoint's own
    /// `model.visual.*` tensors).
    pub mmproj: Option<PathBuf>,
}

/// `clip.projector_type` of the llama.cpp mmproj files of the Qwen3-VL-style towers (Qwen3.5 /
/// 3.6 / 3.8, Qwen3.8-Flash-Next).
pub const MMPROJ_TYPE: &str = "qwen3vl_merger";
/// `clip.projector_type` of MiniMax M3's mmproj files (`crate::minimax_vision`).
pub const MMPROJ_MINIMAX_M3: &str = "minimax_m3";

impl QwenMmConfig {
    /// [`QwenMmConfig::load`] with the tower chosen by `mmproj` (`EngineOptions::mmproj`,
    /// `--mmproj`): the checkpoint's own tower, or for a GGUF model the `mmproj*.gguf` beside it
    /// when it is a Qwen3-VL-style projector (`auto`); a given mmproj file; or no vision (`none`).
    pub fn load_with(model: &Path, mmproj: &crate::pixtral::Mmproj) -> Result<Option<Self>> {
        use crate::pixtral::Mmproj;
        match mmproj {
            Mmproj::None => Ok(None),
            Mmproj::File(p) => Self::from_mmproj(p, model).map(Some),
            Mmproj::Auto if crate::gguf::is_gguf(model) => match crate::pixtral::mmproj_for(model) {
                Some(p) if matches!(crate::pixtral::projector_type(&p).as_deref(), Some(MMPROJ_TYPE | MMPROJ_MINIMAX_M3)) => {
                    Self::from_mmproj(&p, model).map(Some)
                }
                _ => Ok(None),
            },
            Mmproj::Auto if model.is_dir() => Self::load(model),
            Mmproj::Auto => Ok(None),
        }
    }

    /// The settings of llama.cpp's `qwen3vl_merger` mmproj file `path` for the language model
    /// `model` (a GGUF file or checkpoint folder, whose vocabulary gives the placeholder token
    /// ids). The processor settings are the published checkpoints' (pixel budget, 16-pixel
    /// patches, 2x2 merge), with the file's normalization.
    fn from_mmproj(path: &Path, model: &Path) -> Result<Self> {
        let h = crate::gguf::header(path)?;
        let proj = h.get("clip.projector_type").and_then(|v| v.as_str()).unwrap_or("");
        if proj == MMPROJ_MINIMAX_M3 {
            return Self::minimax_m3_mmproj(&h, path, model);
        }
        if proj != MMPROJ_TYPE {
            bail!(
                "{}: projector type {proj:?} is not a Qwen3-VL-style tower ({MMPROJ_TYPE}; pixtral files go with Mistral Large 4)",
                path.display()
            );
        }
        let u = |k: &str| {
            h.get(k)
                .and_then(|v| v.as_u64())
                .map(|x| x as usize)
                .with_context(|| format!("{}: no {k}", path.display()))
        };
        let hidden = u("clip.vision.embedding_length")?;
        let npos = h
            .tensors
            .iter()
            .find(|t| t.name == "v.position_embd.weight")
            .map(|t| t.ne.iter().product::<usize>() / hidden.max(1))
            .with_context(|| format!("{}: no v.position_embd.weight", path.display()))?;
        let deepstack: Vec<usize> = h
            .get("clip.vision.is_deepstack_layers")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .enumerate()
                    .filter(|(_, v)| matches!(v, crate::gguf::Value::Bool(true)))
                    .map(|(i, _)| i)
                    .collect()
            })
            .unwrap_or_default();
        let vision: QwenVisionConfig = serde_json::from_value(serde_json::json!({
            "depth": u("clip.vision.block_count")?,
            "hidden_size": hidden,
            "hidden_act": "gelu_pytorch_tanh",
            "intermediate_size": u("clip.vision.feed_forward_length")?,
            "num_heads": u("clip.vision.attention.head_count")?,
            "patch_size": u("clip.vision.patch_size")?,
            "spatial_merge_size": u("clip.vision.spatial_merge_size").unwrap_or(2),
            "out_hidden_size": u("clip.vision.projection_dim")?,
            "num_position_embeddings": npos,
            "deepstack_visual_indexes": deepstack,
        }))?;
        let triple = |k: &str| -> Option<[f32; 3]> {
            let a = h.get(k)?.as_array()?;
            (a.len() == 3).then(|| [0, 1, 2].map(|i| a[i].as_f64().unwrap_or(0.0) as f32))
        };
        let (mut image, mut video) = if model.is_dir() {
            (ProcessorConfig::load(model, false), ProcessorConfig::load(model, true))
        } else {
            (ProcessorConfig::image_default(), ProcessorConfig::video_default())
        };
        for p in [&mut image, &mut video] {
            p.patch_size = vision.patch_size;
            p.merge_size = vision.spatial_merge_size;
            if let Some(m) = triple("clip.vision.image_mean") {
                p.mean = m;
            }
            if let Some(s) = triple("clip.vision.image_std") {
                p.std = s;
            }
        }
        let tok = if crate::gguf::is_gguf(model) {
            crate::gguf::header(model).ok().and_then(|m| m.tokenizer().ok())
        } else {
            crate::fsx::read(&model.join("tokenizer.json"))
                .ok()
                .and_then(|b| crate::tokenizer::Tokenizer::from_bytes(b).ok())
        };
        let id = |s: &str, d: u32| tok.as_ref().and_then(|t| t.token_to_id(s)).unwrap_or(d);
        Ok(Self {
            vision,
            image,
            video,
            image_token_id: id("<|image_pad|>", 248056),
            video_token_id: id("<|video_pad|>", 248057),
            markers: QWEN_MARKERS,
            mmproj: Some(path.to_path_buf()),
        })
    }

    /// The tensors of the tower when they live in an mmproj file (None: in the checkpoint).
    pub fn open_mmproj(&self) -> Result<Option<ModelFiles>> {
        self.mmproj.as_deref().map(|p| ModelFiles::open(p, false)).transpose()
    }

    /// None when `config.json` has no Qwen-style `vision_config`.
    pub fn load(dir: &Path) -> Result<Option<Self>> {
        let raw: serde_json::Value = serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?)?;
        if raw.get("model_type").and_then(|v| v.as_str()) == Some("minimax_m3_vl") {
            return Self::minimax_m3(&raw);
        }
        let Some(v) = raw.get("vision_config").filter(|v| v.get("spatial_merge_size").is_some()) else {
            return Ok(None);
        };
        let vision: QwenVisionConfig = serde_json::from_value(v.clone()).context("vision_config")?;
        let id = |k: &str, d: u32| raw.get(k).and_then(|v| v.as_u64()).map_or(d, |v| v as u32);
        let (mut image, mut video) = (ProcessorConfig::load(dir, false), ProcessorConfig::load(dir, true));
        if vision.is_qwen25() && !crate::fsx::exists(&dir.join("preprocessor_config.json")) {
            // Qwen2-VL / 2.5-VL processor defaults (the Qwen3.5 ones have 16-pixel patches).
            for p in [&mut image, &mut video] {
                p.patch_size = vision.patch_size;
                p.mean = [0.481_454_66, 0.457_827_5, 0.408_210_73];
                p.std = [0.268_629_54, 0.261_302_6, 0.275_777_1];
                p.min_pixels = 3136;
                p.max_pixels = 12_845_056;
            }
        }
        Ok(Some(Self {
            vision,
            image,
            video,
            image_token_id: id("image_token_id", 248056),
            video_token_id: id("video_token_id", 248057),
            markers: QWEN_MARKERS,
            mmproj: None,
        }))
    }

    /// MiniMax M3's tower from llama.cpp's `minimax_m3` mmproj file.
    fn minimax_m3_mmproj(h: &crate::gguf::Header, path: &Path, model: &Path) -> Result<Self> {
        let u = |k: &str| {
            h.get(k)
                .and_then(|v| v.as_u64())
                .with_context(|| format!("{}: no {k}", path.display()))
        };
        let out = u("clip.vision.projection_dim")?;
        let ph = h
            .tensors
            .iter()
            .find(|t| t.name == "mm.1.weight")
            .map(|t| t.ne[1] as u64)
            .unwrap_or(out);
        let raw = serde_json::json!({
            "model_type": "minimax_m3_vl",
            "text_config": {"hidden_size": out},
            "projector_hidden_size": ph,
            "vision_config": {
                "num_hidden_layers": u("clip.vision.block_count")?,
                "hidden_size": u("clip.vision.embedding_length")?,
                "intermediate_size": u("clip.vision.feed_forward_length")?,
                "num_attention_heads": u("clip.vision.attention.head_count")?,
                "patch_size": u("clip.vision.patch_size")?,
                "spatial_merge_size": u("clip.vision.spatial_merge_size").unwrap_or(2),
                "layer_norm_eps": h.get("clip.vision.attention.layer_norm_epsilon").and_then(|v| v.as_f64()),
            },
        });
        let mut c = Self::minimax_m3(&raw)?.context("minimax_m3 mmproj")?;
        let tok = crate::gguf::header(model).ok().and_then(|m| m.tokenizer().ok());
        if let Some(t) = &tok {
            c.image_token_id = t.token_to_id("]<]image[>[").unwrap_or(c.image_token_id);
            c.video_token_id = t.token_to_id("]<]video[>[").unwrap_or(c.video_token_id);
        }
        c.mmproj = Some(path.to_path_buf());
        Ok(c)
    }

    /// MiniMax M3's (`crate::minimax_vision`): a CLIP-style `vision_config`, the processor its
    /// release uses (Qwen2-VL's resize between 3136 and 451584 pixels, 14-pixel patches, CLIP
    /// mean / std), images as `]<]start of image[>[` + `]<]image[>[` per token + `]<]end of image[>[`.
    fn minimax_m3(raw: &serde_json::Value) -> Result<Option<Self>> {
        let Some(v) = raw.get("vision_config") else { return Ok(None) };
        let text_hidden = raw
            .get("text_config")
            .and_then(|t| t.get("hidden_size"))
            .and_then(|x| x.as_u64())
            .context("minimax_m3_vl config has no text hidden_size")?;
        let u = |k: &str| {
            v.get(k)
                .and_then(|x| x.as_u64())
                .with_context(|| format!("vision_config has no {k}"))
        };
        let comp = v
            .get("img_token_compression_config")
            .or_else(|| raw.get("img_token_compression_config"));
        let merge = comp
            .and_then(|c| c.get("spatial_merge_size"))
            .or_else(|| v.get("spatial_merge_size"))
            .and_then(|x| x.as_u64())
            .unwrap_or(2);
        let temporal = comp
            .and_then(|c| c.get("temporal_patch_size"))
            .or_else(|| v.get("temporal_patch_size"))
            .and_then(|x| x.as_u64())
            .unwrap_or(2);
        let theta = v
            .get("rope_theta")
            .or_else(|| v.get("rope_parameters").and_then(|r| r.get("rope_theta")))
            .and_then(|x| x.as_f64())
            .unwrap_or(10000.0);
        let vision: QwenVisionConfig = serde_json::from_value(serde_json::json!({
            "depth": u("num_hidden_layers")?,
            "hidden_size": u("hidden_size")?,
            "hidden_act": "gelu",
            "intermediate_size": u("intermediate_size")?,
            "num_heads": u("num_attention_heads")?,
            "patch_size": u("patch_size")?,
            "spatial_merge_size": merge,
            "temporal_patch_size": temporal,
            "out_hidden_size": text_hidden,
            "rope_parameters": {"rope_theta": theta},
            "minimax_m3": true,
            "projector_hidden_size": raw.get("projector_hidden_size").and_then(|x| x.as_u64()),
            "layer_norm_eps": v.get("layer_norm_eps").and_then(|x| x.as_f64()),
        }))?;
        let mut image = ProcessorConfig::image_default();
        image.patch_size = vision.patch_size;
        image.temporal_patch_size = vision.temporal_patch_size;
        image.merge_size = vision.spatial_merge_size;
        image.mean = [0.481_454_66, 0.457_827_5, 0.408_210_73];
        image.std = [0.268_629_54, 0.261_302_6, 0.275_777_1];
        image.min_pixels = 4 * 28 * 28;
        image.max_pixels = 451_584;
        let id = |k: &str, d: u32| raw.get(k).and_then(|v| v.as_u64()).map_or(d, |v| v as u32);
        Ok(Some(Self {
            vision,
            video: image.clone(),
            image,
            image_token_id: id("image_token_index", 200025),
            video_token_id: id("video_token_index", 200026),
            markers: ["]<]start of image[>[", "]<]image[>[", "]<]end of image[>["],
            mmproj: None,
        }))
    }
}

/// Qwen's image placeholder: start, one per token, end.
pub const QWEN_MARKERS: [&str; 3] = ["<|vision_start|>", "<|image_pad|>", "<|vision_end|>"];

/// Whether a checkpoint is a Qwen3.5-family vision-language model (`config.json` has a Qwen
/// `vision_config`).
pub fn is_qwen_vl(dir: &Path) -> bool {
    matches!(QwenMmConfig::load(dir), Ok(Some(_)))
}

/// `smart_resize` of the Qwen2-VL image processor: both sides multiples of `factor`, area within
/// `[min_pixels, max_pixels]`, aspect ratio kept as closely as possible. Returns (height, width).
pub fn smart_resize(height: usize, width: usize, factor: usize, min_pixels: usize, max_pixels: usize) -> Result<(usize, usize)> {
    smart_resize_t(1, 1, height, width, factor, min_pixels, max_pixels)
}

/// The video processor's `smart_resize`: the budget covers all `frames` frames (rounded to a
/// multiple of `temporal_factor`). With `frames = temporal_factor = 1` this is the image rule
/// (except for the upscaling of sides below `factor`, which only the video processor does).
pub fn smart_resize_t(
    frames: usize,
    temporal_factor: usize,
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize)> {
    let (mut h, mut w) = (height as f64, width as f64);
    if h == 0.0 || w == 0.0 {
        bail!("empty image");
    }
    if temporal_factor > 1 {
        if frames < temporal_factor {
            bail!("{frames} frames: a video needs at least {temporal_factor}");
        }
        if height < factor || width < factor {
            let s = (factor as f64 / h).max(factor as f64 / w);
            h = (h * s).floor();
            w = (w * s).floor();
        }
    }
    if h.max(w) / h.min(w) > 200.0 {
        bail!("aspect ratio {:.0} is over 200", h.max(w) / h.min(w));
    }
    let f = factor as f64;
    // Python's round() is half-to-even.
    let mut hb = (h / f).round_ties_even() * f;
    let mut wb = (w / f).round_ties_even() * f;
    let t = frames as f64;
    let tb = (t / temporal_factor as f64).round_ties_even() * temporal_factor as f64;
    if tb * hb * wb > max_pixels as f64 {
        let beta = ((t * h * w) / max_pixels as f64).sqrt();
        hb = f.max((h / beta / f).floor() * f);
        wb = f.max((w / beta / f).floor() * f);
    } else if tb * hb * wb < min_pixels as f64 {
        let beta = (min_pixels as f64 / (t * h * w)).sqrt();
        hb = (h * beta / f).ceil() * f;
        wb = (w * beta / f).ceil() * f;
    }
    Ok((hb as usize, wb as usize))
}

/// Preprocessed pixels: `[t * h * w, C * T * p * p]` patches (spatial-merge block order within
/// each temporal group) and the patch grid `(t, h, w)`.
pub struct Pixels {
    pub data: Vec<f32>,
    pub grid: (usize, usize, usize),
}

impl Pixels {
    pub fn patches(&self) -> usize {
        self.grid.0 * self.grid.1 * self.grid.2
    }
}

/// Patchifies frames already resized to `w`x`h` (RGB8, `T` consecutive frames per temporal
/// patch; a short last group repeats the last frame).
fn patchify(frames: &[&[u8]], w: usize, h: usize, pc: &ProcessorConfig) -> Pixels {
    let (p, tp, m) = (pc.patch_size, pc.temporal_patch_size, pc.merge_size);
    let (gh, gw) = (h / p, w / p);
    let gt = frames.len().div_ceil(tp);
    let d = 3 * tp * p * p;
    let per_t = gh * gw;
    let mut data = vec![0f32; gt * per_t * d];
    // torchvision's fused rescale + normalize: (x - 255 mean) / (255 std).
    let mean: [f32; 3] = pc.mean.map(|v| v * 255.0);
    let std: [f32; 3] = pc.std.map(|v| v * 255.0);
    for_chunks(&mut data, d, true, |i, dst| {
        let (ti, k) = (i / per_t, i % per_t);
        // Block order: (block row, block col, row in block, col in block).
        let bw = gw / m;
        let (blk, inb) = (k / (m * m), k % (m * m));
        let (py, px) = ((blk / bw) * m + inb / m, (blk % bw) * m + inb % m);
        for c in 0..3 {
            for tt in 0..tp {
                let f = frames[(ti * tp + tt).min(frames.len() - 1)];
                for y in 0..p {
                    let row = &f[((py * p + y) * w + px * p) * 3..];
                    let o = ((c * tp + tt) * p + y) * p;
                    for x in 0..p {
                        dst[o + x] = (row[x * 3 + c] as f32 - mean[c]) / std[c];
                    }
                }
            }
        }
    });
    Pixels { data, grid: (gt, gh, gw) }
}

/// The image processor: smart resize within `max_pixels` (0: the checkpoint's budget) with
/// torchvision's uint8 bicubic (the processor's default backend; `resample` also has Pillow's),
/// normalization, patches (each frame twice along the temporal patch axis).
pub fn preprocess_image(rgb: &[u8], w: usize, h: usize, pc: &ProcessorConfig, max_pixels: usize) -> Result<Pixels> {
    let factor = pc.patch_size * pc.merge_size;
    let maxp = if max_pixels > 0 { max_pixels } else { pc.max_pixels };
    let (th, tw) = smart_resize(h, w, factor, pc.min_pixels.min(maxp), maxp)?;
    let img = crate::resample::resize_bicubic_rgb8_torchvision(rgb, w, h, tw, th);
    Ok(patchify(&[img.as_slice()], tw, th, pc))
}

/// The video processor on sampled frames (RGB8, all the same size `w`x`h`): one smart resize
/// for the whole clip, frames padded to a multiple of the temporal patch by repeating the last.
pub fn preprocess_video(frames: &[&[u8]], w: usize, h: usize, pc: &ProcessorConfig) -> Result<Pixels> {
    let factor = pc.patch_size * pc.merge_size;
    let (th, tw) = smart_resize_t(frames.len(), pc.temporal_patch_size, h, w, factor, pc.min_pixels, pc.max_pixels)?;
    let resized: Vec<Vec<u8>> = frames
        .iter()
        .map(|f| crate::resample::resize_bicubic_rgb8_torchvision(f, w, h, tw, th))
        .collect();
    let refs: Vec<&[u8]> = resized.iter().map(|v| v.as_slice()).collect();
    Ok(patchify(&refs, tw, th, pc))
}

/// Timestamps (seconds) of the temporal patches of a clip sampled at `indices` from a `fps`
/// video: the mean of each patch's first and last frame (`Qwen3VLProcessor._calculate_timestamps`).
pub fn temporal_patch_times(indices: &[usize], fps: f64, temporal_patch_size: usize) -> Vec<f64> {
    let times: Vec<f64> = indices.iter().map(|&i| i as f64 / fps).collect();
    temporal_patch_times_of(&times, temporal_patch_size)
}

/// [`temporal_patch_times`] from the sampled frames' timestamps in seconds: the mean of the first
/// and last frame of each temporal patch (the last frame repeated to fill the final patch).
pub fn temporal_patch_times_of(times: &[f64], temporal_patch_size: usize) -> Vec<f64> {
    let mut t = times.to_vec();
    while !t.is_empty() && !t.len().is_multiple_of(temporal_patch_size) {
        t.push(*t.last().unwrap());
    }
    t.chunks(temporal_patch_size).map(|c| (c[0] + c[c.len() - 1]) / 2.0).collect()
}

/// A biased linear layer of the tower.
pub(crate) struct VLin<'a> {
    w: Lin<'a>,
    b: Vec<f32>,
    n_out: usize,
    n_in: usize,
}

impl<'a> VLin<'a> {
    /// Loads `{name}.weight` (`[o, i]`) and `{name}.bias`. When `(po, pi)` is larger, the layer
    /// is padded with zero rows / columns to it (the 27-layer towers' MLP width, 4304, is not a
    /// multiple of the GPU tiles): a bf16 copy in memory.
    pub(crate) fn load(files: &'a ModelFiles, name: &str, (o, i): (usize, usize), (po, pi): (usize, usize)) -> Result<Self> {
        let wname = format!("{name}.weight");
        let mut b = files.f32(&format!("{name}.bias"))?;
        if b.len() != o {
            bail!("{name}.bias has {} values, expected {o}", b.len());
        }
        b.resize(po, 0.0);
        let w = if (po, pi) == (o, i) {
            load_lin(files, &wname, o, i, Quant::Bf16)?
        } else {
            let info = files.info(&wname).with_context(|| format!("missing tensor {wname}"))?;
            if info.shape != [o, i] {
                bail!("tensor {wname} has shape {:?}, expected [{o}, {i}]", info.shape);
            }
            let src = files.bf16(&wname)?;
            let mut v = crate::quant::PageVec::<u16>::zeroed(po * pi);
            for r in 0..o {
                v[r * pi..r * pi + i].copy_from_slice(&src[r * i..(r + 1) * i]);
            }
            Lin::bf16_owned(v)
        };
        Ok(Self { w, b, n_out: po, n_in: pi })
    }

    /// An all-zero layer (tests).
    #[cfg(test)]
    pub(crate) fn zero(n_out: usize, n_in: usize) -> Self {
        Self {
            w: Lin::bf16_owned(crate::quant::PageVec::zeroed(n_out * n_in)),
            b: vec![0.0; n_out],
            n_out,
            n_in,
        }
    }

    pub(crate) fn n_out(&self) -> usize {
        self.n_out
    }

    /// [`Self::load`] with the rows reordered: row `perm[j]` of the layer is row `j` of the
    /// stored weight (and bias), a bf16 copy (llama.cpp mmproj files permute some rows).
    pub(crate) fn load_permuted(files: &'a ModelFiles, name: &str, (o, i): (usize, usize), perm: &[usize]) -> Result<Self> {
        let wname = format!("{name}.weight");
        let src = files.bf16(&wname)?;
        let b0 = files.f32(&format!("{name}.bias"))?;
        if src.len() != o * i || b0.len() != o || perm.len() != o {
            bail!("{wname}: unexpected size for a [{o}, {i}] layer");
        }
        let mut w = crate::quant::PageVec::<u16>::zeroed(o * i);
        let mut b = vec![0f32; o];
        for (j, &r) in perm.iter().enumerate() {
            w[r * i..(r + 1) * i].copy_from_slice(&src[j * i..(j + 1) * i]);
            b[r] = b0[j];
        }
        Ok(Self {
            w: Lin::bf16_owned(w),
            b,
            n_out: o,
            n_in: i,
        })
    }

    pub(crate) fn forward(&self, x: &[f32], t: usize, y: &mut [f32]) {
        linear(&self.w, self.n_out, self.n_in, x, t, y);
        for_chunks(y, self.n_out, true, |_, r| add_inplace(r, &self.b));
    }
}

struct VBlock<'a> {
    norm1: LayerNorm,
    norm2: LayerNorm,
    qkv: VLin<'a>,
    proj: VLin<'a>,
    fc1: VLin<'a>,
    fc2: VLin<'a>,
}

/// The vision tower of a Qwen vision-language checkpoint: Qwen3.5 / 3.6 / 3.8, or Qwen2-VL /
/// Qwen2.5-VL (`crate::qwen25_vision`).
pub enum QwenVisionTower<'a> {
    Qwen35(Qwen35Tower<'a>),
    Qwen25(crate::qwen25_vision::Qwen25Tower<'a>),
    MiniMaxM3(Box<crate::minimax_vision::MiniMaxM3Tower<'a>>),
}

impl<'a> QwenVisionTower<'a> {
    /// Loads the tower `cfg` describes (bf16; see [`Qwen35Tower::load`]).
    pub fn load(cfg: QwenVisionConfig, files: &'a ModelFiles) -> Result<Self> {
        // Callers of `encode` alone would drop the DeepStack features without a word.
        if !cfg.deepstack_visual_indexes.is_empty() {
            bail!("DeepStack vision features (deepstack_visual_indexes, Qwen3-VL) are not supported here");
        }
        Self::load_deep(cfg, files)
    }

    /// [`QwenVisionTower::load`] that accepts a Qwen3-VL tower with DeepStack features, for
    /// callers that take them from [`QwenVisionTower::encode_deep`].
    pub fn load_deep(cfg: QwenVisionConfig, files: &'a ModelFiles) -> Result<Self> {
        Ok(if cfg.minimax_m3 {
            Self::MiniMaxM3(Box::new(crate::minimax_vision::MiniMaxM3Tower::load(cfg, files)?))
        } else if cfg.is_qwen25() {
            Self::Qwen25(crate::qwen25_vision::Qwen25Tower::load(cfg, files)?)
        } else {
            Self::Qwen35(Qwen35Tower::load(cfg, files)?)
        })
    }

    pub fn cfg(&self) -> &QwenVisionConfig {
        match self {
            Self::Qwen35(t) => &t.cfg,
            Self::Qwen25(t) => &t.cfg,
            Self::MiniMaxM3(t) => &t.cfg,
        }
    }

    /// Encodes preprocessed pixels into `[tokens, out_hidden_size]` language-model embeddings
    /// (tokens = patches / merge^2, in temporal-group then merged-row-major order).
    pub fn encode(&self, px: &Pixels) -> Vec<f32> {
        match self {
            Self::Qwen35(t) => t.encode(px),
            Self::Qwen25(t) => t.encode(px),
            Self::MiniMaxM3(t) => t.encode(px),
        }
    }

    /// [`QwenVisionTower::encode`], and the DeepStack features of a Qwen3-VL tower (see
    /// [`Qwen35Tower::encode_deep`]; none for the other towers).
    pub fn encode_deep(&self, px: &Pixels) -> (Vec<f32>, Vec<Vec<f32>>) {
        match self {
            Self::Qwen35(t) => t.encode_deep(px),
            _ => (self.encode(px), Vec::new()),
        }
    }
}

/// The Qwen3.5-family tower (see the module docs).
pub struct Qwen35Tower<'a> {
    pub cfg: QwenVisionConfig,
    patch_w: Lin<'a>, // [H, C*T*p*p]
    patch_b: Vec<f32>,
    pos_embed: Vec<f32>, // [num_position_embeddings, H]
    blocks: Vec<VBlock<'a>>,
    merger_norm: LayerNorm,
    merger_fc1: VLin<'a>,
    merger_fc2: VLin<'a>,
    /// DeepStack mergers (Qwen3-VL), one per entry of `deepstack_visual_indexes`: a LayerNorm
    /// over each merge block's concatenated patches, then the merger's two layers.
    deepstack: Vec<(LayerNorm, VLin<'a>, VLin<'a>)>,
    /// MLP width padded to a multiple of 64.
    inter: usize,
    inv_freq: Vec<f32>, // head_dim/4 frequencies per axis
}

impl<'a> Qwen35Tower<'a> {
    /// Loads `model.visual.*` (or `visual.*`) in bf16 whatever the text model's precision:
    /// int8 blocks moved the 27-layer tower's outputs by 4% (relative), against 1e-5 in bf16,
    /// and the tower is small next to the language model. Tensors stored pre-quantized load as
    /// stored.
    pub fn load(cfg: QwenVisionConfig, files: &'a ModelFiles) -> Result<Self> {
        if let Some(a) = &cfg.hidden_act
            && a != "gelu_pytorch_tanh"
        {
            bail!("vision hidden_act {a} is not supported (gelu_pytorch_tanh)");
        }
        // llama.cpp mmproj files (`v.*`, `mm.*`) or the checkpoint's `model.visual.*`.
        let gguf = files.has("v.patch_embd.weight");
        let pre = if gguf {
            ""
        } else if files.has("model.visual.patch_embed.proj.weight") {
            "model.visual."
        } else if files.has("visual.patch_embed.proj.weight") {
            "visual."
        } else {
            bail!("the checkpoint has no vision tower (model.visual.*)");
        };
        if gguf && !cfg.deepstack_visual_indexes.is_empty() {
            bail!("DeepStack vision features (deepstack_visual_indexes) from an mmproj file are not supported");
        }
        let name = |hf: &str| if gguf { mmproj_name(hf) } else { format!("{pre}{hf}") };
        let h = cfg.hidden_size;
        let (c, t, p) = (cfg.in_channels, cfg.temporal_patch_size, cfg.patch_size);
        let dense = |n: &str, o: usize, i: usize| VLin::load(files, &name(n), (o, i), (o, i));
        let (inter, inter_p) = (cfg.intermediate_size, cfg.intermediate_size.next_multiple_of(64));
        let patch_w = if gguf {
            // The Conv3d kernel split by llama.cpp into one Conv2d per temporal slice
            // (`v.patch_embd.weight`, `.weight.1`): interleaved back into [H, C, T, p, p].
            let slices = (0..t)
                .map(|k| {
                    let n = if k == 0 {
                        "v.patch_embd.weight".to_string()
                    } else {
                        format!("v.patch_embd.weight.{k}")
                    };
                    let w = files.bf16(&n)?;
                    if w.len() != h * c * p * p {
                        bail!("{n}: {} values, expected {}", w.len(), h * c * p * p);
                    }
                    Ok(w)
                })
                .collect::<Result<Vec<_>>>()?;
            let pp = p * p;
            let mut v = crate::quant::PageVec::<u16>::zeroed(h * c * t * pp);
            for oc in 0..h * c {
                for (k, w) in slices.iter().enumerate() {
                    v[(oc * t + k) * pp..(oc * t + k + 1) * pp].copy_from_slice(&w[oc * pp..(oc + 1) * pp]);
                }
            }
            Lin::bf16_owned(v)
        } else {
            Lin::bf16(files.bf16_conv(&name("patch_embed.proj.weight"), &[h, c, t, p, p])?)
        };
        let patch_b = files.f32(&name("patch_embed.proj.bias"))?;
        let pos_embed = files.f32(&name("pos_embed.weight"))?;
        if pos_embed.len() != cfg.num_position_embeddings * h || patch_b.len() != h {
            bail!("vision tower: unexpected patch / position embedding sizes");
        }
        let mut blocks = Vec::with_capacity(cfg.depth);
        for l in 0..cfg.depth {
            let b = format!("blocks.{l}.");
            blocks.push(VBlock {
                norm1: LayerNorm::load(files, &name(&format!("{b}norm1")), h)?,
                norm2: LayerNorm::load(files, &name(&format!("{b}norm2")), h)?,
                qkv: dense(&format!("{b}attn.qkv"), 3 * h, h)?,
                proj: dense(&format!("{b}attn.proj"), h, h)?,
                fc1: VLin::load(files, &name(&format!("{b}mlp.linear_fc1")), (inter, h), (inter_p, h))?,
                fc2: VLin::load(files, &name(&format!("{b}mlp.linear_fc2")), (h, inter), (h, inter_p))?,
            });
        }
        let mh = h * cfg.spatial_merge_size * cfg.spatial_merge_size;
        let spatial = cfg.head_dim() / 2;
        let theta = cfg.rope_theta();
        let inv_freq = (0..spatial / 2)
            .map(|i| (1.0 / theta.powf((2 * i) as f64 / spatial as f64)) as f32)
            .collect();
        let deepstack = (0..cfg.deepstack_visual_indexes.len())
            .map(|i| {
                let p = format!("deepstack_merger_list.{i}.");
                Ok((
                    LayerNorm::load(files, &name(&format!("{p}norm")), mh)?,
                    dense(&format!("{p}linear_fc1"), mh, mh)?,
                    dense(&format!("{p}linear_fc2"), cfg.out_hidden_size, mh)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            deepstack,
            patch_w,
            patch_b,
            pos_embed,
            blocks,
            merger_norm: LayerNorm::load(files, &name("merger.norm"), h)?,
            merger_fc1: dense("merger.linear_fc1", mh, mh)?,
            merger_fc2: dense("merger.linear_fc2", cfg.out_hidden_size, mh)?,
            inter: inter_p,
            inv_freq,
            cfg,
        })
    }

    /// (row, col) of the `k`-th patch of a frame in spatial-merge block order.
    fn patch_rc(&self, k: usize, gw: usize) -> (usize, usize) {
        let m = self.cfg.spatial_merge_size;
        let (blk, inb) = (k / (m * m), k % (m * m));
        ((blk / (gw / m)) * m + inb / m, (blk % (gw / m)) * m + inb % m)
    }

    /// The learned position embeddings resampled to a `gh`x`gw` grid (bilinear, align_corners),
    /// in patch order: `[gh * gw, H]`.
    fn pos_embeddings(&self, gh: usize, gw: usize) -> Vec<f32> {
        let h = self.cfg.hidden_size;
        let side = (self.cfg.num_position_embeddings as f64).sqrt() as usize;
        // Per-axis taps (floor, floor + 1 clamped) and weights, as transformers computes them.
        let taps = |i: usize, n: usize| -> [(usize, f32); 2] {
            let src = i as f32 * (side - 1) as f32 / (n.max(2) - 1) as f32;
            let src = if n == 1 { 0.0 } else { src };
            let f = src.floor();
            let t0 = (f as usize).min(side - 1);
            let t1 = (f as usize + 1).min(side - 1);
            [(t0, (1.0 - (src - f).abs()).max(0.0)), (t1, (1.0 - (src - f - 1.0).abs()).max(0.0))]
        };
        let mut out = vec![0f32; gh * gw * h];
        for_chunks(&mut out, h, true, |k, row| {
            let (r, c) = self.patch_rc(k, gw);
            for (ty, wy) in taps(r, gh) {
                for (tx, wx) in taps(c, gw) {
                    let wgt = wy * wx;
                    let e = &self.pos_embed[(ty * side + tx) * h..(ty * side + tx + 1) * h];
                    for (o, &v) in row.iter_mut().zip(e) {
                        *o += v * wgt;
                    }
                }
            }
        });
        out
    }

    /// Encodes preprocessed pixels into `[tokens, out_hidden_size]` language-model embeddings
    /// (tokens = patches / merge^2, in temporal-group then block order).
    pub fn encode(&self, px: &Pixels) -> Vec<f32> {
        self.encode_deep(px).0
    }

    /// [`Qwen35Tower::encode`], and the DeepStack features (Qwen3-VL; `[tokens,
    /// out_hidden_size]` per entry of `deepstack_visual_indexes`, which the language model adds
    /// to the image tokens after its first layers).
    pub fn encode_deep(&self, px: &Pixels) -> (Vec<f32>, Vec<Vec<f32>>) {
        let cfg = &self.cfg;
        let (gt, gh, gw) = px.grid;
        let per_t = gh * gw;
        let n = gt * per_t;
        let h = cfg.hidden_size;
        let (nh, hd) = (cfg.num_heads, cfg.head_dim());
        let d = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
        assert_eq!(px.data.len(), n * d, "pixel buffer does not match the grid");
        let eps = 1e-6;

        // Patch embedding + resampled position embeddings (the same for every temporal group).
        let mut x = vec![0f32; n * h];
        linear(&self.patch_w, h, d, &px.data, n, &mut x);
        let pos = self.pos_embeddings(gh, gw);
        for_chunks(&mut x, h, true, |i, row| {
            let pe = &pos[(i % per_t) * h..(i % per_t + 1) * h];
            for ((o, &b), &e) in row.iter_mut().zip(&self.patch_b).zip(pe) {
                *o += b + e;
            }
        });

        // Axial RoPE angles per patch: the first quarter of each half rotates with the row, the
        // second with the column.
        let q4 = self.inv_freq.len();
        let half = 2 * q4;
        let mut cos = vec![0f32; per_t * half];
        let mut sin = vec![0f32; per_t * half];
        for k in 0..per_t {
            let (r, c) = self.patch_rc(k, gw);
            for (j, &f) in self.inv_freq.iter().enumerate() {
                let (a, b) = (r as f32 * f, c as f32 * f);
                cos[k * half + j] = a.cos();
                sin[k * half + j] = a.sin();
                cos[k * half + q4 + j] = b.cos();
                sin[k * half + q4 + j] = b.sin();
            }
        }
        debug_assert_eq!(2 * half, hd);

        let mut hn = vec![0f32; n * h];
        let mut qkv = vec![0f32; n * 3 * h];
        let mut att = vec![0f32; n * h];
        let mut y = vec![0f32; n * h];
        let mut mid = vec![0f32; n * self.inter];
        let mut deep = Vec::new();
        for (li, b) in self.blocks.iter().enumerate() {
            hn.copy_from_slice(&x);
            b.norm1.apply_rows(&mut hn, eps);
            b.qkv.forward(&hn, n, &mut qkv);
            let (mut q, mut k, mut v) = (vec![0f32; n * h], vec![0f32; n * h], vec![0f32; n * h]);
            for_chunks(&mut q, h, true, |i, row| row.copy_from_slice(&qkv[i * 3 * h..i * 3 * h + h]));
            for_chunks(&mut k, h, true, |i, row| {
                row.copy_from_slice(&qkv[i * 3 * h + h..i * 3 * h + 2 * h])
            });
            for_chunks(&mut v, h, true, |i, row| {
                row.copy_from_slice(&qkv[i * 3 * h + 2 * h..(i + 1) * 3 * h])
            });
            for buf in [&mut q, &mut k] {
                for_chunks(buf, hd, true, |i, head| {
                    let pk = (i / nh) % per_t;
                    apply_rope(head, &cos[pk * half..(pk + 1) * half], &sin[pk * half..(pk + 1) * half]);
                });
            }
            // Bidirectional attention within each temporal group (frame pair).
            for g in 0..gt {
                let r = g * per_t * h..(g + 1) * per_t * h;
                let o = attention_chunked(&q[r.clone()], &k[r.clone()], &v[r.clone()], per_t, nh, hd);
                att[r].copy_from_slice(&o);
            }
            b.proj.forward(&att, n, &mut y);
            add_inplace(&mut x, &y);

            hn.copy_from_slice(&x);
            b.norm2.apply_rows(&mut hn, eps);
            b.fc1.forward(&hn, n, &mut mid);
            for_chunks(&mut mid, self.inter, true, |_, r| {
                for v in r.iter_mut() {
                    *v = gelu_tanh(*v);
                }
            });
            b.fc2.forward(&mid, n, &mut y);
            add_inplace(&mut x, &y);
            if let Some(i) = cfg.deepstack_visual_indexes.iter().position(|&d| d == li) {
                let (norm, fc1, fc2) = &self.deepstack[i];
                let mm = cfg.spatial_merge_size * cfg.spatial_merge_size;
                let (tokens, mh) = (n / mm, h * mm);
                let mut xn = x.clone();
                norm.apply_rows(&mut xn, eps);
                let mut m1 = vec![0f32; tokens * mh];
                fc1.forward(&xn, tokens, &mut m1);
                for_chunks(&mut m1, mh, true, |_, r| r.iter_mut().for_each(|v| *v = gelu_erf(*v)));
                let mut out = vec![0f32; tokens * cfg.out_hidden_size];
                fc2.forward(&m1, tokens, &mut out);
                deep.push(out);
            }
        }

        // Merger: LayerNorm per patch, each merge block's patches concatenated, MLP (exact GELU).
        self.merger_norm.apply_rows(&mut x, eps);
        let mm = cfg.spatial_merge_size * cfg.spatial_merge_size;
        let (tokens, mh) = (n / mm, h * mm);
        let mut m1 = vec![0f32; tokens * mh];
        self.merger_fc1.forward(&x, tokens, &mut m1);
        for_chunks(&mut m1, mh, true, |_, r| {
            for v in r.iter_mut() {
                *v = gelu_erf(*v);
            }
        });
        let mut out = vec![0f32; tokens * cfg.out_hidden_size];
        self.merger_fc2.forward(&m1, tokens, &mut out);
        (out, deep)
    }
}

/// The llama.cpp mmproj name of tower tensor `hf` (a `model.visual.` name without the prefix, or
/// the stem of a norm or linear layer).
fn mmproj_name(hf: &str) -> String {
    if let Some(rest) = hf.strip_prefix("blocks.") {
        let (l, tail) = rest.split_once('.').unwrap_or((rest, ""));
        let (stem, suffix) = match tail.find(".weight").or_else(|| tail.find(".bias")) {
            Some(i) => tail.split_at(i),
            None => (tail, ""),
        };
        let g = match stem {
            "norm1" => "ln1",
            "norm2" => "ln2",
            "attn.qkv" => "attn_qkv",
            "attn.proj" => "attn_out",
            "mlp.linear_fc1" => "ffn_up",
            "mlp.linear_fc2" => "ffn_down",
            s => s,
        };
        return format!("v.blk.{l}.{g}{suffix}");
    }
    match hf {
        "patch_embed.proj.bias" => "v.patch_embd.bias",
        "pos_embed.weight" => "v.position_embd.weight",
        "merger.norm" => "v.post_ln",
        "merger.linear_fc1" => "mm.0",
        "merger.linear_fc2" => "mm.2",
        s => s,
    }
    .to_string()
}

/// Bidirectional multi-head attention over `n` tokens (`[n, heads*hd]` each, scale hd^-0.5),
/// in query chunks that keep the score matrix under ~256 MB (GPU when one is active).
pub(crate) fn attention_chunked(q: &[f32], k: &[f32], v: &[f32], n: usize, nh: usize, hd: usize) -> Vec<f32> {
    let d = nh * hd;
    let chunk = ((1usize << 26) / n.max(1)).clamp(64, n.max(64));
    if chunk >= n {
        return crate::sd::ops::attention(q, k, v, n, n, nh, hd, false);
    }
    let mut out = vec![0f32; n * d];
    let mut s = 0;
    while s < n {
        let e = (s + chunk).min(n);
        let o = crate::sd::ops::attention(&q[s * d..e * d], k, v, e - s, n, nh, hd, false);
        out[s * d..e * d].copy_from_slice(&o);
        s = e;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_matches_reference() {
        // Values from transformers' smart_resize (factor 32, Qwen3.5-0.8B image budget).
        assert_eq!(smart_resize(1024, 1024, 32, 65536, 16777216).unwrap(), (1024, 1024));
        assert_eq!(smart_resize(480, 640, 32, 65536, 16777216).unwrap(), (480, 640));
        assert_eq!(smart_resize(100, 150, 32, 65536, 16777216).unwrap(), (224, 320));
        assert_eq!(smart_resize(3000, 4000, 32, 65536, 1_048_576).unwrap(), (864, 1152));
        // Video: 16 frames of 240x320 within 25165824 pixels.
        assert_eq!(smart_resize_t(16, 2, 240, 320, 32, 4096, 25165824).unwrap(), (256, 320));
    }

    #[test]
    fn mmproj_names() {
        assert_eq!(mmproj_name("blocks.3.norm1"), "v.blk.3.ln1");
        assert_eq!(mmproj_name("blocks.26.attn.qkv"), "v.blk.26.attn_qkv");
        assert_eq!(mmproj_name("blocks.0.mlp.linear_fc2"), "v.blk.0.ffn_down");
        assert_eq!(mmproj_name("blocks.0.attn.proj.weight"), "v.blk.0.attn_out.weight");
        assert_eq!(mmproj_name("merger.linear_fc2"), "mm.2");
        assert_eq!(mmproj_name("merger.norm"), "v.post_ln");
        assert_eq!(mmproj_name("pos_embed.weight"), "v.position_embd.weight");
    }

    #[test]
    fn timestamps_average_pairs() {
        let t = temporal_patch_times(&[0, 15, 30, 45, 60], 30.0, 2);
        assert_eq!(t, vec![0.25, 1.25, 2.0]);
        // The prompt prints them like Python's `f"{t:.1f}"` (exact value, ties to even).
        let s: Vec<String> = [0.25, 1.25, 0.75, 2.5].iter().map(|t| format!("{t:.1}")).collect();
        assert_eq!(s, ["0.2", "1.2", "0.8", "2.5"]);
    }
}
