//! Qwen3.5 / 3.6 / 3.8 vision (`Qwen3_5ForConditionalGeneration`'s `model.visual`): the image and
//! video processors (`Qwen2VLImageProcessor` / `Qwen3VLVideoProcessor`: smart resize to multiples
//! of 32 pixels within a pixel budget, uint8 bicubic, normalization to [-1, 1], 16x16 patches
//! with 2 temporal frames each, emitted in 2x2 spatial-merge blocks), and the ViT: a Conv3d patch
//! embedding, learned position embeddings bilinearly resampled to the patch grid, axial 2D RoPE,
//! pre-LayerNorm blocks with biased projections and bidirectional attention within each frame,
//! and the merger that folds every 2x2 block into one language-model token.
//!
//! The language model places the tokens with M-RoPE (see `QwenSession::add_vision_span`).
//! DeepStack (feature injection into early decoder layers) is not used by these checkpoints
//! (`deepstack_visual_indexes` is empty) and is rejected if present.

use crate::bert::{LayerNorm, gelu_erf};
use crate::kernels::*;
use crate::model::{Lin, Quant, linear, load_lin};
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::path::Path;

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
}

impl QwenVisionConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }
    fn rope_theta(&self) -> f64 {
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
}

impl QwenMmConfig {
    /// None when `config.json` has no Qwen-style `vision_config`.
    pub fn load(dir: &Path) -> Result<Option<Self>> {
        let raw: serde_json::Value = serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?)?;
        let Some(v) = raw.get("vision_config").filter(|v| v.get("spatial_merge_size").is_some()) else {
            return Ok(None);
        };
        let vision: QwenVisionConfig = serde_json::from_value(v.clone()).context("vision_config")?;
        let id = |k: &str, d: u32| raw.get(k).and_then(|v| v.as_u64()).map_or(d, |v| v as u32);
        Ok(Some(Self {
            vision,
            image: ProcessorConfig::load(dir, false),
            video: ProcessorConfig::load(dir, true),
            image_token_id: id("image_token_id", 248056),
            video_token_id: id("video_token_id", 248057),
        }))
    }
}

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
struct VLin<'a> {
    w: Lin<'a>,
    b: Vec<f32>,
    n_out: usize,
    n_in: usize,
}

impl<'a> VLin<'a> {
    /// Loads `{name}.weight` (`[o, i]`) and `{name}.bias`. When `(po, pi)` is larger, the layer
    /// is padded with zero rows / columns to it (the 27-layer towers' MLP width, 4304, is not a
    /// multiple of the GPU tiles): a bf16 copy in memory.
    fn load(files: &'a ModelFiles, name: &str, (o, i): (usize, usize), (po, pi): (usize, usize)) -> Result<Self> {
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

    fn forward(&self, x: &[f32], t: usize, y: &mut [f32]) {
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

pub struct QwenVisionTower<'a> {
    pub cfg: QwenVisionConfig,
    patch_w: Lin<'a>, // [H, C*T*p*p]
    patch_b: Vec<f32>,
    pos_embed: Vec<f32>, // [num_position_embeddings, H]
    blocks: Vec<VBlock<'a>>,
    merger_norm: LayerNorm,
    merger_fc1: VLin<'a>,
    merger_fc2: VLin<'a>,
    /// MLP width padded to a multiple of 64.
    inter: usize,
    inv_freq: Vec<f32>, // head_dim/4 frequencies per axis
}

impl<'a> QwenVisionTower<'a> {
    /// Loads `model.visual.*` (or `visual.*`) in bf16 whatever the text model's precision:
    /// int8 blocks moved the 27-layer tower's outputs by 4% (relative), against 1e-5 in bf16,
    /// and the tower is small next to the language model. Tensors stored pre-quantized load as
    /// stored.
    pub fn load(cfg: QwenVisionConfig, files: &'a ModelFiles) -> Result<Self> {
        if !cfg.deepstack_visual_indexes.is_empty() {
            bail!("DeepStack vision features (deepstack_visual_indexes) are not supported");
        }
        if let Some(a) = &cfg.hidden_act
            && a != "gelu_pytorch_tanh"
        {
            bail!("vision hidden_act {a} is not supported (gelu_pytorch_tanh)");
        }
        let pre = if files.has("model.visual.patch_embed.proj.weight") {
            "model.visual."
        } else if files.has("visual.patch_embed.proj.weight") {
            "visual."
        } else {
            bail!("the checkpoint has no vision tower (model.visual.*)");
        };
        let h = cfg.hidden_size;
        let (c, t, p) = (cfg.in_channels, cfg.temporal_patch_size, cfg.patch_size);
        let dense = |name: &str, o: usize, i: usize| VLin::load(files, &format!("{pre}{name}"), (o, i), (o, i));
        let (inter, inter_p) = (cfg.intermediate_size, cfg.intermediate_size.next_multiple_of(64));
        let pname = format!("{pre}patch_embed.proj.weight");
        let patch_w = Lin::bf16(files.bf16_conv(&pname, &[h, c, t, p, p])?);
        let patch_b = files.f32(&format!("{pre}patch_embed.proj.bias"))?;
        let pos_embed = files.f32(&format!("{pre}pos_embed.weight"))?;
        if pos_embed.len() != cfg.num_position_embeddings * h || patch_b.len() != h {
            bail!("vision tower: unexpected patch / position embedding sizes");
        }
        let mut blocks = Vec::with_capacity(cfg.depth);
        for l in 0..cfg.depth {
            let b = format!("blocks.{l}.");
            blocks.push(VBlock {
                norm1: LayerNorm::load(files, &format!("{pre}{b}norm1"), h)?,
                norm2: LayerNorm::load(files, &format!("{pre}{b}norm2"), h)?,
                qkv: dense(&format!("{b}attn.qkv"), 3 * h, h)?,
                proj: dense(&format!("{b}attn.proj"), h, h)?,
                fc1: VLin::load(files, &format!("{pre}{b}mlp.linear_fc1"), (inter, h), (inter_p, h))?,
                fc2: VLin::load(files, &format!("{pre}{b}mlp.linear_fc2"), (h, inter), (h, inter_p))?,
            });
        }
        let mh = h * cfg.spatial_merge_size * cfg.spatial_merge_size;
        let spatial = cfg.head_dim() / 2;
        let theta = cfg.rope_theta();
        let inv_freq = (0..spatial / 2)
            .map(|i| (1.0 / theta.powf((2 * i) as f64 / spatial as f64)) as f32)
            .collect();
        Ok(Self {
            patch_w,
            patch_b,
            pos_embed,
            blocks,
            merger_norm: LayerNorm::load(files, &format!("{pre}merger.norm"), h)?,
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
        for b in &self.blocks {
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
        out
    }
}

/// Bidirectional multi-head attention over `n` tokens (`[n, heads*hd]` each, scale hd^-0.5),
/// in query chunks that keep the score matrix under ~256 MB (GPU when one is active).
fn attention_chunked(q: &[f32], k: &[f32], v: &[f32], n: usize, nh: usize, hd: usize) -> Vec<f32> {
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
    fn timestamps_average_pairs() {
        let t = temporal_patch_times(&[0, 15, 30, 45, 60], 30.0, 2);
        assert_eq!(t, vec![0.25, 1.25, 2.0]);
        // The prompt prints them like Python's `f"{t:.1f}"` (exact value, ties to even).
        let s: Vec<String> = [0.25, 1.25, 0.75, 2.5].iter().map(|t| format!("{t:.1}")).collect();
        assert_eq!(s, ["0.2", "1.2", "0.8", "2.5"]);
    }
}
