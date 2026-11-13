//! Baidu's Unlimited-OCR-2 (`unlimited-ocr`, `UnlimitedOCRForCausalLM`), as the checkpoint's own
//! code runs it (`modeling_unlimitedocr.py`, `modeling_deepseekv2.py`, `dash_attention.py`,
//! `_sparse_base_impl.py`, `deepencoder.py`; `infer()` is the path followed here):
//!
//! - preprocessing (Pillow): every page as one global view, the image resized to fit 1024x1024
//!   (bicubic) and centered on gray (127); a single page larger than 640 pixels also as 2 to 16
//!   tiles of 640x640 ("gundam" mode: the image squashed to the grid of tiles whose aspect ratio
//!   is closest); `x / 127.5 - 1`, rounded to bfloat16 as the reference does;
//! - the "deepencoder": a SAM ViT-B (windows of 14 with decomposed relative positions, global
//!   attention in four blocks, the neck and two stride-2 convolutions: 1/64 of the pixels) whose
//!   output is the patch embedding of a CLIP ViT-L (class token, 24 blocks, no final norm); the
//!   CLIP output and the SAM features side by side (2048) through one linear projector;
//! - packing: the tiles' features as one grid with `image_newline` after every row, then the
//!   global view's 16 rows with a newline each, then `view_seperator`: `273` tokens for a page
//!   alone, `(10 w + 1) * 10 h` more with `w` x `h` tiles;
//! - the prompt: BOS, `<PAGE>` and the `<image>` placeholders per page, then the instruction
//!   (`Multi page parsing.`); no chat roles. The reply is `<PAGE>` then the page's text, per page;
//! - the decoder: 12 layers of width 1280, a dense MLP then DeepSeek-V2 MoE layers (softmax
//!   router, top 6 of 64 without renormalization, a shared expert), and "DASH" attention: every
//!   query attends to its last `sparse_local_block * sparse_block_size` (128) tokens and to at
//!   most `dash_topk_blocks` (12) earlier blocks of 32 keys, chosen by a router (two small
//!   projections of the layer input, Gemma-style RMS norms, rotated; a learned `head_cls` query
//!   pools each block's router keys into one summary key). The router's scores of the candidate
//!   blocks go through 1.5-entmax over the best 16 and the log of that prior, centered, is added
//!   to the attention scores of the block's keys. Candidates are the image blocks of the query's
//!   own page (all blocks when it has none);
//! - positions: a rotary embedding over two axes, the running index of the token inside its page
//!   (63 frequency pairs) and the parity of the page (one pair), pages counted by the `<PAGE>`
//!   markers of the prompt and of the reply. Without markers (text only, or images without
//!   `<PAGE>`) the reference falls back to plain YaRN positions; so does this.
//!
//! The session is incremental where the reference recomputes: blocks summaries are final once a
//! block is full, and a full block is the only kind a query can select (the last four blocks are
//! its local window). The first forward pass of a session is the prompt (it must come whole:
//! which tokens belong to a page depends on where the last image ends); every later token is a
//! decoding step.

use crate::bert::{LayerNorm, gelu_erf};
use crate::config::YarnParams;
use crate::kernels::*;
use crate::model::{Lin, Quant, Table, linear, linear_multi, load_lin, table_quant};
use crate::moe::{Act, Experts, Shared};
use crate::qwen_vision::VLin;
use crate::safetensors::ModelFiles;
use crate::sd::ops::{Conv, f32_to_bf16, matmul_f32, quick_gelu};
use anyhow::{Context, Result, bail};
use std::path::Path;

/// The instruction `infer()` uses when none is given.
pub const DEFAULT_PROMPT: &str = "Multi page parsing.";
/// The page marker of prompts and replies.
pub const PAGE: &str = "<PAGE>";
/// The image placeholder (one token per feature).
pub const IMAGE: &str = "<image>";
const BOS: &str = "<｜begin▁of▁sentence｜>";
const EOS: &str = "<｜end▁of▁sentence｜>";

/// Side of the global view and of a tile.
const BASE: usize = 1024;
const TILE: usize = 640;
/// Most tiles per page (`max_crops`).
const MAX_TILES: usize = 16;

pub fn is_model_type(mt: &str) -> bool {
    mt == "unlimited-ocr"
}

/// The settings of an Unlimited-OCR-2 checkpoint (`config.json`).
#[derive(Debug, Clone)]
pub struct OcrConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub vocab_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub first_k_dense_replace: usize,
    pub moe_layer_freq: usize,
    pub rms_norm_eps: f64,
    pub routed_scaling_factor: f64,
    pub norm_topk_prob: bool,
    pub eos_token_id: u32,
    pub image_token_id: u32,
    pub page_token_id: u32,
    pub per_page_image_tokens: usize,
    /// Keys per block (`sparse_block_size`).
    pub block_size: usize,
    /// Blocks of the local window (`sparse_local_block`).
    pub local_blocks: usize,
    /// Width of the router's projections (`sparse_index_dim`; the head size here).
    pub index_dim: usize,
    /// Blocks a query reads at most (`dash_topk_blocks`).
    pub topk_blocks: usize,
    /// Best blocks the entmax runs over (`infer(dash_topm=16)`).
    pub topm: usize,
    /// Per layer: the entmax input scale (`dash_scale_per_layer`) and the prior floor.
    pub scale: Vec<f32>,
    pub floor: Vec<f32>,
    pub sigma: f32,
    pub rope_tok_theta: f64,
    pub rope_page_theta: f64,
    pub rope_page_pairs: usize,
    pub rope_page_mod: usize,
    /// Token frequencies spread over their own pairs (`resample`) or cut from the full head's
    /// (`truncate`): the attention's, the router's.
    pub main_resample: bool,
    pub idx_resample: bool,
    /// Positions without page markers.
    pub yarn: YarnParams,
}

impl OcrConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn is_moe(&self, l: usize) -> bool {
        self.n_routed_experts > 0 && l >= self.first_k_dense_replace && l.is_multiple_of(self.moe_layer_freq.max(1))
    }

    pub fn load(dir: &Path) -> Result<Self> {
        let raw: serde_json::Value = serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?)?;
        Self::from_json(&raw)
    }

    pub fn from_json(raw: &serde_json::Value) -> Result<Self> {
        let us = |k: &str| raw[k].as_u64().map(|v| v as usize).with_context(|| format!("config.json: no {k}"));
        let sp = &raw["sparse_attention_config"];
        if !sp.is_object() {
            bail!("config.json: no sparse_attention_config");
        }
        let spu = |k: &str, d: usize| sp[k].as_u64().map(|v| v as usize).unwrap_or(d);
        let spf = |k: &str, d: f64| sp[k].as_f64().unwrap_or(d);
        let sps = |k: &str, d: &str| sp[k].as_str().unwrap_or(d).to_lowercase();
        let layers = us("num_hidden_layers")?;
        let (hidden, heads) = (us("hidden_size")?, us("num_attention_heads")?);
        // What the reference code would run differently and this does not implement.
        let unsupported = |what: &str| -> Result<()> { bail!("Unlimited-OCR: {what} is not supported") };
        if raw["use_mla"].as_bool().unwrap_or(false) {
            unsupported("use_mla")?;
        }
        if us("num_key_value_heads")? != heads {
            unsupported("grouped key / value heads")?;
        }
        if raw["scoring_func"].as_str().unwrap_or("softmax") != "softmax" || raw["topk_method"].as_str().unwrap_or("greedy") != "greedy" {
            unsupported("a router other than softmax / greedy")?;
        }
        if !sp["use_sparse_attention"].as_bool().unwrap_or(false) || sps("training_stage", "") != "dash" {
            unsupported("a checkpoint without DASH attention (training_stage)")?;
        }
        let freq: Vec<u64> = sp["sparse_attention_freq"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_u64()).collect())
            .unwrap_or_default();
        if freq.len() < layers || freq.iter().any(|&f| f != 1) {
            unsupported("layers without sparse attention (sparse_attention_freq)")?;
        }
        if sps("dash_stage0", "indexer") != "head_cls" {
            unsupported("dash_stage0 other than head_cls")?;
        }
        if sps("sparse_rope_mode", "1d") != "page2d" || sps("sparse_main_rope_mode", "1d") != "page2d" {
            unsupported("rope modes other than page2d")?;
        }
        if spu("sparse_rope_page_span", 0) != 0 {
            unsupported("sparse_rope_page_span")?;
        }
        if !sp["dash_image_only"].as_bool().unwrap_or(true) || !sp["image_bucket_constraint"].as_bool().unwrap_or(false) {
            unsupported("routing without the page constraint (dash_image_only, image_bucket_constraint)")?;
        }
        let head_dim = hidden / heads;
        let index_dim = spu("sparse_index_dim", head_dim);
        if index_dim != head_dim || spu("sparse_num_index_heads", 1) != 1 {
            unsupported("a router width other than the head size")?;
        }
        let page_pairs = spu("sparse_rope_page_pairs", 8);
        if page_pairs == 0 || page_pairs >= head_dim / 2 {
            unsupported("sparse_rope_page_pairs out of range")?;
        }
        let per_layer = |k: &str, scalar: f64| -> Vec<f32> {
            let a: Vec<f32> = sp[k]
                .as_array()
                .map(|a| a.iter().filter_map(|v| v.as_f64()).map(|v| v as f32).collect())
                .unwrap_or_default();
            (0..layers).map(|l| a.get(l).copied().unwrap_or(scalar as f32)).collect()
        };
        let theta = raw["rope_theta"].as_f64().unwrap_or(10000.0);
        // The 1D fallback: transformers' YaRN with the factor recomputed from the positions.
        let mut ys = raw["rope_scaling"].clone();
        let yarn: YarnParams = if ys.is_object() {
            ys["rope_theta"] = spf("sparse_rope_theta", theta).into();
            if ys["rope_type"].is_null() {
                ys["rope_type"] = ys["type"].clone();
            }
            let mut y: YarnParams = serde_json::from_value(ys)?;
            if let (Some(o), Some(m)) = (y.original_max_position_embeddings, raw["max_position_embeddings"].as_u64()) {
                y.factor = Some(m as f64 / o as f64);
            }
            y
        } else {
            serde_json::from_value(serde_json::json!({"rope_theta": theta}))?
        };
        Ok(Self {
            hidden_size: hidden,
            num_hidden_layers: layers,
            num_attention_heads: heads,
            vocab_size: us("vocab_size")?,
            intermediate_size: us("intermediate_size")?,
            moe_intermediate_size: us("moe_intermediate_size")?,
            n_routed_experts: raw["n_routed_experts"].as_u64().unwrap_or(0) as usize,
            n_shared_experts: raw["n_shared_experts"].as_u64().unwrap_or(0) as usize,
            num_experts_per_tok: raw["num_experts_per_tok"].as_u64().unwrap_or(0) as usize,
            first_k_dense_replace: raw["first_k_dense_replace"].as_u64().unwrap_or(0) as usize,
            moe_layer_freq: raw["moe_layer_freq"].as_u64().unwrap_or(1) as usize,
            rms_norm_eps: raw["rms_norm_eps"].as_f64().unwrap_or(1e-6),
            routed_scaling_factor: raw["routed_scaling_factor"].as_f64().unwrap_or(1.0),
            norm_topk_prob: raw["norm_topk_prob"].as_bool().unwrap_or(false),
            eos_token_id: raw["eos_token_id"].as_u64().unwrap_or(1) as u32,
            image_token_id: spu("image_token_id", 128815) as u32,
            page_token_id: sp["page_token_id"].as_u64().context("config.json: no page_token_id")? as u32,
            per_page_image_tokens: spu("per_page_image_tokens", 273).max(1),
            block_size: spu("sparse_block_size", 128),
            local_blocks: spu("sparse_local_block", 1).max(1),
            index_dim,
            topk_blocks: spu("dash_topk_blocks", 12),
            topm: 16,
            scale: per_layer("dash_scale_per_layer", spf("dash_scaling_factor", 2.0)),
            floor: per_layer("dash_floor_per_layer", spf("dash_prior_floor", 1e-2)),
            sigma: spf("dash_sigma", 1.0) as f32,
            rope_tok_theta: spf("sparse_rope_tok_theta", theta),
            rope_page_theta: spf("sparse_rope_page_theta", 256.0),
            rope_page_pairs: page_pairs,
            rope_page_mod: spu("sparse_rope_page_mod", 0),
            main_resample: sps("sparse_main_rope_tok_freq_mode", "truncate") != "truncate",
            idx_resample: sps("sparse_rope_tok_freq_mode", "resample") != "truncate",
            yarn,
        })
    }
}

// ---- preprocessing --------------------------------------------------------------------------

/// A page's views: the global one and its tiles (`[3, side, side]` each, in [-1, 1]).
pub struct PagePixels {
    pub global: Vec<f32>,
    pub tiles: Vec<Vec<f32>>,
    /// Tiles across and down (1 x 1: none).
    pub grid: (usize, usize),
}

impl PagePixels {
    /// The `<image>` tokens of this page.
    pub fn tokens(&self) -> usize {
        let g = BASE / 64;
        let n = g * (g + 1) + 1;
        if self.tiles.is_empty() {
            n
        } else {
            let t = TILE / 64;
            n + (t * self.grid.0 + 1) * (t * self.grid.1)
        }
    }
}

/// Python's `round`: halves go to the even neighbor.
fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

/// torchvision's `ToTensor` and `Normalize(0.5, 0.5)` on interleaved RGB8 (a `side` x `side`
/// window of an image `iw` wide at `x0`, `y0`), cast to bfloat16 as `_uocr_build_inputs` does.
fn normalize(img: &[u8], iw: usize, x0: usize, y0: usize, side: usize) -> Vec<f32> {
    let mut out = vec![0f32; 3 * side * side];
    for c in 0..3 {
        for y in 0..side {
            for x in 0..side {
                let v = (img[((y0 + y) * iw + x0 + x) * 3 + c] as f32 / 255.0 - 0.5) / 0.5;
                out[(c * side + y) * side + x] = bf16_to_f32(f32_to_bf16(v));
            }
        }
    }
    out
}

/// `dynamic_preprocess`'s grid of tiles for a `w` x `h` image: of the grids of 2 to 16 tiles, the
/// one whose aspect ratio is closest (on a tie, the later one in area order when the image has
/// more than half its pixels).
pub fn tile_grid(w: usize, h: usize) -> (usize, usize) {
    let aspect = w as f64 / h as f64;
    let mut ratios: Vec<(usize, usize)> = Vec::new();
    for i in 1..=MAX_TILES {
        for j in 1..=MAX_TILES {
            if (2..=MAX_TILES).contains(&(i * j)) {
                ratios.push((i, j));
            }
        }
    }
    // Python sorts a set here: grids of the same area come in its hash order, which only matters
    // for two of them equally close to the image, and those always have a third in between.
    ratios.sort_by_key(|r| r.0 * r.1);
    let (mut best, mut best_diff) = ((1, 1), f64::INFINITY);
    let area = (w * h) as f64;
    for r in ratios {
        let diff = (aspect - r.0 as f64 / r.1 as f64).abs();
        if diff < best_diff {
            (best, best_diff) = (r, diff);
        } else if diff == best_diff && area > 0.5 * (TILE * TILE * r.0 * r.1) as f64 {
            best = r;
        }
    }
    best
}

/// One page's views from interleaved RGB8. `crop`: tiles too ("gundam" mode, what `infer()` does
/// for a single page), when the image is larger than a tile.
pub fn preprocess(rgb: &[u8], w: usize, h: usize, crop: bool) -> Result<PagePixels> {
    if w == 0 || h == 0 || rgb.len() != w * h * 3 {
        bail!("empty image");
    }
    // ImageOps.pad: resized to fit (ImageOps.contain), centered on the mean color.
    let (mut rw, mut rh) = (BASE, BASE);
    if w > h {
        rh = round_half_even(h as f64 / w as f64 * BASE as f64) as usize;
    } else if h > w {
        rw = round_half_even(w as f64 / h as f64 * BASE as f64) as usize;
    }
    if rw == 0 || rh == 0 {
        bail!("image of {w}x{h}: too thin for a {BASE}x{BASE} view");
    }
    let resized = crate::resample::resize_bicubic_rgb8(rgb, w, h, rw, rh);
    let mut canvas = vec![127u8; BASE * BASE * 3];
    let (ox, oy) = if rw != BASE {
        (round_half_even((BASE - rw) as f64 * 0.5) as usize, 0)
    } else {
        (0, round_half_even((BASE - rh) as f64 * 0.5) as usize)
    };
    for y in 0..rh {
        canvas[((oy + y) * BASE + ox) * 3..((oy + y) * BASE + ox + rw) * 3].copy_from_slice(&resized[y * rw * 3..(y + 1) * rw * 3]);
    }
    let global = normalize(&canvas, BASE, 0, 0, BASE);
    let (mut tiles, mut grid) = (Vec::new(), (1, 1));
    if crop && (w > TILE || h > TILE) {
        grid = tile_grid(w, h);
        let (tw, th) = (TILE * grid.0, TILE * grid.1);
        let squashed = crate::resample::resize_bicubic_rgb8(rgb, w, h, tw, th);
        for ty in 0..grid.1 {
            for tx in 0..grid.0 {
                tiles.push(normalize(&squashed, tw, tx * TILE, ty * TILE, TILE));
            }
        }
    }
    Ok(PagePixels { global, tiles, grid })
}

// ---- the vision encoder ---------------------------------------------------------------------

const SAM_DIM: usize = 768;
const SAM_HEADS: usize = 12;
const SAM_HD: usize = SAM_DIM / SAM_HEADS;
const SAM_GRID: usize = 64;
const SAM_WINDOW: usize = 14;
const SAM_GLOBAL: [usize; 4] = [2, 5, 8, 11];
const CLIP_DIM: usize = 1024;
const CLIP_HEADS: usize = 16;
const CLIP_GRID: usize = 16;

/// Keys' bicubic kernel with a = -0.5 (what PyTorch's antialiased `interpolate` uses).
fn cubic(x: f32) -> f32 {
    const A: f32 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// `F.interpolate(mode="bicubic", antialias=True, align_corners=False)` of a square grid
/// `[s, s, c]` to `[t, t, c]`.
fn interp_bicubic_aa(src: &[f32], s: usize, t: usize, c: usize) -> Vec<f32> {
    let scale = s as f32 / t as f32;
    let (support, inv) = if scale >= 1.0 { (2.0 * scale, 1.0 / scale) } else { (2.0, 1.0) };
    // Per output index: the first input index and the normalized weights.
    let taps: Vec<(usize, Vec<f32>)> = (0..t)
        .map(|i| {
            let center = scale * (i as f32 + 0.5);
            let lo = ((center - support + 0.5) as i64).max(0) as usize;
            let hi = ((center + support + 0.5) as i64).min(s as i64) as usize;
            let mut w: Vec<f32> = (lo..hi).map(|j| cubic((j as f32 - center + 0.5) * inv)).collect();
            let total: f32 = w.iter().sum();
            if total != 0.0 {
                w.iter_mut().for_each(|v| *v /= total);
            }
            (lo, w)
        })
        .collect();
    // Columns, then rows.
    let mut mid = vec![0f32; s * t * c];
    for y in 0..s {
        for (x, (lo, w)) in taps.iter().enumerate() {
            let o = &mut mid[(y * t + x) * c..(y * t + x + 1) * c];
            for (k, &wk) in w.iter().enumerate() {
                axpy(o, wk, &src[(y * s + lo + k) * c..(y * s + lo + k + 1) * c]);
            }
        }
    }
    let mut out = vec![0f32; t * t * c];
    for (y, (lo, w)) in taps.iter().enumerate() {
        for (k, &wk) in w.iter().enumerate() {
            axpy(
                &mut out[y * t * c..(y + 1) * t * c],
                wk,
                &mid[(lo + k) * t * c..(lo + k + 1) * t * c],
            );
        }
    }
    out
}

/// `F.interpolate(mode="linear")` of a table `[n, c]` to `[m, c]` (SAM's relative positions for
/// another grid).
fn interp_linear(src: &[f32], n: usize, m: usize, c: usize) -> Vec<f32> {
    if n == m {
        return src.to_vec();
    }
    let scale = n as f32 / m as f32;
    let mut out = vec![0f32; m * c];
    for i in 0..m {
        let x = ((i as f32 + 0.5) * scale - 0.5).max(0.0);
        let i0 = (x as usize).min(n - 1);
        let i1 = (i0 + 1).min(n - 1);
        let f = x - i0 as f32;
        for k in 0..c {
            out[i * c + k] = (1.0 - f) * src[i0 * c + k] + f * src[i1 * c + k];
        }
    }
    out
}

struct SamBlock<'a> {
    norm1: LayerNorm,
    norm2: LayerNorm,
    qkv: VLin<'a>,
    proj: VLin<'a>,
    lin1: VLin<'a>,
    lin2: VLin<'a>,
    /// Relative position tables `[2 * side - 1, head]` (side: the window, or the 64 grid).
    rel_h: Vec<f32>,
    rel_w: Vec<f32>,
    /// 0: global attention.
    window: usize,
}

struct ClipBlock<'a> {
    norm1: LayerNorm,
    norm2: LayerNorm,
    qkv: VLin<'a>,
    out: VLin<'a>,
    fc1: VLin<'a>,
    fc2: VLin<'a>,
}

/// The "deepencoder" (SAM ViT-B into CLIP ViT-L), the projector and the two separators.
pub struct OcrTower<'a> {
    patch_w: Lin<'a>,
    patch_b: Vec<f32>,
    sam_pos: Vec<f32>,
    sam_blocks: Vec<SamBlock<'a>>,
    neck: [Conv; 2],
    neck_norm: [LayerNorm; 2],
    net: [Conv; 2],
    clip_class: Vec<f32>,
    clip_pos: Vec<f32>,
    clip_pre: LayerNorm,
    clip_blocks: Vec<ClipBlock<'a>>,
    proj: VLin<'a>,
    newline: Vec<f32>,
    separator: Vec<f32>,
    hidden: usize,
}

/// SAM's attention over a `gh` x `gw` grid: `qkv` is `[n, 3 * 768]` (q, k, v per token), the
/// scores get the decomposed relative position terms `q . Rh[qy - ky] + q . Rw[qx - kx]`.
fn sam_attention(qkv: &[f32], gh: usize, gw: usize, rel_h: &[f32], rel_w: &[f32], out: &mut [f32]) {
    let (n, hd) = (gh * gw, SAM_HD);
    let np = n.next_multiple_of(32);
    let scale = 1.0 / (hd as f32).sqrt();
    let (mut qh, mut kh, mut vt) = (vec![0f32; n * hd], vec![0f32; n * hd], vec![0f32; hd * np]);
    for h in 0..SAM_HEADS {
        for i in 0..n {
            let row = &qkv[i * 3 * SAM_DIM..(i + 1) * 3 * SAM_DIM];
            qh[i * hd..(i + 1) * hd].copy_from_slice(&row[h * hd..(h + 1) * hd]);
            kh[i * hd..(i + 1) * hd].copy_from_slice(&row[SAM_DIM + h * hd..SAM_DIM + (h + 1) * hd]);
            for c in 0..hd {
                vt[c * np + i] = row[2 * SAM_DIM + h * hd + c];
            }
        }
        let s = matmul_f32(&kh, n, hd, &qh, n);
        let mut p = vec![0f32; n * np];
        for_chunks(&mut p, np, n > 64, |i, r| {
            let q = &qh[i * hd..(i + 1) * hd];
            let (qy, qx) = (i / gw, i % gw);
            let rh: Vec<f32> = (0..gh)
                .map(|ky| dot(q, &rel_h[(qy + gh - 1 - ky) * hd..(qy + gh - ky) * hd]))
                .collect();
            let rw: Vec<f32> = (0..gw)
                .map(|kx| dot(q, &rel_w[(qx + gw - 1 - kx) * hd..(qx + gw - kx) * hd]))
                .collect();
            let row = &s[i * n..(i + 1) * n];
            let mut mx = f32::NEG_INFINITY;
            for j in 0..n {
                r[j] = row[j] * scale + rh[j / gw] + rw[j % gw];
                mx = mx.max(r[j]);
            }
            let mut sum = 0f32;
            for v in &mut r[..n] {
                *v = (*v - mx).exp();
                sum += *v;
            }
            let inv = 1.0 / sum;
            r[..n].iter_mut().for_each(|v| *v *= inv);
        });
        let o = matmul_f32(&vt, hd, np, &p, n);
        for i in 0..n {
            out[i * SAM_DIM + h * hd..i * SAM_DIM + (h + 1) * hd].copy_from_slice(&o[i * hd..(i + 1) * hd]);
        }
    }
}

impl<'a> OcrTower<'a> {
    /// `hidden`: the language model's width.
    pub fn load(files: &'a ModelFiles, hidden: usize) -> Result<Self> {
        let sam = "model.sam_model.";
        let dense = |n: &str, o: usize, i: usize| VLin::load(files, n, (o, i), (o, i));
        let getf = |n: &str, len: usize| -> Result<Vec<f32>> {
            let v = files.f32(n)?;
            if v.len() != len {
                bail!("tensor {n} has {} values, expected {len}", v.len());
            }
            Ok(v)
        };
        let mut sam_blocks = Vec::new();
        for l in 0..12 {
            let b = format!("{sam}blocks.{l}.");
            let window = if SAM_GLOBAL.contains(&l) { 0 } else { SAM_WINDOW };
            let side = if window == 0 { SAM_GRID } else { window };
            sam_blocks.push(SamBlock {
                norm1: LayerNorm::load(files, &format!("{b}norm1"), SAM_DIM)?,
                norm2: LayerNorm::load(files, &format!("{b}norm2"), SAM_DIM)?,
                qkv: dense(&format!("{b}attn.qkv"), 3 * SAM_DIM, SAM_DIM)?,
                proj: dense(&format!("{b}attn.proj"), SAM_DIM, SAM_DIM)?,
                lin1: dense(&format!("{b}mlp.lin1"), 4 * SAM_DIM, SAM_DIM)?,
                lin2: dense(&format!("{b}mlp.lin2"), SAM_DIM, 4 * SAM_DIM)?,
                rel_h: getf(&format!("{b}attn.rel_pos_h"), (2 * side - 1) * SAM_HD)?,
                rel_w: getf(&format!("{b}attn.rel_pos_w"), (2 * side - 1) * SAM_HD)?,
                window,
            });
        }
        let conv = |n: &str, cout: usize, cin: usize, k: usize| -> Result<Conv> {
            Conv::from_kernel(&getf(n, cout * cin * k * k)?, cout, cin, k, None)
        };
        let clip = "model.vision_model.";
        let mut clip_blocks = Vec::new();
        for l in 0..24 {
            let b = format!("{clip}transformer.layers.{l}.");
            clip_blocks.push(ClipBlock {
                norm1: LayerNorm::load(files, &format!("{b}layer_norm1"), CLIP_DIM)?,
                norm2: LayerNorm::load(files, &format!("{b}layer_norm2"), CLIP_DIM)?,
                qkv: dense(&format!("{b}self_attn.qkv_proj"), 3 * CLIP_DIM, CLIP_DIM)?,
                out: dense(&format!("{b}self_attn.out_proj"), CLIP_DIM, CLIP_DIM)?,
                fc1: dense(&format!("{b}mlp.fc1"), 4 * CLIP_DIM, CLIP_DIM)?,
                fc2: dense(&format!("{b}mlp.fc2"), CLIP_DIM, 4 * CLIP_DIM)?,
            });
        }
        Ok(Self {
            patch_w: Lin::bf16(files.bf16_conv(&format!("{sam}patch_embed.proj.weight"), &[SAM_DIM, 3, 16, 16])?),
            patch_b: getf(&format!("{sam}patch_embed.proj.bias"), SAM_DIM)?,
            sam_pos: getf(&format!("{sam}pos_embed"), SAM_GRID * SAM_GRID * SAM_DIM)?,
            sam_blocks,
            neck: [
                conv(&format!("{sam}neck.0.weight"), 256, SAM_DIM, 1)?.with_padding(0, 0),
                conv(&format!("{sam}neck.2.weight"), 256, 256, 3)?,
            ],
            neck_norm: [
                LayerNorm::load(files, &format!("{sam}neck.1"), 256)?,
                LayerNorm::load(files, &format!("{sam}neck.3"), 256)?,
            ],
            net: [
                conv(&format!("{sam}net_2.weight"), 512, 256, 3)?,
                conv(&format!("{sam}net_3.weight"), CLIP_DIM, 512, 3)?,
            ],
            clip_class: getf(&format!("{clip}embeddings.class_embedding"), CLIP_DIM)?,
            clip_pos: getf(
                &format!("{clip}embeddings.position_embedding.weight"),
                (CLIP_GRID * CLIP_GRID + 1) * CLIP_DIM,
            )?,
            clip_pre: LayerNorm::load(files, &format!("{clip}pre_layrnorm"), CLIP_DIM)?,
            clip_blocks,
            proj: dense("model.projector.layers", hidden, 2 * CLIP_DIM)?,
            newline: getf("model.image_newline", hidden)?,
            separator: getf("model.view_seperator", hidden)?,
            hidden,
        })
    }

    /// SAM over one view `[3, s, s]`: `[(s / 64)^2, 1024]` and the side of that grid.
    fn sam(&self, px: &[f32], s: usize) -> (Vec<f32>, usize) {
        let g = s / 16;
        let n = g * g;
        let d = 3 * 16 * 16;
        let mut patches = vec![0f32; n * d];
        for_chunks(&mut patches, d, true, |i, row| {
            let (py, pxi) = (i / g, i % g);
            for c in 0..3 {
                for y in 0..16 {
                    let src = (c * s + py * 16 + y) * s + pxi * 16;
                    row[(c * 16 + y) * 16..(c * 16 + y + 1) * 16].copy_from_slice(&px[src..src + 16]);
                }
            }
        });
        let mut x = vec![0f32; n * SAM_DIM];
        linear(&self.patch_w, SAM_DIM, d, &patches, n, &mut x);
        for_chunks(&mut x, SAM_DIM, true, |_, r| add_inplace(r, &self.patch_b));
        if g == SAM_GRID {
            add_inplace(&mut x, &self.sam_pos);
        } else {
            add_inplace(&mut x, &interp_bicubic_aa(&self.sam_pos, SAM_GRID, g, SAM_DIM));
        }
        for b in &self.sam_blocks {
            self.sam_block(b, &mut x, g);
        }
        let (mut y, _, _) = self.neck[0].forward(&x, g, g, 1);
        self.neck_norm[0].apply_rows(&mut y, 1e-6);
        let (mut y, _, _) = self.neck[1].forward(&y, g, g, 1);
        self.neck_norm[1].apply_rows(&mut y, 1e-6);
        let (y, h2, w2) = self.net[0].forward(&y, g, g, 2);
        let (y, h3, _) = self.net[1].forward(&y, h2, w2, 2);
        (y, h3)
    }

    fn sam_block(&self, b: &SamBlock, x: &mut [f32], g: usize) {
        let n = g * g;
        let mut hn = x.to_vec();
        b.norm1.apply_rows(&mut hn, 1e-6);
        let mut att = vec![0f32; n * SAM_DIM];
        if b.window == 0 {
            let side = 2 * g - 1;
            let (rh, rw) = (
                interp_linear(&b.rel_h, 2 * SAM_GRID - 1, side, SAM_HD),
                interp_linear(&b.rel_w, 2 * SAM_GRID - 1, side, SAM_HD),
            );
            let mut qkv = vec![0f32; n * 3 * SAM_DIM];
            b.qkv.forward(&hn, n, &mut qkv);
            sam_attention(&qkv, g, g, &rh, &rw, &mut att);
        } else {
            // Windows of the grid zero-padded (after the norm) to a multiple of the window.
            let ws = b.window;
            let nw = g.div_ceil(ws);
            let wn = ws * ws;
            let mut wins = vec![0f32; nw * nw * wn * SAM_DIM];
            let cell = |i: usize| {
                let (win, t) = (i / wn, i % wn);
                ((win / nw) * ws + t / ws, (win % nw) * ws + t % ws)
            };
            for_chunks(&mut wins, SAM_DIM, true, |i, r| {
                let (y, xx) = cell(i);
                if y < g && xx < g {
                    r.copy_from_slice(&hn[(y * g + xx) * SAM_DIM..(y * g + xx + 1) * SAM_DIM]);
                }
            });
            let mut qkv = vec![0f32; nw * nw * wn * 3 * SAM_DIM];
            b.qkv.forward(&wins, nw * nw * wn, &mut qkv);
            let mut o = vec![0f32; wn * SAM_DIM];
            for win in 0..nw * nw {
                sam_attention(
                    &qkv[win * wn * 3 * SAM_DIM..(win + 1) * wn * 3 * SAM_DIM],
                    ws,
                    ws,
                    &b.rel_h,
                    &b.rel_w,
                    &mut o,
                );
                for t in 0..wn {
                    let (y, xx) = cell(win * wn + t);
                    if y < g && xx < g {
                        att[(y * g + xx) * SAM_DIM..(y * g + xx + 1) * SAM_DIM].copy_from_slice(&o[t * SAM_DIM..(t + 1) * SAM_DIM]);
                    }
                }
            }
        }
        let mut y = vec![0f32; n * SAM_DIM];
        b.proj.forward(&att, n, &mut y);
        add_inplace(x, &y);
        hn.copy_from_slice(x);
        b.norm2.apply_rows(&mut hn, 1e-6);
        let mut mid = vec![0f32; n * 4 * SAM_DIM];
        b.lin1.forward(&hn, n, &mut mid);
        for_chunks(&mut mid, 4 * SAM_DIM, true, |_, r| r.iter_mut().for_each(|v| *v = gelu_erf(*v)));
        b.lin2.forward(&mid, n, &mut y);
        add_inplace(x, &y);
    }

    /// CLIP over SAM's features (`[g * g, 1024]`, its patch embeddings): `[g * g, 1024]`, the
    /// last block's output without the class token.
    fn clip(&self, feats: &[f32], g: usize) -> Vec<f32> {
        let (h, nh) = (CLIP_DIM, CLIP_HEADS);
        let n = g * g + 1;
        let mut x = vec![0f32; n * h];
        x[..h].copy_from_slice(&self.clip_class);
        x[h..].copy_from_slice(feats);
        if g == CLIP_GRID {
            add_inplace(&mut x, &self.clip_pos);
        } else {
            add_inplace(&mut x[..h], &self.clip_pos[..h]);
            add_inplace(&mut x[h..], &interp_bicubic_aa(&self.clip_pos[h..], CLIP_GRID, g, h));
        }
        self.clip_pre.apply_rows(&mut x, 1e-5);
        let hd = h / nh;
        let scale = (hd as f32).powf(-0.5);
        let (mut hn, mut qkv) = (vec![0f32; n * h], vec![0f32; n * 3 * h]);
        let (mut q, mut k, mut v) = (vec![0f32; n * h], vec![0f32; n * h], vec![0f32; n * h]);
        let (mut att, mut y, mut mid) = (vec![0f32; n * h], vec![0f32; n * h], vec![0f32; n * 4 * h]);
        for b in &self.clip_blocks {
            hn.copy_from_slice(&x);
            b.norm1.apply_rows(&mut hn, 1e-5);
            b.qkv.forward(&hn, n, &mut qkv);
            for i in 0..n {
                let row = &qkv[i * 3 * h..(i + 1) * 3 * h];
                for (o, &s) in q[i * h..(i + 1) * h].iter_mut().zip(&row[..h]) {
                    *o = s * scale;
                }
                k[i * h..(i + 1) * h].copy_from_slice(&row[h..2 * h]);
                v[i * h..(i + 1) * h].copy_from_slice(&row[2 * h..]);
            }
            crate::vision::attention_bidirectional(&q, &k, &v, n, nh, hd, &mut att);
            b.out.forward(&att, n, &mut y);
            add_inplace(&mut x, &y);
            hn.copy_from_slice(&x);
            b.norm2.apply_rows(&mut hn, 1e-5);
            b.fc1.forward(&hn, n, &mut mid);
            for_chunks(&mut mid, 4 * h, true, |_, r| r.iter_mut().for_each(|v| *v = quick_gelu(*v)));
            b.fc2.forward(&mid, n, &mut y);
            add_inplace(&mut x, &y);
        }
        x[h..].to_vec()
    }

    /// One view `[3, s, s]`: the encoder features `[(s / 64)^2, 2048]` (CLIP's, then SAM's) and
    /// the side of the grid.
    pub fn features(&self, px: &[f32], s: usize) -> (Vec<f32>, usize) {
        let (sam, g) = self.sam(px, s);
        let clip = self.clip(&sam, g);
        let mut out = vec![0f32; g * g * 2 * CLIP_DIM];
        for (i, o) in out.chunks_mut(2 * CLIP_DIM).enumerate() {
            o[..CLIP_DIM].copy_from_slice(&clip[i * CLIP_DIM..(i + 1) * CLIP_DIM]);
            o[CLIP_DIM..].copy_from_slice(&sam[i * CLIP_DIM..(i + 1) * CLIP_DIM]);
        }
        (out, g)
    }

    /// Encoder features into the language model's space: `[n, hidden]`.
    pub fn project(&self, feats: &[f32]) -> Vec<f32> {
        let n = feats.len() / (2 * CLIP_DIM);
        let mut out = vec![0f32; n * self.hidden];
        self.proj.forward(feats, n, &mut out);
        out
    }

    /// The packed embeddings of a page, `[page.tokens(), hidden]`: the tiles' grid, the global
    /// view, the separator. `global_feats`: receives the global view's encoder features.
    pub fn encode(&self, page: &PagePixels, global_feats: Option<&mut Vec<f32>>) -> Vec<f32> {
        let d = self.hidden;
        let mut out = Vec::with_capacity(page.tokens() * d);
        if !page.tiles.is_empty() {
            let (tw, th) = page.grid;
            let mut g = 0;
            let tiles: Vec<Vec<f32>> = page
                .tiles
                .iter()
                .map(|t| {
                    let (f, side) = self.features(t, TILE);
                    g = side;
                    self.project(&f)
                })
                .collect();
            for r in 0..th * g {
                for c in 0..tw * g {
                    let f = &tiles[(r / g) * tw + c / g];
                    let i = (r % g) * g + c % g;
                    out.extend_from_slice(&f[i * d..(i + 1) * d]);
                }
                out.extend_from_slice(&self.newline);
            }
        }
        let (f, g) = self.features(&page.global, BASE);
        let emb = self.project(&f);
        if let Some(gf) = global_feats {
            *gf = f;
        }
        for r in 0..g {
            out.extend_from_slice(&emb[r * g * d..(r + 1) * g * d]);
            out.extend_from_slice(&self.newline);
        }
        out.extend_from_slice(&self.separator);
        debug_assert_eq!(out.len(), page.tokens() * d);
        out
    }
}

// ---- prompt and decoding rules --------------------------------------------------------------

/// The prompt text of `infer()` (after BOS) for pages of `counts` image tokens: without `<image>`
/// in `prompt`, `<PAGE>` and the page's placeholders per page, then the instruction; a prompt
/// with one `<image>` per page is used as is, one `<image>` for several pages stands for all of
/// them (with their `<PAGE>` markers).
pub fn prompt_text(prompt: &str, counts: &[usize]) -> Result<String> {
    let prompt = if prompt.is_empty() && !counts.is_empty() {
        DEFAULT_PROMPT
    } else {
        prompt
    };
    if prompt.is_empty() {
        bail!("prompt is empty");
    }
    let marked: String = counts.iter().map(|&n| format!("{PAGE}{}", IMAGE.repeat(n))).collect();
    let parts: Vec<&str> = prompt.split(IMAGE).collect();
    let n_ph = parts.len() - 1;
    Ok(if n_ph == 0 {
        format!("{marked}{prompt}")
    } else if counts.len() > 1 && n_ph == 1 {
        format!("{}{marked}{}", parts[0], parts[1])
    } else if n_ph == counts.len() {
        let mut s = String::new();
        for (i, p) in parts.iter().enumerate() {
            s.push_str(p);
            if let Some(&n) = counts.get(i) {
                s.push_str(&IMAGE.repeat(n));
            }
        }
        s
    } else {
        bail!("the prompt has {n_ph} {IMAGE} placeholders for {} images", counts.len())
    })
}

/// The model's "chat" format: BOS and the prompt as it is, no roles; a reply ends with EOS.
pub struct OcrChat;

impl crate::chat::ChatFormat for OcrChat {
    fn conversation_start(&self, _system: Option<&str>, _think: bool) -> String {
        BOS.to_string()
    }
    /// A message whose attachments come first (`<PAGE>` and the placeholders of every page, as
    /// [`crate::mm`] writes them: runs of at least 273 `<image>`): the placeholders the client
    /// wrote itself (the README's request text has `<PAGE><image>` per page) are dropped, and
    /// pages without an instruction get `infer()`'s. Other messages are fed as they are.
    fn user_turn(&self, content: &str, _think: bool) -> String {
        let (mut pages, mut rest) = (String::new(), content);
        loop {
            let marked = rest.strip_prefix(PAGE);
            let after = marked.unwrap_or(rest);
            let images = after.trim_start_matches(IMAGE);
            let n = (after.len() - images.len()) / IMAGE.len();
            if marked.is_none() && n == 0 {
                break;
            }
            // A single placeholder is the client's.
            if n > 1 {
                pages.push_str(&rest[..rest.len() - images.len()]);
            }
            rest = images;
        }
        if pages.is_empty() {
            return content.to_string();
        }
        let rest = rest.replace(PAGE, "").replace(IMAGE, "");
        format!("{pages}{}", if rest.trim().is_empty() { DEFAULT_PROMPT } else { &rest })
    }
    fn end_of_model_turn(&self) -> String {
        EOS.to_string()
    }
    fn keeps_special_tokens(&self) -> bool {
        true
    }
    fn end_of_turn_token(&self) -> &'static str {
        EOS
    }
    fn think_tokens(&self) -> (&'static str, &'static str) {
        ("", "")
    }
    fn starts_in_thinking(&self, _think: bool) -> bool {
        false
    }
    fn conversation_start_tools(&self, system: Option<&str>, think: bool, _tools: &[crate::tools::ToolSpec]) -> String {
        self.conversation_start(system, think)
    }
    fn tool_call_marker(&self) -> &'static str {
        ""
    }
    fn parse_tool_calls(&self, _raw: &str) -> Vec<crate::tools::ToolCall> {
        Vec::new()
    }
    fn tool_results(&self, _results: &[crate::tools::ToolResult], _think: bool) -> String {
        String::new()
    }
}

/// The logit rules of `infer()` as a sampler filter: the page count guard (as many `<PAGE>`
/// markers in the reply as in the prompt, a cap on a page's tokens) and the optional sliding
/// window no-repeat n-gram rule, both over the generated tokens only.
#[derive(Clone, Debug)]
pub struct OcrGuard {
    /// `<PAGE>` markers of the prompt (0: no guard).
    pub expected_pages: usize,
    pub page_id: u32,
    pub eos_id: u32,
    /// Tokens of one page after which the next marker (or the end) is forced (0: no cap).
    pub max_page_tokens: usize,
    /// n-gram size (0: off) and the window of last tokens searched.
    pub ngram: usize,
    pub window: usize,
    generated: Vec<u32>,
    markers: usize,
    page_len: usize,
}

impl OcrGuard {
    /// `infer()`'s defaults for a prompt of `prompt_ids`: the guard on, 20000 tokens per page,
    /// no n-gram rule.
    pub fn new(cfg: &OcrConfig, prompt_ids: &[u32]) -> Self {
        Self {
            expected_pages: prompt_ids.iter().filter(|&&t| t == cfg.page_token_id).count(),
            page_id: cfg.page_token_id,
            eos_id: cfg.eos_token_id,
            max_page_tokens: 20000,
            ngram: 0,
            window: 100,
            generated: Vec::new(),
            markers: 0,
            page_len: 0,
        }
    }

    /// The no-repeat rule: no n-gram of `ngram` tokens twice within the last `window` tokens.
    pub fn with_ngram(mut self, ngram: usize, window: usize) -> Self {
        self.ngram = ngram;
        self.window = if window > 0 { window } else { 100 };
        self
    }
}

impl crate::sampler::TokenFilter for OcrGuard {
    fn mask(&mut self, logits: &mut [f32]) {
        let g = &self.generated;
        let n = self.ngram;
        if n > 0 && g.len() >= n {
            let prefix = &g[g.len() - (n - 1)..];
            let start = g.len().saturating_sub(self.window);
            for i in start..(g.len() + 1).saturating_sub(n) {
                if &g[i..i + n - 1] == prefix {
                    logits[g[i + n - 1] as usize] = f32::NEG_INFINITY;
                }
            }
        }
        if self.expected_pages == 0 {
            return;
        }
        let only = |logits: &mut [f32], id: u32| {
            let keep = logits[id as usize];
            logits.fill(f32::NEG_INFINITY);
            // A token the n-gram rule banned is still the only one allowed.
            logits[id as usize] = if keep == f32::NEG_INFINITY { 0.0 } else { keep };
        };
        if self.markers > self.expected_pages {
            only(logits, self.eos_id);
        } else if self.max_page_tokens > 0 && self.page_len >= self.max_page_tokens {
            only(
                logits,
                if self.markers < self.expected_pages {
                    self.page_id
                } else {
                    self.eos_id
                },
            );
        } else if self.markers < self.expected_pages {
            logits[self.eos_id as usize] = f32::NEG_INFINITY;
        }
    }
    fn accept(&mut self, token: u32) {
        self.generated.push(token);
        if token == self.page_id {
            self.markers += 1;
            self.page_len = 0;
        } else {
            self.page_len += 1;
        }
    }
    fn fork(&self) -> Box<dyn crate::sampler::TokenFilter> {
        Box::new(self.clone())
    }
    fn complete(&self) -> bool {
        self.markers >= self.expected_pages
    }
}

// ---- the language model ---------------------------------------------------------------------

/// Masked scores of the router (`_ENTMAX_NEG`) and the clamp on its activations.
const ENTMAX_NEG: f32 = -1.0e4;
const ACT_CLAMP: f32 = 1.0e4;

struct Mlp<'a> {
    gate: Lin<'a>,
    up: Lin<'a>,
    down: Lin<'a>,
    inter: usize,
}

impl Mlp<'_> {
    fn forward(&self, x: &[f32], t: usize, h: usize, y: &mut [f32]) {
        let mut g = vec![0f32; t * self.inter];
        let mut u = vec![0f32; t * self.inter];
        linear_multi(
            &mut [(&self.gate, self.inter, &mut g[..]), (&self.up, self.inter, &mut u[..])],
            h,
            x,
            t,
        );
        silu_mul(&mut g, &u);
        linear(&self.down, h, self.inter, &g, t, y);
    }
}

// One per layer, built once.
#[allow(clippy::large_enum_variant)]
enum Ffn<'a> {
    Dense(Mlp<'a>),
    Moe {
        router: Vec<f32>,
        experts: Experts<'a>,
        shared: Option<Mlp<'a>>,
    },
}

/// The block router of a layer (`dash_router`).
struct Router<'a> {
    q: Lin<'a>,
    k: Lin<'a>,
    /// `1 + weight` of the two Gemma-style norms.
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    head_cls: Vec<f32>,
}

struct Layer<'a> {
    ln_in: Vec<f32>,
    ln_post: Vec<f32>,
    q: Lin<'a>,
    k: Lin<'a>,
    v: Lin<'a>,
    o: Lin<'a>,
    router: Router<'a>,
    ffn: Ffn<'a>,
}

/// A layer's cache: rotated keys and values `[pos, heads * head]`, the router's rotated keys
/// `[pos, index]` and the summary key of every full block `[blocks, index]`.
#[derive(Clone, Default)]
struct LayerState {
    k: Vec<f32>,
    v: Vec<f32>,
    ik: Vec<f32>,
    kbar: Vec<f32>,
}

/// Where the pages are, from the prompt (see `_build_page_rope_pos`, `_build_page_aware_ids` and
/// `_uocr_dash_decode_allow`), and the state the reply's tokens advance.
#[derive(Clone, Default)]
struct Pages {
    /// Two-axis positions (the prompt has images and `<PAGE>` markers); else plain positions.
    two_d: bool,
    /// The last token's page and index in it.
    page: usize,
    tok: usize,
    /// Prompt tokens per page (through the last image): where a page's reply continues.
    pf_cnt: Vec<usize>,
    /// Per prompt block: the pages of its image tokens (prefill candidates) and their range
    /// (decode candidates; -1: no image).
    blk_pages: Vec<Vec<i64>>,
    bmin: Vec<i64>,
    bmax: Vec<i64>,
    /// Markers through the last image, and seen so far.
    pp: i64,
    mcum: i64,
}

/// A token's position for the rotary embedding.
#[derive(Clone, Copy)]
enum Pos {
    Page { tok: usize, page: usize },
    Plain(usize),
}

/// The blocks a query may choose from: those with image tokens of page `.0` (all when the page
/// is negative or has none), by the exact pages of the blocks (prompt) or their range (reply).
#[derive(Clone, Copy)]
struct Cand(i64, bool);

pub struct OcrSession {
    pub pos: usize,
    layers: Vec<LayerState>,
    pages: Pages,
}

impl OcrSession {
    /// Drops the cache: the next pass is a prompt again.
    pub fn clear(&mut self) {
        self.pos = 0;
        self.layers.iter_mut().for_each(|l| *l = LayerState::default());
        self.pages = Pages::default();
    }
}

pub struct OcrModel<'a> {
    pub cfg: OcrConfig,
    pub quant: Quant,
    embed: Table<'a>,
    lm_head: Lin<'a>,
    final_norm: Vec<f32>,
    layers: Vec<Layer<'a>>,
    /// Inverse frequencies of the token axis and of the page axis: the attention's, the router's.
    main_freq: (Vec<f32>, Vec<f32>),
    idx_freq: (Vec<f32>, Vec<f32>),
    /// Plain positions: YaRN frequencies and magnitude.
    yarn: (Vec<f32>, f32),
}

/// `1 / base^(i / n_tok)` (`resample`) or `1 / base^(2 i / dim)` (`truncate`) for the token
/// pairs, `1 / page_base^(i / pairs)` for the page pairs, in float32 as the reference.
fn page_freqs(dim: usize, tok_base: f64, page_base: f64, pairs: usize, resample: bool) -> (Vec<f32>, Vec<f32>) {
    let n_tok = dim / 2 - pairs;
    let tok = (0..n_tok)
        .map(|i| {
            let e = if resample {
                i as f32 / n_tok as f32
            } else {
                i as f32 * 2.0 / dim as f32
            };
            1.0 / (tok_base as f32).powf(e)
        })
        .collect();
    let page = (0..pairs).map(|i| 1.0 / (page_base as f32).powf(i as f32 / pairs as f32)).collect();
    (tok, page)
}

/// 1.5-entmax of scores sorted in decreasing order (`dash_attention.entmax15`).
fn entmax15(x: &[f32]) -> Vec<f32> {
    let xs: Vec<f32> = x.iter().map(|v| (v - x[0]) / 2.0).collect();
    let (mut cs, mut cs2, mut support) = (0f32, 0f32, 0usize);
    let mut taus = Vec::with_capacity(xs.len());
    for (k, &v) in xs.iter().enumerate() {
        let rho = (k + 1) as f32;
        cs += v;
        cs2 += v * v;
        let (mean, mean_sq) = (cs / rho, cs2 / rho);
        let delta = (1.0 - rho * (mean_sq - mean * mean)) / rho;
        let tau = mean - delta.max(0.0).sqrt();
        support += usize::from(tau <= v);
        taus.push(tau);
    }
    let tau = taus[support.max(1) - 1];
    xs.iter().map(|v| (v - tau).max(0.0).powi(2)).collect()
}

impl<'a> OcrModel<'a> {
    pub fn load(cfg: OcrConfig, files: &'a ModelFiles, quant: Quant) -> Result<Self> {
        let quant = Quant::stored(files).unwrap_or(quant);
        let (h, v, d) = (cfg.hidden_size, cfg.vocab_size, cfg.index_dim);
        let getf = |n: &str, len: usize| -> Result<Vec<f32>> {
            let t = files.f32(n).with_context(|| format!("missing tensor {n}"))?;
            if t.len() != len {
                bail!("tensor {n} has {} values, expected {len}", t.len());
            }
            Ok(t)
        };
        let lin = |n: &str, o: usize, i: usize| load_lin(files, n, o, i, quant);
        let mlp = |p: &str, inter: usize| -> Result<Mlp<'a>> {
            Ok(Mlp {
                gate: lin(&format!("{p}gate_proj.weight"), inter, h)?,
                up: lin(&format!("{p}up_proj.weight"), inter, h)?,
                down: lin(&format!("{p}down_proj.weight"), h, inter)?,
                inter,
            })
        };
        let one_plus = |mut w: Vec<f32>| {
            w.iter_mut().for_each(|v| *v += 1.0);
            w
        };
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let lp = format!("model.layers.{l}.");
            let (ap, rp, mp) = (
                format!("{lp}self_attn."),
                format!("{lp}self_attn.dash_router."),
                format!("{lp}mlp."),
            );
            let ffn = if cfg.is_moe(l) {
                let (ne, mi) = (cfg.n_routed_experts, cfg.moe_intermediate_size);
                Ffn::Moe {
                    router: getf(&format!("{mp}gate.weight"), ne * h)?,
                    experts: Experts::load(
                        files,
                        &format!("{mp}experts."),
                        ne,
                        cfg.num_experts_per_tok,
                        mi,
                        h,
                        quant,
                        Act::Silu,
                    )?,
                    shared: if cfg.n_shared_experts > 0 {
                        Some(mlp(&format!("{mp}shared_experts."), mi * cfg.n_shared_experts)?)
                    } else {
                        None
                    },
                }
            } else {
                Ffn::Dense(mlp(&mp, cfg.intermediate_size)?)
            };
            layers.push(Layer {
                ln_in: getf(&format!("{lp}input_layernorm.weight"), h)?,
                ln_post: getf(&format!("{lp}post_attention_layernorm.weight"), h)?,
                q: lin(&format!("{ap}q_proj.weight"), h, h)?,
                k: lin(&format!("{ap}k_proj.weight"), h, h)?,
                v: lin(&format!("{ap}v_proj.weight"), h, h)?,
                o: lin(&format!("{ap}o_proj.weight"), h, h)?,
                router: Router {
                    q: load_lin(files, &format!("{rp}q_proj.weight"), d, h, Quant::Bf16)?,
                    k: load_lin(files, &format!("{rp}k_proj.weight"), d, h, Quant::Bf16)?,
                    q_norm: one_plus(getf(&format!("{rp}q_norm.weight"), d)?),
                    k_norm: one_plus(getf(&format!("{rp}k_norm.weight"), d)?),
                    head_cls: getf(&format!("{rp}head_cls"), d)?,
                },
                ffn,
            });
        }
        let hd = cfg.head_dim();
        let (tt, pt, pp) = (cfg.rope_tok_theta, cfg.rope_page_theta, cfg.rope_page_pairs);
        Ok(Self {
            embed: Table::load(files, "model.embed_tokens.weight", v, h, table_quant(quant))?,
            lm_head: lin("lm_head.weight", v, h)?,
            final_norm: getf("model.norm.weight", h)?,
            layers,
            main_freq: page_freqs(hd, tt, pt, pp, cfg.main_resample),
            idx_freq: page_freqs(d, tt, pt, pp, cfg.idx_resample),
            yarn: cfg.yarn.inv_freq(hd),
            quant,
            cfg,
        })
    }

    pub fn new_session(&self) -> OcrSession {
        OcrSession {
            pos: 0,
            layers: vec![LayerState::default(); self.layers.len()],
            pages: Pages::default(),
        }
    }

    /// Logits `[t, vocab]` for final-normed hidden states `[t, hidden]`.
    pub fn logits(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        let c = &self.cfg;
        let mut logits = vec![0f32; t * c.vocab_size];
        linear(&self.lm_head, c.vocab_size, c.hidden_size, hidden, t, &mut logits);
        logits
    }

    /// Runs `tokens` and returns the logits after the last one.
    pub fn forward(&self, sess: &mut OcrSession, tokens: &[u32]) -> Vec<f32> {
        let h = self.cfg.hidden_size;
        let hs = self.forward_hidden_mm(sess, tokens, &[]);
        self.logits(&hs[hs.len() - h..], 1)
    }

    /// Runs `tokens` (positions in `mm` take the given embeddings instead of the token's): the
    /// whole prompt in an empty session, decoding steps after it. Returns the final-normed
    /// hidden states `[t, hidden]`.
    pub fn forward_hidden_mm(&self, sess: &mut OcrSession, tokens: &[u32], mm: &[(usize, &[f32])]) -> Vec<f32> {
        assert!(!tokens.is_empty());
        let h = self.cfg.hidden_size;
        if sess.pos == 0 {
            let (pos, cand) = self.plan_prompt(&mut sess.pages, tokens);
            return self.run(sess, tokens, mm, &pos, &cand);
        }
        let mut out = Vec::with_capacity(tokens.len() * h);
        for (i, &t) in tokens.iter().enumerate() {
            let (pos, cand) = self.plan_step(&mut sess.pages, t, sess.pos);
            let one: Vec<(usize, &[f32])> = mm.iter().filter(|m| m.0 == i).map(|m| (0, m.1)).collect();
            out.extend(self.run(sess, &[t], &one, &[pos], &[cand]));
        }
        out
    }

    /// The positions and candidate pages of a prompt's tokens, and the page state after it.
    fn plan_prompt(&self, pages: &mut Pages, ids: &[u32]) -> (Vec<Pos>, Vec<Cand>) {
        let c = &self.cfg;
        let s = ids.len();
        let bs = c.block_size;
        let img: Vec<bool> = ids.iter().map(|&t| t == c.image_token_id).collect();
        // Markers and image tokens through each position.
        let (mut cum, mut run) = (vec![0i64; s], vec![0i64; s]);
        let (mut m, mut r) = (0i64, 0i64);
        for i in 0..s {
            m += i64::from(ids[i] == c.page_token_id);
            r += i64::from(img[i]);
            (cum[i], run[i]) = (m, r);
        }
        let last_img = img.iter().rposition(|&b| b);
        // Markers through the last image: the prompt's pages.
        let n_pages = last_img.map_or(0, |i| cum[i]);
        let in_prompt = |i: usize| last_img.is_some_and(|l| i <= l);
        *pages = Pages {
            two_d: n_pages > 0,
            pp: n_pages,
            mcum: m,
            ..Pages::default()
        };
        // Positions: the page (0: no page, the prompt's pages from 1, the reply's again from 1)
        // and the running index inside it.
        let pos: Vec<Pos> = if pages.two_d {
            let page: Vec<usize> = (0..s)
                .map(|i| if in_prompt(i) { cum[i] } else { cum[i] - n_pages }.max(0) as usize)
                .collect();
            let n_pg = page.iter().max().unwrap() + 1;
            let mut count = vec![0usize; n_pg];
            pages.pf_cnt = vec![0; n_pg];
            let pos = (0..s)
                .map(|i| {
                    let p = page[i];
                    count[p] += 1;
                    pages.pf_cnt[p] += usize::from(in_prompt(i));
                    Pos::Page {
                        tok: count[p] - 1,
                        page: p,
                    }
                })
                .collect();
            (pages.page, pages.tok) = (page[s - 1], count[page[s - 1]] - 1);
            pos
        } else {
            (0..s).map(Pos::Plain).collect()
        };
        // The page of every image token: by the markers, or (no marker before the last image)
        // in runs of `per_page_image_tokens`. The reply's rule also uses the markers of a prompt
        // whose first images come before any.
        let per = c.per_page_image_tokens as i64;
        let img_page = |i: usize| if n_pages > 0 { cum[i] - 1 } else { (run[i] - 1) / per };
        let dec_page = |i: usize| if cum[i] > 0 { cum[i] - 1 } else { (run[i] - 1) / per };
        for b in 0..s.div_ceil(bs) {
            let (mut exact, mut lo, mut hi) = (Vec::new(), i64::MAX, -1i64);
            for i in (b * bs..((b + 1) * bs).min(s)).filter(|&i| img[i]) {
                if !exact.contains(&img_page(i)) {
                    exact.push(img_page(i));
                }
                (lo, hi) = (lo.min(dec_page(i)), hi.max(dec_page(i)));
            }
            pages.blk_pages.push(exact);
            pages.bmin.push(if hi >= 0 { lo } else { -1 });
            pages.bmax.push(hi);
        }
        // A query's page: through the last image the markers so far, after it the markers of
        // the reply (none yet: the instruction looks at every page).
        let cand = (0..s)
            .map(|i| {
                let raw = cum[i] - 1;
                let q = if in_prompt(i) { raw } else { raw - n_pages };
                Cand(if last_img.is_some() { q } else { -1 }, true)
            })
            .collect();
        (pos, cand)
    }

    /// The position and candidate page of one more token `id` at `at`.
    fn plan_step(&self, pages: &mut Pages, id: u32, at: usize) -> (Pos, Cand) {
        let is_page = id == self.cfg.page_token_id;
        let pos = if pages.two_d {
            if is_page {
                pages.page = (pages.page + 1).min(pages.pf_cnt.len() - 1);
                pages.tok = pages.pf_cnt[pages.page];
            } else {
                pages.tok += 1;
            }
            Pos::Page {
                tok: pages.tok,
                page: pages.page,
            }
        } else {
            Pos::Plain(at)
        };
        pages.mcum += i64::from(is_page);
        (pos, Cand(pages.mcum - 1 - pages.pp, false))
    }

    /// cos / sin rows `[t, dim / 2]` of the positions for one of the two frequency sets.
    fn rope(&self, pos: &[Pos], freq: &(Vec<f32>, Vec<f32>)) -> (Vec<f32>, Vec<f32>) {
        let half = freq.0.len() + freq.1.len();
        let (mut cos, mut sin) = (vec![0f32; pos.len() * half], vec![0f32; pos.len() * half]);
        for (i, p) in pos.iter().enumerate() {
            let (c, s) = (&mut cos[i * half..(i + 1) * half], &mut sin[i * half..(i + 1) * half]);
            match *p {
                Pos::Page { tok, page } => {
                    let pg = if self.cfg.rope_page_mod > 0 {
                        page % self.cfg.rope_page_mod
                    } else {
                        page
                    };
                    let angles = freq.0.iter().map(|f| tok as f32 * f).chain(freq.1.iter().map(|f| pg as f32 * f));
                    for (j, a) in angles.enumerate() {
                        (c[j], s[j]) = (a.cos(), a.sin());
                    }
                }
                Pos::Plain(p) => {
                    for (j, f) in self.yarn.0.iter().enumerate() {
                        let a = p as f32 * f;
                        (c[j], s[j]) = (a.cos() * self.yarn.1, a.sin() * self.yarn.1);
                    }
                }
            }
        }
        (cos, sin)
    }

    /// `tokens` at `sess.pos` with their positions and candidate pages through every layer.
    fn run(&self, sess: &mut OcrSession, tokens: &[u32], mm: &[(usize, &[f32])], pos: &[Pos], cand: &[Cand]) -> Vec<f32> {
        let c = &self.cfg;
        let (t, h) = (tokens.len(), c.hidden_size);
        let eps = c.rms_norm_eps as f32;
        let mut x = vec![0f32; t * h];
        for_chunks(&mut x, h, t > 1, |ti, row| self.embed.row(tokens[ti] as usize, row));
        for &(i, emb) in mm {
            x[i * h..(i + 1) * h].copy_from_slice(emb);
        }
        let main = self.rope(pos, &self.main_freq);
        let idx = self.rope(pos, &self.idx_freq);
        let mut hn = vec![0f32; t * h];
        let mut o = vec![0f32; t * h];
        for (l, layer) in self.layers.iter().enumerate() {
            rmsnorm_f32w_rows(&x, h, &layer.ln_in, eps, &mut hn);
            self.attention(l, layer, &mut sess.layers[l], &sess.pages, &hn, sess.pos, cand, &main, &idx, &mut o);
            add_inplace(&mut x, &o);
            rmsnorm_f32w_rows(&x, h, &layer.ln_post, eps, &mut hn);
            o.fill(0.0);
            match &layer.ffn {
                Ffn::Dense(m) => m.forward(&hn, t, h, &mut o),
                Ffn::Moe { router, experts, shared } => self.moe(router, experts, shared.as_ref(), &hn, t, &mut o),
            }
            add_inplace(&mut x, &o);
        }
        rmsnorm_f32w_rows_inplace(&mut x, h, &self.final_norm, eps);
        sess.pos += t;
        x
    }

    fn moe(&self, router: &[f32], experts: &Experts, shared: Option<&Mlp>, x: &[f32], t: usize, y: &mut [f32]) {
        let c = &self.cfg;
        let (h, ne, k) = (c.hidden_size, c.n_routed_experts, c.num_experts_per_tok);
        let scale = c.routed_scaling_factor as f32;
        let mut probs = vec![0f32; t * ne];
        for_chunks(&mut probs, ne, t > 1, |ti, p| {
            for (v, w) in p.iter_mut().zip(router.chunks(h)) {
                *v = dot(w, &x[ti * h..(ti + 1) * h]);
            }
            softmax_inplace(p);
        });
        // The top k of the softmax over all experts, kept as they are unless `norm_topk_prob`.
        let routes: Vec<Vec<(usize, f32)>> = probs
            .chunks(ne)
            .map(|p| {
                let mut idx: Vec<usize> = (0..ne).collect();
                idx.sort_by(|&a, &b| p[b].total_cmp(&p[a]).then(a.cmp(&b)));
                idx.truncate(k);
                let norm = if k > 1 && c.norm_topk_prob {
                    idx.iter().map(|&i| p[i]).sum::<f32>() + 1e-20
                } else {
                    1.0
                };
                idx.into_iter().map(|i| (i, p[i] / norm * scale)).collect()
            })
            .collect();
        let shared = shared.map(|s| Shared {
            gate: &s.gate,
            up: &s.up,
            down: &s.down,
            inter: s.inter,
            act: Act::Silu,
            hook: None,
            scale: None,
        });
        experts.forward(x, t, h, &routes, y, None, None, shared.as_ref());
    }

    /// The blocks a query at `p` reads beyond its local window, with their score bias
    /// (`_block_scores_infer` and `_infer_route_select`): `iq` is its router query, `kbar` the
    /// summary keys of the full blocks, `nblk` the blocks of the whole sequence so far.
    #[allow(clippy::too_many_arguments)]
    fn select(&self, l: usize, iq: &[f32], p: usize, nblk: usize, kbar: &[f32], pages: &Pages, cand: Cand) -> Vec<(usize, f32)> {
        let c = &self.cfg;
        let d = c.index_dim;
        let qb = p / c.block_size;
        if qb < c.local_blocks {
            return Vec::new();
        }
        let n_remote = qb - c.local_blocks + 1;
        let allowed = |b: usize| {
            if cand.1 {
                pages.blk_pages.get(b).is_some_and(|pg| pg.contains(&cand.0))
            } else {
                b < pages.bmin.len() && pages.bmin[b] >= 0 && pages.bmin[b] <= cand.0 && cand.0 <= pages.bmax[b]
            }
        };
        let any = cand.0 >= 0 && (0..n_remote).any(allowed);
        let inv_sqrt_d = 1.0 / (d as f32).sqrt();
        let idx_scale = c.scale[l] * inv_sqrt_d;
        let mut xs: Vec<(f32, usize)> = (0..n_remote)
            .filter(|&b| !any || allowed(b))
            .map(|b| {
                let z = (dot(iq, &kbar[b * d..(b + 1) * d]) * inv_sqrt_d).clamp(-ACT_CLAMP, ACT_CLAMP);
                (z * idx_scale, b)
            })
            .collect();
        xs.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        // The entmax runs over the best M of all the blocks' scores, the others masked.
        let m = if c.topm > 0 { c.topm.min(nblk).max(1) } else { nblk };
        xs.truncate(m);
        let mut top: Vec<f32> = xs.iter().map(|x| x.0).collect();
        top.resize(m, ENTMAX_NEG);
        let prior = entmax15(&top);
        xs.truncate(c.topk_blocks.min(m).max(1));
        if xs.is_empty() {
            return Vec::new();
        }
        let logp: Vec<f32> = prior[..xs.len()].iter().map(|p| p.max(c.floor[l]).ln()).collect();
        let mu = logp.iter().sum::<f32>() / logp.len() as f32;
        xs.iter().zip(&logp).map(|(x, lp)| (x.1, (lp - mu) / c.sigma)).collect()
    }

    /// One layer's attention over the normed residual `hn` (`[t, H]`) into `out`.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        l: usize,
        layer: &Layer,
        st: &mut LayerState,
        pages: &Pages,
        hn: &[f32],
        start: usize,
        cand: &[Cand],
        main: &(Vec<f32>, Vec<f32>),
        idx: &(Vec<f32>, Vec<f32>),
        out: &mut [f32],
    ) {
        let c = &self.cfg;
        let (h, nh, hd, d, bs) = (c.hidden_size, c.num_attention_heads, c.head_dim(), c.index_dim, c.block_size);
        let t = hn.len() / h;
        let (mut q, mut k, mut v) = (vec![0f32; t * h], vec![0f32; t * h], vec![0f32; t * h]);
        let (mut iq, mut ik) = (vec![0f32; t * d], vec![0f32; t * d]);
        linear_multi(
            &mut [(&layer.q, h, &mut q[..]), (&layer.k, h, &mut k[..]), (&layer.v, h, &mut v[..])],
            h,
            hn,
            t,
        );
        linear_multi(
            &mut [(&layer.router.q, d, &mut iq[..]), (&layer.router.k, d, &mut ik[..])],
            h,
            hn,
            t,
        );
        let rot = |x: &mut [f32], n: usize, tab: &(Vec<f32>, Vec<f32>)| {
            // Rows of `n` (every head of a token, or the router's one row) with the token's angles.
            let (hf, per) = (n / 2, x.len() / (t * n));
            for_chunks(x, n, t > 8, |i, r| {
                let ti = i / per;
                apply_rope(r, &tab.0[ti * hf..(ti + 1) * hf], &tab.1[ti * hf..(ti + 1) * hf]);
            });
        };
        rot(&mut q, hd, main);
        rot(&mut k, hd, main);
        // The router's projections: RMS-normed (scale 1 + weight), clamped, rotated.
        let eps = c.rms_norm_eps as f32;
        let norm = |x: &mut [f32], w: &[f32]| {
            for_chunks(x, d, t > 8, |_, r| {
                let ms = r.iter().map(|v| v * v).sum::<f32>() / d as f32;
                let inv = 1.0 / (ms + eps).sqrt();
                for (v, w) in r.iter_mut().zip(w) {
                    *v = ((*v * inv).clamp(-ACT_CLAMP, ACT_CLAMP) * w).clamp(-ACT_CLAMP, ACT_CLAMP);
                }
            });
        };
        norm(&mut iq, &layer.router.q_norm);
        norm(&mut ik, &layer.router.k_norm);
        rot(&mut iq, d, idx);
        rot(&mut ik, d, idx);
        st.k.extend_from_slice(&k);
        st.v.extend_from_slice(&v);
        st.ik.extend_from_slice(&ik);
        let len = start + t;
        // The summary key of every block now full: its router keys pooled by `head_cls`.
        let inv_sqrt_d = 1.0 / (d as f32).sqrt();
        for b in st.kbar.len() / d..len / bs {
            let keys = &st.ik[b * bs * d..(b + 1) * bs * d];
            let mut w: Vec<f32> = keys
                .chunks(d)
                .map(|kk| (dot(kk, &layer.router.head_cls) * inv_sqrt_d).clamp(-ACT_CLAMP, ACT_CLAMP))
                .collect();
            softmax_inplace(&mut w);
            let mut kbar = vec![0f32; d];
            for (kk, &wj) in keys.chunks(d).zip(&w) {
                axpy(&mut kbar, wj, kk);
            }
            st.kbar.extend_from_slice(&kbar);
        }
        let nblk = len.div_ceil(bs);
        let local = c.local_blocks * bs;
        let scale = 1.0 / (hd as f32).sqrt();
        let st = &*st;
        for_chunks(out, h, t > 1, |i, o| {
            let p = start + i;
            let sel = self.select(l, &iq[i * d..(i + 1) * d], p, nblk, &st.kbar, pages, cand[i]);
            // The keys read: the chosen blocks' (those before the local window) with their bias,
            // then the local window.
            let lo = (p + 1).saturating_sub(local);
            let mut ranges: Vec<(usize, usize, f32)> = sel
                .iter()
                .map(|&(b, bias)| (b * bs, ((b + 1) * bs).min(lo), bias))
                .filter(|r| r.0 < r.1)
                .collect();
            ranges.push((lo, p + 1, 0.0));
            let n: usize = ranges.iter().map(|r| r.1 - r.0).sum();
            let mut s = vec![0f32; n];
            for hh in 0..nh {
                let qh = &q[i * h + hh * hd..i * h + (hh + 1) * hd];
                let mut j = 0;
                for &(a, b, bias) in &ranges {
                    for kp in a..b {
                        s[j] = dot(qh, &st.k[kp * h + hh * hd..kp * h + (hh + 1) * hd]) * scale + bias;
                        j += 1;
                    }
                }
                softmax_inplace(&mut s);
                let oh = &mut o[hh * hd..(hh + 1) * hd];
                oh.fill(0.0);
                let mut j = 0;
                for &(a, b, _) in &ranges {
                    for kp in a..b {
                        axpy(oh, s[j], &st.v[kp * h + hh * hd..kp * h + (hh + 1) * hd]);
                        j += 1;
                    }
                }
            }
        });
        let att = out.to_vec();
        linear(&layer.o, h, h, &att, t, out);
    }
}

impl crate::generate::TextLm for OcrModel<'_> {
    type Sess = OcrSession;
    fn new_session(&self) -> OcrSession {
        OcrModel::new_session(self)
    }
    fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }
    fn n_layers(&self) -> usize {
        self.cfg.num_hidden_layers
    }
    fn session_len(&self, sess: &OcrSession) -> usize {
        sess.pos
    }
    fn forward_hidden_mm(&self, sess: &mut OcrSession, tokens: &[u32], mm: &[(usize, &[f32])]) -> Vec<f32> {
        OcrModel::forward_hidden_mm(self, sess, tokens, mm)
    }
    fn logits(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        OcrModel::logits(self, hidden, t)
    }
    /// The prompt in one pass (see the module docs).
    fn prefill_chunk(&self, _chunk: usize) -> usize {
        usize::MAX
    }
    fn forward(&self, sess: &mut OcrSession, tokens: &[u32]) -> Vec<f32> {
        OcrModel::forward(self, sess, tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::TokenFilter;

    #[test]
    fn python_rounding() {
        assert_eq!(round_half_even(0.5), 0.0);
        assert_eq!(round_half_even(1.5), 2.0);
        assert_eq!(round_half_even(2.5), 2.0);
        assert_eq!(round_half_even(723.4), 723.0);
        assert_eq!(round_half_even(723.6), 724.0);
    }

    /// `dynamic_preprocess` of the checkpoint's code on the same sizes.
    #[test]
    fn tile_grids() {
        assert_eq!(tile_grid(1241, 1754), (2, 3));
        assert_eq!(tile_grid(1670, 1253), (4, 3));
        assert_eq!(tile_grid(1280, 1024), (4, 3));
        assert_eq!(tile_grid(800, 800), (2, 2));
        assert_eq!(tile_grid(641, 641), (2, 2));
        assert_eq!(tile_grid(3000, 3000), (4, 4));
        assert_eq!(tile_grid(4000, 500), (8, 1));
        assert_eq!(tile_grid(600, 2000), (2, 7));
        assert_eq!(tile_grid(1000, 1250), (3, 4));
        assert_eq!(tile_grid(700, 1400), (1, 2));
        assert_eq!(tile_grid(1400, 700), (2, 1));
    }

    #[test]
    fn chat_messages() {
        use crate::chat::ChatFormat;
        let pages = "<PAGE><image><image><PAGE><image><image>";
        let want = format!("{pages}Multi page parsing.");
        assert_eq!(OcrChat.user_turn(pages, false), want);
        assert_eq!(
            OcrChat.user_turn(&format!("{pages}<PAGE><image><PAGE><image>Multi page parsing."), false),
            want
        );
        assert_eq!(OcrChat.user_turn(&format!("{pages}Read it."), false), format!("{pages}Read it."));
        assert_eq!(OcrChat.user_turn("Hello", false), "Hello");
        assert_eq!(OcrChat.first_turn(None, false, &[], "Hello"), format!("{BOS}Hello"));
    }

    #[test]
    fn prompts() {
        assert_eq!(prompt_text("", &[2]).unwrap(), "<PAGE><image><image>Multi page parsing.");
        assert_eq!(prompt_text("Read.", &[1, 1]).unwrap(), "<PAGE><image><PAGE><image>Read.");
        assert_eq!(prompt_text("<image>\nRead.", &[2]).unwrap(), "<image><image>\nRead.");
        assert_eq!(prompt_text("A<image>B", &[1, 1]).unwrap(), "A<PAGE><image><PAGE><image>B");
        assert!(prompt_text("<image><image>", &[1]).is_err());
        assert!(prompt_text("", &[]).is_err());
    }

    /// Values of `dash_attention.entmax15` (float32).
    #[test]
    fn entmax() {
        let p = entmax15(&[2.0, 1.0, 0.5, -1.0e4]);
        let want = [0.814_649_4, 0.162_070_1, 0.023_280_445, 0.0];
        for (a, b) in p.iter().zip(want) {
            assert!((a - b).abs() < 2e-6, "{p:?}");
        }
        assert_eq!(entmax15(&[3.0, -2.0]), vec![1.0, 0.0]);
        let u = entmax15(&[ENTMAX_NEG; 4]);
        assert!(u.iter().all(|v| (v - 0.25).abs() < 1e-6));
    }

    fn guard(expected_pages: usize, max_page_tokens: usize) -> OcrGuard {
        OcrGuard {
            expected_pages,
            page_id: 7,
            eos_id: 1,
            max_page_tokens,
            ngram: 0,
            window: 100,
            generated: Vec::new(),
            markers: 0,
            page_len: 0,
        }
    }

    #[test]
    fn page_guard() {
        let allowed = |g: &mut OcrGuard| {
            let mut l = vec![0f32; 10];
            g.mask(&mut l);
            (0..10u32).filter(|&i| l[i as usize] > f32::NEG_INFINITY).collect::<Vec<_>>()
        };
        let mut g = guard(2, 3);
        // No end before the second page is opened.
        assert!(!allowed(&mut g).contains(&1));
        g.accept(7);
        assert!(!allowed(&mut g).contains(&1));
        // The cap on a page's tokens forces the next marker, then the end.
        (0..3).for_each(|_| g.accept(4));
        assert_eq!(allowed(&mut g), vec![7]);
        g.accept(7);
        assert_eq!(allowed(&mut g).len(), 10);
        (0..3).for_each(|_| g.accept(4));
        assert_eq!(allowed(&mut g), vec![1]);
        // A third marker: only the end.
        g.accept(7);
        assert_eq!(allowed(&mut g), vec![1]);
    }

    #[test]
    fn no_repeat_ngram() {
        let mut g = guard(0, 0).with_ngram(3, 6);
        [5, 6, 2, 9, 5, 6].into_iter().for_each(|t| g.accept(t));
        let mut l = vec![0f32; 10];
        g.mask(&mut l);
        assert!((0..10).all(|i| (l[i] == f32::NEG_INFINITY) == (i == 2)));
        // Outside the window the n-gram is forgotten.
        let mut g = guard(0, 0).with_ngram(3, 4);
        [5, 6, 2, 9, 5, 6].into_iter().for_each(|t| g.accept(t));
        let mut l = vec![0f32; 10];
        g.mask(&mut l);
        assert!(l.iter().all(|v| *v == 0.0));
    }
}
