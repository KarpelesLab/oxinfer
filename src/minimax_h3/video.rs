//! MiniMax-H3: text to video and first / last frame to video (the `t2va` / `fl2va` tasks of the
//! `transformer/` checkpoint), following diffusers' modular `minimax_h3` pipeline:
//!
//! - the prompt verbatim (no chat template, no BOS or other token around it) through the
//!   Qwen3-VL text encoder with the release's tokenizer, which has seven tokens more than Qwen's
//!   ([`EXTRA_TOKENS`]: `<d>` ... `</d>` around a line of dialogue),
//!   conditioning on `hidden_states[50]`; a keyframe prepends `"<Picture i>: "` and its vision
//!   block, whose rows are tagged as video;
//! - `17 n + 5` frames at 24 fps on a canvas whose sides are multiples of 32: `5 n + 2` latent
//!   frames of `h / 16 x w / 16`, 24 channels, in 2x2 patches;
//! - one packed sequence `[text | keyframe rows | audio rows | video rows]`. Rotary positions: text
//!   rows `(i, 0, 0)`; a video frame at time `n_text + 5/3 * (0, 1, 5, 9, 13, 17, 18, ...)` (the
//!   VAE's 1 + 4 + 4 + 4 + 4 frames per 5 latents) with the aspect-normalized `(y, x)` grid of
//!   H3-Image; a first keyframe at the first frame's time, a last one at `n_text + span - 5/3` (`span`:
//!   the sum of every latent frame's `5/3 * (1 | 4)`); the stereo soundtrack's latents
//!   (40 per second and channel, channel-major) at time `n_text + k`, pinned to the left (first
//!   channel) and right (second channel) ends of the width grid;
//! - two rectified-flow schedules stepped in one forward: `linspace(1, 0, steps)` shifted by 12
//!   (video) and 3 (audio), `steps - 1` evaluations, the timestep `1 - sigma`; keyframe rows are
//!   noised to `t = 0.999` and held there (never stepped); Euler as the blend
//!   `x = r x + (1 - r) (x + (1 - t) v)`, `r = sigma_next / sigma`. The checkpoint is
//!   guidance-distilled: one forward per step, no negative prompt;
//! - the video VAE's ViT decoder ([`super::vae::VaeDecoder::decode_video`]), ImageNet
//!   de-normalization, 8-bit rounding.
//!
//! - the audio rows, one sequence per stereo channel, through the audio VAE's decoder
//!   ([`super::audio_vae::AudioDecoder`]): `latents * latents_std + latents_mean`, 800 samples
//!   at 32 kHz per latent, each channel decoded on its own. Without an audio VAE the audio
//!   latents are still denoised (the model is trained on the joint sequence) and the video is
//!   silent.

use super::audio_vae::AudioDecoder;
use super::dit::{Dit, DitConfig, PackedAv, TAG_AUDIO, TAG_IMAGE, TAG_TEXT};
use super::vae::{VaeConfig, VaeDecoder};
use super::video_vae::VideoEncoder;
use crate::clock::Instant;
use crate::safetensors::ModelFiles;
pub use crate::sd::Image;
use crate::sd::ops::Rng;
use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};

pub const FPS: usize = 24;
/// The `t` a keyframe's rows are held at.
pub const KEYFRAME_T: f32 = 0.999;
pub(crate) const AUDIO_CHANNELS: usize = 2;
const PIXEL_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const PIXEL_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Whether `path` is a single-file MiniMax-H3 transformer (ComfyUI's
/// `diffusion_models/minimax_h3_*.safetensors`: the release's tensor names, with the audio stream).
pub fn is_h3_transformer_file(path: &Path) -> bool {
    if !path.is_file() || path.extension().is_none_or(|e| e != "safetensors") {
        return false;
    }
    crate::safetensors::SafeTensors::open(path, false).is_ok_and(|st| {
        [
            "video_patch_proj.weight",
            "audio_patch_proj.weight",
            "blocks.0.attn.qkv_proj.weight",
        ]
        .iter()
        .all(|n| st.info(n).is_some())
    })
}

/// Whether `dir` is a MiniMax-H3 release in diffusers' layout (`MiniMaxH3ModularPipeline`), one of
/// the release's own partitions (`FL2VA/`, `Ref2VA/`: `MiniMaxH3Pipeline`), or a single-file
/// transformer ([`is_h3_transformer_file`]).
pub fn is_h3_video(dir: &Path) -> bool {
    if dir.is_file() {
        return is_h3_transformer_file(dir);
    }
    model_index(dir)
        .and_then(|v| v.get("_class_name").and_then(|c| c.as_str()).map(String::from))
        .is_some_and(|c| c == "MiniMaxH3ModularPipeline" || c == "MiniMaxH3Pipeline")
}

fn model_index(dir: &Path) -> Option<serde_json::Value> {
    serde_json::from_slice(&crate::fsx::read(&dir.join("model_index.json")).ok()?).ok()
}

/// The partition of the release's own layout (`fl2va`, `ref2va`) when `dir` is one: `FL2VA/`
/// holds `transformer/` (the original tensor names), `video_vae/` (its wrapper's `config.json`,
/// the model in `source/`), `audio_vae/` (`config.json` with the latent statistics,
/// `metadata.json` with the constructor's arguments), `text_encoder/` and `tokenizer/`.
pub fn release_partition(dir: &Path) -> Option<String> {
    let v = model_index(dir)?;
    if v.get("_class_name")?.as_str()? != "MiniMaxH3Pipeline" {
        return None;
    }
    Some(
        v.get("_minimax_h3")
            .and_then(|m| m.get("partition"))
            .and_then(|p| p.as_str())
            .unwrap_or("fl2va")
            .to_string(),
    )
}

/// The video VAE configuration of the release's layout (`video_vae/config.json` and
/// `video_vae/source/config.json`, `AutoencoderKLLegacy`) in diffusers' keys
/// (`AutoencoderKLMiniMaxH3`, as `scripts/convert_minimax_h3_to_diffusers.py` maps them).
pub fn release_vae_json(video_vae: &Path) -> Result<serde_json::Value> {
    let read =
        |p: &Path| -> Result<serde_json::Value> { serde_json::from_slice(&crate::fsx::read(p)?).with_context(|| p.display().to_string()) };
    let w = read(&video_vae.join("config.json"))?;
    let src = video_vae.join(w.get("source_path").and_then(|v| v.as_str()).unwrap_or("source"));
    let c = read(&src.join("config.json"))?;
    let list = |v: Option<&serde_json::Value>| -> Vec<u64> {
        v.and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_u64()).collect())
            .unwrap_or_default()
    };
    if c.get("use_vit_decoder").and_then(|v| v.as_bool()) == Some(false) {
        bail!("{}: not the ViT decoder MiniMax-H3 releases", src.display());
    }
    let ch = c.get("ch").and_then(|v| v.as_u64()).unwrap_or(128);
    let channels: Vec<u64> = list(c.get("ch_mult")).iter().map(|m| m * ch).collect();
    let k = c.get("vit_decoder_kwargs").cloned().unwrap_or_default();
    let get = |o: &serde_json::Value, key: &str| o.get(key).cloned().unwrap_or(serde_json::Value::Null);
    let mut o = serde_json::json!({
        "_class_name": "AutoencoderKLMiniMaxH3",
        "latent_channels": get(&w, "latent_channels"),
        "latents_mean": get(&w, "latents_mean"),
        "latents_std": get(&w, "latents_std"),
        "block_out_channels": channels,
        "layers_per_block": c.get("num_res_blocks").and_then(|v| v.as_u64()).unwrap_or(2),
        "spatial_downsample_factors": list(c.get("space_down")),
        "temporal_downsample_factors": list(c.get("time_down")),
        "spatial_padding_mode": c.get("padding_mode").and_then(|v| v.as_str()).unwrap_or("reflect"),
        "decoder_num_layers": get(&k, "num_layers"),
        "decoder_num_attention_heads": get(&k, "heads"),
        "decoder_attention_head_dim": get(&k, "dim_head"),
        "decoder_rope_theta": get(&k, "rope_theta"),
        "decoder_rope_dim_ratio": get(&k, "rope_dim_ratio"),
        "clip_length": get(&w, "vae_clip_length"),
        "token_drop": get(&w, "vae_token_drop"),
        "vae_tile_size": get(&w, "vae_tile_size"),
        "vae_tile_overlap_min": get(&w, "vae_tile_overlap_min"),
    });
    // Unset entries take the loaders' defaults.
    if let Some(m) = o.as_object_mut() {
        m.retain(|_, v| !v.is_null());
    }
    Ok(o)
}

/// Which end of the video a keyframe is anchored to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Anchor {
    First,
    Last,
}

/// `resolve_canvas_size`: `(height, width)` of the canvas for an aspect ratio (short edge 768,
/// at most 768 x 1344 pixels, sides rounded to multiples of 32).
pub fn resolve_canvas(aspect_w: f64, aspect_h: f64, short_edge: usize, max_pixels: usize, multiple: usize) -> Result<(usize, usize)> {
    let ratio = aspect_w / aspect_h;
    if !(0.25..=4.0).contains(&ratio) {
        bail!("MiniMax-H3 supports aspect ratios from 1:4 to 4:1, got {aspect_w}:{aspect_h}");
    }
    let s = short_edge as f64;
    let (mut w, mut h) = if ratio >= 1.0 { (s * ratio, s) } else { (s, s / ratio) };
    let area = w * h;
    if area > max_pixels as f64 {
        let scale = (max_pixels as f64 / area).sqrt();
        (w, h) = (w * scale, h * scale);
    }
    let r = |v: f64| ((v / multiple as f64).round_ties_even() as usize * multiple).max(multiple);
    Ok((r(h), r(w)))
}

/// `align_num_frames`: the next `17 n + 5` (at least 22).
pub fn align_frames(frames: usize, clip: usize, per_clip: usize) -> usize {
    let mut f = frames.max(clip + per_clip);
    while f % clip != per_clip {
        f += 1;
    }
    f
}

/// Latent frames of `17 n + 5` frames: `5 n + 2`.
pub fn latent_frames(frames: usize, clip: usize, per_clip: usize) -> usize {
    (frames - per_clip) / clip * per_clip + 2
}

/// Audio latents per channel covering `frames` frames (40 per second, rounded half to even).
pub fn audio_latents(frames: usize) -> usize {
    (frames as f64 / FPS as f64 * 40.0).round_ties_even() as usize
}

/// `torch.linspace(1, 0, n)` in float32, shifted (`s x / (1 + (s - 1) x)`), consecutive
/// duplicates collapsed: the sigma grid, ending at 0.
pub fn sigmas(n: usize, shift: f32) -> Vec<f32> {
    let step = -1.0f32 / (n as f32 - 1.0);
    let mut out: Vec<f32> = Vec::with_capacity(n);
    for i in 0..n {
        let base = if i < n / 2 {
            1.0 + step * i as f32
        } else {
            0.0 - step * (n - 1 - i) as f32
        };
        let s = shift * base / (1.0 + (shift - 1.0) * base);
        if out.last() != Some(&s) {
            out.push(s);
        }
    }
    out
}

/// numpy's pairwise summation of fewer than 129 float64 values.
fn numpy_sum(v: &[f64]) -> f64 {
    if v.len() < 8 {
        return v.iter().fold(0.0, |a, b| a + b);
    }
    let mut r = [0f64; 8];
    r.copy_from_slice(&v[..8]);
    let whole = v.len() - v.len() % 8;
    for c in v[8..whole].chunks(8) {
        for (a, b) in r.iter_mut().zip(c) {
            *a += b;
        }
    }
    let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for x in &v[whole..] {
        res += x;
    }
    res
}

pub(crate) const FRAME_RESCALE: f64 = 5.0 / 3.0;
pub(crate) const FRAMES_PER_LATENT: [f64; 5] = [1.0, 4.0, 4.0, 4.0, 4.0];

/// The packed sequence `[text | keyframe rows | audio rows | video rows]`.
pub struct Layout {
    pub positions: Vec<[f32; 3]>,
    pub tags: Vec<u8>,
    pub text_index: Vec<usize>,
    /// Keyframe rows first, then the generated rows.
    pub video_index: Vec<usize>,
    pub audio_index: Vec<usize>,
    /// How many leading video rows are keyframe rows.
    pub n_cond: usize,
    /// How many leading audio rows are reference soundtrack rows (`ref2va`): clean, held at
    /// `t = 1` and never stepped.
    pub n_cond_audio: usize,
}

/// `build_packed_sequence`: the layout of `lt` latent frames of `lh x lw` with `n_audio` audio
/// latents per channel; `text_tags` tags the text rows (a keyframe's vision block is tagged as
/// video).
pub fn layout(text_tags: &[u8], lt: usize, lh: usize, lw: usize, n_audio: usize, patch: usize, anchors: &[Anchor]) -> Layout {
    let (gh, gw) = (lh / patch, lw / patch);
    let rpf = gh * gw;
    let nt = text_tags.len();
    let n_cond = anchors.len() * rpf;
    let n_arows = n_audio * AUDIO_CHANNELS;
    let (a0, v0) = (nt + n_cond, nt + n_cond + n_arows);
    let n = v0 + lt * rpf;
    let sa = ((lh * lw) as f64).sqrt();
    let (ys, xs) = (super::axis_grid(lh, patch, sa), super::axis_grid(lw, patch, sa));
    let mut pos = vec![[0f64; 3]; n];
    for (i, p) in pos.iter_mut().take(nt).enumerate() {
        p[0] = i as f64;
    }
    let frame = |pos: &mut [[f64; 3]], start: usize, time: f64| {
        for (i, p) in pos[start..start + rpf].iter_mut().enumerate() {
            *p = [time, ys[i / gw], xs[i % gw]];
        }
    };
    for (k, a) in anchors.iter().enumerate() {
        let time = match a {
            Anchor::First => nt as f64,
            Anchor::Last => {
                let spans: Vec<f64> = (0..lt).map(|i| FRAME_RESCALE * FRAMES_PER_LATENT[i % 5]).collect();
                nt as f64 + numpy_sum(&spans) - FRAME_RESCALE
            }
        };
        frame(&mut pos, nt + k * rpf, time);
    }
    for i in 0..n_arows {
        let w = if i < n_audio { xs[0] } else { xs[gw - 1] };
        pos[a0 + i] = [nt as f64 + (i % n_audio) as f64, 0.0, w];
    }
    let mut cum = 0f64;
    for f in 0..lt {
        frame(&mut pos, v0 + f * rpf, nt as f64 + cum);
        cum += FRAME_RESCALE * FRAMES_PER_LATENT[f % 5];
    }
    let mut tags = vec![TAG_IMAGE; n];
    tags[..nt].copy_from_slice(text_tags);
    tags[a0..v0].fill(TAG_AUDIO);
    Layout {
        positions: pos.iter().map(|p| [p[0] as f32, p[1] as f32, p[2] as f32]).collect(),
        tags,
        text_index: (0..nt).collect(),
        video_index: (nt..a0).chain(v0..n).collect(),
        audio_index: (a0..v0).collect(),
        n_cond,
        n_cond_audio: 0,
    }
}

/// Rows `[C, 1, p, p]` of video latents `[C, t, h, w]`, frame-major then row-major.
pub fn patchify_video(lat: &[f32], c: usize, t: usize, h: usize, w: usize, p: usize) -> Vec<f32> {
    let plane = h * w;
    let mut out = Vec::with_capacity(lat.len());
    let mut frame = vec![0f32; c * plane];
    for f in 0..t {
        for ch in 0..c {
            frame[ch * plane..(ch + 1) * plane].copy_from_slice(&lat[(ch * t + f) * plane..(ch * t + f + 1) * plane]);
        }
        out.extend(super::patchify(&frame, c, h, w, p));
    }
    out
}

/// The inverse of [`patchify_video`].
pub fn unpatchify_video(rows: &[f32], c: usize, t: usize, h: usize, w: usize, p: usize) -> Vec<f32> {
    let plane = h * w;
    let per = rows.len() / t;
    let mut lat = vec![0f32; c * t * plane];
    for f in 0..t {
        let frame = super::unpatchify(&rows[f * per..(f + 1) * per], c, h, w, p);
        for ch in 0..c {
            lat[(ch * t + f) * plane..(ch * t + f + 1) * plane].copy_from_slice(&frame[ch * plane..(ch + 1) * plane]);
        }
    }
    lat
}

/// The latent geometry of one request.
#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    pub lt: usize,
    pub lh: usize,
    pub lw: usize,
    pub n_audio: usize,
}

/// The joint denoising loop over a packed sequence: `video` holds the keyframe rows (already at
/// `t = 0.999`) then the noise rows, `audio` the audio noise rows; `sigmas` the video and audio
/// grids ([`sigmas`]). Returns the final video rows (keyframe rows included) and audio rows.
#[allow(clippy::too_many_arguments)]
pub fn denoise(
    dit: &Dit,
    text: &[f32],
    text_tags: &[u8],
    video: Vec<f32>,
    audio: Vec<f32>,
    g: Geometry,
    anchors: &[Anchor],
    sig: (&[f32], &[f32]),
    progress: &mut dyn FnMut(usize, usize),
) -> Result<(Vec<f32>, Vec<f32>)> {
    let lay = layout(text_tags, g.lt, g.lh, g.lw, g.n_audio, dit.cfg.patch[1], anchors);
    denoise_layout(dit, text, &lay, video, audio, sig, progress)
}

/// [`denoise`] over any packed layout: `video` holds `lay.n_cond` conditioning rows (already at
/// `t = 0.999`) then the noise rows, `audio` `lay.n_cond_audio` clean reference rows then the
/// noise rows. Returns the final rows, conditioning included.
pub fn denoise_layout(
    dit: &Dit,
    text: &[f32],
    lay: &Layout,
    video: Vec<f32>,
    audio: Vec<f32>,
    sig: (&[f32], &[f32]),
    progress: &mut dyn FnMut(usize, usize),
) -> Result<(Vec<f32>, Vec<f32>)> {
    denoise_cancellable(dit, text, lay, video, audio, sig, progress, None)
}

/// [`denoise_layout`] that stops with an error between two steps once `cancel` is set.
#[allow(clippy::too_many_arguments)]
pub fn denoise_cancellable(
    dit: &Dit,
    text: &[f32],
    lay: &Layout,
    mut video: Vec<f32>,
    mut audio: Vec<f32>,
    sig: (&[f32], &[f32]),
    progress: &mut dyn FnMut(usize, usize),
    cancel: Option<&std::sync::atomic::AtomicBool>,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let cancelled = || cancel.is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed));
    let cfg = &dit.cfg;
    let emb = dit.embed_text(text);
    let (pd, ac) = (cfg.patch_dim(), cfg.audio_channels);
    let (sv, sa) = sig;
    let steps = sv.len().min(sa.len()) - 1;
    let cond_idx = &lay.video_index[..lay.n_cond];
    let cond_audio = &lay.audio_index[..lay.n_cond_audio];
    for i in 0..steps {
        if cancelled() {
            bail!("cancelled");
        }
        let (tv, ta) = (1.0 - sv[i], 1.0 - sa[i]);
        // The distinct timesteps, sorted, and every row's index into them (text rows: the video's).
        let tc = tv.max(KEYFRAME_T);
        let mut ts = vec![tv, ta];
        if lay.n_cond > 0 {
            ts.push(tc);
        }
        if lay.n_cond_audio > 0 {
            ts.push(1.0);
        }
        ts.sort_by(|a, b| a.total_cmp(b));
        ts.dedup();
        let at = |t: f32| ts.iter().position(|&v| v == t).unwrap();
        let mut tidx = vec![at(tv); lay.positions.len()];
        cond_idx.iter().for_each(|&r| tidx[r] = at(tc));
        lay.audio_index.iter().for_each(|&r| tidx[r] = at(ta));
        cond_audio.iter().for_each(|&r| tidx[r] = at(1.0));
        let temb: Vec<f32> = ts.iter().flat_map(|&t| dit.time_embed(t)).collect();
        let (v, a) = dit.forward_av(&PackedAv {
            text_embedded: &emb,
            video: &video,
            audio: &audio,
            text_index: &lay.text_index,
            video_index: &lay.video_index,
            audio_index: &lay.audio_index,
            positions: &lay.positions,
            temb: &temb,
            timestep_index: &tidx,
            tags: &lay.tags,
        })?;
        // x0 from the timestep's sigma, the Euler ratio from the grid's.
        let step = |x: &mut [f32], v: &[f32], t: f32, s: f32, s_next: f32| {
            let (st, r) = (1.0 - t, s_next / s);
            for (xv, &vv) in x.iter_mut().zip(v) {
                *xv = r * *xv + (1.0 - r) * (*xv + st * vv);
            }
        };
        let (c0, a0) = (lay.n_cond * pd, lay.n_cond_audio * ac);
        step(&mut video[c0..], &v[c0..], tv, sv[i], sv[i + 1]);
        step(&mut audio[a0..], &a[a0..], ta, sa[i], sa[i + 1]);
        debug_assert_eq!(audio.len() % ac, 0);
        progress(i + 1, steps);
    }
    Ok((video, audio))
}

const LATENTS_MAGIC: &[u8; 4] = b"H3LT";

/// Denoised rows as a file: `H3LT`, then `lt`, `lh`, `lw`, `n_audio` and the two value counts as
/// little-endian u32, then the video rows (generated rows only) and the audio rows in float32.
pub fn write_latents(path: &Path, g: Geometry, video: &[f32], audio: &[f32]) -> Result<()> {
    let mut b = LATENTS_MAGIC.to_vec();
    for x in [g.lt, g.lh, g.lw, g.n_audio, video.len(), audio.len()] {
        b.extend_from_slice(&(x as u32).to_le_bytes());
    }
    video.iter().chain(audio).for_each(|v| b.extend_from_slice(&v.to_le_bytes()));
    std::fs::write(path, b).with_context(|| format!("write {}", path.display()))
}

/// The video and audio rows of [`write_latents`], for the geometry `g`.
pub fn read_latents(path: &Path, g: Geometry) -> Result<(Vec<f32>, Vec<f32>)> {
    let b = std::fs::read(path).with_context(|| path.display().to_string())?;
    if b.len() < 28 || &b[..4] != LATENTS_MAGIC {
        bail!("{}: not a MiniMax-H3 latent file", path.display());
    }
    let h: Vec<usize> = b[4..28]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c) as usize)
        .collect();
    if h[..4] != [g.lt, g.lh, g.lw, g.n_audio] {
        bail!(
            "{}: latents of {} frames of {}x{} with {} audio latents, the request has {} of {}x{} with {}",
            path.display(),
            h[0],
            h[1],
            h[2],
            h[3],
            g.lt,
            g.lh,
            g.lw,
            g.n_audio
        );
    }
    let vals: Vec<f32> = b[28..].as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
    if vals.len() != h[4] + h[5] {
        bail!("{}: truncated", path.display());
    }
    Ok((vals[..h[4]].to_vec(), vals[h[4]..].to_vec()))
}

/// What [`VideoPipeline::generate`] needs.
pub struct VideoParams {
    pub prompt: String,
    /// `(width, height)`; none: 16:9 at a 768 short edge, or the first keyframe's aspect ratio.
    pub size: Option<(usize, usize)>,
    /// Frames at 24 fps, rounded up to `17 n + 5` (the release generates 5 to 15 seconds).
    pub frames: usize,
    /// Points of the sigma grid (the reference's `num_inference_steps`): `steps - 1` forwards.
    pub steps: usize,
    pub seed: u64,
    pub first: Option<Image>,
    pub last: Option<Image>,
    /// The short edge and pixel budget of a canvas resolved from an aspect ratio.
    pub short_edge: usize,
    pub max_pixels: usize,
    /// Keeps the denoised video and audio rows in this file ([`write_latents`]): written when
    /// missing, read instead of denoising when present (the two VAEs can then be run again, or
    /// other ones, on the same rows).
    pub latents: Option<PathBuf>,
    /// Set from another thread to stop the generation between two denoising steps (it then
    /// returns an error saying "cancelled").
    pub cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

impl VideoParams {
    /// Whether [`VideoParams::cancel`] is set.
    pub fn cancelled(&self) -> bool {
        self.cancel.as_ref().is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed))
    }

    pub fn new(prompt: &str) -> Self {
        Self {
            prompt: prompt.to_string(),
            size: None,
            frames: 124,
            steps: 50,
            seed: 0,
            first: None,
            last: None,
            short_edge: 768,
            max_pixels: 768 * 1344,
            latents: None,
            cancel: None,
        }
    }
}

/// The container and codecs a video is written with (`minimax_h3::encode`, the `media` feature).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VideoFormat {
    /// H.264 and AAC in MP4: what every browser and player takes (the default).
    Mp4,
    /// VP9 and Opus in WebM (Matroska).
    Webm,
    /// Motion JPEG and 16-bit PCM in AVI.
    Avi,
}

impl VideoFormat {
    /// From a name or a file extension: `mp4`, `webm`, `avi`.
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s.to_ascii_lowercase().as_str() {
            "mp4" | "m4v" | "h264" => Self::Mp4,
            "webm" | "vp9" => Self::Webm,
            "avi" | "mjpeg" => Self::Avi,
            _ => return None,
        })
    }

    pub fn extension(self) -> &'static str {
        match self {
            Self::Mp4 => "mp4",
            Self::Webm => "webm",
            Self::Avi => "avi",
        }
    }

    pub fn mime(self) -> &'static str {
        match self {
            Self::Mp4 => "video/mp4",
            Self::Webm => "video/webm",
            Self::Avi => "video/x-msvideo",
        }
    }
}

/// Generated frames (RGB8, row-major) at [`FPS`] and their soundtrack.
pub struct Video {
    pub width: usize,
    pub height: usize,
    pub frames: Vec<Vec<u8>>,
    /// `None` without an audio VAE (no `audio_vae/` in the release folder and none named).
    pub audio: Option<Soundtrack>,
}

/// A generated soundtrack: one waveform per channel (left, right), samples in `[-1, 1]`.
pub struct Soundtrack {
    pub sample_rate: u32,
    pub channels: Vec<Vec<f32>>,
}

impl Soundtrack {
    /// Samples per channel.
    pub fn len(&self) -> usize {
        self.channels.first().map_or(0, Vec::len)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The samples `range` of every channel, interleaved.
    pub fn interleaved(&self, range: std::ops::Range<usize>) -> Vec<f32> {
        range.flat_map(|i| self.channels.iter().map(move |c| c[i])).collect()
    }

    /// A 16-bit PCM WAV file.
    pub fn to_wav(&self) -> Vec<u8> {
        crate::tts::wav_bytes_channels(&self.interleaved(0..self.len()), self.sample_rate, self.channels.len() as u16)
    }
}

/// The conditioning of one request: `[n, text_dim]` rows and their modality tags.
pub struct Conditioning {
    pub text: Vec<f32>,
    pub tags: Vec<u8>,
}

/// Where the parts of a [`VideoPipeline`] come from.
#[derive(Default)]
pub struct Sources {
    /// A release folder in diffusers' layout: its `transformer/`, `vae/`, `tokenizer/` and the two
    /// schedulers' shifts. None: every part is given below, the shifts are 12 and 3.
    pub dir: Option<PathBuf>,
    /// The transformer instead of `dir/transformer`: a folder, or a single file in ComfyUI's
    /// layout (bf16, "pruned", FP8 scaled: see [`super::dit`]).
    pub transformer: Option<PathBuf>,
    /// The video VAE instead of `dir/vae`: a folder, or ComfyUI's single file.
    pub vae: Option<PathBuf>,
    /// The audio VAE instead of `dir/audio_vae`: a folder, or ComfyUI's single file. Without
    /// either the video has no soundtrack.
    pub audio_vae: Option<PathBuf>,
    /// The Qwen3-VL-32B text encoder: a folder, or a GGUF file (text-only prompts).
    pub text_encoder: PathBuf,
    /// The precision of the text encoder's weights (float checkpoints are quantized at load).
    pub text_quant: Option<crate::model::Quant>,
}

/// The tensors of a checkpoint folder or of a single safetensors file, mapped.
fn open_part(path: &Path) -> Result<&'static ModelFiles> {
    let files = if path.is_file() {
        ModelFiles::open_file(path, false)
    } else {
        ModelFiles::open(path, false)
    };
    Ok(Box::leak(Box::new(files.with_context(|| path.display().to_string())?)))
}

pub struct VideoPipeline {
    pub dir: PathBuf,
    pub text_encoder_dir: PathBuf,
    pub text_quant: crate::model::Quant,
    pub dit: Dit<'static>,
    pub vae: VaeDecoder<'static>,
    vae_files: &'static ModelFiles,
    /// The audio VAE's decoder, when the folder has `audio_vae/` or one was named.
    pub audio_vae: Option<AudioDecoder>,
    /// Where the audio VAE is (its encoder is loaded on demand: only `ref2va` needs it).
    pub audio_vae_path: Option<PathBuf>,
    /// The release partition the folder is (`fl2va`, `ref2va`), when it says.
    pub partition: Option<String>,
    /// The VAE's configuration in diffusers' keys (the encoder reads its geometry).
    vae_json: serde_json::Value,
    /// The sigma shifts of the video and audio schedules.
    pub shifts: (f32, f32),
}

fn shift_of(dir: &Path, default: f32) -> f32 {
    crate::fsx::read(&dir.join("scheduler_config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .and_then(|v| v.get("shift").and_then(|s| s.as_f64()))
        .map_or(default, |s| s as f32)
}

/// The conditioning of `prompt` with `keyframes` (canvas-sized): `"<Picture i>: "` and a vision
/// block per keyframe, then the prompt, after the 50th layer of the text encoder in
/// `text_encoder` (`model`: the release folder, for its tokenizer).
pub fn encode_prompt(model: &Path, text_encoder: &Path, prompt: &str, keyframes: &[&Image]) -> Result<Conditioning> {
    encode_prompt_quant(model, text_encoder, crate::model::Quant::Bf16, prompt, keyframes)
}

/// [`encode_prompt`] with the text encoder's weights at `quant` (a float checkpoint is quantized
/// at load; a GGUF file is used in its own format). `text_encoder` is a folder or a GGUF file of
/// Qwen3-VL-32B-Instruct, which the release's `text_encoder/` is (the same files, by their
/// SHA-256); a GGUF file brings its own tokenizer and no vision tower, so no keyframes. One of
/// ComfyUI's single files (`qwen3vl_32b_minimax_h3_*.safetensors`: the first 50 layers, in bf16,
/// int8 ConvRot or NVFP4, see [`crate::comfy_quant`]) is read for text-only prompts too, with the
/// release's `tokenizer.json` in `model`'s `tokenizer/`, in a `tokenizer/` folder beside the file
/// or beside the file itself.
pub fn encode_prompt_quant(
    model: &Path,
    text_encoder: &Path,
    quant: crate::model::Quant,
    prompt: &str,
    keyframes: &[&Image],
) -> Result<Conditioning> {
    let vl = load_text_encoder(model, text_encoder, quant, !keyframes.is_empty())?;
    let (mut ids, mut tags, mut images) = (Vec::new(), Vec::new(), Vec::new());
    for (i, k) in keyframes.iter().enumerate() {
        let im = vl.encode_image(&k.rgb, k.width, k.height, 0)?;
        let label = vl.tokenize(&format!("<Picture {}>: ", i + 1))?;
        let block = vl.tokenize(&im.placeholder())?;
        tags.extend(std::iter::repeat_n(TAG_TEXT, label.len()));
        tags.extend(std::iter::repeat_n(TAG_IMAGE, block.len()));
        ids.extend(label);
        ids.extend(block);
        images.push(im);
    }
    let pids = vl.tokenize(prompt)?;
    if pids.is_empty() {
        bail!("empty prompt");
    }
    tags.extend(std::iter::repeat_n(TAG_TEXT, pids.len()));
    ids.extend(pids);
    let mut hs = vl.hidden_states_tokens(ids, &images, None, &[TEXT_LAYER - 1])?;
    let (_, text) = hs.layers.pop().context("layer 50 not captured")?;
    Ok(Conditioning { text, tags })
}

/// The conditioning is transformers' `hidden_states[TEXT_LAYER]`: the residual stream after this
/// many layers of the text encoder.
pub const TEXT_LAYER: usize = 50;

/// The tokens MiniMax's `tokenizer/tokenizer_config.json` adds to Qwen's vocabulary
/// (`additional_special_tokens`): transformers numbers them after the last id of `tokenizer.json`
/// (151668, `</think>`), and ComfyUI's `MINIMAX_EXTRA_TOKENS` names the same ids. Neither the
/// release's `tokenizer.json` nor a GGUF of Qwen3-VL-32B-Instruct holds them (the GGUF's entries
/// there are unused `[PAD151669]`, ...); without them the `<d>` and `</d>` around a line of
/// dialogue are three tokens each.
pub const EXTRA_TOKENS: [(&str, u32); 7] = [
    ("<d>", 151669),
    ("</d>", 151670),
    ("<|cutoff|>", 151671),
    ("<|lyrics_start|>", 151672),
    ("<|lyrics_end|>", 151673),
    ("<|caption_start|>", 151674),
    ("<|caption_end|>", 151675),
];

/// The text encoder of [`encode_prompt_quant`] with the release's tokenizer: the one found for
/// `text_encoder` (see there) with MiniMax's [`EXTRA_TOKENS`]. `vision`: with the vision tower
/// (keyframes).
pub fn load_text_encoder(model: &Path, text_encoder: &Path, quant: crate::model::Quant, vision: bool) -> Result<crate::qwen_vl::QwenVl> {
    use crate::qwen_vl::{QwenVl, QwenVlOptions};
    let parent = text_encoder.parent().map(Path::to_path_buf).unwrap_or_default();
    if crate::gguf::is_gguf(text_encoder) && vision {
        bail!("a GGUF text encoder has no vision tower: keyframes need the safetensors text encoder");
    }
    if text_encoder.is_file() && vision {
        bail!("the vision tower of a single-file text encoder is not read: keyframes need the text encoder's folder");
    }
    let opts = QwenVlOptions {
        quant,
        mmap: true,
        vision,
        tokenizer_dir: [
            model.join("tokenizer"),
            parent.join("tokenizer"),
            text_encoder.to_path_buf(),
            // Beside a single-file text encoder.
            parent.clone(),
        ]
        .into_iter()
        .find(|d| d.join("tokenizer.json").exists()),
        ..Default::default()
    };
    let mut vl = QwenVl::load(text_encoder, &opts)?;
    // ComfyUI's single files stop after the 50th layer.
    if vl.model().cfg.num_hidden_layers < TEXT_LAYER {
        bail!(
            "the text encoder has {} layers; MiniMax-H3 conditions on its {TEXT_LAYER}th",
            vl.model().cfg.num_hidden_layers
        );
    }
    vl.tokenizer.add_special_tokens_at(&EXTRA_TOKENS)?;
    Ok(vl)
}

impl VideoPipeline {
    /// Loads the DiT (`transformer/`) and the video VAE (`vae/`) of the release in `dir` (bf16,
    /// mapped) and the audio VAE's decoder when `audio_vae/` is there; the text encoder from
    /// `text_encoder` when it is used.
    pub fn load(dir: &Path, text_encoder: &Path) -> Result<Self> {
        let src = Sources {
            dir: Some(dir.to_path_buf()),
            text_encoder: text_encoder.to_path_buf(),
            ..Default::default()
        };
        Self::load_from(&src)
    }

    /// Applies LoRA files to the transformer ([`super::lora`]); returns one report line per file.
    pub fn apply_loras(&mut self, specs: &[super::lora::LoraSpec]) -> Result<Vec<String>> {
        super::lora::apply(&mut self.dit, specs)
    }

    /// Loads the DiT and the video VAE from `src` (mapped, in their stored formats) and the audio
    /// VAE's decoder when there is one.
    pub fn load_from(src: &Sources) -> Result<Self> {
        let part = |given: &Option<PathBuf>, sub: &str| -> Result<PathBuf> {
            match (given, &src.dir) {
                (Some(p), _) => Ok(p.clone()),
                (None, Some(d)) => Ok(d.join(sub)),
                (None, None) => bail!("MiniMax-H3: no {sub} given (a single-file transformer needs the VAE named too)"),
            }
        };
        let partition = src.dir.as_deref().and_then(release_partition);
        if let Some(p) = partition.as_deref()
            && p != "fl2va"
            && p != "ref2va"
        {
            bail!("MiniMax-H3: the {p} partition is not supported (the release's FL2VA/ and Ref2VA/ are)");
        }
        let release = partition.is_some();
        let tpath = part(&src.transformer, "transformer")?;
        let dit_files = open_part(&tpath)?;
        let cfg = if tpath.is_file() {
            DitConfig::from_files(dit_files)
                .with_context(|| format!("{}: not a MiniMax-H3 transformer in ComfyUI's layout", tpath.display()))?
        } else {
            DitConfig::load(&tpath)?
        };
        let dit = Dit::load(cfg, dit_files, None)?;
        if !dit.has_audio() {
            bail!("{}: the transformer has no audio stream (not MiniMax-H3)", tpath.display());
        }
        let vpath = match (&src.vae, &src.dir) {
            (None, Some(d)) if release => {
                let w: serde_json::Value = serde_json::from_slice(&crate::fsx::read(&d.join("video_vae/config.json"))?)?;
                d.join("video_vae")
                    .join(w.get("source_path").and_then(|v| v.as_str()).unwrap_or("source"))
            }
            _ => part(&src.vae, "vae")?,
        };
        let vae_files = open_part(&vpath)?;
        let vae_json: serde_json::Value = if release && src.vae.is_none() {
            release_vae_json(vpath.parent().context("video VAE folder")?)?
        } else if vpath.is_file() {
            VaeConfig::comfy_json(vae_files).with_context(|| {
                format!(
                    "{}: not ComfyUI's single-file MiniMax-H3 video VAE (no minimax_h3_video_vae header entry)",
                    vpath.display()
                )
            })??
        } else {
            serde_json::from_slice(&crate::fsx::read(&vpath.join("config.json"))?)?
        };
        let vae = VaeDecoder::load(VaeConfig::from_json(&vae_json)?, vae_files)?;
        if vae.cfg.axes != 3 {
            bail!("{}: not a video VAE", vpath.display());
        }
        // The audio VAE: the one named, or the release folder's; without one there is no soundtrack.
        let apath = match (&src.audio_vae, &src.dir) {
            (Some(p), _) => Some(p.clone()),
            (None, Some(d)) => Some(d.join("audio_vae")).filter(|a| a.join("config.json").exists()),
            (None, None) => None,
        };
        let audio_vae = match &apath {
            Some(apath) => {
                let a = AudioDecoder::load(apath).with_context(|| format!("audio VAE in {}", apath.display()))?;
                if a.cfg.latent_channels != dit.cfg.audio_channels {
                    bail!(
                        "{}: {} latent channels, the transformer's audio rows have {}",
                        apath.display(),
                        a.cfg.latent_channels,
                        dit.cfg.audio_channels
                    );
                }
                Some(a)
            }
            None => None,
        };
        let shift = |sub: &str, key: &str, default: f32| {
            src.dir.as_ref().map_or(default, |d| {
                // The release's layout states them in model_index.json.
                model_index(d)
                    .and_then(|v| v.get("_minimax_h3")?.get("sigma_shift_scales")?.get(key)?.as_f64())
                    .map_or_else(|| shift_of(&d.join(sub), default), |s| s as f32)
            })
        };
        Ok(Self {
            dir: src
                .dir
                .clone()
                .or_else(|| src.text_encoder.parent().map(Path::to_path_buf))
                .unwrap_or_default(),
            text_encoder_dir: src.text_encoder.clone(),
            text_quant: src.text_quant.unwrap_or(crate::model::Quant::Bf16),
            dit,
            vae,
            vae_files,
            audio_vae,
            audio_vae_path: apath,
            partition,
            vae_json,
            shifts: (shift("scheduler", "video", 12.0), shift("audio_scheduler", "audio", 3.0)),
        })
    }

    /// The canvas `(height, width)`, the aligned frame count and the latent geometry of `p`.
    pub fn geometry(&self, p: &VideoParams) -> Result<(usize, usize, usize, Geometry)> {
        let multiple = self.vae.cfg.patch * self.dit.cfg.patch[1];
        let (h, w) = match (p.size, &p.first, &p.last) {
            (Some((w, h)), _, _) => (h, w),
            (None, Some(k), _) | (None, None, Some(k)) => {
                resolve_canvas(k.width as f64, k.height as f64, p.short_edge, p.max_pixels, multiple)?
            }
            (None, None, None) => resolve_canvas(16.0, 9.0, p.short_edge, p.max_pixels, multiple)?,
        };
        if h % multiple != 0 || w % multiple != 0 || h == 0 || w == 0 {
            bail!("the video size must be multiples of {multiple}, got {w}x{h}");
        }
        let (tcs, _, _, _) = self.vae.chunking();
        let frames = align_frames(p.frames, self.vae.cfg.clip_length, tcs);
        let g = Geometry {
            lt: latent_frames(frames, self.vae.cfg.clip_length, tcs),
            lh: h / self.vae.cfg.patch,
            lw: w / self.vae.cfg.patch,
            n_audio: audio_latents(frames),
        };
        Ok((h, w, frames, g))
    }

    /// The keyframes of `p` on a `w x h` canvas with their anchors: the first given one stretched
    /// (Lanczos), a second one cover-cropped.
    pub fn keyframes(p: &VideoParams, w: usize, h: usize) -> Vec<(Anchor, Image)> {
        let mut out = Vec::new();
        for (a, k) in [(Anchor::First, &p.first), (Anchor::Last, &p.last)] {
            let Some(k) = k else { continue };
            let rgb = if (k.width, k.height) == (w, h) {
                k.rgb.clone()
            } else if out.is_empty() {
                crate::resample::resize_lanczos_rgb8(&k.rgb, k.width, k.height, w, h)
            } else {
                let scale = (w as f64 / k.width as f64).max(h as f64 / k.height as f64);
                let rw = w.max((k.width as f64 * scale).round_ties_even() as usize);
                let rh = h.max((k.height as f64 * scale).round_ties_even() as usize);
                let (left, top) = ((rw - w) / 2, (rh - h) / 2);
                let r = crate::resample::resize_lanczos_rgb8(&k.rgb, k.width, k.height, rw, rh);
                let mut o = Vec::with_capacity(w * h * 3);
                for y in 0..h {
                    o.extend_from_slice(&r[((top + y) * rw + left) * 3..((top + y) * rw + left + w) * 3]);
                }
                o
            };
            out.push((a, Image { width: w, height: h, rgb }));
        }
        out
    }

    /// [`encode_prompt`] with this release's text encoder.
    pub fn encode_prompt(&self, prompt: &str, keyframes: &[&Image]) -> Result<Conditioning> {
        encode_prompt_quant(&self.dir, &self.text_encoder_dir, self.text_quant, prompt, keyframes)
    }

    /// A keyframe's conditioning latents `[C, 1, lh, lw]` (normalized): the posterior sampled with
    /// `noise`, rounded through float16.
    pub fn encode_keyframe(&self, enc: &VideoEncoder, k: &Image, noise: &[f32]) -> Result<Vec<f32>> {
        let plane = k.width * k.height;
        let mut px = vec![0f32; 3 * plane];
        for i in 0..plane {
            for ch in 0..3 {
                px[ch * plane + i] = (k.rgb[i * 3 + ch] as f32 / 255.0 - PIXEL_MEAN[ch]) / PIXEL_STD[ch];
            }
        }
        let (m, t, lh, lw) = enc.encode(&px, 1, k.height, k.width)?;
        let c = self.vae.cfg.latent_channels;
        let n = c * t * lh * lw;
        if noise.len() != n {
            bail!("keyframe: {} noise values for {n} latents", noise.len());
        }
        Ok((0..n)
            .map(|i| {
                let std = (0.5 * m[n + i].clamp(-30.0, 20.0)).exp();
                let z = half::f16::from_f32(m[i] + std * noise[i]).to_f32();
                let ch = i / (t * lh * lw);
                (z - self.vae.cfg.latents_mean[ch]) / self.vae.cfg.latents_std[ch]
            })
            .collect())
    }

    /// The conditioning latents `([C, t, lh, lw], t, lh, lw)` (normalized) of RGB8 frames of
    /// `w`x`h` (one: a keyframe or an image reference; `17 n + 5`: a video reference): the
    /// posterior sampled under its own generator (seed 42), rounded through float16.
    pub fn encode_frames(&self, enc: &VideoEncoder, frames: &[&[u8]], w: usize, h: usize) -> Result<(Vec<f32>, usize, usize, usize)> {
        let (plane, nf) = (w * h, frames.len());
        let mut px = vec![0f32; 3 * nf * plane];
        for (f, rgb) in frames.iter().enumerate() {
            for i in 0..plane {
                for ch in 0..3 {
                    px[(ch * nf + f) * plane + i] = (rgb[i * 3 + ch] as f32 / 255.0 - PIXEL_MEAN[ch]) / PIXEL_STD[ch];
                }
            }
        }
        let (m, t, lh, lw) = enc.encode(&px, nf, h, w)?;
        let n = self.vae.cfg.latent_channels * t * lh * lw;
        let noise = Rng::new(42).normals(n);
        let lat = (0..n)
            .map(|i| {
                let std = (0.5 * m[n + i].clamp(-30.0, 20.0)).exp();
                let z = half::f16::from_f32(m[i] + std * noise[i]).to_f32();
                let ch = i / (t * lh * lw);
                (z - self.vae.cfg.latents_mean[ch]) / self.vae.cfg.latents_std[ch]
            })
            .collect();
        Ok((lat, t, lh, lw))
    }

    /// The audio VAE's encoder (loaded on demand: only `ref2va`'s soundtracks need it).
    pub fn audio_encoder(&self) -> Result<super::audio_vae::AudioEncoder> {
        let p = self
            .audio_vae_path
            .as_ref()
            .context("reference audio needs the audio VAE (audio_vae/ or --audio-vae)")?;
        super::audio_vae::AudioEncoder::load(p).with_context(|| format!("audio VAE encoder in {}", p.display()))
    }

    /// The video encoder (loaded on demand: only keyframes need it).
    pub fn encoder(&self) -> Result<VideoEncoder> {
        VideoEncoder::load_json(&self.vae_json, &self.vae.cfg, self.vae_files)
    }

    /// Decodes video rows (generated rows only) into RGB8 frames.
    pub fn decode(&self, rows: &[f32], g: Geometry) -> Result<Video> {
        let c = self.vae.cfg.latent_channels;
        let lat = unpatchify_video(rows, c, g.lt, g.lh, g.lw, self.dit.cfg.patch[1]);
        let (px, nf) = self.vae.decode_video(&lat, g.lt, g.lh, g.lw)?;
        let (h, w) = (g.lh * self.vae.cfg.patch, g.lw * self.vae.cfg.patch);
        let plane = h * w;
        let frames = (0..nf)
            .map(|f| {
                let mut rgb = vec![0u8; plane * 3];
                for ch in 0..3 {
                    let src = &px[(ch * nf + f) * plane..(ch * nf + f + 1) * plane];
                    for (i, &v) in src.iter().enumerate() {
                        rgb[i * 3 + ch] = ((v * PIXEL_STD[ch] + PIXEL_MEAN[ch]).clamp(0.0, 1.0) * 255.0).round_ties_even() as u8;
                    }
                }
                rgb
            })
            .collect();
        Ok(Video {
            width: w,
            height: h,
            frames,
            audio: None,
        })
    }

    /// Decodes the audio rows (`[2 n, channels]`, channel-major) into the stereo soundtrack;
    /// `None` without an audio VAE.
    pub fn decode_audio(&self, rows: &[f32]) -> Result<Option<Soundtrack>> {
        let Some(vae) = &self.audio_vae else {
            return Ok(None);
        };
        Ok(Some(Soundtrack {
            sample_rate: vae.cfg.sampling_rate,
            channels: vae.decode_rows(rows, AUDIO_CHANNELS)?,
        }))
    }

    /// Text (and keyframes) to video. `cond`: a conditioning computed earlier
    /// ([`VideoPipeline::encode_prompt`]) instead of running the text encoder.
    pub fn generate(&self, p: &VideoParams, cond: Option<Conditioning>, progress: &mut dyn FnMut(usize, usize)) -> Result<Video> {
        if self.partition.as_deref() == Some("ref2va") {
            bail!("the release's Ref2VA/ generates from references only (--ref); text and keyframes need FL2VA/");
        }
        if p.steps < 2 {
            bail!("--steps must be at least 2 (the sigma grid includes its end)");
        }
        let (h, w, frames, g) = self.geometry(p)?;
        if let Some(f) = p.latents.as_ref().filter(|f| f.exists()) {
            let (video, audio) = read_latents(f, g)?;
            let t2 = Instant::now();
            let mut out = self.decode(&video, g)?;
            let t_vae = t2.elapsed().as_secs_f64();
            let t3 = Instant::now();
            out.audio = self.decode_audio(&audio)?;
            eprintln!(
                "[MiniMax-H3: {w}x{h}, {frames} frames ({} latent) from the rows in {}, VAE {t_vae:.1}s, audio VAE {:.1}s]",
                g.lt,
                f.display(),
                t3.elapsed().as_secs_f64()
            );
            return Ok(out);
        }
        let keys = Self::keyframes(p, w, h);
        let anchors: Vec<Anchor> = keys.iter().map(|k| k.0).collect();
        let t0 = Instant::now();
        let cond = match cond {
            Some(c) => c,
            None => self.encode_prompt(&p.prompt, &keys.iter().map(|k| &k.1).collect::<Vec<_>>())?,
        };
        let t_enc = t0.elapsed().as_secs_f64();
        let cfg = &self.dit.cfg;
        let (c, patch) = (self.vae.cfg.latent_channels, cfg.patch[1]);
        let plane = g.lh * g.lw;
        // One generator: the posterior and conditioning noise of each keyframe, the video noise,
        // the audio noise (the reference's order; its posterior uses a generator of its own).
        let mut rng = Rng::new(p.seed);
        let mut video = Vec::new();
        if !keys.is_empty() {
            let enc = self.encoder()?;
            for (_, k) in &keys {
                let z = self.encode_keyframe(&enc, k, &Rng::new(42).normals(c * plane))?;
                let noise = rng.normals(c * plane);
                let noised: Vec<f32> = z
                    .iter()
                    .zip(&noise)
                    .map(|(&x, &n)| KEYFRAME_T * x + (1.0 - KEYFRAME_T) * n)
                    .collect();
                video.extend(patchify_video(&noised, c, 1, g.lh, g.lw, patch));
            }
        }
        let n_cond_vals = video.len();
        video.extend(patchify_video(&rng.normals(c * g.lt * plane), c, g.lt, g.lh, g.lw, patch));
        let audio = rng.normals(g.n_audio * AUDIO_CHANNELS * cfg.audio_channels);
        let (sv, sa) = (sigmas(p.steps, self.shifts.0), sigmas(p.steps, self.shifts.1));
        let t1 = Instant::now();
        let lay = layout(&cond.tags, g.lt, g.lh, g.lw, g.n_audio, cfg.patch[1], &anchors);
        let (video, audio) = denoise_cancellable(&self.dit, &cond.text, &lay, video, audio, (&sv, &sa), progress, p.cancel.as_deref())?;
        if p.cancelled() {
            bail!("cancelled");
        }
        let t_dit = t1.elapsed().as_secs_f64();
        if let Some(f) = &p.latents {
            write_latents(f, g, &video[n_cond_vals..], &audio)?;
        }
        let t2 = Instant::now();
        let mut out = self.decode(&video[n_cond_vals..], g)?;
        let t_vae = t2.elapsed().as_secs_f64();
        let t3 = Instant::now();
        out.audio = self.decode_audio(&audio)?;
        eprintln!(
            "\n[MiniMax-H3: {w}x{h}, {frames} frames ({} latent), {} text rows, text {t_enc:.1}s, {} steps {t_dit:.1}s, VAE {t_vae:.1}s, audio VAE {:.1}s]",
            g.lt,
            cond.tags.len(),
            sv.len().min(sa.len()) - 1,
            t3.elapsed().as_secs_f64()
        );
        Ok(out)
    }
}

/// A MiniMax-H3 part named by an option: a local path, `org/name/path/in/repo.safetensors` (or
/// `.gguf`), or a repository id.
fn video_part(spec: &str, cache_dir: Option<&Path>) -> Result<PathBuf> {
    let p = Path::new(spec);
    if p.exists() {
        return Ok(p.to_path_buf());
    }
    if spec.ends_with(".safetensors") || spec.ends_with(".gguf") {
        return crate::engine::resolve_file(spec, cache_dir);
    }
    crate::engine::resolve_model(spec, cache_dir)
}

/// Loads a MiniMax-H3 video model (a release folder, a single-file transformer, or a repo id)
/// with the server's and the C API's options: `text_encoder`, `transformer`, `vae`,
/// `audio_vae` (parts replacing the folder's), `quant` / `text_quant` (the text encoder's
/// precision: bf16 or q8), `lora` (`FILE[:SCALE]`, repeatable; a turbo LoRA sets the step count
/// and the shifts), `shift`, `audio_shift`. Returns the pipeline and the step count a turbo LoRA
/// is distilled for.
pub fn load_with_options(src: &str, opts: &[(&str, &str)], cache_dir: Option<&Path>) -> Result<(VideoPipeline, Option<usize>)> {
    use crate::model::Quant;
    let path = Path::new(src);
    let model = if path.exists() {
        path.to_path_buf()
    } else {
        crate::engine::resolve_model(src, cache_dir)?
    };
    let single = model.is_file();
    let mut s = Sources {
        dir: (!single).then(|| model.clone()),
        transformer: single.then(|| model.clone()),
        text_quant: Some(Quant::Bf16),
        ..Default::default()
    };
    let (mut te, mut loras, mut shift, mut audio_shift) = (None, Vec::new(), None, None);
    let num = |k: &str, v: &str| -> Result<f32> { v.parse().with_context(|| format!("{k}: not a number: {v:?}")) };
    for (k, v) in opts {
        match *k {
            "text_encoder" => te = Some(video_part(v, cache_dir)?),
            "transformer" => s.transformer = Some(video_part(v, cache_dir)?),
            "vae" => s.vae = Some(video_part(v, cache_dir)?),
            "audio_vae" => s.audio_vae = Some(video_part(v, cache_dir)?),
            "quant" | "text_quant" => {
                // Q4 moves the text encoder's hidden states too far (as for Qwen-Image's).
                s.text_quant = Some(match Quant::parse(v).with_context(|| format!("{k}: {v:?} (bf16, q8)"))? {
                    Quant::Q4 => Quant::Q8,
                    q => q,
                })
            }
            "lora" => {
                let mut l = crate::minimax_h3::lora::LoraSpec::parse(v)?;
                l.path = video_part(&l.path.to_string_lossy(), cache_dir)?;
                loras.push(l);
            }
            "shift" => shift = Some(num(k, v)?),
            "audio_shift" => audio_shift = Some(num(k, v)?),
            k => bail!("unknown MiniMax-H3 option {k:?} (text_encoder, transformer, vae, audio_vae, quant, lora, shift, audio_shift)"),
        }
    }
    if single && s.vae.is_none() {
        bail!("{src} is a single-file transformer: name the video VAE with the vae option");
    }
    s.text_encoder = match te {
        Some(t) => t,
        None => crate::minimax_h3::find_text_encoder(&model, cache_dir)?,
    };
    let mut pipe = VideoPipeline::load_from(&s)?;
    for line in pipe.apply_loras(&loras)? {
        eprintln!("{line}");
    }
    let mut steps = None;
    if let Some(t) = loras.iter().find_map(|l| crate::minimax_h3::lora::turbo_settings(&l.path)) {
        steps = Some(t.forwards + 1);
        pipe.shifts = (t.video_shift, t.audio_shift);
    }
    if let Some(v) = shift {
        pipe.shifts.0 = v;
    }
    if let Some(v) = audio_shift {
        pipe.shifts.1 = v;
    }
    Ok((pipe, steps))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_helpers() {
        assert_eq!(resolve_canvas(16.0, 9.0, 768, 768 * 1344, 32).unwrap(), (768, 1344));
        assert_eq!(resolve_canvas(1.0, 1.0, 768, 768 * 1344, 32).unwrap(), (768, 768));
        assert!(resolve_canvas(5.0, 1.0, 768, 768 * 1344, 32).is_err());
        assert_eq!(align_frames(124, 17, 5), 124);
        assert_eq!(align_frames(125, 17, 5), 141);
        assert_eq!(align_frames(1, 17, 5), 22);
        assert_eq!(latent_frames(124, 17, 5), 37);
        assert_eq!(latent_frames(22, 17, 5), 7);
        assert_eq!(audio_latents(124), 207);
    }

    /// A byte-level BPE tokenizer over printable ASCII with Qwen's last added token, as the
    /// release's `tokenizer.json` (and a GGUF's vocabulary) has it: nothing at 151669 and after.
    fn ascii_tokenizer() -> crate::tokenizer::Tokenizer {
        let mut vocab = serde_json::Map::new();
        for (i, c) in (b'!'..=b'~').enumerate() {
            vocab.insert((c as char).to_string(), serde_json::json!(i));
        }
        vocab.insert("\u{120}".into(), serde_json::json!(vocab.len()));
        let byte_level = serde_json::json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false});
        let j = serde_json::json!({
            "version": "1.0",
            "added_tokens": [{"id": 151668, "content": "</think>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false}],
            "normalizer": {"type": "NFC"},
            "pre_tokenizer": byte_level,
            "decoder": byte_level,
            "model": {"type": "BPE", "vocab": vocab, "merges": []},
        });
        crate::tokenizer::Tokenizer::from_bytes(serde_json::to_vec(&j).unwrap()).unwrap()
    }

    #[test]
    fn dialogue_tags_are_single_tokens() {
        let text = "says: <d>[English] Hi.</d> Then";
        let mut tok = ascii_tokenizer();
        let ids = |t: &crate::tokenizer::Tokenizer| t.encode(text, false).unwrap().get_ids().to_vec();
        let plain = ids(&tok);
        assert!(!plain.iter().any(|&i| i >= 151669));
        tok.add_special_tokens_at(&EXTRA_TOKENS).unwrap();
        let got = ids(&tok);
        // The ids of MiniMax's tokenizer_config.json through transformers, and of ComfyUI.
        assert_eq!(tok.token_to_id("<d>"), Some(151669));
        assert_eq!(tok.token_to_id("</d>"), Some(151670));
        assert_eq!(tok.token_to_id("<|caption_end|>"), Some(151675));
        assert_eq!(got.iter().filter(|&&i| i == 151669).count(), 1);
        assert_eq!(got.iter().filter(|&&i| i == 151670).count(), 1);
        // `<d>` and `</d>` were 3 and 4 tokens, the rest of the text is unchanged.
        assert_eq!(got.len() + 5, plain.len());
        let open = got.iter().position(|&i| i == 151669).unwrap();
        assert_eq!(got[..open], plain[..open]);
        assert_eq!(tok.decode(&got, false).unwrap(), text);
        // A second time changes nothing; an id held by another token is refused.
        tok.add_special_tokens_at(&EXTRA_TOKENS).unwrap();
        assert_eq!(ids(&tok), got);
        assert!(tok.add_special_tokens_at(&[("<x>", 151669)]).is_err());
        assert!(tok.add_special_tokens_at(&[("<d>", 7)]).is_err());
    }

    #[test]
    fn sigma_grid() {
        let s = sigmas(50, 12.0);
        assert_eq!(s[0], 1.0);
        assert_eq!(*s.last().unwrap(), 0.0);
        assert!(s.windows(2).all(|w| w[1] < w[0]));
        // linspace(1, 0, 3) = 1, 0.5, 0; shifted by 3: 1, 0.75, 0.
        assert_eq!(sigmas(3, 3.0), vec![1.0, 0.75, 0.0]);
    }

    #[test]
    fn packed_layout() {
        let tags = [TAG_TEXT; 3];
        let l = layout(&tags, 7, 4, 8, 5, 2, &[Anchor::First, Anchor::Last]);
        let rpf = 2 * 4;
        assert_eq!(l.n_cond, 2 * rpf);
        assert_eq!(l.positions.len(), 3 + 2 * rpf + 10 + 7 * rpf);
        assert_eq!(l.video_index.len(), 9 * rpf);
        assert_eq!(l.audio_index, (3 + 2 * rpf..3 + 2 * rpf + 10).collect::<Vec<_>>());
        // Frame times: 3 + 5/3 * (0, 1, 5, 9, 13, 17, 18).
        let v0 = 3 + 2 * rpf + 10;
        let t = |f: usize| l.positions[v0 + f * rpf][0];
        assert!((t(1) - (3.0 + 5.0 / 3.0)).abs() < 1e-5);
        assert!((t(6) - (3.0 + 30.0)).abs() < 1e-5);
        // The first keyframe sits at the first frame's time, the last one at the clip's span less
        // one unit frame: 3 + 5/3 * (22 - 1).
        assert_eq!(l.positions[3][0], 3.0);
        assert!((l.positions[3 + rpf][0] - 38.0).abs() < 1e-5);
        // Audio: channel-major, at the ends of the width grid.
        assert_eq!(l.positions[3 + 2 * rpf + 5][0], 3.0);
        assert!(l.positions[3 + 2 * rpf][2] < l.positions[3 + 2 * rpf + 5][2]);
        assert_eq!(l.tags[3 + 2 * rpf], TAG_AUDIO);
    }

    #[test]
    fn latents_file_roundtrip() {
        let g = Geometry {
            lt: 2,
            lh: 4,
            lw: 6,
            n_audio: 3,
        };
        let video: Vec<f32> = (0..2 * 2 * 3 * 8).map(|i| i as f32 * 0.5).collect();
        let audio: Vec<f32> = (0..2 * 3 * 4).map(|i| -(i as f32)).collect();
        let path = std::env::temp_dir().join(format!("oxinfer_h3_latents_{}.bin", std::process::id()));
        write_latents(&path, g, &video, &audio).unwrap();
        assert_eq!(read_latents(&path, g).unwrap(), (video, audio));
        // Rows of another request are refused.
        assert!(read_latents(&path, Geometry { lt: 3, ..g }).is_err());
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn patch_roundtrip() {
        let (c, t, h, w) = (3, 2, 4, 6);
        let lat: Vec<f32> = (0..c * t * h * w).map(|i| i as f32).collect();
        let rows = patchify_video(&lat, c, t, h, w, 2);
        assert_eq!(unpatchify_video(&rows, c, t, h, w, 2), lat);
    }

    #[test]
    fn release_vae_config_in_diffusers_keys() {
        // `FL2VA/video_vae/config.json` and `source/config.json` (abridged) give the keys of the
        // root `vae/config.json` (`AutoencoderKLMiniMaxH3`).
        let dir = std::env::temp_dir().join(format!("oxinfer-h3-vae-cfg-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("source")).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"_class_name": "MiniMaxH3VideoVAE", "source_path": "source", "vae_clip_length": 17, "vae_token_drop": 3,
                "vae_tile_size": 256, "vae_tile_overlap_min": 64, "latent_channels": 2,
                "latents_mean": [0.5, -0.25], "latents_std": [1.5, 2.0]}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("source/config.json"),
            r#"{"_class_name": "AutoencoderKLLegacy", "ch": 128, "ch_mult": [1, 2, 2, 4, 4, 8], "num_res_blocks": 2,
                "padding_mode": "reflect", "space_down": [2, 2, 2, 2, 1, 1], "time_down": [1, 2, 2, 1, 1, 1],
                "use_vit_decoder": true, "vit_decoder_kwargs": {"dim_head": 64, "heads": 32, "num_layers": 36,
                "rope_dim_ratio": 0.75, "rope_theta": 100.0}}"#,
        )
        .unwrap();
        let c = release_vae_json(&dir).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        let want = serde_json::json!({
            "_class_name": "AutoencoderKLMiniMaxH3", "latent_channels": 2, "latents_mean": [0.5, -0.25],
            "latents_std": [1.5, 2.0], "block_out_channels": [128, 256, 256, 512, 512, 1024], "layers_per_block": 2,
            "spatial_downsample_factors": [2, 2, 2, 2, 1, 1], "temporal_downsample_factors": [1, 2, 2, 1, 1, 1],
            "spatial_padding_mode": "reflect", "decoder_num_layers": 36, "decoder_num_attention_heads": 32,
            "decoder_attention_head_dim": 64, "decoder_rope_theta": 100.0, "decoder_rope_dim_ratio": 0.75,
            "clip_length": 17, "token_drop": 3, "vae_tile_size": 256, "vae_tile_overlap_min": 64,
        });
        assert_eq!(c, want);
        let cfg = VaeConfig::from_json(&c).unwrap();
        assert_eq!(
            (cfg.axes, cfg.patch_t, cfg.dim, cfg.layers, cfg.clip_length, cfg.token_drop),
            (3, 4, 2048, 36, 17, 3)
        );
    }
}
