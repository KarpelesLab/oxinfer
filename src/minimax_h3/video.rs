//! MiniMax-H3: text to video and first / last frame to video (the `t2va` / `fl2va` tasks of the
//! `transformer/` checkpoint), following diffusers' modular `minimax_h3` pipeline:
//!
//! - the prompt verbatim (no chat template, no special tokens) through the Qwen3-VL text encoder,
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
//! The audio latents are denoised with the video (the model is trained on the joint sequence) but
//! not decoded: the audio VAE is not implemented, so the output is silent.

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
const AUDIO_CHANNELS: usize = 2;
const PIXEL_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const PIXEL_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Whether `dir` is a MiniMax-H3 release in diffusers' layout (`MiniMaxH3ModularPipeline`).
pub fn is_h3_video(dir: &Path) -> bool {
    let Ok(b) = crate::fsx::read(&dir.join("model_index.json")) else {
        return false;
    };
    serde_json::from_slice::<serde_json::Value>(&b)
        .ok()
        .and_then(|v| {
            v.get("_class_name")
                .and_then(|c| c.as_str())
                .map(|c| c == "MiniMaxH3ModularPipeline")
        })
        .unwrap_or(false)
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

const FRAME_RESCALE: f64 = 5.0 / 3.0;
const FRAMES_PER_LATENT: [f64; 5] = [1.0, 4.0, 4.0, 4.0, 4.0];

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
    mut video: Vec<f32>,
    mut audio: Vec<f32>,
    g: Geometry,
    anchors: &[Anchor],
    sig: (&[f32], &[f32]),
    progress: &mut dyn FnMut(usize, usize),
) -> Result<(Vec<f32>, Vec<f32>)> {
    let cfg = &dit.cfg;
    let lay = layout(text_tags, g.lt, g.lh, g.lw, g.n_audio, cfg.patch[1], anchors);
    let emb = dit.embed_text(text);
    let (pd, ac) = (cfg.patch_dim(), cfg.audio_channels);
    let (sv, sa) = sig;
    let steps = sv.len().min(sa.len()) - 1;
    let cond_idx = &lay.video_index[..lay.n_cond];
    for i in 0..steps {
        let (tv, ta) = (1.0 - sv[i], 1.0 - sa[i]);
        // The distinct timesteps, sorted, and every row's index into them (text rows: the video's).
        let tc = tv.max(KEYFRAME_T);
        let mut ts = vec![tv, ta];
        if lay.n_cond > 0 {
            ts.push(tc);
        }
        ts.sort_by(|a, b| a.total_cmp(b));
        ts.dedup();
        let at = |t: f32| ts.iter().position(|&v| v == t).unwrap();
        let mut tidx = vec![at(tv); lay.positions.len()];
        cond_idx.iter().for_each(|&r| tidx[r] = at(tc));
        lay.audio_index.iter().for_each(|&r| tidx[r] = at(ta));
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
        let c0 = lay.n_cond * pd;
        step(&mut video[c0..], &v[c0..], tv, sv[i], sv[i + 1]);
        step(&mut audio, &a, ta, sa[i], sa[i + 1]);
        debug_assert_eq!(audio.len() % ac, 0);
        progress(i + 1, steps);
    }
    Ok((video, audio))
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
}

impl VideoParams {
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
        }
    }
}

/// Generated frames (RGB8, row-major) at [`FPS`].
pub struct Video {
    pub width: usize,
    pub height: usize,
    pub frames: Vec<Vec<u8>>,
}

/// The conditioning of one request: `[n, text_dim]` rows and their modality tags.
pub struct Conditioning {
    pub text: Vec<f32>,
    pub tags: Vec<u8>,
}

pub struct VideoPipeline {
    pub dir: PathBuf,
    pub text_encoder_dir: PathBuf,
    pub dit: Dit<'static>,
    pub vae: VaeDecoder<'static>,
    vae_files: &'static ModelFiles,
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
    use crate::qwen_vl::{QwenVl, QwenVlOptions};
    let parent = text_encoder.parent().map(Path::to_path_buf).unwrap_or_default();
    let opts = QwenVlOptions {
        mmap: true,
        vision: !keyframes.is_empty(),
        tokenizer_dir: [model.join("tokenizer"), parent.join("tokenizer"), text_encoder.to_path_buf()]
            .into_iter()
            .find(|d| d.join("tokenizer.json").exists()),
        ..Default::default()
    };
    let vl = QwenVl::load(text_encoder, &opts)?;
    if vl.model().cfg.num_hidden_layers <= 50 {
        bail!(
            "the text encoder has {} layers; MiniMax-H3 conditions on its 50th",
            vl.model().cfg.num_hidden_layers
        );
    }
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
    let mut hs = vl.hidden_states_tokens(ids, &images, None, &[49])?;
    let (_, text) = hs.layers.pop().context("layer 50 not captured")?;
    Ok(Conditioning { text, tags })
}

impl VideoPipeline {
    /// Loads the DiT (`transformer/`) and the video VAE (`vae/`) of the release in `dir` (bf16,
    /// mapped); the text encoder from `text_encoder` when it is used.
    pub fn load(dir: &Path, text_encoder: &Path) -> Result<Self> {
        let tdir = dir.join("transformer");
        let dit_files: &'static ModelFiles = Box::leak(Box::new(ModelFiles::open(&tdir, false)?));
        let dit = Dit::load(DitConfig::load(&tdir)?, dit_files, None)?;
        if !dit.has_audio() {
            bail!("{}: the transformer has no audio stream (not MiniMax-H3)", tdir.display());
        }
        let vdir = dir.join("vae");
        let vae_files: &'static ModelFiles = Box::leak(Box::new(ModelFiles::open(&vdir, false)?));
        let vae = VaeDecoder::load(VaeConfig::load(&vdir)?, vae_files)?;
        if vae.cfg.axes != 3 {
            bail!("{}: not a video VAE", vdir.display());
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            text_encoder_dir: text_encoder.to_path_buf(),
            dit,
            vae,
            vae_files,
            shifts: (shift_of(&dir.join("scheduler"), 12.0), shift_of(&dir.join("audio_scheduler"), 3.0)),
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
        encode_prompt(&self.dir, &self.text_encoder_dir, prompt, keyframes)
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

    /// The video encoder (loaded on demand: only keyframes need it).
    pub fn encoder(&self) -> Result<VideoEncoder> {
        VideoEncoder::load(&self.dir.join("vae"), &self.vae.cfg, self.vae_files)
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
        })
    }

    /// Text (and keyframes) to video. `cond`: a conditioning computed earlier
    /// ([`VideoPipeline::encode_prompt`]) instead of running the text encoder.
    pub fn generate(&self, p: &VideoParams, cond: Option<Conditioning>, progress: &mut dyn FnMut(usize, usize)) -> Result<Video> {
        if p.steps < 2 {
            bail!("--steps must be at least 2 (the sigma grid includes its end)");
        }
        let (h, w, frames, g) = self.geometry(p)?;
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
        let (video, _audio) = denoise(&self.dit, &cond.text, &cond.tags, video, audio, g, &anchors, (&sv, &sa), progress)?;
        let t_dit = t1.elapsed().as_secs_f64();
        let t2 = Instant::now();
        let out = self.decode(&video[n_cond_vals..], g)?;
        eprintln!(
            "\n[MiniMax-H3: {w}x{h}, {frames} frames ({} latent), {} text rows, text {t_enc:.1}s, {} steps {t_dit:.1}s, VAE {:.1}s]",
            g.lt,
            cond.tags.len(),
            sv.len().min(sa.len()) - 1,
            t2.elapsed().as_secs_f64()
        );
        Ok(out)
    }
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
    fn patch_roundtrip() {
        let (c, t, h, w) = (3, 2, 4, 6);
        let lat: Vec<f32> = (0..c * t * h * w).map(|i| i as f32).collect();
        let rows = patchify_video(&lat, c, t, h, w, 2);
        assert_eq!(unpatchify_video(&rows, c, t, h, w, 2), lat);
    }
}
