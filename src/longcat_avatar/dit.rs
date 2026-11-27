//! `LongCatVideoAvatarTransformer3DModel`: 48 single-stream blocks of width 4096 over the
//! 1 x 2 x 2 patches of the video latents. Each block:
//!
//! - AdaLN from the per-frame timestep embedding (one set of shift / scale / gate per latent
//!   frame: conditioning frames are at `t = 0`, the others at the step's timestep);
//! - self-attention (fused `qkv` with bias, RMS-normed queries and keys per head, 3-axis RoPE
//!   over adjacent channel pairs). Conditioning frames only look at themselves: with one
//!   (image to video) the first frame attends to the first frame; with more (continuation) the
//!   reference frame attends to itself and the conditioning frames to each other; the noisy
//!   frames attend to everything (or, within `mask_frame_range` of the reference's position, to
//!   everything but the reference);
//! - cross-attention to the text (all `model_max_length` positions: the padded ones are zero
//!   rows after the caption embedder, which still gives them a key and a value from the biases),
//!   added without a gate, for the noisy frames only;
//! - cross-attention of each noisy frame's tokens to that frame's 32 audio tokens, then a second
//!   AdaLN (norm, scale, shift, gate) on its output;
//! - a SwiGLU feed-forward.
//!
//! With `ref_target_masks` (several speakers) the self-attention also yields, per noisy token,
//! how much it looks at each masked region of the reference frame; the audio cross-attention
//! turns that into a 1-D rotary position for the token (speaker one in `[0, 4]`, speaker two in
//! `[20, 24]`, background at 12) against fixed positions of each speaker's audio tokens.
//!
//! This file is the float32 path on the CPU (weights dequantized a matrix at a time);
//! `cuda.rs` runs the block stack on the device.

use super::lora::{Active, add_side};
use super::weights::{Mat, Shards};
use crate::kernels::{for_chunks, gelu_tanh};
use crate::sd::ops::{attention, matmul_f32, silu};
use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::path::Path;

fn d_window() -> usize {
    5
}
fn d_vae_scale() -> usize {
    4
}
fn d_class_range() -> usize {
    24
}
fn d_class_interval() -> usize {
    4
}
fn d_ctx() -> usize {
    32
}
fn d_max_len() -> usize {
    512
}

/// `base_model/config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct DitConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub hidden_size: usize,
    pub depth: usize,
    pub num_heads: usize,
    pub caption_channels: usize,
    pub adaln_tembed_dim: usize,
    pub frequency_embedding_size: usize,
    pub patch_size: [usize; 3],
    #[serde(default)]
    pub text_tokens_zero_pad: bool,
    #[serde(default = "d_window")]
    pub audio_window: usize,
    pub audio_block: usize,
    pub audio_channel: usize,
    pub output_dim: usize,
    #[serde(default = "d_ctx")]
    pub context_tokens: usize,
    #[serde(default = "d_vae_scale")]
    pub vae_scale: usize,
    #[serde(default)]
    pub audio_prenorm: bool,
    #[serde(default = "d_class_range")]
    pub class_range: usize,
    #[serde(default = "d_class_interval")]
    pub class_interval: usize,
    #[serde(default = "d_max_len")]
    pub model_max_length: usize,
}

/// A layer norm's weight and bias.
pub(crate) struct Affine {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
}

impl Affine {
    fn load(s: &Shards, p: &str) -> Result<Self> {
        Ok(Self {
            w: s.f32(&format!("{p}.weight"))?,
            b: s.f32(&format!("{p}.bias"))?,
        })
    }
}

pub(crate) struct Block<'a> {
    pub ada: Mat<'a>,
    pub audio_ada: Mat<'a>,
    pub qkv: Mat<'a>,
    pub attn_proj: Mat<'a>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub pre_crs: Affine,
    pub cq: Mat<'a>,
    pub ckv: Mat<'a>,
    pub cproj: Mat<'a>,
    pub cq_norm: Vec<f32>,
    pub ck_norm: Vec<f32>,
    pub pre_video: Affine,
    pub pre_audio: Option<Affine>,
    pub aq: Mat<'a>,
    pub akv: Mat<'a>,
    pub aproj: Mat<'a>,
    pub aq_norm: Vec<f32>,
    pub ak_norm: Vec<f32>,
    pub w1: Mat<'a>,
    pub w2: Mat<'a>,
    pub w3: Mat<'a>,
}

pub struct Dit {
    pub cfg: DitConfig,
    pub(crate) blocks: Vec<Block<'static>>,
    x_embed: Mat<'static>,
    t_mlp: [Mat<'static>; 2],
    y_proj: [Mat<'static>; 2],
    audio_proj: [Mat<'static>; 4],
    audio_norm: Affine,
    final_ada: Mat<'static>,
    final_lin: Mat<'static>,
    /// Whether the matrices are the int8 ones.
    pub int8: bool,
    /// Bytes of the weight files.
    pub weight_bytes: usize,
    pub(crate) device: super::Device,
    // Declared last: dropped after every layer borrowing it.
    _files: Box<Shards>,
}

/// What a forward is conditioned on.
#[derive(Clone, Copy)]
pub struct Cond<'c> {
    /// The text encoder's rows of the prompt's tokens (`[n, caption_channels]`); the positions up
    /// to `text_len` are padding.
    pub text: &'c [f32],
    pub text_len: usize,
    /// `[humans, frames, audio_window, audio_block, audio_channel]`: per video frame of the clip
    /// (`1 + vae_scale (latent frames - 1)` of them, not counting a reference latent) the Whisper
    /// features of the frames around it.
    pub audio: &'c [f32],
    pub humans: usize,
    /// Leading latent frames that are given (reference and conditioning frames).
    pub num_cond: usize,
    /// 1 when the first latent frame is the reference image of a continuation.
    pub num_ref: usize,
    /// The temporal position the reference frame is placed at (continuation).
    pub ref_img_index: usize,
    pub mask_frame_range: usize,
    /// Regions of the reference frame per speaker (`[n, h, w]` in pixels of any size).
    pub ref_masks: Option<(&'c [f32], usize, usize, usize)>,
    pub lora: Active<'c>,
}

/// What the block stack reads besides the tokens.
pub(crate) struct Ctx<'c> {
    pub n_t: usize,
    /// Tokens per latent frame.
    pub s: usize,
    pub grid: (usize, usize),
    /// `[n_t, adaln_tembed_dim]`.
    pub temb: Vec<f32>,
    /// `[text_len, hidden]`.
    pub text: Vec<f32>,
    pub text_len: usize,
    /// `[n_t, na, output_dim]`.
    pub audio: Vec<f32>,
    pub na: usize,
    /// `[n, head_dim / 2]`.
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    pub num_cond: usize,
    /// Noisy frames (from, to) that do not see the reference frame.
    pub mask_frames: Option<(usize, usize)>,
    /// `[n_masks, s]` of 0 / 1.
    pub token_masks: Option<(Vec<f32>, usize)>,
    pub humans: usize,
    pub lora: Active<'c>,
}

/// Row ranges of one attention call: queries, keys.
pub(crate) type Seg = (std::ops::Range<usize>, std::ops::Range<usize>);

impl Ctx<'_> {
    /// The self-attention's calls (see the module docs).
    pub(crate) fn segments(&self) -> Vec<Seg> {
        let (s, n) = (self.s, self.n_t * self.s);
        let nc = self.num_cond;
        if nc == 0 {
            return vec![(0..n, 0..n)];
        }
        if nc == 1 {
            return vec![(0..s, 0..s), (s..n, 0..n)];
        }
        let mut v = vec![(0..s, 0..s), (s..nc * s, s..nc * s)];
        if nc < self.n_t {
            match self.mask_frames {
                Some((a, b)) => {
                    let base = nc * s;
                    for (q, k) in [
                        (base..base + a * s, 0..n),
                        (base + a * s..base + b * s, s..n),
                        (base + b * s..n, 0..n),
                    ] {
                        if !q.is_empty() {
                            v.push((q, k));
                        }
                    }
                }
                None => v.push((nc * s..n, 0..n)),
            }
        }
        v
    }
}

/// Sinusoidal timestep features, cosines first (`TimestepEmbedder.timestep_embedding`).
fn timestep_features(t: f32, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let mut e = vec![0f32; dim];
    for i in 0..half {
        let f = (-(10000f32.ln()) * i as f32 / half as f32).exp();
        let a = t * f;
        e[i] = a.cos();
        e[half + i] = a.sin();
    }
    e
}

/// Layer norm without weights per row of `h`, then `* (1 + scale) + shift` of the row's frame
/// (`mods`: `[frames, .. ]` with shift at `shift_at`, scale at `scale_at`; `frame0`: the frame of
/// the first row).
#[allow(clippy::too_many_arguments)]
fn norm_mod(x: &[f32], h: usize, s: usize, mods: &[f32], stride: usize, shift_at: usize, scale_at: usize, frame0: usize) -> Vec<f32> {
    let mut y = x.to_vec();
    for_chunks(&mut y, h, x.len() >= 1 << 14, |i, r| {
        let f = i / s + frame0;
        let m = &mods[f * stride..(f + 1) * stride];
        let (shift, scale) = (&m[shift_at..shift_at + h], &m[scale_at..scale_at + h]);
        let mean = r.iter().map(|&v| v as f64).sum::<f64>() / h as f64;
        let var = r.iter().map(|&v| (v as f64 - mean) * (v as f64 - mean)).sum::<f64>() / h as f64;
        let inv = (1.0 / (var + 1e-6).sqrt()) as f32;
        let mean = mean as f32;
        for ((v, &sh), &sc) in r.iter_mut().zip(shift).zip(scale) {
            *v = (*v - mean) * inv * (1.0 + sc) + sh;
        }
    });
    y
}

/// Layer norm with weight and bias per row.
pub(crate) fn norm_affine(x: &[f32], a: &Affine, eps: f64) -> Vec<f32> {
    let h = a.w.len();
    let mut y = x.to_vec();
    for_chunks(&mut y, h, x.len() >= 1 << 14, |_, r| {
        let mean = r.iter().map(|&v| v as f64).sum::<f64>() / h as f64;
        let var = r.iter().map(|&v| (v as f64 - mean) * (v as f64 - mean)).sum::<f64>() / h as f64;
        let inv = (1.0 / (var + eps).sqrt()) as f32;
        let mean = mean as f32;
        for ((v, &w), &b) in r.iter_mut().zip(&a.w).zip(&a.b) {
            *v = (*v - mean) * inv * w + b;
        }
    });
    y
}

/// `x[row] += gate[frame] * y[row]` (`y` starts at frame `frame0`).
#[allow(clippy::too_many_arguments)]
fn add_gated(x: &mut [f32], y: &[f32], h: usize, s: usize, mods: &[f32], stride: usize, gate_at: usize, frame0: usize) {
    let x = &mut x[frame0 * s * h..];
    debug_assert_eq!(x.len(), y.len());
    for_chunks(x, h, y.len() >= 1 << 14, |i, r| {
        let f = i / s + frame0;
        let g = &mods[f * stride + gate_at..f * stride + gate_at + h];
        for ((v, &a), &gg) in r.iter_mut().zip(&y[i * h..(i + 1) * h]).zip(g) {
            *v += gg * a;
        }
    });
}

/// Section `part` of a fused projection's rows (`[rows, parts * heads * hd]`: all heads of the
/// queries, then of the keys, ...) with each head RMS-normed (`w`: per-channel weights) and
/// rotated (`cs`: `[rows, hd / 2]` cosines and sines of the adjacent pairs).
#[allow(clippy::too_many_arguments)]
pub(crate) fn head_norm(
    src: &[f32],
    stride: usize,
    part: usize,
    heads: usize,
    hd: usize,
    w: Option<&[f32]>,
    cs: Option<(&[f32], &[f32])>,
) -> Vec<f32> {
    let d = heads * hd;
    let rows = src.len() / stride;
    let mut out = vec![0f32; rows * d];
    for_chunks(&mut out, d, rows * d >= 1 << 14, |i, o| {
        o.copy_from_slice(&src[i * stride + part * d..i * stride + (part + 1) * d]);
        let Some(w) = w else { return };
        for hh in o.chunks_mut(hd) {
            let ms = hh.iter().map(|&v| v * v).sum::<f32>() / hd as f32;
            let inv = 1.0 / (ms + 1e-6).sqrt();
            for (v, &g) in hh.iter_mut().zip(w) {
                *v = *v * inv * g;
            }
            if let Some((cos, sin)) = cs {
                let (c, s) = (&cos[i * hd / 2..(i + 1) * hd / 2], &sin[i * hd / 2..(i + 1) * hd / 2]);
                for (p, pair) in hh.chunks_mut(2).enumerate() {
                    let (a, b) = (pair[0], pair[1]);
                    pair[0] = a * c[p] - b * s[p];
                    pair[1] = b * c[p] + a * s[p];
                }
            }
        }
    });
    out
}

/// Cosines and sines of a 1-D rotary position per row: `[rows, hd / 2]`.
fn rope_1d(pos: &[f32], hd: usize) -> (Vec<f32>, Vec<f32>) {
    let half = hd / 2;
    let freqs: Vec<f32> = (0..half).map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / hd as f32)).collect();
    let mut cos = vec![0f32; pos.len() * half];
    let mut sin = vec![0f32; pos.len() * half];
    for (i, &p) in pos.iter().enumerate() {
        for (j, &f) in freqs.iter().enumerate() {
            let a = p * f;
            cos[i * half + j] = a.cos();
            sin[i * half + j] = a.sin();
        }
    }
    (cos, sin)
}

/// Rotates every head of `x` (`[rows, heads * hd]`) by its row's angles.
fn rotate(x: &mut [f32], heads: usize, hd: usize, cos: &[f32], sin: &[f32]) {
    let d = heads * hd;
    for_chunks(x, d, x.len() >= 1 << 14, |i, o| {
        let (c, s) = (&cos[i * hd / 2..(i + 1) * hd / 2], &sin[i * hd / 2..(i + 1) * hd / 2]);
        for hh in o.chunks_mut(hd) {
            for (p, pair) in hh.chunks_mut(2).enumerate() {
                let (a, b) = (pair[0], pair[1]);
                pair[0] = a * c[p] - b * s[p];
                pair[1] = b * c[p] + a * s[p];
            }
        }
    });
}

/// The cosines and sines of the 3-axis RoPE: `[frames * h * w, hd / 2]`, the pairs of the
/// temporal axis first (`hd - 4 (hd / 6)` channels), then height and width (`2 (hd / 6)` each).
/// `times`: the temporal position of each latent frame.
pub(crate) fn rope_3d(times: &[f32], h: usize, w: usize, hd: usize) -> (Vec<f32>, Vec<f32>) {
    let dh = 2 * (hd / 6);
    let dt = hd - 2 * dh;
    let freqs = |dim: usize| -> Vec<f32> { (0..dim / 2).map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / dim as f32)).collect() };
    let (ft, fh) = (freqs(dt), freqs(dh));
    let half = hd / 2;
    let n = times.len() * h * w;
    let mut cos = vec![0f32; n * half];
    let mut sin = vec![0f32; n * half];
    for (f, &tp) in times.iter().enumerate() {
        for y in 0..h {
            for x in 0..w {
                let at = ((f * h + y) * w + x) * half;
                let mut j = 0;
                for (pos, fr) in [(tp, &ft), (y as f32, &fh), (x as f32, &fh)] {
                    for &q in fr.iter() {
                        let a = pos * q;
                        cos[at + j] = a.cos();
                        sin[at + j] = a.sin();
                        j += 1;
                    }
                }
            }
        }
    }
    (cos, sin)
}

/// `torch.nn.functional.interpolate(mode="nearest")` of `[n, h, w]` masks to `[n, th * tw]`, as
/// 0 / 1.
fn token_masks(m: &[f32], n: usize, h: usize, w: usize, th: usize, tw: usize) -> Vec<f32> {
    let mut out = vec![0f32; n * th * tw];
    let (sy, sx) = (h as f32 / th as f32, w as f32 / tw as f32);
    for c in 0..n {
        for y in 0..th {
            let iy = ((y as f32 * sy).floor() as usize).min(h - 1);
            for x in 0..tw {
                let ix = ((x as f32 * sx).floor() as usize).min(w - 1);
                out[(c * th + y) * tw + x] = if m[(c * h + iy) * w + ix] > 0.0 { 1.0 } else { 0.0 };
            }
        }
    }
    out
}

/// Per class and query: the attention it gives to the class's region of the reference frame,
/// averaged over the heads. `q`: `[nq, heads * hd]`, `kref`: `[s, heads * hd]`, `masks`:
/// `[n_masks, s]`; returns `[n_masks, nq]`.
pub(crate) fn ref_attn_map(q: &[f32], kref: &[f32], masks: &[f32], n_masks: usize, heads: usize, hd: usize) -> Vec<f32> {
    let d = heads * hd;
    let (nq, s) = (q.len() / d, kref.len() / d);
    let scale = 1.0 / (hd as f32).sqrt();
    let sums: Vec<f32> = (0..n_masks).map(|c| masks[c * s..(c + 1) * s].iter().sum()).collect();
    let mut out = vec![0f32; n_masks * nq];
    let hdp = hd.next_multiple_of(32);
    const QC: usize = 2048;
    let mut kh = vec![0f32; s * hdp];
    let mut qh = vec![0f32; QC * hdp];
    let mut part = vec![0f32; QC * n_masks];
    for h in 0..heads {
        for_chunks(&mut kh, hdp, true, |j, r| {
            r[..hd].copy_from_slice(&kref[j * d + h * hd..j * d + (h + 1) * hd])
        });
        let mut q0 = 0;
        while q0 < nq {
            let n = QC.min(nq - q0);
            for_chunks(&mut qh[..n * hdp], hdp, true, |i, r| {
                r[..hd].copy_from_slice(&q[(q0 + i) * d + h * hd..(q0 + i) * d + (h + 1) * hd])
            });
            let sc = matmul_f32(&kh, s, hdp, &qh[..n * hdp], n);
            for_chunks(&mut part[..n * n_masks], n_masks, true, |i, o| {
                let row = &sc[i * s..(i + 1) * s];
                let mx = row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
                let mut tot = 0f32;
                o.fill(0.0);
                for (j, &v) in row.iter().enumerate() {
                    let e = ((v - mx) * scale).exp();
                    tot += e;
                    for (c, oc) in o.iter_mut().enumerate() {
                        *oc += e * masks[c * s + j];
                    }
                }
                for oc in o.iter_mut() {
                    *oc /= tot;
                }
            });
            for i in 0..n {
                for c in 0..n_masks {
                    out[c * nq + q0 + i] += part[i * n_masks + c] / sums[c] / heads as f32;
                }
            }
            q0 += n;
        }
    }
    out
}

/// The 1-D rotary positions of the noisy tokens from their attention to the speakers' regions,
/// and of one frame's audio tokens (`SingleStreamAttention._process_cross_attn`).
pub(crate) fn speaker_positions(map: &[f32], n_masks: usize, na: usize, humans: usize, cfg: &DitConfig) -> (Vec<f32>, Vec<f32>) {
    let nq = map.len() / n_masks;
    let (range, interval) = (cfg.class_range as f32, cfg.class_interval as f32);
    let minmax = |c: usize| {
        let r = &map[c * nq..(c + 1) * nq];
        (
            r.iter().fold(f32::INFINITY, |m, &v| m.min(v)),
            r.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v)),
        )
    };
    let (min1, max1) = minmax(0);
    let (min2, max2) = minmax(1);
    let norm = |v: f32, lo: f32, hi: f32, a: f32, b: f32| (v - lo) / (hi - lo + 1e-8) * (b - a) + a;
    let back = if n_masks <= 3 { (cfg.class_range / 2) as f32 } else { 100.0 };
    let mut pos = vec![0f32; nq];
    for (i, p) in pos.iter_mut().enumerate() {
        let mut best = 0;
        for c in 1..n_masks {
            if map[c * nq + i] > map[best * nq + i] {
                best = c;
            }
        }
        *p = match best {
            0 => norm(map[i], min1, max1, 0.0, interval),
            1 => norm(map[nq + i], min2, max2, range - interval, range),
            _ => back,
        };
    }
    let (h1, h2) = (interval / 2.0, range - interval / 2.0);
    let mut keys = vec![0f32; na];
    if humans > 2 {
        let per = na / humans;
        for (j, k) in keys.iter_mut().enumerate() {
            *k = if j < per {
                h1
            } else if j < 2 * per {
                h2
            } else {
                back
            };
        }
    } else {
        for (j, k) in keys.iter_mut().enumerate() {
            *k = if j < na / 2 { h1 } else { h2 };
        }
    }
    (pos, keys)
}

impl Dit {
    /// Loads `dir` (`base_model/` or `base_model_int8/`: `config.json` and the weight shards).
    pub fn load(dir: &Path) -> Result<Self> {
        let cfg: DitConfig = serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?)
            .with_context(|| format!("{}/config.json", dir.display()))?;
        if cfg.patch_size[0] != 1 {
            bail!("a temporal patch of {} is not supported", cfg.patch_size[0]);
        }
        if !cfg.hidden_size.is_multiple_of(cfg.num_heads) || !(cfg.hidden_size / cfg.num_heads).is_multiple_of(8) {
            bail!("hidden size {} with {} heads is not supported", cfg.hidden_size, cfg.num_heads);
        }
        let files = Box::new(Shards::open(dir)?);
        // SAFETY: as in FluxTransformer::load (field order drops the layers first).
        let s: &'static Shards = unsafe { &*(files.as_ref() as *const Shards) };
        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            let p = format!("blocks.{i}");
            let m = |n: &str| Mat::load(s, &format!("{p}.{n}"));
            let v = |n: &str| s.f32(&format!("{p}.{n}"));
            blocks.push(Block {
                ada: m("adaLN_modulation.1")?,
                audio_ada: m("audio_adaLN_modulation.1")?,
                qkv: m("attn.qkv")?,
                attn_proj: m("attn.proj")?,
                q_norm: v("attn.q_norm.weight")?,
                k_norm: v("attn.k_norm.weight")?,
                pre_crs: Affine::load(s, &format!("{p}.pre_crs_attn_norm"))?,
                cq: m("cross_attn.q_linear")?,
                ckv: m("cross_attn.kv_linear")?,
                cproj: m("cross_attn.proj")?,
                cq_norm: v("cross_attn.q_norm.weight")?,
                ck_norm: v("cross_attn.k_norm.weight")?,
                pre_video: Affine::load(s, &format!("{p}.pre_video_crs_attn_norm"))?,
                pre_audio: if cfg.audio_prenorm {
                    Some(Affine::load(s, &format!("{p}.pre_audio_crs_attn_norm"))?)
                } else {
                    None
                },
                aq: m("audio_cross_attn.q_linear")?,
                akv: m("audio_cross_attn.kv_linear")?,
                aproj: m("audio_cross_attn.proj")?,
                aq_norm: v("audio_cross_attn.q_norm.weight")?,
                ak_norm: v("audio_cross_attn.k_norm.weight")?,
                w1: m("ffn.w1")?,
                w2: m("ffn.w2")?,
                w3: m("ffn.w3")?,
            });
        }
        let int8 = blocks.first().is_some_and(|b| b.qkv.is_int8());
        Ok(Self {
            blocks,
            x_embed: Mat::load(s, "x_embedder.proj")?,
            t_mlp: [Mat::load(s, "t_embedder.mlp.0")?, Mat::load(s, "t_embedder.mlp.2")?],
            y_proj: [Mat::load(s, "y_embedder.y_proj.0")?, Mat::load(s, "y_embedder.y_proj.2")?],
            audio_proj: [
                Mat::load(s, "audio_proj.proj1")?,
                Mat::load(s, "audio_proj.proj1_vf")?,
                Mat::load(s, "audio_proj.proj2")?,
                Mat::load(s, "audio_proj.proj3")?,
            ],
            audio_norm: Affine::load(s, "audio_proj.norm")?,
            final_ada: Mat::load(s, "final_layer.adaLN_modulation.1")?,
            final_lin: Mat::load(s, "final_layer.linear")?,
            int8,
            weight_bytes: files.bytes(),
            device: Default::default(),
            cfg,
            _files: files,
        })
    }

    pub fn head_dim(&self) -> usize {
        self.cfg.hidden_size / self.cfg.num_heads
    }

    /// Frees the device copy of the weights (the next forward brings it back).
    pub fn release_device(&self) {
        self.device.release(0);
    }

    /// [`Dit::release_device`] unless the device still has `bytes` free without it: what a stage
    /// that wants the device between two clips (the VAE) calls, so that a device with room for
    /// both keeps the weights where they are.
    pub fn make_device_room(&self, bytes: usize) {
        self.device.release(bytes.max(1));
    }

    /// The audio tokens of each latent frame: `[humans, frames, context_tokens, output_dim]`.
    fn audio_tokens(&self, audio: &[f32], humans: usize) -> Result<(Vec<f32>, usize)> {
        let c = &self.cfg;
        let (w, sb, ch, vs) = (c.audio_window, c.audio_block, c.audio_channel, c.vae_scale);
        let per = w * sb * ch;
        if humans == 0 || !audio.len().is_multiple_of(humans * per) {
            bail!("audio features: {} values are not rows of {w} x {sb} x {ch}", audio.len());
        }
        let frames = audio.len() / (humans * per);
        if frames == 0 || !(frames - 1).is_multiple_of(vs) {
            bail!("audio features of {frames} frames: expected 1 + {vs} n");
        }
        let n_t = (frames - 1) / vs;
        let mid = w / 2;
        let entry = sb * ch;
        let vf = (w - 1 + vs) * entry;
        let mut out = Vec::with_capacity(humans * (1 + n_t) * c.context_tokens * c.output_dim);
        for hmn in 0..humans {
            let a = &audio[hmn * frames * per..(hmn + 1) * frames * per];
            let first = self.audio_proj[0].forward(&a[..per], 1);
            let mut later = vec![0f32; n_t * vf];
            for n in 0..n_t {
                let dst = &mut later[n * vf..(n + 1) * vf];
                let mut at = 0;
                let mut put = |frame: usize, win: usize| {
                    let src = (frame * w + win) * entry;
                    dst[at..at + entry].copy_from_slice(&a[src..src + entry]);
                    at += entry;
                };
                let base = 1 + n * vs;
                for win in 0..=mid {
                    put(base, win);
                }
                for f in 1..vs.saturating_sub(1) {
                    put(base + f, mid);
                }
                for win in mid..w {
                    put(base + vs - 1, win);
                }
            }
            let mut x = first;
            if n_t > 0 {
                x.extend(self.audio_proj[1].forward(&later, n_t));
            }
            x.iter_mut().for_each(|v| *v = v.max(0.0));
            let mut x = self.audio_proj[2].forward(&x, 1 + n_t);
            x.iter_mut().for_each(|v| *v = v.max(0.0));
            let x = self.audio_proj[3].forward(&x, 1 + n_t);
            // nn.LayerNorm's default epsilon.
            out.extend(norm_affine(&x, &self.audio_norm, 1e-5));
        }
        Ok((out, 1 + n_t))
    }

    /// What the blocks read for this forward.
    pub(crate) fn context<'c>(&self, shape: (usize, usize, usize), timesteps: &[f32], c: &Cond<'c>) -> Result<Ctx<'c>> {
        let cfg = &self.cfg;
        let (n_t, lh, lw) = shape;
        let (gh, gw) = (lh / cfg.patch_size[1], lw / cfg.patch_size[2]);
        let hsz = cfg.hidden_size;
        if timesteps.len() != n_t {
            bail!("{} timesteps for {n_t} latent frames", timesteps.len());
        }
        if c.num_cond > n_t || (c.num_cond > 1 && c.num_ref != 1) {
            bail!("{} conditioning latent frames of {n_t} need the reference frame first", c.num_cond);
        }
        // Timestep embedding per latent frame.
        let fdim = cfg.frequency_embedding_size;
        let mut tf = Vec::with_capacity(n_t * fdim);
        for &t in timesteps {
            tf.extend(timestep_features(t, fdim));
        }
        let mut e = self.t_mlp[0].forward(&tf, n_t);
        e.iter_mut().for_each(|v| *v = silu(*v));
        let temb = self.t_mlp[1].forward(&e, n_t);
        // Caption embedding: zero rows at the padded positions.
        let cc = cfg.caption_channels;
        let nt = c.text.len() / cc;
        if nt > c.text_len || !c.text.len().is_multiple_of(cc) {
            bail!("text rows: {} values for at most {} rows of {cc}", c.text.len(), c.text_len);
        }
        let mut text = vec![0f32; c.text_len * hsz];
        if nt > 0 {
            let mut y = self.y_proj[0].forward(c.text, nt);
            y.iter_mut().for_each(|v| *v = gelu_tanh(*v));
            let y = self.y_proj[1].forward(&y, nt);
            text[..nt * hsz].copy_from_slice(&y);
        }
        if !cfg.text_tokens_zero_pad && nt != c.text_len {
            bail!("a checkpoint without text_tokens_zero_pad needs every text position");
        }
        // Audio tokens per latent frame.
        let (tok, frames) = self.audio_tokens(c.audio, c.humans)?;
        let per_frame = cfg.context_tokens * cfg.output_dim;
        if frames + c.num_ref != n_t {
            bail!(
                "audio features cover {frames} latent frames, the latents have {n_t} (with {} reference)",
                c.num_ref
            );
        }
        let multi = c.ref_masks.is_some();
        let na = if multi { c.humans * cfg.context_tokens } else { cfg.context_tokens };
        if !multi && c.humans != 1 {
            bail!("{} audio tracks need the speakers' regions", c.humans);
        }
        let mut audio = vec![0f32; n_t * na * cfg.output_dim];
        for f in 0..n_t {
            // With a reference frame the first frame's tokens are repeated in front.
            let src_f = f.saturating_sub(c.num_ref);
            for hmn in 0..if multi { c.humans } else { 1 } {
                let src = (hmn * frames + src_f) * per_frame;
                let dst = (f * na + hmn * cfg.context_tokens) * cfg.output_dim;
                audio[dst..dst + per_frame].copy_from_slice(&tok[src..src + per_frame]);
            }
        }
        // Rotary tables.
        let times: Vec<f32> = if c.num_ref > 0 {
            std::iter::once(c.ref_img_index as f32)
                .chain((0..n_t - c.num_ref).map(|v| v as f32))
                .collect()
        } else {
            (0..n_t).map(|v| v as f32).collect()
        };
        let (cos, sin) = rope_3d(&times, gh, gw, self.head_dim());
        let mask_frames = if c.num_cond > 1 && c.mask_frame_range > 0 {
            let start = c.ref_img_index as i64 - c.mask_frame_range as i64 - c.num_cond as i64 + c.num_ref as i64;
            let end = c.ref_img_index as i64 + c.mask_frame_range as i64 - c.num_cond as i64 + c.num_ref as i64 + 1;
            (start >= 0 && end > start && end <= (n_t - c.num_cond) as i64).then_some((start as usize, end as usize))
        } else {
            None
        };
        let token_masks = match c.ref_masks {
            Some((m, n, h, w)) => {
                if c.num_cond == 0 {
                    bail!("several speakers need a reference image (image to video or continuation)");
                }
                if m.len() != n * h * w || n < 2 {
                    bail!("speaker regions: {} values for {n} masks of {h}x{w}", m.len());
                }
                Some((token_masks(m, n, h, w, gh, gw), n))
            }
            None => None,
        };
        Ok(Ctx {
            n_t,
            s: gh * gw,
            grid: (gh, gw),
            temb,
            text,
            text_len: c.text_len,
            audio,
            na,
            cos,
            sin,
            num_cond: c.num_cond,
            mask_frames,
            token_masks,
            humans: c.humans,
            lora: c.lora,
        })
    }

    fn lin(&self, m: &Mat, name: &str, x: &[f32], rows: usize, lora: &Active) -> Vec<f32> {
        let mut y = m.forward(x, rows);
        let pairs = lora.of(name);
        if !pairs.is_empty() {
            add_side(&mut y, x, rows, &pairs);
        }
        y
    }

    /// The AdaLN sets of block `i`: `[n_t, 6 H]` and, for the audio branch, `[n_t, 3 H]`.
    pub(crate) fn block_mods(&self, i: usize, ctx: &Ctx) -> (Vec<f32>, Vec<f32>) {
        let b = &self.blocks[i];
        let st: Vec<f32> = ctx.temb.iter().map(|&v| silu(v)).collect();
        (b.ada.forward(&st, ctx.n_t), b.audio_ada.forward(&st, ctx.n_t))
    }

    /// Block `i` on the CPU.
    fn block(&self, i: usize, x: &mut [f32], ctx: &Ctx) {
        let b = &self.blocks[i];
        let cfg = &self.cfg;
        let (h, heads, hd) = (cfg.hidden_size, cfg.num_heads, self.head_dim());
        let (s, n_t) = (ctx.s, ctx.n_t);
        let n = n_t * s;
        let nc = ctx.num_cond;
        let name = |m: &str| format!("blocks.{i}.{m}");
        let (mods, amods) = self.block_mods(i, ctx);

        // Self-attention.
        let xm = norm_mod(x, h, s, &mods, 6 * h, 0, h, 0);
        let qkv = self.lin(&b.qkv, &name("attn.qkv"), &xm, n, &ctx.lora);
        drop(xm);
        let q = head_norm(&qkv, 3 * h, 0, heads, hd, Some(&b.q_norm), Some((&ctx.cos, &ctx.sin)));
        let k = head_norm(&qkv, 3 * h, 1, heads, hd, Some(&b.k_norm), Some((&ctx.cos, &ctx.sin)));
        let v = head_norm(&qkv, 3 * h, 2, heads, hd, None, None);
        drop(qkv);
        let mut a = vec![0f32; n * h];
        for (qr, kr) in ctx.segments() {
            let o = attention(
                &q[qr.start * h..qr.end * h],
                &k[kr.start * h..kr.end * h],
                &v[kr.start * h..kr.end * h],
                qr.len(),
                kr.len(),
                heads,
                hd,
                false,
            );
            a[qr.start * h..qr.end * h].copy_from_slice(&o);
        }
        // Several speakers: where each noisy token looks in the reference frame.
        let map = ctx
            .token_masks
            .as_ref()
            .map(|(m, nm)| (ref_attn_map(&q[nc * s * h..], &k[..s * h], m, *nm, heads, hd), *nm));
        drop((q, k, v));
        let a = self.lin(&b.attn_proj, &name("attn.proj"), &a, n, &ctx.lora);
        add_gated(x, &a, h, s, &mods, 6 * h, 2 * h, 0);
        drop(a);

        let nn = n - nc * s;
        if nn > 0 {
            // Text cross-attention (noisy frames only, no gate).
            let xn = norm_affine(&x[nc * s * h..], &b.pre_crs, 1e-6);
            let q = self.lin(&b.cq, &name("cross_attn.q_linear"), &xn, nn, &ctx.lora);
            drop(xn);
            let q = head_norm(&q, h, 0, heads, hd, Some(&b.cq_norm), None);
            let kv = self.lin(&b.ckv, &name("cross_attn.kv_linear"), &ctx.text, ctx.text_len, &ctx.lora);
            let k = head_norm(&kv, 2 * h, 0, heads, hd, Some(&b.ck_norm), None);
            let v = head_norm(&kv, 2 * h, 1, heads, hd, None, None);
            let o = attention(&q, &k, &v, nn, ctx.text_len, heads, hd, false);
            drop((q, k, v, kv));
            let o = self.lin(&b.cproj, &name("cross_attn.proj"), &o, nn, &ctx.lora);
            crate::kernels::add_inplace(&mut x[nc * s * h..], &o);
            drop(o);

            // Audio cross-attention: each frame's tokens against that frame's audio tokens.
            let xn = norm_affine(&x[nc * s * h..], &b.pre_video, 1e-6);
            let q = b.aq.forward(&xn, nn);
            drop(xn);
            let mut q = head_norm(&q, h, 0, heads, hd, Some(&b.aq_norm), None);
            let (na, od) = (ctx.na, cfg.output_dim);
            let cond = &ctx.audio[nc * na * od..];
            let cond_n;
            let cond = match &b.pre_audio {
                Some(pre) => {
                    cond_n = norm_affine(cond, pre, 1e-6);
                    &cond_n[..]
                }
                None => cond,
            };
            let frames = n_t - nc;
            let kv = b.akv.forward(cond, frames * na);
            let mut k = head_norm(&kv, 2 * h, 0, heads, hd, Some(&b.ak_norm), None);
            let v = head_norm(&kv, 2 * h, 1, heads, hd, None, None);
            if let Some((map, nm)) = &map
                && ctx.humans >= 2
            {
                let (pos, kpos) = speaker_positions(map, *nm, na, ctx.humans, cfg);
                let (c, sn) = rope_1d(&pos, hd);
                rotate(&mut q, heads, hd, &c, &sn);
                let all: Vec<f32> = (0..frames).flat_map(|_| kpos.iter().copied()).collect();
                let (c, sn) = rope_1d(&all, hd);
                rotate(&mut k, heads, hd, &c, &sn);
            }
            let mut o = vec![0f32; nn * h];
            for f in 0..frames {
                let of = attention(
                    &q[f * s * h..(f + 1) * s * h],
                    &k[f * na * h..(f + 1) * na * h],
                    &v[f * na * h..(f + 1) * na * h],
                    s,
                    na,
                    heads,
                    hd,
                    false,
                );
                o[f * s * h..(f + 1) * s * h].copy_from_slice(&of);
            }
            drop((q, k, v, kv));
            let o = b.aproj.forward(&o, nn);
            let o = norm_mod(&o, h, s, &amods, 3 * h, 0, h, nc);
            add_gated(x, &o, h, s, &amods, 3 * h, 2 * h, nc);
        }

        // Feed-forward.
        let xm = norm_mod(x, h, s, &mods, 6 * h, 3 * h, 4 * h, 0);
        let mut g = self.lin(&b.w1, &name("ffn.w1"), &xm, n, &ctx.lora);
        let u = self.lin(&b.w3, &name("ffn.w3"), &xm, n, &ctx.lora);
        drop(xm);
        for_chunks(&mut g, 1 << 14, true, |i, c| {
            for (v, &uu) in c.iter_mut().zip(&u[i << 14..]) {
                *v = silu(*v) * uu;
            }
        });
        drop(u);
        let o = self.lin(&b.w2, &name("ffn.w2"), &g, n, &ctx.lora);
        add_gated(x, &o, h, s, &mods, 6 * h, 5 * h, 0);
    }

    /// The block stack over the tokens `x` (`[n, hidden]`) on the CPU.
    pub(crate) fn blocks_cpu(&self, x: &mut [f32], ctx: &Ctx) {
        for i in 0..self.blocks.len() {
            self.block(i, x, ctx);
        }
    }

    /// The velocity for latents `x` (`[C, n_t, h, w]`) at `timesteps` (one per latent frame, 0
    /// to 1000): `[C, n_t, h, w]`.
    pub fn forward(&self, x: &[f32], shape: (usize, usize, usize), timesteps: &[f32], c: &Cond) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let (n_t, lh, lw) = shape;
        let (ph, pw) = (cfg.patch_size[1], cfg.patch_size[2]);
        let ci = cfg.in_channels;
        if x.len() != ci * n_t * lh * lw || !lh.is_multiple_of(ph) || !lw.is_multiple_of(pw) {
            bail!("latents: {} values for {ci} x {n_t} x {lh} x {lw}", x.len());
        }
        let ctx = self.context(shape, timesteps, c)?;
        let (gh, gw) = ctx.grid;
        let (s, n) = (gh * gw, n_t * gh * gw);
        let h = cfg.hidden_size;
        // Patches: (channel, patch row, patch column) per token.
        let pd = ci * ph * pw;
        let mut patches = vec![0f32; n * pd];
        for_chunks(&mut patches, pd, n * pd >= 1 << 14, |i, o| {
            let (f, r) = (i / s, i % s);
            let (gy, gx) = (r / gw, r % gw);
            for ch in 0..ci {
                for py in 0..ph {
                    for px in 0..pw {
                        o[(ch * ph + py) * pw + px] = x[((ch * n_t + f) * lh + gy * ph + py) * lw + gx * pw + px];
                    }
                }
            }
        });
        let mut tok = self.x_embed.forward(&patches, n);
        drop(patches);
        if !self.device.run(self, &mut tok, &ctx) {
            self.blocks_cpu(&mut tok, &ctx);
        }
        // Final layer.
        let st: Vec<f32> = ctx.temb.iter().map(|&v| silu(v)).collect();
        let fm = self.final_ada.forward(&st, n_t);
        let xm = norm_mod(&tok, h, s, &fm, 2 * h, 0, h, 0);
        let y = self.final_lin.forward(&xm, n);
        let co = cfg.out_channels;
        let mut out = vec![0f32; co * n_t * lh * lw];
        for i in 0..n {
            let (f, r) = (i / s, i % s);
            let (gy, gx) = (r / gw, r % gw);
            for py in 0..ph {
                for px in 0..pw {
                    for ch in 0..co {
                        out[((ch * n_t + f) * lh + gy * ph + py) * lw + gx * pw + px] = y[i * ph * pw * co + (py * pw + px) * co + ch];
                    }
                }
            }
        }
        Ok(out)
    }
}
