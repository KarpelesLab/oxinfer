//! Inworld Voice Isolator (`inworld/voice-isolator`): 48 kHz speech isolation (denoising), audio
//! in, the isolated voice out, written from the reference package `inworld-voice-isolation`
//! (`model.py`, `streaming.py`, `isolator.py`; Apache-2.0).
//!
//! Two checkpoints share the architecture: `denoiser.safetensors` (`voice-isolator-v1`,
//! offline, two-sided, 64M parameters) and `denoiser_realtime.safetensors`
//! (`voice-isolator-v1-realtime`, causal with 0, 20 or 40 ms of trained look-ahead, 58M). The
//! architecture is read from the checkpoint's `config` metadata.
//!
//! The network, a band-split dual-path model over the STFT (n_fft 2048, hop 480, Hann window):
//! - each frame's spectrum is cut into 46 bands (narrow at low frequency); band `b`'s bins
//!   (re, im interleaved) go through a LayerNorm and a linear layer to a 512-wide token, plus a
//!   band embedding (and, realtime, the look-ahead position's embedding);
//! - seven blocks alternate time attention (pre-norm transformer along the frames of each band:
//!   rotary positions, "exclusive" self attention that removes each output's component along
//!   its own value; then a transformer across the bands of each frame) and dual-path Mamba
//!   (a selective state-space layer along time, bidirectional offline, then a bidirectional
//!   one across bands; state 16, conv 4, expand 2);
//! - per-band heads (LayerNorm, linear, GELU, linear) give each bin a bounded complex mask
//!   (`2 tanh`) and a bounded residual (at most `correction_bound` times the bin's magnitude);
//!   the corrected spectrum's inverse STFT is added to the input.
//!
//! Offline, a clip runs in windows of the checkpoint's `chunk_seconds` (8 s) overlapping by
//! `overlap_seconds` (2 s) with a linear crossfade ([`VoiceIsolator::isolate`]). The realtime
//! checkpoint runs as a stream ([`IsolatorStream`]): samples in pieces of any size, a frame is
//! computed as soon as its analysis window is complete (blocks of up to `block_frames`), the
//! time attention over a cache of past keys within the trained attention window (1200 frames),
//! the time Mamba's state carried, the overlap-add released as frames are synthesized; the
//! decoder's output for frame `t` is read at frame `t + lookahead`. By default (as the
//! reference's `VoiceIsolator`) the attention never looks back across 6 s windows: a second
//! lane starts 2 s before each boundary with the Mamba states cloned and an empty attention
//! cache, and the decoder outputs blend linearly over the crossfade. `isolate` with the
//! realtime checkpoint runs that stream over the clip.
//!
//! Everything runs in f32 (the reference's precision off CUDA); the matrices stay f32 by default
//! (`IsolatorOptions::f32_weights`) and run through the shared matmul driver (the GPU backend
//! for large inputs). The selective scan runs on the CPU (vectorized over the 16 states). With an
//! accelerator that has them ([`crate::gpu::vi`]: Metal), a stream's steps run whole on the
//! device instead (encoders, every block of each lane, decoder heads in one submission, the
//! lanes' states kept there), still in f32.

use crate::ggml::{GgmlType, GgufMatrix};
use crate::gpu::vi;
use crate::kernels::{dot, dots, fast_exp, for_chunks, weighted_values};
use crate::model::{Lin, linear};
use crate::platform::DevicePref;
use crate::quant::PageVec;
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The checkpoint files of the Hugging Face repository.
pub const OFFLINE_FILE: &str = "denoiser.safetensors";
pub const REALTIME_FILE: &str = "denoiser_realtime.safetensors";
/// The repository both checkpoints come from.
pub const REPO: &str = "inworld/voice-isolator";
/// The model rate.
pub const SAMPLE_RATE: u32 = 48_000;
/// The reference's streaming windows (`STREAM_WINDOW_FRAMES`, `STREAM_CROSSFADE_FRAMES`).
pub const STREAM_WINDOW_FRAMES: usize = 600;
pub const STREAM_CROSSFADE_FRAMES: usize = 200;

fn d_one() -> usize {
    1
}
fn d_true() -> bool {
    true
}
fn d_time() -> String {
    "time".into()
}

/// The architecture, from the checkpoint's `config` metadata (`VoiceIsolatorConfig`).
#[derive(Clone, Debug, Deserialize)]
pub struct IsolatorConfig {
    pub width: usize,
    pub block_types: Vec<String>,
    pub num_heads: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub sample_rate: u32,
    pub band_plan: Vec<[f64; 2]>,
    pub correction_bound: f32,
    pub chunk_seconds: f64,
    pub overlap_seconds: f64,
    #[serde(default = "d_true")]
    pub exclusive_self_attention: bool,
    #[serde(default = "d_time")]
    pub exclusive_axes: String,
    pub mamba_state: usize,
    pub mamba_conv: usize,
    pub mamba_expand: usize,
    #[serde(default)]
    pub causal: bool,
    #[serde(default)]
    pub lookahead_frames: usize,
    #[serde(default)]
    pub lookahead_frames_choices: Vec<usize>,
    #[serde(default)]
    pub dual_mode: bool,
    #[serde(default)]
    pub mode_film: bool,
    #[serde(default)]
    pub attention_window_frames: usize,
    #[serde(default = "d_one")]
    pub offline_passes: usize,
}

impl IsolatorConfig {
    fn check(&self) -> Result<()> {
        if self.dual_mode || self.mode_film {
            bail!("dual-mode / mode-FiLM voice isolator checkpoints are not supported (the released files are neither)");
        }
        if self.width == 0 || self.num_heads == 0 || !self.width.is_multiple_of(2 * self.num_heads) {
            bail!(
                "width {} must be a positive multiple of twice num_heads {}",
                self.width,
                self.num_heads
            );
        }
        if self.n_fft < 4 || !self.n_fft.is_power_of_two() || self.hop_length == 0 || self.hop_length > self.n_fft / 2 {
            bail!(
                "n_fft {} must be a power of two and hop {} within (0, n_fft / 2]",
                self.n_fft,
                self.hop_length
            );
        }
        if let Some(k) = self.block_types.iter().find(|k| *k != "transformer" && *k != "mamba") {
            bail!("unknown block type {k:?}");
        }
        if !["both", "time", "band"].contains(&self.exclusive_axes.as_str()) {
            bail!("exclusive_axes {:?}", self.exclusive_axes);
        }
        if self.mamba_conv == 0 || self.mamba_state == 0 || self.mamba_expand == 0 {
            bail!("Mamba dimensions must be positive");
        }
        if !self.lookahead_frames_choices.is_empty() && !self.lookahead_frames_choices.contains(&self.lookahead_frames) {
            bail!(
                "lookahead_frames {} not among {:?}",
                self.lookahead_frames,
                self.lookahead_frames_choices
            );
        }
        if self.offline_passes == 0 {
            bail!("offline_passes must be at least 1");
        }
        Ok(())
    }

    /// The look-ahead positions the checkpoint was trained at (none offline).
    pub fn lookahead_choices(&self) -> Vec<usize> {
        if !self.lookahead_frames_choices.is_empty() {
            self.lookahead_frames_choices.clone()
        } else if self.causal {
            vec![self.lookahead_frames]
        } else {
            Vec::new()
        }
    }
}

/// Python's `round` (half to even).
fn round_half_even(x: f64) -> i64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 {
        2 * (x / 2.0).round() as i64
    } else {
        r as i64
    }
}

/// Bin ranges `[start, end)` of the bands (`band_edges`).
pub fn band_edges(cfg: &IsolatorConfig) -> Vec<(usize, usize)> {
    let n_bins = cfg.n_fft / 2 + 1;
    let hz = cfg.sample_rate as f64 / cfg.n_fft as f64;
    let mut edges = Vec::new();
    let mut start = 0usize;
    for &[w_hz, upto_hz] in &cfg.band_plan {
        let width = round_half_even(w_hz / hz).max(1) as usize;
        let upto = (round_half_even(upto_hz / hz).max(0) as usize).min(n_bins);
        while start < upto {
            let end = (start + width).min(upto);
            edges.push((start, end));
            start = end;
        }
    }
    if start < n_bins {
        edges.push((start, n_bins));
    }
    edges
}

/// Loader options. Keys for [`IsolatorOptions::set`]: `mode` (`offline` / `streaming`, which
/// checkpoint a repo id loads), `f32_weights` / `precision` (`f32` or `bf16`), `device`,
/// `threads`, `cache_dir`.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct IsolatorOptions {
    /// Which checkpoint of the repository to load: the offline one (default) or the realtime one.
    pub mode: Mode,
    /// Keep the matrices in f32 (exact; default) rather than bf16.
    pub f32_weights: bool,
    /// Compute device (process-wide; set by the first model loaded).
    pub device: DevicePref,
    /// CPU worker threads (process-wide).
    pub threads: Option<usize>,
    /// Where repo ids are downloaded.
    pub cache_dir: Option<PathBuf>,
}

impl Default for IsolatorOptions {
    fn default() -> Self {
        Self {
            mode: Mode::Offline,
            f32_weights: true,
            device: DevicePref::Auto,
            threads: None,
            cache_dir: None,
        }
    }
}

fn parse_bool(key: &str, v: &str) -> Result<bool> {
    match v {
        "" | "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => bail!("{key}: expected a boolean, got {v:?}"),
    }
}

impl IsolatorOptions {
    pub const KEYS: &'static [&'static str] = &["mode", "f32_weights", "precision", "device", "threads", "cache_dir"];

    /// Sets an option by key (see [`IsolatorOptions::KEYS`]).
    pub fn set(&mut self, key: &str, value: &str) -> Result<()> {
        match key {
            "mode" => self.mode = Mode::parse(value)?,
            "f32_weights" => self.f32_weights = parse_bool(key, value)?,
            "precision" | "quant" => {
                self.f32_weights = match value {
                    "f32" | "fp32" | "float32" => true,
                    "bf16" => false,
                    _ => bail!("precision: f32 or bf16, got {value:?}"),
                }
            }
            "device" => self.device = DevicePref::parse(value).with_context(|| format!("device: unknown device {value:?}"))?,
            "threads" => self.threads = Some(value.parse().with_context(|| format!("threads: {value:?}"))?),
            "cache_dir" => self.cache_dir = Some(PathBuf::from(value)),
            _ => bail!("unknown voice isolator option {key:?} (known: {})", Self::KEYS.join(", ")),
        }
        Ok(())
    }
}

/// Which checkpoint (offline or the realtime stream).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    #[default]
    Offline,
    Streaming,
}

impl Mode {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "offline" | "v1" | "voice-isolator-v1" => Mode::Offline,
            "streaming" | "stream" | "realtime" | "online" | "voice-isolator-v1-realtime" => Mode::Streaming,
            _ => bail!("mode: offline or streaming, got {s:?}"),
        })
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Offline => "offline",
            Mode::Streaming => "streaming",
        }
    }
}

/// Options of one isolation. Deserializes from JSON with every field optional (the C API's and
/// the server's format), e.g. `{"lookahead_frames": 4, "window_frames": 0}`.
#[derive(Clone, Debug, Default, serde::Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct IsolateOptions {
    /// Offline: the render window in seconds (default: the checkpoint's, 8 s).
    pub chunk_seconds: Option<f64>,
    /// Offline: the crossfade between windows in seconds (default: the checkpoint's, 2 s).
    pub overlap_seconds: Option<f64>,
    /// Offline: passes of the model over its own output (default: the checkpoint's, 1).
    pub offline_passes: Option<usize>,
    /// Streaming: the look-ahead in frames (10 ms each), one of the trained positions (realtime
    /// checkpoint: 0, 2 or 4; default 2).
    pub lookahead_frames: Option<usize>,
    /// Streaming: frames computed together, a throughput knob (default 4 for a live stream; a
    /// whole clip runs in blocks of 64, the same output).
    pub block_frames: Option<usize>,
    /// Streaming: the time attention's past in frames (default: the checkpoint's attention
    /// window, 1200).
    pub cache_frames: Option<usize>,
    /// Streaming: windows of the time attention (default 600 frames; 0 turns them off: the plain
    /// stream, the attention looking back over its whole cache).
    pub window_frames: Option<usize>,
    /// Streaming: the crossfade before each window boundary (default 200 frames).
    pub crossfade_frames: Option<usize>,
}

impl IsolateOptions {
    pub const KEYS: &'static [&'static str] = &[
        "chunk_seconds",
        "overlap_seconds",
        "offline_passes",
        "lookahead_frames",
        "block_frames",
        "cache_frames",
        "window_frames",
        "crossfade_frames",
    ];

    /// Sets an option by key (see [`IsolateOptions::KEYS`]; `lookahead` for `lookahead_frames`).
    pub fn set(&mut self, key: &str, value: &str) -> Result<()> {
        let f = |v: &str| -> Result<f64> { v.parse().with_context(|| format!("{key}: expected a number, got {v:?}")) };
        let n = |v: &str| -> Result<usize> { v.parse().with_context(|| format!("{key}: expected a whole number, got {v:?}")) };
        match key {
            "chunk_seconds" => self.chunk_seconds = Some(f(value)?),
            "overlap_seconds" => self.overlap_seconds = Some(f(value)?),
            "offline_passes" | "passes" => self.offline_passes = Some(n(value)?),
            "lookahead_frames" | "lookahead" => self.lookahead_frames = Some(n(value)?),
            "block_frames" => self.block_frames = Some(n(value)?),
            "cache_frames" => self.cache_frames = Some(n(value)?),
            "window_frames" => self.window_frames = Some(n(value)?),
            "crossfade_frames" => self.crossfade_frames = Some(n(value)?),
            _ => bail!("unknown isolation option {key:?} (known: {})", Self::KEYS.join(", ")),
        }
        Ok(())
    }
}

/// What a loaded isolator is.
#[derive(Clone, Debug, serde::Serialize)]
pub struct IsolatorInfo {
    pub path: PathBuf,
    /// `voice-isolator-v1` or `voice-isolator-v1-realtime` (for the released files).
    pub name: String,
    pub mode: Mode,
    pub params: usize,
    pub width: usize,
    pub blocks: usize,
    pub bands: usize,
    pub sample_rate: u32,
    pub lookahead_choices: Vec<usize>,
    pub default_lookahead: usize,
    pub attention_window_frames: usize,
    pub f32_weights: bool,
    pub device: String,
    pub load_secs: f64,
}

// ---------------------------------------------------------------------------------------------
// Layers.

/// A linear layer with its bias.
struct Dense {
    w: Lin<'static>,
    b: Option<Vec<f32>>,
    n_out: usize,
    n_in: usize,
}

impl Dense {
    fn new(w: &[f32], b: Option<Vec<f32>>, n_out: usize, n_in: usize, f32w: bool) -> Result<Self> {
        if w.len() != n_out * n_in {
            bail!("weight of {} values, expected {n_out} x {n_in}", w.len());
        }
        let w = if f32w {
            let bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
            Lin::gguf(GgufMatrix::from_bytes(GgmlType::F32, &bytes, n_out, n_in)?)
        } else {
            let mut v = PageVec::<u16>::zeroed(w.len());
            for (o, &x) in v.iter_mut().zip(w) {
                *o = half::bf16::from_f32(x).to_bits();
            }
            Lin::bf16_owned(v)
        };
        Ok(Self { w, b, n_out, n_in })
    }

    /// The weights as f32 `[n_out][n_in]` (borrowed when stored as f32).
    fn f32w(&self) -> std::borrow::Cow<'_, [f32]> {
        match &self.w.w {
            crate::model::Weight::Bf16(v, _) => v.iter().map(|&u| f32::from_bits((u as u32) << 16)).collect(),
            crate::model::Weight::Gguf(m) if m.kind == GgmlType::F32 && !m.x4 && (m.data.as_ptr() as usize).is_multiple_of(4) => {
                // SAFETY: the matrix's bytes are n_out * n_in little-endian f32, aligned.
                std::borrow::Cow::Borrowed(unsafe { std::slice::from_raw_parts(m.data.as_ptr() as *const f32, m.data.len() / 4) })
            }
            crate::model::Weight::Gguf(m) => {
                let mut v = vec![0f32; self.n_out * self.n_in];
                m.dequantize_rows(0, self.n_out, &mut v);
                v.into()
            }
            _ => unreachable!("voice isolator weights are f32 or bf16"),
        }
    }

    fn dev(&self) -> vi::Dense<'_> {
        vi::Dense {
            w: self.f32w(),
            b: self.b.as_deref(),
            n_out: self.n_out,
            n_in: self.n_in,
        }
    }

    /// `x` `[t, n_in]` -> `[t, n_out]`.
    fn fwd(&self, x: &[f32], t: usize) -> Vec<f32> {
        let _p = crate::prof::phase(0, "vi matmul");
        let mut y = vec![0f32; t * self.n_out];
        if t == 0 {
            return y;
        }
        linear(&self.w, self.n_out, self.n_in, &x[..t * self.n_in], t, &mut y);
        if let Some(b) = &self.b {
            for_rows(&mut y, self.n_out, |_, r| {
                for (v, &bb) in r.iter_mut().zip(b) {
                    *v += bb;
                }
            });
        }
        y
    }
}

/// `f(row index, row)` over the `c`-wide rows of `y`, in parallel blocks of rows.
fn for_rows(y: &mut [f32], c: usize, f: impl Fn(usize, &mut [f32]) + Sync) {
    let rows = (16384 / c.max(1)).max(1);
    let par = y.len() > rows * c;
    for_chunks(y, rows * c, par, |b, block| {
        for (j, row) in block.chunks_mut(c).enumerate() {
            f(b * rows + j, row);
        }
    });
}

struct Norm {
    w: Vec<f32>,
    b: Vec<f32>,
}

impl Norm {
    fn load(f: &ModelFiles, p: &str, n: usize) -> Result<Self> {
        let w = f.f32(&format!("{p}.weight"))?;
        let b = f.f32(&format!("{p}.bias"))?;
        if w.len() != n || b.len() != n {
            bail!("{p}: {} / {} values, expected {n}", w.len(), b.len());
        }
        Ok(Self { w, b })
    }

    fn row(&self, x: &[f32], out: &mut [f32]) {
        layer_norm(x, &self.w, &self.b, out);
    }

    fn dev(&self) -> vi::Norm<'_> {
        vi::Norm { w: &self.w, b: &self.b }
    }

    fn fwd(&self, x: &[f32]) -> Vec<f32> {
        let _p = crate::prof::phase(1, "vi norm");
        let n = self.w.len();
        let mut y = vec![0f32; x.len()];
        let rows = (16384 / n).max(1);
        let par = x.len() > rows * n;
        for_chunks(&mut y, rows * n, par, |bi, block| {
            for (j, out) in block.chunks_mut(n).enumerate() {
                let r = bi * rows + j;
                self.row(&x[r * n..(r + 1) * n], out);
            }
        });
        y
    }
}

/// LayerNorm (eps 1e-5) of one row.
fn layer_norm(x: &[f32], w: &[f32], b: &[f32], out: &mut [f32]) {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + 1e-5).sqrt();
    for (((o, &v), &ww), &bb) in out.iter_mut().zip(x).zip(w).zip(b) {
        *o = (v - mean) * inv * ww + bb;
    }
}

#[inline]
fn gelu(x: f32) -> f32 {
    crate::sd::ops::gelu_erf(x)
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + fast_exp(-x))
}

/// torch's softplus (threshold 20).
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

fn map_inplace(x: &mut [f32], f: impl Fn(f32) -> f32 + Sync) {
    for_chunks(x, 16384, x.len() > 32768, |_, c| c.iter_mut().for_each(|v| *v = f(*v)));
}

fn add_into(a: &mut [f32], b: &[f32]) {
    for_chunks(a, 16384, a.len() > 32768, |i, c| {
        for (v, &w) in c.iter_mut().zip(&b[i * 16384..]) {
            *v += w;
        }
    });
}

/// `y[b]` = `ls[b] x[b]` for the `t`-row blocks `x[b]` of every band (one grouped matmul on the
/// CPU), biases not added.
fn grouped(ls: &[Dense], x: &[f32], t: usize) -> Vec<f32> {
    let lins: Vec<&Lin> = ls.iter().map(|d| &d.w).collect();
    grouped_lins(&lins, x, t, ls[0].n_out, ls[0].n_in)
}

/// [`grouped`] over bare weights of one shape (blocks padded to the GEMM's row tile).
fn grouped_lins(lins: &[&Lin], x: &[f32], t: usize, n_out: usize, n_in: usize) -> Vec<f32> {
    let _p = crate::prof::phase(0, "vi matmul");
    let nb = lins.len();
    let tp = t.next_multiple_of(crate::kernels::TN);
    let segs: Vec<(usize, usize)> = (0..nb).map(|b| (b * tp, tp)).collect();
    let mut yp = vec![0f32; nb * tp * n_out];
    if tp == t {
        crate::model::linear_grouped(lins, &segs, n_out, n_in, x, &mut yp);
        return yp;
    }
    let mut xp = vec![0f32; nb * tp * n_in];
    for b in 0..nb {
        xp[b * tp * n_in..(b * tp + t) * n_in].copy_from_slice(&x[b * t * n_in..(b + 1) * t * n_in]);
    }
    crate::model::linear_grouped(lins, &segs, n_out, n_in, &xp, &mut yp);
    let mut y = vec![0f32; nb * t * n_out];
    for b in 0..nb {
        y[b * t * n_out..(b + 1) * t * n_out].copy_from_slice(&yp[b * tp * n_out..(b * tp + t) * n_out]);
    }
    y
}

/// Whether the per-band layers run as grouped CPU matmuls (small blocks, or no accelerator to
/// take each band's matmul).
fn use_grouped(nf: usize) -> bool {
    nf < 128 || crate::gpu::active().is_none()
}

fn dense(f: &ModelFiles, p: &str, n_out: usize, n_in: usize, bias: bool, f32w: bool) -> Result<Dense> {
    let w = f.f32(&format!("{p}.weight"))?;
    let b = if bias {
        let b = f.f32(&format!("{p}.bias"))?;
        if b.len() != n_out {
            bail!("{p}.bias: {} values, expected {n_out}", b.len());
        }
        Some(b)
    } else {
        None
    };
    Dense::new(&w, b, n_out, n_in, f32w).with_context(|| p.to_string())
}

/// `[a][b][w]` -> `[b][a][w]`.
fn transpose(x: &[f32], a: usize, b: usize, w: usize) -> Vec<f32> {
    let _p = crate::prof::phase(2, "vi transpose");
    let mut y = vec![0f32; x.len()];
    for_chunks(&mut y, b.max(1) * w, false, |_, _| {});
    let yp = SendPtr(y.as_mut_ptr());
    crate::pool::run(b, &|j| {
        let yp = &yp;
        for i in 0..a {
            // SAFETY: task j writes rows [j][*] only.
            let dst = unsafe { std::slice::from_raw_parts_mut(yp.0.add((j * a + i) * w), w) };
            dst.copy_from_slice(&x[(i * b + j) * w..(i * b + j + 1) * w]);
        }
    });
    y
}

#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

// ---------------------------------------------------------------------------------------------
// Attention.

/// Rotary tables: `angle(pos, i) = pos * 10000^(-i / half)`.
struct Rope {
    inv: Vec<f32>,
}

impl Rope {
    fn new(head_dim: usize) -> Self {
        let half = head_dim / 2;
        Self {
            inv: (0..half).map(|i| 10000f32.powf(-(i as f32) / half as f32)).collect(),
        }
    }
    /// Rotates `v` (`[head_dim]`) to position `pos` (`apply_rope`: halves rotated); `sign` -1
    /// rotates back.
    fn rotate(&self, v: &mut [f32], pos: usize, sign: f32) {
        let half = self.inv.len();
        for i in 0..half {
            let (s, c) = ((pos as f32 * self.inv[i]) as f64).sin_cos();
            let (s, c) = (sign * s as f32, c as f32);
            let (x1, x2) = (v[i], v[i + half]);
            v[i] = x1 * c - x2 * s;
            v[i + half] = x1 * s + x2 * c;
        }
    }

    /// The `(cos, sign * sin)` pairs of position `pos`.
    fn table(&self, pos: usize, sign: f32) -> Vec<(f32, f32)> {
        self.inv
            .iter()
            .map(|&f| {
                let (s, c) = ((pos as f32 * f) as f64).sin_cos();
                (c as f32, sign * s as f32)
            })
            .collect()
    }

    /// Rotates `v` by a [`Self::table`].
    fn apply(tab: &[(f32, f32)], v: &mut [f32]) {
        let half = tab.len();
        for (i, &(c, s)) in tab.iter().enumerate() {
            let (x1, x2) = (v[i], v[i + half]);
            v[i] = x1 * c - x2 * s;
            v[i + half] = x1 * s + x2 * c;
        }
    }
}

/// `o -= (o . v) / max(v . v, 1e-6) v`: the output's component along its own value removed.
fn exclusive(o: &mut [f32], v: &[f32]) {
    let ov = dot(o, v);
    let vv = dot(v, v).max(1e-6);
    let c = ov / vv;
    for (a, &b) in o.iter_mut().zip(v) {
        *a -= c * b;
    }
}

/// Small softmax attention: `out` = softmax(q . keys / sqrt(d)) values for one query against
/// `n` keys `stride` apart.
fn attend_one(q: &[f32], keys: &[f32], values: &[f32], stride: usize, n: usize, scores: &mut Vec<f32>, out: &mut [f32]) {
    let d = q.len();
    scores.resize(n, 0.0);
    dots(q, keys, stride, &mut scores[..n]);
    let scale = 1.0 / (d as f32).sqrt();
    let mut m = f32::NEG_INFINITY;
    for s in scores[..n].iter_mut() {
        *s *= scale;
        m = m.max(*s);
    }
    let mut sum = 0f32;
    for s in scores[..n].iter_mut() {
        *s = fast_exp(*s - m);
        sum += *s;
    }
    let inv = 1.0 / sum;
    for s in scores[..n].iter_mut() {
        *s *= inv;
    }
    weighted_values(&scores[..n], values, stride, d, out);
}

/// One lane's time-attention past for one attention block: keys (rotated at their slot) and
/// values `[rows][heads][cap][head_dim]`, `end` slots used.
#[derive(Clone, Default)]
struct KvCache {
    k: Vec<f32>,
    v: Vec<f32>,
    cap: usize,
    end: usize,
}

impl KvCache {
    /// Makes room for `slots` slots (rows, heads and head size given), keeping the used ones.
    fn reserve(&mut self, slots: usize, rows: usize, heads: usize, hd: usize) {
        if slots <= self.cap {
            return;
        }
        let cap = slots.max(2 * self.cap).max(64);
        let mut k = vec![0f32; rows * heads * cap * hd];
        let mut v = vec![0f32; rows * heads * cap * hd];
        if self.end > 0 {
            for rh in 0..rows * heads {
                let n = self.end * hd;
                k[rh * cap * hd..rh * cap * hd + n].copy_from_slice(&self.k[rh * self.cap * hd..rh * self.cap * hd + n]);
                v[rh * cap * hd..rh * cap * hd + n].copy_from_slice(&self.v[rh * self.cap * hd..rh * self.cap * hd + n]);
            }
        }
        self.k = k;
        self.v = v;
        self.cap = cap;
    }

    /// `_compact`: the last `keep` keys and values to the front, the keys rotated back by the
    /// distance they moved.
    fn compact(&mut self, keep: usize, rows: usize, heads: usize, hd: usize, rope: &Rope) {
        let keep = keep.min(self.end);
        let shift = self.end - keep;
        if shift == 0 {
            return;
        }
        let cap = self.cap;
        let tab = rope.table(shift, -1.0);
        for rh in 0..rows * heads {
            let base = rh * cap * hd;
            for s in 0..keep {
                let (src, dst) = (base + (shift + s) * hd, base + s * hd);
                self.k.copy_within(src..src + hd, dst);
                Rope::apply(&tab, &mut self.k[dst..dst + hd]);
                self.v.copy_within(src..src + hd, dst);
            }
        }
        self.end = keep;
    }
}

/// The carried state of a causal time Mamba: per row the last `conv - 1` inputs of the conv and
/// the scan state.
#[derive(Clone, Default)]
struct MambaCarry {
    tail: Vec<f32>, // [rows][conv - 1][inner]
    ssm: Vec<f32>,  // [rows][inner][state]
}

#[derive(Clone)]
enum BlockState {
    Attn(KvCache),
    Mamba(MambaCarry),
}

/// One lane's state for every block.
#[derive(Clone)]
struct Lane {
    blocks: Vec<BlockState>,
}

impl Lane {
    /// The Mamba states cloned, the attention caches empty (a second lane's start).
    fn fork(&self) -> Self {
        Self {
            blocks: self
                .blocks
                .iter()
                .map(|b| match b {
                    BlockState::Attn(_) => BlockState::Attn(KvCache::default()),
                    m => m.clone(),
                })
                .collect(),
        }
    }
}

/// How a block sees time: two-sided over the whole chunk (offline checkpoint), or causal with
/// this lane state (realtime).
struct Causal<'a> {
    state: &'a mut BlockState,
    /// The attention's past in frames (`cache_frames`).
    cache_frames: usize,
    /// The compaction threshold (`capacity` of the reference's buffers).
    capacity: usize,
}

struct TimeAttn {
    norm1: Norm,
    qkv: Dense,
    proj: Dense,
    norm2: Norm,
    ffn1: Dense,
    ffn2: Dense,
    exclusive: bool,
}

struct BandAttn {
    norm1: Norm,
    in_proj: Dense,
    out_proj: Dense,
    norm2: Norm,
    lin1: Dense,
    lin2: Dense,
    exclusive: bool,
}

struct Mamba {
    in_proj: Dense,
    conv_w: Vec<f32>, // [k][inner]
    conv_b: Vec<f32>,
    x_proj: Dense,
    dt_proj: Dense, // no bias in the matmul: added in f32 after (dt_bias)
    dt_bias: Vec<f32>,
    a: Vec<f32>, // -exp(A_log) [inner][state]
    d: Vec<f32>,
    out_proj: Dense,
    inner: usize,
    state: usize,
    k: usize,
    dt_rank: usize,
}

struct BiMamba {
    fwd: Mamba,
    bwd: Option<Mamba>,
    output_proj: Dense,
    norm: Norm,
}

enum Block {
    Attn(Box<TimeAttn>, Box<BandAttn>),
    Mamba(Box<BiMamba>, Box<BiMamba>),
}

struct BandHead {
    norm: Norm,
    l1: Dense,
    l2: Dense,
}

/// The band encoders and decoders' last layers zero-padded to the widest band, for grouped
/// matmuls over every band at once (`_PackedBands`): one call per layer instead of one per band.
struct Packed {
    /// Encoder inputs per band (`2 * widest`).
    enc_in: usize,
    enc: Vec<Dense>, // [width][enc_in]
    /// Decoder outputs per band (`4 * widest`).
    dec_out: usize,
    l2: Vec<Dense>, // [dec_out][width]
}

/// The network of one checkpoint.
struct Net {
    packed: Packed,
    cfg: IsolatorConfig,
    edges: Vec<(usize, usize)>,
    enc_norm: Vec<Norm>,
    enc: Vec<Dense>,
    band_emb: Vec<f32>,         // [bands][width]
    look_emb: Option<Vec<f32>>, // [choices][width]
    blocks: Vec<Block>,
    dec_scale: Vec<f32>, // 1 + scale of the decoder's condition
    dec_shift: Vec<f32>,
    heads: Vec<BandHead>,
    window: Vec<f32>,
    rope: Rope,
    fft: Fft,
    params: usize,
    /// The realtime network's weights on the accelerator (made for the first stream).
    device: std::sync::OnceLock<Option<Arc<dyn vi::Device>>>,
}

fn load_mamba(f: &ModelFiles, p: &str, width: usize, cfg: &IsolatorConfig, f32w: bool) -> Result<Mamba> {
    let inner = cfg.mamba_expand * width;
    let state = cfg.mamba_state;
    let dt_rank = width.div_ceil(16);
    let k = cfg.mamba_conv;
    let conv_w = f.f32(&format!("{p}.conv1d.weight"))?;
    let conv_b = f.f32(&format!("{p}.conv1d.bias"))?;
    if conv_w.len() != inner * k || conv_b.len() != inner {
        bail!("{p}.conv1d: unexpected shape");
    }
    let a_log = f.f32(&format!("{p}.A_log"))?;
    let d = f.f32(&format!("{p}.D"))?;
    if a_log.len() != inner * state || d.len() != inner {
        bail!("{p}: A_log / D shapes");
    }
    let dt_w = f.f32(&format!("{p}.dt_proj.weight"))?;
    let dt_bias = f.f32(&format!("{p}.dt_proj.bias"))?;
    Ok(Mamba {
        in_proj: dense(f, &format!("{p}.in_proj"), 2 * inner, width, false, f32w)?,
        conv_w: (0..k * inner).map(|i| conv_w[(i % inner) * k + i / inner]).collect(),
        conv_b,
        x_proj: dense(f, &format!("{p}.x_proj"), dt_rank + 2 * state, inner, false, f32w)?,
        dt_proj: Dense::new(&dt_w, None, inner, dt_rank, f32w)?,
        dt_bias,
        a: a_log.iter().map(|v| -v.exp()).collect(),
        d,
        out_proj: dense(f, &format!("{p}.out_proj"), width, inner, false, f32w)?,
        inner,
        state,
        k,
        dt_rank,
    })
}

fn load_bimamba(f: &ModelFiles, p: &str, width: usize, cfg: &IsolatorConfig, causal: bool, f32w: bool) -> Result<BiMamba> {
    Ok(BiMamba {
        fwd: load_mamba(f, &format!("{p}.forward_blocks"), width, cfg, f32w)?,
        bwd: if causal {
            None
        } else {
            Some(load_mamba(f, &format!("{p}.backward_blocks"), width, cfg, f32w)?)
        },
        output_proj: dense(
            f,
            &format!("{p}.output_proj"),
            width,
            if causal { width } else { 2 * width },
            true,
            f32w,
        )?,
        norm: Norm::load(f, &format!("{p}.norm"), width)?,
    })
}

impl Net {
    fn load(files: &ModelFiles, cfg: IsolatorConfig, f32w: bool) -> Result<Self> {
        cfg.check()?;
        let w = cfg.width;
        let edges = band_edges(&cfg);
        let nb = edges.len();
        let mut enc_norm = Vec::with_capacity(nb);
        let mut enc = Vec::with_capacity(nb);
        for (b, &(s, e)) in edges.iter().enumerate() {
            let n_in = 2 * (e - s);
            enc_norm.push(Norm::load(files, &format!("encoders.{b}.0"), n_in)?);
            enc.push(dense(files, &format!("encoders.{b}.1"), w, n_in, true, f32w)?);
        }
        let band_emb = files.f32("band_embedding")?;
        if band_emb.len() != nb * w {
            bail!("band_embedding: {} values, expected {nb} x {w}", band_emb.len());
        }
        let look_emb = if cfg.lookahead_frames_choices.is_empty() {
            None
        } else {
            let e = files.f32("lookahead_embedding.weight")?;
            if e.len() != cfg.lookahead_frames_choices.len() * w {
                bail!("lookahead_embedding: unexpected shape");
            }
            Some(e)
        };
        let exc_time = cfg.exclusive_self_attention && (cfg.exclusive_axes == "both" || cfg.exclusive_axes == "time");
        let exc_band = cfg.exclusive_self_attention && (cfg.exclusive_axes == "both" || cfg.exclusive_axes == "band");
        let mut blocks = Vec::new();
        for (i, kind) in cfg.block_types.iter().enumerate() {
            let p = format!("blocks.{i}");
            blocks.push(if kind == "transformer" {
                let t = format!("{p}.time_transformer");
                let b = format!("{p}.band_transformer");
                Block::Attn(
                    Box::new(TimeAttn {
                        norm1: Norm::load(files, &format!("{t}.norm1"), w)?,
                        qkv: dense(files, &format!("{t}.attn.qkv"), 3 * w, w, true, f32w)?,
                        proj: dense(files, &format!("{t}.attn.proj"), w, w, true, f32w)?,
                        norm2: Norm::load(files, &format!("{t}.norm2"), w)?,
                        ffn1: dense(files, &format!("{t}.ffn.0"), 4 * w, w, true, f32w)?,
                        ffn2: dense(files, &format!("{t}.ffn.3"), w, 4 * w, true, f32w)?,
                        exclusive: exc_time,
                    }),
                    Box::new(BandAttn {
                        norm1: Norm::load(files, &format!("{b}.norm1"), w)?,
                        in_proj: {
                            let wt = files.f32(&format!("{b}.self_attn.in_proj_weight"))?;
                            let bs = files.f32(&format!("{b}.self_attn.in_proj_bias"))?;
                            Dense::new(&wt, Some(bs), 3 * w, w, f32w)?
                        },
                        out_proj: dense(files, &format!("{b}.self_attn.out_proj"), w, w, true, f32w)?,
                        norm2: Norm::load(files, &format!("{b}.norm2"), w)?,
                        lin1: dense(files, &format!("{b}.linear1"), 4 * w, w, true, f32w)?,
                        lin2: dense(files, &format!("{b}.linear2"), w, 4 * w, true, f32w)?,
                        exclusive: exc_band,
                    }),
                )
            } else {
                Block::Mamba(
                    Box::new(load_bimamba(files, &format!("{p}.time_mamba"), w, &cfg, cfg.causal, f32w)?),
                    Box::new(load_bimamba(files, &format!("{p}.freq_mamba"), w, &cfg, false, f32w)?),
                )
            });
        }
        // The decoder's FiLM of its fixed controls [1, 0].
        let cw = files.f32("denoise_decoder.condition.weight")?;
        let cb = files.f32("denoise_decoder.condition.bias")?;
        if cw.len() != 4 * w || cb.len() != 2 * w {
            bail!("denoise_decoder.condition: unexpected shape");
        }
        let cond: Vec<f32> = (0..2 * w).map(|o| cw[2 * o] + cb[o]).collect();
        let dec_scale = cond[..w].iter().map(|s| 1.0 + s).collect();
        let dec_shift = cond[w..].to_vec();
        let mut heads = Vec::with_capacity(nb);
        for (b, &(s, e)) in edges.iter().enumerate() {
            let p = format!("denoise_decoder.heads.{b}");
            heads.push(BandHead {
                norm: Norm::load(files, &format!("{p}.0"), w)?,
                l1: dense(files, &format!("{p}.1"), w, w, true, f32w)?,
                l2: dense(files, &format!("{p}.3"), 4 * (e - s), w, true, f32w)?,
            });
        }
        let n = cfg.n_fft;
        // torch.hann_window (periodic).
        let window = (0..n)
            .map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos()) as f32)
            .collect();
        let widest = edges.iter().map(|&(s, e)| e - s).max().unwrap_or(1);
        let (enc_in, dec_out) = (2 * widest, 4 * widest);
        let mut penc = Vec::with_capacity(nb);
        let mut pl2 = Vec::with_capacity(nb);
        for (b, &(s, e)) in edges.iter().enumerate() {
            let n_in = 2 * (e - s);
            let ew = files.f32(&format!("encoders.{b}.1.weight"))?;
            let mut pw = vec![0f32; w * enc_in];
            for o in 0..w {
                pw[o * enc_in..o * enc_in + n_in].copy_from_slice(&ew[o * n_in..(o + 1) * n_in]);
            }
            penc.push(Dense::new(&pw, None, w, enc_in, f32w)?);
            let n_out = 4 * (e - s);
            let lw = files.f32(&format!("denoise_decoder.heads.{b}.3.weight"))?;
            let mut pw = vec![0f32; dec_out * w];
            pw[..n_out * w].copy_from_slice(&lw);
            pl2.push(Dense::new(&pw, None, dec_out, w, f32w)?);
        }
        let packed = Packed {
            enc_in,
            enc: penc,
            dec_out,
            l2: pl2,
        };
        let params = files.metadata("num_params").and_then(|v| v.parse().ok()).unwrap_or(0);
        Ok(Self {
            packed,
            rope: Rope::new(w / cfg.num_heads),
            fft: Fft::new(n),
            edges,
            enc_norm,
            enc,
            band_emb,
            look_emb,
            blocks,
            dec_scale,
            dec_shift,
            heads,
            window,
            params,
            cfg,
            device: std::sync::OnceLock::new(),
        })
    }

    /// The accelerator's copy of the realtime network for whole stream steps, when there is one
    /// (`OXINFER_VI_GPU=0` keeps the streams on the CPU path).
    fn device(&self) -> Option<Arc<dyn vi::Device>> {
        if !self.cfg.causal || std::env::var("OXINFER_VI_GPU").is_ok_and(|v| v == "0") {
            return None;
        }
        let be = crate::gpu::active()?;
        self.device.get_or_init(|| be.vi_device(&self.dev_model())).clone()
    }

    /// The network as [`vi::Model`] describes it.
    fn dev_model(&self) -> vi::Model<'_> {
        fn mamba(m: &Mamba) -> vi::Mamba<'_> {
            vi::Mamba {
                in_proj: m.in_proj.dev(),
                conv_w: &m.conv_w,
                conv_b: &m.conv_b,
                x_proj: m.x_proj.dev(),
                dt_proj: m.dt_proj.dev(),
                dt_bias: &m.dt_bias,
                a: &m.a,
                d: &m.d,
                out_proj: m.out_proj.dev(),
            }
        }
        fn bi(b: &BiMamba) -> vi::BiMamba<'_> {
            vi::BiMamba {
                fwd: mamba(&b.fwd),
                bwd: b.bwd.as_ref().map(mamba),
                output_proj: b.output_proj.dev(),
                norm: b.norm.dev(),
            }
        }
        let blocks = self
            .blocks
            .iter()
            .map(|b| match b {
                Block::Attn(t, a) => vi::Block::Attn(
                    Box::new(vi::TimeAttn {
                        norm1: t.norm1.dev(),
                        qkv: t.qkv.dev(),
                        proj: t.proj.dev(),
                        norm2: t.norm2.dev(),
                        ffn1: t.ffn1.dev(),
                        ffn2: t.ffn2.dev(),
                        exclusive: t.exclusive,
                    }),
                    Box::new(vi::TimeAttn {
                        norm1: a.norm1.dev(),
                        qkv: a.in_proj.dev(),
                        proj: a.out_proj.dev(),
                        norm2: a.norm2.dev(),
                        ffn1: a.lin1.dev(),
                        ffn2: a.lin2.dev(),
                        exclusive: a.exclusive,
                    }),
                ),
                Block::Mamba(t, f) => vi::Block::Mamba(Box::new(bi(t)), Box::new(bi(f))),
            })
            .collect();
        vi::Model {
            width: self.cfg.width,
            heads: self.cfg.num_heads,
            bins: self.bins(),
            edges: &self.edges,
            enc_norm: self.enc_norm.iter().map(Norm::dev).collect(),
            enc: self.enc.iter().map(Dense::dev).collect(),
            band_emb: &self.band_emb,
            look_emb: self.look_emb.as_deref(),
            blocks,
            dec_scale: &self.dec_scale,
            dec_shift: &self.dec_shift,
            heads_dec: self
                .heads
                .iter()
                .map(|h| vi::Head {
                    norm: h.norm.dev(),
                    l1: h.l1.dev(),
                    l2: h.l2.dev(),
                })
                .collect(),
            window: self.cfg.attention_window_frames,
            rope_inv: &self.rope.inv,
        }
    }

    fn bands(&self) -> usize {
        self.edges.len()
    }

    fn bins(&self) -> usize {
        self.cfg.n_fft / 2 + 1
    }

    /// Band tokens `[bands][frames][width]` of spectra `[frames][bins]` (re, im pairs).
    fn encode(&self, spec: &[f32], nf: usize, look: Option<usize>) -> Vec<f32> {
        let (w, nb, bins) = (self.cfg.width, self.bands(), self.bins());
        let mut x;
        if use_grouped(nf) {
            let pi = self.packed.enc_in;
            let mut inp = vec![0f32; nb * nf * pi];
            for_rows(&mut inp, pi, |r, row| {
                let (b, f) = (r / nf, r % nf);
                let (s, e) = self.edges[b];
                let src = &spec[(f * bins + s) * 2..(f * bins + e) * 2];
                self.enc_norm[b].row(src, &mut row[..2 * (e - s)]);
            });
            x = grouped(&self.packed.enc, &inp, nf);
            for_rows(&mut x, w, |r, row| {
                for (v, &bb) in row.iter_mut().zip(self.enc[r / nf].b.as_deref().unwrap_or(&[])) {
                    *v += bb;
                }
            });
        } else {
            x = vec![0f32; nb * nf * w];
            for (b, &(s, e)) in self.edges.iter().enumerate() {
                let n_in = 2 * (e - s);
                let mut inp = vec![0f32; nf * n_in];
                for f in 0..nf {
                    let src = &spec[(f * bins + s) * 2..(f * bins + e) * 2];
                    self.enc_norm[b].row(src, &mut inp[f * n_in..(f + 1) * n_in]);
                }
                let y = self.enc[b].fwd(&inp, nf);
                x[b * nf * w..(b + 1) * nf * w].copy_from_slice(&y);
            }
        }
        let la = look.and_then(|l| {
            let i = self.cfg.lookahead_frames_choices.iter().position(|&c| c == l)?;
            Some(&self.look_emb.as_ref()?[i * w..(i + 1) * w])
        });
        for_rows(&mut x, w, |r, row| {
            let emb = &self.band_emb[(r / nf) * w..(r / nf + 1) * w];
            for (v, &e) in row.iter_mut().zip(emb) {
                *v += e;
            }
            if let Some(la) = la {
                for (v, &e) in row.iter_mut().zip(la) {
                    *v += e;
                }
            }
        });
        x
    }

    /// The decoder's output `[frames][bins][4]` of features `[bands][frames][width]`.
    fn decode(&self, x: &[f32], nf: usize) -> Vec<f32> {
        let (w, bins) = (self.cfg.width, self.bins());
        let mut raw = vec![0f32; nf * bins * 4];
        if use_grouped(nf) {
            let nb = self.bands();
            let mut h = vec![0f32; nb * nf * w];
            for_rows(&mut h, w, |r, row| {
                let src = &x[r * w..(r + 1) * w];
                let hd = &self.heads[r / nf];
                let t: Vec<f32> = src
                    .iter()
                    .zip(&self.dec_scale)
                    .zip(&self.dec_shift)
                    .map(|((v, sc), sh)| v * sc + sh)
                    .collect();
                hd.norm.row(&t, row);
            });
            let lins: Vec<&Lin> = self.heads.iter().map(|h| &h.l1.w).collect();
            let mut h1 = grouped_lins(&lins, &h, nf, w, w);
            for_rows(&mut h1, w, |r, row| {
                for (v, &bb) in row.iter_mut().zip(self.heads[r / nf].l1.b.as_deref().unwrap_or(&[])) {
                    *v = gelu(*v + bb);
                }
            });
            let po = self.packed.dec_out;
            let y = grouped(&self.packed.l2, &h1, nf);
            let rp = SendPtr(raw.as_mut_ptr());
            crate::pool::run(nb, &|b| {
                let rp = &rp;
                let (s, e) = self.edges[b];
                let n = 4 * (e - s);
                let bias = self.heads[b].l2.b.as_deref().unwrap_or(&[]);
                for f in 0..nf {
                    // SAFETY: band b's bins of every frame only.
                    let dst = unsafe { std::slice::from_raw_parts_mut(rp.0.add((f * bins + s) * 4), n) };
                    for ((d, &v), &bb) in dst.iter_mut().zip(&y[(b * nf + f) * po..]).zip(bias) {
                        *d = v + bb;
                    }
                }
            });
            return raw;
        }
        for (b, &(s, e)) in self.edges.iter().enumerate() {
            let mut h = x[b * nf * w..(b + 1) * nf * w].to_vec();
            for_rows(&mut h, w, |_, row| {
                for ((v, &sc), &sh) in row.iter_mut().zip(&self.dec_scale).zip(&self.dec_shift) {
                    *v = *v * sc + sh;
                }
            });
            let h = self.heads[b].norm.fwd(&h);
            let mut h = self.heads[b].l1.fwd(&h, nf);
            map_inplace(&mut h, gelu);
            let y = self.heads[b].l2.fwd(&h, nf);
            let n = 4 * (e - s);
            for f in 0..nf {
                raw[(f * bins + s) * 4..(f * bins + e) * 4].copy_from_slice(&y[f * n..(f + 1) * n]);
            }
        }
        raw
    }

    /// The backbone over band tokens `[bands][frames][width]`: two-sided (`lane` None) or causal
    /// with a lane's state.
    fn backbone(&self, x: &mut Vec<f32>, nf: usize, mut lane: Option<(&mut Lane, usize, usize)>) {
        for (i, block) in self.blocks.iter().enumerate() {
            let causal = lane.as_mut().map(|(l, cache_frames, capacity)| Causal {
                state: &mut l.blocks[i],
                cache_frames: *cache_frames,
                capacity: *capacity,
            });
            match block {
                Block::Attn(t, b) => {
                    self.time_attn(t, x, nf, causal);
                    self.band_attn(b, x, nf);
                }
                Block::Mamba(t, f) => {
                    self.time_mamba(t, x, nf, causal);
                    self.freq_mamba(f, x, nf);
                }
            }
        }
    }

    fn time_attn(&self, blk: &TimeAttn, x: &mut [f32], nf: usize, causal: Option<Causal>) {
        let (w, nb, nh) = (self.cfg.width, self.bands(), self.cfg.num_heads);
        let hd = w / nh;
        let n = nb * nf;
        let h = blk.norm1.fwd(x);
        let qkv = blk.qkv.fwd(&h, n);
        let mut out = vec![0f32; n * w];
        match causal {
            None => {
                // Two-sided: every band's heads as heads of one call, frame-major [frame][band][w].
                let mut q = vec![0f32; n * w];
                let mut k = vec![0f32; n * w];
                let mut v = vec![0f32; n * w];
                let (qp, kp, vp) = (SendPtr(q.as_mut_ptr()), SendPtr(k.as_mut_ptr()), SendPtr(v.as_mut_ptr()));
                crate::pool::run(n, &|r| {
                    let (qp, kp, vp) = (&qp, &kp, &vp);
                    let (b, f) = (r / nf, r % nf);
                    let src = &qkv[r * 3 * w..(r + 1) * 3 * w];
                    let at = (f * nb + b) * w;
                    // SAFETY: each task writes its own token's rows.
                    let (qd, kd, vd) = unsafe {
                        (
                            std::slice::from_raw_parts_mut(qp.0.add(at), w),
                            std::slice::from_raw_parts_mut(kp.0.add(at), w),
                            std::slice::from_raw_parts_mut(vp.0.add(at), w),
                        )
                    };
                    qd.copy_from_slice(&src[..w]);
                    kd.copy_from_slice(&src[w..2 * w]);
                    vd.copy_from_slice(&src[2 * w..]);
                    for hh in 0..nh {
                        self.rope.rotate(&mut qd[hh * hd..(hh + 1) * hd], f, 1.0);
                        self.rope.rotate(&mut kd[hh * hd..(hh + 1) * hd], f, 1.0);
                    }
                });
                let _p = crate::prof::phase(3, "vi time attention");
                let o = crate::sd::ops::attention(&q, &k, &v, nf, nf, nb * nh, hd, false);
                // Back to band-major, exclusive along the value.
                let op = SendPtr(out.as_mut_ptr());
                crate::pool::run(n, &|r| {
                    let op = &op;
                    let (b, f) = (r / nf, r % nf);
                    let at = (f * nb + b) * w;
                    // SAFETY: each task writes its own token's row.
                    let dst = unsafe { std::slice::from_raw_parts_mut(op.0.add(r * w), w) };
                    dst.copy_from_slice(&o[at..at + w]);
                    if blk.exclusive {
                        for hh in 0..nh {
                            exclusive(&mut dst[hh * hd..(hh + 1) * hd], &v[at + hh * hd..at + (hh + 1) * hd]);
                        }
                    }
                });
            }
            Some(c) => {
                let BlockState::Attn(cache) = c.state else {
                    unreachable!("attention block state")
                };
                if cache.end + nf > c.capacity {
                    cache.compact(c.cache_frames, nb, nh, hd, &self.rope);
                }
                cache.reserve(cache.end + nf, nb, nh, hd);
                let first = cache.end;
                let cap = cache.cap;
                // Write the new keys (rotated at their slots) and values.
                {
                    let (kp, vp) = (SendPtr(cache.k.as_mut_ptr()), SendPtr(cache.v.as_mut_ptr()));
                    crate::pool::run(nb * nh, &|rh| {
                        let (kp, vp) = (&kp, &vp);
                        let (b, hh) = (rh / nh, rh % nh);
                        for f in 0..nf {
                            let src = &qkv[(b * nf + f) * 3 * w..];
                            let at = (rh * cap + first + f) * hd;
                            // SAFETY: each task writes its own (row, head) slots.
                            let (kd, vd) = unsafe {
                                (
                                    std::slice::from_raw_parts_mut(kp.0.add(at), hd),
                                    std::slice::from_raw_parts_mut(vp.0.add(at), hd),
                                )
                            };
                            kd.copy_from_slice(&src[w + hh * hd..w + (hh + 1) * hd]);
                            self.rope.rotate(kd, first + f, 1.0);
                            vd.copy_from_slice(&src[2 * w + hh * hd..2 * w + (hh + 1) * hd]);
                        }
                    });
                }
                cache.end = first + nf;
                let past = first.min(c.cache_frames);
                let window = self.cfg.attention_window_frames;
                let cache: &KvCache = cache;
                let _p = crate::prof::phase(3, "vi time attention");
                // One task per (row, head, block of queries).
                let qb = 16usize;
                let nqb = nf.div_ceil(qb);
                let op = SendPtr(out.as_mut_ptr());
                crate::pool::run(nb * nh * nqb, &|i| {
                    let op = &op;
                    let (rh, j) = (i / nqb, i % nqb);
                    let (b, hh) = (rh / nh, rh % nh);
                    let mut scores = Vec::new();
                    let mut q = vec![0f32; hd];
                    for f in j * qb..((j + 1) * qb).min(nf) {
                        let qs = first + f;
                        let src = &qkv[(b * nf + f) * 3 * w..];
                        q.copy_from_slice(&src[hh * hd..(hh + 1) * hd]);
                        self.rope.rotate(&mut q, qs, 1.0);
                        let mut lo = first - past;
                        if window > 0 {
                            lo = lo.max((qs + 1).saturating_sub(window));
                        }
                        let cnt = qs + 1 - lo;
                        let base = (rh * cap + lo) * hd;
                        // SAFETY: each task writes its own (token, head) slices.
                        let dst = unsafe { std::slice::from_raw_parts_mut(op.0.add((b * nf + f) * w + hh * hd), hd) };
                        attend_one(&q, &cache.k[base..], &cache.v[base..], hd, cnt, &mut scores, dst);
                        if blk.exclusive {
                            exclusive(dst, &src[2 * w + hh * hd..2 * w + (hh + 1) * hd]);
                        }
                    }
                });
            }
        }
        let p = blk.proj.fwd(&out, n);
        add_into(x, &p);
        let h = blk.norm2.fwd(x);
        let mut h = blk.ffn1.fwd(&h, n);
        map_inplace(&mut h, gelu);
        let y = blk.ffn2.fwd(&h, n);
        add_into(x, &y);
    }

    /// The transformer across the bands of each frame, on the band-major layout: the bands are
    /// the sequence and each (frame, head) pair one head of a single attention call.
    fn band_attn(&self, blk: &BandAttn, x: &mut [f32], nf: usize) {
        let (w, nb, nh) = (self.cfg.width, self.bands(), self.cfg.num_heads);
        let hd = w / nh;
        let n = nb * nf;
        let h = blk.norm1.fwd(x);
        let qkv = blk.in_proj.fwd(&h, n);
        // Token (band b, frame f) -> row b, head block f: [band][frame][w] is already the layout.
        let mut q = vec![0f32; n * w];
        let mut k = vec![0f32; n * w];
        let mut v = vec![0f32; n * w];
        {
            let (qp, kp, vp) = (SendPtr(q.as_mut_ptr()), SendPtr(k.as_mut_ptr()), SendPtr(v.as_mut_ptr()));
            crate::pool::run(nb, &|b| {
                let (qp, kp, vp) = (&qp, &kp, &vp);
                for f in 0..nf {
                    let r = b * nf + f;
                    let src = &qkv[r * 3 * w..(r + 1) * 3 * w];
                    // SAFETY: band b's rows only.
                    unsafe {
                        std::slice::from_raw_parts_mut(qp.0.add(r * w), w).copy_from_slice(&src[..w]);
                        std::slice::from_raw_parts_mut(kp.0.add(r * w), w).copy_from_slice(&src[w..2 * w]);
                        std::slice::from_raw_parts_mut(vp.0.add(r * w), w).copy_from_slice(&src[2 * w..]);
                    }
                }
            });
        }
        let pa = crate::prof::phase(4, "vi band attention");
        let mut out = crate::sd::ops::attention(&q, &k, &v, nb, nb, nf * nh, hd, false);
        drop(pa);
        if blk.exclusive {
            for_rows(&mut out, hd, |r, o| exclusive(o, &v[r * hd..(r + 1) * hd]));
        }
        let p = blk.out_proj.fwd(&out, n);
        add_into(x, &p);
        let h = blk.norm2.fwd(x);
        let mut h = blk.lin1.fwd(&h, n);
        map_inplace(&mut h, gelu);
        let y = blk.lin2.fwd(&h, n);
        add_into(x, &y);
    }

    /// The time Mamba: `x += norm(output_proj(cat(fwd + x, bwd + x)))` two-sided, or
    /// `x += norm(output_proj(fwd + x))` causal with the carried state.
    fn time_mamba(&self, m: &BiMamba, x: &mut [f32], nf: usize, causal: Option<Causal>) {
        let nb = self.bands();
        let y = match causal {
            None => bi_mamba(m, x, nb, nf, None),
            Some(c) => {
                let BlockState::Mamba(carry) = c.state else {
                    unreachable!("Mamba block state")
                };
                bi_mamba(m, x, nb, nf, Some(carry))
            }
        };
        add_into(x, &y);
    }

    fn freq_mamba(&self, m: &BiMamba, x: &mut Vec<f32>, nf: usize) {
        let (w, nb) = (self.cfg.width, self.bands());
        let mut xf = transpose(x, nb, nf, w);
        let y = bi_mamba(m, &xf, nf, nb, None);
        add_into(&mut xf, &y);
        *x = transpose(&xf, nf, nb, w);
    }

    /// The complex change of the spectrum the decoder's output encodes, `[frames][bins]` pairs.
    fn correction(&self, raw: &[f32], spec: &[f32]) -> Vec<f32> {
        let bound = self.cfg.correction_bound;
        let mut out = vec![0f32; spec.len()];
        let par = out.len() > 4 * self.bins();
        for_chunks(&mut out, 2 * self.bins(), par, |f, row| {
            let bins = row.len() / 2;
            for k in 0..bins {
                let (sr, si) = (spec[(f * bins + k) * 2], spec[(f * bins + k) * 2 + 1]);
                let r = &raw[(f * bins + k) * 4..(f * bins + k) * 4 + 4];
                let (m0, m1) = (2.0 * r[0].tanh(), 2.0 * r[1].tanh());
                let (c0, c1) = (r[2].tanh(), r[3].tanh());
                let norm = (c0 * c0 + c1 * c1).sqrt().max(1.0);
                let mag = (sr * sr + si * si).sqrt();
                let s = bound * mag / norm;
                row[2 * k] = sr * m0 - si * m1 + c0 * s;
                row[2 * k + 1] = sr * m1 + si * m0 + c1 * s;
            }
        });
        out
    }

    /// The windowed spectrum of one frame of `n_fft` samples: `bins` (re, im) pairs.
    fn analyze(&self, seg: &[f32], out: &mut [f32]) {
        let n = self.cfg.n_fft;
        let mut re: Vec<f32> = seg.iter().zip(&self.window).map(|(a, b)| a * b).collect();
        let mut im = vec![0f32; n];
        self.fft.run(&mut re, &mut im, false);
        for k in 0..n / 2 + 1 {
            out[2 * k] = re[k];
            out[2 * k + 1] = im[k];
        }
    }

    /// One frame's inverse transform times the window (`irfft(delta) * window`).
    fn synth(&self, spec: &[f32], out: &mut [f32]) {
        let n = self.cfg.n_fft;
        let mut re = vec![0f32; n];
        let mut im = vec![0f32; n];
        for k in 0..=n / 2 {
            re[k] = spec[2 * k];
            im[k] = if k == 0 || k == n / 2 { 0.0 } else { spec[2 * k + 1] };
        }
        for k in 1..n / 2 {
            re[n - k] = re[k];
            im[n - k] = -im[k];
        }
        self.fft.run(&mut re, &mut im, true);
        let inv = 1.0 / n as f32;
        for ((o, &r), &w) in out.iter_mut().zip(&re).zip(&self.window) {
            *o = r * inv * w;
        }
    }

    /// `forward` of the offline (two-sided) model over one window of audio: the isolated voice.
    fn forward_offline(&self, audio: &[f32]) -> Vec<f32> {
        let (n_fft, hop) = (self.cfg.n_fft, self.cfg.hop_length);
        let len = audio.len();
        let half = n_fft / 2;
        let reflect = len > half;
        // center=True padding.
        let padded: Vec<f32> = (0..len + n_fft)
            .map(|i| {
                let j = i as isize - half as isize;
                if j >= 0 && (j as usize) < len {
                    audio[j as usize]
                } else if reflect {
                    let r = if j < 0 { -j } else { 2 * (len as isize - 1) - j };
                    audio[r as usize]
                } else {
                    0.0
                }
            })
            .collect();
        let nf = 1 + len / hop;
        let bins = self.bins();
        let mut spec = vec![0f32; nf * bins * 2];
        for_chunks(&mut spec, bins * 2, nf > 1, |f, row| {
            self.analyze(&padded[f * hop..f * hop + n_fft], row)
        });
        let mut x = self.encode(&spec, nf, None);
        self.backbone(&mut x, nf, None);
        let raw = self.decode(&x, nf);
        let delta = self.correction(&raw, &spec);
        self.istft_add(audio, &delta, nf)
    }

    /// `audio + istft(delta)` (center=True, `length` = the audio's).
    fn istft_add(&self, audio: &[f32], delta: &[f32], nf: usize) -> Vec<f32> {
        let (n_fft, hop) = (self.cfg.n_fft, self.cfg.hop_length);
        let bins = self.bins();
        let mut frames = vec![0f32; nf * n_fft];
        for_chunks(&mut frames, n_fft, nf > 1, |f, out| {
            self.synth(&delta[f * bins * 2..(f + 1) * bins * 2], out)
        });
        let total = n_fft + hop * (nf - 1);
        let mut y = vec![0f32; total];
        let mut env = vec![0f32; total];
        for f in 0..nf {
            for i in 0..n_fft {
                y[f * hop + i] += frames[f * n_fft + i];
                env[f * hop + i] += self.window[i] * self.window[i];
            }
        }
        let half = n_fft / 2;
        audio
            .iter()
            .enumerate()
            .map(|(i, &a)| {
                let j = i + half;
                if j < total && env[j] > 1e-11 { a + y[j] / env[j] } else { a }
            })
            .collect()
    }

    /// `enhance`: windows of `chunk` samples overlapping by `overlap`, linearly crossfaded.
    fn enhance(&self, audio: &[f32], chunk: usize, overlap: usize, cancel: &dyn Fn() -> bool) -> Result<Vec<f32>> {
        let n = audio.len();
        if n <= chunk {
            return Ok(self.forward_offline(audio));
        }
        let mut out = vec![0f32; n];
        let mut weight = vec![0f32; n];
        let mut starts: Vec<usize> = (0..n - chunk).step_by(chunk - overlap).collect();
        starts.push(n - chunk);
        let lin = |i: usize| if overlap > 1 { i as f32 / (overlap - 1) as f32 } else { 0.0 };
        for (idx, &start) in starts.iter().enumerate() {
            if cancel() {
                bail!("cancelled");
            }
            let end = start + chunk;
            let mut w = vec![1f32; chunk];
            if idx > 0 {
                let fade_in = starts[idx - 1] + chunk - overlap - start;
                w[..fade_in].fill(0.0);
                for i in 0..overlap {
                    w[fade_in + i] = lin(i);
                }
            }
            if overlap > 0 && end < n {
                for i in 0..overlap {
                    w[chunk - overlap + i] = 1.0 - lin(i);
                }
            }
            let y = self.forward_offline(&audio[start..end]);
            for i in 0..chunk {
                out[start + i] += y[i] * w[i];
                weight[start + i] += w[i];
            }
        }
        Ok(out.iter().zip(&weight).map(|(o, w)| o / w.max(1e-6)).collect())
    }

    fn new_lane(&self) -> Lane {
        let (nb, w) = (self.bands(), self.cfg.width);
        Lane {
            blocks: self
                .blocks
                .iter()
                .map(|b| match b {
                    Block::Attn(..) => BlockState::Attn(KvCache::default()),
                    Block::Mamba(t, _) => {
                        let m = &t.fwd;
                        let _ = w;
                        BlockState::Mamba(MambaCarry {
                            tail: vec![0f32; nb * (m.k - 1) * m.inner],
                            ssm: vec![0f32; nb * m.inner * m.state],
                        })
                    }
                })
                .collect(),
        }
    }
}

/// `norm(output_proj(cat(fwd(x) + x, bwd(x) + x)))` over `rows` sequences of `len` tokens
/// (`[rows][len][width]`), or causal with the carried state (`output_proj(fwd(x) + x)`).
fn bi_mamba(m: &BiMamba, x: &[f32], rows: usize, len: usize, carry: Option<&mut MambaCarry>) -> Vec<f32> {
    let w = m.norm.w.len();
    let n = rows * len;
    let causal = m.bwd.is_none();
    let mut f = mamba(&m.fwd, x, rows, len, false, carry);
    add_into(&mut f, x);
    let cat = if causal {
        f
    } else {
        let mut b = mamba(m.bwd.as_ref().unwrap(), x, rows, len, true, None);
        add_into(&mut b, x);
        let mut c = vec![0f32; n * 2 * w];
        for_rows(&mut c, 2 * w, |r, row| {
            row[..w].copy_from_slice(&f[r * w..(r + 1) * w]);
            row[w..].copy_from_slice(&b[r * w..(r + 1) * w]);
        });
        c
    };
    let y = m.output_proj.fwd(&cat, n);
    m.norm.fwd(&y)
}

/// One Mamba layer over `rows` sequences of `len` tokens (`[rows][len][width]`), each run from
/// its end to its start when `reverse` (the reference's flip, forward, flip), from the carried
/// state when given (updated).
fn mamba(m: &Mamba, x: &[f32], rows: usize, len: usize, reverse: bool, carry: Option<&mut MambaCarry>) -> Vec<f32> {
    let (inner, ns, k) = (m.inner, m.state, m.k);
    let n = rows * len;
    let xz = m.in_proj.fwd(x, n); // [n][2 inner]: u, z
    // Causal depthwise conv over each sequence (in its scan order), then SiLU.
    let pc = crate::prof::phase(5, "vi conv");
    let mut u = vec![0f32; n * inner];
    let tail_in: Option<&[f32]> = carry.as_ref().map(|c| &c.tail[..]);
    {
        let up = SendPtr(u.as_mut_ptr());
        crate::pool::run(rows, &|r| {
            let up = &up;
            for step in 0..len {
                let t = if reverse { len - 1 - step } else { step };
                // SAFETY: row r's tokens only.
                let dst = unsafe { std::slice::from_raw_parts_mut(up.0.add((r * len + t) * inner), inner) };
                dst.copy_from_slice(&m.conv_b);
                for j in 0..k {
                    // Input `step + j - (k - 1)` in scan order.
                    let s = step as isize + j as isize - (k as isize - 1);
                    let src: &[f32] = if s >= 0 {
                        let ts = if reverse { len - 1 - s as usize } else { s as usize };
                        &xz[(r * len + ts) * 2 * inner..(r * len + ts) * 2 * inner + inner]
                    } else if let Some(tail) = tail_in {
                        let ti = (k as isize - 1 + s) as usize;
                        &tail[(r * (k - 1) + ti) * inner..(r * (k - 1) + ti + 1) * inner]
                    } else {
                        continue;
                    };
                    let wj = &m.conv_w[j * inner..(j + 1) * inner];
                    for ((d, &wv), &sv) in dst.iter_mut().zip(wj).zip(src) {
                        *d += wv * sv;
                    }
                }
                for v in dst.iter_mut() {
                    *v = silu(*v);
                }
            }
        });
    }
    drop(pc);
    let xp = m.x_proj.fwd(&u, n); // [n][dt_rank + 2 state]
    let xw = m.dt_rank + 2 * ns;
    let dt_in: Vec<f32> = {
        let mut d = vec![0f32; n * m.dt_rank];
        for_rows(&mut d, m.dt_rank, |r, row| row.copy_from_slice(&xp[r * xw..r * xw + m.dt_rank]));
        d
    };
    let mut delta = m.dt_proj.fwd(&dt_in, n);
    for_rows(&mut delta, inner, |_, row| {
        for (v, &b) in row.iter_mut().zip(&m.dt_bias) {
            *v = softplus(*v + b);
        }
    });
    // The scan, one task per (row, block of channels).
    // Channel blocks of 16 to 256, about four tasks per thread.
    let want = (4 * crate::pool::threads()).div_ceil(rows.max(1)).max(1);
    let cb = inner.div_ceil(want).next_multiple_of(16).clamp(16, 256);
    let nbk = inner.div_ceil(cb);
    let mut y = vec![0f32; n * inner];
    let yp = SendPtr(y.as_mut_ptr());
    match carry {
        Some(c) => {
            // The conv's new tail: the last k - 1 inputs (pre-conv u) of the history (the old
            // tail, then this call's inputs).
            let mut nt = vec![0f32; rows * (k - 1) * inner];
            for r in 0..rows {
                for ti in 0..k - 1 {
                    let from_end = k - 1 - ti; // 1..=k-1
                    let dst = &mut nt[(r * (k - 1) + ti) * inner..(r * (k - 1) + ti + 1) * inner];
                    if from_end <= len {
                        let t = len - from_end;
                        dst.copy_from_slice(&xz[(r * len + t) * 2 * inner..(r * len + t) * 2 * inner + inner]);
                    } else {
                        let oi = k - 1 - (from_end - len);
                        dst.copy_from_slice(&c.tail[(r * (k - 1) + oi) * inner..(r * (k - 1) + oi + 1) * inner]);
                    }
                }
            }
            let sp = SendPtr(c.ssm.as_mut_ptr());
            scan(m, &u, &delta, &xp, &xz, rows, len, reverse, &yp, Some(&sp), nbk, cb);
            c.tail = nt;
        }
        None => scan(m, &u, &delta, &xp, &xz, rows, len, reverse, &yp, None, nbk, cb),
    }
    m.out_proj.fwd(&y, n)
}

/// The selective scan `h = exp(delta a) h + delta b u`, `y = (c . h + d u) silu(z)`, written
/// into `yp` (`[n][inner]`), from and into the state `sp` (`[rows][inner][state]`) when given.
#[allow(clippy::too_many_arguments)]
fn scan(
    m: &Mamba,
    u: &[f32],
    delta: &[f32],
    xp: &[f32],
    xz: &[f32],
    rows: usize,
    len: usize,
    reverse: bool,
    yp: &SendPtr,
    sp: Option<&SendPtr>,
    nbk: usize,
    cb: usize,
) {
    let _p = crate::prof::phase(6, "vi scan");
    let (inner, ns) = (m.inner, m.state);
    let xw = m.dt_rank + 2 * ns;
    crate::pool::run(rows * nbk, &|i| {
        let (r, bk) = (i / nbk, i % nbk);
        let c0 = bk * cb;
        let c1 = (c0 + cb).min(inner);
        let mut h = vec![0f32; (c1 - c0) * ns];
        if let Some(sp) = sp {
            for c in c0..c1 {
                // SAFETY: reading this task's own state slice.
                let s = unsafe { std::slice::from_raw_parts(sp.0.add((r * inner + c) * ns), ns) };
                h[(c - c0) * ns..(c - c0 + 1) * ns].copy_from_slice(s);
            }
        }
        for step in 0..len {
            let t = if reverse { len - 1 - step } else { step };
            let tok = r * len + t;
            let bv = &xp[tok * xw + m.dt_rank..tok * xw + m.dt_rank + ns];
            let cv = &xp[tok * xw + m.dt_rank + ns..tok * xw + m.dt_rank + 2 * ns];
            // SAFETY: this task's own channels of token `tok`.
            let yrow = unsafe { std::slice::from_raw_parts_mut(yp.0.add(tok * inner + c0), c1 - c0) };
            for c in c0..c1 {
                let dl = delta[tok * inner + c];
                let x = u[tok * inner + c];
                let drive = dl * x;
                let a = &m.a[c * ns..(c + 1) * ns];
                let hs = &mut h[(c - c0) * ns..(c - c0 + 1) * ns];
                let acc = if ns == 16 {
                    // Fixed size: the 16 states in vector registers.
                    let hs: &mut [f32; 16] = hs.try_into().unwrap();
                    let a: &[f32; 16] = a.try_into().unwrap();
                    let bv: &[f32; 16] = bv.try_into().unwrap();
                    let cv: &[f32; 16] = cv.try_into().unwrap();
                    let mut p = [0f32; 16];
                    for nn in 0..16 {
                        hs[nn] = fast_exp(dl * a[nn]) * hs[nn] + drive * bv[nn];
                        p[nn] = hs[nn] * cv[nn];
                    }
                    p.iter().sum::<f32>()
                } else {
                    let mut acc = 0f32;
                    for nn in 0..ns {
                        hs[nn] = fast_exp(dl * a[nn]) * hs[nn] + drive * bv[nn];
                        acc += hs[nn] * cv[nn];
                    }
                    acc
                };
                let out = acc + x * m.d[c];
                let g = xz[tok * 2 * inner + inner + c];
                yrow[c - c0] = out * silu(g);
            }
        }
        if let Some(sp) = sp {
            for c in c0..c1 {
                // SAFETY: writing this task's own state slice.
                let s = unsafe { std::slice::from_raw_parts_mut(sp.0.add((r * inner + c) * ns), ns) };
                s.copy_from_slice(&h[(c - c0) * ns..(c - c0 + 1) * ns]);
            }
        }
    });
}

// ---------------------------------------------------------------------------------------------
// FFT.

/// Radix-2 complex FFT of one power-of-two size (twiddles in f64, rounded once).
struct Fft {
    n: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
    rev: Vec<usize>,
}

impl Fft {
    fn new(n: usize) -> Self {
        let bits = n.trailing_zeros();
        let rev = (0..n)
            .map(|i| if n > 1 { i.reverse_bits() >> (usize::BITS - bits) } else { 0 })
            .collect();
        let (cos, sin) = (0..n / 2)
            .map(|k| {
                let a = -2.0 * std::f64::consts::PI * k as f64 / n as f64;
                (a.cos() as f32, a.sin() as f32)
            })
            .unzip();
        Self { n, cos, sin, rev }
    }

    /// In place; `inverse` conjugates the twiddles (no 1 / n).
    fn run(&self, re: &mut [f32], im: &mut [f32], inverse: bool) {
        let n = self.n;
        for i in 0..n {
            let j = self.rev[i];
            if j > i {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let sign = if inverse { -1.0 } else { 1.0 };
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let step = n / len;
            for s in (0..n).step_by(len) {
                for k in 0..half {
                    let (wr, wi) = (self.cos[k * step], sign * self.sin[k * step]);
                    let (a, b) = (s + k, s + k + half);
                    let (xr, xi) = (re[b] * wr - im[b] * wi, re[b] * wi + im[b] * wr);
                    re[b] = re[a] - xr;
                    im[b] = im[a] - xi;
                    re[a] += xr;
                    im[a] += xi;
                }
            }
            len *= 2;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The public API.

/// A loaded voice isolator (one checkpoint).
pub struct VoiceIsolator {
    net: Arc<Net>,
    info: IsolatorInfo,
    cancel: Arc<std::sync::atomic::AtomicBool>,
}

/// Whether `path` is a voice isolator checkpoint: a `.safetensors` file whose metadata `format`
/// is `inworld-voice-isolation`, or a directory holding `denoiser*.safetensors`.
pub fn is_voice_isolator(path: &Path) -> bool {
    let file_ok = |p: &Path| {
        p.extension().is_some_and(|e| e == "safetensors")
            && crate::safetensors::SafeTensors::open(p, false)
                .ok()
                .is_some_and(|s| s.metadata("format") == Some("inworld-voice-isolation"))
    };
    if path.is_dir() {
        return [OFFLINE_FILE, REALTIME_FILE].iter().any(|f| file_ok(&path.join(f)));
    }
    file_ok(path)
}

/// Whether `spec` names the voice isolator repository (or a file of it).
pub fn is_repo_id(spec: &str) -> bool {
    let s = spec.trim_end_matches('/');
    s.eq_ignore_ascii_case(REPO) || s.to_ascii_lowercase().starts_with(&format!("{REPO}/"))
}

/// The checkpoint file of `spec` for `mode`: a `.safetensors` file as is; in a directory, the
/// mode's file; a repo id (`inworld/voice-isolator`, or a file of it) fetched into the cache.
pub fn resolve(spec: &str, mode: Mode, cache_dir: Option<&Path>) -> Result<PathBuf> {
    let file = match mode {
        Mode::Offline => OFFLINE_FILE,
        Mode::Streaming => REALTIME_FILE,
    };
    let p = Path::new(spec);
    if p.is_file() {
        return Ok(p.to_path_buf());
    }
    if p.is_dir() {
        let f = p.join(file);
        if f.is_file() {
            return Ok(f);
        }
        bail!("{}: no {file}", p.display());
    }
    #[cfg(feature = "hf")]
    {
        let parts: Vec<&str> = spec.trim_end_matches('/').split('/').collect();
        if parts.len() > 2 {
            return crate::hf::resolve_file(spec, cache_dir);
        }
        let dir = crate::hf::resolve_files(spec, &[file], cache_dir)?;
        let f = dir.join(file);
        if !f.is_file() {
            bail!("{spec}: the repository has no {file}");
        }
        Ok(f)
    }
    #[cfg(not(feature = "hf"))]
    {
        let _ = cache_dir;
        bail!("{spec}: not a local checkpoint (built without the `hf` feature, so repo ids cannot be fetched)")
    }
}

impl VoiceIsolator {
    /// Loads a checkpoint: a `.safetensors` file, a directory with the released files, or (with
    /// the `hf` feature) the repo id `inworld/voice-isolator` (`opts.mode` picks the file).
    pub fn load(spec: &str, opts: &IsolatorOptions) -> Result<Self> {
        let path = resolve(spec, opts.mode, opts.cache_dir.as_deref())?;
        Self::load_file(&path, opts)
    }

    /// Loads one checkpoint file.
    pub fn load_file(path: &Path, opts: &IsolatorOptions) -> Result<Self> {
        let t0 = std::time::Instant::now();
        if let Some(n) = opts.threads {
            let mut o = crate::pool::options();
            o.threads = n;
            crate::pool::configure_with(o);
        }
        let gpu = crate::gpu::ensure(opts.device)?;
        let files = ModelFiles::open_file(path, true)?;
        let raw = files
            .metadata("config")
            .with_context(|| format!("{}: no `config` metadata (not a voice isolator checkpoint)", path.display()))?;
        let cfg: IsolatorConfig = serde_json::from_str(raw).with_context(|| format!("{}: config metadata", path.display()))?;
        let net = Net::load(&files, cfg, opts.f32_weights).with_context(|| path.display().to_string())?;
        let c = &net.cfg;
        let mode = if c.causal { Mode::Streaming } else { Mode::Offline };
        let info = IsolatorInfo {
            path: path.to_path_buf(),
            name: if c.causal {
                "voice-isolator-v1-realtime"
            } else {
                "voice-isolator-v1"
            }
            .to_string(),
            mode,
            params: net.params,
            width: c.width,
            blocks: c.block_types.len(),
            bands: net.bands(),
            sample_rate: c.sample_rate,
            lookahead_choices: c.lookahead_choices(),
            default_lookahead: c.lookahead_frames,
            attention_window_frames: c.attention_window_frames,
            f32_weights: opts.f32_weights,
            device: gpu.map_or("cpu".to_string(), |b| b.name().to_string()),
            load_secs: t0.elapsed().as_secs_f64(),
        };
        Ok(Self {
            net: Arc::new(net),
            info,
            cancel: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        })
    }

    pub fn info(&self) -> &IsolatorInfo {
        &self.info
    }

    pub fn config(&self) -> &IsolatorConfig {
        &self.net.cfg
    }

    /// The model rate (48 kHz).
    pub fn sample_rate(&self) -> u32 {
        self.net.cfg.sample_rate
    }

    /// A flag that stops a running [`VoiceIsolator::isolate`] (it then fails with "cancelled").
    pub fn cancel_handle(&self) -> Arc<std::sync::atomic::AtomicBool> {
        self.cancel.clone()
    }

    /// Isolates the voice of mono `pcm` at `rate` Hz; returns the voice at the model rate
    /// (48 kHz), aligned with the input resampled to it. The offline checkpoint renders in
    /// overlapping windows; the realtime one runs its stream over the clip.
    pub fn isolate(&self, pcm: &[f32], rate: u32, o: &IsolateOptions) -> Result<Vec<f32>> {
        self.cancel.store(false, std::sync::atomic::Ordering::Relaxed);
        if pcm.is_empty() {
            bail!("the audio is empty: there is nothing to isolate");
        }
        if pcm.iter().any(|v| !v.is_finite()) {
            bail!("the audio contains NaN or infinite samples");
        }
        let sr = self.sample_rate();
        let x = if rate == sr {
            pcm.to_vec()
        } else {
            crate::audio::resample_kaiser_best(pcm, rate as usize, sr as usize)
        };
        let cancel = || self.cancel.load(std::sync::atomic::Ordering::Relaxed);
        if self.net.cfg.causal {
            let mut so = o.clone();
            so.block_frames = Some(o.block_frames.unwrap_or(64));
            let mut s = self.stream(&so)?;
            let piece = sr as usize / 2;
            let mut out = Vec::with_capacity(x.len());
            for c in x.chunks(piece) {
                if cancel() {
                    bail!("cancelled");
                }
                out.extend(s.push(c));
            }
            out.extend(s.flush());
            out.truncate(x.len());
            return Ok(out);
        }
        if o.lookahead_frames.is_some_and(|l| l != 0) {
            bail!("lookahead_frames applies to the realtime checkpoint (this one is offline)");
        }
        let c = &self.net.cfg;
        let chunk_s = o.chunk_seconds.unwrap_or(c.chunk_seconds);
        let overlap_s = o.overlap_seconds.unwrap_or(c.overlap_seconds);
        if !chunk_s.is_finite() || !overlap_s.is_finite() || overlap_s < 0.0 {
            bail!("chunk_seconds / overlap_seconds must be finite and the overlap non-negative");
        }
        let chunk = (chunk_s * sr as f64) as usize;
        if chunk < c.hop_length {
            bail!("chunk_seconds={chunk_s} is shorter than one hop");
        }
        let overlap = ((overlap_s * sr as f64) as usize).min(chunk / 2);
        let passes = o.offline_passes.unwrap_or(c.offline_passes).max(1);
        let mut y = self.net.enhance(&x, chunk, overlap, &cancel)?;
        for _ in 1..passes {
            y = self.net.enhance(&y, chunk, overlap, &cancel)?;
        }
        Ok(y)
    }

    /// Starts a streaming session (realtime checkpoint only).
    pub fn stream(&self, o: &IsolateOptions) -> Result<IsolatorStream> {
        IsolatorStream::new(self.net.clone(), o)
    }
}

/// A streaming session: [`IsolatorStream::push`] 48 kHz mono samples in pieces of any size,
/// receive the output samples that became final; [`IsolatorStream::flush`] ends it.
pub struct IsolatorStream {
    net: Arc<Net>,
    lookahead: usize,
    block_frames: usize,
    cache_frames: usize,
    capacity: usize,
    windows: Option<(usize, usize)>,
    lane: Lane,
    lane_b: Option<Lane>,
    /// Input samples from `audio_base`.
    audio: Vec<f32>,
    audio_base: usize,
    /// Frames computed.
    frames: usize,
    /// Frames synthesized.
    emitted: usize,
    /// Decoder output and spectra of frames `pending_base..frames` (`[bins][4]`, `[bins][2]`).
    raw: Vec<Vec<f32>>,
    spectra: Vec<Vec<f32>>,
    pending_base: usize,
    /// Overlap-add and its window envelope, from padded position `ola_base`.
    ola: Vec<f32>,
    env: Vec<f32>,
    ola_base: usize,
    released: usize,
    closed: bool,
    /// Lane boundaries crossed (frames), as the reference's `boundaries`.
    pub boundaries: Vec<usize>,
    /// The lanes on the accelerator (whole steps there): the session, the device lane of
    /// `lane`, whether the other one is the second lane (`lane_b`).
    dev: Option<Box<dyn vi::Session>>,
    dev_lane: usize,
    dev_b: bool,
}

impl IsolatorStream {
    fn new(net: Arc<Net>, o: &IsolateOptions) -> Result<Self> {
        let c = &net.cfg;
        if !c.causal {
            bail!("streaming needs the realtime checkpoint ({REALTIME_FILE}); this one is offline");
        }
        let choices = c.lookahead_choices();
        let lookahead = o.lookahead_frames.unwrap_or(c.lookahead_frames);
        if !choices.contains(&lookahead) {
            bail!("lookahead_frames must be one of {choices:?} (10 ms each), got {lookahead}");
        }
        let block_frames = o.block_frames.unwrap_or(4);
        if block_frames == 0 {
            bail!("block_frames must be at least 1");
        }
        let cache_frames = match o.cache_frames {
            Some(0) => bail!("cache_frames must be at least 1"),
            Some(n) => n,
            None if c.attention_window_frames > 0 => c.attention_window_frames,
            None => ((c.chunk_seconds * c.sample_rate as f64 / c.hop_length as f64).round() as usize).max(1),
        };
        let window = o.window_frames.unwrap_or(STREAM_WINDOW_FRAMES);
        let windows = if window == 0 {
            None
        } else {
            let cf = o.crossfade_frames.unwrap_or(STREAM_CROSSFADE_FRAMES);
            if cf == 0 || cf >= window {
                bail!("crossfade_frames must be positive and shorter than window_frames");
            }
            Some((window, cf))
        };
        let capacity = match windows {
            Some((w, cf)) if w + cf <= cache_frames + block_frames => w + cf,
            _ => cache_frames + (cache_frames / 2).max(block_frames),
        };
        let lane = net.new_lane();
        let n_fft = c.n_fft;
        let look = net
            .look_emb
            .as_ref()
            .and_then(|_| c.lookahead_frames_choices.iter().position(|&x| x == lookahead));
        let dev = net.device().and_then(|d| {
            d.session(&vi::StreamSpec {
                look,
                cache_frames,
                capacity,
                max_frames: block_frames,
            })
        });
        Ok(Self {
            dev,
            dev_lane: 0,
            dev_b: false,
            lookahead,
            block_frames,
            cache_frames,
            capacity,
            windows,
            lane,
            lane_b: None,
            audio: Vec::new(),
            audio_base: 0,
            frames: 0,
            emitted: 0,
            raw: Vec::new(),
            spectra: Vec::new(),
            pending_base: 0,
            ola: vec![0f32; n_fft],
            env: vec![0f32; n_fft],
            ola_base: 0,
            released: 0,
            closed: false,
            boundaries: Vec::new(),
            net,
        })
    }

    /// Input samples that must follow an output sample before it is released: the analysis
    /// window plus the look-ahead.
    pub fn latency_samples(&self) -> usize {
        self.net.cfg.n_fft + self.net.cfg.hop_length * self.lookahead
    }

    fn analyze_frames(&self, first: usize, count: usize) -> Vec<Vec<f32>> {
        let (n_fft, hop) = (self.net.cfg.n_fft, self.net.cfg.hop_length);
        let total = self.audio_base + self.audio.len();
        (first..first + count)
            .map(|f| {
                let start = (f * hop) as isize - (n_fft / 2) as isize;
                let seg: Vec<f32> = (0..n_fft as isize)
                    .map(|i| {
                        let s = start + i;
                        if s >= self.audio_base as isize && (s as usize) < total {
                            self.audio[s as usize - self.audio_base]
                        } else {
                            0.0
                        }
                    })
                    .collect();
                let mut out = vec![0f32; 2 * self.net.bins()];
                self.net.analyze(&seg, &mut out);
                out
            })
            .collect()
    }

    fn next_cut(&self) -> usize {
        match self.windows {
            None => self.frames + self.block_frames,
            Some((w, cf)) => {
                let start = self.frames - self.frames % w;
                let cs = start + w - cf;
                if self.frames < cs { cs } else { start + w }
            }
        }
    }

    /// The backbone and decoder over a block of spectra: `[frames][bins][4]`.
    fn run_block(&mut self, spectra: &[Vec<f32>]) -> Vec<f32> {
        let net = self.net.clone();
        let nf = spectra.len();
        let bins = net.bins();
        let mut spec = Vec::with_capacity(nf * bins * 2);
        for s in spectra {
            spec.extend_from_slice(s);
        }
        // Whether the second lane runs too (the crossfade before a window's end).
        let two = self.windows.is_some_and(|(w, cf)| {
            let first = self.frames % w;
            debug_assert!(first + nf <= w && !(first < w - cf && w - cf < first + nf));
            first >= w - cf
        });
        if two && !self.has_b() {
            if let Some(d) = self.dev.as_mut() {
                d.fork(self.dev_lane, 1 - self.dev_lane);
                self.dev_b = true;
            } else {
                self.lane_b = Some(self.lane.fork());
            }
        }
        let (mut raw, raw_b) = self.run_lanes(&spec, nf, two);
        let (Some((w, cf)), Some(raw_b)) = (self.windows, raw_b) else {
            return raw;
        };
        let first = self.frames % w;
        let cs = w - cf;
        for f in 0..nf {
            let wt = (((first + f) as f32 - cs as f32 + 1.0) / cf as f32).clamp(0.0, 1.0);
            for (a, &b) in raw[f * bins * 4..(f + 1) * bins * 4]
                .iter_mut()
                .zip(&raw_b[f * bins * 4..(f + 1) * bins * 4])
            {
                *a = *a * (1.0 - wt) + b * wt;
            }
        }
        if first + nf == w {
            self.boundaries.push(self.frames + nf);
            if self.dev.is_some() {
                self.dev_lane = 1 - self.dev_lane;
                self.dev_b = false;
            } else {
                self.lane = self.lane_b.take().unwrap();
            }
        }
        raw
    }

    fn has_b(&self) -> bool {
        if self.dev.is_some() { self.dev_b } else { self.lane_b.is_some() }
    }

    /// The decoder outputs of the lane (and, with `two`, of the second lane) over a block of
    /// spectra: one step on the accelerator, else (or when it fails, the lanes then carried on
    /// from its states) the CPU path.
    fn run_lanes(&mut self, spec: &[f32], nf: usize, two: bool) -> (Vec<f32>, Option<Vec<f32>>) {
        if let Some(d) = self.dev.as_mut() {
            let _p = crate::prof::phase(7, "vi device step");
            let lanes = [self.dev_lane, 1 - self.dev_lane];
            let lanes = &lanes[..1 + two as usize];
            let mut raw = vec![Vec::new(); lanes.len()];
            if d.step(spec, nf, lanes, &mut raw) {
                let b = two.then(|| raw.pop().unwrap());
                return (raw.pop().unwrap(), b);
            }
            crate::gpu::warn("voice isolator: a stream step failed on the GPU, the stream carries on on the CPU");
            let import = |blocks: Vec<vi::LaneBlock>| Lane {
                blocks: blocks
                    .into_iter()
                    .map(|b| match b {
                        vi::LaneBlock::Attn { k, v, cap, end } => BlockState::Attn(KvCache { k, v, cap, end }),
                        vi::LaneBlock::Mamba { tail, ssm } => BlockState::Mamba(MambaCarry { tail, ssm }),
                    })
                    .collect(),
            };
            self.lane = import(d.export(self.dev_lane));
            self.lane_b = self.dev_b.then(|| import(d.export(1 - self.dev_lane)));
            self.dev = None;
        }
        let net = self.net.clone();
        let mut feats = net.encode(spec, nf, Some(self.lookahead));
        let mut x = if two { feats.clone() } else { std::mem::take(&mut feats) };
        net.backbone(&mut x, nf, Some((&mut self.lane, self.cache_frames, self.capacity)));
        let raw = net.decode(&x, nf);
        if !two {
            return (raw, None);
        }
        let mut xb = feats;
        net.backbone(&mut xb, nf, Some((self.lane_b.as_mut().unwrap(), self.cache_frames, self.capacity)));
        (raw, Some(net.decode(&xb, nf)))
    }

    fn advance(&mut self, limit: usize) {
        while self.frames < limit {
            let count = self.block_frames.min(limit - self.frames).min(self.next_cut() - self.frames);
            let spectra = self.analyze_frames(self.frames, count);
            let raw = self.run_block(&spectra);
            let bins = self.net.bins();
            for f in 0..count {
                self.raw.push(raw[f * bins * 4..(f + 1) * bins * 4].to_vec());
            }
            self.spectra.extend(spectra);
            self.frames += count;
        }
        while self.emitted + self.lookahead < self.frames {
            self.synthesize();
        }
        self.trim();
    }

    /// Synthesizes the next frame from the decoder's output `lookahead` frames later (the last
    /// computed frame serving every frame beyond it).
    fn synthesize(&mut self) {
        let net = self.net.clone();
        let (n_fft, hop) = (net.cfg.n_fft, net.cfg.hop_length);
        let index = self.emitted;
        let last = self.frames - 1;
        let source = (index + self.lookahead).min(last);
        let raw = &self.raw[source - self.pending_base];
        let spec = &self.spectra[index - self.pending_base];
        let delta = net.correction(raw, spec);
        let mut frame = vec![0f32; n_fft];
        net.synth(&delta, &mut frame);
        let start = index * hop - self.ola_base;
        if start + n_fft > self.ola.len() {
            self.ola.resize(start + n_fft, 0.0);
            self.env.resize(start + n_fft, 0.0);
        }
        for i in 0..n_fft {
            self.ola[start + i] += frame[i];
            self.env[start + i] += net.window[i] * net.window[i];
        }
        self.emitted += 1;
    }

    fn trim(&mut self) {
        let drop = self.emitted - self.pending_base;
        self.raw.drain(..drop);
        self.spectra.drain(..drop);
        self.pending_base = self.emitted;
        let keep_from = self.released.saturating_sub(self.ola_base);
        if keep_from > 0 {
            let k = keep_from.min(self.ola.len());
            self.ola.drain(..k);
            self.env.drain(..k);
            self.ola_base += k;
        }
        let (n_fft, hop) = (self.net.cfg.n_fft, self.net.cfg.hop_length);
        let needed_from = self.released.min((self.frames * hop).saturating_sub(n_fft / 2));
        if needed_from > self.audio_base {
            let d = (needed_from - self.audio_base).min(self.audio.len());
            self.audio.drain(..d);
            self.audio_base += d;
        }
    }

    /// Output samples `[released, upto)`: the input plus the synthesized change.
    fn release(&mut self, upto: usize) -> Vec<f32> {
        let total = self.audio_base + self.audio.len();
        let (start, stop) = (self.released, upto.min(total));
        if stop <= start {
            return Vec::new();
        }
        let half = self.net.cfg.n_fft / 2;
        let out = (start..stop)
            .map(|s| {
                let j = s + half - self.ola_base;
                let (o, e) = (self.ola.get(j).copied().unwrap_or(0.0), self.env.get(j).copied().unwrap_or(0.0));
                self.audio[s - self.audio_base] + o / e.max(1e-11)
            })
            .collect();
        self.released = stop;
        out
    }

    /// Feeds new samples (48 kHz mono); returns the output samples that became final.
    pub fn push(&mut self, samples: &[f32]) -> Vec<f32> {
        if self.closed {
            return Vec::new();
        }
        self.audio.extend_from_slice(samples);
        let (n_fft, hop) = (self.net.cfg.n_fft, self.net.cfg.hop_length);
        let total = self.audio_base + self.audio.len();
        let computable = if total >= n_fft / 2 { (total - n_fft / 2) / hop + 1 } else { 0 };
        self.advance(computable);
        let upto = (self.emitted * hop).saturating_sub(n_fft / 2);
        let out = self.release(upto);
        self.trim();
        out
    }

    /// Ends the stream and returns every remaining output sample.
    pub fn flush(&mut self) -> Vec<f32> {
        if self.closed {
            return Vec::new();
        }
        self.closed = true;
        let total = self.audio_base + self.audio.len();
        let hop = self.net.cfg.hop_length;
        self.advance(total / hop + 1);
        while self.emitted < self.frames {
            self.synthesize();
        }
        self.release(total)
    }
}

/// Mono samples of an audio file at its own rate: WAV (any channels averaged), raw
/// little-endian f32 at 48 kHz for `.f32`, or (with the `media` feature) any format oxideav
/// decodes.
pub fn load_audio(path: &Path) -> Result<(Vec<f32>, u32)> {
    if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("f32")) {
        let b = crate::fsx::read(path)?;
        return Ok((b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect(), SAMPLE_RATE));
    }
    let bytes = crate::fsx::read(path)?;
    match crate::tts::read_wav(&bytes) {
        Ok(a) => Ok((a.samples, a.sample_rate)),
        #[cfg(feature = "media")]
        Err(_) => {
            let clip = crate::media::Media::new().decode_audio(path)?;
            let rate = clip.params.sample_rate.context("the audio has no sample rate")?;
            Ok((clip.to_mono(rate as usize)?, rate))
        }
        #[cfg(not(feature = "media"))]
        Err(e) => Err(e.context(format!("{} (built without `media`: WAV only)", path.display()))),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn lcg(s: &mut u64) -> f32 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }

    /// Writes a random f32 checkpoint of `cfg` with the config metadata.
    pub(crate) fn tiny(tag: &str, causal: bool) -> PathBuf {
        let w = 16usize;
        let cfg = serde_json::json!({
            "width": w, "block_types": ["transformer", "mamba"], "num_heads": 2, "n_fft": 64, "hop_length": 16,
            "sample_rate": 1600, "band_plan": [[100, 400], [200, 800]], "correction_bound": 0.5,
            "chunk_seconds": 0.4, "overlap_seconds": 0.1, "exclusive_self_attention": true, "exclusive_axes": "time",
            "mamba_state": 4, "mamba_conv": 4, "mamba_expand": 2, "causal": causal,
            "lookahead_frames": if causal { 1 } else { 0 }, "lookahead_frames_choices": if causal { vec![0, 1] } else { vec![] },
            "attention_window_frames": if causal { 12 } else { 0 },
        });
        let c: IsolatorConfig = serde_json::from_value(cfg.clone()).unwrap();
        let edges = band_edges(&c);
        let mut t: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
        let mut s = 7u64 + causal as u64;
        let mut rnd = |shape: Vec<usize>, scale: f32| {
            let n: usize = shape.iter().product();
            let v: Vec<f32> = (0..n).map(|_| lcg(&mut s) * scale).collect();
            (shape, v)
        };
        let put = |t: &mut Vec<(String, Vec<usize>, Vec<f32>)>, name: &str, (shape, v): (Vec<usize>, Vec<f32>)| {
            t.push((name.to_string(), shape, v));
        };
        let ln = |t: &mut Vec<(String, Vec<usize>, Vec<f32>)>, p: &str, n: usize| {
            t.push((format!("{p}.weight"), vec![n], vec![1.0; n]));
            t.push((format!("{p}.bias"), vec![n], vec![0.0; n]));
        };
        for (b, &(st, e)) in edges.iter().enumerate() {
            let n_in = 2 * (e - st);
            ln(&mut t, &format!("encoders.{b}.0"), n_in);
            put(&mut t, &format!("encoders.{b}.1.weight"), rnd(vec![w, n_in], 0.5));
            put(&mut t, &format!("encoders.{b}.1.bias"), rnd(vec![w], 0.1));
            ln(&mut t, &format!("denoise_decoder.heads.{b}.0"), w);
            put(&mut t, &format!("denoise_decoder.heads.{b}.1.weight"), rnd(vec![w, w], 0.5));
            put(&mut t, &format!("denoise_decoder.heads.{b}.1.bias"), rnd(vec![w], 0.1));
            put(
                &mut t,
                &format!("denoise_decoder.heads.{b}.3.weight"),
                rnd(vec![4 * (e - st), w], 0.5),
            );
            put(&mut t, &format!("denoise_decoder.heads.{b}.3.bias"), rnd(vec![4 * (e - st)], 0.1));
        }
        put(&mut t, "band_embedding", rnd(vec![edges.len(), w], 0.1));
        if causal {
            put(&mut t, "lookahead_embedding.weight", rnd(vec![2, w], 0.1));
        }
        put(&mut t, "denoise_decoder.condition.weight", rnd(vec![2 * w, 2], 0.1));
        put(&mut t, "denoise_decoder.condition.bias", rnd(vec![2 * w], 0.1));
        let p = "blocks.0.time_transformer";
        ln(&mut t, &format!("{p}.norm1"), w);
        ln(&mut t, &format!("{p}.norm2"), w);
        for (n, o, i) in [
            ("attn.qkv", 3 * w, w),
            ("attn.proj", w, w),
            ("ffn.0", 4 * w, w),
            ("ffn.3", w, 4 * w),
        ] {
            put(&mut t, &format!("{p}.{n}.weight"), rnd(vec![o, i], 0.4));
            put(&mut t, &format!("{p}.{n}.bias"), rnd(vec![o], 0.1));
        }
        let p = "blocks.0.band_transformer";
        ln(&mut t, &format!("{p}.norm1"), w);
        ln(&mut t, &format!("{p}.norm2"), w);
        put(&mut t, &format!("{p}.self_attn.in_proj_weight"), rnd(vec![3 * w, w], 0.4));
        put(&mut t, &format!("{p}.self_attn.in_proj_bias"), rnd(vec![3 * w], 0.1));
        for (n, o, i) in [("self_attn.out_proj", w, w), ("linear1", 4 * w, w), ("linear2", w, 4 * w)] {
            put(&mut t, &format!("{p}.{n}.weight"), rnd(vec![o, i], 0.4));
            put(&mut t, &format!("{p}.{n}.bias"), rnd(vec![o], 0.1));
        }
        let inner = 2 * w;
        for (bm, causal_here) in [("time_mamba", causal), ("freq_mamba", false)] {
            let p = format!("blocks.1.{bm}");
            let dirs: &[&str] = if causal_here {
                &["forward_blocks"]
            } else {
                &["forward_blocks", "backward_blocks"]
            };
            for d in dirs {
                let q = format!("{p}.{d}");
                put(&mut t, &format!("{q}.in_proj.weight"), rnd(vec![2 * inner, w], 0.4));
                put(&mut t, &format!("{q}.conv1d.weight"), rnd(vec![inner, 1, 4], 0.5));
                put(&mut t, &format!("{q}.conv1d.bias"), rnd(vec![inner], 0.1));
                put(&mut t, &format!("{q}.x_proj.weight"), rnd(vec![1 + 2 * 4, inner], 0.4));
                put(&mut t, &format!("{q}.dt_proj.weight"), rnd(vec![inner, 1], 0.5));
                put(&mut t, &format!("{q}.dt_proj.bias"), rnd(vec![inner], 0.5));
                let a: Vec<f32> = (0..inner * 4).map(|i| ((i % 4) as f32 + 1.0).ln()).collect();
                t.push((format!("{q}.A_log"), vec![inner, 4], a));
                t.push((format!("{q}.D"), vec![inner], vec![1.0; inner]));
                put(&mut t, &format!("{q}.out_proj.weight"), rnd(vec![w, inner], 0.4));
            }
            let pin = if causal_here { w } else { 2 * w };
            put(&mut t, &format!("{p}.output_proj.weight"), rnd(vec![w, pin], 0.4));
            put(&mut t, &format!("{p}.output_proj.bias"), rnd(vec![w], 0.1));
            ln(&mut t, &format!("{p}.norm"), w);
        }
        let mut header = serde_json::Map::new();
        header.insert(
            "__metadata__".into(),
            serde_json::json!({ "config": cfg.to_string(), "format": "inworld-voice-isolation" }),
        );
        let mut data = Vec::new();
        for (name, shape, vals) in &t {
            let start = data.len();
            for v in vals {
                data.extend_from_slice(&v.to_le_bytes());
            }
            header.insert(
                name.clone(),
                serde_json::json!({ "dtype": "F32", "shape": shape, "data_offsets": [start, data.len()] }),
            );
        }
        let mut json = serde_json::to_vec(&header).unwrap();
        while !json.len().is_multiple_of(8) {
            json.push(b' ');
        }
        let mut out = (json.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&json);
        out.extend_from_slice(&data);
        let path = std::env::temp_dir().join(format!("oxinfer-vi-{tag}-{}.safetensors", std::process::id()));
        std::fs::write(&path, out).unwrap();
        path
    }

    pub(crate) fn signal(n: usize) -> Vec<f32> {
        let mut s = 3u64;
        (0..n).map(|i| 0.3 * (i as f32 * 0.07).sin() + 0.2 * lcg(&mut s)).collect()
    }

    #[test]
    fn band_edges_match_the_reference() {
        let cfg: IsolatorConfig = serde_json::from_value(serde_json::json!({
            "width": 512, "block_types": ["transformer"], "num_heads": 8, "n_fft": 2048, "hop_length": 480,
            "sample_rate": 48000, "band_plan": [[100, 1000], [250, 4000], [500, 8000], [1000, 16000], [2000, 24000]],
            "correction_bound": 0.5, "chunk_seconds": 8.0, "overlap_seconds": 2.0, "mamba_state": 16, "mamba_conv": 4,
            "mamba_expand": 2,
        }))
        .unwrap();
        let e = band_edges(&cfg);
        assert_eq!(e.len(), 46);
        assert_eq!(e[10], (40, 43));
        assert_eq!(e[11], (43, 54));
        assert_eq!(e[31], (339, 341));
        assert_eq!(e[44], (1023, 1024));
        assert_eq!(e[45], (1024, 1025));
    }

    #[test]
    fn fft_round_trip() {
        let f = Fft::new(64);
        let x: Vec<f32> = (0..64).map(|i| (i as f32 * 0.3).sin() + 0.1 * i as f32).collect();
        let (mut re, mut im) = (x.clone(), vec![0f32; 64]);
        f.run(&mut re, &mut im, false);
        // Bin 3 against the direct DFT.
        let (mut r3, mut i3) = (0f64, 0f64);
        for (n, &v) in x.iter().enumerate() {
            let a = -2.0 * std::f64::consts::PI * 3.0 * n as f64 / 64.0;
            r3 += v as f64 * a.cos();
            i3 += v as f64 * a.sin();
        }
        assert!((re[3] as f64 - r3).abs() < 1e-4 && (im[3] as f64 - i3).abs() < 1e-4);
        f.run(&mut re, &mut im, true);
        for (a, b) in re.iter().zip(&x) {
            assert!((a / 64.0 - b).abs() < 1e-5);
        }
    }

    #[test]
    fn offline_runs_and_crossfades() {
        let path = tiny("off", false);
        let vi = VoiceIsolator::load_file(&path, &IsolatorOptions::default()).unwrap();
        let x = signal(2000); // longer than one 0.4 s window: three windows
        let y = vi.isolate(&x, 1600, &IsolateOptions::default()).unwrap();
        assert_eq!(y.len(), x.len());
        assert!(y.iter().all(|v| v.is_finite()));
        assert!(y.iter().zip(&x).any(|(a, b)| (a - b).abs() > 1e-3), "the network changes the audio");
        // A clip of one window is one pass.
        let one = vi.isolate(&x[..600], 1600, &IsolateOptions::default()).unwrap();
        assert_eq!(one, vi.net.forward_offline(&x[..600]));
        let _ = std::fs::remove_file(path);
    }

    /// The plain stream (no windows, a cache over the attention window) gives the whole-clip
    /// causal forward, whatever the pushes and blocks.
    #[test]
    fn stream_independent_of_pushes_and_blocks() {
        let path = tiny("rt", true);
        let vi = VoiceIsolator::load_file(&path, &IsolatorOptions::default()).unwrap();
        let x = signal(1500);
        let run = |piece: usize, block: usize, window: usize| {
            let o = IsolateOptions {
                block_frames: Some(block),
                window_frames: Some(window),
                crossfade_frames: Some(3),
                ..Default::default()
            };
            let mut s = vi.stream(&o).unwrap();
            let mut out = Vec::new();
            for c in x.chunks(piece) {
                out.extend(s.push(c));
            }
            out.extend(s.flush());
            out
        };
        for window in [0, 10] {
            let a = run(1500, 200, window);
            assert_eq!(a.len(), x.len());
            for (piece, block) in [(7, 1), (100, 4), (333, 3)] {
                let b = run(piece, block, window);
                assert_eq!(b.len(), a.len());
                let d = a.iter().zip(&b).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max);
                assert!(d < 2e-4, "window {window} piece {piece} block {block}: max diff {d}");
            }
        }
        // Windows change the output (the attention is bounded).
        let (a, b) = (run(1500, 200, 0), run(1500, 200, 10));
        assert!(a.iter().zip(&b).any(|(p, q)| (p - q).abs() > 1e-4));
        assert!(vi.isolate(&x, 1600, &IsolateOptions::default()).is_ok());
        let _ = std::fs::remove_file(path);
    }
}
