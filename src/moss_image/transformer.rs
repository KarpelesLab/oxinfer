//! The MOSS-Image DiT (`MOSSImageTransformer` of the release's `pipeline_moss_image.py`): one
//! stream of image tokens followed by text tokens through 42 identical blocks.
//!
//! - Text: Qwen3.5 hidden states (4096 wide) projected to the model width, then two
//!   pre-norm text blocks of their own (fused `qkv` with biases, per-head RMS norms, a SiTU
//!   feed-forward with separate gate / up projections), run once per prompt.
//! - Time: the timestep (`sigma * 1000`) as 256 sinusoids (cosines first, in float64), an
//!   MLP to the model width, then one modulation for every block (`shared_block_modulation`: shift,
//!   scale and gate for the attention and the feed-forward) and one for the output head. There is
//!   no SiLU between the time MLP and the modulations.
//! - Blocks: RMS norm (with weight) `* (1 + scale) + shift`, q / k / v without biases, per-head RMS
//!   norms, rotary embeddings, joint attention over every token, gated residual; the
//!   feed-forward's fused `ffn_up_gate` gives the value half then the gate half, combined by SiTU
//!   `(4 tanh(g / 4) sigmoid(g)) * (25 tanh(h / 25))`.
//! - Rotary embeddings: 4 axes of 16 frequencies each (`base^(-i / 16)`), rotate-half layout
//!   (dimension `i` pairs with `i + 64`). Image token `(y, x)` sits at `(0, L, y - (H - 1) / 2,
//!   x - (W - 1) / 2)` for a prompt of `L` tokens, text token `j` at `(0, j, 0, 0)`.
//! - Output: the image tokens through the final RMS norm, the head modulation and a linear layer to
//!   the 128 latent channels (the FLUX.2 VAE's 32 channels in 2 x 2 patches).
//!
//! Weights load from the diffusers `transformer/` folder (separate `proj.q` / `k` / `v`), from
//! ComfyUI's single file (`proj.qkv` fused, with or without `model.diffusion_model.`) and from GGUF
//! files in that layout (`molbal/Kromoss-GGUF`: Q4_K / Q6_K / Q8_0 blocks). The fused matrix is
//! split by rows without copying.

use super::convrot;
use super::lora::LoraSet;
use crate::flux::transformer::{gated_add, rms_heads, silu};
use crate::flux::weights::{Dense, Loader, forward_many};
use crate::gpu::dit::Lin as DitLin;
use crate::kernels::for_chunks;
use crate::model::{Lin, Quant};
use crate::safetensors::ModelFiles;
use anyhow::{Result, bail};
use std::path::Path;
use std::sync::Mutex;

/// `config.json` of the transformer (the release's defaults where a key is missing).
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub dim: usize,
    pub ffn_dim: usize,
    pub heads: usize,
    pub layers: usize,
    pub text_dim: usize,
    pub freq_dim: usize,
    pub latent_channels: usize,
    pub rope_base: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            dim: 3072,
            ffn_dim: 8192,
            heads: 24,
            layers: 42,
            text_dim: 4096,
            freq_dim: 256,
            latent_channels: 128,
            rope_base: 10000.0,
        }
    }
}

impl Config {
    pub fn from_json(v: &serde_json::Value) -> Self {
        let d = Self::default();
        let u = |k: &str, def: usize| v[k].as_u64().map_or(def, |x| x as usize);
        Self {
            dim: u("dim", d.dim),
            ffn_dim: u("ffn_dim", d.ffn_dim),
            heads: u("num_heads", d.heads),
            layers: u("num_layers", d.layers),
            text_dim: u("text_dim", d.text_dim),
            freq_dim: u("freq_dim", d.freq_dim),
            latent_channels: u("latent_channels", d.latent_channels),
            rope_base: v["rope_base"].as_f64().unwrap_or(d.rope_base),
        }
    }

    pub fn head_dim(&self) -> usize {
        self.dim / self.heads
    }
}

/// Rotary tables of a token sequence: per token, the `head_dim / 2` angles' cosines and sines.
pub struct Rope {
    cos: Vec<f32>,
    sin: Vec<f32>,
    half: usize,
}

impl Rope {
    /// `pos` holds 4 coordinates per token.
    pub fn new(pos: &[[f32; 4]], head_dim: usize, base: f64) -> Self {
        let half = head_dim / 2;
        let bands = half / 4;
        // As the release: float32 frequencies `1 / base^(i / bands)`, float32 phases.
        let freq: Vec<f32> = (0..bands).map(|i| (1.0 / base.powf(i as f64 / bands as f64)) as f32).collect();
        let mut cos = vec![0f32; pos.len() * half];
        let mut sin = vec![0f32; pos.len() * half];
        for (t, p) in pos.iter().enumerate() {
            for a in 0..4 {
                for (b, &f) in freq.iter().enumerate() {
                    let ph = (p[a] * f) as f64;
                    cos[t * half + a * bands + b] = ph.cos() as f32;
                    sin[t * half + a * bands + b] = ph.sin() as f32;
                }
            }
        }
        Self { cos, sin, half }
    }

    /// The table of tokens `r0..r1`.
    pub fn slice(&self, r0: usize, r1: usize) -> Rope {
        let h = self.half;
        Rope {
            cos: self.cos[r0 * h..r1 * h].to_vec(),
            sin: self.sin[r0 * h..r1 * h].to_vec(),
            half: h,
        }
    }

    /// Rotates every head of `x` (`[tokens, heads * 2 half]`): `(a, b) -> (a cos - b sin, b cos + a
    /// sin)` for dimension pairs `(i, i + half)`.
    pub fn apply(&self, x: &mut [f32], heads: usize) {
        let _t = crate::prof::start(crate::prof::Cat::Elementwise);
        let h = self.half;
        let d = heads * 2 * h;
        for_chunks(x, d, x.len() > d * 16, |t, row| {
            let (c, s) = (&self.cos[t * h..(t + 1) * h], &self.sin[t * h..(t + 1) * h]);
            for head in row.chunks_mut(2 * h) {
                let (a, b) = head.split_at_mut(h);
                for i in 0..h {
                    let (x0, x1) = (a[i], b[i]);
                    a[i] = x0 * c[i] - x1 * s[i];
                    b[i] = x1 * c[i] + x0 * s[i];
                }
            }
        });
    }
}

/// Token coordinates: `lh * lw` image tokens (row-major) then `text_len` text tokens.
pub fn positions(lh: usize, lw: usize, text_len: usize) -> Vec<[f32; 4]> {
    let (cy, cx) = ((lh as f32 - 1.0) / 2.0, (lw as f32 - 1.0) / 2.0);
    let mut p = Vec::with_capacity(lh * lw + text_len);
    for y in 0..lh {
        for x in 0..lw {
            p.push([0.0, text_len as f32, y as f32 - cy, x as f32 - cx]);
        }
    }
    p.extend((0..text_len).map(|j| [0.0, j as f32, 0.0, 0.0]));
    p
}

/// `(4 tanh(g / 4) sigmoid(g)) * (25 tanh(h / 25))`, with `tanh(y) = 2 / (1 + e^(-2y)) - 1` on
/// the vectorizable exponential (the feed-forward evaluates it 2.8e8 times a pass at 1024 x 1024).
#[inline(always)]
pub fn situ(h: f32, g: f32) -> f32 {
    let e = crate::kernels::fast_exp;
    let tg = 2.0 / (1.0 + e(-0.5 * g)) - 1.0;
    let sg = 1.0 / (1.0 + e(-g));
    let th = 2.0 / (1.0 + e(-0.08 * h)) - 1.0;
    100.0 * tg * sg * th
}

/// RMS norm (eps 1e-6) with weight `w`, then `* (1 + scale) + shift` when a modulation is given.
fn rms_mod(x: &[f32], w: &[f32], m: Option<(&[f32], &[f32])>) -> Vec<f32> {
    let _t = crate::prof::start(crate::prof::Cat::Norm);
    let d = w.len();
    let mut y = vec![0f32; x.len()];
    for_chunks(&mut y, d, x.len() > d * 16, |r, out| {
        let row = &x[r * d..(r + 1) * d];
        let ms = row.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / d as f64;
        let inv = (1.0 / (ms + 1e-6).sqrt()) as f32;
        match m {
            Some((scale, shift)) => {
                for i in 0..d {
                    out[i] = row[i] * inv * w[i] * (1.0 + scale[i]) + shift[i];
                }
            }
            None => {
                for i in 0..d {
                    out[i] = row[i] * inv * w[i];
                }
            }
        }
    });
    y
}

/// SiTU over `[rows, 2 f]` (value half, then gate half) into `[rows, f]`.
fn situ_halves(hg: &[f32], f: usize) -> Vec<f32> {
    let _t = crate::prof::start(crate::prof::Cat::Elementwise);
    let rows = hg.len() / (2 * f);
    let mut y = vec![0f32; rows * f];
    for_chunks(&mut y, f, rows > 16, |r, out| {
        let (h, g) = hg[r * 2 * f..(r + 1) * 2 * f].split_at(f);
        for i in 0..f {
            out[i] = situ(h[i], g[i]);
        }
    });
    y
}

fn situ_pair(h: &[f32], g: &[f32]) -> Vec<f32> {
    let mut y = vec![0f32; h.len()];
    for_chunks(&mut y, 1 << 14, h.len() > 1 << 15, |c, out| {
        let o = c << 14;
        for (i, v) in out.iter_mut().enumerate() {
            *v = situ(h[o + i], g[o + i]);
        }
    });
    y
}

fn add(x: &mut [f32], y: &[f32]) {
    for_chunks(x, 1 << 14, x.len() > 1 << 15, |c, out| {
        let o = c << 14;
        for (i, v) in out.iter_mut().enumerate() {
            *v += y[o + i];
        }
    });
}

/// A low-rank LoRA path beside a layer, as two matmuls (`up (down x)`, the strengths folded into
/// `up`) over float16 weights, the rank padded to a multiple of 64 with zeros: they run wherever
/// the base matmul runs (on the GPU when there is one, on the matrix units in half precision
/// there; the update is a small part of the output, so its rounding stays well below the bf16
/// base weights').
pub struct Side {
    down: Dense<'static>,
    up: Dense<'static>,
}

/// An f32 matrix `[n_out, n_in]` as a float16 "GGUF" layer.
fn f16_dense(w: &[f32], n_out: usize, n_in: usize) -> Dense<'static> {
    let mut v = crate::quant::PageVec::<u8>::zeroed(w.len() * 2);
    for (o, x) in v.as_chunks_mut::<2>().0.iter_mut().zip(w) {
        *o = half::f16::from_f32(*x).to_le_bytes();
    }
    let m = crate::ggml::GgufMatrix {
        kind: crate::ggml::GgmlType::F16,
        data: v.into(),
        n_out,
        n_in,
        block_scales: None,
        x4: false,
    };
    Dense {
        w: Lin::gguf(m),
        b: None,
        n_out,
        n_in,
    }
}

/// The side path of `down` (`[r, n_in]`) and `up` (`[n_out, r]`), the rank padded to 64.
fn side(a: &[f32], b: &[f32], r: usize, n_in: usize, n_out: usize) -> Side {
    let rp = r.next_multiple_of(64);
    let mut ap = a.to_vec();
    ap.resize(rp * n_in, 0.0);
    let mut bp = vec![0f32; n_out * rp];
    for (o, row) in bp.chunks_mut(rp).zip(b.chunks(r)) {
        o[..r].copy_from_slice(row);
    }
    Side {
        down: f16_dense(&ap, rp, n_in),
        up: f16_dense(&bp, n_out, rp),
    }
}

/// A layer as the device's whole-pass plans see it (its device state uploaded on first use).
fn dit_dense<'a>(d: &'a Dense) -> Option<DitLin<'a>> {
    Some(DitLin {
        w: d.w.as_ref(),
        // A rotated weight's input is rotated by the plan (or, for a side path, is the rotated
        // input of its layer).
        dev: d.w.device_weight_unfolded(d.n_out, d.n_in)?,
        n_out: d.n_out,
        n_in: d.n_in,
        bias: d.b.as_deref(),
        side: None,
        side_gains: Vec::new(),
        rot: None,
    })
}

/// A linear layer with an optional LoRA side path.
pub struct Layer<'a> {
    d: Dense<'a>,
    side: Option<Side>,
    /// ComfyUI's ConvRot: the weight is stored rotated over groups of this many inputs (the
    /// `Lin` carries the input transform for the CPU path).
    rot: Option<usize>,
}

impl Layer<'_> {
    pub fn n_out(&self) -> usize {
        self.d.n_out
    }

    fn dit(&self) -> Option<DitLin<'_>> {
        let mut l = dit_dense(&self.d)?;
        if self.d.w.has_lora() {
            return None;
        }
        l.rot = self.rot;
        if let Some(s) = &self.side {
            l.side = Some((Box::new(dit_dense(&s.down)?), Box::new(dit_dense(&s.up)?)));
        }
        Some(l)
    }

    fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let mut y = self.d.forward(x, rows);
        self.add_side(x, rows, &mut y);
        y
    }

    fn forward_into(&self, x: &[f32], rows: usize, y: &mut [f32]) {
        self.d.forward_into(x, rows, y);
        self.add_side(x, rows, y);
    }

    fn forward_vec(&self, x: &[f32]) -> Vec<f32> {
        let mut y = self.d.forward_vec(x);
        if let Some(s) = &self.side {
            let u = s.down.forward(x, 1);
            for (a, b) in y.iter_mut().zip(s.up.forward(&u, 1)) {
                *a += b;
            }
        }
        y
    }

    fn add_side(&self, x: &[f32], rows: usize, y: &mut [f32]) {
        if let Some(s) = &self.side {
            let u = s.down.forward(x, rows);
            add(y, &s.up.forward(&u, rows));
        }
    }

    fn bytes(&self) -> usize {
        self.d.bytes() + self.side.as_ref().map_or(0, |s| s.down.bytes() + s.up.bytes())
    }
}

/// Several layers over one input (one batch of base matmuls, then the LoRA paths).
fn forward_many_l(ls: &[&Layer], x: &[f32], rows: usize) -> Vec<Vec<f32>> {
    let ds: Vec<&Dense> = ls.iter().map(|l| &l.d).collect();
    let mut ys = forward_many(&ds, x, rows);
    let sides: Vec<(usize, &Side)> = ls.iter().enumerate().filter_map(|(i, l)| l.side.as_ref().map(|s| (i, s))).collect();
    if !sides.is_empty() {
        let downs: Vec<&Dense> = sides.iter().map(|(_, s)| &s.down).collect();
        let us = forward_many(&downs, x, rows);
        for ((i, s), u) in sides.iter().zip(us) {
            add(&mut ys[*i], &s.up.forward(&u, rows));
        }
    }
    ys
}

/// One of the two text-refinement blocks.
struct TextBlock<'a> {
    norm1: Vec<f32>,
    norm2: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    qkv: Layer<'a>,
    proj: Layer<'a>,
    gate: Layer<'a>,
    up: Layer<'a>,
    down: Layer<'a>,
}

/// One main block.
struct Block<'a> {
    norm1: Vec<f32>,
    norm2: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    qkv: [Layer<'a>; 3],
    o: Layer<'a>,
    up_gate: Layer<'a>,
    down: Layer<'a>,
}

/// A prompt ready for the blocks: the text embedder's output and its length.
pub struct TextCond {
    pub tokens: Vec<f32>,
    pub len: usize,
}

/// One pass of [`Transformer::forward_batch`]: a prompt and the blocks left out.
pub struct Pass<'a> {
    pub text: &'a TextCond,
    pub skip: &'a [usize],
}

/// How [`Transformer::forward_batch`] uses the device.
#[derive(Clone, Copy, Debug)]
pub struct GpuPass {
    /// Whole passes on the device when it can run them.
    pub enabled: bool,
    /// Attention with float32 operands there (else half).
    pub exact_attention: bool,
    /// The passes of a step in one batch.
    pub batch: bool,
}

impl Default for GpuPass {
    fn default() -> Self {
        Self {
            enabled: true,
            exact_attention: false,
            batch: true,
        }
    }
}

/// The per-step modulations.
pub struct StepMod {
    /// shift, scale, gate (attention), shift, scale, gate (feed-forward); `dim` each.
    pub block: Vec<f32>,
    /// shift, scale of the output head.
    pub head: Vec<f32>,
}

pub struct Transformer {
    pub cfg: Config,
    video_emb: Layer<'static>,
    text_proj: Layer<'static>,
    text_blocks: Vec<TextBlock<'static>>,
    time: [Layer<'static>; 2],
    shared_mod: Layer<'static>,
    head_mod: Layer<'static>,
    final_norm: Vec<f32>,
    final_linear: Layer<'static>,
    blocks: Vec<Block<'static>>,
    pub weight_bytes: usize,
    /// Report lines of the LoRA files applied at load.
    pub lora_report: Vec<String>,
    /// The device's whole-pass session (activation buffers), made on first use.
    gpu: Mutex<Option<Box<dyn crate::gpu::dit::Session>>>,
    // Declared last: dropped after every layer borrowing it.
    _files: Box<ModelFiles>,
}

/// Prefixes the transformer's tensors carry in single-file checkpoints.
const PREFIXES: [&str; 3] = ["model.diffusion_model.", "diffusion_model.", "transformer."];

/// Optional `OXINFER_MOSS_LAYERS=N`: run only the first N blocks (checks against a truncated
/// reference).
fn layer_limit() -> Option<usize> {
    std::env::var("OXINFER_MOSS_LAYERS").ok()?.trim().parse().ok()
}

/// Reads one logical layer, with the LoRA updates of the set: merged into the weights (full-rank
/// LoKr updates, or every update with `merge`) or kept as a runtime low-rank side path.
struct LayerReader<'f, 'l> {
    ld: Loader<'f>,
    lora: &'l LoraSet,
}

impl<'f> LayerReader<'f, '_> {
    /// Rows `r0..r1` of tensor `{tensor}.weight` as the layer named `name` (diffusers module
    /// path), quantized to the loader's precision when `quantize`.
    fn rows(&self, tensor: &str, r0: usize, r1: usize, name: &str, quantize: bool) -> Result<Layer<'f>> {
        let merged = self.lora.merged(name);
        let cr = convrot::conf(self.ld.files, tensor)?;
        let rot = cr.as_ref().and_then(|c| c.rot);
        let d = if let Some(c) = &cr {
            let base = convrot::rows(self.ld.files, tensor, c, r0, r1)?;
            if merged {
                // Merged in the original basis, then rotated back and quantized to int8 blocks.
                let (n, n_in) = (base.n_out, base.n_in);
                let mut w = vec![0f32; n * n_in];
                let idx: Vec<u32> = (0..n as u32).collect();
                if !base.w.rows(&idx, n_in, &mut w) {
                    bail!("{tensor}: rows cannot be read back for a LoRA merge");
                }
                convrot::rotate_rows(&mut w, rot);
                self.lora.merge_into(name, &mut w, n, n_in)?;
                convrot::rotate_rows(&mut w, rot);
                Dense {
                    w: Lin::q8(crate::quant::quantize_f32(&w, n, n_in)),
                    b: base.b,
                    n_out: n,
                    n_in,
                }
            } else {
                base
            }
        } else if merged {
            let base = self.ld.rows(tensor, r0, r1, false)?;
            let (n, n_in) = (base.n_out, base.n_in);
            let mut w = vec![0f32; n * n_in];
            let idx: Vec<u32> = (0..n as u32).collect();
            if !base.w.rows(&idx, n_in, &mut w) {
                bail!("{tensor}: rows cannot be read back for a LoRA merge");
            }
            self.lora.merge_into(name, &mut w, n, n_in)?;
            let q = if quantize && n_in.is_multiple_of(crate::quant::QBLOCK) {
                self.ld.quant
            } else {
                Quant::Bf16
            };
            let w = match q {
                Quant::Bf16 => {
                    let mut v = crate::quant::PageVec::<u16>::zeroed(n * n_in);
                    for (o, &x) in v.iter_mut().zip(&w) {
                        *o = half::bf16::from_f32(x).to_bits();
                    }
                    Lin::bf16_owned(v)
                }
                Quant::Q8 => Lin::q8(crate::quant::quantize_f32(&w, n, n_in)),
                Quant::Q4 => Lin::q4(crate::quant::quantize_q4_f32(&w, n, n_in)),
            };
            Dense {
                w,
                b: base.b,
                n_out: n,
                n_in,
            }
        } else {
            self.ld.rows(tensor, r0, r1, quantize)?
        };
        let mut d = d;
        let fold = rot.map(convrot::fold);
        d.w.fold = fold.clone();
        let mut side = None;
        if let Some((mut a, b, r)) = self.lora.runtime(name, d.n_out, d.n_in)? {
            if crate::gpu::active().is_some() {
                // On the accelerator: two matmuls beside the base one. Beside a rotated weight
                // the side path takes the rotated input too: `A x = (A H) (H x)`.
                convrot::rotate_rows(&mut a, rot);
                let mut s = self::side(&a, &b, r, d.n_in, d.n_out);
                s.down.w.fold = fold;
                side = Some(s);
            } else {
                // On the CPU: the low-rank kernel after the base matmul.
                let (n_out, n_in) = (d.n_out, d.n_in);
                d.w.set_lora(Some(Box::new(crate::lora::LoraDelta::fixed(a, &b, r, n_in, n_out))));
            }
        }
        Ok(Layer { d, side, rot })
    }

    fn dense(&self, tensor: &str, name: &str, quantize: bool) -> Result<Layer<'f>> {
        let (rows, _) = self.ld.shape2(&format!("{tensor}.weight"))?;
        self.rows(tensor, 0, rows, name, quantize)
    }

    fn norm(&self, name: &str, n: usize) -> Result<Vec<f32>> {
        let v = self.ld.f32(&format!("{name}.weight"))?;
        if v.len() != n {
            bail!("{name}.weight: {} values, expected {n}", v.len());
        }
        Ok(v)
    }
}

impl Transformer {
    /// Loads a diffusers `transformer/` folder, a single safetensors file or a GGUF file, with the
    /// LoRA set applied. Float weights of the blocks are quantized to `quant` (bf16 keeps them as
    /// stored, borrowed from the mapping); GGUF weights run in their own format.
    pub fn load(path: &Path, quant: Quant, lora: &LoraSet, in_memory: bool) -> Result<Self> {
        let gguf = crate::gguf::is_gguf(path);
        let mut files = if !gguf && path.is_file() {
            ModelFiles::open_file(path, in_memory)?
        } else {
            ModelFiles::open(path, in_memory)?
        };
        if let Some(p) = PREFIXES.iter().find(|p| files.has(&format!("{p}video_embedder.weight"))) {
            let p = p.to_string();
            files.rename(|n| Some(n.strip_prefix(&p).map_or_else(|| format!("other.{n}"), str::to_string)));
        }
        if !files.has("video_embedder.weight") {
            bail!("{}: no video_embedder.weight (not a MOSS-Image transformer)", path.display());
        }
        for probe in ["layers.0.proj.qkv", "layers.0.proj.q"] {
            if let Some(i) = files.info(&format!("{probe}.weight"))
                && !matches!(i.dtype.as_str(), "BF16" | "F16" | "F32")
                && files.ggml(&format!("{probe}.weight")).is_none_or(|g| g.is_err())
                && convrot::conf(&files, probe)?.is_none()
            {
                bail!(
                    "{}: block weights stored as {} without a ComfyUI quantization entry are not supported",
                    path.display(),
                    i.dtype
                );
            }
        }
        let files = Box::new(files);
        // SAFETY: the box's heap block does not move, and every layer borrowing it is dropped
        // before `_files` (field order).
        let fs: &'static ModelFiles = unsafe { &*(files.as_ref() as *const ModelFiles) };
        let rd = LayerReader {
            ld: Loader { files: fs, quant },
            lora,
        };
        let cfg_path = path.join("config.json");
        let mut cfg = if !gguf && path.is_dir() && crate::fsx::exists(&cfg_path) {
            Config::from_json(&serde_json::from_slice(&crate::fsx::read(&cfg_path)?)?)
        } else {
            Self::infer_config(&rd.ld)?
        };
        if let Some(n) = layer_limit() {
            cfg.layers = cfg.layers.min(n);
        }
        let (d, f, hd) = (cfg.dim, cfg.ffn_dim, cfg.head_dim());
        let fused = fs.has("layers.0.proj.qkv.weight");
        let top = |t: &str| rd.dense(t, t, false);
        let video_emb = top("video_embedder")?;
        if video_emb.d.n_in != cfg.latent_channels || video_emb.d.n_out != d {
            bail!(
                "video_embedder is [{} x {}], expected [{d} x {}]",
                video_emb.d.n_out,
                video_emb.d.n_in,
                cfg.latent_channels
            );
        }
        let text_proj = top("text_embedder.proj")?;
        let mut text_blocks = Vec::new();
        for j in 0..2 {
            let p = format!("text_embedder.blocks.{j}");
            let l = |n: &str| rd.dense(&format!("{p}.{n}"), &format!("{p}.{n}"), false);
            text_blocks.push(TextBlock {
                norm1: rd.norm(&format!("{p}.norm1"), d)?,
                norm2: rd.norm(&format!("{p}.norm2"), d)?,
                q_norm: rd.norm(&format!("{p}.q_norm"), hd)?,
                k_norm: rd.norm(&format!("{p}.k_norm"), hd)?,
                qkv: l("qkv")?,
                proj: l("proj")?,
                gate: l("ffn_gate")?,
                up: l("ffn_up")?,
                down: l("ffn_down")?,
            });
        }
        let time = [top("time_embedding.0")?, top("time_embedding.2")?];
        let shared_mod = top("shared_block_modulation")?;
        let head_mod = top("video_head_modulation")?;
        if shared_mod.d.n_out != 6 * d || head_mod.d.n_out != 2 * d {
            bail!(
                "modulation layers have {} / {} rows, expected {} / {}",
                shared_mod.d.n_out,
                head_mod.d.n_out,
                6 * d,
                2 * d
            );
        }
        let final_norm = rd.norm("final_norm_video", d)?;
        let final_linear = top("final_linear_video")?;
        let mut blocks = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let p = format!("layers.{i}");
            let name = |n: &str| format!("{p}.proj.{n}");
            let qkv = if fused {
                let t = name("qkv");
                [
                    rd.rows(&t, 0, d, &name("q"), true)?,
                    rd.rows(&t, d, 2 * d, &name("k"), true)?,
                    rd.rows(&t, 2 * d, 3 * d, &name("v"), true)?,
                ]
            } else {
                [
                    rd.dense(&name("q"), &name("q"), true)?,
                    rd.dense(&name("k"), &name("k"), true)?,
                    rd.dense(&name("v"), &name("v"), true)?,
                ]
            };
            let up_gate = rd.dense(&name("ffn_up_gate"), &name("ffn_up_gate"), true)?;
            if up_gate.d.n_out != 2 * f {
                bail!("{p}.proj.ffn_up_gate has {} rows, expected {}", up_gate.d.n_out, 2 * f);
            }
            blocks.push(Block {
                norm1: rd.norm(&format!("{p}.norm1"), d)?,
                norm2: rd.norm(&format!("{p}.norm2"), d)?,
                q_norm: rd.norm(&format!("{p}.q_norm"), hd)?,
                k_norm: rd.norm(&format!("{p}.k_norm"), hd)?,
                qkv,
                o: rd.dense(&name("o"), &name("o"), true)?,
                up_gate,
                down: rd.dense(&name("ffn_down"), &name("ffn_down"), true)?,
            });
        }
        let lora_report = lora.finish(cfg.layers < Self::infer_layers(fs))?;
        let mut weight_bytes = [&video_emb, &text_proj, &time[0], &time[1], &shared_mod, &head_mod, &final_linear]
            .iter()
            .map(|l| l.bytes())
            .sum::<usize>();
        for b in &text_blocks {
            weight_bytes += [&b.qkv, &b.proj, &b.gate, &b.up, &b.down].iter().map(|l| l.bytes()).sum::<usize>();
        }
        for b in &blocks {
            weight_bytes += b.qkv.iter().chain([&b.o, &b.up_gate, &b.down]).map(|l| l.bytes()).sum::<usize>();
        }
        Ok(Self {
            cfg,
            video_emb,
            text_proj,
            text_blocks,
            time,
            shared_mod,
            head_mod,
            final_norm,
            final_linear,
            blocks,
            weight_bytes,
            lora_report,
            gpu: Mutex::new(None),
            _files: files,
        })
    }

    /// The configuration of the checkpoint at `path` without loading it.
    pub fn peek_config(path: &Path) -> Result<Config> {
        let cfg_path = path.join("config.json");
        if path.is_dir() && crate::fsx::exists(&cfg_path) {
            return Ok(Config::from_json(&serde_json::from_slice(&crate::fsx::read(&cfg_path)?)?));
        }
        let mut files = if !crate::gguf::is_gguf(path) && path.is_file() {
            ModelFiles::open_file(path, false)?
        } else {
            ModelFiles::open(path, false)?
        };
        if let Some(p) = PREFIXES.iter().find(|p| files.has(&format!("{p}video_embedder.weight"))) {
            let p = p.to_string();
            files.rename(|n| Some(n.strip_prefix(&p).map_or_else(|| format!("other.{n}"), str::to_string)));
        }
        Self::infer_config(&Loader {
            files: &files,
            quant: Quant::Bf16,
        })
    }

    fn infer_layers(fs: &ModelFiles) -> usize {
        (0..).take_while(|i| fs.has(&format!("layers.{i}.norm1.weight"))).count()
    }

    /// The configuration of a checkpoint without `config.json` (single files, GGUF).
    fn infer_config(ld: &Loader) -> Result<Config> {
        let (d, lat) = ld.shape2("video_embedder.weight")?;
        let (td_out, td) = ld.shape2("text_embedder.proj.weight")?;
        let (_, freq) = ld.shape2("time_embedding.0.weight")?;
        let (f2, _) = ld.shape2("layers.0.proj.ffn_up_gate.weight")?;
        let hd = ld.f32("layers.0.q_norm.weight")?.len();
        if td_out != d || hd == 0 || d % hd != 0 {
            bail!("inconsistent MOSS-Image shapes (dim {d}, text projection {td_out}, head {hd})");
        }
        Ok(Config {
            dim: d,
            ffn_dim: f2 / 2,
            heads: d / hd,
            layers: Self::infer_layers(ld.files),
            text_dim: td,
            freq_dim: freq,
            latent_channels: lat,
            rope_base: 10000.0,
        })
    }

    /// Bytes of weights held (borrowed from the mapping or converted).
    pub fn bytes(&self) -> usize {
        self.weight_bytes
    }

    /// The text embedder's output for Qwen3.5 hidden states `text` (`[len, text_dim]`).
    pub fn embed_text(&self, text: &[f32]) -> Result<TextCond> {
        let c = &self.cfg;
        if text.is_empty() || !text.len().is_multiple_of(c.text_dim) {
            bail!("text conditioning of {} values is not a multiple of {}", text.len(), c.text_dim);
        }
        let len = text.len() / c.text_dim;
        let rope = Rope::new(&positions(0, 0, len), c.head_dim(), c.rope_base);
        let mut x = self.text_proj.forward(text, len);
        for b in &self.text_blocks {
            let y = rms_mod(&x, &b.norm1, None);
            let qkv = b.qkv.forward(&y, len);
            let d = c.dim;
            let (mut q, mut k, mut v) = (vec![0f32; len * d], vec![0f32; len * d], vec![0f32; len * d]);
            for t in 0..len {
                let r = &qkv[t * 3 * d..(t + 1) * 3 * d];
                q[t * d..(t + 1) * d].copy_from_slice(&r[..d]);
                k[t * d..(t + 1) * d].copy_from_slice(&r[d..2 * d]);
                v[t * d..(t + 1) * d].copy_from_slice(&r[2 * d..]);
            }
            rms_heads(&mut q, &b.q_norm, c.head_dim());
            rms_heads(&mut k, &b.k_norm, c.head_dim());
            rope.apply(&mut q, c.heads);
            rope.apply(&mut k, c.heads);
            let a = crate::sd::ops::attention(&q, &k, &v, len, len, c.heads, c.head_dim(), false);
            add(&mut x, &b.proj.forward(&a, len));
            let y = rms_mod(&x, &b.norm2, None);
            let mut hg = forward_many_l(&[&b.up, &b.gate], &y, len);
            let g = hg.pop().unwrap();
            let h = hg.pop().unwrap();
            add(&mut x, &b.down.forward(&situ_pair(&h, &g), len));
        }
        Ok(TextCond { tokens: x, len })
    }

    /// The modulations of timestep `t` (`sigma * 1000`).
    pub fn step_mod(&self, t: f32) -> StepMod {
        let half = self.cfg.freq_dim / 2;
        let mut e = vec![0f32; 2 * half];
        for i in 0..half {
            let f = 10000f64.powf(-(i as f64) / half as f64);
            let ph = t as f64 * f;
            e[i] = ph.cos() as f32;
            e[half + i] = ph.sin() as f32;
        }
        let mut h = self.time[0].forward_vec(&e);
        h.iter_mut().for_each(|v| *v = silu(*v));
        let time = self.time[1].forward_vec(&h);
        StepMod {
            block: self.shared_mod.forward_vec(&time),
            head: self.head_mod.forward_vec(&time),
        }
    }

    /// The velocities of latent tokens `x` (`[lh * lw, latent_channels]`, row-major) under several
    /// prompts / skip lists at once (the guidance branches of a step): with `gpu` set and a
    /// device that runs whole passes, in one batch on the device (attention with half-precision
    /// operands unless `exact_attention`, every branch in one batch with `batch`), else one
    /// [`Self::forward`] each.
    pub fn forward_batch(&self, x: &[f32], lh: usize, lw: usize, m: &StepMod, passes: &[Pass], gpu: GpuPass) -> Vec<Vec<f32>> {
        if gpu.enabled
            && let Some(v) = self.forward_device(x, lh, lw, m, passes, gpu)
        {
            return v;
        }
        passes.iter().map(|p| self.forward(x, lh, lw, p.text, m, p.skip)).collect()
    }

    fn forward_device(&self, x: &[f32], lh: usize, lw: usize, m: &StepMod, passes: &[Pass], gpu: GpuPass) -> Option<Vec<Vec<f32>>> {
        let be = crate::gpu::active()?;
        let _t = crate::prof::start(crate::prof::Cat::Matmul);
        let c = &self.cfg;
        let mut blocks = Vec::with_capacity(self.blocks.len());
        for b in &self.blocks {
            blocks.push(crate::gpu::dit::Block {
                norm1: &b.norm1,
                norm2: &b.norm2,
                q_norm: &b.q_norm,
                k_norm: &b.k_norm,
                q: b.qkv[0].dit()?,
                k: b.qkv[1].dit()?,
                v: b.qkv[2].dit()?,
                o: b.o.dit()?,
                up_gate: b.up_gate.dit()?,
                down: b.down.dit()?,
            });
        }
        let model = crate::gpu::dit::Model {
            dim: c.dim,
            heads: c.heads,
            head_dim: c.head_dim(),
            ffn: c.ffn_dim,
            latent: c.latent_channels,
            eps: 1e-6,
            video: self.video_emb.dit()?,
            blocks,
            final_norm: &self.final_norm,
            final_linear: self.final_linear.dit()?,
            half_attention: !gpu.exact_attention,
        };
        let ropes: Vec<Rope> = passes
            .iter()
            .map(|p| Rope::new(&positions(lh, lw, p.text.len), c.head_dim(), c.rope_base))
            .collect();
        let seqs: Vec<crate::gpu::dit::Seq> = passes
            .iter()
            .zip(&ropes)
            .map(|(p, r)| crate::gpu::dit::Seq {
                text: &p.text.tokens,
                text_len: p.text.len,
                cos: &r.cos,
                sin: &r.sin,
                skip: p.skip,
            })
            .collect();
        let mut sess = self.gpu.lock().unwrap_or_else(|e| e.into_inner());
        if sess.is_none() {
            *sess = Some(be.dit_session()?);
        }
        let s = sess.as_mut()?;
        let mut out = vec![Vec::new(); passes.len()];
        let ok = if gpu.batch {
            s.forward(&model, x, lh * lw, &seqs, &m.block, &m.head, &mut out)
        } else {
            seqs.iter().zip(out.iter_mut()).all(|(q, o)| {
                s.forward(
                    &model,
                    x,
                    lh * lw,
                    std::slice::from_ref(q),
                    &m.block,
                    &m.head,
                    std::slice::from_mut(o),
                )
            })
        };
        ok.then_some(out)
    }

    /// Drops the device session's buffers (they are made again on the next device pass).
    pub fn release_device(&self) {
        *self.gpu.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }

    /// The velocity of latent tokens `x` (`[lh * lw, latent_channels]`, row-major) under `text`,
    /// with the blocks of `skip` left out.
    pub fn forward(&self, x: &[f32], lh: usize, lw: usize, text: &TextCond, m: &StepMod, skip: &[usize]) -> Vec<f32> {
        let c = &self.cfg;
        let (d, hd) = (c.dim, c.head_dim());
        let ni = lh * lw;
        let n = ni + text.len;
        let rope = Rope::new(&positions(lh, lw, text.len), hd, c.rope_base);
        let mut h = vec![0f32; n * d];
        self.video_emb.forward_into(x, ni, &mut h[..ni * d]);
        h[ni * d..].copy_from_slice(&text.tokens);
        let md: Vec<&[f32]> = m.block.chunks(d).collect();
        for (i, b) in self.blocks.iter().enumerate() {
            if skip.contains(&i) {
                continue;
            }
            let y = rms_mod(&h, &b.norm1, Some((md[1], md[0])));
            let mut ys = forward_many_l(&[&b.qkv[0], &b.qkv[1], &b.qkv[2]], &y, n);
            drop(y);
            let v = ys.pop().unwrap();
            let mut k = ys.pop().unwrap();
            let mut q = ys.pop().unwrap();
            rms_heads(&mut q, &b.q_norm, hd);
            rms_heads(&mut k, &b.k_norm, hd);
            rope.apply(&mut q, c.heads);
            rope.apply(&mut k, c.heads);
            let a = {
                let _t = crate::prof::start(crate::prof::Cat::Attention);
                crate::sd::ops::attention(&q, &k, &v, n, n, c.heads, hd, false)
            };
            drop((q, k, v));
            gated_add(&mut h, &b.o.forward(&a, n), md[2]);
            drop(a);
            let y = rms_mod(&h, &b.norm2, Some((md[4], md[3])));
            let hg = b.up_gate.forward(&y, n);
            drop(y);
            let s = situ_halves(&hg, c.ffn_dim);
            drop(hg);
            gated_add(&mut h, &b.down.forward(&s, n), md[5]);
        }
        let (shift, scale) = m.head.split_at(d);
        let y = rms_mod(&h[..ni * d], &self.final_norm, Some((scale, shift)));
        self.final_linear.forward(&y, ni)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn situ_values() {
        // (4 tanh(0.25) sigmoid(1)) * (25 tanh(0.08))
        let want = (4.0 * 0.25f64.tanh() / (1.0 + (-1f64).exp())) * (25.0 * 0.08f64.tanh());
        assert!((situ(2.0, 1.0) as f64 - want).abs() < 1e-5);
        assert_eq!(situ(0.0, 3.0), 0.0);
        for i in -200..200 {
            let (h, g) = (i as f32 * 0.37, i as f32 * -0.23 + 1.0);
            let (hd, gd) = (h as f64, g as f64);
            let want = (4.0 * (gd / 4.0).tanh() / (1.0 + (-gd).exp())) * (25.0 * (hd / 25.0).tanh());
            assert!((situ(h, g) as f64 - want).abs() < 2e-5 * (1.0 + want.abs()), "situ({h}, {g})");
        }
    }

    #[test]
    fn token_positions() {
        let p = positions(2, 3, 4);
        assert_eq!(p.len(), 10);
        assert_eq!(p[0], [0.0, 4.0, -0.5, -1.0]);
        assert_eq!(p[5], [0.0, 4.0, 0.5, 1.0]);
        assert_eq!(p[6], [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(p[9], [0.0, 3.0, 0.0, 0.0]);
    }

    #[test]
    fn rope_rotate_half() {
        // One head of 16 (4 axes x 2 bands): token at (0, 1, 0, 0) rotates dims 2..4 by 1 and
        // 1/sqrt(base) radians against dims 10..12.
        let r = Rope::new(&[[0.0, 1.0, 0.0, 0.0]], 16, 100.0);
        let mut x = vec![0f32; 16];
        x[2] = 1.0;
        x[3] = 1.0;
        r.apply(&mut x, 1);
        assert!((x[2] - 1f32.cos()).abs() < 1e-6 && (x[10] - 1f32.sin()).abs() < 1e-6);
        assert!((x[3] - 0.1f32.cos()).abs() < 1e-6 && (x[11] - 0.1f32.sin()).abs() < 1e-6);
        assert_eq!(x[0], 0.0);
    }
}
