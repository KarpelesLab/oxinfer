//! The image tokenizer of Nemotron-Labs-Image (`Nemotron-Labs-Image-Tokenizer/`, transformers
//! `model_type` `l24sq_vit`, NVIDIA's `L24SQViTModel`): a ViT encoder that turns 16x16 pixel
//! patches into indices of a fixed codebook of 196,560 Leech-lattice vectors, and a ViT decoder
//! that renders a grid of indices back to pixels.
//!
//! Both halves are 24 Llama-style blocks of 1024 (16 heads of 64): RMSNorm (eps 1e-6) and
//! bidirectional attention with 2D RoPE (the first half of each head rotated by the patch's row,
//! the second half by its column, each half split in two as rotate-half RoPE does, base 10,000),
//! then RMSNorm and a SwiGLU MLP of 2752. No biases inside the blocks.
//!
//! - **Encoder**: pixels in `[-1, 1]` (padded by replication to a multiple of 16), a 16x16
//!   stride-16 convolution to 1024, the blocks, a final RMSNorm, `quant_embed` (1024 -> 24), and
//!   the nearest codebook vector by cosine (both sides L2-normalized; the first index on a tie).
//! - **Decoder**: the codebook rows of the indices (L2-normalized), `post_quant_embed` (24 ->
//!   1024), the blocks, a final RMSNorm, `ffn` (1024 -> 3072, tanh), `conv_out` (3072 -> 768 = 3 x
//!   16 x 16, channel-major within the patch) and the patches laid out row by row; values clamped
//!   to `[-1, 1]`.
//!
//! The decoder's `distill_mlp` (a training-time feature head) is not loaded.

use crate::kernels::*;
use crate::model::{Lin, Quant, linear, linear_multi, load_lin};
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};
use serde_json::Value;

/// `config.json` of the tokenizer.
#[derive(Clone, Debug)]
pub struct VitConfig {
    pub patch: usize,
    pub width: usize,
    pub layers: usize,
    pub heads: usize,
    /// SwiGLU width: `int(width * mlp_ratio * 2 / 3)` rounded up to a multiple of 64.
    pub mlp: usize,
    /// The decoder's output FFN width (`use_ffn_output`), else the head reads the blocks' output.
    pub ffn_out: Option<usize>,
    pub n_embed: usize,
    pub embed_dim: usize,
    pub theta: f64,
    pub l2_norm: bool,
    pub clamp: (f32, f32),
}

impl VitConfig {
    pub fn from_json(v: &Value) -> Result<Self> {
        if v["model_type"].as_str().is_some_and(|m| m != "l24sq_vit") {
            bail!("not an L24SQ ViT tokenizer (model_type {})", v["model_type"]);
        }
        let u = |k: &str, d: u64| v[k].as_u64().unwrap_or(d) as usize;
        let width = u("width", 1024);
        let ratio = v["mlp_ratio"].as_f64().unwrap_or(4.0);
        let raw = (width as f64 * ratio * 2.0 / 3.0) as usize;
        let clamp = v["clamp_range"]
            .as_array()
            .and_then(|a| Some((a.first()?.as_f64()? as f32, a.get(1)?.as_f64()? as f32)))
            .unwrap_or((-1.0, 1.0));
        let c = Self {
            patch: u("patch_size", 16),
            width,
            layers: u("layers", 24),
            heads: u("heads", 16),
            mlp: raw.div_ceil(64) * 64,
            ffn_out: v["use_ffn_output"].as_bool().unwrap_or(true).then(|| u("dim_ffn_output", 3072)),
            n_embed: u("n_embed", 196_560),
            embed_dim: u("embed_dim", 24),
            theta: v["vit_rope_theta"].as_f64().unwrap_or(10_000.0),
            l2_norm: v["l2_norm"].as_bool().unwrap_or(true),
            clamp,
        };
        if c.heads == 0 || !c.width.is_multiple_of(c.heads) || !(c.width / c.heads).is_multiple_of(4) {
            bail!(
                "tokenizer config: width {} over {} heads needs a head size that is a multiple of 4",
                c.width,
                c.heads
            );
        }
        if v["zero_mean"].as_bool() == Some(false) {
            bail!("tokenizer config: only zero_mean inputs ([-1, 1]) are supported");
        }
        Ok(c)
    }

    pub fn head_dim(&self) -> usize {
        self.width / self.heads
    }
}

const EPS: f32 = 1e-6;

struct Block<'a> {
    norm1: Vec<f32>,
    q: Lin<'a>,
    k: Lin<'a>,
    v: Lin<'a>,
    o: Lin<'a>,
    norm2: Vec<f32>,
    gate: Lin<'a>,
    up: Lin<'a>,
    down: Lin<'a>,
}

/// A linear layer with a bias.
struct Dense<'a> {
    w: Lin<'a>,
    b: Vec<f32>,
    n_out: usize,
    n_in: usize,
}

impl Dense<'_> {
    fn forward(&self, x: &[f32], t: usize) -> Vec<f32> {
        let mut y = vec![0f32; t * self.n_out];
        linear(&self.w, self.n_out, self.n_in, x, t, &mut y);
        let b = &self.b;
        for_chunks(&mut y, self.n_out, t > 16, |_, r| {
            for (v, bb) in r.iter_mut().zip(b) {
                *v += bb;
            }
        });
        y
    }
}

/// How the tokenizer's matrices are kept.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Precision {
    /// Rounded to bf16 (the reference runs the decoder in bf16); GPU matmuls when one is active.
    #[default]
    Bf16,
    /// Float32 as stored.
    F32,
}

struct Stack<'a> {
    blocks: Vec<Block<'a>>,
    norm: Vec<f32>,
}

struct Encoder<'a> {
    patch: Dense<'a>,
    stack: Stack<'a>,
    quant: Dense<'a>,
}

struct Decoder<'a> {
    post_quant: Dense<'a>,
    stack: Stack<'a>,
    ffn: Option<Dense<'a>>,
    head: Dense<'a>,
}

/// The L24SQ ViT tokenizer: [`Vit::encode`] (pixels to indices) and [`Vit::decode`].
pub struct Vit<'a> {
    pub cfg: VitConfig,
    /// L2-normalized codebook `[n_embed, embed_dim]`.
    codebook: Vec<f32>,
    enc: Option<Encoder<'a>>,
    dec: Option<Decoder<'a>>,
    /// Per head: the inverse frequencies of one axis (`head_dim / 4` of them).
    inv_freq: Vec<f32>,
}

fn normalize(v: &mut [f32]) {
    // F.normalize: v / max(||v||, 1e-12).
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter_mut().for_each(|x| *x /= n);
}

impl<'a> Vit<'a> {
    /// Loads the halves asked for from the tokenizer's weights (`encoder.*`, `decoder.*`,
    /// `quantize.embedding.weight`, `quant_embed`, `post_quant_embed`).
    pub fn load(cfg: VitConfig, files: &'a ModelFiles, precision: Precision, encoder: bool, decoder: bool) -> Result<Self> {
        let d = cfg.width;
        // Matrices `[n_out, n_in]`, or any tensor of that many values (the patch convolution's
        // `[1024, 3, 16, 16]`, read as `[1024, 768]`).
        let mat = |name: &str, n_out: usize, n_in: usize| -> Result<Lin<'a>> {
            let info = files.info(name).with_context(|| format!("missing tensor {name}"))?;
            if info.shape.iter().product::<usize>() != n_out * n_in {
                bail!("tensor {name} has shape {:?}, expected [{n_out}, {n_in}]", info.shape);
            }
            if info.dtype == "F32" && precision == Precision::F32 {
                return Lin::from_gguf(crate::ggml::GgmlType::F32, files.bytes(name)?, n_out, n_in);
            }
            if info.shape.len() == 2 {
                return load_lin(files, name, n_out, n_in, Quant::Bf16);
            }
            let v = files.f32(name)?;
            let mut w = crate::quant::PageVec::<u16>::zeroed(v.len());
            for (o, x) in w.iter_mut().zip(&v) {
                *o = half::bf16::from_f32(*x).to_bits();
            }
            Ok(Lin::bf16_owned(w))
        };
        let vecf = |name: &str, n: usize| -> Result<Vec<f32>> {
            let v = files.f32(name)?;
            if v.len() != n {
                bail!("{name}: {} values, expected {n}", v.len());
            }
            Ok(v)
        };
        let dense = |p: &str, n_out: usize, n_in: usize| -> Result<Dense<'a>> {
            Ok(Dense {
                w: mat(&format!("{p}.weight"), n_out, n_in)?,
                b: vecf(&format!("{p}.bias"), n_out)?,
                n_out,
                n_in,
            })
        };
        let stack = |p: &str| -> Result<Stack<'a>> {
            let mut blocks = Vec::with_capacity(cfg.layers);
            for l in 0..cfg.layers {
                let b = format!("{p}.blocks.{l}.");
                let m = |n: &str, o: usize, i: usize| mat(&format!("{b}{n}.weight"), o, i);
                blocks.push(Block {
                    norm1: vecf(&format!("{b}norm1.weight"), d)?,
                    q: m("attn.q_proj", d, d)?,
                    k: m("attn.k_proj", d, d)?,
                    v: m("attn.v_proj", d, d)?,
                    o: m("attn.out_proj", d, d)?,
                    norm2: vecf(&format!("{b}norm2.weight"), d)?,
                    gate: m("mlp.gate_proj", cfg.mlp, d)?,
                    up: m("mlp.up_proj", cfg.mlp, d)?,
                    down: m("mlp.down_proj", d, cfg.mlp)?,
                });
            }
            Ok(Stack {
                blocks,
                norm: vecf(&format!("{p}.norm.weight"), d)?,
            })
        };
        let (ne, ed) = (cfg.n_embed, cfg.embed_dim);
        let mut codebook = vecf("quantize.embedding.weight", ne * ed)?;
        if cfg.l2_norm {
            codebook.chunks_mut(ed).for_each(normalize);
        }
        let p = cfg.patch;
        let enc = if encoder {
            Some(Encoder {
                patch: dense("encoder.patch_embed", d, 3 * p * p)?,
                stack: stack("encoder")?,
                quant: dense("quant_embed", ed, d)?,
            })
        } else {
            None
        };
        let dec = if decoder {
            let (ffn, head_in) = match cfg.ffn_out {
                Some(f) => (Some(dense("decoder.ffn.0", f, d)?), f),
                None => (None, d),
            };
            Some(Decoder {
                post_quant: dense("post_quant_embed", d, ed)?,
                stack: stack("decoder")?,
                ffn,
                head: dense("decoder.conv_out", 3 * p * p, head_in)?,
            })
        } else {
            None
        };
        let hd = cfg.head_dim();
        let half = hd / 2;
        let inv_freq = (0..hd / 4)
            .map(|j| (1.0 / cfg.theta.powf((2 * j) as f64 / half as f64)) as f32)
            .collect();
        Ok(Self {
            cfg,
            codebook,
            enc,
            dec,
            inv_freq,
        })
    }

    pub fn has_encoder(&self) -> bool {
        self.enc.is_some()
    }

    pub fn has_decoder(&self) -> bool {
        self.dec.is_some()
    }

    /// RoPE tables of a `gh x gw` grid: per token the row angles then the column angles
    /// (`head_dim / 4` each), as cosines and sines.
    fn rope(&self, gh: usize, gw: usize) -> (Vec<f32>, Vec<f32>) {
        let q = self.inv_freq.len();
        let n = gh * gw;
        let (mut cos, mut sin) = (vec![0f32; n * 2 * q], vec![0f32; n * 2 * q]);
        for i in 0..n {
            let (r, c) = ((i / gw) as f32, (i % gw) as f32);
            for (j, &f) in self.inv_freq.iter().enumerate() {
                // torch.outer in float32, then cos / sin of the float32 angle.
                let (a, b) = (r * f, c * f);
                cos[i * 2 * q + j] = a.cos();
                sin[i * 2 * q + j] = a.sin();
                cos[i * 2 * q + q + j] = b.cos();
                sin[i * 2 * q + q + j] = b.sin();
            }
        }
        (cos, sin)
    }

    /// The blocks and the final norm over `x` (`[n, width]`, `n = gh * gw`).
    fn run_stack(&self, s: &Stack, x: &mut [f32], gh: usize, gw: usize) {
        let c = &self.cfg;
        let (d, nh, hd) = (c.width, c.heads, c.head_dim());
        let n = gh * gw;
        let (cos, sin) = self.rope(gh, gw);
        let q4 = hd / 4;
        let mut hn = vec![0f32; n * d];
        let (mut q, mut k, mut v) = (vec![0f32; n * d], vec![0f32; n * d], vec![0f32; n * d]);
        let mut g = vec![0f32; n * c.mlp];
        let mut u = vec![0f32; n * c.mlp];
        let mut y = vec![0f32; n * d];
        for b in &s.blocks {
            hn.copy_from_slice(x);
            rmsnorm_f32w_rows_inplace(&mut hn, d, &b.norm1, EPS);
            linear_multi(
                &mut [(&b.q, d, &mut q[..]), (&b.k, d, &mut k[..]), (&b.v, d, &mut v[..])],
                d,
                &hn,
                n,
            );
            for buf in [&mut q, &mut k] {
                for_chunks(buf, hd, true, |i, head| {
                    let r = i / nh;
                    let (cr, sr) = (&cos[r * 2 * q4..(r + 1) * 2 * q4], &sin[r * 2 * q4..(r + 1) * 2 * q4]);
                    let (h1, h2) = head.split_at_mut(hd / 2);
                    apply_rope(h1, &cr[..q4], &sr[..q4]);
                    apply_rope(h2, &cr[q4..2 * q4], &sr[q4..2 * q4]);
                });
            }
            let att = crate::sd::ops::attention(&q, &k, &v, n, n, nh, hd, false);
            linear(&b.o, d, d, &att, n, &mut y);
            add_inplace(x, &y);
            hn.copy_from_slice(x);
            rmsnorm_f32w_rows_inplace(&mut hn, d, &b.norm2, EPS);
            linear_multi(&mut [(&b.gate, c.mlp, &mut g[..]), (&b.up, c.mlp, &mut u[..])], d, &hn, n);
            silu_mul(&mut g, &u);
            linear(&b.down, d, c.mlp, &g, n, &mut y);
            add_inplace(x, &y);
        }
        rmsnorm_f32w_rows_inplace(x, d, &s.norm, EPS);
    }

    /// Codebook indices of an image (`[3, h, w]` planes in `[-1, 1]`; `h` and `w` multiples of
    /// the patch size), row by row, and the grid `(rows, columns)`.
    pub fn encode(&self, px: &[f32], h: usize, w: usize) -> Result<(Vec<u32>, (usize, usize))> {
        let e = self.enc.as_ref().context("the tokenizer was loaded without its encoder")?;
        let c = &self.cfg;
        let p = c.patch;
        if h == 0 || w == 0 || !h.is_multiple_of(p) || !w.is_multiple_of(p) || px.len() != 3 * h * w {
            bail!("encode: a [3, {h}, {w}] image needs sides that are multiples of {p}");
        }
        let (gh, gw) = (h / p, w / p);
        let n = gh * gw;
        let pd = 3 * p * p;
        let mut patches = vec![0f32; n * pd];
        for_chunks(&mut patches, pd, n > 64, |i, row| {
            let (gy, gx) = (i / gw, i % gw);
            for ch in 0..3 {
                for ky in 0..p {
                    let src = (ch * h + gy * p + ky) * w + gx * p;
                    row[(ch * p + ky) * p..(ch * p + ky + 1) * p].copy_from_slice(&px[src..src + p]);
                }
            }
        });
        let mut x = e.patch.forward(&patches, n);
        self.run_stack(&e.stack, &mut x, gh, gw);
        let mut z = e.quant.forward(&x, n);
        let ed = c.embed_dim;
        if c.l2_norm {
            z.chunks_mut(ed).for_each(normalize);
        }
        Ok((self.nearest(&z), (gh, gw)))
    }

    /// The index of the closest codebook vector to each row of `z` (`[n, embed_dim]`): the
    /// largest dot product with the normalized codebook (cosine, as both are normalized), or the
    /// smallest squared distance without `l2_norm`.
    fn nearest(&self, z: &[f32]) -> Vec<u32> {
        let ed = self.cfg.embed_dim;
        let n = z.len() / ed;
        let cb = &self.codebook;
        let l2 = self.cfg.l2_norm;
        let sq: Vec<f32> = if l2 {
            Vec::new()
        } else {
            cb.chunks(ed).map(|r| r.iter().map(|x| x * x).sum()).collect()
        };
        let mut out = vec![0u32; n];
        let op = OutPtr(out.as_mut_ptr());
        crate::pool::run(n, &|i| {
            let zi = &z[i * ed..(i + 1) * ed];
            let (mut best, mut bi) = (f32::NEG_INFINITY, 0usize);
            for (j, r) in cb.chunks_exact(ed).enumerate() {
                let dot: f32 = zi.iter().zip(r).map(|(a, b)| a * b).sum();
                let s = if l2 { dot } else { 2.0 * dot - sq[j] };
                if s > best {
                    best = s;
                    bi = j;
                }
            }
            // SAFETY: each task writes its own index of `out`.
            unsafe { op.set(i, bi as u32) };
        });
        out
    }

    /// Pixels of a grid of codebook indices (`rows x columns`, row by row): `[3, 16 rows, 16
    /// columns]` planes, clamped to the configured range (`[-1, 1]`).
    pub fn decode(&self, codes: &[u32], grid: (usize, usize)) -> Result<Vec<f32>> {
        let dd = self.dec.as_ref().context("the tokenizer was loaded without its decoder")?;
        let c = &self.cfg;
        let (gh, gw) = grid;
        let n = gh * gw;
        if n == 0 || codes.len() != n {
            bail!("decode: {} codes for a {gh}x{gw} grid", codes.len());
        }
        let ed = c.embed_dim;
        let mut zq = vec![0f32; n * ed];
        for (row, &i) in zq.chunks_mut(ed).zip(codes) {
            let i = i as usize;
            if i >= c.n_embed {
                bail!("decode: code {i} is past the codebook ({})", c.n_embed);
            }
            row.copy_from_slice(&self.codebook[i * ed..(i + 1) * ed]);
        }
        let mut x = dd.post_quant.forward(&zq, n);
        self.run_stack(&dd.stack, &mut x, gh, gw);
        let x = match &dd.ffn {
            Some(f) => {
                let mut y = f.forward(&x, n);
                for_chunks(&mut y, f.n_out, n > 16, |_, r| r.iter_mut().for_each(|v| *v = v.tanh()));
                y
            }
            None => x,
        };
        let out = dd.head.forward(&x, n);
        let p = c.patch;
        let (h, w) = (gh * p, gw * p);
        let (lo, hi) = c.clamp;
        let mut img = vec![0f32; 3 * h * w];
        for i in 0..n {
            let (gy, gx) = (i / gw, i % gw);
            for ch in 0..3 {
                for ky in 0..p {
                    let src = i * 3 * p * p + (ch * p + ky) * p;
                    let dst = (ch * h + gy * p + ky) * w + gx * p;
                    for (o, &v) in img[dst..dst + p].iter_mut().zip(&out[src..src + p]) {
                        *o = v.clamp(lo, hi);
                    }
                }
            }
        }
        Ok(img)
    }
}

struct OutPtr(*mut u32);
// SAFETY: the tasks write disjoint entries.
unsafe impl Send for OutPtr {}
unsafe impl Sync for OutPtr {}

impl OutPtr {
    /// # Safety
    /// `i` is in bounds and no other task writes it.
    unsafe fn set(&self, i: usize, v: u32) {
        unsafe { *self.0.add(i) = v };
    }
}

/// `[3, h, w]` planes in `[-1, 1]` as RGB8, `round((x + 1) / 2 * 255)` as the reference writes
/// its PNG.
pub fn to_rgb8(px: &[f32], h: usize, w: usize) -> Vec<u8> {
    let n = h * w;
    let mut rgb = vec![0u8; 3 * n];
    for i in 0..n {
        for ch in 0..3 {
            let v = ((px[ch * n + i].clamp(-1.0, 1.0) + 1.0) / 2.0 * 255.0).round();
            rgb[3 * i + ch] = v as u8;
        }
    }
    rgb
}

/// RGB8 pixels as `[3, h, w]` planes in `[-1, 1]` (`x / 255 * 2 - 1`).
pub fn from_rgb8(rgb: &[u8], h: usize, w: usize) -> Vec<f32> {
    let n = h * w;
    let mut px = vec![0f32; 3 * n];
    for i in 0..n {
        for ch in 0..3 {
            px[ch * n + i] = rgb[3 * i + ch] as f32 / 255.0 * 2.0 - 1.0;
        }
    }
    px
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen::batch::tests::write_safetensors;

    fn lcg(s: &mut u64) -> f32 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }

    // A tiny random tokenizer: width 32 (2 heads of 16), 2 blocks, 4x4 patches, 64 codes of 8.
    const W: usize = 32;
    const HEADS: usize = 2;
    const LAYERS: usize = 2;
    const P: usize = 4;
    const NE: usize = 64;
    const ED: usize = 8;
    const FFN: usize = 48;

    fn config() -> Value {
        serde_json::json!({
            "model_type": "l24sq_vit", "patch_size": P, "width": W, "layers": LAYERS, "heads": HEADS,
            "mlp_ratio": 4.0, "use_ffn_output": true, "dim_ffn_output": FFN, "n_embed": NE, "embed_dim": ED,
            "l2_norm": true, "vit_rope_theta": 10000.0, "zero_mean": true, "clamp_range": [-1.0, 1.0],
        })
    }

    /// The checkpoint's tensors, rounded to bf16 (as written).
    fn tensors(mlp: usize) -> Vec<(String, Vec<usize>, Vec<f32>)> {
        let mut s = 7u64;
        let mut t = Vec::new();
        let mut add = |name: String, shape: Vec<usize>, scale: f32, s: &mut u64| {
            let n: usize = shape.iter().product();
            let v: Vec<f32> = (0..n).map(|_| half::bf16::from_f32(lcg(s) * scale).to_f32()).collect();
            t.push((name, shape, v));
        };
        for half in ["encoder", "decoder"] {
            for l in 0..LAYERS {
                let b = format!("{half}.blocks.{l}.");
                for n in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                    add(format!("{b}attn.{n}.weight"), vec![W, W], 0.4, &mut s);
                }
                add(format!("{b}mlp.gate_proj.weight"), vec![mlp, W], 0.4, &mut s);
                add(format!("{b}mlp.up_proj.weight"), vec![mlp, W], 0.4, &mut s);
                add(format!("{b}mlp.down_proj.weight"), vec![W, mlp], 0.2, &mut s);
                add(format!("{b}norm1.weight"), vec![W], 2.0, &mut s);
                add(format!("{b}norm2.weight"), vec![W], 2.0, &mut s);
            }
            add(format!("{half}.norm.weight"), vec![W], 2.0, &mut s);
        }
        add("encoder.patch_embed.weight".into(), vec![W, 3, P, P], 0.3, &mut s);
        add("encoder.patch_embed.bias".into(), vec![W], 0.3, &mut s);
        add("decoder.ffn.0.weight".into(), vec![FFN, W], 0.4, &mut s);
        add("decoder.ffn.0.bias".into(), vec![FFN], 0.4, &mut s);
        add("decoder.conv_out.weight".into(), vec![3 * P * P, FFN], 0.4, &mut s);
        add("decoder.conv_out.bias".into(), vec![3 * P * P], 0.2, &mut s);
        add("quant_embed.weight".into(), vec![ED, W], 0.4, &mut s);
        add("quant_embed.bias".into(), vec![ED], 0.1, &mut s);
        add("post_quant_embed.weight".into(), vec![W, ED], 0.8, &mut s);
        add("post_quant_embed.bias".into(), vec![W], 0.1, &mut s);
        add("quantize.embedding.weight".into(), vec![NE, ED], 1.0, &mut s);
        t
    }

    fn get<'t>(t: &'t [(String, Vec<usize>, Vec<f32>)], name: &str) -> &'t [f32] {
        &t.iter().find(|x| x.0 == name).unwrap().2
    }

    /// `y[o] = sum_i w[o][i] x[i] (+ b[o])`.
    fn mat(w: &[f32], x: &[f32], b: Option<&[f32]>) -> Vec<f32> {
        let n_in = x.len();
        w.chunks(n_in)
            .enumerate()
            .map(|(o, r)| r.iter().zip(x).map(|(a, b)| (*a as f64) * (*b as f64)).sum::<f64>() as f32 + b.map_or(0.0, |b| b[o]))
            .collect()
    }

    fn rms(x: &[f32], w: &[f32]) -> Vec<f32> {
        let m = x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / x.len() as f64;
        let k = 1.0 / (m + 1e-6).sqrt();
        x.iter().zip(w).map(|(v, w)| (*v as f64 * k) as f32 * w).collect()
    }

    /// The reference's `apply_2d_rope_packed` for one head of a token at grid (r, c): the
    /// `cos` / `sin` rows `cat(cat(ah, ah), cat(aw, aw))`, rotate-half within each half.
    fn rope(v: &[f32], r: usize, c: usize) -> Vec<f32> {
        let hd = v.len();
        let (half, q) = (hd / 2, hd / 4);
        let ang = |p: usize, j: usize| p as f64 * 10000f64.powf(-((2 * j) as f64) / half as f64);
        let mut cos = vec![0f64; hd];
        let mut sin = vec![0f64; hd];
        for i in 0..hd {
            let (p, j) = if i < half { (r, i % q) } else { (c, (i - half) % q) };
            cos[i] = ang(p, j).cos();
            sin[i] = ang(p, j).sin();
        }
        let mut rot = vec![0f32; hd];
        for h0 in [0, half] {
            for i in 0..q {
                rot[h0 + i] = -v[h0 + q + i];
                rot[h0 + q + i] = v[h0 + i];
            }
        }
        (0..hd).map(|i| (v[i] as f64 * cos[i] + rot[i] as f64 * sin[i]) as f32).collect()
    }

    fn block(t: &[(String, Vec<usize>, Vec<f32>)], b: &str, x: &mut [Vec<f32>], gh: usize, gw: usize) {
        let n = gh * gw;
        let hd = W / HEADS;
        let hn: Vec<Vec<f32>> = x.iter().map(|r| rms(r, get(t, &format!("{b}norm1.weight")))).collect();
        let pr = |name: &str| -> Vec<Vec<f32>> { hn.iter().map(|r| mat(get(t, &format!("{b}attn.{name}.weight")), r, None)).collect() };
        let (q, k, v) = (pr("q_proj"), pr("k_proj"), pr("v_proj"));
        let mut att = vec![vec![0f32; W]; n];
        for h in 0..HEADS {
            let head = |m: &Vec<Vec<f32>>, i: usize| m[i][h * hd..(h + 1) * hd].to_vec();
            let qs: Vec<Vec<f32>> = (0..n).map(|i| rope(&head(&q, i), i / gw, i % gw)).collect();
            let ks: Vec<Vec<f32>> = (0..n).map(|i| rope(&head(&k, i), i / gw, i % gw)).collect();
            for i in 0..n {
                let s: Vec<f64> = (0..n)
                    .map(|j| qs[i].iter().zip(&ks[j]).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() / (hd as f64).sqrt())
                    .collect();
                let m = s.iter().cloned().fold(f64::MIN, f64::max);
                let e: Vec<f64> = s.iter().map(|v| (v - m).exp()).collect();
                let z: f64 = e.iter().sum();
                for d in 0..hd {
                    att[i][h * hd + d] = (0..n).map(|j| e[j] / z * v[j][h * hd + d] as f64).sum::<f64>() as f32;
                }
            }
        }
        for i in 0..n {
            let o = mat(get(t, &format!("{b}attn.out_proj.weight")), &att[i], None);
            x[i].iter_mut().zip(o).for_each(|(a, b)| *a += b);
            let h2 = rms(&x[i], get(t, &format!("{b}norm2.weight")));
            let g = mat(get(t, &format!("{b}mlp.gate_proj.weight")), &h2, None);
            let u = mat(get(t, &format!("{b}mlp.up_proj.weight")), &h2, None);
            let a: Vec<f32> = g.iter().zip(&u).map(|(g, u)| g / (1.0 + (-g).exp()) * u).collect();
            let d = mat(get(t, &format!("{b}mlp.down_proj.weight")), &a, None);
            x[i].iter_mut().zip(d).for_each(|(a, b)| *a += b);
        }
    }

    fn normalize_row(v: &[f32]) -> Vec<f32> {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        v.iter().map(|x| x / n).collect()
    }

    /// The reference's decoder, written out token by token in f64 sums.
    fn ref_decode(t: &[(String, Vec<usize>, Vec<f32>)], codes: &[u32], gh: usize, gw: usize) -> Vec<f32> {
        let cb = get(t, "quantize.embedding.weight");
        let mut x: Vec<Vec<f32>> = codes
            .iter()
            .map(|&c| {
                let z = normalize_row(&cb[c as usize * ED..(c as usize + 1) * ED]);
                mat(get(t, "post_quant_embed.weight"), &z, Some(get(t, "post_quant_embed.bias")))
            })
            .collect();
        for l in 0..LAYERS {
            block(t, &format!("decoder.blocks.{l}."), &mut x, gh, gw);
        }
        let (h, w) = (gh * P, gw * P);
        let mut img = vec![0f32; 3 * h * w];
        for (i, r) in x.iter().enumerate() {
            let r = rms(r, get(t, "decoder.norm.weight"));
            let f: Vec<f32> = mat(get(t, "decoder.ffn.0.weight"), &r, Some(get(t, "decoder.ffn.0.bias")))
                .iter()
                .map(|v| v.tanh())
                .collect();
            let o = mat(get(t, "decoder.conv_out.weight"), &f, Some(get(t, "decoder.conv_out.bias")));
            // "(hh ww) (c sh sw) -> c (hh sh) (ww sw)"
            for ch in 0..3 {
                for sy in 0..P {
                    for sx in 0..P {
                        let v = o[(ch * P + sy) * P + sx].clamp(-1.0, 1.0);
                        img[(ch * h + (i / gw) * P + sy) * w + (i % gw) * P + sx] = v;
                    }
                }
            }
        }
        img
    }

    #[test]
    fn tiny_decoder_and_encoder_match_a_naive_reference() {
        let cfg = VitConfig::from_json(&config()).unwrap();
        assert_eq!(cfg.mlp, 128); // int(32 * 4 * 2 / 3) = 85, rounded up to 64s
        let t = tensors(cfg.mlp);
        let dir = std::env::temp_dir().join(format!("oxinfer-tiny-l24sq-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        write_safetensors(&dir.join("model.safetensors"), &t);
        let files = crate::safetensors::ModelFiles::open(&dir, true).unwrap();
        let vit = Vit::load(cfg, &files, Precision::Bf16, true, true).unwrap();
        // A non-square grid (3 rows of 5): the row / column halves of the RoPE differ.
        let (gh, gw) = (3, 5);
        let mut s = 11u64;
        let codes: Vec<u32> = (0..gh * gw).map(|_| ((lcg(&mut s) + 0.5) * NE as f32) as u32 % NE as u32).collect();
        let got = vit.decode(&codes, (gh, gw)).unwrap();
        let want = ref_decode(&t, &codes, gh, gw);
        let err = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        // (bf16 activations in the CPU's bf16 matmuls.)
        assert!(err < 3e-2, "decoder: max difference {err}");
        // The encoder: patches through the blocks, then the codebook's nearest vector by cosine.
        let px = got.clone();
        let (enc, grid) = vit.encode(&px, gh * P, gw * P).unwrap();
        assert_eq!(grid, (gh, gw));
        let mut x: Vec<Vec<f32>> = (0..gh * gw)
            .map(|i| {
                let mut patch = vec![0f32; 3 * P * P];
                for ch in 0..3 {
                    for sy in 0..P {
                        for sx in 0..P {
                            patch[(ch * P + sy) * P + sx] = px[(ch * gh * P + (i / gw) * P + sy) * gw * P + (i % gw) * P + sx];
                        }
                    }
                }
                mat(
                    get(&t, "encoder.patch_embed.weight"),
                    &patch,
                    Some(get(&t, "encoder.patch_embed.bias")),
                )
            })
            .collect();
        for l in 0..LAYERS {
            block(&t, &format!("encoder.blocks.{l}."), &mut x, gh, gw);
        }
        let cb: Vec<Vec<f32>> = get(&t, "quantize.embedding.weight").chunks(ED).map(normalize_row).collect();
        let mut agree = 0;
        for (i, r) in x.iter().enumerate() {
            let z = normalize_row(&mat(
                get(&t, "quant_embed.weight"),
                &rms(r, get(&t, "encoder.norm.weight")),
                Some(get(&t, "quant_embed.bias")),
            ));
            let best = (0..NE)
                .max_by(|&a, &b| {
                    let d = |j: usize| cb[j].iter().zip(&z).map(|(x, y)| x * y).sum::<f32>();
                    d(a).total_cmp(&d(b)).then(b.cmp(&a))
                })
                .unwrap();
            agree += (best as u32 == enc[i]) as usize;
        }
        // (A near tie can go either way after bf16 rounding.)
        assert!(agree + 1 >= enc.len(), "encoder: {agree}/{} codes agree", enc.len());
        // Pixels round trip through RGB8.
        let rgb = to_rgb8(&got, gh * P, gw * P);
        let back = from_rgb8(&rgb, gh * P, gw * P);
        assert!(back.iter().zip(&got).all(|(a, b)| (a - b).abs() <= 1.0 / 255.0 + 1e-6));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
