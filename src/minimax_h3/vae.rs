//! MiniMax-H3-Image's VAE decoder (diffusers' `AutoencoderKLMiniMaxH3Image.decode`): a 1x1
//! `post_quant_conv`, then a ViT over the latent pixels: `proj_in`, four learned register tokens and
//! one all-zero token appended (at position 0), blocks of RMSNorm + attention (affine-free per-head
//! RMSNorm on queries and keys; 2D RoPE on the first half of each head: coordinates in `[-1, 1)`
//! scaled by `2 pi`, `theta` 100) and RMSNorm + SwiGLU, each output scaled by a learned vector,
//! then LayerNorm and `proj_out` to a 16x16 RGB patch per latent pixel.
//!
//! Decoding is tiled as released: 256-pixel tiles overlapping by at least 64 (the slack spread in
//! whole 16-pixel steps), each decoded on its own and blended linearly with the *undecoded order's*
//! neighbours (`_stitch_tiles` blends a tile with the original tile above and the original tile to
//! its left).
//!
//! The release's layout (`decoder.x_embedder`, `attn.to_qkv` interleaved per head, `ff.w1` as
//! `[gate; value]`) loads as stored, as does diffusers'.

use super::dit::DLin;
use crate::kernels::*;
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};

pub struct VaeConfig {
    pub latent_channels: usize,
    pub dim: usize,
    pub heads: usize,
    pub layers: usize,
    pub registers: usize,
    pub patch: usize,
    pub rope_theta: f64,
    pub rope_ratio: f64,
    pub eps: f32,
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
    pub tile: usize,
    pub overlap: usize,
}

impl VaeConfig {
    /// `vae/config.json` (the release's `MiniMaxH3ImageVAE` keys or diffusers').
    pub fn load(dir: &std::path::Path) -> Result<Self> {
        let c: serde_json::Value = serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?)?;
        let k = c.get("vit_decoder_kwargs");
        let u = |a: &str, b: &str, d: usize| {
            c.get(a)
                .or_else(|| k.and_then(|k| k.get(b)))
                .and_then(|v| v.as_u64())
                .map_or(d, |v| v as usize)
        };
        let f = |a: &str, b: &str, d: f64| c.get(a).or_else(|| k.and_then(|k| k.get(b))).and_then(|v| v.as_f64()).unwrap_or(d);
        let vec = |n: &str| -> Result<Vec<f32>> {
            Ok(c.get(n)
                .and_then(|v| v.as_array())
                .with_context(|| format!("VAE config has no {n}"))?
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect())
        };
        let heads = u("decoder_num_attention_heads", "heads", 16);
        let hd = u("decoder_attention_head_dim", "dim_head", 64);
        Ok(Self {
            latent_channels: u("latent_channels", "latent_channels", 24),
            dim: heads * hd,
            heads,
            layers: u("decoder_num_layers", "num_layers", 24),
            registers: u("decoder_num_register_tokens", "num_register_tokens", 4),
            patch: 16,
            rope_theta: f("decoder_rope_theta", "rope_theta", 100.0),
            rope_ratio: f("decoder_rope_dim_ratio", "rope_dim_ratio", 0.5),
            eps: f("decoder_norm_eps", "norm_eps", 1e-5) as f32,
            latents_mean: vec("latents_mean")?,
            latents_std: vec("latents_std")?,
            tile: c.get("vae_tile_size").and_then(|v| v.as_u64()).map_or(256, |v| v as usize),
            overlap: c.get("vae_tile_overlap_min").and_then(|v| v.as_u64()).map_or(64, |v| v as usize),
        })
    }
}

enum Qkv<'a> {
    Split([DLin<'a>; 3]),
    /// `[q k v]` per head.
    Interleaved(DLin<'a>),
}

struct Block<'a> {
    norm1: Vec<f32>,
    qkv: Qkv<'a>,
    out: DLin<'a>,
    scale1: Vec<f32>,
    norm2: Vec<f32>,
    fc1: DLin<'a>,
    gate_first: bool,
    fc2: DLin<'a>,
    scale2: Vec<f32>,
}

pub struct VaeDecoder<'a> {
    pub cfg: VaeConfig,
    post_quant: DLin<'a>,
    proj_in: DLin<'a>,
    registers: Vec<f32>,
    blocks: Vec<Block<'a>>,
    norm_out: (Vec<f32>, Vec<f32>),
    proj_out: DLin<'a>,
    inv_freq: Vec<f32>,
}

impl<'a> VaeDecoder<'a> {
    pub fn load(mut cfg: VaeConfig, files: &'a ModelFiles) -> Result<Self> {
        let original = files.has("decoder.x_embedder.weight");
        // The release's config does not state the register count: the tensor does.
        if let Some(i) = files.info("decoder.register_tokens") {
            cfg.registers = i.shape.iter().product::<usize>() / cfg.dim;
        }
        let (d, c) = (cfg.dim, cfg.latent_channels);
        let hd = d / cfg.heads;
        let ffn = 4 * d;
        let v = |n: &str, len: usize| -> Result<Vec<f32>> {
            let x = files.f32(n)?;
            if x.len() != len {
                bail!("{n}: {} values, expected {len}", x.len());
            }
            Ok(x)
        };
        let mut blocks = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            let p = format!("decoder.transformer_blocks.{l}.");
            let (qkv, out, fc1, gate_first, fc2) = if original {
                (
                    Qkv::Interleaved(DLin::load(files, &format!("{p}attn.to_qkv"), 3 * d, d, true)?),
                    DLin::load(files, &format!("{p}attn.to_out"), d, d, true)?,
                    DLin::load(files, &format!("{p}ff.w1"), 2 * ffn, d, true)?,
                    true,
                    DLin::load(files, &format!("{p}ff.w2"), d, ffn, true)?,
                )
            } else {
                (
                    Qkv::Split([
                        DLin::load(files, &format!("{p}attn.to_q"), d, d, true)?,
                        DLin::load(files, &format!("{p}attn.to_k"), d, d, true)?,
                        DLin::load(files, &format!("{p}attn.to_v"), d, d, true)?,
                    ]),
                    DLin::load(files, &format!("{p}attn.to_out.0"), d, d, true)?,
                    DLin::load(files, &format!("{p}ff.net.0.proj"), 2 * ffn, d, true)?,
                    false,
                    DLin::load(files, &format!("{p}ff.net.2"), d, ffn, true)?,
                )
            };
            blocks.push(Block {
                norm1: v(&format!("{p}norm1.weight"), d)?,
                qkv,
                out,
                scale1: v(&format!("{p}scale1"), d)?,
                norm2: v(&format!("{p}norm2.weight"), d)?,
                fc1,
                gate_first,
                fc2,
                scale2: v(&format!("{p}scale2"), d)?,
            });
        }
        let rd = (hd as f64 * cfg.rope_ratio) as usize;
        // inv_freq = 1 / theta ** arange(0, 1, 2 * axes / dim): rd / 4 frequencies per axis.
        let step = 4.0 / rd as f64;
        let inv_freq = (0..rd / 4).map(|i| (1.0 / cfg.rope_theta.powf(i as f64 * step)) as f32).collect();
        let post_quant = {
            let w = files.f32("post_quant_conv.weight")?;
            if w.len() != c * c {
                bail!("post_quant_conv: unexpected size");
            }
            DLin::from_f32(&w, Some(v("post_quant_conv.bias", c)?), c, c)?
        };
        let pin = if original { "decoder.x_embedder" } else { "decoder.proj_in" };
        Ok(Self {
            post_quant,
            proj_in: DLin::load(files, pin, d, c, true)?,
            registers: v("decoder.register_tokens", cfg.registers * d)?,
            blocks,
            norm_out: (v("decoder.norm_out.weight", d)?, v("decoder.norm_out.bias", d)?),
            proj_out: DLin::load(files, "decoder.proj_out", 3 * cfg.patch * cfg.patch, d, true)?,
            inv_freq,
            cfg,
        })
    }

    /// Decodes one `[C, h, w]` latent tile into `[3, 16h, 16w]` (no tiling).
    fn decode_tile(&self, z: &[f32], h: usize, w: usize) -> Vec<f32> {
        let cfg = &self.cfg;
        let (c, d, nh) = (cfg.latent_channels, cfg.dim, cfg.heads);
        let hd = d / nh;
        let np = h * w;
        // Channels-last pixels through the 1x1 conv.
        let mut px = vec![0f32; np * c];
        for (i, row) in px.chunks_mut(c).enumerate() {
            for ch in 0..c {
                row[ch] = z[ch * np + i];
            }
        }
        let px = self.post_quant.forward(&px, np);
        let n = np + cfg.registers + 1;
        let mut x = vec![0f32; n * d];
        x[..np * d].copy_from_slice(&self.proj_in.forward(&px, np));
        x[np * d..(np + cfg.registers) * d].copy_from_slice(&self.registers);
        // RoPE: (h, w) in [-1, 1), the extra tokens at 0; angles [h | w] duplicated.
        let f = self.inv_freq.len();
        let half = 2 * f;
        let (mut cos, mut sin) = (vec![1f32; n * half], vec![0f32; n * half]);
        let two_pi = 2.0 * std::f32::consts::PI;
        for i in 0..np {
            let (r, col) = (i / w, i % w);
            let pos = [
                2.0 * ((r as f32 + 0.5) / h as f32) - 1.0,
                2.0 * ((col as f32 + 0.5) / w as f32) - 1.0,
            ];
            for (axis, &p) in pos.iter().enumerate() {
                for (j, &fr) in self.inv_freq.iter().enumerate() {
                    let a = two_pi * p * fr;
                    cos[i * half + axis * f + j] = a.cos();
                    sin[i * half + axis * f + j] = a.sin();
                }
            }
        }
        let scale = (hd as f32).powf(-0.5);
        for b in &self.blocks {
            let mut hn = x.clone();
            rmsnorm_f32w_rows_inplace(&mut hn, d, &b.norm1, cfg.eps);
            let (mut q, mut k, v) = match &b.qkv {
                Qkv::Split([a, bk, cv]) => (a.forward(&hn, n), bk.forward(&hn, n), cv.forward(&hn, n)),
                Qkv::Interleaved(wq) => {
                    let fused = wq.forward(&hn, n);
                    let (mut q, mut k, mut v) = (vec![0f32; n * d], vec![0f32; n * d], vec![0f32; n * d]);
                    for r in 0..n {
                        for hh in 0..nh {
                            let s = (r * nh + hh) * 3 * hd;
                            let o = r * d + hh * hd;
                            q[o..o + hd].copy_from_slice(&fused[s..s + hd]);
                            k[o..o + hd].copy_from_slice(&fused[s + hd..s + 2 * hd]);
                            v[o..o + hd].copy_from_slice(&fused[s + 2 * hd..s + 3 * hd]);
                        }
                    }
                    (q, k, v)
                }
            };
            for buf in [&mut q, &mut k] {
                for_chunks(buf, hd, true, |i, head| {
                    rmsnorm_inplace(head, None, cfg.eps);
                    let r = i / nh;
                    apply_rope(
                        &mut head[..2 * half],
                        &cos[r * half..(r + 1) * half],
                        &sin[r * half..(r + 1) * half],
                    );
                });
            }
            q.iter_mut().for_each(|v| *v *= scale);
            let mut att = vec![0f32; n * d];
            crate::vision::attention_bidirectional(&q, &k, &v, n, nh, hd, &mut att);
            let y = b.out.forward(&att, n);
            for_chunks(&mut x, d, true, |r, row| {
                for ((xv, &yv), &s) in row.iter_mut().zip(&y[r * d..(r + 1) * d]).zip(&b.scale1) {
                    *xv += yv * s;
                }
            });
            let mut hn = x.clone();
            rmsnorm_f32w_rows_inplace(&mut hn, d, &b.norm2, cfg.eps);
            let gu = b.fc1.forward(&hn, n);
            let ffn = 4 * d;
            let mut act = vec![0f32; n * ffn];
            for_chunks(&mut act, ffn, true, |r, row| {
                let src = &gu[r * 2 * ffn..(r + 1) * 2 * ffn];
                let (g, u) = if b.gate_first {
                    (&src[..ffn], &src[ffn..])
                } else {
                    (&src[ffn..], &src[..ffn])
                };
                for ((o, &gv), &uv) in row.iter_mut().zip(g).zip(u) {
                    *o = uv * gv / (1.0 + (-gv).exp());
                }
            });
            let y = b.fc2.forward(&act, n);
            for_chunks(&mut x, d, true, |r, row| {
                for ((xv, &yv), &s) in row.iter_mut().zip(&y[r * d..(r + 1) * d]).zip(&b.scale2) {
                    *xv += yv * s;
                }
            });
        }
        // LayerNorm, the patch head over the latent pixels, then [3, 16h, 16w].
        let (nw, nb) = &self.norm_out;
        for_chunks(&mut x[..np * d], d, true, |_, r| {
            let mean = r.iter().sum::<f32>() / d as f32;
            let var = r.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
            let inv = (var + cfg.eps).sqrt().recip();
            for ((v, w), b) in r.iter_mut().zip(nw).zip(nb) {
                *v = (*v - mean) * inv * w + b;
            }
        });
        let p = cfg.patch;
        let out = self.proj_out.forward(&x[..np * d], np);
        let (oh, ow) = (h * p, w * p);
        let mut img = vec![0f32; 3 * oh * ow];
        for i in 0..np {
            let (r, col) = (i / w, i % w);
            for ch in 0..3 {
                for y in 0..p {
                    for xx in 0..p {
                        img[(ch * oh + r * p + y) * ow + col * p + xx] = out[i * 3 * p * p + (ch * p + y) * p + xx];
                    }
                }
            }
        }
        img
    }

    /// `_split_tiles`: tile starts and overlaps over `length` pixels.
    fn split(&self, length: usize) -> (Vec<usize>, usize, Vec<usize>) {
        let (tile, min_ov, ratio) = (self.cfg.tile, self.cfg.overlap, self.cfg.patch);
        if tile >= length {
            return (vec![0], length, Vec::new());
        }
        let mut n = length.div_ceil(tile);
        while (tile * n) < min_ov * (n - 1) + length {
            n += 1;
        }
        let mut ov = vec![min_ov; n - 1];
        let remaining = tile * n - ov.iter().sum::<usize>() - length;
        for i in 0..remaining / ratio {
            ov[i % (n - 1)] += ratio;
        }
        let mut starts = vec![0];
        for i in 0..n - 1 {
            let s = starts[i] + tile - ov[i];
            starts.push(s);
        }
        (starts, tile, ov)
    }

    /// Decodes normalized latents `[C, h, w]` (denormalized here) into RGB `[3, 16h, 16w]` in the
    /// VAE's output range (ImageNet-normalized), tiled as released.
    pub fn decode(&self, latents: &[f32], h: usize, w: usize) -> Vec<f32> {
        let cfg = &self.cfg;
        let c = cfg.latent_channels;
        let np = h * w;
        let z: Vec<f32> = latents
            .iter()
            .enumerate()
            .map(|(i, &v)| v * cfg.latents_std[i / np] + cfg.latents_mean[i / np])
            .collect();
        let p = cfg.patch;
        let (ys, th, yov) = self.split(h * p);
        let (xs, tw, xov) = self.split(w * p);
        // Every tile decoded on its own.
        let tiles: Vec<Vec<Vec<f32>>> = ys
            .iter()
            .map(|&y0| {
                xs.iter()
                    .map(|&x0| {
                        let (lh, lw) = (th / p, tw / p);
                        let (ly, lx) = (y0 / p, x0 / p);
                        let mut t = vec![0f32; c * lh * lw];
                        for ch in 0..c {
                            for r in 0..lh {
                                t[(ch * lh + r) * lw..(ch * lh + r + 1) * lw]
                                    .copy_from_slice(&z[(ch * h + ly + r) * w + lx..(ch * h + ly + r) * w + lx + lw]);
                            }
                        }
                        self.decode_tile(&t, lh, lw)
                    })
                    .collect()
            })
            .collect();
        stitch(&tiles, th, tw, &yov, &xov, h * p, w * p)
    }
}

/// `_stitch_tiles`: each tile blended with the original tile above and the original tile to its
/// left over the overlap, trimmed of its trailing overlaps, then laid out.
fn stitch(tiles: &[Vec<Vec<f32>>], th: usize, tw: usize, yov: &[usize], xov: &[usize], oh: usize, ow: usize) -> Vec<f32> {
    let mut out = vec![0f32; 3 * oh * ow];
    let mut oy = 0;
    for (i, row) in tiles.iter().enumerate() {
        let keep_h = if i + 1 < tiles.len() { th - yov[i] } else { th };
        let mut ox = 0;
        for (j, tile) in row.iter().enumerate() {
            let mut t = tile.clone();
            if i > 0 {
                let e = yov[i - 1].min(th);
                let above = &tiles[i - 1][j];
                for ch in 0..3 {
                    for y in 0..e {
                        let wb = y as f32 / e as f32;
                        for x in 0..tw {
                            let a = above[(ch * th + th - e + y) * tw + x];
                            let b = &mut t[(ch * th + y) * tw + x];
                            *b = a * (1.0 - wb) + *b * wb;
                        }
                    }
                }
            }
            if j > 0 {
                let e = xov[j - 1].min(tw);
                let left = &row[j - 1];
                for ch in 0..3 {
                    for y in 0..th {
                        for x in 0..e {
                            let wb = x as f32 / e as f32;
                            let a = left[(ch * th + y) * tw + tw - e + x];
                            let b = &mut t[(ch * th + y) * tw + x];
                            *b = a * (1.0 - wb) + *b * wb;
                        }
                    }
                }
            }
            let keep_w = if j + 1 < row.len() { tw - xov[j] } else { tw };
            for ch in 0..3 {
                for y in 0..keep_h.min(oh - oy) {
                    for x in 0..keep_w.min(ow - ox) {
                        out[(ch * oh + oy + y) * ow + ox + x] = t[(ch * th + y) * tw + x];
                    }
                }
            }
            ox += keep_w;
        }
        oy += keep_h;
    }
    out
}
