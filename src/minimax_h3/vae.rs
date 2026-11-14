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
    /// Rotary axes: 2 (`(h, w)`, the image decoder) or 3 (`(t, h, w)`, the video decoder).
    pub axes: usize,
    /// Pixel frames per latent frame (1 for the image decoder).
    pub patch_t: usize,
    /// The video VAE's temporal chunking: pixel frames per encoded clip and the latent frames
    /// dropped at the end of an encode.
    pub clip_length: usize,
    pub token_drop: usize,
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
        // The video VAE (`AutoencoderKLMiniMaxH3`) states its temporal geometry.
        let video = c.get("clip_length").is_some() || c.get("temporal_downsample_factors").is_some();
        let patch_t = c
            .get("temporal_downsample_factors")
            .and_then(|v| v.as_array())
            .map_or(if video { 4 } else { 1 }, |a| {
                a.iter().filter_map(|v| v.as_u64()).product::<u64>() as usize
            });
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
            axes: if video { 3 } else { 2 },
            patch_t: if video { patch_t } else { 1 },
            clip_length: u("clip_length", "clip_length", 17),
            token_drop: u("token_drop", "token_drop", 3),
        })
    }
}

enum Qkv<'a> {
    Split(Box<[DLin<'a>; 3]>),
    /// `[q k v]` per head.
    Interleaved(Box<DLin<'a>>),
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
                    Qkv::Interleaved(Box::new(DLin::load(files, &format!("{p}attn.to_qkv"), 3 * d, d, true)?)),
                    DLin::load(files, &format!("{p}attn.to_out"), d, d, true)?,
                    DLin::load(files, &format!("{p}ff.w1"), 2 * ffn, d, true)?,
                    true,
                    DLin::load(files, &format!("{p}ff.w2"), d, ffn, true)?,
                )
            } else {
                (
                    Qkv::Split(Box::new([
                        DLin::load(files, &format!("{p}attn.to_q"), d, d, true)?,
                        DLin::load(files, &format!("{p}attn.to_k"), d, d, true)?,
                        DLin::load(files, &format!("{p}attn.to_v"), d, d, true)?,
                    ])),
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
        // inv_freq = 1 / theta ** arange(0, 1, 2 * axes / dim): rd / (2 * axes) frequencies per axis.
        let step = (2 * cfg.axes) as f64 / rd as f64;
        let inv_freq = (0..rd / (2 * cfg.axes))
            .map(|i| (1.0 / cfg.rope_theta.powf(i as f64 * step)) as f32)
            .collect();
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
            proj_out: DLin::load(files, "decoder.proj_out", 3 * cfg.patch_t * cfg.patch * cfg.patch, d, true)?,
            inv_freq,
            cfg,
        })
    }

    /// Decodes one `[C, h, w]` latent tile into `[3, 16h, 16w]` (no tiling).
    fn decode_tile(&self, z: &[f32], h: usize, w: usize) -> Vec<f32> {
        self.decode_block(z, 1, h, w)
    }

    /// Decodes one `[C, t, h, w]` latent block into `[3, t * patch_t, 16h, 16w]` (no tiling): every
    /// latent voxel is one token.
    fn decode_block(&self, z: &[f32], t: usize, h: usize, w: usize) -> Vec<f32> {
        let cfg = &self.cfg;
        let (c, d, nh) = (cfg.latent_channels, cfg.dim, cfg.heads);
        let hd = d / nh;
        let np = t * h * w;
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
        // RoPE: (h, w) or (t, h, w) in [-1, 1), the extra tokens at 0; angles [t | h | w] duplicated.
        let f = self.inv_freq.len();
        let half = cfg.axes * f;
        let centre = |i: usize, size: usize| 2.0 * ((i as f32 + 0.5) / size as f32) - 1.0;
        let (mut cos, mut sin) = (vec![1f32; n * half], vec![0f32; n * half]);
        let two_pi = 2.0 * std::f32::consts::PI;
        for i in 0..np {
            let (fr, r, col) = (i / (h * w), i / w % h, i % w);
            let pos3 = [centre(fr, t), centre(r, h), centre(col, w)];
            for (axis, &p) in pos3[3 - cfg.axes..].iter().enumerate() {
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
                Qkv::Split(w) => (w[0].forward(&hn, n), w[1].forward(&hn, n), w[2].forward(&hn, n)),
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
        // LayerNorm, the patch head over the latent voxels, then [3, t * pt, 16h, 16w].
        let (nw, nb) = &self.norm_out;
        for_chunks(&mut x[..np * d], d, true, |_, r| {
            let mean = r.iter().sum::<f32>() / d as f32;
            let var = r.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
            let inv = (var + cfg.eps).sqrt().recip();
            for ((v, w), b) in r.iter_mut().zip(nw).zip(nb) {
                *v = (*v - mean) * inv * w + b;
            }
        });
        let (p, pt) = (cfg.patch, cfg.patch_t);
        let out = self.proj_out.forward(&x[..np * d], np);
        let (of, oh, ow) = (t * pt, h * p, w * p);
        let row = 3 * pt * p * p;
        let mut img = vec![0f32; 3 * of * oh * ow];
        for i in 0..np {
            let (fr, r, col) = (i / (h * w), i / w % h, i % w);
            for ch in 0..3 {
                for tt in 0..pt {
                    for y in 0..p {
                        let src = i * row + ((ch * pt + tt) * p + y) * p;
                        let dst = ((ch * of + fr * pt + tt) * oh + r * p + y) * ow + col * p;
                        img[dst..dst + p].copy_from_slice(&out[src..src + p]);
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
        stitch(&tiles, 3, th, tw, &yov, &xov, h * p, w * p)
    }

    /// `_decode_clip`: one temporal clip `[C, t, h, w]` (denormalized) into `[3, t * patch_t, 16h,
    /// 16w]`, spatially tiled like [`VaeDecoder::decode`].
    fn decode_clip(&self, z: &[f32], t: usize, h: usize, w: usize) -> Vec<f32> {
        let cfg = &self.cfg;
        let (c, p) = (cfg.latent_channels, cfg.patch);
        let (ys, th, yov) = self.split(h * p);
        let (xs, tw, xov) = self.split(w * p);
        let (lh, lw) = (th / p, tw / p);
        let of = t * cfg.patch_t;
        let tiles: Vec<Vec<Vec<f32>>> = ys
            .iter()
            .map(|&y0| {
                xs.iter()
                    .map(|&x0| {
                        let (ly, lx) = (y0 / p, x0 / p);
                        let mut tile = vec![0f32; c * t * lh * lw];
                        for cf in 0..c * t {
                            for r in 0..lh {
                                let src = (cf * h + ly + r) * w + lx;
                                tile[(cf * lh + r) * lw..(cf * lh + r + 1) * lw].copy_from_slice(&z[src..src + lw]);
                            }
                        }
                        self.decode_block(&tile, t, lh, lw)
                    })
                    .collect()
            })
            .collect();
        // `[3, F, th, tw]` tiles stitch as `3 * F` planes.
        stitch(&tiles, 3 * of, th, tw, &yov, &xov, h * p, w * p)
    }

    /// The temporal chunking (`AutoencoderKLMiniMaxH3`): latent frames per clip, the latent frames a
    /// decoded clip overlaps the next one by, the pixel frames cut at the start of a clip and the
    /// pixel frames two clips cross-fade over.
    pub fn chunking(&self) -> (usize, usize, usize, usize) {
        let (clip, drop, r) = (self.cfg.clip_length, self.cfg.token_drop, self.cfg.patch_t);
        let pre = (r - clip % r) % r;
        let tcs = clip.div_ceil(r);
        let token_overlap = (tcs - drop % tcs) % tcs;
        (tcs, token_overlap, pre, (token_overlap * r).saturating_sub(pre))
    }

    /// Decodes normalized video latents `[C, t, h, w]` (denormalized here) into `[3, F, 16h, 16w]`
    /// in the VAE's output range (ImageNet-normalized) and `F` (`AutoencoderKLMiniMaxH3._decode`):
    /// clips of `tokens_chunk_size + token_overlap` latent frames, each decoded on its own
    /// (spatially tiled), consecutive clips cross-faded over `frame_overlap` pixel frames.
    pub fn decode_video(&self, latents: &[f32], t: usize, h: usize, w: usize) -> Result<(Vec<f32>, usize)> {
        let cfg = &self.cfg;
        let (c, r, drop) = (cfg.latent_channels, cfg.patch_t, cfg.token_drop);
        let (tcs, token_overlap, pre, frame_overlap) = self.chunking();
        if cfg.axes != 3 || latents.len() != c * t * h * w {
            bail!("video decode: {} latent values for [{c}, {t}, {h}, {w}]", latents.len());
        }
        let vol = t * h * w;
        let plane = h * w;
        let pad = (tcs - (t + drop) % tcs) % tcs;
        let chunks = ((t + drop + pad) / tcs).saturating_sub(usize::from(drop > 0));
        if chunks == 0 {
            bail!(
                "video decode: {t} latent frames are fewer than one clip ({} needed)",
                2 * tcs - drop
            );
        }
        // Denormalized, the last frame repeated `pad` times: frame-major per channel.
        let tp = t + pad;
        let mut z = vec![0f32; c * tp * plane];
        for ch in 0..c {
            for f in 0..tp {
                let src = ch * vol + f.min(t - 1) * plane;
                let dst = (ch * tp + f) * plane;
                for i in 0..plane {
                    z[dst + i] = latents[src + i] * cfg.latents_std[ch] + cfg.latents_mean[ch];
                }
            }
        }
        let (oh, ow) = (h * cfg.patch, w * cfg.patch);
        let fplane = oh * ow;
        // Frames as `[3, oh, ow]` each.
        let mut frames: Vec<Vec<f32>> = Vec::new();
        let mut overlap: Vec<Vec<f32>> = Vec::new();
        let cnf = tcs * r;
        for i in 0..chunks {
            let (t0, tl) = (i * tcs, (tcs + token_overlap).min(tp - i * tcs));
            let mut clip = vec![0f32; c * tl * plane];
            for ch in 0..c {
                clip[ch * tl * plane..(ch + 1) * tl * plane].copy_from_slice(&z[(ch * tp + t0) * plane..(ch * tp + t0 + tl) * plane]);
            }
            let dec = self.decode_clip(&clip, tl, h, w);
            let nf = tl * r;
            let frame = |f: usize| -> Vec<f32> {
                let mut o = vec![0f32; 3 * fplane];
                for ch in 0..3 {
                    o[ch * fplane..(ch + 1) * fplane].copy_from_slice(&dec[(ch * nf + f) * fplane..(ch * nf + f + 1) * fplane]);
                }
                o
            };
            for j in 0..1 + usize::from(drop > 0) {
                let mut chunk: Vec<Vec<f32>> = ((j * cnf + pre).min(nf)..((j + 1) * cnf).min(nf)).map(&frame).collect();
                if j == 0 {
                    let e = overlap.len().min(chunk.len()).min(frame_overlap);
                    for (k, b) in chunk.iter_mut().take(e).enumerate() {
                        let wb = k as f32 / e as f32;
                        let a = &overlap[overlap.len() - e + k];
                        for (bv, &av) in b.iter_mut().zip(a) {
                            *bv = av * (1.0 - wb) + *bv * wb;
                        }
                    }
                    frames.append(&mut chunk);
                } else {
                    overlap = chunk;
                }
            }
        }
        frames.append(&mut overlap);
        // The repeated latent frames decoded into frames nobody asked for.
        let tail = cfg.clip_length % r;
        let cut: usize = (0..pad)
            .map(|k| if tail != 0 && (t + k).is_multiple_of(tcs) { tail } else { r })
            .sum();
        frames.truncate(frames.len().saturating_sub(cut));
        let nf = frames.len();
        let mut out = vec![0f32; 3 * nf * fplane];
        for (f, fr) in frames.iter().enumerate() {
            for ch in 0..3 {
                out[(ch * nf + f) * fplane..(ch * nf + f + 1) * fplane].copy_from_slice(&fr[ch * fplane..(ch + 1) * fplane]);
            }
        }
        Ok((out, nf))
    }
}

/// `_stitch_tiles` over `planes` planes per tile: each tile blended with the original tile above and the original tile to its
/// left over the overlap, trimmed of its trailing overlaps, then laid out.
#[allow(clippy::too_many_arguments)]
pub(crate) fn stitch(
    tiles: &[Vec<Vec<f32>>],
    planes: usize,
    th: usize,
    tw: usize,
    yov: &[usize],
    xov: &[usize],
    oh: usize,
    ow: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; planes * oh * ow];
    let mut oy = 0;
    for (i, row) in tiles.iter().enumerate() {
        let keep_h = if i + 1 < tiles.len() { th - yov[i] } else { th };
        let mut ox = 0;
        for (j, tile) in row.iter().enumerate() {
            let mut t = tile.clone();
            if i > 0 {
                let e = yov[i - 1].min(th);
                let above = &tiles[i - 1][j];
                for ch in 0..planes {
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
                for ch in 0..planes {
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
            for ch in 0..planes {
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
