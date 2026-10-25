//! Gemma 4 vision: image preprocessing (aspect-preserving bicubic resize into 16x16 patches),
//! the ViT encoder with axial 2D RoPE and bidirectional attention, 3x3 patch pooling, and the
//! projection of soft tokens into the language model's embedding space.

use crate::kernels::*;
use crate::mm_config::VisionConfig;
use crate::pool;
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};

pub const SUPPORTED_SOFT_TOKENS: [usize; 5] = [70, 140, 280, 560, 1120];

/// A linear layer with optional input/output clamping (`use_clipped_linears`).
pub struct ClipLin<'a> {
    pub w: &'a [u16],
    pub n_out: usize,
    pub n_in: usize,
    pub bounds: Option<(f32, f32, f32, f32)>, // in_min, in_max, out_min, out_max
}

impl ClipLin<'_> {
    pub fn apply(&self, x: &[f32], t: usize, y: &mut [f32]) {
        debug_assert_eq!(x.len(), t * self.n_in);
        match self.bounds {
            Some((imin, imax, omin, omax)) => {
                let xc: Vec<f32> = x.iter().map(|v| v.clamp(imin, imax)).collect();
                matmul(self.w, self.n_out, self.n_in, &xc, t, y);
                for v in y.iter_mut() {
                    *v = v.clamp(omin, omax);
                }
            }
            None => matmul(self.w, self.n_out, self.n_in, x, t, y),
        }
    }
}

/// Loads `prefix.weight` and, when `clipped`, the four `prefix.{input,output}_{min,max}` scalars.
pub fn load_clip_lin<'a>(files: &'a ModelFiles, prefix: &str, n_out: usize, n_in: usize, clipped: bool) -> Result<ClipLin<'a>> {
    let wname = format!("{prefix}.linear.weight");
    let info = files.info(&wname).with_context(|| format!("missing tensor {wname}"))?;
    if info.shape != [n_out, n_in] {
        bail!("tensor {wname} has shape {:?}, expected {:?}", info.shape, [n_out, n_in]);
    }
    let w = files.bf16(&wname)?;
    let bounds = if clipped {
        let s = |n: &str| -> Result<f32> {
            let t = files.bf16(&format!("{prefix}.{n}"))?;
            Ok(bf16_to_f32(t[0]))
        };
        Some((s("input_min")?, s("input_max")?, s("output_min")?, s("output_max")?))
    } else {
        None
    };
    Ok(ClipLin { w, n_out, n_in, bounds })
}

/// Target (height, width) that preserves aspect ratio within the patch budget and is divisible
/// by `pooling_kernel_size * patch_size` (mirrors the HF image processor).
pub fn target_size(height: usize, width: usize, patch: usize, max_patches: usize, pool_k: usize) -> Result<(usize, usize)> {
    let total_px = (height * width) as f64;
    let target_px = (max_patches * patch * patch) as f64;
    let factor = (target_px / total_px).sqrt();
    let side_mult = pool_k * patch;
    let mut th = ((factor * height as f64 / side_mult as f64).floor() as usize) * side_mult;
    let mut tw = ((factor * width as f64 / side_mult as f64).floor() as usize) * side_mult;
    if th == 0 && tw == 0 {
        bail!("image too thin to resize into patches");
    }
    let max_side = (max_patches / (pool_k * pool_k)) * side_mult;
    if th == 0 {
        th = side_mult;
        tw = ((width / height) * side_mult).min(max_side);
    } else if tw == 0 {
        tw = side_mult;
        th = ((height / width) * side_mult).min(max_side);
    }
    if (th * tw) as f64 > target_px {
        bail!("resize target exceeds the patch budget");
    }
    Ok((th, tw))
}

/// Keys cubic kernel with a = -0.5 (PIL / torchvision "bicubic").
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

/// Per-axis antialiased resampling weights (PIL-style: support scales with the downscale factor).
fn resample_weights(n_in: usize, n_out: usize) -> Vec<(usize, Vec<f32>)> {
    let scale = n_in as f32 / n_out as f32;
    let filter_scale = scale.max(1.0);
    let support = 2.0 * filter_scale;
    (0..n_out)
        .map(|i| {
            let center = (i as f32 + 0.5) * scale;
            let lo = ((center - support + 0.5).floor().max(0.0)) as usize;
            let hi = ((center + support + 0.5).floor() as usize).min(n_in);
            let mut w: Vec<f32> = (lo..hi).map(|j| cubic((j as f32 + 0.5 - center) / filter_scale)).collect();
            let s: f32 = w.iter().sum();
            if s != 0.0 {
                for v in w.iter_mut() {
                    *v /= s;
                }
            }
            (lo, w)
        })
        .collect()
}

/// Bicubic, antialiased resize of an interleaved RGB8 image to f32 RGB in [0, 1].
pub fn resize_rgb(rgb: &[u8], w: usize, h: usize, tw: usize, th: usize) -> Vec<f32> {
    let wx = resample_weights(w, tw);
    let wy = resample_weights(h, th);
    // Horizontal pass: [h, tw, 3]
    let mut tmp = vec![0f32; h * tw * 3];
    for y in 0..h {
        let row = &rgb[y * w * 3..(y + 1) * w * 3];
        for (ox, (lo, ws)) in wx.iter().enumerate() {
            let mut acc = [0f32; 3];
            for (k, &wgt) in ws.iter().enumerate() {
                let px = &row[(lo + k) * 3..(lo + k) * 3 + 3];
                acc[0] += wgt * px[0] as f32;
                acc[1] += wgt * px[1] as f32;
                acc[2] += wgt * px[2] as f32;
            }
            tmp[(y * tw + ox) * 3..(y * tw + ox) * 3 + 3].copy_from_slice(&acc);
        }
    }
    // Vertical pass: [th, tw, 3]
    let mut out = vec![0f32; th * tw * 3];
    for (oy, (lo, ws)) in wy.iter().enumerate() {
        for x in 0..tw * 3 {
            let mut acc = 0f32;
            for (k, &wgt) in ws.iter().enumerate() {
                acc += wgt * tmp[(lo + k) * tw * 3 + x];
            }
            out[oy * tw * 3 + x] = (acc / 255.0).clamp(0.0, 1.0);
        }
    }
    out
}

/// Preprocessed image: flattened 16x16x3 patches in [0, 1] and their (x, y) grid positions.
pub struct ImagePatches {
    pub patches: Vec<f32>, // [n, patch*patch*3], (row, col, channel) order
    pub pos: Vec<(usize, usize)>,
    pub n: usize,
    pub pw: usize,
    pub ph: usize,
}

pub fn preprocess(rgb: &[u8], w: usize, h: usize, cfg: &VisionConfig, max_soft_tokens: usize) -> Result<ImagePatches> {
    let p = cfg.patch_size;
    let k = cfg.pooling_kernel_size;
    let (th, tw) = target_size(h, w, p, max_soft_tokens * k * k, k)?;
    let img = if (th, tw) == (h, w) {
        rgb.iter().map(|&v| v as f32 / 255.0).collect()
    } else {
        resize_rgb(rgb, w, h, tw, th)
    };
    Ok(patchify(&img, tw, th, p))
}

/// A video frame as the video processor prepares it: resized to the patch budget with Pillow's
/// 8-bit bicubic (torchvision's uint8 resize in the reference), then scaled to [0, 1].
pub fn preprocess_frame(rgb: &[u8], w: usize, h: usize, cfg: &VisionConfig, max_soft_tokens: usize) -> Result<ImagePatches> {
    let p = cfg.patch_size;
    let k = cfg.pooling_kernel_size;
    let (th, tw) = target_size(h, w, p, max_soft_tokens * k * k, k)?;
    let img: Vec<f32> = crate::resample::resize_bicubic_rgb8(rgb, w, h, tw, th)
        .iter()
        .map(|&v| v as f32 / 255.0)
        .collect();
    Ok(patchify(&img, tw, th, p))
}

/// Cuts an RGB image in [0, 1] (`tw` x `th`, multiples of `p`) into `p` x `p` patches.
fn patchify(img: &[f32], tw: usize, th: usize, p: usize) -> ImagePatches {
    let (ph, pw) = (th / p, tw / p);
    let n = ph * pw;
    let d = p * p * 3;
    let mut patches = vec![0f32; n * d];
    let mut pos = Vec::with_capacity(n);
    for py in 0..ph {
        for px in 0..pw {
            let i = py * pw + px;
            let dst = &mut patches[i * d..(i + 1) * d];
            for r in 0..p {
                for c in 0..p {
                    let src = ((py * p + r) * tw + (px * p + c)) * 3;
                    let o = (r * p + c) * 3;
                    dst[o..o + 3].copy_from_slice(&img[src..src + 3]);
                }
            }
            pos.push((px, py));
        }
    }
    ImagePatches { patches, pos, n, pw, ph }
}

struct VLayer<'a> {
    q: ClipLin<'a>,
    k: ClipLin<'a>,
    v: ClipLin<'a>,
    o: ClipLin<'a>,
    q_norm: &'a [u16],
    k_norm: &'a [u16],
    ln_in: &'a [u16],
    ln_post_attn: &'a [u16],
    ln_pre_ffn: &'a [u16],
    ln_post_ffn: &'a [u16],
    gate: ClipLin<'a>,
    up: ClipLin<'a>,
    down: ClipLin<'a>,
}

pub struct VisionTower<'a> {
    pub cfg: VisionConfig,
    input_proj: &'a [u16], // [H, 3*p*p]
    pos_table: &'a [u16],  // [2, P, H]
    layers: Vec<VLayer<'a>>,
    inv_freq: Vec<f32>,    // head_dim/4 frequencies per spatial axis
    embed_proj: &'a [u16], // [text_hidden, H]
    pub text_hidden: usize,
}

impl<'a> VisionTower<'a> {
    pub fn load(cfg: VisionConfig, files: &'a ModelFiles, text_hidden: usize) -> Result<Self> {
        let pre = "model.vision_tower.";
        let h = cfg.hidden_size;
        let hd = cfg.head_dim;
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let get = |name: &str, shape: &[usize]| -> Result<&'a [u16]> {
            let full = format!("{pre}{name}");
            let info = files.info(&full).with_context(|| format!("missing tensor {full}"))?;
            if info.shape != shape {
                bail!("tensor {full} has shape {:?}, expected {:?}", info.shape, shape);
            }
            files.bf16(&full)
        };
        let clipped = cfg.use_clipped_linears;
        let mut layers = Vec::new();
        for l in 0..cfg.num_hidden_layers {
            let lp = format!("{pre}encoder.layers.{l}.");
            let cl = |n: &str, o: usize, i: usize| load_clip_lin(files, &format!("{lp}{n}"), o, i, clipped);
            let g = |n: &str, s: &[usize]| get(&format!("encoder.layers.{l}.{n}"), s);
            layers.push(VLayer {
                q: cl("self_attn.q_proj", nh * hd, h)?,
                k: cl("self_attn.k_proj", nkv * hd, h)?,
                v: cl("self_attn.v_proj", nkv * hd, h)?,
                o: cl("self_attn.o_proj", h, nh * hd)?,
                q_norm: g("self_attn.q_norm.weight", &[hd])?,
                k_norm: g("self_attn.k_norm.weight", &[hd])?,
                ln_in: g("input_layernorm.weight", &[h])?,
                ln_post_attn: g("post_attention_layernorm.weight", &[h])?,
                ln_pre_ffn: g("pre_feedforward_layernorm.weight", &[h])?,
                ln_post_ffn: g("post_feedforward_layernorm.weight", &[h])?,
                gate: cl("mlp.gate_proj", cfg.intermediate_size, h)?,
                up: cl("mlp.up_proj", cfg.intermediate_size, h)?,
                down: cl("mlp.down_proj", h, cfg.intermediate_size)?,
            });
        }
        // Axial RoPE: spatial_dim = head_dim/2, inv_freq over arange(0, spatial_dim, 2)/spatial_dim.
        let spatial = hd / 2;
        let theta = cfg.rope_theta();
        let inv_freq: Vec<f32> = (0..spatial / 2)
            .map(|i| (1.0 / theta.powf((2 * i) as f64 / spatial as f64)) as f32)
            .collect();
        let embed_name = "model.embed_vision.embedding_projection.weight";
        let embed_proj = {
            let info = files.info(embed_name).context("missing embed_vision projection")?;
            if info.shape != [text_hidden, h] {
                bail!("embed_vision has shape {:?}", info.shape);
            }
            files.bf16(embed_name)?
        };
        Ok(Self {
            input_proj: get("patch_embedder.input_proj.weight", &[h, 3 * cfg.patch_size * cfg.patch_size])?,
            pos_table: get("patch_embedder.position_embedding_table", &[2, cfg.position_embedding_size, h])?,
            layers,
            inv_freq,
            embed_proj,
            text_hidden,
            cfg,
        })
    }

    /// Encodes one image into soft tokens projected to the text embedding space: `[n_soft, text_hidden]`.
    pub fn encode(&self, img: &ImagePatches) -> Vec<f32> {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let hd = cfg.head_dim;
        let nh = cfg.num_attention_heads;
        let n = img.n;
        let eps = cfg.rms_norm_eps as f32;
        let d = cfg.patch_size * cfg.patch_size * 3;

        // Patch embedding: 2*(p-0.5) -> linear -> + x/y position embeddings.
        let pix: Vec<f32> = img.patches.iter().map(|v| 2.0 * (v - 0.5)).collect();
        let mut x = vec![0f32; n * h];
        matmul(self.input_proj, h, d, &pix, n, &mut x);
        let psz = cfg.position_embedding_size;
        for (i, &(px, py)) in img.pos.iter().enumerate() {
            let ex = &self.pos_table[px * h..(px + 1) * h];
            let ey = &self.pos_table[(psz + py) * h..(psz + py + 1) * h];
            let row = &mut x[i * h..(i + 1) * h];
            for j in 0..h {
                row[j] += bf16_to_f32(ex[j]) + bf16_to_f32(ey[j]);
            }
        }

        // RoPE tables per patch: quarter = head_dim/4 angles for x and for y.
        let quarter = self.inv_freq.len();
        let xs: Vec<usize> = img.pos.iter().map(|p| p.0).collect();
        let ys: Vec<usize> = img.pos.iter().map(|p| p.1).collect();
        let (cos_x, sin_x) = rope_tables(&xs, &self.inv_freq);
        let (cos_y, sin_y) = rope_tables(&ys, &self.inv_freq);
        let rope_2d = |v: &mut [f32], i: usize| {
            let (a, b) = v.split_at_mut(hd / 2);
            apply_rope(a, &cos_x[i * quarter..(i + 1) * quarter], &sin_x[i * quarter..(i + 1) * quarter]);
            apply_rope(b, &cos_y[i * quarter..(i + 1) * quarter], &sin_y[i * quarter..(i + 1) * quarter]);
        };

        let mut hn = vec![0f32; n * h];
        let mut q = vec![0f32; n * nh * hd];
        let mut k = vec![0f32; n * nh * hd];
        let mut v = vec![0f32; n * nh * hd];
        let mut att = vec![0f32; n * nh * hd];
        let mut o = vec![0f32; n * h];
        let inter = cfg.intermediate_size;
        let mut g = vec![0f32; n * inter];
        let mut u = vec![0f32; n * inter];
        for layer in &self.layers {
            rmsnorm_rows(&x, h, Some(layer.ln_in), eps, &mut hn);
            layer.q.apply(&hn, n, &mut q);
            layer.k.apply(&hn, n, &mut k);
            layer.v.apply(&hn, n, &mut v);
            for_chunks(&mut q, hd, true, |i, qh| {
                rmsnorm_inplace(qh, Some(layer.q_norm), eps);
                rope_2d(qh, i / nh);
            });
            for_chunks(&mut k, hd, true, |i, kh| {
                rmsnorm_inplace(kh, Some(layer.k_norm), eps);
                rope_2d(kh, i / nh);
            });
            for_chunks(&mut v, hd, true, |_, vh| rmsnorm_inplace(vh, None, eps));
            attention_bidirectional(&q, &k, &v, n, nh, hd, &mut att);
            layer.o.apply(&att, n, &mut o);
            rmsnorm_rows_inplace(&mut o, h, Some(layer.ln_post_attn), eps);
            add_inplace(&mut x, &o);

            rmsnorm_rows(&x, h, Some(layer.ln_pre_ffn), eps, &mut hn);
            layer.gate.apply(&hn, n, &mut g);
            layer.up.apply(&hn, n, &mut u);
            gelu_mul(&mut g, &u);
            layer.down.apply(&g, n, &mut o);
            rmsnorm_rows_inplace(&mut o, h, Some(layer.ln_post_ffn), eps);
            add_inplace(&mut x, &o);
        }

        // 3x3 average pooling by grid position, scaled by sqrt(hidden).
        let kk = cfg.pooling_kernel_size;
        let (gw, gh) = (img.pw / kk, img.ph / kk);
        let n_soft = gw * gh;
        let mut pooled = vec![0f32; n_soft * h];
        for (i, &(px, py)) in img.pos.iter().enumerate() {
            let cell = (px / kk) + gw * (py / kk);
            let dst = &mut pooled[cell * h..(cell + 1) * h];
            for (dj, &xj) in dst.iter_mut().zip(&x[i * h..(i + 1) * h]) {
                *dj += xj;
            }
        }
        let scale = (h as f32).sqrt() / (kk * kk) as f32;
        for v in pooled.iter_mut() {
            *v *= scale;
        }

        // Embedder: RMSNorm without scale, then projection to the text hidden size.
        rmsnorm_rows_inplace(&mut pooled, h, None, eps);
        let mut out = vec![0f32; n_soft * self.text_hidden];
        matmul(self.embed_proj, self.text_hidden, h, &pooled, n_soft, &mut out);
        out
    }
}

/// Full (non-causal) attention over `n` tokens; q/k/v are `[n, nh, hd]`, scaling 1.0.
pub fn attention_bidirectional(q: &[f32], k: &[f32], v: &[f32], n: usize, nh: usize, hd: usize, out: &mut [f32]) {
    let op = SendPtrF(out.as_mut_ptr());
    pool::run(n * nh, &|idx| {
        let (ti, hh) = (idx / nh, idx % nh);
        let qv = &q[(ti * nh + hh) * hd..(ti * nh + hh + 1) * hd];
        let mut scores: Vec<f32> = (0..n).map(|j| dot(qv, &k[(j * nh + hh) * hd..(j * nh + hh + 1) * hd])).collect();
        softmax_inplace(&mut scores);
        // SAFETY: each task owns one disjoint [hd] slice of `out`.
        let o = unsafe { std::slice::from_raw_parts_mut(op.get().add((ti * nh + hh) * hd), hd) };
        o.fill(0.0);
        for (j, &p) in scores.iter().enumerate() {
            axpy(o, p, &v[(j * nh + hh) * hd..(j * nh + hh + 1) * hd]);
        }
    });
}

#[derive(Clone, Copy)]
struct SendPtrF(*mut f32);
unsafe impl Send for SendPtrF {}
unsafe impl Sync for SendPtrF {}
impl SendPtrF {
    #[inline(always)]
    fn get(self) -> *mut f32 {
        self.0
    }
}
