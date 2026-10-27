//! Gemma 4 vision: image preprocessing (aspect-preserving bicubic resize into 16x16 patches),
//! the ViT encoder with axial 2D RoPE and bidirectional attention, 3x3 patch pooling, and the
//! projection of soft tokens into the language model's embedding space.

use crate::kernels::*;
use crate::mm_config::VisionConfig;
use crate::model::{Lin, linear};
use crate::pool;
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};

pub const SUPPORTED_SOFT_TOKENS: [usize; 5] = [70, 140, 280, 560, 1120];

/// A linear layer with optional input/output clamping (`use_clipped_linears`). The matmul runs
/// on the active GPU backend when it takes the size (prefill-sized inputs), else on the CPU.
pub struct ClipLin<'a> {
    pub n_out: usize,
    pub n_in: usize,
    pub bounds: Option<(f32, f32, f32, f32)>, // in_min, in_max, out_min, out_max
    lin: Lin<'a>,
}

impl ClipLin<'_> {
    pub fn apply(&self, x: &[f32], t: usize, y: &mut [f32]) {
        debug_assert_eq!(x.len(), t * self.n_in);
        match self.bounds {
            Some((imin, imax, omin, omax)) => {
                let xc: Vec<f32> = x.iter().map(|v| v.clamp(imin, imax)).collect();
                linear(&self.lin, self.n_out, self.n_in, &xc, t, y);
                for v in y.iter_mut() {
                    *v = v.clamp(omin, omax);
                }
            }
            None => linear(&self.lin, self.n_out, self.n_in, x, t, y),
        }
    }
}

impl ClipLin<'_> {
    /// A copy zero-padded to `n_out` x `n_in` (accelerator kernels take whole tiles: the 12B /
    /// 31B / 26B towers' MLP is 4304 wide). Padded outputs are 0 (or the clamp of 0) and meet
    /// zero weights in the next layer, so results do not change.
    fn padded(&self, n_out: usize, n_in: usize) -> Self {
        let crate::model::Weight::Bf16(src, _) = &self.lin.w else {
            unreachable!("vision weights are bf16")
        };
        let mut w = crate::quant::PageVec::<u16>::zeroed(n_out * n_in);
        for (dst, row) in w.chunks_mut(n_in).zip(src.chunks(self.n_in)) {
            dst[..self.n_in].copy_from_slice(row);
        }
        Self {
            n_out,
            n_in,
            bounds: self.bounds,
            lin: Lin::bf16_owned(w),
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
    Ok(ClipLin {
        n_out,
        n_in,
        bounds,
        lin: Lin::bf16(w),
    })
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

/// Preprocessed image: flattened 16x16x3 patches in [0, 1] and their (x, y) grid positions.
pub struct ImagePatches {
    pub patches: Vec<f32>, // [n, patch*patch*3], (row, col, channel) order
    pub pos: Vec<(usize, usize)>,
    pub n: usize,
    pub pw: usize,
    pub ph: usize,
}

/// The image and video-frame processor: an aspect-preserving resize to the patch budget with
/// torchvision's uint8 antialiased bicubic (the reference processors' default backend, exactly),
/// then scaling to [0, 1] and 16x16 patches.
pub fn preprocess(rgb: &[u8], w: usize, h: usize, cfg: &VisionConfig, max_soft_tokens: usize) -> Result<ImagePatches> {
    let p = cfg.patch_size;
    let k = cfg.pooling_kernel_size;
    let (th, tw) = target_size(h, w, p, max_soft_tokens * k * k, k)?;
    let img: Vec<f32> = crate::resample::resize_bicubic_rgb8_torchvision(rgb, w, h, tw, th)
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
    /// `standardize` (12B / 31B / 26B-A4B): pooled features become `(x - bias) * scale`.
    standardize: Option<(Vec<f32>, Vec<f32>)>,
    pub text_hidden: usize,
    /// MLP width as run (padded, see `ClipLin::padded`).
    inter_p: usize,
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
        // With an accelerator, an MLP width that is not a whole number of tiles is zero-padded
        // so its matmuls run there (copies of those weights, 0.8 GB for the 26B tower).
        let inter_p = match crate::gpu::active() {
            Some(_) => cfg.intermediate_size.next_multiple_of(64),
            None => cfg.intermediate_size,
        };
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
            if inter_p != cfg.intermediate_size {
                let l = layers.last_mut().unwrap();
                l.gate = l.gate.padded(inter_p, h);
                l.up = l.up.padded(inter_p, h);
                l.down = l.down.padded(h, inter_p);
            }
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
        let standardize = if cfg.standardize {
            let f = |n: &str| -> Result<Vec<f32>> {
                let full = format!("{pre}{n}");
                let info = files.info(&full).with_context(|| format!("missing tensor {full}"))?;
                if info.shape != [h] {
                    bail!("tensor {full} has shape {:?}, expected [{h}]", info.shape);
                }
                files.f32(&full)
            };
            Some((f("std_bias")?, f("std_scale")?))
        } else {
            None
        };
        Ok(Self {
            standardize,
            input_proj: get("patch_embedder.input_proj.weight", &[h, 3 * cfg.patch_size * cfg.patch_size])?,
            pos_table: get("patch_embedder.position_embedding_table", &[2, cfg.position_embedding_size, h])?,
            layers,
            inv_freq,
            embed_proj,
            text_hidden,
            inter_p,
            cfg,
        })
    }

    /// Encodes one image into soft tokens projected to the text embedding space: `[n_soft, text_hidden]`.
    pub fn encode(&self, img: &ImagePatches) -> Vec<f32> {
        self.encode_many(&[img]).pop().unwrap()
    }

    /// Encodes several images (video frames) in one pass: the linear layers run over all their
    /// patches at once, attention stays within each image. One `[n_soft, text_hidden]` per image.
    pub fn encode_many(&self, imgs: &[&ImagePatches]) -> Vec<Vec<f32>> {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let hd = cfg.head_dim;
        let nh = cfg.num_attention_heads;
        let n: usize = imgs.iter().map(|i| i.n).sum();
        let eps = cfg.rms_norm_eps as f32;
        let d = cfg.patch_size * cfg.patch_size * 3;
        let pos: Vec<(usize, usize)> = imgs.iter().flat_map(|i| i.pos.iter().copied()).collect();

        // Patch embedding: 2*(p-0.5) -> linear -> + x/y position embeddings.
        let pix: Vec<f32> = imgs.iter().flat_map(|i| i.patches.iter().map(|v| 2.0 * (v - 0.5))).collect();
        let mut x = vec![0f32; n * h];
        matmul(self.input_proj, h, d, &pix, n, &mut x);
        let psz = cfg.position_embedding_size;
        for (i, &(px, py)) in pos.iter().enumerate() {
            let ex = &self.pos_table[px * h..(px + 1) * h];
            let ey = &self.pos_table[(psz + py) * h..(psz + py + 1) * h];
            let row = &mut x[i * h..(i + 1) * h];
            for j in 0..h {
                row[j] += bf16_to_f32(ex[j]) + bf16_to_f32(ey[j]);
            }
        }

        // RoPE tables per patch: quarter = head_dim/4 angles for x and for y.
        let quarter = self.inv_freq.len();
        let xs: Vec<usize> = pos.iter().map(|p| p.0).collect();
        let ys: Vec<usize> = pos.iter().map(|p| p.1).collect();
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
        let inter = self.inter_p;
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
            let mut s0 = 0;
            for img in imgs {
                let r = s0 * nh * hd..(s0 + img.n) * nh * hd;
                attention_bidirectional(&q[r.clone()], &k[r.clone()], &v[r.clone()], img.n, nh, hd, &mut att[r]);
                s0 += img.n;
            }
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

        // 3x3 average pooling by grid position, scaled by sqrt(hidden), per image.
        let kk = cfg.pooling_kernel_size;
        let scale = (h as f32).sqrt() / (kk * kk) as f32;
        let mut pooled = Vec::new();
        let mut counts = Vec::with_capacity(imgs.len());
        let mut s0 = 0;
        for img in imgs {
            let (gw, gh) = (img.pw / kk, img.ph / kk);
            let base = pooled.len();
            pooled.resize(base + gw * gh * h, 0f32);
            for (i, &(px, py)) in img.pos.iter().enumerate() {
                let cell = (px / kk) + gw * (py / kk);
                let dst = &mut pooled[base + cell * h..base + (cell + 1) * h];
                for (dj, &xj) in dst.iter_mut().zip(&x[(s0 + i) * h..(s0 + i + 1) * h]) {
                    *dj += xj;
                }
            }
            counts.push(gw * gh);
            s0 += img.n;
        }
        for v in pooled.iter_mut() {
            *v *= scale;
        }
        if let Some((bias, sc)) = &self.standardize {
            for row in pooled.chunks_mut(h) {
                for ((v, b), s) in row.iter_mut().zip(bias).zip(sc) {
                    *v = (*v - b) * s;
                }
            }
        }

        // Embedder: RMSNorm without scale, then projection to the text hidden size.
        let n_soft: usize = counts.iter().sum();
        rmsnorm_rows_inplace(&mut pooled, h, None, eps);
        let mut out = vec![0f32; n_soft * self.text_hidden];
        matmul(self.embed_proj, self.text_hidden, h, &pooled, n_soft, &mut out);
        let mut res = Vec::with_capacity(imgs.len());
        let mut s0 = 0;
        for c in counts {
            res.push(out[s0 * self.text_hidden..(s0 + c) * self.text_hidden].to_vec());
            s0 += c;
        }
        res
    }
}

/// Full (non-causal) attention over `n` tokens; q/k/v are `[n, nh, hd]`, scaling 1.0. Longer
/// inputs (image towers) go through the GEMM-based attention (GPU when one is active) with q
/// pre-scaled by sqrt(hd) to cancel its 1/sqrt(hd).
pub fn attention_bidirectional(q: &[f32], k: &[f32], v: &[f32], n: usize, nh: usize, hd: usize, out: &mut [f32]) {
    if n >= 256 {
        let s = (hd as f32).sqrt();
        let qs: Vec<f32> = q.iter().map(|x| x * s).collect();
        out.copy_from_slice(&crate::sd::ops::attention(&qs, k, v, n, n, nh, hd, false));
        return;
    }
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
