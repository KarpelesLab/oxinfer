//! The Qwen2-VL / Qwen2.5-VL vision tower (`Qwen2_5_VisionTransformerPretrainedModel`, the
//! `visual.*` weights): a Conv3d patch embedding without bias (14x14 pixels, 2 temporal frames),
//! no learned position embeddings, axial 2D RoPE (the first half of each head's rotary
//! frequencies with the patch row, the second with the column), RMSNorm pre-norm blocks with a
//! biased SwiGLU MLP, window attention (windows of `window_size` pixels, i.e. 4x4 merged tokens)
//! in every block but `fullatt_block_indexes`, which attend over the whole frame, and the patch
//! merger (RMSNorm, then every 2x2 block of patches concatenated through a GELU MLP into one
//! language-model token).
//!
//! The image processor is the Qwen3.5 one (`crate::qwen_vision::preprocess_image`) with this
//! checkpoint's settings (14-pixel patches, CLIP mean / std, 28-pixel resize factor).

use crate::kernels::*;
use crate::model::{Lin, linear};
use crate::pool;
use crate::qwen_vision::{Pixels, QwenVisionConfig, VLin};
use crate::safetensors::ModelFiles;
use anyhow::{Result, bail};

struct Block<'a> {
    norm1: Vec<f32>,
    norm2: Vec<f32>,
    qkv: VLin<'a>,
    proj: VLin<'a>,
    gate: VLin<'a>,
    up: VLin<'a>,
    down: VLin<'a>,
}

/// The Qwen2.5-VL vision tower (see the module docs).
pub struct Qwen25Tower<'a> {
    pub cfg: QwenVisionConfig,
    patch_w: Lin<'a>, // [H, C*T*p*p]
    blocks: Vec<Block<'a>>,
    merger_norm: Vec<f32>,
    merger_fc1: VLin<'a>,
    merger_fc2: VLin<'a>,
    /// MLP width padded to a multiple of 64.
    inter: usize,
    inv_freq: Vec<f32>, // head_dim / 4 frequencies per axis
}

impl<'a> Qwen25Tower<'a> {
    /// Loads `visual.*` (or `model.visual.*`) in bf16 (pre-quantized tensors load as stored).
    pub fn load(cfg: QwenVisionConfig, files: &'a ModelFiles) -> Result<Self> {
        if let Some(a) = &cfg.hidden_act
            && a != "silu"
        {
            bail!("Qwen2.5-VL vision hidden_act {a} is not supported (silu)");
        }
        let pre = if files.has("visual.patch_embed.proj.weight") {
            "visual."
        } else if files.has("model.visual.patch_embed.proj.weight") {
            "model.visual."
        } else {
            bail!("the checkpoint has no vision tower (visual.*)");
        };
        let h = cfg.hidden_size;
        let (c, t, p) = (cfg.in_channels, cfg.temporal_patch_size, cfg.patch_size);
        if !h.is_multiple_of(cfg.num_heads) || !(h / cfg.num_heads).is_multiple_of(4) {
            bail!("vision head size {} is not a multiple of 4", h / cfg.num_heads.max(1));
        }
        let (inter, ip) = (cfg.intermediate_size, cfg.intermediate_size.next_multiple_of(64));
        let lin = |name: &str, o: usize, i: usize, po: usize, pi: usize| VLin::load(files, &format!("{pre}{name}"), (o, i), (po, pi));
        let norm = |name: &str| -> Result<Vec<f32>> {
            let w = files.f32(&format!("{pre}{name}.weight"))?;
            if w.len() != h {
                bail!("{pre}{name}.weight has {} values, expected {h}", w.len());
            }
            Ok(w)
        };
        let patch_w = Lin::bf16(files.bf16_conv(&format!("{pre}patch_embed.proj.weight"), &[h, c, t, p, p])?);
        let mut blocks = Vec::with_capacity(cfg.depth);
        for l in 0..cfg.depth {
            let b = format!("blocks.{l}.");
            blocks.push(Block {
                norm1: norm(&format!("{b}norm1"))?,
                norm2: norm(&format!("{b}norm2"))?,
                qkv: lin(&format!("{b}attn.qkv"), 3 * h, h, 3 * h, h)?,
                proj: lin(&format!("{b}attn.proj"), h, h, h, h)?,
                gate: lin(&format!("{b}mlp.gate_proj"), inter, h, ip, h)?,
                up: lin(&format!("{b}mlp.up_proj"), inter, h, ip, h)?,
                down: lin(&format!("{b}mlp.down_proj"), h, inter, h, ip)?,
            });
        }
        let mh = h * cfg.spatial_merge_size * cfg.spatial_merge_size;
        let spatial = cfg.head_dim() / 2;
        let theta = cfg
            .rope_parameters
            .as_ref()
            .and_then(|r| r.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .unwrap_or(10000.0);
        let inv_freq = (0..spatial / 2)
            .map(|i| (1.0 / theta.powf((2 * i) as f64 / spatial as f64)) as f32)
            .collect();
        let merger_norm = files.f32(&format!("{pre}merger.ln_q.weight"))?;
        if merger_norm.len() != h {
            bail!("merger.ln_q.weight: {} values, expected {h}", merger_norm.len());
        }
        Ok(Self {
            patch_w,
            blocks,
            merger_norm,
            merger_fc1: lin("merger.mlp.0", mh, mh, mh, mh)?,
            merger_fc2: lin("merger.mlp.2", cfg.out_hidden_size, mh, cfg.out_hidden_size, mh)?,
            inter: ip,
            inv_freq,
            cfg,
        })
    }

    /// The window order of the merged tokens of one `gh x gw` patch frame (`get_vision_window_index`):
    /// windows of `vw x vw` merged tokens row-major, tokens row-major inside each, with the
    /// window lengths in merged tokens.
    fn windows(&self, gh: usize, gw: usize) -> (Vec<usize>, Vec<usize>) {
        let m = self.cfg.spatial_merge_size;
        let vw = (self.cfg.window_size.unwrap_or(112) / m / self.cfg.patch_size).max(1);
        let (lh, lw) = (gh / m, gw / m);
        let mut order = Vec::with_capacity(lh * lw);
        let mut lens = Vec::new();
        for wy in 0..lh.div_ceil(vw) {
            for wx in 0..lw.div_ceil(vw) {
                let n0 = order.len();
                for y in wy * vw..((wy + 1) * vw).min(lh) {
                    for x in wx * vw..((wx + 1) * vw).min(lw) {
                        order.push(y * lw + x);
                    }
                }
                lens.push(order.len() - n0);
            }
        }
        (order, lens)
    }

    /// Encodes preprocessed pixels into `[tokens, out_hidden_size]` language-model embeddings
    /// (tokens = patches / merge^2, in temporal-group then merged-row-major order).
    pub fn encode(&self, px: &Pixels) -> Vec<f32> {
        let cfg = &self.cfg;
        let (gt, gh, gw) = px.grid;
        let per_t = gh * gw;
        let n = gt * per_t;
        let h = cfg.hidden_size;
        let (nh, hd) = (cfg.num_heads, cfg.head_dim());
        let mu = cfg.spatial_merge_size * cfg.spatial_merge_size;
        let d = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
        assert_eq!(px.data.len(), n * d, "pixel buffer does not match the grid");
        let eps = 1e-6;

        // Window order of the patches: whole merge units moved together, frame by frame.
        let (worder, wlens) = self.windows(gh, gw);
        let units_per_t = per_t / mu;
        let mut perm = Vec::with_capacity(n); // perm[j] = original patch index at window position j
        let mut wins = Vec::new(); // (start, len) in patches
        for g in 0..gt {
            for &u in &worder {
                for k in 0..mu {
                    perm.push((g * units_per_t + u) * mu + k);
                }
            }
            let mut s = g * per_t;
            for &l in &wlens {
                wins.push((s, l * mu));
                s += l * mu;
            }
        }
        let frames: Vec<(usize, usize)> = (0..gt).map(|g| (g * per_t, per_t)).collect();

        // Patch embedding, already in window order.
        let mut xp = vec![0f32; n * h];
        linear(&self.patch_w, h, d, &px.data, n, &mut xp);
        let mut x = vec![0f32; n * h];
        for_chunks(&mut x, h, true, |j, row| row.copy_from_slice(&xp[perm[j] * h..(perm[j] + 1) * h]));
        drop(xp);

        // Axial RoPE angles per patch (in window order): the first quarter of each half rotates
        // with the row, the second with the column.
        let q4 = self.inv_freq.len();
        let half = 2 * q4;
        debug_assert_eq!(2 * half, hd);
        let m = cfg.spatial_merge_size;
        let bw = gw / m;
        let mut cos = vec![0f32; n * half];
        let mut sin = vec![0f32; n * half];
        for (j, &p) in perm.iter().enumerate() {
            let k = p % per_t;
            let (blk, inb) = (k / mu, k % mu);
            let (r, c) = ((blk / bw) * m + inb / m, (blk % bw) * m + inb % m);
            for (i, &f) in self.inv_freq.iter().enumerate() {
                let (a, b) = (r as f32 * f, c as f32 * f);
                cos[j * half + i] = a.cos();
                sin[j * half + i] = a.sin();
                cos[j * half + q4 + i] = b.cos();
                sin[j * half + q4 + i] = b.sin();
            }
        }

        let mut hn = vec![0f32; n * h];
        let mut qkv = vec![0f32; n * 3 * h];
        let mut y = vec![0f32; n * h];
        let mut g = vec![0f32; n * self.inter];
        let mut u = vec![0f32; n * self.inter];
        for (li, b) in self.blocks.iter().enumerate() {
            rmsnorm_f32w_rows(&x, h, &b.norm1, eps, &mut hn);
            b.qkv.forward(&hn, n, &mut qkv);
            let (mut q, mut k, mut v) = (vec![0f32; n * h], vec![0f32; n * h], vec![0f32; n * h]);
            for_chunks(&mut q, h, true, |i, row| row.copy_from_slice(&qkv[i * 3 * h..i * 3 * h + h]));
            for_chunks(&mut k, h, true, |i, row| {
                row.copy_from_slice(&qkv[i * 3 * h + h..i * 3 * h + 2 * h])
            });
            for_chunks(&mut v, h, true, |i, row| {
                row.copy_from_slice(&qkv[i * 3 * h + 2 * h..(i + 1) * 3 * h])
            });
            for buf in [&mut q, &mut k] {
                for_chunks(buf, hd, true, |i, head| {
                    let j = i / nh;
                    apply_rope(head, &cos[j * half..(j + 1) * half], &sin[j * half..(j + 1) * half]);
                });
            }
            let segs = if cfg.fullatt_block_indexes.contains(&li) { &frames } else { &wins };
            let att = segment_attention(&q, &k, &v, segs, nh, hd);
            b.proj.forward(&att, n, &mut y);
            add_inplace(&mut x, &y);

            rmsnorm_f32w_rows(&x, h, &b.norm2, eps, &mut hn);
            b.gate.forward(&hn, n, &mut g);
            b.up.forward(&hn, n, &mut u);
            silu_mul(&mut g, &u);
            b.down.forward(&g, n, &mut y);
            add_inplace(&mut x, &y);
        }

        // Merger over consecutive merge units (window order), then back to the original order.
        rmsnorm_f32w_rows_inplace(&mut x, h, &self.merger_norm, eps);
        let (tokens, mh) = (n / mu, h * mu);
        let mut m1 = vec![0f32; tokens * mh];
        self.merger_fc1.forward(&x, tokens, &mut m1);
        for_chunks(&mut m1, mh, true, |_, r| {
            for v in r.iter_mut() {
                *v = crate::bert::gelu_erf(*v);
            }
        });
        let od = cfg.out_hidden_size;
        let mut mo = vec![0f32; tokens * od];
        self.merger_fc2.forward(&m1, tokens, &mut mo);
        let mut out = vec![0f32; tokens * od];
        for j in 0..tokens {
            let t = perm[j * mu] / mu;
            out[t * od..(t + 1) * od].copy_from_slice(&mo[j * od..(j + 1) * od]);
        }
        out
    }
}

/// Bidirectional attention within each segment `(start, len)` of the rows (`[n, heads*hd]`
/// each, scale hd^-0.5). Long segments (whole frames) go through the shared attention (GPU when
/// active); short ones (windows) run here, one (segment, head) per task.
fn segment_attention(q: &[f32], k: &[f32], v: &[f32], segs: &[(usize, usize)], nh: usize, hd: usize) -> Vec<f32> {
    let d = nh * hd;
    let n = q.len() / d;
    let mut out = vec![0f32; n * d];
    let small: Vec<(usize, usize)> = segs.iter().copied().filter(|s| s.1 <= 256).collect();
    for &(s, l) in segs.iter().filter(|s| s.1 > 256) {
        let r = s * d..(s + l) * d;
        let o = crate::qwen_vision::attention_chunked(&q[r.clone()], &k[r.clone()], &v[r.clone()], l, nh, hd);
        out[r].copy_from_slice(&o);
    }
    if small.is_empty() {
        return out;
    }
    let scale = (hd as f32).powf(-0.5);
    let op = SyncPtr(out.as_mut_ptr());
    pool::run(small.len() * nh, &|task| {
        let (s, l) = small[task / nh];
        let hh = task % nh;
        let mut sc = vec![0f32; l];
        for i in 0..l {
            let qi = &q[(s + i) * d + hh * hd..(s + i) * d + (hh + 1) * hd];
            let mut mx = f32::NEG_INFINITY;
            for j in 0..l {
                let e = dot(qi, &k[(s + j) * d + hh * hd..(s + j) * d + (hh + 1) * hd]) * scale;
                sc[j] = e;
                mx = mx.max(e);
            }
            let mut sum = 0f32;
            for e in sc.iter_mut() {
                *e = (*e - mx).exp();
                sum += *e;
            }
            let inv = 1.0 / sum;
            // SAFETY: each (row, head) slice is written by this task only.
            let o = unsafe { std::slice::from_raw_parts_mut(op.get().add((s + i) * d + hh * hd), hd) };
            o.fill(0.0);
            for (j, &p) in sc.iter().enumerate() {
                let vj = &v[(s + j) * d + hh * hd..(s + j) * d + (hh + 1) * hd];
                let w = p * inv;
                for (a, &b) in o.iter_mut().zip(vj) {
                    *a += w * b;
                }
            }
        }
    });
    out
}

struct SyncPtr(*mut f32);
unsafe impl Sync for SyncPtr {}
unsafe impl Send for SyncPtr {}
impl SyncPtr {
    fn get(&self) -> *mut f32 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn window_order_matches_reference() {
        // get_vision_window_index for a 1x8x12 patch grid (4x6 merged tokens), window 112 px,
        // patch 14, merge 2: windows of 4x4 merged tokens -> [4x4, 4x2].
        let cfg: crate::qwen_vision::QwenVisionConfig = serde_json::from_value(serde_json::json!({
            "depth": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 1, "out_hidden_size": 8,
            "patch_size": 14, "window_size": 112, "fullatt_block_indexes": [0]
        }))
        .unwrap();
        let t = super::Qwen25Tower {
            cfg,
            patch_w: crate::model::Lin::bf16_owned(crate::quant::PageVec::zeroed(1)),
            blocks: Vec::new(),
            merger_norm: Vec::new(),
            merger_fc1: crate::qwen_vision::VLin::zero(1, 1),
            merger_fc2: crate::qwen_vision::VLin::zero(1, 1),
            inter: 0,
            inv_freq: Vec::new(),
        };
        let (order, lens) = t.windows(8, 12);
        assert_eq!(lens, vec![16, 8]);
        assert_eq!(&order[..6], &[0, 1, 2, 3, 6, 7]);
        assert_eq!(&order[16..20], &[4, 5, 10, 11]);
    }
}
