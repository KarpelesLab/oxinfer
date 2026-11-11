//! MiniMax-VL-01's image input (LLaVA-NeXT "anyres" on a CLIP ViT-L/14-336), as MiniMax's own
//! code (`modeling_minimax_vl_01.py`, `modeling_clip.py`, `image_processor.py`) does it:
//!
//! - preprocessing (Pillow): the best of `image_grid_pinpoints` for the image (largest effective
//!   resolution, then least waste); a base view squashed to 336x336 (bicubic), then the image
//!   resized into the best resolution keeping its aspect ratio (bicubic) on black, centered, cut
//!   into 336x336 tiles row by row; CLIP mean / std;
//! - every view through the CLIP tower on its own (class token + learned positions, pre-LayerNorm,
//!   exact-GELU MLP blocks), the last layer's output without the final norm and without the class
//!   token: 576 features per view, each through the projector (GELU MLP);
//! - packing: the base view's 576 features, then the tiles' features as one grid
//!   (`tiles_h * 24` rows of `tiles_w * 24`), cropped back to the image's aspect ratio
//!   (`unpad_image`) with the learned `image_newline` after every row. The prompt holds as many
//!   `<image>` tokens: `576 + (w + 1) * h`.

use crate::bert::{LayerNorm, gelu_erf};
use crate::kernels::*;
use crate::model::{Lin, linear};
use crate::qwen_vision::VLin;
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};
use std::path::Path;

/// The settings of a MiniMax-VL-01 checkpoint (`config.json` of `minimax_vl_01`).
#[derive(Debug, Clone)]
pub struct Vl01Config {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub inter: usize,
    pub patch: usize,
    pub image_size: usize,
    pub text_hidden: usize,
    /// `(height, width)` pairs (symmetric in the release).
    pub pinpoints: Vec<(usize, usize)>,
    pub image_token_id: u32,
    pub eps: f32,
}

impl Vl01Config {
    /// None unless `dir` is a MiniMax-VL-01 checkpoint.
    pub fn load(dir: &Path) -> Result<Option<Self>> {
        if !dir.is_dir() {
            return Ok(None);
        }
        let Ok(b) = crate::fsx::read(&dir.join("config.json")) else {
            return Ok(None);
        };
        let raw: serde_json::Value = serde_json::from_slice(&b)?;
        if raw.get("model_type").and_then(|v| v.as_str()) != Some("minimax_vl_01") {
            return Ok(None);
        }
        let v = raw.get("vision_config").context("minimax_vl_01 config has no vision_config")?;
        let u = |o: &serde_json::Value, k: &str| {
            o.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .with_context(|| format!("no {k}"))
        };
        let text = raw.get("text_config").context("no text_config")?;
        let pinpoints = raw
            .get("image_grid_pinpoints")
            .and_then(|p| p.as_array())
            .context("no image_grid_pinpoints")?
            .iter()
            .filter_map(|p| Some((p.get(0)?.as_u64()? as usize, p.get(1)?.as_u64()? as usize)))
            .collect();
        if raw.get("vision_feature_layer").and_then(|x| x.as_i64()).unwrap_or(-1) != -1
            || raw
                .get("vision_feature_select_strategy")
                .and_then(|x| x.as_str())
                .unwrap_or("default")
                != "default"
        {
            bail!("MiniMax-VL-01: only the last layer's features without the class token are supported");
        }
        Ok(Some(Self {
            hidden: u(v, "hidden_size")?,
            layers: u(v, "num_hidden_layers")?,
            heads: u(v, "num_attention_heads")?,
            inter: u(v, "intermediate_size")?,
            patch: u(v, "patch_size")?,
            image_size: u(v, "image_size")?,
            text_hidden: u(text, "hidden_size")?,
            pinpoints,
            image_token_id: raw.get("image_token_index").and_then(|x| x.as_u64()).unwrap_or(200025) as u32,
            eps: v.get("layer_norm_eps").and_then(|x| x.as_f64()).unwrap_or(1e-5) as f32,
        }))
    }

    fn grid(&self) -> usize {
        self.image_size / self.patch
    }

    /// `select_best_resolution` (as `(width, height)`) of an image of `w`x`h`.
    pub fn best_resolution(&self, w: usize, h: usize) -> (usize, usize) {
        let (mut best, mut max_eff, mut min_waste) = ((0, 0), 0usize, usize::MAX);
        // The image processor reads the pinpoints as (width, height).
        for &(pw, ph) in &self.pinpoints {
            let scale = (pw as f64 / w as f64).min(ph as f64 / h as f64);
            let (dw, dh) = ((w as f64 * scale) as usize, (h as f64 * scale) as usize);
            let eff = (dw * dh).min(w * h);
            let waste = pw * ph - eff;
            if eff > max_eff || (eff == max_eff && waste < min_waste) {
                (best, max_eff, min_waste) = ((pw, ph), eff, waste);
            }
        }
        best
    }

    /// The kept feature grid `(cols, rows)` of the tiles of an image of `w`x`h` (`unpad_image` /
    /// `get_w_h_num` over the `tiles * 24` grid).
    pub fn unpadded(&self, w: usize, h: usize) -> (usize, usize) {
        let (bw, bh) = self.best_resolution(w, h);
        let g = self.grid();
        let (cw, ch) = (bw / self.image_size * g, bh / self.image_size * g);
        if w as f64 / h as f64 > cw as f64 / ch as f64 {
            let nh = h * cw / w;
            let pad = (ch - nh) / 2;
            (cw, ch - 2 * pad)
        } else {
            let nw = w * ch / h;
            let pad = (cw - nw) / 2;
            (cw - 2 * pad, ch)
        }
    }

    /// The number of `<image>` tokens of an image of `w`x`h`.
    pub fn tokens(&self, w: usize, h: usize) -> usize {
        let (uw, uh) = self.unpadded(w, h);
        self.grid() * self.grid() + (uw + 1) * uh
    }
}

/// The views of an image (RGB8 `w`x`h`): the base view, then the tiles row by row, each
/// `[3, 336, 336]` normalized, and the tile grid `(cols, rows)`.
pub fn preprocess(cfg: &Vl01Config, rgb: &[u8], w: usize, h: usize) -> (Vec<Vec<f32>>, (usize, usize)) {
    let s = cfg.image_size;
    const MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
    const STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];
    // torchvision ToTensor (x / 255) then Normalize.
    let norm = |img: &[u8], iw: usize, x0: usize, y0: usize| -> Vec<f32> {
        let mut out = vec![0f32; 3 * s * s];
        for c in 0..3 {
            for y in 0..s {
                for x in 0..s {
                    let v = img[((y0 + y) * iw + x0 + x) * 3 + c] as f32 / 255.0;
                    out[(c * s + y) * s + x] = (v - MEAN[c]) / STD[c];
                }
            }
        }
        out
    };
    let base = crate::resample::resize_bicubic_rgb8(rgb, w, h, s, s);
    let mut views = vec![norm(&base, s, 0, 0)];
    // keepratio_resize: fit inside the best resolution, centered on black.
    let (bw, bh) = cfg.best_resolution(w, h);
    let (rw, rh, ox, oy) = if w as f64 / h as f64 > bw as f64 / bh as f64 {
        let nh = bw * h / w;
        (bw, nh, 0, (bh - nh) / 2)
    } else {
        let nw = bh * w / h;
        (nw, bh, (bw - nw) / 2, 0)
    };
    let resized = crate::resample::resize_bicubic_rgb8(rgb, w, h, rw, rh);
    let mut canvas = vec![0u8; bw * bh * 3];
    for y in 0..rh {
        canvas[((oy + y) * bw + ox) * 3..((oy + y) * bw + ox + rw) * 3].copy_from_slice(&resized[y * rw * 3..(y + 1) * rw * 3]);
    }
    let (tw, th) = (bw / s, bh / s);
    for ty in 0..th {
        for tx in 0..tw {
            views.push(norm(&canvas, bw, tx * s, ty * s));
        }
    }
    (views, (tw, th))
}

struct Block<'a> {
    norm1: LayerNorm,
    norm2: LayerNorm,
    q: VLin<'a>,
    k: VLin<'a>,
    v: VLin<'a>,
    o: VLin<'a>,
    fc1: VLin<'a>,
    fc2: VLin<'a>,
}

/// The CLIP tower, the projector and `image_newline`.
pub struct Vl01Tower<'a> {
    pub cfg: Vl01Config,
    patch_w: Lin<'a>,
    class: Vec<f32>,
    pos: Vec<f32>,
    pre_norm: LayerNorm,
    blocks: Vec<Block<'a>>,
    proj1: VLin<'a>,
    proj2: VLin<'a>,
    newline: Vec<f32>,
}

impl<'a> Vl01Tower<'a> {
    pub fn load(cfg: Vl01Config, files: &'a ModelFiles) -> Result<Self> {
        let pre = "vision_tower.vision_model.";
        let (h, p) = (cfg.hidden, cfg.patch);
        let g = cfg.grid();
        let dense = |n: &str, o: usize, i: usize| VLin::load(files, n, (o, i), (o, i));
        let mut blocks = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            let b = format!("{pre}encoder.layers.{l}.");
            blocks.push(Block {
                norm1: LayerNorm::load(files, &format!("{b}layer_norm1"), h)?,
                norm2: LayerNorm::load(files, &format!("{b}layer_norm2"), h)?,
                q: dense(&format!("{b}self_attn.q_proj"), h, h)?,
                k: dense(&format!("{b}self_attn.k_proj"), h, h)?,
                v: dense(&format!("{b}self_attn.v_proj"), h, h)?,
                o: dense(&format!("{b}self_attn.out_proj"), h, h)?,
                fc1: dense(&format!("{b}mlp.fc1"), cfg.inter, h)?,
                fc2: dense(&format!("{b}mlp.fc2"), h, cfg.inter)?,
            });
        }
        let pos = files.f32(&format!("{pre}embeddings.position_embedding.weight"))?;
        let class = files.f32(&format!("{pre}embeddings.class_embedding"))?;
        let newline = files.f32("image_newline")?;
        if pos.len() != (g * g + 1) * h || class.len() != h || newline.len() != cfg.text_hidden {
            bail!("MiniMax-VL-01: unexpected embedding sizes");
        }
        Ok(Self {
            patch_w: Lin::bf16(files.bf16_conv(&format!("{pre}embeddings.patch_embedding.weight"), &[h, 3, p, p])?),
            class,
            pos,
            pre_norm: LayerNorm::load(files, &format!("{pre}pre_layrnorm"), h)?,
            blocks,
            proj1: dense("multi_modal_projector.linear_1", cfg.text_hidden, h)?,
            proj2: dense("multi_modal_projector.linear_2", cfg.text_hidden, cfg.text_hidden)?,
            newline,
            cfg,
        })
    }

    /// One `[3, 336, 336]` view into `[576, text_hidden]` projected features.
    fn view(&self, px: &[f32]) -> Vec<f32> {
        let cfg = &self.cfg;
        let (h, p, nh) = (cfg.hidden, cfg.patch, cfg.heads);
        let (g, s) = (cfg.grid(), cfg.image_size);
        let n = g * g + 1;
        // Patches as rows of [3 * p * p] (channel, y, x).
        let d = 3 * p * p;
        let mut patches = vec![0f32; g * g * d];
        for (i, row) in patches.chunks_mut(d).enumerate() {
            let (py, pxi) = (i / g, i % g);
            for c in 0..3 {
                for y in 0..p {
                    for x in 0..p {
                        row[(c * p + y) * p + x] = px[(c * s + py * p + y) * s + pxi * p + x];
                    }
                }
            }
        }
        let mut x = vec![0f32; n * h];
        x[..h].copy_from_slice(&self.class);
        linear(&self.patch_w, h, d, &patches, g * g, &mut x[h..]);
        add_inplace(&mut x, &self.pos);
        self.pre_norm.apply_rows(&mut x, cfg.eps);
        let hd = h / nh;
        let scale = (hd as f32).powf(-0.5);
        let (mut hn, mut q, mut k, mut v) = (vec![0f32; n * h], vec![0f32; n * h], vec![0f32; n * h], vec![0f32; n * h]);
        let (mut att, mut y, mut mid) = (vec![0f32; n * h], vec![0f32; n * h], vec![0f32; n * cfg.inter]);
        for b in &self.blocks {
            hn.copy_from_slice(&x);
            b.norm1.apply_rows(&mut hn, cfg.eps);
            b.q.forward(&hn, n, &mut q);
            b.k.forward(&hn, n, &mut k);
            b.v.forward(&hn, n, &mut v);
            q.iter_mut().for_each(|v| *v *= scale);
            crate::vision::attention_bidirectional(&q, &k, &v, n, nh, hd, &mut att);
            b.o.forward(&att, n, &mut y);
            add_inplace(&mut x, &y);
            hn.copy_from_slice(&x);
            b.norm2.apply_rows(&mut hn, cfg.eps);
            b.fc1.forward(&hn, n, &mut mid);
            for_chunks(&mut mid, cfg.inter, true, |_, r| r.iter_mut().for_each(|v| *v = gelu_erf(*v)));
            b.fc2.forward(&mid, n, &mut y);
            add_inplace(&mut x, &y);
        }
        // The last layer's output without the class token, projected.
        let th = cfg.text_hidden;
        let mut p1 = vec![0f32; (n - 1) * th];
        self.proj1.forward(&x[h..], n - 1, &mut p1);
        p1.iter_mut().for_each(|v| *v = gelu_erf(*v));
        let mut out = vec![0f32; (n - 1) * th];
        self.proj2.forward(&p1, n - 1, &mut out);
        out
    }

    /// The packed features of an image of `w`x`h` (see the module docs): `[tokens(w, h),
    /// text_hidden]`.
    pub fn encode(&self, views: &[Vec<f32>], (tw, th): (usize, usize), w: usize, h: usize) -> Vec<f32> {
        let cfg = &self.cfg;
        let (g, d) = (cfg.grid(), cfg.text_hidden);
        let feats: Vec<Vec<f32>> = views.iter().map(|v| self.view(v)).collect();
        let mut out = feats[0].clone();
        // The tiles as one (th * g) x (tw * g) grid, cropped to the image's aspect, a newline per row.
        let (cw, ch) = (tw * g, th * g);
        let (uw, uh) = cfg.unpadded(w, h);
        let (x0, y0) = ((cw - uw) / 2, (ch - uh) / 2);
        for r in y0..y0 + uh {
            for c in x0..x0 + uw {
                let (tile, pr, pc) = ((r / g) * tw + c / g, r % g, c % g);
                let f = &feats[1 + tile];
                out.extend_from_slice(&f[(pr * g + pc) * d..(pr * g + pc + 1) * d]);
            }
            out.extend_from_slice(&self.newline);
        }
        out
    }
}
