//! Inkling-2's vision tower, as transformers' `Inkling2VisionModel` and `Inkling2ImageProcessor`
//! (the text decoder is [`crate::inkling`]).
//!
//! An image is not resized: it is padded on the right and at the bottom to whole tiles of
//! `merge * patch` pixels (40; the padding is -1 on the 0..255 scale), and each tile becomes
//! one token of the language model. A tile is `merge x merge` native patches of `patch` pixels
//! (20), embedded by a linear layer plus a learned position (a `grid x grid` table, bilinearly
//! resampled to the image's patch grid); a class token leads the sequence. The encoder is a
//! pre-norm ViT with LayerScale and axial rotary positions (half of each head turns with the
//! column, half with the row; patches are numbered from 1, the class token sits at 0). The
//! patches of a tile, concatenated, go through a two-layer GELU MLP to the text width.
//!
//! Images with an alpha channel are composited onto a checkerboard by the reference processor;
//! here images are decoded to RGB before they arrive, so transparency is not reproduced.

use crate::kernels::*;
use crate::model::{Lin, Quant, linear, linear_multi, load_lin};
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};
use serde_json::Value;
use std::path::Path;

const P: &str = "model.visual.";

#[derive(Debug, Clone)]
pub struct Inkling2VisionConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    /// Pixels per side of a native patch.
    pub patch_size: usize,
    /// Native patches per side of a tile (one language model token).
    pub spatial_merge_size: usize,
    /// Side of the learned position grid.
    pub posemb_grid_size: usize,
    pub projection_hidden_dim: usize,
    pub layer_norm_eps: f64,
    pub rope_theta: f64,
    /// The placeholder token an image feature replaces.
    pub image_token_id: u32,
}

impl Inkling2VisionConfig {
    /// The vision settings of the checkpoint in `dir`, None when it is not an Inkling-2 model
    /// with a vision tower.
    pub fn load(dir: &Path) -> Result<Option<Self>> {
        let Ok(bytes) = crate::fsx::read(&dir.join("config.json")) else {
            return Ok(None);
        };
        let raw: Value = serde_json::from_slice(&bytes)?;
        if raw.get("model_type").and_then(Value::as_str) != Some("inkling2") {
            return Ok(None);
        }
        match raw.get("vision_config").filter(|v| v.is_object()) {
            Some(v) => Self::from_json(v, raw.get("image_token_id").and_then(Value::as_u64).unwrap_or(200054) as u32).map(Some),
            None => Ok(None),
        }
    }

    pub fn from_json(v: &Value, image_token_id: u32) -> Result<Self> {
        let int = |keys: &[&str], default: usize| -> usize {
            keys.iter()
                .find_map(|k| v.get(*k).and_then(Value::as_u64))
                .map_or(default, |x| x as usize)
        };
        if v.get("projector_bias").and_then(Value::as_bool) == Some(true) {
            bail!("Inkling-2 vision: a projector with biases is not supported");
        }
        if int(&["num_channels"], 3) != 3 {
            bail!("Inkling-2 vision: images of 3 channels only");
        }
        let hidden_size = int(&["hidden_size"], 1024);
        let ratio = v.get("mlp_ratio").and_then(Value::as_f64).unwrap_or(4.0);
        let c = Self {
            hidden_size,
            num_hidden_layers: int(&["num_hidden_layers"], 23),
            num_attention_heads: int(&["num_attention_heads"], 16),
            intermediate_size: int(&["intermediate_size"], (hidden_size as f64 * ratio) as usize),
            patch_size: int(&["patch_size", "native_patch_size"], 20),
            spatial_merge_size: int(&["spatial_merge_size", "adapter_merge_size"], 2),
            posemb_grid_size: int(&["posemb_grid_size"], 32),
            projection_hidden_dim: int(&["projection_hidden_dim"], 4096),
            layer_norm_eps: v.get("layer_norm_eps").and_then(Value::as_f64).unwrap_or(1e-5),
            rope_theta: v
                .get("rope_parameters")
                .and_then(|r| r.get("rope_theta"))
                .and_then(Value::as_f64)
                .unwrap_or(10000.0),
            image_token_id,
        };
        if c.num_attention_heads == 0 || !c.hidden_size.is_multiple_of(c.num_attention_heads) || !c.head_dim().is_multiple_of(4) {
            bail!(
                "Inkling-2 vision: {} heads over a width of {}",
                c.num_attention_heads,
                c.hidden_size
            );
        }
        Ok(c)
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Pixels per side of a tile.
    pub fn tile(&self) -> usize {
        self.patch_size * self.spatial_merge_size
    }
}

/// A linear layer with a bias.
struct Affine<'a> {
    w: Lin<'a>,
    b: Vec<f32>,
}

impl Affine<'_> {
    fn add_bias(&self, y: &mut [f32]) {
        let n = self.b.len();
        for_chunks(y, n, y.len() > 64 * n, |_, r| add_inplace(r, &self.b));
    }
}

struct Block<'a> {
    ln_1: (Vec<f32>, Vec<f32>),
    ln_2: (Vec<f32>, Vec<f32>),
    q: Affine<'a>,
    k: Affine<'a>,
    v: Affine<'a>,
    o: Affine<'a>,
    fc: Affine<'a>,
    proj: Affine<'a>,
    /// The LayerScale factors of the attention and of the MLP.
    ls_1: Vec<f32>,
    ls_2: Vec<f32>,
}

pub struct Inkling2Vision<'a> {
    pub cfg: Inkling2VisionConfig,
    patch: Lin<'a>,     // [H, 3 * p * p]
    position: Vec<f32>, // [grid * grid, H]
    /// The class token: its embedding plus its position.
    class: Vec<f32>,
    ln_pre: (Vec<f32>, Vec<f32>),
    blocks: Vec<Block<'a>>,
    proj_in: Lin<'a>,  // [proj, m * m * H]
    proj_out: Lin<'a>, // [text, proj]
    text_hidden: usize,
}

/// A preprocessed image: `rows x cols` tiles, and the normalized native patches (`[3 * p * p]`
/// each: row, column, channel) tile by tile, the patches of a tile in raster order.
pub struct Inkling2Image {
    pub patches: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

/// LayerNorm with weight and bias over each row of `x` (`n` wide) into `y`.
fn layer_norm_rows(x: &[f32], n: usize, wb: &(Vec<f32>, Vec<f32>), eps: f32, y: &mut [f32]) {
    for_chunks(y, n, x.len() > 16 * n, |i, o| {
        let r = &x[i * n..(i + 1) * n];
        let mean = r.iter().sum::<f32>() / n as f32;
        let var = r.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for (((o, &v), &w), &b) in o.iter_mut().zip(r).zip(&wb.0).zip(&wb.1) {
            *o = (v - mean) * inv * w + b;
        }
    });
}

/// The two taps and weights of `F.interpolate(mode="bilinear", align_corners=False)` for
/// position `i` of `size` on a table of `side`.
fn taps(i: usize, size: usize, side: usize) -> [(usize, f32); 2] {
    let src = (i as f32 + 0.5) * side as f32 / size as f32 - 0.5;
    let f = src.floor();
    let clamp = |t: f32| (t.max(0.0) as usize).min(side - 1);
    let d = src - f;
    [(clamp(f), 1.0 - d), (clamp(f + 1.0), d)]
}

impl<'a> Inkling2Vision<'a> {
    pub fn load(cfg: Inkling2VisionConfig, files: &'a ModelFiles, text_hidden: usize) -> Result<Self> {
        let (h, inter) = (cfg.hidden_size, cfg.intermediate_size);
        let getf = |name: &str, n: usize| -> Result<Vec<f32>> {
            let full = format!("{P}{name}");
            let info = files.info(&full).with_context(|| format!("missing tensor {full}"))?;
            if info.shape.iter().product::<usize>() != n {
                bail!("tensor {full} has shape {:?}, expected {n} values", info.shape);
            }
            files.f32(&full)
        };
        let lin = |name: &str, n_out: usize, n_in: usize| load_lin(files, &format!("{P}{name}"), n_out, n_in, Quant::Bf16);
        let affine = |name: &str, n_out: usize, n_in: usize| -> Result<Affine<'a>> {
            Ok(Affine {
                w: lin(&format!("{name}.weight"), n_out, n_in)?,
                b: getf(&format!("{name}.bias"), n_out)?,
            })
        };
        let norm =
            |name: &str| -> Result<(Vec<f32>, Vec<f32>)> { Ok((getf(&format!("{name}.weight"), h)?, getf(&format!("{name}.bias"), h)?)) };
        if !files.has(&format!("{P}backbone.embeddings.patch_embedding.weight")) {
            bail!("this checkpoint has no vision tower ({P}backbone.*)");
        }
        let d = 3 * cfg.patch_size * cfg.patch_size;
        let m2 = cfg.spatial_merge_size * cfg.spatial_merge_size;
        let mut class = getf("backbone.embeddings.class_embedding", h)?;
        add_inplace(&mut class, &getf("backbone.embeddings.cls_position_embedding", h)?);
        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let p = format!("backbone.layers.{l}.");
            blocks.push(Block {
                ln_1: norm(&format!("{p}ln_1"))?,
                ln_2: norm(&format!("{p}ln_2"))?,
                q: affine(&format!("{p}self_attn.q_proj"), h, h)?,
                k: affine(&format!("{p}self_attn.k_proj"), h, h)?,
                v: affine(&format!("{p}self_attn.v_proj"), h, h)?,
                o: affine(&format!("{p}self_attn.out_proj"), h, h)?,
                fc: affine(&format!("{p}mlp.c_fc"), inter, h)?,
                proj: affine(&format!("{p}mlp.c_proj"), h, inter)?,
                ls_1: getf(&format!("{p}ls_1.gamma"), h)?,
                ls_2: getf(&format!("{p}ls_2.gamma"), h)?,
            });
        }
        Ok(Self {
            patch: lin("backbone.embeddings.patch_embedding.weight", h, d)?,
            position: getf(
                "backbone.embeddings.position_embedding",
                cfg.posemb_grid_size * cfg.posemb_grid_size * h,
            )?,
            class,
            ln_pre: norm("backbone.ln_pre")?,
            blocks,
            proj_in: lin("proj_in.weight", cfg.projection_hidden_dim, m2 * h)?,
            proj_out: lin("proj_out.weight", text_hidden, cfg.projection_hidden_dim)?,
            text_hidden,
            cfg,
        })
    }

    /// Cuts an RGB image (`[h][w][3]` bytes) into tiles, padding the last row and column.
    pub fn preprocess(&self, rgb: &[u8], w: usize, h: usize) -> Result<Inkling2Image> {
        if w == 0 || h == 0 || rgb.len() != w * h * 3 {
            bail!("image of {w}x{h} with {} bytes", rgb.len());
        }
        let (p, m) = (self.cfg.patch_size, self.cfg.spatial_merge_size);
        let tile = p * m;
        let (rows, cols) = (h.div_ceil(tile), w.div_ceil(tile));
        let d = 3 * p * p;
        // (x / 255 - 0.5) / 0.5, the padding being -1 before the rescale.
        let value = |x: f32| (x / 255.0 - 0.5) / 0.5;
        let pad = value(-1.0);
        let mut patches = vec![0f32; rows * cols * m * m * d];
        for_chunks(&mut patches, d, true, |i, o| {
            let (t, k) = (i / (m * m), i % (m * m));
            let (y0, x0) = ((t / cols * m + k / m) * p, (t % cols * m + k % m) * p);
            for (j, px) in o.as_chunks_mut::<3>().0.iter_mut().enumerate() {
                let (y, x) = (y0 + j / p, x0 + j % p);
                if y < h && x < w {
                    let s = &rgb[(y * w + x) * 3..][..3];
                    *px = [value(s[0] as f32), value(s[1] as f32), value(s[2] as f32)];
                } else {
                    *px = [pad; 3];
                }
            }
        });
        Ok(Inkling2Image { patches, rows, cols })
    }

    /// The features of one image, `[rows * cols, text_hidden]`.
    pub fn encode(&self, img: &Inkling2Image) -> Vec<f32> {
        let c = &self.cfg;
        let (h, nh, hd, m) = (c.hidden_size, c.num_attention_heads, c.head_dim(), c.spatial_merge_size);
        let (gh, gw) = (img.rows * m, img.cols * m);
        let np = gh * gw;
        let n = np + 1;
        let d = 3 * c.patch_size * c.patch_size;
        let eps = c.layer_norm_eps as f32;
        // The native grid position of patch `i` (tile by tile, raster inside a tile).
        let at = |i: usize| -> (usize, usize) {
            let (t, k) = (i / (m * m), i % (m * m));
            (t / img.cols * m + k / m, t % img.cols * m + k % m)
        };
        let mut x = vec![0f32; n * h];
        x[..h].copy_from_slice(&self.class);
        linear(&self.patch, h, d, &img.patches, np, &mut x[h..]);
        let side = c.posemb_grid_size;
        for_chunks(&mut x[h..], h, true, |i, row| {
            let (r, col) = at(i);
            for (tr, wr) in taps(r, gh, side) {
                for (tc, wc) in taps(col, gw, side) {
                    axpy(row, wr * wc, &self.position[(tr * side + tc) * h..(tr * side + tc + 1) * h]);
                }
            }
        });
        let mut y = vec![0f32; n * h];
        layer_norm_rows(&x, h, &self.ln_pre, eps, &mut y);
        std::mem::swap(&mut x, &mut y);

        // Axial rotary angles: the first half of a head's pairs turns with the column, the
        // second with the row; patches are numbered from 1, the class token stays at 0.
        let half = hd / 2;
        let quarter = half / 2;
        let inv: Vec<f32> = (0..quarter)
            .map(|j| 1.0 / (c.rope_theta as f32).powf((2 * j) as f32 / half as f32))
            .collect();
        let (mut cos, mut sin) = (vec![1f32; n * half], vec![0f32; n * half]);
        for i in 0..np {
            let (r, col) = at(i);
            for j in 0..half {
                let a = if j < quarter {
                    (col + 1) as f32 * inv[j]
                } else {
                    (r + 1) as f32 * inv[j - quarter]
                };
                cos[(i + 1) * half + j] = a.cos();
                sin[(i + 1) * half + j] = a.sin();
            }
        }

        let inter = c.intermediate_size;
        let mut hn = vec![0f32; n * h];
        let (mut q, mut k, mut v) = (vec![0f32; n * h], vec![0f32; n * h], vec![0f32; n * h]);
        let mut a = vec![0f32; n * inter];
        for b in &self.blocks {
            layer_norm_rows(&x, h, &b.ln_1, eps, &mut hn);
            linear_multi(
                &mut [(&b.q.w, h, &mut q[..]), (&b.k.w, h, &mut k[..]), (&b.v.w, h, &mut v[..])],
                h,
                &hn,
                n,
            );
            b.q.add_bias(&mut q);
            b.k.add_bias(&mut k);
            b.v.add_bias(&mut v);
            for buf in [&mut q, &mut k] {
                for_chunks(buf, hd, true, |i, head| {
                    let j = i / nh;
                    apply_rope_interleaved(head, &cos[j * half..(j + 1) * half], &sin[j * half..(j + 1) * half]);
                });
            }
            let att = crate::sd::ops::attention(&q, &k, &v, n, n, nh, hd, false);
            linear(&b.o.w, h, h, &att, n, &mut y);
            b.o.add_bias(&mut y);
            for_chunks(&mut x, h, true, |i, r| {
                for ((o, &v), &s) in r.iter_mut().zip(&y[i * h..(i + 1) * h]).zip(&b.ls_1) {
                    *o += v * s;
                }
            });
            layer_norm_rows(&x, h, &b.ln_2, eps, &mut hn);
            linear(&b.fc.w, inter, h, &hn, n, &mut a);
            b.fc.add_bias(&mut a);
            for_chunks(&mut a, inter, true, |_, r| {
                r.iter_mut().for_each(|v| *v = crate::bert::gelu_erf(*v))
            });
            linear(&b.proj.w, h, inter, &a, n, &mut y);
            b.proj.add_bias(&mut y);
            for_chunks(&mut x, h, true, |i, r| {
                for ((o, &v), &s) in r.iter_mut().zip(&y[i * h..(i + 1) * h]).zip(&b.ls_2) {
                    *o += v * s;
                }
            });
        }
        // The patches of a tile are consecutive: concatenated, they are the projector's input.
        let (t, md, ph) = (img.rows * img.cols, m * m * h, c.projection_hidden_dim);
        let mut mid = vec![0f32; t * ph];
        linear(&self.proj_in, ph, md, &x[h..], t, &mut mid);
        for_chunks(&mut mid, ph, true, |_, r| r.iter_mut().for_each(|v| *v = crate::bert::gelu_erf(*v)));
        let mut out = vec![0f32; t * self.text_hidden];
        linear(&self.proj_out, self.text_hidden, ph, &mid, t, &mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bilinear taps against `F.interpolate(align_corners=False)` on a table of 4.
    #[test]
    fn bilinear_taps() {
        // Upsampling 4 -> 8: position 0 reads the border twice, position 1 mixes 0 and 1.
        assert_eq!(taps(0, 8, 4), [(0, 0.25), (0, 0.75)]);
        assert_eq!(taps(1, 8, 4), [(0, 0.75), (1, 0.25)]);
        assert_eq!(taps(7, 8, 4), [(3, 0.75), (3, 0.25)]);
        // Same size: the table itself.
        assert_eq!(taps(2, 4, 4), [(2, 1.0), (3, 0.0)]);
    }
}
