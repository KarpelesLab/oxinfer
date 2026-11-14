//! LFM3-Omni's image input (`Lfm3VlVisionModel` and the multimodal projector): a linear
//! patchifier over 16x16 patches (each frame twice along the temporal patch axis), an RMSNorm,
//! then gated depthwise 3x3 convolutions over the patch grid interleaved with bidirectional
//! attention (query / key RMSNorms, grouped key / value heads, axial 2D RoPE), and a projector
//! that folds each 2x2 block of patches into one language-model token.
//!
//! The patches travel in 2x2 merge-block order, as the Qwen processors emit them
//! (`crate::qwen_vision::preprocess_image`, the same smart resize and normalization).

use super::blocks::StackConfig;
use crate::bert::{LayerNorm, gelu_erf};
use crate::kernels::*;
use crate::model::{Lin, Quant, linear, load_lin};
use crate::qwen_vision::{Pixels, VLin, attention_chunked};
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};

/// The tower's shape (`vision_config`).
#[derive(Clone, Debug)]
pub struct VisionConfig {
    pub hidden: usize,
    pub inter: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub patch_dim: usize,
    pub merge: usize,
    pub eps: f32,
    pub rope_theta: f64,
    /// `true` for a conv layer.
    pub conv: Vec<bool>,
    pub projector_hidden: usize,
    pub projector_eps: f32,
}

impl VisionConfig {
    /// From the root `config.json` (the projector's settings are beside `vision_config`).
    pub fn from_json(root: &serde_json::Value) -> Result<Self> {
        let v = &root["vision_config"];
        let int = |k: &str| -> Result<usize> {
            v.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .with_context(|| format!("vision_config: missing {k}"))
        };
        let types: Vec<String> = serde_json::from_value(v["layer_types"].clone()).context("vision layer_types")?;
        if v.get("conv_kernel_size").and_then(|x| x.as_u64()).unwrap_or(3) != 3 {
            bail!("LFM3 vision tower: only 3x3 convolutions are supported");
        }
        if v["rope_parameters"]["rope_type"].as_str().is_some_and(|t| t != "axial") {
            bail!("LFM3 vision tower: only axial RoPE is supported");
        }
        if root
            .get("projector_hidden_act")
            .and_then(|x| x.as_str())
            .is_some_and(|a| a != "gelu")
        {
            bail!(
                "LFM3 vision projector: activation {:?} is not supported",
                root["projector_hidden_act"]
            );
        }
        if root.get("projector_bias").and_then(|x| x.as_bool()) == Some(false) {
            bail!("LFM3 vision projector without biases is not supported");
        }
        let patch = int("patch_size")?;
        let merge = int("spatial_merge_size")?;
        if root
            .get("downsample_factor")
            .and_then(|x| x.as_u64())
            .is_some_and(|d| d as usize != merge)
        {
            bail!("LFM3 vision projector: downsample_factor differs from the tower's merge size");
        }
        Ok(Self {
            hidden: int("hidden_size")?,
            inter: int("intermediate_size")?,
            heads: int("num_attention_heads")?,
            kv_heads: int("num_key_value_heads")?,
            patch_dim: int("in_channels")? * int("temporal_patch_size")? * patch * patch,
            merge,
            eps: v.get("norm_eps").and_then(|x| x.as_f64()).unwrap_or(1e-5) as f32,
            rope_theta: v["rope_parameters"]["rope_theta"].as_f64().unwrap_or(1_000_000.0),
            conv: StackConfig::conv_flags(&types)?,
            projector_hidden: root
                .get("projector_hidden_size")
                .and_then(|x| x.as_u64())
                .context("missing projector_hidden_size")? as usize,
            projector_eps: root.get("projector_layer_norm_eps").and_then(|x| x.as_f64()).unwrap_or(1e-6) as f32,
        })
    }
}

#[allow(clippy::large_enum_variant)]
enum Op<'a> {
    Conv {
        in_proj: Lin<'a>,
        out_proj: Lin<'a>,
        /// `[hidden, 9]`.
        w: Vec<f32>,
    },
    Attn {
        q: Lin<'a>,
        k: Lin<'a>,
        v: Lin<'a>,
        o: Lin<'a>,
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
    },
}

struct Layer<'a> {
    op: Op<'a>,
    op_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    w1: Lin<'a>,
    w3: Lin<'a>,
    w2: Lin<'a>,
}

/// The vision tower and the projector.
pub struct VisionTower<'a> {
    pub cfg: VisionConfig,
    patchifier: Lin<'a>,
    embed_norm: Vec<f32>,
    layers: Vec<Layer<'a>>,
    proj_norm: LayerNorm,
    proj_1: VLin<'a>,
    proj_2: VLin<'a>,
    text_hidden: usize,
}

impl<'a> VisionTower<'a> {
    pub fn load(cfg: VisionConfig, files: &'a ModelFiles, text_hidden: usize) -> Result<Self> {
        let p = "model.vision_tower.";
        let h = cfg.hidden;
        let hd = h / cfg.heads;
        let lin = |n: &str, o: usize, i: usize| load_lin(files, n, o, i, Quant::Bf16);
        let vec_of = |n: &str, len: usize| -> Result<Vec<f32>> {
            let v = files.f32(n)?;
            if v.len() != len {
                bail!("tensor {n} has {} values, expected {len}", v.len());
            }
            Ok(v)
        };
        let mut layers = Vec::with_capacity(cfg.conv.len());
        for (l, &conv) in cfg.conv.iter().enumerate() {
            let lp = format!("{p}layers.{l}.");
            let op = if conv {
                Op::Conv {
                    in_proj: lin(&format!("{lp}conv.in_proj.weight"), 3 * h, h)?,
                    out_proj: lin(&format!("{lp}conv.out_proj.weight"), h, h)?,
                    w: vec_of(&format!("{lp}conv.spatial_conv.weight"), h * 9)?,
                }
            } else {
                let ap = format!("{lp}self_attn.");
                Op::Attn {
                    q: lin(&format!("{ap}q_proj.weight"), h, h)?,
                    k: lin(&format!("{ap}k_proj.weight"), cfg.kv_heads * hd, h)?,
                    v: lin(&format!("{ap}v_proj.weight"), cfg.kv_heads * hd, h)?,
                    o: lin(&format!("{ap}out_proj.weight"), h, h)?,
                    q_norm: vec_of(&format!("{ap}q_layernorm.weight"), hd)?,
                    k_norm: vec_of(&format!("{ap}k_layernorm.weight"), hd)?,
                }
            };
            layers.push(Layer {
                op,
                op_norm: vec_of(&format!("{lp}operator_norm.weight"), h)?,
                ffn_norm: vec_of(&format!("{lp}ffn_norm.weight"), h)?,
                w1: lin(&format!("{lp}feed_forward.w1.weight"), cfg.inter, h)?,
                w3: lin(&format!("{lp}feed_forward.w3.weight"), cfg.inter, h)?,
                w2: lin(&format!("{lp}feed_forward.w2.weight"), h, cfg.inter)?,
            });
        }
        let mm = cfg.merge * cfg.merge;
        let ph = cfg.projector_hidden;
        Ok(Self {
            patchifier: lin(&format!("{p}patchifier.weight"), h, cfg.patch_dim)?,
            embed_norm: vec_of(&format!("{p}embedding_norm.weight"), h)?,
            layers,
            proj_norm: LayerNorm::load(files, "model.multi_modal_projector.layer_norm", h)?,
            proj_1: VLin::load(files, "model.multi_modal_projector.linear_1", (ph, h * mm), (ph, h * mm))?,
            proj_2: VLin::load(files, "model.multi_modal_projector.linear_2", (text_hidden, ph), (text_hidden, ph))?,
            text_hidden,
            cfg,
        })
    }

    /// Row and column of patch `k` (merge-block order) in a grid `gw` patches wide.
    fn patch_rc(&self, k: usize, gw: usize) -> (usize, usize) {
        let m = self.cfg.merge;
        let (blk, inb) = (k / (m * m), k % (m * m));
        let bw = gw / m;
        ((blk / bw) * m + inb / m, (blk % bw) * m + inb % m)
    }

    /// The tower's output `[patches, hidden]` (merge-block order) for one image.
    pub fn features(&self, px: &Pixels) -> Vec<f32> {
        let c = &self.cfg;
        let (gt, gh, gw) = px.grid;
        assert_eq!(gt, 1, "the LFM3 vision tower takes one frame at a time");
        let n = gh * gw;
        let (h, nh, nkv) = (c.hidden, c.heads, c.kv_heads);
        let hd = h / nh;
        assert_eq!(px.data.len(), n * c.patch_dim, "pixel buffer does not match the grid");
        let rc: Vec<(usize, usize)> = (0..n).map(|k| self.patch_rc(k, gw)).collect();

        let mut x = vec![0f32; n * h];
        linear(&self.patchifier, h, c.patch_dim, &px.data, n, &mut x);
        rmsnorm_f32w_rows_inplace(&mut x, h, &self.embed_norm, c.eps);

        // Axial RoPE: the first quarter of the head rotates with the row, the second with the
        // column (and again for the second half).
        let q4 = hd / 4;
        let inv_freq: Vec<f32> = (0..q4)
            .map(|i| (1.0 / c.rope_theta.powf((2 * i) as f64 / (hd / 2) as f64)) as f32)
            .collect();
        let half = hd / 2;
        let mut cos = vec![0f32; n * half];
        let mut sin = vec![0f32; n * half];
        for (k, &(r, col)) in rc.iter().enumerate() {
            for (j, &f) in inv_freq.iter().enumerate() {
                let (a, b) = (r as f32 * f, col as f32 * f);
                cos[k * half + j] = a.cos();
                sin[k * half + j] = a.sin();
                cos[k * half + q4 + j] = b.cos();
                sin[k * half + q4 + j] = b.sin();
            }
        }

        let mut hn = vec![0f32; n * h];
        let mut y = vec![0f32; n * h];
        let mut a = vec![0f32; n * c.inter];
        let mut b = vec![0f32; n * c.inter];
        let rep = nh / nkv;
        for layer in &self.layers {
            rmsnorm_f32w_rows(&x, h, &layer.op_norm, c.eps, &mut hn);
            match &layer.op {
                Op::Conv { in_proj, out_proj, w } => {
                    let mut bcx = vec![0f32; n * 3 * h];
                    linear(in_proj, 3 * h, h, &hn, n, &mut bcx);
                    // B * x on the raster grid.
                    let mut grid = vec![0f32; n * h];
                    for (k, &(r, col)) in rc.iter().enumerate() {
                        let src = &bcx[k * 3 * h..(k + 1) * 3 * h];
                        let dst = &mut grid[(r * gw + col) * h..(r * gw + col + 1) * h];
                        for (d, o) in dst.iter_mut().enumerate() {
                            *o = src[d] * src[2 * h + d];
                        }
                    }
                    let mut gated = vec![0f32; n * h];
                    for_chunks(&mut gated, h, true, |k, row| {
                        let (r, col) = rc[k];
                        let cg = &bcx[k * 3 * h + h..k * 3 * h + 2 * h];
                        for ky in 0..3 {
                            let yy = (r + ky) as isize - 1;
                            if yy < 0 || yy >= gh as isize {
                                continue;
                            }
                            for kx in 0..3 {
                                let xx = (col + kx) as isize - 1;
                                if xx < 0 || xx >= gw as isize {
                                    continue;
                                }
                                let src = &grid[(yy as usize * gw + xx as usize) * h..][..h];
                                let kk = ky * 3 + kx;
                                for (d, o) in row.iter_mut().enumerate() {
                                    *o += w[d * 9 + kk] * src[d];
                                }
                            }
                        }
                        for (o, &g) in row.iter_mut().zip(cg) {
                            *o *= g;
                        }
                    });
                    linear(out_proj, h, h, &gated, n, &mut y);
                }
                Op::Attn {
                    q,
                    k,
                    v,
                    o,
                    q_norm,
                    k_norm,
                } => {
                    let mut qs = vec![0f32; n * h];
                    let mut ks = vec![0f32; n * nkv * hd];
                    let mut vs = vec![0f32; n * nkv * hd];
                    linear(q, h, h, &hn, n, &mut qs);
                    linear(k, nkv * hd, h, &hn, n, &mut ks);
                    linear(v, nkv * hd, h, &hn, n, &mut vs);
                    for (buf, heads, w) in [(&mut qs, nh, q_norm), (&mut ks, nkv, k_norm)] {
                        for_chunks(buf, hd, true, |i, head| {
                            rmsnorm_f32w_inplace(head, w, c.eps);
                            let p = i / heads;
                            apply_rope(head, &cos[p * half..(p + 1) * half], &sin[p * half..(p + 1) * half]);
                        });
                    }
                    // Each key / value head serves `rep` query heads.
                    let expand = |src: &[f32]| -> Vec<f32> {
                        let mut out = vec![0f32; n * h];
                        for_chunks(&mut out, hd, true, |i, head| {
                            let (p, hh) = (i / nh, i % nh);
                            head.copy_from_slice(&src[(p * nkv + hh / rep) * hd..][..hd]);
                        });
                        out
                    };
                    let att = attention_chunked(&qs, &expand(&ks), &expand(&vs), n, nh, hd);
                    linear(o, h, h, &att, n, &mut y);
                }
            }
            add_inplace(&mut x, &y);
            rmsnorm_f32w_rows(&x, h, &layer.ffn_norm, c.eps, &mut hn);
            linear(&layer.w1, c.inter, h, &hn, n, &mut a);
            linear(&layer.w3, c.inter, h, &hn, n, &mut b);
            silu_mul(&mut a, &b);
            linear(&layer.w2, h, c.inter, &a, n, &mut y);
            add_inplace(&mut x, &y);
        }
        x
    }

    /// Soft tokens `[patches / merge^2, text_hidden]` of one image.
    pub fn encode(&self, px: &Pixels) -> Vec<f32> {
        let c = &self.cfg;
        let mut x = self.features(px);
        self.proj_norm.apply_rows(&mut x, c.projector_eps);
        let mm = c.merge * c.merge;
        let tokens = x.len() / (c.hidden * mm);
        let ph = c.projector_hidden;
        let mut a = vec![0f32; tokens * ph];
        self.proj_1.forward(&x, tokens, &mut a);
        for_chunks(&mut a, ph, true, |_, r| r.iter_mut().for_each(|v| *v = gelu_erf(*v)));
        let mut out = vec![0f32; tokens * self.text_hidden];
        self.proj_2.forward(&a, tokens, &mut out);
        out
    }
}
