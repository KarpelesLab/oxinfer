//! MiniMax-H3's diffusion transformer (the H3 DiT, `MiniMaxH3DiTModel`; diffusers'
//! `MiniMaxH3ImageTransformer2DModel` / `MiniMaxH3Transformer3DModel`): one packed sequence of
//! rows (text, then latent patches) through full self-attention blocks, no cross-attention.
//!
//! - text: `context_embedder` then a token refiner (plain pre-norm blocks, SwiGLU, no RoPE) and its
//!   final RMSNorm;
//! - latent patches (`[C, pt, ph, pw]` per row): `proj_in` (float32 in the release);
//! - every block: RMSNorm, AdaLN shift / scale, attention (per-head RMSNorm on queries and keys, a
//!   3-axis RoPE: one `inv_freq` of `rope_freq_dim` shared by the `(t, h, w)` axes, `[t | h | w]`
//!   angles rotating the first `2 * 3 * rope_freq_dim` channels half against half), gated residual;
//!   then the same around a SwiGLU feed-forward. The six AdaLN vectors come from one projection of
//!   the timestep embedding with one set per `(timestep, modality)`: a row selects
//!   `timestep_index * 3 + tag` (tag 0 image / video, 1 text, 2 audio);
//! - output: RMSNorm with a per-timestep shift / scale, then the output head (H3-Image: one per
//!   sampling step, the "PDD" heads of `pdd_fused_heads.safetensors`).
//!
//! MiniMax-H3 (video) adds an audio stream to the same stack: `audio_proj_in` / `audio_proj_out`
//! over audio latent rows (tag 2), rows of every modality scattered through one sequence
//! ([`Dit::forward_av`]).
//!
//! Loads the release's layout (`blocks.N.attn.qkv_proj` interleaved per head as `[q k v]`,
//! `mlp.fc1` as `[gate; value]`, `final_layer.*`) as stored, or diffusers' (`transformer_blocks.N`,
//! `to_q` / `to_k` / `to_v`, `ff.net.0.proj` as `[value; gate]`), or ComfyUI's single files
//! ([`DitConfig::from_files`]): the release's tensor names with `qkv_proj` as `[q; k; v]`
//! (`comfy/ldm/minimax/model.py`: `self.qkv_proj(x).split(heads * head_dim, dim=-1)`; the same
//! 535 names as the release, so only where the file comes from tells the two apart), and
//!
//! - "pruned" files, which drop the time embedder and the `[6 H * 3, time_dim]` AdaLN weights (26
//!   of the 66 GB) for their curve form (`adaln_curve_grid` in `comfy/model_detection.py`,
//!   `MiniMaxH3Model._time_embedding`): `adaln_t_table` `[grid, k]` holds the coordinates of
//!   `silu(time_embedder(t))` at `t = i / (grid - 1)` in a basis of `k` directions (1025 x 8 in
//!   the release: the curve's principal components, its mean folded into the biases), a timestep
//!   reads it by linear interpolation, and each `adaln_proj.linear` is `[.., k]` (float16, used
//!   in float32) applied to those coordinates without the SiLU;
//! - FP8 "scaled" weights (`float8_e4m3fn` times the scalar `<module>.weight_scale`, ComfyUI's
//!   `QUANTIZATION.md`), used as stored. `input_scale` (a calibrated activation scale for
//!   ComfyUI's FP8 kernels) and `comfy_quant` are not read: the product runs in float32 on the
//!   decoded weights, ComfyUI's `full_precision_matrix_mult`;
//! - `int8_tensorwise` weights with ConvRot (`*_int8_convrot.safetensors`: int8 rows with a scale
//!   per row, stored rotated by a grouped regular Hadamard matrix; used as stored, the input
//!   rotated first) and `w6a8_int8` ones (`*_w6a8.safetensors`: 6-bit codes, decoded to int8 at
//!   load), read by [`crate::comfy_quant`] from each module's `comfy_quant` entry. Float32
//!   products too: ComfyUI's int8 kernels also round the rotated input to int8, which is not done.

use crate::kernels::*;
use crate::model::{Lin, Quant, linear, load_lin};
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};
use std::path::Path;

/// The modalities a row can carry (AdaLN sets per timestep).
pub const MODALITIES: usize = 3;
pub const TAG_IMAGE: u8 = 0;
pub const TAG_TEXT: u8 = 1;
pub const TAG_AUDIO: u8 = 2;

#[derive(Debug, Clone)]
pub struct DitConfig {
    pub hidden: usize,
    pub layers: usize,
    pub refiner_layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub in_channels: usize,
    /// Channels of an audio latent row (MiniMax-H3; 0 for H3-Image).
    pub audio_channels: usize,
    pub patch: [usize; 3],
    pub text_dim: usize,
    pub freq_dim: usize,
    pub time_hidden: usize,
    pub time_dim: usize,
    pub rope_freq_dim: usize,
    pub rope_theta: f64,
    pub norm_eps: f32,
    pub qk_eps: f32,
    pub final_eps: f32,
    /// `qkv_proj` rows are `[q; k; v]` (ComfyUI's single files) instead of `[q k v]` per head
    /// (the release). Diffusers' layout has three matrices and ignores it.
    pub qkv_stacked: bool,
}

impl DitConfig {
    /// `config.json` of either layout (the release's `MiniMaxH3DiTModel` keys or diffusers').
    pub fn load(dir: &Path) -> Result<Self> {
        let c: serde_json::Value = serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?)?;
        let u = |keys: &[&str], d: Option<usize>| -> Result<usize> {
            keys.iter()
                .find_map(|k| c.get(*k).and_then(|v| v.as_u64()))
                .map(|v| v as usize)
                .or(d)
                .with_context(|| format!("DiT config has none of {keys:?}"))
        };
        let f = |keys: &[&str], d: f64| keys.iter().find_map(|k| c.get(*k).and_then(|v| v.as_f64())).unwrap_or(d);
        let patch: Vec<usize> = c
            .get("patch_size")
            .and_then(|p| p.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_u64().map(|v| v as usize)).collect())
            .unwrap_or_else(|| vec![1, 2, 2]);
        if patch.len() != 3 {
            bail!("DiT patch_size must have 3 entries");
        }
        Ok(Self {
            hidden: u(&["hidden_size"], None)?,
            layers: u(&["num_layers"], None)?,
            refiner_layers: u(&["num_refiner_layers", "token_refiner_num_layers"], Some(2))?,
            heads: u(&["num_attention_heads"], None)?,
            head_dim: u(&["attention_head_dim"], None)?,
            ffn: u(&["ffn_dim", "ffn_hidden_size"], None)?,
            in_channels: u(&["in_channels", "latents_dim"], None)?,
            audio_channels: u(&["audio_in_channels", "audio_latents_dim"], Some(0))?,
            patch: [patch[0], patch[1], patch[2]],
            text_dim: u(&["text_dim"], None)?,
            freq_dim: u(&["freq_dim", "timestep_input_dim"], Some(256))?,
            time_hidden: u(&["time_embed_hidden_dim", "time_embed_hidden_size"], None)?,
            time_dim: u(&["time_embed_dim"], None)?,
            rope_freq_dim: u(&["rope_freq_dim", "rope_inv_freq_len"], Some(16))?,
            rope_theta: f(&["rope_theta"], 10000.0),
            norm_eps: f(&["norm_eps"], 1e-5) as f32,
            qk_eps: f(&["qk_norm_eps"], 1e-5) as f32,
            final_eps: f(&["final_norm_eps"], 1e-5) as f32,
            qkv_stacked: false,
        })
    }

    /// The configuration of a ComfyUI single-file checkpoint, from its tensor shapes
    /// (`comfy/model_detection.py`, the MiniMax H3 branch) and, where the file has one, the
    /// `config` entry of its header (`{"transformer": {...}}` with the release's keys).
    pub fn from_files(files: &ModelFiles) -> Result<Self> {
        let dim = |n: &str, i: usize| -> Result<usize> {
            files
                .info(n)
                .with_context(|| format!("missing tensor {n}"))?
                .shape
                .get(i)
                .copied()
                .with_context(|| format!("tensor {n} has no dimension {i}"))
        };
        let count = |prefix: &str| (0..).take_while(|i| files.has(&format!("{prefix}{i}.norm1.weight"))).count();
        let meta: serde_json::Value = files
            .metadata("config")
            .and_then(|m| serde_json::from_str::<serde_json::Value>(m).ok())
            .and_then(|v| v.get("transformer").cloned())
            .unwrap_or(serde_json::Value::Null);
        if meta.get("pdd_steps").and_then(|v| v.as_u64()).unwrap_or(1) > 1 {
            bail!("a checkpoint with stacked PDD output heads (MiniMax-H3-Image) is not read from a single file");
        }
        let eps = |k: &str| meta.get(k).and_then(|v| v.as_f64()).unwrap_or(1e-5) as f32;
        let head_dim = dim("blocks.0.attn.q_norm.weight", 0)?;
        // The curve form has no time embedder: the AdaLN projections read `k` coordinates.
        let (freq_dim, time_hidden, time_dim) = if files.has("adaln_t_table") {
            (0, 0, dim("adaln_t_table", 1)?)
        } else {
            (
                dim("time_embedder.proj_in.weight", 1)?,
                dim("time_embedder.proj_in.weight", 0)?,
                dim("time_embedder.proj_out.weight", 0)?,
            )
        };
        let patch = [1, 2, 2];
        Ok(Self {
            hidden: dim("video_patch_proj.weight", 0)?,
            layers: count("blocks."),
            refiner_layers: count("token_refiner.blocks."),
            heads: dim("blocks.0.attn.qkv_proj.weight", 0)? / (3 * head_dim),
            head_dim,
            ffn: dim("blocks.0.mlp.fc1.weight", 0)? / 2,
            in_channels: dim("final_layer.video_out.weight", 0)? / patch.iter().product::<usize>(),
            audio_channels: if files.has("audio_patch_proj.weight") {
                dim("final_layer.audio_out.weight", 0)?
            } else {
                0
            },
            patch,
            text_dim: dim("condition_proj.weight", 1)?,
            freq_dim,
            time_hidden,
            time_dim,
            rope_freq_dim: dim("rope.inv_freq", 0)?,
            rope_theta: 10000.0,
            norm_eps: eps("norm_eps"),
            qk_eps: eps("qk_norm_eps"),
            final_eps: eps("final_norm_eps"),
            qkv_stacked: true,
        })
    }
    pub fn inner(&self) -> usize {
        self.heads * self.head_dim
    }
    pub fn patch_dim(&self) -> usize {
        self.in_channels * self.patch.iter().product::<usize>()
    }
}

/// A linear layer with an optional bias, float32 weights kept in float32 (the release keeps its
/// patch, timestep and output layers so), float16 ones in float16 (ComfyUI's fp16 VAE and curve
/// AdaLN projections), FP8 "scaled" ones as stored, others bf16 (or as stored).
pub(crate) struct DLin<'a> {
    pub(crate) w: Lin<'a>,
    pub(crate) b: Option<Vec<f32>>,
    pub(crate) n_out: usize,
    pub(crate) n_in: usize,
}

impl<'a> DLin<'a> {
    pub(crate) fn load(files: &'a ModelFiles, name: &str, n_out: usize, n_in: usize, bias: bool) -> Result<Self> {
        let wn = format!("{name}.weight");
        let info = files.info(&wn).with_context(|| format!("missing tensor {wn}"))?;
        let w = match info.dtype.as_str() {
            "F32" => f32_lin(&files.f32(&wn)?, n_out, n_in)?,
            "F16" if info.shape == [n_out, n_in] => {
                Lin::from_gguf(crate::ggml::GgmlType::F16, files.bytes(&wn)?, n_out, n_in).with_context(|| format!("tensor {wn}"))?
            }
            // ComfyUI's scalar `weight_scale` (DeepSeek-style block scales go through `load_lin`).
            "F8_E4M3" if !files.has(&format!("{name}.weight_scale_inv")) => crate::model::fp8_scaled(files, &wn, n_out, n_in)?,
            _ => load_lin(files, &wn, n_out, n_in, Quant::Bf16)?,
        };
        let b = if bias {
            let b = files.f32(&format!("{name}.bias"))?;
            if b.len() != n_out {
                bail!("{name}.bias: {} values, expected {n_out}", b.len());
            }
            Some(b)
        } else {
            None
        };
        Ok(Self { w, b, n_out, n_in })
    }
    pub(crate) fn from_f32(w: &[f32], b: Option<Vec<f32>>, n_out: usize, n_in: usize) -> Result<Self> {
        Ok(Self {
            w: f32_lin(w, n_out, n_in)?,
            b,
            n_out,
            n_in,
        })
    }
    pub(crate) fn forward(&self, x: &[f32], t: usize) -> Vec<f32> {
        let mut y = vec![0f32; t * self.n_out];
        linear(&self.w, self.n_out, self.n_in, x, t, &mut y);
        if let Some(b) = &self.b {
            for_chunks(&mut y, self.n_out, t > 1, |_, r| r.iter_mut().zip(b).for_each(|(v, &bv)| *v += bv));
        }
        y
    }
}

/// A float32 matrix as a linear layer (exact: its rows are dequantized as stored).
pub(crate) fn f32_lin(w: &[f32], n_out: usize, n_in: usize) -> Result<Lin<'static>> {
    if w.len() != n_out * n_in {
        bail!("f32 weight of {} values for [{n_out}, {n_in}]", w.len());
    }
    let mut pv = crate::quant::PageVec::<u8>::zeroed(w.len() * 4);
    for (d, v) in pv.as_chunks_mut::<4>().0.iter_mut().zip(w) {
        d.copy_from_slice(&v.to_le_bytes());
    }
    Ok(Lin::gguf(crate::ggml::GgufMatrix::new(
        crate::ggml::GgmlType::F32,
        pv.into(),
        n_out,
        n_in,
    )?))
}

enum Qkv<'a> {
    /// `to_q`, `to_k`, `to_v`.
    Split(Box<[DLin<'a>; 3]>),
    /// One projection whose rows are `[q k v]` per head (the release's `qkv_proj`).
    Interleaved(Box<DLin<'a>>),
    /// One projection whose rows are `[q; k; v]` (ComfyUI's `qkv_proj`).
    Stacked(Box<DLin<'a>>),
}

struct Attn<'a> {
    qkv: Qkv<'a>,
    norm_q: Vec<f32>,
    norm_k: Vec<f32>,
    out: DLin<'a>,
}

struct Ff<'a> {
    /// `[2 * ffn, H]`, gate rows first (`gate_first`) or value rows first.
    fc1: DLin<'a>,
    gate_first: bool,
    fc2: DLin<'a>,
}

struct Block<'a> {
    norm1: Vec<f32>,
    attn: Attn<'a>,
    norm2: Vec<f32>,
    ff: Ff<'a>,
    /// `[6 * H * 3, time_dim]` and its bias: per modality the six AdaLN vectors.
    adaln: Option<DLin<'a>>,
}

/// The DiT (see the module docs).
pub struct Dit<'a> {
    pub cfg: DitConfig,
    proj_in: DLin<'a>,
    context: DLin<'a>,
    /// The time embedder's two layers; none in the curve form.
    time: Option<(DLin<'a>, DLin<'a>)>,
    /// The curve form's `adaln_t_table` (`[grid, time_dim]`, see the module docs).
    curve: Option<Vec<f32>>,
    refiner: Vec<Block<'a>>,
    refiner_norm: Vec<f32>,
    blocks: Vec<Block<'a>>,
    norm_out: Vec<f32>,
    norm_out_ada: DLin<'a>,
    /// The output head of each sampling step (H3-Image), or the one head.
    heads_out: Vec<DLin<'a>>,
    /// MiniMax-H3's audio projections (in, out).
    audio: Option<(DLin<'a>, DLin<'a>)>,
    inv_freq: Vec<f32>,
}

/// One forward's inputs.
pub struct Packed<'p> {
    /// Text conditioning `[n_text, text_dim]`, rows `0..n_text` of the sequence.
    pub text: &'p [f32],
    /// Latent patch rows `[n_img, patch_dim]`, rows `n_text..` of the sequence.
    pub image: &'p [f32],
    /// `(t, h, w)` rotary coordinates of every row.
    pub positions: &'p [[f32; 3]],
    /// The distinct timesteps (`1 - sigma`), and each row's index into them.
    pub timesteps: &'p [f32],
    pub timestep_index: &'p [usize],
    /// Each row's modality tag.
    pub tags: &'p [u8],
}

impl<'a> Dit<'a> {
    /// Loads the DiT from `files` (`transformer/`), with the per-step output heads of `heads`
    /// (`pdd_fused_heads.safetensors`: `video_out.weight` `[steps, out, H]`, `video_out.bias`) when
    /// given; diffusers' `pdd_proj_out.N` otherwise, else the single head.
    pub fn load(cfg: DitConfig, files: &'a ModelFiles, heads: Option<&ModelFiles>) -> Result<Self> {
        let original = files.has("blocks.0.attn.qkv_proj.weight");
        let (h, inner, ffn, td) = (cfg.hidden, cfg.inner(), cfg.ffn, cfg.time_dim);
        let hd = cfg.head_dim;
        let norm = |n: &str, len: usize| -> Result<Vec<f32>> {
            let v = files.f32(n)?;
            if v.len() != len {
                bail!("{n}: {} values, expected {len}", v.len());
            }
            Ok(v)
        };
        let block = |p: &str, adaln: bool| -> Result<Block<'a>> {
            Ok(if original {
                Block {
                    norm1: norm(&format!("{p}norm1.weight"), h)?,
                    attn: Attn {
                        qkv: {
                            let w = Box::new(DLin::load(files, &format!("{p}attn.qkv_proj"), 3 * inner, h, false)?);
                            if cfg.qkv_stacked { Qkv::Stacked(w) } else { Qkv::Interleaved(w) }
                        },
                        norm_q: norm(&format!("{p}attn.q_norm.weight"), hd)?,
                        norm_k: norm(&format!("{p}attn.k_norm.weight"), hd)?,
                        out: DLin::load(files, &format!("{p}attn.out_proj"), h, inner, false)?,
                    },
                    norm2: norm(&format!("{p}norm2.weight"), h)?,
                    ff: Ff {
                        fc1: DLin::load(files, &format!("{p}mlp.fc1"), 2 * ffn, h, false)?,
                        gate_first: true,
                        fc2: DLin::load(files, &format!("{p}mlp.fc2"), h, ffn, false)?,
                    },
                    adaln: adaln
                        .then(|| DLin::load(files, &format!("{p}adaln_proj.linear"), 6 * h * MODALITIES, td, true))
                        .transpose()?,
                }
            } else {
                Block {
                    norm1: norm(&format!("{p}norm1.weight"), h)?,
                    attn: Attn {
                        qkv: Qkv::Split(Box::new([
                            DLin::load(files, &format!("{p}attn.to_q"), inner, h, false)?,
                            DLin::load(files, &format!("{p}attn.to_k"), inner, h, false)?,
                            DLin::load(files, &format!("{p}attn.to_v"), inner, h, false)?,
                        ])),
                        norm_q: norm(&format!("{p}attn.norm_q.weight"), hd)?,
                        norm_k: norm(&format!("{p}attn.norm_k.weight"), hd)?,
                        out: DLin::load(files, &format!("{p}attn.to_out.0"), h, inner, false)?,
                    },
                    norm2: norm(&format!("{p}norm2.weight"), h)?,
                    ff: Ff {
                        fc1: DLin::load(files, &format!("{p}ff.net.0.proj"), 2 * ffn, h, false)?,
                        gate_first: false,
                        fc2: DLin::load(files, &format!("{p}ff.net.2"), h, ffn, false)?,
                    },
                    adaln: adaln
                        .then(|| DLin::load(files, &format!("{p}adaln_proj.linear"), 6 * h * MODALITIES, td, true))
                        .transpose()?,
                }
            })
        };
        let (rp, bp) = if original {
            ("token_refiner.blocks.", "blocks.")
        } else {
            ("token_refiner.refiner_blocks.", "transformer_blocks.")
        };
        let refiner = (0..cfg.refiner_layers)
            .map(|l| block(&format!("{rp}{l}."), false))
            .collect::<Result<Vec<_>>>()?;
        let blocks = (0..cfg.layers)
            .map(|l| block(&format!("{bp}{l}."), true))
            .collect::<Result<Vec<_>>>()?;
        let pd = cfg.patch_dim();
        let n = |orig: &str, diff: &str| if original { orig.to_string() } else { diff.to_string() };
        let heads_out = match heads {
            Some(hf) => {
                let w = hf.f32("video_out.weight")?;
                let b = hf.f32("video_out.bias")?;
                let steps = b.len() / pd;
                if steps == 0 || w.len() != steps * pd * h || b.len() != steps * pd {
                    bail!("PDD heads: unexpected sizes");
                }
                (0..steps)
                    .map(|s| DLin::from_f32(&w[s * pd * h..(s + 1) * pd * h], Some(b[s * pd..(s + 1) * pd].to_vec()), pd, h))
                    .collect::<Result<Vec<_>>>()?
            }
            None if files.has("pdd_proj_out.0.weight") => (0..)
                .take_while(|s| files.has(&format!("pdd_proj_out.{s}.weight")))
                .map(|s| DLin::load(files, &format!("pdd_proj_out.{s}"), pd, h, true))
                .collect::<Result<Vec<_>>>()?,
            None => vec![DLin::load(files, &n("final_layer.video_out", "proj_out"), pd, h, true)?],
        };
        let ain = n("audio_patch_proj", "audio_proj_in");
        let audio = if cfg.audio_channels > 0 && files.has(&format!("{ain}.weight")) {
            Some((
                DLin::load(files, &ain, h, cfg.audio_channels, true)?,
                DLin::load(files, &n("final_layer.audio_out", "audio_proj_out"), cfg.audio_channels, h, true)?,
            ))
        } else {
            None
        };
        let rd = cfg.rope_freq_dim;
        let inv_freq: Vec<f32> = (0..rd)
            .map(|i| (1.0 / cfg.rope_theta.powf((2 * i) as f64 / (2 * rd) as f64)) as f32)
            .collect();
        if 6 * rd > hd {
            bail!("DiT heads of {hd} are too narrow for the 3-axis RoPE");
        }
        // ComfyUI's files store the frequencies: they must be the ones computed here.
        if files.has("rope.inv_freq") {
            let stored = files.f32("rope.inv_freq")?;
            if stored.len() != rd || stored.iter().zip(&inv_freq).any(|(a, b)| (a - b).abs() > 1e-5 * b.abs()) {
                bail!("rope.inv_freq is not 1 / {} ^ (i / {rd})", cfg.rope_theta);
            }
        }
        let curve = if files.has("adaln_t_table") {
            let t = files.f32("adaln_t_table")?;
            if t.len() < 2 * td || !t.len().is_multiple_of(td) {
                bail!("adaln_t_table: {} values for rows of {td}", t.len());
            }
            Some(t)
        } else {
            None
        };
        let time = if curve.is_some() {
            None
        } else {
            Some((
                DLin::load(
                    files,
                    &n("time_embedder.proj_in", "time_embedder.linear_1"),
                    cfg.time_hidden,
                    cfg.freq_dim,
                    true,
                )?,
                DLin::load(
                    files,
                    &n("time_embedder.proj_out", "time_embedder.linear_2"),
                    td,
                    cfg.time_hidden,
                    true,
                )?,
            ))
        };
        Ok(Self {
            proj_in: DLin::load(files, &n("video_patch_proj", "proj_in"), h, pd, true)?,
            context: DLin::load(files, &n("condition_proj", "context_embedder"), h, cfg.text_dim, true)?,
            time,
            curve,
            refiner,
            refiner_norm: norm("token_refiner.final_norm.weight", h)?,
            blocks,
            norm_out: norm(&n("final_layer.norm.weight", "norm_out.norm.weight"), h)?,
            norm_out_ada: DLin::load(files, &n("final_layer.adaln_proj.linear", "norm_out.linear"), 2 * h, td, true)?,
            heads_out,
            audio,
            inv_freq,
            cfg,
        })
    }

    /// The number of output heads (sampling steps of a PDD release; 1 otherwise).
    pub fn n_heads(&self) -> usize {
        self.heads_out.len()
    }

    /// The refined text stream `[n, H]` (independent of the timestep: computed once).
    pub fn embed_text(&self, text: &[f32]) -> Vec<f32> {
        let h = self.cfg.hidden;
        let n = text.len() / self.cfg.text_dim;
        let mut x = self.context.forward(text, n);
        for b in &self.refiner {
            self.block(b, &mut x, n, None, None);
        }
        rmsnorm_f32w_rows_inplace(&mut x, h, &self.refiner_norm, self.cfg.final_eps);
        x
    }

    /// The timestep embedding (`[time_dim]`) of a `t` in `[0, 1]`; in the curve form the row of
    /// `adaln_t_table` at `t` (linear interpolation, `t` clamped to `[0, 1]`).
    pub fn time_embed(&self, t: f32) -> Vec<f32> {
        if let Some(table) = &self.curve {
            let td = self.cfg.time_dim;
            let grid = table.len() / td;
            let pos = t.clamp(0.0, 1.0) * (grid - 1) as f32;
            let i0 = (pos.floor() as usize).min(grid - 2);
            let w = pos - i0 as f32;
            let (a, b) = (&table[i0 * td..(i0 + 1) * td], &table[(i0 + 1) * td..(i0 + 2) * td]);
            return a.iter().zip(b).map(|(&x, &y)| x + w * (y - x)).collect();
        }
        let (time1, time2) = self.time.as_ref().expect("a DiT has a time embedder or a curve");
        let e = crate::sd::ops::timestep_embedding(t, self.cfg.freq_dim, true, 0.0);
        let mut a = time1.forward(&e, 1);
        a.iter_mut().for_each(|v| *v = crate::sd::ops::silu(*v));
        time2.forward(&a, 1)
    }

    /// What the AdaLN projections read of timestep embeddings ([`Dit::time_embed`]): their SiLU,
    /// or the curve form's coordinates as they are.
    fn time_act(&self, temb: &[f32]) -> Vec<f32> {
        if self.curve.is_some() {
            temb.to_vec()
        } else {
            temb.iter().map(|&v| crate::sd::ops::silu(v)).collect()
        }
    }

    /// RoPE `cos` / `sin` (`3 * rope_freq_dim` each) of every row.
    fn rope(&self, positions: &[[f32; 3]]) -> (Vec<f32>, Vec<f32>) {
        let f = self.inv_freq.len();
        let half = 3 * f;
        let (mut cos, mut sin) = (vec![0f32; positions.len() * half], vec![0f32; positions.len() * half]);
        for (i, p) in positions.iter().enumerate() {
            for (axis, &pv) in p.iter().enumerate() {
                for (j, &fr) in self.inv_freq.iter().enumerate() {
                    let a = pv * fr;
                    cos[i * half + axis * f + j] = a.cos();
                    sin[i * half + axis * f + j] = a.sin();
                }
            }
        }
        (cos, sin)
    }

    /// One block over `x` (`[n, H]`). `ada`: the AdaLN vectors of every row (shift / scale / gate
    /// for attention then feed-forward), none for the refiner; `rope`: per-row cos / sin.
    fn block(&self, b: &Block, x: &mut [f32], n: usize, ada: Option<Ada>, rope: Option<(&[f32], &[f32])>) {
        let cfg = &self.cfg;
        let (h, nh, hd) = (cfg.hidden, cfg.heads, cfg.head_dim);
        let inner = nh * hd;
        let mut hn = x.to_vec();
        rmsnorm_f32w_rows_inplace(&mut hn, h, &b.norm1, cfg.norm_eps);
        if let Some(a) = ada {
            modulate(&mut hn, a, h, 0);
        }
        // q, k, v as [n, inner] each.
        let (mut q, mut k, v) = match &b.attn.qkv {
            Qkv::Split(w) => (w[0].forward(&hn, n), w[1].forward(&hn, n), w[2].forward(&hn, n)),
            Qkv::Interleaved(w) => {
                let f = w.forward(&hn, n);
                let (mut q, mut k, mut v) = (vec![0f32; n * inner], vec![0f32; n * inner], vec![0f32; n * inner]);
                for r in 0..n {
                    for hh in 0..nh {
                        let s = (r * nh + hh) * 3 * hd;
                        let d = r * inner + hh * hd;
                        q[d..d + hd].copy_from_slice(&f[s..s + hd]);
                        k[d..d + hd].copy_from_slice(&f[s + hd..s + 2 * hd]);
                        v[d..d + hd].copy_from_slice(&f[s + 2 * hd..s + 3 * hd]);
                    }
                }
                (q, k, v)
            }
            Qkv::Stacked(w) => {
                let f = w.forward(&hn, n);
                let (mut q, mut k, mut v) = (vec![0f32; n * inner], vec![0f32; n * inner], vec![0f32; n * inner]);
                for r in 0..n {
                    let s = r * 3 * inner;
                    q[r * inner..(r + 1) * inner].copy_from_slice(&f[s..s + inner]);
                    k[r * inner..(r + 1) * inner].copy_from_slice(&f[s + inner..s + 2 * inner]);
                    v[r * inner..(r + 1) * inner].copy_from_slice(&f[s + 2 * inner..s + 3 * inner]);
                }
                (q, k, v)
            }
        };
        let half = 3 * self.inv_freq.len();
        for (buf, w) in [(&mut q, &b.attn.norm_q), (&mut k, &b.attn.norm_k)] {
            for_chunks(buf, hd, true, |i, head| {
                rmsnorm_f32w_inplace(head, w, cfg.qk_eps);
                if let Some((cos, sin)) = rope {
                    let r = i / nh;
                    apply_rope(
                        &mut head[..2 * half],
                        &cos[r * half..(r + 1) * half],
                        &sin[r * half..(r + 1) * half],
                    );
                }
            });
        }
        let att = crate::qwen_vision::attention_chunked(&q, &k, &v, n, nh, hd);
        let y = b.attn.out.forward(&att, n);
        residual(x, &y, ada, h, 2);

        let mut hn = x.to_vec();
        rmsnorm_f32w_rows_inplace(&mut hn, h, &b.norm2, cfg.norm_eps);
        if let Some(a) = ada {
            modulate(&mut hn, a, h, 3);
        }
        let ffn = cfg.ffn;
        let gu = b.ff.fc1.forward(&hn, n);
        let mut act = vec![0f32; n * ffn];
        for_chunks(&mut act, ffn, true, |r, row| {
            let src = &gu[r * 2 * ffn..(r + 1) * 2 * ffn];
            let (g, u) = if b.ff.gate_first {
                (&src[..ffn], &src[ffn..])
            } else {
                (&src[ffn..], &src[..ffn])
            };
            for ((o, &gv), &uv) in row.iter_mut().zip(g).zip(u) {
                *o = uv * gv / (1.0 + (-gv).exp());
            }
        });
        let y = b.ff.fc2.forward(&act, n);
        residual(x, &y, ada, h, 5);
    }

    /// The block stack over the packed sequence `x` (`[n, H]`): `st` holds the SiLU of each distinct
    /// timestep embedding (`[nt, time_dim]`), a row reads the AdaLN set of its `(timestep, tag)`.
    fn stack(&self, x: &mut [f32], st: &[f32], timestep_index: &[usize], tags: &[u8], positions: &[[f32; 3]], dump: Option<&Dump>) {
        let h = self.cfg.hidden;
        let n = x.len() / h;
        let nt = st.len() / self.cfg.time_dim;
        let (cos, sin) = self.rope(positions);
        // Each row's AdaLN set, `(timestep, modality)`: read from the table, not copied out per row
        // (that would be n * 6H floats per block, 4.8 GB at a 1344x768 video's 37k rows).
        let key: Vec<usize> = (0..n).map(|r| timestep_index[r] * MODALITIES + tags[r] as usize).collect();
        let dump = dump.filter(|d| d.blocks);
        if let Some(d) = dump {
            d.write("x_in.bin", x);
        }
        for (bi, b) in self.blocks.iter().enumerate() {
            // [nt * 3, 6H]: the six vectors of each (timestep, modality).
            let table = b.adaln.as_ref().unwrap().forward(st, nt);
            self.block(
                b,
                x,
                n,
                Some(Ada {
                    table: &table,
                    key: &key,
                    h,
                }),
                Some((&cos, &sin)),
            );
            if let Some(d) = dump {
                d.write(&format!("blk_{bi}.bin"), x);
            }
        }
    }

    /// The output norm of the rows `rows` of `x` (RMSNorm, then the shift / scale of each row's
    /// timestep).
    fn norm_out_rows(&self, x: &[f32], rows: &[usize], st: &[f32], timestep_index: &[usize]) -> Vec<f32> {
        let h = self.cfg.hidden;
        let nt = st.len() / self.cfg.time_dim;
        let so = self.norm_out_ada.forward(st, nt);
        let mut hn = vec![0f32; rows.len() * h];
        for_chunks(&mut hn, h, true, |r, row| row.copy_from_slice(&x[rows[r] * h..(rows[r] + 1) * h]));
        rmsnorm_f32w_rows_inplace(&mut hn, h, &self.norm_out, self.cfg.final_eps);
        for_chunks(&mut hn, h, true, |r, row| {
            let k = timestep_index[rows[r]];
            let (shift, scale) = (&so[k * 2 * h..k * 2 * h + h], &so[k * 2 * h + h..(k + 1) * 2 * h]);
            for ((v, &s), &c) in row.iter_mut().zip(scale).zip(shift) {
                *v = *v * (1.0 + s) + c;
            }
        });
        hn
    }

    /// The velocity of the latent rows for one step (`head`: which output head).
    pub fn forward(&self, p: &Packed, text_embedded: &[f32], head: usize) -> Vec<f32> {
        let cfg = &self.cfg;
        let h = cfg.hidden;
        let n_text = text_embedded.len() / h;
        let n_img = p.image.len() / cfg.patch_dim();
        let n = n_text + n_img;
        let mut x = vec![0f32; n * h];
        x[..n_text * h].copy_from_slice(text_embedded);
        x[n_text * h..].copy_from_slice(&self.proj_in.forward(p.image, n_img));
        let temb: Vec<f32> = p.timesteps.iter().flat_map(|&t| self.time_embed(t)).collect();
        let st = self.time_act(&temb);
        self.stack(&mut x, &st, p.timestep_index, p.tags, p.positions, None);
        // Output norm with the per-timestep shift / scale, then the head over the latent rows.
        let rows: Vec<usize> = (n_text..n).collect();
        let hn = self.norm_out_rows(&x, &rows, &st, p.timestep_index);
        self.heads_out[head.min(self.heads_out.len() - 1)].forward(&hn, n_img)
    }

    /// Whether the checkpoint has MiniMax-H3's audio stream.
    pub fn has_audio(&self) -> bool {
        self.audio.is_some()
    }

    /// MiniMax-H3's joint forward: the velocities of the video rows and of the audio rows, in the
    /// order of `video_index` / `audio_index`.
    pub fn forward_av(&self, p: &PackedAv) -> Result<(Vec<f32>, Vec<f32>)> {
        let cfg = &self.cfg;
        let h = cfg.hidden;
        let (ain, aout) = self.audio.as_ref().map(|(a, b)| (a, b)).context("this DiT has no audio stream")?;
        let n = p.positions.len();
        let (nv, na) = (p.video_index.len(), p.audio_index.len());
        let n_text = p.text_embedded.len() / h;
        if p.video.len() != nv * cfg.patch_dim()
            || p.audio.len() != na * cfg.audio_channels
            || n_text != p.text_index.len()
            || n != n_text + nv + na
            || p.tags.len() != n
            || p.timestep_index.len() != n
            || !p.temb.len().is_multiple_of(cfg.time_dim)
        {
            bail!("packed sequence: the rows, indices and positions do not agree");
        }
        let mut x = vec![0f32; n * h];
        for (src, index) in [
            (p.text_embedded, p.text_index),
            (&self.proj_in.forward(p.video, nv)[..], p.video_index),
            (&ain.forward(p.audio, na)[..], p.audio_index),
        ] {
            for (r, &i) in index.iter().enumerate() {
                x[i * h..(i + 1) * h].copy_from_slice(&src[r * h..(r + 1) * h]);
            }
        }
        let st = self.time_act(p.temb);
        let dump = Dump::from_env();
        self.stack(&mut x, &st, p.timestep_index, p.tags, p.positions, dump.as_ref());
        let v = self.heads_out[0].forward(&self.norm_out_rows(&x, p.video_index, &st, p.timestep_index), nv);
        let a = aout.forward(&self.norm_out_rows(&x, p.audio_index, &st, p.timestep_index), na);
        if let Some(d) = &dump {
            let i = d.call;
            d.write(&format!("s{i}_video_in.bin"), p.video);
            d.write(&format!("s{i}_audio_in.bin"), p.audio);
            d.write(&format!("s{i}_v.bin"), &v);
            d.write(&format!("s{i}_a.bin"), &a);
            d.write(&format!("s{i}_temb.bin"), p.temb);
            d.write(
                &format!("s{i}_tidx.bin"),
                &p.timestep_index.iter().map(|&t| t as f32).collect::<Vec<f32>>(),
            );
            if i == 0 {
                d.write("text_emb.bin", p.text_embedded);
                d.write("positions.bin", p.positions.as_flattened());
                d.write("tags.bin", &p.tags.iter().map(|&t| t as f32).collect::<Vec<f32>>());
            }
        }
        Ok((v, a))
    }
}

/// `OXINFER_H3_DUMP=DIR`: what [`Dit::forward_av`] computes, kept as little-endian float32 files in
/// DIR for a comparison with a reference on the same inputs (`tools/minimax_h3_video_ref.py`).
/// Call `i` of the process (a sampling step) writes its video and audio rows (`s{i}_video_in.bin`,
/// `s{i}_audio_in.bin`), its velocities (`s{i}_v.bin`, `s{i}_a.bin`), the timestep embeddings
/// (`s{i}_temb.bin`) and each row's index into them (`s{i}_tidx.bin`); the first call also the
/// refined text rows (`text_emb.bin`), the rotary positions (`positions.bin`) and the modality tags
/// (`tags.bin`). With `OXINFER_H3_DUMP_BLOCKS=i`, call `i` also writes the packed sequence before
/// the blocks (`x_in.bin`) and after each one (`blk_{n}.bin`: the whole sequence, 100 MB per
/// block for 4682 rows).
struct Dump {
    dir: std::path::PathBuf,
    call: usize,
    blocks: bool,
}

impl Dump {
    fn from_env() -> Option<Self> {
        static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let dir = std::path::PathBuf::from(std::env::var_os("OXINFER_H3_DUMP")?);
        let call = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let blocks = std::env::var("OXINFER_H3_DUMP_BLOCKS").is_ok_and(|v| v.parse() == Ok(call));
        Some(Self { dir, call, blocks })
    }

    fn write(&self, name: &str, v: &[f32]) {
        let path = self.dir.join(name);
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        if let Err(e) = std::fs::write(&path, bytes) {
            eprintln!("OXINFER_H3_DUMP: {}: {e}", path.display());
        }
    }
}

/// The inputs of [`Dit::forward_av`]: rows of three modalities scattered through one sequence.
pub struct PackedAv<'p> {
    /// The refined text stream ([`Dit::embed_text`]), `[n_text, H]`.
    pub text_embedded: &'p [f32],
    /// Video latent patch rows `[n_video, patch_dim]` (conditioning rows first).
    pub video: &'p [f32],
    /// Audio latent rows `[n_audio, audio_channels]`.
    pub audio: &'p [f32],
    /// The sequence position of every text / video / audio row.
    pub text_index: &'p [usize],
    pub video_index: &'p [usize],
    pub audio_index: &'p [usize],
    /// `(t, h, w)` rotary coordinates of every row of the sequence.
    pub positions: &'p [[f32; 3]],
    /// The embeddings of the distinct timesteps (`[nt, time_dim]`, [`Dit::time_embed`]) and each
    /// row's index into them.
    pub temb: &'p [f32],
    pub timestep_index: &'p [usize],
    /// Each row's modality tag.
    pub tags: &'p [u8],
}

/// The AdaLN vectors of every row: row `r` reads set `key[r]` of `table` (`[sets, 6H]`).
#[derive(Clone, Copy)]
struct Ada<'t> {
    table: &'t [f32],
    key: &'t [usize],
    h: usize,
}

impl Ada<'_> {
    /// Vector `which` (0..6: shift, scale, gate for attention, then for the feed-forward) of row `r`.
    fn get(&self, r: usize, which: usize) -> &[f32] {
        let o = self.key[r] * 6 * self.h + which * self.h;
        &self.table[o..o + self.h]
    }
}

/// `x = x * (1 + scale) + shift` with the AdaLN vectors of each row (`which`: 0 attention, 3
/// feed-forward; `[shift, scale, gate]` follow).
fn modulate(x: &mut [f32], ada: Ada, h: usize, which: usize) {
    for_chunks(x, h, true, |r, row| {
        let (shift, scale) = (ada.get(r, which), ada.get(r, which + 1));
        for ((v, &s), &c) in row.iter_mut().zip(scale).zip(shift) {
            *v = *v * (1.0 + s) + c;
        }
    });
}

/// `x += gate * y` (AdaLN gate `which`: 2 attention, 5 feed-forward), or `x += y`.
fn residual(x: &mut [f32], y: &[f32], ada: Option<Ada>, h: usize, which: usize) {
    match ada {
        Some(a) => for_chunks(x, h, true, |r, row| {
            let g = a.get(r, which);
            for ((v, &yv), &gv) in row.iter_mut().zip(&y[r * h..(r + 1) * h]).zip(g) {
                *v += gv * yv;
            }
        }),
        None => add_inplace(x, y),
    }
}

/// A stored linear layer of the DiT, for LoRA files ([`super::lora`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct LayerId {
    refiner: bool,
    /// `usize::MAX`: the final layer.
    block: usize,
    /// 0 - 2: the attention's stored projections (one fused matrix, or q / k / v), 3: its output,
    /// 4 / 5: the feed-forward's two layers, 6: the AdaLN projection.
    slot: u8,
}

impl<'a> Dit<'a> {
    /// The stored layers behind a module in ComfyUI's naming (`blocks.N.attn.qkv_proj`,
    /// `.attn.out_proj`, `.mlp.fc1`, `.mlp.fc2`, `.adaln_proj.linear`, the same under
    /// `token_refiner.`, `final_layer.adaln_proj.linear`) and, for every output row of each, the
    /// row of the module's output it holds, in ComfyUI's order: `qkv_proj` as `[q; k; v]`, `fc1` as
    /// `[gate; value]`. None: no such module.
    pub(crate) fn lora_parts(&self, module: &str) -> Option<Vec<(LayerId, Vec<usize>)>> {
        let (inner, hd, ffn) = (self.cfg.inner(), self.cfg.head_dim, self.cfg.ffn);
        if module == "final_layer.adaln_proj.linear" {
            let id = LayerId {
                refiner: false,
                block: usize::MAX,
                slot: 6,
            };
            return Some(vec![(id, (0..self.norm_out_ada.n_out).collect())]);
        }
        let (refiner, rest) = match module.strip_prefix("token_refiner.") {
            Some(r) => (true, r),
            None => (false, module),
        };
        let (idx, tail) = rest.strip_prefix("blocks.")?.split_once('.')?;
        let block: usize = idx.parse().ok()?;
        let b = if refiner { &self.refiner } else { &self.blocks }.get(block)?;
        let id = |slot: u8| LayerId { refiner, block, slot };
        Some(match tail {
            "attn.qkv_proj" => match &b.attn.qkv {
                Qkv::Stacked(_) => vec![(id(0), (0..3 * inner).collect())],
                // Row `(head * 3 + j) * hd + i` of the release's matrix is row `i` of head `head`
                // of q, k or v (`j`).
                Qkv::Interleaved(_) => {
                    let rows = (0..3 * inner).map(|o| (o / hd % 3) * inner + o / (3 * hd) * hd + o % hd).collect();
                    vec![(id(0), rows)]
                }
                Qkv::Split(_) => (0..3).map(|j| (id(j as u8), (j * inner..(j + 1) * inner).collect())).collect(),
            },
            "attn.out_proj" => vec![(id(3), (0..b.attn.out.n_out).collect())],
            "mlp.fc1" if b.ff.gate_first => vec![(id(4), (0..2 * ffn).collect())],
            "mlp.fc1" => vec![(id(4), (ffn..2 * ffn).chain(0..ffn).collect())],
            "mlp.fc2" => vec![(id(5), (0..b.ff.fc2.n_out).collect())],
            "adaln_proj.linear" => vec![(id(6), (0..b.adaln.as_ref()?.n_out).collect())],
            _ => return None,
        })
    }

    /// The layer of a [`LayerId`] from [`Dit::lora_parts`].
    pub(crate) fn layer_mut(&mut self, id: LayerId) -> &mut DLin<'a> {
        if id.block == usize::MAX {
            return &mut self.norm_out_ada;
        }
        let b = &mut if id.refiner { &mut self.refiner } else { &mut self.blocks }[id.block];
        match (id.slot, &mut b.attn.qkv) {
            (0..=2, Qkv::Split(w)) => &mut w[id.slot as usize],
            (0..=2, Qkv::Interleaved(w) | Qkv::Stacked(w)) => w,
            (3, _) => &mut b.attn.out,
            (4, _) => &mut b.ff.fc1,
            (5, _) => &mut b.ff.fc2,
            _ => b.adaln.as_mut().expect("the AdaLN projection of a LayerId"),
        }
    }

    /// Whether the AdaLN projections are in the curve form of ComfyUI's "pruned" files.
    pub fn is_curve(&self) -> bool {
        self.curve.is_some()
    }
}
