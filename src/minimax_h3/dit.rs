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
//! Loads the release's layout (`blocks.N.attn.qkv_proj` interleaved per head as `[q k v]`,
//! `mlp.fc1` as `[gate; value]`, `final_layer.*`) as stored, or diffusers' (`transformer_blocks.N`,
//! `to_q` / `to_k` / `to_v`, `ff.net.0.proj` as `[value; gate]`).

use crate::kernels::*;
use crate::model::{Lin, Quant, linear, load_lin};
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};
use std::path::Path;

/// The modalities a row can carry (AdaLN sets per timestep).
pub const MODALITIES: usize = 3;
pub const TAG_IMAGE: u8 = 0;
pub const TAG_TEXT: u8 = 1;

#[derive(Debug, Clone)]
pub struct DitConfig {
    pub hidden: usize,
    pub layers: usize,
    pub refiner_layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub in_channels: usize,
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
/// patch, timestep and output layers so), others bf16 (or as stored).
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
        let w = if info.dtype == "F32" {
            f32_lin(&files.f32(&wn)?, n_out, n_in)?
        } else {
            load_lin(files, &wn, n_out, n_in, Quant::Bf16)?
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
            for_chunks(&mut y, self.n_out, t > 1, |_, r| add_inplace(r, b));
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
    Split([DLin<'a>; 3]),
    /// One projection whose rows are `[q k v]` per head (the release's `qkv_proj`).
    Interleaved(DLin<'a>),
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
    time1: DLin<'a>,
    time2: DLin<'a>,
    refiner: Vec<Block<'a>>,
    refiner_norm: Vec<f32>,
    blocks: Vec<Block<'a>>,
    norm_out: Vec<f32>,
    norm_out_ada: DLin<'a>,
    /// The output head of each sampling step (H3-Image), or the one head.
    heads_out: Vec<DLin<'a>>,
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
                        qkv: Qkv::Interleaved(DLin::load(files, &format!("{p}attn.qkv_proj"), 3 * inner, h, false)?),
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
                        qkv: Qkv::Split([
                            DLin::load(files, &format!("{p}attn.to_q"), inner, h, false)?,
                            DLin::load(files, &format!("{p}attn.to_k"), inner, h, false)?,
                            DLin::load(files, &format!("{p}attn.to_v"), inner, h, false)?,
                        ]),
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
        let rd = cfg.rope_freq_dim;
        let inv_freq = (0..rd)
            .map(|i| (1.0 / cfg.rope_theta.powf((2 * i) as f64 / (2 * rd) as f64)) as f32)
            .collect();
        if 6 * rd > hd {
            bail!("DiT heads of {hd} are too narrow for the 3-axis RoPE");
        }
        Ok(Self {
            proj_in: DLin::load(files, &n("video_patch_proj", "proj_in"), h, pd, true)?,
            context: DLin::load(files, &n("condition_proj", "context_embedder"), h, cfg.text_dim, true)?,
            time1: DLin::load(
                files,
                &n("time_embedder.proj_in", "time_embedder.linear_1"),
                cfg.time_hidden,
                cfg.freq_dim,
                true,
            )?,
            time2: DLin::load(
                files,
                &n("time_embedder.proj_out", "time_embedder.linear_2"),
                td,
                cfg.time_hidden,
                true,
            )?,
            refiner,
            refiner_norm: norm("token_refiner.final_norm.weight", h)?,
            blocks,
            norm_out: norm(&n("final_layer.norm.weight", "norm_out.norm.weight"), h)?,
            norm_out_ada: DLin::load(files, &n("final_layer.adaln_proj.linear", "norm_out.linear"), 2 * h, td, true)?,
            heads_out,
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

    /// The timestep embedding (`[time_dim]`) of a `t` in `[0, 1]`.
    fn time_embed(&self, t: f32) -> Vec<f32> {
        let e = crate::sd::ops::timestep_embedding(t, self.cfg.freq_dim, true, 0.0);
        let mut a = self.time1.forward(&e, 1);
        a.iter_mut().for_each(|v| *v = crate::sd::ops::silu(*v));
        self.time2.forward(&a, 1)
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

    /// One block over `x` (`[n, H]`). `ada`: the per-row AdaLN vectors (`[n, 6H]`, shift / scale /
    /// gate for attention then feed-forward), none for the refiner; `rope`: per-row cos / sin.
    fn block(&self, b: &Block, x: &mut [f32], n: usize, ada: Option<&[f32]>, rope: Option<(&[f32], &[f32])>) {
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
            Qkv::Split([wq, wk, wv]) => (wq.forward(&hn, n), wk.forward(&hn, n), wv.forward(&hn, n)),
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
        let temb: Vec<Vec<f32>> = p.timesteps.iter().map(|&t| self.time_embed(t)).collect();
        let st: Vec<f32> = temb.iter().flat_map(|e| e.iter().map(|&v| crate::sd::ops::silu(v))).collect();
        let nt = p.timesteps.len();
        let (cos, sin) = self.rope(p.positions);
        let six = 6 * h;
        for b in &self.blocks {
            // [nt, 3 * 6H]: the six vectors of each (timestep, modality), then one row per token.
            let table = b.adaln.as_ref().unwrap().forward(&st, nt);
            let mut ada = vec![0f32; n * six];
            for_chunks(&mut ada, six, true, |r, row| {
                let k = p.timestep_index[r] * MODALITIES + p.tags[r] as usize;
                row.copy_from_slice(&table[k * six..(k + 1) * six]);
            });
            self.block(b, &mut x, n, Some(&ada), Some((&cos, &sin)));
        }
        // Output norm with the per-timestep shift / scale, then the head over the latent rows.
        let so = self.norm_out_ada.forward(&st, nt);
        let mut hn = x[n_text * h..].to_vec();
        rmsnorm_f32w_rows_inplace(&mut hn, h, &self.norm_out, cfg.final_eps);
        for_chunks(&mut hn, h, true, |r, row| {
            let k = p.timestep_index[n_text + r];
            let (shift, scale) = (&so[k * 2 * h..k * 2 * h + h], &so[k * 2 * h + h..(k + 1) * 2 * h]);
            for ((v, &s), &c) in row.iter_mut().zip(scale).zip(shift) {
                *v = *v * (1.0 + s) + c;
            }
        });
        self.heads_out[head.min(self.heads_out.len() - 1)].forward(&hn, n_img)
    }
}

/// `x = x * (1 + scale) + shift` with the AdaLN vectors of each row (`which`: 0 attention, 3
/// feed-forward; `[shift, scale, gate]` follow).
fn modulate(x: &mut [f32], ada: &[f32], h: usize, which: usize) {
    for_chunks(x, h, true, |r, row| {
        let a = &ada[r * 6 * h..(r + 1) * 6 * h];
        let (shift, scale) = (&a[which * h..(which + 1) * h], &a[(which + 1) * h..(which + 2) * h]);
        for ((v, &s), &c) in row.iter_mut().zip(scale).zip(shift) {
            *v = *v * (1.0 + s) + c;
        }
    });
}

/// `x += gate * y` (AdaLN gate `which`: 2 attention, 5 feed-forward), or `x += y`.
fn residual(x: &mut [f32], y: &[f32], ada: Option<&[f32]>, h: usize, which: usize) {
    match ada {
        Some(a) => for_chunks(x, h, true, |r, row| {
            let g = &a[r * 6 * h + which * h..r * 6 * h + (which + 1) * h];
            for ((v, &yv), &gv) in row.iter_mut().zip(&y[r * h..(r + 1) * h]).zip(g) {
                *v += gv * yv;
            }
        }),
        None => add_inplace(x, y),
    }
}
