//! MiniMax M3's vision tower (`MiniMaxM3VLVisionModel` plus its projector): Qwen2-VL's
//! preprocessing (14-pixel patches, two temporal frames, 2x2 merge windows), a bias-free Conv3d
//! patch embedding, a CLIP encoder (pre-LayerNorm `pre_layrnorm`, LayerNorm blocks with biased
//! attention and an exact-GELU MLP, no final norm) whose queries and keys carry a partial 3D RoPE:
//! each of time, row and column gets `2 * (hd / 3 / 2)` dimensions' worth of frequencies (13 of
//! them for the 80-wide heads), laid out `[t | h | w]` and rotated half against half over the
//! first `3 * 2 * (hd / 3 / 2)` dimensions; the rest pass through. Every patch of a request attends
//! to every other (one sequence, no mask). The projector maps each patch to the text width (GELU
//! MLP), then fuses each merge window's four patches (concatenated) into one token with a second
//! GELU MLP (`patch_merge_mlp` in the release, `merge_linear_*` in transformers).

use crate::bert::{LayerNorm, gelu_erf};
use crate::kernels::*;
use crate::model::{Lin, linear};
use crate::qwen_vision::{Pixels, QwenVisionConfig, VLin};
use crate::safetensors::ModelFiles;
use anyhow::{Result, bail};

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

/// MiniMax M3's tower and projector (see the module docs).
pub struct MiniMaxM3Tower<'a> {
    pub cfg: QwenVisionConfig,
    patch_w: Lin<'a>, // [H, C*T*p*p]
    pre_norm: LayerNorm,
    blocks: Vec<Block<'a>>,
    proj1: VLin<'a>,
    proj2: VLin<'a>,
    merge1: VLin<'a>,
    merge2: VLin<'a>,
    /// Frequencies of one axis (`hd / 3 / 2` of them).
    inv_freq: Vec<f32>,
    eps: f32,
}

impl<'a> MiniMaxM3Tower<'a> {
    /// Loads `vision_tower.vision_model.*`, `multi_modal_projector.*` and `patch_merge_mlp.*`
    /// (the release's names) or transformers' (`model.vision_tower.*`, `merge_linear_*`), bf16.
    pub fn load(cfg: QwenVisionConfig, files: &'a ModelFiles) -> Result<Self> {
        if files.has("v.patch_embd.weight") {
            return Self::load_mmproj(cfg, files);
        }
        let pre = ["vision_tower.vision_model.", "model.vision_tower.", "vision_tower."]
            .into_iter()
            .find(|p| files.has(&format!("{p}embeddings.patch_embedding.weight")) || files.has(&format!("{p}embeddings.proj.weight")))
            .ok_or_else(|| anyhow::anyhow!("the checkpoint has no MiniMax vision tower (vision_tower.vision_model.*)"))?;
        let mp = if files.has("model.multi_modal_projector.linear_1.weight") {
            "model."
        } else {
            ""
        };
        let (h, nh) = (cfg.hidden_size, cfg.num_heads);
        let (c, t, p) = (cfg.in_channels, cfg.temporal_patch_size, cfg.patch_size);
        let out = cfg.out_hidden_size;
        let pw = [
            format!("{pre}embeddings.patch_embedding.weight"),
            format!("{pre}embeddings.proj.weight"),
        ]
        .into_iter()
        .find(|n| files.has(n))
        .unwrap();
        let patch_w = Lin::bf16(files.bf16_conv(&pw, &[h, c, t, p, p])?);
        let dense = |n: &str, o: usize, i: usize| VLin::load(files, n, (o, i), (o, i));
        let inter = cfg.intermediate_size;
        let mut blocks = Vec::with_capacity(cfg.depth);
        for l in 0..cfg.depth {
            let b = format!("{pre}encoder.layers.{l}.");
            blocks.push(Block {
                norm1: LayerNorm::load(files, &format!("{b}layer_norm1"), h)?,
                norm2: LayerNorm::load(files, &format!("{b}layer_norm2"), h)?,
                q: dense(&format!("{b}self_attn.q_proj"), h, h)?,
                k: dense(&format!("{b}self_attn.k_proj"), h, h)?,
                v: dense(&format!("{b}self_attn.v_proj"), h, h)?,
                o: dense(&format!("{b}self_attn.out_proj"), h, h)?,
                fc1: dense(&format!("{b}mlp.fc1"), inter, h)?,
                fc2: dense(&format!("{b}mlp.fc2"), h, inter)?,
            });
        }
        let m2 = cfg.spatial_merge_size * cfg.spatial_merge_size;
        let ph = cfg.projector_hidden_size.unwrap_or(out);
        let (merge1, merge2) = if files.has("patch_merge_mlp.linear_1.weight") {
            ("patch_merge_mlp.linear_1".to_string(), "patch_merge_mlp.linear_2".to_string())
        } else {
            (
                format!("{mp}multi_modal_projector.merge_linear_1"),
                format!("{mp}multi_modal_projector.merge_linear_2"),
            )
        };
        let hd = h / nh;
        let sd = 2 * ((hd / 3) / 2);
        if sd == 0 || 3 * sd > hd {
            bail!("vision heads of {hd} are too narrow for the 3D RoPE");
        }
        let theta = cfg.rope_theta();
        Ok(Self {
            pre_norm: LayerNorm::load(files, &format!("{pre}pre_layrnorm"), h)?,
            patch_w,
            blocks,
            proj1: dense(&format!("{mp}multi_modal_projector.linear_1"), ph, h)?,
            proj2: dense(&format!("{mp}multi_modal_projector.linear_2"), out, ph)?,
            merge1: dense(&merge1, ph, m2 * out)?,
            merge2: dense(&merge2, out, ph)?,
            inv_freq: (0..sd / 2).map(|i| (1.0 / theta.powf((2 * i) as f64 / sd as f64)) as f32).collect(),
            eps: cfg.layer_norm_eps.unwrap_or(1e-5) as f32,
            cfg,
        })
    }

    /// The tower from llama.cpp's `minimax_m3` mmproj file: `v.*` / `mm.*` names, the Conv3d as
    /// one Conv2d per frame, and the query / key rows of each head reordered from Hugging Face's
    /// `[t h w | t h w | rest]` halves to `[t t | h h | w w | rest]` (undone here).
    fn load_mmproj(cfg: QwenVisionConfig, files: &'a ModelFiles) -> Result<Self> {
        let (h, nh) = (cfg.hidden_size, cfg.num_heads);
        let (c, t, p) = (cfg.in_channels, cfg.temporal_patch_size, cfg.patch_size);
        let out = cfg.out_hidden_size;
        let pp = p * p;
        let slices = (0..t)
            .map(|k| {
                let n = if k == 0 {
                    "v.patch_embd.weight".to_string()
                } else {
                    format!("v.patch_embd.weight.{k}")
                };
                let w = files.bf16(&n)?;
                if w.len() != h * c * pp {
                    bail!("{n}: {} values, expected {}", w.len(), h * c * pp);
                }
                Ok(w)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut pw = crate::quant::PageVec::<u16>::zeroed(h * c * t * pp);
        for oc in 0..h * c {
            for (k, w) in slices.iter().enumerate() {
                pw[(oc * t + k) * pp..(oc * t + k + 1) * pp].copy_from_slice(&w[oc * pp..(oc + 1) * pp]);
            }
        }
        // llama.cpp's `_permute_vit_qk`: stored row j of a head is Hugging Face's row perm[j].
        let hd = h / nh;
        let axis = 2 * ((2 * (hd / 2) / 3) / 2);
        let ah = axis / 2;
        let half = 3 * ah;
        let mut head_perm: Vec<usize> = Vec::with_capacity(hd);
        for a in 0..3 {
            head_perm.extend(a * ah..(a + 1) * ah);
            head_perm.extend(half + a * ah..half + (a + 1) * ah);
        }
        head_perm.extend(2 * half..hd);
        let perm: Vec<usize> = (0..nh).flat_map(|n| head_perm.iter().map(move |&r| n * hd + r)).collect();
        let dense = |n: &str, o: usize, i: usize| VLin::load(files, n, (o, i), (o, i));
        let inter = cfg.intermediate_size;
        let mut blocks = Vec::with_capacity(cfg.depth);
        for l in 0..cfg.depth {
            let b = format!("v.blk.{l}.");
            blocks.push(Block {
                norm1: LayerNorm::load(files, &format!("{b}ln1"), h)?,
                norm2: LayerNorm::load(files, &format!("{b}ln2"), h)?,
                q: VLin::load_permuted(files, &format!("{b}attn_q"), (h, h), &perm)?,
                k: VLin::load_permuted(files, &format!("{b}attn_k"), (h, h), &perm)?,
                v: dense(&format!("{b}attn_v"), h, h)?,
                o: dense(&format!("{b}attn_out"), h, h)?,
                fc1: dense(&format!("{b}ffn_up"), inter, h)?,
                fc2: dense(&format!("{b}ffn_down"), h, inter)?,
            });
        }
        let m2 = cfg.spatial_merge_size * cfg.spatial_merge_size;
        let ph = cfg.projector_hidden_size.unwrap_or(out);
        let sd = 2 * ((hd / 3) / 2);
        let theta = cfg.rope_theta();
        Ok(Self {
            patch_w: Lin::bf16_owned(pw),
            pre_norm: LayerNorm::load(files, "v.pre_ln", h)?,
            blocks,
            proj1: dense("mm.1", ph, h)?,
            proj2: dense("mm.2", out, ph)?,
            merge1: dense("mm.merger.fc1", ph, m2 * out)?,
            merge2: dense("mm.merger.fc2", out, ph)?,
            inv_freq: (0..sd / 2).map(|i| (1.0 / theta.powf((2 * i) as f64 / sd as f64)) as f32).collect(),
            eps: cfg.layer_norm_eps.unwrap_or(1e-5) as f32,
            cfg,
        })
    }

    /// Encodes the patches of one image or video (`[t * h * w]`, merge-window order) into
    /// `[patches / merge^2, out_hidden_size]` language-model embeddings.
    pub fn encode(&self, px: &Pixels) -> Vec<f32> {
        let cfg = &self.cfg;
        let (h, nh) = (cfg.hidden_size, cfg.num_heads);
        let hd = h / nh;
        let (gt, gh, gw) = px.grid;
        let n = px.patches();
        let per_t = gh * gw;
        let d = px.data.len() / n;
        let mut x = vec![0f32; n * h];
        linear(&self.patch_w, h, d, &px.data, n, &mut x);
        self.pre_norm.apply_rows(&mut x, self.eps);

        // 3D RoPE per patch: `[t | h | w]` frequencies, rotated half against half.
        let m = cfg.spatial_merge_size;
        let f = self.inv_freq.len();
        let half = 3 * f;
        let (mut cos, mut sin) = (vec![0f32; n * half], vec![0f32; n * half]);
        for i in 0..n {
            let (ti, k) = (i / per_t, i % per_t);
            let bw = gw / m;
            let (blk, inb) = (k / (m * m), k % (m * m));
            let (r, c) = ((blk / bw) * m + inb / m, (blk % bw) * m + inb % m);
            for (axis, pos) in [ti, r, c].into_iter().enumerate() {
                for (j, &fr) in self.inv_freq.iter().enumerate() {
                    let a = pos as f32 * fr;
                    cos[i * half + axis * f + j] = a.cos();
                    sin[i * half + axis * f + j] = a.sin();
                }
            }
        }
        let scale = (hd as f32).powf(-0.5);
        let inter = cfg.intermediate_size;
        let mut hn = vec![0f32; n * h];
        let (mut q, mut k, mut v) = (vec![0f32; n * h], vec![0f32; n * h], vec![0f32; n * h]);
        let mut att = vec![0f32; n * h];
        let mut y = vec![0f32; n * h];
        let mut mid = vec![0f32; n * inter];
        for b in &self.blocks {
            hn.copy_from_slice(&x);
            b.norm1.apply_rows(&mut hn, self.eps);
            b.q.forward(&hn, n, &mut q);
            b.k.forward(&hn, n, &mut k);
            b.v.forward(&hn, n, &mut v);
            for (buf, s) in [(&mut q, scale), (&mut k, 1.0)] {
                for_chunks(buf, hd, true, |i, head| {
                    let pk = i / nh;
                    apply_rope(
                        &mut head[..2 * half],
                        &cos[pk * half..(pk + 1) * half],
                        &sin[pk * half..(pk + 1) * half],
                    );
                    if s != 1.0 {
                        head.iter_mut().for_each(|v| *v *= s);
                    }
                });
            }
            crate::vision::attention_bidirectional(&q, &k, &v, n, nh, hd, &mut att);
            b.o.forward(&att, n, &mut y);
            add_inplace(&mut x, &y);
            hn.copy_from_slice(&x);
            b.norm2.apply_rows(&mut hn, self.eps);
            b.fc1.forward(&hn, n, &mut mid);
            for_chunks(&mut mid, inter, true, |_, r| r.iter_mut().for_each(|v| *v = gelu_erf(*v)));
            b.fc2.forward(&mid, n, &mut y);
            add_inplace(&mut x, &y);
        }
        let _ = gt;

        // Projector: every patch to the text width, then each merge window's patches fused.
        let out = cfg.out_hidden_size;
        let ph = self.proj1.n_out();
        let mut p1 = vec![0f32; n * ph];
        self.proj1.forward(&x, n, &mut p1);
        p1.iter_mut().for_each(|v| *v = gelu_erf(*v));
        let mut p2 = vec![0f32; n * out];
        self.proj2.forward(&p1, n, &mut p2);
        let tokens = n / (m * m);
        let mut m1 = vec![0f32; tokens * ph];
        self.merge1.forward(&p2, tokens, &mut m1);
        m1.iter_mut().for_each(|v| *v = gelu_erf(*v));
        let mut emb = vec![0f32; tokens * out];
        self.merge2.forward(&m1, tokens, &mut emb);
        emb
    }
}
