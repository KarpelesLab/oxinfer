//! LTX-2.5's duration head (`model_patches/ltx-2.5-duration-head-bf16.safetensors`,
//! `ltx_core.duration_head.DurationHead`): the clip length a prompt implies, from the
//! connectors' context rows of both streams.
//!
//! Each stream's rows projected to 256 plus a learned modality embedding, both streams' rows as
//! one sequence, one learned query attending over them (`nn.MultiheadAttention`, 4 heads), a
//! GELU (tanh) MLP to one value: the log of the duration in seconds. The pipeline clamps it to
//! 1 to 20 s and snaps the frame count down to `8 k + 1` (up when that falls under the minimum).

use crate::minimax_h3::dit::DLin;
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};

pub struct DurationHead<'a> {
    video_in: DLin<'a>,
    audio_in: DLin<'a>,
    video_emb: Vec<f32>,
    audio_emb: Vec<f32>,
    query: Vec<f32>,
    /// `[3 * hidden, hidden]` q, k, v rows.
    in_proj: DLin<'a>,
    out_proj: DLin<'a>,
    mlp_hidden: DLin<'a>,
    mlp_out: DLin<'a>,
    hidden: usize,
    heads: usize,
}

impl<'a> DurationHead<'a> {
    pub fn load(f: &'a ModelFiles) -> Result<Self> {
        let p = "duration_head";
        let shape = |n: &str| -> Result<Vec<usize>> { Ok(f.info(n).with_context(|| format!("missing tensor {n}"))?.shape.clone()) };
        let vi = shape(&format!("{p}.video_input_proj.weight"))?;
        let ai = shape(&format!("{p}.audio_input_proj.weight"))?;
        let (hidden, vd, ad) = (vi[0], vi[1], ai[1]);
        if hidden % 4 != 0 {
            bail!("duration head: pooler width {hidden} is not a multiple of its 4 heads");
        }
        let mh = shape(&format!("{p}.mlp_hidden.weight"))?[0];
        let lin = |n: &str, o: usize, i: usize| DLin::load(f, &format!("{p}.{n}"), o, i, true);
        let lin_named = |n: &str, o: usize, i: usize| -> Result<DLin<'a>> {
            // `nn.MultiheadAttention` names its fused projection `in_proj_weight` / `in_proj_bias`.
            let w = f.f32(&format!("{p}.{n}_weight"))?;
            let b = f.f32(&format!("{p}.{n}_bias"))?;
            DLin::from_f32(&w, Some(b), o, i)
        };
        Ok(Self {
            video_in: lin("video_input_proj", hidden, vd)?,
            audio_in: lin("audio_input_proj", hidden, ad)?,
            video_emb: f.f32(&format!("{p}.video_modality_emb"))?,
            audio_emb: f.f32(&format!("{p}.audio_modality_emb"))?,
            query: f.f32(&format!("{p}.attention_pooler.query_tokens"))?,
            in_proj: lin_named("attention_pooler.cross_attn.in_proj", 3 * hidden, hidden)?,
            out_proj: lin("attention_pooler.cross_attn.out_proj", hidden, hidden)?,
            mlp_hidden: lin("mlp_hidden", mh, hidden)?,
            mlp_out: lin("mlp_out", 1, mh)?,
            hidden,
            heads: 4,
        })
    }

    /// The predicted duration in seconds from the context rows (`[n, video_dim]`,
    /// `[m, audio_dim]`).
    pub fn seconds(&self, video: &[f32], audio: &[f32]) -> f32 {
        let h = self.hidden;
        let nv = video.len() / self.video_in.n_in;
        let na = audio.len() / self.audio_in.n_in;
        let mut tokens = self.video_in.forward(video, nv);
        for r in tokens.chunks_mut(h) {
            r.iter_mut().zip(&self.video_emb).for_each(|(a, &b)| *a += b);
        }
        let mut at = self.audio_in.forward(audio, na);
        for r in at.chunks_mut(h) {
            r.iter_mut().zip(&self.audio_emb).for_each(|(a, &b)| *a += b);
        }
        tokens.extend_from_slice(&at);
        let n = nv + na;
        // q from the query token, k and v from the tokens (the fused projection's row blocks).
        let qkv_q = self.in_proj.forward(&self.query, 1);
        let qkv = self.in_proj.forward(&tokens, n);
        let q = &qkv_q[..h];
        let hd = h / self.heads;
        let scale = 1.0 / (hd as f32).sqrt();
        let mut pooled = vec![0f32; h];
        for hh in 0..self.heads {
            let qs = &q[hh * hd..(hh + 1) * hd];
            let scores: Vec<f32> = (0..n)
                .map(|t| {
                    let k = &qkv[t * 3 * h + h + hh * hd..][..hd];
                    qs.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * scale
                })
                .collect();
            let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
            let z: f32 = e.iter().sum();
            for (t, w) in e.iter().enumerate() {
                let v = &qkv[t * 3 * h + 2 * h + hh * hd..][..hd];
                for (o, &vv) in pooled[hh * hd..(hh + 1) * hd].iter_mut().zip(v) {
                    *o += w / z * vv;
                }
            }
        }
        let pooled = self.out_proj.forward(&pooled, 1);
        let mut hidden = self.mlp_hidden.forward(&pooled, 1);
        hidden.iter_mut().for_each(|v| *v = crate::kernels::gelu_tanh(*v));
        self.mlp_out.forward(&hidden, 1)[0].exp()
    }
}

/// `seconds_to_clamped_num_frames` with the predictor's 1 to 20 s range.
pub fn frames_for(seconds: f32, fps: f32) -> usize {
    let (min_f, max_f) = ((1.0 * fps).round_ties_even() as i64, (20.0 * fps).round_ties_even() as i64);
    let raw = ((seconds * fps).round_ties_even() as i64).clamp(min_f, max_f).max(1) as usize;
    let f = super::snap_frames(raw);
    if (f as i64) < min_f {
        let ts = super::TIME_SCALE as i64;
        (((min_f - 1 + ts - 1) / ts * ts + 1).min(max_f)) as usize
    } else {
        f
    }
}
