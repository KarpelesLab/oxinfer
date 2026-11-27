//! The "H3 corrector" of SulphurAI's Sulphur-3 (`s3_corrector.safetensors`,
//! `experimental/corrector2s.safetensors`): a separate network, trained against one MiniMax-H3
//! checkpoint (S3), that reads at every denoising step the noisy video latent, H3's video
//! velocity, the step's sigma interval and the raw Qwen3-VL caption states, and predicts a
//! residual of the velocity. The residual's RMS is soft-capped at a fraction of the velocity's,
//! scaled by a strength and added to the video velocity; the audio rows keep H3's.
//!
//! Follows the release's ComfyUI node pack (`S3_ComfyUI.zip`: `ComfyUI-H3-Corrector/corrector.py`,
//! `__init__.py`; the `experimental/H3_Corrector_Advanced_Motion_STG` pack carries the same
//! network byte for byte):
//!
//! - the architecture is in the file's `h3_corrector` metadata entry (S3's: 24 latent channels,
//!   text width 5120, width 1280, 55 blocks of 20 heads of 64, MLP ratio 4, time embedding 256,
//!   patches of 1 x 2 x 2, LayerNorm eps 1e-5), weights float32 (2.0B: 8.0 GB; the node keeps
//!   them float32, "bf16 weights change its output several-fold");
//! - `patch_in`: a 1 x 2 x 2 convolution of the latent and the velocity concatenated over the
//!   channels (48) to `T x H/2 x W/2` tokens of 1280 (the DiT's video rows, frame-major);
//! - the caption: LayerNorm then a linear to 1280; the time: sinusoidal embeddings (256, cos
//!   then sin, of `1000 sigma`) of `sigma`, `sigma_next` and their difference, through
//!   `Linear(768, 1280)`, SiLU, `Linear(1280, 1280)`;
//! - each block: `modulation(time)` (no SiLU) gives shift / scale / gate for the self-attention
//!   and the MLP; joint self-attention over every token with rotary positions on three
//!   segments of each head (time: 16 channels, rows: 24, columns: 24; consecutive pairs,
//!   `10000^(-j / (w/2))`), gated by `1 + gate`; cross-attention to the caption (unmodulated,
//!   ungated); an MLP with tanh GELU, gated by `1 + gate`;
//! - the final LayerNorm modulated by `final_modulation(time)` and `patch_out` back to
//!   `[24, T, H, W]`.
//!
//! What it is given and how its output is used ([`Corrector::correct`], `ApplyH3Corrector` in
//! the pack): the velocity is H3's data-ward one, `(x0 - z) / sigma`, which is what the DiT
//! returns here; `sigma_next` the schedule's next point; the residual `r` is capped as
//! `r / sqrt(1 + mean(r^2) / budget^2)` with `budget = budget_fraction * max(rms(v),
//! velocity_floor)` (trained at 0.5 and 0.001) and the velocity becomes `v + strength r`; on the
//! steps `start_step ..= end_step` (0-based) only.

use super::dit::DLin;
use crate::kernels::{for_chunks, gelu_tanh};
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};
use std::path::Path;

/// The architecture (the file's `h3_corrector` entry).
#[derive(Clone, Debug, PartialEq)]
pub struct CorrectorConfig {
    pub video_channels: usize,
    pub text_dim: usize,
    pub width: usize,
    pub depth: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub mlp_ratio: usize,
    pub time_embed_dim: usize,
    pub patch: [usize; 3],
    pub norm_eps: f32,
}

impl CorrectorConfig {
    fn from_json(v: &serde_json::Value) -> Result<Self> {
        let u = |k: &str, d: usize| v.get(k).and_then(|x| x.as_u64()).map_or(d, |x| x as usize);
        let patch = match v.get("patch").and_then(|p| p.as_array()) {
            Some(p) if p.len() == 3 => {
                let p: Vec<usize> = p.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect();
                [p[0], p[1], p[2]]
            }
            _ => [1, 2, 2],
        };
        let c = Self {
            video_channels: u("video_channels", 24),
            text_dim: u("text_dim", 5120),
            width: u("width", 1280),
            depth: u("depth", 55),
            heads: u("heads", 20),
            head_dim: u("head_dim", 64),
            mlp_ratio: u("mlp_ratio", 4),
            time_embed_dim: u("time_embed_dim", 256),
            patch,
            norm_eps: v.get("norm_eps").and_then(|x| x.as_f64()).unwrap_or(1e-5) as f32,
        };
        if c.width != c.heads * c.head_dim || c.patch[0] != 1 || c.patch.contains(&0) || !c.head_dim.is_multiple_of(8) {
            bail!("H3 corrector: unsupported architecture {c:?} (width = heads x head_dim, a time patch of 1)");
        }
        Ok(c)
    }
}

/// How a corrector is applied (the `ApplyH3Corrector` node's inputs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CorrectorSettings {
    /// Multiplier on the capped residual (1: as trained; 0: off).
    pub strength: f32,
    /// Cap on the residual's RMS as a fraction of the velocity's (trained at 0.5).
    pub budget_fraction: f32,
    /// Lower bound on the velocity RMS the cap is taken from.
    pub velocity_floor: f32,
    /// First step (0-based) corrected.
    pub start_step: usize,
    /// Last step (0-based, inclusive) corrected; None: through the last.
    pub end_step: Option<usize>,
}

impl Default for CorrectorSettings {
    fn default() -> Self {
        Self {
            strength: 1.0,
            budget_fraction: 0.5,
            velocity_floor: 1e-3,
            start_step: 0,
            end_step: None,
        }
    }
}

impl CorrectorSettings {
    /// Whether step `i` (0-based) is corrected.
    pub fn active(&self, i: usize) -> bool {
        self.strength != 0.0 && i >= self.start_step && self.end_step.is_none_or(|e| i <= e)
    }

    /// Sets one option by name (`corrector_strength`, `corrector_budget`, `corrector_floor`,
    /// `corrector_start`, `corrector_end`; -1 for the end: through the last step). Returns
    /// false for another name.
    pub fn set(&mut self, key: &str, v: &str) -> Result<bool> {
        let f = || -> Result<f32> { v.parse().with_context(|| format!("{key}: not a number: {v:?}")) };
        match key {
            "corrector_strength" => self.strength = f()?,
            "corrector_budget" | "corrector_budget_fraction" => self.budget_fraction = f()?,
            "corrector_floor" | "corrector_velocity_floor" => self.velocity_floor = f()?,
            "corrector_start" | "corrector_start_step" => self.start_step = v.parse().with_context(|| format!("{key}: {v:?}"))?,
            "corrector_end" | "corrector_end_step" => {
                let e: i64 = v.parse().with_context(|| format!("{key}: {v:?}"))?;
                self.end_step = (e >= 0).then_some(e as usize);
            }
            _ => return Ok(false),
        }
        if !(self.budget_fraction > 0.0 && self.velocity_floor > 0.0) {
            bail!("corrector_budget and corrector_floor must be positive");
        }
        Ok(true)
    }
}

struct Norm {
    w: Vec<f32>,
    b: Vec<f32>,
}

struct Block<'a> {
    norm_self: Norm,
    norm_cross: Norm,
    norm_mlp: Norm,
    qkv: DLin<'a>,
    out: DLin<'a>,
    cq: DLin<'a>,
    ck: DLin<'a>,
    cv: DLin<'a>,
    cout: DLin<'a>,
    fc1: DLin<'a>,
    fc2: DLin<'a>,
    modulation: DLin<'a>,
}

/// A loaded corrector (weights mapped from the file, float32 as stored).
pub struct Corrector {
    pub cfg: CorrectorConfig,
    pub path: std::path::PathBuf,
    patch_in: DLin<'static>,
    text_norm: Norm,
    text_proj: DLin<'static>,
    time1: DLin<'static>,
    time2: DLin<'static>,
    blocks: Vec<Block<'static>>,
    final_norm: Norm,
    final_mod: DLin<'static>,
    patch_out: DLin<'static>,
}

/// LayerNorm of each row of `x` (`[rows, w.len()]`) with weight and bias, float64 sums.
fn layer_norm(x: &[f32], n: &Norm, eps: f32) -> Vec<f32> {
    let c = n.w.len();
    let mut y = x.to_vec();
    for_chunks(&mut y, c, x.len() > c, |_, r| {
        let mean = r.iter().map(|&v| v as f64).sum::<f64>() / c as f64;
        let var = r.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / c as f64;
        let inv = 1.0 / (var + eps as f64).sqrt();
        for (i, v) in r.iter_mut().enumerate() {
            *v = ((*v as f64 - mean) * inv) as f32 * n.w[i] + n.b[i];
        }
    });
    y
}

/// `x * (1 + scale) + shift` on every row.
fn modulate(x: &mut [f32], shift: &[f32], scale: &[f32]) {
    let c = shift.len();
    for_chunks(x, c, true, |_, r| {
        for ((v, &sh), &sc) in r.iter_mut().zip(shift).zip(scale) {
            *v = *v * (1.0 + sc) + sh;
        }
    });
}

/// `h += (1 + gate) * y` on every row (`gate` None: `h += y`).
fn add_gated(h: &mut [f32], y: &[f32], gate: Option<&[f32]>) {
    match gate {
        Some(g) => {
            let c = g.len();
            for_chunks(h, c, true, |i, r| {
                let yr = &y[i * c..(i + 1) * c];
                for ((v, &yv), &gv) in r.iter_mut().zip(yr).zip(g) {
                    *v += (1.0 + gv) * yv;
                }
            });
        }
        None => h.iter_mut().zip(y).for_each(|(v, &yv)| *v += yv),
    }
}

/// The sinusoidal embedding of one sigma (`_time_embedding`): `width / 2` frequencies
/// `10000^(-i / half)`, angles `sigma * f * 1000`, cosines then sines, zero padding to `width`.
fn time_embedding(sigma: f32, width: usize) -> Vec<f32> {
    let half = width / 2;
    let ln = -(10_000f64.ln() as f32);
    let mut out = vec![0f32; width];
    for i in 0..half {
        let f = (ln * i as f32 / half as f32).exp();
        let a = sigma * f * 1000.0;
        out[i] = a.cos();
        out[half + i] = a.sin();
    }
    out
}

/// Rotates channel pairs `start..start + width` of every head of `x` (`[n, heads * dh]`) by the
/// token's position along one axis (`pos[token]`).
fn rope_segment(x: &mut [f32], pos: &[f32], heads: usize, dh: usize, start: usize, width: usize) {
    if width == 0 {
        return;
    }
    let half = width / 2;
    let ln = -(10_000f64.ln() as f32);
    let inv: Vec<f32> = (0..half).map(|j| (ln * j as f32 / half.max(1) as f32).exp()).collect();
    for_chunks(x, heads * dh, true, |t, r| {
        let p = pos[t];
        for j in 0..half {
            let (s, c) = (p * inv[j]).sin_cos();
            for h in 0..heads {
                let o = h * dh + start + 2 * j;
                let (a, b) = (r[o], r[o + 1]);
                r[o] = a * c - b * s;
                r[o + 1] = a * s + b * c;
            }
        }
    });
}

impl Corrector {
    /// Loads a corrector file (its architecture from the `h3_corrector` metadata entry).
    pub fn load(path: &Path) -> Result<Self> {
        let files: &'static ModelFiles = Box::leak(Box::new(
            ModelFiles::open_file(path, false).with_context(|| format!("H3 corrector {}", path.display()))?,
        ));
        let meta = files
            .metadata("h3_corrector")
            .with_context(|| format!("{}: not an H3 corrector (no h3_corrector metadata entry)", path.display()))?;
        let cfg = CorrectorConfig::from_json(&serde_json::from_str(meta).context("h3_corrector entry: not JSON")?)?;
        let (w, c) = (cfg.width, cfg.video_channels);
        let pp = cfg.patch[1] * cfg.patch[2];
        let lin = |n: &str, o: usize, i: usize| DLin::load(files, n, o, i, true).with_context(|| format!("{}: {n}", path.display()));
        let norm = |n: &str, d: usize| -> Result<Norm> {
            let (wv, bv) = (files.f32(&format!("{n}.weight"))?, files.f32(&format!("{n}.bias"))?);
            if wv.len() != d || bv.len() != d {
                bail!("{n}: {} / {} values for a norm of {d}", wv.len(), bv.len());
            }
            Ok(Norm { w: wv, b: bv })
        };
        // The convolution as a linear over the `[channel, kh, kw]` patch vector.
        let pw = files.f32("patch_in.weight")?;
        if pw.len() != w * 2 * c * pp {
            bail!(
                "patch_in.weight: {} values for [{w}, {}, 1, {}, {}]",
                pw.len(),
                2 * c,
                cfg.patch[1],
                cfg.patch[2]
            );
        }
        let patch_in = DLin::from_f32(&pw, Some(files.f32("patch_in.bias")?), w, 2 * c * pp)?;
        let mut blocks = Vec::with_capacity(cfg.depth);
        let ff = w * cfg.mlp_ratio;
        for i in 0..cfg.depth {
            let p = |n: &str| format!("blocks.{i}.{n}");
            blocks.push(Block {
                norm_self: norm(&p("norm_self"), w)?,
                norm_cross: norm(&p("norm_cross"), w)?,
                norm_mlp: norm(&p("norm_mlp"), w)?,
                qkv: lin(&p("self_attention.qkv"), 3 * w, w)?,
                out: lin(&p("self_attention.out"), w, w)?,
                cq: lin(&p("cross_attention.q"), w, w)?,
                ck: lin(&p("cross_attention.k"), w, w)?,
                cv: lin(&p("cross_attention.v"), w, w)?,
                cout: lin(&p("cross_attention.out"), w, w)?,
                fc1: lin(&p("mlp.fc1"), ff, w)?,
                fc2: lin(&p("mlp.fc2"), w, ff)?,
                modulation: lin(&p("modulation"), 6 * w, w)?,
            });
        }
        Ok(Self {
            patch_in,
            text_norm: norm("text_norm", cfg.text_dim)?,
            text_proj: lin("text_projection", w, cfg.text_dim)?,
            time1: lin("time_mlp.0", w, 3 * cfg.time_embed_dim)?,
            time2: lin("time_mlp.2", w, w)?,
            blocks,
            final_norm: norm("final_norm", w)?,
            final_mod: lin("final_modulation", 2 * w, w)?,
            patch_out: lin("patch_out", c * pp, w)?,
            path: path.to_path_buf(),
            cfg,
        })
    }

    /// The raw residual (`[C, T, H, W]`) for latent `z` and velocity `v` (both `[C, T, H, W]`)
    /// on the step `sigma -> sigma_next`; `text` the caption states (`[L, text_dim]`).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        z: &[f32],
        v: &[f32],
        t: usize,
        h: usize,
        w: usize,
        sigma: f32,
        sigma_next: f32,
        text: &[f32],
    ) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let (c, d, heads, dh) = (cfg.video_channels, cfg.width, cfg.heads, cfg.head_dim);
        let (ph, pw) = (cfg.patch[1], cfg.patch[2]);
        if z.len() != c * t * h * w
            || v.len() != z.len()
            || !h.is_multiple_of(ph)
            || !w.is_multiple_of(pw)
            || !text.len().is_multiple_of(cfg.text_dim)
        {
            bail!(
                "H3 corrector: a latent of {} values for [{c}, {t}, {h}, {w}] or text of {} values",
                z.len(),
                text.len()
            );
        }
        let (hp, wp) = (h / ph, w / pw);
        let n = t * hp * wp;
        let l = text.len() / cfg.text_dim;
        let plane = t * h * w;
        // Patch vectors: [z channels; v channels] x (kh, kw).
        let kin = 2 * c * ph * pw;
        let mut x = vec![0f32; n * kin];
        for_chunks(&mut x, kin, true, |tok, r| {
            let (f, rest) = (tok / (hp * wp), tok % (hp * wp));
            let (y0, x0) = ((rest / wp) * ph, (rest % wp) * pw);
            for ch in 0..2 * c {
                let src = if ch < c { &z[ch * plane..] } else { &v[(ch - c) * plane..] };
                for a in 0..ph {
                    for b in 0..pw {
                        r[(ch * ph + a) * pw + b] = src[(f * h + y0 + a) * w + x0 + b];
                    }
                }
            }
        });
        let mut hid = self.patch_in.forward(&x, n);
        drop(x);
        let txt = self.text_proj.forward(&layer_norm(text, &self.text_norm, cfg.norm_eps), l);
        let te: Vec<f32> = [sigma, sigma_next, sigma - sigma_next]
            .iter()
            .flat_map(|&s| time_embedding(s, cfg.time_embed_dim))
            .collect();
        let mut t1 = self.time1.forward(&te, 1);
        t1.iter_mut().for_each(|v| *v = crate::sd::ops::silu(*v));
        let time = self.time2.forward(&t1, 1);
        // Rotary positions per token and the three segments of a head.
        let pos_t: Vec<f32> = (0..n).map(|i| (i / (hp * wp)) as f32).collect();
        let pos_y: Vec<f32> = (0..n).map(|i| (i % (hp * wp) / wp) as f32).collect();
        let pos_x: Vec<f32> = (0..n).map(|i| (i % wp) as f32).collect();
        let tw = 2 * (dh / 8);
        let hw = 2 * ((dh - tw) / 4);
        let segs = [(0, tw, &pos_t), (tw, hw, &pos_y), (tw + hw, dh - tw - hw, &pos_x)];
        for b in &self.blocks {
            let m = b.modulation.forward(&time, 1);
            let ch = |k: usize| &m[k * d..(k + 1) * d];
            // Self-attention.
            let mut a = layer_norm(&hid, &b.norm_self, cfg.norm_eps);
            modulate(&mut a, ch(0), ch(1));
            let qkv = b.qkv.forward(&a, n);
            let (mut q, mut k, mut vv) = (vec![0f32; n * d], vec![0f32; n * d], vec![0f32; n * d]);
            for (i, row) in qkv.chunks_exact(3 * d).enumerate() {
                q[i * d..(i + 1) * d].copy_from_slice(&row[..d]);
                k[i * d..(i + 1) * d].copy_from_slice(&row[d..2 * d]);
                vv[i * d..(i + 1) * d].copy_from_slice(&row[2 * d..]);
            }
            drop(qkv);
            for &(start, width, pos) in &segs {
                rope_segment(&mut q, pos, heads, dh, start, width);
                rope_segment(&mut k, pos, heads, dh, start, width);
            }
            let att = crate::sd::ops::attention(&q, &k, &vv, n, n, heads, dh, false);
            let o = b.out.forward(&att, n);
            add_gated(&mut hid, &o, Some(ch(2)));
            // Cross-attention to the caption.
            let cq = b.cq.forward(&layer_norm(&hid, &b.norm_cross, cfg.norm_eps), n);
            let (ck, cv) = (b.ck.forward(&txt, l), b.cv.forward(&txt, l));
            let att = crate::sd::ops::attention(&cq, &ck, &cv, n, l, heads, dh, false);
            add_gated(&mut hid, &b.cout.forward(&att, n), None);
            // MLP.
            let mut a = layer_norm(&hid, &b.norm_mlp, cfg.norm_eps);
            modulate(&mut a, ch(3), ch(4));
            let mut f = b.fc1.forward(&a, n);
            for_chunks(&mut f, d * cfg.mlp_ratio, true, |_, r| {
                r.iter_mut().for_each(|v| *v = gelu_tanh(*v))
            });
            add_gated(&mut hid, &b.fc2.forward(&f, n), Some(ch(5)));
        }
        let fm = self.final_mod.forward(&time, 1);
        let mut a = layer_norm(&hid, &self.final_norm, cfg.norm_eps);
        modulate(&mut a, &fm[..d], &fm[d..]);
        let p = self.patch_out.forward(&a, n);
        // [token][c, kh, kw] -> [C, T, H, W].
        let mut out = vec![0f32; c * plane];
        let pp = ph * pw;
        for (tok, row) in p.chunks_exact(c * pp).enumerate() {
            let (f, rest) = (tok / (hp * wp), tok % (hp * wp));
            let (y0, x0) = ((rest / wp) * ph, (rest % wp) * pw);
            for ch in 0..c {
                for a in 0..ph {
                    for bb in 0..pw {
                        out[ch * plane + (f * h + y0 + a) * w + x0 + bb] = row[(ch * ph + a) * pw + bb];
                    }
                }
            }
        }
        Ok(out)
    }

    /// The velocity correction of one step: the residual for `z` / `v` (`[C, T, H, W]`), capped
    /// ([`bound_residual`]) and scaled by the strength, added to `v` in place. Returns the ratio
    /// of the applied residual's RMS to the velocity's.
    #[allow(clippy::too_many_arguments)]
    pub fn correct(
        &self,
        s: &CorrectorSettings,
        z: &[f32],
        v: &mut [f32],
        t: usize,
        h: usize,
        w: usize,
        sigma: f32,
        sigma_next: f32,
        text: &[f32],
    ) -> Result<f32> {
        let raw = self.forward(z, v, t, h, w, sigma, sigma_next, text)?;
        let (applied, ratio) = bound_residual(&raw, v, s.budget_fraction, s.velocity_floor);
        v.iter_mut().zip(&applied).for_each(|(x, &r)| *x += s.strength * r);
        Ok(ratio)
    }
}

/// `bound_residual` of the node pack: `raw` scaled by `1 / sqrt(1 + mean(raw^2) / budget^2)`,
/// `budget = budget_fraction * max(rms(velocity), velocity_floor)`; returns it and the ratio of
/// its RMS to the velocity's.
pub fn bound_residual(raw: &[f32], velocity: &[f32], budget_fraction: f32, velocity_floor: f32) -> (Vec<f32>, f32) {
    let ms = |x: &[f32]| x.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / x.len().max(1) as f64;
    let base = ms(velocity).sqrt();
    let budget = budget_fraction as f64 * base.max(velocity_floor as f64);
    let scale = 1.0 / (1.0 + ms(raw) / (budget * budget).max(1e-12)).sqrt();
    let applied: Vec<f32> = raw.iter().map(|&v| (v as f64 * scale) as f32).collect();
    let ratio = (ms(&applied).sqrt() / base.max(1e-12)) as f32;
    (applied, ratio)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bound_caps_at_the_budget() {
        let v = vec![1.0f32; 64];
        // A residual far above the budget comes out near it (0.5 of the velocity's RMS).
        let (a, r) = bound_residual(&vec![100.0f32; 64], &v, 0.5, 1e-3);
        assert!((r - 0.5).abs() < 1e-3, "{r}");
        assert!(a.iter().all(|&x| (x - a[0]).abs() < 1e-6));
        // A small one passes nearly unchanged.
        let (a, _) = bound_residual(&vec![0.01f32; 64], &v, 0.5, 1e-3);
        assert!((a[0] - 0.01).abs() < 1e-5);
    }

    #[test]
    fn time_embedding_layout() {
        let e = time_embedding(0.5, 256);
        assert!((e[0] - (500f32).cos()).abs() < 1e-6 && (e[128] - (500f32).sin()).abs() < 1e-6);
        assert_eq!(time_embedding(0.0, 8), vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn settings_parse() {
        let mut s = CorrectorSettings::default();
        assert!(s.set("corrector_end", "4").unwrap());
        assert!(s.set("corrector_strength", "0.5").unwrap());
        assert!(!s.set("shift", "3").unwrap());
        assert!(s.active(4) && !s.active(5));
        s.set("corrector_end", "-1").unwrap();
        assert!(s.active(99));
        assert!(s.set("corrector_floor", "0").is_err());
    }
}
