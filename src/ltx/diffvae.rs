//! LTX-2.5's diffusion video decoder ("DiffVAE", `vae/ltx-2.5-video-vae-bf16.safetensors`,
//! `ltx_core.model.video_vae.DiffusionVideoDecoder`): a neighborhood-attention transformer that
//! upsamples the latents into a context volume, then denoises patchified pixels guided by it.
//!
//! - Stages 1 to 4 (deterministic): latents un-normalized, `conv_in` (a linear layer per row),
//!   per stage `NABlock`s (RMS norm, 3D neighborhood attention with absolute RoPE over the row's
//!   `(t, h, w)`, RMS norm, SwiGLU) and a pixel-shuffle upsampler (`LinearPixelShuffleUpsample`;
//!   a doubled first frame dropped). The last latent frame is repeated twice before stage 1
//!   (NATTEN's last-frame border) and the extra frames cropped from the context before stage 5.
//! - Stage 5 (diffusion): Gaussian noise over the 4 x 4-patchified pixels (timestep 1), `conv_in_x_t`,
//!   then per block the context projected and added, AdaLN-modulated neighborhood attention and
//!   SwiGLU (one shared timestep modulation plus each block's table); `norm_out`, `conv_out`,
//!   un-patchify. The release's checkpoint predicts the clean pixels in one step.
//!
//! Neighborhood attention is NATTEN's (`natten.na3d`): each query attends to the `k_t x k_h x k_w`
//! keys around it, the window shifted inward at the volume's borders. Stage 5 follows the
//! pipelines' default decode mode (`DiffVAEMode.CHUNKED_EAGER`): along the width its blocks see
//! the volume edge-replicated by half a window on both sides (the replicated keys at their own,
//! out-of-range positions), so the windows of the outer columns stay centered.
//!
//! Latents smaller than every stage's window are edge-padded first and the padding cropped from
//! the pixels. The decode runs untiled.

use super::vae::Vid;
use crate::kernels::{dot, for_chunks};
use crate::minimax_h3::dit::DLin;
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};

/// The decoder's configuration (`config` metadata, `vae.decoder`).
#[derive(Clone, Debug)]
pub struct DiffVaeConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub patch: usize,
    pub head_dim: usize,
    pub stage_channels: Vec<usize>,
    pub stage_depths: Vec<usize>,
    pub stage_kernels: Vec<[usize; 3]>,
    /// Per upsampler: its stride and channel reduction.
    pub upsamples: Vec<([usize; 3], usize)>,
    pub stage5_kernel: [usize; 3],
    pub timestep_mult: f32,
    pub steps: usize,
    /// The model predicts clean pixels (else velocities).
    pub x0: bool,
}

fn config_json(f: &ModelFiles) -> Option<serde_json::Value> {
    let s = f.metadata("config")?;
    let c: serde_json::Value = serde_json::from_str(s).ok()?;
    Some(c.get("vae").cloned().unwrap_or(c))
}

/// Whether `f` holds a diffusion video decoder (`CausalDiffusionVAE`).
pub fn is_diffusion_vae(f: &ModelFiles) -> bool {
    config_json(f).is_some_and(|v| v.get("_class_name").and_then(|c| c.as_str()) == Some("CausalDiffusionVAE"))
        || f.has("decoder.diff_blocks.0.attn.qkv.weight")
}

impl DiffVaeConfig {
    pub fn from_files(f: &ModelFiles) -> Result<Self> {
        let v = config_json(f).context("LTX DiffVAE: no `config` metadata")?;
        let d = v.get("decoder").context("LTX DiffVAE: no decoder config")?;
        let u = |k: &str, def: usize| d.get(k).and_then(|x| x.as_u64()).map_or(def, |x| x as usize);
        let list = |k: &str| -> Result<Vec<usize>> {
            d.get(k)
                .and_then(|x| x.as_array())
                .with_context(|| format!("LTX DiffVAE: no {k}"))?
                .iter()
                .map(|x| x.as_u64().map(|x| x as usize).context("not a number"))
                .collect()
        };
        let triple = |x: &serde_json::Value| -> Result<[usize; 3]> {
            let a = x.as_array().context("not a triple")?;
            if a.len() != 3 {
                bail!("not a triple");
            }
            let g = |i: usize| a[i].as_u64().map(|x| x as usize).context("not a number");
            Ok([g(0)?, g(1)?, g(2)?])
        };
        let kernels = d
            .get("stage_kernels")
            .and_then(|x| x.as_array())
            .context("LTX DiffVAE: no stage_kernels")?
            .iter()
            .map(triple)
            .collect::<Result<Vec<_>>>()?;
        let upsamples = d
            .get("upsamples")
            .and_then(|x| x.as_array())
            .context("LTX DiffVAE: no upsamples")?
            .iter()
            .map(|x| {
                let a = x.as_array().context("upsample")?;
                Ok((triple(&a[0])?, a.get(1).and_then(|r| r.as_u64()).unwrap_or(1) as usize))
            })
            .collect::<Result<Vec<_>>>()?;
        let stage_channels = list("stage_channels")?;
        let stage_depths = list("stage_depths")?;
        if stage_channels.len() != stage_depths.len()
            || kernels.len() != stage_channels.len()
            || upsamples.len() + 1 != stage_channels.len()
        {
            bail!("LTX DiffVAE: inconsistent stage lists");
        }
        let stage5_kernel = match d.get("stage5_kernel") {
            Some(k) => triple(k)?,
            None => [3, 7, 7],
        };
        let out = v
            .get("model_output_type")
            .or_else(|| d.get("model_output_type"))
            .and_then(|x| x.as_str());
        Ok(Self {
            in_channels: u("in_channels", 128),
            out_channels: u("out_channels", 3),
            patch: u("patch_size", 4),
            head_dim: u("head_dim", 64),
            stage_channels,
            stage_depths,
            stage_kernels: kernels,
            upsamples,
            stage5_kernel,
            timestep_mult: d.get("timestep_scale_multiplier").and_then(|x| x.as_f64()).unwrap_or(1.0) as f32,
            steps: u("default_num_inference_steps", 2).max(1),
            x0: out == Some("x0"),
        })
    }

    /// Per-axis latent size below which some stage's window would not fit
    /// (`all_stages_min_tile_size`).
    pub fn min_latent(&self) -> [usize; 3] {
        let mut cum = [1usize; 3];
        let mut mins = [1usize; 3];
        for (i, (st, _)) in self.upsamples.iter().enumerate() {
            for a in 0..3 {
                mins[a] = mins[a].max(self.stage_kernels[i][a].div_ceil(cum[a]));
            }
            for a in 0..3 {
                cum[a] *= st[a];
            }
        }
        for a in 0..3 {
            mins[a] = mins[a].max(self.stage5_kernel[a].div_ceil(cum[a]));
        }
        mins
    }
}

/// `default_rope_dim_split`: the head's dimensions for time, height and width.
fn rope_split(hd: usize) -> [usize; 3] {
    let mut dt = (hd / 4) / 2 * 2;
    let mut dhw = (hd - dt) / 2;
    if !dhw.is_multiple_of(2) {
        dt -= 2;
        dhw = (hd - dt) / 2;
    }
    [dt, dhw, dhw]
}

/// `rope_inv_freqs(dim)`: `1 / 10000^(i / dim)` for even `i`.
fn inv_freqs(dim: usize) -> Vec<f32> {
    (0..dim / 2)
        .map(|i| (1.0 / 10000f64.powf((2 * i) as f64 / dim as f64)) as f32)
        .collect()
}

/// Cosines and sines of `pos * inv` (float32, as the reference computes them).
fn angles(pos: &[f32], inv: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut c = Vec::with_capacity(pos.len() * inv.len());
    let mut s = Vec::with_capacity(pos.len() * inv.len());
    for &p in pos {
        for &f in inv {
            let a = p * f;
            c.push(a.cos());
            s.push(a.sin());
        }
    }
    (c, s)
}

/// Absolute 3D RoPE tables: per axis, cos / sin of each position.
struct Rope3 {
    split: [usize; 3],
    tabs: [(Vec<f32>, Vec<f32>); 3],
}

impl Rope3 {
    fn new(hd: usize, pos: [&[f32]; 3]) -> Self {
        let split = rope_split(hd);
        let tabs = std::array::from_fn(|a| angles(pos[a], &inv_freqs(split[a])));
        Self { split, tabs }
    }

    /// Rotates every head of one row at positions `(t, h, w)` (indices into the tables);
    /// pairs are adjacent values.
    fn apply(&self, row: &mut [f32], hd: usize, idx: [usize; 3]) {
        for head in row.chunks_mut(hd) {
            let mut off = 0;
            for a in 0..3 {
                let d = self.split[a];
                let half = d / 2;
                let (c, s) = (&self.tabs[a].0[idx[a] * half..][..half], &self.tabs[a].1[idx[a] * half..][..half]);
                for i in 0..half {
                    let (xe, xo) = (head[off + 2 * i], head[off + 2 * i + 1]);
                    head[off + 2 * i] = xe * c[i] - xo * s[i];
                    head[off + 2 * i + 1] = xe * s[i] + xo * c[i];
                }
                off += d;
            }
        }
    }
}

/// `natten.na3d` window start on one axis: centered, shifted inward at the borders.
#[inline]
fn win_start(i: usize, off: usize, k: usize, len: usize) -> usize {
    (i + off).saturating_sub(k / 2).min(len - k)
}

/// 3D neighborhood attention: queries `q` over a `qd` grid, keys / values over a `kd` grid
/// (query `i` centered at key `i + off` per axis), `heads` heads of `hd` (rows `heads * hd`),
/// queries already scaled.
#[allow(clippy::too_many_arguments)]
pub(crate) fn na3d_cpu(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    qd: [usize; 3],
    kd: [usize; 3],
    off: [usize; 3],
    ker: [usize; 3],
    heads: usize,
    hd: usize,
) -> Vec<f32> {
    let dim = heads * hd;
    let n = qd[0] * qd[1] * qd[2];
    let ker: [usize; 3] = std::array::from_fn(|a| ker[a].min(kd[a]));
    let nk = ker[0] * ker[1] * ker[2];
    let mut out = vec![0f32; n * dim];
    // Rows of 16 queries along the width per task.
    let rows = 16;
    for_chunks(&mut out, rows * dim, n > rows, |ci, o| {
        let mut sc = vec![0f32; nk];
        for (r, orow) in o.chunks_mut(dim).enumerate() {
            let qi = ci * rows + r;
            let (t, h, w) = (qi / (qd[1] * qd[2]), (qi / qd[2]) % qd[1], qi % qd[2]);
            let s = [
                win_start(t, off[0], ker[0], kd[0]),
                win_start(h, off[1], ker[1], kd[1]),
                win_start(w, off[2], ker[2], kd[2]),
            ];
            for hh in 0..heads {
                let qv = &q[qi * dim + hh * hd..][..hd];
                let mut j = 0;
                let mut mx = f32::NEG_INFINITY;
                for a in 0..ker[0] {
                    for b in 0..ker[1] {
                        let base = ((s[0] + a) * kd[1] + s[1] + b) * kd[2] + s[2];
                        for c in 0..ker[2] {
                            let x = dot(qv, &k[(base + c) * dim + hh * hd..][..hd]);
                            sc[j] = x;
                            mx = mx.max(x);
                            j += 1;
                        }
                    }
                }
                let mut sum = 0f32;
                for x in sc.iter_mut() {
                    *x = (*x - mx).exp();
                    sum += *x;
                }
                let acc = &mut orow[hh * hd..(hh + 1) * hd];
                let mut j = 0;
                for a in 0..ker[0] {
                    for b in 0..ker[1] {
                        let base = ((s[0] + a) * kd[1] + s[1] + b) * kd[2] + s[2];
                        for c in 0..ker[2] {
                            let p = sc[j] / sum;
                            let vr = &v[(base + c) * dim + hh * hd..][..hd];
                            acc.iter_mut().zip(vr).for_each(|(o, &x)| *o += p * x);
                            j += 1;
                        }
                    }
                }
            }
        }
    });
    out
}

/// Neighborhood attention on the accelerator when it has a kernel for it, else on the CPU.
#[allow(clippy::too_many_arguments)]
fn na3d(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    qd: [usize; 3],
    kd: [usize; 3],
    off: [usize; 3],
    ker: [usize; 3],
    heads: usize,
    hd: usize,
) -> Vec<f32> {
    let ker: [usize; 3] = std::array::from_fn(|a| ker[a].min(kd[a]));
    if let Some(be) = crate::gpu::active() {
        let mut out = vec![0f32; qd.iter().product::<usize>() * heads * hd];
        if be.na3d(q, k, v, qd, kd, off, ker, heads, hd, &mut out) {
            return out;
        }
    }
    na3d_cpu(q, k, v, qd, kd, off, ker, heads, hd)
}

fn rms_w(x: &[f32], n: usize, w: &[f32]) -> Vec<f32> {
    let mut y = x.to_vec();
    crate::kernels::rmsnorm_f32w_rows_inplace(&mut y, n, w, 1e-6);
    y
}

/// `x += y` (both rows of the same width).
fn add_into(x: &mut [f32], y: &[f32]) {
    let n = 1 << 16;
    for_chunks(x, n, x.len() > n, |i, r| r.iter_mut().zip(&y[i * n..]).for_each(|(a, &b)| *a += b));
}

/// A 3D neighborhood attention (`NeighborhoodAttention3D`): fused q / k / v projection, RMS
/// norms of each head's queries and keys, absolute RoPE, the output projection.
struct NaAttn<'a> {
    qkv: DLin<'a>,
    proj: DLin<'a>,
    qn: Vec<f32>,
    kn: Vec<f32>,
    dim: usize,
    hd: usize,
    kernel: [usize; 3],
}

impl<'a> NaAttn<'a> {
    fn load(f: &'a ModelFiles, p: &str, dim: usize, hd: usize, kernel: [usize; 3]) -> Result<Self> {
        Ok(Self {
            qkv: DLin::load(f, &format!("{p}.qkv"), 3 * dim, dim, true)?,
            proj: DLin::load(f, &format!("{p}.proj"), dim, dim, true)?,
            qn: f.f32(&format!("{p}.q_norm.weight"))?,
            kn: f.f32(&format!("{p}.k_norm.weight"))?,
            dim,
            hd,
            kernel,
        })
    }

    /// The attention's output (projected, bias included) of rows `y` over the `d` grid. With
    /// `replicate_w`, the keys extend half a window past both width edges (the edge columns
    /// repeated at their own positions).
    fn forward(&self, y: &[f32], d: [usize; 3], replicate_w: bool) -> Vec<f32> {
        let (dim, hd) = (self.dim, self.hd);
        let n = d[0] * d[1] * d[2];
        let heads = dim / hd;
        let mut pr = Prof::new("attn");
        let qkv = self.qkv.forward(y, n);
        pr.lap("qkv");
        let mut q = vec![0f32; n * dim];
        let mut k = vec![0f32; n * dim];
        let mut v = vec![0f32; n * dim];
        let split = |dst: &mut Vec<f32>, part: usize| {
            for_chunks(dst, dim * 64, n > 64, |ci, c| {
                for (r, row) in c.chunks_mut(dim).enumerate() {
                    let i = ci * 64 + r;
                    row.copy_from_slice(&qkv[i * 3 * dim + part * dim..][..dim]);
                }
            });
        };
        split(&mut q, 0);
        split(&mut k, 1);
        split(&mut v, 2);
        drop(qkv);
        let scale = 1.0 / (hd as f32).sqrt();
        let halo = if replicate_w { self.kernel[2] / 2 } else { 0 };
        let kw = d[2] + 2 * halo;
        let pos = |len: usize, from: i64| -> Vec<f32> { (0..len).map(|i| (i as i64 + from) as f32).collect() };
        let (pt, ph) = (pos(d[0], 0), pos(d[1], 0));
        let pw = pos(kw, -(halo as i64));
        let rope = Rope3::new(hd, [&pt, &ph, &pw]);
        let (qn, kn) = (&self.qn, &self.kn);
        for_chunks(&mut q, dim * 64, n > 64, |ci, c| {
            crate::kernels::rmsnorm_f32w_rows_inplace(c, hd, qn, 1e-6);
            for (r, row) in c.chunks_mut(dim).enumerate() {
                let i = ci * 64 + r;
                row.iter_mut().for_each(|x| *x *= scale);
                rope.apply(row, hd, [i / (d[1] * d[2]), (i / d[2]) % d[1], i % d[2] + halo]);
            }
        });
        crate::kernels::rmsnorm_f32w_rows_inplace(&mut k, hd, kn, 1e-6);
        let (k, v) = if halo == 0 {
            for_chunks(&mut k, dim * 64, n > 64, |ci, c| {
                for (r, row) in c.chunks_mut(dim).enumerate() {
                    let i = ci * 64 + r;
                    rope.apply(row, hd, [i / (d[1] * d[2]), (i / d[2]) % d[1], i % d[2]]);
                }
            });
            (k, v)
        } else {
            let nk = d[0] * d[1] * kw;
            let mut kp = vec![0f32; nk * dim];
            let mut vp = vec![0f32; nk * dim];
            for_chunks(&mut kp, dim * kw, nk > kw, |ri, c| {
                for (j, row) in c.chunks_mut(dim).enumerate() {
                    let src = ri * d[2] + j.saturating_sub(halo).min(d[2] - 1);
                    row.copy_from_slice(&k[src * dim..][..dim]);
                    rope.apply(row, hd, [ri / d[1], ri % d[1], j]);
                }
            });
            for_chunks(&mut vp, dim * kw, nk > kw, |ri, c| {
                for (j, row) in c.chunks_mut(dim).enumerate() {
                    let src = ri * d[2] + j.saturating_sub(halo).min(d[2] - 1);
                    row.copy_from_slice(&v[src * dim..][..dim]);
                }
            });
            (kp, vp)
        };
        let kd = [d[0], d[1], kw];
        pr.lap("prep");
        let o = na3d(&q, &k, &v, d, kd, [0, 0, halo], self.kernel, heads, hd);
        drop((q, k, v));
        pr.lap("na");
        let o = self.proj.forward(&o, n);
        pr.lap("proj");
        o
    }
}

/// `d` as the device sees it (None: no device state for it).
fn dev_lin<'s>(d: &'s DLin) -> Option<crate::gpu::dit::Lin<'s>> {
    if d.w.fold.is_some() || d.w.has_lora() {
        return None;
    }
    Some(crate::gpu::dit::Lin {
        w: d.w.as_ref(),
        dev: d.w.device_weight_unfolded(d.n_out, d.n_in)?,
        n_out: d.n_out,
        n_in: d.n_in,
        bias: d.b.as_deref(),
        side: None,
        rot: None,
    })
}

/// A block as the device sees it.
#[allow(clippy::too_many_arguments)]
fn dev_block<'s>(
    attn: &'s NaAttn,
    mlp: &'s SwiGlu,
    norm1: &'s [f32],
    norm2: &'s [f32],
    replicate_w: bool,
    ctx: Option<&'s DLin>,
    mods: Option<[&'s [f32]; 4]>,
) -> Option<crate::gpu::dit::NaBlock<'s>> {
    Some(crate::gpu::dit::NaBlock {
        dim: attn.dim,
        kernel: attn.kernel,
        replicate_w,
        norm1,
        norm2,
        qkv: dev_lin(&attn.qkv)?,
        q_norm: &attn.qn,
        k_norm: &attn.kn,
        proj: dev_lin(&attn.proj)?,
        gate: dev_lin(&mlp.gate)?,
        up: dev_lin(&mlp.up)?,
        down: dev_lin(&mlp.down)?,
        ctx: match ctx {
            Some(c) => Some(dev_lin(c)?),
            None => None,
        },
        mods,
    })
}

/// SwiGLU without biases: `down(silu(gate x) * up x)`.
struct SwiGlu<'a> {
    gate: DLin<'a>,
    up: DLin<'a>,
    down: DLin<'a>,
}

impl<'a> SwiGlu<'a> {
    fn load(f: &'a ModelFiles, p: &str, dim: usize) -> Result<Self> {
        let hidden = (dim * 4).div_ceil(16) * 16;
        Ok(Self {
            gate: DLin::load(f, &format!("{p}.w_gate"), hidden, dim, false)?,
            up: DLin::load(f, &format!("{p}.w_up"), hidden, dim, false)?,
            down: DLin::load(f, &format!("{p}.w_down"), dim, hidden, false)?,
        })
    }

    /// `x += swiglu(norm(x))` over row chunks (bounded memory), `norm` producing the input.
    fn residual(&self, x: &mut [f32], dim: usize, norm: &dyn Fn(&[f32]) -> Vec<f32>) {
        let n = x.len() / dim;
        let chunk = 1 << 16;
        let mut r0 = 0;
        while r0 < n {
            let r1 = (r0 + chunk).min(n);
            let xs = &mut x[r0 * dim..r1 * dim];
            let y = norm(xs);
            let rows = r1 - r0;
            let ys = crate::ltx::nn::lins(&[&self.gate, &self.up], &y, rows);
            let (mut g, u) = (ys[0].clone(), &ys[1]);
            crate::kernels::silu_mul(&mut g, u);
            let o = self.down.forward(&g, rows);
            add_into(xs, &o);
            r0 = r1;
        }
    }
}

/// A deterministic stage's block (`NABlock`): `x + attn(norm1 x)`, then `x + mlp(norm2 x)`.
struct NaBlock<'a> {
    norm1: Vec<f32>,
    norm2: Vec<f32>,
    attn: NaAttn<'a>,
    mlp: SwiGlu<'a>,
}

impl<'a> NaBlock<'a> {
    fn load(f: &'a ModelFiles, p: &str, dim: usize, hd: usize, kernel: [usize; 3]) -> Result<Self> {
        Ok(Self {
            norm1: f.f32(&format!("{p}.norm1.weight"))?,
            norm2: f.f32(&format!("{p}.norm2.weight"))?,
            attn: NaAttn::load(f, &format!("{p}.attn"), dim, hd, kernel)?,
            mlp: SwiGlu::load(f, &format!("{p}.mlp"), dim)?,
        })
    }

    fn forward(&self, x: &mut Vid) {
        let d = [x.t, x.h, x.w];
        let c = x.c;
        let y = rms_w(&x.data, c, &self.norm1);
        let a = self.attn.forward(&y, d, false);
        drop(y);
        add_into(&mut x.data, &a);
        drop(a);
        let w2 = &self.norm2;
        self.mlp.residual(&mut x.data, c, &|xs| rms_w(xs, c, w2));
    }
}

/// A stage-5 block (`DiffusionNABlock`, the ungated AdaLN-Zero form): the context projected and
/// added, modulated attention and SwiGLU.
struct DiffBlock<'a> {
    ctx: DLin<'a>,
    /// `[7, dim]`: scale / shift / gate of the attention, of the MLP, the context gate.
    sst: Vec<f32>,
    norm1: Vec<f32>,
    norm2: Vec<f32>,
    attn: NaAttn<'a>,
    mlp: SwiGlu<'a>,
}

/// `rms(x) * w * (1 + scale) + shift` over rows of `n`.
fn modulated(x: &[f32], n: usize, w: &[f32], scale: &[f32], shift: &[f32]) -> Vec<f32> {
    let mut y = rms_w(x, n, w);
    let par = y.len() > n * 64;
    for_chunks(&mut y, n * 64, par, |_, c| {
        for row in c.chunks_mut(n) {
            for ((v, &s), &h) in row.iter_mut().zip(scale).zip(shift) {
                *v = *v * (1.0 + s) + h;
            }
        }
    });
    y
}

impl<'a> DiffBlock<'a> {
    /// Scale / shift of the attention's norm and of the MLP's: the shared modulation `m` plus
    /// the block's table.
    fn mods(&self, m: &[f32]) -> [Vec<f32>; 4] {
        let c = self.attn.dim;
        let row = |i: usize| -> Vec<f32> { (0..c).map(|j| m[i * c + j] + self.sst[i * c + j]).collect() };
        [row(0), row(1), row(3), row(4)]
    }

    fn forward(&self, x: &mut Vid, ctx: &[f32], m: &[f32]) {
        let c = x.c;
        let d = [x.t, x.h, x.w];
        let n = d[0] * d[1] * d[2];
        let [sc_a, sh_a, sc_m, sh_m] = self.mods(m);
        let p = self.ctx.forward(ctx, n);
        add_into(&mut x.data, &p);
        drop(p);
        let y = modulated(&x.data, c, &self.norm1, &sc_a, &sh_a);
        let a = self.attn.forward(&y, d, true);
        drop(y);
        add_into(&mut x.data, &a);
        drop(a);
        let w2 = &self.norm2;
        self.mlp.residual(&mut x.data, c, &|xs| modulated(xs, c, w2, &sc_m, &sh_m));
    }
}

/// `LinearPixelShuffleUpsample`: a linear layer then depth-to-space (channels `(c, t, h, w)`).
struct Upsample<'a> {
    proj: DLin<'a>,
    stride: [usize; 3],
    out: usize,
}

impl Upsample<'_> {
    fn forward(&self, x: &Vid, drop_first: bool) -> Vid {
        let [p1, p2, p3] = self.stride;
        let n = x.t * x.h * x.w;
        let y = self.proj.forward(&x.data, n);
        let oc = self.out;
        let pc = self.proj.n_out;
        let mut o = Vid::new(x.t * p1, x.h * p2, x.w * p3, oc);
        let (oh, ow) = (o.h, o.w);
        let fr = oh * ow * oc;
        for_chunks(&mut o.data, fr, o.t > 1, |ot, frame| {
            let (t, a) = (ot / p1, ot % p1);
            for oy in 0..oh {
                let (h, b) = (oy / p2, oy % p2);
                for ox in 0..ow {
                    let (w, dd) = (ox / p3, ox % p3);
                    let src = &y[((t * x.h + h) * x.w + w) * pc..][..pc];
                    let dst = &mut frame[(oy * ow + ox) * oc..][..oc];
                    for (ch, v) in dst.iter_mut().enumerate() {
                        *v = src[((ch * p1 + a) * p2 + b) * p3 + dd];
                    }
                }
            }
        });
        if p1 == 2 && drop_first {
            super::vae::drop_first_frame(&mut o);
        }
        o
    }
}

pub struct DiffDecoder<'a> {
    pub cfg: DiffVaeConfig,
    mean: Vec<f32>,
    std: Vec<f32>,
    conv_in: DLin<'a>,
    det: Vec<Vec<NaBlock<'a>>>,
    ups: Vec<Upsample<'a>>,
    t1: DLin<'a>,
    t2: DLin<'a>,
    adaln: DLin<'a>,
    conv_in_x: DLin<'a>,
    blocks: Vec<DiffBlock<'a>>,
    norm_out: Vec<f32>,
    conv_out: DLin<'a>,
    /// Run the blocks on the accelerator's session when there is one (`OXINFER_DIFFVAE_GPU=0`
    /// keeps them on the CPU).
    pub device: bool,
}

/// Pads or crops axis `axis` (0: t, 1: h, 2: w) of `x` to `size`: `repeat_last` repeats or drops
/// the end, else symmetric (`before = need / 2`), edges replicated. Returns the rows before.
fn resize_axis(x: &Vid, axis: usize, size: usize, repeat_last: bool) -> (Vid, usize) {
    let len = [x.t, x.h, x.w][axis];
    if len == size {
        return (x.clone(), 0);
    }
    let before = if repeat_last {
        0
    } else if size > len {
        (size - len) / 2
    } else {
        (len - size) / 2
    };
    let mut d = [x.t, x.h, x.w];
    d[axis] = size;
    let mut o = Vid::new(d[0], d[1], d[2], x.c);
    let c = x.c;
    for t in 0..d[0] {
        for h in 0..d[1] {
            for w in 0..d[2] {
                let mut s = [t, h, w];
                // Source index along the axis.
                let i = s[axis] as i64 + if size > len { -(before as i64) } else { before as i64 };
                s[axis] = i.clamp(0, len as i64 - 1) as usize;
                let src = ((s[0] * x.h + s[1]) * x.w + s[2]) * c;
                let dst = ((t * d[1] + h) * d[2] + w) * c;
                o.data[dst..dst + c].copy_from_slice(&x.data[src..src + c]);
            }
        }
    }
    (o, before)
}

impl<'a> DiffDecoder<'a> {
    pub fn load(f: &'a ModelFiles) -> Result<Self> {
        let cfg = DiffVaeConfig::from_files(f)?;
        let pfx = if f.has("decoder.conv_in.weight") { "" } else { "vae." };
        let n = |s: &str| format!("{pfx}decoder.{s}");
        let ch = &cfg.stage_channels;
        let hd = cfg.head_dim;
        let stats = |k: &str| -> Result<Vec<f32>> {
            let name = format!("per_channel_statistics.{k}");
            f.f32(&name).or_else(|_| f.f32(&format!("vae.{name}")))
        };
        let (mean, std) = (stats("mean-of-means")?, stats("std-of-means")?);
        let mut det = Vec::new();
        let mut ups = Vec::new();
        for (i, (st, red)) in cfg.upsamples.iter().enumerate() {
            let c = ch[i];
            det.push(
                (0..cfg.stage_depths[i])
                    .map(|j| NaBlock::load(f, &n(&format!("det_stages.{i}.{j}")), c, hd, cfg.stage_kernels[i]))
                    .collect::<Result<Vec<_>>>()?,
            );
            let pc = st.iter().product::<usize>() * c / red;
            ups.push(Upsample {
                proj: DLin::load(f, &n(&format!("upsamples.{i}.proj")), pc, c, true)?,
                stride: *st,
                out: pc / st.iter().product::<usize>(),
            });
        }
        let c5 = *ch.last().context("no stages")?;
        let ctx_c = c5;
        let pix = cfg.out_channels * cfg.patch * cfg.patch;
        let temb = f.info(&n("t_embedder.mlp.0.weight")).context("LTX DiffVAE: no t_embedder")?.shape[0];
        let blocks = (0..*cfg.stage_depths.last().unwrap())
            .map(|j| -> Result<DiffBlock> {
                let p = n(&format!("diff_blocks.{j}"));
                Ok(DiffBlock {
                    ctx: DLin::load(f, &format!("{p}.context_proj"), c5, ctx_c, true)?,
                    sst: f.f32(&format!("{p}.scale_shift_table"))?,
                    norm1: f.f32(&format!("{p}.norm1.weight"))?,
                    norm2: f.f32(&format!("{p}.norm2.weight"))?,
                    attn: NaAttn::load(f, &format!("{p}.attn"), c5, hd, cfg.stage5_kernel)?,
                    mlp: SwiGlu::load(f, &format!("{p}.mlp"), c5)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            conv_in: DLin::load(f, &n("conv_in"), ch[0], cfg.in_channels, true)?,
            t1: DLin::load(f, &n("t_embedder.mlp.0"), temb, 256, true)?,
            t2: DLin::load(f, &n("t_embedder.mlp.2"), temb, temb, true)?,
            adaln: DLin::load(f, &n("shared_adaln.proj"), 7 * c5, temb, true)?,
            conv_in_x: DLin::load(f, &n("conv_in_x_t"), c5, pix, true)?,
            norm_out: f.f32(&n("norm_out.weight"))?,
            conv_out: DLin::load(f, &n("conv_out"), pix, c5, true)?,
            mean,
            std,
            det,
            ups,
            blocks,
            cfg,
            device: true,
        })
    }

    /// The accelerator's block session for one decode, when blocks run there.
    fn session(&self) -> Option<Box<dyn crate::gpu::dit::NaSession>> {
        if !self.device || std::env::var("OXINFER_DIFFVAE_GPU").is_ok_and(|v| v == "0") {
            return None;
        }
        crate::gpu::active()?.na_session()
    }

    /// Deterministic stage `i`'s blocks over `v`.
    fn det_stage(&self, i: usize, v: &mut Vid, sess: &mut Option<Box<dyn crate::gpu::dit::NaSession>>, tick: &mut dyn FnMut(usize)) {
        let blocks = &self.det[i];
        if let Some(s) = sess.as_mut() {
            let dev: Option<Vec<_>> = blocks
                .iter()
                .map(|b| dev_block(&b.attn, &b.mlp, &b.norm1, &b.norm2, false, None, None))
                .collect();
            if let Some(dev) = dev
                && s.blocks(&dev, &mut v.data, None, [v.t, v.h, v.w])
            {
                tick(blocks.len());
                return;
            }
        }
        for b in blocks {
            b.forward(v);
            tick(1);
        }
    }

    /// Stage 5's blocks over `x` with context `ctx` and the shared modulation `m`.
    fn diff_stage(
        &self,
        x: &mut Vid,
        ctx: &Vid,
        m: &[f32],
        sess: &mut Option<Box<dyn crate::gpu::dit::NaSession>>,
        tick: &mut dyn FnMut(usize),
    ) {
        if let Some(s) = sess.as_mut() {
            let mods: Vec<[Vec<f32>; 4]> = self.blocks.iter().map(|b| b.mods(m)).collect();
            let dev: Option<Vec<_>> = self
                .blocks
                .iter()
                .zip(&mods)
                .map(|(b, md)| {
                    dev_block(
                        &b.attn,
                        &b.mlp,
                        &b.norm1,
                        &b.norm2,
                        true,
                        Some(&b.ctx),
                        Some([&md[0][..], &md[1][..], &md[2][..], &md[3][..]]),
                    )
                })
                .collect();
            if let Some(dev) = dev
                && s.blocks(&dev, &mut x.data, Some(&ctx.data), [x.t, x.h, x.w])
            {
                tick(self.blocks.len());
                return;
            }
        }
        for b in &self.blocks {
            b.forward(x, &ctx.data, m);
            tick(1);
        }
    }

    /// Pixel noise values `decode` draws for latents of `t x h x w` (one draw).
    pub fn noise_len(&self, t: usize, h: usize, w: usize) -> usize {
        let g = self.geometry(t, h, w);
        self.cfg.out_channels * g.f5 * g.h5 * g.w5
    }

    fn geometry(&self, t: usize, h: usize, w: usize) -> Geometry {
        let mins = self.cfg.min_latent();
        let (tp, hp, wp) = (t.max(mins[0]), h.max(mins[1]), w.max(mins[2]));
        let (mut s4t, mut s4h, mut s4w) = (tp, hp, wp);
        for (st, _) in &self.cfg.upsamples[..3] {
            s4t = s4t * st[0] - usize::from(st[0] == 2);
            s4h *= st[1];
            s4w *= st[2];
        }
        let st = self.cfg.upsamples[3].0;
        let p = self.cfg.patch;
        let mut f5 = if st[0] == 2 { s4t * 2 - 1 } else { s4t * st[0] };
        f5 = f5.max(self.cfg.stage5_kernel[0]);
        Geometry {
            pad: [tp, hp, wp],
            f5,
            h5: s4h * st[1] * p,
            w5: s4w * st[2] * p,
        }
    }

    /// Frames in `[-1, 1]` (`[t, h, w, 3]` rows: `(t - 1) * 8 + 1` frames of 32 times the
    /// latent's size) from latents `[c, t, h, w]`; `noise(n)` gives the `n` standard normal values
    /// of the starting pixels (`[3, frames, h, w]`, the reference's draw order).
    pub fn decode(
        &self,
        lat: &[f32],
        t: usize,
        h: usize,
        w: usize,
        noise: &mut dyn FnMut(usize) -> Vec<f32>,
        progress: &mut dyn FnMut(usize, usize),
    ) -> Vid {
        let cfg = &self.cfg;
        let c = cfg.in_channels;
        let g = self.geometry(t, h, w);
        let nd = cfg.stage_depths.len() - 1;
        let total = cfg.stage_depths[..nd].iter().sum::<usize>() + cfg.stage_depths[nd] * cfg.steps.max(1) + 1;
        let mut done = 0;
        let mut tick = |k: usize| {
            done += k;
            progress(done.min(total), total);
        };
        // Un-normalized latents, padded to every stage's window (time: the last frame repeated;
        // space: the edges, half before), then the last frame twice more (NATTEN's border).
        let mut x = Vid::from_cthw(lat, c, t, h, w);
        for row in x.data.chunks_mut(c) {
            for (ch, v) in row.iter_mut().enumerate() {
                *v = *v * self.std[ch] + self.mean[ch];
            }
        }
        let (x1, _) = resize_axis(&x, 0, g.pad[0], true);
        let (x1, h_before) = resize_axis(&x1, 1, g.pad[1], false);
        let (x1, w_before) = resize_axis(&x1, 2, g.pad[2], false);
        let ghost = (cfg.stage_kernels[0][0] / 2) * 2;
        let (x1, _) = resize_axis(&x1, 0, g.pad[0] + ghost, true);
        drop(x);
        let n = x1.t * x1.h * x1.w;
        let mut v = Vid {
            data: self.conv_in.forward(&x1.data, n),
            c: cfg.stage_channels[0],
            ..x1
        };
        // Stages 1 to 4.
        let mut pr = Prof::new("stage");
        let mut sess = self.session();
        for (i, up) in self.ups.iter().enumerate() {
            self.det_stage(i, &mut v, &mut sess, &mut tick);
            v = up.forward(&v, true);
            pr.lap(["1", "2", "3", "4"][i.min(3)]);
            if i == 3 {
                // The ghost frames off the context, keeping at least stage 5's window.
                let tscale: usize = cfg.upsamples.iter().map(|u| u.0[0]).product();
                let content = v.t.saturating_sub(ghost * tscale).max(1);
                let keep = v.t.min(content.max(cfg.stage5_kernel[0]));
                v = resize_axis(&v, 0, keep, true).0;
            }
        }
        let ctx = v;
        // Stage 5 over the patchified pixels.
        let p = cfg.patch;
        let oc = cfg.out_channels;
        let (f5, h5, w5) = (g.f5, g.h5, g.w5);
        if ctx.t != f5 || ctx.h * p != h5 || ctx.w * p != w5 {
            // The geometry above mirrors the stages; a mismatch would be a bug.
            panic!("LTX DiffVAE: context {}x{}x{} for pixels {f5}x{h5}x{w5}", ctx.t, ctx.h, ctx.w);
        }
        let mut xt = noise(oc * f5 * h5 * w5);
        let ts: Vec<f32> = (0..cfg.steps)
            .map(|i| 1.0 - i as f32 * (1.0 - 1.0 / cfg.steps as f32) / (cfg.steps.max(2) - 1) as f32)
            .collect();
        let mut pixels = Vec::new();
        for (si, &tnow) in ts.iter().enumerate() {
            let out = self.step(&ctx, &xt, tnow, &mut sess, &mut tick);
            pr.lap("5");
            let tnext = ts.get(si + 1).copied().unwrap_or(0.0);
            if si + 1 == ts.len() && cfg.x0 {
                pixels = out;
                break;
            }
            // Euler to the next timestep with the velocity (from x0 when the model predicts it).
            for (x, o) in xt.iter_mut().zip(&out) {
                let vel = if cfg.x0 { (*x - o) / tnow } else { *o };
                *x -= (tnow - tnext) * vel;
            }
            if si + 1 == ts.len() {
                pixels = std::mem::take(&mut xt);
            }
        }
        // Crop to the content: frames from the start, height / width past the padding.
        let (fo, ho, wo) = ((t - 1) * 8 + 1, h * 32, w * 32);
        let (hb, wb) = (h_before * 32, w_before * 32);
        let mut out = Vid::new(fo, ho, wo, oc);
        let fr = ho * wo * oc;
        for_chunks(&mut out.data, fr, fo > 1, |f, frame| {
            let sf = f.min(f5 - 1);
            for y in 0..ho {
                for x in 0..wo {
                    for ch in 0..oc {
                        frame[(y * wo + x) * oc + ch] = pixels[((ch * f5 + sf) * h5 + y + hb) * w5 + x + wb];
                    }
                }
            }
        });
        tick(1);
        out
    }

    /// One stage-5 step: the model's output over pixels `xt` (`[3, f, h, w]`) at timestep `t`.
    fn step(
        &self,
        ctx: &Vid,
        xt: &[f32],
        t: f32,
        sess: &mut Option<Box<dyn crate::gpu::dit::NaSession>>,
        tick: &mut dyn FnMut(usize),
    ) -> Vec<f32> {
        let cfg = &self.cfg;
        let p = cfg.patch;
        let oc = cfg.out_channels;
        let (f5, h5, w5) = (ctx.t, ctx.h * p, ctx.w * p);
        let pc = oc * p * p;
        let n = ctx.t * ctx.h * ctx.w;
        // Patchify: channel (c, w offset, h offset).
        let mut patches = vec![0f32; n * pc];
        let (hh, ww) = (ctx.h, ctx.w);
        for_chunks(&mut patches, hh * ww * pc, f5 > 1, |f, fr| {
            for y in 0..hh {
                for x in 0..ww {
                    let row = &mut fr[(y * ww + x) * pc..][..pc];
                    for c in 0..oc {
                        for r in 0..p {
                            for q in 0..p {
                                row[(c * p + r) * p + q] = xt[((c * f5 + f) * h5 + y * p + q) * w5 + x * p + r];
                            }
                        }
                    }
                }
            }
        });
        let c5 = *cfg.stage_channels.last().unwrap();
        let mut x = Vid {
            t: ctx.t,
            h: ctx.h,
            w: ctx.w,
            c: c5,
            data: self.conv_in_x.forward(&patches, n),
        };
        drop(patches);
        // The shared timestep modulation.
        let s = super::nn::timestep_sinusoid(t * cfg.timestep_mult);
        let mut e = self.t1.forward(&s, 1);
        e.iter_mut().for_each(|v| *v = crate::sd::ops::silu(*v));
        let e = self.t2.forward(&e, 1);
        let act: Vec<f32> = e.iter().map(|&v| crate::sd::ops::silu(v)).collect();
        let m = self.adaln.forward(&act, 1);
        self.diff_stage(&mut x, ctx, &m, sess, tick);
        let y = rms_w(&x.data, c5, &self.norm_out);
        drop(x);
        let o = self.conv_out.forward(&y, n);
        // Un-patchify into `[3, f, h, w]`.
        let mut px = vec![0f32; oc * f5 * h5 * w5];
        for_chunks(&mut px, h5 * w5, true, |cf, plane| {
            let (c, f) = (cf / f5, cf % f5);
            for y in 0..hh {
                for x in 0..ww {
                    let row = &o[((f * hh + y) * ww + x) * pc..][..pc];
                    for r in 0..p {
                        for q in 0..p {
                            plane[(y * p + q) * w5 + x * p + r] = row[(c * p + r) * p + q];
                        }
                    }
                }
            }
        });
        px
    }
}

/// Timings of the decode's parts on stderr with `OXINFER_DIFFVAE_PROF=1`.
struct Prof {
    on: bool,
    what: &'static str,
    t: crate::clock::Instant,
}

impl Prof {
    fn new(what: &'static str) -> Self {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let on = *ON.get_or_init(|| std::env::var("OXINFER_DIFFVAE_PROF").is_ok_and(|v| v != "0"));
        Self {
            on,
            what,
            t: crate::clock::Instant::now(),
        }
    }

    fn lap(&mut self, part: &str) {
        if self.on {
            eprintln!("[diffvae {} {part}: {:.3}s]", self.what, self.t.elapsed().as_secs_f64());
            self.t = crate::clock::Instant::now();
        }
    }
}

struct Geometry {
    /// The latent's size after the window padding.
    pad: [usize; 3],
    f5: usize,
    h5: usize,
    w5: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_split_of_64() {
        assert_eq!(rope_split(64), [16, 24, 24]);
    }

    /// Neighborhood attention against masked dense attention over the same windows (NATTEN's
    /// clamped windows; keys extended past the width edges with an offset).
    #[test]
    fn na3d_matches_masked_attention() {
        let (heads, hd) = (2, 64);
        let dim = heads * hd;
        let qd = [4, 5, 9];
        let (halo, ker) = (2, [3, 3, 5]);
        let kd = [4, 5, 9 + 2 * halo];
        let mut g = crate::sd::ops::Rng::new(3);
        let nq: usize = qd.iter().product();
        let nk: usize = kd.iter().product();
        let (q, k, v) = (g.normals(nq * dim), g.normals(nk * dim), g.normals(nk * dim));
        let out = na3d_cpu(&q, &k, &v, qd, kd, [0, 0, halo], ker, heads, hd);
        for qi in 0..nq {
            let c = [qi / (qd[1] * qd[2]), (qi / qd[2]) % qd[1], qi % qd[2]];
            let lo: Vec<usize> = (0..3)
                .map(|a| {
                    let off = if a == 2 { halo } else { 0 };
                    ((c[a] + off) as i64 - (ker[a] / 2) as i64).clamp(0, (kd[a] - ker[a]) as i64) as usize
                })
                .collect();
            for h in 0..heads {
                let mut w = Vec::new();
                for ki in 0..nk {
                    let kc = [ki / (kd[1] * kd[2]), (ki / kd[2]) % kd[1], ki % kd[2]];
                    if (0..3).all(|a| kc[a] >= lo[a] && kc[a] < lo[a] + ker[a]) {
                        let s: f32 = (0..hd).map(|i| q[qi * dim + h * hd + i] * k[ki * dim + h * hd + i]).sum();
                        w.push((ki, s));
                    }
                }
                let mx = w.iter().map(|x| x.1).fold(f32::NEG_INFINITY, f32::max);
                let z: f32 = w.iter().map(|x| (x.1 - mx).exp()).sum();
                for i in 0..hd {
                    let want: f32 = w.iter().map(|&(ki, s)| (s - mx).exp() / z * v[ki * dim + h * hd + i]).sum();
                    let got = out[qi * dim + h * hd + i];
                    assert!((want - got).abs() < 1e-4, "query {qi} head {h}: {got} vs {want}");
                }
            }
        }
    }
}
