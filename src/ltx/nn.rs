//! Layers shared by LTX-2.5's transformer and its text connectors (`ltx_core.model.transformer`):
//! linear layers with biases, the "split" rotary embedding over fractional positions, attention
//! with an RMS norm over all heads' queries / keys and an optional per-head sigmoid gate, the
//! GELU (tanh) feed-forward, and the PixArt-alpha timestep embedder.

use crate::kernels::{for_chunks, gelu_tanh, rmsnorm_f32w_rows_inplace};
use crate::minimax_h3::dit::DLin;
use crate::safetensors::ModelFiles;
use anyhow::{Result, bail};

/// `y = W x + b` for several layers over one input (one fused matmul call).
pub(crate) fn lins(ls: &[&DLin], x: &[f32], t: usize) -> Vec<Vec<f32>> {
    let n_in = ls[0].n_in;
    let mut ys: Vec<Vec<f32>> = ls.iter().map(|l| vec![0f32; t * l.n_out]).collect();
    {
        let mut jobs: Vec<(&crate::model::Lin, usize, &mut [f32])> =
            ls.iter().zip(ys.iter_mut()).map(|(l, y)| (&l.w, l.n_out, &mut y[..])).collect();
        crate::model::linear_multi(&mut jobs, n_in, x, t);
    }
    for (l, y) in ls.iter().zip(ys.iter_mut()) {
        if let Some(b) = &l.b {
            add_bias(y, b);
        }
    }
    ys
}

pub(crate) fn add_bias(y: &mut [f32], b: &[f32]) {
    let n = b.len();
    for_chunks(y, n, y.len() > 8 * n, |_, r| r.iter_mut().zip(b).for_each(|(v, &bv)| *v += bv));
}

/// Root-mean-square normalization of every `n`-wide row without a weight
/// (`torch.nn.functional.rms_norm`).
pub(crate) fn rms_rows(x: &[f32], n: usize, eps: f32) -> Vec<f32> {
    let mut y = x.to_vec();
    rms_rows_inplace(&mut y, n, eps);
    y
}

pub(crate) fn rms_rows_inplace(x: &mut [f32], n: usize, eps: f32) {
    for_chunks(x, n, x.len() > 8 * n, |_, r| {
        let ss = r.iter().map(|v| v * v).sum::<f32>() / n as f32;
        let s = 1.0 / (ss + eps).sqrt();
        r.iter_mut().for_each(|v| *v *= s);
    });
}

/// Rotary tables of a sequence: per token `inner / 2` cosines and sines (head `h` reads
/// `[h * hd / 2, (h + 1) * hd / 2)`).
pub struct Rope {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    pub half: usize,
}

/// `generate_freq_grid_np`: `theta ^ linspace(0, 1, inner / (2 dims)) * pi / 2`, in float64 then
/// float32.
pub(crate) fn freq_grid(theta: f64, dims: usize, inner: usize) -> Vec<f32> {
    let n = inner / (2 * dims);
    (0..n)
        .map(|i| {
            let e = if n == 1 { 0.0 } else { i as f64 / (n - 1) as f64 };
            (theta.powf(e) * std::f64::consts::FRAC_PI_2) as f32
        })
        .collect()
}

impl Rope {
    /// `precompute_freqs_cis` with the split layout: `frac` holds per token one fractional
    /// position per axis (`position / max_pos`), the angles of frequency `j` and axis `d` at
    /// `j * dims + d` after `inner / 2 - n_freq * dims` padding entries (cos 1, sin 0).
    pub fn new(frac: &[f32], dims: usize, inner: usize, theta: f64) -> Self {
        let grid = freq_grid(theta, dims, inner);
        let half = inner / 2;
        let n_tok = frac.len() / dims;
        let pad = half - grid.len() * dims;
        let mut cos = vec![1f32; n_tok * half];
        let mut sin = vec![0f32; n_tok * half];
        for t in 0..n_tok {
            let f = &frac[t * dims..(t + 1) * dims];
            let (c, s) = (&mut cos[t * half..(t + 1) * half], &mut sin[t * half..(t + 1) * half]);
            for (j, &g) in grid.iter().enumerate() {
                for d in 0..dims {
                    let a = g * (f[d] * 2.0 - 1.0);
                    c[pad + j * dims + d] = a.cos();
                    s[pad + j * dims + d] = a.sin();
                }
            }
        }
        Self { cos, sin, half }
    }

    /// Rotates `x` (`[n, inner]`, heads of `hd`) in place: per head the first half against the
    /// second (`apply_split_rotary_emb`).
    pub fn apply(&self, x: &mut [f32], heads: usize, hd: usize) {
        let inner = heads * hd;
        let h2 = hd / 2;
        debug_assert_eq!(self.half, inner / 2);
        for_chunks(x, inner, x.len() > 8 * inner, |r, row| {
            let (c, s) = (
                &self.cos[r * self.half..(r + 1) * self.half],
                &self.sin[r * self.half..(r + 1) * self.half],
            );
            for h in 0..heads {
                let xh = &mut row[h * hd..(h + 1) * hd];
                let (c, s) = (&c[h * h2..(h + 1) * h2], &s[h * h2..(h + 1) * h2]);
                for i in 0..h2 {
                    let (a, b) = (xh[i], xh[i + h2]);
                    xh[i] = a * c[i] - s[i] * b;
                    xh[i + h2] = b * c[i] + s[i] * a;
                }
            }
        });
    }
}

/// Multi-head attention without a mask, queries in chunks that keep a score matrix under
/// ~512 MB (on the accelerator when one is active, see [`crate::sd::ops::attention`]).
pub(crate) fn attention(q: &[f32], k: &[f32], v: &[f32], tq: usize, tk: usize, heads: usize, hd: usize) -> Vec<f32> {
    let d = heads * hd;
    let chunk = ((1usize << 27) / (tk * heads).max(1)).clamp(64, tq.max(64));
    if chunk >= tq {
        return crate::sd::ops::attention(q, k, v, tq, tk, heads, hd, false);
    }
    let mut out = vec![0f32; tq * d];
    let mut s = 0;
    while s < tq {
        let e = (s + chunk).min(tq);
        let o = crate::sd::ops::attention(&q[s * d..e * d], k, v, e - s, tk, heads, hd, false);
        out[s * d..e * d].copy_from_slice(&o);
        s = e;
    }
    out
}

fn vec_f32(f: &ModelFiles, name: &str, n: usize) -> Result<Vec<f32>> {
    let v = f.f32(name)?;
    if v.len() != n {
        bail!("{name}: {} values, expected {n}", v.len());
    }
    Ok(v)
}

pub(crate) fn table(f: &ModelFiles, name: &str, rows: usize, n: usize) -> Result<Vec<f32>> {
    vec_f32(f, name, rows * n)
}

/// `ltx_core.model.transformer.attention.Attention`.
pub(crate) struct Attn<'a> {
    pub q: DLin<'a>,
    pub k: DLin<'a>,
    pub v: DLin<'a>,
    pub o: DLin<'a>,
    /// `to_gate_logits`: one logit per head from the query input, `2 sigmoid` scales the head.
    pub gate: Option<DLin<'a>>,
    pub qn: Vec<f32>,
    pub kn: Vec<f32>,
    pub heads: usize,
    pub hd: usize,
    pub eps: f32,
}

impl<'a> Attn<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn load(f: &'a ModelFiles, p: &str, q_dim: usize, ctx_dim: usize, heads: usize, hd: usize, eps: f32) -> Result<Self> {
        let inner = heads * hd;
        let gate_name = format!("{p}.to_gate_logits");
        let gate = if f.has(&format!("{gate_name}.weight")) {
            Some(DLin::load(f, &gate_name, heads, q_dim, true)?)
        } else {
            None
        };
        Ok(Self {
            q: DLin::load(f, &format!("{p}.to_q"), inner, q_dim, true)?,
            k: DLin::load(f, &format!("{p}.to_k"), inner, ctx_dim, true)?,
            v: DLin::load(f, &format!("{p}.to_v"), inner, ctx_dim, true)?,
            o: DLin::load(f, &format!("{p}.to_out.0"), q_dim, inner, true)?,
            gate,
            qn: vec_f32(f, &format!("{p}.q_norm.weight"), inner)?,
            kn: vec_f32(f, &format!("{p}.k_norm.weight"), inner)?,
            heads,
            hd,
            eps,
        })
    }

    /// Attention of `x` (`[n, q_dim]`) over `ctx` (`[nk, ctx_dim]`; none: self-attention), the
    /// queries rotated by `pe` and the keys by `k_pe` (else `pe`).
    pub fn forward(&self, x: &[f32], n: usize, ctx: Option<(&[f32], usize)>, pe: Option<&Rope>, k_pe: Option<&Rope>) -> Vec<f32> {
        let inner = self.heads * self.hd;
        let (mut q, mut k, v, nk) = match ctx {
            None => {
                let mut ys = lins(&[&self.q, &self.k, &self.v], x, n);
                let v = ys.pop().unwrap();
                let k = ys.pop().unwrap();
                (ys.pop().unwrap(), k, v, n)
            }
            Some((c, nk)) => {
                let q = self.q.forward(x, n);
                let mut ys = lins(&[&self.k, &self.v], c, nk);
                let v = ys.pop().unwrap();
                (q, ys.pop().unwrap(), v, nk)
            }
        };
        rmsnorm_f32w_rows_inplace(&mut q, inner, &self.qn, self.eps);
        rmsnorm_f32w_rows_inplace(&mut k, inner, &self.kn, self.eps);
        if let Some(pe) = pe {
            pe.apply(&mut q, self.heads, self.hd);
            k_pe.unwrap_or(pe).apply(&mut k, self.heads, self.hd);
        }
        let mut out = attention(&q, &k, &v, n, nk, self.heads, self.hd);
        drop((q, k, v));
        if let Some(g) = &self.gate {
            let logits = g.forward(x, n);
            let (heads, hd) = (self.heads, self.hd);
            for_chunks(&mut out, inner, n > 8, |r, row| {
                for h in 0..heads {
                    let s = 2.0 / (1.0 + (-logits[r * heads + h]).exp());
                    row[h * hd..(h + 1) * hd].iter_mut().for_each(|v| *v *= s);
                }
            });
        }
        self.o.forward(&out, n)
    }

    /// The attention skipped (STG's perturbation, `all_perturbed`): the value projection of `x`
    /// straight through the output projection, no per-head gate.
    pub fn passthrough(&self, x: &[f32], n: usize) -> Vec<f32> {
        let v = self.v.forward(x, n);
        self.o.forward(&v, n)
    }
}

/// `FeedForward`: GELU (tanh) between two linear layers (`ff.net.0.proj`, `ff.net.2`).
pub(crate) struct Ff<'a> {
    pub up: DLin<'a>,
    pub down: DLin<'a>,
}

impl<'a> Ff<'a> {
    pub fn load(f: &'a ModelFiles, p: &str, dim: usize, inner: usize) -> Result<Self> {
        let bias = f.has(&format!("{p}.net.0.proj.bias"));
        Ok(Self {
            up: DLin::load(f, &format!("{p}.net.0.proj"), inner, dim, bias)?,
            down: DLin::load(f, &format!("{p}.net.2"), dim, inner, bias)?,
        })
    }

    pub fn forward(&self, x: &[f32], n: usize) -> Vec<f32> {
        let mut h = self.up.forward(x, n);
        let w = self.up.n_out;
        for_chunks(&mut h, w, n > 1, |_, r| r.iter_mut().for_each(|v| *v = gelu_tanh(*v)));
        self.down.forward(&h, n)
    }
}

/// `get_timestep_embedding(t, 256, flip_sin_to_cos=True, downscale_freq_shift=0)`: `[cos | sin]`.
pub(crate) fn timestep_sinusoid(t: f32) -> Vec<f32> {
    let half = 128usize;
    let l = -(10000f64.ln() as f32);
    let mut out = vec![0f32; 2 * half];
    for i in 0..half {
        let e = (l * i as f32 / half as f32).exp();
        let a = t * e;
        out[i] = a.cos();
        out[half + i] = a.sin();
    }
    out
}

/// `AdaLayerNormSingle`: the timestep embedder (`emb.timestep_embedder.linear_1` / `_2`) and the
/// modulation projection (`linear`) of its SiLU.
pub(crate) struct AdaSingle<'a> {
    pub l1: DLin<'a>,
    pub l2: DLin<'a>,
    pub lin: DLin<'a>,
}

impl<'a> AdaSingle<'a> {
    pub fn load(f: &'a ModelFiles, p: &str, dim: usize, coeff: usize) -> Result<Self> {
        Ok(Self {
            l1: DLin::load(f, &format!("{p}.emb.timestep_embedder.linear_1"), dim, 256, true)?,
            l2: DLin::load(f, &format!("{p}.emb.timestep_embedder.linear_2"), dim, dim, true)?,
            lin: DLin::load(f, &format!("{p}.linear"), coeff * dim, dim, true)?,
        })
    }

    /// `(modulation, embedded timestep)` of timestep `t` (already times the multiplier).
    pub fn forward(&self, t: f32) -> (Vec<f32>, Vec<f32>) {
        let s = timestep_sinusoid(t);
        let mut h = self.l1.forward(&s, 1);
        h.iter_mut().for_each(|v| *v = crate::sd::ops::silu(*v));
        let emb = self.l2.forward(&h, 1);
        let act: Vec<f32> = emb.iter().map(|&v| crate::sd::ops::silu(v)).collect();
        (self.lin.forward(&act, 1), emb)
    }
}
