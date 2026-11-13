//! The audio side of Haru's language model (ports of `haru/models/meanflow_head.py` and of
//! `AudioProjector` in `haru/models/haru.py`):
//!
//! - [`Projector`]: an audio latent into the language model's embedding space (a linear layer,
//!   an RMS norm, a linear layer), one for the model's own audio and one for the user's;
//! - [`Head`]: the MeanFlow head, a small MLP that turns the language model's hidden state and
//!   64 values of noise into the next audio latent in one step. It predicts the mean velocity of
//!   the flow from noise to data over an interval; the time, the interval and the guidance scale
//!   are inputs of the network (sinusoidal features through two linear layers), so guided
//!   sampling is a single pass.

use crate::kernels::matmul;
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};
use serde::Deserialize;

fn d_steps() -> usize {
    1
}
fn d_guidance() -> f32 {
    1.5
}
fn d_interval() -> f32 {
    7.0
}
fn d_eps() -> f32 {
    1e-5
}
fn d_range() -> [f32; 2] {
    [1.0, 2.0]
}

/// `head` of a Haru `config.json`.
#[derive(Clone, Debug, Deserialize)]
pub struct HeadConfig {
    pub hidden_size: usize,
    pub latent_dim: usize,
    pub num_layers: usize,
    pub ffn_ratio: f64,
    #[serde(default = "d_eps")]
    pub norm_eps: f32,
    #[serde(default = "d_interval")]
    pub interval_embed: f32,
    #[serde(default = "d_steps")]
    pub num_steps: usize,
    #[serde(default = "d_guidance")]
    pub guidance_scale: f32,
    #[serde(default = "d_range")]
    pub guidance_range: [f32; 2],
}

const FREQUENCY_DIM: usize = 256;

fn matrix(f: &'static ModelFiles, name: &str, n_out: usize, n_in: usize) -> Result<&'static [u16]> {
    let s = &f.info(name).with_context(|| format!("missing tensor {name}"))?.shape;
    if s[..] != [n_out, n_in] {
        bail!("{name}: shape {s:?}, expected [{n_out}, {n_in}]");
    }
    f.bf16(name)
}

fn vector(f: &ModelFiles, name: &str, n: usize) -> Result<Vec<f32>> {
    let v = f.f32(name)?;
    if v.len() != n {
        bail!("{name}: {} values, expected {n}", v.len());
    }
    Ok(v)
}

/// `w x` for one row `x`.
fn apply(w: &[u16], n_out: usize, x: &[f32]) -> Vec<f32> {
    let mut y = vec![0f32; n_out];
    matmul(w, n_out, x.len(), x, 1, &mut y);
    y
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

/// `x * rsqrt(mean(x^2) + eps)`.
fn rms(x: &[f32], eps: f32) -> Vec<f32> {
    let s = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt();
    x.iter().map(|v| v * s).collect()
}

/// An audio latent to the embedding space of the language model.
pub struct Projector {
    fc1: &'static [u16],
    fc1_b: Vec<f32>,
    norm: Vec<f32>,
    fc2: &'static [u16],
    fc2_b: Vec<f32>,
    hidden: usize,
}

impl Projector {
    pub fn load(f: &'static ModelFiles, p: &str, latent: usize, hidden: usize) -> Result<Self> {
        Ok(Self {
            fc1: matrix(f, &format!("{p}.fc1.weight"), hidden, latent)?,
            fc1_b: vector(f, &format!("{p}.fc1.bias"), hidden)?,
            norm: vector(f, &format!("{p}.norm.weight"), hidden)?,
            fc2: matrix(f, &format!("{p}.fc2.weight"), hidden, hidden)?,
            fc2_b: vector(f, &format!("{p}.fc2.bias"), hidden)?,
            hidden,
        })
    }

    /// A normalized latent (`latent_dim` values) to `hidden` values.
    pub fn forward(&self, latent: &[f32]) -> Vec<f32> {
        let mut h = apply(self.fc1, self.hidden, latent);
        h.iter_mut().zip(&self.fc1_b).for_each(|(v, b)| *v += b);
        let mut n = rms(&h, 1e-6);
        n.iter_mut().zip(&self.norm).for_each(|(v, w)| *v *= w);
        let mut y = apply(self.fc2, self.hidden, &n);
        y.iter_mut().zip(&self.fc2_b).for_each(|(v, b)| *v += b);
        y
    }
}

/// Sinusoidal features of a scalar, then two linear layers.
struct ScalarEmbed {
    fc1: &'static [u16],
    fc2: &'static [u16],
}

impl ScalarEmbed {
    fn load(f: &'static ModelFiles, p: &str, hidden: usize) -> Result<Self> {
        Ok(Self {
            fc1: matrix(f, &format!("{p}.fc1.weight"), hidden, FREQUENCY_DIM)?,
            fc2: matrix(f, &format!("{p}.fc2.weight"), hidden, hidden)?,
        })
    }

    fn forward(&self, value: f32, hidden: usize) -> Vec<f32> {
        let half = FREQUENCY_DIM / 2;
        let scale = -(10000f64.ln()) as f32;
        let args: Vec<f32> = (0..half).map(|k| value * (scale * k as f32 / half as f32).exp()).collect();
        let features: Vec<f32> = args.iter().map(|a| a.cos()).chain(args.iter().map(|a| a.sin())).collect();
        let mut h = apply(self.fc1, hidden, &features);
        h.iter_mut().for_each(|v| *v = silu(*v));
        apply(self.fc2, hidden, &h)
    }
}

struct HeadLayer {
    norm: Vec<f32>,
    /// `[3 * hidden, hidden]`: shift, scale and gate from the condition.
    ada: &'static [u16],
    gate: &'static [u16],
    up: &'static [u16],
    down: &'static [u16],
}

pub struct Head {
    pub cfg: HeadConfig,
    pub weight_bytes: usize,
    ffn: usize,
    latent_proj: &'static [u16],
    cond_proj: &'static [u16],
    t_embed: ScalarEmbed,
    dt_embed: ScalarEmbed,
    h_embed: ScalarEmbed,
    omega_embed: ScalarEmbed,
    layers: Vec<HeadLayer>,
    /// `[2 * hidden, hidden]`: shift and scale.
    final_ada: &'static [u16],
    final_proj: &'static [u16],
}

impl Head {
    pub fn load(f: &'static ModelFiles, cfg: HeadConfig) -> Result<Self> {
        let (h, d) = (cfg.hidden_size, cfg.latent_dim);
        let ffn = (h as f64 * cfg.ffn_ratio) as usize;
        let layers = (0..cfg.num_layers)
            .map(|i| {
                let p = format!("head.layers.{i}");
                Ok(HeadLayer {
                    norm: vector(f, &format!("{p}.norm.weight"), h)?,
                    ada: matrix(f, &format!("{p}.ada.weight"), 3 * h, h)?,
                    gate: matrix(f, &format!("{p}.ffn.gate_proj.weight"), ffn, h)?,
                    up: matrix(f, &format!("{p}.ffn.up_proj.weight"), ffn, h)?,
                    down: matrix(f, &format!("{p}.ffn.down_proj.weight"), h, ffn)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if cfg.num_steps == 0 {
            bail!("haru head: num_steps is 0");
        }
        let weight_bytes = f
            .names()
            .filter(|n| n.starts_with("head."))
            .filter_map(|n| f.info(n))
            .map(|i| i.end - i.start)
            .sum();
        Ok(Self {
            ffn,
            weight_bytes,
            latent_proj: matrix(f, "head.latent_proj.weight", h, d)?,
            cond_proj: matrix(f, "head.cond_proj.weight", h, h)?,
            t_embed: ScalarEmbed::load(f, "head.t_embed", h)?,
            dt_embed: ScalarEmbed::load(f, "head.dt_embed", h)?,
            h_embed: ScalarEmbed::load(f, "head.h_embed", h)?,
            omega_embed: ScalarEmbed::load(f, "head.omega_embed", h)?,
            layers,
            final_ada: matrix(f, "head.final.ada.weight", 2 * h, h)?,
            final_proj: matrix(f, "head.final.proj.weight", d, h)?,
            cfg,
        })
    }

    /// The mean velocity at `z` for the projected condition `cond`, from time `t` over `h`.
    fn velocity(&self, z: &[f32], cond: &[f32], t: f32, h: f32, guidance: f32) -> Vec<f32> {
        let n = self.cfg.hidden_size;
        let mut c = cond.to_vec();
        for e in [
            self.t_embed.forward(t, n),
            self.dt_embed.forward(self.cfg.interval_embed, n),
            self.h_embed.forward(h, n),
            self.omega_embed.forward(1.0 - 1.0 / guidance, n),
        ] {
            c.iter_mut().zip(&e).for_each(|(v, e)| *v += e);
        }
        c.iter_mut().for_each(|v| *v = silu(*v));
        let mut x = apply(self.latent_proj, n, z);
        for l in &self.layers {
            let ada = apply(l.ada, 3 * n, &c);
            let (shift, rest) = ada.split_at(n);
            let (scale, gate) = rest.split_at(n);
            let hn = rms(&x, self.cfg.norm_eps);
            let inp: Vec<f32> = (0..n).map(|i| hn[i] * l.norm[i] * (1.0 + scale[i]) + shift[i]).collect();
            let mut a = apply(l.gate, self.ffn, &inp);
            let u = apply(l.up, self.ffn, &inp);
            a.iter_mut().zip(&u).for_each(|(a, u)| *a = silu(*a) * u);
            let o = apply(l.down, n, &a);
            x.iter_mut().zip(gate).zip(&o).for_each(|((x, g), o)| *x += g * o);
        }
        let ada = apply(self.final_ada, 2 * n, &c);
        let (shift, scale) = ada.split_at(n);
        let hn = rms(&x, self.cfg.norm_eps);
        let inp: Vec<f32> = (0..n).map(|i| hn[i] * (1.0 + scale[i]) + shift[i]).collect();
        apply(self.final_proj, self.cfg.latent_dim, &inp)
    }

    /// The next latent (normalized, `latent_dim` values) from the language model's hidden state
    /// and as many values of standard normal noise.
    pub fn sample(&self, hidden: &[f32], noise: &[f32], guidance: f32) -> Vec<f32> {
        let cond = apply(self.cond_proj, self.cfg.hidden_size, hidden);
        let steps = self.cfg.num_steps;
        let h = 1.0 / steps as f32;
        let mut z = noise.to_vec();
        for i in 0..steps {
            let v = self.velocity(&z, &cond, i as f32 * h, h, guidance);
            z.iter_mut().zip(&v).for_each(|(z, v)| *z += h * v);
        }
        z
    }
}
