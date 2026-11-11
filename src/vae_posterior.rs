//! Sampling the VAE posterior of an input image, as the reference pipelines of OmniGen2
//! (`latent_dist.sample()`), BAGEL (`DiagonalGaussian`) and Step1X-Edit
//! (`retrieve_latents(..., sample_mode="sample")`) do: `z = mean + exp(0.5 * logvar) * noise`,
//! then the latent normalization. oxinfer's default is the posterior mean (`noise = 0`).
//!
//! The noise is drawn from oxinfer's own generator ([`crate::sd::ops::Rng`]: xoshiro256**
//! seeded through splitmix64, Box-Muller normals), in channel-first order (`[c, h, w]`, the
//! order `torch.randn` fills that shape), from a stream of its own per input image
//! ([`noise_seed`]). It is not torch's stream (Mersenne Twister / Philox on a GPU), which the
//! crate does not reproduce: as with the initial latents, a sampled encoding is the same
//! distribution as the reference's, not the same draw. (OmniGen2 and BAGEL do not even draw it
//! from the run's seeded generator, but from torch's global one.) Checks against the reference
//! pass the reference's noise in.

use crate::flux::vae::Vae;
use anyhow::{Context, Result, bail};

/// The seed of the posterior noise of input image `index` in a run seeded with `seed`: a stream
/// apart from the initial latents' (`Rng::new(seed)`).
pub fn noise_seed(seed: u64, index: usize) -> u64 {
    (seed ^ 0x5641_4550_6f73_7400).wrapping_add((index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

/// The posterior noise of input image `index`: `c * lh * lw` standard normals, channel first.
pub fn noise(seed: u64, index: usize, n: usize) -> Vec<f32> {
    crate::sd::ops::Rng::new(noise_seed(seed, index)).normals(n)
}

/// A sampled encoding.
pub struct Sampled {
    /// Normalized latents `[lh * lw, c]`, as [`Vae::encode`].
    pub latents: Vec<f32>,
    pub lh: usize,
    pub lw: usize,
    /// The posterior's standard deviations relative to its (shifted) means, as an L2 ratio over
    /// the latent: how far a sample sits from the mean encoding.
    pub relative_std: f64,
}

/// Normalized latents of `x` (`[h*w, 3]` RGB in [-1, 1]) sampled from the posterior with `noise`
/// (`[c, lh, lw]` standard normals). `clamp_logvar`: diffusers'
/// `DiagonalGaussianDistribution` clamps the log-variance to [-30, 20]; the BFL autoencoder
/// (BAGEL) does not.
pub fn encode_sampled(vae: &Vae, x: &[f32], h: usize, w: usize, noise: &[f32], clamp_logvar: bool) -> Result<Sampled> {
    let enc = vae.encoder.as_ref().context("the VAE was loaded without its encoder")?;
    let (m, lh, lw) = enc.moments(x, h, w);
    let c = enc.latent_channels;
    let n = lh * lw;
    if noise.len() != c * n {
        bail!("posterior noise: {} values for [{c}, {lh}, {lw}]", noise.len());
    }
    let mut z = vec![0f32; n * c];
    let (s, sh) = (vae.scaling_factor, vae.shift_factor);
    let (mut var, mut base) = (0f64, 0f64);
    for (p, r) in z.chunks_mut(c).enumerate() {
        for (i, v) in r.iter_mut().enumerate() {
            let mean = m[p * 2 * c + i];
            let mut logvar = m[p * 2 * c + c + i];
            if clamp_logvar {
                logvar = logvar.clamp(-30.0, 20.0);
            }
            let std = (0.5 * logvar).exp();
            *v = (mean + std * noise[i * n + p] - sh) * s;
            var += (std as f64).powi(2);
            base += ((mean - sh) as f64).powi(2);
        }
    }
    Ok(Sampled {
        latents: z,
        lh,
        lw,
        relative_std: (var / base).sqrt(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streams_are_apart() {
        let a = noise(42, 0, 8);
        assert_eq!(a, noise(42, 0, 8));
        assert_ne!(a, noise(42, 1, 8));
        assert_ne!(a, noise(43, 0, 8));
        assert_ne!(a, crate::sd::ops::Rng::new(42).normals(8));
    }
}
