//! The VAE's encoder and decoder on CUDA with the activations kept on the device
//! ([`crate::cuda_vae`]): the layers of [`super`] walked over device frames. The mid block's
//! attention runs on the host. The residual groups of Qwen-Image-2.1's VAE (`is_residual`) have
//! no kernel there: such a stack is declined.
//!
//! `OXINFER_QWEN_VAE_GPU=0` keeps the per-layer path; a pass that cannot run on the device
//! returns `None` and the host path runs.

use super::{Layer, Res, Stack};
use crate::cuda_vae::{DFrame, Out, Pass};
use anyhow::{Result, bail};

pub(super) use crate::cuda_vae::Engine;

/// The variable that turns the device passes off.
pub(super) const OFF: &str = "OXINFER_QWEN_VAE_GPU";

fn res<'p>(p: &mut Pass<'p, '_>, r: &Res, x: &DFrame<'p>, h: usize, w: usize) -> Result<DFrame<'p>> {
    let sc = match &r.shortcut {
        Some(c) => Some(p.conv(c, &[Some(x)], h, w, 1, false, None, Out::Frame)?.0),
        None => None,
    };
    let a = p.rms(&r.norm1.g, x, h * w, true)?;
    let a = p.conv(&r.conv1, &[Some(&a)], h, w, 1, false, None, Out::Frame)?.0;
    let a = p.rms(&r.norm2.g, &a, h * w, true)?;
    Ok(
        p.conv(&r.conv2, &[Some(&a)], h, w, 1, false, Some(sc.as_ref().unwrap_or(x)), Out::Frame)?
            .0,
    )
}

fn layer<'p>(p: &mut Pass<'p, '_>, l: &Layer, x: DFrame<'p>, h: usize, w: usize) -> Result<(DFrame<'p>, usize, usize)> {
    match l {
        Layer::Group(..) => bail!("a residual group"),
        Layer::Res(r) => Ok((res(p, r, &x, h, w)?, h, w)),
        Layer::Attn(a) => {
            let v = p.download(&x, h * w, a.c)?;
            Ok((p.upload(&a.forward(&v, h, w), a.c)?, h, w))
        }
        Layer::Down(c) => p.conv(c, &[Some(&x)], h, w, 2, false, None, Out::Frame),
        Layer::Up(c) => Ok((p.conv(c, &[Some(&x)], h, w, 1, true, None, Out::Frame)?.0, 2 * h, 2 * w)),
    }
}

/// [`Stack::forward`] on the device.
pub(super) fn forward(e: &Engine, s: &Stack, x: &[f32], h: usize, w: usize) -> Option<(Vec<f32>, usize, usize)> {
    if s.layers.iter().any(|l| matches!(l, Layer::Group(..))) {
        return None;
    }
    e.pass(|p| {
        let x = p.upload(x, s.conv_in.cin)?;
        let (mut x, mut h, mut w) = p.conv(&s.conv_in, &[Some(&x)], h, w, 1, false, None, Out::Frame)?;
        for l in &s.layers {
            (x, h, w) = layer(p, l, x, h, w)?;
        }
        let a = p.rms(&s.norm_out.g, &x, h * w, true)?;
        drop(x);
        let y = p.conv(&s.conv_out, &[Some(&a)], h, w, 1, false, None, Out::Dense)?.0;
        Ok((p.download(&y, h * w, s.conv_out.cout)?, h, w))
    })
}

#[cfg(test)]
mod tests {
    use super::super::Vae;
    use super::*;
    use crate::cuda::CudaBackend;

    fn backend() -> Option<&'static CudaBackend> {
        static B: std::sync::OnceLock<Option<CudaBackend>> = std::sync::OnceLock::new();
        B.get_or_init(|| CudaBackend::new(0).ok()).as_ref()
    }

    /// The device's encoder and decoder against the host's on a tiny VAE of the same layout (the
    /// LongCat-Video-Avatar tiny release's; skipped without an NVIDIA GPU).
    #[test]
    fn stacks_match_host() {
        let Some(be) = backend() else { return };
        let dir = std::env::temp_dir().join(format!("oxinfer-qwen-vae-cuda-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (_, base) = crate::longcat_avatar::tiny::write_tiny_release(&dir).unwrap();
        let vae = Vae::load(&base.join("vae"), true).unwrap();
        let e = Engine::build(be).unwrap();
        let mut seed = 9u64;
        let mut rnd = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        };
        let (h, w) = (40, 56);
        let enc = vae.enc.as_ref().unwrap();
        let px: Vec<f32> = (0..h * w * enc.conv_in.cin).map(|_| rnd()).collect();
        let lat: Vec<f32> = (0..(h / 8) * (w / 8) * vae.dec.conv_in.cin).map(|_| rnd()).collect();
        for (what, s, x, h, w) in [("encode", enc, &px, h, w), ("decode", &vae.dec, &lat, h / 8, w / 8)] {
            let (dev, dh, dw) = forward(&e, s, x, h, w).expect("the engine declined");
            let (host, hh, hw) = s.forward(x, h, w);
            assert_eq!((dh, dw, dev.len()), (hh, hw, host.len()), "{what}: size");
            let scale = host.iter().map(|v| v.abs()).sum::<f32>() / host.len() as f32;
            let max = dev.iter().zip(&host).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(max < 1e-4 * scale.max(1.0), "{what}: {max:.3e} (mean |v| {scale:.3e})");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
