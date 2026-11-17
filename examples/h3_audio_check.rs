//! Runs MiniMax-H3's audio VAE decoder (`oxinfer::minimax_h3::audio_vae`) on the audio rows
//! written by `tools/minimax_h3_audio_tiny.py` and compares with diffusers' waveform:
//!
//! ```text
//! h3_audio_check AUDIO_VAE DIR [OUT.wav | --write]
//! ```
//!
//! `AUDIO_VAE` is the release's `audio_vae/` folder or ComfyUI's single file. `DIR` holds
//! `audio_rows.bin` and `audio_decoded.bin`; `OUT.wav` receives the decoded stereo soundtrack.
//! `--write` writes `audio_decoded.bin` from this decoder instead of comparing, so that another
//! form of the same VAE can be compared with it.

use anyhow::{Context, Result};
use oxinfer::minimax_h3::audio_vae::AudioDecoder;
use oxinfer::minimax_h3::video::Soundtrack;
use std::path::Path;
use std::time::Instant;

fn read(p: &Path) -> Result<Vec<f32>> {
    let b = std::fs::read(p).with_context(|| p.display().to_string())?;
    Ok(b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: h3_audio_check AUDIO_VAE DIR [OUT.wav | --write]";
    let (vdir, dir) = (args.next().context(usage)?, args.next().context(usage)?);
    let dir = Path::new(&dir);
    oxinfer::gpu::init(&oxinfer::platform::detect(), oxinfer::platform::DevicePref::Cpu)?;
    let t0 = Instant::now();
    let vae = AudioDecoder::load(Path::new(&vdir))?;
    let t_load = t0.elapsed().as_secs_f64();
    let rows = read(&dir.join("audio_rows.bin"))?;
    let out = args.next();
    let t1 = Instant::now();
    let channels = vae.decode_rows(&rows, 2)?;
    let secs = channels[0].len() as f64 / vae.cfg.sampling_rate as f64;
    println!(
        "{} latents per channel -> 2 x {} samples ({secs:.2} s at {} Hz); loaded in {t_load:.1}s, decoded in {:.1}s",
        rows.len() / 2 / vae.cfg.latent_channels,
        channels[0].len(),
        vae.cfg.sampling_rate,
        t1.elapsed().as_secs_f64()
    );
    let got: Vec<f32> = channels.concat();
    if out.as_deref() == Some("--write") {
        let b: Vec<u8> = got.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(dir.join("audio_decoded.bin"), b)?;
        println!("wrote {}", dir.join("audio_decoded.bin").display());
        return Ok(());
    }
    let want = read(&dir.join("audio_decoded.bin"))?;
    assert_eq!(got.len(), want.len(), "sample count");
    let n = got.len() as f64;
    let max = got.iter().zip(&want).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let mse = got.iter().zip(&want).map(|(x, y)| ((x - y) as f64).powi(2)).sum::<f64>() / n;
    let power = want.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / n;
    let clamped = want.iter().filter(|v| v.abs() >= 1.0).count() as f64 / n;
    println!(
        "audio VAE decode: max abs diff {max:.3e}, ref RMS {:.3e} ({:.2}% of the samples at the clamp), SNR {:.1} dB",
        power.sqrt(),
        100.0 * clamped,
        10.0 * (power / mse.max(1e-30)).log10()
    );
    if let Some(out) = out {
        let s = Soundtrack {
            sample_rate: vae.cfg.sampling_rate,
            channels,
        };
        std::fs::write(&out, s.to_wav()).with_context(|| format!("write {out}"))?;
        println!("wrote {out}");
    }
    Ok(())
}
