//! Runs MiniMax-H3's video VAE (encoder and decoder) and its DiT with the joint video / audio
//! schedule (`oxinfer::minimax_h3`) on inputs written by `tools/minimax_h3_video_tiny.py` and
//! compares with diffusers' outputs:
//!
//! ```text
//! h3_video_check DIR
//! ```

use anyhow::{Context, Result};
use oxinfer::minimax_h3::dit::{Dit, DitConfig};
use oxinfer::minimax_h3::vae::{VaeConfig, VaeDecoder};
use oxinfer::minimax_h3::video::{Anchor, Geometry, denoise, layout, sigmas};
use oxinfer::minimax_h3::video_vae::VideoEncoder;
use oxinfer::safetensors::ModelFiles;
use std::path::Path;

fn read(p: &Path) -> Result<Vec<f32>> {
    let b = std::fs::read(p).with_context(|| p.display().to_string())?;
    Ok(b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
}

fn cmp(name: &str, a: &[f32], b: &[f32]) {
    let n = a.len().min(b.len());
    let max = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let scale = b.iter().map(|v| v.abs()).sum::<f32>() / n.max(1) as f32;
    let mse = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum::<f64>() / n.max(1) as f64;
    let peak = b.iter().fold(0f32, |m, v| m.max(v.abs())) as f64;
    println!(
        "{name}: {} values (ref {}), max abs diff {max:.3e}, ref mean |v| {scale:.3e}, PSNR {:.1} dB",
        a.len(),
        b.len(),
        10.0 * (peak * peak / mse.max(1e-30)).log10()
    );
}

fn main() -> Result<()> {
    let dir = std::env::args().nth(1).context("usage: h3_video_check DIR")?;
    let dir = Path::new(&dir);
    oxinfer::gpu::init(&oxinfer::platform::detect(), oxinfer::platform::DevicePref::Cpu)?;
    let case: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("case.json"))?)?;
    let u = |k: &str| case[k].as_u64().unwrap() as usize;

    // Video VAE: encoder (a clip, a single frame), decoder (one clip, two clips, padded).
    let vdir = dir.join("vae");
    let vfiles = ModelFiles::open(&vdir, false)?;
    let vae = VaeDecoder::load(VaeConfig::load(&vdir)?, &vfiles)?;
    let enc = VideoEncoder::load(&vdir, &vae.cfg, &vfiles)?;
    let (h, w, frames) = (u("h"), u("w"), u("frames"));
    let video = read(&dir.join("video.bin"))?;
    let (m, t, lh, lw) = enc.encode(&video, frames, h, w)?;
    println!("encoded {frames} frames -> [{}, {t}, {lh}, {lw}]", enc.channels);
    cmp("VAE encode (clip)", &m, &read(&dir.join("moments.bin"))?);
    let plane = h * w;
    let first: Vec<f32> = (0..3)
        .flat_map(|ch| video[ch * frames * plane..ch * frames * plane + plane].to_vec())
        .collect();
    let (m1, _, _, _) = enc.encode(&first, 1, h, w)?;
    cmp("VAE encode (one frame)", &m1, &read(&dir.join("moments1.bin"))?);
    for (t, nf) in case["decode"].as_object().context("decode cases")? {
        let t: usize = t.parse()?;
        let (px, got) = vae.decode_video(&read(&dir.join(format!("latents_{t}.bin")))?, t, lh, lw)?;
        assert_eq!(got as u64, nf.as_u64().unwrap(), "decoded frame count");
        cmp(
            &format!("VAE decode ({t} latent frames -> {got})"),
            &px,
            &read(&dir.join(format!("decoded_{t}.bin")))?,
        );
    }

    // DiT: layout, the first step's velocities, the rows after the whole schedule.
    let tdir = dir.join("transformer");
    if !tdir.exists() {
        // A VAE-only case (the real VAE).
        return Ok(());
    }
    let files = ModelFiles::open(&tdir, false)?;
    let dit = Dit::load(DitConfig::load(&tdir)?, &files, None)?;
    let g = Geometry {
        lt: u("lt"),
        lh: u("lh"),
        lw: u("lw"),
        n_audio: u("n_audio"),
    };
    let text = read(&dir.join("dit_text.bin"))?;
    let tags: Vec<u8> = read(&dir.join("dit_text_tags.bin"))?.iter().map(|&v| v as u8).collect();
    let anchors = [Anchor::First, Anchor::Last];
    let lay = layout(&tags, g.lt, g.lh, g.lw, g.n_audio, dit.cfg.patch[1], &anchors);
    let pos: Vec<f32> = lay.positions.iter().flatten().copied().collect();
    cmp("layout positions", &pos, &read(&dir.join("dit_positions.bin"))?);
    let (sv, sa) = (sigmas(u("steps"), 12.0), sigmas(u("steps"), 3.0));
    cmp("video sigmas", &sv, &read(&dir.join("sigmas_video.bin"))?);
    cmp("audio sigmas", &sa, &read(&dir.join("sigmas_audio.bin"))?);
    let video = read(&dir.join("dit_video.bin"))?;
    let audio = read(&dir.join("dit_audio.bin"))?;
    // One step: x1 = r x + (1 - r) (x + (1 - t) v), so v follows from the rows after one step.
    let one = |s: &[f32]| vec![s[0], s[1]];
    let (v1, a1) = denoise(
        &dit,
        &text,
        &tags,
        video.clone(),
        audio.clone(),
        g,
        &anchors,
        (&one(&sv), &one(&sa)),
        &mut |_, _| {},
    )?;
    let velocity = |x0: &[f32], x1: &[f32], s: &[f32]| -> Vec<f32> {
        let (st, r) = (1.0 - (1.0 - s[0]), s[1] / s[0]);
        x0.iter().zip(x1).map(|(a, b)| (b - a) / ((1.0 - r) * st)).collect()
    };
    let c0 = lay.n_cond * dit.cfg.patch_dim();
    cmp(
        "DiT video velocity (step 1)",
        &velocity(&video[c0..], &v1[c0..], &sv),
        &read(&dir.join("dit_v0.bin"))?[c0..],
    );
    cmp(
        "DiT audio velocity (step 1)",
        &velocity(&audio, &a1, &sa),
        &read(&dir.join("dit_a0.bin"))?,
    );
    let (vf, af) = denoise(&dit, &text, &tags, video, audio, g, &anchors, (&sv, &sa), &mut |_, _| {})?;
    cmp("video rows after the schedule", &vf, &read(&dir.join("dit_video_final.bin"))?);
    cmp("audio rows after the schedule", &af, &read(&dir.join("dit_audio_final.bin"))?);
    Ok(())
}
