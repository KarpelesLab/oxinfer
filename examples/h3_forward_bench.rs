//! Times one MiniMax-H3 transformer forward on a request kept by `OXINFER_H3_DUMP=DIR` and
//! compares its velocities with the kept ones:
//!
//! ```text
//! h3_forward_bench TRANSFORMER.safetensors DUMP_DIR [--lora FILE]... [--step I[,J...]] [--blocks N]
//!     [--device cpu|cuda] [--write DIR] [--ref DIR]
//! ```
//!
//! `--step` names the forwards to run, one after the other (default 0).
//! `--blocks N` runs the first N blocks only (then `--ref DIR` names the velocities of an earlier
//! `--write DIR` of the same N, the kept ones being the whole stack's).
//!
//! The video VAE's decoder on the rows kept by `OXINFER_H3_LATENTS=FILE` (`--write DIR` keeps
//! the decoded frames as float32 in `DIR/vae_px.bin`, `--ref DIR` compares with an earlier one):
//!
//! ```text
//! h3_forward_bench vae VAE.safetensors FILE.latents [--device cpu|cuda] [--write DIR] [--ref DIR]
//! ```

use anyhow::{Context, Result};
use oxinfer::minimax_h3::dit::{Dit, DitConfig, PackedAv, stage};
use oxinfer::minimax_h3::vae::{VaeConfig, VaeDecoder};
use oxinfer::safetensors::ModelFiles;
use std::path::{Path, PathBuf};

fn read(p: &Path) -> Result<Vec<f32>> {
    let b = std::fs::read(p).with_context(|| p.display().to_string())?;
    Ok(b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
}

fn write(p: &Path, v: &[f32]) -> Result<()> {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(p, b).with_context(|| p.display().to_string())
}

fn cmp(name: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "{name}: sizes differ");
    let max = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let mean = a.iter().zip(b).map(|(x, y)| (x - y).abs() as f64).sum::<f64>() / a.len() as f64;
    let scale = b.iter().map(|v| v.abs() as f64).sum::<f64>() / b.len() as f64;
    println!(
        "{name}: {} values, max abs diff {max:.3e}, mean abs diff {mean:.3e}, ref mean |v| {scale:.3e}",
        a.len()
    );
}

/// The `vae` form: decodes the video rows of `latents` with the VAE in `vae`.
fn vae(vae: &Path, latents: &Path, out: Option<&Path>, reference: Option<&Path>) -> Result<()> {
    let files = ModelFiles::open_file(vae, false)?;
    let json = VaeConfig::comfy_json(&files).context("not ComfyUI's single-file video VAE")??;
    let dec = VaeDecoder::load(VaeConfig::from_json(&json)?, &files)?;
    let b = std::fs::read(latents).with_context(|| latents.display().to_string())?;
    anyhow::ensure!(b.len() >= 28 && &b[..4] == b"H3LT", "{}: not a latent file", latents.display());
    let hdr: Vec<usize> = b[4..28]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c) as usize)
        .collect();
    let (lt, lh, lw) = (hdr[0], hdr[1], hdr[2]);
    let rows: Vec<f32> = b[28..28 + 4 * hdr[4]]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    let c = dec.cfg.latent_channels;
    anyhow::ensure!(rows.len() == c * lt * lh * lw, "{}: not {c} channels", latents.display());
    // The transformer's rows are 2x2 patches.
    let lat = oxinfer::minimax_h3::video::unpatchify_video(&rows, c, lt, lh, lw, 2);
    let t = std::time::Instant::now();
    let (px, frames) = dec.decode_video(&lat, lt, lh, lw)?;
    println!(
        "[{lt} latent frames of {lh}x{lw}: {frames} frames decoded in {:.2}s]",
        t.elapsed().as_secs_f64()
    );
    if let Some(d) = out {
        std::fs::create_dir_all(d)?;
        write(&d.join("vae_px.bin"), &px)?;
    }
    if let Some(r) = reference {
        cmp("decoded frames", &px, &read(&r.join("vae_px.bin"))?);
    }
    Ok(())
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut model = PathBuf::from(args.next().context("usage: h3_forward_bench TRANSFORMER DUMP_DIR [options]")?);
    let decode = model.as_os_str() == "vae";
    if decode {
        model = PathBuf::from(args.next().context("VAE.safetensors")?);
    }
    let dump = PathBuf::from(args.next().context("DUMP_DIR (or FILE.latents)")?);
    let (mut loras, mut steps, mut blocks, mut device) = (Vec::new(), vec![0usize], None, "cpu".to_string());
    let (mut out, mut reference) = (None, None);
    while let Some(a) = args.next() {
        let mut v = || args.next().with_context(|| format!("{a} needs a value"));
        match a.as_str() {
            "--lora" => loras.push(oxinfer::minimax_h3::lora::LoraSpec::parse(&v()?)?),
            "--step" => steps = v()?.split(',').map(|s| s.parse()).collect::<Result<_, _>>()?,
            "--blocks" => blocks = Some(v()?.parse::<usize>()?),
            "--device" => device = v()?,
            "--write" => out = Some(PathBuf::from(v()?)),
            "--ref" => reference = Some(PathBuf::from(v()?)),
            _ => anyhow::bail!("unknown option {a}"),
        }
    }
    let plat = oxinfer::platform::detect();
    let pref = oxinfer::platform::DevicePref::parse(&device).context("--device: cpu or cuda")?;
    let gpu = oxinfer::gpu::init(&plat, pref)?;
    println!("[device: {}]", gpu.map_or("cpu", |g| g.name()));
    if decode {
        return vae(&model, &dump, out.as_deref(), reference.as_deref());
    }
    let t0 = std::time::Instant::now();
    let files = ModelFiles::open_file(&model, false)?;
    let mut dit = Dit::load(DitConfig::from_files(&files)?, &files, None)?;
    for line in oxinfer::minimax_h3::lora::apply(&mut dit, &loras)? {
        println!("{line}");
    }
    if let Some(n) = blocks {
        dit.truncate_blocks(n);
    }
    println!("[loaded in {:.1}s]", t0.elapsed().as_secs_f64());
    let h = dit.cfg.hidden;
    let text = read(&dump.join("text_emb.bin"))?;
    let tags: Vec<u8> = read(&dump.join("tags.bin"))?.iter().map(|&t| t as u8).collect();
    let positions: Vec<[f32; 3]> = read(&dump.join("positions.bin"))?.as_chunks::<3>().0.to_vec();
    // `[text | audio | video]` (no keyframes).
    let nt = text.len() / h;
    let na = tags.iter().filter(|&&t| t == oxinfer::minimax_h3::dit::TAG_AUDIO).count();
    let n = tags.len();
    let (ti, ai, vi): (Vec<usize>, Vec<usize>, Vec<usize>) = ((0..nt).collect(), (nt..nt + na).collect(), (nt + na..n).collect());
    println!(
        "[{n} rows: {nt} text, {na} audio, {} video; {} blocks]",
        vi.len(),
        blocks.unwrap_or(dit.cfg.layers)
    );
    for step in steps {
        let video = read(&dump.join(format!("s{step}_video_in.bin")))?;
        let audio = read(&dump.join(format!("s{step}_audio_in.bin")))?;
        let temb = read(&dump.join(format!("s{step}_temb.bin")))?;
        let tidx: Vec<usize> = read(&dump.join(format!("s{step}_tidx.bin")))?.iter().map(|&t| t as usize).collect();
        let t1 = std::time::Instant::now();
        let (v, a) = dit.forward_av(&PackedAv {
            text_embedded: &text,
            video: &video,
            audio: &audio,
            text_index: &ti,
            video_index: &vi,
            audio_index: &ai,
            positions: &positions,
            temb: &temb,
            timestep_index: &tidx,
            tags: &tags,
            skip_blocks: &[],
        })?;
        println!("[forward {step}: {:.2}s]", t1.elapsed().as_secs_f64());
        if let Some(d) = &out {
            std::fs::create_dir_all(d)?;
            write(&d.join(format!("s{step}_v.bin")), &v)?;
            write(&d.join(format!("s{step}_a.bin")), &a)?;
        }
        let r = reference.as_deref().or(blocks.is_none().then_some(&dump));
        if let Some(r) = r {
            cmp("video velocity", &v, &read(&r.join(format!("s{step}_v.bin")))?);
            cmp("audio velocity", &a, &read(&r.join(format!("s{step}_a.bin")))?);
        }
    }
    print!("{}", stage::report());
    if let Some(s) = oxinfer::gpu::active().and_then(|b| b.stats()) {
        println!("  gpu: {s}");
    }
    Ok(())
}
