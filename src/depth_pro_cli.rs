//! `oxinfer depth IMAGE...`: metric depth with Apple's Depth Pro 2 ([`oxinfer::depth_pro`]), as
//! the reference's `depth-pro-2` command writes it: `NAME-depth.png` (inverse depth in color),
//! with `--png16` `NAME-depth16.png` (16-bit inverse depth), with `--raw` `NAME-depth.f32` (depth
//! in meters, little-endian float32, row-major), and a summary (focal length, depth range). A
//! video (or a folder of frames) gets `NAME-depth.mp4` and per-frame folders, its frames
//! attending to each other (`--mode buffered` clips or `unbuffered` live frames).

use anyhow::{Context, Result, bail};
use oxinfer::depth_pro::{DepthOptions, DepthPro, VideoOptions};
use std::path::PathBuf;
use std::time::Instant;

pub const USAGE: &str = "usage: oxinfer depth IMAGE|VIDEO|FRAMES_DIR... [options]\n\
\n\
Predicts metric depth with Depth Pro 2 and writes NAME-depth.png (inverse depth through the turbo\n\
color map) beside each image or into --out; for a video (.mp4, .mov, .mkv, .webm, .avi, or a folder\n\
of frames) NAME-depth.mp4, the frames attending to each other.\n\
\n\
  --model SPEC      apple/DepthPro2-S (default), -B, -L, -H, or a folder\n\
  --focal SPEC      the focal length estimator (default apple/DepthPro2-focal; none: no estimate)\n\
  --focal-px F      the focal length in pixels (skips EXIF and the estimator)\n\
  --no-exif         ignore the EXIF 35 mm focal length\n\
  --out DIR         where the files go (default: beside each image)\n\
  --png16           also NAME-depth16.png: 16-bit inverse depth (the summary gives meters = k / value)\n\
  --raw             also NAME-depth.f32: depth in meters, float32, row-major (infinite: far)\n\
  --mode M          video: buffered (clips of --window frames overlapping by --overlap, the\n\
                    default) or unbuffered (each frame from itself and the window - 1 before it)\n\
  --window N        video: frames attended together (default 16)\n\
  --overlap K       video, buffered: frames shared by consecutive clips (default 4)\n\
  --frames-fps F    frame rate of a folder of frames (default 24)\n\
  --profile         time per phase
  --device D        cpu, gpu, metal, ... (default auto)\n\
  --threads N       CPU threads\n\
  --cache-dir DIR   download cache";

/// Runs the arguments after `depth`.
pub fn run(args: &[String]) -> Result<()> {
    let mut o = DepthOptions::default();
    let mut model = "apple/DepthPro2-S".to_string();
    let (mut focal_px, mut exif, mut png16, mut raw, mut profile) = (None::<f32>, true, false, false, false);
    let mut out_dir: Option<PathBuf> = None;
    let mut images = Vec::new();
    let mut vo = VideoOptions::default();
    let mut frames_fps: Option<f64> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let (flag, inline) = match a.split_once('=') {
            Some((f, v)) if f.starts_with("--") => (f, Some(v.to_string())),
            _ => (a.as_str(), None),
        };
        let mut val = || -> Result<String> {
            match &inline {
                Some(v) => Ok(v.clone()),
                None => it.next().cloned().with_context(|| format!("{flag} needs a value")),
            }
        };
        match flag {
            "-h" | "--help" => {
                eprintln!("{USAGE}");
                return Ok(());
            }
            "--model" | "-m" => model = val()?,
            "--focal" => o.set("focal", &val()?)?,
            "--focal-px" => focal_px = Some(val()?.parse().context("--focal-px")?),
            "--no-exif" => exif = false,
            "--out" | "-o" => out_dir = Some(val()?.into()),
            "--png16" => png16 = true,
            "--raw" => raw = true,
            "--profile" => profile = true,
            "--mode" => vo.set("mode", &val()?)?,
            "--window" => vo.set("window", &val()?)?,
            "--overlap" => vo.set("overlap", &val()?)?,
            "--frames-fps" => frames_fps = Some(val()?.parse().context("--frames-fps")?),
            "--device" => o.set("device", &val()?)?,
            "--threads" => o.set("threads", &val()?)?,
            "--cache-dir" => o.set("cache_dir", &val()?)?,
            s if s.starts_with('-') => bail!("unknown option {s}\n{USAGE}"),
            s => images.push(PathBuf::from(s)),
        }
    }
    if images.is_empty() {
        bail!("{USAGE}");
    }
    let t0 = Instant::now();
    let net = DepthPro::load(&model, &o)?;
    let info = net.info();
    eprintln!(
        "[loaded Depth Pro 2 ({}, {:.0}M parameters{}) on {} in {:.2}s]",
        info.backbone,
        info.params as f64 / 1e6,
        if net.has_focal_estimator() {
            ", with the focal length estimator"
        } else {
            ""
        },
        info.device,
        t0.elapsed().as_secs_f64()
    );
    vo.focal_px = focal_px;
    for path in &images {
        let stem = path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "image".into());
        let dir = out_dir
            .clone()
            .unwrap_or_else(|| path.parent().map(|p| p.to_path_buf()).unwrap_or_default());
        if oxinfer::depth_pro::is_video_path(path) {
            video(&net, path, &dir, &stem, &vo, frames_fps, png16, raw, profile)?;
            continue;
        }
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let img = oxinfer::depth_pro::decode_image(&bytes)?;
        let (f, source) = match (focal_px, exif.then_some(img.focal_px).flatten()) {
            (Some(f), _) => (Some(f), "given"),
            (None, Some(f)) => (Some(f), "from EXIF"),
            (None, None) => (None, "estimated"),
        };
        let t1 = Instant::now();
        let d = net.infer_rgb(&img.rgb, img.width, img.height, f)?;
        let secs = t1.elapsed().as_secs_f64();
        if profile {
            oxinfer::prof::report(secs, 1);
            eprint!("{}", oxinfer::sd::ops::timing::report());
            oxinfer::prof::reset();
        }
        let mut written = Vec::new();
        let png = dir.join(format!("{stem}-depth.png"));
        std::fs::write(&png, oxinfer::png::encode_rgb(d.width, d.height, &d.colorize()))?;
        written.push(png.display().to_string());
        let (g16, k) = d.gray16();
        if png16 {
            let p = dir.join(format!("{stem}-depth16.png"));
            std::fs::write(&p, oxinfer::png::encode_gray16(d.width, d.height, &g16))?;
            written.push(p.display().to_string());
        }
        if raw && let Some(depth) = d.depth() {
            let p = dir.join(format!("{stem}-depth.f32"));
            std::fs::write(&p, depth.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
            written.push(p.display().to_string());
        }
        println!("Wrote {} ({}x{}, {:.2}s).", written.join(", "), d.width, d.height, secs);
        match d.focal_px {
            Some(f) => println!("Focal length: {f:.1} px, {source}."),
            None => println!("Focal length: none (no estimator): inverse depth only."),
        }
        if let Some((lo, hi, far)) = d.depth_range() {
            println!(
                "Depth: {lo:.2} m to {hi:.2} m (5th to 95th percentile), {:.1}% infinitely far.",
                100.0 * far
            );
        }
        if png16 && d.focal_px.is_some() {
            println!("16-bit PNG: depth in meters = {k:.2} / value, 0 is infinitely far.");
        }
    }
    Ok(())
}

/// One video (a file or a folder of frames): every frame's depth, the files, the summary.
#[allow(clippy::too_many_arguments)]
fn video(
    net: &DepthPro,
    path: &std::path::Path,
    dir: &std::path::Path,
    stem: &str,
    vo: &VideoOptions,
    frames_fps: Option<f64>,
    png16: bool,
    raw: bool,
    profile: bool,
) -> Result<()> {
    #[cfg(feature = "media")]
    {
        let clip = oxinfer::depth_pro::load_video(path, frames_fps)?;
        let frames: Vec<&[u8]> = clip.frames.iter().map(|f| &f[..]).collect();
        let t1 = Instant::now();
        let maps = net.infer_video(&frames, clip.width, clip.height, vo)?;
        let secs = t1.elapsed().as_secs_f64();
        if profile {
            oxinfer::prof::report(secs, maps.len());
            eprint!("{}", oxinfer::sd::ops::timing::report());
            oxinfer::prof::reset();
        }
        let (written, k) = oxinfer::depth_pro::write_video(&maps, dir, stem, clip.fps, png16, raw)?;
        let names: Vec<String> = written.iter().map(|p| p.display().to_string()).collect();
        println!(
            "Wrote {} frames ({}x{}, {} mode, window {}) at {:.5} fps to {} ({:.2}s, {:.2}s a frame).",
            maps.len(),
            clip.width,
            clip.height,
            vo.mode.as_str(),
            vo.window,
            clip.fps,
            names.join(", "),
            secs,
            secs / maps.len().max(1) as f64
        );
        let last = maps.last().context("no frames decoded")?;
        match last.focal_px {
            Some(f) if vo.focal_px.is_some() => println!("Focal length: {f:.1} px, given."),
            Some(f) => println!(
                "Focal length: {f:.1} px, estimated from the first {} frames.",
                vo.window.min(maps.len())
            ),
            None => println!("Focal length: none (no estimator): inverse depth only."),
        }
        if let Some((lo, hi, far)) = last.depth_range() {
            println!(
                "Depth (last frame): {lo:.2} m to {hi:.2} m (5th to 95th percentile), {:.1}% infinitely far.",
                100.0 * far
            );
        }
        if let Some(k) = k {
            println!("16-bit PNGs: depth in meters = {k:.2} / value, 0 is infinitely far.");
        }
        Ok(())
    }
    #[cfg(not(feature = "media"))]
    {
        let _ = (net, dir, stem, vo, frames_fps, png16, raw, profile);
        bail!("{}: video input needs the `media` feature", path.display())
    }
}
