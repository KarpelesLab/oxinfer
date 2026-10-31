//! `oxinfer --model stepfun-ai/Step1X-Edit-v1p2 --init-image in.png --prompt "make it winter" -o
//! out.png`: Step1X-Edit instruction-based editing through `oxinfer::step1x` (without
//! `--init-image`: text to image).

use crate::Args;
use anyhow::{Context, Result};
use oxinfer::step1x::{Options, Params, Pipeline};
use std::io::Read;
use std::path::PathBuf;
use std::time::Instant;

pub fn run(args: &Args) -> Result<()> {
    let stream = args.stream.then(|| {
        let mut o = oxinfer::stream::StreamOptions::default().with_env();
        o.cache_budget = args.cache_budget_gb.map(|g| (g * 1e9) as u64);
        o.resident_bytes = args.resident_mb.map(|m| m << 20);
        if let Some(p) = args.prefetch_mb {
            o.prefetch_bytes = p << 20;
        }
        o.resident_mode = args.resident_mode;
        if let Some(f) = args.flash_mbps {
            o.flash_bytes_per_sec = Some(f * 1e6);
        }
        o
    });
    let opts = Options {
        quant: args.quant,
        // The text encoder at Q4 moves its hidden states visibly: keep it at Q8 then.
        text_quant: if args.quant == oxinfer::model::Quant::Q4 {
            oxinfer::model::Quant::Q8
        } else {
            args.quant
        },
        mmap: args.mmap,
        transformer: args
            .transformer
            .as_ref()
            .map(|p| oxinfer::engine::resolve_file(&p.to_string_lossy(), args.cache_dir.as_deref()))
            .transpose()?,
        stream,
        device: args.device,
        // OXINFER_EXACT_ATTENTION=0: the GPU's half-precision attention kernels.
        exact_attention: std::env::var("OXINFER_EXACT_ATTENTION").map_or(true, |v| v != "0"),
        ..Default::default()
    };
    let t0 = Instant::now();
    let mut pipe = Pipeline::load(&args.model.to_string_lossy(), opts)?;
    eprintln!(
        "[Step1X-Edit {:?}: {} double + {} single blocks, transformer {} | VAE loaded in {:.2}s]",
        pipe.version,
        pipe.tcfg.num_layers,
        pipe.tcfg.num_single_layers,
        match &args.transformer {
            Some(p) => p.display().to_string(),
            None => args.quant.as_str().to_string(),
        },
        t0.elapsed().as_secs_f64()
    );
    let prompt = match &args.prompt {
        Some(p) => p.clone(),
        None => {
            let mut s = String::new();
            std::io::stdin().read_to_string(&mut s)?;
            s.trim().to_string()
        }
    };
    let mut p = Params::new(&prompt);
    if let Some(s) = args.steps {
        p.steps = s;
    }
    if let Some(g) = args.guidance {
        p.true_cfg = g;
    }
    if !args.negative.is_empty() {
        p.negative = Some(args.negative.clone());
    }
    p.seed = args.seed;
    p.think = args.think;
    // --size WxH: the output size, and the working resolution's area (the input's aspect ratio
    // is kept while denoising: the reference latents share the target's grid).
    if let Some((w, h)) = args.size {
        p.width = Some(w);
        p.height = Some(h);
        p.size_level = ((w * h) as f64).sqrt().round() as usize;
    }
    let image = match args.init_images.first().or(args.images.first()) {
        None => None,
        Some(path) => {
            #[cfg(feature = "media")]
            {
                Some(
                    oxinfer::media::Media::new()
                        .load_image(path)
                        .with_context(|| format!("load {}", path.display()))?,
                )
            }
            #[cfg(not(feature = "media"))]
            anyhow::bail!("{}: decoding images needs the `media` feature", path.display())
        }
    };
    // A --size of another aspect ratio: the input is center-cropped to it first.
    let image = match (image, args.size) {
        (Some(img), Some((w, h))) => Some(crop_to_aspect(img, w, h)),
        (img, _) => img,
    };
    if image.is_none() {
        eprintln!("[no --init-image: generating from the prompt alone]");
    }
    eprintln!(
        "[{} steps, true CFG {} (negative {:?}), size level {}, seed {}{}]",
        p.steps,
        p.true_cfg,
        p.negative.as_deref().unwrap_or(if image.is_some() { "" } else { "<default>" }),
        p.size_level,
        p.seed,
        if p.think { ", thinking" } else { "" }
    );
    let tp = Instant::now();
    let mut progress = |i: usize, n: usize| eprint!("\r[step {i}/{n}, {:.1}s]", tp.elapsed().as_secs_f64());
    let (img, tm) = match &image {
        Some(im) => pipe.edit(im, &p, &mut progress)?,
        None => pipe.generate(&p, &mut progress)?,
    };
    eprintln!();
    if let Some(r) = &tm.rewritten {
        eprintln!("[rewritten instruction: {r}]");
    }
    let out = args.output.clone().unwrap_or_else(|| PathBuf::from("image.png"));
    std::fs::write(&out, oxinfer::png::encode_rgb(img.width, img.height, &img.rgb)).with_context(|| format!("write {}", out.display()))?;
    eprintln!(
        "[load text {:.1}s{}, encode {:.1}s | vae encode {:.1}s | load transformer {:.1}s | denoise {:.1}s ({} passes of {} tokens, {:.2}s each) at {}x{} | decode {:.1}s | {}x{} -> {}]",
        tm.load_text,
        if p.think {
            format!(", think {:.1}s", tm.think)
        } else {
            String::new()
        },
        tm.text,
        tm.vae_encode,
        tm.load_transformer,
        tm.denoise,
        tm.transformer_calls,
        tm.tokens,
        tm.denoise / tm.transformer_calls.max(1) as f64,
        tm.size.0,
        tm.size.1,
        tm.decode,
        img.width,
        img.height,
        out.display()
    );
    if args.profile {
        eprint!("[time by operation]\n{}", oxinfer::sd::ops::timing::report());
        if let Some(s) = oxinfer::gpu::active().and_then(|b| b.stats()) {
            eprintln!("  gpu: {s}");
        }
    }
    Ok(())
}

/// The largest centered `w:h` crop of `img` (unchanged when the aspect ratio already matches
/// within a pixel).
fn crop_to_aspect(img: oxinfer::video::RgbImage, w: usize, h: usize) -> oxinfer::video::RgbImage {
    let (iw, ih) = (img.width, img.height);
    let (cw, ch) = if iw * h > ih * w {
        ((ih * w).div_ceil(h).min(iw), ih)
    } else {
        (iw, (iw * h).div_ceil(w).min(ih))
    };
    if iw.abs_diff(cw) <= 1 && ih.abs_diff(ch) <= 1 {
        return img;
    }
    let (x0, y0) = ((iw - cw) / 2, (ih - ch) / 2);
    let mut data = Vec::with_capacity(cw * ch * 3);
    for y in y0..y0 + ch {
        data.extend_from_slice(&img.data[(y * iw + x0) * 3..(y * iw + x0 + cw) * 3]);
    }
    oxinfer::video::RgbImage {
        width: cw,
        height: ch,
        data,
    }
}
