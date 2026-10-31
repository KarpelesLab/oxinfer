//! `oxinfer --model Qwen/Qwen-Image-Edit-2511 --init-image in.png --prompt "make it winter" -o out.png`:
//! instruction-based editing (one or more `--init-image`) and text to image through
//! `oxinfer::qwen_image`.

use crate::Args;
use anyhow::{Context, Result};
use oxinfer::qwen_image::{Kind, LoraSpec, Options, Pipeline};
use std::io::Read;
use std::path::PathBuf;
use std::time::Instant;

pub fn run(args: &Args) -> Result<()> {
    let loras: Vec<LoraSpec> = args
        .loras
        .iter()
        .map(|l| LoraSpec {
            path: l.path.clone(),
            scale: l.scale,
        })
        .collect();
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
        // The text encoder at Q4 moves its hidden states by ~25%: keep it at Q8 then.
        text_quant: if args.quant == oxinfer::model::Quant::Q4 {
            oxinfer::model::Quant::Q8
        } else {
            args.quant
        },
        mmap: args.mmap,
        loras: loras.clone(),
        transformer: args
            .transformer
            .as_ref()
            .map(|p| oxinfer::engine::resolve_file(&p.to_string_lossy(), args.cache_dir.as_deref()))
            .transpose()?,
        stream,
        keep_text_encoder: false,
        device: args.device,
    };
    let t0 = Instant::now();
    let mut pipe = Pipeline::load(&args.model.to_string_lossy(), opts)?;
    eprintln!(
        "[Qwen-Image {:?}: {} blocks, transformer {} | VAE loaded in {:.2}s]",
        pipe.kind,
        pipe.tcfg.num_layers,
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
    let lightning = pipe.opts.lightning_steps();
    let mut p = pipe.default_params(&prompt);
    if let Some(s) = args.steps {
        p.steps = s;
    }
    if let Some(g) = args.guidance {
        p.true_cfg = g;
        if g > 1.0 && p.negative.is_none() {
            p.negative = Some(" ".into());
        }
    }
    if !args.negative.is_empty() {
        p.negative = Some(args.negative.clone());
    }
    p.seed = args.seed;
    if let Some((w, h)) = args.size {
        p.width = Some(w);
        p.height = Some(h);
    }
    let images = load_images(&args.init_images)?;
    if images.is_empty() && pipe.kind != Kind::TextToImage {
        eprintln!("[no --init-image: generating from the prompt alone]");
    }
    eprintln!(
        "[{} steps, true CFG {}{}, seed {}{}]",
        p.steps,
        p.true_cfg,
        if p.true_cfg > 1.0 {
            format!(" (negative {:?})", p.negative.as_deref().unwrap_or(""))
        } else {
            String::new()
        },
        p.seed,
        if lightning.is_some() {
            ", Lightning scheduler (shift 3)"
        } else {
            ""
        }
    );
    let mut progress = |i: usize, n: usize| eprint!("\r[step {i}/{n}]");
    let (img, tm) = if images.is_empty() {
        pipe.generate(&p, &mut progress)?
    } else {
        pipe.edit(&images, &p, &mut progress)?
    };
    eprintln!();
    for l in &pipe.lora_report {
        eprintln!("{l}");
    }
    let out = args.output.clone().unwrap_or_else(|| PathBuf::from("image.png"));
    std::fs::write(&out, oxinfer::png::encode_rgb(img.width, img.height, &img.rgb)).with_context(|| format!("write {}", out.display()))?;
    eprintln!(
        "[load text {:.1}s, encode {:.1}s | vae encode {:.1}s | load transformer {:.1}s | denoise {:.1}s ({} passes, {:.2}s each) | decode {:.1}s | {}x{} -> {}]",
        tm.load_text,
        tm.text,
        tm.vae_encode,
        tm.load_transformer,
        tm.denoise,
        tm.transformer_calls,
        tm.denoise / tm.transformer_calls.max(1) as f64,
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

fn load_images(paths: &[PathBuf]) -> Result<Vec<oxinfer::video::RgbImage>> {
    if paths.is_empty() {
        return Ok(Vec::new());
    }
    #[cfg(feature = "media")]
    {
        let media = oxinfer::media::Media::new();
        paths
            .iter()
            .map(|p| media.load_image(p).with_context(|| format!("load {}", p.display())))
            .collect()
    }
    #[cfg(not(feature = "media"))]
    anyhow::bail!("built without the `media` feature: input images cannot be decoded")
}
