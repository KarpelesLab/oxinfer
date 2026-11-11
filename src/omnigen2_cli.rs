//! `oxinfer --model OmniGen2/OmniGen2 --init-image in.png --prompt "make it winter" -o out.png`:
//! instruction editing (one `--init-image`), in-context generation (several), text to image (none)
//! through `oxinfer::omnigen2`. Without `-o` but with `--image`, the prompt is a question about
//! the images, answered by the model's Qwen2.5-VL (`mllm/`), as for any Qwen2.5-VL chat model.

use crate::Args;
use anyhow::{Context, Result};
use oxinfer::omnigen2::{Options, Params, Pipeline};
use std::io::Read;
use std::path::PathBuf;
use std::time::Instant;

/// Whether these arguments ask a question about images (the understanding side) rather than
/// for an image.
pub fn is_understanding(args: &Args) -> bool {
    args.output.is_none() && args.init_images.is_empty() && (!args.images.is_empty() || args.chat)
}

/// The arguments of the understanding side: the `mllm/` folder as a Qwen2.5-VL chat model (its
/// float32 weights mapped; the tokenizer comes from `processor/`).
pub fn understanding_args(mut args: Args) -> Args {
    args.model = args.model.join("mllm");
    args.mmap = true;
    args
}

pub fn run(args: &Args) -> Result<()> {
    let opts = Options {
        quant: args.quant,
        // Qwen2.5-VL at Q4 moves the conditioning by 43% (Q8: 5%): keep it at Q8 then.
        text_quant: args.flux.text_quant.unwrap_or(if args.quant == oxinfer::model::Quant::Q4 {
            oxinfer::model::Quant::Q8
        } else {
            args.quant
        }),
        mmap: args.mmap,
        transformer: args
            .transformer
            .as_ref()
            .map(|p| oxinfer::engine::resolve_file(&p.to_string_lossy(), args.cache_dir.as_deref()))
            .transpose()?,
        text_encoder: args.flux.text_encoder.clone(),
        keep_text_encoder: false,
        device: args.device,
    };
    let t0 = Instant::now();
    let mut pipe = Pipeline::load(&args.model.to_string_lossy(), opts)?;
    eprintln!(
        "[OmniGen2: {} blocks, transformer {} | VAE loaded in {:.2}s]",
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
    let mut p = Params::new(&prompt);
    if let Some(s) = args.steps {
        p.steps = s;
    }
    if let Some(g) = args.guidance {
        p.text_guidance = g;
    }
    if let Some(g) = args.image_guidance {
        p.image_guidance = g;
    }
    if !args.negative.is_empty() {
        p.negative = args.negative.clone();
    }
    p.seed = args.seed;
    let e = &args.bagel.edit;
    e.only("OmniGen2", true, true, true)?;
    if let Some(s) = &e.scheduler {
        p.scheduler = oxinfer::omnigen2::Scheduler::parse(s).with_context(|| format!("--scheduler expects euler|dpmsolver++, got {s}"))?;
    }
    // As the official script: TaylorSeer wins when both caches are asked for.
    if e.taylorseer {
        if e.teacache.is_some() {
            eprintln!("[--teacache and --taylorseer are mutually exclusive: using TaylorSeer]");
        }
        p = p.with_taylorseer();
    } else if let Some(t) = e.teacache {
        p = p.with_teacache(t);
    }
    p.sample_posterior = e.sample_posterior;
    if let Some((w, h)) = args.size {
        p.width = Some(w);
        p.height = Some(h);
    }
    let mut paths = args.init_images.clone();
    paths.extend(args.images.iter().cloned());
    let images = load_images(&paths)?;
    eprintln!(
        "[{} steps ({}{}), text guidance {}, image guidance {}, seed {}, {} input image{}{}]",
        p.steps,
        p.scheduler.as_str(),
        match p.cache {
            oxinfer::step_cache::StepCache::None => String::new(),
            oxinfer::step_cache::StepCache::TeaCache { rel_l1_thresh } => format!(", TeaCache {rel_l1_thresh}"),
            oxinfer::step_cache::StepCache::TaylorSeer(_) => ", TaylorSeer".to_string(),
        },
        p.text_guidance,
        if images.is_empty() { 1.0 } else { p.image_guidance },
        p.seed,
        images.len(),
        if images.len() == 1 { "" } else { "s" },
        if p.sample_posterior && !images.is_empty() {
            ", sampled VAE posterior"
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
    for (i, s) in tm.posterior_std.iter().enumerate() {
        eprintln!("[input image {}: posterior std / latents (L2) {s:.3e}]", i + 1);
    }
    if let Some(t) = pipe.transformer() {
        eprintln!("[transformer weights {:.2} GB]", t.bytes() as f64 / 1e9);
    }
    let out = args.output.clone().unwrap_or_else(|| PathBuf::from("image.png"));
    std::fs::write(&out, oxinfer::png::encode_rgb(img.width, img.height, &img.rgb)).with_context(|| format!("write {}", out.display()))?;
    eprintln!(
        "[load text {:.1}s, encode {:.1}s | vae encode {:.1}s | load transformer {:.1}s | denoise {:.1}s ({} passes, {:.2}s each{}) | decode {:.1}s | {}x{} -> {}]",
        tm.load_text,
        tm.text,
        tm.vae_encode,
        tm.load_transformer,
        tm.denoise,
        tm.transformer_calls,
        tm.denoise / tm.transformer_calls.max(1) as f64,
        if tm.cached_passes > 0 {
            format!("; {} more from the step cache", tm.cached_passes)
        } else {
            String::new()
        },
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
