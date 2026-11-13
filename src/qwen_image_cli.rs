//! `oxinfer --model Qwen/Qwen-Image-Edit-2511 --init-image in.png --prompt "make it winter" -o out.png`:
//! instruction-based editing (one or more `--init-image`), text to image, image to image and
//! inpainting (`--strength`, `--mask`), ControlNet (`--controlnet`, `--control-image`) and layer
//! decomposition (Qwen-Image-Layered, `--layers`) through `oxinfer::qwen_image`.

use crate::Args;
use anyhow::{Context, Result};
use oxinfer::qwen_image::{Control, Kind, LoraSpec, Options, Pipeline};
use std::io::Read;
use std::path::PathBuf;
use std::time::Instant;

/// The command line's Qwen-Image-only flags.
#[derive(Default)]
pub struct Flags {
    controlnet: Option<PathBuf>,
    control_images: Vec<PathBuf>,
    control_scales: Vec<f32>,
    control_start: Option<f32>,
    control_end: Option<f32>,
    layers: Option<usize>,
    resolution: Option<usize>,
}

impl Flags {
    pub fn set(&mut self, flag: &str, value: &str) -> Result<()> {
        match flag {
            "--controlnet" => self.controlnet = Some(PathBuf::from(value)),
            "--control-image" => self.control_images.push(PathBuf::from(value)),
            "--control-scale" => self.control_scales.push(value.parse()?),
            "--control-start" => self.control_start = Some(value.parse()?),
            "--control-end" => self.control_end = Some(value.parse()?),
            "--layers" => self.layers = Some(value.parse()?),
            "--resolution" => self.resolution = Some(value.parse()?),
            _ => anyhow::bail!("unknown flag {flag}"),
        }
        Ok(())
    }
}

pub fn run(args: &Args) -> Result<()> {
    let fl = &args.qwen_image;
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
        controlnet: fl.controlnet.clone(),
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
        // Qwen-Image-Layered without a prompt captions the image itself.
        None if pipe.kind == Kind::Layered => String::new(),
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
    if pipe.kind == Kind::Layered {
        return decompose(args, &mut pipe, p, images);
    }
    if images.is_empty() && !matches!(pipe.kind, Kind::TextToImage | Kind::V21) {
        eprintln!("[no --init-image: generating from the prompt alone]");
    }
    p.strength = args.strength;
    if let Some(m) = &args.mask {
        p.mask = load_images(std::slice::from_ref(m))?.pop();
    }
    if !matches!(fl.control_scales.len(), 0 | 1) && fl.control_scales.len() != fl.control_images.len() {
        anyhow::bail!(
            "--control-scale: {} values for {} control images",
            fl.control_scales.len(),
            fl.control_images.len()
        );
    }
    for (i, image) in load_images(&fl.control_images)?.into_iter().enumerate() {
        let scale = *fl.control_scales.get(i).or(fl.control_scales.first()).unwrap_or(&1.0);
        p.control.push(Control { image, scale });
    }
    p.control_start = fl.control_start.unwrap_or(p.control_start);
    p.control_end = fl.control_end.unwrap_or(p.control_end);
    if !p.control.is_empty() {
        eprintln!(
            "[ControlNet {}: {} control image(s), scale {:?}, steps {}..{} of the schedule{}]",
            fl.controlnet.as_ref().map(|c| c.display().to_string()).unwrap_or_default(),
            p.control.len(),
            p.control.iter().map(|c| c.scale).collect::<Vec<_>>(),
            p.control_start,
            p.control_end,
            if p.mask.is_some() { ", inpainting mask" } else { "" }
        );
    } else if !images.is_empty() && (pipe.kind == Kind::TextToImage || p.strength.is_some() || p.mask.is_some()) {
        eprintln!(
            "[{}, strength {}]",
            if p.mask.is_some() {
                "inpainting (mask blending)"
            } else {
                "image to image"
            },
            p.strength.unwrap_or(if pipe.kind == Kind::TextToImage {
                oxinfer::qwen_image::DEFAULT_STRENGTH
            } else {
                1.0
            })
        );
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
        match (&pipe.sample_sigmas, lightning) {
            (Some(s), _) if s.len() == p.steps => ", the checkpoint's saved sigmas",
            (Some(_), _) => ", uniform sigmas (not the checkpoint's saved schedule)",
            (None, Some(_)) => ", Lightning scheduler (shift 3)",
            (None, None) => "",
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
    // An RGBA model (Qwen-Image-2.1) that drew transparency: a PNG with its alpha channel (an
    // opaque picture's alpha still dips a few levels under 255 here and there).
    let png = match &pipe.alpha {
        Some(alpha) if alpha.iter().any(|&v| v < 128) => {
            let rgba: Vec<u8> = img
                .rgb
                .as_chunks::<3>()
                .0
                .iter()
                .zip(alpha)
                .flat_map(|(p, &a)| [p[0], p[1], p[2], a])
                .collect();
            eprintln!("[the image has transparency: writing RGBA]");
            oxinfer::png::encode_rgba(img.width, img.height, &rgba)
        }
        _ => oxinfer::png::encode_rgb(img.width, img.height, &img.rgb),
    };
    std::fs::write(&out, png).with_context(|| format!("write {}", out.display()))?;
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

/// Qwen-Image-Layered: `--init-image` into `--layers` RGBA PNG files (`-o out.png` writes
/// `out_0.png`, `out_1.png`, ...).
fn decompose(args: &Args, pipe: &mut Pipeline, mut p: oxinfer::qwen_image::Params, images: Vec<oxinfer::video::RgbImage>) -> Result<()> {
    let fl = &args.qwen_image;
    let [image] = &images[..] else {
        anyhow::bail!("Qwen-Image-Layered decomposes one image (--init-image FILE)");
    };
    if args.steps.is_none() {
        p.steps = 50;
    }
    // The layered pipeline's default: guidance without the norm rescale.
    p.cfg_normalize = false;
    p.layers = fl.layers.unwrap_or(p.layers);
    p.resolution = fl.resolution.unwrap_or(p.resolution);
    eprintln!(
        "[{} layers at resolution {}, {} steps, true CFG {}, seed {}]",
        p.layers, p.resolution, p.steps, p.true_cfg, p.seed
    );
    let mut progress = |i: usize, n: usize| eprint!("\r[step {i}/{n}]");
    let (layers, tm) = pipe.decompose(image, &p, &mut progress)?;
    eprintln!();
    if let (true, Some(c)) = (p.prompt.trim().is_empty(), &pipe.caption) {
        eprintln!("[caption: {c}]");
    }
    let out = args.output.clone().unwrap_or_else(|| PathBuf::from("layer.png"));
    let stem = out.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    for (i, l) in layers.iter().enumerate() {
        let f = out.with_file_name(format!("{stem}_{i}.png"));
        std::fs::write(&f, oxinfer::png::encode_rgba(l.width, l.height, &l.rgba)).with_context(|| format!("write {}", f.display()))?;
        eprintln!("[layer {i}: {}x{} -> {}]", l.width, l.height, f.display());
    }
    eprintln!(
        "[load text {:.1}s, encode {:.1}s | vae encode {:.1}s | load transformer {:.1}s | denoise {:.1}s ({} passes, {:.2}s each) | decode {:.1}s]",
        tm.load_text,
        tm.text,
        tm.vae_encode,
        tm.load_transformer,
        tm.denoise,
        tm.transformer_calls,
        tm.denoise / tm.transformer_calls.max(1) as f64,
        tm.decode
    );
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
