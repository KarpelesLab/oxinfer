//! `oxinfer --model unsloth/FLUX.1-Kontext-dev --init-image in.png -p "make it night" -o out.png`:
//! FLUX.1 Kontext editing through `oxinfer::flux` (without `--init-image`: text to image), from a
//! diffusers folder, a FLUX GGUF transformer (`--model QuantStack/FLUX.1-Kontext-dev-GGUF:Q4_0`) or
//! a BFL single-file checkpoint (`--model flux1-dev.safetensors`). `--strength` / `--mask` make
//! `--init-image` an image-to-image / inpainting (FLUX.1 Fill) input, `--control-image` feeds
//! FLUX.1 Canny / Depth, `--image-prompt` FLUX.1 Redux, `--lora` applies LoRA files.

use crate::Args;
use anyhow::{Context, Result};
use oxinfer::flux::{Inputs, LoadOptions, Options, Pipeline, RefLayout};
use oxinfer::model::Quant;
use std::io::Read;
use std::path::PathBuf;
use std::time::Instant;

/// FLUX-specific flags.
#[derive(Default, Debug)]
pub struct FluxArgs {
    pub transformer: Option<String>,
    pub text_encoder: Option<String>,
    pub base: Option<String>,
    pub max_area: Option<usize>,
    pub no_auto_resize: bool,
    pub text_quant: Option<Quant>,
    pub ref_layout: Option<RefLayout>,
    pub control_image: Option<PathBuf>,
    pub image_prompts: Vec<PathBuf>,
    pub image_prompt_scale: Option<f32>,
    pub vae: Option<String>,
    pub redux: Option<String>,
}

impl FluxArgs {
    /// Takes `flag` (with its value from `val`) if it is a FLUX flag.
    pub fn parse(&mut self, flag: &str, val: &mut dyn FnMut() -> Result<String>) -> Result<bool> {
        match flag {
            "--transformer" => self.transformer = Some(val()?),
            "--text-encoder" => self.text_encoder = Some(val()?),
            "--base" => self.base = Some(val()?),
            "--max-area" => self.max_area = Some(val()?.parse()?),
            "--no-auto-resize" => self.no_auto_resize = true,
            "--ref-layout" => {
                let v = val()?;
                self.ref_layout =
                    Some(RefLayout::parse(&v).with_context(|| format!("--ref-layout expects offset|index|diffusers, got {v}"))?);
            }
            "--control-image" => self.control_image = Some(PathBuf::from(val()?)),
            "--image-prompt" => self.image_prompts.push(PathBuf::from(val()?)),
            "--image-prompt-scale" => self.image_prompt_scale = Some(val()?.parse()?),
            "--vae" => self.vae = Some(val()?),
            "--redux" => self.redux = Some(val()?),
            "--text-quant" => {
                let v = val()?;
                self.text_quant = Some(Quant::parse(&v).with_context(|| format!("--text-quant expects bf16|q8|q4, got {v}"))?);
            }
            _ => return Ok(false),
        }
        Ok(true)
    }
}

pub const USAGE: &str = "\
FLUX.1 (a diffusers FLUX folder or repo, e.g. unsloth/FLUX.1-Kontext-dev, a FLUX GGUF\n\
transformer such as QuantStack/FLUX.1-Kontext-dev-GGUF:Q4_0 or city96/FLUX.1-schnell-gguf:Q4_0,\n\
or a BFL single-file checkpoint such as flux1-dev.safetensors):\n\
\x20 --init-image FILE     Kontext: edit this image following --prompt (else: text to image);\n\
\x20                      repeat it for several reference images\n\
\x20 --ref-layout L       several references: offset (default, ComfyUI's) | index | diffusers\n\
\x20 --strength F         image to image from --init-image instead of a Kontext edit: how much\n\
\x20                      is redrawn, 0..1 (default 0.6; 1 for FLUX.1 Fill)\n\
\x20 --mask FILE          inpaint --init-image: white = repaint (FLUX.1 Fill, or any FLUX.1 by\n\
\x20                      blending the kept region back at every step)\n\
\x20 --control-image FILE FLUX.1 Canny / Depth (or [dev] with BFL's control LoRA): the edge / depth map\n\
\x20 --image-prompt FILE  FLUX.1 Redux: an image as (part of) the prompt; needs --redux; repeatable\n\
\x20 --redux SPEC         the FLUX.1 Redux folder or repo (image_encoder/ + image_embedder/)\n\
\x20 --lora FILE[:SCALE]  apply a LoRA to the transformer (diffusers, kohya, BFL / XLabs keys); repeatable\n\
\x20 --vae SPEC           the autoencoder: BFL's ae.safetensors or a diffusers vae folder\n\
\x20 -o, --output FILE     PNG to write (default image.png)\n\
\x20 --steps N            denoising steps (default 28; 4 for schnell)\n\
\x20 --guidance F         distilled guidance (default 3.5; BFL suggests 2.5 for edits)\n\
\x20 --size WxH           output size, rounded to multiples of 16 (default: the input's aspect\n\
\x20                      ratio at --max-area pixels; text to image 1024x1024)\n\
\x20 --max-area N         editing: output pixels (default 1048576 = 1024^2)\n\
\x20 --no-auto-resize     editing: keep the input size (rounded to 16) instead of the closest\n\
\x20                      preferred Kontext resolution (about 1 MP)\n\
\x20 --quant Q            transformer precision when stored as float: bf16 (default) | q8 | q4\n\
\x20 --text-quant Q       T5 encoder precision (default: --quant)\n\
\x20 --transformer SPEC   use this transformer (GGUF file or org/name:QUANT) with --model's other parts\n\
\x20 --text-encoder SPEC a T5 encoder GGUF (e.g. city96/t5-v1_1-xxl-encoder-gguf:Q8_0) instead of the base's\n\
\x20 --base SPEC          with a GGUF --model: where the text encoders / VAE come from\n\
\x20                      (default unsloth/FLUX.1-Kontext-dev)\n\
\x20 --seed N             noise seed (default 42)\n";

fn load_image(p: &std::path::Path) -> Result<oxinfer::video::RgbImage> {
    #[cfg(feature = "media")]
    {
        oxinfer::media::Media::new()
            .load_image(p)
            .with_context(|| format!("load {}", p.display()))
    }
    #[cfg(not(feature = "media"))]
    {
        anyhow::bail!("{}: decoding images needs the `media` feature", p.display())
    }
}

pub fn run(args: &Args) -> Result<()> {
    let f = &args.flux;
    // The images to edit: every --init-image (shared with Stable Diffusion editing), or the --image ones.
    let init: &[PathBuf] = if args.init_images.is_empty() {
        &args.images
    } else {
        &args.init_images
    };
    let opts = LoadOptions {
        quant: args.quant,
        text_quant: f.text_quant,
        transformer: f.transformer.clone(),
        text_encoder: f.text_encoder.clone(),
        base: f.base.clone(),
        device: args.device,
        cache_dir: args.cache_dir.clone(),
        vae_encoder: !init.is_empty() || f.control_image.is_some(),
        loras: args
            .loras
            .iter()
            .map(|l| oxinfer::flux::lora::LoraSpec {
                path: l.path.clone(),
                scale: l.scale,
            })
            .collect(),
        vae: f.vae.clone(),
        redux: f.redux.clone(),
    };
    if !f.image_prompts.is_empty() && f.redux.is_none() {
        anyhow::bail!("--image-prompt needs the Redux encoder: --redux DIR (a FLUX.1-Redux-dev folder or repo)");
    }
    let t0 = Instant::now();
    let pipe = Pipeline::load(&args.model.to_string_lossy(), &opts)?;
    let c = &pipe.transformer.cfg;
    let variant = match c.in_channels / 64 {
        6 => ", Fill",
        2 => ", control (Canny / Depth)",
        _ => "",
    };
    eprintln!(
        "[FLUX.1 ({}): {} double + {} single blocks, {}{}{variant} | transformer {:.1} GB, T5 {:.1} GB | loaded in {:.1}s]",
        pipe.class,
        c.num_layers,
        c.num_single_layers,
        if pipe.transformer.layout == oxinfer::flux::transformer::Layout::Bfl {
            "GGUF/BFL layout"
        } else {
            args.quant.as_str()
        },
        if c.guidance_embeds { ", guidance-distilled" } else { "" },
        pipe.transformer.weight_bytes as f64 / 1e9,
        pipe.t5.weight_bytes as f64 / 1e9,
        t0.elapsed().as_secs_f64()
    );
    for l in &pipe.lora_report {
        eprintln!("{l}");
    }
    let prompt = match &args.prompt {
        Some(p) => p.clone(),
        None => {
            let mut s = String::new();
            std::io::stdin().read_to_string(&mut s)?;
            s.trim().to_string()
        }
    };
    let mut o = Options {
        steps: args.steps,
        guidance: args.guidance,
        seed: args.seed,
        auto_resize: !f.no_auto_resize,
        ref_layout: f.ref_layout.unwrap_or_default(),
        ..Default::default()
    };
    if let Some(s) = f.image_prompt_scale {
        o.image_prompt_scale = s;
    }
    // An explicit --size is the output area too, unless --max-area says otherwise (diffusers
    // would rescale it to 1024x1024 pixels).
    if let Some(m) = f.max_area.or(args.size.map(|(w, h)| w * h)) {
        o.max_area = m;
    }
    let mut images = init.iter().map(|p| load_image(p)).collect::<Result<Vec<_>>>()?;
    let mask = args.mask.as_deref().map(load_image).transpose()?;
    let control = f.control_image.as_deref().map(load_image).transpose()?;
    let image_prompts = f.image_prompts.iter().map(|p| load_image(p)).collect::<Result<Vec<_>>>()?;
    // --init-image is a Kontext reference unless --strength / --mask (or a Fill transformer) make
    // it the image an image-to-image / inpainting run starts from.
    let img2img = !images.is_empty() && (args.strength.is_some() || mask.is_some() || c.in_channels == 384);
    if mask.is_some() && images.is_empty() {
        anyhow::bail!("--mask needs the image to inpaint (--init-image FILE)");
    }
    if img2img {
        o.strength = args.strength;
        (o.width, o.height) = (args.size.map(|s| s.0), args.size.map(|s| s.1));
    } else if !images.is_empty() {
        // Editing keeps the input's aspect ratio unless --size says otherwise. With --size the
        // references are resized too: Kontext gives the reference tokens the same positions as the
        // output grid, so a ~1 MP reference under a smaller output comes out zoomed and shifted.
        // The first takes the output size, the others its area at their own aspect ratio.
        if let Some((w, h)) = args.size
            && !f.no_auto_resize
        {
            let (w, h) = (w / 16 * 16, h / 16 * 16);
            o.auto_resize = false;
            for (i, img) in images.iter_mut().enumerate() {
                let (tw, th) = if i == 0 {
                    (w, h)
                } else {
                    oxinfer::flux::kontext_size(img.width, img.height, w * h)
                };
                let data = oxinfer::resample::resize_lanczos_rgb8(&img.data, img.width, img.height, tw, th);
                *img = oxinfer::video::RgbImage {
                    width: tw,
                    height: th,
                    data,
                };
            }
        }
        let (w, h) = args.size.unwrap_or((images[0].width, images[0].height));
        (o.width, o.height) = (Some(w), Some(h));
    } else {
        (o.width, o.height) = (args.size.map(|s| s.0), args.size.map(|s| s.1));
    }
    let inputs = Inputs {
        references: if img2img { &[] } else { &images },
        init: if img2img { images.first() } else { None },
        mask: mask.as_ref(),
        control: control.as_ref(),
        image_prompts: &image_prompts,
    };
    let steps = o.steps.unwrap_or(if pipe.has_guidance() { 28 } else { 4 });
    let mode = match (&inputs, c.in_channels) {
        (i, 384) if i.init.is_some() => "fill".to_string(),
        (i, _) if i.mask.is_some() => format!("inpainting (mask blending), strength {}", o.strength.unwrap_or(0.6)),
        (i, _) if i.init.is_some() => format!("image to image, strength {}", o.strength.unwrap_or(0.6)),
        (i, _) if i.references.len() > 1 => format!("edit, {} references ({:?} layout)", i.references.len(), o.ref_layout).to_lowercase(),
        (i, _) if !i.references.is_empty() => "edit".to_string(),
        (i, _) if i.control.is_some() => "control".to_string(),
        _ => "text to image".to_string(),
    };
    eprintln!(
        "[{mode}{}: {} steps, guidance {}, seed {}]",
        if image_prompts.is_empty() {
            String::new()
        } else {
            format!(", {} image prompt(s)", image_prompts.len())
        },
        steps,
        if pipe.has_guidance() {
            format!("{}", o.guidance.unwrap_or(3.5))
        } else {
            "none".into()
        },
        o.seed
    );
    let tp = Instant::now();
    let mut progress = |i: usize, n: usize| {
        eprint!("\r[step {i}/{n}, {:.1}s]", tp.elapsed().as_secs_f64());
        true
    };
    let (img, tm) = pipe.render(&prompt, &inputs, &o, &mut progress)?;
    eprintln!();
    let out = args.output.clone().unwrap_or_else(|| PathBuf::from("image.png"));
    std::fs::write(&out, oxinfer::png::encode_rgb(img.width, img.height, &img.rgb)).with_context(|| format!("write {}", out.display()))?;
    eprintln!(
        "[{}x{} | text {:.2}s | encode {:.2}s | denoise {:.2}s ({} transformer passes of {} tokens, {:.2}s each) | decode {:.2}s | wrote {}]",
        img.width,
        img.height,
        tm.text,
        tm.encode,
        tm.denoise,
        tm.transformer_calls,
        tm.tokens,
        tm.denoise / tm.transformer_calls.max(1) as f64,
        tm.decode,
        out.display()
    );
    if args.profile {
        eprint!("[time by operation]\n{}", oxinfer::sd::ops::timing::report());
        if let Some(s) = oxinfer::gpu::active().and_then(|b| b.stats()) {
            eprintln!("  gpu: {s}");
        }
    }
    if oxinfer::sd::ops::gpu_check::on() {
        eprintln!("[gpu check: {}]", oxinfer::sd::ops::gpu_check::summary());
    }
    Ok(())
}
