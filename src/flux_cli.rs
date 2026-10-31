//! `oxinfer --model unsloth/FLUX.1-Kontext-dev --init-image in.png -p "make it night" -o out.png`:
//! FLUX.1 Kontext editing through `oxinfer::flux` (without `--init-image`: text to image), from a
//! diffusers folder or a FLUX GGUF transformer (`--model QuantStack/FLUX.1-Kontext-dev-GGUF:Q4_0`).

use crate::Args;
use anyhow::{Context, Result};
use oxinfer::flux::{LoadOptions, Options, Pipeline};
use oxinfer::model::Quant;
use std::io::Read;
use std::path::PathBuf;
use std::time::Instant;

/// FLUX-specific flags.
#[derive(Default, Debug)]
pub struct FluxArgs {
    pub init_image: Option<PathBuf>,
    pub transformer: Option<String>,
    pub base: Option<String>,
    pub max_area: Option<usize>,
    pub no_auto_resize: bool,
    pub text_quant: Option<Quant>,
}

impl FluxArgs {
    /// Takes `flag` (with its value from `val`) if it is a FLUX flag.
    pub fn parse(&mut self, flag: &str, val: &mut dyn FnMut() -> Result<String>) -> Result<bool> {
        match flag {
            "--init-image" => self.init_image = Some(PathBuf::from(val()?)),
            "--transformer" => self.transformer = Some(val()?),
            "--base" => self.base = Some(val()?),
            "--max-area" => self.max_area = Some(val()?.parse()?),
            "--no-auto-resize" => self.no_auto_resize = true,
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
FLUX.1 (a diffusers FLUX folder or repo, e.g. unsloth/FLUX.1-Kontext-dev, or a FLUX GGUF\n\
transformer such as QuantStack/FLUX.1-Kontext-dev-GGUF:Q4_0 or city96/FLUX.1-schnell-gguf:Q4_0):\n\
\x20 --init-image FILE     Kontext: edit this image following --prompt (else: text to image)\n\
\x20 -o, --output FILE     PNG to write (default image.png)\n\
\x20 --steps N            denoising steps (default 28; 4 for schnell)\n\
\x20 --guidance F         distilled guidance (default 3.5; BFL suggests 2.5 for edits)\n\
\x20 --size WxH           output size (editing: rescaled to --max-area at that aspect ratio;\n\
\x20                      default the input image's size, text to image 1024x1024)\n\
\x20 --max-area N         editing: output pixels (default 1048576 = 1024^2)\n\
\x20 --no-auto-resize     editing: keep the input size (rounded to 16) instead of the closest\n\
\x20                      preferred Kontext resolution (about 1 MP)\n\
\x20 --quant Q            transformer precision when stored as float: bf16 (default) | q8 | q4\n\
\x20 --text-quant Q       T5 encoder precision (default: --quant)\n\
\x20 --transformer SPEC   use this transformer (GGUF file or org/name:QUANT) with --model's other parts\n\
\x20 --base SPEC          with a GGUF --model: where the text encoders / VAE come from\n\
\x20                      (default unsloth/FLUX.1-Kontext-dev)\n\
\x20 --seed N             noise seed (default 42)\n";

pub fn run(args: &Args) -> Result<()> {
    let f = &args.flux;
    let opts = LoadOptions {
        quant: args.quant,
        text_quant: f.text_quant,
        transformer: f.transformer.clone(),
        base: f.base.clone(),
        device: args.device,
        cache_dir: args.cache_dir.clone(),
        vae_encoder: f.init_image.is_some(),
    };
    let t0 = Instant::now();
    let pipe = Pipeline::load(&args.model.to_string_lossy(), &opts)?;
    let c = &pipe.transformer.cfg;
    eprintln!(
        "[FLUX.1 ({}): {} double + {} single blocks, {}{} | transformer {:.1} GB, T5 {:.1} GB | loaded in {:.1}s]",
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
        ..Default::default()
    };
    if let Some(m) = f.max_area {
        o.max_area = m;
    }
    let image = match &f.init_image {
        None => None,
        Some(p) => {
            #[cfg(feature = "media")]
            {
                Some(
                    oxinfer::media::Media::new()
                        .load_image(p)
                        .with_context(|| format!("load {}", p.display()))?,
                )
            }
            #[cfg(not(feature = "media"))]
            {
                anyhow::bail!("{}: decoding images needs the `media` feature", p.display())
            }
        }
    };
    // Editing keeps the input's aspect ratio unless --size says otherwise.
    let (w, h) = args.size.or(image.as_ref().map(|i| (i.width, i.height))).unwrap_or((1024, 1024));
    (o.width, o.height) = (Some(w), Some(h));
    let (steps, mode) = (
        o.steps.unwrap_or(if pipe.has_guidance() { 28 } else { 4 }),
        if image.is_some() { "edit" } else { "text to image" },
    );
    eprintln!(
        "[{mode}: {} steps, guidance {}, seed {}]",
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
    let (img, tm) = match &image {
        Some(im) => pipe.edit(im, &prompt, &o, &mut progress)?,
        None => pipe.generate(&prompt, &o, &mut progress)?,
    };
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
