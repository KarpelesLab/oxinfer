//! `oxinfer --model DIR -p "a red fox in the snow" -o fox.png`: Stable Diffusion 3 / 3.5 text to
//! image through `oxinfer::sd3` (with `--init-image`: image to image), from a diffusers folder or
//! an SD 3 GGUF transformer (`--model city96/stable-diffusion-3.5-medium-gguf:Q4_0 --base DIR`).
//! `--transformer`, `--text-encoder`, `--base` and `--text-quant` are the FLUX flags.

use crate::Args;
use anyhow::{Context, Result};
use oxinfer::sd3::{LoadOptions, Options, Pipeline};
use std::io::Read;
use std::path::PathBuf;
use std::time::Instant;

/// SD 3-specific flags.
#[derive(Default, Debug)]
pub struct Sd3Args {
    pub no_t5: bool,
    pub max_sequence_length: Option<usize>,
}

impl Sd3Args {
    /// Takes `flag` (with its value from `val`) if it is an SD 3 flag.
    pub fn parse(&mut self, flag: &str, val: &mut dyn FnMut() -> Result<String>) -> Result<bool> {
        match flag {
            "--no-t5" => self.no_t5 = true,
            "--max-sequence-length" => self.max_sequence_length = Some(val()?.parse()?),
            _ => return Ok(false),
        }
        Ok(true)
    }
}

pub const USAGE: &str = "\
Stable Diffusion 3 / 3.5 (a diffusers StableDiffusion3Pipeline folder or repo, or an SD 3 GGUF\n\
transformer such as city96/stable-diffusion-3.5-medium-gguf:Q4_0 with --base):\n\
\x20 --init-image FILE     image to image: start from this image (else: text to image)\n\
\x20 --strength F         image to image: how much of the schedule is run, 0..1 (default 0.6)\n\
\x20 -o, --output FILE     PNG to write (default image.png)\n\
\x20 --steps N            denoising steps (default 28; 50 image to image; 4 for turbo models)\n\
\x20 --guidance F         classifier-free guidance (default 7; 0 for turbo models)\n\
\x20 --negative TEXT      negative prompt (with guidance > 1; default empty)\n\
\x20 --size WxH           output size, rounded to multiples of 16 (default 1024x1024; image to\n\
\x20                      image: the input's size)\n\
\x20 --no-t5              run without the T5 encoder (zeros in its place, as diffusers allows)\n\
\x20 --max-sequence-length N  T5 tokens (default 256, at most 512)\n\
\x20 --quant Q            transformer precision when stored as float: bf16 (default) | q8 | q4\n\
\x20 --text-quant Q       T5 encoder precision (default: --quant)\n\
\x20 --transformer SPEC   use this transformer (GGUF file or org/name:QUANT) with --model's other parts\n\
\x20 --text-encoder SPEC a T5 encoder GGUF (e.g. city96/t5-v1_1-xxl-encoder-gguf:Q8_0) instead of the base's\n\
\x20 --base SPEC          with a GGUF --model: the diffusers folder the text encoders / VAE come from\n\
\x20 --seed N             noise seed (default 42)\n";

pub fn run(args: &Args) -> Result<()> {
    let f = &args.flux;
    let init = args.init_images.first().or(args.images.first());
    let opts = LoadOptions {
        quant: args.quant,
        text_quant: f.text_quant,
        // main's own `--transformer` (Qwen-Image, Step1X-Edit, OmniGen2) takes the flag first.
        transformer: args
            .transformer
            .as_ref()
            .map(|p| p.to_string_lossy().into_owned())
            .or_else(|| f.transformer.clone()),
        text_encoder: f.text_encoder.clone(),
        base: f.base.clone(),
        t5: !args.sd3.no_t5,
        device: args.device,
        cache_dir: args.cache_dir.clone(),
        vae_encoder: init.is_some(),
    };
    let t0 = Instant::now();
    let pipe = Pipeline::load(&args.model.to_string_lossy(), &opts)?;
    let c = &pipe.transformer.cfg;
    eprintln!(
        "[Stable Diffusion 3 ({}): {} joint blocks of {} x {}{}{}, {} | transformer {:.1} GB, T5 {} | loaded in {:.1}s]",
        pipe.class,
        c.num_layers,
        c.heads,
        c.head_dim,
        if c.qk_norm { ", QK norm" } else { "" },
        if c.dual_attention_layers.is_empty() {
            String::new()
        } else {
            format!(", {} with dual attention", c.dual_attention_layers.len())
        },
        if pipe.transformer.layout == oxinfer::sd3::transformer::Layout::Sai {
            "GGUF/original layout"
        } else {
            args.quant.as_str()
        },
        pipe.transformer.weight_bytes as f64 / 1e9,
        match &pipe.t5 {
            Some(t) => format!("{:.1} GB", t.weight_bytes as f64 / 1e9),
            None => "off".into(),
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
    let mut o = Options {
        steps: args.steps,
        guidance: args.guidance,
        negative_prompt: (!args.negative.is_empty()).then(|| args.negative.clone()),
        seed: args.seed,
        ..Default::default()
    };
    if let Some(n) = args.sd3.max_sequence_length {
        o.max_sequence_length = n;
    }
    if let Some(s) = args.strength {
        o.strength = s;
    }
    if let Some((w, h)) = args.size {
        (o.width, o.height) = (Some(w), Some(h));
    }
    let image: Option<oxinfer::video::RgbImage> = match init {
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
    let (steps, guidance) = pipe.defaults(&o, image.is_some());
    eprintln!(
        "[{}: {steps} steps, guidance {guidance}, seed {}]",
        match &image {
            Some(_) => format!("image to image, strength {}", o.strength),
            None => "text to image".into(),
        },
        o.seed
    );
    let tp = Instant::now();
    let mut progress = |i: usize, n: usize| {
        eprint!("\r[step {i}/{n}, {:.1}s]", tp.elapsed().as_secs_f64());
        true
    };
    let (img, tm) = match &image {
        Some(im) => pipe.img2img(im, &prompt, &o, &mut progress)?,
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
