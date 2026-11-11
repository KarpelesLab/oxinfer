//! `oxinfer --model ByteDance-Seed/BAGEL-7B-MoT --init-image in.png -p "make it night" -o out.png`:
//! BAGEL through `oxinfer::bagel`. With `-o` and an image: editing; with `-o` alone: text to
//! image; without `-o`: an answer about the image (or a plain text reply).

use crate::Args;
use anyhow::{Context, Result};
use oxinfer::bagel::{Bagel, ImageOptions, Input, LoadOptions, Renorm, TextOptions};
use oxinfer::generate::Flow;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::time::Instant;

/// BAGEL-specific flags.
#[derive(Default, Debug)]
pub struct BagelArgs {
    pub cfg_text: Option<f32>,
    pub cfg_img: Option<f32>,
    pub cfg_interval: Option<[f32; 2]>,
    pub renorm: Option<Renorm>,
    pub renorm_min: Option<f32>,
    pub shift: Option<f32>,
    pub max_size: Option<usize>,
    pub min_size: Option<usize>,
    /// Flags shared with the other instruction-editing pipelines (parsed here so that they need
    /// no hook of their own in the argument loop).
    pub edit: EditArgs,
}

/// Opt-in options of the instruction-editing pipelines (OmniGen2, BAGEL, Step1X-Edit) that
/// change the result: step caches, the scheduler, posterior sampling.
#[derive(Default, Debug)]
pub struct EditArgs {
    /// `--teacache F`: TeaCache with this relative-L1 threshold (OmniGen2).
    pub teacache: Option<f32>,
    /// `--taylorseer`: TaylorSeer step caching (OmniGen2, BAGEL).
    pub taylorseer: bool,
    /// `--scheduler euler|dpmsolver++` (OmniGen2).
    pub scheduler: Option<String>,
    /// `--sample-posterior`: sample the VAE posterior of input images instead of its mean.
    pub sample_posterior: bool,
}

impl EditArgs {
    fn parse(&mut self, flag: &str, val: &mut dyn FnMut() -> Result<String>) -> Result<bool> {
        match flag {
            "--teacache" => self.teacache = Some(val()?.parse()?),
            "--taylorseer" => self.taylorseer = true,
            "--scheduler" => self.scheduler = Some(val()?),
            "--sample-posterior" => self.sample_posterior = true,
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// Fails when a flag the running pipeline does not have was given.
    pub fn only(&self, model: &str, teacache: bool, taylorseer: bool, scheduler: bool) -> Result<()> {
        let bad = |on: bool, ok: bool, flag: &str| -> Result<()> {
            if on && !ok {
                anyhow::bail!("{flag} is not an option of {model}");
            }
            Ok(())
        };
        bad(self.teacache.is_some(), teacache, "--teacache")?;
        bad(self.taylorseer, taylorseer, "--taylorseer")?;
        bad(self.scheduler.is_some(), scheduler, "--scheduler")
    }
}

impl BagelArgs {
    /// Takes `flag` (with its value from `val`) if it is a BAGEL flag.
    pub fn parse(&mut self, flag: &str, val: &mut dyn FnMut() -> Result<String>) -> Result<bool> {
        match flag {
            "--cfg-text" => self.cfg_text = Some(val()?.parse()?),
            "--cfg-img" => self.cfg_img = Some(val()?.parse()?),
            "--cfg-interval" => {
                let v = val()?;
                let (a, b) = v
                    .split_once(',')
                    .with_context(|| format!("--cfg-interval expects LO,HI, got {v}"))?;
                self.cfg_interval = Some([a.trim().parse()?, b.trim().parse()?]);
            }
            "--cfg-renorm" => {
                let v = val()?;
                self.renorm =
                    Some(Renorm::parse(&v).with_context(|| format!("--cfg-renorm expects global|channel|text_channel, got {v}"))?);
            }
            "--cfg-renorm-min" => self.renorm_min = Some(val()?.parse()?),
            "--timestep-shift" => self.shift = Some(val()?.parse()?),
            "--max-size" => self.max_size = Some(val()?.parse()?),
            "--min-size" => self.min_size = Some(val()?.parse()?),
            _ => return self.edit.parse(flag, val),
        }
        Ok(true)
    }
}

pub const USAGE: &str = "\
BAGEL (ByteDance-Seed/BAGEL-7B-MoT): editing, text to image and image understanding:\n\
\x20 --init-image FILE     with -o: edit this image following --prompt; without -o: ask about it\n\
\x20                      (repeat it for several images: the instruction follows them all)\n\
\x20 -o, --output FILE     PNG to write (editing / text to image; without it the answer is text)\n\
\x20 --steps N            timesteps (default 50)\n\
\x20 --cfg-text F         text guidance (default 4)\n\
\x20 --cfg-img F          image guidance (default 2 when editing, 1 otherwise)\n\
\x20 --cfg-interval LO,HI guidance timesteps (default 0,1 editing; 0.4,1 text to image)\n\
\x20 --cfg-renorm T       global | channel | text_channel (default text_channel editing, global otherwise)\n\
\x20 --cfg-renorm-min F   renorm lower bound (default 0; 1 disables)\n\
\x20 --timestep-shift F   (default 3)\n\
\x20 --size WxH           text to image size (default 1024x1024)\n\
\x20 --max-size N         input images: longer side at most N (default 1024; 512 is ~4x cheaper)\n\
\x20 --min-size N         input images: shorter side at least N where possible (default 512)\n\
\x20 --think              write a plan (images) or reasoning (answers) first\n\
\x20 --taylorseer         TaylorSeer step cache: after 5 steps the model runs every 3rd step (a\n\
\x20                      different image; off by default)\n\
\x20 --sample-posterior   sample the VAE posterior of input images (default: its mean)\n\
\x20 --quant Q            bf16 (default) | q8 | q4      --mmap   map the weights\n\
\x20 --seed N             noise seed (default 42)\n";

pub fn run(args: &Args) -> Result<()> {
    let b = &args.bagel;
    b.edit.only("BAGEL", false, true, false)?;
    // The images to edit or ask about: every --init-image, else every --image.
    let inits: Vec<&PathBuf> = if args.init_images.is_empty() {
        args.images.iter().collect()
    } else {
        args.init_images.iter().collect()
    };
    let init = inits.first();
    let image_out = args.output.is_some() || args.generate_image.is_some();
    let opts = LoadOptions {
        quant: args.quant,
        mmap: args.mmap,
        device: args.device,
        cache_dir: args.cache_dir.clone(),
        generation: image_out,
        understanding: init.is_some(),
        text: !image_out || args.think,
    };
    let t0 = Instant::now();
    let m = Bagel::load(&args.model.to_string_lossy(), &opts)?;
    let c = m.config();
    eprintln!(
        "[BAGEL: {} layers x 2 experts, hidden {}, {} | loaded in {:.1}s]",
        c.num_hidden_layers,
        c.hidden_size,
        args.quant.as_str(),
        t0.elapsed().as_secs_f64()
    );
    let prompt = match args.generate_image.clone().or_else(|| args.prompt.clone()) {
        Some(p) => p,
        None => {
            let mut s = String::new();
            std::io::stdin().read_to_string(&mut s)?;
            s.trim().to_string()
        }
    };
    let images = inits
        .iter()
        .map(|p| {
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
        })
        .collect::<Result<Vec<_>>>()?;
    let image = images.first();
    let mut inputs: Vec<Input> = images.iter().map(Input::Image).collect();
    inputs.push(Input::Text(&prompt));
    let defaults = ImageOptions::default();
    let (max_size, min_size) = (
        b.max_size.unwrap_or(defaults.max_image_size),
        b.min_size.unwrap_or(defaults.min_image_size.min(b.max_size.unwrap_or(usize::MAX))),
    );
    if !image_out {
        let o = TextOptions {
            max_tokens: args.max_tokens_set.unwrap_or(1000),
            temperature: if args.temp_given { args.temperature } else { 0.0 },
            top_k: args.top_k,
            top_p: args.top_p,
            seed: args.seed,
            think: args.think,
            max_image_size: max_size,
            min_image_size: min_size,
        };
        let t1 = Instant::now();
        let mut out = std::io::stdout();
        let text = m.understand_interleaved(&inputs, &o, &mut |d| {
            let _ = out.write_all(d.as_bytes());
            let _ = out.flush();
            Flow::Continue
        })?;
        println!();
        eprintln!("[{} chars in {:.2}s]", text.len(), t1.elapsed().as_secs_f64());
        return Ok(());
    }
    let mut o = ImageOptions {
        steps: args.steps.unwrap_or(50),
        timestep_shift: b.shift.unwrap_or(3.0),
        cfg_text_scale: b.cfg_text.or(args.guidance),
        cfg_img_scale: b.cfg_img.or(args.image_guidance),
        cfg_interval: b.cfg_interval,
        cfg_renorm_min: b.renorm_min.unwrap_or(0.0),
        cfg_renorm_type: b.renorm,
        seed: args.seed,
        think: args.think,
        max_image_size: max_size,
        min_image_size: min_size,
        taylorseer: b.edit.taylorseer,
        sample_posterior: b.edit.sample_posterior,
        ..Default::default()
    };
    if let Some((w, h)) = args.size {
        (o.width, o.height) = (Some(w), Some(h));
    }
    eprintln!(
        "[{}: {} steps, seed {}{}{}{}]",
        if image.is_some() { "edit" } else { "text to image" },
        o.steps,
        o.seed,
        if o.think { ", thinking" } else { "" },
        if o.taylorseer { ", TaylorSeer" } else { "" },
        if o.sample_posterior && image.is_some() {
            ", sampled VAE posterior"
        } else {
            ""
        }
    );
    let tp = Instant::now();
    let mut progress = |i: usize, n: usize| {
        eprint!("\r[step {i}/{n}, {:.1}s]", tp.elapsed().as_secs_f64());
        true
    };
    let (img, st) = m.generate_interleaved(&inputs, &o, &mut progress)?;
    eprintln!();
    for (i, s) in st.posterior_std.iter().enumerate() {
        eprintln!("[input image {}: posterior std / latents (L2) {s:.3e}]", i + 1);
    }
    if let Some(t) = &st.thinking {
        println!("{t}");
    }
    let out = args.output.clone().unwrap_or_else(|| PathBuf::from("image.png"));
    std::fs::write(&out, oxinfer::png::encode_rgb(img.width, img.height, &img.rgb)).with_context(|| format!("write {}", out.display()))?;
    eprintln!(
        "[{}x{} | context {} tokens {:.2}s | think {:.2}s | denoise {:.2}s ({} passes of {} latent tokens, {:.2}s each{}) | decode {:.2}s | wrote {}]",
        img.width,
        img.height,
        st.context_tokens,
        st.context,
        st.think,
        st.denoise,
        st.passes,
        st.latent_tokens,
        st.denoise / st.passes.max(1) as f64,
        if st.cached_passes > 0 {
            format!("; {} more from the step cache", st.cached_passes)
        } else {
            String::new()
        },
        st.decode,
        out.display()
    );
    if args.profile {
        oxinfer::prof::report(st.context + st.denoise, st.passes);
        if let Some(s) = oxinfer::gpu::active().and_then(|b| b.stats()) {
            eprintln!("  gpu: {s}");
        }
    }
    Ok(())
}
