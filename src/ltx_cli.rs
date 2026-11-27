//! `oxinfer --model Lightricks/LTX-2.5 -p "..." -o clip.mp4`: LTX-2.5 text (and image) to video
//! with its soundtrack (`oxinfer::ltx`; docs/ltx.md).
//!
//! ```text
//! oxinfer --model Lightricks/LTX-2.5 -p "A red fox trots across a snowy clearing" \
//!     --size 768x512 --frames 121 -o fox.mp4
//! oxinfer --model Lightricks/LTX-2.5 --init-image first.png -p "..." -o clip.mp4
//! ```
//!
//! The release is gated; its files come from the public mirror `comfyicu/LTX-2.5` (only the
//! parts a run reads). `--model` may also be a local folder in the release's layout or one of its
//! transformer files (the other parts are then taken from the folder above it).

use crate::Args;
use anyhow::{Context, Result, bail};
use oxinfer::clock::Instant;
use oxinfer::ltx::{self, FetchOptions, ImageCond, Noise, Params, Paths, Pipeline, Variant};
use std::path::{Path, PathBuf};

/// The LTX-2.5 flags.
#[derive(Default, Clone)]
pub struct LtxArgs {
    /// `--fps F` (default 24).
    pub fps: Option<f32>,
    /// `--one-stage`: the distilled schedule at the full size, no spatial upsampler.
    pub one_stage: bool,
    /// `--temporal-upscale`: twice the frames and rate after the last stage.
    pub temporal_upscale: bool,
    /// `--no-audio`: no soundtrack decoded.
    pub no_audio: bool,
    /// `--image FILE[@FRAME[:STRENGTH]]` (repeatable).
    pub images: Vec<String>,
    /// `--eta F`: the ancestral sampler's noise (1: the release's, 0: plain Euler).
    pub eta: Option<f32>,
    /// `--sigmas a,b,..` / `--stage2-sigmas a,b,..`: replace the distilled schedules.
    pub sigmas: Option<Vec<f32>>,
    pub stage2_sigmas: Option<Vec<f32>>,
    /// `--text-bf16`: the bf16 text encoder (when downloading).
    pub text_bf16: bool,
    /// `--upsampler FILE`, `--temporal-upsampler FILE`, `--duration-head FILE`.
    pub upsampler: Option<PathBuf>,
    pub temporal_upsampler: Option<PathBuf>,
    pub duration_head: Option<PathBuf>,
    /// `--distilled-lora`: the release's distilled LoRA on the dev transformer.
    pub distilled_lora: bool,
    /// `--exact-attention`: the GPU's video attention on float32 operands.
    pub exact_attention: bool,
}

fn floats(v: &str) -> Result<Vec<f32>> {
    v.split(',')
        .map(|x| x.trim().parse::<f32>().with_context(|| format!("not a number: {x}")))
        .collect()
}

impl LtxArgs {
    /// Takes `flag` (with its value from `val`) if it is an LTX-2.5 flag.
    pub fn parse(&mut self, flag: &str, val: &mut dyn FnMut() -> Result<String>) -> Result<bool> {
        match flag {
            "--fps" => self.fps = Some(val()?.parse()?),
            "--one-stage" => self.one_stage = true,
            "--temporal-upscale" => self.temporal_upscale = true,
            "--no-audio" => self.no_audio = true,
            "--image" => self.images.push(val()?),
            "--eta" => self.eta = Some(val()?.parse()?),
            "--sigmas" => self.sigmas = Some(floats(&val()?)?),
            "--stage2-sigmas" => self.stage2_sigmas = Some(floats(&val()?)?),
            "--text-bf16" => self.text_bf16 = true,
            "--upsampler" => self.upsampler = Some(PathBuf::from(val()?)),
            "--temporal-upsampler" => self.temporal_upsampler = Some(PathBuf::from(val()?)),
            "--duration-head" => self.duration_head = Some(PathBuf::from(val()?)),
            "--distilled-lora" => self.distilled_lora = true,
            "--exact-attention" => self.exact_attention = true,
            _ => return Ok(false),
        }
        Ok(true)
    }
}

/// Whether the command line runs LTX-2.5: the release's repository id (or its mirror's), or a
/// folder / transformer file of it.
pub fn wants(args: &Args) -> bool {
    ltx::is_repo(&args.model.to_string_lossy()) || ltx::is_ltx(&args.model)
}

fn load_image(path: &Path) -> Result<oxinfer::sd::Image> {
    #[cfg(feature = "media")]
    {
        oxinfer::minimax_h3::encode::load_image(path)
    }
    #[cfg(not(feature = "media"))]
    bail!("built without the `media` feature: {} cannot be decoded", path.display())
}

/// A part named on the command line: a local path or `org/name/path/in/repo.safetensors`.
fn part(p: &Path, args: &Args) -> Result<PathBuf> {
    if p.exists() {
        return Ok(p.to_path_buf());
    }
    oxinfer::engine::resolve_file(&p.to_string_lossy(), args.cache_dir.as_deref())
}

pub fn run(args: &Args) -> Result<()> {
    let t0 = Instant::now();
    let a = &args.ltx;
    let spec = args.model.to_string_lossy().to_string();
    // `--transformer int8|bf16|nvfp4|dev|dev-bf16` picks a release file; a path names one.
    let variant = args.transformer.as_ref().and_then(|t| Variant::parse(&t.to_string_lossy()));
    let mut paths = if ltx::is_repo(&spec) && !args.model.exists() {
        let o = FetchOptions {
            variant: variant.unwrap_or_default(),
            text_bf16: a.text_bf16,
            temporal_upsampler: a.temporal_upscale,
            distilled_lora: a.distilled_lora,
        };
        let dir = ltx::fetch(&spec, &o, args.cache_dir.as_deref())?;
        Paths::resolve_with(&dir, Some(o.variant))?
    } else {
        Paths::resolve_with(&args.model, variant)?
    };
    if args.download_only {
        eprintln!("[LTX-2.5 parts: {}]", paths.transformer.display());
        return Ok(());
    }
    if let Some(t) = args.transformer.as_ref().filter(|_| variant.is_none()) {
        paths.transformer = part(t, args)?;
    }
    if let Some(t) = &args.flux.text_encoder {
        paths.text_encoder = Some(part(Path::new(t), args)?);
    }
    if let Some(v) = &args.flux.vae {
        paths.video_vae = Some(part(Path::new(v), args)?);
    }
    if let Some(v) = &args.audio_vae {
        paths.audio_vae = Some(part(Path::new(v), args)?);
    }
    for (slot, flag) in [
        (&mut paths.spatial_upsampler, &a.upsampler),
        (&mut paths.temporal_upsampler, &a.temporal_upsampler),
        (&mut paths.duration_head, &a.duration_head),
    ] {
        if let Some(f) = flag {
            *slot = Some(part(f, args)?);
        }
    }
    if a.distilled_lora {
        let root = paths
            .transformer
            .parent()
            .and_then(|p| p.parent())
            .map(Path::to_path_buf)
            .unwrap_or_default();
        let l = root.join(ltx::files::DISTILLED_LORA);
        if !l.exists() {
            bail!("--distilled-lora: {} is missing", l.display());
        }
        paths.loras.push((l, 1.0));
    }
    for l in &args.loras {
        paths.loras.push((l.path.clone(), l.scale));
    }
    let prompt = args.prompt.clone().context("--prompt is required")?;
    let mut p = Params::new(&prompt);
    if let Some((w, h)) = args.size {
        p.width = w;
        p.height = h;
    }
    p.frames = args.frames;
    if let Some(f) = a.fps {
        p.fps = f;
    }
    p.seed = args.seed;
    p.two_stage = !a.one_stage;
    p.temporal_upscale = a.temporal_upscale;
    p.audio = !a.no_audio;
    if let Some(e) = a.eta {
        p.eta = e;
    }
    if let Some(s) = &a.sigmas {
        p.stage1_sigmas = s.clone();
    }
    if let Some(s) = &a.stage2_sigmas {
        p.stage2_sigmas = s.clone();
    }
    p.noise = match args.moss.noise.as_deref() {
        None | Some("oxinfer") => Noise::Oxinfer,
        Some("torch") => Noise::Torch,
        Some(n) => bail!("--noise {n}: oxinfer or torch"),
    };
    if let Some(f) = args.init_images.first() {
        p.images.push(ImageCond {
            image: load_image(f)?,
            frame: 0,
            strength: args.strength.unwrap_or(1.0),
        });
    }
    for s in &a.images {
        let (f, frame, strength) = ltx::api::parse_image(s)?;
        p.images.push(ImageCond {
            image: load_image(&f)?,
            frame,
            strength,
        });
    }
    if let Some(f) = &args.last_image {
        // The clip's last frame: of --frames (or the default 121).
        p.images.push(ImageCond {
            image: load_image(f)?,
            frame: ltx::snap_frames(p.frames.unwrap_or(ltx::DEFAULT_FRAMES)) - 1,
            strength: 1.0,
        });
    }
    for s in [&p.stage1_sigmas, &p.stage2_sigmas] {
        if s.len() < 2 || s.last() != Some(&0.0) {
            bail!("a sigma schedule needs at least two values and ends at 0");
        }
    }
    let t1 = Instant::now();
    let mut pipe = Pipeline::load(paths)?;
    pipe.dit.exact_attention = a.exact_attention;
    eprintln!(
        "[LTX-2.5: {} ({} blocks), loaded in {:.1}s]",
        pipe.paths.transformer.display(),
        pipe.dit.cfg.layers,
        t1.elapsed().as_secs_f64()
    );
    let tp = Instant::now();
    let mut progress = |stage: &str, i: usize, n: usize| eprint!("\r[{stage}: {i}/{n}, {:.1}s]   ", tp.elapsed().as_secs_f64());
    let video = pipe.generate(&p, &mut progress)?;
    eprintln!();
    let default = if cfg!(feature = "media") { "video.mp4" } else { "video.png" };
    let out = args.output.clone().unwrap_or_else(|| PathBuf::from(default));
    let what = crate::minimax_h3_video_cli::write_video(&video, &out)?;
    eprintln!(
        "[{}x{}, {} frames | {:.1}s | wrote {what}{}]",
        video.width,
        video.height,
        video.frames.len(),
        t0.elapsed().as_secs_f64(),
        if video.audio.is_some() { "" } else { "; no audio track" }
    );
    if args.profile {
        eprint!("[time by operation]\n{}", oxinfer::sd::ops::timing::report());
    }
    Ok(())
}
