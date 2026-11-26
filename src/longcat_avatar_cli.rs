//! `oxinfer --model meituan-longcat/LongCat-Video-Avatar-1.6` (or `--preset longcat-avatar-int8`):
//! audio-driven avatar video (`oxinfer::longcat_avatar`; docs/longcat-avatar.md).
//!
//! ```text
//! oxinfer --preset longcat-avatar-int8 --init-image person.png --audio speech.wav \
//!     -p "A woman is speaking to the camera in a bright studio." -o talk.mp4
//! ```

use crate::Args;
use anyhow::{Context, Result, bail};
use oxinfer::clock::Instant;
use oxinfer::longcat_avatar::pipeline::{AvatarParams, AvatarPipeline, COND_FRAMES, FPS, SPEECH_RATE, Stage};
use oxinfer::longcat_avatar::{AVATAR_REPO, PIPELINE};
use oxinfer::minimax_h3::video::{Soundtrack, Video};
use std::path::{Path, PathBuf};

/// The avatar flags.
#[derive(Default, Clone)]
pub struct AvatarArgs {
    /// `--int8`: the int8 transformer (`base_model_int8/`).
    pub int8: bool,
    /// `--segments N`: clips (each after the first adds `frames - 13` frames).
    pub segments: Option<usize>,
    /// `--hd`: the 720p buckets.
    pub hd: bool,
    /// `--mix-steps N`, `--mix-alpha F`.
    pub mix_steps: Option<usize>,
    pub mix_alpha: Option<f32>,
    /// `--no-distill`: the base model without the LoRAs (`--steps`, `--guidance`, `--audio-guidance`).
    pub no_distill: bool,
    pub audio_guidance: Option<f32>,
    /// `--ref-index N`, `--mask-frames N`.
    pub ref_index: Option<usize>,
    pub mask_frames: Option<usize>,
    /// `--bbox Y0,X0,Y1,X1` (twice: speaker one, speaker two).
    pub bboxes: Vec<[usize; 4]>,
    /// `--audio-mode para|add`.
    pub audio_mode: Option<String>,
    /// `--avatar-base DIR|REPO`: where `text_encoder/`, `tokenizer/` and `vae/` are.
    pub base: Option<String>,
}

impl AvatarArgs {
    /// Takes `flag` (with its value from `val`) if it is an avatar flag.
    pub fn parse(&mut self, flag: &str, val: &mut dyn FnMut() -> Result<String>) -> Result<bool> {
        match flag {
            "--int8" => self.int8 = true,
            "--segments" => self.segments = Some(val()?.parse()?),
            "--hd" => self.hd = true,
            "--mix-steps" => self.mix_steps = Some(val()?.parse()?),
            "--mix-alpha" => self.mix_alpha = Some(val()?.parse()?),
            "--no-distill" => self.no_distill = true,
            "--audio-guidance" => self.audio_guidance = Some(val()?.parse()?),
            "--ref-index" => self.ref_index = Some(val()?.parse()?),
            "--mask-frames" => self.mask_frames = Some(val()?.parse()?),
            "--bbox" => {
                let v = val()?;
                let n: Vec<usize> = v
                    .split(',')
                    .map(|x| x.trim().parse::<usize>())
                    .collect::<Result<_, _>>()
                    .with_context(|| format!("--bbox expects Y0,X0,Y1,X1, got {v:?}"))?;
                let [a, b, c, d] = n[..] else {
                    bail!("--bbox expects Y0,X0,Y1,X1, got {v:?}")
                };
                self.bboxes.push([a, b, c, d]);
            }
            "--audio-mode" => self.audio_mode = Some(val()?),
            "--avatar-base" => self.base = Some(val()?),
            _ => return Ok(false),
        }
        Ok(true)
    }
}

fn preset_of(args: &Args) -> Option<&'static oxinfer::advisor::Preset> {
    let is = |s: &str| oxinfer::advisor::preset(s).filter(|p| p.pipeline == PIPELINE);
    args.moss
        .preset
        .as_deref()
        .and_then(is)
        .or_else(|| is(&args.model.to_string_lossy()))
}

/// Whether the command line runs the avatar pipeline: one of its presets, the release's
/// repository id or a folder holding it.
pub fn wants(args: &Args) -> bool {
    preset_of(args).is_some() || args.model.to_string_lossy() == AVATAR_REPO || oxinfer::longcat_avatar::is_avatar(&args.model)
}

fn load_image(path: &Path) -> Result<oxinfer::sd::Image> {
    #[cfg(feature = "media")]
    {
        oxinfer::minimax_h3::encode::load_image(path)
    }
    #[cfg(not(feature = "media"))]
    bail!("built without the `media` feature: {} cannot be decoded", path.display())
}

/// `path` as mono samples at `rate`.
fn load_audio(path: &Path, rate: usize) -> Result<Vec<f32>> {
    #[cfg(feature = "media")]
    {
        oxinfer::media::Media::new().load_audio(path, rate)
    }
    #[cfg(not(feature = "media"))]
    {
        let _ = rate;
        bail!("built without the `media` feature: {} cannot be decoded", path.display())
    }
}

/// The speech tracks of the request at `rate`: one, or two arranged as `mode` says (`para`: both
/// from the start, cut to the shorter; `add`: one after the other).
fn speech_tracks(paths: &[PathBuf], mode: &str, rate: usize) -> Result<Vec<Vec<f32>>> {
    let mut t: Vec<Vec<f32>> = paths.iter().map(|p| load_audio(p, rate)).collect::<Result<_>>()?;
    if t.len() == 2 {
        match mode {
            "para" => {
                let n = t[0].len().min(t[1].len());
                t.iter_mut().for_each(|x| x.truncate(n));
            }
            "add" => {
                let (a, b) = (t[0].len(), t[1].len());
                t[0].resize(a + b, 0.0);
                let mut second = vec![0f32; a];
                second.extend_from_slice(&t[1]);
                t[1] = second;
            }
            m => bail!("--audio-mode {m}: para or add"),
        }
    }
    Ok(t)
}

pub fn run(args: &Args) -> Result<()> {
    let t0 = Instant::now();
    let a = &args.avatar;
    let preset = preset_of(args);
    let setting = |k: &str| preset.and_then(|p| p.defaults.iter().find(|(n, _)| *n == k).map(|(_, v)| *v));
    let int8 = a.int8 || preset.is_some_and(|p| p.load.iter().any(|(k, v)| *k == "transformer" && v.contains("int8")));
    let spec = match preset {
        Some(p) if !args.model.is_dir() => p.base.to_string(),
        _ => args.model.to_string_lossy().to_string(),
    };
    if args.audios.is_empty() || args.audios.len() > 2 {
        bail!("--audio FILE is required: the speech that drives the character (twice for two speakers)");
    }
    let mut p = AvatarParams {
        prompt: args.prompt.clone().context("--prompt is required")?,
        hd: a.hd,
        size: args.size,
        seed: args.seed,
        ..Default::default()
    };
    if !args.negative.is_empty() {
        p.negative_prompt = args.negative.clone();
    }
    let num = |k: &str| setting(k).and_then(|v| v.parse::<f32>().ok());
    p.distill = !a.no_distill;
    p.steps = args
        .steps
        .or(num("steps").map(|v| v as usize))
        .unwrap_or(if p.distill { 8 } else { 50 });
    p.mix_steps = a.mix_steps.or(num("mix_steps").map(|v| v as usize)).unwrap_or(p.mix_steps);
    p.mix_alpha = a.mix_alpha.or(num("mix_alpha")).unwrap_or(p.mix_alpha);
    let guidance = args.guidance.unwrap_or(if p.distill { 1.0 } else { 4.0 });
    p.text_guidance = guidance;
    p.audio_guidance = a.audio_guidance.unwrap_or(guidance);
    p.frames = args.frames.unwrap_or(p.frames);
    p.segments = a.segments.unwrap_or(1).max(1);
    p.ref_img_index = a.ref_index.unwrap_or(p.ref_img_index);
    p.mask_frame_range = a.mask_frames.unwrap_or(p.mask_frame_range);
    if let Some(i) = args.init_images.first() {
        p.image = Some(load_image(i)?);
    }
    p.speech = speech_tracks(&args.audios, a.audio_mode.as_deref().unwrap_or("para"), SPEECH_RATE)?;
    p.regions = match a.bboxes[..] {
        [] => None,
        [x, y] => Some([x, y]),
        _ => bail!("--bbox is given twice (speaker one, speaker two) or not at all"),
    };

    let src = oxinfer::longcat_avatar::resolve(&spec, a.base.as_deref(), int8, args.quant, args.cache_dir.as_deref())?;
    let pipe = AvatarPipeline::load(&src)?;
    let (w, h) = pipe.canvas(&p)?;
    let shape = pipe.latent_shape(w, h, p.frames);
    let total = p.frames + (p.segments - 1) * (p.frames - COND_FRAMES.min(p.frames));
    eprintln!(
        "[LongCat-Video-Avatar: {} blocks ({}), {w}x{h}, {total} frames at {FPS} fps in {} clip(s) of {} ({} latent frames, {} tokens), {} steps{}; loaded in {:.1}s]",
        pipe.dit.cfg.depth,
        if pipe.dit.int8 { "int8" } else { "bf16" },
        p.segments,
        p.frames,
        shape.0,
        shape.0 * (shape.1 / 2) * (shape.2 / 2),
        p.steps,
        if p.distill {
            format!(", DMD x{} then OPSD (mix alpha {})", p.mix_steps, p.mix_alpha)
        } else {
            format!(", guidance {} (audio {})", p.text_guidance, p.audio_guidance)
        },
        pipe.load_secs
    );
    let mut last = Instant::now();
    let clip = pipe.generate(&p, &mut |s| match s {
        Stage::Text => eprintln!("[text encoder]"),
        Stage::Audio { done, total } => {
            eprintln!("[audio features: window {done}/{total}, {:.1}s]", last.elapsed().as_secs_f64());
            last = Instant::now();
        }
        Stage::Encode => {}
        Stage::Step {
            segment,
            segments,
            step,
            steps,
            secs,
        } => {
            eprintln!("[clip {segment}/{segments} step {step}/{steps}: {secs:.1}s]")
        }
        Stage::Decode { segment, done, total } => {
            if done == total {
                eprintln!("[clip {segment}: decoded {total} latent frames]");
            }
        }
    })?;

    // The soundtrack: the driving audio (both speakers mixed), cut to the video's length.
    const OUT_RATE: usize = 48_000;
    let tracks = speech_tracks(&args.audios, a.audio_mode.as_deref().unwrap_or("para"), OUT_RATE)?;
    let n = clip.frames.len() * OUT_RATE / FPS;
    let mut mix = vec![0f32; n];
    for t in &tracks {
        for (o, &s) in mix.iter_mut().zip(t) {
            *o += s;
        }
    }
    mix.iter_mut().for_each(|v| *v = v.clamp(-1.0, 1.0));
    let video = Video {
        width: clip.width,
        height: clip.height,
        frames: clip.frames,
        fps: FPS,
        audio: Some(Soundtrack {
            sample_rate: OUT_RATE as u32,
            channels: vec![mix.clone(), mix],
        }),
    };
    let out = args.output.clone().unwrap_or_else(|| PathBuf::from("avatar.mp4"));
    let secs = clip.secs;
    let t1 = Instant::now();
    let what = crate::minimax_h3_video_cli::write_video(&video, &out)?;
    eprintln!(
        "[wrote {what}: {} frames, {:.1}s of video in {:.1}s (text encoder {:.1}s, audio encoder {:.1}s, denoising {:.1}s, VAE decode {:.1}s, file {:.1}s)]",
        video.frames.len(),
        video.frames.len() as f64 / FPS as f64,
        t0.elapsed().as_secs_f64(),
        secs[0],
        secs[1],
        secs[2],
        secs[3],
        t1.elapsed().as_secs_f64()
    );
    Ok(())
}
