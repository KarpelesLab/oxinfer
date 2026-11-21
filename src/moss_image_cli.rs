//! `oxinfer --model lodestones/Kromoss --prompt "..." -o out.png` (MOSS-Image text to image, through
//! `oxinfer::moss_image`), and `oxinfer --preset kromoss-famegrid --prompt "..." -o out.png`
//! (a curated stack of `oxinfer::advisor::PRESETS`).

use crate::Args;
use anyhow::{Context, Result};
use oxinfer::moss_image::{Options, Params, Pipeline};
use std::io::Read;
use std::path::PathBuf;
use std::time::Instant;

/// MOSS-Image flags (beyond the shared `--steps`, `--guidance`, `--negative`, `--size`, `--seed`,
/// `--lora`, `--sampler`, `--shift`, `--scheduler`, `--transformer`, `--text-encoder`, `--vae`).
#[derive(Default, Debug)]
pub struct MossArgs {
    /// `--preset ID`.
    pub preset: Option<String>,
    /// `--uncond-skip LAYERS` (comma-separated, `none`).
    pub uncond_skip: Option<String>,
    /// `--slg-layers LAYERS`, `--slg-scale F`, `--slg-start F`, `--slg-stop F`.
    pub slg_layers: Option<String>,
    pub slg_scale: Option<f32>,
    pub slg_start: Option<f32>,
    pub slg_stop: Option<f32>,
    /// `--trigger WORD` (`""`: none).
    pub trigger: Option<String>,
    /// `--noise oxinfer|torch`.
    pub noise: Option<String>,
    /// `--lora-merge`: merge low-rank LoRAs into the weights.
    pub lora_merge: bool,
    /// `--moss KEY=VALUE`: any load option or generation setting by name.
    pub set: Vec<(String, String)>,
}

impl MossArgs {
    /// Takes `flag` (with its value from `val`) if it is a MOSS-Image flag.
    pub fn parse(&mut self, flag: &str, val: &mut dyn FnMut() -> Result<String>) -> Result<bool> {
        match flag {
            "--preset" => self.preset = Some(val()?),
            "--uncond-skip" => self.uncond_skip = Some(val()?),
            "--slg-layers" => self.slg_layers = Some(val()?),
            "--slg-scale" => self.slg_scale = Some(val()?.parse()?),
            "--slg-start" => self.slg_start = Some(val()?.parse()?),
            "--slg-stop" => self.slg_stop = Some(val()?.parse()?),
            "--trigger" => self.trigger = Some(val()?),
            "--noise" => self.noise = Some(val()?),
            "--lora-merge" => self.lora_merge = true,
            "--moss" => {
                let v = val()?;
                let (k, x) = v.split_once('=').with_context(|| format!("--moss expects KEY=VALUE, got {v:?}"))?;
                self.set.push((k.trim().to_string(), x.trim().to_string()));
            }
            _ => return Ok(false),
        }
        Ok(true)
    }
}

/// Whether the command line runs MOSS-Image: a preset of that pipeline, a MOSS-Image folder or a
/// MOSS-Image GGUF transformer.
pub fn wants(args: &Args) -> bool {
    let preset = |s: &str| oxinfer::advisor::preset(s).is_some_and(oxinfer::moss_image::is_moss_preset);
    args.moss.preset.as_deref().is_some_and(preset)
        || preset(&args.model.to_string_lossy())
        || oxinfer::moss_image::is_moss_image(&args.model)
        || oxinfer::moss_image::is_moss_gguf(&args.model)
}

pub fn run(args: &Args) -> Result<()> {
    let mut o = Options {
        quant: args.quant,
        text_quant: args.flux.text_quant.unwrap_or(oxinfer::model::Quant::Bf16),
        transformer: args
            .transformer
            .as_ref()
            .map(|p| p.to_string_lossy().into_owned())
            .or(args.flux.transformer.clone()),
        text_encoder: args.flux.text_encoder.clone(),
        vae: args.flux.vae.clone(),
        loras: Vec::new(),
        lora_merge: args.moss.lora_merge,
        in_memory: args.preload,
        device: args.device,
        cache_dir: args.cache_dir.clone(),
        ..Default::default()
    };
    let mut model = args.model.to_string_lossy().into_owned();
    // A GGUF transformer given as the model: the rest from the preset's base or Kromoss.
    if oxinfer::moss_image::is_moss_gguf(&args.model) {
        o.transformer = Some(model.clone());
        model = "lodestones/Kromoss".into();
    }
    if let Some(p) = &args.moss.preset {
        o.apply_preset(p)?;
        if oxinfer::advisor::preset(&model).is_none() && !oxinfer::moss_image::is_moss_image(std::path::Path::new(&model)) {
            model = o.preset.map(|p| p.base.to_string()).unwrap_or(model);
        }
    }
    for l in &args.loras {
        o.loras.push(oxinfer::moss_image::lora::LoraSpec {
            path: l.path.clone(),
            scale: l.scale,
        });
    }
    let mut gen_set = Vec::new();
    for (k, v) in &args.moss.set {
        if Options::KEYS.contains(&k.as_str()) {
            o.set(k, v)?;
        } else {
            gen_set.push((k.clone(), v.clone()));
        }
    }
    let t0 = Instant::now();
    let mut pipe = Pipeline::load(&model, o)?;
    for l in pipe.lora_report() {
        eprintln!("{l}");
    }
    eprintln!(
        "[MOSS-Image {}: {} blocks, transformer {:.2} GB{}, text encoder {} | loaded in {:.1}s]",
        pipe.dir.display(),
        pipe.transformer().cfg.layers,
        pipe.transformer().bytes() as f64 / 1e9,
        pipe.opts.preset.map(|p| format!(", preset {}", p.id)).unwrap_or_default(),
        pipe.text_encoder_dir().display(),
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
    let mut p: Params = pipe.params(&prompt);
    if let Some(s) = args.steps {
        p.steps = s;
    }
    if let Some(g) = args.guidance {
        p.guidance = g;
    }
    if !args.negative.is_empty() {
        p.negative = args.negative.clone();
    }
    p.seed = args.seed;
    if let Some((w, h)) = args.size {
        (p.width, p.height) = (w, h);
    }
    if let Some(s) = &args.sampler {
        p.set("sampler", s)?;
    }
    if let Some(s) = &args.bagel.edit.scheduler {
        p.set("scheduler", s)?;
    }
    if let Some(s) = args.shift {
        p.shift = s;
    }
    let m = &args.moss;
    if let Some(v) = &m.uncond_skip {
        p.set("uncond_skip_layers", v)?;
    }
    if let Some(v) = &m.slg_layers {
        p.set("slg_layers", v)?;
    }
    if let Some(v) = m.slg_scale {
        p.slg_scale = v;
    }
    if let Some(v) = m.slg_start {
        p.slg_start = v;
    }
    if let Some(v) = m.slg_stop {
        p.slg_stop = v;
    }
    if let Some(t) = &m.trigger {
        p.trigger = t.clone();
    }
    if let Some(n) = &m.noise {
        p.set("noise", n)?;
    }
    for (k, v) in &gen_set {
        p.set(k, v)?;
    }
    eprintln!(
        "[{}x{}, {} steps ({} / {}, shift {}), guidance {}{}{}, seed {}]",
        p.width,
        p.height,
        p.steps,
        p.sampler.as_str(),
        p.schedule.as_str(),
        p.shift,
        p.guidance,
        if p.guidance != 1.0 && !p.uncond_skip_layers.is_empty() {
            format!(" (negative branch without blocks {:?})", p.uncond_skip_layers)
        } else {
            String::new()
        },
        if p.slg_scale > 0.0 && !p.slg_layers.is_empty() {
            format!(", SLG {} on {:?}", p.slg_scale, p.slg_layers)
        } else {
            String::new()
        },
        p.seed
    );
    if p.full_prompt() != p.prompt {
        eprintln!("[prompt: {}]", p.full_prompt());
    }
    let mut progress = |i: usize, n: usize| eprint!("\r[step {i}/{n}]");
    let (img, tm) = pipe.generate(&p, &mut progress)?;
    eprintln!();
    let out = args.output.clone().unwrap_or_else(|| PathBuf::from("image.png"));
    std::fs::write(&out, oxinfer::png::encode_rgb(img.width, img.height, &img.rgb)).with_context(|| format!("write {}", out.display()))?;
    eprintln!(
        "[load text encoder {:.1}s, encode {:.1}s | denoise {:.1}s ({} passes, {:.2}s each) | decode {:.1}s | {}x{} -> {}]",
        tm.load_text,
        tm.text,
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
