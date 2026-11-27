//! LTX-2.5 for the server and the C API: loading from a name with `KEY=VALUE` options, and the
//! generation parameters from a JSON object (the same fields everywhere).

use super::dit::Blocks;
use super::{FetchOptions, Guidance, GuidanceMode, ImageCond, Noise, Params, Paths, Pipeline, Variant};
use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};

/// A part named in an option: a local path or `org/name/path/in/repo.safetensors`.
fn part(v: &str, cache_dir: Option<&Path>) -> Result<PathBuf> {
    let p = Path::new(v);
    if p.exists() {
        return Ok(p.to_path_buf());
    }
    crate::engine::resolve_file(v, cache_dir)
}

/// Whether `src` (a repo id, folder or file) is LTX-2.5.
pub fn is_ltx_source(src: &str) -> bool {
    super::is_repo(src) || super::is_ltx(Path::new(src))
}

/// Loads LTX-2.5 from `src` (`Lightricks/LTX-2.5` or its mirror: the parts a run reads are
/// fetched; a release folder; a transformer file) with options: `transformer` (`int8`, `bf16`,
/// `nvfp4`, `dev`, `dev-bf16` or a file), `text_encoder`, `vae`, `audio_vae`, `upsampler`,
/// `temporal_upsampler`, `duration_head` (files), `text_bf16` (fetch the bf16 text encoder),
/// `temporal` (fetch the temporal upsampler), `distilled_lora` (the release's, on the dev
/// transformer), `lora` (`FILE[:SCALE]`, repeatable), `device` (`cpu`: the transformer on the
/// CPU), `exact_attention` (the device's video attention on float32 operands).
pub fn load_with_options(src: &str, opts: &[(&str, &str)], cache_dir: Option<&Path>) -> Result<Pipeline> {
    let flag = |v: &str| matches!(v, "1" | "true" | "yes" | "on");
    let mut fo = FetchOptions::default();
    let mut transformer_file = None;
    for (k, v) in opts {
        match *k {
            "transformer" => match Variant::parse(v) {
                Some(x) => fo.variant = x,
                None => transformer_file = Some(part(v, cache_dir)?),
            },
            "text_bf16" => fo.text_bf16 = flag(v),
            "temporal" => fo.temporal_upsampler = flag(v),
            "distilled_lora" => fo.distilled_lora = flag(v),
            _ => {}
        }
    }
    let mut paths = if super::is_repo(src) && !Path::new(src).exists() {
        let dir = super::fetch(src, &fo, cache_dir)?;
        Paths::resolve_with(&dir, Some(fo.variant))?
    } else {
        let variant = opts.iter().find(|(k, _)| *k == "transformer").and_then(|(_, v)| Variant::parse(v));
        Paths::resolve_with(Path::new(src), variant)?
    };
    if let Some(t) = transformer_file {
        paths.transformer = t;
    }
    let (mut device, mut exact) = (true, false);
    for (k, v) in opts {
        match *k {
            "transformer" | "text_bf16" | "distilled_lora" | "temporal" => {}
            "text_encoder" => paths.text_encoder = Some(part(v, cache_dir)?),
            "vae" => paths.video_vae = Some(part(v, cache_dir)?),
            "audio_vae" => paths.audio_vae = Some(part(v, cache_dir)?),
            "upsampler" => paths.spatial_upsampler = Some(part(v, cache_dir)?),
            "temporal_upsampler" => paths.temporal_upsampler = Some(part(v, cache_dir)?),
            "duration_head" => paths.duration_head = Some(part(v, cache_dir)?),
            "lora" => {
                let l = crate::sd::lora::LoraSpec::parse(v)?;
                paths.loras.push((part(&l.path.to_string_lossy(), cache_dir)?, l.scale));
            }
            "device" => device = !v.eq_ignore_ascii_case("cpu"),
            "exact_attention" => exact = flag(v),
            k => bail!(
                "unknown LTX-2.5 option {k:?} (transformer, text_encoder, vae, audio_vae, upsampler, temporal_upsampler, \
                 duration_head, text_bf16, temporal, distilled_lora, lora, device, exact_attention)"
            ),
        }
    }
    if fo.distilled_lora {
        let root = paths
            .transformer
            .parent()
            .and_then(|p| p.parent())
            .map(Path::to_path_buf)
            .unwrap_or_default();
        paths.distilled_lora = Some(root.join(super::files::DISTILLED_LORA));
    }
    let mut pipe = Pipeline::load(paths)?;
    pipe.dit.device = device;
    pipe.dit.exact_attention = exact;
    Ok(pipe)
}

/// `WxH`.
pub fn parse_size(s: &str) -> Result<(usize, usize)> {
    s.split_once(['x', 'X'])
        .and_then(|(w, h)| Some((w.trim().parse().ok()?, h.trim().parse().ok()?)))
        .with_context(|| format!("size {s:?}: expected WIDTHxHEIGHT"))
}

fn floats(v: &serde_json::Value) -> Option<Vec<f32>> {
    match v {
        serde_json::Value::Array(a) => a.iter().map(|x| x.as_f64().map(|f| f as f32)).collect(),
        serde_json::Value::String(s) => s.split(',').map(|x| x.trim().parse().ok()).collect(),
        _ => None,
    }
}

/// The guidance keys ([`guidance_option`]).
pub const GUIDANCE_KEYS: [&str; 14] = [
    "cfg",
    "audio_cfg",
    "stg",
    "audio_stg",
    "stg_blocks",
    "audio_stg_blocks",
    "rescale",
    "audio_rescale",
    "modality",
    "audio_modality",
    "skip_step",
    "audio_skip_step",
    "steps",
    "negative_prompt",
];

/// `all`, `none` or a comma list of block indices.
pub fn parse_blocks(v: &str) -> Result<Blocks> {
    Ok(match v.trim() {
        "all" => Blocks::All,
        "" | "none" => Blocks::None,
        l => Blocks::List(
            l.split(',')
                .map(|x| x.trim().parse().with_context(|| format!("block index {x:?}")))
                .collect::<Result<_>>()?,
        ),
    })
}

/// Sets one guidance setting of `g` (the release's names: `cfg` / `audio_cfg` (1: off),
/// `stg` / `audio_stg` (0: off), `stg_blocks` / `audio_stg_blocks` (`all`, `none` or a list),
/// `rescale` / `audio_rescale`, `modality` / `audio_modality` (the a2v / v2a scale, 1: off),
/// `skip_step` / `audio_skip_step`, `steps`, `negative_prompt`); false when `key` is not one.
pub fn guidance_option(g: &mut Guidance, key: &str, v: &str) -> Result<bool> {
    let f = || -> Result<f32> { v.trim().parse().with_context(|| format!("{key}: not a number: {v}")) };
    let n = || -> Result<usize> { v.trim().parse().with_context(|| format!("{key}: not a count: {v}")) };
    match key {
        "cfg" | "guidance_scale" | "video_cfg" => g.video.cfg = f()?,
        "audio_cfg" | "audio_guidance_scale" => g.audio.cfg = f()?,
        "stg" | "stg_scale" | "video_stg" => g.video.stg = f()?,
        "audio_stg" | "audio_stg_scale" => g.audio.stg = f()?,
        "stg_blocks" | "video_stg_blocks" => g.video.stg_blocks = parse_blocks(v)?,
        "audio_stg_blocks" => g.audio.stg_blocks = parse_blocks(v)?,
        "rescale" | "video_rescale" => g.video.rescale = f()?,
        "audio_rescale" => g.audio.rescale = f()?,
        "modality" | "modality_scale" | "a2v_scale" => g.video.modality = f()?,
        "audio_modality" | "audio_modality_scale" | "v2a_scale" => g.audio.modality = f()?,
        "skip_step" | "video_skip_step" => g.video.skip_step = n()?,
        "audio_skip_step" => g.audio.skip_step = n()?,
        "steps" | "num_inference_steps" => g.steps = n()?.max(1),
        "negative_prompt" | "negative" => g.negative_prompt = v.to_string(),
        _ => return Ok(false),
    }
    Ok(true)
}

/// The guidance mode of `mode` (`auto` / none, `on`, `off`) and the release's guidance with
/// `opts` ([`guidance_option`]) on top; a scale or block setting (not `steps` or
/// `negative_prompt`) turns `auto` into `on`.
pub fn guidance_from(mode: Option<&str>, opts: &[(&str, String)]) -> Result<(GuidanceMode, Guidance)> {
    let mut m = match mode {
        None => GuidanceMode::Auto,
        Some(x) => GuidanceMode::parse(x).with_context(|| format!("guidance {x:?}: auto, on or off"))?,
    };
    let mut g = Guidance::default();
    for (k, v) in opts {
        if !guidance_option(&mut g, k, v)? {
            bail!("unknown guidance setting {k:?}");
        }
        if m == GuidanceMode::Auto && !matches!(*k, "steps" | "num_inference_steps" | "negative_prompt" | "negative") {
            m = GuidanceMode::On;
        }
    }
    Ok((m, g))
}

/// [`Params`] from a JSON object: `prompt` (required), `size` (`"WxH"`) or `width` / `height`,
/// `frames` or `seconds`, `fps`, `seed`, `one_stage`, `temporal_upscale`, `audio` (false: no
/// soundtrack), `eta`, `sigmas` / `stage2_sigmas` (arrays or comma lists), `noise`
/// (`oxinfer` / `torch`), `strength` (of the first frame), `guidance` (`auto`, `on`, `off`) and
/// the guidance settings of [`guidance_option`] (`cfg`, `stg`, `steps`, `negative_prompt`, ...;
/// any of them but `steps` and `negative_prompt` turns guidance on), `refine_audio`, `image_crf`
/// (the H.264 CRF conditioning images are re-compressed at; 0: as they are). Images are the
/// caller's to load ([`images_from_json`]).
pub fn params_from_json(o: &serde_json::Value) -> Result<Params> {
    let prompt = o["prompt"].as_str().context("no prompt")?;
    let mut p = Params::new(prompt);
    if let Some(s) = o["size"].as_str().filter(|s| !s.is_empty() && *s != "auto") {
        (p.width, p.height) = parse_size(s)?;
    }
    if let (Some(w), Some(h)) = (o["width"].as_u64(), o["height"].as_u64()) {
        (p.width, p.height) = (w as usize, h as usize);
    }
    let num = |k: &str| o[k].as_f64().or_else(|| o[k].as_str().and_then(|s| s.trim().parse().ok()));
    if let Some(f) = num("fps") {
        p.fps = f as f32;
    }
    if let Some(f) = num("frames") {
        p.frames = Some(f as usize);
    } else if let Some(s) = num("seconds") {
        p.frames = Some((s * p.fps as f64).round().max(1.0) as usize);
    }
    if let Some(s) = o["seed"].as_u64() {
        p.seed = s;
    }
    let b = |k: &str| o[k].as_bool().or_else(|| o[k].as_str().map(|s| matches!(s, "1" | "true" | "yes")));
    if let Some(x) = b("one_stage") {
        p.two_stage = !x;
    }
    if let Some(x) = b("temporal_upscale") {
        p.temporal_upscale = x;
    }
    if let Some(x) = b("audio") {
        p.audio = x;
    }
    if let Some(e) = num("eta") {
        p.eta = Some(e as f32);
    }
    if let Some(s) = floats(&o["sigmas"]) {
        p.stage1_sigmas = Some(s);
    }
    let mut gopts: Vec<(&str, String)> = Vec::new();
    for k in GUIDANCE_KEYS {
        match &o[k] {
            serde_json::Value::Null => {}
            serde_json::Value::String(v) => gopts.push((k, v.clone())),
            serde_json::Value::Array(a) => gopts.push((
                k,
                a.iter()
                    .map(|x| x.to_string().trim_matches('"').to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            )),
            v => gopts.push((k, v.to_string())),
        }
    }
    let gmode = match &o["guidance"] {
        serde_json::Value::Bool(b) => Some(if *b { "on".to_string() } else { "off".to_string() }),
        serde_json::Value::String(s) => Some(s.clone()),
        _ => None,
    };
    (p.guidance_mode, p.guidance) = guidance_from(gmode.as_deref(), &gopts)?;
    if let Some(x) = b("refine_audio") {
        p.refine_audio = x;
    }
    if let Some(c) = num("image_crf") {
        p.image_crf = Some(c.max(0.0) as u32);
    }
    if let Some(s) = floats(&o["stage2_sigmas"]) {
        p.stage2_sigmas = s;
    }
    p.noise = match o["noise"].as_str() {
        None => Noise::Oxinfer,
        Some(n) => Noise::parse(n)?,
    };
    for s in p.stage1_sigmas.iter().chain([&p.stage2_sigmas]) {
        if s.len() < 2 || s.last() != Some(&0.0) {
            bail!("a sigma schedule needs at least two values and ends at 0");
        }
    }
    Ok(p)
}

/// The image conditionings of a JSON object, as `(file, frame, strength)`: `first_frame` /
/// `input_reference` (frame 0, strength `strength` or 1), `last_frame` (the clip's last frame),
/// `images` (an array of `{"path", "frame", "strength"}` or of `FILE@FRAME[:STRENGTH]`).
pub fn images_from_json(o: &serde_json::Value) -> Result<Vec<(PathBuf, Option<usize>, f32)>> {
    let mut v = Vec::new();
    let strength = o["strength"].as_f64().unwrap_or(1.0) as f32;
    for k in ["first_frame", "input_reference"] {
        if let Some(f) = o[k].as_str() {
            v.push((PathBuf::from(f), Some(0), strength));
            break;
        }
    }
    if let Some(f) = o["last_frame"].as_str() {
        v.push((PathBuf::from(f), None, 1.0));
    }
    if let Some(a) = o["images"].as_array() {
        for i in a {
            match i {
                serde_json::Value::String(s) => {
                    let (f, frame, st) = parse_image(s)?;
                    v.push((f, Some(frame), st));
                }
                x => v.push((
                    PathBuf::from(x["path"].as_str().context("images: each needs a path")?),
                    Some(x["frame"].as_u64().unwrap_or(0) as usize),
                    x["strength"].as_f64().unwrap_or(1.0) as f32,
                )),
            }
        }
    }
    Ok(v)
}

/// `FILE[@FRAME[:STRENGTH]]`.
pub fn parse_image(s: &str) -> Result<(PathBuf, usize, f32)> {
    let (file, rest) = match s.rsplit_once('@') {
        Some((f, r)) if !r.contains('/') => (f, Some(r)),
        _ => (s, None),
    };
    let (frame, strength) = match rest {
        Some(r) => match r.split_once(':') {
            Some((f, st)) => (f.parse().context("image frame")?, st.parse().context("image strength")?),
            None => (r.parse().context("image frame")?, 1.0),
        },
        None => (0, 1.0),
    };
    Ok((PathBuf::from(file), frame, strength))
}

/// Adds the images (loaded by `load`) to `p`; a frame of None is the clip's last (resolved with
/// the pipeline's frame count: `p.frames`, else 121).
pub fn add_images(p: &mut Params, imgs: Vec<(PathBuf, Option<usize>, f32)>, load: &dyn Fn(&Path) -> Result<super::Image>) -> Result<()> {
    for (f, frame, strength) in imgs {
        let frame = match frame {
            Some(x) => x,
            None => super::snap_frames(p.frames.unwrap_or(super::DEFAULT_FRAMES)) - 1,
        };
        p.images.push(ImageCond {
            image: load(&f)?,
            frame,
            strength,
        });
    }
    Ok(())
}
