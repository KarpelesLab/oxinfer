//! Image, video and audio attachments: decoding through oxideav (the `media` feature),
//! encoding with the vision / audio towers into soft tokens, and mapping them onto the
//! placeholder tokens of the prompt. Gemma 4 takes images, videos (frames as images with
//! `mm:ss` timestamps) and audio; Qwen3.5 / 3.6 / 3.8 take images and videos (temporal patches
//! of two frames with `<t seconds>` timestamps, placed with M-RoPE).

#[cfg(feature = "media")]
use crate::clock::Instant;
use crate::model::Quant;
use crate::safetensors::ModelFiles;
use crate::video::VideoOptions;
#[cfg(feature = "media")]
use crate::{audio, media, mm_config, qwen_vision, video, vision};
use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};

/// Attachments for one prompt.
pub struct MmRequest<'a> {
    /// The model directory (for `config.json` / processor configs).
    pub model: &'a Path,
    pub images: &'a [PathBuf],
    pub audios: &'a [PathBuf],
    /// Video files, or directories of frames.
    pub videos: &'a [PathBuf],
    /// Video frame sampling.
    pub video: VideoOptions,
    /// Image token budget (0: the model's default). Gemma 4: soft tokens per image (70, 140,
    /// 280, 560 or 1120); Qwen: merged tokens per image (each covers 32x32 pixels).
    pub max_soft_tokens: usize,
    /// Precision for a Qwen vision tower that is not pre-quantized (see
    /// `qwen_vision::QwenVisionTower::load`).
    pub quant: Quant,
    /// Write the first image's / clip's inputs and soft tokens as f32 files with this prefix.
    pub dump_mm: Option<&'a Path>,
}

impl<'a> MmRequest<'a> {
    /// A request for `model` with the given images and audio clips and default settings.
    pub fn new(model: &'a Path, images: &'a [PathBuf], audios: &'a [PathBuf]) -> Self {
        Self {
            model,
            images,
            audios,
            videos: &[],
            video: VideoOptions::default(),
            max_soft_tokens: 0,
            quant: Quant::Bf16,
            dump_mm: None,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.images.is_empty() && self.audios.is_empty() && self.videos.is_empty()
    }
}

/// Soft tokens for one attachment: placeholder token id and `[n, hidden]` embeddings.
pub struct MmFeature {
    /// The placeholder token the embeddings replace.
    pub token_id: u32,
    /// `[n, hidden]` soft-token embeddings.
    pub emb: Vec<f32>,
    pub n: usize,
    /// Qwen (M-RoPE): the `(rows, cols)` token grid of each run of placeholders, in order (one
    /// per image, one per temporal patch of a video). Empty for Gemma.
    pub grids: Vec<(usize, usize)>,
}

/// Decodes and encodes all attachments. Returns the placeholder text to prepend to the user
/// message and the features in the same order.
#[cfg(not(feature = "media"))]
pub fn prepare(_req: &MmRequest, _files: &ModelFiles, _text_hidden: usize) -> Result<(String, Vec<MmFeature>)> {
    bail!("built without the `media` feature: image, video and audio inputs are unavailable")
}

#[cfg(feature = "media")]
pub fn prepare(args: &MmRequest, files: &ModelFiles, text_hidden: usize) -> Result<(String, Vec<MmFeature>)> {
    if let Some(qc) = qwen_vision::QwenMmConfig::load(args.model)? {
        return prepare_qwen(args, files, text_hidden, qc);
    }
    let mmc = mm_config::MmConfig::load(args.model)?;
    let media = media::Media::new();
    let mut text = String::new();
    let mut feats = Vec::new();
    if !args.images.is_empty() || !args.videos.is_empty() {
        let vcfg = mmc.vision.clone().context("checkpoint has no vision tower")?;
        let t = Instant::now();
        let tower = vision::VisionTower::load(vcfg, files, text_hidden)?;
        let budget = if args.max_soft_tokens > 0 {
            args.max_soft_tokens
        } else {
            mmc.vision_soft_tokens_per_image
        };
        if !vision::SUPPORTED_SOFT_TOKENS.contains(&budget) {
            bail!("--max-soft-tokens must be one of {:?}", vision::SUPPORTED_SOFT_TOKENS);
        }
        for (i, path) in args.images.iter().enumerate() {
            let img = media.load_image(path)?;
            let patches = vision::preprocess(&img.data, img.width, img.height, &tower.cfg, budget)?;
            let emb = tower.encode(&patches);
            let n = emb.len() / text_hidden;
            eprintln!(
                "[image {}: {}x{} -> {}x{} patches -> {} soft tokens, {:.2}s]",
                path.display(),
                img.width,
                img.height,
                patches.pw,
                patches.ph,
                n,
                t.elapsed().as_secs_f64()
            );
            if let (Some(prefix), 0) = (args.dump_mm, i) {
                dump(&format!("{}_pixels.bin", prefix.display()), &patches.patches)?;
                dump(&format!("{}_image.bin", prefix.display()), &emb)?;
            }
            text.push_str("<|image>");
            for _ in 0..n {
                text.push_str("<|image|>");
            }
            text.push_str("<image|>");
            feats.push(MmFeature {
                token_id: mmc.image_token_id,
                emb,
                n,
                grids: Vec::new(),
            });
        }
        // Videos: sampled frames through the image tower at the video processor's budget, each
        // as `mm:ss <|image>...<image|>`, separated by spaces.
        let vp = GemmaVideoConfig::load(args.model);
        for (i, path) in args.videos.iter().enumerate() {
            let t = Instant::now();
            let num_frames = args.video.max_frames.unwrap_or(vp.num_frames);
            let select = |total: usize, fps: f64| video::sample_uniform(total, fps, args.video.fps, num_frames);
            let clip = video::load_clip(&media, path, &select, args.video.frames_fps)?;
            let mut emb = Vec::new();
            let mut parts = Vec::with_capacity(clip.frames.len());
            for (f, secs) in clip.frames.iter().zip(clip.times()) {
                let patches = vision::preprocess_frame(f, clip.width, clip.height, &tower.cfg, vp.max_soft_tokens)?;
                if let (Some(prefix), 0, true) = (args.dump_mm, i, emb.is_empty()) {
                    dump(&format!("{}_vpixels.bin", prefix.display()), &patches.patches)?;
                }
                let e = tower.encode(&patches);
                let n = e.len() / text_hidden;
                emb.extend_from_slice(&e);
                let s = secs as u64;
                parts.push(format!("{:02}:{:02} <|image>{}<image|>", s / 60, s % 60, "<|video|>".repeat(n)));
            }
            let n = emb.len() / text_hidden;
            eprintln!(
                "[video {}: {} of {} frames ({}x{}, {:.2} fps) -> {} soft tokens, {:.2}s]",
                path.display(),
                clip.frames.len(),
                clip.total,
                clip.width,
                clip.height,
                clip.fps,
                n,
                t.elapsed().as_secs_f64()
            );
            if let (Some(prefix), 0) = (args.dump_mm, i) {
                dump(&format!("{}_video.bin", prefix.display()), &emb)?;
            }
            text.push_str(&parts.join(" "));
            feats.push(MmFeature {
                token_id: mmc.video_token_id,
                emb,
                n,
                grids: Vec::new(),
            });
        }
    }
    if !args.audios.is_empty() {
        let acfg = mmc.audio.clone().context("checkpoint has no audio tower")?;
        let t = Instant::now();
        let tower = audio::AudioTower::load(acfg, files, text_hidden)?;
        for (i, path) in args.audios.iter().enumerate() {
            let mut pcm = media.load_audio(path, audio::SAMPLE_RATE)?;
            // The reference feature extractor pads the waveform to a multiple of 128 samples.
            let padded = pcm.len().div_ceil(128) * 128;
            pcm.resize(padded, 0.0);
            let frames = audio::num_mel_frames(pcm.len());
            if frames == 0 {
                bail!("{}: audio too short", path.display());
            }
            let mel = audio::mel_features(&pcm);
            let emb = tower.encode(&mel, frames);
            let n = emb.len() / text_hidden;
            eprintln!(
                "[audio {}: {:.2}s -> {} mel frames -> {} soft tokens, {:.2}s]",
                path.display(),
                pcm.len() as f64 / audio::SAMPLE_RATE as f64,
                frames,
                n,
                t.elapsed().as_secs_f64()
            );
            if let (Some(prefix), 0) = (args.dump_mm, i) {
                dump(&format!("{}_mel.bin", prefix.display()), &mel)?;
                dump(&format!("{}_audio.bin", prefix.display()), &emb)?;
            }
            text.push_str("<|audio>");
            for _ in 0..n {
                text.push_str("<|audio|>");
            }
            text.push_str("<audio|>");
            feats.push(MmFeature {
                token_id: mmc.audio_token_id,
                emb,
                n,
                grids: Vec::new(),
            });
        }
    }
    Ok((text, feats))
}

#[cfg(feature = "media")]
fn dump(path: &str, v: &[f32]) -> Result<()> {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(path, b).with_context(|| format!("write {path}"))
}

/// Gemma 4's video processor settings (`processor_config.json`'s `video_processor`).
#[cfg(feature = "media")]
struct GemmaVideoConfig {
    num_frames: usize,
    max_soft_tokens: usize,
}

#[cfg(feature = "media")]
impl GemmaVideoConfig {
    fn load(dir: &Path) -> Self {
        let v: Option<serde_json::Value> = crate::fsx::read(&dir.join("processor_config.json"))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok());
        let vp = v.as_ref().and_then(|v| v.get("video_processor"));
        let int = |k: &str, d: usize| vp.and_then(|v| v.get(k)).and_then(|x| x.as_u64()).map_or(d, |x| x as usize);
        Self {
            num_frames: int("num_frames", 32),
            max_soft_tokens: int("max_soft_tokens", 70),
        }
    }
}

/// Qwen3.5-family attachments: `<|vision_start|><|image_pad|>...<|vision_end|>` per image, and per
/// video `<|vision_start|>` + `<t seconds><|vision_start|><|video_pad|>...<|vision_end|>` per
/// temporal patch + `<|vision_end|>` (the chat template's placeholder expanded by
/// `Qwen3VLProcessor`).
#[cfg(feature = "media")]
fn prepare_qwen(
    args: &MmRequest,
    files: &ModelFiles,
    text_hidden: usize,
    qc: qwen_vision::QwenMmConfig,
) -> Result<(String, Vec<MmFeature>)> {
    if !args.audios.is_empty() {
        bail!("this model takes no audio input");
    }
    let mut text = String::new();
    let mut feats = Vec::new();
    if args.images.is_empty() && args.videos.is_empty() {
        return Ok((text, feats));
    }
    if qc.vision.out_hidden_size != text_hidden {
        bail!(
            "vision tower output {} does not match the text model's {text_hidden}",
            qc.vision.out_hidden_size
        );
    }
    let media = media::Media::new();
    let t = Instant::now();
    let tower = qwen_vision::QwenVisionTower::load(qc.vision.clone(), files, args.quant)?;
    let m = qc.vision.spatial_merge_size;
    let unit = qc.image.patch_size * m;
    let max_pixels = args.max_soft_tokens * unit * unit;
    for (i, path) in args.images.iter().enumerate() {
        let img = media.load_image(path)?;
        let px = qwen_vision::preprocess_image(&img.data, img.width, img.height, &qc.image, max_pixels)?;
        let t1 = Instant::now();
        let emb = tower.encode(&px);
        let (_, gh, gw) = px.grid;
        let n = emb.len() / text_hidden;
        eprintln!(
            "[image {}: {}x{} -> {}x{} patches -> {} tokens, tower {:.2}s, {:.2}s]",
            path.display(),
            img.width,
            img.height,
            gw,
            gh,
            n,
            t1.elapsed().as_secs_f64(),
            t.elapsed().as_secs_f64()
        );
        if let (Some(prefix), 0) = (args.dump_mm, i) {
            dump(&format!("{}_pixels.bin", prefix.display()), &px.data)?;
            dump(&format!("{}_image.bin", prefix.display()), &emb)?;
        }
        text.push_str("<|vision_start|>");
        text.push_str(&"<|image_pad|>".repeat(n));
        text.push_str("<|vision_end|>");
        feats.push(MmFeature {
            token_id: qc.image_token_id,
            emb,
            n,
            grids: vec![(gh / m, gw / m)],
        });
    }
    let vp = &qc.video;
    for (i, path) in args.videos.iter().enumerate() {
        let t1 = Instant::now();
        let fps = args.video.fps.unwrap_or(vp.fps);
        let max_frames = args.video.max_frames.unwrap_or(vp.max_frames);
        let select = |total: usize, vfps: f64| video::sample_qwen(total, vfps, fps, vp.min_frames.min(max_frames), max_frames);
        let clip = video::load_clip(&media, path, &select, args.video.frames_fps)?;
        let refs: Vec<&[u8]> = clip.frames.iter().map(|f| f.as_slice()).collect();
        let px = qwen_vision::preprocess_video(&refs, clip.width, clip.height, vp)?;
        let emb = tower.encode(&px);
        let (gt, gh, gw) = px.grid;
        let n = emb.len() / text_hidden;
        let per = (gh / m) * (gw / m);
        let times = qwen_vision::temporal_patch_times(&clip.indices, clip.fps, vp.temporal_patch_size);
        eprintln!(
            "[video {}: {} of {} frames ({}x{}, {:.2} fps) -> {gt}x{gw}x{gh} patches -> {n} tokens, {:.2}s]",
            path.display(),
            clip.frames.len(),
            clip.total,
            clip.width,
            clip.height,
            clip.fps,
            t1.elapsed().as_secs_f64()
        );
        if let (Some(prefix), 0) = (args.dump_mm, i) {
            dump(&format!("{}_vpixels.bin", prefix.display()), &px.data)?;
            dump(&format!("{}_video.bin", prefix.display()), &emb)?;
        }
        text.push_str("<|vision_start|>");
        for &secs in &times {
            text.push_str(&format!("<{secs:.1} seconds><|vision_start|>"));
            text.push_str(&"<|video_pad|>".repeat(per));
            text.push_str("<|vision_end|>");
        }
        text.push_str("<|vision_end|>");
        feats.push(MmFeature {
            token_id: qc.video_token_id,
            emb,
            n,
            grids: vec![(gh / m, gw / m); gt],
        });
    }
    Ok((text, feats))
}

/// Maps placeholder token positions in `ids` to soft-token embeddings, in attachment order.
pub fn overrides(ids: &[u32], feats: &[MmFeature], hidden: usize) -> Result<Vec<(usize, Vec<f32>)>> {
    let mut out = Vec::new();
    let mut cursor = 0usize;
    for f in feats {
        let mut k = 0;
        while k < f.n {
            let pos = ids[cursor..]
                .iter()
                .position(|&t| t == f.token_id)
                .context("fewer placeholder tokens than soft tokens")?
                + cursor;
            out.push((pos, f.emb[k * hidden..(k + 1) * hidden].to_vec()));
            cursor = pos + 1;
            k += 1;
        }
    }
    Ok(out)
}

/// Qwen M-RoPE: `(position in ids, rows, cols)` of each run of placeholder tokens (see
/// `MmFeature::grids`), to declare with `QwenSession::add_vision_span` before the prefill.
pub fn vision_spans(ids: &[u32], feats: &[MmFeature]) -> Result<Vec<(usize, usize, usize)>> {
    let mut out = Vec::new();
    let mut cursor = 0usize;
    for f in feats {
        for &(gh, gw) in &f.grids {
            let pos = ids[cursor..]
                .iter()
                .position(|&t| t == f.token_id)
                .context("fewer placeholder tokens than vision tokens")?
                + cursor;
            let run = gh * gw;
            if pos + run > ids.len() || ids[pos..pos + run].iter().any(|&t| t != f.token_id) {
                bail!("vision placeholder run at {pos} is shorter than {run} tokens");
            }
            out.push((pos, gh, gw));
            cursor = pos + run;
        }
    }
    Ok(out)
}

/// Prepares a prompt's attachments for a prefill at the session's current position: declares
/// the vision spans (Qwen M-RoPE) on `sess` and returns the embedding overrides for
/// `generate::prefill`.
pub fn bind<M: crate::generate::TextLm + ?Sized>(
    model: &M,
    sess: &mut M::Sess,
    ids: &[u32],
    feats: &[MmFeature],
) -> Result<Vec<(usize, Vec<f32>)>> {
    if feats.is_empty() {
        return Ok(Vec::new());
    }
    let base = model.session_len(sess);
    for (p, gh, gw) in vision_spans(ids, feats)? {
        model.add_vision_span(sess, base + p, gh, gw);
    }
    overrides(ids, feats, model.hidden_size())
}
