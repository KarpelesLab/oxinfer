//! Image, video and audio attachments: decoding (files through oxideav with the `media` feature,
//! decoded oxideav frames with the `oxideav` feature), encoding with the vision / audio towers
//! into soft tokens, and mapping them onto the placeholder tokens of the prompt. Gemma 4 takes
//! images, videos (frames as images with `mm:ss` timestamps) and audio; Qwen3.5 / 3.6 / 3.8 take
//! images and videos (temporal patches of two frames with `<t seconds>` timestamps, placed with
//! M-RoPE).

use crate::clock::Instant;
use crate::safetensors::ModelFiles;
use crate::video::{self, Clip, RgbImage, Sampler, VideoOptions};
use crate::{audio, mm_config, qwen_vision, vision};
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
            dump_mm: None,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.images.is_empty() && self.audios.is_empty() && self.videos.is_empty()
    }
}

/// Decoded attachments (the `oxideav` feature) for [`prepare_frames`]: each kind goes after the
/// files of the same kind in [`MmRequest`].
#[cfg(feature = "oxideav")]
#[derive(Clone, Copy, Debug, Default)]
pub struct MmFrames<'a> {
    pub images: &'a [crate::av::Picture],
    /// Audio clips (Gemma 4 only).
    pub audios: &'a [crate::av::AudioClip],
    pub videos: &'a [crate::av::VideoClip],
}

#[cfg(feature = "oxideav")]
impl MmFrames<'_> {
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

/// The attachments of a request in prompt order (files first, then decoded frames), decoded on
/// demand. The oxideav registries are built only when a file is read.
struct Sources<'a> {
    req: &'a MmRequest<'a>,
    #[cfg(feature = "oxideav")]
    frames: MmFrames<'a>,
    #[cfg(feature = "media")]
    media: std::cell::OnceCell<crate::media::Media>,
}

impl<'a> Sources<'a> {
    fn new(req: &'a MmRequest<'a>) -> Self {
        Self {
            req,
            #[cfg(feature = "oxideav")]
            frames: MmFrames::default(),
            #[cfg(feature = "media")]
            media: std::cell::OnceCell::new(),
        }
    }

    #[cfg(feature = "media")]
    fn media(&self) -> &crate::media::Media {
        self.media.get_or_init(crate::media::Media::new)
    }

    fn n_images(&self) -> usize {
        #[cfg(feature = "oxideav")]
        let n = self.frames.images.len();
        #[cfg(not(feature = "oxideav"))]
        let n = 0;
        self.req.images.len() + n
    }

    fn n_videos(&self) -> usize {
        #[cfg(feature = "oxideav")]
        let n = self.frames.videos.len();
        #[cfg(not(feature = "oxideav"))]
        let n = 0;
        self.req.videos.len() + n
    }

    fn n_audios(&self) -> usize {
        #[cfg(feature = "oxideav")]
        let n = self.frames.audios.len();
        #[cfg(not(feature = "oxideav"))]
        let n = 0;
        self.req.audios.len() + n
    }

    /// Image `i`: a label for the logs and its pixels.
    fn image(&self, i: usize) -> Result<(String, RgbImage)> {
        let Some(path) = self.req.images.get(i) else {
            return self.frame_image(i - self.req.images.len());
        };
        Ok((path.display().to_string(), self.load_image(path)?))
    }

    /// Video `i`: a label and the frames `select` picks.
    fn video(&self, i: usize, select: Sampler) -> Result<(String, Clip)> {
        let Some(path) = self.req.videos.get(i) else {
            return self.frame_video(i - self.req.videos.len(), select);
        };
        Ok((path.display().to_string(), self.load_clip(path, select)?))
    }

    /// Audio clip `i`: a label and its mono samples at `rate`.
    fn audio(&self, i: usize, rate: usize) -> Result<(String, Vec<f32>)> {
        let Some(path) = self.req.audios.get(i) else {
            return self.frame_audio(i - self.req.audios.len(), rate);
        };
        Ok((path.display().to_string(), self.load_audio(path, rate)?))
    }

    #[cfg(feature = "media")]
    fn load_image(&self, path: &Path) -> Result<RgbImage> {
        self.media().load_image(path)
    }

    #[cfg(feature = "media")]
    fn load_clip(&self, path: &Path, select: Sampler) -> Result<Clip> {
        video::load_clip(self.media(), path, select, self.req.video.frames_fps)
    }

    #[cfg(feature = "media")]
    fn load_audio(&self, path: &Path, rate: usize) -> Result<Vec<f32>> {
        self.media().load_audio(path, rate)
    }

    #[cfg(not(feature = "media"))]
    fn load_image(&self, path: &Path) -> Result<RgbImage> {
        bail!(
            "{}: built without the `media` feature, so image files cannot be decoded",
            path.display()
        )
    }

    #[cfg(not(feature = "media"))]
    fn load_clip(&self, path: &Path, _select: Sampler) -> Result<Clip> {
        bail!(
            "{}: built without the `media` feature, so video files cannot be decoded",
            path.display()
        )
    }

    #[cfg(not(feature = "media"))]
    fn load_audio(&self, path: &Path, _rate: usize) -> Result<Vec<f32>> {
        bail!(
            "{}: built without the `media` feature, so audio files cannot be decoded",
            path.display()
        )
    }

    #[cfg(feature = "oxideav")]
    fn frame_image(&self, k: usize) -> Result<(String, RgbImage)> {
        let img = self.frames.images[k].to_rgb().with_context(|| format!("image frame {}", k + 1))?;
        Ok((format!("frame {}", k + 1), img))
    }

    #[cfg(feature = "oxideav")]
    fn frame_video(&self, k: usize, select: Sampler) -> Result<(String, Clip)> {
        let clip = self.frames.videos[k]
            .sample(select)
            .with_context(|| format!("video clip {}", k + 1))?;
        Ok((format!("clip {}", k + 1), clip))
    }

    #[cfg(feature = "oxideav")]
    fn frame_audio(&self, k: usize, rate: usize) -> Result<(String, Vec<f32>)> {
        let pcm = self.frames.audios[k]
            .to_mono(rate)
            .with_context(|| format!("audio clip {}", k + 1))?;
        Ok((format!("audio frames {}", k + 1), pcm))
    }

    #[cfg(not(feature = "oxideav"))]
    fn frame_image(&self, k: usize) -> Result<(String, RgbImage)> {
        bail!("no image {k}")
    }

    #[cfg(not(feature = "oxideav"))]
    fn frame_video(&self, k: usize, _select: Sampler) -> Result<(String, Clip)> {
        bail!("no video {k}")
    }

    #[cfg(not(feature = "oxideav"))]
    fn frame_audio(&self, k: usize, _rate: usize) -> Result<(String, Vec<f32>)> {
        bail!("no audio clip {k}")
    }
}

/// Decodes and encodes all attachments. Returns the placeholder text to prepend to the user
/// message and the features in the same order. Files need the `media` feature.
pub fn prepare(args: &MmRequest, files: &ModelFiles, text_hidden: usize) -> Result<(String, Vec<MmFeature>)> {
    prepare_sources(&Sources::new(args), files, text_hidden)
}

/// [`prepare`] with decoded frames too, each kind after the files of `args` (the `oxideav`
/// feature; the files still need `media`).
#[cfg(feature = "oxideav")]
pub fn prepare_frames(args: &MmRequest, frames: MmFrames, files: &ModelFiles, text_hidden: usize) -> Result<(String, Vec<MmFeature>)> {
    let mut src = Sources::new(args);
    src.frames = frames;
    prepare_sources(&src, files, text_hidden)
}

fn prepare_sources(src: &Sources, files: &ModelFiles, text_hidden: usize) -> Result<(String, Vec<MmFeature>)> {
    let args = src.req;
    if let Some(qc) = qwen_vision::QwenMmConfig::load(args.model)? {
        return prepare_qwen(src, files, text_hidden, qc);
    }
    let mmc = mm_config::MmConfig::load(args.model)?;
    let mut text = String::new();
    let mut feats = Vec::new();
    if src.n_images() > 0 || src.n_videos() > 0 {
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
        for i in 0..src.n_images() {
            let (what, img) = src.image(i)?;
            let patches = vision::preprocess(&img.data, img.width, img.height, &tower.cfg, budget)?;
            let emb = tower.encode(&patches);
            let n = emb.len() / text_hidden;
            eprintln!(
                "[image {what}: {}x{} -> {}x{} patches -> {} soft tokens, {:.2}s]",
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
        for i in 0..src.n_videos() {
            let t = Instant::now();
            let num_frames = args.video.max_frames.unwrap_or(vp.num_frames);
            let select = |total: usize, fps: f64| video::sample_uniform(total, fps, args.video.fps, num_frames);
            let (what, clip) = src.video(i, &select)?;
            let frames = clip
                .frames
                .iter()
                .map(|f| vision::preprocess(f, clip.width, clip.height, &tower.cfg, vp.max_soft_tokens))
                .collect::<Result<Vec<_>>>()?;
            if let (Some(prefix), 0) = (args.dump_mm, i) {
                dump(&format!("{}_vpixels.bin", prefix.display()), &frames[0].patches)?;
            }
            // Frames go through the tower in batches of up to ~16k patches.
            let per = frames[0].n.max(1);
            let mut embs = Vec::with_capacity(frames.len());
            for batch in frames.chunks((16384 / per).max(1)) {
                let refs: Vec<&vision::ImagePatches> = batch.iter().collect();
                embs.extend(tower.encode_many(&refs));
            }
            let mut emb = Vec::new();
            let mut parts = Vec::with_capacity(frames.len());
            for (e, secs) in embs.iter().zip(clip.times()) {
                let n = e.len() / text_hidden;
                emb.extend_from_slice(e);
                let s = secs as u64;
                parts.push(format!("{:02}:{:02} <|image>{}<image|>", s / 60, s % 60, "<|video|>".repeat(n)));
            }
            let n = emb.len() / text_hidden;
            eprintln!(
                "[video {what}: {} of {} frames ({}x{}, {:.2} fps) -> {} soft tokens, {:.2}s]",
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
    if src.n_audios() > 0 {
        let acfg = mmc.audio.clone().context("checkpoint has no audio tower")?;
        let t = Instant::now();
        let tower = audio::AudioTower::load(acfg, files, text_hidden)?;
        for i in 0..src.n_audios() {
            let (what, mut pcm) = src.audio(i, audio::SAMPLE_RATE)?;
            // The reference feature extractor pads the waveform to a multiple of 128 samples.
            let padded = pcm.len().div_ceil(128) * 128;
            pcm.resize(padded, 0.0);
            let frames = audio::num_mel_frames(pcm.len());
            if frames == 0 {
                bail!("{what}: audio too short");
            }
            let mel = audio::mel_features(&pcm);
            let emb = tower.encode(&mel, frames);
            let n = emb.len() / text_hidden;
            eprintln!(
                "[audio {what}: {:.2}s -> {} mel frames -> {} soft tokens, {:.2}s]",
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

fn dump(path: &str, v: &[f32]) -> Result<()> {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(path, b).with_context(|| format!("write {path}"))
}

/// Gemma 4's video processor settings (`processor_config.json`'s `video_processor`).
struct GemmaVideoConfig {
    num_frames: usize,
    max_soft_tokens: usize,
}

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
fn prepare_qwen(src: &Sources, files: &ModelFiles, text_hidden: usize, qc: qwen_vision::QwenMmConfig) -> Result<(String, Vec<MmFeature>)> {
    let args = src.req;
    if src.n_audios() > 0 {
        bail!("this model takes no audio input");
    }
    let mut text = String::new();
    let mut feats = Vec::new();
    if src.n_images() == 0 && src.n_videos() == 0 {
        return Ok((text, feats));
    }
    if qc.vision.out_hidden_size != text_hidden {
        bail!(
            "vision tower output {} does not match the text model's {text_hidden}",
            qc.vision.out_hidden_size
        );
    }
    let t = Instant::now();
    let tower = qwen_vision::QwenVisionTower::load(qc.vision.clone(), files)?;
    let m = qc.vision.spatial_merge_size;
    let unit = qc.image.patch_size * m;
    let max_pixels = args.max_soft_tokens * unit * unit;
    for i in 0..src.n_images() {
        let (what, img) = src.image(i)?;
        let (px, ph, feat) = qwen_image(&tower, &qc, text_hidden, &img, max_pixels, &what, t)?;
        if let (Some(prefix), 0) = (args.dump_mm, i) {
            dump(&format!("{}_pixels.bin", prefix.display()), &px.data)?;
            dump(&format!("{}_image.bin", prefix.display()), &feat.emb)?;
        }
        text.push_str(&ph);
        feats.push(feat);
    }
    let vp = &qc.video;
    for i in 0..src.n_videos() {
        let t1 = Instant::now();
        let fps = args.video.fps.unwrap_or(vp.fps);
        let max_frames = args.video.max_frames.unwrap_or(vp.max_frames);
        let select = |total: usize, vfps: f64| video::sample_qwen(total, vfps, fps, vp.min_frames.min(max_frames), max_frames);
        let (what, clip) = src.video(i, &select)?;
        let refs: Vec<&[u8]> = clip.frames.iter().map(|f| f.as_slice()).collect();
        let px = qwen_vision::preprocess_video(&refs, clip.width, clip.height, vp)?;
        let emb = tower.encode(&px);
        let (gt, gh, gw) = px.grid;
        let n = emb.len() / text_hidden;
        let per = (gh / m) * (gw / m);
        let times = qwen_vision::temporal_patch_times_of(&clip.times(), vp.temporal_patch_size);
        eprintln!(
            "[video {what}: {} of {} frames ({}x{}, {:.2} fps) -> {gt}x{gw}x{gh} patches -> {n} tokens, {:.2}s]",
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

/// One Qwen image through the tower: its pixels, its placeholder text
/// (`<|vision_start|><|image_pad|>...<|vision_end|>`) and its feature.
fn qwen_image(
    tower: &qwen_vision::QwenVisionTower,
    qc: &qwen_vision::QwenMmConfig,
    text_hidden: usize,
    img: &RgbImage,
    max_pixels: usize,
    what: &str,
    t: Instant,
) -> Result<(qwen_vision::Pixels, String, MmFeature)> {
    let m = qc.vision.spatial_merge_size;
    let px = qwen_vision::preprocess_image(&img.data, img.width, img.height, &qc.image, max_pixels)?;
    let t1 = Instant::now();
    let emb = tower.encode(&px);
    let (_, gh, gw) = px.grid;
    let n = emb.len() / text_hidden;
    eprintln!(
        "[image {what}: {}x{} -> {}x{} patches -> {} tokens, tower {:.2}s, {:.2}s]",
        img.width,
        img.height,
        gw,
        gh,
        n,
        t1.elapsed().as_secs_f64(),
        t.elapsed().as_secs_f64()
    );
    let text = format!("<|vision_start|>{}<|vision_end|>", "<|image_pad|>".repeat(n));
    let feat = MmFeature {
        token_id: qc.image_token_id,
        emb,
        n,
        grids: vec![(gh / m, gw / m)],
    };
    Ok((px, text, feat))
}

/// Images for a Qwen3.5-family vision checkpoint in `model`: per image, its placeholder text
/// (`<|vision_start|><|image_pad|>...<|vision_end|>`, to place anywhere in a prompt) and its
/// feature. `max_soft_tokens` as in [`MmRequest`]. For prompts that are not chat turns (typed
/// decisions).
pub fn encode_rgb_images(
    model: &Path,
    files: &ModelFiles,
    text_hidden: usize,
    images: &[RgbImage],
    max_soft_tokens: usize,
) -> Result<Vec<(String, MmFeature)>> {
    let qc = qwen_vision::QwenMmConfig::load(model)?
        .with_context(|| format!("{}: no Qwen vision tower (config.json has no vision_config)", model.display()))?;
    if qc.vision.out_hidden_size != text_hidden {
        bail!(
            "vision tower output {} does not match the text model's {text_hidden}",
            qc.vision.out_hidden_size
        );
    }
    let t = Instant::now();
    let tower = qwen_vision::QwenVisionTower::load(qc.vision.clone(), files).context("load the vision tower")?;
    let unit = qc.image.patch_size * qc.vision.spatial_merge_size;
    let mut out = Vec::with_capacity(images.len());
    for (i, img) in images.iter().enumerate() {
        let (_, text, feat) = qwen_image(
            &tower,
            &qc,
            text_hidden,
            img,
            max_soft_tokens * unit * unit,
            &(i + 1).to_string(),
            t,
        )?;
        out.push((text, feat));
    }
    Ok(out)
}

/// [`encode_rgb_images`] for encoded images (PNG, JPEG, WebP, GIF, BMP bytes; `media` feature).
pub fn encode_images(
    model: &Path,
    files: &ModelFiles,
    text_hidden: usize,
    images: &[Vec<u8>],
    max_soft_tokens: usize,
) -> Result<Vec<(String, MmFeature)>> {
    let rgb = decode_images(images)?;
    encode_rgb_images(model, files, text_hidden, &rgb, max_soft_tokens)
}

/// Decodes encoded images (PNG, JPEG, WebP, GIF, BMP; the `media` feature).
pub fn decode_images(images: &[Vec<u8>]) -> Result<Vec<RgbImage>> {
    #[cfg(feature = "media")]
    {
        let media = crate::media::Media::new();
        images
            .iter()
            .enumerate()
            .map(|(i, b)| media.decode_image(b).with_context(|| format!("image {}", i + 1)))
            .collect()
    }
    #[cfg(not(feature = "media"))]
    {
        if images.is_empty() {
            return Ok(Vec::new());
        }
        bail!("built without the `media` feature: encoded images cannot be decoded")
    }
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
