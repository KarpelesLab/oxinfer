//! LTX-2.5 (`Lightricks/LTX-2.5`): a 22B audio-video diffusion transformer that makes a video
//! with its soundtrack from a prompt (text to video) and from images (image to video), following
//! the release's `ltx_pipelines.distilled` (`DistilledPipeline`):
//!
//! 1. the prompt through the Gemma 4 12B text encoder and its projections ([`text`]), then the
//!    transformer's two connectors (1024 context rows per stream, padding replaced by learned
//!    registers);
//! 2. stage 1 at half the width and height: noise (video first, then audio), the distilled
//!    schedule ([`DISTILLED_SIGMAS`], 8 forwards) with the ancestral Euler sampler (eta 1, the
//!    sampler of LTX-2.5 checkpoints), the transformer predicting velocities (`X0Model`:
//!    `x - t v` per row);
//! 3. the spatial x2 latent upsampler ([`upsample`]), then stage 2 at the full size: the
//!    upsampled latents (and the stage's audio latents) noised to `sigma = 0.909375` and
//!    [`STAGE_2_SIGMAS`] (3 forwards);
//! 4. the video VAE decoder ([`vae`], or the diffusion decoder [`diffvae`]) and the audio VAE, vocoder and bandwidth extension
//!    ([`audio`], 48 kHz stereo).
//!
//! An image conditions the latent frame it falls in: its VAE latents replace that frame's rows,
//! which then keep timestep `1 - strength` (0: never denoised). Frame counts are `8 k + 1` (the
//! VAE's grid, rounded down); with the duration head (`model_patches/`) and no frame count, the
//! clip's length is predicted from the prompt ([`duration`]).
//!
//! Single-stage generation (`two_stage = false`) runs the distilled schedule at the full size and
//! skips the upsampler. Temporal upscaling runs the release's DFR temporal rounds ([`dfr`]):
//! the clip made on a keyframe canvas with generated keyframe slots, then per round the temporal
//! x2 upsampler and a tiled refinement between the keyframes.

pub mod api;
pub mod audio;
pub mod audio_enc;
pub(crate) mod device;
pub mod dfr;
pub mod diffvae;
pub mod dit;
pub mod duration;
pub mod enhance;
pub mod guidance;
pub mod inputs;
pub mod lora;
pub(crate) mod nn;
pub mod text;
#[doc(hidden)]
pub mod tiny;
pub mod upsample;
pub mod vae;

use crate::clock::Instant;
use crate::minimax_h3::video::{Soundtrack, Video};
use crate::safetensors::ModelFiles;
pub use crate::sd::Image;
use anyhow::{Context, Result, bail};
pub use diffvae::DiffDecoder;
use dit::{Dit, Perturb, StreamIn};
pub use guidance::{DEFAULT_NEGATIVE_PROMPT, Guidance, GuidanceMode, GuiderParams};
pub use inputs::{AudioInput, Retake, VideoInput};
use std::path::{Path, PathBuf};

/// The distilled stage-1 schedule (`DISTILLED_SIGMA_VALUES`).
pub const DISTILLED_SIGMAS: [f32; 9] = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0];
/// The stage-2 schedule (`STAGE_2_DISTILLED_SIGMA_VALUES`).
pub const STAGE_2_SIGMAS: [f32; 4] = [0.909375, 0.725, 0.421875, 0.0];
/// Seed offsets of the ancestral sampler's noise per stage (`ANCESTRAL_*_NOISE_SEED_OFFSET`).
const STAGE_1_NOISE_OFFSET: u64 = 10000;
const STAGE_2_NOISE_OFFSET: u64 = 20000;
/// Frames of video per latent frame, pixels per latent row and column.
pub const TIME_SCALE: usize = 8;
pub const SPACE_SCALE: usize = 32;
/// Audio latents per second (16 kHz, hop 160, 4 mel frames per latent).
pub const AUDIO_LATENTS_PER_SECOND: f64 = 25.0;
/// The release's default length and rate.
pub const DEFAULT_FRAMES: usize = 121;
pub const DEFAULT_FPS: f32 = 24.0;

/// The release's file names (`Lightricks/LTX-2.5`, mirrored publicly as `comfyicu/LTX-2.5`).
pub mod files {
    pub const DISTILLED_INT8: &str = "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors";
    pub const DISTILLED_BF16: &str = "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors";
    pub const DISTILLED_NVFP4: &str = "diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors";
    pub const DEV_INT8: &str = "diffusion_models/ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors";
    pub const DEV_BF16: &str = "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors";
    pub const TEXT_INT8: &str = "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors";
    pub const TEXT_BF16: &str = "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors";
    pub const VIDEO_VAE: &str = "vae/ltx-2.5-video-vae-conv-bf16.safetensors";
    /// The diffusion video decoder ("DiffVAE"; its encoder is the convolutional VAE's).
    pub const VIDEO_VAE_DIFF: &str = "vae/ltx-2.5-video-vae-bf16.safetensors";
    pub const AUDIO_VAE: &str = "vae/ltx-2.5-audio-vae-bf16.safetensors";
    pub const SPATIAL_UPSAMPLER: &str = "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors";
    pub const TEMPORAL_UPSAMPLER: &str = "latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors";
    pub const DURATION_HEAD: &str = "model_patches/ltx-2.5-duration-head-bf16.safetensors";
    pub const DISTILLED_LORA: &str = "loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors";
    /// The public mirror auto-downloads come from (the release itself is gated).
    pub const MIRROR: &str = "comfyicu/LTX-2.5";
}

/// The transformer files of the release.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Variant {
    /// Distilled, ComfyUI int8 ConvRot (21.5 GB): the default.
    #[default]
    DistilledInt8,
    DistilledBf16,
    DistilledNvfp4,
    /// The full model (for the distilled LoRA, or LoRAs trained on it).
    DevInt8,
    DevBf16,
}

impl Variant {
    /// `int8` / `distilled`, `bf16`, `nvfp4`, `dev` / `dev-int8`, `dev-bf16`.
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s.to_ascii_lowercase().as_str() {
            "int8" | "distilled" | "distilled-int8" => Self::DistilledInt8,
            "bf16" | "distilled-bf16" => Self::DistilledBf16,
            "nvfp4" | "fp4" | "distilled-nvfp4" => Self::DistilledNvfp4,
            "dev" | "dev-int8" => Self::DevInt8,
            "dev-bf16" => Self::DevBf16,
            _ => return None,
        })
    }

    pub fn file(self) -> &'static str {
        match self {
            Self::DistilledInt8 => files::DISTILLED_INT8,
            Self::DistilledBf16 => files::DISTILLED_BF16,
            Self::DistilledNvfp4 => files::DISTILLED_NVFP4,
            Self::DevInt8 => files::DEV_INT8,
            Self::DevBf16 => files::DEV_BF16,
        }
    }

    pub fn is_dev(self) -> bool {
        matches!(self, Self::DevInt8 | Self::DevBf16)
    }
}

/// The video decoder of a run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum VideoVae {
    /// The convolutional VAE (`vae/ltx-2.5-video-vae-conv-bf16.safetensors`, 1.45 GB).
    #[default]
    Conv,
    /// The diffusion decoder (`vae/ltx-2.5-video-vae-bf16.safetensors`, 1.47 GB; [`diffvae`]).
    Diff,
}

impl VideoVae {
    /// `conv`, `diff` (`diffvae`, `diffusion`).
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s.to_ascii_lowercase().as_str() {
            "conv" | "convolutional" => Self::Conv,
            "diff" | "diffvae" | "diffusion" => Self::Diff,
            _ => return None,
        })
    }

    pub fn file(self) -> &'static str {
        match self {
            Self::Conv => files::VIDEO_VAE,
            Self::Diff => files::VIDEO_VAE_DIFF,
        }
    }
}

/// Whether `spec` names the release (`Lightricks/LTX-2.5`, gated, or its public mirror).
pub fn is_repo(spec: &str) -> bool {
    spec == "Lightricks/LTX-2.5" || spec == files::MIRROR
}

/// What [`fetch`] downloads beside the always-needed parts (text encoder, VAEs, spatial
/// upsampler, duration head).
#[derive(Clone, Debug, Default)]
pub struct FetchOptions {
    pub variant: Variant,
    /// The bf16 text encoder (26 GB) instead of the int8 one (15 GB).
    pub text_bf16: bool,
    pub temporal_upsampler: bool,
    /// The distilled LoRA (for the dev transformer).
    pub distilled_lora: bool,
    /// The video decoder.
    pub vae: VideoVae,
}

/// The files of a run fetched from the release's public mirror ([`files::MIRROR`]; the release
/// itself is gated) into the download cache; returns the folder (the release's layout).
pub fn fetch(spec: &str, o: &FetchOptions, cache_dir: Option<&Path>) -> Result<PathBuf> {
    let repo = if spec == "Lightricks/LTX-2.5" { files::MIRROR } else { spec };
    let mut names = vec![
        o.variant.file(),
        if o.text_bf16 { files::TEXT_BF16 } else { files::TEXT_INT8 },
        o.vae.file(),
        files::AUDIO_VAE,
        files::SPATIAL_UPSAMPLER,
        files::DURATION_HEAD,
    ];
    if o.temporal_upsampler {
        names.push(files::TEMPORAL_UPSAMPLER);
    }
    if o.distilled_lora {
        names.push(files::DISTILLED_LORA);
    }
    #[cfg(feature = "hf")]
    {
        crate::hf::resolve_files(repo, &names, cache_dir)
    }
    #[cfg(not(feature = "hf"))]
    {
        let _ = (cache_dir, names);
        bail!("{repo}: built without the `hf` feature, so the release's files cannot be fetched (name a local folder)")
    }
}

/// Whether `path` is an LTX-2.5 release folder (`diffusion_models/ltx-*` files) or one of its
/// transformer files.
pub fn is_ltx(path: &Path) -> bool {
    if path.is_dir() {
        return std::fs::read_dir(path.join("diffusion_models")).is_ok_and(|rd| {
            rd.flatten().any(|e| {
                let n = e.file_name().to_string_lossy().to_string();
                n.starts_with("ltx-") && n.ends_with(".safetensors")
            })
        });
    }
    is_transformer_file(path)
}

/// Whether `path` is a safetensors file of an LTX audio-video transformer (its `config`
/// metadata names `AVTransformer3DModel`).
pub fn is_transformer_file(path: &Path) -> bool {
    path.is_file()
        && path.extension().is_some_and(|e| e == "safetensors")
        && crate::safetensors::SafeTensors::open(path, false)
            .is_ok_and(|st| st.metadata("config").is_some_and(|c| c.contains("\"AVTransformer3DModel\"")))
}

/// Where the parts of a run come from.
#[derive(Clone, Debug, Default)]
pub struct Paths {
    pub transformer: PathBuf,
    pub text_encoder: Option<PathBuf>,
    pub video_vae: Option<PathBuf>,
    pub audio_vae: Option<PathBuf>,
    pub spatial_upsampler: Option<PathBuf>,
    pub temporal_upsampler: Option<PathBuf>,
    pub duration_head: Option<PathBuf>,
    /// LoRA files and their strengths, on for every stage.
    pub loras: Vec<(PathBuf, f32)>,
    /// The release's distilled LoRA (on the dev transformer): on for every stage of the distilled
    /// schedule; with guidance (the dev pipeline), off in the first stage and on in the second
    /// (`ti2vid_two_stages`).
    pub distilled_lora: Option<PathBuf>,
}

/// Whether `path` names a dev (non-distilled) transformer file of the release (`*-dev-*`).
pub fn is_dev_file(path: &Path) -> bool {
    path.file_name().is_some_and(|n| n.to_string_lossy().contains("-dev-"))
}

impl Paths {
    /// The parts of a release folder: the smallest distilled transformer present (int8 ConvRot,
    /// NVFP4, bf16; else a dev one), the int8 or bf16 text encoder, the convolutional VAE, the
    /// audio VAE, the upsamplers and the duration head when present. A transformer file alone
    /// takes the other parts from the folder above its own (`diffusion_models/..`).
    pub fn resolve(path: &Path) -> Result<Self> {
        Self::resolve_with(path, None)
    }

    /// [`Self::resolve_with`] with the video decoder `vae` (the other one when it is missing).
    pub fn resolve_vae(path: &Path, variant: Option<Variant>, vae: VideoVae) -> Result<Self> {
        let mut p = Self::resolve_with(path, variant)?;
        let root = if path.is_dir() {
            path.to_path_buf()
        } else {
            path.parent().and_then(|p| p.parent()).map(Path::to_path_buf).unwrap_or_default()
        };
        let want = root.join(vae.file());
        if want.exists() {
            p.video_vae = Some(want);
        }
        Ok(p)
    }

    /// [`Self::resolve`] with the transformer `variant` of a folder (none: the first present).
    pub fn resolve_with(path: &Path, variant: Option<Variant>) -> Result<Self> {
        let (root, transformer) = if path.is_dir() {
            let order = [
                files::DISTILLED_INT8,
                files::DISTILLED_NVFP4,
                files::DISTILLED_BF16,
                files::DEV_INT8,
                files::DEV_BF16,
            ];
            let pick = match variant {
                Some(v) => Some(path.join(v.file())).filter(|p| p.exists()),
                None => order.iter().map(|f| path.join(f)).find(|p| p.exists()),
            }
            .with_context(|| format!("{}: no LTX-2.5 transformer in diffusion_models/", path.display()))?;
            (path.to_path_buf(), pick)
        } else {
            let root = path.parent().and_then(|p| p.parent()).map(Path::to_path_buf).unwrap_or_default();
            (root, path.to_path_buf())
        };
        let first = |names: &[&str]| names.iter().map(|n| root.join(n)).find(|p| p.exists());
        Ok(Self {
            transformer,
            text_encoder: first(&[files::TEXT_INT8, files::TEXT_BF16]),
            video_vae: first(&[files::VIDEO_VAE, files::VIDEO_VAE_DIFF]),
            audio_vae: first(&[files::AUDIO_VAE]),
            spatial_upsampler: first(&[files::SPATIAL_UPSAMPLER]),
            temporal_upsampler: first(&[files::TEMPORAL_UPSAMPLER]),
            duration_head: first(&[files::DURATION_HEAD]),
            loras: Vec::new(),
            distilled_lora: None,
        })
    }
}

/// An image conditioning: the frame it shows and how strongly (1: kept as is).
#[derive(Clone)]
pub struct ImageCond {
    pub image: Image,
    pub frame: usize,
    pub strength: f32,
}

/// The noise generator.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum Noise {
    /// oxinfer's generator ([`crate::sd::ops::Rng`]).
    #[default]
    Oxinfer,
    /// PyTorch's CPU generator ([`crate::torch_rng`]), as `torch.Generator("cpu")`.
    Torch,
    /// Normals read from files: the generator seeded `s` reads `noise_{s}.f32` in this folder
    /// (little-endian float32) from its start, in the order it draws (for comparisons with the
    /// reference fed the same files).
    Files(PathBuf),
}

impl Noise {
    /// `oxinfer`, `torch` or `files:DIR`.
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "" | "oxinfer" => Self::Oxinfer,
            "torch" => Self::Torch,
            _ => match s.strip_prefix("files:") {
                Some(d) => Self::Files(PathBuf::from(d)),
                None => bail!("noise {s:?}: oxinfer, torch or files:DIR"),
            },
        })
    }
}

/// What [`Pipeline::generate`] needs.
#[derive(Clone)]
pub struct Params {
    pub prompt: String,
    pub width: usize,
    pub height: usize,
    /// Frames (rounded down to `8 k + 1`); none: the duration head's prediction, else 121.
    pub frames: Option<usize>,
    pub fps: f32,
    pub seed: u64,
    /// Half-size stage, spatial upsampler, full-size stage (the release's pipeline); else one
    /// stage at the full size. The dev transformer without the distilled LoRA always runs one
    /// stage (`ti2vid_one_stage`).
    pub two_stage: bool,
    /// The first (or only) stage's schedule; none: [`DISTILLED_SIGMAS`], or with guidance
    /// [`guidance::ltx2_sigmas`] of its steps.
    pub stage1_sigmas: Option<Vec<f32>>,
    pub stage2_sigmas: Vec<f32>,
    /// Whether the first (or only) stage is guided (`ti2vid_*`: by default with a dev
    /// transformer), and how ([`Guidance`], the release's by default).
    pub guidance_mode: GuidanceMode,
    pub guidance: Guidance,
    /// Rewrite the prompt first with this Gemma 4 instruct model (a repository id, folder or
    /// file; [`enhance::DEFAULT_ENHANCER`] is the reference's), the release's prompt enhancer
    /// ([`enhance`]); none: the prompt as it is.
    pub enhance_prompt: Option<String>,
    /// Denoise the soundtrack again in the guided pipeline's second stage (`--refine-audio`;
    /// otherwise the first stage's is kept).
    pub refine_audio: bool,
    /// Temporal rounds after the last stage (0 to 2; [`dfr`]): each doubles the frame count and
    /// rate with the temporal upsampler, refining the clip in tiles between generated keyframes.
    pub temporal_upscale: usize,
    pub images: Vec<ImageCond>,
    /// Decode the soundtrack (the audio stream is denoised either way: the model is joint).
    pub audio: bool,
    /// Ancestral noise: 1 is the distilled pipeline's sampler, 0 plain Euler (the guided
    /// pipeline's); none: the pipeline's.
    pub eta: Option<f32>,
    /// The H.264 CRF conditioning images are re-compressed at before they are encoded (the
    /// compression the model was trained against); none: the model's ([`Pipeline::image_crf`]:
    /// 18 for LTX-2.4 and later, 33 before), 0: as they are.
    pub image_crf: Option<u32>,
    pub noise: Noise,
    /// Audio to video (`a2vid_two_stage`): the soundtrack the video is made for. Its latents
    /// (the audio VAE's encoder) are frozen conditioning in both stages and the output keeps the
    /// waveform itself; without [`Self::frames`], the clip lasts as long as the audio (snapped).
    pub audio_input: Option<AudioInput>,
    /// Retake (`RetakePipeline`): a time window of a video regenerated, one stage at the video's
    /// own size and rate.
    pub retake: Option<Retake>,
    /// Text to audio (`t2a_one_stage`): the audio stream alone, no video.
    pub audio_only: bool,
    /// The sampler of the stages ([`Sampler::Res2s`]: `ti2vid_two_stages_hq`'s).
    pub sampler: Sampler,
    /// The distilled LoRA's strength in the first and the second stage (none: off in a guided
    /// first stage, else 1; the HQ pipeline's are 0.25 and 0.5).
    pub distilled_lora_strengths: Option<(f32, f32)>,
    pub cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

/// The sampler of the denoising stages.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Sampler {
    /// Euler steps, ancestral by [`Params::eta`] (the release's default pipelines).
    #[default]
    Euler,
    /// `res2s_audio_video_denoising_loop`: second-order exponential Runge-Kutta steps with SDE
    /// noise at the midpoint and the step (two transformer evaluations a step).
    Res2s,
}

impl Sampler {
    /// `euler` or `res2s`.
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s.to_ascii_lowercase().as_str() {
            "euler" | "" => Self::Euler,
            "res2s" | "res_2s" => Self::Res2s,
            _ => return None,
        })
    }
}

impl Params {
    pub fn new(prompt: &str) -> Self {
        Self {
            prompt: prompt.to_string(),
            width: 768,
            height: 512,
            frames: None,
            fps: DEFAULT_FPS,
            seed: 42,
            two_stage: true,
            stage1_sigmas: None,
            stage2_sigmas: STAGE_2_SIGMAS.to_vec(),
            guidance_mode: GuidanceMode::Auto,
            guidance: Guidance::default(),
            refine_audio: false,
            enhance_prompt: None,
            temporal_upscale: 0,
            images: Vec::new(),
            audio: true,
            eta: None,
            image_crf: None,
            noise: Noise::Oxinfer,
            audio_input: None,
            retake: None,
            audio_only: false,
            sampler: Sampler::Euler,
            distilled_lora_strengths: None,
            cancel: None,
        }
    }

    /// `ti2vid_two_stages_hq`'s settings (`LTX_2_3_HQ_PARAMS`): the res2s sampler, 15 guided
    /// steps (video CFG 3, rescale 0.45, modality 3; audio CFG 7, rescale 1, modality 3; no STG)
    /// and the distilled LoRA at 0.25 then 0.5.
    pub fn hq(mut self) -> Self {
        self.sampler = Sampler::Res2s;
        self.guidance.steps = 15;
        for (g, cfg, rescale) in [(&mut self.guidance.video, 3.0, 0.45), (&mut self.guidance.audio, 7.0, 1.0)] {
            g.cfg = cfg;
            g.stg = 0.0;
            g.stg_blocks = dit::Blocks::None;
            g.rescale = rescale;
            g.modality = 3.0;
            g.skip_step = 0;
        }
        self.distilled_lora_strengths = Some((0.25, 0.5));
        self
    }

    pub fn cancelled(&self) -> bool {
        self.cancel.as_ref().is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed))
    }
}

/// `snap_frames_to_grid`: `frames` rounded down to `8 k + 1`.
pub fn snap_frames(frames: usize) -> usize {
    (frames.max(1) - 1) / TIME_SCALE * TIME_SCALE + 1
}

/// Audio latents for a clip (`AudioLatentShape.from_video_pixel_shape`).
pub fn audio_latents(frames: usize, fps: f32) -> usize {
    ((frames as f64 / fps as f64) * AUDIO_LATENTS_PER_SECOND).round().max(1.0) as usize
}

/// Video row positions (`get_pixel_coords` with the causal fix, time divided by the rate; the
/// middle of each patch): `[t, y, x]` per row, frame-major.
pub fn video_positions(lt: usize, lh: usize, lw: usize, fps: f32) -> Vec<f32> {
    let mut p = Vec::with_capacity(lt * lh * lw * 3);
    for f in 0..lt {
        let s = ((f * TIME_SCALE) as f32 + 1.0 - TIME_SCALE as f32).max(0.0) / fps;
        let e = (((f + 1) * TIME_SCALE) as f32 + 1.0 - TIME_SCALE as f32).max(0.0) / fps;
        let t = (s + e) / 2.0;
        for y in 0..lh {
            for x in 0..lw {
                let yy = ((y * SPACE_SCALE) as f32 + ((y + 1) * SPACE_SCALE) as f32) / 2.0;
                let xx = ((x * SPACE_SCALE) as f32 + ((x + 1) * SPACE_SCALE) as f32) / 2.0;
                p.extend_from_slice(&[t, yy, xx]);
            }
        }
    }
    p
}

/// The positions of a guide image's rows for pixel frame `frame` (`VideoConditionByKeyframeIndex`
/// with one pixel frame: time `[frame, frame + 1)` over the rate, no causal fix).
pub fn guide_positions(frame: usize, lh: usize, lw: usize, fps: f32) -> Vec<f32> {
    let t = (frame as f32 / fps + (frame + 1) as f32 / fps) / 2.0;
    let mut p = video_positions(1, lh, lw, fps);
    for r in p.chunks_mut(3) {
        r[0] = t;
    }
    p
}

/// Audio row positions in seconds (`AudioPatchifier`, causal: 4 mel frames per latent, hop 160
/// at 16 kHz; the middle of each latent's span).
pub fn audio_positions(n: usize) -> Vec<f32> {
    let at = |i: usize| ((i * 4) as f32 + 1.0 - 4.0).max(0.0) * 160.0 / 16000.0;
    (0..n).map(|i| (at(i) + at(i + 1)) / 2.0).collect()
}

/// `[c, t, h, w]` latents to rows `[t * h * w, c]` (the transformer's patch size is 1).
pub fn patchify(lat: &[f32], c: usize, t: usize, h: usize, w: usize) -> Vec<f32> {
    let n = t * h * w;
    let mut rows = vec![0f32; n * c];
    for ch in 0..c {
        for i in 0..n {
            rows[i * c + ch] = lat[ch * n + i];
        }
    }
    rows
}

/// The inverse of [`patchify`].
pub fn unpatchify(rows: &[f32], c: usize, t: usize, h: usize, w: usize) -> Vec<f32> {
    let n = t * h * w;
    let mut lat = vec![0f32; n * c];
    for ch in 0..c {
        for i in 0..n {
            lat[ch * n + i] = rows[i * c + ch];
        }
    }
    lat
}

/// A noise source drawn in the reference's order.
pub enum NoiseGen {
    Ox(crate::sd::ops::Rng),
    Torch(Box<crate::torch_rng::TorchGenerator>),
    /// A file's values and the next one to give.
    File(Vec<f32>, usize, PathBuf),
}

impl NoiseGen {
    pub fn new(kind: &Noise, seed: u64) -> Self {
        match kind {
            Noise::Oxinfer => Self::Ox(crate::sd::ops::Rng::new(seed)),
            Noise::Torch => Self::Torch(Box::new(crate::torch_rng::TorchGenerator::new(seed))),
            Noise::Files(dir) => {
                let f = dir.join(format!("noise_{seed}.f32"));
                let v = std::fs::read(&f)
                    .map(|b| b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
                    .unwrap_or_default();
                Self::File(v, 0, f)
            }
        }
    }

    pub fn normals(&mut self, n: usize) -> Vec<f32> {
        match self {
            Self::Ox(r) => r.normals(n),
            Self::Torch(g) => g.randn(n),
            Self::File(v, at, f) => {
                assert!(*at + n <= v.len(), "{}: {} values, {} drawn", f.display(), v.len(), *at + n);
                *at += n;
                v[*at - n..*at].to_vec()
            }
        }
    }
}

/// One stream's state through a denoising stage: the rows, the clean rows conditioned ones keep,
/// and each row's denoising mask.
pub struct State {
    pub x: Vec<f32>,
    pub clean: Vec<f32>,
    pub mask: Vec<f32>,
    pub width: usize,
}

impl State {
    fn zeros(n: usize, width: usize) -> Self {
        Self {
            x: vec![0f32; n * width],
            clean: vec![0f32; n * width],
            mask: vec![1f32; n],
            width,
        }
    }

    fn rows(&self) -> usize {
        self.mask.len()
    }

    /// `GaussianNoiser`: `lerp(clean, lerp(x, noise, scale), mask)` per row.
    fn noise(&mut self, g: &mut NoiseGen, scale: f32) {
        let noise = g.normals(self.x.len());
        let w = self.width;
        for r in 0..self.rows() {
            let m = self.mask[r];
            for c in 0..w {
                let i = r * w + c;
                let l = self.x[i] + (noise[i] - self.x[i]) * scale;
                self.x[i] = self.clean[i] + (l - self.clean[i]) * m;
            }
        }
    }

    /// Each row's timestep at `sigma`.
    fn timesteps(&self, sigma: f32) -> Vec<f32> {
        self.mask.iter().map(|&m| m * sigma).collect()
    }
}

/// `EulerAncestralDiffusionStep` (rectified flow) then the conditioning mask
/// (`post_process_latent`).
fn ancestral_step(s: &mut State, denoised: &[f32], sigma: f32, next: f32, eta: f32, noise: Option<&[f32]>) {
    let w = s.width;
    if next == 0.0 {
        s.x.copy_from_slice(denoised);
        return;
    }
    // Without noise (the plain Euler loop), the denoised rows are blended with the clean ones by
    // the mask first (`post_process_latent`); with it, the stepped rows after.
    let blended: Vec<f32>;
    let denoised = if noise.is_none() && s.mask.iter().any(|&m| m != 1.0) {
        blended = (0..s.x.len())
            .map(|i| {
                let m = s.mask[i / w];
                denoised[i] * m + s.clean[i] * (1.0 - m)
            })
            .collect();
        &blended[..]
    } else {
        denoised
    };
    let down = next * (1.0 + (next / sigma - 1.0) * eta);
    let r = down / sigma;
    let (an, ad) = (1.0 - next, 1.0 - down);
    let renoise = (next * next - down * down * an * an / (ad * ad)).max(0.0).sqrt();
    for row in 0..s.rows() {
        let m = s.mask[row];
        for c in 0..w {
            let i = row * w + c;
            let mut x = r * s.x[i] + (1.0 - r) * denoised[i];
            if let Some(n) = noise {
                x = (an / ad) * x + n[i] * renoise;
                x = x * m + s.clean[i] * (1.0 - m);
            }
            s.x[i] = x;
        }
    }
}

/// The prompt's context rows for both streams.
pub struct PromptContext {
    pub video: Vec<f32>,
    pub audio: Vec<f32>,
    /// The prompt's tokens (the duration head reads them).
    pub tokens: usize,
}

pub struct Pipeline {
    pub paths: Paths,
    pub dit: Dit<'static>,
    /// A dev (non-distilled) transformer: guided by default ([`Params::guidance`]).
    pub dev: bool,
    /// The distilled LoRA's switch and strength (when [`Paths::distilled_lora`] is loaded).
    distilled_switch: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    distilled_gain: std::sync::Arc<std::sync::atomic::AtomicU32>,
    vae_files: Option<&'static ModelFiles>,
}

/// How one denoising stage runs.
struct Run<'a> {
    sigmas: &'a [f32],
    /// The ancestral sampler's noise seed.
    seed: u64,
    eta: f32,
    /// The audio rows are kept as they are (conditioning, timestep 0, sigma 0).
    frozen_audio: bool,
    /// The video rows likewise (a retake of the soundtrack alone).
    frozen_video: bool,
    guidance: Option<&'a Guidance>,
    name: &'a str,
    /// The video rows' positions (none: the clip's at the run's rate, then the images' guide
    /// rows').
    pos: Option<&'a [f32]>,
    /// Generated keyframe slot rows (they get the keyframe embedding).
    slots: std::ops::Range<usize>,
    sampler: Sampler,
}

/// `get_res2s_coefficients(h, c2)`: `(a21, b1, b2)` from the phi functions of `-h c2` and `-h`.
pub fn res2s_coefficients(h: f64, c2: f64) -> (f64, f64, f64) {
    let fact = |k: u32| (1..=k).map(|x| x as f64).product::<f64>();
    let phi = |j: u32, z: f64| -> f64 {
        if z.abs() < 1e-10 {
            return 1.0 / fact(j);
        }
        let rem: f64 = (0..j).map(|k| z.powi(k as i32) / fact(k)).sum();
        (z.exp() - rem) / z.powi(j as i32)
    };
    let a21 = c2 * phi(1, -h * c2);
    let b2 = phi(2, -h) / c2;
    let b1 = phi(1, -h) - b2;
    (a21, b1, b2)
}

fn open_static(p: &Path) -> Result<&'static ModelFiles> {
    Ok(Box::leak(Box::new(
        ModelFiles::open_file(p, false).with_context(|| format!("open {}", p.display()))?,
    )))
}

/// Progress of a run: `(stage, done, total)`.
pub type Progress<'a> = &'a mut dyn FnMut(&str, usize, usize);

impl Pipeline {
    pub fn load(paths: Paths) -> Result<Self> {
        let files = open_static(&paths.transformer)?;
        let mut dit = Dit::load(files)?;
        if !paths.loras.is_empty() {
            lora::apply(&mut dit, &paths.loras)?;
        }
        let distilled_gain = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(1f32.to_bits()));
        let distilled_switch = match &paths.distilled_lora {
            Some(l) => {
                let sw = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
                lora::apply_switched(&mut dit, &[(l.clone(), 1.0)], Some(&sw), Some(&distilled_gain))?;
                dit.lora_switch = Some(sw.clone());
                dit.lora_gain = Some(distilled_gain.clone());
                Some(sw)
            }
            None => None,
        };
        let vae_files = paths.video_vae.as_deref().map(open_static).transpose()?;
        Ok(Self {
            dev: is_dev_file(&paths.transformer),
            paths,
            dit,
            distilled_switch,
            distilled_gain,
            vae_files,
        })
    }

    /// The guidance a run gets (none: the distilled pipeline): by [`Params::guidance_mode`];
    /// automatically for a dev transformer, except with the distilled LoRA on one stage.
    pub fn guidance_of(&self, p: &Params) -> Option<Guidance> {
        let on = match p.guidance_mode {
            GuidanceMode::On => true,
            GuidanceMode::Off => false,
            GuidanceMode::Auto => self.dev && !(self.distilled_switch.is_some() && !p.two_stage),
        };
        on.then(|| {
            let mut g = p.guidance.clone();
            if p.audio_input.is_some() {
                // `a2vid_two_stage`: the audio guider has the default parameters (the audio is
                // frozen conditioning).
                g.audio = GuiderParams::off();
            }
            if p.audio_only {
                // `t2a_one_stage`: no video guider, and no v2a modality guidance.
                g.video = GuiderParams::off();
                g.audio.modality = 1.0;
            }
            g
        })
    }

    /// The H.264 CRF conditioning images are re-compressed at for this model (`detect_params`
    /// on the VAE's `model_version`: 18 from LTX-2.4 on, 33 before).
    pub fn image_crf(&self) -> u32 {
        let version = self
            .vae_files
            .and_then(|f| f.metadata("model_version"))
            .map(|v| {
                v.replace('-', ".")
                    .split('.')
                    .map_while(|x| x.parse::<u32>().ok())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if version.as_slice() >= [2, 4].as_slice() { 18 } else { 33 }
    }

    /// `p`'s images re-compressed as the reference does (`load_conditioning_still`: cropped to
    /// even sides, one H.264 intra picture at the run's CRF, decoded back). oxideav's encoder has
    /// constant-QP rate control only: the picture is coded at QP `crf + 5`, the frame QP x264
    /// (`-preset veryfast`, the reference's) gives a lone intra picture at that CRF (its
    /// MB-tree offset) before adaptive quantization moves single macroblocks.
    fn recompress_images(&self, p: &Params) -> Result<Vec<ImageCond>> {
        let crf = p.image_crf.unwrap_or_else(|| self.image_crf());
        if crf == 0 || p.images.is_empty() {
            return Ok(p.images.clone());
        }
        #[cfg(feature = "media")]
        {
            p.images
                .iter()
                .map(|im| {
                    let (w, h) = (im.image.width / 2 * 2, im.image.height / 2 * 2);
                    if w < 2 || h < 2 {
                        return Ok(im.clone());
                    }
                    let mut rgb = Vec::with_capacity(w * h * 3);
                    for y in 0..h {
                        let r = &im.image.rgb[y * im.image.width * 3..];
                        rgb.extend_from_slice(&r[..w * 3]);
                    }
                    let rgb = crate::minimax_h3::encode::h264_roundtrip(&rgb, w, h, crf + 5)?;
                    Ok(ImageCond {
                        image: Image { width: w, height: h, rgb },
                        ..im.clone()
                    })
                })
                .collect()
        }
        #[cfg(not(feature = "media"))]
        {
            eprintln!("[LTX-2.5: built without the `media` feature: conditioning images are not re-compressed (CRF {crf})]");
            Ok(p.images.clone())
        }
    }

    /// The distilled LoRA at `strength` (0: off).
    fn set_distilled_lora(&self, strength: f32) {
        if let Some(s) = &self.distilled_switch {
            s.store(strength != 0.0, std::sync::atomic::Ordering::Relaxed);
            self.distilled_gain.store(strength.to_bits(), std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// The context rows of `prompt` (text encoder, then the connectors).
    pub fn encode_prompt(&self, prompt: &str) -> Result<PromptContext> {
        Ok(self.encode_prompts(&[prompt])?.remove(0))
    }

    /// The context rows of each of `prompts` (the text encoder loaded once).
    pub fn encode_prompts(&self, prompts: &[&str]) -> Result<Vec<PromptContext>> {
        let te = self
            .paths
            .text_encoder
            .as_ref()
            .context("LTX-2.5 needs its text encoder (text_encoders/gemma4-12b-with-proj-ltx-2.5-*.safetensors)")?;
        let files = ModelFiles::open_file(te, false).with_context(|| format!("open {}", te.display()))?;
        let enc = text::TextEncoder::load(&files)?;
        let feats = prompts.iter().map(|p| enc.encode(p)).collect::<Result<Vec<_>>>()?;
        drop(enc);
        Ok(feats.iter().map(|f| self.context(f)).collect())
    }

    /// The connectors over prompt features.
    pub fn context(&self, f: &text::Features) -> PromptContext {
        PromptContext {
            video: self.dit.connect(true, &f.video, text::MAX_TOKENS),
            audio: self.dit.connect(false, &f.audio, text::MAX_TOKENS),
            tokens: f.tokens,
        }
    }

    /// The clip's frame count: `p.frames` (snapped), the duration head's prediction, or 121.
    pub fn resolve_frames(&self, p: &Params, ctx: &PromptContext) -> Result<usize> {
        if let Some(f) = p.frames {
            return Ok(snap_frames(f));
        }
        if let Some(d) = &self.paths.duration_head {
            let files = ModelFiles::open_file(d, false)?;
            let head = duration::DurationHead::load(&files)?;
            // Text to audio predicts from the audio context alone.
            let secs = head.seconds(if p.audio_only { &[] } else { &ctx.video }, &ctx.audio);
            eprintln!("[LTX-2.5 duration head: {secs:.2} s]");
            return Ok(duration::frames_for(secs, p.fps));
        }
        Ok(DEFAULT_FRAMES)
    }

    fn vae_files(&self) -> Result<&'static ModelFiles> {
        self.vae_files
            .context("LTX-2.5 needs its video VAE (vae/ltx-2.5-video-vae-conv-bf16.safetensors or vae/ltx-2.5-video-vae-bf16.safetensors)")
    }

    /// Image latents `[c, 1, h / 32, w / 32]` of `img` resized and center-cropped to `w` x `h`.
    pub fn encode_image(&self, enc: &vae::VideoEncoder, img: &Image, w: usize, h: usize) -> Vec<f32> {
        let px = resize_center_crop(img, w, h);
        let mut v = vae::Vid::new(1, h, w, 3);
        for (i, &b) in px.iter().enumerate() {
            v.data[i] = b / 127.5 - 1.0;
        }
        enc.encode(&v).to_cthw()
    }

    /// One evaluation of the denoised rows of both streams at `s` (step `i` of the guidance's
    /// skip rule): one transformer pass, or the guidance passes combined. `s64`: the sigma in
    /// float64 (the res2s midpoints' timestep sinusoids are float64 in the reference).
    #[allow(clippy::too_many_arguments)]
    fn predict(
        &self,
        ctx: &PromptContext,
        neg: Option<&PromptContext>,
        video: &State,
        audio: &State,
        (pos, apos, keyframe_rows): (&[f32], &[f32], usize),
        (s, s64): (f32, Option<f64>),
        i: usize,
        run: &Run,
        last: &Option<(Vec<f32>, Vec<f32>)>,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let (fv, fa) = (run.frozen_video, run.frozen_audio);
        let (vs, as_) = (if fv { 0.0 } else { s }, if fa { 0.0 } else { s });
        let vt = video.timesteps(vs);
        let at = audio.timesteps(as_);
        let t64 = |st: &State, frozen: bool| {
            s64.filter(|_| !frozen)
                .map(|g| st.mask.iter().map(|&m| m as f64 * g).collect::<Vec<f64>>())
        };
        let (vt64, at64) = (t64(video, fv), t64(audio, fa));
        // One transformer pass: both streams' denoised rows (`X0Model`: `x - t v`).
        let pass = |c: &PromptContext, ptb: &Perturb| -> (Vec<f32>, Vec<f32>) {
            let (vv, av) = self.dit.forward_with(
                &StreamIn {
                    x: &video.x,
                    t: &vt,
                    sigma: vs,
                    pos,
                    ctx: &c.video,
                    keyframe_rows,
                    slot_rows: run.slots.clone(),
                    t64: vt64.as_deref(),
                    sigma64: s64.filter(|_| !fv),
                },
                &StreamIn {
                    x: &audio.x,
                    t: &at,
                    sigma: as_,
                    pos: apos,
                    ctx: &c.audio,
                    keyframe_rows: 0,
                    slot_rows: 0..0,
                    t64: at64.as_deref(),
                    sigma64: s64.filter(|_| !fa),
                },
                ptb,
            );
            let x0 = |st: &State, v: &[f32], t: &[f32]| -> Vec<f32> {
                let w = st.width;
                (0..st.x.len()).map(|i| st.x[i] - v[i] * t[i / w]).collect()
            };
            (x0(video, &vv, &vt), x0(audio, &av, &at))
        };
        let Some(gd) = run.guidance else {
            return Ok(pass(ctx, &Perturb::default()));
        };
        let (vskip, askip) = (gd.video.skips(i), gd.audio.skips(i));
        if let Some(l) = last
            && vskip
            && askip
        {
            return Ok(l.clone());
        }
        let cond = pass(ctx, &Perturb::default());
        let un = match gd.uses_uncond() {
            true => Some(pass(
                neg.context("guidance needs the negative prompt's context")?,
                &Perturb::default(),
            )),
            false => None,
        };
        let pt = gd.perturbed().map(|q| pass(ctx, &q));
        let iso = gd.uses_isolated().then(|| {
            pass(
                ctx,
                &Perturb {
                    cross: true,
                    ..Perturb::default()
                },
            )
        });
        let vd = match (last, vskip) {
            (Some(l), true) => l.0.clone(),
            _ => gd.video.calculate(
                &cond.0,
                un.as_ref().map(|u| &u.0[..]),
                pt.as_ref().map(|u| &u.0[..]),
                iso.as_ref().map(|u| &u.0[..]),
                &video.mask,
                video.width,
            ),
        };
        let ad = match (last, askip) {
            (Some(l), true) => l.1.clone(),
            _ => gd.audio.calculate(
                &cond.1,
                un.as_ref().map(|u| &u.1[..]),
                pt.as_ref().map(|u| &u.1[..]),
                iso.as_ref().map(|u| &u.1[..]),
                &audio.mask,
                audio.width,
            ),
        };
        Ok((vd, ad))
    }

    /// Denoises one stage: the video state (`lt x lh x lw` rows, then any guide rows) and the
    /// audio state over `run.sigmas`, with the guidance passes (`neg`: the negative prompt's
    /// context) when there is guidance, by the run's sampler.
    #[allow(clippy::too_many_arguments)]
    fn denoise(
        &self,
        ctx: &PromptContext,
        neg: Option<&PromptContext>,
        video: &mut State,
        audio: &mut State,
        (lt, lh, lw): (usize, usize, usize),
        p: &Params,
        run: &Run,
        progress: Progress,
    ) -> Result<()> {
        let pos = match run.pos {
            Some(v) => v.to_vec(),
            None if video.rows() == 0 => Vec::new(),
            None => {
                let mut pos = video_positions(lt, lh, lw, p.fps);
                for im in p.images.iter().filter(|im| im.frame != 0) {
                    pos.extend(guide_positions(im.frame, lh, lw, p.fps));
                }
                pos
            }
        };
        let apos = audio_positions(audio.rows());
        let geo = (&pos[..], &apos[..], lh * lw);
        if run.sampler == Sampler::Res2s {
            return self.denoise_res2s(ctx, neg, video, audio, geo, p, run, progress);
        }
        let mut g = NoiseGen::new(&p.noise, run.seed);
        let sigmas = run.sigmas;
        let steps = sigmas.len() - 1;
        let mut last: Option<(Vec<f32>, Vec<f32>)> = None;
        for i in 0..steps {
            if p.cancelled() {
                bail!("cancelled");
            }
            let (s, next) = (sigmas[i], sigmas[i + 1]);
            let (vd, ad) = self.predict(ctx, neg, video, audio, geo, (s, None), i, run, &last)?;
            if run.guidance.is_some_and(|g| g.video.skip_step != 0 || g.audio.skip_step != 0) {
                last = Some((vd.clone(), ad.clone()));
            }
            let draw = run.eta > 0.0 && next != 0.0;
            let vn = draw.then(|| g.normals(video.x.len()));
            let an = draw.then(|| g.normals(audio.x.len()));
            if !run.frozen_video {
                ancestral_step(video, &vd, s, next, run.eta, vn.as_deref());
            }
            if !run.frozen_audio {
                ancestral_step(audio, &ad, s, next, run.eta, an.as_deref());
            }
            progress(run.name, i + 1, steps);
        }
        Ok(())
    }

    /// `res2s_audio_video_denoising_loop` (`legacy_mode`, `bongmath`, eta 0.5): per step the
    /// denoised rows at `sigma`, a midpoint at `sqrt(sigma next)` by the exponential integrator
    /// (`a21 = c2 phi_1(-h c2)`), SDE noise there (`Res2sDiffusionStep`), the anchor refined
    /// (bong iterations), the denoised rows at the midpoint, the step (`b1`, `b2`), SDE noise
    /// again; a final evaluation at 0.0011 when the schedule ends at 0. The noise comes from two
    /// generators seeded -1 (steps) and 9999 (midpoints), each draw normalized to zero mean and
    /// unit variance; the arithmetic in float64 as the reference's on the CPU.
    #[allow(clippy::too_many_arguments)]
    fn denoise_res2s(
        &self,
        ctx: &PromptContext,
        neg: Option<&PromptContext>,
        video: &mut State,
        audio: &mut State,
        geo: (&[f32], &[f32], usize),
        p: &Params,
        run: &Run,
        progress: Progress,
    ) -> Result<()> {
        let mut step_gen = NoiseGen::new(&p.noise, u64::MAX);
        let mut sub_gen = NoiseGen::new(&p.noise, 9999);
        let n_full = run.sigmas.len() - 1;
        let mut sig: Vec<f32> = run.sigmas.to_vec();
        let terminal = sig.last() == Some(&0.0);
        if terminal {
            sig.pop();
            sig.extend([0.0011, 0.0]);
        }
        let hs: Vec<f64> = sig.windows(2).map(|w| -(w[1] as f64 / w[0] as f64).ln()).collect();
        let total = 2 * n_full + terminal as usize;
        let mut done = 0;
        let blend = |st: &State, d: &mut [f32]| {
            let w = st.width;
            for (r, row) in d.chunks_mut(w).enumerate() {
                let m = st.mask[r];
                if m != 1.0 {
                    for (c, v) in row.iter_mut().enumerate() {
                        *v = *v * m + st.clean[r * w + c] * (1.0 - m);
                    }
                }
            }
        };
        // `_get_new_noise`: float64 normals normalized twice (the whole tensor, then over its
        // last two axes: the whole tensor again).
        let new_noise = |g: &mut NoiseGen, n: usize| -> Vec<f64> {
            let mut v: Vec<f64> = g.normals(n).into_iter().map(|x| x as f64).collect();
            for _ in 0..2 {
                let m = v.iter().sum::<f64>() / n as f64;
                let sd = (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n as f64 - 1.0).max(1.0)).sqrt();
                v.iter_mut().for_each(|x| *x = (*x - m) / sd);
            }
            v
        };
        // `Res2sDiffusionStep.step` from `sigma` to `next`, then `post_process_latent`.
        let sde = |st: &State, anchor: &[f64], den: &[f64], sigma: f64, next: f64, noise: &[f64], eta: f64| -> Vec<f64> {
            let up = (next * eta).min(next * 0.9999);
            let resid = (next * next - up * up).max(0.0).sqrt();
            let alpha = (1.0 - next) + resid;
            let down = resid / alpha;
            let plain = up == 0.0 || next == 0.0;
            let w = st.width;
            (0..anchor.len())
                .map(|i| {
                    let x = if plain {
                        den[i]
                    } else {
                        let eps = (anchor[i] - den[i]) / (sigma - next);
                        let d = anchor[i] - sigma * eps;
                        alpha * (d + down * eps) + up * noise[i]
                    };
                    let m = st.mask[i / w] as f64;
                    if m == 1.0 { x } else { x * m + st.clean[i] as f64 * (1.0 - m) }
                })
                .collect()
        };
        let streams = [!run.frozen_video, !run.frozen_audio];
        for i in 0..n_full {
            if p.cancelled() {
                bail!("cancelled");
            }
            let (sigma, next) = (sig[i] as f64, sig[i + 1] as f64);
            let h = hs[i];
            let (a21, b1, b2) = res2s_coefficients(h, 0.5);
            let sub = (sigma * next).sqrt();
            let (mut d1v, mut d1a) = self.predict(ctx, neg, video, audio, geo, (sig[i], None), i, run, &None)?;
            blend(video, &mut d1v);
            blend(audio, &mut d1a);
            done += 1;
            progress(run.name, done, total);
            let d1 = [d1v, d1a];
            let mut anchors: Vec<Vec<f64>> = [&*video, &*audio].iter().map(|s| s.x.iter().map(|&v| v as f64).collect()).collect();
            let mut eps1: Vec<Vec<f64>> = (0..2)
                .map(|k| d1[k].iter().zip(&anchors[k]).map(|(&d, &a)| d as f64 - a).collect())
                .collect();
            let mut mids: Vec<Vec<f64>> = (0..2)
                .map(|k| anchors[k].iter().zip(&eps1[k]).map(|(&a, &e)| a + h * a21 * e).collect())
                .collect();
            for (k, st) in [&*video, &*audio].into_iter().enumerate() {
                // A frozen stream draws its noise too (the reference steps it, then keeps it).
                let nz = new_noise(&mut sub_gen, st.x.len());
                if streams[k] {
                    mids[k] = sde(st, &anchors[k], &mids[k], sigma, sub, &nz, 0.5);
                }
            }
            if h < 0.5 && sigma > 0.03 {
                for k in 0..2 {
                    for _ in 0..100 {
                        for j in 0..anchors[k].len() {
                            anchors[k][j] = mids[k][j] - h * a21 * eps1[k][j];
                            eps1[k][j] = d1[k][j] as f64 - anchors[k][j];
                        }
                    }
                }
            }
            let mid_state = |st: &State, m: &[f64]| State {
                x: m.iter().map(|&v| v as f32).collect(),
                clean: st.clean.clone(),
                mask: st.mask.clone(),
                width: st.width,
            };
            let (mv, ma) = (mid_state(video, &mids[0]), mid_state(audio, &mids[1]));
            let (mut d2v, mut d2a) = self.predict(ctx, neg, &mv, &ma, geo, (sub as f32, Some(sub)), 0, run, &None)?;
            blend(video, &mut d2v);
            blend(audio, &mut d2a);
            done += 1;
            progress(run.name, done, total);
            let d2 = [d2v, d2a];
            for (k, st) in [&mut *video, &mut *audio].into_iter().enumerate() {
                let nz = new_noise(&mut step_gen, st.x.len());
                if !streams[k] {
                    continue;
                }
                let xn: Vec<f64> = (0..anchors[k].len())
                    .map(|j| anchors[k][j] + h * (b1 * eps1[k][j] + b2 * (d2[k][j] as f64 - anchors[k][j])))
                    .collect();
                let x = sde(st, &anchors[k], &xn, sigma, next, &nz, 0.5);
                st.x = x.iter().map(|&v| v as f32).collect();
            }
        }
        if terminal {
            let (mut dv, mut da) = self.predict(ctx, neg, video, audio, geo, (sig[n_full], None), n_full, run, &None)?;
            blend(video, &mut dv);
            blend(audio, &mut da);
            if streams[0] {
                video.x = dv;
            }
            if streams[1] {
                audio.x = da;
            }
            progress(run.name, total, total);
        }
        Ok(())
    }

    /// Text (and images) to video.
    pub fn generate(&self, p: &Params, progress: Progress) -> Result<Video> {
        let t0 = Instant::now();
        let enhanced;
        let p = match &p.enhance_prompt {
            Some(model) => {
                let first = p.images.iter().find(|im| im.frame == 0).map(|im| &im.image);
                let text = enhance::enhance(model, &p.prompt, first, None)?;
                eprintln!("[LTX-2.5 enhanced prompt: {text}]");
                enhanced = Params { prompt: text, ..p.clone() };
                &enhanced
            }
            None => p,
        };
        let guidance = self.guidance_of(p);
        let neg = guidance.as_ref().filter(|g| g.uses_uncond()).map(|g| g.negative_prompt.as_str());
        let mut ctxs = self.encode_prompts(&[&p.prompt, neg.unwrap_or_default()][..1 + neg.is_some() as usize])?;
        let neg = (ctxs.len() > 1).then(|| ctxs.pop().unwrap());
        let t_text = t0.elapsed().as_secs_f64();
        self.generate_guided(p, &ctxs[0], neg.as_ref(), t_text, progress)
    }

    /// [`Self::generate`] from context rows already made (the negative prompt's encoded here when
    /// the guidance needs it).
    pub fn generate_with(&self, p: &Params, ctx: &PromptContext, t_text: f64, progress: Progress) -> Result<Video> {
        let neg = match self.guidance_of(p).filter(|g| g.uses_uncond()) {
            Some(g) => Some(self.encode_prompt(&g.negative_prompt)?),
            None => None,
        };
        self.generate_guided(p, ctx, neg.as_ref(), t_text, progress)
    }

    /// [`Self::generate`] from the prompt's and the negative prompt's context rows.
    pub fn generate_guided(
        &self,
        p: &Params,
        ctx: &PromptContext,
        neg: Option<&PromptContext>,
        t_text: f64,
        progress: Progress,
    ) -> Result<Video> {
        let guidance = self.guidance_of(p);
        dump("ctx_video", &ctx.video);
        dump("ctx_audio", &ctx.audio);
        if p.retake.is_some() {
            return self.retake(p, ctx, neg, guidance.as_ref(), t_text, progress);
        }
        if p.audio_only {
            return self.text_to_audio(p, ctx, neg, guidance.as_ref(), t_text, progress);
        }
        // Audio to video: the clip's length from the audio when not given (`num_frames_from_audio_duration`),
        // the audio clipped to the clip, and the negative pass on the prompt's audio context (the
        // audio guider has no negative context).
        let source = p.audio_input.as_ref().map(|a| {
            let frames = p.frames.unwrap_or_else(|| ((a.duration() * p.fps as f64) as usize).clamp(1, 1024));
            (a.window(0.0, Some(snap_frames(frames) as f64 / p.fps as f64)), snap_frames(frames))
        });
        let neg_a2v;
        let neg = match (&source, neg) {
            (Some(_), Some(n)) => {
                neg_a2v = PromptContext {
                    video: n.video.clone(),
                    audio: ctx.audio.clone(),
                    tokens: n.tokens,
                };
                Some(&neg_a2v)
            }
            (_, n) => n,
        };
        // The dev transformer's second stage needs the distilled LoRA (`ti2vid_two_stages`);
        // without it, one stage at the full size (`ti2vid_one_stage`).
        let two_stage = p.two_stage && !(guidance.is_some() && self.dev && self.distilled_switch.is_none());
        if p.two_stage && !two_stage {
            eprintln!("[LTX-2.5: the dev transformer without the distilled LoRA runs one stage at the full size]");
        }
        let images = self.recompress_images(p)?;
        let p = &Params {
            two_stage,
            images,
            frames: source.as_ref().map(|s| s.1).or(p.frames),
            ..p.clone()
        };
        let div = if p.two_stage { 2 * SPACE_SCALE } else { SPACE_SCALE };
        if p.width == 0 || p.height == 0 || !p.width.is_multiple_of(div) || !p.height.is_multiple_of(div) {
            bail!(
                "LTX-2.5: width and height must be multiples of {div} ({} pipeline), got {}x{}",
                if p.two_stage { "two-stage" } else { "one-stage" },
                p.width,
                p.height
            );
        }
        let requested = self.resolve_frames(p, ctx)?;
        for im in &p.images {
            if im.frame >= requested {
                bail!("LTX-2.5: an image conditions frame {} of a {requested}-frame clip", im.frame);
            }
        }
        // Temporal rounds: the clip is made on the keyframe canvas (`resolve_canvas`) with a
        // generated keyframe slot at each segment border.
        let rounds = p.temporal_upscale.min(dfr::MAX_ROUNDS);
        if rounds > 0 && !self.dit.has_keyframe_embedding() {
            bail!("LTX-2.5: temporal upscaling needs a transformer with the keyframe embedding (generated keyframe slots)");
        }
        let (frames, keyframes_at) = if rounds > 0 {
            let (n, _, at) = dfr::resolve_canvas(requested);
            (n, at)
        } else {
            (requested, Vec::new())
        };
        // The transformer's rate (60 above 30 with the rounds' canvas; the soundtrack keeps `fps`).
        let cfps = if rounds > 0 { dfr::conditioning_fps(p.fps) } else { p.fps };
        let c = self.dit.cfg.in_channels;
        let ac = self.dit.cfg.audio_in_channels;
        let lt = (frames - 1) / TIME_SCALE + 1;
        let na = audio_latents(frames, p.fps);
        let mut rng = NoiseGen::new(&p.noise, p.seed);
        let vfiles = self.vae_files()?;
        let encoder = if p.images.is_empty() {
            None
        } else {
            Some(vae::VideoEncoder::load(vfiles)?)
        };
        let t1 = Instant::now();
        // Conditioning: an image for frame 0 replaces the first latent frame's rows
        // (`VideoConditionByLatentIndex`); one for any other frame is appended as guide rows
        // (`VideoConditionByKeyframeIndex`: clean latents at that frame's time), dropped after
        // each stage.
        let condition_with = |st: &mut State, w: usize, h: usize, images: &[ImageCond]| {
            if let Some(enc) = &encoder {
                let (lh, lw) = (h / SPACE_SCALE, w / SPACE_SCALE);
                for im in images {
                    let lat = self.encode_image(enc, &im.image, w, h);
                    let rows = patchify(&lat, c, 1, lh, lw);
                    if im.frame == 0 {
                        st.clean[..lh * lw * c].copy_from_slice(&rows);
                        st.mask[..lh * lw].fill(1.0 - im.strength);
                    } else {
                        st.x.extend(std::iter::repeat_n(0.0, rows.len()));
                        st.clean.extend_from_slice(&rows);
                        st.mask.extend(std::iter::repeat_n(1.0 - im.strength, lh * lw));
                    }
                }
            }
        };
        let condition = |st: &mut State, w: usize, h: usize| condition_with(st, w, h, &p.images);
        // Generated keyframe slots (`VideoGeneratedKeyframeSlots`): a latent frame of rows per
        // keyframe at the end, denoised with the clip from `init` (or zeros).
        let add_slots = |st: &mut State, n: usize, frame_rows: usize, init: Option<&[f32]>| -> std::ops::Range<usize> {
            let start = st.rows();
            let rows = n * frame_rows;
            match init {
                Some(v) => st.x.extend_from_slice(&v[..rows * c]),
                None => st.x.extend(std::iter::repeat_n(0.0, rows * c)),
            }
            st.clean.extend(std::iter::repeat_n(0.0, rows * c));
            st.mask.extend(std::iter::repeat_n(1.0, rows));
            start..start + rows
        };
        // The video rows' positions with slots: the clip's, the images' guide rows', the slots'.
        let positions = |lh: usize, lw: usize| -> Vec<f32> {
            let mut pos = video_positions(lt, lh, lw, cfps);
            for im in p.images.iter().filter(|im| im.frame != 0) {
                pos.extend(guide_positions(im.frame, lh, lw, cfps));
            }
            for &k in &keyframes_at {
                pos.extend(guide_positions(k, lh, lw, cfps));
            }
            pos
        };
        let slot_rows = |st: &State, r: &std::ops::Range<usize>| st.x[r.start * c..r.end * c].to_vec();
        // The rows of the clip itself (guide rows dropped).
        let clip = |st: &mut State, rows: usize| {
            st.x.truncate(rows * c);
            st.clean.truncate(rows * c);
            st.mask.truncate(rows);
        };
        let (w1, h1) = if p.two_stage {
            (p.width / 2, p.height / 2)
        } else {
            (p.width, p.height)
        };
        let (lh, lw) = (h1 / SPACE_SCALE, w1 / SPACE_SCALE);
        let mut video = State::zeros(lt * lh * lw, c);
        condition(&mut video, w1, h1);
        let slots1 = if rounds > 0 {
            add_slots(&mut video, keyframes_at.len(), lh * lw, None)
        } else {
            0..0
        };
        let pos1 = (rounds > 0).then(|| positions(lh, lw));
        video.noise(&mut rng, 1.0);
        let mut audio = State::zeros(na, ac);
        // Audio to video: the soundtrack's latents, frozen (noised at scale 0, which still draws).
        let source_rows = match &source {
            Some((a, _)) => {
                let af = ModelFiles::open_file(self.paths.audio_vae.as_ref().context("audio to video needs the audio VAE")?, false)?;
                let enc = audio_enc::AudioEncoder::load(&af)?;
                let rows = enc.encode(&a.channels, a.rate)?;
                Some(audio_enc::conform_rows(rows, ac, na))
            }
            None => None,
        };
        if let Some(r) = &source_rows {
            dump("source_audio", r);
            audio.x = r.clone();
            audio.noise(&mut rng, 0.0);
            audio.clean = audio.x.clone();
            audio.mask.fill(0.0);
        } else {
            audio.noise(&mut rng, 1.0);
        }
        let guided = guidance.is_some();
        let sig1 = match (&p.stage1_sigmas, &guidance) {
            (Some(s), _) => s.clone(),
            // The HQ pipeline shifts its schedule for the first stage's latent tokens.
            (None, Some(g)) if p.sampler == Sampler::Res2s => guidance::ltx2_sigmas_for(g.steps, lt * lh * lw),
            (None, Some(g)) => guidance::ltx2_sigmas(g.steps),
            (None, None) => DISTILLED_SIGMAS.to_vec(),
        };
        let eta = p.eta.unwrap_or(if guided { 0.0 } else { 1.0 });
        if let Some(g) = &guidance {
            eprintln!(
                "[LTX-2.5 guidance: {} steps, {} passes each (video cfg {} stg {} rescale {} modality {}; audio cfg {} stg {} rescale {} modality {})]",
                sig1.len() - 1,
                g.passes(),
                g.video.cfg,
                g.video.stg,
                g.video.rescale,
                g.video.modality,
                g.audio.cfg,
                g.audio.stg,
                g.audio.rescale,
                g.audio.modality
            );
        }
        // The distilled LoRA is off while the guided first stage runs (the HQ pipeline's runs at
        // its own strengths).
        let (lora1, lora2) = p.distilled_lora_strengths.unwrap_or((if guided { 0.0 } else { 1.0 }, 1.0));
        self.set_distilled_lora(lora1);
        let r = self.denoise(
            ctx,
            neg,
            &mut video,
            &mut audio,
            (lt, lh, lw),
            p,
            &Run {
                sigmas: &sig1,
                seed: p.seed + STAGE_1_NOISE_OFFSET,
                eta,
                frozen_audio: source_rows.is_some(),
                frozen_video: false,
                guidance: guidance.as_ref(),
                name: "stage 1",
                pos: pos1.as_deref(),
                slots: slots1.clone(),
                sampler: p.sampler,
            },
            progress,
        );
        self.set_distilled_lora(lora2);
        r?;
        if let Some(r) = &source_rows {
            audio.x = r.clone();
        }
        let mut planes = slot_rows(&video, &slots1);
        let stage1_audio = audio.x.clone();
        clip(&mut video, lt * lh * lw);
        dump("stage1_video", &video.x);
        dump("stage1_audio", &audio.x);
        let (mut lh, mut lw) = (lh, lw);
        let t_s1 = t1.elapsed().as_secs_f64();
        let mut t_s2 = 0.0;
        let vfiles_stats = vae::stats_of(vfiles)?;
        if p.two_stage {
            let up = self
                .paths
                .spatial_upsampler
                .as_ref()
                .context("the two-stage pipeline needs the spatial upsampler (latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors)")?;
            let t2 = Instant::now();
            let uf = ModelFiles::open_file(up, false)?;
            let ups = upsample::LatentUpsampler::load(&uf)?;
            let x = upsample_rows(&ups, &vfiles_stats, &video.x, c, (lt, lh, lw));
            // The slots' rows upsampled as a clip of their own (`upsampler(stacked_keyframes)`).
            let up_planes = (rounds > 0).then(|| upsample_rows(&ups, &vfiles_stats, &planes, c, (keyframes_at.len(), lh, lw)));
            drop(ups);
            dump("upsampled_video", &x);
            lh *= 2;
            lw *= 2;
            let mut v2 = State::zeros(lt * lh * lw, c);
            v2.x = x;
            condition(&mut v2, p.width, p.height);
            let slots2 = match &up_planes {
                Some(u) => add_slots(&mut v2, keyframes_at.len(), lh * lw, Some(u)),
                None => 0..0,
            };
            let pos2 = (rounds > 0).then(|| positions(lh, lw));
            let s0 = p.stage2_sigmas[0];
            v2.noise(&mut rng, s0);
            let mut a2 = State::zeros(na, ac);
            a2.x = audio.x.clone();
            // The guided pipeline keeps the first stage's soundtrack (frozen: noised at scale 0,
            // which still draws its noise, then timestep 0) unless asked to refine it; the HQ
            // pipeline denoises it again but keeps the first stage's too; audio to video keeps the
            // source's.
            let keep = source_rows.is_some() || (guided && !p.refine_audio);
            let frozen = keep && (source_rows.is_some() || p.sampler != Sampler::Res2s);
            a2.noise(&mut rng, if frozen { 0.0 } else { s0 });
            if frozen {
                a2.clean = a2.x.clone();
                a2.mask.fill(0.0);
            }
            self.denoise(
                ctx,
                None,
                &mut v2,
                &mut a2,
                (lt, lh, lw),
                p,
                &Run {
                    sigmas: &p.stage2_sigmas,
                    seed: p.seed + STAGE_2_NOISE_OFFSET,
                    eta,
                    frozen_audio: frozen,
                    frozen_video: false,
                    guidance: None,
                    name: "stage 2",
                    pos: pos2.as_deref(),
                    slots: slots2.clone(),
                    sampler: p.sampler,
                },
                progress,
            )?;
            if keep {
                a2.x = audio.x.clone();
            }
            if rounds > 0 {
                planes = slot_rows(&v2, &slots2);
            }
            clip(&mut v2, lt * lh * lw);
            dump("stage2_video", &v2.x);
            dump("stage2_audio", &a2.x);
            video = v2;
            audio = a2;
            t_s2 = t2.elapsed().as_secs_f64();
        }
        let (mut lt, mut fps, mut frames) = (lt, p.fps, frames);
        if rounds > 0 {
            let up = self
                .paths
                .temporal_upsampler
                .as_ref()
                .context("temporal upscaling needs the temporal upsampler (latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors)")?;
            let uf = ModelFiles::open_file(up, false)?;
            let ups = upsample::LatentUpsampler::load(&uf)?;
            let fr = lh * lw * c;
            let mut keyframes: std::collections::BTreeMap<usize, Vec<f32>> =
                keyframes_at.iter().zip(planes.chunks(fr)).map(|(&k, v)| (k, v.to_vec())).collect();
            // The stage-1 soundtrack, frozen, for the tiles' cross-attention (`audio_latent_for_tile`).
            let source_duration = frames as f32 / p.fps;
            let full_t = stage1_audio.len() / ac;
            for r in 1..=rounds {
                let x = upsample_rows(&ups, &vfiles_stats, &video.x, c, (lt, lh, lw));
                lt = 2 * lt - 1;
                frames = 2 * (frames - 1) + 1;
                fps *= 2.0;
                let rfps = dfr::conditioning_fps(fps);
                let scaled: std::collections::BTreeMap<usize, Vec<f32>> = keyframes.iter().map(|(k, v)| (2 * k, v.clone())).collect();
                let seams: Vec<usize> = scaled.keys().copied().collect();
                let tiles = dfr::tile_plan(&seams, frames, 1 << r);
                let mut plane_at = scaled.clone();
                let mut stitched: Vec<f32> = Vec::with_capacity(lt * fr);
                let mut previous: Option<(usize, Vec<f32>)> = None;
                for (ti, tile) in tiles.iter().enumerate() {
                    let prefix = match ti {
                        0 => None,
                        _ => Some(dfr::tile_prefix((tile.start - 1) * TIME_SCALE, plane_at.keys().copied())?),
                    };
                    // The tile's frames: the canvas', or the plane before its border then the
                    // canvas from there (`_temporal_tile_video_input`).
                    let (tile_x, pixel_start, resume, pinned) = match prefix {
                        None => (x[tile.start * fr..tile.end * fr].to_vec(), 0, 0, 0),
                        Some(pf) => {
                            let mut v = plane_at[&pf.keyframe].clone();
                            v.extend_from_slice(&x[pf.video_start * fr..tile.end * fr]);
                            (v, pf.keyframe, pf.resume, pf.cells)
                        }
                    };
                    let tlt = tile_x.len() / fr;
                    let local_frames = (tlt - 1) * TIME_SCALE + 1;
                    let pixel_end = pixel_start + local_frames - 1;
                    let mut st = State::zeros(tlt * lh * lw, c);
                    st.x = tile_x.clone();
                    let mut pos = video_positions(tlt, lh, lw, rfps);
                    // The images that fall in the tile (`rebase_image_conditionings`).
                    let imgs: Vec<ImageCond> = p
                        .images
                        .iter()
                        .filter_map(|im| {
                            let f = im.frame << r;
                            (pixel_start <= f && f <= pixel_end).then(|| ImageCond {
                                frame: f - pixel_start,
                                ..im.clone()
                            })
                        })
                        .collect();
                    condition_with(&mut st, lw * SPACE_SCALE, lh * SPACE_SCALE, &imgs);
                    for im in imgs.iter().filter(|im| im.frame != 0) {
                        pos.extend(guide_positions(im.frame, lh, lw, rfps));
                    }
                    // The keyframes at its borders, as guides (strength 0.95).
                    for &a in tile.anchors.iter().filter(|&&a| a >= resume) {
                        st.x.extend(std::iter::repeat_n(0.0, fr));
                        st.clean.extend_from_slice(&scaled[&a]);
                        st.mask.extend(std::iter::repeat_n(1.0 - dfr::ANCHOR_STRENGTH, lh * lw));
                        pos.extend(guide_positions(a - pixel_start, lh, lw, rfps));
                    }
                    // New slots mid-segment, started from the nearest latent frame
                    // (`slot_initials_from_video`, Python's round half to even).
                    let local: Vec<usize> = tile.slots.iter().map(|&s| s - pixel_start).collect();
                    let slots = if local.is_empty() {
                        0..0
                    } else {
                        let mut init = Vec::with_capacity(local.len() * fr);
                        for &s in &local {
                            let i = ((s as f64 / TIME_SCALE as f64).round_ties_even() as usize).min(tlt - 1);
                            init.extend_from_slice(&tile_x[i * fr..(i + 1) * fr]);
                            pos.extend(guide_positions(s, lh, lw, rfps));
                        }
                        add_slots(&mut st, local.len(), lh * lw, Some(&init))
                    };
                    // The pinned prefix: the plane, then the previous tile's frames up to the
                    // border (`lead_in_carryover`, strength 1).
                    if let (Some(pf), Some((base, prev))) = (prefix, &previous) {
                        let offset = pf.video_start - base;
                        let cells = pf.cells - 1;
                        if offset + cells > prev.len() / fr {
                            bail!("temporal round {r}: tile {ti}'s pinned frames fall outside the previous tile");
                        }
                        st.clean[..fr].copy_from_slice(&plane_at[&pf.keyframe]);
                        st.clean[fr..pf.cells * fr].copy_from_slice(&prev[offset * fr..(offset + cells) * fr]);
                        st.mask[..pf.cells * lh * lw].fill(0.0);
                    }
                    st.noise(&mut rng, dfr::TEMPORAL_SIGMAS[0]);
                    // The soundtrack of the tile's window, frozen (noised at scale 0: drawn).
                    let src0 = pixel_start as f32 / fps / source_duration * full_t as f32;
                    let src1 = (pixel_start + local_frames) as f32 / fps / source_duration * full_t as f32;
                    let mut at = State::zeros(audio_latents(local_frames, rfps), ac);
                    at.x = dfr::resample_rows(&stage1_audio, ac, src0, src1, at.rows());
                    at.noise(&mut rng, 0.0);
                    at.clean = at.x.clone();
                    at.mask.fill(0.0);
                    let pt = Params {
                        fps: rfps,
                        images: imgs.clone(),
                        ..p.clone()
                    };
                    self.denoise(
                        ctx,
                        None,
                        &mut st,
                        &mut at,
                        (tlt, lh, lw),
                        &pt,
                        &Run {
                            sigmas: &dfr::TEMPORAL_SIGMAS,
                            seed: p.seed + 1000 * r as u64 + ti as u64,
                            eta: p.eta.unwrap_or(dfr::TEMPORAL_ETA),
                            frozen_audio: true,
                            frozen_video: false,
                            guidance: None,
                            name: "temporal",
                            pos: Some(&pos),
                            slots: slots.clone(),
                            sampler: Sampler::Euler,
                        },
                        progress,
                    )?;
                    for (k, &s) in tile.slots.iter().enumerate() {
                        let at_row = (slots.start + k * lh * lw) * c;
                        plane_at.entry(s).or_insert_with(|| st.x[at_row..at_row + fr].to_vec());
                    }
                    let tile_lat = st.x[..tlt * fr].to_vec();
                    stitched.extend_from_slice(&tile_lat[pinned * fr..]);
                    previous = Some((prefix.map_or(tile.start, |pf| pf.video_start - 1), tile_lat));
                }
                if stitched.len() != lt * fr {
                    bail!("temporal round {r}: stitched {} latent frames, expected {lt}", stitched.len() / fr);
                }
                video.x = stitched;
                keyframes = plane_at;
                dump(&format!("temporal{r}_video"), &video.x);
            }
            // The canvas' padding cut: `(frames - 1) 2^rounds + 1` frames.
            let target = (requested - 1) * (1 << rounds) + 1;
            let keep = (target - 1) / TIME_SCALE + 1;
            video.x.truncate(keep * fr);
            lt = keep;
            frames = target;
        }
        self.finish(
            p,
            &video.x,
            &audio.x,
            (lt, lh, lw),
            frames,
            fps,
            &mut rng,
            source.as_ref().map(|s| &s.0),
            [t_text, t_s1, t_s2],
            progress,
        )
    }

    /// The decoders: the video rows `[lt lh lw, c]` to frames (none: no video), the audio rows to
    /// the soundtrack, or `soundtrack` itself (audio to video: the source's waveform cut to the
    /// clip).
    #[allow(clippy::too_many_arguments)]
    fn finish(
        &self,
        p: &Params,
        video_rows: &[f32],
        audio_rows: &[f32],
        (lt, lh, lw): (usize, usize, usize),
        frames: usize,
        fps: f32,
        rng: &mut NoiseGen,
        soundtrack: Option<&AudioInput>,
        [t_text, t_s1, t_s2]: [f64; 3],
        progress: Progress,
    ) -> Result<Video> {
        let c = self.dit.cfg.in_channels;
        let t3 = Instant::now();
        let (frames_rgb, w, h, n_frames) = if lt == 0 || video_rows.is_empty() {
            (Vec::new(), 0, 0, frames)
        } else {
            let vfiles = self.vae_files()?;
            let lat = unpatchify(video_rows, c, lt, lh, lw);
            let mut on_vae = |d: usize, t: usize| progress("vae", d, t);
            let px = if diffvae::is_diffusion_vae(vfiles) {
                // The diffusion decoder's starting pixels continue the run's noise stream, as the
                // reference's decoder continues its generator.
                let mut dec = diffvae::DiffDecoder::load(vfiles)?;
                dec.device = self.dit.device;
                dec.decode(&lat, lt, lh, lw, &mut |n| rng.normals(n), &mut on_vae)
            } else {
                vae::VideoDecoder::load(vfiles)?.decode(&lat, lt, lh, lw, &mut on_vae)
            };
            dump("pixels", &px.data);
            let n_frames = px.t.min(frames);
            let (w, h) = (px.w, px.h);
            let frames_rgb: Vec<Vec<u8>> = (0..n_frames)
                .map(|f| {
                    px.data[f * w * h * 3..(f + 1) * w * h * 3]
                        .iter()
                        .map(|&v| (((v + 1.0) / 2.0).clamp(0.0, 1.0) * 255.0).round() as u8)
                        .collect()
                })
                .collect();
            (frames_rgb, w, h, n_frames)
        };
        let t_vae = t3.elapsed().as_secs_f64();
        let t4 = Instant::now();
        let soundtrack = match (soundtrack, &self.paths.audio_vae, p.audio) {
            (Some(a), _, true) => {
                // `replace_chunks_audio`: the source's samples for the clip's length.
                let keep = ((n_frames as f64 / fps as f64) * a.rate as f64).round() as usize;
                let two = [&a.channels[0], a.channels.get(1).unwrap_or(&a.channels[0])];
                Some(Soundtrack {
                    sample_rate: a.rate,
                    channels: two.iter().map(|c| c[..keep.min(c.len())].to_vec()).collect(),
                })
            }
            (None, Some(path), true) => {
                let af = ModelFiles::open_file(path, false)?;
                let dec = audio::AudioDecoder::load(&af)?;
                let mel = dec.decode(audio_rows)?;
                let t_mel = mel.len() / (2 * dec.mel_bins);
                let voc = audio::Vocoder::load(&af)?;
                let mut ch = voc.forward(&mel, t_mel);
                dump("mel", &mel);
                dump("waveform", &ch.concat());
                // Cut to the video's length (audio alone: as decoded, as the reference).
                if !frames_rgb.is_empty() {
                    let keep = ((n_frames as f64 / fps as f64) * voc.output_rate as f64).round() as usize;
                    for c in ch.iter_mut() {
                        c.truncate(keep);
                    }
                }
                Some(Soundtrack {
                    sample_rate: voc.output_rate,
                    channels: ch,
                })
            }
            _ => None,
        };
        eprintln!(
            "[LTX-2.5: {w}x{h}, {} frames at {fps} fps, text {t_text:.1}s, stage 1 {t_s1:.1}s, stage 2 {t_s2:.1}s, VAE {t_vae:.1}s, audio {:.1}s]",
            frames_rgb.len(),
            t4.elapsed().as_secs_f64()
        );
        Ok(Video {
            width: w,
            height: h,
            frames: frames_rgb,
            fps: fps.round() as usize,
            audio: soundtrack,
        })
    }
}

/// Writes `v` as `name.f32` (little-endian float32) into the folder `OXINFER_LTX_DUMP` names, if
/// any: a run's intermediate states, for comparisons with the reference.
fn dump(name: &str, v: &[f32]) {
    if let Some(d) = std::env::var_os("OXINFER_LTX_DUMP") {
        let d = PathBuf::from(d);
        let _ = std::fs::create_dir_all(&d);
        let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        if let Err(e) = std::fs::write(d.join(format!("{name}.f32")), b) {
            eprintln!("[OXINFER_LTX_DUMP: {name}: {e}]");
        }
    }
}

/// The latent frame a pixel frame falls in (frame 0 alone in the first).
pub fn latent_index(frame: usize) -> usize {
    if frame == 0 { 0 } else { (frame - 1) / TIME_SCALE + 1 }
}

/// `upsample_video`: rows un-normalized by the VAE's statistics, upsampled, normalized again.
fn upsample_rows(
    ups: &upsample::LatentUpsampler,
    (mean, std): &(Vec<f32>, Vec<f32>),
    rows: &[f32],
    c: usize,
    (lt, lh, lw): (usize, usize, usize),
) -> Vec<f32> {
    let mut v = vae::Vid::new(lt, lh, lw, c);
    for (o, r) in v.data.chunks_mut(c).zip(rows.chunks(c)) {
        for ch in 0..c {
            o[ch] = r[ch] * std[ch] + mean[ch];
        }
    }
    let y = ups.forward(&v);
    let mut out = y.data;
    for r in out.chunks_mut(c) {
        for ch in 0..c {
            r[ch] = (r[ch] - mean[ch]) / std[ch];
        }
    }
    out
}

/// `resize_and_center_crop`: scaled (bilinear, `align_corners=False`) to cover `w` x `h`, then
/// center-cropped; `[h, w, 3]` floats in 0..255.
pub fn resize_center_crop(img: &Image, w: usize, h: usize) -> Vec<f32> {
    let (sw, sh) = (img.width, img.height);
    let scale = (h as f64 / sh as f64).max(w as f64 / sw as f64);
    let nh = (sh as f64 * scale).ceil() as usize;
    let nw = (sw as f64 * scale).ceil() as usize;
    let (top, left) = ((nh - h) / 2, (nw - w) / 2);
    let tap = |o: usize, n_out: usize, n_in: usize| -> (usize, usize, f32) {
        let s = n_in as f64 / n_out as f64;
        let x = ((o as f64 + 0.5) * s - 0.5).max(0.0);
        let i0 = (x.floor() as usize).min(n_in - 1);
        let i1 = (i0 + 1).min(n_in - 1);
        (i0, i1, (x - i0 as f64) as f32)
    };
    let mut out = vec![0f32; w * h * 3];
    for y in 0..h {
        let (y0, y1, fy) = tap(y + top, nh, sh);
        for x in 0..w {
            let (x0, x1, fx) = tap(x + left, nw, sw);
            for c in 0..3 {
                let p = |yy: usize, xx: usize| img.rgb[(yy * sw + xx) * 3 + c] as f32;
                let top_v = p(y0, x0) * (1.0 - fx) + p(y0, x1) * fx;
                let bot_v = p(y1, x0) * (1.0 - fx) + p(y1, x1) * fx;
                out[(y * w + x) * 3 + c] = top_v * (1.0 - fy) + bot_v * fy;
            }
        }
    }
    out
}

impl Pipeline {
    /// `RetakePipeline`: the source video encoded whole ([`vae::VideoEncoder::encode_tiled`]),
    /// its soundtrack by the audio encoder, the rows inside the retake window noised and
    /// regenerated (`TemporalRegionMask`: a latent frame or audio latent whose span meets the
    /// window), the rest kept; one stage at the source's size and rate with plain Euler steps on
    /// the distilled schedule (guided with the dev transformer). A stream not regenerated is
    /// frozen whole (noised at scale 0: the reference noises it at scale 1 before freezing it); a
    /// source without a soundtrack gets a new one.
    fn retake(
        &self,
        p: &Params,
        ctx: &PromptContext,
        neg: Option<&PromptContext>,
        guidance: Option<&Guidance>,
        t_text: f64,
        progress: Progress,
    ) -> Result<Video> {
        let rt = p.retake.as_ref().context("no retake")?;
        let src = &rt.source;
        if rt.start.partial_cmp(&rt.end) != Some(std::cmp::Ordering::Less) {
            bail!("retake: the start ({}) must come before the end ({})", rt.start, rt.end);
        }
        if src.frames.is_empty() || src.fps <= 0.0 {
            bail!("retake: the source video has no frames");
        }
        let fps = src.fps;
        let frames = snap_frames(src.frames.len());
        let (w, h) = (src.width / SPACE_SCALE * SPACE_SCALE, src.height / SPACE_SCALE * SPACE_SCALE);
        if w == 0 || h == 0 {
            bail!("retake: the source video ({}x{}) is smaller than 32x32", src.width, src.height);
        }
        if (w, h) != (src.width, src.height) || frames != src.frames.len() {
            eprintln!(
                "[LTX-2.5 retake: the source ({}x{}, {} frames) cropped to {w}x{h}, {frames} frames]",
                src.width,
                src.height,
                src.frames.len()
            );
        }
        let p = &Params { fps, ..p.clone() };
        let t1 = Instant::now();
        let c = self.dit.cfg.in_channels;
        let ac = self.dit.cfg.audio_in_channels;
        let (lt, lh, lw) = ((frames - 1) / TIME_SCALE + 1, h / SPACE_SCALE, w / SPACE_SCALE);
        let na = audio_latents(frames, fps);
        // `video_latent_from_file`: the frames resized and center-cropped, in [-1, 1], encoded.
        let vfiles = self.vae_files()?;
        let enc = vae::VideoEncoder::load(vfiles)?;
        let mut v = vae::Vid::new(frames, h, w, 3);
        for (f, rgb) in src.frames.iter().take(frames).enumerate() {
            let img = Image {
                width: src.width,
                height: src.height,
                rgb: rgb.clone(),
            };
            let px = resize_center_crop(&img, w, h);
            for (o, &b) in v.data[f * w * h * 3..(f + 1) * w * h * 3].iter_mut().zip(&px) {
                *o = b / 127.5 - 1.0;
            }
        }
        let z = enc.encode_tiled(&v, &mut |d, t| progress("encode", d, t));
        drop((enc, v));
        let in_window = |t0: f32, t1: f32| inputs::in_region(t0, t1, rt.start, rt.end);
        let mut video = State::zeros(lt * lh * lw, c);
        video.x = z.data.clone();
        video.clean = z.data;
        for f in 0..lt {
            let px = |k: usize| ((k * TIME_SCALE) as f32 + 1.0 - TIME_SCALE as f32).max(0.0);
            let m = (rt.video && in_window(px(f) / fps, px(f + 1) / fps)) as u8 as f32;
            video.mask[f * lh * lw..(f + 1) * lh * lw].fill(m);
        }
        dump("source_video", &video.clean);
        // The source's soundtrack (`audio_latent_from_file`), else a new one from scratch.
        let source_audio = match (&src.audio, &self.paths.audio_vae) {
            (Some(a), Some(path)) => {
                let a = a.window(0.0, Some(frames as f64 / fps as f64));
                let af = ModelFiles::open_file(path, false)?;
                let rows = audio_enc::AudioEncoder::load(&af)?.encode(&a.channels, a.rate)?;
                Some(audio_enc::conform_rows(rows, ac, na))
            }
            _ => None,
        };
        let mut audio = State::zeros(na, ac);
        let frozen_audio = source_audio.is_some() && !rt.audio;
        if let Some(r) = &source_audio {
            dump("source_audio", r);
            audio.x = r.clone();
            audio.clean = r.clone();
            let at = |i: usize| ((i * 4) as f32 + 1.0 - 4.0).max(0.0) * 160.0 / 16000.0;
            for (i, m) in audio.mask.iter_mut().enumerate() {
                *m = (rt.audio && in_window(at(i), at(i + 1))) as u8 as f32;
            }
        }
        let frozen_video = !rt.video;
        let sig = match (&p.stage1_sigmas, guidance) {
            (Some(s), _) => s.clone(),
            (None, Some(g)) => guidance::ltx2_sigmas(g.steps),
            (None, None) => DISTILLED_SIGMAS.to_vec(),
        };
        let mut rng = NoiseGen::new(&p.noise, p.seed);
        video.noise(&mut rng, if frozen_video { 0.0 } else { sig[0] });
        audio.noise(&mut rng, if frozen_audio { 0.0 } else { sig[0] });
        let lora = p
            .distilled_lora_strengths
            .map_or(if guidance.is_some() { 0.0 } else { 1.0 }, |s| s.0);
        self.set_distilled_lora(lora);
        let r = self.denoise(
            ctx,
            neg,
            &mut video,
            &mut audio,
            (lt, lh, lw),
            p,
            &Run {
                sigmas: &sig,
                seed: p.seed + STAGE_1_NOISE_OFFSET,
                eta: p.eta.unwrap_or(0.0),
                frozen_audio,
                frozen_video,
                guidance,
                name: "retake",
                pos: None,
                slots: 0..0,
                sampler: p.sampler,
            },
            progress,
        );
        self.set_distilled_lora(1.0);
        r?;
        if frozen_video {
            video.x = video.clean.clone();
        }
        if let (true, Some(r)) = (frozen_audio, &source_audio) {
            audio.x = r.clone();
        }
        dump("stage1_video", &video.x);
        dump("stage1_audio", &audio.x);
        let t_s1 = t1.elapsed().as_secs_f64();
        self.finish(
            p,
            &video.x,
            &audio.x,
            (lt, lh, lw),
            frames,
            fps,
            &mut rng,
            None,
            [t_text, t_s1, 0.0],
            progress,
        )
    }

    /// `T2AOneStagePipeline`: the audio stream alone (no video rows, so no audio-video
    /// cross-attention), guided on the audio with CFG, STG and rescale (no modality guidance)
    /// with the dev transformer, on `LTX2Scheduler`'s schedule with Euler steps; the length from
    /// [`Params::frames`] or the duration head on the audio context alone. The result has no
    /// frames, and the whole decoded soundtrack.
    fn text_to_audio(
        &self,
        p: &Params,
        ctx: &PromptContext,
        neg: Option<&PromptContext>,
        guidance: Option<&Guidance>,
        t_text: f64,
        progress: Progress,
    ) -> Result<Video> {
        let frames = self.resolve_frames(p, ctx)?;
        let t1 = Instant::now();
        let (c, ac) = (self.dit.cfg.in_channels, self.dit.cfg.audio_in_channels);
        let na = audio_latents(frames, p.fps);
        let mut rng = NoiseGen::new(&p.noise, p.seed);
        let mut video = State::zeros(0, c);
        let mut audio = State::zeros(na, ac);
        audio.noise(&mut rng, 1.0);
        let sig = match (&p.stage1_sigmas, guidance) {
            (Some(s), _) => s.clone(),
            (None, Some(g)) => guidance::ltx2_sigmas(g.steps),
            (None, None) => DISTILLED_SIGMAS.to_vec(),
        };
        let lora = p
            .distilled_lora_strengths
            .map_or(if guidance.is_some() { 0.0 } else { 1.0 }, |s| s.0);
        self.set_distilled_lora(lora);
        let r = self.denoise(
            ctx,
            neg,
            &mut video,
            &mut audio,
            (0, 0, 0),
            p,
            &Run {
                sigmas: &sig,
                seed: p.seed + STAGE_1_NOISE_OFFSET,
                eta: p.eta.unwrap_or(if guidance.is_some() { 0.0 } else { 1.0 }),
                frozen_audio: false,
                frozen_video: false,
                guidance,
                name: "stage 1",
                pos: None,
                slots: 0..0,
                sampler: p.sampler,
            },
            progress,
        );
        self.set_distilled_lora(1.0);
        r?;
        dump("stage1_audio", &audio.x);
        let t_s1 = t1.elapsed().as_secs_f64();
        self.finish(
            p,
            &[],
            &audio.x,
            (0, 0, 0),
            frames,
            p.fps,
            &mut rng,
            None,
            [t_text, t_s1, 0.0],
            progress,
        )
    }
}
