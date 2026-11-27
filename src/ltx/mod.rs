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
//! skips the upsampler. The temporal x2 upsampler doubles the frame count and rate after the
//! last stage and refines the result with `DISTILLED_SIGMAS[4..]` (the audio kept as it was): a
//! simplified form of the release's DFR temporal rounds.

pub mod api;
pub mod audio;
pub(crate) mod device;
pub mod diffvae;
pub mod dit;
pub mod duration;
pub mod guidance;
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
use std::path::{Path, PathBuf};

/// The distilled stage-1 schedule (`DISTILLED_SIGMA_VALUES`).
pub const DISTILLED_SIGMAS: [f32; 9] = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0];
/// The stage-2 schedule (`STAGE_2_DISTILLED_SIGMA_VALUES`).
pub const STAGE_2_SIGMAS: [f32; 4] = [0.909375, 0.725, 0.421875, 0.0];
/// Seed offsets of the ancestral sampler's noise per stage (`ANCESTRAL_*_NOISE_SEED_OFFSET`).
const STAGE_1_NOISE_OFFSET: u64 = 10000;
const STAGE_2_NOISE_OFFSET: u64 = 20000;
const TEMPORAL_NOISE_OFFSET: u64 = 30000;
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
    /// Denoise the soundtrack again in the guided pipeline's second stage (`--refine-audio`;
    /// otherwise the first stage's is kept).
    pub refine_audio: bool,
    /// Double the frame count and rate with the temporal upsampler after the last stage.
    pub temporal_upscale: bool,
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
    pub cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
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
            temporal_upscale: false,
            images: Vec::new(),
            audio: true,
            eta: None,
            image_crf: None,
            noise: Noise::Oxinfer,
            cancel: None,
        }
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
    /// The distilled LoRA's switch (when [`Paths::distilled_lora`] is loaded).
    distilled_switch: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    vae_files: Option<&'static ModelFiles>,
}

/// How one denoising stage runs.
struct Run<'a> {
    sigmas: &'a [f32],
    /// The ancestral sampler's noise seed.
    seed: u64,
    eta: f32,
    /// The audio rows are kept as they are (conditioning, timestep 0).
    frozen_audio: bool,
    guidance: Option<&'a Guidance>,
    name: &'a str,
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
        let distilled_switch = match &paths.distilled_lora {
            Some(l) => {
                let sw = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
                lora::apply_switched(&mut dit, &[(l.clone(), 1.0)], Some(&sw))?;
                dit.lora_switch = Some(sw.clone());
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
        on.then(|| p.guidance.clone())
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

    fn set_distilled_lora(&self, on: bool) {
        if let Some(s) = &self.distilled_switch {
            s.store(on, std::sync::atomic::Ordering::Relaxed);
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
            video: self.dit.video_connector.forward(&f.video, text::MAX_TOKENS),
            audio: self.dit.audio_connector.forward(&f.audio, text::MAX_TOKENS),
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
            let secs = head.seconds(&ctx.video, &ctx.audio);
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

    /// Denoises one stage: the video state (`lt x lh x lw` rows, then any guide rows) and the
    /// audio state over `run.sigmas`, with the guidance passes (`neg`: the negative prompt's
    /// context) when there is guidance.
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
        let mut pos = video_positions(lt, lh, lw, p.fps);
        for im in p.images.iter().filter(|im| im.frame != 0) {
            pos.extend(guide_positions(im.frame, lh, lw, p.fps));
        }
        let apos = audio_positions(audio.rows());
        let mut g = NoiseGen::new(&p.noise, run.seed);
        let sigmas = run.sigmas;
        let steps = sigmas.len() - 1;
        let frozen = run.frozen_audio;
        let mut last: Option<(Vec<f32>, Vec<f32>)> = None;
        for i in 0..steps {
            if p.cancelled() {
                bail!("cancelled");
            }
            let (s, next) = (sigmas[i], sigmas[i + 1]);
            let vt = video.timesteps(s);
            let at = audio.timesteps(if frozen { 0.0 } else { s });
            // One transformer pass: both streams' denoised rows (`X0Model`: `x - t v`).
            let pass = |c: &PromptContext, ptb: &Perturb| -> (Vec<f32>, Vec<f32>) {
                let (vv, av) = self.dit.forward_with(
                    &StreamIn {
                        x: &video.x,
                        t: &vt,
                        sigma: s,
                        pos: &pos,
                        ctx: &c.video,
                        keyframe_rows: lh * lw,
                    },
                    &StreamIn {
                        x: &audio.x,
                        t: &at,
                        sigma: if frozen { 0.0 } else { s },
                        pos: &apos,
                        ctx: &c.audio,
                        keyframe_rows: 0,
                    },
                    ptb,
                );
                let x0 = |st: &State, v: &[f32], t: &[f32]| -> Vec<f32> {
                    let w = st.width;
                    (0..st.x.len()).map(|i| st.x[i] - v[i] * t[i / w]).collect()
                };
                (x0(video, &vv, &vt), x0(audio, &av, &at))
            };
            let (vd, ad) = match run.guidance {
                None => pass(ctx, &Perturb::default()),
                Some(gd) => {
                    let (vskip, askip) = (gd.video.skips(i), gd.audio.skips(i));
                    match &last {
                        Some(l) if vskip && askip => l.clone(),
                        _ => {
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
                            let vd = match (&last, vskip) {
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
                            let ad = match (&last, askip) {
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
                            (vd, ad)
                        }
                    }
                }
            };
            if run.guidance.is_some_and(|g| g.video.skip_step != 0 || g.audio.skip_step != 0) {
                last = Some((vd.clone(), ad.clone()));
            }
            let draw = run.eta > 0.0 && next != 0.0;
            let vn = draw.then(|| g.normals(video.x.len()));
            let an = draw.then(|| g.normals(audio.x.len()));
            ancestral_step(video, &vd, s, next, run.eta, vn.as_deref());
            if !frozen {
                ancestral_step(audio, &ad, s, next, run.eta, an.as_deref());
            }
            progress(run.name, i + 1, steps);
        }
        Ok(())
    }

    /// Text (and images) to video.
    pub fn generate(&self, p: &Params, progress: Progress) -> Result<Video> {
        let t0 = Instant::now();
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
        let frames = self.resolve_frames(p, ctx)?;
        for im in &p.images {
            if im.frame >= frames {
                bail!("LTX-2.5: an image conditions frame {} of a {frames}-frame clip", im.frame);
            }
        }
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
        let condition = |st: &mut State, w: usize, h: usize| {
            if let Some(enc) = &encoder {
                let (lh, lw) = (h / SPACE_SCALE, w / SPACE_SCALE);
                for im in &p.images {
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
        video.noise(&mut rng, 1.0);
        let mut audio = State::zeros(na, ac);
        audio.noise(&mut rng, 1.0);
        let guided = guidance.is_some();
        let sig1 = match (&p.stage1_sigmas, &guidance) {
            (Some(s), _) => s.clone(),
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
        // The distilled LoRA is off while the guided first stage runs.
        self.set_distilled_lora(!guided);
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
                frozen_audio: false,
                guidance: guidance.as_ref(),
                name: "stage 1",
            },
            progress,
        );
        self.set_distilled_lora(true);
        r?;
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
            drop(ups);
            dump("upsampled_video", &x);
            lh *= 2;
            lw *= 2;
            let mut v2 = State::zeros(lt * lh * lw, c);
            v2.x = x;
            condition(&mut v2, p.width, p.height);
            let s0 = p.stage2_sigmas[0];
            v2.noise(&mut rng, s0);
            let mut a2 = State::zeros(na, ac);
            a2.x = audio.x.clone();
            // The guided pipeline keeps the first stage's soundtrack (frozen: noised at scale 0,
            // which still draws its noise, then timestep 0) unless asked to refine it.
            let frozen = guided && !p.refine_audio;
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
                    guidance: None,
                    name: "stage 2",
                },
                progress,
            )?;
            if frozen {
                a2.x = audio.x.clone();
            }
            clip(&mut v2, lt * lh * lw);
            dump("stage2_video", &v2.x);
            dump("stage2_audio", &a2.x);
            video = v2;
            audio = a2;
            t_s2 = t2.elapsed().as_secs_f64();
        }
        let (mut lt, mut fps, mut frames) = (lt, p.fps, frames);
        if p.temporal_upscale {
            let up = self
                .paths
                .temporal_upsampler
                .as_ref()
                .context("temporal upscaling needs the temporal upsampler (latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors)")?;
            let uf = ModelFiles::open_file(up, false)?;
            let ups = upsample::LatentUpsampler::load(&uf)?;
            let x = upsample_rows(&ups, &vfiles_stats, &video.x, c, (lt, lh, lw));
            drop(ups);
            lt = 2 * lt - 1;
            fps *= 2.0;
            frames = (lt - 1) * TIME_SCALE + 1;
            let mut v3 = State::zeros(lt * lh * lw, c);
            v3.x = x;
            let sig: Vec<f32> = DISTILLED_SIGMAS[4..].to_vec();
            v3.noise(&mut rng, sig[0]);
            let mut a3 = State::zeros(na, ac);
            a3.x = audio.x.clone();
            a3.clean = audio.x.clone();
            a3.mask.fill(0.0);
            let pt = Params {
                fps,
                images: Vec::new(),
                ..p.clone()
            };
            self.denoise(
                ctx,
                None,
                &mut v3,
                &mut a3,
                (lt, lh, lw),
                &pt,
                &Run {
                    sigmas: &sig,
                    seed: p.seed + TEMPORAL_NOISE_OFFSET,
                    eta: p.eta.unwrap_or(1.0),
                    frozen_audio: true,
                    guidance: None,
                    name: "temporal",
                },
                progress,
            )?;
            video = v3;
        }
        // Decode.
        let t3 = Instant::now();
        let lat = unpatchify(&video.x, c, lt, lh, lw);
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
        let t_vae = t3.elapsed().as_secs_f64();
        let t4 = Instant::now();
        let soundtrack = match (&self.paths.audio_vae, p.audio) {
            (Some(path), true) => {
                let af = ModelFiles::open_file(path, false)?;
                let dec = audio::AudioDecoder::load(&af)?;
                let mel = dec.decode(&audio.x)?;
                let t_mel = mel.len() / (2 * dec.mel_bins);
                let voc = audio::Vocoder::load(&af)?;
                let mut ch = voc.forward(&mel, t_mel);
                dump("mel", &mel);
                dump("waveform", &ch.concat());
                let keep = ((n_frames as f64 / fps as f64) * voc.output_rate as f64).round() as usize;
                for c in ch.iter_mut() {
                    c.truncate(keep);
                }
                Some(Soundtrack {
                    sample_rate: voc.output_rate,
                    channels: ch,
                })
            }
            _ => None,
        };
        eprintln!(
            "[LTX-2.5: {w}x{h}, {n_frames} frames at {fps} fps, text {t_text:.1}s, stage 1 {t_s1:.1}s, stage 2 {t_s2:.1}s, VAE {t_vae:.1}s, audio {:.1}s]",
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
