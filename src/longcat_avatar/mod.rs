//! LongCat-Video-Avatar 1.6 (`meituan-longcat/LongCat-Video-Avatar-1.6`): audio-driven avatar
//! video. A reference image (or only a prompt), a speech recording and a text prompt become a
//! 25 fps video of the character speaking; a second speaker with their own recording is
//! possible, and clips chain into long videos. Ported from Meituan's reference
//! (github.com/meituan-longcat/LongCat-Video, `pipeline_longcat_video_avatar.py` and the two
//! `run_demo_avatar_*` scripts with `--model_type avatar-v1.6`):
//!
//! - text: umT5-XXL of the base repository `meituan-longcat/LongCat-Video` (`text_encoder/`,
//!   `tokenizer/`), 512 positions ([`crate::flux::t5`], which holds a relative-position bias per
//!   block for umT5);
//! - audio: Whisper-large-v3 encoder hidden states, five groups of layers per 20 ms, resampled
//!   to one row per video frame; a clip of 93 frames reads the rows at -2..2 around each frame
//!   ([`audio`]);
//! - latents: the Wan 2.1 VAE of the base repository (`vae/`: 16 channels, 8x in space, 4x in
//!   time; [`vae`]), normalized with its per-channel mean and standard deviation;
//! - the transformer ([`dit`]): `base_model/` in bf16 or `base_model_int8/` (int8 rows with a
//!   scale each, read as stored), with the two LoRAs as side paths ([`lora`]);
//! - sampling ([`pipeline`]): the distilled 8-step rectified-flow schedule (shift 7) without
//!   classifier-free guidance, the first `mix_steps` steps with the DMD LoRA (`loraA`) and the
//!   rest with the OPSD one (`loraB`); or the base model with text and audio guidance;
//! - long videos: each further clip is conditioned on the last 13 frames of the previous one
//!   (re-encoded) and on the first clip's first latent frame as a reference placed at temporal
//!   position `ref_img_index`.
//!
//! Not ported: the vocal separator (`vocal_separator/Kim_Vocal_2.onnx`, an MDX-Net run by ONNX
//! Runtime in the reference: the speech goes to Whisper as given), `ftfy`'s mojibake repair of the prompt, the
//! key / value cache of the undistilled continuation (`use_kv_cache`, which the mixed-LoRA
//! setting turns off), `enhance_hf`, block-sparse attention, FP8 feed-forwards and the 1.0
//! checkpoints (wav2vec2 features).

pub mod audio;
#[cfg(all(feature = "gpu", not(any(target_os = "macos", target_os = "ios", target_arch = "wasm32"))))]
mod cuda;
pub mod dit;
pub mod lora;
pub mod pipeline;
#[doc(hidden)]
pub mod tiny;
pub mod vae;
pub mod weights;

pub use pipeline::{AvatarParams, AvatarPipeline, Clip, FPS, Sources, Stage, is_avatar};

/// The hub repositories of the release and of the parts it borrows.
pub const AVATAR_REPO: &str = "meituan-longcat/LongCat-Video-Avatar-1.6";
pub const BASE_REPO: &str = "meituan-longcat/LongCat-Video";

/// The `pipeline` name of this model's presets in [`crate::advisor::PRESETS`].
pub const PIPELINE: &str = "longcat-avatar";

/// The files of the release this crate reads, besides the transformer's shards (which the index
/// of the chosen `base_model*/` names).
#[cfg(feature = "hf")]
const AVATAR_FILES: &[&str] = &[
    "config.json",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "loraA/lora.safetensors",
    "loraB/lora.safetensors",
    "whisper-large-v3/config.json",
    "whisper-large-v3/preprocessor_config.json",
    "whisper-large-v3/model.safetensors",
];

/// The files of the base repository this crate reads, besides the text encoder's shards.
#[cfg(feature = "hf")]
const BASE_FILES: &[&str] = &[
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "tokenizer/tokenizer.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors.index.json",
];

/// The shard files a safetensors index names, under `folder/`.
#[cfg(feature = "hf")]
fn shards_of(dir: &std::path::Path, folder: &str, index: &str) -> anyhow::Result<Vec<String>> {
    use anyhow::Context;
    let p = dir.join(folder).join(index);
    let v: serde_json::Value = serde_json::from_slice(&crate::fsx::read(&p)?).with_context(|| p.display().to_string())?;
    let mut names: Vec<String> = v["weight_map"]
        .as_object()
        .with_context(|| format!("{}: no weight_map", p.display()))?
        .values()
        .filter_map(|f| f.as_str().map(|f| format!("{folder}/{f}")))
        .collect();
    names.sort();
    names.dedup();
    Ok(names)
}

/// Where the parts of a request are: `spec` is the release as a folder or a repository id
/// (fetched into the cache: the chosen transformer, the two LoRAs, the scheduler's config and
/// Whisper's `model.safetensors`, not the other transformer nor Whisper's other weight formats),
/// `base` the base repository likewise (default: `LongCat-Video` next to a release folder, as
/// the reference lays them out, else [`BASE_REPO`]: its text encoder, tokenizer and VAE, not its
/// own 54 GB transformer).
pub fn resolve(
    spec: &str,
    base: Option<&str>,
    int8: bool,
    text_quant: crate::model::Quant,
    cache_dir: Option<&std::path::Path>,
) -> anyhow::Result<Sources> {
    use std::path::{Path, PathBuf};
    let sub = if int8 { "base_model_int8" } else { "base_model" };
    let local = Path::new(spec).is_dir();
    let avatar: PathBuf = if local {
        PathBuf::from(spec)
    } else {
        #[cfg(feature = "hf")]
        {
            let index = if int8 {
                "quantized_model.safetensors.index.json"
            } else {
                "diffusion_pytorch_model.safetensors.index.json"
            };
            let (cfg, idx) = (format!("{sub}/config.json"), format!("{sub}/{index}"));
            let mut names: Vec<&str> = AVATAR_FILES.to_vec();
            names.extend([cfg.as_str(), idx.as_str()]);
            let dir = crate::hf::resolve_files(spec, &names, cache_dir)?;
            let shards = shards_of(&dir, sub, index)?;
            crate::hf::resolve_files(spec, &shards.iter().map(String::as_str).collect::<Vec<_>>(), cache_dir)?
        }
        #[cfg(not(feature = "hf"))]
        anyhow::bail!("{spec}: not a folder (built without the `hf` feature, repository ids are not fetched)")
    };
    let sibling = avatar.parent().map(|p| p.join("LongCat-Video")).filter(|p| p.join("vae").is_dir());
    let base: PathBuf = match (base, sibling) {
        (Some(b), _) if Path::new(b).is_dir() => PathBuf::from(b),
        (None, Some(s)) if local => s,
        (b, _) => {
            let repo = b.unwrap_or(BASE_REPO);
            #[cfg(feature = "hf")]
            {
                let dir = crate::hf::resolve_files(repo, BASE_FILES, cache_dir)?;
                let shards = shards_of(&dir, "text_encoder", "model.safetensors.index.json")?;
                crate::hf::resolve_files(repo, &shards.iter().map(String::as_str).collect::<Vec<_>>(), cache_dir)?
            }
            #[cfg(not(feature = "hf"))]
            anyhow::bail!("{repo}: the base repository's folder is needed (text_encoder/, tokenizer/, vae/): name it with --avatar-base")
        }
    };
    let _ = (cache_dir, sub);
    Ok(Sources {
        avatar,
        base,
        int8,
        text_quant,
    })
}

/// The transformer's device engine: decided at the first forward.
#[derive(Default)]
pub(crate) struct Device {
    #[cfg(all(feature = "gpu", not(any(target_os = "macos", target_os = "ios", target_arch = "wasm32"))))]
    engine: std::sync::OnceLock<Option<cuda::Engine>>,
}

impl Device {
    /// The block stack over `x` on the device; false (with `x` untouched) when the host runs it.
    pub(crate) fn run(&self, dit: &dit::Dit, x: &mut [f32], ctx: &dit::Ctx) -> bool {
        #[cfg(all(feature = "gpu", not(any(target_os = "macos", target_os = "ios", target_arch = "wasm32"))))]
        if let Some(e) = self.engine.get_or_init(|| cuda::Engine::new(dit)) {
            return e.run(dit, x, ctx);
        }
        let _ = (dit, x, ctx);
        false
    }

    /// Frees the engine's device memory, unless the device still has `keep` bytes free (0:
    /// in any case).
    pub(crate) fn release(&self, keep: usize) {
        #[cfg(all(feature = "gpu", not(any(target_os = "macos", target_os = "ios", target_arch = "wasm32"))))]
        if let Some(Some(e)) = self.engine.get() {
            e.release(keep);
        }
        let _ = keep;
    }
}
