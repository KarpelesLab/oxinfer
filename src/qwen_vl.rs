//! A Qwen vision-language model as a conditioning encoder: Qwen2.5-VL (or Qwen2-VL / Qwen3.5)
//! language model, vision tower, tokenizer and image processor in one value, returning hidden
//! states rather than tokens. Diffusion pipelines use it as their text encoder (Qwen-Image,
//! Step1X-Edit, OmniGen2); the chat path goes through [`crate::engine::Engine`] instead.
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! use oxinfer::qwen_vl::QwenVl;
//! let vl = QwenVl::load("Qwen/Qwen2.5-VL-3B-Instruct".as_ref(), &Default::default())?;
//! let img = vl.encode_image(&vec![128u8; 64 * 64 * 3], 64, 64, 0)?;
//! let text = format!("<|im_start|>user\n{}Describe it.<|im_end|>\n<|im_start|>assistant\n", img.placeholder());
//! let h = vl.hidden_states(&text, &[img], &[])?;
//! println!("{} tokens x {}", h.ids.len(), h.hidden);
//! # Ok(()) }
//! ```

use crate::model::Quant;
use crate::qwen::QwenModel;
use crate::qwen_vision::{QwenMmConfig, QwenVisionTower};
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};

/// How to load a [`QwenVl`].
#[derive(Clone, Debug)]
pub struct QwenVlOptions {
    /// Precision of the language model's matmul weights (the vision tower stays bf16).
    pub quant: Quant,
    /// Map the weights from disk instead of reading them into memory.
    pub mmap: bool,
    /// Directory holding `tokenizer.json` (default: the model directory, or a sibling
    /// `processor/` / `tokenizer/` folder of a diffusers pipeline).
    pub tokenizer_dir: Option<PathBuf>,
    /// Directory holding `preprocessor_config.json` (same defaults).
    pub processor_dir: Option<PathBuf>,
    /// Load the vision tower (images); false for text-only conditioning.
    pub vision: bool,
}

impl Default for QwenVlOptions {
    fn default() -> Self {
        Self {
            quant: Quant::Bf16,
            mmap: false,
            tokenizer_dir: None,
            processor_dir: None,
            vision: true,
        }
    }
}

/// An image run through the vision tower, ready to be placed in a prompt.
#[derive(Clone, Debug)]
pub struct VlImage {
    /// `[n, hidden]` language-model embeddings.
    pub emb: Vec<f32>,
    pub n: usize,
    /// Merged-token grid (rows, columns).
    pub grid: (usize, usize),
    /// The size the image processor resized it to.
    pub size: (usize, usize),
    /// DeepStack features (Qwen3-VL): `[n, hidden]` added to the image's tokens after language
    /// model layer 0, 1, ...; empty for the other towers.
    pub deep: Vec<Vec<f32>>,
}

impl VlImage {
    /// `<|vision_start|>` + one `<|image_pad|>` per token + `<|vision_end|>`.
    pub fn placeholder(&self) -> String {
        format!("<|vision_start|>{}<|vision_end|>", "<|image_pad|>".repeat(self.n))
    }
}

/// Hidden states of one prompt.
#[derive(Clone, Debug)]
pub struct Hidden {
    pub ids: Vec<u32>,
    pub hidden: usize,
    /// `[ids.len(), hidden]` after the final norm (transformers' `hidden_states[-1]` /
    /// `last_hidden_state`).
    pub last: Vec<f32>,
    /// `(layer, [ids.len(), hidden])` residual stream after each requested layer
    /// (transformers' `hidden_states[layer + 1]`).
    pub layers: Vec<(usize, Vec<f32>)>,
}

/// A Qwen vision-language model loaded for conditioning (see the module docs).
pub struct QwenVl {
    // Field order: the model and tower borrow `files` and are dropped first.
    model: QwenModel<'static>,
    tower: Option<QwenVisionTower<'static>>,
    pub mm: Option<QwenMmConfig>,
    pub tokenizer: crate::tokenizer::Tokenizer,
    pub dir: PathBuf,
    _files: Box<ModelFiles>,
}

/// The DeepStack rows of `images` for a session (`QwenSession::deepstack`): each image's
/// features after the first layers, at its tokens (`refs`: the placeholder positions, in order).
fn deepstack(images: &[VlImage], refs: &[(usize, &[f32])]) -> Result<Option<Box<crate::qwen::DeepStack>>> {
    let mut out = Vec::new();
    let mut at = 0;
    for im in images {
        let start = refs.get(at).map_or(0, |r| r.0);
        if !im.deep.is_empty() && refs[at..at + im.n].iter().enumerate().any(|(k, r)| r.0 != start + k) {
            bail!("an image's placeholder tokens are not contiguous (DeepStack features need one run)");
        }
        out.extend(im.deep.iter().enumerate().map(|(l, rows)| (l, start, rows.clone())));
        at += im.n;
    }
    Ok((!out.is_empty()).then(|| Box::new(crate::qwen::DeepStack(out))))
}

/// The first of `cands` that holds `file`.
fn find_dir(cands: &[PathBuf], file: &str) -> Option<PathBuf> {
    cands.iter().find(|d| crate::fsx::exists(&d.join(file))).cloned()
}

impl QwenVl {
    /// Loads a Qwen VL checkpoint folder (or a Hugging Face repo id, fetched into the cache).
    pub fn load(path_or_repo: &Path, opts: &QwenVlOptions) -> Result<Self> {
        let dir = crate::engine::resolve_model(&path_or_repo.to_string_lossy(), None)?;
        let parent = dir.parent().map(Path::to_path_buf).unwrap_or_default();
        let cands = [dir.clone(), parent.join("processor"), parent.join("tokenizer")];
        let tok_dir = opts.tokenizer_dir.clone().or_else(|| find_dir(&cands, "tokenizer.json"));
        let tokenizer = match tok_dir {
            Some(d) => crate::engine::load_tokenizer(&d)?,
            None => {
                let d = find_dir(&cands, "vocab.json").with_context(|| format!("{}: no tokenizer.json or vocab.json", dir.display()))?;
                qwen2_tokenizer_from_vocab(&d)?
            }
        };
        let cfg = crate::config::QwenConfig::load(&dir)?;
        let files = Box::new(ModelFiles::open(&dir, !opts.mmap)?);
        // SAFETY: the box's heap block does not move, and `QwenVl` drops the model and tower
        // (every borrow of it) before `_files`.
        let fs: &'static ModelFiles = unsafe { &*(files.as_ref() as *const ModelFiles) };
        let model = QwenModel::load(cfg, fs, opts.quant)?;
        let mut mm = QwenMmConfig::load(&dir)?;
        if let Some(m) = &mut mm {
            let pdir = opts.processor_dir.clone().or_else(|| find_dir(&cands, "preprocessor_config.json"));
            if let Some(p) = pdir {
                m.image = crate::qwen_vision::ProcessorConfig::load(&p, false);
            }
        }
        let tower = match (&mm, opts.vision) {
            (Some(m), true) => {
                if m.vision.out_hidden_size != model.cfg.hidden_size {
                    bail!(
                        "vision tower output {} does not match the text model's {}",
                        m.vision.out_hidden_size,
                        model.cfg.hidden_size
                    );
                }
                Some(QwenVisionTower::load_deep(m.vision.clone(), fs)?)
            }
            _ => None,
        };
        Ok(Self {
            model,
            tower,
            mm,
            tokenizer,
            dir,
            _files: files,
        })
    }

    pub fn model(&self) -> &QwenModel<'static> {
        &self.model
    }

    pub fn hidden_size(&self) -> usize {
        self.model.cfg.hidden_size
    }

    /// Whether the vision tower is loaded.
    pub fn has_vision(&self) -> bool {
        self.tower.is_some()
    }

    /// Runs an RGB8 image through the image processor (smart resize within `max_pixels`, 0 for
    /// the processor's budget) and the vision tower.
    pub fn encode_image(&self, rgb: &[u8], w: usize, h: usize, max_pixels: usize) -> Result<VlImage> {
        let (Some(tower), Some(mm)) = (&self.tower, &self.mm) else {
            bail!("{}: loaded without a vision tower", self.dir.display());
        };
        let px = crate::qwen_vision::preprocess_image(rgb, w, h, &mm.image, max_pixels)?;
        let (emb, deep) = tower.encode_deep(&px);
        let m = mm.vision.spatial_merge_size;
        let (_, gh, gw) = px.grid;
        let n = emb.len() / self.hidden_size();
        let p = mm.image.patch_size;
        Ok(VlImage {
            emb,
            n,
            grid: (gh / m, gw / m),
            size: (gw * p, gh * p),
            deep,
        })
    }

    /// Tokenizes `text` (special tokens parsed, nothing added).
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?
            .get_ids()
            .to_vec())
    }

    /// Hidden states of `text`, whose image placeholders (`VlImage::placeholder`, in order) take
    /// the embeddings of `images` (with M-RoPE positions); `layers` asks for intermediate
    /// residual streams too.
    pub fn hidden_states(&self, text: &str, images: &[VlImage], layers: &[usize]) -> Result<Hidden> {
        self.hidden_states_tokens(self.tokenize(text)?, images, None, layers)
    }

    /// [`QwenVl::hidden_states`] of token ids (image placeholders already expanded), optionally
    /// with explicit `[time, height, width]` RoPE positions per token (transformers'
    /// `position_ids`) instead of the ones `get_rope_index` derives.
    pub fn hidden_states_tokens(
        &self,
        ids: Vec<u32>,
        images: &[VlImage],
        positions: Option<Vec<[usize; 3]>>,
        layers: &[usize],
    ) -> Result<Hidden> {
        if positions.as_ref().is_some_and(|p| p.len() != ids.len()) {
            bail!("{} positions for {} tokens", positions.map_or(0, |p| p.len()), ids.len());
        }
        if ids.is_empty() {
            bail!("empty prompt");
        }
        let h = self.hidden_size();
        let feats: Vec<crate::mm::MmFeature> = match &self.mm {
            Some(mm) => images
                .iter()
                .map(|im| crate::mm::MmFeature {
                    token_id: mm.image_token_id,
                    emb: im.emb.clone(),
                    n: im.n,
                    grids: vec![im.grid],
                })
                .collect(),
            None if images.is_empty() => Vec::new(),
            None => bail!("{}: not a vision-language model", self.dir.display()),
        };
        let mut sess = self.model.new_session();
        sess.capture = layers.to_vec();
        sess.positions = positions;
        for (p, gh, gw) in crate::mm::vision_spans(&ids, &feats)? {
            sess.add_vision_span(p, gh, gw);
        }
        let ov = crate::mm::overrides(&ids, &feats, h)?;
        let refs: Vec<(usize, &[f32])> = ov.iter().map(|(i, v)| (*i, v.as_slice())).collect();
        sess.deepstack = deepstack(images, &refs)?;
        let last = self.model.forward_hidden_mm(&mut sess, &ids, &refs);
        Ok(Hidden {
            ids,
            hidden: h,
            last,
            layers: std::mem::take(&mut sess.captured),
        })
    }

    /// Continues `text` (image placeholders as in [`QwenVl::hidden_states`]) greedily for at most
    /// `max_new_tokens` tokens, stopping before any of `eos`: the new token ids. Logits of tokens
    /// already in the sequence are divided (when positive) or multiplied by `repetition_penalty`,
    /// as transformers' `generate` does (Qwen2.5-VL's generation config: top-k 1, penalty 1.05).
    pub fn generate(
        &self,
        text: &str,
        images: &[VlImage],
        max_new_tokens: usize,
        eos: &[u32],
        repetition_penalty: f32,
    ) -> Result<Vec<u32>> {
        let ids = self.tokenize(text)?;
        if ids.is_empty() {
            bail!("empty prompt");
        }
        let h = self.hidden_size();
        let feats: Vec<crate::mm::MmFeature> = match &self.mm {
            Some(mm) => images
                .iter()
                .map(|im| crate::mm::MmFeature {
                    token_id: mm.image_token_id,
                    emb: im.emb.clone(),
                    n: im.n,
                    grids: vec![im.grid],
                })
                .collect(),
            None if images.is_empty() => Vec::new(),
            None => bail!("{}: not a vision-language model", self.dir.display()),
        };
        let mut sess = self.model.new_session();
        for (p, gh, gw) in crate::mm::vision_spans(&ids, &feats)? {
            sess.add_vision_span(p, gh, gw);
        }
        let ov = crate::mm::overrides(&ids, &feats, h)?;
        let refs: Vec<(usize, &[f32])> = ov.iter().map(|(i, v)| (*i, v.as_slice())).collect();
        sess.deepstack = deepstack(images, &refs)?;
        let last = self.model.forward_hidden_mm(&mut sess, &ids, &refs);
        let mut logits = self.model.logits(&last[(ids.len() - 1) * h..], 1);
        let mut seen = vec![false; logits.len()];
        ids.iter().for_each(|&i| seen[i as usize] = true);
        let mut out = Vec::new();
        while out.len() < max_new_tokens {
            if repetition_penalty != 1.0 {
                for (l, _) in logits.iter_mut().zip(&seen).filter(|(_, s)| **s) {
                    *l = if *l > 0.0 {
                        *l / repetition_penalty
                    } else {
                        *l * repetition_penalty
                    };
                }
            }
            // The first of equal maxima, as torch's argmax.
            let next = (0..logits.len()).rev().max_by(|&a, &b| logits[a].total_cmp(&logits[b])).unwrap() as u32;
            if eos.contains(&next) {
                break;
            }
            out.push(next);
            seen[next as usize] = true;
            if out.len() < max_new_tokens {
                logits = self.model.forward(&mut sess, &[next]);
            }
        }
        Ok(out)
    }
}

/// A Qwen2 byte-level BPE tokenizer from `vocab.json` + `merges.txt` and the added tokens of
/// `tokenizer_config.json` (folders without `tokenizer.json`, e.g. Qwen-Image's `tokenizer/`).
pub fn qwen2_tokenizer_from_vocab(dir: &Path) -> Result<crate::tokenizer::Tokenizer> {
    use serde_json::json;
    let vocab: serde_json::Value = serde_json::from_slice(&crate::fsx::read(&dir.join("vocab.json"))?)?;
    let merges_txt = String::from_utf8(crate::fsx::read(&dir.join("merges.txt"))?)?;
    let merges: Vec<&str> = merges_txt.lines().filter(|l| !l.starts_with("#version") && !l.is_empty()).collect();
    let cfg: serde_json::Value = crate::fsx::read(&dir.join("tokenizer_config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default();
    let mut added = Vec::new();
    if let Some(m) = cfg["added_tokens_decoder"].as_object() {
        for (id, t) in m {
            added.push(json!({
                "id": id.parse::<u64>()?, "content": t["content"], "single_word": false, "lstrip": false,
                "rstrip": false, "normalized": false, "special": t["special"].as_bool().unwrap_or(true),
            }));
        }
    }
    let regex = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    let byte_level = json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false});
    let j = json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": added,
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": regex}, "behavior": "Isolated", "invert": false}, byte_level,
        ]},
        "post_processor": byte_level, "decoder": byte_level,
        "model": {
            "type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": "", "end_of_word_suffix": "",
            "fuse_unk": false, "byte_fallback": false, "ignore_merges": false, "vocab": vocab, "merges": merges,
        },
    });
    crate::tokenizer::Tokenizer::from_bytes(serde_json::to_vec(&j)?).map_err(|e| anyhow::anyhow!("tokenizer from {}: {e}", dir.display()))
}
