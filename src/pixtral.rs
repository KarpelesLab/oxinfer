//! The Pixtral vision tower and Mistral 3 / Large 4 projector (`PixtralVisionModel` +
//! `MistralLarge4MultiModalProjector`): a 14x14 patch convolution without bias, RMSNorm, pre-norm
//! blocks with 2D axial RoPE (the first half of each head's rotary pairs with the patch row, the
//! second with the column) and a SiLU-gated MLP, every patch of an image attending to every other
//! one; then the projector: RMSNorm, the patch merger (each 2x2 block of patches concatenated,
//! channel-major as `torch.nn.functional.unfold` orders them, through a linear layer), and a
//! GELU MLP into the language model's width.
//!
//! Weights come from the checkpoint itself (`vision_encoder.*` in Mistral's layout, read under
//! transformers' names, or a transformers-saved `model.vision_tower.*`) or from llama.cpp's
//! `mmproj` GGUF file (`v.*`, `mm.*`). Mistral's layout and the GGUF keep the query / key rows in
//! the interleaved RoPE order; transformers' own files have them permuted for rotate-half RoPE.
//!
//! Prompt layout (`PixtralProcessor`): rows of `[IMG]` placeholders, one per merged token, each
//! row closed by `[IMG_BREAK]`, the last by `[IMG_END]`.

use crate::kernels::*;
use crate::model::{Lin, linear, load_lin};
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};

/// The tower and projector shapes.
#[derive(Clone, Debug)]
pub struct PixtralConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub patch_size: usize,
    /// Longest image side after resizing.
    pub image_size: usize,
    pub rope_theta: f64,
    pub spatial_merge_size: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    /// The placeholder, row break and image end token ids.
    pub image_token_id: u32,
    pub image_break_token_id: u32,
    pub image_end_token_id: u32,
}

impl PixtralConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// The vision config of a Mistral Large 4 / Mistral 3 checkpoint folder (`config.json`
    /// `vision_config`, processor settings from `processor_config.json`); None without one.
    pub fn load(dir: &Path) -> Result<Option<Self>> {
        Self::load_with(dir, &Mmproj::Auto)
    }

    /// [`PixtralConfig::load`] with the projector chosen by `mmproj`: an explicit file replaces
    /// the checkpoint's tower (None when it is not a Pixtral `mmproj`, e.g. a Qwen one), `None`
    /// gives no vision.
    pub fn load_with(dir: &Path, mmproj: &Mmproj) -> Result<Option<Self>> {
        match mmproj {
            Mmproj::None => return Ok(None),
            Mmproj::File(p) if projector_type(p).as_deref() != Some("pixtral") => return Ok(None),
            Mmproj::File(p) => return Self::from_mmproj(p, dir).map(Some),
            Mmproj::Auto => {}
        }
        if crate::gguf::is_gguf(dir) {
            return match mmproj_for(dir) {
                Some(p) if crate::gguf::header(&p)?.get("clip.projector_type").and_then(|v| v.as_str()) == Some("pixtral") => {
                    Self::from_mmproj(&p, dir).map(Some)
                }
                _ => Ok(None),
            };
        }
        let Ok(b) = crate::fsx::read(&dir.join("config.json")) else {
            return Ok(None);
        };
        let c: serde_json::Value = serde_json::from_slice(&b)?;
        let v = &c["vision_config"];
        if v.get("model_type").and_then(|t| t.as_str()) != Some("pixtral") {
            return Ok(None);
        }
        let u = |o: &serde_json::Value, k: &str, d: usize| o.get(k).and_then(|x| x.as_u64()).map_or(d, |x| x as usize);
        let proc: serde_json::Value = crate::fsx::read(&dir.join("processor_config.json"))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default();
        let ip = &proc["image_processor"];
        let arr = |k: &str, d: [f32; 3]| -> [f32; 3] {
            match ip.get(k).and_then(|x| x.as_array()) {
                Some(a) if a.len() == 3 => std::array::from_fn(|i| a[i].as_f64().unwrap_or(d[i] as f64) as f32),
                _ => d,
            }
        };
        let tok = load_vocab(dir);
        let id = |s: &str, d: u32| tok.as_ref().and_then(|t| t.token_to_id(s)).unwrap_or(d);
        Ok(Some(Self {
            hidden_size: u(v, "hidden_size", 1024),
            intermediate_size: u(v, "intermediate_size", 4096),
            num_hidden_layers: u(v, "num_hidden_layers", 24),
            num_attention_heads: u(v, "num_attention_heads", 16),
            patch_size: u(v, "patch_size", 16),
            image_size: ip["size"]["longest_edge"].as_u64().map_or(u(v, "image_size", 1024), |x| x as usize),
            rope_theta: v["rope_parameters"]["rope_theta"]
                .as_f64()
                .or_else(|| v["rope_theta"].as_f64())
                .unwrap_or(10_000.0),
            spatial_merge_size: u(&c, "spatial_merge_size", 2),
            image_mean: arr("image_mean", CLIP_MEAN),
            image_std: arr("image_std", CLIP_STD),
            image_token_id: c["image_token_index"].as_u64().map_or_else(|| id("[IMG]", 10), |x| x as u32),
            image_break_token_id: id("[IMG_BREAK]", 12),
            image_end_token_id: id("[IMG_END]", 13),
        }))
    }

    /// The config of llama.cpp's `clip` / `pixtral` mmproj file `path`, for the language model
    /// `model` (a GGUF file, whose vocabulary gives the image token ids).
    fn from_mmproj(path: &Path, model: &Path) -> Result<Self> {
        let h = crate::gguf::header(path)?;
        if h.get("clip.projector_type").and_then(|v| v.as_str()) != Some("pixtral") {
            bail!("{}: not a pixtral mmproj file", path.display());
        }
        let u = |k: &str| {
            h.get(k)
                .and_then(|v| v.as_u64())
                .map(|x| x as usize)
                .with_context(|| format!("{}: no {k}", path.display()))
        };
        let arr = |k: &str, d: [f32; 3]| -> [f32; 3] {
            match h.get(k).and_then(|v| v.as_array()) {
                Some(a) if a.len() == 3 => std::array::from_fn(|i| a[i].as_f64().unwrap_or(d[i] as f64) as f32),
                _ => d,
            }
        };
        let tok = match crate::gguf::is_gguf(model) {
            true => crate::gguf::header(model).ok().and_then(|m| m.tokenizer().ok()),
            false => load_vocab(model),
        };
        let id = |s: &str, d: u32| tok.as_ref().and_then(|t| t.token_to_id(s)).unwrap_or(d);
        Ok(Self {
            hidden_size: u("clip.vision.embedding_length")?,
            intermediate_size: u("clip.vision.feed_forward_length")?,
            num_hidden_layers: u("clip.vision.block_count")?,
            num_attention_heads: u("clip.vision.attention.head_count")?,
            patch_size: u("clip.vision.patch_size")?,
            image_size: u("clip.vision.image_size")?,
            rope_theta: h.get("clip.rope.freq_base").and_then(|v| v.as_f64()).unwrap_or(10_000.0),
            spatial_merge_size: u("clip.vision.spatial_merge_size").unwrap_or(2),
            image_mean: arr("clip.vision.image_mean", CLIP_MEAN),
            image_std: arr("clip.vision.image_std", CLIP_STD),
            image_token_id: id("[IMG]", 10),
            image_break_token_id: id("[IMG_BREAK]", 12),
            image_end_token_id: id("[IMG_END]", 13),
        })
    }
}

const CLIP_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
const CLIP_STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];

fn load_vocab(dir: &Path) -> Option<crate::tokenizer::Tokenizer> {
    crate::tokenizer::Tokenizer::from_bytes(crate::fsx::read(&dir.join("tokenizer.json")).ok()?).ok()
}

/// Where the vision tower of a Mistral Large 4 or Qwen3.5-family (Qwen3.8-Flash-Next) checkpoint
/// comes from (`EngineOptions::mmproj`, `--mmproj`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Mmproj {
    /// The checkpoint's own tower, or for a GGUF model the `mmproj*.gguf` beside it
    /// ([`mmproj_for`]).
    #[default]
    Auto,
    /// This llama.cpp `mmproj` GGUF file.
    File(PathBuf),
    /// No vision tower: image input is refused.
    None,
}

impl Mmproj {
    /// `auto` (or empty), `none` / `off`, or a file path.
    pub fn parse(s: &str) -> Self {
        match s.trim() {
            "" | "auto" => Mmproj::Auto,
            "none" | "off" | "0" => Mmproj::None,
            p => Mmproj::File(PathBuf::from(p)),
        }
    }

    /// The `mmproj` file that holds the tower of `model`, if one does (None: no file, the
    /// checkpoint's own tower or no vision).
    pub fn file_for(&self, model: &Path) -> Option<PathBuf> {
        match self {
            Mmproj::Auto if crate::gguf::is_gguf(model) => mmproj_for(model),
            Mmproj::File(p) => Some(p.clone()),
            _ => None,
        }
    }
}

/// `clip.projector_type` of llama.cpp mmproj file `path` (`pixtral`, `qwen3vl_merger`, ...); None
/// when it cannot be read or has none.
pub fn projector_type(path: &Path) -> Option<String> {
    let h = crate::gguf::header(path).ok()?;
    h.get("clip.projector_type").and_then(|v| v.as_str()).map(str::to_string)
}

/// The llama.cpp `mmproj*.gguf` vision file beside GGUF model `model` (in its folder or the
/// folder above, as unsloth lays out split models), if there is exactly one.
pub fn mmproj_for(model: &Path) -> Option<PathBuf> {
    let dir = model.parent()?;
    for d in [Some(dir), dir.parent()].into_iter().flatten() {
        let mut v: Vec<PathBuf> = std::fs::read_dir(d)
            .ok()?
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                let n = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                n.starts_with("mmproj") && n.ends_with(".gguf")
            })
            .collect();
        v.sort();
        if v.len() == 1 {
            return v.pop();
        }
    }
    None
}

struct Block<'a> {
    attention_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    q: Lin<'a>,
    k: Lin<'a>,
    v: Lin<'a>,
    o: Lin<'a>,
    gate: Lin<'a>,
    up: Lin<'a>,
    down: Lin<'a>,
}

/// The Pixtral tower and projector (see the module docs).
pub struct PixtralTower<'a> {
    pub cfg: PixtralConfig,
    patch: Lin<'a>, // [H, 3 * p * p]
    ln_pre: Vec<f32>,
    blocks: Vec<Block<'a>>,
    proj_norm: Vec<f32>,
    merger: Lin<'a>, // [H, H * m * m]
    linear_1: Lin<'a>,
    linear_2: Lin<'a>,
    text_hidden: usize,
    /// Query / key rows in interleaved RoPE order (Mistral's layout, GGUF) rather than
    /// rotate-half (transformers' files).
    interleaved: bool,
}

/// Preprocessed image: normalized patches `[ph * pw, 3 * p * p]` (channel, row, column within a
/// patch) on a `ph x pw` grid.
pub struct PixtralImage {
    pub patches: Vec<f32>,
    pub ph: usize,
    pub pw: usize,
}

impl<'a> PixtralTower<'a> {
    /// Loads the tower from `files` (the checkpoint, or an opened mmproj GGUF), producing
    /// `text_hidden`-wide tokens.
    pub fn load(cfg: PixtralConfig, files: &'a ModelFiles, text_hidden: usize) -> Result<Self> {
        let gguf = files.has("v.patch_embd.weight");
        let (h, p, inter) = (cfg.hidden_size, cfg.patch_size, cfg.intermediate_size);
        let mm = cfg.spatial_merge_size * cfg.spatial_merge_size;
        let pre = if files.has("model.vision_tower.ln_pre.weight") {
            "model.vision_tower."
        } else if files.has("vision_tower.ln_pre.weight") {
            "vision_tower."
        } else if gguf {
            ""
        } else {
            bail!("the checkpoint has no Pixtral vision tower");
        };
        let mp = if pre.starts_with("model.") {
            "model.multi_modal_projector."
        } else {
            "multi_modal_projector."
        };
        let name = |hf: &str, gg: &str| if gguf { gg.to_string() } else { format!("{pre}{hf}") };
        let pname = |hf: &str, gg: &str| if gguf { gg.to_string() } else { format!("{mp}{hf}") };
        // Float16 weights (the mmproj file) stay float16 (dense f16 rows, no rounding to bf16).
        let f16 = |n: &str, o: usize, i: usize| -> Result<Option<Lin<'a>>> {
            match files.info(n) {
                Some(t) if t.dtype == "F16" && t.shape.iter().product::<usize>() == o * i => {
                    files.note_use(n, true, false);
                    // SAFETY: the matrix borrows `files`, which outlives the tower.
                    let m = unsafe { crate::ggml::GgufMatrix::borrowed(crate::ggml::GgmlType::F16, files.bytes(n)?, o, i)? };
                    Ok(Some(Lin::gguf(m)))
                }
                _ => Ok(None),
            }
        };
        let lin = |n: String, o: usize, i: usize| -> Result<Lin<'a>> {
            match f16(&n, o, i)? {
                Some(l) => Ok(l),
                None => load_lin(files, &n, o, i, crate::model::Quant::Bf16),
            }
        };
        let vecf = |n: String, len: usize| -> Result<Vec<f32>> {
            let w = files.f32(&n)?;
            if w.len() != len {
                bail!("{n}: {} values, expected {len}", w.len());
            }
            Ok(w)
        };
        let patch = if gguf {
            match f16("v.patch_embd.weight", h, 3 * p * p)? {
                Some(l) => l,
                None => {
                    let (kind, bytes) = files
                        .ggml("v.patch_embd.weight")
                        .context("v.patch_embd.weight: unsupported type")??;
                    Lin::from_gguf(kind, bytes, h, 3 * p * p)?
                }
            }
        } else {
            let n = format!("{pre}patch_conv.weight");
            match files.info(&n).map(|i| i.dtype.as_str()) {
                Some("F16") | Some("F32") => {
                    // Float weights: as f32 rows of a dense matrix (no bf16 rounding).
                    let w = files.f32(&n)?;
                    let bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
                    Lin::gguf(crate::ggml::GgufMatrix::from_bytes(
                        crate::ggml::GgmlType::F32,
                        &bytes,
                        h,
                        3 * p * p,
                    )?)
                }
                _ => Lin::bf16(files.bf16_conv(&n, &[h, 3, p, p])?),
            }
        };
        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let b = format!("transformer.layers.{l}.");
            let g = format!("v.blk.{l}.");
            blocks.push(Block {
                attention_norm: vecf(name(&format!("{b}attention_norm.weight"), &format!("{g}ln1.weight")), h)?,
                ffn_norm: vecf(name(&format!("{b}ffn_norm.weight"), &format!("{g}ln2.weight")), h)?,
                q: lin(name(&format!("{b}attention.q_proj.weight"), &format!("{g}attn_q.weight")), h, h)?,
                k: lin(name(&format!("{b}attention.k_proj.weight"), &format!("{g}attn_k.weight")), h, h)?,
                v: lin(name(&format!("{b}attention.v_proj.weight"), &format!("{g}attn_v.weight")), h, h)?,
                o: lin(name(&format!("{b}attention.o_proj.weight"), &format!("{g}attn_out.weight")), h, h)?,
                gate: lin(
                    name(&format!("{b}feed_forward.gate_proj.weight"), &format!("{g}ffn_gate.weight")),
                    inter,
                    h,
                )?,
                up: lin(
                    name(&format!("{b}feed_forward.up_proj.weight"), &format!("{g}ffn_up.weight")),
                    inter,
                    h,
                )?,
                down: lin(
                    name(&format!("{b}feed_forward.down_proj.weight"), &format!("{g}ffn_down.weight")),
                    h,
                    inter,
                )?,
            });
        }
        let interleaved = gguf || files.metadata("naming") == Some("mistral");
        Ok(Self {
            patch,
            ln_pre: vecf(name("ln_pre.weight", "v.pre_ln.weight"), h)?,
            blocks,
            proj_norm: vecf(pname("norm.weight", "mm.input_norm.weight"), h)?,
            merger: lin(pname("patch_merger.merging_layer.weight", "mm.patch_merger.weight"), h, h * mm)?,
            linear_1: lin(pname("linear_1.weight", "mm.1.weight"), text_hidden, h)?,
            linear_2: lin(pname("linear_2.weight", "mm.2.weight"), text_hidden, text_hidden)?,
            text_hidden,
            interleaved,
            cfg,
        })
    }

    /// The processor's resize target for a `w x h` image: within `image_size` on the longer
    /// side, rounded up to whole merged tokens (`patch_size * spatial_merge_size` pixels).
    pub fn target_size(&self, w: usize, h: usize) -> (usize, usize) {
        let c = &self.cfg;
        let (mut fw, mut fh) = (w as f64, h as f64);
        let ratio = (fh / c.image_size as f64).max(fw / c.image_size as f64);
        if ratio > 1.0 {
            fh = (fh / ratio).floor();
            fw = (fw / ratio).floor();
        }
        let unit = c.patch_size * c.spatial_merge_size;
        let up = |v: f64| ((v as usize).max(1) - 1) / unit * unit + unit;
        (up(fw), up(fh))
    }

    /// Resizes (bicubic, antialiased, as torchvision), rescales and normalizes an RGB image into
    /// patches.
    pub fn preprocess(&self, rgb: &[u8], w: usize, h: usize) -> PixtralImage {
        let c = &self.cfg;
        let (tw, th) = self.target_size(w, h);
        let img = if (tw, th) == (w, h) {
            rgb.to_vec()
        } else {
            crate::resample::resize_bicubic_rgb8_torchvision(rgb, w, h, tw, th)
        };
        let p = c.patch_size;
        let (pw, ph) = (tw / p, th / p);
        let d = 3 * p * p;
        let mut patches = vec![0f32; ph * pw * d];
        for (i, row) in patches.chunks_mut(d).enumerate() {
            let (pr, pc) = (i / pw, i % pw);
            for ch in 0..3 {
                for y in 0..p {
                    for x in 0..p {
                        let v = img[((pr * p + y) * tw + pc * p + x) * 3 + ch] as f32 / 255.0;
                        row[(ch * p + y) * p + x] = (v - c.image_mean[ch]) / c.image_std[ch];
                    }
                }
            }
        }
        PixtralImage { patches, ph, pw }
    }

    /// The image's soft tokens `[ph/m * pw/m, text_hidden]`, row-major.
    pub fn encode(&self, img: &PixtralImage) -> Vec<f32> {
        let c = &self.cfg;
        let (h, nh, hd) = (c.hidden_size, c.num_attention_heads, c.head_dim());
        let (ph, pw) = (img.ph, img.pw);
        let n = ph * pw;
        let eps = 1e-5;
        let d = 3 * c.patch_size * c.patch_size;
        let mut x = vec![0f32; n * h];
        linear(&self.patch, h, d, &img.patches, n, &mut x);
        rmsnorm_f32w_rows_inplace(&mut x, h, &self.ln_pre, eps);

        // Axial angles per patch: rotary pair j < hd/4 turns with the row at frequency 2j,
        // the others with the column at frequency 2(j - hd/4) + 1 (of hd/2 frequencies).
        let half = hd / 2;
        let inv: Vec<f32> = (0..half)
            .map(|i| 1.0 / (c.rope_theta as f32).powf((2 * i) as f32 / hd as f32))
            .collect();
        let mut cos = vec![0f32; n * half];
        let mut sin = vec![0f32; n * half];
        for i in 0..n {
            let (r, col) = ((i / pw) as f32, (i % pw) as f32);
            for j in 0..half {
                let a = if j < half / 2 {
                    r * inv[2 * j]
                } else {
                    col * inv[2 * (j - half / 2) + 1]
                };
                cos[i * half + j] = a.cos();
                sin[i * half + j] = a.sin();
            }
        }

        let inter = c.intermediate_size;
        let mut hn = vec![0f32; n * h];
        let mut y = vec![0f32; n * h];
        let mut g = vec![0f32; n * inter];
        let mut u = vec![0f32; n * inter];
        let (mut q, mut k, mut v) = (vec![0f32; n * h], vec![0f32; n * h], vec![0f32; n * h]);
        for b in &self.blocks {
            rmsnorm_f32w_rows(&x, h, &b.attention_norm, eps, &mut hn);
            crate::model::linear_multi(
                &mut [(&b.q, h, &mut q[..]), (&b.k, h, &mut k[..]), (&b.v, h, &mut v[..])],
                h,
                &hn,
                n,
            );
            for buf in [&mut q, &mut k] {
                for_chunks(buf, hd, true, |i, head| {
                    let j = i / nh;
                    let (cs, sn) = (&cos[j * half..(j + 1) * half], &sin[j * half..(j + 1) * half]);
                    if self.interleaved {
                        apply_rope_interleaved(head, cs, sn);
                    } else {
                        apply_rope(head, cs, sn);
                    }
                });
            }
            let att = crate::sd::ops::attention(&q, &k, &v, n, n, nh, hd, false);
            linear(&b.o, h, h, &att, n, &mut y);
            add_inplace(&mut x, &y);
            rmsnorm_f32w_rows(&x, h, &b.ffn_norm, eps, &mut hn);
            crate::model::linear_multi(&mut [(&b.gate, inter, &mut g[..]), (&b.up, inter, &mut u[..])], h, &hn, n);
            silu_mul(&mut g, &u);
            linear(&b.down, h, inter, &g, n, &mut y);
            add_inplace(&mut x, &y);
        }

        // Projector: norm, 2x2 patch merge (channel-major unfold), GELU MLP.
        rmsnorm_f32w_rows_inplace(&mut x, h, &self.proj_norm, eps);
        let m = c.spatial_merge_size;
        let (mh, mw) = (ph / m, pw / m);
        let t = mh * mw;
        let md = h * m * m;
        let mut merged = vec![0f32; t * md];
        for_chunks(&mut merged, md, true, |ti, row| {
            let (r, col) = (ti / mw, ti % mw);
            for ki in 0..m {
                for kj in 0..m {
                    let src = &x[((r * m + ki) * pw + col * m + kj) * h..][..h];
                    for (ch, &val) in src.iter().enumerate() {
                        row[ch * m * m + ki * m + kj] = val;
                    }
                }
            }
        });
        let mut mo = vec![0f32; t * h];
        linear(&self.merger, h, md, &merged, t, &mut mo);
        let th = self.text_hidden;
        let mut a = vec![0f32; t * th];
        linear(&self.linear_1, th, h, &mo, t, &mut a);
        for_chunks(&mut a, th, true, |_, r| {
            for v in r.iter_mut() {
                *v = crate::bert::gelu_erf(*v);
            }
        });
        let mut out = vec![0f32; t * th];
        linear(&self.linear_2, th, th, &a, t, &mut out);
        out
    }

    /// The prompt text of an image of `ph x pw` patches: rows of `[IMG]`, `[IMG_BREAK]` after each
    /// row but the last, which ends with `[IMG_END]`.
    pub fn placeholder(&self, img: &PixtralImage) -> String {
        let m = self.cfg.spatial_merge_size;
        let (rows, cols) = (img.ph / m, img.pw / m);
        let mut s = String::new();
        for r in 0..rows {
            s.push_str(&"[IMG]".repeat(cols));
            s.push_str(if r + 1 == rows { "[IMG_END]" } else { "[IMG_BREAK]" });
        }
        s
    }
}
