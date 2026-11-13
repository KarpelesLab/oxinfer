//! Text embeddings for sentence-transformers style models: a backbone (Gemma 3 / EmbeddingGemma,
//! Qwen3 / Qwen3-Embedding, Qwen2 / gte-Qwen2 and stella, BERT-family encoders such as BGE-M3)
//! followed by the modules listed in `modules.json`: pooling, dense projections, normalization.
//! Prompts (`query`, `document`, ...) come from `config_sentence_transformers.json`. BGE-M3's
//! sparse (lexical weights) and multi-vector (ColBERT) heads are loaded when their files are
//! present; so are stella's alternative projection heads (`2_Dense_256`, ..., one per output
//! size).

use crate::bert::{BertConfig, BertModel};
use crate::config::{QwenConfig, TextConfig};
use crate::embed_gemma2::{EmbGemma2Config, EmbGemma2Model};
use crate::model::{Model, Quant};
use crate::modernbert::{ModernBertConfig, ModernBertModel};
use crate::mpnet::{MpnetConfig, MpnetModel};
use crate::nomic_bert::{NomicBertConfig, NomicBertModel};
use crate::qwen::QwenModel;
use crate::safetensors::{ModelFiles, SafeTensors};
use crate::tokenizer::Tokenizer;
use anyhow::{Context, Result, anyhow, bail};
use std::path::Path;

enum Backbone<'a> {
    Gemma(Box<Model<'a>>),
    /// EmbeddingGemma 2 (`crate::embed_gemma2`): its rows are already projected to the output size.
    Gemma2(Box<EmbGemma2Model<'a>>),
    Qwen(Box<QwenModel<'a>>),
    Bert(Box<BertModel<'a>>),
    Mpnet(Box<MpnetModel<'a>>),
    Nomic(Box<NomicBertModel<'a>>),
    ModernBert(Box<ModernBertModel<'a>>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pooling {
    Cls,
    Mean,
    Max,
    MeanSqrtLen,
    LastToken,
}

#[derive(Clone, Copy, Debug)]
enum Act {
    Identity,
    Tanh,
}

/// A sentence-transformers `Dense` module (f32, applied to one pooled vector).
struct Dense {
    w: Vec<f32>, // [n_out, n_in]
    b: Option<Vec<f32>>,
    act: Act,
    n_in: usize,
    n_out: usize,
}

impl Dense {
    fn apply(&self, x: &[f32]) -> Vec<f32> {
        (0..self.n_out)
            .map(|o| {
                let mut v = crate::kernels::dot(&self.w[o * self.n_in..(o + 1) * self.n_in], x);
                if let Some(b) = &self.b {
                    v += b[o];
                }
                match self.act {
                    Act::Identity => v,
                    Act::Tanh => v.tanh(),
                }
            })
            .collect()
    }
}

/// One text's embeddings: the dense vector, plus BGE-M3's lexical weights (token id, weight; one
/// entry per distinct token) and multi-vector embeddings (one per token after the first) when
/// the model has those heads.
#[derive(Debug, Clone)]
pub struct Embedding {
    pub dense: Vec<f32>,
    pub sparse: Option<Vec<(u32, f32)>>,
    pub colbert: Option<Vec<Vec<f32>>>,
}

impl Embedding {
    /// Matryoshka truncation: keeps the first `dim` values of the dense vector and renormalizes.
    pub fn truncate(&mut self, dim: usize) -> Result<()> {
        if dim == 0 || dim > self.dense.len() {
            bail!("cannot truncate a {}-dimensional embedding to {dim}", self.dense.len());
        }
        self.dense.truncate(dim);
        l2_normalize(&mut self.dense);
        Ok(())
    }
}

pub struct Embedder<'a> {
    backbone: Backbone<'a>,
    pub tok: Tokenizer,
    pub pooling: Pooling,
    include_prompt: bool,
    dense: Vec<Dense>,
    pub normalize: bool,
    /// Named prompts prepended to the text, and the one used by default.
    pub prompts: Vec<(String, String)>,
    pub default_prompt: Option<String>,
    pub max_len: usize,
    /// Other projection heads that can stand in for the last dense module, by output size
    /// (stella: `2_Dense_256` ... `2_Dense_8192` beside the one `modules.json` names), read on
    /// first use.
    heads: Vec<Head>,
    /// A token appended to every input: the tokenizer's `add_eos_token` when `tokenizer.json`
    /// has no template for it (gte-Qwen2, whose own tokenizer class adds the template at load).
    append: Option<u32>,
    sparse: Option<(Vec<f32>, f32)>,
    colbert: Option<Dense>,
    /// Tokens left out of the lexical weights (CLS, EOS, PAD, UNK).
    unused: Vec<u32>,
}

fn read_json(path: &Path) -> Option<serde_json::Value> {
    serde_json::from_slice(&crate::fsx::read(path).ok()?).ok()
}

/// A float tensor of a standalone file as f32 (small heads: `sparse_linear.pt`, dense modules).
fn tensor_f32(st: &SafeTensors, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let info = st.info(name).with_context(|| format!("missing tensor {name}"))?.clone();
    let b = st.bytes(name)?;
    let v = match info.dtype.as_str() {
        "F32" => b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect(),
        "BF16" => b
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| crate::kernels::bf16_to_f32(u16::from_le_bytes(*c)))
            .collect(),
        "F16" => b.as_chunks::<2>().0.iter().map(|c| half::f16::from_le_bytes(*c).to_f32()).collect(),
        d => bail!("tensor {name}: dtype {d}"),
    };
    Ok((v, info.shape))
}

/// `nn.Linear` weights from a file holding `{prefix}weight` and maybe `{prefix}bias`.
fn load_linear(path: &Path, prefix: &str, act: Act) -> Result<Dense> {
    let st = SafeTensors::open(path, true)?;
    let (w, shape) = tensor_f32(&st, &format!("{prefix}weight"))?;
    if shape.len() != 2 {
        bail!("{}: weight shape {shape:?}", path.display());
    }
    let b = if st.info(&format!("{prefix}bias")).is_some() {
        Some(tensor_f32(&st, &format!("{prefix}bias"))?.0)
    } else {
        None
    };
    Ok(Dense {
        w,
        b,
        act,
        n_out: shape[0],
        n_in: shape[1],
    })
}

/// The weights file of a sentence-transformers module folder.
fn module_weights(dir: &Path) -> Result<std::path::PathBuf> {
    ["model.safetensors", "pytorch_model.bin"]
        .iter()
        .map(|f| dir.join(f))
        .find(|p| crate::fsx::exists(p))
        .with_context(|| format!("{}: no model.safetensors or pytorch_model.bin", dir.display()))
}

/// Whether the Qwen2 checkpoint in `dir` runs without the causal mask: the modeling code that
/// gte-Qwen2 and stella ship (`modeling_qwen.py`) makes `Qwen2Model.forward` bidirectional unless
/// it is asked otherwise, which sentence-transformers never does; gte-Qwen2 also says so in its
/// config (`is_causal: false`). Any other Qwen2 embedder is the stock causal decoder.
fn qwen2_bidirectional(dir: &Path) -> bool {
    let cfg = read_json(&dir.join("config.json")).unwrap_or_default();
    cfg["is_causal"].as_bool() == Some(false)
        || cfg["auto_map"]["AutoModel"]
            .as_str()
            .is_some_and(|m| m.ends_with("modeling_qwen.Qwen2Model"))
}

/// An alternative projection head: its output size, its weights file, and the weights once read.
type Head = (usize, std::path::PathBuf, std::sync::OnceLock<Dense>);

/// `NAME_N` module folders beside the dense module `path` = `NAME_M` (stella's `2_Dense_256`,
/// `2_Dense_768`, ...): projections of the same input to `N` outputs.
fn sibling_heads(dir: &Path, path: &str, n_in: usize, n_out: usize) -> Vec<Head> {
    let Some((stem, m)) = path.rsplit_once('_') else {
        return Vec::new();
    };
    if m.parse::<usize>().ok() != Some(n_out) {
        return Vec::new();
    }
    let mut out = Vec::new();
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let Some(n) = name
            .strip_prefix(stem)
            .and_then(|r| r.strip_prefix('_'))
            .and_then(|n| n.parse::<usize>().ok())
        else {
            continue;
        };
        let c = read_json(&e.path().join("config.json")).unwrap_or_default();
        if n == n_out || c["in_features"].as_u64() != Some(n_in as u64) || c["out_features"].as_u64() != Some(n as u64) {
            continue;
        }
        if let Ok(w) = module_weights(&e.path()) {
            out.push((n, w, std::sync::OnceLock::new()));
        }
    }
    out.sort_by_key(|h| h.0);
    out
}

/// Whether `dir` holds a sentence-transformers model (or another embedding-only checkpoint).
pub fn is_embedding_model(dir: &Path) -> bool {
    crate::fsx::exists(&dir.join("modules.json"))
}

impl<'a> Embedder<'a> {
    pub fn load(dir: &Path, files: &'a ModelFiles, quant: Quant) -> Result<Self> {
        let model_type = crate::config::model_type(dir)?;
        let backbone = if model_type.starts_with("qwen3") || model_type == "qwen2" {
            let mut cfg = QwenConfig::load(dir)?;
            cfg.bidirectional = model_type == "qwen2" && qwen2_bidirectional(dir);
            Backbone::Qwen(Box::new(QwenModel::load(cfg, files, quant)?))
        } else if model_type == "embedding_gemma2" {
            Backbone::Gemma2(Box::new(EmbGemma2Model::load(EmbGemma2Config::load(dir)?, files, quant)?))
        } else if model_type.starts_with("gemma") {
            Backbone::Gemma(Box::new(Model::load(TextConfig::load(dir)?, files, quant)?))
        } else if crate::bert::is_bert(&model_type) {
            Backbone::Bert(Box::new(BertModel::load(BertConfig::load(dir)?, files, quant)?))
        } else if model_type == "mpnet" {
            Backbone::Mpnet(Box::new(MpnetModel::load(MpnetConfig::load(dir)?, files, quant)?))
        } else if model_type == "nomic_bert" {
            Backbone::Nomic(Box::new(NomicBertModel::load(NomicBertConfig::load(dir)?, files, quant)?))
        } else if crate::modernbert::is_modernbert(&model_type) {
            Backbone::ModernBert(Box::new(ModernBertModel::load(ModernBertConfig::load(dir)?, files, quant)?))
        } else {
            bail!("model_type {model_type} is not supported for embeddings");
        };
        let hidden = match &backbone {
            Backbone::Gemma(m) => m.cfg.hidden_size,
            Backbone::Gemma2(m) => m.cfg.hidden_size,
            Backbone::Qwen(m) => m.cfg.hidden_size,
            Backbone::Bert(m) => m.cfg.hidden_size,
            Backbone::Mpnet(m) => m.cfg.hidden_size,
            Backbone::Nomic(m) => m.cfg.hidden_size,
            Backbone::ModernBert(m) => m.cfg.hidden_size,
        };

        // Modules after the transformer, in order.
        let mut pooling = Pooling::Mean;
        let mut include_prompt = true;
        let mut dense = Vec::new();
        let mut heads = Vec::new();
        let mut normalize = false;
        let modules = read_json(&dir.join("modules.json")).unwrap_or_default();
        for m in modules.as_array().into_iter().flatten() {
            let ty = m["type"].as_str().unwrap_or_default();
            let mdir = dir.join(m["path"].as_str().unwrap_or_default());
            match ty.rsplit('.').next().unwrap_or_default() {
                "Transformer" => {}
                "Pooling" => {
                    let c = read_json(&mdir.join("config.json")).context("pooling config")?;
                    // sentence-transformers 6 writes one `pooling_mode` name, older ones a flag each.
                    let mode = c["pooling_mode"].as_str();
                    let on = |k: &str| {
                        c[k].as_bool().unwrap_or(false)
                            || mode.is_some_and(|m| {
                                k.strip_prefix("pooling_mode_")
                                    .is_some_and(|f| f == m || f.strip_suffix("_token").or(f.strip_suffix("_tokens")) == Some(m))
                            })
                    };
                    pooling = if on("pooling_mode_cls_token") {
                        Pooling::Cls
                    } else if on("pooling_mode_lasttoken") {
                        Pooling::LastToken
                    } else if on("pooling_mode_max_tokens") {
                        Pooling::Max
                    } else if on("pooling_mode_mean_sqrt_len_tokens") {
                        Pooling::MeanSqrtLen
                    } else if on("pooling_mode_mean_tokens") {
                        Pooling::Mean
                    } else {
                        bail!("{}: unsupported pooling mode", mdir.display());
                    };
                    include_prompt = c["include_prompt"].as_bool().unwrap_or(true);
                }
                "Dense" => {
                    let c = read_json(&mdir.join("config.json")).context("dense config")?;
                    let act = match c["activation_function"].as_str().unwrap_or("torch.nn.modules.linear.Identity") {
                        a if a.ends_with("Identity") => Act::Identity,
                        a if a.ends_with("Tanh") => Act::Tanh,
                        a => bail!("dense activation {a} is not supported"),
                    };
                    let d = load_linear(&module_weights(&mdir)?, "linear.", act)?;
                    let want_in = dense.last().map_or(hidden, |d: &Dense| d.n_out);
                    if d.n_in != want_in {
                        bail!("{}: dense input {} does not follow {want_in}", mdir.display(), d.n_in);
                    }
                    heads = sibling_heads(dir, m["path"].as_str().unwrap_or_default(), want_in, d.n_out);
                    dense.push(d);
                }
                "Normalize" => normalize = true,
                other => bail!("sentence-transformers module {other} is not supported"),
            }
        }

        let st = read_json(&dir.join("config_sentence_transformers.json")).unwrap_or_default();
        let prompts: Vec<(String, String)> = st["prompts"]
            .as_object()
            .map(|o| {
                o.iter()
                    .map(|(k, v)| (k.clone(), v.as_str().unwrap_or_default().to_string()))
                    .collect()
            })
            .unwrap_or_default();
        let default_prompt = st["default_prompt_name"].as_str().map(str::to_string);
        let cfg = read_json(&dir.join("config.json")).unwrap_or_default();
        let max_len = read_json(&dir.join("sentence_bert_config.json"))
            .and_then(|c| c["max_seq_length"].as_u64())
            .or_else(|| cfg["max_position_embeddings"].as_u64())
            .or_else(|| cfg["text_config"]["max_position_embeddings"].as_u64())
            .unwrap_or(512) as usize;

        let mut tok = Tokenizer::from_bytes(crate::fsx::read(&dir.join("tokenizer.json"))?).map_err(|e| anyhow!("tokenizer: {e}"))?;
        tok.with_truncation(Some(crate::tokenizer::TruncationParams {
            max_length: max_len,
            ..Default::default()
        }))
        .map_err(|e| anyhow!("tokenizer truncation: {e}"))?;
        let _ = tok.with_padding(None);
        let tc = read_json(&dir.join("tokenizer_config.json")).unwrap_or_default();
        let append = if tc["add_eos_token"].as_bool() == Some(true) {
            let eos = &tc["eos_token"];
            let eos = eos
                .as_str()
                .or_else(|| eos["content"].as_str())
                .context("add_eos_token without an eos_token")?;
            let id = tok
                .token_to_id(eos)
                .with_context(|| format!("eos token {eos} is not in the vocabulary"))?;
            let probe = tok.encode("a", true).map_err(|e| anyhow!("encode: {e}"))?;
            (probe.get_ids().last() != Some(&id)).then_some(id)
        } else {
            None
        };

        // BGE-M3 heads.
        let sparse = match dir.join("sparse_linear.pt") {
            p if crate::fsx::exists(&p) => {
                let d = load_linear(&p, "", Act::Identity)?;
                if d.n_out != 1 || d.n_in != hidden {
                    bail!("sparse_linear.pt: shape [{}, {}]", d.n_out, d.n_in);
                }
                Some((d.w, d.b.map_or(0.0, |b| b[0])))
            }
            _ => None,
        };
        let colbert = match dir.join("colbert_linear.pt") {
            p if crate::fsx::exists(&p) => Some(load_linear(&p, "", Act::Identity)?),
            _ => None,
        };
        let special = |k: &str| -> Option<u32> {
            let tc = read_json(&dir.join("tokenizer_config.json"))?;
            let s = match &tc[k] {
                serde_json::Value::String(s) => s.clone(),
                v => v["content"].as_str()?.to_string(),
            };
            tok.token_to_id(&s)
        };
        let unused = ["cls_token", "eos_token", "pad_token", "unk_token"]
            .iter()
            .filter_map(|k| special(k))
            .collect();

        Ok(Self {
            backbone,
            tok,
            pooling,
            include_prompt,
            dense,
            normalize,
            prompts,
            default_prompt,
            max_len,
            heads,
            append,
            sparse,
            colbert,
            unused,
        })
    }

    /// Output sizes of the model's alternative projection heads (empty for most models).
    pub fn head_dims(&self) -> Vec<usize> {
        self.heads.iter().map(|h| h.0).collect()
    }

    /// Output dimension of the dense embedding.
    pub fn dim(&self) -> usize {
        match (self.dense.last(), &self.backbone) {
            (Some(d), _) => d.n_out,
            (None, Backbone::Gemma(m)) => m.cfg.hidden_size,
            (None, Backbone::Gemma2(m)) => m.cfg.hidden_size,
            (None, Backbone::Qwen(m)) => m.cfg.hidden_size,
            (None, Backbone::Bert(m)) => m.cfg.hidden_size,
            (None, Backbone::Mpnet(m)) => m.cfg.hidden_size,
            (None, Backbone::Nomic(m)) => m.cfg.hidden_size,
            (None, Backbone::ModernBert(m)) => m.cfg.hidden_size,
        }
    }

    pub fn has_sparse(&self) -> bool {
        self.sparse.is_some()
    }
    pub fn has_colbert(&self) -> bool {
        self.colbert.is_some()
    }

    /// The lexical weights of `e` keyed by token text (as FlagEmbedding reports them).
    pub fn sparse_tokens(&self, e: &Embedding) -> Option<Vec<(String, f32)>> {
        let s = e.sparse.as_ref()?;
        Some(
            s.iter()
                .map(|&(id, w)| (self.tok.id_to_token(id).unwrap_or_else(|| id.to_string()), w))
                .collect(),
        )
    }

    /// The text of prompt `name` (`query`, `document`, ...), if the model defines it.
    pub fn prompt(&self, name: &str) -> Option<&str> {
        self.prompts.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }

    /// Final hidden states `[t, hidden]` of `ids`.
    fn hidden(&self, ids: &[u32]) -> Vec<f32> {
        match &self.backbone {
            Backbone::Gemma(m) => {
                let mut s = m.new_session();
                m.forward_hidden(&mut s, ids)
            }
            Backbone::Gemma2(m) => m.forward(ids),
            Backbone::Qwen(m) => {
                let mut s = m.new_session();
                m.forward_hidden_mm(&mut s, ids, &[])
            }
            Backbone::Bert(m) => m.forward(ids),
            Backbone::Mpnet(m) => m.forward(ids),
            Backbone::Nomic(m) => m.forward(ids),
            Backbone::ModernBert(m) => m.forward(ids),
        }
    }

    /// Whether the model embeds images and audio too (EmbeddingGemma 2).
    pub fn multimodal(&self) -> bool {
        matches!(self.backbone, Backbone::Gemma2(_))
    }

    /// One embedding of the attachments of `req` (images and audio clips; EmbeddingGemma 2,
    /// whose checkpoint `files` holds Gemma 4's vision and audio towers) followed by `text`
    /// (which may be empty), with prompt `prompt_name`: the attachments' soft tokens stand where
    /// the processor writes them (`<|image>`, one `<|image|>` per soft token, `<image|>`; audio
    /// alike), and the pooling covers them as it covers text tokens.
    pub fn embed_media(&self, req: &crate::mm::MmRequest, files: &ModelFiles, text: &str, prompt_name: Option<&str>) -> Result<Embedding> {
        let Backbone::Gemma2(m) = &self.backbone else {
            bail!("this embedding model takes text only");
        };
        if self.pooling != Pooling::Mean || !self.dense.is_empty() {
            bail!("attachments are embedded with mean pooling and no dense module");
        }
        // Its video processor lays frames out its own way (`<|video|>` soft tokens, no timestamps).
        if !req.videos.is_empty() {
            bail!("video embeddings are not supported (images and audio are)");
        }
        let prompt = match prompt_name.or(self.default_prompt.as_deref()) {
            Some(n) => self.prompt(n).with_context(|| format!("the model has no prompt named {n}"))?,
            None => "",
        };
        let width = m.cfg.width;
        let (mm_text, feats) = crate::mm::prepare(req, files, width)?;
        let full = format!("{prompt}{mm_text}{text}");
        let enc = self.tok.encode(full.as_str(), true).map_err(|e| anyhow!("encode: {e}"))?;
        let ids = enc.get_ids();
        let mut x = m.embed(ids);
        for (pos, emb) in crate::mm::overrides(ids, &feats, width)? {
            x[pos * width..(pos + 1) * width].copy_from_slice(&emb);
        }
        let hs = m.forward_embeds(x);
        let h = m.cfg.hidden_size;
        let mut v = vec![0f32; h];
        for r in hs.chunks(h) {
            crate::kernels::add_inplace(&mut v, r);
        }
        v.iter_mut().for_each(|x| *x /= ids.len() as f32);
        if self.normalize {
            l2_normalize(&mut v);
        }
        Ok(Embedding {
            dense: v,
            sparse: None,
            colbert: None,
        })
    }

    /// Embeds `text` with prompt `prompt_name` (None: the model's default prompt, if any).
    pub fn embed(&self, text: &str, prompt_name: Option<&str>) -> Result<Embedding> {
        self.embed_dim(text, prompt_name, None)
    }

    /// [`Self::embed`] with `dim` output dimensions: a model with one projection head per output
    /// size (stella) uses the head of that size; any other model truncates the vector to its
    /// first `dim` values and renormalizes (Matryoshka, [`Embedding::truncate`]).
    pub fn embed_dim(&self, text: &str, prompt_name: Option<&str>, dim: Option<usize>) -> Result<Embedding> {
        let head = match dim {
            Some(d) if !self.heads.is_empty() && d != self.dim() => {
                let Some((n_out, path, cell)) = self.heads.iter().find(|h| h.0 == d) else {
                    let mut dims = self.head_dims();
                    dims.push(self.dim());
                    dims.sort_unstable();
                    bail!("the model has no {d}-dimensional head (it has {dims:?})");
                };
                if cell.get().is_none() {
                    let last = self.dense.last().unwrap();
                    let d = load_linear(path, "linear.", last.act)?;
                    if d.n_in != last.n_in || d.n_out != *n_out {
                        bail!("{}: a [{}, {}] projection", path.display(), d.n_out, d.n_in);
                    }
                    let _ = cell.set(d);
                }
                cell.get()
            }
            _ => None,
        };
        let mut e = self.embed_head(text, prompt_name, head)?;
        if let Some(d) = dim
            && d != e.dense.len()
        {
            e.truncate(d)?;
        }
        Ok(e)
    }

    /// Embeds `text`; `head` replaces the last dense module.
    fn embed_head(&self, text: &str, prompt_name: Option<&str>, head: Option<&Dense>) -> Result<Embedding> {
        let name = prompt_name.or(self.default_prompt.as_deref());
        let prompt = match name {
            Some(n) => self.prompt(n).with_context(|| format!("the model has no prompt named {n}"))?,
            None => "",
        };
        let full = format!("{prompt}{text}");
        let enc = self.tok.encode(full.as_str(), true).map_err(|e| anyhow!("encode: {e}"))?;
        let mut ids: Vec<u32> = enc.get_ids().to_vec();
        if let Some(eos) = self.append {
            // As the template would: the text is cut to leave room for the token.
            ids.truncate(self.max_len.saturating_sub(1));
            ids.push(eos);
        }
        if ids.is_empty() {
            bail!("empty input");
        }
        // Prompt tokens to skip when pooling without them (sentence-transformers counts the
        // prompt tokenized alone, minus a trailing special token).
        let skip = if !self.include_prompt && !prompt.is_empty() {
            let p = self.tok.encode(prompt, true).map_err(|e| anyhow!("encode: {e}"))?;
            let n = p.get_ids().len();
            let trailing = p.get_special_tokens_mask().last().copied().unwrap_or(0) as usize;
            n - trailing
        } else {
            0
        };
        let hs = self.hidden(&ids);
        let h = hs.len() / ids.len();
        let t = ids.len();
        let rows = || hs.chunks(h).enumerate().skip(skip);
        let mut v = match self.pooling {
            Pooling::Cls => hs[..h].to_vec(),
            Pooling::LastToken => hs[(t - 1) * h..].to_vec(),
            Pooling::Mean | Pooling::MeanSqrtLen => {
                let mut acc = vec![0f32; h];
                let mut n = 0usize;
                for (_, r) in rows() {
                    crate::kernels::add_inplace(&mut acc, r);
                    n += 1;
                }
                let d = if self.pooling == Pooling::Mean {
                    n as f32
                } else {
                    (n as f32).sqrt()
                };
                acc.iter_mut().for_each(|x| *x /= d.max(1.0));
                acc
            }
            Pooling::Max => {
                let mut acc = vec![f32::NEG_INFINITY; h];
                for (_, r) in rows() {
                    for (a, &x) in acc.iter_mut().zip(r) {
                        *a = a.max(x);
                    }
                }
                acc
            }
        };
        let last = self.dense.len().saturating_sub(1);
        for (i, d) in self.dense.iter().enumerate() {
            v = match head {
                Some(h) if i == last => h.apply(&v),
                _ => d.apply(&v),
            };
        }
        if self.normalize {
            l2_normalize(&mut v);
        }
        let sparse = self.sparse.as_ref().map(|(w, b)| {
            let mut out: Vec<(u32, f32)> = Vec::new();
            for (i, r) in hs.chunks(h).enumerate() {
                if self.unused.contains(&ids[i]) {
                    continue;
                }
                let x = (crate::kernels::dot(w, r) + b).max(0.0);
                match out.iter_mut().find(|(id, _)| *id == ids[i]) {
                    Some(e) => e.1 = e.1.max(x),
                    None => out.push((ids[i], x)),
                }
            }
            out.retain(|&(_, x)| x > 0.0);
            out.sort_by_key(|&(id, _)| id);
            out
        });
        let colbert = self.colbert.as_ref().map(|c| {
            hs.chunks(h)
                .skip(1)
                .map(|r| {
                    let mut v = c.apply(r);
                    l2_normalize(&mut v);
                    v
                })
                .collect()
        });
        Ok(Embedding { dense: v, sparse, colbert })
    }
}

pub fn l2_normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.iter_mut().for_each(|x| *x /= n);
    }
}

/// Cosine similarity.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let d = crate::kernels::dot(a, b);
    let n = (crate::kernels::dot(a, a) * crate::kernels::dot(b, b)).sqrt();
    if n > 0.0 { d / n } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// stella's layout: `modules.json` names `2_Dense_1024`; the other `2_Dense_N` folders with the
    /// same input are alternative heads, anything else (another input size, no weights, the
    /// unsuffixed copy) is not.
    #[test]
    fn sibling_heads_by_output_size() {
        let dir = std::env::temp_dir().join(format!("oxinfer-embed-heads-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let module = |name: &str, n_in: usize, n_out: usize, weights: bool| {
            let d = dir.join(name);
            std::fs::create_dir_all(&d).unwrap();
            let c = serde_json::json!({"in_features": n_in, "out_features": n_out, "bias": true});
            std::fs::write(d.join("config.json"), c.to_string()).unwrap();
            if weights {
                std::fs::write(d.join("model.safetensors"), b"").unwrap();
            }
        };
        module("2_Dense", 16, 64, true);
        module("2_Dense_8", 16, 8, true);
        module("2_Dense_32", 16, 32, true);
        module("2_Dense_64", 16, 64, true);
        module("2_Dense_12", 24, 12, true);
        module("2_Dense_48", 16, 48, false);
        let dims = |path: &str, n_out| sibling_heads(&dir, path, 16, n_out).iter().map(|h| h.0).collect::<Vec<_>>();
        assert_eq!(dims("2_Dense_32", 32), [8, 64]);
        assert!(dims("2_Dense", 64).is_empty());
        assert!(dims("3_Dense_32", 32).is_empty());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
