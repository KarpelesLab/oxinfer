//! The whitelist's models: kind detection, lazy loading, one request at a time per model, and
//! least-recently-used unloading under the memory budget.

use super::{ApiError, ApiResult, ModelSpec};
use crate::engine::{Engine, EngineOptions};
use crate::model::Quant;
use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};

/// What a model does (which endpoints take it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Text generation: `/v1/chat/completions`, `/v1/completions`.
    Chat,
    /// `/v1/embeddings`.
    Embedding,
    /// `/v1/images/generations`, `/v1/images/edits`.
    Image,
    /// Speech to text: `/v1/audio/transcriptions`, `/v1/audio/translations`.
    Transcription,
    /// Text to speech: `/v1/audio/speech`.
    Speech,
}

impl Kind {
    pub fn parse(s: &str) -> Option<Kind> {
        Some(match s {
            "chat" | "text" | "llm" => Kind::Chat,
            "embedding" | "embeddings" | "embed" => Kind::Embedding,
            "image" | "images" => Kind::Image,
            "transcription" | "asr" | "whisper" | "stt" => Kind::Transcription,
            "speech" | "tts" => Kind::Speech,
            _ => return None,
        })
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Chat => "chat",
            Kind::Embedding => "embedding",
            Kind::Image => "image",
            Kind::Transcription => "transcription",
            Kind::Speech => "speech",
        }
    }
}

/// The image pipelines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ImageKind {
    Sd,
    Flux,
    QwenImage,
    OmniGen2,
    Step1x,
    Bagel,
    Janus,
}

/// An embedding model with the checkpoint it borrows.
pub(crate) struct EmbedModel {
    /// Declared first: dropped before the files it points into.
    pub emb: crate::embed::Embedder<'static>,
    _files: Box<crate::safetensors::ModelFiles>,
}

// Boxed as a whole in `Loaded::Image`.
#[allow(clippy::large_enum_variant)]
pub(crate) enum ImageModel {
    Sd(crate::sd::Pipeline),
    Flux(crate::flux::Pipeline),
    QwenImage(crate::qwen_image::Pipeline),
    OmniGen2(crate::omnigen2::Pipeline),
    Step1x(crate::step1x::Pipeline),
    Bagel(crate::bagel::Bagel),
    Janus(crate::janus::Janus),
}

/// A loaded model.
pub(crate) enum Loaded {
    Chat(Box<Engine>),
    Embedding(Box<EmbedModel>),
    Image(Box<ImageModel>),
    Transcription(Box<crate::whisper::Transcriber>),
    Speech(Box<crate::tts::Speaker>),
}

impl Loaded {
    fn kind(&self) -> Kind {
        match self {
            Loaded::Chat(_) => Kind::Chat,
            Loaded::Embedding(_) => Kind::Embedding,
            Loaded::Image(_) => Kind::Image,
            Loaded::Transcription(_) => Kind::Transcription,
            Loaded::Speech(_) => Kind::Speech,
        }
    }
}

/// One model of the whitelist.
pub(crate) struct Entry {
    pub spec: ModelSpec,
    /// Held for the whole of a request: requests to one model wait their turn.
    slot: Mutex<Option<Loaded>>,
    /// The kind and the local path, once known.
    detected: Mutex<Option<(Kind, Option<ImageKind>)>>,
    /// Registry clock of the last use.
    last_used: AtomicU64,
    /// Memory counted for the model while loaded (0: not loaded).
    bytes: AtomicU64,
}

pub(crate) struct Registry {
    entries: Vec<Entry>,
    budget: Option<u64>,
    cache_dir: Option<PathBuf>,
    /// The server-wide prompt cache of chat models (a model's own `prompt_cache` overrides it).
    pub prompt_cache: Option<PathBuf>,
    pub prompt_cache_gb: Option<f64>,
    clock: AtomicU64,
    /// Serializes loads (and the memory accounting around them).
    loading: Mutex<()>,
    log: bool,
}

/// A model in use: the lock on its slot, held until dropped.
pub(crate) struct Lease<'a> {
    pub entry: &'a Entry,
    guard: MutexGuard<'a, Option<Loaded>>,
}

impl Lease<'_> {
    pub fn model(&mut self) -> &mut Loaded {
        self.guard.as_mut().expect("a lease holds a loaded model")
    }
    pub fn spec(&self) -> &ModelSpec {
        &self.entry.spec
    }
}

/// Server-side model keys (not passed to the loaders).
fn server_key(k: &str) -> bool {
    ModelSpec::KEYS.contains(&k)
}

impl Registry {
    pub fn new(specs: &[ModelSpec], budget: Option<u64>, cache_dir: Option<PathBuf>, log: bool) -> Self {
        let entries = specs
            .iter()
            .map(|s| Entry {
                spec: s.clone(),
                slot: Mutex::new(None),
                // Models on disk already: their kind for the listing, without fetching anything.
                detected: Mutex::new(local_path(&s.source, cache_dir.as_deref()).and_then(|p| {
                    let d = detect_path(&p, &s.source);
                    match s.option("kind").and_then(Kind::parse) {
                        Some(k) if k != d.0 => None,
                        _ => Some(d),
                    }
                })),
                last_used: AtomicU64::new(0),
                bytes: AtomicU64::new(0),
            })
            .collect();
        Self {
            entries,
            budget,
            cache_dir,
            prompt_cache: None,
            prompt_cache_gb: None,
            clock: AtomicU64::new(1),
            loading: Mutex::new(()),
            log,
        }
    }

    pub fn find(&self, id: &str) -> ApiResult<&Entry> {
        self.entries.iter().find(|e| e.spec.id == id).ok_or_else(|| {
            let known: Vec<&str> = self.entries.iter().map(|e| e.spec.id.as_str()).collect();
            ApiError::new(404, format!("the model {id:?} does not exist (available: {})", known.join(", ")))
                .param("model")
                .code("model_not_found")
        })
    }

    /// Checks that `id` is listed and, when its kind is already known, that it is `want`.
    pub fn check(&self, id: &str, want: Kind) -> ApiResult<()> {
        let e = self.find(id)?;
        let kind = lock(&e.detected)
            .map(|d| d.0)
            .or_else(|| e.spec.option("kind").and_then(Kind::parse));
        match kind {
            Some(k) if k != want => Err(wrong_kind(id, k, want)),
            _ => Ok(()),
        }
    }

    /// Waits for the model, loads it if needed, and checks its kind.
    pub fn acquire(&self, id: &str, want: Kind) -> ApiResult<Lease<'_>> {
        let entry = self.find(id)?;
        let mut guard = lock(&entry.slot);
        if guard.is_none() {
            let (kind, image) = self.detect(entry).map_err(|e| ApiError::new(500, format!("model {id}: {e:#}")))?;
            if kind != want {
                return Err(wrong_kind(id, kind, want));
            }
            *guard = Some(
                self.load(entry, kind, image)
                    .map_err(|e| ApiError::new(500, format!("load {id}: {e:#}")))?,
            );
        }
        let kind = guard.as_ref().unwrap().kind();
        if kind != want {
            return Err(wrong_kind(id, kind, want));
        }
        entry.last_used.store(self.clock.fetch_add(1, Ordering::Relaxed), Ordering::Relaxed);
        Ok(Lease { entry, guard })
    }

    /// Writes the snapshot a loaded chat model's prompt cache has waiting (see
    /// [`crate::engine::Engine::flush_prompt_cache`]); nothing if the model is not loaded.
    pub fn flush_prompt_cache(&self, id: &str) {
        let Ok(e) = self.find(id) else { return };
        let mut g = lock(&e.slot);
        if let Some(Loaded::Chat(engine)) = g.as_mut()
            && let Err(err) = engine.flush_prompt_cache()
        {
            eprintln!("[{id}: prompt cache: {err:#}]");
        }
    }

    /// `(id, kind, loaded)` of every model.
    pub fn list(&self) -> Vec<(String, Option<Kind>, bool)> {
        self.entries
            .iter()
            .map(|e| {
                let kind = lock(&e.detected)
                    .map(|d| d.0)
                    .or_else(|| e.spec.option("kind").and_then(Kind::parse));
                (e.spec.id.clone(), kind, e.bytes.load(Ordering::Relaxed) > 0)
            })
            .collect()
    }

    /// Fetches every model (and detects its kind).
    pub fn download(&self) -> Result<()> {
        for e in &self.entries {
            self.detect(e)?;
            if e.spec.source.contains('/') && !Path::new(&e.spec.source).exists() {
                crate::engine::resolve_model(&e.spec.source, self.cache_dir.as_deref()).with_context(|| e.spec.id.clone())?;
            }
        }
        Ok(())
    }

    /// Loads every model (as far as the memory budget allows: later ones may unload earlier ones).
    pub fn preload(&self) -> Result<()> {
        for e in &self.entries {
            let mut g = lock(&e.slot);
            if g.is_none() {
                let (kind, image) = self.detect(e)?;
                *g = Some(self.load(e, kind, image).with_context(|| format!("load {}", e.spec.id))?);
            }
            e.last_used.store(self.clock.fetch_add(1, Ordering::Relaxed), Ordering::Relaxed);
        }
        Ok(())
    }

    /// Unloads idle models, least recently used first, until `need` more bytes fit in the budget
    /// (`u64::MAX`: all of them). Models in use are skipped.
    pub fn unload_idle(&self, need: u64) {
        self.make_room(None, need);
    }

    fn make_room(&self, except: Option<&Entry>, need: u64) {
        let budget = self.budget.unwrap_or(u64::MAX);
        let mut tried: Vec<*const Entry> = Vec::new();
        loop {
            let used: u64 = self.entries.iter().map(|e| e.bytes.load(Ordering::Relaxed)).sum();
            if need != u64::MAX && used.saturating_add(need) <= budget {
                return;
            }
            let victim = self
                .entries
                .iter()
                .filter(|e| e.bytes.load(Ordering::Relaxed) > 0)
                .filter(|e| except.is_none_or(|x| !std::ptr::eq(*e, x)) && !tried.contains(&(*e as *const Entry)))
                .min_by_key(|e| e.last_used.load(Ordering::Relaxed));
            let Some(v) = victim else { return };
            tried.push(v as *const Entry);
            // A model in use is not unloaded.
            if let Ok(mut g) = v.slot.try_lock()
                && g.is_some()
            {
                *g = None;
                let b = v.bytes.swap(0, Ordering::Relaxed);
                if self.log {
                    eprintln!("[unloaded {} ({:.1} GB) to make room]", v.spec.id, b as f64 / 1e9);
                }
            }
        }
    }

    /// The model's kind (and image pipeline), from the `kind` option and its files; a repo id
    /// that is not in the cache yet has its small files fetched for this.
    fn detect(&self, e: &Entry) -> Result<(Kind, Option<ImageKind>)> {
        if let Some(d) = *lock(&e.detected) {
            return Ok(d);
        }
        let given = e.spec.option("kind").and_then(Kind::parse);
        let d = match given {
            Some(k) if k != Kind::Image => (k, None),
            _ => {
                let path = self.probe_path(&e.spec.source)?;
                let (k, img) = detect_path(&path, &e.spec.source);
                match given {
                    Some(Kind::Image) if img.is_none() => bail!("{}: not an image pipeline oxinfer knows", e.spec.source),
                    _ => (k, img),
                }
            }
        };
        *lock(&e.detected) = Some(d);
        Ok(d)
    }

    /// A local path holding enough of the model to tell its kind.
    fn probe_path(&self, source: &str) -> Result<PathBuf> {
        if let Some(p) = local_path(source, self.cache_dir.as_deref()) {
            return Ok(p);
        }
        if source.contains(':') {
            // org/name:QUANT, a GGUF file: the model itself.
            return crate::engine::resolve_model(source, self.cache_dir.as_deref());
        }
        #[cfg(feature = "hf")]
        {
            const PROBE: [&str; 7] = [
                "model_index.json",
                "config.json",
                "modules.json",
                "llm_config.json",
                "vit_config.json",
                "meta.yaml",
                "tokenizer_config.json",
            ];
            crate::hf::resolve_files(source, &PROBE, self.cache_dir.as_deref())
        }
        #[cfg(not(feature = "hf"))]
        bail!("{source}: no such file or directory (built without the `hf` feature, so repo ids cannot be fetched)")
    }

    fn load(&self, e: &Entry, kind: Kind, image: Option<ImageKind>) -> Result<Loaded> {
        let _l = lock(&self.loading);
        let spec = &e.spec;
        let est = match spec.option("memory_gb") {
            Some(g) => (g.parse::<f64>().context("memory_gb")? * 1e9) as u64,
            None => estimate(spec, self.cache_dir.as_deref()),
        };
        let budget = self.budget.unwrap_or(u64::MAX);
        self.make_room(Some(e), est.min(budget));
        if self.log {
            eprintln!("[loading {} ({}, ~{:.1} GB)]", spec.id, kind.as_str(), est as f64 / 1e9);
        }
        let t0 = std::time::Instant::now();
        let opts: Vec<(&str, &str)> = spec
            .options
            .iter()
            .filter(|(k, _)| !server_key(k))
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        let cache = self.cache_dir.as_ref().map(|p| p.to_string_lossy().into_owned());
        let src = spec.source.as_str();
        let loaded = match kind {
            Kind::Chat => {
                let mut o = EngineOptions {
                    cache_dir: self.cache_dir.clone(),
                    prompt_cache: self.prompt_cache.clone(),
                    prompt_cache_gb: self.prompt_cache_gb,
                    ..Default::default()
                };
                for (k, v) in &opts {
                    o.set(k, v)?;
                }
                Loaded::Chat(Box::new(Engine::load(src, &o)?))
            }
            Kind::Embedding => {
                let mut quant = Quant::Bf16;
                for (k, v) in &opts {
                    match *k {
                        "quant" => quant = Quant::parse(v).with_context(|| format!("quant={v}: expected bf16, q8 or q4"))?,
                        _ => bail!("unknown embedding model option {k:?} (quant)"),
                    }
                }
                let dir = crate::engine::resolve_model(src, self.cache_dir.as_deref())?;
                let files = Box::new(crate::safetensors::ModelFiles::open(&dir, false)?);
                // SAFETY: the files are boxed (their address is stable) and dropped after the
                // embedder, which is declared first in EmbedModel.
                let files_ref: &'static crate::safetensors::ModelFiles = unsafe { &*(&*files as *const _) };
                let emb = crate::embed::Embedder::load(&dir, files_ref, quant)?;
                Loaded::Embedding(Box::new(EmbedModel { emb, _files: files }))
            }
            Kind::Transcription => {
                let mut o = crate::whisper::TranscriberOptions {
                    cache_dir: self.cache_dir.clone(),
                    ..Default::default()
                };
                for (k, v) in &opts {
                    o.set(k, v)?;
                }
                Loaded::Transcription(Box::new(crate::whisper::Transcriber::load(src, &o)?))
            }
            Kind::Speech => {
                let mut o = crate::tts::SpeakerOptions {
                    cache_dir: self.cache_dir.clone(),
                    ..Default::default()
                };
                for (k, v) in &opts {
                    o.set(k, v)?;
                }
                Loaded::Speech(Box::new(crate::tts::Speaker::load(src, &o)?))
            }
            Kind::Image => {
                let ik = image.context("not an image pipeline oxinfer knows")?;
                Loaded::Image(Box::new(load_image(ik, src, &opts, self.cache_dir.as_deref(), cache.as_deref())?))
            }
        };
        // Repo ids are resolved now: count what is on disk.
        let est = match spec.option("memory_gb") {
            Some(_) => est,
            None => estimate(spec, self.cache_dir.as_deref()),
        };
        e.bytes.store(est.clamp(1, budget), Ordering::Relaxed);
        if self.log {
            eprintln!("[loaded {} in {:.1}s]", spec.id, t0.elapsed().as_secs_f64());
        }
        Ok(loaded)
    }
}

fn load_image(ik: ImageKind, src: &str, opts: &[(&str, &str)], cache_dir: Option<&Path>, cache: Option<&str>) -> Result<ImageModel> {
    // Applies the options to a loader's option struct (with the cache directory when it takes one).
    macro_rules! options {
        ($o:expr, $cache:expr) => {{
            let mut o = $o;
            if $cache && let Some(c) = cache {
                o.set("cache_dir", c)?;
            }
            for (k, v) in opts {
                o.set(k, v)?;
            }
            o
        }};
    }
    Ok(match ik {
        ImageKind::Sd => {
            let (mut device, mut loras) = (crate::platform::DevicePref::Auto, Vec::new());
            for (k, v) in opts {
                match *k {
                    "device" => device = crate::platform::DevicePref::parse(v).with_context(|| format!("device: unknown device {v:?}"))?,
                    "lora" => loras.push(crate::sd::lora::LoraSpec::parse(v)?),
                    k => bail!("unknown Stable Diffusion option {k:?} (device, lora)"),
                }
            }
            for l in &mut loras {
                l.path = crate::engine::resolve_file(&l.path.to_string_lossy(), cache_dir)?;
            }
            let dir = crate::engine::resolve_model(src, cache_dir)?;
            crate::gpu::ensure(device)?;
            ImageModel::Sd(crate::sd::Pipeline::load_with_loras(&dir, &loras, &mut |_| {})?)
        }
        ImageKind::Flux => ImageModel::Flux(crate::flux::Pipeline::load(
            src,
            &options!(crate::flux::LoadOptions::default(), true),
        )?),
        ImageKind::QwenImage => ImageModel::QwenImage(crate::qwen_image::Pipeline::load(
            src,
            options!(crate::qwen_image::Options::default(), false),
        )?),
        ImageKind::OmniGen2 => ImageModel::OmniGen2(crate::omnigen2::Pipeline::load(
            src,
            options!(crate::omnigen2::Options::default(), false),
        )?),
        ImageKind::Step1x => ImageModel::Step1x(crate::step1x::Pipeline::load(
            src,
            options!(crate::step1x::Options::default(), false),
        )?),
        ImageKind::Bagel => ImageModel::Bagel(crate::bagel::Bagel::load(
            src,
            &options!(crate::bagel::LoadOptions::default(), true),
        )?),
        ImageKind::Janus => ImageModel::Janus(crate::janus::Janus::load(
            src,
            &options!(crate::janus::JanusOptions::default(), true),
        )?),
    })
}

fn wrong_kind(id: &str, kind: Kind, want: Kind) -> ApiError {
    ApiError::bad(format!(
        "the model {id:?} is a{} {} model; this endpoint needs a{} {} model",
        if kind == Kind::Embedding { "n" } else { "" },
        kind.as_str(),
        if want == Kind::Embedding { "n" } else { "" },
        want.as_str()
    ))
    .param("model")
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// The kind of the model at `path` (a directory or a GGUF file).
pub(crate) fn detect_path(path: &Path, source: &str) -> (Kind, Option<ImageKind>) {
    let img = |k| (Kind::Image, Some(k));
    if crate::gguf::is_gguf(path) {
        if crate::flux::is_flux_gguf(path) {
            return img(ImageKind::Flux);
        }
        // Orpheus GGUFs are plain Llama files; their names tell them apart.
        let name = format!("{source} {}", path.display()).to_ascii_lowercase();
        return if name.contains("orpheus") {
            (Kind::Speech, None)
        } else {
            (Kind::Chat, None)
        };
    }
    if crate::omnigen2::is_omnigen2(path) {
        img(ImageKind::OmniGen2)
    } else if crate::flux::is_flux_dir(path) {
        img(ImageKind::Flux)
    } else if crate::qwen_image::is_qwen_image(path) {
        img(ImageKind::QwenImage)
    } else if crate::bagel::is_bagel(path) {
        img(ImageKind::Bagel)
    } else if crate::step1x::is_step1x(path) {
        img(ImageKind::Step1x)
    } else if crate::sd::is_diffusers(path) {
        img(ImageKind::Sd)
    } else if crate::janus::is_janus(path) {
        img(ImageKind::Janus)
    } else if crate::embed::is_embedding_model(path) {
        (Kind::Embedding, None)
    } else if crate::whisper::is_whisper(path) {
        (Kind::Transcription, None)
    } else if is_orpheus(path) {
        (Kind::Speech, None)
    } else {
        (Kind::Chat, None)
    }
}

/// Orpheus checkpoints: Llama with the SNAC audio codes as extra vocabulary (156940 tokens).
fn is_orpheus(dir: &Path) -> bool {
    let cfg: Option<serde_json::Value> = crate::fsx::read(&dir.join("config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok());
    cfg.is_some_and(|c| c["model_type"] == "llama" && c["vocab_size"].as_u64().is_some_and(|v| v >= 156_000))
}

/// The model's files when they are on disk: a local path, or a repo in the cache (with its
/// configuration, or for `org/name:QUANT` the GGUF file).
fn local_path(source: &str, cache_dir: Option<&Path>) -> Option<PathBuf> {
    let p = Path::new(source);
    if p.exists() {
        return Some(crate::gguf::model_in_dir(p).unwrap_or_else(|| p.to_path_buf()));
    }
    let (repo, quant) = match source.split_once(':') {
        Some((r, q)) => (r, Some(q)),
        None => (source, None),
    };
    if repo.split('/').count() != 2 || repo.starts_with('/') || repo.contains("..") {
        return None;
    }
    let dir = cache_dir.map(Path::to_path_buf).or_else(cache_root)?.join(repo.replace('/', "--"));
    match quant {
        Some(q) => {
            let want = format!("{}.gguf", q.to_ascii_lowercase());
            std::fs::read_dir(&dir).ok()?.flatten().map(|e| e.path()).find(|f| {
                f.file_name()
                    .is_some_and(|n| n.to_string_lossy().to_ascii_lowercase().ends_with(&want))
            })
        }
        None => ["config.json", "model_index.json", "modules.json", "meta.yaml"]
            .iter()
            .any(|f| dir.join(f).is_file())
            .then_some(dir),
    }
}

/// The memory a model takes, from the size of its weight files (a repo id not fetched yet: 0),
/// scaled down when it is quantized at load.
fn estimate(spec: &ModelSpec, cache_dir: Option<&Path>) -> u64 {
    let Some(path) = local_path(&spec.source, cache_dir) else {
        return 0;
    };
    let bytes = disk_bytes(&path, 0);
    let factor = match (crate::gguf::is_gguf(&path), spec.option("quant")) {
        (false, Some("q4" | "q4_0")) => 0.3,
        (false, Some("q8" | "int8")) => 0.55,
        _ => 1.0,
    };
    (bytes as f64 * factor) as u64
}

fn cache_root() -> Option<PathBuf> {
    #[cfg(feature = "hf")]
    {
        Some(crate::hf::cache_root())
    }
    #[cfg(not(feature = "hf"))]
    None
}

/// Bytes of the files over 1 MB under `path` (the weights).
fn disk_bytes(path: &Path, depth: usize) -> u64 {
    let Ok(md) = std::fs::metadata(path) else { return 0 };
    if md.is_file() {
        return md.len();
    }
    if depth > 4 {
        return 0;
    }
    std::fs::read_dir(path)
        .map(|rd| {
            rd.flatten()
                .map(|e| {
                    let p = e.path();
                    match std::fs::metadata(&p) {
                        Ok(m) if m.is_file() && m.len() > 1 << 20 => m.len(),
                        Ok(m) if m.is_dir() => disk_bytes(&p, depth + 1),
                        _ => 0,
                    }
                })
                .sum()
        })
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds() {
        for k in [Kind::Chat, Kind::Embedding, Kind::Image, Kind::Transcription, Kind::Speech] {
            assert_eq!(Kind::parse(k.as_str()), Some(k));
        }
        assert_eq!(Kind::parse("video"), None);
        let e = wrong_kind("m", Kind::Embedding, Kind::Chat);
        assert!(
            e.message.contains("is an embedding model; this endpoint needs a chat model"),
            "{}",
            e.message
        );
    }

    #[test]
    fn detect_local_folders() {
        let dir = std::env::temp_dir().join(format!("oxinfer-detect-{}", std::process::id()));
        let sub = |name: &str, file: &str, body: &str| {
            let d = dir.join(name);
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(d.join(file), body).unwrap();
            d
        };
        let st = sub("st", "modules.json", "[]");
        assert_eq!(detect_path(&st, "st").0, Kind::Embedding);
        let w = sub("w", "config.json", r#"{"model_type": "whisper"}"#);
        assert_eq!(detect_path(&w, "w").0, Kind::Transcription);
        let o = sub("o", "config.json", r#"{"model_type": "llama", "vocab_size": 156940}"#);
        assert_eq!(detect_path(&o, "o").0, Kind::Speech);
        let l = sub("l", "config.json", r#"{"model_type": "llama", "vocab_size": 128256}"#);
        assert_eq!(detect_path(&l, "l").0, Kind::Chat);
        let sd = sub("sd", "model_index.json", r#"{"_class_name": "StableDiffusionPipeline"}"#);
        assert_eq!(detect_path(&sd, "sd"), (Kind::Image, Some(ImageKind::Sd)));
        let fx = sub("fx", "model_index.json", r#"{"_class_name": "FluxKontextPipeline"}"#);
        assert_eq!(detect_path(&fx, "fx"), (Kind::Image, Some(ImageKind::Flux)));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn lru_unloading_order() {
        let specs: Vec<ModelSpec> = ["a", "b", "c"].iter().map(|s| ModelSpec::new(s).unwrap()).collect();
        let r = Registry::new(&specs, Some(100), None, false);
        // Pretend every model is loaded (a slot holding a model is only needed to unload it).
        for (i, e) in r.entries.iter().enumerate() {
            e.bytes.store(40, Ordering::Relaxed);
            e.last_used.store([3, 1, 2][i], Ordering::Relaxed);
        }
        // Nothing is in the slots, so nothing can be unloaded: the loop ends.
        r.make_room(None, 50);
        assert!(r.entries.iter().all(|e| e.bytes.load(Ordering::Relaxed) == 40));
    }
}
