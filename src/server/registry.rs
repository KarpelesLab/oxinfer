//! The whitelist's models: kind detection, lazy loading, one request at a time per model, and
//! least-recently-used unloading under the memory budget.

use super::{ApiError, ApiResult, ModelSpec};
use crate::engine::{Engine, EngineOptions};
use crate::model::Quant;
use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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
    /// Text to speech: `/v1/audio/speech` (and Haru's speech to speech: `/v1/chat/completions`
    /// with `modalities: ["text", "audio"]`).
    Speech,
    /// Video generation (MiniMax-H3): `/v1/videos`, an asynchronous job.
    Video,
}

impl Kind {
    pub fn parse(s: &str) -> Option<Kind> {
        Some(match s {
            "chat" | "text" | "llm" => Kind::Chat,
            "embedding" | "embeddings" | "embed" => Kind::Embedding,
            "image" | "images" => Kind::Image,
            "transcription" | "asr" | "whisper" | "stt" => Kind::Transcription,
            "speech" | "tts" => Kind::Speech,
            "video" | "videos" => Kind::Video,
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
            Kind::Video => "video",
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
    Moss,
    /// Stable Diffusion 3 / 3.5 (a diffusers folder or an SD 3 GGUF transformer).
    Sd3,
    /// MiniMax-H3-Image.
    H3Image,
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
    Moss(crate::moss_image::Pipeline),
    Sd3(crate::sd3::Pipeline),
    H3Image(crate::minimax_h3::Pipeline),
}

/// A MiniMax-H3 video pipeline with the settings its options gave.
pub(crate) struct VideoModel {
    pub pipe: crate::minimax_h3::video::VideoPipeline,
    /// The step count a turbo LoRA is distilled for (requests without `steps` take it).
    pub steps: Option<usize>,
}
/// A loaded model.
pub(crate) enum Loaded {
    Chat(Box<Engine>),
    Embedding(Box<EmbedModel>),
    Image(Box<ImageModel>),
    Transcription(Box<crate::whisper::Transcriber>),
    Speech(Box<crate::tts::Speaker>),
    /// ZONOS3 (also `/v1/audio/speech`).
    Zonos(Box<crate::zonos::Zonos>),
    /// Fish Audio Drama 3 (also `/v1/audio/speech`).
    Drama(Box<crate::fish::Drama>),
    /// Audio8 TTS (also `/v1/audio/speech`).
    Arktts(Box<crate::arktts::Arktts>),
    /// Haru (speech to speech: `/v1/chat/completions` with audio in and out).
    Haru(Box<crate::haru::Haru>),
    Video(Box<VideoModel>),
}

impl Loaded {
    fn kind(&self) -> Kind {
        match self {
            Loaded::Chat(_) => Kind::Chat,
            Loaded::Embedding(_) => Kind::Embedding,
            Loaded::Image(_) => Kind::Image,
            Loaded::Transcription(_) => Kind::Transcription,
            Loaded::Speech(_) | Loaded::Zonos(_) | Loaded::Drama(_) | Loaded::Arktts(_) | Loaded::Haru(_) => Kind::Speech,
            Loaded::Video(_) => Kind::Video,
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
    /// Replaced by a later entry with the same id ([`Registry::add`]): no longer found by id.
    retired: AtomicBool,
}

impl Entry {
    fn new(spec: &ModelSpec, cache_dir: Option<&Path>) -> Self {
        Entry {
            spec: spec.clone(),
            slot: Mutex::new(None),
            // Models on disk already: their kind for the listing, without fetching anything.
            detected: Mutex::new(local_path(&spec.source, cache_dir).and_then(|p| {
                let d = detect_path(&p, &spec.source);
                match spec.option("kind").and_then(Kind::parse) {
                    Some(k) if k != d.0 => None,
                    _ => Some(d),
                }
            })),
            last_used: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
            retired: AtomicBool::new(false),
        }
    }

    fn kind(&self) -> Option<Kind> {
        lock(&self.detected)
            .map(|d| d.0)
            .or_else(|| self.spec.option("kind").and_then(Kind::parse))
    }

    fn busy(&self) -> bool {
        matches!(self.slot.try_lock(), Err(std::sync::TryLockError::WouldBlock))
    }
}

/// A model of the registry as [`Registry::status`] reports it.
#[derive(Clone, Debug)]
pub(crate) struct EntryStatus {
    pub spec: ModelSpec,
    pub kind: Option<Kind>,
    pub loaded: bool,
    /// A request (or a load) holds it.
    pub busy: bool,
    /// Memory counted for it while loaded.
    pub bytes: u64,
}

pub(crate) struct Registry {
    /// Boxed, and never removed or replaced while the registry lives (a replaced model is only
    /// marked `retired`), so references to entries stay valid for as long as `&self`.
    #[allow(clippy::vec_box)] // the boxes are what keeps the entries' addresses stable
    entries: Mutex<Vec<Box<Entry>>>,
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

/// What a request is waiting for before its first token ([`Registry::acquire_with`], and the
/// prompt pass of the generation that follows).
#[derive(Clone, Copy, Debug)]
pub(crate) enum Stage<'a> {
    /// Another request is using the model.
    Waiting,
    /// A file of the model is being fetched (`index` of `count` large files; 0 for metadata).
    #[cfg_attr(not(feature = "hf"), allow(dead_code))] // only downloads report it
    Downloading {
        file: &'a str,
        index: usize,
        count: usize,
        done: u64,
        total: Option<u64>,
    },
    /// The model is being read into memory.
    Loading,
    /// The prompt is being read by the model: tokens done of its total.
    Prompt { done: usize, total: usize },
}

/// Receives the [`Stage`]s of a request; false when the client is gone (a download is then
/// cancelled, a wait given up).
pub(crate) type OnStage<'a, 'f> = &'a mut (dyn FnMut(&Stage) -> bool + 'f);

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
        let entries = specs.iter().map(|s| Box::new(Entry::new(s, cache_dir.as_deref()))).collect();
        Self {
            entries: Mutex::new(entries),
            budget,
            cache_dir,
            prompt_cache: None,
            prompt_cache_gb: None,
            clock: AtomicU64::new(1),
            loading: Mutex::new(()),
            log,
        }
    }

    /// Every entry, retired ones included.
    fn all(&self) -> Vec<&Entry> {
        let g = lock(&self.entries);
        // SAFETY: each entry is boxed (its address does not move when the vector grows) and is
        // never removed or replaced before the registry is dropped, which `&self` outlives.
        g.iter().map(|b| unsafe { &*(&**b as *const Entry) }).collect()
    }

    /// The entries clients can use.
    fn active(&self) -> Vec<&Entry> {
        self.all().into_iter().filter(|e| !e.retired.load(Ordering::Relaxed)).collect()
    }

    pub fn find(&self, id: &str) -> ApiResult<&Entry> {
        let active = self.active();
        active.iter().find(|e| e.spec.id == id).copied().ok_or_else(|| {
            let known: Vec<&str> = active.iter().map(|e| e.spec.id.as_str()).collect();
            ApiError::new(404, format!("the model {id:?} does not exist (available: {})", known.join(", ")))
                .param("model")
                .code("model_not_found")
        })
    }

    /// Adds a model to the whitelist (the web GUI's models). An entry with the same id and the
    /// same source and options is kept as it is (returns false); one that differs is replaced
    /// once it is idle (it is unloaded first; a 409 while a request uses it).
    pub fn add(&self, spec: ModelSpec) -> ApiResult<bool> {
        // Entry::new reads the model's configuration when it is on disk: before the lock.
        let fresh = Box::new(Entry::new(&spec, self.cache_dir.as_deref()));
        let mut entries = lock(&self.entries);
        let old = entries.iter().find(|e| !e.retired.load(Ordering::Relaxed) && e.spec.id == spec.id);
        if let Some(old) = old {
            if old.spec == spec {
                return Ok(false);
            }
            // (try_lock: never waits for a slot while holding the list.)
            let mut g = old
                .slot
                .try_lock()
                .map_err(|_| ApiError::new(409, format!("{} is in use; try again once its request is done", spec.id)))?;
            *g = None;
            old.bytes.store(0, Ordering::Relaxed);
            old.retired.store(true, Ordering::Relaxed);
        }
        entries.push(fresh);
        Ok(true)
    }

    /// Unloads model `id` now: false when it was not loaded, a 409 while a request uses it.
    pub fn unload(&self, id: &str) -> ApiResult<bool> {
        let e = self.find(id)?;
        let mut g = e
            .slot
            .try_lock()
            .map_err(|_| ApiError::new(409, format!("{id} is in use; try again once its request is done")))?;
        let was = g.take().is_some();
        e.bytes.store(0, Ordering::Relaxed);
        if was && self.log {
            eprintln!("[unloaded {id}]");
        }
        Ok(was)
    }

    /// Loads model `id` now if it is not loaded (fetching it first when needed), reporting to
    /// `on` as [`Registry::acquire_with`] does; returns its kind.
    pub fn load_now(&self, id: &str, on: OnStage<'_, '_>) -> ApiResult<Kind> {
        let e = self.find(id)?;
        let kind = match e.kind() {
            Some(k) => k,
            None => self.detect(e).map_err(|err| ApiError::new(500, format!("model {id}: {err:#}")))?.0,
        };
        drop(self.acquire_with(id, kind, Some(on))?);
        Ok(kind)
    }

    /// The whitelist's models with their state.
    pub fn status(&self) -> Vec<EntryStatus> {
        self.active()
            .into_iter()
            .map(|e| EntryStatus {
                spec: e.spec.clone(),
                kind: e.kind(),
                loaded: e.bytes.load(Ordering::Relaxed) > 0,
                busy: e.busy(),
                bytes: e.bytes.load(Ordering::Relaxed),
            })
            .collect()
    }

    /// Whether a model whose files are under `folder` is loaded or in use.
    pub fn uses_folder(&self, folder: &Path) -> bool {
        let real = std::fs::canonicalize(folder).unwrap_or_else(|_| folder.to_path_buf());
        self.all().into_iter().any(|e| {
            (e.bytes.load(Ordering::Relaxed) > 0 || e.busy())
                && crate::model_cache::folder_of(&e.spec.source, self.cache_dir.as_deref())
                    .and_then(|f| std::fs::canonicalize(f).ok())
                    .is_some_and(|f| f == real)
        })
    }

    /// The memory the loaded models may take together, in bytes (None: unlimited).
    pub fn budget(&self) -> Option<u64> {
        self.budget
    }

    /// The model's kind, detected now when it is not known yet (a repo id not fetched has its
    /// small files fetched for this).
    pub fn kind_of(&self, id: &str) -> ApiResult<Kind> {
        let e = self.find(id)?;
        match e.kind() {
            Some(k) => Ok(k),
            None => Ok(self.detect(e).map_err(|err| ApiError::new(500, format!("model {id}: {err:#}")))?.0),
        }
    }

    /// `f` on the loaded model when it is loaded and idle (never waits, never loads).
    pub fn peek<R>(&self, id: &str, f: impl FnOnce(&mut Loaded) -> R) -> Option<R> {
        let e = self.find(id).ok()?;
        let mut g = e.slot.try_lock().ok()?;
        g.as_mut().map(f)
    }

    /// The model's files when they are on disk (a local path, or a repo in the cache).
    pub fn local_path(&self, id: &str) -> Option<PathBuf> {
        let e = self.find(id).ok()?;
        local_path(&e.spec.source, self.cache_dir.as_deref())
    }

    /// The cache directory repo ids are fetched to (None: the default).
    pub fn cache_dir(&self) -> Option<&Path> {
        self.cache_dir.as_deref()
    }

    /// Checks that `id` is listed and, when its kind is already known, that it is `want`.
    pub fn check(&self, id: &str, want: Kind) -> ApiResult<()> {
        let e = self.find(id)?;
        match e.kind() {
            Some(k) if k != want => Err(wrong_kind(id, k, want)),
            _ => Ok(()),
        }
    }

    /// Waits for the model, loads it if needed, and checks its kind.
    pub fn acquire(&self, id: &str, want: Kind) -> ApiResult<Lease<'_>> {
        self.acquire_with(id, want, None)
    }

    /// [`Registry::acquire`] reporting what it waits for to `on`: another request holding the
    /// model, the files being fetched, the load.
    pub fn acquire_with(&self, id: &str, want: Kind, mut on: Option<OnStage<'_, '_>>) -> ApiResult<Lease<'_>> {
        let gone = || ApiError::new(499, "the client closed the request");
        let entry = self.find(id)?;
        let mut guard = match (&mut on, entry.slot.try_lock()) {
            (_, Ok(g)) => g,
            (_, Err(std::sync::TryLockError::Poisoned(p))) => p.into_inner(),
            (None, Err(std::sync::TryLockError::WouldBlock)) => lock(&entry.slot),
            // In use: say so every two seconds until it is free.
            (Some(on), Err(std::sync::TryLockError::WouldBlock)) => {
                let mut next = std::time::Instant::now();
                loop {
                    if std::time::Instant::now() >= next {
                        if !on(&Stage::Waiting) {
                            return Err(gone());
                        }
                        next = std::time::Instant::now() + std::time::Duration::from_secs(2);
                    }
                    std::thread::sleep(std::time::Duration::from_millis(20));
                    match entry.slot.try_lock() {
                        Ok(g) => break g,
                        Err(std::sync::TryLockError::Poisoned(p)) => break p.into_inner(),
                        Err(std::sync::TryLockError::WouldBlock) => {}
                    }
                }
            }
        };
        if guard.is_none() {
            let load = || -> ApiResult<Loaded> {
                let (kind, image) = self.detect(entry).map_err(|e| ApiError::new(500, format!("model {id}: {e:#}")))?;
                if kind != want {
                    return Err(wrong_kind(id, kind, want));
                }
                self.load(entry, kind, image)
                    .map_err(|e| ApiError::new(500, format!("load {id}: {e:#}")))
            };
            *guard = Some(match &mut on {
                None => load()?,
                Some(on) => {
                    if !on(&Stage::Loading) {
                        return Err(gone());
                    }
                    // Downloads report through the hook; once the last file is whole the rest
                    // of the wait is the load.
                    #[cfg(feature = "hf")]
                    {
                        let mut hook = |p: &crate::hf::Progress| {
                            let last = p.count > 0 && p.index == p.count && p.total == Some(p.done);
                            on(&Stage::Downloading {
                                file: p.file,
                                index: p.index,
                                count: p.count,
                                done: p.done,
                                total: p.total,
                            }) && (!last || on(&Stage::Loading))
                        };
                        crate::hf::with_progress(&mut hook, || Ok(load())).map_err(|_| gone())??
                    }
                    // Without `hf` nothing is downloaded: the wait is the load alone.
                    #[cfg(not(feature = "hf"))]
                    load()?
                }
            });
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
        self.active()
            .into_iter()
            .map(|e| (e.spec.id.clone(), e.kind(), e.bytes.load(Ordering::Relaxed) > 0))
            .collect()
    }

    /// Fetches every model (and detects its kind).
    pub fn download(&self) -> Result<()> {
        for e in self.active() {
            self.detect(e)?;
            if e.spec.source.contains('/') && !Path::new(&e.spec.source).exists() {
                crate::engine::resolve_model(&e.spec.source, self.cache_dir.as_deref()).with_context(|| e.spec.id.clone())?;
            }
        }
        Ok(())
    }

    /// Loads every model (as far as the memory budget allows: later ones may unload earlier ones).
    pub fn preload(&self) -> Result<()> {
        for e in self.active() {
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
        let entries = self.all();
        loop {
            let used: u64 = entries.iter().map(|e| e.bytes.load(Ordering::Relaxed)).sum();
            if need != u64::MAX && used.saturating_add(need) <= budget {
                return;
            }
            let victim = entries
                .iter()
                .copied()
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
            // Told apart by its name: the repository has no config.json to probe.
            _ if crate::zonos::is_zonos(Path::new(&e.spec.source)) => (Kind::Speech, None),
            _ if !Path::new(&e.spec.source).exists() && crate::fish::is_drama(Path::new(&e.spec.source)) => (Kind::Speech, None),
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
            Kind::Speech if crate::zonos::is_zonos(Path::new(src)) => {
                let mut o = crate::zonos::ZonosOptions {
                    cache_dir: self.cache_dir.clone(),
                    ..Default::default()
                };
                for (k, v) in &opts {
                    match *k {
                        "threads" => o.threads = Some(v.parse().with_context(|| format!("threads: {v:?}"))?),
                        "decoder" => o.decoder = Some(v.to_string()),
                        "exact" => o.exact = matches!(*v, "1" | "true" | "yes" | "on"),
                        _ => bail!("unknown ZONOS3 option {k:?} (threads, decoder, exact)"),
                    }
                }
                Loaded::Zonos(Box::new(crate::zonos::Zonos::load(src, &o)?))
            }
            Kind::Speech if crate::fish::is_drama(Path::new(src)) => {
                let mut o = crate::fish::DramaOptions {
                    cache_dir: self.cache_dir.clone(),
                    ..Default::default()
                };
                for (k, v) in &opts {
                    match *k {
                        "threads" => o.threads = Some(v.parse().with_context(|| format!("threads: {v:?}"))?),
                        "mmap" => o.mmap = matches!(*v, "1" | "true" | "yes" | "on"),
                        "quant" => o.quant = crate::model::Quant::parse(v).with_context(|| format!("quant: {v:?} (bf16, q8, q4)"))?,
                        _ => bail!("unknown Drama 3 option {k:?} (threads, mmap, quant)"),
                    }
                }
                Loaded::Drama(Box::new(crate::fish::Drama::load(src, &o)?))
            }
            Kind::Speech if local_path(src, self.cache_dir.as_deref()).is_some_and(|p| crate::haru::is_haru(&p)) => {
                let mut o = crate::haru::HaruOptions {
                    cache_dir: self.cache_dir.clone(),
                    ..Default::default()
                };
                for (k, v) in &opts {
                    match *k {
                        "quant" => o.quant = Quant::parse(v).with_context(|| format!("quant: {v:?} (bf16, q8, q4)"))?,
                        "mmap" => o.mmap = matches!(*v, "1" | "true" | "yes" | "on"),
                        _ => bail!("unknown Haru option {k:?} (quant, mmap)"),
                    }
                }
                Loaded::Haru(Box::new(crate::haru::Haru::load(src, &o)?))
            }
            Kind::Speech if local_path(src, self.cache_dir.as_deref()).is_some_and(|p| crate::arktts::is_arktts(&p)) => {
                Loaded::Arktts(Box::new(crate::arktts::load_for_server(src, &opts, self.cache_dir.clone())?))
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
            Kind::Video => Loaded::Video(Box::new(load_video(src, &opts, self.cache_dir.as_deref())?)),
        };
        // Repo ids are resolved now: count what is on disk.
        let est = match spec.option("memory_gb") {
            Some(_) => est,
            None => estimate(spec, self.cache_dir.as_deref()),
        };
        e.bytes.store(est.clamp(1, budget), Ordering::Relaxed);
        crate::model_cache::touch(src, self.cache_dir.as_deref());
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
        ImageKind::Moss => ImageModel::Moss(crate::moss_image::Pipeline::load(
            src,
            options!(crate::moss_image::Options::default(), true),
        )?),
        ImageKind::Janus => ImageModel::Janus(crate::janus::Janus::load(
            src,
            &options!(crate::janus::JanusOptions::default(), true),
        )?),
        ImageKind::Sd3 => ImageModel::Sd3(crate::sd3::Pipeline::load(
            src,
            &options!(crate::sd3::LoadOptions::default(), true),
        )?),
        ImageKind::H3Image => {
            let dir = crate::engine::resolve_model(src, cache_dir)?;
            let mut te = None;
            for (k, v) in opts {
                match *k {
                    "text_encoder" => te = Some(crate::engine::resolve_model(v, cache_dir)?),
                    k => bail!("unknown MiniMax-H3-Image option {k:?} (text_encoder)"),
                }
            }
            let te = match te {
                Some(t) => t,
                None => crate::minimax_h3::find_text_encoder(&dir, cache_dir)?,
            };
            ImageModel::H3Image(crate::minimax_h3::Pipeline::load(&dir, &te)?)
        }
    })
}

/// Loads a MiniMax-H3 video model with its options
/// ([`crate::minimax_h3::video::load_with_options`]).
fn load_video(src: &str, opts: &[(&str, &str)], cache_dir: Option<&Path>) -> Result<VideoModel> {
    let (pipe, steps) = crate::minimax_h3::video::load_with_options(src, opts, cache_dir)?;
    Ok(VideoModel { pipe, steps })
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
    if crate::zonos::is_zonos(path) || crate::fish::is_drama(path) {
        return (Kind::Speech, None);
    }
    if crate::gguf::is_gguf(path) {
        if crate::flux::is_flux_gguf(path) {
            return img(ImageKind::Flux);
        }
        if crate::sd3::is_sd3_gguf(path) {
            return img(ImageKind::Sd3);
        }
        // Orpheus GGUFs are plain Llama files; their names tell them apart.
        let name = format!("{source} {}", path.display()).to_ascii_lowercase();
        return if name.contains("orpheus") {
            (Kind::Speech, None)
        } else {
            (Kind::Chat, None)
        };
    }
    if crate::minimax_h3::video::is_h3_video(path) {
        return (Kind::Video, None);
    }
    if crate::moss_image::is_moss_image(path) {
        img(ImageKind::Moss)
    } else if crate::omnigen2::is_omnigen2(path) {
        img(ImageKind::OmniGen2)
    } else if crate::flux::is_flux_dir(path) {
        img(ImageKind::Flux)
    } else if crate::qwen_image::is_qwen_image(path) {
        img(ImageKind::QwenImage)
    } else if crate::bagel::is_bagel(path) {
        img(ImageKind::Bagel)
    } else if crate::step1x::is_step1x(path) {
        img(ImageKind::Step1x)
    } else if crate::minimax_h3::is_h3_image(path) {
        img(ImageKind::H3Image)
    } else if crate::sd3::is_sd3_dir(path) {
        img(ImageKind::Sd3)
    } else if crate::sd::is_diffusers(path) {
        img(ImageKind::Sd)
    } else if crate::janus::is_janus(path) {
        img(ImageKind::Janus)
    } else if crate::embed::is_embedding_model(path) {
        (Kind::Embedding, None)
    } else if crate::whisper::is_whisper(path) || crate::nemotron_asr::is_speech_lm(path) {
        (Kind::Transcription, None)
    } else if is_orpheus(path) || crate::arktts::is_arktts(path) || crate::haru::is_haru(path) {
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
    // (A repo without a config file oxinfer probes, such as ZONOS3: its cache folder.)
    let found =
        local_path(&spec.source, cache_dir).or_else(|| crate::model_cache::folder_of(&spec.source, cache_dir).filter(|f| f.is_dir()));
    let Some(path) = found else {
        return 0;
    };
    let bytes = disk_bytes(&path, 0);
    let factor = match (crate::gguf::is_gguf(&path), spec.option("quant")) {
        // Drama 3: nine tenths of it are FP8 experts, kept as stored.
        _ if crate::fish::is_drama(&path) => 1.0,
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
        for k in [
            Kind::Chat,
            Kind::Embedding,
            Kind::Image,
            Kind::Transcription,
            Kind::Speech,
            Kind::Video,
        ] {
            assert_eq!(Kind::parse(k.as_str()), Some(k));
        }
        assert_eq!(Kind::parse("audio"), None);
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
        let sd3 = sub("sd3", "model_index.json", r#"{"_class_name": "StableDiffusion3Pipeline"}"#);
        assert_eq!(detect_path(&sd3, "sd3"), (Kind::Image, Some(ImageKind::Sd3)));
        let h3i = sub("h3i", "model_index.json", r#"{"_class_name": "MiniMaxH3ImagePipeline"}"#);
        assert_eq!(detect_path(&h3i, "h3i"), (Kind::Image, Some(ImageKind::H3Image)));
        let h3 = sub("h3", "model_index.json", r#"{"_class_name": "MiniMaxH3ModularPipeline"}"#);
        assert_eq!(detect_path(&h3, "h3"), (Kind::Video, None));
        let haru = sub("haru", "config.json", r#"{"model_type": "haru"}"#);
        assert_eq!(detect_path(&haru, "haru").0, Kind::Speech);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn lru_unloading_order() {
        let specs: Vec<ModelSpec> = ["a", "b", "c"].iter().map(|s| ModelSpec::new(s).unwrap()).collect();
        let r = Registry::new(&specs, Some(100), None, false);
        // Pretend every model is loaded (a slot holding a model is only needed to unload it).
        for (i, e) in r.all().into_iter().enumerate() {
            e.bytes.store(40, Ordering::Relaxed);
            e.last_used.store([3, 1, 2][i], Ordering::Relaxed);
        }
        // Nothing is in the slots, so nothing can be unloaded: the loop ends.
        r.make_room(None, 50);
        assert!(r.all().iter().all(|e| e.bytes.load(Ordering::Relaxed) == 40));
    }
}
