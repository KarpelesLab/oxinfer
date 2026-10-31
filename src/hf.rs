//! Fetching and caching HuggingFace model repositories by name, over rsurl (pure-Rust HTTP).
//!
//! `--model google/gemma-4-E4B-it` resolves to `$OXINFER_CACHE/google--gemma-4-E4B-it` (default
//! `~/.cache/oxinfer/hub`), downloading the config, tokenizer, template and safetensors files that
//! are missing or incomplete. Downloads are resumable. Gated repos need `HF_TOKEN`.
//!
//! Files come from the ATOnline mirror when it carries the repo (`AI/Model:get`, which lists
//! every file with a download URL), otherwise from huggingface.co; `OXINFER_HUB` overrides.

use anyhow::{Context, Result, anyhow, bail};
use rsurl::{DownloadOptions, Request};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// `$OXINFER_CACHE`, else `~/.cache/oxinfer/hub` (on iOS, where `HOME` is the app's sandbox:
/// `Library/Application Support/oxinfer/hub` in it, which the system does not purge).
pub fn cache_root() -> PathBuf {
    if let Ok(p) = std::env::var("OXINFER_CACHE") {
        return PathBuf::from(p);
    }
    let home = PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".to_string()));
    if cfg!(target_os = "ios") {
        return home.join("Library").join("Application Support").join("oxinfer").join("hub");
    }
    home.join(".cache").join("oxinfer").join("hub")
}

/// Download progress reported to a [`with_progress`] hook.
#[derive(Clone, Copy, Debug)]
pub struct Progress<'a> {
    pub repo: &'a str,
    /// The file being fetched (path in the repository).
    pub file: &'a str,
    /// 1-based index of the file among the `count` large files of this fetch (0 for metadata).
    pub index: usize,
    pub count: usize,
    /// Bytes on disk so far (including a resumed part) and the file size when known.
    pub done: u64,
    pub total: Option<u64>,
}

type ProgressHook<'f> = dyn FnMut(&Progress) -> bool + 'f;

thread_local! {
    static HOOK: std::cell::Cell<Option<*mut ProgressHook<'static>>> = const { std::cell::Cell::new(None) };
}

/// Unwinding payload of a cancelled download.
struct Cancelled;

/// Runs `f` (a [`resolve`], [`resolve_file`] or [`resolve_gguf`] call on this thread) with
/// `hook` receiving download progress; returning false from the hook cancels the download (the
/// partial file stays and a later fetch resumes it) and makes this return an error.
pub fn with_progress<T>(hook: &mut ProgressHook<'_>, f: impl FnOnce() -> Result<T>) -> Result<T> {
    let p: *mut ProgressHook<'_> = hook;
    // SAFETY: the pointer is only dereferenced on this thread while `hook` is borrowed (it is
    // removed before returning, also when unwinding through `catch_unwind`).
    let p: *mut ProgressHook<'static> = unsafe { std::mem::transmute(p) };
    let prev = HOOK.with(|h| h.replace(Some(p)));
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    HOOK.with(|h| h.set(prev));
    match r {
        Ok(r) => r,
        Err(e) if e.is::<Cancelled>() => bail!("download cancelled"),
        Err(e) => std::panic::resume_unwind(e),
    }
}

/// Reports progress to the hook of [`with_progress`], if any; unwinds out of the download when it
/// asks to cancel.
fn report(p: &Progress) {
    // Taken while called, so the hook is never re-entered.
    let Some(h) = HOOK.with(|c| c.take()) else {
        return;
    };
    // SAFETY: see `with_progress`.
    let keep = unsafe { (*h)(p) };
    HOOK.with(|c| c.set(Some(h)));
    if !keep {
        std::panic::resume_unwind(Box::new(Cancelled));
    }
}

/// A local directory is used as is; otherwise `spec` must be a repo id (`org/name`), which is
/// fetched into the cache (if needed) and its directory returned.
pub fn resolve(spec: &str, cache_dir: Option<&Path>) -> Result<PathBuf> {
    let p = Path::new(spec);
    if p.is_dir() || crate::gguf::is_gguf(p) {
        return Ok(p.to_path_buf());
    }
    if let Some((repo, quant)) = spec.split_once(':')
        && repo.split('/').count() == 2
    {
        return resolve_gguf(repo, quant, cache_dir);
    }
    let parts: Vec<&str> = spec.split('/').collect();
    if parts.len() != 2 || parts.iter().any(|s| s.is_empty() || s.starts_with('.')) || spec.starts_with('/') {
        bail!("{spec}: not an existing directory and not a HuggingFace repo id of the form org/name");
    }
    let root = cache_dir.map(Path::to_path_buf).unwrap_or_else(cache_root);
    let dir = root.join(spec.replace('/', "--"));
    std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
    fetch_repo(spec, &dir)?;
    Ok(dir)
}

/// Some files of a repo: a local directory is used as is; otherwise the files of `names` that
/// repo `spec` (`org/name`) has are fetched into its cache directory, which is returned. For
/// add-ons such as decision assets, without the weights.
pub fn resolve_files(spec: &str, names: &[&str], cache_dir: Option<&Path>) -> Result<PathBuf> {
    let p = Path::new(spec);
    if p.is_dir() {
        return Ok(p.to_path_buf());
    }
    let parts: Vec<&str> = spec.split('/').collect();
    if parts.len() != 2 || parts.iter().any(|s| s.is_empty() || s.starts_with('.')) || spec.starts_with('/') {
        bail!("{spec}: not an existing directory and not a HuggingFace repo id of the form org/name");
    }
    let root = cache_dir.map(Path::to_path_buf).unwrap_or_else(cache_root);
    let dir = root.join(spec.replace('/', "--"));
    let list = listing(spec)?;
    let present: Vec<&&str> = names.iter().filter(|n| list.contains_key(**n)).collect();
    for (i, n) in present.iter().enumerate() {
        let (size, src) = &list[**n];
        fetch_file(spec, &dir, n, *size, src, i + 1, present.len())?;
    }
    Ok(dir)
}

/// A single file: a local path is used as is; otherwise `spec` is `org/name/path/in/repo` (or
/// `org/name` when the repo holds exactly one safetensors file), fetched into the repo's cache
/// directory. For LoRA files and other single-file downloads.
pub fn resolve_file(spec: &str, cache_dir: Option<&Path>) -> Result<PathBuf> {
    let p = Path::new(spec);
    if p.is_file() {
        return Ok(p.to_path_buf());
    }
    if let Some((repo, quant)) = spec.split_once(':')
        && repo.split('/').count() == 2
    {
        return resolve_gguf(repo, quant, cache_dir);
    }
    let parts: Vec<&str> = spec.split('/').collect();
    if parts.len() < 2 || parts.iter().any(|s| s.is_empty() || s.starts_with('.')) || spec.starts_with('/') {
        bail!("{spec}: not an existing file and not a HuggingFace path of the form org/name/file");
    }
    let repo = format!("{}/{}", parts[0], parts[1]);
    let list = listing(&repo)?;
    let name = if parts.len() > 2 {
        parts[2..].join("/")
    } else {
        let mut st: Vec<&String> = list.keys().filter(|k| k.ends_with(".safetensors")).collect();
        st.sort();
        match st.as_slice() {
            [one] => one.to_string(),
            [] => bail!("{repo}: no safetensors file in the repository"),
            many => bail!(
                "{repo} has several safetensors files; name one as {repo}/FILE: {}",
                many.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
            ),
        }
    };
    let (size, src) = list.get(&name).with_context(|| format!("{repo}: no file {name}"))?;
    let root = cache_dir.map(Path::to_path_buf).unwrap_or_else(cache_root);
    let dir = root.join(repo.replace('/', "--"));
    fetch_file(&repo, &dir, &name, *size, src, 1, 1)?;
    Ok(dir.join(name))
}

/// The GGUF files of a listing that hold a model (not vision projectors), sorted.
fn gguf_models(list: &Listing) -> Vec<&String> {
    let mut v: Vec<&String> = list
        .keys()
        .filter(|k| k.ends_with(".gguf") && !k.rsplit('/').next().unwrap_or(k).starts_with("mmproj"))
        .collect();
    v.sort();
    v
}

/// `org/name:QUANT` for a repository of GGUF files (as llama.cpp's `-hf`): the file whose name
/// ends in `QUANT.gguf` (case-insensitive, e.g. `Q4_K_M` or `FastMTP-32K`), fetched alone.
pub fn resolve_gguf(repo: &str, quant: &str, cache_dir: Option<&Path>) -> Result<PathBuf> {
    let list = listing(repo)?;
    let ggufs = gguf_models(&list);
    let want = format!("{}.gguf", quant.to_ascii_lowercase());
    let hits: Vec<&&String> = ggufs
        .iter()
        .filter(|f| {
            let f = f.to_ascii_lowercase();
            f.ends_with(&want) && f[..f.len() - want.len()].ends_with(['-', '.', '_', '/'])
        })
        .collect();
    let name = match hits.as_slice() {
        [one] => one.to_string(),
        [] => bail!(
            "{repo}: no GGUF file for {quant}; available: {}",
            ggufs.iter().map(|f| f.as_str()).collect::<Vec<_>>().join(", ")
        ),
        many => bail!(
            "{repo}: several GGUF files match {quant}: {}",
            many.iter().map(|f| f.as_str()).collect::<Vec<_>>().join(", ")
        ),
    };
    if name.contains("-of-") {
        bail!("{name}: split GGUF files are not supported yet");
    }
    let (size, src) = &list[&name];
    let root = cache_dir.map(Path::to_path_buf).unwrap_or_else(cache_root);
    let dir = root.join(repo.replace('/', "--"));
    std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
    fetch_file(repo, &dir, &name, *size, src, 1, 1)?;
    Ok(dir.join(name))
}

/// Percent-encodes a repository path for a URL (keeps `/` and unreserved characters).
fn url_path(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b"-._~/".contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

fn token() -> Option<String> {
    std::env::var("HF_TOKEN")
        .ok()
        .or_else(|| std::env::var("HUGGING_FACE_HUB_TOKEN").ok())
        .filter(|t| !t.is_empty())
}

/// A GET request; `auth` adds the HuggingFace token (never sent to other hosts).
fn request(url: &str, auth: bool) -> Result<Request> {
    let mut r = Request::get(url)
        .map_err(|e| anyhow!("{url}: {e}"))?
        .follow_redirects(true)
        .max_redirs(10)
        .header("User-Agent", "oxinfer/0.1")
        // The resumable downloader stores raw bytes; refuse compressed transfer encodings.
        .header("Accept-Encoding", "identity");
    if auth && let Some(t) = token() {
        r = r.header("Authorization", &format!("Bearer {t}"));
    }
    Ok(r)
}

/// The body of a plain GET of `url` (no HuggingFace token), for image URLs in decision requests.
pub fn http_get(url: &str) -> Result<Vec<u8>> {
    let resp = request(url, false)?.send().map_err(|e| anyhow!("{url}: {e}"))?;
    if resp.status != 200 {
        bail!("{url}: HTTP {}", resp.status);
    }
    Ok(resp.body)
}

#[derive(Deserialize)]
struct Sibling {
    rfilename: String,
    #[serde(default)]
    size: Option<u64>,
}

#[derive(Deserialize)]
struct RepoInfo {
    #[serde(default)]
    siblings: Vec<Sibling>,
}

const SMALL_FILES: [&str; 9] = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "processor_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
];

/// Where a repository file comes from.
enum Source {
    Url(String),
    /// Small files the mirror inlines as `data:` URLs.
    Inline(Vec<u8>),
}

/// A repository's file listing: name -> (size, source).
type Listing = HashMap<String, (Option<u64>, Source)>;

/// The ATOnline model mirror: `AI/Model:get` lists a repo's files with download URLs.
pub const MIRROR_API: &str = "https://ws.atonline.com/_special/rest/AI/Model:get";

#[derive(Deserialize)]
struct MirrorFile {
    #[serde(rename = "Filename")]
    filename: String,
    #[serde(rename = "Size")]
    size: String,
    #[serde(rename = "Url")]
    url: String,
}

#[derive(Deserialize)]
struct MirrorModel {
    #[serde(rename = "Files")]
    files: Vec<MirrorFile>,
}

#[derive(Deserialize)]
struct MirrorReply {
    data: MirrorModel,
}

/// Lists `repo` on the mirror; Ok(None) when the mirror does not carry it.
fn list_mirror(repo: &str) -> Result<Option<Listing>> {
    let api = format!("{MIRROR_API}?key={repo}");
    let resp = request(&api, false)?.send().map_err(|e| anyhow!("{api}: {e}"))?;
    match resp.status {
        200 => {}
        404 => return Ok(None),
        s => bail!("{api}: HTTP {s}"),
    }
    let reply: MirrorReply = serde_json::from_slice(&resp.body).with_context(|| format!("parse {api}"))?;
    let mut out = Listing::new();
    for f in reply.data.files {
        let src = match f.url.strip_prefix("data:") {
            Some(d) => {
                let b64 = d
                    .split_once(";base64,")
                    .map(|(_, b)| b)
                    .with_context(|| format!("{}: unsupported data URL", f.filename))?;
                Source::Inline(crate::b64::decode(b64).with_context(|| format!("{}: bad base64", f.filename))?)
            }
            None => Source::Url(f.url),
        };
        out.insert(f.filename, (f.size.parse().ok(), src));
    }
    Ok(Some(out))
}

/// Lists `repo` on huggingface.co.
fn list_hf(repo: &str) -> Result<Listing> {
    let api = format!("https://huggingface.co/api/models/{repo}?blobs=true");
    let resp = request(&api, true)?.send().map_err(|e| anyhow!("{api}: {e}"))?;
    match resp.status {
        200 => {}
        401 | 403 => bail!(
            "{repo}: access denied (HTTP {}). If the repo is gated, accept its license on huggingface.co and set HF_TOKEN.",
            resp.status
        ),
        404 => bail!("{repo}: repository not found on huggingface.co"),
        s => bail!("{api}: HTTP {s}"),
    }
    let info: RepoInfo = serde_json::from_slice(&resp.body).with_context(|| format!("parse {api}"))?;
    Ok(info
        .siblings
        .into_iter()
        .map(|s| {
            let url = format!("https://huggingface.co/{repo}/resolve/main/{}", url_path(&s.rfilename));
            (s.rfilename, (s.size, Source::Url(url)))
        })
        .collect())
}

/// Where repositories are fetched from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hub {
    /// The mirror, then huggingface.co for repos it does not carry (the default).
    Auto,
    /// The ATOnline mirror only.
    Mirror,
    /// huggingface.co only.
    HuggingFace,
}

static HUB: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Selects the hub for later fetches, overriding `OXINFER_HUB`.
pub fn set_hub(hub: Hub) {
    HUB.store(hub as u8 + 1, std::sync::atomic::Ordering::Relaxed);
}

/// The hub in use: the one given to [`set_hub`], else `OXINFER_HUB=auto|mirror|hf`, else `Auto`.
pub fn hub() -> Result<Hub> {
    Ok(match HUB.load(std::sync::atomic::Ordering::Relaxed) {
        1 => Hub::Auto,
        2 => Hub::Mirror,
        3 => Hub::HuggingFace,
        _ => match std::env::var("OXINFER_HUB").unwrap_or_default().as_str() {
            "" | "auto" => Hub::Auto,
            "mirror" => Hub::Mirror,
            "hf" => Hub::HuggingFace,
            h => bail!("OXINFER_HUB={h}: expected auto, mirror or hf"),
        },
    })
}

fn listing(repo: &str) -> Result<Listing> {
    match hub()? {
        Hub::HuggingFace => list_hf(repo),
        Hub::Mirror => list_mirror(repo)?.with_context(|| format!("{repo}: not on the mirror ({MIRROR_API})")),
        Hub::Auto => match list_mirror(repo) {
            Ok(Some(l)) => Ok(l),
            Ok(None) => list_hf(repo),
            Err(e) => {
                eprintln!("[mirror unavailable ({e:#}); trying huggingface.co]");
                list_hf(repo)
            }
        },
    }
}

/// An ANEMLL Core ML repository: `meta.yaml` plus compiled `.mlmodelc` directories.
fn is_anemll_listing(list: &Listing) -> bool {
    list.contains_key("meta.yaml") && list.keys().any(|k| k.contains(".mlmodelc/"))
}

/// Files of an ANEMLL repository worth fetching: everything but its Python chat scripts,
/// uncompiled `.mlpackage`s and Finder / git metadata.
fn anemll_wanted(name: &str) -> bool {
    let base = name.rsplit('/').next().unwrap_or(name);
    !(base == ".DS_Store"
        || base == ".gitattributes"
        || (base.starts_with("chat") && base.ends_with(".py"))
        || name.contains(".mlpackage/"))
}

/// Files of a diffusers pipeline repository (`model_index.json` at the root, as for
/// `timbrooks/instruct-pix2pix`): the index, each component's configs and tokenizer files, and one
/// safetensors weight file per component (the fp16 variant when there is one: half the download,
/// and the weights are kept in bf16 anyway). The safety checker, the full single-file checkpoints
/// at the root and PyTorch `.bin` duplicates are skipped.
fn diffusers_files(list: &Listing) -> Vec<String> {
    let mut files = vec!["model_index.json".to_string()];
    let mut dirs: Vec<&str> = list
        .keys()
        .filter_map(|k| k.split_once('/').map(|(d, _)| d))
        .filter(|d| !matches!(*d, "safety_checker" | "feature_extractor"))
        .collect();
    dirs.sort();
    dirs.dedup();
    for d in dirs {
        let mut names: Vec<&String> = list.keys().filter(|k| k.split_once('/').is_some_and(|(p, _)| p == d)).collect();
        names.sort();
        for n in &names {
            if n.ends_with(".json") || n.ends_with(".txt") || n.ends_with(".model") {
                files.push(n.to_string());
            }
        }
        let st: Vec<&&String> = names.iter().filter(|n| n.ends_with(".safetensors")).collect();
        if let Some(f) = st.iter().find(|n| n.ends_with(".fp16.safetensors")).or_else(|| st.first()) {
            files.push(f.to_string());
        }
    }
    files
}

fn fetch_repo(repo: &str, dir: &Path) -> Result<()> {
    let list = listing(repo)?;
    if is_anemll_listing(&list) {
        let mut files: Vec<&String> = list.keys().filter(|k| anemll_wanted(k)).collect();
        files.sort();
        let count = files.len();
        for (i, f) in files.iter().enumerate() {
            let (size, src) = &list[*f];
            fetch_file(repo, dir, f, *size, src, i + 1, count)?;
        }
        return Ok(());
    }
    if list.contains_key("model_index.json") {
        let files = diffusers_files(&list);
        let count = files.len();
        for (i, f) in files.iter().enumerate() {
            let (size, src) = &list[f];
            fetch_file(repo, dir, f, *size, src, i + 1, count)?;
        }
        return Ok(());
    }
    let mut files: Vec<String> = SMALL_FILES
        .iter()
        .filter(|f| list.contains_key(**f))
        .map(|f| f.to_string())
        .collect();
    // Jev decision bundles: the head, adapter and configs (`crate::decide`); the LM-head
    // readout's adapter only when there is no slot head.
    let mut decision: Vec<&str> = crate::decide::ASSET_FILES.to_vec();
    if !list.contains_key("head.safetensors") {
        decision.extend(crate::decide::LM_ADAPTER_FILES);
    }
    decision.retain(|f| list.contains_key(*f) && !files.iter().any(|g| g == f));
    files.extend(decision.into_iter().map(str::to_string));
    // sentence-transformers layout (embedding models): the module list, the prompts, and each
    // module's folder (pooling, dense projections); BGE-M3's sparse and multi-vector heads.
    if let Some((size, src)) = list.get("modules.json") {
        fetch_file(repo, dir, "modules.json", *size, src, 0, 0)?;
        let modules: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("modules.json"))?)?;
        for f in [
            "config_sentence_transformers.json",
            "sentence_bert_config.json",
            "sparse_linear.pt",
            "colbert_linear.pt",
        ] {
            if list.contains_key(f) {
                files.push(f.into());
            }
        }
        for m in modules.as_array().into_iter().flatten() {
            let path = m["path"].as_str().unwrap_or_default();
            if path.is_empty() {
                continue;
            }
            for f in ["config.json", "model.safetensors", "pytorch_model.bin"] {
                let name = format!("{path}/{f}");
                if list.contains_key(&name) && !(f == "pytorch_model.bin" && list.contains_key(&format!("{path}/model.safetensors"))) {
                    files.push(name);
                }
            }
        }
    }
    if list.contains_key("model.safetensors") {
        files.push("model.safetensors".into());
    } else if let Some((size, src)) = list.get("model.safetensors.index.json") {
        fetch_file(repo, dir, "model.safetensors.index.json", *size, src, 0, 0)?;
        let idx: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("model.safetensors.index.json"))?)?;
        let mut shards: Vec<String> = idx["weight_map"]
            .as_object()
            .context("weight_map")?
            .values()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
        shards.sort();
        shards.dedup();
        files.extend(shards);
    } else if list.contains_key("pytorch_model.bin") {
        files.push("pytorch_model.bin".into());
    } else if let Some((size, src)) = list.get("pytorch_model.bin.index.json") {
        // Sharded PyTorch checkpoints (e.g. deepseek-ai/Janus-Pro-7B).
        fetch_file(repo, dir, "pytorch_model.bin.index.json", *size, src, 0, 0)?;
        let idx: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("pytorch_model.bin.index.json"))?)?;
        let mut shards: Vec<String> = idx["weight_map"]
            .as_object()
            .context("weight_map")?
            .values()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
        shards.sort();
        shards.dedup();
        files.extend(shards);
    } else if !gguf_models(&list).is_empty() {
        bail!(
            "{repo} holds GGUF files; pick one as {repo}:QUANT (e.g. {repo}:Q4_K_M): {}",
            gguf_models(&list).iter().map(|f| f.as_str()).collect::<Vec<_>>().join(", ")
        );
    } else {
        bail!("{repo}: no safetensors or pytorch_model.bin weights found");
    }
    let count = files.len();
    for (i, f) in files.iter().enumerate() {
        let (size, src) = list.get(f).with_context(|| format!("{repo}: {f} is not in the listing"))?;
        fetch_file(repo, dir, f, *size, src, i + 1, count)?;
    }
    Ok(())
}

fn human_gb(b: u64) -> String {
    if b >= 1_000_000_000 {
        format!("{:.2} GB", b as f64 / 1e9)
    } else {
        format!("{:.1} MB", b as f64 / 1e6)
    }
}

/// Renders a progress line: `[=====>     ]  42.0%  3.21/7.64 GB  31 MB/s  eta 2m10s`.
fn progress_line(label: &str, done: u64, total: Option<u64>, secs: f64) -> String {
    let rate = done as f64 / secs.max(1e-3);
    let speed = format!("{:5.0} MB/s", rate / 1e6);
    match total {
        Some(t) if t > 0 => {
            let frac = (done as f64 / t as f64).clamp(0.0, 1.0);
            let width = 30usize;
            let filled = (frac * width as f64) as usize;
            let bar: String = (0..width)
                .map(|i| {
                    if i < filled {
                        '='
                    } else if i == filled {
                        '>'
                    } else {
                        ' '
                    }
                })
                .collect();
            let eta = if rate > 0.0 { ((t - done) as f64 / rate) as u64 } else { 0 };
            // The rate is noise for the first second; show no estimate yet.
            let eta_s = if secs < 1.0 {
                "--".to_string()
            } else if eta >= 3600 {
                format!("{}h{:02}m", eta / 3600, (eta % 3600) / 60)
            } else if eta >= 60 {
                format!("{}m{:02}s", eta / 60, eta % 60)
            } else {
                format!("{eta}s")
            };
            format!(
                "  {label} [{bar}] {:5.1}%  {}/{}  {speed}  eta {eta_s}",
                frac * 100.0,
                human_gb(done),
                human_gb(t)
            )
        }
        _ => format!("  {label} {}  {speed}", human_gb(done)),
    }
}

fn fetch_file(repo: &str, dir: &Path, name: &str, expected: Option<u64>, src: &Source, index: usize, count: usize) -> Result<()> {
    use std::io::IsTerminal;
    let path = dir.join(name);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let progress = |done: u64, total: Option<u64>| {
        report(&Progress {
            repo,
            file: name,
            index,
            count,
            done,
            total,
        })
    };
    if let Ok(md) = std::fs::metadata(&path)
        && md.len() > 0
        && expected.is_none_or(|s| s == md.len())
    {
        progress(md.len(), Some(md.len()));
        return Ok(());
    }
    let url = match src {
        Source::Inline(b) => {
            std::fs::write(&path, b).with_context(|| format!("write {}", path.display()))?;
            progress(b.len() as u64, Some(b.len() as u64));
            return Ok(());
        }
        Source::Url(u) => u.as_str(),
    };
    progress(0, expected);
    let (prepo, pname) = (repo.to_string(), name.to_string());
    let t0 = Instant::now();
    let label = format!("[{index}/{count}] {name}");
    let tty = std::io::stderr().is_terminal();
    let mut last = Instant::now();
    let mut last_pct = -1i64;
    let plabel = label.clone();
    let opts = DownloadOptions {
        progress: Some(Box::new(move |done: u64, total: Option<u64>| {
            report(&Progress {
                repo: &prepo,
                file: &pname,
                index,
                count,
                done,
                total,
            });
            let secs = t0.elapsed().as_secs_f64();
            if tty {
                if last.elapsed().as_millis() >= 200 {
                    last = Instant::now();
                    eprint!("\r\x1b[K{}", progress_line(&plabel, done, total, secs));
                }
            } else {
                // Not a terminal: one log line per 5% (or every 30 s when the size is unknown).
                let pct = total.filter(|&t| t > 0).map(|t| (done * 100 / t) as i64 / 5 * 5);
                let due = match pct {
                    Some(p) => p != last_pct,
                    None => last.elapsed().as_secs() >= 30,
                };
                if due {
                    last = Instant::now();
                    last_pct = pct.unwrap_or(-1);
                    eprintln!("{}", progress_line(&plabel, done, total, secs));
                }
            }
        })),
        ..Default::default()
    };
    eprintln!(
        "[fetching {repo}/{name}{}]",
        expected.map(|s| format!(" ({})", human_gb(s))).unwrap_or_default()
    );
    let out = request(url, url.starts_with("https://huggingface.co/"))?
        .download_resumable(&path, opts)
        .map_err(|e| anyhow!("download {repo}/{name}: {e}"))?;
    let secs = t0.elapsed().as_secs_f64().max(1e-3);
    if tty {
        eprint!("\r\x1b[K");
    }
    eprintln!(
        "  {label}: done, {} in {:.0}s ({:.0} MB/s){}",
        human_gb(out.bytes_written + out.resumed_from),
        secs,
        out.bytes_written as f64 / secs / 1e6,
        if out.resumed_from > 0 {
            format!(", resumed from {}", human_gb(out.resumed_from))
        } else {
            String::new()
        }
    );
    if let (Some(exp), Ok(md)) = (expected, std::fs::metadata(&path))
        && md.len() != exp
    {
        bail!("{}: size {} after download, expected {exp}", path.display(), md.len());
    }
    Ok(())
}
