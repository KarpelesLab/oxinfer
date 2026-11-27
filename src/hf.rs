//! Fetching and caching HuggingFace model repositories by name, over rsurl (pure-Rust HTTP).
//!
//! `--model google/gemma-4-E4B-it` resolves to `$OXINFER_CACHE/google--gemma-4-E4B-it` (default
//! `~/.cache/oxinfer/hub`), downloading the config, tokenizer, template and safetensors files that
//! are missing or incomplete. Downloads are resumable. Gated repos need `HF_TOKEN`.
//!
//! Files come from the ATOnline mirror when it carries the repo (`AI/Model:get`, which lists
//! every file with a download URL), otherwise from huggingface.co; `OXINFER_HUB` overrides.
//!
//! Every load asks the hub for the repository's file list, within a few seconds when the cache
//! already holds what the load needs: a hub that does not answer in that time is passed over with
//! one note and the cached files are used. Offline ([`set_offline`], `HF_HUB_OFFLINE=1`) nothing
//! is asked at all. No step of a request waits without a bound (the `net` submodule).

use anyhow::{Context, Result, anyhow, bail};
use rsurl::DownloadOptions;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

mod net;
#[cfg(test)]
mod offline_tests;

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
/// fetched into the cache (if needed) and its directory returned. For `org/name:QUANT` the
/// repository's vision projector comes along when it has one ([`resolve_with`] with
/// [`Mmproj::Auto`](crate::pixtral::Mmproj::Auto)).
pub fn resolve(spec: &str, cache_dir: Option<&Path>) -> Result<PathBuf> {
    resolve_with(spec, cache_dir, &crate::pixtral::Mmproj::Auto)
}

/// [`resolve`] with the model's `mmproj` choice: for a GGUF repository (`org/name:QUANT`),
/// `auto` also fetches one of its top-level `mmproj*.gguf` vision files beside the model
/// ([`resolve_gguf_with`]); `none` or a given file fetch none.
pub fn resolve_with(spec: &str, cache_dir: Option<&Path>, mmproj: &crate::pixtral::Mmproj) -> Result<PathBuf> {
    let p = Path::new(spec);
    if p.is_dir() || crate::gguf::is_gguf(p) {
        return Ok(p.to_path_buf());
    }
    crate::gguf::check_magic(p)?;
    if let Some((repo, quant)) = spec.split_once(':')
        && repo.split('/').count() == 2
    {
        return resolve_gguf_with(repo, quant, cache_dir, mmproj);
    }
    let parts: Vec<&str> = spec.split('/').collect();
    if parts.len() != 2 || parts.iter().any(|s| s.is_empty() || s.starts_with('.')) || spec.starts_with('/') {
        bail!("{spec}: not an existing directory and not a HuggingFace repo id of the form org/name");
    }
    let dir = repo_dir(spec, cache_dir);
    fetch_repo(spec, &dir)?;
    Ok(dir)
}

/// Errors unless `spec` has the form of a repo id (`org/name`).
fn check_repo_id(spec: &str) -> Result<()> {
    let parts: Vec<&str> = spec.split('/').collect();
    if parts.len() != 2 || parts.iter().any(|s| s.is_empty() || s.starts_with('.')) || spec.starts_with('/') {
        bail!("{spec}: not an existing directory and not a HuggingFace repo id of the form org/name");
    }
    Ok(())
}

/// The folder of `repo` in the cache.
fn repo_dir(repo: &str, cache_dir: Option<&Path>) -> PathBuf {
    cache_dir
        .map(Path::to_path_buf)
        .unwrap_or_else(cache_root)
        .join(repo.replace('/', "--"))
}

/// Fetches the files of `names` that the listing has.
fn fetch_present(repo: &str, dir: &Path, list: &Listing, names: &[&str]) -> Result<()> {
    let present: Vec<&&str> = names.iter().filter(|n| list.contains_key(**n)).collect();
    for (i, n) in present.iter().enumerate() {
        let (size, src) = &list[**n];
        fetch_file(repo, dir, n, *size, src, i + 1, present.len())?;
    }
    Ok(())
}

/// Some files of a repo: a local directory is used as is; otherwise the files of `names` that
/// repo `spec` (`org/name`) has are fetched into its cache directory, which is returned. For
/// add-ons such as decision assets, without the weights.
pub fn resolve_files(spec: &str, names: &[&str], cache_dir: Option<&Path>) -> Result<PathBuf> {
    let p = Path::new(spec);
    if p.is_dir() {
        return Ok(p.to_path_buf());
    }
    check_repo_id(spec)?;
    let dir = repo_dir(spec, cache_dir);
    with_listing(spec, &dir, &|list| fetch_present(spec, &dir, list, names))?;
    Ok(dir)
}

/// Every file of repo `spec` under the folder `prefix` (`audio_tokenizer/`), fetched into the
/// repo's cache directory, which is returned: the parts of a model its main download leaves out.
pub fn resolve_prefix(spec: &str, prefix: &str, cache_dir: Option<&Path>) -> Result<PathBuf> {
    check_repo_id(spec)?;
    let dir = repo_dir(spec, cache_dir);
    let prefix = format!("{}/", prefix.trim_end_matches('/'));
    with_listing(spec, &dir, &|list| {
        let mut names: Vec<&str> = list.keys().map(String::as_str).filter(|n| n.starts_with(&prefix)).collect();
        if names.is_empty() {
            bail!("{spec} has no {prefix} folder");
        }
        names.sort_unstable();
        fetch_present(spec, &dir, list, &names)
    })?;
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
        return resolve_gguf_with(repo, quant, cache_dir, &crate::pixtral::Mmproj::None);
    }
    let parts: Vec<&str> = spec.split('/').collect();
    if parts.len() < 2 || parts.iter().any(|s| s.is_empty() || s.starts_with('.')) || spec.starts_with('/') {
        bail!("{spec}: not an existing file and not a HuggingFace path of the form org/name/file");
    }
    let repo = format!("{}/{}", parts[0], parts[1]);
    let dir = repo_dir(&repo, cache_dir);
    with_listing(&repo, &dir, &|list| {
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
        fetch_file(&repo, &dir, &name, *size, src, 1, 1)?;
        Ok(dir.join(name))
    })
}

/// The GGUF files of a listing that hold a model (not vision projectors), sorted.
fn gguf_models(list: &Listing) -> Vec<&String> {
    let mut v: Vec<&String> = list
        .keys()
        .filter(|k| k.ends_with(".gguf") && !crate::gguf::is_mmproj_name(k))
        .collect();
    v.sort();
    v
}

/// `org/name:QUANT` for a repository of GGUF files (as llama.cpp's `-hf`): the file whose name
/// ends in `QUANT.gguf` (case-insensitive, e.g. `Q4_K_M` or `FastMTP-32K`) and its split parts,
/// with the repository's vision projector when it has one ([`resolve_gguf_with`] with
/// [`Mmproj::Auto`](crate::pixtral::Mmproj::Auto)).
pub fn resolve_gguf(repo: &str, quant: &str, cache_dir: Option<&Path>) -> Result<PathBuf> {
    resolve_gguf_with(repo, quant, cache_dir, &crate::pixtral::Mmproj::Auto)
}

/// [`resolve_gguf`] with the model's `mmproj` choice: `auto` also fetches the repository's
/// top-level `mmproj*.gguf` (BF16, else F16, else F32, else another) beside the model, where
/// [`crate::pixtral::mmproj_for`] finds it, unless one is there already; `none` or a given file
/// fetch the model alone.
pub fn resolve_gguf_with(repo: &str, quant: &str, cache_dir: Option<&Path>, mmproj: &crate::pixtral::Mmproj) -> Result<PathBuf> {
    let dir = repo_dir(repo, cache_dir);
    with_listing(repo, &dir, &|list| fetch_gguf(repo, quant, &dir, list, mmproj))
}

/// The fetch of [`resolve_gguf_with`] from one listing. The model's parts are needed; the vision
/// projector is an extra, and a failure to fetch it is a note, not an error (the model then loads
/// without image input, and a later load tries again).
fn fetch_gguf(repo: &str, quant: &str, dir: &Path, list: &Listing, mmproj: &crate::pixtral::Mmproj) -> Result<PathBuf> {
    let (name, parts) = gguf_parts(repo, quant, list)?;
    let proj = gguf_mmproj(list, &name, &mmprojs_present(dir, &name), mmproj);
    let count = parts.len() + usize::from(proj.is_some());
    for (i, p) in parts.iter().enumerate() {
        let (size, src) = list
            .get(p)
            .with_context(|| format!("{repo}: split GGUF part {p} is not in the listing"))?;
        fetch_file(repo, dir, p, *size, src, i + 1, count)?;
    }
    if let Some(p) = &proj {
        let (size, src) = &list[p];
        let have = std::fs::metadata(dir.join(p)).map_or(0, |m| m.len());
        if have == 0 || size.is_some_and(|s| s != have) {
            eprintln!(
                "note: also fetching {p}{}, the repository's vision projector for image input (mmproj none skips it)",
                size.map_or(String::new(), |s| format!(" ({})", human_gb(s)))
            );
        }
        if let Err(e) = fetch_file(repo, dir, p, *size, src, count, count) {
            eprintln!("note: {p} was not fetched ({e:#}); {name} loads without image input until a later load completes it");
        }
    }
    // The shared part also under the name of the part it stands for, as llama.cpp opens the
    // parts by name (oxinfer finds it by its header either way).
    #[cfg(unix)]
    for (link, target) in shared_part_links(&name, &parts) {
        let at = dir.join(&link);
        if !at.exists() && std::os::unix::fs::symlink(&target, &at).is_err() {
            eprintln!("note: could not link {link} to {target}");
        }
    }
    Ok(dir.join(name))
}

/// The `mmproj*.gguf` files already in the cache beside GGUF model `name` of the repository cached
/// in `dir`, as repository paths: in the model's folder and, for a model in a subfolder, the
/// repository's top level (the folders [`crate::pixtral::mmproj_for`] looks in).
fn mmprojs_present(dir: &Path, name: &str) -> Vec<String> {
    let mut folders = vec![String::new()];
    if let Some((sub, _)) = name.rsplit_once('/') {
        folders.push(format!("{sub}/"));
    }
    let mut out = Vec::new();
    for f in folders {
        let Ok(rd) = std::fs::read_dir(dir.join(&f)) else { continue };
        for e in rd.flatten() {
            let n = e.file_name().to_string_lossy().into_owned();
            if crate::gguf::is_mmproj_name(&n) {
                out.push(format!("{f}{n}"));
            }
        }
    }
    out.sort();
    out
}

/// The vision projector of a GGUF repository to fetch beside its model `name`, given the
/// `mmproj*.gguf` files the cache has beside it already (`present`, as repository paths) and the
/// model's `mmproj` choice. With `auto`, one of the listing's top-level `mmproj*.gguf` files:
/// BF16, else F16, else F32, else the first other one; never one in a subfolder (unsloth's
/// `Old_Support/`) or one not named `.gguf`. None with `none` or a given file, when the
/// repository has none, when the model is more than one folder down (where
/// [`crate::pixtral::mmproj_for`] would not find it), or when another projector is present
/// already; a listed one that is present (perhaps partly) is returned, so that it completes.
fn gguf_mmproj(list: &Listing, name: &str, present: &[String], mmproj: &crate::pixtral::Mmproj) -> Option<String> {
    if *mmproj != crate::pixtral::Mmproj::Auto || name.matches('/').count() > 1 {
        return None;
    }
    let mut cands: Vec<&String> = list.keys().filter(|k| !k.contains('/') && crate::gguf::is_mmproj_name(k)).collect();
    if !present.is_empty() {
        return present.iter().find(|p| cands.contains(p)).cloned();
    }
    let rank = |n: &str| {
        let n = n.to_ascii_lowercase();
        if n.contains("bf16") {
            0
        } else if n.contains("f16") {
            1
        } else if n.contains("f32") {
            2
        } else {
            3
        }
    };
    cands.sort_by(|a, b| rank(a).cmp(&rank(b)).then_with(|| a.cmp(b)));
    cands.first().map(|s| s.to_string())
}

/// The split parts of `name` that `parts` fetches under another name (see [`gguf_parts`]), as
/// (part name, fetched name).
#[cfg_attr(not(unix), allow(dead_code))]
fn shared_part_links(name: &str, parts: &[String]) -> Vec<(String, String)> {
    let Some((pre, _, count, digits)) = crate::gguf::split_name(name) else {
        return Vec::new();
    };
    (1..=count)
        .zip(parts)
        .map(|(i, p)| (format!("{pre}-{i:0digits$}-of-{count:0digits$}.gguf"), p.clone()))
        .filter(|(want, p)| want != p && !p.contains('/'))
        .collect()
}

/// The GGUF file of `repo` for `quant` and its parts (itself alone unless it is split).
fn gguf_parts(repo: &str, quant: &str, list: &Listing) -> Result<(String, Vec<String>)> {
    let ggufs = gguf_models(list);
    let want = format!("{}.gguf", quant.to_ascii_lowercase());
    // A split model is matched by its first part's name without `-00001-of-0000N`; the later
    // parts are not models of their own.
    let model_name = |f: &str| -> Option<String> {
        match crate::gguf::split_name(f) {
            Some((pre, 1, _, _)) => Some(format!("{pre}.gguf")),
            Some(_) => None,
            None => Some(f.to_string()),
        }
    };
    let hits: Vec<&&String> = ggufs
        .iter()
        .filter(|f| {
            let Some(f) = model_name(f) else { return false };
            let f = f.to_ascii_lowercase();
            f.ends_with(&want) && f[..f.len() - want.len()].ends_with(['-', '.', '_', '/'])
        })
        .collect();
    // A repository's draft sidecars (`mtp-*.gguf`, `dflash-*.gguf`) often carry the same quant
    // suffix as its models: with several matches, they give way to the model.
    let is_sidecar = |f: &str| {
        let base = f.rsplit('/').next().unwrap_or(f).to_ascii_lowercase();
        ["mtp-", "dflash-", "dspark-"].iter().any(|p| base.starts_with(p))
    };
    let models: Vec<&&String> = hits.iter().copied().filter(|f| !is_sidecar(f)).collect();
    let hits = if hits.len() > 1 && !models.is_empty() { models } else { hits };
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
    let mut parts: Vec<String> = match crate::gguf::split_name(&name) {
        Some((pre, _, count, digits)) => (1..=count)
            .map(|i| format!("{pre}-{i:0digits$}-of-{count:0digits$}.gguf"))
            .collect(),
        None => vec![name.clone()],
    };
    // A part the listing does not have can be a file every quantization of the repository shares
    // and that is named as none of them: ISTA-DASLab's DeepSeek V4.1 builds count
    // `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` (the engram tables, at the repository's root) as
    // their last part. One unsplit GGUF whose name says "engram", in the folder of the parts,
    // stands for one missing part.
    let folder = |f: &str| f.rsplit_once('/').map_or("", |(d, _)| d).to_string();
    let missing: Vec<usize> = (0..parts.len()).filter(|&i| !list.contains_key(&parts[i])).collect();
    let shared: Vec<&&String> = ggufs
        .iter()
        .filter(|f| {
            let base = f.rsplit('/').next().unwrap_or(f);
            crate::gguf::split_name(base).is_none() && base.to_ascii_lowercase().contains("engram") && folder(f) == folder(&name)
        })
        .collect();
    if let ([i], [f]) = (missing.as_slice(), shared.as_slice()) {
        parts[*i] = f.to_string();
    }
    Ok((name, parts))
}

/// One file of a [`Plan`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct PlanFile {
    /// The path in the repository.
    pub name: String,
    /// Its size, when the listing gives it.
    pub size: Option<u64>,
    /// Bytes of it already in the cache (a whole file, or the part an interrupted download left).
    pub have: u64,
}

/// What fetching a model takes ([`plan`]).
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct Plan {
    /// The repository (`org/name`).
    pub repo: String,
    /// Its folder in the cache.
    pub dir: PathBuf,
    /// The files [`resolve`] fetches, in order (the small configuration files it reads to decide
    /// are not counted).
    pub files: Vec<PlanFile>,
}

impl Plan {
    /// Bytes of all the files (those of unknown size not counted).
    pub fn total(&self) -> u64 {
        self.files.iter().map(|f| f.size.unwrap_or(f.have)).sum()
    }
    /// Bytes still to download.
    pub fn missing(&self) -> u64 {
        self.files.iter().map(|f| f.size.map_or(0, |s| s.saturating_sub(f.have))).sum()
    }
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "repo": self.repo,
            "dir": self.dir.display().to_string(),
            "total": self.total(),
            "missing": self.missing(),
            "files": self.files.iter().map(|f| serde_json::json!({"name": f.name, "size": f.size, "have": f.have})).collect::<Vec<_>>(),
        })
    }
}

/// The files a [`resolve`] of `spec` (`org/name` or `org/name:QUANT`) would fetch, with their
/// sizes and what the cache already has, without downloading anything (the few small files that
/// decide which weights are needed, such as a weight index, are read into memory). Errors for a
/// local path.
pub fn plan(spec: &str, cache_dir: Option<&Path>) -> Result<Plan> {
    if Path::new(spec).exists() {
        bail!("{spec}: a local path, nothing to fetch");
    }
    let (repo, quant) = match spec.split_once(':') {
        Some((r, q)) if r.split('/').count() == 2 => (r, Some(q)),
        _ => (spec, None),
    };
    let parts: Vec<&str> = repo.split('/').collect();
    if parts.len() != 2 || parts.iter().any(|s| s.is_empty() || s.starts_with('.')) || repo.starts_with('/') {
        bail!("{spec}: not a HuggingFace repo id of the form org/name (or org/name:QUANT)");
    }
    let dir = repo_dir(repo, cache_dir);
    let files = with_listing(repo, &dir, &|list| {
        let names = match quant {
            Some(q) => {
                let (name, mut parts) = gguf_parts(repo, q, list)?;
                parts.extend(gguf_mmproj(
                    list,
                    &name,
                    &mmprojs_present(&dir, &name),
                    &crate::pixtral::Mmproj::Auto,
                ));
                parts
            }
            None => repo_files(repo, list, &mut |name| peek_meta(repo, &dir, list, name), Some(&dir))?,
        };
        // A file the cache's own listing lacks is one the cache cannot give.
        if let Some(gone) = names.iter().find(|n| is_cached_listing(list) && !list.contains_key(*n)) {
            bail!("{repo}: {gone} is not in the cache");
        }
        Ok(names
            .into_iter()
            .map(|name| {
                let size = list.get(&name).and_then(|(s, _)| *s);
                let have = std::fs::metadata(dir.join(&name)).map(|m| m.len()).unwrap_or(0);
                let have = size.map_or(have, |s| have.min(s));
                PlanFile { name, size, have }
            })
            .collect())
    })?;
    Ok(Plan {
        repo: repo.to_string(),
        dir,
        files,
    })
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

/// The body of a plain GET of `url` (no HuggingFace token), for image URLs in decision requests
/// and release archives: any size, so without a deadline for the whole of it, but with the
/// client's bounds on the lookup, each connection attempt and each silence of the server.
pub fn http_get(url: &str) -> Result<Vec<u8>> {
    let resp = net::request(&net::cfg(), url, false)?.send().map_err(|e| anyhow!("{url}: {e}"))?;
    if resp.status != 200 {
        bail!("{url}: HTTP {}", resp.status);
    }
    Ok(resp.body)
}

/// 0 = unset (the environment decides), 1 = offline, 2 = online.
static OFFLINE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Offline mode for later loads, overriding the environment: a repo id is served from the cache
/// without asking any hub, and a load the cache cannot serve fails at once.
pub fn set_offline(offline: bool) {
    OFFLINE.store(if offline { 1 } else { 2 }, std::sync::atomic::Ordering::Relaxed);
}

/// Whether the hub client is offline: what [`set_offline`] was given, else `HF_HUB_OFFLINE` or
/// `OXINFER_OFFLINE` set to `1`, `true`, `yes` or `on`.
pub fn offline() -> bool {
    match OFFLINE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => ["HF_HUB_OFFLINE", "OXINFER_OFFLINE"]
            .iter()
            .any(|v| std::env::var(v).is_ok_and(|v| net::truthy(&v))),
    }
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
    /// A file of the cache's own listing ([`cached_listing`]): on disk, with nowhere to fetch it
    /// from.
    Cached,
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

/// A status that says the server could not answer now (as opposed to "no"): the hub counts as
/// not reachable.
fn try_later(status: u16) -> bool {
    status == 408 || status == 429 || status >= 500
}

/// Lists `repo` on the mirror within `total`; Ok(None) when the mirror does not carry it.
fn list_mirror(cfg: &net::Cfg, repo: &str, total: Duration) -> Result<Option<Listing>> {
    let api = format!("{}?key={repo}", cfg.mirror_api);
    let resp = net::send(net::request(cfg, &api, false)?, total).with_context(|| api.clone())?;
    match resp.status {
        200 => {}
        404 => return Ok(None),
        s if try_later(s) => return Err(net::unreachable(format!("{api}: HTTP {s}"))),
        s => bail!("{api}: HTTP {s}"),
    }
    // Not the mirror's JSON: something on the way answered in its place (a captive portal).
    let reply: MirrorReply =
        serde_json::from_slice(&resp.body).map_err(|e| net::unreachable(format!("{api}: not the mirror's answer ({e})")))?;
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

/// Lists `repo` on huggingface.co (or the `HF_ENDPOINT` server) within `total`.
fn list_hf(cfg: &net::Cfg, repo: &str, total: Duration) -> Result<Listing> {
    let hf = &cfg.hf;
    let api = format!("{hf}/api/models/{repo}?blobs=true");
    let resp = net::send(net::request(cfg, &api, true)?, total).with_context(|| api.clone())?;
    match resp.status {
        200 => {}
        401 | 403 => bail!(
            "{repo}: access denied (HTTP {}). If the repo is gated, accept its license on huggingface.co and set HF_TOKEN.",
            resp.status
        ),
        404 => bail!("{repo}: repository not found on {hf}"),
        s if try_later(s) => return Err(net::unreachable(format!("{api}: HTTP {s}"))),
        s => bail!("{api}: HTTP {s}"),
    }
    let info: RepoInfo = serde_json::from_slice(&resp.body).map_err(|e| net::unreachable(format!("{api}: not the hub's answer ({e})")))?;
    Ok(info
        .siblings
        .into_iter()
        .map(|s| {
            let url = format!("{hf}/{repo}/resolve/main/{}", url_path(&s.rfilename));
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

/// The hub's listing of `repo`, the whole of it within `total`. An error that is
/// [`net::is_unreachable`] says that no hub answered; any other is a hub's answer.
fn listing(cfg: &net::Cfg, repo: &str, total: Duration) -> Result<Listing> {
    let t0 = Instant::now();
    let left = || total.saturating_sub(t0.elapsed());
    let hub = match cfg.hub {
        Some(h) => h,
        None => hub()?,
    };
    match hub {
        Hub::HuggingFace => list_hf(cfg, repo, total),
        Hub::Mirror => list_mirror(cfg, repo, total)?.with_context(|| format!("{repo}: not on the mirror ({})", cfg.mirror_api)),
        // The mirror gets half of the time, so that huggingface.co can still be asked.
        Hub::Auto => match list_mirror(cfg, repo, total / 2) {
            Ok(Some(l)) => Ok(l),
            Ok(None) => list_hf(cfg, repo, left()),
            Err(mirror) => match list_hf(cfg, repo, left()) {
                Ok(l) => {
                    eprintln!("[mirror unavailable ({mirror:#}); using {}]", cfg.hf);
                    Ok(l)
                }
                // A mirror that did not answer may be the one place that carries the repo:
                // whatever the other hub says, the listing is not known.
                Err(hf) if net::is_unreachable(&mirror) => Err(net::unreachable(format!("{mirror:#}; {hf:#}"))),
                Err(hf) => Err(hf),
            },
        },
    }
}

/// The cache's own listing of a repository: every file of its folder `dir` under the name the
/// repository gives it, with its size on disk. A download in progress is kept in a
/// `.rsurlpart` file beside the final name until it is whole, so a file under its final name is
/// a complete one; empty files (never a download's result) and the part files are left out.
fn cached_listing(dir: &Path) -> Listing {
    fn walk(dir: &Path, prefix: &str, out: &mut Listing) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            let name = format!("{prefix}{}", e.file_name().to_string_lossy());
            // Through links: a shared GGUF part is also there under the name of the part it
            // stands for.
            let Ok(md) = std::fs::metadata(e.path()) else { continue };
            if md.is_dir() {
                walk(&e.path(), &format!("{name}/"), out);
            } else if md.len() > 0 && !name.ends_with(".rsurlpart") {
                out.insert(name, (Some(md.len()), Source::Cached));
            }
        }
    }
    let mut out = Listing::new();
    walk(dir, "", &mut out);
    out
}

/// Whether `list` is the cache's own ([`cached_listing`]) rather than a hub's.
fn is_cached_listing(list: &Listing) -> bool {
    list.values().any(|(_, s)| matches!(s, Source::Cached))
}

/// Until when each hub that did not answer is left alone (see [`with_listing`]), by endpoint.
static HUB_DOWN: std::sync::Mutex<Vec<(String, Instant)>> = std::sync::Mutex::new(Vec::new());

/// Runs `f`, the part of a load that reads a repository's listing and fetches from it, on the
/// hub's listing of `repo` or, when no hub is to be asked or none answers, on the cache's own
/// ([`cached_listing`] of `dir`).
///
/// A load asks the hub every time, as huggingface_hub does, so that a repository that changed is
/// fetched again; what bounds the cost is the time given to the question. When the cache holds
/// files of the repository the hub has `check` (5 s) for the whole listing, both hubs together,
/// and a hub that does not answer is passed over: one note, then the cached files, and for the
/// next minute loads the cache can serve do not ask again. Only when the cache cannot serve the
/// load (nothing cached, or `f` fails on the cached files) does the listing get `meta` (30 s),
/// and its failure is the load's. Offline, `f` runs on the cached files alone. `f` may run twice.
fn with_listing<T>(repo: &str, dir: &Path, f: &dyn Fn(&Listing) -> Result<T>) -> Result<T> {
    let cfg = net::cfg();
    let cached = cached_listing(dir);
    let at = dir.display();
    if cfg.offline {
        const WHY: &str = "offline (HF_HUB_OFFLINE, OXINFER_OFFLINE or --offline)";
        if cached.is_empty() {
            bail!("{repo}: {WHY}, and the repository is not in the cache ({at})");
        }
        return f(&cached).map_err(|e| anyhow!("{repo}: {WHY}, and the cache ({at}) cannot serve this load: {e:#}"));
    }
    let key = format!("{} {}", cfg.mirror_api, cfg.hf);
    // Why the cached files do not do, when they were tried.
    let mut lacking = None;
    if !cached.is_empty() {
        let down = {
            let d = HUB_DOWN.lock().unwrap_or_else(|e| e.into_inner());
            d.iter().any(|(k, until)| *k == key && Instant::now() < *until)
        };
        if down {
            match f(&cached) {
                Ok(v) => return Ok(v),
                Err(e) => lacking = Some(e),
            }
        } else {
            match listing(&cfg, repo, cfg.check) {
                Ok(list) => return f(&list),
                Err(e) if net::is_unreachable(&e) => match f(&cached) {
                    Ok(v) => {
                        let mut d = HUB_DOWN.lock().unwrap_or_else(|e| e.into_inner());
                        d.retain(|(k, _)| *k != key);
                        d.push((key, Instant::now() + cfg.down_for));
                        eprintln!(
                            "[the hub did not answer ({e:#}); using the cached files of {repo}, and not asking again for {} s]",
                            cfg.down_for.as_secs()
                        );
                        return Ok(v);
                    }
                    Err(e) => lacking = Some(e),
                },
                Err(e) => return Err(e),
            }
        }
    }
    // The cache cannot serve this load: the listing is needed, with the time that takes.
    match (listing(&cfg, repo, cfg.meta), lacking) {
        (Ok(list), _) => {
            // It answers again.
            HUB_DOWN.lock().unwrap_or_else(|e| e.into_inner()).retain(|(k, _)| *k != key);
            f(&list)
        }
        (Err(e), Some(lacking)) if net::is_unreachable(&e) => {
            bail!("{repo}: the hub did not answer ({e:#}), and the cache ({at}) cannot serve this load: {lacking:#}")
        }
        (Err(e), _) => Err(e),
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

fn fetch_repo(repo: &str, dir: &Path) -> Result<()> {
    with_listing(repo, dir, &|list| {
        let files = repo_files(repo, list, &mut |name| fetch_meta(repo, dir, list, name), Some(dir))?;
        fetch_listed(repo, dir, list, &files)
    })
}

/// Reads a small file of the listing that decides which files a fetch takes (`modules.json`,
/// a weight index), fetching it into the repository's folder.
type Meta<'a> = dyn FnMut(&str) -> Result<Vec<u8>> + 'a;

fn fetch_meta(repo: &str, dir: &Path, list: &Listing, name: &str) -> Result<Vec<u8>> {
    let (size, src) = list.get(name).with_context(|| format!("{repo}: {name} is not in the listing"))?;
    fetch_file(repo, dir, name, *size, src, 0, 0)?;
    std::fs::read(dir.join(name)).with_context(|| format!("read {}", dir.join(name).display()))
}

/// The same without writing anything: the cached copy, else the file fetched into memory.
fn peek_meta(repo: &str, dir: &Path, list: &Listing, name: &str) -> Result<Vec<u8>> {
    let (size, src) = list.get(name).with_context(|| format!("{repo}: {name} is not in the listing"))?;
    if let Ok(b) = std::fs::read(dir.join(name))
        && size.is_none_or(|s| s == b.len() as u64)
    {
        return Ok(b);
    }
    match src {
        Source::Inline(b) => Ok(b.clone()),
        Source::Cached => bail!("{repo}/{name}: not whole in the cache"),
        Source::Url(u) => {
            let cfg = net::cfg();
            let resp = net::send(net::request(&cfg, u, cfg.is_hf(u))?, cfg.meta).with_context(|| format!("{repo}/{name}"))?;
            if resp.status != 200 {
                bail!("{repo}/{name}: HTTP {}", resp.status);
            }
            Ok(resp.body)
        }
    }
}

/// Fetches `files` of the listing (counted 1 to N in the progress reports).
fn fetch_listed(repo: &str, dir: &Path, list: &Listing, files: &[String]) -> Result<()> {
    let count = files.len();
    for (i, f) in files.iter().enumerate() {
        let (size, src) = list.get(f).with_context(|| format!("{repo}: {f} is not in the listing"))?;
        fetch_file(repo, dir, f, *size, src, i + 1, count)?;
    }
    Ok(())
}

/// The files a fetch of the repository takes (besides the small ones `meta` reads).
fn repo_files(repo: &str, list: &Listing, meta: &mut Meta, dir: Option<&Path>) -> Result<Vec<String>> {
    if list.contains_key("model_index.json") {
        return diffusers_component_files(list, &[], meta, dir);
    }
    if is_anemll_listing(list) {
        let mut files: Vec<String> = list.keys().filter(|k| anemll_wanted(k)).cloned().collect();
        files.sort();
        return Ok(files);
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
    if list.contains_key("modules.json") {
        let modules: serde_json::Value = serde_json::from_slice(&meta("modules.json")?)?;
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
            // A dense module named `X_N` may have siblings `X_M`: the same projection to other
            // output sizes (stella's `2_Dense_256` ... `2_Dense_8192`), a few MB each.
            let mut paths = vec![path.to_string()];
            if let Some((stem, n)) = path.rsplit_once('_')
                && n.parse::<usize>().is_ok()
            {
                let mut sib: Vec<String> = list
                    .keys()
                    .filter_map(|k| k.strip_suffix("/config.json"))
                    .filter(|p| *p != path)
                    .filter(|p| {
                        p.strip_prefix(stem)
                            .and_then(|r| r.strip_prefix('_'))
                            .is_some_and(|m| m.parse::<usize>().is_ok())
                    })
                    .map(str::to_string)
                    .collect();
                sib.sort();
                paths.extend(sib);
            }
            for path in &paths {
                for f in ["config.json", "model.safetensors", "pytorch_model.bin"] {
                    let name = format!("{path}/{f}");
                    if list.contains_key(&name) && !(f == "pytorch_model.bin" && list.contains_key(&format!("{path}/model.safetensors"))) {
                        files.push(name);
                    }
                }
            }
        }
    }
    // Mistral's own layout (`consolidated*.safetensors`, read by transformers through
    // config.json's `transformers_weights`) when there is no transformers-named copy.
    let index = ["model.safetensors.index.json", "consolidated.safetensors.index.json"]
        .into_iter()
        .find(|n| list.contains_key(*n) && (n.starts_with("model") || list.contains_key("config.json")));
    if list.contains_key("model.safetensors") {
        files.push("model.safetensors".into());
    } else if let Some(index) = index {
        let idx: serde_json::Value = serde_json::from_slice(&meta(index)?)?;
        let mut shards: Vec<String> = idx["weight_map"]
            .as_object()
            .context("weight_map")?
            .values()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
        shards.sort();
        shards.dedup();
        files.extend(shards);
    } else if list.keys().any(|n| crate::safetensors::is_shard_name(n)) {
        // Shards without an index (baidu/ERNIE-5.5-Flash-Base).
        let mut shards: Vec<String> = list.keys().filter(|n| crate::safetensors::is_shard_name(n)).cloned().collect();
        shards.sort();
        files.extend(shards);
    } else if list.contains_key("diffusion_pytorch_model.safetensors") {
        // A single diffusers component (a ControlNet repo).
        files.push("diffusion_pytorch_model.safetensors".into());
    } else if list.contains_key("pytorch_model.bin") {
        files.push("pytorch_model.bin".into());
    } else if list.contains_key("pytorch_model.bin.index.json") {
        // Sharded PyTorch checkpoints (e.g. deepseek-ai/Janus-Pro-7B).
        let idx: serde_json::Value = serde_json::from_slice(&meta("pytorch_model.bin.index.json")?)?;
        let mut shards: Vec<String> = idx["weight_map"]
            .as_object()
            .context("weight_map")?
            .values()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
        shards.sort();
        shards.dedup();
        files.extend(shards);
    } else if !gguf_models(list).is_empty() {
        bail!(
            "{repo} holds GGUF files; pick one as {repo}:QUANT (e.g. {repo}:Q4_K_M): {}",
            gguf_models(list).iter().map(|f| f.as_str()).collect::<Vec<_>>().join(", ")
        );
    } else {
        bail!("{repo}: no safetensors or pytorch_model.bin weights found");
    }
    Ok(files)
}

/// A diffusers pipeline repository (`model_index.json`) without the components in `skip` (e.g.
/// `["transformer"]` when its weights come from elsewhere): a local directory is used as is.
pub fn resolve_diffusers(spec: &str, skip: &[&str], cache_dir: Option<&Path>) -> Result<PathBuf> {
    let p = Path::new(spec);
    if p.is_dir() {
        return Ok(p.to_path_buf());
    }
    check_repo_id(spec)?;
    let dir = repo_dir(spec, cache_dir);
    with_listing(spec, &dir, &|list| {
        if !list.contains_key("model_index.json") {
            bail!("{spec}: not a diffusers pipeline (no model_index.json)");
        }
        let files = diffusers_component_files(list, skip, &mut |name| fetch_meta(spec, &dir, list, name), Some(&dir))?;
        fetch_listed(spec, &dir, list, &files)
    })?;
    Ok(dir)
}

/// The files of a diffusers pipeline: `model_index.json` and, for each component it names (but
/// those in `skip`), the component folder's configs, tokenizer files and weights (sharded or
/// not; the full-precision file over an `.fp16` variant, safetensors over `.bin`). Top-level
/// single-file checkpoints (`flux1-dev.safetensors`, `ae.safetensors`, ...) are left out.
fn diffusers_component_files(list: &Listing, skip: &[&str], meta: &mut Meta, dir: Option<&Path>) -> Result<Vec<String>> {
    let index: serde_json::Value = serde_json::from_slice(&meta("model_index.json")?)?;
    // MOSS-Image's text encoder is Qwen/Qwen3.5-9B unchanged: not fetched again when the cache
    // holds that repo (the pipeline reads it from there).
    let mut skip = skip.to_vec();
    if index["_class_name"] == "MOSSImagePipeline"
        && dir.is_some_and(|dir| {
            dir.parent().is_some_and(|root| {
                let q = root.join(crate::moss_image::QWEN35_9B.replace('/', "--"));
                ["model.safetensors.index.json", "model.safetensors"]
                    .iter()
                    .any(|n| q.join(n).exists())
            }) && !dir.join("text_encoder").join("model.safetensors.index.json").exists()
        })
    {
        skip.push("text_encoder");
    }
    let mut files: Vec<String> = Vec::new();
    for (comp, v) in index.as_object().context("model_index.json is not an object")? {
        let named = v.as_array().is_some_and(|a| a.first().is_some_and(|x| !x.is_null()));
        if comp.starts_with('_') || skip.contains(&comp.as_str()) || !named {
            continue;
        }
        let pre = format!("{comp}/");
        let mut names: Vec<&String> = list
            .keys()
            .filter(|k| k.starts_with(&pre) && !k[pre.len()..].contains('/'))
            .collect();
        names.sort();
        let has = |n: &str| list.contains_key(&format!("{pre}{n}"));
        let any_st = names.iter().any(|n| n.ends_with(".safetensors"));
        let weights_index = names.iter().find(|n| n.ends_with(".safetensors.index.json")).map(|s| s.to_string());
        for n in names {
            let base = &n[pre.len()..];
            let keep = if base.ends_with(".safetensors") {
                // Shards come from the index below; prefer full precision over `.fp16`.
                weights_index.is_none()
                    && base
                        .strip_suffix(".fp16.safetensors")
                        .is_none_or(|full| !has(&format!("{full}.safetensors")))
            } else if base.ends_with(".bin") {
                !any_st
            } else {
                [".json", ".txt", ".model", ".tiktoken"].iter().any(|e| base.ends_with(e))
            };
            if keep {
                files.push(n.clone());
            }
        }
        if let Some(idx) = weights_index {
            let v: serde_json::Value = serde_json::from_slice(&meta(&idx)?)?;
            let mut shards: Vec<String> = v["weight_map"]
                .as_object()
                .context("weight_map")?
                .values()
                .filter_map(|x| x.as_str().map(|s| format!("{pre}{s}")))
                .collect();
            shards.sort();
            shards.dedup();
            files.extend(shards);
        }
    }
    Ok(files)
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
        Source::Cached => bail!("{repo}/{name}: not whole in the cache"),
        Source::Url(u) => u.as_str(),
    };
    let cfg = net::cfg();
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
        // A connection that says nothing for `stall` is dropped (the request's read timeout)
        // and the download resumes on a new one from the bytes it has, a few times over: the
        // count starts again whenever an attempt adds bytes, so only a download that cannot
        // advance at all gives up.
        max_retries: cfg.retries,
        initial_backoff: cfg.backoff,
        max_backoff: cfg.backoff * 16,
        ..Default::default()
    };
    eprintln!(
        "[fetching {repo}/{name}{}]",
        expected.map(|s| format!(" ({})", human_gb(s))).unwrap_or_default()
    );
    let out = net::request(&cfg, url, cfg.is_hf(url))?
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

#[cfg(test)]
mod tests {
    use super::*;

    fn listing_of(names: &[&str]) -> Listing {
        names
            .iter()
            .map(|n| (n.to_string(), (Some(1), Source::Inline(Vec::new()))))
            .collect()
    }

    #[test]
    fn gguf_mmproj_choice() {
        use crate::pixtral::Mmproj;
        let auto = Mmproj::Auto;
        // unsloth/Inkling-2-Mini-GGUF's layout.
        let mut names = vec![
            "Inkling-2-Mini-IQ3_XXS.gguf",
            "Inkling-2-Mini-Q4_K_M.gguf",
            "mmproj-F32.gguf",
            "mmproj-F16.gguf",
            "mmproj-BF16.gguf",
            "Old_Support/mmproj-BF16.gguf_file",
            "Old_Support/mmproj-F16.gguf_file",
            "README.md",
        ];
        let list = listing_of(&names);
        let m = "Inkling-2-Mini-IQ3_XXS.gguf";
        assert_eq!(gguf_mmproj(&list, m, &[], &auto).as_deref(), Some("mmproj-BF16.gguf"));
        // The model choice is unchanged by the projectors.
        assert_eq!(gguf_parts("r/x", "IQ3_XXS", &list).unwrap().1, [m]);
        // BF16, then F16, then F32, then any other.
        names.retain(|n| *n != "mmproj-BF16.gguf");
        assert_eq!(gguf_mmproj(&listing_of(&names), m, &[], &auto).as_deref(), Some("mmproj-F16.gguf"));
        names.retain(|n| *n != "mmproj-F16.gguf");
        assert_eq!(gguf_mmproj(&listing_of(&names), m, &[], &auto).as_deref(), Some("mmproj-F32.gguf"));
        names.retain(|n| *n != "mmproj-F32.gguf");
        names.push("mmproj-model-Q8_0.gguf");
        assert_eq!(
            gguf_mmproj(&listing_of(&names), m, &[], &auto).as_deref(),
            Some("mmproj-model-Q8_0.gguf")
        );
        // Only files in subfolders, or not named .gguf: none.
        let only_old = listing_of(&[
            m,
            "Old_Support/mmproj-BF16.gguf_file",
            "Old_Support/mmproj-F16.gguf",
            "mmproj-F16.gguf_file",
        ]);
        assert_eq!(gguf_mmproj(&only_old, m, &[], &auto), None);
        // A text-only repository: none.
        assert_eq!(gguf_mmproj(&listing_of(&[m]), m, &[], &auto), None);
        // The mmproj option: none, or a file of the user's, fetch nothing.
        assert_eq!(gguf_mmproj(&list, m, &[], &Mmproj::None), None);
        assert_eq!(gguf_mmproj(&list, m, &[], &Mmproj::File("/x/mmproj-F16.gguf".into())), None);
        // A projector already beside the model: none fetched, whatever its name; a listed one
        // that is present (perhaps an interrupted download) is completed instead.
        assert_eq!(gguf_mmproj(&list, m, &["mmproj-mine.gguf".into()], &auto), None);
        assert_eq!(
            gguf_mmproj(&list, m, &["mmproj-F16.gguf".into()], &auto).as_deref(),
            Some("mmproj-F16.gguf")
        );
        // A split model one folder down (unsloth's large models): the top-level projector, which
        // mmproj_for finds in the folder above; deeper, none.
        let split = listing_of(&[
            "Q4_K_M/x-Q4_K_M-00001-of-00002.gguf",
            "Q4_K_M/x-Q4_K_M-00002-of-00002.gguf",
            "mmproj-F16.gguf",
        ]);
        assert_eq!(
            gguf_mmproj(&split, "Q4_K_M/x-Q4_K_M-00001-of-00002.gguf", &[], &auto).as_deref(),
            Some("mmproj-F16.gguf")
        );
        assert_eq!(gguf_mmproj(&split, "a/b/x-Q4_K_M.gguf", &[], &auto), None);
        assert_eq!(
            gguf_mmproj(
                &split,
                "Q4_K_M/x-Q4_K_M-00001-of-00002.gguf",
                &["Q4_K_M/mmproj-x.gguf".into()],
                &auto
            ),
            None
        );
    }

    #[test]
    fn mmprojs_on_disk() {
        let dir = std::env::temp_dir().join(format!("oxinfer-mmproj-present-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("Q4")).unwrap();
        let m = "Q4/x-Q4-00001-of-00002.gguf";
        assert!(mmprojs_present(&dir, m).is_empty());
        for f in ["mmproj-F16.gguf", "mmproj-F16.gguf_file", "x.gguf", "Q4/mmproj-own.gguf"] {
            std::fs::write(dir.join(f), b"x").unwrap();
        }
        assert_eq!(mmprojs_present(&dir, m), ["Q4/mmproj-own.gguf", "mmproj-F16.gguf"]);
        assert_eq!(mmprojs_present(&dir, "x.gguf"), ["mmproj-F16.gguf"]);
        // What resolve_gguf_with fetches is what mmproj_for then finds.
        assert_eq!(crate::pixtral::mmproj_for(&dir.join("x.gguf")), Some(dir.join("mmproj-F16.gguf")));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn shared_engram_part() {
        // ISTA-DASLab's DeepSeek V4.1 builds: five named parts of six, the sixth shared.
        let pre = "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M";
        let mut names: Vec<String> = (1..=5).map(|i| format!("{pre}-0000{i}-of-00006.gguf")).collect();
        names.push("DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00001-of-00006.gguf".into());
        names.push("DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf".into());
        names.push("mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf".into());
        let list = listing_of(&names.iter().map(String::as_str).collect::<Vec<_>>());
        let (name, parts) = gguf_parts("r/x", "IQ1_M", &list).unwrap();
        assert_eq!(name, format!("{pre}-00001-of-00006.gguf"));
        assert_eq!(parts.len(), 6);
        assert_eq!(parts[5], "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf");
        assert_eq!(
            shared_part_links(&name, &parts),
            vec![(
                format!("{pre}-00006-of-00006.gguf"),
                "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf".to_string()
            )]
        );
        // Every part listed: nothing replaced, nothing linked.
        let list = listing_of(&["x-Q4-00001-of-00002.gguf", "x-Q4-00002-of-00002.gguf", "Engram.gguf"]);
        let (name, parts) = gguf_parts("r/x", "Q4", &list).unwrap();
        assert_eq!(parts, ["x-Q4-00001-of-00002.gguf", "x-Q4-00002-of-00002.gguf"]);
        assert!(shared_part_links(&name, &parts).is_empty());
    }
}
