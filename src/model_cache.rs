//! The model cache on disk: the folders repo ids are fetched into (`$OXINFER_CACHE`, default
//! `~/.cache/oxinfer/hub`, one `org--name` folder per repository; see [`crate::hf`]), listed with
//! their size and when they were last used, and removed one at a time (`oxinfer models list`,
//! `oxinfer models rm`, the web GUI's storage page, `oxi_cache_list` / `oxi_cache_remove` in the C
//! API).
//!
//! Nothing here fetches anything, so it works without the `hf` feature. [`remove`] only ever
//! deletes a direct child of the cache directory, named by its folder name (`org--name`) or its
//! repo id (`org/name`): names with path separators, `..`, a leading dot or a NUL are refused, and
//! the folder's resolved location must be inside the resolved cache directory (a symbolic link in
//! the cache is removed as a link, never followed).

use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

/// The file a load writes in a model's folder to record when it was last used ([`touch`]).
pub const LAST_USED_FILE: &str = ".oxinfer-last-used";

/// One folder of the cache.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct Entry {
    /// The folder name (`Qwen--Qwen3.5-0.8B`).
    pub name: String,
    /// The repo id it holds (`Qwen/Qwen3.5-0.8B`), for folders named `org--name`.
    pub repo: Option<String>,
    pub path: PathBuf,
    /// Bytes of its files (symbolic links not followed).
    pub bytes: u64,
    pub files: usize,
    /// Seconds since the epoch of the newest file.
    pub modified: Option<u64>,
    /// Seconds since the epoch of the last load by oxinfer ([`touch`]), when recorded.
    pub last_used: Option<u64>,
    /// The GGUF model files in it (relative paths; vision projectors left out), sorted.
    pub ggufs: Vec<String>,
    /// It holds a model's configuration (`config.json`, `model_index.json`, `modules.json`,
    /// `meta.yaml`) or a GGUF file: something a loader can open, though weights may still be
    /// missing (an interrupted download).
    pub usable: bool,
}

impl Entry {
    /// The model specs that load what is in the folder: `org/name` for a repository with a
    /// configuration, `org/name:QUANT` for each GGUF file (or its path when no quant suffix can
    /// be told), the folder's path for the others.
    pub fn specs(&self) -> Vec<String> {
        let mut out = Vec::new();
        for g in &self.ggufs {
            match (&self.repo, gguf_quant(g)) {
                (Some(r), Some(q)) => out.push(format!("{r}:{q}")),
                _ => out.push(self.path.join(g).to_string_lossy().into_owned()),
            }
        }
        if self.ggufs.is_empty() || has_config(&self.path) {
            match &self.repo {
                Some(r) if self.ggufs.is_empty() => out.push(r.clone()),
                _ => out.push(self.path.to_string_lossy().into_owned()),
            }
        }
        out.dedup();
        out
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "name": self.name,
            "repo": self.repo,
            "path": self.path.display().to_string(),
            "bytes": self.bytes,
            "files": self.files,
            "modified": self.modified,
            "last_used": self.last_used,
            "ggufs": self.ggufs,
            "usable": self.usable,
            "specs": self.specs(),
        })
    }
}

/// The quant suffix of a GGUF file name, as `org/name:QUANT` matches it (`Llama-3.2-1B-Q4_K_M.gguf`
/// gives `Q4_K_M`; the first part of a split file gives the quant of the whole).
pub fn gguf_quant(file: &str) -> Option<String> {
    let base = file.rsplit('/').next().unwrap_or(file);
    let stem = base.strip_suffix(".gguf").or_else(|| base.strip_suffix(".GGUF"))?;
    let stem = match crate::gguf::split_name(base) {
        Some((pre, 1, _, _)) => pre.to_string(),
        Some(_) => return None,
        None => stem.to_string(),
    };
    let (_, q) = stem.rsplit_once(['-', '.'])?;
    (!q.is_empty()).then(|| q.to_string())
}

/// The cache directory: `cache_dir`, else `$OXINFER_CACHE`, else `~/.cache/oxinfer/hub`.
pub fn root(cache_dir: Option<&Path>) -> PathBuf {
    cache_dir.map(Path::to_path_buf).unwrap_or_else(crate::advisor::default_cache_dir)
}

fn has_config(dir: &Path) -> bool {
    ["config.json", "model_index.json", "modules.json", "meta.yaml"]
        .iter()
        .any(|f| dir.join(f).is_file())
}

fn secs(t: std::io::Result<std::time::SystemTime>) -> Option<u64> {
    t.ok()?.duration_since(UNIX_EPOCH).ok().map(|d| d.as_secs())
}

/// Walks `dir` without following links: (bytes, files, newest modification, GGUF files).
fn walk(dir: &Path, rel: &str, depth: usize, acc: &mut (u64, usize, Option<u64>, Vec<String>)) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let Ok(md) = e.path().symlink_metadata() else { continue };
        let name = e.file_name().to_string_lossy().into_owned();
        let path = if rel.is_empty() { name.clone() } else { format!("{rel}/{name}") };
        if md.is_dir() {
            if depth < 8 {
                walk(&e.path(), &path, depth + 1, acc);
            }
        } else if md.is_file() {
            if name == LAST_USED_FILE {
                continue;
            }
            acc.0 += md.len();
            acc.1 += 1;
            acc.2 = acc.2.max(secs(md.modified()));
            let lower = name.to_ascii_lowercase();
            let later_part = matches!(crate::gguf::split_name(&name), Some((_, i, _, _)) if i > 1);
            if lower.ends_with(".gguf") && !lower.starts_with("mmproj") && !later_part {
                acc.3.push(path);
            }
        }
    }
}

/// The entry for the folder at `path` (a direct child of the cache).
pub fn entry_at(path: &Path) -> Result<Entry> {
    let md = path.symlink_metadata().with_context(|| format!("{}", path.display()))?;
    let name = path.file_name().context("no folder name")?.to_string_lossy().into_owned();
    let mut acc = (0u64, 0usize, None, Vec::new());
    let target = if md.file_type().is_symlink() {
        std::fs::canonicalize(path).ok()
    } else {
        Some(path.to_path_buf())
    };
    if let Some(t) = &target {
        if t.is_dir() {
            walk(t, "", 0, &mut acc);
        } else if let Ok(m) = t.metadata() {
            acc = (m.len(), 1, secs(m.modified()), Vec::new());
        }
    }
    acc.3.sort();
    let last_used = std::fs::read_to_string(path.join(LAST_USED_FILE))
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .or_else(|| secs(std::fs::metadata(path.join(LAST_USED_FILE)).and_then(|m| m.modified())));
    let repo = name
        .split_once("--")
        .filter(|(o, n)| !o.is_empty() && !n.is_empty() && !n.contains("--"))
        .map(|(o, n)| format!("{o}/{n}"));
    Ok(Entry {
        usable: has_config(path) || !acc.3.is_empty(),
        repo,
        path: path.to_path_buf(),
        bytes: acc.0,
        files: acc.1,
        modified: acc.2,
        last_used,
        ggufs: acc.3,
        name,
    })
}

/// Every folder of the cache, by name. A cache directory that does not exist is empty.
pub fn list(cache_dir: Option<&Path>) -> Result<Vec<Entry>> {
    let root = root(cache_dir);
    let rd = match std::fs::read_dir(&root) {
        Ok(rd) => rd,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e).with_context(|| format!("read {}", root.display())),
    };
    let mut out: Vec<Entry> = rd
        .flatten()
        .filter(|e| !e.file_name().to_string_lossy().starts_with('.'))
        .filter(|e| e.path().symlink_metadata().is_ok_and(|m| m.is_dir() || m.file_type().is_symlink()))
        .filter_map(|e| entry_at(&e.path()).ok())
        .collect();
    out.sort_by_key(|a| a.name.to_lowercase());
    Ok(out)
}

/// The folder name of a cache entry given as a folder name (`org--name`) or a repo id
/// (`org/name`, `org/name:QUANT`): refused unless it is one plain path component.
pub fn folder_name(name: &str) -> Result<String> {
    let name = name.trim();
    let name = match name.split_once(':') {
        // org/name:QUANT names the repo's folder.
        Some((repo, _)) if repo.split('/').count() == 2 => repo,
        _ => name,
    };
    let folder = match name.split('/').collect::<Vec<_>>()[..] {
        [one] => one.to_string(),
        [org, repo] if !org.is_empty() && !repo.is_empty() => format!("{org}--{repo}"),
        _ => bail!("{name:?}: not a cache folder name or a repo id (org/name)"),
    };
    if folder.is_empty()
        || folder.starts_with('.')
        || folder.contains(['/', '\\', '\0', ':'])
        || folder == ".."
        || Path::new(&folder).components().count() != 1
    {
        bail!("{name:?}: not a cache folder name");
    }
    Ok(folder)
}

/// The path of the cache entry `name` ([`folder_name`]), checked to be a direct child of the cache
/// directory once resolved. Errors when there is no such entry.
pub fn locate(name: &str, cache_dir: Option<&Path>) -> Result<PathBuf> {
    let folder = folder_name(name)?;
    let root = root(cache_dir);
    let path = root.join(&folder);
    let md = path
        .symlink_metadata()
        .with_context(|| format!("{name}: not in the cache ({})", root.display()))?;
    if !md.file_type().is_symlink() {
        // Resolved, the entry must sit directly in the resolved cache directory.
        let real_root = std::fs::canonicalize(&root).with_context(|| format!("{}", root.display()))?;
        let real = std::fs::canonicalize(&path).with_context(|| format!("{}", path.display()))?;
        if real.parent() != Some(real_root.as_path()) || real == real_root {
            bail!("{name}: resolves outside the cache directory");
        }
    }
    Ok(path)
}

/// The top-level cache folder holding the files of model `source` (a repo id, `org/name:QUANT`,
/// or a path inside the cache), if it is in the cache.
pub fn folder_of(source: &str, cache_dir: Option<&Path>) -> Option<PathBuf> {
    let root = root(cache_dir);
    let p = Path::new(source);
    if p.is_absolute() || p.exists() {
        let real_root = std::fs::canonicalize(&root).ok()?;
        let real = std::fs::canonicalize(p).ok()?;
        let rel = real.strip_prefix(&real_root).ok()?;
        let first = rel.components().next()?;
        return Some(root.join(first.as_os_str()));
    }
    let folder = folder_name(source).ok()?;
    let path = root.join(folder);
    path.exists().then_some(path)
}

/// Records now as the last use of the cache folder holding `source` (a small file in it), when
/// it is in the cache. Errors are ignored: this is bookkeeping.
pub fn touch(source: &str, cache_dir: Option<&Path>) {
    if let Some(dir) = folder_of(source, cache_dir)
        && dir.is_dir()
    {
        let now = std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let _ = std::fs::write(dir.join(LAST_USED_FILE), format!("{now}\n"));
    }
}

/// Deletes the cache entry `name` (a folder name `org--name` or a repo id `org/name`; see
/// [`locate`] for what is refused) with everything in it; returns the bytes freed. A symbolic link
/// is removed as a link. The caller checks that the model is not in use.
pub fn remove(name: &str, cache_dir: Option<&Path>) -> Result<u64> {
    let path = locate(name, cache_dir)?;
    let md = path.symlink_metadata()?;
    if md.file_type().is_symlink() {
        std::fs::remove_file(&path).with_context(|| format!("remove {}", path.display()))?;
        return Ok(0);
    }
    let bytes = entry_at(&path).map(|e| e.bytes).unwrap_or(0);
    if md.is_dir() {
        // std's remove_dir_all removes links inside without following them.
        std::fs::remove_dir_all(&path).with_context(|| format!("remove {}", path.display()))?;
    } else {
        std::fs::remove_file(&path).with_context(|| format!("remove {}", path.display()))?;
    }
    Ok(bytes)
}

/// `(free, total)` bytes of the file system holding the cache directory, when the platform says.
pub fn disk_space(cache_dir: Option<&Path>) -> Option<(u64, u64)> {
    crate::platform::disk_space(&root(cache_dir))
}

/// The cache as JSON: `{"path", "free", "total", "entries": [...]}` (the C API's and the web
/// GUI's format).
pub fn to_json(cache_dir: Option<&Path>) -> Result<serde_json::Value> {
    let entries = list(cache_dir)?;
    let disk = disk_space(cache_dir);
    Ok(serde_json::json!({
        "path": root(cache_dir).display().to_string(),
        "free": disk.map(|d| d.0),
        "total": disk.map(|d| d.1),
        "bytes": entries.iter().map(|e| e.bytes).sum::<u64>(),
        "entries": entries.iter().map(Entry::to_json).collect::<Vec<_>>(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("oxinfer-cache-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn names() {
        assert_eq!(folder_name("Qwen/Qwen3-0.6B").unwrap(), "Qwen--Qwen3-0.6B");
        assert_eq!(folder_name("org/repo:Q4_K_M").unwrap(), "org--repo");
        assert_eq!(folder_name("Qwen--Qwen3-0.6B").unwrap(), "Qwen--Qwen3-0.6B");
        for bad in [
            "", "..", ".", "../x", "a/b/c", "/etc", "a\\b", ".hidden", "../..", "a\0b", "/", "c:x",
        ] {
            assert!(folder_name(bad).is_err(), "{bad:?} accepted");
        }
        assert_eq!(gguf_quant("Llama-3.2-1B-Instruct-Q4_K_M.gguf").as_deref(), Some("Q4_K_M"));
        assert_eq!(gguf_quant("model.Q8_0.gguf").as_deref(), Some("Q8_0"));
        assert_eq!(gguf_quant("sub/x-UD-Q2_K_XL-00001-of-00003.gguf").as_deref(), Some("Q2_K_XL"));
        assert_eq!(gguf_quant("x-UD-Q2_K_XL-00002-of-00003.gguf"), None);
    }

    #[test]
    fn list_and_remove() {
        let root = tmp("lr");
        let outside = tmp("lr-outside");
        std::fs::write(outside.join("keep.bin"), b"precious").unwrap();
        let m = root.join("org--model");
        std::fs::create_dir_all(m.join("sub")).unwrap();
        std::fs::write(m.join("config.json"), b"{}").unwrap();
        std::fs::write(m.join("sub/w.safetensors"), vec![0u8; 1000]).unwrap();
        let g = root.join("org--gguf");
        std::fs::create_dir_all(&g).unwrap();
        std::fs::write(g.join("m-Q4_K_M.gguf"), b"GGUF").unwrap();
        std::fs::write(g.join("mmproj-f16.gguf"), b"GGUF").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(&outside, root.join("link")).unwrap();
        let l = list(Some(&root)).unwrap();
        let e = l.iter().find(|e| e.name == "org--model").unwrap();
        assert_eq!((e.bytes, e.files, e.usable, e.repo.as_deref()), (1002, 2, true, Some("org/model")));
        assert_eq!(e.specs(), vec!["org/model".to_string()]);
        let e = l.iter().find(|e| e.name == "org--gguf").unwrap();
        assert_eq!(e.ggufs, vec!["m-Q4_K_M.gguf".to_string()]);
        assert_eq!(e.specs(), vec!["org/gguf:Q4_K_M".to_string()]);
        touch("org/model", Some(&root));
        assert!(entry_at(&m).unwrap().last_used.is_some());
        assert_eq!(folder_of("org/model", Some(&root)), Some(root.join("org--model")));
        assert_eq!(
            folder_of(&m.join("sub").to_string_lossy(), Some(&root)),
            Some(root.join("org--model"))
        );
        // Traversal and missing entries are refused, and nothing outside is touched.
        for bad in ["../lr-outside", "..", "org--model/sub", "/", "missing"] {
            assert!(remove(bad, Some(&root)).is_err(), "{bad:?} removed");
        }
        #[cfg(unix)]
        {
            // A link is removed as a link.
            assert_eq!(remove("link", Some(&root)).unwrap(), 0);
            assert!(outside.join("keep.bin").exists());
        }
        assert_eq!(remove("org/model", Some(&root)).unwrap(), 1002);
        assert!(!m.exists());
        assert!(root.exists() && g.exists());
        std::fs::remove_dir_all(&root).ok();
        std::fs::remove_dir_all(&outside).ok();
    }
}
