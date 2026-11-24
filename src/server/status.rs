//! The status page's report ([`status_json`], `GET /api/web/activity`) and its actions on the
//! caches: the prompt caches' snapshots ([`clear_prompt_cache`]) and the streamed models' expert
//! caches ([`free_caches`]).

use super::registry::EntryStatus;
use super::{Api, ApiError, ApiResult, activity};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// What [`Api::clear_prompt_cache`] removed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct PromptCacheCleared {
    /// Snapshots removed, and their bytes.
    pub entries: usize,
    pub bytes: u64,
}

/// One snapshot file as the status page knows it.
#[derive(Clone, Debug)]
struct Snap {
    bytes: u64,
    modified: SystemTime,
    /// The `model` of its metadata (a fingerprint); empty when it cannot be read.
    fingerprint: String,
    model_type: Option<String>,
    tokens: u64,
}

/// The snapshots' metadata by file, read again when a file's size or time changes.
#[derive(Default)]
pub(crate) struct SnapIndex(HashMap<PathBuf, Snap>);

/// The prompt-cache directories: the server's and those of the models' own `prompt_cache`
/// options (and of the loaded engines).
fn cache_dirs(api: &Api, models: &[EntryStatus]) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = api.opts.prompt_cache.iter().cloned().collect();
    for m in models {
        let own = match m.spec.option("prompt_cache") {
            None | Some("" | "none" | "off" | "0") => None,
            Some("auto" | "on" | "1") => Some(crate::snapshot::default_dir()),
            Some(v) => Some(PathBuf::from(v)),
        };
        let loaded = m.details.as_ref().and_then(|d| d.prompt_cache.clone());
        for d in own.into_iter().chain(loaded) {
            if !dirs.contains(&d) {
                dirs.push(d);
            }
        }
    }
    dirs
}

/// The snapshots in `dir` (`*.oxsnap` regular files directly in it), through the index.
fn scan(index: &mut SnapIndex, dir: &Path) -> Vec<(PathBuf, Snap)> {
    let Ok(rd) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut out = Vec::new();
    for e in rd.flatten() {
        let p = e.path();
        if p.extension().and_then(|x| x.to_str()) != Some(crate::snapshot::EXT) {
            continue;
        }
        let Ok(md) = std::fs::symlink_metadata(&p) else { continue };
        if !md.file_type().is_file() {
            continue;
        }
        let modified = md.modified().unwrap_or(SystemTime::UNIX_EPOCH);
        let known = index.0.get(&p).filter(|s| s.bytes == md.len() && s.modified == modified).cloned();
        let snap = known.unwrap_or_else(|| {
            let meta = crate::snapshot::SnapReader::open(&p).map(|r| r.meta().clone()).ok();
            let m = meta.as_ref();
            let s = Snap {
                bytes: md.len(),
                modified,
                fingerprint: m.and_then(|m| m["model"].as_str()).unwrap_or_default().to_string(),
                model_type: m.and_then(|m| m["model_type"].as_str()).map(str::to_string),
                tokens: m.and_then(|m| m["tokens"].as_u64()).unwrap_or(0),
            };
            index.0.insert(p.clone(), s.clone());
            s
        });
        out.push((p, snap));
    }
    out
}

/// The prompt caches: per directory its snapshots and bytes, grouped by model (fingerprint),
/// with the ids of the models that store under it.
pub(crate) fn prompt_cache_json(api: &Api, models: &[EntryStatus]) -> Value {
    let dirs = cache_dirs(api, models);
    let mut index = api.snap_index.lock().unwrap_or_else(|p| p.into_inner());
    let mut seen = Vec::new();
    let (mut total_n, mut total_b) = (0usize, 0u64);
    let list: Vec<Value> = dirs
        .iter()
        .map(|dir| {
            let snaps = scan(&mut index, dir);
            seen.extend(snaps.iter().map(|(p, _)| p.clone()));
            // Per fingerprint: the snapshot count, and their sums (the last use the latest).
            let mut groups: Vec<(usize, Snap)> = Vec::new();
            for (_, s) in &snaps {
                match groups.iter_mut().find(|g| g.1.fingerprint == s.fingerprint) {
                    Some((n, g)) => {
                        *n += 1;
                        g.bytes += s.bytes;
                        g.tokens += s.tokens;
                        g.modified = g.modified.max(s.modified);
                    }
                    None => groups.push((1, s.clone())),
                }
            }
            groups.sort_by_key(|g| std::cmp::Reverse(g.1.bytes));
            let bytes: u64 = snaps.iter().map(|(_, s)| s.bytes).sum();
            total_n += snaps.len();
            total_b += bytes;
            let unix = |t: SystemTime| t.duration_since(SystemTime::UNIX_EPOCH).map(|d| d.as_secs()).ok();
            json!({
                "path": dir.display().to_string(),
                "exists": dir.is_dir(),
                "entries": snaps.len(),
                "bytes": bytes,
                "models": groups.iter().map(|(n, g)| json!({
                    "fingerprint": g.fingerprint,
                    "model_type": g.model_type,
                    "ids": models.iter().filter(|m| m.fingerprint.as_deref() == Some(g.fingerprint.as_str())).map(|m| m.spec.id.clone()).collect::<Vec<_>>(),
                    "entries": n,
                    "bytes": g.bytes,
                    "tokens": g.tokens,
                    "last_used": unix(g.modified),
                })).collect::<Vec<_>>(),
            })
        })
        .collect();
    index.0.retain(|p, _| seen.contains(p));
    json!({
        "dirs": list,
        "entries": total_n,
        "bytes": total_b,
        "cap": (api.opts.prompt_cache_gb.unwrap_or(crate::snapshot::DEFAULT_CAP_GB) * 1e9) as u64,
    })
}

/// Removes prompt-cache snapshots: of the model with id `model` or with fingerprint
/// `fingerprint` (else all of them), in directory `dir` (one of the prompt-cache directories;
/// else all of them). The loaded idle chat models whose cache is concerned forget them too.
pub(crate) fn clear_prompt_cache(
    api: &Api,
    model: Option<&str>,
    fingerprint: Option<&str>,
    dir: Option<&str>,
) -> ApiResult<PromptCacheCleared> {
    let models = api.models.status();
    let mut dirs = cache_dirs(api, &models);
    if let Some(d) = dir {
        dirs.retain(|x| x.to_str() == Some(d));
        if dirs.is_empty() {
            return Err(ApiError::bad(format!("{d:?} is not a prompt-cache directory of this server")).param("dir"));
        }
    }
    let fp: Option<String> = match (model, fingerprint) {
        (Some(id), _) => {
            let m = models.iter().find(|m| m.spec.id == id).ok_or_else(|| {
                ApiError::new(404, format!("the model {id:?} does not exist"))
                    .param("model")
                    .code("model_not_found")
            })?;
            Some(m.fingerprint.clone().ok_or_else(|| {
                ApiError::new(
                    409,
                    format!("{id} has not been loaded since the server started, so its snapshots cannot be told apart: clear them by fingerprint, or all of them"),
                )
                .param("model")
            })?)
        }
        (None, Some(f)) => Some(f.to_string()),
        (None, None) => None,
    };
    let mut out = PromptCacheCleared::default();
    {
        let mut index = api.snap_index.lock().unwrap_or_else(|p| p.into_inner());
        for d in &dirs {
            for (p, s) in scan(&mut index, d) {
                if fp.as_deref().is_none_or(|f| s.fingerprint == f) && std::fs::remove_file(&p).is_ok() {
                    out.entries += 1;
                    out.bytes += s.bytes;
                    index.0.remove(&p);
                }
            }
        }
    }
    // The engines' own index of the directory (and the conversation each has waiting).
    let r = api.models.each_idle_engine(|e| {
        let concerned = fp.as_deref().is_none_or(|f| e.prompt_cache_fingerprint() == f)
            && e.prompt_cache_info().is_some_and(|(d, _, _)| dirs.contains(&d));
        if concerned { e.clear_prompt_cache() } else { (0, 0) }
    });
    for (_, (n, b)) in r {
        out.entries += n;
        out.bytes += b;
    }
    if api.opts.log {
        eprintln!(
            "[prompt cache: removed {} snapshots ({:.2} GB){}]",
            out.entries,
            out.bytes as f64 / 1e9,
            model.map(|m| format!(" of {m}")).unwrap_or_default()
        );
    }
    Ok(out)
}

/// Lets go of the expert caches of the streamed models (model `id`, or all); returns the bytes.
pub(crate) fn free_caches(api: &Api, id: Option<&str>) -> ApiResult<u64> {
    let models = api.models.status();
    if let Some(id) = id
        && !models.iter().any(|m| m.spec.id == id)
    {
        return Err(ApiError::new(404, format!("the model {id:?} does not exist"))
            .param("model")
            .code("model_not_found"));
    }
    let mut freed = 0;
    for m in models.iter().filter(|m| id.is_none_or(|i| m.spec.id == i)) {
        if let Some(s) = m.details.as_ref().and_then(|d| d.streamer.as_ref()).and_then(|w| w.upgrade()) {
            freed += s.release_part_cache();
        }
    }
    if api.opts.log && freed > 0 {
        eprintln!("[freed {:.2} GB of cached experts]", freed as f64 / 1e9);
    }
    Ok(freed)
}

/// A streamed model's figures: the page-cache budget, the resident and streamed weights, the
/// part (expert) cache, and what was read so far.
fn stream_json(s: &crate::stream::Streamer) -> Value {
    let plan = s.plan();
    let st = s.stats();
    let (used, cap) = s.part_cache_usage();
    json!({
        "budget": s.options().cache_budget,
        "resident_bytes": plan.resident_bytes,
        "streamed_bytes": plan.streamed_bytes,
        "part_bytes": plan.part_bytes,
        "part_cache": {"used": used, "cap": cap},
        "bytes_read": st.bytes_read,
        "parts": st.parts,
        "parts_cached": st.parts_cached,
    })
}

/// The whole report (see [`Api::status`]).
pub(crate) fn status_json(api: &Api) -> Value {
    let a = &api.activity;
    let models = api.models.status();
    let requests = a.requests();
    let (served, cancelled) = a.counters();
    let model_list: Vec<Value> = models
        .iter()
        .map(|m| {
            let d = m.details.as_ref();
            let stream = d.and_then(|d| d.streamer.as_ref()).and_then(|w| w.upgrade());
            let running = requests.iter().filter(|r| r.model.as_deref() == Some(m.spec.id.as_str()));
            json!({
                "id": m.spec.id,
                "source": m.spec.source,
                "kind": m.kind.map(super::Kind::as_str),
                "loaded": m.loaded,
                "busy": m.busy,
                "state": match (m.loaded, m.busy) {
                    (true, true) => "busy",
                    (true, false) => "ready",
                    (false, true) => "loading",
                    (false, false) => "unloaded",
                },
                "memory": m.bytes,
                "vram": m.vram,
                "device": d.and_then(|d| d.device.clone()),
                "model_type": d.and_then(|d| d.model_type.clone()),
                "quant": d.and_then(|d| d.quant.clone()),
                "format": d.and_then(|d| d.format),
                "loaded_at": d.map(|d| d.loaded_at),
                "load_secs": d.map(|d| (d.load_secs * 10.0).round() / 10.0),
                "idle_secs": (!m.busy && m.loaded).then_some(m.idle_secs.round()),
                "requests": running.count(),
                "fingerprint": m.fingerprint,
                "stream": stream.as_deref().map(stream_json),
            })
        })
        .collect();
    let web = api.web.as_ref();
    json!({
        "version": env!("CARGO_PKG_VERSION"),
        "started": a.started_unix,
        "uptime": a.started.elapsed().as_secs(),
        "listen": api.opts.listen,
        "loopback": api.loopback,
        "admin": web.map(|w| !w.read_only()),
        "auth_required": api.opts.api_key.is_some(),
        "requests": requests.iter().map(activity::RequestInfo::to_json).collect::<Vec<_>>(),
        "videos": api.videos.running(),
        "recent": a.recent(),
        "served": served,
        "cancelled": cancelled,
        "models": model_list,
        "memory_budget": api.models.budget(),
        "models_memory": models.iter().map(|m| m.bytes).sum::<u64>(),
        "idle_unload": api.opts.idle_unload.map(|d| d.as_secs()),
        "process": activity::process_json(a),
        "machine": activity::machine_json(),
        "gpu": activity::gpu_json(),
        "prompt_cache": prompt_cache_json(api, &models),
    })
}
