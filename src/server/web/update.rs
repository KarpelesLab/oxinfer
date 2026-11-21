//! The web GUI's release check and one-click update (`GET` / `POST /api/web/update`).
//!
//! The check reads the latest release (`update::latest_from`: GitHub's release API, an anonymous
//! GET that carries nothing about this machine or its user; or [`super::super::ServerOptions::update_feed`]
//! / `$OXINFER_UPDATE_FEED`) at most once every [`CHECK_EVERY`] (every [`RETRY_AFTER`] after a
//! failure), in the background: the first GET answers `"state": "checking"` and starts it, the
//! next ones answer from memory. With the default source the answer is also kept in the cache
//! directory (`update-release.json`, like `--auto-update`'s stamp), so restarts do not ask again.
//! `update_check=0` (`--no-update-check`) makes no request at all.
//!
//! The update downloads the release's archive for this platform, replaces the running executable
//! (`update::install`) and restarts into it through the [`Restart`] given to
//! [`allow_self_update`] (`oxinfer web` gives one: the same arguments on the same address).
//! Without that call (an application embedding the library or the C API, whose executable is not
//! oxinfer's) nothing is replaced and the GUI shows the command `oxinfer --update`. Refused while
//! a download, a model load or a request runs.

use super::{Api, ApiError, ApiResult, Output, WebState, json_ok, lock};
use serde_json::{Value, json};
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime};

/// The least time between two release checks.
#[cfg_attr(not(feature = "update"), allow(dead_code))]
pub const CHECK_EVERY: Duration = Duration::from_secs(6 * 3600);
/// The least time before another check after a failed one.
#[cfg_attr(not(feature = "update"), allow(dead_code))]
pub const RETRY_AFTER: Duration = Duration::from_secs(30 * 60);

/// The environment variable that tells a restarted `oxinfer web` the address it listened on
/// (it then listens there again, and opens no browser).
pub const RESTART_ENV: &str = "OXINFER_WEB_RESTART";

/// Starts the new executable (its path) in place of this process after `POST /api/web/update`
/// installed it: on Unix by `exec`, so it returns only on failure; elsewhere by starting it and
/// exiting. An error is reported in the update's status ("restart oxinfer to use it").
pub type Restart = Box<dyn Fn(&Path) -> anyhow::Result<()> + Send + Sync>;

/// What the last check found.
#[derive(Default)]
#[cfg_attr(not(feature = "update"), allow(dead_code))]
struct Check {
    checking: bool,
    /// When the answer was obtained, and when the next check may start.
    checked: Option<SystemTime>,
    next: Option<Instant>,
    latest: Option<String>,
    url: Option<String>,
    newer: bool,
    /// Why the latest release cannot be installed here (no binary for this platform, a
    /// directory the executable cannot be replaced in).
    cannot: Option<String>,
    error: Option<String>,
}

/// The update state of a [`WebState`].
#[cfg_attr(not(feature = "update"), allow(dead_code))]
pub(super) struct Updates {
    enabled: bool,
    feed: Option<String>,
    check: Arc<Mutex<Check>>,
    /// The update started from the GUI: `state` (`checking`, `downloading`, `restarting`,
    /// `updated`, `up_to_date`, `error`), `version`, `message`, `error`, `started`.
    job: Mutex<Option<Arc<Mutex<Value>>>>,
    /// Set by [`allow_self_update`]: this process is the oxinfer program.
    restart: OnceLock<Option<Restart>>,
}

impl Updates {
    pub(super) fn new(opts: &super::super::ServerOptions) -> Self {
        Updates {
            enabled: opts.update_check,
            feed: opts.update_feed.clone(),
            check: Arc::new(Mutex::new(Check::default())),
            job: Mutex::new(None),
            restart: OnceLock::new(),
        }
    }
}

/// Marks the server's process as the `oxinfer` program, whose executable `POST /api/web/update`
/// may replace. `restart` then starts the new executable in its place; without one the update's
/// status says to restart oxinfer by hand. Never called (an application embedding the library
/// or the C API), the GUI is told to run `oxinfer --update` instead. False when the server has no
/// web GUI or this was called already.
pub fn allow_self_update(api: &Api, restart: Option<Restart>) -> bool {
    api.web.as_ref().is_some_and(|w| w.updates.restart.set(restart).is_ok())
}

fn unix(t: SystemTime) -> u64 {
    t.duration_since(SystemTime::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

const RUNNING: [&str; 3] = ["checking", "downloading", "restarting"];

/// What keeps the server from restarting now: a download, a model load, a request.
fn busy(api: &Api, w: &WebState) -> Option<String> {
    if lock(&w.jobs).iter().any(|j| j.running()) {
        return Some("a download is running".into());
    }
    let loading = lock(&w.loads).iter().find_map(|l| {
        matches!(
            lock(&l.st)["state"].as_str(),
            Some("queued" | "waiting" | "downloading" | "loading")
        )
        .then(|| l.id.clone())
    });
    if let Some(id) = loading {
        return Some(format!("{id} is loading"));
    }
    api.models
        .status()
        .into_iter()
        .find(|s| s.busy)
        .map(|s| format!("a request is using {}", s.spec.id))
}

/// `GET /api/web/update`.
pub(super) fn update_status(api: &Arc<Api>, w: &WebState) -> Value {
    let u = &w.updates;
    let job = lock(&u.job).as_ref().map(|j| lock(j).clone());
    let mut v = json!({
        "enabled": u.enabled,
        "state": "disabled",
        "current": env!("CARGO_PKG_VERSION"),
        "latest": null,
        "newer": null,
        "url": null,
        "checked": null,
        "checking": false,
        "error": null,
        "can_update": false,
        "why_not": null,
        "command": "oxinfer --update",
        "job": job,
    });
    if u.enabled {
        fill(api, w, &mut v);
    }
    v
}

#[cfg(not(feature = "update"))]
fn fill(_api: &Arc<Api>, _w: &WebState, v: &mut Value) {
    v["state"] = json!("unsupported");
    v["error"] = json!("this build has no release check (the `update` feature)");
}

#[cfg(feature = "update")]
fn fill(api: &Arc<Api>, w: &WebState, v: &mut Value) {
    let u = &w.updates;
    let mut c = lock(&u.check);
    if !c.checking && c.next.is_none_or(|n| Instant::now() >= n) {
        c.checking = true;
        let (check, feed) = (u.check.clone(), u.feed.clone());
        let stamp = (u.feed.is_none() && std::env::var_os(crate::update::FEED_ENV).is_none())
            .then(|| crate::model_cache::root(api.models.cache_dir()).join("update-release.json"));
        let spawned = std::thread::Builder::new()
            .name("oxinfer-update-check".into())
            .spawn(move || run_check(&check, feed.as_deref(), stamp.as_deref()));
        if spawned.is_err() {
            c.checking = false;
        }
    }
    let state = match (&c.latest, &c.error) {
        (None, _) if c.checking => "checking",
        (None, Some(_)) => "error",
        (None, None) => "unknown",
        (Some(_), _) if c.newer => "available",
        (Some(_), _) => "up_to_date",
    };
    let restart = u.restart.get();
    let why_not = if !c.newer {
        None
    } else if !w.admin {
        Some("the management API is read-only: a non-loopback address without an API key".to_string())
    } else if restart.is_none() {
        Some("this server runs inside another program, through the oxinfer library or C API".to_string())
    } else {
        c.cannot.clone()
    };
    v["state"] = json!(state);
    v["latest"] = json!(c.latest);
    v["newer"] = json!(c.latest.as_ref().map(|_| c.newer));
    v["url"] = json!(c.url);
    v["checked"] = json!(c.checked.map(unix));
    v["checking"] = json!(c.checking);
    v["error"] = json!(c.error);
    v["can_update"] = json!(c.newer && why_not.is_none());
    v["why_not"] = json!(why_not);
}

/// One release check, from the stamp file when it is recent enough.
#[cfg(feature = "update")]
fn run_check(check: &Mutex<Check>, feed: Option<&str>, stamp: Option<&Path>) {
    let cached = stamp.and_then(|s| {
        let age = std::fs::metadata(s).and_then(|m| m.modified()).ok()?;
        let since = age.elapsed().ok().filter(|d| *d < CHECK_EVERY)?;
        let rel: crate::update::Release = serde_json::from_slice(&std::fs::read(s).ok()?).ok()?;
        Some((rel, age, CHECK_EVERY - since))
    });
    let r = match cached {
        Some(c) => Ok(c),
        None => crate::update::latest_from(feed).map(|rel| {
            if let Some(s) = stamp
                && let Ok(bytes) = serde_json::to_vec(&rel)
            {
                let _ = s.parent().map(std::fs::create_dir_all);
                let _ = std::fs::write(s, bytes);
            }
            (rel, SystemTime::now(), CHECK_EVERY)
        }),
    };
    // Before taking the lock: probing the executable's directory touches the disk.
    let cannot = r
        .as_ref()
        .ok()
        .and_then(|(rel, ..)| rel.is_newer().then(|| crate::update::cannot_install(rel)).flatten());
    let mut c = lock(check);
    c.checking = false;
    match r {
        Ok((rel, when, wait)) => {
            c.latest = Some(rel.version().to_string());
            c.url = Some(rel.html_url.clone()).filter(|u| !u.is_empty());
            c.newer = rel.is_newer();
            c.cannot = cannot;
            c.error = None;
            c.checked = Some(when);
            c.next = Some(Instant::now() + wait);
        }
        Err(e) => {
            c.error = Some(format!("{e:#}"));
            c.checked = Some(SystemTime::now());
            c.next = Some(Instant::now() + RETRY_AFTER);
        }
    }
}

/// `POST /api/web/update` (the management API's admin rule already applied).
pub(super) fn start_update(api: &Arc<Api>, w: &WebState) -> ApiResult<Output> {
    let u = &w.updates;
    if !u.enabled {
        return Err(ApiError::new(403, "updates are turned off (update_check=0, --no-update-check)").code("update_check_off"));
    }
    if u.restart.get().is_none() || !cfg!(feature = "update") {
        let why = if cfg!(feature = "update") {
            "this server runs inside another program (the oxinfer library or C API), which cannot replace itself: run `oxinfer --update`"
        } else {
            "this build has no self-update (the `update` feature)"
        };
        return Err(ApiError::new(409, why).code("cannot_update"));
    }
    {
        let job = lock(&u.job);
        if let Some(j) = job.as_ref() {
            let v = lock(j).clone();
            // One at a time; once installed, a restart (by hand) is what is left to do.
            if v["state"].as_str().is_some_and(|s| RUNNING.contains(&s) || s == "updated") {
                return json_ok(v);
            }
        }
    }
    if let Some(why) = busy(api, w) {
        return Err(ApiError::new(409, format!("{why}: update once it is done")).code("busy"));
    }
    let job = Arc::new(Mutex::new(json!({"state": "checking", "started": unix(SystemTime::now())})));
    *lock(&u.job) = Some(job.clone());
    #[cfg(feature = "update")]
    {
        let api2 = api.clone();
        let j = job.clone();
        std::thread::Builder::new()
            .name("oxinfer-update".into())
            .spawn(move || run_update(&api2, &j))
            .map_err(|e| ApiError::internal(e.into()))?;
    }
    let v = lock(&job).clone();
    json_ok(v)
}

/// Sets the job's fields (the others stay).
#[cfg(feature = "update")]
fn set(job: &Mutex<Value>, fields: Value) {
    let mut v = lock(job);
    if let (Some(o), Some(f)) = (v.as_object_mut(), fields.as_object()) {
        for (k, x) in f {
            o.insert(k.clone(), x.clone());
        }
    }
}

#[cfg(feature = "update")]
fn run_update(api: &Arc<Api>, job: &Mutex<Value>) {
    use crate::update::{Outcome, cannot_install, install, latest_from};
    let Some(w) = api.web.as_ref() else { return };
    let log = api.opts.log;
    let fail = |e: String| {
        if log {
            eprintln!("[update: {e}]");
        }
        set(job, json!({"state": "error", "error": e}));
    };
    let rel = match latest_from(w.updates.feed.as_deref()) {
        Ok(r) => r,
        Err(e) => return fail(format!("{e:#}")),
    };
    let version = rel.version().to_string();
    if !rel.is_newer() {
        set(job, json!({"state": "up_to_date", "version": version}));
        return;
    }
    if let Some(why) = cannot_install(&rel) {
        return fail(why);
    }
    let asset = rel.asset().map(|a| (a.name.clone(), a.size));
    set(
        job,
        json!({"state": "downloading", "version": version, "file": asset.as_ref().map(|a| &a.0), "size": asset.as_ref().map(|a| a.1)}),
    );
    if log {
        eprintln!("[update: installing oxinfer {version} from {}]", rel.html_url);
    }
    let exe = match install(&rel, &mut |line| set(job, json!({"message": line}))) {
        Ok(Outcome::Updated(exe, _)) => exe,
        Ok(Outcome::UpToDate(_)) => return set(job, json!({"state": "up_to_date"})),
        Err(e) => return fail(format!("{e:#}")),
    };
    let manual = |why: Option<String>| {
        let mut msg = format!("oxinfer {version} is installed: restart oxinfer to use it");
        if let Some(w) = why {
            msg = format!("{msg} ({w})");
        }
        if log {
            eprintln!("[update: {msg}]");
        }
        set(job, json!({"state": "updated", "message": msg}));
    };
    let Some(Some(restart)) = w.updates.restart.get() else {
        return manual(None);
    };
    set(
        job,
        json!({"state": "restarting", "message": format!("restarting into oxinfer {version}")}),
    );
    if log {
        eprintln!("[update: {} is now oxinfer {version}; restarting]", exe.display());
    }
    // The page sees the state before the server goes; a request that started since waits for
    // its end (two minutes at most).
    std::thread::sleep(Duration::from_millis(1500));
    let t0 = Instant::now();
    while busy(api, w).is_some() && t0.elapsed() < Duration::from_secs(120) {
        std::thread::sleep(Duration::from_millis(250));
    }
    if let Err(e) = restart(&exe) {
        manual(Some(format!("the restart failed: {e:#}")));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::{ApiRequest, ServerOptions};

    #[test]
    fn refused_while_a_model_loads() {
        let dir = std::env::temp_dir().join(format!("oxinfer-web-update-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let feed = dir.join("release.json");
        std::fs::write(
            &feed,
            r#"{"tag_name": "v999.0.0", "html_url": "https://example.invalid/r", "assets": []}"#,
        )
        .unwrap();
        let o = ServerOptions::from_pairs(&format!(
            "web=1; log=0; cache_dir={}; update_feed={}",
            dir.display(),
            feed.display()
        ))
        .unwrap();
        let api = Api::new(o).unwrap();
        let post = ApiRequest::post_json("/api/web/update", "{}");
        // Not the oxinfer program: nothing to replace.
        let r = api.handle(&post).collect();
        assert_eq!(r.status, 409);
        assert!(String::from_utf8_lossy(&r.body).contains("cannot_update"));
        assert!(allow_self_update(&api, None));
        assert!(!allow_self_update(&api, None), "once");
        let w = api.web.as_ref().unwrap();
        w.loads.lock().unwrap().push(Arc::new(super::super::LoadJob {
            id: "m".into(),
            cancel: Default::default(),
            started: Instant::now(),
            st: Mutex::new(json!({"state": "loading"})),
        }));
        let r = api.handle(&post).collect();
        assert_eq!(r.status, 409);
        let v: Value = serde_json::from_slice(&r.body).unwrap();
        assert_eq!(v["error"]["code"], "busy", "{v}");
        assert!(v["error"]["message"].as_str().unwrap().contains("m is loading"), "{v}");
        assert!(lock(&w.updates.job).is_none(), "no job started");
        std::fs::remove_dir_all(&dir).ok();
    }
}
