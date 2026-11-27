//! Learning from outcomes, the server's half (docs/learning.md): a chat request for
//! `MODEL@PROFILE` runs model `MODEL` with the profile's LoRA adapter when it has one and is
//! logged as an episode (the tokens the model read and the tokens it wrote); a later request
//! reports how it went (`POST /v1/episodes/{id}/outcome`, `POST /v1/outcomes`). An offline trainer
//! turns a profile's episodes and outcomes into its adapter. Requests for plain `MODEL` are
//! untouched: no adapter, no log.
//!
//! A profile is the directory `<learn dir>/<model>/<profile>` ([`Learn::profile_dir`]):
//! `episodes.jsonl` and `outcomes.jsonl`, appended to here, and `adapter/`, a PEFT folder the
//! trainer writes, attached to the loaded model ([`crate::engine::EnginePool::attach_adapter`])
//! and attached again when its files change.

use super::registry::ChatLease;
use super::{Api, ApiError, ApiRequest, ApiResponse, ApiResult, Output, now};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// What separates a model's id from the profile in a request's `model`.
pub(crate) const SEPARATOR: char = '@';

/// The longest profile name.
const MAX_PROFILE: usize = 64;

/// Whether `name` can be a profile: 1 to 64 of `A-Z a-z 0-9 _ . -`, not starting with a dot (it
/// is a directory's name).
pub(crate) fn valid_profile(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= MAX_PROFILE
        && !name.starts_with('.')
        && name.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-'))
}

/// The directory name of model `id`: its bytes other than `A-Z a-z 0-9 _ . -` as `%XX` (two
/// upper-case hex digits; `Qwen/Qwen3-0.6B` gives `Qwen%2FQwen3-0.6B`), a leading dot too. Two
/// ids never share a name.
pub(crate) fn model_dir_name(id: &str) -> String {
    let mut out = String::with_capacity(id.len());
    for (i, b) in id.bytes().enumerate() {
        if (b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-')) || (b == b'.' && i > 0) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// [`model_dir_name`] back to the id (None: not such a name).
fn model_of_dir(name: &str) -> Option<String> {
    let b = name.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' {
            let h = std::str::from_utf8(b.get(i + 1..i + 3)?).ok()?;
            out.push(u8::from_str_radix(h, 16).ok()?);
            i += 3;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

/// The longest conversation name kept ([`session_name`]).
const MAX_SESSION: usize = 128;

/// The name a request gives its conversation (the `X-Session-Id` header, as opencode sends it):
/// 1 to 128 printable ASCII characters without spaces, else none.
pub(crate) fn session_name(header: Option<&str>) -> Option<String> {
    let s = header?.trim();
    ((1..=MAX_SESSION).contains(&s.len()) && s.bytes().all(|b| b.is_ascii_graphic())).then(|| s.to_string())
}

/// The adapter a logged reply was sampled with.
#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct EpisodeAdapter {
    pub path: String,
    pub strength: f32,
    /// What told the adapter's files apart when it was attached ([`adapter_version`]).
    pub version: String,
}

/// One line of `episodes.jsonl`.
#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct Episode<'a> {
    pub id: &'a str,
    pub time: u64,
    pub model: &'a str,
    pub profile: &'a str,
    /// The conversation the request named ([`session_name`]); absent when it named none.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session: Option<&'a str>,
    pub adapters: Vec<EpisodeAdapter>,
    pub prompt_tokens: &'a [u32],
    pub completion_tokens: &'a [u32],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Vec<f32>>,
    pub finish_reason: &'a str,
}

/// What is read back of a line of `episodes.jsonl`.
#[derive(serde::Deserialize)]
struct EpisodeLine {
    id: String,
    #[serde(default)]
    time: u64,
    #[serde(default)]
    session: Option<String>,
    #[serde(default)]
    adapters: Vec<serde::de::IgnoredAny>,
    #[serde(default)]
    prompt_tokens: Vec<serde::de::IgnoredAny>,
    #[serde(default)]
    completion_tokens: Vec<serde::de::IgnoredAny>,
    #[serde(default)]
    finish_reason: String,
}

/// One line of `outcomes.jsonl`.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct Outcome {
    pub id: String,
    pub time: u64,
    pub reward: f64,
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub note: Option<String>,
}

/// An episode in the index.
#[derive(Clone, Debug)]
struct Meta {
    id: String,
    time: u64,
    model: String,
    profile: String,
    /// The conversation it belongs to, when the request named one.
    session: Option<String>,
    prompt: usize,
    completion: usize,
    finish: String,
    /// Sampled with an adapter.
    adapted: bool,
    /// The outcomes reported, and the last of them.
    outcomes: usize,
    last: Option<Outcome>,
}

impl Meta {
    fn json(&self) -> Value {
        json!({
            "id": self.id,
            "object": "episode",
            "time": self.time,
            "model": self.model,
            "profile": self.profile,
            "session": self.session,
            "adapter": self.adapted,
            "prompt_tokens": self.prompt,
            "completion_tokens": self.completion,
            "finish_reason": self.finish,
            "outcomes": self.outcomes,
            "outcome": self.last.as_ref().map(|o| json!({ "time": o.time, "reward": o.reward, "kind": o.kind, "note": o.note })),
        })
    }
}

#[derive(Default)]
struct State {
    /// The logs on disk were read ([`Learn::index`]).
    loaded: bool,
    /// The episodes, oldest first.
    episodes: Vec<Meta>,
    by_id: HashMap<String, usize>,
    /// Adapters that did not load, by profile directory: the version that failed (not tried
    /// again until the files change).
    failed: HashMap<PathBuf, String>,
}

impl State {
    fn push(&mut self, m: Meta) {
        match self.by_id.get(&m.id) {
            // (An id logged twice: the later line stands.)
            Some(&i) => self.episodes[i] = m,
            None => {
                self.by_id.insert(m.id.clone(), self.episodes.len());
                self.episodes.push(m);
            }
        }
    }

    fn outcome(&mut self, o: Outcome) {
        if let Some(&i) = self.by_id.get(&o.id) {
            let m = &mut self.episodes[i];
            m.outcomes += 1;
            m.last = Some(o);
        }
    }
}

/// The profiles' logs and adapters ([`super::ServerOptions::learn_dir`]).
pub(crate) struct Learn {
    dir: Option<PathBuf>,
    /// Held while a log is appended to.
    state: Mutex<State>,
}

/// A profile of a model, as listed.
pub(crate) struct ProfileInfo {
    pub name: String,
    pub episodes: usize,
    pub outcomes: usize,
    /// The adapter folder's version when it holds one ([`adapter_version`]).
    pub adapter: Option<String>,
}

/// What tells the files of the PEFT folder `dir` apart from another training's: the size and
/// modification time (nanoseconds) of `adapter_model.safetensors` and `adapter_config.json`.
/// None when it is not an adapter folder.
pub(crate) fn adapter_version(dir: &Path) -> Option<String> {
    let mut v = String::new();
    for f in ["adapter_model.safetensors", "adapter_config.json"] {
        let md = std::fs::metadata(dir.join(f)).ok().filter(|m| m.is_file())?;
        let t = md
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_nanos());
        if !v.is_empty() {
            v.push('-');
        }
        v.push_str(&format!("{:x}.{:x}", md.len(), t));
    }
    Some(v)
}

fn append(path: &Path, line: &str) -> std::io::Result<()> {
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    // One write per line: lines of concurrent writers do not mix.
    f.write_all(format!("{line}\n").as_bytes())
}

fn lines(path: &Path, mut f: impl FnMut(&str)) {
    use std::io::BufRead;
    let Ok(file) = std::fs::File::open(path) else { return };
    for line in std::io::BufReader::new(file).lines() {
        match line {
            Ok(l) if !l.trim().is_empty() => f(&l),
            Ok(_) => {}
            Err(_) => break,
        }
    }
}

fn subdirs(dir: &Path) -> Vec<(String, PathBuf)> {
    let mut v: Vec<(String, PathBuf)> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().is_dir())
        .filter_map(|e| Some((e.file_name().to_str()?.to_string(), e.path())))
        .collect();
    v.sort();
    v
}

impl Learn {
    pub fn new(dir: Option<PathBuf>) -> Self {
        Self {
            dir,
            state: Mutex::new(State::default()),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// `<learn dir>/<model>/<profile>` (None when learning is off).
    pub fn profile_dir(&self, model: &str, profile: &str) -> Option<PathBuf> {
        Some(self.dir.as_ref()?.join(model_dir_name(model)).join(profile))
    }

    /// The state with the logs on disk read into its index (once: at the first use after the
    /// start).
    fn index(&self) -> std::sync::MutexGuard<'_, State> {
        let mut st = self.lock();
        if st.loaded {
            return st;
        }
        st.loaded = true;
        let Some(dir) = &self.dir else { return st };
        for (mname, mdir) in subdirs(dir) {
            let Some(model) = model_of_dir(&mname) else { continue };
            for (profile, pdir) in subdirs(&mdir) {
                if !valid_profile(&profile) {
                    continue;
                }
                lines(&pdir.join("episodes.jsonl"), |l| {
                    // (A line cut short by a crash is skipped.)
                    if let Ok(e) = serde_json::from_str::<EpisodeLine>(l) {
                        st.push(Meta {
                            id: e.id,
                            time: e.time,
                            model: model.clone(),
                            profile: profile.clone(),
                            session: e.session,
                            prompt: e.prompt_tokens.len(),
                            completion: e.completion_tokens.len(),
                            finish: e.finish_reason,
                            adapted: !e.adapters.is_empty(),
                            outcomes: 0,
                            last: None,
                        });
                    }
                });
            }
        }
        for (_, mdir) in subdirs(dir) {
            for (_, pdir) in subdirs(&mdir) {
                lines(&pdir.join("outcomes.jsonl"), |l| {
                    if let Ok(o) = serde_json::from_str::<Outcome>(l) {
                        st.outcome(o);
                    }
                });
            }
        }
        // By time, as they are listed (the files of several profiles were read one after the
        // other).
        st.episodes.sort_by(|a, b| (a.time, &a.id).cmp(&(b.time, &b.id)));
        st.by_id = st.episodes.iter().enumerate().map(|(i, m)| (m.id.clone(), i)).collect();
        st
    }

    /// Appends an episode to its profile's `episodes.jsonl`.
    pub fn log(&self, e: &Episode) -> anyhow::Result<()> {
        use anyhow::Context;
        let dir = self.profile_dir(e.model, e.profile).context("learning is off")?;
        let mut st = self.index();
        crate::snapshot::private_dir(&dir)?;
        let path = dir.join("episodes.jsonl");
        append(&path, &serde_json::to_string(e)?).with_context(|| path.display().to_string())?;
        st.push(Meta {
            id: e.id.to_string(),
            time: e.time,
            model: e.model.to_string(),
            profile: e.profile.to_string(),
            session: e.session.map(str::to_string),
            prompt: e.prompt_tokens.len(),
            completion: e.completion_tokens.len(),
            finish: e.finish_reason.to_string(),
            adapted: !e.adapters.is_empty(),
            outcomes: 0,
            last: None,
        });
        Ok(())
    }

    /// The episodes of conversation `session` logged after its last one that has an outcome,
    /// oldest first: what a verdict on "the replies since the previous verdict" is about.
    fn unjudged(&self, session: &str) -> Vec<String> {
        let st = self.index();
        let mut ids: Vec<String> = st
            .episodes
            .iter()
            .rev()
            .filter(|m| m.session.as_deref() == Some(session))
            .take_while(|m| m.outcomes == 0)
            .map(|m| m.id.clone())
            .collect();
        ids.reverse();
        ids
    }

    /// Appends the outcomes to their episodes' profiles' `outcomes.jsonl`, all or none: a 404
    /// naming the ids that are not episodes.
    fn report(&self, outcomes: Vec<Outcome>) -> ApiResult<()> {
        let mut st = self.index();
        let unknown: Vec<&str> = outcomes
            .iter()
            .map(|o| o.id.as_str())
            .filter(|id| !st.by_id.contains_key(*id))
            .collect();
        if !unknown.is_empty() {
            return Err(ApiError::new(404, format!("no episode {}", unknown.join(", "))).code("episode_not_found"));
        }
        for o in outcomes {
            let m = &st.episodes[st.by_id[&o.id]];
            let dir = self
                .profile_dir(&m.model, &m.profile)
                .ok_or_else(|| ApiError::new(404, "learning is off"))?;
            let path = dir.join("outcomes.jsonl");
            let line = serde_json::to_string(&o).map_err(|e| ApiError::internal(e.into()))?;
            append(&path, &line).map_err(|e| ApiError::new(500, format!("{}: {e}", path.display())))?;
            st.outcome(o);
        }
        Ok(())
    }

    /// The profiles of `model` that exist (their directories), by name.
    pub fn profiles(&self, model: &str) -> Vec<ProfileInfo> {
        let Some(dir) = &self.dir else { return Vec::new() };
        let st = self.index();
        subdirs(&dir.join(model_dir_name(model)))
            .into_iter()
            .filter(|(name, _)| valid_profile(name))
            .map(|(name, pdir)| {
                let of = st.episodes.iter().filter(|m| m.model == model && m.profile == name);
                ProfileInfo {
                    episodes: of.clone().count(),
                    outcomes: of.map(|m| m.outcomes).sum(),
                    adapter: adapter_version(&pdir.join("adapter")),
                    name,
                }
            })
            .collect()
    }

    /// The names of [`Learn::profiles`] (without reading the logs).
    pub fn profile_names(&self, model: &str) -> Vec<String> {
        let Some(dir) = &self.dir else { return Vec::new() };
        subdirs(&dir.join(model_dir_name(model)))
            .into_iter()
            .map(|(n, _)| n)
            .filter(|n| valid_profile(n))
            .collect()
    }

    /// Brings the loaded model's adapter of `profile` in line with the profile's `adapter/`
    /// folder before a request: attached when the folder holds one that is not attached (or
    /// whose files changed since: the trainer wrote a new one), detached when the folder is gone.
    /// Returns the adapter's name in the pool (the profile's) when the request runs with it.
    ///
    /// An adapter that does not load is reported on stderr once per version, and the request
    /// runs on the base model.
    pub fn sync_adapter(&self, lease: &ChatLease<'_>, model: &str, profile: &str, log: bool) -> Option<String> {
        let dir = self.profile_dir(model, profile)?;
        let folder = dir.join("adapter");
        let attached = lease.pool.adapter(profile);
        let Some(version) = adapter_version(&folder) else {
            if attached.is_some() {
                lease.pool.detach_adapter(profile);
                if log {
                    eprintln!("[{model}{SEPARATOR}{profile}: adapter detached ({} is gone)]", folder.display());
                }
            }
            return None;
        };
        if attached.as_ref().is_some_and(|a| a.version == version) {
            return Some(profile.to_string());
        }
        if self.lock().failed.get(&dir) == Some(&version) {
            return None;
        }
        match lease.pool.attach_adapter(profile, &folder, 1.0, &version) {
            Ok(a) => {
                if log {
                    eprintln!("[{model}{SEPARATOR}{profile}: adapter attached: {}]", a.info.summary());
                }
                Some(profile.to_string())
            }
            Err(e) => {
                eprintln!("[{model}{SEPARATOR}{profile}: adapter not used (the base model answers): {e:#}]");
                self.lock().failed.insert(dir, version);
                None
            }
        }
    }
}

/// What a request's `model` names: the listed model and the profile after [`SEPARATOR`]. A
/// listed id is itself, whatever it holds.
pub(crate) fn resolve(api: &Api, name: &str) -> ApiResult<(String, Option<String>)> {
    let plain = || Ok((name.to_string(), None));
    if api.models.find(name).is_ok() {
        return plain();
    }
    let Some((model, profile)) = name.rsplit_once(SEPARATOR) else {
        return plain();
    };
    if model.is_empty() || api.models.find(model).is_err() {
        return plain();
    }
    let bad = |status: u16, msg: String| Err(ApiError::new(status, msg).param("model"));
    if !valid_profile(profile) {
        return bad(
            400,
            format!("profile {profile:?}: expected 1 to {MAX_PROFILE} of A-Z a-z 0-9 _ . - (not starting with a dot)"),
        );
    }
    if api.learn.dir.is_none() {
        return bad(404, format!("{name}: profiles are off on this server (learn_dir=none)"));
    }
    if !api.admin() {
        return bad(
            403,
            format!(
                "{name}: profiles are logged and their adapters loaded only on a loopback listener or with an API key (start the server with --api-key KEY)"
            ),
        );
    }
    Ok((model.to_string(), Some(profile.to_string())))
}

/// The model id of a request's `model` (without its profile), for the checks made before the
/// request is parsed.
pub(crate) fn base_model(name: &str) -> &str {
    match name.rsplit_once(SEPARATOR) {
        Some((m, p)) if !m.is_empty() && valid_profile(p) => m,
        _ => name,
    }
}

fn json_ok(v: Value) -> ApiResult<Output> {
    Ok(Output::Full(ApiResponse::json(200, &v)))
}

/// One reported outcome: `{"id": ...}`, `{"ids": [...]}` (`id`: the path's) or
/// `{"session": ...}` (the conversation's responses that `unjudged` gives: those since its last
/// outcome, possibly none), with `reward` (-1 to 1) or `result` (`good`: 1, `bad`: -1), and
/// optionally `kind` and `note`.
fn parse_outcome(v: &Value, id: Option<&str>, time: u64, unjudged: &dyn Fn(&str) -> Vec<String>, out: &mut Vec<Outcome>) -> ApiResult<()> {
    let o = v.as_object().ok_or_else(|| ApiError::bad("an outcome must be a JSON object"))?;
    for k in o.keys() {
        if !["id", "ids", "session", "reward", "result", "kind", "note"].contains(&k.as_str()) {
            return Err(ApiError::bad(format!(
                "unknown outcome field {k:?} (id, ids, session, reward, result, kind, note)"
            )));
        }
    }
    let reward = match (o.get("reward").filter(|v| !v.is_null()), o.get("result").filter(|v| !v.is_null())) {
        (Some(r), None) => r
            .as_f64()
            .filter(|r| (-1.0..=1.0).contains(r))
            .ok_or_else(|| ApiError::bad("reward must be a number from -1 to 1").param("reward"))?,
        (None, Some(r)) => match r.as_str() {
            Some("good") => 1.0,
            Some("bad") => -1.0,
            _ => return Err(ApiError::bad("result must be \"good\" or \"bad\"").param("result")),
        },
        (Some(_), Some(_)) => return Err(ApiError::bad("give reward or result, not both")),
        (None, None) => return Err(ApiError::bad("an outcome needs reward (-1 to 1) or result (\"good\", \"bad\")").param("reward")),
    };
    let text = |k: &'static str| -> ApiResult<Option<String>> {
        match o.get(k) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::String(s)) => Ok(Some(s.clone())),
            Some(_) => Err(ApiError::bad(format!("{k} must be a string")).param(k)),
        }
    };
    let (kind, note) = (text("kind")?, text("note")?);
    let mut ids: Vec<String> = Vec::new();
    if let Some(id) = id {
        ids.push(id.to_string());
    }
    if let Some(id) = text("id")? {
        ids.push(id);
    }
    match o.get("ids") {
        None | Some(Value::Null) => {}
        Some(Value::Array(a)) => {
            for i in a {
                ids.push(
                    i.as_str()
                        .ok_or_else(|| ApiError::bad("ids must be an array of response ids").param("ids"))?
                        .to_string(),
                );
            }
        }
        Some(_) => return Err(ApiError::bad("ids must be an array of response ids").param("ids")),
    }
    let session = text("session")?;
    if let Some(s) = &session {
        // (An outcome of this request already names them: they are judged once.)
        ids.extend(unjudged(s).into_iter().filter(|i| !out.iter().any(|o| o.id == *i)));
    }
    let mut seen = std::collections::HashSet::new();
    ids.retain(|i| seen.insert(i.clone()));
    if ids.is_empty() && session.is_none() {
        return Err(ApiError::bad("an outcome needs the id of a response (id, ids) or a session").param("id"));
    }
    out.extend(ids.into_iter().map(|id| Outcome {
        id,
        time,
        reward,
        kind: kind.clone(),
        note: note.clone(),
    }));
    Ok(())
}

fn body(req: &ApiRequest) -> ApiResult<Value> {
    serde_json::from_slice(&req.body).map_err(|e| ApiError::bad(format!("invalid JSON body: {e}")))
}

fn recorded(outcomes: &[Outcome]) -> Value {
    json!({
        "object": "list",
        "recorded": outcomes.len(),
        "data": outcomes.iter().map(|o| json!({ "id": o.id, "object": "outcome", "time": o.time, "reward": o.reward, "kind": o.kind, "note": o.note })).collect::<Vec<_>>(),
    })
}

/// `GET /v1/episodes`: the episodes, newest first (`order=asc`: oldest first), `limit` of them
/// (default 50, at most 1000) after episode `after`, of `model`, `profile` and `session` when
/// given.
fn list(api: &Api, req: &ApiRequest) -> ApiResult<Output> {
    let limit = match req.query_param("limit") {
        Some(l) => l
            .parse::<usize>()
            .ok()
            .filter(|n| (1..=1000).contains(n))
            .ok_or_else(|| ApiError::bad("limit must be 1 to 1000").param("limit"))?,
        None => 50,
    };
    let asc = match req.query_param("order").as_deref() {
        None | Some("desc") => false,
        Some("asc") => true,
        Some(_) => return Err(ApiError::bad("order must be asc or desc").param("order")),
    };
    let (model, profile, after) = (req.query_param("model"), req.query_param("profile"), req.query_param("after"));
    let session = req.query_param("session");
    let st = api.learn.index();
    let start = match &after {
        Some(a) => Some(
            *st.by_id
                .get(a)
                .ok_or_else(|| ApiError::new(404, format!("no episode {a}")).param("after"))?,
        ),
        None => None,
    };
    let wanted = |m: &&Meta| {
        model.as_ref().is_none_or(|x| *x == m.model)
            && profile.as_ref().is_none_or(|x| *x == m.profile)
            && session.as_ref().is_none_or(|x| Some(x.as_str()) == m.session.as_deref())
    };
    let mut it: Box<dyn Iterator<Item = &Meta>> = match (asc, start) {
        (true, s) => Box::new(st.episodes[s.map_or(0, |i| i + 1)..].iter()),
        (false, s) => Box::new(st.episodes[..s.unwrap_or(st.episodes.len())].iter().rev()),
    };
    let mut data: Vec<Value> = Vec::new();
    let mut more = false;
    for m in it.by_ref().filter(wanted) {
        if data.len() == limit {
            more = true;
            break;
        }
        data.push(m.json());
    }
    json_ok(json!({
        "object": "list",
        "first_id": data.first().map(|d| d["id"].clone()),
        "last_id": data.last().map(|d| d["id"].clone()),
        "has_more": more,
        "data": data,
    }))
}

fn profiles_json(api: &Api, model: &str) -> Value {
    let dir = api.learn.dir.as_ref().map(|d| d.join(model_dir_name(model)));
    let data: Vec<Value> = api
        .learn
        .profiles(model)
        .into_iter()
        .map(|p| {
            json!({
                "id": format!("{model}{SEPARATOR}{}", p.name),
                "object": "profile",
                "model": model,
                "profile": p.name,
                "path": dir.as_ref().map(|d| d.join(&p.name)),
                "episodes": p.episodes,
                "outcomes": p.outcomes,
                "adapter": p.adapter.map(|v| json!({ "version": v })),
            })
        })
        .collect();
    json!({ "object": "list", "data": data })
}

/// The learning endpoints (None: another endpoint's path).
pub(crate) fn route(api: &Api, req: &ApiRequest, path: &str) -> Option<ApiResult<Output>> {
    let get = req.method.eq_ignore_ascii_case("GET");
    let post = req.method.eq_ignore_ascii_case("POST");
    let delete = req.method.eq_ignore_ascii_case("DELETE");
    let wrong = || ApiError::new(405, format!("{} is not allowed on {path}", req.method));
    let off = || ApiError::new(404, "profiles are off on this server (learn_dir=none)");
    let unjudged = |s: &str| api.learn.unjudged(s);
    let run = |write: bool, f: &mut dyn FnMut() -> ApiResult<Output>| -> ApiResult<Output> {
        api.manage(req, write)?;
        if api.learn.dir.is_none() {
            return Err(off());
        }
        f()
    };
    if path == "/v1/episodes" {
        return Some(if get { run(false, &mut || list(api, req)) } else { Err(wrong()) });
    }
    if path == "/v1/outcomes" {
        return Some(if !post {
            Err(wrong())
        } else {
            run(true, &mut || {
                let v = body(req)?;
                let time = now();
                let mut out = Vec::new();
                // A batch, or one outcome as it is.
                match v.get("outcomes") {
                    Some(Value::Array(a)) => {
                        for o in a {
                            parse_outcome(o, None, time, &unjudged, &mut out)?;
                        }
                    }
                    Some(_) => return Err(ApiError::bad("outcomes must be an array").param("outcomes")),
                    None => parse_outcome(&v, None, time, &unjudged, &mut out)?,
                }
                api.learn.report(out.clone())?;
                json_ok(recorded(&out))
            })
        });
    }
    if let Some(rest) = path.strip_prefix("/v1/episodes/") {
        if let Some(id) = rest.strip_suffix("/outcome") {
            return Some(if !post {
                Err(wrong())
            } else {
                run(true, &mut || {
                    let mut out = Vec::new();
                    parse_outcome(&body(req)?, Some(id), now(), &unjudged, &mut out)?;
                    api.learn.report(out.clone())?;
                    json_ok(recorded(&out))
                })
            });
        }
        return Some(if !get {
            Err(wrong())
        } else {
            run(false, &mut || {
                let st = api.learn.index();
                match st.by_id.get(rest) {
                    Some(&i) => json_ok(st.episodes[i].json()),
                    None => Err(ApiError::new(404, format!("no episode {rest}")).code("episode_not_found")),
                }
            })
        });
    }
    let rest = path.strip_prefix("/v1/models/")?;
    if let Some(model) = rest.strip_suffix("/profiles") {
        return Some(if !get {
            Err(wrong())
        } else {
            run(false, &mut || {
                api.models.find(model)?;
                json_ok(profiles_json(api, model))
            })
        });
    }
    let (head, profile) = rest.strip_suffix("/adapter")?.rsplit_once('/')?;
    let model = head.strip_suffix("/profiles")?;
    Some(if !delete {
        Err(wrong())
    } else {
        run(true, &mut || {
            api.models.find(model)?;
            let missing = || ApiError::new(404, format!("{model} has no profile {profile:?}"));
            if !valid_profile(profile) {
                return Err(missing());
            }
            let dir = api.learn.profile_dir(model, profile).ok_or_else(off)?;
            if !dir.is_dir() {
                return Err(missing());
            }
            let folder = dir.join("adapter");
            let was = folder.exists();
            if was {
                std::fs::remove_dir_all(&folder).map_err(|e| ApiError::new(500, format!("{}: {e}", folder.display())))?;
            }
            api.learn.lock().failed.remove(&dir);
            json_ok(json!({ "id": format!("{model}{SEPARATOR}{profile}"), "object": "profile.adapter", "deleted": was }))
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        for ok in ["a", "run-1", "A_b.c", &"x".repeat(64)] {
            assert!(valid_profile(ok), "{ok}");
        }
        for bad in ["", ".", "..", ".a", "a/b", "a b", "a@b", "é", &"x".repeat(65)] {
            assert!(!valid_profile(bad), "{bad}");
        }
        for id in ["Qwen/Qwen3-0.6B", "org/repo:Q4_K_M", "..", ".hidden", "a%2Fb", "/abs/path", "é"] {
            let d = model_dir_name(id);
            assert!(
                d.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-' | b'%')),
                "{d}"
            );
            assert!(!d.starts_with('.'), "{d}");
            assert_eq!(model_of_dir(&d).as_deref(), Some(id));
        }
        assert_eq!(model_dir_name("Qwen/Qwen3-0.6B"), "Qwen%2FQwen3-0.6B");
        assert_eq!(base_model("a/b@p"), "a/b");
        assert_eq!(base_model("a/b@p/q"), "a/b@p/q");
        assert_eq!(base_model("a/b"), "a/b");
    }

    #[test]
    fn outcomes_parse() {
        let parse = |v: Value, id: Option<&str>| {
            let mut out = Vec::new();
            let unjudged = |s: &str| {
                if s == "s1" {
                    vec!["u1".to_string(), "u2".to_string()]
                } else {
                    Vec::new()
                }
            };
            parse_outcome(&v, id, 7, &unjudged, &mut out).map(|_| out).map_err(|e| e.message)
        };
        let o = parse(json!({"result": "bad", "kind": "tests", "note": "2 failed"}), Some("a")).unwrap();
        assert_eq!(
            o,
            vec![Outcome {
                id: "a".into(),
                time: 7,
                reward: -1.0,
                kind: Some("tests".into()),
                note: Some("2 failed".into())
            }]
        );
        let o = parse(json!({"ids": ["a", "b", "a"], "id": "c", "reward": 0.5}), None).unwrap();
        assert_eq!(o.iter().map(|o| o.id.as_str()).collect::<Vec<_>>(), ["c", "a", "b"]);
        assert!(o.iter().all(|o| o.reward == 0.5 && o.kind.is_none()));
        assert_eq!(parse(json!({"result": "good"}), Some("a")).unwrap()[0].reward, 1.0);
        for bad in [
            json!({"reward": 2}),
            json!({"reward": "1"}),
            json!({"result": "ok"}),
            json!({"reward": 1, "result": "good"}),
            json!({}),
            json!({"reward": 1, "ids": "a"}),
            json!({"reward": 1, "score": 3}),
            json!({"reward": 1, "kind": 3}),
        ] {
            assert!(parse(bad.clone(), Some("a")).is_err(), "{bad}");
        }
        assert!(parse(json!({"reward": 1}), None).is_err());
        // A conversation: its responses since the last outcome, with those named beside it.
        let ids = |v: Value| parse(v, None).unwrap().into_iter().map(|o| o.id).collect::<Vec<_>>();
        assert_eq!(ids(json!({"session": "s1", "result": "good"})), ["u1", "u2"]);
        assert_eq!(
            ids(json!({"session": "s1", "id": "u2", "ids": ["x"], "result": "good"})),
            ["u2", "x", "u1"]
        );
        assert!(ids(json!({"session": "other", "result": "bad"})).is_empty());
        assert!(parse(json!({"session": 3, "result": "bad"}), None).is_err());
    }

    /// The logs are appended to and read back by a server started later.
    #[test]
    fn logs_and_index() {
        let dir = std::env::temp_dir().join(format!("oxinfer-learn-unit-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let learn = Learn::new(Some(dir.clone()));
        let ep = |id: &'static str, time: u64, profile: &'static str| Episode {
            id,
            time,
            model: "org/m",
            profile,
            session: None,
            adapters: Vec::new(),
            prompt_tokens: &[1, 2, 3],
            completion_tokens: &[4, 5],
            logprobs: None,
            finish_reason: "stop",
        };
        learn.log(&ep("e1", 10, "p")).unwrap();
        learn.log(&ep("e2", 11, "q")).unwrap();
        let o = |id: &str, reward: f64| Outcome {
            id: id.into(),
            time: 12,
            reward,
            kind: Some("tests".into()),
            note: None,
        };
        learn.report(vec![o("e1", -1.0)]).unwrap();
        learn.report(vec![o("e1", 1.0), o("e2", 0.25)]).unwrap();
        // All or none.
        let e = learn.report(vec![o("e2", 1.0), o("nope", 1.0)]).unwrap_err();
        assert_eq!((e.status, e.message.as_str()), (404, "no episode nope"));
        let pdir = dir.join("org%2Fm").join("p");
        let text = std::fs::read_to_string(pdir.join("episodes.jsonl")).unwrap();
        assert_eq!(
            text,
            "{\"id\":\"e1\",\"time\":10,\"model\":\"org/m\",\"profile\":\"p\",\"adapters\":[],\"prompt_tokens\":[1,2,3],\"completion_tokens\":[4,5],\"finish_reason\":\"stop\"}\n"
        );
        let text = std::fs::read_to_string(pdir.join("outcomes.jsonl")).unwrap();
        assert_eq!(
            text,
            "{\"id\":\"e1\",\"time\":12,\"reward\":-1.0,\"kind\":\"tests\",\"note\":null}\n{\"id\":\"e1\",\"time\":12,\"reward\":1.0,\"kind\":\"tests\",\"note\":null}\n"
        );
        for learn in [learn, Learn::new(Some(dir.clone()))] {
            let st = learn.index();
            let e1 = &st.episodes[st.by_id["e1"]];
            assert_eq!((e1.profile.as_str(), e1.prompt, e1.completion, e1.outcomes), ("p", 3, 2, 2));
            assert_eq!(e1.last.as_ref().unwrap().reward, 1.0);
            assert_eq!(st.episodes[st.by_id["e2"]].last.as_ref().unwrap().reward, 0.25);
            drop(st);
            let p = learn.profiles("org/m");
            assert_eq!(
                p.iter().map(|p| (p.name.as_str(), p.episodes, p.outcomes)).collect::<Vec<_>>(),
                [("p", 1, 2), ("q", 1, 1)]
            );
            assert!(p[0].adapter.is_none());
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A conversation's episodes carry its name; a verdict on the conversation is about those
    /// since its last one with an outcome, also after a restart.
    #[test]
    fn sessions() {
        assert_eq!(session_name(Some(" ses_1aB ")).as_deref(), Some("ses_1aB"));
        for bad in [None, Some(""), Some("a b"), Some("é"), Some(&*"x".repeat(129))] {
            assert_eq!(session_name(bad), None, "{bad:?}");
        }
        let dir = std::env::temp_dir().join(format!("oxinfer-learn-sessions-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let learn = Learn::new(Some(dir.clone()));
        let ep = |id: &'static str, time: u64, session: Option<&'static str>| Episode {
            id,
            time,
            model: "m",
            profile: "p",
            session,
            adapters: Vec::new(),
            prompt_tokens: &[1],
            completion_tokens: &[2],
            logprobs: None,
            finish_reason: "stop",
        };
        for (id, t, s) in [("a1", 1, Some("A")), ("b1", 2, Some("B")), ("a2", 3, Some("A")), ("n1", 4, None)] {
            learn.log(&ep(id, t, s)).unwrap();
        }
        let text = std::fs::read_to_string(dir.join("m").join("p").join("episodes.jsonl")).unwrap();
        assert!(
            text.starts_with("{\"id\":\"a1\",\"time\":1,\"model\":\"m\",\"profile\":\"p\",\"session\":\"A\",\"adapters\":[],"),
            "{text}"
        );
        let last = text.lines().last().unwrap();
        assert!(
            last.starts_with("{\"id\":\"n1\",\"time\":4,\"model\":\"m\",\"profile\":\"p\",\"adapters\""),
            "{last}"
        );
        assert_eq!(learn.unjudged("A"), ["a1", "a2"]);
        assert_eq!(learn.unjudged("B"), ["b1"]);
        assert!(learn.unjudged("C").is_empty());
        let o = |id: &str| Outcome {
            id: id.into(),
            time: 5,
            reward: 1.0,
            kind: None,
            note: None,
        };
        learn.report(vec![o("a1")]).unwrap();
        assert_eq!(learn.unjudged("A"), ["a2"]);
        learn.log(&ep("a3", 6, Some("A"))).unwrap();
        // (Read again from the files.)
        let learn = Learn::new(Some(dir.clone()));
        assert_eq!(learn.unjudged("A"), ["a2", "a3"]);
        learn.report(vec![o("a2"), o("a3")]).unwrap();
        assert!(learn.unjudged("A").is_empty());
        assert_eq!(learn.unjudged("B"), ["b1"]);
        std::fs::remove_dir_all(&dir).ok();
    }
}
