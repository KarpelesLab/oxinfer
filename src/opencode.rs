//! [opencode](https://opencode.ai) integration: adds oxinfer's models to opencode's global
//! configuration as models of an `oxinfer` provider (`@ai-sdk/openai-compatible` pointed at the
//! server's `/v1`), and removes them again.
//!
//! The configuration directory is `$XDG_CONFIG_HOME/opencode`, else `~/.config/opencode` (on
//! Windows too, where `%USERPROFILE%` stands in for `$HOME`), as opencode looks it up
//! ([`config_dir`], [`detect`]). Its file is `opencode.jsonc` when there is one, else
//! `opencode.json` ([`Config::path`]); both are read as JSONC (comments, trailing commas).
//!
//! Edits are minimal text edits of the file, so every other key, provider, comment and the
//! file's indentation stay as they were; the file is backed up before each write
//! (`opencode.json.oxinfer-<unix time>.bak`, the last five kept) and replaced atomically (a
//! temporary file renamed over it, with the same permissions; a symbolic link keeps pointing at
//! the file it pointed at, which is the one written).
//!
//! A model's entry may also be `MODEL@PROFILE` ([`profile_id`]): the same model, whose requests
//! the server logs and runs with the profile's adapter (docs/learning.md). [`Config::install_plugin`]
//! writes the opencode plugin that reports the outcomes of such a session
//! (`plugins/oxinfer-outcomes.js`, [`PLUGIN_SOURCE`]).
//!
//! ```no_run
//! use oxinfer::opencode::{Capabilities, Config, Model, Provider};
//! let cfg = oxinfer::opencode::detect().expect("opencode is not set up for this user");
//! let caps = Capabilities { vision: true, thinking: true, tools: true, context: Some(131072) };
//! let provider = Provider { base_url: "http://127.0.0.1:8080/v1".into(), api_key: None };
//! let change = cfg.add(&provider, &Model::new("org/model", &caps)).unwrap();
//! println!("{} changed: {}", change.path.display(), change.changed);
//! cfg.remove("org/model").unwrap();
//! ```

use anyhow::{Context, Result, bail};
use serde::Serialize;
use serde_json::{Value, json};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub use crate::engine::Capabilities;

/// The provider id in opencode's `provider` map.
pub const PROVIDER: &str = "oxinfer";
/// The AI SDK package opencode talks to the server with.
pub const NPM: &str = "@ai-sdk/openai-compatible";
/// opencode's configuration schema (`$schema` of a file created here).
pub const SCHEMA: &str = "https://opencode.ai/config.json";
/// The context written when the model's is not known.
pub const DEFAULT_CONTEXT: u64 = 32768;
/// A cap on the context written for a model without a `max_context` option ([`Model::capped`])
/// that suits a machine which cannot hold a model's trained context (the server applies none
/// unless asked: `opencode_context`). opencode compacts a conversation as it nears
/// `limit.context`.
pub const CONTEXT_CAP: u64 = 131072;
/// The outcome plugin's file name in opencode's `plugins` directory.
pub const PLUGIN_FILE: &str = "oxinfer-outcomes.js";
/// The outcome plugin ([`Config::install_plugin`]): it reports the test, build and lint results
/// of a `MODEL@PROFILE` session to the server (`POST /v1/outcomes`).
pub const PLUGIN_SOURCE: &str = include_str!("opencode/oxinfer-outcomes.js");
/// Backups kept per configuration file.
const BACKUPS: usize = 5;

/// Writes are serialized (two requests editing the file at once would lose one's edit).
static WRITE: Mutex<()> = Mutex::new(());

/// opencode's global configuration directory for this user, whether or not it exists:
/// `$XDG_CONFIG_HOME/opencode` (an absolute `XDG_CONFIG_HOME`), else `$HOME/.config/opencode`.
pub fn config_dir() -> Option<PathBuf> {
    config_dir_from(|k| std::env::var_os(k))
}

fn config_dir_from(env: impl Fn(&str) -> Option<OsString>) -> Option<PathBuf> {
    let set = |k: &str| env(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    if let Some(x) = set("XDG_CONFIG_HOME").filter(|p| p.is_absolute()) {
        return Some(x.join("opencode"));
    }
    let home = set("HOME").or_else(|| if cfg!(windows) { set("USERPROFILE") } else { None })?;
    Some(home.join(".config").join("opencode"))
}

/// opencode's configuration when its directory exists (opencode has been run or set up).
pub fn detect() -> Option<Config> {
    config_dir().filter(|d| d.is_dir()).map(Config::at)
}

/// Where the provider's models are served: the server's `/v1` URL and its API key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Provider {
    /// `http://127.0.0.1:8080/v1`.
    pub base_url: String,
    /// Written as the provider's `options.apiKey` (when the server has one).
    pub api_key: Option<String>,
}

/// The id of `model` run under `profile` (`MODEL@PROFILE`), as a chat request names it.
pub fn profile_id(model: &str, profile: &str) -> String {
    format!("{model}@{profile}")
}

/// One model's entry under `provider.oxinfer.models`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Model {
    /// The id the server lists it under (`/v1/models`), the key of the entry.
    pub id: String,
    /// What opencode shows; written when the entry is created and kept afterwards.
    pub name: String,
    /// It writes a thinking block.
    pub reasoning: bool,
    /// Its chat format has tool calls.
    pub tool_call: bool,
    /// It takes images (its vision tower or mmproj file is there).
    pub attachment: bool,
    /// `modalities.input`: `text`, and `image` with [`Model::attachment`].
    pub input: Vec<String>,
    /// `limit.context` and `limit.output`, in tokens.
    pub context: u64,
    pub output: u64,
}

impl Model {
    /// The entry of a model with these capabilities: the context is the model's (its
    /// `max_context` option or its trained context), [`DEFAULT_CONTEXT`] when unknown; the
    /// output a quarter of it, from 1024 to 32768 tokens.
    pub fn new(id: &str, caps: &Capabilities) -> Model {
        Model::capped(id, caps, None)
    }

    /// [`Model::new`] with the context limited to `cap` tokens: for a model whose context is its
    /// trained one, not a `max_context` the user chose (the server's `opencode_context`; none by default).
    pub fn capped(id: &str, caps: &Capabilities, cap: Option<u64>) -> Model {
        let context = caps.context.map_or(DEFAULT_CONTEXT, |c| c as u64).max(1);
        let context = cap.map_or(context, |c| context.min(c.max(1)));
        let output = (context / 4).clamp(1024.min(context), 32768);
        let mut input = vec!["text".to_string()];
        if caps.vision {
            input.push("image".to_string());
        }
        Model {
            id: id.to_string(),
            name: format!("{id} (oxinfer)"),
            reasoning: caps.thinking,
            tool_call: caps.tools,
            attachment: caps.vision,
            input,
            context,
            output,
        }
    }

    /// The fields kept up to date on an existing entry (all of the entry but its name).
    fn managed(&self) -> Vec<(&'static str, Value)> {
        vec![
            ("reasoning", json!(self.reasoning)),
            ("tool_call", json!(self.tool_call)),
            ("attachment", json!(self.attachment)),
            ("temperature", json!(true)),
            ("modalities", json!({"input": self.input, "output": ["text"]})),
            ("limit", json!({"context": self.context, "output": self.output})),
        ]
    }

    /// The entry as written to the file.
    pub fn to_json(&self) -> Value {
        let mut m = serde_json::Map::new();
        m.insert("name".into(), json!(self.name));
        for (k, v) in self.managed() {
            m.insert(k.into(), v);
        }
        Value::Object(m)
    }
}

/// A model's entry in the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EntryState {
    Absent,
    /// There, and what [`Config::add`] would write (the provider's URL and key included).
    Current,
    /// There, but [`Config::add`] would change it or the provider.
    Outdated,
}

/// What [`Config::install_plugin`] or [`Config::remove_plugin`] did.
#[derive(Clone, Debug, Serialize)]
pub struct PluginChange {
    /// The plugin's file.
    pub path: PathBuf,
    /// It was written or removed (false: it already was as asked).
    pub changed: bool,
}

/// What [`Config::add`] or [`Config::remove`] did.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Change {
    /// The file.
    pub path: PathBuf,
    /// The file was written (false: it already said that).
    pub changed: bool,
    /// The file did not exist and was created.
    pub created: bool,
    /// The copy made before writing.
    pub backup: Option<PathBuf>,
    /// The provider had no model left and was removed with the model.
    pub provider_removed: bool,
    /// opencode's `model` / `small_model` named the removed model (they are left as they are).
    pub default_model: Vec<String>,
    /// Comments that were inside the replaced or removed entries (gone with them).
    pub comments_removed: usize,
}

/// opencode's configuration in a directory.
#[derive(Clone, Debug)]
pub struct Config {
    dir: PathBuf,
}

impl Config {
    /// The configuration in `dir` (which [`config_dir`] gives, or another one, for tests).
    pub fn at(dir: impl Into<PathBuf>) -> Config {
        Config { dir: dir.into() }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The file edited: `opencode.jsonc` when it exists (opencode reads it last, so it wins),
    /// unless only `opencode.json` has the `oxinfer` provider; else `opencode.json`, which may
    /// not exist yet.
    pub fn path(&self) -> PathBuf {
        let json = self.dir.join("opencode.json");
        let jsonc = self.dir.join("opencode.jsonc");
        let has = |p: &Path| {
            std::fs::read_to_string(p)
                .ok()
                .and_then(|t| parse_value(&t).ok())
                .is_some_and(|v| v["provider"][PROVIDER].is_object())
        };
        match (json.is_file(), jsonc.is_file()) {
            (true, true) if has(&json) && !has(&jsonc) => json,
            (_, true) => jsonc,
            _ => json,
        }
    }

    /// The file's text (None when it does not exist).
    fn text(&self, path: &Path) -> Result<Option<String>> {
        match std::fs::read_to_string(path) {
            Ok(t) => Ok(Some(t)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e).with_context(|| format!("read {}", path.display())),
        }
    }

    /// The configuration as JSON (comments and trailing commas dropped); None without a file.
    pub fn read(&self) -> Result<Option<Value>> {
        let path = self.path();
        self.text(&path)?
            .map(|t| parse_value(&t).with_context(|| format!("{}", path.display())))
            .transpose()
    }

    /// The ids under `provider.oxinfer.models`.
    pub fn models(&self) -> Result<Vec<String>> {
        Ok(self
            .read()?
            .and_then(|v| v["provider"][PROVIDER]["models"].as_object().map(|m| m.keys().cloned().collect()))
            .unwrap_or_default())
    }

    /// Whether `model` has an entry, and whether it is what [`Config::add`] would write.
    pub fn state(&self, provider: &Provider, model: &Model) -> Result<EntryState> {
        Ok(entry_state(&self.read()?.unwrap_or_else(|| json!({})), provider, model))
    }

    /// Adds `model` to the `oxinfer` provider (created when missing), or brings its entry and
    /// the provider's URL and key up to date. Nothing is written when nothing changes.
    pub fn add(&self, provider: &Provider, model: &Model) -> Result<Change> {
        let _w = WRITE.lock().unwrap_or_else(|p| p.into_inner());
        let path = self.path();
        let old = self.text(&path)?;
        let base = old.clone().unwrap_or_else(|| format!("{{\n  \"$schema\": \"{SCHEMA}\"\n}}\n"));
        let (text, comments_removed) = add_text(&base, provider, model).with_context(|| format!("{}", path.display()))?;
        let mut c = Change {
            path: path.clone(),
            comments_removed,
            ..Change::default()
        };
        if old.as_deref() != Some(text.as_str()) {
            c.backup = save(&path, old.is_some(), &text)?;
            c.changed = true;
            c.created = old.is_none();
        }
        Ok(c)
    }

    /// Where the outcome plugin is or would be: `plugins/oxinfer-outcomes.js` in the
    /// configuration directory (`plugin/` when it is already there; opencode loads both).
    pub fn plugin_path(&self) -> PathBuf {
        let old = self.dir.join("plugin").join(PLUGIN_FILE);
        if old.is_file() {
            old
        } else {
            self.dir.join("plugins").join(PLUGIN_FILE)
        }
    }

    /// Whether the outcome plugin is installed, and whether it is this version's.
    pub fn plugin_state(&self) -> EntryState {
        match std::fs::read_to_string(self.plugin_path()) {
            Ok(t) if t == PLUGIN_SOURCE => EntryState::Current,
            Ok(_) => EntryState::Outdated,
            Err(_) => EntryState::Absent,
        }
    }

    /// Writes the outcome plugin ([`PLUGIN_SOURCE`]) to [`Config::plugin_path`], replacing
    /// another version of it. opencode loads it at its next start.
    pub fn install_plugin(&self) -> Result<PluginChange> {
        let _w = WRITE.lock().unwrap_or_else(|p| p.into_inner());
        let path = self.plugin_path();
        let state = self.plugin_state();
        if state != EntryState::Current {
            let dir = path.parent().context("the file has no directory")?;
            std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
            let tmp = dir.join(format!(".{PLUGIN_FILE}.oxinfer-{}.tmp", std::process::id()));
            let _ = std::fs::remove_file(&tmp);
            let r = replace_with(&tmp, &path, state != EntryState::Absent, PLUGIN_SOURCE);
            if r.is_err() {
                let _ = std::fs::remove_file(&tmp);
            }
            r?;
        }
        Ok(PluginChange {
            path,
            changed: state != EntryState::Current,
        })
    }

    /// Deletes the outcome plugin's file.
    pub fn remove_plugin(&self) -> Result<PluginChange> {
        let _w = WRITE.lock().unwrap_or_else(|p| p.into_inner());
        let path = self.plugin_path();
        let changed = match std::fs::remove_file(&path) {
            Ok(()) => true,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => false,
            Err(e) => return Err(e).with_context(|| format!("remove {}", path.display())),
        };
        Ok(PluginChange { path, changed })
    }

    /// Removes the entry of model `id`, and the provider when it has no model left. opencode's
    /// default model is left alone ([`Change::default_model`] says when it named this one).
    pub fn remove(&self, id: &str) -> Result<Change> {
        let _w = WRITE.lock().unwrap_or_else(|p| p.into_inner());
        let path = self.path();
        let mut c = Change {
            path: path.clone(),
            ..Change::default()
        };
        let Some(old) = self.text(&path)? else { return Ok(c) };
        let r = remove_text(&old, id).with_context(|| format!("{}", path.display()))?;
        c.provider_removed = r.provider_removed;
        c.default_model = r.default_model;
        c.comments_removed = r.comments_removed;
        if r.text != old {
            c.backup = save(&path, true, &r.text)?;
            c.changed = true;
        }
        Ok(c)
    }
}

/// The JSON of a JSONC text (an empty text is `{}`).
pub fn parse_value(text: &str) -> Result<Value> {
    jsonc::parse(text)?.value(text)
}

/// The edits [`Config::add`] makes: (path, value) pairs, applied in order.
fn wanted(v: &Value, p: &Provider, m: &Model) -> Vec<(Vec<String>, Value)> {
    let path = |ks: &[&str]| ks.iter().map(|k| k.to_string()).collect::<Vec<_>>();
    let prov = &v["provider"][PROVIDER];
    if !prov.is_object() {
        let mut options = json!({"baseURL": p.base_url});
        if let Some(k) = &p.api_key {
            options["apiKey"] = json!(k);
        }
        let mut models = serde_json::Map::new();
        models.insert(m.id.clone(), m.to_json());
        let value = json!({"npm": NPM, "name": "oxinfer (local)", "options": options, "models": models});
        return vec![(path(&["provider", PROVIDER]), value)];
    }
    let mut out = Vec::new();
    if prov["npm"] != NPM {
        out.push((path(&["provider", PROVIDER, "npm"]), json!(NPM)));
    }
    if prov["options"]["baseURL"] != p.base_url.as_str() {
        out.push((path(&["provider", PROVIDER, "options", "baseURL"]), json!(p.base_url)));
    }
    if let Some(k) = &p.api_key
        && prov["options"]["apiKey"] != k.as_str()
    {
        out.push((path(&["provider", PROVIDER, "options", "apiKey"]), json!(k)));
    }
    let entry = &prov["models"][m.id.as_str()];
    if !entry.is_object() {
        out.push((path(&["provider", PROVIDER, "models", m.id.as_str()]), m.to_json()));
    } else {
        for (k, val) in m.managed() {
            if entry[k] != val {
                out.push((path(&["provider", PROVIDER, "models", m.id.as_str(), k]), val));
            }
        }
    }
    out
}

/// [`Config::state`] of an already read configuration ([`Config::read`]).
pub fn entry_state(v: &Value, p: &Provider, m: &Model) -> EntryState {
    if !v["provider"][PROVIDER]["models"][m.id.as_str()].is_object() {
        EntryState::Absent
    } else if wanted(v, p, m).is_empty() {
        EntryState::Current
    } else {
        EntryState::Outdated
    }
}

/// `text` with `model` added (and the comments that were inside replaced values).
fn add_text(text: &str, p: &Provider, m: &Model) -> Result<(String, usize)> {
    let mut t = text.to_string();
    let mut removed = 0;
    let todo = wanted(&parse_value(&t)?, p, m);
    for (path, value) in todo {
        let doc = jsonc::parse(&t)?;
        let keys: Vec<&str> = path.iter().map(String::as_str).collect();
        let e = jsonc::set(&t, &doc, &keys, &value)?;
        removed += doc.comments_in(&[&e]);
        t = jsonc::apply(&t, vec![e]);
    }
    Ok((t, removed))
}

struct Removed {
    text: String,
    provider_removed: bool,
    default_model: Vec<String>,
    comments_removed: usize,
}

fn remove_text(text: &str, id: &str) -> Result<Removed> {
    let v = parse_value(text)?;
    let full = format!("{PROVIDER}/{id}");
    let default_model = ["model", "small_model"]
        .iter()
        .filter(|k| v[**k] == full.as_str())
        .map(|k| k.to_string())
        .collect();
    let mut r = Removed {
        text: text.to_string(),
        provider_removed: false,
        default_model,
        comments_removed: 0,
    };
    if v["provider"][PROVIDER]["models"].get(id).is_none() {
        return Ok(r);
    }
    let doc = jsonc::parse(text)?;
    if let Some(edits) = jsonc::remove(text, &doc, &["provider", PROVIDER, "models", id])? {
        r.comments_removed += doc.comments_in(&edits.iter().collect::<Vec<_>>());
        r.text = jsonc::apply(text, edits);
    }
    let v = parse_value(&r.text)?;
    if v["provider"][PROVIDER]["models"].as_object().is_some_and(|m| m.is_empty()) {
        let doc = jsonc::parse(&r.text)?;
        if let Some(edits) = jsonc::remove(&r.text, &doc, &["provider", PROVIDER])? {
            r.comments_removed += doc.comments_in(&edits.iter().collect::<Vec<_>>());
            r.text = jsonc::apply(&r.text, edits);
            r.provider_removed = true;
        }
    }
    Ok(r)
}

/// Unix time in seconds.
fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// Copies `target` to `<name>.oxinfer-<time>.bak` beside it and keeps the last [`BACKUPS`].
fn backup(target: &Path) -> Result<PathBuf> {
    let dir = target.parent().context("the file has no directory")?;
    let name = target.file_name().context("the file has no name")?.to_string_lossy().into_owned();
    let prefix = format!("{name}.oxinfer-");
    let now = unix_now();
    let mut path = dir.join(format!("{prefix}{now}.bak"));
    let mut n = 1;
    while path.exists() {
        n += 1;
        path = dir.join(format!("{prefix}{now}-{n}.bak"));
    }
    std::fs::copy(target, &path).with_context(|| format!("back up {} to {}", target.display(), path.display()))?;
    // The oldest go (by their time, then their counter).
    let key = |s: &str| -> (u64, u64) {
        let t = s.strip_prefix(&prefix).and_then(|r| r.strip_suffix(".bak")).unwrap_or("");
        let (a, b) = t.split_once('-').unwrap_or((t, "1"));
        (a.parse().unwrap_or(0), b.parse().unwrap_or(0))
    };
    let mut old: Vec<String> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|n| n.starts_with(&prefix) && n.ends_with(".bak"))
                .collect()
        })
        .unwrap_or_default();
    old.sort_by_key(|n| key(n));
    let extra = old.len().saturating_sub(BACKUPS);
    for n in &old[..extra] {
        let _ = std::fs::remove_file(dir.join(n));
    }
    Ok(path)
}

/// Writes `text` to `path` atomically (a temporary file beside it, renamed over it) with the
/// old file's permissions, after backing the old file up. Returns the backup.
fn save(path: &Path, exists: bool, text: &str) -> Result<Option<PathBuf>> {
    // A symbolic link (dotfiles): the file it points at is the one replaced.
    let target = if exists {
        std::fs::canonicalize(path).with_context(|| format!("resolve {}", path.display()))?
    } else {
        path.to_path_buf()
    };
    let dir = target.parent().context("the file has no directory")?;
    if !dir.is_dir() {
        bail!("{} does not exist", dir.display());
    }
    let backup = if exists { Some(backup(&target)?) } else { None };
    let name = target.file_name().context("the file has no name")?.to_string_lossy().into_owned();
    let tmp = dir.join(format!(".{name}.oxinfer-{}.tmp", std::process::id()));
    let _ = std::fs::remove_file(&tmp);
    let r = replace_with(&tmp, &target, exists, text);
    if r.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    r.map(|_| backup)
}

/// Writes `text` to `tmp`, gives it the permissions of `target` (or the owner's only, for a new
/// file) and renames it over `target`.
fn replace_with(tmp: &Path, target: &Path, exists: bool, text: &str) -> Result<()> {
    use std::io::Write;
    {
        // Closed at the end of the block, before the rename.
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(tmp)
            .with_context(|| format!("create {}", tmp.display()))?;
        f.write_all(text.as_bytes())?;
        f.sync_all()?;
    }
    if exists {
        std::fs::set_permissions(tmp, std::fs::metadata(target)?.permissions())?;
    } else {
        // It may hold the server's API key: readable by its owner only.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(tmp, std::fs::Permissions::from_mode(0o600))?;
        }
    }
    std::fs::rename(tmp, target).with_context(|| format!("replace {}", target.display()))?;
    Ok(())
}

/// JSONC (JSON with `//` and `/* */` comments and trailing commas) with the byte spans of its
/// values, and minimal edits of it that keep the rest of the text as it is.
pub(crate) mod jsonc {
    use anyhow::{Result, anyhow, bail};
    use serde::Serialize;
    use serde_json::Value;

    #[derive(Debug)]
    pub struct Node {
        /// Byte range of the value (`end` is past its last byte).
        pub start: usize,
        pub end: usize,
        pub kind: Kind,
    }

    #[derive(Debug)]
    pub enum Kind {
        Object(Vec<Member>),
        Array,
        Scalar,
    }

    #[derive(Debug)]
    pub struct Member {
        pub key: String,
        /// Where the key's opening quote is.
        pub key_start: usize,
        pub value: Node,
        /// The comma after the value.
        pub comma: Option<usize>,
    }

    #[derive(Debug)]
    pub struct Doc {
        /// None for a text of only whitespace and comments.
        pub root: Option<Node>,
        /// Byte ranges of the comments.
        pub comments: Vec<(usize, usize)>,
        /// Positions of the trailing commas and of a byte-order mark's bytes.
        blanks: Vec<(usize, usize)>,
    }

    /// One replacement of a byte range.
    #[derive(Debug)]
    pub struct Edit {
        pub start: usize,
        pub end: usize,
        pub text: String,
    }

    struct P<'a> {
        s: &'a str,
        b: &'a [u8],
        i: usize,
        comments: Vec<(usize, usize)>,
        blanks: Vec<(usize, usize)>,
    }

    impl P<'_> {
        fn err(&self, msg: &str) -> anyhow::Error {
            let before = &self.b[..self.i.min(self.b.len())];
            let line = before.iter().filter(|&&c| c == b'\n').count() + 1;
            let col = before.len() - before.iter().rposition(|&c| c == b'\n').map_or(0, |p| p + 1) + 1;
            anyhow!("line {line}, column {col}: {msg}")
        }

        fn peek(&self) -> Option<u8> {
            self.b.get(self.i).copied()
        }

        fn ws(&mut self) -> Result<()> {
            loop {
                match (self.peek(), self.b.get(self.i + 1)) {
                    (Some(b' ' | b'\t' | b'\n' | b'\r'), _) => self.i += 1,
                    (Some(b'/'), Some(b'/')) => {
                        let s = self.i;
                        while self.peek().is_some_and(|c| c != b'\n') {
                            self.i += 1;
                        }
                        self.comments.push((s, self.i));
                    }
                    (Some(b'/'), Some(b'*')) => {
                        let s = self.i;
                        self.i += 2;
                        loop {
                            match (self.peek(), self.b.get(self.i + 1)) {
                                (Some(b'*'), Some(b'/')) => {
                                    self.i += 2;
                                    break;
                                }
                                (Some(_), _) => self.i += 1,
                                (None, _) => {
                                    self.i = s;
                                    return Err(self.err("unterminated /* comment"));
                                }
                            }
                        }
                        self.comments.push((s, self.i));
                    }
                    _ => return Ok(()),
                }
            }
        }

        fn string(&mut self) -> Result<()> {
            let s = self.i;
            self.i += 1;
            loop {
                match self.peek() {
                    None => {
                        self.i = s;
                        return Err(self.err("unterminated string"));
                    }
                    Some(b'\\') => self.i += 2,
                    Some(b'"') => {
                        self.i += 1;
                        return Ok(());
                    }
                    Some(_) => self.i += 1,
                }
            }
        }

        fn value(&mut self, depth: usize) -> Result<Node> {
            if depth > 512 {
                return Err(self.err("nested too deeply"));
            }
            let start = self.i;
            let kind = match self.peek() {
                None => return Err(self.err("unexpected end of the file")),
                Some(b'{') => {
                    self.i += 1;
                    let mut members = Vec::new();
                    self.ws()?;
                    if self.peek() == Some(b'}') {
                        self.i += 1;
                    } else {
                        loop {
                            if self.peek() != Some(b'"') {
                                return Err(self.err("expected a key in quotes"));
                            }
                            let key_start = self.i;
                            self.string()?;
                            let key: String =
                                serde_json::from_str(&self.s[key_start..self.i]).map_err(|e| self.err(&format!("bad key: {e}")))?;
                            self.ws()?;
                            if self.peek() != Some(b':') {
                                return Err(self.err("expected ':'"));
                            }
                            self.i += 1;
                            self.ws()?;
                            let value = self.value(depth + 1)?;
                            self.ws()?;
                            match self.peek() {
                                Some(b',') => {
                                    let c = self.i;
                                    self.i += 1;
                                    self.ws()?;
                                    members.push(Member {
                                        key,
                                        key_start,
                                        value,
                                        comma: Some(c),
                                    });
                                    if self.peek() == Some(b'}') {
                                        self.blanks.push((c, c + 1));
                                        self.i += 1;
                                        break;
                                    }
                                }
                                Some(b'}') => {
                                    members.push(Member {
                                        key,
                                        key_start,
                                        value,
                                        comma: None,
                                    });
                                    self.i += 1;
                                    break;
                                }
                                _ => return Err(self.err("expected ',' or '}'")),
                            }
                        }
                    }
                    Kind::Object(members)
                }
                Some(b'[') => {
                    self.i += 1;
                    self.ws()?;
                    if self.peek() == Some(b']') {
                        self.i += 1;
                    } else {
                        loop {
                            self.value(depth + 1)?;
                            self.ws()?;
                            match self.peek() {
                                Some(b',') => {
                                    let c = self.i;
                                    self.i += 1;
                                    self.ws()?;
                                    if self.peek() == Some(b']') {
                                        self.blanks.push((c, c + 1));
                                        self.i += 1;
                                        break;
                                    }
                                }
                                Some(b']') => {
                                    self.i += 1;
                                    break;
                                }
                                _ => return Err(self.err("expected ',' or ']'")),
                            }
                        }
                    }
                    Kind::Array
                }
                Some(b'"') => {
                    self.string()?;
                    Kind::Scalar
                }
                Some(_) => {
                    while self
                        .peek()
                        .is_some_and(|c| !matches!(c, b',' | b']' | b'}' | b':' | b' ' | b'\t' | b'\n' | b'\r' | b'/'))
                    {
                        self.i += 1;
                    }
                    if self.i == start {
                        return Err(self.err("unexpected character"));
                    }
                    Kind::Scalar
                }
            };
            Ok(Node { start, end: self.i, kind })
        }
    }

    pub fn parse(text: &str) -> Result<Doc> {
        let mut p = P {
            s: text,
            b: text.as_bytes(),
            i: 0,
            comments: Vec::new(),
            blanks: Vec::new(),
        };
        if text.starts_with('\u{feff}') {
            p.blanks.push((0, 3));
            p.i = 3;
        }
        p.ws()?;
        let root = if p.i >= p.b.len() {
            None
        } else {
            let n = p.value(0)?;
            p.ws()?;
            if p.i < p.b.len() {
                return Err(p.err("unexpected text after the end"));
            }
            Some(n)
        };
        Ok(Doc {
            root,
            comments: p.comments,
            blanks: p.blanks,
        })
    }

    impl Doc {
        /// The text's JSON value (scalars are checked here, by serde_json).
        pub fn value(&self, text: &str) -> Result<Value> {
            if self.root.is_none() {
                return Ok(Value::Object(Default::default()));
            }
            let mut b = text.as_bytes().to_vec();
            for &(s, e) in self.comments.iter().chain(&self.blanks) {
                b[s..e].fill(b' ');
            }
            let plain = String::from_utf8(b).map_err(|e| anyhow!("{e}"))?;
            serde_json::from_str(&plain).map_err(|e| anyhow!("not valid JSON(C): {e}"))
        }

        /// The comments wholly inside the ranges the edits replace.
        pub fn comments_in(&self, edits: &[&Edit]) -> usize {
            self.comments
                .iter()
                .filter(|&&(s, e)| edits.iter().any(|x| x.start <= s && e <= x.end))
                .count()
        }

        fn root_object(&self) -> Result<&Node> {
            match &self.root {
                Some(n @ Node { kind: Kind::Object(_), .. }) => Ok(n),
                Some(_) => bail!("the top level is not an object"),
                None => bail!("the file is empty"),
            }
        }
    }

    /// Applies non-overlapping edits.
    pub fn apply(text: &str, mut edits: Vec<Edit>) -> String {
        edits.sort_by_key(|e| std::cmp::Reverse(e.start));
        let mut t = text.to_string();
        for e in edits {
            t.replace_range(e.start..e.end, &e.text);
        }
        t
    }

    fn line_start(text: &str, pos: usize) -> usize {
        text[..pos].rfind('\n').map_or(0, |p| p + 1)
    }

    /// The whitespace that starts the line of `pos`.
    fn line_indent(text: &str, pos: usize) -> &str {
        let ls = line_start(text, pos);
        let rest = &text[ls..];
        &rest[..rest.len() - rest.trim_start_matches([' ', '\t']).len()]
    }

    /// The whitespace before `pos` when only whitespace stands between it and its line's start.
    fn own_line_indent(text: &str, pos: usize) -> Option<&str> {
        let ls = line_start(text, pos);
        let before = &text[ls..pos];
        before.bytes().all(|c| c == b' ' || c == b'\t').then_some(before)
    }

    /// The file's indentation step: that of its first indented line (two spaces when none).
    fn unit(text: &str) -> String {
        for line in text.lines() {
            let ws = &line[..line.len() - line.trim_start_matches([' ', '\t']).len()];
            if !ws.is_empty() && ws.len() < line.len() {
                return if ws.starts_with('\t') { "\t".into() } else { ws.into() };
            }
        }
        "  ".into()
    }

    /// `v` pretty-printed with `unit` per level, its lines after the first indented by `base`.
    fn render(v: &Value, unit: &str, base: &str) -> String {
        let mut buf = Vec::new();
        let mut ser = serde_json::Serializer::with_formatter(&mut buf, serde_json::ser::PrettyFormatter::with_indent(unit.as_bytes()));
        v.serialize(&mut ser).expect("a JSON value serializes");
        String::from_utf8(buf)
            .expect("serde_json writes UTF-8")
            .replace('\n', &format!("\n{base}"))
    }

    fn nest(rest: &[&str], value: &Value) -> Value {
        rest.iter().rev().fold(value.clone(), |v, k| {
            let mut m = serde_json::Map::new();
            m.insert(k.to_string(), v);
            Value::Object(m)
        })
    }

    /// The edit that sets the value at `path` (objects along it are created as needed; a
    /// non-object in the way is replaced).
    pub fn set(text: &str, doc: &Doc, path: &[&str], value: &Value) -> Result<Edit> {
        let unit = unit(text);
        let mut obj = doc.root_object()?;
        for (k, key) in path.iter().enumerate() {
            let Kind::Object(members) = &obj.kind else {
                unreachable!("only objects are walked")
            };
            // A repeated key: the last one is what JSON readers take.
            match members.iter().rev().find(|m| m.key == *key) {
                Some(m) if k + 1 < path.len() && matches!(m.value.kind, Kind::Object(_)) => obj = &m.value,
                Some(m) => {
                    let base = own_line_indent(text, m.key_start).unwrap_or_else(|| line_indent(text, m.key_start));
                    return Ok(Edit {
                        start: m.value.start,
                        end: m.value.end,
                        text: render(&nest(&path[k + 1..], value), &unit, base),
                    });
                }
                None => return Ok(insert(text, obj, key, &nest(&path[k + 1..], value), &unit)),
            }
        }
        bail!("an empty path")
    }

    /// The edit that adds member `key` at the end of object `obj`.
    fn insert(text: &str, obj: &Node, key: &str, value: &Value, unit: &str) -> Edit {
        let Kind::Object(members) = &obj.kind else {
            unreachable!("insert into an object")
        };
        let key = serde_json::to_string(key).expect("a string serializes");
        let close = obj.end - 1;
        match members.last() {
            None => {
                let outer = line_indent(text, obj.start).to_string();
                let inner = format!("{outer}{unit}");
                let member = format!("\n{inner}{key}: {}", render(value, unit, &inner));
                if text[obj.start + 1..close].trim().is_empty() {
                    Edit {
                        start: obj.start + 1,
                        end: close,
                        text: format!("{member}\n{outer}"),
                    }
                } else {
                    // Comments inside: they stay after the new member.
                    Edit {
                        start: obj.start + 1,
                        end: obj.start + 1,
                        text: member,
                    }
                }
            }
            Some(last) => {
                let (pos, comma) = match last.comma {
                    Some(c) => (c + 1, ""),
                    None => (last.value.end, ","),
                };
                let member = match own_line_indent(text, last.key_start) {
                    Some(ind) => format!("{comma}\n{ind}{key}: {}", render(value, unit, ind)),
                    // Members on one line: one more on it.
                    None => format!("{comma} {key}: {}", serde_json::to_string(value).expect("a JSON value serializes")),
                };
                Edit {
                    start: pos,
                    end: pos,
                    text: member,
                }
            }
        }
    }

    /// The edits that remove the member at `path` (None when it is not there).
    pub fn remove(text: &str, doc: &Doc, path: &[&str]) -> Result<Option<Vec<Edit>>> {
        let Some((last_key, parents)) = path.split_last() else {
            bail!("an empty path")
        };
        let mut obj = doc.root_object()?;
        for key in parents {
            let Kind::Object(members) = &obj.kind else {
                unreachable!("only objects are walked")
            };
            match members.iter().rev().find(|m| m.key == *key) {
                Some(m) if matches!(m.value.kind, Kind::Object(_)) => obj = &m.value,
                _ => return Ok(None),
            }
        }
        let Kind::Object(members) = &obj.kind else {
            unreachable!("only objects are walked")
        };
        let Some(i) = members.iter().rposition(|m| m.key == *last_key) else {
            return Ok(None);
        };
        let b = text.as_bytes();
        // The whitespace before `pos` (not past a comment; after a line comment, its newline stays).
        let ws_before = |pos: usize| {
            let mut p = pos;
            while p > 0 && matches!(b[p - 1], b' ' | b'\t' | b'\r' | b'\n') {
                p -= 1;
            }
            if doc.comments.iter().any(|&(s, e)| e == p && text[s..].starts_with("//")) {
                p += text[p..pos].find('\n').map_or(0, |n| n + 1);
            }
            p
        };
        let m = &members[i];
        let end = m.comma.map_or(m.value.end, |c| c + 1);
        let from = ws_before(m.key_start);
        Ok(Some(if members.len() == 1 {
            // The only member: `{}` when nothing else was inside.
            let close = obj.end - 1;
            let to = if from == obj.start + 1 && text[end..close].trim().is_empty() {
                close
            } else {
                end
            };
            vec![Edit {
                start: from,
                end: to,
                text: String::new(),
            }]
        } else if i + 1 == members.len() {
            let pc = members[i - 1].comma.expect("a member followed by another has a comma");
            vec![
                Edit {
                    start: pc,
                    end: pc + 1,
                    text: String::new(),
                },
                Edit {
                    start: from,
                    end,
                    text: String::new(),
                },
            ]
        } else {
            vec![Edit {
                start: from,
                end,
                text: String::new(),
            }]
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps(vision: bool) -> Capabilities {
        Capabilities {
            vision,
            thinking: true,
            tools: true,
            context: Some(131072),
        }
    }

    fn provider() -> Provider {
        Provider {
            base_url: "http://127.0.0.1:8080/v1".into(),
            api_key: None,
        }
    }

    fn temp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("oxinfer-opencode-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    const USER: &str = r#"{
    // my settings
    "$schema": "https://opencode.ai/config.json",
    "theme": "dark", /* trailing */
    "provider": {
        "other": {
            "npm": "@ai-sdk/openai-compatible",
            "options": { "baseURL": "http://10.0.0.1/v1", "apiKey": "secret", },
            "models": { "m": { "name": "M" } },
        },
    },
    "model": "oxinfer/org/a",
}
"#;

    #[test]
    fn jsonc_parsing() {
        let v = parse_value(USER).unwrap();
        assert_eq!(v["theme"], "dark");
        assert_eq!(v["provider"]["other"]["options"]["apiKey"], "secret");
        assert_eq!(parse_value("").unwrap(), json!({}));
        assert_eq!(parse_value("// only a comment\n").unwrap(), json!({}));
        assert_eq!(parse_value("\u{feff}{\"a\": [1, 2,],}").unwrap(), json!({"a": [1, 2]}));
        assert_eq!(
            parse_value(r#"{"u": "// not a comment", "s": "/* nor this */"}"#).unwrap()["u"],
            "// not a comment"
        );
        assert_eq!(parse_value("{\"é\": \"/* ü */\" /* ü */}").unwrap(), json!({"é": "/* ü */"}));
        for bad in ["{", "{\"a\" 1}", "{\"a\": 1} x", "/* open", "{\"a\": tru}", "[1,,2]", "{,}"] {
            assert!(parse_value(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn model_entry() {
        let m = Model::new("org/a", &caps(true));
        assert_eq!(m.to_json()["modalities"]["input"], json!(["text", "image"]));
        assert_eq!(m.to_json()["limit"], json!({"context": 131072, "output": 32768}));
        let m = Model::new("org/a", &Capabilities::default());
        assert!(!m.attachment && !m.reasoning && !m.tool_call);
        assert_eq!(m.to_json()["modalities"]["input"], json!(["text"]));
        assert_eq!((m.context, m.output), (DEFAULT_CONTEXT, 8192));
        let small = Model::new(
            "s",
            &Capabilities {
                context: Some(2048),
                ..Default::default()
            },
        );
        assert_eq!((small.context, small.output), (2048, 1024));
        // A cap on the context: a million trained tokens, an unknown context, a smaller one.
        let big = Capabilities {
            context: Some(1 << 20),
            ..Default::default()
        };
        let m = Model::new("b", &big);
        assert_eq!((m.context, m.output), (1 << 20, 32768));
        let m = Model::capped("b", &big, Some(CONTEXT_CAP));
        assert_eq!((m.context, m.output), (131072, 32768));
        let m = Model::capped("b", &Capabilities::default(), Some(CONTEXT_CAP));
        assert_eq!(m.context, DEFAULT_CONTEXT);
        assert_eq!(
            Model::capped("s", &caps(true), Some(4096)).to_json()["limit"],
            json!({"context": 4096, "output": 1024})
        );
        assert_eq!(profile_id("org/a", "repo"), "org/a@repo");
    }

    #[test]
    fn plugin_file() {
        let d = temp("plugin");
        let cfg = Config::at(&d);
        assert_eq!(cfg.plugin_state(), EntryState::Absent);
        assert!(!cfg.remove_plugin().unwrap().changed);
        let c = cfg.install_plugin().unwrap();
        assert_eq!((c.changed, c.path.clone()), (true, d.join("plugins").join(PLUGIN_FILE)));
        assert_eq!(std::fs::read_to_string(&c.path).unwrap(), PLUGIN_SOURCE);
        assert_eq!(cfg.plugin_state(), EntryState::Current);
        assert!(!cfg.install_plugin().unwrap().changed);
        std::fs::write(&c.path, "old").unwrap();
        assert_eq!(cfg.plugin_state(), EntryState::Outdated);
        assert!(cfg.install_plugin().unwrap().changed);
        assert_eq!(std::fs::read_to_string(&c.path).unwrap(), PLUGIN_SOURCE);
        assert_eq!(
            std::fs::read_dir(d.join("plugins")).unwrap().count(),
            1,
            "no temporary file is left"
        );
        assert!(cfg.remove_plugin().unwrap().changed);
        assert_eq!(cfg.plugin_state(), EntryState::Absent);
        // One already in opencode's other plugin directory is the one kept up to date.
        std::fs::create_dir_all(d.join("plugin")).unwrap();
        std::fs::write(d.join("plugin").join(PLUGIN_FILE), "old").unwrap();
        assert_eq!(cfg.install_plugin().unwrap().path, d.join("plugin").join(PLUGIN_FILE));
        // The plugin's only export is the plugin (opencode calls every export of the file).
        assert_eq!(PLUGIN_SOURCE.matches("\nexport ").count(), 1);
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn add_update_remove_keep_the_rest() {
        let d = temp("edit");
        let path = d.join("opencode.json");
        std::fs::write(&path, USER).unwrap();
        let cfg = Config::at(&d);
        assert_eq!(cfg.path(), path);
        let m = Model::new("org/a", &caps(false));
        assert_eq!(cfg.state(&provider(), &m).unwrap(), EntryState::Absent);
        let c = cfg.add(&provider(), &m).unwrap();
        assert!(c.changed && !c.created);
        let backup = c.backup.clone().unwrap();
        assert_eq!(
            std::fs::read_to_string(&backup).unwrap(),
            USER,
            "the backup is the file before the write"
        );
        let text = std::fs::read_to_string(&path).unwrap();
        // Comments, other keys and providers, and the indentation are kept.
        assert!(text.starts_with("{\n    // my settings\n"), "{text}");
        assert!(
            text.contains("/* trailing */") && text.contains("\"apiKey\": \"secret\", }"),
            "{text}"
        );
        assert!(text.contains("\n        \"oxinfer\": {\n            \"npm\""), "{text}");
        let v = parse_value(&text).unwrap();
        assert_eq!(v["provider"]["other"]["models"]["m"]["name"], "M");
        assert_eq!(v["provider"]["oxinfer"]["options"]["baseURL"], "http://127.0.0.1:8080/v1");
        assert!(v["provider"]["oxinfer"]["options"].get("apiKey").is_none());
        assert_eq!(v["provider"]["oxinfer"]["models"]["org/a"]["attachment"], false);
        assert_eq!(v["provider"]["oxinfer"]["models"]["org/a"]["name"], "org/a (oxinfer)");
        assert_eq!(cfg.state(&provider(), &m).unwrap(), EntryState::Current);
        // Again: nothing to write.
        let c = cfg.add(&provider(), &m).unwrap();
        assert!(!c.changed && c.backup.is_none());
        // A renamed entry keeps its name; a changed capability, URL or key is updated.
        let renamed = text.replace("\"org/a (oxinfer)\"", "\"Mine\"");
        std::fs::write(&path, &renamed).unwrap();
        let m2 = Model::new("org/a", &caps(true));
        let p2 = Provider {
            base_url: "http://127.0.0.1:9090/v1".into(),
            api_key: Some("k".into()),
        };
        assert_eq!(cfg.state(&p2, &m2).unwrap(), EntryState::Outdated);
        let c = cfg.add(&p2, &m2).unwrap();
        assert!(c.changed);
        let v = cfg.read().unwrap().unwrap();
        let e = &v["provider"]["oxinfer"]["models"]["org/a"];
        assert_eq!(e["name"], "Mine");
        assert_eq!(e["attachment"], true);
        assert_eq!(e["modalities"]["input"], json!(["text", "image"]));
        assert_eq!(
            v["provider"]["oxinfer"]["options"],
            json!({"baseURL": "http://127.0.0.1:9090/v1", "apiKey": "k"})
        );
        assert_eq!(cfg.state(&p2, &m2).unwrap(), EntryState::Current);
        // A second model, then removing both: the provider goes with the last one.
        cfg.add(&p2, &Model::new("org/b", &caps(false))).unwrap();
        assert_eq!(cfg.models().unwrap(), ["org/a", "org/b"]);
        let c = cfg.remove("org/a").unwrap();
        assert!(c.changed && !c.provider_removed);
        assert_eq!(c.default_model, ["model"], "opencode's model named it");
        assert_eq!(cfg.models().unwrap(), ["org/b"]);
        let c = cfg.remove("org/b").unwrap();
        assert!(c.changed && c.provider_removed && c.default_model.is_empty());
        let text = std::fs::read_to_string(&path).unwrap();
        let v = parse_value(&text).unwrap();
        assert!(v["provider"].get("oxinfer").is_none());
        assert_eq!(v["provider"]["other"]["options"]["apiKey"], "secret");
        assert_eq!(v["model"], "oxinfer/org/a", "the default model is the user's to change");
        assert!(text.contains("// my settings") && text.contains("/* trailing */"));
        // Removing what is not there writes nothing.
        assert!(!cfg.remove("org/b").unwrap().changed);
        // Backups: one per write, the last five kept.
        let baks = std::fs::read_dir(&d)
            .unwrap()
            .filter(|e| e.as_ref().unwrap().file_name().to_string_lossy().ends_with(".bak"))
            .count();
        assert_eq!(baks, 5);
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn new_and_jsonc_files() {
        let d = temp("new");
        let cfg = Config::at(&d);
        let c = cfg.add(&provider(), &Model::new("org/a", &caps(true))).unwrap();
        assert!(c.created && c.backup.is_none());
        assert_eq!(c.path, d.join("opencode.json"));
        let v = cfg.read().unwrap().unwrap();
        assert_eq!(v["$schema"], SCHEMA);
        assert_eq!(v["provider"]["oxinfer"]["npm"], NPM);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(&c.path).unwrap().permissions().mode() & 0o777, 0o600);
        }
        // opencode.jsonc wins when there is one (unless only opencode.json has the provider).
        std::fs::write(d.join("opencode.jsonc"), "{\n\t// tabs\n\t\"theme\": \"x\",\n}\n").unwrap();
        assert_eq!(cfg.path(), d.join("opencode.json"));
        cfg.remove("org/a").unwrap();
        assert_eq!(cfg.path(), d.join("opencode.jsonc"));
        cfg.add(&provider(), &Model::new("org/a", &caps(false))).unwrap();
        let text = std::fs::read_to_string(d.join("opencode.jsonc")).unwrap();
        assert!(
            text.contains("\t// tabs\n\t\"theme\": \"x\",\n\t\"provider\": {\n\t\t\"oxinfer\""),
            "{text}"
        );
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn minimal_removals() {
        let rm = |t: &str, path: &[&str]| {
            let doc = jsonc::parse(t).unwrap();
            let e = jsonc::remove(t, &doc, path).unwrap().unwrap();
            let out = jsonc::apply(t, e);
            parse_value(&out).unwrap_or_else(|e| panic!("{out}: {e}"));
            out
        };
        assert_eq!(rm("{\n  \"a\": 1,\n  \"b\": 2\n}", &["a"]), "{\n  \"b\": 2\n}");
        assert_eq!(rm("{\n  \"a\": 1,\n  \"b\": 2\n}", &["b"]), "{\n  \"a\": 1\n}");
        assert_eq!(rm("{\n  \"a\": 1\n}", &["a"]), "{}");
        assert_eq!(rm("{\"a\": 1, // about a\n  \"b\": 2}", &["b"]), "{\"a\": 1 // about a\n}");
        assert_eq!(rm("{ // c\n  \"a\": 1 }", &["a"]), "{ // c\n }");
        assert_eq!(rm("{\"x\": {\"a\": 1, \"b\": 2,}}", &["x", "a"]), "{\"x\": { \"b\": 2,}}");
        let doc = jsonc::parse("{\"a\": 1}").unwrap();
        assert!(jsonc::remove("{\"a\": 1}", &doc, &["z"]).unwrap().is_none());
    }

    #[test]
    #[cfg(unix)]
    fn config_dir_lookup() {
        let env = |pairs: &'static [(&'static str, &'static str)]| {
            move |k: &str| pairs.iter().find(|(n, _)| *n == k).map(|(_, v)| OsString::from(v))
        };
        let x = config_dir_from(env(&[("XDG_CONFIG_HOME", "/x"), ("HOME", "/h")])).unwrap();
        assert_eq!(x, Path::new("/x").join("opencode"));
        let h = config_dir_from(env(&[("XDG_CONFIG_HOME", "rel"), ("HOME", "/h")])).unwrap();
        assert_eq!(h, Path::new("/h").join(".config").join("opencode"));
        assert!(config_dir_from(env(&[])).is_none());
    }
}
