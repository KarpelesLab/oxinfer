//! `GET /api/web/opencode`, `POST /api/web/opencode/add`, `/remove` and `/plugin`: the server's
//! chat models in opencode's global configuration ([`crate::opencode`]), as models of an `oxinfer`
//! provider pointed at this server, plain or under a profile (`MODEL@PROFILE`), and the plugin
//! that reports a profile session's outcomes.
//!
//! Only for requests from this machine (a loopback listener, or a loopback client): the file
//! written is in the home directory of the user running the server.

use super::{Api, ApiError, ApiRequest, ApiResult, Output, WebState, json_ok, str_field};
use crate::opencode::{Capabilities, Config, EntryState, Model, Provider};
use crate::server::Kind;
use crate::server::registry::EntryStatus;
use serde_json::{Value, json};
use std::path::Path;

/// The request comes from this machine: the server listens on loopback, or the client's address
/// is a loopback one.
fn local(w: &WebState, req: &ApiRequest) -> bool {
    w.loopback
        || req
            .peer
            .as_deref()
            .and_then(|p| p.parse::<std::net::SocketAddr>().ok())
            .is_some_and(|a| a.ip().to_canonical().is_loopback())
}

const NOT_LOCAL: &str =
    "opencode's configuration can only be changed from the machine the server runs on (open the GUI at http://localhost)";

/// The provider's `baseURL` (this server's `/v1` as a client on this machine reaches it) and key.
fn provider(api: &Api, req: &ApiRequest) -> Provider {
    let o = &api.opts;
    let scheme = if o.tls_cert.is_some() || o.self_signed { "https" } else { "http" };
    let addr = std::net::ToSocketAddrs::to_socket_addrs(&o.listen).ok().and_then(|mut a| a.next());
    // Port 0 picks one at start: the one the browser used.
    let host_port = req
        .host
        .as_deref()
        .and_then(|h| h.rsplit_once(':'))
        .and_then(|(_, p)| p.parse::<u16>().ok());
    let port = addr.map(|a| a.port()).filter(|&p| p != 0).or(host_port).unwrap_or(8080);
    let host = match addr.map(|a| a.ip()) {
        Some(ip) if !ip.is_unspecified() && !ip.is_loopback() => match ip {
            std::net::IpAddr::V6(v6) => format!("[{v6}]"),
            v4 => v4.to_string(),
        },
        Some(std::net::IpAddr::V6(_)) => "[::1]".to_string(),
        _ => "127.0.0.1".to_string(),
    };
    Provider {
        base_url: format!("{scheme}://{host}:{port}/v1"),
        api_key: o.api_key.clone(),
    }
}

/// The catalog's entry of a model source (`org/name`, `org/name:QUANT`).
fn catalog(source: &str) -> Option<&'static crate::advisor::Model> {
    let base = |s: &str| s.split(':').next().unwrap_or(s).to_ascii_lowercase();
    let cat = crate::advisor::CATALOG;
    cat.iter()
        .find(|m| m.repo.eq_ignore_ascii_case(source))
        .or_else(|| cat.iter().find(|m| base(m.repo) == base(source)))
}

fn is_chat(s: &EntryStatus) -> bool {
    match s.kind {
        Some(k) => k == Kind::Chat,
        None => catalog(&s.spec.source).is_some_and(|m| m.kind == crate::advisor::Kind::Chat),
    }
}

/// The chat template of the model at `path` (a GGUF file, or a folder's
/// `tokenizer_config.json` / `chat_template.jinja`).
fn chat_template(path: &Path) -> Option<String> {
    if crate::gguf::is_gguf(path) {
        return crate::gguf::header(path).ok()?.chat_template().map(str::to_string);
    }
    let dir = if path.is_dir() { path } else { path.parent()? };
    let mut t = std::fs::read_to_string(dir.join("chat_template.jinja")).unwrap_or_default();
    if let Ok(b) = std::fs::read(dir.join("tokenizer_config.json"))
        && let Ok(v) = serde_json::from_slice::<Value>(&b)
    {
        match &v["chat_template"] {
            Value::String(s) => t.push_str(s),
            Value::Array(list) => list.iter().filter_map(|x| x["template"].as_str()).for_each(|s| t.push_str(s)),
            _ => {}
        }
    }
    (!t.is_empty()).then_some(t)
}

/// Whether the vision tower of the model at `path` is there: an mmproj file beside a GGUF
/// model or given with the `mmproj` option (as the engine finds it), a `vision_config` in a
/// folder's `config.json`.
fn vision_files(path: &Path, mmproj: &crate::pixtral::Mmproj) -> bool {
    if *mmproj == crate::pixtral::Mmproj::None {
        return false;
    }
    if crate::gguf::is_gguf(path) {
        return mmproj.file_for(path).is_some();
    }
    let dir = if path.is_dir() { path } else { path.parent().unwrap_or(path) };
    std::fs::read(dir.join("config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
        .is_some_and(|v| v.get("vision_config").is_some_and(|c| !c.is_null()))
}

/// What model `s` can do, for its opencode entry. The one place the entry's capabilities come
/// from: the engine's own ([`crate::engine::Engine::capabilities`], images as
/// [`crate::engine::Engine::accepts_images`] tells) once the model has been loaded
/// (`"loaded"`); before that its files and the catalog (`"files"`): the chat template's
/// thinking and tools, the vision tower or mmproj file on disk, or a vision model of the
/// catalog whose tower its load fetches (a GGUF repository's mmproj), the trained context and
/// the `max_context` option.
pub(super) fn capabilities(api: &Api, s: &EntryStatus) -> (Capabilities, &'static str) {
    if let Some(c) = s.caps {
        return (c, "loaded");
    }
    let spec = &s.spec;
    let path = api.models.local_path(&spec.id);
    let caps = catalog(&spec.source).map_or(&[][..], |m| m.caps);
    let has = |c: crate::advisor::Cap| caps.contains(&c);
    let template = path.as_deref().and_then(chat_template).unwrap_or_default();
    let thinking = has(crate::advisor::Cap::Thinking)
        || template.contains("<think>")
        || template.contains("enable_thinking")
        || template.contains("reasoning_content");
    let tools = has(crate::advisor::Cap::Tools) || template.contains("tools");
    // A vision model of the catalog whose tower its load fetches: a GGUF repository's mmproj
    // file comes with the model unless `mmproj` is `none` (or another file).
    let mmproj = spec.option("mmproj").map(crate::pixtral::Mmproj::parse).unwrap_or_default();
    let image = has(crate::advisor::Cap::ImageInput);
    let fetched = image && mmproj == crate::pixtral::Mmproj::Auto;
    let vision = match path.as_deref() {
        Some(p) => vision_files(p, &mmproj) || (fetched && crate::gguf::is_gguf(p) && spec.source.contains(':')),
        None => fetched || (image && matches!(mmproj, crate::pixtral::Mmproj::File(_))),
    };
    let trained = path.as_deref().and_then(crate::engine::trained_context);
    let max = spec.option("max_context").and_then(|v| v.parse::<usize>().ok()).filter(|&n| n > 0);
    let context = match (max, trained) {
        (Some(m), Some(t)) => Some(m.min(t)),
        (m, t) => m.or(t),
    };
    (
        Capabilities {
            vision,
            thinking,
            tools,
            context,
        },
        "files",
    )
}

/// The opencode entry of model `s`, under `profile` when given: its capabilities with the
/// context capped at [`crate::server::ServerOptions::opencode_context`] unless the model has a
/// `max_context` option (the user's choice stands).
fn entry(api: &Api, s: &EntryStatus, profile: Option<&str>) -> (Model, Capabilities, &'static str) {
    let (caps, source) = capabilities(api, s);
    let chosen = s
        .spec
        .option("max_context")
        .and_then(|v| v.parse::<usize>().ok())
        .is_some_and(|n| n > 0);
    let cap = if chosen { None } else { api.opts.opencode_context };
    let id = match profile {
        Some(p) => crate::opencode::profile_id(&s.spec.id, p),
        None => s.spec.id.clone(),
    };
    (Model::capped(&id, &caps, cap), caps, source)
}

/// Whether requests for `MODEL@PROFILE` are served (as `learn::resolve` decides).
fn profiles_served(api: &Api) -> bool {
    api.opts.learn_dir.is_some() && api.admin()
}

fn plugin_json(cfg: &Config) -> Value {
    let state = cfg.plugin_state();
    json!({
        "path": cfg.plugin_path(),
        "state": state,
        "installed": state != EntryState::Absent,
        "up_to_date": state == EntryState::Current,
    })
}

/// `GET /api/web/opencode`.
pub(super) fn status(api: &Api, w: &WebState, req: &ApiRequest) -> Value {
    if !local(w, req) {
        return json!({"available": false, "detected": false, "reason": NOT_LOCAL});
    }
    let Some(dir) = crate::opencode::config_dir().filter(|d| d.is_dir()) else {
        return json!({"available": true, "detected": false, "dir": crate::opencode::config_dir()});
    };
    let cfg = Config::at(&dir);
    let path = cfg.path();
    let prov = provider(api, req);
    let (value, error) = match cfg.read() {
        Ok(v) => (v.unwrap_or_else(|| json!({})), None),
        Err(e) => (json!({}), Some(format!("{e:#}"))),
    };
    let mut listed = Vec::new();
    let entries: Vec<String> = value["provider"][crate::opencode::PROVIDER]["models"]
        .as_object()
        .map(|o| o.keys().cloned().collect())
        .unwrap_or_default();
    let status = api.models.status();
    let ids: Vec<&str> = status.iter().map(|s| s.spec.id.as_str()).collect();
    let models: Vec<Value> = status
        .iter()
        .filter(|s| is_chat(s))
        .map(|s| {
            let (m, caps, source) = entry(api, s, None);
            let state = crate::opencode::entry_state(&value, &prov, &m);
            listed.push(s.spec.id.clone());
            // Its entries under a profile (`ID@PROFILE`; a listed id is itself).
            let profiles: Vec<Value> = entries
                .iter()
                .filter(|k| !ids.contains(&k.as_str()))
                .filter_map(|k| {
                    let p = k.strip_prefix(s.spec.id.as_str())?.strip_prefix(crate::server::learn::SEPARATOR)?;
                    crate::server::learn::valid_profile(p).then_some((k, p))
                })
                .map(|(k, p)| {
                    let (m, _, _) = entry(api, s, Some(p));
                    let state = crate::opencode::entry_state(&value, &prov, &m);
                    listed.push(k.clone());
                    json!({
                        "id": k,
                        "profile": p,
                        "state": state,
                        "up_to_date": state == EntryState::Current,
                        "entry": m.to_json(),
                    })
                })
                .collect();
            json!({
                "id": s.spec.id,
                "state": state,
                "present": state != EntryState::Absent,
                "up_to_date": state == EntryState::Current,
                "capabilities": caps,
                "source": source,
                "entry": m.to_json(),
                "profiles": profiles,
            })
        })
        .collect();
    let other: Vec<&String> = entries.iter().filter(|k| !listed.contains(k)).collect();
    json!({
        "available": true,
        "detected": true,
        "writable": w.admin && error.is_none(),
        "dir": dir,
        "path": path,
        "exists": path.is_file(),
        "error": error,
        "provider": value["provider"][crate::opencode::PROVIDER].is_object(),
        "base_url": prov.base_url,
        "default_model": value.get("model"),
        "context_cap": api.opts.opencode_context,
        "profiles": profiles_served(api),
        "plugin": plugin_json(&cfg),
        "models": models,
        "other": other,
    })
}

fn detected() -> ApiResult<Config> {
    crate::opencode::detect().ok_or_else(|| {
        let dir = crate::opencode::config_dir().map_or_else(|| "~/.config/opencode".to_string(), |d| d.display().to_string());
        ApiError::new(404, format!("opencode is not set up for this user (no {dir}); run opencode once")).code("opencode_not_found")
    })
}

fn change_json(id: &str, c: &crate::opencode::Change, added: bool) -> Value {
    let mut msg = if !c.changed {
        format!("{id}: {} already says that", c.path.display())
    } else if added {
        format!("{id}: written to {}", c.path.display())
    } else {
        format!("{id}: removed from {}", c.path.display())
    };
    if c.provider_removed {
        msg.push_str(" (the oxinfer provider had no model left and was removed)");
    }
    for k in &c.default_model {
        msg.push_str(&format!(
            ". opencode's \"{k}\" still names {}/{id}: change it there if you want another default",
            crate::opencode::PROVIDER
        ));
    }
    if c.comments_removed > 0 {
        msg.push_str(&format!(
            ". {} comment(s) inside the replaced entry went with it",
            c.comments_removed
        ));
    }
    let mut v = serde_json::to_value(c).unwrap_or_else(|_| json!({}));
    v["model"] = json!(id);
    v["message"] = json!(msg);
    v
}

/// `POST /api/web/opencode/plugin`: `{"install": true}` (the default) writes the outcome plugin
/// to opencode's `plugins` directory, `{"install": false}` deletes it.
pub(super) fn plugin(api: &Api, w: &WebState, req: &ApiRequest, b: &Value) -> ApiResult<Output> {
    if !local(w, req) {
        return Err(ApiError::new(403, NOT_LOCAL).code("opencode_local_only"));
    }
    let install = match b.get("install") {
        None | Some(Value::Null) => true,
        Some(Value::Bool(i)) => *i,
        Some(_) => return Err(ApiError::bad("install must be true or false").param("install")),
    };
    let cfg = detected()?;
    let c = if install { cfg.install_plugin() } else { cfg.remove_plugin() }
        .map_err(|e| ApiError::new(409, format!("opencode's plugin: {e:#}")).code("opencode_config"))?;
    if api.opts.log && c.changed {
        eprintln!("[opencode: {} {}]", if install { "wrote" } else { "removed" }, c.path.display());
    }
    let message = match (install, c.changed) {
        (true, true) => format!("written to {} (opencode loads it at its next start)", c.path.display()),
        (true, false) => format!("{} is already this version", c.path.display()),
        (false, true) => format!("{} removed", c.path.display()),
        (false, false) => format!("{} was not there", c.path.display()),
    };
    json_ok(json!({"path": c.path, "changed": c.changed, "installed": install, "message": message}))
}

/// `POST /api/web/opencode/add` and `/remove`: `{"model"}`, with `"profile"` for the model's
/// entry under that profile (`MODEL@PROFILE`). Adding one also installs the outcome plugin
/// unless `"plugin": false`.
pub(super) fn change(api: &Api, w: &WebState, req: &ApiRequest, b: &Value, add: bool) -> ApiResult<Output> {
    if !local(w, req) {
        return Err(ApiError::new(403, NOT_LOCAL).code("opencode_local_only"));
    }
    let model = str_field(b, &["model", "id"]).ok_or_else(|| ApiError::bad("opencode: no \"model\"").param("model"))?;
    let profile = match b.get("profile") {
        None | Some(Value::Null) => None,
        Some(Value::String(p)) if crate::server::learn::valid_profile(p) => Some(p.as_str()),
        Some(p) => {
            return Err(ApiError::bad(format!(
                "profile {p}: expected 1 to 64 of A-Z a-z 0-9 _ . - (not starting with a dot)"
            ))
            .param("profile"));
        }
    };
    let id = &match profile {
        Some(p) => crate::opencode::profile_id(model, p),
        None => model.to_string(),
    };
    let cfg = detected()?;
    let edit_err = |e: anyhow::Error| ApiError::new(409, format!("cannot edit opencode's configuration: {e:#}")).code("opencode_config");
    if !add {
        let c = cfg.remove(id).map_err(edit_err)?;
        if api.opts.log && c.changed {
            eprintln!("[opencode: removed {id} from {}]", c.path.display());
        }
        return json_ok(change_json(id, &c, false));
    }
    let s = api
        .models
        .status()
        .into_iter()
        .find(|s| s.spec.id == model)
        .ok_or_else(|| ApiError::new(404, format!("no model {model:?} is listed")).code("model_not_found"))?;
    if !is_chat(&s) {
        return Err(ApiError::bad(format!("{model} is not a chat model")).param("model"));
    }
    if profile.is_some() && !profiles_served(api) {
        return Err(ApiError::new(
            409,
            format!("{id}: this server does not serve profiles (learn_dir=none, or a non-loopback listener without an API key)"),
        )
        .code("profiles_off")
        .param("profile"));
    }
    let (m, _, source) = entry(api, &s, profile);
    let c = cfg.add(&provider(api, req), &m).map_err(edit_err)?;
    if api.opts.log && c.changed {
        eprintln!("[opencode: wrote {id} to {}]", c.path.display());
    }
    let mut v = change_json(id, &c, true);
    v["entry"] = m.to_json();
    v["source"] = json!(source);
    if let Some(p) = profile {
        v["profile"] = json!(p);
        let mut msg = format!(
            "{}. In opencode, pick {}/{id}: its requests are logged under profile {p}",
            v["message"].as_str().unwrap_or_default(),
            crate::opencode::PROVIDER
        );
        if b.get("plugin") != Some(&Value::Bool(false)) {
            let pc = cfg
                .install_plugin()
                .map_err(|e| ApiError::new(409, format!("{id} was written, but not opencode's plugin: {e:#}")).code("opencode_config"))?;
            if api.opts.log && pc.changed {
                eprintln!("[opencode: wrote {}]", pc.path.display());
            }
            msg.push_str(&format!(
                ", and {} reports its test, build and lint results{}",
                pc.path.display(),
                if pc.changed { " (written now: restart opencode)" } else { "" }
            ));
            v["plugin"] = json!({"path": pc.path, "changed": pc.changed});
        }
        v["message"] = json!(msg);
    }
    json_ok(v)
}
