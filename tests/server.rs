//! The OpenAI-compatible server over real HTTP on a random port. The routing and error checks
//! run by default; the ones with a model are ignored by default (they fetch or take from the
//! cache `Qwen/Qwen3-0.6B` and `BAAI/bge-small-en-v1.5`; override with `OXINFER_E2E_CHAT` /
//! `OXINFER_E2E_EMBED`):
//!
//! ```text
//! cargo test --release --test server -- --include-ignored --nocapture
//! ```
//! `OXINFER_E2E_OPTS` (`,draft=...`, `,lookup=1`) adds model options to the constrained-decoding
//! test, to run it through speculative decoding.
//! `tools/openai_check.py` checks every endpoint with the official `openai` Python client.

#![cfg(feature = "server")]

use oxinfer::server::{Api, ServerOptions, http};
use serde_json::{Value, json};
use std::io::{Read, Write};
use std::net::SocketAddr;

/// One HTTP/1.1 request; returns the status and the (de-chunked) body.
fn request(addr: SocketAddr, method: &str, path: &str, body: &str) -> (u16, String) {
    let mut s = std::net::TcpStream::connect(addr).unwrap();
    write!(
        s,
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut raw = Vec::new();
    s.read_to_end(&mut raw).unwrap();
    let text = String::from_utf8_lossy(&raw).into_owned();
    let (head, rest) = text.split_once("\r\n\r\n").unwrap();
    let status = head.split(' ').nth(1).unwrap().parse().unwrap();
    if !head.to_ascii_lowercase().contains("transfer-encoding: chunked") {
        return (status, rest.to_string());
    }
    let (mut out, mut rest) = (String::new(), rest);
    while let Some((size, after)) = rest.split_once("\r\n") {
        let n = usize::from_str_radix(size.trim(), 16).unwrap();
        if n == 0 {
            break;
        }
        out.push_str(&after[..n]);
        rest = &after[n + 2..];
    }
    (status, out)
}

fn serve(opts: &str) -> http::Running {
    let mut o = ServerOptions::from_pairs(opts).unwrap();
    o.set("listen", "127.0.0.1:0").unwrap();
    o.set("log", "0").unwrap();
    http::start(Api::new(o).unwrap()).unwrap()
}

#[test]
fn routing_and_errors() {
    let s = serve("model=m=/nonexistent/model; api_key=k");
    let (st, body) = request(s.addr(), "GET", "/v1/models", "");
    assert_eq!(st, 401, "{body}");
    let s2 = serve("model=m=/nonexistent/model");
    let (st, body) = request(s2.addr(), "GET", "/v1/models", "");
    assert_eq!(st, 200);
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["data"][0]["id"], "m");
    let chat = json!({"model": "x", "messages": [{"role": "user", "content": "hi"}]}).to_string();
    let (st, body) = request(s2.addr(), "POST", "/v1/chat/completions", &chat);
    assert_eq!(st, 404);
    assert_eq!(serde_json::from_str::<Value>(&body).unwrap()["error"]["code"], "model_not_found");
    let chat = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]}).to_string();
    let (st, body) = request(s2.addr(), "POST", "/v1/chat/completions", &chat);
    assert_eq!(st, 500, "{body}");
    let (st, _) = request(s2.addr(), "DELETE", "/v1/embeddings", "");
    assert_eq!(st, 405);
    // A schema outside the supported subset is refused before anything runs.
    let schema = json!({"type": "object", "properties": {"n": {"type": "integer", "multipleOf": 2}}});
    let chat = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "n", "strict": true, "schema": schema}}})
    .to_string();
    let (st, body) = request(s2.addr(), "POST", "/v1/chat/completions", &chat);
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!((st, &v["error"]["param"]), (400, &json!("response_format")), "{body}");
    assert!(
        v["error"]["message"].as_str().unwrap().contains("\"multipleOf\" is not supported"),
        "{body}"
    );
    let chat = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}], "top_logprobs": 2}).to_string();
    let (st, _) = request(s2.addr(), "POST", "/v1/chat/completions", &chat);
    assert_eq!(st, 400);
    s.stop().unwrap();
    s2.stop().unwrap();
}

/// One HTTP/1.1 request with its own header lines (`Host` included); the status and the body.
fn request_with(addr: SocketAddr, method: &str, path: &str, headers: &[&str], body: &str) -> (u16, String) {
    let mut s = std::net::TcpStream::connect(addr).unwrap();
    let mut head = format!("{method} {path} HTTP/1.1\r\n");
    for h in headers {
        head.push_str(h);
        head.push_str("\r\n");
    }
    write!(s, "{head}Content-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
    let mut raw = Vec::new();
    s.read_to_end(&mut raw).unwrap();
    let text = String::from_utf8_lossy(&raw).into_owned();
    let (head, rest) = text.split_once("\r\n\r\n").unwrap();
    (head.split(' ').nth(1).unwrap().parse().unwrap(), rest.to_string())
}

/// A temporary model cache: `org--model` (a configuration and a weight file), `org--gguf` (a
/// GGUF file), and a folder beside the cache that must never be touched.
fn temp_cache(tag: &str) -> (std::path::PathBuf, std::path::PathBuf) {
    let base = std::env::temp_dir().join(format!("oxinfer-webtest-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let cache = base.join("hub");
    let m = cache.join("org--model");
    std::fs::create_dir_all(&m).unwrap();
    std::fs::write(m.join("config.json"), r#"{"model_type": "bert"}"#).unwrap();
    std::fs::write(m.join("model.safetensors"), vec![0u8; 4096]).unwrap();
    std::fs::create_dir_all(cache.join("org--gguf")).unwrap();
    std::fs::write(cache.join("org--gguf/m-Q4_K_M.gguf"), b"GGUF").unwrap();
    let outside = base.join("outside");
    std::fs::create_dir_all(&outside).unwrap();
    std::fs::write(outside.join("keep.bin"), b"precious").unwrap();
    (cache, outside)
}

#[test]
fn web_gui_and_management_api() {
    let (cache, outside) = temp_cache("api");
    let s = serve(&format!("web=1; cache_dir={}", cache.display()));
    let a = s.addr();
    let local = ["Host: localhost", "Content-Type: application/json"];
    // The GUI (when this build embeds it) and the info endpoint need nothing.
    if oxinfer::server::web::has_gui() {
        let (st, body) = request_with(a, "GET", "/", &["Host: localhost"], "");
        assert_eq!(st, 200);
        assert!(body.contains("<div id=\"app\">"), "{body}");
    }
    // The catalog's presets, for the Image page and the catalog (none downloaded in this cache).
    let (st, body) = request_with(a, "GET", "/api/web/presets", &local, "");
    assert_eq!(st, 200, "{body}");
    let presets: Value = serde_json::from_str(&body).unwrap();
    let dmd2 = presets["data"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["id"] == "kromoss-dmd2")
        .expect("kromoss-dmd2");
    assert_eq!(
        (dmd2["base"].as_str(), dmd2["base_cached"].as_bool(), dmd2["listed"].as_bool()),
        (Some("lodestones/Kromoss"), Some(false), Some(false))
    );
    // (The avatar presets are not listed: the server has no route for that pipeline.)
    assert!(
        presets["data"]
            .as_array()
            .unwrap()
            .iter()
            .all(|p| p["pipeline"] != "longcat-avatar"),
        "{body}"
    );
    let (st, body) = request_with(a, "GET", "/api/web/info", &local, "");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(
        (st, &v["oxinfer"], &v["auth_required"], &v["admin"]),
        (200, &json!(true), &json!(false), &json!(true))
    );
    // The catalog: the tested models, with their kinds and what fits.
    let (st, body) = request_with(a, "GET", "/api/web/catalog", &local, "");
    assert_eq!(st, 200);
    let v: Value = serde_json::from_str(&body).unwrap();
    let models = v["models"].as_array().unwrap();
    assert!(models.len() > 20);
    assert!(models.iter().any(|m| m["kind"] == "chat" && m["server_kind"] == "chat"));
    assert!(
        models
            .iter()
            .any(|m| m["kind"] == "speech-to-text" && m["server_kind"] == "transcription")
    );
    assert!(
        models.iter().all(|m| m["cached"] == false),
        "nothing of the catalog is in this cache"
    );
    // Which accelerator each variant uses, and the machine's accelerators.
    assert!(models.iter().all(|m| {
        m["accelerator"].is_string()
            && m["variants"]
                .as_array()
                .unwrap()
                .iter()
                .all(|v| v["accelerator"].is_string() && v["accel"]["decode"].is_string())
    }));
    assert_eq!(v["hardware"]["accelerators"][0]["backend"], "cpu");
    // The status lists them too: the CPU first, then the GPUs (or a line saying there is none).
    let (st, body) = request_with(a, "GET", "/api/web/status", &local, "");
    assert_eq!(st, 200);
    let v: Value = serde_json::from_str(&body).unwrap();
    let acc = v["accelerators"].as_array().unwrap();
    assert_eq!((&acc[0]["backend"], &acc[0]["used"]), (&json!("cpu"), &json!(true)));
    assert!(acc.len() >= 2 && acc.iter().all(|x| x["status"].is_string() && x["prefill"].is_boolean()));
    // The cache: both folders, sizes, kinds and the specs that load them.
    let (st, body) = request_with(a, "GET", "/api/web/cache", &local, "");
    assert_eq!(st, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    let entries = v["entries"].as_array().unwrap();
    assert_eq!(entries.len(), 2, "{body}");
    let m = entries.iter().find(|e| e["name"] == "org--model").unwrap();
    assert_eq!(
        (&m["repo"], &m["bytes"], &m["loaded"]),
        (&json!("org/model"), &json!(4096 + 22), &json!(false))
    );
    let g = entries.iter().find(|e| e["name"] == "org--gguf").unwrap();
    assert_eq!(g["specs"], json!(["org/gguf:Q4_K_M"]));
    let (st, body) = request_with(a, "GET", "/api/web/disk", &local, "");
    assert_eq!(st, 200, "{body}");
    // Deletion: names that leave the cache are refused, and nothing outside is touched.
    for bad in ["../outside", "..", "/", "org--model/config.json", "a/b/c", ".hidden"] {
        let (st, body) = request_with(a, "POST", "/api/web/cache/delete", &local, &json!({"name": bad}).to_string());
        assert!(st == 400 || st == 404, "{bad:?}: {st} {body}");
    }
    let (st, _) = request_with(
        a,
        "POST",
        "/api/web/cache/delete",
        &local,
        &json!({"name": "org--missing"}).to_string(),
    );
    assert_eq!(st, 404);
    assert!(outside.join("keep.bin").exists() && cache.join("org--model").exists());
    // Cross-site requests: another origin, no JSON content type, a rebinding Host.
    let del = json!({"name": "org/model"}).to_string();
    let (st, _) = request_with(
        a,
        "POST",
        "/api/web/cache/delete",
        &["Host: localhost", "Content-Type: application/json", "Origin: http://evil.example"],
        &del,
    );
    assert_eq!(st, 403);
    let (st, _) = request_with(
        a,
        "POST",
        "/api/web/cache/delete",
        &["Host: localhost", "Content-Type: text/plain"],
        &del,
    );
    assert_eq!(st, 415);
    let (st, _) = request_with(
        a,
        "POST",
        "/api/web/cache/delete",
        &["Host: evil.example", "Content-Type: application/json"],
        &del,
    );
    assert_eq!(st, 403);
    assert!(cache.join("org--model").exists());
    // The page's own origin may.
    let own = "Origin: http://localhost";
    let (st, body) = request_with(
        a,
        "POST",
        "/api/web/cache/delete",
        &["Host: localhost", "Content-Type: application/json", own],
        &del,
    );
    assert_eq!(st, 200, "{body}");
    assert_eq!(serde_json::from_str::<Value>(&body).unwrap()["freed"], json!(4096 + 22));
    assert!(!cache.join("org--model").exists() && cache.join("org--gguf").exists() && outside.join("keep.bin").exists());
    // A model added from the GUI joins /v1/models; a load of a missing path fails, reported.
    let load = json!({"model": "/nonexistent/oxinfer-model", "id": "x"}).to_string();
    let (st, body) = request_with(a, "POST", "/api/web/models/load", &local, &load);
    assert_eq!(st, 200, "{body}");
    let (_, body) = request_with(a, "GET", "/v1/models", &local, "");
    assert_eq!(serde_json::from_str::<Value>(&body).unwrap()["data"][0]["id"], "x");
    let mut state = Value::Null;
    for _ in 0..100 {
        let (_, body) = request_with(a, "GET", "/api/web/models", &local, "");
        state = serde_json::from_str::<Value>(&body).unwrap()["data"][0].clone();
        if state["state"] == "error" {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    assert_eq!(state["state"], "error", "{state}");
    // Downloads of a local path, and unknown jobs.
    let (st, _) = request_with(
        a,
        "POST",
        "/api/web/downloads",
        &local,
        &json!({"model": outside.display().to_string()}).to_string(),
    );
    assert_eq!(st, 400);
    let (st, _) = request_with(a, "GET", "/api/web/downloads/nope", &local, "");
    assert_eq!(st, 404);
    let (st, _) = request_with(a, "GET", "/api/web/nothing", &local, "");
    assert_eq!(st, 404);
    s.stop().unwrap();
    std::fs::remove_dir_all(cache.parent().unwrap()).ok();
}

#[test]
fn web_auth_and_read_only() {
    let (cache, _) = temp_cache("auth");
    let s = serve(&format!("web=1; api_key=k; cache_dir={}", cache.display()));
    let a = s.addr();
    let (st, body) = request_with(a, "GET", "/api/web/info", &["Host: 127.0.0.1"], "");
    assert_eq!(st, 200);
    assert_eq!(serde_json::from_str::<Value>(&body).unwrap()["auth_required"], true);
    for p in [
        "/api/web/status",
        "/api/web/catalog",
        "/api/web/cache",
        "/api/web/models",
        "/v1/models",
    ] {
        let (st, _) = request_with(a, "GET", p, &["Host: 127.0.0.1"], "");
        assert_eq!(st, 401, "{p}");
        let (st, _) = request_with(a, "GET", p, &["Host: 127.0.0.1", "Authorization: Bearer k"], "");
        assert_eq!(st, 200, "{p}");
    }
    let (st, _) = request_with(
        a,
        "POST",
        "/api/web/cache/delete",
        &["Host: 127.0.0.1", "Content-Type: application/json"],
        r#"{"name": "org--gguf"}"#,
    );
    assert_eq!(st, 401);
    assert!(cache.join("org--gguf").exists());
    if oxinfer::server::web::has_gui() {
        let (st, _) = request_with(a, "GET", "/", &["Host: 127.0.0.1"], "");
        assert_eq!(st, 200, "the page itself needs no key (it asks for it)");
    }
    s.stop().unwrap();
    // Without the web option nothing of it is served.
    let s = serve("model=m=/nonexistent/model");
    let (st, _) = request_with(s.addr(), "GET", "/api/web/info", &["Host: localhost"], "");
    assert_eq!(st, 404);
    s.stop().unwrap();
    // A non-loopback listener without a key: the management API reads, but changes nothing
    // (through the API itself: no socket is bound).
    let mut o = ServerOptions::from_pairs(&format!("web=1; log=0; listen=0.0.0.0:0; cache_dir={}", cache.display())).unwrap();
    o.set("workers", "2").unwrap();
    let api = Api::new(o).unwrap();
    assert!(api.web_read_only());
    let mut req = oxinfer::server::ApiRequest::new("GET", "/api/web/cache", None, vec![]);
    assert_eq!(api.handle(&req).collect().status, 200);
    req = oxinfer::server::ApiRequest::post_json("/api/web/cache/delete", r#"{"name": "org--gguf"}"#);
    let r = api.handle(&req).collect();
    assert_eq!(r.status, 403);
    assert_eq!(serde_json::from_slice::<Value>(&r.body).unwrap()["error"]["code"], "read_only");
    req = oxinfer::server::ApiRequest::post_json("/api/web/downloads", r#"{"model": "org/x"}"#);
    assert_eq!(api.handle(&req).collect().status, 403);
    assert!(cache.join("org--gguf").exists());
    std::fs::remove_dir_all(cache.parent().unwrap()).ok();
}

/// A release feed (GitHub's release JSON) at `dir/name`: version `tag`, with an archive for this
/// platform at `asset_url` when given.
#[cfg(feature = "update")]
fn release_feed(dir: &std::path::Path, name: &str, tag: &str, asset_url: Option<&str>) -> String {
    let assets: Vec<Value> = match (asset_url, oxinfer::update::platform_suffix()) {
        (Some(u), Some(s)) => vec![json!({"name": format!("oxinfer-{tag}-{s}.tar.gz"), "browser_download_url": u, "size": 1000})],
        _ => vec![],
    };
    let path = dir.join(name);
    let v = json!({"tag_name": tag, "html_url": format!("https://github.com/KarpelesLab/oxinfer/releases/tag/{tag}"), "assets": assets});
    std::fs::write(&path, v.to_string()).unwrap();
    path.display().to_string()
}

/// `GET /api/web/update` of `api` until its check (or its update job) is done.
#[cfg(feature = "update")]
fn update_state(api: &std::sync::Arc<Api>, job: bool) -> Value {
    let req = oxinfer::server::ApiRequest::new("GET", "/api/web/update", None, vec![]);
    for _ in 0..200 {
        let r = api.handle(&req).collect();
        assert_eq!(r.status, 200);
        let v: Value = serde_json::from_slice(&r.body).unwrap();
        let running = if job {
            ["checking", "downloading", "restarting"].contains(&v["job"]["state"].as_str().unwrap_or(""))
        } else {
            v["checking"] == true
        };
        if !running {
            return v;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    panic!("the update check did not end");
}

#[test]
#[cfg(feature = "update")]
fn web_update_check_and_update() {
    let (cache, _) = temp_cache("update");
    let dir = cache.parent().unwrap().to_path_buf();
    // Nothing listens on port 9: the download fails, so the test binary is never replaced.
    let newer = release_feed(&dir, "newer.json", "v999.0.0", Some("http://127.0.0.1:9/oxinfer.tar.gz"));
    let older = release_feed(&dir, "older.json", "v0.0.1", None);
    let api_with = |extra: &str| {
        let o = ServerOptions::from_pairs(&format!("web=1; log=0; cache_dir={}; {extra}", cache.display())).unwrap();
        Api::new(o).unwrap()
    };
    let post = || oxinfer::server::ApiRequest::post_json("/api/web/update", "{}");
    let error = |r: &oxinfer::server::ApiResponse| serde_json::from_slice::<Value>(&r.body).unwrap()["error"].clone();

    // Over HTTP: the first answer is "checking", then the release from the feed.
    let s = serve(&format!("web=1; cache_dir={}; update_feed={newer}", cache.display()));
    let local = ["Host: localhost", "Content-Type: application/json"];
    let (st, body) = request_with(s.addr(), "GET", "/api/web/update", &local, "");
    assert_eq!(st, 200);
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["current"], env!("CARGO_PKG_VERSION"));
    assert!(v["state"] == "checking" || v["state"] == "available", "{v}");
    let mut v = v;
    for _ in 0..200 {
        if v["state"] != "checking" {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
        v = serde_json::from_str(&request_with(s.addr(), "GET", "/api/web/update", &local, "").1).unwrap();
    }
    assert_eq!(v["state"], "available", "{v}");
    assert_eq!((v["latest"].as_str(), v["newer"].as_bool()), (Some("999.0.0"), Some(true)));
    assert_eq!(v["url"], "https://github.com/KarpelesLab/oxinfer/releases/tag/v999.0.0");
    assert!(v["checked"].as_u64().is_some());
    // Not the oxinfer program (a library user, as here): the GUI shows the command instead.
    assert_eq!(v["can_update"], false);
    assert!(v["why_not"].as_str().unwrap().contains("another program"), "{v}");
    assert_eq!(v["command"], "oxinfer --update");
    let (st, body) = request_with(s.addr(), "POST", "/api/web/update", &local, "{}");
    assert_eq!(st, 409, "{body}");
    assert!(body.contains("cannot_update"), "{body}");
    s.stop().unwrap();

    // Not newer.
    let api = api_with(&format!("update_feed={older}"));
    let v = update_state(&api, false);
    assert_eq!(
        (v["state"].as_str(), v["newer"].as_bool(), v["latest"].as_str()),
        (Some("up_to_date"), Some(false), Some("0.0.1")),
        "{v}"
    );
    assert!(v["why_not"].is_null() && v["can_update"] == false, "{v}");
    // A failed check.
    let api = api_with(&format!("update_feed={}", dir.join("missing.json").display()));
    let v = update_state(&api, false);
    assert_eq!(v["state"], "error", "{v}");
    assert!(v["error"].as_str().unwrap().contains("release feed"), "{v}");
    assert!(v["latest"].is_null());

    // Turned off: no check (the feed would fail) and no update.
    let api = api_with(&format!("update_check=0; update_feed={}", dir.join("missing.json").display()));
    let v = update_state(&api, false);
    assert_eq!(
        (v["enabled"].as_bool(), v["state"].as_str()),
        (Some(false), Some("disabled")),
        "{v}"
    );
    assert!(v["error"].is_null() && v["latest"].is_null());
    oxinfer::server::web::allow_self_update(&api, None);
    let r = api.handle(&post()).collect();
    assert_eq!((r.status, error(&r)["code"].as_str()), (403, Some("update_check_off")));

    // Read-only (a non-loopback address without a key): refused, whatever the program.
    let api = api_with(&format!("listen=0.0.0.0:0; update_feed={newer}"));
    oxinfer::server::web::allow_self_update(&api, None);
    let v = update_state(&api, false);
    assert_eq!(v["can_update"], false);
    assert!(v["why_not"].as_str().unwrap().contains("read-only"), "{v}");
    let r = api.handle(&post()).collect();
    assert_eq!((r.status, error(&r)["code"].as_str()), (403, Some("read_only")));

    // The oxinfer program (allowed, no restart hook): the update job runs and fails to download.
    let api = api_with(&format!("update_feed={newer}"));
    oxinfer::server::web::allow_self_update(&api, None);
    let v = update_state(&api, false);
    assert_eq!(v["can_update"], oxinfer::update::platform_suffix().is_some(), "{v}");
    let r = api.handle(&post()).collect();
    assert_eq!(r.status, 200);
    let job: Value = serde_json::from_slice(&r.body).unwrap();
    assert!(
        ["checking", "downloading", "error"].contains(&job["state"].as_str().unwrap()),
        "{job}"
    );
    let v = update_state(&api, true);
    assert_eq!(v["job"]["state"], "error", "{v}");
    let exe = std::env::current_exe().unwrap();
    assert!(exe.is_file(), "the test binary is still there");
    // Not newer: the job says so, nothing is downloaded.
    let api = api_with(&format!("update_feed={older}"));
    oxinfer::server::web::allow_self_update(&api, None);
    assert_eq!(api.handle(&post()).collect().status, 200);
    let v = update_state(&api, true);
    assert_eq!(v["job"]["state"], "up_to_date", "{v}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn anthropic_messages_routing_and_errors() {
    let s = serve("model=m=/nonexistent/model; api_key=k");
    let a = s.addr();
    let msg = json!({"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]}).to_string();
    // Anthropic's clients send the key as x-api-key (or as a bearer token).
    let (st, body) = request_with(
        a,
        "POST",
        "/v1/messages",
        &["Host: localhost", "Content-Type: application/json"],
        &msg,
    );
    assert_eq!(st, 401, "{body}");
    // In Anthropic's error shape, as every error of these endpoints.
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(
        (&v["type"], &v["error"]["type"]),
        (&json!("error"), &json!("authentication_error")),
        "{body}"
    );
    let ok_key = [
        "Host: localhost",
        "Content-Type: application/json",
        "x-api-key: k",
        "anthropic-version: 2023-06-01",
    ];
    let (st, body) = request_with(a, "POST", "/v1/messages?beta=true", &ok_key, &msg);
    // The model's files are missing: a load failure, in Anthropic's error shape.
    assert_eq!(st, 500, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!((&v["type"], &v["error"]["type"]), (&json!("error"), &json!("api_error")), "{body}");
    let other = json!({"model": "x", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]}).to_string();
    let (st, body) = request_with(a, "POST", "/v1/messages", &ok_key, &other);
    assert_eq!(st, 404);
    assert_eq!(serde_json::from_str::<Value>(&body).unwrap()["error"]["type"], "not_found_error");
    let no_max = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]}).to_string();
    let (st, body) = request_with(a, "POST", "/v1/messages", &ok_key, &no_max);
    assert_eq!(st, 400);
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap()["error"]["type"],
        "invalid_request_error"
    );
    let (st, _) = request_with(a, "GET", "/v1/messages", &ok_key, "");
    assert_eq!(st, 405);
    let (st, body) = request_with(a, "POST", "/v1/messages/count_tokens", &ok_key, &other);
    assert_eq!(st, 404, "{body}");
    let (st, body) = request_with(a, "POST", "/v1/messages", &ok_key, "{not json");
    assert_eq!(st, 400, "{body}");
    assert_eq!(serde_json::from_str::<Value>(&body).unwrap()["type"], "error", "{body}");
    // What Claude Code sends besides the conversation: beta headers, cache-control marks,
    // metadata, thinking blocks of earlier turns (signed by another server here, or redacted),
    // interleaved with tool calls and results. All accepted: the request gets as far as the load.
    let claude_code = json!({
        "model": "m", "max_tokens": 10, "stream": true, "metadata": {"user_id": "session"},
        "thinking": {"type": "enabled", "budget_tokens": 4000},
        "system": [{"type": "text", "text": "You are a coding agent.", "cache_control": {"type": "ephemeral"}}],
        "tools": [{"name": "Bash", "description": "Runs a command", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}},
                   "cache_control": {"type": "ephemeral"}}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "List the files", "cache_control": {"type": "ephemeral"}}]},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "I should run ls.", "signature": "AaGVsbG8gd29ybGQgdGhpcyBpcyBub3QgYSBzaWduYXR1cmU="},
                {"type": "redacted_thinking", "data": "EmwKAhgBEgy3va3pzix"},
                {"type": "tool_use", "id": "toolu_01", "name": "Bash", "input": {"command": "ls"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_01", "content": "a.txt", "cache_control": {"type": "ephemeral"}}]}
        ]
    })
    .to_string();
    let mut beta = ok_key.to_vec();
    beta.push("anthropic-beta: interleaved-thinking-2025-05-14,prompt-caching-2024-07-31,fine-grained-tool-streaming-2025-05-14");
    let (st, body) = request_with(a, "POST", "/v1/messages?beta=true", &beta, &claude_code);
    // Streamed: the stream starts, and the load failure comes as its error event.
    assert_eq!(st, 200, "{body}");
    assert!(body.contains("event: message_start") && body.contains("event: error"), "{body}");
    assert!(sse_events(&body).iter().any(|e| e["error"]["type"] == "api_error"), "{body}");
    s.stop().unwrap();
    // The signing key is a server option.
    let mut o = ServerOptions::default();
    o.set("thinking_key", "secret").unwrap();
    assert_eq!(o.thinking_key.as_deref(), Some("secret"));
    let o = ServerOptions::from_json(&json!({"thinking_key": "k2"})).unwrap();
    assert_eq!(o.thinking_key.as_deref(), Some("k2"));
}

#[test]
#[ignore = "fetches Qwen/Qwen3-0.6B"]
fn anthropic_messages_with_a_model() {
    let model = env("OXINFER_E2E_CHAT", "Qwen/Qwen3-0.6B");
    let s = serve(&format!("model=m={model}"));
    let a = s.addr();
    let h = ["Host: localhost", "Content-Type: application/json"];
    let tools = json!([{"name": "get_weather", "description": "The weather in a city", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]);
    let body = json!({"model": "m", "max_tokens": 200, "temperature": 0, "tools": tools, "tool_choice": {"type": "any"},
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}]});
    let (st, out) = request_with(a, "POST", "/v1/messages", &h, &body.to_string());
    assert_eq!(st, 200, "{out}");
    let v: Value = serde_json::from_str(&out).unwrap();
    assert_eq!(v["stop_reason"], "tool_use", "{out}");
    let call = v["content"]
        .as_array()
        .unwrap()
        .iter()
        .find(|b| b["type"] == "tool_use")
        .unwrap()
        .clone();
    assert_eq!(call["name"], "get_weather");
    // The result back, streamed this time.
    let body = json!({"model": "m", "max_tokens": 100, "temperature": 0, "stream": true, "tools": tools,
        "messages": [{"role": "user", "content": "What is the weather in Paris?"},
            {"role": "assistant", "content": [call.clone()]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call["id"], "content": "Sunny, 21 C"}]}]});
    let (st, out) = request(a, "POST", "/v1/messages", &body.to_string());
    assert_eq!(st, 200);
    for e in [
        "event: message_start",
        "event: content_block_start",
        "event: content_block_delta",
        "event: message_delta",
        "event: message_stop",
    ] {
        assert!(out.contains(e), "{e} missing: {out}");
    }
    let text: String = out
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter_map(|d| serde_json::from_str::<Value>(d).ok())
        .filter_map(|v| v["delta"]["text"].as_str().map(str::to_string))
        .collect();
    assert!(text.to_lowercase().contains("sunny") || text.contains("21"), "{text}");
    let (st, out) = request(
        a,
        "POST",
        "/v1/messages/count_tokens",
        &json!({"model": "m", "messages": [{"role": "user", "content": "Hello there"}]}).to_string(),
    );
    assert_eq!(st, 200, "{out}");
    assert!(serde_json::from_str::<Value>(&out).unwrap()["input_tokens"].as_u64().unwrap() > 2);
    s.stop().unwrap();
}

/// The events of an Anthropic stream, as JSON.
fn sse_events(out: &str) -> Vec<Value> {
    out.lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter_map(|d| serde_json::from_str::<Value>(d).ok())
        .collect()
}

#[test]
#[ignore = "fetches Qwen/Qwen3-0.6B"]
fn anthropic_thinking_signatures() {
    let model = env("OXINFER_E2E_CHAT", "Qwen/Qwen3-0.6B");
    let s = serve(&format!("model=m={model}; thinking_key=shared"));
    let a = s.addr();
    let tools = json!([{"name": "get_weather", "description": "The weather in a city", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]);
    let q = json!({"role": "user", "content": "What is the weather in Paris? Use the tool."});
    let body = json!({"model": "m", "max_tokens": 600, "temperature": 0, "tools": tools, "thinking": {"type": "enabled", "budget_tokens": 500},
        "messages": [q]});
    let (st, out) = request(a, "POST", "/v1/messages", &body.to_string());
    assert_eq!(st, 200, "{out}");
    let v: Value = serde_json::from_str(&out).unwrap();
    let content = v["content"].as_array().unwrap().clone();
    let thinking = content.iter().find(|b| b["type"] == "thinking").expect("a thinking block").clone();
    assert!(!thinking["thinking"].as_str().unwrap().is_empty());
    let sig = thinking["signature"].as_str().unwrap().to_string();
    assert!(!sig.is_empty(), "{out}");
    assert_eq!(v["stop_reason"], "tool_use", "{out}");
    let call = content.iter().find(|b| b["type"] == "tool_use").unwrap().clone();

    // Sent back (thinking, then the call) with the tool's result: the tool loop goes on.
    let turn = |thinking: &Value| {
        json!({"model": "m", "max_tokens": 300, "temperature": 0, "tools": tools, "thinking": {"type": "enabled", "budget_tokens": 200},
            "messages": [q, {"role": "assistant", "content": [thinking, call]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call["id"], "content": "Sunny, 21 C"}]}]})
    };
    let mut tampered = thinking.clone();
    tampered["signature"] = json!(format!("{}x", &sig[..sig.len() - 1]));
    // count_tokens counts the reasoning of the current tool loop when its signature verifies.
    let count = |b: &Value| {
        let (st, out) = request(a, "POST", "/v1/messages/count_tokens", &b.to_string());
        assert_eq!(st, 200, "{out}");
        serde_json::from_str::<Value>(&out).unwrap()["input_tokens"].as_u64().unwrap()
    };
    assert!(count(&turn(&thinking)) > count(&turn(&tampered)));
    // Another server with the same key verifies it too; one with its own random key does not.
    let s2 = serve(&format!("model=m={model}; thinking_key=shared"));
    let (st, out) = request(s2.addr(), "POST", "/v1/messages/count_tokens", &turn(&thinking).to_string());
    assert_eq!(
        (st, serde_json::from_str::<Value>(&out).unwrap()["input_tokens"].as_u64().unwrap()),
        (200, count(&turn(&thinking)))
    );
    s2.stop().unwrap();
    let s3 = serve(&format!("model=m={model}"));
    let (st, out) = request(s3.addr(), "POST", "/v1/messages/count_tokens", &turn(&thinking).to_string());
    assert_eq!(
        (st, serde_json::from_str::<Value>(&out).unwrap()["input_tokens"].as_u64().unwrap()),
        (200, count(&turn(&tampered)))
    );
    s3.stop().unwrap();

    // Streamed: the thinking block ends with a signature_delta that verifies its deltas.
    let mut b = turn(&tampered);
    b["stream"] = json!(true);
    let (st, out) = request(a, "POST", "/v1/messages", &b.to_string());
    assert_eq!(st, 200, "{out}");
    let ev = sse_events(&out);
    let text: String = ev.iter().filter_map(|e| e["delta"]["thinking"].as_str()).collect();
    let sigs: Vec<&str> = ev.iter().filter_map(|e| e["delta"]["signature"].as_str()).collect();
    assert_eq!(sigs.len(), 1, "{out}");
    let i = ev.iter().position(|e| e["delta"]["type"] == "signature_delta").unwrap();
    assert_eq!(ev[i + 1]["type"], "content_block_stop", "the signature closes the block");
    let answer: String = ev.iter().filter_map(|e| e["delta"]["text"].as_str()).collect();
    assert!(answer.to_lowercase().contains("sunny") || answer.contains("21"), "{answer}");
    // That signature verifies: the streamed block counts in a next tool loop's tokens.
    let streamed = json!({"type": "thinking", "thinking": text, "signature": sigs[0]});
    let bad = json!({"type": "thinking", "thinking": text, "signature": sig});
    assert!(count(&turn(&streamed)) > count(&turn(&bad)));
    s.stop().unwrap();
}

fn env(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

#[test]
#[ignore = "fetches Qwen/Qwen3-0.6B and BAAI/bge-small-en-v1.5"]
fn chat_and_embeddings() {
    let chat = env("OXINFER_E2E_CHAT", "Qwen/Qwen3-0.6B");
    let embed = env("OXINFER_E2E_EMBED", "BAAI/bge-small-en-v1.5");
    let s = serve(&format!("model=chat={chat}; model=embed={embed}"));
    let ask = |messages: Value, extra: Value| {
        let mut req = json!({"model": "chat", "messages": messages, "temperature": 0, "max_tokens": 30});
        for (k, v) in extra.as_object().unwrap() {
            req[k] = v.clone();
        }
        request(s.addr(), "POST", "/v1/chat/completions", &req.to_string())
    };

    // A reply, then the same conversation extended: its prefix comes from the cache.
    let first = json!([{"role": "user", "content": "My name is Ada. Reply with just OK."}]);
    let (st, body) = ask(first.clone(), json!({}));
    assert_eq!(st, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    let reply = v["choices"][0]["message"]["content"].as_str().unwrap().to_string();
    let mut convo = first.as_array().unwrap().clone();
    convo.push(json!({"role": "assistant", "content": reply}));
    convo.push(json!({"role": "user", "content": "What is my name?"}));
    let (_, body) = ask(Value::Array(convo), json!({}));
    let v: Value = serde_json::from_str(&body).unwrap();
    assert!(v["choices"][0]["message"]["content"].as_str().unwrap().contains("Ada"), "{body}");
    assert!(v["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap() > 0, "{body}");

    // Streamed: server-sent events ending with [DONE], the text in the deltas.
    let (st, body) = ask(json!([{"role": "user", "content": "Count from 1 to 5."}]), json!({"stream": true}));
    assert_eq!(st, 200);
    assert!(body.trim_end().ends_with("data: [DONE]"), "{body}");
    let text: String = body
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter_map(|d| serde_json::from_str::<Value>(d).ok())
        .filter_map(|c| c["choices"][0]["delta"]["content"].as_str().map(str::to_string))
        .collect();
    assert!(text.contains('3'), "{text}");

    // A tool call.
    let tools = json!([{"type": "function", "function": {"name": "get_weather", "description": "Current weather of a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]);
    let (_, body) = ask(
        json!([{"role": "user", "content": "What is the weather in Paris?"}]),
        json!({"tools": tools, "max_tokens": 200}),
    );
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls", "{body}");
    assert_eq!(v["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "get_weather");

    // Embeddings, and a chat request to an embedding model.
    let req = json!({"model": "embed", "input": ["a cat", "a dog"]}).to_string();
    let (st, body) = request(s.addr(), "POST", "/v1/embeddings", &req);
    assert_eq!(st, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["data"].as_array().unwrap().len(), 2);
    let req = json!({"model": "embed", "messages": [{"role": "user", "content": "hi"}]}).to_string();
    let (st, _) = request(s.addr(), "POST", "/v1/chat/completions", &req);
    assert_eq!(st, 400);
    s.stop().unwrap();
}

/// Whether `v` is a value of `schema` (the keywords the tests use).
fn valid(schema: &Value, v: &Value) -> bool {
    if let Some(e) = schema.get("enum") {
        return e.as_array().unwrap().contains(v);
    }
    if let Some(a) = schema.get("anyOf") {
        return a.as_array().unwrap().iter().any(|s| valid(s, v));
    }
    match schema["type"].as_str().unwrap() {
        "object" => {
            let (Some(o), Some(props)) = (v.as_object(), schema["properties"].as_object()) else {
                return false;
            };
            let required = schema["required"].as_array().unwrap();
            required.iter().all(|k| o.contains_key(k.as_str().unwrap())) && o.iter().all(|(k, x)| props.get(k).is_some_and(|s| valid(s, x)))
        }
        "array" => v.as_array().is_some_and(|a| a.iter().all(|x| valid(&schema["items"], x))),
        "string" => v.is_string(),
        "integer" => v.is_i64() || v.is_u64(),
        "number" => v.is_number(),
        "boolean" => v.is_boolean(),
        "null" => v.is_null(),
        t => panic!("type {t}"),
    }
}

/// Constrained decoding and log probabilities on a real model. `OXINFER_E2E_OPTS` adds model
/// options (`draft=...`, `lookup=1`: the constraint then runs through speculative decoding).
#[test]
#[ignore = "fetches Qwen/Qwen3-0.6B"]
fn structured_outputs_forced_calls_and_logprobs() {
    let chat = env("OXINFER_E2E_CHAT", "Qwen/Qwen3-0.6B");
    let s = serve(&format!("model=chat={chat}{}", env("OXINFER_E2E_OPTS", "")));
    let ask = |messages: Value, extra: Value| {
        let mut req = json!({"model": "chat", "messages": messages, "temperature": 0, "max_tokens": 300});
        for (k, v) in extra.as_object().unwrap() {
            req[k] = v.clone();
        }
        let (st, body) = request(s.addr(), "POST", "/v1/chat/completions", &req.to_string());
        assert_eq!(st, 200, "{body}");
        body
    };
    let user = |t: &str| json!([{"role": "user", "content": t}]);
    let content = |body: &str| -> (String, Value) {
        let v: Value = serde_json::from_str(body).unwrap();
        (v["choices"][0]["message"]["content"].as_str().unwrap_or_default().to_string(), v)
    };

    // A schema with nesting, an enum, a union and an array: the reply always parses and
    // validates, greedy or sampled, though the prompts never describe the schema.
    let schema = json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "population": {"type": "integer"},
            "continent": {"type": "string", "enum": ["Africa", "America", "Asia", "Europe", "Oceania"]},
            "landmarks": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "year": {"anyOf": [{"type": "integer"}, {"type": "null"}]}},
                          "required": ["name", "year"], "additionalProperties": false}},
            "capital": {"type": "boolean"},
        },
        "required": ["name", "population", "continent", "landmarks", "capital"],
        "additionalProperties": false,
    });
    let format = json!({"type": "json_schema", "json_schema": {"name": "city", "strict": true, "schema": schema}});
    let prompts = [
        "Tell me about Paris.",
        "Describe Tokyo.",
        "What is 2 + 2?",
        "Écris un poème sur Kyoto (京都) 🌸.",
    ];
    let (mut complete, mut calls) = (0, 0);
    for (i, p) in prompts.iter().enumerate() {
        for temperature in [0.0, 1.0] {
            let body = ask(user(p), json!({"response_format": format, "temperature": temperature, "seed": i}));
            let (text, v) = content(&body);
            if v["choices"][0]["finish_reason"] == "length" {
                continue;
            }
            complete += 1;
            let doc: Value = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{e}: {text}"));
            assert!(valid(&schema, &doc), "{text}");
            let keys: Vec<&String> = doc.as_object().unwrap().keys().collect();
            assert_eq!(keys, ["name", "population", "continent", "landmarks", "capital"]);
        }
    }
    // Any JSON object.
    let (text, _) = content(&ask(user("Say hello."), json!({"response_format": {"type": "json_object"}})));
    assert!(serde_json::from_str::<Value>(&text).unwrap().is_object(), "{text}");
    // Streamed, the pieces add up to the document.
    let body = ask(user("Tell me about Rome."), json!({"response_format": format, "stream": true}));
    let chunks: Vec<Value> = body
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter_map(|d| serde_json::from_str(d).ok())
        .collect();
    let text: String = chunks.iter().filter_map(|c| c["choices"][0]["delta"]["content"].as_str()).collect();
    assert!(valid(&schema, &serde_json::from_str(&text).unwrap()), "{text}");
    // Cut by the token limit: a start of a document, reported as such.
    let (text, v) = content(&ask(
        user("Tell me about Paris."),
        json!({"response_format": format, "max_tokens": 8}),
    ));
    assert_eq!(v["choices"][0]["finish_reason"], "length", "{text}");
    assert!(text.starts_with('{') && serde_json::from_str::<Value>(&text).is_err(), "{text}");
    // With thinking: the reasoning is free text, the answer follows the schema.
    let body = ask(
        user("What is 12 * 13? Tell me about the result."),
        json!({"response_format": format, "reasoning_effort": "high", "max_tokens": 1500}),
    );
    let (text, v) = content(&body);
    if v["choices"][0]["finish_reason"] != "length" {
        assert!(
            v["choices"][0]["message"]["reasoning_content"]
                .as_str()
                .is_some_and(|r| !r.is_empty()),
            "{body}"
        );
        assert!(valid(&schema, &serde_json::from_str(text.trim_start()).unwrap()), "{text}");
    }

    // Forced calls: a question that needs no tool still calls one, with arguments of its schema.
    let weather = json!({"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        "required": ["city", "days", "unit"], "additionalProperties": false});
    let add = json!({"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]});
    let tools = json!([
        {"type": "function", "function": {"name": "get_weather", "description": "Weather forecast of a city", "parameters": weather}},
        {"type": "function", "function": {"name": "add", "description": "Adds two numbers", "parameters": add}},
    ]);
    let call = |body: &str| -> (String, Value) {
        let v: Value = serde_json::from_str(body).unwrap();
        let c = &v["choices"][0];
        assert_eq!(c["finish_reason"], "tool_calls", "{body}");
        assert_eq!(c["message"]["tool_calls"].as_array().unwrap().len(), 1, "{body}");
        let f = &c["message"]["tool_calls"][0]["function"];
        (
            f["name"].as_str().unwrap().to_string(),
            serde_json::from_str(f["arguments"].as_str().unwrap()).unwrap(),
        )
    };
    for (i, p) in ["Hello!", "What is 17.5 plus 4?", "Will it rain in Lisbon this week?"]
        .iter()
        .enumerate()
    {
        for temperature in [0.0, 1.0] {
            let (name, args) = call(&ask(
                user(p),
                json!({"tools": tools, "tool_choice": "required", "temperature": temperature, "seed": i}),
            ));
            let schema = if name == "add" { &add } else { &weather };
            assert!((name == "add" || name == "get_weather") && valid(schema, &args), "{name}({args})");
            let named = json!({"type": "function", "function": {"name": "get_weather"}});
            let (name, args) = call(&ask(
                user(p),
                json!({"tools": tools, "tool_choice": named, "temperature": temperature, "seed": i}),
            ));
            assert!(name == "get_weather" && valid(&weather, &args), "{name}({args})");
            calls += 2;
        }
    }
    eprintln!(
        "schema replies: {complete} of {} complete, all valid; forced calls: {calls}, all valid",
        2 * prompts.len()
    );
    assert!(complete >= prompts.len());
    // A forced call cut short is no call.
    let body = ask(user("Hello!"), json!({"tools": tools, "tool_choice": "required", "max_tokens": 6}));
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["choices"][0]["finish_reason"], "length", "{body}");
    assert!(v["choices"][0]["message"]["tool_calls"].is_null(), "{body}");

    // Log probabilities: one entry per token of the answer, whose bytes are the answer's.
    let body = ask(
        user("Name three colors."),
        json!({"logprobs": true, "top_logprobs": 3, "max_tokens": 30}),
    );
    let (text, v) = content(&body);
    let check = |entries: &[Value], text: &str| {
        let mut bytes = Vec::new();
        for e in entries {
            let lp = e["logprob"].as_f64().unwrap();
            let top = e["top_logprobs"].as_array().unwrap();
            assert!(lp <= 0.0 && top.len() == 3, "{e}");
            // Greedy: the token is the most likely one, and alternatives are in order.
            assert_eq!(top[0]["token"], e["token"], "{e}");
            assert!((top[0]["logprob"].as_f64().unwrap() - lp).abs() < 1e-6);
            assert!(top.windows(2).all(|w| w[0]["logprob"].as_f64() >= w[1]["logprob"].as_f64()));
            assert!(top.iter().map(|t| t["logprob"].as_f64().unwrap().exp()).sum::<f64>() <= 1.0 + 1e-4);
            bytes.extend(e["bytes"].as_array().unwrap().iter().map(|b| b.as_u64().unwrap() as u8));
        }
        assert_eq!(String::from_utf8_lossy(&bytes).trim(), text.trim());
    };
    let entries = v["choices"][0]["logprobs"]["content"].as_array().unwrap().clone();
    assert_eq!(entries.len() as u64, v["usage"]["completion_tokens"].as_u64().unwrap(), "{body}");
    check(&entries, &text);
    // Streamed: the same entries, spread over the chunks.
    let body = ask(
        user("Name three colors."),
        json!({"logprobs": true, "top_logprobs": 3, "max_tokens": 30, "stream": true}),
    );
    let streamed: Vec<Value> = body
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter_map(|d| serde_json::from_str::<Value>(d).ok())
        .filter_map(|c| c["choices"][0]["logprobs"]["content"].as_array().cloned())
        .flatten()
        .collect();
    // The same tokens. This request repeats the previous one, so the values are compared in
    // `logprobs_streamed_and_repeated`, which separates streaming from cache reuse.
    let key = |l: &[Value]| l.iter().map(|e| (e["token"].clone(), e["bytes"].clone())).collect::<Vec<_>>();
    assert_eq!(key(&streamed), key(&entries));
    // Without the option there are none; under a constraint they are still the model's own
    // (a forced token can be unlikely).
    assert!(content(&ask(user("Hi"), json!({"max_tokens": 5}))).1["choices"][0]["logprobs"].is_null());
    let (text, v) = content(&ask(
        user("Hello!"),
        json!({"logprobs": true, "response_format": {"type": "json_object"}}),
    ));
    let entries = v["choices"][0]["logprobs"]["content"].as_array().unwrap();
    let bytes: Vec<u8> = entries
        .iter()
        .flat_map(|e| e["bytes"].as_array().unwrap().iter().map(|b| b.as_u64().unwrap() as u8))
        .collect();
    assert_eq!(String::from_utf8_lossy(&bytes).trim(), text.trim());
    // Completions: the legacy shape.
    let req = json!({"model": "chat", "prompt": "The capital of France is", "max_tokens": 4, "temperature": 0, "logprobs": 2}).to_string();
    let (st, body) = request(s.addr(), "POST", "/v1/completions", &req);
    assert_eq!(st, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    let lp = &v["choices"][0]["logprobs"];
    assert_eq!(lp["tokens"].as_array().unwrap().len(), 4, "{body}");
    assert_eq!(
        lp["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t.as_str().unwrap())
            .collect::<String>(),
        v["choices"][0]["text"].as_str().unwrap()
    );
    assert!(
        lp["top_logprobs"][0].as_object().unwrap().len() == 2
            && lp["text_offset"][0] == 0
            && lp["token_logprobs"][0].as_f64().unwrap() <= 0.0
    );
    s.stop().unwrap();
}

/// Log probabilities do not depend on streaming: the same request on a newly loaded model gives
/// the same entries streamed or not, bit for bit, and so does a repeat that reuses nothing
/// (`cached_tokens` 0: models with recurrent state, such as Qwen3.5). A repeat that reuses the
/// cache (Qwen3, Gemma, Llama) feeds only the last prompt token, alone instead of inside the
/// prompt's pass, on the CPU's decode kernels instead of the prefill ones: the reply is then
/// that of a slightly different computation, as it always was for the text, and its values
/// move with the weights' precision (measured: up to 0.42 on Qwen3-4B at 4 bits, where the
/// logits of the two pass shapes differ by up to 0.63, KL 1.3e-4, in `examples/prefill_kl`;
/// 2.4e-5 on bf16 weights). Whether it is streamed changes nothing there either.
#[test]
#[ignore = "fetches Qwen/Qwen3-0.6B"]
fn logprobs_streamed_and_repeated() {
    let chat = env("OXINFER_E2E_CHAT", "Qwen/Qwen3-0.6B");
    let opts = format!("model=chat={chat}{}", env("OXINFER_E2E_OPTS", ""));
    let req = |stream: bool| {
        json!({"model": "chat", "messages": [{"role": "user", "content": "Name three colors."}], "temperature": 0, "max_tokens": 30,
            "logprobs": true, "top_logprobs": 3, "stream": stream, "stream_options": {"include_usage": true}})
        .to_string()
    };
    // The entries of a reply and the prompt tokens taken from the cache.
    let run = |s: &http::Running, stream: bool| -> (Vec<Value>, u64) {
        let (st, body) = request(s.addr(), "POST", "/v1/chat/completions", &req(stream));
        assert_eq!(st, 200, "{body}");
        let chunks: Vec<Value> = if stream {
            body.lines()
                .filter_map(|l| l.strip_prefix("data: "))
                .filter_map(|d| serde_json::from_str(d).ok())
                .collect()
        } else {
            vec![serde_json::from_str(&body).unwrap()]
        };
        let entries = chunks
            .iter()
            .filter_map(|c| c["choices"][0]["logprobs"]["content"].as_array().cloned())
            .flatten()
            .collect();
        let cached = chunks
            .iter()
            .filter_map(|c| c["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64())
            .max();
        (entries, cached.unwrap())
    };
    let tokens = |l: &[Value]| l.iter().map(|e| (e["token"].clone(), e["bytes"].clone())).collect::<Vec<_>>();
    let diff = |a: &[Value], b: &[Value]| {
        assert_eq!(tokens(a), tokens(b));
        let d = |x: &Value, y: &Value| (x["logprob"].as_f64().unwrap() - y["logprob"].as_f64().unwrap()).abs();
        a.iter()
            .zip(b)
            .flat_map(|(x, y)| {
                let tops = x["top_logprobs"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .zip(y["top_logprobs"].as_array().unwrap());
                std::iter::once(d(x, y)).chain(tops.map(|(p, q)| d(p, q))).collect::<Vec<_>>()
            })
            .fold(0.0, f64::max)
    };
    // Each on a newly loaded model: no cache to reuse.
    let s = serve(&opts);
    let (plain, c0) = run(&s, false);
    // The same server again, not streamed then streamed: both may reuse the cache.
    let (plain2, c1) = run(&s, false);
    let (streamed2, c2) = run(&s, true);
    s.stop().unwrap();
    let s = serve(&opts);
    let (streamed, c3) = run(&s, true);
    let (plain3, c4) = run(&s, false);
    s.stop().unwrap();
    assert!(!plain.is_empty() && c0 == 0 && c3 == 0);
    eprintln!(
        "logprobs of {} tokens; new model, streamed vs not: {:.2e}; repeated not streamed ({c1} cached): {:.2e}; then streamed ({c2} cached): {:.2e} vs the first, {:.2e} vs the repeat; streamed then not ({c4} cached): {:.2e}",
        plain.len(),
        diff(&plain, &streamed),
        diff(&plain, &plain2),
        diff(&plain, &streamed2),
        diff(&plain2, &streamed2),
        diff(&streamed, &plain3),
    );
    assert_eq!(plain, streamed, "streaming changed the log probabilities");
    // The two repeats on one server reuse the same cache: the same passes, streamed or not.
    assert_eq!(
        (c1, &plain2),
        (c2, &streamed2),
        "streaming changed the log probabilities of a repeat"
    );
    for (cached, other) in [(c1, &plain2), (c4, &plain3)] {
        if cached == 0 {
            // Nothing reused: the passes are the same ones.
            assert_eq!(&plain, other);
        } else {
            // Another pass shape for the last prompt token: the same tokens (checked by
            // `diff`), values within what the weights' precision gives.
            assert!(diff(&plain, other) < 1.0);
        }
    }
}

#[test]
#[ignore = "fetches Qwen/Qwen3-0.6B"]
fn prompt_cache_across_restarts() {
    let chat = env("OXINFER_E2E_CHAT", "Qwen/Qwen3-0.6B");
    let dir = std::env::temp_dir().join(format!("oxinfer-pc-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let opts = format!("model=chat={chat}; prompt_cache={}", dir.display());
    let system = "You are a careful assistant for a library. ".repeat(12);
    let req = json!({"model": "chat", "temperature": 0, "max_tokens": 20, "messages": [
        {"role": "system", "content": system}, {"role": "user", "content": "Name one famous library."}]})
    .to_string();
    let snapshots = || {
        std::fs::read_dir(&dir)
            .map(|d| d.flatten().filter(|e| e.path().extension().is_some_and(|x| x == "oxsnap")).count())
            .unwrap_or(0)
    };
    let s = serve(&opts);
    let (st, first) = request(s.addr(), "POST", "/v1/chat/completions", &req);
    assert_eq!(st, 200, "{first}");
    // The conversation is stored after the response, on another thread.
    for _ in 0..300 {
        if snapshots() > 0 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    assert!(snapshots() > 0, "no snapshot stored");
    s.stop().unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777, 0o700);
    }
    // A new server (the model loads again): the same request restores the stored prompt and
    // gives the same reply.
    let s = serve(&opts);
    let (st, again) = request(s.addr(), "POST", "/v1/chat/completions", &req);
    assert_eq!(st, 200, "{again}");
    let (a, b): (Value, Value) = (serde_json::from_str(&first).unwrap(), serde_json::from_str(&again).unwrap());
    assert_eq!(a["choices"][0]["message"]["content"], b["choices"][0]["message"]["content"]);
    let cached = b["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap();
    assert!(cached > 100, "{again}");
    s.stop().unwrap();
    std::fs::remove_dir_all(&dir).ok();
}

/// A thinking conversation on a tiny random Inkling-2-Mini: each request extends the previous
/// one with the reply as returned (its `reasoning_content` too, which the re-rendered
/// conversation leaves out, so the reply's tokens differ from those in the cache). The second
/// and third turns still reuse the cache up to the previous prompt.
#[test]
fn inkling_chat_reuses_the_cache_after_thinking() {
    let dir = std::env::temp_dir().join(format!("oxinfer-ink-server-{}", std::process::id()));
    oxinfer::inkling::tiny::write_tiny(&dir, 5).unwrap();
    let s = serve(&format!("model=ink={},device=cpu", dir.display()));
    let ask = |messages: &[Value]| {
        let req = json!({"model": "ink", "messages": messages, "temperature": 0, "max_tokens": 80});
        let (st, body) = request(s.addr(), "POST", "/v1/chat/completions", &req.to_string());
        assert_eq!(st, 200, "{body}");
        serde_json::from_str::<Value>(&body).unwrap()
    };
    let mut convo = vec![
        json!({"role": "system", "content": "You are a careful assistant. ".repeat(8)}),
        json!({"role": "user", "content": "What is the tallest mountain?"}),
    ];
    let mut prompt = 0;
    for (turn, next) in ["And the second tallest?", "Thanks. Which one is harder to climb?", ""]
        .into_iter()
        .enumerate()
    {
        let v = ask(&convo);
        let usage = &v["usage"];
        let cached = usage["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap_or(0) as usize;
        if turn > 0 {
            assert!(
                cached + 1 >= prompt,
                "turn {turn}: {cached} cached tokens of the previous prompt's {prompt}: {v}"
            );
        }
        prompt = usage["prompt_tokens"].as_u64().unwrap() as usize;
        let m = &v["choices"][0]["message"];
        convo.push(json!({
            "role": "assistant",
            "content": m["content"].as_str().unwrap_or_default(),
            "reasoning_content": m["reasoning_content"].as_str().unwrap_or("Let me think about mountains."),
        }));
        convo.push(json!({"role": "user", "content": next}));
    }
    s.stop().unwrap();
    std::fs::remove_dir_all(&dir).ok();
}

/// A 24-bit BMP of `w x h` pixels, a gradient that `seed` shifts.
fn bmp(w: usize, h: usize, seed: u8) -> Vec<u8> {
    let row = (w * 3).next_multiple_of(4);
    let size = 54 + row * h;
    let mut b = Vec::with_capacity(size);
    b.extend_from_slice(b"BM");
    b.extend_from_slice(&(size as u32).to_le_bytes());
    b.extend_from_slice(&[0, 0, 0, 0]);
    b.extend_from_slice(&54u32.to_le_bytes());
    b.extend_from_slice(&40u32.to_le_bytes());
    b.extend_from_slice(&(w as i32).to_le_bytes());
    b.extend_from_slice(&(h as i32).to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&24u16.to_le_bytes());
    b.extend_from_slice(&[0; 4]);
    b.extend_from_slice(&((row * h) as u32).to_le_bytes());
    b.extend_from_slice(&[0; 16]);
    for y in 0..h {
        for x in 0..w {
            let v = (x * 13 + y * 7) as u8;
            b.extend_from_slice(&[v.wrapping_add(seed), v.wrapping_mul(3), seed.wrapping_mul(5).wrapping_add(y as u8)]);
        }
        b.resize(b.len() + row - w * 3, 0);
    }
    b
}

/// An agent's long conversation on a tiny random Inkling-2-Mini with a vision tower, as opencode
/// sends it: tool calls, their results (thousands of tokens each), and after some of them a user
/// message holding the pictures a tool returned. Every request reuses the cache up to the
/// previous prompt, the pictures in it included (they used to make each request start over from
/// the system prompt), and after a restart the last one comes back from the prompt cache.
#[test]
fn inkling_agent_conversation_with_images_reuses_the_cache() {
    let dir = std::env::temp_dir().join(format!("oxinfer-ink-server-vision-{}", std::process::id()));
    let cache = std::env::temp_dir().join(format!("oxinfer-ink-server-vision-pc-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&cache);
    oxinfer::inkling::tiny::write_tiny_vision(&dir, 5).unwrap();
    let spec = format!(
        "model=ink={},device=cpu,prompt_cache={},prompt_cache_min_tokens=8",
        dir.display(),
        cache.display()
    );
    let tools = json!([{"type": "function", "function": {"name": "read", "description": "Reads a file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]);
    let ask = |addr: SocketAddr, messages: &[Value]| {
        let req = json!({"model": "ink", "messages": messages, "tools": tools, "temperature": 0, "max_tokens": 6});
        let (st, body) = request(addr, "POST", "/v1/chat/completions", &req.to_string());
        assert_eq!(st, 200, "{body}");
        serde_json::from_str::<Value>(&body).unwrap()
    };
    let s = serve(&spec);
    let mut convo = vec![
        json!({"role": "system", "content": "You are a coding agent. ".repeat(20)}),
        json!({"role": "user", "content": "Check the figures of the paper."}),
    ];
    let (mut prompt, mut last) = (0, Value::Null);
    for turn in 0..5 {
        let v = ask(s.addr(), &convo);
        let usage = &v["usage"];
        let cached = usage["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap_or(0) as usize;
        if turn > 0 {
            assert!(
                cached + 1 >= prompt,
                "turn {turn}: {cached} cached tokens of the previous prompt's {prompt}: {usage}"
            );
        }
        prompt = usage["prompt_tokens"].as_u64().unwrap() as usize;
        last = v["choices"][0]["message"].clone();
        let id = format!("call_{turn}");
        convo.push(json!({"role": "assistant", "content": last["content"].as_str().unwrap_or_default(),
            "tool_calls": [{"id": id, "type": "function", "function": {"name": "read", "arguments": format!("{{\"path\":\"fig{turn}.png\"}}")}}]}));
        let text: String = (0..120).map(|i| format!("{turn} {i:4}: some file content here\n")).collect();
        convo.push(json!({"role": "tool", "tool_call_id": id, "content": text}));
        if turn % 2 == 0 {
            let url = format!("data:image/bmp;base64,{}", b64_encode(&bmp(16 + 8 * turn, 16, turn as u8 * 30)));
            convo.push(json!({"role": "user", "content": [
                {"type": "text", "text": "Attached media from tool result:"},
                {"type": "image_url", "image_url": {"url": url}}]}));
        }
    }
    assert!(prompt > 15_000, "{prompt} tokens");
    s.stop().unwrap();
    // After a restart: the last request again, from the disk (its prompt but the last token).
    convo.truncate(convo.len() - 3);
    let s = serve(&spec);
    let v = ask(s.addr(), &convo);
    let cached = v["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap_or(0) as usize;
    assert!(cached + 1 >= prompt, "{cached} of {prompt} restored: {v}");
    assert_eq!(v["choices"][0]["message"]["content"], last["content"]);
    s.stop().unwrap();
    std::fs::remove_dir_all(&cache).ok();
    std::fs::remove_dir_all(&dir).ok();
}

/// The same on tiny random DeepSeek V4 and V4.1 models, thinking on, with a prompt cache: the
/// second turn reuses the cache up to the previous prompt (in-memory checkpoints), and
/// after a restart the conversation comes back from the disk snapshots with the same reply.
#[test]
fn deepseek_chat_reuses_the_cache_after_thinking() {
    for v41 in [false, true] {
        let dir = std::env::temp_dir().join(format!("oxinfer-ds4-server-{v41}-{}", std::process::id()));
        let cache = std::env::temp_dir().join(format!("oxinfer-ds4-server-pc-{v41}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&cache);
        oxinfer::deepseek4::tiny::write_chat(&dir, v41, json!({})).unwrap();
        let spec = format!(
            "model=ds={},device=cpu,prompt_cache={},prompt_cache_min_tokens=8",
            dir.display(),
            cache.display()
        );
        let ask = |addr: SocketAddr, messages: &[Value]| {
            let req = json!({"model": "ds", "messages": messages, "temperature": 0, "max_tokens": 80, "think": true});
            let (st, body) = request(addr, "POST", "/v1/chat/completions", &req.to_string());
            assert_eq!(st, 200, "{body}");
            serde_json::from_str::<Value>(&body).unwrap()
        };
        let s = serve(&spec);
        let mut convo = vec![
            json!({"role": "system", "content": "You are a careful assistant. ".repeat(8)}),
            json!({"role": "user", "content": "What is the tallest mountain?"}),
        ];
        let mut prompt = 0;
        let mut last = Value::Null;
        for (turn, next) in ["And the second tallest?", "Thanks. Which one is harder to climb?"]
            .into_iter()
            .enumerate()
        {
            let v = ask(s.addr(), &convo);
            let usage = &v["usage"];
            let cached = usage["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap_or(0) as usize;
            if turn > 0 {
                assert!(
                    cached + 1 >= prompt,
                    "v41 {v41}, turn {turn}: {cached} cached tokens of the previous prompt's {prompt}: {v}"
                );
            }
            prompt = usage["prompt_tokens"].as_u64().unwrap() as usize;
            let m = &v["choices"][0]["message"];
            convo.push(json!({
                "role": "assistant",
                "content": m["content"].as_str().unwrap_or_default(),
                "reasoning_content": m["reasoning_content"].as_str().unwrap_or("Let me think about mountains."),
            }));
            convo.push(json!({"role": "user", "content": next}));
            last = v;
        }
        s.stop().unwrap();
        // After a restart: the last request again, from the disk.
        convo.truncate(convo.len() - 2);
        let s = serve(&spec);
        let v = ask(s.addr(), &convo);
        let cached = v["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap_or(0) as usize;
        assert!(cached + 1 >= prompt, "v41 {v41}: {cached} of {prompt} restored: {v}");
        assert_eq!(v["choices"][0]["message"], last["choices"][0]["message"], "v41 {v41}");
        s.stop().unwrap();
        std::fs::remove_dir_all(&cache).ok();
        std::fs::remove_dir_all(&dir).ok();
    }
}

/// The same on tiny random Qwen3.5 models (dense, and MoE with MTP), thinking on, with a prompt
/// cache: the second and third turns reuse the cache up to the previous prompt (in-memory
/// checkpoints of the recurrent states), and after a restart the conversation comes back from
/// the disk snapshots with the same reply.
#[test]
fn qwen35_chat_reuses_the_cache_after_thinking() {
    for (moe, mtp) in [(false, false), (true, true)] {
        let tag = format!("{moe}-{mtp}");
        let dir = std::env::temp_dir().join(format!("oxinfer-q35-server-{tag}-{}", std::process::id()));
        let cache = std::env::temp_dir().join(format!("oxinfer-q35-server-pc-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&cache);
        oxinfer::qwen::tiny::write(&dir, moe, mtp).unwrap();
        let spec = format!(
            "model=q={},device=cpu,mtp={},prompt_cache={},prompt_cache_min_tokens=8",
            dir.display(),
            mtp as u8,
            cache.display()
        );
        let ask = |addr: SocketAddr, messages: &[Value]| {
            let req = json!({"model": "q", "messages": messages, "temperature": 0, "max_tokens": 40, "think": true});
            let (st, body) = request(addr, "POST", "/v1/chat/completions", &req.to_string());
            assert_eq!(st, 200, "{body}");
            serde_json::from_str::<Value>(&body).unwrap()
        };
        let s = serve(&spec);
        let mut convo = vec![
            json!({"role": "system", "content": "You are a careful assistant. ".repeat(8)}),
            json!({"role": "user", "content": "What is the tallest mountain?"}),
        ];
        let mut prompt = 0;
        let mut last = Value::Null;
        for (turn, next) in ["And the second tallest?", "Thanks. Which one is harder to climb?"]
            .into_iter()
            .enumerate()
        {
            let v = ask(s.addr(), &convo);
            let usage = &v["usage"];
            let cached = usage["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap_or(0) as usize;
            if turn > 0 {
                assert!(
                    cached + 1 >= prompt,
                    "{tag}, turn {turn}: {cached} cached tokens of the previous prompt's {prompt}: {v}"
                );
            }
            prompt = usage["prompt_tokens"].as_u64().unwrap() as usize;
            let m = &v["choices"][0]["message"];
            convo.push(json!({
                "role": "assistant",
                "content": m["content"].as_str().unwrap_or_default(),
                "reasoning_content": m["reasoning_content"].as_str().unwrap_or("Let me think about mountains."),
            }));
            convo.push(json!({"role": "user", "content": next}));
            last = v;
        }
        s.stop().unwrap();
        // After a restart: the last request again, from the disk.
        convo.truncate(convo.len() - 2);
        let s = serve(&spec);
        let v = ask(s.addr(), &convo);
        let cached = v["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap_or(0) as usize;
        assert!(cached + 1 >= prompt, "{tag}: {cached} of {prompt} restored: {v}");
        assert_eq!(v["choices"][0]["message"], last["choices"][0]["message"], "{tag}");
        s.stop().unwrap();
        std::fs::remove_dir_all(&cache).ok();
        std::fs::remove_dir_all(&dir).ok();
    }
}

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn b64_decode(s: &str) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    let (mut acc, mut bits) = (0u32, 0);
    for c in s.bytes().filter(|&c| c != b'=') {
        acc = acc << 6 | B64.iter().position(|&x| x == c)? as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Some(out)
}

fn b64_encode(b: &[u8]) -> String {
    let mut s = String::new();
    for c in b.chunks(3) {
        let n = c.iter().enumerate().fold(0u32, |a, (i, &x)| a | (x as u32) << (16 - 8 * i));
        for i in 0..4 {
            s.push(if i <= c.len() {
                B64[(n >> (18 - 6 * i) & 63) as usize] as char
            } else {
                '='
            });
        }
    }
    s
}

/// A folder with only a `model_index.json` / `config.json`: detected as a kind, fails to load.
fn fake_model(tag: &str, file: &str, body: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("oxinfer-fake-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    std::fs::write(d.join(file), body).unwrap();
    d
}

fn wait_video(addr: SocketAddr, id: &str, until: &[&str]) -> Value {
    for _ in 0..600 {
        let (st, body) = request(addr, "GET", &format!("/v1/videos/{id}"), "");
        assert_eq!(st, 200, "{body}");
        let v: Value = serde_json::from_str(&body).unwrap();
        if until.contains(&v["status"].as_str().unwrap()) {
            return v;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    panic!("video {id} did not reach {until:?}");
}

#[test]
fn videos_routing_and_jobs() {
    let h3 = fake_model("h3", "model_index.json", r#"{"_class_name": "MiniMaxH3ModularPipeline"}"#);
    let chat = fake_model("chat", "config.json", r#"{"model_type": "llama", "vocab_size": 128256}"#);
    let s = serve(&format!("model=v={}; model=c={}", h3.display(), chat.display()));
    let a = s.addr();
    let (st, body) = request(a, "GET", "/v1/models/v", "");
    assert_eq!(
        (st, serde_json::from_str::<Value>(&body).unwrap()["kind"].clone()),
        (200, json!("video"))
    );
    // Checked before a job is made.
    for (req, code, param) in [
        (json!({"model": "v"}), 400, "prompt"),
        (json!({"prompt": "x"}), 400, "model"),
        (json!({"model": "nope", "prompt": "x"}), 404, "model"),
        (json!({"model": "c", "prompt": "x"}), 400, "model"),
        (json!({"model": "v", "prompt": "x", "size": "100x100"}), 400, "size"),
        (json!({"model": "v", "prompt": "x", "seconds": 0}), 400, "seconds"),
        (json!({"model": "v", "prompt": "x", "steps": 1}), 400, "steps"),
        (
            json!({"model": "v", "prompt": "x", "first_frame": "/etc/passwd"}),
            400,
            "first_frame",
        ),
    ] {
        let (st, body) = request(a, "POST", "/v1/videos", &req.to_string());
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(st, code, "{req}: {body}");
        if param != "first_frame" {
            assert_eq!(v["error"]["param"], param, "{req}: {body}");
        }
    }
    // A job: created at once, then failing to load the model (the folder has no weights).
    let (st, body) = request(
        a,
        "POST",
        "/v1/videos",
        &json!({"model": "v", "prompt": "a fox", "seconds": "5", "size": "448x256"}).to_string(),
    );
    assert_eq!(st, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(
        (v["object"].as_str(), v["status"].as_str(), v["size"].as_str()),
        (Some("video"), Some("queued"), Some("448x256"))
    );
    let id = v["id"].as_str().unwrap().to_string();
    assert!(id.starts_with("video_"));
    let done = wait_video(a, &id, &["failed", "completed"]);
    assert_eq!(done["status"], "failed", "{done}");
    assert!(done["error"]["message"].as_str().unwrap().contains("v"), "{done}");
    let (st, body) = request(a, "GET", &format!("/v1/videos/{id}/content?variant=video"), "");
    assert_eq!(st, 409, "{body}");
    // The multipart form (OpenAI's shape) and the alias route.
    let form = "--B\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nv\r\n--B\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nwaves\r\n--B--\r\n";
    let (st, body) = request_with(
        a,
        "POST",
        "/v1/videos/generations",
        &["Host: localhost", "Content-Type: multipart/form-data; boundary=B"],
        form,
    );
    assert_eq!(st, 200, "{body}");
    let id2 = serde_json::from_str::<Value>(&body).unwrap()["id"].as_str().unwrap().to_string();
    let (st, body) = request(a, "POST", &format!("/v1/videos/{id2}/cancel"), "");
    assert_eq!(st, 200, "{body}");
    let v = wait_video(a, &id2, &["failed", "cancelled", "completed"]);
    assert_ne!(v["status"], "completed");
    // The list, newest first, then deletion.
    let (st, body) = request(a, "GET", "/v1/videos?limit=10", "");
    let list: Value = serde_json::from_str(&body).unwrap();
    assert_eq!((st, list["object"].as_str()), (200, Some("list")));
    assert_eq!(list["data"][0]["id"], id2.as_str());
    assert_eq!(list["data"][1]["id"], id.as_str());
    let (st, body) = request(a, "GET", "/v1/videos?limit=1", "");
    assert_eq!(
        (st, serde_json::from_str::<Value>(&body).unwrap()["has_more"].clone()),
        (200, json!(true))
    );
    let (st, body) = request(a, "GET", &format!("/v1/videos/{id}/events"), "");
    assert_eq!(st, 200);
    assert!(body.contains("event: video") && body.contains("\"failed\""), "{body}");
    let (st, body) = request(a, "DELETE", &format!("/v1/videos/{id}"), "");
    assert_eq!(
        (st, serde_json::from_str::<Value>(&body).unwrap()["deleted"].clone()),
        (200, json!(true))
    );
    let (st, _) = request(a, "GET", &format!("/v1/videos/{id}"), "");
    assert_eq!(st, 404);
    let (st, _) = request(a, "PUT", "/v1/videos", "");
    assert_eq!(st, 405);
    s.stop().unwrap();
    for d in [h3, chat] {
        std::fs::remove_dir_all(d).ok();
    }
}

#[test]
fn voices_spoken_chat_and_image_streams() {
    let chat = fake_model("chat2", "config.json", r#"{"model_type": "llama", "vocab_size": 128256}"#);
    let tts = fake_model("tts", "config.json", r#"{"model_type": "llama", "vocab_size": 156940}"#);
    let sd = fake_model("sd", "model_index.json", r#"{"_class_name": "StableDiffusionPipeline"}"#);
    let s = serve(&format!(
        "model=c={}; model=t={}; model=sd={}",
        chat.display(),
        tts.display(),
        sd.display()
    ));
    let a = s.addr();
    let (st, body) = request(a, "GET", "/v1/audio/voices", "");
    assert_eq!(st, 400, "{body}");
    let (st, body) = request(a, "GET", "/v1/audio/voices?model=c", "");
    assert_eq!(st, 400, "{body}");
    assert!(body.contains("not a speech model"), "{body}");
    // A speech model is loaded to list its voices: this one has no weights.
    let (st, body) = request(a, "GET", "/v1/audio/voices?model=t", "");
    assert_eq!(st, 500, "{body}");
    // Audio out from a text-only model, a format the server does not write, a streamed WAV.
    let msgs = json!([{"role": "user", "content": "hi"}]);
    for (extra, param) in [
        (json!({"model": "c", "audio": {"format": "wav"}}), "modalities"),
        (json!({"model": "c", "audio": {"format": "mp3"}}), "audio"),
        (json!({"model": "c", "stream": true, "audio": {"format": "wav"}}), "audio"),
        (json!({"model": "t"}), "model"),
    ] {
        let mut req = extra.clone();
        req["messages"] = msgs.clone();
        req["modalities"] = json!(["text", "audio"]);
        let (st, body) = request(a, "POST", "/v1/chat/completions", &req.to_string());
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(st, if param == "model" { 500 } else { 400 }, "{req}: {body}");
        if st == 400 {
            assert_eq!(v["error"]["param"], param, "{req}: {body}");
        }
    }
    // An image stream: the error comes as an event once the stream has started.
    let (st, body) = request(
        a,
        "POST",
        "/v1/images/generations",
        &json!({"model": "sd", "prompt": "x", "stream": true}).to_string(),
    );
    assert_eq!(st, 200, "{body}");
    assert!(
        body.contains("event: image_generation.progress") && body.contains("\"stage\":\"loading\""),
        "{body}"
    );
    assert!(body.contains("event: error"), "{body}");
    let (st, _) = request(
        a,
        "POST",
        "/v1/images/generations",
        &json!({"model": "sd", "prompt": "x", "stream": "maybe"}).to_string(),
    );
    assert_eq!(st, 400);
    let (st, _) = request(
        a,
        "POST",
        "/v1/images/generations",
        &json!({"model": "nope", "prompt": "x", "stream": true}).to_string(),
    );
    assert_eq!(st, 404);
    s.stop().unwrap();
    for d in [chat, tts, sd] {
        std::fs::remove_dir_all(d).ok();
    }
}

/// The SSE events of a body: (event name, JSON data).
fn events(body: &str) -> Vec<(String, Value)> {
    body.split("\n\n")
        .filter_map(|ev| {
            let name = ev.lines().find_map(|l| l.strip_prefix("event: "))?.to_string();
            let data = ev.lines().find_map(|l| l.strip_prefix("data: "))?;
            Some((name, serde_json::from_str(data).ok()?))
        })
        .collect()
}

#[test]
fn sd3_images_with_step_progress() {
    let dir = std::env::temp_dir().join(format!("oxinfer-sd3-tiny-{}", std::process::id()));
    oxinfer::sd3::tiny::write_tiny_release(&dir).unwrap();
    let s = serve(&format!("model=sd3={}", dir.display()));
    let a = s.addr();
    let (_, body) = request(a, "GET", "/v1/models/sd3", "");
    assert_eq!(serde_json::from_str::<Value>(&body).unwrap()["kind"], "image", "{body}");
    // The plain request and response are as before.
    let req = json!({"model": "sd3", "prompt": "a red fox", "size": "64x64", "steps": 3, "seed": 1});
    let (st, body) = request(a, "POST", "/v1/images/generations", &req.to_string());
    assert_eq!(st, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    let png = b64_decode(v["data"][0]["b64_json"].as_str().unwrap()).unwrap();
    assert!(png.starts_with(b"\x89PNG"));
    // Streamed: one progress event per step, then the image.
    let mut sreq = req.clone();
    sreq["stream"] = json!(true);
    sreq["n"] = json!(2);
    let (st, body) = request(a, "POST", "/v1/images/generations", &sreq.to_string());
    assert_eq!(st, 200, "{body}");
    let ev = events(&body);
    let steps: Vec<(u64, u64, u64)> = ev
        .iter()
        .filter(|(n, v)| n == "image_generation.progress" && v["stage"] == "generating")
        .map(|(_, v)| {
            (
                v["image_index"].as_u64().unwrap(),
                v["step"].as_u64().unwrap(),
                v["steps"].as_u64().unwrap(),
            )
        })
        .collect();
    assert_eq!(steps, [(0, 1, 3), (0, 2, 3), (0, 3, 3), (1, 1, 3), (1, 2, 3), (1, 3, 3)], "{body}");
    let done: Vec<&Value> = ev
        .iter()
        .filter(|(n, _)| n == "image_generation.completed")
        .map(|(_, v)| v)
        .collect();
    assert_eq!(done.len(), 2);
    assert_eq!(done[0]["size"], "64x64");
    assert_eq!(
        b64_decode(done[0]["b64_json"].as_str().unwrap()).unwrap(),
        png,
        "same seed, same image"
    );
    // An edit (image to image), streamed as image_edit events.
    let form = "--B\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nsd3\r\n--B\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nwinter\r\n--B\r\nContent-Disposition: form-data; name=\"stream\"\r\n\r\ntrue\r\n--B\r\nContent-Disposition: form-data; name=\"steps\"\r\n\r\n5\r\n--B\r\nContent-Disposition: form-data; name=\"image\"; filename=\"in.png\"\r\nContent-Type: image/png\r\n\r\n";
    let mut raw = form.as_bytes().to_vec();
    raw.extend_from_slice(&png);
    raw.extend_from_slice(b"\r\n--B--\r\n");
    let mut c = std::net::TcpStream::connect(a).unwrap();
    write!(
        c,
        "POST /v1/images/edits HTTP/1.1\r\nHost: localhost\r\nContent-Type: multipart/form-data; boundary=B\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        raw.len()
    )
    .unwrap();
    c.write_all(&raw).unwrap();
    let mut out = Vec::new();
    c.read_to_end(&mut out).unwrap();
    let text = String::from_utf8_lossy(&out);
    assert!(
        text.contains("event: image_edit.progress") && text.contains("event: image_edit.completed"),
        "{text}"
    );
    s.stop().unwrap();
    std::fs::remove_dir_all(&dir).ok();
}

/// Decodes a WAV body: (sample rate, samples).
fn wav(bytes: &[u8]) -> (u32, usize) {
    assert_eq!(&bytes[..4], b"RIFF");
    let rate = u32::from_le_bytes(bytes[24..28].try_into().unwrap());
    (rate, (bytes.len() - 44) / 2)
}

#[test]
#[ignore = "fetches LiquidAI/LFM3-Omni-2B (4.5 GB)"]
fn lfm3_omni_speech_and_voice_chat() {
    let m = env("OXINFER_E2E_OMNI", "LiquidAI/LFM3-Omni-2B");
    let s = serve(&format!("model=omni={m}"));
    let a = s.addr();
    let (st, body) = request(a, "GET", "/v1/audio/voices?model=omni", "");
    assert_eq!(st, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!((v["default"].as_str(), v["data"].as_array().unwrap().len()), (Some("female"), 2));
    // Text to speech through /v1/audio/speech.
    let req = json!({"model": "omni", "input": "Hello from the server.", "voice": "male", "seed": 3});
    let mut c = std::net::TcpStream::connect(a).unwrap();
    let b = req.to_string();
    write!(c, "POST /v1/audio/speech HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}", b.len()).unwrap();
    let mut raw = Vec::new();
    c.read_to_end(&mut raw).unwrap();
    let at = raw.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
    assert!(String::from_utf8_lossy(&raw[..at]).contains(" 200 "));
    let (rate, n) = wav(&raw[at..]);
    assert_eq!(rate, 24000);
    assert!(n > 24000 / 2, "{n} samples");
    // A voice chat: the answer as text and speech.
    let req = json!({"model": "omni", "modalities": ["text", "audio"], "audio": {"voice": "alloy", "format": "wav"}, "seed": 1,
        "messages": [{"role": "user", "content": "What is the capital of France? Answer in one short sentence."}]});
    let (st, body) = request(a, "POST", "/v1/chat/completions", &req.to_string());
    assert_eq!(st, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    let audio = &v["choices"][0]["message"]["audio"];
    eprintln!("transcript: {}", audio["transcript"]);
    assert!(audio["transcript"].as_str().unwrap().to_lowercase().contains("paris"), "{body}");
    let (rate, n) = wav(&b64_decode(audio["data"].as_str().unwrap()).unwrap());
    assert!(rate == 24000 && n > 24000, "{n} samples");
    // Streamed: transcript pieces, then PCM16.
    let mut sreq = req.clone();
    sreq["stream"] = json!(true);
    sreq["audio"]["format"] = json!("pcm16");
    let (st, body) = request(a, "POST", "/v1/chat/completions", &sreq.to_string());
    assert_eq!(st, 200, "{body}");
    assert!(
        body.contains("\"transcript\"") && body.contains("\"data\"") && body.ends_with("data: [DONE]\n\n"),
        "{body}"
    );
    s.stop().unwrap();
}

#[test]
#[ignore = "fetches higurashi-ai/Haru-4B (10.5 GB) and LiquidAI/LFM3-Omni-2B (for a recording)"]
fn haru_speech_to_speech() {
    let haru = env("OXINFER_E2E_HARU", "higurashi-ai/Haru-4B");
    let omni = env("OXINFER_E2E_OMNI", "LiquidAI/LFM3-Omni-2B");
    let s = serve(&format!("model=haru={haru}; model=omni={omni}"));
    let a = s.addr();
    // A recording of a question, spoken by LFM3-Omni.
    let req = json!({"model": "omni", "input": "Hi! How are you doing today?", "seed": 2});
    let mut c = std::net::TcpStream::connect(a).unwrap();
    let b = req.to_string();
    write!(c, "POST /v1/audio/speech HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}", b.len()).unwrap();
    let mut raw = Vec::new();
    c.read_to_end(&mut raw).unwrap();
    let at = raw.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
    let question = &raw[at..];
    let (st, body) = request(a, "GET", "/v1/audio/voices?model=haru", "");
    assert_eq!(st, 200, "{body}");
    assert_eq!(serde_json::from_str::<Value>(&body).unwrap()["family"], "haru");
    let req = json!({"model": "haru", "modalities": ["text", "audio"], "seed": 1, "max_reply_seconds": 8,
        "messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": b64_encode(question), "format": "wav"}}]}]});
    let (st, body) = request(a, "POST", "/v1/chat/completions", &req.to_string());
    assert_eq!(st, 200, "{}", &body[..body.len().min(400)]);
    let v: Value = serde_json::from_str(&body).unwrap();
    let audio = &v["choices"][0]["message"]["audio"];
    eprintln!("haru: {} ({}s)", audio["transcript"], audio["duration"]);
    let (rate, n) = wav(&b64_decode(audio["data"].as_str().unwrap()).unwrap());
    assert!(rate == 24000 && n > 2400, "{n} samples");
    // Text to speech is not Haru's.
    let (st, body) = request(a, "POST", "/v1/audio/speech", &json!({"model": "haru", "input": "hi"}).to_string());
    assert_eq!(st, 400, "{body}");
    s.stop().unwrap();
}

/// A GET returning raw bytes: (status, content type, body).
fn get_bytes(addr: SocketAddr, path: &str) -> (u16, String, Vec<u8>) {
    let mut s = std::net::TcpStream::connect(addr).unwrap();
    write!(s, "GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n").unwrap();
    let mut raw = Vec::new();
    s.read_to_end(&mut raw).unwrap();
    let at = raw.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
    let head = String::from_utf8_lossy(&raw[..at]).to_ascii_lowercase();
    let status = head.split(' ').nth(1).unwrap().parse().unwrap();
    let ct = head
        .lines()
        .find_map(|l| l.strip_prefix("content-type: "))
        .unwrap_or("")
        .trim()
        .to_string();
    let mut body = raw[at..].to_vec();
    if head.contains("transfer-encoding: chunked") {
        let (mut out, mut rest) = (Vec::new(), &body[..]);
        while let Some(nl) = rest.windows(2).position(|w| w == b"\r\n") {
            let n = usize::from_str_radix(std::str::from_utf8(&rest[..nl]).unwrap().trim(), 16).unwrap();
            if n == 0 {
                break;
            }
            out.extend_from_slice(&rest[nl + 2..nl + 2 + n]);
            rest = &rest[nl + 4 + n..];
        }
        body = out;
    }
    (status, ct, body)
}

#[test]
fn video_jobs_with_a_tiny_minimax_h3() {
    let base = std::env::temp_dir().join(format!("oxinfer-h3-tiny-{}", std::process::id()));
    let (video, image) = (base.join("video"), base.join("image"));
    oxinfer::minimax_h3::tiny::write_tiny_video_release(&video).unwrap();
    oxinfer::minimax_h3::tiny::write_tiny_image_release(&image).unwrap();
    let s = serve(&format!("model=h3={}; model=h3i={}", video.display(), image.display()));
    let a = s.addr();
    // Text to video, then first and last frame to video (images as data: URLs).
    let req = json!({"model": "h3", "prompt": "A red fox trots across the snow.", "size": "64x64", "frames": 22, "steps": 4, "seed": 3});
    let (st, body) = request(a, "POST", "/v1/videos", &req.to_string());
    assert_eq!(st, 200, "{body}");
    let id = serde_json::from_str::<Value>(&body).unwrap()["id"].as_str().unwrap().to_string();
    let (st, stream) = request(a, "GET", &format!("/v1/videos/{id}/events"), "");
    assert_eq!(st, 200);
    let v = wait_video(a, &id, &["completed", "failed"]);
    assert_eq!(v["status"], "completed", "{v}");
    assert_eq!(
        (v["progress"].as_u64(), v["frames"].as_u64(), v["size"].as_str()),
        (Some(100), Some(22), Some("64x64"))
    );
    assert_eq!((v["steps"].as_u64(), v["has_audio"].as_bool()), (Some(3), Some(true)), "{v}");
    let steps: Vec<u64> = stream
        .split("\n\n")
        .filter_map(|e| e.strip_prefix("event: video\ndata: "))
        .filter_map(|d| serde_json::from_str::<Value>(d).ok())
        .filter(|e| e["stage"] == "generating")
        .filter_map(|e| e["step"].as_u64())
        .collect();
    assert!(steps.windows(2).all(|w| w[0] <= w[1]), "{steps:?}");
    // MP4 by default: H.264 (an `avc1` sample entry) and AAC (`mp4a`).
    let (st, ct, mp4) = get_bytes(a, &format!("/v1/videos/{id}/content"));
    assert_eq!((st, ct.as_str(), &mp4[4..8]), (200, "video/mp4", &b"ftyp"[..]));
    let has = |b: &[u8], tag: &[u8]| b.windows(tag.len()).any(|w| w == tag);
    assert!(has(&mp4, b"avc1") && has(&mp4, b"mp4a"), "H.264 and AAC tracks");
    // AVI (Motion JPEG and PCM) when asked for.
    let req = json!({"model": "h3", "prompt": "A red fox.", "size": "64x64", "frames": 22, "steps": 3, "response_format": "avi"});
    let (st, body) = request(a, "POST", "/v1/videos", &req.to_string());
    assert_eq!(st, 200, "{body}");
    let ida = serde_json::from_str::<Value>(&body).unwrap()["id"].as_str().unwrap().to_string();
    assert_eq!(wait_video(a, &ida, &["completed", "failed"])["status"], "completed");
    let (st, ct, avi) = get_bytes(a, &format!("/v1/videos/{ida}/content"));
    assert_eq!(
        (st, ct.as_str(), &avi[..4], &avi[8..12]),
        (200, "video/x-msvideo", &b"RIFF"[..], &b"AVI "[..])
    );
    assert_eq!(
        avi.windows(4).filter(|w| w == b"00dc").count(),
        2 * 22,
        "22 frames and their index entries"
    );
    let (st, _) = request(
        a,
        "POST",
        "/v1/videos",
        &json!({"model": "h3", "prompt": "x", "response_format": "gif"}).to_string(),
    );
    assert_eq!(st, 400);
    let (st, ct, wav) = get_bytes(a, &format!("/v1/videos/{id}/content?variant=audio"));
    assert_eq!((st, ct.as_str(), &wav[..4]), (200, "audio/wav", &b"RIFF"[..]));
    assert_eq!(u16::from_le_bytes([wav[22], wav[23]]), 2, "stereo");
    let (st, ct, png) = get_bytes(a, &format!("/v1/videos/{id}/content?variant=thumbnail"));
    assert_eq!((st, ct.as_str(), &png[..4]), (200, "image/png", &b"\x89PNG"[..]));
    let first = format!("data:image/png;base64,{}", b64_encode(&png));
    let req = json!({"model": "h3", "prompt": "The fox turns around.", "size": "64x64", "frames": 22, "steps": 3, "input_reference": {"image_url": first}, "last_frame": first});
    let (st, body) = request(a, "POST", "/v1/videos", &req.to_string());
    assert_eq!(st, 200, "{body}");
    let id2 = serde_json::from_str::<Value>(&body).unwrap()["id"].as_str().unwrap().to_string();
    let v = wait_video(a, &id2, &["completed", "failed"]);
    assert_eq!(
        (v["status"].as_str(), v["size"].as_str()),
        (Some("completed"), Some("64x64")),
        "{v}"
    );
    // References (ref2va): an image and an audio clip, uploaded.
    let mut form = Vec::new();
    for (k, v) in [
        ("model", "h3"),
        ("prompt", "subject_definitions: <Picture 1>, voice: <Audio 1>"),
        ("frames", "22"),
        ("steps", "3"),
        ("size", "64x64"),
        ("ref_image_edge", "64"),
    ] {
        form.extend_from_slice(format!("--B\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n").as_bytes());
    }
    for (name, ct, data) in [("a.png", "image/png", &png), ("b.wav", "audio/wav", &wav)] {
        form.extend_from_slice(
            format!("--B\r\nContent-Disposition: form-data; name=\"references[]\"; filename=\"{name}\"\r\nContent-Type: {ct}\r\n\r\n")
                .as_bytes(),
        );
        form.extend_from_slice(data);
        form.extend_from_slice(b"\r\n");
    }
    form.extend_from_slice(b"--B--\r\n");
    let mut c = std::net::TcpStream::connect(a).unwrap();
    write!(
        c,
        "POST /v1/videos HTTP/1.1\r\nHost: localhost\r\nContent-Type: multipart/form-data; boundary=B\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        form.len()
    )
    .unwrap();
    c.write_all(&form).unwrap();
    let mut out = String::new();
    c.read_to_string(&mut out).unwrap();
    let body = out.split_once("\r\n\r\n").unwrap().1;
    let id3 = serde_json::from_str::<Value>(body).unwrap()["id"].as_str().unwrap().to_string();
    let v = wait_video(a, &id3, &["completed", "failed"]);
    assert_eq!(v["status"], "completed", "{v}");
    // Cancelled between steps: a large canvas and many steps (slow enough even for the tiny
    // model), cancelled once it denoises.
    let req = json!({"model": "h3", "prompt": "x", "size": "768x768", "frames": 22, "steps": 200});
    let (_, body) = request(a, "POST", "/v1/videos", &req.to_string());
    let id4 = serde_json::from_str::<Value>(&body).unwrap()["id"].as_str().unwrap().to_string();
    for _ in 0..2000 {
        let (_, b) = request(a, "GET", &format!("/v1/videos/{id4}"), "");
        if serde_json::from_str::<Value>(&b).unwrap()["step"].as_u64() > Some(1) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    request(a, "POST", &format!("/v1/videos/{id4}/cancel"), "");
    let v = wait_video(a, &id4, &["completed", "failed", "cancelled"]);
    assert_eq!(v["status"], "cancelled", "{v}");
    // MiniMax-H3-Image as an image model (a small canvas: OXINFER_H3_MIN_AREA).
    // SAFETY: set before any other thread of this test reads it; no other test reads it.
    unsafe { std::env::set_var("OXINFER_H3_MIN_AREA", "4096") };
    let (st, body) = request(
        a,
        "POST",
        "/v1/images/generations",
        &json!({"model": "h3i", "prompt": "a fox", "size": "64x64", "stream": true}).to_string(),
    );
    assert_eq!(st, 200, "{body}");
    let ev = events(&body);
    assert!(
        ev.iter()
            .filter(|(n, v)| n == "image_generation.progress" && v["stage"] == "generating")
            .count()
            >= 1,
        "{body}"
    );
    assert!(
        ev.iter().any(|(n, v)| n == "image_generation.completed" && v["size"] == "64x64"),
        "{body}"
    );
    let (st, body) = request(
        a,
        "POST",
        "/v1/images/generations",
        &json!({"model": "h3i", "prompt": "a fox", "steps": 4}).to_string(),
    );
    assert_eq!(st, 500, "{body}");
    s.stop().unwrap();
    std::fs::remove_dir_all(&base).ok();
}

#[test]
fn video_jobs_with_a_tiny_ltx() {
    let base = std::env::temp_dir().join(format!("oxinfer-ltx-tiny-{}", std::process::id()));
    oxinfer::ltx::tiny::write_tiny_release(&base).unwrap();
    let s = serve(&format!("model=ltx={}", base.display()));
    let a = s.addr();
    // Two stages; the frame count from `seconds` at the request's rate.
    let req = json!({"model": "ltx", "prompt": "A red fox trots across the snow.", "size": "64x128", "seconds": 0.5, "fps": 16, "seed": 3});
    let (st, body) = request(a, "POST", "/v1/videos", &req.to_string());
    assert_eq!(st, 200, "{body}");
    let id = serde_json::from_str::<Value>(&body).unwrap()["id"].as_str().unwrap().to_string();
    let v = wait_video(a, &id, &["completed", "failed"]);
    assert_eq!(v["status"], "completed", "{v}");
    assert_eq!(
        (
            v["frames"].as_u64(),
            v["size"].as_str(),
            v["fps"].as_f64(),
            v["has_audio"].as_bool()
        ),
        (Some(1), Some("64x128"), Some(16.0), Some(true)),
        "{v}"
    );
    let (st, ct, png) = get_bytes(a, &format!("/v1/videos/{id}/content?variant=thumbnail"));
    assert_eq!((st, ct.as_str(), &png[..4]), (200, "image/png", &b"\x89PNG"[..]));
    // One stage with a first frame (a data: URL).
    let first = format!("data:image/png;base64,{}", b64_encode(&png));
    let req = json!({"model": "ltx", "prompt": "The fox turns around.", "size": "64x32", "frames": 9, "one_stage": true, "input_reference": first});
    let (st, body) = request(a, "POST", "/v1/videos", &req.to_string());
    assert_eq!(st, 200, "{body}");
    let id2 = serde_json::from_str::<Value>(&body).unwrap()["id"].as_str().unwrap().to_string();
    let v = wait_video(a, &id2, &["completed", "failed"]);
    assert_eq!(
        (v["status"].as_str(), v["frames"].as_u64(), v["size"].as_str()),
        (Some("completed"), Some(9), Some("64x32")),
        "{v}"
    );
    let (st, ct, mp4) = get_bytes(a, &format!("/v1/videos/{id2}/content"));
    assert_eq!((st, ct.as_str(), &mp4[4..8]), (200, "video/mp4", &b"ftyp"[..]));
    // A size the two-stage pipeline cannot take fails the job.
    let req = json!({"model": "ltx", "prompt": "x", "size": "96x64", "frames": 9});
    let (st, body) = request(a, "POST", "/v1/videos", &req.to_string());
    assert_eq!(st, 200, "{body}");
    let id3 = serde_json::from_str::<Value>(&body).unwrap()["id"].as_str().unwrap().to_string();
    assert_eq!(wait_video(a, &id3, &["completed", "failed"])["status"], "failed");
    s.stop().unwrap();
    std::fs::remove_dir_all(&base).ok();
}

/// Models a request names without them being listed (`auto_models`): a repo in the cache joins
/// the list; not without `auto_models`, not for another site's page, not past the memory budget.
#[test]
fn auto_models_admission() {
    let base = std::env::temp_dir().join(format!("oxinfer-auto-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let m = base.join("hub/org--model");
    std::fs::create_dir_all(&m).unwrap();
    std::fs::write(m.join("config.json"), r#"{"model_type": "bert"}"#).unwrap();
    std::fs::write(m.join("modules.json"), "[]").unwrap();
    // Over 1 MB: counted as weights.
    std::fs::write(m.join("model.safetensors"), vec![0u8; 3 << 20]).unwrap();
    let cache = format!("cache_dir={}", base.join("hub").display());
    let embed = json!({"model": "org/model", "input": "hi"}).to_string();
    let ids = |a: SocketAddr| -> Vec<String> {
        let (_, body) = request(a, "GET", "/v1/models", "");
        let v: Value = serde_json::from_str(&body).unwrap();
        v["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["id"].as_str().unwrap().to_string())
            .collect()
    };

    let off = serve(&cache);
    let (st, body) = request(off.addr(), "POST", "/v1/embeddings", &embed);
    assert_eq!(st, 404, "{body}");
    off.stop().unwrap();

    let on = serve(&format!("{cache}; auto_models=1"));
    let a = on.addr();
    // Another site's page: not listed.
    let headers = ["Host: localhost", "Origin: http://evil.example", "Content-Type: application/json"];
    let (st, _) = request_with(a, "POST", "/v1/embeddings", &headers, &embed);
    assert_eq!(st, 404);
    // A path is never listed on request.
    let path = json!({"model": m.display().to_string(), "input": "hi"}).to_string();
    let (st, _) = request(a, "POST", "/v1/embeddings", &path);
    assert_eq!(st, 404);
    assert!(ids(a).is_empty());
    // The repo in the cache: listed and loaded (its weights are not real: the load fails).
    let (st, body) = request(a, "POST", "/v1/embeddings", &embed);
    assert_eq!(st, 500, "{body}");
    assert!(body.contains("load org/model"), "{body}");
    assert_eq!(ids(a), ["org/model"]);
    on.stop().unwrap();

    let small = serve(&format!("{cache}; auto_models=1; memory_gb=0.001"));
    let (st, body) = request(small.addr(), "POST", "/v1/embeddings", &embed);
    assert_eq!(st, 507, "{body}");
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap()["error"]["code"],
        "insufficient_memory"
    );
    assert!(ids(small.addr()).is_empty());
    small.stop().unwrap();
    std::fs::remove_dir_all(&base).ok();
}

/// A model fetched and loaded because a request names it, then unloaded once idle.
#[test]
#[ignore = "fetches Qwen/Qwen3-0.6B"]
fn auto_models_fetch_and_idle_unload() {
    let chat = env("OXINFER_E2E_CHAT", "Qwen/Qwen3-0.6B");
    // 3 s of idleness (the upkeep looks every 5 s).
    let s = serve("auto_models=1; idle_unload_min=0.05");
    let a = s.addr();
    let req = json!({"model": chat, "messages": [{"role": "user", "content": "Reply with just OK."}], "max_tokens": 20});
    let (st, body) = request(a, "POST", "/v1/chat/completions", &req.to_string());
    assert_eq!(st, 200, "{body}");
    let loaded = || {
        let (_, body) = request(a, "GET", "/v1/models", "");
        let v: Value = serde_json::from_str(&body).unwrap();
        let m = v["data"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["id"] == chat.as_str())
            .cloned()
            .unwrap();
        m["loaded"].as_bool().unwrap()
    };
    assert!(loaded());
    let t0 = std::time::Instant::now();
    while loaded() {
        assert!(t0.elapsed().as_secs() < 30, "still loaded after 30 s");
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
    // Asked again: loaded again.
    let (st, body) = request(a, "POST", "/v1/chat/completions", &req.to_string());
    assert_eq!(st, 200, "{body}");
    s.stop().unwrap();
}

// ---- the status page: requests in flight, cancel, unload, the caches ---------------------------

/// A prompt-cache snapshot as an engine stores it (`model`: its fingerprint), without a model.
fn fake_snapshot(dir: &std::path::Path, name: &str, model: &str, model_type: &str, tokens: &[u32]) {
    std::fs::create_dir_all(dir).unwrap();
    oxinfer::snapshot::write_atomic(&dir.join(format!("{name}.oxsnap")), |f| {
        let mut w = oxinfer::snapshot::SnapWriter::new(f)?;
        w.u32s("tokens", tokens)?;
        w.finish(json!({"model": model, "model_type": model_type, "tokens": tokens.len()}))?;
        Ok(())
    })
    .unwrap();
}

fn json_of(body: &str) -> Value {
    serde_json::from_str(body).unwrap_or_else(|e| panic!("{e}: {body}"))
}

#[test]
fn status_page_and_prompt_cache() {
    let base = std::env::temp_dir().join(format!("oxinfer-status-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let pc = base.join("pc");
    fake_snapshot(&pc, "a", "fp-one", "qwen3", &[1, 2, 3]);
    fake_snapshot(&pc, "b", "fp-one", "qwen3", &[1, 2, 3, 4]);
    fake_snapshot(&pc, "c", "fp-two", "gemma3", &[5]);
    std::fs::write(pc.join("notes.txt"), "not a snapshot").unwrap();
    let s = serve(&format!("web=1; model=m=/nonexistent/model; prompt_cache={}", pc.display()));
    let a = s.addr();
    let (st, body) = request(a, "GET", "/api/web/activity", "");
    assert_eq!(st, 200, "{body}");
    let v = json_of(&body);
    assert_eq!(v["version"], env!("CARGO_PKG_VERSION"));
    assert!(v["uptime"].is_u64() && v["started"].is_u64());
    assert_eq!(v["requests"], json!([]));
    assert_eq!((&v["served"], &v["cancelled"]), (&json!(0), &json!(0)));
    assert_eq!(v["process"]["pid"], std::process::id());
    assert!(v["machine"].is_object());
    assert_eq!((&v["models"][0]["id"], &v["models"][0]["state"]), (&json!("m"), &json!("unloaded")));
    let cache = &v["prompt_cache"];
    assert_eq!(cache["entries"], 3, "{cache}");
    let d = &cache["dirs"][0];
    assert_eq!(d["path"], pc.display().to_string());
    let one = d["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["fingerprint"] == "fp-one")
        .unwrap();
    assert_eq!(
        (&one["entries"], &one["model_type"], &one["tokens"]),
        (&json!(2), &json!("qwen3"), &json!(7))
    );
    assert_eq!(one["ids"], json!([]), "no model loaded with this fingerprint");
    let (st, body) = request(a, "GET", "/api/web/prompt-cache", "");
    assert_eq!((st, json_of(&body)["entries"].as_u64()), (200, Some(3)));

    // Nothing in flight.
    let (st, body) = request(a, "GET", "/api/web/requests", "");
    assert_eq!((st, &json_of(&body)["data"]), (200, &json!([])));
    let (st, body) = request(a, "POST", "/api/web/requests/req-999/cancel", "{}");
    assert_eq!((st, &json_of(&body)["error"]["code"]), (404, &json!("request_not_found")));

    // Clearing: a model never loaded cannot be told apart; a directory must be the cache's.
    let clear = |b: Value| request(a, "POST", "/api/web/prompt-cache/clear", &b.to_string());
    let (st, body) = clear(json!({"model": "m"}));
    assert_eq!(st, 409, "{body}");
    let (st, _) = clear(json!({"model": "nope"}));
    assert_eq!(st, 404);
    let (st, body) = clear(json!({"dir": base.display().to_string()}));
    assert_eq!(st, 400, "{body}");
    assert_eq!(std::fs::read_dir(&pc).unwrap().count(), 4);
    let (st, body) = clear(json!({"fingerprint": "fp-one", "dir": pc.display().to_string()}));
    assert_eq!((st, &json_of(&body)["removed"]), (200, &json!(2)), "{body}");
    assert!(!pc.join("a.oxsnap").exists() && !pc.join("b.oxsnap").exists() && pc.join("c.oxsnap").exists());
    let (st, body) = clear(json!({}));
    assert_eq!((st, &json_of(&body)["removed"]), (200, &json!(1)), "{body}");
    assert!(!pc.join("c.oxsnap").exists());
    assert!(pc.join("notes.txt").exists(), "only snapshots are removed");
    let (_, body) = request(a, "GET", "/api/web/activity", "");
    assert_eq!(json_of(&body)["prompt_cache"]["entries"], 0);

    // No model streams from disk here: nothing to free.
    let (st, body) = request(a, "POST", "/api/web/caches/free", "{}");
    assert_eq!((st, &json_of(&body)["freed"]), (200, &json!(0)), "{body}");
    let (st, _) = request(a, "POST", "/api/web/caches/free", r#"{"model": "nope"}"#);
    assert_eq!(st, 404);

    // Another site's page changes nothing.
    let evil = ["Host: localhost", "Origin: http://evil.example", "Content-Type: application/json"];
    fake_snapshot(&pc, "d", "fp-two", "gemma3", &[5]);
    for p in [
        "/api/web/prompt-cache/clear",
        "/api/web/requests/req-1/cancel",
        "/api/web/caches/free",
    ] {
        let (st, _) = request_with(a, "POST", p, &evil, "{}");
        assert_eq!(st, 403, "{p}");
    }
    assert!(pc.join("d.oxsnap").exists());
    s.stop().unwrap();

    // With an API key: the key is needed for all of it.
    let s = serve(&format!("web=1; api_key=k; prompt_cache={}", pc.display()));
    for (m, p) in [
        ("GET", "/api/web/activity"),
        ("GET", "/api/web/requests"),
        ("POST", "/api/web/prompt-cache/clear"),
    ] {
        let (st, _) = request(s.addr(), m, p, "{}");
        assert_eq!(st, 401, "{p}");
    }
    let (st, _) = request_with(
        s.addr(),
        "GET",
        "/api/web/activity",
        &["Host: localhost", "Authorization: Bearer k"],
        "",
    );
    assert_eq!(st, 200);
    s.stop().unwrap();

    // A non-loopback listener without a key: the status reads, the actions are refused.
    let o = ServerOptions::from_pairs(&format!("web=1; log=0; listen=0.0.0.0:0; prompt_cache={}", pc.display())).unwrap();
    let api = Api::new(o).unwrap();
    let get = oxinfer::server::ApiRequest::new("GET", "/api/web/activity", None, vec![]);
    assert_eq!(api.handle(&get).collect().status, 200);
    for p in [
        "/api/web/prompt-cache/clear",
        "/api/web/requests/req-1/cancel",
        "/api/web/caches/free",
    ] {
        let r = api.handle(&oxinfer::server::ApiRequest::post_json(p, "{}")).collect();
        assert_eq!(r.status, 403, "{p}");
        assert_eq!(serde_json::from_slice::<Value>(&r.body).unwrap()["error"]["code"], "read_only");
    }
    assert!(pc.join("d.oxsnap").exists());
    // The library: the same report and actions.
    assert_eq!(api.status()["prompt_cache"]["entries"], 1);
    assert!(api.requests().is_empty() && !api.cancel("req-1"));
    assert_eq!(api.clear_prompt_cache(None).unwrap().entries, 1);
    assert_eq!(api.free_caches(None).unwrap(), 0);
    std::fs::remove_dir_all(&base).ok();
}

/// Polls `/api/web/requests` until a request matches `pred`; returns it.
fn wait_request(a: SocketAddr, what: &str, pred: impl Fn(&Value) -> bool) -> Value {
    let t0 = std::time::Instant::now();
    loop {
        let (_, body) = request(a, "GET", "/api/web/requests", "");
        let v = json_of(&body);
        if let Some(r) = v["data"].as_array().unwrap().iter().find(|r| pred(r)) {
            return r.clone();
        }
        if t0.elapsed().as_secs() >= 120 {
            let (_, act) = request(a, "GET", "/api/web/activity", "");
            panic!("no request {what} after 120 s: {v}; recent: {}", json_of(&act)["recent"]);
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

/// Cancelling requests (running, queued, streamed) and unloading a model, with a tiny random
/// SD 3.5 (denoising steps are its stopping points).
#[test]
fn cancel_requests_and_unload_models() {
    let dir = std::env::temp_dir().join(format!("oxinfer-cancel-sd3-{}", std::process::id()));
    oxinfer::sd3::tiny::write_tiny_release(&dir).unwrap();
    let s = serve(&format!("web=1; model=sd3={}", dir.display()));
    let a = s.addr();
    // Far more steps than the test waits for.
    let slow = json!({"model": "sd3", "prompt": "a fox", "size": "256x256", "steps": 100000, "seed": 1}).to_string();
    let spawn = |body: String| std::thread::spawn(move || request(a, "POST", "/v1/images/generations", &body));
    let cancel = |id: &str| request(a, "POST", &format!("/api/web/requests/{id}/cancel"), "{}");

    let running = spawn(slow.clone());
    let r = wait_request(a, "denoising", |r| r["step"]["done"].as_u64() > Some(0));
    assert_eq!(
        (&r["kind"], &r["model"], &r["stage"]),
        (&json!("image"), &json!("sd3"), &json!("generating")),
        "{r}"
    );
    assert_eq!((&r["endpoint"], &r["stream"]), (&json!("/v1/images/generations"), &json!(false)));
    assert!(r["client"].as_str().unwrap().starts_with("127.0.0.1:"), "{r}");
    let id = r["id"].as_str().unwrap().to_string();

    // A second request waits for the model; cancelled, it gives up at once.
    let queued = spawn(slow.clone());
    let q = wait_request(a, "queued", |r| r["stage"] == "queued");
    let (st, body) = cancel(q["id"].as_str().unwrap());
    assert_eq!(st, 200, "{body}");
    let (st, body) = queued.join().unwrap();
    assert_eq!((st, &json_of(&body)["error"]["code"]), (499, &json!("cancelled")), "{body}");

    // The running one stops at its next step.
    let t0 = std::time::Instant::now();
    let (st, body) = cancel(&id);
    assert_eq!((st, &json_of(&body)["cancelled"]), (200, &json!(true)), "{body}");
    let (st, body) = running.join().unwrap();
    assert_eq!((st, &json_of(&body)["error"]["code"]), (499, &json!("cancelled")), "{body}");
    assert!(t0.elapsed().as_secs() < 30, "stopped after {:?}", t0.elapsed());
    let (_, body) = request(a, "GET", "/api/web/activity", "");
    let v = json_of(&body);
    assert_eq!(v["requests"], json!([]), "{v}");
    assert_eq!((&v["served"], &v["cancelled"]), (&json!(2), &json!(2)));
    assert_eq!(v["recent"][0]["outcome"], "cancelled");
    assert_eq!(v["models"][0]["state"], "ready", "its slot is free again");

    // Streamed: the stream ends with an error event.
    let mut sreq = json_of(&slow);
    sreq["stream"] = json!(true);
    let streamed = spawn(sreq.to_string());
    let r = wait_request(a, "streamed", |r| r["stream"] == true && r["step"]["done"].as_u64() > Some(0));
    cancel(r["id"].as_str().unwrap());
    let (st, body) = streamed.join().unwrap();
    assert_eq!(st, 200);
    let ev = events(&body);
    let (name, err) = ev.last().unwrap();
    assert_eq!((name.as_str(), &err["error"]["code"]), ("error", &json!("cancelled")), "{body}");

    // The model serves the next request as usual.
    let quick = json!({"model": "sd3", "prompt": "a fox", "size": "64x64", "steps": 2, "seed": 1}).to_string();
    let (st, body) = request(a, "POST", "/v1/images/generations", &quick);
    assert_eq!(st, 200, "{body}");

    // Unloading: refused while a request uses the model, unless its requests are cancelled.
    let busy = spawn(slow.clone());
    wait_request(a, "busy", |r| r["step"]["done"].as_u64() > Some(0));
    let (st, body) = request(a, "POST", "/api/web/models/unload", r#"{"id": "sd3"}"#);
    assert_eq!(st, 409, "{body}");
    let (st, body) = request(a, "POST", "/api/web/models/unload", r#"{"id": "sd3", "cancel": true}"#);
    let v = json_of(&body);
    assert_eq!((st, &v["unloaded"], &v["cancelled"]), (200, &json!(true), &json!(1)), "{body}");
    assert_eq!(busy.join().unwrap().0, 499);
    let (_, body) = request(a, "GET", "/api/web/activity", "");
    let m = &json_of(&body)["models"][0];
    assert_eq!(
        (&m["loaded"], &m["state"], &m["memory"]),
        (&json!(false), &json!("unloaded"), &json!(0)),
        "{m}"
    );
    // Loaded again by a request, then unloaded while idle.
    let (st, _) = request(a, "POST", "/v1/images/generations", &quick);
    assert_eq!(st, 200);
    let (_, body) = request(a, "GET", "/api/web/activity", "");
    let m = &json_of(&body)["models"][0];
    assert_eq!(m["loaded"], true, "{m}");
    assert!(m["loaded_at"].is_u64() && m["load_secs"].is_number(), "{m}");
    let (st, body) = request(a, "POST", "/api/web/models/unload", r#"{"id": "sd3"}"#);
    assert_eq!((st, &json_of(&body)["unloaded"]), (200, &json!(true)), "{body}");
    s.stop().unwrap();
    std::fs::remove_dir_all(&dir).ok();
}

/// Cancelling a chat completion mid-reply, and clearing a loaded model's prompt-cache snapshots.
#[test]
#[ignore = "fetches Qwen/Qwen3-0.6B"]
fn cancel_chat_and_clear_prompt_cache_with_a_model() {
    let chat = env("OXINFER_E2E_CHAT", "Qwen/Qwen3-0.6B");
    let pc = std::env::temp_dir().join(format!("oxinfer-status-pc-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&pc);
    fake_snapshot(&pc, "other", "another-model", "gemma3", &[1, 2, 3]);
    let s = serve(&format!("web=1; model=chat={chat}; prompt_cache={}", pc.display()));
    let a = s.addr();
    let system = "You are a careful assistant for a library. ".repeat(12);
    let long = json!({"model": "chat", "max_tokens": 4000, "temperature": 0.7, "messages": [
        {"role": "system", "content": system.clone()}, {"role": "user", "content": "Write a very long story about a lighthouse."}]})
    .to_string();
    let t = std::thread::spawn(move || request(a, "POST", "/v1/chat/completions", &long));
    let r = wait_request(a, "generating", |r| r["tokens"].as_u64() > Some(5));
    assert_eq!((&r["kind"], &r["model"]), (&json!("chat"), &json!("chat")), "{r}");
    let t0 = std::time::Instant::now();
    let (st, _) = request(a, "POST", &format!("/api/web/requests/{}/cancel", r["id"].as_str().unwrap()), "{}");
    assert_eq!(st, 200);
    let (st, body) = t.join().unwrap();
    assert_eq!((st, &json_of(&body)["error"]["code"]), (499, &json!("cancelled")), "{body}");
    assert!(t0.elapsed().as_secs() < 5, "stopped after {:?}", t0.elapsed());
    let (_, body) = request(a, "GET", "/api/web/activity", "");
    let v = json_of(&body);
    let m = &v["models"][0];
    assert_eq!((&m["loaded"], &m["state"]), (&json!(true), &json!("ready")), "{m}");
    assert!(m["device"].is_string() && m["model_type"].is_string(), "{m}");
    assert!(v["recent"][0]["tokens"].as_u64() > Some(5));
    // A finished reply is stored; clearing the model's snapshots leaves the other model's.
    let short = json!({"model": "chat", "max_tokens": 10, "temperature": 0, "messages": [
        {"role": "system", "content": system}, {"role": "user", "content": "Name one famous library."}]})
    .to_string();
    let (st, body) = request(a, "POST", "/v1/chat/completions", &short);
    assert_eq!(st, 200, "{body}");
    let mine = || {
        let (_, body) = request(a, "GET", "/api/web/prompt-cache", "");
        let v = json_of(&body);
        v["dirs"][0]["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|g| g["ids"] == json!(["chat"]))
            .map(|g| g["entries"].as_u64().unwrap())
            .unwrap_or(0)
    };
    let t0 = std::time::Instant::now();
    while mine() == 0 {
        assert!(t0.elapsed().as_secs() < 30, "no snapshot stored");
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    let (st, body) = request(a, "POST", "/api/web/prompt-cache/clear", r#"{"model": "chat"}"#);
    assert_eq!(st, 200, "{body}");
    assert!(json_of(&body)["removed"].as_u64() > Some(0), "{body}");
    assert_eq!(mine(), 0);
    assert!(pc.join("other.oxsnap").exists());
    s.stop().unwrap();
    std::fs::remove_dir_all(&pc).ok();
}

/// Concurrent chat requests on one model (`parallel`, `crate::batch`): each streamed reply equals
/// the one its request gets alone (greedy), with the decode steps batched.
#[test]
#[ignore = "fetches Qwen/Qwen3-0.6B"]
fn concurrent_chats_match_alone() {
    let chat = env("OXINFER_E2E_CHAT", "Qwen/Qwen3-0.6B");
    let prompts: Vec<String> = (0..4)
        .map(|i| {
            format!(
                "Request {i}. {} Write a short story about it.",
                "The river and the lantern. ".repeat(20 * i + 3)
            )
        })
        .collect();
    let replies = |s: &http::Running, at_once: bool| -> Vec<String> {
        let addr = s.addr();
        let ask = move |p: String| {
            let req =
                json!({"model": "chat", "messages": [{"role": "user", "content": p}], "temperature": 0, "max_tokens": 48, "stream": true});
            let (st, body) = request(addr, "POST", "/v1/chat/completions", &req.to_string());
            assert_eq!(st, 200, "{body}");
            body.lines()
                .filter_map(|l| l.strip_prefix("data: "))
                .filter_map(|d| serde_json::from_str::<Value>(d).ok())
                .filter_map(|c| c["choices"][0]["delta"]["content"].as_str().map(str::to_string))
                .collect::<String>()
        };
        if at_once {
            let hs: Vec<_> = prompts.iter().cloned().map(|p| std::thread::spawn(move || ask(p))).collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        } else {
            prompts.iter().cloned().map(ask).collect()
        }
    };
    let s = serve(&format!("model=chat={chat},parallel=4"));
    let together = replies(&s, true);
    s.stop().unwrap();
    let s = serve(&format!("model=chat={chat},parallel=1"));
    let alone = replies(&s, false);
    s.stop().unwrap();
    for (i, (a, b)) in together.iter().zip(&alone).enumerate() {
        assert!(!a.is_empty(), "request {i}: empty reply");
        assert_eq!(a, b, "request {i}");
    }
}

/// Replaces `dir/adapter` with a new PEFT folder the way a trainer does: written beside it, then
/// renamed into place.
fn put_adapter(profile: &std::path::Path, seed: u64) {
    let tmp = profile.join("adapter.tmp");
    let _ = std::fs::remove_dir_all(&tmp);
    oxinfer::qwen::tiny::write_adapter(&tmp, seed, 0.6).unwrap();
    let _ = std::fs::remove_dir_all(profile.join("adapter"));
    std::fs::rename(&tmp, profile.join("adapter")).unwrap();
}

fn jsonl(path: &std::path::Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

/// A harness that cannot see response ids (opencode's plugins): the conversation a request names
/// with `X-Session-Id` is logged with its episode, and an outcome for the session is one for its
/// episodes since its last outcome.
#[test]
fn outcomes_for_a_session() {
    let base = std::env::temp_dir().join(format!("oxinfer-learn-session-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let (model, learn) = (base.join("model"), base.join("learn"));
    oxinfer::qwen::tiny::write(&model, false, false).unwrap();
    let s = serve(&format!("model=q={},device=cpu; learn_dir={}", model.display(), learn.display()));
    let a = s.addr();
    let chat = |path: &str, header: Option<&str>, stream: bool| -> String {
        let req = json!({"model": "q@dev", "messages": [{"role": "user", "content": "Hi"}], "temperature": 0,
            "max_tokens": 4, "stream": stream});
        let mut headers = vec!["Host: localhost", "Content-Type: application/json"];
        headers.extend(header);
        let (st, body) = request_with(a, "POST", path, &headers, &req.to_string());
        assert_eq!(st, 200, "{body}");
        // (The episode's id is read from the log: a streamed body is not parsed here.)
        jsonl(&learn.join("q/dev/episodes.jsonl")).pop().unwrap()["id"]
            .as_str()
            .unwrap()
            .to_string()
    };
    let a1 = chat("/v1/chat/completions", Some("X-Session-Id: ses_A"), false);
    let b1 = chat("/v1/chat/completions", Some("x-session-id: ses_B"), false);
    let a2 = chat("/v1/chat/completions", Some("X-Session-Id: ses_A"), true);
    let n1 = chat("/v1/chat/completions", None, false);
    let n2 = chat("/v1/chat/completions", Some("X-Session-Id: not a name"), false);
    let eps = jsonl(&learn.join("q/dev/episodes.jsonl"));
    let sessions: Vec<&Value> = eps.iter().map(|e| &e["session"]).collect();
    assert_eq!(
        sessions,
        [&json!("ses_A"), &json!("ses_B"), &json!("ses_A"), &Value::Null, &Value::Null]
    );
    assert!(eps[3].get("session").is_none(), "an episode without a session has no such key");

    let post = |body: Value| {
        let (st, out) = request(a, "POST", "/v1/outcomes", &body.to_string());
        (st, json_of(&out))
    };
    let ids = |v: &Value| -> Vec<String> {
        v["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|o| o["id"].as_str().unwrap().to_string())
            .collect()
    };
    let (st, v) = post(json!({"session": "ses_A", "result": "bad", "kind": "tests", "note": "exit 1: make test"}));
    assert_eq!((st, v["recorded"].as_u64()), (200, Some(2)), "{v}");
    assert_eq!(ids(&v), [a1.clone(), a2.clone()]);
    // Nothing new since: nothing recorded, and no error (also for a session never seen).
    for session in ["ses_A", "nobody"] {
        let (st, v) = post(json!({"session": session, "result": "good"}));
        assert_eq!((st, v["recorded"].as_u64()), (200, Some(0)), "{v}");
    }
    // The replies since, of both APIs and under opencode's other header.
    let a3 = chat("/v1/messages", Some("X-Session-Id: ses_A"), false);
    let a4 = chat("/v1/chat/completions", Some("X-Opencode-Session-Id: ses_A"), false);
    let (_, v) = post(json!({"outcomes": [{"session": "ses_A", "result": "good", "kind": "tests"},
        {"session": "ses_A", "result": "good", "kind": "build"}, {"id": n1, "reward": 0.5}]}));
    assert_eq!(
        ids(&v),
        [a3.clone(), a4.clone(), n1.clone()],
        "a batch judges a session's replies once"
    );
    let outs = jsonl(&learn.join("q/dev/outcomes.jsonl"));
    assert_eq!(outs.len(), 5);
    assert_eq!(
        outs[0],
        json!({"id": a1, "time": outs[0]["time"], "reward": -1.0, "kind": "tests", "note": "exit 1: make test"})
    );
    // The listing by session; the other conversation is still unjudged.
    let (st, body) = request(a, "GET", "/v1/episodes?session=ses_A&order=asc", "");
    assert_eq!(st, 200, "{body}");
    let list = json_of(&body);
    let d = list["data"].as_array().unwrap();
    assert_eq!(
        d.iter().map(|e| e["id"].as_str().unwrap()).collect::<Vec<_>>(),
        [a1.as_str(), a2.as_str(), a3.as_str(), a4.as_str()]
    );
    assert!(d.iter().all(|e| e["session"] == "ses_A" && e["outcomes"] == 1), "{list}");
    let (_, v) = post(json!({"session": "ses_B", "reward": -0.5}));
    assert_eq!(ids(&v), [b1]);
    let (_, body) = request(a, "GET", &format!("/v1/episodes/{n2}"), "");
    assert_eq!(json_of(&body)["session"], Value::Null);
    s.stop().unwrap();
    std::fs::remove_dir_all(&base).ok();
}

/// `MODEL@PROFILE` (docs/learning.md): the episodes logged for chat requests of both APIs,
/// streamed or not, the outcomes reported for them, and the profile's adapter attached, replaced
/// and removed while the model stays loaded; requests for `MODEL` are the base model's and are
/// not logged.
#[test]
fn profiles_log_episodes_take_outcomes_and_switch_adapters() {
    let base = std::env::temp_dir().join(format!("oxinfer-learn-server-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let (model, learn) = (base.join("model"), base.join("learn"));
    oxinfer::qwen::tiny::write(&model, false, false).unwrap();
    let (ad1, ad2) = (base.join("ad1"), base.join("ad2"));
    oxinfer::qwen::tiny::write_adapter(&ad1, 21, 0.6).unwrap();
    oxinfer::qwen::tiny::write_adapter(&ad2, 22, 0.6).unwrap();
    let spec = |extra: &str| format!("model=q={},device=cpu{extra}; learn_dir={}", model.display(), learn.display());
    let chat = |addr: SocketAddr, name: &str, extra: Value| -> Value {
        let mut req = json!({"model": name, "messages": [{"role": "user", "content": "Hello there, how are you?"}],
            "temperature": 0, "max_tokens": 12});
        for (k, v) in extra.as_object().unwrap() {
            req[k] = v.clone();
        }
        let (st, body) = request(addr, "POST", "/v1/chat/completions", &req.to_string());
        assert_eq!(st, 200, "{body}");
        serde_json::from_str(&body).unwrap()
    };
    let text = |v: &Value| v["choices"][0]["message"]["content"].as_str().unwrap().to_string();
    // What the model says alone, and loaded with each adapter.
    let reference = |extra: &str| {
        let s = serve(&spec(extra));
        let t = text(&chat(s.addr(), "q", json!({})));
        s.stop().unwrap();
        t
    };
    let plain = reference("");
    let with1 = reference(&format!(",lora={}", ad1.display()));
    let with2 = reference(&format!(",lora={}", ad2.display()));
    assert!(plain != with1 && with1 != with2 && plain != with2, "the adapters change nothing");
    assert!(!learn.exists(), "a request without a profile wrote to the learn directory");

    let s = serve(&spec(""));
    let a = s.addr();
    let pdir = learn.join("q").join("dev");
    // A profile without an adapter: the base model, logged.
    let r = chat(a, "q@dev", json!({}));
    assert_eq!((r["model"].as_str(), text(&r)), (Some("q@dev"), plain.clone()));
    let id1 = r["id"].as_str().unwrap().to_string();
    let eps = jsonl(&pdir.join("episodes.jsonl"));
    assert_eq!(eps.len(), 1);
    let e = &eps[0];
    assert_eq!(
        (e["id"].as_str(), e["model"].as_str(), e["profile"].as_str()),
        (Some(id1.as_str()), Some("q"), Some("dev"))
    );
    assert_eq!(e["adapters"], json!([]));
    assert_eq!(
        e["prompt_tokens"].as_array().unwrap().len() as u64,
        r["usage"]["prompt_tokens"].as_u64().unwrap()
    );
    let (logged, counted) = (
        e["completion_tokens"].as_array().unwrap().len() as u64,
        r["usage"]["completion_tokens"].as_u64().unwrap(),
    );
    // (The stop token, when one ended the reply, is logged and not counted.)
    assert!(
        logged == counted || logged == counted + 1,
        "{logged} tokens logged, {counted} counted"
    );
    assert_eq!(e["finish_reason"], r["choices"][0]["finish_reason"]);
    assert!(e.get("logprobs").is_none());
    // The byte-level tokenizer: the prompt's tokens hold the message's bytes.
    let bytes: Vec<u8> = e["prompt_tokens"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|t| u8::try_from(t.as_u64()?).ok())
        .collect();
    assert!(String::from_utf8_lossy(&bytes).contains("Hello there, how are you?"));
    // With log probabilities: one per sampled token.
    let r = chat(a, "q@dev", json!({"logprobs": true}));
    let e = jsonl(&pdir.join("episodes.jsonl")).pop().unwrap();
    assert_eq!(e["id"], r["id"]);
    assert_eq!(
        e["logprobs"].as_array().unwrap().len(),
        e["completion_tokens"].as_array().unwrap().len()
    );
    assert!(
        e["logprobs"]
            .as_array()
            .unwrap()
            .iter()
            .all(|l| l.as_f64().is_some_and(|l| l <= 0.0))
    );
    // Streamed: every chunk carries the id the episode is logged under.
    let req = json!({"model": "q@dev", "messages": [{"role": "user", "content": "Hi"}], "temperature": 0, "max_tokens": 6, "stream": true});
    let (st, body) = request(a, "POST", "/v1/chat/completions", &req.to_string());
    assert_eq!(st, 200, "{body}");
    let chunks = sse_events(&body);
    let id2 = chunks[0]["id"].as_str().unwrap().to_string();
    assert!(chunks.iter().all(|c| c["id"] == id2.as_str() && c["model"] == "q@dev"), "{body}");
    // Anthropic's API, plain and streamed (the id is in `message_start`).
    let msg = |stream: bool| {
        json!({"model": "q@other", "max_tokens": 6, "temperature": 0, "stream": stream,
        "messages": [{"role": "user", "content": "Hi"}]})
    };
    let (st, body) = request(a, "POST", "/v1/messages", &msg(false).to_string());
    assert_eq!(st, 200, "{body}");
    let id3 = json_of(&body)["id"].as_str().unwrap().to_string();
    let (st, body) = request(a, "POST", "/v1/messages", &msg(true).to_string());
    assert_eq!(st, 200, "{body}");
    let start = sse_events(&body).into_iter().find(|e| e["type"] == "message_start").unwrap();
    let id4 = start["message"]["id"].as_str().unwrap().to_string();
    let logged = |p: &str| -> Vec<String> {
        jsonl(&learn.join("q").join(p).join("episodes.jsonl"))
            .iter()
            .map(|e| e["id"].as_str().unwrap().to_string())
            .collect()
    };
    assert_eq!(logged("dev")[2], id2);
    assert_eq!(logged("other"), [id3.clone(), id4.clone()]);
    let (st, body) = request(a, "POST", "/v1/messages/count_tokens", &msg(false).to_string());
    assert_eq!(st, 200, "{body}");
    // Names that are not profiles.
    for bad in ["q@", "q@a b", "q@.x", "q@a/b"] {
        let req = json!({"model": bad, "messages": [{"role": "user", "content": "Hi"}]});
        let (st, body) = request(a, "POST", "/v1/chat/completions", &req.to_string());
        assert!(st == 400 || st == 404, "{bad}: {st} {body}");
    }
    assert_eq!(subdirs_of(&learn.join("q")), ["dev", "other"]);

    // Outcomes: by path, with the shorthand; several ids at once; a batch; unknown ids.
    let post = |path: &str, v: Value| {
        let (st, body) = request(a, "POST", path, &v.to_string());
        (st, json_of(&body))
    };
    let (st, v) = post(
        &format!("/v1/episodes/{id1}/outcome"),
        json!({"result": "bad", "kind": "tests", "note": "2 failed"}),
    );
    assert_eq!((st, v["recorded"].as_u64()), (200, Some(1)), "{v}");
    let (st, v) = post("/v1/episodes/nope/outcome", json!({"reward": 1}));
    assert_eq!((st, v["error"]["code"].as_str()), (404, Some("episode_not_found")), "{v}");
    let (st, _) = post(&format!("/v1/episodes/{id1}/outcome"), json!({"reward": 3}));
    assert_eq!(st, 400);
    let (st, v) = post("/v1/outcomes", json!({"ids": [id1, id2], "reward": 0.5, "kind": "build"}));
    assert_eq!((st, v["recorded"].as_u64()), (200, Some(2)), "{v}");
    let batch = json!({"outcomes": [{"id": id3, "result": "good", "kind": "user_accepted"}, {"ids": [id4, id2], "result": "bad"}]});
    let (st, v) = post("/v1/outcomes", batch);
    assert_eq!((st, v["recorded"].as_u64()), (200, Some(3)), "{v}");
    // All or none.
    let (st, _) = post(
        "/v1/outcomes",
        json!({"outcomes": [{"id": id3, "result": "bad"}, {"id": "nope", "result": "bad"}]}),
    );
    assert_eq!(st, 404);
    let outs = jsonl(&pdir.join("outcomes.jsonl"));
    assert_eq!(outs.len(), 4);
    assert_eq!(
        outs[0],
        json!({"id": id1, "time": outs[0]["time"], "reward": -1.0, "kind": "tests", "note": "2 failed"})
    );
    assert_eq!(
        (outs[3]["id"].as_str(), outs[3]["reward"].as_f64(), &outs[3]["kind"]),
        (Some(id2.as_str()), Some(-1.0), &Value::Null)
    );
    assert_eq!(jsonl(&learn.join("q/other/outcomes.jsonl")).len(), 2);

    // The listing: newest first, paged, filtered, with each episode's last outcome.
    let get = |path: &str| {
        let (st, body) = request(a, "GET", path, "");
        assert_eq!(st, 200, "{body}");
        json_of(&body)
    };
    let all = get("/v1/episodes");
    assert_eq!((all["data"].as_array().unwrap().len(), all["has_more"].as_bool()), (5, Some(false)));
    let dev = get("/v1/episodes?profile=dev&order=asc");
    let d = dev["data"].as_array().unwrap();
    assert_eq!(d.len(), 3);
    assert_eq!(
        (d[0]["id"].as_str(), d[0]["outcomes"].as_u64(), d[0]["outcome"]["reward"].as_f64()),
        (Some(id1.as_str()), Some(2), Some(0.5))
    );
    assert_eq!((d[0]["outcome"]["kind"].as_str(), d[1]["outcome"].is_null()), (Some("build"), true));
    assert!(d[0]["prompt_tokens"].as_u64().unwrap() > 20 && d[0]["completion_tokens"].as_u64().unwrap() > 0);
    let page = get("/v1/episodes?limit=2");
    assert_eq!(
        (page["data"].as_array().unwrap().len(), page["has_more"].as_bool()),
        (2, Some(true))
    );
    let next = get(&format!("/v1/episodes?limit=10&after={}", page["last_id"].as_str().unwrap()));
    assert_eq!(next["data"].as_array().unwrap().len(), 3);
    assert_eq!(get(&format!("/v1/episodes/{id3}"))["outcome"]["kind"], "user_accepted");
    // The profiles, in the model list and on their own.
    let models = get("/v1/models");
    assert_eq!(models["data"][0]["profiles"], json!(["dev", "other"]));
    let p = get("/v1/models/q/profiles");
    assert_eq!(p["data"][0]["id"], "q@dev");
    assert_eq!(
        (
            p["data"][0]["episodes"].as_u64(),
            p["data"][0]["outcomes"].as_u64(),
            p["data"][0]["adapter"].is_null()
        ),
        (Some(3), Some(4), true)
    );

    // The trainer writes the profile's adapter: later requests of the profile run with it, the
    // model's own requests and the other profile's do not, the weights stay loaded.
    put_adapter(&pdir, 21);
    let r = chat(a, "q@dev", json!({}));
    assert_eq!(text(&r), with1);
    let e = jsonl(&pdir.join("episodes.jsonl")).pop().unwrap();
    assert_eq!(e["adapters"][0]["path"].as_str(), Some(pdir.join("adapter").to_str().unwrap()));
    assert_eq!(e["adapters"][0]["strength"].as_f64(), Some(1.0));
    let v1 = e["adapters"][0]["version"].as_str().unwrap().to_string();
    assert_eq!(get("/v1/models/q/profiles")["data"][0]["adapter"]["version"], v1.as_str());
    assert_eq!(text(&chat(a, "q", json!({}))), plain);
    assert_eq!(text(&chat(a, "q@other", json!({}))), plain);
    assert_eq!(text(&chat(a, "q@dev", json!({}))), with1);
    // Trained again: the new files are picked up.
    put_adapter(&pdir, 22);
    assert_eq!(text(&chat(a, "q@dev", json!({}))), with2);
    let e = jsonl(&pdir.join("episodes.jsonl")).pop().unwrap();
    assert_ne!(e["adapters"][0]["version"].as_str(), Some(v1.as_str()));
    // A second profile's adapter beside it.
    put_adapter(&learn.join("q/other"), 21);
    assert_eq!(text(&chat(a, "q@other", json!({}))), with1);
    assert_eq!(text(&chat(a, "q@dev", json!({}))), with2);
    // An adapter that does not load: the base model answers.
    std::fs::write(pdir.join("adapter/adapter_model.safetensors"), b"not a safetensors file").unwrap();
    assert_eq!(text(&chat(a, "q@dev", json!({}))), plain);
    assert_eq!(jsonl(&pdir.join("episodes.jsonl")).pop().unwrap()["adapters"], json!([]));
    assert_eq!(text(&chat(a, "q@other", json!({}))), with1);
    // Deleted (the logs stay): the base model again.
    put_adapter(&pdir, 22);
    assert_eq!(text(&chat(a, "q@dev", json!({}))), with2);
    let (st, body) = request(a, "DELETE", "/v1/models/q/profiles/dev/adapter", "");
    assert_eq!((st, json_of(&body)["deleted"].as_bool()), (200, Some(true)), "{body}");
    assert!(!pdir.join("adapter").exists() && pdir.join("episodes.jsonl").exists());
    assert_eq!(text(&chat(a, "q@dev", json!({}))), plain);
    let (st, body) = request(a, "DELETE", "/v1/models/q/profiles/dev/adapter", "");
    assert_eq!((st, json_of(&body)["deleted"].as_bool()), (200, Some(false)), "{body}");
    let (st, _) = request(a, "DELETE", "/v1/models/q/profiles/nope/adapter", "");
    assert_eq!(st, 404);
    s.stop().unwrap();

    // After a restart the episodes are still known: outcomes are taken, the listing has them.
    let s = serve(&spec(""));
    let a = s.addr();
    let (st, body) = request(a, "POST", &format!("/v1/episodes/{id4}/outcome"), r#"{"result": "good"}"#);
    assert_eq!(st, 200, "{body}");
    let (st, body) = request(a, "GET", &format!("/v1/episodes/{id4}"), "");
    assert_eq!((st, json_of(&body)["outcome"]["reward"].as_f64()), (200, Some(1.0)), "{body}");
    assert_eq!(text(&chat(a, "q@other", json!({}))), with1);
    s.stop().unwrap();

    // Profiles off; and on a listener that is neither loopback nor keyed.
    let off = serve(&format!("model=q={},device=cpu; learn_dir=none", model.display()));
    let req = json!({"model": "q@dev", "messages": [{"role": "user", "content": "Hi"}]}).to_string();
    let (st, _) = request(off.addr(), "POST", "/v1/chat/completions", &req);
    assert_eq!(st, 404);
    let (st, _) = request(off.addr(), "GET", "/v1/episodes", "");
    assert_eq!(st, 404);
    off.stop().unwrap();
    let mut o = ServerOptions::from_pairs(&spec("")).unwrap();
    o.set("listen", "0.0.0.0:1").unwrap();
    o.set("log", "0").unwrap();
    let api = Api::new(o).unwrap();
    let handle = |method: &str, path: &str, body: &str| {
        let r = api
            .handle(&oxinfer::server::ApiRequest::new(
                method,
                path,
                Some("application/json"),
                body.as_bytes().to_vec(),
            ))
            .collect();
        r.status
    };
    assert_eq!(handle("POST", "/v1/chat/completions", &req), 403);
    assert_eq!(handle("POST", &format!("/v1/episodes/{id1}/outcome"), r#"{"result": "good"}"#), 403);
    assert_eq!(handle("DELETE", "/v1/models/q/profiles/other/adapter", ""), 403);
    assert_eq!(handle("GET", "/v1/episodes", ""), 200);
    std::fs::remove_dir_all(&base).ok();
}

fn subdirs_of(dir: &std::path::Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    v.sort();
    v
}
