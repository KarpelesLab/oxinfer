//! The web GUI's opencode endpoints (`/api/web/opencode`) over real HTTP, with opencode's
//! configuration in a temporary `XDG_CONFIG_HOME` (the only test of this binary, so that setting
//! the environment races nothing).

#![cfg(feature = "server")]

use oxinfer::server::{Api, ApiRequest, ServerOptions, http};
use serde_json::{Value, json};
use std::io::{Read, Write};
use std::net::SocketAddr;

fn request(addr: SocketAddr, method: &str, path: &str, headers: &[&str], body: &str) -> (u16, Value) {
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
    let status = head.split(' ').nth(1).unwrap().parse().unwrap();
    (status, serde_json::from_str(rest).unwrap_or_else(|_| json!({"raw": rest})))
}

const CONFIG: &str = r#"{
  // the user's comment
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "other": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://10.0.0.1/v1", "apiKey": "theirs" },
      "models": { "m": { "name": "M" } },
    },
  },
  "model": "oxinfer/c",
}
"#;

#[test]
fn opencode_endpoints() {
    let base = std::env::temp_dir().join(format!("oxinfer-opencode-server-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let xdg = base.join("xdg");
    std::fs::create_dir_all(&xdg).unwrap();
    // SAFETY: the only test of this binary, before any server thread starts.
    unsafe { std::env::set_var("XDG_CONFIG_HOME", &xdg) };
    // A chat model folder: a vision tower, a template with thinking and tools, a 40960 context.
    let model = base.join("chat");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::write(
        model.join("config.json"),
        r#"{"model_type": "qwen3", "max_position_embeddings": 40960, "vision_config": {"depth": 1}}"#,
    )
    .unwrap();
    std::fs::write(
        model.join("tokenizer_config.json"),
        r#"{"chat_template": "{% if tools %}{{ tools }}{% endif %}<think>"}"#,
    )
    .unwrap();

    let mut o = ServerOptions::from_pairs(&format!("web=1; log=0; model=c={}", model.display())).unwrap();
    o.set("listen", "127.0.0.1:0").unwrap();
    let s = http::start(Api::new(o).unwrap()).unwrap();
    let a = s.addr();
    let host = format!("Host: localhost:{}", a.port());
    let local = [host.as_str(), "Content-Type: application/json"];

    // No opencode directory yet.
    let (st, v) = request(a, "GET", "/api/web/opencode", &local, "");
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["available"], true);
    assert_eq!(v["detected"], false);
    let (st, v) = request(a, "POST", "/api/web/opencode/add", &local, r#"{"model": "c"}"#);
    assert_eq!((st, v["error"]["code"].as_str()), (404, Some("opencode_not_found")), "{v}");

    let dir = xdg.join("opencode");
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("opencode.json");
    std::fs::write(&file, CONFIG).unwrap();
    let (st, v) = request(a, "GET", "/api/web/opencode", &local, "");
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["detected"], true);
    assert_eq!(v["provider"], false);
    assert_eq!(v["base_url"], format!("http://127.0.0.1:{}/v1", a.port()));
    let m = &v["models"][0];
    assert_eq!(m["id"], "c");
    assert_eq!(m["state"], "absent");
    assert_eq!(m["source"], "files");
    assert_eq!(
        m["capabilities"],
        json!({"vision": true, "thinking": true, "tools": true, "context": 40960})
    );
    assert_eq!(m["entry"]["limit"], json!({"context": 40960, "output": 10240}));
    assert!(!v.to_string().contains("theirs"), "no key of the file is returned");

    // Add: the entry and the provider, the rest of the file kept, a backup made.
    let (st, v) = request(a, "POST", "/api/web/opencode/add", &local, r#"{"model": "c"}"#);
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["changed"], true);
    let backup = std::path::PathBuf::from(v["backup"].as_str().unwrap());
    assert_eq!(std::fs::read_to_string(&backup).unwrap(), CONFIG);
    let text = std::fs::read_to_string(&file).unwrap();
    assert!(text.contains("// the user's comment"), "{text}");
    let cfg = oxinfer::opencode::parse_value(&text).unwrap();
    let p = &cfg["provider"]["oxinfer"];
    assert_eq!(p["npm"], "@ai-sdk/openai-compatible");
    assert_eq!(p["options"], json!({"baseURL": format!("http://127.0.0.1:{}/v1", a.port())}));
    let e = &p["models"]["c"];
    assert_eq!(
        (&e["reasoning"], &e["tool_call"], &e["attachment"]),
        (&json!(true), &json!(true), &json!(true))
    );
    assert_eq!(e["modalities"], json!({"input": ["text", "image"], "output": ["text"]}));
    assert_eq!(cfg["provider"]["other"]["options"]["apiKey"], "theirs");
    let (_, v) = request(a, "GET", "/api/web/opencode", &local, "");
    assert_eq!(
        (v["models"][0]["present"].clone(), v["models"][0]["up_to_date"].clone()),
        (json!(true), json!(true))
    );
    // Again: nothing written.
    let (_, v) = request(a, "POST", "/api/web/opencode/add", &local, r#"{"model": "c"}"#);
    assert_eq!((v["changed"].clone(), v["backup"].clone()), (json!(false), Value::Null), "{v}");
    let (st, v) = request(a, "POST", "/api/web/opencode/add", &local, r#"{"model": "nope"}"#);
    assert_eq!(st, 404, "{v}");
    // A cross-site page cannot do it.
    let (st, _) = request(
        a,
        "POST",
        "/api/web/opencode/remove",
        &[host.as_str(), "Content-Type: application/json", "Origin: http://evil.example"],
        r#"{"model": "c"}"#,
    );
    assert_eq!(st, 403);

    // Remove: the provider goes with its last model; opencode's default model is reported, kept.
    let (st, v) = request(a, "POST", "/api/web/opencode/remove", &local, r#"{"model": "c"}"#);
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["provider_removed"], true);
    assert_eq!(v["default_model"], json!(["model"]));
    assert!(v["message"].as_str().unwrap().contains("oxinfer/c"), "{v}");
    let cfg = oxinfer::opencode::parse_value(&std::fs::read_to_string(&file).unwrap()).unwrap();
    assert!(cfg["provider"].get("oxinfer").is_none());
    assert_eq!(cfg["provider"]["other"]["models"]["m"]["name"], "M");
    assert_eq!(cfg["model"], "oxinfer/c");
    s.stop().unwrap();

    // A non-loopback listener without a key: read-only, nothing offered (no socket is bound).
    let before = std::fs::read_to_string(&file).unwrap();
    let o = ServerOptions::from_pairs(&format!("web=1; log=0; listen=0.0.0.0:0; model=c={}", model.display())).unwrap();
    let api = Api::new(o).unwrap();
    let r = api
        .handle(&ApiRequest::post_json("/api/web/opencode/add", r#"{"model": "c"}"#))
        .collect();
    assert_eq!(r.status, 403);
    assert_eq!(serde_json::from_slice::<Value>(&r.body).unwrap()["error"]["code"], "read_only");
    let r = api.handle(&ApiRequest::new("GET", "/api/web/opencode", None, vec![])).collect();
    let v: Value = serde_json::from_slice(&r.body).unwrap();
    assert_eq!((v["available"].clone(), v["detected"].clone()), (json!(false), json!(false)));
    // With a key: only a client on this machine.
    let o = ServerOptions::from_pairs(&format!("web=1; log=0; api_key=k; listen=0.0.0.0:0; model=c={}", model.display())).unwrap();
    let api = Api::new(o).unwrap();
    let req = |peer: &str| {
        let mut r = ApiRequest::post_json("/api/web/opencode/add", r#"{"model": "c"}"#);
        r.authorization = Some("Bearer k".into());
        r.peer = Some(peer.into());
        r
    };
    let r = api.handle(&req("192.168.1.9:5000")).collect();
    assert_eq!(r.status, 403);
    assert_eq!(
        serde_json::from_slice::<Value>(&r.body).unwrap()["error"]["code"],
        "opencode_local_only"
    );
    assert_eq!(std::fs::read_to_string(&file).unwrap(), before);
    let r = api.handle(&req("127.0.0.1:5000")).collect();
    assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
    let cfg = oxinfer::opencode::parse_value(&std::fs::read_to_string(&file).unwrap()).unwrap();
    assert_eq!(cfg["provider"]["oxinfer"]["options"]["apiKey"], "k");

    context_cap_profiles_and_plugin(&base, &dir);
    std::fs::remove_dir_all(&base).ok();
}

/// The context written for a model trained for a million tokens (capped unless the model has a
/// `max_context`), the model's entry under a profile, and the outcome plugin's file.
fn context_cap_profiles_and_plugin(base: &std::path::Path, dir: &std::path::Path) {
    let file = dir.join("opencode.json");
    let big = base.join("big");
    std::fs::create_dir_all(&big).unwrap();
    std::fs::write(
        big.join("config.json"),
        r#"{"model_type": "qwen3", "max_position_embeddings": 1048576}"#,
    )
    .unwrap();
    let api = |extra: &str| {
        let learn = base.join("learn");
        let spec = format!("web=1; log=0; learn_dir={}; model=big={}{extra}", learn.display(), big.display());
        Api::new(ServerOptions::from_pairs(&spec).unwrap()).unwrap()
    };
    let call = |api: &std::sync::Arc<Api>, path: &str, body: Option<Value>| -> (u16, Value) {
        let r = match body {
            Some(b) => api.handle(&ApiRequest::post_json(path, &b.to_string())),
            None => api.handle(&ApiRequest::new("GET", path, None, vec![])),
        }
        .collect();
        (r.status, serde_json::from_slice(&r.body).unwrap())
    };
    let limit = |api: &std::sync::Arc<Api>| {
        let (st, v) = call(api, "/api/web/opencode", None);
        assert_eq!(st, 200, "{v}");
        let m = &v["models"][0];
        (m["entry"]["limit"].clone(), m["state"].as_str().unwrap().to_string(), v)
    };
    let entry =
        || oxinfer::opencode::parse_value(&std::fs::read_to_string(&file).unwrap()).unwrap()["provider"]["oxinfer"]["models"].clone();

    // No cap, the default: the trained context.
    std::fs::write(&file, "{}\n").unwrap();
    for extra in ["", "; opencode_context=none"] {
        let (l, state, v) = limit(&api(extra));
        assert_eq!(
            (l, state.as_str(), &v["context_cap"]),
            (json!({"context": 1048576, "output": 32768}), "absent", &Value::Null),
            "{extra}"
        );
    }
    let a = api("");
    let (st, v) = call(&a, "/api/web/opencode/add", Some(json!({"model": "big"})));
    assert_eq!(st, 200, "{v}");
    assert_eq!(entry()["big"]["limit"]["context"], 1048576);
    // A cap: that entry needs an update, which writes 131072.
    let a = api("; opencode_context=131072");
    let (l, state, v) = limit(&a);
    assert_eq!(
        (l, state.as_str(), &v["context_cap"]),
        (json!({"context": 131072, "output": 32768}), "outdated", &json!(131072))
    );
    let (_, v) = call(&a, "/api/web/opencode/add", Some(json!({"model": "big"})));
    assert_eq!(v["changed"], true, "{v}");
    assert_eq!(entry()["big"]["limit"], json!({"context": 131072, "output": 32768}));
    assert_eq!(limit(&a).1, "current");
    // Another cap; and the model's own max_context, which wins over any cap.
    assert_eq!(
        limit(&api("; opencode_context=65536")).0,
        json!({"context": 65536, "output": 16384})
    );
    for extra in [",max_context=200000", ",max_context=200000; opencode_context=65536"] {
        assert_eq!(limit(&api(extra)).0, json!({"context": 200000, "output": 32768}), "{extra}");
    }
    assert!(ServerOptions::from_pairs("opencode_context=many").is_err());

    // The model under a profile: its own entry, and the plugin's file.
    let plugin = dir.join("plugins").join("oxinfer-outcomes.js");
    let (_, _, v) = limit(&a);
    assert_eq!((&v["profiles"], &v["plugin"]["installed"]), (&json!(true), &json!(false)), "{v}");
    assert_eq!(v["models"][0]["profiles"], json!([]));
    let (st, v) = call(&a, "/api/web/opencode/add", Some(json!({"model": "big", "profile": "my repo"})));
    assert_eq!((st, v["error"]["param"].as_str()), (400, Some("profile")), "{v}");
    let (st, v) = call(&a, "/api/web/opencode/add", Some(json!({"model": "big", "profile": "repo"})));
    assert_eq!(st, 200, "{v}");
    assert_eq!(
        (v["model"].as_str(), v["profile"].as_str(), &v["plugin"]["changed"]),
        (Some("big@repo"), Some("repo"), &json!(true)),
        "{v}"
    );
    assert_eq!(std::fs::read_to_string(&plugin).unwrap(), oxinfer::opencode::PLUGIN_SOURCE);
    let e = entry();
    assert_eq!(e["big@repo"]["name"], "big@repo (oxinfer)");
    assert_eq!(e["big@repo"]["limit"], e["big"]["limit"]);
    let (_, _, v) = limit(&a);
    assert_eq!(
        v["models"][0]["profiles"],
        json!([{"id": "big@repo", "profile": "repo", "state": "current", "up_to_date": true, "entry": v["models"][0]["profiles"][0]["entry"]}])
    );
    assert_eq!((&v["other"], &v["plugin"]["up_to_date"]), (&json!([]), &json!(true)), "{v}");
    // Without the plugin when asked; an edited plugin file is another version, written again.
    std::fs::write(&plugin, "// edited\n").unwrap();
    let (_, v) = call(
        &a,
        "/api/web/opencode/add",
        Some(json!({"model": "big", "profile": "two", "plugin": false})),
    );
    assert!(v.get("plugin").is_none(), "{v}");
    assert_eq!(limit(&a).2["plugin"]["state"], "outdated");
    let (st, v) = call(&a, "/api/web/opencode/plugin", Some(json!({})));
    assert_eq!((st, &v["changed"]), (200, &json!(true)), "{v}");
    assert_eq!(std::fs::read_to_string(&plugin).unwrap(), oxinfer::opencode::PLUGIN_SOURCE);
    let (_, v) = call(&a, "/api/web/opencode/plugin", Some(json!({"install": true})));
    assert_eq!(v["changed"], false, "{v}");
    // Removing a profile's entry leaves the model's and the plugin; the plugin goes when asked.
    let (st, v) = call(&a, "/api/web/opencode/remove", Some(json!({"model": "big", "profile": "repo"})));
    assert_eq!(
        (st, &v["changed"], v["model"].as_str()),
        (200, &json!(true), Some("big@repo")),
        "{v}"
    );
    let e = entry();
    assert!(
        e.get("big@repo").is_none() && e["big"].is_object() && e["big@two"].is_object(),
        "{e}"
    );
    assert!(plugin.is_file());
    let (_, v) = call(&a, "/api/web/opencode/plugin", Some(json!({"install": false})));
    assert_eq!(v["changed"], true, "{v}");
    assert!(!plugin.exists());
    // A server that does not serve profiles does not write one.
    let off = Api::new(ServerOptions::from_pairs(&format!("web=1; log=0; learn_dir=none; model=big={}", big.display())).unwrap()).unwrap();
    let (st, v) = call(&off, "/api/web/opencode/add", Some(json!({"model": "big", "profile": "repo"})));
    assert_eq!((st, v["error"]["code"].as_str()), (409, Some("profiles_off")), "{v}");
    assert_eq!(limit(&off).2["profiles"], false);
    assert!(!plugin.exists() && entry().get("big@repo").is_none());
}
