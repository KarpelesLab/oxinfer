//! Peers (`ServerOptions::peers`, docs/cluster.md): two servers of this process on loopback
//! ports, one naming the other as its peer, with tiny random checkpoints on the CPU.

#![cfg(feature = "server")]

use oxinfer::server::{Api, ServerOptions, http};
use serde_json::{Value, json};
use std::io::{Read, Write};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// A running server and its handler.
struct Node {
    api: Arc<Api>,
    http: Option<http::Running>,
}

impl Node {
    fn addr(&self) -> SocketAddr {
        self.http.as_ref().unwrap().addr()
    }
    fn stop(&mut self) {
        if let Some(h) = self.http.take() {
            h.stop().unwrap();
        }
    }
}

impl Drop for Node {
    fn drop(&mut self) {
        // (A failed test leaves requests running: the listener would wait for them.)
        for r in self.api.requests() {
            self.api.cancel(&r.id);
        }
        self.stop();
    }
}

fn node_at(listen: &str, opts: &str) -> Node {
    let mut o = ServerOptions::from_pairs(opts).unwrap();
    o.set("listen", listen).unwrap();
    o.set("log", "0").unwrap();
    if !opts.contains("learn_dir") {
        o.set("learn_dir", "none").unwrap();
    }
    let api = Api::new(o).unwrap();
    let http = http::start(api.clone()).unwrap();
    Node { api, http: Some(http) }
}

fn node(opts: &str) -> Node {
    node_at("127.0.0.1:0", opts)
}

/// A loopback port nothing listens on.
fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

fn dechunk(head: &str, rest: &str) -> String {
    if !head.to_ascii_lowercase().contains("transfer-encoding: chunked") {
        return rest.to_string();
    }
    let (mut out, mut rest) = (String::new(), rest);
    while let Some((size, after)) = rest.split_once("\r\n") {
        let Ok(n) = usize::from_str_radix(size.trim(), 16) else { break };
        if n == 0 || after.len() < n + 2 {
            break;
        }
        out.push_str(&after[..n]);
        rest = &after[n + 2..];
    }
    out
}

/// One HTTP/1.1 request with an optional key; the status and the body.
fn request(addr: SocketAddr, method: &str, path: &str, key: Option<&str>, body: &str) -> (u16, String) {
    request_as(addr, method, path, key.map(|k| format!("Authorization: Bearer {k}\r\n")), body)
}

/// [`request`] with its own header lines (each ending with CRLF).
fn request_as(addr: SocketAddr, method: &str, path: &str, headers: Option<String>, body: &str) -> (u16, String) {
    let mut s = std::net::TcpStream::connect(addr).unwrap();
    let auth = headers.unwrap_or_default();
    write!(
        s,
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\n{auth}Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut raw = Vec::new();
    s.read_to_end(&mut raw).unwrap();
    let text = String::from_utf8_lossy(&raw).into_owned();
    let (head, rest) = text.split_once("\r\n\r\n").unwrap();
    (head.split(' ').nth(1).unwrap().parse().unwrap(), dechunk(head, rest))
}

fn json_of(body: &str) -> Value {
    serde_json::from_str(body).unwrap_or_else(|e| panic!("{e}: {body}"))
}

fn ids(v: &Value) -> Vec<String> {
    v["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap().to_string())
        .collect()
}

fn tiny_chat(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("oxinfer-cluster-{tag}-{}", std::process::id()));
    oxinfer::inkling::tiny::write_tiny(&dir, 5).unwrap();
    dir
}

fn wait_for(what: &str, mut ok: impl FnMut() -> bool) {
    let t0 = Instant::now();
    while !ok() {
        assert!(t0.elapsed() < Duration::from_secs(120), "{what}: not after 120 s");
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// The model list is the union, a model only the peer has is run there (a complete response
/// and a streamed one), a model both have keeps a conversation on one machine, and requests
/// without the peer's key are refused by it.
#[test]
fn union_proxying_and_auth() {
    let dir = tiny_chat("union");
    let m = dir.display();
    let b = node(&format!("api_key=link; model=both={m},device=cpu; model=only-b={m},device=cpu"));
    let a = node(&format!(
        "model=both={m},device=cpu; model=only-a=/nonexistent/model; peer={},key=link",
        b.addr()
    ));
    let peer = b.addr().to_string();

    // The peer takes nothing without its key: not from clients, not as a peer.
    let chat = |model: &str, text: &str| {
        json!({"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": 8, "temperature": 0}).to_string()
    };
    assert_eq!(request(b.addr(), "GET", "/v1/models", None, "").0, 401);
    assert_eq!(request(b.addr(), "GET", "/v1/cluster", None, "").0, 401);
    assert_eq!(
        request(b.addr(), "POST", "/v1/chat/completions", Some("wrong"), &chat("only-b", "hi")).0,
        401
    );

    // The union, with where each model is listed.
    let (st, body) = request(a.addr(), "GET", "/v1/models", None, "");
    assert_eq!(st, 200, "{body}");
    let v = json_of(&body);
    assert_eq!(ids(&v), ["both", "only-a", "only-b"], "{v}");
    let nodes = |id: &str| v["data"].as_array().unwrap().iter().find(|m| m["id"] == id).unwrap()["nodes"].clone();
    assert_eq!(nodes("both"), json!(["local", peer]));
    assert_eq!(nodes("only-a"), json!(["local"]));
    assert_eq!(nodes("only-b"), json!([peer]));
    let (st, body) = request(a.addr(), "GET", "/v1/models/only-b", None, "");
    assert_eq!((st, &json_of(&body)["kind"]), (200, &json!("chat")), "{body}");

    // A model only the peer has: the same reply as from the peer itself.
    let (st, direct) = request(b.addr(), "POST", "/v1/chat/completions", Some("link"), &chat("only-b", "hello"));
    assert_eq!(st, 200, "{direct}");
    let (st, via) = request(a.addr(), "POST", "/v1/chat/completions", None, &chat("only-b", "hello"));
    assert_eq!(st, 200, "{via}");
    let (d, v) = (json_of(&direct), json_of(&via));
    assert_eq!(d["choices"][0]["message"], v["choices"][0]["message"]);
    assert_eq!(d["usage"]["completion_tokens"], v["usage"]["completion_tokens"]);
    assert_eq!(a.api.status()["peers"][0]["forwarded"], 1);
    assert_eq!(a.api.status()["peers"][0]["up"], true);
    // Streamed: the peer's events as they are, to the end.
    let mut req = json_of(&chat("only-b", "hello"));
    req["stream"] = json!(true);
    let (st, sse) = request(a.addr(), "POST", "/v1/chat/completions", None, &req.to_string());
    assert_eq!(st, 200, "{sse}");
    assert!(sse.trim_end().ends_with("data: [DONE]"), "{sse}");
    let text: String = sse
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .filter_map(|d| json_of(d)["choices"][0]["delta"]["content"].as_str().map(str::to_string))
        .collect();
    assert_eq!(text, d["choices"][0]["message"]["content"].as_str().unwrap_or_default());
    // Its errors too, as the peer words them.
    let bad = json!({"model": "only-b", "messages": "nope"}).to_string();
    let (st, body) = request(a.addr(), "POST", "/v1/chat/completions", None, &bad);
    let (st_b, body_b) = request(b.addr(), "POST", "/v1/chat/completions", Some("link"), &bad);
    assert_eq!((st, json_of(&body)), (st_b, json_of(&body_b)));
    assert_eq!(st, 400, "{body}");
    // Other endpoints with a model go the same way (the tokens of a prompt, counted there).
    let count = json!({"model": "only-b", "messages": [{"role": "user", "content": "hello there"}]}).to_string();
    let (st, body) = request(a.addr(), "POST", "/v1/messages/count_tokens", None, &count);
    assert_eq!(st, 200, "{body}");
    assert!(json_of(&body)["input_tokens"].as_u64() > Some(0), "{body}");

    // A model both have: the first request runs here (nothing is loaded anywhere, this machine
    // first), and the conversation stays here.
    let before = a.api.status()["peers"][0]["forwarded"].as_u64().unwrap();
    let turn1 = json!({"model": "both", "max_tokens": 4, "temperature": 0, "messages": [
        {"role": "system", "content": "You are terse."}, {"role": "user", "content": "one"}]});
    let (st, body) = request(a.addr(), "POST", "/v1/chat/completions", None, &turn1.to_string());
    assert_eq!(st, 200, "{body}");
    let mut turn2 = turn1.clone();
    let msgs = turn2["messages"].as_array_mut().unwrap();
    msgs.push(json!({"role": "assistant", "content": "x"}));
    msgs.push(json!({"role": "user", "content": "two"}));
    for _ in 0..3 {
        let (st, body) = request(a.addr(), "POST", "/v1/chat/completions", None, &turn2.to_string());
        assert_eq!(st, 200, "{body}");
    }
    assert_eq!(
        a.api.status()["peers"][0]["forwarded"].as_u64().unwrap(),
        before,
        "the conversation left its machine"
    );
    assert!(a.api.models().iter().any(|(id, _, loaded)| id == "both" && *loaded));

    // The peer's view of itself, and the front's view of the peer.
    let (st, body) = request(b.addr(), "GET", "/v1/cluster", Some("link"), "");
    let v = json_of(&body);
    assert_eq!((st, &v["object"]), (200, &json!("cluster.node")), "{body}");
    assert_eq!(v["models"][1]["id"], "only-b");
    assert_eq!(
        (&v["models"][1]["loaded"], &v["models"][1]["parallel"]),
        (&json!(true), &json!(4)),
        "{v}"
    );
    a.api.refresh_peers();
    let p = &a.api.status()["peers"][0];
    assert_eq!((&p["addr"], &p["error"]), (&json!(peer), &Value::Null), "{p}");
    assert_eq!(
        p["models"][1],
        json!({"id": "only-b", "kind": "chat", "loaded": true, "requests": 0}),
        "{p}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// A peer that is not there costs one warning and no waiting; a model only it listed is
/// answered with a 502 while it is away; it is used again once it answers.
#[test]
fn dead_peer_and_recovery() {
    let dir = tiny_chat("dead");
    let m = dir.display();
    let port = free_port();
    let t0 = Instant::now();
    let a = node(&format!("api_key=link; model=both={m},device=cpu; peer=127.0.0.1:{port}"));
    assert!(
        t0.elapsed() < Duration::from_secs(10),
        "starting with a dead peer took {:?}",
        t0.elapsed()
    );
    let health = || json_of(&request(a.addr(), "GET", "/health", None, "").1);
    let h = health();
    assert_eq!(h["degraded"], true, "{h}");
    let w = h["warnings"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|w| w.as_str().unwrap().contains("peer 127.0.0.1"))
        .count();
    assert_eq!(w, 1, "{h}");
    let p = &a.api.status()["peers"][0];
    assert_eq!(p["up"], false, "{p}");
    assert!(p["error"].as_str().unwrap().contains("unreachable"), "{p}");
    // Its own models are served as without peers.
    let chat = |model: &str| json!({"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}).to_string();
    let t0 = Instant::now();
    let (st, body) = request(a.addr(), "POST", "/v1/chat/completions", Some("link"), &chat("both"));
    assert_eq!(st, 200, "{body}");
    assert!(t0.elapsed() < Duration::from_secs(60));
    assert_eq!(ids(&json_of(&request(a.addr(), "GET", "/v1/models", Some("link"), "").1)), ["both"]);

    // The peer starts: listed and used from the next poll on.
    let mut b = node_at(
        &format!("127.0.0.1:{port}"),
        &format!("api_key=link; model=both={m},device=cpu; model=only-b={m},device=cpu"),
    );
    a.api.refresh_peers();
    assert_eq!(health()["degraded"], oxinfer::gpu::degraded());
    assert_eq!(
        ids(&json_of(&request(a.addr(), "GET", "/v1/models", Some("link"), "").1)),
        ["both", "only-b"]
    );
    let (st, body) = request(a.addr(), "POST", "/v1/chat/completions", Some("link"), &chat("only-b"));
    assert_eq!(st, 200, "{body}");

    // It goes away again, without a poll in between: the request that finds it gone runs here
    // when the model is here too, and says so when it is not.
    b.stop();
    drop(b);
    let (st, body) = request(a.addr(), "POST", "/v1/chat/completions", Some("link"), &chat("only-b"));
    assert_eq!((st, &json_of(&body)["error"]["code"]), (502, &json!("peer_unavailable")), "{body}");
    assert_eq!(a.api.status()["peers"][0]["up"], false);
    let (st, body) = request(a.addr(), "POST", "/v1/chat/completions", Some("link"), &chat("only-b"));
    assert_eq!((st, &json_of(&body)["error"]["type"]), (502, &json!("peer_error")), "{body}");
    let (st, body) = request(a.addr(), "POST", "/v1/chat/completions", Some("link"), &chat("both"));
    assert_eq!(st, 200, "{body}");
    assert_eq!(ids(&json_of(&request(a.addr(), "GET", "/v1/models", Some("link"), "").1)), ["both"]);
    std::fs::remove_dir_all(&dir).ok();
}

/// A peer without an API key, and one with another key, are not used; a server with peers and
/// no key for them does not start.
#[test]
fn peers_need_the_key() {
    let open = node("model=m=/nonexistent/model");
    let keyed = node("api_key=theirs; model=k=/nonexistent/model");
    let a = node(&format!(
        "model=mine=/nonexistent/model; peer={},key=x; peer={},key=not-theirs",
        open.addr(),
        keyed.addr()
    ));
    let v = json_of(&request(a.addr(), "GET", "/v1/models", None, "").1);
    assert_eq!(ids(&v), ["mine"], "{v}");
    let peers = a.api.status()["peers"].clone();
    assert_eq!((&peers[0]["up"], &peers[1]["up"]), (&json!(false), &json!(false)), "{peers}");
    assert!(peers[0]["error"].as_str().unwrap().contains("without an API key"), "{peers}");
    assert!(peers[1]["error"].as_str().unwrap().contains("refuses this server's key"), "{peers}");
    let chat = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]}).to_string();
    let (st, body) = request(a.addr(), "POST", "/v1/chat/completions", None, &chat);
    assert_eq!((st, &json_of(&body)["error"]["code"]), (404, &json!("model_not_found")), "{body}");

    let mut o = ServerOptions::from_pairs("peer=127.0.0.1:9").unwrap();
    o.set("log", "0").unwrap();
    let e = Api::new(o).err().expect("a peer without a key").to_string();
    assert!(e.contains("no key"), "{e}");
    // The front's own key is the peers' when none is given.
    let both = node(&format!("api_key=theirs; peer={}", keyed.addr()));
    assert_eq!(both.api.status()["peers"][0]["up"], true);
    // Two servers that name each other do not send a request back and forth.
    let x_port = free_port();
    let y = node(&format!("api_key=s; model=only-y=/nonexistent/model; peer=127.0.0.1:{x_port}"));
    let x = node_at(&format!("127.0.0.1:{x_port}"), &format!("api_key=s; peer={}", y.addr()));
    y.api.refresh_peers();
    let chat = json!({"model": "only-y", "messages": [{"role": "user", "content": "hi"}]}).to_string();
    let (st, body) = request(x.addr(), "POST", "/v1/chat/completions", Some("s"), &chat);
    assert_eq!(st, 500, "the peer's own failure to load a model that does not exist: {body}");
}

/// Requests for a model both machines have are spread by load, a cancel on the front stops the
/// peer's work, and so does a client that goes away. With a tiny random SD 3.5, whose requests
/// last as long as they are given steps.
#[test]
fn balancing_and_cancel() {
    let dir = std::env::temp_dir().join(format!("oxinfer-cluster-sd3-{}", std::process::id()));
    oxinfer::sd3::tiny::write_tiny_release(&dir).unwrap();
    let m = dir.display();
    let b = node(&format!("api_key=link; model=sd3={m}"));
    let a = node(&format!("model=sd3={m}; peer={},key=link", b.addr()));
    let addr = a.addr();
    let slow = |stream: bool| {
        json!({"model": "sd3", "prompt": "a fox", "size": "256x256", "steps": 100000, "seed": 1, "stream": stream}).to_string()
    };
    let spawn = |body: String| std::thread::spawn(move || request(addr, "POST", "/v1/images/generations", None, &body));
    let running = |api: &Arc<Api>| api.requests().into_iter().filter(|r| r.step.is_some_and(|s| s.0 > 0)).count();

    // The first runs here; the second, while this machine's model is busy, on the peer.
    let first = spawn(slow(false));
    wait_for("the first request denoising here", || running(&a.api) == 1);
    assert!(a.api.requests()[0].peer.is_none());
    assert_eq!(b.api.requests().len(), 0);
    let second = spawn(slow(true));
    wait_for("the second request denoising on the peer", || running(&b.api) == 1);
    let reqs = a.api.requests();
    let fwd = reqs.iter().find(|r| r.peer.is_some()).expect("a forwarded request");
    assert_eq!(fwd.peer.as_deref(), Some(b.addr().to_string().as_str()));
    assert_eq!(a.api.status()["peers"][0]["inflight"], 1);

    // Cancelled on the front: its client gets the error event, and the peer stops.
    assert!(a.api.cancel(&fwd.id));
    let (st, body) = second.join().unwrap();
    assert_eq!(st, 200, "{body}");
    assert!(body.contains("\"cancelled\""), "{body}");
    wait_for("the peer's request to stop", || b.api.requests().is_empty());
    assert_eq!(a.api.status()["peers"][0]["inflight"], 0);

    // A client that goes away in the middle of a forwarded stream: the peer stops too.
    let mut s = std::net::TcpStream::connect(addr).unwrap();
    let body = slow(true);
    write!(
        s,
        "POST /v1/images/generations HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    wait_for("the third request denoising on the peer", || running(&b.api) == 1);
    let mut some = [0u8; 64];
    s.read_exact(&mut some).unwrap();
    drop(s);
    wait_for("the peer's request to stop after its client left", || b.api.requests().is_empty());
    wait_for("the front to forget it", || a.api.requests().len() == 1);

    // The one running here all along is cancelled as before.
    let id = a.api.requests()[0].id.clone();
    assert!(a.api.cancel(&id));
    assert_eq!(first.join().unwrap().0, 499);
    wait_for("both machines idle", || a.api.requests().is_empty() && b.api.requests().is_empty());

    // Two requests arriving together: one each (the second is counted against the first's
    // machine before the first has started there).
    for round in 0..3 {
        let go = Arc::new(std::sync::Barrier::new(2));
        let pair: Vec<_> = (0..2)
            .map(|_| {
                let (go, body) = (go.clone(), slow(false));
                std::thread::spawn(move || {
                    go.wait();
                    request(addr, "POST", "/v1/images/generations", None, &body)
                })
            })
            .collect();
        wait_for("two requests in flight", || a.api.requests().len() == 2);
        wait_for("one of them on the peer", || b.api.requests().len() == 1);
        let here = a.api.requests().iter().filter(|r| r.peer.is_none()).count();
        assert_eq!(here, 1, "round {round}: {:?}", a.api.requests());
        for r in a.api.requests() {
            assert!(a.api.cancel(&r.id));
        }
        for t in pair {
            assert_eq!(t.join().unwrap().0, 499);
        }
        wait_for("both machines idle again", || {
            a.api.requests().is_empty() && b.api.requests().is_empty()
        });
    }

    // A conversation (named by its client) that started on the peer, this machine being busy,
    // goes back there when this machine is idle again; a request without the name runs here.
    let quick = json!({"model": "sd3", "prompt": "a fox", "size": "64x64", "steps": 2, "seed": 1}).to_string();
    let named = |body: &str| request_as(addr, "POST", "/v1/images/generations", Some("X-Session-Id: ses_1\r\n".into()), body);
    let forwarded = || a.api.status()["peers"][0]["forwarded"].as_u64().unwrap();
    let hold = spawn(slow(false));
    wait_for("a request denoising here", || running(&a.api) == 1 && b.api.requests().is_empty());
    let n = forwarded();
    assert_eq!(named(&quick).0, 200);
    assert_eq!(forwarded(), n + 1, "this machine was busy");
    for r in a.api.requests() {
        assert!(a.api.cancel(&r.id));
    }
    assert_eq!(hold.join().unwrap().0, 499);
    wait_for("this machine idle", || a.api.requests().is_empty());
    assert_eq!(named(&quick).0, 200);
    assert_eq!(forwarded(), n + 2, "the conversation's machine");
    assert_eq!(request(addr, "POST", "/v1/images/generations", None, &quick).0, 200);
    assert_eq!(forwarded(), n + 2, "no conversation: this machine first");
    std::fs::remove_dir_all(&dir).ok();
}

/// A model no machine lists goes to the machine with the memory for it; a profile always to the
/// machine that has it.
#[test]
fn models_on_request_and_profiles() {
    // A repo in the cache both machines read (its weights are not real: loading it fails, on
    // the machine that was chosen).
    let base = std::env::temp_dir().join(format!("oxinfer-cluster-auto-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let m = base.join("hub/org--model");
    std::fs::create_dir_all(&m).unwrap();
    std::fs::write(m.join("config.json"), r#"{"model_type": "bert"}"#).unwrap();
    std::fs::write(m.join("modules.json"), "[]").unwrap();
    std::fs::write(m.join("model.safetensors"), vec![0u8; 3 << 20]).unwrap();
    let cache = format!("cache_dir={}; auto_models=1", base.join("hub").display());
    let embed = json!({"model": "org/model", "input": "hi"}).to_string();
    let listed = |n: &Node| n.api.models().iter().any(|(id, ..)| id == "org/model");
    {
        // The front has no memory for it, the peer has.
        let b = node(&format!("api_key=k; {cache}"));
        let a = node(&format!("{cache}; memory_gb=0.001; peer={},key=k", b.addr()));
        let (st, body) = request(a.addr(), "POST", "/v1/embeddings", None, &embed);
        assert_eq!(st, 500, "{body}");
        assert!(body.contains("load org/model"), "{body}");
        assert!(listed(&b) && !listed(&a));
    }
    {
        // Both have: the front takes it.
        let b = node(&format!("api_key=k; {cache}"));
        let a = node(&format!("{cache}; peer={},key=k", b.addr()));
        let (st, body) = request(a.addr(), "POST", "/v1/embeddings", None, &embed);
        assert_eq!(st, 500, "{body}");
        assert!(listed(&a) && !listed(&b));
    }
    {
        // Neither has: the last machine's refusal.
        let b = node(&format!("api_key=k; {cache}; memory_gb=0.001"));
        let a = node(&format!("{cache}; memory_gb=0.001; peer={},key=k", b.addr()));
        let (st, body) = request(a.addr(), "POST", "/v1/embeddings", None, &embed);
        assert_eq!(st, 507, "{body}");
        assert!(!listed(&a) && !listed(&b));
    }

    // Profiles: `p1` exists on the peer only, `p2` nowhere (it is then the front's).
    let dir = tiny_chat("profiles");
    let (la, lb) = (base.join("learn-a"), base.join("learn-b"));
    std::fs::create_dir_all(lb.join("both/p1")).unwrap();
    let mdl = format!("model=both={},device=cpu", dir.display());
    let b = node(&format!("api_key=k; {mdl}; learn_dir={}", lb.display()));
    let a = node(&format!("api_key=k; {mdl}; learn_dir={}; peer={}", la.display(), b.addr()));
    let chat = |model: &str| {
        let body = json!({"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}).to_string();
        let (st, body) = request(a.addr(), "POST", "/v1/chat/completions", Some("k"), &body);
        assert_eq!(st, 200, "{model}: {body}");
    };
    let forwarded = || a.api.status()["peers"][0]["forwarded"].as_u64().unwrap();
    for n in 1..=3 {
        chat("both@p1");
        assert_eq!(forwarded(), n, "both@p1 runs where the profile is");
    }
    for _ in 0..3 {
        chat("both@p2");
    }
    assert_eq!(forwarded(), 3, "both@p2 stays on the front");
    assert!(la.join("both/p2").is_dir() && !lb.join("both/p2").exists());
    assert!(!la.join("both/p1").exists());
    drop((a, b));
    std::fs::remove_dir_all(&base).ok();
    std::fs::remove_dir_all(&dir).ok();
}
