//! The OpenAI-compatible server over real HTTP on a random port. The routing and error checks
//! run by default; the ones with a model are ignored by default (they fetch or take from the
//! cache `Qwen/Qwen3-0.6B` and `BAAI/bge-small-en-v1.5`; override with `OXINFER_E2E_CHAT` /
//! `OXINFER_E2E_EMBED`):
//!
//! ```text
//! cargo test --release --test server -- --include-ignored --nocapture
//! ```
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
    s.stop().unwrap();
    s2.stop().unwrap();
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
