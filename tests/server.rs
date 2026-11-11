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
