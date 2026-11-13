//! Token-level checks on a small vocabulary: every byte as a token (so any text, and any split
//! of a multi-byte character, can be written), tokens that span several grammar symbols, and
//! special tokens.

use super::*;
use crate::chat::{self, ChatFormat};
use crate::sampler::Sampler;
use crate::tools::ToolCall;
use serde_json::{Map, json};

const MULTI: &[&str] = &[
    "{\"",
    "\":",
    "\": ",
    "\",",
    "\", \"",
    "\"}",
    "\"]}",
    "name",
    "age",
    " true",
    "123",
    "[]",
    "{}",
    "}}",
    "é",
    " \"",
    "\n",
    "\n\n",
    "\":\"",
    "null",
    ",\"",
    "tags",
    "\":[",
    "\"],\"",
    "\"]",
    "true",
    "false",
    "},{",
    "\":{\"",
    "1.5",
    "-",
    "e",
    "Paris",
    "city",
    "{\"name\": \"",
    "\"}\n",
    "<",
    "</",
    ">\n",
];
/// Part of a character: the first two bytes of 日 (E6 97 A5).
const HALF: &[u8] = &[0xe6, 0x97];
const SPECIAL: &[&str] = &[
    "<eos>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<|tool_call>",
    "<tool_call|>",
    "<|\"|>",
    "[TOOL_CALLS]",
    "[ARGS]",
];

fn vocab() -> Arc<Vocab> {
    let mut tokens: Vec<(Vec<u8>, bool)> = (0..=255u8).map(|b| (vec![b], true)).collect();
    tokens.extend(MULTI.iter().map(|t| (t.as_bytes().to_vec(), true)));
    tokens.push((HALF.to_vec(), true));
    let first = tokens.len() as u32;
    tokens.extend(SPECIAL.iter().map(|t| (t.as_bytes().to_vec(), false)));
    let added = SPECIAL.iter().enumerate().map(|(i, t)| (t.to_string(), first + i as u32)).collect();
    Arc::new(Vocab::from_tokens(tokens, added))
}

fn special(v: &Vocab, name: &str) -> u32 {
    v.added.iter().find(|a| a.0 == name).unwrap().1
}

fn multi(name: &str) -> u32 {
    256 + MULTI.iter().position(|t| *t == name).unwrap() as u32
}

/// `text` as tokens: special tokens where it names them, the longest vocabulary token elsewhere.
fn tokenize(v: &Vocab, text: &str) -> Vec<u32> {
    let mut out = Vec::new();
    let mut rest = text.as_bytes();
    while !rest.is_empty() {
        if let Some((t, id)) = v.added.iter().find(|a| rest.starts_with(a.0.as_bytes())) {
            out.push(*id);
            rest = &rest[t.len()..];
            continue;
        }
        let best = (0..v.len() as u32)
            .filter(|&t| v.plain(t) && rest.starts_with(v.bytes(t)))
            .max_by_key(|&t| v.bytes(t).len())
            .unwrap();
        out.push(best);
        rest = &rest[v.bytes(best).len()..];
    }
    out
}

fn compiled(v: &Arc<Vocab>, format: ResponseFormat) -> Arc<Compiled> {
    Arc::new(compile(v, &format, &ToolChoice::Auto, &[], None).unwrap().unwrap())
}

/// Follows `text` token by token, checking at each step that the mask (trie walk, position
/// and state caches) is exactly the set of tokens whose bytes the grammar can follow; returns
/// the constraint after it.
fn follow(c: &Arc<Compiled>, text: &str) -> Constraint {
    let v = c.vocab.clone();
    let eos = special(&v, "<eos>");
    let mut con = Constraint::new(c.clone(), None, false, &[eos]);
    let check = |con: &Constraint, at: &str| {
        let mut logits = vec![0f32; v.len()];
        con.clone().mask(&mut logits);
        for t in 0..v.len() as u32 {
            let direct = if t == eos {
                con.state.contains(&DONE)
            } else {
                !c.advance(&con.state, t).is_empty()
            };
            assert_eq!(
                logits[t as usize] == 0.0,
                direct,
                "token {:?} after {at:?}",
                String::from_utf8_lossy(v.bytes(t))
            );
            assert_eq!(con.allows(t), direct);
        }
    };
    let mut done = String::new();
    check(&con, &done);
    for t in tokenize(&v, text) {
        assert!(con.allows(t), "{:?} after {done:?}", String::from_utf8_lossy(v.bytes(t)));
        con.accept(t);
        done.push_str(&String::from_utf8_lossy(v.bytes(t)));
        check(&con, &done);
    }
    con
}

fn person() -> Value {
    json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "home": {"anyOf": [{"type": "null"}, {"$ref": "#/$defs/place"}]},
        },
        "required": ["name", "age", "tags", "home"],
        "additionalProperties": false,
        "$defs": {"place": {"type": "object", "properties": {"city": {"type": "string"}, "in": {"$ref": "#/$defs/place"}},
                            "required": ["city"], "additionalProperties": false}},
    })
}

#[test]
fn masks_are_exact_along_documents() {
    let v = vocab();
    let c = compiled(&v, ResponseFormat::JsonSchema(person()));
    for doc in [
        r#"{"name":"Ada","age":123,"tags":[],"home":null}"#,
        r#"{"name": "é日\n\"x\"", "age": -7, "tags": ["a", "b"], "home": {"city": "Paris", "in": {"city": "France"}}}"#,
        "{\n  \"name\": \"\",\n  \"age\": 0,\n  \"tags\": [\n    \"tags\"\n  ],\n  \"home\": {\"city\": \"\\u00e9\"}\n}",
    ] {
        let con = follow(&c, doc);
        assert!(con.complete(), "{doc}");
        // Complete: only the stop token may follow.
        let allowed: Vec<u32> = (0..v.len() as u32).filter(|&t| con.allows(t)).collect();
        assert_eq!(allowed, vec![special(&v, "<eos>")]);
    }
    let c = compiled(&v, ResponseFormat::JsonObject);
    for doc in [r#"{"a": [1.5, -2e3, true, false, null, {}], "b": {"c": "\ud83d\ude00"}}"#, "{}"] {
        assert!(follow(&c, doc).complete(), "{doc}");
    }
    // A number at the top may end or go on.
    let c = compiled(&v, ResponseFormat::JsonSchema(json!({"type": "integer"})));
    let con = follow(&c, "12");
    assert!(con.complete() && con.allows(b'3' as u32) && con.allows(special(&v, "<eos>")) && !con.allows(b'.' as u32));
}

#[test]
fn tokens_spanning_symbols() {
    let v = vocab();
    let c = compiled(&v, ResponseFormat::JsonSchema(person()));
    // One token opens the object and the first key's quote; another is a whole key and its string's start.
    let con = follow(&c, "");
    assert!(con.allows(multi("{\"")) && con.allows(multi("{\"name\": \"")) && !con.allows(multi("{}")) && !con.allows(multi("\"}")));
    // Inside the name: the quote may close it with what follows, but not the object.
    let con = follow(&c, r#"{"name":"Ada"#);
    assert!(con.allows(multi("\",")) && con.allows(multi("\", \"")) && con.allows(multi("name")));
    assert!(!con.allows(multi("\"}")) && !con.allows(multi("\"]}")) && !con.allows(multi("\":")) && !con.allows(multi("\"}\n")));
    // The key that follows is fixed: `,"` then `age`, not `name`.
    let con = follow(&c, r#"{"name":"Ada","#);
    assert!(con.allows(b'"' as u32) && con.allows(multi(" \"")) && !con.allows(multi("name")));
    let con = follow(&c, r#"{"name":"Ada",""#);
    assert!(con.allows(multi("age")) && con.allows(b'a' as u32) && !con.allows(multi("name")) && !con.allows(multi("tags")));
    // In the last string of the document, one token closes the string, the array and the object.
    let con = follow(&c, r#"{"name":"","age":1,"tags":["x"],"home":{"city":"P"#);
    assert!(con.allows(multi("\"}")) && con.allows(multi("}}")) && !con.allows(multi("\"]}")) && con.allows(multi("\"}\n")));
    let con = follow(&c, r#"{"name":"","age":1,"tags":["x"],"home":{"city":"P""#);
    assert!(con.allows(multi("}}")) && !con.allows(multi("},{")));
    let con = follow(&c, r#"{"name":"","age":1,"tags":["x"#);
    assert!(con.allows(multi("\"],\"")) && con.allows(multi("\"]")) && !con.allows(multi("\"]}")));
}

#[test]
fn characters_split_across_tokens() {
    let v = vocab();
    let c = compiled(&v, ResponseFormat::JsonObject);
    let half = 256 + MULTI.len() as u32;
    let eos = special(&v, "<eos>");
    let mut con = follow(&c, r#"{"k":"a"#);
    // The first two bytes of a three-byte character, as one token.
    assert!(con.allows(half) && con.allows(0xe6) && !con.allows(0x97) && !con.allows(0xff) && !con.allows(0xc0));
    con.accept(half);
    // Only the byte that completes the character may follow: no quote, no other character.
    let allowed: Vec<u32> = (0..v.len() as u32).filter(|&t| con.allows(t)).collect();
    assert_eq!(allowed, (0x80..=0xbf).collect::<Vec<u32>>());
    assert!(!con.allows(eos));
    con.accept(0xa5);
    assert!(con.allows(b'"' as u32) && con.allows(multi("\"}")) && con.allows(half));
    con.accept(multi("\"}"));
    assert!(con.complete());
    // Byte by byte: a four-byte character, with the second byte's range depending on the first.
    let mut con = follow(&c, r#"{"k":""#);
    con.accept(0xf0);
    assert!(con.allows(0x9f) && !con.allows(0x80) && !con.allows(b'"' as u32));
    for b in [0x9f, 0x98] {
        con.accept(b);
        assert!(!con.allows(b'"' as u32) && !con.complete());
    }
    con.accept(0x80);
    assert!(con.allows(b'"' as u32));
    // Surrogates are not characters.
    let mut con = follow(&c, r#"{"k":""#);
    con.accept(0xed);
    assert!(con.allows(0x9f) && !con.allows(0xa0));
}

#[test]
fn thinking_is_free_and_the_answer_follows() {
    let v = vocab();
    let c = compiled(&v, ResponseFormat::JsonObject);
    let (eos, open, close) = (special(&v, "<eos>"), special(&v, "<think>"), special(&v, "</think>"));
    // The reply starts inside a thinking block: anything but the end.
    let mut con = Constraint::new(c.clone(), Some((open, close)), true, &[eos]);
    for t in tokenize(&v, "Let me think: {not json] \"") {
        assert!(con.allows(t) && !con.allows(eos) && !con.allows(open) && con.allows(close) && !con.complete());
        con.accept(t);
    }
    con.accept(close);
    // After it: up to two newlines, then the document.
    assert!(con.allows(multi("\n\n")) && con.allows(b'{' as u32) && !con.allows(b'x' as u32) && !con.allows(close) && !con.allows(open));
    con.accept(multi("\n\n"));
    assert!(!con.allows(multi("\n")) && con.allows(multi("{}")));
    con.accept(multi("{}"));
    assert!(con.complete() && con.allows(eos) && !con.allows(multi("\n")));
    // The model opens the block itself, or answers at once.
    let mut con = Constraint::new(c.clone(), Some((open, close)), false, &[eos]);
    assert!(con.allows(open) && con.allows(b'{' as u32) && !con.allows(b'x' as u32) && !con.allows(multi("\n\n")));
    let mut direct = con.clone();
    direct.accept(b'{' as u32);
    assert!(!direct.allows(open) && direct.allows(b'}' as u32));
    con.accept(open);
    assert!(con.allows(b'x' as u32) && !con.allows(eos));
    // Without thinking, the delimiters never pass.
    let con = Constraint::new(c, None, false, &[eos]);
    assert!(!con.allows(open) && !con.allows(close) && con.allows(b'{' as u32));
}

/// Samples a reply from random logits under the constraint.
fn sample(c: &Arc<Compiled>, seed: u64, temperature: f32) -> (Vec<u8>, bool) {
    let v = c.vocab.clone();
    let eos = special(&v, "<eos>");
    let mut s = Sampler::new(temperature, 0, 1.0, seed + 1);
    s.set_guide(Some(Box::new(Constraint::new(c.clone(), None, false, &[eos]))), None);
    let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut text = Vec::new();
    for _ in 0..4000 {
        let mut logits: Vec<f32> = (0..v.len())
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 40) as f32 / (1u64 << 24) as f32 * 6.0 - 3.0
            })
            .collect();
        // Lean towards closing things, so that documents end.
        for t in *b"\"}]," {
            logits[t as usize] += 5.0;
        }
        logits[eos as usize] += 5.0;
        let t = s.sample(&logits);
        if t == eos {
            return (text, s.take_guide().unwrap().complete == Some(true));
        }
        text.extend_from_slice(v.bytes(t));
    }
    (text, false)
}

#[test]
fn sampled_replies_are_documents_of_the_schema() {
    let v = vocab();
    let schema = person();
    let mut g = Grammar::new();
    let rule = schema::compile(&mut g, &mut Prims::default(), &schema, &Dialect::Json).unwrap();
    let c = compiled(&v, ResponseFormat::JsonSchema(schema));
    for seed in 0..60 {
        for temperature in [0.0, 1.0, 2.0] {
            let (text, complete) = sample(&c, seed, temperature);
            let shown = String::from_utf8_lossy(&text).into_owned();
            assert!(complete, "seed {seed}: {shown}");
            let doc: Value = serde_json::from_slice(&text).unwrap_or_else(|e| panic!("seed {seed}: {e}: {shown}"));
            assert_eq!(Matcher::matches(&g, rule, &text), (true, true), "{shown}");
            let o = doc.as_object().unwrap();
            assert_eq!(o.keys().collect::<Vec<_>>(), ["name", "age", "tags", "home"], "{shown}");
            assert!(o["name"].is_string() && o["age"].is_i64() && o["tags"].as_array().unwrap().iter().all(Value::is_string));
            assert!(o["home"].is_null() || o["home"]["city"].is_string(), "{shown}");
        }
    }
    let c = compiled(&v, ResponseFormat::JsonObject);
    for seed in 0..40 {
        let (text, complete) = sample(&c, seed, 1.0);
        assert!(
            complete && serde_json::from_slice::<Value>(&text).unwrap().is_object(),
            "{}",
            String::from_utf8_lossy(&text)
        );
    }
}

fn weather() -> ToolSpec {
    ToolSpec {
        name: "get_weather".into(),
        description: "Weather of a city".into(),
        parameters: json!({
            "type": "object",
            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}, "unit": {"type": "string", "enum": ["c", "f"]},
                           "detail": {"type": ["boolean", "null"]}},
            "required": ["city", "days"],
        }),
    }
}

/// Feeds `text` to a new constraint of a forced call; None when a token is refused.
fn call_accepts(c: &Arc<Compiled>, text: &str) -> Option<bool> {
    tokens_accepted(c, &tokenize(&c.vocab, text))
}

fn tokens_accepted(c: &Arc<Compiled>, tokens: &[u32]) -> Option<bool> {
    let mut con = Constraint::new(c.clone(), None, false, &[special(&c.vocab, "<eos>")]);
    for &t in tokens {
        if !con.allows(t) {
            return None;
        }
        con.accept(t);
    }
    Some(con.complete())
}

#[test]
fn forced_calls_are_what_each_format_writes_and_parses() {
    let v = vocab();
    let tools = [
        weather(),
        ToolSpec {
            name: "now".into(),
            description: String::new(),
            parameters: json!({"type": "object", "properties": {}}),
        },
    ];
    let args = |v: Value| v.as_object().unwrap().clone();
    let full = ToolCall {
        name: "get_weather".into(),
        arguments: args(json!({"city": "São Paulo, \"SP\" <b>", "days": 3, "unit": "c", "detail": null})),
    };
    let short = ToolCall {
        name: "get_weather".into(),
        arguments: args(json!({"city": "Paris", "days": 10})),
    };
    let none = ToolCall {
        name: "now".into(),
        arguments: Map::new(),
    };
    let formats: [(&dyn ChatFormat, ToolSyntax); 6] = [
        (&chat::QWEN3, ToolSyntax::QwenJson),
        (&chat::QWEN, ToolSyntax::QwenXml),
        (&chat::GemmaChat, ToolSyntax::Gemma),
        (&chat::MistralChat, ToolSyntax::Mistral),
        (&chat::DeepSeekChat, ToolSyntax::Dsml),
        (&chat::DeepSeek41Chat, ToolSyntax::Dsml41),
    ];
    for (fmt, syntax) in formats {
        assert_eq!(fmt.tool_syntax(), Some(syntax));
        let any = Arc::new(
            compile(&v, &ResponseFormat::Text, &ToolChoice::Required, &tools, Some(syntax))
                .unwrap()
                .unwrap(),
        );
        let one = ToolChoice::Function("get_weather".into());
        let named = Arc::new(compile(&v, &ResponseFormat::Text, &one, &tools, Some(syntax)).unwrap().unwrap());
        for call in [&full, &short, &none] {
            // What the format renders for a past call is a complete forced call, and parses back.
            let text = fmt.assistant_text("", std::slice::from_ref(call));
            assert_eq!(call_accepts(&any, &text), Some(true), "{syntax:?}: {text}");
            assert_eq!(fmt.parse_tool_calls_for(&text, &tools), vec![call.clone()], "{syntax:?}: {text}");
            let expect = if call.name == "now" { None } else { Some(true) };
            assert_eq!(call_accepts(&named, &text), expect, "{syntax:?}: {text}");
            // Cut anywhere, it is a valid start that is not complete.
            let tokens = tokenize(&v, &text);
            for n in 0..tokens.len() {
                assert_eq!(tokens_accepted(&any, &tokens[..n]), Some(false), "{syntax:?}: {n} tokens of {text}");
            }
        }
        // Arguments outside the schema are refused: a missing required parameter, a wrong
        // type, an unknown name, a value outside the enum, text before the call.
        let bad = [
            ToolCall {
                name: "get_weather".into(),
                arguments: args(json!({"city": "Paris"})),
            },
            ToolCall {
                name: "get_weather".into(),
                arguments: args(json!({"city": "Paris", "days": 1.5})),
            },
            ToolCall {
                name: "get_weather".into(),
                arguments: args(json!({"city": "Paris", "days": 1, "unit": "k"})),
            },
            ToolCall {
                name: "get_weather".into(),
                arguments: args(json!({"city": "Paris", "days": 1, "zone": 1})),
            },
            ToolCall {
                name: "weather".into(),
                arguments: args(json!({"city": "Paris", "days": 1})),
            },
            ToolCall {
                name: "now".into(),
                arguments: args(json!({"city": "Paris"})),
            },
        ];
        for call in &bad {
            let text = fmt.assistant_text("", std::slice::from_ref(call));
            assert_ne!(call_accepts(&any, &text), Some(true), "{syntax:?}: {text}");
        }
        let text = format!("Sure.{}", fmt.assistant_text("", std::slice::from_ref(&short)));
        assert_eq!(call_accepts(&any, &text), None);
    }
}

#[test]
fn forced_calls_need_tools_a_syntax_and_a_supported_schema() {
    let v = vocab();
    let tools = [weather()];
    let err = |choice: ToolChoice, tools: &[ToolSpec], syntax| match compile(&v, &ResponseFormat::Text, &choice, tools, syntax) {
        Err(e) => format!("{e:#}"),
        Ok(_) => panic!("expected an error"),
    };
    assert!(err(ToolChoice::Required, &[], Some(ToolSyntax::QwenJson)).contains("no tools"));
    assert!(err(ToolChoice::Required, &tools, None).contains("no tool calling"));
    assert!(err(ToolChoice::Function("x".into()), &tools, Some(ToolSyntax::QwenJson)).contains("not among the tools"));
    let odd = [ToolSpec {
        name: "odd".into(),
        description: String::new(),
        parameters: json!({"type": "object", "properties": {"n": {"type": "integer", "minimum": 0}}}),
    }];
    for syntax in [
        ToolSyntax::QwenJson,
        ToolSyntax::QwenXml,
        ToolSyntax::Gemma,
        ToolSyntax::Mistral,
        ToolSyntax::Dsml,
        ToolSyntax::Dsml41,
    ] {
        let e = err(ToolChoice::Required, &odd, Some(syntax));
        assert!(e.contains("tool \"odd\"") && e.contains("\"minimum\" is not supported"), "{e}");
        // With a response format and no forced call, the model picks the document or a call; a
        // tool whose schema is not supported takes any arguments.
        let c = Arc::new(
            compile(&v, &ResponseFormat::JsonObject, &ToolChoice::Auto, &odd, Some(syntax))
                .unwrap()
                .unwrap(),
        );
        assert_eq!(call_accepts(&c, "{\"a\": 1}"), Some(true));
        let call = ToolCall {
            name: "odd".into(),
            arguments: json!({"n": -1, "other": "x"}).as_object().unwrap().clone(),
        };
        let fmt: &dyn ChatFormat = match syntax {
            ToolSyntax::QwenJson => &chat::QWEN3,
            ToolSyntax::QwenXml => &chat::QWEN,
            ToolSyntax::Gemma => &chat::GemmaChat,
            ToolSyntax::Mistral => &chat::MistralChat,
            ToolSyntax::Dsml => &chat::DeepSeekChat,
            ToolSyntax::Dsml41 => &chat::DeepSeek41Chat,
        };
        let text = fmt.assistant_text("", std::slice::from_ref(&call));
        assert_eq!(call_accepts(&c, &text), Some(true), "{syntax:?}: {text}");
    }
    // Nothing to constrain.
    assert!(
        compile(&v, &ResponseFormat::Text, &ToolChoice::Auto, &tools, Some(ToolSyntax::QwenJson))
            .unwrap()
            .is_none()
    );
    assert!(
        ResponseFormat::from_json(
            &json!({"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "string", "pattern": "a"}}})
        )
        .is_err()
    );
    assert_eq!(
        ResponseFormat::from_json(&json!({"type": "json_object"})).unwrap(),
        ResponseFormat::JsonObject
    );
    assert!(ResponseFormat::from_json(&json!({"type": "xml"})).is_err());
}

#[test]
fn byte_level_alphabet() {
    let a = byte_level();
    assert_eq!(a.len(), 256);
    assert_eq!((a[&'Ġ'], a[&'Ċ'], a[&'a'], a[&'é'], a[&'ĉ']), (b' ', b'\n', b'a', 0xe9, b'\t'));
}
