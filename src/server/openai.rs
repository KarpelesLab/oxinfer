//! The text endpoints: chat completions, (legacy) completions and embeddings.

use super::media::{TempFiles, random_seed, to_file};
use super::registry::{Kind, Loaded};
use super::{Api, ApiError, ApiResponse, ApiResult, Output, Sink, now};
use crate::engine::{Attachments, Engine, Flow, Message, Reply, Role, SamplingParams, StopReason, SystemPrompt};
use crate::tools::{ToolCall, ToolSpec};
use serde_json::{Map, Value, json};
use std::sync::Arc;

// ---- requests ---------------------------------------------------------------------------------

/// A media part of a user message, kept as sent until the request runs.
#[derive(Clone, Debug, PartialEq)]
enum MediaPart {
    Image(String),
    Video(String),
    /// Base64 audio and its format (`wav`, `mp3`, ...), or a URL.
    Audio(String, Option<String>),
}

#[derive(Clone, Debug, PartialEq)]
struct InMessage {
    role: Role,
    text: String,
    media: Vec<MediaPart>,
    tool_calls: Vec<(String, ToolCall)>,
    /// Tool messages: the name of the tool, from the call they answer.
    name: Option<String>,
}

/// Sampling and length settings a request may set (unset: the model's defaults).
#[derive(Clone, Debug, Default, PartialEq)]
struct Gen {
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    repetition_penalty: Option<f32>,
    seed: Option<u64>,
    max_tokens: Option<usize>,
    stop: Vec<String>,
    n: usize,
    stream: bool,
    include_usage: bool,
}

#[derive(Clone, Debug)]
struct ChatRequest {
    model: String,
    messages: Vec<InMessage>,
    tools: Vec<ToolSpec>,
    think: Option<bool>,
    params: Gen,
}

fn num(v: &Value, k: &'static str) -> ApiResult<Option<f64>> {
    match v.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(n)) => Ok(n.as_f64()),
        Some(_) => Err(ApiError::bad(format!("{k} must be a number")).param(k)),
    }
}

fn string<'a>(v: &'a Value, k: &'static str) -> ApiResult<Option<&'a str>> {
    match v.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s)),
        Some(_) => Err(ApiError::bad(format!("{k} must be a string")).param(k)),
    }
}

fn boolean(v: &Value, k: &'static str) -> ApiResult<Option<bool>> {
    match v.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(b)) => Ok(Some(*b)),
        Some(_) => Err(ApiError::bad(format!("{k} must be a boolean")).param(k)),
    }
}

impl Gen {
    fn parse(v: &Value, max_n: usize) -> ApiResult<Self> {
        let stop = match v.get("stop") {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::String(s)) => vec![s.clone()],
            Some(Value::Array(a)) => a
                .iter()
                .map(|s| s.as_str().map(str::to_string))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| ApiError::bad("stop must be a string or an array of strings").param("stop"))?,
            Some(_) => return Err(ApiError::bad("stop must be a string or an array of strings").param("stop")),
        };
        if stop.len() > 16 || stop.iter().any(String::is_empty) {
            return Err(ApiError::bad("stop: at most 16 non-empty sequences").param("stop"));
        }
        let n = num(v, "n")?.unwrap_or(1.0) as usize;
        if !(1..=max_n).contains(&n) {
            return Err(ApiError::bad(format!("n must be between 1 and {max_n}")).param("n"));
        }
        let stream = boolean(v, "stream")?.unwrap_or(false);
        if stream && n > 1 {
            return Err(ApiError::bad("n > 1 cannot be streamed").param("n"));
        }
        if boolean(v, "logprobs")?.unwrap_or(false) || v.get("top_logprobs").is_some_and(|t| !t.is_null()) {
            return Err(ApiError::bad("logprobs are not supported").param("logprobs"));
        }
        let max_tokens = num(v, "max_completion_tokens")?.or(num(v, "max_tokens")?);
        if max_tokens.is_some_and(|m| m < 1.0) {
            return Err(ApiError::bad("max_tokens must be at least 1").param("max_tokens"));
        }
        let temperature = num(v, "temperature")?;
        if temperature.is_some_and(|t| !(0.0..=2.0).contains(&t)) {
            return Err(ApiError::bad("temperature must be between 0 and 2").param("temperature"));
        }
        let top_p = num(v, "top_p")?;
        if top_p.is_some_and(|p| !(0.0..=1.0).contains(&p)) {
            return Err(ApiError::bad("top_p must be between 0 and 1").param("top_p"));
        }
        Ok(Self {
            temperature: temperature.map(|t| t as f32),
            top_p: top_p.map(|p| p as f32),
            top_k: num(v, "top_k")?.map(|k| k as usize),
            repetition_penalty: num(v, "repetition_penalty")?.map(|p| p as f32),
            seed: match v.get("seed") {
                Some(Value::Number(n)) => n.as_u64().or_else(|| n.as_i64().map(|i| i as u64)),
                None | Some(Value::Null) => None,
                Some(_) => return Err(ApiError::bad("seed must be an integer").param("seed")),
            },
            max_tokens: max_tokens.map(|m| m as usize),
            stop,
            n,
            stream,
            include_usage: v.pointer("/stream_options/include_usage").and_then(Value::as_bool).unwrap_or(false),
        })
    }

    /// The sampler for choice `i`, the request's values over the model's defaults (the `temperature`,
    /// `top_p`, `top_k` and `repetition_penalty` options) over OpenAI's (temperature 1, top-p 1).
    fn sampling(&self, spec: &super::ModelSpec, seed: u64) -> SamplingParams {
        let opt = |k: &str| spec.option(k).and_then(|v| v.parse::<f64>().ok());
        SamplingParams {
            temperature: self.temperature.or(opt("temperature").map(|v| v as f32)).unwrap_or(1.0),
            top_k: self.top_k.or(opt("top_k").map(|v| v as usize)).unwrap_or(0),
            top_p: self.top_p.or(opt("top_p").map(|v| v as f32)).unwrap_or(1.0),
            seed,
            repetition_penalty: self
                .repetition_penalty
                .or(opt("repetition_penalty").map(|v| v as f32))
                .unwrap_or(1.0),
        }
    }
}

/// The text of a message's `content` (a string or text parts) and its media parts.
fn content(v: &Value, role: Role) -> ApiResult<(String, Vec<MediaPart>)> {
    let bad = |m: &str| ApiError::bad(m.to_string()).param("messages");
    match v {
        Value::Null => Ok((String::new(), Vec::new())),
        Value::String(s) => Ok((s.clone(), Vec::new())),
        Value::Array(parts) => {
            let (mut text, mut media) = (String::new(), Vec::new());
            for p in parts {
                let ty = p.get("type").and_then(Value::as_str).unwrap_or("text");
                // A URL given as {"url": ...} or as the string itself.
                let url = |k: &str| -> ApiResult<String> {
                    let u = &p[k];
                    u.get("url")
                        .and_then(Value::as_str)
                        .or_else(|| u.as_str())
                        .map(str::to_string)
                        .ok_or_else(|| bad(&format!("{ty} part without a url")))
                };
                match ty {
                    "text" | "input_text" => {
                        if !text.is_empty() {
                            text.push('\n');
                        }
                        text.push_str(p.get("text").and_then(Value::as_str).ok_or_else(|| bad("text part without text"))?);
                    }
                    "image_url" | "input_image" => media.push(MediaPart::Image(url("image_url").or_else(|_| url("image"))?)),
                    "video_url" | "input_video" => media.push(MediaPart::Video(url("video_url").or_else(|_| url("video"))?)),
                    "input_audio" => {
                        let a = &p["input_audio"];
                        let data = a
                            .get("data")
                            .and_then(Value::as_str)
                            .ok_or_else(|| bad("input_audio part without data"))?;
                        let format = a.get("format").and_then(Value::as_str).map(str::to_string);
                        media.push(MediaPart::Audio(data.to_string(), format));
                    }
                    "audio_url" => media.push(MediaPart::Audio(url("audio_url")?, None)),
                    other => return Err(bad(&format!("unsupported content part type {other:?}"))),
                }
            }
            if !media.is_empty() && role != Role::User {
                return Err(bad("images, audio and video go in user messages"));
            }
            Ok((text, media))
        }
        _ => Err(bad("content must be a string, an array of parts or null")),
    }
}

/// Tool definitions (`[{"type": "function", "function": {...}}]`), as `tool_choice` allows.
fn tools(v: &Value) -> ApiResult<Vec<ToolSpec>> {
    let bad = |m: String| ApiError::bad(m).param("tools");
    let mut out = Vec::new();
    if let Some(list) = v.get("tools").filter(|t| !t.is_null()) {
        for t in list.as_array().ok_or_else(|| bad("tools must be an array".into()))? {
            if t.get("type").and_then(Value::as_str).unwrap_or("function") != "function" {
                return Err(bad("only function tools are supported".into()));
            }
            let f = t.get("function").unwrap_or(t);
            let name = f
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| bad("a tool has no name".into()))?;
            out.push(ToolSpec {
                name: name.to_string(),
                description: f.get("description").and_then(Value::as_str).unwrap_or_default().to_string(),
                parameters: f
                    .get("parameters")
                    .cloned()
                    .filter(|p| !p.is_null())
                    .unwrap_or_else(|| json!({ "type": "object", "properties": {} })),
            });
        }
    }
    match v.get("tool_choice") {
        None | Some(Value::Null) => {}
        Some(Value::String(s)) => match s.as_str() {
            "none" => out.clear(),
            "auto" | "required" => {}
            other => {
                return Err(
                    ApiError::bad(format!("tool_choice {other:?}: expected none, auto, required or a function")).param("tool_choice"),
                );
            }
        },
        Some(o) => {
            let name = o.pointer("/function/name").and_then(Value::as_str).ok_or_else(|| {
                ApiError::bad("tool_choice: expected {\"type\": \"function\", \"function\": {\"name\": ...}}").param("tool_choice")
            })?;
            out.retain(|t| t.name == name);
            if out.is_empty() {
                return Err(ApiError::bad(format!("tool_choice names {name:?}, which is not in tools")).param("tool_choice"));
            }
        }
    }
    Ok(out)
}

impl ChatRequest {
    fn parse(v: &Value) -> ApiResult<Self> {
        let model = string(v, "model")?.ok_or_else(|| ApiError::bad("model is required").param("model"))?;
        let msgs = v
            .get("messages")
            .and_then(Value::as_array)
            .filter(|m| !m.is_empty())
            .ok_or_else(|| ApiError::bad("messages must be a non-empty array").param("messages"))?;
        let mut messages: Vec<InMessage> = Vec::new();
        for m in msgs {
            let role = match m.get("role").and_then(Value::as_str) {
                Some("system" | "developer") => Role::System,
                Some("user") => Role::User,
                Some("assistant") => Role::Assistant,
                Some("tool" | "function") => Role::Tool,
                r => return Err(ApiError::bad(format!("unknown message role {r:?}")).param("messages")),
            };
            let (text, media) = content(m.get("content").unwrap_or(&Value::Null), role)?;
            let mut calls = Vec::new();
            if let Some(tc) = m.get("tool_calls").filter(|t| !t.is_null()) {
                for c in tc
                    .as_array()
                    .ok_or_else(|| ApiError::bad("tool_calls must be an array").param("messages"))?
                {
                    let f = c.get("function").unwrap_or(c);
                    let name = f.get("name").and_then(Value::as_str).unwrap_or_default().to_string();
                    let arguments = match f.get("arguments") {
                        Some(Value::String(s)) if s.trim().is_empty() => Map::new(),
                        Some(Value::String(s)) => match serde_json::from_str(s) {
                            Ok(Value::Object(o)) => o,
                            _ => return Err(ApiError::bad(format!("tool call {name}: arguments are not a JSON object")).param("messages")),
                        },
                        Some(Value::Object(o)) => o.clone(),
                        _ => Map::new(),
                    };
                    let id = c.get("id").and_then(Value::as_str).unwrap_or_default().to_string();
                    calls.push((id, ToolCall { name, arguments }));
                }
            }
            let name = if role == Role::Tool {
                let id = m.get("tool_call_id").and_then(Value::as_str);
                // The call this answers, by id (else by order in the message's handler).
                let by_id = id.and_then(|id| {
                    messages
                        .iter()
                        .rev()
                        .flat_map(|p| p.tool_calls.iter())
                        .find(|(cid, _)| cid == id)
                        .map(|(_, c)| c.name.clone())
                });
                by_id.or_else(|| m.get("name").and_then(Value::as_str).map(str::to_string))
            } else {
                None
            };
            messages.push(InMessage {
                role,
                text,
                media,
                tool_calls: calls,
                name,
            });
        }
        let effort = string(v, "reasoning_effort")?;
        let think = boolean(v, "think")?
            .or(boolean(v, "enable_thinking")?)
            .or(v.pointer("/chat_template_kwargs/enable_thinking").and_then(Value::as_bool))
            .or(effort.map(|e| !matches!(e, "none" | "minimal")));
        Ok(Self {
            model: model.to_string(),
            messages,
            tools: tools(v)?,
            think,
            params: Gen::parse(v, 8)?,
        })
    }

    /// The engine's messages, with the media written to temporary files.
    fn messages(&self, api: &Api, temp: &mut TempFiles) -> ApiResult<Vec<Message>> {
        self.messages
            .iter()
            .map(|m| {
                let mut a = Attachments::default();
                for p in &m.media {
                    match p {
                        MediaPart::Image(u) => a.images.push(to_file(api, u, temp)?),
                        MediaPart::Video(u) => a.videos.push(to_file(api, u, temp)?),
                        MediaPart::Audio(d, _) if d.contains("://") || d.starts_with("data:") => a.audios.push(to_file(api, d, temp)?),
                        MediaPart::Audio(d, fmt) => {
                            let bytes = crate::b64::decode(d).ok_or_else(|| ApiError::bad("input_audio: bad base64").param("messages"))?;
                            let ext = super::media::extension(&bytes, None, fmt.as_deref().map(|f| format!("x.{f}")).as_deref());
                            a.audios.push(temp.write(&bytes, &ext)?);
                        }
                    }
                }
                let mut msg = Message::new(m.role, m.text.clone()).with_media(a);
                msg.tool_calls = m.tool_calls.iter().map(|(_, c)| c.clone()).collect();
                msg.name = m.name.clone();
                Ok(msg)
            })
            .collect()
    }
}

// ---- stop sequences ---------------------------------------------------------------------------

/// Cuts generated text at the first stop sequence, holding back what could be the start of one.
struct StopFilter<'a> {
    stops: &'a [String],
    held: String,
    hit: bool,
}

impl<'a> StopFilter<'a> {
    fn new(stops: &'a [String]) -> Self {
        Self {
            stops,
            held: String::new(),
            hit: false,
        }
    }

    /// The text that can be released after `s`; sets `hit` at a stop sequence.
    fn push(&mut self, s: &str) -> String {
        if self.hit {
            return String::new();
        }
        if self.stops.is_empty() {
            return s.to_string();
        }
        self.held.push_str(s);
        if let Some(i) = self.stops.iter().filter_map(|st| self.held.find(st.as_str())).min() {
            self.hit = true;
            let out = self.held[..i].to_string();
            self.held.clear();
            return out;
        }
        // Keep the longest tail that is the start of a stop sequence.
        let keep = self
            .stops
            .iter()
            .map(|st| {
                (1..st.len().min(self.held.len() + 1))
                    .rev()
                    .find(|&k| st.is_char_boundary(k) && self.held.ends_with(&st[..k]))
                    .unwrap_or(0)
            })
            .max()
            .unwrap_or(0);
        let cut = self.held.len() - keep;
        let out = self.held[..cut].to_string();
        self.held.drain(..cut);
        out
    }

    fn finish(&mut self) -> String {
        std::mem::take(&mut self.held)
    }
}

// ---- chat completions -------------------------------------------------------------------------

/// One generated choice.
struct Choice {
    text: String,
    thinking: String,
    tool_calls: Vec<ToolCall>,
    finish: &'static str,
    prompt_tokens: usize,
    cached_tokens: usize,
    completion_tokens: usize,
}

fn finish_reason(r: &Reply, stopped: bool) -> &'static str {
    match r.finish {
        _ if stopped => "stop",
        StopReason::Stop if !r.tool_calls.is_empty() => "tool_calls",
        StopReason::Length => "length",
        _ => "stop",
    }
}

/// A streamed piece: (choice, text, thinking).
type Emit<'a> = &'a mut dyn FnMut(usize, &str, bool) -> bool;

/// Runs the request's `n` choices on its model, streaming the text to `emit`.
fn run_chat(api: &Api, req: &ChatRequest, emit: Emit) -> ApiResult<Vec<Choice>> {
    let mut temp = TempFiles::default();
    let messages = req.messages(api, &mut temp)?;
    let mut lease = api.models.acquire(&req.model, Kind::Chat)?;
    let spec = lease.spec().clone();
    let Loaded::Chat(engine) = lease.model() else { unreachable!() };
    configure(engine, &spec, req.think, &req.tools, req.params.max_tokens);
    let seed = req.params.seed.unwrap_or_else(random_seed);
    let mut out = Vec::new();
    for i in 0..req.params.n {
        engine.set_sampling(req.params.sampling(&spec, seed.wrapping_add(i as u64)));
        let mut filter = StopFilter::new(&req.params.stop);
        let (mut text, mut thinking) = (String::new(), String::new());
        let mut gone = false;
        let reply = engine
            .chat(&messages, &mut |d| {
                let ok = if d.thinking {
                    thinking.push_str(d.text);
                    d.text.is_empty() || emit(i, d.text, true)
                } else {
                    let t = filter.push(d.text);
                    text.push_str(&t);
                    t.is_empty() || emit(i, &t, false)
                };
                gone |= !ok;
                if gone || filter.hit { Flow::Stop } else { Flow::Continue }
            })
            .map_err(|e| ApiError::bad(format!("{e:#}")))?;
        let rest = filter.finish();
        if !rest.is_empty() {
            text.push_str(&rest);
            emit(i, &rest, false);
        }
        if gone {
            break;
        }
        out.push(Choice {
            finish: finish_reason(&reply, filter.hit),
            text,
            thinking,
            tool_calls: if filter.hit { Vec::new() } else { reply.tool_calls.clone() },
            prompt_tokens: reply.stats.prompt_tokens + reply.stats.cached_tokens,
            cached_tokens: reply.stats.cached_tokens,
            completion_tokens: reply.stats.generated_tokens,
        });
    }
    Ok(out)
}

/// The request's conversation settings over the model's (`system`, `think`, `max_tokens`).
fn configure(engine: &mut Engine, spec: &super::ModelSpec, think: Option<bool>, tools: &[ToolSpec], max_tokens: Option<usize>) {
    let c = engine.config_mut();
    c.system = match spec.option("system") {
        None | Some("") => SystemPrompt::None,
        Some("default") => SystemPrompt::Default,
        Some(t) => SystemPrompt::Text(t.to_string()),
    };
    c.think = think.unwrap_or_else(|| spec.option("think").is_some_and(|v| matches!(v, "1" | "true" | "yes" | "on")));
    c.tools = tools.to_vec();
    c.max_tokens = max_tokens
        .or_else(|| spec.option("max_tokens").and_then(|v| v.parse().ok()))
        .unwrap_or(4096);
}

fn tool_calls_json(api: &Api, calls: &[ToolCall], indexed: bool) -> Value {
    Value::Array(
        calls
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let mut v = json!({
                    "id": api.new_id("call_"),
                    "type": "function",
                    "function": { "name": c.name, "arguments": Value::Object(c.arguments.clone()).to_string() },
                });
                if indexed {
                    v["index"] = i.into();
                }
                v
            })
            .collect(),
    )
}

fn usage(choices: &[Choice]) -> Value {
    let prompt = choices.first().map_or(0, |c| c.prompt_tokens);
    let cached = choices.first().map_or(0, |c| c.cached_tokens);
    let completion: usize = choices.iter().map(|c| c.completion_tokens).sum();
    json!({
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": { "cached_tokens": cached },
    })
}

/// `POST /v1/chat/completions`.
pub(super) fn chat(api: &Arc<Api>, body: &Value) -> ApiResult<Output> {
    let req = ChatRequest::parse(body)?;
    api.models.check(&req.model, Kind::Chat)?;
    let id = api.new_id("chatcmpl-");
    let created = now();
    if req.params.stream {
        let api = api.clone();
        return Ok(Output::Stream {
            content_type: "text/event-stream".into(),
            producer: Box::new(move |sink| {
                stream_chat(&api, &req, &id, created, sink);
                api.flush_prompt_cache_later(&req.model);
            }),
        });
    }
    let choices = run_chat(api, &req, &mut |_, _, _| true)?;
    api.flush_prompt_cache_later(&req.model);
    let list: Vec<Value> = choices
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let mut msg = json!({
                "role": "assistant",
                "content": if c.text.is_empty() && !c.tool_calls.is_empty() { Value::Null } else { c.text.clone().into() },
            });
            if !c.thinking.is_empty() {
                msg["reasoning_content"] = c.thinking.clone().into();
            }
            if !c.tool_calls.is_empty() {
                msg["tool_calls"] = tool_calls_json(api, &c.tool_calls, false);
            }
            json!({ "index": i, "message": msg, "logprobs": null, "finish_reason": c.finish })
        })
        .collect();
    Ok(Output::Full(ApiResponse::json(
        200,
        &json!({
            "id": id,
            "object": "chat.completion",
            "created": created,
            "model": req.model,
            "choices": list,
            "usage": usage(&choices),
            "system_fingerprint": null,
        }),
    )))
}

fn sse(sink: Sink, v: &Value) -> bool {
    sink(format!("data: {v}\n\n").as_bytes())
}

fn stream_chat(api: &Api, req: &ChatRequest, id: &str, created: u64, sink: Sink) {
    let chunk = |delta: Value, finish: Value| {
        json!({
            "id": id, "object": "chat.completion.chunk", "created": created, "model": req.model,
            "choices": [{ "index": 0, "delta": delta, "logprobs": null, "finish_reason": finish }],
        })
    };
    if !sse(sink, &chunk(json!({ "role": "assistant", "content": "" }), Value::Null)) {
        return;
    }
    let r = run_chat(api, req, &mut |_, text, thinking| {
        let delta = if thinking {
            json!({ "reasoning_content": text })
        } else {
            json!({ "content": text })
        };
        sse(sink, &chunk(delta, Value::Null))
    });
    match r {
        Ok(choices) => {
            let Some(c) = choices.first() else { return };
            if !c.tool_calls.is_empty() {
                sse(
                    sink,
                    &chunk(json!({ "tool_calls": tool_calls_json(api, &c.tool_calls, true) }), Value::Null),
                );
            }
            sse(sink, &chunk(json!({}), c.finish.into()));
            if req.params.include_usage {
                let u = json!({
                    "id": id, "object": "chat.completion.chunk", "created": created, "model": req.model,
                    "choices": [], "usage": usage(&choices),
                });
                sse(sink, &u);
            }
        }
        Err(e) => {
            sse(sink, &e.to_json());
        }
    }
    sink(b"data: [DONE]\n\n");
}

// ---- completions ------------------------------------------------------------------------------

/// `POST /v1/completions`: raw continuations, no chat template.
pub(super) fn completions(api: &Arc<Api>, body: &Value) -> ApiResult<Output> {
    let model = string(body, "model")?
        .ok_or_else(|| ApiError::bad("model is required").param("model"))?
        .to_string();
    let prompts: Vec<Value> = match body.get("prompt") {
        Some(Value::String(s)) => vec![Value::String(s.clone())],
        Some(Value::Array(a)) if a.iter().all(Value::is_number) && !a.is_empty() => vec![Value::Array(a.clone())],
        Some(Value::Array(a)) if !a.is_empty() => a.clone(),
        _ => return Err(ApiError::bad("prompt must be a string, an array of strings or of token ids").param("prompt")),
    };
    if prompts.len() > 32 {
        return Err(ApiError::bad("at most 32 prompts").param("prompt"));
    }
    if string(body, "suffix")?.is_some_and(|s| !s.is_empty()) {
        return Err(ApiError::bad("suffix is not supported").param("suffix"));
    }
    if num(body, "best_of")?.is_some_and(|b| b > 1.0) {
        return Err(ApiError::bad("best_of is not supported").param("best_of"));
    }
    let mut params = Gen::parse(body, 8)?;
    // OpenAI's default for this endpoint.
    params.max_tokens.get_or_insert(16);
    let echo = boolean(body, "echo")?.unwrap_or(false);
    if params.stream && prompts.len() > 1 {
        return Err(ApiError::bad("several prompts cannot be streamed").param("prompt"));
    }
    api.models.check(&model, Kind::Chat)?;
    let id = api.new_id("cmpl-");
    let created = now();
    let job = Completion {
        model,
        prompts,
        params,
        echo,
    };
    if job.params.stream {
        let api = api.clone();
        return Ok(Output::Stream {
            content_type: "text/event-stream".into(),
            producer: Box::new(move |sink| {
                let chunk = |text: &str, finish: Value| {
                    json!({
                        "id": id, "object": "text_completion", "created": created, "model": job.model,
                        "choices": [{ "index": 0, "text": text, "logprobs": null, "finish_reason": finish }],
                    })
                };
                match job.run(&api, &mut |_, t| sse(sink, &chunk(t, Value::Null))) {
                    Ok(choices) => {
                        if let Some(c) = choices.first() {
                            sse(sink, &chunk("", c.finish.into()));
                        }
                        if job.params.include_usage {
                            let u = json!({
                                "id": id, "object": "text_completion", "created": created, "model": job.model,
                                "choices": [], "usage": usage(&choices),
                            });
                            sse(sink, &u);
                        }
                    }
                    Err(e) => {
                        sse(sink, &e.to_json());
                    }
                }
                sink(b"data: [DONE]\n\n");
            }),
        });
    }
    let choices = job.run(api, &mut |_, _| true)?;
    let list: Vec<Value> = choices
        .iter()
        .enumerate()
        .map(|(i, c)| json!({ "index": i, "text": c.text, "logprobs": null, "finish_reason": c.finish }))
        .collect();
    let completion: usize = choices.iter().map(|c| c.completion_tokens).sum();
    let prompt: usize = choices.iter().step_by(job.params.n).map(|c| c.prompt_tokens).sum();
    Ok(Output::Full(ApiResponse::json(
        200,
        &json!({
            "id": id, "object": "text_completion", "created": created, "model": job.model, "choices": list,
            "usage": { "prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion },
        }),
    )))
}

struct Completion {
    model: String,
    prompts: Vec<Value>,
    params: Gen,
    echo: bool,
}

impl Completion {
    /// Every prompt `n` times; `emit(choice, text)` streams.
    fn run(&self, api: &Api, emit: &mut dyn FnMut(usize, &str) -> bool) -> ApiResult<Vec<Choice>> {
        let mut lease = api.models.acquire(&self.model, Kind::Chat)?;
        let spec = lease.spec().clone();
        let Loaded::Chat(engine) = lease.model() else { unreachable!() };
        engine.config_mut().max_tokens = self.params.max_tokens.unwrap_or(16);
        let seed = self.params.seed.unwrap_or_else(random_seed);
        let mut out = Vec::new();
        for p in &self.prompts {
            let prompt = match p {
                Value::String(s) => s.clone(),
                Value::Array(ids) => {
                    let ids: Option<Vec<u32>> = ids.iter().map(|t| t.as_u64().map(|t| t as u32)).collect();
                    let ids = ids.ok_or_else(|| ApiError::bad("token prompts must be arrays of integers").param("prompt"))?;
                    engine
                        .tokenizer()
                        .decode(&ids, false)
                        .map_err(|e| ApiError::bad(format!("decode the prompt tokens: {e}")).param("prompt"))?
                }
                _ => return Err(ApiError::bad("prompt must be a string, an array of strings or of token ids").param("prompt")),
            };
            for _ in 0..self.params.n {
                let k = out.len();
                engine.reset();
                engine.set_sampling(self.params.sampling(&spec, seed.wrapping_add(k as u64)));
                let mut filter = StopFilter::new(&self.params.stop);
                let mut text = if self.echo { prompt.clone() } else { String::new() };
                if self.echo && !emit(k, &prompt) {
                    break;
                }
                let mut gone = false;
                let reply = engine
                    .complete(&prompt, &mut |d| {
                        let t = filter.push(d.text);
                        text.push_str(&t);
                        gone |= !t.is_empty() && !emit(k, &t);
                        if gone || filter.hit { Flow::Stop } else { Flow::Continue }
                    })
                    .map_err(|e| ApiError::bad(format!("{e:#}")))?;
                let rest = filter.finish();
                if !rest.is_empty() {
                    text.push_str(&rest);
                    emit(k, &rest);
                }
                if gone {
                    return Ok(out);
                }
                out.push(Choice {
                    finish: finish_reason(&reply, filter.hit),
                    text,
                    thinking: String::new(),
                    tool_calls: Vec::new(),
                    prompt_tokens: reply.stats.prompt_tokens,
                    cached_tokens: 0,
                    completion_tokens: reply.stats.generated_tokens,
                });
            }
        }
        Ok(out)
    }
}

// ---- embeddings -------------------------------------------------------------------------------

/// `POST /v1/embeddings` (`encoding_format` float or base64, `dimensions`; extension:
/// `prompt_name` for models with named prompts).
pub(super) fn embeddings(api: &Arc<Api>, body: &Value) -> ApiResult<Output> {
    let model = string(body, "model")?.ok_or_else(|| ApiError::bad("model is required").param("model"))?;
    let bad_input = || ApiError::bad("input must be a string, an array of strings, or token ids").param("input");
    let inputs: Vec<Value> = match body.get("input") {
        Some(Value::String(s)) => vec![Value::String(s.clone())],
        Some(Value::Array(a)) if !a.is_empty() && a.iter().all(Value::is_number) => vec![Value::Array(a.clone())],
        Some(Value::Array(a)) if !a.is_empty() => a.clone(),
        _ => return Err(bad_input()),
    };
    if inputs.len() > 2048 {
        return Err(ApiError::bad("at most 2048 inputs").param("input"));
    }
    let base64 = match string(body, "encoding_format")? {
        None | Some("float") => false,
        Some("base64") => true,
        Some(f) => return Err(ApiError::bad(format!("encoding_format {f:?}: expected float or base64")).param("encoding_format")),
    };
    let dims = num(body, "dimensions")?.map(|d| d as usize);
    api.models.check(model, Kind::Embedding)?;
    let mut lease = api.models.acquire(model, Kind::Embedding)?;
    let prompt_name = string(body, "prompt_name")?
        .map(str::to_string)
        .or_else(|| lease.spec().option("prompt_name").map(str::to_string));
    let Loaded::Embedding(m) = lease.model() else { unreachable!() };
    let (mut data, mut tokens) = (Vec::new(), 0);
    for (i, inp) in inputs.iter().enumerate() {
        let text = match inp {
            Value::String(s) => s.clone(),
            Value::Array(ids) => {
                let ids: Option<Vec<u32>> = ids.iter().map(|t| t.as_u64().map(|t| t as u32)).collect();
                m.emb
                    .tok
                    .decode(&ids.ok_or_else(bad_input)?, true)
                    .map_err(|e| ApiError::bad(format!("decode tokens: {e}")))?
            }
            _ => return Err(bad_input()),
        };
        tokens += m.emb.tok.encode(text.as_str(), true).map(|e| e.len()).unwrap_or(0);
        let mut e = m
            .emb
            .embed(&text, prompt_name.as_deref())
            .map_err(|e| ApiError::bad(format!("{e:#}")))?;
        if let Some(d) = dims {
            e.truncate(d).map_err(|e| ApiError::bad(format!("{e:#}")).param("dimensions"))?;
        }
        let embedding = if base64 {
            let bytes: Vec<u8> = e.dense.iter().flat_map(|x| x.to_le_bytes()).collect();
            Value::String(crate::b64::encode(&bytes))
        } else {
            json!(e.dense)
        };
        data.push(json!({ "object": "embedding", "index": i, "embedding": embedding }));
    }
    Ok(Output::Full(ApiResponse::json(
        200,
        &json!({ "object": "list", "data": data, "model": model, "usage": { "prompt_tokens": tokens, "total_tokens": tokens } }),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_filter_holds_partial_matches() {
        let stops = vec!["END".to_string(), "\n\n".to_string()];
        let mut f = StopFilter::new(&stops);
        assert_eq!(f.push("Hello E"), "Hello ");
        assert_eq!(f.push("N"), "");
        assert_eq!(f.push("X"), "ENX");
        assert_eq!(f.push(" and\nmore"), " and\nmore");
        assert_eq!(f.push("\n\nlater"), "");
        assert!(f.hit);
        assert_eq!(f.push("ignored"), "");
        let mut f = StopFilter::new(&stops);
        assert_eq!(f.push("abcE"), "abc");
        assert_eq!(f.finish(), "E");
        let none: Vec<String> = Vec::new();
        assert_eq!(StopFilter::new(&none).push("x"), "x");
    }

    #[test]
    fn chat_request_parsing() {
        let v = json!({
            "model": "m",
            "messages": [
                {"role": "developer", "content": "Be brief."},
                {"role": "user", "content": [{"type": "text", "text": "What is this?"},
                                              {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "look", "arguments": "{\"x\": 1}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "a cat"},
            ],
            "tools": [{"type": "function", "function": {"name": "look", "parameters": {"type": "object"}}},
                      {"type": "function", "function": {"name": "other"}}],
            "tool_choice": {"type": "function", "function": {"name": "look"}},
            "max_completion_tokens": 20, "stop": "\n", "seed": 3, "temperature": 0,
            "stream": true, "stream_options": {"include_usage": true},
            "reasoning_effort": "high",
        });
        let r = ChatRequest::parse(&v).unwrap();
        assert_eq!(r.messages.len(), 4);
        assert_eq!(r.messages[0].role, Role::System);
        assert_eq!(r.messages[1].media, vec![MediaPart::Image("data:image/png;base64,AAAA".into())]);
        assert_eq!(r.messages[2].tool_calls[0].1.arguments["x"], 1);
        assert_eq!(r.messages[3].name.as_deref(), Some("look"));
        assert_eq!(r.tools.len(), 1);
        assert_eq!(r.think, Some(true));
        let g = &r.params;
        assert_eq!(
            (g.max_tokens, g.seed, g.temperature, g.stream, g.include_usage),
            (Some(20), Some(3), Some(0.0), true, true)
        );
        assert_eq!(g.stop, vec!["\n".to_string()]);

        let bad = |v: Value| ChatRequest::parse(&v).err().unwrap();
        assert_eq!(bad(json!({"model": "m", "messages": []})).param, Some("messages"));
        assert_eq!(
            bad(json!({"model": "m", "messages": [{"role": "robot", "content": "x"}]})).status,
            400
        );
        assert_eq!(
            bad(json!({"model": "m", "messages": [{"role": "user", "content": "x"}], "n": 2, "stream": true})).param,
            Some("n")
        );
        assert_eq!(
            bad(json!({"model": "m", "messages": [{"role": "user", "content": "x"}], "temperature": 3})).param,
            Some("temperature")
        );
        assert_eq!(
            bad(json!({"model": "m", "messages": [{"role": "assistant", "content": [{"type": "image_url", "image_url": "x"}]}]})).param,
            Some("messages")
        );
        let none = json!({"model": "m", "messages": [{"role": "user", "content": "x"}], "tools": v["tools"], "tool_choice": "none"});
        assert!(ChatRequest::parse(&none).unwrap().tools.is_empty());
    }

    #[test]
    fn sampling_defaults() {
        let spec = super::super::ModelSpec::parse("m=x/y,temperature=0.6,top_k=20").unwrap();
        let g = Gen::parse(&json!({"top_p": 0.9}), 1).unwrap();
        let s = g.sampling(&spec, 5);
        assert_eq!((s.temperature, s.top_k, s.top_p, s.seed), (0.6, 20, 0.9, 5));
        let s = Gen::parse(&json!({}), 1)
            .unwrap()
            .sampling(&super::super::ModelSpec::new("x").unwrap(), 1);
        assert_eq!((s.temperature, s.top_k, s.top_p), (1.0, 0, 1.0));
    }
}
