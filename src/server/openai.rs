//! The text endpoints: chat completions, (legacy) completions and embeddings.

use super::media::{TempFiles, random_seed, to_file};
use super::registry::{Kind, Loaded, OnStage, Stage};
use super::{Api, ApiError, ApiResponse, ApiResult, Output, Sink, now};
use crate::constrain::Vocab;
use crate::engine::{
    Attachments, Engine, Flow, Message, Reply, ResponseFormat, Role, SamplingParams, StopReason, SystemPrompt, TokenLogprob, ToolChoice,
};
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
    /// `stream_options.progress`: what the request waits for as `progress` events (an
    /// extension) instead of comment lines.
    progress: bool,
    /// Log probabilities with this many alternatives per token.
    logprobs: Option<usize>,
}

#[derive(Clone, Debug)]
struct ChatRequest {
    model: String,
    messages: Vec<InMessage>,
    tools: Vec<ToolSpec>,
    tool_choice: ToolChoice,
    response_format: ResponseFormat,
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
        // Chat: `logprobs: true` and `top_logprobs: N`; completions: `logprobs: N`.
        let logprobs = match v.get("logprobs") {
            None | Some(Value::Null | Value::Bool(false)) => None,
            Some(Value::Bool(true)) => Some(0),
            Some(Value::Number(n)) => match n.as_u64() {
                Some(n) if n <= 20 => Some(n as usize),
                _ => return Err(ApiError::bad("logprobs must be between 0 and 20").param("logprobs")),
            },
            Some(_) => return Err(ApiError::bad("logprobs must be a boolean (chat) or a count (completions)").param("logprobs")),
        };
        let logprobs = match num(v, "top_logprobs")? {
            None => logprobs,
            Some(t) if !(0.0..=20.0).contains(&t) => {
                return Err(ApiError::bad("top_logprobs must be between 0 and 20").param("top_logprobs"));
            }
            Some(_) if logprobs.is_none() => return Err(ApiError::bad("top_logprobs requires logprobs to be true").param("top_logprobs")),
            Some(t) => Some(t as usize),
        };
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
            progress: v.pointer("/stream_options/progress").and_then(Value::as_bool).unwrap_or(false),
            logprobs,
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

/// Tool definitions (`[{"type": "function", "function": {...}}]`), as `tool_choice` allows, and
/// whether a call is forced (the parameter schemas of the tools it may be are then checked).
fn tools(v: &Value) -> ApiResult<(Vec<ToolSpec>, ToolChoice)> {
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
    let mut choice = ToolChoice::Auto;
    match v.get("tool_choice") {
        None | Some(Value::Null) => {}
        Some(Value::String(s)) => match s.as_str() {
            "none" => out.clear(),
            "auto" => {}
            "required" if out.is_empty() => {
                return Err(ApiError::bad("tool_choice is required, but no tools are given").param("tool_choice"));
            }
            "required" => choice = ToolChoice::Required,
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
            choice = ToolChoice::Function(name.to_string());
        }
    }
    if choice != ToolChoice::Auto {
        for t in &out {
            crate::constrain::check_tool(t).map_err(|e| ApiError::bad(format!("{e:#}")).param("tools"))?;
        }
    }
    Ok((out, choice))
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
        let (tools, tool_choice) = tools(v)?;
        let response_format = match v.get("response_format") {
            None | Some(Value::Null) => ResponseFormat::Text,
            Some(f) => ResponseFormat::from_json(f).map_err(|e| ApiError::bad(format!("{e:#}")).param("response_format"))?,
        };
        Ok(Self {
            model: model.to_string(),
            messages,
            tools,
            tool_choice,
            response_format,
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
    /// The stop sequence that was hit.
    which: Option<String>,
}

impl<'a> StopFilter<'a> {
    fn new(stops: &'a [String]) -> Self {
        Self {
            stops,
            held: String::new(),
            hit: false,
            which: None,
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
        if let Some((i, st)) = self
            .stops
            .iter()
            .filter_map(|st| self.held.find(st.as_str()).map(|i| (i, st)))
            .min_by_key(|&(i, _)| i)
        {
            self.hit = true;
            self.which = Some(st.clone());
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

/// A token's log probability as the API shows it: the token's text and bytes, and the most
/// likely tokens at its position.
#[derive(Clone, Debug)]
struct LogEntry {
    token: String,
    bytes: Vec<u8>,
    logprob: f32,
    top: Vec<(String, Vec<u8>, f32)>,
}

impl LogEntry {
    fn new(vocab: &Vocab, l: &TokenLogprob) -> Self {
        let text = |id: u32| (String::from_utf8_lossy(vocab.bytes(id)).into_owned(), vocab.bytes(id).to_vec());
        // JSON has no infinity: a token the model gave no probability (one a constraint
        // forced, at most) shows OpenAI's floor.
        let finite = |p: f32| if p.is_finite() { p.max(-9999.0) } else { -9999.0 };
        let (token, bytes) = text(l.token);
        Self {
            token,
            bytes,
            logprob: finite(l.logprob),
            top: l
                .top
                .iter()
                .map(|&(id, p)| {
                    let (t, b) = text(id);
                    (t, b, finite(p))
                })
                .collect(),
        }
    }

    /// Chat completions: `{"content": [{token, logprob, bytes, top_logprobs}]}`.
    fn chat(entries: &[LogEntry]) -> Value {
        let list: Vec<Value> = entries
            .iter()
            .map(|e| {
                let top: Vec<Value> = e
                    .top
                    .iter()
                    .map(|(t, b, p)| json!({ "token": t, "logprob": p, "bytes": b }))
                    .collect();
                json!({ "token": e.token, "logprob": e.logprob, "bytes": e.bytes, "top_logprobs": top })
            })
            .collect();
        json!({ "content": list, "refusal": null })
    }

    /// Completions: parallel `tokens`, `token_logprobs`, `top_logprobs` and `text_offset`
    /// (characters, from `offset`).
    fn legacy(entries: &[LogEntry], mut offset: usize) -> Value {
        let mut offsets = Vec::new();
        for e in entries {
            offsets.push(offset);
            offset += e.token.chars().count();
        }
        json!({
            "tokens": entries.iter().map(|e| e.token.as_str()).collect::<Vec<_>>(),
            "token_logprobs": entries.iter().map(|e| e.logprob).collect::<Vec<_>>(),
            "top_logprobs": entries.iter().map(|e| e.top.iter().map(|(t, _, p)| (t.clone(), json!(p))).collect::<Map<String, Value>>()).collect::<Vec<_>>(),
            "text_offset": offsets,
        })
    }
}

/// One generated choice.
struct Choice {
    /// With `logprobs`: the entries of the answer's tokens.
    logprobs: Option<Vec<LogEntry>>,
    text: String,
    thinking: String,
    tool_calls: Vec<ToolCall>,
    finish: &'static str,
    /// The stop sequence that ended the reply.
    stop_sequence: Option<String>,
    prompt_tokens: usize,
    cached_tokens: usize,
    completion_tokens: usize,
    /// Characters of `text` that were generated (completions: the rest is the echoed prompt).
    generated_chars: usize,
}

fn finish_reason(r: &Reply, stopped: bool) -> &'static str {
    match r.finish {
        _ if stopped => "stop",
        StopReason::Stop if !r.tool_calls.is_empty() => "tool_calls",
        StopReason::Length => "length",
        _ => "stop",
    }
}

/// A streamed piece: (choice, text, thinking, the log probabilities of the tokens it adds).
type Emit<'a> = &'a mut dyn FnMut(usize, &str, bool, &[LogEntry]) -> bool;

/// Collects the log probabilities of a reply's answer tokens: all of them, and those not yet
/// sent with a streamed piece.
struct LogCollector {
    vocab: Option<Arc<Vocab>>,
    all: Vec<LogEntry>,
    pending: Vec<LogEntry>,
}

impl LogCollector {
    fn new(engine: &mut Engine, on: bool) -> Self {
        Self {
            vocab: on.then(|| engine.token_vocab()),
            all: Vec::new(),
            pending: Vec::new(),
        }
    }

    fn push(&mut self, l: Option<&TokenLogprob>) {
        if let (Some(v), Some(l)) = (&self.vocab, l.filter(|l| l.content)) {
            self.pending.push(LogEntry::new(v, l));
        }
    }

    /// The entries to send with the next piece.
    fn take(&mut self) -> Vec<LogEntry> {
        self.all.extend(self.pending.iter().cloned());
        std::mem::take(&mut self.pending)
    }

    /// The entries of the reply; with `cut` (a stop sequence), without the tokens whose text
    /// was never released.
    fn finish(mut self, cut: bool) -> (Option<Vec<LogEntry>>, Vec<LogEntry>) {
        let rest = if cut { Vec::new() } else { self.take() };
        (self.vocab.is_some().then_some(self.all), rest)
    }
}

/// A byte count for a progress line: `412 kB`, `35.2 MB`, `1.74 GB`.
fn size(n: u64) -> String {
    match n {
        0..1_000_000 => format!("{} kB", n.div_ceil(1000)),
        1_000_000..1_000_000_000 => format!("{:.1} MB", n as f64 / 1e6),
        _ => format!("{:.2} GB", n as f64 / 1e9),
    }
}

/// What a streamed response says before its first token, so that a client is not left without a
/// byte while a model downloads, loads, is busy with another request or reads a long prompt on a
/// slow device. By default SSE comment lines (`: loading qwen`), which every SSE client skips
/// and which keep the connection alive; with `stream_options.progress` an `event: progress`
/// with a JSON body instead (an extension: OpenAI's clients do not expect named events here).
/// At most one a second per stage, plus the one that completes a download or a prompt.
struct Stages<'a> {
    model: &'a str,
    events: bool,
    last: Option<(std::mem::Discriminant<Stage<'static>>, std::time::Instant)>,
}

impl<'a> Stages<'a> {
    fn new(model: &'a str, events: bool) -> Self {
        Stages { model, events, last: None }
    }

    /// The bytes to send for `st`, or None when one of its kind went out less than a second ago.
    fn render(&mut self, st: &Stage) -> Option<String> {
        let kind = std::mem::discriminant(&match *st {
            Stage::Waiting => Stage::Waiting,
            Stage::Loading => Stage::Loading,
            Stage::Downloading { .. } => Stage::Downloading {
                file: "",
                index: 0,
                count: 0,
                done: 0,
                total: None,
            },
            Stage::Prompt { .. } => Stage::Prompt { done: 0, total: 0 },
        });
        // The small files of a repository (configuration, tokenizer) come in a moment.
        if matches!(*st, Stage::Downloading { index, total, .. } if index == 0 || total.is_some_and(|t| t < 1_000_000)) {
            return None;
        }
        let complete = match *st {
            Stage::Downloading { done, total, .. } => total == Some(done),
            Stage::Prompt { done, total } => done == total,
            _ => false,
        };
        let now = std::time::Instant::now();
        if !complete
            && self
                .last
                .is_some_and(|(k, t)| k == kind && now.duration_since(t) < std::time::Duration::from_secs(1))
        {
            return None;
        }
        self.last = Some((kind, now));
        let m = self.model;
        let (v, text) = match *st {
            Stage::Waiting => (json!({ "stage": "waiting" }), format!("waiting: {m} is busy with another request")),
            Stage::Loading => (json!({ "stage": "loading" }), format!("loading {m}")),
            Stage::Downloading {
                file,
                index,
                count,
                done,
                total,
            } => (
                json!({ "stage": "downloading", "file": file, "index": index, "count": count, "done": done, "total": total }),
                match total {
                    Some(t) if t > 0 => format!(
                        "downloading {m}: {file} {:.0}% ({} of {})",
                        100.0 * done as f64 / t as f64,
                        size(done),
                        size(t)
                    ),
                    _ => format!("downloading {m}: {file} ({})", size(done)),
                },
            ),
            Stage::Prompt { done, total } => (
                json!({ "stage": "prompt", "done": done, "total": total }),
                format!("reading the prompt: {done} of {total} tokens"),
            ),
        };
        Some(if self.events {
            let mut v = v;
            v["object"] = "progress".into();
            v["model"] = m.into();
            format!("event: progress\ndata: {v}\n\n")
        } else {
            format!(": {text}\n\n")
        })
    }
}

/// Runs the request's `n` choices on its model, streaming the text to `emit` and what it waits
/// for before that to `on`.
fn run_chat(api: &Api, req: &ChatRequest, emit: Emit, on: Option<OnStage<'_, '_>>) -> ApiResult<Vec<Choice>> {
    let mut temp = TempFiles::default();
    let messages = req.messages(api, &mut temp)?;
    let mut on = on;
    let mut lease = api.models.acquire_with(&req.model, Kind::Chat, on.as_deref_mut())?;
    let mut prompt = |done: usize, total: usize| {
        super::activity::prompt(done, total);
        if let Some(on) = on.as_deref_mut() {
            on(&Stage::Prompt { done, total });
        }
    };
    let spec = lease.spec().clone();
    let Loaded::Chat(engine) = lease.model() else { unreachable!() };
    configure(engine, &spec, req.think, &req.tools, req.params.max_tokens);
    let c = engine.config_mut();
    c.response_format = req.response_format.clone();
    c.tool_choice = req.tool_choice.clone();
    c.logprobs = req.params.logprobs;
    let seed = req.params.seed.unwrap_or_else(random_seed);
    let mut out = Vec::new();
    for i in 0..req.params.n {
        engine.set_sampling(req.params.sampling(&spec, seed.wrapping_add(i as u64)));
        let mut filter = StopFilter::new(&req.params.stop);
        let (mut text, mut thinking) = (String::new(), String::new());
        let mut gone = false;
        let mut logs = LogCollector::new(engine, req.params.logprobs.is_some());
        let reply = crate::generate::with_prefill_progress(&mut prompt, || {
            engine.chat(&messages, &mut |d| {
                logs.push(d.logprob);
                super::activity::token();
                let ok = if d.thinking {
                    thinking.push_str(d.text);
                    d.text.is_empty() || emit(i, d.text, true, &[])
                } else {
                    let t = filter.push(d.text);
                    text.push_str(&t);
                    t.is_empty() || emit(i, &t, false, &logs.take())
                };
                // A cancel from the status page stops it as a client gone away would.
                gone |= !ok || super::activity::cancelled();
                if gone || filter.hit { Flow::Stop } else { Flow::Continue }
            })
        })
        .map_err(|e| ApiError::bad(format!("{e:#}")))?;
        let rest = filter.finish();
        let (logprobs, last) = logs.finish(filter.hit);
        if !rest.is_empty() || !last.is_empty() {
            text.push_str(&rest);
            emit(i, &rest, false, &last);
        }
        if gone {
            break;
        }
        out.push(Choice {
            logprobs,
            finish: finish_reason(&reply, filter.hit),
            stop_sequence: filter.which.take(),
            text,
            thinking,
            tool_calls: if filter.hit { Vec::new() } else { reply.tool_calls.clone() },
            prompt_tokens: reply.stats.prompt_tokens + reply.stats.cached_tokens,
            cached_tokens: reply.stats.cached_tokens,
            completion_tokens: reply.stats.generated_tokens,
            generated_chars: 0,
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

/// What a spoken reply reads of a chat request ([`super::speech`]): the model, the system
/// prompt (the last system message, if any), and the last user message's text and attachments
/// (written to `temp`). Earlier turns are not replayed.
pub(super) struct SpokenInput {
    pub model: String,
    pub system: Option<String>,
    pub text: String,
    pub media: Attachments,
    pub seed: Option<u64>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub max_tokens: Option<usize>,
    pub stream: bool,
    pub include_usage: bool,
}

pub(super) fn spoken_input(api: &Api, body: &Value, temp: &mut TempFiles) -> ApiResult<SpokenInput> {
    let req = ChatRequest::parse(body)?;
    let msgs = req.messages(api, temp)?;
    let system = msgs.iter().rev().find(|m| m.role == Role::System).map(|m| m.content.clone());
    let user = msgs
        .iter()
        .rev()
        .find(|m| m.role == Role::User)
        .ok_or_else(|| ApiError::bad("a spoken reply answers a user message: none was given").param("messages"))?;
    Ok(SpokenInput {
        model: req.model.clone(),
        system,
        text: user.content.clone(),
        media: user.media.clone(),
        seed: req.params.seed,
        temperature: req.params.temperature,
        top_p: req.params.top_p,
        top_k: req.params.top_k,
        max_tokens: req.params.max_tokens,
        stream: req.params.stream,
        include_usage: req.params.include_usage,
    })
}

/// `POST /v1/chat/completions`.
pub(super) fn chat(api: &Arc<Api>, body: &Value) -> ApiResult<Output> {
    // Audio out (OpenAI's `modalities: ["text", "audio"]`): LFM3-Omni, Haru.
    if body
        .get("modalities")
        .and_then(Value::as_array)
        .is_some_and(|m| m.iter().any(|x| x == "audio"))
    {
        return super::speech::spoken_chat(api, body);
    }
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
    let choices = run_chat(api, &req, &mut |_, _, _, _| true, None)?;
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
            json!({ "index": i, "message": msg, "logprobs": c.logprobs.as_deref().map(LogEntry::chat), "finish_reason": c.finish })
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
    let chunk_with = |delta: Value, finish: Value, logprobs: Value| {
        json!({
            "id": id, "object": "chat.completion.chunk", "created": created, "model": req.model,
            "choices": [{ "index": 0, "delta": delta, "logprobs": logprobs, "finish_reason": finish }],
        })
    };
    let chunk = |delta: Value, finish: Value| chunk_with(delta, finish, Value::Null);
    if !sse(sink, &chunk(json!({ "role": "assistant", "content": "" }), Value::Null)) {
        return;
    }
    // The text and the stages before it go out through the same sink.
    let out = std::cell::RefCell::new(&mut *sink);
    let mut stages = Stages::new(&req.model, req.params.progress);
    let r = run_chat(
        api,
        req,
        &mut |_, text, thinking, logprobs| {
            let delta = if thinking {
                json!({ "reasoning_content": text })
            } else {
                json!({ "content": text })
            };
            // The entries of the tokens whose text this piece (or an earlier, empty one) holds.
            let logprobs = if req.params.logprobs.is_some() && !thinking {
                LogEntry::chat(logprobs)
            } else {
                Value::Null
            };
            sse(&mut **out.borrow_mut(), &chunk_with(delta, Value::Null, logprobs))
        },
        Some(&mut |st| stages.render(st).is_none_or(|s| (out.borrow_mut())(s.as_bytes()))),
    );
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

// ---- Anthropic Messages API -------------------------------------------------------------------

/// The text of an Anthropic content value: a string, or the text of its `text` blocks.
fn anthropic_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter(|b| b.get("type").and_then(Value::as_str).unwrap_or("text") == "text")
            .filter_map(|b| b.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// An Anthropic image block's source as a URL (`data:` for base64 images).
fn anthropic_image(b: &Value) -> ApiResult<String> {
    let s = b
        .get("source")
        .ok_or_else(|| ApiError::bad("image block without a source").param("messages"))?;
    match s.get("type").and_then(Value::as_str) {
        Some("base64") => {
            let mt = s.get("media_type").and_then(Value::as_str).unwrap_or("image/png");
            let data = s.get("data").and_then(Value::as_str).unwrap_or_default();
            Ok(format!("data:{mt};base64,{data}"))
        }
        Some("url") => Ok(s.get("url").and_then(Value::as_str).unwrap_or_default().to_string()),
        t => Err(ApiError::bad(format!("image source type {t:?} is not supported (base64, url)")).param("messages")),
    }
}

/// An Anthropic Messages request (`/v1/messages`) as the OpenAI chat request it means: the
/// system prompt as a system message, `tool_use` blocks as tool calls, `tool_result` blocks as tool
/// messages, images as image parts, `thinking` as the thinking switch, the tools and tool choice
/// in OpenAI's shape. Consecutive messages of one role are one turn, as Anthropic merges them.
///
/// Thinking blocks of earlier replies whose signature `signer` verifies become the assistant
/// message's `reasoning_content` (the engine does not feed it again: its own record of the
/// conversation holds the reasoning; `count_tokens` counts it within the current tool loop).
/// Blocks that fail (edited, signed by another server or before a restart with a random key, or
/// no signer) are dropped without an error, as are `redacted_thinking` blocks, Anthropic's server
/// tool blocks, and the fields that do not change the reply (`cache_control`, `metadata`, ...).
fn anthropic_to_openai(v: &Value, signer: Option<&super::sign::ThinkingSigner>) -> ApiResult<Value> {
    let bad = |m: &str, p: &'static str| ApiError::bad(m.to_string()).param(p);
    let mut msgs: Vec<Value> = Vec::new();
    match v.get("system") {
        None | Some(Value::Null) => {}
        Some(s) => {
            let t = anthropic_text(s);
            if !t.is_empty() {
                msgs.push(json!({ "role": "system", "content": t }));
            }
        }
    }
    for m in v
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| bad("messages must be an array", "messages"))?
    {
        let role = m.get("role").and_then(Value::as_str).unwrap_or_default();
        let content = m.get("content").unwrap_or(&Value::Null);
        let blocks: Vec<Value> = match content {
            Value::String(s) => vec![json!({ "type": "text", "text": s })],
            Value::Array(a) => a.clone(),
            _ => return Err(bad("content must be a string or an array of blocks", "messages")),
        };
        match role {
            "assistant" => {
                let mut text = String::new();
                let mut thinking = String::new();
                let mut calls = Vec::new();
                for b in &blocks {
                    match b.get("type").and_then(Value::as_str) {
                        Some("text") => text.push_str(b.get("text").and_then(Value::as_str).unwrap_or_default()),
                        Some("tool_use") => calls.push(json!({
                            "id": b.get("id").cloned().unwrap_or(Value::Null),
                            "type": "function",
                            "function": { "name": b.get("name").cloned().unwrap_or(Value::Null), "arguments": b.get("input").cloned().unwrap_or(json!({})) },
                        })),
                        Some("thinking") => {
                            let t = b.get("thinking").and_then(Value::as_str).unwrap_or_default();
                            let sig = b.get("signature").and_then(Value::as_str).unwrap_or_default();
                            if !t.is_empty() && signer.is_some_and(|s| s.verify(t, sig)) {
                                thinking.push_str(t);
                            }
                        }
                        _ => {} // redacted_thinking, server tool blocks
                    }
                }
                // Consecutive assistant messages are one turn.
                if let Some(prev) = msgs
                    .last_mut()
                    .filter(|m| m["role"] == "assistant" && m.get("tool_calls").is_none())
                {
                    let mut t = prev["content"].as_str().unwrap_or_default().to_string();
                    t.push_str(&text);
                    prev["content"] = t.into();
                    if !thinking.is_empty() {
                        let mut r = prev
                            .get("reasoning_content")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string();
                        r.push_str(&thinking);
                        prev["reasoning_content"] = r.into();
                    }
                    if !calls.is_empty() {
                        prev["tool_calls"] = Value::Array(calls);
                    }
                    continue;
                }
                let mut msg = json!({ "role": "assistant", "content": text });
                if !thinking.is_empty() {
                    msg["reasoning_content"] = thinking.into();
                }
                if !calls.is_empty() {
                    msg["tool_calls"] = Value::Array(calls);
                }
                msgs.push(msg);
            }
            "user" => {
                let mut parts: Vec<Value> = Vec::new();
                for b in &blocks {
                    match b.get("type").and_then(Value::as_str) {
                        Some("text") => {
                            parts.push(json!({ "type": "text", "text": b.get("text").and_then(Value::as_str).unwrap_or_default() }))
                        }
                        Some("image") => parts.push(json!({ "type": "image_url", "image_url": { "url": anthropic_image(b)? } })),
                        Some("tool_result") => {
                            let c = b.get("content").unwrap_or(&Value::Null);
                            let mut t = anthropic_text(c);
                            if b.get("is_error").and_then(Value::as_bool) == Some(true) {
                                t = format!("Error: {t}");
                            }
                            msgs.push(json!({ "role": "tool", "tool_call_id": b.get("tool_use_id").cloned().unwrap_or(Value::Null), "content": t }));
                        }
                        Some("document") => {
                            // A plain-text document: its text.
                            if let Some(d) = b.pointer("/source/data").and_then(Value::as_str)
                                && b.pointer("/source/type").and_then(Value::as_str) == Some("text")
                            {
                                parts.push(json!({ "type": "text", "text": d }));
                            }
                        }
                        _ => {}
                    }
                }
                if !parts.is_empty() {
                    // Consecutive user messages are one turn.
                    if let Some(prev) = msgs.last_mut().filter(|m| m["role"] == "user")
                        && let Some(a) = prev["content"].as_array_mut()
                    {
                        a.extend(parts);
                    } else {
                        msgs.push(json!({ "role": "user", "content": parts }));
                    }
                }
            }
            // (Some clients put system turns among the messages.)
            "system" => {
                let t = anthropic_text(content);
                if !t.is_empty() {
                    msgs.push(json!({ "role": "system", "content": t }));
                }
            }
            r => return Err(bad(&format!("unknown message role {r:?} (user, assistant)"), "messages")),
        }
    }
    let mut out = json!({ "model": v.get("model").cloned().unwrap_or(Value::Null), "messages": msgs });
    for (from, to) in [
        ("max_tokens", "max_tokens"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("top_k", "top_k"),
        ("stop_sequences", "stop"),
        ("stream", "stream"),
    ] {
        if let Some(x) = v.get(from).filter(|x| !x.is_null()) {
            out[to] = x.clone();
        }
    }
    match v.pointer("/thinking/type").and_then(Value::as_str) {
        Some("enabled" | "adaptive") => out["think"] = true.into(),
        Some("disabled") => out["think"] = false.into(),
        _ => {}
    }
    if let Some(tools) = v.get("tools").and_then(Value::as_array) {
        // Client tools only: Anthropic's server tools (web search, code execution) have a type.
        let list: Vec<Value> = tools
            .iter()
            .filter(|t| t.get("type").and_then(Value::as_str).is_none_or(|ty| ty == "custom"))
            .map(|t| {
                json!({ "type": "function", "function": {
                    "name": t.get("name").cloned().unwrap_or(Value::Null),
                    "description": t.get("description").cloned().unwrap_or(Value::Null),
                    "parameters": t.get("input_schema").cloned().unwrap_or(json!({ "type": "object", "properties": {} })),
                }})
            })
            .collect();
        if !list.is_empty() {
            out["tools"] = Value::Array(list);
        }
    }
    if let Some(tc) = v.get("tool_choice").filter(|t| !t.is_null())
        && out.get("tools").is_some()
    {
        out["tool_choice"] = match tc.get("type").and_then(Value::as_str) {
            Some("any") => "required".into(),
            Some("none") => "none".into(),
            Some("tool") => json!({ "type": "function", "function": { "name": tc.get("name").cloned().unwrap_or(Value::Null) } }),
            _ => "auto".into(),
        };
    }
    Ok(out)
}

fn anthropic_stop(c: &Choice) -> &'static str {
    match c.finish {
        _ if c.stop_sequence.is_some() => "stop_sequence",
        "tool_calls" => "tool_use",
        "length" => "max_tokens",
        _ => "end_turn",
    }
}

/// An error in Anthropic's shape (`{"type": "error", "error": {"type", "message"}}`).
fn anthropic_error(e: &ApiError) -> Value {
    let kind = match e.status {
        400 | 405 | 413 | 415 | 422 => "invalid_request_error",
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        503 => "overloaded_error",
        _ => "api_error",
    };
    json!({ "type": "error", "error": { "type": kind, "message": e.message } })
}

/// Whether `path` is an endpoint of Anthropic's API (errors in its shape).
pub(super) fn is_anthropic_path(path: &str) -> bool {
    let p = path.trim_end_matches('/');
    let p = p.strip_prefix("/v1").unwrap_or(p);
    p == "/messages" || p == "/messages/count_tokens"
}

/// An error of an Anthropic endpoint as its response.
pub(super) fn anthropic_error_response(e: &ApiError) -> ApiResponse {
    ApiResponse::json(e.status, &anthropic_error(e))
}

fn anthropic_tool_id(api: &Api) -> String {
    api.new_id("toolu_")
}

/// `POST /v1/messages`: Anthropic's Messages API (what Claude Code and the `anthropic` SDKs
/// speak) over the chat models: system prompt, text and image blocks, tools with `tool_use` /
/// `tool_result` blocks, thinking (`thinking` blocks), `max_tokens`, `stop_sequences`, sampling,
/// and streaming in Anthropic's event format. Errors have Anthropic's shape.
pub(super) fn messages(api: &Arc<Api>, body: &Value) -> ApiResult<Output> {
    match messages_inner(api, body) {
        Ok(o) => Ok(o),
        Err(e) => Ok(Output::Full(ApiResponse::json(e.status, &anthropic_error(&e)))),
    }
}

/// The thinking blocks (and redacted ones) of a request's assistant turns: (verified, dropped).
fn thinking_blocks(api: &Api, body: &Value) -> (usize, usize) {
    let (mut ok, mut dropped) = (0, 0);
    let msgs = body.get("messages").and_then(Value::as_array).into_iter().flatten();
    for b in msgs.filter_map(|m| m.get("content").and_then(Value::as_array)).flatten() {
        match b.get("type").and_then(Value::as_str) {
            Some("thinking") => {
                let t = b.get("thinking").and_then(Value::as_str).unwrap_or_default();
                let s = b.get("signature").and_then(Value::as_str).unwrap_or_default();
                if api.signer.verify(t, s) { ok += 1 } else { dropped += 1 }
            }
            Some("redacted_thinking") => dropped += 1,
            _ => {}
        }
    }
    (ok, dropped)
}

fn messages_inner(api: &Arc<Api>, body: &Value) -> ApiResult<Output> {
    if body.get("max_tokens").is_none() {
        return Err(ApiError::bad("max_tokens is required").param("max_tokens"));
    }
    let req = ChatRequest::parse(&anthropic_to_openai(body, Some(&api.signer))?)?;
    api.models.check(&req.model, Kind::Chat)?;
    if api.opts.log {
        let (ok, dropped) = thinking_blocks(api, body);
        if ok + dropped > 0 {
            eprintln!("[messages: thinking blocks sent back: {ok} verified, {dropped} dropped]");
        }
    }
    let id = api.new_id("msg_");
    if req.params.stream {
        let api = api.clone();
        return Ok(Output::Stream {
            content_type: "text/event-stream".into(),
            producer: Box::new(move |sink| {
                stream_messages(&api, &req, &id, sink);
                api.flush_prompt_cache_later(&req.model);
            }),
        });
    }
    let choices = run_chat(api, &req, &mut |_, _, _, _| true, None)?;
    api.flush_prompt_cache_later(&req.model);
    let c = choices.first().ok_or_else(|| ApiError::new(499, "the client closed the request"))?;
    let mut content = Vec::new();
    if !c.thinking.is_empty() {
        content.push(json!({ "type": "thinking", "thinking": c.thinking, "signature": api.signer.sign(&c.thinking) }));
    }
    if !c.text.is_empty() || c.tool_calls.is_empty() {
        content.push(json!({ "type": "text", "text": c.text }));
    }
    for t in &c.tool_calls {
        content
            .push(json!({ "type": "tool_use", "id": anthropic_tool_id(api), "name": t.name, "input": Value::Object(t.arguments.clone()) }));
    }
    Ok(Output::Full(ApiResponse::json(
        200,
        &json!({
            "id": id,
            "type": "message",
            "role": "assistant",
            "model": req.model,
            "content": content,
            "stop_reason": anthropic_stop(c),
            "stop_sequence": c.stop_sequence,
            "usage": {
                "input_tokens": c.prompt_tokens - c.cached_tokens,
                "cache_read_input_tokens": c.cached_tokens,
                "cache_creation_input_tokens": 0,
                "output_tokens": c.completion_tokens,
            },
        }),
    )))
}

fn event(sink: Sink, name: &str, v: &Value) -> bool {
    sink(format!("event: {name}\ndata: {v}\n\n").as_bytes())
}

/// The streamed form: `message_start`, a `thinking` and / or `text` content block with its
/// deltas (a thinking block ends with its `signature_delta`), a `tool_use` block per call (its
/// input in one `input_json_delta`), `message_delta` with the stop reason and `message_stop`;
/// `ping` while the model loads or reads the prompt.
fn stream_messages(api: &Api, req: &ChatRequest, id: &str, sink: Sink) {
    let start = json!({
        "type": "message_start",
        "message": {
            "id": id, "type": "message", "role": "assistant", "model": req.model, "content": [],
            "stop_reason": null, "stop_sequence": null, "usage": { "input_tokens": 0, "output_tokens": 0 },
        },
    });
    if !event(sink, "message_start", &start) {
        return;
    }
    let out = std::cell::RefCell::new(&mut *sink);
    // The open block: its index and whether it is thinking.
    let mut block: Option<(usize, bool)> = None;
    // The open thinking block's text, signed when it closes.
    let thought = std::cell::RefCell::new(String::new());
    let close = |s: &mut dyn FnMut(&[u8]) -> bool, (i, thinking): (usize, bool)| {
        if thinking {
            let sig = api.signer.sign(&std::mem::take(&mut *thought.borrow_mut()));
            let d = json!({ "type": "content_block_delta", "index": i, "delta": { "type": "signature_delta", "signature": sig } });
            if !event(s, "content_block_delta", &d) {
                return false;
            }
        }
        event(s, "content_block_stop", &json!({ "type": "content_block_stop", "index": i }))
    };
    let mut next = 0usize;
    let mut last_ping: Option<std::time::Instant> = None;
    let r = run_chat(
        api,
        req,
        &mut |_, text, thinking, _| {
            let mut s = out.borrow_mut();
            if block.is_some_and(|(_, t)| t != thinking) && !close(&mut **s, block.take().unwrap()) {
                return false;
            }
            if block.is_none() {
                let cb = if thinking {
                    json!({ "type": "thinking", "thinking": "", "signature": "" })
                } else {
                    json!({ "type": "text", "text": "" })
                };
                if !event(
                    &mut **s,
                    "content_block_start",
                    &json!({ "type": "content_block_start", "index": next, "content_block": cb }),
                ) {
                    return false;
                }
                block = Some((next, thinking));
                next += 1;
            }
            let delta = if thinking {
                thought.borrow_mut().push_str(text);
                json!({ "type": "thinking_delta", "thinking": text })
            } else {
                json!({ "type": "text_delta", "text": text })
            };
            let i = block.unwrap().0;
            event(
                &mut **s,
                "content_block_delta",
                &json!({ "type": "content_block_delta", "index": i, "delta": delta }),
            )
        },
        // At most one ping a second.
        Some(&mut |_| {
            if last_ping.is_some_and(|t| t.elapsed() < std::time::Duration::from_secs(1)) {
                return true;
            }
            last_ping = Some(std::time::Instant::now());
            (out.borrow_mut())(b"event: ping\ndata: {\"type\": \"ping\"}\n\n")
        }),
    );
    let sink = out.into_inner();
    match r {
        Ok(choices) => {
            let Some(c) = choices.first() else { return };
            if let Some(b) = block.take() {
                close(sink, b);
            }
            for t in &c.tool_calls {
                let cb = json!({ "type": "tool_use", "id": anthropic_tool_id(api), "name": t.name, "input": {} });
                event(
                    sink,
                    "content_block_start",
                    &json!({ "type": "content_block_start", "index": next, "content_block": cb }),
                );
                let args = Value::Object(t.arguments.clone()).to_string();
                let d =
                    json!({ "type": "content_block_delta", "index": next, "delta": { "type": "input_json_delta", "partial_json": args } });
                event(sink, "content_block_delta", &d);
                event(sink, "content_block_stop", &json!({ "type": "content_block_stop", "index": next }));
                next += 1;
            }
            let d = json!({
                "type": "message_delta",
                "delta": { "stop_reason": anthropic_stop(c), "stop_sequence": c.stop_sequence },
                "usage": { "input_tokens": c.prompt_tokens - c.cached_tokens, "cache_read_input_tokens": c.cached_tokens, "output_tokens": c.completion_tokens },
            });
            event(sink, "message_delta", &d);
            event(sink, "message_stop", &json!({ "type": "message_stop" }));
        }
        Err(e) => {
            event(sink, "error", &anthropic_error(&e));
        }
    }
}

/// `POST /v1/messages/count_tokens`: the tokens of the request's system prompt, messages and tool
/// definitions with the model's tokenizer (loading the model if needed), plus a few per message
/// for the template's markers: an estimate within a few tokens of the formatted prompt.
pub(super) fn count_tokens(api: &Arc<Api>, body: &Value) -> ApiResult<Output> {
    let r = (|| -> ApiResult<Value> {
        let oa = anthropic_to_openai(body, Some(&api.signer))?;
        let model = oa
            .get("model")
            .and_then(Value::as_str)
            .ok_or_else(|| ApiError::bad("model is required").param("model"))?;
        api.models.check(model, Kind::Chat)?;
        let mut text = String::new();
        let mut n_msgs = 0;
        let msgs = oa["messages"].as_array().cloned().unwrap_or_default();
        // The current tool loop (after the last user turn): its replies' reasoning is in the context.
        let loop_start = msgs.iter().rposition(|m| m["role"] == "user").unwrap_or(0);
        for (k, m) in msgs.iter().enumerate() {
            n_msgs += 1;
            if k > loop_start
                && let Some(r) = m["reasoning_content"].as_str()
            {
                text.push_str(r);
            }
            match &m["content"] {
                Value::String(s) => text.push_str(s),
                Value::Array(parts) => parts.iter().filter_map(|p| p["text"].as_str()).for_each(|t| text.push_str(t)),
                _ => {}
            }
            for c in m["tool_calls"].as_array().into_iter().flatten() {
                text.push_str(&c["function"].to_string());
            }
            text.push('\n');
        }
        if let Some(t) = oa.get("tools") {
            text.push_str(&t.to_string());
        }
        let mut lease = api.models.acquire(model, Kind::Chat)?;
        let Loaded::Chat(engine) = lease.model() else { unreachable!() };
        let n = engine.tokenizer().encode(text.as_str(), false).map_err(ApiError::internal)?.len();
        Ok(json!({ "input_tokens": n + 4 * n_msgs }))
    })();
    Ok(Output::Full(match r {
        Ok(v) => ApiResponse::json(200, &v),
        Err(e) => ApiResponse::json(e.status, &anthropic_error(&e)),
    }))
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
                let chunk = |text: &str, finish: Value, logprobs: Value| {
                    json!({
                        "id": id, "object": "text_completion", "created": created, "model": job.model,
                        "choices": [{ "index": 0, "text": text, "logprobs": logprobs, "finish_reason": finish }],
                    })
                };
                let mut offset = 0;
                let mut piece = |sink: Sink, _: usize, t: &str, l: &[LogEntry]| {
                    let logprobs = if job.params.logprobs.is_some() {
                        LogEntry::legacy(l, offset)
                    } else {
                        Value::Null
                    };
                    offset += t.chars().count();
                    sse(sink, &chunk(t, Value::Null, logprobs))
                };
                let out = std::cell::RefCell::new(&mut *sink);
                let mut stages = Stages::new(&job.model, job.params.progress);
                let r = job.run(
                    &api,
                    &mut |i, t, l| piece(&mut **out.borrow_mut(), i, t, l),
                    Some(&mut |st| stages.render(st).is_none_or(|s| (out.borrow_mut())(s.as_bytes()))),
                );
                match r {
                    Ok(choices) => {
                        if let Some(c) = choices.first() {
                            sse(sink, &chunk("", c.finish.into(), Value::Null));
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
    let choices = job.run(api, &mut |_, _, _| true, None)?;
    let list: Vec<Value> = choices
        .iter()
        .enumerate()
        .map(|(i, c)| {
            // Offsets count from the start of `text`, after an echoed prompt.
            let echoed = if job.echo {
                c.text.chars().count().saturating_sub(c.generated_chars)
            } else {
                0
            };
            let logprobs = c.logprobs.as_deref().map(|l| LogEntry::legacy(l, echoed));
            json!({ "index": i, "text": c.text, "logprobs": logprobs, "finish_reason": c.finish })
        })
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
    /// Every prompt `n` times; `emit(choice, text, log probabilities)` streams.
    fn run(
        &self,
        api: &Api,
        emit: &mut dyn FnMut(usize, &str, &[LogEntry]) -> bool,
        on: Option<OnStage<'_, '_>>,
    ) -> ApiResult<Vec<Choice>> {
        let mut on = on;
        let mut lease = api.models.acquire_with(&self.model, Kind::Chat, on.as_deref_mut())?;
        let mut reading = |done: usize, total: usize| {
            super::activity::prompt(done, total);
            if let Some(on) = on.as_deref_mut() {
                on(&Stage::Prompt { done, total });
            }
        };
        let spec = lease.spec().clone();
        let Loaded::Chat(engine) = lease.model() else { unreachable!() };
        let c = engine.config_mut();
        c.max_tokens = self.params.max_tokens.unwrap_or(16);
        // A raw continuation: nothing of the last chat request constrains it.
        c.response_format = ResponseFormat::Text;
        c.tool_choice = ToolChoice::Auto;
        c.logprobs = self.params.logprobs;
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
                if self.echo && !emit(k, &prompt, &[]) {
                    break;
                }
                let mut gone = false;
                let echoed = text.chars().count();
                let mut logs = LogCollector::new(engine, self.params.logprobs.is_some());
                let reply = crate::generate::with_prefill_progress(&mut reading, || {
                    engine.complete(&prompt, &mut |d| {
                        logs.push(d.logprob);
                        super::activity::token();
                        let t = filter.push(d.text);
                        text.push_str(&t);
                        gone |= (!t.is_empty() && !emit(k, &t, &logs.take())) || super::activity::cancelled();
                        if gone || filter.hit { Flow::Stop } else { Flow::Continue }
                    })
                })
                .map_err(|e| ApiError::bad(format!("{e:#}")))?;
                let rest = filter.finish();
                let (logprobs, last) = logs.finish(filter.hit);
                if !rest.is_empty() || !last.is_empty() {
                    text.push_str(&rest);
                    emit(k, &rest, &last);
                }
                if gone {
                    return Ok(out);
                }
                out.push(Choice {
                    logprobs,
                    generated_chars: text.chars().count() - echoed,
                    finish: finish_reason(&reply, filter.hit),
                    stop_sequence: filter.which.take(),
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
        if let Some(d) = dims
            && (d == 0 || (d > m.emb.dim() && !m.emb.head_dims().contains(&d)))
        {
            return Err(ApiError::bad(format!("cannot give {d} dimensions (the model has {})", m.emb.dim())).param("dimensions"));
        }
        let e = m
            .emb
            .embed_dim(&text, prompt_name.as_deref(), dims)
            .map_err(|e| ApiError::bad(format!("{e:#}")))?;
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
    fn stages_are_comments_or_events_and_throttled() {
        let dl = |done| Stage::Downloading {
            file: "model.safetensors",
            index: 1,
            count: 1,
            done,
            total: Some(2_000_000_000),
        };
        let mut s = Stages::new("qwen", false);
        assert_eq!(s.render(&Stage::Loading).unwrap(), ": loading qwen\n\n");
        assert_eq!(
            s.render(&dl(500_000_000)).unwrap(),
            ": downloading qwen: model.safetensors 25% (500.0 MB of 2.00 GB)\n\n"
        );
        // The same stage again within the second is dropped, unless it completes it.
        assert!(s.render(&dl(600_000_000)).is_none());
        assert!(s.render(&dl(2_000_000_000)).unwrap().contains("100%"));
        assert_eq!(
            s.render(&Stage::Waiting).unwrap(),
            ": waiting: qwen is busy with another request\n\n"
        );
        assert_eq!(
            s.render(&Stage::Prompt { done: 256, total: 900 }).unwrap(),
            ": reading the prompt: 256 of 900 tokens\n\n"
        );
        assert!(s.render(&Stage::Prompt { done: 512, total: 900 }).is_none());
        assert!(s.render(&Stage::Prompt { done: 900, total: 900 }).is_some());
        // As events (stream_options.progress): one JSON object per event.
        let mut s = Stages::new("qwen", true);
        let e = s.render(&Stage::Prompt { done: 256, total: 900 }).unwrap();
        let data = e.strip_prefix("event: progress\ndata: ").unwrap().strip_suffix("\n\n").unwrap();
        let v: Value = serde_json::from_str(data).unwrap();
        assert_eq!(
            v,
            json!({"object": "progress", "model": "qwen", "stage": "prompt", "done": 256, "total": 900})
        );
        let p = ChatRequest::parse(&json!({
            "model": "m", "messages": [{"role": "user", "content": "x"}],
            "stream": true, "stream_options": {"progress": true},
        }))
        .unwrap();
        assert!(p.params.progress);
    }

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
        assert_eq!(r.tool_choice, ToolChoice::Function("look".into()));
        assert_eq!(r.response_format, ResponseFormat::Text);
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
    fn constraints_and_logprobs_parsing() {
        let with = |extra: Value| {
            let mut v = json!({"model": "m", "messages": [{"role": "user", "content": "x"}]});
            for (k, x) in extra.as_object().unwrap() {
                v[k] = x.clone();
            }
            ChatRequest::parse(&v)
        };
        let tools = json!([{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {"a": {"type": "integer"}}}}},
                           {"type": "function", "function": {"name": "g"}}]);
        let r = with(json!({"tools": tools, "tool_choice": "required", "logprobs": true, "top_logprobs": 5})).unwrap();
        assert_eq!(
            (r.tool_choice, r.tools.len(), r.params.logprobs),
            (ToolChoice::Required, 2, Some(5))
        );
        assert_eq!(with(json!({"logprobs": true})).unwrap().params.logprobs, Some(0));
        assert_eq!(with(json!({"logprobs": false})).unwrap().params.logprobs, None);
        assert_eq!(with(json!({"top_logprobs": 3})).unwrap_err().param, Some("top_logprobs"));
        assert_eq!(
            with(json!({"logprobs": true, "top_logprobs": 21})).unwrap_err().param,
            Some("top_logprobs")
        );
        assert_eq!(with(json!({"tool_choice": "required"})).unwrap_err().param, Some("tool_choice"));
        // A forced call needs a parameter schema the grammar can follow.
        let odd = json!([{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {"a": {"type": "integer", "minimum": 1}}}}}]);
        let e = with(json!({"tools": odd, "tool_choice": "required"})).unwrap_err();
        assert!(
            e.param == Some("tools") && e.status == 400 && e.message.contains("\"minimum\" is not supported"),
            "{}",
            e.message
        );
        assert!(with(json!({"tools": odd})).is_ok());

        let schema = json!({"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"], "additionalProperties": false});
        let r = with(json!({"response_format": {"type": "json_schema", "json_schema": {"name": "n", "strict": true, "schema": schema}}}))
            .unwrap();
        assert_eq!(r.response_format, ResponseFormat::JsonSchema(schema));
        assert_eq!(
            with(json!({"response_format": {"type": "json_object"}})).unwrap().response_format,
            ResponseFormat::JsonObject
        );
        assert_eq!(
            with(json!({"response_format": {"type": "text"}})).unwrap().response_format,
            ResponseFormat::Text
        );
        let e = with(json!({"response_format": {"type": "json_schema", "json_schema": {"name": "n", "schema": {"type": "string", "pattern": "^a"}}}})).unwrap_err();
        assert!(
            e.status == 400 && e.param == Some("response_format") && e.message.contains("\"pattern\" is not supported"),
            "{}",
            e.message
        );
        assert!(with(json!({"response_format": {"type": "json_schema"}})).is_err());
        assert!(with(json!({"response_format": {"type": "yaml"}})).is_err());
        // Completions take the number of alternatives.
        assert_eq!(Gen::parse(&json!({"logprobs": 3}), 1).unwrap().logprobs, Some(3));
        assert!(Gen::parse(&json!({"logprobs": 99}), 1).is_err());
    }

    #[test]
    fn logprob_shapes() {
        let tokens = vec![(b"Hel".to_vec(), true), (b"lo".to_vec(), true), (vec![0xe6, 0x97], true)];
        let vocab = Vocab::from_tokens(tokens, Vec::new());
        let entry = |token, logprob: f32, top: &[(u32, f32)]| {
            let l = TokenLogprob {
                token,
                logprob,
                top: top.to_vec(),
                content: true,
            };
            LogEntry::new(&vocab, &l)
        };
        let list = [entry(0, -0.5, &[(0, -0.5), (1, -2.0)]), entry(2, f32::NEG_INFINITY, &[])];
        let chat = LogEntry::chat(&list);
        assert_eq!(
            chat["content"][0],
            json!({"token": "Hel", "logprob": -0.5, "bytes": [72, 101, 108],
            "top_logprobs": [{"token": "Hel", "logprob": -0.5, "bytes": [72, 101, 108]}, {"token": "lo", "logprob": -2.0, "bytes": [108, 111]}]})
        );
        // Part of a character: its bytes, and the floor for a probability of zero.
        assert_eq!(chat["content"][1]["bytes"], json!([0xe6, 0x97]));
        assert_eq!(chat["content"][1]["logprob"], json!(-9999.0));
        let legacy = LogEntry::legacy(&list[..1], 4);
        assert_eq!(
            legacy,
            json!({"tokens": ["Hel"], "token_logprobs": [-0.5], "top_logprobs": [{"Hel": -0.5, "lo": -2.0}], "text_offset": [4]})
        );
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

    #[test]
    fn anthropic_messages_as_chat() {
        let v = json!({
            "model": "m", "max_tokens": 100, "stream": true, "stop_sequences": ["END"],
            "system": [{"type": "text", "text": "Be brief.", "cache_control": {"type": "ephemeral"}}],
            "thinking": {"type": "enabled", "budget_tokens": 1000},
            "tools": [
                {"name": "read", "description": "Reads a file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}},
                {"type": "web_search_20250305", "name": "web_search"}
            ],
            "tool_choice": {"type": "tool", "name": "read"},
            "messages": [
                {"role": "user", "content": "Read a.txt"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "...", "signature": "x"},
                    {"type": "text", "text": "Reading."},
                    {"type": "tool_use", "id": "toolu_1", "name": "read", "input": {"path": "a.txt"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "hello"}]},
                    {"type": "text", "text": "And now?"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}
                ]}
            ]
        });
        let o = anthropic_to_openai(&v, None).unwrap();
        assert!(o["messages"][2].get("reasoning_content").is_none(), "unsigned thinking is dropped");
        let r = ChatRequest::parse(&o).unwrap();
        assert_eq!(r.messages.len(), 5);
        assert_eq!((r.messages[0].role, r.messages[0].text.as_str()), (Role::System, "Be brief."));
        assert_eq!(r.messages[2].text, "Reading.");
        assert_eq!(r.messages[2].tool_calls[0].1.name, "read");
        assert_eq!(r.messages[2].tool_calls[0].1.arguments["path"], "a.txt");
        assert_eq!(
            (r.messages[3].role, r.messages[3].text.as_str(), r.messages[3].name.as_deref()),
            (Role::Tool, "hello", Some("read"))
        );
        assert_eq!(r.messages[4].text, "And now?");
        assert_eq!(r.messages[4].media, vec![MediaPart::Image("data:image/png;base64,AAAA".into())]);
        assert_eq!(r.tools.len(), 1, "server tools are left out");
        assert_eq!(r.tool_choice, ToolChoice::Function("read".into()));
        assert_eq!((r.think, r.params.stream, r.params.max_tokens), (Some(true), true, Some(100)));
        assert_eq!(r.params.stop, vec!["END".to_string()]);
        let choice = |finish, stop: Option<&str>| Choice {
            logprobs: None,
            text: String::new(),
            thinking: String::new(),
            tool_calls: Vec::new(),
            finish,
            stop_sequence: stop.map(str::to_string),
            prompt_tokens: 0,
            cached_tokens: 0,
            completion_tokens: 0,
            generated_chars: 0,
        };
        assert_eq!(anthropic_stop(&choice("tool_calls", None)), "tool_use");
        assert_eq!(anthropic_stop(&choice("length", None)), "max_tokens");
        assert_eq!(anthropic_stop(&choice("stop", None)), "end_turn");
        assert_eq!(anthropic_stop(&choice("stop", Some("END"))), "stop_sequence");
        let any = anthropic_to_openai(
            &json!({"model": "m", "messages": [], "tools": v["tools"], "tool_choice": {"type": "any"}}),
            None,
        )
        .unwrap();
        assert_eq!(any["tool_choice"], "required");
        assert!(anthropic_to_openai(&json!({"model": "m", "messages": [{"role": "tool", "content": "x"}]}), None).is_err());
        let sys = anthropic_to_openai(
            &json!({"model": "m", "messages": [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}]}),
            None,
        )
        .unwrap();
        assert_eq!(sys["messages"][0], json!({"role": "system", "content": "x"}));
    }

    #[test]
    fn anthropic_thinking_blocks() {
        let signer = super::super::sign::ThinkingSigner::new(Some("k"));
        let good = signer.sign("First I list the files.");
        let v = json!({
            "model": "m", "max_tokens": 10, "metadata": {"user_id": "u"},
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "List", "cache_control": {"type": "ephemeral"}}]},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "First I list the files.", "signature": good},
                    {"type": "redacted_thinking", "data": "opaque"},
                    {"type": "thinking", "thinking": "Edited.", "signature": good},
                    {"type": "tool_use", "id": "toolu_1", "name": "ls", "input": {}}
                ]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "a.txt"}]},
                // Interleaved: a second reply of the same tool loop, split in two messages.
                {"role": "assistant", "content": [{"type": "text", "text": "Found "}]},
                {"role": "assistant", "content": [{"type": "text", "text": "a.txt."}]},
                {"role": "user", "content": "Thanks"},
                {"role": "user", "content": [{"type": "text", "text": "Bye"}]}
            ]
        });
        let o = anthropic_to_openai(&v, Some(&signer)).unwrap();
        let m = o["messages"].as_array().unwrap();
        assert_eq!(m.len(), 5, "{m:?}");
        assert_eq!(m[1]["reasoning_content"], "First I list the files.", "only the verified block");
        assert_eq!(m[1]["tool_calls"][0]["function"]["name"], "ls");
        assert_eq!(m[2]["role"], "tool");
        assert_eq!(m[3], json!({"role": "assistant", "content": "Found a.txt."}));
        assert_eq!(m[4]["content"].as_array().unwrap().len(), 2);
        // Another server's key: dropped, no error.
        let o = anthropic_to_openai(&v, Some(&super::super::sign::ThinkingSigner::new(Some("other")))).unwrap();
        assert!(o["messages"][1].get("reasoning_content").is_none());
        let r = ChatRequest::parse(&o).unwrap();
        assert_eq!(r.messages.len(), 5);
        assert!(is_anthropic_path("/v1/messages") && is_anthropic_path("/messages/count_tokens/") && !is_anthropic_path("/v1/models"));
    }
}
