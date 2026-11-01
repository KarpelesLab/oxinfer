//! Tool (function) calling: tool declarations in each model family's own syntax, parsing of the
//! calls a model emits, and a few built-in tools for the command line. The host runs the tools;
//! `chat::ChatFormat` feeds the results back in the format the model was trained on.
//!
//! Gemma 4 declares tools as `<|tool>declaration:NAME{...}<tool|>` in the system turn and calls
//! them with `<|tool_call>call:NAME{key:<|"|>value<|"|>}<tool_call|>`; Qwen 3.5 / 3.8 lists JSON
//! schemas inside `<tools>` and calls with `<tool_call><function=NAME><parameter=key>value
//! </parameter></function></tool_call>`.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

/// A tool the model may call: a name, what it does, and a JSON schema (`type: object`) for its
/// arguments.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    #[serde(default = "empty_object")]
    pub parameters: Value,
}

fn empty_object() -> Value {
    json!({ "type": "object", "properties": {} })
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Map<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolResult {
    pub name: String,
    pub content: String,
}

// ---- Gemma 4 ------------------------------------------------------------------------------------

pub mod gemma {
    use super::*;

    const Q: &str = "<|\"|>";

    /// A value in Gemma's argument syntax (strings in `<|"|>` quotes, keys bare).
    pub fn format_argument(v: &Value) -> String {
        match v {
            Value::Null => "null".into(),
            Value::Bool(b) => b.to_string(),
            Value::Number(n) => n.to_string(),
            Value::String(s) => format!("{Q}{s}{Q}"),
            Value::Array(a) => format!("[{}]", a.iter().map(format_argument).collect::<Vec<_>>().join(",")),
            Value::Object(o) => {
                let mut keys: Vec<&String> = o.keys().collect();
                keys.sort();
                format!(
                    "{{{}}}",
                    keys.iter()
                        .map(|k| format!("{k}:{}", format_argument(&o[*k])))
                        .collect::<Vec<_>>()
                        .join(",")
                )
            }
        }
    }

    fn quoted_list(items: &[Value]) -> String {
        items
            .iter()
            .filter_map(|v| v.as_str())
            .map(|s| format!("{Q}{s}{Q}"))
            .collect::<Vec<_>>()
            .join(",")
    }

    /// The template's `format_parameters`: one entry per property, sorted by name.
    fn format_parameters(props: &Map<String, Value>) -> String {
        let mut keys: Vec<&String> = props.keys().collect();
        keys.sort();
        let mut out = Vec::new();
        for k in keys {
            let p = &props[k];
            let ty = p["type"].as_str().unwrap_or("string").to_uppercase();
            let mut fields = Vec::new();
            if let Some(d) = p["description"].as_str() {
                fields.push(format!("description:{Q}{d}{Q}"));
            }
            if ty == "STRING"
                && let Some(e) = p["enum"].as_array()
            {
                fields.push(format!("enum:{}", format_argument(&Value::Array(e.clone()))));
            }
            if ty == "ARRAY"
                && let Some(items) = p["items"].as_object()
            {
                let mut item_keys: Vec<&String> = items.keys().collect();
                item_keys.sort();
                let mut parts = Vec::new();
                for ik in item_keys {
                    let iv = &items[ik];
                    match ik.as_str() {
                        "properties" => parts.push(format!(
                            "properties:{{{}}}",
                            iv.as_object().map(format_parameters).unwrap_or_default()
                        )),
                        "required" => parts.push(format!("required:[{}]", quoted_list(iv.as_array().map(|a| &a[..]).unwrap_or(&[])))),
                        "type" => parts.push(format!(
                            "type:{}",
                            format_argument(&Value::String(iv.as_str().unwrap_or("string").to_uppercase()))
                        )),
                        _ if !iv.is_null() => parts.push(format!("{ik}:{}", format_argument(iv))),
                        _ => {}
                    }
                }
                fields.push(format!("items:{{{}}}", parts.join(",")));
            }
            if p["nullable"].as_bool() == Some(true) {
                fields.push("nullable:true".into());
            }
            if ty == "OBJECT" {
                if let Some(sub) = p["properties"].as_object() {
                    fields.push(format!("properties:{{{}}}", format_parameters(sub)));
                }
                if let Some(r) = p["required"].as_array().filter(|r| !r.is_empty()) {
                    fields.push(format!("required:[{}]", quoted_list(r)));
                }
            }
            fields.push(format!("type:{Q}{ty}{Q}"));
            out.push(format!("{k}:{{{}}}", fields.join(",")));
        }
        out.join(",")
    }

    /// `<|tool>declaration:...<tool|>` for one tool (the template's `format_function_declaration`).
    pub fn declaration(t: &ToolSpec) -> String {
        let mut s = format!("<|tool>declaration:{}{{description:{Q}{}{Q}", t.name, t.description);
        let p = &t.parameters;
        if p.is_object() && !p.as_object().unwrap().is_empty() {
            s.push_str(",parameters:{");
            if let Some(props) = p["properties"].as_object().filter(|m| !m.is_empty()) {
                s.push_str(&format!("properties:{{{}}},", format_parameters(props)));
            }
            if let Some(r) = p["required"].as_array().filter(|r| !r.is_empty()) {
                s.push_str(&format!("required:[{}],", quoted_list(r)));
            }
            if let Some(ty) = p["type"].as_str() {
                s.push_str(&format!("type:{Q}{}{Q}}}", ty.to_uppercase()));
            }
        }
        s.push_str("}<tool|>");
        s
    }

    /// A tool result block, `<|tool_response>response:NAME{value:...}<tool_response|>`.
    pub fn response(r: &ToolResult) -> String {
        format!(
            "<|tool_response>response:{}{{value:{}}}<tool_response|>",
            r.name,
            format_argument(&Value::String(r.content.clone()))
        )
    }

    /// Parses Gemma's argument syntax: `{key:<|"|>text<|"|>,n:3,flag:true,list:[...],obj:{...}}`.
    struct Parser<'a> {
        s: &'a str,
        i: usize,
    }

    impl Parser<'_> {
        fn rest(&self) -> &str {
            &self.s[self.i..]
        }
        fn skip_ws(&mut self) {
            while self.rest().starts_with(char::is_whitespace) {
                self.i += self.rest().chars().next().unwrap().len_utf8();
            }
        }
        fn value(&mut self) -> Option<Value> {
            self.skip_ws();
            let r = self.rest();
            if let Some(after) = r.strip_prefix(Q) {
                let end = after.find(Q)?;
                let v = after[..end].to_string();
                self.i += Q.len() + end + Q.len();
                return Some(Value::String(v));
            }
            if r.starts_with('{') {
                return self.object().map(Value::Object);
            }
            if r.starts_with('[') {
                self.i += 1;
                let mut items = Vec::new();
                loop {
                    self.skip_ws();
                    if self.rest().starts_with(']') {
                        self.i += 1;
                        return Some(Value::Array(items));
                    }
                    items.push(self.value()?);
                    self.skip_ws();
                    if self.rest().starts_with(',') {
                        self.i += 1;
                    }
                }
            }
            // A bare literal up to the next delimiter.
            let end = r.find([',', '}', ']']).unwrap_or(r.len());
            let lit = r[..end].trim().to_string();
            self.i += end;
            Some(serde_json::from_str(&lit).unwrap_or(Value::String(lit)))
        }
        fn object(&mut self) -> Option<Map<String, Value>> {
            self.skip_ws();
            if !self.rest().starts_with('{') {
                return None;
            }
            self.i += 1;
            let mut m = Map::new();
            loop {
                self.skip_ws();
                if self.rest().starts_with('}') {
                    self.i += 1;
                    return Some(m);
                }
                let colon = self.rest().find(':')?;
                let key = self.rest()[..colon].trim().trim_matches('"').replace(Q, "");
                self.i += colon + 1;
                let v = self.value()?;
                m.insert(key, v);
                self.skip_ws();
                if self.rest().starts_with(',') {
                    self.i += 1;
                }
            }
        }
    }

    /// Every `<|tool_call>call:NAME{...}<tool_call|>` in `raw` (decoded with special tokens).
    pub fn parse_calls(raw: &str) -> Vec<ToolCall> {
        let mut calls = Vec::new();
        let mut rest = raw;
        while let Some(start) = rest.find("<|tool_call>") {
            let body = &rest[start + "<|tool_call>".len()..];
            let end = body.find("<tool_call|>").unwrap_or(body.len());
            let call = body[..end].trim();
            let call = call.strip_prefix("call:").unwrap_or(call);
            let brace = call.find('{').unwrap_or(call.len());
            let name = call[..brace].trim().to_string();
            let mut p = Parser { s: call, i: brace };
            let arguments = if brace < call.len() {
                p.object().unwrap_or_default()
            } else {
                Map::new()
            };
            if !name.is_empty() {
                calls.push(ToolCall { name, arguments });
            }
            rest = &body[end..];
        }
        calls
    }
}

// ---- Mistral (Large 4) ---------------------------------------------------------------------------

pub mod mistral {
    use super::*;

    /// `[AVAILABLE_TOOLS][...][/AVAILABLE_TOOLS]`: the function schemas as the template's `tojson`
    /// writes them.
    pub fn available_tools(tools: &[ToolSpec]) -> String {
        let list: Vec<Value> = tools
            .iter()
            .map(
                |t| json!({ "type": "function", "function": { "name": t.name, "description": t.description, "parameters": t.parameters } }),
            )
            .collect();
        format!("[AVAILABLE_TOOLS]{}[/AVAILABLE_TOOLS]", qwen::py_json(&Value::Array(list)))
    }

    /// Every `[TOOL_CALLS]NAME[ARGS]{...}` (arguments as JSON; a JSON string holding an object is
    /// read too).
    pub fn parse_calls(raw: &str) -> Vec<ToolCall> {
        let mut calls = Vec::new();
        for part in raw.split("[TOOL_CALLS]").skip(1) {
            let Some((name, args)) = part.split_once("[ARGS]") else { continue };
            let args = args.trim();
            // The arguments run to the end of the JSON value (the reply may go on after it).
            let mut de = serde_json::Deserializer::from_str(args).into_iter::<Value>();
            let arguments = match de.next() {
                Some(Ok(Value::Object(o))) => o,
                Some(Ok(Value::String(s))) => serde_json::from_str(&s).unwrap_or_default(),
                _ => Map::new(),
            };
            calls.push(ToolCall {
                name: name.trim().to_string(),
                arguments,
            });
        }
        calls
    }

    /// A tool message: `[TOOL_RESULTS]content[/TOOL_RESULTS]`.
    pub fn response(r: &ToolResult) -> String {
        format!("[TOOL_RESULTS]{}[/TOOL_RESULTS]", r.content)
    }
}

// ---- Qwen 3.5 / 3.8 -----------------------------------------------------------------------------

pub mod qwen {
    use super::*;

    /// JSON as Python's `json.dumps` writes it (", " and ": " separators, keys in order, non-ASCII
    /// kept): what chat templates produce with `tojson`.
    pub fn py_json(v: &Value) -> String {
        match v {
            Value::Array(a) => format!("[{}]", a.iter().map(py_json).collect::<Vec<_>>().join(", ")),
            Value::Object(o) => format!(
                "{{{}}}",
                o.iter()
                    .map(|(k, v)| format!("{}: {}", Value::String(k.clone()), py_json(v)))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            other => other.to_string(),
        }
    }

    /// The system-prompt block the Qwen template inserts when tools are given.
    pub fn tools_block(tools: &[ToolSpec]) -> String {
        let mut s = String::from("# Tools\n\nYou have access to the following functions:\n\n<tools>");
        for t in tools {
            let v = json!({ "type": "function", "function": { "name": t.name, "description": t.description, "parameters": t.parameters } });
            s.push('\n');
            s.push_str(&py_json(&v));
        }
        s.push_str("\n</tools>");
        s.push_str(
            "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>",
        );
        s
    }

    /// Qwen3's tools block: JSON signatures, and calls answered as JSON objects.
    pub fn tools_block_json(tools: &[ToolSpec]) -> String {
        let mut s = String::from(
            "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>",
        );
        for t in tools {
            let v = json!({ "type": "function", "function": { "name": t.name, "description": t.description, "parameters": t.parameters } });
            s.push('\n');
            s.push_str(&py_json(&v));
        }
        s.push_str("\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>");
        s
    }

    /// Every `<tool_call><function=NAME><parameter=K>V</parameter>...</function></tool_call>`, or
    /// (Qwen3) `<tool_call>{"name": NAME, "arguments": {...}}</tool_call>`.
    /// Parameter values are JSON when they parse as JSON, strings otherwise (as the template writes).
    pub fn parse_calls(raw: &str) -> Vec<ToolCall> {
        let mut calls = Vec::new();
        let mut rest = raw;
        while let Some(start) = rest.find("<tool_call>") {
            let body = &rest[start + "<tool_call>".len()..];
            let end = body.find("</tool_call>").unwrap_or(body.len());
            let block = &body[..end];
            if let Ok(Value::Object(o)) = serde_json::from_str::<Value>(block.trim()) {
                let name = o.get("name").and_then(Value::as_str).unwrap_or_default().to_string();
                let arguments = match o.get("arguments") {
                    Some(Value::Object(a)) => a.clone(),
                    // Some models write the arguments as a JSON string.
                    Some(Value::String(a)) => serde_json::from_str(a).unwrap_or_default(),
                    _ => Map::new(),
                };
                if !name.is_empty() {
                    calls.push(ToolCall { name, arguments });
                }
            } else if let Some(f) = block.find("<function=") {
                let after = &block[f + "<function=".len()..];
                let name_end = after.find('>').unwrap_or(after.len());
                let name = after[..name_end].trim().to_string();
                let mut arguments = Map::new();
                let mut params = &after[name_end.min(after.len())..];
                while let Some(ps) = params.find("<parameter=") {
                    let a = &params[ps + "<parameter=".len()..];
                    let ke = a.find('>').unwrap_or(a.len());
                    let key = a[..ke].trim().to_string();
                    let v = &a[(ke + 1).min(a.len())..];
                    let ve = v.find("</parameter>").unwrap_or(v.len());
                    let raw_v = v[..ve].trim_matches('\n');
                    let value = serde_json::from_str::<Value>(raw_v.trim())
                        .ok()
                        .filter(|j| !j.is_string())
                        .unwrap_or_else(|| Value::String(raw_v.to_string()));
                    arguments.insert(key, value);
                    params = &v[ve.min(v.len())..];
                }
                if !name.is_empty() {
                    calls.push(ToolCall { name, arguments });
                }
            }
            rest = &body[end..];
        }
        calls
    }
}

// ---- built-in tools (command line) --------------------------------------------------------------

/// Local wall-clock time: (`YYYY-MM-DD HH:MM:SS`, weekday, UTC offset in seconds, zone name).
pub fn local_now() -> (String, &'static str, i64, String) {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let (offset, zone) = local_offset(secs);
    let t = secs + offset;
    let (days, rem) = (t.div_euclid(86400), t.rem_euclid(86400));
    // Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    const DAYS: [&str; 7] = ["Thursday", "Friday", "Saturday", "Sunday", "Monday", "Tuesday", "Wednesday"];
    let wd = DAYS[days.rem_euclid(7) as usize];
    (
        format!("{y:04}-{m:02}-{d:02} {:02}:{:02}:{:02}", rem / 3600, rem % 3600 / 60, rem % 60),
        wd,
        offset,
        zone,
    )
}

#[cfg(unix)]
// `c_long` is i64 on 64-bit targets and i32 on 32-bit ones: the conversions are not no-ops everywhere.
#[allow(clippy::useless_conversion, clippy::unnecessary_cast)]
fn local_offset(secs: i64) -> (i64, String) {
    // struct tm as laid out by glibc, musl, bionic and Darwin (tm_gmtoff and tm_zone at the end).
    #[repr(C)]
    struct Tm {
        sec: i32,
        min: i32,
        hour: i32,
        mday: i32,
        mon: i32,
        year: i32,
        wday: i32,
        yday: i32,
        isdst: i32,
        gmtoff: std::ffi::c_long,
        zone: *const std::ffi::c_char,
    }
    unsafe extern "C" {
        // time_t is `long` on the Unix targets built here (glibc, musl, bionic, Darwin).
        fn localtime_r(t: *const std::ffi::c_long, out: *mut Tm) -> *mut Tm;
    }
    let mut tm = std::mem::MaybeUninit::<Tm>::zeroed();
    // SAFETY: localtime_r fills `tm` and returns it, or null on failure.
    let t = secs as std::ffi::c_long;
    let ok = unsafe { !localtime_r(&t, tm.as_mut_ptr()).is_null() };
    if !ok {
        return (0, "UTC".into());
    }
    let tm = unsafe { tm.assume_init() };
    let zone = if tm.zone.is_null() {
        String::new()
    } else {
        unsafe { std::ffi::CStr::from_ptr(tm.zone) }.to_string_lossy().into_owned()
    };
    (i64::from(tm.gmtoff), zone)
}

#[cfg(not(unix))]
fn local_offset(_secs: i64) -> (i64, String) {
    (0, "UTC".into())
}

/// "2026-10-24 14:05:09 Friday (UTC+09:00, JST)".
pub fn now_string() -> String {
    let (t, wd, off, zone) = local_now();
    let sign = if off < 0 { '-' } else { '+' };
    let a = off.abs();
    let z = if zone.is_empty() { String::new() } else { format!(", {zone}") };
    format!("{t} {wd} (UTC{sign}{:02}:{:02}{z})", a / 3600, a % 3600 / 60)
}

/// The default system prompt of the command line.
pub fn default_system_prompt(tools: bool) -> String {
    let mut s = format!(
        "You are a helpful assistant running locally on the user's computer with oxinfer, an open-source inference engine. You have no internet access. The current local date and time is {}.",
        now_string()
    );
    if tools {
        s.push_str(" Use the provided tools when they help answer accurately; otherwise answer directly.");
    }
    s
}

pub fn builtin_specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: "get_current_datetime".into(),
            description: "Returns the current local date, time, weekday and time zone.".into(),
            parameters: empty_object(),
        },
        ToolSpec {
            name: "calculate".into(),
            description: "Evaluates an arithmetic expression exactly. Supports + - * / % ^, parentheses, and sqrt, sin, cos, tan, ln, log10, exp, abs, floor, ceil, round, pi, e.".into(),
            parameters: json!({
                "type": "object",
                "properties": { "expression": { "type": "string", "description": "The expression, for example (3 + 4) * 2 ^ 10" } },
                "required": ["expression"]
            }),
        },
    ]
}

/// Runs a built-in tool; unknown tools and bad arguments come back as error text for the model.
pub fn run_builtin(call: &ToolCall) -> String {
    match call.name.as_str() {
        "get_current_datetime" => now_string(),
        "calculate" => match call.arguments.get("expression").and_then(|v| v.as_str()) {
            Some(e) => match calc::eval(e) {
                Ok(v) => calc::format(v),
                Err(err) => format!("error: {err}"),
            },
            None => "error: missing argument \"expression\"".into(),
        },
        other => format!("error: unknown tool {other}"),
    }
}

/// A small arithmetic evaluator for the `calculate` tool.
pub mod calc {
    pub fn format(v: f64) -> String {
        if v.fract() == 0.0 && v.abs() < 1e15 {
            format!("{}", v as i64)
        } else {
            format!("{v}")
        }
    }

    pub fn eval(s: &str) -> Result<f64, String> {
        let toks = lex(s)?;
        let mut p = P { t: &toks, i: 0 };
        let v = p.expr()?;
        if p.i != toks.len() {
            return Err(format!("unexpected {:?}", toks[p.i]));
        }
        if !v.is_finite() {
            return Err("result is not a finite number".into());
        }
        Ok(v)
    }

    #[derive(Debug, Clone, PartialEq)]
    enum T {
        Num(f64),
        Id(String),
        Op(char),
    }

    fn lex(s: &str) -> Result<Vec<T>, String> {
        let c: Vec<char> = s.chars().collect();
        let mut out = Vec::new();
        let mut i = 0;
        while i < c.len() {
            let ch = c[i];
            if ch.is_whitespace() {
                i += 1;
            } else if ch.is_ascii_digit() || ch == '.' {
                let st = i;
                while i < c.len() && (c[i].is_ascii_digit() || c[i] == '.' || c[i] == '_') {
                    i += 1;
                }
                if i < c.len()
                    && (c[i] == 'e' || c[i] == 'E')
                    && i + 1 < c.len()
                    && (c[i + 1].is_ascii_digit() || c[i + 1] == '-' || c[i + 1] == '+')
                {
                    i += 2;
                    while i < c.len() && c[i].is_ascii_digit() {
                        i += 1;
                    }
                }
                let txt: String = c[st..i].iter().filter(|&&x| x != '_').collect();
                out.push(T::Num(txt.parse().map_err(|_| format!("bad number {txt}"))?));
            } else if ch.is_alphabetic() {
                let st = i;
                while i < c.len() && (c[i].is_alphanumeric() || c[i] == '_') {
                    i += 1;
                }
                out.push(T::Id(c[st..i].iter().collect::<String>().to_lowercase()));
            } else if "+-*/%^()×÷".contains(ch) {
                out.push(T::Op(match ch {
                    '×' => '*',
                    '÷' => '/',
                    o => o,
                }));
                i += 1;
            } else {
                return Err(format!("unexpected character {ch:?}"));
            }
        }
        Ok(out)
    }

    struct P<'a> {
        t: &'a [T],
        i: usize,
    }

    impl P<'_> {
        fn peek(&self) -> Option<&T> {
            self.t.get(self.i)
        }
        fn eat(&mut self, op: char) -> bool {
            if self.peek() == Some(&T::Op(op)) {
                self.i += 1;
                true
            } else {
                false
            }
        }
        fn expr(&mut self) -> Result<f64, String> {
            let mut v = self.term()?;
            loop {
                if self.eat('+') {
                    v += self.term()?;
                } else if self.eat('-') {
                    v -= self.term()?;
                } else {
                    return Ok(v);
                }
            }
        }
        fn term(&mut self) -> Result<f64, String> {
            let mut v = self.unary()?;
            loop {
                if self.eat('*') {
                    v *= self.unary()?;
                } else if self.eat('/') {
                    v /= self.unary()?;
                } else if self.eat('%') {
                    v %= self.unary()?;
                } else {
                    return Ok(v);
                }
            }
        }
        fn unary(&mut self) -> Result<f64, String> {
            if self.eat('-') {
                return Ok(-self.unary()?);
            }
            if self.eat('+') {
                return self.unary();
            }
            self.power()
        }
        fn power(&mut self) -> Result<f64, String> {
            let base = self.atom()?;
            if self.eat('^') {
                // Right-associative, and binds tighter than a leading minus on its exponent.
                let e = self.unary()?;
                return Ok(base.powf(e));
            }
            Ok(base)
        }
        fn atom(&mut self) -> Result<f64, String> {
            match self.peek().cloned() {
                Some(T::Num(n)) => {
                    self.i += 1;
                    Ok(n)
                }
                Some(T::Op('(')) => {
                    self.i += 1;
                    let v = self.expr()?;
                    if !self.eat(')') {
                        return Err("missing )".into());
                    }
                    Ok(v)
                }
                Some(T::Id(id)) => {
                    self.i += 1;
                    match id.as_str() {
                        "pi" => return Ok(std::f64::consts::PI),
                        "e" => return Ok(std::f64::consts::E),
                        _ => {}
                    }
                    let arg = self.atom()?;
                    Ok(match id.as_str() {
                        "sqrt" => arg.sqrt(),
                        "sin" => arg.sin(),
                        "cos" => arg.cos(),
                        "tan" => arg.tan(),
                        "ln" => arg.ln(),
                        "log" | "log10" => arg.log10(),
                        "exp" => arg.exp(),
                        "abs" => arg.abs(),
                        "floor" => arg.floor(),
                        "ceil" => arg.ceil(),
                        "round" => arg.round(),
                        other => return Err(format!("unknown function {other}")),
                    })
                }
                other => Err(format!("unexpected {other:?}")),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma_declaration_matches_the_template() {
        let t = ToolSpec {
            name: "get_weather".into(),
            description: "Weather for a city".into(),
            parameters: json!({"type":"object","properties":{"unit":{"type":"string","enum":["c","f"]},"city":{"type":"string","description":"City name"}},"required":["city"]}),
        };
        assert_eq!(
            gemma::declaration(&t),
            "<|tool>declaration:get_weather{description:<|\"|>Weather for a city<|\"|>,parameters:{properties:{city:{description:<|\"|>City name<|\"|>,type:<|\"|>STRING<|\"|>},unit:{enum:[<|\"|>c<|\"|>,<|\"|>f<|\"|>],type:<|\"|>STRING<|\"|>}},required:[<|\"|>city<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|>"
        );
    }

    #[test]
    fn gemma_calls_parse() {
        let raw = "Let me check.<|tool_call>call:get_weather{city:<|\"|>Paris, FR<|\"|>,days:3,metric:true,tags:[<|\"|>a<|\"|>,2]}<tool_call|><|tool_response>";
        let c = gemma::parse_calls(raw);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].name, "get_weather");
        assert_eq!(c[0].arguments["city"], json!("Paris, FR"));
        assert_eq!(c[0].arguments["days"], json!(3));
        assert_eq!(c[0].arguments["metric"], json!(true));
        assert_eq!(c[0].arguments["tags"], json!(["a", 2]));
    }

    #[test]
    fn qwen3_json_calls_parse() {
        let raw = "<think>\n\n</think>\n\n<tool_call>\n{\"name\": \"calculate\", \"arguments\": {\"expression\": \"2 ^ 10\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"get_current_datetime\", \"arguments\": \"{}\"}\n</tool_call>";
        let c = qwen::parse_calls(raw);
        assert_eq!(c.len(), 2);
        assert_eq!(c[0].name, "calculate");
        assert_eq!(c[0].arguments["expression"], json!("2 ^ 10"));
        assert_eq!(c[1].name, "get_current_datetime");
        assert!(c[1].arguments.is_empty());
    }

    #[test]
    fn qwen_calls_parse() {
        let raw = "<tool_call>\n<function=run_javascript>\n<parameter=code>\nconst x = 1;\nreturn x + 1;\n</parameter>\n<parameter=timeout>\n5\n</parameter>\n</function>\n</tool_call>";
        let c = qwen::parse_calls(raw);
        assert_eq!(
            c,
            vec![ToolCall {
                name: "run_javascript".into(),
                arguments: json!({"code": "const x = 1;\nreturn x + 1;", "timeout": 5})
                    .as_object()
                    .unwrap()
                    .clone()
            }]
        );
    }

    #[test]
    fn calculator() {
        assert_eq!(calc::eval("(3 + 4) * 2 ^ 10").unwrap(), 7168.0);
        assert_eq!(calc::eval("-2^2").unwrap(), -4.0);
        assert_eq!(calc::eval("2^-1").unwrap(), 0.5);
        assert!((calc::eval("sqrt(2) * sqrt 2").unwrap() - 2.0).abs() < 1e-12);
        assert_eq!(calc::format(calc::eval("10 % 4 + 1_000").unwrap()), "1002");
        assert!(calc::eval("1 +").is_err());
    }
}
