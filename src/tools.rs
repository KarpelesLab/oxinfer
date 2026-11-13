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

    /// A call as the model writes it, `<|tool_call>call:NAME{...}<tool_call|>` (for a past reply
    /// rendered again).
    pub fn call(c: &ToolCall) -> String {
        format!(
            "<|tool_call>call:{}{}<tool_call|>",
            c.name,
            format_argument(&Value::Object(c.arguments.clone()))
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

    /// A call as the model writes it: `[TOOL_CALLS]NAME[ARGS]{...}`.
    pub fn call(c: &ToolCall) -> String {
        format!("[TOOL_CALLS]{}[ARGS]{}", c.name, Value::Object(c.arguments.clone()))
    }

    /// A tool message: `[TOOL_RESULTS]content[/TOOL_RESULTS]`.
    pub fn response(r: &ToolResult) -> String {
        format!("[TOOL_RESULTS]{}[/TOOL_RESULTS]", r.content)
    }
}

// ---- DeepSeek V4 (DSML) -------------------------------------------------------------------------

pub mod deepseek {
    use super::*;

    const DSML: &str = "｜DSML｜";

    /// The names of the DSML tags: the block of calls, a call, a parameter.
    pub(super) struct Tags {
        pub block: &'static str,
        pub invoke: &'static str,
        pub parameter: &'static str,
    }

    const V4: Tags = Tags {
        block: "tool_calls",
        invoke: "invoke",
        parameter: "parameter",
    };

    /// The "## Tools" section DeepSeek V4's encoder (`render_tools`) adds to the system prompt.
    pub fn tools_block(tools: &[ToolSpec]) -> String {
        tools_block_with(&V4, tools)
    }

    pub(super) fn tools_block_with(tags: &Tags, tools: &[ToolSpec]) -> String {
        let (b, i, p) = (tags.block, tags.invoke, tags.parameter);
        let schemas: Vec<String> = tools
            .iter()
            .map(|t| qwen::py_json(&json!({ "name": t.name, "description": t.description, "parameters": t.parameters })))
            .collect();
        format!(
            "## Tools\n\nYou have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<{DSML}{b}>\" block like the following:\n\n<{DSML}{b}>\n<{DSML}{i} name=\"$TOOL_NAME\">\n<{DSML}{p} name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</{DSML}{p}>\n...\n</{DSML}{i}>\n<{DSML}{i} name=\"$TOOL_NAME2\">\n...\n</{DSML}{i}>\n</{DSML}{b}>\n\nString parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\nIf thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\nOtherwise, output directly after </think> with tool calls or final response.\n\n### Available Tool Schemas\n\n{}\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n",
            schemas.join("\n")
        )
    }

    /// The marker that opens the calls in a reply.
    pub const MARKER: &str = "<｜DSML｜tool_calls>";

    /// Every `<｜DSML｜invoke name="..">` with its `<｜DSML｜parameter name=".." string="true|false">
    /// ..</｜DSML｜parameter>` (string values as is, others as JSON).
    pub fn parse_calls(raw: &str) -> Vec<ToolCall> {
        parse_calls_with(&V4, raw)
    }

    pub(super) fn parse_calls_with(tags: &Tags, raw: &str) -> Vec<ToolCall> {
        let mut calls = Vec::new();
        let marker = format!("<{DSML}{}>", tags.block);
        let Some((_, body)) = raw.split_once(marker.as_str()) else {
            return calls;
        };
        let open_inv = format!("<{DSML}{} name=\"", tags.invoke);
        let open_par = format!("<{DSML}{} name=\"", tags.parameter);
        let close_par = format!("</{DSML}{}>", tags.parameter);
        for inv in body.split(open_inv.as_str()).skip(1) {
            let Some((name, mut rest)) = inv.split_once("\">") else { continue };
            let mut arguments = Map::new();
            while let Some(i) = rest.find(open_par.as_str()) {
                rest = &rest[i + open_par.len()..];
                let Some((pname, r)) = rest.split_once('"') else { break };
                let Some(r) = r.strip_prefix(" string=\"") else { break };
                let Some((is_str, r)) = r.split_once("\">") else { break };
                let Some((value, r)) = r.split_once(close_par.as_str()) else {
                    break;
                };
                let v = if is_str == "true" {
                    Value::String(value.to_string())
                } else {
                    serde_json::from_str(value).unwrap_or_else(|_| Value::String(value.to_string()))
                };
                arguments.insert(pname.to_string(), v);
                rest = r;
            }
            calls.push(ToolCall {
                name: name.to_string(),
                arguments,
            });
        }
        calls
    }

    /// Calls as the model writes them: one `<｜DSML｜tool_calls>` block (strings as is, other values
    /// as JSON).
    pub fn calls(calls: &[ToolCall]) -> String {
        calls_with(&V4, calls)
    }

    pub(super) fn calls_with(tags: &Tags, calls: &[ToolCall]) -> String {
        let (b, i, p) = (tags.block, tags.invoke, tags.parameter);
        let mut s = format!("\n\n<{DSML}{b}>\n");
        for c in calls {
            s.push_str(&format!("<{DSML}{i} name=\"{}\">\n", c.name));
            for (k, v) in &c.arguments {
                let (is_str, text) = match v {
                    Value::String(t) => ("true", t.clone()),
                    other => ("false", other.to_string()),
                };
                s.push_str(&format!("<{DSML}{p} name=\"{k}\" string=\"{is_str}\">{text}</{DSML}{p}>\n"));
            }
            s.push_str(&format!("</{DSML}{i}>\n"));
        }
        s.push_str(&format!("</{DSML}{b}>"));
        s
    }

    /// A tool result block (the results of one round are joined with blank lines in a user turn).
    pub fn response(r: &ToolResult) -> String {
        format!("<tool_result>{}</tool_result>", r.content)
    }
}

/// DeepSeek V4.1 (its `encoding.py`): V4's DSML with tag names that start with a space
/// (`<｜DSML｜ calls>`, `<｜DSML｜ invoke name="..">`, `<｜DSML｜ parameter ..>`).
pub mod deepseek41 {
    use super::deepseek::{Tags, calls_with, parse_calls_with, tools_block_with};
    use super::*;

    const V41: Tags = Tags {
        block: " calls",
        invoke: " invoke",
        parameter: " parameter",
    };

    pub fn tools_block(tools: &[ToolSpec]) -> String {
        tools_block_with(&V41, tools)
    }

    /// The marker that opens the calls in a reply.
    pub const MARKER: &str = "<｜DSML｜ calls>";

    pub fn parse_calls(raw: &str) -> Vec<ToolCall> {
        parse_calls_with(&V41, raw)
    }

    pub fn calls(calls: &[ToolCall]) -> String {
        calls_with(&V41, calls)
    }
}

// ---- MiniMax M2 / M3 ----------------------------------------------------------------------------

pub mod minimax {
    use super::*;

    /// The marker that opens the calls in a reply.
    pub const MARKER: &str = "<minimax:tool_call>";
    const END: &str = "</minimax:tool_call>";

    /// The "# Tools" section the MiniMax template appends to the system turn: one
    /// `<tool>{function JSON}</tool>` per tool, then the call syntax.
    pub fn tools_block(tools: &[ToolSpec]) -> String {
        let mut s = String::from(
            "\n\n# Tools\nYou may call one or more tools to assist with the user query.\nHere are the tools available in JSONSchema format:\n\n<tools>\n",
        );
        for t in tools {
            let f = json!({ "name": t.name, "description": t.description, "parameters": t.parameters });
            s.push_str(&format!("<tool>{}</tool>\n", qwen::py_json(&f)));
        }
        s.push_str("</tools>\n\nWhen making tool calls, use XML format to invoke tools and pass parameters:\n\n");
        s.push_str(MARKER);
        s.push_str("\n<invoke name=\"tool-name-1\">\n<parameter name=\"param-key-1\">param-value-1</parameter>\n<parameter name=\"param-key-2\">param-value-2</parameter>\n...\n</invoke>\n");
        s.push_str(END);
        s
    }

    /// Every `<invoke name="..">` with its `<parameter name="..">..</parameter>` inside
    /// `<minimax:tool_call>` blocks. Values are JSON when they parse as JSON (other than a bare
    /// string), strings otherwise (the template writes strings as is, other values as JSON).
    pub fn parse_calls(raw: &str) -> Vec<ToolCall> {
        let mut calls = Vec::new();
        for block in raw.split(MARKER).skip(1) {
            let block = block.split(END).next().unwrap_or(block);
            for inv in block.split("<invoke name=\"").skip(1) {
                let Some((name, mut rest)) = inv.split_once("\">") else { continue };
                rest = rest.split("</invoke>").next().unwrap_or(rest);
                let mut arguments = Map::new();
                while let Some(i) = rest.find("<parameter name=\"") {
                    rest = &rest[i + "<parameter name=\"".len()..];
                    let Some((key, r)) = rest.split_once("\">") else { break };
                    let (value, r) = r.split_once("</parameter>").unwrap_or((r, ""));
                    let v = value.trim_matches('\n');
                    let parsed = serde_json::from_str::<Value>(v.trim()).ok().filter(|j| !j.is_string());
                    arguments.insert(key.to_string(), parsed.unwrap_or_else(|| Value::String(v.to_string())));
                    rest = r;
                }
                if !name.is_empty() {
                    calls.push(ToolCall {
                        name: name.to_string(),
                        arguments,
                    });
                }
            }
        }
        calls
    }

    /// Calls as the template renders them after a reply's text.
    pub fn calls(calls: &[ToolCall]) -> String {
        let mut s = format!("\n{MARKER}\n");
        for c in calls {
            s.push_str(&format!("<invoke name=\"{}\">\n", c.name));
            for (k, v) in &c.arguments {
                let text = match v {
                    Value::String(t) => t.clone(),
                    other => qwen::py_json(other),
                };
                s.push_str(&format!("<parameter name=\"{k}\">{text}</parameter>\n"));
            }
            s.push_str("</invoke>\n");
        }
        s.push_str(END);
        s
    }
}

/// MiniMax M3 / M3.1: every tag of a call is preceded by the `]<]minimax[>[` token; arguments are
/// elements named after them, objects and arrays expanded into nested elements (`<item>` per
/// array entry), scalars as text (booleans as JSON).
pub mod minimax3 {
    use super::*;

    pub const NS: &str = "]<]minimax[>[";

    /// The tools section of the developer turn.
    pub fn tools_block(tools: &[ToolSpec]) -> String {
        let mut s = String::from(
            "\n\n# Tools\nYou may call one or more tools to assist with the user query.\nHere are the tools available in JSONSchema format:\n\n<tools>\n",
        );
        for t in tools {
            let f = json!({ "name": t.name, "description": t.description, "parameters": t.parameters });
            s.push_str(&format!("<tool>{}</tool>\n", qwen::py_json(&f)));
        }
        s.push_str("</tools>\n\n");
        s.push_str(&format!(
            "To call tools, wrap all invocations in a single {NS}<tool_call>{NS}</tool_call> block. Parameter values containing nested objects or arrays are recursively expanded into XML elements. Example:\n\n{NS}<tool_call>\n{NS}<invoke name=\"tool-name-1\">{NS}<param-1>value-1{NS}</param-1>{NS}<param-2>{NS}<item>{NS}<key-a>val-a{NS}</key-a>{NS}<key-b>val-b{NS}</key-b>{NS}</item>{NS}</param-2>{NS}</invoke>\n{NS}<invoke name=\"tool-name-2\">{NS}<param-1>value-1{NS}</param-1>{NS}</invoke>\n{NS}</tool_call>"
        ));
        s
    }

    /// A value as the template's `to_xml` writes it.
    fn to_xml(v: &Value, out: &mut String) {
        match v {
            Value::Object(o) => {
                for (k, v) in o.iter().filter(|(_, v)| !v.is_null()) {
                    out.push_str(&format!("{NS}<{k}>"));
                    to_xml(v, out);
                    out.push_str(&format!("{NS}</{k}>"));
                }
            }
            Value::Array(a) => {
                for v in a {
                    out.push_str(&format!("{NS}<item>"));
                    to_xml(v, out);
                    out.push_str(&format!("{NS}</item>"));
                }
            }
            Value::Null => {}
            Value::String(s) => out.push_str(s),
            other => out.push_str(&other.to_string()),
        }
    }

    /// Calls as the template renders them after a reply's text.
    pub fn calls(calls: &[ToolCall]) -> String {
        let mut s = format!("{NS}<tool_call>\n");
        for c in calls {
            s.push_str(&format!("{NS}<invoke name=\"{}\">", c.name));
            for (k, v) in c.arguments.iter().filter(|(_, v)| !v.is_null()) {
                s.push_str(&format!("{NS}<{k}>"));
                to_xml(v, &mut s);
                s.push_str(&format!("{NS}</{k}>"));
            }
            s.push_str(&format!("{NS}</invoke>\n"));
        }
        s.push_str(&format!("{NS}</tool_call>"));
        s
    }

    /// The children of an element body (after the `NS` markers were removed): `(name, body)`
    /// pairs, or None when the body is text.
    fn children(mut body: &str) -> Option<Vec<(&str, &str)>> {
        let mut out = Vec::new();
        loop {
            let b = body.trim_start_matches(['\n', ' ']);
            if b.is_empty() {
                return (!out.is_empty()).then_some(out);
            }
            let rest = b.strip_prefix('<')?;
            let (name, rest) = rest.split_once('>')?;
            if name.is_empty() || name.starts_with('/') || name.contains(' ') {
                return None;
            }
            // The matching close tag, skipping nested elements of the same name.
            let (open, close) = (format!("<{name}>"), format!("</{name}>"));
            let (mut depth, mut i) = (1usize, 0usize);
            loop {
                let c = rest[i..].find(&close)? + i;
                depth += rest[i..c].matches(&open).count();
                depth -= 1;
                if depth == 0 {
                    out.push((name, &rest[..c]));
                    body = &rest[c + close.len()..];
                    break;
                }
                i = c + close.len();
            }
        }
    }

    fn value(body: &str) -> Value {
        match children(body) {
            Some(ch) if ch.iter().all(|(n, _)| *n == "item") => Value::Array(ch.into_iter().map(|(_, b)| value(b)).collect()),
            Some(ch) => Value::Object(ch.into_iter().map(|(n, b)| (n.to_string(), value(b))).collect()),
            None => serde_json::from_str::<Value>(body.trim())
                .ok()
                .filter(|j| !j.is_string() && !j.is_object() && !j.is_array())
                .unwrap_or_else(|| Value::String(body.to_string())),
        }
    }

    /// Every `<invoke name="..">` in `<tool_call>` blocks.
    pub fn parse_calls(raw: &str) -> Vec<ToolCall> {
        let text = raw.replace(NS, "");
        let mut calls = Vec::new();
        for block in text.split("<tool_call>").skip(1) {
            let block = block.split("</tool_call>").next().unwrap_or(block);
            for inv in block.split("<invoke name=\"").skip(1) {
                let Some((name, rest)) = inv.split_once("\">") else { continue };
                let body = rest.split("</invoke>").next().unwrap_or(rest);
                let arguments = match value(body) {
                    Value::Object(o) => o,
                    _ => Map::new(),
                };
                if !name.is_empty() {
                    calls.push(ToolCall {
                        name: name.to_string(),
                        arguments,
                    });
                }
            }
        }
        calls
    }
}

/// MiniMax-M1: tools as JSON lines in a `tool_setting` system turn, calls as JSON lines in
/// `<tool_calls>`; MiniMax-Text-01: one `function_setting` turn per tool, calls as
/// `<function_call>` followed by a TypeScript block `functions.NAME({json})`.
pub mod minimax01 {
    use super::*;

    /// The `tool_setting` turn of MiniMax-M1's template (whose `''name''` renders unquoted).
    pub fn m1_tools_turn(tools: &[ToolSpec]) -> String {
        let mut s = String::from("<beginning_of_sentence>system tool_setting=tools\nYou are provided with these tools:\n<tools>\n");
        for t in tools {
            let v = json!({ "type": "function", "function": { "name": t.name, "description": t.description, "parameters": t.parameters } });
            s.push_str(&qwen::py_json(&v));
            s.push('\n');
        }
        s.push_str("</tools>\n\nIf you need to call tools, please respond with <tool_calls></tool_calls> XML tags, and provide tool-name and json-object of arguments, following the format below:\n<tool_calls>\n{name: <tool-name-1>, arguments: <args-json-object-1>}\n...\n</tool_calls><end_of_sentence>\n");
        s
    }

    /// The `function_setting` turns of MiniMax-Text-01's template.
    pub fn text01_tools_turns(tools: &[ToolSpec]) -> String {
        tools
            .iter()
            .map(|t| {
                let v =
                    json!({ "type": "function", "function": { "name": t.name, "description": t.description, "parameters": t.parameters } });
                format!(
                    "<beginning_of_sentence>system function_setting=functions\n{}<end_of_sentence>\n",
                    qwen::py_json(&v)
                )
            })
            .collect()
    }

    fn call_of(v: &Value) -> Option<ToolCall> {
        let o = v.as_object()?;
        let name = o.get("name")?.as_str()?.to_string();
        let arguments = match o.get("arguments") {
            Some(Value::Object(a)) => a.clone(),
            Some(Value::String(a)) => serde_json::from_str(a).unwrap_or_default(),
            _ => Map::new(),
        };
        Some(ToolCall { name, arguments })
    }

    /// M1: one JSON object per line inside `<tool_calls>`.
    pub fn parse_m1(raw: &str) -> Vec<ToolCall> {
        raw.split("<tool_calls>")
            .skip(1)
            .flat_map(|b| {
                b.split("</tool_calls>")
                    .next()
                    .unwrap_or(b)
                    .lines()
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
            .filter_map(|l| serde_json::from_str::<Value>(l.trim()).ok().as_ref().and_then(call_of))
            .collect()
    }

    pub fn m1_calls(calls: &[ToolCall]) -> String {
        let lines: Vec<String> = calls
            .iter()
            .map(|c| qwen::py_json(&json!({ "name": c.name, "arguments": Value::Object(c.arguments.clone()) })))
            .collect();
        format!("<tool_calls>\n{}\n</tool_calls>", lines.join("\n"))
    }

    /// Text-01: `functions.NAME({json})` lines in TypeScript blocks after `<function_call>`.
    pub fn parse_text01(raw: &str) -> Vec<ToolCall> {
        let mut calls = Vec::new();
        for part in raw.split("<function_call>").skip(1) {
            for line in part.lines() {
                let Some(rest) = line.trim().strip_prefix("functions.") else {
                    continue;
                };
                let Some((name, args)) = rest.split_once('(') else { continue };
                let args = args.trim_end().strip_suffix(')').unwrap_or(args);
                let arguments = serde_json::from_str::<Value>(args)
                    .ok()
                    .and_then(|v| v.as_object().cloned())
                    .unwrap_or_default();
                calls.push(ToolCall {
                    name: name.to_string(),
                    arguments,
                });
            }
        }
        calls
    }

    pub fn text01_calls(calls: &[ToolCall]) -> String {
        let lines: Vec<String> = calls
            .iter()
            .map(|c| format!("functions.{}({})", c.name, qwen::py_json(&Value::Object(c.arguments.clone()))))
            .collect();
        format!("<function_call>```typescript\n{}\n```", lines.join("\n"))
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

    /// A call as Qwen 3.5 / 3.8 write it (`<tool_call>` with `<function=...>` and one
    /// `<parameter=...>` per argument: strings as is, other values as JSON), or with `json` as
    /// Qwen3 writes it (`{"name": ..., "arguments": {...}}`).
    pub fn call(c: &ToolCall, json: bool) -> String {
        if json {
            let v = json!({ "name": c.name, "arguments": Value::Object(c.arguments.clone()) });
            return format!("<tool_call>\n{}\n</tool_call>", py_json(&v));
        }
        let mut s = format!("<tool_call>\n<function={}>\n", c.name);
        for (k, v) in &c.arguments {
            let text = match v {
                Value::String(t) => t.clone(),
                other => other.to_string(),
            };
            s.push_str(&format!("<parameter={k}>\n{text}\n</parameter>\n"));
        }
        s.push_str("</function>\n</tool_call>");
        s
    }

    /// Every `<tool_call><function=NAME><parameter=K>V</parameter>...</function></tool_call>`, or
    /// (Qwen3) `<tool_call>{"name": NAME, "arguments": {...}}</tool_call>`.
    /// Parameter values are JSON when they parse as JSON, strings otherwise (as the template writes).
    pub fn parse_calls(raw: &str) -> Vec<ToolCall> {
        parse_calls_for(raw, &[])
    }

    /// [`parse_calls`] with the declared tools: a parameter whose schema says `"type": "string"`
    /// stays the string the model wrote, also when it reads as JSON (`42`, `true`, `null`).
    pub fn parse_calls_for(raw: &str, tools: &[ToolSpec]) -> Vec<ToolCall> {
        let declared_string = |tool: &str, key: &str| {
            let t = tools.iter().find(|t| t.name == tool);
            t.and_then(|t| t.parameters.get("properties")?.get(key)?.get("type")?.as_str()) == Some("string")
        };
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
                        .filter(|j| !j.is_string() && !declared_string(&name, &key))
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

// ---- GLM-5.3 ------------------------------------------------------------------------------------

/// GLM-5.3 / GLM-5.3-Flash: JSON declarations inside `<tools>` in a system turn of their own;
/// calls as `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>` (each
/// tag a token of the vocabulary), strings bare and other values as JSON.
pub mod glm {
    use super::*;

    /// The marker that opens a call in a reply.
    pub const MARKER: &str = "<tool_call>";
    const END: &str = "</tool_call>";

    /// The tools turn the template writes after the reasoning-effort line.
    pub fn tools_block(tools: &[ToolSpec]) -> String {
        let mut s = String::from(
            "<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n",
        );
        for t in tools {
            s.push_str(&format!(
                "{{\"name\": {}, \"description\": {}, \"parameters\": {}}}\n",
                qwen::py_json(&Value::String(t.name.clone())),
                qwen::py_json(&Value::String(t.description.clone())),
                qwen::py_json(&t.parameters)
            ));
        }
        s.push_str("</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>");
        s
    }

    /// Every `<tool_call>` of a reply. Values are JSON when they parse as JSON (other than a bare
    /// string), strings otherwise; a parameter declared `"type": "string"` in `tools` stays the
    /// string the model wrote.
    pub fn parse_calls_for(raw: &str, tools: &[ToolSpec]) -> Vec<ToolCall> {
        let declared_string = |tool: &str, key: &str| {
            let t = tools.iter().find(|t| t.name == tool);
            t.and_then(|t| t.parameters.get("properties")?.get(key)?.get("type")?.as_str()) == Some("string")
        };
        let mut calls = Vec::new();
        for block in raw.split(MARKER).skip(1) {
            let block = block.split(END).next().unwrap_or(block);
            let (name, mut rest) = match block.find("<arg_key>") {
                Some(i) => (block[..i].trim(), &block[i..]),
                None => (block.trim(), ""),
            };
            let mut arguments = Map::new();
            while let Some(r) = rest.strip_prefix("<arg_key>") {
                let Some((key, r)) = r.split_once("</arg_key>") else { break };
                let Some(r) = r.trim_start().strip_prefix("<arg_value>") else {
                    break;
                };
                let (value, r) = r.split_once("</arg_value>").unwrap_or((r, ""));
                let parsed = serde_json::from_str::<Value>(value.trim())
                    .ok()
                    .filter(|j| !j.is_string() && !declared_string(name, key.trim()));
                arguments.insert(key.trim().to_string(), parsed.unwrap_or_else(|| Value::String(value.to_string())));
                rest = r.trim_start();
            }
            if !name.is_empty() {
                calls.push(ToolCall {
                    name: name.to_string(),
                    arguments,
                });
            }
        }
        calls
    }

    pub fn parse_calls(raw: &str) -> Vec<ToolCall> {
        parse_calls_for(raw, &[])
    }

    /// Calls as the template renders them after a reply's text.
    pub fn calls(calls: &[ToolCall]) -> String {
        let mut s = String::new();
        for c in calls {
            s.push_str(MARKER);
            s.push_str(&c.name);
            for (k, v) in &c.arguments {
                let text = match v {
                    Value::String(t) => t.clone(),
                    other => qwen::py_json(other),
                };
                s.push_str(&format!("<arg_key>{k}</arg_key><arg_value>{text}</arg_value>"));
            }
            s.push_str(END);
        }
        s
    }

    pub fn response(r: &ToolResult) -> String {
        format!("<tool_response>{}</tool_response>", r.content)
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

    /// DeepSeek V4's DSML calls, as its encoder writes them (two calls, string and JSON values).
    #[test]
    fn qwen_string_parameters_stay_strings() {
        let raw =
            "<tool_call>\n<function=f>\n<parameter=zip>\n75001\n</parameter>\n<parameter=n>\n3\n</parameter>\n</function>\n</tool_call>";
        let tool = ToolSpec {
            name: "f".into(),
            description: String::new(),
            parameters: json!({"type": "object", "properties": {"zip": {"type": "string"}, "n": {"type": "integer"}}}),
        };
        let typed = qwen::parse_calls_for(raw, &[tool]);
        assert_eq!(typed[0].arguments["zip"], json!("75001"));
        assert_eq!(typed[0].arguments["n"], json!(3));
        assert_eq!(qwen::parse_calls(raw)[0].arguments["zip"], json!(75001));
    }

    #[test]
    fn deepseek_calls_parse() {
        let raw = "Let me compute.\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"calculate\">\n<｜DSML｜parameter name=\"expression\" string=\"true\">2+2</｜DSML｜parameter>\n</｜DSML｜invoke>\n<｜DSML｜invoke name=\"search\">\n<｜DSML｜parameter name=\"q\" string=\"true\">a \"b\"</｜DSML｜parameter>\n<｜DSML｜parameter name=\"n\" string=\"false\">[1, 2]</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls><｜end▁of▁sentence｜>";
        let c = deepseek::parse_calls(raw);
        let obj = |v: Value| v.as_object().unwrap().clone();
        assert_eq!(
            c,
            vec![
                ToolCall {
                    name: "calculate".into(),
                    arguments: obj(json!({"expression": "2+2"}))
                },
                ToolCall {
                    name: "search".into(),
                    arguments: obj(json!({"q": "a \"b\"", "n": [1, 2]}))
                }
            ]
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
