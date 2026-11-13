//! LongCat-2.5's chat format (its `chat_template.jinja`) and tool calls (the `longcat_xml`
//! parser of Meituan's SGLang-FluentLLM, which the model card's deployment uses).
//!
//! A conversation has no opening token. Tools are declared first, in a
//! `<longcat_tool_declare>` block; then `<longcat_system>`, `<longcat_user>`,
//! `<longcat_assistant>` and `<longcat_observation>` turns follow each other with nothing
//! between them. The turn before a reply ends with ` /think_on ` or ` /think_off `, and the reply
//! opens with `<longcat_think>`: left open when thinking, closed at once (`\n\n</longcat_think>\n`)
//! otherwise. A reply ends with `</longcat_s>`, the end-of-sequence token.
//!
//! A call is `<longcat_tool_call>NAME`, then `<longcat_arg_key>K</longcat_arg_key>` and
//! `<longcat_arg_value>V</longcat_arg_value>` on a line each, then `</longcat_tool_call>`:
//! strings bare, other values as JSON. Results come in one `<longcat_observation>` turn, each as
//! `<longcat_tool_response>NAME\nCONTENT</longcat_tool_response>`.
//!
//! Not rendered: the template's `thinking_budget` line, `files`, and the `code_interpreter`
//! tool type (every tool is a function).

use crate::chat::ChatFormat;
use crate::tools::qwen::py_json;
use crate::tools::{ToolCall, ToolResult, ToolSpec};
use serde_json::{Map, Value};

/// The marker that opens a call in a reply.
pub const MARKER: &str = "<longcat_tool_call>";
const END: &str = "</longcat_tool_call>";
/// The token a reply ends on.
pub const EOS: &str = "</longcat_s>";

/// The declaration block the template writes at the start of the conversation.
pub fn tools_block(tools: &[ToolSpec]) -> String {
    let mut s = String::from("<longcat_tool_declare>\n# Tools\nYou have access to the following tools:\n\n## Tool namespace: function\n\n");
    for t in tools {
        s.push_str(&format!(
            "### Tool name: {}\nDescription: {}\n\nInputSchema: {}\n\n",
            t.name,
            t.description,
            py_json(&t.parameters)
        ));
    }
    s.push_str(concat!(
        "**Note**: For each function call, output the function name and arguments within the following XML format:\n",
        "<longcat_tool_call>{function-name}\n",
        "<longcat_arg_key>{arg-key-1}</longcat_arg_key>\n<longcat_arg_value>{arg-value-1}</longcat_arg_value>\n",
        "<longcat_arg_key>{arg-key-2}</longcat_arg_key>\n<longcat_arg_value>{arg-value-2}</longcat_arg_value>\n",
        "...\n</longcat_tool_call>\n</longcat_tool_declare>"
    ));
    s
}

/// Every `<longcat_tool_call>` of a reply, as SGLang-FluentLLM's `LongCatXMLDetector` reads them:
/// the name is the rest of the first line; keys and values are trimmed; a value is JSON when it
/// parses as JSON, the text otherwise, and the text always for a parameter declared
/// `"type": "string"` in `tools`.
pub fn parse_calls_for(raw: &str, tools: &[ToolSpec]) -> Vec<ToolCall> {
    let declared_string = |tool: &str, key: &str| {
        let t = tools.iter().find(|t| t.name == tool);
        t.and_then(|t| t.parameters.get("properties")?.get(key)?.get("type")?.as_str()) == Some("string")
    };
    let mut calls = Vec::new();
    for block in raw.split(MARKER).skip(1) {
        let block = block.split(END).next().unwrap_or(block);
        let (name, mut rest) = match block.split_once('\n') {
            Some((n, r)) => (n.trim(), r),
            None => (block.trim(), ""),
        };
        let mut arguments = Map::new();
        while let Some(i) = rest.find("<longcat_arg_key>") {
            let r = &rest[i + "<longcat_arg_key>".len()..];
            // (A missing `</longcat_arg_key>`: the key runs up to the value, as the detector's
            // fallback patterns read it.)
            let Some(v) = r.find("<longcat_arg_value>") else { break };
            let key = r[..v].split("</longcat_arg_key>").next().unwrap_or("").trim();
            let r = &r[v + "<longcat_arg_value>".len()..];
            let (value, r) = r.split_once("</longcat_arg_value>").unwrap_or((r, ""));
            let value = value.trim();
            let parsed = if declared_string(name, key) {
                None
            } else {
                serde_json::from_str::<Value>(value).ok()
            };
            arguments.insert(key.to_string(), parsed.unwrap_or_else(|| Value::String(value.to_string())));
            rest = r;
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

/// Calls as the template renders them after a reply's text.
pub fn calls(calls: &[ToolCall]) -> String {
    let mut s = String::new();
    for c in calls {
        s.push_str(MARKER);
        s.push_str(&c.name);
        s.push('\n');
        for (k, v) in &c.arguments {
            let text = match v {
                Value::String(t) => t.clone(),
                other => py_json(other),
            };
            s.push_str(&format!(
                "<longcat_arg_key>{k}</longcat_arg_key>\n<longcat_arg_value>{text}</longcat_arg_value>\n"
            ));
        }
        s.push_str(END);
        s.push('\n');
    }
    s
}

pub fn response(r: &ToolResult) -> String {
    format!("<longcat_tool_response>{}\n{}</longcat_tool_response>", r.name, r.content)
}

/// The chat format of a LongCat checkpoint (see the module docs), as its template has it.
pub struct LongcatChat {
    /// The template marks the turn before a reply with ` /think_on ` / ` /think_off ` and opens
    /// the reply with a think block (LongCat-2.5, LongCat-2.0). The LongCat-Flash-Lite templates
    /// do so only when asked, and then for the last turn alone, which a conversation that is
    /// extended turn by turn cannot follow: for them a reply opens with `<longcat_assistant>`
    /// alone (their default; they are not thinking models) and `think` is ignored.
    thinking: bool,
    /// Tool results are a `<longcat_observation>` turn with the tools' names (otherwise, as
    /// LongCat-Flash-Lite, a `<longcat_user>` turn of bare `<longcat_tool_response>` blocks).
    observation: bool,
}

/// LongCat-2.5 and LongCat-2.0.
pub static LONGCAT: LongcatChat = LongcatChat {
    thinking: true,
    observation: true,
};
/// LongCat-Flash-Lite-Sparse.
pub static LONGCAT_LITE_SPARSE: LongcatChat = LongcatChat {
    thinking: false,
    observation: true,
};
/// LongCat-Flash-Lite.
pub static LONGCAT_LITE: LongcatChat = LongcatChat {
    thinking: false,
    observation: false,
};

impl LongcatChat {
    /// The format of the checkpoint in `dir`, from its chat template (`chat_template.jinja`, or
    /// the one in `tokenizer_config.json`; LongCat-2.5's when it has none).
    pub fn for_model(dir: &std::path::Path) -> anyhow::Result<&'static LongcatChat> {
        let template = crate::fsx::read(&dir.join("chat_template.jinja"))
            .ok()
            .map(|b| String::from_utf8_lossy(&b).into_owned())
            .or_else(|| {
                let cfg: Value = serde_json::from_slice(&crate::fsx::read(&dir.join("tokenizer_config.json")).ok()?).ok()?;
                cfg["chat_template"].as_str().map(str::to_string)
            });
        Self::for_template(template.as_deref())
    }

    /// The format matching a chat template's source (None: LongCat-2.5's).
    pub fn for_template(template: Option<&str>) -> anyhow::Result<&'static LongcatChat> {
        let Some(t) = template else { return Ok(&LONGCAT) };
        if !t.contains("<longcat_user>") {
            anyhow::bail!(
                "this checkpoint's chat template is not one of LongCat-2.x's or LongCat-Flash-Lite's (LongCat-Flash-Chat's `[Round n] USER:` format is not supported)"
            );
        }
        let thinking = t.contains("enable_thinking | default(true)") || t.contains("enable_thinking | default(false)");
        Ok(match (thinking, t.contains("<longcat_observation>")) {
            (true, _) => &LONGCAT,
            (false, true) => &LONGCAT_LITE_SPARSE,
            (false, false) => &LONGCAT_LITE,
        })
    }

    /// What ends the turn before a reply, and opens the reply.
    fn assistant(&self, think: bool) -> &'static str {
        match (self.thinking, think) {
            (false, _) => "<longcat_assistant>",
            (true, true) => " /think_on <longcat_assistant><longcat_think>",
            (true, false) => " /think_off <longcat_assistant><longcat_think>\n\n</longcat_think>\n",
        }
    }
}

impl ChatFormat for LongcatChat {
    fn conversation_start(&self, system: Option<&str>, think: bool) -> String {
        self.conversation_start_tools(system, think, &[])
    }
    fn user_turn(&self, content: &str, think: bool) -> String {
        format!("<longcat_user>{content}{}", self.assistant(think))
    }
    fn end_of_model_turn(&self) -> String {
        EOS.to_string()
    }
    fn end_of_turn_token(&self) -> &'static str {
        EOS
    }
    fn think_tokens(&self) -> (&'static str, &'static str) {
        ("<longcat_think>", "</longcat_think>")
    }
    fn starts_in_thinking(&self, think: bool) -> bool {
        self.thinking && think
    }
    fn conversation_start_tools(&self, system: Option<&str>, _think: bool, tools: &[ToolSpec]) -> String {
        let mut s = String::new();
        if !tools.is_empty() {
            s.push_str(&tools_block(tools));
        }
        if let Some(sys) = system.filter(|s| !s.is_empty()) {
            s.push_str("<longcat_system>");
            s.push_str(sys);
        }
        s
    }
    fn tool_call_marker(&self) -> &'static str {
        MARKER
    }
    fn parse_tool_calls(&self, raw: &str) -> Vec<ToolCall> {
        parse_calls_for(raw, &[])
    }
    fn parse_tool_calls_for(&self, raw: &str, tools: &[ToolSpec]) -> Vec<ToolCall> {
        parse_calls_for(raw, tools)
    }
    fn tool_results(&self, results: &[ToolResult], think: bool) -> String {
        // The reply ended (`</longcat_s>` was sampled, not fed); the results are one turn.
        let mut s = format!(
            "{EOS}{}",
            if self.observation {
                "<longcat_observation>"
            } else {
                "<longcat_user>"
            }
        );
        for r in results {
            if self.observation {
                s.push_str(&response(r));
            } else {
                s.push_str(&format!("<longcat_tool_response>{}</longcat_tool_response>", r.content));
            }
        }
        s.push_str(self.assistant(think));
        s
    }
    fn assistant_text(&self, content: &str, calls: &[ToolCall]) -> String {
        format!("{content}{}", self::calls(calls))
    }
    fn assistant_text_after(&self, content: &str, calls: &[ToolCall], think: bool) -> String {
        // After a generation prompt that left the think block open: it closes empty, as the
        // template renders a past reply without its reasoning.
        let close = if self.starts_in_thinking(think) {
            "\n\n</longcat_think>\n"
        } else {
            ""
        };
        format!("{close}{}", self.assistant_text(content, calls))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The renderings of LongCat-2.5's chat template (transformers' jinja environment).
    #[test]
    fn matches_template() {
        let f = &LONGCAT;
        assert_eq!(
            f.first_turn(None, false, &[], "What is the capital of France?"),
            "<longcat_user>What is the capital of France? /think_off <longcat_assistant><longcat_think>\n\n</longcat_think>\n"
        );
        assert_eq!(
            f.first_turn(Some("Be brief."), true, &[], "What is the capital of France?"),
            "<longcat_system>Be brief.<longcat_user>What is the capital of France? /think_on <longcat_assistant><longcat_think>"
        );
        let tools = [
            ToolSpec {
                name: "get_time".into(),
                description: "Current time in a timezone.".into(),
                parameters: serde_json::json!({"type": "object", "properties": {"tz": {"type": "string", "description": "IANA name"}, "n": {"type": "integer"}}, "required": ["tz"]}),
            },
            ToolSpec {
                name: "calc".into(),
                description: "Evaluate \"é\".".into(),
                parameters: serde_json::json!({"type": "object", "properties": {"expr": {"type": "string"}}}),
            },
        ];
        let declare = concat!(
            "<longcat_tool_declare>\n# Tools\nYou have access to the following tools:\n\n## Tool namespace: function\n\n",
            "### Tool name: get_time\nDescription: Current time in a timezone.\n\n",
            "InputSchema: {\"type\": \"object\", \"properties\": {\"tz\": {\"type\": \"string\", \"description\": \"IANA name\"}, \"n\": {\"type\": \"integer\"}}, \"required\": [\"tz\"]}\n\n",
            "### Tool name: calc\nDescription: Evaluate \"é\".\n\n",
            "InputSchema: {\"type\": \"object\", \"properties\": {\"expr\": {\"type\": \"string\"}}}\n\n",
            "**Note**: For each function call, output the function name and arguments within the following XML format:\n",
            "<longcat_tool_call>{function-name}\n<longcat_arg_key>{arg-key-1}</longcat_arg_key>\n<longcat_arg_value>{arg-value-1}</longcat_arg_value>\n",
            "<longcat_arg_key>{arg-key-2}</longcat_arg_key>\n<longcat_arg_value>{arg-value-2}</longcat_arg_value>\n...\n</longcat_tool_call>\n</longcat_tool_declare>"
        );
        assert_eq!(
            f.first_turn(Some("Be brief."), true, &tools, "Time in Tokyo?"),
            format!("{declare}<longcat_system>Be brief.<longcat_user>Time in Tokyo? /think_on <longcat_assistant><longcat_think>")
        );
        assert_eq!(
            f.first_turn(None, false, &tools, "Time in Tokyo?"),
            format!("{declare}<longcat_user>Time in Tokyo? /think_off <longcat_assistant><longcat_think>\n\n</longcat_think>\n")
        );
        // A reply with reasoning, text and two calls as the template renders it after the
        // generation prompt, its parse, the results and the next turns.
        let reply = concat!(
            "Let me check.<longcat_tool_call>get_time\n<longcat_arg_key>tz</longcat_arg_key>\n<longcat_arg_value>Asia/Tokyo</longcat_arg_value>\n",
            "<longcat_arg_key>n</longcat_arg_key>\n<longcat_arg_value>3</longcat_arg_value>\n</longcat_tool_call>\n",
            "<longcat_tool_call>calc\n<longcat_arg_key>expr</longcat_arg_key>\n<longcat_arg_value>1+1</longcat_arg_value>\n",
            "<longcat_arg_key>Opt</longcat_arg_key>\n<longcat_arg_value>{\"z\": 1, \"a\": [1.5, true, null]}</longcat_arg_value>\n</longcat_tool_call>\n"
        );
        let calls = f.parse_tool_calls_for(&format!("\nneed tool\n</longcat_think>\n{reply}"), &tools);
        assert_eq!(calls.len(), 2);
        assert_eq!((calls[0].name.as_str(), calls[1].name.as_str()), ("get_time", "calc"));
        assert_eq!(calls[0].arguments["tz"], "Asia/Tokyo");
        assert_eq!(calls[0].arguments["n"], 3);
        assert_eq!(calls[1].arguments["expr"], "1+1");
        assert_eq!(calls[1].arguments["Opt"]["a"][0], 1.5);
        assert_eq!(f.assistant_text("Let me check.", &calls), reply);
        let results = [
            ToolResult {
                name: "get_time".into(),
                content: "12:00".into(),
            },
            ToolResult {
                name: "calc".into(),
                content: "2".into(),
            },
        ];
        let observed = "</longcat_s><longcat_observation><longcat_tool_response>get_time\n12:00</longcat_tool_response><longcat_tool_response>calc\n2</longcat_tool_response>";
        assert_eq!(
            f.tool_results(&results, true),
            format!("{observed} /think_on <longcat_assistant><longcat_think>")
        );
        assert_eq!(
            f.tool_results(&results, false),
            format!("{observed} /think_off <longcat_assistant><longcat_think>\n\n</longcat_think>\n")
        );
        // The whole conversation again, without thinking: byte for byte the template's.
        let whole = format!(
            "{}{}{}{}{}{}",
            f.first_turn(Some("Be brief."), false, &tools, "Time in Tokyo?"),
            f.assistant_text_after("Let me check.", &calls, false),
            f.tool_results(&results, false),
            f.assistant_text_after("It is noon.", &[], false),
            f.end_of_model_turn(),
            f.user_turn("Thanks, and in Paris?", false)
        );
        assert_eq!(
            whole,
            format!(
                "{declare}<longcat_system>Be brief.<longcat_user>Time in Tokyo? /think_off <longcat_assistant><longcat_think>\n\n</longcat_think>\n{reply}{observed} /think_off <longcat_assistant><longcat_think>\n\n</longcat_think>\nIt is noon.</longcat_s><longcat_user>Thanks, and in Paris? /think_off <longcat_assistant><longcat_think>\n\n</longcat_think>\n"
            )
        );
        // With thinking, a past reply without its reasoning (the template's rendering of a
        // message that has no `reasoning_content`).
        assert_eq!(
            format!(
                "{}{}{}{}",
                f.first_turn(None, true, &[], "What is the capital of France?"),
                f.assistant_text_after("Paris.", &[], true),
                f.end_of_model_turn(),
                f.user_turn("And Italy?", true)
            ),
            "<longcat_user>What is the capital of France? /think_on <longcat_assistant><longcat_think>\n\n</longcat_think>\nParis.</longcat_s><longcat_user>And Italy? /think_on <longcat_assistant><longcat_think>"
        );
    }

    /// The templates of LongCat-Flash-Lite-Sparse and LongCat-Flash-Lite in their default mode
    /// (no thinking marks), as transformers renders them.
    #[test]
    fn lite_templates() {
        let sparse = "{%- set enable_thinking = enable_thinking | default(none) -%} <longcat_user> <longcat_observation>";
        let f = LongcatChat::for_template(Some(sparse)).unwrap();
        assert!(!f.thinking && f.observation);
        let call = "<longcat_tool_call>get_time\n<longcat_arg_key>tz</longcat_arg_key>\n<longcat_arg_value>Asia/Tokyo</longcat_arg_value>\n<longcat_arg_key>n</longcat_arg_key>\n<longcat_arg_value>3</longcat_arg_value>\n</longcat_tool_call>\n";
        let calls = f.parse_tool_calls(&format!("Let me check.{call}"));
        let result = [ToolResult {
            name: "get_time".into(),
            content: "12:00".into(),
        }];
        let conv = |f: &LongcatChat| {
            format!(
                "{}{}{}{}{}{}",
                f.first_turn(Some("Be brief."), true, &[], "Time in Tokyo?"),
                f.assistant_text_after("Let me check.", &calls, true),
                f.tool_results(&result, true),
                f.assistant_text_after("It is noon.", &[], true),
                f.end_of_model_turn(),
                f.user_turn("Thanks, and in Paris?", true)
            )
        };
        assert_eq!(
            conv(f),
            format!(
                "<longcat_system>Be brief.<longcat_user>Time in Tokyo?<longcat_assistant>Let me check.{call}</longcat_s><longcat_observation><longcat_tool_response>get_time\n12:00</longcat_tool_response><longcat_assistant>It is noon.</longcat_s><longcat_user>Thanks, and in Paris?<longcat_assistant>"
            )
        );
        let lite = LongcatChat::for_template(Some("{{- \"<longcat_user>\" }} {%- if enable_thinking == true %}")).unwrap();
        assert!(!lite.thinking && !lite.observation);
        assert_eq!(
            conv(lite),
            format!(
                "<longcat_system>Be brief.<longcat_user>Time in Tokyo?<longcat_assistant>Let me check.{call}</longcat_s><longcat_user><longcat_tool_response>12:00</longcat_tool_response><longcat_assistant>It is noon.</longcat_s><longcat_user>Thanks, and in Paris?<longcat_assistant>"
            )
        );
        assert!(std::ptr::eq(LongcatChat::for_template(None).unwrap(), &LONGCAT));
        let v25 = "{%- set enable_thinking = enable_thinking | default(false) -%} <longcat_user> <longcat_observation>";
        assert!(std::ptr::eq(LongcatChat::for_template(Some(v25)).unwrap(), &LONGCAT));
        assert!(LongcatChat::for_template(Some("[Round 0] USER:")).is_err());
    }

    #[test]
    fn parses_calls_as_the_detector() {
        let tools = [ToolSpec {
            name: "f".into(),
            description: String::new(),
            parameters: serde_json::json!({"type": "object", "properties": {"s": {"type": "string"}, "n": {"type": "integer"}}}),
        }];
        // A string parameter keeps its text; others are JSON when they parse; values are trimmed;
        // a call without its closing tags is read up to the end.
        let raw = "<longcat_tool_call>f\n<longcat_arg_key>s</longcat_arg_key>\n<longcat_arg_value> 42 </longcat_arg_value>\n<longcat_arg_key>n</longcat_arg_key><longcat_arg_value>42</longcat_arg_value>\n<longcat_arg_key>x<longcat_arg_value>[1, 2]</longcat_arg_value>\n<longcat_arg_key>y</longcat_arg_key>\n<longcat_arg_value>a\nb";
        let c = &parse_calls_for(raw, &tools)[0];
        assert_eq!(c.name, "f");
        assert_eq!(c.arguments["s"], "42");
        assert_eq!(c.arguments["n"], 42);
        assert_eq!(c.arguments["x"], serde_json::json!([1, 2]));
        assert_eq!(c.arguments["y"], "a\nb");
        let none = parse_calls_for("no call here", &tools);
        assert!(none.is_empty());
        let bare = parse_calls_for("<longcat_tool_call>ping\n</longcat_tool_call>\n", &tools);
        assert_eq!((bare[0].name.as_str(), bare[0].arguments.len()), ("ping", 0));
    }
}
