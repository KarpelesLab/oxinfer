//! Chat formatting for each model family (mirrors their chat templates), including tool calling.

use crate::tools::{ToolCall, ToolResult, ToolSpec};

pub const EOT: &str = "<turn|>";

/// Text for the start of a conversation: `<bos>` plus the optional system turn.
pub fn conversation_start(system: Option<&str>, think: bool) -> String {
    let mut s = String::from("<bos>");
    if think || system.is_some() {
        s.push_str("<|turn>system\n");
        if think {
            s.push_str("<|think|>\n");
        }
        if let Some(sys) = system {
            s.push_str(sys.trim());
        }
        s.push_str("<turn|>\n");
    }
    s
}

/// A user turn followed by the generation prompt for the model.
pub fn user_turn(content: &str) -> String {
    format!("<|turn>user\n{}<turn|>\n<|turn>model\n", content.trim())
}

/// Text to append after the model has finished its reply (before the next user turn).
pub fn end_of_model_turn() -> String {
    format!("{EOT}\n")
}

/// Chat formatting for a model family.
pub trait ChatFormat {
    fn conversation_start(&self, system: Option<&str>, think: bool) -> String;
    /// A user turn followed by the generation prompt.
    fn user_turn(&self, content: &str, think: bool) -> String;
    fn end_of_model_turn(&self) -> String;
    /// The special token that closes a model turn (generation stops on it).
    fn end_of_turn_token(&self) -> &'static str;
    /// Tokens that open / close a thinking block in generated text ("" if none), and whether
    /// generation starts inside one when thinking is enabled.
    fn think_tokens(&self) -> (&'static str, &'static str);
    fn starts_in_thinking(&self, think: bool) -> bool;

    /// The start of a conversation through its first user turn and the generation prompt.
    fn first_turn(&self, system: Option<&str>, think: bool, tools: &[ToolSpec], content: &str) -> String {
        let mut s = self.conversation_start_tools(system, think, tools);
        s.push_str(&self.user_turn(content, think));
        s
    }

    /// Conversation start with tool declarations in the system turn (as the model's template).
    fn conversation_start_tools(&self, system: Option<&str>, think: bool, tools: &[ToolSpec]) -> String;
    /// The text (a special token for Gemma) that opens a tool call in generated output.
    fn tool_call_marker(&self) -> &'static str;
    /// Tool calls in a reply decoded with its special tokens.
    fn parse_tool_calls(&self, raw: &str) -> Vec<ToolCall>;
    /// [`ChatFormat::parse_tool_calls`] knowing the declared tools: formats that write string
    /// arguments bare keep a parameter declared as a string a string (`"42"`, `"true"`).
    fn parse_tool_calls_for(&self, raw: &str, tools: &[ToolSpec]) -> Vec<ToolCall> {
        let _ = tools;
        self.parse_tool_calls(raw)
    }
    /// How the model writes a tool call, for a forced one ([`crate::constrain`]); None when
    /// the format has no tool calling.
    fn tool_syntax(&self) -> Option<crate::constrain::ToolSyntax> {
        None
    }
    /// What to feed after a reply that ended on its tool calls (its stop token was sampled but not
    /// fed): the results, up to where the model continues.
    fn tool_results(&self, results: &[ToolResult], think: bool) -> String;
    /// A past reply as the model wrote it after the generation prompt, up to (not including) the
    /// token it stopped on: its text, then its tool calls (when it made any). Rendering a whole
    /// conversation again ([`crate::engine::Engine::chat`]) gives the same text as the turns that
    /// produced it.
    fn assistant_text(&self, content: &str, calls: &[ToolCall]) -> String {
        let _ = calls;
        content.to_string()
    }
}

pub struct GemmaChat;
impl ChatFormat for GemmaChat {
    fn conversation_start(&self, system: Option<&str>, think: bool) -> String {
        conversation_start(system, think)
    }
    fn user_turn(&self, content: &str, _think: bool) -> String {
        user_turn(content)
    }
    fn end_of_model_turn(&self) -> String {
        end_of_model_turn()
    }
    fn end_of_turn_token(&self) -> &'static str {
        EOT
    }
    fn think_tokens(&self) -> (&'static str, &'static str) {
        ("<|channel>", "<channel|>")
    }
    fn starts_in_thinking(&self, _think: bool) -> bool {
        false
    }
    fn conversation_start_tools(&self, system: Option<&str>, think: bool, tools: &[ToolSpec]) -> String {
        if tools.is_empty() {
            return conversation_start(system, think);
        }
        let mut s = String::from("<bos><|turn>system\n");
        if think {
            s.push_str("<|think|>\n");
        }
        if let Some(sys) = system {
            s.push_str(sys.trim());
        }
        for t in tools {
            s.push_str(&crate::tools::gemma::declaration(t));
        }
        s.push_str("<turn|>\n");
        s
    }
    fn tool_call_marker(&self) -> &'static str {
        "<|tool_call>"
    }
    fn parse_tool_calls(&self, raw: &str) -> Vec<ToolCall> {
        crate::tools::gemma::parse_calls(raw)
    }
    fn tool_syntax(&self) -> Option<crate::constrain::ToolSyntax> {
        Some(crate::constrain::ToolSyntax::Gemma)
    }
    fn tool_results(&self, results: &[ToolResult], think: bool) -> String {
        // The model stopped on `<|tool_response>`; the responses continue the same model turn.
        let mut s: String = results.iter().map(crate::tools::gemma::response).collect();
        if think {
            s.push_str("<|channel>thought\n");
        }
        s
    }
    fn assistant_text(&self, content: &str, calls: &[ToolCall]) -> String {
        let mut s = content.to_string();
        s.extend(calls.iter().map(crate::tools::gemma::call));
        s
    }
}

impl GemmaChat {
    /// The format of the Gemma 4 checkpoint in `dir` (a folder or a GGUF file): [`GemmaChatLarge`]
    /// when its template closes an empty thought channel in the generation prompt, else this.
    pub fn for_model(dir: &std::path::Path) -> &'static (dyn ChatFormat + Sync) {
        let template = if crate::gguf::is_gguf(dir) {
            crate::gguf::header(dir).ok().and_then(|h| h.chat_template().map(str::to_string))
        } else {
            crate::fsx::read(&dir.join("chat_template.jinja"))
                .ok()
                .map(|b| String::from_utf8_lossy(&b).into_owned())
                .or_else(|| {
                    let cfg: serde_json::Value =
                        serde_json::from_slice(&crate::fsx::read(&dir.join("tokenizer_config.json")).ok()?).ok()?;
                    cfg["chat_template"].as_str().map(str::to_string)
                })
        };
        match template {
            Some(t) if t.contains(r"'<|channel>thought\n<channel|>'") => &GemmaChatLarge,
            _ => &GemmaChat,
        }
    }
}

/// Gemma 4 12B / 31B / 26B-A4B: as [`GemmaChat`], but with thinking off the generation prompt
/// opens and closes an empty thought channel (`<|turn>model\n<|channel>thought\n<channel|>`), as
/// these models' chat template does.
pub struct GemmaChatLarge;
impl ChatFormat for GemmaChatLarge {
    fn conversation_start(&self, system: Option<&str>, think: bool) -> String {
        GemmaChat.conversation_start(system, think)
    }
    fn user_turn(&self, content: &str, think: bool) -> String {
        let mut s = user_turn(content);
        if !think {
            s.push_str("<|channel>thought\n<channel|>");
        }
        s
    }
    fn end_of_model_turn(&self) -> String {
        GemmaChat.end_of_model_turn()
    }
    fn end_of_turn_token(&self) -> &'static str {
        GemmaChat.end_of_turn_token()
    }
    fn think_tokens(&self) -> (&'static str, &'static str) {
        GemmaChat.think_tokens()
    }
    fn starts_in_thinking(&self, think: bool) -> bool {
        GemmaChat.starts_in_thinking(think)
    }
    fn conversation_start_tools(&self, system: Option<&str>, think: bool, tools: &[ToolSpec]) -> String {
        GemmaChat.conversation_start_tools(system, think, tools)
    }
    fn tool_call_marker(&self) -> &'static str {
        GemmaChat.tool_call_marker()
    }
    fn parse_tool_calls(&self, raw: &str) -> Vec<ToolCall> {
        GemmaChat.parse_tool_calls(raw)
    }
    fn tool_syntax(&self) -> Option<crate::constrain::ToolSyntax> {
        GemmaChat.tool_syntax()
    }
    fn tool_results(&self, results: &[ToolResult], think: bool) -> String {
        GemmaChat.tool_results(results, think)
    }
    fn assistant_text(&self, content: &str, calls: &[ToolCall]) -> String {
        GemmaChat.assistant_text(content, calls)
    }
}

/// Gemma 3: `<start_of_turn>` turns, no system role (the system prompt opens the first user
/// turn), no thinking and no tool tokens.
pub struct Gemma3Chat;
impl ChatFormat for Gemma3Chat {
    fn conversation_start(&self, _system: Option<&str>, _think: bool) -> String {
        "<bos>".into()
    }
    fn user_turn(&self, content: &str, _think: bool) -> String {
        format!("<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n", content.trim())
    }
    fn first_turn(&self, system: Option<&str>, _think: bool, _tools: &[ToolSpec], content: &str) -> String {
        let prefix = system.map(|s| format!("{s}\n\n")).unwrap_or_default();
        format!(
            "<bos><start_of_turn>user\n{prefix}{}<end_of_turn>\n<start_of_turn>model\n",
            content.trim()
        )
    }
    fn end_of_model_turn(&self) -> String {
        "<end_of_turn>\n".into()
    }
    fn end_of_turn_token(&self) -> &'static str {
        "<end_of_turn>"
    }
    fn think_tokens(&self) -> (&'static str, &'static str) {
        ("", "")
    }
    fn starts_in_thinking(&self, _think: bool) -> bool {
        false
    }
    fn conversation_start_tools(&self, system: Option<&str>, think: bool, _tools: &[ToolSpec]) -> String {
        self.conversation_start(system, think)
    }
    fn tool_call_marker(&self) -> &'static str {
        ""
    }
    fn parse_tool_calls(&self, _raw: &str) -> Vec<ToolCall> {
        Vec::new()
    }
    fn tool_results(&self, _results: &[ToolResult], _think: bool) -> String {
        String::new()
    }
}

/// Qwen ChatML templates. Qwen3.5 / 3.8: thinking is on by default in the model, so a non-thinking
/// request closes the think block in the generation prompt, and a thinking one opens it; Qwen3.8's
/// also adds a reasoning-effort instruction to the system turn in thinking mode (and a small model
/// given that instruction anyway can deliberate about it endlessly). Qwen3: the model opens its
/// own think block, and tools are declared and called in JSON.
pub struct QwenChat {
    pub reasoning_effort: bool,
    pub qwen3: bool,
}

/// Qwen3.5, Qwen3.8 (with the reasoning-effort instruction) and Qwen3 formats.
pub static QWEN: QwenChat = QwenChat {
    reasoning_effort: false,
    qwen3: false,
};
pub static QWEN_EFFORT: QwenChat = QwenChat {
    reasoning_effort: true,
    qwen3: false,
};
pub static QWEN3: QwenChat = QwenChat {
    reasoning_effort: false,
    qwen3: true,
};

impl QwenChat {
    /// The format matching the model's own chat template (`chat_template.jinja`, or the template
    /// embedded in `tokenizer_config.json`).
    pub fn for_model(dir: &std::path::Path) -> &'static QwenChat {
        if crate::gguf::is_gguf(dir) {
            return Self::for_template(crate::gguf::header(dir).ok().as_deref().and_then(|h| h.chat_template()));
        }
        let jinja = crate::fsx::read(&dir.join("chat_template.jinja"))
            .ok()
            .map(|b| String::from_utf8_lossy(&b).into_owned());
        let template = jinja.or_else(|| {
            let cfg: serde_json::Value = serde_json::from_slice(&crate::fsx::read(&dir.join("tokenizer_config.json")).ok()?).ok()?;
            cfg["chat_template"].as_str().map(str::to_string)
        });
        Self::for_template(template.as_deref())
    }

    /// The format matching a chat template's source (None: plain Qwen).
    pub fn for_template(template: Option<&str>) -> &'static QwenChat {
        match template {
            Some(t) if t.contains("reasoning_effort") => &QWEN_EFFORT,
            Some(t) if t.contains("<function-name>") => &QWEN3,
            _ => &QWEN,
        }
    }

    /// The instruction this template puts first in the system turn when thinking.
    fn effort(&self, think: bool) -> Option<&'static str> {
        (think && self.reasoning_effort).then_some(QWEN_REASONING_XHIGH)
    }
}
const QWEN_REASONING_XHIGH: &str = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
impl ChatFormat for QwenChat {
    fn conversation_start(&self, system: Option<&str>, think: bool) -> String {
        let mut s = String::new();
        match (self.effort(think), system) {
            (Some(e), Some(sys)) => s.push_str(&format!("<|im_start|>system\n{e}\n\n{}<|im_end|>\n", sys.trim())),
            (Some(e), None) => s.push_str(&format!("<|im_start|>system\n{e}<|im_end|>\n")),
            (None, Some(sys)) if self.qwen3 => s.push_str(&format!("<|im_start|>system\n{sys}<|im_end|>\n")),
            (None, Some(sys)) => s.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", sys.trim())),
            (None, None) => {}
        }
        s
    }
    fn user_turn(&self, content: &str, think: bool) -> String {
        let content = if self.qwen3 { content } else { content.trim() };
        format!(
            "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n{}",
            self.think_tail(think)
        )
    }
    fn end_of_model_turn(&self) -> String {
        "<|im_end|>\n".to_string()
    }
    fn end_of_turn_token(&self) -> &'static str {
        "<|im_end|>"
    }
    fn think_tokens(&self) -> (&'static str, &'static str) {
        ("<think>", "</think>")
    }
    fn starts_in_thinking(&self, think: bool) -> bool {
        think && !self.qwen3
    }
    fn conversation_start_tools(&self, system: Option<&str>, think: bool, tools: &[ToolSpec]) -> String {
        if tools.is_empty() {
            return self.conversation_start(system, think);
        }
        let mut s = String::from("<|im_start|>system\n");
        if self.qwen3 {
            if let Some(sys) = system {
                s.push_str(sys);
                s.push_str("\n\n");
            }
            s.push_str(&crate::tools::qwen::tools_block_json(tools));
            s.push_str("<|im_end|>\n");
            return s;
        }
        if let Some(e) = self.effort(think) {
            s.push_str(e);
            s.push_str("\n\n");
        }
        s.push_str(&crate::tools::qwen::tools_block(tools));
        if let Some(sys) = system.map(str::trim).filter(|s| !s.is_empty()) {
            s.push_str("\n\n");
            s.push_str(sys);
        }
        s.push_str("<|im_end|>\n");
        s
    }
    fn tool_call_marker(&self) -> &'static str {
        "<tool_call>"
    }
    fn parse_tool_calls(&self, raw: &str) -> Vec<ToolCall> {
        crate::tools::qwen::parse_calls(raw)
    }
    fn parse_tool_calls_for(&self, raw: &str, tools: &[ToolSpec]) -> Vec<ToolCall> {
        crate::tools::qwen::parse_calls_for(raw, tools)
    }
    fn tool_syntax(&self) -> Option<crate::constrain::ToolSyntax> {
        Some(if self.qwen3 {
            crate::constrain::ToolSyntax::QwenJson
        } else {
            crate::constrain::ToolSyntax::QwenXml
        })
    }
    fn tool_results(&self, results: &[ToolResult], think: bool) -> String {
        // The assistant turn ended (<|im_end|> was sampled, not fed); results come as a user turn.
        let mut s = String::from("<|im_end|>\n<|im_start|>user");
        for r in results {
            s.push_str("\n<tool_response>\n");
            s.push_str(&r.content);
            s.push_str("\n</tool_response>");
        }
        s.push_str("<|im_end|>\n<|im_start|>assistant\n");
        s.push_str(self.think_tail(think));
        s
    }
    fn assistant_text(&self, content: &str, calls: &[ToolCall]) -> String {
        if calls.is_empty() {
            return content.to_string();
        }
        // The template's layout: the text, a blank line, the calls one per line.
        let mut s = content.trim().to_string();
        if !s.is_empty() {
            s.push_str("\n\n");
        }
        let calls: Vec<String> = calls.iter().map(|c| crate::tools::qwen::call(c, self.qwen3)).collect();
        s.push_str(&calls.join("\n"));
        s
    }
}

impl QwenChat {
    /// What follows the assistant header in the generation prompt.
    fn think_tail(&self, think: bool) -> &'static str {
        match (think, self.qwen3) {
            (false, _) => "<think>\n\n</think>\n\n",
            (true, false) => "<think>\n",
            (true, true) => "",
        }
    }
}

/// Qwen2 / Qwen2.5 ChatML: like Qwen3 without thinking, and the template's default system
/// prompt when none is given.
pub struct Qwen2Chat {
    pub default_system: Option<String>,
}

impl Qwen2Chat {
    /// The format of the checkpoint in `dir`, from its chat template (`chat_template.jinja`,
    /// `chat_template.json` or `tokenizer_config.json`): Qwen2.5-VL's template writes its default
    /// system turn inline ("You are a helpful assistant.").
    pub fn for_model(dir: &std::path::Path) -> Qwen2Chat {
        let read = |name: &str| crate::fsx::read(&dir.join(name)).ok();
        let json_template = |name: &str| -> Option<String> {
            let v: serde_json::Value = serde_json::from_slice(&read(name)?).ok()?;
            v["chat_template"].as_str().map(str::to_string)
        };
        let template = read("chat_template.jinja")
            .map(|b| String::from_utf8_lossy(&b).into_owned())
            .or_else(|| json_template("chat_template.json"))
            .or_else(|| json_template("tokenizer_config.json"))
            .unwrap_or_default();
        let fmt = Self::for_template(&template);
        if fmt.default_system.is_some() {
            return fmt;
        }
        const OPEN: &str = "%}<|im_start|>system\n";
        let inline = template.find(OPEN).and_then(|i| {
            let rest = &template[i + OPEN.len()..];
            let body = &rest[..rest.find("<|im_end|>")?];
            (!body.contains(['{', '}', '\''])).then(|| body.to_string())
        });
        Qwen2Chat { default_system: inline }
    }

    /// The format of a chat template: its default system prompt is the literal in
    /// `'<|im_start|>system\n...<|im_end|>\n'` (Qwen2.5: "You are Qwen, created by Alibaba
    /// Cloud. You are a helpful assistant."; fine-tunes change it).
    pub fn for_template(template: &str) -> Qwen2Chat {
        const OPEN: &str = "'<|im_start|>system\\n";
        const CLOSE: &str = "<|im_end|>\\n'";
        let default_system = template.match_indices(OPEN).find_map(|(i, _)| {
            let rest = &template[i + OPEN.len()..];
            let body = &rest[..rest.find(CLOSE)?];
            // Skip the `'<|im_start|>system\n' + messages[0]['content']` concatenation.
            if body.contains('\'') {
                return None;
            }
            let mut out = String::new();
            let mut it = body.chars();
            while let Some(c) = it.next() {
                match (c, c == '\\') {
                    (_, true) => match it.next() {
                        Some('n') => out.push('\n'),
                        Some(o) => out.push(o),
                        None => {}
                    },
                    (c, false) => out.push(c),
                }
            }
            Some(out)
        });
        Qwen2Chat { default_system }
    }
}

impl ChatFormat for Qwen2Chat {
    fn conversation_start(&self, system: Option<&str>, _think: bool) -> String {
        match system.or(self.default_system.as_deref()) {
            Some(s) => format!("<|im_start|>system\n{s}<|im_end|>\n"),
            None => String::new(),
        }
    }
    fn user_turn(&self, content: &str, _think: bool) -> String {
        format!("<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n")
    }
    fn end_of_model_turn(&self) -> String {
        "<|im_end|>\n".to_string()
    }
    fn end_of_turn_token(&self) -> &'static str {
        "<|im_end|>"
    }
    fn think_tokens(&self) -> (&'static str, &'static str) {
        ("", "")
    }
    fn starts_in_thinking(&self, _think: bool) -> bool {
        false
    }
    fn conversation_start_tools(&self, system: Option<&str>, think: bool, _tools: &[ToolSpec]) -> String {
        self.conversation_start(system, think)
    }
    fn tool_call_marker(&self) -> &'static str {
        ""
    }
    fn parse_tool_calls(&self, _raw: &str) -> Vec<ToolCall> {
        Vec::new()
    }
    fn tool_results(&self, _results: &[ToolResult], _think: bool) -> String {
        String::new()
    }
}

/// Llama 3.x instruct: header-delimited turns; the template always opens with a system turn
/// carrying the knowledge cutoff and today's date ("25 Oct 2026", as `strftime_now("%d %b %Y")`).
pub struct Llama3Chat;
impl Llama3Chat {
    /// Whether the model's chat template is this format: always for safetensors folders (the
    /// family is Llama 3), and for GGUF files whose `tokenizer.chat_template` has no template or
    /// Llama 3's header tokens (other Llama-architecture models, such as SmolLM or Mistral, use
    /// other formats).
    pub fn matches_model(dir: &std::path::Path) -> bool {
        if !crate::gguf::is_gguf(dir) {
            return true;
        }
        let h = crate::gguf::header(dir).ok();
        h.as_deref()
            .and_then(|h| h.chat_template())
            .is_none_or(|t| t.contains("<|start_header_id|>"))
    }

    fn today() -> String {
        const MONTHS: [&str; 12] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
        let (t, _, _, _) = crate::tools::local_now();
        let m: usize = t[5..7].parse().unwrap_or(1);
        format!("{} {} {}", &t[8..10], MONTHS[m.clamp(1, 12) - 1], &t[..4])
    }
}
impl ChatFormat for Llama3Chat {
    fn conversation_start(&self, system: Option<&str>, _think: bool) -> String {
        format!(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: {}\n\n{}<|eot_id|>",
            Self::today(),
            system.map(str::trim).unwrap_or("")
        )
    }
    fn user_turn(&self, content: &str, _think: bool) -> String {
        format!(
            "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            content.trim()
        )
    }
    fn end_of_model_turn(&self) -> String {
        "<|eot_id|>".to_string()
    }
    fn end_of_turn_token(&self) -> &'static str {
        "<|eot_id|>"
    }
    fn think_tokens(&self) -> (&'static str, &'static str) {
        ("", "")
    }
    fn starts_in_thinking(&self, _think: bool) -> bool {
        false
    }
    fn conversation_start_tools(&self, system: Option<&str>, think: bool, _tools: &[ToolSpec]) -> String {
        self.conversation_start(system, think)
    }
    fn tool_call_marker(&self) -> &'static str {
        ""
    }
    fn parse_tool_calls(&self, _raw: &str) -> Vec<ToolCall> {
        Vec::new()
    }
    fn tool_results(&self, _results: &[ToolResult], _think: bool) -> String {
        String::new()
    }
}

/// Mistral Large 4 (its `chat_template.jinja`): `<s>[SYSTEM_PROMPT]...[/SYSTEM_PROMPT]`, the
/// tools and `[MODEL_SETTINGS]{"reasoning_effort": "..."}[/MODEL_SETTINGS]` before the first
/// user turn, `[INST]...[/INST]` user turns, replies closed by `</s>`. The template always sets
/// a reasoning effort (default "high"); thinking off asks for "low". Thinking is written as
/// `[THINK]...[/THINK]` at the start of the reply.
pub struct MistralChat;

impl MistralChat {
    fn settings(think: bool) -> String {
        format!(
            "[MODEL_SETTINGS]{{\"reasoning_effort\": \"{}\"}}[/MODEL_SETTINGS]",
            if think { "high" } else { "low" }
        )
    }
}

impl ChatFormat for MistralChat {
    fn conversation_start(&self, system: Option<&str>, think: bool) -> String {
        self.conversation_start_tools(system, think, &[])
    }
    fn user_turn(&self, content: &str, _think: bool) -> String {
        format!("[INST]{content}[/INST]")
    }
    fn end_of_model_turn(&self) -> String {
        "</s>".to_string()
    }
    fn end_of_turn_token(&self) -> &'static str {
        "</s>"
    }
    fn think_tokens(&self) -> (&'static str, &'static str) {
        ("[THINK]", "[/THINK]")
    }
    fn starts_in_thinking(&self, _think: bool) -> bool {
        false
    }
    fn conversation_start_tools(&self, system: Option<&str>, think: bool, tools: &[ToolSpec]) -> String {
        let mut s = String::from("<s>");
        if let Some(sys) = system {
            s.push_str(&format!("[SYSTEM_PROMPT]{sys}[/SYSTEM_PROMPT]"));
        }
        if !tools.is_empty() {
            s.push_str(&crate::tools::mistral::available_tools(tools));
        }
        s.push_str(&Self::settings(think));
        s
    }
    fn tool_call_marker(&self) -> &'static str {
        "[TOOL_CALLS]"
    }
    fn parse_tool_calls(&self, raw: &str) -> Vec<ToolCall> {
        crate::tools::mistral::parse_calls(raw)
    }
    fn tool_syntax(&self) -> Option<crate::constrain::ToolSyntax> {
        Some(crate::constrain::ToolSyntax::Mistral)
    }
    fn tool_results(&self, results: &[ToolResult], _think: bool) -> String {
        // The reply ended on `</s>` (sampled, not fed); the results follow it.
        let mut s = String::from("</s>");
        s.extend(results.iter().map(crate::tools::mistral::response));
        s
    }
    fn assistant_text(&self, content: &str, calls: &[ToolCall]) -> String {
        let mut s = content.to_string();
        s.extend(calls.iter().map(crate::tools::mistral::call));
        s
    }
}

/// DeepSeek V4 (its `encoding_dsv4.py`): `<｜begin▁of▁sentence｜>` then the system prompt as is
/// (tool schemas appended after a blank line), user turns `<｜User｜>...<｜Assistant｜>` followed by
/// `<think>` (thinking) or `</think>` (chat), replies closed by `<｜end▁of▁sentence｜>`. Tool
/// calls are DSML blocks; their results come back as `<tool_result>` blocks of a user turn.
pub struct DeepSeekChat;

impl DeepSeekChat {
    const EOS: &'static str = "<｜end▁of▁sentence｜>";
    fn assistant(think: bool) -> &'static str {
        if think {
            "<｜Assistant｜><think>"
        } else {
            "<｜Assistant｜></think>"
        }
    }
}

impl ChatFormat for DeepSeekChat {
    fn conversation_start(&self, system: Option<&str>, think: bool) -> String {
        self.conversation_start_tools(system, think, &[])
    }
    fn user_turn(&self, content: &str, think: bool) -> String {
        format!("<｜User｜>{content}{}", Self::assistant(think))
    }
    fn end_of_model_turn(&self) -> String {
        Self::EOS.to_string()
    }
    fn end_of_turn_token(&self) -> &'static str {
        Self::EOS
    }
    fn think_tokens(&self) -> (&'static str, &'static str) {
        ("<think>", "</think>")
    }
    fn starts_in_thinking(&self, think: bool) -> bool {
        think
    }
    fn conversation_start_tools(&self, system: Option<&str>, _think: bool, tools: &[ToolSpec]) -> String {
        let mut s = String::from("<｜begin▁of▁sentence｜>");
        if let Some(sys) = system {
            s.push_str(sys);
        }
        if !tools.is_empty() {
            if system.is_some() {
                s.push_str("\n\n");
            }
            s.push_str(&crate::tools::deepseek::tools_block(tools));
        }
        s
    }
    fn tool_call_marker(&self) -> &'static str {
        crate::tools::deepseek::MARKER
    }
    fn parse_tool_calls(&self, raw: &str) -> Vec<ToolCall> {
        crate::tools::deepseek::parse_calls(raw)
    }
    fn tool_syntax(&self) -> Option<crate::constrain::ToolSyntax> {
        Some(crate::constrain::ToolSyntax::Dsml)
    }
    fn tool_results(&self, results: &[ToolResult], think: bool) -> String {
        // The reply ended on its end-of-sentence token (sampled, not fed); the results are a user
        // turn.
        let blocks: Vec<String> = results.iter().map(crate::tools::deepseek::response).collect();
        format!("{}<｜User｜>{}{}", Self::EOS, blocks.join("\n\n"), Self::assistant(think))
    }
    fn assistant_text(&self, content: &str, calls: &[ToolCall]) -> String {
        if calls.is_empty() {
            return content.to_string();
        }
        format!("{}{}", content.trim_end(), crate::tools::deepseek::calls(calls))
    }
}

/// DeepSeek V4.1 (its `encoding/encoding.py`): V4's format with a `<｜System｜>` token in front of
/// the system part (the system prompt, the tool schemas after a blank line, and in thinking mode
/// a first line giving the reasoning effort as a number, 75 of 100 by default), and DSML tags
/// whose names start with a space (`<｜DSML｜ calls>`).
pub struct DeepSeek41Chat;

impl DeepSeek41Chat {
    /// The effort line of thinking mode (`"high"`, the encoder's default).
    const EFFORT: &'static str = "Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n";
}

impl ChatFormat for DeepSeek41Chat {
    fn conversation_start(&self, system: Option<&str>, think: bool) -> String {
        self.conversation_start_tools(system, think, &[])
    }
    fn user_turn(&self, content: &str, think: bool) -> String {
        DeepSeekChat.user_turn(content, think)
    }
    fn end_of_model_turn(&self) -> String {
        DeepSeekChat::EOS.to_string()
    }
    fn end_of_turn_token(&self) -> &'static str {
        DeepSeekChat::EOS
    }
    fn think_tokens(&self) -> (&'static str, &'static str) {
        ("<think>", "</think>")
    }
    fn starts_in_thinking(&self, think: bool) -> bool {
        think
    }
    fn conversation_start_tools(&self, system: Option<&str>, think: bool, tools: &[ToolSpec]) -> String {
        let mut s = String::from("<｜begin▁of▁sentence｜>");
        // (Tools are declared in a system message: an empty one when there is no system prompt.)
        let has_system = system.is_some() || !tools.is_empty();
        if think || has_system {
            s.push_str("<｜System｜>");
        }
        if think {
            s.push_str(Self::EFFORT);
        }
        s.push_str(system.unwrap_or(""));
        if !tools.is_empty() {
            s.push_str("\n\n");
            s.push_str(&crate::tools::deepseek41::tools_block(tools));
        }
        s
    }
    fn tool_call_marker(&self) -> &'static str {
        crate::tools::deepseek41::MARKER
    }
    fn parse_tool_calls(&self, raw: &str) -> Vec<ToolCall> {
        crate::tools::deepseek41::parse_calls(raw)
    }
    fn tool_syntax(&self) -> Option<crate::constrain::ToolSyntax> {
        Some(crate::constrain::ToolSyntax::Dsml41)
    }
    fn tool_results(&self, results: &[ToolResult], think: bool) -> String {
        DeepSeekChat.tool_results(results, think)
    }
    fn assistant_text(&self, content: &str, calls: &[ToolCall]) -> String {
        if calls.is_empty() {
            return content.to_string();
        }
        format!("{}{}", content.trim_end(), crate::tools::deepseek41::calls(calls))
    }
}

/// MiniMax M2 / M2.1 / M2.5 / M2.7 (and M3's text turns): `]~!b[` then `]~b]ROLE\n...[e~[\n` turns;
/// replies always open a thinking block (`<think>\n` ends the generation prompt); tool results
/// are a `tool` turn of `<response>` blocks. Without a system prompt the template puts the
/// model's identity there, which differs between releases (read from the model's template).
pub struct MiniMaxChat {
    pub identity: String,
}

impl MiniMaxChat {
    const BOS: &'static str = "]~!b[";
    const EOT: &'static str = "[e~[";

    /// The format of the model in `dir` (its template's `model_identity`).
    pub fn for_model(dir: &std::path::Path) -> &'static MiniMaxChat {
        Box::leak(Box::new(Self::for_template(model_template(dir).as_deref())))
    }

    pub fn for_template(template: Option<&str>) -> MiniMaxChat {
        let identity = template
            .and_then(|t| t.split("set model_identity = \"").nth(1))
            .and_then(|r| r.split_once('"'))
            .map(|(id, _)| id.to_string())
            .unwrap_or_else(|| "You are a helpful assistant.".to_string());
        MiniMaxChat { identity }
    }

    fn ai(think: bool) -> &'static str {
        // The template always opens a thinking block; without thinking it is closed empty.
        if think {
            "]~b]ai\n<think>\n"
        } else {
            "]~b]ai\n<think>\n\n</think>\n\n"
        }
    }
}

/// MiniMax M3 / M3.1: a system turn with the model's identity, its thinking instructions and id,
/// then a `developer` turn with the system prompt (default "You are a helpful assistant.") and the
/// tools; thinking in `<mm:think>`; tool calls as `]<]minimax[>[`-tagged XML (see
/// `crate::tools::minimax3`). M3's template has thinking modes: on (`<mm:think>` opens the reply)
/// or off (`</mm:think>` does); M3.1's is always adaptive (the model decides; off prefills
/// `</mm:think>`, as its instructions describe).
pub struct MiniMax3Chat {
    /// The system turn's first paragraph ("Your model version is MiniMax-M3.1, ...").
    pub identity: String,
    /// The template takes `thinking_mode` (M3).
    pub modes: bool,
}

impl MiniMax3Chat {
    const ADAPTIVE: &'static str = "Current thinking mode: adaptive. You are encouraged to think for complex decision-making, multi-step reasoning, or when analyzing function/tool results.\n";

    /// Whether a chat template is M3's (rather than M2's).
    pub fn is_template(t: &str) -> bool {
        t.contains("<mm:think>")
    }

    pub fn for_template(t: &str) -> MiniMax3Chat {
        let identity = t
            .split("'Your model version is ")
            .nth(1)
            .and_then(|r| r.split_once('\''))
            .map(|(id, _)| format!("Your model version is {id}"))
            .unwrap_or_else(|| "Your model version is MiniMax-M3, developed by MiniMax.".to_string());
        MiniMax3Chat {
            identity,
            modes: t.contains("thinking_mode"),
        }
    }

    fn mode(&self, think: bool) -> &'static str {
        match (self.modes, think) {
            (false, _) => Self::ADAPTIVE,
            (true, true) => {
                "Current thinking mode: enabled. You MUST think step by step before every response, including after receiving function/tool results.\n"
            }
            (true, false) => "Current thinking mode: disabled. Do not output any thinking process.\n",
        }
    }

    fn ai(&self, think: bool) -> &'static str {
        match (self.modes, think) {
            (true, true) => "]~b]ai\n<mm:think>",
            (false, true) => "]~b]ai\n",
            (_, false) => "]~b]ai\n</mm:think>",
        }
    }
}

impl ChatFormat for MiniMax3Chat {
    fn conversation_start(&self, system: Option<&str>, think: bool) -> String {
        self.conversation_start_tools(system, think, &[])
    }
    fn user_turn(&self, content: &str, think: bool) -> String {
        format!("]~b]user\n{content}[e~[\n{}", self.ai(think))
    }
    fn end_of_model_turn(&self) -> String {
        "[e~[\n".to_string()
    }
    fn end_of_turn_token(&self) -> &'static str {
        "[e~["
    }
    fn think_tokens(&self) -> (&'static str, &'static str) {
        ("<mm:think>", "</mm:think>")
    }
    fn starts_in_thinking(&self, think: bool) -> bool {
        think && self.modes
    }
    fn conversation_start_tools(&self, system: Option<&str>, think: bool, tools: &[ToolSpec]) -> String {
        let mut s = format!(
            "]~!b[]~b]system\n{}\n\n<thinking_instructions>\nYou have a thinking capability that allows you to reason step by step before responding. When thinking is enabled, wrap your reasoning in <mm:think></mm:think> tags before your response. When thinking is disabled, begin your response directly after the </mm:think> prefix. When thinking is adaptive, decide on your own whether to think for the current turn.\n{}</thinking_instructions>\n\n<model_id>8bb8149fe</model_id>[e~[\n]~b]developer\n",
            self.identity,
            self.mode(think)
        );
        s.push_str(system.filter(|s| !s.is_empty()).unwrap_or("You are a helpful assistant."));
        if !tools.is_empty() {
            s.push_str(&crate::tools::minimax3::tools_block(tools));
        }
        s.push_str("[e~[\n");
        s
    }
    fn tool_call_marker(&self) -> &'static str {
        crate::tools::minimax3::NS
    }
    fn parse_tool_calls(&self, raw: &str) -> Vec<ToolCall> {
        crate::tools::minimax3::parse_calls(raw)
    }
    fn tool_results(&self, results: &[ToolResult], think: bool) -> String {
        let mut s = String::from("[e~[\n]~b]tool");
        for r in results {
            s.push_str(&format!("\n<response>{}</response>", r.content));
        }
        s.push_str("[e~[\n");
        s.push_str(self.ai(think));
        s
    }
    fn assistant_text(&self, content: &str, calls: &[ToolCall]) -> String {
        if calls.is_empty() {
            return content.to_string();
        }
        format!("{content}{}", crate::tools::minimax3::calls(calls))
    }
}

/// MiniMax-Text-01, VL-01 and M1: `<beginning_of_sentence>ROLE name=...\n...<end_of_sentence>\n`
/// turns. M1 opens the document with `<begin_of_document>`, declares tools in a system turn and
/// thinks in plain-text `<think>` blocks; Text-01 declares each tool in its own system turn after
/// the conversation (here after the first user turn) and answers tool calls with
/// `function_response` system turns.
pub struct MiniMax01Chat {
    pub m1: bool,
}

impl MiniMax01Chat {
    const AI: &'static str = "<beginning_of_sentence>ai name=assistant\n";
    const EOS: &'static str = "<end_of_sentence>";

    /// Whether a chat template is this family's.
    pub fn is_template(t: &str) -> bool {
        t.contains("<beginning_of_sentence>")
    }

    pub fn for_template(t: &str) -> MiniMax01Chat {
        MiniMax01Chat {
            m1: t.contains("<begin_of_document>"),
        }
    }
}

impl ChatFormat for MiniMax01Chat {
    fn conversation_start(&self, system: Option<&str>, think: bool) -> String {
        self.conversation_start_tools(system, think, &[])
    }
    fn first_turn(&self, system: Option<&str>, think: bool, tools: &[ToolSpec], content: &str) -> String {
        if self.m1 {
            let mut s = self.conversation_start_tools(system, think, tools);
            s.push_str(&self.user_turn(content, think));
            return s;
        }
        // Text-01's template lists the tools after the messages.
        let mut s = self.conversation_start_tools(system, think, &[]);
        s.push_str(&format!("<beginning_of_sentence>user name=user\n{content}{}\n", Self::EOS));
        s.push_str(&crate::tools::minimax01::text01_tools_turns(tools));
        s.push_str(Self::AI);
        s
    }
    fn user_turn(&self, content: &str, _think: bool) -> String {
        format!("<beginning_of_sentence>user name=user\n{content}{}\n{}", Self::EOS, Self::AI)
    }
    fn end_of_model_turn(&self) -> String {
        format!("{}\n", Self::EOS)
    }
    fn end_of_turn_token(&self) -> &'static str {
        Self::EOS
    }
    fn think_tokens(&self) -> (&'static str, &'static str) {
        if self.m1 { ("<think>", "</think>") } else { ("", "") }
    }
    fn starts_in_thinking(&self, _think: bool) -> bool {
        false
    }
    fn conversation_start_tools(&self, system: Option<&str>, _think: bool, tools: &[ToolSpec]) -> String {
        let mut s = String::from(if self.m1 { "<begin_of_document>" } else { "" });
        if let Some(sys) = system.filter(|s| !s.is_empty()) {
            s.push_str(&format!("<beginning_of_sentence>system ai_setting=assistant\n{sys}{}\n", Self::EOS));
        }
        if self.m1 && !tools.is_empty() {
            s.push_str(&crate::tools::minimax01::m1_tools_turn(tools));
        }
        s
    }
    fn tool_call_marker(&self) -> &'static str {
        if self.m1 { "<tool_calls>" } else { "<function_call>" }
    }
    fn parse_tool_calls(&self, raw: &str) -> Vec<ToolCall> {
        if self.m1 {
            crate::tools::minimax01::parse_m1(raw)
        } else {
            crate::tools::minimax01::parse_text01(raw)
        }
    }
    fn tool_results(&self, results: &[ToolResult], _think: bool) -> String {
        let mut s = format!("{}\n", Self::EOS);
        if self.m1 {
            s.push_str("<beginning_of_sentence>tool name=tools\n");
            for r in results {
                s.push_str(&format!("tool name: {}\ntool result: {}\n\n", r.name, r.content));
            }
            s.push_str(&format!("{}\n", Self::EOS));
        } else {
            for r in results {
                s.push_str(&format!(
                    "<beginning_of_sentence>system function_response=functions\n{{\"name\": \"{}\", \"response\": {}}}{}\n",
                    r.name,
                    r.content,
                    Self::EOS
                ));
            }
        }
        s.push_str(Self::AI);
        s
    }
    fn assistant_text(&self, content: &str, calls: &[ToolCall]) -> String {
        if calls.is_empty() {
            return content.to_string();
        }
        let c = if self.m1 {
            crate::tools::minimax01::m1_calls(calls)
        } else {
            crate::tools::minimax01::text01_calls(calls)
        };
        format!("{}\n{c}", content.trim_end())
    }
}

/// The chat format of a MiniMax model: M3's, MiniMax-01's / M1's or M2's, by its template.
pub fn minimax_for_model(dir: &std::path::Path) -> &'static (dyn ChatFormat + Sync) {
    let t = model_template(dir);
    match t.as_deref() {
        Some(t) if MiniMax3Chat::is_template(t) => Box::leak(Box::new(MiniMax3Chat::for_template(t))),
        Some(t) if MiniMax01Chat::is_template(t) => Box::leak(Box::new(MiniMax01Chat::for_template(t))),
        t => Box::leak(Box::new(MiniMaxChat::for_template(t))),
    }
}

/// The chat template source of the model in `dir` (`chat_template.jinja`, the one in
/// `tokenizer_config.json`, or a GGUF file's).
pub fn model_template(dir: &std::path::Path) -> Option<String> {
    if crate::gguf::is_gguf(dir) {
        return crate::gguf::header(dir)
            .ok()
            .as_deref()
            .and_then(|h| h.chat_template())
            .map(str::to_string);
    }
    let jinja = crate::fsx::read(&dir.join("chat_template.jinja"))
        .ok()
        .map(|b| String::from_utf8_lossy(&b).into_owned());
    jinja.or_else(|| {
        let cfg: serde_json::Value = serde_json::from_slice(&crate::fsx::read(&dir.join("tokenizer_config.json")).ok()?).ok()?;
        cfg["chat_template"].as_str().map(str::to_string)
    })
}

impl ChatFormat for MiniMaxChat {
    fn conversation_start(&self, system: Option<&str>, think: bool) -> String {
        self.conversation_start_tools(system, think, &[])
    }
    fn user_turn(&self, content: &str, think: bool) -> String {
        format!("]~b]user\n{content}{}\n{}", Self::EOT, Self::ai(think))
    }
    fn end_of_model_turn(&self) -> String {
        format!("{}\n", Self::EOT)
    }
    fn end_of_turn_token(&self) -> &'static str {
        Self::EOT
    }
    fn think_tokens(&self) -> (&'static str, &'static str) {
        ("<think>", "</think>")
    }
    fn starts_in_thinking(&self, think: bool) -> bool {
        think
    }
    fn conversation_start_tools(&self, system: Option<&str>, _think: bool, tools: &[ToolSpec]) -> String {
        let mut s = format!("{}]~b]system\n", Self::BOS);
        match system.filter(|s| !s.is_empty()) {
            Some(sys) => s.push_str(sys),
            None => s.push_str(&self.identity),
        }
        if !tools.is_empty() {
            s.push_str(&crate::tools::minimax::tools_block(tools));
        }
        s.push_str(Self::EOT);
        s.push('\n');
        s
    }
    fn tool_call_marker(&self) -> &'static str {
        crate::tools::minimax::MARKER
    }
    fn parse_tool_calls(&self, raw: &str) -> Vec<ToolCall> {
        crate::tools::minimax::parse_calls(raw)
    }
    fn tool_results(&self, results: &[ToolResult], think: bool) -> String {
        // The reply ended on `[e~[` (sampled, not fed); the results are one tool turn.
        let mut s = format!("{}\n]~b]tool", Self::EOT);
        for r in results {
            s.push_str(&format!("\n<response>{}</response>", r.content));
        }
        s.push_str(Self::EOT);
        s.push('\n');
        s.push_str(Self::ai(think));
        s
    }
    fn assistant_text(&self, content: &str, calls: &[ToolCall]) -> String {
        if calls.is_empty() {
            return content.to_string();
        }
        // As the template: the text as is, then a newline and the calls block.
        format!("{content}{}", crate::tools::minimax::calls(calls))
    }
}

/// GLM-5.3 and GLM-5.3-Flash (zai-org; Nex-N2.6 has the same template): `[gMASK]<sop>`, a system
/// turn with the reasoning effort (the template's default, Max), the tools in a system turn of
/// their own, then `<|system|>` / `<|user|>` / `<|assistant|>` / `<|observation|>` turns with
/// nothing between them. A reply ends on the next role token (`<|user|>`, or `<|observation|>`
/// after tool calls). The template always opens the reply with `<think>`; without thinking the
/// block is closed at once, as the template renders past replies that have no reasoning.
pub struct GlmChat;

impl GlmChat {
    fn assistant(think: bool) -> &'static str {
        if think {
            "<|assistant|><think>"
        } else {
            "<|assistant|><think></think>"
        }
    }

    /// `eos` plus the role tokens a reply can end on.
    pub fn stop_ids(tok: &crate::tokenizer::Tokenizer, mut eos: Vec<u32>) -> Vec<u32> {
        for t in ["<|user|>", "<|observation|>", "<|endoftext|>"] {
            if let Some(id) = tok.token_to_id(t)
                && !eos.contains(&id)
            {
                eos.push(id);
            }
        }
        eos
    }
}

impl ChatFormat for GlmChat {
    fn conversation_start(&self, system: Option<&str>, think: bool) -> String {
        self.conversation_start_tools(system, think, &[])
    }
    fn user_turn(&self, content: &str, think: bool) -> String {
        format!("<|user|>{content}{}", Self::assistant(think))
    }
    fn end_of_model_turn(&self) -> String {
        String::new()
    }
    fn end_of_turn_token(&self) -> &'static str {
        "<|user|>"
    }
    fn think_tokens(&self) -> (&'static str, &'static str) {
        ("<think>", "</think>")
    }
    fn starts_in_thinking(&self, think: bool) -> bool {
        think
    }
    fn conversation_start_tools(&self, system: Option<&str>, _think: bool, tools: &[ToolSpec]) -> String {
        let mut s = String::from("[gMASK]<sop><|system|>Reasoning Effort: Max");
        if !tools.is_empty() {
            s.push_str(&crate::tools::glm::tools_block(tools));
        }
        if let Some(sys) = system.filter(|s| !s.is_empty()) {
            s.push_str("<|system|>");
            s.push_str(sys);
        }
        s
    }
    fn tool_call_marker(&self) -> &'static str {
        crate::tools::glm::MARKER
    }
    fn parse_tool_calls(&self, raw: &str) -> Vec<ToolCall> {
        crate::tools::glm::parse_calls(raw)
    }
    fn parse_tool_calls_for(&self, raw: &str, tools: &[ToolSpec]) -> Vec<ToolCall> {
        crate::tools::glm::parse_calls_for(raw, tools)
    }
    fn tool_results(&self, results: &[ToolResult], think: bool) -> String {
        let mut s = String::from("<|observation|>");
        for r in results {
            s.push_str(&crate::tools::glm::response(r));
        }
        s.push_str(Self::assistant(think));
        s
    }
    fn assistant_text(&self, content: &str, calls: &[ToolCall]) -> String {
        format!("{content}{}", crate::tools::glm::calls(calls))
    }
}

/// `eos` plus the chat format's end-of-turn token, looked up in the tokenizer.
pub fn with_turn_end(fmt: &dyn ChatFormat, tok: &crate::tokenizer::Tokenizer, mut eos: Vec<u32>) -> Vec<u32> {
    if let Some(id) = tok.token_to_id(fmt.end_of_turn_token())
        && !eos.contains(&id)
    {
        eos.push(id);
    }
    eos
}

#[cfg(test)]
mod tests {
    use super::*;

    const M27: &str = "You are a helpful assistant. Your name is MiniMax-M2.7 and is built by MiniMax.";

    fn m27() -> MiniMaxChat {
        MiniMaxChat::for_template(Some(&format!("{{%- set model_identity = \"{M27}\" -%}}")))
    }

    /// The renderings of MiniMax-M2.7's own chat template (transformers' apply_chat_template).
    #[test]
    fn minimax_matches_template() {
        let f = m27();
        assert_eq!(f.identity, M27);
        assert_eq!(
            f.first_turn(None, true, &[], "Hi"),
            format!("]~!b[]~b]system\n{M27}[e~[\n]~b]user\nHi[e~[\n]~b]ai\n<think>\n")
        );
        assert_eq!(
            f.first_turn(Some("Be brief."), true, &[], "Hi"),
            "]~!b[]~b]system\nBe brief.[e~[\n]~b]user\nHi[e~[\n]~b]ai\n<think>\n"
        );
        let tool = ToolSpec {
            name: "get_time".into(),
            description: "Current time".into(),
            parameters: serde_json::json!({"type": "object", "properties": {"tz": {"type": "string"}}, "required": ["tz"]}),
        };
        let start = format!(
            "]~!b[]~b]system\n{M27}\n\n# Tools\nYou may call one or more tools to assist with the user query.\nHere are the tools available in JSONSchema format:\n\n<tools>\n<tool>{{\"name\": \"get_time\", \"description\": \"Current time\", \"parameters\": {{\"type\": \"object\", \"properties\": {{\"tz\": {{\"type\": \"string\"}}}}, \"required\": [\"tz\"]}}}}</tool>\n</tools>\n\nWhen making tool calls, use XML format to invoke tools and pass parameters:\n\n<minimax:tool_call>\n<invoke name=\"tool-name-1\">\n<parameter name=\"param-key-1\">param-value-1</parameter>\n<parameter name=\"param-key-2\">param-value-2</parameter>\n...\n</invoke>\n</minimax:tool_call>[e~[\n]~b]user\nTime?[e~[\n]~b]ai\n<think>\n"
        );
        assert_eq!(f.first_turn(None, true, std::slice::from_ref(&tool), "Time?"), start);
        // A call as the template renders it, its parse, and the results turn.
        let reply = "need tool\n</think>\n\n\n<minimax:tool_call>\n<invoke name=\"get_time\">\n<parameter name=\"tz\">UTC</parameter>\n<parameter name=\"n\">3</parameter>\n</invoke>\n</minimax:tool_call>";
        let calls = f.parse_tool_calls(reply);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_time");
        assert_eq!(calls[0].arguments["tz"], "UTC");
        assert_eq!(calls[0].arguments["n"], 3);
        let results = [ToolResult {
            name: "get_time".into(),
            content: "12:00".into(),
        }];
        assert_eq!(
            f.tool_results(&results, true),
            "[e~[\n]~b]tool\n<response>12:00</response>[e~[\n]~b]ai\n<think>\n"
        );
        assert_eq!(f.assistant_text("need tool\n</think>\n\n", &calls), reply);
    }

    /// MiniMax-M3.1-Flash's template, as transformers renders it (adaptive thinking).
    #[test]
    fn minimax3_matches_template() {
        let id = "Your model version is MiniMax-M3.1, developed by MiniMax. Knowledge cutoff: January 2026. Founded in early 2022, MiniMax is a global AI foundation model company committed to advancing the frontiers of AI towards AGI.";
        let f = MiniMax3Chat::for_template(&format!("{{{{- '{id}' }}}} <mm:think>"));
        assert!(!f.modes);
        let sys = format!(
            "]~!b[]~b]system\n{id}\n\n<thinking_instructions>\nYou have a thinking capability that allows you to reason step by step before responding. When thinking is enabled, wrap your reasoning in <mm:think></mm:think> tags before your response. When thinking is disabled, begin your response directly after the </mm:think> prefix. When thinking is adaptive, decide on your own whether to think for the current turn.\nCurrent thinking mode: adaptive. You are encouraged to think for complex decision-making, multi-step reasoning, or when analyzing function/tool results.\n</thinking_instructions>\n\n<model_id>8bb8149fe</model_id>[e~[\n]~b]developer\nYou are a helpful assistant."
        );
        assert_eq!(
            f.first_turn(None, true, &[], "Hi"),
            format!("{sys}[e~[\n]~b]user\nHi[e~[\n]~b]ai\n")
        );
        let tool = ToolSpec {
            name: "get_time".into(),
            description: "Current time".into(),
            parameters: serde_json::json!({"type": "object", "properties": {"tz": {"type": "string"}}, "required": ["tz"]}),
        };
        let ns = "]<]minimax[>[";
        let tools = format!(
            "\n\n# Tools\nYou may call one or more tools to assist with the user query.\nHere are the tools available in JSONSchema format:\n\n<tools>\n<tool>{{\"name\": \"get_time\", \"description\": \"Current time\", \"parameters\": {{\"type\": \"object\", \"properties\": {{\"tz\": {{\"type\": \"string\"}}}}, \"required\": [\"tz\"]}}}}</tool>\n</tools>\n\nTo call tools, wrap all invocations in a single {ns}<tool_call>{ns}</tool_call> block. Parameter values containing nested objects or arrays are recursively expanded into XML elements. Example:\n\n{ns}<tool_call>\n{ns}<invoke name=\"tool-name-1\">{ns}<param-1>value-1{ns}</param-1>{ns}<param-2>{ns}<item>{ns}<key-a>val-a{ns}</key-a>{ns}<key-b>val-b{ns}</key-b>{ns}</item>{ns}</param-2>{ns}</invoke>\n{ns}<invoke name=\"tool-name-2\">{ns}<param-1>value-1{ns}</param-1>{ns}</invoke>\n{ns}</tool_call>"
        );
        assert_eq!(
            f.first_turn(None, true, std::slice::from_ref(&tool), "Time?"),
            format!("{sys}{tools}[e~[\n]~b]user\nTime?[e~[\n]~b]ai\n")
        );
        let reply = format!(
            "<mm:think>r</mm:think>{ns}<tool_call>\n{ns}<invoke name=\"get_time\">{ns}<tz>UTC{ns}</tz>{ns}</invoke>\n{ns}</tool_call>"
        );
        let calls = f.parse_tool_calls(&reply);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["tz"], "UTC");
        assert_eq!(f.assistant_text("<mm:think>r</mm:think>", &calls), reply);
        let results = [ToolResult {
            name: "get_time".into(),
            content: "12:00".into(),
        }];
        assert_eq!(
            f.tool_results(&results, true),
            "[e~[\n]~b]tool\n<response>12:00</response>[e~[\n]~b]ai\n"
        );
        // Nested arguments round-trip.
        let mut args = serde_json::Map::new();
        args.insert("q".into(), "a < b".into());
        args.insert("n".into(), 3.into());
        args.insert("flag".into(), true.into());
        args.insert("list".into(), serde_json::json!([1, "x", {"k": "v"}]));
        args.insert("obj".into(), serde_json::json!({"a": {"b": 2.5}}));
        let call = ToolCall {
            name: "f".into(),
            arguments: args,
        };
        let t = f.assistant_text("", std::slice::from_ref(&call));
        assert_eq!(f.parse_tool_calls(&t), vec![call]);
    }

    /// MiniMax-M1's and Text-01's templates (rendered with Jinja).
    #[test]
    fn minimax01_matches_templates() {
        let tool = ToolSpec {
            name: "get_time".into(),
            description: "Current time".into(),
            parameters: serde_json::json!({"type": "object", "properties": {"tz": {"type": "string"}}}),
        };
        let decl = r#"{"type": "function", "function": {"name": "get_time", "description": "Current time", "parameters": {"type": "object", "properties": {"tz": {"type": "string"}}}}}"#;
        let m1 = MiniMax01Chat::for_template("{{ '<begin_of_document>' }}<beginning_of_sentence>");
        assert!(m1.m1);
        assert_eq!(
            m1.first_turn(Some("Be brief."), false, std::slice::from_ref(&tool), "Hi"),
            format!(
                "<begin_of_document><beginning_of_sentence>system ai_setting=assistant\nBe brief.<end_of_sentence>\n<beginning_of_sentence>system tool_setting=tools\nYou are provided with these tools:\n<tools>\n{decl}\n</tools>\n\nIf you need to call tools, please respond with <tool_calls></tool_calls> XML tags, and provide tool-name and json-object of arguments, following the format below:\n<tool_calls>\n{{name: <tool-name-1>, arguments: <args-json-object-1>}}\n...\n</tool_calls><end_of_sentence>\n<beginning_of_sentence>user name=user\nHi<end_of_sentence>\n<beginning_of_sentence>ai name=assistant\n"
            )
        );
        let t01 = MiniMax01Chat::for_template("<beginning_of_sentence>");
        assert!(!t01.m1);
        assert_eq!(
            t01.first_turn(Some("Be brief."), false, std::slice::from_ref(&tool), "Hi"),
            format!(
                "<beginning_of_sentence>system ai_setting=assistant\nBe brief.<end_of_sentence>\n<beginning_of_sentence>user name=user\nHi<end_of_sentence>\n<beginning_of_sentence>system function_setting=functions\n{decl}<end_of_sentence>\n<beginning_of_sentence>ai name=assistant\n"
            )
        );
        // Calls as each writes them.
        let m1_out = "<think>\nok\n</think>\n<tool_calls>\n{\"name\": \"get_time\", \"arguments\": {\"tz\": \"UTC\"}}\n</tool_calls>";
        assert_eq!(m1.parse_tool_calls(m1_out)[0].arguments["tz"], "UTC");
        let t01_out = "<function_call>```typescript\nfunctions.get_time({\"tz\": \"UTC\"})\n```";
        let c = t01.parse_tool_calls(t01_out);
        assert_eq!((c[0].name.as_str(), &c[0].arguments["tz"]), ("get_time", &serde_json::json!("UTC")));
        for f in [&m1, &t01] {
            let t = f.assistant_text("Let me check.", &c);
            assert_eq!(f.parse_tool_calls(&t), c, "{t}");
        }
    }

    /// The renderings of DeepSeek V4.1's own encoder (`encoding/encoding.py` of the checkpoint,
    /// `encode_messages`): chat and thinking mode, tools with and without a system prompt, a
    /// round of tool calling.
    #[test]
    fn deepseek41_matches_encoder() {
        let f = DeepSeek41Chat;
        const EFFORT: &str = "Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n";
        assert_eq!(
            f.first_turn(Some("Be brief."), false, &[], "Hi"),
            "<｜begin▁of▁sentence｜><｜System｜>Be brief.<｜User｜>Hi<｜Assistant｜></think>"
        );
        assert_eq!(
            f.first_turn(None, false, &[], "Hi"),
            "<｜begin▁of▁sentence｜><｜User｜>Hi<｜Assistant｜></think>"
        );
        assert_eq!(
            f.first_turn(None, true, &[], "Hi"),
            format!("<｜begin▁of▁sentence｜><｜System｜>{EFFORT}<｜User｜>Hi<｜Assistant｜><think>")
        );
        assert_eq!(
            f.first_turn(Some("Be brief."), true, &[], "Hi"),
            format!("<｜begin▁of▁sentence｜><｜System｜>{EFFORT}Be brief.<｜User｜>Hi<｜Assistant｜><think>")
        );
        let tool = ToolSpec {
            name: "get_weather".into(),
            description: "Weather of a city.".into(),
            parameters: serde_json::json!({"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}),
        };
        let tools = "## Tools\n\nYou have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<｜DSML｜ calls>\" block like the following:\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"$TOOL_NAME\">\n<｜DSML｜ parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜ parameter>\n...\n</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"$TOOL_NAME2\">\n...\n</｜DSML｜ invoke>\n</｜DSML｜ calls>\n\nString parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\nIf thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\nOtherwise, output directly after </think> with tool calls or final response.\n\n### Available Tool Schemas\n\n{\"name\": \"get_weather\", \"description\": \"Weather of a city.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}, \"days\": {\"type\": \"integer\"}}, \"required\": [\"city\"]}}\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n";
        // Without a system prompt the tools are declared in an empty system message.
        assert_eq!(
            f.first_turn(None, true, std::slice::from_ref(&tool), "Paris?"),
            format!("<｜begin▁of▁sentence｜><｜System｜>{EFFORT}\n\n{tools}<｜User｜>Paris?<｜Assistant｜><think>")
        );
        let call = ToolCall {
            name: "get_weather".into(),
            arguments: serde_json::json!({"city": "Paris", "days": 3}).as_object().unwrap().clone(),
        };
        let result = ToolResult {
            name: "get_weather".into(),
            content: "sunny".into(),
        };
        let mut s = f.first_turn(Some("Be brief."), false, std::slice::from_ref(&tool), "Paris?");
        let reply = f.assistant_text("Let me check.", std::slice::from_ref(&call));
        assert_eq!(f.parse_tool_calls(&reply), vec![call]);
        s.push_str(&reply);
        s.push_str(&f.tool_results(&[result], false));
        s.push_str("Sunny.");
        s.push_str(&f.end_of_model_turn());
        s.push_str(&f.user_turn("Thanks", false));
        assert_eq!(
            s,
            format!(
                "<｜begin▁of▁sentence｜><｜System｜>Be brief.\n\n{tools}<｜User｜>Paris?<｜Assistant｜></think>Let me check.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n<｜DSML｜ parameter name=\"city\" string=\"true\">Paris</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"days\" string=\"false\">3</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls><｜end▁of▁sentence｜><｜User｜><tool_result>sunny</tool_result><｜Assistant｜></think>Sunny.<｜end▁of▁sentence｜><｜User｜>Thanks<｜Assistant｜></think>"
            )
        );
    /// The renderings of GLM-5.3-Flash's chat template (transformers' jinja environment; the
    /// templates of GLM-5.3 and Nex-N2.6 give the same text on these conversations).
    #[test]
    fn glm_matches_template() {
        let f = GlmChat;
        assert_eq!(
            f.first_turn(None, true, &[], "Hi"),
            "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>Hi<|assistant|><think>"
        );
        assert_eq!(
            f.first_turn(Some("Be brief."), true, &[], "Hi"),
            "[gMASK]<sop><|system|>Reasoning Effort: Max<|system|>Be brief.<|user|>Hi<|assistant|><think>"
        );
        let tools = [
            ToolSpec {
                name: "get_time".into(),
                description: "Current time in a timezone.".into(),
                parameters: serde_json::json!({"type": "object", "properties": {"tz": {"type": "string", "description": "IANA name"}, "n": {"type": "integer"}}, "required": ["tz"]}),
            },
            ToolSpec {
                name: "calc".into(),
                description: "Evaluate.".into(),
                parameters: serde_json::json!({"type": "object", "properties": {"expr": {"type": "string"}}}),
            },
        ];
        let block = concat!(
            "<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n",
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n",
            r#"{"name": "get_time", "description": "Current time in a timezone.", "parameters": {"type": "object", "properties": {"tz": {"type": "string", "description": "IANA name"}, "n": {"type": "integer"}}, "required": ["tz"]}}"#,
            "\n",
            r#"{"name": "calc", "description": "Evaluate.", "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}}}"#,
            "\n</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n",
            "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>"
        );
        assert_eq!(
            f.first_turn(Some("Be brief."), true, &tools, "Time in Tokyo?"),
            format!("[gMASK]<sop><|system|>Reasoning Effort: Max{block}<|system|>Be brief.<|user|>Time in Tokyo?<|assistant|><think>")
        );
        // A reply with two calls as the template renders it, its parse, and the results turn.
        let reply = "need tool</think>Let me check.<tool_call>get_time<arg_key>tz</arg_key><arg_value>Asia/Tokyo</arg_value><arg_key>n</arg_key><arg_value>3</arg_value></tool_call><tool_call>calc<arg_key>expr</arg_key><arg_value>1+1</arg_value></tool_call>";
        let calls = f.parse_tool_calls_for(reply, &tools);
        assert_eq!(calls.len(), 2);
        assert_eq!((calls[0].name.as_str(), calls[1].name.as_str()), ("get_time", "calc"));
        assert_eq!(calls[0].arguments["tz"], "Asia/Tokyo");
        assert_eq!(calls[0].arguments["n"], 3);
        assert_eq!(calls[1].arguments["expr"], "1+1");
        assert_eq!(f.assistant_text("need tool</think>Let me check.", &calls), reply);
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
        assert_eq!(
            f.tool_results(&results, true),
            "<|observation|><tool_response>12:00</tool_response><tool_response>2</tool_response><|assistant|><think>"
        );
        // A string parameter keeps what the model wrote, also when it reads as JSON.
        let c = f.parse_tool_calls_for(
            "<tool_call>calc<arg_key>expr</arg_key><arg_value>42</arg_value></tool_call>",
            &tools,
        );
        assert_eq!(c[0].arguments["expr"], "42");
        assert_eq!(
            f.parse_tool_calls("<tool_call>calc<arg_key>expr</arg_key><arg_value>42</arg_value></tool_call>")[0].arguments["expr"],
            42
        );
        // A second turn (the reply ended on `<|user|>`, which the next turn writes), and a reply
        // without thinking.
        let two = format!(
            "{}greet</think>Hello!{}{}",
            f.first_turn(None, true, &[], "Hi"),
            f.end_of_model_turn(),
            f.user_turn("Bye", true)
        );
        assert_eq!(
            two,
            "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>Hi<|assistant|><think>greet</think>Hello!<|user|>Bye<|assistant|><think>"
        );
        assert_eq!(f.user_turn("Hi", false), "<|user|>Hi<|assistant|><think></think>");
    }
}
