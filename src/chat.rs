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
    /// What to feed after a reply that ended on its tool calls (its stop token was sampled but not
    /// fed): the results, up to where the model continues.
    fn tool_results(&self, results: &[ToolResult], think: bool) -> String;
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
    fn tool_results(&self, results: &[ToolResult], think: bool) -> String {
        // The model stopped on `<|tool_response>`; the responses continue the same model turn.
        let mut s: String = results.iter().map(crate::tools::gemma::response).collect();
        if think {
            s.push_str("<|channel>thought\n");
        }
        s
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

/// `eos` plus the chat format's end-of-turn token, looked up in the tokenizer.
pub fn with_turn_end(fmt: &dyn ChatFormat, tok: &tokenizers::Tokenizer, mut eos: Vec<u32>) -> Vec<u32> {
    if let Some(id) = tok.token_to_id(fmt.end_of_turn_token())
        && !eos.contains(&id)
    {
        eos.push(id);
    }
    eos
}
