//! [`Engine::chat`]: a whole conversation (system, user, assistant and tool messages) rendered in
//! the model's chat format and run, as stateless chat APIs (OpenAI's chat completions) send it.
//! When the conversation extends the one of the previous call, its tokens are already in the
//! cache and only the new ones are fed.

use super::*;

/// Who wrote a [`Message`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Role {
    /// Instructions; all system messages are joined into the system prompt.
    System,
    #[default]
    User,
    /// A past reply of the model (its text and tool calls).
    Assistant,
    /// The result of a tool call of the assistant message before it.
    Tool,
}

/// One message of a conversation for [`Engine::chat`].
#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct Message {
    pub role: Role,
    pub content: String,
    /// User messages: attachments placed before the text (see [`Engine::send_with_media`]).
    pub media: Attachments,
    /// Assistant messages: the calls the reply ended on.
    pub tool_calls: Vec<ToolCall>,
    /// Tool messages: the name of the tool whose result this is.
    pub name: Option<String>,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            ..Self::default()
        }
    }
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(Role::System, content)
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(Role::User, content)
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(Role::Assistant, content)
    }
    /// The result of the call of tool `name`.
    pub fn tool(name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            name: Some(name.into()),
            ..Self::new(Role::Tool, content)
        }
    }
    pub fn with_media(mut self, media: Attachments) -> Self {
        self.media = media;
        self
    }
    pub fn with_tool_calls(mut self, calls: Vec<ToolCall>) -> Self {
        self.tool_calls = calls;
        self
    }

    /// Parses `{"role": "system" | "user" | "assistant" | "tool", "content": "...", "media": {...}
    /// (the [`Attachments::from_json`] format), "tool_calls": [{"name": ..., "arguments": {...}}],
    /// "name": "..." (a tool message's tool)}` (every key but `role` optional; the C API's
    /// format).
    pub fn from_json(v: &serde_json::Value) -> Result<Self> {
        let o = v.as_object().context("message: expected a JSON object")?;
        let role = match o.get("role").and_then(|r| r.as_str()) {
            Some("system") => Role::System,
            Some("user") => Role::User,
            Some("assistant") => Role::Assistant,
            Some("tool") => Role::Tool,
            r => bail!("message: unknown role {r:?} (system, user, assistant, tool)"),
        };
        let mut m = Message::new(role, o.get("content").and_then(|c| c.as_str()).unwrap_or_default());
        for (k, val) in o {
            match k.as_str() {
                "role" | "content" => {}
                "media" => m.media = Attachments::from_json(val)?,
                "tool_calls" => m.tool_calls = serde_json::from_value(val.clone()).context("message: tool_calls")?,
                "name" => m.name = val.as_str().map(str::to_string),
                _ => bail!("message: unknown key {k:?}"),
            }
        }
        Ok(m)
    }
}

impl Engine {
    /// Runs a whole conversation and generates the next reply, streaming it to `on_delta`. The
    /// conversation replaces the engine's: system messages (joined) are the system prompt
    /// ([`ChatConfig::system`] when there are none), consecutive user messages form one turn,
    /// assistant messages are past replies (their thinking is not part of the conversation) and
    /// tool messages the results of the calls before them. The last message is usually a user
    /// or tool message; after an assistant message the model continues that reply. [`ChatConfig`]
    /// gives thinking, the tools and the reply length.
    ///
    /// When the conversation's tokens start with those the cache holds from the previous call
    /// (the same conversation with the reply and new messages appended, the reply rendered as the
    /// model wrote it), only the new tokens are fed ([`Stats::cached_tokens`]); otherwise the
    /// conversation starts over. Prompts with attachments before the new messages always start
    /// over (the cached soft tokens might be of other media).
    pub fn chat(&mut self, messages: &[Message], on_delta: &mut dyn FnMut(&Delta) -> Flow) -> Result<Reply> {
        let (text, feats) = self.render(messages)?;
        let ids = self.tok.encode(text, false).map_err(|e| anyhow!("encode: {e}"))?.get_ids().to_vec();
        if ids.is_empty() {
            bail!("empty prompt");
        }
        let placeholder = |t: &u32| feats.iter().any(|f| f.token_id == *t);
        let skip = match self.chat_ids.take() {
            Some(c) if c.len() < ids.len() && ids.starts_with(&c) && self.context_len() == c.len() && !c.iter().any(placeholder) => c.len(),
            _ => 0,
        };
        if skip == 0 {
            self.reset();
        }
        // The cache ends where the conversation's tokens are fed from: the token the last reply
        // stopped on, if any, is among them.
        self.unfed = None;
        let think = self.config.think;
        let mut reply = self.turn_ids(ids[skip..].to_vec(), &feats, think)?.finish(self, on_delta)?;
        reply.stats.cached_tokens = skip;
        self.stats.cached_tokens = skip;
        // The cache holds the prompt and the reply up to the tokens the model has not run (the
        // one it stopped on, at least).
        let (mut fed, ctx) = (ids, self.context_len());
        fed.extend_from_slice(&reply.tokens);
        self.chat_ids = (fed.len() - reply.tokens.len() <= ctx && ctx <= fed.len()).then(|| {
            fed.truncate(ctx);
            fed
        });
        self.turns = messages.iter().filter(|m| m.role == Role::User).count().max(1);
        Ok(reply)
    }

    /// The conversation's prompt text and the soft tokens of its attachments, in order.
    fn render(&self, messages: &[Message]) -> Result<(String, Vec<mm::MmFeature>)> {
        let (fmt, c) = (self.fmt, &self.config);
        let think = c.think;
        let sys: Vec<&str> = messages
            .iter()
            .filter(|m| m.role == Role::System)
            .map(|m| m.content.as_str())
            .collect();
        let system = if sys.is_empty() {
            match &c.system {
                SystemPrompt::Default => Some(crate::tools::default_system_prompt(!c.tools.is_empty())),
                SystemPrompt::None => None,
                SystemPrompt::Text(t) => Some(t.clone()),
            }
        } else {
            Some(sys.join("\n\n"))
        };
        let rest: Vec<&Message> = messages.iter().filter(|m| m.role != Role::System).collect();
        let (mut text, mut feats) = (String::new(), Vec::new());
        let mut started = false;
        let mut i = 0;
        while i < rest.len() {
            let m = rest[i];
            if m.role != Role::User && !m.media.is_empty() {
                bail!("attachments go in user messages");
            }
            match m.role {
                Role::User => {
                    let mut content = String::new();
                    while i < rest.len() && rest[i].role == Role::User {
                        let (mm_text, f) = self.prepare_media(&rest[i].media)?;
                        feats.extend(f);
                        if !content.is_empty() {
                            content.push_str("\n\n");
                        }
                        content.push_str(&mm_text);
                        content.push_str(&rest[i].content);
                        i += 1;
                    }
                    if started {
                        // After a reply: its stop token, then the new turn.
                        text.push_str(&fmt.end_of_model_turn());
                        text.push_str(&fmt.user_turn(&content, think));
                    } else {
                        text = fmt.first_turn(system.as_deref(), think, &c.tools, &content);
                        started = true;
                    }
                }
                Role::Assistant => {
                    if !started {
                        bail!("the conversation must start with a user message (after the system messages)");
                    }
                    text.push_str(&fmt.assistant_text(&m.content, &m.tool_calls));
                    i += 1;
                }
                Role::Tool => {
                    let prev = (i > 0).then(|| rest[i - 1]);
                    if !prev.is_some_and(|p| p.role == Role::Assistant && !p.tool_calls.is_empty()) {
                        bail!("tool messages must follow an assistant message with tool calls");
                    }
                    let mut results = Vec::new();
                    while i < rest.len() && rest[i].role == Role::Tool {
                        let r = rest[i];
                        let name = r.name.clone().unwrap_or_else(|| {
                            // Unnamed results answer the calls in order.
                            let calls = &prev.unwrap().tool_calls;
                            calls[results.len().min(calls.len() - 1)].name.clone()
                        });
                        results.push(ToolResult {
                            name,
                            content: r.content.clone(),
                        });
                        i += 1;
                    }
                    let t = fmt.tool_results(&results, think);
                    if t.is_empty() {
                        bail!("this model's chat format has no tool calling");
                    }
                    text.push_str(&t);
                }
                Role::System => unreachable!(),
            }
        }
        if !started {
            bail!("the conversation has no user message");
        }
        Ok((text, feats))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::ToolCall;

    #[test]
    fn assistant_text_round_trips_tool_calls() {
        let mut args = serde_json::Map::new();
        args.insert("city".into(), "Paris".into());
        args.insert("days".into(), 3.into());
        let call = ToolCall {
            name: "weather".into(),
            arguments: args,
        };
        let formats: [&dyn ChatFormat; 4] = [&chat::GemmaChat, &chat::QWEN, &chat::QWEN3, &chat::MistralChat];
        for f in formats {
            let t = f.assistant_text("Let me check.", std::slice::from_ref(&call));
            assert_eq!(f.parse_tool_calls(&t), vec![call.clone()], "{t}");
        }
        let t = chat::DeepSeekChat.assistant_text("", std::slice::from_ref(&call));
        assert_eq!(chat::DeepSeekChat.parse_tool_calls(&t), vec![call]);
        assert_eq!(chat::QWEN.assistant_text("plain", &[]), "plain");
    }
}
