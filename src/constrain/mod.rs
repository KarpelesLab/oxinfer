//! Constrained decoding: the answer of a reply is kept inside a grammar by masking, at every
//! step, the logits of the tokens that would leave it.
//!
//! * [`ResponseFormat`]: any JSON object, or a JSON Schema ([`schema`] lists the subset).
//! * [`ToolChoice`]: a forced tool call, in the model family's own call syntax ([`ToolSyntax`])
//!   with the tool's parameter schema.
//!
//! The grammar ([`grammar`]) is over bytes, so a token that spans several grammar symbols
//! (`","`, `"}]`) is followed through all of them, and a multi-byte character split across
//! tokens is tracked byte by byte: only tokens that keep the text valid UTF-8 inside the
//! grammar pass. Special tokens (anything in the tokenizer's added tokens) have no text for the
//! grammar: they pass only where the grammar names them (the call markers of a tool call), and
//! the stop tokens only once the document is complete. A thinking block is not constrained
//! (but cannot end the reply): the grammar starts after it closes.
//!
//! [`Constraint`] is the [`TokenFilter`] the [`crate::sampler::Sampler`] applies, so every
//! decoding path that samples through it (plain, prompt lookup, MTP and DSpark rounds, on the
//! CPU or the GPU) is constrained: speculative drafts are drawn under the mask of a forked
//! state, and verification masks the target's distribution at every position.
//!
//! What a token may do is found by walking the vocabulary as a byte trie. Per grammar position
//! the walk is done once: it gives the tokens accepted whatever lies under that position, and
//! the few that run past it (a string's closing quote and what follows), which are checked
//! against the actual stack. Masks are then cached per matcher state, so the tokens of a long
//! string cost one lookup each.

pub mod grammar;
pub mod schema;

use crate::chat::ChatFormat;
use crate::sampler::TokenFilter;
use crate::tokenizer::Tokenizer;
use crate::tools::ToolSpec;
use anyhow::{Context, Result, bail};
use grammar::{DONE, Grammar, Matcher, Pos, RuleId, Sym};
use schema::{Compiler, Dialect, Prims};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// What the answer of a reply must be ([`crate::engine::ChatConfig::response_format`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub enum ResponseFormat {
    /// Anything.
    #[default]
    Text,
    /// A JSON object.
    JsonObject,
    /// A JSON value of this schema (see [`schema`] for the supported subset).
    JsonSchema(Value),
}

impl ResponseFormat {
    /// Parses OpenAI's `response_format`: `{"type": "text"}`, `{"type": "json_object"}` or
    /// `{"type": "json_schema", "json_schema": {"schema": {...}}}` (a bare schema under `schema`
    /// works too). The schema is checked.
    pub fn from_json(v: &Value) -> Result<Self> {
        let f = match v.get("type").and_then(Value::as_str) {
            Some("text") => Self::Text,
            Some("json_object") => Self::JsonObject,
            Some("json_schema") => {
                let s = v
                    .pointer("/json_schema/schema")
                    .or_else(|| v.get("schema"))
                    .context("response_format: json_schema needs json_schema.schema")?;
                Self::JsonSchema(s.clone())
            }
            t => bail!("response_format: unknown type {t:?} (text, json_object, json_schema)"),
        };
        f.check()?;
        Ok(f)
    }

    /// Whether the schema is in the supported subset.
    pub fn check(&self) -> Result<()> {
        match self {
            Self::JsonSchema(s) => schema::check(s).context("response_format schema"),
            _ => Ok(()),
        }
    }
}

/// Whether a reply must call a tool ([`crate::engine::ChatConfig::tool_choice`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub enum ToolChoice {
    /// The model decides.
    #[default]
    Auto,
    /// The reply is a call of one of the tools.
    Required,
    /// The reply is a call of this tool.
    Function(String),
}

/// How a model family writes a tool call ([`ChatFormat::tool_syntax`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolSyntax {
    /// Qwen3: `<tool_call>\n{"name": NAME, "arguments": {...}}\n</tool_call>`.
    QwenJson,
    /// Qwen 3.5 / 3.8: `<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n
    /// </function>\n</tool_call>` (strings bare, other values as JSON).
    QwenXml,
    /// Gemma 4: `<|tool_call>call:NAME{key:<|"|>text<|"|>,n:1}<tool_call|>`.
    Gemma,
    /// Mistral: `[TOOL_CALLS]NAME[ARGS]{...}`.
    Mistral,
    /// DeepSeek V4: a `<｜DSML｜tool_calls>` block.
    Dsml,
}

// ---- the vocabulary as bytes ----------------------------------------------------------------------

#[derive(Clone, Copy, Default)]
struct TrieNode {
    edges: (u32, u32),
    ends: (u32, u32),
}

/// The bytes of every token and the trie of those the grammar reads as text.
pub struct Vocab {
    bytes: Vec<Vec<u8>>,
    plain: Vec<bool>,
    /// Added tokens by decreasing length of their text.
    added: Vec<(String, u32)>,
    nodes: Vec<TrieNode>,
    edges: Vec<(u8, u32)>,
    ends: Vec<u32>,
}

/// GPT-2's byte-level alphabet: the byte a token character stands for.
fn byte_level() -> HashMap<char, u8> {
    let mut m = HashMap::new();
    let mut n = 0u32;
    for b in 0..=255u8 {
        let own = matches!(b, 0x21..=0x7e | 0xa1..=0xac | 0xae..=0xff);
        let c = if own {
            b as u32
        } else {
            n += 1;
            255 + n
        };
        m.insert(char::from_u32(c).unwrap(), b);
    }
    m
}

impl Vocab {
    /// The tokens of `tok`: what each contributes to decoded text, found by decoding it after
    /// another token (so that no start-of-text rule applies). A token that is part of a
    /// multi-byte character decodes to U+FFFD: its bytes then come from the byte-level
    /// alphabet or the `<0xNN>` form, checked against that decoding.
    pub fn new(tok: &Tokenizer) -> Self {
        let added = tok.get_added_tokens_decoder();
        let vocab = tok.get_vocab(true);
        let n = vocab.values().copied().max().map_or(0, |m| m as usize + 1);
        let mut tokens: Vec<(Vec<u8>, bool)> = vec![(Vec::new(), false); n];
        let decode = |ids: &[u32]| tok.decode(ids, false).unwrap_or_default();
        let anchor = ["a", "A", "x", "0"]
            .iter()
            .filter_map(|t| tok.token_to_id(t))
            .find(|id| !added.contains_key(id) && !decode(&[*id]).is_empty());
        let base = anchor.map(|a| decode(&[a])).unwrap_or_default();
        let alphabet = byte_level();
        for (text, &id) in &vocab {
            if let Some(a) = added.get(&id) {
                tokens[id as usize] = (a.content.clone().into_bytes(), false);
                continue;
            }
            let Some(anchor) = anchor else { continue };
            let both = decode(&[anchor, id]);
            let Some(rest) = both.strip_prefix(base.as_str()) else { continue };
            if rest.is_empty() {
                continue;
            }
            if !rest.contains('\u{fffd}') || text.contains('\u{fffd}') {
                tokens[id as usize] = (rest.as_bytes().to_vec(), true);
                continue;
            }
            let raw: Option<Vec<u8>> = if text.len() == 6 && text.starts_with("<0x") && text.ends_with('>') {
                u8::from_str_radix(&text[3..5], 16).ok().map(|b| vec![b])
            } else {
                text.chars().map(|c| alphabet.get(&c).copied()).collect()
            };
            tokens[id as usize] = match raw {
                Some(raw) if String::from_utf8_lossy(&raw) == rest => (raw, true),
                _ => (rest.as_bytes().to_vec(), false),
            };
        }
        Self::from_tokens(tokens, added.into_iter().map(|(id, a)| (a.content, id)).collect())
    }

    /// A vocabulary from each token's bytes and whether the grammar reads it as text, and the
    /// added (special) tokens' texts.
    pub fn from_tokens(tokens: Vec<(Vec<u8>, bool)>, mut added: Vec<(String, u32)>) -> Self {
        added.retain(|a| !a.0.is_empty());
        added.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));
        let plain: Vec<bool> = tokens.iter().map(|(b, p)| *p && !b.is_empty()).collect();
        let bytes: Vec<Vec<u8>> = tokens.into_iter().map(|t| t.0).collect();
        let mut items: Vec<(&[u8], u32)> = bytes
            .iter()
            .enumerate()
            .filter(|(i, _)| plain[*i])
            .map(|(i, b)| (b.as_slice(), i as u32))
            .collect();
        items.sort_unstable();
        let (mut nodes, mut edges, mut ends) = (Vec::new(), Vec::new(), Vec::new());
        fn build(items: &[(&[u8], u32)], depth: usize, nodes: &mut Vec<TrieNode>, edges: &mut Vec<(u8, u32)>, ends: &mut Vec<u32>) -> u32 {
            let id = nodes.len();
            nodes.push(TrieNode::default());
            let here = items.iter().take_while(|i| i.0.len() == depth).count();
            let e0 = ends.len() as u32;
            ends.extend(items[..here].iter().map(|i| i.1));
            let mut kids = Vec::new();
            let mut i = here;
            while i < items.len() {
                let b = items[i].0[depth];
                let j = i + items[i..].iter().take_while(|x| x.0[depth] == b).count();
                kids.push((b, build(&items[i..j], depth + 1, nodes, edges, ends)));
                i = j;
            }
            let s = edges.len() as u32;
            edges.extend(kids);
            nodes[id] = TrieNode {
                edges: (s, edges.len() as u32),
                ends: (e0, e0 + here as u32),
            };
            id as u32
        }
        build(&items, 0, &mut nodes, &mut edges, &mut ends);
        Self {
            bytes,
            plain,
            added,
            nodes,
            edges,
            ends,
        }
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// The bytes token `id` adds to the text (a special token's own text; empty if unknown).
    pub fn bytes(&self, id: u32) -> &[u8] {
        self.bytes.get(id as usize).map_or(&[], |b| b.as_slice())
    }

    /// Whether the grammar reads token `id` as text.
    pub fn plain(&self, id: u32) -> bool {
        self.plain.get(id as usize).copied().unwrap_or(false)
    }

    /// `text` as grammar symbols: special tokens where it names them, bytes elsewhere.
    fn text(&self, g: &mut Grammar, text: &str) -> Vec<Sym> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            if let Some((t, id)) = self.added.iter().find(|a| rest.starts_with(a.0.as_str())) {
                out.push(Sym::Token(*id));
                rest = &rest[t.len()..];
            } else {
                let n = rest.chars().next().map_or(1, char::len_utf8);
                out.extend(g.lit(&rest.as_bytes()[..n]));
                rest = &rest[n..];
            }
        }
        out
    }
}

// ---- a compiled constraint --------------------------------------------------------------------------

/// What one grammar position accepts whatever is under it on the stack.
struct PosMask {
    /// Tokens that stay inside the position's rule (or end exactly with it).
    accept: Vec<u64>,
    /// Tokens that run past the end of the rule: they depend on the rest of the stack.
    past: Vec<u32>,
}

/// The tokens a matcher state allows.
struct StateMask {
    bits: Vec<u64>,
    /// The document is complete here: a stop token may follow.
    complete: bool,
}

struct Inner {
    m: Matcher,
    positions: HashMap<Pos, Arc<PosMask>>,
    states: HashMap<Vec<u32>, Arc<StateMask>>,
}

/// A grammar with the caches of its token masks for one vocabulary.
pub struct Compiled {
    g: Grammar,
    root: RuleId,
    /// The root after a thinking block: up to two newlines, then the root.
    root_after_thinking: RuleId,
    vocab: Arc<Vocab>,
    inner: Mutex<Inner>,
}

fn set(bits: &mut [u64], i: u32) {
    bits[i as usize >> 6] |= 1 << (i & 63);
}

fn get(bits: &[u64], i: usize) -> bool {
    bits.get(i >> 6).is_some_and(|w| (w >> (i & 63)) & 1 == 1)
}

impl Compiled {
    fn new(g: Grammar, root: RuleId, root_after_thinking: RuleId, vocab: Arc<Vocab>) -> Self {
        Self {
            g,
            root,
            root_after_thinking,
            vocab,
            inner: Mutex::new(Inner {
                m: Matcher::new(),
                positions: HashMap::new(),
                states: HashMap::new(),
            }),
        }
    }

    fn inner(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn start(&self, after_thinking: bool) -> Vec<u32> {
        let rule = if after_thinking { self.root_after_thinking } else { self.root };
        self.inner().m.start(&self.g, rule, DONE)
    }

    /// The state after `token` (empty when the token is not allowed).
    fn advance(&self, state: &[u32], token: u32) -> Vec<u32> {
        let mut inner = self.inner();
        let m = &mut inner.m;
        if !self.vocab.plain(token) {
            let mut next = Vec::new();
            m.step_token(&self.g, state, token, &mut next);
            return next;
        }
        let mut state = state.to_vec();
        for &b in self.vocab.bytes(token) {
            let mut next = Vec::new();
            m.step_byte(&self.g, &state, b, &mut next);
            state = next;
            if state.is_empty() {
                break;
            }
        }
        state
    }

    /// Walks the trie under `node` with the stacks `state` (over the unknown stack
    /// [`grammar::OUT`]); `past` says a stack already ran out of the rule.
    fn walk(&self, m: &mut Matcher, node: u32, state: &[u32], past: bool, pm: &mut PosMask) {
        let v = &self.vocab;
        let n = v.nodes[node as usize];
        for &(b, child) in &v.edges[n.edges.0 as usize..n.edges.1 as usize] {
            let mut next = Vec::new();
            let past = m.step_byte(&self.g, state, b, &mut next) || past;
            if next.is_empty() && !past {
                continue;
            }
            let c = v.nodes[child as usize];
            for &t in &v.ends[c.ends.0 as usize..c.ends.1 as usize] {
                if !next.is_empty() {
                    set(&mut pm.accept, t);
                } else {
                    pm.past.push(t);
                }
            }
            self.walk(m, child, &next, past, pm);
        }
    }

    fn position(&self, inner: &mut Inner, pos: Pos) -> Arc<PosMask> {
        if let Some(p) = inner.positions.get(&pos) {
            return p.clone();
        }
        let mut pm = PosMask {
            accept: vec![0; self.vocab.len().div_ceil(64)],
            past: Vec::new(),
        };
        let top = inner.m.at(pos, grammar::OUT);
        self.walk(&mut inner.m, 0, &[top], false, &mut pm);
        let pm = Arc::new(pm);
        inner.positions.insert(pos, pm.clone());
        pm
    }

    fn state_mask(&self, state: &[u32]) -> Arc<StateMask> {
        let mut guard = self.inner();
        let inner = &mut *guard;
        if let Some(s) = inner.states.get(state) {
            return s.clone();
        }
        let mut bits = vec![0u64; self.vocab.len().div_ceil(64)];
        let mut complete = false;
        for &s in state {
            match inner.m.top(&self.g, s) {
                None => complete |= s == DONE,
                Some((_, Sym::Token(t))) => {
                    if (t as usize) < self.vocab.len() {
                        set(&mut bits, t);
                    }
                }
                Some((pos, _)) => {
                    let pm = self.position(inner, pos);
                    for (b, a) in bits.iter_mut().zip(&pm.accept) {
                        *b |= a;
                    }
                    for &t in &pm.past {
                        if get(&bits, t as usize) {
                            continue;
                        }
                        let mut st = vec![s];
                        for &b in self.vocab.bytes(t) {
                            let mut next = Vec::new();
                            inner.m.step_byte(&self.g, &st, b, &mut next);
                            st = next;
                            if st.is_empty() {
                                break;
                            }
                        }
                        if !st.is_empty() {
                            set(&mut bits, t);
                        }
                    }
                }
            }
        }
        if inner.states.len() >= 512 {
            inner.states.clear();
        }
        let sm = Arc::new(StateMask { bits, complete });
        inner.states.insert(state.to_vec(), sm.clone());
        sm
    }
}

// ---- building the grammar of a reply ---------------------------------------------------------------

struct Builder<'a> {
    g: Grammar,
    prims: Prims,
    vocab: &'a Vocab,
    bare: Option<RuleId>,
}

static NO_SCHEMA: Value = Value::Null;

impl Builder<'_> {
    fn text(&mut self, s: &str) -> Vec<Sym> {
        self.vocab.text(&mut self.g, s)
    }

    /// Bare text that never holds `</` (so that it cannot hold the closing tag after it):
    /// any characters, newlines included.
    fn bare(&mut self) -> RuleId {
        if let Some(r) = self.bare {
            return r;
        }
        let (text, angle) = (self.g.new_rule(), self.g.new_rule());
        let lt = self.g.byte(b'<');
        let mut c = Compiler::new(&mut self.g, &mut self.prims, &NO_SCHEMA, &Dialect::Json);
        let (plain, after) = (c.char_then(b"<", true, Sym::Rule(text)), c.char_then(b"</", true, Sym::Rule(text)));
        self.g.add_alt(text, Vec::new());
        for a in plain {
            self.g.add_alt(text, a);
        }
        self.g.add_alt(text, vec![lt, Sym::Rule(angle)]);
        self.g.add_alt(angle, vec![lt, Sym::Rule(angle)]);
        for a in after {
            self.g.add_alt(angle, a);
        }
        self.bare = Some(text);
        text
    }

    /// A parameter name where the schema declares none.
    fn name(&mut self) -> Vec<Sym> {
        let first = self.g.class(&[(b'a', b'z'), (b'A', b'Z'), (b'_', b'_')]);
        let more = self.g.class(&[(b'a', b'z'), (b'A', b'Z'), (b'_', b'_'), (b'0', b'9')]);
        let mut k = vec![first];
        k.extend(self.g.repeat(vec![more], 0, Some(63)));
        k
    }

    /// The parameters of `tool` for the syntaxes that write one element per parameter: each
    /// with its bare-string and JSON alternatives and whether it is required; None when the
    /// schema declares no properties (any parameters).
    #[allow(clippy::type_complexity)]
    fn parameters(&mut self, tool: &ToolSpec) -> Result<Option<Vec<(String, Vec<Vec<Sym>>, Vec<Vec<Sym>>, bool)>>> {
        let bare = self.bare();
        let mut c = Compiler::new(&mut self.g, &mut self.prims, &tool.parameters, &Dialect::Json);
        let o = c.object_schema(&tool.parameters, "#")?;
        let Ok(props) = Compiler::properties(o, "#")? else {
            return Ok(None);
        };
        let mut out = Vec::new();
        for (k, s, required) in props {
            let (text, json) = c.split(s, &format!("#/properties/{k}"), bare, 0)?;
            out.push((k.clone(), text, json, required));
        }
        Ok(Some(out))
    }

    /// One call of `tool` in `syntax`.
    fn tool_call(&mut self, syntax: ToolSyntax, tool: &ToolSpec) -> Result<Vec<Sym>> {
        let name = &tool.name;
        let object =
            |b: &mut Self, d: &Dialect| Compiler::new(&mut b.g, &mut b.prims, &tool.parameters, d).object_value(&tool.parameters, "#");
        Ok(match syntax {
            ToolSyntax::QwenJson => {
                let mut s = self.text("<tool_call>\n{\"name\": ");
                s.extend(self.g.lit(Value::String(name.clone()).to_string().as_bytes()));
                s.extend(self.text(", \"arguments\": "));
                s.push(Sym::Rule(object(self, &Dialect::Json)?));
                s.extend(self.text("}\n</tool_call>"));
                s
            }
            ToolSyntax::Mistral => {
                let mut s = self.text(&format!("[TOOL_CALLS]{name}[ARGS]"));
                s.push(Sym::Rule(object(self, &Dialect::Json)?));
                s
            }
            ToolSyntax::Gemma => {
                let quote = self.text("<|\"|>");
                let mut s = self.text(&format!("<|tool_call>call:{name}"));
                s.push(Sym::Rule(object(self, &Dialect::Gemma { quote })?));
                s.extend(self.text("<tool_call|>"));
                s
            }
            ToolSyntax::QwenXml => {
                let mut s = self.text(&format!("<tool_call>\n<function={name}>\n"));
                let close = self.text("\n</parameter>\n");
                let member = |b: &mut Self, key: Vec<Sym>, values: Vec<Vec<Sym>>| {
                    let mut m = b.text("<parameter=");
                    m.extend(key);
                    m.extend(b.text(">\n"));
                    m.push(Sym::Rule(b.g.rule(values)));
                    m.extend(close.clone());
                    m
                };
                match self.parameters(tool)? {
                    Some(params) => {
                        for (k, mut values, json, required) in params {
                            values.extend(json);
                            let key = self.g.lit(k.as_bytes());
                            let m = member(self, key, values);
                            s.extend(if required { m } else { vec![self.g.opt(m)] });
                        }
                    }
                    None => {
                        let (key, bare) = (self.name(), self.bare());
                        let m = member(self, key, vec![vec![Sym::Rule(bare)]]);
                        s.extend(self.g.repeat(m, 0, None));
                    }
                }
                s.extend(self.text("</function>\n</tool_call>"));
                s
            }
            ToolSyntax::Dsml => {
                const D: &str = "｜DSML｜";
                let mut s = self.text(&format!("\n\n<{D}tool_calls>\n<{D}invoke name=\"{name}\">\n"));
                let close = self.text(&format!("</{D}parameter>\n"));
                let member = |b: &mut Self, key: Vec<Sym>, text: Vec<Vec<Sym>>, json: Vec<Vec<Sym>>| {
                    let mut alts = Vec::new();
                    for (values, string) in [(text, "true"), (json, "false")] {
                        if values.is_empty() {
                            continue;
                        }
                        let mut m = b.text(&format!("<{D}parameter name=\""));
                        m.extend(key.clone());
                        m.extend(b.text(&format!("\" string=\"{string}\">")));
                        m.push(Sym::Rule(b.g.rule(values)));
                        m.extend(close.clone());
                        alts.push(m);
                    }
                    vec![Sym::Rule(b.g.rule(alts))]
                };
                match self.parameters(tool)? {
                    Some(params) => {
                        for (k, text, json, required) in params {
                            let key = self.g.lit(k.as_bytes());
                            let m = member(self, key, text, json);
                            s.extend(if required { m } else { vec![self.g.opt(m)] });
                        }
                    }
                    None => {
                        let (key, bare) = (self.name(), self.bare());
                        let any = Compiler::new(&mut self.g, &mut self.prims, &NO_SCHEMA, &Dialect::Json).any();
                        let m = member(self, key, vec![vec![Sym::Rule(bare)]], vec![vec![Sym::Rule(any)]]);
                        s.extend(self.g.repeat(m, 0, None));
                    }
                }
                s.extend(self.text(&format!("</{D}invoke>\n</{D}tool_calls>")));
                s
            }
        })
    }
}

/// Compiles what a reply must be; None when nothing constrains it. With a forced tool choice
/// the reply is one call; with a response format and tools the model chooses between the
/// document and a call (a tool whose parameter schema is not supported then takes any
/// arguments).
pub fn compile(
    vocab: &Arc<Vocab>,
    format: &ResponseFormat,
    choice: &ToolChoice,
    tools: &[ToolSpec],
    syntax: Option<ToolSyntax>,
) -> Result<Option<Compiled>> {
    let forced = *choice != ToolChoice::Auto;
    let mut b = Builder {
        g: Grammar::new(),
        prims: Prims::default(),
        vocab,
        bare: None,
    };
    let mut alts = Vec::new();
    if !forced {
        alts.push(vec![Sym::Rule(match format {
            ResponseFormat::Text => return Ok(None),
            ResponseFormat::JsonObject => schema::json_object(&mut b.g, &mut b.prims),
            ResponseFormat::JsonSchema(s) => {
                schema::compile(&mut b.g, &mut b.prims, s, &Dialect::Json).context("response_format schema")?
            }
        })]);
    }
    let callable: Vec<&ToolSpec> = match choice {
        ToolChoice::Function(name) => {
            let t = tools.iter().find(|t| t.name == *name);
            vec![t.with_context(|| format!("tool_choice names {name:?}, which is not among the tools"))?]
        }
        _ => tools.iter().collect(),
    };
    if forced && callable.is_empty() {
        bail!("tool_choice requires a call, but no tools are declared");
    }
    match syntax {
        None if forced => bail!("this model's chat format has no tool calling"),
        None => {}
        Some(syntax) => {
            for t in callable {
                // A schema that fails leaves rules behind: go back to the grammar before it.
                let before = (!forced).then(|| (b.g.clone(), b.prims.clone(), b.bare));
                alts.push(match b.tool_call(syntax, t) {
                    Ok(call) => call,
                    Err(e) if forced => return Err(e.context(format!("tool {:?}: parameters schema", t.name))),
                    Err(_) => {
                        if let Some((g, prims, bare)) = before {
                            (b.g, b.prims, b.bare) = (g, prims, bare);
                        }
                        let any = ToolSpec {
                            parameters: serde_json::json!({ "type": "object" }),
                            ..t.clone()
                        };
                        b.tool_call(syntax, &any)?
                    }
                });
            }
        }
    }
    let root = b.g.rule(alts);
    let nl = b.g.byte(b'\n');
    let mut after = b.g.repeat(vec![nl], 0, Some(2));
    after.push(Sym::Rule(root));
    let after = b.g.rule(vec![after]);
    b.g.validate()?;
    Ok(Some(Compiled::new(b.g, root, after, vocab.clone())))
}

/// Whether a forced call of `tool` can be constrained: its parameters must be an object
/// schema of the supported subset.
pub fn check_tool(tool: &ToolSpec) -> Result<()> {
    let (mut g, mut prims) = (Grammar::new(), Prims::default());
    Compiler::new(&mut g, &mut prims, &tool.parameters, &Dialect::Json)
        .object_value(&tool.parameters, "#")
        .and_then(|_| g.validate())
        .with_context(|| format!("tool {:?}: parameters schema", tool.name))
}

// ---- the filter -------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// Nothing generated yet; a thinking block may open.
    Start,
    Thinking,
    Answer,
}

/// The state of one constrained reply ([`TokenFilter`]).
#[derive(Clone)]
pub struct Constraint {
    compiled: Arc<Compiled>,
    phase: Phase,
    state: Vec<u32>,
    stops: Vec<u32>,
    /// The tokens that open and close a thinking block, when the reply may think.
    think: Option<(u32, u32)>,
}

impl Constraint {
    /// A constraint at the start of a reply. `think` gives the tokens that open and close a
    /// thinking block when the model may write one, `thinking` whether the reply starts inside
    /// one; `stops` are the tokens that end a reply.
    pub fn new(compiled: Arc<Compiled>, think: Option<(u32, u32)>, thinking: bool, stops: &[u32]) -> Self {
        let state = compiled.start(false);
        Self {
            compiled,
            phase: if thinking && think.is_some() {
                Phase::Thinking
            } else {
                Phase::Start
            },
            state,
            stops: stops.to_vec(),
            think,
        }
    }

    /// Whether `token` may come next.
    pub fn allows(&self, token: u32) -> bool {
        let think = self.think.filter(|_| self.phase != Phase::Answer);
        match self.phase {
            Phase::Thinking => !self.stops.contains(&token) && think.map(|t| t.0) != Some(token),
            _ => {
                let sm = self.compiled.state_mask(&self.state);
                get(&sm.bits, token as usize) || (sm.complete && self.stops.contains(&token)) || think.map(|t| t.0) == Some(token)
            }
        }
    }
}

impl TokenFilter for Constraint {
    fn mask(&mut self, logits: &mut [f32]) {
        let open = self.think.filter(|_| self.phase != Phase::Answer).map(|t| t.0 as usize);
        if self.phase == Phase::Thinking {
            // Thinking is free, but the reply cannot end in it.
            for s in self.stops.iter().map(|&s| s as usize).chain(open) {
                if let Some(l) = logits.get_mut(s) {
                    *l = f32::NEG_INFINITY;
                }
            }
            return;
        }
        let sm = self.compiled.state_mask(&self.state);
        let stops: Vec<(usize, f32)> = self
            .stops
            .iter()
            .filter_map(|&s| logits.get(s as usize).map(|&l| (s as usize, l)))
            .collect();
        let mut any = false;
        for (i, l) in logits.iter_mut().enumerate() {
            if get(&sm.bits, i) || open == Some(i) {
                any = true;
            } else {
                *l = f32::NEG_INFINITY;
            }
        }
        // Nothing can follow (the document is complete, or a dead end): the reply ends.
        if sm.complete || !any {
            for (s, l) in stops {
                logits[s] = l;
            }
        }
    }

    fn accept(&mut self, token: u32) {
        match (self.phase, self.think) {
            (Phase::Thinking, Some((_, close))) => {
                if token == close {
                    self.phase = Phase::Answer;
                    self.state = self.compiled.start(true);
                }
            }
            (Phase::Start, Some((open, _))) if token == open => self.phase = Phase::Thinking,
            _ => {
                self.phase = Phase::Answer;
                if !self.stops.contains(&token) {
                    self.state = self.compiled.advance(&self.state, token);
                }
            }
        }
    }

    fn fork(&self) -> Box<dyn TokenFilter> {
        Box::new(self.clone())
    }

    fn complete(&self) -> bool {
        self.phase != Phase::Thinking && self.state.contains(&DONE)
    }
}

/// An engine's vocabulary and its last compiled constraints (requests often repeat a schema).
#[derive(Default)]
pub struct Cache {
    vocab: Option<Arc<Vocab>>,
    compiled: Vec<(String, Arc<Compiled>)>,
}

impl Cache {
    /// The vocabulary of `tok` as bytes (computed once).
    pub fn vocab(&mut self, tok: &Tokenizer) -> Arc<Vocab> {
        self.vocab.get_or_insert_with(|| Arc::new(Vocab::new(tok))).clone()
    }

    /// The constraint of a reply in chat format `fmt`, or None when nothing constrains it.
    /// `think` is the thinking mode of the turn, `stops` the tokens that end a reply.
    #[allow(clippy::too_many_arguments)]
    pub fn constraint(
        &mut self,
        tok: &Tokenizer,
        fmt: &dyn ChatFormat,
        format: &ResponseFormat,
        choice: &ToolChoice,
        tools: &[ToolSpec],
        think: bool,
        stops: &[u32],
    ) -> Result<Option<Constraint>> {
        if *format == ResponseFormat::Text && *choice == ToolChoice::Auto {
            return Ok(None);
        }
        let syntax = fmt.tool_syntax();
        let key = format!(
            "{format:?}|{choice:?}|{syntax:?}|{}",
            serde_json::to_string(tools).unwrap_or_default()
        );
        let compiled = match self.compiled.iter().find(|c| c.0 == key) {
            Some(c) => c.1.clone(),
            None => {
                let vocab = self.vocab(tok);
                let Some(c) = compile(&vocab, format, choice, tools, syntax)? else {
                    return Ok(None);
                };
                let c = Arc::new(c);
                self.compiled.truncate(7);
                self.compiled.insert(0, (key, c.clone()));
                c
            }
        };
        let (open, close) = fmt.think_tokens();
        let id = |t: &str| if t.is_empty() { None } else { tok.token_to_id(t) };
        let starts = fmt.starts_in_thinking(think);
        // A block may open when the reply starts in one or thinking is on.
        let tokens = id(open).zip(id(close)).filter(|_| think || starts);
        Ok(Some(Constraint::new(compiled, tokens, starts, stops)))
    }
}

#[cfg(test)]
mod tests;
