//! Runner for ANEMLL models: LLMs pre-converted to Core ML for the Apple Neural Engine
//! (<https://github.com/Anemll/Anemll>), driven through [`crate::coreml`].
//!
//! # Layout
//!
//! A model directory holds `meta.yaml` (`model_info.parameters`, read by [`Meta::load`]), the
//! tokenizer files, and compiled models:
//! - chunked: `<prefix>_embeddings[_lutN].mlmodelc` (token ids -> hidden states),
//!   `<prefix>_FFN_PF[_lutN]_chunk_NNofMM.mlmodelc` (the transformer layers split in MM parts,
//!   each a multifunction model with `infer` and `prefill`, plus `infer_rotate` /
//!   `prefill_rotate` for sliding-window models), `<prefix>_lm_head[_lutN].mlmodelc` (hidden
//!   states -> logits); `split_rotate` models keep `infer*` in `..._FFN_..._combined` files and
//!   `prefill*` in `..._PF_...` files;
//! - monolithic (`model_info.model_type: monolithic`): one `<prefix>_monolithic_full[_lutN]`
//!   model with the same functions, taking token ids and returning logits.
//!
//! # Contract (what this runner feeds and reads; same as ANEMLL's `tests/chat.py` and Swift CLI)
//!
//! Every function of the FFN chunks / monolithic model shares one Core ML state (the KV caches:
//! `kv_cache_0`, or `kv_cache_local` + `kv_cache_global` for Gemma 3; any set of states works),
//! made once by `newState` from the chunk that declares the most states. Inputs are filled by
//! name, for `T` tokens at positions `p..p+T` (`T` = 1 for `infer*`, `batch_size` for
//! `prefill*`):
//! - `input_ids` int32 `[1, T]` (embeddings, monolithic) or `hidden_states` fp16 `[1, T, H]`
//!   (chunks: the previous part's output);
//! - `position_ids` int32 `[T]` = `p..p+T`; `current_pos` int32 `[1]` = `p`;
//! - `causal_mask` fp16 `[1, 1, T, W]`: row `i` is 0 for columns `<= p+i`, -inf after;
//! - `update_mask` fp16 `[1, 1, L, T]` (only when the function declares it): 1 at `[p+i, i]`.
//!
//! Other inputs are an error unless an [`InputHook`] fills them (for converters that add their
//! own inputs). Outputs: `hidden_states` (embeddings), `output_hidden_states` (chunks), and from
//! the LM head / monolithic `infer`: `logits1..N` (concatenated in order) or `logits`, or
//! `argmax_idx` int32 + `argmax_val` fp16 (one per vocabulary slice, `argmax_in_model`; slice
//! offsets from `lm_head_chunk_sizes`, else `vocab_size` split evenly).
//!
//! Prefill: full batches go through `prefill` (`prefill_rotate` from `sliding_window` on, when
//! the context is longer than the window); a batch never straddles the window boundary, and the
//! remaining tokens go one by one through `infer` / `infer_rotate`. Then the last prompt token is
//! run once more through `infer` to produce the first prediction (as ANEMLL does: prefill
//! functions only return the last chunk's hidden state of the batch, not logits).
//! [`PrefillMode::Single`] sends every token through `infer`; it is what [`PrefillMode::Auto`]
//! picks, like `chat.py`, for models declaring neither `update_mask_prefill` nor
//! `prefill_dynamic_slice` (older conversions whose batch prefill may be wrong). Only
//! [`PrefillMode::Padded`] pads the last partial batch (with token 0; `update_mask` then marks only
//! the real tokens, as the Swift CLI does for `update_mask` models).
//!
//! Packages written by oxinfer's converter (`tools/coreml_convert.py`, contract in
//! `docs/coreml-format.md`) extend this:
//! - `prefill_last_token_output`: the last chunk's `prefill` returns the final-normed hidden state
//!   of the batch's last token, so every prompt token is fed exactly once: full batches, then the
//!   rest through `infer`, the prediction from the call that processed the last token (the LM
//!   head runs on the `prefill` output when the prompt ends on a batch boundary);
//! - `recurrent_state` (Qwen3.5 / 3.6 / 3.8 dense: Gated DeltaNet and convolution states next to
//!   the KV cache): the same, and never a padded batch (`prefill_full_batches_only`). The state
//!   cannot be rewound: [`AnemllModel::reset`] starts a new one, and a conversation continues it;
//! - Gemma 4: the embeddings model's second output `per_layer_inputs` is wired to the input of
//!   that name of every chunk (generally: an input another part outputs is bound to that output),
//!   with separate local / global KV-cache states.
//!
//! # Extending
//!
//! New converters plug in through the same contract: any number of Core ML states is handled (the
//! runner never names them), extra `meta.yaml` parameters are in [`Meta::params`], and inputs
//! beyond the ones above are filled by an [`InputHook`] passed to [`AnemllModel::load_with`]
//! (loading fails early, naming the input, when there is none).

use crate::coreml::{ComputeUnits, DataType, FeatureKind, Model, MultiArray, Prepared, State};
use anyhow::{Context, Result, anyhow, bail};
use std::path::{Path, PathBuf};

/// A YAML subset parser, enough for `meta.yaml`: block mappings and sequences by indentation,
/// flow sequences of scalars, quoted and plain scalars, `|` / `>` block scalars, comments.
pub mod yaml {
    use anyhow::{Result, bail};

    #[derive(Clone, Debug, PartialEq)]
    pub enum Value {
        Null,
        Bool(bool),
        Int(i64),
        Float(f64),
        Str(String),
        List(Vec<Value>),
        Map(Vec<(String, Value)>),
    }

    impl Value {
        pub fn get(&self, key: &str) -> Option<&Value> {
            match self {
                Value::Map(m) => m.iter().find(|(k, _)| k == key).map(|(_, v)| v),
                _ => None,
            }
        }
        pub fn as_str(&self) -> Option<&str> {
            match self {
                Value::Str(s) => Some(s),
                _ => None,
            }
        }
        pub fn as_i64(&self) -> Option<i64> {
            match self {
                Value::Int(i) => Some(*i),
                Value::Float(f) if f.fract() == 0.0 => Some(*f as i64),
                Value::Str(s) => s.trim().parse().ok(),
                _ => None,
            }
        }
        pub fn as_usize(&self) -> Option<usize> {
            self.as_i64().and_then(|v| usize::try_from(v).ok())
        }
        pub fn as_f64(&self) -> Option<f64> {
            match self {
                Value::Int(i) => Some(*i as f64),
                Value::Float(f) => Some(*f),
                Value::Str(s) => s.trim().parse().ok(),
                _ => None,
            }
        }
        pub fn as_bool(&self) -> Option<bool> {
            match self {
                Value::Bool(b) => Some(*b),
                _ => None,
            }
        }
        /// A list of integers, also from a string such as `"[1, 2]"` or `"1,2"`.
        pub fn as_usize_list(&self) -> Option<Vec<usize>> {
            match self {
                Value::List(l) => l.iter().map(Value::as_usize).collect(),
                Value::Str(s) => s
                    .trim()
                    .trim_matches(['[', ']'])
                    .split(',')
                    .filter(|p| !p.trim().is_empty())
                    .map(|p| p.trim().parse().ok())
                    .collect(),
                _ => None,
            }
        }
    }

    struct Line<'a> {
        indent: usize,
        text: &'a str,
        no: usize,
    }

    /// Strips a comment: `#` at the start or after whitespace, outside quotes.
    fn strip_comment(s: &str) -> &str {
        let (mut sq, mut dq) = (false, false);
        let b = s.as_bytes();
        for (i, &c) in b.iter().enumerate() {
            match c {
                b'\'' if !dq => sq = !sq,
                b'"' if !sq => dq = !dq,
                b'#' if !sq && !dq && (i == 0 || b[i - 1] == b' ' || b[i - 1] == b'\t') => return &s[..i],
                _ => {}
            }
        }
        s
    }

    pub fn parse(src: &str) -> Result<Value> {
        let raw: Vec<&str> = src.lines().collect();
        let mut p = Parser { raw, i: 0 };
        p.skip_blank();
        if p.i >= p.raw.len() {
            return Ok(Value::Map(Vec::new()));
        }
        let ind = p.peek().map(|l| l.indent).unwrap_or(0);
        p.block(ind)
    }

    struct Parser<'a> {
        raw: Vec<&'a str>,
        i: usize,
    }

    impl<'a> Parser<'a> {
        fn line(&self, i: usize) -> Option<Line<'a>> {
            let r = *self.raw.get(i)?;
            if r.trim_start().starts_with("---") && r.trim().chars().all(|c| c == '-') {
                return None;
            }
            let t = strip_comment(r).trim_end();
            let body = t.trim_start();
            if body.is_empty() {
                return None;
            }
            Some(Line {
                indent: t.len() - body.len(),
                text: body,
                no: i + 1,
            })
        }
        fn skip_blank(&mut self) {
            while self.i < self.raw.len() && self.line(self.i).is_none() {
                self.i += 1;
            }
        }
        fn peek(&mut self) -> Option<Line<'a>> {
            self.skip_blank();
            self.line(self.i)
        }

        /// A block (mapping or sequence) whose lines are indented by exactly `ind`.
        fn block(&mut self, ind: usize) -> Result<Value> {
            let first = match self.peek() {
                Some(l) => l,
                None => return Ok(Value::Null),
            };
            if first.text == "-" || first.text.starts_with("- ") {
                let mut items = Vec::new();
                while let Some(l) = self.peek() {
                    if l.indent != ind || !(l.text == "-" || l.text.starts_with("- ")) {
                        if l.indent > ind {
                            bail!("line {}: unexpected indentation", l.no);
                        }
                        break;
                    }
                    self.i += 1;
                    let rest = l.text[1..].trim_start();
                    if rest.is_empty() {
                        let child = self.peek().filter(|c| c.indent > ind).map(|c| c.indent);
                        items.push(match child {
                            Some(ci) => self.block(ci)?,
                            None => Value::Null,
                        });
                    } else if key_split(rest).is_some() {
                        // `- key: value` starts an inline mapping item; further keys are indented
                        // to the column after "- ".
                        let col = ind + (l.text.len() - rest.len());
                        let mut m = Vec::new();
                        self.entry(rest, col, l.no, &mut m)?;
                        while let Some(n) = self.peek() {
                            if n.indent != col {
                                break;
                            }
                            self.i += 1;
                            self.entry(n.text, col, n.no, &mut m)?;
                        }
                        items.push(Value::Map(m));
                    } else {
                        items.push(scalar(rest));
                    }
                }
                return Ok(Value::List(items));
            }
            let mut m = Vec::new();
            while let Some(l) = self.peek() {
                if l.indent < ind {
                    break;
                }
                if l.indent > ind {
                    bail!("line {}: unexpected indentation", l.no);
                }
                self.i += 1;
                self.entry(l.text, ind, l.no, &mut m)?;
            }
            Ok(Value::Map(m))
        }

        /// One `key: value` entry at indentation `ind` (its line already consumed).
        fn entry(&mut self, text: &str, ind: usize, no: usize, m: &mut Vec<(String, Value)>) -> Result<()> {
            let (k, v) = match key_split(text) {
                Some(kv) => kv,
                None => bail!("line {no}: expected `key: value`"),
            };
            let key = unquote(k);
            let v = v.trim();
            let val = if v.is_empty() {
                match self.peek() {
                    Some(c) if c.indent > ind => {
                        let ci = c.indent;
                        self.block(ci)?
                    }
                    // A sequence may sit at the key's own indentation.
                    Some(c) if c.indent == ind && (c.text == "-" || c.text.starts_with("- ")) => self.block(ind)?,
                    _ => Value::Null,
                }
            } else if v.starts_with('|') || v.starts_with('>') {
                Value::Str(self.block_scalar(ind, v.starts_with('>')))
            } else if (v.starts_with('\'') || v.starts_with('"')) && !quote_closed(v) {
                Value::Str(self.multiline_quoted(v))
            } else {
                scalar(v)
            };
            m.push((key, val));
            Ok(())
        }

        /// A quoted scalar continued over the next lines (folded: a line break is a space, an
        /// empty line a newline).
        fn multiline_quoted(&mut self, first: &str) -> String {
            let mut s = first.to_string();
            while self.i < self.raw.len() && !quote_closed(&s) {
                let l = self.raw[self.i].trim();
                self.i += 1;
                if l.is_empty() {
                    s.push('\n');
                } else {
                    if !s.ends_with('\n') {
                        s.push(' ');
                    }
                    s.push_str(l);
                }
            }
            unquote(&s)
        }

        /// The lines of a `|` / `>` block scalar: everything indented deeper than `ind`.
        fn block_scalar(&mut self, ind: usize, folded: bool) -> String {
            let mut lines: Vec<&str> = Vec::new();
            let mut base = None;
            while self.i < self.raw.len() {
                let r = self.raw[self.i];
                let body = r.trim_start();
                let indent = r.len() - body.len();
                if !body.is_empty() && indent <= ind {
                    break;
                }
                if !body.is_empty() && base.is_none() {
                    base = Some(indent);
                }
                lines.push(r);
                self.i += 1;
            }
            while lines.last().is_some_and(|l| l.trim().is_empty()) {
                lines.pop();
            }
            let b = base.unwrap_or(0);
            let parts: Vec<&str> = lines.iter().map(|l| if l.len() >= b { &l[b..] } else { l.trim() }).collect();
            let mut s = parts.join(if folded { " " } else { "\n" });
            s.push('\n');
            s
        }
    }

    /// Splits `key: value` at the first `:` followed by a space or the end, outside quotes.
    fn key_split(s: &str) -> Option<(&str, &str)> {
        let (mut sq, mut dq) = (false, false);
        let b = s.as_bytes();
        for (i, &c) in b.iter().enumerate() {
            match c {
                b'\'' if !dq => sq = !sq,
                b'"' if !sq => dq = !dq,
                b':' if !sq && !dq && (i + 1 == b.len() || b[i + 1] == b' ') => return Some((&s[..i], &s[i + 1..])),
                b'[' | b'{' if i == 0 => return None,
                _ => {}
            }
        }
        None
    }

    /// Whether a scalar starting with a quote also ends its quoted part.
    fn quote_closed(s: &str) -> bool {
        let q = s.as_bytes()[0];
        let b = &s.as_bytes()[1..];
        let mut i = 0;
        while i < b.len() {
            match b[i] {
                b'\\' if q == b'"' => i += 1,
                b'\'' if q == b'\'' && b.get(i + 1) == Some(&b'\'') => i += 1,
                c if c == q => return true,
                _ => {}
            }
            i += 1;
        }
        false
    }

    fn unquote(s: &str) -> String {
        let s = s.trim();
        if s.len() >= 2 && ((s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\''))) {
            let inner = &s[1..s.len() - 1];
            if s.starts_with('\'') {
                return inner.replace("''", "'");
            }
            let mut out = String::new();
            let mut it = inner.chars();
            while let Some(c) = it.next() {
                if c == '\\' {
                    match it.next() {
                        Some('n') => out.push('\n'),
                        Some('t') => out.push('\t'),
                        Some(o) => out.push(o),
                        None => {}
                    }
                } else {
                    out.push(c);
                }
            }
            return out;
        }
        s.to_string()
    }

    fn scalar(s: &str) -> Value {
        let s = s.trim();
        if s.starts_with('[') && s.ends_with(']') {
            let inner = &s[1..s.len() - 1];
            return Value::List(split_flow(inner).into_iter().map(scalar).collect());
        }
        if s.starts_with('"') || s.starts_with('\'') {
            return Value::Str(unquote(s));
        }
        match s {
            "~" | "null" | "Null" | "NULL" => return Value::Null,
            "true" | "True" | "TRUE" => return Value::Bool(true),
            "false" | "False" | "FALSE" => return Value::Bool(false),
            _ => {}
        }
        if let Ok(i) = s.parse::<i64>() {
            return Value::Int(i);
        }
        if s.chars().any(|c| c.is_ascii_digit())
            && let Ok(f) = s.parse::<f64>()
        {
            return Value::Float(f);
        }
        Value::Str(s.to_string())
    }

    /// Splits a flow sequence body at top-level commas.
    fn split_flow(s: &str) -> Vec<&str> {
        let mut out = Vec::new();
        let (mut depth, mut sq, mut dq, mut start) = (0i32, false, false, 0);
        for (i, c) in s.char_indices() {
            match c {
                '\'' if !dq => sq = !sq,
                '"' if !sq => dq = !dq,
                '[' | '{' if !sq && !dq => depth += 1,
                ']' | '}' if !sq && !dq => depth -= 1,
                ',' if depth == 0 && !sq && !dq => {
                    out.push(&s[start..i]);
                    start = i + 1;
                }
                _ => {}
            }
        }
        if !s[start..].trim().is_empty() {
            out.push(&s[start..]);
        }
        out
    }
}

/// Model parameters from `meta.yaml`, with ANEMLL's defaults and file-name rules applied.
#[derive(Clone, Debug)]
pub struct Meta {
    pub dir: PathBuf,
    pub name: String,
    pub version: String,
    /// `model_info.architecture` (e.g. `qwen3`, `gemma3_text`, `llama`).
    pub architecture: String,
    pub model_prefix: String,
    pub monolithic: bool,
    pub context_length: usize,
    pub state_length: usize,
    pub batch_size: usize,
    pub num_chunks: usize,
    pub split_lm_head: usize,
    pub vocab_size: Option<usize>,
    pub lm_head_chunk_sizes: Option<Vec<usize>>,
    pub argmax_in_model: bool,
    /// Gemma 3 local attention window; rotate functions are used from this position on.
    pub sliding_window: Option<usize>,
    pub update_mask_prefill: bool,
    pub prefill_dynamic_slice: bool,
    /// The last chunk's `prefill` returns the final-normed hidden state of the batch's last token
    /// (oxinfer's converter, docs/coreml-format.md): the prompt is then fed exactly once, the LM
    /// head reading that output when the prompt ends on a batch boundary.
    pub prefill_last_token_output: bool,
    /// States that cannot be rewound (Qwen3.5 Gated DeltaNet): every token is fed exactly once,
    /// in order.
    pub recurrent_state: bool,
    /// `prefill` only takes full batches of real tokens (no padding).
    pub prefill_full_batches_only: bool,
    /// Gemma 4 long contexts: the sliding-window layers keep a shifting cache of this many
    /// positions, which every call advances by its batch, so (as for `recurrent_state`) every
    /// token is fed exactly once, in order, and never padded.
    pub local_cache_length: Option<usize>,
    pub split_rotate: bool,
    /// Resolved model paths (empty for the parts a layout does not have).
    pub embeddings: PathBuf,
    pub lm_head: PathBuf,
    /// FFN chunks, in order (the monolithic model alone for monolithic layouts).
    pub ffn: Vec<PathBuf>,
    /// Prefill chunks of `split_rotate` layouts.
    pub pf: Vec<PathBuf>,
    /// Directory with `tokenizer.json` (`tokenizer_path`, else the model directory).
    pub tokenizer_dir: PathBuf,
    /// `recommended_sampling` (temperature, top_k, top_p, do_sample), when given.
    pub recommended_sampling: Option<(f32, usize, f32, bool)>,
    /// The whole `model_info.parameters` mapping, for extensions.
    pub params: yaml::Value,
}

/// Whether `dir` holds an ANEMLL model (a `meta.yaml` describing Core ML parts).
pub fn is_anemll(dir: &Path) -> bool {
    let Ok(text) = std::fs::read_to_string(dir.join("meta.yaml")) else {
        return false;
    };
    // Compiled parts next to it are enough (the loader then reports any meta.yaml problem).
    if has_mlmodelc(dir) {
        return true;
    }
    let Ok(v) = yaml::parse(&text) else {
        return false;
    };
    let info = v.get("model_info");
    let framework = info.and_then(|i| i.get("framework")).and_then(yaml::Value::as_str).unwrap_or("");
    info.and_then(|i| i.get("parameters")).is_some()
        && (framework.eq_ignore_ascii_case("core ml") || framework.eq_ignore_ascii_case("coreml"))
}

fn has_mlmodelc(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .map(|rd| rd.flatten().any(|e| e.file_name().to_string_lossy().ends_with(".mlmodelc")))
        .unwrap_or(false)
}

fn strip_ext(s: &str) -> &str {
    s.strip_suffix(".mlmodelc").or_else(|| s.strip_suffix(".mlpackage")).unwrap_or(s)
}

/// `dir/name[.mlmodelc]` when it exists.
fn find_model(dir: &Path, name: &str) -> Option<PathBuf> {
    let base = strip_ext(name);
    [dir.join(format!("{base}.mlmodelc")), dir.join(name)]
        .into_iter()
        .find(|p| p.is_dir())
}

/// `(chunk index, chunk count)` from a `..._chunk_NNofMM...` name, with the text around it.
fn chunk_parts(name: &str) -> Option<(String, usize, usize, String)> {
    let at = name.rfind("_chunk_")?;
    let rest = &name[at + 7..];
    let of = rest.find("of")?;
    let i: usize = rest[..of].parse().ok()?;
    let digits = rest[of + 2..].chars().take_while(char::is_ascii_digit).count();
    let n: usize = rest[of + 2..of + 2 + digits].parse().ok()?;
    Some((name[..at].to_string(), i, n, rest[of + 2 + digits..].to_string()))
}

/// All chunks of a chunked model path, in order.
fn all_chunks(first: &Path) -> Result<Vec<PathBuf>> {
    let name = first.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    let dir = first.parent().unwrap_or(Path::new("."));
    let Some((pre, _, n, post)) = chunk_parts(&name) else {
        return Ok(vec![first.to_path_buf()]);
    };
    (1..=n)
        .map(|i| {
            let p = dir.join(format!("{pre}_chunk_{i:02}of{n:02}{post}"));
            if p.is_dir() {
                Ok(p)
            } else {
                bail!("missing FFN chunk {}", p.display())
            }
        })
        .collect()
}

impl Meta {
    /// States that only move forward: a recurrent state or a shifting local cache. Every token
    /// is then fed exactly once, in order, with no padded batches.
    pub fn no_rewind(&self) -> bool {
        self.recurrent_state || self.local_cache_length.is_some()
    }

    pub fn load(dir: &Path) -> Result<Meta> {
        let path = dir.join("meta.yaml");
        let text = std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        let root = yaml::parse(&text).with_context(|| format!("parse {}", path.display()))?;
        let info = root
            .get("model_info")
            .with_context(|| format!("{}: no model_info", path.display()))?;
        let params = info
            .get("parameters")
            .cloned()
            .with_context(|| format!("{}: no model_info.parameters", path.display()))?;
        let p = |k: &str| params.get(k);
        let s = |v: Option<&yaml::Value>| v.and_then(|v| v.as_str().map(str::to_string));
        let prefix = s(p("model_prefix")).unwrap_or_else(|| "llama".into());
        let monolithic = s(info.get("model_type")).as_deref() == Some("monolithic");
        let context_length = p("context_length").and_then(yaml::Value::as_usize).unwrap_or(2048);
        let lut = |k: &str| match p(k) {
            None | Some(yaml::Value::Null) => String::new(),
            Some(v) if v.as_str() == Some("none") => String::new(),
            Some(v) => format!(
                "_lut{}",
                v.as_i64()
                    .map(|i| i.to_string())
                    .or_else(|| v.as_str().map(str::to_string))
                    .unwrap_or_default()
            ),
        };
        let num_chunks = if monolithic {
            1
        } else {
            p("num_chunks").and_then(yaml::Value::as_usize).unwrap_or(1)
        };
        let split_rotate = p("split_rotate").and_then(yaml::Value::as_bool).unwrap_or(false);
        let sliding_window = p("sliding_window")
            .and_then(yaml::Value::as_usize)
            .or_else(|| prefix.to_lowercase().starts_with("gemma3").then_some(512));
        let need = |what: &str, name: &str, found: Option<PathBuf>| -> Result<PathBuf> {
            found.with_context(|| format!("{}: {what} model {name}[.mlmodelc] not found", dir.display()))
        };
        let (mut embeddings, mut lm_head, mut ffn, mut pf) = (PathBuf::new(), PathBuf::new(), Vec::new(), Vec::new());
        if monolithic {
            let name = s(p("monolithic_model")).unwrap_or_else(|| format!("{prefix}_monolithic_full{}.mlmodelc", lut("lut_bits")));
            ffn.push(need("monolithic", &name, find_model(dir, &name))?);
        } else {
            let name = s(p("embeddings")).unwrap_or_else(|| format!("{prefix}_embeddings{}", lut("lut_embeddings")));
            let mut found = find_model(dir, &name);
            if found.is_none() && name.contains("_lut") {
                found = find_model(dir, name.split("_lut").next().unwrap_or(&name));
            }
            embeddings = need("embeddings", &name, found)?;
            let name = s(p("lm_head")).unwrap_or_else(|| format!("{prefix}_lm_head{}", lut("lut_lmhead")));
            lm_head = need("LM head", &name, find_model(dir, &name))?;
            let lut_ffn = lut("lut_ffn");
            let canonical = format!("{prefix}_FFN_PF{lut_ffn}_chunk_01of{num_chunks:02}");
            let ffn_name = if split_rotate {
                s(p("ffn")).unwrap_or_else(|| format!("{prefix}_FFN{lut_ffn}_chunk_01of{num_chunks:02}_combined"))
            } else {
                match s(p("ffn")) {
                    Some(f) if !f.contains("_chunk_") && find_model(dir, &canonical).is_some() => canonical.clone(),
                    Some(f) => f,
                    None => canonical.clone(),
                }
            };
            let mut first = find_model(dir, &ffn_name);
            if first.is_none() && !ffn_name.contains("_chunk_") {
                // An unchunked name in meta.yaml for a chunked file set.
                let base = strip_ext(&ffn_name);
                first = find_model(dir, &format!("{base}_chunk_01of{num_chunks:02}"));
            }
            ffn = all_chunks(&need("FFN", &ffn_name, first)?)?;
            if split_rotate {
                let pf_name = s(p("pf")).unwrap_or_else(|| format!("{prefix}_PF{lut_ffn}_chunk_01of{num_chunks:02}"));
                pf = all_chunks(&need("prefill", &pf_name, find_model(dir, &pf_name))?)?;
                if pf.len() != ffn.len() {
                    bail!("{} FFN chunks but {} prefill chunks", ffn.len(), pf.len());
                }
            }
        }
        let tokenizer_dir = match s(p("tokenizer_path")) {
            Some(t) if dir.join(&t).join("tokenizer.json").exists() => dir.join(t),
            _ => dir.to_path_buf(),
        };
        let recommended_sampling = p("recommended_sampling").and_then(|r| {
            Some((
                r.get("temperature")?.as_f64()? as f32,
                r.get("top_k").or_else(|| r.get("topK"))?.as_usize()?,
                r.get("top_p").or_else(|| r.get("topP"))?.as_f64()? as f32,
                r.get("do_sample").and_then(yaml::Value::as_bool).unwrap_or(true),
            ))
        });
        Ok(Meta {
            dir: dir.to_path_buf(),
            name: s(info.get("name")).unwrap_or_default(),
            version: info
                .get("version")
                .map(|v| match v {
                    yaml::Value::Str(s) => s.clone(),
                    yaml::Value::Float(f) => f.to_string(),
                    yaml::Value::Int(i) => i.to_string(),
                    _ => String::new(),
                })
                .unwrap_or_default(),
            architecture: s(info.get("architecture")).unwrap_or_default(),
            model_prefix: prefix,
            monolithic,
            context_length,
            state_length: p("state_length").and_then(yaml::Value::as_usize).unwrap_or(context_length),
            batch_size: p("batch_size").and_then(yaml::Value::as_usize).unwrap_or(64),
            num_chunks,
            split_lm_head: p("split_lm_head").and_then(yaml::Value::as_usize).unwrap_or(8),
            vocab_size: p("vocab_size").and_then(yaml::Value::as_usize),
            lm_head_chunk_sizes: p("lm_head_chunk_sizes")
                .and_then(yaml::Value::as_usize_list)
                .filter(|v| !v.is_empty()),
            argmax_in_model: p("argmax_in_model").and_then(yaml::Value::as_bool).unwrap_or(false),
            sliding_window,
            update_mask_prefill: p("update_mask_prefill").and_then(yaml::Value::as_bool).unwrap_or(false),
            prefill_dynamic_slice: p("prefill_dynamic_slice").and_then(yaml::Value::as_bool).unwrap_or(false),
            prefill_last_token_output: p("prefill_last_token_output").and_then(yaml::Value::as_bool).unwrap_or(false),
            recurrent_state: p("recurrent_state").and_then(yaml::Value::as_bool).unwrap_or(false),
            prefill_full_batches_only: p("prefill_full_batches_only").and_then(yaml::Value::as_bool).unwrap_or(false),
            local_cache_length: p("local_cache_length").and_then(yaml::Value::as_usize),
            split_rotate,
            embeddings,
            lm_head,
            ffn,
            pf,
            tokenizer_dir,
            recommended_sampling,
            params,
        })
    }

    /// The model's tokenizer (`tokenizer.json`).
    pub fn tokenizer(&self) -> Result<tokenizers::Tokenizer> {
        let p = self.tokenizer_dir.join("tokenizer.json");
        tokenizers::Tokenizer::from_file(&p).map_err(|e| anyhow!("{}: {e}", p.display()))
    }

    fn tokenizer_config(&self) -> serde_json::Value {
        std::fs::read(self.tokenizer_dir.join("tokenizer_config.json"))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default()
    }

    /// The chat format matching the model's template (`chat_template.jinja` or
    /// `tokenizer_config.json`): Llama 3, Gemma 4, Gemma 3, Qwen2 / 2.5 or Qwen3 ChatML.
    pub fn chat_format(&self) -> Result<&'static (dyn crate::chat::ChatFormat + Sync)> {
        use crate::chat;
        let cfg = self.tokenizer_config();
        let jinja = std::fs::read_to_string(self.tokenizer_dir.join("chat_template.jinja")).ok();
        let tpl = jinja.as_deref().or(cfg["chat_template"].as_str()).unwrap_or("");
        let arch = format!("{} {}", self.architecture, self.model_prefix).to_lowercase();
        Ok(if tpl.contains("<|start_header_id|>") {
            &chat::Llama3Chat
        } else if tpl.contains("<|turn>") || (tpl.is_empty() && arch.contains("gemma4")) {
            &chat::GemmaChat
        } else if tpl.contains("<start_of_turn>") || arch.contains("gemma3") {
            &chat::Gemma3Chat
        } else if (tpl.contains("<|im_start|>") && !tpl.contains("</think>")) || (tpl.is_empty() && arch.contains("qwen2")) {
            // Qwen2 / 2.5 ChatML (no thinking block); made once per model, a few bytes leaked.
            Box::leak(Box::new(chat::Qwen2Chat::for_template(tpl)))
        } else if tpl.contains("<|im_start|>") || arch.contains("qwen") {
            chat::QwenChat::for_template(Some(tpl))
        } else {
            bail!("no chat format for this model ({}); format the prompt yourself", self.architecture)
        })
    }

    /// Stop tokens as ANEMLL's `chat.py` collects them: the tokenizer's EOS token, then
    /// `<|endoftext|>`, `<end_of_turn>` and `<|eot_id|>` when the vocabulary has them, plus the
    /// chat format's end-of-turn token.
    pub fn stop_tokens(&self, tok: &tokenizers::Tokenizer, fmt: &dyn crate::chat::ChatFormat) -> Vec<u32> {
        let cfg = self.tokenizer_config();
        let eos = &cfg["eos_token"];
        let mut ids = Vec::new();
        let names = [
            eos.as_str().or(eos["content"].as_str()).unwrap_or(""),
            "<|endoftext|>",
            "<end_of_turn>",
            "<|eot_id|>",
        ];
        for n in names.into_iter().filter(|n| !n.is_empty()) {
            if let Some(id) = tok.token_to_id(n)
                && !ids.contains(&id)
            {
                ids.push(id);
            }
        }
        crate::chat::with_turn_end(fmt, tok, ids)
    }

    /// Per-slice sizes of an `argmax_in_model` head with `n` slices (ANEMLL's rule).
    pub fn argmax_layout(&self, n: usize) -> Vec<usize> {
        if let Some(s) = &self.lm_head_chunk_sizes
            && s.len() == n
        {
            return s.clone();
        }
        let vocab = self.vocab_size.or_else(|| {
            let a = if self.architecture.is_empty() {
                self.model_prefix.to_lowercase()
            } else {
                self.architecture.to_lowercase()
            };
            if a.starts_with("gemma") {
                Some(262144)
            } else if a.starts_with("qwen") {
                Some(151936)
            } else if a.starts_with("llama") {
                Some(128257)
            } else {
                None
            }
        });
        match vocab {
            Some(v) if n > 0 => (0..n).map(|i| v / n + usize::from(i < v % n)).collect(),
            _ => vec![0; n],
        }
    }
}

/// How the prompt is written into the KV cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillMode {
    /// `Batched` when `meta.yaml` declares `update_mask_prefill` or `prefill_dynamic_slice`,
    /// else `Single` (as ANEMLL's `chat.py`: the batch prefill of older conversions can be
    /// wrong, e.g. `anemll-google-gemma-3-270m-it-ctx512-monolithic_0.3.5` produces garbage).
    /// `Padded` for packages of oxinfer's converter (`prefill_last_token_output`) without
    /// recurrent state or shifting local cache.
    Auto,
    /// Full batches through the `prefill` functions, the rest token by token.
    Batched,
    /// As `Batched`, with the last partial batch padded (outside rotate mode): one prediction
    /// instead of up to `batch_size - 1`, much faster for short prompts, but the KV entries of
    /// those tokens differ from ANEMLL's runtimes in fp16 rounding (the Swift CLI does the same
    /// for `update_mask` models). Same as `Batched` for recurrent models (`recurrent_state`,
    /// `prefill_full_batches_only`), whose state would absorb the padding, and for shifting local
    /// caches (`local_cache_length`), whose window it would advance.
    Padded,
    /// Every token through `infer`. Slow (decode speed).
    Single,
}

/// Loading options.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub units: ComputeUnits,
    pub prefill: PrefillMode,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            units: ComputeUnits::CpuAndNeuralEngine,
            prefill: PrefillMode::Auto,
        }
    }
}

/// The model's prediction for the next token.
#[derive(Clone, Debug)]
pub enum Next {
    /// Full logits (vocabulary order, possibly padded past the tokenizer's vocabulary).
    Logits(Vec<f32>),
    /// The head computed the argmax in the model (`argmax_in_model`): token id and its logit.
    Argmax { token: u32, logit: f32 },
}

impl Next {
    /// The greedy token.
    pub fn argmax(&self) -> u32 {
        match self {
            Next::Logits(l) => crate::sampler::argmax(l),
            Next::Argmax { token, .. } => *token,
        }
    }
}

/// Fills inputs the runner does not know (for converters adding their own inputs).
pub trait InputHook: Send {
    /// Fills `arr` for input `name` of `function` (`infer`, `prefill`, ...; `embeddings`,
    /// `lm_head`) for `tokens` tokens starting at position `pos`. Returns false if it does not
    /// know the input.
    fn fill(&mut self, function: &str, name: &str, pos: usize, tokens: usize, arr: &mut MultiArray) -> Result<bool>;
}

/// One loaded function.
struct Func {
    /// `embeddings`, `lm_head`, or the function name (`infer`, `prefill`, ...).
    name: String,
    model: Model,
}

/// The functions of one FFN chunk (or of the monolithic model), as indices into `funcs`.
struct Chunk {
    infer: usize,
    prefill: usize,
    infer_rotate: Option<usize>,
    prefill_rotate: Option<usize>,
    /// Index into `AnemllModel::states`.
    state: usize,
}

/// One prediction of a plan, bound to its arrays.
struct Step {
    func: usize,
    prep: Prepared,
    state: Option<usize>,
    /// Inputs filled by the [`InputHook`]: (name, array index).
    hooked: Vec<(String, usize)>,
}

/// The chain of predictions for one token (`infer*`) or one batch (`prefill*`): embeddings,
/// chunks and LM head with each part's output backing bound as the next part's input, so a step
/// only writes the token / position / mask arrays and runs.
struct Plan {
    tokens: usize,
    steps: Vec<Step>,
    /// Whether the last step is the separate LM head (skipped by KV-only calls).
    has_head: bool,
    /// Output arrays of the last step (LM head or monolithic model): (name, array index).
    out: Vec<(String, usize)>,
}

impl Plan {
    fn empty() -> Plan {
        Plan {
            tokens: 0,
            steps: Vec::new(),
            has_head: false,
            out: Vec::new(),
        }
    }
}

/// Loading / prediction timings and counts.
#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub load_secs: f64,
    pub predictions: u64,
}

/// An ANEMLL model with its KV-cache state and current position.
pub struct AnemllModel {
    pub meta: Meta,
    pub options: Options,
    funcs: Vec<Func>,
    chunks: Vec<Chunk>,
    embed: Option<usize>,
    lm_head: Option<usize>,
    /// Every input / output array of the plans.
    arrays: Vec<MultiArray>,
    /// Shared input arrays by (name, shape): written once per step, read by every part.
    shared: Vec<(String, Vec<usize>, usize)>,
    infer_plan: Plan,
    prefill_plan: Plan,
    infer_rotate_plan: Option<Plan>,
    prefill_rotate_plan: Option<Plan>,
    states: Vec<State>,
    /// Next position to write (tokens `0..pos` are in the cache).
    pos: usize,
    /// Whether the rotate functions are used from `sliding_window` on.
    rotation: bool,
    /// Every prompt token is fed exactly once (no re-run of the last one): packages with
    /// `prefill_last_token_output`, `recurrent_state` or `local_cache_length`.
    exact: bool,
    /// The plan whose head outputs hold the latest prediction.
    head_plan: PlanKind,
    hook: Option<Box<dyn InputHook>>,
    pub stats: Stats,
}

const KNOWN_INPUTS: [&str; 6] = [
    "input_ids",
    "hidden_states",
    "position_ids",
    "causal_mask",
    "current_pos",
    "update_mask",
];
const F16_NEG_INF: u16 = 0xFC00;

/// A fresh output array: IOSurface-backed for fp16 (what the Neural Engine writes directly).
fn output_array(shape: &[usize], dtype: DataType) -> Result<MultiArray> {
    if dtype == DataType::F16 {
        MultiArray::surface(shape)
    } else {
        MultiArray::new(shape, dtype)
    }
}

impl AnemllModel {
    /// Loads the model in `dir` (see the module docs for the layout).
    pub fn load(dir: &Path, options: Options) -> Result<AnemllModel> {
        Self::load_with(dir, options, None)
    }

    /// As [`load`](Self::load), with a hook for inputs this runner does not know.
    pub fn load_with(dir: &Path, options: Options, hook: Option<Box<dyn InputHook>>) -> Result<AnemllModel> {
        crate::coreml::available()?;
        let t0 = std::time::Instant::now();
        let meta = Meta::load(dir)?;
        let units = options.units;
        let mut funcs: Vec<Func> = Vec::new();
        let mut load = |p: &Path, f: Option<&str>, label: &str| -> Result<usize> {
            funcs.push(Func {
                name: f.unwrap_or(label).to_string(),
                model: Model::load(p, units, f)?,
            });
            Ok(funcs.len() - 1)
        };
        let want_rotation = meta.sliding_window.is_some_and(|sw| meta.context_length > sw);
        let (embed, lm_head) = if meta.monolithic {
            (None, None)
        } else {
            (
                Some(load(&meta.embeddings, None, "embeddings")?),
                Some(load(&meta.lm_head, None, "lm_head")?),
            )
        };
        let mut chunks = Vec::new();
        for (i, path) in meta.ffn.iter().enumerate() {
            let pf_path = if meta.split_rotate { &meta.pf[i] } else { path };
            let infer = load(path, Some("infer"), "")?;
            let prefill = load(pf_path, Some("prefill"), "")?;
            // Optional functions: absent in 2-function models.
            let (infer_rotate, prefill_rotate) = if want_rotation {
                (
                    load(path, Some("infer_rotate"), "").ok(),
                    load(pf_path, Some("prefill_rotate"), "").ok(),
                )
            } else {
                (None, None)
            };
            chunks.push(Chunk {
                infer,
                prefill,
                infer_rotate,
                prefill_rotate,
                state: 0,
            });
        }
        if chunks.is_empty() {
            bail!("{}: no FFN chunks", dir.display());
        }
        // Rotation needs both rotate functions in every chunk.
        let rotation = want_rotation && chunks.iter().all(|c| c.infer_rotate.is_some() && c.prefill_rotate.is_some());
        let exact = meta.prefill_last_token_output || meta.no_rewind();
        if meta.recurrent_state && rotation {
            bail!(
                "{}: recurrent states cannot be combined with sliding-window rotation",
                dir.display()
            );
        }
        let mut options = options;
        if options.prefill == PrefillMode::Auto {
            let no_pad = meta.no_rewind() || meta.prefill_full_batches_only;
            options.prefill = if meta.prefill_last_token_output && meta.prefill_dynamic_slice && !no_pad {
                // oxinfer's converter: padding the last partial batch gives the same logits as
                // feeding those tokens through `infer` (verified bit-identical on Qwen3 and
                // Gemma 4 E2B / E4B) at a fraction of the calls.
                PrefillMode::Padded
            } else if meta.update_mask_prefill || meta.prefill_dynamic_slice {
                PrefillMode::Batched
            } else {
                PrefillMode::Single
            };
        }
        let mut m = AnemllModel {
            meta,
            options,
            funcs,
            chunks,
            embed,
            lm_head,
            arrays: Vec::new(),
            shared: Vec::new(),
            infer_plan: Plan::empty(),
            prefill_plan: Plan::empty(),
            infer_rotate_plan: None,
            prefill_rotate_plan: None,
            states: Vec::new(),
            pos: 0,
            rotation,
            exact,
            head_plan: PlanKind::Infer,
            hook,
            stats: Stats::default(),
        };
        m.check_inputs()?;
        m.make_states()?;
        let b = m.meta.batch_size;
        m.infer_plan = m.plan(|c| Some(c.infer), 1, true)?;
        // With `prefill_last_token_output` the last chunk's prefill output feeds the LM head.
        m.prefill_plan = m.plan(|c| Some(c.prefill), b, m.meta.prefill_last_token_output)?;
        if m.rotation {
            m.infer_rotate_plan = Some(m.plan(|c| c.infer_rotate, 1, true)?);
            m.prefill_rotate_plan = Some(m.plan(|c| c.prefill_rotate, b, false)?);
        }
        m.stats.load_secs = t0.elapsed().as_secs_f64();
        Ok(m)
    }

    /// One shared state from the chunk declaring the most states, if it covers them all;
    /// otherwise one state per chunk (models from older converters).
    fn make_states(&mut self) -> Result<()> {
        let names = |c: &Chunk| -> Vec<String> {
            let mut v: Vec<String> = [c.prefill, c.infer]
                .iter()
                .flat_map(|&f| self.funcs[f].model.states.iter().map(|s| s.name.clone()))
                .collect();
            v.sort();
            v.dedup();
            v
        };
        let all: Vec<Vec<String>> = self.chunks.iter().map(names).collect();
        let best = (0..all.len()).max_by_key(|&i| (all[i].len(), usize::MAX - i)).unwrap_or(0);
        let covers = all.iter().all(|n| n.iter().all(|s| all[best].contains(s)));
        // ANEMLL makes the state from `prefill` (from `infer` for monolithic models).
        let maker = |c: &Chunk| if self.meta.monolithic { c.infer } else { c.prefill };
        self.states.clear();
        if covers {
            self.states.push(self.funcs[maker(&self.chunks[best])].model.new_state()?);
            for c in &mut self.chunks {
                c.state = 0;
            }
        } else {
            for i in 0..self.chunks.len() {
                let s = self.funcs[maker(&self.chunks[i])].model.new_state()?;
                self.states.push(s);
                self.chunks[i].state = i;
            }
        }
        Ok(())
    }

    /// Fails early on inputs nobody can fill.
    fn check_inputs(&self) -> Result<()> {
        if self.hook.is_some() {
            return Ok(());
        }
        // Inputs another part produces (Gemma 4's `per_layer_inputs`) are wired by name.
        let produced = |name: &str| self.funcs.iter().any(|f| f.model.output(name).is_some());
        for f in &self.funcs {
            for d in &f.model.inputs {
                if d.kind == FeatureKind::MultiArray && !KNOWN_INPUTS.contains(&d.name.as_str()) && !produced(&d.name) {
                    bail!(
                        "{} takes an input `{}` {:?} this runner does not know (load with an InputHook)",
                        f.name,
                        d.name,
                        d.shape
                    );
                }
            }
        }
        Ok(())
    }

    /// The shared array for input `name` with `shape`, created on first use.
    fn shared_input(&mut self, name: &str, shape: Vec<usize>, dtype: DataType) -> Result<usize> {
        if let Some(&(_, _, i)) = self.shared.iter().find(|(n, s, _)| n == name && *s == shape) {
            return Ok(i);
        }
        // fp16 masks live in IOSurfaces like the hidden states; int32 arrays cannot.
        let a = if dtype == DataType::F16 {
            MultiArray::surface(&shape)?
        } else {
            MultiArray::new(&shape, dtype)?
        };
        self.arrays.push(a);
        let i = self.arrays.len() - 1;
        self.shared.push((name.to_string(), shape, i));
        Ok(i)
    }

    /// Wires the prediction chain for `t` tokens through the chunk functions `pick` selects.
    fn plan(&mut self, pick: impl Fn(&Chunk) -> Option<usize>, t: usize, head: bool) -> Result<Plan> {
        let mut chain: Vec<(usize, Option<usize>)> = Vec::new();
        if let Some(e) = self.embed {
            chain.push((e, None));
        }
        for c in &self.chunks {
            chain.push((pick(c).context("missing chunk function")?, Some(c.state)));
        }
        if head && let Some(h) = self.lm_head {
            chain.push((h, None));
        }
        let mut steps = Vec::new();
        let mut prev: Option<usize> = None;
        // Outputs of earlier steps other than the hidden-state chain, by name.
        let mut produced: Vec<(String, usize)> = Vec::new();
        let mut out = Vec::new();
        for (k, &(fi, state)) in chain.iter().enumerate() {
            let fname = self.funcs[fi].name.clone();
            let inputs = self.funcs[fi].model.inputs.clone();
            let mut bound: Vec<(String, usize)> = Vec::new();
            let mut hooked = Vec::new();
            for d in inputs.iter().filter(|d| d.kind == FeatureKind::MultiArray) {
                let idx = match d.name.as_str() {
                    "hidden_states" => {
                        let p = prev.with_context(|| format!("{fname}: hidden_states input but nothing produces it"))?;
                        if !d.shape.is_empty() && self.arrays[p].shape != d.shape {
                            bail!(
                                "{fname}: hidden_states input {:?} does not match the previous part's output {:?}",
                                d.shape,
                                self.arrays[p].shape
                            );
                        }
                        p
                    }
                    "input_ids" => self.shared_input("input_ids", vec![1, t], DataType::I32)?,
                    "position_ids" => self.shared_input("position_ids", vec![t], DataType::I32)?,
                    "current_pos" => self.shared_input("current_pos", vec![1], DataType::I32)?,
                    "causal_mask" => {
                        let w = *d.shape.last().context("causal_mask shape")?;
                        self.shared_input("causal_mask", vec![1, 1, t, w], DataType::F16)?
                    }
                    "update_mask" => {
                        let l = d.shape.get(2).copied().unwrap_or(self.meta.context_length);
                        self.shared_input("update_mask", vec![1, 1, l, t], DataType::F16)?
                    }
                    other if produced.iter().any(|(n, _)| n == other) => {
                        let p = produced.iter().find(|(n, _)| n == other).map(|&(_, i)| i).unwrap_or_default();
                        if !d.shape.is_empty() && self.arrays[p].shape != d.shape {
                            bail!(
                                "{fname}: input {other} {:?} does not match the output {:?} of an earlier part",
                                d.shape,
                                self.arrays[p].shape
                            );
                        }
                        p
                    }
                    other => {
                        self.arrays.push(MultiArray::new(&d.shape, d.dtype.unwrap_or(DataType::F16))?);
                        hooked.push((other.to_string(), self.arrays.len() - 1));
                        self.arrays.len() - 1
                    }
                };
                bound.push((d.name.clone(), idx));
            }
            // Output backings. The embeddings' output shapes are flexible: take the next part's
            // `hidden_states` input, or for other outputs the first later input of that name.
            let outputs = self.funcs[fi].model.outputs.clone();
            let next_in = chain
                .get(k + 1)
                .and_then(|&(nf, _)| self.funcs[nf].model.input("hidden_states").map(|d| d.shape.clone()));
            let later_in = |name: &str| {
                chain[k + 1..]
                    .iter()
                    .find_map(|&(nf, _)| self.funcs[nf].model.input(name).map(|d| d.shape.clone()))
                    .filter(|s| !s.is_empty())
            };
            let mut backs: Vec<(String, usize)> = Vec::new();
            for d in outputs.iter().filter(|d| d.kind == FeatureKind::MultiArray) {
                let shape = if d.shape.is_empty() {
                    match (&next_in, d.name.as_str()) {
                        (Some(s), "hidden_states") if s.len() == 3 => vec![1, t, s[2]],
                        (_, name) if name != "hidden_states" && later_in(name).is_some() => later_in(name).unwrap_or_default(),
                        _ => bail!("{fname}: output {} has no fixed shape", d.name),
                    }
                } else {
                    d.shape.clone()
                };
                self.arrays.push(output_array(&shape, d.dtype.unwrap_or(DataType::F16))?);
                backs.push((d.name.clone(), self.arrays.len() - 1));
            }
            prev = backs
                .iter()
                .find(|(n, _)| n == "output_hidden_states" || n == "hidden_states")
                .map(|&(_, i)| i);
            produced.extend(
                backs
                    .iter()
                    .filter(|(n, _)| n != "output_hidden_states" && n != "hidden_states")
                    .cloned(),
            );
            let ins: Vec<(&str, &MultiArray)> = bound.iter().map(|(n, i)| (n.as_str(), &self.arrays[*i])).collect();
            let outs: Vec<(&str, &MultiArray)> = backs.iter().map(|(n, i)| (n.as_str(), &self.arrays[*i])).collect();
            let prep = self.funcs[fi].model.prepare(&ins, &outs)?;
            if k + 1 == chain.len() {
                out = backs;
            }
            steps.push(Step {
                func: fi,
                prep,
                state,
                hooked,
            });
        }
        let has_head = head && self.lm_head.is_some();
        Ok(Plan {
            tokens: t,
            steps,
            has_head,
            out,
        })
    }

    /// Tokens currently in the KV cache.
    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn context_length(&self) -> usize {
        self.meta.context_length
    }

    /// Whether predictions are full logits (false: `argmax_in_model`, greedy only).
    pub fn has_logits(&self) -> bool {
        !self.infer_plan.out.iter().any(|(n, _)| n == "argmax_idx")
    }

    /// Whether the sliding-window rotate functions are in use.
    pub fn rotation(&self) -> bool {
        self.rotation
    }

    /// The loaded functions and their inputs / outputs / states, for diagnostics.
    pub fn describe(&self) -> String {
        let mut s = String::new();
        for f in &self.funcs {
            let fmt = |v: &[crate::coreml::FeatureDesc]| v.iter().map(|d| format!("{}{:?}", d.name, d.shape)).collect::<Vec<_>>().join(" ");
            s.push_str(&format!(
                "{}: in {} | out {} | state {}\n",
                f.name,
                fmt(&f.model.inputs),
                fmt(&f.model.outputs),
                fmt(&f.model.states)
            ));
        }
        s
    }

    /// Clears the KV cache (fresh states) and starts again at position 0.
    pub fn reset(&mut self) -> Result<()> {
        // Plans hold no state references; new states take effect on the next prediction.
        self.make_states()?;
        self.pos = 0;
        Ok(())
    }

    /// Room left in the context.
    pub fn remaining(&self) -> usize {
        self.meta.context_length.saturating_sub(self.pos)
    }

    /// Writes `tokens` into the cache from the current position and returns the prediction after
    /// the last one. Errors (and changes nothing) if they do not fit in the context.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Next> {
        let Some(&last) = tokens.last() else {
            bail!("prefill: no tokens");
        };
        let start = self.pos;
        let end = start + tokens.len();
        if end > self.meta.context_length {
            bail!(
                "context full: {} tokens in the cache + {} new > context length {} of this Core ML model",
                start,
                tokens.len(),
                self.meta.context_length
            );
        }
        if self.exact {
            return self.prefill_exact(tokens, start);
        }
        self.write_kv(tokens, start)?;
        // ANEMLL runs the last prompt token once more through `infer` for its prediction.
        self.pos = end - 1;
        self.step(last)
    }

    /// Prefill feeding every token exactly once (docs/coreml-format.md): full batches through
    /// `prefill` (ending on the last token only when its output feeds the head), the rest through
    /// `infer`, the prediction from the call that processed the last token.
    fn prefill_exact(&mut self, tokens: &[u32], start: usize) -> Result<Next> {
        let b = self.meta.batch_size;
        let end = start + tokens.len();
        let tok = |p: usize| &tokens[p - start..];
        let last_out = self.prefill_plan.has_head;
        let mut p = start;
        if self.options.prefill != PrefillMode::Single {
            while p + b <= end && (last_out || p + b < end) {
                let head = p + b == end;
                self.fill(&tok(p)[..b], p, b);
                self.exec(PlanKind::Prefill, p, !head)?;
                p += b;
            }
            if p == end {
                self.pos = end;
                return self.read_head();
            }
            // A padded batch for all but the last token, unless padding would enter a state that
            // cannot be rewound.
            let no_pad = self.meta.no_rewind() || self.meta.prefill_full_batches_only;
            if self.options.prefill == PrefillMode::Padded && !no_pad && p + 1 < end && p + b <= self.meta.context_length {
                let mut ids = tok(p)[..end - 1 - p].to_vec();
                let valid = ids.len();
                ids.resize(b, 0);
                self.fill(&ids, p, valid);
                self.exec(PlanKind::Prefill, p, true)?;
                p = end - 1;
            }
        }
        while p + 1 < end {
            self.single(tok(p)[0], p, false)?;
            p += 1;
        }
        self.pos = end - 1;
        self.step(tokens[tokens.len() - 1])
    }

    /// Feeds one token at the current position and returns the prediction for the next one.
    pub fn step(&mut self, token: u32) -> Result<Next> {
        let p = self.pos;
        if p >= self.meta.context_length {
            bail!(
                "context full: position {p} reaches the context length {} of this Core ML model",
                self.meta.context_length
            );
        }
        self.single(token, p, true)?;
        self.pos = p + 1;
        self.read_head()
    }

    /// Prefill of `tokens` at positions `start..` (no prediction).
    fn write_kv(&mut self, tokens: &[u32], start: usize) -> Result<()> {
        let b = self.meta.batch_size;
        let end = start + tokens.len();
        let tok = |p: usize| &tokens[p - start..];
        if self.options.prefill == PrefillMode::Single {
            for p in start..end {
                self.single(tok(p)[0], p, false)?;
            }
            return Ok(());
        }
        // Batches never straddle the window boundary (the rotate functions start there).
        let boundary = match self.meta.sliding_window {
            Some(sw) if self.rotation => sw,
            _ => usize::MAX,
        };
        let mut p = start;
        let fill_end = end.min(boundary);
        while p + b <= fill_end {
            self.batch(&tok(p)[..b], p, false)?;
            p += b;
        }
        // The last partial batch, padded with token 0: the pad positions' KV entries lie past
        // the prompt, hidden by the causal mask and overwritten by the next tokens. Not in
        // rotate mode, where the window would shift them in.
        let cap = self.meta.context_length.min(boundary);
        if self.options.prefill == PrefillMode::Padded && p < fill_end && p + b <= cap {
            let mut ids = tok(p)[..fill_end - p].to_vec();
            let valid = ids.len();
            ids.resize(b, 0);
            self.fill(&ids, p, valid);
            self.exec(PlanKind::Prefill, p, false)?;
            p = fill_end;
        }
        while p < fill_end {
            self.single(tok(p)[0], p, false)?;
            p += 1;
        }
        while p + b <= end {
            self.batch(&tok(p)[..b], p, true)?;
            p += b;
        }
        while p < end {
            self.single(tok(p)[0], p, false)?;
            p += 1;
        }
        Ok(())
    }

    fn use_rotate(&self, p: usize) -> bool {
        self.rotation && self.meta.sliding_window.is_some_and(|sw| p >= sw)
    }

    /// Writes the shared inputs for `ids` at positions `p..p+ids.len()`.
    /// `update_mask` marks only the first `valid` tokens (the rest are padding).
    fn fill(&mut self, ids: &[u32], p: usize, valid: usize) {
        let t = ids.len();
        for k in 0..self.shared.len() {
            let (name, shape, i) = (&self.shared[k].0, &self.shared[k].1, self.shared[k].2);
            let a = &mut self.arrays[i];
            match name.as_str() {
                "input_ids" if shape[1] == t => a.write_i32(&ids.iter().map(|&x| x as i32).collect::<Vec<_>>()),
                "position_ids" if shape[0] == t => a.write_i32(&(p..p + t).map(|x| x as i32).collect::<Vec<_>>()),
                "current_pos" => a.write_i32(&[p as i32]),
                "causal_mask" if shape[2] == t => {
                    let w = shape[3];
                    let mut bits = vec![F16_NEG_INF; t * w];
                    for r in 0..t {
                        bits[r * w..r * w + (p + r + 1).min(w)].fill(0);
                    }
                    a.write_f16_bits(&bits);
                }
                "update_mask" if shape[3] == t => {
                    let l = shape[2];
                    let mut bits = vec![0u16; l * t];
                    for r in 0..valid {
                        if p + r < l {
                            bits[(p + r) * t + r] = 0x3C00; // 1.0
                        }
                    }
                    a.write_f16_bits(&bits);
                }
                _ => {}
            }
        }
    }

    /// Runs a plan at position `p` (inputs already filled); `skip_head` leaves out the separate LM
    /// head (KV writes need no prediction).
    fn exec(&mut self, which: PlanKind, p: usize, skip_head: bool) -> Result<()> {
        let plan = match which {
            PlanKind::Infer => &self.infer_plan,
            PlanKind::InferRotate => self.infer_rotate_plan.as_ref().context("no infer_rotate")?,
            PlanKind::Prefill => &self.prefill_plan,
            PlanKind::PrefillRotate => self.prefill_rotate_plan.as_ref().context("no prefill_rotate")?,
        };
        let n = plan.steps.len() - usize::from(skip_head && plan.has_head);
        // Monolithic `infer*` functions are their own head.
        if n == plan.steps.len() && (plan.has_head || matches!(which, PlanKind::Infer | PlanKind::InferRotate)) {
            self.head_plan = which;
        }
        for step in &plan.steps[..n] {
            if !step.hooked.is_empty() {
                let hook = self.hook.as_mut().context("input hook")?;
                for (name, i) in &step.hooked {
                    if !hook.fill(&self.funcs[step.func].name, name, p, plan.tokens, &mut self.arrays[*i])? {
                        bail!("{}: no value for input {name}", self.funcs[step.func].name);
                    }
                }
            }
            self.funcs[step.func].model.run(&step.prep, step.state.map(|s| &self.states[s]))?;
            self.stats.predictions += 1;
        }
        Ok(())
    }

    /// One token at position `p` through `infer` (`infer_rotate` past the window), and the LM
    /// head when `head` (its outputs are then in the plan's arrays).
    fn single(&mut self, token: u32, p: usize, head: bool) -> Result<()> {
        self.fill(&[token], p, 1);
        let which = if self.use_rotate(p) {
            PlanKind::InferRotate
        } else {
            PlanKind::Infer
        };
        self.exec(which, p, !head)
    }

    /// A full batch at positions `p..p+batch` through the prefill functions.
    fn batch(&mut self, ids: &[u32], p: usize, rotate: bool) -> Result<()> {
        self.fill(ids, p, ids.len());
        self.exec(if rotate { PlanKind::PrefillRotate } else { PlanKind::Prefill }, p, false)
    }

    /// Logits or argmax from the head outputs of the last `infer` step.
    fn read_head(&self) -> Result<Next> {
        let plan = match self.head_plan {
            PlanKind::Prefill => &self.prefill_plan,
            PlanKind::InferRotate => self.infer_rotate_plan.as_ref().unwrap_or(&self.infer_plan),
            _ => &self.infer_plan,
        };
        let get = |name: &str| plan.out.iter().find(|(n, _)| n == name).map(|&(_, i)| &self.arrays[i]);
        if let (Some(idx), Some(val)) = (get("argmax_idx"), get("argmax_val")) {
            let (idx, val) = (idx.read_i32(), val.read_f32());
            let sizes = self.meta.argmax_layout(idx.len());
            let mut best = (0usize, f32::NEG_INFINITY);
            for (i, &v) in val.iter().enumerate() {
                if v > best.1 {
                    best = (i, v);
                }
            }
            let offset: usize = sizes[..best.0].iter().sum();
            let token = (idx[best.0].max(0) as usize + offset) as u32;
            return Ok(Next::Argmax { token, logit: best.1 });
        }
        let mut parts: Vec<(usize, usize)> = plan
            .out
            .iter()
            .filter_map(|(n, i)| n.strip_prefix("logits").and_then(|k| k.parse::<usize>().ok()).map(|k| (k, *i)))
            .collect();
        parts.sort();
        if parts.is_empty() {
            let i = ["logits", "output_logits"]
                .into_iter()
                .find_map(|n| plan.out.iter().find(|(o, _)| o == n).map(|&(_, i)| i))
                .ok_or_else(|| anyhow!("the head has no logits or argmax outputs"))?;
            parts.push((0, i));
        }
        let mut logits = Vec::with_capacity(self.meta.vocab_size.unwrap_or(0));
        for (_, i) in parts {
            self.arrays[i].read_f32_into(&mut logits);
        }
        Ok(Next::Logits(logits))
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PlanKind {
    Infer,
    InferRotate,
    Prefill,
    PrefillRotate,
}

#[cfg(test)]
mod tests {
    use super::yaml::{Value, parse};

    #[test]
    fn meta_yaml_subset() {
        let src = "model_info:\n  name: anemll-x\n  version: 0.3.5\n  description: |\n    Line one\n    Context length: 512\n  summary: 'Two\n\n    lines\n\n    '\n  license: MIT\n  framework: Core ML\n  model_type: monolithic\n  parameters:\n    context_length: 4096\n    lut_ffn: 6\n    lut_embeddings: none\n    argmax_in_model: true\n    lm_head_chunk_sizes: [16384, 16384]\n    quoted: \"a: b # c\"\n    functions:\n      - infer\n      - prefill\n    sizes:\n    - 3\n    - 4\n    recommended_sampling:\n      temperature: 0.7\n      top_k: 64\n\n# trailing comment\n# key: not a key\n";
        let v = parse(src).unwrap();
        let info = v.get("model_info").unwrap();
        assert_eq!(info.get("version"), Some(&Value::Str("0.3.5".into())));
        assert_eq!(
            info.get("description").and_then(Value::as_str),
            Some("Line one\nContext length: 512\n")
        );
        assert_eq!(info.get("summary").and_then(Value::as_str), Some("Two\nlines\n"));
        assert_eq!(info.get("framework").and_then(Value::as_str), Some("Core ML"));
        let p = info.get("parameters").unwrap();
        assert_eq!(p.get("context_length").and_then(Value::as_usize), Some(4096));
        assert_eq!(p.get("lut_embeddings").and_then(Value::as_str), Some("none"));
        assert_eq!(p.get("argmax_in_model").and_then(Value::as_bool), Some(true));
        assert_eq!(
            p.get("lm_head_chunk_sizes").and_then(Value::as_usize_list),
            Some(vec![16384, 16384])
        );
        assert_eq!(p.get("quoted").and_then(Value::as_str), Some("a: b # c"));
        assert_eq!(p.get("sizes").and_then(Value::as_usize_list), Some(vec![3, 4]));
        assert_eq!(
            p.get("functions"),
            Some(&Value::List(vec![Value::Str("infer".into()), Value::Str("prefill".into())]))
        );
        let rs = p.get("recommended_sampling").unwrap();
        assert_eq!(rs.get("temperature").and_then(Value::as_f64), Some(0.7));
        assert_eq!(Value::Str("[1, 2,3]".into()).as_usize_list(), Some(vec![1, 2, 3]));
    }

    #[test]
    fn chunk_names() {
        assert_eq!(
            super::chunk_parts("qwen_FFN_PF_lut6_chunk_02of04.mlmodelc"),
            Some(("qwen_FFN_PF_lut6".into(), 2, 4, ".mlmodelc".into()))
        );
        assert_eq!(
            super::chunk_parts("g_FFN_lut6_chunk_01of02_combined.mlmodelc"),
            Some(("g_FFN_lut6".into(), 1, 2, "_combined.mlmodelc".into()))
        );
    }
}
