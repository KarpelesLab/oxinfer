//! Text tokenization from HuggingFace `tokenizer.json` files, implemented in-crate.
//!
//! [`Tokenizer`] loads the JSON format the HuggingFace `tokenizers` library writes and encodes /
//! decodes the way that library does: added tokens are split out first (the non-normalized ones
//! on the raw text, the normalized ones after normalization), the rest goes through the
//! normalizer, the pre-tokenizer and the model (BPE, Unigram, WordPiece or WordLevel), then the
//! post-processor adds the special tokens. Supported components:
//!
//! - normalizers: NFC / NFD / NFKC / NFKD, Lowercase, Strip, StripAccents, Replace, Prepend,
//!   BertNormalizer, Precompiled (SentencePiece charsmaps), Nmt, ByteLevel, Sequence;
//! - pre-tokenizers: ByteLevel, Split (string or regex, every split behavior), Whitespace,
//!   WhitespaceSplit, BertPreTokenizer, Metaspace, Digits, Punctuation, CharDelimiterSplit,
//!   FixedLength, Sequence;
//! - models: BPE (byte fallback, unknown-token fusing, `ignore_merges`, subword prefix / suffix),
//!   Unigram (Viterbi, byte fallback), WordPiece, WordLevel;
//! - post-processors: TemplateProcessing, RobertaProcessing, BertProcessing, ByteLevel, Sequence;
//! - decoders: ByteLevel, Metaspace, WordPiece, Replace, ByteFallback, Fuse, Strip, BPEDecoder,
//!   CTC, Sequence.
//!
//! Regex patterns run on a small backtracking engine ([`regex`]) that supports the lookahead the
//! common split patterns use. Offsets are not tracked: encodings carry ids and the special-token
//! mask.

mod decode;
mod models;
mod normalize;
mod pretok;
pub mod regex;
mod unicode;

use anyhow::{Context, Result, anyhow, bail};
use decode::{Decoder, PostProcessor};
use models::{Model, ModelJson};
use normalize::Normalizer;
use pretok::{Piece, PreTokenizer};
use serde::Deserialize;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// A token added on top of the model vocabulary (special tokens, chat markers, ...).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AddedToken {
    pub content: String,
    /// Only matched as a whole word.
    pub single_word: bool,
    /// Swallows the whitespace before it.
    pub lstrip: bool,
    /// Swallows the whitespace after it.
    pub rstrip: bool,
    /// Matched on the normalized text (else on the raw input).
    pub normalized: bool,
    /// A special token (skipped by `decode(.., true)`).
    pub special: bool,
}

#[derive(Deserialize)]
struct AddedTokenJson {
    id: u32,
    content: String,
    #[serde(default)]
    single_word: bool,
    #[serde(default)]
    lstrip: bool,
    #[serde(default)]
    rstrip: bool,
    #[serde(default = "yes")]
    normalized: bool,
    #[serde(default)]
    special: bool,
}

fn yes() -> bool {
    true
}

#[derive(Deserialize)]
struct FileJson {
    #[serde(default)]
    added_tokens: Vec<AddedTokenJson>,
    #[serde(default)]
    normalizer: Option<Value>,
    #[serde(default)]
    pre_tokenizer: Option<Value>,
    #[serde(default)]
    post_processor: Option<Value>,
    #[serde(default)]
    decoder: Option<Value>,
    #[serde(default)]
    truncation: Option<Value>,
    #[serde(default)]
    padding: Option<Value>,
    model: ModelJson,
}

/// Which end truncation and padding apply to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Direction {
    Left,
    #[default]
    Right,
}

/// Truncation of encodings to at most `max_length` ids (special tokens included).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TruncationParams {
    pub max_length: usize,
    pub direction: Direction,
}

impl Default for TruncationParams {
    fn default() -> Self {
        TruncationParams {
            max_length: 512,
            direction: Direction::Right,
        }
    }
}

/// Padding of single encodings: to a fixed length (`length`), rounded up to `pad_to_multiple_of`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PaddingParams {
    /// None: pad to the encoding's own length (only `pad_to_multiple_of` then applies).
    pub length: Option<usize>,
    pub pad_to_multiple_of: Option<usize>,
    pub pad_id: u32,
    pub direction: Direction,
}

impl Default for PaddingParams {
    fn default() -> Self {
        PaddingParams {
            length: None,
            pad_to_multiple_of: None,
            pad_id: 0,
            direction: Direction::Right,
        }
    }
}

/// The result of [`Tokenizer::encode`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Encoding {
    ids: Vec<u32>,
    special: Vec<u32>,
    attention: Vec<u32>,
}

impl Encoding {
    pub fn get_ids(&self) -> &[u32] {
        &self.ids
    }

    /// 1 for the tokens the post-processor or padding added, 0 for the input's.
    pub fn get_special_tokens_mask(&self) -> &[u32] {
        &self.special
    }

    /// 0 for padding, 1 elsewhere.
    pub fn get_attention_mask(&self) -> &[u32] {
        &self.attention
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

/// Leftmost-longest matching over a set of strings (the added tokens).
#[derive(Clone)]
struct Matcher {
    /// (node, byte) -> node; node 0 is the root.
    next: HashMap<(u32, u8), u32>,
    /// node -> added-token id for nodes that end a token.
    end: HashMap<u32, u32>,
    first: [bool; 256],
    nodes: u32,
}

impl Default for Matcher {
    fn default() -> Self {
        Matcher {
            next: HashMap::new(),
            end: HashMap::new(),
            first: [false; 256],
            nodes: 1,
        }
    }
}

impl Matcher {
    fn new<'a>(words: impl IntoIterator<Item = (&'a str, u32)>) -> Matcher {
        let mut m = Matcher::default();
        for (w, id) in words {
            if w.is_empty() {
                continue;
            }
            m.first[w.as_bytes()[0] as usize] = true;
            let mut n = 0u32;
            for &b in w.as_bytes() {
                n = match m.next.get(&(n, b)) {
                    Some(&x) => x,
                    None => {
                        let x = m.nodes;
                        m.nodes += 1;
                        m.next.insert((n, b), x);
                        x
                    }
                };
            }
            m.end.insert(n, id);
        }
        m
    }

    fn is_empty(&self) -> bool {
        self.end.is_empty()
    }

    /// The leftmost-longest match at or after `from`: (start, end, id).
    fn find(&self, s: &[u8], from: usize) -> Option<(usize, usize, u32)> {
        for i in from..s.len() {
            if !self.first[s[i] as usize] {
                continue;
            }
            let mut n = 0u32;
            let mut best = None;
            for (k, &b) in s[i..].iter().enumerate() {
                match self.next.get(&(n, b)) {
                    Some(&x) => n = x,
                    None => break,
                }
                if let Some(&id) = self.end.get(&n) {
                    best = Some((i, i + k + 1, id));
                }
            }
            if best.is_some() {
                return best;
            }
        }
        None
    }
}

/// A piece of the input: text still to tokenize, or an added token already identified.
enum Seg {
    Text(String, bool),
    Token(u32),
}

/// A tokenizer loaded from a `tokenizer.json`.
#[derive(Clone)]
pub struct Tokenizer {
    model: Model,
    normalizer: Option<Normalizer>,
    pre_tokenizer: Option<PreTokenizer>,
    post_processor: Option<PostProcessor>,
    decoder: Option<Decoder>,
    added: HashMap<u32, AddedToken>,
    added_ids: HashMap<String, u32>,
    /// Normalized forms of normalized added tokens (when they differ from the content).
    normalized_cache: HashMap<u32, String>,
    special: HashSet<String>,
    raw_matcher: Matcher,
    norm_matcher: Matcher,
    truncation: Option<TruncationParams>,
    padding: Option<PaddingParams>,
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokenizer")
            .field("vocab_size", &self.model.vocab_size())
            .field("added_tokens", &self.added.len())
            .finish_non_exhaustive()
    }
}

fn direction(v: &Value) -> Direction {
    if v.as_str() == Some("Left") {
        Direction::Left
    } else {
        Direction::Right
    }
}

impl Tokenizer {
    /// Loads a `tokenizer.json` file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Tokenizer> {
        let path = path.as_ref();
        Self::from_bytes(crate::fsx::read(path)?).with_context(|| format!("tokenizer {}", path.display()))
    }

    /// Parses the contents of a `tokenizer.json`.
    pub fn from_bytes(bytes: impl AsRef<[u8]>) -> Result<Tokenizer> {
        let f: FileJson = serde_json::from_slice(bytes.as_ref()).context("tokenizer.json")?;
        let some = |v: Option<Value>| v.filter(|v| !v.is_null());
        let model = Model::from_json(f.model)?;
        let normalizer = some(f.normalizer).map(|v| Normalizer::from_json(&v)).transpose()?;
        let pre_tokenizer = some(f.pre_tokenizer).map(|v| PreTokenizer::from_json(&v)).transpose()?;
        let post_processor = some(f.post_processor).map(|v| PostProcessor::from_json(&v)).transpose()?;
        let decoder = some(f.decoder).map(|v| Decoder::from_json(&v)).transpose()?;
        let truncation = some(f.truncation).map(|t| TruncationParams {
            max_length: t["max_length"].as_u64().unwrap_or(512) as usize,
            direction: direction(&t["direction"]),
        });
        let padding = some(f.padding).map(|p| PaddingParams {
            length: p["strategy"]["Fixed"].as_u64().map(|n| n as usize),
            pad_to_multiple_of: p["pad_to_multiple_of"].as_u64().map(|n| n as usize),
            pad_id: p["pad_id"].as_u64().unwrap_or(0) as u32,
            direction: direction(&p["direction"]),
        });
        let mut t = Tokenizer {
            model,
            normalizer,
            pre_tokenizer,
            post_processor,
            decoder,
            added: HashMap::new(),
            added_ids: HashMap::new(),
            normalized_cache: HashMap::new(),
            special: HashSet::new(),
            raw_matcher: Matcher::default(),
            norm_matcher: Matcher::default(),
            truncation,
            padding,
        };
        for a in f.added_tokens {
            if a.content.is_empty() {
                continue;
            }
            let id = t.token_to_id(&a.content).unwrap_or(a.id);
            if a.normalized
                && let Some(n) = &t.normalizer
            {
                let normed = n.normalize(a.content.clone());
                if normed != a.content {
                    t.normalized_cache.insert(id, normed);
                }
            }
            if a.special {
                t.special.insert(a.content.clone());
            }
            t.added_ids.insert(a.content.clone(), id);
            t.added.insert(
                id,
                AddedToken {
                    content: a.content,
                    single_word: a.single_word,
                    lstrip: a.lstrip,
                    rstrip: a.rstrip,
                    normalized: a.normalized,
                    special: a.special,
                },
            );
        }
        t.raw_matcher = Matcher::new(
            t.added
                .iter()
                .filter(|(_, a)| !a.normalized)
                .map(|(&id, a)| (a.content.as_str(), id)),
        );
        t.norm_matcher = Matcher::new(
            t.added
                .iter()
                .filter(|(_, a)| a.normalized)
                .map(|(&id, a)| (t.normalized_cache.get(&id).map(String::as_str).unwrap_or(a.content.as_str()), id)),
        );
        Ok(t)
    }

    /// Sets (or clears) truncation.
    pub fn with_truncation(&mut self, t: Option<TruncationParams>) -> Result<&mut Self> {
        if let Some(t) = &t
            && t.max_length == 0
        {
            bail!("truncation max_length must be positive");
        }
        self.truncation = t;
        Ok(self)
    }

    /// Sets (or clears) padding.
    pub fn with_padding(&mut self, p: Option<PaddingParams>) -> &mut Self {
        self.padding = p;
        self
    }

    /// The id of a token (added tokens first, then the model vocabulary).
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.added_ids.get(token).copied().or_else(|| self.model.token_to_id(token))
    }

    /// The text of a token id (an added token's normalized form when it has one).
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.token_str(id).map(str::to_string)
    }

    fn token_str(&self, id: u32) -> Option<&str> {
        match self.added.get(&id) {
            Some(a) => Some(self.normalized_cache.get(&id).map(String::as_str).unwrap_or(&a.content)),
            None => self.model.id_to_token(id),
        }
    }

    /// The vocabulary size: the model's, plus the added tokens it does not already contain.
    pub fn get_vocab_size(&self, with_added_tokens: bool) -> usize {
        let base = self.model.vocab_size();
        if !with_added_tokens {
            return base;
        }
        base + self.added_ids.keys().filter(|t| self.model.token_to_id(t).is_none()).count()
    }

    /// The added tokens by id.
    pub fn get_added_tokens_decoder(&self) -> HashMap<u32, AddedToken> {
        self.added.clone()
    }

    /// Splits `s` at the added tokens `m` finds (honoring single_word / lstrip / rstrip).
    fn split_added(&self, s: &str, first: bool, m: &Matcher, out: &mut Vec<Seg>) {
        if m.is_empty() {
            out.push(Seg::Text(s.to_string(), first));
            return;
        }
        let b = s.as_bytes();
        let mut done = 0;
        let mut pos = 0;
        while let Some((mut start, mut stop, id)) = m.find(b, pos) {
            pos = stop;
            let a = &self.added[&id];
            if a.single_word {
                let start_ok = start == 0 || !s[..start].chars().next_back().is_some_and(unicode::is_word_char);
                let stop_ok = stop == s.len() || !s[stop..].chars().next().is_some_and(unicode::is_word_char);
                if !start_ok || !stop_ok {
                    continue;
                }
            }
            if a.lstrip {
                let ws = s[..start].trim_end_matches(char::is_whitespace).len();
                start = ws.max(done);
            }
            if a.rstrip {
                // The search resumes at the token's own end, as in the reference implementation.
                stop = s.len() - s[stop..].trim_start_matches(char::is_whitespace).len();
            }
            if done < start {
                out.push(Seg::Text(s[done..start].to_string(), first && done == 0));
            }
            out.push(Seg::Token(id));
            done = stop;
        }
        if done < s.len() || s.is_empty() {
            out.push(Seg::Text(s[done..].to_string(), first && done == 0));
        }
    }

    /// Encodes `text`; `add_special_tokens` lets the post-processor add its tokens (BOS, ...).
    pub fn encode(&self, text: impl AsRef<str>, add_special_tokens: bool) -> Result<Encoding> {
        let text = text.as_ref();
        let mut raw = Vec::new();
        self.split_added(text, true, &self.raw_matcher, &mut raw);
        let mut ids = Vec::with_capacity(text.len() / 3 + 4);
        for seg in raw {
            match seg {
                Seg::Token(id) => ids.push(id),
                Seg::Text(s, first) => {
                    let s = match &self.normalizer {
                        Some(n) => n.normalize(s),
                        None => s,
                    };
                    let mut segs = Vec::new();
                    self.split_added(&s, first, &self.norm_matcher, &mut segs);
                    for seg in segs {
                        match seg {
                            Seg::Token(id) => ids.push(id),
                            Seg::Text(s, first) => {
                                if s.is_empty() {
                                    continue;
                                }
                                let pieces = vec![Piece { s, first }];
                                let pieces = match &self.pre_tokenizer {
                                    Some(p) => p.apply(pieces),
                                    None => pieces,
                                };
                                for p in pieces {
                                    self.model.tokenize(&p.s, &mut ids)?;
                                }
                            }
                        }
                    }
                }
            }
        }
        if let Some(t) = &self.truncation {
            let added = match &self.post_processor {
                Some(p) if add_special_tokens => p.added_tokens(),
                _ => 0,
            };
            let max = t.max_length.saturating_sub(added);
            if ids.len() > max {
                match t.direction {
                    Direction::Right => ids.truncate(max),
                    Direction::Left => {
                        ids.drain(..ids.len() - max);
                    }
                }
            }
        }
        let mask = vec![0; ids.len()];
        let (mut ids, mut special) = match &self.post_processor {
            Some(p) if add_special_tokens => p.process(ids, mask),
            _ => (ids, mask),
        };
        let mut attention = vec![1; ids.len()];
        if let Some(p) = &self.padding {
            let mut target = p.length.unwrap_or(ids.len()).max(ids.len());
            if let Some(m) = p.pad_to_multiple_of.filter(|&m| m > 0) {
                target = target.div_ceil(m) * m;
            }
            let n = target - ids.len();
            if n > 0 {
                let pad = |v: &mut Vec<u32>, x: u32| match p.direction {
                    Direction::Right => v.extend(std::iter::repeat_n(x, n)),
                    Direction::Left => {
                        v.splice(0..0, std::iter::repeat_n(x, n));
                    }
                };
                pad(&mut ids, p.pad_id);
                pad(&mut special, 1);
                pad(&mut attention, 0);
            }
        }
        Ok(Encoding { ids, special, attention })
    }

    /// Decodes ids to text; `skip_special_tokens` drops the special added tokens.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let tokens: Vec<String> = ids
            .iter()
            .filter_map(|&id| self.token_str(id))
            .filter(|t| !skip_special_tokens || !self.special.contains(*t))
            .map(str::to_string)
            .collect();
        Ok(match &self.decoder {
            Some(d) => d.decode(tokens),
            None => tokens.join(" "),
        })
    }
}

impl std::str::FromStr for Tokenizer {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Tokenizer> {
        Tokenizer::from_bytes(s.as_bytes())
    }
}

/// Builds a tokenizer from a JSON value (as the GGUF and vocabulary converters produce).
pub fn from_value(v: &Value) -> Result<Tokenizer> {
    Tokenizer::from_bytes(serde_json::to_vec(v).map_err(|e| anyhow!("tokenizer json: {e}"))?)
}

#[cfg(test)]
mod tests;
