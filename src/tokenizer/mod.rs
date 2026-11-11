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
//!   FixedLength, UnicodeScripts, Sequence;
//! - models: BPE (byte fallback, unknown-token fusing, `ignore_merges`, dropout, subword prefix /
//!   suffix), Unigram (Viterbi, byte fallback), WordPiece, WordLevel;
//! - post-processors: TemplateProcessing, RobertaProcessing, BertProcessing, ByteLevel, Sequence;
//! - decoders: ByteLevel, Metaspace, WordPiece, Replace, ByteFallback, Fuse, Strip, BPEDecoder,
//!   CTC, Sequence;
//! - truncation (longest-first / only-first / only-second, stride, overflowing parts) and padding
//!   (fixed or batch-longest, `pad_to_multiple_of`), added tokens with every flag.
//!
//! Inputs can be single sequences or pairs, raw or pre-split into words ([`EncodeInput`]); there
//! are batch encode / decode and a streaming decoder ([`DecodeStream`]).
//!
//! Regex patterns run on [`regex`], a backtracking engine with the syntax and semantics of
//! `fancy-regex` (the reference's engine): lookaround including variable-length lookbehind,
//! atomic groups, possessive quantifiers, backreferences, conditionals, `\K` and `\G`. Offsets
//! are not tracked: encodings carry ids, type ids, the special-token and attention masks.

mod decode;
mod models;
mod normalize;
mod pretok;
pub mod regex;
mod scripts;
mod unicode;

use anyhow::{Context, Result, anyhow, bail};
use decode::{Decoder, PostProcessor};
use models::{Model, ModelJson};
use normalize::Normalizer;
use pretok::{Piece, PreTokenizer};
use serde::Deserialize;
use serde_json::Value;
use std::borrow::Cow;
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
    #[allow(dead_code)]
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

/// How a pair of sequences is truncated.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TruncationStrategy {
    /// Shorten the longer sequence first (balanced when both are long).
    #[default]
    LongestFirst,
    /// Only the first sequence (an error when it is too short to reach the length).
    OnlyFirst,
    /// Only the second sequence.
    OnlySecond,
}

/// Truncation of encodings to at most `max_length` ids (special tokens included). The cut-off
/// parts become overflowing encodings, consecutive ones sharing `stride` ids.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TruncationParams {
    pub max_length: usize,
    pub direction: Direction,
    pub strategy: TruncationStrategy,
    pub stride: usize,
}

impl Default for TruncationParams {
    fn default() -> Self {
        TruncationParams {
            max_length: 512,
            direction: Direction::Right,
            strategy: TruncationStrategy::LongestFirst,
            stride: 0,
        }
    }
}

/// Padding of encodings: to a fixed length (`length`), or to the longest of a batch (`None`),
/// rounded up to `pad_to_multiple_of`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PaddingParams {
    /// None: pad to the batch's longest encoding (a single encoding: its own length).
    pub length: Option<usize>,
    pub pad_to_multiple_of: Option<usize>,
    pub pad_id: u32,
    pub pad_type_id: u32,
    pub pad_token: String,
    pub direction: Direction,
}

impl Default for PaddingParams {
    fn default() -> Self {
        PaddingParams {
            length: None,
            pad_to_multiple_of: None,
            pad_id: 0,
            pad_type_id: 0,
            pad_token: "[PAD]".into(),
            direction: Direction::Right,
        }
    }
}

/// The result of [`Tokenizer::encode`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Encoding {
    pub(crate) ids: Vec<u32>,
    pub(crate) type_ids: Vec<u32>,
    pub(crate) special: Vec<u32>,
    pub(crate) attention: Vec<u32>,
    pub(crate) overflowing: Vec<Encoding>,
}

impl Encoding {
    fn new(ids: Vec<u32>, type_id: u32) -> Encoding {
        let n = ids.len();
        Encoding {
            ids,
            type_ids: vec![type_id; n],
            special: vec![0; n],
            attention: vec![1; n],
            overflowing: Vec::new(),
        }
    }

    pub fn get_ids(&self) -> &[u32] {
        &self.ids
    }

    /// The sequence type of each token (0 for the first sequence, 1 for the pair, as the
    /// post-processor assigns them).
    pub fn get_type_ids(&self) -> &[u32] {
        &self.type_ids
    }

    /// 1 for the tokens the post-processor or padding added, 0 for the input's.
    pub fn get_special_tokens_mask(&self) -> &[u32] {
        &self.special
    }

    /// 0 for padding, 1 elsewhere.
    pub fn get_attention_mask(&self) -> &[u32] {
        &self.attention
    }

    /// The parts truncation cut off (each post-processed and padded like the encoding).
    pub fn get_overflowing(&self) -> &[Encoding] {
        &self.overflowing
    }

    pub fn take_overflowing(&mut self) -> Vec<Encoding> {
        std::mem::take(&mut self.overflowing)
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    fn slice(&self, a: usize, b: usize) -> Encoding {
        Encoding {
            ids: self.ids[a..b].to_vec(),
            type_ids: self.type_ids[a..b].to_vec(),
            special: self.special[a..b].to_vec(),
            attention: self.attention[a..b].to_vec(),
            overflowing: Vec::new(),
        }
    }

    /// Keeps `max_len` ids; the rest becomes overflowing windows of `max_len` overlapping by
    /// `stride`.
    fn truncate(&mut self, max_len: usize, stride: usize, direction: Direction) -> Result<()> {
        let n = self.ids.len();
        if max_len >= n {
            return Ok(());
        }
        if max_len == 0 {
            let o = std::mem::take(self);
            self.overflowing.push(o);
            return Ok(());
        }
        if stride >= max_len {
            bail!("truncation stride {stride} must be smaller than the length {max_len} it applies to");
        }
        let step = max_len - stride;
        let mut parts = Vec::new();
        match direction {
            Direction::Right => {
                for start in (0..n).step_by(step) {
                    let stop = (start + max_len).min(n);
                    parts.push((start, stop));
                    if stop == n {
                        break;
                    }
                }
            }
            Direction::Left => {
                for stop in (0..n).rev().step_by(step) {
                    let stop = stop + 1;
                    let start = stop.saturating_sub(max_len);
                    if start < stop {
                        parts.push((start, stop));
                        if start == 0 {
                            break;
                        }
                    }
                }
            }
        }
        let mut first = self.slice(parts[0].0, parts[0].1);
        first.overflowing = parts[1..].iter().map(|&(a, b)| self.slice(a, b)).collect();
        *self = first;
        Ok(())
    }

    /// Appends `other`; overflowing parts combine (ours with `other`, ours with its
    /// overflowing ones, us with its overflowing ones).
    fn merge_with(&mut self, other: Encoding) {
        let mut over = Vec::new();
        for o in &self.overflowing {
            let mut x = o.clone();
            x.merge_with(other.clone());
            over.push(x);
            for oo in &other.overflowing {
                let mut x = o.clone();
                x.merge_with(oo.clone());
                over.push(x);
            }
        }
        for oo in &other.overflowing {
            let mut x = self.clone();
            x.merge_with(oo.clone());
            over.push(x);
        }
        self.ids.extend(other.ids);
        self.type_ids.extend(other.type_ids);
        self.special.extend(other.special);
        self.attention.extend(other.attention);
        self.overflowing = over;
    }

    pub(crate) fn merge(encodings: impl IntoIterator<Item = Encoding>) -> Encoding {
        let mut e = Encoding::default();
        for x in encodings {
            e.merge_with(x);
        }
        e
    }

    /// Pads to `target` ids (the overflowing parts too).
    fn pad(&mut self, target: usize, p: &PaddingParams) {
        for o in &mut self.overflowing {
            o.pad(target, p);
        }
        if self.ids.len() >= target {
            return;
        }
        let n = target - self.ids.len();
        let pad = |v: &mut Vec<u32>, x: u32| match p.direction {
            Direction::Right => v.extend(std::iter::repeat_n(x, n)),
            Direction::Left => {
                v.splice(0..0, std::iter::repeat_n(x, n));
            }
        };
        pad(&mut self.ids, p.pad_id);
        pad(&mut self.type_ids, p.pad_type_id);
        pad(&mut self.special, 1);
        pad(&mut self.attention, 0);
    }
}

/// Pads every encoding to the fixed length or the longest one, rounded up to the multiple.
fn pad_encodings(encodings: &mut [Encoding], p: &PaddingParams) {
    let Some(longest) = encodings.iter().map(Encoding::len).max() else {
        return;
    };
    let mut target = p.length.unwrap_or(longest);
    if let Some(m) = p.pad_to_multiple_of
        && m > 0
        && !target.is_multiple_of(m)
    {
        target += m - target % m;
    }
    for e in encodings {
        e.pad(target, p);
    }
}

/// Truncates an encoding (and its pair) to `params.max_length` ids in total.
fn truncate_encodings(encoding: &mut Encoding, pair: Option<&mut Encoding>, t: &TruncationParams) -> Result<()> {
    if t.max_length == 0 {
        encoding.truncate(0, t.stride, t.direction)?;
        if let Some(p) = pair {
            p.truncate(0, t.stride, t.direction)?;
        }
        return Ok(());
    }
    let total = encoding.len() + pair.as_ref().map_or(0, |p| p.len());
    if total <= t.max_length {
        return Ok(());
    }
    let to_remove = total - t.max_length;
    match t.strategy {
        TruncationStrategy::LongestFirst => match pair {
            Some(p) => {
                let (mut n1, mut n2) = (encoding.len(), p.len());
                let swap = n1 > n2;
                if swap {
                    std::mem::swap(&mut n1, &mut n2);
                }
                n2 = if n1 > t.max_length { n1 } else { n1.max(t.max_length - n1) };
                if n1 + n2 > t.max_length {
                    n1 = t.max_length / 2;
                    n2 = n1 + t.max_length % 2;
                }
                if swap {
                    std::mem::swap(&mut n1, &mut n2);
                }
                encoding.truncate(n1, t.stride, t.direction)?;
                p.truncate(n2, t.stride, t.direction)?;
            }
            None => encoding.truncate(total - to_remove, t.stride, t.direction)?,
        },
        TruncationStrategy::OnlyFirst | TruncationStrategy::OnlySecond => {
            let target = if t.strategy == TruncationStrategy::OnlyFirst {
                encoding
            } else {
                pair.ok_or_else(|| anyhow!("truncation: the second sequence was not provided"))?
            };
            let n = target.len();
            if n <= to_remove {
                bail!("truncation: the sequence to truncate is too short to respect max_length");
            }
            target.truncate(n - to_remove, t.stride, t.direction)?;
        }
    }
    Ok(())
}

/// What [`Tokenizer::encode`] accepts: a text, or a text already split into words (each one
/// tokenized on its own, as `is_split_into_words` inputs are).
pub enum InputSequence<'a> {
    Raw(Cow<'a, str>),
    PreTokenized(Vec<Cow<'a, str>>),
}

impl<'a> From<&'a str> for InputSequence<'a> {
    fn from(s: &'a str) -> Self {
        InputSequence::Raw(Cow::Borrowed(s))
    }
}

impl<'a> From<&'a &'a str> for InputSequence<'a> {
    fn from(s: &'a &'a str) -> Self {
        InputSequence::Raw(Cow::Borrowed(s))
    }
}

impl<'a> From<&'a String> for InputSequence<'a> {
    fn from(s: &'a String) -> Self {
        InputSequence::Raw(Cow::Borrowed(s))
    }
}

impl From<String> for InputSequence<'_> {
    fn from(s: String) -> Self {
        InputSequence::Raw(Cow::Owned(s))
    }
}

impl<'a> From<&'a [&'a str]> for InputSequence<'a> {
    fn from(s: &'a [&'a str]) -> Self {
        InputSequence::PreTokenized(s.iter().map(|&w| Cow::Borrowed(w)).collect())
    }
}

impl<'a> From<Vec<&'a str>> for InputSequence<'a> {
    fn from(s: Vec<&'a str>) -> Self {
        InputSequence::PreTokenized(s.into_iter().map(Cow::Borrowed).collect())
    }
}

impl<'a> From<&'a [String]> for InputSequence<'a> {
    fn from(s: &'a [String]) -> Self {
        InputSequence::PreTokenized(s.iter().map(|w| Cow::Borrowed(w.as_str())).collect())
    }
}

impl From<Vec<String>> for InputSequence<'_> {
    fn from(s: Vec<String>) -> Self {
        InputSequence::PreTokenized(s.into_iter().map(Cow::Owned).collect())
    }
}

/// A single sequence or a pair (`(first, second)`), for [`Tokenizer::encode`].
pub enum EncodeInput<'a> {
    Single(InputSequence<'a>),
    Dual(InputSequence<'a>, InputSequence<'a>),
}

impl<'a, I: Into<InputSequence<'a>>> From<I> for EncodeInput<'a> {
    fn from(i: I) -> Self {
        EncodeInput::Single(i.into())
    }
}

impl<'a, A: Into<InputSequence<'a>>, B: Into<InputSequence<'a>>> From<(A, B)> for EncodeInput<'a> {
    fn from((a, b): (A, B)) -> Self {
        EncodeInput::Dual(a.into(), b.into())
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
    Text(Piece),
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
    /// Special added tokens in the input are tokenized as plain text.
    encode_special_tokens: bool,
    /// Whether pieces need to know if they start at the input's start through the normalizer
    /// (Metaspace `prepend_scheme: first`).
    track_start: bool,
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
        let truncation = some(f.truncation)
            .map(|t| -> Result<TruncationParams> {
                Ok(TruncationParams {
                    max_length: t["max_length"].as_u64().unwrap_or(512) as usize,
                    direction: direction(&t["direction"]),
                    strategy: match t["strategy"].as_str().unwrap_or("LongestFirst") {
                        "LongestFirst" => TruncationStrategy::LongestFirst,
                        "OnlyFirst" => TruncationStrategy::OnlyFirst,
                        "OnlySecond" => TruncationStrategy::OnlySecond,
                        s => bail!("unsupported truncation strategy {s}"),
                    },
                    stride: t["stride"].as_u64().unwrap_or(0) as usize,
                })
            })
            .transpose()?;
        let padding = some(f.padding).map(|p| PaddingParams {
            length: p["strategy"]["Fixed"].as_u64().map(|n| n as usize),
            pad_to_multiple_of: p["pad_to_multiple_of"].as_u64().map(|n| n as usize),
            pad_id: p["pad_id"].as_u64().unwrap_or(0) as u32,
            pad_type_id: p["pad_type_id"].as_u64().unwrap_or(0) as u32,
            pad_token: p["pad_token"].as_str().unwrap_or("[PAD]").to_string(),
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
            encode_special_tokens: false,
            track_start: false,
        };
        t.track_start = t.pre_tokenizer.as_ref().is_some_and(PreTokenizer::uses_first);
        // Ids as the reference assigns them (the file's ids are not used): a token in the
        // vocabulary (or added before) keeps that id, the others are numbered from the
        // vocabulary size on.
        let mut next_id = t.model.vocab_size() as u32;
        for a in f.added_tokens {
            if a.content.is_empty() {
                continue;
            }
            let tok = AddedToken {
                content: a.content,
                single_word: a.single_word,
                lstrip: a.lstrip,
                rstrip: a.rstrip,
                normalized: a.normalized,
                special: a.special,
            };
            if let Some(id) = t.added_ids.get(&tok.content)
                && t.added.get(id) == Some(&tok)
            {
                continue;
            }
            let id = match t.token_to_id(&tok.content) {
                Some(id) => id,
                None => {
                    next_id += 1;
                    next_id - 1
                }
            };
            if tok.normalized
                && let Some(n) = &t.normalizer
            {
                let normed = n.normalize(tok.content.clone());
                if normed != tok.content {
                    t.normalized_cache.insert(id, normed);
                }
            }
            if tok.special {
                t.special.insert(tok.content.clone());
            }
            t.added_ids.insert(tok.content.clone(), id);
            t.added.insert(id, tok);
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

    /// Sets (or clears) truncation (an error when the stride does not fit in the length left
    /// once the special tokens are counted).
    pub fn with_truncation(&mut self, t: Option<TruncationParams>) -> Result<&mut Self> {
        if let Some(t) = &t {
            let added = self.post_processor.as_ref().map_or(0, |p| p.added_tokens(false));
            match t.max_length.checked_sub(added) {
                Some(n) if n >= t.stride => {}
                _ => bail!(
                    "truncation stride {} does not fit in max_length {} minus {added} special tokens",
                    t.stride,
                    t.max_length
                ),
            }
        }
        self.truncation = t;
        Ok(self)
    }

    pub fn get_truncation(&self) -> Option<&TruncationParams> {
        self.truncation.as_ref()
    }

    /// Sets (or clears) padding.
    pub fn with_padding(&mut self, p: Option<PaddingParams>) -> &mut Self {
        self.padding = p;
        self
    }

    pub fn get_padding(&self) -> Option<&PaddingParams> {
        self.padding.as_ref()
    }

    /// Whether special added tokens in the input are tokenized as plain text (instead of
    /// becoming their own id).
    pub fn set_encode_special_tokens(&mut self, value: bool) {
        self.encode_special_tokens = value;
    }

    pub fn get_encode_special_tokens(&self) -> bool {
        self.encode_special_tokens
    }

    /// The whole vocabulary (token -> id), optionally with the added tokens.
    pub fn get_vocab(&self, with_added_tokens: bool) -> HashMap<String, u32> {
        let mut v = self.model.get_vocab();
        if with_added_tokens {
            v.extend(self.added_ids.iter().map(|(t, &id)| (t.clone(), id)));
        }
        v
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

    /// Splits `p` at the added tokens `m` finds (honoring single_word / lstrip / rstrip).
    fn split_added(&self, p: Piece, m: &Matcher, out: &mut Vec<Seg>) {
        if m.is_empty() {
            out.push(Seg::Text(p));
            return;
        }
        let s = p.s.as_str();
        let b = s.as_bytes();
        let mut done = 0;
        let mut pos = 0;
        while let Some((mut start, mut stop, id)) = m.find(b, pos) {
            pos = stop;
            let a = &self.added[&id];
            if self.encode_special_tokens && a.special {
                continue;
            }
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
                out.push(Seg::Text(p.slice(done, start)));
            }
            out.push(Seg::Token(id));
            done = stop;
        }
        if done < s.len() || s.is_empty() {
            out.push(Seg::Text(p.slice(done, s.len())));
        }
    }

    /// The pieces of one text, in order: added tokens and pre-tokenized text.
    fn splits(&self, text: &str, out: &mut Vec<Seg>) {
        let mut raw = Vec::new();
        self.split_added(Piece::segment(text.to_string(), 0), &self.raw_matcher, &mut raw);
        for seg in raw {
            match seg {
                Seg::Token(id) => out.push(Seg::Token(id)),
                Seg::Text(p) => {
                    let p = match &self.normalizer {
                        Some(n) if self.track_start => {
                            let (s, zero) = n.normalize_tracked(p.s, p.shift0, p.zero);
                            Piece { s, zero, ..p }
                        }
                        // Untracked: nothing reads where the start went, but the old count
                        // would not even fall on a char boundary of the new text.
                        Some(n) => Piece {
                            s: n.normalize(p.s),
                            zero: 0,
                            ..p
                        },
                        None => p,
                    };
                    let mut segs = Vec::new();
                    // The reference re-slices the normalized text here, matches or not.
                    let p = if p.s.is_empty() { p } else { p.slice(0, p.s.len()) };
                    self.split_added(p, &self.norm_matcher, &mut segs);
                    for seg in segs {
                        match seg {
                            Seg::Token(id) => out.push(Seg::Token(id)),
                            Seg::Text(p) => {
                                if p.s.is_empty() {
                                    continue;
                                }
                                match &self.pre_tokenizer {
                                    Some(t) => out.extend(t.apply(vec![p]).into_iter().map(Seg::Text)),
                                    None => out.push(Seg::Text(p)),
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// The ids of one text (added tokens, normalizer, pre-tokenizer, model). With `limit`, the
    /// pieces are tokenized from the truncated end only until that many ids are reached (the
    /// reference's early truncation: later pieces are dropped).
    fn tokenize(&self, text: &str, limit: Option<(usize, Direction)>, ids: &mut Vec<u32>) -> Result<()> {
        let mut segs = Vec::new();
        self.splits(text, &mut segs);
        let Some((max, dir)) = limit else {
            for seg in segs {
                match seg {
                    Seg::Token(id) => ids.push(id),
                    Seg::Text(p) => self.model.tokenize(&p.s, ids)?,
                }
            }
            return Ok(());
        };
        let mut parts: Vec<Vec<u32>> = Vec::new();
        let mut total = 0;
        let order: Box<dyn Iterator<Item = Seg>> = match dir {
            Direction::Right => Box::new(segs.into_iter()),
            Direction::Left => Box::new(segs.into_iter().rev()),
        };
        for seg in order {
            match seg {
                Seg::Token(id) => {
                    total += 1;
                    parts.push(vec![id]);
                }
                Seg::Text(p) => {
                    let mut v = Vec::new();
                    self.model.tokenize(&p.s, &mut v)?;
                    total += v.len();
                    parts.push(v);
                    if total >= max {
                        break;
                    }
                }
            }
        }
        if dir == Direction::Left {
            parts.reverse();
        }
        ids.extend(parts.into_iter().flatten());
        Ok(())
    }

    fn encode_sequence(&self, seq: &InputSequence, type_id: u32) -> Result<Encoding> {
        let mut ids = Vec::new();
        // Early truncation (not for the first sequence when only the second is truncated).
        let limit = self
            .truncation
            .as_ref()
            .filter(|t| t.strategy != TruncationStrategy::OnlySecond || type_id != 0)
            .map(|t| (t.max_length, t.direction));
        match seq {
            InputSequence::Raw(s) => {
                ids.reserve(s.len() / 3 + 4);
                self.tokenize(s, limit, &mut ids)?;
            }
            InputSequence::PreTokenized(words) => {
                for w in words {
                    self.tokenize(w, limit, &mut ids)?;
                }
            }
        }
        Ok(Encoding::new(ids, type_id))
    }

    /// Encodes a text, a pre-tokenized text (a list of words) or a pair of them (`(a, b)`);
    /// `add_special_tokens` lets the post-processor add its tokens (BOS, `[CLS]`, ...). Then
    /// truncation and padding apply, when set.
    pub fn encode<'s>(&self, input: impl Into<EncodeInput<'s>>, add_special_tokens: bool) -> Result<Encoding> {
        let (a, b) = match input.into() {
            EncodeInput::Single(a) => (a, None),
            EncodeInput::Dual(a, b) => (a, Some(b)),
        };
        let mut encoding = self.encode_sequence(&a, 0)?;
        let mut pair = b.map(|b| self.encode_sequence(&b, 1)).transpose()?;
        if let Some(t) = &self.truncation {
            let added = match &self.post_processor {
                Some(p) if add_special_tokens => p.added_tokens(pair.is_some()),
                _ => 0,
            };
            let mut t = t.clone();
            if added > 0 {
                t.max_length = t
                    .max_length
                    .checked_sub(added)
                    .ok_or_else(|| anyhow!("truncation max_length {} is below the {added} special tokens", t.max_length))?;
            }
            truncate_encodings(&mut encoding, pair.as_mut(), &t)?;
        }
        let mut encoding = match &self.post_processor {
            Some(p) => p.process(encoding, pair, add_special_tokens)?,
            None => Encoding::merge(std::iter::once(encoding).chain(pair)),
        };
        if let Some(p) = &self.padding {
            pad_encodings(std::slice::from_mut(&mut encoding), p);
        }
        Ok(encoding)
    }

    /// Encodes several inputs (in parallel); with padding to the longest, they all get the
    /// length of the longest.
    pub fn encode_batch<'s, E: Into<EncodeInput<'s>> + Send>(&self, inputs: Vec<E>, add_special_tokens: bool) -> Result<Vec<Encoding>> {
        let inputs: Vec<EncodeInput<'s>> = inputs.into_iter().map(Into::into).collect();
        let threads = if cfg!(target_arch = "wasm32") {
            1
        } else {
            std::thread::available_parallelism().map_or(1, |n| n.get()).min(inputs.len())
        };
        let mut out: Vec<Encoding> = if threads <= 1 {
            inputs
                .into_iter()
                .map(|i| self.encode(i, add_special_tokens))
                .collect::<Result<_>>()?
        } else {
            let mut slots: Vec<Option<EncodeInput<'s>>> = inputs.into_iter().map(Some).collect();
            let chunk = slots.len().div_ceil(threads);
            let parts: Vec<Result<Vec<Encoding>>> = std::thread::scope(|sc| {
                let handles: Vec<_> = slots
                    .chunks_mut(chunk)
                    .map(|c| {
                        sc.spawn(move || {
                            c.iter_mut()
                                .map(|i| self.encode(i.take().unwrap(), add_special_tokens))
                                .collect::<Result<Vec<_>>>()
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().expect("tokenizer thread panicked")).collect()
            });
            let mut out = Vec::new();
            for p in parts {
                out.extend(p?);
            }
            out
        };
        if let Some(p) = &self.padding {
            pad_encodings(&mut out, p);
        }
        Ok(out)
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

impl Tokenizer {
    /// Decodes several id sequences.
    pub fn decode_batch(&self, sentences: &[&[u32]], skip_special_tokens: bool) -> Result<Vec<String>> {
        sentences.iter().map(|ids| self.decode(ids, skip_special_tokens)).collect()
    }

    /// A streaming decoder: feed ids one by one, get the text they complete.
    pub fn decode_stream(&self, skip_special_tokens: bool) -> DecodeStream<'_> {
        DecodeStream {
            tokenizer: self,
            skip_special_tokens,
            ids: Vec::new(),
            prefix: String::new(),
            prefix_index: 0,
        }
    }
}

/// Incremental decoding (see [`Tokenizer::decode_stream`]): each [`step`](Self::step) returns
/// the new text the id completes, or None while it is pending (an incomplete UTF-8 sequence,
/// a token that only changes how the previous ones decode).
pub struct DecodeStream<'t> {
    tokenizer: &'t Tokenizer,
    skip_special_tokens: bool,
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
}

impl DecodeStream<'_> {
    pub fn step(&mut self, id: u32) -> Result<Option<String>> {
        let t = self.tokenizer;
        let skip = self.skip_special_tokens;
        if self.prefix.is_empty() && !self.ids.is_empty() {
            let p = t.decode(&self.ids, skip)?;
            if !p.ends_with('\u{fffd}') {
                self.prefix = p;
                self.prefix_index = self.ids.len();
            }
        }
        self.ids.push(id);
        let s = t.decode(&self.ids, skip)?;
        if s.len() > self.prefix.len() && !s.ends_with('\u{fffd}') {
            if !s.starts_with(&self.prefix) {
                bail!(
                    "decode stream: token {id} changed the text already returned ({:?} is not a prefix of {s:?})",
                    self.prefix
                );
            }
            let new = s[self.prefix.len()..].to_string();
            let new_index = self.ids.len() - self.prefix_index;
            self.ids.drain(..self.prefix_index);
            self.prefix = t.decode(&self.ids, skip)?;
            self.prefix_index = new_index;
            Ok(Some(new))
        } else {
            Ok(None)
        }
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
