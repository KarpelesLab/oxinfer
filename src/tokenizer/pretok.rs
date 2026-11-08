//! Pre-tokenizers: how normalized text is cut into the pieces the model tokenizes one by one.

use super::normalize::Pattern;
use super::regex::Regex;
use super::unicode;
use anyhow::{Result, anyhow, bail};
use serde_json::Value;
use std::sync::OnceLock;

/// A piece of text; `first` is set when it starts the input (Metaspace's `first` scheme).
#[derive(Clone, Debug)]
pub struct Piece {
    pub s: String,
    pub first: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Behavior {
    Removed,
    Isolated,
    MergedWithPrevious,
    MergedWithNext,
    Contiguous,
}

impl Behavior {
    fn from_json(v: &Value) -> Result<Behavior> {
        Ok(match v.as_str().unwrap_or("Isolated") {
            "Removed" => Behavior::Removed,
            "Isolated" => Behavior::Isolated,
            "MergedWithPrevious" => Behavior::MergedWithPrevious,
            "MergedWithNext" => Behavior::MergedWithNext,
            "Contiguous" => Behavior::Contiguous,
            b => bail!("unsupported split behavior {b}"),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prepend {
    Always,
    Never,
    First,
}

#[derive(Clone, Debug)]
pub enum PreTokenizer {
    ByteLevel {
        add_prefix_space: bool,
        use_regex: bool,
    },
    Split {
        pattern: Pattern,
        behavior: Behavior,
        invert: bool,
    },
    Whitespace,
    WhitespaceSplit,
    Bert,
    Metaspace {
        replacement: char,
        prepend: Prepend,
        split: bool,
    },
    Digits {
        individual: bool,
    },
    Punctuation(Behavior),
    CharDelimiter(char),
    FixedLength(usize),
    Sequence(Vec<PreTokenizer>),
}

/// The GPT-2 byte-level split pattern.
fn gpt2_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+").unwrap())
}

fn whitespace_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\w+|[^\w\s]+").unwrap())
}

/// Byte -> char of the GPT-2 byte-level alphabet.
pub fn bytes_char() -> &'static [char; 256] {
    static T: OnceLock<[char; 256]> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = ['\0'; 256];
        let mut n = 0;
        for b in 0..256u32 {
            let printable = (b'!' as u32..=b'~' as u32).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
            t[b as usize] = if printable {
                char::from_u32(b).unwrap()
            } else {
                n += 1;
                char::from_u32(255 + n).unwrap()
            };
        }
        t
    })
}

/// `s` in the byte-level alphabet (each UTF-8 byte as one char).
pub fn byte_level_chars(s: &str) -> String {
    let t = bytes_char();
    s.bytes().map(|b| t[b as usize]).collect()
}

/// Splits `s` at the `(range, is_match)` pieces `matches` per `behavior`, keeping non-empty
/// pieces.
fn split_by(p: &Piece, matches: Vec<((usize, usize), bool)>, behavior: Behavior, out: &mut Vec<Piece>) {
    let mut spans: Vec<((usize, usize), bool)> = Vec::with_capacity(matches.len());
    match behavior {
        Behavior::Isolated => spans = matches.into_iter().map(|(o, _)| (o, false)).collect(),
        Behavior::Removed => spans = matches,
        Behavior::Contiguous => {
            let mut prev = false;
            for (o, m) in matches {
                match spans.last_mut() {
                    Some(((_, e), _)) if m == prev => *e = o.1,
                    _ => spans.push((o, false)),
                }
                prev = m;
            }
        }
        Behavior::MergedWithPrevious => {
            let mut prev = false;
            for (o, m) in matches {
                match spans.last_mut() {
                    Some(((_, e), _)) if m && !prev => *e = o.1,
                    _ => spans.push((o, false)),
                }
                prev = m;
            }
        }
        Behavior::MergedWithNext => {
            let mut prev = false;
            for (o, m) in matches.into_iter().rev() {
                match spans.last_mut() {
                    Some(((s, _), _)) if m && !prev => *s = o.0,
                    _ => spans.push((o, false)),
                }
                prev = m;
            }
            spans.reverse();
        }
    }
    for ((a, b), remove) in spans {
        if !remove && a < b {
            out.push(Piece {
                s: p.s[a..b].to_string(),
                first: p.first && a == 0,
            });
        }
    }
}

/// `(range, is_match)` pieces of `s` for a per-char predicate.
fn char_matches(s: &str, f: impl Fn(char) -> bool) -> Vec<((usize, usize), bool)> {
    let mut out = Vec::new();
    let mut last = 0;
    for (i, c) in s.char_indices() {
        if f(c) {
            if last < i {
                out.push(((last, i), false));
            }
            out.push(((i, i + c.len_utf8()), true));
            last = i + c.len_utf8();
        }
    }
    if last < s.len() {
        out.push(((last, s.len()), false));
    }
    out
}

fn regex_matches(re: &Regex, s: &str) -> Vec<((usize, usize), bool)> {
    let mut out = Vec::new();
    let mut prev = 0;
    for (a, b) in re.find_iter(s) {
        if prev != a {
            out.push(((prev, a), false));
        }
        out.push(((a, b), true));
        prev = b;
    }
    if prev != s.len() {
        out.push(((prev, s.len()), false));
    }
    out
}

impl PreTokenizer {
    pub fn from_json(v: &Value) -> Result<PreTokenizer> {
        let ty = v["type"].as_str().ok_or_else(|| anyhow!("pre-tokenizer without a type: {v}"))?;
        Ok(match ty {
            "ByteLevel" => PreTokenizer::ByteLevel {
                add_prefix_space: v["add_prefix_space"].as_bool().unwrap_or(true),
                use_regex: v["use_regex"].as_bool().unwrap_or(true),
            },
            "Split" => PreTokenizer::Split {
                pattern: Pattern::from_json(&v["pattern"])?,
                behavior: Behavior::from_json(&v["behavior"])?,
                invert: v["invert"].as_bool().unwrap_or(false),
            },
            "Whitespace" => PreTokenizer::Whitespace,
            "WhitespaceSplit" => PreTokenizer::WhitespaceSplit,
            "BertPreTokenizer" => PreTokenizer::Bert,
            "Metaspace" => {
                let replacement = v["replacement"].as_str().and_then(|s| s.chars().next()).unwrap_or('▁');
                let mut prepend = match v["prepend_scheme"].as_str().unwrap_or("always") {
                    "always" => Prepend::Always,
                    "never" => Prepend::Never,
                    "first" => Prepend::First,
                    s => bail!("unsupported Metaspace prepend_scheme {s}"),
                };
                if v["add_prefix_space"].as_bool() == Some(false) {
                    prepend = Prepend::Never;
                }
                PreTokenizer::Metaspace {
                    replacement,
                    prepend,
                    split: v["split"].as_bool().unwrap_or(true),
                }
            }
            "Digits" => PreTokenizer::Digits {
                individual: v["individual_digits"].as_bool().unwrap_or(false),
            },
            "Punctuation" => PreTokenizer::Punctuation(Behavior::from_json(&v["behavior"])?),
            "CharDelimiterSplit" => PreTokenizer::CharDelimiter(
                v["delimiter"]
                    .as_str()
                    .and_then(|s| s.chars().next())
                    .ok_or_else(|| anyhow!("CharDelimiterSplit without a delimiter"))?,
            ),
            "FixedLength" => PreTokenizer::FixedLength(v["length"].as_u64().unwrap_or(5) as usize),
            "Sequence" => PreTokenizer::Sequence(
                v["pretokenizers"]
                    .as_array()
                    .map(|a| a.iter().map(PreTokenizer::from_json).collect())
                    .unwrap_or(Ok(vec![]))?,
            ),
            t => bail!("unsupported pre-tokenizer {t}"),
        })
    }

    /// Splits every piece of `pieces`.
    pub fn apply(&self, pieces: Vec<Piece>) -> Vec<Piece> {
        if let PreTokenizer::Sequence(v) = self {
            return v.iter().fold(pieces, |p, t| t.apply(p));
        }
        let mut out = Vec::with_capacity(pieces.len());
        for p in &pieces {
            self.split_one(p, &mut out);
        }
        if let PreTokenizer::ByteLevel { .. } = self {
            for p in &mut out {
                p.s = byte_level_chars(&p.s);
            }
        }
        out
    }

    fn split_one(&self, p: &Piece, out: &mut Vec<Piece>) {
        match self {
            PreTokenizer::ByteLevel {
                add_prefix_space,
                use_regex,
            } => {
                let p = if *add_prefix_space && !p.s.starts_with(' ') {
                    &Piece {
                        s: format!(" {}", p.s),
                        first: p.first,
                    }
                } else {
                    p
                };
                if *use_regex {
                    split_by(p, regex_matches(gpt2_regex(), &p.s), Behavior::Isolated, out);
                } else {
                    out.push(p.clone());
                }
            }
            PreTokenizer::Split { pattern, behavior, invert } => {
                let mut m = pattern.find_matches(&p.s);
                if *invert {
                    m.iter_mut().for_each(|(_, f)| *f = !*f);
                }
                split_by(p, m, *behavior, out);
            }
            PreTokenizer::Whitespace => {
                let m = regex_matches(whitespace_regex(), &p.s).into_iter().map(|(o, f)| (o, !f)).collect();
                split_by(p, m, Behavior::Removed, out);
            }
            PreTokenizer::WhitespaceSplit => split_by(p, char_matches(&p.s, char::is_whitespace), Behavior::Removed, out),
            PreTokenizer::Bert => {
                let mut tmp = Vec::new();
                split_by(p, char_matches(&p.s, char::is_whitespace), Behavior::Removed, &mut tmp);
                for q in &tmp {
                    split_by(q, char_matches(&q.s, unicode::is_punctuation), Behavior::Isolated, out);
                }
            }
            PreTokenizer::Metaspace {
                replacement,
                prepend,
                split,
            } => {
                let mut s: String = p.s.chars().map(|c| if c == ' ' { *replacement } else { c }).collect();
                let add = match prepend {
                    Prepend::Always => true,
                    Prepend::First => p.first,
                    Prepend::Never => false,
                };
                if add && !s.is_empty() && !s.starts_with(*replacement) {
                    s.insert(0, *replacement);
                }
                let q = Piece { s, first: p.first };
                if *split {
                    let r = *replacement;
                    split_by(&q, char_matches(&q.s, |c| c == r), Behavior::MergedWithNext, out);
                } else if !q.s.is_empty() {
                    out.push(q);
                }
            }
            PreTokenizer::Digits { individual } => {
                let b = if *individual { Behavior::Isolated } else { Behavior::Contiguous };
                split_by(p, char_matches(&p.s, char::is_numeric), b, out);
            }
            PreTokenizer::Punctuation(b) => split_by(p, char_matches(&p.s, unicode::is_punctuation), *b, out),
            PreTokenizer::CharDelimiter(d) => {
                let d = *d;
                split_by(p, char_matches(&p.s, |c| c == d), Behavior::Removed, out)
            }
            PreTokenizer::FixedLength(n) => {
                let chars: Vec<(usize, char)> = p.s.char_indices().collect();
                for (k, chunk) in chars.chunks((*n).max(1)).enumerate() {
                    let a = chunk[0].0;
                    let last = chunk[chunk.len() - 1];
                    let b = last.0 + last.1.len_utf8();
                    out.push(Piece {
                        s: p.s[a..b].to_string(),
                        first: p.first && k == 0,
                    });
                }
            }
            PreTokenizer::Sequence(_) => unreachable!(),
        }
    }
}
