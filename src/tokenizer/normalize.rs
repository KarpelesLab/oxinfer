//! Normalizers: the text transformations applied before pre-tokenization.

use super::regex::Regex;
use super::unicode;
use anyhow::{Context, Result, anyhow, bail};
use serde_json::Value;

/// A `{"String": ..}` / `{"Regex": ..}` pattern.
#[derive(Clone, Debug)]
pub enum Pattern {
    Str(String),
    Re(Regex),
}

impl Pattern {
    pub fn from_json(v: &Value) -> Result<Pattern> {
        if let Some(s) = v["String"].as_str() {
            Ok(Pattern::Str(s.to_string()))
        } else if let Some(r) = v["Regex"].as_str() {
            Ok(Pattern::Re(Regex::new(r)?))
        } else {
            bail!("unsupported pattern {v}")
        }
    }

    /// The text split into `(range, is_match)` pieces covering all of it.
    pub fn find_matches(&self, s: &str) -> Vec<((usize, usize), bool)> {
        if s.is_empty() {
            return vec![((0, 0), false)];
        }
        let hits: Vec<(usize, usize)> = match self {
            Pattern::Str(p) if p.is_empty() => return vec![((0, s.len()), false)],
            Pattern::Str(p) => s.match_indices(p.as_str()).map(|(i, m)| (i, i + m.len())).collect(),
            Pattern::Re(r) => r.find_iter(s),
        };
        let mut out = Vec::with_capacity(hits.len() * 2 + 1);
        let mut prev = 0;
        for (a, b) in hits {
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

    /// `s` with every match replaced by `content`.
    pub fn replace(&self, s: &str, content: &str) -> String {
        let mut out = String::with_capacity(s.len());
        for ((a, b), m) in self.find_matches(s) {
            out.push_str(if m { content } else { &s[a..b] });
        }
        out
    }
}

#[derive(Clone, Debug)]
pub enum Normalizer {
    Nfc,
    Nfd,
    Nfkc,
    Nfkd,
    Nmt,
    Lowercase,
    Strip {
        left: bool,
        right: bool,
    },
    StripAccents,
    Replace {
        pattern: Pattern,
        content: String,
    },
    Prepend(String),
    Bert {
        clean_text: bool,
        chinese: bool,
        strip_accents: bool,
        lowercase: bool,
    },
    Precompiled(Precompiled),
    ByteLevel,
    Sequence(Vec<Normalizer>),
}

impl Normalizer {
    pub fn from_json(v: &Value) -> Result<Normalizer> {
        let ty = v["type"].as_str().ok_or_else(|| anyhow!("normalizer without a type: {v}"))?;
        let flag = |k: &str, d: bool| v[k].as_bool().unwrap_or(d);
        Ok(match ty {
            "NFC" => Normalizer::Nfc,
            "NFD" => Normalizer::Nfd,
            "NFKC" => Normalizer::Nfkc,
            "NFKD" => Normalizer::Nfkd,
            "Nmt" => Normalizer::Nmt,
            "Lowercase" => Normalizer::Lowercase,
            "Strip" => Normalizer::Strip {
                left: flag("strip_left", true),
                right: flag("strip_right", true),
            },
            "StripAccents" => Normalizer::StripAccents,
            "Replace" => Normalizer::Replace {
                pattern: Pattern::from_json(&v["pattern"])?,
                content: v["content"].as_str().unwrap_or_default().to_string(),
            },
            "Prepend" => Normalizer::Prepend(v["prepend"].as_str().unwrap_or_default().to_string()),
            "BertNormalizer" => {
                let lowercase = flag("lowercase", true);
                Normalizer::Bert {
                    clean_text: flag("clean_text", true),
                    chinese: flag("handle_chinese_chars", true),
                    strip_accents: v["strip_accents"].as_bool().unwrap_or(lowercase),
                    lowercase,
                }
            }
            "Precompiled" => {
                let b64 = v["precompiled_charsmap"].as_str().unwrap_or_default();
                if b64.is_empty() {
                    // An empty charsmap is the identity.
                    Normalizer::Sequence(vec![])
                } else {
                    let raw = crate::b64::decode(b64).ok_or_else(|| anyhow!("precompiled_charsmap: bad base64"))?;
                    Normalizer::Precompiled(Precompiled::new(&raw).context("precompiled_charsmap")?)
                }
            }
            "ByteLevel" => Normalizer::ByteLevel,
            "Sequence" => Normalizer::Sequence(
                v["normalizers"]
                    .as_array()
                    .map(|a| a.iter().map(Normalizer::from_json).collect())
                    .unwrap_or(Ok(vec![]))?,
            ),
            t => bail!("unsupported normalizer {t}"),
        })
    }

    pub fn normalize(&self, s: String) -> String {
        match self {
            Normalizer::Nfc => unicode::nfc(&s),
            Normalizer::Nfd => unicode::nfd(&s),
            Normalizer::Nfkc => unicode::nfkc(&s),
            Normalizer::Nfkd => unicode::nfkd(&s),
            Normalizer::Nmt => s
                .chars()
                .filter(|&c| !matches!(c as u32, 0x1..=0x8 | 0xB | 0xE..=0x1F | 0x7F | 0x8F | 0x9F))
                .map(|c| match c as u32 {
                    0x9 | 0xA | 0xC | 0xD | 0x1680 | 0x200B..=0x200F | 0x2028 | 0x2029 | 0x2581 | 0xFEFF | 0xFFFD => ' ',
                    _ => c,
                })
                .collect(),
            Normalizer::Lowercase => s.chars().flat_map(char::to_lowercase).collect(),
            Normalizer::Strip { left, right } => {
                let mut t = s.as_str();
                if *left {
                    t = t.trim_start_matches(char::is_whitespace);
                }
                if *right {
                    t = t.trim_end_matches(char::is_whitespace);
                }
                t.to_string()
            }
            Normalizer::StripAccents => s.chars().filter(|&c| !unicode::is_mark(c)).collect(),
            Normalizer::Replace { pattern, content } => pattern.replace(&s, content),
            Normalizer::Prepend(p) => {
                if s.is_empty() {
                    s
                } else {
                    format!("{p}{s}")
                }
            }
            Normalizer::Bert {
                clean_text,
                chinese,
                strip_accents,
                lowercase,
            } => {
                let mut s = s;
                if *clean_text {
                    s = s
                        .chars()
                        .filter(|&c| {
                            let control = !matches!(c, '\t' | '\n' | '\r') && unicode::is_other(c);
                            !(c == '\0' || c == '\u{fffd}' || control)
                        })
                        .map(|c| {
                            if matches!(c, '\t' | '\n' | '\r') || c.is_whitespace() {
                                ' '
                            } else {
                                c
                            }
                        })
                        .collect();
                }
                if *chinese {
                    let mut t = String::with_capacity(s.len());
                    for c in s.chars() {
                        if is_chinese_char(c) {
                            t.push(' ');
                            t.push(c);
                            t.push(' ');
                        } else {
                            t.push(c);
                        }
                    }
                    s = t;
                }
                if *strip_accents {
                    s = unicode::nfd(&s).chars().filter(|&c| !unicode::is_nonspacing_mark(c)).collect();
                }
                if *lowercase {
                    s = s.chars().flat_map(char::to_lowercase).collect();
                }
                s
            }
            Normalizer::Precompiled(p) => p.normalize(&s),
            Normalizer::ByteLevel => super::pretok::byte_level_chars(&s),
            Normalizer::Sequence(v) => v.iter().fold(s, |s, n| n.normalize(s)),
        }
    }
}

fn is_chinese_char(c: char) -> bool {
    matches!(c as u32,
        0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0x20000..=0x2A6DF | 0x2A700..=0x2B73F | 0x2B740..=0x2B81F
        | 0x2B920..=0x2CEAF | 0xF900..=0xFAFF | 0x2F800..=0x2FA1F)
}

/// SentencePiece's precompiled normalization map: a Darts double-array trie over UTF-8 inputs
/// whose leaves index NUL-terminated replacement strings.
#[derive(Clone, Debug)]
pub struct Precompiled {
    trie: Vec<u32>,
    normalized: Vec<u8>,
}

impl Precompiled {
    pub fn new(raw: &[u8]) -> Result<Precompiled> {
        if raw.len() < 4 {
            bail!("too short");
        }
        let trie_size = u32::from_le_bytes(raw[..4].try_into().unwrap()) as usize;
        if raw.len() < 4 + trie_size {
            bail!("truncated trie");
        }
        let trie = raw[4..4 + trie_size]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| u32::from_le_bytes(c))
            .collect();
        let normalized = raw[4 + trie_size..].to_vec();
        if std::str::from_utf8(&normalized).is_err() {
            bail!("replacement strings are not UTF-8");
        }
        Ok(Precompiled { trie, normalized })
    }

    /// The first (shortest) prefix match of `key`.
    fn lookup(&self, key: &[u8]) -> Option<usize> {
        let a = &self.trie;
        let offset = |u: u32| ((u >> 10) << ((u & (1 << 9)) >> 6)) as usize;
        let label = |u: u32| (u & ((1 << 31) | 0xFF)) as usize;
        let has_leaf = |u: u32| (u >> 8) & 1 == 1;
        let value = |u: u32| (u & ((1 << 31) - 1)) as usize;
        let mut pos = 0usize;
        let mut unit = *a.first()?;
        pos ^= offset(unit);
        for &c in key {
            if c == 0 {
                break;
            }
            pos ^= c as usize;
            unit = *a.get(pos)?;
            if label(unit) != c as usize {
                return None;
            }
            pos ^= offset(unit);
            if has_leaf(unit) {
                return Some(value(*a.get(pos)?));
            }
        }
        None
    }

    fn transform(&self, chunk: &str) -> Option<&str> {
        let i = self.lookup(chunk.as_bytes())?;
        let rest = self.normalized.get(i..)?;
        let end = rest.iter().position(|&b| b == 0).unwrap_or(rest.len());
        std::str::from_utf8(&rest[..end]).ok()
    }

    /// Grapheme by grapheme: a short grapheme mapped as a whole, else char by char (as
    /// SentencePiece and the reference implementation do).
    pub fn normalize(&self, s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        for g in unicode::graphemes(s) {
            if g.len() < 6
                && let Some(n) = self.transform(g)
            {
                out.push_str(n);
                continue;
            }
            for (i, c) in g.char_indices() {
                match self.transform(&g[i..i + c.len_utf8()]) {
                    Some(n) => out.push_str(n),
                    None => out.push(c),
                }
            }
        }
        out
    }
}
