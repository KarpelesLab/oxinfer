//! Decoders (token strings back to text) and post-processors (special tokens around a sequence).

use super::normalize::Pattern;
use super::pretok::bytes_char;
use anyhow::{Result, anyhow, bail};
use serde_json::Value;
use std::sync::OnceLock;

#[derive(Clone, Debug)]
pub enum Decoder {
    ByteLevel,
    Metaspace { replacement: char, strip_first: bool },
    WordPiece { prefix: String, cleanup: bool },
    Replace { pattern: Pattern, content: String },
    ByteFallback,
    Fuse,
    Strip { content: char, start: usize, stop: usize },
    Bpe { suffix: String },
    Ctc { pad: String, delimiter: String, cleanup: bool },
    Sequence(Vec<Decoder>),
}

/// Char of the byte-level alphabet -> byte.
fn char_bytes() -> &'static [Option<u8>] {
    static T: OnceLock<Vec<Option<u8>>> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = vec![None; 0x200];
        for (b, &c) in bytes_char().iter().enumerate() {
            t[c as usize] = Some(b as u8);
        }
        t
    })
}

fn wordpiece_cleanup(s: &str) -> String {
    s.replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" do not", " don't")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
}

impl Decoder {
    pub fn from_json(v: &Value) -> Result<Decoder> {
        let ty = v["type"].as_str().ok_or_else(|| anyhow!("decoder without a type: {v}"))?;
        let str_or = |k: &str, d: &str| v[k].as_str().unwrap_or(d).to_string();
        Ok(match ty {
            "ByteLevel" => Decoder::ByteLevel,
            "Metaspace" => {
                let never = v["prepend_scheme"].as_str() == Some("never") || v["add_prefix_space"].as_bool() == Some(false);
                Decoder::Metaspace {
                    replacement: v["replacement"].as_str().and_then(|s| s.chars().next()).unwrap_or('▁'),
                    strip_first: !never,
                }
            }
            "WordPiece" => Decoder::WordPiece {
                prefix: str_or("prefix", "##"),
                cleanup: v["cleanup"].as_bool().unwrap_or(true),
            },
            "Replace" => Decoder::Replace {
                pattern: Pattern::from_json(&v["pattern"])?,
                content: str_or("content", ""),
            },
            "ByteFallback" => Decoder::ByteFallback,
            "Fuse" => Decoder::Fuse,
            "Strip" => Decoder::Strip {
                content: v["content"].as_str().and_then(|s| s.chars().next()).unwrap_or(' '),
                start: v["start"].as_u64().unwrap_or(0) as usize,
                stop: v["stop"].as_u64().unwrap_or(0) as usize,
            },
            "BPEDecoder" => Decoder::Bpe {
                suffix: str_or("suffix", "</w>"),
            },
            "CTC" => Decoder::Ctc {
                pad: str_or("pad_token", "<pad>"),
                delimiter: str_or("word_delimiter_token", "|"),
                cleanup: v["cleanup"].as_bool().unwrap_or(true),
            },
            "Sequence" => Decoder::Sequence(
                v["decoders"]
                    .as_array()
                    .map(|a| a.iter().map(Decoder::from_json).collect())
                    .unwrap_or(Ok(vec![]))?,
            ),
            t => bail!("unsupported decoder {t}"),
        })
    }

    pub fn decode(&self, tokens: Vec<String>) -> String {
        self.chain(tokens).concat()
    }

    fn chain(&self, tokens: Vec<String>) -> Vec<String> {
        match self {
            Decoder::ByteLevel => {
                let map = char_bytes();
                let mut bytes = Vec::new();
                for t in &tokens {
                    let mapped: Option<Vec<u8>> = t.chars().map(|c| map.get(c as usize).copied().flatten()).collect();
                    match mapped {
                        Some(b) => bytes.extend(b),
                        None => bytes.extend_from_slice(t.as_bytes()),
                    }
                }
                vec![String::from_utf8_lossy(&bytes).into_owned()]
            }
            Decoder::Metaspace { replacement, strip_first } => tokens
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    t.chars()
                        .filter_map(|c| {
                            if c == *replacement {
                                if i == 0 && *strip_first { None } else { Some(' ') }
                            } else {
                                Some(c)
                            }
                        })
                        .collect()
                })
                .collect(),
            Decoder::WordPiece { prefix, cleanup } => tokens
                .into_iter()
                .enumerate()
                .map(|(i, t)| {
                    let t = if i == 0 {
                        t
                    } else if let Some(s) = t.strip_prefix(prefix.as_str()) {
                        s.to_string()
                    } else {
                        format!(" {t}")
                    };
                    if *cleanup { wordpiece_cleanup(&t) } else { t }
                })
                .collect(),
            Decoder::Replace { pattern, content } => tokens.iter().map(|t| pattern.replace(t, content)).collect(),
            Decoder::ByteFallback => {
                let mut out = Vec::with_capacity(tokens.len());
                let mut pending: Vec<u8> = Vec::new();
                let flush = |pending: &mut Vec<u8>, out: &mut Vec<String>| {
                    if pending.is_empty() {
                        return;
                    }
                    match String::from_utf8(pending.clone()) {
                        Ok(s) => out.push(s),
                        Err(_) => out.extend(std::iter::repeat_n("\u{fffd}".to_string(), pending.len())),
                    }
                    pending.clear();
                };
                for t in tokens {
                    let byte = (t.len() == 6 && t.starts_with("<0x") && t.ends_with('>'))
                        .then(|| u8::from_str_radix(&t[3..5], 16).ok())
                        .flatten();
                    match byte {
                        Some(b) => pending.push(b),
                        None => {
                            flush(&mut pending, &mut out);
                            out.push(t);
                        }
                    }
                }
                flush(&mut pending, &mut out);
                out
            }
            Decoder::Fuse => vec![tokens.concat()],
            Decoder::Strip { content, start, stop } => tokens
                .into_iter()
                .map(|t| {
                    let chars: Vec<char> = t.chars().collect();
                    let mut a = 0;
                    for (i, &c) in chars.iter().enumerate().take(*start) {
                        if c == *content {
                            a = i + 1;
                        } else {
                            break;
                        }
                    }
                    let mut b = chars.len();
                    for i in 0..(*stop).min(chars.len()) {
                        let k = chars.len() - i - 1;
                        if chars[k] == *content {
                            b = k;
                        } else {
                            break;
                        }
                    }
                    chars[a..b.max(a)].iter().collect()
                })
                .collect(),
            Decoder::Bpe { suffix } => {
                let n = tokens.len().saturating_sub(1);
                tokens
                    .into_iter()
                    .enumerate()
                    .map(|(i, t)| t.replace(suffix.as_str(), if i == n { "" } else { " " }))
                    .collect()
            }
            Decoder::Ctc { pad, delimiter, cleanup } => {
                let mut out = Vec::new();
                let mut last: Option<String> = None;
                for t in tokens {
                    if last.as_ref() == Some(&t) {
                        continue;
                    }
                    last = Some(t.clone());
                    let mut r = t.replace(pad.as_str(), "");
                    if *cleanup {
                        r = wordpiece_cleanup(&r).replace(delimiter.as_str(), " ");
                    }
                    if !r.is_empty() {
                        out.push(r);
                    }
                }
                out
            }
            Decoder::Sequence(v) => v.iter().fold(tokens, |t, d| d.chain(t)),
        }
    }
}

/// One element of a template: a special token's ids, or the sequence (A or B).
#[derive(Clone, Debug)]
pub enum Piece {
    Special(Vec<u32>),
    Seq(bool),
}

#[derive(Clone, Debug)]
pub enum PostProcessor {
    Template {
        single: Vec<Piece>,
    },
    /// `[cls] A [sep]` (RoBERTa and BERT processing of a single sequence).
    Wrap {
        cls: u32,
        sep: u32,
    },
    Noop,
    Sequence(Vec<PostProcessor>),
}

impl PostProcessor {
    pub fn from_json(v: &Value) -> Result<PostProcessor> {
        let ty = v["type"].as_str().ok_or_else(|| anyhow!("post-processor without a type: {v}"))?;
        let tok = |k: &str| -> Result<u32> { v[k][1].as_u64().map(|x| x as u32).ok_or_else(|| anyhow!("{ty}: missing {k}")) };
        Ok(match ty {
            "TemplateProcessing" => {
                let specials = &v["special_tokens"];
                let parse = |t: &Value| -> Result<Vec<Piece>> {
                    let mut out = Vec::new();
                    for p in t.as_array().into_iter().flatten() {
                        if let Some(s) = p.get("SpecialToken") {
                            let name = s["id"].as_str().unwrap_or_default();
                            let ids = specials[name]["ids"]
                                .as_array()
                                .ok_or_else(|| anyhow!("template special token {name:?} is not defined"))?
                                .iter()
                                .map(|x| x.as_u64().map(|x| x as u32).ok_or_else(|| anyhow!("bad id for {name:?}")))
                                .collect::<Result<Vec<u32>>>()?;
                            out.push(Piece::Special(ids));
                        } else if let Some(s) = p.get("Sequence") {
                            out.push(Piece::Seq(s["id"].as_str() == Some("B")));
                        } else {
                            bail!("bad template piece {p}");
                        }
                    }
                    Ok(out)
                };
                PostProcessor::Template {
                    single: parse(&v["single"])?,
                }
            }
            "RobertaProcessing" => PostProcessor::Wrap {
                cls: tok("cls")?,
                sep: tok("sep")?,
            },
            "BertProcessing" => PostProcessor::Wrap {
                cls: tok("cls")?,
                sep: tok("sep")?,
            },
            "ByteLevel" => PostProcessor::Noop,
            "Sequence" => PostProcessor::Sequence(
                v["processors"]
                    .as_array()
                    .map(|a| a.iter().map(PostProcessor::from_json).collect())
                    .unwrap_or(Ok(vec![]))?,
            ),
            t => bail!("unsupported post-processor {t}"),
        })
    }

    /// How many special tokens a single sequence gets.
    pub fn added_tokens(&self) -> usize {
        match self {
            PostProcessor::Template { single } => single.iter().map(|p| if let Piece::Special(ids) = p { ids.len() } else { 0 }).sum(),
            PostProcessor::Wrap { .. } => 2,
            PostProcessor::Noop => 0,
            PostProcessor::Sequence(v) => v.iter().map(PostProcessor::added_tokens).sum(),
        }
    }

    /// Adds the special tokens around a single sequence: (ids, special-token mask).
    pub fn process(&self, ids: Vec<u32>, mask: Vec<u32>) -> (Vec<u32>, Vec<u32>) {
        match self {
            PostProcessor::Template { single } => {
                let mut o = Vec::with_capacity(ids.len() + 4);
                let mut m = Vec::with_capacity(ids.len() + 4);
                for p in single {
                    match p {
                        Piece::Special(s) => {
                            o.extend_from_slice(s);
                            m.extend(std::iter::repeat_n(1, s.len()));
                        }
                        Piece::Seq(false) => {
                            o.extend_from_slice(&ids);
                            m.extend_from_slice(&mask);
                        }
                        Piece::Seq(true) => {}
                    }
                }
                (o, m)
            }
            PostProcessor::Wrap { cls, sep } => {
                let mut o = Vec::with_capacity(ids.len() + 2);
                o.push(*cls);
                o.extend_from_slice(&ids);
                o.push(*sep);
                let mut m = vec![1];
                m.extend_from_slice(&mask);
                m.push(1);
                (o, m)
            }
            PostProcessor::Noop => (ids, mask),
            PostProcessor::Sequence(v) => v.iter().fold((ids, mask), |(i, m), p| p.process(i, m)),
        }
    }
}
