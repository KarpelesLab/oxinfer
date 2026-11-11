//! Decoders (token strings back to text) and post-processors (special tokens around a sequence).

use super::Encoding;
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

/// One element of a template: a special token's ids, or a sequence (A or B), with the type id
/// its tokens get.
#[derive(Clone, Debug)]
pub enum Piece {
    Special(Vec<u32>, u32),
    Seq(bool, u32),
}

#[derive(Clone, Debug)]
pub enum PostProcessor {
    Template {
        single: Vec<Piece>,
        pair: Vec<Piece>,
    },
    /// `[CLS] A [SEP]` / `[CLS] A [SEP] B [SEP]` (B gets type id 1).
    Bert {
        cls: u32,
        sep: u32,
    },
    /// `<s> A </s>` / `<s> A </s> </s> B </s>` (all type id 0).
    Roberta {
        cls: u32,
        sep: u32,
    },
    /// Offsets trimming only: nothing to do on ids.
    ByteLevel,
    Sequence(Vec<PostProcessor>),
}

/// Parses a template piece: `{"Sequence": {"id": "A", "type_id": 0}}`,
/// `{"SpecialToken": {"id": "[CLS]", "type_id": 0}}`, or the string forms `$A`, `$B:1`, `$1`,
/// `[CLS]:0`.
fn template_piece(p: &Value, specials: &Value) -> Result<Piece> {
    let special = |name: &str, type_id: u32| -> Result<Piece> {
        let ids = specials[name]["ids"]
            .as_array()
            .ok_or_else(|| anyhow!("template special token {name:?} is not defined"))?
            .iter()
            .map(|x| x.as_u64().map(|x| x as u32).ok_or_else(|| anyhow!("bad id for {name:?}")))
            .collect::<Result<Vec<u32>>>()?;
        Ok(Piece::Special(ids, type_id))
    };
    if let Some(s) = p.get("SpecialToken") {
        return special(s["id"].as_str().unwrap_or_default(), s["type_id"].as_u64().unwrap_or(0) as u32);
    }
    if let Some(s) = p.get("Sequence") {
        let b = match s["id"].as_str() {
            Some("A") | None => false,
            Some("B") => true,
            Some(x) => bail!("bad template sequence id {x:?}"),
        };
        return Ok(Piece::Seq(b, s["type_id"].as_u64().unwrap_or(0) as u32));
    }
    let Some(s) = p.as_str() else { bail!("bad template piece {p}") };
    let (id, type_id) = match s.split(':').collect::<Vec<_>>()[..] {
        [id] => (id, None),
        [id, t] => (id, Some(t.parse::<u32>().map_err(|_| anyhow!("bad template piece {s:?}"))?)),
        _ => bail!("bad template piece {s:?}"),
    };
    if let Some(rest) = id.strip_prefix('$') {
        let (b, t) = match rest {
            "" | "A" | "a" => (false, 0),
            "B" | "b" => (true, 0),
            n => (false, n.parse::<u32>().map_err(|_| anyhow!("bad template piece {s:?}"))?),
        };
        Ok(Piece::Seq(b, type_id.unwrap_or(t)))
    } else {
        special(id, type_id.unwrap_or(0))
    }
}

fn template(t: &Value, specials: &Value, default: &str) -> Result<Vec<Piece>> {
    match t {
        Value::Array(a) => a.iter().map(|p| template_piece(p, specials)).collect(),
        Value::String(s) => s
            .split(' ')
            .map(|p| template_piece(&Value::String(p.to_string()), specials))
            .collect(),
        Value::Null => default
            .split(' ')
            .map(|p| template_piece(&Value::String(p.to_string()), specials))
            .collect(),
        _ => bail!("bad template {t}"),
    }
}

/// `encoding` wrapped as `[pre..] encoding [post]` (special tokens marked, attention reset), the
/// same for its overflowing parts; `ty` gives the type ids (`None`: keep the encoding's).
fn wrap(e: &Encoding, pre: &[u32], post: u32, ty: (u32, Option<u32>, u32)) -> Encoding {
    let one = |e: &Encoding| {
        let n = e.ids.len();
        let mut ids = Vec::with_capacity(n + pre.len() + 1);
        ids.extend_from_slice(pre);
        ids.extend_from_slice(&e.ids);
        ids.push(post);
        let mut type_ids = vec![ty.0; pre.len()];
        match ty.1 {
            Some(t) => type_ids.extend(std::iter::repeat_n(t, n)),
            None => type_ids.extend_from_slice(&e.type_ids),
        }
        type_ids.push(ty.2);
        let mut special = vec![1; pre.len()];
        special.extend(std::iter::repeat_n(0, n));
        special.push(1);
        let attention = vec![1; ids.len()];
        Encoding {
            ids,
            type_ids,
            special,
            attention,
            overflowing: Vec::new(),
        }
    };
    let mut out = one(e);
    out.overflowing = e.overflowing.iter().map(one).collect();
    out
}

impl PostProcessor {
    pub fn from_json(v: &Value) -> Result<PostProcessor> {
        let ty = v["type"].as_str().ok_or_else(|| anyhow!("post-processor without a type: {v}"))?;
        let tok = |k: &str| -> Result<u32> { v[k][1].as_u64().map(|x| x as u32).ok_or_else(|| anyhow!("{ty}: missing {k}")) };
        Ok(match ty {
            "TemplateProcessing" => {
                let specials = &v["special_tokens"];
                PostProcessor::Template {
                    single: template(&v["single"], specials, "$0")?,
                    pair: template(&v["pair"], specials, "$A:0 $B:1")?,
                }
            }
            "RobertaProcessing" => PostProcessor::Roberta {
                cls: tok("cls")?,
                sep: tok("sep")?,
            },
            "BertProcessing" => PostProcessor::Bert {
                cls: tok("cls")?,
                sep: tok("sep")?,
            },
            "ByteLevel" => PostProcessor::ByteLevel,
            "Sequence" => PostProcessor::Sequence(
                v["processors"]
                    .as_array()
                    .map(|a| a.iter().map(PostProcessor::from_json).collect())
                    .unwrap_or(Ok(vec![]))?,
            ),
            t => bail!("unsupported post-processor {t}"),
        })
    }

    /// How many special tokens a single sequence (or a pair) gets.
    pub fn added_tokens(&self, is_pair: bool) -> usize {
        match self {
            PostProcessor::Template { single, pair } => (if is_pair { pair } else { single })
                .iter()
                .map(|p| if let Piece::Special(ids, _) = p { ids.len() } else { 0 })
                .sum(),
            PostProcessor::Bert { .. } => 2 + is_pair as usize,
            PostProcessor::Roberta { .. } => 2 + 2 * is_pair as usize,
            PostProcessor::ByteLevel => 0,
            PostProcessor::Sequence(v) => v.iter().map(|p| p.added_tokens(is_pair)).sum(),
        }
    }

    /// Processes one encoding (or a pair) into the final one: each gets its sequence's type id,
    /// the processor adds its special tokens, then the parts are concatenated.
    pub fn process(&self, mut encoding: Encoding, pair: Option<Encoding>, add_special_tokens: bool) -> Result<Encoding> {
        encoding.type_ids = vec![0; encoding.ids.len()];
        let mut encodings = vec![encoding];
        if let Some(mut p) = pair {
            p.type_ids = vec![1; p.ids.len()];
            encodings.push(p);
        }
        let encodings = self.process_encodings(encodings, add_special_tokens)?;
        Ok(Encoding::merge(encodings))
    }

    fn process_encodings(&self, encodings: Vec<Encoding>, add_special_tokens: bool) -> Result<Vec<Encoding>> {
        match self {
            PostProcessor::Template { single, pair } => {
                let t = match encodings.len() {
                    1 => single,
                    2 => pair,
                    n => bail!("cannot post-process {n} sequences"),
                };
                let mut out = Vec::with_capacity(t.len());
                for p in t {
                    match p {
                        Piece::Seq(b, type_id) => {
                            let mut e = encodings
                                .get(*b as usize)
                                .ok_or_else(|| anyhow!("the template uses sequence B without a pair"))?
                                .clone();
                            e.type_ids = vec![*type_id; e.ids.len()];
                            out.push(e);
                        }
                        Piece::Special(ids, type_id) if add_special_tokens => out.push(Encoding {
                            ids: ids.clone(),
                            type_ids: vec![*type_id; ids.len()],
                            special: vec![1; ids.len()],
                            attention: vec![1; ids.len()],
                            overflowing: Vec::new(),
                        }),
                        Piece::Special(..) => {}
                    }
                }
                Ok(out)
            }
            PostProcessor::Bert { cls, sep } | PostProcessor::Roberta { cls, sep } => {
                let roberta = matches!(self, PostProcessor::Roberta { .. });
                let mut encodings = encodings;
                if roberta {
                    // RoBERTa has a single type: every sequence (but not its overflowing parts,
                    // which only get fixed when wrapped) becomes type 0, even without specials.
                    for e in &mut encodings {
                        e.type_ids = vec![0; e.ids.len()];
                    }
                }
                if !add_special_tokens {
                    return Ok(encodings);
                }
                Ok(encodings
                    .iter()
                    .enumerate()
                    .map(|(i, e)| match (i, roberta) {
                        (0, true) => wrap(e, &[*cls], *sep, (0, Some(0), 0)),
                        (0, false) => wrap(e, &[*cls], *sep, (0, None, 0)),
                        (_, false) => wrap(e, &[], *sep, (1, None, 1)),
                        (_, true) => wrap(e, &[*sep], *sep, (0, Some(0), 0)),
                    })
                    .collect())
            }
            PostProcessor::ByteLevel => Ok(encodings),
            PostProcessor::Sequence(v) => {
                let mut e = encodings;
                for p in v {
                    e = p.process_encodings(e, add_special_tokens)?;
                }
                Ok(e)
            }
        }
    }
}
