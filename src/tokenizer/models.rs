//! Tokenization models: BPE, Unigram, WordPiece and WordLevel.

use anyhow::{Result, anyhow, bail};
use serde::Deserialize;
use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};
use std::collections::{BinaryHeap, HashMap};
use std::fmt;

/// A model vocabulary as it appears in `tokenizer.json`: a token -> id map (BPE, WordPiece,
/// WordLevel) or a list of (token, score) (Unigram).
pub enum VocabJson {
    Map(Vec<(String, u32)>),
    List(Vec<(String, f64)>),
}

impl<'de> Deserialize<'de> for VocabJson {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = VocabJson;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a vocabulary map or list")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut m: A) -> std::result::Result<VocabJson, A::Error> {
                let mut v = Vec::with_capacity(m.size_hint().unwrap_or(0));
                while let Some((k, id)) = m.next_entry::<String, u32>()? {
                    v.push((k, id));
                }
                Ok(VocabJson::Map(v))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut s: A) -> std::result::Result<VocabJson, A::Error> {
                let mut v = Vec::with_capacity(s.size_hint().unwrap_or(0));
                while let Some(e) = s.next_element::<(String, f64)>()? {
                    v.push(e);
                }
                Ok(VocabJson::List(v))
            }
        }
        d.deserialize_any(V)
    }
}

/// A BPE merge: `"a b"` (legacy) or `["a", "b"]`.
pub struct MergeJson(pub String, pub String);

impl<'de> Deserialize<'de> for MergeJson {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = MergeJson;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a merge (\"a b\" or [\"a\", \"b\"])")
            }
            fn visit_str<E: de::Error>(self, s: &str) -> std::result::Result<MergeJson, E> {
                let mut it = s.split(' ');
                match (it.next(), it.next(), it.next()) {
                    (Some(a), Some(b), None) => Ok(MergeJson(a.to_string(), b.to_string())),
                    _ => Err(E::custom(format!("bad merge {s:?}"))),
                }
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut s: A) -> std::result::Result<MergeJson, A::Error> {
                let a: String = s.next_element()?.ok_or_else(|| de::Error::custom("empty merge"))?;
                let b: String = s.next_element()?.ok_or_else(|| de::Error::custom("merge with one part"))?;
                Ok(MergeJson(a, b))
            }
        }
        d.deserialize_any(V)
    }
}

/// The `model` object of `tokenizer.json`.
#[derive(Deserialize)]
pub struct ModelJson {
    #[serde(rename = "type")]
    ty: Option<String>,
    vocab: Option<VocabJson>,
    #[serde(default)]
    merges: Option<Vec<MergeJson>>,
    #[serde(default)]
    unk_token: Option<String>,
    #[serde(default)]
    unk_id: Option<usize>,
    #[serde(default)]
    continuing_subword_prefix: Option<String>,
    #[serde(default)]
    end_of_word_suffix: Option<String>,
    #[serde(default)]
    fuse_unk: Option<bool>,
    #[serde(default)]
    byte_fallback: Option<bool>,
    #[serde(default)]
    ignore_merges: Option<bool>,
    #[serde(default)]
    max_input_chars_per_word: Option<usize>,
}

#[derive(Clone)]
pub enum Model {
    Bpe(Bpe),
    Unigram(Unigram),
    WordPiece(WordPiece),
    WordLevel(WordLevel),
}

impl Model {
    pub fn from_json(m: ModelJson) -> Result<Model> {
        let ty = match m.ty.as_deref() {
            Some(t) => t.to_string(),
            None => match (&m.vocab, &m.merges) {
                (Some(VocabJson::List(_)), _) => "Unigram".into(),
                (_, Some(_)) => "BPE".into(),
                _ if m.continuing_subword_prefix.is_some() => "WordPiece".into(),
                _ => "WordLevel".into(),
            },
        };
        let vocab = m.vocab.ok_or_else(|| anyhow!("model without a vocabulary"))?;
        let map = |v: VocabJson| -> Result<Vec<(String, u32)>> {
            match v {
                VocabJson::Map(v) => Ok(v),
                VocabJson::List(_) => bail!("{ty} model with a list vocabulary"),
            }
        };
        Ok(match ty.as_str() {
            "BPE" => Model::Bpe(Bpe::new(
                map(vocab)?,
                m.merges.unwrap_or_default(),
                BpeOptions {
                    unk_token: m.unk_token,
                    prefix: m.continuing_subword_prefix.filter(|p| !p.is_empty()),
                    suffix: m.end_of_word_suffix.filter(|p| !p.is_empty()),
                    fuse_unk: m.fuse_unk.unwrap_or(false),
                    byte_fallback: m.byte_fallback.unwrap_or(false),
                    ignore_merges: m.ignore_merges.unwrap_or(false),
                },
            )?),
            "Unigram" => match vocab {
                VocabJson::List(v) => Model::Unigram(Unigram::new(v, m.unk_id, m.byte_fallback.unwrap_or(false))?),
                VocabJson::Map(_) => bail!("Unigram model with a map vocabulary"),
            },
            "WordPiece" => Model::WordPiece(WordPiece {
                vocab: Vocab::new(map(vocab)?),
                unk_token: m.unk_token.unwrap_or_else(|| "[UNK]".into()),
                prefix: m.continuing_subword_prefix.unwrap_or_else(|| "##".into()),
                max_chars: m.max_input_chars_per_word.unwrap_or(100),
            }),
            "WordLevel" => Model::WordLevel(WordLevel {
                vocab: Vocab::new(map(vocab)?),
                unk_token: m.unk_token.unwrap_or_else(|| "<unk>".into()),
            }),
            t => bail!("unsupported model {t}"),
        })
    }

    /// Appends the ids of `word` (one pre-tokenized piece) to `out`.
    pub fn tokenize(&self, word: &str, out: &mut Vec<u32>) -> Result<()> {
        if word.is_empty() {
            return Ok(());
        }
        match self {
            Model::Bpe(m) => m.tokenize(word, out),
            Model::Unigram(m) => m.tokenize(word, out),
            Model::WordPiece(m) => m.tokenize(word, out),
            Model::WordLevel(m) => m.tokenize(word, out),
        }
    }

    pub fn token_to_id(&self, t: &str) -> Option<u32> {
        match self {
            Model::Bpe(m) => m.vocab.id(t),
            Model::Unigram(m) => m.ids.get(t).copied(),
            Model::WordPiece(m) => m.vocab.id(t),
            Model::WordLevel(m) => m.vocab.id(t),
        }
    }

    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        match self {
            Model::Bpe(m) => m.vocab.token(id),
            Model::Unigram(m) => m.pieces.get(id as usize).map(|(t, _)| t.as_str()),
            Model::WordPiece(m) => m.vocab.token(id),
            Model::WordLevel(m) => m.vocab.token(id),
        }
    }

    pub fn vocab_size(&self) -> usize {
        match self {
            Model::Bpe(m) => m.vocab.len(),
            Model::Unigram(m) => m.pieces.len(),
            Model::WordPiece(m) => m.vocab.len(),
            Model::WordLevel(m) => m.vocab.len(),
        }
    }
}

/// A token <-> id map.
#[derive(Clone)]
pub struct Vocab {
    ids: HashMap<String, u32>,
    tokens: HashMap<u32, String>,
}

impl Vocab {
    fn new(v: Vec<(String, u32)>) -> Vocab {
        let mut ids = HashMap::with_capacity(v.len());
        let mut tokens = HashMap::with_capacity(v.len());
        for (t, id) in v {
            tokens.insert(id, t.clone());
            ids.insert(t, id);
        }
        Vocab { ids, tokens }
    }

    fn id(&self, t: &str) -> Option<u32> {
        self.ids.get(t).copied()
    }

    fn token(&self, id: u32) -> Option<&str> {
        self.tokens.get(&id).map(String::as_str)
    }

    fn len(&self) -> usize {
        self.ids.len()
    }
}

#[derive(Clone)]
pub struct BpeOptions {
    unk_token: Option<String>,
    prefix: Option<String>,
    suffix: Option<String>,
    fuse_unk: bool,
    byte_fallback: bool,
    ignore_merges: bool,
}

#[derive(Clone)]
pub struct Bpe {
    vocab: Vocab,
    /// (left, right) -> (rank, merged id).
    merges: HashMap<(u32, u32), (u32, u32)>,
    o: BpeOptions,
    /// `<0xXX>` byte tokens (byte fallback).
    bytes: Box<[Option<u32>; 256]>,
}

impl Bpe {
    fn new(vocab: Vec<(String, u32)>, merges: Vec<MergeJson>, o: BpeOptions) -> Result<Bpe> {
        let vocab = Vocab::new(vocab);
        let plen = o.prefix.as_ref().map_or(0, String::len);
        let mut map = HashMap::with_capacity(merges.len());
        for (rank, MergeJson(a, b)) in merges.into_iter().enumerate() {
            let ia = vocab.id(&a).ok_or_else(|| anyhow!("merge token {a:?} is not in the vocabulary"))?;
            let ib = vocab.id(&b).ok_or_else(|| anyhow!("merge token {b:?} is not in the vocabulary"))?;
            let merged = format!("{a}{}", b.get(plen..).unwrap_or(""));
            let im = vocab
                .id(&merged)
                .ok_or_else(|| anyhow!("merged token {merged:?} is not in the vocabulary"))?;
            map.insert((ia, ib), (rank as u32, im));
        }
        let mut bytes = Box::new([None; 256]);
        for (b, slot) in bytes.iter_mut().enumerate() {
            *slot = vocab.id(&format!("<0x{b:02X}>"));
        }
        Ok(Bpe {
            vocab,
            merges: map,
            o,
            bytes,
        })
    }

    fn tokenize(&self, word: &str, out: &mut Vec<u32>) -> Result<()> {
        if self.o.ignore_merges
            && let Some(id) = self.vocab.id(word)
        {
            out.push(id);
            return Ok(());
        }
        // Initial symbols: one per char (prefixed / suffixed as configured), byte fallback or
        // unknown tokens for the rest.
        let mut syms: Vec<u32> = Vec::with_capacity(word.len());
        let mut unk: Option<u32> = None;
        let unk_id = || -> Result<u32> {
            let t = self.o.unk_token.as_deref().unwrap_or_default();
            self.vocab
                .id(t)
                .ok_or_else(|| anyhow!("unknown token {t:?} is not in the vocabulary"))
        };
        let mut buf = String::new();
        let mut it = word.char_indices().peekable();
        while let Some((i, c)) = it.next() {
            let is_last = it.peek().is_none();
            let s = &word[i..i + c.len_utf8()];
            let key: &str = if (i > 0 && self.o.prefix.is_some()) || (is_last && self.o.suffix.is_some()) {
                buf.clear();
                if i > 0
                    && let Some(p) = &self.o.prefix
                {
                    buf.push_str(p);
                }
                buf.push_str(s);
                if is_last && let Some(sf) = &self.o.suffix {
                    buf.push_str(sf);
                }
                &buf
            } else {
                s
            };
            if let Some(id) = self.vocab.id(key) {
                if let Some(u) = unk.take() {
                    syms.push(u);
                }
                syms.push(id);
                continue;
            }
            if self.o.byte_fallback {
                let ids: Option<Vec<u32>> = key.bytes().map(|b| self.bytes[b as usize]).collect();
                if let Some(ids) = ids {
                    // The reference implementation leaves a pending unknown token in place here.
                    syms.extend(ids);
                    continue;
                }
            }
            if self.o.unk_token.is_some() {
                match (unk, self.o.fuse_unk) {
                    (Some(_), true) => {}
                    (Some(u), false) => {
                        syms.push(u);
                        unk = Some(unk_id()?);
                    }
                    (None, _) => unk = Some(unk_id()?),
                }
            }
        }
        if let Some(u) = unk {
            syms.push(u);
        }
        self.merge(&mut syms);
        out.extend_from_slice(&syms);
        Ok(())
    }

    /// Applies merges by rank (lowest first, leftmost on ties) until none applies.
    fn merge(&self, syms: &mut Vec<u32>) {
        let n = syms.len();
        if n < 2 {
            return;
        }
        let mut prev: Vec<isize> = (0..n as isize).map(|i| i - 1).collect();
        let mut next: Vec<isize> = (0..n as isize).map(|i| if i + 1 < n as isize { i + 1 } else { -1 }).collect();
        let mut alive = vec![true; n];
        let mut heap: BinaryHeap<std::cmp::Reverse<(u32, usize, u32)>> = BinaryHeap::new();
        for i in 0..n - 1 {
            if let Some(&(rank, id)) = self.merges.get(&(syms[i], syms[i + 1])) {
                heap.push(std::cmp::Reverse((rank, i, id)));
            }
        }
        while let Some(std::cmp::Reverse((rank, pos, new_id))) = heap.pop() {
            if !alive[pos] || next[pos] < 0 {
                continue;
            }
            let right = next[pos] as usize;
            match self.merges.get(&(syms[pos], syms[right])) {
                Some(&(r, id)) if id == new_id && r == rank => {}
                _ => continue,
            }
            syms[pos] = new_id;
            alive[right] = false;
            next[pos] = next[right];
            if next[right] >= 0 {
                prev[next[right] as usize] = pos as isize;
            }
            if prev[pos] >= 0 {
                let p = prev[pos] as usize;
                if let Some(&(r, id)) = self.merges.get(&(syms[p], syms[pos])) {
                    heap.push(std::cmp::Reverse((r, p, id)));
                }
            }
            if next[pos] >= 0 {
                let q = next[pos] as usize;
                if let Some(&(r, id)) = self.merges.get(&(syms[pos], syms[q])) {
                    heap.push(std::cmp::Reverse((r, pos, id)));
                }
            }
        }
        let mut k = 0;
        for i in 0..n {
            if alive[i] {
                syms[k] = syms[i];
                k += 1;
            }
        }
        syms.truncate(k);
    }
}

#[derive(Clone)]
pub struct WordPiece {
    vocab: Vocab,
    unk_token: String,
    prefix: String,
    max_chars: usize,
}

impl WordPiece {
    fn tokenize(&self, word: &str, out: &mut Vec<u32>) -> Result<()> {
        let unk = || {
            self.vocab
                .id(&self.unk_token)
                .ok_or_else(|| anyhow!("unknown token {:?} is not in the vocabulary", self.unk_token))
        };
        if word.chars().count() > self.max_chars {
            out.push(unk()?);
            return Ok(());
        }
        let mut subs = Vec::new();
        let mut start = 0;
        let mut buf = String::new();
        while start < word.len() {
            let mut end = word.len();
            let mut found = None;
            while start < end {
                let sub = &word[start..end];
                let key = if start > 0 {
                    buf.clear();
                    buf.push_str(&self.prefix);
                    buf.push_str(sub);
                    buf.as_str()
                } else {
                    sub
                };
                if let Some(id) = self.vocab.id(key) {
                    found = Some(id);
                    break;
                }
                end -= sub.chars().next_back().map_or(1, char::len_utf8);
            }
            match found {
                Some(id) => subs.push(id),
                None => {
                    out.push(unk()?);
                    return Ok(());
                }
            }
            start = end;
        }
        out.extend(subs);
        Ok(())
    }
}

#[derive(Clone)]
pub struct WordLevel {
    vocab: Vocab,
    unk_token: String,
}

impl WordLevel {
    fn tokenize(&self, word: &str, out: &mut Vec<u32>) -> Result<()> {
        match self.vocab.id(word).or_else(|| self.vocab.id(&self.unk_token)) {
            Some(id) => {
                out.push(id);
                Ok(())
            }
            None => bail!("unknown token {:?} is not in the vocabulary", self.unk_token),
        }
    }
}

#[derive(Clone)]
pub struct Unigram {
    pieces: Vec<(String, f64)>,
    ids: HashMap<String, u32>,
    /// Distinct pieces sorted by bytes, with the id the piece maps to (the last duplicate).
    sorted: Vec<u32>,
    /// The range of `sorted` whose pieces start with each byte.
    first: Box<[(u32, u32); 256]>,
    unk_id: Option<usize>,
    byte_fallback: bool,
    min_score: f64,
}

const UNK_PENALTY: f64 = 10.0;

impl Unigram {
    fn new(pieces: Vec<(String, f64)>, unk_id: Option<usize>, byte_fallback: bool) -> Result<Unigram> {
        if let Some(u) = unk_id
            && u >= pieces.len()
        {
            bail!("Unigram unk_id {u} is out of the vocabulary");
        }
        let mut ids = HashMap::with_capacity(pieces.len());
        let mut min_score = f64::INFINITY;
        for (i, (t, s)) in pieces.iter().enumerate() {
            ids.insert(t.clone(), i as u32);
            min_score = min_score.min(*s);
        }
        let mut sorted: Vec<u32> = ids.values().copied().collect();
        sorted.sort_unstable_by(|&a, &b| pieces[a as usize].0.as_bytes().cmp(pieces[b as usize].0.as_bytes()));
        let mut first = Box::new([(0u32, 0u32); 256]);
        for (i, &id) in sorted.iter().enumerate() {
            if let Some(&b) = pieces[id as usize].0.as_bytes().first() {
                let r = &mut first[b as usize];
                if r.1 == 0 {
                    r.0 = i as u32;
                }
                r.1 = i as u32 + 1;
            }
        }
        Ok(Unigram {
            pieces,
            ids,
            sorted,
            first,
            unk_id,
            byte_fallback,
            min_score,
        })
    }

    fn key(&self, i: usize) -> &[u8] {
        self.pieces[self.sorted[i] as usize].0.as_bytes()
    }

    /// Calls `f(len, id)` for every vocabulary piece that is a prefix of `s`, shortest first.
    fn prefixes(&self, s: &[u8], mut f: impl FnMut(usize, u32)) {
        let Some(&c0) = s.first() else { return };
        let (lo0, hi0) = self.first[c0 as usize];
        if lo0 == hi0 {
            return;
        }
        let (mut lo, mut hi) = (lo0 as usize, hi0 as usize);
        if self.key(lo).len() == 1 {
            f(1, self.sorted[lo]);
        }
        for (d, &c) in s.iter().enumerate().skip(1) {
            // Keys in [lo, hi) share the prefix s[..d]; a key of length d sorts first.
            let start = lo
                + self.sorted[lo..hi].partition_point(|&id| {
                    let k = self.pieces[id as usize].0.as_bytes();
                    k.len() <= d || k[d] < c
                });
            let end = start + self.sorted[start..hi].partition_point(|&id| self.pieces[id as usize].0.as_bytes()[d] <= c);
            if start == end {
                return;
            }
            lo = start;
            hi = end;
            if self.key(lo).len() == d + 1 {
                f(d + 1, self.sorted[lo]);
            }
        }
    }

    fn tokenize(&self, s: &str, out: &mut Vec<u32>) -> Result<()> {
        #[derive(Clone, Copy)]
        struct Node {
            id: usize,
            score: f64,
            start: Option<usize>,
        }
        let n = s.len();
        let unk_score = self.min_score - UNK_PENALTY;
        let mut best = vec![
            Node {
                id: 0,
                score: 0.0,
                start: None
            };
            n + 1
        ];
        let b = s.as_bytes();
        let mut at = 0;
        while at < n {
            let base = best[at].score;
            let mblen = s[at..].chars().next().map_or(1, char::len_utf8);
            let mut single = false;
            self.prefixes(&b[at..], |len, id| {
                let t = &mut best[at + len];
                let cand = self.pieces[id as usize].1 + base;
                if t.start.is_none() || cand > t.score {
                    *t = Node {
                        id: id as usize,
                        score: cand,
                        start: Some(at),
                    };
                }
                if len == mblen {
                    single = true;
                }
            });
            if !single {
                let t = &mut best[at + mblen];
                let cand = unk_score + base;
                if t.start.is_none() || cand > t.score {
                    let unk = self.unk_id.ok_or_else(|| anyhow!("Unigram model without an unknown token"))?;
                    *t = Node {
                        id: unk,
                        score: cand,
                        start: Some(at),
                    };
                }
            }
            at += mblen;
        }
        // Back-track, fusing consecutive unknown pieces.
        let mut pieces: Vec<(usize, usize)> = Vec::new();
        let mut end = n;
        let mut unk_end: Option<usize> = None;
        while end > 0 {
            let node = best[end];
            let start = node.start.unwrap();
            if Some(node.id) == self.unk_id {
                unk_end.get_or_insert(end);
            } else {
                if let Some(ue) = unk_end.take() {
                    pieces.push((end, ue));
                }
                pieces.push((start, end));
            }
            end = start;
        }
        if let Some(ue) = unk_end {
            pieces.push((0, ue));
        }
        for &(a, e) in pieces.iter().rev() {
            let t = &s[a..e];
            if let Some(&id) = self.ids.get(t) {
                out.push(id);
                continue;
            }
            if self.byte_fallback {
                let ids: Option<Vec<u32>> = t.bytes().map(|b| self.ids.get(&format!("<0x{b:02X}>")).copied()).collect();
                if let Some(ids) = ids {
                    out.extend(ids);
                    continue;
                }
            }
            out.push(self.unk_id.ok_or_else(|| anyhow!("Unigram model without an unknown token"))? as u32);
        }
        Ok(())
    }
}
