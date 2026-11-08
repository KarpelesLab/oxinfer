//! A small backtracking regular-expression engine for the patterns `tokenizer.json` files carry
//! (pre-tokenizer splits, `Replace` normalizers and decoders).
//!
//! The syntax is the Perl / Oniguruma subset those patterns use: literals and escapes, `.`,
//! classes (`[...]`, `\s \d \w`, `\p{..}` general categories), groups (capturing, `(?:..)`,
//! `(?i:..)`, inline `(?i)`), lookahead (`(?=..)`, `(?!..)`), greedy and lazy quantifiers, and the
//! `^ $ \A \z \b \B` anchors. Matching follows backtracking (leftmost-first) semantics, the same
//! as the `fancy-regex` engine the reference tokenizers library uses, so alternation order and
//! greediness give the same splits. `$` is the end of the text (no multi-line mode).

use super::unicode::{Gc, gc_mask, is_word_char};
use anyhow::{Result, bail};

#[derive(Clone, Debug)]
enum Item {
    Range(char, char),
    Gc(u64, bool),
    Space(bool),
    Digit(bool),
    Word(bool),
    Alphabetic(bool),
    Class(Class),
}

#[derive(Clone, Debug)]
struct Class {
    items: Vec<Item>,
    negated: bool,
    fold: bool,
}

impl Item {
    fn matches(&self, c: char) -> bool {
        match self {
            Item::Range(a, b) => *a <= c && c <= *b,
            Item::Gc(m, neg) => (gc_mask(c) & m != 0) != *neg,
            Item::Space(neg) => c.is_whitespace() != *neg,
            Item::Digit(neg) => (gc_mask(c) & Gc::ND != 0) != *neg,
            Item::Word(neg) => is_word_char(c) != *neg,
            Item::Alphabetic(neg) => intl::unicode::is_alphabetic(c) != *neg,
            Item::Class(k) => k.matches(c),
        }
    }
}

impl Class {
    fn matches(&self, c: char) -> bool {
        let hit = |c: char| self.items.iter().any(|i| i.matches(c));
        let mut m = hit(c);
        if !m && self.fold {
            m = fold_variants(c).any(hit);
        }
        m != self.negated
    }
}

/// The other-case forms of `c` (single-char mappings only), for case-insensitive matching.
fn fold_variants(c: char) -> impl Iterator<Item = char> {
    let one = |mut it: std::char::ToLowercase| -> Option<char> {
        let a = it.next()?;
        it.next().is_none().then_some(a)
    };
    let lo = one(c.to_lowercase()).filter(|&l| l != c);
    let mut up = c.to_uppercase();
    let u = match (up.next(), up.next()) {
        (Some(u), None) if u != c => Some(u),
        _ => None,
    };
    lo.into_iter().chain(u)
}

#[derive(Clone, Debug)]
enum Node {
    Empty,
    Char(char, bool),
    Any,
    Class(Class),
    Start,
    End,
    WordBoundary(bool),
    Concat(Vec<Node>),
    Alt(Vec<Node>),
    Repeat { node: Box<Node>, min: u32, max: u32, greedy: bool },
    Look { node: Box<Node>, negated: bool },
}

impl Node {
    /// Whether the node matches exactly one character (a fast path for repetitions).
    fn single(&self) -> bool {
        matches!(self, Node::Char(..) | Node::Any | Node::Class(_))
    }
}

/// A compiled pattern.
#[derive(Clone, Debug)]
pub struct Regex {
    root: Node,
}

struct Parser<'a> {
    s: &'a [char],
    i: usize,
    fold: bool,
    src: &'a str,
}

impl Parser<'_> {
    fn peek(&self) -> Option<char> {
        self.s.get(self.i).copied()
    }

    fn eat(&mut self, c: char) -> bool {
        if self.peek() == Some(c) {
            self.i += 1;
            true
        } else {
            false
        }
    }

    fn next(&mut self) -> Result<char> {
        let c = self.peek().ok_or_else(|| anyhow::anyhow!("regex {:?}: unexpected end", self.src))?;
        self.i += 1;
        Ok(c)
    }

    fn alt(&mut self) -> Result<Node> {
        let mut alts = vec![self.concat()?];
        while self.eat('|') {
            alts.push(self.concat()?);
        }
        Ok(if alts.len() == 1 { alts.pop().unwrap() } else { Node::Alt(alts) })
    }

    fn concat(&mut self) -> Result<Node> {
        let mut seq = Vec::new();
        while let Some(c) = self.peek() {
            if c == '|' || c == ')' {
                break;
            }
            let atom = match self.atom()? {
                Some(a) => a,
                None => continue,
            };
            let atom = self.quantifier(atom)?;
            seq.push(atom);
        }
        Ok(match seq.len() {
            0 => Node::Empty,
            1 => seq.pop().unwrap(),
            _ => Node::Concat(seq),
        })
    }

    fn quantifier(&mut self, atom: Node) -> Result<Node> {
        let (min, max) = match self.peek() {
            Some(c @ ('*' | '+' | '?')) => {
                self.i += 1;
                match c {
                    '*' => (0, u32::MAX),
                    '+' => (1, u32::MAX),
                    _ => (0, 1),
                }
            }
            Some('{') => match self.bounds() {
                Some(b) => b,
                None => return Ok(atom),
            },
            _ => return Ok(atom),
        };
        let greedy = !self.eat('?');
        if self.peek() == Some('+') {
            bail!("regex {:?}: possessive quantifiers are not supported", self.src);
        }
        Ok(Node::Repeat {
            node: Box::new(atom),
            min,
            max,
            greedy,
        })
    }

    /// Parses `{n}`, `{n,}` or `{n,m}` (consuming it), or returns None (`{` is then a literal).
    fn bounds(&mut self) -> Option<(u32, u32)> {
        let save = self.i;
        self.i += 1;
        let num = |p: &mut Self| -> Option<u32> {
            let st = p.i;
            while matches!(p.peek(), Some('0'..='9')) {
                p.i += 1;
            }
            if p.i == st {
                return None;
            }
            p.s[st..p.i].iter().collect::<String>().parse().ok()
        };
        let a = num(self);
        let r = a.and_then(|a| {
            let b = if self.eat(',') { num(self).unwrap_or(u32::MAX) } else { a };
            (self.peek() == Some('}')).then_some((a, b))
        });
        match r {
            Some(r) => {
                self.i += 1;
                Some(r)
            }
            None => {
                self.i = save;
                None
            }
        }
    }

    fn atom(&mut self) -> Result<Option<Node>> {
        let c = self.next()?;
        Ok(Some(match c {
            '.' => Node::Any,
            '^' => Node::Start,
            '$' => Node::End,
            '[' => Node::Class(self.class()?),
            '(' => return self.group(),
            '\\' => self.escape()?,
            '*' | '+' | '?' => bail!("regex {:?}: quantifier without operand", self.src),
            c => self.lit(c),
        }))
    }

    fn lit(&self, c: char) -> Node {
        Node::Char(c, self.fold && fold_variants(c).next().is_some())
    }

    fn group(&mut self) -> Result<Option<Node>> {
        if self.eat('?') {
            match self.next()? {
                ':' => return self.sub(self.fold).map(Some),
                '=' | '!' => {
                    let negated = self.s[self.i - 1] == '!';
                    let node = self.sub(self.fold)?;
                    return Ok(Some(Node::Look {
                        node: Box::new(node),
                        negated,
                    }));
                }
                '<' if matches!(self.peek(), Some('=' | '!')) => {
                    bail!("regex {:?}: lookbehind is not supported", self.src)
                }
                '<' | 'P' | '\'' => {
                    // Named group: skip the name.
                    if self.s[self.i - 1] == 'P' {
                        self.eat('<');
                    }
                    while let Some(c) = self.peek() {
                        self.i += 1;
                        if c == '>' || c == '\'' {
                            break;
                        }
                    }
                    return self.sub(self.fold).map(Some);
                }
                _ => {
                    // Flags: (?i) (?-i) (?i:..)
                    self.i -= 1;
                    let mut on = true;
                    let mut fold = self.fold;
                    loop {
                        match self.next()? {
                            'i' => fold = on,
                            '-' => on = false,
                            'm' | 's' | 'x' | 'u' => {}
                            ':' => return self.sub(fold).map(Some),
                            ')' => {
                                self.fold = fold;
                                return Ok(None);
                            }
                            f => bail!("regex {:?}: unsupported group flag {f:?}", self.src),
                        }
                    }
                }
            }
        }
        self.sub(self.fold).map(Some)
    }

    fn sub(&mut self, fold: bool) -> Result<Node> {
        let saved = self.fold;
        self.fold = fold;
        let n = self.alt()?;
        self.fold = saved;
        if !self.eat(')') {
            bail!("regex {:?}: unbalanced parenthesis", self.src);
        }
        Ok(n)
    }

    fn hex(&mut self, n: usize) -> Result<char> {
        let mut v = 0u32;
        if self.eat('{') {
            while !self.eat('}') {
                v = v * 16
                    + self
                        .next()?
                        .to_digit(16)
                        .ok_or_else(|| anyhow::anyhow!("regex {:?}: bad hex", self.src))?;
            }
        } else {
            for _ in 0..n {
                v = v * 16
                    + self
                        .next()?
                        .to_digit(16)
                        .ok_or_else(|| anyhow::anyhow!("regex {:?}: bad hex", self.src))?;
            }
        }
        char::from_u32(v).ok_or_else(|| anyhow::anyhow!("regex {:?}: bad code point", self.src))
    }

    /// An escape after `\`: a class item or a literal char.
    fn escape_item(&mut self) -> Result<Result<Item, char>> {
        let c = self.next()?;
        Ok(match c {
            's' => Ok(Item::Space(false)),
            'S' => Ok(Item::Space(true)),
            'd' => Ok(Item::Digit(false)),
            'D' => Ok(Item::Digit(true)),
            'w' => Ok(Item::Word(false)),
            'W' => Ok(Item::Word(true)),
            'p' | 'P' => {
                let name: String = if self.eat('{') {
                    let mut n = String::new();
                    while !self.eat('}') {
                        n.push(self.next()?);
                    }
                    n
                } else {
                    self.next()?.to_string()
                };
                let (name, neg2) = match name.strip_prefix('^') {
                    Some(n) => (n.to_string(), true),
                    None => (name, false),
                };
                let neg = (c == 'P') != neg2;
                Ok(property(&name, neg).ok_or_else(|| anyhow::anyhow!("regex {:?}: unknown property {name:?}", self.src))?)
            }
            'n' => Err('\n'),
            'r' => Err('\r'),
            't' => Err('\t'),
            'f' => Err('\x0c'),
            'v' => Err('\x0b'),
            'a' => Err('\x07'),
            'e' => Err('\x1b'),
            '0' => Err('\0'),
            'x' => Err(self.hex(2)?),
            'u' => Err(self.hex(4)?),
            c if c.is_ascii_alphanumeric() => bail!("regex {:?}: unsupported escape \\{c}", self.src),
            c => Err(c),
        })
    }

    fn escape(&mut self) -> Result<Node> {
        match self.peek() {
            Some('A') => {
                self.i += 1;
                return Ok(Node::Start);
            }
            Some('z' | 'Z') => {
                self.i += 1;
                return Ok(Node::End);
            }
            Some('b') => {
                self.i += 1;
                return Ok(Node::WordBoundary(true));
            }
            Some('B') => {
                self.i += 1;
                return Ok(Node::WordBoundary(false));
            }
            _ => {}
        }
        Ok(match self.escape_item()? {
            Ok(item) => Node::Class(Class {
                items: vec![item],
                negated: false,
                fold: false,
            }),
            Err(c) => self.lit(c),
        })
    }

    fn class(&mut self) -> Result<Class> {
        let negated = self.eat('^');
        let mut items = Vec::new();
        let mut first = true;
        loop {
            let c = self.next()?;
            if c == ']' && !first {
                break;
            }
            first = false;
            let lo = match c {
                '[' if self.peek() == Some(':') => bail!("regex {:?}: POSIX classes are not supported", self.src),
                '[' => {
                    items.push(Item::Class(self.class()?));
                    continue;
                }
                '&' if self.peek() == Some('&') => bail!("regex {:?}: class intersection is not supported", self.src),
                '\\' => match self.escape_item()? {
                    Ok(item) => {
                        items.push(item);
                        continue;
                    }
                    Err(c) => c,
                },
                c => c,
            };
            if self.peek() == Some('-') && self.s.get(self.i + 1).is_some_and(|&n| n != ']') {
                self.i += 1;
                let hi = match self.next()? {
                    '\\' => match self.escape_item()? {
                        Err(c) => c,
                        Ok(_) => bail!("regex {:?}: bad class range", self.src),
                    },
                    c => c,
                };
                items.push(Item::Range(lo, hi));
            } else {
                items.push(Item::Range(lo, lo));
            }
        }
        Ok(Class {
            items,
            negated,
            fold: self.fold,
        })
    }
}

/// A `\p{name}` property as a class item.
fn property(name: &str, neg: bool) -> Option<Item> {
    let norm: String = name
        .chars()
        .filter(|c| !matches!(c, ' ' | '_' | '-'))
        .collect::<String>()
        .to_ascii_lowercase();
    let norm = norm
        .strip_prefix("gc=")
        .or_else(|| norm.strip_prefix("generalcategory="))
        .unwrap_or(&norm)
        .to_string();
    let m = match norm.as_str() {
        "l" | "letter" => Gc::L,
        "lu" | "uppercaseletter" => Gc::LU,
        "ll" | "lowercaseletter" => Gc::LL,
        "lt" | "titlecaseletter" => Gc::LT,
        "lm" | "modifierletter" => Gc::LM,
        "lo" | "otherletter" => Gc::LO,
        "lc" | "l&" | "casedletter" => Gc::LU | Gc::LL | Gc::LT,
        "m" | "mark" | "combiningmark" => Gc::M,
        "mn" | "nonspacingmark" => Gc::MN,
        "mc" | "spacingmark" => Gc::MC,
        "me" | "enclosingmark" => Gc::ME,
        "n" | "number" => Gc::N,
        "nd" | "decimalnumber" | "digit" => Gc::ND,
        "nl" | "letternumber" => Gc::NL,
        "no" | "othernumber" => Gc::NO,
        "p" | "punctuation" | "punct" => Gc::P,
        "pc" | "connectorpunctuation" => Gc::PC,
        "pd" | "dashpunctuation" => Gc::PD,
        "ps" | "openpunctuation" => Gc::PS,
        "pe" | "closepunctuation" => Gc::PE,
        "pi" | "initialpunctuation" => Gc::PI,
        "pf" | "finalpunctuation" => Gc::PF,
        "po" | "otherpunctuation" => Gc::PO,
        "s" | "symbol" => Gc::S,
        "sm" | "mathsymbol" => Gc::SM,
        "sc" | "currencysymbol" => Gc::SC,
        "sk" | "modifiersymbol" => Gc::SK,
        "so" | "othersymbol" => Gc::SO,
        "z" | "separator" => Gc::Z,
        "zs" | "spaceseparator" => Gc::ZS,
        "zl" | "lineseparator" => Gc::ZL,
        "zp" | "paragraphseparator" => Gc::ZP,
        "c" | "other" => Gc::C,
        "cc" | "control" | "cntrl" => Gc::CC,
        "cf" | "format" => Gc::CF,
        "cs" | "surrogate" => Gc::CS,
        "co" | "privateuse" => Gc::CO,
        "cn" | "unassigned" => Gc::CN,
        "alphabetic" | "alpha" => return Some(Item::Alphabetic(neg)),
        "whitespace" | "space" | "wspace" => return Some(Item::Space(neg)),
        "any" => {
            return Some(Item::Range('\0', char::MAX)).map(|r| {
                if neg {
                    Item::Class(Class {
                        items: vec![r],
                        negated: true,
                        fold: false,
                    })
                } else {
                    r
                }
            });
        }
        _ => return None,
    };
    Some(Item::Gc(m, neg))
}

impl Regex {
    pub fn new(pattern: &str) -> Result<Regex> {
        let chars: Vec<char> = pattern.chars().collect();
        let mut p = Parser {
            s: &chars,
            i: 0,
            fold: false,
            src: pattern,
        };
        let root = p.alt()?;
        if p.i != chars.len() {
            bail!("regex {pattern:?}: unbalanced parenthesis");
        }
        Ok(Regex { root })
    }

    /// A pattern matching `s` literally.
    pub fn literal(s: &str) -> Regex {
        Regex {
            root: Node::Concat(s.chars().map(|c| Node::Char(c, false)).collect()),
        }
    }

    /// The leftmost match at or after byte `from`, as a byte range.
    pub fn find_at(&self, text: &str, from: usize) -> Option<(usize, usize)> {
        let m = Matcher { t: text };
        let mut start = from;
        loop {
            let mut end = None;
            if m.node(&self.root, start, &mut |p| {
                end = Some(p);
                true
            }) {
                return end.map(|e| (start, e));
            }
            let c = text[start..].chars().next()?;
            start += c.len_utf8();
        }
    }

    /// Non-overlapping matches, left to right (an empty match right after a previous match is
    /// skipped, as in the `regex` crate).
    pub fn find_iter(&self, text: &str) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let mut pos = 0;
        let mut last_end = None;
        while pos <= text.len() {
            let Some((s, e)) = self.find_at(text, pos) else { break };
            if s == e && last_end == Some(e) {
                match text[s..].chars().next() {
                    Some(c) => {
                        pos = s + c.len_utf8();
                        continue;
                    }
                    None => break,
                }
            }
            out.push((s, e));
            last_end = Some(e);
            pos = if e == s {
                match text[e..].chars().next() {
                    Some(c) => e + c.len_utf8(),
                    None => break,
                }
            } else {
                e
            };
        }
        out
    }

    /// Whether the pattern matches somewhere in `text`.
    pub fn is_match(&self, text: &str) -> bool {
        self.find_at(text, 0).is_some()
    }
}

struct Matcher<'t> {
    t: &'t str,
}

impl Matcher<'_> {
    fn char_at(&self, pos: usize) -> Option<char> {
        self.t.get(pos..)?.chars().next()
    }

    fn one(&self, n: &Node, pos: usize) -> Option<usize> {
        let c = self.char_at(pos)?;
        let ok = match n {
            Node::Char(x, fold) => c == *x || (*fold && fold_variants(c).any(|v| v == *x)),
            Node::Any => c != '\n',
            Node::Class(k) => k.matches(c),
            _ => unreachable!(),
        };
        ok.then(|| pos + c.len_utf8())
    }

    fn is_word_at(&self, pos: usize) -> bool {
        self.char_at(pos).is_some_and(is_word_char)
    }

    fn is_word_before(&self, pos: usize) -> bool {
        self.t[..pos].chars().next_back().is_some_and(is_word_char)
    }

    fn node(&self, n: &Node, pos: usize, k: &mut dyn FnMut(usize) -> bool) -> bool {
        match n {
            Node::Empty => k(pos),
            Node::Char(..) | Node::Any | Node::Class(_) => match self.one(n, pos) {
                Some(p) => k(p),
                None => false,
            },
            Node::Start => pos == 0 && k(pos),
            Node::End => pos == self.t.len() && k(pos),
            Node::WordBoundary(want) => ((self.is_word_before(pos) != self.is_word_at(pos)) == *want) && k(pos),
            Node::Concat(v) => self.seq(v, pos, k),
            Node::Alt(v) => v.iter().any(|a| self.node(a, pos, k)),
            Node::Look { node, negated } => {
                let hit = self.node(node, pos, &mut |_| true);
                hit != *negated && k(pos)
            }
            Node::Repeat { node, min, max, greedy } => {
                if node.single() {
                    let mut ends = vec![pos];
                    let mut p = pos;
                    while (ends.len() as u64) <= *max as u64 {
                        match self.one(node, p) {
                            Some(q) => {
                                p = q;
                                ends.push(p);
                            }
                            None => break,
                        }
                    }
                    let lo = *min as usize;
                    if ends.len() <= lo {
                        return false;
                    }
                    if *greedy {
                        ends[lo..].iter().rev().any(|&e| k(e))
                    } else {
                        ends[lo..].iter().any(|&e| k(e))
                    }
                } else {
                    self.rep(node, *min, *max, *greedy, 0, pos, k)
                }
            }
        }
    }

    fn seq(&self, v: &[Node], pos: usize, k: &mut dyn FnMut(usize) -> bool) -> bool {
        match v.split_first() {
            None => k(pos),
            Some((h, t)) => self.node(h, pos, &mut |p| self.seq(t, p, k)),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn rep(&self, n: &Node, min: u32, max: u32, greedy: bool, count: u32, pos: usize, k: &mut dyn FnMut(usize) -> bool) -> bool {
        let more = |k: &mut dyn FnMut(usize) -> bool| {
            count < max
                && self.node(n, pos, &mut |p| {
                    // An empty iteration cannot make progress: stop repeating.
                    if p == pos && count >= min {
                        false
                    } else {
                        self.rep(n, min, max, greedy, count + 1, p, k)
                    }
                })
        };
        // Greedy: one more iteration first, then stopping here; lazy: the other way round.
        let stop = |k: &mut dyn FnMut(usize) -> bool| count >= min && k(pos);
        if greedy {
            if more(k) {
                return true;
            }
            stop(k)
        } else {
            if stop(k) {
                return true;
            }
            more(k)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all(p: &str, t: &str) -> Vec<String> {
        Regex::new(p)
            .unwrap()
            .find_iter(t)
            .into_iter()
            .map(|(a, b)| t[a..b].to_string())
            .collect()
    }

    #[test]
    fn gpt2_pattern_with_lookahead() {
        let p = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
        assert_eq!(
            all(p, "Hello world's  end   x\n"),
            ["Hello", " world", "'s", " ", " end", "  ", " x", "\n"]
        );
    }

    #[test]
    fn mistral_digit_grouping() {
        let p = r"\p{N}{1,3}(?=(?:\p{N}{3})*(?!\p{N}))|\p{N}{1,2}";
        assert_eq!(all(p, "1234567"), ["1", "234", "567"]);
        assert_eq!(all(p, "12"), ["12"]);
    }

    #[test]
    fn classes_flags_and_anchors() {
        assert_eq!(all(r"(?i:'S|'t)|[a-c]+", "'s'T abcx"), ["'s", "'T", "abc"]);
        assert_eq!(all(r"\s+$", "a  b  "), ["  "]);
        assert_eq!(all(r"[\r\n]", "a\r\nb"), ["\r", "\n"]);
        assert_eq!(all(r"[一-龥]+", "ab漢字c"), ["漢字"]);
        assert_eq!(all(r" {2,}", "a b  c   d"), ["  ", "   "]);
        assert_eq!(all(r"\w+|[^\w\s]+", "hi, you!"), ["hi", ",", "you", "!"]);
        assert_eq!(all(r"a*?b", "aab"), ["aab"]);
        assert_eq!(all(r"x*", "ab"), ["", "", ""]);
    }
}
