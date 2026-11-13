//! A backtracking regular-expression engine for the patterns `tokenizer.json` files carry
//! (pre-tokenizer splits, `Replace` normalizers and decoders).
//!
//! It follows the `fancy-regex` engine the reference tokenizers library uses, syntax and
//! semantics: Perl / Oniguruma style patterns with lookahead and lookbehind (variable length
//! included), atomic groups, greedy / lazy / possessive quantifiers, capture groups (numbered and
//! named), backreferences (`\1`, `\k<name>`, relative `\k<-1>`), conditionals (`(?(1)a|b)`), `\K`,
//! `\G`, the `\A \z \Z \b \B \< \> \b{start}..` assertions, inline flags `i m s x U` (scoped
//! `(?i:..)` or to the end of the pattern `(?i)`) and comments. Matching is leftmost-first
//! backtracking on an explicit stack (no recursion), with fancy-regex's backtrack limit. One
//! place follows Oniguruma (which the Python tokenizers library uses) where fancy-regex differs:
//! `+` after an interval repeats it (`\d{3}+` is `(?:\d{3})+`, not a possessive `\d{3}`).
//!
//! Character classes, `\d \s \w`, `\p{..}` properties (general categories, scripts, script
//! extensions, binary properties, ...) and case-insensitive literals are compiled by
//! `regex-syntax`, exactly as fancy-regex hands them to the `regex` crate, so their Unicode
//! semantics (tables, case folding, set operations `&& -- ~~`, POSIX classes) are the same.
//! Subroutine calls (`\g<1>`, `(?P>name)`) are rejected, as in fancy-regex.

use anyhow::{Result, anyhow, bail};
use regex_syntax::hir::{Class, ClassUnicode, ClassUnicodeRange, HirKind};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::OnceLock;

/// Maximum group nesting.
const MAX_DEPTH: usize = 64;
/// Backtracks allowed per search before giving up (fancy-regex's default).
const BACKTRACK_LIMIT: usize = 1_000_000;
/// Maximum backtrack-stack depth.
const MAX_STACK: usize = 1_000_000;

// ----------------------------------------------------------------------------------------------
// Character sets

/// A set of chars: a bitmap of the BMP plus sorted ranges above it.
#[derive(Clone)]
struct Set {
    bmp: Box<[u64; 1024]>,
    astral: Vec<(u32, u32)>,
}

impl std::fmt::Debug for Set {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Set")
    }
}

impl Set {
    fn new(ranges: impl IntoIterator<Item = (u32, u32)>) -> Set {
        let mut bmp = Box::new([0u64; 1024]);
        let mut astral = Vec::new();
        for (a, b) in ranges {
            let mut lo = a;
            let hi = b.min(0xFFFF);
            while lo <= hi {
                if lo & 63 == 0 && hi - lo >= 63 {
                    bmp[(lo >> 6) as usize] = u64::MAX;
                    lo += 64;
                } else {
                    bmp[(lo >> 6) as usize] |= 1 << (lo & 63);
                    lo += 1;
                }
            }
            if b > 0xFFFF {
                astral.push((a.max(0x10000), b));
            }
        }
        Set { bmp, astral }
    }

    #[inline]
    fn contains(&self, c: char) -> bool {
        let u = c as u32;
        if u < 0x10000 {
            (self.bmp[(u >> 6) as usize] >> (u & 63)) & 1 != 0
        } else {
            self.astral
                .binary_search_by(|&(a, b)| {
                    if b < u {
                        Ordering::Less
                    } else if a > u {
                        Ordering::Greater
                    } else {
                        Ordering::Equal
                    }
                })
                .is_ok()
        }
    }
}

/// Compiles a single-character pattern (a class, `\d`, `\p{..}`, a literal) with `regex-syntax`.
fn compile_set(pattern: &str, casei: bool) -> Result<Set> {
    let hir = regex_syntax::ParserBuilder::new()
        .case_insensitive(casei)
        .build()
        .parse(pattern)
        .map_err(|e| anyhow!("{e}"))?;
    match hir.kind() {
        HirKind::Class(Class::Unicode(c)) => Ok(Set::new(c.ranges().iter().map(|r| (r.start() as u32, r.end() as u32)))),
        HirKind::Literal(l) => {
            let s = std::str::from_utf8(&l.0)?;
            let mut it = s.chars();
            match (it.next(), it.next()) {
                (Some(c), None) => Ok(Set::new([(c as u32, c as u32)])),
                _ => bail!("{pattern:?} is not a single character"),
            }
        }
        _ => bail!("{pattern:?} is not a single-character class"),
    }
}

/// `\w` (Unicode), for the word-boundary assertions.
fn word_set() -> &'static Set {
    static W: OnceLock<Set> = OnceLock::new();
    W.get_or_init(|| compile_set(r"\w", false).expect("\\w"))
}

/// Whether `a` and `b` are equal under simple case folding.
fn eq_fold(a: char, b: char) -> bool {
    if a == b {
        return true;
    }
    let mut c = ClassUnicode::new([ClassUnicodeRange::new(a, a)]);
    c.case_fold_simple();
    c.ranges().iter().any(|r| r.start() <= b && b <= r.end())
}

// ----------------------------------------------------------------------------------------------
// Syntax tree

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Assertion {
    StartText,
    EndText,
    StartLine,
    EndLine,
    Word,
    NotWord,
    WordStart,
    WordEnd,
    WordStartHalf,
    WordEndHalf,
}

#[derive(Debug)]
enum Expr {
    Empty,
    Any {
        nl: bool,
    },
    Assert(Assertion),
    Lit(char),
    Set(Box<Set>),
    Concat(Vec<Expr>),
    Alt(Vec<Expr>),
    Group(usize, Box<Expr>),
    Repeat {
        child: Box<Expr>,
        lo: usize,
        hi: usize,
        greedy: bool,
    },
    Look {
        child: Box<Expr>,
        behind: bool,
        neg: bool,
    },
    Backref {
        group: usize,
        casei: bool,
    },
    Atomic(Box<Expr>),
    KeepOut,
    Continue,
    BackrefExists(usize),
    Cond {
        cond: Box<Expr>,
        yes: Box<Expr>,
        no: Box<Expr>,
    },
}

impl Expr {
    fn repeatable(&self) -> bool {
        !matches!(
            self,
            Expr::Look { .. } | Expr::Empty | Expr::KeepOut | Expr::Continue | Expr::BackrefExists(_)
        )
    }
}

/// What an escape produced: a literal char, a delegated single-char class (pattern text and
/// case-insensitivity), or another expression.
enum Esc {
    Lit(char),
    Class(String, bool),
    Expr(Expr),
}

const F_CASEI: u32 = 1;
const F_MULTI: u32 = 2;
const F_DOTNL: u32 = 4;
const F_SWAP_GREED: u32 = 8;
const F_SPACE: u32 = 16;

fn codepoint_len(b: u8) -> usize {
    match b {
        b if b < 0x80 => 1,
        b if b < 0xE0 => 2,
        b if b < 0xF0 => 3,
        _ => 4,
    }
}

fn parse_decimal(s: &str, ix: usize) -> Option<(usize, usize)> {
    let mut end = ix;
    while end < s.len() && s.as_bytes()[end].is_ascii_digit() {
        end += 1;
    }
    s[ix..end].parse::<usize>().ok().map(|v| (end, v))
}

fn is_id_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// An identifier between `open` and `close`, optionally with a relative `+n` / `-n` suffix:
/// (id, relative, bytes used).
fn parse_id<'a>(s: &'a str, open: &str, close: &str, allow_relative: bool) -> Option<(&'a str, Option<isize>, usize)> {
    if !s.starts_with(open) || s.len() <= open.len() + close.len() {
        return None;
    }
    let id_start = open.len();
    let after = s[id_start..].char_indices().find(|&(_, c)| !is_id_char(c)).map(|(i, _)| i);
    let id_len = match after {
        Some(n) => n,
        None if close.is_empty() => s.len() - id_start,
        None => 0,
    };
    let id_end = id_start + id_len;
    if id_len > 0 && s[id_end..].starts_with(close) {
        return Some((&s[id_start..id_end], None, id_end + close.len()));
    } else if !allow_relative {
        return None;
    }
    let sign = *s.as_bytes().get(id_end)?;
    if sign == b'+' || sign == b'-' {
        let (end, n) = parse_decimal(s, id_end + 1)?;
        if s[end..].starts_with(close) {
            if n == 0 && id_len == 0 {
                return None;
            }
            let n = if sign == b'-' { -(n as isize) } else { n as isize };
            return Some((&s[id_start..id_end], Some(n), end + close.len()));
        }
    }
    None
}

struct Parser<'a> {
    re: &'a str,
    flags: u32,
    groups: usize,
    named: HashMap<String, usize>,
    numeric_backrefs: bool,
    backrefs: Vec<usize>,
}

impl Parser<'_> {
    fn err(&self, ix: usize, msg: impl std::fmt::Display) -> anyhow::Error {
        anyhow!("regex {:?}: {msg} (at byte {ix})", self.re)
    }

    fn flag(&self, f: u32) -> bool {
        self.flags & f != 0
    }

    fn byte(&self, ix: usize) -> Option<u8> {
        self.re.as_bytes().get(ix).copied()
    }

    /// Skips whitespace and `#` comments (in `x` mode) and `(?#..)` comments.
    fn ws(&self, mut ix: usize) -> Result<usize> {
        let b = self.re.as_bytes();
        loop {
            if ix == b.len() {
                return Ok(ix);
            }
            match b[ix] {
                b'#' if self.flag(F_SPACE) => match b[ix..].iter().position(|&c| c == b'\n') {
                    Some(x) => ix += x + 1,
                    None => return Ok(b.len()),
                },
                b' ' | b'\r' | b'\n' | b'\t' if self.flag(F_SPACE) => ix += 1,
                b'(' if b[ix..].starts_with(b"(?#") => {
                    ix += 3;
                    loop {
                        if ix >= b.len() {
                            return Err(self.err(ix, "unclosed comment"));
                        }
                        match b[ix] {
                            b')' => {
                                ix += 1;
                                break;
                            }
                            b'\\' => ix += 2,
                            _ => ix += 1,
                        }
                    }
                }
                _ => return Ok(ix),
            }
        }
    }

    fn parse_re(&mut self, ix: usize, depth: usize) -> Result<(usize, Expr)> {
        let (ix, child) = self.parse_branch(ix, depth)?;
        let mut ix = self.ws(ix)?;
        if self.byte(ix) == Some(b'|') {
            let mut children = vec![child];
            while self.byte(ix) == Some(b'|') {
                let (next, child) = self.parse_branch(ix + 1, depth)?;
                children.push(child);
                ix = self.ws(next)?;
            }
            return Ok((ix, Expr::Alt(children)));
        }
        if self.numeric_backrefs && !self.named.is_empty() {
            bail!("regex {:?}: numbered backreferences cannot be used with named groups", self.re);
        }
        Ok((ix, child))
    }

    fn parse_branch(&mut self, mut ix: usize, depth: usize) -> Result<(usize, Expr)> {
        let mut children = Vec::new();
        while ix < self.re.len() {
            let (next, child) = self.parse_piece(ix, depth)?;
            if next == ix {
                break;
            }
            if !matches!(child, Expr::Empty) {
                children.push(child);
            }
            ix = next;
        }
        Ok(match children.len() {
            0 => (ix, Expr::Empty),
            1 => (ix, children.pop().unwrap()),
            _ => (ix, Expr::Concat(children)),
        })
    }

    fn parse_piece(&mut self, ix: usize, depth: usize) -> Result<(usize, Expr)> {
        let (ix, child) = self.parse_atom(ix, depth)?;
        let mut ix = self.ws(ix)?;
        let Some(b) = self.byte(ix) else { return Ok((ix, child)) };
        let (lo, hi) = match b {
            b'?' => (0, 1),
            b'*' => (0, usize::MAX),
            b'+' => (1, usize::MAX),
            b'{' => match self.parse_repeat(ix) {
                Some((next, lo, hi)) => {
                    ix = next - 1;
                    (lo, hi)
                }
                // Not a valid repetition: `{` is a literal.
                None => return Ok((ix, child)),
            },
            _ => return Ok((ix, child)),
        };
        if !child.repeatable() {
            return Err(self.err(ix, "target of repeat operator is invalid"));
        }
        ix = self.ws(ix + 1)?;
        let mut greedy = true;
        if self.byte(ix) == Some(b'?') {
            greedy = false;
            ix += 1;
        }
        greedy ^= self.flag(F_SWAP_GREED);
        let mut node = Expr::Repeat {
            child: Box::new(child),
            lo,
            hi,
            greedy,
        };
        if self.byte(ix) == Some(b'+') {
            ix += 1;
            // `a++`, `a*+`, `a?+` are possessive. After an interval, `+` repeats it (`\d{3}+` is
            // `(?:\d{3})+`): Oniguruma's Ruby syntax, which the tokenizers library of Python
            // runs its patterns with (fancy-regex reads a possessive interval there), and what
            // tokenizer files are written against (ERNIE 5.5 groups digits with `\p{Nd}{3}+\z`).
            node = if b == b'{' {
                Expr::Repeat {
                    child: Box::new(node),
                    lo: 1,
                    hi: usize::MAX,
                    greedy: true,
                }
            } else {
                Expr::Atomic(Box::new(node))
            };
        }
        Ok((ix, node))
    }

    /// `{n}`, `{n,}`, `{,m}`, `{n,m}` at `ix`: (index after `}`, lo, hi).
    fn parse_repeat(&self, ix: usize) -> Option<(usize, usize, usize)> {
        let ix = self.ws(ix + 1).ok()?;
        let b = self.re.as_bytes();
        if ix == b.len() {
            return None;
        }
        let mut end = ix;
        let lo = if b[ix] == b',' {
            0
        } else {
            let (next, lo) = parse_decimal(self.re, ix)?;
            end = next;
            lo
        };
        let ix = self.ws(end).ok()?;
        if ix == b.len() {
            return None;
        }
        end = ix;
        let hi = match b[ix] {
            b'}' => lo,
            b',' => {
                end = self.ws(ix + 1).ok()?;
                match parse_decimal(self.re, end) {
                    Some((next, hi)) => {
                        end = next;
                        hi
                    }
                    None => usize::MAX,
                }
            }
            _ => return None,
        };
        let ix = self.ws(end).ok()?;
        if ix == b.len() || b[ix] != b'}' {
            return None;
        }
        Some((ix + 1, lo, hi))
    }

    fn literal(&self, c: char) -> Result<Expr> {
        if self.flag(F_CASEI) {
            let mut buf = String::new();
            regex_syntax::escape_into(c.encode_utf8(&mut [0; 4]), &mut buf);
            Ok(Expr::Set(Box::new(compile_set(&buf, true)?)))
        } else {
            Ok(Expr::Lit(c))
        }
    }

    fn class_expr(&self, ix: usize, pattern: &str, casei: bool) -> Result<Expr> {
        compile_set(pattern, casei)
            .map(|s| Expr::Set(Box::new(s)))
            .map_err(|e| self.err(ix, e))
    }

    fn parse_atom(&mut self, ix: usize, depth: usize) -> Result<(usize, Expr)> {
        let ix = self.ws(ix)?;
        let Some(b) = self.byte(ix) else { return Ok((ix, Expr::Empty)) };
        match b {
            b'.' => Ok((ix + 1, Expr::Any { nl: self.flag(F_DOTNL) })),
            b'^' => Ok((
                ix + 1,
                Expr::Assert(if self.flag(F_MULTI) {
                    Assertion::StartLine
                } else {
                    Assertion::StartText
                }),
            )),
            b'$' => Ok((
                ix + 1,
                Expr::Assert(if self.flag(F_MULTI) {
                    Assertion::EndLine
                } else {
                    Assertion::EndText
                }),
            )),
            b'(' => self.parse_group(ix, depth),
            b'\\' => {
                let (end, e) = self.parse_escape(ix, false)?;
                Ok((
                    end,
                    match e {
                        Esc::Lit(c) => self.literal(c)?,
                        Esc::Class(p, casei) => self.class_expr(ix, &p, casei)?,
                        Esc::Expr(e) => e,
                    },
                ))
            }
            b'+' | b'*' | b'?' | b'|' | b')' => Ok((ix, Expr::Empty)),
            b'[' => self.parse_class(ix),
            _ => {
                let c = self.re[ix..].chars().next().unwrap();
                Ok((ix + c.len_utf8(), self.literal(c)?))
            }
        }
    }

    fn backref(&mut self, group: usize) -> Expr {
        self.backrefs.push(group);
        Expr::Backref {
            group,
            casei: self.flag(F_CASEI),
        }
    }

    /// A named / numbered / relative backreference `open id close` at `ix`.
    fn parse_named_backref(&mut self, ix: usize, open: &str, close: &str, allow_relative: bool) -> Result<(usize, Expr)> {
        let re = self.re;
        let (id, mut relative, skip) =
            parse_id(&re[ix..], open, close, allow_relative).ok_or_else(|| self.err(ix, "invalid group name"))?;
        let group = if let Some(&g) = self.named.get(id) {
            Some(g)
        } else if let Ok(g) = id.parse::<usize>() {
            Some(g)
        } else if let Some(r) = relative {
            if id.is_empty() {
                relative = None;
                self.groups.checked_add_signed(if r < 0 { r + 1 } else { r })
            } else {
                None
            }
        } else {
            None
        };
        if relative.is_some() {
            return Err(self.err(ix, "backreferences with a recursion level are not supported"));
        }
        match group {
            Some(g) => Ok((ix + skip, self.backref(g))),
            None => Err(self.err(ix, format!("invalid group name {id:?} in backreference"))),
        }
    }

    fn parse_escape(&mut self, ix: usize, in_class: bool) -> Result<(usize, Esc)> {
        let b = self.byte(ix + 1).ok_or_else(|| self.err(ix, "trailing backslash"))?;
        let end = ix + 1 + codepoint_len(b);
        let e = |x: Expr| Ok((end, Esc::Expr(x)));
        if b.is_ascii_digit() {
            let (end, group) = parse_decimal(self.re, ix + 1)
                .filter(|&(_, g)| g < self.re.len() / 2)
                .ok_or_else(|| self.err(ix, "invalid backreference"))?;
            self.numeric_backrefs = true;
            return Ok((end, Esc::Expr(self.backref(group))));
        }
        if !in_class {
            match b {
                b'k' => {
                    let (end, x) = if self.byte(end) == Some(b'\'') {
                        self.parse_named_backref(end, "'", "'", true)?
                    } else {
                        self.parse_named_backref(end, "<", ">", true)?
                    };
                    return Ok((end, Esc::Expr(x)));
                }
                b'A' => return e(Expr::Assert(Assertion::StartText)),
                b'z' => return e(Expr::Assert(Assertion::EndText)),
                // `\Z`: `(?=\n*$)`.
                b'Z' => {
                    return e(Expr::Look {
                        child: Box::new(Expr::Concat(vec![
                            Expr::Repeat {
                                child: Box::new(Expr::Lit('\n')),
                                lo: 0,
                                hi: usize::MAX,
                                greedy: true,
                            },
                            Expr::Assert(Assertion::EndText),
                        ])),
                        behind: false,
                        neg: false,
                    });
                }
                b'b' | b'B' => {
                    let check = self.ws(end)?;
                    if self.byte(check) == Some(b'{') {
                        let next = self.ws(check + 1)?;
                        if !matches!(self.byte(next), Some(c) if c.is_ascii_digit() || c == b',') {
                            return self.parse_word_boundary_brace(ix);
                        }
                    }
                    return e(Expr::Assert(if b == b'b' { Assertion::Word } else { Assertion::NotWord }));
                }
                b'<' => return e(Expr::Assert(Assertion::WordStart)),
                b'>' => return e(Expr::Assert(Assertion::WordEnd)),
                b'K' => return e(Expr::KeepOut),
                b'G' => return e(Expr::Continue),
                b'O' => return e(Expr::Any { nl: true }),
                b'g' => return Err(self.err(ix, "subroutine calls are not supported")),
                _ => {}
            }
        }
        if matches!(b | 32, b'd' | b's' | b'w') {
            return Ok((end, Esc::Class(self.re[ix..end].to_string(), self.flag(F_CASEI))));
        }
        if b | 32 == b'h' {
            let s = if b == b'h' { "[0-9A-Fa-f]" } else { "[^0-9A-Fa-f]" };
            return Ok((end, Esc::Class(s.to_string(), false)));
        }
        match b {
            b'x' => return self.parse_hex(self.ws(end)?, 2),
            b'u' => return self.parse_hex(self.ws(end)?, 4),
            b'U' => return self.parse_hex(self.ws(end)?, 8),
            _ => {}
        }
        if b | 32 == b'p' && end != self.re.len() {
            let bytes = self.re.as_bytes();
            let nb = bytes[end];
            let mut stop = end + codepoint_len(nb);
            if nb == b'{' {
                loop {
                    let c = *bytes.get(stop).ok_or_else(|| self.err(ix, "unclosed Unicode property name"))?;
                    if c == b'}' {
                        stop += 1;
                        break;
                    }
                    stop += codepoint_len(c);
                }
            }
            return Ok((stop, Esc::Class(self.re[ix..stop].to_string(), self.flag(F_CASEI))));
        }
        let c = match b {
            b'a' => '\x07',
            b'b' => '\x08',
            b'f' => '\x0c',
            b'n' => '\n',
            b'r' => '\r',
            b't' => '\t',
            b'v' => '\x0b',
            b'e' => '\x1b',
            _ => {
                if b.is_ascii_alphabetic() && !matches!(b, b'k' | b'A' | b'z' | b'b' | b'B' | b'<' | b'>' | b'K' | b'G') {
                    return Err(self.err(ix, format!("invalid escape \\{}", &self.re[ix + 1..end])));
                }
                self.re[ix + 1..end].chars().next().unwrap()
            }
        };
        Ok((end, Esc::Lit(c)))
    }

    /// Hex digits after `\x` / `\u` / `\U`: `digits` of them, or `{..}`.
    fn parse_hex(&self, ix: usize, digits: usize) -> Result<(usize, Esc)> {
        let b = self.re.as_bytes();
        let hex = |x: u8| x.is_ascii_hexdigit();
        let lit = |s: &str, end: usize| -> Result<(usize, Esc)> {
            let v = u32::from_str_radix(s, 16).map_err(|_| self.err(ix, "invalid hex escape"))?;
            let c = char::from_u32(v).ok_or_else(|| self.err(ix, "invalid code point"))?;
            Ok((end, Esc::Lit(c)))
        };
        if ix >= b.len() {
            return Err(self.err(ix, "invalid hex escape"));
        }
        if ix + digits <= b.len() && b[ix..ix + digits].iter().all(|&x| hex(x)) {
            return lit(&self.re[ix..ix + digits], ix + digits);
        }
        if b[ix] == b'{' {
            let mut pos = ix + 1;
            let mut s = String::new();
            while pos < b.len() {
                pos = self.ws(pos)?;
                let Some(&x) = b.get(pos) else { break };
                if x == b'}' && !s.is_empty() {
                    return lit(&s, pos + 1);
                }
                if hex(x) && s.len() < 8 {
                    s.push(x as char);
                    pos += 1;
                } else {
                    break;
                }
            }
        }
        Err(self.err(ix, "invalid hex escape"))
    }

    /// `\b{start}`, `\b{end}`, `\b{start-half}`, `\b{end-half}` at `ix`.
    fn parse_word_boundary_brace(&self, ix: usize) -> Result<(usize, Esc)> {
        let b = self.re.as_bytes();
        let start = self.ws(ix + 2)?;
        let mut pos = start + 1;
        let mut content = String::new();
        while pos < b.len() && b[pos] != b'}' {
            let next = self.ws(pos)?;
            if next > pos {
                pos = next;
                continue;
            }
            content.push(b[pos] as char);
            pos += codepoint_len(b[pos]);
        }
        if pos >= b.len() || b[ix + 1] == b'B' {
            return Err(self.err(ix, "invalid word boundary"));
        }
        let a = match content.as_str() {
            "start" => Assertion::WordStart,
            "end" => Assertion::WordEnd,
            "start-half" => Assertion::WordStartHalf,
            "end-half" => Assertion::WordEndHalf,
            _ => return Err(self.err(ix, format!("invalid word boundary \\b{{{content}}}"))),
        };
        Ok((pos + 1, Esc::Expr(Expr::Assert(a))))
    }

    /// A bracketed class: its text (escapes resolved the fancy-regex way) is compiled by
    /// `regex-syntax`.
    fn parse_class(&mut self, ix: usize) -> Result<(usize, Expr)> {
        let b = self.re.as_bytes();
        let start = ix;
        let mut ix = ix + 1;
        let mut class = String::from("[");
        let mut nest = 1;
        if b.get(ix) == Some(&b'^') {
            class.push('^');
            ix += 1;
        }
        if b.get(ix) == Some(&b']') {
            class.push(']');
            ix += 1;
        }
        loop {
            let Some(&c) = b.get(ix) else {
                return Err(self.err(ix, "unclosed character class"));
            };
            ix = match c {
                b'\\' => {
                    let (end, e) = self.parse_escape(ix, true)?;
                    match e {
                        Esc::Lit(c) => regex_syntax::escape_into(c.encode_utf8(&mut [0; 4]), &mut class),
                        Esc::Class(p, _) => class.push_str(&p),
                        Esc::Expr(_) => return Err(self.err(ix, "invalid escape in a character class")),
                    }
                    end
                }
                b'[' => {
                    nest += 1;
                    class.push('[');
                    ix + 1
                }
                b']' => {
                    nest -= 1;
                    class.push(']');
                    if nest == 0 {
                        break;
                    }
                    ix + 1
                }
                c => {
                    let end = ix + codepoint_len(c);
                    class.push_str(&self.re[ix..end]);
                    end
                }
            };
        }
        let e = self.class_expr(start, &class, self.flag(F_CASEI))?;
        Ok((ix + 1, e))
    }

    fn close_paren(&self, ix: usize) -> Result<usize> {
        let ix = self.ws(ix)?;
        match self.byte(ix) {
            Some(b')') => Ok(ix + 1),
            None => Err(self.err(ix, "unclosed group")),
            _ => Err(self.err(ix, "expected a closing parenthesis")),
        }
    }

    fn parse_group(&mut self, ix: usize, depth: usize) -> Result<(usize, Expr)> {
        let depth = depth + 1;
        if depth >= MAX_DEPTH {
            return Err(self.err(ix, "groups nested too deeply"));
        }
        let ix = self.ws(ix + 1)?;
        let re = self.re;
        let rest = &re[ix..];
        enum Kind {
            Look(bool, bool),
            Capture(usize),
            Atomic,
        }
        let (kind, skip) = if rest.starts_with("?=") {
            (Kind::Look(false, false), 2)
        } else if rest.starts_with("?!") {
            (Kind::Look(false, true), 2)
        } else if rest.starts_with("?<=") {
            (Kind::Look(true, false), 3)
        } else if rest.starts_with("?<!") {
            (Kind::Look(true, true), 3)
        } else if rest.starts_with("?<") || rest.starts_with("?'") {
            self.groups += 1;
            let (open, close) = if rest.starts_with("?<") { ("<", ">") } else { ("'", "'") };
            match parse_id(&rest[1..], open, close, false) {
                Some((id, None, skip)) => {
                    self.named.insert(id.to_string(), self.groups);
                    (Kind::Capture(self.groups), skip + 1)
                }
                _ => return Err(self.err(ix, "invalid group name")),
            }
        } else if rest.starts_with("?P<") {
            self.groups += 1;
            match parse_id(&rest[2..], "<", ">", false) {
                Some((id, None, skip)) => {
                    self.named.insert(id.to_string(), self.groups);
                    (Kind::Capture(self.groups), skip + 2)
                }
                _ => return Err(self.err(ix, "invalid group name")),
            }
        } else if rest.starts_with("?P=") {
            return self.parse_named_backref(ix + 3, "", ")", false);
        } else if rest.starts_with("?>") {
            (Kind::Atomic, 2)
        } else if rest.starts_with("?(") {
            return self.parse_conditional(ix + 2, depth);
        } else if rest.starts_with("?P>") {
            return Err(self.err(ix, "subroutine calls are not supported"));
        } else if rest.starts_with('?') {
            return self.parse_flags(ix, depth);
        } else {
            self.groups += 1;
            (Kind::Capture(self.groups), 0)
        };
        let (ix, child) = self.parse_re(ix + skip, depth)?;
        let ix = self.close_paren(ix)?;
        let child = Box::new(child);
        Ok((
            ix,
            match kind {
                Kind::Look(behind, neg) => Expr::Look { child, behind, neg },
                Kind::Capture(g) => Expr::Group(g, child),
                Kind::Atomic => Expr::Atomic(child),
            },
        ))
    }

    /// `(?flags)` (to the end of the pattern) or `(?flags:..)`; `ix` is at the `?`.
    fn parse_flags(&mut self, ix: usize, depth: usize) -> Result<(usize, Expr)> {
        let start = ix + 1;
        let mut ix = start;
        let mut neg = false;
        let old = self.flags;
        loop {
            ix = self.ws(ix)?;
            let b = self.byte(ix).ok_or_else(|| self.err(ix, "unclosed group"))?;
            let f = match b {
                b'i' => F_CASEI,
                b'm' => F_MULTI,
                b's' => F_DOTNL,
                b'U' => F_SWAP_GREED,
                b'x' => F_SPACE,
                _ => 0,
            };
            if f != 0 {
                if neg {
                    self.flags &= !f;
                } else {
                    self.flags |= f;
                }
                ix += 1;
                continue;
            }
            match b {
                b'u' if !neg => {}
                b'-' if !neg => neg = true,
                b')' => {
                    if ix == start || neg && ix == start + 1 {
                        return Err(self.err(ix, "unknown flag"));
                    }
                    return Ok((ix + 1, Expr::Empty));
                }
                b':' => {
                    if neg && ix == start + 1 {
                        return Err(self.err(ix, "unknown flag"));
                    }
                    let (ix, child) = self.parse_re(ix + 1, depth)?;
                    match self.byte(ix) {
                        Some(b')') => {}
                        None => return Err(self.err(ix, "unclosed group")),
                        _ => return Err(self.err(ix, "expected a closing parenthesis")),
                    }
                    self.flags = old;
                    return Ok((ix + 1, child));
                }
                _ => return Err(self.err(ix, "unknown flag")),
            }
            ix += 1;
        }
    }

    /// `(?(cond)yes|no)`; `ix` is after `(?(`.
    fn parse_conditional(&mut self, ix: usize, depth: usize) -> Result<(usize, Expr)> {
        let b = self.byte(ix).ok_or_else(|| self.err(ix, "unclosed group"))?;
        let (next, cond) = if b == b'\'' {
            self.parse_named_backref(ix, "'", "')", true)?
        } else if b == b'<' {
            self.parse_named_backref(ix, "<", ">)", true)?
        } else if b == b'+' || b == b'-' || b.is_ascii_digit() {
            self.parse_named_backref(ix, "", ")", true)?
        } else {
            let (next, cond) = self.parse_re(ix, depth)?;
            (self.close_paren(next)?, cond)
        };
        let (end, child) = self.parse_re(next, depth)?;
        if end == next {
            return match cond {
                Expr::Backref { group, .. } => Ok((self.close_paren(end)?, Expr::BackrefExists(group))),
                _ => Err(self.err(end, "expected a backreference or a branch in the conditional")),
            };
        }
        let (yes, no) = match child {
            Expr::Alt(mut v) => {
                let yes = v.remove(0);
                let no = if v.len() == 1 { v.pop().unwrap() } else { Expr::Alt(v) };
                (yes, no)
            }
            c => (c, Expr::Empty),
        };
        let cond = match cond {
            Expr::Backref { group, .. } => Expr::BackrefExists(group),
            c => c,
        };
        let after = self.close_paren(end)?;
        if matches!((&yes, &no), (Expr::Empty, Expr::Empty)) {
            return Ok((after, cond));
        }
        Ok((
            after,
            Expr::Cond {
                cond: Box::new(cond),
                yes: Box::new(yes),
                no: Box::new(no),
            },
        ))
    }
}

// ----------------------------------------------------------------------------------------------
// Program

/// A single-character matcher (the fast path of greedy repeats).
#[derive(Clone, Copy, Debug)]
enum One {
    Char(char),
    Set(usize),
    Any,
    AnyNoNl,
}

#[derive(Clone, Debug)]
enum Insn {
    Match,
    One(One),
    /// Greedy repetition of one character.
    Star {
        one: One,
        lo: usize,
        hi: usize,
    },
    Assert(Assertion),
    Split(usize, usize),
    Jmp(usize),
    Save(usize),
    Save0(usize),
    Restore(usize),
    RepeatGr {
        lo: usize,
        hi: usize,
        next: usize,
        repeat: usize,
    },
    RepeatNg {
        lo: usize,
        hi: usize,
        next: usize,
        repeat: usize,
    },
    RepeatEpsGr {
        lo: usize,
        next: usize,
        repeat: usize,
        check: usize,
    },
    RepeatEpsNg {
        lo: usize,
        next: usize,
        repeat: usize,
        check: usize,
    },
    GoBack(usize),
    FailNegativeLook,
    Backref {
        slot: usize,
        casei: bool,
    },
    BackrefExists(usize),
    BeginAtomic,
    EndAtomic,
    /// Variable-length lookbehind: sub-program `prog` must match a span ending here (searched
    /// at most `max` chars back).
    Behind {
        prog: usize,
        neg: bool,
        max: Option<usize>,
    },
    Continue,
}

#[derive(Clone, Debug, Default)]
struct Prog {
    insns: Vec<Insn>,
}

/// (minimum size in chars, whether every match has that size).
#[derive(Clone, Copy)]
struct Size {
    min: usize,
    exact: bool,
}

struct Compiler {
    insns: Vec<Insn>,
    n_saves: usize,
    sets: Vec<Set>,
    subs: Vec<Prog>,
    group_sizes: HashMap<usize, Size>,
}

impl Compiler {
    fn pc(&self) -> usize {
        self.insns.len()
    }

    fn new_save(&mut self) -> usize {
        self.n_saves += 1;
        self.n_saves - 1
    }

    fn size(&mut self, e: &Expr) -> Size {
        let exact = |min| Size { min, exact: true };
        match e {
            Expr::Empty | Expr::Assert(_) | Expr::Look { .. } | Expr::KeepOut | Expr::Continue | Expr::BackrefExists(_) => exact(0),
            Expr::Any { .. } | Expr::Lit(_) | Expr::Set(_) => exact(1),
            Expr::Concat(v) => {
                let mut s = exact(0);
                for c in v {
                    let x = self.size(c);
                    s.min = s.min.saturating_add(x.min);
                    s.exact &= x.exact;
                }
                s
            }
            Expr::Alt(v) => {
                let sizes: Vec<Size> = v.iter().map(|c| self.size(c)).collect();
                let min = sizes.iter().map(|s| s.min).min().unwrap_or(0);
                Size {
                    min,
                    exact: sizes.iter().all(|s| s.exact && s.min == min),
                }
            }
            Expr::Group(g, c) => {
                let s = self.size(c);
                self.group_sizes.insert(*g, s);
                s
            }
            Expr::Repeat { child, lo, hi, .. } => {
                let s = self.size(child);
                Size {
                    min: s.min.saturating_mul(*lo),
                    exact: s.exact && lo == hi,
                }
            }
            Expr::Backref { group, .. } => self.group_sizes.get(group).copied().unwrap_or(Size { min: 0, exact: false }),
            Expr::Atomic(c) => self.size(c),
            Expr::Cond { cond, yes, no } => {
                let (c, y, n) = (self.size(cond), self.size(yes), self.size(no));
                let a = c.min + y.min;
                Size {
                    min: a.min(n.min),
                    exact: c.exact && y.exact && n.exact && a == n.min,
                }
            }
        }
    }

    /// Maximum size in chars (None: unbounded or unknown).
    fn max_size(e: &Expr) -> Option<usize> {
        Some(match e {
            Expr::Empty | Expr::Assert(_) | Expr::Look { .. } | Expr::KeepOut | Expr::Continue | Expr::BackrefExists(_) => 0,
            Expr::Any { .. } | Expr::Lit(_) | Expr::Set(_) => 1,
            Expr::Concat(v) => v.iter().map(Self::max_size).try_fold(0usize, |a, m| a.checked_add(m?))?,
            Expr::Alt(v) => v.iter().map(Self::max_size).try_fold(0usize, |a, m| Some(a.max(m?)))?,
            Expr::Group(_, c) | Expr::Atomic(c) => Self::max_size(c)?,
            Expr::Repeat { child, hi, .. } => {
                let m = Self::max_size(child)?;
                if m == 0 {
                    0
                } else if *hi == usize::MAX {
                    return None;
                } else {
                    m.checked_mul(*hi)?
                }
            }
            Expr::Backref { .. } => return None,
            Expr::Cond { cond, yes, no } => (Self::max_size(cond)? + Self::max_size(yes)?).max(Self::max_size(no)?),
        })
    }

    fn one(&mut self, e: &Expr) -> Option<One> {
        Some(match e {
            Expr::Lit(c) => One::Char(*c),
            Expr::Set(s) => {
                self.sets.push((**s).clone());
                One::Set(self.sets.len() - 1)
            }
            Expr::Any { nl: true } => One::Any,
            Expr::Any { nl: false } => One::AnyNoNl,
            _ => return None,
        })
    }

    fn visit(&mut self, e: &Expr) -> Result<()> {
        match e {
            Expr::Empty => {}
            Expr::Any { .. } | Expr::Lit(_) | Expr::Set(_) => {
                let o = self.one(e).unwrap();
                self.insns.push(Insn::One(o));
            }
            Expr::Assert(a) => self.insns.push(Insn::Assert(*a)),
            Expr::Concat(v) => {
                for c in v {
                    self.visit(c)?;
                }
            }
            Expr::Alt(v) => self.alt(v.len(), |c, i| c.visit(&v[i]))?,
            Expr::Group(g, c) => {
                self.size(e);
                self.insns.push(Insn::Save(g * 2));
                self.visit(c)?;
                self.insns.push(Insn::Save(g * 2 + 1));
            }
            Expr::Repeat { child, lo, hi, greedy } => self.repeat(child, *lo, *hi, *greedy)?,
            Expr::Look { child, behind, neg } => self.look(child, *behind, *neg)?,
            Expr::Backref { group, casei } => {
                if *group == 0 {
                    bail!("invalid backreference to group 0");
                }
                self.insns.push(Insn::Backref {
                    slot: group * 2,
                    casei: *casei,
                })
            }
            Expr::BackrefExists(g) => self.insns.push(Insn::BackrefExists(*g)),
            Expr::Atomic(c) => {
                self.insns.push(Insn::BeginAtomic);
                self.visit(c)?;
                self.insns.push(Insn::EndAtomic);
            }
            Expr::KeepOut => self.insns.push(Insn::Save(0)),
            Expr::Continue => self.insns.push(Insn::Continue),
            Expr::Cond { cond, yes, no } => {
                // The condition is matched atomically; when it fails the `no` branch runs from
                // the same position, when it succeeds `yes` runs after it (with no way back to
                // `no`).
                self.insns.push(Insn::BeginAtomic);
                let split = self.pc();
                self.insns.push(Insn::Split(split + 1, usize::MAX));
                self.visit(cond)?;
                self.insns.push(Insn::EndAtomic);
                self.visit(yes)?;
                let jmp = self.pc();
                self.insns.push(Insn::Jmp(0));
                let no_pc = self.pc();
                self.insns[split] = Insn::Split(split + 1, no_pc);
                self.visit(no)?;
                let end = self.pc();
                self.insns[jmp] = Insn::Jmp(end);
            }
        }
        Ok(())
    }

    fn alt(&mut self, n: usize, mut f: impl FnMut(&mut Compiler, usize) -> Result<()>) -> Result<()> {
        let mut jmps = Vec::new();
        let mut last_split: Option<usize> = None;
        for i in 0..n {
            let pc = self.pc();
            if let Some(s) = last_split.take()
                && let Insn::Split(x, _) = self.insns[s]
            {
                self.insns[s] = Insn::Split(x, pc);
            }
            if i + 1 < n {
                self.insns.push(Insn::Split(pc + 1, usize::MAX));
                last_split = Some(pc);
            }
            f(self, i)?;
            if i + 1 < n {
                jmps.push(self.pc());
                self.insns.push(Insn::Jmp(0));
            }
        }
        let end = self.pc();
        for j in jmps {
            self.insns[j] = Insn::Jmp(end);
        }
        Ok(())
    }

    fn repeat(&mut self, child: &Expr, lo: usize, hi: usize, greedy: bool) -> Result<()> {
        if greedy && let Some(one) = self.one(child) {
            self.insns.push(Insn::Star { one, lo, hi });
            return Ok(());
        }
        if lo == 0 && hi == 1 {
            let pc = self.pc();
            self.insns.push(Insn::Split(pc + 1, pc + 1));
            self.visit(child)?;
            let next = self.pc();
            self.insns[pc] = if greedy {
                Insn::Split(pc + 1, next)
            } else {
                Insn::Split(next, pc + 1)
            };
            return Ok(());
        }
        let min = self.size(child).min;
        if hi == usize::MAX && min == 0 {
            // Unbounded repetition of something that can match empty: stop on an empty
            // iteration.
            let repeat = self.new_save();
            let check = self.new_save();
            self.insns.push(Insn::Save0(repeat));
            let pc = self.pc();
            self.insns.push(Insn::Jmp(0));
            self.visit(child)?;
            self.insns.push(Insn::Jmp(pc));
            let next = self.pc();
            self.insns[pc] = if greedy {
                Insn::RepeatEpsGr { lo, next, repeat, check }
            } else {
                Insn::RepeatEpsNg { lo, next, repeat, check }
            };
        } else if lo == 0 && hi == usize::MAX {
            let pc = self.pc();
            self.insns.push(Insn::Jmp(0));
            self.visit(child)?;
            self.insns.push(Insn::Jmp(pc));
            let next = self.pc();
            self.insns[pc] = if greedy {
                Insn::Split(pc + 1, next)
            } else {
                Insn::Split(next, pc + 1)
            };
        } else if lo == 1 && hi == usize::MAX {
            let pc = self.pc();
            self.visit(child)?;
            let next = self.pc() + 1;
            self.insns.push(if greedy { Insn::Split(pc, next) } else { Insn::Split(next, pc) });
        } else {
            let repeat = self.new_save();
            self.insns.push(Insn::Save0(repeat));
            let pc = self.pc();
            self.insns.push(Insn::Jmp(0));
            self.visit(child)?;
            self.insns.push(Insn::Jmp(pc));
            let next = self.pc();
            self.insns[pc] = if greedy {
                Insn::RepeatGr { lo, hi, next, repeat }
            } else {
                Insn::RepeatNg { lo, hi, next, repeat }
            };
        }
        Ok(())
    }

    fn look(&mut self, child: &Expr, behind: bool, neg: bool) -> Result<()> {
        if behind {
            let size = self.size(child);
            if !size.exact
                && let Expr::Alt(v) = child
            {
                // `(?<=a|bb)` -> `(?<=a)|(?<=bb)`; `(?<!a|bb)` -> `(?<!a)(?<!bb)`.
                if neg {
                    for a in v {
                        self.look(a, true, true)?;
                    }
                    return Ok(());
                }
                return self.alt(v.len(), |c, i| c.look(&v[i], true, false));
            }
            if !size.exact {
                let mut sub = Compiler {
                    insns: Vec::new(),
                    n_saves: 0,
                    sets: std::mem::take(&mut self.sets),
                    subs: std::mem::take(&mut self.subs),
                    group_sizes: std::mem::take(&mut self.group_sizes),
                };
                std::mem::swap(&mut sub.n_saves, &mut self.n_saves);
                let r = sub.visit(child);
                sub.insns.push(Insn::Match);
                self.sets = std::mem::take(&mut sub.sets);
                self.subs = std::mem::take(&mut sub.subs);
                self.group_sizes = std::mem::take(&mut sub.group_sizes);
                self.n_saves = sub.n_saves;
                r?;
                self.subs.push(Prog { insns: sub.insns });
                let prog = self.subs.len() - 1;
                self.insns.push(Insn::Behind {
                    prog,
                    neg,
                    max: Self::max_size(child),
                });
                return Ok(());
            }
            return self.lookaround(child, neg, Some(size.min));
        }
        self.lookaround(child, neg, None)
    }

    fn lookaround(&mut self, child: &Expr, neg: bool, back: Option<usize>) -> Result<()> {
        if neg {
            let pc = self.pc();
            self.insns.push(Insn::Split(pc + 1, usize::MAX));
            if let Some(n) = back {
                self.insns.push(Insn::GoBack(n));
            }
            self.visit(child)?;
            self.insns.push(Insn::FailNegativeLook);
            let next = self.pc();
            self.insns[pc] = Insn::Split(pc + 1, next);
        } else {
            let save = self.new_save();
            self.insns.push(Insn::Save(save));
            if let Some(n) = back {
                self.insns.push(Insn::GoBack(n));
            }
            self.visit(child)?;
            self.insns.push(Insn::Restore(save));
        }
        Ok(())
    }
}

// ----------------------------------------------------------------------------------------------
// The machine

struct Branch {
    pc: usize,
    ix: usize,
    nsave: usize,
}

/// Backtracking state: the save slots (copy-on-write against the branch stack) and the branch
/// stack, as in fancy-regex's VM.
struct State {
    saves: Vec<usize>,
    stack: Vec<Branch>,
    oldsave: Vec<(usize, usize)>,
    nsave: usize,
    explicit_sp: usize,
}

struct Overflow;

impl State {
    fn new(n_saves: usize) -> State {
        State {
            saves: vec![usize::MAX; n_saves],
            stack: Vec::new(),
            oldsave: Vec::new(),
            nsave: 0,
            explicit_sp: n_saves,
        }
    }

    fn reset(&mut self) {
        self.saves.truncate(self.explicit_sp);
        self.saves.fill(usize::MAX);
        self.stack.clear();
        self.oldsave.clear();
        self.nsave = 0;
    }

    fn push(&mut self, pc: usize, ix: usize) -> Result<(), Overflow> {
        if self.stack.len() >= MAX_STACK {
            return Err(Overflow);
        }
        self.stack.push(Branch { pc, ix, nsave: self.nsave });
        self.nsave = 0;
        Ok(())
    }

    fn pop(&mut self) -> (usize, usize) {
        for _ in 0..self.nsave {
            let (slot, v) = self.oldsave.pop().unwrap();
            self.saves[slot] = v;
        }
        let b = self.stack.pop().unwrap();
        self.nsave = b.nsave;
        (b.pc, b.ix)
    }

    fn save(&mut self, slot: usize, v: usize) {
        let n = self.oldsave.len();
        if self.oldsave[n - self.nsave..].iter().any(|&(s, _)| s == slot) {
            self.saves[slot] = v;
            return;
        }
        self.oldsave.push((slot, self.saves[slot]));
        self.nsave += 1;
        self.saves[slot] = v;
    }

    fn stack_push(&mut self, v: usize) {
        if self.saves.len() == self.explicit_sp {
            self.saves.push(self.explicit_sp + 1);
        }
        let sp = self.saves[self.explicit_sp];
        if self.saves.len() == sp {
            self.saves.push(v);
        } else {
            self.save(sp, v);
        }
        self.save(self.explicit_sp, sp + 1);
    }

    fn stack_pop(&mut self) -> usize {
        let sp = self.saves[self.explicit_sp] - 1;
        let v = self.saves[sp];
        self.save(self.explicit_sp, sp);
        v
    }

    /// Drops the branches pushed after the first `count`, keeping the current saves.
    fn backtrack_cut(&mut self, count: usize) {
        if self.stack.len() == count {
            return;
        }
        let (start, end) = {
            let mut end = self.oldsave.len() - self.nsave;
            for b in &self.stack[count + 1..] {
                end -= b.nsave;
            }
            (end - self.stack[count].nsave, end)
        };
        let mut seen: Vec<usize> = self.oldsave[start..end].iter().map(|&(s, _)| s).collect();
        let mut keep = end;
        for i in end..self.oldsave.len() {
            let slot = self.oldsave[i].0;
            if !seen.contains(&slot) {
                seen.push(slot);
                self.oldsave.swap(keep, i);
                keep += 1;
            }
        }
        self.stack.truncate(count);
        self.oldsave.truncate(keep);
        self.nsave = keep - start;
    }
}

/// A compiled pattern.
#[derive(Clone, Debug)]
pub struct Regex {
    main: Prog,
    subs: Vec<Prog>,
    sets: Vec<Set>,
    n_saves: usize,
    /// The pattern starts with `\A` / `^` (single-line): only position 0 can match.
    anchored: bool,
}

fn prev_char(s: &str, ix: usize) -> Option<char> {
    s[..ix].chars().next_back()
}

fn next_char(s: &str, ix: usize) -> Option<char> {
    s[ix..].chars().next()
}

impl Regex {
    pub fn new(pattern: &str) -> Result<Regex> {
        let mut p = Parser {
            re: pattern,
            flags: 0,
            groups: 0,
            named: HashMap::new(),
            numeric_backrefs: false,
            backrefs: Vec::new(),
        };
        let (ix, expr) = p.parse_re(0, 0)?;
        if ix < pattern.len() {
            return Err(p.err(ix, "unmatched closing parenthesis"));
        }
        if let Some(&g) = p.backrefs.iter().find(|&&g| g > p.groups) {
            bail!("regex {pattern:?}: backreference to undefined group {g}");
        }
        let mut c = Compiler {
            insns: Vec::new(),
            n_saves: (p.groups + 1) * 2,
            sets: Vec::new(),
            subs: Vec::new(),
            group_sizes: HashMap::new(),
        };
        c.visit(&expr).map_err(|e| anyhow!("regex {pattern:?}: {e}"))?;
        c.insns.push(Insn::Match);
        let anchored = matches!(c.insns.first(), Some(Insn::Assert(Assertion::StartText)));
        Ok(Regex {
            main: Prog { insns: c.insns },
            subs: c.subs,
            sets: c.sets,
            n_saves: c.n_saves,
            anchored,
        })
    }

    /// A pattern matching `s` literally.
    pub fn literal(s: &str) -> Regex {
        let mut insns: Vec<Insn> = s.chars().map(|c| Insn::One(One::Char(c))).collect();
        insns.push(Insn::Match);
        Regex {
            main: Prog { insns },
            subs: Vec::new(),
            sets: Vec::new(),
            n_saves: 2,
            anchored: false,
        }
    }

    #[inline]
    fn one(&self, o: One, c: char) -> bool {
        match o {
            One::Char(x) => c == x,
            One::Set(i) => self.sets[i].contains(c),
            One::Any => true,
            One::AnyNoNl => c != '\n',
        }
    }

    fn assert(s: &str, ix: usize, a: Assertion) -> bool {
        let w = word_set();
        let before = || prev_char(s, ix).is_some_and(|c| w.contains(c));
        let after = || next_char(s, ix).is_some_and(|c| w.contains(c));
        match a {
            Assertion::StartText => ix == 0,
            Assertion::EndText => ix == s.len(),
            Assertion::StartLine => ix == 0 || s.as_bytes()[ix - 1] == b'\n',
            Assertion::EndLine => ix == s.len() || s.as_bytes()[ix] == b'\n',
            Assertion::Word => before() != after(),
            Assertion::NotWord => before() == after(),
            Assertion::WordStart => !before() && after(),
            Assertion::WordEnd => before() && !after(),
            Assertion::WordStartHalf => !before(),
            Assertion::WordEndHalf => !after(),
        }
    }

    /// Runs `prog` anchored at `start`: the match `(start, end)` (with `end == end_at` when
    /// given). `from` is where the search began (`\G`).
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        prog: &Prog,
        st: &mut State,
        s: &str,
        start: usize,
        from: usize,
        skipped_empty: bool,
        end_at: Option<usize>,
        budget: &mut usize,
    ) -> Result<Option<(usize, usize)>, Overflow> {
        st.reset();
        st.saves[0] = start;
        let b = s.as_bytes();
        let mut pc = 0;
        let mut ix = start;
        loop {
            'fail: loop {
                match prog.insns[pc] {
                    Insn::Match => {
                        if end_at.is_some_and(|e| e != ix) {
                            break 'fail;
                        }
                        return Ok(Some((st.saves[0].min(ix), ix)));
                    }
                    Insn::One(o) => match next_char(s, ix) {
                        Some(c) if self.one(o, c) => ix += c.len_utf8(),
                        _ => break 'fail,
                    },
                    Insn::Star { one, lo, hi } => {
                        // Consume as many as possible, leaving a branch for each shorter count.
                        let mut n = 0;
                        let mut p = ix;
                        let first_branch = st.stack.len();
                        while n < hi {
                            match next_char(s, p) {
                                Some(c) if self.one(one, c) => {
                                    if n >= lo {
                                        st.push(pc + 1, p)?;
                                    }
                                    p += c.len_utf8();
                                    n += 1;
                                }
                                _ => break,
                            }
                        }
                        if n < lo {
                            // Drop the (empty) branches this pushed and fail.
                            debug_assert_eq!(st.stack.len(), first_branch);
                            break 'fail;
                        }
                        ix = p;
                    }
                    Insn::Assert(a) => {
                        if !Self::assert(s, ix, a) {
                            break 'fail;
                        }
                    }
                    Insn::Split(x, y) => {
                        st.push(y, ix)?;
                        pc = x;
                        continue;
                    }
                    Insn::Jmp(t) => {
                        pc = t;
                        continue;
                    }
                    Insn::Save(slot) => st.save(slot, ix),
                    Insn::Save0(slot) => st.save(slot, 0),
                    Insn::Restore(slot) => ix = st.saves[slot],
                    Insn::RepeatGr { lo, hi, next, repeat } => {
                        let n = st.saves[repeat];
                        if n == hi {
                            pc = next;
                            continue;
                        }
                        st.save(repeat, n + 1);
                        if n >= lo {
                            st.push(next, ix)?;
                        }
                    }
                    Insn::RepeatNg { lo, hi, next, repeat } => {
                        let n = st.saves[repeat];
                        if n == hi {
                            pc = next;
                            continue;
                        }
                        st.save(repeat, n + 1);
                        if n >= lo {
                            st.push(pc + 1, ix)?;
                            pc = next;
                            continue;
                        }
                    }
                    Insn::RepeatEpsGr { lo, next, repeat, check } => {
                        let n = st.saves[repeat];
                        if n > 0 && st.saves[check] == ix {
                            pc = next;
                            continue;
                        }
                        st.save(repeat, n + 1);
                        if n >= lo {
                            st.save(check, ix);
                            st.push(next, ix)?;
                        }
                    }
                    Insn::RepeatEpsNg { lo, next, repeat, check } => {
                        let n = st.saves[repeat];
                        if n > 0 && st.saves[check] == ix {
                            pc = next;
                            continue;
                        }
                        st.save(repeat, n + 1);
                        if n >= lo {
                            st.save(check, ix);
                            st.push(pc + 1, ix)?;
                            pc = next;
                            continue;
                        }
                    }
                    Insn::GoBack(n) => {
                        for _ in 0..n {
                            match prev_char(s, ix) {
                                Some(c) => ix -= c.len_utf8(),
                                None => break 'fail,
                            }
                        }
                    }
                    Insn::FailNegativeLook => {
                        // The body of a negative lookaround matched: drop its branches, then
                        // fail past the lookaround.
                        loop {
                            let (p, _) = st.pop();
                            if p == pc + 1 {
                                break;
                            }
                        }
                        break 'fail;
                    }
                    Insn::Backref { slot, casei } => {
                        let (lo, hi) = (st.saves[slot], st.saves[slot + 1]);
                        if lo == usize::MAX || hi == usize::MAX {
                            break 'fail;
                        }
                        let end = ix + (hi - lo);
                        if end > s.len() {
                            break 'fail;
                        }
                        let ok = if b[ix..end] == b[lo..hi] {
                            true
                        } else if casei && s.is_char_boundary(end) {
                            let (x, y) = (&s[lo..hi], &s[ix..end]);
                            x.chars().count() == y.chars().count() && x.chars().zip(y.chars()).all(|(a, c)| eq_fold(a, c))
                        } else {
                            false
                        };
                        if !ok {
                            break 'fail;
                        }
                        ix = end;
                    }
                    Insn::BackrefExists(g) => {
                        if st.saves[g * 2] == usize::MAX {
                            break 'fail;
                        }
                    }
                    Insn::BeginAtomic => {
                        let n = st.stack.len();
                        st.stack_push(n);
                    }
                    Insn::EndAtomic => {
                        let n = st.stack_pop();
                        st.backtrack_cut(n);
                    }
                    Insn::Behind { prog: sub, neg, max } => {
                        let mut found = false;
                        let mut sub_state = State::new(self.n_saves);
                        let mut p = ix;
                        let mut steps = 0;
                        loop {
                            if self
                                .run(&self.subs[sub], &mut sub_state, s, p, from, skipped_empty, Some(ix), budget)?
                                .is_some()
                            {
                                found = true;
                                break;
                            }
                            if max.is_some_and(|m| steps >= m) {
                                break;
                            }
                            match prev_char(s, p) {
                                Some(c) => p -= c.len_utf8(),
                                None => break,
                            }
                            steps += 1;
                        }
                        if found == neg {
                            break 'fail;
                        }
                    }
                    Insn::Continue => {
                        if ix > from || skipped_empty {
                            break 'fail;
                        }
                    }
                }
                pc += 1;
            }
            if st.stack.is_empty() {
                return Ok(None);
            }
            if *budget == 0 {
                return Err(Overflow);
            }
            *budget -= 1;
            let (p, i) = st.pop();
            pc = p;
            ix = i;
        }
    }

    /// The leftmost match at or after byte `from`, as a byte range (None also when the
    /// backtrack limit is hit).
    pub fn find_at(&self, text: &str, from: usize) -> Option<(usize, usize)> {
        self.search(text, from, false).ok().flatten()
    }

    fn search(&self, text: &str, from: usize, skipped_empty: bool) -> Result<Option<(usize, usize)>, Overflow> {
        let mut st = State::new(self.n_saves);
        let mut budget = BACKTRACK_LIMIT;
        let mut start = from;
        loop {
            if self.anchored && start > 0 {
                return Ok(None);
            }
            if let Some(m) = self.run(&self.main, &mut st, text, start, from, skipped_empty, None, &mut budget)? {
                return Ok(Some(m));
            }
            match next_char(text, start) {
                Some(c) => start += c.len_utf8(),
                None => return Ok(None),
            }
        }
    }

    /// Non-overlapping matches, left to right (an empty match right after a previous match is
    /// skipped). Like the reference, the iteration stops when the backtrack limit is hit.
    pub fn find_iter(&self, text: &str) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let mut pos = 0;
        let mut last_match: Option<usize> = None;
        while pos <= text.len() {
            let skipped = last_match.is_some_and(|m| pos > m);
            let Ok(Some((s, e))) = self.search(text, pos, skipped) else { break };
            if s == e {
                pos = match text[e..].chars().next() {
                    Some(c) => e + c.len_utf8(),
                    None => text.len() + 1,
                };
                if last_match == Some(e) {
                    continue;
                }
            } else {
                pos = e;
            }
            last_match = Some(e);
            out.push((s, e));
        }
        out
    }

    /// Whether the pattern matches somewhere in `text`.
    pub fn is_match(&self, text: &str) -> bool {
        self.find_at(text, 0).is_some()
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
        assert_eq!(all(r"(?m)^\w+$", "ab\ncd"), ["ab", "cd"]);
        assert_eq!(all(r"(?s)a.b", "a\nb"), ["a\nb"]);
        assert_eq!(all(r"(?x) a b  # comment", "ab"), ["ab"]);
        assert_eq!(all(r"[[:digit:]&&[^5]]+", "12567"), ["12", "67"]);
        assert_eq!(all(r"\p{Han}+|\p{Hiragana}+", "漢字ひらがな"), ["漢字", "ひらがな"]);
        assert_eq!(all(r"(?i)straße", "STRASSE Straße"), ["Straße"]);
    }

    #[test]
    fn lookbehind_atomic_possessive_backrefs() {
        assert_eq!(all(r"(?<=\$)\d+", "a $12 b 34"), ["12"]);
        assert_eq!(all(r"(?<!\d)\d{2}", "123 45"), ["12", "45"]);
        assert_eq!(all(r"(?<=ab|c)x", "abx cx bx"), ["x", "x"]);
        assert_eq!(all(r"(?<=a\d*)x", "a123x bx"), ["x"]);
        assert_eq!(all(r"(?>a+)b", "aaab"), ["aaab"]);
        assert_eq!(all(r"a++a", "aaaa"), Vec::<String>::new());
        // `+` after an interval repeats it (Oniguruma's Ruby syntax): digits in groups of three
        // from the right, as ERNIE 5.5's tokenizer splits them.
        assert_eq!(all(r"\A\d{1,2}(?=\d{3}+\z)", "1234567"), ["1"]);
        assert_eq!(all(r"\A\d{1,2}(?=\d{3}+\z)", "123456"), Vec::<String>::new());
        assert_eq!(all(r"a{2}+", "aaaaa"), ["aaaa"]);
        assert_eq!(all(r"(\w)\1", "abccdd"), ["cc", "dd"]);
        assert_eq!(all(r"(?<q>['x])\w+\k<q>", "'ab' xcdx"), ["'ab'", "xcdx"]);
        assert_eq!(all(r"(a)?(?(1)b|c)", "ab c"), ["ab", "c"]);
        assert_eq!(all(r"foo\Kbar", "foobar"), ["bar"]);
        assert_eq!(all(r"\Gab", "ababxab"), ["ab", "ab"]);
        assert_eq!(all(r"\bab\b", "ab cab ab"), ["ab", "ab"]);
        assert_eq!(all(r"a\Z", "a\n\n"), ["a"]);
    }

    #[test]
    fn long_inputs_do_not_recurse() {
        let t = "1".repeat(200_000);
        let p = r"\p{N}{1,3}(?=(?:\p{N}{3})*(?!\p{N}))|\p{N}{1,2}";
        assert_eq!(Regex::new(p).unwrap().find_iter(&t).len(), 200_000 / 3 + 1);
    }

    #[test]
    fn errors() {
        assert!(Regex::new(r"a)").is_err());
        assert!(Regex::new(r"(a").is_err());
        assert!(Regex::new(r"*a").is_err());
        assert!(Regex::new(r"\2(a)").is_err());
        assert!(Regex::new(r"\g<1>(a)").is_err());
        assert!(Regex::new(r"\q").is_err());
    }
}
