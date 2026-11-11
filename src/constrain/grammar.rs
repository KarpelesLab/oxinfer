//! Context-free grammars over bytes (and single tokens) and their incremental matcher.
//!
//! A [`Grammar`] is a list of rules; a rule is a choice between alternatives; an alternative is a
//! sequence of symbols: a byte out of a set, one token id (special tokens have no text), or another
//! rule. The [`Matcher`] follows every parse at once: a state is a set of stacks, each the
//! chain of positions to return to, with a terminal on top. Stacks are interned in an arena
//! (`(position, parent)` pairs), so equal stacks have equal ids, pushing and popping copy
//! nothing, and loops such as the characters of a string come back to the same ids.

use anyhow::{Result, bail};
use std::collections::HashMap;

pub type RuleId = u32;

/// One symbol of an alternative.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sym {
    /// One byte out of class `n` of the grammar.
    Bytes(u32),
    /// Exactly this token (a special token).
    Token(u32),
    Rule(RuleId),
}

/// A set of bytes.
pub type Class = [u64; 4];

fn class_has(c: &Class, b: u8) -> bool {
    (c[b as usize >> 6] >> (b & 63)) & 1 == 1
}

/// A symbol position: before symbol `idx` of alternative `alt` of `rule`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Pos {
    pub rule: u32,
    pub alt: u16,
    pub idx: u16,
}

#[derive(Clone, Debug, Default)]
pub struct Grammar {
    rules: Vec<Vec<Vec<Sym>>>,
    classes: Vec<Class>,
    class_ids: HashMap<Class, u32>,
}

impl Grammar {
    pub fn new() -> Self {
        Self::default()
    }

    /// A rule without alternatives yet (for recursion); fill it with [`Grammar::add_alt`].
    pub fn new_rule(&mut self) -> RuleId {
        self.rules.push(Vec::new());
        self.rules.len() as RuleId - 1
    }

    pub fn add_alt(&mut self, rule: RuleId, alt: Vec<Sym>) {
        self.rules[rule as usize].push(alt);
    }

    pub fn rule(&mut self, alts: Vec<Vec<Sym>>) -> RuleId {
        self.rules.push(alts);
        self.rules.len() as RuleId - 1
    }

    pub fn alts(&self, rule: RuleId) -> &[Vec<Sym>] {
        &self.rules[rule as usize]
    }

    /// A byte in one of the inclusive `ranges`.
    pub fn class(&mut self, ranges: &[(u8, u8)]) -> Sym {
        let mut c: Class = [0; 4];
        for &(a, b) in ranges {
            for x in a..=b {
                c[x as usize >> 6] |= 1 << (x & 63);
            }
        }
        let n = self.classes.len() as u32;
        let id = *self.class_ids.entry(c).or_insert(n);
        if id == n {
            self.classes.push(c);
        }
        Sym::Bytes(id)
    }

    pub fn byte(&mut self, b: u8) -> Sym {
        self.class(&[(b, b)])
    }

    /// The bytes of `s` in sequence.
    pub fn lit(&mut self, s: &[u8]) -> Vec<Sym> {
        s.iter().map(|&b| self.byte(b)).collect()
    }

    /// `seq` or nothing.
    pub fn opt(&mut self, seq: Vec<Sym>) -> Sym {
        Sym::Rule(self.rule(vec![seq, Vec::new()]))
    }

    /// `seq` between `min` and `max` times (no upper bound when `max` is None). `seq` must
    /// consume input.
    pub fn repeat(&mut self, seq: Vec<Sym>, min: usize, max: Option<usize>) -> Vec<Sym> {
        let item = if seq.len() == 1 { seq[0] } else { Sym::Rule(self.rule(vec![seq])) };
        let mut out = vec![item; min];
        match max {
            None => {
                let r = self.new_rule();
                self.add_alt(r, vec![item, Sym::Rule(r)]);
                self.add_alt(r, Vec::new());
                out.push(Sym::Rule(r));
            }
            Some(max) => {
                // rest(k): up to k more.
                let mut rest: Option<RuleId> = None;
                for _ in min..max {
                    let mut more = vec![item];
                    more.extend(rest.map(Sym::Rule));
                    rest = Some(self.rule(vec![more, Vec::new()]));
                }
                out.extend(rest.map(Sym::Rule));
            }
        }
        out
    }

    fn sym(&self, p: Pos) -> Option<Sym> {
        self.rules[p.rule as usize][p.alt as usize].get(p.idx as usize).copied()
    }

    /// Checks what the matcher relies on: every rule has an alternative, positions fit their
    /// types, and no rule reaches itself without consuming input (a `$ref` cycle, for example),
    /// which would never stop expanding.
    pub fn validate(&self) -> Result<()> {
        let n = self.rules.len();
        for r in &self.rules {
            if r.is_empty() {
                bail!("the grammar has a rule that matches nothing");
            }
            if r.len() > u16::MAX as usize || r.iter().any(|a| a.len() >= u16::MAX as usize) {
                bail!("the grammar has a rule with too many alternatives or symbols");
            }
        }
        let mut nullable = vec![false; n];
        loop {
            let mut changed = false;
            for (i, r) in self.rules.iter().enumerate() {
                if !nullable[i]
                    && r.iter()
                        .any(|a| a.iter().all(|s| matches!(s, Sym::Rule(q) if nullable[*q as usize])))
                {
                    nullable[i] = true;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        // Rules reachable at the left edge of each rule; a cycle is left recursion.
        let left: Vec<Vec<u32>> = self
            .rules
            .iter()
            .map(|r| {
                let mut l = Vec::new();
                for a in r {
                    for s in a {
                        match s {
                            Sym::Rule(q) => {
                                l.push(*q);
                                if !nullable[*q as usize] {
                                    break;
                                }
                            }
                            _ => break,
                        }
                    }
                }
                l
            })
            .collect();
        // 0 = new, 1 = on the path, 2 = done.
        let mut color = vec![0u8; n];
        for start in 0..n {
            if color[start] != 0 {
                continue;
            }
            let mut path: Vec<(usize, usize)> = vec![(start, 0)];
            color[start] = 1;
            while let Some(&(r, i)) = path.last() {
                let Some(&q) = left[r].get(i) else {
                    color[r] = 2;
                    path.pop();
                    continue;
                };
                path.last_mut().unwrap().1 += 1;
                match color[q as usize] {
                    0 => {
                        color[q as usize] = 1;
                        path.push((q as usize, 0));
                    }
                    1 => bail!("the grammar is left-recursive (a reference cycle that consumes no input)"),
                    _ => {}
                }
            }
        }
        Ok(())
    }
}

/// Stack id of the finished parse: the start rule is complete.
pub const DONE: u32 = 0;
/// Stack id of "outside": the bottom of a stack whose parent is unknown (used to find what a
/// position accepts whatever is under it).
pub const OUT: u32 = 1;

/// The arena of interned stacks of a [`Grammar`].
#[derive(Clone, Debug)]
pub struct Matcher {
    nodes: Vec<(Pos, u32)>,
    index: HashMap<(Pos, u32), u32>,
}

impl Default for Matcher {
    fn default() -> Self {
        Self::new()
    }
}

impl Matcher {
    pub fn new() -> Self {
        let none = Pos {
            rule: u32::MAX,
            alt: 0,
            idx: 0,
        };
        Self {
            nodes: vec![(none, 0), (none, 1)],
            index: HashMap::new(),
        }
    }

    fn intern(&mut self, pos: Pos, parent: u32) -> u32 {
        let n = self.nodes.len() as u32;
        let id = *self.index.entry((pos, parent)).or_insert(n);
        if id == n {
            self.nodes.push((pos, parent));
        }
        id
    }

    /// The stacks at the start of `rule` over the stack `parent` ([`DONE`] for a whole parse).
    pub fn start(&mut self, g: &Grammar, rule: RuleId, parent: u32) -> Vec<u32> {
        let mut out = Vec::new();
        for a in 0..g.rules[rule as usize].len() {
            let p = Pos {
                rule,
                alt: a as u16,
                idx: 0,
            };
            self.enter(g, parent, p, &mut out);
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// The single stack at `pos` over `parent` (`pos` must be a terminal).
    pub fn at(&mut self, pos: Pos, parent: u32) -> u32 {
        self.intern(pos, parent)
    }

    /// Continues at `pos` over `parent`, down to the terminals.
    fn enter(&mut self, g: &Grammar, parent: u32, pos: Pos, out: &mut Vec<u32>) {
        let alt = &g.rules[pos.rule as usize][pos.alt as usize];
        match alt.get(pos.idx as usize) {
            None => self.resume(g, parent, out),
            Some(Sym::Rule(r)) => {
                // A rule in last position needs no frame to return to.
                let ret = if pos.idx as usize + 1 == alt.len() {
                    parent
                } else {
                    self.intern(pos, parent)
                };
                for a in 0..g.rules[*r as usize].len() {
                    let p = Pos {
                        rule: *r,
                        alt: a as u16,
                        idx: 0,
                    };
                    self.enter(g, ret, p, out);
                }
            }
            Some(_) => {
                let id = self.intern(pos, parent);
                out.push(id);
            }
        }
    }

    /// The rule called from the frame `node` is complete.
    fn resume(&mut self, g: &Grammar, node: u32, out: &mut Vec<u32>) {
        if node <= OUT {
            out.push(node);
            return;
        }
        let (pos, parent) = self.nodes[node as usize];
        self.enter(g, parent, Pos { idx: pos.idx + 1, ..pos }, out);
    }

    /// The position and terminal on top of a stack (None for [`DONE`] and [`OUT`]).
    pub fn top(&self, g: &Grammar, stack: u32) -> Option<(Pos, Sym)> {
        if stack <= OUT {
            return None;
        }
        let pos = self.nodes[stack as usize].0;
        g.sym(pos).map(|s| (pos, s))
    }

    /// The stacks of `state` after byte `b`, appended to `out` (sorted, without duplicates).
    /// Returns whether `state` held [`OUT`], which cannot be followed.
    pub fn step_byte(&mut self, g: &Grammar, state: &[u32], b: u8, out: &mut Vec<u32>) -> bool {
        let mut escaped = false;
        for &s in state {
            if s == OUT {
                escaped = true;
            }
            if s <= OUT {
                continue;
            }
            let (pos, parent) = self.nodes[s as usize];
            if let Some(Sym::Bytes(c)) = g.sym(pos)
                && class_has(&g.classes[c as usize], b)
            {
                self.enter(g, parent, Pos { idx: pos.idx + 1, ..pos }, out);
            }
        }
        out.sort_unstable();
        out.dedup();
        escaped
    }

    /// The stacks of `state` after the special token `token`.
    pub fn step_token(&mut self, g: &Grammar, state: &[u32], token: u32, out: &mut Vec<u32>) {
        for &s in state {
            if s <= OUT {
                continue;
            }
            let (pos, parent) = self.nodes[s as usize];
            if g.sym(pos) == Some(Sym::Token(token)) {
                self.enter(g, parent, Pos { idx: pos.idx + 1, ..pos }, out);
            }
        }
        out.sort_unstable();
        out.dedup();
    }

    /// Whether `text` is a whole match of `rule`, and whether it is the start of one.
    pub fn matches(g: &Grammar, rule: RuleId, text: &[u8]) -> (bool, bool) {
        let mut m = Matcher::new();
        let mut state = m.start(g, rule, DONE);
        for &b in text {
            let mut next = Vec::new();
            m.step_byte(g, &state, b, &mut next);
            state = next;
            if state.is_empty() {
                return (false, false);
            }
        }
        (state.contains(&DONE), true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequences_choices_and_repeats() {
        let mut g = Grammar::new();
        // root = "a" ("b" | "cd"){1,2} "e"*
        let b = g.lit(b"b");
        let cd = g.lit(b"cd");
        let choice = g.rule(vec![b, cd]);
        let mut root = g.lit(b"a");
        root.extend(g.repeat(vec![Sym::Rule(choice)], 1, Some(2)));
        let e = g.byte(b'e');
        root.extend(g.repeat(vec![e], 0, None));
        let root = g.rule(vec![root]);
        g.validate().unwrap();
        for ok in ["ab", "acd", "abcd", "acdbeee", "abe"] {
            assert_eq!(Matcher::matches(&g, root, ok.as_bytes()), (true, true), "{ok}");
        }
        for prefix in ["", "a", "ac", "abc"] {
            assert_eq!(Matcher::matches(&g, root, prefix.as_bytes()), (false, true), "{prefix}");
        }
        for bad in ["b", "ae", "abbb", "abcdcd", "abeb"] {
            assert_eq!(Matcher::matches(&g, root, bad.as_bytes()), (false, false), "{bad}");
        }
    }

    #[test]
    fn recursion_keeps_the_stack_and_loops_reuse_it() {
        let mut g = Grammar::new();
        // nested = "(" nested ")" | "x"+
        let nested = g.new_rule();
        let (open, close, x) = (g.byte(b'('), g.byte(b')'), g.byte(b'x'));
        g.add_alt(nested, vec![open, Sym::Rule(nested), close]);
        let xs = g.repeat(vec![x], 1, None);
        g.add_alt(nested, xs);
        g.validate().unwrap();
        assert_eq!(Matcher::matches(&g, nested, b"((xxx))"), (true, true));
        assert_eq!(Matcher::matches(&g, nested, b"((xxx)"), (false, true));
        assert_eq!(Matcher::matches(&g, nested, b"((xxx)))"), (false, false));
        // A long run of x comes back to the same stacks: the arena does not grow.
        let mut m = Matcher::new();
        let mut state = m.start(&g, nested, DONE);
        let mut sizes = Vec::new();
        for _ in 0..50 {
            let mut next = Vec::new();
            m.step_byte(&g, &state, b'x', &mut next);
            state = next;
            sizes.push(m.nodes.len());
        }
        assert_eq!(sizes[2], sizes[49]);
        assert!(state.contains(&DONE));
    }

    #[test]
    fn left_recursion_is_refused() {
        let mut g = Grammar::new();
        let a = g.new_rule();
        let x = g.byte(b'x');
        g.add_alt(a, vec![Sym::Rule(a), x]);
        g.add_alt(a, vec![x]);
        assert!(g.validate().is_err());
        // Through a rule that can be empty.
        let mut g = Grammar::new();
        let x = g.byte(b'x');
        let maybe = g.opt(vec![x]);
        let a = g.new_rule();
        g.add_alt(a, vec![maybe, Sym::Rule(a)]);
        g.add_alt(a, vec![x]);
        assert!(g.validate().is_err());
        let mut g = Grammar::new();
        g.new_rule();
        assert!(g.validate().is_err());
    }

    #[test]
    fn tokens_are_not_bytes() {
        let mut g = Grammar::new();
        let mut seq = vec![Sym::Token(7)];
        seq.extend(g.lit(b"ok"));
        let root = g.rule(vec![seq]);
        let mut m = Matcher::new();
        let state = m.start(&g, root, DONE);
        let mut next = Vec::new();
        m.step_byte(&g, &state, b'o', &mut next);
        assert!(next.is_empty());
        m.step_token(&g, &state, 8, &mut next);
        assert!(next.is_empty());
        m.step_token(&g, &state, 7, &mut next);
        assert_eq!(next.len(), 1);
    }
}
