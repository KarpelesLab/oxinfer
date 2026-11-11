//! Prompt-lookup speculative decoding: drafts copied from the context itself, with no draft
//! model. When the last tokens of the conversation (prompt and reply so far) occurred before, the
//! tokens that followed that earlier occurrence are proposed as the continuation, and the target
//! verifies them in one multi-token pass ([`crate::dspark::Target::forward_verify`]). Replies that
//! quote, edit or summarize their prompt (code edits, refactors, extraction, rewrites, repeated
//! structure) accept long runs of such drafts; free prose rarely finds a match, and the round is
//! then a plain one-token step, so the drafter costs next to nothing there.
//!
//! - **Matching** ([`Lookup::draft`]): an index from each bigram of the context to its positions;
//!   the candidates of the last two tokens (the most recent ones first) are extended backwards to
//!   their full match length, and the longest match wins (the most recent among equals). The
//!   draft is what followed it, up to [`LookupOptions::max_draft`] tokens. Other occurrences
//!   that agree on the next token add confidence ("consensus").
//! - **Verification-outcome memory** ([`LookupOptions::memory`]): a small direct-mapped table,
//!   keyed by a hash of the four tokens before a position, remembers the draft the target last
//!   rejected there and the token it chose instead. A draft token the target refuted in the same
//!   context, and has not accepted there since, ends the draft (a veto), so the drafter does not
//!   repeat a continuation it was just told is wrong.
//! - **Cost-aware length** ([`LookupOptions::cost_aware`]): each draft token's chance of being
//!   accepted (when the ones before it were) is estimated per match class (match length,
//!   consensus, depth in the draft), calibrated online from verification outcomes; together with
//!   the measured time of verifying each number of drafts, the round verifies the length with the
//!   most expected tokens per second, which is 0 (a plain step) when nothing is worth it.
//!
//! Verification is exact: greedy decoding gives the plain greedy tokens (bit for bit with
//! [`LookupOptions::fast_verify`] off; with it, the GPU's batched kernels may round differently,
//! as for DSpark and MTP), and sampling uses standard speculative sampling with a draft
//! distribution of one token ([`crate::qwen_mtp::speculative_accept`]), which keeps the model's
//! distribution.
//!
//! The drafter runs on its own over any [`crate::dspark::Target`] ([`LookupRounds`]), or inside
//! the DSpark and MTP decoders, where a long context match supplies a round's drafts in place of
//! the draft model's when it promises more accepted tokens.

use crate::dspark::Target;
use crate::qwen_mtp::{MtpStats, speculative_accept};
use crate::sampler::Sampler;
use std::collections::HashMap;

/// Prompt-lookup settings.
#[derive(Clone, Debug)]
pub struct LookupOptions {
    /// Most draft tokens per round (at most [`crate::qwen_mtp::MAX_DRAFT`]).
    pub max_draft: usize,
    /// Shortest match (in tokens, at least 2) that may give a draft.
    pub min_match: usize,
    /// Veto draft tokens the target refuted in the same context ([`Memory`]).
    pub memory: bool,
    /// Verify the length with the most expected tokens per second (0: a plain step) from the
    /// measured verification costs and the calibrated acceptance of each match class; off,
    /// every match is verified to its full length.
    pub cost_aware: bool,
    /// Verify with the GPU's fast batched kernels (see
    /// [`crate::dspark::DsparkOptions::fast_verify`]).
    pub fast_verify: bool,
    /// Tokens that end generation: nothing after one is committed within a round.
    pub stop: Vec<u32>,
}

impl Default for LookupOptions {
    fn default() -> Self {
        Self {
            max_draft: DEFAULT_MAX_DRAFT,
            min_match: 2,
            memory: true,
            cost_aware: true,
            fast_verify: true,
            stop: Vec::new(),
        }
    }
}

/// Default most draft tokens per round.
pub const DEFAULT_MAX_DRAFT: usize = 10;

/// Whether the engine uses prompt lookup unless told otherwise (`lookup` option).
pub const BY_DEFAULT: bool = true;

/// Shortest context match whose draft replaces a draft model's round ([`Lookup::draft_over`]).
pub const LONG_MATCH: usize = 8;

/// Most recent occurrences of a bigram examined per lookup.
const MAX_CANDIDATES: usize = 48;
/// Match lengths are counted up to this.
const MAX_MATCH: usize = 64;
/// Context tokens hashed for the verification-outcome memory.
const MEMORY_CONTEXT: usize = 4;
/// Entries of the memory (a power of two).
const MEMORY_SLOTS: usize = 4096;

/// A proposed draft.
#[derive(Clone, Debug, Default)]
pub struct LookupDraft {
    pub tokens: Vec<u32>,
    /// Each token's estimated chance of acceptance, given the ones before it were accepted.
    pub chances: Vec<f64>,
    /// Length of the context match the draft continues (0: no draft).
    pub matched: usize,
    /// The draft was cut by the memory's veto.
    pub vetoed: bool,
    /// Calibration class of each token.
    class: Vec<usize>,
}

impl LookupDraft {
    /// Expected tokens a round verifying the whole draft commits (the bonus token included).
    pub fn expected(&self) -> f64 {
        expected(&self.chances, self.chances.len())
    }
}

fn expected(chances: &[f64], k: usize) -> f64 {
    let (mut e, mut p) = (1.0, 1.0);
    for &c in &chances[..k] {
        p *= c;
        e += p;
    }
    e
}

/// Drafter statistics.
#[derive(Clone, Debug, Default)]
pub struct LookupStats {
    /// Rounds that looked up a draft, and those that found a match.
    pub lookups: usize,
    pub matches: usize,
    /// Rounds that verified lookup drafts, and the drafts verified and accepted.
    pub rounds: usize,
    pub drafted: usize,
    pub accepted: usize,
    /// Drafts the memory cut short.
    pub vetoed: usize,
}

/// The verification-outcome memory: per context (a hash of the last [`MEMORY_CONTEXT`] tokens,
/// direct-mapped, tagged), the draft token the target last rejected there and the token it
/// produced instead.
#[derive(Clone)]
pub struct Memory {
    slots: Vec<Slot>,
}

#[derive(Clone, Copy, Default)]
struct Slot {
    /// Upper half of the context hash (0: empty).
    tag: u32,
    /// The refuted draft token (`u32::MAX`: none).
    refuted: u32,
    /// The token the target produced the last time.
    best: u32,
}

impl Default for Memory {
    fn default() -> Self {
        Self {
            slots: vec![Slot::default(); MEMORY_SLOTS],
        }
    }
}

impl Memory {
    fn hash(ctx: &[u32]) -> u64 {
        // FNV-1a over whole 32-bit tokens, then a SplitMix64 finalizer to spread the bits.
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &t in ctx {
            h ^= u64::from(t);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h ^= h >> 30;
        h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        h ^= h >> 27;
        h = h.wrapping_mul(0x94d0_49bb_1331_11eb);
        h ^ (h >> 31)
    }

    fn slot(&mut self, ctx: &[u32]) -> (&mut Slot, u32) {
        let h = Self::hash(ctx);
        let tag = ((h >> 32) as u32).max(1);
        (&mut self.slots[h as usize & (MEMORY_SLOTS - 1)], tag)
    }

    /// Whether the target refuted `token` after `ctx` and has not accepted it there since.
    pub fn refuted(&self, ctx: &[u32], token: u32) -> bool {
        let h = Self::hash(ctx);
        let s = &self.slots[h as usize & (MEMORY_SLOTS - 1)];
        s.tag == ((h >> 32) as u32).max(1) && s.refuted == token && s.best != token
    }

    /// The target produced `actual` after `ctx`, where the draft proposed `drafted`.
    pub fn observe(&mut self, ctx: &[u32], drafted: u32, actual: u32) {
        let (s, tag) = self.slot(ctx);
        if s.tag != tag {
            *s = Slot {
                tag,
                refuted: u32::MAX,
                best: actual,
            };
        }
        s.best = actual;
        if drafted != actual {
            s.refuted = drafted;
        } else if s.refuted == actual {
            s.refuted = u32::MAX;
        }
    }

    pub fn clear(&mut self) {
        self.slots.fill(Slot::default());
    }
}

/// Match classes of the calibration: match length buckets x consensus x depth in the draft.
const LEN_BUCKETS: usize = 6;
const DEPTHS: usize = 4;
const CLASSES: usize = LEN_BUCKETS * 2 * DEPTHS;

fn len_bucket(m: usize) -> usize {
    match m {
        0..=2 => 0,
        3 => 1,
        4 => 2,
        5..=7 => 3,
        8..=15 => 4,
        _ => 5,
    }
}

/// Prior acceptance chance of a class before any outcome is measured.
fn prior(class: usize) -> f64 {
    let (b, rest) = (class / (2 * DEPTHS), class % (2 * DEPTHS));
    let (cons, depth) = (rest / DEPTHS, rest % DEPTHS);
    let base = [0.3, 0.45, 0.55, 0.65, 0.75, 0.85][b];
    (base + 0.05 * cons as f64 + if depth > 0 { 0.1 } else { 0.0 }).min(0.95)
}

/// The cost-aware length policy: verification costs by number of drafts and acceptance by class.
#[derive(Clone, Debug)]
pub struct Policy {
    /// Moving averages of a round's verification time with `k` drafts (0: a plain step), with the
    /// number of samples.
    pub verify: Vec<(f64, usize)>,
    /// Decayed (accepted, verified) counts per class.
    pub classes: Vec<(f64, f64)>,
    /// Rounds in a row with a match but no draft verified, and the current number before a probe.
    pub idle: usize,
    pub probe: usize,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            verify: Vec::new(),
            classes: vec![(0.0, 0.0); CLASSES],
            idle: 0,
            probe: Self::PROBE,
        }
    }
}

impl Policy {
    const DECAY: f64 = 0.8;
    const PRIOR: f64 = 2.0;
    const CLASS_DECAY: f64 = 0.995;
    const PROBE: usize = 8;
    const PROBE_MAX: usize = 128;

    fn chance(&self, class: usize) -> f64 {
        let (a, n) = self.classes[class];
        (a + Self::PRIOR * prior(class)) / (n + Self::PRIOR)
    }

    /// Records a round's verification time with `k` drafts (a moving average whose second
    /// sample replaces the first, which may include compiling a GPU plan).
    pub fn record_time(&mut self, k: usize, secs: f64) {
        if self.verify.len() <= k {
            self.verify.resize(k + 1, (0.0, 0));
        }
        let e = &mut self.verify[k];
        *e = (
            if e.1 < 2 {
                secs
            } else {
                Self::DECAY * e.0 + (1.0 - Self::DECAY) * secs
            },
            e.1 + 1,
        );
    }

    fn record_outcome(&mut self, class: &[usize], accepted: usize) {
        for (j, &c) in class.iter().enumerate().take(accepted + 1) {
            for e in &mut self.classes {
                e.0 *= Self::CLASS_DECAY;
                e.1 *= Self::CLASS_DECAY;
            }
            self.classes[c].0 += f64::from(u8::from(j < accepted));
            self.classes[c].1 += 1.0;
        }
    }

    /// Verification time of `k` drafts: measured, else interpolated between the nearest measured
    /// lengths; beyond the longest one measured, its time grown by 5% per draft (an optimistic
    /// guess, so that longer lengths get tried). Before any verification of drafts was measured,
    /// twice a plain step (the first multi-token pass may also prepare its GPU plan, so it is only
    /// worth it for a good match); before a plain step was measured, unknown.
    fn verify_time(&self, k: usize) -> Option<f64> {
        if k > 0 && self.verify.iter().skip(1).all(|e| e.1 == 0) {
            let plain = self.verify.first().filter(|e| e.1 > 0)?;
            return Some(plain.0 * (2.0 + 0.05 * k as f64));
        }
        let m: Vec<(usize, f64)> = self
            .verify
            .iter()
            .enumerate()
            .filter(|e| e.1.1 > 0)
            .map(|(j, e)| (j, e.0))
            .collect();
        let lo = m.iter().rev().find(|e| e.0 <= k);
        let hi = m.iter().find(|e| e.0 >= k);
        match (lo, hi) {
            (Some(a), Some(b)) if a.0 == b.0 => Some(a.1),
            (Some(a), Some(b)) => Some(a.1 + (b.1 - a.1) * (k - a.0) as f64 / (b.0 - a.0) as f64),
            (Some(a), None) => Some(a.1 * (1.0 + 0.05 * (k - a.0) as f64)),
            (None, Some(a)) => Some(a.1),
            (None, None) => None,
        }
    }

    /// How many of the draft's tokens to verify: the length with the most expected tokens per
    /// second (0 = a plain step, also until a plain step was measured).
    pub fn cut(&self, chances: &[f64]) -> usize {
        let mut best = (0, 0.0);
        for k in 0..=chances.len() {
            let Some(v) = self.verify_time(k) else { return 0 };
            let rate = expected(chances, k) / v.max(1e-9);
            if rate > best.1 {
                best = (k, rate);
            }
        }
        best.0
    }
}

/// The drafter: the context tokens with their bigram index, the memory and the length policy.
#[derive(Clone)]
pub struct Lookup {
    toks: Vec<u32>,
    /// Bigram `(a << 32) | b` to the positions of `b` (increasing).
    index: HashMap<u64, Vec<u32>>,
    memory: Memory,
    pub policy: Policy,
    pub opts: LookupOptions,
    pub stats: LookupStats,
}

fn bigram(a: u32, b: u32) -> u64 {
    (u64::from(a) << 32) | u64::from(b)
}

impl Lookup {
    pub fn new(opts: LookupOptions) -> Self {
        let opts = LookupOptions {
            max_draft: opts.max_draft.min(crate::qwen_mtp::MAX_DRAFT),
            min_match: opts.min_match.max(2),
            ..opts
        };
        Self {
            toks: Vec::new(),
            index: HashMap::new(),
            memory: Memory::default(),
            policy: Policy::default(),
            opts,
            stats: LookupStats::default(),
        }
    }

    /// Most drafts a round verifies. A verification on the GPU's decode plans takes up to
    /// [`LookupOptions::max_draft`]; on the CPU, more tokens than the batch-invariant decode
    /// kernels take ([`crate::kernels::decode_max_tokens`], 7 by default) would go through the
    /// prefill GEMM, whose rounding differs from one-token decoding, so with fast verification
    /// (where that is acceptable) the drafts are capped to stay exact on the CPU without changing
    /// the process-wide limit; exact verification raises the limit instead ([`Lookup::prepare`]).
    fn round_cap(&self) -> usize {
        let gpu = crate::gpu::active().is_some() && crate::gpu::decode_enabled();
        if gpu || !self.opts.fast_verify {
            self.opts.max_draft
        } else {
            self.opts.max_draft.min(crate::kernels::decode_max_tokens().saturating_sub(1))
        }
    }

    /// With exact verification, keeps verifications of up to [`LookupOptions::max_draft`] drafts
    /// on the batch-invariant decode kernels (raises the process-wide
    /// [`crate::kernels::set_decode_max_tokens`], as the DSpark and MTP decoders do).
    pub fn prepare(&self) {
        if !self.opts.fast_verify {
            exact_up_to(self.opts.max_draft + 1);
        }
    }

    /// The context tokens.
    pub fn tokens(&self) -> &[u32] {
        &self.toks
    }

    pub fn len(&self) -> usize {
        self.toks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.toks.is_empty()
    }

    /// Appends tokens to the context.
    pub fn extend(&mut self, tokens: &[u32]) {
        for &t in tokens {
            if let Some(&p) = self.toks.last() {
                self.index.entry(bigram(p, t)).or_default().push(self.toks.len() as u32);
            }
            self.toks.push(t);
        }
    }

    /// Drops the context from position `len` on.
    pub fn truncate(&mut self, len: usize) {
        while self.toks.len() > len {
            let i = self.toks.len() - 1;
            if i > 0 {
                let k = bigram(self.toks[i - 1], self.toks[i]);
                if let Some(v) = self.index.get_mut(&k) {
                    debug_assert_eq!(v.last().copied(), Some(i as u32));
                    v.pop();
                    if v.is_empty() {
                        self.index.remove(&k);
                    }
                }
            }
            self.toks.pop();
        }
    }

    /// A new conversation: the context and the memory are cleared (the calibration and the
    /// measured verification costs are kept: they describe the model and the machine).
    pub fn reset(&mut self) {
        self.toks.clear();
        self.index.clear();
        self.memory.clear();
    }

    /// Replaces the context (a restored session).
    pub fn set_tokens(&mut self, tokens: &[u32]) {
        let common = self.toks.iter().zip(tokens).take_while(|(a, b)| a == b).count();
        self.truncate(common);
        self.extend(&tokens[common..]);
    }

    /// The longest earlier match of the context's suffix: `(position of its last token, match
    /// length, occurrences agreeing on the next token)`.
    fn find(&self) -> Option<(usize, usize, usize)> {
        let n = self.toks.len();
        if n < 3 {
            return None;
        }
        let cands = self.index.get(&bigram(self.toks[n - 2], self.toks[n - 1]))?;
        let mut best: Option<(usize, usize)> = None;
        let mut found: Vec<(usize, usize)> = Vec::new();
        for &p in cands.iter().rev().take(MAX_CANDIDATES + 1) {
            let p = p as usize;
            if p + 1 >= n {
                continue; // the suffix itself
            }
            let mut m = 2;
            while m < MAX_MATCH && m <= p && self.toks[p - m] == self.toks[n - 1 - m] {
                m += 1;
            }
            if m < self.opts.min_match {
                continue;
            }
            found.push((p, m));
            if best.is_none_or(|b| m > b.1) {
                best = Some((p, m));
            }
        }
        let (p, m) = best?;
        let next = self.toks[p + 1];
        let floor = m.min(4);
        let agree = found.iter().filter(|&&(q, mq)| mq >= floor && self.toks[q + 1] == next).count();
        Some((p, m, agree))
    }

    /// The draft for the context (whose last token is the one the target runs next), at most
    /// `k_max` tokens, with each token's calibrated chance of acceptance; empty without a match.
    pub fn draft(&mut self, k_max: usize) -> LookupDraft {
        self.stats.lookups += 1;
        let k_max = k_max.min(self.round_cap());
        let mut d = LookupDraft::default();
        if k_max == 0 {
            return d;
        }
        let Some((p, m, agree)) = self.find() else { return d };
        self.stats.matches += 1;
        d.matched = m;
        let n = self.toks.len();
        let cons = usize::from(agree >= 2);
        let mut ctx: Vec<u32> = self.toks[n.saturating_sub(MEMORY_CONTEXT)..].to_vec();
        for (i, &t) in self.toks[p + 1..n.min(p + 1 + k_max)].iter().enumerate() {
            if self.opts.memory && ctx.len() == MEMORY_CONTEXT && self.memory.refuted(&ctx, t) {
                d.vetoed = true;
                self.stats.vetoed += 1;
                break;
            }
            let class = (len_bucket(m) * 2 + cons) * DEPTHS + i.min(DEPTHS - 1);
            d.tokens.push(t);
            d.chances.push(self.policy.chance(class));
            d.class.push(class);
            if ctx.len() == MEMORY_CONTEXT {
                ctx.remove(0);
            }
            ctx.push(t);
        }
        d
    }

    /// How many of a draft's tokens to verify (see [`LookupOptions::cost_aware`]).
    pub fn cut(&mut self, d: &LookupDraft) -> usize {
        if d.tokens.is_empty() {
            return 0;
        }
        if !self.opts.cost_aware {
            return d.tokens.len();
        }
        let k = self.policy.cut(&d.chances);
        if k > 0 {
            self.policy.idle = 0;
            return k;
        }
        // Probe now and then so that a poor early estimate does not stick.
        self.policy.idle += 1;
        if self.policy.idle >= self.policy.probe {
            self.policy.idle = 0;
            self.policy.probe = (self.policy.probe * 2).min(Policy::PROBE_MAX);
            return d.tokens.len().min(2);
        }
        0
    }

    /// Records a verification round: `drafts` (the verified part of a draft, `class` its
    /// calibration classes, empty for drafts of another source), the committed tokens `out`
    /// (`accepted` drafts, then the target's own token if any), before `out` joins the context
    /// (whose last token is the one the round ran first).
    fn observe(&mut self, drafts: &[u32], class: &[usize], out: &[u32], accepted: usize) {
        if !class.is_empty() {
            self.policy.record_outcome(class, accepted);
            self.stats.rounds += 1;
            self.stats.drafted += drafts.len();
            self.stats.accepted += accepted;
            if accepted == drafts.len() {
                self.policy.probe = Policy::PROBE;
            }
        }
        if !self.opts.memory || drafts.is_empty() {
            return;
        }
        let n = self.toks.len();
        let mut ctx: Vec<u32> = self.toks[n.saturating_sub(MEMORY_CONTEXT)..].to_vec();
        for (j, &d) in drafts.iter().enumerate().take(accepted + 1) {
            let Some(&actual) = out.get(j) else { break };
            if ctx.len() == MEMORY_CONTEXT {
                self.memory.observe(&ctx, d, actual);
                ctx.remove(0);
            }
            ctx.push(actual);
        }
    }

    /// For a decoder with its own draft source: the lookup draft to verify this round instead
    /// (its tokens and calibration classes), if the context match is long (at least
    /// [`LONG_MATCH`] tokens: shorter ones are where draft models do better) and it promises more
    /// tokens per second than the other source's round would (`other_tokens` committed in
    /// `other_secs`; 0 seconds: compare tokens only).
    pub fn draft_over(&mut self, k_max: usize, other_tokens: f64, other_secs: f64) -> Option<(Vec<u32>, Vec<usize>)> {
        self.prepare();
        let mut d = self.draft(k_max);
        if d.matched < LONG_MATCH {
            return None;
        }
        let k = self.cut(&d);
        if k == 0 {
            return None;
        }
        let e = expected(&d.chances, k);
        let better = match self.policy.verify_time(k) {
            Some(v) if other_secs > 0.0 => e / v.max(1e-9) > other_tokens / other_secs,
            _ => e > other_tokens,
        };
        if !better {
            return None;
        }
        d.tokens.truncate(k);
        d.class.truncate(k);
        Some((d.tokens, d.class))
    }

    /// Records a round of a decoder that verified `drafts` (from this drafter when `class` is
    /// not empty) and committed `out`, then appends `out` to the context.
    pub fn commit(&mut self, drafts: &[u32], class: &[usize], out: &[u32], accepted: usize) {
        self.observe(drafts, class, out, accepted);
        self.extend(out);
    }
}

/// Keeps verification passes of up to `t` tokens on the batch-invariant decode kernels (the
/// process-wide [`crate::kernels::set_decode_max_tokens`], which never lowers it).
pub fn exact_up_to(t: usize) {
    if t > crate::kernels::decode_max_tokens() {
        crate::kernels::set_decode_max_tokens(t);
    }
}

/// The speculative test of `drafts` (each proposed with probability 1) against the target's
/// `logits` (`[drafts + 1, vocab]`): the committed tokens (ending at the first stop token, if any)
/// and the number of drafts accepted. With a repetition penalty, the accepted tokens count as
/// seen for the next positions, as one-token decoding would have it.
pub fn accept(sampler: &mut Sampler, logits: &[f32], v: usize, drafts: &[u32], stop: &[u32]) -> (Vec<u32>, usize) {
    let mut out = Vec::with_capacity(drafts.len() + 1);
    let mut accepted = 0;
    let penalty = sampler.repetition_penalty != 1.0;
    for (j, &d) in drafts.iter().enumerate() {
        let p = sampler.distribution(&logits[j * v..(j + 1) * v]);
        let t = speculative_accept(sampler, &p, &[(d, 1.0)], d).unwrap_or(d);
        sampler.commit(t);
        if penalty {
            sampler.observe(&[t]);
        }
        out.push(t);
        if t != d {
            break;
        }
        accepted += 1;
        if stop.contains(&d) {
            break;
        }
    }
    if accepted == drafts.len() && !out.last().is_some_and(|t| stop.contains(t)) {
        out.push(sampler.sample(&logits[drafts.len() * v..(drafts.len() + 1) * v]));
    }
    if let Some(i) = out.iter().position(|t| stop.contains(t)) {
        out.truncate(i + 1);
    }
    (out, accepted)
}

/// Prompt-lookup speculative decoding over a target model's session ([`crate::generate::RoundDecoder`];
/// see the module docs). The drafter's context should end with the session's tokens (drafts
/// are looked up in it): the decoder appends the tokens it commits, and when dropped removes the
/// last committed one again if the target has not run it.
pub struct LookupRounds<'a, M: Target> {
    model: &'a M,
    pub sess: &'a mut M::Sess,
    pub lookup: &'a mut Lookup,
    next: Option<u32>,
    /// Context length and session length when the rounds started.
    base: (usize, usize),
    pub stats: MtpStats,
}

impl<'a, M: Target> LookupRounds<'a, M> {
    pub fn new(model: &'a M, sess: &'a mut M::Sess, lookup: &'a mut Lookup) -> Self {
        let base = (lookup.len(), model.session_len(sess));
        lookup.prepare();
        Self {
            model,
            sess,
            lookup,
            next: None,
            base,
            stats: MtpStats::default(),
        }
    }
}

impl<M: Target> Drop for LookupRounds<'_, M> {
    fn drop(&mut self) {
        let n = self.model.session_len(self.sess);
        self.lookup.truncate(self.base.0 + n.saturating_sub(self.base.1));
    }
}

impl<M: Target> crate::generate::RoundDecoder for LookupRounds<'_, M> {
    fn push(&mut self, token: u32) {
        assert!(self.next.is_none(), "push follows prefill; step commits its own tokens");
        self.next = Some(token);
        self.lookup.extend(&[token]);
    }

    fn step(&mut self, sampler: &mut Sampler, max_new: usize) -> Vec<u32> {
        let next = self.next.expect("push a first token before step");
        let model = self.model;
        let n = model.session_len(self.sess);
        let t0 = crate::clock::Instant::now();
        let mut d = self.lookup.draft(max_new.max(1) - 1);
        let k = self.lookup.cut(&d);
        d.tokens.truncate(k);
        d.class.truncate(k);
        let draft_secs = t0.elapsed().as_secs_f64();
        let v = model.vocab_size();
        let stop = std::mem::take(&mut self.lookup.opts.stop);
        let (out, accepted) = if k == 0 {
            let logits = model.forward(self.sess, &[next]);
            let t = sampler.sample(&logits);
            (vec![t], 0)
        } else {
            let mut seq = Vec::with_capacity(k + 1);
            seq.push(next);
            seq.extend_from_slice(&d.tokens);
            let logits = model.forward_verify(self.sess, &seq, &[], &mut Vec::new(), self.lookup.opts.fast_verify);
            let (out, a) = accept(sampler, &logits, v, &d.tokens, &stop);
            model.truncate(self.sess, n + out.len());
            (out, a)
        };
        self.lookup.opts.stop = stop;
        let secs = t0.elapsed().as_secs_f64();
        self.lookup.policy.record_time(k, secs - draft_secs);
        self.lookup.commit(&d.tokens, &d.class, &out, accepted);
        self.next = out.last().copied();
        let st = &mut self.stats;
        st.rounds += 1;
        st.drafted += k;
        st.accepted += accepted;
        st.emitted += out.len();
        if st.per_round.len() <= accepted {
            st.per_round.resize(accepted + 1, 0);
        }
        st.per_round[accepted] += 1;
        st.draft_secs += draft_secs;
        st.verify_secs += secs - draft_secs;
        if std::env::var_os("OXINFER_DEBUG_DRAFT").is_some() && k > 0 {
            eprintln!(
                "[lookup: match {} k {k} accepted {accepted}{}; {:.1} ms]",
                d.matched,
                if d.vetoed { " (vetoed)" } else { "" },
                (secs - draft_secs) * 1e3
            );
        }
        out
    }

    fn rounds(&self) -> usize {
        self.stats.rounds
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate::{Flow, RoundDecoder, TextLm};

    const V: usize = 50;
    const H: usize = 4;

    /// A toy target whose greedy continuation is a function of the last three tokens (so text
    /// repeats and prompt lookup matches), with some spread for sampling.
    struct Toy;

    #[derive(Clone, Default)]
    struct ToySess(Vec<u32>);

    fn next_of(ctx: &[u32]) -> u32 {
        let n = ctx.len();
        let (a, b, c) = (ctx[n.saturating_sub(3)], ctx[n.saturating_sub(2)], ctx[n - 1]);
        // Mostly a cycle through the prompt's material, sometimes a jump.
        if (a + b + c) % 7 == 0 {
            (c * 3 + 1) % V as u32
        } else {
            (c + 1) % 12
        }
    }

    impl TextLm for Toy {
        type Sess = ToySess;
        fn new_session(&self) -> ToySess {
            ToySess::default()
        }
        fn hidden_size(&self) -> usize {
            H
        }
        fn n_layers(&self) -> usize {
            1
        }
        fn session_len(&self, s: &ToySess) -> usize {
            s.0.len()
        }
        fn forward_hidden_mm(&self, s: &mut ToySess, tokens: &[u32], _mm: &[(usize, &[f32])]) -> Vec<f32> {
            let mut out = vec![0f32; tokens.len() * H];
            for (i, &t) in tokens.iter().enumerate() {
                s.0.push(t);
                out[i * H] = next_of(&s.0) as f32;
                out[i * H + 1] = s.0.len() as f32;
            }
            out
        }
        fn logits(&self, hidden: &[f32], t: usize) -> Vec<f32> {
            let mut l = vec![0f32; t * V];
            for i in 0..t {
                let (next, pos) = (hidden[i * H] as usize, hidden[i * H + 1] as usize);
                for (j, x) in l[i * V..(i + 1) * V].iter_mut().enumerate() {
                    *x = ((j * 7 + pos * 3) % 11) as f32 * 0.1;
                }
                l[i * V + next] += 2.5;
            }
            l
        }
    }

    impl Target for Toy {
        fn forward_feats(
            &self,
            s: &mut ToySess,
            tokens: &[u32],
            mm: &[(usize, &[f32])],
            _layers: &[usize],
            feats: &mut Vec<f32>,
        ) -> Vec<f32> {
            feats.clear();
            self.forward_hidden_mm(s, tokens, mm)
        }
        fn embed_row(&self, _token: u32, out: &mut [f32]) {
            out.fill(0.0);
        }
        fn head_raw(&self, hidden: &[f32], t: usize) -> Vec<f32> {
            self.logits(hidden, t)
        }
        fn vocab_size(&self) -> usize {
            V
        }
        fn truncate(&self, s: &mut ToySess, len: usize) {
            s.0.truncate(len);
        }
    }

    #[test]
    fn index_and_matching() {
        let mut lk = Lookup::new(LookupOptions::default());
        lk.extend(&[1, 2, 3, 4, 5, 9, 9, 1, 2, 3]);
        let d = lk.draft(4);
        assert_eq!(d.tokens, [4, 5, 9, 9]);
        assert_eq!(d.matched, 3);
        lk.truncate(8);
        assert_eq!(lk.tokens(), [1, 2, 3, 4, 5, 9, 9, 1]);
        lk.extend(&[2, 7]);
        assert!(lk.draft(4).tokens.is_empty());
        // The longest match wins over a more recent shorter one.
        let mut lk = Lookup::new(LookupOptions::default());
        lk.extend(&[8, 1, 2, 3, 40, 0, 5, 2, 3, 41, 0, 8, 1, 2, 3]);
        assert_eq!(lk.draft(2).tokens, [40, 0]);
        lk.set_tokens(&[8, 1, 2, 3, 40, 0, 5, 2, 3]);
        assert_eq!(lk.draft(2).tokens, [40, 0]);
    }

    #[test]
    fn memory_vetoes_refuted_drafts() {
        let mut lk = Lookup::new(LookupOptions::default());
        lk.extend(&[10, 11, 12, 13, 14, 15, 10, 11, 12, 13]);
        assert_eq!(lk.draft(2).tokens, [14, 15]);
        // The target refuted 14 after [10, 11, 12, 13] and produced 20.
        let ctx = [10, 11, 12, 13];
        lk.memory.observe(&ctx, 14, 20);
        let d = lk.draft(2);
        assert!(d.tokens.is_empty() && d.vetoed);
        // Accepted there later: no veto any more.
        lk.memory.observe(&ctx, 14, 14);
        assert_eq!(lk.draft(2).tokens, [14, 15]);
        lk.opts.memory = false;
        lk.memory.observe(&ctx, 14, 20);
        assert_eq!(lk.draft(2).tokens, [14, 15]);
    }

    fn plain(prompt: &[u32], s: &mut Sampler, n: usize) -> Vec<u32> {
        let toy = Toy;
        let mut ps = toy.new_session();
        let hs = crate::generate::prefill(&toy, &mut ps, prompt, 3, &[]);
        let lg = toy.logits(&hs[hs.len() - H..], 1);
        crate::generate::decode(&toy, &mut ps, lg, s, &[], n, &mut |_| Flow::Continue).tokens
    }

    /// Greedy prompt-lookup decoding gives the plain greedy tokens, with every option, and keeps
    /// the context in step with the session.
    #[test]
    fn greedy_lookup_matches_plain_decoding() {
        let prompt: Vec<u32> = (0..12).chain(0..12).chain([3, 4, 5]).collect();
        let want = plain(&prompt, &mut Sampler::new(0.0, 0, 1.0, 1), 150);
        for (cost_aware, memory) in [(false, false), (false, true), (true, true), (true, false)] {
            let toy = Toy;
            let mut sess = toy.new_session();
            // The toy target is exact either way; fast verification leaves the process-wide
            // decode-kernel limit alone for the tests running alongside.
            let mut lk = Lookup::new(LookupOptions {
                cost_aware,
                memory,
                ..Default::default()
            });
            let hs = crate::generate::prefill(&toy, &mut sess, &prompt, 3, &[]);
            lk.extend(&prompt);
            let lg = toy.logits(&hs[hs.len() - H..], 1);
            let mut s = Sampler::new(0.0, 0, 1.0, 1);
            let (d, st) = {
                let mut dec = LookupRounds::new(&toy, &mut sess, &mut lk);
                let d = crate::generate::decode_rounds(&mut dec, &lg, &mut s, &[], 150, &mut |_| Flow::Continue);
                (d, dec.stats.clone())
            };
            assert_eq!(d.tokens, want, "cost_aware {cost_aware} memory {memory}");
            assert_eq!(lk.tokens(), &sess.0[..], "context in step");
            assert!(st.accepted > 50 && st.rounds < 120, "{st:?}");
            if !cost_aware {
                assert!(st.accepted < st.drafted, "some drafts rejected: {st:?}");
            }
        }
    }

    /// A filter whose mask depends on the token before: forbids the tokens whose sum with it is
    /// a multiple of 5 (the toy's usual successor of 2 and 7 among them).
    #[derive(Clone)]
    struct NotLikeLast(u32);
    impl crate::sampler::TokenFilter for NotLikeLast {
        fn mask(&mut self, logits: &mut [f32]) {
            for (i, l) in logits.iter_mut().enumerate() {
                if (i as u32 + self.0).is_multiple_of(5) {
                    *l = f32::NEG_INFINITY;
                }
            }
        }
        fn accept(&mut self, token: u32) {
            self.0 = token;
        }
        fn fork(&self) -> Box<dyn crate::sampler::TokenFilter> {
            Box::new(self.clone())
        }
        fn complete(&self) -> bool {
            true
        }
    }

    /// With a filter and a log-probability record on the sampler (constrained decoding),
    /// prompt-lookup rounds give the tokens and the record of plain decoding under the same
    /// filter: verification masks every position, and drafts outside the filter are rejected.
    #[test]
    fn guided_lookup_matches_guided_plain_decoding() {
        let prompt: Vec<u32> = (0..12).chain(0..12).chain([3, 4, 5]).collect();
        let guided = || {
            let mut s = Sampler::new(0.0, 0, 1.0, 1);
            let log = s.set_guide(Some(Box::new(NotLikeLast(5))), Some(2));
            (s, log)
        };
        let (mut s, want_log) = guided();
        let want = plain(&prompt, &mut s, 150);
        assert_eq!(s.take_guide().unwrap().tokens, want);
        assert!(want.windows(2).all(|w| !(w[0] + w[1]).is_multiple_of(5)) && !want[0].is_multiple_of(5));
        // The filter changes the reply: the test would pass trivially otherwise.
        assert_ne!(want, plain(&prompt, &mut Sampler::new(0.0, 0, 1.0, 1), 150));

        let toy = Toy;
        let mut sess = toy.new_session();
        let mut lk = Lookup::new(LookupOptions {
            cost_aware: false,
            ..Default::default()
        });
        let hs = crate::generate::prefill(&toy, &mut sess, &prompt, 3, &[]);
        lk.extend(&prompt);
        let lg = toy.logits(&hs[hs.len() - H..], 1);
        let (mut s, log) = guided();
        let (d, st) = {
            let mut dec = LookupRounds::new(&toy, &mut sess, &mut lk);
            let d = crate::generate::decode_rounds(&mut dec, &lg, &mut s, &[], 150, &mut |_| Flow::Continue);
            (d, dec.stats.clone())
        };
        assert_eq!(d.tokens, want);
        assert_eq!(s.take_guide().unwrap().tokens, want, "every committed token reached the guide");
        assert_eq!(*log.lock().unwrap(), *want_log.lock().unwrap());
        assert!(
            st.drafted > 0 && st.accepted > 0 && st.accepted < st.drafted,
            "drafts accepted and rejected: {st:?}"
        );
    }

    /// Sampled prompt-lookup decoding keeps the target's distribution: the first token after a
    /// prompt whose lookup always proposes the same continuation.
    #[test]
    fn sampled_lookup_keeps_the_distribution() {
        let prompt: Vec<u32> = (0..12).chain(0..5).collect();
        let toy = Toy;
        // Two samples of the same distribution over 50 tokens differ by about 0.02 in total
        // variation at this size.
        let n = 40_000;
        let (mut hist, mut want) = (vec![0usize; V], vec![0usize; V]);
        let mut s = Sampler::new(1.0, 0, 1.0, 11);
        for _ in 0..n {
            want[plain(&prompt, &mut s, 2)[1] as usize] += 1;
        }
        for _ in 0..n {
            let mut sess = toy.new_session();
            let mut lk = Lookup::new(LookupOptions {
                cost_aware: false,
                ..Default::default()
            });
            let hs = crate::generate::prefill(&toy, &mut sess, &prompt, 4, &[]);
            lk.extend(&prompt);
            let lg = toy.logits(&hs[hs.len() - H..], 1);
            let mut dec = LookupRounds::new(&toy, &mut sess, &mut lk);
            let first = s.sample(&lg);
            dec.push(first);
            let out = dec.step(&mut s, 3);
            hist[out[0] as usize] += 1;
        }
        let tv: f64 = hist.iter().zip(&want).map(|(&a, &b)| (a as f64 - b as f64).abs()).sum::<f64>() / (2.0 * n as f64);
        assert!(tv < 0.035, "total variation {tv}");
    }

    #[test]
    fn cost_aware_cut() {
        let mut p = Policy::default();
        assert_eq!(p.cut(&[0.9, 0.9]), 0, "unmeasured: a plain step first");
        p.record_time(0, 0.010);
        p.record_time(0, 0.010);
        // No verification measured yet: twice a plain step, so only good matches.
        assert_eq!(p.cut(&[0.5, 0.5]), 0);
        assert_eq!(p.cut(&[0.95, 0.95, 0.95, 0.95]), 4);
        p.record_time(4, 0.030);
        p.record_time(4, 0.030);
        // Verifying 4 costs 3 plain steps: only worth it with high chances.
        assert_eq!(p.cut(&[0.2, 0.2, 0.2, 0.2]), 0);
        assert_eq!(p.cut(&[0.95, 0.95, 0.95, 0.95]), 4);
    }
}
