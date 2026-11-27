//! Token sampling: greedy, or temperature + top-k + top-p, optionally with a repetition penalty,
//! a filter on what may be sampled (constrained decoding) and a record of log probabilities.

use std::borrow::Cow;
use std::cell::RefCell;
use std::sync::{Arc, Mutex};

/// Restricts what can be sampled ([`Sampler::set_guide`]): [`crate::constrain::Constraint`]
/// keeps a reply inside a grammar.
pub trait TokenFilter: Send {
    /// Sets the logits of the tokens that cannot come next to negative infinity (at least one
    /// must stay).
    fn mask(&mut self, logits: &mut [f32]);
    /// Records the token that came next.
    fn accept(&mut self, token: u32);
    /// A copy of the state, to follow draft tokens that may be thrown away.
    fn fork(&self) -> Box<dyn TokenFilter>;
    /// Whether what was accepted is complete (the reply may end).
    fn complete(&self) -> bool;
}

/// transformers' `NoRepeatNGramLogitsProcessor` (`no_repeat_ngram_size`): a token that would
/// complete an n-gram of `n` tokens already in the sequence (the prompt included) is banned,
/// on top of an optional inner filter.
pub struct NoRepeatNgram {
    n: usize,
    tokens: Vec<u32>,
    /// The tokens that followed each `n - 1`-token prefix.
    next: std::collections::HashMap<Vec<u32>, Vec<u32>>,
    inner: Option<Box<dyn TokenFilter>>,
}

impl NoRepeatNgram {
    /// The ban of n-grams of `n` tokens (at least 1) over `context` and what follows it.
    pub fn new(n: usize, context: &[u32], inner: Option<Box<dyn TokenFilter>>) -> Self {
        let mut f = Self {
            n: n.max(1),
            tokens: Vec::with_capacity(context.len() + 1024),
            next: std::collections::HashMap::new(),
            inner,
        };
        for &t in context {
            f.push(t);
        }
        f
    }

    fn push(&mut self, t: u32) {
        self.tokens.push(t);
        let n = self.n;
        if self.tokens.len() >= n {
            let g = &self.tokens[self.tokens.len() - n..];
            let e = self.next.entry(g[..n - 1].to_vec()).or_default();
            if !e.contains(&g[n - 1]) {
                e.push(g[n - 1]);
            }
        }
    }

    /// The tokens that cannot come next.
    pub fn banned(&self) -> &[u32] {
        let k = self.n - 1;
        if self.tokens.len() < k {
            return &[];
        }
        self.next.get(&self.tokens[self.tokens.len() - k..]).map_or(&[], |v| v.as_slice())
    }
}

impl TokenFilter for NoRepeatNgram {
    fn mask(&mut self, logits: &mut [f32]) {
        if let Some(f) = &mut self.inner {
            f.mask(logits);
        }
        let banned = self.banned();
        // At least one token stays.
        let left = logits
            .iter()
            .enumerate()
            .filter(|&(i, &l)| l > f32::NEG_INFINITY && !banned.contains(&(i as u32)))
            .count();
        if left > 0 {
            for &t in banned {
                if let Some(l) = logits.get_mut(t as usize) {
                    *l = f32::NEG_INFINITY;
                }
            }
        }
    }

    fn accept(&mut self, token: u32) {
        if let Some(f) = &mut self.inner {
            f.accept(token);
        }
        self.push(token);
    }

    fn fork(&self) -> Box<dyn TokenFilter> {
        Box::new(Self {
            n: self.n,
            tokens: self.tokens.clone(),
            next: self.next.clone(),
            inner: self.inner.as_ref().map(|f| f.fork()),
        })
    }

    fn complete(&self) -> bool {
        self.inner.as_ref().is_none_or(|f| f.complete())
    }
}

/// The log probability of a generated token and of the most likely tokens at its position, in
/// the distribution the model produced: the softmax of its logits, before the repetition
/// penalty, temperature, top-k, top-p and any constraint.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TokenLogprob {
    pub token: u32,
    pub logprob: f32,
    /// The most likely tokens, by decreasing probability (as many as were asked for).
    pub top: Vec<(u32, f32)>,
    /// Whether the token is part of the visible answer (not thinking, a delimiter, a tool call
    /// or the stop token); set by [`crate::engine::Engine`].
    pub content: bool,
}

impl TokenLogprob {
    /// The entry of `token` for `logits`, with the `top_n` most likely tokens.
    pub fn of(logits: &[f32], token: u32, top_n: usize) -> Self {
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f64 = logits.iter().map(|&l| ((l - max) as f64).exp()).sum();
        let lz = max as f64 + sum.ln();
        let mut top: Vec<(u32, f32)> = Vec::with_capacity(top_n + 1);
        if top_n > 0 {
            for (i, &l) in logits.iter().enumerate() {
                if top.len() < top_n || l > top[top.len() - 1].1 {
                    let at = top.partition_point(|t| t.1 >= l);
                    top.insert(at, (i as u32, l));
                    top.truncate(top_n);
                }
            }
        }
        for t in top.iter_mut() {
            t.1 = (t.1 as f64 - lz) as f32;
        }
        Self {
            token,
            logprob: logits.get(token as usize).map_or(f32::NAN, |&l| (l as f64 - lz) as f32),
            top,
            content: true,
        }
    }
}

/// The log probabilities recorded so far, one per token committed ([`Sampler::set_guide`]).
pub type LogprobLog = Arc<Mutex<Vec<TokenLogprob>>>;

/// What [`Sampler::take_guide`] reports.
#[derive(Clone, Debug, Default)]
pub struct GuideReport {
    /// Every token committed while the guide was set.
    pub tokens: Vec<u32>,
    /// Whether the filter's document is complete (None without a filter).
    pub complete: Option<bool>,
}

/// A filter and a log-probability record, with the state both need between calls.
struct Guide {
    filter: Option<Box<dyn TokenFilter>>,
    /// The filter's state along the draft tokens of a speculative round.
    draft: Option<Box<dyn TokenFilter>>,
    top_n: Option<usize>,
    /// The logits the next committed token is chosen from.
    last: Vec<f32>,
    tokens: Vec<u32>,
    log: LogprobLog,
}

pub struct Sampler {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    /// Min-p (0 = off): after top-k and top-p, tokens less likely than `min_p` times the most
    /// likely one are cut (transformers' `MinPLogitsWarper`, llama.cpp's `--min-p`).
    pub min_p: f32,
    /// Repetition penalty as transformers and vLLM apply it (1 = off): before anything else,
    /// the logit of every token seen so far (passed to [`Sampler::observe`], or sampled by
    /// [`Sampler::sample`]) is divided by it when positive and multiplied by it when negative.
    pub repetition_penalty: f32,
    rng: u64,
    /// Bit set of the tokens seen, for the repetition penalty.
    seen: Vec<u64>,
    guide: Option<RefCell<Guide>>,
}

impl Sampler {
    pub fn new(temperature: f32, top_k: usize, top_p: f32, seed: u64) -> Self {
        Self {
            temperature,
            top_k,
            top_p,
            min_p: 0.0,
            repetition_penalty: 1.0,
            rng: seed.max(1),
            seen: Vec::new(),
            guide: None,
        }
    }

    /// Guides the tokens sampled from now on: `filter` masks what cannot come next, and
    /// `logprobs` records each token's log probability with that many alternatives (in the
    /// returned log, as the tokens are committed). Both follow the tokens [`Sampler::sample`]
    /// returns and those given to [`Sampler::commit`]. With neither, any guide is removed.
    pub fn set_guide(&mut self, filter: Option<Box<dyn TokenFilter>>, logprobs: Option<usize>) -> LogprobLog {
        let log = LogprobLog::default();
        self.guide = (filter.is_some() || logprobs.is_some()).then(|| {
            RefCell::new(Guide {
                filter,
                draft: None,
                top_n: logprobs,
                last: Vec::new(),
                tokens: Vec::new(),
                log: log.clone(),
            })
        });
        log
    }

    /// Removes the guide and reports what it saw (None when none was set).
    pub fn take_guide(&mut self) -> Option<GuideReport> {
        let g = self.guide.take()?.into_inner();
        Some(GuideReport {
            tokens: g.tokens,
            complete: g.filter.map(|f| f.complete()),
        })
    }

    /// Whether a filter or a log-probability record is set.
    pub fn guided(&self) -> bool {
        self.guide.is_some()
    }

    /// Tells the guide that `token` was chosen from the logits last given to
    /// [`Sampler::distribution`] (speculative decoding commits tokens it did not get from
    /// [`Sampler::sample`]). Nothing to do without a guide.
    pub fn commit(&mut self, token: u32) {
        let Some(g) = &mut self.guide else { return };
        let g = g.get_mut();
        g.draft = None;
        if let Some(f) = &mut g.filter {
            f.accept(token);
        }
        g.tokens.push(token);
        if let Some(n) = g.top_n {
            let entry = TokenLogprob::of(&g.last, token, n);
            g.log.lock().unwrap_or_else(|p| p.into_inner()).push(entry);
        }
        g.last.clear();
    }

    /// Tells the guide that `token` was drafted after the ones committed and drafted so far:
    /// the next [`Sampler::draft_distribution`] is masked for what may follow it.
    pub fn draft_commit(&self, token: u32) {
        if let Some(g) = &self.guide {
            let g = &mut *g.borrow_mut();
            if g.draft.is_none() {
                g.draft = g.filter.as_ref().map(|f| f.fork());
            }
            if let Some(d) = &mut g.draft {
                d.accept(token);
            }
        }
    }

    /// `logits` with the repetition penalty and the guide's filter applied (the filter's state
    /// along the drafts when `draft`), and whether a filter masked them.
    fn prepared<'a>(&self, logits: &'a [f32], draft: bool) -> (Cow<'a, [f32]>, bool) {
        let mut l = self.penalized(logits);
        let Some(g) = &self.guide else { return (l, false) };
        let g = &mut *g.borrow_mut();
        if draft && g.draft.is_none() {
            g.draft = g.filter.as_ref().map(|f| f.fork());
        }
        if !draft && g.top_n.is_some() {
            g.last.clear();
            g.last.extend_from_slice(logits);
        }
        match if draft { &mut g.draft } else { &mut g.filter } {
            Some(f) => {
                f.mask(l.to_mut());
                (l, true)
            }
            None => (l, false),
        }
    }

    /// Sets the repetition penalty (see [`Sampler::repetition_penalty`]).
    pub fn with_repetition_penalty(mut self, penalty: f32) -> Self {
        self.repetition_penalty = penalty;
        self
    }

    /// Sets min-p (see [`Sampler::min_p`]).
    pub fn with_min_p(mut self, min_p: f32) -> Self {
        self.min_p = min_p;
        self
    }

    /// The token bit set of the repetition penalty (bit `t % 64` of word `t / 64`).
    pub fn seen_words(&self) -> &[u64] {
        &self.seen
    }

    /// Records tokens for the repetition penalty (e.g. the prompt, as transformers counts it).
    pub fn observe(&mut self, tokens: &[u32]) {
        for &t in tokens {
            let (w, b) = (t as usize / 64, t % 64);
            if w >= self.seen.len() {
                self.seen.resize(w + 1, 0);
            }
            self.seen[w] |= 1 << b;
        }
    }

    /// Forgets the tokens seen so far.
    pub fn clear_history(&mut self) {
        self.seen.clear();
    }

    /// Whether `token` was seen (repetition penalty).
    pub fn seen(&self, token: u32) -> bool {
        self.seen.get(token as usize / 64).is_some_and(|w| (w >> (token % 64)) & 1 == 1)
    }

    /// `logits` with the repetition penalty applied (borrowed when it is off).
    fn penalized<'a>(&self, logits: &'a [f32]) -> Cow<'a, [f32]> {
        if self.repetition_penalty == 1.0 || self.seen.iter().all(|&w| w == 0) {
            return Cow::Borrowed(logits);
        }
        let mut l = logits.to_vec();
        apply_repetition_penalty(&mut l, &self.seen, self.repetition_penalty);
        Cow::Owned(l)
    }

    fn next_f32(&mut self) -> f32 {
        // xorshift64*
        let mut x = self.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        let r = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (r >> 40) as f32 / (1u64 << 24) as f32
    }

    /// A uniform draw in [0, 1) from the sampler's generator (speculative acceptance tests).
    pub fn uniform(&mut self) -> f32 {
        self.next_f32()
    }

    /// Whether a temperature draw takes every token of a vocabulary of `n` as a candidate (no
    /// top-k or top-p cut): then the tokens are taken in token order ([`sample_full`]).
    fn whole_vocab(&self, n: usize) -> bool {
        (self.top_k == 0 || self.top_k >= n) && self.top_p >= 1.0 && self.min_p <= 0.0
    }

    /// Whether `sample` is a plain temperature draw over the whole vocabulary ([`sample_full`]
    /// with the generator's next uniform): above temperature 0, no top-k or top-p cut, no
    /// repetition penalty, no guide. A device can then sample itself from uniforms the host
    /// draws ([`Sampler::uniform`]).
    pub fn plain_sampled(&self) -> bool {
        self.temperature > 0.0
            && self.top_k == 0
            && self.top_p >= 1.0
            && self.min_p <= 0.0
            && self.repetition_penalty == 1.0
            && self.guide.is_none()
    }

    /// What a device needs to choose the tokens itself as [`Sampler::sample`] would (greedy or a
    /// draw, with top-k, top-p, min-p and the repetition penalty; [`DeviceSampling::pick`]),
    /// from uniforms the host draws ([`Sampler::uniform`]); None with a guide (a grammar's mask
    /// or a log-probability record stays on the host).
    pub fn device_sampling(&self) -> Option<DeviceSampling> {
        if self.guide.is_some() || self.temperature.is_nan() || !(self.plain_greedy() || device_sampling_enabled()) {
            return None;
        }
        Some(DeviceSampling {
            temperature: self.temperature.max(0.0),
            top_k: self.top_k,
            top_p: self.top_p,
            min_p: self.min_p,
            repetition_penalty: self.repetition_penalty,
            seen: if self.repetition_penalty != 1.0 {
                self.seen.clone()
            } else {
                Vec::new()
            },
        })
    }

    /// Whether a device can choose the tokens ([`Sampler::device_sampling`]).
    pub fn chainable(&self) -> bool {
        self.guide.is_none() && !self.temperature.is_nan() && (self.plain_greedy() || device_sampling_enabled())
    }

    /// The generator's state, to replay draws ([`Sampler::set_state`]): a device chain draws a
    /// uniform for every pass it queues and keeps the draws of the passes it keeps only.
    pub fn state(&self) -> u64 {
        self.rng
    }

    /// Restores a state [`Sampler::state`] returned, then skips `draws` uniforms.
    pub fn set_state(&mut self, state: u64, draws: usize) {
        self.rng = state;
        for _ in 0..draws {
            self.next_f32();
        }
    }

    /// Whether `sample` is greedy (argmax).
    pub fn greedy(&self) -> bool {
        self.temperature <= 0.0
    }

    /// Whether `sample` is the argmax of the logits as given (greedy, without a repetition
    /// penalty or a guide).
    pub fn plain_greedy(&self) -> bool {
        self.greedy() && self.repetition_penalty == 1.0 && self.guide.is_none()
    }

    /// The candidates `sample` draws from: token ids by decreasing probability (in token order
    /// when every token is one: no top-k or top-p cut), the
    /// (unnormalized over the cut) probabilities, how many survive top-p, and their total.
    /// With `masked`, tokens of probability zero (masked out) are never candidates.
    fn candidates(&self, logits: &[f32], masked: bool) -> (Vec<u32>, Vec<f32>, usize, f32) {
        if self.whole_vocab(logits.len()) {
            // Every token a candidate: in token order, no sort (a vocabulary of 200k took
            // milliseconds to sort per token).
            let (w, total) = full_weights(logits, 1.0 / self.temperature);
            let inv = (1.0 / total) as f32;
            let probs: Vec<f32> = w.iter().map(|&x| x * inv).collect();
            let total = probs.iter().map(|&p| p as f64).sum::<f64>() as f32;
            return ((0..logits.len() as u32).collect(), probs, logits.len(), total);
        }
        let (idx, w, cut, total) = cut_candidates(logits, 1.0 / self.temperature, self.top_k, self.top_p, self.min_p, masked);
        let inv = if total > 0.0 { 1.0 / total } else { 0.0 };
        let probs = w.iter().map(|&x| x * inv).collect();
        (idx, probs, cut, if total > 0.0 { 1.0 } else { 0.0 })
    }

    /// The exact distribution `sample` draws from (temperature, top-k, top-p applied), as
    /// `(token, probability)` pairs summing to 1 by decreasing probability (in token order
    /// without a top-k or top-p cut, every token listed); one token with
    /// probability 1 when greedy. With a guide, the distribution is the filtered one, and the
    /// token chosen from it must be given to [`Sampler::commit`].
    pub fn distribution(&self, logits: &[f32]) -> Vec<(u32, f32)> {
        self.dist(logits, false)
    }

    /// [`Sampler::distribution`] for a draft model's logits over the same vocabulary: with a
    /// filter, masked for what may follow the tokens committed and drafted so far
    /// ([`Sampler::draft_commit`]). Draft logits over another vocabulary go through
    /// [`Sampler::unguided_distribution`]; verification filters either way.
    pub fn draft_distribution(&self, logits: &[f32]) -> Vec<(u32, f32)> {
        self.dist(logits, true)
    }

    /// The distribution of `logits` without the guide: for draft logits over another
    /// vocabulary.
    pub fn unguided_distribution(&self, logits: &[f32]) -> Vec<(u32, f32)> {
        let logits = &*self.penalized(logits);
        if self.greedy() {
            return vec![(argmax(logits), 1.0)];
        }
        let (idx, probs, cut, total) = self.candidates(logits, false);
        idx[..cut].iter().zip(&probs[..cut]).map(|(&i, &p)| (i, p / total)).collect()
    }

    fn dist(&self, logits: &[f32], draft: bool) -> Vec<(u32, f32)> {
        let (logits, masked) = self.prepared(logits, draft);
        if self.greedy() {
            return vec![(argmax(&logits), 1.0)];
        }
        let (idx, probs, cut, total) = self.candidates(&logits, masked);
        idx[..cut].iter().zip(&probs[..cut]).map(|(&i, &p)| (i, p / total)).collect()
    }

    /// [`Sampler::distribution`] (`draft`: [`Sampler::draft_distribution`]) as a [`Dist`]:
    /// dense for a draw over the whole vocabulary without a guide's mask (the same
    /// probabilities, in token order, without building pairs).
    pub fn dist_of(&self, logits: &[f32], draft: bool) -> Dist {
        if !self.greedy() && self.whole_vocab(logits.len()) {
            let (logits, masked) = self.prepared(logits, draft);
            if !masked {
                let (mut w, total) = full_weights(&logits, 1.0 / self.temperature);
                let inv = if total > 0.0 { (1.0 / total) as f32 } else { 0.0 };
                for x in w.iter_mut() {
                    *x *= inv;
                }
                return Dist::Dense(w);
            }
        }
        Dist::Pairs(self.dist(logits, draft))
    }

    /// Draws from a [`Dist`] (dense ones in token order, as [`sample_full`]).
    pub fn sample_dist(&mut self, dist: &Dist) -> u32 {
        match dist {
            Dist::Pairs(p) => self.sample_from(p),
            Dist::Dense(p) => {
                let total = p.iter().map(|&x| x as f64).sum::<f64>();
                let u = self.next_f32();
                draw_weights(p, total, u)
            }
        }
    }

    /// Draws from a distribution such as `distribution` returns.
    pub fn sample_from(&mut self, dist: &[(u32, f32)]) -> u32 {
        let total: f32 = dist.iter().map(|d| d.1).sum();
        let mut r = self.next_f32() * total;
        for &(i, p) in dist {
            r -= p;
            if r <= 0.0 {
                return i;
            }
        }
        dist.last().map_or(0, |d| d.0)
    }

    /// Draws a token from `logits` (all tokens of the vocabulary). With a repetition penalty,
    /// the token is recorded as seen; with a guide, it is committed.
    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        let id = self.sample_inner(logits);
        if self.repetition_penalty != 1.0 {
            self.observe(&[id]);
        }
        self.commit(id);
        id
    }

    fn sample_inner(&mut self, logits: &[f32]) -> u32 {
        let (logits, masked) = self.prepared(logits, false);
        if self.greedy() {
            return argmax(&logits);
        }
        if !masked && self.whole_vocab(logits.len()) {
            let (w, total) = full_weights(&logits, 1.0 / self.temperature);
            let u = self.next_f32();
            return draw_weights(&w, total, u);
        }
        let (idx, w, cut, _) = cut_candidates(&logits, 1.0 / self.temperature, self.top_k, self.top_p, self.min_p, masked);
        let u = self.next_f32();
        idx[draw_prefix(&prefix_sums(&w[..cut]), u)]
    }
}

/// Applies a repetition penalty to `logits` (indexed by token id) for the tokens set in the bit
/// set `seen` (bit `t % 64` of word `t / 64`): positive logits are divided by `penalty`,
/// negative ones multiplied (transformers' `RepetitionPenaltyLogitsProcessor`).
pub fn apply_repetition_penalty(logits: &mut [f32], seen: &[u64], penalty: f32) {
    for (w, &bits) in seen.iter().enumerate() {
        let mut b = bits;
        while b != 0 {
            let t = w * 64 + b.trailing_zeros() as usize;
            b &= b - 1;
            if let Some(l) = logits.get_mut(t) {
                *l = if *l < 0.0 { *l * penalty } else { *l / penalty };
            }
        }
    }
}

/// The key a cut draw ranks logits by: decreasing value (NaN as -inf), then increasing token id
/// ([`f32::total_cmp`]'s order: -0 below +0; devices rank by the same order on the bits).
fn rank_cmp(logits: &[f32], a: u32, b: u32) -> std::cmp::Ordering {
    let key = |i: u32| {
        let l = logits[i as usize];
        if l.is_nan() { f32::NEG_INFINITY } else { l }
    };
    key(b).total_cmp(&key(a)).then(a.cmp(&b))
}

/// The candidates of a draw with top-k, top-p or min-p cuts, in rank order (decreasing logit,
/// then increasing token id: [`rank_cmp`]), with their weights `exp((l - max) * inv_t)`, how
/// many survive the cuts and the sum of their weights:
///
/// - top-k: the first `top_k` (0: all);
/// - top-p: then the shortest prefix whose weight reaches `top_p` times the top-k set's;
/// - min-p: then those of weight at least `min_p` (relative to the most likely token's 1);
/// - at least one; with `masked` (a guide's -inf), never a token of weight zero but the first.
///
/// [`draw_prefix`] then draws in rank order. Devices cut and draw the same way (Metal:
/// `ink_fsamp`), to rounding.
pub(crate) fn cut_candidates(
    logits: &[f32],
    inv_t: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    masked: bool,
) -> (Vec<u32>, Vec<f32>, usize, f32) {
    let n = logits.len();
    let kk = if top_k == 0 { n } else { top_k.min(n) };
    let cmp = |a: &u32, b: &u32| rank_cmp(logits, *a, *b);
    let max = logits[argmax(logits) as usize];
    let weight = |i: u32| {
        let l = logits[i as usize];
        if l > f32::NEG_INFINITY && max > f32::NEG_INFINITY {
            crate::kernels::fast_exp((l - max) * inv_t)
        } else {
            0.0
        }
    };
    let (mut idx, total_w): (Vec<u32>, f32) = if kk < n {
        let mut idx: Vec<u32> = (0..n as u32).collect();
        idx.select_nth_unstable_by(kk - 1, cmp);
        idx.truncate(kk);
        (idx, 0.0)
    } else {
        // Without top-k, top-p's set is a share of the whole vocabulary's weight W: tokens of
        // weight below (1 - top_p) W / 2n together weigh less than half of what top-p leaves
        // out, so the cut is among the others (and min-p's below min_p are cut anyway); only
        // those are sorted.
        let (w, total) = full_weights(logits, inv_t);
        let mut tau = if min_p > 0.0 { min_p } else { 0.0 };
        if top_p < 1.0 {
            tau = tau.max(((1.0 - top_p as f64) * total / (2 * n) as f64) as f32);
        }
        let idx = (0..n as u32)
            .filter(|&i| w[i as usize] >= tau && !(masked && w[i as usize] <= 0.0))
            .collect();
        (idx, total as f32)
    };
    if idx.is_empty() {
        idx.push(argmax(logits));
    }
    idx.sort_unstable_by(cmp);
    let w: Vec<f32> = idx.iter().map(|&i| weight(i)).collect();
    let p = prefix_sums(&w);
    let mut m = idx.len();
    if top_p < 1.0 {
        let d = if kk < n { p[kk - 1] } else { total_w };
        let target = top_p * d;
        if let Some(r) = p.iter().position(|&x| x >= target) {
            m = m.min(r + 1);
        }
    }
    if min_p > 0.0
        && let Some(r) = w.iter().position(|&x| x < min_p)
    {
        m = m.min(r);
    }
    let mut m = m.max(1);
    if masked {
        while m > 1 && w[m - 1] <= 0.0 {
            m -= 1;
        }
    }
    let total = p[m - 1];
    (idx, w, m, total)
}

/// The inclusive prefix sums of `w`.
fn prefix_sums(w: &[f32]) -> Vec<f32> {
    let mut acc = 0f32;
    w.iter()
        .map(|&x| {
            acc += x;
            acc
        })
        .collect()
}

/// The first index whose prefix sum (`p`, nondecreasing) reaches `u` times the last (the last
/// index when rounding leaves it short).
fn draw_prefix(p: &[f32], u: f32) -> usize {
    let r = u * p[p.len() - 1];
    p.iter().position(|&x| x >= r).unwrap_or(p.len() - 1)
}

/// Whether devices may choose sampled tokens themselves ([`Sampler::device_sampling`]; default
/// on, `OXINFER_DEVICE_SAMPLING=0` keeps every draw on the host's sampler, whose rounding differs
/// from a device's: the same distribution, but not always the same tokens from a seed).
pub fn device_sampling_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("OXINFER_DEVICE_SAMPLING").is_none_or(|v| v != "0"))
}

/// A device's sampling settings ([`Sampler::device_sampling`]): it chooses each token itself
/// as [`DeviceSampling::pick`] does, from a uniform the host draws, and adds it to `seen`.
#[derive(Clone, Debug, Default)]
pub struct DeviceSampling {
    /// 0: greedy (the argmax).
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    /// The tokens seen (repetition penalty; bit `t % 64` of word `t / 64`), empty without one.
    pub seen: Vec<u64>,
}

impl DeviceSampling {
    /// Whether a draw takes every token of a vocabulary of `n` (no top-k, top-p or min-p cut).
    pub fn whole(&self, n: usize) -> bool {
        (self.top_k == 0 || self.top_k >= n) && self.top_p >= 1.0 && self.min_p <= 0.0
    }

    /// Whether this is the plain argmax (greedy, no repetition penalty).
    pub fn plain_greedy(&self) -> bool {
        self.temperature <= 0.0 && self.repetition_penalty == 1.0
    }

    /// Whether a token's logit is penalized ([`Sampler::repetition_penalty`]).
    pub fn penalized(&self) -> bool {
        self.repetition_penalty != 1.0
    }

    /// Records `t` as seen (the repetition penalty's next logits).
    pub fn observe(&mut self, t: u32) {
        if !self.penalized() {
            return;
        }
        let (w, b) = (t as usize / 64, t % 64);
        if w >= self.seen.len() {
            self.seen.resize(w + 1, 0);
        }
        self.seen[w] |= 1 << b;
    }

    /// The token [`Sampler::sample`] chooses from logits the repetition penalty was already
    /// applied to (no guide), given the uniform it would draw: the reference of the device's
    /// choice.
    pub fn pick(&self, penalized: &[f32], u: f32) -> u32 {
        if self.temperature <= 0.0 {
            return argmax(penalized);
        }
        let inv_t = 1.0 / self.temperature;
        if self.whole(penalized.len()) {
            return sample_full(penalized, inv_t, u);
        }
        let (idx, w, m, _) = cut_candidates(penalized, inv_t, self.top_k, self.top_p, self.min_p, false);
        idx[draw_prefix(&prefix_sums(&w[..m]), u)]
    }
}

/// The weights `exp((l - max) * inv_t)` of every logit (0 for NaN and -inf) and their sum.
fn full_weights(logits: &[f32], inv_t: f32) -> (Vec<f32>, f64) {
    let max = logits[argmax(logits) as usize];
    let mut w = vec![0f32; logits.len()];
    let mut total = 0f64;
    if max == f32::NEG_INFINITY || max.is_nan() {
        return (w, total);
    }
    for (wc, lc) in w.chunks_mut(1024).zip(logits.chunks(1024)) {
        let mut s = 0f32;
        for (o, &l) in wc.iter_mut().zip(lc) {
            // (fast_exp clamps: -inf and NaN by hand.)
            let e = if l > f32::NEG_INFINITY {
                crate::kernels::fast_exp((l - max) * inv_t)
            } else {
                0.0
            };
            *o = e;
            s += e;
        }
        total += s as f64;
    }
    (w, total)
}

/// The first index in order at which the cumulative weight of `w` reaches `u` times its total
/// (the last of nonzero weight when rounding leaves `u` past the end; 0 when all are zero).
fn draw_weights(w: &[f32], total: f64, u: f32) -> u32 {
    let mut r = u as f64 * total;
    let mut last = 0;
    for (i, &x) in w.iter().enumerate() {
        if x > 0.0 {
            r -= x as f64;
            last = i;
            if r <= 0.0 {
                return i as u32;
            }
        }
    }
    last as u32
}

/// A temperature draw over the whole vocabulary from the uniform `u` in [0, 1): the first token
/// in token order at which the cumulative weight `exp((l - max) * inv_t)` reaches `u` times
/// the total (the inverse of the distribution's CDF in token order; the last token of nonzero
/// weight when rounding leaves `u` past the end). Device samplers draw the same way.
pub fn sample_full(logits: &[f32], inv_t: f32, u: f32) -> u32 {
    let (w, total) = full_weights(logits, inv_t);
    draw_weights(&w, total, u)
}

/// A distribution a token is drawn from ([`Sampler::dist_of`]): `(token, probability)` pairs as
/// [`Sampler::distribution`] gives them, or, for a draw over the whole vocabulary, every
/// token's probability in token order (no pairs to build: speculative rounds take one a
/// verified position).
#[derive(Clone, Debug, PartialEq)]
pub enum Dist {
    Pairs(Vec<(u32, f32)>),
    Dense(Vec<f32>),
}

impl Dist {
    /// One token with probability 1 (a greedy or a looked-up draft).
    pub fn point(t: u32) -> Self {
        Dist::Pairs(vec![(t, 1.0)])
    }

    /// The probability of `t`.
    pub fn prob(&self, t: u32) -> f32 {
        match self {
            Dist::Pairs(p) => p.iter().find(|e| e.0 == t).map_or(0.0, |e| e.1),
            Dist::Dense(p) => p.get(t as usize).copied().unwrap_or(0.0),
        }
    }

    /// The first token listed (the most likely of pairs; a greedy distribution's only one).
    pub fn first(&self) -> u32 {
        match self {
            Dist::Pairs(p) => p.first().map_or(0, |e| e.0),
            Dist::Dense(p) => argmax(p),
        }
    }
}

/// The index of the first largest value (NaNs never win; 0 when nothing beats -inf).
pub fn argmax(x: &[f32]) -> u32 {
    #[cfg(target_arch = "aarch64")]
    if x.len() >= 64 {
        // SAFETY: NEON is baseline on aarch64; every load is inside `x`.
        return unsafe { argmax_neon(x) };
    }
    argmax_scalar(x)
}

fn argmax_scalar(x: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in x.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best as u32
}

/// [`argmax_scalar`] in two passes: the largest value (FMAXNM ignores NaNs as `>` does), then
/// its first index. A vocabulary of logits per decoded token: the scalar loop's compare and
/// branch per element took ~70 us over Llama 3's 128k.
#[cfg(target_arch = "aarch64")]
unsafe fn argmax_neon(x: &[f32]) -> u32 {
    unsafe {
        use std::arch::aarch64::*;
        let (n, p) = (x.len(), x.as_ptr());
        let mut m = [vdupq_n_f32(f32::NEG_INFINITY); 4];
        let mut i = 0;
        while i + 16 <= n {
            for k in 0..4 {
                m[k] = vmaxnmq_f32(m[k], vld1q_f32(p.add(i + 4 * k)));
            }
            i += 16;
        }
        let mut best = vmaxnmvq_f32(vmaxnmq_f32(vmaxnmq_f32(m[0], m[1]), vmaxnmq_f32(m[2], m[3])));
        for &v in &x[i..] {
            if v > best {
                best = v;
            }
        }
        // Nothing above -inf (all -inf or NaN): the scalar loop keeps index 0.
        if best.is_nan() || best == f32::NEG_INFINITY {
            return 0;
        }
        let b = vdupq_n_f32(best);
        let mut i = 0;
        while i + 16 <= n {
            let e = [0, 1, 2, 3].map(|k| vceqq_f32(vld1q_f32(p.add(i + 4 * k)), b));
            if vmaxvq_u32(vorrq_u32(vorrq_u32(e[0], e[1]), vorrq_u32(e[2], e[3]))) != 0 {
                break;
            }
            i += 16;
        }
        (i + x[i..].iter().position(|&v| v == best).unwrap_or(0)) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The n-gram ban over the context and the accepted tokens (transformers' rule).
    #[test]
    fn no_repeat_ngram_bans_completions() {
        // Context 1 2 3 4 5 ... 1 2 3: with n = 4, 4 cannot follow (1 2 3 4 is there).
        let mut f = NoRepeatNgram::new(4, &[1, 2, 3, 4, 5, 9, 1, 2, 3], None);
        let mut l = vec![0f32; 10];
        f.mask(&mut l);
        assert_eq!(l[4], f32::NEG_INFINITY);
        assert!(l.iter().enumerate().all(|(i, &v)| i == 4 || v == 0.0));
        f.accept(7);
        // Now 2 3 7 then 3 7 ...: nothing banned.
        assert!(f.banned().is_empty());
        let mut g = NoRepeatNgram::new(2, &[5, 6], None);
        g.accept(5);
        assert_eq!(g.banned(), &[6]);
        assert!(g.fork().complete());
    }

    /// Cut draws: top-k, then top-p over the top-k set's weight, then min-p against the most
    /// likely token, ties ranked by token id, the draw in rank order; `DeviceSampling::pick`
    /// gives what `Sampler::sample` gives from the same uniform.
    #[test]
    fn cut_draws_follow_the_cuts() {
        // Weights at temperature 1: e^0, e^-1 (twice, tied), e^-2, e^-3, and a far tail.
        let mut logits = vec![-30.0f32; 64];
        logits[9] = 0.0;
        logits[4] = -1.0;
        logits[2] = -1.0;
        logits[7] = -2.0;
        logits[30] = -3.0;
        let e = |x: f32| x.exp();
        let cut = |k, p, mp| {
            let (idx, _, m, _) = cut_candidates(&logits, 1.0, k, p, mp, false);
            idx[..m].to_vec()
        };
        assert_eq!(cut(3, 1.0, 0.0), vec![9, 2, 4]);
        assert_eq!(cut(2, 1.0, 0.0), vec![9, 2]);
        // top-p: 1 / (1 + 2/e + ...) of the top-4 weight is 0.52.
        let w4 = 1.0 + 2.0 * e(-1.0) + e(-2.0);
        assert_eq!(cut(4, 0.5, 0.0), vec![9]);
        assert_eq!(cut(4, (1.0 + e(-1.0)) / w4 - 0.01, 0.0), vec![9, 2]);
        assert_eq!(cut(0, 0.9, 0.0), vec![9, 2, 4]);
        assert_eq!(cut(0, 0.95, 0.0), vec![9, 2, 4, 7]);
        // min-p: weight at least min_p (e^-1 = 0.37 kept at 0.3, cut at 0.4).
        assert_eq!(cut(0, 1.0, 0.3), vec![9, 2, 4]);
        assert_eq!(cut(0, 1.0, 0.4), vec![9]);
        assert_eq!(cut(40, 0.99, 0.1), vec![9, 2, 4, 7]);
        // Draw frequencies over a cut, and the device reference against the sampler.
        let mut s = Sampler::new(1.0, 4, 1.0, 5).with_min_p(0.2);
        let mut ds = s.device_sampling().unwrap();
        ds.seen.clear();
        let mut hist = [0usize; 64];
        let n = 100_000;
        for _ in 0..n {
            let st = s.state();
            let u = s.uniform();
            s.set_state(st, 0);
            let t = s.sample(&logits);
            assert_eq!(ds.pick(&logits, u), t);
            hist[t as usize] += 1;
        }
        let z = 1.0 + 2.0 * e(-1.0);
        for (t, want) in [(9usize, 1.0 / z), (2, e(-1.0) / z), (4, e(-1.0) / z), (7, 0.0)] {
            let f = hist[t] as f32 / n as f32;
            assert!((f - want).abs() < 0.01, "token {t}: {f} vs {want}");
        }
        // The repetition penalty: the device's settings carry the tokens seen.
        let mut s = Sampler::new(0.0, 0, 1.0, 1).with_repetition_penalty(2.0);
        s.observe(&[9]);
        let ds = s.device_sampling().unwrap();
        assert!(ds.penalized() && !ds.plain_greedy() && ds.seen[0] == 1 << 9);
        assert!(Sampler::new(0.0, 0, 1.0, 1).device_sampling().unwrap().plain_greedy());
    }

    /// Whole-vocabulary temperature draws ([`sample_full`], no sort) keep the softmax's
    /// frequencies, and the distribution lists every token in token order.
    #[test]
    fn whole_vocab_draws_follow_the_softmax() {
        let logits = [1.0f32, f32::NEG_INFINITY, 0.0, 2.0, -1.0, f32::NAN];
        let mut s = Sampler::new(0.8, 0, 1.0, 11);
        let d = s.distribution(&logits);
        let z: f64 = logits
            .iter()
            .filter(|l| l.is_finite())
            .map(|&l| ((l - 2.0) / 0.8).exp() as f64)
            .sum();
        for (t, p) in &d {
            let l = logits[*t as usize];
            let want = if l.is_finite() { ((l - 2.0) / 0.8).exp() as f64 / z } else { 0.0 };
            assert!((*p as f64 - want).abs() < 1e-6, "{t}: {p} vs {want}");
        }
        assert!(d.windows(2).all(|w| w[0].0 < w[1].0));
        let n = 200_000;
        let mut hist = [0usize; 6];
        for _ in 0..n {
            hist[s.sample(&logits) as usize] += 1;
        }
        assert_eq!((hist[1], hist[5]), (0, 0));
        for (t, p) in &d {
            let f = hist[*t as usize] as f32 / n as f32;
            assert!((f - p).abs() < 0.005, "token {t}: {f} vs {p}");
        }
        // The end of the range and replayed states.
        assert_eq!(sample_full(&logits, 1.0, 0.0), 0);
        assert_eq!(sample_full(&logits, 1.0, 0.999_999_9), 4);
        let st = s.state();
        let a: Vec<u32> = (0..8).map(|_| s.sample(&logits)).collect();
        s.set_state(st, 3);
        let b: Vec<u32> = (0..5).map(|_| s.sample(&logits)).collect();
        assert_eq!(&a[3..], &b[..]);
    }

    /// The NEON argmax against the scalar loop: ties (first index), NaNs, infinities.
    #[test]
    fn argmax_matches_scalar() {
        let mut seed = 9u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 40) as u32
        };
        for n in [64usize, 65, 100, 1000, 128256] {
            for case in 0..6 {
                let mut x: Vec<f32> = (0..n).map(|_| (rnd() % 1000) as f32 / 7.0 - 50.0).collect();
                match case {
                    1 => x.iter_mut().step_by(3).for_each(|v| *v = f32::NAN),
                    2 => x.fill(f32::NEG_INFINITY),
                    3 => {
                        x.fill(f32::NAN);
                        x[n - 1] = f32::NEG_INFINITY;
                    }
                    4 => {
                        x[n / 2] = f32::INFINITY;
                        x[n - 1] = f32::INFINITY;
                    }
                    5 => x[0] = f32::NAN,
                    _ => {}
                }
                assert_eq!(argmax(&x), argmax_scalar(&x), "n {n} case {case}");
            }
        }
    }

    #[test]
    fn repetition_penalty() {
        let mut s = Sampler::new(0.0, 0, 1.0, 1).with_repetition_penalty(2.0);
        // Token 1 wins until it has been seen; then 3.0 / 2 < 2.0.
        let logits = [0.5, 3.0, 2.0, -1.0];
        assert_eq!(s.sample(&logits), 1);
        assert!(s.seen(1) && !s.seen(2));
        assert_eq!(s.sample(&logits), 2);
        let mut l = logits.to_vec();
        s.observe(&[3]);
        apply_repetition_penalty(&mut l, &s.seen, 2.0);
        assert_eq!(l, vec![0.5, 1.5, 1.0, -2.0]);
        s.clear_history();
        assert_eq!(s.sample(&logits), 1);
    }

    /// Allows the tokens of a list, one position after the other.
    #[derive(Clone)]
    struct Script(Vec<Vec<u32>>, usize);
    impl TokenFilter for Script {
        fn mask(&mut self, logits: &mut [f32]) {
            for (i, l) in logits.iter_mut().enumerate() {
                if !self.0[self.1].contains(&(i as u32)) {
                    *l = f32::NEG_INFINITY;
                }
            }
        }
        fn accept(&mut self, _token: u32) {
            self.1 += 1;
        }
        fn fork(&self) -> Box<dyn TokenFilter> {
            Box::new(self.clone())
        }
        fn complete(&self) -> bool {
            self.1 == self.0.len()
        }
    }

    #[test]
    fn guide_masks_commits_and_records() {
        let logits = [2.0f32, 1.0, 0.0, -1.0];
        let script = Script(vec![vec![1, 2], vec![3], vec![0, 3]], 0);
        // Greedy: the best allowed token at each position.
        let mut s = Sampler::new(0.0, 0, 1.0, 1);
        let log = s.set_guide(Some(Box::new(script.clone())), Some(2));
        assert_eq!(s.sample(&logits), 1);
        // A speculative round: the target's distribution, then the token committed.
        assert_eq!(s.distribution(&logits), vec![(3, 1.0)]);
        // Drafts follow the filter's state without moving it.
        s.draft_commit(3);
        assert_eq!(s.draft_distribution(&logits), vec![(0, 1.0)]);
        assert_eq!(s.distribution(&logits), vec![(3, 1.0)]);
        s.commit(3);
        assert_eq!(s.sample(&logits), 0);
        let r = s.take_guide().unwrap();
        assert_eq!((r.tokens, r.complete), (vec![1, 3, 0], Some(true)));
        assert!(!s.guided() && s.take_guide().is_none());
        // Log probabilities are of the unmasked softmax.
        let log = log.lock().unwrap();
        let z: f32 = logits.iter().map(|l| l.exp()).sum();
        assert_eq!(log.len(), 3);
        assert!((log[0].logprob - (1.0 - z.ln())).abs() < 1e-5 && (log[1].logprob - (-1.0 - z.ln())).abs() < 1e-5);
        assert_eq!(log[1].top.iter().map(|t| t.0).collect::<Vec<_>>(), vec![0, 1]);
        assert!((log[1].top[0].1 - (2.0 - z.ln())).abs() < 1e-5);

        // Sampling never leaves the allowed set, whatever top-k and top-p are.
        for seed in 1..200 {
            let mut s = Sampler::new(1.5, 3, 0.9, seed);
            s.set_guide(Some(Box::new(script.clone())), None);
            let a = s.sample(&logits);
            assert!(a == 1 || a == 2);
            assert!(s.distribution(&logits).iter().all(|d| d.0 == 3));
            s.commit(3);
            let c = s.sample(&logits);
            assert!(c == 0 || c == 3);
        }
        // Without a guide nothing changes.
        let mut s = Sampler::new(0.0, 0, 1.0, 1);
        s.commit(5);
        assert_eq!(s.sample(&logits), 0);
    }
}
