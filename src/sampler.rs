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

    /// Whether `sample` is greedy (argmax).
    pub fn greedy(&self) -> bool {
        self.temperature <= 0.0
    }

    /// The candidates `sample` draws from: token ids by decreasing probability, the
    /// (unnormalized over the cut) probabilities, how many survive top-p, and their total.
    /// With `masked`, tokens of probability zero (masked out) are never candidates.
    fn candidates(&self, logits: &[f32], masked: bool) -> (Vec<u32>, Vec<f32>, usize, f32) {
        let k = if self.top_k == 0 {
            logits.len()
        } else {
            self.top_k.min(logits.len())
        };
        // Partial selection of the top-k indices.
        let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
        idx.select_nth_unstable_by(k - 1, |&a, &b| {
            logits[b as usize]
                .partial_cmp(&logits[a as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        idx.truncate(k);
        idx.sort_unstable_by(|&a, &b| {
            logits[b as usize]
                .partial_cmp(&logits[a as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let inv_t = 1.0 / self.temperature;
        let max = logits[idx[0] as usize];
        let mut probs: Vec<f32> = idx.iter().map(|&i| ((logits[i as usize] - max) * inv_t).exp()).collect();
        let sum: f32 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= sum;
        }
        // top-p
        let mut cut = probs.len();
        if self.top_p < 1.0 {
            let mut acc = 0f32;
            for (i, &p) in probs.iter().enumerate() {
                acc += p;
                if acc >= self.top_p {
                    cut = i + 1;
                    break;
                }
            }
        }
        if masked {
            while cut > 1 && probs[cut - 1] <= 0.0 {
                cut -= 1;
            }
        }
        let total: f32 = probs[..cut].iter().sum();
        (idx, probs, cut, total)
    }

    /// The exact distribution `sample` draws from (temperature, top-k, top-p applied), as
    /// `(token, probability)` pairs summing to 1 by decreasing probability; one token with
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
        let (idx, probs, cut, total) = self.candidates(&logits, masked);
        let mut r = self.next_f32() * total;
        for i in 0..cut {
            r -= probs[i];
            if r <= 0.0 {
                return idx[i];
            }
        }
        idx[cut - 1]
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

pub fn argmax(x: &[f32]) -> u32 {
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

#[cfg(test)]
mod tests {
    use super::*;

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
