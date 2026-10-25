//! Token sampling: greedy, or temperature + top-k + top-p, optionally with a repetition penalty.

use std::borrow::Cow;

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
    fn candidates(&self, logits: &[f32]) -> (Vec<u32>, Vec<f32>, usize, f32) {
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
        let total: f32 = probs[..cut].iter().sum();
        (idx, probs, cut, total)
    }

    /// The exact distribution `sample` draws from (temperature, top-k, top-p applied), as
    /// `(token, probability)` pairs summing to 1 by decreasing probability; one token with
    /// probability 1 when greedy.
    pub fn distribution(&self, logits: &[f32]) -> Vec<(u32, f32)> {
        let logits = &*self.penalized(logits);
        if self.greedy() {
            return vec![(argmax(logits), 1.0)];
        }
        let (idx, probs, cut, total) = self.candidates(logits);
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
    /// the token is recorded as seen.
    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        let id = self.sample_inner(logits);
        if self.repetition_penalty != 1.0 {
            self.observe(&[id]);
        }
        id
    }

    fn sample_inner(&mut self, logits: &[f32]) -> u32 {
        let logits = self.penalized(logits);
        if self.greedy() {
            return argmax(&logits);
        }
        let (idx, probs, cut, total) = self.candidates(&logits);
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
}
