//! Decoding of encoded windows, several at once: greedy or sampled sequences (the most likely of
//! `best_of` samples) and beam search, as openai-whisper's `GreedyDecoder`, `BeamSearchDecoder`
//! and `MaximumLikelihoodRanker` do them.
//!
//! Every hypothesis of every window is one row of a decoder step
//! ([`WhisperModel::decode_step`]), and a row's values do not depend on the rows it is computed
//! with, so a window decodes to the same tokens alone or in a batch. Beam search keeps one
//! self-attention cache per beam: a beam continued once hands its cache on, a beam continued
//! several times is copied (openai-whisper's `rearrange_kv_cache`).

use super::model::{CrossKv, DecState, WhisperModel};
use super::tokens::Special;
use super::{Rules, log_softmax, log_softmax_at, process, softmax_prob};
use crate::sampler::{Sampler, argmax};
use std::sync::atomic::{AtomicBool, Ordering};

/// How the tokens of a window are chosen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Search {
    /// The most likely token (temperature 0) or a sampled one, for `n` independent sequences
    /// of which the best ranked is kept.
    Sample { temperature: f32, n: usize },
    /// Beam search over `size` beams until `max_candidates` hypotheses have ended.
    Beam { size: usize, max_candidates: usize },
}

/// One window to decode.
pub(super) struct Job<'a> {
    pub cross: &'a CrossKv,
    /// The decoder prompt: context, the start-of-transcript sequence at `sot_index`, a prefix.
    pub input: &'a [u32],
    pub sot_index: usize,
    pub rules: &'a Rules<'a>,
    pub search: Search,
    /// Ranking of finished hypotheses: None divides the log-probability by the length, Some(a)
    /// by `((5 + length) / 6)^a` (the Google NMT penalty).
    pub length_penalty: Option<f32>,
    /// Tokens generated at most.
    pub max_new: usize,
    pub seed: u64,
}

/// One decoding attempt of a window.
pub(super) struct Attempt {
    /// Generated tokens, `<|endoftext|>` included when the sequence ended.
    pub tokens: Vec<u32>,
    /// log-softmax of the processed logits at each chosen token.
    pub logprobs: Vec<f32>,
    pub no_speech_prob: f32,
    pub cancelled: bool,
    /// Decoder passes that produced a token, over every hypothesis.
    pub steps: usize,
}

/// A sequence being generated.
struct Hyp {
    st: DecState,
    tokens: Vec<u32>,
    logprobs: Vec<f32>,
    /// Sum of `logprobs`, accumulated in generation order.
    sum: f32,
    /// Logits of the next token (unprocessed).
    logits: Vec<f32>,
    sampler: Option<Sampler>,
    live: bool,
}

/// A hypothesis that ended.
struct Done {
    tokens: Vec<u32>,
    logprobs: Vec<f32>,
    sum: f32,
}

struct Run<'a> {
    job: &'a Job<'a>,
    hyps: Vec<Hyp>,
    /// Beam search: ended hypotheses, in the order they were found.
    finished: Vec<Done>,
    no_speech_prob: f32,
    steps: usize,
    /// Token choices made (beam search: updates).
    iters: usize,
    complete: bool,
}

/// Decodes the windows together and returns one attempt per job, in order. Stops at the next
/// token when `cancel` is set (every attempt is then `cancelled`).
pub(super) fn run(m: &WhisperModel, sp: &Special, jobs: &[Job], cancel: &AtomicBool) -> Vec<Attempt> {
    let d = m.cfg.d_model;
    let v = m.cfg.vocab_size;
    // The prompts: one pass per window, exactly as when it is decoded alone.
    let mut runs: Vec<Run> = jobs
        .iter()
        .map(|job| {
            let mut st = m.new_state();
            let h = m.decode(job.input, &mut st, job.cross);
            let t = job.input.len();
            // No-speech probability: the raw logits at <|startoftranscript|>.
            let no_speech_prob = softmax_prob(&m.logits(&h[job.sot_index * d..(job.sot_index + 1) * d]), sp.no_speech as usize);
            let logits = m.logits(&h[(t - 1) * d..t * d]);
            let hyps = match job.search {
                Search::Sample { temperature, n } => (0..n.max(1))
                    .map(|j| Hyp {
                        st: st.clone(),
                        tokens: Vec::new(),
                        logprobs: Vec::new(),
                        sum: 0.0,
                        logits: logits.clone(),
                        sampler: Some(Sampler::new(
                            temperature,
                            0,
                            1.0,
                            job.seed.wrapping_add((j as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)).max(1),
                        )),
                        live: true,
                    })
                    .collect(),
                // The beams all start from the prompt: one row until the first tokens tell
                // them apart (openai-whisper's identical rows collapse to the same candidates).
                Search::Beam { .. } => vec![Hyp {
                    st,
                    tokens: Vec::new(),
                    logprobs: Vec::new(),
                    sum: 0.0,
                    logits,
                    sampler: None,
                    live: true,
                }],
            };
            Run {
                job,
                hyps,
                finished: Vec::new(),
                no_speech_prob,
                steps: 0,
                iters: 0,
                complete: job.max_new == 0,
            }
        })
        .collect();

    let mut cancelled = false;
    while runs.iter().any(|r| !r.complete) {
        if cancel.load(Ordering::SeqCst) {
            cancelled = true;
            break;
        }
        for r in runs.iter_mut().filter(|r| !r.complete) {
            r.advance(m, sp);
        }
        // One decoder step for every sequence still going, of every window.
        let mut toks: Vec<u32> = Vec::new();
        let mut sts: Vec<&mut DecState> = Vec::new();
        let mut cross: Vec<&CrossKv> = Vec::new();
        for r in runs.iter_mut().filter(|r| !r.complete) {
            for h in r.hyps.iter_mut().filter(|h| h.live) {
                toks.push(*h.tokens.last().unwrap());
                sts.push(&mut h.st);
                cross.push(r.job.cross);
            }
        }
        if toks.is_empty() {
            continue;
        }
        let logits = m.decode_step(&toks, &mut sts, &cross);
        let mut rows = logits.chunks(v);
        for r in runs.iter_mut().filter(|r| !r.complete) {
            for h in r.hyps.iter_mut().filter(|h| h.live) {
                h.logits.copy_from_slice(rows.next().unwrap());
            }
        }
    }
    runs.into_iter()
        .map(|r| {
            let (no_speech_prob, steps) = (r.no_speech_prob, r.steps);
            let best = r.finalize(sp.eot);
            Attempt {
                tokens: best.tokens,
                logprobs: best.logprobs,
                no_speech_prob,
                cancelled,
                steps,
            }
        })
        .collect()
}

impl Run<'_> {
    /// Chooses the next token of every sequence from its logits.
    fn advance(&mut self, m: &WhisperModel, sp: &Special) {
        let job = self.job;
        let max_pos = m.cfg.max_target_positions;
        self.iters += 1;
        match job.search {
            Search::Sample { temperature, .. } => {
                for h in self.hyps.iter_mut().filter(|h| h.live) {
                    process(sp, &mut h.logits, &h.tokens, job.rules);
                    let tok = if temperature > 0.0 {
                        h.sampler.as_mut().unwrap().sample(&h.logits)
                    } else {
                        argmax(&h.logits)
                    };
                    let lp = log_softmax_at(&h.logits, tok as usize);
                    h.logprobs.push(lp);
                    h.sum += lp;
                    h.tokens.push(tok);
                    self.steps += 1;
                    h.live = tok != sp.eot && h.st.pos < max_pos && h.tokens.len() < job.max_new;
                }
                self.complete = self.hyps.iter().all(|h| !h.live);
            }
            Search::Beam { size, max_candidates } => {
                // Candidates: the size + 1 most likely continuations of every beam.
                let mut cands: Vec<Cand> = Vec::with_capacity(self.hyps.len() * (size + 1));
                for (src, h) in self.hyps.iter_mut().enumerate() {
                    process(sp, &mut h.logits, &h.tokens, job.rules);
                    let lp = log_softmax(&h.logits);
                    for tok in top_k(&lp, size + 1) {
                        cands.push(Cand {
                            score: h.sum + lp[tok],
                            src,
                            tok: tok as u32,
                            lp: lp[tok],
                        });
                    }
                }
                self.steps += self.hyps.len();
                let scores: Vec<(f32, bool)> = cands.iter().map(|c| (c.score, c.tok == sp.eot)).collect();
                let (next, ended) = beam_select(&scores, size);
                for &i in &ended {
                    if self.finished.len() >= max_candidates {
                        break;
                    }
                    let c = &cands[i];
                    let h = &self.hyps[c.src];
                    self.finished.push(Done {
                        tokens: h.tokens.iter().copied().chain([c.tok]).collect(),
                        logprobs: h.logprobs.iter().copied().chain([c.lp]).collect(),
                        sum: c.score,
                    });
                }
                // The new beams take over their source's cache: moved for its last use, copied
                // before.
                let mut uses = vec![0usize; self.hyps.len()];
                for &i in &next {
                    uses[cands[i].src] += 1;
                }
                let mut old: Vec<Option<Hyp>> = std::mem::take(&mut self.hyps).into_iter().map(Some).collect();
                for &i in &next {
                    let c = &cands[i];
                    uses[c.src] -= 1;
                    let mut h = if uses[c.src] == 0 {
                        old[c.src].take().unwrap()
                    } else {
                        let o = old[c.src].as_ref().unwrap();
                        Hyp {
                            st: o.st.clone(),
                            tokens: o.tokens.clone(),
                            logprobs: o.logprobs.clone(),
                            sum: o.sum,
                            logits: vec![0f32; o.logits.len()],
                            sampler: None,
                            live: true,
                        }
                    };
                    h.tokens.push(c.tok);
                    h.logprobs.push(c.lp);
                    h.sum = c.score;
                    self.hyps.push(h);
                }
                let full = self.hyps.first().is_none_or(|h| h.st.pos >= max_pos);
                self.complete = self.finished.len() >= max_candidates || self.iters >= job.max_new || full;
            }
        }
    }

    /// The best ranked hypothesis.
    fn finalize(mut self, eot: u32) -> Done {
        let job = self.job;
        if let Search::Beam { size, .. } = job.search {
            // Not enough ended: the most likely beams end where they are.
            if self.finished.len() < size {
                let mut order: Vec<usize> = (0..self.hyps.len()).collect();
                order.sort_by(|&a, &b| self.hyps[b].sum.partial_cmp(&self.hyps[a].sum).unwrap_or(std::cmp::Ordering::Equal));
                for i in order {
                    let h = &self.hyps[i];
                    self.finished.push(Done {
                        tokens: h.tokens.iter().copied().chain([eot]).collect(),
                        logprobs: h.logprobs.iter().copied().chain([0.0]).collect(),
                        sum: h.sum,
                    });
                    if self.finished.len() >= size {
                        break;
                    }
                }
            }
        } else {
            self.finished = self
                .hyps
                .drain(..)
                .map(|h| Done {
                    tokens: h.tokens,
                    logprobs: h.logprobs,
                    sum: h.sum,
                })
                .collect();
        }
        if self.finished.len() == 1 {
            return self.finished.pop().unwrap();
        }
        let scored: Vec<(f32, usize)> = self
            .finished
            .iter()
            .map(|f| (f.sum, f.tokens.iter().position(|&t| t == eot).unwrap_or(f.tokens.len())))
            .collect();
        let best = rank(&scored, job.length_penalty);
        self.finished.swap_remove(best)
    }
}

/// A continuation of beam `src` by `tok`.
struct Cand {
    /// The beam's log-probability with the token's.
    score: f32,
    src: usize,
    tok: u32,
    lp: f32,
}

/// The indices of the `k` largest values, largest first (the lower index first among equals).
fn top_k(x: &[f32], k: usize) -> Vec<usize> {
    let mut best: Vec<usize> = Vec::with_capacity(k + 1);
    for (i, &v) in x.iter().enumerate() {
        if best.len() == k && v.partial_cmp(&x[best[k - 1]]) != Some(std::cmp::Ordering::Greater) {
            continue;
        }
        let at = best.iter().position(|&j| v > x[j]).unwrap_or(best.len());
        best.insert(at, i);
        best.truncate(k);
    }
    best
}

/// One update of openai-whisper's beam search over the candidates `(score, ends)` listed beam by
/// beam: walks them by decreasing score (equal scores in listing order) until `size` of them
/// that go on are kept. Returns those, and the ones that end met on the way, both best first.
fn beam_select(cands: &[(f32, bool)], size: usize) -> (Vec<usize>, Vec<usize>) {
    let mut order: Vec<usize> = (0..cands.len()).collect();
    order.sort_by(|&a, &b| cands[b].0.partial_cmp(&cands[a].0).unwrap_or(std::cmp::Ordering::Equal));
    let (mut next, mut ended) = (Vec::with_capacity(size), Vec::new());
    for i in order {
        if cands[i].1 {
            ended.push(i);
        } else {
            next.push(i);
            if next.len() == size {
                break;
            }
        }
    }
    (next, ended)
}

/// openai-whisper's `MaximumLikelihoodRanker`: the index of the hypothesis `(log-probability,
/// length without <|endoftext|>)` with the highest log-probability over its length, or over the
/// length penalty `((5 + length) / 6)^a`. The first one wins among equals.
fn rank(hyps: &[(f32, usize)], length_penalty: Option<f32>) -> usize {
    let mut best = (0usize, f64::NEG_INFINITY);
    for (i, &(sum, len)) in hyps.iter().enumerate() {
        let penalty = match length_penalty {
            None => len as f64,
            Some(a) => ((5.0 + len as f64) / 6.0).powf(a as f64),
        };
        // An empty hypothesis has no score under the plain length rule.
        let score = if penalty > 0.0 { sum as f64 / penalty } else { f64::NEG_INFINITY };
        if score > best.1 {
            best = (i, score);
        }
    }
    best.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_orders_by_value_then_index() {
        let x = [0.5, f32::NEG_INFINITY, 2.0, 0.5, -1.0, 2.0];
        assert_eq!(top_k(&x, 3), vec![2, 5, 0]);
        assert_eq!(top_k(&x, 5), vec![2, 5, 0, 3, 4]);
        // Masked tokens fill what is left, as torch.topk returns them.
        assert_eq!(top_k(&x, 6), vec![2, 5, 0, 3, 4, 1]);
        assert_eq!(top_k(&[f32::NEG_INFINITY; 3], 2), vec![0, 1]);
    }

    #[test]
    fn beam_update_like_openai() {
        // Two beams, three candidates each (beam_size 2): (score, ends).
        // Beam 0: -0.1 goes on, -0.3 ends, -2.0 goes on; beam 1: -0.2 ends, -0.4 goes on, -3.0.
        let c = [
            (-0.1, false),
            (-0.3, true),
            (-2.0, false),
            (-0.2, true),
            (-0.4, false),
            (-3.0, false),
        ];
        let (next, ended) = beam_select(&c, 2);
        // By score: 0 (on), 3 (ends), 1 (ends), 4 (on): two kept, the walk stops there.
        assert_eq!(next, vec![0, 4]);
        assert_eq!(ended, vec![3, 1]);
        // An ending candidate ranked below the last kept one is not seen.
        let c = [(-0.1, false), (-0.2, false), (-0.3, true)];
        assert_eq!(beam_select(&c, 2), (vec![0, 1], vec![]));
        // Equal scores keep the listing order (Python's stable sort).
        let c = [(-1.0, false), (-1.0, false), (-1.0, false)];
        assert_eq!(beam_select(&c, 2).0, vec![0, 1]);
        // Impossible continuations (-inf) are taken when nothing else is left.
        let c = [(-0.5, true), (f32::NEG_INFINITY, false), (-0.7, false)];
        assert_eq!(beam_select(&c, 2), (vec![2, 1], vec![0]));
    }

    #[test]
    fn ranking_like_openai() {
        // Plain length normalization: -4 / 4 beats -3 / 2.
        assert_eq!(rank(&[(-3.0, 2), (-4.0, 4)], None), 1);
        // Google NMT penalty with alpha 1: -3 / (7 / 6) = -2.57 against -4 / (9 / 6) = -2.67.
        assert_eq!(rank(&[(-3.0, 2), (-4.0, 4)], Some(1.0)), 0);
        // Alpha 0: the raw log-probability.
        assert_eq!(rank(&[(-3.0, 2), (-4.0, 4), (-3.0, 9)], Some(0.0)), 0);
        // The first of equals; an empty hypothesis never wins the plain rule.
        assert_eq!(rank(&[(-2.0, 2), (-2.0, 2)], None), 0);
        assert_eq!(rank(&[(-0.1, 0), (-5.0, 3)], None), 1);
    }
}
