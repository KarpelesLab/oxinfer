//! Text generation building blocks shared by the CLI and [`crate::engine`]: the common surface of
//! the model families ([`TextLm`]), chunked prefill, the decoding loops (plain sampling, Qwen MTP
//! and Gemma 4 drafter speculative decoding) with a per-token callback that can stop them, and
//! incremental detokenization for streaming ([`TextStream`]).
//!
//! The loops never print; the callback receives the tokens generated so far (the new one last)
//! for every token except a stop token, and returns [`Flow::Stop`] to end generation early.

use crate::chat::ChatFormat;
use crate::clock::Instant;
use crate::drafter::Drafter;
use crate::model::{Model, Session};
use crate::prof;
use crate::qwen::{QwenModel, QwenSession};
use crate::qwen_mtp::MtpDecoder;
use crate::sampler::{Sampler, argmax};
use tokenizers::Tokenizer;

/// Common surface of the model families (Gemma 4 / Gemma 3, Qwen 3.5/3.8 and Qwen3) used by the
/// generation loops.
pub trait TextLm {
    type Sess;
    fn new_session(&self) -> Self::Sess;
    fn hidden_size(&self) -> usize;
    /// Tokens in the session's cache (the position of the next token).
    fn session_len(&self, sess: &Self::Sess) -> usize;
    /// Runs `tokens` through the model, with the embeddings of some positions replaced by
    /// multimodal soft tokens; returns the final-normed hidden states `[tokens, hidden]`.
    fn forward_hidden_mm(&self, sess: &mut Self::Sess, tokens: &[u32], mm: &[(usize, &[f32])]) -> Vec<f32>;
    /// LM-head logits `[t, vocab]` for the last `t` hidden states given.
    fn logits(&self, hidden: &[f32], t: usize) -> Vec<f32>;
    /// Declares `gh` x `gw` vision tokens at sequence position `start` before they are fed
    /// (Qwen's M-RoPE positions); models without multimodal positions ignore it.
    fn add_vision_span(&self, _sess: &mut Self::Sess, _start: usize, _gh: usize, _gw: usize) {}
    /// Runs `tokens` and returns the logits after the last one.
    fn forward(&self, sess: &mut Self::Sess, tokens: &[u32]) -> Vec<f32> {
        let h = self.hidden_size();
        let hs = self.forward_hidden_mm(sess, tokens, &[]);
        self.logits(&hs[hs.len() - h..], 1)
    }
}

impl TextLm for Model<'_> {
    type Sess = Session;
    fn new_session(&self) -> Session {
        Model::new_session(self)
    }
    fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }
    fn session_len(&self, sess: &Session) -> usize {
        sess.pos
    }
    fn forward_hidden_mm(&self, sess: &mut Session, tokens: &[u32], mm: &[(usize, &[f32])]) -> Vec<f32> {
        Model::forward_hidden_mm(self, sess, tokens, mm)
    }
    fn logits(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        Model::logits(self, hidden, t)
    }
}

impl TextLm for QwenModel<'_> {
    type Sess = QwenSession;
    fn new_session(&self) -> QwenSession {
        QwenModel::new_session(self)
    }
    fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }
    fn session_len(&self, sess: &QwenSession) -> usize {
        sess.pos
    }
    fn forward_hidden_mm(&self, sess: &mut QwenSession, tokens: &[u32], mm: &[(usize, &[f32])]) -> Vec<f32> {
        QwenModel::forward_hidden_mm(self, sess, tokens, mm)
    }
    fn logits(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        QwenModel::logits(self, hidden, t)
    }
    fn add_vision_span(&self, sess: &mut QwenSession, start: usize, gh: usize, gw: usize) {
        sess.add_vision_span(start, gh, gw)
    }
}

/// Prefills `ids` in chunks of `chunk` tokens with multimodal overrides (`(position in ids,
/// embedding)`); returns the hidden states of the last chunk.
pub fn prefill<M: TextLm + ?Sized>(model: &M, sess: &mut M::Sess, ids: &[u32], chunk: usize, overrides: &[(usize, Vec<f32>)]) -> Vec<f32> {
    let mut hs = Vec::new();
    let mut start = 0;
    for c in ids.chunks(chunk.max(1)) {
        let end = start + c.len();
        let local: Vec<(usize, &[f32])> = overrides
            .iter()
            .filter(|(p, _)| *p >= start && *p < end)
            .map(|(p, e)| (p - start, e.as_slice()))
            .collect();
        hs = model.forward_hidden_mm(sess, c, &local);
        start = end;
    }
    hs
}

/// What a per-token callback wants next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flow {
    Continue,
    Stop,
}

/// Why a decoding loop ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// A stop token was generated (it is the last of `tokens`).
    Stop,
    /// The token limit was reached.
    Length,
    /// The callback returned [`Flow::Stop`].
    Cancelled,
}

/// The result of a decoding loop.
#[derive(Clone, Debug)]
pub struct Decoded {
    /// Generated tokens, including the stop token if one ended generation.
    pub tokens: Vec<u32>,
    /// Main-model forward passes run while decoding (one per token for plain decoding, one per
    /// verification round for speculative decoding).
    pub forwards: usize,
    /// Wall time of the loop.
    pub secs: f64,
    pub reason: StopReason,
    /// The last generated token when it was not run through the model and is not a stop token:
    /// a later turn must feed it first to keep the cache consistent with the conversation.
    pub unfed: Option<u32>,
}

/// Plain decoding: samples a token from `logits`, runs it, repeats. Stops after a stop token (not
/// run), `max_tokens` tokens (all run, as the CLI always has), or when `on_token` says so (the
/// last token is then not run: see [`Decoded::unfed`]).
#[allow(clippy::too_many_arguments)]
pub fn decode<M: TextLm + ?Sized>(
    model: &M,
    sess: &mut M::Sess,
    mut logits: Vec<f32>,
    sampler: &mut Sampler,
    stop: &[u32],
    max_tokens: usize,
    on_token: &mut dyn FnMut(&[u32]) -> Flow,
) -> Decoded {
    let t0 = Instant::now();
    let mut out: Vec<u32> = Vec::new();
    let mut forwards = 0usize;
    let mut reason = StopReason::Length;
    let mut unfed = None;
    for _ in 0..max_tokens {
        let id = sampler.sample(&logits);
        out.push(id);
        if stop.contains(&id) {
            reason = StopReason::Stop;
            break;
        }
        if on_token(&out) == Flow::Stop {
            reason = StopReason::Cancelled;
            unfed = Some(id);
            break;
        }
        logits = model.forward(sess, &[id]);
        forwards += 1;
    }
    Decoded {
        tokens: out,
        forwards,
        secs: t0.elapsed().as_secs_f64(),
        reason,
        unfed,
    }
}

/// Qwen MTP speculative decoding after `dec.prefill(..)` returned `logits`: samples the first
/// token, then runs verification rounds until a stop token, `max_tokens` tokens (at least one is
/// always generated) or `on_token` stops it (the tokens of the current round are still committed
/// and returned, without further callbacks). `forwards` counts verification rounds.
pub fn decode_mtp(
    dec: &mut MtpDecoder,
    logits: &[f32],
    sampler: &mut Sampler,
    stop: &[u32],
    max_tokens: usize,
    on_token: &mut dyn FnMut(&[u32]) -> Flow,
) -> Decoded {
    let t0 = Instant::now();
    let rounds0 = dec.stats.rounds;
    let first = sampler.sample(logits);
    let mut out = vec![first];
    dec.push(first);
    let mut cancelled = !stop.contains(&first) && on_token(&out) == Flow::Stop;
    while !cancelled && !stop.contains(out.last().unwrap()) && out.len() < max_tokens {
        for id in dec.step(sampler, max_tokens - out.len()) {
            out.push(id);
            if !cancelled && !stop.contains(&id) && on_token(&out) == Flow::Stop {
                cancelled = true;
            }
        }
    }
    let last = *out.last().unwrap();
    let reason = if stop.contains(&last) {
        StopReason::Stop
    } else if cancelled {
        StopReason::Cancelled
    } else {
        StopReason::Length
    };
    Decoded {
        tokens: out,
        forwards: dec.stats.rounds - rounds0,
        secs: t0.elapsed().as_secs_f64(),
        reason,
        // The decoder never runs the last committed token until the next round or prefill.
        unfed: (reason != StopReason::Stop).then_some(last),
    }
}

/// Statistics of Gemma 4 drafter speculative decoding.
#[derive(Clone, Debug, Default)]
pub struct SpecStats {
    pub forwards: usize,
    pub drafted: usize,
    pub accepted: usize,
    /// Histogram of accepted-draft counts per round.
    pub per_pos: [usize; 16],
}

/// Greedy speculative decoding with the Gemma 4 drafter. Produces exactly the plain greedy
/// sequence. `hidden_last` is the final-normed hidden state of the last prompt token, whose
/// forward the caller already ran. `k` is the maximum draft length (adaptive from 3). The cache
/// may hold accepted drafts past the last returned token when the limit or the callback ended the
/// loop inside a round: truncate it to the conversation if it continues.
#[allow(clippy::too_many_arguments)]
pub fn decode_speculative(
    model: &Model,
    drafter: &Drafter,
    sess: &mut Session,
    hidden_last: Vec<f32>,
    k: usize,
    eos: &[u32],
    max_tokens: usize,
    on_token: &mut dyn FnMut(&[u32]) -> Flow,
) -> (Decoded, SpecStats) {
    let t0 = Instant::now();
    let h = model.cfg.hidden_size;
    let (ls, lg) = model.cfg.shared_kv_layers();
    let mut stats = SpecStats::default();
    let mut out: Vec<u32> = Vec::new();
    let mut cancelled = false;

    // First token from the prompt's logits (the prefill forward already counted by the caller).
    let logits = model.logits(&hidden_last, 1);
    let mut last = argmax(&logits);
    let mut h_prev = hidden_last; // hidden state of the token that produced `last`
    out.push(last);
    if !eos.contains(&last) && on_token(&out) == Flow::Stop {
        cancelled = true;
    }

    // Adaptive draft length: grow after a fully accepted round, shrink after a fully rejected one.
    let (k_min, k_max) = (2usize.min(k.max(1)), k.max(1));
    let mut k_cur = k_min.max(3).min(k_max);
    'outer: while !cancelled && !eos.contains(&last) && out.len() < max_tokens {
        // Draft k tokens from the drafter, chaining its projected hidden state.
        let pos = sess.pos; // index of `last` (not yet in the cache)
        let k = k_cur.min(max_tokens - out.len());
        let mut drafts = Vec::with_capacity(k);
        let mut hid = h_prev.clone();
        let mut cur = last;
        for _ in 0..k.max(1) {
            let _t = prof::start(prof::Cat::Other);
            let emb = model.embed_scaled(cur);
            let (d, nh) = drafter.step(&emb, &hid, pos, sess.cache(ls), sess.cache(lg));
            drafts.push(d);
            hid = nh;
            cur = d;
        }
        stats.drafted += drafts.len();
        if stats.forwards == 0 && std::env::var("OXINFER_DEBUG_DRAFT").is_ok() {
            eprintln!("[debug: pos {pos} last {last} drafts {drafts:?}]");
        }

        // Verify: one main forward over [last, drafts...].
        let mut seq = Vec::with_capacity(drafts.len() + 1);
        seq.push(last);
        seq.extend_from_slice(&drafts);
        let check = std::env::var("OXINFER_CHECK_SPEC").is_ok();
        let sess_before = if check { Some(sess.clone()) } else { None };
        let hs = model.forward_hidden(sess, &seq);
        let logits = model.logits(&hs, seq.len());
        stats.forwards += 1;
        let v = model.cfg.vocab_size;
        if let Some(mut s2) = sess_before {
            // Debug: recompute every position through single-token forwards and compare argmaxes.
            for (i, &tk) in seq.iter().enumerate() {
                let l1 = model.forward(&mut s2, &[tk]);
                let lk = &logits[i * v..(i + 1) * v];
                let (a1, ak) = (argmax(&l1), argmax(lk));
                if a1 != ak {
                    eprintln!(
                        "[check: pos {} token {}: t=1 argmax {} ({:.4}) vs t={} argmax {} ({:.4}); cross: t1[{}]={:.4} tk[{}]={:.4}]",
                        pos + i,
                        tk,
                        a1,
                        l1[a1 as usize],
                        seq.len(),
                        ak,
                        lk[ak as usize],
                        ak,
                        l1[ak as usize],
                        a1,
                        lk[a1 as usize]
                    );
                }
            }
        }
        let mut n = 0;
        while n < drafts.len() && argmax(&logits[n * v..(n + 1) * v]) == drafts[n] {
            n += 1;
        }
        stats.accepted += n;
        if n < stats.per_pos.len() {
            stats.per_pos[n] += 1;
        }
        if n == drafts.len() {
            k_cur = (k_cur + 1).min(k_max);
        } else if n == 0 {
            k_cur = (k_cur - 1).max(k_min);
        }
        if stats.forwards == 1 && std::env::var("OXINFER_DEBUG_DRAFT").is_ok() {
            let truth: Vec<u32> = (0..seq.len()).map(|i| argmax(&logits[i * v..(i + 1) * v])).collect();
            eprintln!("[debug: main argmax after each of {seq:?} = {truth:?}; accepted {n}]");
        }
        for &d in &drafts[..n] {
            if out.len() >= max_tokens {
                sess.truncate(pos + n + 1);
                break 'outer;
            }
            out.push(d);
            if eos.contains(&d) {
                sess.truncate(pos + n);
                break 'outer;
            }
            if on_token(&out) == Flow::Stop {
                sess.truncate(pos + n + 1);
                cancelled = true;
                break 'outer;
            }
        }
        let bonus = argmax(&logits[n * v..(n + 1) * v]);
        // Keep only [last, drafts[..n]] in the cache; `bonus` is processed by the next round.
        sess.truncate(pos + n + 1);
        h_prev = hs[n * h..(n + 1) * h].to_vec();
        last = bonus;
        if out.len() >= max_tokens {
            break;
        }
        out.push(bonus);
        if !eos.contains(&bonus) && on_token(&out) == Flow::Stop {
            cancelled = true;
        }
    }
    let end = *out.last().unwrap();
    let reason = if eos.contains(&end) {
        StopReason::Stop
    } else if cancelled {
        StopReason::Cancelled
    } else {
        StopReason::Length
    };
    let d = Decoded {
        tokens: out,
        forwards: stats.forwards,
        secs: t0.elapsed().as_secs_f64(),
        reason,
        unfed: (reason != StopReason::Stop).then_some(end),
    };
    (d, stats)
}

/// Incremental detokenizer for streaming: turns the growing list of generated tokens into text
/// deltas, holding back an incomplete UTF-8 sequence, dropping the thinking delimiters and
/// stopping at the first tool call (the host runs the call; its text is not shown).
pub struct TextStream {
    printed: usize,
    soc: u32,
    eoc: u32,
    call: Option<u32>,
    thinking: bool,
}

impl TextStream {
    pub fn new(fmt: &dyn ChatFormat, think: bool, tok: &Tokenizer) -> Self {
        let (soc, eoc) = fmt.think_tokens();
        let id = |t: &str| if t.is_empty() { None } else { tok.token_to_id(t) };
        let (soc, eoc) = (id(soc).unwrap_or(u32::MAX), id(eoc).unwrap_or(u32::MAX));
        Self {
            printed: 0,
            soc,
            eoc,
            call: id(fmt.tool_call_marker()),
            thinking: fmt.starts_in_thinking(think),
        }
    }

    /// The token ids that open and close a thinking block (`u32::MAX` if none).
    pub fn think_tokens(&self) -> (u32, u32) {
        (self.soc, self.eoc)
    }

    /// Whether the text so far ends inside a thinking block.
    pub fn thinking(&self) -> bool {
        self.thinking
    }

    /// Tracks thinking blocks; call for each new token before [`TextStream::delta`]. Returns
    /// true if the token opened or closed a thinking block.
    pub fn on_token(&mut self, id: u32) -> bool {
        if id == self.soc {
            self.thinking = true;
        } else if id == self.eoc {
            self.thinking = false;
        } else {
            return false;
        }
        true
    }

    /// The new stable text of `out` since the last call (empty if none; `final_` flushes an
    /// incomplete character too). `out` must not include a stop token.
    pub fn delta(&mut self, tok: &Tokenizer, out: &[u32], final_: bool) -> String {
        // Thinking delimiters are rendered as dimming, never as text (some tokenizers keep them).
        let visible = match self.call.and_then(|c| out.iter().position(|&t| t == c)) {
            Some(i) => &out[..i],
            None => out,
        };
        let shown: Vec<u32> = visible.iter().copied().filter(|&t| t != self.soc && t != self.eoc).collect();
        if let Ok(text) = tok.decode(&shown, true) {
            let stable = if !final_ && text.ends_with('\u{FFFD}') {
                text.len().saturating_sub(3)
            } else {
                text.len()
            };
            if stable > self.printed && text.is_char_boundary(stable) {
                let from = std::mem::replace(&mut self.printed, stable);
                return text[from..stable].to_string();
            }
        }
        String::new()
    }
}
