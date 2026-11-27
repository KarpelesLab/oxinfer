//! Text generation building blocks shared by the CLI and [`crate::engine`]: the common surface of
//! the model families ([`TextLm`]), chunked prefill, the decoding loops (plain sampling, Qwen MTP
//! and DSpark rounds, Gemma 4 drafter speculative decoding) with a per-token callback that can
//! stop them, and incremental detokenization for streaming ([`TextStream`]).
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
use crate::sampler::{DeviceSampling, Sampler, argmax};
use crate::tokenizer::Tokenizer;

/// Tokens per prefill chunk on the CPU (the CLI's and the engine's default without a GPU
/// backend). Each matmul's weight rows are unpacked once per chunk and a mixture of experts'
/// experts get twice the tokens of 256: on an M5 Max, pp512 / pp2048 +5-10% against 256 (Qwen3-8B
/// Q8_0, Gemma 4 12B and 26B-A4B, Llama 3.2 1B), 1024 no faster.
pub const CPU_PREFILL_CHUNK: usize = 512;

/// Common surface of the model families (Gemma 4 / Gemma 3, Qwen 3.5/3.8 and Qwen3) used by the
/// generation loops.
pub trait TextLm: Sync {
    /// (Sent: a batched step of [`crate::batch`] may run a session on another request's thread.)
    type Sess: Send;
    fn new_session(&self) -> Self::Sess;
    fn hidden_size(&self) -> usize;
    /// Decoder layers.
    fn n_layers(&self) -> usize;
    /// The session's intervention state ([`crate::intervene`]), when the family supports it.
    fn iv_state<'s>(&self, _sess: &'s mut Self::Sess) -> Option<&'s mut crate::intervene::State> {
        None
    }
    /// Tokens in the session's cache (the position of the next token).
    fn session_len(&self, sess: &Self::Sess) -> usize;
    /// Runs `tokens` through the model, with the embeddings of some positions replaced by
    /// multimodal soft tokens; returns the final-normed hidden states `[tokens, hidden]`.
    fn forward_hidden_mm(&self, sess: &mut Self::Sess, tokens: &[u32], mm: &[(usize, &[f32])]) -> Vec<f32>;
    /// [`Self::forward_hidden_mm`] for a prompt chunk whose hidden states nothing reads but the
    /// last token's (`last`), or none: a model may then skip work only those rows need (Gemma's
    /// KV-sharing layers) and return the last token's alone (or, without `last`, anything).
    fn prefill_hidden(&self, sess: &mut Self::Sess, tokens: &[u32], mm: &[(usize, &[f32])], _last: bool) -> Vec<f32> {
        self.forward_hidden_mm(sess, tokens, mm)
    }
    /// LM-head logits `[t, vocab]` for the last `t` hidden states given.
    fn logits(&self, hidden: &[f32], t: usize) -> Vec<f32>;
    /// Declares `gh` x `gw` vision tokens at sequence position `start` before they are fed
    /// (Qwen's M-RoPE positions); models without multimodal positions ignore it.
    fn add_vision_span(&self, _sess: &mut Self::Sess, _start: usize, _gh: usize, _gw: usize) {}
    /// The soft tokens of one image attend to each other both ways, so a forward pass must hold
    /// a whole run of them (prefill chunks are stretched to the end of a run).
    fn mm_runs_whole(&self) -> bool {
        false
    }
    /// The prefill chunk the model prefers over `chunk` (see `QwenModel::prefill_chunk`).
    fn prefill_chunk(&self, chunk: usize) -> usize {
        chunk
    }
    /// Runs `tokens` and returns the logits after the last one.
    fn forward(&self, sess: &mut Self::Sess, tokens: &[u32]) -> Vec<f32> {
        let h = self.hidden_size();
        let hs = self.forward_hidden_mm(sess, tokens, &[]);
        self.logits(&hs[hs.len() - h..], 1)
    }
    /// Greedy decoding with the passes chained on the device (see
    /// `Inkling2Model::decode_greedy`): runs `first` and the argmax tokens after it, at most
    /// `max` passes, `on` getting each kept pass's token in order ([`Chain`]). The passes kept
    /// (the session advanced by them), or None (nothing run) when the model does not do it.
    fn decode_greedy(&self, _sess: &mut Self::Sess, _first: u32, _max: usize, _on: &mut dyn FnMut(u32) -> Chain) -> Option<usize> {
        None
    }
    /// [`TextLm::decode_greedy`] for the sampling `samp` ([`Sampler::device_sampling`]: greedy
    /// or a temperature draw, with top-k, top-p, min-p and a repetition penalty): each pass's
    /// token chosen on the device as [`DeviceSampling::pick`] chooses it, from the uniform
    /// `draw(j)` returns for the chain's pass `j` (asked for in order as passes are queued; a
    /// pass queued past the last kept one draws too, and a pass run again draws its index
    /// again).
    fn decode_sampled(
        &self,
        _sess: &mut Self::Sess,
        _first: u32,
        _max: usize,
        _samp: &DeviceSampling,
        _draw: &mut dyn FnMut(usize) -> f32,
        _on: &mut dyn FnMut(u32) -> Chain,
    ) -> Option<usize> {
        None
    }
    /// One decode step of several independent sessions (`tokens[i]` appended to `sess[i]`):
    /// the logits after each, as [`Self::forward`] of the one token would give them. Families
    /// with a batched pass read the weights once for all of them ([`crate::batch`]); by default
    /// the sessions run one after the other. A batched pass is exact where the family's batched
    /// kernels are (each row the bits of its one-token step; see the implementations).
    fn forward_batch(&self, sess: &mut [&mut Self::Sess], tokens: &[u32]) -> Vec<Vec<f32>> {
        sess.iter_mut().zip(tokens).map(|(s, &t)| self.forward(s, &[t])).collect()
    }
}

/// Runs `run` (a chain of passes that choose their tokens on the device: [`TextLm::decode_sampled`]
/// and its kin) with the sampler's [`DeviceSampling`] and its generator's uniforms, then leaves the
/// sampler as one-token steps would have for the first `used()` tokens the chain gave `on`: the
/// generator advanced by their draws only, those tokens seen by the repetition penalty. None
/// (nothing run) without device sampling for this sampler (a guide) or when `run` returns None.
pub fn chain_sampled(
    sampler: &mut Sampler,
    on: &mut dyn FnMut(u32) -> Chain,
    used: &dyn Fn() -> usize,
    run: &mut ChainRun<'_>,
) -> Option<usize> {
    let ds = sampler.device_sampling()?;
    let st = sampler.state();
    let mut toks: Vec<u32> = Vec::new();
    let n = {
        let mut us: Vec<f32> = Vec::new();
        let s = &mut *sampler;
        let mut draw = |j: usize| -> f32 {
            while us.len() <= j {
                us.push(s.uniform());
            }
            us[j]
        };
        let mut on2 = |t: u32| -> Chain {
            toks.push(t);
            on(t)
        };
        run(&ds, &mut draw, &mut on2)
    };
    let u = if n.is_some() { used().min(toks.len()) } else { 0 };
    sampler.set_state(st, if ds.temperature > 0.0 { u } else { 0 });
    if ds.penalized() {
        sampler.observe(&toks[..u]);
    }
    n
}

/// A chain run by [`chain_sampled`]: given the device's sampling, the uniform of each pass and
/// the per-token callback, the passes kept (None: nothing run).
pub type ChainRun<'a> = dyn FnMut(&DeviceSampling, &mut dyn FnMut(usize) -> f32, &mut dyn FnMut(u32) -> Chain) -> Option<usize> + 'a;

/// Chained device passes for `sampler` ([`TextLm::decode_greedy`], or
/// [`TextLm::decode_sampled`] with uniforms from the sampler's generator); `used` says, after
/// the chain, how many of its tokens were taken: the generator is left as if it had drawn for
/// those only, as one-token steps would have. None (nothing run) when the model does not chain
/// for this sampler.
pub fn decode_chain<M: TextLm + ?Sized>(
    model: &M,
    sess: &mut M::Sess,
    sampler: &mut Sampler,
    first: u32,
    max: usize,
    on: &mut dyn FnMut(u32) -> Chain,
    used: &dyn Fn() -> usize,
) -> Option<usize> {
    if sampler.plain_greedy() {
        return model.decode_greedy(sess, first, max, on);
    }
    chain_sampled(sampler, on, used, &mut |ds, draw, on| {
        model.decode_sampled(sess, first, max, ds, draw, on)
    })
}

/// What a chain of greedy passes on the device does after a pass's token ([`TextLm::decode_greedy`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Chain {
    /// Run the token, and go on.
    Go,
    /// Do not run the token (a pass already queued for it is undone).
    Stop,
    /// Queue no more passes: the passes already queued run (their tokens still come).
    Last,
}

impl TextLm for Model<'_> {
    type Sess = Session;
    fn new_session(&self) -> Session {
        Model::new_session(self)
    }
    fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }
    fn n_layers(&self) -> usize {
        self.cfg.num_hidden_layers
    }
    fn iv_state<'s>(&self, sess: &'s mut Session) -> Option<&'s mut crate::intervene::State> {
        Some(&mut sess.iv)
    }
    fn session_len(&self, sess: &Session) -> usize {
        sess.pos
    }
    fn forward_hidden_mm(&self, sess: &mut Session, tokens: &[u32], mm: &[(usize, &[f32])]) -> Vec<f32> {
        Model::forward_hidden_mm(self, sess, tokens, mm)
    }
    fn prefill_hidden(&self, sess: &mut Session, tokens: &[u32], mm: &[(usize, &[f32])], last: bool) -> Vec<f32> {
        Model::prefill_hidden(self, sess, tokens, mm, last)
    }
    fn logits(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        Model::logits(self, hidden, t)
    }
    fn mm_runs_whole(&self) -> bool {
        self.cfg.vision_bidirectional()
    }
    fn forward(&self, sess: &mut Session, tokens: &[u32]) -> Vec<f32> {
        Model::forward(self, sess, tokens)
    }
    fn decode_greedy(&self, sess: &mut Session, first: u32, max: usize, on: &mut dyn FnMut(u32) -> Chain) -> Option<usize> {
        Model::decode_greedy(self, sess, first, max, on)
    }
    fn forward_batch(&self, sess: &mut [&mut Session], tokens: &[u32]) -> Vec<Vec<f32>> {
        Model::forward_batch(self, sess, tokens)
    }
    fn prefill_chunk(&self, chunk: usize) -> usize {
        Model::prefill_chunk(self, chunk)
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
    fn n_layers(&self) -> usize {
        self.cfg.num_hidden_layers
    }
    fn iv_state<'s>(&self, sess: &'s mut QwenSession) -> Option<&'s mut crate::intervene::State> {
        Some(&mut sess.iv)
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
    fn prefill_chunk(&self, chunk: usize) -> usize {
        QwenModel::prefill_chunk(self, chunk)
    }
    fn prefill_hidden(&self, sess: &mut QwenSession, tokens: &[u32], mm: &[(usize, &[f32])], _last: bool) -> Vec<f32> {
        QwenModel::prefill_hidden(self, sess, tokens, mm)
    }
    fn forward(&self, sess: &mut QwenSession, tokens: &[u32]) -> Vec<f32> {
        QwenModel::forward(self, sess, tokens)
    }
    fn decode_greedy(&self, sess: &mut QwenSession, first: u32, max: usize, on: &mut dyn FnMut(u32) -> Chain) -> Option<usize> {
        QwenModel::decode_greedy(self, sess, first, max, on)
    }
    fn forward_batch(&self, sess: &mut [&mut QwenSession], tokens: &[u32]) -> Vec<Vec<f32>> {
        QwenModel::forward_batch(self, sess, tokens)
    }
}

type PrefillHook<'f> = dyn FnMut(usize, usize) + 'f;

thread_local! {
    static PREFILL_HOOK: std::cell::Cell<Option<*mut PrefillHook<'static>>> = const { std::cell::Cell::new(None) };
}

/// Runs `f` on this thread with `hook` receiving the progress of every [`prefill`] of more than
/// one token inside it: `(tokens done, tokens of the prompt)`, before the first chunk and after
/// each one. For callers that want to show something while a long prompt is read on a slow
/// device (the server's streamed responses).
pub fn with_prefill_progress<T>(hook: &mut PrefillHook<'_>, f: impl FnOnce() -> T) -> T {
    struct Restore(Option<*mut PrefillHook<'static>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            PREFILL_HOOK.with(|h| h.set(self.0));
        }
    }
    let p: *mut PrefillHook<'_> = hook;
    // SAFETY: the pointer is only dereferenced on this thread while `hook` is borrowed: the
    // guard removes it before this returns, also when unwinding.
    let p: *mut PrefillHook<'static> = unsafe { std::mem::transmute(p) };
    let _restore = Restore(PREFILL_HOOK.with(|h| h.replace(Some(p))));
    f()
}

fn report_prefill(done: usize, total: usize) {
    // Taken while called, so the hook is never re-entered.
    if let Some(h) = PREFILL_HOOK.with(|c| c.take()) {
        // SAFETY: see `with_prefill_progress`.
        unsafe { (*h)(done, total) };
        PREFILL_HOOK.with(|c| c.set(Some(h)));
    }
}

/// Prefills `ids` in chunks of `chunk` tokens with multimodal overrides (`(position in ids,
/// embedding)`); returns the hidden states of the last chunk (at least its last token's: models
/// may skip work only the others' need, see [`TextLm::prefill_hidden`]). A chunk ending inside a run of
/// consecutive overrides is stretched to the run's end when the model needs whole runs
/// ([`TextLm::mm_runs_whole`]).
pub fn prefill<M: TextLm + ?Sized>(model: &M, sess: &mut M::Sess, ids: &[u32], chunk: usize, overrides: &[(usize, Vec<f32>)]) -> Vec<f32> {
    crate::kernels::lm_prefill(|| prefill_inner(model, sess, ids, chunk, overrides))
}

thread_local! {
    /// Prompt tokens still to come after the chunk [`prefill`] is running on this thread.
    static PROMPT_AHEAD: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Prompt tokens [`prefill`] will feed after the chunk it is running (0 outside it): a model
/// whose chunks append to caches takes room for all of them at the first chunk, instead of
/// growing (and moving) the caches every few chunks of a long prompt.
pub(crate) fn prompt_ahead() -> usize {
    PROMPT_AHEAD.with(|c| c.get())
}

fn prefill_inner<M: TextLm + ?Sized>(
    model: &M,
    sess: &mut M::Sess,
    ids: &[u32],
    chunk: usize,
    overrides: &[(usize, Vec<f32>)],
) -> Vec<f32> {
    // Interventions limited to prompt / generated positions: the last prompt token is the first
    // "generated" position (its output is sampled).
    let base = model.session_len(sess);
    if let Some(st) = model.iv_state(sess) {
        st.gen_from = base + ids.len().saturating_sub(1);
    }
    let mut hs = Vec::new();
    let mut start = 0;
    let chunk = model.prefill_chunk(chunk);
    let whole = model.mm_runs_whole() && !overrides.is_empty();
    let mut mm_pos: Vec<usize> = if whole {
        overrides.iter().map(|o| o.0).collect()
    } else {
        Vec::new()
    };
    mm_pos.sort_unstable();
    let is_mm = |p: usize| mm_pos.binary_search(&p).is_ok();
    let report = ids.len() > 1;
    if report {
        report_prefill(0, ids.len());
    }
    while start < ids.len() {
        // Other requests on the model ([`crate::batch`]) go between the chunks, the shortest
        // prompt left first.
        let tw = Instant::now();
        if start > 0 {
            crate::batch::set_remaining(ids.len() - start);
            crate::batch::yield_now();
        }
        let waited = tw.elapsed();
        let mut end = (start + crate::batch::chunk_for(chunk).max(1)).min(ids.len());
        if whole {
            while end < ids.len() && is_mm(end - 1) && is_mm(end) {
                end += 1;
            }
        }
        let c = &ids[start..end];
        let local: Vec<(usize, &[f32])> = overrides
            .iter()
            .filter(|(p, _)| *p >= start && *p < end)
            .map(|(p, e)| (p - start, e.as_slice()))
            .collect();
        let tr = Instant::now();
        PROMPT_AHEAD.with(|a| a.set(ids.len() - end));
        hs = model.prefill_hidden(sess, c, &local, end == ids.len());
        PROMPT_AHEAD.with(|a| a.set(0));
        if crate::batch::trace() {
            eprintln!(
                "[prompt chunk: {} tokens at {start} of {}, waited {:.2} ms, ran in {:.2} ms]",
                c.len(),
                ids.len(),
                waited.as_secs_f64() * 1e3,
                tr.elapsed().as_secs_f64() * 1e3
            );
        }
        start = end;
        if start == ids.len() {
            crate::batch::set_remaining(0);
        }
        if report {
            report_prefill(end, ids.len());
        }
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
pub fn decode<M: TextLm>(
    model: &M,
    sess: &mut M::Sess,
    logits: Vec<f32>,
    sampler: &mut Sampler,
    stop: &[u32],
    max_tokens: usize,
    on_token: &mut dyn FnMut(&[u32]) -> Flow,
) -> Decoded {
    decode_from(model, sess, Vec::new(), Start::Logits(logits), sampler, stop, max_tokens, on_token)
}

/// Where [`decode_from`] starts.
pub enum Start {
    /// The logits to sample the next token from.
    Logits(Vec<f32>),
    /// A token to run first (the last of the reply so far, which the model has not run).
    Run(u32),
}

/// [`decode`] continuing a reply: `out` holds its tokens so far (`on_token` gets them all, the
/// new ones after them; `max_tokens` counts them too). Speculative rounds hand over to it when
/// other requests start decoding on the model ([`decode_rounds_until`]).
#[allow(clippy::too_many_arguments)]
pub fn decode_from<M: TextLm>(
    model: &M,
    sess: &mut M::Sess,
    mut out: Vec<u32>,
    start: Start,
    sampler: &mut Sampler,
    stop: &[u32],
    max_tokens: usize,
    on_token: &mut dyn FnMut(&[u32]) -> Flow,
) -> Decoded {
    let t0 = Instant::now();
    let mut forwards = 0usize;
    let mut reason = StopReason::Length;
    let mut unfed = None;
    // With other requests on the model, the steps join theirs ([`crate::batch::decode_step`]).
    let _batched = crate::batch::decoding();
    let mut logits = match start {
        Start::Logits(l) => l,
        Start::Run(t) => {
            forwards += 1;
            let _decode = crate::pool::decode_scope();
            crate::batch::decode_step(model, sess, t)
        }
    };
    while out.len() < max_tokens {
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
        // The token to run next.
        let mut run = id;
        // Decoding the device chains (the rest of the reply in one call), alone on the model:
        // the chain ends when another request comes, its last token not run.
        if sampler.chainable() && !crate::batch::contended() {
            let n0 = out.len();
            let budget = max_tokens - (n0 - 1);
            let taken = std::cell::Cell::new(0usize);
            let mut stopped = None;
            let mut yielded = false;
            let mut on = |t: u32| -> Chain {
                if out.len() >= max_tokens {
                    // (The last pass: its token is not one of the reply's.)
                    return Chain::Stop;
                }
                out.push(t);
                taken.set(out.len() - n0);
                if stop.contains(&t) {
                    stopped = Some((StopReason::Stop, None));
                } else if on_token(&out) == Flow::Stop {
                    stopped = Some((StopReason::Cancelled, Some(t)));
                } else if crate::batch::contended() {
                    yielded = true;
                }
                if stopped.is_none() && !yielded { Chain::Go } else { Chain::Stop }
            };
            if let Some(n) = decode_chain(model, sess, sampler, id, budget, &mut on, &|| taken.get()) {
                forwards += n;
                if let Some((r, u)) = stopped {
                    (reason, unfed) = (r, u);
                    break;
                }
                if !yielded {
                    break;
                }
                run = *out.last().unwrap();
            }
        }
        // A one-token step: on fewer threads where that leaves cores to the rest of the system.
        let _decode = crate::pool::decode_scope();
        logits = crate::batch::decode_step(model, sess, run);
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

/// A speculative decoder run in verification rounds (Qwen MTP, DSpark): `push` commits the
/// first token of a reply, `step` drafts, verifies and returns the committed tokens.
pub trait RoundDecoder {
    fn push(&mut self, token: u32);
    fn step(&mut self, sampler: &mut Sampler, max_new: usize) -> Vec<u32>;
    /// [`RoundDecoder::step`] that hands each committed token to `emit` as soon as it has it
    /// (a decoder that commits many tokens in one step streams them); `emit` returns false to
    /// stop there (a stop token, or the caller's cancel). By default after the step.
    fn step_emit(&mut self, sampler: &mut Sampler, max_new: usize, emit: &mut dyn FnMut(u32) -> bool) -> Vec<u32> {
        let out = self.step(sampler, max_new);
        for &id in &out {
            emit(id);
        }
        out
    }
    /// Verification rounds run so far.
    fn rounds(&self) -> usize;
}

impl RoundDecoder for MtpDecoder<'_> {
    fn push(&mut self, token: u32) {
        MtpDecoder::push(self, token)
    }
    fn step(&mut self, sampler: &mut Sampler, max_new: usize) -> Vec<u32> {
        MtpDecoder::step(self, sampler, max_new)
    }
    fn rounds(&self) -> usize {
        self.stats.rounds
    }
}

/// Qwen MTP speculative decoding after `dec.prefill(..)` returned `logits` (see
/// [`decode_rounds`]).
pub fn decode_mtp(
    dec: &mut MtpDecoder,
    logits: &[f32],
    sampler: &mut Sampler,
    stop: &[u32],
    max_tokens: usize,
    on_token: &mut dyn FnMut(&[u32]) -> Flow,
) -> Decoded {
    decode_rounds(dec, logits, sampler, stop, max_tokens, on_token)
}

/// Speculative decoding in rounds after the decoder's prefill returned `logits`: samples the
/// first token, then runs verification rounds until a stop token, `max_tokens` tokens (at least
/// one is always generated) or `on_token` stops it (the tokens of the current round are still
/// committed and returned, without further callbacks). `forwards` counts verification rounds.
pub fn decode_rounds<D: RoundDecoder + ?Sized>(
    dec: &mut D,
    logits: &[f32],
    sampler: &mut Sampler,
    stop: &[u32],
    max_tokens: usize,
    on_token: &mut dyn FnMut(&[u32]) -> Flow,
) -> Decoded {
    decode_rounds_until(dec, logits, sampler, stop, max_tokens, on_token, &|| false).0
}

/// [`decode_rounds`] that ends between two rounds once `yield_if` says so (true with the
/// result: its last token was committed but not run, [`Decoded::unfed`]), for the caller to go
/// on with plain steps ([`decode_from`] with [`Start::Run`]): what speculative decoders do when
/// other requests start decoding on the model ([`crate::batch::contended`]), so that their steps
/// join the batched ones.
#[allow(clippy::too_many_arguments)]
pub fn decode_rounds_until<D: RoundDecoder + ?Sized>(
    dec: &mut D,
    logits: &[f32],
    sampler: &mut Sampler,
    stop: &[u32],
    max_tokens: usize,
    on_token: &mut dyn FnMut(&[u32]) -> Flow,
    yield_if: &dyn Fn() -> bool,
) -> (Decoded, bool) {
    let mut yielded = false;
    let t0 = Instant::now();
    let rounds0 = dec.rounds();
    let first = sampler.sample(logits);
    let mut out = vec![first];
    dec.push(first);
    let mut cancelled = !stop.contains(&first) && on_token(&out) == Flow::Stop;
    while !cancelled && !stop.contains(out.last().unwrap()) && out.len() < max_tokens {
        if yield_if() {
            yielded = true;
            break;
        }
        // Other requests on the model ([`crate::batch`]) go between the rounds.
        crate::batch::yield_now();
        dec.step_emit(sampler, max_tokens - out.len(), &mut |id| {
            out.push(id);
            if !cancelled && !stop.contains(&id) && on_token(&out) == Flow::Stop {
                cancelled = true;
            }
            !cancelled && !stop.contains(&id)
        });
    }
    let last = *out.last().unwrap();
    let reason = if stop.contains(&last) {
        StopReason::Stop
    } else if cancelled {
        StopReason::Cancelled
    } else {
        StopReason::Length
    };
    (
        Decoded {
            tokens: out,
            forwards: dec.rounds() - rounds0,
            secs: t0.elapsed().as_secs_f64(),
            reason,
            // The decoder never runs the last committed token until the next round or prefill.
            unfed: (reason != StopReason::Stop).then_some(last),
        },
        yielded,
    )
}

impl Decoded {
    /// This reply's first part followed by `rest`, which went on from it ([`decode_from`]).
    pub fn then(self, rest: Decoded) -> Decoded {
        Decoded {
            forwards: self.forwards + rest.forwards,
            secs: self.secs + rest.secs,
            ..rest
        }
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
/// forward the caller already ran. `k` is the maximum draft length: with `fixed` every round
/// drafts it, else each round drafts the length (0 = a plain step) with the most expected tokens
/// per second, starting from 3 ([`crate::spec_cost::ChainPolicy`]). The cache
/// may hold accepted drafts past the last returned token when the limit or the callback ended the
/// loop inside a round: truncate it to the conversation if it continues.
#[allow(clippy::too_many_arguments)]
pub fn decode_speculative(
    model: &Model,
    drafter: &Drafter,
    sess: &mut Session,
    hidden_last: Vec<f32>,
    k: usize,
    fixed: bool,
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

    // The draft length policy: the verification passes run the exact kernels.
    let k_max = k.max(1);
    let mut policy = crate::spec_cost::ChainPolicy::new(crate::spec_cost::CostPrior::detect(false));
    'outer: while !cancelled && !eos.contains(&last) && out.len() < max_tokens {
        // Draft k tokens from the drafter, chaining its projected hidden state.
        let pos = sess.pos; // index of `last` (not yet in the cache)
        let k_cur = if fixed { k_max } else { policy.choose(k_max, 3) };
        let k = k_cur.min(max_tokens - out.len() - 1);
        let t_round = Instant::now();
        let mut drafts = Vec::with_capacity(k);
        let mut hid = h_prev.clone();
        let mut cur = last;
        for _ in 0..k {
            let _t = prof::start(prof::Cat::Other);
            let emb = model.embed_scaled(cur);
            let (d, nh) = drafter.step(&emb, &hid, pos, sess.cache(ls), sess.cache(lg));
            drafts.push(d);
            hid = nh;
            cur = d;
        }
        stats.drafted += drafts.len();
        let draft_secs = t_round.elapsed().as_secs_f64();
        if stats.forwards == 0 && std::env::var("OXINFER_DEBUG_DRAFT").is_ok() {
            eprintln!("[debug: pos {pos} last {last} drafts {drafts:?}]");
        }

        // Verify: one main forward over [last, drafts...].
        let mut seq = Vec::with_capacity(drafts.len() + 1);
        seq.push(last);
        seq.extend_from_slice(&drafts);
        let check = std::env::var("OXINFER_CHECK_SPEC").is_ok();
        let sess_before = if check { Some(sess.clone()) } else { None };
        let (hs, logits) = model.verify_hidden(sess, &seq);
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
        if k == k_cur {
            policy.record(k, n, draft_secs, t_round.elapsed().as_secs_f64() - draft_secs);
        }
        if std::env::var_os("OXINFER_DEBUG_DRAFT").is_some() {
            eprintln!(
                "[drafter: k {k} accepted {n}; ms: draft {:.1} verify {:.1}]",
                draft_secs * 1e3,
                (t_round.elapsed().as_secs_f64() - draft_secs) * 1e3
            );
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
    tags: Option<Tags>,
    /// The token that opens a message header and those that end it ([`ChatFormat::message_header`]).
    header: Option<(u32, Vec<u32>)>,
    /// Special tokens are part of the text ([`ChatFormat::keeps_special_tokens`]).
    keep_special: bool,
    marks: Option<Marks>,
}

/// The state of a format whose blocks are marked in plain text ([`crate::chat::TextMarks`]).
struct Marks {
    m: crate::chat::TextMarks,
    /// Bytes of the decoded reply read so far.
    raw: usize,
    called: bool,
    /// A thinking block opened or closed since the last [`TextStream::on_token`].
    flipped: bool,
    /// The line break after a mark is still to drop.
    skip_nl: bool,
}

/// The length of the longest suffix of `text` that starts `mark` without completing it.
fn partial_suffix(text: &str, mark: &str) -> usize {
    (1..mark.len().min(text.len() + 1))
        .rev()
        .find(|&i| text.is_char_boundary(text.len() - i) && mark.starts_with(&text[text.len() - i..]))
        .unwrap_or(0)
}

/// The token ids of a format whose blocks are tags of several tokens
/// ([`crate::chat::TagTokens`]), and the tag being read (whether it opens, its name's tokens).
struct Tags {
    open: u32,
    close: u32,
    sep: u32,
    think: Vec<u32>,
    calls: Vec<u32>,
    tag: Option<(bool, Vec<u32>)>,
    called: bool,
}

impl TextStream {
    pub fn new(fmt: &dyn ChatFormat, think: bool, tok: &Tokenizer) -> Self {
        let (soc, eoc) = fmt.think_tokens();
        let id = |t: &str| if t.is_empty() { None } else { tok.token_to_id(t) };
        let (soc, eoc) = (id(soc).unwrap_or(u32::MAX), id(eoc).unwrap_or(u32::MAX));
        let tags = fmt.tag_tokens().and_then(|t| {
            let name = |s: &str| tok.encode(s, false).ok().map(|e| e.get_ids().to_vec());
            Some(Tags {
                open: tok.token_to_id(t.open)?,
                close: tok.token_to_id(t.close)?,
                sep: tok.token_to_id(t.sep)?,
                think: name(t.think)?,
                calls: name(t.calls)?,
                tag: None,
                called: false,
            })
        });
        Self {
            printed: 0,
            soc,
            eoc,
            call: id(fmt.tool_call_marker()),
            thinking: fmt.starts_in_thinking(think),
            tags,
            header: fmt
                .message_header()
                .and_then(|(open, ends)| Some((tok.token_to_id(open)?, ends.iter().filter_map(|t| tok.token_to_id(t)).collect()))),
            keep_special: fmt.keeps_special_tokens(),
            marks: fmt.text_marks().map(|m| Marks {
                m,
                raw: 0,
                called: false,
                flipped: false,
                skip_nl: false,
            }),
        }
    }

    /// Whether the reply has opened its tool calls (formats with tags; the others open them
    /// with one token, [`ChatFormat::tool_call_marker`]).
    pub fn called(&self) -> bool {
        self.tags.as_ref().is_some_and(|t| t.called) || self.marks.as_ref().is_some_and(|m| m.called)
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
        // Marks in plain text are found in the decoded text ([`TextStream::delta`]): the change
        // is reported with the next token.
        if let Some(m) = &mut self.marks {
            return std::mem::take(&mut m.flipped);
        }
        if let Some(t) = &mut self.tags {
            if id == t.open || id == t.close {
                t.tag = Some((id == t.open, Vec::new()));
            } else if id == t.sep {
                if let Some((opens, name)) = t.tag.take() {
                    if name == t.think {
                        self.thinking = opens;
                        return true;
                    }
                    t.called |= opens && name == t.calls;
                }
            } else if let Some((_, name)) = &mut t.tag {
                name.push(id);
            }
            return false;
        }
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
        if self.marks.is_some() {
            return self.delta_marks(tok, out, final_);
        }
        // Thinking delimiters are rendered as dimming, never as text (some tokenizers keep them).
        let visible = match self.call.and_then(|c| out.iter().position(|&t| t == c)) {
            Some(i) => &out[..i],
            None => out,
        };
        let shown: Vec<u32> = match &self.tags {
            // Tags are not text, and nothing is shown from the tag that opens the tool calls.
            Some(t) => {
                let (mut shown, mut tag): (Vec<u32>, Option<(bool, Vec<u32>)>) = (Vec::new(), None);
                for &id in visible {
                    if id == t.open || id == t.close {
                        tag = Some((id == t.open, Vec::new()));
                    } else if id == t.sep && tag.is_some() {
                        if tag.take().is_some_and(|(opens, name)| opens && name == t.calls) {
                            break;
                        }
                    } else if let Some((_, name)) = &mut tag {
                        name.push(id);
                    } else {
                        shown.push(id);
                    }
                }
                shown
            }
            None => {
                // What a message header holds between its opening token and its content kind is a name.
                let mut in_header = false;
                visible
                    .iter()
                    .copied()
                    .filter(|&t| {
                        if let Some((open, ends)) = &self.header {
                            if t == *open {
                                in_header = true;
                            } else if in_header && !ends.contains(&t) {
                                return false;
                            } else {
                                in_header = false;
                            }
                        }
                        t != self.soc && t != self.eoc
                    })
                    .collect()
            }
        };
        if let Ok(text) = tok.decode(&shown, !self.keep_special) {
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

    /// [`TextStream::delta`] for marks in plain text: the marks are not shown, nor the line break
    /// that leads to the closing one or to the tool calls, nor the one after a thinking mark;
    /// nothing is shown from the tool calls on. The text stops at a thinking mark, which is
    /// applied by the next call (so that [`TextStream::thinking`] says what the returned text
    /// is), and the start of a possible mark is held back until `final_`.
    fn delta_marks(&mut self, tok: &Tokenizer, out: &[u32], final_: bool) -> String {
        let Ok(text) = tok.decode(out, true) else {
            return String::new();
        };
        let stable = if !final_ && text.ends_with('\u{FFFD}') {
            text.len().saturating_sub(3)
        } else {
            text.len()
        };
        let m = self.marks.as_mut().unwrap();
        let mut shown = String::new();
        while !m.called && m.raw < stable && text.is_char_boundary(stable) {
            let rest = &text[m.raw..stable];
            if m.skip_nl {
                m.skip_nl = false;
                m.raw += usize::from(rest.starts_with('\n'));
                continue;
            }
            let toggle = if self.thinking { m.m.think_close } else { m.m.think_open };
            let at = |mark: &str| rest.find(mark).filter(|_| !mark.is_empty());
            // (Calls are not looked for inside a thinking block.)
            let call = if self.thinking { None } else { at(m.m.call) };
            let next = match (at(toggle), call) {
                (Some(t), Some(c)) if c < t => Some((c, true)),
                (Some(t), _) => Some((t, false)),
                (None, Some(c)) => Some((c, true)),
                (None, None) => None,
            };
            let Some((i, is_call)) = next else {
                let hold = if final_ {
                    0
                } else {
                    let lead = |mark: &str| partial_suffix(rest, &format!("\n{mark}")).max(partial_suffix(rest, mark));
                    lead(toggle).max(if self.thinking { 0 } else { lead(m.m.call) })
                };
                shown.push_str(&rest[..rest.len() - hold]);
                m.raw += rest.len() - hold;
                break;
            };
            let before = &rest[..i];
            let lead = self.thinking || is_call;
            shown.push_str(if lead {
                before.strip_suffix('\n').unwrap_or(before)
            } else {
                before
            });
            m.raw += i;
            if is_call {
                m.called = true;
            } else if shown.is_empty() || final_ {
                m.raw += toggle.len();
                self.thinking = !self.thinking;
                (m.flipped, m.skip_nl) = (true, true);
                if !final_ {
                    break;
                }
            } else {
                break;
            }
        }
        shown
    }
}
