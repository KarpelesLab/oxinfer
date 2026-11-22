//! A per-reply token hook ([`crate::engine::Engine::set_token_hook`]): code that sees each token
//! of a reply as it is committed and may make the model read tokens of its own next (forced
//! tokens), after which sampling resumes.
//!
//! The hook rides on the sampler's guide ([`crate::sampler::Sampler::set_guide`]), which every
//! decoding loop samples through: plain decoding, prompt lookup, MTP and DSpark rounds, on every
//! device. A forced token is the only one the guide lets the sampler choose at its position, so
//! it goes through the model like a sampled one (it is in the cache, the conversation's history,
//! the streamed deltas and the reply's tokens), and the tokens after it are conditioned on it.
//! Speculative rounds may draft forced tokens (the draft filter knows the queue) but the hook is
//! only told about committed tokens, never about drafts.

use crate::sampler::TokenFilter;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Sees each committed token of a reply ([`crate::engine::Engine::set_token_hook`]).
pub trait TokenHook: Send {
    /// Called once per committed token, with the reply's tokens so far (the new one last,
    /// forced tokens included). Returns the tokens the model must read next, in order (empty:
    /// none). Tokens returned while earlier forced tokens are still queued go after them.
    fn on_token(&mut self, reply: &[u32]) -> Vec<u32>;

    /// Called when a reply starts, before its first token is sampled: the tokens returned are
    /// the reply's first tokens. Default: none.
    fn on_start(&mut self) -> Vec<u32> {
        Vec::new()
    }
}

impl<F: FnMut(&[u32]) -> Vec<u32> + Send> TokenHook for F {
    fn on_token(&mut self, reply: &[u32]) -> Vec<u32> {
        self(reply)
    }
}

/// The hook shared between turns.
pub(crate) type SharedHook = Arc<Mutex<Box<dyn TokenHook>>>;

/// The sampler filter of one reply: the forced tokens queued by the hook, else `inner` (a
/// constraint, or the model's reply guard). Forced tokens are not given to `inner`.
pub(crate) struct HookFilter {
    /// None in a draft copy ([`TokenFilter::fork`]): drafts never reach the hook.
    hook: Option<SharedHook>,
    inner: Option<Box<dyn TokenFilter>>,
    queue: VecDeque<u32>,
    reply: Vec<u32>,
    /// Tokens whose id is past the logits (the hook's mistake): skipped.
    vocab: usize,
}

impl HookFilter {
    pub fn new(hook: SharedHook, inner: Option<Box<dyn TokenFilter>>, vocab: usize) -> Self {
        let queue = lock(&hook).on_start().into();
        HookFilter {
            hook: Some(hook),
            inner,
            queue,
            reply: Vec::new(),
            vocab,
        }
    }

    /// Drops queued tokens outside the vocabulary.
    fn clean(&mut self) {
        while self.queue.front().is_some_and(|&t| t as usize >= self.vocab) {
            let t = self.queue.pop_front().unwrap();
            eprintln!("[token hook: token {t} is outside the vocabulary ({}); skipped]", self.vocab);
        }
    }
}

fn lock<T: ?Sized>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

impl TokenFilter for HookFilter {
    fn mask(&mut self, logits: &mut [f32]) {
        self.vocab = logits.len();
        self.clean();
        match self.queue.front() {
            Some(&t) => {
                logits.fill(f32::NEG_INFINITY);
                // Finite whatever the model gave it.
                logits[t as usize] = 0.0;
            }
            None => {
                if let Some(f) = &mut self.inner {
                    f.mask(logits);
                }
            }
        }
    }

    fn accept(&mut self, token: u32) {
        self.clean();
        if self.queue.front() == Some(&token) {
            self.queue.pop_front();
        } else if let Some(f) = &mut self.inner {
            f.accept(token);
        }
        self.reply.push(token);
        if let Some(h) = &self.hook {
            let more = lock(h).on_token(&self.reply);
            self.queue.extend(more);
        }
    }

    fn fork(&self) -> Box<dyn TokenFilter> {
        Box::new(HookFilter {
            hook: None,
            inner: self.inner.as_ref().map(|f| f.fork()),
            queue: self.queue.clone(),
            reply: Vec::new(),
            vocab: self.vocab,
        })
    }

    fn complete(&self) -> bool {
        self.queue.is_empty() && self.inner.as_ref().is_none_or(|f| f.complete())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::Sampler;

    /// A toy model: the next token's logits favour `(last + 1) % 8`.
    fn logits(last: u32) -> Vec<f32> {
        let mut l = vec![0.0f32; 8];
        l[((last + 1) % 8) as usize] = 5.0;
        l
    }

    #[test]
    fn forced_tokens_follow_the_trigger_then_sampling_resumes() {
        let seen = Arc::new(Mutex::new(Vec::<Vec<u32>>::new()));
        let s2 = seen.clone();
        // After token 3: force 7, 7.
        let hook: SharedHook = Arc::new(Mutex::new(Box::new(move |r: &[u32]| {
            lock(&s2).push(r.to_vec());
            if r.last() == Some(&3) { vec![7, 7] } else { Vec::new() }
        })));
        let mut s = Sampler::new(0.0, 0, 1.0, 1);
        s.set_guide(Some(Box::new(HookFilter::new(hook, None, 8))), None);
        let mut last = 0;
        let mut out = Vec::new();
        for _ in 0..6 {
            last = s.sample(&logits(last));
            out.push(last);
        }
        assert_eq!(out, [1, 2, 3, 7, 7, 0]);
        let r = s.take_guide().unwrap();
        assert_eq!(r.tokens, out);
        assert_eq!(r.complete, Some(true));
        assert_eq!(lock(&seen).len(), 6);
        assert_eq!(lock(&seen)[3], [1, 2, 3, 7]);
    }

    #[test]
    fn drafts_see_the_queue_but_not_the_hook() {
        let calls = Arc::new(Mutex::new(0usize));
        let c2 = calls.clone();
        let hook: SharedHook = Arc::new(Mutex::new(Box::new(move |_: &[u32]| {
            *lock(&c2) += 1;
            vec![5]
        })));
        let mut f = HookFilter::new(hook, None, 8);
        f.accept(1);
        assert_eq!(*lock(&calls), 1);
        let mut d = f.fork();
        let mut l = logits(1);
        d.mask(&mut l);
        assert_eq!(crate::sampler::argmax(&l), 5);
        d.accept(5);
        // The draft's queue is empty now: the toy logits again; the hook was not called.
        let mut l = logits(5);
        d.mask(&mut l);
        assert_eq!(crate::sampler::argmax(&l), 6);
        assert_eq!(*lock(&calls), 1);
    }

    #[test]
    fn start_tokens_and_out_of_vocabulary() {
        struct H;
        impl TokenHook for H {
            fn on_token(&mut self, _: &[u32]) -> Vec<u32> {
                Vec::new()
            }
            fn on_start(&mut self) -> Vec<u32> {
                vec![99, 4]
            }
        }
        let hook: SharedHook = Arc::new(Mutex::new(Box::new(H)));
        let mut s = Sampler::new(0.0, 0, 1.0, 1);
        s.set_guide(Some(Box::new(HookFilter::new(hook, None, 8))), None);
        // 99 is skipped, 4 forced, then the toy model's 5.
        assert_eq!(s.sample(&logits(0)), 4);
        assert_eq!(s.sample(&logits(4)), 5);
    }
}
