//! [`EnginePool`]: several conversations of one loaded model at the same time, each in its own
//! session (its own caches) over the same weights, scheduled together by a [`crate::batch::Group`]
//! (prompt chunks interleaved with the others' decoding, plain decode steps batched).

use super::*;
use crate::batch::{BatchOptions, Group};
use std::sync::atomic::AtomicU64;
use std::sync::{Condvar, Mutex, MutexGuard};

impl Engine {
    /// Another session of the loaded model for a conversation of its own: the same weights,
    /// plans, draft and settings (sampling, chat config, prompt cache directory), and empty
    /// caches. It shares this engine's [`crate::batch::Group`] (if any), so their requests can
    /// run at the same time. Engines with interventions or a token hook, Core ML models and the
    /// families with a single conversation per load (Unlimited-OCR) cannot fork.
    pub fn fork(&self) -> Result<Engine> {
        if self.iv.spec.is_some() || self.token_hook.is_some() {
            bail!("an engine with interventions or a token hook cannot fork");
        }
        let lm = self.lm.fork().context("this model runs one conversation per load")?;
        let pcache = match &self.pcache {
            Some(pc) => Some(crate::snapshot::PromptCache::open(pc.dir(), pc.cap())?),
            None => None,
        };
        Ok(Engine {
            lm,
            tok: self.tok.clone(),
            fmt: self.fmt,
            stop: self.stop.clone(),
            sampler: sampler_for(&self.sampling),
            sampling: self.sampling,
            config: self.config.clone(),
            info: self.info.clone(),
            files: self.files,
            mmproj: self.mmproj.clone(),
            prefill_chunk: self.prefill_chunk,
            max_context: self.max_context,
            memory_limit: self.memory_limit,
            draft_tokens: self.draft_tokens,
            draft_fixed: self.draft_fixed,
            turns: 0,
            unfed: None,
            chat_ids: None,
            cancel: Arc::new(AtomicBool::new(false)),
            stats: Stats::default(),
            // The first engine stops it when it drops.
            streamer: None,
            decision: None,
            iv: IvState {
                cache_dir: self.iv.cache_dir.clone(),
                ..Default::default()
            },
            lookup: reset_lookup(self.lookup.clone()),
            history: Vec::new(),
            history_media: Vec::new(),
            constrain: Default::default(),
            token_hook: None,
            pcache,
            pcache_min: self.pcache_min,
            pcache_pending: false,
            checkpoints: self.checkpoints,
            omni_voice: None,
            fingerprint: self.fingerprint.clone(),
            fingerprint_source: self.fingerprint_source.clone(),
            loras: self.loras.clone(),
            group: self.group.clone(),
            _arena: self._arena.clone(),
        })
    }

    /// The scheduler this engine shares with its forks ([`EnginePool`]).
    pub fn group(&self) -> Option<&Arc<Group>> {
        self.group.as_ref()
    }
}

/// One session of an [`EnginePool`].
struct Slot {
    engine: Mutex<Engine>,
    /// The messages of the last conversation it ran (what the next request is matched against).
    last: Mutex<Vec<Message>>,
    /// Pool clock of the last use.
    used: AtomicU64,
}

/// Up to [`EnginePool::len`] conversations of one model at a time: engines sharing the weights
/// ([`Engine::fork`]) and a [`crate::batch::Group`]. A request takes the idle session whose last
/// conversation shares the longest beginning with its own ([`EnginePool::acquire`]), so that each
/// client's conversation keeps its cache; when all are busy it waits for one.
pub struct EnginePool {
    slots: Vec<Slot>,
    tok: Tokenizer,
    group: Arc<Group>,
    clock: AtomicU64,
    free: (Mutex<()>, Condvar),
}

/// A session of an [`EnginePool`] in use (released on drop).
pub struct PoolGuard<'a> {
    pool: &'a EnginePool,
    index: usize,
    engine: MutexGuard<'a, Engine>,
}

impl std::ops::Deref for PoolGuard<'_> {
    type Target = Engine;
    fn deref(&self) -> &Engine {
        &self.engine
    }
}

impl std::ops::DerefMut for PoolGuard<'_> {
    fn deref_mut(&mut self) -> &mut Engine {
        &mut self.engine
    }
}

impl PoolGuard<'_> {
    /// Which session of the pool this is.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Records the conversation the session now holds (matched against the next requests).
    pub fn remember(&self, messages: &[Message]) {
        *self.pool.slots[self.index].last.lock().unwrap_or_else(|p| p.into_inner()) = messages.to_vec();
    }
}

impl Drop for PoolGuard<'_> {
    fn drop(&mut self) {
        let s = &self.pool.slots[self.index];
        s.used.store(self.pool.clock.fetch_add(1, Ordering::Relaxed) + 1, Ordering::Relaxed);
        let _g = self.pool.free.0.lock().unwrap_or_else(|p| p.into_inner());
        self.pool.free.1.notify_all();
    }
}

/// How much of `new`'s beginning `old` has: the length of the leading messages they share and of
/// the common start of the first that differs (bytes of content).
fn shared(old: &[Message], new: &[Message]) -> usize {
    let mut n = 0;
    for (a, b) in old.iter().zip(new) {
        if a.role != b.role {
            break;
        }
        if a == b {
            n += a.content.len() + 1;
            continue;
        }
        if a.media == b.media {
            n += a.content.bytes().zip(b.content.bytes()).take_while(|(x, y)| x == y).count();
        }
        break;
    }
    n
}

impl EnginePool {
    /// A pool of `slots` sessions: `engine` and `slots - 1` forks of it ([`Engine::fork`]; fewer
    /// when the engine cannot fork, which [`EnginePool::len`] tells), their requests scheduled with
    /// `opts`.
    pub fn new(mut engine: Engine, slots: usize, opts: BatchOptions) -> Self {
        let group = Group::new(opts);
        engine.group = Some(group.clone());
        let mut engines = vec![engine];
        for _ in 1..slots.max(1) {
            match engines[0].fork() {
                Ok(e) => engines.push(e),
                Err(e) => {
                    eprintln!("[one conversation at a time: {e:#}]");
                    break;
                }
            }
        }
        Self {
            tok: engines[0].tok.clone(),
            slots: engines
                .into_iter()
                .map(|e| Slot {
                    engine: Mutex::new(e),
                    last: Mutex::new(Vec::new()),
                    used: AtomicU64::new(0),
                })
                .collect(),
            group,
            clock: AtomicU64::new(0),
            free: (Mutex::new(()), Condvar::new()),
        }
    }

    /// The model's tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }

    /// Sessions in the pool.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// The scheduler of the pool's sessions.
    pub fn group(&self) -> &Arc<Group> {
        &self.group
    }

    /// Sessions running a request now.
    pub fn busy(&self) -> usize {
        self.slots.iter().filter(|s| s.engine.try_lock().is_err()).count()
    }

    /// The idle session best for a conversation of `messages` (None: all are busy): the one whose
    /// last conversation shares the most of its beginning, else the least recently used.
    pub fn try_acquire(&self, messages: &[Message]) -> Option<PoolGuard<'_>> {
        let mut best: Option<(usize, u64, usize, MutexGuard<'_, Engine>)> = None;
        for (i, s) in self.slots.iter().enumerate() {
            let g = match s.engine.try_lock() {
                Ok(g) => g,
                Err(std::sync::TryLockError::Poisoned(p)) => p.into_inner(),
                Err(std::sync::TryLockError::WouldBlock) => continue,
            };
            let score = shared(&s.last.lock().unwrap_or_else(|p| p.into_inner()), messages);
            let used = s.used.load(Ordering::Relaxed);
            // More shared first; among equals the least recently used.
            if best.as_ref().is_none_or(|b| score > b.0 || (score == b.0 && used < b.1)) {
                best = Some((score, used, i, g));
            }
        }
        best.map(|(_, _, index, engine)| PoolGuard { pool: self, index, engine })
    }

    /// [`EnginePool::try_acquire`], waiting for a session to be free; `wait` is called about
    /// every `every` while all are busy and stops the wait (None) when it returns false.
    pub fn acquire(&self, messages: &[Message], every: std::time::Duration, wait: &mut dyn FnMut() -> bool) -> Option<PoolGuard<'_>> {
        loop {
            if let Some(g) = self.try_acquire(messages) {
                return Some(g);
            }
            if !wait() {
                return None;
            }
            let g = self.free.0.lock().unwrap_or_else(|p| p.into_inner());
            // (Re-checked under the lock a release notifies under: no wake-up is missed for long.)
            let _ = self.free.1.wait_timeout(g, every).unwrap_or_else(|p| p.into_inner());
        }
    }

    /// Session `i`, waiting for it.
    pub fn slot(&self, i: usize) -> PoolGuard<'_> {
        PoolGuard {
            pool: self,
            index: i,
            engine: self.slots[i].engine.lock().unwrap_or_else(|p| p.into_inner()),
        }
    }

    /// `f` on each idle session (never waits).
    pub fn each_idle<R>(&self, mut f: impl FnMut(&mut Engine) -> R) -> Vec<R> {
        self.slots
            .iter()
            .filter_map(|s| s.engine.try_lock().ok().map(|mut e| f(&mut e)))
            .collect()
    }
}
