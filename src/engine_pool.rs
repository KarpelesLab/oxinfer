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

    /// Bytes of memory the conversation's caches hold (the KV caches on the host and their device
    /// copies, recurrent states, checkpoints); None for the families that do not tell.
    pub fn cache_bytes(&self) -> Option<u64> {
        self.lm.cache_bytes().map(|b| b as u64)
    }

    /// Lets go of the conversation's caches (as [`Engine::reset`]) to free their memory, after
    /// storing the conversation in the prompt cache ([`EngineOptions::prompt_cache`]) when one
    /// is set and it is not there yet, so that the next request continuing it restores it from
    /// there instead of reading it again. Returns the bytes the caches held (0 when they were
    /// empty, or not measured).
    pub fn evict_cache(&mut self) -> Result<u64> {
        if self.context_len() == 0 {
            return Ok(0);
        }
        let bytes = self.cache_bytes().unwrap_or(0);
        let stored = if self.pcache.is_some() && self.lm.snapshots() {
            self.pcache_pending = false;
            let _run = run_lock();
            self.store_snapshot()
        } else {
            Ok(false)
        };
        self.reset();
        stored.map(|_| bytes)
    }
}

/// The clock of the sessions' last uses, shared by every pool so that the least recently used
/// session of several models can be told ([`EnginePool::idle_caches`]).
static CLOCK: AtomicU64 = AtomicU64::new(0);

/// The default memory budget of the sessions' caches ([`EnginePool::set_cache_budget`], the
/// server's `session_cache_gb`): a quarter of the machine's memory, at least 2 GB (None when it
/// is not known).
pub fn default_cache_budget() -> Option<u64> {
    crate::platform::total_memory().map(|m| (m / 4).max(2_000_000_000))
}

/// One session of an [`EnginePool`].
struct Slot {
    engine: Mutex<Engine>,
    /// The messages of the last conversation it ran (what the next request is matched against).
    last: Mutex<Vec<Message>>,
    /// Clock ([`CLOCK`]) of the last use.
    used: AtomicU64,
    /// [`Engine::cache_bytes`] when it was last released (0: empty or not measured).
    bytes: AtomicU64,
}

/// Up to [`EnginePool::len`] conversations of one model at a time: engines sharing the weights
/// ([`Engine::fork`]) and a [`crate::batch::Group`]. A request takes the idle session whose last
/// conversation shares the longest beginning with its own ([`EnginePool::acquire`]), so that each
/// client's conversation keeps its cache; when all are busy it waits for one.
pub struct EnginePool {
    slots: Vec<Slot>,
    tok: Tokenizer,
    group: Arc<Group>,
    free: (Mutex<()>, Condvar),
    /// [`EnginePool::set_cache_budget`] (`u64::MAX`: none).
    budget: AtomicU64,
}

/// An idle session's cache ([`EnginePool::idle_caches`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IdleCache {
    /// The session ([`PoolGuard::index`]).
    pub index: usize,
    /// When it was last used (a clock shared by all pools: lower is older).
    pub used: u64,
    /// Bytes its caches hold ([`Engine::cache_bytes`]; 0 when not measured).
    pub bytes: u64,
    /// Tokens of its conversation.
    pub tokens: usize,
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
        s.used.store(CLOCK.fetch_add(1, Ordering::Relaxed) + 1, Ordering::Relaxed);
        s.bytes.store(self.engine.cache_bytes().unwrap_or(0), Ordering::Relaxed);
        // The other idle sessions over the budget (this one, still held, is kept: the budget
        // bounds the others).
        let budget = self.pool.budget.load(Ordering::Relaxed);
        if budget != u64::MAX {
            self.pool.trim(budget);
        }
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
                    bytes: AtomicU64::new(0),
                })
                .collect(),
            group,
            free: (Mutex::new(()), Condvar::new()),
            budget: AtomicU64::new(u64::MAX),
        }
    }

    /// The memory the sessions' caches may hold together (None: no limit, the default for a
    /// pool): when a request ends and the sessions hold more, the least recently used idle ones
    /// let go of their caches ([`Engine::evict_cache`]: stored in the prompt cache first when one
    /// is set) until they fit, the session just released kept whatever its size. Sessions of
    /// families that do not measure their caches ([`Engine::cache_bytes`]) count for nothing.
    /// (The server sets none per pool: it bounds all its models' sessions together, see
    /// `ServerOptions::session_cache`.)
    pub fn set_cache_budget(&self, bytes: Option<u64>) {
        self.budget.store(bytes.unwrap_or(u64::MAX), Ordering::Relaxed);
        if let Some(b) = bytes {
            self.trim(b);
        }
    }

    /// [`EnginePool::set_cache_budget`].
    pub fn cache_budget(&self) -> Option<u64> {
        Some(self.budget.load(Ordering::Relaxed)).filter(|&b| b != u64::MAX)
    }

    /// Bytes the sessions' caches hold, as last measured (a busy session's: when it was last
    /// released).
    pub fn cache_bytes(&self) -> u64 {
        self.slots.iter().map(|s| s.bytes.load(Ordering::Relaxed)).sum()
    }

    /// The idle sessions holding a conversation, least recently used first (never waits).
    pub fn idle_caches(&self) -> Vec<IdleCache> {
        let mut v: Vec<IdleCache> = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(index, s)| {
                let e = s.engine.try_lock().ok()?;
                let tokens = e.context_len();
                (tokens > 0).then(|| IdleCache {
                    index,
                    used: s.used.load(Ordering::Relaxed),
                    bytes: s.bytes.load(Ordering::Relaxed),
                    tokens,
                })
            })
            .collect();
        v.sort_by_key(|c| c.used);
        v
    }

    /// Lets idle session `index` go of its caches ([`Engine::evict_cache`]); None when it is
    /// busy (or does not exist). Returns the bytes freed.
    pub fn evict(&self, index: usize) -> Option<Result<u64>> {
        let s = self.slots.get(index)?;
        let mut e = s.engine.try_lock().ok()?;
        let r = e.evict_cache();
        s.bytes.store(e.cache_bytes().unwrap_or(0), Ordering::Relaxed);
        Some(r)
    }

    /// Evicts the least recently used idle sessions' caches until the sessions hold at most
    /// `budget` bytes (or no idle one is left); returns the bytes freed.
    pub fn trim(&self, budget: u64) -> u64 {
        let mut freed = 0;
        for c in self.idle_caches() {
            if self.cache_bytes() <= budget {
                break;
            }
            if c.bytes == 0 {
                continue;
            }
            match self.evict(c.index) {
                Some(Ok(b)) => freed += b,
                Some(Err(e)) => eprintln!("[session cache: {e:#}]"),
                None => {}
            }
        }
        freed
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

#[cfg(test)]
mod tests {
    use super::*;

    fn quiet(_: &Delta) -> Flow {
        Flow::Continue
    }

    /// Over the budget, the least recently used idle session lets go of its cache, stored in the
    /// prompt cache: its conversation, when it comes back, is restored from there and gets the
    /// reply it would have got from the cache in memory.
    #[test]
    fn idle_sessions_over_the_budget_let_go_of_their_caches() {
        let dir = std::env::temp_dir().join(format!("oxinfer-pool-evict-{}", std::process::id()));
        let cache = dir.join("pc");
        crate::inkling::tiny::write_tiny(&dir, 3).unwrap();
        let mut plain = EngineOptions::default();
        plain.set("device", "cpu").unwrap();
        let mut o = plain.clone();
        o.set("prompt_cache", cache.to_str().unwrap()).unwrap();
        o.set("prompt_cache_min_tokens", "8").unwrap();
        let load = |o: &EngineOptions| {
            let mut e = Engine::load(dir.to_str().unwrap(), o).unwrap();
            e.config_mut().max_tokens = 6;
            e
        };
        let a = vec![
            Message::system("You answer briefly. ".repeat(6)),
            Message::user("Hello there, how are you?"),
        ];
        let b = vec![
            Message::system("Tell stories. ".repeat(8)),
            Message::user("A story about a lantern."),
        ];
        let pool = EnginePool::new(load(&o), 2, BatchOptions::default());
        assert_eq!(pool.len(), 2);
        pool.set_cache_budget(Some(0));
        let turn = |m: &[Message]| -> (usize, Reply) {
            let mut g = pool.try_acquire(m).unwrap();
            let r = g.chat(m, &mut quiet).unwrap();
            g.remember(m);
            (g.index(), r)
        };
        let (ia, ra) = turn(&a);
        assert!(pool.cache_bytes() > 0);
        // The other conversation on the other session: the first, now the least recently used
        // and idle, lets go of its cache (the one just released is kept).
        let (ib, _) = turn(&b);
        assert_ne!(ia, ib);
        let idle = pool.idle_caches();
        assert_eq!(idle.iter().map(|c| c.index).collect::<Vec<_>>(), vec![ib]);
        assert_eq!(pool.slot(ia).context_len(), 0);
        // The first conversation goes on: restored from the prompt cache, as it would have gone
        // on from memory.
        let mut a2 = a.clone();
        a2.push(Message::assistant(format!("{}.", ra.text)));
        a2.push(Message::user("And then?"));
        let (i2, r2) = turn(&a2);
        assert_eq!(i2, ia);
        assert!(r2.stats.restored_tokens > 0, "not restored");
        let mut e = load(&plain);
        e.chat(&a, &mut quiet).unwrap();
        let r = e.chat(&a2, &mut quiet).unwrap();
        assert_eq!(r.tokens, r2.tokens);
        // No budget: nothing is evicted.
        pool.set_cache_budget(None);
        turn(&b);
        assert_eq!(pool.idle_caches().len(), 2);
        std::fs::remove_dir_all(&dir).ok();
    }
}
