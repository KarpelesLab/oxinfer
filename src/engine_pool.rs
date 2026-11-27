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
            default_systems: self.default_systems.clone(),
            #[cfg(test)]
            clock_override: self.clock_override,
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
            adapter_tag: String::new(),
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
    /// The adapter of the last conversation it ran ([`EnginePool::acquire_with`]; empty: none).
    adapter: Mutex<String>,
}

/// An adapter attached to a pool's model while it is loaded ([`EnginePool::attach_adapter`]).
#[derive(Clone, Debug)]
pub struct PoolAdapter {
    /// The name requests select it by ([`EnginePool::acquire_with`]).
    pub name: String,
    /// The caller's mark of the file's state when it was attached (its modification time, say).
    pub version: String,
    /// The adapter as loaded: path, format, rank, alpha, strength, updates, bytes.
    pub info: crate::lora::Attached,
    switch: crate::lora::Switched,
}

impl PoolAdapter {
    /// What sessions and the prompt cache tell this adapter's conversations apart by
    /// ([`Engine::set_adapter_tag`]).
    fn tag(&self) -> String {
        format!("{}|{}|{}", self.info.path.display(), self.version, self.info.strength)
    }
}

/// Which adapter the requests running now use: all the same one (or none), since the switches
/// are the model's. A request for another waits for those to end; the requests arriving after
/// it wait behind it.
#[derive(Default)]
struct Gate {
    /// The adapter switched on (None: none). [`EXCLUSIVE`] while the adapters change.
    mode: Option<String>,
    running: usize,
    /// What the waiting requests want.
    waiters: Vec<Option<String>>,
}

/// [`Gate::mode`] while the set of adapters changes (no adapter has this name).
const EXCLUSIVE: &str = "\0exclusive";

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
    /// The adapters attached since the load, by name.
    adapters: Mutex<Vec<PoolAdapter>>,
    gate: (Mutex<Gate>, Condvar),
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
    /// The adapter its request runs with.
    adapter: Option<PoolAdapter>,
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

    /// The adapter switched on while this session is held ([`EnginePool::acquire_with`]; None:
    /// none, also when the one asked for was not attached).
    pub fn adapter(&self) -> Option<&PoolAdapter> {
        self.adapter.as_ref()
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
        *lock(&s.adapter) = self.engine.adapter_tag().to_string();
        self.pool.leave();
        let _g = self.pool.free.0.lock().unwrap_or_else(|p| p.into_inner());
        self.pool.free.1.notify_all();
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// The common start of two differing messages that counts in [`shared`] (bytes): below it, the
/// conversations only share a few words ("Request 1" / "Request 2"), a session holding the
/// other one is worth more kept than reused (its own client may continue it), and a new
/// conversation takes an empty or the least recently used session instead.
const MIN_PARTIAL: usize = 256;

/// How much of `new`'s beginning `old` has: the length of the leading messages they share and of
/// the common start of the first that differs if at least [`MIN_PARTIAL`] (bytes of content).
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
            let c = a.content.bytes().zip(b.content.bytes()).take_while(|(x, y)| x == y).count();
            if c >= MIN_PARTIAL {
                n += c;
            }
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
                    adapter: Mutex::new(String::new()),
                })
                .collect(),
            group,
            free: (Mutex::new(()), Condvar::new()),
            budget: AtomicU64::new(u64::MAX),
            adapters: Mutex::new(Vec::new()),
            gate: (Mutex::new(Gate::default()), Condvar::new()),
        }
    }

    /// Enters the gate for requests running with adapter `want` (None: none), setting the
    /// switches when no request runs: at once when it can (`wait` None: false otherwise), else
    /// waiting, `wait.1` called about every `wait.0` and giving up (false) when it returns
    /// false. [`EnginePool::leave`] must follow a true.
    fn enter(&self, want: Option<&str>, mut wait: Option<(std::time::Duration, &mut dyn FnMut() -> bool)>) -> bool {
        let want = want.map(str::to_string);
        let mut g = lock(&self.gate.0);
        let mut queued = false;
        loop {
            // The same adapter as the running requests, unless a request for another waits.
            if g.running == 0 || (g.mode == want && !g.waiters.iter().any(|w| *w != g.mode)) {
                break;
            }
            let Some((every, more)) = wait.as_mut() else { return false };
            if !queued {
                g.waiters.push(want.clone());
                queued = true;
            }
            g = self.gate.1.wait_timeout(g, *every).unwrap_or_else(|p| p.into_inner()).0;
            if g.running != 0 && !more() {
                if let Some(i) = g.waiters.iter().position(|w| *w == want) {
                    g.waiters.remove(i);
                }
                self.gate.1.notify_all();
                return false;
            }
        }
        if queued && let Some(i) = g.waiters.iter().position(|w| *w == want) {
            g.waiters.remove(i);
        }
        if g.running == 0 {
            for a in lock(&self.adapters).iter() {
                a.switch.set(Some(a.name.as_str()) == want.as_deref());
            }
            g.mode = want;
        }
        g.running += 1;
        true
    }

    fn leave(&self) {
        let mut g = lock(&self.gate.0);
        g.running = g.running.saturating_sub(1);
        drop(g);
        self.gate.1.notify_all();
    }

    /// `f` on every session while no request runs (waiting for those in flight to end; the
    /// requests arriving meanwhile wait): where the model's adapters change.
    fn exclusive<R>(&self, f: impl FnOnce(&mut [MutexGuard<'_, Engine>]) -> R) -> R {
        self.enter(Some(EXCLUSIVE), Some((std::time::Duration::from_secs(1), &mut || true)));
        let mut engines: Vec<MutexGuard<'_, Engine>> = self.slots.iter().map(|s| lock(&s.engine)).collect();
        let r = f(&mut engines);
        for s in &self.slots {
            lock(&s.last).clear();
            lock(&s.adapter).clear();
            s.bytes.store(0, Ordering::Relaxed);
        }
        drop(engines);
        self.leave();
        r
    }

    /// Attaches the LoRA adapter at `path` (a PEFT folder or a llama.cpp GGUF adapter, see
    /// [`crate::lora`]) to the loaded model as `name`, without reloading its weights: runtime
    /// updates of `strength`, switched off but for the requests that select it
    /// ([`EnginePool::acquire_with`]). `version` is the caller's mark of the file's state
    /// ([`PoolAdapter::version`]). An adapter of that name is replaced.
    ///
    /// Waits for the requests in flight to end (none sees the change); then every session
    /// starts again (their conversations are read again, or restored from the prompt cache) and
    /// the GPU plans are compiled again on their next use. For Inkling-2 models and for
    /// Qwen-path and Gemma models decoded without MTP or a DSpark draft, not streamed from disk
    /// (`Engine::change_adapters`); others return an error and stay as they were. When the
    /// adapter does not load or fit, the model computes what it did before, and an adapter it
    /// was to replace is gone.
    ///
    /// A replaced or detached adapter's updates leave the GPU plans and add nothing on the CPU,
    /// but stay in memory until the model is unloaded.
    pub fn attach_adapter(&self, name: &str, path: &std::path::Path, strength: f32, version: &str) -> Result<PoolAdapter> {
        if name.is_empty() || name == EXCLUSIVE {
            bail!("adapter name {name:?}");
        }
        self.exclusive(|engines| {
            self.retire(name);
            for e in engines.iter_mut() {
                if let Err(err) = e.flush_prompt_cache() {
                    eprintln!("[prompt cache: {err:#}]");
                }
            }
            // SAFETY: every session of the model is held here, and the others are rebound below.
            let r = unsafe { engines[0].change_adapters(Some((path, strength))) };
            for e in engines[1..].iter_mut() {
                e.rebind();
            }
            let switch = r?.context("no adapter attached")?;
            let a = PoolAdapter {
                name: name.to_string(),
                version: version.to_string(),
                info: switch.adapter.clone(),
                switch,
            };
            lock(&self.adapters).push(a.clone());
            Ok(a)
        })
    }

    /// Takes adapter `name` off the model ([`EnginePool::attach_adapter`] tells what that
    /// frees); false when there is none. Waits for the requests in flight, as attaching does.
    pub fn detach_adapter(&self, name: &str) -> bool {
        if !lock(&self.adapters).iter().any(|a| a.name == name) {
            return false;
        }
        self.exclusive(|engines| {
            if !self.retire(name) {
                return false;
            }
            for e in engines.iter_mut() {
                if let Err(err) = e.flush_prompt_cache() {
                    eprintln!("[prompt cache: {err:#}]");
                }
            }
            // SAFETY: as in `attach_adapter`.
            if let Err(err) = unsafe { engines[0].change_adapters(None) } {
                eprintln!("[detach adapter {name}: {err:#}]");
            }
            for e in engines[1..].iter_mut() {
                e.rebind();
            }
            true
        })
    }

    /// Removes adapter `name` from the list and retires its updates.
    fn retire(&self, name: &str) -> bool {
        let mut list = lock(&self.adapters);
        let Some(i) = list.iter().position(|a| a.name == name) else {
            return false;
        };
        list.remove(i).switch.retire();
        true
    }

    /// The adapters attached since the load ([`EnginePool::attach_adapter`]).
    pub fn adapters(&self) -> Vec<PoolAdapter> {
        lock(&self.adapters).clone()
    }

    /// Adapter `name`, when attached.
    pub fn adapter(&self, name: &str) -> Option<PoolAdapter> {
        lock(&self.adapters).iter().find(|a| a.name == name).cloned()
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
        if !self.enter(None, None) {
            return None;
        }
        let g = self.take(messages, None);
        if g.is_none() {
            self.leave();
        }
        g
    }

    /// The idle session best for `messages` run with `adapter`, the gate entered: the sessions
    /// whose last conversation ran with the same adapter first.
    fn take(&self, messages: &[Message], adapter: Option<&str>) -> Option<PoolGuard<'_>> {
        let adapter = adapter.and_then(|n| self.adapter(n));
        let tag = adapter.as_ref().map(PoolAdapter::tag).unwrap_or_default();
        let mut best: Option<(usize, u64, usize, MutexGuard<'_, Engine>)> = None;
        for (i, s) in self.slots.iter().enumerate() {
            let g = match s.engine.try_lock() {
                Ok(g) => g,
                Err(std::sync::TryLockError::Poisoned(p)) => p.into_inner(),
                Err(std::sync::TryLockError::WouldBlock) => continue,
            };
            // (A session that ran another adapter starts over: it shares nothing.)
            let score = if *lock(&s.adapter) == tag {
                1 + shared(&s.last.lock().unwrap_or_else(|p| p.into_inner()), messages)
            } else {
                0
            };
            let used = s.used.load(Ordering::Relaxed);
            // More shared first; among equals the least recently used.
            if best.as_ref().is_none_or(|b| score > b.0 || (score == b.0 && used < b.1)) {
                best = Some((score, used, i, g));
            }
        }
        let (_, _, index, mut engine) = best?;
        if let Err(e) = engine.set_adapter_tag(&tag) {
            eprintln!("[prompt cache: {e:#}]");
        }
        Some(PoolGuard {
            pool: self,
            index,
            engine,
            adapter,
        })
    }

    /// [`EnginePool::try_acquire`], waiting for a session to be free; `wait` is called about
    /// every `every` while all are busy and stops the wait (None) when it returns false.
    pub fn acquire(&self, messages: &[Message], every: std::time::Duration, wait: &mut dyn FnMut() -> bool) -> Option<PoolGuard<'_>> {
        self.acquire_with(messages, None, every, wait)
    }

    /// [`EnginePool::acquire`] for a request run with adapter `adapter`
    /// ([`EnginePool::attach_adapter`]; None, or a name that is not attached: none, the base
    /// model): it is switched on for the request, and off for the requests without it. The
    /// switches are the model's, so the requests running at one time all use the same adapter (or
    /// none): a request for another one waits for them to end, and those arriving after it wait
    /// behind it. [`PoolGuard::adapter`] tells what the request got.
    pub fn acquire_with(
        &self,
        messages: &[Message],
        adapter: Option<&str>,
        every: std::time::Duration,
        wait: &mut dyn FnMut() -> bool,
    ) -> Option<PoolGuard<'_>> {
        // (An adapter detached since counts as none from here on.)
        let adapter = adapter.filter(|n| self.adapter(n).is_some());
        if !self.enter(adapter, Some((every, &mut *wait))) {
            return None;
        }
        loop {
            if let Some(g) = self.take(messages, adapter) {
                return Some(g);
            }
            if !wait() {
                self.leave();
                return None;
            }
            let g = self.free.0.lock().unwrap_or_else(|p| p.into_inner());
            // (Re-checked under the lock a release notifies under: no wake-up is missed for long.)
            let _ = self.free.1.wait_timeout(g, every).unwrap_or_else(|p| p.into_inner());
        }
    }

    /// Session `i`, waiting for it.
    pub fn slot(&self, i: usize) -> PoolGuard<'_> {
        self.enter(None, Some((std::time::Duration::from_secs(1), &mut || true)));
        let mut engine = self.slots[i].engine.lock().unwrap_or_else(|p| p.into_inner());
        if let Err(e) = engine.set_adapter_tag("") {
            eprintln!("[prompt cache: {e:#}]");
        }
        PoolGuard {
            pool: self,
            index: i,
            engine,
            adapter: None,
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

    /// An adapter attached to a loaded model: off for the requests without it (the replies of
    /// before), on for those that select it (the replies of a model loaded with it), per request
    /// on the same weights; replaced and detached without a reload.
    #[test]
    fn adapters_attach_to_a_loaded_model_and_switch_per_request() {
        let dir = std::env::temp_dir().join(format!("oxinfer-pool-adapters-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (model, ad1, ad2) = (dir.join("model"), dir.join("a1"), dir.join("a2"));
        crate::qwen::tiny::write(&model, false, false).unwrap();
        crate::qwen::tiny::write_adapter(&ad1, 11, 0.6).unwrap();
        crate::qwen::tiny::write_adapter(&ad2, 12, 0.6).unwrap();
        let mut o = EngineOptions::default();
        o.set("device", "cpu").unwrap();
        let load = |o: &EngineOptions| {
            let mut e = Engine::load(model.to_str().unwrap(), o).unwrap();
            e.config_mut().max_tokens = 12;
            e
        };
        // A system message of its own: the default system prompt depends on the clock (the date;
        // the time to the second when this was written, so the replies of two requests a second
        // apart were those of two different prompts).
        let m = vec![Message::system("You answer briefly."), Message::user("Hello there, how are you?")];
        let alone = |lora: Option<&std::path::Path>| {
            let mut o = o.clone();
            if let Some(l) = lora {
                o.set("lora", l.to_str().unwrap()).unwrap();
            }
            load(&o).chat(&m, &mut quiet).unwrap().tokens
        };
        let (base, with1, with2) = (alone(None), alone(Some(&ad1)), alone(Some(&ad2)));
        assert_ne!(base, with1, "the adapter changes nothing");
        assert_ne!(with1, with2);
        let pool = EnginePool::new(load(&o), 2, BatchOptions::default());
        let turn = |adapter: Option<&str>| -> (Vec<u32>, Option<String>) {
            let mut g = pool
                .acquire_with(&m, adapter, std::time::Duration::from_millis(50), &mut || true)
                .unwrap();
            let r = g.chat(&m, &mut quiet).unwrap();
            g.remember(&m);
            (r.tokens, g.adapter().map(|a| a.version.clone()))
        };
        assert_eq!(turn(None).0, base);
        // A name that is not attached is the base model.
        assert_eq!(turn(Some("p")), (base.clone(), None));
        let a = pool.attach_adapter("p", &ad1, 1.0, "v1").unwrap();
        assert_eq!((a.info.rank, a.info.modules, a.info.merged), (4, 12, 0));
        assert_eq!(turn(None).0, base);
        assert_eq!(turn(Some("p")), (with1.clone(), Some("v1".into())));
        assert_eq!(turn(None).0, base);
        // A second adapter beside it, each on for its own requests only.
        pool.attach_adapter("q", &ad2, 1.0, "v1").unwrap();
        assert_eq!(turn(Some("q")).0, with2);
        assert_eq!(turn(Some("p")).0, with1);
        assert_eq!(turn(None).0, base);
        // Replaced: the old one is gone.
        pool.attach_adapter("p", &ad2, 1.0, "v2").unwrap();
        assert_eq!(turn(Some("p")), (with2.clone(), Some("v2".into())));
        assert_eq!(pool.adapters().len(), 2);
        // An adapter that does not load leaves the model as it was, without the one it replaced.
        assert!(pool.attach_adapter("p", &dir.join("missing"), 1.0, "v3").is_err());
        assert_eq!(turn(Some("p")), (base.clone(), None));
        assert_eq!(turn(Some("q")).0, with2);
        assert!(pool.detach_adapter("q"));
        assert!(!pool.detach_adapter("q"));
        assert_eq!(turn(Some("q")), (base.clone(), None));
        assert!(pool.adapters().is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The same on Inkling-2 (the tiny looped checkpoint): an adapter attached after the load is
    /// off for the requests without it and on for those that select it, with the replies of a
    /// model loaded with `lora=` and of the base model; replaced and detached without a reload.
    #[test]
    fn inkling_adapters_attach_to_a_loaded_model_and_switch_per_request() {
        let dir = std::env::temp_dir().join(format!("oxinfer-pool-ink-adapters-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (model, ad1, ad2) = (dir.join("model"), dir.join("a1"), dir.join("a2"));
        crate::inkling::tiny::write_tiny(&model, 3).unwrap();
        crate::inkling::tiny::write_adapter(&ad1, 11, 0.4).unwrap();
        crate::inkling::tiny::write_adapter(&ad2, 12, 0.4).unwrap();
        let mut o = EngineOptions::default();
        o.set("device", "cpu").unwrap();
        let load = |o: &EngineOptions| {
            let mut e = Engine::load(model.to_str().unwrap(), o).unwrap();
            e.config_mut().max_tokens = 12;
            e
        };
        // (With a system message: the default one carries the date (once the time of day), and this random
        // model's reply follows every byte of its prompt.)
        let m = vec![Message::system("Be brief."), Message::user("Hello there, how are you?")];
        let alone = |lora: Option<&std::path::Path>| {
            let mut o = o.clone();
            if let Some(l) = lora {
                o.set("lora", l.to_str().unwrap()).unwrap();
            }
            load(&o).chat(&m, &mut quiet).unwrap().tokens
        };
        let (base, with1, with2) = (alone(None), alone(Some(&ad1)), alone(Some(&ad2)));
        assert_ne!(base, with1, "the adapter changes nothing");
        assert_ne!(with1, with2);
        let pool = EnginePool::new(load(&o), 2, BatchOptions::default());
        let turn = |adapter: Option<&str>| -> (Vec<u32>, Option<String>) {
            let mut g = pool
                .acquire_with(&m, adapter, std::time::Duration::from_millis(50), &mut || true)
                .unwrap();
            let r = g.chat(&m, &mut quiet).unwrap();
            g.remember(&m);
            (r.tokens, g.adapter().map(|a| a.version.clone()))
        };
        assert_eq!(turn(None).0, base);
        let a = pool.attach_adapter("p", &ad1, 1.0, "v1").unwrap();
        assert_eq!((a.info.rank, a.info.modules, a.info.merged), (4, 12, 0));
        assert_eq!(turn(None).0, base);
        assert_eq!(turn(Some("p")), (with1.clone(), Some("v1".into())));
        assert_eq!(turn(None).0, base);
        // A second adapter beside it, each on for its own requests only.
        pool.attach_adapter("q", &ad2, 1.0, "v1").unwrap();
        assert_eq!(turn(Some("q")).0, with2);
        assert_eq!(turn(Some("p")).0, with1);
        assert_eq!(turn(None).0, base);
        // Replaced, then detached.
        pool.attach_adapter("p", &ad2, 1.0, "v2").unwrap();
        assert_eq!(turn(Some("p")), (with2.clone(), Some("v2".into())));
        assert!(pool.detach_adapter("q"));
        assert_eq!(turn(Some("q")), (base.clone(), None));
        assert_eq!(turn(Some("p")).0, with2);
        assert_eq!(turn(None).0, base);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// With the model's MTP head drafting (the tiny looped checkpoint and its tiny drafter): an
    /// adapter attaches as without it, the drafts go on, and the replies are those of the model
    /// loaded with `lora=` or with none, decoded without drafts.
    #[test]
    fn inkling_adapters_attach_while_the_mtp_head_drafts() {
        let dir = std::env::temp_dir().join(format!("oxinfer-pool-ink-mtp-adapters-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (model, ad1, ad2) = (dir.join("model"), dir.join("a1"), dir.join("a2"));
        crate::inkling::tiny::write_tiny(&model, 3).unwrap();
        crate::inkling::tiny::write_tiny_mtp(&model, 9, &[0, 2]).unwrap();
        crate::inkling::tiny::write_adapter(&ad1, 11, 0.4).unwrap();
        crate::inkling::tiny::write_adapter(&ad2, 12, 0.4).unwrap();
        let mut o = EngineOptions::default();
        o.set("device", "cpu").unwrap();
        o.set("draft", "none").unwrap();
        o.set("draft_device", "cpu").unwrap();
        let load = |o: &EngineOptions| {
            let mut e = Engine::load(model.to_str().unwrap(), o).unwrap();
            e.config_mut().max_tokens = 40;
            e
        };
        let m = vec![Message::system("Be brief."), Message::user("Hello there, how are you?")];
        let alone = |lora: Option<&std::path::Path>| {
            let mut o = o.clone();
            if let Some(l) = lora {
                o.set("lora", l.to_str().unwrap()).unwrap();
            }
            load(&o).chat(&m, &mut quiet).unwrap().tokens
        };
        let (base, with1, with2) = (alone(None), alone(Some(&ad1)), alone(Some(&ad2)));
        assert_ne!(base, with1, "the adapter changes nothing");
        assert_ne!(with1, with2);
        let mut mtp = o.clone();
        mtp.set("mtp", "on").unwrap();
        let pool = EnginePool::new(load(&mtp), 2, BatchOptions::default());
        // The reply's tokens and its drafted tokens.
        let turn = |adapter: Option<&str>| -> (Vec<u32>, usize) {
            let mut g = pool
                .acquire_with(&m, adapter, std::time::Duration::from_millis(50), &mut || true)
                .unwrap();
            let r = g.chat(&m, &mut quiet).unwrap();
            g.remember(&m);
            (r.tokens, r.stats.drafted)
        };
        let (t, drafted) = turn(None);
        assert_eq!(t, base);
        assert!(drafted > 0, "no drafts");
        pool.attach_adapter("p", &ad1, 1.0, "v1").unwrap();
        let (t, drafted) = turn(Some("p"));
        assert_eq!(t, with1);
        assert!(drafted > 0, "no drafts with the adapter on");
        let (t, drafted) = turn(None);
        assert_eq!(t, base);
        assert!(drafted > 0, "no drafts after the attach");
        pool.attach_adapter("q", &ad2, 1.0, "v1").unwrap();
        assert_eq!(turn(Some("q")).0, with2);
        assert_eq!(turn(Some("p")).0, with1);
        assert!(pool.detach_adapter("p"));
        assert_eq!(turn(Some("p")).0, base);
        assert_eq!(turn(Some("q")).0, with2);
        assert_eq!(turn(None).0, base);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Requests with different adapters do not overlap: one waits for the other to end.
    #[test]
    fn requests_with_different_adapters_take_turns() {
        let dir = std::env::temp_dir().join(format!("oxinfer-pool-gate-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        crate::qwen::tiny::write(&dir.join("model"), false, false).unwrap();
        crate::qwen::tiny::write_adapter(&dir.join("a"), 11, 0.6).unwrap();
        let mut o = EngineOptions::default();
        o.set("device", "cpu").unwrap();
        let pool = EnginePool::new(
            Engine::load(dir.join("model").to_str().unwrap(), &o).unwrap(),
            2,
            BatchOptions::default(),
        );
        pool.attach_adapter("p", &dir.join("a"), 1.0, "v1").unwrap();
        let m = [Message::user("Hi")];
        let every = std::time::Duration::from_millis(10);
        let base = pool.acquire_with(&m, None, every, &mut || true).unwrap();
        // Another base request runs beside it; one with the adapter does not (a session is free).
        let base2 = pool.acquire_with(&m, None, every, &mut || true).unwrap();
        drop(base2);
        let mut tries = 0;
        let none = pool.acquire_with(&m, Some("p"), every, &mut || {
            tries += 1;
            tries < 3
        });
        assert!(none.is_none());
        assert!(pool.try_acquire(&m).is_some(), "the request that gave up still holds the gate");
        drop(base);
        let with = pool.acquire_with(&m, Some("p"), every, &mut || true).unwrap();
        assert!(with.adapter().is_some());
        assert!(pool.try_acquire(&m).is_none());
        drop(with);
        assert!(pool.try_acquire(&m).is_some());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_few_shared_words_do_not_count() {
        let (a, b) = ([Message::user("Request 1. notes")], [Message::user("Request 2. notes")]);
        assert_eq!(shared(&a, &b), 0);
        let long = "x".repeat(MIN_PARTIAL);
        let a = [Message::user(format!("{long} one"))];
        let b = [Message::user(format!("{long} two"))];
        assert_eq!(shared(&a, &b), MIN_PARTIAL + 1);
        // Whole messages count, however short.
        let sys = Message::system("Be brief.");
        let a = [sys.clone(), Message::user("a")];
        let b = [sys, Message::user("b")];
        assert_eq!(shared(&a, &b), "Be brief.".len() + 1);
    }
}
