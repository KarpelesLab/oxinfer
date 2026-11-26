//! Concurrent conversations on one model: turns on the device and batched decode steps.
//!
//! Several sessions of one loaded model ([`crate::engine::EnginePool`]: engines sharing the
//! weights, each with its own caches) serve requests at the same time, each on its own thread.
//! They share a [`Group`]: the model runs one *job* at a time, and the group decides whose:
//!
//! - a request **holds the turn** while it runs anything but plain decode steps (a prefill
//!   chunk, a speculative round, the head of the prompt's last token...), and gives it up between
//!   prompt chunks and speculative rounds when another request waits ([`yield_now`]), so that a
//!   long prompt does not stall the replies being decoded;
//! - **plain decode steps** of the requests decoding at that moment run as one batched step
//!   ([`decode_step`]: [`crate::generate::TextLm::forward_batch`]), the weights read once for all
//!   of them, as soon as every decoding request has its next token (or after
//!   [`BatchOptions::gather_us`]); a request samples and streams its token outside the turn;
//! - when both wait, batched steps and other jobs alternate (first come, first served among
//!   the other jobs).
//!
//! A request alone on its model takes none of this: its steps run as they would without a group
//! (speculative decoding, greedy passes chained on the device), so single-request latency is
//! unchanged. A greedy chain stops queuing passes when another request arrives.
//!
//! Each batched row equals the request's own one-token step where the model's batched step is
//! batch-invariant (see [`crate::generate::TextLm::forward_batch`]), so a reply does not depend
//! on the requests it was decoded with.
//!
//! Turns imply the engine-wide compute lock (`crate::engine::RUN`): a job holds both.

use crate::clock::Instant;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Duration;

/// Settings of a [`Group`].
#[derive(Clone, Debug, PartialEq)]
pub struct BatchOptions {
    /// Most sequences in one batched decode step (default 8; 1 runs decode steps one by one, in
    /// turns, still interleaved with the other requests' prompts).
    pub max_batch: usize,
    /// How long a decode step waits for the other decoding requests' tokens before the batch runs
    /// without them, in microseconds (default 3000).
    pub gather_us: u64,
    /// Batched steps on the GPU use the fast-verification kernels (matrix units; see
    /// `crate::gpu::Backend::decode_plan_multi_fast`) instead of the exact multi-token ones: a
    /// row's logits then differ from its one-token step's by rounding (~1e-4), so a reply may
    /// depend on the requests decoded with it. Default off (process-wide, see [`set_fast`]).
    pub fast: bool,
    /// The most prompt tokens a chunk reads while other requests of the model are decoding (0:
    /// the model's own prefill chunk; see [`chunk_for`]). Their steps wait for each chunk: a
    /// smaller one lets them run more often, at the cost of prompt speed where a model reads
    /// big chunks faster (mixtures of experts).
    pub decode_chunk: usize,
}

static FAST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Sets [`BatchOptions::fast`] for the process (the models' batched steps read it).
pub fn set_fast(on: bool) {
    FAST.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether batched GPU steps use the fast-verification kernels ([`BatchOptions::fast`]).
pub fn fast() -> bool {
    FAST.load(std::sync::atomic::Ordering::Relaxed)
}

/// Whether a batched GPU step of `rows` rows takes the fast-verification kernels: with
/// [`BatchOptions::fast`], from 5 rows on (up to 4 the exact multi-token kernels, one sweep of
/// the weights, are as fast or faster: Qwen3-8B on an M5 Max, 2 rows 61 against 49 tok/s a
/// stream, 4 rows 46 against 47; 8 rows 20 against 42).
pub fn fast_for(rows: usize) -> bool {
    fast() && rows > 4
}

impl Default for BatchOptions {
    fn default() -> Self {
        Self {
            max_batch: 8,
            gather_us: 3000,
            fast: false,
            decode_chunk: 0,
        }
    }
}

/// Runs the batched step of the sessions `sess` (type-erased `&M`, `&mut M::Sess`), one token
/// each: the logits of each row.
type RunFn = unsafe fn(model: usize, sess: &[usize], tokens: &[u32]) -> Vec<Vec<f32>>;

struct Pending {
    id: u64,
    since: Instant,
    model: usize,
    sess: usize,
    token: u32,
    run: RunFn,
}

/// What the group's scheduler hands the model to next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Grant {
    Batch,
    Solo(u64),
    /// A batch waits for its rows (until the deadline), or nothing waits.
    Wait,
}

#[derive(Default)]
struct St {
    /// The member holding the turn ([`BATCH`]: a batched step runs).
    holder: Option<u64>,
    /// Members waiting for the turn, in arrival order, with the prompt tokens each has left to
    /// read ([`set_remaining`]: the fewest go first).
    queue: VecDeque<(u64, usize)>,
    /// Members in a turn ([`enter`] .. drop).
    members: usize,
    /// Members inside a plain decoding loop ([`decoding`]).
    decoding: usize,
    pending: Vec<Pending>,
    /// Logits of finished batched rows by member (None: the batch failed).
    done: Vec<(u64, Option<Vec<f32>>)>,
    /// The last job was a batched step.
    last_batch: bool,
    /// When the last job ended: rows submitted while it ran wait for the others from then on
    /// (they are sampling their token from that very job).
    freed: Option<Instant>,
    next_id: u64,
    stats: GroupStats,
}

/// What a [`Group`] has run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GroupStats {
    /// Batched decode steps, and the rows they ran.
    pub batches: u64,
    pub rows: u64,
    /// Turns given to other jobs.
    pub turns: u64,
}

const BATCH: u64 = u64::MAX;

/// The scheduler of a model's concurrent sessions (see the module docs).
pub struct Group {
    st: Mutex<St>,
    cv: Condvar,
    opts: BatchOptions,
}

impl Group {
    pub fn new(opts: BatchOptions) -> Arc<Self> {
        if opts.fast {
            set_fast(true);
        }
        Arc::new(Self {
            st: Mutex::new(St::default()),
            cv: Condvar::new(),
            opts: BatchOptions {
                max_batch: opts.max_batch.max(1),
                ..opts
            },
        })
    }

    pub fn options(&self) -> &BatchOptions {
        &self.opts
    }

    /// Requests in a turn right now.
    pub fn members(&self) -> usize {
        self.lock().members
    }

    pub fn stats(&self) -> GroupStats {
        self.lock().stats
    }

    fn lock(&self) -> MutexGuard<'_, St> {
        self.st.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn grant(&self, st: &St) -> Grant {
        if st.holder.is_some() {
            return Grant::Wait;
        }
        let ready = !st.pending.is_empty()
            && (st.pending.len() >= st.decoding.min(self.opts.max_batch)
                || self.gathered(st) >= Duration::from_micros(self.opts.gather_us));
        // The shortest prompt left first (the first come among equals): short requests get their
        // first token without waiting for long prompts.
        let next = st.queue.iter().enumerate().min_by_key(|(i, q)| (q.1, *i)).map(|(_, q)| q.0);
        match next {
            Some(id) if !ready || st.last_batch => Grant::Solo(id),
            _ if ready => Grant::Batch,
            _ => Grant::Wait,
        }
    }

    /// How long the oldest pending row has waited for the others: since it came, or since the
    /// last job ended if it came while that ran.
    fn gathered(&self, st: &St) -> Duration {
        let Some(p) = st.pending.first() else { return Duration::ZERO };
        match st.freed {
            Some(f) if f > p.since => f.elapsed(),
            _ => p.since.elapsed(),
        }
    }

    /// Waits on the condition variable, waking up by the gather deadline when a batch waits for
    /// its rows.
    fn wait<'a>(&self, st: MutexGuard<'a, St>) -> MutexGuard<'a, St> {
        let timeout = st.pending.first().map(|_| {
            Duration::from_micros(self.opts.gather_us)
                .saturating_sub(self.gathered(&st))
                .max(Duration::from_micros(100))
        });
        match timeout {
            Some(t) => self.cv.wait_timeout(st, t).unwrap_or_else(|p| p.into_inner()).0,
            None => self.cv.wait(st).unwrap_or_else(|p| p.into_inner()),
        }
    }

    /// Takes the turn for member `id` (first come, first served, batched steps in between).
    fn acquire(&self, id: u64, remaining: usize) {
        let mut st = self.lock();
        if !st.queue.iter().any(|q| q.0 == id) {
            st.queue.push_back((id, remaining));
        }
        loop {
            match self.grant(&st) {
                Grant::Solo(g) if g == id => {
                    st.queue.retain(|q| q.0 != id);
                    st.holder = Some(id);
                    st.last_batch = false;
                    st.stats.turns += 1;
                    return;
                }
                // (A batch runs on one of its rows' threads: this one may be inside a prompt,
                // whose thread settings are not a decode step's.)
                _ => st = self.wait(st),
            }
        }
    }

    fn release(&self) {
        let mut st = self.lock();
        st.holder = None;
        st.freed = Some(Instant::now());
        drop(st);
        self.cv.notify_all();
    }

    /// Runs the oldest pending rows (at most `max_batch`) as one step, the lock released while
    /// it runs; their logits go to `done`.
    fn run_batch<'a>(&'a self, mut st: MutexGuard<'a, St>) -> MutexGuard<'a, St> {
        let n = st.pending.len().min(self.opts.max_batch);
        let rows: Vec<Pending> = st.pending.drain(..n).collect();
        let (t0, waited) = (Instant::now(), rows[0].since.elapsed());
        st.holder = Some(BATCH);
        st.last_batch = true;
        st.stats.batches += 1;
        st.stats.rows += n as u64;
        drop(st);
        // Any failure (a panic in the model) fails the rows instead of leaving them waiting.
        struct Fail<'g>(&'g Group, Vec<u64>);
        impl Drop for Fail<'_> {
            fn drop(&mut self) {
                if self.1.is_empty() {
                    return;
                }
                let mut st = self.0.lock();
                st.done.extend(self.1.iter().map(|&id| (id, None)));
                st.holder = None;
                drop(st);
                self.0.cv.notify_all();
            }
        }
        let mut fail = Fail(self, rows.iter().map(|r| r.id).collect());
        let out = {
            let _run = crate::engine::run_lock();
            let sess: Vec<usize> = rows.iter().map(|r| r.sess).collect();
            let toks: Vec<u32> = rows.iter().map(|r| r.token).collect();
            // SAFETY: every row's thread waits in `decode_step` until its result is in `done`, so
            // its session is not used elsewhere, and the model outlives the step; the rows of a
            // group are sessions of one model (one `RunFn`).
            unsafe { (rows[0].run)(rows[0].model, &sess, &toks) }
        };
        fail.1.clear();
        if trace() {
            eprintln!(
                "[batch: {n} rows ({} decoding), gathered in {:.2} ms, ran in {:.2} ms]",
                self.lock().decoding,
                waited.as_secs_f64() * 1e3,
                t0.elapsed().as_secs_f64() * 1e3
            );
        }
        let mut st = self.lock();
        for (r, l) in rows.iter().zip(out) {
            st.done.push((r.id, Some(l)));
        }
        st.holder = None;
        st.freed = Some(Instant::now());
        self.cv.notify_all();
        st
    }
}

/// `OXINFER_BATCH_TRACE=1`: a line per batched step on stderr.
pub(crate) fn trace() -> bool {
    static T: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *T.get_or_init(|| std::env::var_os("OXINFER_BATCH_TRACE").is_some())
}

/// This thread's membership of a group.
struct Current {
    group: Arc<Group>,
    id: u64,
    holding: bool,
    /// The engine-wide compute lock, held with the turn.
    run: Option<MutexGuard<'static, ()>>,
    /// Inside a plain decoding loop.
    decoding: bool,
    /// Prompt tokens left to read ([`set_remaining`]).
    remaining: usize,
}

thread_local! {
    static CURRENT: RefCell<Option<Current>> = const { RefCell::new(None) };
}

/// A request's membership of a group ([`enter`]); leaving gives up the turn.
pub struct Member {
    active: bool,
}

impl Drop for Member {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let Some(c) = CURRENT.with(|c| c.borrow_mut().take()) else { return };
        let mut st = c.group.lock();
        st.members -= 1;
        if c.decoding {
            st.decoding -= 1;
        }
        st.queue.retain(|q| q.0 != c.id);
        if c.holding {
            st.holder = None;
        }
        drop(st);
        drop(c.run);
        c.group.cv.notify_all();
    }
}

/// Joins `group` for a request on this thread and takes the turn (waiting for it): what the
/// request runs on the model until the member drops is scheduled by the group. A thread already
/// in a group stays in that one (the returned member does nothing).
pub fn enter(group: &Arc<Group>) -> Member {
    enter_with(group, 0)
}

/// [`enter`] for a request with `remaining` prompt tokens to read (see [`set_remaining`]).
pub fn enter_with(group: &Arc<Group>, remaining: usize) -> Member {
    if CURRENT.with(|c| c.borrow().is_some()) {
        return Member { active: false };
    }
    let id = {
        let mut st = group.lock();
        st.next_id += 1;
        st.members += 1;
        st.next_id
    };
    CURRENT.with(|c| {
        *c.borrow_mut() = Some(Current {
            group: group.clone(),
            id,
            holding: false,
            run: None,
            decoding: false,
            remaining,
        })
    });
    ensure_turn();
    Member { active: true }
}

/// Whether this thread holds a group's turn (and with it the compute lock).
pub fn holding() -> bool {
    CURRENT.with(|c| c.borrow().as_ref().is_some_and(|c| c.holding))
}

/// Takes the turn back if this thread is in a group and gave it up.
pub fn ensure_turn() {
    let Some((g, id, rem)) = CURRENT.with(|c| {
        c.borrow()
            .as_ref()
            .filter(|c| !c.holding)
            .map(|c| (c.group.clone(), c.id, c.remaining))
    }) else {
        return;
    };
    g.acquire(id, rem);
    let run = crate::engine::run_lock_static();
    CURRENT.with(|c| {
        if let Some(c) = c.borrow_mut().as_mut() {
            c.holding = true;
            c.run = Some(run);
        }
    });
}

/// Tells the group how many prompt tokens this thread's request has left to read (0: none, its
/// jobs are short): among the requests waiting for the turn, the fewest go first.
pub fn set_remaining(n: usize) {
    CURRENT.with(|c| {
        if let Some(c) = c.borrow_mut().as_mut() {
            c.remaining = n;
        }
    });
}

/// The size of the next prompt chunk of this thread's request: `chunk`, or at most
/// [`BatchOptions::decode_chunk`] tokens while other requests of its group are decoding.
pub fn chunk_for(chunk: usize) -> usize {
    CURRENT.with(|c| {
        let b = c.borrow();
        let Some(c) = b.as_ref() else { return chunk };
        let dc = c.group.opts.decode_chunk;
        if dc > 0 && c.group.lock().decoding > 0 {
            chunk.min(dc)
        } else {
            chunk
        }
    })
}

/// Gives up the turn (keeping the membership).
fn give_up() {
    let Some((g, run)) = CURRENT.with(|c| {
        let mut b = c.borrow_mut();
        let c = b.as_mut().filter(|c| c.holding)?;
        c.holding = false;
        Some((c.group.clone(), c.run.take()))
    }) else {
        return;
    };
    drop(run);
    g.release();
}

/// Between two jobs of a request (prompt chunks, speculative rounds): lets the requests waiting
/// for the model go first, then takes the turn back.
pub fn yield_now() {
    let Some(g) = CURRENT.with(|c| c.borrow().as_ref().filter(|c| c.holding).map(|c| c.group.clone())) else {
        return;
    };
    let waiting = {
        let st = g.lock();
        !st.queue.is_empty() || !st.pending.is_empty()
    };
    if waiting {
        give_up();
        ensure_turn();
    }
}

/// Whether other requests share this thread's group now (greedy chains on the device stop
/// queuing passes, so that the others' steps can join).
pub fn contended() -> bool {
    CURRENT.with(|c| c.borrow().as_ref().is_some_and(|c| c.group.lock().members > 1))
}

/// A plain decoding loop's registration ([`decoding`]): its steps are batched with the other
/// requests' while it lives; dropping it takes the turn back.
pub struct Decoding {
    active: bool,
}

impl Drop for Decoding {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let g = CURRENT.with(|c| {
            let mut b = c.borrow_mut();
            let c = b.as_mut()?;
            c.decoding = false;
            Some(c.group.clone())
        });
        if let Some(g) = g {
            g.lock().decoding -= 1;
            g.cv.notify_all();
        }
        ensure_turn();
    }
}

/// Registers a plain decoding loop on this thread (see [`decode_step`]).
pub fn decoding() -> Decoding {
    let g = CURRENT.with(|c| {
        let mut b = c.borrow_mut();
        let c = b.as_mut().filter(|c| !c.decoding)?;
        c.decoding = true;
        Some(c.group.clone())
    });
    match g {
        Some(g) => {
            g.lock().decoding += 1;
            Decoding { active: true }
        }
        None => Decoding { active: false },
    }
}

unsafe fn run_rows<M: crate::generate::TextLm>(model: usize, sess: &[usize], tokens: &[u32]) -> Vec<Vec<f32>> {
    // SAFETY: see `Group::run_batch`: `model` is a `&M`, each `sess` a distinct `&mut M::Sess`.
    let model = unsafe { &*(model as *const M) };
    let mut ss: Vec<&mut M::Sess> = sess.iter().map(|&p| unsafe { &mut *(p as *mut M::Sess) }).collect();
    model.forward_batch(&mut ss, tokens)
}

/// One plain decode step: runs `token` and returns the logits after it. Alone on its model (or
/// outside a group) the step runs at once; otherwise it joins the next batched step of the
/// group's decoding requests and returns without the turn (sampling and streaming the token need
/// none; the loop's [`Decoding`] takes it back when it ends).
pub fn decode_step<M: crate::generate::TextLm>(model: &M, sess: &mut M::Sess, token: u32) -> Vec<f32> {
    let cur = CURRENT.with(|c| c.borrow().as_ref().map(|c| (c.group.clone(), c.id, c.decoding)));
    let Some((g, id, true)) = cur.filter(|(g, ..)| g.lock().members > 1) else {
        ensure_turn();
        return model.forward(sess, &[token]);
    };
    give_up();
    let mut st = g.lock();
    st.pending.push(Pending {
        id,
        since: Instant::now(),
        model: model as *const M as usize,
        sess: sess as *mut M::Sess as usize,
        token,
        run: run_rows::<M>,
    });
    g.cv.notify_all();
    loop {
        if let Some(i) = st.done.iter().position(|d| d.0 == id) {
            let (_, r) = st.done.swap_remove(i);
            drop(st);
            return r.expect("a batched decode step failed");
        }
        match g.grant(&st) {
            Grant::Batch => st = g.run_batch(st),
            _ => st = g.wait(st),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate::TextLm;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A model whose "logits" are the token plus the session's position, and that counts the
    /// rows of each batched step.
    struct Toy {
        steps: AtomicUsize,
        widest: AtomicUsize,
        running: AtomicUsize,
    }
    impl crate::generate::TextLm for Toy {
        type Sess = usize;
        fn new_session(&self) -> usize {
            0
        }
        fn hidden_size(&self) -> usize {
            1
        }
        fn n_layers(&self) -> usize {
            1
        }
        fn session_len(&self, s: &usize) -> usize {
            *s
        }
        fn forward_hidden_mm(&self, s: &mut usize, t: &[u32], _: &[(usize, &[f32])]) -> Vec<f32> {
            assert_eq!(self.running.fetch_add(1, Ordering::SeqCst), 0, "two jobs at once");
            std::thread::sleep(Duration::from_micros(300));
            *s += t.len();
            self.running.fetch_sub(1, Ordering::SeqCst);
            vec![(*s + *t.last().unwrap() as usize) as f32]
        }
        fn logits(&self, h: &[f32], _: usize) -> Vec<f32> {
            h.to_vec()
        }
        fn forward_batch(&self, s: &mut [&mut usize], t: &[u32]) -> Vec<Vec<f32>> {
            self.steps.fetch_add(1, Ordering::SeqCst);
            self.widest.fetch_max(t.len(), Ordering::SeqCst);
            assert_eq!(self.running.fetch_add(1, Ordering::SeqCst), 0, "two jobs at once");
            std::thread::sleep(Duration::from_micros(300));
            self.running.fetch_sub(1, Ordering::SeqCst);
            s.iter_mut()
                .zip(t)
                .map(|(s, &t)| {
                    **s += 1;
                    vec![(**s + t as usize) as f32]
                })
                .collect()
        }
    }

    #[test]
    fn concurrent_decoders_batch_and_get_their_own_results() {
        let toy = Arc::new(Toy {
            steps: AtomicUsize::new(0),
            widest: AtomicUsize::new(0),
            running: AtomicUsize::new(0),
        });
        let g = Group::new(BatchOptions {
            max_batch: 4,
            gather_us: 20_000,
            ..Default::default()
        });
        let start = Arc::new(std::sync::Barrier::new(4));
        let hs: Vec<_> = (0..4)
            .map(|k| {
                let (toy, g, start) = (toy.clone(), g.clone(), start.clone());
                std::thread::spawn(move || {
                    start.wait();
                    let _m = enter(&g);
                    // A "prompt" of a few chunks, yielding between them.
                    let mut s = 0usize;
                    for _ in 0..3 {
                        yield_now();
                        toy.forward_hidden_mm(&mut s, &[1], &[]);
                    }
                    let _d = decoding();
                    let mut got = Vec::new();
                    for i in 0..50u32 {
                        let tok = 1000 * k + i;
                        let l = decode_step(&*toy, &mut s, tok);
                        got.push(l[0] as usize);
                        assert_eq!(l[0] as usize, s + tok as usize);
                    }
                    assert_eq!(s, 53);
                    got
                })
            })
            .collect();
        for h in hs {
            h.join().unwrap();
        }
        let st = g.stats();
        assert!(toy.widest.load(Ordering::SeqCst) > 1, "no step was batched: {st:?}");
        assert!(toy.widest.load(Ordering::SeqCst) <= 4);
        assert_eq!(st.rows as usize, toy_rows(&toy, &st));
        assert_eq!(g.members(), 0);
    }

    fn toy_rows(_t: &Toy, st: &GroupStats) -> usize {
        st.rows as usize
    }

    #[test]
    fn prompt_chunks_shrink_while_others_decode() {
        let g = Group::new(BatchOptions {
            decode_chunk: 512,
            ..Default::default()
        });
        assert_eq!(chunk_for(2048), 2048, "outside a group");
        let _m = enter(&g);
        assert_eq!(chunk_for(2048), 2048, "nobody decoding");
        g.lock().decoding += 1;
        assert_eq!(chunk_for(2048), 512);
        assert_eq!(chunk_for(256), 256);
        g.lock().decoding -= 1;
    }

    #[test]
    fn alone_a_step_runs_at_once() {
        let toy = Toy {
            steps: AtomicUsize::new(0),
            widest: AtomicUsize::new(0),
            running: AtomicUsize::new(0),
        };
        let g = Group::new(BatchOptions::default());
        let _m = enter(&g);
        assert!(holding());
        let _d = decoding();
        let mut s = 0usize;
        for i in 0..5 {
            decode_step(&toy, &mut s, i);
            assert!(holding());
        }
        assert_eq!(toy.steps.load(Ordering::SeqCst), 0);
        assert!(!contended());
    }
}
