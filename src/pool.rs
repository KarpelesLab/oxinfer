//! Minimal spin-waiting thread pool for low-latency parallel-for.
//!
//! Workers spin on a generation counter while work is flowing, so dispatching a job
//! costs about a microsecond. After a quiet period they back off to parking.
//! A region ends as soon as every *claimed* block is done; a worker that never
//! joined the region (e.g. because it was preempted) does not hold up the barrier.
//! Inside a [`decode_scope`] regions may use fewer workers (see [`decode_threads`]); the others
//! back off to parking as if no work came.

use crate::clock::Instant;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering::*};
use std::thread::{self, Thread};
use std::time::Duration;

type JobFn<'a> = &'a (dyn Fn(usize) + Sync);

/// One parallel-for, living on the caller's stack for the duration of `run`. The claim counter
/// has a cache line of its own (128 bytes: Apple's lines), so that the claims do not take the
/// job's line away from the threads about to read it.
#[repr(C, align(128))]
struct Region {
    f: JobFn<'static>,
    n: usize,
    next: Padded<AtomicUsize>,
}

/// A value alone on its cache line.
#[repr(align(128))]
struct Padded<T>(T);

impl<T> std::ops::Deref for Padded<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

struct Shared {
    /// What the waiting workers read, on a line of its own: the generation, the region and how
    /// many workers it takes.
    generation: Padded<AtomicU64>,
    region: AtomicPtr<Region>,
    workers: AtomicUsize,
    exit: AtomicBool,
    sleepers: Padded<AtomicUsize>,
    /// Per worker, on its own line: set while it may be inside a region. (One shared counter
    /// that every worker incremented on joining and decremented on leaving queued 17
    /// read-modify-writes on one line per region.)
    busy: Box<[Padded<AtomicBool>]>,
}

pub struct Pool {
    shared: Arc<Shared>,
    handles: Vec<Thread>,
    workers: usize,
    opts: PoolOptions,
}

/// Scheduling class of compute threads. On Apple platforms this is the thread's QoS class,
/// which also picks the cores: `Background` threads run on the efficiency cores only, the other
/// classes on any core (performance cores first). Elsewhere it is ignored.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Qos {
    /// The default: as urgent as a UI thread.
    #[default]
    UserInteractive,
    UserInitiated,
    Utility,
    /// Efficiency cores only (Apple), lowest priority.
    Background,
}

/// Thread pool settings (see [`init_with`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolOptions {
    /// Threads in total, the calling thread included.
    pub threads: usize,
    /// QoS of the calling thread and the workers.
    pub qos: Qos,
    /// Of the workers, how many run at `Qos::Background` instead (kept on the efficiency cores
    /// on Apple platforms: with work claimed block by block they add their throughput without
    /// the performance cores waiting for them).
    pub efficiency_threads: usize,
}

impl PoolOptions {
    /// `threads` threads at the default QoS.
    pub fn threads(threads: usize) -> Self {
        Self {
            threads,
            qos: Qos::UserInteractive,
            efficiency_threads: 0,
        }
    }
}

/// Cores by kind: `(performance, efficiency)` physical cores (Apple: `hw.perflevel0/1`; other
/// platforms report every logical core as a performance core).
pub fn cpu_cores() -> (usize, usize) {
    let all = thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        unsafe extern "C" {
            fn sysctlbyname(
                name: *const i8,
                old: *mut std::ffi::c_void,
                oldlen: *mut usize,
                new: *const std::ffi::c_void,
                newlen: usize,
            ) -> i32;
        }
        let get = |name: &std::ffi::CStr| -> Option<usize> {
            let mut v = 0i32;
            let mut len = std::mem::size_of::<i32>();
            let r = unsafe { sysctlbyname(name.as_ptr(), (&mut v as *mut i32).cast(), &mut len, std::ptr::null(), 0) };
            (r == 0 && v > 0).then_some(v as usize)
        };
        if let Some(p) = get(c"hw.perflevel0.physicalcpu") {
            return (p, get(c"hw.perflevel1.physicalcpu").unwrap_or(0));
        }
    }
    (all, 0)
}

/// Threads the pool takes by default: one per physical core where logical CPUs are SMT siblings
/// (Linux, from sysfs), else every logical CPU. The second thread of a core adds nothing to
/// kernels bound by memory bandwidth or by the FMA units, and halves the cache each one has.
pub fn default_threads() -> usize {
    let all = thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    #[cfg(any(target_os = "linux", target_os = "fullrust"))]
    if let Some(cores) = linux_physical_cores(all) {
        return cores;
    }
    all
}

/// The number of physical cores among the CPUs this process may run on, when it differs from
/// `all` (the logical CPUs available): the distinct `thread_siblings_list` of the allowed CPUs.
/// None when anything is unreadable or a quota (not the affinity mask) limits the process.
#[cfg(any(target_os = "linux", target_os = "fullrust"))]
fn linux_physical_cores(all: usize) -> Option<usize> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let list = status.lines().find_map(|l| l.strip_prefix("Cpus_allowed_list:"))?.trim();
    let mut cpus = Vec::new();
    for part in list.split(',') {
        let (a, b) = part.split_once('-').unwrap_or((part, part));
        cpus.extend(a.trim().parse::<usize>().ok()?..=b.trim().parse::<usize>().ok()?);
    }
    if cpus.len() != all {
        return None;
    }
    let mut cores = std::collections::BTreeSet::new();
    for c in cpus {
        cores.insert(std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list")).ok()?);
    }
    (!cores.is_empty() && cores.len() < all).then_some(cores.len())
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn set_qos(q: Qos) {
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    }
    let class = match q {
        Qos::UserInteractive => 0x21,
        Qos::UserInitiated => 0x19,
        Qos::Utility => 0x11,
        Qos::Background => 0x09,
    };
    unsafe {
        pthread_set_qos_class_self_np(class, 0);
    }
}
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
fn set_qos(_q: Qos) {}

/// QoS of the pool most recently created (the reader threads of weight streaming use it).
static POOL_QOS: AtomicUsize = AtomicUsize::new(0);

/// Gives the calling thread the compute threads' QoS (default `Qos::UserInteractive`).
pub(crate) fn raise_qos() {
    set_qos(match POOL_QOS.load(Relaxed) {
        1 => Qos::UserInitiated,
        2 => Qos::Utility,
        3 => Qos::Background,
        _ => Qos::UserInteractive,
    });
}

thread_local! {
    /// Set while this thread runs a pool job: a nested `Pool::run` would wait on itself.
    // Already const; clippy (1.100) still flags it on Android's thread-local implementation.
    #[allow(clippy::missing_const_for_thread_local)]
    static IN_JOB: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs `f` with this thread marked as inside a job.
#[inline]
fn in_job(f: impl FnOnce()) {
    let prev = IN_JOB.with(|c| c.replace(true));
    f();
    IN_JOB.with(|c| c.set(prev));
}

/// Claims and runs blocks of the current region, if any. Returns false if no region was published.
#[inline]
fn participate(shared: &Shared, id: usize) -> bool {
    let busy = &shared.busy[id];
    busy.store(true, SeqCst);
    let rp = shared.region.load(SeqCst);
    if rp.is_null() {
        busy.store(false, Release);
        return false;
    }
    // SAFETY: the publisher keeps the region alive until it has cleared the pointer and then
    // seen every `busy` flag clear, and this one was set before the pointer was loaded (both
    // sequentially consistent: either the publisher sees the flag or we see the null).
    let region = unsafe { &*rp };
    in_job(|| {
        loop {
            let i = region.next.fetch_add(1, Relaxed);
            if i >= region.n {
                break;
            }
            (region.f)(i);
        }
    });
    busy.store(false, Release);
    true
}

fn worker(shared: Arc<Shared>, qos: Qos, id: usize) {
    set_qos(qos);
    let mut seen = 0u64;
    loop {
        // Wait for a new generation: spin, then yield, then park with growing timeouts.
        let mut spins = 0u32;
        let mut t0: Option<Instant> = None;
        let mut park = Duration::from_micros(100);
        loop {
            if shared.exit.load(Acquire) {
                return;
            }
            let g = shared.generation.load(Acquire);
            if g != seen {
                seen = g;
                // A region for fewer workers is not for this one, which goes on backing off.
                if id < shared.workers.load(Relaxed) {
                    break;
                }
            }
            spins += 1;
            if spins < 4096 {
                std::hint::spin_loop();
                continue;
            }
            let now = Instant::now();
            let start = *t0.get_or_insert(now);
            if now.duration_since(start) < Duration::from_millis(3) {
                thread::yield_now();
                continue;
            }
            shared.sleepers.fetch_add(1, SeqCst);
            if shared.generation.load(SeqCst) == seen && !shared.exit.load(SeqCst) {
                thread::park_timeout(park);
                park = (park * 2).min(Duration::from_millis(10));
            }
            shared.sleepers.fetch_sub(1, SeqCst);
        }
        participate(&shared, id);
    }
}

impl Pool {
    /// A pool using `threads` threads in total (the caller counts as one).
    pub fn new(threads: usize) -> Self {
        Self::with(PoolOptions::threads(threads))
    }

    /// A pool with the given settings.
    pub fn with(o: PoolOptions) -> Self {
        let threads = o.threads;
        POOL_QOS.store(o.qos as usize, Relaxed);
        set_qos(o.qos);
        let shared = Arc::new(Shared {
            generation: Padded(AtomicU64::new(0)),
            region: AtomicPtr::new(std::ptr::null_mut()),
            workers: AtomicUsize::new(usize::MAX),
            exit: AtomicBool::new(false),
            sleepers: Padded(AtomicUsize::new(0)),
            busy: (0..threads.max(1)).map(|_| Padded(AtomicBool::new(false))).collect(),
        });
        // Spawning can fail where the platform has no threads (wasm32 outside wasi-threads, where
        // `spawn` returns Unsupported); whatever was spawned is the pool, the rest runs inline.
        let mut handles: Vec<thread::Thread> = Vec::with_capacity(threads.max(1) - 1);
        let n = threads.max(1) - 1;
        for i in 0..n {
            let s = shared.clone();
            // The last `efficiency_threads` workers run at background QoS.
            let q = if i + o.efficiency_threads.min(n) >= n {
                Qos::Background
            } else {
                o.qos
            };
            match thread::Builder::new().name(format!("oxinfer-{i}")).spawn(move || worker(s, q, i)) {
                Ok(h) => handles.push(h.thread().clone()),
                Err(_) => break,
            }
        }
        let workers = handles.len();
        Self {
            shared,
            handles,
            workers,
            opts: o,
        }
    }

    pub fn threads(&self) -> usize {
        self.workers + 1
    }

    /// Runs `f(i)` for every `i in 0..n` across the pool, returning when all are done.
    /// From inside a job it runs serially on the calling thread.
    pub fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync)) {
        if n == 0 {
            return;
        }
        let nested = IN_JOB.with(|c| c.get());
        if nested && std::env::var_os("OXINFER_POOL_DEBUG").is_some() {
            eprintln!("[pool: nested run]\n{}", std::backtrace::Backtrace::force_capture());
        }
        // Inside a decode scope: that many threads, the caller included.
        let workers = match DECODE.with(|d| d.get()) {
            0 => self.workers,
            t => t.min(self.workers + 1) - 1,
        };
        if n == 1 || workers == 0 || nested {
            for i in 0..n {
                f(i);
            }
            return;
        }
        REGIONS.fetch_add(1, Relaxed);
        // The caller's prompt-processing flag (`kernels::lm_prefill`) holds in the workers too.
        let lm = crate::kernels::in_lm_prefill();
        let scoped = move |i: usize| crate::kernels::with_lm_prefill(lm, || f(i));
        let f: &(dyn Fn(usize) + Sync) = &scoped;
        let s = &self.shared;
        // SAFETY: the region outlives every access: the pointer is cleared and every `busy`
        // flag observed clear before this function returns.
        let region = Region {
            f: unsafe { std::mem::transmute::<JobFn<'_>, JobFn<'static>>(f) },
            n,
            next: Padded(AtomicUsize::new(0)),
        };
        // (Any value of `workers` a worker reads is safe: whoever joins is waited for below.)
        s.workers.store(workers, Relaxed);
        s.region.store(&region as *const Region as *mut Region, SeqCst);
        s.generation.fetch_add(1, SeqCst);
        if s.sleepers.load(SeqCst) > 0 {
            for h in &self.handles[..workers] {
                h.unpark();
            }
        }
        in_job(|| {
            loop {
                let i = region.next.fetch_add(1, Relaxed);
                if i >= n {
                    break;
                }
                f(i);
            }
        });
        s.region.store(std::ptr::null_mut(), SeqCst);
        // Every worker, of this region's or not, that may have loaded the pointer has left.
        for b in &s.busy[..self.workers] {
            while b.load(SeqCst) {
                std::hint::spin_loop();
            }
        }
    }

    pub fn shutdown(&self) {
        self.shared.exit.store(true, SeqCst);
        for h in &self.handles {
            h.unpark();
        }
    }
}

/// Parallel regions run so far (by any pool; runs inline on the caller are not counted).
static REGIONS: AtomicU64 = AtomicU64::new(0);

/// The number of parallel regions run so far (see [`Pool::run`]), for profiling.
pub fn regions() -> u64 {
    REGIONS.load(Relaxed)
}

thread_local! {
    /// Inside a [`decode_scope`] on this (the calling) thread: the threads its regions use.
    // Already const; clippy (1.100) still flags it on Android's thread-local implementation.
    #[allow(clippy::missing_const_for_thread_local)]
    static DECODE: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Threads of a decode step (0: automatic; `usize::MAX`: not read from the environment yet).
static DECODE_THREADS: AtomicUsize = AtomicUsize::new(usize::MAX);

/// Sets the threads the regions of a [`decode_scope`] use, the caller included (at most the
/// pool's), or 0 (the default; also `OXINFER_DECODE_THREADS`) for [`decode_threads`]'s choice.
pub fn set_decode_threads(n: usize) {
    DECODE_THREADS.store(n, Relaxed);
}

/// The threads the regions of a [`decode_scope`] use with a pool of `threads`: those set by
/// [`set_decode_threads`] or `OXINFER_DECODE_THREADS`, else all but two from 12 threads on.
///
/// A one-token step is about a hundred short regions bound by memory bandwidth, and on a machine
/// that also runs other work, one of all-the-cores' threads losing its core in the middle of a
/// region holds the region up. On an 18-core M5 Max shared with other jobs (Llama 3.2 1B Q8_0),
/// 16 threads decoded about 2% faster than 18 on average over seven comparisons (from 1.4% slower
/// to 4.9% faster); the fastest count moved between 12 and 18 with the load, and a tuner
/// choosing among 12-18 by measured step times was too noisy to beat the fixed choice.
pub fn decode_threads(threads: usize) -> usize {
    let mut n = DECODE_THREADS.load(Relaxed);
    if n == usize::MAX {
        n = std::env::var("OXINFER_DECODE_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        DECODE_THREADS.store(n, Relaxed);
    }
    match n {
        0 if threads >= 12 => threads - 2,
        0 => threads,
        n => n.min(threads),
    }
}

/// While the guard lives, the parallel regions this thread starts use [`decode_threads`]
/// threads: for the steps of decoding loops (prompts keep every thread).
pub fn decode_scope() -> DecodeScope {
    let n = decode_threads(threads());
    DecodeScope(DECODE.with(|d| d.replace(n)))
}

/// See [`decode_scope`].
pub struct DecodeScope(usize);

impl Drop for DecodeScope {
    fn drop(&mut self) {
        DECODE.with(|d| d.set(self.0));
    }
}

static GLOBAL: AtomicPtr<Pool> = AtomicPtr::new(std::ptr::null_mut());

/// Installs a global pool with `threads` threads (replacing any previous one).
pub fn init(threads: usize) {
    init_with(PoolOptions::threads(threads))
}

/// Installs a global pool with the given settings (replacing any previous one). For phones,
/// e.g. an iPhone's 2 performance + 4 efficiency cores: `threads: 6` uses all of them (work is
/// claimed block by block, so slower cores take fewer blocks), `threads: 2` only the
/// performance cores, and `efficiency_threads` pins some workers to the efficiency cores.
pub fn init_with(o: PoolOptions) {
    let p = Box::into_raw(Box::new(Pool::with(o)));
    let old = GLOBAL.swap(p, SeqCst);
    if !old.is_null() {
        // Leaked on purpose: in-flight references may still exist.
        unsafe { (*old).shutdown() };
    }
}

/// Uses `threads` threads: installs a new global pool unless the current one has that many.
pub fn configure(threads: usize) {
    let p = GLOBAL.load(Acquire);
    if p.is_null() || unsafe { (*p).threads() } != threads.max(1) {
        init(threads);
    }
}

/// Uses these settings: installs a new global pool unless the current one has them.
pub fn configure_with(o: PoolOptions) {
    let p = GLOBAL.load(Acquire);
    if p.is_null() || unsafe { (*p).opts } != o {
        init_with(o);
    }
}

/// The settings of the global pool.
pub fn options() -> PoolOptions {
    get().opts
}

pub fn get() -> &'static Pool {
    let p = GLOBAL.load(Acquire);
    if p.is_null() {
        init(default_threads());
        return get();
    }
    unsafe { &*p }
}

#[inline]
pub fn run(n: usize, f: &(dyn Fn(usize) + Sync)) {
    get().run(n, f)
}

pub fn threads() -> usize {
    get().threads()
}

#[cfg(test)]
mod tests {
    /// The default is at least one thread and at most the logical CPUs (one per core with SMT).
    #[test]
    fn default_threads_is_within_the_logical_cpus() {
        let all = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        let d = super::default_threads();
        assert!((1..=all).contains(&d), "{d} threads for {all} logical CPUs");
        eprintln!("default threads: {d} of {all} logical CPUs");
    }
}
