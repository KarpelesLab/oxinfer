//! Minimal spin-waiting thread pool for low-latency parallel-for.
//!
//! Workers spin on a generation counter while work is flowing, so dispatching a job
//! costs about a microsecond. After a quiet period they back off to parking.
//! A region ends as soon as every *claimed* block is done; a worker that never
//! joined the region (e.g. because it was preempted) does not hold up the barrier.

use crate::clock::Instant;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering::*};
use std::thread::{self, Thread};
use std::time::Duration;

type JobFn<'a> = &'a (dyn Fn(usize) + Sync);

/// One parallel-for, living on the caller's stack for the duration of `run`.
struct Region {
    f: JobFn<'static>,
    n: usize,
    next: AtomicUsize,
}

struct Shared {
    generation: AtomicU64,
    region: AtomicPtr<Region>,
    active: AtomicUsize,
    sleepers: AtomicUsize,
    exit: AtomicBool,
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

/// Claims and runs blocks of the current region, if any. Returns false if no region was published.
#[inline]
fn participate(shared: &Shared) -> bool {
    shared.active.fetch_add(1, SeqCst);
    let rp = shared.region.load(SeqCst);
    if rp.is_null() {
        shared.active.fetch_sub(1, SeqCst);
        return false;
    }
    // SAFETY: the publisher keeps the region alive until `active` drops to zero after
    // it has cleared the pointer, and we incremented `active` before loading it.
    let region = unsafe { &*rp };
    loop {
        let i = region.next.fetch_add(1, Relaxed);
        if i >= region.n {
            break;
        }
        (region.f)(i);
    }
    shared.active.fetch_sub(1, SeqCst);
    true
}

fn worker(shared: Arc<Shared>, qos: Qos) {
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
                break;
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
        participate(&shared);
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
            generation: AtomicU64::new(0),
            region: AtomicPtr::new(std::ptr::null_mut()),
            active: AtomicUsize::new(0),
            sleepers: AtomicUsize::new(0),
            exit: AtomicBool::new(false),
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
            match thread::Builder::new().name(format!("oxinfer-{i}")).spawn(move || worker(s, q)) {
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
    /// Must not be called from inside a job.
    pub fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync)) {
        if n == 0 {
            return;
        }
        if n == 1 || self.workers == 0 {
            for i in 0..n {
                f(i);
            }
            return;
        }
        let s = &self.shared;
        // SAFETY: the region outlives every access: the pointer is cleared and `active`
        // observed zero before this function returns.
        let region = Region {
            f: unsafe { std::mem::transmute::<JobFn<'_>, JobFn<'static>>(f) },
            n,
            next: AtomicUsize::new(0),
        };
        s.region.store(&region as *const Region as *mut Region, SeqCst);
        s.generation.fetch_add(1, SeqCst);
        if s.sleepers.load(SeqCst) > 0 {
            for h in &self.handles {
                h.unpark();
            }
        }
        loop {
            let i = region.next.fetch_add(1, Relaxed);
            if i >= n {
                break;
            }
            f(i);
        }
        s.region.store(std::ptr::null_mut(), SeqCst);
        while s.active.load(SeqCst) != 0 {
            std::hint::spin_loop();
        }
    }

    pub fn shutdown(&self) {
        self.shared.exit.store(true, SeqCst);
        for h in &self.handles {
            h.unpark();
        }
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
        init(thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
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
