//! Lightweight always-on profiler: accumulated wall time per category.

use crate::clock::Instant;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub enum Cat {
    Matmul = 0,
    Attention,
    Norm,
    Elementwise,
    Embed,
    Head,
    Other,
}

const NAMES: [&str; 7] = ["matmul", "attention", "norm", "elementwise", "embed+ple", "lm_head", "drafter"];
static NANOS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];
static CALLS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];
static BYTES: AtomicU64 = AtomicU64::new(0);

pub struct Timer(Cat, Instant);

#[inline]
pub fn start(c: Cat) -> Timer {
    Timer(c, Instant::now())
}

impl Drop for Timer {
    #[inline]
    fn drop(&mut self) {
        NANOS[self.0 as usize].fetch_add(self.1.elapsed().as_nanos() as u64, Ordering::Relaxed);
        CALLS[self.0 as usize].fetch_add(1, Ordering::Relaxed);
    }
}

/// Record bytes of weights streamed (for bandwidth reporting).
#[inline]
pub fn add_bytes(n: usize) {
    BYTES.fetch_add(n as u64, Ordering::Relaxed);
}

/// Named phases of a model's forward pass, timed beside the categories above (which they
/// overlap): `--profile` prints the ones used. Slot `i` is named by its first `phase(i, name)`.
const PHASES: usize = 24;
static PHASE_NAMES: [std::sync::OnceLock<&'static str>; PHASES] = [const { std::sync::OnceLock::new() }; PHASES];
static PHASE_NANOS: [AtomicU64; PHASES] = [const { AtomicU64::new(0) }; PHASES];
static PHASE_CALLS: [AtomicU64; PHASES] = [const { AtomicU64::new(0) }; PHASES];

pub struct PhaseTimer(usize, Instant);

/// Times phase `slot` (named `name` in the report) until the timer drops.
#[inline]
pub fn phase(slot: usize, name: &'static str) -> PhaseTimer {
    PHASE_NAMES[slot].get_or_init(|| name);
    PhaseTimer(slot, Instant::now())
}

impl Drop for PhaseTimer {
    #[inline]
    fn drop(&mut self) {
        PHASE_NANOS[self.0].fetch_add(self.1.elapsed().as_nanos() as u64, Ordering::Relaxed);
        PHASE_CALLS[self.0].fetch_add(1, Ordering::Relaxed);
    }
}

pub fn reset() {
    for a in PHASE_NANOS.iter().chain(PHASE_CALLS.iter()) {
        a.store(0, Ordering::Relaxed);
    }
    for a in NANOS.iter().chain(CALLS.iter()) {
        a.store(0, Ordering::Relaxed);
    }
    BYTES.store(0, Ordering::Relaxed);
}

pub fn report(total_secs: f64, tokens: usize) {
    eprintln!(
        "[profile over {tokens} forward tokens, {:.1} ms/token]",
        total_secs * 1e3 / tokens.max(1) as f64
    );
    if let Some(s) = crate::gpu::active().and_then(|b| b.stats()) {
        eprintln!("  gpu: {s}");
    }
    let mut acc = 0f64;
    for i in 0..NAMES.len() {
        let s = NANOS[i].load(Ordering::Relaxed) as f64 * 1e-9;
        let c = CALLS[i].load(Ordering::Relaxed);
        if c == 0 {
            continue;
        }
        acc += s;
        eprintln!(
            "  {:<12} {:>8.1} ms/tok  {:>6} calls/tok  {:>7.1} us/call",
            NAMES[i],
            s * 1e3 / tokens.max(1) as f64,
            c / tokens.max(1) as u64,
            s * 1e6 / c as f64
        );
    }
    let bytes = BYTES.load(Ordering::Relaxed) as f64;
    let mm = NANOS[Cat::Matmul as usize].load(Ordering::Relaxed) as f64 * 1e-9;
    eprintln!(
        "  weights streamed {:.2} GB/tok, matmul bandwidth {:.0} GB/s, accounted {:.0}% of wall",
        bytes / tokens.max(1) as f64 / 1e9,
        bytes / mm.max(1e-9) / 1e9,
        acc / total_secs.max(1e-9) * 100.0
    );
    for i in 0..PHASES {
        let c = PHASE_CALLS[i].load(Ordering::Relaxed);
        let Some(name) = PHASE_NAMES[i].get().filter(|_| c > 0) else {
            continue;
        };
        let s = PHASE_NANOS[i].load(Ordering::Relaxed) as f64 * 1e-9;
        eprintln!(
            "  phase {:<16} {:>6.1} ms/tok  {:>6} calls/tok  {:>7.1} us/call",
            name,
            s * 1e3 / tokens.max(1) as f64,
            c / tokens.max(1) as u64,
            s * 1e6 / c as f64
        );
    }
}
