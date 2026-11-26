//! The cost model shared by the speculative decoding policies (DSpark's [`crate::dspark::Adaptive`],
//! prompt lookup's [`crate::lookup::Policy`], MTP's [`crate::qwen_mtp::DraftCosts`] and the Gemma 4
//! assistant's [`ChainPolicy`]): what verifying `k` drafts costs, as a multiple of a plain
//! one-token step.
//!
//! Each policy measures the verification passes it runs (a moving average per number of drafts),
//! and a length it has not measured is estimated from the measured ones through the shape of a
//! small prior table, keyed on where the passes run ([`VerifyBackend`]): a pass of `t` tokens on
//! the exact multi-token matvecs of the Metal decode library costs far more per token than on its
//! fast batched kernels, and the CPU's are different again. A policy therefore needs only a plain
//! step measured to price every length, and its first measurement of a length replaces the
//! guess. The tables are measured on an M5 Max (Qwen3-8B Q4_K_M and Q8_0 verification passes,
//! the mean of the two formats; see docs/performance.md, `OXINFER_SPEC_SWEEP`), the CUDA one is
//! a guess (no measurement): online measurements correct them within a few rounds of each
//! length.

/// Where verification passes run, for [`CostPrior::detect`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerifyBackend {
    /// Metal decode plans, exact multi-token matvecs (`fast_verify` off).
    MetalExact,
    /// Metal decode plans, fast batched kernels (`fast_verify`, the default).
    MetalFast,
    /// CUDA decode plans.
    Cuda,
    /// The CPU kernels (also any other backend without decode plans).
    Cpu,
}

impl VerifyBackend {
    /// The backend verification passes run on in this process, with fast verification asked
    /// for (`fast`) or not.
    pub fn detect(fast: bool) -> Self {
        match crate::gpu::active() {
            Some(b) if crate::gpu::decode_enabled() && b.decodes() => {
                if b.name() == "metal" {
                    if fast { Self::MetalFast } else { Self::MetalExact }
                } else if b.name().starts_with("cuda") {
                    Self::Cuda
                } else {
                    Self::Cpu
                }
            }
            _ => Self::Cpu,
        }
    }
}

/// Prior cost of a verification pass of `t` tokens (`TABLE[t - 1]`, t = 1 to 16) as a multiple
/// of a one-token step (see the module docs). Measured up to 8 tokens (the mean of Qwen3-8B
/// Q4_K_M and Q8_0, whose own ratios differ by up to 1.6 times: Q4_K's exact multi-token rows
/// cost the most per token), extended at the last measured growth per token beyond.
const METAL_EXACT: [f64; 16] = [
    1.0, 1.15, 1.36, 1.63, 2.0, 2.4, 2.87, 3.39, 3.94, 4.49, 5.04, 5.59, 6.14, 6.69, 7.24, 7.79,
];
/// The fast batched kernels: a step from one token to two (the matrix kernels replace the
/// one-token ones; Q4_K_M 1.92, Q8_0 1.26), then nearly flat up to 8 tokens; beyond, a guess.
const METAL_FAST: [f64; 16] = [
    1.0, 1.59, 1.61, 1.62, 1.63, 1.64, 1.66, 1.68, 1.9, 1.97, 2.04, 2.11, 2.18, 2.25, 2.32, 2.4,
];
/// Not measured: the Metal exact shape.
const CUDA: [f64; 16] = METAL_EXACT;
/// The CPU's batch-invariant decode kernels (8 threads, Q4_K_M and Q8_0 as above).
const CPU: [f64; 16] = [
    1.0, 1.05, 1.35, 1.53, 1.88, 2.06, 2.41, 2.57, 2.87, 3.17, 3.47, 3.77, 4.07, 4.37, 4.67, 4.97,
];

/// The prior shape of verification costs by token count ([`VerifyBackend`]'s table).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CostPrior {
    pub backend: VerifyBackend,
}

impl Default for CostPrior {
    /// The prior of the active backend with fast verification (the default).
    fn default() -> Self {
        Self::detect(true)
    }
}

impl CostPrior {
    pub fn new(backend: VerifyBackend) -> Self {
        Self { backend }
    }

    /// [`VerifyBackend::detect`]'s prior.
    pub fn detect(fast: bool) -> Self {
        Self::new(VerifyBackend::detect(fast))
    }

    fn table(&self) -> &'static [f64; 16] {
        match self.backend {
            VerifyBackend::MetalExact => &METAL_EXACT,
            VerifyBackend::MetalFast => &METAL_FAST,
            VerifyBackend::Cuda => &CUDA,
            VerifyBackend::Cpu => &CPU,
        }
    }

    /// A pass of `t` tokens (at least 1) as a multiple of a one-token step; beyond the table,
    /// its last step per token.
    pub fn ratio(&self, t: usize) -> f64 {
        let r = self.table();
        let t = t.max(1);
        if t <= r.len() {
            return r[t - 1];
        }
        let n = r.len();
        r[n - 1] + (r[n - 1] - r[n - 2]) * (t - n) as f64
    }

    /// The time of verifying `k` drafts (a pass of `k + 1` tokens) from measured times
    /// `measured[j] = (moving average, count)` of `j` drafts: the measurement, else between the
    /// nearest measured lengths in the prior's shape, else the nearest one scaled by the prior.
    /// None when nothing is measured.
    pub fn verify_time(&self, measured: &[(f64, usize)], k: usize) -> Option<f64> {
        if let Some(&(t, n)) = measured.get(k)
            && n > 0
        {
            return Some(t);
        }
        let mut lo = None;
        let mut hi = None;
        for (j, e) in measured.iter().enumerate().filter(|e| e.1.1 > 0) {
            if j < k {
                lo = Some((j, e.0));
            } else if hi.is_none() {
                hi = Some((j, e.0));
            }
        }
        let r = |j: usize| self.ratio(j + 1);
        match (lo, hi) {
            (Some(a), Some(b)) => {
                let f = (r(k) - r(a.0)) / (r(b.0) - r(a.0)).max(1e-9);
                Some(a.1 + (b.1 - a.1) * f)
            }
            (Some(a), None) | (None, Some(a)) => Some(a.1 * r(k) / r(a.0)),
            (None, None) => None,
        }
    }
}

/// A decayed acceptance count per draft position: `(accepted, verified)` of the drafts at each
/// position that were verified after the ones before them were accepted.
#[derive(Clone, Debug, Default)]
pub struct SlotRates {
    pub slots: Vec<(f64, f64)>,
}

impl SlotRates {
    /// Records a round that verified `verified` drafts and accepted the first `accepted`.
    pub fn record(&mut self, verified: usize, accepted: usize, decay: f64) {
        if self.slots.len() < verified {
            self.slots.resize(verified, (0.0, 0.0));
        }
        for (j, s) in self.slots.iter_mut().enumerate() {
            s.0 *= decay;
            s.1 *= decay;
            if j < verified.min(accepted + 1) {
                s.0 += f64::from(u8::from(j < accepted));
                s.1 += 1.0;
            }
        }
    }

    /// Verified drafts recorded at the first position (decayed).
    pub fn weight(&self) -> f64 {
        self.slots.first().map_or(0.0, |s| s.1)
    }

    /// Expected tokens a round with `k` drafts commits (the target's own token included): each
    /// position's rate given the ones before it were accepted, with one pseudo-count at the
    /// previous position's rate (`first` before the first), so that a position rarely reached
    /// keeps a rate near the one before it.
    pub fn expected(&self, k: usize, first: f64) -> f64 {
        let (mut e, mut p, mut rate) = (1.0, 1.0, first);
        for j in 0..k {
            let (a, n) = self.slots.get(j).copied().unwrap_or((0.0, 0.0));
            rate = (a + rate) / (n + 1.0);
            p *= rate;
            e += p;
        }
        e
    }
}

/// The draft length policy of a drafter that drafts token after token (the Gemma 4 assistant,
/// whose `k` drafts cost `k` steps): each round, the length (0 = a plain step) with the most
/// expected tokens per second, from the measured cost of a draft step and of verifying each
/// length ([`CostPrior::verify_time`]) and the running acceptance rate of each draft position.
/// While plain steps win it drafts again now and then, as rarely as keeps the time spent on
/// such probes within [`ChainPolicy::PROBE_BUDGET`] of the plain steps' time; while drafting,
/// one round in 8 drafts one more than the best length, so that the next length's cost and
/// acceptance stay measured.
#[derive(Clone, Debug)]
pub struct ChainPolicy {
    pub prior: CostPrior,
    /// Moving average of verifying (and committing) `k` drafts, with the number of samples.
    pub verify: Vec<(f64, usize)>,
    /// Moving average of one draft step.
    pub step: (f64, usize),
    pub rates: SlotRates,
    /// Plain rounds since the last round with drafts.
    pub plain_run: usize,
    /// Rounds with drafts so far.
    pub rounds: usize,
}

impl ChainPolicy {
    const DECAY: f64 = 0.8;
    /// Decay of the per-position acceptance counts per round with drafts (a window of about 30
    /// rounds).
    const RATE_DECAY: f64 = 0.97;
    /// Share of the plain steps' time the probes of a drafter that loses may cost.
    pub const PROBE_BUDGET: f64 = 0.02;
    /// Acceptance assumed at the first position before any is measured.
    const FIRST: f64 = 0.6;

    pub fn new(prior: CostPrior) -> Self {
        Self {
            prior,
            verify: Vec::new(),
            step: (0.0, 0),
            rates: SlotRates::default(),
            plain_run: 0,
            rounds: 0,
        }
    }

    /// A moving average whose second sample replaces the first (which may include compiling a
    /// GPU plan or warming caches).
    pub fn ema(e: &mut (f64, usize), v: f64) {
        *e = (if e.1 < 2 { v } else { Self::DECAY * e.0 + (1.0 - Self::DECAY) * v }, e.1 + 1);
    }

    /// Records a round: `k` drafts (taking `draft_secs` in all), `accepted` of them, verified in
    /// `verify_secs`.
    pub fn record(&mut self, k: usize, accepted: usize, draft_secs: f64, verify_secs: f64) {
        if self.verify.len() <= k {
            self.verify.resize(k + 1, (0.0, 0));
        }
        Self::ema(&mut self.verify[k], verify_secs);
        if k == 0 {
            self.plain_run += 1;
            return;
        }
        Self::ema(&mut self.step, draft_secs / k as f64);
        self.rates.record(k, accepted, Self::RATE_DECAY);
        self.plain_run = 0;
        self.rounds += 1;
    }

    /// Expected tokens per second of a round with `k` drafts (None: not measured enough).
    fn rate(&self, k: usize) -> Option<f64> {
        let v = self.prior.verify_time(&self.verify, k)?;
        let d = if k > 0 { self.step.0 * k as f64 } else { 0.0 };
        Some(self.rates.expected(k, Self::FIRST) / (d + v).max(1e-9))
    }

    /// The next round's number of drafts, 0 to `k_max`: `k_start` until a round with drafts
    /// and a plain step have been measured twice each.
    pub fn choose(&self, k_max: usize, k_start: usize) -> usize {
        if k_max == 0 {
            return 0;
        }
        if self.step.1 < 2 {
            return k_start.clamp(1, k_max);
        }
        if self.verify.first().is_none_or(|e| e.1 < 2) {
            return 0;
        }
        let mut best = (0, 0.0);
        for k in 0..=k_max {
            let Some(r) = self.rate(k) else { return k_start.clamp(1, k_max) };
            if r > best.1 {
                best = (k, r);
            }
        }
        if best.0 > 0 {
            return if self.rounds % 8 == 7 { (best.0 + 1).min(k_max) } else { best.0 };
        }
        // Plain steps win: probe with the length that loses least, once the plain steps since
        // the last probe have earned its expected loss.
        let plain = self.verify[0].0;
        let (mut k_probe, mut loss) = (1, f64::INFINITY);
        for k in 1..=k_max {
            let v = self.prior.verify_time(&self.verify, k).unwrap_or(plain);
            let l = self.step.0 * k as f64 + v - self.rates.expected(k, Self::FIRST) * plain;
            if l < loss {
                (k_probe, loss) = (k, l);
            }
        }
        if probe_due(self.plain_run, loss, plain, self.rounds) {
            k_probe
        } else {
            0
        }
    }
}

/// Whether `plain_run` plain steps of `plain` seconds have earned a probe expected to lose
/// `loss` seconds against plain steps, at [`ChainPolicy::PROBE_BUDGET`] (at least every 4 and at
/// most every 256 steps), when the loss was estimated from `samples` rounds with drafts: at most
/// every `4 * samples` steps, since a loss measured on a few rounds (the first of which may also
/// compile a GPU plan) is no reason to stop looking.
pub fn probe_due(plain_run: usize, loss: f64, plain: f64, samples: usize) -> bool {
    probe_due_every(plain_run, loss, plain, samples, 4)
}

/// [`probe_due`] with at most every `per_sample * samples` steps (at least every `per_sample`)
/// instead of 4: a drafter whose probes cost a whole draft pass explores less eagerly.
pub fn probe_due_every(plain_run: usize, loss: f64, plain: f64, samples: usize, per_sample: usize) -> bool {
    let every = (loss.max(0.0) / (ChainPolicy::PROBE_BUDGET * plain).max(1e-12)).ceil();
    let most = (per_sample * samples).clamp(per_sample, 256) as f64;
    plain_run as f64 >= every.clamp(4.0, most.max(4.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_time_follows_the_prior_between_and_beyond_measurements() {
        let p = CostPrior::new(VerifyBackend::MetalExact);
        let mut m = vec![(0.0, 0); 8];
        assert_eq!(p.verify_time(&m, 3), None);
        m[0] = (10.0, 3);
        // One plain step measured: every length from the prior.
        assert!((p.verify_time(&m, 3).unwrap() - 10.0 * p.ratio(4)).abs() < 1e-9);
        m[7] = (40.0, 2);
        let v3 = p.verify_time(&m, 3).unwrap();
        assert!(v3 > 10.0 && v3 < 40.0);
        // Measured lengths are returned as they are; beyond the last one, scaled.
        assert_eq!(p.verify_time(&m, 7), Some(40.0));
        let v9 = p.verify_time(&m[..8], 9).unwrap();
        assert!((v9 - 40.0 * p.ratio(10) / p.ratio(8)).abs() < 1e-9);
    }

    #[test]
    fn chain_policy_drafts_when_accepted_and_probes_rarely_when_not() {
        let mut c = ChainPolicy::new(CostPrior::new(VerifyBackend::MetalFast));
        // Cheap drafts, nearly always accepted: drafts.
        for _ in 0..10 {
            c.record(3, 3, 0.003, 0.012);
            c.record(0, 0, 0.0, 0.010);
        }
        assert!(c.choose(6, 3) >= 3);
        // Drafts always rejected: plain steps, with probes at most at the budget.
        let mut c = ChainPolicy::new(CostPrior::new(VerifyBackend::MetalFast));
        for _ in 0..40 {
            c.record(2, 0, 0.004, 0.013);
        }
        for _ in 0..3 {
            c.record(0, 0, 0.0, 0.010);
        }
        let mut probes = 0;
        for _ in 0..400 {
            let k = c.choose(6, 3);
            if k > 0 {
                probes += 1;
                c.record(k, 0, 0.002 * k as f64, 0.013);
            } else {
                c.record(0, 0, 0.0, 0.010);
            }
        }
        assert!(probes > 0 && probes < 20, "{probes} probes");
    }
}
