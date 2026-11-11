//! Step caches of diffusion transformers: skipping the transformer blocks on some denoising
//! steps and standing in a cheap estimate. Both change the result by design, so the pipelines
//! only use them when asked.
//!
//! - [`TeaCache`] (ali-vilab/TeaCache, as wired into the official `OmniGen2Transformer2DModel`):
//!   the relative L1 change of the first block's modulated input since the previous step,
//!   rescaled by a fitted polynomial, accumulates; while the sum stays under a threshold the
//!   blocks are skipped and the previous step's residual (blocks' output minus input) is added
//!   to the input instead.
//! - [`TaylorSeer`] (Shenyi-Z/TaylorSeer, as in the official OmniGen2 and BAGEL code): the blocks
//!   run every `interval`-th step (and on each of the first `first_enhance` steps); finite
//!   differences of the hidden states between those steps give derivatives up to `max_order`,
//!   and the steps in between take the Taylor expansion.
//!
//! The references keep the features of every layer. On a skipped step every layer's output is
//! replaced by its own prediction, so only the last layer's features reach the output: that is
//! the one feature kept here, the same arithmetic on the same values.

/// Which step cache a pipeline uses.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum StepCache {
    /// Every step runs the transformer (the default).
    #[default]
    None,
    /// [`TeaCache`] with this accumulated relative-L1 threshold (the reference's default: 0.05).
    TeaCache { rel_l1_thresh: f32 },
    /// [`TaylorSeer`].
    TaylorSeer(TaylorSeerConfig),
}

impl StepCache {
    pub fn is_none(&self) -> bool {
        *self == StepCache::None
    }
}

/// The coefficients of OmniGen2's rescale polynomial (`np.poly1d`, highest degree first).
pub const OMNIGEN2_TEACACHE_POLY: [f64; 5] = [-5.48259225, 11.48772289, -4.47407401, 2.47730926, -0.03316487];

/// `np.poly1d(coeffs)(x)`.
pub fn poly(coeffs: &[f64], x: f64) -> f64 {
    coeffs.iter().fold(0.0, |acc, &c| acc * x + c)
}

/// TeaCache state of one guidance branch.
#[derive(Debug, Default)]
pub struct TeaCache {
    pub rel_l1_thresh: f64,
    coeffs: Vec<f64>,
    accumulated: f64,
    prev_modulated: Option<Vec<f32>>,
    residual: Option<Vec<f32>>,
    /// The rescaled distance of the last decision (for logs and checks).
    pub last_distance: f64,
}

impl TeaCache {
    pub fn new(rel_l1_thresh: f32, coeffs: &[f64]) -> Self {
        Self {
            rel_l1_thresh: rel_l1_thresh as f64,
            coeffs: coeffs.to_vec(),
            ..Default::default()
        }
    }

    /// Whether the blocks must run this step. `modulated` is the first block's modulated input
    /// (every token of the pass); `first_or_last` forces a run.
    pub fn should_calc(&mut self, modulated: Vec<f32>, first_or_last: bool) -> bool {
        let calc = match (&self.prev_modulated, first_or_last) {
            (Some(prev), false) if prev.len() == modulated.len() && self.residual.is_some() => {
                let (mut diff, mut base) = (0f64, 0f64);
                for (&a, &b) in modulated.iter().zip(prev) {
                    diff += (a - b).abs() as f64;
                    base += b.abs() as f64;
                }
                self.last_distance = poly(&self.coeffs, diff / base);
                self.accumulated += self.last_distance;
                self.accumulated >= self.rel_l1_thresh
            }
            _ => true,
        };
        if calc {
            self.accumulated = 0.0;
        }
        self.prev_modulated = Some(modulated);
        calc
    }

    /// Keeps the residual of a step that ran the blocks.
    pub fn set_residual(&mut self, residual: Vec<f32>) {
        self.residual = Some(residual);
    }

    /// The residual of the last step that ran the blocks.
    pub fn residual(&self) -> Option<&[f32]> {
        self.residual.as_deref()
    }
}

/// TaylorSeer settings (the references' `cache_init`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TaylorSeerConfig {
    /// Highest derivative order kept (OmniGen2: 4, BAGEL: 6).
    pub max_order: usize,
    /// The blocks run every `interval`-th step (`fresh_threshold`, 3).
    pub interval: usize,
    /// The first steps that always run the blocks (`first_enhance`, 5).
    pub first_enhance: usize,
}

impl TaylorSeerConfig {
    /// OmniGen2's values.
    pub const OMNIGEN2: Self = Self {
        max_order: 4,
        interval: 3,
        first_enhance: 5,
    };
    /// BAGEL's values.
    pub const BAGEL: Self = Self {
        max_order: 6,
        interval: 3,
        first_enhance: 5,
    };
}

/// What a step does (`cal_type`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Calc {
    /// Run the blocks, then [`TaylorSeer::update`].
    Full,
    /// Take [`TaylorSeer::predict`].
    Taylor,
}

/// TaylorSeer state of one guidance branch: call [`TaylorSeer::begin`] at each pass of the
/// branch, [`TaylorSeer::update`] or [`TaylorSeer::predict`] as it says, then
/// [`TaylorSeer::end`].
#[derive(Debug)]
pub struct TaylorSeer {
    cfg: TaylorSeerConfig,
    step: usize,
    counter: usize,
    /// The last two steps that ran the blocks (`activated_steps[-2..]`, which starts as `[0]`).
    activated: [usize; 2],
    /// Finite-difference derivatives of the feature, order 0 first.
    factors: Vec<Vec<f32>>,
}

impl TaylorSeer {
    pub fn new(cfg: TaylorSeerConfig) -> Self {
        Self {
            cfg,
            step: 0,
            counter: 0,
            activated: [0, 0],
            factors: Vec::new(),
        }
    }

    /// The pass index of this branch (passes begun and ended so far).
    pub fn step(&self) -> usize {
        self.step
    }

    /// `cal_type`: what this pass does.
    pub fn begin(&mut self) -> Calc {
        let first = self.step < self.cfg.first_enhance;
        if first || self.counter + 1 == self.cfg.interval.max(1) || self.factors.is_empty() {
            self.counter = 0;
            self.activated = [self.activated[1], self.step];
            Calc::Full
        } else {
            self.counter += 1;
            Calc::Taylor
        }
    }

    /// `derivative_approximation` after a full pass: `feature` becomes order 0 and each higher
    /// order the difference with the previous value of the order below, over the distance (in
    /// steps) between the last two full passes.
    pub fn update(&mut self, feature: Vec<f32>) {
        let dist = (self.activated[1] - self.activated[0]) as f32;
        let old = std::mem::take(&mut self.factors);
        let mut new = vec![feature];
        for i in 0..self.cfg.max_order {
            // As the reference: derivatives start once `step > first_enhance - 2`.
            if i >= old.len() || old[i].len() != new[i].len() || self.step + 2 <= self.cfg.first_enhance {
                break;
            }
            let mut d = vec![0f32; new[i].len()];
            let (n, o) = (&new[i], &old[i]);
            crate::kernels::for_chunks(&mut d, 4096, n.len() > (1 << 16), |c, out| {
                let at = c * 4096;
                for (k, v) in out.iter_mut().enumerate() {
                    *v = (n[at + k] - o[at + k]) / dist;
                }
            });
            new.push(d);
        }
        self.factors = new;
    }

    /// The feature of the last full pass.
    pub fn feature(&self) -> &[f32] {
        &self.factors[0]
    }

    /// The derivative orders held (1 = the feature only).
    pub fn orders(&self) -> usize {
        self.factors.len()
    }

    /// `taylor_formula`: the expansion at this step, `sum_i (1 / i!) * f_i * x^i` with `x` the
    /// steps since the last full pass.
    pub fn predict(&self) -> Vec<f32> {
        let x = (self.step - self.activated[1]) as i64;
        let n = self.factors[0].len();
        let mut out = vec![0f32; n];
        let (mut fact, mut xp) = (1f64, 1i64);
        for (i, f) in self.factors.iter().enumerate() {
            if i > 0 {
                fact *= i as f64;
                xp *= x;
            }
            // `(1 / i!) * f` in the feature's precision, then `* x**i` (an integer), as torch.
            let (c, xi) = ((1.0 / fact) as f32, xp as f32);
            crate::kernels::for_chunks(&mut out, 4096, n > (1 << 16), |ch, o| {
                let at = ch * 4096;
                for (k, v) in o.iter_mut().enumerate() {
                    *v += (c * f[at + k]) * xi;
                }
            });
        }
        out
    }

    /// The end of a pass of this branch.
    pub fn end(&mut self) {
        self.step += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poly_is_numpy_poly1d() {
        // np.poly1d([-5.48259225, 11.48772289, -4.47407401, 2.47730926, -0.03316487])(0.1)
        assert!((poly(&OMNIGEN2_TEACACHE_POLY, 0.1) - 0.180_764_779_565).abs() < 1e-12);
        assert_eq!(poly(&OMNIGEN2_TEACACHE_POLY, 0.0), -0.03316487);
    }

    #[test]
    fn teacache_skips_until_the_threshold() {
        // Identity polynomial: the accumulated distance is the plain relative L1 change.
        let mut t = TeaCache::new(0.25, &[1.0, 0.0]);
        assert!(t.should_calc(vec![1.0; 4], true));
        t.set_residual(vec![0.5; 4]);
        // +10% twice stays under 0.25 (0.1, then 0.1 / 1.1 more), the third change crosses it.
        assert!(!t.should_calc(vec![1.1; 4], false));
        assert!((t.last_distance - 0.1).abs() < 1e-6);
        assert!(!t.should_calc(vec![1.2; 4], false));
        assert!(t.should_calc(vec![1.3; 4], false));
        t.set_residual(vec![0.25; 4]);
        // The sum restarted.
        assert!(!t.should_calc(vec![1.35; 4], false));
        // The last step always runs.
        assert!(t.should_calc(vec![1.35; 4], true));
        assert_eq!(t.residual().unwrap(), &[0.25; 4]);
    }

    /// The schedule of the reference's `cal_type` with `fresh_threshold` 3, `first_enhance` 5.
    #[test]
    fn taylorseer_schedule() {
        let mut t = TaylorSeer::new(TaylorSeerConfig::OMNIGEN2);
        let mut kinds = String::new();
        for _ in 0..14 {
            match t.begin() {
                Calc::Full => {
                    kinds.push('F');
                    t.update(vec![0.0]);
                }
                Calc::Taylor => kinds.push('t'),
            }
            t.end();
        }
        assert_eq!(kinds, "FFFFFttFttFttF");
    }

    /// A linear feature is extrapolated exactly; a quadratic one takes the reference's backward
    /// differences (over 1 step at step 4, over 3 steps from then on), which are not its
    /// derivatives at the last full step.
    #[test]
    fn taylorseer_expansion() {
        let f = |s: usize| {
            let s = s as f32;
            vec![2.0 + 3.0 * s, 1.0 - 0.5 * s + 0.25 * s * s]
        };
        let mut t = TaylorSeer::new(TaylorSeerConfig {
            max_order: 2,
            interval: 3,
            first_enhance: 5,
        });
        for s in 0..10 {
            match t.begin() {
                Calc::Full => t.update(f(s)),
                Calc::Taylor => {
                    let p = t.predict();
                    // Order 1 from step 4, order 2 from the full pass at step 7 on.
                    assert_eq!(t.orders(), if s >= 8 { 3 } else { 2 });
                    assert!((p[0] - f(s)[0]).abs() < 1e-4, "step {s}: {p:?}");
                    // At 7: f = 9.75, f' = (9.75 - 3) / 3 = 2.25, f'' = (2.25 - 1.25) / 3.
                    let want = match s {
                        8 => Some(9.75 + 2.25 + 1.0 / 6.0),
                        9 => Some(9.75 + 4.5 + 4.0 / 6.0),
                        _ => None,
                    };
                    if let Some(want) = want {
                        assert!((p[1] - want).abs() < 1e-4, "step {s}: {} vs {want}", p[1]);
                    }
                }
            }
            t.end();
        }
    }

    /// Derivatives only start at step 4 (`step > first_enhance - 2`), as the reference.
    #[test]
    fn taylorseer_first_steps_hold_no_derivative() {
        let mut t = TaylorSeer::new(TaylorSeerConfig::BAGEL);
        for s in 0..5 {
            assert_eq!(t.begin(), Calc::Full);
            t.update(vec![s as f32]);
            assert_eq!(t.orders(), if s < 4 { 1 } else { 2 });
            t.end();
        }
        assert_eq!(t.begin(), Calc::Taylor);
        // f0 = 4, f1 = (4 - 3) / 1, one step later.
        assert_eq!(t.predict(), vec![5.0]);
    }
}
