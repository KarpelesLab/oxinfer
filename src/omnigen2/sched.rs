//! OmniGen2's samplers over its schedule `t = 0 (noise) .. 1 (image)`: the default Euler steps
//! and the official `--scheduler dpmsolver++` option, the repository's
//! `DPMSolverMultistepScheduler(algorithm_type="dpmsolver++", solver_type="midpoint",
//! solver_order=2, prediction_type="flow_prediction")`
//! (`omnigen2/schedulers/scheduling_dpmsolver_multistep.py`).
//!
//! That scheduler works on `sigma = 1 - t` (`alpha = 1 - sigma`), turns each velocity into a
//! data prediction `x0 = x + sigma * v`, and takes second-order multistep DPM-Solver++ steps in
//! `lambda = ln(alpha) - ln(sigma)`. The first step (`sigma = 1`, `lambda = -inf`) and the last
//! (`sigma_next = 0`, `final_sigmas_type = "zero"`) are first order; so is, in effect, the
//! second step, whose difference term is divided by an infinite `lambda` ratio. Scalars are
//! computed in f32 as the reference's 0-dimensional tensors.

/// The sampler of a run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Deserialize)]
pub enum Scheduler {
    /// `FlowMatchEulerDiscreteScheduler`: `x += (t_next - t) * v` (the default).
    #[default]
    #[serde(rename = "euler")]
    Euler,
    /// DPM-Solver++ (2M, midpoint) on the same timesteps.
    #[serde(rename = "dpmsolver++", alias = "dpmsolver")]
    DpmSolver,
}

impl Scheduler {
    /// `euler` or `dpmsolver++` (also `dpmsolver`, `dpm`), as the official `--scheduler`.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "euler" => Some(Self::Euler),
            "dpmsolver++" | "dpmsolver" | "dpm" | "dpm++" => Some(Self::DpmSolver),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Euler => "euler",
            Self::DpmSolver => "dpmsolver++",
        }
    }
}

/// The DPM-Solver++ state of one run.
pub struct DpmSolver {
    /// `1 - t` per step, then 0.
    pub sigmas: Vec<f32>,
    /// The previous step's data prediction.
    prev: Option<Vec<f32>>,
}

/// `lambda = ln(alpha) - ln(sigma)` with `alpha = 1 - sigma`.
fn lambda(sigma: f32) -> f32 {
    (1.0 - sigma).ln() - sigma.ln()
}

impl DpmSolver {
    /// `timesteps`: the `t` of each step (without the final 1), as [`super::timesteps`] gives
    /// them.
    pub fn new(timesteps: &[f32]) -> Self {
        let mut sigmas: Vec<f32> = timesteps.iter().map(|&t| 1.0 - t).collect();
        sigmas.push(0.0);
        Self { sigmas, prev: None }
    }

    /// Step `i`: `x` (the latents at `timesteps[i]`) becomes the latents at the next timestep
    /// given the velocity `v`.
    pub fn step(&mut self, i: usize, v: &[f32], x: &mut [f32]) {
        let n = self.sigmas.len() - 1;
        let (sigma_s0, sigma_t) = (self.sigmas[i], self.sigmas[i + 1]);
        // convert_model_output (flow_prediction): x0 = sample + sigma * model_output.
        let m0: Vec<f32> = x.iter().zip(v).map(|(&a, &b)| a + sigma_s0 * b).collect();
        let alpha_t = 1.0 - sigma_t;
        let (lambda_t, lambda_s0) = (lambda(sigma_t), lambda(sigma_s0));
        let h = lambda_t - lambda_s0;
        let a = sigma_t / sigma_s0;
        let b = alpha_t * ((-h).exp() - 1.0);
        match &self.prev {
            // Second order (the last step is first order: `final_sigmas_type == "zero"`).
            Some(m1) if i + 1 != n && m1.len() == m0.len() => {
                let h_0 = lambda_s0 - lambda(self.sigmas[i - 1]);
                let inv_r0 = 1.0 / (h_0 / h);
                let c = 0.5 * b;
                for ((xo, &d0), &p) in x.iter_mut().zip(&m0).zip(m1) {
                    let d1 = inv_r0 * (d0 - p);
                    *xo = a * *xo - b * d0 - c * d1;
                }
            }
            _ => {
                for (xo, &d0) in x.iter_mut().zip(&m0) {
                    *xo = a * *xo - b * d0;
                }
            }
        }
        self.prev = Some(m0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The model output of the checks below at step `i`, element `k` (exact in f32).
    fn v(i: usize, k: usize) -> f32 {
        ((i * 7 + k * 3) % 11) as f32 / 8.0 - 0.5
    }

    /// Against the official scheduler (tools/ref_omnigen2_sched.py prints these): 6 steps on 256
    /// latent pixels (`num_tokens`), sample `[0.5, -1, 0.25, 2]`, model outputs [`v`].
    #[test]
    fn dpmsolver_matches_the_official_scheduler() {
        let ts = crate::omnigen2::timesteps(6, 256);
        let want_sigmas: [f32; 7] = REF_SIGMAS;
        let mut s = DpmSolver::new(&ts[..6]);
        for (a, b) in s.sigmas.iter().zip(&want_sigmas) {
            assert!((a - b).abs() <= 1e-7, "sigmas {:?} vs {want_sigmas:?}", s.sigmas);
        }
        let mut x = vec![0.5f32, -1.0, 0.25, 2.0];
        for (i, want) in REF_SAMPLES.iter().enumerate() {
            let vi: Vec<f32> = (0..4).map(|k| v(i, k)).collect();
            s.step(i, &vi, &mut x);
            for (a, b) in x.iter().zip(want) {
                assert!((a - b).abs() <= 2e-6 * b.abs().max(1.0), "step {i}: {x:?} vs {want:?}");
            }
        }
    }

    // From tools/ref_omnigen2_sched.py (steps 6, num_tokens 256).
    const REF_SIGMAS: [f32; 7] = [1.0, 0.6666666, 0.44444442, 0.28571427, 0.16666663, 0.07407409, 0.0];
    const REF_SAMPLES: [[f32; 4]; 6] = [
        [0.3333333, -1.0416667, 0.33333334, 2.2083335],
        [0.41666666, -0.8750001, 0.2777778, 2.2361112],
        [0.3668067, -0.8653363, 0.42951685, 2.1465685],
        [0.5019219, -0.931114, 0.4255802, 2.251922],
        [0.51129854, -0.8329185, 0.353348, 2.2381504],
        [0.49278003, -0.82365924, 0.39038503, 2.2011132],
    ];

    /// One step is the data prediction itself; a constant data prediction is kept by every step.
    #[test]
    fn dpmsolver_limits() {
        let mut s = DpmSolver::new(&[0.0]);
        let mut x = vec![1.0f32, -2.0];
        s.step(0, &[0.5, 0.25], &mut x);
        assert_eq!(x, vec![1.5, -1.75]);
        // v = (x0 - x) / sigma: every step lands on the line from x to x0.
        let ts = crate::omnigen2::timesteps(5, 1024);
        let mut s = DpmSolver::new(&ts[..5]);
        let (x_start, x0) = (2.0f32, -1.0f32);
        let mut x = vec![x_start];
        for i in 0..5 {
            let v = (x0 - x[0]) / s.sigmas[i];
            s.step(i, &[v], &mut x);
            let want = s.sigmas[i + 1] * x_start + (1.0 - s.sigmas[i + 1]) * x0;
            assert!((x[0] - want).abs() < 1e-5, "step {i}: {} vs {want}", x[0]);
        }
        assert!((x[0] - x0).abs() < 1e-6);
    }

    #[test]
    fn names() {
        assert_eq!(Scheduler::parse("dpmsolver++"), Some(Scheduler::DpmSolver));
        assert_eq!(Scheduler::parse("Euler"), Some(Scheduler::Euler));
        assert_eq!(Scheduler::parse("ddim"), None);
    }
}
