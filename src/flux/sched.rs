//! Flow-matching Euler sampling (diffusers `FlowMatchEulerDiscreteScheduler`), as used by
//! FLUX, Qwen-Image, Step1X-Edit, OmniGen2 and other rectified-flow models: the model predicts a
//! velocity `v`, and each step moves the sample by `(sigma_next - sigma) * v`. Sigmas run from 1
//! (pure noise) to 0, optionally shifted by a constant (`shift`) or by the image sequence length
//! (`use_dynamic_shifting`, the `mu` of [`calculate_shift`]).

use anyhow::{Result, bail};
use std::path::Path;

/// `scheduler_config.json` of a flow-matching pipeline (diffusers defaults for missing keys).
#[derive(Clone, Debug, PartialEq)]
pub struct FlowMatchEuler {
    pub num_train_timesteps: usize,
    pub shift: f64,
    pub use_dynamic_shifting: bool,
    pub base_shift: f64,
    pub max_shift: f64,
    pub base_image_seq_len: usize,
    pub max_image_seq_len: usize,
    /// "exponential" (default) or "linear" time shift with `mu`.
    pub time_shift_linear: bool,
    pub shift_terminal: Option<f64>,
    pub invert_sigmas: bool,
}

impl Default for FlowMatchEuler {
    fn default() -> Self {
        Self {
            num_train_timesteps: 1000,
            shift: 1.0,
            use_dynamic_shifting: false,
            base_shift: 0.5,
            max_shift: 1.15,
            base_image_seq_len: 256,
            max_image_seq_len: 4096,
            time_shift_linear: false,
            shift_terminal: None,
            invert_sigmas: false,
        }
    }
}

/// The resolution-dependent shift `mu` (diffusers `calculate_shift`): linear in the number of
/// image tokens between (`base_seq_len`, `base_shift`) and (`max_seq_len`, `max_shift`).
pub fn calculate_shift(image_seq_len: usize, base_seq_len: usize, max_seq_len: usize, base_shift: f64, max_shift: f64) -> f64 {
    let m = (max_shift - base_shift) / (max_seq_len as f64 - base_seq_len as f64);
    let b = base_shift - m * base_seq_len as f64;
    image_seq_len as f64 * m + b
}

/// The denoising plan: `timesteps[i]` (what the model is given, `sigma * 1000`) and `sigmas`
/// (one more than the steps, ending at 0).
#[derive(Clone, Debug)]
pub struct Plan {
    pub timesteps: Vec<f32>,
    pub sigmas: Vec<f32>,
}

impl FlowMatchEuler {
    pub fn load(path: &Path) -> Result<Self> {
        let cfg: serde_json::Value = serde_json::from_slice(&crate::fsx::read(path)?)?;
        Self::from_json(&cfg)
    }

    pub fn from_json(cfg: &serde_json::Value) -> Result<Self> {
        let d = Self::default();
        let f = |k: &str, v: f64| cfg[k].as_f64().unwrap_or(v);
        let u = |k: &str, v: usize| cfg[k].as_u64().map(|x| x as usize).unwrap_or(v);
        let b = |k: &str| cfg[k].as_bool().unwrap_or(false);
        for k in [
            "use_karras_sigmas",
            "use_exponential_sigmas",
            "use_beta_sigmas",
            "stochastic_sampling",
        ] {
            if b(k) {
                bail!("scheduler option {k} is not supported");
            }
        }
        Ok(Self {
            num_train_timesteps: u("num_train_timesteps", d.num_train_timesteps),
            shift: f("shift", d.shift),
            use_dynamic_shifting: b("use_dynamic_shifting"),
            base_shift: f("base_shift", d.base_shift),
            max_shift: f("max_shift", d.max_shift),
            base_image_seq_len: u("base_image_seq_len", d.base_image_seq_len),
            max_image_seq_len: u("max_image_seq_len", d.max_image_seq_len),
            time_shift_linear: cfg["time_shift_type"].as_str() == Some("linear"),
            shift_terminal: cfg["shift_terminal"].as_f64(),
            invert_sigmas: b("invert_sigmas"),
        })
    }

    /// `mu` for `image_seq_len` image tokens with this config's bounds.
    pub fn mu(&self, image_seq_len: usize) -> f64 {
        calculate_shift(
            image_seq_len,
            self.base_image_seq_len,
            self.max_image_seq_len,
            self.base_shift,
            self.max_shift,
        )
    }

    /// `set_timesteps(sigmas=..., mu=...)`: the given base sigmas (FLUX pipelines pass
    /// `linspace(1, 1/steps, steps)`, see [`flux_sigmas`]) shifted by `mu` (dynamic shifting) or by
    /// the constant `shift`. Arithmetic in f32 like diffusers (numpy float32 arrays).
    pub fn plan(&self, base: &[f32], mu: Option<f64>) -> Result<Plan> {
        if self.use_dynamic_shifting && mu.is_none() {
            bail!("this scheduler needs `mu` (dynamic shifting)");
        }
        let mut s: Vec<f32> = if self.use_dynamic_shifting {
            let mu = mu.unwrap();
            let e = mu.exp() as f32;
            base.iter()
                .map(|&t| {
                    let r = 1.0 / t - 1.0;
                    if self.time_shift_linear {
                        mu as f32 / (mu as f32 + r)
                    } else {
                        e / (e + r)
                    }
                })
                .collect()
        } else {
            let k = self.shift as f32;
            base.iter().map(|&t| k * t / (1.0 + (k - 1.0) * t)).collect()
        };
        if let Some(st) = self.shift_terminal
            && let Some(&last) = s.last()
        {
            let scale = (1.0 - last) / (1.0 - st as f32);
            for v in s.iter_mut() {
                *v = 1.0 - (1.0 - *v) / scale;
            }
        }
        let n = self.num_train_timesteps as f32;
        let mut timesteps: Vec<f32> = s.iter().map(|&v| v * n).collect();
        if self.invert_sigmas {
            for v in s.iter_mut() {
                *v = 1.0 - *v;
            }
            timesteps = s.iter().map(|&v| v * n).collect();
            s.push(1.0);
        } else {
            s.push(0.0);
        }
        Ok(Plan { timesteps, sigmas: s })
    }

    /// The default plan of FLUX-style pipelines for `steps` steps over `image_seq_len` tokens.
    pub fn flux_plan(&self, steps: usize, image_seq_len: usize) -> Result<Plan> {
        self.plan(&flux_sigmas(steps), Some(self.mu(image_seq_len)))
    }

    /// One Euler step: `x += (sigmas[i + 1] - sigmas[i]) * v`.
    pub fn step(plan: &Plan, i: usize, x: &mut [f32], v: &[f32]) {
        let dt = plan.sigmas[i + 1] - plan.sigmas[i];
        for (a, &b) in x.iter_mut().zip(v) {
            *a += dt * b;
        }
    }

    /// `scale_noise`: the sample at noise level `sigma` between clean `x0` and `noise`
    /// (`sigma * noise + (1 - sigma) * x0`), the start of image-to-image runs.
    pub fn scale_noise(x0: &[f32], noise: &[f32], sigma: f32) -> Vec<f32> {
        x0.iter().zip(noise).map(|(&a, &n)| sigma * n + (1.0 - sigma) * a).collect()
    }
}

/// `np.linspace(1.0, 1 / steps, steps)` as float32 (the base sigmas FLUX pipelines pass).
pub fn flux_sigmas(steps: usize) -> Vec<f32> {
    let steps = steps.max(1);
    if steps == 1 {
        return vec![1.0];
    }
    let (a, b) = (1.0f64, 1.0 / steps as f64);
    let d = (b - a) / (steps - 1) as f64;
    (0..steps)
        .map(|i| if i == steps - 1 { b as f32 } else { (a + i as f64 * d) as f32 })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kontext_schedule() {
        // diffusers 0.42 FlowMatchEulerDiscreteScheduler (FLUX.1 Kontext config), 4 steps, 4096
        // image tokens: set_timesteps(sigmas=np.linspace(1, 0.25, 4), mu=calculate_shift(4096)).
        let s = FlowMatchEuler {
            use_dynamic_shifting: true,
            shift: 3.0,
            ..Default::default()
        };
        let mu = s.mu(4096);
        assert!((mu - 1.15).abs() < 1e-12);
        let p = s.flux_plan(4, 4096).unwrap();
        let want = [1.0f32, 0.904_530_76, 0.759_510_9, 0.512_844_15, 0.0];
        for (a, b) in p.sigmas.iter().zip(want) {
            assert!((a - b).abs() < 2e-6, "{:?}", p.sigmas);
        }
        assert!((p.timesteps[1] - 904.530_76).abs() < 2e-3);
        // 28 steps over 1024 tokens (mu 0.63).
        let p = s.flux_plan(28, 1024).unwrap();
        assert!((p.sigmas[1] - 0.980_655_97).abs() < 2e-6 && (p.sigmas[27] - 0.065_019_6).abs() < 2e-6);
        // Schnell: no shifting at all (shift 1).
        let p = FlowMatchEuler::default().plan(&flux_sigmas(4), None).unwrap();
        assert_eq!(p.sigmas, vec![1.0, 0.75, 0.5, 0.25, 0.0]);
    }
}
