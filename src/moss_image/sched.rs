//! Noise schedules and samplers for MOSS-Image (a rectified flow: `x_sigma = (1 - sigma) x0 +
//! sigma noise`, the transformer predicting the velocity `noise - x0` at timestep `1000 sigma`).
//!
//! - [`Schedule::Moss`], the release's `MOSSImageScheduler`: `steps` values of the shifted grid
//!   `shift v / (1 + (shift - 1) v)` over `v = linspace(1, shift / (1000 + shift - 1))`, each
//!   snapped to the nearest of the 1000 training sigmas `linspace(1, 0.001)`, repeats dropped,
//!   then 0. Float32 throughout, as the release computes it.
//! - ComfyUI's schedules over its model sampling (`time_snr_shift(shift, i / 1000)`, `i = 1..=1000`):
//!   [`Schedule::Simple`], [`Schedule::Beta`] (alpha = beta = 0.6) and [`Schedule::Beta57`]
//!   (RES4LYF's alpha 0.5, beta 0.7).
//! - [`Sampler::Euler`] (the release's, and ComfyUI's `euler`) and [`Sampler::ErSde`] (ComfyUI's
//!   `er_sde`: the VP ER-SDE-Solver-3 on data predictions, with fresh noise each step).

use anyhow::{Result, bail};

/// Where the sigmas come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Schedule {
    #[default]
    Moss,
    Simple,
    Beta,
    Beta57,
}

impl Schedule {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s.to_ascii_lowercase().as_str() {
            "moss" | "default" => Self::Moss,
            "simple" => Self::Simple,
            "beta" => Self::Beta,
            "beta57" => Self::Beta57,
            _ => return None,
        })
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Moss => "moss",
            Self::Simple => "simple",
            Self::Beta => "beta",
            Self::Beta57 => "beta57",
        }
    }
}

/// How each step moves the latents.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Sampler {
    #[default]
    Euler,
    ErSde,
}

impl Sampler {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s.to_ascii_lowercase().as_str() {
            "euler" => Self::Euler,
            "er_sde" | "er-sde" | "ersde" => Self::ErSde,
            _ => return None,
        })
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Euler => "euler",
            Self::ErSde => "er_sde",
        }
    }
}

/// Where the noise comes from: oxinfer's own generator, or PyTorch's CPU generator
/// ([`crate::torch_rng`]) as diffusers and ComfyUI draw it: the initial latents
/// `torch.randn([1, 128, h / 16, w / 16])` from `manual_seed(seed)` (diffusers' pipeline with a
/// CPU generator, ComfyUI's `prepare_noise`), and ER-SDE's step noise from a second CPU generator
/// seeded `seed + 1`, one `randn` of the latent's shape per step (ComfyUI's `default_noise_sampler`
/// when it samples on the CPU; on CUDA or MPS ComfyUI draws step noise from the device's own
/// generator, which is not reproduced).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Noise {
    #[default]
    Oxinfer,
    Torch,
}

impl Noise {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s.to_ascii_lowercase().as_str() {
            "oxinfer" | "default" => Self::Oxinfer,
            "torch" | "comfy" | "comfyui" | "diffusers" => Self::Torch,
            _ => return None,
        })
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Oxinfer => "oxinfer",
            Self::Torch => "torch",
        }
    }
}

/// `torch.linspace(start, end, steps)` in float32 (ATen's CPU kernel: the first half counted
/// from `start`, the second from `end`).
pub fn linspace_f32(start: f32, end: f32, steps: usize) -> Vec<f32> {
    if steps == 1 {
        return vec![start];
    }
    let step = (end - start) / (steps - 1) as f32;
    let half = steps / 2;
    (0..steps)
        .map(|i| {
            if i < half {
                start + step * i as f32
            } else {
                end - step * (steps - 1 - i) as f32
            }
        })
        .collect()
}

/// The release's schedule (see the module docs), with the final 0.
pub fn moss_sigmas(steps: usize, shift: f32) -> Result<Vec<f32>> {
    if steps == 0 || shift <= 0.0 {
        bail!("steps and shift must be positive");
    }
    let total = 1000usize;
    let grid = |count: usize, shift: f32| -> Vec<f32> {
        let lower = (shift as f64 / (total as f64 + shift as f64 - 1.0)) as f32;
        linspace_f32(1.0, lower, count)
            .into_iter()
            .map(|v| shift * v / (1.0 + (shift - 1.0) * v))
            .collect()
    };
    let sig = grid(steps, shift);
    let mut training = grid(total, 1.0);
    training.reverse();
    let mut out: Vec<f32> = Vec::new();
    for s in sig {
        let upper = training.partition_point(|&t| t < s).min(total - 1);
        let lower = upper.saturating_sub(1);
        let pick = if (s - training[lower]).abs() <= (training[upper] - s).abs() {
            lower
        } else {
            upper
        };
        let v = training[pick];
        if out.last() != Some(&v) {
            out.push(v);
        }
    }
    out.push(0.0);
    Ok(out)
}

/// ComfyUI's `time_snr_shift`.
fn time_snr_shift(shift: f64, t: f64) -> f64 {
    if shift == 1.0 { t } else { shift * t / (1.0 + (shift - 1.0) * t) }
}

/// ComfyUI's model-sampling sigma table: `time_snr_shift(shift, i / 1000)`, `i = 1..=1000` (as
/// float32 values).
fn comfy_table(shift: f32) -> Vec<f32> {
    (1..=1000)
        .map(|i| time_snr_shift(shift as f64, (i as f32 / 1000.0) as f64) as f32)
        .collect()
}

/// The sigmas of a ComfyUI scheduler, with the final 0.
pub fn comfy_sigmas(s: Schedule, steps: usize, shift: f32) -> Result<Vec<f32>> {
    if steps == 0 {
        bail!("steps must be at least 1");
    }
    let table = comfy_table(shift);
    let mut out = Vec::new();
    match s {
        Schedule::Moss => return moss_sigmas(steps, shift),
        Schedule::Simple => {
            let ss = table.len() as f64 / steps as f64;
            for x in 0..steps {
                out.push(table[table.len() - 1 - (x as f64 * ss) as usize]);
            }
        }
        Schedule::Beta | Schedule::Beta57 => {
            let (a, b) = if s == Schedule::Beta { (0.6, 0.6) } else { (0.5, 0.7) };
            let total = (table.len() - 1) as f64;
            let mut last = -1.0;
            for i in 0..steps {
                let p = 1.0 - i as f64 / steps as f64;
                let t = (beta_ppf(p, a, b) * total).round_ties_even();
                if t != last {
                    out.push(table[t as usize]);
                }
                last = t;
            }
        }
    }
    out.push(0.0);
    Ok(out)
}

/// `ln Gamma(x)` (Lanczos, g = 7).
fn ln_gamma(x: f64) -> f64 {
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + 7.5;
    for (i, &c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// The continued fraction of the incomplete beta function (Lentz).
fn beta_cf(x: f64, a: f64, b: f64) -> f64 {
    let tiny = 1e-300;
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..1000 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-15 {
            break;
        }
    }
    h
}

/// The regularized incomplete beta function `I_x(a, b)`.
pub fn beta_cdf(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let lbt = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln();
    if x < (a + 1.0) / (a + b + 2.0) {
        (lbt.exp() * beta_cf(x, a, b)) / a
    } else {
        1.0 - (lbt.exp() * beta_cf(1.0 - x, b, a)) / b
    }
}

/// The beta distribution's quantile (`scipy.stats.beta.ppf`), by bisection.
pub fn beta_ppf(p: f64, a: f64, b: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if beta_cdf(mid, a, b) < p {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < 1e-16 {
            break;
        }
    }
    0.5 * (lo + hi)
}

/// The state of one sampler run over `sigmas`.
pub struct Run {
    sampler: Sampler,
    sigmas: Vec<f32>,
    /// ER-SDE: `lambda = sigma / (1 - sigma)` per step, and the earlier data predictions.
    lambdas: Vec<f64>,
    old: Option<Vec<f32>>,
    old_d: Option<Vec<f32>>,
}

impl Run {
    /// A run over `sigmas` (with the final 0) for a model sampled with `shift` (ER-SDE moves a
    /// first sigma of 1, which has no SNR, to ComfyUI's `percent_to_sigma(1e-4)`).
    pub fn new(sampler: Sampler, mut sigmas: Vec<f32>, shift: f32) -> Self {
        if sampler == Sampler::ErSde && sigmas.len() > 1 && sigmas[0] >= 1.0 {
            sigmas[0] = time_snr_shift(shift as f64, 1.0 - 1e-4) as f32;
        }
        let lambdas = sigmas.iter().map(|&s| s as f64 / (1.0 - s as f64)).collect();
        Self {
            sampler,
            sigmas,
            lambdas,
            old: None,
            old_d: None,
        }
    }

    pub fn steps(&self) -> usize {
        self.sigmas.len() - 1
    }

    /// The sigma the model sees at step `i`.
    pub fn sigma(&self, i: usize) -> f32 {
        self.sigmas[i]
    }

    /// One step from velocity `v` at step `i`; `noise` gives fresh standard normal values
    /// (stochastic samplers).
    pub fn step(&mut self, i: usize, x: &mut [f32], v: &[f32], noise: &mut dyn FnMut(usize) -> Vec<f32>) {
        let (s, sn) = (self.sigmas[i], self.sigmas[i + 1]);
        match self.sampler {
            Sampler::Euler => {
                let d = sn - s;
                for (a, &b) in x.iter_mut().zip(v) {
                    *a += d * b;
                }
            }
            Sampler::ErSde => {
                let den: Vec<f32> = x.iter().zip(v).map(|(&a, &b)| a - b * s).collect();
                let stage = (i + 1).min(3);
                if sn == 0.0 {
                    x.copy_from_slice(&den);
                } else {
                    let ls = &self.lambdas;
                    let (l_s, l_t) = (ls[i], ls[i + 1]);
                    let alpha_s = s as f64 / l_s;
                    let alpha_t = sn as f64 / l_t;
                    let r_alpha = alpha_t / alpha_s;
                    let scaler = |x: f64| x * (x.powf(0.3).exp() + 10.0);
                    let r = scaler(l_t) / scaler(l_s);
                    let (c1, c2) = ((r_alpha * r) as f32, (alpha_t * (1.0 - r)) as f32);
                    for (a, &d) in x.iter_mut().zip(&den) {
                        *a = c1 * *a + c2 * d;
                    }
                    if stage >= 2 {
                        let old = self.old.as_ref().expect("ER-SDE: previous prediction");
                        let dt = l_t - l_s;
                        let n_pts = 200usize;
                        let h = -dt / n_pts as f64;
                        let pos: Vec<f64> = (0..n_pts).map(|k| l_t + k as f64 * h).collect();
                        let sum_inv: f64 = pos.iter().map(|&p| 1.0 / scaler(p)).sum::<f64>() * h;
                        let dd: Vec<f32> = den
                            .iter()
                            .zip(old)
                            .map(|(&a, &b)| ((a - b) as f64 / (l_s - ls[i - 1])) as f32)
                            .collect();
                        let c = (alpha_t * (dt + sum_inv * scaler(l_t))) as f32;
                        for (a, &d) in x.iter_mut().zip(&dd) {
                            *a += c * d;
                        }
                        if stage >= 3 {
                            let old_d = self.old_d.as_ref().expect("ER-SDE: previous derivative");
                            let s_u: f64 = pos.iter().map(|&p| (p - l_s) / scaler(p)).sum::<f64>() * h;
                            let den_scale = (l_s - ls[i - 2]) / 2.0;
                            let c = (alpha_t * (dt * dt / 2.0 + s_u * scaler(l_t))) as f32;
                            for ((a, &d), &od) in x.iter_mut().zip(&dd).zip(old_d) {
                                *a += c * ((d - od) as f64 / den_scale) as f32;
                            }
                        }
                        self.old_d = Some(dd);
                    }
                    let var = l_t * l_t - l_s * l_s * r * r;
                    let amp = if var > 0.0 { (alpha_t * var.sqrt()) as f32 } else { 0.0 };
                    if amp > 0.0 {
                        let z = noise(x.len());
                        for (a, &n) in x.iter_mut().zip(&z) {
                            *a += amp * n;
                        }
                    }
                }
                self.old = Some(den);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_schedule() {
        // The reference's MOSSImageScheduler (shift 5).
        let s = moss_sigmas(4, 5.0).unwrap();
        assert_eq!(s.len(), 5);
        assert_eq!(s[0], 1.0);
        assert!((s[1] - 0.91).abs() < 1e-6 && (s[2] - 0.717).abs() < 1e-6 && (s[3] - 0.024).abs() < 1e-6);
        let s = moss_sigmas(8, 5.0).unwrap();
        let t: Vec<f32> = s.iter().map(|v| (v * 1000.0).round()).collect();
        assert_eq!(&t[..6], &[1000.0, 968.0, 926.0, 871.0, 791.0, 671.0]);
        assert_eq!(moss_sigmas(50, 5.0).unwrap().len(), 51);
    }

    #[test]
    fn beta_quantiles() {
        // scipy.stats.beta.ppf(0.3, 0.5, 0.7) = 0.10914...; the cdf of the symmetric beta at 1/2.
        assert!((beta_cdf(0.5, 0.6, 0.6) - 0.5).abs() < 1e-12);
        let x = beta_ppf(0.3, 0.5, 0.7);
        assert!((beta_cdf(x, 0.5, 0.7) - 0.3).abs() < 1e-12);
        let s = comfy_sigmas(Schedule::Beta57, 24, 5.0).unwrap();
        assert_eq!(s[0], 1.0);
        assert_eq!(*s.last().unwrap(), 0.0);
        assert!(s.windows(2).all(|w| w[0] > w[1]));
        let s = comfy_sigmas(Schedule::Simple, 8, 5.0).unwrap();
        assert_eq!(s.len(), 9);
        assert_eq!(s[0], 1.0);
    }

    #[test]
    fn comfy_schedules_match() {
        // ComfyUI's beta_scheduler (scipy's beta.ppf) / simple_scheduler over the MOSS model
        // sampling (shift 5), printed by the verification script.
        let close = |a: &[f32], b: &[f32]| a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 2e-6);
        let b57 = [
            1.0, 0.995106, 0.986731, 0.975877, 0.962247, 0.946307, 0.927835, 0.906216, 0.881051, 0.852417, 0.819559, 0.782186, 0.739379,
            0.689964, 0.634842, 0.573593, 0.504177, 0.42979, 0.349424, 0.267296, 0.187075, 0.113636, 0.057252, 0.019685, 0.0,
        ];
        assert!(close(&comfy_sigmas(Schedule::Beta57, 24, 5.0).unwrap(), &b57));
        let beta = [1.0, 0.988055, 0.959032, 0.910695, 0.833888, 0.711207, 0.51815, 0.23539, 0.0];
        assert!(close(&comfy_sigmas(Schedule::Beta, 8, 5.0).unwrap(), &beta));
        let simple = [1.0, 0.972222, 0.9375, 0.892857, 0.833333, 0.75, 0.625, 0.416667, 0.0];
        assert!(close(&comfy_sigmas(Schedule::Simple, 8, 5.0).unwrap(), &simple));
    }

    #[test]
    fn er_sde_matches_comfy() {
        // ComfyUI's sample_er_sde without noise on a toy model whose prediction is
        // `x - sigma (0.9 x - c)`, over the simple 8-step schedule.
        let c = [0.3f32, -0.7, 1.2];
        let mut x = vec![1.0f32, -2.0, 0.5];
        let mut run = Run::new(Sampler::ErSde, comfy_sigmas(Schedule::Simple, 8, 5.0).unwrap(), 5.0);
        for i in 0..run.steps() {
            let v: Vec<f32> = x.iter().zip(&c).map(|(&a, &b)| 0.9 * a - b).collect();
            run.step(i, &mut x, &v, &mut |n| vec![0.0; n]);
        }
        let want = [0.211_168, -0.492_566_7, 0.843_005_8];
        assert!(x.iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-4), "{x:?}");
    }
}
