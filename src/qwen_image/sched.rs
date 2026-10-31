//! The flow-matching Euler scheduler of the Qwen-Image pipelines: [`crate::flux::sched`]'s
//! `FlowMatchEulerDiscreteScheduler`, driven as diffusers' Qwen-Image pipelines drive it (sigmas
//! `linspace(1, 1/steps, steps)`, exponential time shift with `mu` from the image token count,
//! the stretch to `shift_terminal`, Euler steps), plus the Lightning LoRAs' settings.

pub use crate::flux::sched::{FlowMatchEuler, calculate_shift};
use anyhow::Result;

/// The Qwen-Image scheduler.
pub type FlowMatch = FlowMatchEuler;

impl FlowMatchEuler {
    /// The scheduler the Lightning (step-distilled) LoRAs were trained with: a fixed shift of 3
    /// (`base_shift = max_shift = ln 3`), no terminal stretch.
    pub fn lightning() -> Self {
        Self {
            base_image_seq_len: 256,
            max_image_seq_len: 8192,
            base_shift: 3f64.ln(),
            max_shift: 3f64.ln(),
            use_dynamic_shifting: true,
            ..Default::default()
        }
    }

    /// The sigmas of `steps` steps for `seq_len` image tokens, with the trailing 0.
    pub fn sigmas(&self, steps: usize, seq_len: usize) -> Result<Vec<f32>> {
        Ok(self.flux_plan(steps, seq_len)?.sigmas)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmas_match_diffusers() {
        // Qwen-Image-Edit-2511's scheduler config, 4 steps at 4096 image tokens (diffusers 0.42).
        let f = FlowMatch::from_json(&serde_json::json!({
            "base_image_seq_len": 256, "max_image_seq_len": 8192, "base_shift": 0.5, "max_shift": 0.9,
            "shift": 1.0, "shift_terminal": 0.02, "use_dynamic_shifting": true, "time_shift_type": "exponential"
        }))
        .unwrap();
        let s = f.sigmas(4, 4096).unwrap();
        for (a, b) in s.iter().zip([1.0, 0.7667093, 0.4556137, 0.02, 0.0]) {
            assert!((a - b).abs() < 2e-6, "{s:?}");
        }
        // Lightning: fixed shift 3, no terminal stretch.
        let s = FlowMatch::lightning().sigmas(4, 1024).unwrap();
        for (a, b) in s.iter().zip([1.0, 0.9, 0.75, 0.5, 0.0]) {
            assert!((a - b).abs() < 2e-6, "{s:?}");
        }
    }
}
