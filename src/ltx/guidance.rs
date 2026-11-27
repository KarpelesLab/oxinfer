//! Guidance for the dev transformer, as the release's `ti2vid_*` pipelines do it
//! (`ltx_core.components.guiders.MultiModalGuider`, `ltx_pipelines.utils.denoisers`): per step,
//! besides the prompt's pass, a pass on the negative prompt (CFG), one with the self-attention of
//! some blocks skipped (STG: the value projection passed straight through), and one with the
//! audio-video cross-attention skipped (the isolated-modality pass); each stream's denoised rows
//!
//! ```text
//! pred = cond + (cfg - 1) (cond - uncond) + stg (cond - perturbed) + (modality - 1) (cond - isolated)
//! ```
//!
//! rescaled toward the prompt pass's spread (`rescale`: `pred *= rescale std(cond) / std(pred) +
//! 1 - rescale`, the standard deviations over the rows being denoised).

use super::dit::{Blocks, Perturb};

/// The release's negative prompt (`DEFAULT_NEGATIVE_PROMPT`).
pub const DEFAULT_NEGATIVE_PROMPT: &str = "has_subtitles, has_blurbox, transition from black, transition to black, speech_ending_short, \
blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, \
grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, \
deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, \
wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of \
field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent \
lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny \
valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, \
mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, \
off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward \
pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, \
inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.";

/// The release's stage-1 steps for the dev transformer (`num_inference_steps` of LTX-2.3 and
/// later).
pub const DEFAULT_STEPS: usize = 30;

/// One stream's guidance (`MultiModalGuiderParams`).
#[derive(Clone, Debug, PartialEq)]
pub struct GuiderParams {
    /// Classifier-free guidance (1: off).
    pub cfg: f32,
    /// Spatio-temporal guidance (0: off), perturbing the self-attention of `stg_blocks`.
    pub stg: f32,
    pub stg_blocks: Blocks,
    /// How far the guided rows are rescaled toward the prompt pass's spread (0: not at all).
    pub rescale: f32,
    /// Guidance away from the pass without audio-video cross-attention (1: off).
    pub modality: f32,
    /// Guide only every `skip_step + 1`-th step (the others reuse the last guided rows).
    pub skip_step: usize,
}

impl GuiderParams {
    /// No guidance: the prompt's pass alone.
    pub fn off() -> Self {
        Self {
            cfg: 1.0,
            stg: 0.0,
            stg_blocks: Blocks::None,
            rescale: 0.0,
            modality: 1.0,
            skip_step: 0,
        }
    }

    /// The release's video guidance for LTX-2.3 and later checkpoints (`LTX_2_3_PARAMS`, which
    /// LTX-2.5's `model_version` selects): CFG 3, STG 1 on block 28, rescale 0.7, modality 3.
    pub fn video() -> Self {
        Self {
            cfg: 3.0,
            stg: 1.0,
            stg_blocks: Blocks::List(vec![28]),
            rescale: 0.7,
            modality: 3.0,
            skip_step: 0,
        }
    }

    /// The release's audio guidance: as the video's with CFG 7.
    pub fn audio() -> Self {
        Self { cfg: 7.0, ..Self::video() }
    }

    pub fn uses_uncond(&self) -> bool {
        self.cfg != 1.0
    }

    pub fn uses_perturbed(&self) -> bool {
        self.stg != 0.0 && !self.stg_blocks.is_none()
    }

    pub fn uses_isolated(&self) -> bool {
        self.modality != 1.0
    }

    /// Whether step `i` reuses the last guided rows (`should_skip_step`).
    pub fn skips(&self, i: usize) -> bool {
        self.skip_step != 0 && !i.is_multiple_of(self.skip_step + 1)
    }

    /// The guided rows (`MultiModalGuider.calculate`): `cond` and the other passes' denoised rows
    /// (`[rows, width]`), `mask` each row's denoising mask (the rescale's spread is over rows
    /// with a nonzero mask).
    pub fn calculate(
        &self,
        cond: &[f32],
        uncond: Option<&[f32]>,
        ptb: Option<&[f32]>,
        iso: Option<&[f32]>,
        mask: &[f32],
        width: usize,
    ) -> Vec<f32> {
        let term = |scale: f32, other: Option<&[f32]>, i: usize, c: f32| other.map_or(0.0, |o| scale * (c - o[i]));
        let mut pred: Vec<f32> = cond
            .iter()
            .enumerate()
            .map(|(i, &c)| c + term(self.cfg - 1.0, uncond, i, c) + term(self.stg, ptb, i, c) + term(self.modality - 1.0, iso, i, c))
            .collect();
        if self.rescale != 0.0 {
            let std = |x: &[f32]| -> Option<f64> {
                let rows = || x.chunks(width).zip(mask).filter(|(_, m)| **m != 0.0).map(|(r, _)| r);
                let n: usize = rows().map(|r| r.len()).sum();
                if n < 2 {
                    return None;
                }
                let mean = rows().flatten().map(|&v| v as f64).sum::<f64>() / n as f64;
                let ss: f64 = rows().flatten().map(|&v| (v as f64 - mean).powi(2)).sum();
                Some((ss / (n - 1) as f64).sqrt())
            };
            let factor = match (std(cond), std(&pred)) {
                (Some(c), Some(p)) if p > 0.0 => (c / p) as f32,
                _ => 1.0,
            };
            let f = self.rescale * factor + (1.0 - self.rescale);
            pred.iter_mut().for_each(|v| *v *= f);
        }
        pred
    }
}

/// Whether a run is guided ([`super::Params::guidance_mode`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GuidanceMode {
    /// Guided with a dev transformer (except with the distilled LoRA on one stage, which runs
    /// the distilled schedule), not with a distilled one.
    #[default]
    Auto,
    On,
    /// The distilled pipeline (its schedule and sampler) whatever the transformer.
    Off,
}

impl GuidanceMode {
    /// `auto`, `on` / `default`, `off` / `none`.
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Self::Auto,
            "on" | "default" | "true" | "1" | "yes" => Self::On,
            "off" | "none" | "false" | "0" | "no" => Self::Off,
            _ => return None,
        })
    }
}

/// Guidance of both streams for the dev transformer's first stage (or its only one).
#[derive(Clone, Debug, PartialEq)]
pub struct Guidance {
    pub video: GuiderParams,
    pub audio: GuiderParams,
    /// The negative prompt of the CFG pass.
    pub negative_prompt: String,
    /// Stage-1 steps on [`ltx2_sigmas`] (when the run gives no schedule of its own).
    pub steps: usize,
}

impl Default for Guidance {
    /// The release's (`ti2vid_two_stages` / `ti2vid_one_stage` defaults for LTX-2.5).
    fn default() -> Self {
        Self {
            video: GuiderParams::video(),
            audio: GuiderParams::audio(),
            negative_prompt: DEFAULT_NEGATIVE_PROMPT.to_string(),
            steps: DEFAULT_STEPS,
        }
    }
}

impl Guidance {
    /// No guidance (the prompt's pass alone), `steps` steps.
    pub fn off() -> Self {
        Self {
            video: GuiderParams::off(),
            audio: GuiderParams::off(),
            ..Self::default()
        }
    }

    pub fn uses_uncond(&self) -> bool {
        self.video.uses_uncond() || self.audio.uses_uncond()
    }

    /// The perturbation of the STG pass (None: no such pass).
    pub fn perturbed(&self) -> Option<Perturb> {
        let p = Perturb {
            video_self: if self.video.uses_perturbed() {
                self.video.stg_blocks.clone()
            } else {
                Blocks::None
            },
            audio_self: if self.audio.uses_perturbed() {
                self.audio.stg_blocks.clone()
            } else {
                Blocks::None
            },
            cross: false,
        };
        (!p.is_none()).then_some(p)
    }

    pub fn uses_isolated(&self) -> bool {
        self.video.uses_isolated() || self.audio.uses_isolated()
    }

    /// Transformer passes per step.
    pub fn passes(&self) -> usize {
        1 + self.uses_uncond() as usize + self.perturbed().is_some() as usize + self.uses_isolated() as usize
    }
}

/// `LTX2Scheduler.execute(steps)`: `steps + 1` sigmas from 1 to 0, shifted for 4096 tokens
/// (`exp(shift) / (exp(shift) + 1 / s - 1)` with the shift interpolated between 0.95 at 1024
/// tokens and 2.05 at 4096) and stretched so that the last nonzero one is 0.1. Computed in float32
/// as the reference does.
pub fn ltx2_sigmas(steps: usize) -> Vec<f32> {
    ltx2_sigmas_for(steps, 4096)
}

/// [`ltx2_sigmas`] shifted for `tokens` latent tokens.
pub fn ltx2_sigmas_for(steps: usize, tokens: usize) -> Vec<f32> {
    let steps = steps.max(1);
    let (x1, x2) = (1024.0f64, 4096.0f64);
    let (max_shift, base_shift) = (2.05f64, 0.95f64);
    let mm = (max_shift - base_shift) / (x2 - x1);
    let b = base_shift - mm * x1;
    let shift = tokens as f64 * mm + b;
    let e = shift.exp() as f32;
    // torch.linspace(1, 0, steps + 1) in float32: the first half from the start, the rest from
    // the end.
    let n = steps + 1;
    let step = (0.0f32 - 1.0) / steps as f32;
    let half = n / 2;
    let lin: Vec<f32> = (0..n)
        .map(|i| {
            if i < half {
                1.0 + step * i as f32
            } else {
                0.0 - step * (steps - i) as f32
            }
        })
        .collect();
    let mut s: Vec<f32> = lin
        .iter()
        .map(|&v| if v != 0.0 { e / (e + (1.0 / v - 1.0)) } else { 0.0 })
        .collect();
    let last = s.iter().rev().find(|&&v| v != 0.0).copied().unwrap_or(1.0);
    let scale = (1.0 - last) / (1.0 - 0.1f32);
    for v in s.iter_mut().filter(|v| **v != 0.0) {
        *v = 1.0 - (1.0 - *v) / scale;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_ends() {
        let s = ltx2_sigmas(30);
        assert_eq!(s.len(), 31);
        assert_eq!(s[30], 0.0);
        assert!((s[29] - 0.1).abs() < 1e-6, "{}", s[29]);
        assert!((s[0] - 1.0).abs() < 1e-6);
        assert!(s.windows(2).all(|w| w[0] > w[1]));
    }

    #[test]
    fn guidance_terms() {
        let g = GuiderParams {
            rescale: 0.0,
            ..GuiderParams::video()
        };
        let cond = [1.0f32, 2.0];
        let un = [0.0f32, 1.0];
        let p = g.calculate(&cond, Some(&un), None, None, &[1.0], 2);
        assert_eq!(p, vec![3.0, 4.0]);
    }
}
