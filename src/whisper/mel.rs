//! Whisper's log-mel front end, following transformers' `WhisperFeatureExtractor` (its torch
//! path): 16 kHz mono, a periodic Hann window of 400 samples, hop 160, `center=True` with reflect
//! padding, power spectrum, a Slaney-scale / Slaney-normalized mel filterbank (80 or 128 bins,
//! 0-8 kHz), `log10(max(mel, 1e-10))`, the last frame dropped, clamped to `max - 8` over the whole
//! input, then `(x + 4) / 4`.
//!
//! Clips of up to 30 s are zero-padded to 30 s first (3000 frames), as the processor does by
//! default; longer audio is featurized whole (transformers' long-form call: `truncation=False,
//! padding="longest"`), so the clamp sees all of it.
//!
//! The DFT runs in f64 (a mixed-radix FFT, 400 = 2^4 * 5^2) and the filterbank sums in f64, so the
//! only differences from torch's f32 pipeline are its own rounding errors (about 1e-6 relative).
//! Gemma 4's front end (`crate::audio`) uses a different window, FFT size, mel scale and log, so
//! only the overall shape is shared.

use crate::pool;

pub const SAMPLE_RATE: usize = 16_000;
pub const N_FFT: usize = 400;
pub const HOP: usize = 160;
/// One decoding window: 30 s of samples, 3000 mel frames.
pub const CHUNK_SAMPLES: usize = 30 * SAMPLE_RATE;
pub const CHUNK_FRAMES: usize = CHUNK_SAMPLES / HOP;
const BINS: usize = N_FFT / 2 + 1;

/// Log-mel features, time-major: `data[t * n_mels + m]`.
#[derive(Clone, Debug)]
pub struct Features {
    pub n_mels: usize,
    pub frames: usize,
    pub data: Vec<f32>,
}

impl Features {
    /// The frame rows `t0..t0 + CHUNK_FRAMES`, zero-filled past the end (transformers pads the
    /// last window's features with 0, not with the features of silence).
    pub fn window(&self, t0: usize) -> Vec<f32> {
        let mut out = vec![0f32; CHUNK_FRAMES * self.n_mels];
        let t1 = (t0 + CHUNK_FRAMES).min(self.frames);
        if t1 > t0 {
            out[..(t1 - t0) * self.n_mels].copy_from_slice(&self.data[t0 * self.n_mels..t1 * self.n_mels]);
        }
        out
    }
}

/// numpy's `linspace(a, b, n)` (endpoint included, last value exact).
fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    let step = (b - a) / (n - 1) as f64;
    let mut v: Vec<f64> = (0..n).map(|i| a + i as f64 * step).collect();
    v[n - 1] = b;
    v
}

const MIN_LOG_HZ: f64 = 1000.0;
const MIN_LOG_MEL: f64 = 15.0;

fn hz_to_mel(f: f64) -> f64 {
    if f >= MIN_LOG_HZ {
        MIN_LOG_MEL + (f / MIN_LOG_HZ).ln() * (27.0 / 6.4f64.ln())
    } else {
        3.0 * f / 200.0
    }
}

fn mel_to_hz(m: f64) -> f64 {
    if m >= MIN_LOG_MEL {
        MIN_LOG_HZ * ((6.4f64.ln() / 27.0) * (m - MIN_LOG_MEL)).exp()
    } else {
        200.0 * m / 3.0
    }
}

/// transformers' `mel_filter_bank(201, n_mels, 0, 8000, 16000, norm="slaney", mel_scale="slaney")`
/// as f64, `[BINS][n_mels]`.
pub fn mel_filters(n_mels: usize) -> Vec<f64> {
    let mel = linspace(hz_to_mel(0.0), hz_to_mel(8000.0), n_mels + 2);
    let filt: Vec<f64> = mel.iter().map(|&m| mel_to_hz(m)).collect();
    let fft: Vec<f64> = linspace(0.0, (SAMPLE_RATE / 2) as f64, BINS);
    let mut out = vec![0f64; BINS * n_mels];
    for (b, &f) in fft.iter().enumerate() {
        for m in 0..n_mels {
            let down = -(filt[m] - f) / (filt[m + 1] - filt[m]);
            let up = (filt[m + 2] - f) / (filt[m + 2] - filt[m + 1]);
            let enorm = 2.0 / (filt[m + 2] - filt[m]);
            out[b * n_mels + m] = down.min(up).max(0.0) * enorm;
        }
    }
    out
}

/// A mixed-radix complex FFT of one fixed size (recursive decimation in time).
pub(crate) struct Fft {
    n: usize,
    factors: Vec<usize>,
    /// `exp(-2 pi i k / n)` for `k < n`.
    tw: Vec<(f64, f64)>,
}

impl Fft {
    pub(crate) fn new(n: usize) -> Self {
        let mut factors = Vec::new();
        let mut m = n;
        for p in [4, 2, 3, 5] {
            while m.is_multiple_of(p) {
                factors.push(p);
                m /= p;
            }
        }
        let mut p = 7;
        while m > 1 {
            while m.is_multiple_of(p) {
                factors.push(p);
                m /= p;
            }
            p += 2;
        }
        let tw = (0..n)
            .map(|k| {
                let a = -2.0 * std::f64::consts::PI * k as f64 / n as f64;
                (a.cos(), a.sin())
            })
            .collect();
        Self { n, factors, tw }
    }

    /// `out = DFT(x)`, both of length `n`.
    pub(crate) fn run(&self, x: &[(f64, f64)], out: &mut [(f64, f64)]) {
        debug_assert_eq!(x.len(), self.n);
        let mut scratch = vec![(0f64, 0f64); self.factors.iter().copied().max().unwrap_or(1)];
        self.rec(x, 1, self.n, out, 0, &mut scratch);
    }

    fn rec(&self, x: &[(f64, f64)], stride: usize, n: usize, out: &mut [(f64, f64)], fi: usize, scratch: &mut [(f64, f64)]) {
        if n == 1 {
            out[0] = x[0];
            return;
        }
        let p = self.factors[fi];
        let m = n / p;
        for r in 0..p {
            self.rec(&x[r * stride..], stride * p, m, &mut out[r * m..(r + 1) * m], fi + 1, scratch);
        }
        // X[k + q m] = sum_r W_n^(r (k + q m)) Y_r[k]
        let step = self.n / n;
        for k in 0..m {
            for q in 0..p {
                let (mut sr, mut si) = (0f64, 0f64);
                for r in 0..p {
                    let (yr, yi) = out[r * m + k];
                    let (wr, wi) = self.tw[(r * (k + q * m) * step) % self.n];
                    sr += yr * wr - yi * wi;
                    si += yr * wi + yi * wr;
                }
                scratch[q] = (sr, si);
            }
            for q in 0..p {
                out[k + q * m] = scratch[q];
            }
        }
    }
}

/// Log-mel features of 16 kHz mono samples; see the module documentation for the padding rules.
pub fn log_mel(pcm: &[f32], n_mels: usize) -> Features {
    let padded;
    let x: &[f32] = if pcm.len() <= CHUNK_SAMPLES {
        let mut v = pcm.to_vec();
        v.resize(CHUNK_SAMPLES, 0.0);
        padded = v;
        &padded
    } else {
        pcm
    };
    let n = x.len();
    // torch.stft(center=True) gives n / hop + 1 frames; the extractor drops the last.
    let frames = n / HOP;
    let filters = mel_filters(n_mels);
    let window: Vec<f64> = (0..N_FFT)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64).cos())
        .collect();
    let fft = Fft::new(N_FFT);
    let half = (N_FFT / 2) as isize;
    // Reflect padding by n_fft / 2 on both sides (the edge sample is not repeated).
    let sample = |i: isize| -> f64 {
        let last = n as isize - 1;
        let j = if i < 0 {
            -i
        } else if i > last {
            2 * last - i
        } else {
            i
        };
        x[j.clamp(0, last) as usize] as f64
    };
    let mut data = vec![0f32; frames * n_mels];
    const ROWS: usize = 64;
    let tasks = frames.div_ceil(ROWS);
    let dp = SendPtr(data.as_mut_ptr());
    pool::run(tasks, &|task| {
        let (t0, t1) = (task * ROWS, ((task + 1) * ROWS).min(frames));
        let mut buf = vec![(0f64, 0f64); N_FFT];
        let mut spec = vec![(0f64, 0f64); N_FFT];
        let mut power = [0f64; BINS];
        for t in t0..t1 {
            let start = (t * HOP) as isize - half;
            for (i, b) in buf.iter_mut().enumerate() {
                *b = (sample(start + i as isize) * window[i], 0.0);
            }
            fft.run(&buf, &mut spec);
            for (p, &(re, im)) in power.iter_mut().zip(&spec[..BINS]) {
                *p = re * re + im * im;
            }
            // SAFETY: each task writes its own frame rows.
            let row = unsafe { std::slice::from_raw_parts_mut(dp.get().add(t * n_mels), n_mels) };
            for (m, r) in row.iter_mut().enumerate() {
                let mut s = 0f64;
                for b in 0..BINS {
                    s += filters[b * n_mels + m] * power[b];
                }
                *r = s.max(1e-10).log10() as f32;
            }
        }
    });
    let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for v in data.iter_mut() {
        *v = ((*v).max(max - 8.0) + 4.0) / 4.0;
    }
    Features { n_mels, frames, data }
}

#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    fn get(self) -> *mut f32 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_matches_direct_dft() {
        for n in [400usize, 12, 7, 64] {
            let x: Vec<(f64, f64)> = (0..n).map(|i| (((i * 7919) % 13) as f64 - 6.0, ((i * 31) % 5) as f64)).collect();
            let mut y = vec![(0.0, 0.0); n];
            Fft::new(n).run(&x, &mut y);
            for k in 0..n {
                let (mut r, mut im) = (0f64, 0f64);
                for (j, &(a, b)) in x.iter().enumerate() {
                    let ang = -2.0 * std::f64::consts::PI * (j * k) as f64 / n as f64;
                    r += a * ang.cos() - b * ang.sin();
                    im += a * ang.sin() + b * ang.cos();
                }
                assert!((r - y[k].0).abs() < 1e-9 && (im - y[k].1).abs() < 1e-9, "n={n} k={k}");
            }
        }
    }

    #[test]
    fn filters_are_slaney_normalized() {
        // Each triangle has area ~1 in Hz (slaney norm 2 / width): sum * bin width ~ 1.
        let f = mel_filters(80);
        let bin_hz = 8000.0 / (BINS - 1) as f64;
        for m in 10..80 {
            let s: f64 = (0..BINS).map(|b| f[b * 80 + m]).sum::<f64>() * bin_hz;
            assert!((s - 1.0).abs() < 0.1, "mel {m}: area {s}");
        }
    }

    #[test]
    fn silence_and_shape() {
        let f = log_mel(&vec![0f32; 16000], 80);
        assert_eq!(f.frames, 3000);
        // All -10 before the clamp, so (-10 + 4) / 4.
        assert!(f.data.iter().all(|&v| (v + 1.5).abs() < 1e-6));
        let long = log_mel(&vec![0.1f32; CHUNK_SAMPLES + 1000], 128);
        assert_eq!(long.frames, (CHUNK_SAMPLES + 1000) / HOP);
        assert_eq!(long.window(long.frames - 10).len(), CHUNK_FRAMES * 128);
    }
}
