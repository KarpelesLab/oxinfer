//! PyTorch's CPU random generator (`torch.Generator("cpu")`, `torch.manual_seed`): the Mersenne
//! Twister of `at::mt19937` and `torch.randn`'s float32 path (`normal_fill` of
//! `aten/src/ATen/native/cpu/DistributionTemplates.h`), so that pipelines can draw the same noise
//! as diffusers (`torch.randn(shape, generator=torch.Generator("cpu").manual_seed(seed))`) and
//! ComfyUI (`comfy.sample.prepare_noise`; its samplers' `default_noise_sampler` on the CPU).
//!
//! `randn` of `n >= 16` float32 values: `n` uniforms `(u32 & 0xFFFFFF) / 2^24`, then each block of
//! 16 turned into normals by Box-Muller over pairs `(j, j + 8)`: `r = sqrt(-2 ln(1 - u_j))`, the
//! angle `2 pi u_{j+8}` (computed in float64, kept in float32), all in float32: `x_j = r cos`, `x_{j+8} = r
//! sin`; when `n` is not a multiple of 16 the last 16 values are drawn again and redone. The same
//! libm as PyTorch's build gives the same bits (the system's on macOS and Linux); PyTorch on x86
//! CPUs with AVX2 takes a vectorized variant whose last bits can differ.

/// `at::mt19937` with PyTorch's seeding.
pub struct TorchGenerator {
    state: [u32; 624],
    next: usize,
}

impl TorchGenerator {
    /// `torch.Generator("cpu").manual_seed(seed)` (the low 32 bits seed the twister, as in
    /// PyTorch).
    pub fn new(seed: u64) -> Self {
        let mut state = [0u32; 624];
        state[0] = seed as u32;
        for j in 1..624 {
            state[j] = 1_812_433_253u32
                .wrapping_mul(state[j - 1] ^ (state[j - 1] >> 30))
                .wrapping_add(j as u32);
        }
        Self { state, next: 624 }
    }

    fn twist(&mut self) {
        const UPPER: u32 = 0x8000_0000;
        const LOWER: u32 = 0x7fff_ffff;
        let s = &mut self.state;
        for i in 0..624 {
            let y = (s[i] & UPPER) | (s[(i + 1) % 624] & LOWER);
            let mut v = s[(i + 397) % 624] ^ (y >> 1);
            if y & 1 != 0 {
                v ^= 0x9908_b0df;
            }
            s[i] = v;
        }
        self.next = 0;
    }

    /// The next 32-bit output.
    pub fn random(&mut self) -> u32 {
        if self.next >= 624 {
            self.twist();
        }
        let mut y = self.state[self.next];
        self.next += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// `at::uniform_real_distribution<float>(0, 1)`.
    fn uniform(&mut self) -> f32 {
        ((self.random() & 0x00ff_ffff) as f64 * (1.0 / (1u64 << 24) as f64)) as f32
    }

    /// `torch.randn(n)` in float32 from this generator (see the module docs; `n >= 16`).
    pub fn randn(&mut self, n: usize) -> Vec<f32> {
        assert!(n >= 16, "torch's normal_fill path needs at least 16 values");
        let mut d: Vec<f32> = (0..n).map(|_| self.uniform()).collect();
        for b in d.as_chunks_mut::<16>().0 {
            fill16(b);
        }
        if !n.is_multiple_of(16) {
            let tail = &mut d[n - 16..];
            for v in tail.iter_mut() {
                *v = self.uniform();
            }
            fill16(tail);
        }
        d
    }
}

/// `normal_fill_16` with mean 0, std 1.
fn fill16(d: &mut [f32]) {
    for j in 0..8 {
        let u1 = 1.0f32 - d[j];
        let u2 = d[j + 8];
        let radius = (-2.0f32 * u1.ln()).sqrt();
        // `scalar_t theta = 2.0f * c10::pi<double> * u2`: the product in float64, kept as float32.
        let theta = (2.0f64 * std::f64::consts::PI * u2 as f64) as f32;
        d[j] = radius * theta.cos();
        d[j + 8] = radius * theta.sin();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt19937_reference_outputs() {
        // std::mt19937 seeded with 5489 (the C++ standard's check value: the 10000th output).
        let mut g = TorchGenerator::new(5489);
        let mut v = 0;
        for _ in 0..10000 {
            v = g.random();
        }
        assert_eq!(v, 4_123_659_995);
    }

    #[test]
    fn randn_matches_torch() {
        // torch.randn(20, generator=torch.Generator("cpu").manual_seed(42)) (PyTorch 2.15, macOS
        // arm64), bit for bit: the first 16 values and the redone tail (system libm).
        let want: [i32; 20] = [
            1073128743,
            1069440852,
            1063687527,
            -1073299235,
            -1086188318,
            1066010192,
            1062011203,
            1071062675,
            1052128065,
            -1087388026,
            -1090741964,
            1048004756,
            -1100140819,
            1026231296,
            -1098854791,
            1063002032,
            -1096903545,
            -1094018415,
            1062054972,
            -1088479008,
        ];
        let v = TorchGenerator::new(42).randn(20);
        // Bit for bit with macOS's libm; another system's `ln` / `cos` can round a last bit
        // otherwise (glibc on x86-64: one value 1 ulp away; PyTorch itself there, on its AVX2
        // path, is up to 5 ulps from these).
        let ulps = if cfg!(target_os = "macos") { 0 } else { 2 };
        for (i, (a, b)) in v.iter().zip(want).enumerate() {
            let d = (a.to_bits() as i64 - b as u32 as i64).abs();
            assert!(d <= ulps, "{i}: {a} vs {} ({d} ulps)", f32::from_bits(b as u32));
        }
    }

    /// `OXINFER_TORCH_RANDN=FILE:SEED`: a dump of `torch.randn(..., generator=...manual_seed(SEED))`
    /// (raw f32) compared value for value.
    #[test]
    #[ignore]
    fn randn_matches_torch_dump() {
        let spec = std::env::var("OXINFER_TORCH_RANDN").expect("OXINFER_TORCH_RANDN=FILE:SEED");
        let (f, seed) = spec.rsplit_once(':').unwrap();
        let b = std::fs::read(f).unwrap();
        let want: Vec<f32> = b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
        let v = TorchGenerator::new(seed.parse().unwrap()).randn(want.len());
        let same = v.iter().zip(&want).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
        let worst = v.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        eprintln!("{same} of {} bit-identical, worst |diff| {worst:e}", want.len());
        assert_eq!(same, want.len());
    }
}
