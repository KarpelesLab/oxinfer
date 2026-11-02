//! Times the GPU backend's unmasked attention (128-wide heads) three ways and reports each one's
//! relative difference from the f32 simdgroup kernel: that kernel (`gpu::set_exact`), exact
//! attention on the matrix units (f32 operands, `gpu::set_exact_attention`) and the default
//! (half operands): `cargo run --release --example attn_bench -- [TOKENS] [HEADS] [LOGIT_STD]`
//! (defaults 8192, 24, 4: q and k entries are scaled so q.k / sqrt(128) has that spread).

use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let t: usize = args.next().map_or(Ok(8192), |a| a.parse())?;
    let heads: usize = args.next().map_or(Ok(24), |a| a.parse())?;
    let spread: f32 = args.next().map_or(Ok(4.0), |a| a.parse())?;
    let plat = oxinfer::platform::detect();
    oxinfer::pool::init(std::thread::available_parallelism().map_or(4, |n| n.get()));
    let gpu = oxinfer::gpu::init(&plat, oxinfer::platform::DevicePref::Auto)?.ok_or_else(|| anyhow::anyhow!("no GPU backend"))?;
    let d = heads * 128;
    let mut seed = 0x9e3779b97f4a7c15u64;
    let mut gauss = move || {
        // Box-Muller from a xorshift stream.
        let mut u = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        let (a, b) = (u(), u());
        ((-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()) as f32
    };
    // q.k / sqrt(128) has standard deviation s^2 for entries ~ N(0, s^2).
    let s = spread.sqrt();
    let q: Vec<f32> = (0..t * d).map(|_| gauss() * s).collect();
    let k: Vec<f32> = (0..t * d).map(|_| gauss() * s).collect();
    let v: Vec<f32> = (0..t * d).map(|_| gauss()).collect();
    let run = |exact: bool, exact_attention: bool| {
        oxinfer::gpu::set_exact(exact);
        oxinfer::gpu::set_exact_attention(exact_attention);
        let mut out = vec![0f32; t * d];
        let mut best = f64::MAX;
        for _ in 0..4 {
            let t0 = Instant::now();
            assert!(
                gpu.attention(&q, &k, &v, t, t, heads, 128, false, &mut out),
                "attention not handled"
            );
            best = best.min(t0.elapsed().as_secs_f64());
        }
        (out, best)
    };
    let (reference, tr) = run(true, true);
    println!("{t} tokens, {heads} heads, best of 4 (copies included):");
    println!("  f32 simdgroup kernel: {:.1} ms", tr * 1e3);
    for (name, exact_attention) in [("matrix units, f32", true), ("matrix units, half", false)] {
        let (out, time) = run(false, exact_attention);
        let (mut num, mut den) = (0f64, 0f64);
        for (a, b) in out.iter().zip(&reference) {
            num += ((a - b) as f64).powi(2);
            den += (*b as f64).powi(2);
        }
        println!("  {name}: {:.1} ms, relative difference {:.2e}", time * 1e3, (num / den).sqrt());
    }
    Ok(())
}
