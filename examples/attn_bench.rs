//! Times `oxinfer::sd::ops::attention` (bidirectional) on random inputs:
//!
//! ```text
//! attn_bench ROWS HEADS HEAD_DIM [REPEATS]
//! ```
//!
//! `OXINFER_ATTN_TWO_PASS=1` selects the one-head-at-a-time kernel instead of the blocked one.

fn main() {
    let a: Vec<usize> = std::env::args().skip(1).map(|s| s.parse().expect("a number")).collect();
    let (n, heads, dh) = (a[0], a[1], a[2]);
    let reps = a.get(3).copied().unwrap_or(3);
    oxinfer::gpu::init(&oxinfer::platform::detect(), oxinfer::platform::DevicePref::Cpu).unwrap();
    let mut rng = oxinfer::sd::ops::Rng::new(1);
    let d = heads * dh;
    let (q, k, v) = (rng.normals(n * d), rng.normals(n * d), rng.normals(n * d));
    let _ = oxinfer::sd::ops::attention(&q, &k, &v, n, n, heads, dh, false);
    let t = std::time::Instant::now();
    for _ in 0..reps {
        std::hint::black_box(oxinfer::sd::ops::attention(&q, &k, &v, n, n, heads, dh, false));
    }
    let s = t.elapsed().as_secs_f64() / reps as f64;
    let flops = 4.0 * (n * n * d) as f64;
    println!("{n} rows, {heads} x {dh}: {:.1} ms, {:.0} GFLOP/s", s * 1e3, flops / s / 1e9);
}
