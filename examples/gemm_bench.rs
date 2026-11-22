//! Times the CPU prefill GEMMs (random weights): sme, hybrid, int8x2, f32, int8 and fast:
//! `cargo run --release --example gemm_bench -- [TYPES] [THREADS] [N_OUT] [N_IN] [TOKENS]`
//! (TYPES: GGUF types like `Q8_0,Q4_K`, or `OXQ8` / `OXQ4`; defaults Q8_0, 1 thread,
//! 4096 x 4096, 512 tokens). Prints GOP/s (2 per multiply-add), best of 7 interleaved rounds.

use oxinfer::ggml::{GgmlType, GgufMatrix, random_blocks};
use oxinfer::kernels::{PrefillGemm, QMat, matmul_quant_multi, set_prefill_gemm};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let types = args.next().unwrap_or_else(|| "Q8_0".into());
    let threads: usize = args.next().map_or(Ok(1), |t| t.parse())?;
    let n_out: usize = args.next().map_or(Ok(4096), |t| t.parse())?;
    let n_in: usize = args.next().map_or(Ok(4096), |t| t.parse())?;
    let t: usize = args.next().map_or(Ok(512), |t| t.parse())?;
    let rounds: usize = std::env::var("ROUNDS").ok().and_then(|r| r.parse().ok()).unwrap_or(7);
    oxinfer::pool::init(threads);
    let wf = || -> Vec<f32> {
        let mut s = 7u64;
        (0..n_out * n_in)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 40) % 2001) as f32 / 1000.0 - 1.0
            })
            .collect()
    };
    for name in types.split(',') {
        let mut seed = 1u64;
        let a = if name.eq_ignore_ascii_case("oxq8") {
            bench(&oxinfer::quant::quantize_f32(&wf(), n_out, n_in), t, rounds)
        } else if name.eq_ignore_ascii_case("oxq4") {
            bench(&oxinfer::quant::quantize_q4_f32(&wf(), n_out, n_in), t, rounds)
        } else {
            let kind = GgmlType::ALL
                .into_iter()
                .find(|k| k.name().eq_ignore_ascii_case(name))
                .ok_or_else(|| anyhow::anyhow!("unknown type {name}"))?;
            let m = GgufMatrix::from_bytes(kind, &random_blocks(kind, n_out * n_in / kind.block_size(), &mut seed), n_out, n_in)?;
            bench(&m, t, rounds)
        };
        let ops = 2.0 * (n_out * n_in * t) as f64 / 1e9;
        let g = |s: f64| ops / s;
        println!(
            "{name:<8} sme {:7.1}  hybrid {:7.1}  int8x2 {:7.1}  f32 {:7.1}  int8 {:7.1}  fast {:7.1} GOP/s",
            g(a[0]),
            g(a[1]),
            g(a[2]),
            g(a[3]),
            g(a[4]),
            g(a[5])
        );
    }
    Ok(())
}

fn bench<M: QMat>(m: &M, t: usize, rounds: usize) -> [f64; 6] {
    let n_in = m.n_in();
    let x: Vec<f32> = (0..t * n_in).map(|i| ((i * 7919) % 2001) as f32 / 1000.0 - 1.0).collect();
    let mut y = vec![0f32; t * m.n_out()];
    let mut best = [f64::MAX; 6];
    for _ in 0..rounds {
        for (i, g) in [
            PrefillGemm::Sme,
            PrefillGemm::Hybrid,
            PrefillGemm::Int8x2,
            PrefillGemm::F32,
            PrefillGemm::Int8,
            PrefillGemm::Fast,
        ]
        .into_iter()
        .enumerate()
        {
            set_prefill_gemm(g);
            let t0 = Instant::now();
            matmul_quant_multi(&mut [(m, &mut y[..])], &x, t);
            best[i] = best[i].min(t0.elapsed().as_secs_f64());
        }
    }
    best
}
