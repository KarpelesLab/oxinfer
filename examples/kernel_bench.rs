//! Times one GGUF-format matmul (random weights) for several token counts with each
//! multi-token kernel choice: `cargo run --release --example kernel_bench -- [TYPES] [THREADS] [N_OUT] [N_IN]`
//! (TYPES like `IQ3_XXS,Q5_K`; defaults: the K-quants and i-quants of the HauhauCS GGUFs, 1
//! thread, 4096 x 5120).

use oxinfer::ggml::{GgmlType, GgufMatrix, random_blocks};
use oxinfer::kernels::{MultiTokenKernels, matmul_quant_multi, set_multi_token_kernels};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let types = args
        .next()
        .unwrap_or_else(|| "IQ3_XXS,IQ3_S,Q5_K,IQ4_XS,Q6_K,Q4_K,Q3_K,Q2_K".into());
    let threads: usize = args.next().map_or(Ok(1), |t| t.parse())?;
    let n_out: usize = args.next().map_or(Ok(4096), |t| t.parse())?;
    let n_in: usize = args.next().map_or(Ok(5120), |t| t.parse())?;
    oxinfer::pool::init(threads);
    oxinfer::kernels::set_decode_max_tokens(15);
    let modes = [
        ("pairs", MultiTokenKernels::Pairs),
        ("dot", MultiTokenKernels::Dot),
        ("auto", MultiTokenKernels::Auto),
    ];
    println!(
        "i8mm: {}; ns per super-block (256 weights) per token, best of 15",
        oxinfer::kernels::cpu_has_i8mm()
    );
    for name in types.split(',') {
        let kind = GgmlType::ALL
            .into_iter()
            .find(|k| k.name().eq_ignore_ascii_case(name))
            .ok_or_else(|| anyhow::anyhow!("unknown type {name}"))?;
        let mut seed = 1u64;
        let m = GgufMatrix::from_bytes(kind, &random_blocks(kind, n_out * n_in / kind.block_size(), &mut seed), n_out, n_in)?;
        let mut line = format!("{:<8}", kind.name());
        for t in [1usize, 2, 3, 4, 6, 9] {
            let x: Vec<f32> = (0..t * n_in).map(|i| ((i * 7919) % 2001) as f32 / 1000.0 - 1.0).collect();
            let mut y = vec![0f32; t * n_out];
            // Modes interleaved round by round (the machine may be shared), best of all rounds.
            let mut best = [f64::MAX; 3];
            for _ in 0..15 {
                for (i, (_, mode)) in modes.iter().enumerate() {
                    if t == 1 && *mode != MultiTokenKernels::Pairs {
                        continue;
                    }
                    set_multi_token_kernels(*mode);
                    let t0 = Instant::now();
                    matmul_quant_multi(&mut [(&m, &mut y[..])], &x, t);
                    best[i] = best[i].min(t0.elapsed().as_secs_f64());
                }
            }
            for (i, (mname, _)) in modes.iter().enumerate() {
                if best[i] < f64::MAX {
                    let ns = best[i] * 1e9 / (n_out * n_in * t) as f64 * 256.0;
                    line += &format!(" t{t} {mname} {ns:.2}");
                }
            }
        }
        println!("{line}");
    }
    Ok(())
}
