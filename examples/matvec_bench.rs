//! Times one-token decode matmuls (random weights) the way a decode step streams them: a ring of
//! copies larger than the caches, cycled, so every call reads its weights from memory:
//! `cargo run --release --example matvec_bench -- [TYPE] [THREADS,..] [N_OUTxN_IN,..] [MB]`
//! (defaults Q8_0, 8 threads, Llama 3.2 1B's shapes, a 1 GB ring). Prints us per call and GB/s,
//! and the cost of an empty parallel region. `STREAM=1`: plain streaming reads of the ring
//! instead, in parallel regions of `REGION_KB` split into blocks of `BLOCK_KB` (the machine's
//! read bandwidth for a given split). `X4=1`: the same ring interleaved four rows at a time
//! (`oxinfer::ggml::repack`) too, timed alternately with the plain one.

use oxinfer::ggml::{GgmlType, GgufMatrix, random_blocks};
use oxinfer::kernels::matmul_quant_multi;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let kind_name = args.next().unwrap_or_else(|| "Q8_0".into());
    let threads: Vec<usize> = args
        .next()
        .unwrap_or_else(|| "8".into())
        .split(',')
        .map(|s| s.parse())
        .collect::<Result<_, _>>()?;
    let shapes: Vec<(usize, usize)> = args
        .next()
        .unwrap_or_else(|| "2048x2048,512x2048,8192x2048,2048x8192".into())
        .split(',')
        .map(|s| {
            let (a, b) = s.split_once('x').expect("shape NOUTxNIN");
            (a.parse().unwrap(), b.parse().unwrap())
        })
        .collect();
    let ring_mb: usize = args.next().map_or(Ok(1024), |s| s.parse())?;
    let kind = GgmlType::ALL
        .into_iter()
        .find(|k| k.name().eq_ignore_ascii_case(&kind_name))
        .ok_or_else(|| anyhow::anyhow!("unknown type {kind_name}"))?;
    if std::env::var_os("STREAM").is_some() {
        // Plain streaming reads (sum of u64 words) over a 1 GB buffer, split among the threads.
        let buf: Vec<u64> = (0..(ring_mb << 17)).map(|i| i as u64).collect();
        for &th in &threads {
            oxinfer::pool::init(th);
            // Regions of REGION_KB (default: all of it), each split into blocks of BLOCK_KB.
            let e = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
            let region = (e("REGION_KB", buf.len() / 128) * 128).min(buf.len());
            let chunk = (e("BLOCK_KB", 0) * 128).max(1);
            let chunk = if e("BLOCK_KB", 0) == 0 { region.div_ceil(th * 16) } else { chunk };
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t0 = Instant::now();
                for r in buf.chunks(region) {
                    oxinfer::pool::run(r.len().div_ceil(chunk), &|i| {
                        let s: u64 = r[i * chunk..((i + 1) * chunk).min(r.len())]
                            .as_chunks::<8>()
                            .0
                            .iter()
                            .fold(0u64, |a, c| a ^ c[0] ^ c[1] ^ c[2] ^ c[3] ^ c[4] ^ c[5] ^ c[6] ^ c[7]);
                        std::hint::black_box(s);
                    });
                }
                best = best.min(t0.elapsed().as_secs_f64());
            }
            println!("{th} threads: stream {:.1} GB/s", (buf.len() * 8) as f64 / best / 1e9);
        }
        return Ok(());
    }
    for &th in &threads {
        oxinfer::pool::init(th);
        // Empty regions.
        let n = 20000;
        let t0 = Instant::now();
        for _ in 0..n {
            oxinfer::pool::run(th * 4, &|_| {});
        }
        println!("{th} threads: empty region {:.2} us", t0.elapsed().as_secs_f64() * 1e6 / n as f64);
        for &(n_out, n_in) in &shapes {
            let mut seed = 1u64;
            let one = GgufMatrix::from_bytes(kind, &random_blocks(kind, n_out * n_in / kind.block_size(), &mut seed), n_out, n_in)?;
            let copies = (ring_mb << 20).div_ceil(one.bytes()).max(2);
            let ring: Vec<GgufMatrix> = (0..copies)
                .map(|i| {
                    let mut s = i as u64 + 2;
                    GgufMatrix::from_bytes(kind, &random_blocks(kind, n_out * n_in / kind.block_size(), &mut s), n_out, n_in).unwrap()
                })
                .collect();
            let x: Vec<f32> = (0..n_in).map(|i| ((i * 7919) % 2001) as f32 / 1000.0 - 1.0).collect();
            let mut y = vec![0f32; n_out];
            let x4 = std::env::var("X4").is_ok_and(|v| v == "1") && oxinfer::ggml::can_interleave(kind, n_out, n_in);
            let ring4: Vec<GgufMatrix> = if x4 {
                ring.iter().map(|m| m.to_interleaved().unwrap()).collect()
            } else {
                Vec::new()
            };
            let rings: Vec<(&str, &[GgufMatrix])> = if x4 {
                vec![("plain", &ring[..]), ("x4", &ring4[..])]
            } else {
                vec![("", &ring[..])]
            };
            for (_, r) in &rings {
                for m in r.iter() {
                    matmul_quant_multi(&mut [(m, &mut y[..])], &x, 1);
                }
            }
            let mut best = vec![f64::MAX; rings.len()];
            for _ in 0..5 {
                for (b, (_, r)) in best.iter_mut().zip(&rings) {
                    let t0 = Instant::now();
                    for m in r.iter() {
                        matmul_quant_multi(&mut [(m, &mut y[..])], &x, 1);
                    }
                    *b = b.min(t0.elapsed().as_secs_f64() / copies as f64);
                }
            }
            for (b, (name, _)) in best.iter().zip(&rings) {
                println!(
                    "  {kind_name} {name} {n_out}x{n_in} ({:.1} MB): {:7.1} us  {:6.1} GB/s",
                    one.bytes() as f64 / 1e6,
                    b * 1e6,
                    one.bytes() as f64 / b / 1e9
                );
            }
        }
    }
    Ok(())
}
