//! Memory bandwidth and matmul-kernel microbenchmark.
//! usage: membench [GB]   (default 4 GB buffer)
// The tile macros expand `r + $R <= rows` with `$R` = 1 in one instance.
#![allow(clippy::int_plus_one)]

use oxinfer::quant;
use oxinfer::{kernels, pool};
use std::time::Instant;

fn bench_read(buf: &[u64], threads: usize) -> f64 {
    pool::init(threads);
    let bytes = buf.len() * 8;
    let chunk = (buf.len() / (threads * 16)).max(1 << 16);
    let n = buf.len().div_ceil(chunk);
    let mut best = 0f64;
    for _ in 0..3 {
        let t = Instant::now();
        let total = std::sync::atomic::AtomicU64::new(0);
        pool::run(n, &|i| {
            let c = &buf[i * chunk..((i + 1) * chunk).min(buf.len())];
            total.fetch_add(c.iter().fold(0u64, |a, &b| a ^ b), std::sync::atomic::Ordering::Relaxed);
        });
        std::hint::black_box(total);
        let gbs = bytes as f64 / t.elapsed().as_secs_f64() / 1e9;
        best = best.max(gbs);
    }
    best
}

fn bench_matmul(w: &[u16], n_out: usize, n_in: usize, t: usize, threads: usize) -> f64 {
    pool::init(threads);
    let x = vec![0.5f32; t * n_in];
    let mut y = vec![0f32; t * n_out];
    let mut best = 0f64;
    for _ in 0..3 {
        let ti = Instant::now();
        kernels::matmul(w, n_out, n_in, &x, t, &mut y);
        let gbs = (w.len() * 2) as f64 / ti.elapsed().as_secs_f64() / 1e9;
        best = best.max(gbs);
    }
    std::hint::black_box(&y);
    best
}

/// `membench gpu [n_out n_in]`: checks and times the accelerator matmul against the CPU kernels.
#[cfg(feature = "gpu")]
fn bench_gpu(n_out: usize, n_in: usize) {
    use oxinfer::gpu::{self, WeightRef};
    let plat = oxinfer::platform::detect();
    println!("{}", plat.summary());
    let Ok(Some(be)) = gpu::init(&plat, oxinfer::platform::DevicePref::Auto) else {
        println!("no accelerator backend");
        return;
    };
    pool::init(pool::threads().max(1));
    let w: Vec<u16> = (0..n_out * n_in)
        .map(|i| ((((i as u32).wrapping_mul(2654435761) >> 8) as f32 / (1u32 << 24) as f32 - 0.5).to_bits() >> 16) as u16)
        .collect();
    let q8 = quant::quantize_bf16(&w, n_out, n_in);
    let q4 = quant::quantize_q4(&w, n_out, n_in);
    // KINDS=Q4_K,IQ3_S,..: GGUF formats to check as well (random blocks with sane scales), and
    // F8_E4M3: FP8 with 128 x 128 block scales, as the official Mistral Large 4 / DeepSeek
    // checkpoints store them.
    let kinds = std::env::var("KINDS").unwrap_or_default();
    let mut gg: Vec<oxinfer::ggml::GgufMatrix> = kinds
        .split(',')
        .filter_map(|k| oxinfer::ggml::GgmlType::ALL.into_iter().find(|t| t.name() == k))
        .map(|k| {
            let raw = oxinfer::ggml::random_blocks(k, n_out * n_in / k.block_size(), &mut 1);
            oxinfer::ggml::GgufMatrix::from_bytes(k, &raw, n_out, n_in).unwrap()
        })
        .collect();
    if kinds.split(',').any(|k| k == "F8_E4M3") {
        let mut q = quant::PageVec::<u8>::zeroed(n_out * n_in);
        for (i, b) in q.iter_mut().enumerate() {
            // Any byte but the NaNs (0x7f, 0xff).
            let v = ((i as u32).wrapping_mul(2654435761) >> 13) as u8;
            *b = if v & 0x7f == 0x7f { v ^ 1 } else { v };
        }
        let ns = n_out.div_ceil(128) * n_in.div_ceil(128);
        let mut s = quant::PageVec::<f32>::zeroed(ns);
        for (i, v) in s.iter_mut().enumerate() {
            *v = 1e-3 * (1.0 + (i % 7) as f32 / 7.0);
        }
        gg.push(oxinfer::ggml::GgufMatrix::fp8(q.into(), s.into(), n_out, n_in, (128, 128)).unwrap());
    }
    let mut refs: Vec<(&str, WeightRef)> = vec![
        ("bf16", WeightRef::Bf16(&w)),
        ("q8", WeightRef::Q8(&q8)),
        ("q4", WeightRef::Q4(&q4)),
    ];
    refs.extend(gg.iter().map(|m| (m.kind.name(), WeightRef::Gguf(m))));
    for (name, wr) in refs.iter() {
        let dev = be.upload(wr, n_out, n_in).expect("upload");
        println!("\n{name} [{n_out} x {n_in}] on {}:", be.name());
        // T=a,b,c overrides the token counts (e.g. convolution-sized batches).
        let ts: Vec<usize> = std::env::var("T")
            .ok()
            .map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect())
            .unwrap_or_else(|| vec![1, 2, 4, 8, 16, 64, 256, 512]);
        for &t in &ts {
            let x: Vec<f32> = (0..t * n_in).map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0).collect();
            let mut y_ref = vec![0f32; t * n_out];
            match wr {
                WeightRef::Bf16(w) => kernels::matmul(w, n_out, n_in, &x, t, &mut y_ref),
                WeightRef::Q8(m) => kernels::matmul_quant_multi(&mut [(*m, &mut y_ref[..])], &x, t),
                WeightRef::Q4(m) => kernels::matmul_quant_multi(&mut [(*m, &mut y_ref[..])], &x, t),
                WeightRef::Gguf(m) => kernels::matmul_quant_multi(&mut [(*m, &mut y_ref[..])], &x, t),
            }
            let mut y = vec![0f32; t * n_out];
            let mut best_gpu = f64::MAX;
            let reps: usize = std::env::var("REPS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(if t >= 256 { 3 } else { 5 });
            let wanted = be.wants(n_out, n_in, t);
            for _ in 0..reps {
                let ti = Instant::now();
                let wr2 = match wr {
                    WeightRef::Bf16(w) => WeightRef::Bf16(w),
                    WeightRef::Q8(m) => WeightRef::Q8(m),
                    WeightRef::Q4(m) => WeightRef::Q4(m),
                    WeightRef::Gguf(m) => WeightRef::Gguf(m),
                };
                let mut jobs = [gpu::Job {
                    w: wr2,
                    dev: &dev,
                    n_out,
                    y: &mut y,
                }];
                if !be.matmul(&x, t, n_in, &mut jobs) {
                    best_gpu = f64::NAN;
                    break;
                }
                best_gpu = best_gpu.min(ti.elapsed().as_secs_f64());
            }
            let mut best_cpu = f64::MAX;
            for _ in 0..reps {
                let ti = Instant::now();
                match wr {
                    WeightRef::Bf16(w) => kernels::matmul(w, n_out, n_in, &x, t, &mut y_ref),
                    WeightRef::Q8(m) => kernels::matmul_quant_multi(&mut [(*m, &mut y_ref[..])], &x, t),
                    WeightRef::Q4(m) => kernels::matmul_quant_multi(&mut [(*m, &mut y_ref[..])], &x, t),
                    WeightRef::Gguf(m) => kernels::matmul_quant_multi(&mut [(*m, &mut y_ref[..])], &x, t),
                }
                best_cpu = best_cpu.min(ti.elapsed().as_secs_f64());
            }
            if std::env::var("SHOW").is_ok() {
                println!("    gpu {:?}\n    cpu {:?}", &y[t * n_out - 4..], &y_ref[t * n_out - 4..]);
            }
            let scale = y_ref.iter().fold(0f32, |a, &b| a.max(b.abs())).max(1e-6);
            let max_rel = y.iter().zip(&y_ref).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max) / scale;
            let flops = 2.0 * (t * n_out * n_in) as f64;
            println!(
                "  t={t:>3}: gpu {:>8.3} ms ({:>7.1} GB/s, {:>7.1} GFLOP/s) | cpu {:>8.3} ms ({:>7.1} GFLOP/s) | max rel err {max_rel:.2e}{}",
                best_gpu * 1e3,
                wr.bytes() as f64 / best_gpu / 1e9,
                flops / best_gpu / 1e9,
                best_cpu * 1e3,
                flops / best_cpu / 1e9,
                if wanted { "" } else { "  (backend would decline this size)" }
            );
        }
    }
    if let Some(s) = be.stats() {
        println!("\ngpu totals: {s}");
    }
}

/// `membench gguf [n_out n_in [threads]]`: decode matmul throughput of the GGUF formats against
/// oxinfer's Q4 / Q8, over enough copies of the matrix (>= 1.5 GB) that the caches do not help.
fn bench_gguf(n_out: usize, n_in: usize, threads: usize) {
    use oxinfer::ggml::{GgmlType, GgufMatrix};
    use oxinfer::kernels::QMat;
    pool::init(threads);
    let mut seed = 1u64;
    let mut rnd = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 33) as u32
    };
    fn run<M: QMat>(name: &str, ms: &[M], n_in: usize, rnd: &mut dyn FnMut() -> u32) {
        let n_out = ms[0].n_out();
        let mut line = format!("  {name:<8} {:>6.1} MB", ms[0].bytes() as f64 / 1e6);
        let ts: Vec<usize> = match std::env::var("T") {
            Ok(v) => v.split(',').filter_map(|x| x.parse().ok()).collect(),
            Err(_) => vec![1, 2, 4],
        };
        for t in ts {
            let x: Vec<f32> = (0..t * n_in).map(|_| (rnd() % 2001) as f32 / 1000.0 - 1.0).collect();
            let mut y = vec![0f32; t * n_out];
            let mut best = 0f64;
            // REPEAT=n: n passes per timing (for cache-resident runs with COPIES=1).
            let repeat: usize = std::env::var("REPEAT").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
            for _ in 0..if repeat > 1 { 20 } else { 5 } {
                let t0 = Instant::now();
                for _ in 0..repeat {
                    for m in ms {
                        kernels::matmul_quant_multi(&mut [(m, &mut y[..])], &x, t);
                    }
                }
                best = best.max((ms[0].bytes() * ms.len() * repeat) as f64 / t0.elapsed().as_secs_f64() / 1e9);
            }
            std::hint::black_box(&y);
            line += &format!(" | t={t} {best:>6.1} GB/s {:>6.2} ms", ms[0].bytes() as f64 / best / 1e6);
        }
        println!("{line}");
    }
    let copies = |bytes: usize| match std::env::var("COPIES").ok().and_then(|v| v.parse().ok()) {
        Some(n) => n,
        None => (1_500_000_000 / bytes).max(2),
    };
    println!("decode matmul [{n_out} x {n_in}], {threads} threads (GB/s of weight bytes, ms per matrix):");
    {
        let wb: Vec<u16> = (0..n_out * n_in)
            .map(|_| (0x3c00 + rnd() % 0x800) as u16 ^ ((rnd() & 1) << 15) as u16)
            .collect();
        let q4 = quant::quantize_q4(&wb, n_out, n_in);
        let n = copies(q4.bytes());
        let q4s: Vec<_> = (0..n).map(|_| quant::quantize_q4(&wb, n_out, n_in)).collect();
        run("Q4", &q4s, n_in, &mut rnd);
        drop(q4s);
        let q8 = quant::quantize_bf16(&wb, n_out, n_in);
        let q8s: Vec<_> = (0..copies(q8.bytes())).map(|_| quant::quantize_bf16(&wb, n_out, n_in)).collect();
        run("Q8", &q8s, n_in, &mut rnd);
    }
    let kinds: Vec<GgmlType> = match std::env::var("KINDS") {
        Ok(v) => v
            .split(',')
            .filter_map(|k| GgmlType::ALL.into_iter().find(|t| t.name() == k))
            .collect(),
        Err(_) => GgmlType::ALL.into_iter().filter(|k| !k.is_float()).collect(),
    };
    for kind in kinds {
        if !n_in.is_multiple_of(kind.block_size()) {
            continue;
        }
        let bytes = n_out * kind.row_bytes(n_in);
        let raw: Vec<u8> = (0..bytes).map(|_| rnd() as u8).collect();
        let ms: Vec<_> = (0..copies(bytes))
            .map(|_| GgufMatrix::from_bytes(kind, &raw, n_out, n_in).unwrap())
            .collect();
        run(kind.name(), &ms, n_in, &mut rnd);
    }
}

fn main() {
    if std::env::args().nth(1).as_deref() == Some("gguf") {
        let arg = |i: usize, d: usize| std::env::args().nth(i).and_then(|s| s.parse().ok()).unwrap_or(d);
        bench_gguf(arg(2, 17408), arg(3, 5120), arg(4, 12));
        return;
    }
    if std::env::args().nth(1).as_deref() == Some("gpu") {
        #[cfg(feature = "gpu")]
        {
            let n_out: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(10240);
            let n_in: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(2560);
            bench_gpu(n_out, n_in);
        }
        #[cfg(not(feature = "gpu"))]
        println!("built without the gpu feature");
        return;
    }
    let gb: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(4);
    let n = gb << 27; // u64 count
    println!("filling {gb} GB buffer...");
    let buf: Vec<u64> = (0..n as u64).map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15)).collect();
    let threads = [1usize, 2, 4, 6, 8, 12, 18];
    println!("\nplain read bandwidth (GB/s):");
    for &th in &threads {
        println!("  {th:>2} threads: {:>6.1}", bench_read(&buf, th));
    }
    // Reinterpret the buffer as a bf16 weight matrix [n_out, n_in].
    let n_in = 4096;
    let w: &[u16] = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u16, buf.len() * 4) };
    let n_out = w.len() / n_in;
    let n_out = n_out.min(262_144);
    let w = &w[..n_out * n_in];
    println!("\nbf16 matmul [{n_out} x {n_in}], t=1 (GB/s of weights):");
    for &th in &threads {
        println!("  {th:>2} threads: {:>6.1}", bench_matmul(w, n_out, n_in, 1, th));
    }
    println!("\nbf16 matmul, t=4:");
    for &th in &[6usize, 12, 18] {
        println!("  {th:>2} threads: {:>6.1}", bench_matmul(w, n_out, n_in, 4, th));
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::env::var("VARIANTS").is_err() {
            println!("(set VARIANTS=1 for single-thread kernel variants)");
        } else {
            println!("\nsingle-thread kernel variants, t=1 (GB/s):");
            let rows = n_out;
            {
                pool::init(1);
                let x = vec![0.5f32; n_in];
                let mut y = vec![0f32; n_out];
                let mut best = 0f64;
                for _ in 0..3 {
                    let ti = Instant::now();
                    kernels::matmul(w, n_out, n_in, &x, 1, &mut y);
                    best = best.max((w.len() * 2) as f64 / ti.elapsed().as_secs_f64() / 1e9);
                }
                println!("  {:<22} {:>6.1}", "matmul() 1 thread", best);
            }
            let x = vec![0.5f32; n_in];
            macro_rules! var {
                ($name:expr, $R:expr, $TC:expr, $G:expr, $A:expr) => {{
                    let mut best = 0f64;
                    let mut sink = 0f32;
                    for _ in 0..3 {
                        let ti = Instant::now();
                        let mut r = 0;
                        while r + $R <= rows {
                            let o = unsafe {
                                kernels::simd::kernel::<u16, $R, $TC, $G, $A>(w.as_ptr().add(r * n_in), n_in, x.as_ptr(), n_in, n_in)
                            };
                            for rr in 0..$R {
                                for c in 0..$TC {
                                    sink += o[rr][c];
                                }
                            }
                            r += $R;
                        }
                        best = best.max((rows * n_in * 2) as f64 / ti.elapsed().as_secs_f64() / 1e9);
                    }
                    std::hint::black_box(sink);
                    println!("  {:<22} {:>6.1}", $name, best);
                }};
            }
            var!("R4 G1 A2 (current)", 4, 1, 1, 2);
            var!("R4 G1 A1", 4, 1, 1, 1);
            var!("R4 G2 A4", 4, 1, 2, 4);
            var!("R2 G2 A4", 2, 1, 2, 4);
            var!("R2 G4 A8", 2, 1, 4, 8);
            var!("R1 G4 A8", 1, 1, 4, 8);
            var!("R8 G1 A1", 8, 1, 1, 1);
            var!("R8 G1 A2", 8, 1, 1, 2);
            var!("R8 G2 A2", 8, 1, 2, 2);
            var!("R12 G1 A1", 12, 1, 1, 1);
            var!("R16 G1 A1", 16, 1, 1, 1);
            var!("R6 G1 A2", 6, 1, 1, 2);
            var!("R8 TC2 G1 A1", 8, 2, 1, 1);
            var!("R4 TC2 G1 A2 (cur)", 4, 2, 1, 2);
            var!("R4 TC4 G1 A1 (cur)", 4, 4, 1, 1);
            var!("R8 TC4 G1 A1", 8, 4, 1, 1);
            var!("R1 G2 A4", 1, 1, 2, 4);
            var!("R1 G4 A8", 1, 1, 4, 8);
            var!("R1 G8 A16", 1, 1, 8, 16);
            var!("R2 G2 A4", 2, 1, 2, 4);
            var!("R2 G4 A8", 2, 1, 4, 8);
            // AND/SHL conversion variant, single row, A accumulators.
            {
                use std::arch::aarch64::*;
                let half = n_in / 2;
                let xe: Vec<f32> = (0..half).map(|j| x[2 * j]).collect();
                let xo: Vec<f32> = (0..half).map(|j| x[2 * j + 1]).collect();
                macro_rules! andshl {
                    ($name:expr, $A:expr) => {{
                        let mut best = 0f64;
                        let mut sink = 0f32;
                        for _ in 0..3 {
                            let ti = Instant::now();
                            for r in 0..rows {
                                unsafe {
                                    let wp = w.as_ptr().add(r * n_in) as *const u32;
                                    let mask = vdupq_n_u32(0xFFFF_0000);
                                    let mut acc = [vdupq_n_f32(0.0); $A];
                                    let mut i = 0;
                                    while i < n_in {
                                        for g in 0..$A / 2 {
                                            let w32 = vld1q_u32(wp.add((i + 8 * g) / 2));
                                            let hi = vreinterpretq_f32_u32(vandq_u32(w32, mask));
                                            let lo = vreinterpretq_f32_u32(vshlq_n_u32::<16>(w32));
                                            acc[2 * g] = vfmaq_f32(acc[2 * g], lo, vld1q_f32(xe.as_ptr().add((i + 8 * g) / 2)));
                                            acc[2 * g + 1] = vfmaq_f32(acc[2 * g + 1], hi, vld1q_f32(xo.as_ptr().add((i + 8 * g) / 2)));
                                        }
                                        i += 4 * $A;
                                    }
                                    let mut sum = acc[0];
                                    for a in 1..$A {
                                        sum = vaddq_f32(sum, acc[a]);
                                    }
                                    sink += vaddvq_f32(sum);
                                }
                            }
                            best = best.max((rows * n_in * 2) as f64 / ti.elapsed().as_secs_f64() / 1e9);
                        }
                        std::hint::black_box(sink);
                        println!("  {:<22} {:>6.1}", $name, best);
                    }};
                }
                andshl!("ANDSHL R1 A4", 4);
                andshl!("ANDSHL R1 A8", 8);
                andshl!("ANDSHL R1 A16", 16);
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // Prefill GEMM kernel compute ceiling: single thread, operands resident in L1 / L2.
        for (name, rows, k, toks) in [
            ("L1: 8 rows x 512 k x 8 tok", 8usize, 512usize, 8usize),
            ("L2: 8 rows x 4096 k x 64 tok", 8, 4096, 64),
            ("L2: 64 rows x 2560 k x 64 tok", 64, 2560, 64),
        ] {
            let wq: Vec<u16> = (0..rows * k).map(|i| 0x3f80 ^ (i as u16 & 0x3f)).collect();
            let xt: Vec<f32> = vec![0.5f32; toks * k]; // packed [tile][k][8]
            let mut y = vec![0f32; toks * rows];
            let flops_per_pass = 2.0 * rows as f64 * k as f64 * toks as f64;
            let passes = (2e10 / flops_per_pass).max(1.0) as usize;
            let t0 = Instant::now();
            for _ in 0..passes {
                let mut r = 0;
                while r + 8 <= rows {
                    for tile in 0..toks / 8 {
                        unsafe {
                            kernels::simd::gemm_8x8::<u16>(
                                wq.as_ptr().add(r * k),
                                k,
                                xt.as_ptr().add(tile * k * 8),
                                y.as_mut_ptr(),
                                rows,
                                r,
                                tile * 8,
                                8,
                            );
                        }
                    }
                    r += 8;
                }
            }
            std::hint::black_box(&y);
            println!(
                "gemm_8x8 1 thread {name}: {:.0} GFLOP/s",
                flops_per_pass * passes as f64 / t0.elapsed().as_secs_f64() / 1e9
            );
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // Quantized decode kernels, single thread, over ~1 GB of weights.
        let n_in = 4096usize;
        let rows = 131072usize; // q8: 512 MB; q4: 256 MB
        let wb: Vec<u16> = (0..rows * n_in).map(|i| 0x3f80 ^ (i as u16 & 0x7f)).collect();
        pool::init(1);
        let q8 = quant::quantize_bf16(&wb, rows, n_in);
        let q4 = quant::quantize_q4(&wb, rows, n_in);
        let x: Vec<f32> = (0..4 * n_in).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
        let xa = quant::quantize_act(&x, 4, n_in);
        let bpr = n_in / 32;
        println!("\nquantized decode kernels, 1 thread (GB/s of weight bytes | weight-equivalent bf16 GB/s):");
        macro_rules! bench {
            ($name:expr, $bytes:expr, $body:expr) => {{
                let mut best = 0f64;
                let mut sink = 0f32;
                for _ in 0..3 {
                    let t0 = Instant::now();
                    sink += $body;
                    best = best.max($bytes as f64 / t0.elapsed().as_secs_f64() / 1e9);
                }
                std::hint::black_box(sink);
                println!(
                    "  {:<24} {:>6.1} | {:>6.1}",
                    $name,
                    best,
                    best * (rows * n_in * 2) as f64 / $bytes as f64
                );
            }};
        }
        unsafe {
            bench!("q8 tc=4 (R2)", q8.bytes(), {
                let mut s = 0f32;
                let mut r = 0;
                while r + 2 <= rows {
                    let o = kernels::simd::q8_dot_rows_tc::<2, 4>(
                        q8.q.as_ptr().add(r * n_in),
                        q8.scales.as_ptr().add(r * bpr),
                        n_in,
                        xa.q.as_ptr(),
                        xa.scales.as_ptr(),
                    );
                    s += o[0][0] + o[0][1] + o[0][2] + o[0][3] + o[1][0] + o[1][1] + o[1][2] + o[1][3];
                    r += 2;
                }
                s
            });
            bench!("q4 tc=4 (R2)", q4.bytes(), {
                let mut s = 0f32;
                let mut r = 0;
                while r + 2 <= rows {
                    let o = kernels::simd::q4_dot_rows_tc::<2, 4>(
                        q4.q.as_ptr().add(r * n_in / 2),
                        q4.scales.as_ptr().add(r * bpr),
                        n_in,
                        xa.q.as_ptr(),
                        xa.scales.as_ptr(),
                        xa.sums.as_ptr(),
                    );
                    s += o[0][0] + o[0][1] + o[0][2] + o[0][3] + o[1][0] + o[1][1] + o[1][2] + o[1][3];
                    r += 2;
                }
                s
            });
            bench!("q8 t=1 (R2)", q8.bytes(), {
                let mut s = 0f32;
                let mut r = 0;
                while r + 2 <= rows {
                    let o = kernels::simd::q8_dot_rows::<2>(
                        q8.q.as_ptr().add(r * n_in),
                        q8.scales.as_ptr().add(r * bpr),
                        n_in,
                        xa.q.as_ptr(),
                        xa.scales.as_ptr(),
                    );
                    s += o[0] + o[1];
                    r += 2;
                }
                s
            });
            for tc in [2usize, 4] {
                let name = format!("q8 tc={tc} (R1)");
                bench!(name, q8.bytes(), {
                    let mut s = 0f32;
                    for r in 0..rows {
                        let o: [f32; 4] = if tc == 4 {
                            kernels::simd::q8_dot_rows_tc::<1, 4>(
                                q8.q.as_ptr().add(r * n_in),
                                q8.scales.as_ptr().add(r * bpr),
                                n_in,
                                xa.q.as_ptr(),
                                xa.scales.as_ptr(),
                            )[0]
                        } else {
                            let o2 = kernels::simd::q8_dot_rows_tc::<1, 2>(
                                q8.q.as_ptr().add(r * n_in),
                                q8.scales.as_ptr().add(r * bpr),
                                n_in,
                                xa.q.as_ptr(),
                                xa.scales.as_ptr(),
                            )[0];
                            [o2[0], o2[1], 0.0, 0.0]
                        };
                        s += o[0] + o[1] + o[2] + o[3];
                    }
                    s
                });
            }
            bench!("q4 t=1 (R2)", q4.bytes(), {
                let mut s = 0f32;
                let mut r = 0;
                while r + 2 <= rows {
                    let o = kernels::simd::q4_dot_rows::<2>(
                        q4.q.as_ptr().add(r * n_in / 2),
                        q4.scales.as_ptr().add(r * bpr),
                        n_in,
                        xa.q.as_ptr(),
                        xa.scales.as_ptr(),
                        xa.sums.as_ptr(),
                    );
                    s += o[0] + o[1];
                    r += 2;
                }
                s
            });
            for tc in [2usize, 4] {
                let name = format!("q4 tc={tc} (R1)");
                bench!(name, q4.bytes(), {
                    let mut s = 0f32;
                    for r in 0..rows {
                        let o: [f32; 4] = if tc == 4 {
                            kernels::simd::q4_dot_rows_tc::<1, 4>(
                                q4.q.as_ptr().add(r * n_in / 2),
                                q4.scales.as_ptr().add(r * bpr),
                                n_in,
                                xa.q.as_ptr(),
                                xa.scales.as_ptr(),
                                xa.sums.as_ptr(),
                            )[0]
                        } else {
                            let o2 = kernels::simd::q4_dot_rows_tc::<1, 2>(
                                q4.q.as_ptr().add(r * n_in / 2),
                                q4.scales.as_ptr().add(r * bpr),
                                n_in,
                                xa.q.as_ptr(),
                                xa.scales.as_ptr(),
                                xa.sums.as_ptr(),
                            )[0];
                            [o2[0], o2[1], 0.0, 0.0]
                        };
                        s += o[0] + o[1] + o[2] + o[3];
                    }
                    s
                });
            }
        }
    }
    println!("\nbf16 matmul, t=32 (f32 conversion path), GFLOP/s:");
    for &th in &[6usize, 12, 18] {
        let gbs = bench_matmul(w, n_out, n_in, 32, th);
        println!("  {th:>2} threads: {:>6.1} GB/s = {:>7.0} GFLOP/s", gbs, gbs / 2.0 * 2.0 * 32.0);
    }
}
