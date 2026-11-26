//! Prompt processing and generation speed of an Inkling-2 model, as `llama-bench` measures them:
//! `pp N` runs a prompt of N tokens (random ids) into a fresh session, `tg N` generates N tokens
//! after a 1-token prompt (greedy), each after a warm-up pass of the same size, the median of `-r`
//! runs.
//!
//! The prompt passes run as the library runs them outside a generation: on the CPU with the f32
//! GEMM. `OXINFER_PREFILL_GEMM=fast` gives the rounding a prompt has from the command line and
//! the engine (`prefill_gemm`'s default there: int8 activations).
//!
//! ```text
//! inkling2_bench MODEL [-p 512,2048] [-n 128] [-r 3] [--kv f32|f16|q8] [--cpu] [--threads N] [--chunk N] [-v 2,4,8] [-b 2,4,8] [--chain] [--exact 2,4,8]
//! ```
//!
//! `--chain`: `tg N` as greedy decoding with the passes chained on the device
//! (`Inkling2Model::decode_greedy`, as the CLI decodes greedily) rather than a host loop of
//! `forward` and argmax. `--exact N,..`: after prompts of 64 and of 200 random tokens, N tokens as
//! one verification pass against the same N one at a time: the largest logit difference and how
//! many logits differ (speculative decoding gives the plain greedy text when there are none).
//! `-b N,..`: batched decode steps (the server's concurrent requests: `forward_rows`) of N
//! sessions after prompts of 64 random tokens each, ms per step.

use anyhow::{Context, Result};
use oxinfer::inkling::{Inkling2Config, Inkling2Model, KvCache};
use oxinfer::safetensors::ModelFiles;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: inkling2_bench MODEL [-p 512,2048] [-n 128] [-r 3] [--kv f32|f16|q8] [--cpu] [--threads N] [--chunk N] [-v 2,4,8] [-b 2,4,8] [--chain] [--exact 2,4,8]";
    let model = args.next().context(usage)?;
    let mut verify: Vec<usize> = Vec::new();
    let mut rows: Vec<usize> = Vec::new();
    let mut exact: Vec<usize> = Vec::new();
    let mut chain = false;
    let (mut pp, mut tg, mut reps, mut kv, mut cpu, mut chunk) = (vec![512usize], 128usize, 3usize, KvCache::F32, false, 2048usize);
    while let Some(a) = args.next() {
        match a.as_str() {
            "-p" => {
                pp = args
                    .next()
                    .context("-p N,..")?
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.parse())
                    .collect::<Result<_, _>>()?
            }
            "-n" => tg = args.next().context("-n N")?.parse()?,
            "-r" => reps = args.next().context("-r N")?.parse::<usize>()?.max(1),
            "--kv" => kv = KvCache::parse(&args.next().context("--kv MODE")?)?,
            "--chunk" => chunk = args.next().context("--chunk N")?.parse::<usize>()?.max(1),
            "--cpu" => cpu = true,
            "--chain" => chain = true,
            "--threads" => oxinfer::pool::configure(args.next().context("--threads N")?.parse::<usize>()?.max(1)),
            "--exact" => {
                exact = args
                    .next()
                    .context("--exact N,..")?
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.parse())
                    .collect::<Result<_, _>>()?
            }
            "-b" => {
                rows = args
                    .next()
                    .context("-b N,..")?
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.parse())
                    .collect::<Result<_, _>>()?
            }
            "-v" => {
                verify = args
                    .next()
                    .context("-v N,..")?
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.parse())
                    .collect::<Result<_, _>>()?
            }
            other => anyhow::bail!("unknown argument {other}"),
        }
    }
    let dev = if cpu {
        oxinfer::platform::DevicePref::Cpu
    } else {
        oxinfer::platform::DevicePref::Auto
    };
    oxinfer::gpu::init(&oxinfer::platform::detect(), dev)?;
    let dir = oxinfer::engine::resolve_model(&model, None)?;
    let files = ModelFiles::open(&dir, true)?;
    let mut m = Inkling2Model::load(Inkling2Config::load(&dir)?, &files, oxinfer::model::Quant::Bf16)?;
    m.kv = kv;
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % 199_000) as u32 + 100
    };
    // (`INKLING2_BENCH_PROF`: the per-category time of the timed runs, as `--profile` prints it.)
    let prof = std::env::var_os("INKLING2_BENCH_PROF").is_some();
    for &n in &pp {
        let mut times = Vec::new();
        for rep in 0..=reps {
            let ids: Vec<u32> = (0..n).map(|_| rnd()).collect();
            let mut sess = m.new_session();
            if rep == 1 {
                oxinfer::prof::reset();
            }
            let t0 = std::time::Instant::now();
            for c in ids.chunks(chunk) {
                m.forward_hidden(&mut sess, c);
            }
            if rep > 0 {
                times.push(t0.elapsed().as_secs_f64());
            }
        }
        if prof {
            oxinfer::prof::report(times.iter().sum(), n * reps);
        }
        let t = median(times);
        println!("pp{n}: {:.1} tok/s ({:.3} s)", n as f64 / t, t);
    }
    if tg > 0 {
        let am = |v: &[f32]| v.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0 as u32;
        let mut times = Vec::new();
        // (`INKLING2_BENCH_SUM`: the greedy tokens of the last run and a digest of its logits'
        // bits, to see whether two builds or settings compute the same steps.)
        let mut digest = 0u64;
        let mut toks = Vec::new();
        for rep in 0..=reps {
            let mut sess = m.new_session();
            let mut tok = am(&m.forward(&mut sess, &[rnd()]));
            (digest, toks) = (0, Vec::new());
            if rep == 1 {
                oxinfer::prof::reset();
            }
            let t0 = std::time::Instant::now();
            let n = if rep == 0 { 8 } else { tg };
            if chain {
                m.decode_greedy(&mut sess, tok, n, &mut |t| {
                    toks.push(t);
                    oxinfer::generate::Chain::Go
                })
                .context("--chain: the device does not chain greedy passes")?;
            } else {
                for _ in 0..n {
                    let l = m.forward(&mut sess, &[tok]);
                    tok = am(&l);
                    toks.push(tok);
                    digest = l.iter().fold(digest, |d, v| d.rotate_left(7) ^ u64::from(v.to_bits()));
                }
            }
            if rep > 0 {
                times.push(t0.elapsed().as_secs_f64());
            }
        }
        if prof {
            oxinfer::prof::report(times.iter().sum(), tg * reps);
        }
        let t = median(times);
        println!("tg{tg}: {:.1} tok/s ({:.2} ms/token)", tg as f64 / t, t * 1e3 / tg as f64);
        if std::env::var_os("INKLING2_BENCH_SUM").is_some() {
            println!("tokens {toks:?} logits digest {digest:016x}");
        }
    }
    // Verification passes (speculative decoding): N tokens at once after a short prompt, rolled
    // back after each.
    for &n in &verify {
        let mut sess = m.new_session();
        let ids: Vec<u32> = (0..64).map(|_| rnd()).collect();
        m.forward(&mut sess, &ids);
        let base = sess.pos;
        let mut times = Vec::new();
        for rep in 0..=reps.max(5) {
            let toks: Vec<u32> = (0..n).map(|_| rnd()).collect();
            let t0 = std::time::Instant::now();
            oxinfer::dspark::Target::forward_verify(&m, &mut sess, &toks, &[], &mut Vec::new(), false);
            if rep > 0 {
                times.push(t0.elapsed().as_secs_f64());
            }
            sess.truncate(base)?;
        }
        let t = median(times);
        println!("verify{n}: {:.2} ms per pass", t * 1e3);
    }
    // Batched decode steps of n sessions.
    for &n in &rows {
        let mut sess: Vec<_> = (0..n)
            .map(|_| {
                let mut s = m.new_session();
                let ids: Vec<u32> = (0..64).map(|_| rnd()).collect();
                m.forward(&mut s, &ids);
                s
            })
            .collect();
        let mut times = Vec::new();
        for rep in 0..=reps.max(5) * 4 {
            let toks: Vec<u32> = (0..n).map(|_| rnd()).collect();
            let mut refs: Vec<&mut _> = sess.iter_mut().collect();
            let t0 = std::time::Instant::now();
            m.forward_rows(&mut refs, &toks).context("-b: no batched passes on this device")?;
            if rep > 0 {
                times.push(t0.elapsed().as_secs_f64());
            }
        }
        let t = median(times);
        println!("rows{n}: {:.2} ms per step ({:.1} tok/s)", t * 1e3, n as f64 / t);
    }
    // A verification pass against one-token passes.
    for &n in &exact {
        for plen in [64, 200] {
            let mut sess = m.new_session();
            let ids: Vec<u32> = (0..plen).map(|_| rnd()).collect();
            m.forward(&mut sess, &ids);
            let base = sess.pos;
            let toks: Vec<u32> = (0..n).map(|_| rnd()).collect();
            let mut one = Vec::new();
            for &t in &toks {
                one.extend(m.forward(&mut sess, &[t]));
            }
            sess.truncate(base)?;
            let all = oxinfer::dspark::Target::forward_verify(&m, &mut sess, &toks, &[], &mut Vec::new(), false);
            anyhow::ensure!(all.len() == one.len(), "verification logits: {} against {}", all.len(), one.len());
            let diff = all.iter().zip(&one).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            let nd = all.iter().zip(&one).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            println!(
                "exact{n} after {plen}: max logit diff {diff:.3e}, {nd} of {} logits differ",
                all.len()
            );
        }
    }
    Ok(())
}
