//! Times main-model forward passes of several token counts (the speculative verification path:
//! a recording forward plus the LM head over every token), best of a few repeats, for each
//! multi-token kernel choice in turn (interleaved pass by pass, so that a shared machine's load
//! hits them alike):
//! `cargo run --release --example pass_bench -- MODEL [THREADS] [TOKENS,..] [REPEATS] [MODES]`
//! (THREADS: `N`, `NeM` with M of them at background QoS, or `Nb` all at background QoS; MODES:
//! comma-separated `pairs`, `dot`, `auto`, default `pairs,auto`). Also prints the weight types
//! of a GGUF checkpoint; `PASS_PROFILE=1` adds a per-operation profile of the last pass.

use oxinfer::config::QwenConfig;
use oxinfer::kernels::{MultiTokenKernels, set_multi_token_kernels};
use oxinfer::model::Quant;
use oxinfer::qwen::QwenModel;
use oxinfer::safetensors::ModelFiles;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(
        args.next()
            .expect("usage: pass_bench MODEL [THREADS] [TOKENS,..] [REPEATS] [MODES]"),
    );
    let spec = args.next().unwrap_or_else(|| "3".into());
    let mut po = oxinfer::pool::PoolOptions::threads(3);
    if let Some(n) = spec.strip_suffix('b') {
        po.threads = n.parse()?;
        po.qos = oxinfer::pool::Qos::Background;
    } else if let Some((n, e)) = spec.split_once('e') {
        po.threads = n.parse()?;
        po.efficiency_threads = e.parse()?;
    } else {
        po.threads = spec.parse()?;
    }
    let counts: Vec<usize> = args
        .next()
        .unwrap_or_else(|| "1,2,4,9".into())
        .split(',')
        .map(|s| s.parse())
        .collect::<Result<_, _>>()?;
    let repeats: usize = args.next().map_or(Ok(3), |t| t.parse())?;
    let modes: Vec<(String, MultiTokenKernels)> = args
        .next()
        .unwrap_or_else(|| "pairs,auto".into())
        .split(',')
        .map(|m| {
            let k = match m {
                "pairs" => MultiTokenKernels::Pairs,
                "dot" => MultiTokenKernels::Dot,
                _ => MultiTokenKernels::Auto,
            };
            (m.to_string(), k)
        })
        .collect();
    oxinfer::pool::init_with(po);
    oxinfer::kernels::set_decode_max_tokens(15);
    if oxinfer::gguf::is_gguf(&path) {
        let h = oxinfer::gguf::read_header(&path)?;
        let mut types: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        for t in &h.tensors {
            let name = oxinfer::ggml::GgmlType::from_id(t.ggml_type).map_or(format!("{}", t.ggml_type), |k| k.name().to_string());
            let e = types.entry(name).or_default();
            e.0 += 1;
            e.1 += t.end - t.start;
        }
        for (k, (n, b)) in types {
            eprintln!("  {k:<8} {n:>4} tensors {:>7.2} GB", b as f64 / 1e9);
        }
    }
    let files = ModelFiles::open(&path, false)?;
    files.preload();
    let model = QwenModel::load(QwenConfig::load(&path)?, &files, Quant::Q4)?;
    let mut sess = model.new_session();
    let prompt: Vec<u32> = (0..64).map(|i| 1000 + i * 37).collect();
    model.forward(&mut sess, &prompt);
    let prof = std::env::var_os("PASS_PROFILE").is_some();
    for &t in &counts {
        let toks: Vec<u32> = (0..t as u32).map(|i| 2000 + i * 11).collect();
        // One-token passes use the same kernels whatever the mode.
        let ms = if t == 1 { &modes[..1] } else { &modes[..] };
        let mut times: Vec<Vec<f64>> = vec![Vec::new(); ms.len()];
        for i in 0..repeats {
            for (mi, (_, mode)) in ms.iter().enumerate() {
                set_multi_token_kernels(*mode);
                if prof && i + 1 == repeats {
                    oxinfer::prof::reset();
                }
                let mut s = sess.clone();
                let t0 = Instant::now();
                let hs = model.forward_hidden_rec(&mut s, &toks);
                let l = model.logits(&hs, t);
                std::hint::black_box(&l);
                times[mi].push(t0.elapsed().as_secs_f64());
                if prof && i + 1 == repeats {
                    eprintln!("[{} profile]", ms[mi].0);
                    oxinfer::prof::report(*times[mi].last().unwrap(), t);
                }
            }
        }
        let mut line = format!("threads {spec} tokens {t}:");
        for ((name, _), ts) in ms.iter().zip(&times) {
            let best = ts.iter().copied().fold(f64::MAX, f64::min);
            let all: Vec<String> = ts.iter().map(|v| format!("{v:.3}")).collect();
            line += &format!(" {name} {best:.3} s ({})", all.join(" "));
        }
        println!("{line}");
    }
    Ok(())
}
