//! Times main-model forward passes of several token counts (the speculative verification path:
//! a recording forward plus the LM head over every token), best of a few repeats:
//! `cargo run --release --example pass_bench -- MODEL [THREADS] [TOKENS,..] [REPEATS]`.
//! Also prints the weight types of a GGUF checkpoint.

use oxinfer::config::QwenConfig;
use oxinfer::model::Quant;
use oxinfer::qwen::QwenModel;
use oxinfer::safetensors::ModelFiles;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(args.next().expect("usage: pass_bench MODEL [THREADS] [TOKENS,..] [REPEATS]"));
    let threads: usize = args.next().map_or(Ok(3), |t| t.parse())?;
    let counts: Vec<usize> = args
        .next()
        .unwrap_or_else(|| "1,2,4,9".into())
        .split(',')
        .map(|s| s.parse())
        .collect::<Result<_, _>>()?;
    let repeats: usize = args.next().map_or(Ok(3), |t| t.parse())?;
    oxinfer::pool::init(threads);
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
    for &t in &counts {
        let toks: Vec<u32> = (0..t as u32).map(|i| 2000 + i * 11).collect();
        let mut best = f64::MAX;
        let mut all = Vec::new();
        let prof = std::env::var_os("PASS_PROFILE").is_some();
        for i in 0..repeats {
            if prof && i + 1 == repeats {
                oxinfer::prof::reset();
            }
            let mut s = sess.clone();
            let t0 = Instant::now();
            let hs = model.forward_hidden_rec(&mut s, &toks);
            let l = model.logits(&hs, t);
            std::hint::black_box(&l);
            let dt = t0.elapsed().as_secs_f64();
            best = best.min(dt);
            all.push(format!("{dt:.3}"));
        }
        if prof {
            oxinfer::prof::report(all.last().unwrap().parse::<f64>()?, t);
        }
        println!("threads {threads} tokens {t}: best {best:.3} s ({})", all.join(" "));
    }
    Ok(())
}
