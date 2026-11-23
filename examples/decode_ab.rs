//! Same-process A/B timing of one-token CPU decode steps: greedy decoding in rounds of 128
//! tokens after a short prompt (llama-bench's tg128 contexts), the settings switched as it goes
//! so that a shared machine's load hits all of them alike, with each setting's median, mean and
//! best step and the parallel regions a step runs:
//! `cargo run --release --example decode_ab -- MODEL [THREADS] [STEPS] [SETTINGS]`
//! (a Qwen-family GGUF or checkpoint, e.g. Llama 3.2; THREADS: the pool's, default all; STEPS per
//! setting; SETTINGS: comma-separated `gG` / `tN` / `gGtN`, the gated MLP fused (`g1`, the
//! default) or not (`g0`) and the step's threads (`t0`: the default decode threads), default
//! `g0,g1`). Settings switch every token, or every 64 tokens when their threads differ (so that
//! the workers left out have parked). The settings are bit-identical: `AB_CHECK=1` first decodes
//! with each one alone and asserts that the tokens are the same.

use oxinfer::config::QwenConfig;
use oxinfer::model::Quant;
use oxinfer::qwen::QwenModel;
use oxinfer::safetensors::ModelFiles;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Clone, Copy, PartialEq)]
struct Setting {
    glu: bool,
    threads: usize,
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(args.next().expect("usage: decode_ab MODEL [THREADS] [STEPS] [SETTINGS]"));
    let threads: usize = args.next().map_or(Ok(oxinfer::pool::default_threads()), |s| s.parse())?;
    let steps: usize = args.next().map_or(Ok(256), |s| s.parse())?;
    let names: Vec<String> = args.next().unwrap_or_else(|| "g0,g1".into()).split(',').map(String::from).collect();
    let settings: Vec<Setting> = names
        .iter()
        .map(|s| {
            let mut st = Setting { glu: true, threads: 0 };
            let (g, t) = match s.split_once('t') {
                Some((g, t)) => (g, Some(t)),
                None => (s.as_str(), None),
            };
            if let Some(g) = g.strip_prefix('g') {
                st.glu = g != "0";
            }
            if let Some(t) = t {
                st.threads = t.parse().expect("tN: threads");
            }
            st
        })
        .collect();
    oxinfer::pool::init(threads);
    let files = ModelFiles::open(&path, false)?;
    files.preload();
    let model = QwenModel::load(QwenConfig::load(&path)?, &files, Quant::Q8)?;
    let prompt: Vec<u32> = (0..32).map(|i| 1000 + i * 37).collect();
    let block = if settings.iter().any(|s| s.threads != settings[0].threads) {
        64
    } else {
        1
    };
    let decode = |set: &[Setting], n: usize, times: &mut [Vec<f64>], regions: &mut [u64]| -> Vec<u32> {
        let mut sess = model.new_session();
        let mut tok = 0;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            if i % 128 == 0 {
                sess = model.new_session();
                tok = oxinfer::sampler::argmax(&model.forward(&mut sess, &prompt));
            }
            let m = (i / block) % set.len();
            oxinfer::kernels::set_glu_fused(set[m].glu);
            oxinfer::pool::set_decode_threads(set[m].threads);
            let r0 = oxinfer::pool::regions();
            let t0 = Instant::now();
            let _scope = oxinfer::pool::decode_scope();
            let l = model.forward(&mut sess, &[tok]);
            tok = oxinfer::sampler::argmax(&l);
            times[m].push(t0.elapsed().as_secs_f64() * 1e3);
            regions[m] += oxinfer::pool::regions() - r0;
            out.push(tok);
        }
        out
    };
    if std::env::var_os("AB_CHECK").is_some() {
        let mut first: Option<Vec<u32>> = None;
        for (s, name) in settings.iter().zip(&names) {
            let toks = decode(std::slice::from_ref(s), steps, &mut [Vec::new()], &mut [0]);
            match &first {
                None => first = Some(toks),
                Some(f) => assert_eq!(f, &toks, "setting {name} decodes other tokens"),
            }
        }
        println!("{} settings decode the same {steps} tokens", settings.len());
    }
    let mut times = vec![Vec::new(); settings.len()];
    let mut regions = vec![0u64; settings.len()];
    // A warm-up pass, then the measured one.
    decode(&settings, 16, &mut times.clone(), &mut regions.clone());
    decode(
        &settings,
        (steps * settings.len()).next_multiple_of(128 * block),
        &mut times,
        &mut regions,
    );
    for ((name, ts), r) in names.iter().zip(&mut times).zip(&regions) {
        ts.sort_by(f64::total_cmp);
        let med = ts[ts.len() / 2];
        let mean = ts.iter().sum::<f64>() / ts.len() as f64;
        println!(
            "{name:>8}: median {med:.3} ms ({:.1} tok/s), mean {mean:.3}, best tenth {:.3}, best {:.3}; {:.1} regions a step",
            1e3 / med,
            ts[ts.len() / 10],
            ts[0],
            *r as f64 / ts.len() as f64
        );
    }
    Ok(())
}
