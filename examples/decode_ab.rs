//! Same-process A/B timing of one-token CPU decode steps: greedy decoding in rounds of 128
//! tokens after a short prompt (llama-bench's tg128 contexts), the settings switched as it goes
//! so that a shared machine's load hits all of them alike, with each setting's median, mean and
//! best step and the parallel regions a step runs:
//! `cargo run --release --example decode_ab -- MODEL [THREADS] [STEPS] [SETTINGS]`
//! (a Qwen-family GGUF or checkpoint, e.g. Llama 3.2; THREADS: the pool's, default all; STEPS per
//! setting; SETTINGS: comma-separated combinations of `gG`, `tN` and `rR` such as `g1t8r0`, the
//! gated MLP fused (`g1`, the default) or not (`g0`), the step's threads (`t0`: the default
//! decode threads), and the weights interleaved for the CPU kernels (`r1`, the default where
//! the target has them) or in the file's layout (`r0`; both variants are loaded when the
//! settings ask for both), default `g0,g1`). The checkpoint is read into memory, as the engine
//! does by default. Settings switch every token, or every 64 tokens when their threads differ
//! (so that the workers left out have parked). The settings are bit-identical: `AB_CHECK=1`
//! first decodes with each one alone and asserts that the tokens are the same.

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
    repack: bool,
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
            let mut st = Setting {
                glu: true,
                threads: 0,
                repack: true,
            };
            // Letters, each followed by its number.
            let mut rest = s.as_str();
            while let Some(c) = rest.chars().next() {
                let n = rest[1..].find(|c: char| !c.is_ascii_digit()).map_or(rest.len(), |i| i + 1);
                let v: usize = rest[1..n].parse().expect("a number after each letter");
                match c {
                    'g' => st.glu = v != 0,
                    't' => st.threads = v,
                    'r' => st.repack = v != 0,
                    _ => panic!("unknown setting letter {c}"),
                }
                rest = &rest[n..];
            }
            st
        })
        .collect();
    oxinfer::pool::init(threads);
    // The models with the weights in the file's layout and interleaved (each loaded if a setting
    // asks for it), each from its own copy of the checkpoint in memory.
    let mut files = Vec::new();
    for on in [false, true] {
        files.push(
            settings
                .iter()
                .any(|s| s.repack == on)
                .then(|| ModelFiles::open(&path, true))
                .transpose()?,
        );
    }
    let mut models = Vec::new();
    for (on, f) in [false, true].into_iter().zip(&files) {
        // (The layout is chosen as the model loads.)
        oxinfer::ggml::set_repack(Some(on));
        models.push(match f {
            Some(f) => Some(QwenModel::load(QwenConfig::load(&path)?, f, Quant::Q8)?),
            None => None,
        });
    }
    let model = models.iter().flatten().next().unwrap();
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
            let model = models[set[m].repack as usize].as_ref().unwrap();
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
