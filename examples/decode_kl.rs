//! How far CPU decoding's int8 activations move the output distributions: a text is decoded one
//! token at a time (teacher-forced, the decode kernels) with the K-quants' activations per 32
//! and per 256 (`oxinfer::quant::set_act_q8k`), and the same text is run in passes of 8 tokens
//! on the exact f32 prompt GEMM (`prefill_gemm=exact`: float activations), the reference. At
//! the last token of every pass, the KL divergence of each run's next-token distribution from
//! the reference's (the whole vocabulary) and whether their argmax agrees; also the `fast`
//! prompt GEMM (int8 activations as llama.cpp rounds them) against the reference, for scale:
//! `cargo run --release --example decode_kl -- MODEL TEXT_FILE [TOKENS]` (a Qwen-family or
//! Gemma GGUF / checkpoint; TOKENS default 256).

use oxinfer::kernels::{PrefillGemm, set_prefill_gemm};
use oxinfer::model::Quant;
use oxinfer::safetensors::ModelFiles;
use std::path::PathBuf;

/// The next-token logits after each pass of `tokens` (in passes of `step`).
type Run<'a> = dyn FnMut(&[u32], usize) -> Vec<Vec<f32>> + 'a;

fn kl(p: &[f32], q: &[f32]) -> f64 {
    let lse = |v: &[f32]| {
        let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        m + v.iter().map(|&t| (t as f64 - m).exp()).sum::<f64>().ln()
    };
    let (lp, lq) = (lse(p), lse(q));
    p.iter()
        .zip(q)
        .map(|(&a, &b)| {
            let pa = a as f64 - lp;
            pa.exp() * (pa - (b as f64 - lq))
        })
        .sum()
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: decode_kl MODEL TEXT_FILE [TOKENS]";
    let path = PathBuf::from(args.next().expect(usage));
    let text = std::fs::read_to_string(args.next().expect(usage))?;
    let n: usize = args.next().map_or(Ok(256), |s| s.parse())?;
    oxinfer::pool::init(oxinfer::pool::default_threads());
    let tok = oxinfer::engine::load_tokenizer(&path)?;
    let ids = tok.encode(text.as_str(), true)?.get_ids().to_vec();
    let ids = &ids[..n.min(ids.len())];
    let files = ModelFiles::open(&path, true)?;
    let gemma = path.to_string_lossy().to_lowercase().contains("gemma");
    const STEP: usize = 8;
    // Runs `ids` in passes of `step` tokens on a new session: the logits after each pass.
    let mut run: Box<Run> = if gemma {
        let m = oxinfer::model::Model::load(oxinfer::config::TextConfig::load(&path)?, &files, Quant::Q8)?;
        Box::new(move |ids: &[u32], step: usize| {
            let mut s = m.new_session();
            ids.chunks(step).map(|c| m.forward(&mut s, c)).collect()
        })
    } else {
        let m = oxinfer::qwen::QwenModel::load(oxinfer::config::QwenConfig::load(&path)?, &files, Quant::Q8)?;
        Box::new(move |ids: &[u32], step: usize| {
            let mut s = m.new_session();
            ids.chunks(step).map(|c| m.forward(&mut s, c)).collect()
        })
    };
    let ids = &ids[..ids.len() / STEP * STEP];
    set_prefill_gemm(PrefillGemm::Exact);
    let reference = run(ids, STEP);
    set_prefill_gemm(PrefillGemm::Fast);
    let fast = run(ids, STEP);
    set_prefill_gemm(PrefillGemm::Auto);
    // One token at a time: the logits after every STEP-th.
    let mut decode = |q8k: bool| -> Vec<Vec<f32>> {
        oxinfer::quant::set_act_q8k(Some(q8k));
        run(ids, 1).into_iter().skip(STEP - 1).step_by(STEP).collect()
    };
    let (q32, q8k) = (decode(false), decode(true));
    let report = |name: &str, a: &[Vec<f32>], b: &[Vec<f32>]| {
        let k: Vec<f64> = a.iter().zip(b).map(|(p, q)| kl(p, q)).collect();
        let agree = a.iter().zip(b).filter(|(p, q)| argmax(p) == argmax(q)).count();
        let mut s = k.clone();
        s.sort_by(f64::total_cmp);
        println!(
            "{name:>24}: KL mean {:.2e} median {:.2e} max {:.2e}; argmax agrees {agree}/{}",
            k.iter().sum::<f64>() / k.len() as f64,
            s[s.len() / 2],
            s[s.len() - 1],
            k.len()
        );
    };
    println!("{} ({} positions of {} tokens)", path.display(), reference.len(), ids.len());
    report("decode per 32 vs f32", &reference, &q32);
    report("decode per 256 vs f32", &reference, &q8k);
    report("prompt fast vs f32", &reference, &fast);
    report("per 256 vs per 32", &q32, &q8k);
    Ok(())
}
