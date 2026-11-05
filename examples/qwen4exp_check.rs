//! Compares a Qwen4-Exp (Qwen3.8-Flash-Next) checkpoint run by oxinfer with reference logits, as
//! written by a llama.cpp greedy run on the same GGUF: the prompt ids, then `[1 + n, vocab]` f32
//! rows (the prompt's last position, then after feeding each generated token) and the generated
//! ids:
//!
//! ```text
//! qwen4exp_check MODEL PROMPT_IDS REF.bin GEN_IDS [--chunk N] [--cpu]
//! ```
//!
//! The id files hold whitespace-separated token ids. Prints the largest logit difference, the
//! KL divergence and whether the top token agrees, for every step.

use anyhow::{Context, Result};
use oxinfer::qwen::QwenModel;
use oxinfer::safetensors::ModelFiles;
use std::path::Path;

/// Max |a - b|, KL(softmax(b) || softmax(a)), whether the argmaxes agree.
fn compare(a: &[f32], b: &[f32]) -> (f32, f64, bool) {
    let max = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let lse = |v: &[f32]| {
        let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        m + v.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln()
    };
    let (la, lb) = (lse(a), lse(b));
    let kl = b
        .iter()
        .zip(a)
        .map(|(&y, &x)| {
            let pb = (y as f64 - lb).exp();
            pb * ((y as f64 - lb) - (x as f64 - la))
        })
        .sum::<f64>();
    let am = |v: &[f32]| v.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0;
    (max, kl, am(a) == am(b))
}

fn ids(p: &str) -> Result<Vec<u32>> {
    let s = std::fs::read_to_string(p).with_context(|| p.to_string())?;
    Ok(s.split_whitespace().map(|v| v.parse::<u32>()).collect::<Result<_, _>>()?)
}

fn main() -> Result<()> {
    let mut args: Box<dyn Iterator<Item = String>> = Box::new(std::env::args().skip(1));
    let usage = "usage: qwen4exp_check MODEL PROMPT_IDS REF.bin GEN_IDS [--chunk N] [--cpu] | qwen4exp_check DIR [...]";
    let mut model = args.next().context(usage)?;
    // A folder holding model/, prompt.txt, ref.bin and gen.txt (tools/qwen4exp_tiny.py).
    let bundle = Path::new(&model).join("prompt.txt").exists();
    if bundle {
        let d = model.clone();
        let rest: Vec<String> = args.collect();
        let f = |n: &str| format!("{d}/{n}");
        model = f("model");
        args = Box::new([f("prompt.txt"), f("ref.bin"), f("gen.txt")].into_iter().chain(rest));
    }
    let prompt = ids(&args.next().context(usage)?)?;
    let refs: Vec<f32> = std::fs::read(args.next().context(usage)?)?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    let gen_ids = ids(&args.next().context(usage)?)?;
    let (mut chunk, mut cpu, mut mtp, mut mtp_from) = (usize::MAX, false, None, None);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--chunk" => chunk = args.next().context("--chunk N")?.parse()?,
            "--cpu" => cpu = true,
            // Greedy MTP speculative decoding of the reference's tokens (draft length K).
            "--mtp" => mtp = Some(args.next().context("--mtp K")?.parse::<usize>()?),
            "--mtp-from" => mtp_from = Some(args.next().context("--mtp-from FILE")?),
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
    let dir = Path::new(&dir);
    let cfg = oxinfer::config::QwenConfig::load(dir)?;
    let files = ModelFiles::open(dir, false)?;
    let t0 = std::time::Instant::now();
    let m = QwenModel::load(cfg, &files, oxinfer::model::Quant::Bf16)?;
    let vocab = m.cfg.vocab_size;
    let h = m.cfg.hidden_size;
    eprintln!("[loaded in {:.1}s]", t0.elapsed().as_secs_f64());
    if let Some(k) = mtp {
        use oxinfer::qwen_mtp::{MtpDecoder, MtpOptions};
        let side = match &mtp_from {
            Some(p) => Some(ModelFiles::open(&oxinfer::engine::resolve_model(p, None)?, false)?),
            None => None,
        };
        let head = m.load_mtp(side.as_ref().unwrap_or(&files))?;
        let opts = MtpOptions {
            draft_tokens: k,
            ..Default::default()
        };
        let mut dec = MtpDecoder::new(&m, &head, opts);
        let mut sampler = oxinfer::sampler::Sampler::new(0.0, 0, 1.0, 1);
        let t1 = std::time::Instant::now();
        let l = dec.prefill(&prompt, chunk.min(prompt.len()));
        let first = oxinfer::sampler::argmax(&l);
        dec.push(first);
        let mut out = vec![first];
        while out.len() < gen_ids.len() {
            out.extend(dec.step(&mut sampler, gen_ids.len() - out.len()));
        }
        out.truncate(gen_ids.len());
        let s = &dec.stats;
        println!(
            "MTP k={k}: {} tokens in {:.2}s, same as the reference: {}, acceptance {:.1}%, {:.2} tokens/round",
            out.len(),
            t1.elapsed().as_secs_f64(),
            out == gen_ids,
            100.0 * s.acceptance(),
            s.tokens_per_round()
        );
        if out != gen_ids {
            println!("ours {out:?}\nref  {gen_ids:?}");
        }
        return Ok(());
    }
    let mut sess = m.new_session();
    let t1 = std::time::Instant::now();
    let mut hs = Vec::new();
    for c in prompt.chunks(chunk.min(prompt.len())) {
        hs = m.forward_hidden_mm(&mut sess, c, &[]);
    }
    let l = m.logits(&hs[hs.len() - h..], 1);
    eprintln!("[prefill {} tokens in {:.2}s]", prompt.len(), t1.elapsed().as_secs_f64());
    let n = (refs.len() / vocab).min(gen_ids.len() + 1);
    let (mut worst, mut kls, mut same) = (0f32, 0f64, 0);
    let mut step = |i: usize, l: &[f32]| {
        let (mx, kl, s) = compare(l, &refs[i * vocab..(i + 1) * vocab]);
        let am = l.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0;
        println!(
            "step {i}: max diff {mx:.3e}, KL {kl:.3e}, top-1 {} (ours {am})",
            if s { "same" } else { "DIFFERENT" }
        );
        worst = worst.max(mx);
        kls += kl;
        same += s as usize;
    };
    step(0, &l);
    let t2 = std::time::Instant::now();
    for i in 1..n {
        let l = m.forward(&mut sess, &gen_ids[i - 1..i]);
        step(i, &l);
    }
    let dt = t2.elapsed().as_secs_f64();
    println!(
        "{n} steps: max logit diff {worst:.3e}, mean KL {:.3e}, same top-1 {same}/{n}; decode {:.2} tok/s",
        kls / n as f64,
        (n - 1) as f64 / dt.max(1e-9)
    );
    Ok(())
}
