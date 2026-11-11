//! Checks a runtime LoRA on the routed experts of a MoE model against reference logits, on the
//! folder `tools/moe_lora_tiny.py` writes (`model/`, `adapter/`, `prompt.txt`, `base.bin` and
//! `ref.bin`: `[tokens, vocab]` f32 logits without and with the adapter):
//!
//! ```text
//! moe_lora_check DIR [--model SUB] [--adapter SUB] [--quant q8|q4] [--cpu] [--tol X]
//! ```
//!
//! `--model model_split` / `--adapter adapter_split` take the same weights / adapter with one
//! module per expert. The prompt runs as one prefill pass (grouped expert matmuls, or the
//! accelerator), in chunks of 5 tokens (short passes) and token by token (decode), with the
//! adapter on, then off; every position's logits are compared. With `--quant`, the batched
//! expert kernels must also match the one-at-a-time path bit for bit. Fails when a bf16 run
//! differs from the reference by more than `--tol` (default 1e-3).

use anyhow::{Context, Result, bail};
use oxinfer::generate::TextLm;
use oxinfer::lora::{Active, LoraAdapter};
use oxinfer::model::Quant;
use oxinfer::safetensors::ModelFiles;
use std::path::Path;

/// Max |a - b|, the largest KL(softmax(b) || softmax(a)) of a row, rows whose argmaxes agree.
fn compare(a: &[f32], b: &[f32], vocab: usize) -> (f32, f64, usize) {
    let max = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let lse = |v: &[f32]| {
        let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        m + v.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln()
    };
    let am = |v: &[f32]| v.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0;
    let (mut kl, mut same) = (0f64, 0);
    for (a, b) in a.chunks(vocab).zip(b.chunks(vocab)) {
        let (la, lb) = (lse(a), lse(b));
        let k: f64 = b
            .iter()
            .zip(a)
            .map(|(&y, &x)| (y as f64 - lb).exp() * ((y as f64 - lb) - (x as f64 - la)))
            .sum();
        kl = kl.max(k);
        same += (am(a) == am(b)) as usize;
    }
    (max, kl, same)
}

/// Logits of every position of `ids`, fed in chunks of `chunk` tokens.
fn run<M: TextLm>(m: &M, ids: &[u32], chunk: usize) -> Vec<f32> {
    let mut sess = m.new_session();
    let mut out = Vec::new();
    for c in ids.chunks(chunk) {
        let hs = m.forward_hidden_mm(&mut sess, c, &[]);
        out.extend(m.logits(&hs, c.len()));
    }
    out
}

fn check<M: TextLm>(
    m: &M,
    switch: &std::sync::Arc<std::sync::atomic::AtomicBool>,
    ids: &[u32],
    base: &[f32],
    refs: &[f32],
    exact: bool,
    tol: f32,
) -> Result<()> {
    let (t, vocab) = (ids.len(), base.len() / ids.len());
    let mut worst = 0f32;
    let mut report = |what: &str, l: &[f32], want: &[f32]| {
        let (mx, kl, same) = compare(l, want, vocab);
        println!("{what}: max |dlogit| {mx:.3e}, max KL {kl:.3e}, top-1 {same}/{t}");
        worst = worst.max(mx);
    };
    let off = run(m, ids, t);
    report("adapter off, one pass, vs base     ", &off, base);
    {
        let _on = Active::new(switch);
        for (what, chunk) in [("one pass       ", t), ("chunks of 5    ", 5), ("token by token ", 1)] {
            let l = run(m, ids, chunk);
            report(&format!("adapter on, {what}, vs peft"), &l, refs);
            if chunk < t {
                oxinfer::kernels::set_moe_batched(false);
                let one = run(m, ids, chunk);
                oxinfer::kernels::set_moe_batched(true);
                let same = l.iter().zip(&one).all(|(a, b)| a.to_bits() == b.to_bits());
                println!("    batched expert kernels == one expert at a time, bit for bit: {same}");
                if !same {
                    bail!("batched and one-at-a-time experts differ");
                }
            }
        }
    }
    let again = run(m, ids, t);
    let same = off.iter().zip(&again).all(|(a, b)| a.to_bits() == b.to_bits());
    println!("adapter off again == before, bit for bit: {same}");
    if !same {
        bail!("the adapter leaves a trace when switched off");
    }
    let (changed, _, _) = compare(base, refs, vocab);
    println!("(the adapter moves the reference logits by up to {changed:.3})");
    if exact && worst > tol {
        bail!("max difference {worst:.3e} exceeds {tol:.1e}");
    }
    Ok(())
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: moe_lora_check DIR [--model SUB] [--adapter SUB] [--quant q8|q4] [--cpu] [--tol X]";
    let dir = args.next().context(usage)?;
    let (mut model, mut adapter, mut quant, mut cpu, mut tol) = ("model".to_string(), "adapter".to_string(), Quant::Bf16, false, 1e-3f32);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model" => model = args.next().context(usage)?,
            "--adapter" => adapter = args.next().context(usage)?,
            "--quant" => quant = Quant::parse(&args.next().context(usage)?).context("--quant bf16|q8|q4")?,
            "--cpu" => cpu = true,
            "--tol" => tol = args.next().context(usage)?.parse()?,
            other => bail!("unknown argument {other}\n{usage}"),
        }
    }
    let dev = if cpu {
        oxinfer::platform::DevicePref::Cpu
    } else {
        oxinfer::platform::DevicePref::Auto
    };
    oxinfer::gpu::init(&oxinfer::platform::detect(), dev)?;
    let dir = Path::new(&dir);
    let f32s = |name: &str| -> Result<Vec<f32>> {
        let b = std::fs::read(dir.join(name)).with_context(|| format!("read {name}"))?;
        Ok(b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
    };
    let ids: Vec<u32> = std::fs::read_to_string(dir.join("prompt.txt"))?
        .split_whitespace()
        .map(|v| v.parse::<u32>())
        .collect::<Result<_, _>>()?;
    let (base, refs) = (f32s("base.bin")?, f32s("ref.bin")?);
    let mdir = dir.join(&model);
    let files = ModelFiles::open(&mdir, false)?;
    let mut ad = LoraAdapter::load(&dir.join(&adapter))?;
    let switch = ad.switch();
    let exact = quant == Quant::Bf16;
    let device = oxinfer::gpu::active().map_or("cpu".to_string(), |b| b.name().to_string());
    if oxinfer::config::model_type(&mdir)?.starts_with("gemma") {
        let mut m = oxinfer::model::Model::load(oxinfer::config::TextConfig::load(&mdir)?, &files, quant)?;
        let n = m.attach_lora(&mut ad)?;
        println!(
            "[{model} ({}), {adapter}: {n} layers patched, rank {}, on {device}]",
            quant.as_str(),
            ad.rank
        );
        check(&m, &switch, &ids, &base, &refs, exact, tol)
    } else {
        let mut m = oxinfer::qwen::QwenModel::load(oxinfer::config::QwenConfig::load(&mdir)?, &files, quant)?;
        let n = m.attach_lora(&mut ad)?;
        println!(
            "[{model} ({}), {adapter}: {n} layers patched, rank {}, on {device}]",
            quant.as_str(),
            ad.rank
        );
        check(&m, &switch, &ids, &base, &refs, exact, tol)
    }
}
