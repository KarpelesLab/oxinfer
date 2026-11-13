//! Dumps what `tools/ref_fish_drama.py` compares with Fish Audio's PyTorch reference of Drama 3:
//!
//! ```text
//! fish_check prompt MODEL CASES.json OUT.json
//! fish_check gen MODEL OUT TEXT [--steps N] [--temp T [--seed S]] [--ref-text T --ref-codes CODES.npy]
//!                [--mmap] [--quant q8|q4] [--profile]
//! fish_check decode MODEL CODES.npy OUT.f32
//! fish_check encode MODEL CLIP OUT.npy
//! ```
//!
//! `prompt` writes the prompt rows of every case of CASES.json (`text`, `references` as
//! `[transcript, frames]`, `history` as `[text, frames]`, `seed`; the codes are a fixed function
//! of the seed). `gen` runs the greedy generation of TEXT (sampled with `--temp T [--seed S]`) and writes `OUT.json` (the text, the
//! prompt rows, the references and the generated frames), `OUT.logits` (f32: per step the logits
//! over the end token and the semantic band, then those of the nine codebook heads) and
//! `OUT.full` (f32: the backbone's logits over the whole vocabulary for the last prompt row).
//! `decode` writes the codec's output for the frames of a `.npy` (`[frames, 10]`), `encode` the
//! codes of an audio clip (`[frames, 10]`, int64). MODEL may be `codec.pth` alone for those two.

use anyhow::{Context, Result, bail};
use oxinfer::engine::Flow;
use oxinfer::fish::codec::Codec;
use oxinfer::fish::{Drama, DramaOptions, DramaSpeak};
use oxinfer::zonos::read_npy;
use std::path::Path;

/// The codes (`[frames, n_cb]`) of clip `k` of a case, as `ref_fish_drama.py prompt` derives them.
fn case_codes(seed: u64, k: u64, frames: u64, n_cb: u64) -> Vec<u32> {
    (0..frames)
        .flat_map(|f| (0..n_cb).map(move |c| ((seed * 7919 + k * 104729 + f * 31 + c * 17 + f * c) % 4096) as u32))
        .collect()
}

fn write_npy_i64(path: &str, shape: [usize; 2], v: &[u32]) -> Result<()> {
    let mut h = format!(
        "{{'descr': '<i8', 'fortran_order': False, 'shape': ({}, {}), }}",
        shape[0], shape[1]
    );
    while !(10 + h.len() + 1).is_multiple_of(64) {
        h.push(' ');
    }
    h.push('\n');
    let mut b = b"\x93NUMPY\x01\x00".to_vec();
    b.extend_from_slice(&(h.len() as u16).to_le_bytes());
    b.extend_from_slice(h.as_bytes());
    b.extend(v.iter().flat_map(|&x| (x as i64).to_le_bytes()));
    std::fs::write(path, b).with_context(|| format!("write {path}"))
}

fn codec(model: &str) -> Result<Codec> {
    let p = Path::new(model);
    Codec::load(&if p.is_dir() { p.join("codec.pth") } else { p.to_path_buf() })
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let mut lo = DramaOptions::default();
    lo.mmap = args.iter().any(|a| a == "--mmap");
    if let Some(q) = flag("--quant") {
        lo.quant = oxinfer::model::Quant::parse(&q).context("--quant bf16|q8|q4")?;
    }
    match args.first().map(String::as_str) {
        Some("prompt") if args.len() >= 4 => {
            let d = Drama::load(&args[1], &lo)?;
            let n_cb = d.info().n_codebooks as u64;
            let cases: Vec<serde_json::Value> = serde_json::from_slice(&std::fs::read(&args[2])?)?;
            let mut out = Vec::new();
            for case in &cases {
                let seed = case["seed"].as_u64().unwrap_or(0);
                let clips = |key: &str, k0: u64| -> Vec<(String, Vec<u32>)> {
                    case[key]
                        .as_array()
                        .map(|a| {
                            a.iter()
                                .enumerate()
                                .map(|(k, e)| {
                                    (
                                        e[0].as_str().unwrap_or_default().to_string(),
                                        case_codes(seed, k0 + k as u64, e[1].as_u64().unwrap_or(1), n_cb),
                                    )
                                })
                                .collect()
                        })
                        .unwrap_or_default()
                };
                let refs = clips("references", 0);
                let hist = clips("history", refs.len() as u64);
                out.push(d.prompt(case["text"].as_str().context("case without text")?, 0, &refs, &hist)?);
            }
            std::fs::write(&args[3], serde_json::to_vec(&out)?)?;
        }
        Some("gen") if args.len() >= 4 => {
            let d = Drama::load(&args[1], &lo)?;
            eprintln!("[loaded in {:.1}s]", d.info().load_secs);
            let mut o = DramaSpeak::default();
            // Greedy unless a temperature is given; then the reference is led along the frames
            // sampled here.
            o.temperature = flag("--temp").map(|t| t.parse()).transpose()?.unwrap_or(0.0);
            if let Some(s) = flag("--seed") {
                o.seed = s.parse()?;
            }
            let steps: usize = flag("--steps").map(|s| s.parse()).transpose()?.unwrap_or(32);
            let mut refs = Vec::new();
            if let (Some(t), Some(c)) = (flag("--ref-text"), flag("--ref-codes")) {
                refs.push((t, d.reference_codes(Path::new(&c))?));
            }
            let prompt = d.prompt(&args[3], 0, &refs, &[])?;
            let full = d.prompt_logits(&prompt);
            oxinfer::prof::reset();
            let g = d.generate(&prompt, &o, steps, true, &mut |_| Flow::Continue)?;
            if args.iter().any(|a| a == "--profile") {
                oxinfer::prof::report(
                    g.prefill_secs + g.generate_secs,
                    g.prompt_rows + g.rows.len() / (1 + d.info().n_codebooks),
                );
            }
            let w = 1 + d.info().n_codebooks;
            eprintln!(
                "[{} prompt rows in {:.2}s, {} frames in {:.2}s ({:.2} frames/s), {:?}]",
                g.prompt_rows,
                g.prefill_secs,
                g.rows.len() / w,
                g.generate_secs,
                (g.rows.len() / w) as f64 / g.generate_secs.max(1e-9),
                g.finish
            );
            let json = serde_json::json!({
                "text": args[3], "prompt": prompt, "references": refs, "frames": g.rows,
                "sampled": o.temperature > 0.0, "ended": g.finish == oxinfer::engine::StopReason::Stop,
            });
            std::fs::write(format!("{}.json", args[2]), serde_json::to_vec(&json)?)?;
            let bytes: Vec<u8> = g.logits.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(format!("{}.logits", args[2]), bytes)?;
            let bytes: Vec<u8> = full.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(format!("{}.full", args[2]), bytes)?;
        }
        Some("decode") if args.len() >= 4 => {
            let c = codec(&args[1])?;
            let (shape, v) = read_npy(Path::new(&args[2]))?;
            if shape.len() != 2 || shape[1] != c.n_codebooks() {
                bail!("{}: shape {shape:?}, expected [frames, {}]", args[2], c.n_codebooks());
            }
            let codes: Vec<u32> = v.iter().map(|&x| x as u32).collect();
            let t0 = std::time::Instant::now();
            let pcm = c.decode(&codes)?;
            eprintln!(
                "[{} frames ({:.2} s of audio) decoded in {:.2}s]",
                shape[0],
                pcm.len() as f64 / oxinfer::fish::codec::SAMPLE_RATE as f64,
                t0.elapsed().as_secs_f64()
            );
            let bytes: Vec<u8> = pcm.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(&args[3], bytes).with_context(|| format!("write {}", args[3]))?;
        }
        Some("encode") if args.len() >= 4 => {
            let c = codec(&args[1])?;
            let a = oxinfer::tts::read_wav(&std::fs::read(&args[2])?)?;
            if a.sample_rate != oxinfer::fish::codec::SAMPLE_RATE {
                bail!("{}: {} Hz, the codec takes 44100", args[2], a.sample_rate);
            }
            let t0 = std::time::Instant::now();
            let codes = c.encode(&a.samples)?;
            let n = c.n_codebooks();
            eprintln!(
                "[{:.2} s of audio to {} frames in {:.2}s]",
                a.duration_secs(),
                codes.len() / n,
                t0.elapsed().as_secs_f64()
            );
            write_npy_i64(&args[3], [codes.len() / n, n], &codes)?;
        }
        _ => bail!(
            "usage: fish_check prompt MODEL CASES.json OUT.json | fish_check gen MODEL OUT TEXT [options] | fish_check decode MODEL CODES.npy OUT.f32 | fish_check encode MODEL CLIP.wav OUT.npy"
        ),
    }
    Ok(())
}
