//! Times Whisper's encoder on the CPU (with the decoder's cross-attention keys and values, as
//! the transcription's encoder time counts them) over the first 30 s window of a clip, the
//! attention kernels alternated in one process (so the load of a shared machine hits them
//! alike):
//!
//! ```text
//! whisper_enc_bench MODEL AUDIO ROUNDS [KERNEL ...] [--threads N] [--profile] [--dump FILE]
//! ```
//!
//! KERNEL is passed to `oxinfer::whisper::model::set_encoder_attention` before each of its
//! runs (0: the default, 1: the blocked attention used before, 2: the NEON prompt-attention
//! kernel; default: 0 alone). Prints the best and median time of each, the largest difference
//! of its encoder output from the first one's, and with `--profile` the per-stage phases over
//! all runs. `--dump FILE` writes the first one's encoder output (f32).

use anyhow::Context;
use oxinfer::whisper::{Transcriber, TranscriberOptions, mel};
use std::path::Path;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: whisper_enc_bench MODEL AUDIO ROUNDS [KERNEL ...] [--threads N] [--profile] [--dump FILE]";
    let (mut kernels, mut profile, mut dump) = (Vec::new(), false, None);
    let mut lo = TranscriberOptions::default();
    lo.set("device", "cpu")?;
    let mut i = 3;
    while i < a.len() {
        match a[i].as_str() {
            "--threads" => {
                lo.set("threads", a.get(i + 1).context(usage)?)?;
                i += 1;
            }
            "--profile" => profile = true,
            "--dump" => {
                dump = Some(a.get(i + 1).context(usage)?.clone());
                i += 1;
            }
            v => kernels.push(v.parse::<u32>().context(usage)?),
        }
        i += 1;
    }
    if kernels.is_empty() {
        kernels.push(0);
    }
    let rounds: usize = a.get(2).context(usage)?.parse()?;
    oxinfer::hf::set_hub(oxinfer::hf::Hub::Auto);
    let asr = Transcriber::load(a.first().context(usage)?, &lo)?;
    let m = asr.model();
    let pcm = Transcriber::load_audio(Path::new(a.get(1).context(usage)?))?;
    let pcm = &pcm[..pcm.len().min(mel::CHUNK_SAMPLES)];
    let window = mel::log_mel(pcm, m.cfg.num_mel_bins).window(0);
    // Warm-up.
    let _ = m.encode(&window);
    oxinfer::prof::reset();
    let mut times = vec![Vec::new(); kernels.len()];
    let mut outs: Vec<Option<Vec<f32>>> = vec![None; kernels.len()];
    let t_all = Instant::now();
    for _ in 0..rounds {
        for (k, &v) in kernels.iter().enumerate() {
            oxinfer::whisper::model::set_encoder_attention(v);
            let t0 = Instant::now();
            let y = m.encode(&window);
            drop(m.cross_kv(&y));
            times[k].push(t0.elapsed().as_secs_f64());
            outs[k].get_or_insert(y);
        }
    }
    let wall = t_all.elapsed().as_secs_f64();
    let base = outs[0].as_ref().unwrap();
    if let Some(f) = dump {
        std::fs::write(f, base.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
    }
    for (k, &v) in kernels.iter().enumerate() {
        let t = &mut times[k];
        t.sort_by(f64::total_cmp);
        let y = outs[k].as_ref().unwrap();
        let diff = base.iter().zip(y).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let same = base.iter().zip(y).all(|(a, b)| a.to_bits() == b.to_bits());
        println!(
            "kernel {v}: best {:.1} ms, median {:.1} ms, max diff from the first {diff:.3e}{}",
            t[0] * 1e3,
            t[t.len() / 2] * 1e3,
            if same { " (identical)" } else { "" }
        );
    }
    if profile {
        oxinfer::prof::report(wall, rounds * kernels.len());
    }
    Ok(())
}
