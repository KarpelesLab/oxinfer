//! Compares Nemotron 3.5 Transcribe run by oxinfer with reference outputs on one recording (as
//! written by `tools/ref_nemotron_asr.py` on the release, or `tools/nemotron_asr_tiny.py` on a
//! tiny random model): `pcm`, `features`, `valid`, `asr_states`, `speaker_logits`,
//! `audio_embeds`, `ids`, `logits` (the last rows of the prompt), `gen` and `decode_logits`.
//!
//! ```text
//! nemotron_asr_check MODEL REF.safetensors [--quant bf16|q8|q4] [--chunk N] [--audio-only] [--cpu] [--mmap]
//! ```
//!
//! Prints the largest difference of the features, of both encoders' outputs and of the soft
//! tokens (with the largest value compared, for scale), then for the language model the largest
//! logit difference, the mean KL divergence and the agreement of the top token over the prompt
//! rows and over the greedy decode, which feeds the reference's tokens: with the Mamba time step
//! as the config has it (no clamp), and with transformers' floor in the prompt (its multi-token
//! scan floors the time step, its cached steps do not).

use anyhow::{Context, Result};
use oxinfer::nemotron::{NemotronConfig, NemotronModel};
use oxinfer::nemotron_asr::{Features, Speech, SpeechConfig};
use oxinfer::safetensors::{ModelFiles, SafeTensors};
use std::path::Path;

fn i64s(st: &SafeTensors, name: &str) -> Result<Vec<u32>> {
    Ok(st
        .bytes(name)?
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| i64::from_le_bytes(*c) as u32)
        .collect())
}

fn f32s(st: &SafeTensors, name: &str) -> Result<Vec<f32>> {
    Ok(st.bytes(name)?.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
}

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

fn report(what: &str, ours: &[f32], refs: &[f32], vocab: usize) {
    let n = refs.len() / vocab;
    let (mut max, mut kl, mut same, mut worst) = (0f32, 0f64, 0, 0);
    for i in 0..n {
        let (m, k, s) = compare(&ours[i * vocab..(i + 1) * vocab], &refs[i * vocab..(i + 1) * vocab]);
        if m > max {
            (max, worst) = (m, i);
        }
        kl += k;
        same += s as usize;
    }
    println!(
        "{what}: {n} positions, max logit diff {max:.3e} (at {worst}), mean KL {:.3e}, same top-1 {same}/{n}",
        kl / n as f64
    );
}

/// The largest difference of two tensors and the largest reference value.
fn diff(what: &str, ours: &[f32], refs: &[f32]) {
    if ours.len() != refs.len() {
        println!("{what}: {} values, the reference has {}", ours.len(), refs.len());
        return;
    }
    let (mut max, mut at, mut big, mut sum) = (0f32, 0, 0f32, 0f64);
    for (i, (a, b)) in ours.iter().zip(refs).enumerate() {
        big = big.max(b.abs());
        sum += (a - b).abs() as f64;
        if (a - b).abs() > max {
            (max, at) = ((a - b).abs(), i);
        }
    }
    println!(
        "{what}: {} values, max diff {max:.3e} (at {at}), mean diff {:.3e}, largest value {big:.2}",
        ours.len(),
        sum / ours.len().max(1) as f64
    );
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: nemotron_asr_check MODEL REF.safetensors [--quant bf16|q8|q4] [--chunk N] [--audio-only] [--cpu] [--mmap]";
    let model = args.next().context(usage)?;
    let refp = args.next().context(usage)?;
    let (mut chunk, mut cpu, mut mmap, mut audio_only) = (None, false, false, false);
    let mut quant = oxinfer::model::Quant::Bf16;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--chunk" => chunk = Some(args.next().context("--chunk N")?.parse::<usize>()?),
            "--quant" => quant = oxinfer::model::Quant::parse(&args.next().context("--quant Q")?).context("--quant bf16|q8|q4")?,
            "--audio-only" => audio_only = true,
            "--cpu" => cpu = true,
            "--mmap" => mmap = true,
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
    let files = ModelFiles::open(&dir, !mmap)?;
    let st = SafeTensors::open(Path::new(&refp), true)?;
    let t0 = std::time::Instant::now();
    let speech = Speech::load(SpeechConfig::load(&dir)?, &files)?;
    eprintln!("[audio side loaded in {:.2}s]", t0.elapsed().as_secs_f64());

    let pcm = f32s(&st, "pcm")?;
    let t0 = std::time::Instant::now();
    let feats = speech.features(&pcm);
    let valid = i64s(&st, "valid")?[0] as usize;
    println!(
        "{:.2}s of audio: {} frames ({} valid; the reference has {valid}), features in {:.2}s",
        pcm.len() as f64 / speech.cfg.sample_rate as f64,
        feats.frames,
        feats.valid,
        t0.elapsed().as_secs_f64()
    );
    let rf = f32s(&st, "features")?;
    diff("features", &feats.data, &rf);
    // Both encoders on our features, and on the reference's (the encoders alone).
    for (what, f) in [
        ("", &feats),
        (
            " (on the reference's features)",
            &Features {
                data: rf.clone(),
                frames: rf.len() / speech.cfg.n_mels,
                valid,
            },
        ),
    ] {
        let t0 = std::time::Instant::now();
        let enc = speech.encode(f);
        eprintln!("[encoders: {} chunks, {:.2}s]", enc.chunks, t0.elapsed().as_secs_f64());
        diff(&format!("speech encoder states{what}"), &enc.asr_states, &f32s(&st, "asr_states")?);
        diff(&format!("speaker logits{what}"), &enc.speaker_logits, &f32s(&st, "speaker_logits")?);
        diff(&format!("audio soft tokens{what}"), &enc.embeds, &f32s(&st, "audio_embeds")?);
    }
    if audio_only {
        return Ok(());
    }

    let enc = speech.encode(&feats);
    let t0 = std::time::Instant::now();
    let m = NemotronModel::load(NemotronConfig::load(&dir)?, &files, quant)?;
    eprintln!(
        "[language model: {} blocks ({}), {:?}, loaded in {:.2}s]",
        m.cfg.num_hidden_layers(),
        m.cfg.describe(),
        m.quant,
        t0.elapsed().as_secs_f64()
    );
    let ids = i64s(&st, "ids")?;
    let (h, vocab) = (m.cfg.hidden_size, m.cfg.vocab_size);
    let at: Vec<usize> = (0..ids.len()).filter(|&i| ids[i] == speech.cfg.audio_token_id).collect();
    anyhow::ensure!(
        at.len() == enc.tokens,
        "{} audio tokens in the prompt, {} soft tokens",
        at.len(),
        enc.tokens
    );
    let overrides: Vec<(usize, Vec<f32>)> = at
        .iter()
        .enumerate()
        .map(|(i, &p)| (p, enc.embeds[i * h..(i + 1) * h].to_vec()))
        .collect();
    let chunk = chunk.unwrap_or(ids.len()).max(1);
    let refs = f32s(&st, "logits")?;
    let rows = refs.len() / vocab;
    // The time step: unclamped (the config's limit), or with transformers' floor of
    // `time_step_min`, which its multi-token scan (the prompt) applies and its cached steps do not.
    let floor = m.cfg.time_step_min as f32;
    let prefill = |dt_min: f32| {
        let mut sess = m.new_session();
        sess.dt_min = dt_min;
        let mut ours: Vec<f32> = Vec::new();
        let t0 = std::time::Instant::now();
        let mut start = 0;
        while start < ids.len() {
            let end = (start + chunk).min(ids.len());
            let local: Vec<(usize, &[f32])> = overrides
                .iter()
                .filter(|(p, _)| *p >= start && *p < end)
                .map(|(p, e)| (p - start, e.as_slice()))
                .collect();
            let hs = m.forward_hidden_mm(&mut sess, &ids[start..end], &local);
            // Only the rows compared.
            let keep = (end - start).min(rows.saturating_sub(ids.len() - end));
            ours.extend(m.logits(&hs[(end - start - keep) * h..], keep));
            start = end;
        }
        eprintln!(
            "[prefill {} tokens ({} audio) in chunks of {chunk}: {:.2}s]",
            ids.len(),
            enc.tokens,
            t0.elapsed().as_secs_f64()
        );
        (sess, ours)
    };
    let (plain, ours) = prefill(m.new_session().dt_min);
    report("prompt (no floor)", &ours, &refs, vocab);
    let (floored, with_floor) = prefill(floor);
    report("prompt (floor, as transformers)", &with_floor, &refs, vocab);

    if st.info("gen").is_some() && st.info("decode_logits").is_some() {
        let gen_ids = i64s(&st, "gen")?;
        let dref = f32s(&st, "decode_logits")?;
        let am = |v: &[f32]| v.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0 as u32;
        for (what, sess, last) in [
            ("no floor", plain, &ours),
            ("floor in the prompt only, as transformers", floored, &with_floor),
        ] {
            let mut s = sess;
            s.dt_min = 0.0;
            let mut got = Vec::new();
            let mut greedy = vec![am(&last[last.len() - vocab..])];
            let t0 = std::time::Instant::now();
            for &tk in &gen_ids {
                let lg = m.forward(&mut s, &[tk]);
                greedy.push(am(&lg));
                got.extend(lg);
            }
            let same = greedy.iter().zip(&gen_ids).filter(|(a, b)| a == b).count();
            report(&format!("decode ({what})"), &got, &dref, vocab);
            println!(
                "  greedy tokens equal to the reference's: {same}/{} ({:.2}s per token)",
                gen_ids.len(),
                t0.elapsed().as_secs_f64() / gen_ids.len().max(1) as f64
            );
            // Where they differ: the step, and the reference's margin between its two best
            // tokens there (step 0 is the prompt's last position).
            let margin = |row: &[f32]| {
                let mut v: Vec<f32> = row.to_vec();
                v.sort_by(|a, b| b.total_cmp(a));
                v[0] - v[1]
            };
            let differ: Vec<String> = (0..gen_ids.len())
                .filter(|&i| greedy[i] != gen_ids[i])
                .map(|i| {
                    let row = if i == 0 {
                        &refs[refs.len() - vocab..]
                    } else {
                        &dref[(i - 1) * vocab..i * vocab]
                    };
                    format!("{i} ({:.2})", margin(row))
                })
                .collect();
            if !differ.is_empty() {
                println!("  steps that differ (the reference's top-1 margin): {}", differ.join(", "));
            }
        }
    }
    Ok(())
}
