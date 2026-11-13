//! Compares oxinfer's Haru (`oxinfer::haru`) with the outputs `tools/ref_haru.py` wrote from
//! the PyTorch reference:
//!
//! ```text
//! haru_check MODEL_DIR REF_DIR
//! ```
//!
//! Stage by stage: the trimming of the voice reference, the codec's latent means of the
//! reference (a whole clip) and of the user's audio (frame by frame), the codec's waveform for
//! the reference's latents, the prompt's tokens, then the conversation itself with the
//! reference's noise replayed and a greedy text stream: per frame the text token, the language
//! model's hidden state, the audio latent and the audio. Exits with an error when prompt tokens
//! or frame tokens differ.

use anyhow::{Context, Result, bail};
use oxinfer::haru::{DuplexConfig, DuplexSession, Haru, HaruOptions, Replay, prepare_reference};
use std::path::Path;
use std::time::Instant;

fn f32s(p: &Path) -> Result<Vec<f32>> {
    let b = std::fs::read(p).with_context(|| format!("read {}", p.display()))?;
    Ok(b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
}

fn ids(v: &serde_json::Value) -> Vec<u32> {
    v.as_array()
        .map(|a| a.iter().filter_map(|x| x.as_u64().map(|x| x as u32)).collect())
        .unwrap_or_default()
}

/// The largest difference and the signal-to-noise ratio in dB of `a` against `b`.
fn compare(a: &[f32], b: &[f32]) -> (f32, f64) {
    let max = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let sig: f64 = b.iter().map(|&y| (y as f64) * (y as f64)).sum();
    let noise: f64 = a.iter().zip(b).map(|(&x, &y)| ((x - y) as f64).powi(2)).sum();
    (max, 10.0 * (sig / noise.max(1e-300)).log10())
}

fn peak(a: &[f32]) -> f32 {
    a.iter().fold(0f32, |m, v| m.max(v.abs()))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        bail!("usage: haru_check MODEL_DIR REF_DIR");
    }
    let dir = Path::new(&args[1]);
    let man: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("manifest.json"))?)?;
    let model = Haru::load(&args[0], &HaruOptions::default())?;
    let codec = model.codec();
    let (d, fs) = (codec.latent_dim, codec.frame_samples);
    let h = model.info().hidden_size;
    let mut failed = false;

    // 1. The voice reference.
    let reference = if dir.join("reference.f32").exists() {
        let want = f32s(&dir.join("reference.f32"))?;
        let got = prepare_reference(&f32s(&dir.join("reference_raw.f32"))?, 180.0, model.info().sample_rate);
        println!(
            "reference: {} samples ({:.2} s) after trimming, {}",
            want.len(),
            want.len() as f64 / model.info().sample_rate as f64,
            if got == want { "identical" } else { "DIFFERENT" }
        );
        failed |= got != want;
        let t0 = Instant::now();
        let mean = codec.encode(&want);
        let secs = t0.elapsed().as_secs_f64();
        let want_mean = f32s(&dir.join("ref_mean.f32"))?;
        if mean.len() != want_mean.len() {
            println!(
                "encoder: {} latents, the reference has {}: DIFFERENT",
                mean.len() / d,
                want_mean.len() / d
            );
            failed = true;
        } else {
            let (max, snr) = compare(&mean, &want_mean);
            println!(
                "encoder (whole clip): {} latents in {secs:.2} s; largest difference {max:.3e} (peak {:.2}), SNR {snr:.1} dB",
                mean.len() / d,
                peak(&want_mean)
            );
        }
        Some(want)
    } else {
        None
    };

    // 2. The user's frames through the streaming encoder.
    let user = f32s(&dir.join("user.f32"))?;
    let count = user.len() / fs;
    let want_mean = f32s(&dir.join("user_mean.f32"))?;
    let mut state = codec.encoder_state();
    let mut mean = Vec::new();
    for f in user.chunks(fs).take(want_mean.len() / d) {
        mean.extend(codec.encode_stream(f, &mut state));
    }
    if !want_mean.is_empty() {
        let (max, snr) = compare(&mean, &want_mean);
        println!(
            "encoder (frame by frame): {} latents; largest difference {max:.3e} (peak {:.2}), SNR {snr:.1} dB",
            mean.len() / d,
            peak(&want_mean)
        );
    }

    // 3. The decoder on the reference's latents.
    let want_latents = f32s(&dir.join("latents.f32"))?;
    let want_audio = f32s(&dir.join("audio.f32"))?;
    let t0 = Instant::now();
    let mut state = codec.decoder_state();
    let mut audio = Vec::new();
    for l in want_latents.chunks(d) {
        audio.extend(codec.decode_stream(l, &mut state));
    }
    audio.iter_mut().for_each(|v| *v = v.clamp(-1.0, 1.0));
    let secs = t0.elapsed().as_secs_f64();
    let (max, snr) = compare(&audio, &want_audio);
    println!(
        "decoder (frame by frame, the reference's latents): {:.2} s of audio in {secs:.2} s; largest difference {max:.3e} (peak {:.2}), SNR {snr:.1} dB",
        audio.len() as f64 / model.info().sample_rate as f64,
        peak(&want_audio)
    );
    let whole: Vec<f32> = codec.decode(&want_latents).iter().map(|v| v.clamp(-1.0, 1.0)).collect();
    println!(
        "decoder (16 frames at a time) against frame by frame: {:.3e}",
        compare(&whole, &audio).0
    );

    // 4. The conversation, with the reference's noise.
    let mut config = DuplexConfig::default();
    config.temperature = 0.0;
    config.speak_first = man["speak_first"].as_bool().unwrap_or(false);
    config.max_frames = count + 1;
    let noise = Replay::new(f32s(&dir.join("noise.f32"))?);
    let system = man["system_prompt"].as_str().unwrap_or("");
    let mut session = DuplexSession::with_noise(&model, system, reference.as_deref(), config, Box::new(noise));
    let t0 = Instant::now();
    let mut frames = vec![session.start()?];
    let start_secs = t0.elapsed().as_secs_f64();
    let same = session.prompt_tokens == ids(&man["prompt_tokens"]);
    println!(
        "prompt: {} positions in {start_secs:.2} s (the reference took {:.2} s), tokens {}",
        session.prompt_length,
        man["start_secs"].as_f64().unwrap_or(0.0),
        if same { "identical" } else { "DIFFERENT" }
    );
    failed |= !same;
    let t0 = Instant::now();
    for i in 0..count - 1 {
        frames.push(session.step(&user[i * fs..(i + 1) * fs])?);
    }
    let step_secs = t0.elapsed().as_secs_f64();
    let want_tokens = ids(&man["tokens"]);
    let got_tokens: Vec<u32> = frames.iter().map(|f| f.token).collect();
    let first = got_tokens.iter().zip(&want_tokens).position(|(a, b)| a != b);
    println!(
        "frames: {count} in {step_secs:.2} s ({:.0} ms per frame; the reference took {:.2} s); tokens {}",
        1000.0 * step_secs / (count.max(2) - 1) as f64,
        man["step_secs"].as_f64().unwrap_or(0.0),
        match first {
            None if got_tokens.len() == want_tokens.len() => "identical".to_string(),
            None => "DIFFERENT in number".to_string(),
            Some(f) => format!("DIFFERENT from frame {f}"),
        }
    );
    failed |= first.is_some() || got_tokens.len() != want_tokens.len();
    let upto = first.unwrap_or(count).min(frames.len());
    let want_hidden = f32s(&dir.join("hidden.f32"))?;
    let (mut dh, mut dl, mut da) = (Vec::new(), Vec::new(), Vec::new());
    for (i, f) in frames.iter().enumerate().take(upto) {
        dh.push(compare(&f.hidden, &want_hidden[i * h..(i + 1) * h]).0);
        dl.push(compare(&f.latent, &want_latents[i * d..(i + 1) * d]).0);
        da.push(compare(&f.audio, &want_audio[i * fs..(i + 1) * fs]).0);
    }
    let max = |v: &[f32]| v.iter().cloned().fold(0f32, f32::max);
    let tenth = |v: &[f32]| max(&v[..v.len().min(10)]);
    println!(
        "over the {upto} frames with identical tokens: hidden state within {:.3e} (first 10 frames {:.3e}; peak {:.1}), latent within {:.3e} (first 10: {:.3e}; peak {:.2}), audio within {:.3e} (first 10: {:.3e})",
        max(&dh),
        tenth(&dh),
        peak(&want_hidden),
        max(&dl),
        tenth(&dl),
        peak(&want_latents),
        max(&da),
        tenth(&da)
    );
    let got_audio: Vec<f32> = frames.iter().take(upto).flat_map(|f| f.audio.iter().copied()).collect();
    println!(
        "the conversation's audio against the reference's: SNR {:.1} dB over {:.1} s",
        compare(&got_audio, &want_audio[..got_audio.len()]).1,
        got_audio.len() as f64 / model.info().sample_rate as f64
    );
    let text: String = frames.iter().map(|f| f.text.as_str()).collect();
    println!(
        "text: {:?}{}",
        text.trim(),
        if text == man["text"].as_str().unwrap_or("") {
            " (identical)"
        } else {
            " (DIFFERENT)"
        }
    );
    if failed {
        bail!("differences from the reference");
    }
    Ok(())
}
