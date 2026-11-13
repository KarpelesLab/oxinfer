//! `oxinfer --model higurashi-ai/Haru-4B --audio user.wav -o reply.wav`: a recording of the
//! user is played to Haru frame by frame, as in a live conversation, and what the model says
//! meanwhile is written as a WAV of the same length (24 kHz); its text goes to standard output.
//! The shape of the reference's `haru-inference`: `--system` is the character card, `--ref-audio`
//! the voice the model speaks with, `--guidance` the audio guidance scale, `--temp`, `--top-k`,
//! `--top-p` and `--seed` the sampling of the text stream, `--speak-first` (given as
//! `--speak-json '{"speak_first": true}'`) lets the model open the conversation.

use crate::Args;
use anyhow::{Context, Result, bail};
use oxinfer::haru::{DuplexConfig, DuplexSession, Haru, HaruOptions};
use oxinfer::tts::Audio;
use std::io::Write;
use std::time::Instant;

pub fn run(args: &Args) -> Result<()> {
    let [input] = args.audios.as_slice() else {
        bail!("Haru is a speech-to-speech model: --audio USER.wav (one recording of the user) -o REPLY.wav");
    };
    let mut lo = HaruOptions::default();
    lo.quant = args.quant;
    lo.mmap = args.mmap;
    lo.threads = args.threads;
    lo.cache_dir = args.cache_dir.clone();
    let model = Haru::load(&args.model.to_string_lossy(), &lo)?;
    let info = model.info();
    eprintln!(
        "[haru: {} layers, hidden {}, {:.2} GB of weights from {}; {} Hz, frames of {} samples; cpu, {} threads; loaded in {:.2}s]",
        info.layers,
        info.hidden_size,
        info.weight_bytes as f64 / 1e9,
        info.dir.display(),
        info.sample_rate,
        info.frame_samples,
        info.threads,
        info.load_secs
    );
    let fs = info.frame_samples;
    let mut user = oxinfer::tts::load_audio_file(input, info.sample_rate).with_context(|| input.display().to_string())?;
    let count = user.len().div_ceil(fs).max(1);
    user.resize(count * fs, 0.0);
    let reference = match args.ref_audio.as_slice() {
        [] => None,
        [clip] => Some(model.load_reference(clip)?),
        _ => bail!("--ref-audio: Haru takes one clip of the voice it should speak with"),
    };
    let mut config = DuplexConfig::default();
    if let Some(j) = &args.speak_json {
        let v: serde_json::Value = serde_json::from_str(j).context("--speak-json")?;
        config.speak_first = v["speak_first"].as_bool().unwrap_or(false);
        if let Some(b) = v["eop_bias"].as_f64() {
            config.eop_bias = b as f32;
        }
    }
    if let Some(g) = args.guidance {
        config.guidance_scale = g;
    }
    if let Some(t) = args.temperature_set {
        config.temperature = t;
    }
    if let Some(p) = args.top_p_set {
        config.top_p = p;
    }
    // (64 is the command line's default for chat models: the reference's 25 unless another is given.)
    config.top_k = if args.top_k == 64 { config.top_k } else { args.top_k };
    config.seed = args.seed;
    config.max_frames = count + 1;
    let mut session = DuplexSession::new(&model, args.system.as_deref().unwrap_or(""), reference.as_deref(), config);

    // The model's frame t is heard while the user speaks frame t and was computed from the
    // user's frames before it: the last user frame needs no step.
    let t0 = Instant::now();
    let mut frame = session.start()?;
    let prompt_secs = t0.elapsed().as_secs_f64();
    let mut reply = Vec::with_capacity(count * fs);
    let mut out = std::io::stdout().lock();
    let t1 = Instant::now();
    for index in 0..count {
        reply.extend_from_slice(&frame.audio);
        out.write_all(frame.text.as_bytes())?;
        out.flush()?;
        if args.show_tokens {
            eprintln!("[frame {}: token {} {:?}]", frame.index, frame.token, frame.text);
        }
        if index + 1 < count {
            frame = session.step(&user[index * fs..(index + 1) * fs])?;
        }
    }
    writeln!(out)?;
    let secs = t1.elapsed().as_secs_f64();
    let audio = Audio {
        samples: reply,
        sample_rate: info.sample_rate,
    };
    let path = args.output.clone().unwrap_or_else(|| "reply.wav".into());
    audio.write_wav(&path)?;
    eprintln!(
        "[{} prompt positions in {prompt_secs:.2}s | {count} frames ({:.1} s) in {secs:.2}s = {:.0} ms per frame (real time: {:.0} ms) | wrote {}]",
        session.prompt_length,
        audio.duration_secs(),
        1000.0 * secs / (count.max(2) - 1) as f64,
        1000.0 * fs as f64 / info.sample_rate as f64,
        path.display()
    );
    Ok(())
}
