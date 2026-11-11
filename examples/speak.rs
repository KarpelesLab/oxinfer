//! Text to speech through the library:
//! `cargo run --release --example speak -- TEXT [options]`
//! (Orpheus TTS; the model defaults to unsloth/orpheus-3b-0.1-ft, an ungated copy of
//! canopylabs/orpheus-3b-0.1-ft, and the SNAC decoder to hubertsiuzdak/snac_24khz).
//!
//! Options: `--model M`, `--voice NAME`, `-o FILE` (WAV, default speech.wav), `--stream` (audio
//! chunks as they are generated, with their latency), `--options JSON` (any `SpeakOptions` field),
//! `--quant bf16|q8|q4`, `--device cpu|metal|...`, `--threads N`, `--stop-after SECS` (cancel from
//! another thread after that long), `--also TEXT` (repeatable: more utterances, generated with the
//! first as one batch and written to FILE-1.wav, FILE-2.wav, ...; `--voices a,b,c` gives them
//! voices in turn).

use oxinfer::engine::Flow;
use oxinfer::tts::{SpeakOptions, Speaker, SpeakerOptions, Utterance};
use std::time::{Duration, Instant};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: speak TEXT [--model M] [--voice NAME] [-o FILE] [--stream] [--options JSON] [--quant Q] [--device D] [--threads N] [--stop-after SECS] [--also TEXT]... [--voices A,B,...]";
    let text = args.next().expect(usage);
    let (mut model, mut out, mut stream, mut stop_after) = ("unsloth/orpheus-3b-0.1-ft".to_string(), "speech.wav".to_string(), false, None);
    let mut lo = SpeakerOptions::default();
    let mut so = SpeakOptions::default();
    let (mut also, mut voices) = (Vec::new(), Vec::new());
    while let Some(a) = args.next() {
        let mut val = || args.next().unwrap_or_else(|| panic!("{a} needs a value"));
        match a.as_str() {
            "--model" => model = val(),
            "--voice" => so.voice = Some(val()),
            "-o" | "--output" => out = val(),
            "--stream" => stream = true,
            "--also" => also.push(val()),
            "--voices" => voices = val().split(',').map(str::to_string).collect(),
            // Replaces the options set so far (put --voice after it).
            "--options" => so = SpeakOptions::from_json(&val())?,
            "--quant" => lo.set("quant", &val())?,
            "--device" => lo.set("device", &val())?,
            "--threads" => lo.set("threads", &val())?,
            "--stop-after" => stop_after = Some(val().parse::<f64>()?),
            _ => panic!("unknown option {a}\n{usage}"),
        }
    }
    let speaker = Speaker::load(&model, &lo)?;
    if let Some(secs) = stop_after {
        let cancel = speaker.cancel_handle();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs_f64(secs));
            cancel.cancel();
        });
    }
    if !also.is_empty() {
        // A batch: every utterance is what it would be alone, generated in shared passes.
        let texts: Vec<&str> = std::iter::once(text.as_str()).chain(also.iter().map(String::as_str)).collect();
        let opts: Vec<SpeakOptions> = (0..texts.len())
            .map(|i| {
                let mut o = so.clone();
                if !voices.is_empty() {
                    o.voice = Some(voices[i % voices.len()].clone());
                }
                o
            })
            .collect();
        let items: Vec<Utterance> = texts.iter().zip(&opts).map(|(t, o)| Utterance::new(t, o)).collect();
        let (speeches, st) = speaker.speak_batch_detailed(&items)?;
        eprintln!(
            "{} utterances: {} tokens in {} steps at {:.1} tok/s, {:.2} s of audio in {:.2} s (real-time factor {:.2})",
            st.utterances,
            st.tokens,
            st.steps,
            st.tok_s(),
            st.audio_secs,
            st.total_secs,
            st.rtf()
        );
        let stem = out.strip_suffix(".wav").unwrap_or(&out);
        for (i, s) in speeches.iter().enumerate() {
            let path = format!("{stem}-{}.wav", i + 1);
            s.audio.write_wav(&path)?;
            eprintln!("wrote {path} ({:.2} s, {:?})", s.audio.duration_secs(), s.finish);
        }
        return Ok(());
    }
    let t0 = Instant::now();
    let speech = if stream {
        speaker.speak_stream(&text, &so, &mut |c| {
            eprintln!(
                "chunk at {:.3}s: {} samples ({:.2}-{:.2} s){}",
                t0.elapsed().as_secs_f64(),
                c.samples.len(),
                c.start as f64 / c.sample_rate as f64,
                (c.start + c.samples.len()) as f64 / c.sample_rate as f64,
                if c.last { ", last" } else { "" }
            );
            Flow::Continue
        })?
    } else {
        speaker.speak_detailed(&text, &so)?
    };
    let st = &speech.stats;
    eprintln!(
        "{} tokens at {:.1} tok/s, {:.2} s of audio in {:.2} s (real-time factor {:.2}), {:?}",
        st.tokens,
        st.tok_s(),
        st.audio_secs,
        st.total_secs,
        st.rtf(),
        speech.finish
    );
    speech.audio.write_wav(&out)?;
    eprintln!("wrote {out}");
    Ok(())
}
