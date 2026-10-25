//! Text to speech through the library:
//! `cargo run --release --example speak -- TEXT [options]`
//! (Orpheus TTS; the model defaults to unsloth/orpheus-3b-0.1-ft, an ungated copy of
//! canopylabs/orpheus-3b-0.1-ft, and the SNAC decoder to hubertsiuzdak/snac_24khz).
//!
//! Options: `--model M`, `--voice NAME`, `-o FILE` (WAV, default speech.wav), `--stream` (audio
//! chunks as they are generated, with their latency), `--options JSON` (any `SpeakOptions` field),
//! `--quant bf16|q8|q4`, `--device cpu|metal|...`, `--threads N`, `--stop-after SECS` (cancel from
//! another thread after that long).

use oxinfer::engine::Flow;
use oxinfer::tts::{SpeakOptions, Speaker, SpeakerOptions};
use std::time::{Duration, Instant};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: speak TEXT [--model M] [--voice NAME] [-o FILE] [--stream] [--options JSON] [--quant Q] [--device D] [--threads N] [--stop-after SECS]";
    let text = args.next().expect(usage);
    let (mut model, mut out, mut stream, mut stop_after) = ("unsloth/orpheus-3b-0.1-ft".to_string(), "speech.wav".to_string(), false, None);
    let mut lo = SpeakerOptions::default();
    let mut so = SpeakOptions::default();
    while let Some(a) = args.next() {
        let mut val = || args.next().unwrap_or_else(|| panic!("{a} needs a value"));
        match a.as_str() {
            "--model" => model = val(),
            "--voice" => so.voice = Some(val()),
            "-o" | "--output" => out = val(),
            "--stream" => stream = true,
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
