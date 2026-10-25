//! `oxinfer --model <Whisper model> --transcribe FILE`: speech to text through
//! `oxinfer::whisper`, printing the text, timed segments, JSON or subtitles.

use crate::Args;
use anyhow::{Context, Result, bail};
use oxinfer::engine::Flow;
use oxinfer::model::Quant;
use oxinfer::whisper::{Precision, Segment, Task, TranscribeOptions, Transcriber, TranscriberOptions};
use std::io::Write;

/// Output format of `--format`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Format {
    Text,
    Json,
    Srt,
    Vtt,
}

fn stamp(t: f64, sep: char) -> String {
    let ms = (t.max(0.0) * 1000.0).round() as u64;
    format!(
        "{:02}:{:02}:{:02}{sep}{:03}",
        ms / 3_600_000,
        ms / 60_000 % 60,
        ms / 1000 % 60,
        ms % 1000
    )
}

pub fn run(args: &Args) -> Result<()> {
    let file = args
        .transcribe
        .as_ref()
        .context("a Whisper model transcribes audio: use --transcribe FILE")?;
    let format = match args.format.as_deref().unwrap_or("text") {
        "text" | "txt" => Format::Text,
        "json" => Format::Json,
        "srt" => Format::Srt,
        "vtt" => Format::Vtt,
        f => bail!("unknown --format {f} (text, json, srt, vtt)"),
    };
    let mut lo = TranscriberOptions::default();
    if let Some(p) = &args.precision {
        lo.set("precision", p)?;
    }
    lo.decoder = match args.quant {
        Quant::Q8 => Precision::Q8,
        Quant::Q4 => Precision::Q4,
        Quant::Bf16 => lo.decoder,
    };
    lo.mmap = true;
    let t0 = std::time::Instant::now();
    let asr = Transcriber::load_dir(&args.model, &lo)?;
    let info = asr.info();
    eprintln!(
        "[whisper: d_model {}, {} encoder / {} decoder layers, {} mel bins, {}; encoder {}, decoder {}; loaded in {:.2}s]",
        info.d_model,
        info.encoder_layers,
        info.decoder_layers,
        info.n_mels,
        if info.multilingual { "multilingual" } else { "English-only" },
        info.encoder.as_str(),
        info.decoder.as_str(),
        t0.elapsed().as_secs_f64()
    );

    let mut o = if args.fallback {
        TranscribeOptions::default().with_fallback()
    } else {
        TranscribeOptions::default()
    };
    if args.temperature > 0.0 && !args.fallback {
        o.temperatures = vec![args.temperature];
    }
    o.seed = args.seed;
    o.language = args.language.clone();
    if args.translate {
        o.task = Some(Task::Translate);
    }
    o.word_timestamps = args.word_timestamps;
    o.timestamps = args.timestamps || args.word_timestamps || matches!(format, Format::Srt | Format::Vtt);
    o.initial_prompt = args.initial_prompt.clone();
    o.prefix = args.prefix.clone();
    o.condition_on_prev_tokens = args.condition_on_prev;

    let t1 = std::time::Instant::now();
    let pcm = Transcriber::load_audio(file)?;
    let load_audio = t1.elapsed().as_secs_f64();
    if format == Format::Vtt {
        println!("WEBVTT\n");
    }
    let mut out = std::io::stdout();
    let mut n = 0usize;
    let mut on_segment = |s: &Segment| {
        n += 1;
        match format {
            Format::Text if o.timestamps => {
                let _ = writeln!(out, "[{} --> {}] {}", stamp(s.start, '.'), stamp(s.end, '.'), s.text);
                for w in &s.words {
                    let _ = writeln!(out, "    [{:8.2} {:8.2}] {:4.2} {}", w.start, w.end, w.probability, w.word);
                }
            }
            Format::Srt => {
                let _ = writeln!(out, "{n}\n{} --> {}\n{}\n", stamp(s.start, ','), stamp(s.end, ','), s.text);
            }
            Format::Vtt => {
                let _ = writeln!(out, "{} --> {}\n{}\n", stamp(s.start, '.'), stamp(s.end, '.'), s.text);
            }
            _ => {}
        }
        let _ = out.flush();
        Flow::Continue
    };
    let t = asr.transcribe_with(&pcm, &o, &mut on_segment)?;
    match format {
        Format::Text if !o.timestamps => println!("{}", t.text),
        Format::Json => println!("{}", serde_json::to_string_pretty(&t)?),
        _ => {}
    }
    let s = &t.stats;
    eprintln!(
        "[{} ({}) | {:.1}s of audio, decoded in {:.2}s | features {:.2}s, encoder {:.2}s ({} windows), decoder {:.2}s ({} tokens) | {:.2}s total, {:.1}x real time]",
        t.language,
        t.language_probability
            .map_or("given".to_string(), |p| format!("detected, p={p:.2}")),
        s.audio_secs,
        load_audio,
        s.mel_secs,
        s.encode_secs,
        s.windows,
        s.decode_secs,
        s.tokens,
        s.total_secs,
        1.0 / s.rtf.max(1e-9)
    );
    if args.profile {
        oxinfer::prof::report(s.total_secs, s.tokens);
    }
    Ok(())
}
