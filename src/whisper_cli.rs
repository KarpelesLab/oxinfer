//! `oxinfer --model <Whisper model> --transcribe FILE`: speech to text through
//! `oxinfer::whisper`, printing the text, timed segments, JSON or subtitles. Several
//! `--transcribe` files are decoded together as a batch.

use crate::Args;
use anyhow::{Context, Result, bail};
use oxinfer::engine::Flow;
use oxinfer::model::Quant;
use oxinfer::whisper::{Precision, Segment, Task, TranscribeOptions, Transcriber, TranscriberOptions, Transcript};
use std::io::Write;
use std::path::PathBuf;

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

/// What a finished segment prints: a timed line (and its words) or a subtitle cue; nothing for
/// plain text and JSON, which are printed at the end. `n` numbers the cue from 1.
fn segment_text(format: Format, timestamps: bool, n: usize, s: &Segment) -> String {
    match format {
        Format::Text if timestamps => {
            let mut out = format!("[{} --> {}] {}\n", stamp(s.start, '.'), stamp(s.end, '.'), s.text);
            for w in &s.words {
                out.push_str(&format!("    [{:8.2} {:8.2}] {:4.2} {}\n", w.start, w.end, w.probability, w.word));
            }
            out
        }
        Format::Srt => format!("{n}\n{} --> {}\n{}\n\n", stamp(s.start, ','), stamp(s.end, ','), s.text),
        Format::Vtt => format!("{} --> {}\n{}\n\n", stamp(s.start, '.'), stamp(s.end, '.'), s.text),
        _ => String::new(),
    }
}

/// The statistics line of one transcript.
fn report(t: &Transcript, decoded: &str) {
    let s = &t.stats;
    eprintln!(
        "[{} ({}) | {:.1}s of audio{decoded} | features {:.2}s, encoder {:.2}s ({} windows), decoder {:.2}s ({} tokens) | {:.2}s total, {:.1}x real time]",
        t.language,
        t.language_probability
            .map_or("given".to_string(), |p| format!("detected, p={p:.2}")),
        s.audio_secs,
        s.mel_secs,
        s.encode_secs,
        s.windows,
        s.decode_secs,
        s.tokens,
        s.total_secs,
        1.0 / s.rtf.max(1e-9)
    );
}

/// Several files as a batch: the output of each is kept until all are done and printed in the
/// order given (text and JSON), or written next to the input (subtitles).
fn run_batch(asr: &Transcriber, o: &TranscribeOptions, format: Format, files: &[&PathBuf], profile: bool) -> Result<()> {
    let t0 = std::time::Instant::now();
    let pcms: Vec<Vec<f32>> = files
        .iter()
        .map(|f| Transcriber::load_audio(f).with_context(|| format!("{}", f.display())))
        .collect::<Result<_>>()?;
    let load_audio = t0.elapsed().as_secs_f64();
    let refs: Vec<&[f32]> = pcms.iter().map(|p| &p[..]).collect();
    let mut bodies = vec![String::new(); files.len()];
    let mut cues = vec![0usize; files.len()];
    let t1 = std::time::Instant::now();
    let ts = asr.transcribe_batch_with(&refs, o, &mut |i, s| {
        cues[i] += 1;
        bodies[i].push_str(&segment_text(format, o.timestamps, cues[i], s));
        Flow::Continue
    })?;
    let wall = t1.elapsed().as_secs_f64();
    match format {
        Format::Json => {
            let mut all = Vec::with_capacity(ts.len());
            for (f, t) in files.iter().zip(&ts) {
                let mut v = serde_json::json!({ "file": f.display().to_string() });
                v.as_object_mut()
                    .unwrap()
                    .extend(serde_json::to_value(t)?.as_object().cloned().unwrap_or_default());
                all.push(v);
            }
            println!("{}", serde_json::to_string_pretty(&all)?);
        }
        Format::Srt | Format::Vtt => {
            for (f, body) in files.iter().zip(&bodies) {
                let (ext, head) = if format == Format::Vtt {
                    ("vtt", "WEBVTT\n\n")
                } else {
                    ("srt", "")
                };
                let path = f.with_extension(ext);
                std::fs::write(&path, format!("{head}{body}")).with_context(|| format!("write {}", path.display()))?;
                eprintln!("[wrote {}]", path.display());
            }
        }
        Format::Text => {
            for (i, ((f, t), body)) in files.iter().zip(&ts).zip(&bodies).enumerate() {
                if i > 0 {
                    println!();
                }
                println!("== {} ==", f.display());
                if o.timestamps {
                    print!("{body}");
                } else {
                    println!("{}", t.text);
                }
            }
        }
    }
    for t in &ts {
        report(t, "");
    }
    let audio: f64 = ts.iter().map(|t| t.stats.audio_secs).sum();
    let tokens: usize = ts.iter().map(|t| t.stats.tokens).sum();
    eprintln!(
        "[{} files, {:.1}s of audio decoded in {:.2}s | transcribed together ({} at a time) in {:.2}s, {:.1}x real time]",
        ts.len(),
        audio,
        load_audio,
        o.batch_size.max(1).min(ts.len()),
        wall,
        audio / wall.max(1e-9)
    );
    if profile {
        oxinfer::prof::report(wall, tokens);
    }
    Ok(())
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
    o.beam_size = args.beam;
    o.patience = args.patience;
    o.length_penalty = args.length_penalty;
    o.best_of = args.best_of;
    if let Some(n) = args.batch {
        o.batch_size = n;
    }
    if !args.transcribe_more.is_empty() {
        let files: Vec<&PathBuf> = std::iter::once(file).chain(&args.transcribe_more).collect();
        return run_batch(&asr, &o, format, &files, args.profile);
    }

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
        let _ = out.write_all(segment_text(format, o.timestamps, n, s).as_bytes());
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
    report(&t, &format!(", decoded in {load_audio:.2}s"));
    if args.profile {
        oxinfer::prof::report(s.total_secs, s.tokens);
    }
    Ok(())
}
