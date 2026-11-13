//! `oxinfer --model fishaudio/drama-3 --speak TEXT -o out.wav`: text to speech through
//! `oxinfer::fish`, with the options of the Orpheus command line where they apply (`--temp`,
//! `--top-p`, `--max-tokens`, `--seed`, `--speak-json`, `--speak-file`, `--speak-stream`,
//! `--ref-audio` with `--ref-text`, `--dump-pcm`, `--mmap`, `--quant`). `--voice N` is the speaker tag
//! (0 to 15) of text that carries none.

use crate::Args;
use anyhow::{Context, Result, bail};
use oxinfer::engine::Flow;
use oxinfer::fish::{Drama, DramaOptions, DramaSpeak, Reference};
use oxinfer::tts::Audio;
use std::io::Read;
use std::time::Instant;

/// `path` with `-N` before its extension (`out.wav` -> `out-3.wav`).
fn numbered(path: &std::path::Path, n: usize) -> std::path::PathBuf {
    let stem = path.file_stem().unwrap_or_default().to_string_lossy();
    let name = match path.extension() {
        Some(e) => format!("{stem}-{n}.{}", e.to_string_lossy()),
        None => format!("{stem}-{n}"),
    };
    path.with_file_name(name)
}

fn write_audio(out: &std::path::Path, dump_pcm: Option<&std::path::Path>, audio: &Audio) -> Result<()> {
    audio.write_wav(out)?;
    eprintln!(
        "[wrote {} ({:.2} s of audio at {} Hz)]",
        out.display(),
        audio.duration_secs(),
        audio.sample_rate
    );
    if let Some(p) = dump_pcm {
        let bytes: Vec<u8> = audio.samples.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(p, bytes)?;
        eprintln!("[wrote {} f32 samples to {}]", audio.samples.len(), p.display());
    }
    Ok(())
}

pub fn run(args: &Args) -> Result<()> {
    if args.decode_tokens.is_some() || args.snac_encode.is_some() || args.decode_codes.is_some() {
        bail!("--decode-tokens, --snac-encode and --decode-codes are for Orpheus and SNAC models");
    }
    let stdin = || -> Result<String> {
        let mut t = String::new();
        std::io::stdin().read_to_string(&mut t)?;
        Ok(t)
    };
    let mut texts: Vec<String> = Vec::new();
    match args.speak.as_deref() {
        Some("-") => texts.push(stdin()?),
        Some(t) => texts.push(t.to_string()),
        None => {}
    }
    texts.extend(args.speak_more.iter().cloned());
    if let Some(f) = &args.speak_file {
        let all = if f.as_os_str() == "-" {
            stdin()?
        } else {
            std::fs::read_to_string(f).with_context(|| format!("read {}", f.display()))?
        };
        texts.extend(all.lines().map(str::trim).filter(|l| !l.is_empty()).map(str::to_string));
    }
    if texts.is_empty() {
        bail!("--speak TEXT or --speak-file FILE: nothing to say");
    }
    // Before the 40 GB are read.
    for t in &texts {
        oxinfer::fish::validate_text(t, "text")?;
    }
    let batch = texts.len() > 1 || args.speak_file.is_some();
    let mut so = match &args.speak_json {
        Some(j) => DramaSpeak::from_json(j)?,
        None => DramaSpeak::default(),
    };
    if let Some(v) = args.voice.as_deref().filter(|v| !v.is_empty()) {
        so.speaker = v.parse().ok().filter(|n| *n <= 15).with_context(|| {
            format!("--voice {v}: Drama 3 takes a speaker tag, 0 to 15 (a voice is cloned with --ref-audio and --ref-text)")
        })?;
    }
    if args.ref_audio.len() != args.ref_text.len() {
        bail!(
            "--ref-audio ({}) and --ref-text ({}): every reference clip needs its transcript",
            args.ref_audio.len(),
            args.ref_text.len()
        );
    }
    for (a, t) in args.ref_audio.iter().zip(&args.ref_text) {
        so.references.push(Reference {
            audio: a.clone(),
            text: t.clone(),
        });
    }
    if let Some(t) = args.temperature_set {
        so.temperature = t;
    }
    if let Some(p) = args.top_p_set {
        so.top_p = p;
    }
    if let Some(n) = args.max_tokens_set {
        so.max_tokens = n;
    }
    so.seed = args.seed;

    let mut lo = DramaOptions::default();
    lo.cache_dir = args.cache_dir.clone();
    lo.mmap = args.mmap;
    lo.quant = args.quant;
    let d = Drama::load(&args.model.to_string_lossy(), &lo)?;
    let info = d.info();
    eprintln!(
        "[drama 3: {} layers, hidden {}, {:.1} GB of weights from {} (FP8 experts, the rest {}); codec at {} Hz, {:.1} frames/s; cpu, {} threads; loaded in {:.1}s]",
        info.layers,
        info.hidden_size,
        info.weight_bytes as f64 / 1e9,
        info.path.display(),
        info.quant.as_str(),
        info.sample_rate,
        info.frame_rate,
        info.threads,
        info.load_secs
    );
    let out = args.output.clone().unwrap_or_else(|| "speech.wav".into());
    for (i, text) in texts.iter().enumerate() {
        let t0 = Instant::now();
        let mut chunks = 0;
        let speech = d.speak_stream(text.trim(), &so, &mut |c| {
            chunks += 1;
            if args.speak_stream || args.show_tokens {
                eprintln!(
                    "[chunk {chunks}: {:.3}-{:.3} s at {:.3} s]",
                    c.start as f64 / c.sample_rate as f64,
                    (c.start + c.samples.len()) as f64 / c.sample_rate as f64,
                    t0.elapsed().as_secs_f64()
                );
            }
            Flow::Continue
        })?;
        if args.show_tokens {
            eprintln!(
                "[frames ({} x {} codes): {:?}]",
                speech.stats.frames, info.n_codebooks, speech.tokens
            );
        }
        let st = &speech.stats;
        eprintln!(
            "[{}{} prompt rows, prefill {:.2}s | {} frames in {:.2}s = {:.2} frames/s (real time needs {:.1}) | codec {:.2}s | {:.2}s audio in {:.2}s: RTF {:.2} | {:?}]",
            if batch { format!("{}: ", i + 1) } else { String::new() },
            st.prompt_tokens,
            st.prefill_secs,
            st.frames,
            st.generate_secs,
            st.tok_s(),
            info.frame_rate,
            st.snac_secs,
            st.audio_secs,
            st.total_secs,
            st.rtf(),
            speech.finish
        );
        if batch {
            write_audio(&numbered(&out, i + 1), None, &speech.audio)?;
        } else {
            write_audio(&out, args.dump_pcm.as_deref(), &speech.audio)?;
        }
    }
    Ok(())
}
