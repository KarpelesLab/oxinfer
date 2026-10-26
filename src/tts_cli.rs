//! `oxinfer --model <Orpheus model> --speak TEXT -o out.wav`: text to speech through
//! `oxinfer::tts`; `--decode-tokens FILE` decodes audio tokens without the language model.

use crate::Args;
use anyhow::{Context, Result, bail};
use oxinfer::engine::Flow;
use oxinfer::snac::{Noise, Snac, SnacOptions};
use oxinfer::tts::{Audio, SpeakOptions, Speaker, SpeakerOptions, audio_frames, frames_to_codes, load_audio_file};
use std::io::Read;
use std::time::Instant;

/// Token ids from a file: a JSON array, or numbers separated by whitespace or commas.
fn read_ids(path: &std::path::Path) -> Result<Vec<u32>> {
    let s = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    if let Ok(v) = serde_json::from_str::<Vec<u32>>(&s) {
        return Ok(v);
    }
    s.split(|c: char| c.is_whitespace() || c == ',' || c == '[' || c == ']')
        .filter(|t| !t.is_empty())
        .map(|t| t.parse().with_context(|| format!("not a token id: {t:?}")))
        .collect()
}

fn write_outputs(args: &Args, audio: &Audio) -> Result<()> {
    let out = args.output.clone().unwrap_or_else(|| "speech.wav".into());
    audio.write_wav(&out)?;
    eprintln!(
        "[wrote {} ({:.2} s of audio at {} Hz)]",
        out.display(),
        audio.duration_secs(),
        audio.sample_rate
    );
    if let Some(p) = &args.dump_pcm {
        let bytes: Vec<u8> = audio.samples.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(p, bytes)?;
        eprintln!("[wrote {} f32 samples to {}]", audio.samples.len(), p.display());
    }
    Ok(())
}

pub fn run(args: &Args) -> Result<()> {
    let snac_spec = args.snac.clone();
    let mut lo = SpeakerOptions::default();
    lo.quant = args.quant;
    lo.mmap = args.mmap;
    lo.device = args.device;
    lo.snac = snac_spec.clone();
    lo.cache_dir = args.cache_dir.clone();
    if let Some(p) = &args.precision {
        lo.snac_f32 = match p.as_str() {
            "f32" | "fp32" => true,
            "bf16" => false,
            _ => bail!("--precision {p}: the SNAC decoder takes f32 or bf16"),
        };
    }
    let noise = if args.no_noise { Noise::Off } else { Noise::Seeded(args.seed) };

    // The SNAC codec on its own: audio to codes, codes to audio.
    if args.snac_encode.is_some() || args.decode_codes.is_some() {
        let mut so = SnacOptions::default();
        so.f32_weights = lo.snac_f32;
        let dir = oxinfer::engine::resolve_model(&args.model.to_string_lossy(), args.cache_dir.as_deref())?;
        let snac = Snac::load(&dir, &so)?;
        if let Some(f) = &args.snac_encode {
            let pcm = load_audio_file(f, snac.sample_rate())?;
            let t = Instant::now();
            let codes = snac.encode(&pcm)?;
            let lens: Vec<usize> = codes.iter().map(Vec::len).collect();
            eprintln!(
                "[encoded {:.2} s at {} Hz to codes {lens:?} in {:.3}s]",
                pcm.len() as f64 / snac.sample_rate() as f64,
                snac.sample_rate(),
                t.elapsed().as_secs_f64()
            );
            let json = serde_json::to_string(&codes)?;
            match &args.output {
                Some(o) => std::fs::write(o, json).with_context(|| format!("write {}", o.display()))?,
                None => println!("{json}"),
            }
            return Ok(());
        }
        let f = args.decode_codes.as_ref().unwrap();
        let codes: Vec<Vec<u32>> = serde_json::from_slice(&std::fs::read(f).with_context(|| format!("read {}", f.display()))?)
            .context("codes: a JSON array per codebook")?;
        let refs: Vec<&[u32]> = codes.iter().map(Vec::as_slice).collect();
        let t = Instant::now();
        let samples = snac.decode(&refs, noise)?;
        eprintln!("[decoded {} coarse frames in {:.3}s]", codes[0].len(), t.elapsed().as_secs_f64());
        let audio = Audio {
            samples,
            sample_rate: snac.sample_rate(),
        };
        return write_outputs(args, &audio);
    }

    // Audio tokens (or, with a SNAC model, the same) decoded without the language model.
    if let Some(f) = &args.decode_tokens {
        let ids = read_ids(f)?;
        let frames = audio_frames(&ids);
        let (audio, secs) = if oxinfer::snac::is_snac(&args.model) {
            let mut so = SnacOptions::default();
            so.f32_weights = lo.snac_f32;
            let snac = Snac::load(&args.model, &so)?;
            let c = frames_to_codes(&frames);
            let t = Instant::now();
            let samples = snac.decode(&[&c[0], &c[1], &c[2]], noise)?;
            let audio = Audio {
                samples,
                sample_rate: snac.sample_rate(),
            };
            (audio, t.elapsed().as_secs_f64())
        } else {
            let speaker = Speaker::load(&args.model.to_string_lossy(), &lo)?;
            let t = Instant::now();
            (speaker.decode_frames(&frames, noise)?, t.elapsed().as_secs_f64())
        };
        eprintln!("[decoded {} frames in {secs:.3}s]", frames.len());
        return write_outputs(args, &audio);
    }

    let mut text = args.speak.clone().context("--speak TEXT")?;
    if text == "-" {
        text.clear();
        std::io::stdin().read_to_string(&mut text)?;
    }
    let mut so = match &args.speak_json {
        Some(j) => SpeakOptions::from_json(j)?,
        None => SpeakOptions::default(),
    };
    if let Some(v) = &args.voice {
        so.voice = (!v.is_empty()).then(|| v.clone());
    }
    if let Some(t) = args.temperature_set {
        so.temperature = t;
    }
    if let Some(p) = args.top_p_set {
        so.top_p = p;
    }
    if let Some(r) = args.repetition_penalty {
        so.repetition_penalty = r;
    }
    if let Some(n) = args.max_tokens_set {
        so.max_tokens = n;
    }
    so.seed = args.seed;
    so.noise = !args.no_noise;
    if args.unconstrained {
        so.constrained = false;
    }

    let speaker = Speaker::load(&args.model.to_string_lossy(), &lo)?;
    if args.ref_audio.len() != args.ref_text.len() {
        bail!(
            "--ref-audio and --ref-text go in pairs ({} vs {})",
            args.ref_audio.len(),
            args.ref_text.len()
        );
    }
    for (a, t) in args.ref_audio.iter().zip(&args.ref_text) {
        let ts = Instant::now();
        let r = speaker.encode_reference_file(a, t)?;
        eprintln!(
            "[reference {}: {} frames ({:.2} s) encoded in {:.2}s]",
            a.display(),
            r.frames.len(),
            r.duration_secs(),
            ts.elapsed().as_secs_f64()
        );
        so.references.push(r);
    }
    let info = speaker.info();
    eprintln!(
        "[orpheus{}{}: {} layers, hidden {}, {}; SNAC {} Hz from {}; device {}, {} threads; loaded in {:.2}s]",
        if info.pretrained { " (pretrained)" } else { "" },
        if info.languages.is_empty() {
            String::new()
        } else {
            format!(" [{}]", info.languages.join(", "))
        },
        info.layers,
        info.hidden_size,
        if oxinfer::gguf::is_gguf(&info.path) {
            "GGUF".to_string()
        } else {
            format!("{:?}", info.quant)
        },
        info.sample_rate,
        info.snac_path.display(),
        info.device,
        info.threads,
        info.load_secs
    );
    let speech = if args.speak_stream {
        let t0 = Instant::now();
        let mut chunks = 0;
        speaker.speak_stream(text.trim(), &so, &mut |c| {
            chunks += 1;
            if args.show_tokens {
                eprintln!(
                    "[chunk {chunks}: {:.3}-{:.3} s at {:.3} s]",
                    c.start as f64 / c.sample_rate as f64,
                    (c.start + c.samples.len()) as f64 / c.sample_rate as f64,
                    t0.elapsed().as_secs_f64()
                );
            }
            Flow::Continue
        })?
    } else {
        speaker.speak_detailed(text.trim(), &so)?
    };
    if args.show_tokens {
        eprintln!(
            "[prompt ids: {:?}]",
            speaker.prompt_tokens_with(text.trim(), so.voice.as_deref(), &so.references)?
        );
        eprintln!("[output ids ({}): {:?}]", speech.tokens.len(), speech.tokens);
    }
    let st = &speech.stats;
    eprintln!(
        "[{} prompt tok, prefill {:.2}s | {} tok ({} frames, {} dropped{}) in {:.2}s = {:.1} tok/s (real time needs {:.1}) | SNAC {:.2}s | {:.2}s audio in {:.2}s: RTF {:.2}{} | {:?}]",
        st.prompt_tokens,
        st.prefill_secs,
        st.tokens,
        st.frames,
        st.dropped_tokens,
        if st.loops > 0 {
            format!(", {} loops", st.loops)
        } else {
            String::new()
        },
        st.generate_secs,
        st.tok_s(),
        7.0 * speaker.snac().sample_rate() as f64 / speaker.snac().samples_per_frame() as f64,
        st.snac_secs,
        st.audio_secs,
        st.total_secs,
        st.rtf(),
        st.first_chunk_secs
            .map(|s| format!(", first audio after {s:.2}s"))
            .unwrap_or_default(),
        speech.finish
    );
    write_outputs(args, &speech.audio)
}
