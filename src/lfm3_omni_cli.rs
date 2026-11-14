//! Spoken replies from LFM3-Omni (see docs/lfm3.md). Text replies go through the normal chat
//! path (`--audio` / `--image` with `-p`); this is for the two shapes that write a WAV:
//!
//! ```sh
//! # text to speech: the reply is speech only
//! oxinfer --model LiquidAI/LFM3-Omni-2B --speak "The quick brown fox." -o out.wav
//! # voice chat: a spoken (or typed) question, the answer as text on stdout and speech in the WAV
//! oxinfer --model LiquidAI/LFM3-Omni-2B --audio question.wav -o reply.wav
//! ```
//!
//! `--system` replaces the default system prompt ("Perform TTS in American English using a
//! female voice.", with `--voice male` for the other voice; "Respond with interleaved text and
//! audio."). `--temp`, `--top-k`, `--repetition-penalty` and `--seed` sample the speech codes
//! (the checkpoint's defaults: 0.4, none, 1.2); the text tokens are greedy. `--max-tokens`
//! bounds the generated slots (a text token or a frame of 80 ms). `--speak-json` takes
//! `{"mode": "audio" | "interleaved", "fixed_order": true}`; `--dump-pcm FILE` writes the f32
//! samples and `FILE.codes.json` the codes.

use crate::Args;
use anyhow::{Context, Result, bail};
use oxinfer::chat::{ChatFormat, Lfm3Chat};
use oxinfer::lfm3_omni::{self as omni, Mode, OmniConfig, Speech, SpeechOptions};
use oxinfer::{config, generate, qwen, safetensors};
use std::io::Write;
use std::time::Instant;

/// Whether the command line asks this model for speech: `--speak`, or an output WAV.
pub fn wants_speech(args: &Args) -> bool {
    args.speak.is_some()
        || args
            .output
            .as_ref()
            .is_some_and(|o| o.extension().is_some_and(|e| e.eq_ignore_ascii_case("wav")))
}

pub fn run(args: &Args) -> Result<()> {
    let t0 = Instant::now();
    if !args.videos.is_empty() || args.speak_file.is_some() || !args.speak_more.is_empty() {
        bail!("LFM3-Omni speech: one --speak TEXT, or --audio / --image / -p with -o reply.wav");
    }
    let json: serde_json::Value = match &args.speak_json {
        Some(j) => serde_json::from_str(j).context("--speak-json")?,
        None => serde_json::Value::Null,
    };
    let mode = match (json["mode"].as_str(), &args.speak) {
        (Some("audio"), _) | (None, Some(_)) => Mode::Audio,
        (Some("interleaved"), _) | (None, None) => Mode::Interleaved,
        (Some(m), _) => bail!("--speak-json: mode {m:?} (audio or interleaved)"),
    };
    let user = match (&args.speak, &args.prompt) {
        (Some(_), Some(_)) => bail!("--speak TEXT is the text to say: leave -p out"),
        (Some(s), None) => s.clone(),
        (None, p) => p.clone().unwrap_or_default(),
    };
    let system = match (&args.system, &args.speak) {
        (Some(s), _) => s.clone(),
        (None, Some(_)) if mode == Mode::Audio => {
            let voice = args.voice.as_deref().unwrap_or("female");
            if !matches!(voice, "female" | "male") {
                bail!("--voice: female or male");
            }
            format!("Perform TTS in American English using a {voice} voice.")
        }
        _ => "Respond with interleaved text and audio.".to_string(),
    };

    let cfg = config::QwenConfig::load(&args.model)?;
    let ocfg = OmniConfig::load(&args.model)?;
    let files = safetensors::ModelFiles::open(&args.model, !args.mmap)?;
    let model = qwen::QwenModel::load(cfg, &files, args.quant)?;
    let speech = Speech::load(ocfg, &files)?;
    let codec_dir = args.model.join("audio_tokenizer");
    let codec_files = safetensors::ModelFiles::open(&codec_dir, true).with_context(|| format!("the codec in {}", codec_dir.display()))?;
    let codec = omni::codec::CodecDecoder::load(omni::codec::CodecConfig::load(&codec_dir)?, &codec_files)?;
    let tok = oxinfer::engine::load_tokenizer(&args.model)?;
    let eos = config::load_eos_ids(&args.model);
    eprintln!(
        "[lfm3-omni: model, speech head and codec loaded in {:.2}s]",
        t0.elapsed().as_secs_f64()
    );

    let (mm_text, feats) = if args.images.is_empty() && args.audios.is_empty() {
        Default::default()
    } else {
        oxinfer::mm::prepare(&crate::mm_request(args), &files, model.cfg.hidden_size)?
    };
    let fmt = Lfm3Chat;
    let mut text = fmt.conversation_start(Some(&system).filter(|s| !s.is_empty()).map(|s| s.as_str()), false);
    text.push_str(&fmt.user_turn(&format!("{mm_text}{user}"), false));
    if args.print_prompt {
        println!("{text}");
        return Ok(());
    }
    let mut ids = tok
        .encode(text.as_str(), false)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?
        .get_ids()
        .to_vec();
    ids.push(mode.marker(&speech.cfg));

    let mut opts = SpeechOptions::load(&args.model);
    if let Some(t) = args.temperature_set {
        opts.temperature = t;
    }
    if args.top_k != 64 {
        opts.top_k = args.top_k;
    }
    if let Some(p) = args.repetition_penalty {
        opts.repetition_penalty = p;
    }
    opts.seed = args.seed;
    opts.max_new = args.max_tokens_set.unwrap_or(opts.max_new);
    opts.fixed_order = json["fixed_order"].as_bool().unwrap_or(false);

    let t1 = Instant::now();
    let mut sess = model.new_session();
    let overrides = oxinfer::mm::bind(&model, &mut sess, &ids, &feats)?;
    let hs = generate::prefill(&model, &mut sess, &ids, args.prefill_chunk.unwrap_or(256), &overrides);
    let prompt_secs = t1.elapsed().as_secs_f64();

    let t2 = Instant::now();
    let mut out = std::io::stdout().lock();
    let mut shown = 0;
    let mut text_ids: Vec<u32> = Vec::new();
    let reply = omni::generate(
        &model,
        &speech,
        &mut sess,
        &hs,
        mode,
        &opts,
        &eos,
        |t| {
            text_ids.push(t);
            if args.show_tokens {
                eprintln!("[text {t} {:?}]", tok.id_to_token(t).unwrap_or_default());
            }
            // The stable decoded prefix, as it grows.
            if let Ok(s) = tok.decode(&text_ids, true)
                && !s.ends_with('\u{fffd}')
                && s.len() > shown
            {
                let _ = out.write_all(&s.as_bytes()[shown..]);
                let _ = out.flush();
                shown = s.len();
            }
        },
        |codes| {
            if args.show_tokens {
                eprintln!("[frame {codes:?}]");
            }
        },
    );
    if shown > 0 {
        writeln!(out)?;
    }
    let gen_secs = t2.elapsed().as_secs_f64();

    let t3 = Instant::now();
    let pcm = codec.decode(&reply.codes, reply.frames)?;
    let rate = codec.cfg.sample_rate;
    let secs = pcm.len() as f64 / rate as f64;
    let path = args.output.clone().unwrap_or_else(|| "speech.wav".into());
    std::fs::write(&path, oxinfer::tts::wav_bytes(&pcm, rate as u32)).with_context(|| format!("write {}", path.display()))?;
    if let Some(p) = &args.dump_pcm {
        let b: Vec<u8> = pcm.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(p, b).with_context(|| format!("write {}", p.display()))?;
        let rows: Vec<&[u32]> = reply.codes.chunks(speech.cfg.num_codebooks).collect();
        let cp = format!("{}.codes.json", p.display());
        std::fs::write(&cp, serde_json::to_vec(&rows)?).with_context(|| format!("write {cp}"))?;
    }
    eprintln!(
        "[{} prompt tokens in {prompt_secs:.2}s; {} text tokens and {} frames in {gen_secs:.2}s{}; {secs:.2}s of audio decoded in {:.2}s ({:.1}x real time overall) -> {}]",
        ids.len(),
        reply.text.len(),
        reply.frames,
        if reply.finished { "" } else { " (stopped at --max-tokens)" },
        t3.elapsed().as_secs_f64(),
        secs / (gen_secs + t3.elapsed().as_secs_f64()).max(1e-9),
        path.display()
    );
    Ok(())
}
