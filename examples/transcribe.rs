//! Speech to text through the library:
//! `cargo run --release --example transcribe -- MODEL AUDIO [options]`
//! (MODEL is a local directory or a repo id such as openai/whisper-small; AUDIO is any file
//! oxideav decodes, or raw 16 kHz f32 samples with `--raw`).
//!
//! Options: `--language xx`, `--translate`, `--timestamps`, `--words`, `--fallback`,
//! `--prompt TEXT`, `--condition` (on previous text), `--json`, `--precision f32|f16|bf16`, `--quant q8|q4` (decoder),
//! `--device cpu|metal|...`, `--threads N`, `--profile`, `--dump DIR` (writes the first window's log-mel features, encoder
//! output and first-step logits as f32 files, plus the PCM, for reference comparisons).

use oxinfer::engine::Flow;
use oxinfer::whisper::{Task, TranscribeOptions, Transcriber, TranscriberOptions, mel};
use std::path::{Path, PathBuf};

fn write_f32(path: &Path, v: &[f32]) -> anyhow::Result<()> {
    std::fs::write(path, v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: transcribe MODEL AUDIO [--language xx] [--translate] [--timestamps] [--words] [--fallback] [--prompt TEXT] [--condition] [--json] [--raw] [--precision P] [--quant Q] [--device D] [--threads N] [--dump DIR]";
    let model = args.next().expect(usage);
    let audio = PathBuf::from(args.next().expect(usage));
    let mut lo = TranscriberOptions::default();
    let mut o = TranscribeOptions::default();
    let (mut json, mut raw, mut dump, mut profile) = (false, false, None::<PathBuf>, false);
    while let Some(a) = args.next() {
        let mut val = || args.next().unwrap_or_else(|| panic!("{a} needs a value"));
        match a.as_str() {
            "--language" => o.language = Some(val()),
            "--translate" => o.task = Some(Task::Translate),
            "--timestamps" => o.timestamps = true,
            "--words" => o.word_timestamps = true,
            "--fallback" => o = o.with_fallback(),
            "--prompt" => o.initial_prompt = Some(val()),
            "--condition" => o.condition_on_prev_tokens = true,
            "--json" => json = true,
            "--raw" => raw = true,
            "--precision" => lo.set("precision", &val())?,
            "--quant" => lo.set("decoder", &val())?,
            "--device" => lo.set("device", &val())?,
            "--threads" => lo.set("threads", &val())?,
            "--dump" => dump = Some(PathBuf::from(val())),
            "--profile" => profile = true,
            _ => panic!("unknown option {a}\n{usage}"),
        }
    }
    oxinfer::hf::set_hub(oxinfer::hf::Hub::Auto);
    let asr = Transcriber::load(&model, &lo)?;
    let info = asr.info();
    eprintln!(
        "[whisper d={} enc {}L dec {}L, {} mels, encoder {}, decoder {}, device {}, {:.2}s]",
        info.d_model,
        info.encoder_layers,
        info.decoder_layers,
        info.n_mels,
        info.encoder.as_str(),
        info.decoder.as_str(),
        info.device,
        info.load_secs
    );
    let pcm: Vec<f32> = if raw {
        std::fs::read(&audio)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    } else {
        Transcriber::load_audio(&audio)?
    };
    if let Some(d) = &dump {
        std::fs::create_dir_all(d)?;
        write_f32(&d.join("pcm.f32"), &pcm)?;
        let m = asr.model();
        let feats = mel::log_mel(&pcm, m.cfg.num_mel_bins);
        write_f32(&d.join("mel.f32"), &feats.data)?;
        let enc = m.encode(&feats.window(0));
        write_f32(&d.join("enc.f32"), &enc)?;
        // First-step logits after <|startoftranscript|> [<|en|> <|transcribe|>] <|notimestamps|>.
        let sp = asr.special();
        let mut ids = vec![sp.sot];
        if sp.multilingual {
            ids.push(sp.language_token(o.language.as_deref().unwrap_or("en"))?);
            ids.push(sp.transcribe.unwrap());
        }
        ids.push(sp.no_timestamps);
        let cross = m.cross_kv(&enc);
        let mut st = m.new_state();
        let h = m.decode(&ids, &mut st, &cross);
        let d_model = m.cfg.d_model;
        write_f32(&d.join("logits.f32"), &m.logits(&h[(ids.len() - 1) * d_model..]))?;
        eprintln!(
            "[dumped pcm, mel ({} frames), encoder output and logits to {}]",
            feats.frames,
            d.display()
        );
    }
    oxinfer::prof::reset();
    let t = asr.transcribe_with(&pcm, &o, &mut |s| {
        if !json && (o.timestamps || o.word_timestamps) {
            println!("[{:7.2} -> {:7.2}] {}", s.start, s.end, s.text);
            for w in &s.words {
                println!("    {:7.2} {:7.2} {:.2} {}", w.start, w.end, w.probability, w.word);
            }
        }
        Flow::Continue
    })?;
    if json {
        println!("{}", serde_json::to_string_pretty(&t)?);
    } else if !(o.timestamps || o.word_timestamps) {
        println!("{}", t.text);
    }
    let s = &t.stats;
    if profile {
        oxinfer::prof::report(s.total_secs, s.tokens);
        eprint!("[encoder attention]\n{}", oxinfer::sd::ops::timing::report());
    }
    if oxinfer::sd::ops::gpu_check::on() {
        eprintln!("[gpu check: {}]", oxinfer::sd::ops::gpu_check::summary());
    }
    eprintln!(
        "[language {}{} | {:.1}s audio, {} windows, {} tokens | mel {:.2}s, encode {:.2}s, decode {:.2}s, total {:.2}s, RTF {:.3}]",
        t.language,
        t.language_probability.map(|p| format!(" (p={p:.2})")).unwrap_or_default(),
        s.audio_secs,
        s.windows,
        s.tokens,
        s.mel_secs,
        s.encode_secs,
        s.decode_secs,
        s.total_secs,
        s.rtf
    );
    Ok(())
}
