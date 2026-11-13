//! Dumps what `tools/zonos_ref.py` compares with Zyphra's PyTorch reference of ZONOS3-Mini:
//!
//! ```text
//! zonos_check gen MODEL OUT TEXT [--voice V] [--mode embedding|reference] [--steps N]
//!                 [--prefill N] [--cfg S] [--int8]
//! zonos_check decode MODEL CODES.npy OUT.f32 [--decoder full]
//! zonos_check ids MODEL TEXTS.txt OUT.json
//! zonos_check embed MODEL CLIP OUT.f32
//! ```
//!
//! `gen` runs the greedy generation of TEXT and writes `OUT.json` (the prompt rows, the speaker
//! embedding, the condition rows and the generated rows) and `OUT.logits` (f32: the logits of
//! every step, `[steps, codebooks, audio_vocab]`). The quantized layers are computed in f32 on
//! the dequantized weights unless `--int8` is given. `decode` writes the decoder's raw output for
//! the frames of a `.npy` (`[frames, 10]`), without the output gain. `ids` writes the text ids of
//! every line of a file, `embed` the speaker embedding of an audio clip (f32).

use anyhow::{Context, Result, bail};
use oxinfer::engine::Flow;
use oxinfer::zonos::{SpeakerMode, Zonos, ZonosOptions, ZonosSpeak, read_npy};
use std::path::Path;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let has = |name: &str| args.iter().any(|a| a == name);
    let mut lo = ZonosOptions::default();
    lo.exact = !has("--int8");
    lo.decoder = flag("--decoder");
    match args.first().map(String::as_str) {
        Some("gen") if args.len() >= 4 => {
            let z = Zonos::load(&args[1], &lo)?;
            let mut o = ZonosSpeak::default();
            o.temperature = 0.0;
            o.voice = flag("--voice");
            if flag("--mode").as_deref() == Some("reference") {
                o.speaker_mode = SpeakerMode::Reference;
            }
            if let Some(n) = flag("--steps") {
                o.max_tokens = n.parse()?;
            }
            if let Some(n) = flag("--prefill") {
                o.silence_prefill_frames = n.parse()?;
            }
            if let Some(s) = flag("--cfg") {
                o.speaker_cfg_scale = s.parse()?;
            }
            let (prompt, twin) = z.prompt(&args[3], &o)?;
            let t0 = std::time::Instant::now();
            let g = z.generate(&prompt, twin.as_ref(), &o, true, &mut |_, _| Flow::Continue)?;
            let n_cb = z.lm().cfg.n_codebooks;
            eprintln!(
                "[{} prompt rows, {} generated rows in {:.2}s, end frame {:?}]",
                prompt.rows.len() / (n_cb + 1),
                g.rows.len() / n_cb,
                t0.elapsed().as_secs_f64(),
                g.eos_frame
            );
            let json = serde_json::json!({
                "text": args[3],
                "n_codebooks": n_cb,
                "prompt": prompt.rows,
                "speaker": prompt.speaker,
                "prefix_rows": prompt.prefix_rows,
                "cond": prompt.cond,
                "prefill_rows": prompt.prefill_rows,
                "twin": twin.as_ref().map(|t| &t.rows),
                "rows": g.rows,
                "eos_frame": g.eos_frame,
            });
            std::fs::write(format!("{}.json", args[2]), serde_json::to_vec(&json)?)?;
            let bytes: Vec<u8> = g.logits.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(format!("{}.logits", args[2]), bytes)?;
        }
        Some("decode") if args.len() >= 4 => {
            let z = Zonos::load(&args[1], &lo)?;
            let (shape, v) = read_npy(Path::new(&args[2]))?;
            if shape.len() != 2 || shape[1] != z.decoder().n_codebooks {
                bail!("{}: shape {shape:?}", args[2]);
            }
            let codes: Vec<u32> = v.iter().map(|&x| x as u32).collect();
            let t0 = std::time::Instant::now();
            let pcm = z.decoder().decode(&codes, shape[0])?;
            eprintln!(
                "[{} frames ({:.2} s of audio) decoded in {:.2}s]",
                shape[0],
                pcm.len() as f64 / z.decoder().sample_rate as f64,
                t0.elapsed().as_secs_f64()
            );
            let bytes: Vec<u8> = pcm.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(&args[3], bytes).with_context(|| format!("write {}", args[3]))?;
        }
        Some("embed") if args.len() >= 4 => {
            let z = Zonos::load(&args[1], &lo)?;
            let t0 = std::time::Instant::now();
            let e = z.speaker_embedding_file(Path::new(&args[2]))?;
            eprintln!("[speaker embedding of {} values in {:.2}s]", e.len(), t0.elapsed().as_secs_f64());
            let bytes: Vec<u8> = e.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(&args[3], bytes).with_context(|| format!("write {}", args[3]))?;
        }
        Some("ids") if args.len() >= 4 => {
            let z = Zonos::load(&args[1], &lo)?;
            let text = std::fs::read_to_string(&args[2]).with_context(|| format!("read {}", args[2]))?;
            let ids = text
                .split('\n')
                .filter(|l| !l.trim().is_empty())
                .map(|l| z.text_ids(l))
                .collect::<Result<Vec<_>>>()?;
            std::fs::write(&args[3], serde_json::to_vec(&ids)?)?;
        }
        _ => bail!(
            "usage: zonos_check gen MODEL OUT TEXT [options] | zonos_check decode MODEL CODES.npy OUT.f32 | zonos_check ids MODEL TEXTS.txt OUT.json | zonos_check embed MODEL CLIP OUT.f32"
        ),
    }
    Ok(())
}
