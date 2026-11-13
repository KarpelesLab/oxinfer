//! Compares oxinfer's Audio8 TTS (`oxinfer::arktts`) with the outputs `tools/ref_arktts.py`
//! wrote from the checkpoint's own Python code:
//!
//! ```text
//! arktts_check MODEL_DIR REF_DIR [--no-encode]
//! arktts_check MODEL_DIR REF_DIR --ids
//! ```
//!
//! `--ids` compares only the token ids of the texts of `REF_DIR/ids.json`. Otherwise,
//! Stage by stage: the ids of the prompt's two halves, the RoPE table, the codes of the
//! reference clip (the codec's encoder), the prompt rows, the greedy generation (codes and the
//! logits of every step, from the reference's prompt), and the codec's waveform for the
//! reference's codes (whole, in windows as `decode` does, and piece by piece as streaming does).
//! Exits with an error when ids, prompt rows or generated codes differ.

use anyhow::{Context, Result, bail};
use oxinfer::arktts::{Arktts, ArkttsOptions, ArkttsSpeak, Reference};
use oxinfer::engine::Flow;
use std::path::Path;
use std::time::Instant;

fn f32s(p: &Path) -> Result<Vec<f32>> {
    let b = std::fs::read(p).with_context(|| format!("read {}", p.display()))?;
    Ok(b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
}

fn u32s(p: &Path) -> Result<Vec<u32>> {
    let b = std::fs::read(p).with_context(|| format!("read {}", p.display()))?;
    Ok(b.as_chunks::<4>().0.iter().map(|c| i32::from_le_bytes(*c) as u32).collect())
}

fn ids(v: &serde_json::Value) -> Vec<u32> {
    v.as_array()
        .map(|a| a.iter().filter_map(|x| x.as_u64().map(|x| x as u32)).collect())
        .unwrap_or_default()
}

/// The largest difference and the signal-to-noise ratio in dB of `a` against `b`.
fn compare(a: &[f32], b: &[f32]) -> (f32, f64) {
    let max = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let sig: f64 = b.iter().map(|&y| (y as f64) * (y as f64)).sum();
    let noise: f64 = a.iter().zip(b).map(|(&x, &y)| ((x - y) as f64).powi(2)).sum();
    (max, 10.0 * (sig / noise.max(1e-300)).log10())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        bail!("usage: arktts_check MODEL_DIR REF_DIR [--no-encode]");
    }
    let dir = Path::new(&args[1]);
    let m = Arktts::load(&args[0], &ArkttsOptions::default())?;
    if args.iter().any(|a| a == "--ids") {
        let rows: Vec<serde_json::Value> = serde_json::from_slice(&std::fs::read(dir.join("ids.json"))?)?;
        let mut bad = 0;
        for r in &rows {
            let text = r["text"].as_str().context("ids.json: text")?;
            let same = m.prompt_ids(text, None)?.0 == ids(&r["ids"]) && m.prompt_ids("x", Some(text))?.0 == ids(&r["reference_prefix"]);
            if !same {
                bad += 1;
                println!("DIFFERENT: {text:?}");
            }
        }
        println!("token ids: {} texts, {} identical", rows.len(), rows.len() - bad);
        if bad > 0 {
            bail!("differences from the reference");
        }
        return Ok(());
    }
    let man: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("manifest.json"))?)?;
    let c = &m.lm().cfg;
    let (n, cs) = (c.num_codebooks, c.codebook_size);
    let text = man["text"].as_str().context("manifest: text")?;
    let ref_text = man["ref_text"].as_str();
    let mut failed = false;

    // 1. Token ids.
    let (prefix, suffix) = m.prompt_ids(text, ref_text)?;
    let same = prefix == ids(&man["prefix_ids"]) && suffix == ids(&man["suffix_ids"]);
    println!(
        "token ids: {} + {} ids, {}",
        prefix.len(),
        suffix.len(),
        if same { "identical" } else { "DIFFERENT" }
    );
    failed |= !same;

    // 2. The RoPE table.
    let rope = f32s(&dir.join("rope.f32"))?;
    let diff = m.lm().rope().iter().zip(&rope).filter(|(a, b)| a != b).count();
    println!(
        "rope table: {} values, {diff} different{}",
        rope.len(),
        if rope.len() == m.lm().rope().len() {
            ""
        } else {
            " (LENGTHS DIFFER)"
        }
    );

    // 3. The reference clip's codes.
    let mut reference = None;
    if let Some(rt) = ref_text {
        let want = u32s(&dir.join("ref_codes.i32"))?;
        let frames = want.len() / n;
        if !args.iter().any(|a| a == "--no-encode") {
            let audio = f32s(&dir.join("ref_audio.f32"))?;
            let t0 = Instant::now();
            let (got, got_frames) = m.codec().encode(&audio)?;
            let secs = t0.elapsed().as_secs_f64();
            if got_frames != frames {
                println!("encoder: {got_frames} frames, the reference has {frames}: DIFFERENT");
                failed = true;
            } else {
                let per: Vec<usize> = (0..n)
                    .map(|j| (0..frames).filter(|f| got[f * n + j] == want[f * n + j]).count())
                    .collect();
                let all: usize = per.iter().sum();
                println!(
                    "encoder: {frames} frames of {:.2} s in {secs:.2} s, {all} of {} codes identical ({:.2}%); per codebook {per:?}",
                    audio.len() as f64 / m.codec().sample_rate as f64,
                    frames * n,
                    100.0 * all as f64 / (frames * n) as f64
                );
            }
        }
        reference = Some(Reference {
            text: rt.to_string(),
            codes: want,
            frames,
        });
    }

    // 4. The prompt rows.
    let prompt = m.prompt(text, reference.as_ref())?;
    let want = u32s(&dir.join("prompt.i32"))?;
    println!(
        "prompt: {} rows, {}",
        prompt.len() / (n + 1),
        if prompt == want { "identical" } else { "DIFFERENT" }
    );
    failed |= prompt != want;

    // 5. Greedy generation from the reference's prompt.
    let want_codes = u32s(&dir.join("codes.i32"))?;
    let frames = want_codes.len() / n;
    let mut o = ArkttsSpeak::default();
    o.temperature = Some(0.0);
    o.max_tokens = Some(frames + usize::from(man["finished"].as_bool().unwrap_or(false)));
    let t0 = Instant::now();
    let g = m.generate(&want, &o, true, &mut |_| Flow::Continue)?;
    let secs = t0.elapsed().as_secs_f64();
    let first = (0..g.frames.min(frames)).find(|&f| g.codes[f * n..(f + 1) * n] != want_codes[f * n..(f + 1) * n]);
    let same = g.codes == want_codes;
    println!(
        "greedy generation: {} frames in {secs:.2} s ({:.1} frames/s; the reference took {:.2} s), {:?}; the reference has {frames}: {}",
        g.frames,
        g.frames as f64 / secs,
        man["generate_secs"].as_f64().unwrap_or(0.0),
        g.finish,
        match (same, first) {
            (true, _) => "identical".to_string(),
            (false, Some(f)) => format!("DIFFERENT from frame {f}"),
            (false, None) => "DIFFERENT in length".to_string(),
        }
    );
    failed |= !same;
    let slow = f32s(&dir.join("slow.f32"))?;
    let fast = f32s(&dir.join("fast.f32"))?;
    let upto = first.unwrap_or(g.frames.min(frames));
    let (mut ds, mut df) = (0f32, 0f32);
    for (i, l) in g.logits.iter().enumerate().take(upto + 1) {
        if let Some(w) = slow.get(i * (cs + 1)..(i + 1) * (cs + 1)) {
            ds = ds.max(compare(&l[..cs + 1], w).0);
        }
        if l.len() > cs + 1
            && i < upto
            && let Some(w) = fast.get(i * (n - 1) * cs..(i + 1) * (n - 1) * cs)
        {
            df = df.max(compare(&l[cs + 1..], w).0);
        }
    }
    println!("logits over the identical steps: slow within {ds:.3e}, fast within {df:.3e}");

    // 6. The codec's waveform for the reference's codes.
    let wave = f32s(&dir.join("audio.f32"))?;
    let t0 = Instant::now();
    let whole = m.codec().decode_windows(&want_codes, frames, 0, 0)?;
    let t_whole = t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    let windows = m.codec().decode(&want_codes, frames)?;
    let t_win = t0.elapsed().as_secs_f64();
    if whole.len() != wave.len() {
        println!("decoder: {} samples, the reference has {}: DIFFERENT", whole.len(), wave.len());
        failed = true;
    } else {
        let (max, snr) = compare(&whole, &wave);
        println!(
            "decoder: {} samples ({:.2} s) in {t_whole:.2} s (the reference took {:.2} s); largest difference {max:.3e}, SNR {snr:.1} dB",
            wave.len(),
            wave.len() as f64 / m.codec().sample_rate as f64,
            man["decode_secs"].as_f64().unwrap_or(0.0)
        );
        let (max, snr) = compare(&windows, &wave);
        let (maxw, _) = compare(&windows, &whole);
        println!(
            "decoder in windows: {t_win:.2} s; against the reference {max:.3e}, SNR {snr:.1} dB; against the whole decoding {maxw:.3e}"
        );
        let mut pieces = Vec::new();
        let mut f = 0;
        while f < frames {
            let upto = (f + 32).min(frames);
            pieces.extend(m.codec().decode_tail(&want_codes[..upto * n], upto, f)?);
            f = upto;
        }
        let (max, snr) = compare(&pieces, &wave);
        println!("decoder piece by piece (32 frames): against the reference {max:.3e}, SNR {snr:.1} dB");
    }
    if failed {
        bail!("differences from the reference");
    }
    Ok(())
}
