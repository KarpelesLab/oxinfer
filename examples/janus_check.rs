//! Dumps for checking Janus-Pro against the reference (tools/ref_janus.py writes the same files):
//!
//! ```text
//! janus_check MODEL text QUESTION OUTDIR          prompt ids + last-token text logits
//! janus_check MODEL gen PROMPT OUTDIR             greedy CFG image tokens (576) + image.png
//! janus_check MODEL forced PROMPT TOKENS OUT.bin  guided logits per step, teacher-forced
//! janus_check MODEL vq TOKENS OUT.bin             VQ decoder output (HWC f32 in [-1, 1])
//! janus_check MODEL vision IMAGE QUESTION OUTDIR  features, embeddings, logits, greedy answer
//! ```
//!
//! `OXINFER_QUANT=q8|q4` quantizes the language model.

use anyhow::{Context, Result, bail};
use oxinfer::generate::Flow;
use oxinfer::janus::{ChatOptions, ImageOptions, Janus, JanusOptions};
use std::path::Path;

fn save(path: impl AsRef<Path>, v: &[f32]) -> Result<()> {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(path.as_ref(), b).with_context(|| format!("write {}", path.as_ref().display()))
}

fn tokens(path: &str) -> Result<Vec<u32>> {
    Ok(std::fs::read_to_string(path)?
        .split_whitespace()
        .map(|t| t.parse())
        .collect::<Result<_, _>>()?)
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 3 {
        bail!("usage: janus_check MODEL text|gen|forced|vq|vision ARGS... (see the source)");
    }
    let mut opts = JanusOptions::default();
    if let Ok(q) = std::env::var("OXINFER_QUANT") {
        opts.quant = oxinfer::model::Quant::parse(&q).context("OXINFER_QUANT")?;
    }
    let jm = Janus::load(&a[0], &opts)?;
    eprintln!("[loaded in {:.1}s]", jm.load_secs);
    match a[1].as_str() {
        "text" => {
            std::fs::create_dir_all(&a[3])?;
            let lg = jm.understanding_logits(&a[2], &[], None)?;
            save(format!("{}/logits.bin", a[3]), &lg)?;
        }
        "gen" => {
            std::fs::create_dir_all(&a[3])?;
            let o = ImageOptions {
                temperature: 0.0,
                ..Default::default()
            };
            let (grids, st) = jm.generate_tokens(&a[2], &o, &mut |_| Flow::Continue)?;
            let img = jm.decode_tokens(&grids[0])?;
            eprintln!(
                "[{} prompt tokens: prefill {:.2}s, tokens {:.2}s ({:.1} ms/step)]",
                st.prompt_tokens,
                st.prefill,
                st.tokens,
                st.tokens * 1e3 / st.steps as f64
            );
            let s: Vec<String> = grids[0].iter().map(|t| t.to_string()).collect();
            std::fs::write(format!("{}/tokens.txt", a[3]), s.join(" "))?;
            std::fs::write(
                format!("{}/image.png", a[3]),
                oxinfer::png::encode_rgb(img.width, img.height, &img.rgb),
            )?;
        }
        "forced" => {
            let lg = jm.image_logits(&a[2], 5.0, &tokens(&a[3])?)?;
            save(&a[4], &lg.concat())?;
        }
        "vq" => {
            let t = tokens(&a[2])?;
            let (rgb, _, _) = jm.decode_tokens_f32(&t)?;
            save(&a[3], &rgb)?;
            let img = jm.decode_tokens(&t)?;
            std::fs::write(format!("{}.png", a[3]), oxinfer::png::encode_rgb(img.width, img.height, &img.rgb))?;
        }
        "vision" => {
            std::fs::create_dir_all(&a[4])?;
            let img = oxinfer::media::Media::new().load_image(Path::new(&a[2]))?;
            let (emb, feats) = jm.encode_image(&img)?;
            save(format!("{}/features.bin", a[4]), &feats)?;
            save(format!("{}/embeds.bin", a[4]), &emb)?;
            let lg = jm.understanding_logits(&a[3], std::slice::from_ref(&img), None)?;
            save(format!("{}/logits.bin", a[4]), &lg)?;
            let o = ChatOptions {
                max_tokens: 64,
                ..Default::default()
            };
            let ans = jm.understand(&a[3], &[img], &o, &mut |_| Flow::Continue)?;
            std::fs::write(format!("{}/answer.txt", a[4]), &ans)?;
            println!("{ans}");
        }
        m => bail!("unknown mode {m}"),
    }
    Ok(())
}
