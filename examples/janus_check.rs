//! Dumps for checking Janus-Pro and JanusFlow against the reference (`tools/ref_janus.py` writes
//! the same files):
//!
//! ```text
//! janus_check MODEL text QUESTION OUTDIR          prompt ids + last-token text logits
//! janus_check MODEL gen PROMPT OUTDIR             greedy CFG image tokens (576) + image.png
//! janus_check MODEL forced PROMPT TOKENS OUT.bin  guided logits per step, teacher-forced
//! janus_check MODEL vq TOKENS OUT.bin             VQ decoder output (HWC f32 in [-1, 1])
//! janus_check MODEL vision IMAGE QUESTION OUTDIR  features, embeddings, logits, greedy answer
//! janus_check MODEL chat IMAGE Q1 Q2 OUTDIR       two turns: the greedy answer to Q1 about the
//!                                                 image (answer1.txt), then the conversation's
//!                                                 ids and the logits after Q2
//! janus_check MODEL flow PROMPT NOISE.bin STEPS CFG OUTDIR
//!                                                 JanusFlow from the given noise ([48*48, 4]
//!                                                 f32): the velocities of every step
//!                                                 (velocity.bin: per step, conditional then
//!                                                 unconditional), latents.bin, pixels.bin (the
//!                                                 VAE's output, HWC) and image.png
//! ```
//!
//! `OXINFER_QUANT=q8|q4` quantizes the language model; `OXINFER_DEVICE=cpu|cuda|...` picks the
//! device.

use anyhow::{Context, Result, bail};
use oxinfer::generate::Flow;
use oxinfer::janus::{ChatOptions, Conversation, ImageOptions, Janus, JanusOptions};
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
        bail!("usage: janus_check MODEL text|gen|forced|vq|vision|chat|flow ARGS... (see the source)");
    }
    let mut opts = JanusOptions::default();
    if let Ok(q) = std::env::var("OXINFER_QUANT") {
        opts.quant = oxinfer::model::Quant::parse(&q).context("OXINFER_QUANT")?;
    }
    if let Ok(d) = std::env::var("OXINFER_DEVICE") {
        opts.set("device", &d)?;
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
        "chat" => {
            std::fs::create_dir_all(&a[5])?;
            let img = oxinfer::media::Media::new().load_image(Path::new(&a[2]))?;
            let o = ChatOptions {
                max_tokens: 48,
                ..Default::default()
            };
            let mut conv = Conversation::new();
            let ans = jm.ask(&mut conv, &a[3], &[img], &o, &mut |_| Flow::Continue)?;
            std::fs::write(format!("{}/answer1.txt", a[5]), &ans)?;
            let lg = jm.ask_logits(&mut conv, &a[4], &[], None)?;
            save(format!("{}/logits.bin", a[5]), &lg)?;
            let s: Vec<String> = conv.ids().iter().map(|t| t.to_string()).collect();
            std::fs::write(format!("{}/ids.txt", a[5]), s.join(" "))?;
            println!("{ans}");
        }
        "flow" => {
            let out = &a[6];
            std::fs::create_dir_all(out)?;
            let noise: Vec<f32> = std::fs::read(&a[3])?
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            let o = ImageOptions {
                steps: a[4].parse()?,
                cfg_weight: Some(a[5].parse()?),
                ..Default::default()
            };
            let mut trace = Vec::new();
            let (z, st) = jm.flow_latents(&a[2], &o, Some(&noise), Some(&mut trace), &mut |p| {
                eprint!("\r[step {}/{}]", p.step, p.steps);
                Flow::Continue
            })?;
            eprintln!(
                "\n[{} prompt tokens: prefill {:.2}s, steps {:.2}s]",
                st.prompt_tokens, st.prefill, st.tokens
            );
            save(format!("{out}/velocity.bin"), &trace.concat())?;
            save(format!("{out}/latents.bin"), &z[0])?;
            let (px, _, _) = jm.decode_latents_f32(&z[0])?;
            save(format!("{out}/pixels.bin"), &px)?;
            let img = jm.decode_latents(&z[0])?;
            std::fs::write(
                format!("{out}/image.png"),
                oxinfer::png::encode_rgb(img.width, img.height, &img.rgb),
            )?;
        }
        m => bail!("unknown mode {m}"),
    }
    Ok(())
}
