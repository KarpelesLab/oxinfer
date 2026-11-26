//! Runs the parts of `oxinfer::longcat_avatar` on the inputs written by
//! `tools/longcat_avatar_tiny.py` (tiny random checkpoints built with Meituan's own modules) or
//! `tools/longcat_avatar_ref.py` (the released weights) and compares with the reference outputs:
//!
//! ```text
//! longcat_avatar_check DIR [part ...]
//! ```
//!
//! Parts (all by default, each skipped when its case is not in `DIR/case.json`): `vae`, `dit`,
//! `audio`, `text`, `sched`, `e2e`.

use anyhow::{Context, Result};
use oxinfer::longcat_avatar::audio::{AudioEncoder, clip_window};
use oxinfer::longcat_avatar::dit::{Cond, Dit};
use oxinfer::longcat_avatar::lora::{Active, LoraNet};
use oxinfer::longcat_avatar::pipeline::{self, AvatarParams, AvatarPipeline, Sources, Stage};
use oxinfer::longcat_avatar::vae::VideoVae;
use serde_json::Value;
use std::path::Path;

fn read(p: &Path) -> Result<Vec<f32>> {
    let b = std::fs::read(p).with_context(|| p.display().to_string())?;
    Ok(b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
}

fn cmp(name: &str, a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let max = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let mean = a.iter().zip(b).map(|(x, y)| (x - y).abs() as f64).sum::<f64>() / n.max(1) as f64;
    let scale = b.iter().map(|v| v.abs() as f64).sum::<f64>() / n.max(1) as f64;
    let mse = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum::<f64>() / n.max(1) as f64;
    let peak = b.iter().fold(0f32, |m, v| m.max(v.abs())) as f64;
    println!(
        "{name}: {} values (ref {}), max abs diff {max:.3e}, mean abs diff {mean:.3e}, ref mean |v| {scale:.3e}, PSNR {:.1} dB",
        a.len(),
        b.len(),
        10.0 * (peak * peak / mse.max(1e-30)).log10()
    );
    max
}

/// Frames `[n][h * w * 3]` channels-last from a channel-first `[1, 3, n, h, w]` dump.
fn frames_cl(v: &[f32], n: usize, h: usize, w: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|f| {
            let mut o = vec![0f32; h * w * 3];
            for p in 0..h * w {
                for c in 0..3 {
                    o[p * 3 + c] = v[(c * n + f) * h * w + p];
                }
            }
            o
        })
        .collect()
}

fn frames_cf(fr: &[Vec<f32>], h: usize, w: usize) -> Vec<f32> {
    let n = fr.len();
    let mut o = vec![0f32; 3 * n * h * w];
    for (f, x) in fr.iter().enumerate() {
        for p in 0..h * w {
            for c in 0..3 {
                o[(c * n + f) * h * w + p] = x[p * 3 + c];
            }
        }
    }
    o
}

fn dims(v: &Value) -> Vec<usize> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect()
}

fn check_vae(dir: &Path, case: &Value, base: &Path) -> Result<()> {
    let vae = VideoVae::load(&base.join("vae"), true)?;
    for name in ["vae_enc_video", "vae_enc_image"] {
        if case[name].is_null() {
            continue;
        }
        let s = dims(&case[name]);
        let x = read(&dir.join(format!("{name}.bin")))?;
        let lat = vae.encode(&frames_cl(&x, s[2], s[3], s[4]), s[3], s[4])?;
        cmp(name, &lat, &read(&dir.join(format!("{name}_mode.bin")))?);
        // (The reference with float32 activations on its bf16-rounded weights, when the case has it.)
        if let Ok(r) = read(&dir.join(format!("{name}_mode_f32.bin"))) {
            cmp(&format!("{name} (float32 reference)"), &lat, &r);
        }
    }
    if !case["vae_dec_latent"].is_null() {
        let s = dims(&case["vae_dec_latent"]);
        let z = read(&dir.join("vae_dec_latent.bin"))?;
        let fr = vae.decode(&z, s[2], s[3], s[4], &mut |_, _| {})?;
        let o = dims(&case["vae_dec_video"]);
        let got = frames_cf(&fr, o[3], o[4]);
        cmp("vae_dec", &got, &read(&dir.join("vae_dec_video.bin"))?);
        if let Ok(r) = read(&dir.join("vae_dec_video_f32.bin")) {
            cmp("vae_dec (float32 reference)", &got, &r);
        }
    }
    Ok(())
}

fn check_dit(dir: &Path, case: &Value, avatar: &Path) -> Result<()> {
    let la = avatar.join("loraA/lora.safetensors");
    let loras = if la.exists() {
        Some((LoraNet::load(&la)?, LoraNet::load(&avatar.join("loraB/lora.safetensors"))?))
    } else {
        None
    };
    for (sub, int8) in [("base_model", false), ("base_model_int8", true)] {
        if !avatar.join(sub).join("config.json").exists() {
            continue;
        }
        let mut dit: Option<Dit> = None;
        for (name, mix) in [
            ("dit_t2v", None),
            ("dit_i2v", None),
            ("dit_avc", None),
            ("dit_avc_mask", None),
            ("dit_multi", None),
            ("dit_lora_mix", Some((0.3f32, 0.7f32))),
            ("dit_lora_b", Some((0.0, 1.0))),
            ("dit_lora_a", Some((1.0, 0.0))),
        ] {
            let name = if int8 { format!("{name}_int8") } else { name.to_string() };
            let c = &case[&name];
            if c.is_null() {
                continue;
            }
            if dit.is_none() {
                dit = Some(Dit::load(&avatar.join(sub))?);
            }
            let dit = dit.as_ref().unwrap();
            let xs = dims(&c["x"]);
            let x = read(&dir.join(format!("{name}_x.bin")))?;
            let text = read(&dir.join(format!("{name}_text.bin")))?;
            let mask: Vec<u64> = c["mask"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
            let valid = mask.iter().filter(|&&m| m != 0).count();
            let audio = read(&dir.join(format!("{name}_audio.bin")))?;
            let ts: Vec<f32> = c["t"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
            let kw = &c["kw"];
            let u = |k: &str| kw[k].as_u64().unwrap_or(0) as usize;
            let masks = if kw.get("ref_target_masks").is_some() {
                let m = read(&dir.join("dit_multi_masks.bin"))?;
                let s = dims(&case["dit_multi_masks"]);
                Some((m, s[0], s[1], s[2]))
            } else {
                None
            };
            let nets: Vec<(&LoraNet, f32)> = match (mix, &loras) {
                (Some((a, b)), Some((la, lb))) => vec![(la, a), (lb, b)],
                _ => Vec::new(),
            };
            let cc = dit.cfg.caption_channels;
            let t0 = std::time::Instant::now();
            let y = dit.forward(
                &x,
                (xs[2], xs[3], xs[4]),
                &ts,
                &Cond {
                    text: &text[..valid * cc],
                    text_len: mask.len(),
                    audio: &audio,
                    humans: dims(&c["audio"])[0],
                    num_cond: u("num_cond_latents"),
                    num_ref: u("num_ref_latents"),
                    ref_img_index: u("ref_img_index"),
                    mask_frame_range: u("mask_frame_range"),
                    ref_masks: masks.as_ref().map(|(m, n, h, w)| (&m[..], *n, *h, *w)),
                    lora: Active { nets: &nets },
                },
            )?;
            let secs = t0.elapsed().as_secs_f64();
            cmp(&name, &y, &read(&dir.join(format!("{name}_out.bin")))?);
            if let Ok(r) = read(&dir.join(format!("{name}_out_f32.bin"))) {
                cmp(&format!("{name} (float32 reference)"), &y, &r);
            }
            if secs > 1.0 {
                println!("  ({secs:.1} s)");
            }
        }
    }
    Ok(())
}

fn check_audio(dir: &Path, case: &Value, avatar: &Path) -> Result<()> {
    if case["audio_emb"].is_null() {
        return Ok(());
    }
    let enc = AudioEncoder::load(&avatar.join("whisper-large-v3"))?;
    let speech = read(&dir.join("speech.bin"))?;
    let t0 = std::time::Instant::now();
    let (emb, rows) = enc.embed(&speech, pipeline::FPS, &mut |_, _| {})?;
    println!("audio: {rows} rows in {:.1} s", t0.elapsed().as_secs_f64());
    cmp("audio_emb", &emb, &read(&dir.join("audio_emb.bin"))?);
    Ok(())
}

fn check_text(dir: &Path, case: &Value, base: &Path) -> Result<()> {
    let Some(prompts) = case["prompts"].as_array() else { return Ok(()) };
    let enc = oxinfer::flux::t5::T5Encoder::load(&base.join("text_encoder"), &base.join("tokenizer"), oxinfer::model::Quant::Bf16)?;
    for (i, p) in prompts.iter().enumerate() {
        let p = p.as_str().unwrap();
        let ids = enc.tokenize(&pipeline::clean_prompt(p), 512)?;
        let want: Vec<u32> = case[format!("prompt{i}_ids")]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let n = ids.iter().rposition(|&t| t != enc.pad_id).map_or(0, |q| q + 1);
        println!(
            "prompt {i}: {} tokens, ids {}",
            n,
            if ids[..n] == want[..] { "match" } else { "DIFFER" }
        );
        if ids[..n] != want[..] {
            println!("  got  {:?}\n  want {:?}", &ids[..n], want);
        }
        let t0 = std::time::Instant::now();
        let e = pipeline::encode_text(&enc, p, 512)?;
        let r = read(&dir.join(format!("prompt{i}_embeds.bin")))?;
        cmp(
            &format!("prompt{i}_embeds ({:.1} s)", t0.elapsed().as_secs_f64()),
            &e,
            &r[..e.len().min(r.len())],
        );
        if let Ok(r) = read(&dir.join(format!("prompt{i}_embeds_f32.bin"))) {
            cmp(&format!("prompt{i}_embeds (float32 reference)"), &e, &r[..e.len().min(r.len())]);
        }
    }
    Ok(())
}

fn check_sched(case: &Value) {
    let Some(want) = case["sched_sigmas"].as_array() else { return };
    let want: Vec<f32> = want.iter().map(|v| v.as_f64().unwrap() as f32).collect();
    cmp("sigmas (distilled, shift 7)", &pipeline::sigmas(8, true, 7.0), &want);
}

fn check_e2e(dir: &Path, case: &Value, avatar: &Path, base: &Path) -> Result<()> {
    if case["e2e_noise"].is_null() && case["e2e_noise_int8"].is_null() {
        return Ok(());
    }
    let prompts = case["prompts"].as_array().context("no prompts")?;
    let prompt = prompts[0].as_str().unwrap();
    let img = oxinfer::minimax_h3::encode::load_image(&dir.join("ref.png"))?;
    let speech = read(&dir.join("speech.bin"))?;
    for (tag, int8) in [("", false), ("_int8", true)] {
        if case[format!("e2e_noise{tag}")].is_null() {
            continue;
        }
        let pipe = AvatarPipeline::load(&Sources {
            avatar: avatar.to_path_buf(),
            base: base.to_path_buf(),
            int8,
            text_quant: oxinfer::model::Quant::Bf16,
        })?;
        let ns = dims(&case[format!("e2e_noise{tag}")]);
        let frames = 1 + 4 * (ns[2] - 1);
        let p = AvatarParams {
            prompt: prompt.to_string(),
            image: Some(oxinfer::sd::Image {
                width: img.width,
                height: img.height,
                rgb: img.rgb.clone(),
            }),
            speech: vec![speech.clone()],
            frames,
            ..Default::default()
        };
        let (w, h) = pipe.canvas(&p)?;
        let shape = pipe.latent_shape(w, h, frames);
        let text = pipeline::encode_text(&pipe.text_encoder()?, prompt, 512)?;
        let (emb, rows) = pipe.audio_encoder()?.embed(&speech, pipeline::FPS, &mut |_, _| {})?;
        let image = pipeline::preprocess_image(p.image.as_ref().unwrap(), w, h);
        let mut prog = |s: Stage| {
            if let Stage::Step { step, steps, secs, .. } = s {
                eprintln!("  step {step}/{steps} {secs:.1}s");
            }
        };
        let noise = read(&dir.join(format!("e2e_noise{tag}.bin")))?;
        let lat = pipe.first_clip(
            &p,
            &text,
            None,
            (w, h),
            Some(&image),
            &clip_window(&emb, rows, 0, frames, 1, 5),
            None,
            noise.clone(),
            &mut prog,
        )?;
        cmp(&format!("e2e_latent{tag}"), &lat, &read(&dir.join(format!("e2e_latent{tag}.bin")))?);
        let fr = pipe.decode(&lat, shape, 1, &mut prog)?;
        let got: Vec<f32> = fr.iter().flat_map(|f| f.iter().map(|&v| v as f32 / 255.0)).collect();
        let want = read(&dir.join(format!("e2e_frames{tag}.bin")))?;
        let want8: Vec<f32> = want.iter().map(|&v| ((v * 255.0) as u8) as f32 / 255.0).collect();
        cmp(&format!("e2e_frames{tag} (8-bit)"), &got, &want8);
        if !case[format!("avc_noise{tag}")].is_null() {
            let n2 = read(&dir.join(format!("avc_noise{tag}.bin")))?;
            let frame = shape.1 * shape.2;
            let reference: Vec<f32> = (0..pipe.dit.cfg.in_channels)
                .flat_map(|ch| lat[ch * shape.0 * frame..ch * shape.0 * frame + frame].to_vec())
                .collect();
            let audio = clip_window(&emb, rows, frames - pipeline::COND_FRAMES, frames, 1, 5);
            let lat2 = pipe.next_clip(&p, &text, None, (w, h), &fr, &reference, &audio, None, n2, 2, &mut prog)?;
            cmp(
                &format!("avc_latent{tag}"),
                &lat2,
                &read(&dir.join(format!("avc_latent{tag}.bin")))?,
            );
            let fr2 = pipe.decode(&lat2, shape, 2, &mut prog)?;
            let got: Vec<f32> = fr2.iter().flat_map(|f| f.iter().map(|&v| v as f32 / 255.0)).collect();
            let want = read(&dir.join(format!("avc_frames{tag}.bin")))?;
            let want8: Vec<f32> = want.iter().map(|&v| ((v * 255.0) as u8) as f32 / 255.0).collect();
            cmp(&format!("avc_frames{tag} (8-bit)"), &got, &want8);
        }
        if tag.is_empty() && !case["cfg_latent"].is_null() {
            let neg = pipeline::encode_text(&pipe.text_encoder()?, "blurry", 512)?;
            let pg = AvatarParams {
                steps: 3,
                distill: false,
                text_guidance: 4.0,
                audio_guidance: 4.0,
                ..p.clone()
            };
            let lat3 = pipe.first_clip(
                &pg,
                &text,
                Some(&neg),
                (w, h),
                Some(&image),
                &clip_window(&emb, rows, 0, frames, 1, 5),
                None,
                noise,
                &mut prog,
            )?;
            cmp("cfg_latent", &lat3, &read(&dir.join("cfg_latent.bin"))?);
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = Path::new(args.first().context("usage: longcat_avatar_check DIR [part ...]")?);
    let parts: Vec<&str> = args[1..].iter().map(String::as_str).filter(|a| !a.starts_with("--")).collect();
    // `--device=cuda` (default: cpu).
    let device = args.iter().find_map(|a| a.strip_prefix("--device=")).unwrap_or("cpu");
    let pref = oxinfer::platform::DevicePref::parse(device).context("--device=cpu|cuda")?;
    let gpu = oxinfer::gpu::init(&oxinfer::platform::detect(), pref)?;
    println!("[device: {}]", gpu.map_or("cpu", |g| g.name()));
    let on = |p: &str| parts.is_empty() || parts.contains(&p);
    let case: Value = serde_json::from_slice(&std::fs::read(dir.join("case.json"))?)?;
    // `--avatar=DIR` and `--base=DIR`: the released weights (default: the tiny folders in DIR).
    let opt = |k: &str| args.iter().find_map(|a| a.strip_prefix(k)).map(std::path::PathBuf::from);
    let avatar = opt("--avatar=").unwrap_or_else(|| dir.join("LongCat-Video-Avatar-1.6"));
    let base = opt("--base=").unwrap_or_else(|| dir.join("LongCat-Video"));
    // `bench-vae` (only when named): the time to decode 5 latent frames of 64 x 96 (17 frames of
    // 768 x 512) and to encode 13 of them back.
    if parts.contains(&"bench-vae") {
        let vae = VideoVae::load(&base.join("vae"), true)?;
        let (t, h, w) = (5, 64, 96);
        let z: Vec<f32> = (0..vae.cfg.z_dim * t * h * w)
            .map(|i| ((i * 2654435761usize) % 1000) as f32 / 500.0 - 1.0)
            .collect();
        let t0 = std::time::Instant::now();
        let fr = vae.decode(&z, t, h, w, &mut |_, _| {})?;
        println!(
            "VAE decode: {} frames of {}x{} in {:.1} s",
            fr.len(),
            w * 8,
            h * 8,
            t0.elapsed().as_secs_f64()
        );
        print!("{}", oxinfer::sd::ops::timing::report());
        let t0 = std::time::Instant::now();
        let lat = vae.encode(&fr[..13], h * 8, w * 8)?;
        println!(
            "VAE encode: 13 frames in {:.1} s ({} values)",
            t0.elapsed().as_secs_f64(),
            lat.len()
        );
        return Ok(());
    }
    if on("sched") {
        check_sched(&case);
    }
    if on("vae") {
        check_vae(dir, &case, &base)?;
    }
    if on("dit") {
        check_dit(dir, &case, &avatar)?;
    }
    if on("audio") {
        check_audio(dir, &case, &avatar)?;
    }
    if on("text") {
        check_text(dir, &case, &base)?;
    }
    if on("e2e") {
        check_e2e(dir, &case, &avatar, &base)?;
    }
    Ok(())
}
