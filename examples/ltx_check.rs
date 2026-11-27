//! Checks oxinfer's LTX-2.5 components against reference dumps (the scripts of the
//! verification notes in `docs/ltx.md`): one transformer forward over its first blocks, the text
//! connectors, the video VAE, the audio decoder and vocoder, the latent upsampler.
//!
//! ```text
//! ltx_check dit FILE REF_DIR        # ref_dit.py's dump (FILE: a transformer file)
//! ltx_check connector FILE REF_DIR  # ref_connector.py
//! ltx_check vae FILE REF_DIR        # ref_vae.py (FILE: the video VAE)
//! ltx_check diffvae FILE REF_DIR    # tools/ltx_ref_diffvae.py (FILE: the DiffVAE)
//! ltx_check diffvae_time FILE T H W # a timed decode of random latents (either VAE)
//! ltx_check audio FILE REF_DIR      # ref_audio.py (FILE: the audio VAE)
//! ltx_check upsample FILE VAE REF_DIR  # ref_upsample.py
//! ltx_check text FILE REF_DIR       # ref_text.py (FILE: the text encoder)
//! ltx_check guider - REF_DIR        # ref_guider.py (guidance arithmetic, the dev schedule)
//! ltx_check crf IMAGE QP OUT_DIR    # the H.264 round trip of a conditioning image
//! ltx_check vaebench FILE T H W DIR # random latents decoded (timed); DIR/px_ref.f32: compared
//! ltx_check audiobench FILE N DIR   # audio decoder and vocoder on N random rows (timed)
//! ```

use anyhow::{Context, Result, bail};
use oxinfer::ltx;
use oxinfer::safetensors::ModelFiles;
use std::path::Path;

fn read_f32(p: &Path) -> Result<Vec<f32>> {
    let b = std::fs::read(p).with_context(|| format!("read {}", p.display()))?;
    Ok(b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
}

/// Relative L2 error, max abs error and the reference's RMS.
fn compare(name: &str, ours: &[f32], want: &[f32]) {
    if ours.len() != want.len() {
        println!("{name}: length {} vs {}", ours.len(), want.len());
        return;
    }
    let (mut d2, mut w2, mut mx) = (0f64, 0f64, 0f64);
    for (&a, &b) in ours.iter().zip(want) {
        let d = (a - b) as f64;
        d2 += d * d;
        w2 += (b as f64) * (b as f64);
        mx = mx.max(d.abs());
    }
    println!(
        "{name}: rel L2 {:.3e}, max abs {:.3e}, ref rms {:.3e}",
        (d2 / w2.max(1e-30)).sqrt(),
        mx,
        (w2 / want.len() as f64).sqrt()
    );
}

fn meta(dir: &Path) -> Result<serde_json::Value> {
    Ok(serde_json::from_slice(&std::fs::read(dir.join("meta.json"))?)?)
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        bail!("usage: ltx_check dit|connector|vae|vaebench|audio|audiobench|upsample|text FILE [ARGS] DIR");
    }
    let file = Path::new(&a[2]);
    let dir = Path::new(a.last().unwrap());
    // LTX_DEVICE=metal|auto runs on the accelerator (the transformer's whole-pass session).
    let dev = std::env::var("LTX_DEVICE").unwrap_or_else(|_| "cpu".into());
    let pref = oxinfer::platform::DevicePref::parse(&dev).context("LTX_DEVICE: cpu, auto, metal")?;
    oxinfer::gpu::init(&oxinfer::platform::detect(), pref)?;
    let r = |n: &str| read_f32(&dir.join(format!("{n}.f32")));
    match a[1].as_str() {
        "dit" => {
            let m = meta(dir)?;
            let u = |k: &str| m[k].as_u64().unwrap_or(0) as usize;
            let (lt, lh, lw) = (u("lt"), u("lh"), u("lw"));
            let fps = m["fps"].as_f64().unwrap_or(24.0) as f32;
            let sigma = m["sigma"].as_f64().unwrap_or(1.0) as f32;
            let files = ModelFiles::open_file(file, false)?;
            let mut dit = ltx::dit::Dit::load(&files)?;
            dit.truncate_blocks(u("blocks"));
            let pos = ltx::video_positions(lt, lh, lw, fps);
            compare("video positions", &pos, &r("vpos")?);
            let apos = ltx::audio_positions(u("na"));
            compare("audio positions", &apos, &r("apos")?);
            let (vx, ax, vctx, actx, vt, at) = (r("vx")?, r("ax")?, r("vctx")?, r("actx")?, r("vt")?, r("at")?);
            let t0 = std::time::Instant::now();
            let (vo, ao) = dit.forward(
                &ltx::dit::StreamIn {
                    x: &vx,
                    t: &vt,
                    sigma,
                    pos: &pos,
                    ctx: &vctx,
                    keyframe_rows: lh * lw,
                    slot_rows: 0..0,
                },
                &ltx::dit::StreamIn {
                    x: &ax,
                    t: &at,
                    sigma,
                    pos: &apos,
                    ctx: &actx,
                    keyframe_rows: 0,
                    slot_rows: 0..0,
                },
            );
            println!("forward {:.2}s", t0.elapsed().as_secs_f64());
            compare("video velocity", &vo, &r("vout")?);
            compare("audio velocity", &ao, &r("aout")?);
            // LTX_SAVE=DIR: this run's velocities as DIR/vout.f32, DIR/aout.f32 (another run's
            // reference: CPU against Metal on the same file).
            if let Ok(out) = std::env::var("LTX_SAVE") {
                let out = Path::new(&out);
                std::fs::create_dir_all(out)?;
                for (n, v) in [("vout", &vo), ("aout", &ao)] {
                    std::fs::write(
                        out.join(format!("{n}.f32")),
                        v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(),
                    )?;
                }
            }
            // The guidance passes (ref_dit_ptb.py): STG (video self-attention skipped in block 1,
            // audio in block 0) and the isolated-modality pass (no audio-video cross-attention).
            if dir.join("vout_ptb.f32").exists() {
                fn si<'s>(
                    x: &'s [f32],
                    t: &'s [f32],
                    sigma: f32,
                    pos: &'s [f32],
                    ctx: &'s [f32],
                    keyframe_rows: usize,
                ) -> ltx::dit::StreamIn<'s> {
                    ltx::dit::StreamIn {
                        x,
                        t,
                        sigma,
                        pos,
                        ctx,
                        keyframe_rows,
                        slot_rows: 0..0,
                    }
                }
                use ltx::dit::{Blocks, Perturb};
                let ptb = Perturb {
                    video_self: Blocks::List(vec![1]),
                    audio_self: Blocks::List(vec![0]),
                    cross: false,
                };
                let (vo, ao) = dit.forward_with(
                    &si(&vx, &vt, sigma, &pos, &vctx, lh * lw),
                    &si(&ax, &at, sigma, &apos, &actx, 0),
                    &ptb,
                );
                compare("video velocity, STG pass", &vo, &r("vout_ptb")?);
                compare("audio velocity, STG pass", &ao, &r("aout_ptb")?);
                let iso = Perturb {
                    cross: true,
                    ..Perturb::default()
                };
                let (vo, ao) = dit.forward_with(
                    &si(&vx, &vt, sigma, &pos, &vctx, lh * lw),
                    &si(&ax, &at, sigma, &apos, &actx, 0),
                    &iso,
                );
                compare("video velocity, isolated pass", &vo, &r("vout_mod")?);
                compare("audio velocity, isolated pass", &ao, &r("aout_mod")?);
            }
        }
        "crf" => {
            // ltx_check crf IMAGE QP OUT_DIR: the image cropped to even sides, through H.264 at
            // constant QP and back (what LTX-2.5 conditions on), as raw RGB `crf.rgb` (and the
            // cropped original `orig.rgb`), for comparison with x264's CRF encode.
            #[cfg(feature = "media")]
            {
                let img = oxinfer::minimax_h3::encode::load_image(file)?;
                let qp: u32 = a[3].parse()?;
                let (w, h) = (img.width / 2 * 2, img.height / 2 * 2);
                let rgb: Vec<u8> = (0..h).flat_map(|y| img.rgb[y * img.width * 3..][..w * 3].to_vec()).collect();
                let t0 = std::time::Instant::now();
                let out = oxinfer::minimax_h3::encode::h264_roundtrip(&rgb, w, h, qp)?;
                println!("{w}x{h} at QP {qp}: {:.2}s", t0.elapsed().as_secs_f64());
                std::fs::create_dir_all(dir)?;
                std::fs::write(dir.join("crf.rgb"), &out)?;
                std::fs::write(dir.join("orig.rgb"), &rgb)?;
                std::fs::write(dir.join("meta.json"), serde_json::json!({"w": w, "h": h}).to_string())?;
            }
            #[cfg(not(feature = "media"))]
            bail!("built without the media feature");
        }
        "guider" => {
            // ref_guider.py: MultiModalGuider.calculate with the release's video / audio params,
            // and LTX2Scheduler's 30-step schedule (FILE is unused).
            let m = meta(dir)?;
            let w = m["width"].as_u64().unwrap_or(128) as usize;
            let (cond, un, ptb, md, mask) = (r("cond")?, r("uncond")?, r("ptb")?, r("mod")?, r("mask")?);
            for (name, g, want) in [
                ("video guidance", ltx::GuiderParams::video(), "gv"),
                ("audio guidance", ltx::GuiderParams::audio(), "ga"),
            ] {
                let out = g.calculate(&cond, Some(&un), Some(&ptb), Some(&md), &mask, w);
                compare(name, &out, &r(want)?);
            }
            compare("LTX2Scheduler(30)", &ltx::guidance::ltx2_sigmas(30), &r("sigmas30")?);
        }
        "connector" => {
            let files = ModelFiles::open_file(file, false)?;
            let dit = ltx::dit::Dit::load(&files)?;
            // (On the device's session with LTX_DEVICE=metal.)
            let t0 = std::time::Instant::now();
            let v = dit.connect(true, &r("vfeat")?, 1024);
            let a = dit.connect(false, &r("afeat")?, 1024);
            println!("connectors {:.2}s", t0.elapsed().as_secs_f64());
            compare("video connector", &v, &r("vconn")?);
            compare("audio connector", &a, &r("aconn")?);
        }
        "vae" => {
            let m = meta(dir)?;
            let u = |k: &str| m[k].as_u64().unwrap_or(0) as usize;
            let files = ModelFiles::open_file(file, false)?;
            let dec = ltx::vae::VideoDecoder::load(&files)?;
            let t0 = std::time::Instant::now();
            let px = dec.decode(&r("lat")?, u("t"), u("h"), u("w"), &mut |_, _| {});
            println!("decode {:.2}s", t0.elapsed().as_secs_f64());
            compare("decoded frames", &px.to_cthw(), &r("px")?);
            if dir.join("img.f32").exists() {
                let enc = ltx::vae::VideoEncoder::load(&files)?;
                let img = r("img")?;
                let (ih, iw) = (u("ih"), u("iw"));
                let v = ltx::vae::Vid::from_cthw(&img, 3, 1, ih, iw);
                let z = enc.encode(&v);
                compare("encoded image", &z.to_cthw(), &r("z")?);
            }
        }
        "diffvae" => {
            // ref_diffvae.py's decode: the same latents and the reference's recorded noise.
            let m = meta(dir)?;
            let u = |k: &str| m[k].as_u64().unwrap_or(0) as usize;
            let files = ModelFiles::open_file(file, false)?;
            let dec = ltx::diffvae::DiffDecoder::load(&files)?;
            let (t, h, w) = (u("t"), u("h"), u("w"));
            let noise = r("noise")?;
            println!("noise: {} values, decode draws {}", noise.len(), dec.noise_len(t, h, w));
            let mut at = 0;
            let mut draw = |n: usize| {
                let v = noise[at..at + n].to_vec();
                at += n;
                v
            };
            let t0 = std::time::Instant::now();
            let px = dec.decode(&r("lat")?, t, h, w, &mut draw, &mut |_, _| {});
            println!("decode {:.2}s ({}x{}x{} frames)", t0.elapsed().as_secs_f64(), px.t, px.h, px.w);
            compare("decoded frames", &px.to_cthw(), &r("px")?);
        }
        "diffvae_time" => {
            // ltx_check diffvae_time FILE T H W: a decode of random latents, timed (either VAE).
            let n = |i: usize| -> Result<usize> { a.get(i).context("T H W")?.parse::<usize>().context("T H W") };
            let (t, h, w) = (n(3)?, n(4)?, n(5)?);
            let files = ModelFiles::open_file(file, false)?;
            let mut g = ltx::NoiseGen::new(&ltx::Noise::Oxinfer, 1);
            let lat = g.normals(128 * t * h * w);
            let t0 = std::time::Instant::now();
            // The convolutional VAE's decode for comparison when FILE is that one.
            let px = if ltx::diffvae::is_diffusion_vae(&files) {
                let dec = ltx::diffvae::DiffDecoder::load(&files)?;
                dec.decode(&lat, t, h, w, &mut |n| g.normals(n), &mut |_, _| {})
            } else {
                ltx::vae::VideoDecoder::load(&files)?.decode(&lat, t, h, w, &mut |_, _| {})
            };
            println!("decode {:.2}s ({}x{}x{} frames)", t0.elapsed().as_secs_f64(), px.t, px.h, px.w);
        }
        "vaebench" => {
            // ltx_check vaebench FILE T H W DIR: decode fixed random latents [128, T, H, W]
            // (timed), the frames written to DIR/px.f32; compared with DIR/px_ref.f32 when
            // present (another run's: CPU against GPU).
            let (t, h, w) = (a[3].parse::<usize>()?, a[4].parse::<usize>()?, a[5].parse::<usize>()?);
            let files = ModelFiles::open_file(file, false)?;
            let dec = ltx::vae::VideoDecoder::load(&files)?;
            let mut rng = oxinfer::sd::ops::Rng::new(7);
            let lat = rng.normals(128 * t * h * w);
            let t0 = std::time::Instant::now();
            let px = dec.decode(&lat, t, h, w, &mut |_, _| {});
            println!("decode {}x{}x{} frames {:.2}s", px.t, px.h, px.w, t0.elapsed().as_secs_f64());
            std::fs::create_dir_all(dir)?;
            std::fs::write(
                dir.join("px.f32"),
                px.data.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(),
            )?;
            if dir.join("px_ref.f32").exists() {
                compare("decoded frames", &px.data, &r("px_ref")?);
            }
            let enc = ltx::vae::VideoEncoder::load(&files)?;
            let img = ltx::vae::Vid {
                t: 1,
                h: px.h,
                w: px.w,
                c: 3,
                data: px.data[..px.h * px.w * 3].to_vec(),
            };
            let t0 = std::time::Instant::now();
            let z = enc.encode(&img);
            println!("encode one {}x{} frame {:.2}s", px.w, px.h, t0.elapsed().as_secs_f64());
            std::fs::write(dir.join("z.f32"), z.data.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
            if dir.join("z_ref.f32").exists() {
                compare("encoded frame", &z.data, &r("z_ref")?);
            }
        }
        "audiobench" => {
            // ltx_check audiobench FILE N DIR: N random audio latent rows through the audio
            // decoder and the vocoder (timed), the waveform written to DIR/wave.f32 and compared
            // with DIR/wave_ref.f32 when present.
            let n = a[3].parse::<usize>()?;
            let files = ModelFiles::open_file(file, false)?;
            let dec = ltx::audio::AudioDecoder::load(&files)?;
            let mut rng = oxinfer::sd::ops::Rng::new(9);
            let rows = rng.normals(n * 128);
            let t0 = std::time::Instant::now();
            let mel = dec.decode(&rows)?;
            println!("audio decoder {:.2}s", t0.elapsed().as_secs_f64());
            let t_mel = mel.len() / (2 * dec.mel_bins);
            let voc = ltx::audio::Vocoder::load(&files)?;
            let t0 = std::time::Instant::now();
            let w = voc.forward(&mel, t_mel);
            println!("vocoder {:.2}s", t0.elapsed().as_secs_f64());
            let flat: Vec<f32> = mel.iter().chain(w.concat().iter()).copied().collect();
            std::fs::create_dir_all(dir)?;
            std::fs::write(dir.join("wave.f32"), flat.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
            if dir.join("wave_ref.f32").exists() {
                let want = r("wave_ref")?;
                compare("mel", &mel, &want[..mel.len().min(want.len())]);
                compare("mel and waveform", &flat, &want);
            }
        }
        "audio" => {
            let files = ModelFiles::open_file(file, false)?;
            let dec = ltx::audio::AudioDecoder::load(&files)?;
            let mel = dec.decode(&r("rows")?)?;
            compare("mel", &mel, &r("mel")?);
            let t_mel = mel.len() / (2 * dec.mel_bins);
            let voc = ltx::audio::Vocoder::load(&files)?;
            let t0 = std::time::Instant::now();
            let w = voc.forward(&r("mel")?, t_mel);
            println!("vocoder {:.2}s", t0.elapsed().as_secs_f64());
            let flat: Vec<f32> = w.concat();
            compare("waveform", &flat, &r("wave")?);
        }
        "upsample" => {
            let vae = Path::new(&a[3]);
            let m = meta(dir)?;
            let u = |k: &str| m[k].as_u64().unwrap_or(0) as usize;
            let files = ModelFiles::open_file(file, false)?;
            let up = ltx::upsample::LatentUpsampler::load(&files)?;
            let vf = ModelFiles::open_file(vae, false)?;
            let (mean, std) = ltx::vae::stats_of(&vf)?;
            let (t, h, w) = (u("t"), u("h"), u("w"));
            let lat = r("lat")?;
            let mut v = ltx::vae::Vid::from_cthw(&lat, 128, t, h, w);
            for row in v.data.chunks_mut(128) {
                for c in 0..128 {
                    row[c] = row[c] * std[c] + mean[c];
                }
            }
            let mut y = up.forward(&v);
            for row in y.data.chunks_mut(128) {
                for c in 0..128 {
                    row[c] = (row[c] - mean[c]) / std[c];
                }
            }
            compare("upsampled", &y.to_cthw(), &r("out")?);
        }
        "rows" => {
            // ltx_check rows FILE_A FILE_B MODULE: a layer's first rows as two files store them
            // (bf16, NVFP4: not ConvRot), decoded and compared.
            let (fb, module) = (Path::new(&a[3]), a[4].as_str());
            let decode = |p: &Path| -> Result<Vec<f32>> {
                let files = ModelFiles::open_file(p, false)?;
                let name = format!("model.diffusion_model.{module}.weight");
                let (n_out, n_in) = if files.has(&format!("model.diffusion_model.{module}.comfy_quant")) {
                    let s = files
                        .info(&format!("model.diffusion_model.{module}.weight_scale"))
                        .context("scale")?
                        .shape
                        .clone();
                    let w = files.info(&name).context("weight")?.shape.clone();
                    // NVFP4: [N, K / 2] bytes; int8: [N, K].
                    if s.len() == 2 && s[1] > 1 { (w[0], w[1] * 2) } else { (w[0], w[1]) }
                } else {
                    let w = files.info(&name).context("weight")?.shape.clone();
                    (w[0], w[1])
                };
                let l = oxinfer::model::load_lin(&files, &name, n_out, n_in, oxinfer::model::Quant::Bf16)?;
                // Rows decoded as stored (not for ConvRot rows: those are in the rotated basis).
                let rows: Vec<u32> = (0..256.min(n_out) as u32).collect();
                let mut w = vec![0f32; rows.len() * n_in];
                if !l.rows(&rows, n_in, &mut w) {
                    bail!("{}: rows stored rotated", p.display());
                }
                Ok(w)
            };
            let (wa, wb) = (decode(file)?, decode(fb)?);
            compare(&format!("{module} (first 256 rows)"), &wa, &wb);
            let (ma, mb) = (wa.iter().sum::<f32>() / wa.len() as f32, wb.iter().sum::<f32>() / wb.len() as f32);
            let cov: f64 = wa.iter().zip(&wb).map(|(&x, &y)| ((x - ma) * (y - mb)) as f64).sum();
            let va: f64 = wa.iter().map(|&x| ((x - ma) * (x - ma)) as f64).sum();
            let vb: f64 = wb.iter().map(|&y| ((y - mb) * (y - mb)) as f64).sum();
            println!("correlation {:.5}", cov / (va * vb).sqrt());
        }
        "gemma" => {
            // The first layers of the text encoder against ref_text.py's truncated model.
            let m = meta(dir)?;
            let nl = m["layers"].as_u64().unwrap_or(6) as usize;
            let files = ModelFiles::open_file(file, false)?;
            let mut raw = ltx::text::gemma_config(&files)?;
            let tc = raw.get_mut("text_config").context("no text_config")?;
            tc["num_hidden_layers"] = nl.into();
            let lt: Vec<serde_json::Value> = tc["layer_types"].as_array().context("layer_types")?[..nl].to_vec();
            tc["layer_types"] = lt.into();
            let cfg = oxinfer::config::TextConfig::from_json(raw)?;
            let h = cfg.hidden_size;
            let model = oxinfer::model::Model::load(cfg, &files, oxinfer::model::Quant::Bf16)?;
            let ids: Vec<u32> = std::fs::read(dir.join("ids.i32"))?
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| i32::from_le_bytes(*c) as u32)
                .collect();
            let emb: Vec<f32> = ids.iter().flat_map(|&t| model.embed_scaled(t)).collect();
            compare("hidden 0", &emb, &r("hs0")?);
            let mut sess = model.new_session();
            sess.capture = (0..nl - 1).collect();
            let last = model.forward_hidden(&mut sess, &ids);
            for (l, x) in &sess.captured {
                compare(&format!("hidden {}", l + 1), x, &r(&format!("hs{}", l + 1))?);
            }
            compare(&format!("hidden {nl} (normed)"), &last, &r(&format!("hs{nl}"))?);
            let _ = h;
        }
        "text" => {
            // Tokens against ref_text.py's; writes every hidden state and both features for
            // ref_features.py.
            let files = ModelFiles::open_file(file, false)?;
            let enc = ltx::text::TextEncoder::load(&files)?;
            let prompt = std::fs::read_to_string(dir.join("prompt.txt"))?;
            let ids = enc.tokens(&prompt)?;
            let want: Vec<u32> = std::fs::read(dir.join("ids.i32"))?
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| i32::from_le_bytes(*c) as u32)
                .collect();
            println!("tokens: {} ours, {} reference, equal {}", ids.len(), want.len(), ids == want);
            let t0 = std::time::Instant::now();
            let states = enc.hidden_states(&ids);
            println!("text encoder {:.2}s", t0.elapsed().as_secs_f64());
            let h = enc.model.cfg.hidden_size;
            let f = enc.features(&states, ids.len(), h);
            let out = dir.join("full");
            std::fs::create_dir_all(&out)?;
            let w = |n: &str, v: &[f32]| {
                std::fs::write(
                    out.join(format!("{n}.f32")),
                    v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(),
                )
            };
            for (i, s) in states.iter().enumerate() {
                w(&format!("hs{i}"), s)?;
            }
            w("vfeat", &f.video)?;
            w("afeat", &f.audio)?;
            println!("wrote {} hidden states and the features to {}", states.len(), out.display());
        }
        c => bail!("unknown check {c}"),
    }
    Ok(())
}
