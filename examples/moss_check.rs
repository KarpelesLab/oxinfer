//! Checks oxinfer's MOSS-Image components against reference dumps (`ref_moss.py` of the
//! verification notes in `docs/moss-image.md`): token ids and text-encoder features, one
//! transformer forward (conditional, and unconditional with block 22 left out), and a VAE decode.
//!
//! ```text
//! moss_check MODEL REF_DIR [text|step|vae|all] [KEY=VALUE options...]
//! ```

use anyhow::{Context, Result};
use oxinfer::moss_image::{Options, Pipeline};
use std::path::Path;

fn read_f32(p: &Path) -> Result<Vec<f32>> {
    let b = std::fs::read(p).with_context(|| format!("read {}", p.display()))?;
    Ok(b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
}

fn read_i32(p: &Path) -> Result<Vec<u32>> {
    let b = std::fs::read(p).with_context(|| format!("read {}", p.display()))?;
    Ok(b.as_chunks::<4>().0.iter().map(|c| i32::from_le_bytes(*c) as u32).collect())
}

/// Relative L2 error, max abs error and the reference's RMS.
fn compare(name: &str, ours: &[f32], want: &[f32]) {
    assert_eq!(ours.len(), want.len(), "{name}: length");
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

fn psnr(ours: &[f32], want: &[f32]) -> f64 {
    // Both in [-1, 1]: range 2.
    let mse = ours.iter().zip(want).map(|(&a, &b)| ((a - b) as f64).powi(2)).sum::<f64>() / ours.len() as f64;
    10.0 * (4.0 / mse).log10()
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model = args
        .get(1)
        .context("usage: moss_check MODEL REF_DIR [text|step|vae|all] [KEY=VALUE...]")?;
    let refd = Path::new(args.get(2).context("REF_DIR")?);
    let what = args.get(3).map(String::as_str).unwrap_or("all");
    oxinfer::pool::init(std::thread::available_parallelism().map_or(1, |n| n.get()));
    let mut o = Options::default();
    for kv in args.iter().skip(4) {
        let (k, v) = kv.split_once('=').context("options are KEY=VALUE")?;
        o.set(k, v)?;
    }
    let t0 = std::time::Instant::now();
    let mut pipe = Pipeline::load(model, o)?;
    println!(
        "[loaded in {:.1}s; text encoder {}]",
        t0.elapsed().as_secs_f64(),
        pipe.text_encoder_dir().display()
    );
    let all = what == "all";
    if all || what == "text" {
        pipe.load_text_encoder()?;
        let prompt = std::fs::read_to_string(refd.join("prompt.txt"))
            .unwrap_or_else(|_| "a portrait photo of a woman in a red coat on a city street at dusk".to_string());
        for (tag, text) in [("cond", prompt.trim()), ("uncond", "")] {
            let ids = pipe.prompt_ids(text)?;
            let want = read_i32(&refd.join(format!("{tag}_ids.bin")))?;
            println!("{tag}: {} tokens, ids {}", ids.len(), if ids == want { "equal" } else { "DIFFER" });
            let t = std::time::Instant::now();
            let h = pipe.encode_prompt(text)?;
            println!("  encoded in {:.2}s", t.elapsed().as_secs_f64());
            compare(
                &format!("{tag} text features"),
                &h,
                &read_f32(&refd.join(format!("{tag}_text.bin")))?,
            );
        }
        pipe.unload_text_encoder();
    }
    if all || what == "step" {
        let tr = pipe.transformer();
        let c = tr.cfg.latent_channels;
        let x = read_f32(&refd.join("step_x.bin"))?;
        let n = x.len() / c;
        let side = (n as f64).sqrt() as usize;
        let (lh, lw) = (side, n / side);
        let xt = oxinfer::flux::vae::chw_to_hwc(&x, c, lh, lw);
        let t = read_f32(&refd.join("step_t.bin"))?[0];
        let m = tr.step_mod(t);
        oxinfer::prof::reset();
        let t_all = std::time::Instant::now();
        for (tag, skip) in [("cond", vec![]), ("uncond", vec![22usize])] {
            let text = tr.embed_text(&read_f32(&refd.join(format!("{tag}_text.bin")))?)?;
            let t1 = std::time::Instant::now();
            let v = tr.forward(&xt, lh, lw, &text, &m, &skip);
            let secs = t1.elapsed().as_secs_f64();
            let v = oxinfer::flux::vae::hwc_to_chw(&v, c, lh, lw);
            compare(
                &format!("{tag} velocity (t = {t}, {secs:.2}s)"),
                &v,
                &read_f32(&refd.join(format!("step_{tag}.bin")))?,
            );
        }
        oxinfer::prof::report(t_all.elapsed().as_secs_f64(), 2);
    }
    if all || what == "batch" {
        // Both branches through `forward_batch` (whole passes on the device when it runs them),
        // three times: the first builds the session and maps the weights.
        let tr = pipe.transformer();
        let c = tr.cfg.latent_channels;
        let x = read_f32(&refd.join("step_x.bin"))?;
        let n = x.len() / c;
        let side = (n as f64).sqrt() as usize;
        let (lh, lw) = (side, n / side);
        let xt = oxinfer::flux::vae::chw_to_hwc(&x, c, lh, lw);
        let t = read_f32(&refd.join("step_t.bin"))?[0];
        let m = tr.step_mod(t);
        let cond = tr.embed_text(&read_f32(&refd.join("cond_text.bin"))?)?;
        let unc = tr.embed_text(&read_f32(&refd.join("uncond_text.bin"))?)?;
        let skip = [22usize];
        let passes = [
            oxinfer::moss_image::transformer::Pass { text: &cond, skip: &[] },
            oxinfer::moss_image::transformer::Pass { text: &unc, skip: &skip },
        ];
        // MOSS_CHECK_REPS=N: N rounds; MOSS_CHECK_SINGLE=1: the conditional branch alone.
        let reps = std::env::var("MOSS_CHECK_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
        let n = if std::env::var_os("MOSS_CHECK_SINGLE").is_some() { 1 } else { 2 };
        for rep in 0..reps {
            let t1 = std::time::Instant::now();
            let vs = tr.forward_batch(&xt, lh, lw, &m, &passes[..n], pipe.opts.gpu);
            let secs = t1.elapsed().as_secs_f64();
            for (tag, v) in ["cond", "uncond"].iter().zip(&vs) {
                let v = oxinfer::flux::vae::hwc_to_chw(v, c, lh, lw);
                compare(
                    &format!("[{rep}] batch {tag} velocity ({secs:.2}s for both)"),
                    &v,
                    &read_f32(&refd.join(format!("step_{tag}.bin")))?,
                );
            }
        }
    }
    if all || what == "vae" {
        let vae = pipe.vae();
        let c = vae.mean.len();
        let z = read_f32(&refd.join("vae_z.bin"))?;
        let n = z.len() / c;
        let side = (n as f64).sqrt() as usize;
        let (lh, lw) = (side, n / side);
        let t1 = std::time::Instant::now();
        oxinfer::prof::reset();
        let img = vae.decode(&oxinfer::flux::vae::chw_to_hwc(&z, c, lh, lw), lh, lw);
        oxinfer::prof::report(t1.elapsed().as_secs_f64(), 1);
        eprint!("{}", oxinfer::sd::ops::timing::report());
        let secs = t1.elapsed().as_secs_f64();
        let want = read_f32(&refd.join("vae_img.bin"))?;
        let img = oxinfer::flux::vae::hwc_to_chw(&img, 3, 16 * lh, 16 * lw);
        let clamped: Vec<f32> = img.iter().map(|v| v.clamp(-1.0, 1.0)).collect();
        compare(&format!("VAE decode ({secs:.2}s)"), &clamped, &want);
        println!("VAE decode PSNR {:.1} dB", psnr(&clamped, &want));
    }
    Ok(())
}
