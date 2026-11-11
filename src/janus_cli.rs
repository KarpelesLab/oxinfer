//! `oxinfer --model deepseek-ai/Janus-Pro-1B --generate-image TEXT -o out.png`: text to image
//! through `oxinfer::janus` (Janus-Pro and JanusFlow); with `--prompt` (and `--image FILE`),
//! questions about images, and with `--chat` a conversation about them.

use crate::Args;
use anyhow::{Context, Result, bail};
use oxinfer::generate::Flow;
use oxinfer::janus::{ChatOptions, Conversation, ImageOptions, Janus, JanusOptions, Variant};
use std::io::BufRead;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

pub fn run(args: &Args) -> Result<()> {
    let generate = args.generate_image.is_some() || (args.output.is_some() && args.images.is_empty() && !args.chat);
    let opts = JanusOptions {
        quant: args.quant,
        mmap: args.mmap,
        device: args.device,
        cache_dir: args.cache_dir.clone(),
        generation: generate,
        understanding: !generate,
        vae: None,
    };
    let jm = Janus::load(&args.model.to_string_lossy(), &opts)?;
    let c = jm.config();
    eprintln!(
        "[{}: {} layers, hidden {}, {} | loaded in {:.1}s]",
        jm.variant.name(),
        c.num_hidden_layers,
        c.hidden_size,
        args.quant.as_str(),
        jm.load_secs
    );
    let prompt = match args.generate_image.clone().or_else(|| args.prompt.clone()) {
        Some(p) => p,
        None if args.chat && !generate => String::new(),
        None => {
            let mut s = String::new();
            std::io::stdin().read_to_string(&mut s)?;
            s.trim().to_string()
        }
    };
    if generate {
        image(args, &jm, &prompt)
    } else {
        answer(args, &jm, &prompt)
    }
}

/// `out.png` for one image, `out-0.png`, `out-1.png`, ... for several.
fn numbered(out: &Path, i: usize, n: usize) -> PathBuf {
    if n == 1 {
        return out.to_path_buf();
    }
    let stem = out.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let ext = out
        .extension()
        .map(|e| e.to_string_lossy().to_string())
        .unwrap_or_else(|| "png".into());
    out.with_file_name(format!("{stem}-{i}.{ext}"))
}

fn image(args: &Args, jm: &Janus, prompt: &str) -> Result<()> {
    let flow = jm.variant == Variant::Flow;
    let defaults = ImageOptions::default();
    let (width, height) = args.size.unwrap_or((0, 0));
    let o = ImageOptions {
        cfg_weight: args.guidance,
        temperature: if args.temp_given { args.temperature } else { 1.0 },
        seed: args.seed,
        parallel_size: args.num_images.max(1),
        steps: args.steps.unwrap_or(defaults.steps),
        width,
        height,
        ..defaults
    };
    let native = jm.native_size();
    let size = if width == 0 {
        format!("{native}x{native}")
    } else {
        format!("{native}x{native} resampled to {width}x{height}")
    };
    if flow {
        eprintln!(
            "[{} image(s), {size}, {} steps, cfg {}, seed {}]",
            o.parallel_size,
            o.steps,
            o.cfg_weight.unwrap_or(oxinfer::janus::flow::DEFAULT_CFG),
            o.seed
        );
    } else {
        eprintln!(
            "[{} image(s), {size}, cfg {}, temperature {}, seed {}]",
            o.parallel_size,
            o.cfg_weight.unwrap_or(5.0),
            o.temperature,
            o.seed
        );
    }
    let t0 = Instant::now();
    oxinfer::prof::reset();
    let (imgs, st) = jm.generate_images(prompt, &o, &mut |p| {
        if flow {
            eprint!("\r[step {}/{}]", p.step, p.steps);
        } else if p.step % 8 == 0 || p.step == p.steps {
            eprint!("\r[image token {}/{}]", p.step, p.steps);
        }
        Flow::Continue
    })?;
    eprintln!();
    let out = args.output.clone().unwrap_or_else(|| PathBuf::from("image.png"));
    let mut written = Vec::new();
    for (i, img) in imgs.iter().enumerate() {
        let p = numbered(&out, i, imgs.len());
        std::fs::write(&p, oxinfer::png::encode_rgb(img.width, img.height, &img.rgb)).with_context(|| format!("write {}", p.display()))?;
        written.push(p.display().to_string());
    }
    eprintln!(
        "[{} prompt tokens | prefill {:.2}s | {} steps x {} sequences: {:.2}s ({:.1} ms/step) | {} decode {:.2}s | total {:.2}s | wrote {}]",
        st.prompt_tokens,
        st.prefill,
        st.steps,
        2 * imgs.len(),
        st.tokens,
        st.tokens * 1e3 / st.steps.max(1) as f64,
        if flow { "VAE" } else { "VQ" },
        st.decode,
        t0.elapsed().as_secs_f64(),
        written.join(", ")
    );
    if args.profile {
        // Prefill and every image-token step (the VQ decoder's convolutions are not counted).
        oxinfer::prof::report(st.prefill + st.tokens, st.steps);
    }
    Ok(())
}

fn answer(args: &Args, jm: &Janus, question: &str) -> Result<()> {
    if !args.audios.is_empty() || !args.videos.is_empty() {
        bail!("Janus takes images only");
    }
    #[cfg(feature = "media")]
    let images = {
        let m = oxinfer::media::Media::new();
        args.images.iter().map(|p| m.load_image(p)).collect::<Result<Vec<_>>>()?
    };
    #[cfg(not(feature = "media"))]
    let images: Vec<oxinfer::video::RgbImage> = if args.images.is_empty() {
        Vec::new()
    } else {
        bail!("--image needs the `media` feature (image decoding)")
    };
    let o = ChatOptions {
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        seed: args.seed,
        system_prompt: args.system.clone(),
    };
    let mut out = std::io::stdout();
    let mut conv = Conversation::new();
    let mut ask = |question: &str, images: &[oxinfer::video::RgbImage]| -> Result<()> {
        let t0 = Instant::now();
        let text = jm.ask(&mut conv, question, images, &o, &mut |d| {
            let _ = out.write_all(d.as_bytes());
            let _ = out.flush();
            Flow::Continue
        })?;
        println!();
        eprintln!("[{} chars in {:.2}s]", text.len(), t0.elapsed().as_secs_f64());
        Ok(())
    };
    if !args.chat {
        return ask(question, &images);
    }
    // A conversation: the images belong to the first question (`--prompt`, or the first line).
    let mut images = images;
    if !question.trim().is_empty() {
        ask(question, &std::mem::take(&mut images))?;
    }
    let stdin = std::io::stdin();
    loop {
        eprint!("> ");
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break;
        }
        if line.trim().is_empty() {
            continue;
        }
        ask(line.trim(), &std::mem::take(&mut images))?;
    }
    Ok(())
}
