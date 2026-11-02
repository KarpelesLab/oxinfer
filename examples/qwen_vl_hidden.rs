//! Hidden states of a Qwen2.5-VL (or Qwen3.5) model for a prompt with images, through
//! `oxinfer::qwen_vl` (the conditioning API diffusion pipelines use):
//!
//! ```text
//! qwen_vl_hidden MODEL [--image FILE]... [--layer N]... [--dump DIR] [--cpu] PROMPT
//! ```
//!
//! The prompt is a user turn of the Qwen chat format; `{image}` in it marks where each image goes
//! (default: all images first). With `--dump DIR`, writes `DIR/image{i}.bin` (tower output),
//! `DIR/last.bin` (final-normed hidden states) and `DIR/layer{N}.bin`, all f32. `OXINFER_QUANT`
//! quantizes the language model.

use anyhow::{Context, Result, bail};
use oxinfer::qwen_vl::{QwenVl, QwenVlOptions};
use std::path::PathBuf;

fn save(path: PathBuf, v: &[f32]) -> Result<()> {
    std::fs::write(&path, v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).with_context(|| format!("write {}", path.display()))
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model = args
        .next()
        .context("usage: qwen_vl_hidden MODEL [--image F]... [--layer N]... [--dump DIR] PROMPT")?;
    let (mut images, mut layers, mut dump, mut prompt, mut cpu) = (Vec::new(), Vec::new(), None, None, false);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--image" => images.push(PathBuf::from(args.next().context("--image FILE")?)),
            "--layer" => layers.push(args.next().context("--layer N")?.parse()?),
            "--dump" => dump = Some(PathBuf::from(args.next().context("--dump DIR")?)),
            "--cpu" => cpu = true,
            _ => prompt = Some(a),
        }
    }
    let prompt = prompt.context("no prompt")?;
    let mut opts = QwenVlOptions::default();
    if let Ok(q) = std::env::var("OXINFER_QUANT") {
        opts.quant = oxinfer::model::Quant::parse(&q).context("OXINFER_QUANT")?;
    }
    let dev = if cpu {
        oxinfer::platform::DevicePref::Cpu
    } else {
        oxinfer::platform::DevicePref::Auto
    };
    oxinfer::gpu::init(&oxinfer::platform::detect(), dev)?;
    let t0 = std::time::Instant::now();
    let vl = QwenVl::load(std::path::Path::new(&model), &opts)?;
    eprintln!("[loaded in {:.1}s]", t0.elapsed().as_secs_f64());
    let media = oxinfer::media::Media::new();
    let mut vis = Vec::new();
    for (i, p) in images.iter().enumerate() {
        let img = media.load_image(p)?;
        let v = vl.encode_image(&img.data, img.width, img.height, 0)?;
        eprintln!(
            "[image {}: {}x{} -> {:?} grid, {} tokens]",
            p.display(),
            v.size.0,
            v.size.1,
            v.grid,
            v.n
        );
        if let Some(d) = &dump {
            save(d.join(format!("image{i}.bin")), &v.emb)?;
        }
        vis.push(v);
    }
    let mut user = prompt.clone();
    if !user.contains("{image}") {
        user = "{image}".repeat(vis.len()) + &user;
    }
    for v in &vis {
        user = user.replacen("{image}", &v.placeholder(), 1);
    }
    if user.contains("{image}") {
        bail!("more {{image}} marks than images");
    }
    let text =
        format!("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n");
    let t1 = std::time::Instant::now();
    let h = vl.hidden_states(&text, &vis, &layers)?;
    let t = h.ids.len();
    let rms = |v: &[f32]| (v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
    eprintln!(
        "[{t} tokens x {} in {:.2}s; last hidden rms {:.4}]",
        h.hidden,
        t1.elapsed().as_secs_f64(),
        rms(&h.last)
    );
    if let Some(d) = &dump {
        save(d.join("last.bin"), &h.last)?;
        for (l, v) in &h.layers {
            save(d.join(format!("layer{l}.bin")), v)?;
        }
    }
    Ok(())
}
