//! Stable Diffusion XL through the library, with its size conditioning:
//! `cargo run --release --example sd_xl -- MODEL PROMPT OUT.png [WIDTHxHEIGHT]`
//!
//! `MODEL` is an SDXL diffusers folder or repo id (`stabilityai/stable-diffusion-xl-base-1.0`,
//! `stabilityai/sdxl-turbo`). The pipeline is the Stable Diffusion one: the model type comes from
//! `model_index.json`, and the defaults (sampler, guidance, steps) from the model.

use oxinfer::sd::xl::SizeCond;
use oxinfer::sd::{Params, Pipeline, default_steps};

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (model, prompt, out, size) = match a.as_slice() {
        [m, p, o] => (m, p, o, None),
        [m, p, o, s] => (m, p, o, s.split_once('x')),
        _ => anyhow::bail!("usage: sd_xl MODEL PROMPT OUT.png [WIDTHxHEIGHT]"),
    };
    // Convolutions, attention and norms on the GPU when there is one (Metal, CUDA).
    oxinfer::gpu::init(&oxinfer::platform::detect(), oxinfer::platform::DevicePref::Auto)?;
    let mut pipe = Pipeline::load(&oxinfer::hf::resolve(model, None)?)?;
    anyhow::ensure!(pipe.is_xl(), "{model} is not an SDXL pipeline");
    let side = pipe.default_size();
    let (width, height) = match size {
        Some((w, h)) => (w.parse()?, h.parse()?),
        None => (side, side),
    };
    // What the model is told about the picture: by default (None) the output size and no crop.
    // Here: a source picture twice as large as the output.
    pipe.size_cond = Some(SizeCond {
        original: (2 * height, 2 * width),
        ..SizeCond::for_image(width, height)
    });
    let params = Params {
        prompt: prompt.clone(),
        negative: String::new(),
        steps: default_steps(pipe.turbo, None, 1.0),
        guidance: pipe.default_guidance(),
        seed: 1,
        width,
        height,
        sampler: pipe.default_sampler(),
    };
    let (img, t) = pipe.generate(&params, &mut |i, n| eprint!("\r{i}/{n}"))?;
    eprintln!("\ntext {:.2}s, denoise {:.2}s, decode {:.2}s", t.text, t.denoise, t.decode);
    std::fs::write(out, oxinfer::png::encode_rgb(img.width, img.height, &img.rgb))?;
    Ok(())
}
