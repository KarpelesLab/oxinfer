//! Stable Diffusion 3 / 3.5 text to image through the library:
//! `cargo run --release --example sd3_generate -- MODEL "a red fox in the snow" OUT.png [bf16|q8|q4]`
//! MODEL: a diffusers `StableDiffusion3Pipeline` folder or repo (SD 3 medium, SD 3.5 medium /
//! large / large-turbo).

use oxinfer::sd3::{LoadOptions, Options, Pipeline};

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (model, prompt, out) = match a.as_slice() {
        [m, p, o] | [m, p, o, _] => (m, p, o),
        _ => anyhow::bail!("usage: sd3_generate MODEL PROMPT OUT.png [bf16|q8|q4]"),
    };
    let mut lo = LoadOptions {
        vae_encoder: false,
        ..Default::default()
    };
    if let Some(q) = a.get(3) {
        lo.set("quant", q)?;
    }
    let pipe = Pipeline::load(model, &lo)?;
    let opts = Options {
        negative_prompt: Some("blurry, low quality".into()),
        ..Default::default()
    };
    let (img, t) = pipe.generate(prompt, &opts, &mut |i, n| {
        eprint!("\r{i}/{n}");
        true
    })?;
    eprintln!("\n{}x{} in {:.1}s", img.width, img.height, t.text + t.denoise + t.decode);
    std::fs::write(out, oxinfer::png::encode_rgb(img.width, img.height, &img.rgb))?;
    Ok(())
}
