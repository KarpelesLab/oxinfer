//! FLUX.1 Kontext image editing through the library:
//! `cargo run --release --example flux_edit -- MODEL IN.png "make it night" OUT.png [q8|q4]`
//! MODEL: a diffusers folder or repo (unsloth/FLUX.1-Kontext-dev), or a GGUF transformer
//! (QuantStack/FLUX.1-Kontext-dev-GGUF:Q4_0; the rest then comes from unsloth/FLUX.1-Kontext-dev).

use oxinfer::flux::{LoadOptions, Options, Pipeline};

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (model, input, prompt, out) = match a.as_slice() {
        [m, i, p, o] | [m, i, p, o, _] => (m, i, p, o),
        _ => anyhow::bail!("usage: flux_edit MODEL IN.png INSTRUCTION OUT.png [bf16|q8|q4]"),
    };
    let mut lo = LoadOptions::default();
    if let Some(q) = a.get(4) {
        lo.set("quant", q)?;
    }
    let pipe = Pipeline::load(model, &lo)?;
    let image = oxinfer::media::Media::new().load_image(input.as_ref())?;
    let opts = Options {
        guidance: Some(2.5),
        // Keep the input's aspect ratio (diffusers' default output is 1024x1024).
        width: Some(image.width),
        height: Some(image.height),
        ..Default::default()
    };
    let (img, t) = pipe.edit(&image, prompt, &opts, &mut |i, n| {
        eprint!("\r{i}/{n}");
        true
    })?;
    eprintln!("\n{}x{} in {:.1}s", img.width, img.height, t.text + t.encode + t.denoise + t.decode);
    std::fs::write(out, oxinfer::png::encode_rgb(img.width, img.height, &img.rgb))?;
    Ok(())
}
