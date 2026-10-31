//! Step1X-Edit image editing through the library:
//! `cargo run --release --example step1x_edit -- MODEL IN.png "make it night" OUT.png [bf16|q8|q4] [STEPS]`
//! MODEL: a diffusers folder or repo (stepfun-ai/Step1X-Edit-v1p2).

use oxinfer::step1x::{Options, Params, Pipeline};

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (model, input, prompt, out) = match a.as_slice() {
        [m, i, p, o, ..] => (m, i, p, o),
        _ => anyhow::bail!("usage: step1x_edit MODEL IN.png INSTRUCTION OUT.png [bf16|q8|q4] [STEPS]"),
    };
    let mut o = Options {
        mmap: true,
        ..Default::default()
    };
    if let Some(q) = a.get(4) {
        o.set("quant", q)?;
    }
    let mut pipe = Pipeline::load(model, o)?;
    let image = oxinfer::media::Media::new().load_image(input.as_ref())?;
    let mut p = Params::new(prompt);
    if let Some(s) = a.get(5) {
        p.steps = s.parse()?;
    }
    let (img, t) = pipe.edit(&image, &p, &mut |i, n| eprint!("\r{i}/{n}"))?;
    eprintln!(
        "\n{}x{} (worked at {}x{}) in {:.1}s",
        img.width,
        img.height,
        t.size.0,
        t.size.1,
        t.text + t.vae_encode + t.denoise + t.decode
    );
    std::fs::write(out, oxinfer::png::encode_rgb(img.width, img.height, &img.rgb))?;
    Ok(())
}
