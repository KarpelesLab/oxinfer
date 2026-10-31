//! BAGEL through the library: edit an image, then ask about the result.
//! `cargo run --release --example bagel_edit -- MODEL IN.png "put a hat on the dog" OUT.png [q8|q4]`
//! MODEL: a BAGEL folder or `ByteDance-Seed/BAGEL-7B-MoT`.

use oxinfer::bagel::{Bagel, ImageOptions, LoadOptions, TextOptions};
use oxinfer::generate::Flow;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (model, input, instruction, out) = match a.as_slice() {
        [m, i, p, o] | [m, i, p, o, _] => (m, i, p, o),
        _ => anyhow::bail!("usage: bagel_edit MODEL IN.png INSTRUCTION OUT.png [bf16|q8|q4]"),
    };
    let mut lo = LoadOptions {
        mmap: true,
        ..Default::default()
    };
    if let Some(q) = a.get(4) {
        lo.set("quant", q)?;
    }
    let m = Bagel::load(model, &lo)?;
    let image = oxinfer::media::Media::new().load_image(input.as_ref())?;
    // The reference's editing settings (text guidance 4, image guidance 2, text_channel renorm,
    // 50 timesteps with shift 3); a smaller max_image_size makes it much cheaper.
    let opts = ImageOptions::default();
    let (img, st) = m.edit(&image, instruction, &opts, &mut |i, n| {
        eprint!("\r{i}/{n}");
        true
    })?;
    eprintln!("\n{}x{} in {:.1}s", img.width, img.height, st.context + st.denoise + st.decode);
    std::fs::write(out, oxinfer::png::encode_rgb(img.width, img.height, &img.rgb))?;
    // The same model reads images too.
    let edited = oxinfer::video::RgbImage {
        width: img.width,
        height: img.height,
        data: img.rgb,
    };
    let answer = m.understand(
        Some(&edited),
        "Describe this image in one sentence.",
        &TextOptions::default(),
        &mut |_| Flow::Continue,
    )?;
    println!("{answer}");
    Ok(())
}
