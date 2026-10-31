//! Editing an image with Stable Diffusion through the library:
//! `cargo run --release --example sd_edit -- MODEL IN.png PROMPT OUT.png [MASK.png]`
//!
//! With `timbrooks/instruct-pix2pix` the prompt is an instruction ("make it snowy"); with an
//! inpainting model (`stable-diffusion-v1-5/stable-diffusion-inpainting`) or any model plus a mask,
//! the mask's white part is repainted; otherwise the image is redrawn at strength 0.6.

use oxinfer::sd::{EditInput, EditKind, Params, Pipeline, sched::Sampler};

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (model, input, prompt, out, mask) = match a.as_slice() {
        [m, i, p, o] => (m, i, p, o, None),
        [m, i, p, o, k] => (m, i, p, o, Some(k)),
        _ => anyhow::bail!("usage: sd_edit MODEL IN.png PROMPT OUT.png [MASK.png]"),
    };
    let media = oxinfer::media::Media::new();
    let mut edit = EditInput::new(media.load_image(input.as_ref())?);
    if let Some(m) = mask {
        edit = edit.with_mask(media.load_image(m.as_ref())?);
    }
    // Convolutions, attention and norms on the GPU when there is one (Metal, CUDA).
    oxinfer::gpu::init(&oxinfer::platform::detect(), oxinfer::platform::DevicePref::Auto);
    let pipe = Pipeline::load(&oxinfer::hf::resolve(model, None)?)?;
    let kind = pipe.edit_kind(edit.mask.is_some())?;
    if kind == EditKind::Img2Img {
        edit = edit.with_strength(0.6);
    }
    let params = Params {
        prompt: prompt.clone(),
        negative: String::new(),
        steps: if kind == EditKind::Instruct { 20 } else { 25 },
        guidance: 7.5,
        seed: 1,
        // 0 x 0: the model's area at the input's aspect ratio (Pipeline::edit_size).
        width: 0,
        height: 0,
        sampler: Sampler::EulerAncestral,
    };
    let (img, t) = pipe.generate_from(&params, &edit, &mut |i, n| eprint!("\r{kind:?} {i}/{n}"))?;
    eprintln!("\nencode {:.2}s, denoise {:.2}s, decode {:.2}s", t.encode, t.denoise, t.decode);
    std::fs::write(out, oxinfer::png::encode_rgb(img.width, img.height, &img.rgb))?;
    Ok(())
}
