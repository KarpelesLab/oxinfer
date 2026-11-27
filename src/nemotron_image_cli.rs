//! `oxinfer --model nvidia/Nemotron-Labs-Image-30B-A3B --generate-image TEXT -o out.png`: text to
//! image through `oxinfer::nemotron_image` (with `--image FILE`, source images the instruction
//! refers to).

use crate::Args;
use anyhow::{Context, Result, bail};
use oxinfer::generate::Flow;
use oxinfer::nemotron_image::{GenerateOptions, LoadOptions, NemotronImage, Structure};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// `--nemo-image KEY=VALUE` settings: any load option ([`LoadOptions::set`]) or generation option
/// ([`GenerateOptions::set`]) by name.
#[derive(Default, Debug)]
pub struct NemoImageArgs {
    pub set: Vec<(String, String)>,
}

impl NemoImageArgs {
    pub fn parse(&mut self, flag: &str, val: &mut dyn FnMut() -> Result<String>) -> Result<bool> {
        match flag {
            "--nemo-image" => {
                let v = val()?;
                let (k, x) = v
                    .split_once('=')
                    .with_context(|| format!("--nemo-image expects KEY=VALUE, got {v:?}"))?;
                self.set.push((k.trim().to_string(), x.trim().to_string()));
            }
            "--image-tokenizer" => self.set.push(("tokenizer".into(), val()?)),
            _ => return Ok(false),
        }
        Ok(true)
    }
}

const LOAD_KEYS: [&str; 7] = ["quant", "mmap", "device", "cache_dir", "tokenizer", "vit_precision", "encoder"];

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

pub fn run(args: &Args) -> Result<()> {
    let mut lo = LoadOptions {
        quant: args.quant,
        mmap: args.mmap,
        device: args.device,
        cache_dir: args.cache_dir.clone(),
        encoder: !args.images.is_empty(),
        ..Default::default()
    };
    let mut o = GenerateOptions::default();
    if let Some((w, h)) = args.size {
        (o.width, o.height) = (w, h);
    }
    o.seed = args.seed;
    o.num_images = args.num_images.max(1);
    if args.temp_given {
        o.temperature = args.temperature;
    }
    if let Some(p) = args.top_p_set {
        o.top_p = p;
    }
    if std::env::args().any(|a| a == "--top-k") {
        o.top_k = args.top_k;
    }
    if let Some(n) = args.max_tokens_set {
        o.max_tokens = n;
    }
    if args.unconstrained {
        o.structure = Structure::Free;
    }
    if args.no_system {
        o.system_prompt = Some(String::new());
    } else if let Some(s) = &args.system {
        o.system_prompt = Some(s.clone());
    }
    for (k, v) in &args.nemo_image.set {
        if LOAD_KEYS.contains(&k.as_str()) {
            lo.set(k, v)?;
        } else {
            o.set(k, v)?;
        }
    }
    let prompt = match args.generate_image.clone().or_else(|| args.prompt.clone()) {
        Some(p) => p,
        None => {
            let mut s = String::new();
            std::io::stdin().read_to_string(&mut s)?;
            s.trim().to_string()
        }
    };
    if prompt.trim().is_empty() {
        bail!("no prompt: give one with --generate-image TEXT or --prompt TEXT");
    }
    #[cfg(feature = "media")]
    let sources = {
        let m = oxinfer::media::Media::new();
        args.images.iter().map(|p| m.load_image(p)).collect::<Result<Vec<_>>>()?
    };
    #[cfg(not(feature = "media"))]
    let sources: Vec<oxinfer::video::RgbImage> = if args.images.is_empty() {
        Vec::new()
    } else {
        bail!("--image needs the `media` feature (image decoding)")
    };
    let m = NemotronImage::load(&args.model.to_string_lossy(), &lo)?;
    let c = m.config();
    eprintln!(
        "[Nemotron-Labs-Image: {} blocks, hidden {}, {} experts (top {}) | {} | loaded in {:.1}s]",
        c.blocks.len(),
        c.hidden_size,
        c.n_routed_experts,
        c.num_experts_per_tok,
        m.language_model().quant.as_str(),
        m.load_secs
    );
    eprintln!(
        "[{} image(s), {}x{}, temperature {}, top-p {}, seed {}{}{}]",
        o.num_images,
        o.width,
        o.height,
        o.temperature,
        o.top_p,
        o.seed,
        if o.structure == Structure::Free { ", free structure" } else { "" },
        if sources.is_empty() {
            String::new()
        } else {
            format!(", {} source image(s)", sources.len())
        }
    );
    if o.width * o.height < 640 * 640 {
        eprintln!(
            "[note: the model was trained toward 1024x1024-class areas; smaller sizes tend to come out as a blurred corner of a larger picture]"
        );
    }
    let t0 = Instant::now();
    oxinfer::prof::reset();
    let mut last = Instant::now();
    let mut last_step = 0;
    let (imgs, st) = m.generate(&prompt, &sources, &o, &mut |p| {
        if (p.step == p.steps && p.step != last_step) || last.elapsed().as_secs_f64() > 0.5 {
            last_step = p.step;
            last = Instant::now();
            eprint!("\r[image {}/{}: token {}/{}]", p.image + 1, p.images, p.step, p.steps);
        }
        Flow::Continue
    })?;
    eprintln!();
    let out = args.output.clone().unwrap_or_else(|| PathBuf::from("image.png"));
    let mut written = Vec::new();
    for (i, g) in imgs.iter().enumerate() {
        let p = numbered(&out, i, imgs.len());
        let img = &g.image;
        if o.structure == Structure::Free {
            let v = m.vocab;
            let head: Vec<u32> = g
                .tokens
                .iter()
                .copied()
                .take_while(|&t| t != v.vis_row && !v.is_visual(t))
                .collect();
            let text = m.tokenizer().decode(&head, false).unwrap_or_default();
            eprintln!(
                "[image {i}: {} tokens, {:?} before the first row, grid {}x{}]",
                g.tokens.len(),
                text,
                g.grid.1,
                g.grid.0
            );
        }
        std::fs::write(&p, oxinfer::png::encode_rgb(img.width, img.height, &img.rgb)).with_context(|| format!("write {}", p.display()))?;
        written.push(format!("{} ({}x{})", p.display(), img.width, img.height));
    }
    eprintln!(
        "[{} prompt tokens | prefill {:.2}s | {} tokens in {:.2}s ({:.1} tok/s) | decode {:.2}s | total {:.2}s | wrote {}]",
        st.prompt_tokens,
        st.prefill,
        st.tokens,
        st.token_secs,
        st.tokens as f64 / st.token_secs.max(1e-9),
        st.decode,
        t0.elapsed().as_secs_f64(),
        written.join(", ")
    );
    if args.profile {
        oxinfer::prof::report(st.prefill + st.token_secs, st.tokens);
    }
    Ok(())
}
