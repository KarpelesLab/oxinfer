//! `oxinfer ocr IMAGE...`: OCR with OCRia-nano-1 ([`oxinfer::ocria`]): the text of each image in
//! reading order, or with `--json` its lines (boxes, text, confidence, script, words).

use anyhow::{Context, Result, bail};
use oxinfer::ocria::{OcrOptions, OcrRunOptions, Ocria, REPO};
use std::time::Instant;

pub const USAGE: &str = "usage: oxinfer ocr IMAGE... [options]\n\
\n\
OCR of images (PNG, JPEG, WebP, GIF, BMP) with OCRia-nano-1 (llamaindex/OCRia-nano-1): a line\n\
detector, a script router and four line recognizers (Latin, CJK, right-to-left, world scripts).\n\
Prints each image's text in reading order, or its lines as JSON.\n\
\n\
  --model SPEC          the repo id (default llamaindex/OCRia-nano-1) or a folder of its files\n\
  --scripts LIST        recognizers to load: latin,cjk,rtl,world or all (default all)\n\
  --long-side N         detector input long side (default 1600)\n\
  --json                JSON: one object per image (lines with box, bbox, text, confidence, script,\n\
                        words; text; script; upside_down; sideways), an array for several images\n\
  --plain               the models' outputs only: no second detection pass, gutter split,\n\
                        orientation retry, script rescue, junk filter, faint tokens, lookalike\n\
                        folding, right-to-left reordering or column order\n\
  --no-adaptive, --no-gutter, --no-orient, --no-rescue, --no-junk, --no-faint, --no-clean,\n\
  --no-bidi, --no-layout   turn one of them off\n\
  --no-exif             read photos as stored (default: upright per their EXIF orientation)\n\
  --thresh T            detector threshold (default det.meta.json's, 0.55)\n\
  --merge G             collinear merge gap in line heights (default 0.6; 0 off)\n\
  --repeat N            run each image N times (timing)\n\
  --device D            where the models run: auto (default: on the GPU for large pages and\n\
                        batches of lines), cpu, metal (always the GPU)\n\
  --threads N           CPU threads\n\
  --cache-dir DIR       download cache\n\
  --profile             time per stage and per operator kind\n\
  -q, --quiet           no timing lines";

/// Runs the arguments after `ocr`.
pub fn run(args: &[String]) -> Result<()> {
    let mut lo = OcrOptions::default();
    let mut ro = OcrRunOptions::default();
    let mut model = REPO.to_string();
    let (mut json, mut quiet, mut repeat, mut profile) = (false, false, 1usize, false);
    let mut images = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let (flag, inline) = match a.split_once('=') {
            Some((f, v)) if f.starts_with("--") => (f, Some(v.to_string())),
            _ => (a.as_str(), None),
        };
        let mut val = || -> Result<String> {
            match &inline {
                Some(v) => Ok(v.clone()),
                None => it.next().cloned().with_context(|| format!("{flag} needs a value")),
            }
        };
        match flag {
            "-h" | "--help" => {
                eprintln!("{USAGE}");
                return Ok(());
            }
            "--model" | "-m" => model = val()?,
            "--scripts" => lo.set("scripts", &val()?)?,
            "--long-side" => {
                let v = val()?;
                lo.set("det_long_side", &v)?;
                ro.set("det_long_side", &v)?;
            }
            "--json" => json = true,
            "--plain" => {
                let side = ro.det_long_side;
                ro = OcrRunOptions::plain();
                ro.det_long_side = side;
            }
            "--thresh" => ro.set("det_thresh", &val()?)?,
            "--merge" => ro.set("det_merge", &val()?)?,
            "--repeat" => repeat = val()?.parse::<usize>().context("--repeat")?.max(1),
            "--threads" => lo.set("threads", &val()?)?,
            "--device" => lo.set("device", &val()?)?,
            "--cache-dir" => lo.set("cache_dir", &val()?)?,
            "-q" | "--quiet" => quiet = true,
            "--profile" => profile = true,
            s if s.starts_with("--no-") && OcrRunOptions::KEYS.contains(&&s[5..]) => ro.set(&s[5..], "0")?,
            s if s.starts_with('-') => bail!("unknown option {s}\n{USAGE}"),
            s => images.push(s.to_string()),
        }
    }
    if images.is_empty() {
        bail!("{USAGE}");
    }
    let t0 = Instant::now();
    let ocr = Ocria::load(&model, &lo)?;
    if !quiet {
        let i = ocr.info();
        eprintln!(
            "[loaded {} ({}; {:.2}M parameters) in {:.2}s]",
            i.name,
            i.scripts.join(", "),
            i.params as f64 / 1e6,
            t0.elapsed().as_secs_f64()
        );
    }
    oxinfer::ocria::set_profile(profile);
    oxinfer::onnx::set_profile(profile);
    let mut pages = Vec::new();
    for path in &images {
        let bytes = std::fs::read(path).with_context(|| format!("read {path}"))?;
        let t1 = Instant::now();
        let img = oxinfer::ocria::decode_image(&bytes, ro.exif).with_context(|| path.clone())?;
        let decode = t1.elapsed().as_secs_f64();
        let mut best = f64::INFINITY;
        let mut page = None;
        for _ in 0..repeat {
            let mut p = ocr.ocr_rgb(&img.rgb, img.width, img.height, &ro).with_context(|| path.clone())?;
            p.exif_orientation = img.orientation;
            best = best.min(p.seconds);
            page = Some(p);
        }
        let page = page.unwrap();
        if profile {
            eprint!("{}{}", oxinfer::ocria::profile_report(), oxinfer::onnx::profile_report());
        }
        if !quiet {
            eprintln!(
                "[{path}: {} x {}, {} lines, {:.1} ms{} (decode {:.1} ms)]",
                img.width,
                img.height,
                page.lines.len(),
                page.seconds * 1e3,
                if repeat > 1 {
                    format!(", best of {repeat} {:.1} ms", best * 1e3)
                } else {
                    String::new()
                },
                decode * 1e3
            );
        }
        if json {
            pages.push(serde_json::to_value(&page)?);
        } else {
            if images.len() > 1 {
                println!("==> {path} <==");
            }
            println!("{}", page.text);
        }
    }
    if json {
        let v = if pages.len() == 1 {
            pages.remove(0)
        } else {
            serde_json::Value::Array(pages)
        };
        println!("{}", serde_json::to_string_pretty(&v)?);
    }
    Ok(())
}
