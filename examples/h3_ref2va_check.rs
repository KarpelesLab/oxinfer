//! Compares MiniMax-H3's `ref2va` packed layout (`oxinfer::minimax_h3::ref2va::layout`) with
//! diffusers' on the request `tools/minimax_h3_ref2va_layout.py` writes:
//!
//! ```text
//! h3_ref2va_check DIR
//! ```

use anyhow::{Context, Result};
use oxinfer::minimax_h3::ref2va::{RefBlock, layout};
use oxinfer::minimax_h3::video::Geometry;
use serde_json::Value;

fn main() -> Result<()> {
    let dir = std::env::args().nth(1).context("usage: h3_ref2va_check DIR")?;
    let c: Value = serde_json::from_slice(&std::fs::read(std::path::Path::new(&dir).join("ref_layout.json"))?)?;
    let us = |v: &Value| -> Vec<usize> { v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect() };
    let tags: Vec<u8> = us(&c["text_tags"]).iter().map(|&t| t as u8).collect();
    let (mut vis, mut aud) = (c["visual"].as_array().unwrap().iter(), us(&c["audio"]).into_iter());
    let blocks: Vec<RefBlock> = c["refs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            let kind = r["kind"].as_str().unwrap();
            let visual = (kind != "audio").then(|| {
                let v = us(vis.next().unwrap());
                (v[0], v[1], v[2])
            });
            let audio = if kind != "image" { aud.next().unwrap() } else { 0 };
            RefBlock { visual, audio, image: kind == "image" }
        })
        .collect();
    let t = us(&c["target"]);
    let g = Geometry { lt: t[0], lh: t[1], lw: t[2], n_audio: t[3] };
    let lay = layout(&tags, &blocks, g, 2);
    let want: Vec<[f64; 3]> = c["positions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| {
            let p = p.as_array().unwrap();
            [p[0].as_f64().unwrap(), p[1].as_f64().unwrap(), p[2].as_f64().unwrap()]
        })
        .collect();
    assert_eq!(lay.positions.len(), want.len(), "row count");
    let max = lay
        .positions
        .iter()
        .zip(&want)
        .flat_map(|(a, b)| (0..3).map(move |k| (a[k] as f64 - b[k] as f32 as f64).abs()))
        .fold(0f64, f64::max);
    let tags_ok = lay.tags.iter().map(|&t| t as usize).collect::<Vec<_>>() == us(&c["tags"]);
    let vidx_ok = lay.video_index == us(&c["video_index"]);
    let aidx_ok = lay.audio_index == us(&c["audio_index"]);
    println!(
        "ref2va layout: {} rows, positions max diff {max:.3e} (as float32), tags {}, video rows {}, audio rows {}, reference rows {} / {} (ref {} / {})",
        want.len(),
        if tags_ok { "equal" } else { "DIFFER" },
        if vidx_ok { "equal" } else { "DIFFER" },
        if aidx_ok { "equal" } else { "DIFFER" },
        lay.n_cond,
        lay.n_cond_audio,
        c["n_cond"],
        c["n_cond_audio"]
    );
    anyhow::ensure!(max == 0.0 && tags_ok && vidx_ok && aidx_ok, "the layouts differ");
    Ok(())
}
