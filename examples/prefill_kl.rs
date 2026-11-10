//! Prefill agreement between devices: writes the last position's output distribution after a
//! prefill of a text file (raw tokens, no template), or compares two such files.
//!
//! ```text
//! prefill_kl MODEL TOKENIZER TEXT OUT.bin [--cpu] [--chunk N] [--repeat N]
//! prefill_kl --compare A.bin B.bin
//! ```

use anyhow::{Context, Result};
use oxinfer::qwen::QwenModel;
use oxinfer::safetensors::ModelFiles;
use std::path::Path;

fn read(p: &str) -> Result<Vec<f32>> {
    Ok(std::fs::read(p)?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.first().map(String::as_str) == Some("--compare") {
        let (x, y) = (read(&a[1])?, read(&a[2])?);
        let lse = |v: &[f32]| {
            let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            m + v.iter().map(|&t| (t as f64 - m).exp()).sum::<f64>().ln()
        };
        let (lx, ly) = (lse(&x), lse(&y));
        let kl: f64 = x
            .iter()
            .zip(&y)
            .map(|(&p, &q)| {
                let pp = (p as f64 - lx).exp();
                pp * ((p as f64 - lx) - (q as f64 - ly))
            })
            .sum();
        let mx = x.iter().zip(&y).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max);
        let am = |v: &[f32]| v.iter().enumerate().max_by(|p, q| p.1.total_cmp(q.1)).unwrap().0;
        println!("KL {kl:.3e}, max diff {mx:.3e}, top-1 {} vs {}", am(&x), am(&y));
        return Ok(());
    }
    let usage = "usage: prefill_kl MODEL TOKENIZER TEXT OUT.bin [--cpu] [--chunk N] [--repeat N]";
    let (model, tok, text, out) = (
        a.first().context(usage)?,
        a.get(1).context(usage)?,
        a.get(2).context(usage)?,
        a.get(3).context(usage)?,
    );
    let cpu = a.iter().any(|x| x == "--cpu");
    let chunk: usize = a
        .iter()
        .position(|x| x == "--chunk")
        .and_then(|i| a.get(i + 1))
        .map_or(Ok(1024), |v| v.parse())?;
    let dev = if cpu {
        oxinfer::platform::DevicePref::Cpu
    } else {
        oxinfer::platform::DevicePref::Auto
    };
    oxinfer::gpu::init(&oxinfer::platform::detect(), dev)?;
    let tk = oxinfer::engine::load_tokenizer(Path::new(tok))?;
    let ids: Vec<u32> = tk.encode(std::fs::read_to_string(text)?, false)?.get_ids().to_vec();
    let dir = oxinfer::engine::resolve_model(model, None)?;
    let dir = Path::new(&dir);
    let cfg = oxinfer::config::QwenConfig::load(dir)?;
    let files = ModelFiles::open(dir, false)?;
    let m = QwenModel::load(cfg, &files, oxinfer::model::Quant::Bf16)?;
    let h = m.cfg.hidden_size;
    let repeat: usize = a
        .iter()
        .position(|x| x == "--repeat")
        .and_then(|i| a.get(i + 1))
        .map_or(Ok(1), |v| v.parse())?;
    let (mut hs, mut dt) = (Vec::new(), 0.0);
    // Later runs (a new session each) leave out first-use costs (GPU plans).
    for r in 0..repeat.max(1) {
        let mut sess = m.new_session();
        let t0 = std::time::Instant::now();
        for c in ids.chunks(chunk) {
            hs = m.forward_hidden_mm(&mut sess, c, &[]);
        }
        dt = t0.elapsed().as_secs_f64();
        let again = if r > 0 { " (again)" } else { "" };
        eprintln!(
            "[prefill {} tokens in {dt:.2}s = {:.1} tok/s{again}]",
            ids.len(),
            ids.len() as f64 / dt
        );
    }
    let l = m.logits(&hs[hs.len() - h..], 1);
    if std::env::var_os("OXINFER_PROFILE").is_some() {
        oxinfer::prof::report(dt, ids.len());
        oxinfer::gpu::pf_report();
    }
    std::fs::write(out, l.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
    Ok(())
}
