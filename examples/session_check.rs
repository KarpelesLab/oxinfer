//! Checks session snapshots ([`oxinfer::snapshot`]) on a real model: the logits after a
//! restored session are bit for bit those of the session it was saved from, and of the same
//! computation redone from scratch; and times the restore against recomputing the prompt.
//!
//! ```text
//! session_check MODEL PROMPT_FILE [--steps N] [--quant Q] [--cpu] [--dir DIR]
//! ```
//!
//! Prefills the chat-formatted prompt into a session (A), writes it to a snapshot in `DIR`
//! (default: a private directory in the prompt-cache location), reads it into a fresh session
//! (B), prefills the prompt again into a third session (C, the recomputation), then decodes
//! `N` greedy steps (default 32) feeding the same token to all three and compares their logits.

use anyhow::{Context, Result, bail};
use oxinfer::chat::ChatFormat;
use oxinfer::generate::TextLm;
use oxinfer::model::Quant;
use oxinfer::safetensors::ModelFiles;
use oxinfer::sampler::argmax;
use oxinfer::snapshot::{SessionState, SnapReader, SnapWriter};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn check<M: TextLm>(m: &M, ids: &[u32], steps: usize, dir: &Path) -> Result<()>
where
    M::Sess: SessionState,
{
    let chunk = 1024;
    let h = m.hidden_size();
    let t0 = Instant::now();
    let mut a = m.new_session();
    let hs = oxinfer::generate::prefill(m, &mut a, ids, chunk, &[]);
    let la = m.logits(&hs[hs.len() - h..], 1);
    let prefill = t0.elapsed().as_secs_f64();
    let path = dir.join("session-check.oxsnap");
    let t1 = Instant::now();
    let bytes = oxinfer::snapshot::write_atomic(&path, |f| {
        let mut w = SnapWriter::new(f)?;
        a.write_state(&mut w)?;
        w.finish(serde_json::json!({ "model": "check", "tokens": ids.len() }))?;
        Ok(())
    })?;
    let save = t1.elapsed().as_secs_f64();
    let t2 = Instant::now();
    let mut b = m.new_session();
    let mut r = SnapReader::open(&path)?;
    b.read_state(&mut r)?;
    r.finish()?;
    let restore = t2.elapsed().as_secs_f64();
    println!(
        "{} prompt tokens: prefill {prefill:.3}s ({:.0} tok/s); snapshot {:.1} MB, saved in {save:.3}s, restored in {restore:.3}s ({:.1}x faster than the prefill)",
        ids.len(),
        ids.len() as f64 / prefill,
        bytes as f64 / 1e6,
        prefill / restore
    );
    let mut c = m.new_session();
    let hs = oxinfer::generate::prefill(m, &mut c, ids, chunk, &[]);
    let lc = m.logits(&hs[hs.len() - h..], 1);
    if la != lc {
        println!("note: two prefills of the prompt differ (nondeterministic prefill)");
    }
    let mut tok = argmax(&la);
    let (mut same_ab, mut same_ac) = (0, 0);
    for s in 0..steps {
        let la = m.forward(&mut a, &[tok]);
        let lb = m.forward(&mut b, &[tok]);
        let lc = m.forward(&mut c, &[tok]);
        let d = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max);
        same_ab += usize::from(la == lb);
        same_ac += usize::from(la == lc);
        if la != lb || la != lc {
            println!("step {s}: max |A - B| {:e}, max |A - C| {:e}", d(&la, &lb), d(&la, &lc));
        }
        tok = argmax(&la);
    }
    println!("{steps} decode steps: restored = original bit for bit at {same_ab}, recomputed = original at {same_ac}");
    std::fs::remove_file(&path).ok();
    if same_ab != steps {
        bail!("the restored session differs");
    }
    Ok(())
}

fn main() -> Result<()> {
    let usage = "usage: session_check MODEL PROMPT_FILE [--steps N] [--quant Q] [--cpu] [--dir DIR]";
    let mut args = std::env::args().skip(1);
    let model = args.next().context(usage)?;
    let prompt = std::fs::read_to_string(args.next().context(usage)?)?;
    let (mut steps, mut quant, mut cpu, mut dir) = (32usize, Quant::Bf16, false, None::<PathBuf>);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--steps" => steps = args.next().context("--steps N")?.parse()?,
            "--quant" => quant = Quant::parse(&args.next().context("--quant Q")?).context("bad --quant")?,
            "--cpu" => cpu = true,
            "--dir" => dir = Some(PathBuf::from(args.next().context("--dir DIR")?)),
            other => bail!("unknown argument {other}"),
        }
    }
    let dev = if cpu {
        oxinfer::platform::DevicePref::Cpu
    } else {
        oxinfer::platform::DevicePref::Auto
    };
    oxinfer::gpu::init(&oxinfer::platform::detect(), dev)?;
    let dir = dir.unwrap_or_else(|| oxinfer::snapshot::default_dir().join("check"));
    oxinfer::snapshot::private_dir(&dir)?;
    let path = oxinfer::engine::resolve_model(&model, None)?;
    let files = ModelFiles::open(&path, true)?;
    let tok = oxinfer::engine::load_tokenizer(&path)?;
    let mt = oxinfer::config::model_type(&path)?;
    let fmt: &dyn ChatFormat = if mt.starts_with("gemma") {
        oxinfer::chat::GemmaChat::for_model(&path)
    } else {
        oxinfer::chat::QwenChat::for_model(&path)
    };
    let ids = tok
        .encode(fmt.first_turn(None, false, &[], &prompt), false)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .get_ids()
        .to_vec();
    if mt.starts_with("gemma") {
        let m = oxinfer::model::Model::load(oxinfer::config::TextConfig::load(&path)?, &files, quant)?;
        check(&m, &ids, steps, &dir)
    } else {
        let m = oxinfer::qwen::QwenModel::load(oxinfer::config::QwenConfig::load(&path)?, &files, quant)?;
        check(&m, &ids, steps, &dir)
    }
}
