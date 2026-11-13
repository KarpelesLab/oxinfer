//! Compares an Inkling-2 checkpoint run by oxinfer with reference outputs (as written by
//! `tools/inkling2_tiny.py`: `ids` (I64), `logits` `[n, vocab]` at every prompt position, and
//! optionally `gen` (I64, greedy tokens) with `decode_logits` `[len(gen), vocab]`, the logits after
//! feeding each of them one at a time; `vocab` is the unpadded vocabulary):
//!
//! ```text
//! inkling2_check MODEL REF.safetensors [--chunk N] [--image FILE] [--dump FILE] [--cpu] [--mmap] [--stream]
//! ```
//!
//! Prints, for the prefill (in chunks of N tokens, default the whole prompt) and for the
//! token-by-token decode, the largest logit difference, the mean KL divergence and the agreement
//! of the top token, and whether greedy decoding from the prompt gives the reference's tokens.
//! With `image_features` in the reference (`[n, hidden]`, one row per image placeholder of the
//! prompt), the placeholders take those rows; `--image FILE` takes them from oxinfer's own vision
//! tower instead and reports how far they are from the reference's.
//! `--vision-only` (with `--image`) compares the tower's features alone, for a checkpoint that has
//! only the vision tower's tensors (`tools/ref_inkling2_vision.py`).
//! `--dump` writes the logits compared (prefill rows, then decode rows; f32). `--stream` reads
//! the weights through the streamer (see docs/streaming.md), as an oversized checkpoint is.

use anyhow::{Context, Result};
use oxinfer::inkling::{Inkling2Config, Inkling2Model};
use oxinfer::safetensors::{ModelFiles, SafeTensors};
use std::path::Path;

fn i64s(st: &SafeTensors, name: &str) -> Result<Vec<u32>> {
    Ok(st
        .bytes(name)?
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| i64::from_le_bytes(*c) as u32)
        .collect())
}

fn f32s(st: &SafeTensors, name: &str) -> Result<Vec<f32>> {
    Ok(st.bytes(name)?.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
}

/// Max |a - b|, KL(softmax(b) || softmax(a)), whether the argmaxes agree.
fn compare(a: &[f32], b: &[f32]) -> (f32, f64, bool) {
    let max = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let lse = |v: &[f32]| {
        let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        m + v.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln()
    };
    let (la, lb) = (lse(a), lse(b));
    let kl = b
        .iter()
        .zip(a)
        .map(|(&y, &x)| {
            let pb = (y as f64 - lb).exp();
            pb * ((y as f64 - lb) - (x as f64 - la))
        })
        .sum::<f64>();
    let am = |v: &[f32]| v.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0;
    (max, kl, am(a) == am(b))
}

fn report(what: &str, ours: &[f32], refs: &[f32], vocab: usize) {
    let n = refs.len() / vocab;
    let (mut max, mut kl, mut same, mut worst) = (0f32, 0f64, 0, 0);
    for i in 0..n {
        let (m, k, s) = compare(&ours[i * vocab..(i + 1) * vocab], &refs[i * vocab..(i + 1) * vocab]);
        if m > max {
            (max, worst) = (m, i);
        }
        kl += k;
        same += s as usize;
    }
    println!(
        "{what}: {n} positions, max logit diff {max:.3e} (at {worst}), mean KL {:.3e}, same top-1 {same}/{n}",
        kl / n as f64
    );
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: inkling2_check MODEL REF.safetensors [--chunk N] [--dump FILE] [--cpu] [--mmap] [--stream]";
    let model = args.next().context(usage)?;
    let refp = args.next().context(usage)?;
    let (mut chunk, mut cpu, mut mmap, mut stream, mut vision_only) = (None, false, false, false, false);
    let mut dump: Option<String> = None;
    let mut image: Option<String> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--chunk" => chunk = Some(args.next().context("--chunk N")?.parse::<usize>()?),
            "--cpu" => cpu = true,
            "--mmap" => mmap = true,
            "--stream" => stream = true,
            "--vision-only" => vision_only = true,
            "--dump" => dump = Some(args.next().context("--dump FILE")?),
            "--image" => image = Some(args.next().context("--image FILE")?),
            other => anyhow::bail!("unknown argument {other}"),
        }
    }
    let dev = if cpu {
        oxinfer::platform::DevicePref::Cpu
    } else {
        oxinfer::platform::DevicePref::Auto
    };
    oxinfer::gpu::init(&oxinfer::platform::detect(), dev)?;
    let dir = oxinfer::engine::resolve_model(&model, None)?;
    let cfg = Inkling2Config::load(&dir)?;
    let files = ModelFiles::open(&dir, !(mmap || stream))?;
    if vision_only {
        let st = SafeTensors::open(Path::new(&refp), true)?;
        let (h, feats) = (cfg.hidden_size, f32s(&st, "image_features")?);
        let images = [std::path::PathBuf::from(image.context("--vision-only needs --image FILE")?)];
        let (_, mine) = oxinfer::mm::prepare(&oxinfer::mm::MmRequest::new(&dir, &images, &[]), &files, h)?;
        let mine: Vec<f32> = mine.into_iter().flat_map(|f| f.emb).collect();
        anyhow::ensure!(
            mine.len() == feats.len(),
            "{} image features, the reference has {}",
            mine.len() / h,
            feats.len() / h
        );
        let max = mine.iter().zip(&feats).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let big = feats.iter().fold(0f32, |m, v| m.max(v.abs()));
        let rms = |v: &[f32]| (v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
        let diff: Vec<f32> = mine.iter().zip(&feats).map(|(a, b)| a - b).collect();
        println!(
            "image features: {} tokens, max diff {max:.3e}, rms diff {:.3e} (largest value {big:.2}, rms {:.3})",
            mine.len() / h,
            rms(&diff),
            rms(&feats)
        );
        return Ok(());
    }
    let mut m = Inkling2Model::load(cfg, &files, oxinfer::model::Quant::Bf16)?;
    let streamer = if stream {
        Some(oxinfer::stream::attach_inkling(&mut m, &files, oxinfer::inkling::default_stream())?)
    } else {
        None
    };
    let st = SafeTensors::open(Path::new(&refp), true)?;
    let ids = i64s(&st, "ids")?;
    let refs = f32s(&st, "logits")?;
    let vocab = m.cfg.unpadded_vocab_size;
    let chunk = chunk.unwrap_or(ids.len()).max(1);

    // Image features for the placeholders of the prompt: the reference's, or our tower's.
    let h = m.cfg.hidden_size;
    let mut feats: Vec<f32> = if st.info("image_features").is_some() {
        f32s(&st, "image_features")?
    } else {
        Vec::new()
    };
    if let Some(path) = &image {
        let images = [std::path::PathBuf::from(path)];
        let req = oxinfer::mm::MmRequest::new(&dir, &images, &[]);
        let (_, mine) = oxinfer::mm::prepare(&req, &files, h)?;
        let mine: Vec<f32> = mine.into_iter().flat_map(|f| f.emb).collect();
        anyhow::ensure!(
            mine.len() == feats.len(),
            "{} image features, the reference has {}",
            mine.len() / h,
            feats.len() / h
        );
        let max = mine.iter().zip(&feats).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let big = feats.iter().fold(0f32, |m, v| m.max(v.abs()));
        println!(
            "image features: {} tokens, max diff {max:.3e} (largest value {big:.2})",
            mine.len() / h
        );
        feats = mine;
    }
    let places: Vec<usize> = (0..ids.len()).filter(|&i| ids[i] == m.cfg.image_token_id).collect();
    if !feats.is_empty() {
        anyhow::ensure!(
            places.len() * h == feats.len(),
            "{} placeholders for {} image features",
            places.len(),
            feats.len() / h
        );
    }
    // The placeholders of tokens `from..from + n` as overrides relative to `from`.
    let overrides = |from: usize, n: usize| -> Vec<(usize, &[f32])> {
        if feats.is_empty() {
            return Vec::new();
        }
        places
            .iter()
            .enumerate()
            .filter(|(_, p)| (from..from + n).contains(p))
            .map(|(k, &p)| (p - from, &feats[k * h..(k + 1) * h]))
            .collect()
    };
    let mut sess = m.new_session();
    let mut ours = Vec::with_capacity(refs.len());
    let t0 = std::time::Instant::now();
    for (ci, c) in ids.chunks(chunk).enumerate() {
        let hs = m.forward_capture(&mut sess, c, &overrides(ci * chunk, c.len()), &[], &mut Vec::new());
        ours.extend(m.logits(&hs, c.len()));
    }
    eprintln!(
        "[prefill {} tokens in chunks of {chunk}: {:.2}s]",
        ids.len(),
        t0.elapsed().as_secs_f64()
    );
    report("prefill", &ours, &refs, vocab);

    if st.info("gen").is_some() {
        let gen_ids = i64s(&st, "gen")?;
        let dref = f32s(&st, "decode_logits")?;
        let mut got = Vec::with_capacity(dref.len());
        for &tk in &gen_ids {
            got.extend(m.forward(&mut sess, &[tk]));
        }
        report("decode", &got, &dref, vocab);
        // Greedy decoding on our own: the token after the prompt, then after each of ours.
        let am = |v: &[f32]| v.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0 as u32;
        let mut sess = m.new_session();
        let hs = m.forward_capture(&mut sess, &ids, &overrides(0, ids.len()), &[], &mut Vec::new());
        let mut last = m.logits(&hs[hs.len() - h..], 1);
        let mut mine = Vec::with_capacity(gen_ids.len());
        for _ in 0..gen_ids.len() {
            let tk = am(&last);
            mine.push(tk);
            last = m.forward(&mut sess, &[tk]);
        }
        let same = mine.iter().zip(&gen_ids).take_while(|(a, b)| a == b).count();
        println!(
            "greedy: {same}/{} tokens match the reference{}",
            gen_ids.len(),
            if same == gen_ids.len() { "" } else { " (MISMATCH)" }
        );
        ours.extend(got);
    }
    if let Some(path) = dump {
        let bytes: Vec<u8> = ours.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(&path, bytes).with_context(|| format!("write {path}"))?;
    }
    m.set_stream(None);
    drop(streamer);
    Ok(())
}
