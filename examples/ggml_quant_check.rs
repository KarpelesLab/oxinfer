//! Compares oxinfer's ggml block quantizers (`oxinfer::ggml_quant`) with llama.cpp's.
//!
//! `ggml_quant_check gen OUT.bin [MODEL.gguf]` writes test rows (`u32 rows, u32 n_per_row`, the
//! f32 rows, then an f32 importance matrix of `n_per_row` positive weights): Gaussian-like rows
//! with outliers, plus rows of a few tensors of a GGUF model, dequantized.
//! `ggml_quant_check cmp IN.bin DIR...` reads `DIR/<TYPE>.bin` / `DIR/<TYPE>_im.bin` (the same
//! rows quantized by llama.cpp's `ggml_quantize_chunk`, without and with the importance matrix)
//! and reports per type the share of identical blocks and the RMSE of both quantizations.

use oxinfer::ggml::{GgmlType, dequantize_row};
use oxinfer::ggml_quant::quantize_row;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("gen") => gen_rows(Path::new(&args[2]), args.get(3).map(Path::new)),
        Some("cmp") => {
            for d in &args[3..] {
                println!("== {d}");
                cmp(Path::new(&args[2]), Path::new(d))?;
            }
            Ok(())
        }
        _ => anyhow::bail!("usage: ggml_quant_check gen OUT.bin [MODEL.gguf] | cmp IN.bin DIR..."),
    }
}

const N: usize = 2048;

fn gen_rows(out: &Path, model: Option<&Path>) -> anyhow::Result<()> {
    let mut seed = 0x5eed_u64;
    let mut u = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    let mut rows: Vec<f32> = Vec::new();
    for r in 0..96 {
        let scale = 0.002 * (1 + r % 7) as f32;
        for i in 0..N {
            let g = (u() + u() + u() + u()) * scale;
            rows.push(if (i * 31 + r) % 211 == 0 { g * 8.0 } else { g });
        }
    }
    if let Some(m) = model {
        let h = oxinfer::gguf::read_header(m)?;
        let bytes = std::fs::read(m)?;
        for name in [
            "blk.0.attn_q.weight",
            "blk.3.ffn_gate.weight",
            "blk.7.attn_v.weight",
            "blk.15.ffn_down.weight",
        ] {
            let t = h
                .tensors
                .iter()
                .find(|t| t.name == name)
                .ok_or_else(|| anyhow::anyhow!("{name} missing"))?;
            let kind = GgmlType::from_id(t.ggml_type).unwrap();
            let n_in = t.ne[0];
            let rb = kind.row_bytes(n_in);
            let mut row = vec![0f32; n_in];
            for r in 0..64 {
                let o = t.start + r * 37 % t.ne[1] * rb;
                dequantize_row(kind, &bytes[o..o + rb], &mut row);
                // Rows of N: the first N columns (or the row repeated for narrower tensors).
                rows.extend((0..N).map(|i| row[i % n_in]));
            }
        }
    }
    let imat: Vec<f32> = (0..N).map(|_| u().abs() * 4.0 + 0.05).collect();
    let mut b = Vec::new();
    b.extend(((rows.len() / N) as u32).to_le_bytes());
    b.extend((N as u32).to_le_bytes());
    for v in rows.iter().chain(&imat) {
        b.extend(v.to_le_bytes());
    }
    std::fs::write(out, b)?;
    println!("{} rows of {N}", rows.len() / N);
    Ok(())
}

fn cmp(input: &Path, dir: &Path) -> anyhow::Result<()> {
    use GgmlType::*;
    let b = std::fs::read(input)?;
    let rd = |i: usize| u32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap()) as usize;
    let (rows, n) = (rd(0), rd(1));
    let f: Vec<f32> = b[8..].as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
    let (x, imat) = f.split_at(rows * n);
    println!(
        "{:<8} {:>4} {:>9} {:>12} {:>12} {:>9}",
        "type", "im", "blocks=", "rmse ours", "rmse lcpp", "ours/lcpp"
    );
    for kind in [
        Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, TQ1_0, TQ2_0,
        Q1_0, Q2_0, MXFP4,
    ] {
        for im in [false, true] {
            let path = dir.join(format!("{}{}.bin", kind.name(), if im { "_im" } else { "" }));
            let Ok(theirs) = std::fs::read(&path) else { continue };
            let rb = kind.row_bytes(n);
            let mut ours = vec![0u8; rows * rb];
            for r in 0..rows {
                quantize_row(kind, &x[r * n..(r + 1) * n], &mut ours[r * rb..(r + 1) * rb], im.then_some(imat));
            }
            if let Some(d) = std::env::var_os("OXQ_DUMP") {
                std::fs::write(Path::new(&d).join(path.file_name().unwrap()), &ours)?;
            }
            let ts = kind.type_size();
            let same = ours.chunks(ts).zip(theirs.chunks(ts)).filter(|(a, b)| a == b).count();
            let total = ours.len() / ts;
            let rmse = |q: &[u8]| {
                let mut d = vec![0f32; n];
                let (mut e, mut s) = (0f64, 0f64);
                for r in 0..rows {
                    dequantize_row(kind, &q[r * rb..(r + 1) * rb], &mut d);
                    for (a, b) in x[r * n..(r + 1) * n].iter().zip(&d) {
                        e += ((a - b) as f64).powi(2);
                        s += (*a as f64).powi(2);
                    }
                }
                (e / s).sqrt()
            };
            let (ro, rt) = (rmse(&ours), rmse(&theirs));
            println!(
                "{:<8} {:>4} {:>8.3}% {:>12.6} {:>12.6} {:>9.5}",
                kind.name(),
                if im { "yes" } else { "no" },
                100.0 * same as f64 / total as f64,
                ro,
                rt,
                ro / rt
            );
        }
    }
    Ok(())
}
