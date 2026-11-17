//! MiniMax-H3's text conditioning (the residual stream of Qwen3-VL-32B after its 50th layer)
//! against a transformers reference (`tools/qwen3vl_comfy.py`), from the model's folder or from one
//! of ComfyUI's single-file text encoders (bf16, int8 ConvRot, NVFP4;
//! `oxinfer::qwen_vl::single_file_config`).
//!
//! ```text
//! h3_text_check MODEL_DIR|FILE.safetensors REF.bin --ids IDS.txt [--layer N]
//! h3_text_check MODEL_DIR|FILE.safetensors REF.bin --prompt-file PROMPT.txt --tokenizer-in DIR
//! h3_text_check dump TEXT_ENCODER PROMPT.txt OUT_DIR [--tokenizer-in DIR] [--hidden 1,2,50] [--ids IDS.txt]
//! ```
//!
//! With `--ids`, the language model alone on the given token ids, compared after layer `N`
//! (0-based: transformers' `hidden_states[N + 1]`; default 49). With `--prompt-file`, the video
//! pipeline's own text path (`minimax_h3::video::encode_prompt_quant`: the tokenizer of
//! `DIR/tokenizer`, the 50th layer). Float32 on the CPU.
//!
//! `dump` is for the real weights, where the reference is too large to store beside a test: the
//! pipeline's text encoder (a folder, a GGUF file, a single file) on a prompt, writing the token
//! ids (`OUT_DIR/ids.txt`; with `--ids`, compared with the reference's, and `ids_plain.txt` has
//! what the tokenizer's file alone gives) and `hidden_states[i]` for the `i` of `--hidden`
//! (`OUT_DIR/hs_<i>.bin`, float32 rows), which `tools/minimax_h3_text_ref.py` computes in
//! transformers from the same GGUF and compares.

use anyhow::{Context, Result};
use oxinfer::qwen::QwenModel;
use oxinfer::safetensors::ModelFiles;
use std::path::{Path, PathBuf};

/// `dump TEXT_ENCODER PROMPT.txt OUT_DIR [--tokenizer-in DIR] [--hidden 1,2,50] [--ids IDS.txt]`.
fn dump(mut args: impl Iterator<Item = String>) -> Result<()> {
    use oxinfer::minimax_h3::video::{EXTRA_TOKENS, TEXT_LAYER, load_text_encoder};
    let usage = "usage: h3_text_check dump TEXT_ENCODER PROMPT.txt OUT_DIR [--tokenizer-in DIR] [--hidden 1,2,50] [--ids IDS.txt]";
    let enc = PathBuf::from(args.next().context(usage)?);
    let prompt = args.next().context(usage)?;
    let out = PathBuf::from(args.next().context(usage)?);
    let (mut tok, mut hidden, mut want) = (PathBuf::new(), vec![TEXT_LAYER], None);
    while let Some(a) = args.next() {
        let v = args.next().with_context(|| format!("{a} needs a value"))?;
        match a.as_str() {
            "--tokenizer-in" => tok = PathBuf::from(v),
            "--hidden" => hidden = v.split(',').map(str::parse).collect::<Result<_, _>>()?,
            "--ids" => want = Some(v),
            _ => anyhow::bail!("unknown option {a}"),
        }
    }
    anyhow::ensure!(
        hidden.iter().all(|&l| l >= 1),
        "--hidden counts from 1 (hidden_states[0] is the embedding)"
    );
    oxinfer::gpu::init(&oxinfer::platform::detect(), oxinfer::platform::DevicePref::Cpu)?;
    let text = std::fs::read_to_string(&prompt).with_context(|| prompt.clone())?;
    let vl = load_text_encoder(&tok, &enc, oxinfer::model::Quant::Bf16, false)?;
    let ids = vl.tokenize(&text)?;
    std::fs::create_dir_all(&out)?;
    let line = |ids: &[u32]| ids.iter().map(u32::to_string).collect::<Vec<_>>().join(" ") + "\n";
    std::fs::write(out.join("ids.txt"), line(&ids))?;
    println!("{} tokens", ids.len());
    if let Some(w) = want {
        let w: Vec<u32> = std::fs::read_to_string(&w)?
            .split_whitespace()
            .map(str::parse)
            .collect::<Result<_, _>>()?;
        let differ = ids.len().abs_diff(w.len()) + ids.iter().zip(&w).filter(|(a, b)| a != b).count();
        println!(
            "against {} reference ids: {}",
            w.len(),
            if differ == 0 {
                "identical".into()
            } else {
                format!("{differ} differ")
            }
        );
        // The same text through the tokenizer as its file gives it (what ran before the tokens
        // of `EXTRA_TOKENS` were added).
        let plain = oxinfer::engine::load_tokenizer(&if oxinfer::gguf::is_gguf(&enc) {
            enc.clone()
        } else {
            tok.join("tokenizer")
        })?;
        let p = plain.encode(text.as_str(), false).map_err(|e| anyhow::anyhow!("{e}"))?;
        let extra = ids.iter().filter(|i| EXTRA_TOKENS.iter().any(|t| t.1 == **i)).count();
        println!(
            "without MiniMax's added tokens: {} tokens ({extra} of them in this prompt)",
            p.get_ids().len()
        );
        std::fs::write(out.join("ids_plain.txt"), line(p.get_ids()))?;
        anyhow::ensure!(differ == 0, "the token ids differ from the reference's");
    }
    let layers: Vec<usize> = hidden.iter().map(|l| l - 1).collect();
    let hs = vl.hidden_states_tokens(ids, &[], None, &layers)?;
    for (l, rows) in &hs.layers {
        let bytes: Vec<u8> = rows.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(out.join(format!("hs_{}.bin", l + 1)), bytes)?;
        let rms = (rows.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / rows.len() as f64).sqrt();
        println!("hidden_states[{}]: rms {rms:.4}", l + 1);
    }
    Ok(())
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: h3_text_check MODEL_DIR|FILE.safetensors REF.bin (--ids IDS.txt [--layer N] | --prompt-file F --tokenizer-in DIR)";
    let model = args.next().context(usage)?;
    if model == "dump" {
        return dump(args);
    }
    let want: Vec<f32> = std::fs::read(args.next().context(usage)?)?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    let (mut layer, mut ids, mut prompt, mut tok) = (49usize, None, None, PathBuf::new());
    while let Some(a) = args.next() {
        let v = args.next().with_context(|| format!("{a} needs a value"))?;
        match a.as_str() {
            "--layer" => layer = v.parse()?,
            "--ids" => ids = Some(v),
            "--prompt-file" => prompt = Some(v),
            "--tokenizer-in" => tok = PathBuf::from(v),
            _ => anyhow::bail!("unknown option {a}"),
        }
    }
    oxinfer::gpu::init(&oxinfer::platform::detect(), oxinfer::platform::DevicePref::Cpu)?;
    let path = Path::new(&model);
    let (got, n) = if let Some(p) = prompt {
        let text = std::fs::read_to_string(&p).with_context(|| p.clone())?;
        let c = oxinfer::minimax_h3::video::encode_prompt_quant(&tok, path, oxinfer::model::Quant::Bf16, &text, &[])?;
        (c.text, c.tags.len())
    } else {
        let ids: Vec<u32> = std::fs::read_to_string(ids.context(usage)?)?
            .split_whitespace()
            .map(|v| v.parse::<u32>())
            .collect::<Result<_, _>>()?;
        let (files, cfg) = if path.is_file() {
            let f = ModelFiles::open_file(path, false)?;
            let c = oxinfer::qwen_vl::single_file_config(&f)?;
            (f, c)
        } else {
            (ModelFiles::open(path, false)?, oxinfer::config::QwenConfig::load(path)?)
        };
        println!(
            "{} layers of width {}, {} heads of {} ({} for keys and values), MLP {}",
            cfg.num_hidden_layers, cfg.hidden_size, cfg.num_attention_heads, cfg.head_dim, cfg.num_key_value_heads, cfg.intermediate_size
        );
        let m = QwenModel::load(cfg, &files, oxinfer::model::Quant::Bf16)?;
        let mut sess = m.new_session();
        sess.capture = vec![layer];
        m.forward_hidden_mm(&mut sess, &ids, &[]);
        let (_, got) = sess.captured.pop().context("the layer was not captured")?;
        (got, ids.len())
    };
    anyhow::ensure!(got.len() == want.len(), "{} values, the reference has {}", got.len(), want.len());
    let max = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let scale = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
    println!("{n} tokens: max abs diff {max:.3e} (reference values up to {scale:.2})");
    Ok(())
}
