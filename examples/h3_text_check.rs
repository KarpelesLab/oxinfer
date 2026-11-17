//! MiniMax-H3's text conditioning (the residual stream of Qwen3-VL-32B after its 50th layer)
//! against a transformers reference (`tools/qwen3vl_comfy.py`), from the model's folder or from one
//! of ComfyUI's single-file text encoders (bf16, int8 ConvRot, NVFP4;
//! `oxinfer::qwen_vl::single_file_config`).
//!
//! ```text
//! h3_text_check MODEL_DIR|FILE.safetensors REF.bin --ids IDS.txt [--layer N]
//! h3_text_check MODEL_DIR|FILE.safetensors REF.bin --prompt-file PROMPT.txt --tokenizer-in DIR
//! ```
//!
//! With `--ids`, the language model alone on the given token ids, compared after layer `N`
//! (0-based: transformers' `hidden_states[N + 1]`; default 49). With `--prompt-file`, the video
//! pipeline's own text path (`minimax_h3::video::encode_prompt_quant`: the tokenizer of
//! `DIR/tokenizer`, the 50th layer). Float32 on the CPU.

use anyhow::{Context, Result};
use oxinfer::qwen::QwenModel;
use oxinfer::safetensors::ModelFiles;
use std::path::{Path, PathBuf};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: h3_text_check MODEL_DIR|FILE.safetensors REF.bin (--ids IDS.txt [--layer N] | --prompt-file F --tokenizer-in DIR)";
    let model = args.next().context(usage)?;
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
