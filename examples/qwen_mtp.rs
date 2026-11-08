//! Qwen3.5 / Qwen3.8 speculative decoding with the checkpoint's MTP layer, through the library:
//! `cargo run --release --example qwen_mtp -- MODEL PROMPT [TEMPERATURE]`
//! (MODEL is a local directory or a repo id such as Qwen/Qwen3.5-0.8B).

use oxinfer::chat::{self, ChatFormat};
use oxinfer::config::{QwenConfig, load_eos_ids};
use oxinfer::model::Quant;
use oxinfer::qwen::QwenModel;
use oxinfer::qwen_mtp::{MtpDecoder, MtpOptions};
use oxinfer::safetensors::ModelFiles;
use oxinfer::sampler::Sampler;
use std::io::Write;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let spec = args.next().expect("usage: qwen_mtp MODEL PROMPT [TEMPERATURE]");
    let prompt = args.next().expect("usage: qwen_mtp MODEL PROMPT [TEMPERATURE]");
    let temperature: f32 = args.next().map_or(Ok(0.0), |t| t.parse())?;
    oxinfer::hf::set_hub(oxinfer::hf::Hub::Auto);
    let dir = oxinfer::hf::resolve(&spec, None)?;
    oxinfer::pool::init(std::thread::available_parallelism()?.get());

    let files = ModelFiles::open(&dir, false)?;
    let model = QwenModel::load(QwenConfig::load(&dir)?, &files, Quant::Q8)?;
    // The MTP layer can also come from a sidecar `ModelFiles` with the same tensor names.
    let mtp = model.load_mtp(&files)?;
    let tok = oxinfer::tokenizer::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("{e}"))?;
    let fmt = chat::QwenChat::for_model(&dir);
    let stop = chat::with_turn_end(fmt, &tok, load_eos_ids(&dir));

    let opts = MtpOptions {
        draft_tokens: 3,
        adaptive: true,
        stop: stop.clone(),
        ..Default::default()
    };
    let mut dec = MtpDecoder::new(&model, &mtp, opts);
    let mut sampler = Sampler::new(temperature, 20, 0.95, 42);
    let text = fmt.first_turn(None, false, &[], &prompt);
    let ids = tok
        .encode(text.as_str(), false)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .get_ids()
        .to_vec();

    let logits = dec.prefill(&ids, 256);
    let first = sampler.sample(&logits);
    dec.push(first);
    let mut out = vec![first];
    while !stop.contains(out.last().unwrap()) && out.len() < 512 {
        out.extend(dec.step(&mut sampler, 512 - out.len()));
    }
    let shown: Vec<u32> = out.iter().copied().filter(|t| !stop.contains(t)).collect();
    println!("{}", tok.decode(&shown, true).map_err(|e| anyhow::anyhow!("{e}"))?);
    std::io::stdout().flush()?;
    eprintln!(
        "[{} tokens in {} rounds: {:.2} tokens/forward, {:.0}% of drafts accepted]",
        dec.stats.emitted + 1,
        dec.stats.rounds,
        dec.stats.tokens_per_round(),
        100.0 * dec.stats.acceptance()
    );
    Ok(())
}
