//! The release's prompt enhancer (`PromptEncoder(enhance_first_prompt=True)`,
//! `LTXGemmaTextEncoder.enhance_t2v` / `enhance_i2v`): a Gemma 4 instruct model rewrites a short
//! prompt into a detailed audio-visual caption in the style of the training captions, with the
//! release's system prompts (`gemma4_t2v_system_prompt.txt`, `gemma4_i2v_system_prompt.txt`,
//! Apache-2.0, from github.com/Lightricks/LTX-2) and greedy decoding up to 600 tokens; with a
//! conditioning image, the image goes first in the user turn. LTX-2.5's own text encoder is not
//! an instruct model, so the reference takes a separate one (`--prompt-enhancer-gemma-root`, e.g.
//! Gemma 4 E2B-it); so does oxinfer (any Gemma 4 instruct model it runs, by name or path).

use super::Image;
use crate::engine::{Engine, EngineOptions, Message, SamplingParams};
use crate::generate::Flow;
use anyhow::{Context, Result};

/// The text-to-video system prompt (`default_gemma4_t2v_system_prompt`).
pub const T2V_SYSTEM_PROMPT: &str = include_str!("prompts/gemma4_t2v_system_prompt.txt");
/// The image-to-video system prompt (`default_gemma4_i2v_system_prompt`).
pub const I2V_SYSTEM_PROMPT: &str = include_str!("prompts/gemma4_i2v_system_prompt.txt");
/// The reference's default enhancer: Gemma 4 E2B-it.
pub const DEFAULT_ENHANCER: &str = "google/gemma-4-E2B-it";
/// `GEMMA4_ENHANCE_GENERATION_KWARGS["max_new_tokens"]`.
pub const MAX_TOKENS: usize = 600;

/// `clean_response`: curly quotes and dashes made plain, leading non-letters dropped.
pub fn clean_response(text: &str) -> String {
    let t: String = text
        .chars()
        .map(|c| match c {
            '\u{2018}' | '\u{2019}' | '\u{2032}' => '\'',
            '\u{201c}' | '\u{201d}' => '"',
            '\u{2014}' | '\u{2013}' | '\u{2212}' => '-',
            '\u{a0}' => ' ',
            c => c,
        })
        .collect();
    match t.char_indices().find(|(_, c)| c.is_alphabetic()) {
        Some((i, _)) => t[i..].to_string(),
        None => t,
    }
}

/// `prompt` enhanced by the instruct model `enhancer` (a repository id, folder or file oxinfer
/// loads as a chat model): with `image`, the image-to-video form.
pub fn enhance(enhancer: &str, prompt: &str, image: Option<&Image>, cache_dir: Option<&std::path::Path>) -> Result<String> {
    let opts = EngineOptions {
        cache_dir: cache_dir.map(|p| p.to_path_buf()),
        ..EngineOptions::default()
    };
    let mut engine = Engine::load(enhancer, &opts).with_context(|| format!("the prompt enhancer {enhancer}"))?;
    engine.set_sampling(SamplingParams {
        temperature: 0.0,
        ..SamplingParams::default()
    });
    engine.config_mut().max_tokens = MAX_TOKENS;
    engine.config_mut().think = false;
    let mut tmp = None;
    let messages = match image {
        None => vec![Message::system(T2V_SYSTEM_PROMPT), Message::user(format!("user prompt: {prompt}"))],
        Some(img) => {
            let f = std::env::temp_dir().join(format!("oxinfer-ltx-enhance-{}.png", std::process::id()));
            std::fs::write(&f, crate::png::encode_rgb(img.width, img.height, &img.rgb))?;
            tmp = Some(f.clone());
            let media = crate::engine::Attachments {
                images: vec![f],
                ..Default::default()
            };
            vec![
                Message::system(I2V_SYSTEM_PROMPT),
                Message::user(format!("User Raw Input Prompt: {prompt}.")).with_media(media),
            ]
        }
    };
    let r = engine.chat(&messages, &mut |_| Flow::Continue);
    if let Some(f) = tmp {
        let _ = std::fs::remove_file(f);
    }
    Ok(clean_response(r?.text.trim()))
}

#[cfg(test)]
mod tests {
    #[test]
    fn clean() {
        assert_eq!(
            super::clean_response("  \u{201c}A dog\u{2019}s run \u{2014} fast"),
            "A dog's run - fast"
        );
        assert!(super::T2V_SYSTEM_PROMPT.len() > 1000 && super::I2V_SYSTEM_PROMPT.len() > 1000);
    }
}
