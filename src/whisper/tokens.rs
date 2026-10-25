//! Whisper's special tokens and generation settings, from `generation_config.json` (with the
//! tokenizer's added tokens as the fallback for older checkpoints), and the language table.

use anyhow::{Context, Result, bail};
use std::path::Path;
use tokenizers::Tokenizer;

/// The languages Whisper was trained on: (code, English name), in token order.
pub const LANGUAGES: [(&str, &str); 100] = [
    ("en", "english"),
    ("zh", "chinese"),
    ("de", "german"),
    ("es", "spanish"),
    ("ru", "russian"),
    ("ko", "korean"),
    ("fr", "french"),
    ("ja", "japanese"),
    ("pt", "portuguese"),
    ("tr", "turkish"),
    ("pl", "polish"),
    ("ca", "catalan"),
    ("nl", "dutch"),
    ("ar", "arabic"),
    ("sv", "swedish"),
    ("it", "italian"),
    ("id", "indonesian"),
    ("hi", "hindi"),
    ("fi", "finnish"),
    ("vi", "vietnamese"),
    ("he", "hebrew"),
    ("uk", "ukrainian"),
    ("el", "greek"),
    ("ms", "malay"),
    ("cs", "czech"),
    ("ro", "romanian"),
    ("da", "danish"),
    ("hu", "hungarian"),
    ("ta", "tamil"),
    ("no", "norwegian"),
    ("th", "thai"),
    ("ur", "urdu"),
    ("hr", "croatian"),
    ("bg", "bulgarian"),
    ("lt", "lithuanian"),
    ("la", "latin"),
    ("mi", "maori"),
    ("ml", "malayalam"),
    ("cy", "welsh"),
    ("sk", "slovak"),
    ("te", "telugu"),
    ("fa", "persian"),
    ("lv", "latvian"),
    ("bn", "bengali"),
    ("sr", "serbian"),
    ("az", "azerbaijani"),
    ("sl", "slovenian"),
    ("kn", "kannada"),
    ("et", "estonian"),
    ("mk", "macedonian"),
    ("br", "breton"),
    ("eu", "basque"),
    ("is", "icelandic"),
    ("hy", "armenian"),
    ("ne", "nepali"),
    ("mn", "mongolian"),
    ("bs", "bosnian"),
    ("kk", "kazakh"),
    ("sq", "albanian"),
    ("sw", "swahili"),
    ("gl", "galician"),
    ("mr", "marathi"),
    ("pa", "punjabi"),
    ("si", "sinhala"),
    ("km", "khmer"),
    ("sn", "shona"),
    ("yo", "yoruba"),
    ("so", "somali"),
    ("af", "afrikaans"),
    ("oc", "occitan"),
    ("ka", "georgian"),
    ("be", "belarusian"),
    ("tg", "tajik"),
    ("sd", "sindhi"),
    ("gu", "gujarati"),
    ("am", "amharic"),
    ("yi", "yiddish"),
    ("lo", "lao"),
    ("uz", "uzbek"),
    ("fo", "faroese"),
    ("ht", "haitian creole"),
    ("ps", "pashto"),
    ("tk", "turkmen"),
    ("nn", "nynorsk"),
    ("mt", "maltese"),
    ("sa", "sanskrit"),
    ("lb", "luxembourgish"),
    ("my", "myanmar"),
    ("bo", "tibetan"),
    ("tl", "tagalog"),
    ("mg", "malagasy"),
    ("as", "assamese"),
    ("tt", "tatar"),
    ("haw", "hawaiian"),
    ("ln", "lingala"),
    ("ha", "hausa"),
    ("ba", "bashkir"),
    ("jw", "javanese"),
    ("su", "sundanese"),
    ("yue", "cantonese"),
];

/// Other names transformers accepts for a language (`TO_LANGUAGE_CODE`).
const ALIASES: [(&str, &str); 12] = [
    ("burmese", "my"),
    ("valencian", "ca"),
    ("flemish", "nl"),
    ("haitian", "ht"),
    ("letzeburgesch", "lb"),
    ("pushto", "ps"),
    ("panjabi", "pa"),
    ("moldavian", "ro"),
    ("moldovan", "ro"),
    ("sinhalese", "si"),
    ("castilian", "es"),
    ("mandarin", "zh"),
];

/// The English name of a language code ("fr" -> "french").
pub fn language_name(code: &str) -> Option<&'static str> {
    LANGUAGES.iter().find(|l| l.0 == code).map(|l| l.1)
}

/// A language code from a code, an English name or a `<|xx|>` token ("French" -> "fr").
pub fn language_code(s: &str) -> Option<&'static str> {
    let s = s.trim().to_lowercase();
    let s = s.strip_prefix("<|").and_then(|r| r.strip_suffix("|>")).unwrap_or(&s);
    LANGUAGES
        .iter()
        .find(|l| l.0 == s || l.1 == s)
        .map(|l| l.0)
        .or_else(|| ALIASES.iter().find(|a| a.0 == s).map(|a| a.1))
}

/// Token ids and generation defaults of one checkpoint.
#[derive(Clone, Debug)]
pub struct Special {
    pub eot: u32,
    pub sot: u32,
    pub sot_prev: Option<u32>,
    pub no_timestamps: u32,
    /// `<|nospeech|>` / `<|nocaptions|>` (the token before `<|notimestamps|>`).
    pub no_speech: u32,
    pub transcribe: Option<u32>,
    pub translate: Option<u32>,
    /// The first timestamp token (`<|0.00|>`), the one after `<|notimestamps|>`.
    pub timestamp_begin: u32,
    /// (code, token id) of the languages this checkpoint knows, by token id.
    pub languages: Vec<(String, u32)>,
    pub multilingual: bool,
    pub suppress: Vec<u32>,
    pub begin_suppress: Vec<u32>,
    pub max_initial_timestamp_index: Option<u32>,
    /// Decoder-wide length limit (`max_length`, prompt included).
    pub max_length: usize,
    /// (decoder layer, head) pairs whose cross-attention follows the audio (word timestamps).
    pub alignment_heads: Vec<(usize, usize)>,
    /// The checkpoint's default language (distil-whisper English models set one).
    pub default_language: Option<String>,
    pub default_task: Option<String>,
}

fn read_json(path: &Path) -> Option<serde_json::Value> {
    serde_json::from_slice(&crate::fsx::read(path).ok()?).ok()
}

fn ids(v: &serde_json::Value) -> Vec<u32> {
    v.as_array()
        .map(|a| a.iter().filter_map(|x| x.as_u64().map(|x| x as u32)).collect())
        .unwrap_or_default()
}

impl Special {
    pub fn load(dir: &Path, tok: &Tokenizer, vocab_size: usize, max_target: usize) -> Result<Self> {
        let gen_cfg = read_json(&dir.join("generation_config.json")).unwrap_or_default();
        let cfg = read_json(&dir.join("config.json")).unwrap_or_default();
        let get = |k: &str| {
            gen_cfg
                .get(k)
                .filter(|v| !v.is_null())
                .or_else(|| cfg.get(k).filter(|v| !v.is_null()))
        };
        let id =
            |k: &str, token: &str| -> Option<u32> { get(k).and_then(|v| v.as_u64()).map(|v| v as u32).or_else(|| tok.token_to_id(token)) };
        let eot = id("eos_token_id", "<|endoftext|>").context("no <|endoftext|> token")?;
        let sot = id("decoder_start_token_id", "<|startoftranscript|>").context("no <|startoftranscript|> token")?;
        let no_timestamps = id("no_timestamps_token_id", "<|notimestamps|>").context("no <|notimestamps|> token")?;
        let sot_prev = id("prev_sot_token_id", "<|startofprev|>");
        let task_to_id = get("task_to_id");
        let task = |name: &str| {
            task_to_id
                .and_then(|t| t.get(name))
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .or_else(|| tok.token_to_id(&format!("<|{name}|>")))
        };
        let mut languages: Vec<(String, u32)> = match get("lang_to_id").and_then(|v| v.as_object()) {
            Some(m) => m
                .iter()
                .filter_map(|(k, v)| {
                    let code = k.strip_prefix("<|")?.strip_suffix("|>")?;
                    Some((code.to_string(), v.as_u64()? as u32))
                })
                .collect(),
            None => LANGUAGES
                .iter()
                .filter_map(|(c, _)| tok.token_to_id(&format!("<|{c}|>")).map(|i| (c.to_string(), i)))
                .filter(|(_, i)| *i > sot && *i < no_timestamps)
                .collect(),
        };
        languages.sort_by_key(|l| l.1);
        let multilingual = get("is_multilingual").and_then(|v| v.as_bool()).unwrap_or(vocab_size >= 51865) && !languages.is_empty();
        let alignment_heads = get("alignment_heads")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|p| Some((p.get(0)?.as_u64()? as usize, p.get(1)?.as_u64()? as usize)))
                    .collect()
            })
            .unwrap_or_default();
        let max_length = get("max_length")
            .and_then(|v| v.as_u64())
            .map_or(max_target, |v| v as usize)
            .min(max_target);
        let s = Self {
            eot,
            sot,
            sot_prev,
            no_timestamps,
            no_speech: no_timestamps - 1,
            transcribe: task("transcribe"),
            translate: task("translate"),
            timestamp_begin: no_timestamps + 1,
            languages,
            multilingual,
            suppress: get("suppress_tokens").map(ids).unwrap_or_default(),
            begin_suppress: get("begin_suppress_tokens").map(ids).unwrap_or_default(),
            max_initial_timestamp_index: get("max_initial_timestamp_index").and_then(|v| v.as_u64()).map(|v| v as u32),
            max_length,
            alignment_heads,
            default_language: get("language").and_then(|v| v.as_str()).and_then(language_code).map(str::to_string),
            default_task: get("task").and_then(|v| v.as_str()).map(str::to_string),
        };
        if s.timestamp_begin as usize >= vocab_size {
            bail!("the vocabulary has no timestamp tokens");
        }
        Ok(s)
    }

    /// The token of a language given as a code, a name or a `<|xx|>` token.
    pub fn language_token(&self, lang: &str) -> Result<u32> {
        let code = language_code(lang).with_context(|| format!("unknown language {lang:?}"))?;
        self.languages
            .iter()
            .find(|l| l.0 == code)
            .map(|l| l.1)
            .with_context(|| format!("this model does not support language {code}"))
    }

    /// The language code of a language token.
    pub fn language_of(&self, token: u32) -> Option<&str> {
        self.languages.iter().find(|l| l.1 == token).map(|l| l.0.as_str())
    }
}

/// The tokenizer of a checkpoint folder.
pub fn load_tokenizer(dir: &Path) -> Result<Tokenizer> {
    Tokenizer::from_bytes(crate::fsx::read(&dir.join("tokenizer.json"))?).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))
}

/// Maps token ids to text, keeping only text tokens (below `<|endoftext|>`).
pub fn decode_text(tok: &Tokenizer, sp: &Special, ids: &[u32]) -> String {
    let text: Vec<u32> = ids.iter().copied().filter(|&t| t < sp.eot).collect();
    tok.decode(&text, true).unwrap_or_default()
}
