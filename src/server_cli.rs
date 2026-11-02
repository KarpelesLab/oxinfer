//! `oxinfer serve`: the OpenAI-compatible HTTP server ([`oxinfer::server`]).

use anyhow::{Context, Result, bail};
use oxinfer::server::{Api, ServerOptions, http};

pub const USAGE: &str = "usage: oxinfer serve --model [ID=]SOURCE[,KEY=VALUE...] [--model ...] [options]\n\
\n\
OpenAI-compatible API (/v1/models, /v1/chat/completions, /v1/completions, /v1/embeddings,\n\
/v1/images/generations, /v1/images/edits, /v1/audio/transcriptions, /v1/audio/translations,\n\
/v1/audio/speech) over the listed models only. SOURCE is a local directory or file, a repo id\n\
(org/name, fetched on first use) or org/name:QUANT (a GGUF file of the repo); ID is the name\n\
clients use (default: SOURCE). Model options: the engine's (quant, device, mmap, stream, mmproj,\n\
draft, mtp, max_context, ...) or the loader's for other kinds, and kind=chat|embedding|image|\n\
transcription|speech, system=TEXT, think=1, max_tokens=N, temperature=F, top_p=F, top_k=N,\n\
repetition_penalty=F, prompt_name=NAME, memory_gb=F.\n\
\n\
options:\n\
\x20 --listen ADDR         address to listen on (default 127.0.0.1:8080; port 0 picks one)\n\
\x20 --config FILE         JSON options: {\"listen\": ..., \"models\": [\"ID=SOURCE,...\" or\n\
\x20                       {\"id\": ..., \"model\": ..., \"options\": {...}}], \"api_key\": ..., ...}\n\
\x20 --api-key KEY         require Authorization: Bearer KEY (default: $OXINFER_API_KEY if set)\n\
\x20 --cors ORIGIN         send CORS headers allowing ORIGIN (* for any)\n\
\x20 --max-body-mb N       largest request body (default 16, also the most the listener reads)\n\
\x20 --preload             load every model at start (default: on first use)\n\
\x20 --download            fetch every model at start\n\
\x20 --memory-gb N         memory for loaded models together; least recently used ones are\n\
\x20                       unloaded to make room (default 80% of RAM)\n\
\x20 --cache-dir DIR       where repo ids are fetched to\n\
\x20 --threads N           CPU threads (default: all cores)\n\
\x20 --no-remote-media     do not fetch http(s) image / audio / video URLs (data: URLs only)\n\
\x20 --local-media         also accept local paths and file:// URLs as media (trusted clients only)\n\
\x20 --tls-cert FILE --tls-key FILE  HTTPS with this certificate chain and key (PEM)\n\
\x20 --self-signed         HTTPS with a generated self-signed certificate\n\
\x20 --workers N           connections served at once (default 32)\n\
\x20 --quiet               no request log";

/// Runs the server with the arguments after `serve`.
pub fn run(args: &[String]) -> Result<()> {
    let mut o = ServerOptions::default();
    if let Ok(k) = std::env::var("OXINFER_API_KEY")
        && !k.is_empty()
    {
        o.set("api_key", &k)?;
    }
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let flag = a.strip_prefix("--").or_else(|| a.strip_prefix('-')).unwrap_or(a);
        let (flag, inline) = match flag.split_once('=') {
            Some((f, v)) => (f, Some(v.to_string())),
            None => (flag, None),
        };
        let mut val = || -> Result<String> {
            match &inline {
                Some(v) => Ok(v.clone()),
                None => it.next().cloned().with_context(|| format!("{a} needs a value")),
            }
        };
        match flag {
            "h" | "help" => {
                eprintln!("{USAGE}");
                std::process::exit(2);
            }
            "m" | "model" => o.set("model", &val()?)?,
            "l" => o.set("listen", &val()?)?,
            "preload" | "download" | "self-signed" | "local-media" => o.set(&flag.replace('-', "_"), "1")?,
            "no-remote-media" => o.set("remote_media", "0")?,
            "quiet" | "q" => o.set("log", "0")?,
            f if f.starts_with(|c: char| c.is_ascii_alphabetic()) => {
                let key = f.replace('-', "_");
                if !ServerOptions::KEYS.contains(&key.as_str()) {
                    bail!("unknown option {a} (oxinfer serve --help)");
                }
                o.set(&key, &val()?)?
            }
            _ => bail!("unexpected argument {a} (oxinfer serve --help)"),
        }
    }
    if o.models.is_empty() {
        eprintln!("{USAGE}");
        bail!("no models: list them with --model (only the listed models can be requested)");
    }
    let plat = oxinfer::platform::detect();
    eprintln!("[platform: {}]", plat.summary());
    let api = Api::new(o)?;
    let running = http::start(api.clone())?;
    let models: Vec<String> = api
        .models()
        .into_iter()
        .map(|(id, kind, loaded)| match (kind, loaded) {
            (Some(k), true) => format!("{id} ({}, loaded)", k.as_str()),
            (Some(k), false) => format!("{id} ({})", k.as_str()),
            (None, _) => id,
        })
        .collect();
    eprintln!("[serving {} on {}/v1]", models.join(", "), running.url());
    running.wait()
}
