//! `oxinfer serve` and `oxinfer web`: one server. The browser GUI with this machine as its
//! backend ([`oxinfer::server::web`]), its management API, and the OpenAI- and
//! Anthropic-compatible APIs on the same port. `web` also opens the GUI in the browser.

use anyhow::{Result, bail};
use oxinfer::server::{Api, ServerOptions, http};

pub const USAGE: &str = "usage: oxinfer serve|web [--host ADDR] [--port N] [--model [ID=]SOURCE[,KEY=VALUE...]]... [options]\n\
\n\
One server for both: the web GUI (chat, images, speech to text, text to speech, voice chat,\n\
video, embeddings; the model catalog, downloads with progress, the model cache) at\n\
http://HOST:PORT/, the OpenAI-compatible API at /v1 (/v1/models, /v1/chat/completions,\n\
/v1/completions, /v1/embeddings, /v1/images/*, /v1/audio/*, /v1/videos) and Anthropic's\n\
/v1/messages. `oxinfer web` also opens the GUI in the browser; `oxinfer serve` does not\n\
(--open / --no-open change that). Models picked in the GUI join the API's model list.\n\
\n\
Requests may also name models not listed: a repo id (org/name, org/name:QUANT) or a preset of the\n\
catalog is fetched when it is not in the cache (if it fits on the disk and in --memory-gb), loaded\n\
and run, within the request (--no-auto-models: listed models only). Loaded models are unloaded\n\
after --idle-unload minutes unused, when another load needs their memory (RAM or GPU), and while\n\
the system runs short of memory.\n\
\n\
--model SOURCE lists a model from the start: SOURCE is a local directory or file, a repo id\n\
(org/name, fetched on first use) or org/name:QUANT (a GGUF file of the repo); ID is the name\n\
clients use (default: SOURCE). Model options: the engine's (quant, device, mmap, stream,\n\
mmproj, draft, mtp, lookup, lora, prompt_cache, max_context, ...) or the loader's for other\n\
kinds, and kind=chat|embedding|image|transcription|speech, system=TEXT, think=1, max_tokens=N,\n\
temperature=F, top_p=F, top_k=N, repetition_penalty=F, prompt_name=NAME, memory_gb=F.\n\
\n\
options:\n\
\x20 --host ADDR           address to listen on (default 127.0.0.1)\n\
\x20 --port N              port (default 8080, or the next free one; 0 picks one)\n\
\x20 --listen ADDR:PORT    both at once\n\
\x20 --open, --no-open     open the GUI in the browser or not (default: web opens it, serve not)\n\
\x20 --model [ID=]SOURCE   a model listed (and loadable) from the start; repeatable\n\
\x20 --config FILE         JSON options: {\"listen\": ..., \"models\": [\"ID=SOURCE,...\" or\n\
\x20                       {\"id\": ..., \"model\": ..., \"options\": {...}}], \"api_key\": ..., ...}\n\
\x20 --api-key KEY         require Authorization: Bearer KEY for the APIs and the GUI (default:\n\
\x20                       $OXINFER_API_KEY if set). Without one on a non-loopback address the\n\
\x20                       GUI's management API is read-only (no downloads, deletions or loads)\n\
\x20 --cors ORIGIN         send CORS headers allowing ORIGIN (* for any)\n\
\x20 --max-body-mb N       largest request body (default 16, also the most the listener reads)\n\
\x20 --preload             load every listed model at start (default: on first use)\n\
\x20 --download            fetch every listed model at start\n\
\x20 --memory-gb N         memory for loaded models together; least recently used ones are\n\
\x20                       unloaded to make room (default 80% of RAM)\n\
\x20 --idle-unload MIN     unload models unused for MIN minutes (default 30; 0: never)\n\
\x20 --no-auto-models      only the listed models (and those loaded in the GUI) can be requested\n\
\x20 --cache-dir DIR       where repo ids are fetched to\n\
\x20 --prompt-cache DIR    keep chat models' session snapshots in DIR (auto:\n\
\x20                       ~/.cache/oxinfer/prompt-cache) across requests and restarts\n\
\x20 --prompt-cache-gb F   size cap of the prompt cache (default 20; least recently used go)\n\
\x20 --threads N           CPU threads (default: all cores)\n\
\x20 --parallel N          conversations each chat model runs at once, each with its own caches\n\
\x20                       (default 4; prompts interleaved with the others' decoding, decode\n\
\x20                       steps batched; 1: one request at a time). Per model: parallel=N\n\
\x20 --session-cache-gb F  memory the sessions' caches may hold together (default auto: a quarter\n\
\x20                       of RAM; none: no limit); beyond it the least recently used idle\n\
\x20                       sessions let go of theirs (stored in the prompt cache first, if set)\n\
\x20 --no-remote-media     do not fetch http(s) image / audio / video URLs (data: URLs only)\n\
\x20 --local-media         also accept local paths and file:// URLs as media (trusted clients only)\n\
\x20 --tls-cert FILE --tls-key FILE  HTTPS with this certificate chain and key (PEM)\n\
\x20 --self-signed         HTTPS with a generated self-signed certificate\n\
\x20 --workers N           connections served at once (default 32)\n\
\x20 --thinking-key KEY    key of the signatures of /v1/messages thinking blocks (default: random\n\
\x20                       per start; set it to keep them valid across restarts)\n\
\x20 --no-update-check     do not ask GitHub for the latest release (the GUI tells when a newer\n\
\x20                       oxinfer is out, checking at most every 6 hours, and can install it)\n\
\x20 --quiet               no request log";

const DEFAULT_PORT: u16 = 8080;

/// Opens `url` in the default browser (best effort).
fn open_browser(url: &str) {
    #[cfg(target_os = "macos")]
    let r = std::process::Command::new("open").arg(url).spawn();
    #[cfg(target_os = "windows")]
    let r = std::process::Command::new("cmd").args(["/C", "start", "", url]).spawn();
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let r = std::process::Command::new("xdg-open").arg(url).spawn();
    if let Err(e) = r {
        eprintln!("[could not open a browser ({e}); open {url}]");
    }
}

/// Runs the server with the arguments after `serve` or `web` (`command`); `open`: whether the
/// GUI opens in the browser unless `--open` / `--no-open` says otherwise.
pub fn run(args: &[String], command: &str, open: bool) -> Result<()> {
    let mut o = ServerOptions::default();
    let (mut host, mut port, mut listen, mut open) = (None::<String>, None::<u16>, None::<String>, open);
    super::server_cli::parse_options(&mut o, args, USAGE, command, &mut |flag, val| {
        match flag {
            "host" => host = Some(val()?),
            "port" | "p" => port = Some(val()?.parse().map_err(|_| anyhow::anyhow!("--port: expected a port number"))?),
            "listen" | "l" => listen = Some(val()?),
            "no-open" | "no-browser" => open = false,
            "open" => open = true,
            _ => return Ok(false),
        }
        Ok(true)
    })?;
    if listen.is_some() && (host.is_some() || port.is_some()) {
        bail!("--listen and --host / --port: give one or the other");
    }
    // Restarted after an update from the GUI: the address it had, no new browser tab.
    let restarted = std::env::var(oxinfer::server::web::RESTART_ENV).ok().filter(|a| !a.is_empty());
    if let Some(a) = &restarted {
        // SAFETY: nothing else runs yet (no thread reads the environment).
        unsafe { std::env::remove_var(oxinfer::server::web::RESTART_ENV) };
        (listen, host, port, open) = (Some(a.clone()), None, None, false);
        // The previous process may not have let the port go yet (Windows: it exits after
        // starting this one).
        for _ in 0..100 {
            if std::net::TcpListener::bind(a.as_str()).is_ok() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        eprintln!("[restarted as oxinfer {}]", env!("CARGO_PKG_VERSION"));
    }
    o.set("web", "1")?;
    if !oxinfer::server::web::has_gui() {
        eprintln!("[this build has no web GUI files (src/server/webui); only the APIs are served]");
    }
    let plat = oxinfer::platform::detect();
    eprintln!("[platform: {}]", plat.summary());
    // The default port, or the next free one when it is taken (not when a port was asked for).
    let host = host.unwrap_or_else(|| "127.0.0.1".into());
    let candidates: Vec<String> = match (&listen, port) {
        (Some(l), _) => vec![l.clone()],
        (None, Some(p)) => vec![join(&host, p)],
        (None, None) => (0..20).map(|i| join(&host, DEFAULT_PORT + i)).collect(),
    };
    let addr = candidates
        .iter()
        .find(|a| std::net::TcpListener::bind(a.as_str()).is_ok())
        .unwrap_or(&candidates[0]);
    o.set("listen", addr)?;
    let api = Api::new(o.clone())?;
    let running = http::start(api.clone())?;
    // The GUI may update this executable, then start the new one on the same address.
    let addr = running.addr();
    oxinfer::server::web::allow_self_update(&api, Some(Box::new(move |exe| restart(exe, addr))));
    let url = running.url();
    let shown = match running.addr().ip() {
        ip if ip.is_unspecified() => url.replacen(&ip.to_string(), "localhost", 1).replace("[::]", "localhost"),
        _ => url.clone(),
    };
    if api.web_read_only() {
        eprintln!(
            "\n[WARNING: listening on {} without an API key: anyone who can reach this address can use the\n\
             loaded models through the GUI and /v1. Downloads, deletions and model loads are disabled for\n\
             everyone; start with --api-key KEY (or OXINFER_API_KEY) to manage models remotely.]\n",
            running.addr()
        );
    }
    let models: Vec<String> = api.models().into_iter().map(|(id, _, _)| id).collect();
    if !models.is_empty() {
        eprintln!("[models: {}]", models.join(", "));
    }
    eprintln!("[GUI: {shown}/]");
    eprintln!(
        "[API: {shown}/v1 (OpenAI-compatible){}]",
        if o.api_key.is_some() { ", Authorization: Bearer KEY" } else { "" }
    );
    if open {
        // The key in the fragment (never sent to the server): the page keeps it and drops it
        // from the address.
        let target = match &o.api_key {
            Some(k) => format!("{shown}/#key={}", encode(k)),
            None => format!("{shown}/"),
        };
        open_browser(&target);
    }
    running.wait()
}

/// Runs `exe` (this program's path, where an update put the new binary) in place of this process,
/// with the same arguments, on `addr` ([`oxinfer::server::web::RESTART_ENV`]). Unix: `exec`, the
/// same process (its id, its terminal); the listening socket closes with the exec (sockets are
/// close-on-exec) and the new one binds again (`SO_REUSEADDR`). Elsewhere: starts the new
/// process and exits (it waits for the port). Returns only on failure.
fn restart(exe: &std::path::Path, addr: std::net::SocketAddr) -> Result<()> {
    let mut cmd = std::process::Command::new(exe);
    cmd.args(std::env::args_os().skip(1))
        .env(oxinfer::server::web::RESTART_ENV, addr.to_string());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let e = cmd.exec();
        Err(anyhow::anyhow!("exec {}: {e}", exe.display()))
    }
    #[cfg(not(unix))]
    {
        cmd.spawn().map_err(|e| anyhow::anyhow!("start {}: {e}", exe.display()))?;
        std::process::exit(0)
    }
}

fn join(host: &str, port: u16) -> String {
    if host.contains(':') && !host.starts_with('[') {
        format!("[{host}]:{port}")
    } else {
        format!("{host}:{port}")
    }
}

/// Percent-encodes a URL fragment value.
fn encode(s: &str) -> String {
    s.bytes()
        .map(|b| {
            if b.is_ascii_alphanumeric() || b"-._~".contains(&b) {
                (b as char).to_string()
            } else {
                format!("%{b:02X}")
            }
        })
        .collect()
}
