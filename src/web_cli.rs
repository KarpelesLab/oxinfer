//! `oxinfer web`: the browser GUI with this machine as its backend ([`oxinfer::server::web`]),
//! next to the OpenAI-compatible API of `oxinfer serve` on the same port.

use anyhow::{Result, bail};
use oxinfer::server::{Api, ServerOptions, http};

pub const USAGE: &str = "usage: oxinfer web [--host ADDR] [--port N] [--no-open] [--api-key KEY] [options]\n\
\n\
Serves the web GUI (chat, images, speech to text, text to speech, voice chat, video, embeddings;\n\
the model catalog, downloads with progress, the model cache) at http://HOST:PORT/, and the\n\
OpenAI-compatible API of `oxinfer serve` at http://HOST:PORT/v1 for other clients at the same\n\
time. Models picked in the GUI join the API's model list.\n\
\n\
options:\n\
\x20 --host ADDR           address to listen on (default 127.0.0.1)\n\
\x20 --port N              port (default 8090, or the next free one; 0 picks one)\n\
\x20 --listen ADDR:PORT    both at once\n\
\x20 --no-open             do not open the GUI in the browser\n\
\x20 --api-key KEY         require Authorization: Bearer KEY for the API and the GUI (default:\n\
\x20                       $OXINFER_API_KEY if set). Without one on a non-loopback address the\n\
\x20                       GUI's management API is read-only (no downloads, deletions or loads)\n\
\x20 --model [ID=]SOURCE   a model listed (and loadable) from the start; repeatable\n\
\x20 --no-update-check     do not ask GitHub for the latest release (the GUI tells when a newer\n\
\x20                       oxinfer is out, checking at most every 6 hours, and can install it)\n\
\x20 and the other options of `oxinfer serve` (--memory-gb, --cache-dir, --threads,\n\
\x20 --prompt-cache, --cors, --tls-cert/--tls-key, --self-signed, --preload, --quiet, ...)";

const DEFAULT_PORT: u16 = 8090;

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

/// Runs the GUI's server with the arguments after `web`.
pub fn run(args: &[String]) -> Result<()> {
    let mut o = ServerOptions::default();
    let (mut host, mut port, mut listen, mut open) = (None::<String>, None::<u16>, None::<String>, true);
    super::server_cli::parse_options(&mut o, args, USAGE, "web", &mut |flag, val| {
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
