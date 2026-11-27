//! [`Api`] over HTTP(S) with [httpsd](https://crates.io/crates/httpsd) (its thread-pool runtime):
//! streamed responses (server-sent events, raw audio) are written from their own thread, a
//! client that goes away stops the generation, and CORS headers are added when
//! [`super::ServerOptions::cors`] is set.

use super::{Api, ApiRequest, Output};
use anyhow::{Context, Result, anyhow, bail};
use httpsd::{Body, Request, Response, StatusCode};
use std::net::SocketAddr;
use std::sync::Arc;

/// A running server ([`start`]).
pub struct Running {
    addr: SocketAddr,
    https: bool,
    shutdown: httpsd::Shutdown,
    thread: Option<std::thread::JoinHandle<Result<()>>>,
}

impl Running {
    /// The address it listens on (with the port chosen when the configured one was 0).
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// `http://ADDR` or `https://ADDR`.
    pub fn url(&self) -> String {
        format!("{}://{}", if self.https { "https" } else { "http" }, self.addr)
    }

    /// Blocks until the server stops (an error, or [`Running::stop`] from another handle).
    pub fn wait(mut self) -> Result<()> {
        match self.thread.take().map(|t| t.join()) {
            Some(Ok(r)) => r,
            Some(Err(_)) => bail!("the server thread panicked"),
            None => Ok(()),
        }
    }

    /// Stops accepting connections, lets the requests in flight finish (up to httpsd's grace
    /// period) and returns.
    pub fn stop(self) -> Result<()> {
        self.shutdown.trigger();
        self.wait()
    }
}

/// Starts serving `api` on [`super::ServerOptions::listen`] (HTTPS with `tls_cert` / `tls_key`
/// or `self_signed`) in a background thread.
pub fn start(api: Arc<Api>) -> Result<Running> {
    let o = api.options().clone();
    // Resolve a port of 0 to a free port now, so that it can be reported.
    let mut addr: SocketAddr = std::net::ToSocketAddrs::to_socket_addrs(&o.listen)
        .with_context(|| format!("listen address {:?}", o.listen))?
        .next()
        .with_context(|| format!("listen address {:?} resolves to nothing", o.listen))?;
    if addr.port() == 0 {
        addr = std::net::TcpListener::bind(addr)
            .and_then(|l| l.local_addr())
            .with_context(|| format!("bind {addr}"))?;
    }
    let mut server = httpsd::Server::bind(addr)
        .map_err(|e| anyhow!("bind {addr}: {e}"))?
        .workers(o.workers.max(2))
        .max_body_bytes(o.max_body)
        .server_name(Some(format!("oxinfer/{}", env!("CARGO_PKG_VERSION"))));
    let https = match (&o.tls_cert, &o.tls_key, o.self_signed) {
        (Some(c), Some(k), _) => {
            let acc = httpsd::tls::TlsAcceptor::from_pem_files(c, k).map_err(|e| anyhow!("TLS certificate: {e}"))?;
            server = server.tls(acc);
            true
        }
        (None, None, true) => {
            let host = match addr.ip() {
                ip if ip.is_unspecified() => "localhost".to_string(),
                ip => ip.to_string(),
            };
            let names = ["localhost", host.as_str()];
            let acc = httpsd::tls::TlsAcceptor::self_signed(&names).map_err(|e| anyhow!("self-signed certificate: {e}"))?;
            server = server.tls(acc);
            true
        }
        (None, None, false) => false,
        _ => bail!("HTTPS needs both tls_cert and tls_key"),
    };
    let shutdown = httpsd::Shutdown::new();
    let (tx, rx) = std::sync::mpsc::channel();
    let server = server
        .graceful(shutdown.clone())
        .notify_bound(tx)
        .handler(move |req: &Request| handle(&api, req));
    let thread = std::thread::Builder::new()
        .name("oxinfer-http".into())
        .spawn(move || server.run().map_err(|e| anyhow!("server: {e}")))?;
    // Wait for the listener (or the thread's failure to bind).
    if rx.recv_timeout(std::time::Duration::from_secs(10)).is_err() {
        if thread.is_finished() {
            return match thread.join() {
                Ok(Err(e)) => Err(e),
                _ => bail!("the server stopped while starting"),
            };
        }
        bail!("the server did not start listening on {addr}");
    }
    Ok(Running {
        addr,
        https,
        shutdown,
        thread: Some(thread),
    })
}

fn status(code: u16) -> StatusCode {
    let reason = match code {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        409 => "Conflict",
        405 => "Method Not Allowed",
        413 => "Payload Too Large",
        415 => "Unsupported Media Type",
        499 => "Client Closed Request",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        507 => "Insufficient Storage",
        503 => "Service Unavailable",
        _ => "Error",
    };
    StatusCode::new(code, reason)
}

fn with_cors(api: &Api, mut r: Response) -> Response {
    if let Some(origin) = &api.options().cors {
        r = r
            .header("Access-Control-Allow-Origin", origin.clone())
            .header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
            .header(
                "Access-Control-Allow-Headers",
                "Authorization, Content-Type, X-Api-Key, Anthropic-Version, Anthropic-Beta",
            )
            .header("Access-Control-Max-Age", "86400");
        if origin != "*" {
            r = r.header("Vary", "Origin");
        }
    }
    r
}

fn handle(api: &Arc<Api>, req: &Request) -> Response {
    let method = req.method().as_str().to_string();
    if method == "OPTIONS" {
        return with_cors(api, Response::new(status(204)));
    }
    let h = req.headers();
    let full = match req.query() {
        Some(q) if !q.is_empty() => format!("{}?{q}", req.path()),
        _ => req.path().to_string(),
    };
    let mut r = ApiRequest::new(&method, &full, h.get("content-type"), req.body().to_vec());
    // Anthropic's clients send the key as x-api-key.
    r.authorization = h
        .get("authorization")
        .map(str::to_string)
        .or_else(|| h.get("x-api-key").map(|k| format!("Bearer {k}")));
    r.origin = h.get("origin").map(str::to_string);
    r.host = h.get("host").map(str::to_string);
    r.peer = req.peer_addr().map(|a| a.to_string());
    r.user_agent = h.get("user-agent").map(str::to_string);
    r.session = h.get("x-session-id").or_else(|| h.get("x-opencode-session-id")).map(str::to_string);
    r.forwarded = h.get("x-oxinfer-forwarded").map(str::to_string);
    let resp = match api.handle(&r) {
        Output::Full(full) => {
            let mut resp = Response::new(status(full.status)).header("Content-Type", full.content_type.clone());
            if api.options().web && !req.path().starts_with("/v1/") && !req.path().starts_with("/api/") {
                // The GUI's files: those under assets/ have their hash in their name.
                let cc = if req.path().starts_with("/assets/") {
                    "public, max-age=31536000, immutable"
                } else {
                    "no-cache"
                };
                resp = resp.header("Cache-Control", cc);
            }
            if api.options().web {
                resp = resp
                    .header("X-Content-Type-Options", "nosniff")
                    .header("Referrer-Policy", "no-referrer");
            }
            resp.body(full.body)
        }
        Output::Stream { content_type, producer } => streamed(content_type, producer),
    };
    with_cors(api, resp)
}

/// A 200 response whose body `producer` writes from its own thread.
fn streamed(content_type: String, producer: super::Producer) -> Response {
    let (tx, body) = Body::channel();
    std::thread::spawn(move || {
        producer(&mut |b: &[u8]| tx.send(b.to_vec()).is_ok());
        tx.finish();
    });
    Response::new(status(200))
        .header("Content-Type", content_type)
        .header("Cache-Control", "no-cache")
        .body(body)
}
