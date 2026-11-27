//! Peers ([`super::ServerOptions::peers`]): other oxinfer servers this one sends requests to, so
//! that clients talk to one endpoint and get every machine's models and capacity (docs/cluster.md).
//!
//! Each peer is an ordinary `oxinfer serve` with an API key. This server polls its `GET
//! /v1/cluster` (its models with the requests they are running, its free memory), lists the union
//! of the models, and for each model request picks where it runs: the only machine that lists the
//! model, else the machine that holds the conversation ([`Peers::plan`]), else the least loaded
//! one. A request for a peer is forwarded as it came (plain HTTP/1.1 over TCP, the shared key in
//! `Authorization`), its response passed through as it arrives, server-sent events included. A
//! forwarded request carries [`FORWARDED`] and is never forwarded again, so two servers may name
//! each other.

use super::{Api, ApiError, ApiRequest, ApiResponse, Kind, Output, activity, learn};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{Shutdown, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// The header of a request another oxinfer server forwarded (its value: that server's address).
pub(crate) const FORWARDED: &str = "X-Oxinfer-Forwarded";

/// The longest a connection to a peer may take to open.
const CONNECT: Duration = Duration::from_secs(2);
/// The longest a peer may take to answer `GET /v1/cluster`, and to take a request's bytes.
const ANSWER: Duration = Duration::from_secs(3);
/// How often the peers are asked for their state.
pub(crate) const POLL: Duration = Duration::from_secs(2);
/// A peer that has not answered for this long has the requests forwarded to it ended.
const STALLED: Duration = Duration::from_secs(30);
/// Conversations remembered ([`Routes`]).
const MEMORY: usize = 8192;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// One peer of [`super::ServerOptions::peers`]: `ADDR[,key=KEY]`.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct PeerSpec {
    /// `HOST:PORT` of the peer's listener (`http://` in front is accepted and dropped).
    pub addr: String,
    /// The peer's API key (default: this server's own [`super::ServerOptions::api_key`]).
    pub key: Option<String>,
}

impl PeerSpec {
    /// Parses `ADDR[,key=KEY]`.
    pub fn parse(s: &str) -> anyhow::Result<Self> {
        let mut items = s.split(',').map(str::trim);
        let addr = items.next().unwrap_or_default();
        let addr = addr.strip_prefix("http://").unwrap_or(addr).trim_end_matches('/');
        if addr.starts_with("https://") {
            anyhow::bail!("peer {s:?}: peers are reached over plain HTTP (see docs/cluster.md)");
        }
        if addr.is_empty() || !addr.contains(':') {
            anyhow::bail!("peer {s:?}: expected HOST:PORT[,key=KEY]");
        }
        let mut key = None;
        for item in items.filter(|i| !i.is_empty()) {
            match item.split_once('=') {
                Some(("key", v)) if !v.is_empty() => key = Some(v.to_string()),
                _ => anyhow::bail!("peer {s:?}: unknown option {item:?} (key=KEY)"),
            }
        }
        Ok(Self {
            addr: addr.to_string(),
            key,
        })
    }

    /// From a configuration file entry: a spec string, or `{"addr": ..., "key": ...}`.
    pub fn from_json(v: &Value) -> anyhow::Result<Self> {
        if let Some(s) = v.as_str() {
            return Self::parse(s);
        }
        let o = v
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("peer: expected a string or an object"))?;
        let addr = o
            .get("addr")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("peer: no \"addr\""))?;
        let mut p = Self::parse(addr)?;
        for (k, v) in o {
            match (k.as_str(), v.as_str()) {
                ("addr", _) => {}
                ("key", Some(key)) if !key.is_empty() => p.key = Some(key.to_string()),
                _ => anyhow::bail!("peer: unknown key {k:?} (addr, key)"),
            }
        }
        Ok(p)
    }
}

/// A model of a peer, from its `GET /v1/cluster`.
#[derive(Clone, Debug)]
struct PeerModel {
    id: String,
    loaded: bool,
    /// Requests it is running or has waiting.
    requests: usize,
    /// Conversations it runs at once.
    parallel: usize,
    profiles: Vec<String>,
    /// Its `/v1/models` entry.
    entry: Value,
}

/// What a peer last said of itself.
#[derive(Clone, Debug, Default)]
struct Info {
    version: String,
    models: Vec<PeerModel>,
    auto_models: bool,
    /// Memory it can still give to models, and its budget for them.
    free: Option<u64>,
    budget: Option<u64>,
}

#[derive(Debug, Default)]
struct State {
    info: Option<Info>,
    /// Usable: it answered its last poll, with the key, and refuses requests without one.
    up: bool,
    /// Why it is not usable (one line; logged once per change).
    error: Option<String>,
    /// When it last answered, and since when it does not.
    seen: Option<Instant>,
    silent_since: Option<Instant>,
    /// Checked to refuse requests without a key (again after each outage).
    key_checked: bool,
}

pub(crate) struct Peer {
    pub addr: String,
    key: String,
    st: Mutex<State>,
    /// Requests forwarded to it and not finished, by model.
    inflight: Mutex<HashMap<String, usize>>,
    /// Their connections, to end them when the peer is gone.
    conns: Mutex<HashMap<u64, TcpStream>>,
    forwarded: AtomicU64,
}

/// Where conversations ran last: a key of the conversation ([`conversation_key`]) to its machine
/// (0: this one, `i + 1`: peer `i`).
#[derive(Default)]
struct Routes {
    last: HashMap<u64, (usize, u64)>,
    /// Requests of each conversation in flight.
    inflight: HashMap<u64, usize>,
    tick: u64,
}

pub(crate) struct Peers {
    list: Vec<Arc<Peer>>,
    routes: Mutex<Routes>,
    next_conn: AtomicU64,
    /// This server's address, as [`FORWARDED`] names it.
    me: String,
    log: bool,
}

/// Where a request may run, in the order to try.
pub(crate) struct Plan {
    /// 0: this machine, `i + 1`: peer `i`.
    pub order: Vec<usize>,
    /// The model's id without its profile.
    model: String,
    key: Option<u64>,
    /// The model is listed nowhere and may be listed on request: a machine that cannot take it
    /// (no memory, no disk, cannot fetch it) passes it on.
    pub auto: bool,
}

/// Counts a conversation's request in flight until dropped.
pub(crate) struct KeyGuard {
    peers: Arc<Peers>,
    key: Option<u64>,
}

impl Drop for KeyGuard {
    fn drop(&mut self) {
        if let Some(k) = self.key {
            let mut r = lock(&self.peers.routes);
            if let Some(n) = r.inflight.get_mut(&k) {
                *n -= 1;
                if *n == 0 {
                    r.inflight.remove(&k);
                }
            }
        }
    }
}

/// A request forwarded to a peer, until its response is done.
struct Flight {
    peer: Arc<Peer>,
    model: String,
    conn: u64,
    stream: TcpStream,
}

impl Drop for Flight {
    fn drop(&mut self) {
        // (Every clone of the socket: the peer sees the connection end and stops the request.)
        let _ = self.stream.shutdown(Shutdown::Both);
        lock(&self.peer.conns).remove(&self.conn);
        let mut f = lock(&self.peer.inflight);
        if let Some(n) = f.get_mut(&self.model) {
            *n -= 1;
            if *n == 0 {
                f.remove(&self.model);
            }
        }
    }
}

// ---- a small HTTP/1.1 client ------------------------------------------------------------------

fn io_err(msg: impl Into<String>) -> std::io::Error {
    std::io::Error::other(msg.into())
}

fn connect(addr: &str) -> std::io::Result<TcpStream> {
    let mut last = io_err(format!("{addr} resolves to nothing"));
    for a in addr.to_socket_addrs()? {
        match TcpStream::connect_timeout(&a, CONNECT) {
            Ok(s) => {
                s.set_nodelay(true)?;
                s.set_write_timeout(Some(ANSWER))?;
                return Ok(s);
            }
            Err(e) => last = e,
        }
    }
    Err(last)
}

enum Framing {
    Length(usize),
    Chunked { left: usize },
    Close,
    Done,
}

/// A response whose head was read.
struct Reply {
    status: u16,
    content_type: String,
    r: BufReader<TcpStream>,
    framing: Framing,
}

fn read_line(r: &mut BufReader<TcpStream>) -> std::io::Result<String> {
    let mut line = Vec::new();
    r.by_ref().take(16 << 10).read_until(b'\n', &mut line)?;
    if !line.ends_with(b"\n") {
        return Err(io_err("the connection ended in a header"));
    }
    Ok(String::from_utf8_lossy(&line).trim_end().to_string())
}

/// Sends a request and reads the response's head.
fn request(mut s: TcpStream, method: &str, target: &str, headers: &[(&str, &str)], body: &[u8]) -> std::io::Result<Reply> {
    let mut head = format!(
        "{method} {target} HTTP/1.1\r\nConnection: close\r\nContent-Length: {}\r\n",
        body.len()
    );
    for (k, v) in headers {
        // (Header values come from this server's options and the client's request line.)
        if !v.contains(['\r', '\n']) {
            head.push_str(&format!("{k}: {v}\r\n"));
        }
    }
    head.push_str("\r\n");
    s.write_all(head.as_bytes())?;
    s.write_all(body)?;
    s.flush()?;
    let mut r = BufReader::with_capacity(64 << 10, s);
    let status_line = read_line(&mut r)?;
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|c| c.parse().ok())
        .ok_or_else(|| io_err(format!("not an HTTP response: {status_line:?}")))?;
    let (mut content_type, mut framing) = (String::new(), Framing::Close);
    for _ in 0..200 {
        let line = read_line(&mut r)?;
        if line.is_empty() {
            return Ok(Reply {
                status,
                content_type,
                r,
                framing,
            });
        }
        let Some((k, v)) = line.split_once(':') else { continue };
        let v = v.trim();
        match k.to_ascii_lowercase().as_str() {
            "content-type" => content_type = v.to_string(),
            "content-length" => {
                if let (Framing::Close, Ok(n)) = (&framing, v.parse()) {
                    framing = Framing::Length(n);
                }
            }
            "transfer-encoding" if v.to_ascii_lowercase().contains("chunked") => framing = Framing::Chunked { left: 0 },
            _ => {}
        }
    }
    Err(io_err("too many response headers"))
}

impl Reply {
    /// Appends the body bytes that have arrived (waiting for some); false at the body's end.
    fn read(&mut self, out: &mut Vec<u8>) -> std::io::Result<bool> {
        loop {
            let left = match &mut self.framing {
                Framing::Done => return Ok(false),
                Framing::Length(0) => {
                    self.framing = Framing::Done;
                    return Ok(false);
                }
                Framing::Length(n) => n,
                Framing::Close => {
                    let buf = self.r.fill_buf()?;
                    if buf.is_empty() {
                        self.framing = Framing::Done;
                        return Ok(false);
                    }
                    let n = buf.len();
                    out.extend_from_slice(buf);
                    self.r.consume(n);
                    return Ok(true);
                }
                Framing::Chunked { left } if *left == 0 => {
                    let line = read_line(&mut self.r)?;
                    let size = line.split(';').next().unwrap_or_default().trim();
                    let n = usize::from_str_radix(size, 16).map_err(|_| io_err(format!("bad chunk size {line:?}")))?;
                    if n == 0 {
                        // The trailers, to the empty line.
                        while !read_line(&mut self.r)?.is_empty() {}
                        self.framing = Framing::Done;
                        return Ok(false);
                    }
                    *left = n;
                    continue;
                }
                Framing::Chunked { left } => left,
            };
            let buf = self.r.fill_buf()?;
            if buf.is_empty() {
                return Err(io_err("the peer closed the connection in the middle of its response"));
            }
            let n = buf.len().min(*left);
            out.extend_from_slice(&buf[..n]);
            self.r.consume(n);
            *left -= n;
            if matches!(self.framing, Framing::Chunked { left: 0 }) {
                let mut crlf = [0u8; 2];
                self.r.read_exact(&mut crlf)?;
            }
            return Ok(true);
        }
    }

    fn read_all(&mut self, cap: usize) -> std::io::Result<Vec<u8>> {
        let mut out = Vec::new();
        while self.read(&mut out)? {
            if out.len() > cap {
                return Err(io_err("the peer's response is too large"));
            }
        }
        Ok(out)
    }
}

// ---- the peers' state ------------------------------------------------------------------------

impl Peer {
    /// `GET /v1/cluster`: the status and the body, or the error with whether the peer took the
    /// connection.
    fn get_cluster(&self, with_key: bool) -> Result<(u16, Vec<u8>), (bool, std::io::Error)> {
        let s = connect(&self.addr).map_err(|e| (false, e))?;
        let run = || -> std::io::Result<(u16, Vec<u8>)> {
            s.set_read_timeout(Some(ANSWER))?;
            let auth = format!("Bearer {}", self.key);
            let mut headers = vec![("Host", self.addr.as_str()), (FORWARDED, "poll")];
            if with_key {
                headers.push(("Authorization", auth.as_str()));
            }
            let mut r = request(s.try_clone()?, "GET", "/v1/cluster", &headers, &[])?;
            Ok((r.status, r.read_all(4 << 20)?))
        };
        run().map_err(|e| (true, e))
    }

    /// Asks the peer for its state; returns what changed, to log (`Some(None)`: it is back).
    fn poll(&self) -> Option<Option<String>> {
        let fail = |st: &mut State, connected: bool, msg: String| {
            let now = Instant::now();
            st.silent_since.get_or_insert(now);
            st.key_checked &= connected;
            let was = st.error.replace(msg.clone());
            st.up = false;
            (was.as_deref() != Some(msg.as_str())).then_some(Some(msg))
        };
        let (status, body) = match self.get_cluster(true) {
            Ok(r) => r,
            Err((connected, e)) => {
                let change = fail(&mut lock(&self.st), connected, format!("unreachable ({e})"));
                // Gone (nothing listens there), or silent for too long: the requests forwarded
                // to it will get no answer.
                let stalled = lock(&self.st).silent_since.is_some_and(|t| t.elapsed() > STALLED);
                if !connected || stalled {
                    self.end_flights();
                }
                return change;
            }
        };
        let mut st = lock(&self.st);
        if status == 401 || status == 403 {
            return fail(
                &mut st,
                true,
                "refuses this server's key (give the peer's --api-key as key=KEY)".into(),
            );
        }
        let info = serde_json::from_slice::<Value>(&body)
            .ok()
            .filter(|_| status == 200)
            .and_then(|v| parse_info(&v));
        let Some(info) = info else {
            return fail(
                &mut st,
                true,
                format!("answered GET /v1/cluster with status {status}: not an oxinfer server with peers support?"),
            );
        };
        if !st.key_checked {
            drop(st);
            // The link is not an open door: a peer that answers without the key is not used.
            let open = matches!(self.get_cluster(false), Ok((200, _)));
            st = lock(&self.st);
            if open {
                return fail(
                    &mut st,
                    true,
                    "answers requests without an API key: start it with --api-key KEY (it is not used until then)".into(),
                );
            }
            st.key_checked = true;
        }
        st.info = Some(info);
        st.seen = Some(Instant::now());
        st.silent_since = None;
        let was_down = !st.up;
        st.up = true;
        st.error = None;
        was_down.then_some(None)
    }

    fn end_flights(&self) {
        for s in lock(&self.conns).values() {
            let _ = s.shutdown(Shutdown::Both);
        }
    }

    /// Whether the peer, which does not answer now, listed the model when it last did.
    fn had(&self, id: &str) -> bool {
        let st = lock(&self.st);
        !st.up && st.info.as_ref().is_some_and(|i| i.models.iter().any(|m| m.id == id))
    }

    fn inflight(&self, model: &str) -> usize {
        lock(&self.inflight).get(model).copied().unwrap_or(0)
    }

    fn model(&self, id: &str) -> Option<PeerModel> {
        let st = lock(&self.st);
        if !st.up {
            return None;
        }
        st.info.as_ref()?.models.iter().find(|m| m.id == id).cloned()
    }

    fn status_json(&self) -> Value {
        let st = lock(&self.st);
        let info = st.info.as_ref();
        json!({
            "addr": self.addr,
            "up": st.up,
            "error": st.error,
            "version": info.map(|i| i.version.clone()),
            "seen_secs": st.seen.map(|t| t.elapsed().as_secs()),
            "memory_free": info.and_then(|i| i.free),
            "memory_budget": info.and_then(|i| i.budget),
            "auto_models": info.map(|i| i.auto_models),
            "forwarded": self.forwarded.load(Ordering::Relaxed),
            "inflight": lock(&self.inflight).values().sum::<usize>(),
            "models": info.map(|i| i.models.iter().map(|m| json!({
                "id": m.id,
                "kind": m.entry.get("kind"),
                "loaded": m.loaded,
                "requests": m.requests,
            })).collect::<Vec<_>>()).unwrap_or_default(),
        })
    }
}

fn parse_info(v: &Value) -> Option<Info> {
    if v.get("object")?.as_str()? != "cluster.node" {
        return None;
    }
    let models = v
        .get("models")?
        .as_array()?
        .iter()
        .filter_map(|m| {
            Some(PeerModel {
                id: m.get("id")?.as_str()?.to_string(),
                loaded: m.get("loaded").and_then(Value::as_bool).unwrap_or(false),
                requests: m.get("requests").and_then(Value::as_u64).unwrap_or(0) as usize,
                parallel: m.get("parallel").and_then(Value::as_u64).unwrap_or(1).max(1) as usize,
                profiles: m
                    .get("profiles")
                    .and_then(Value::as_array)
                    .map(|a| a.iter().filter_map(|p| p.as_str().map(str::to_string)).collect())
                    .unwrap_or_default(),
                entry: m.clone(),
            })
        })
        .collect();
    Some(Info {
        version: v.get("version").and_then(Value::as_str).unwrap_or_default().to_string(),
        models,
        auto_models: v.get("auto_models").and_then(Value::as_bool).unwrap_or(false),
        free: v.pointer("/memory/free").and_then(Value::as_u64),
        budget: v.pointer("/memory/budget").and_then(Value::as_u64),
    })
}

fn fnv(h: &mut u64, bytes: &[u8]) {
    for b in bytes {
        *h ^= *b as u64;
        *h = h.wrapping_mul(0x100000001b3);
    }
    // (A separator, so that ("ab", "c") and ("a", "bc") differ.)
    *h ^= 0xff;
    *h = h.wrapping_mul(0x100000001b3);
}

/// What tells a conversation from the others across its requests: the name its client gives it
/// (`X-Session-Id`), else how it starts (the system prompt and the first message, which every
/// later request of the conversation repeats).
fn conversation_key(req: &ApiRequest, path: &str, model: &str) -> Option<u64> {
    let mut h = 0xcbf29ce484222325u64;
    fnv(&mut h, model.as_bytes());
    if let Some(name) = learn::session_name(req.session.as_deref()) {
        fnv(&mut h, b"session");
        fnv(&mut h, name.as_bytes());
        return Some(h);
    }
    if !matches!(path, "/v1/chat/completions" | "/v1/messages") {
        return None;
    }
    let v: Value = serde_json::from_slice(&req.body).ok()?;
    let messages = v.get("messages")?.as_array()?;
    if let Some(sys) = v.get("system") {
        fnv(&mut h, sys.to_string().as_bytes());
    }
    // Up to the first message that is not the system's.
    let mut any = false;
    for m in messages {
        fnv(&mut h, m.to_string().as_bytes());
        any = true;
        if !matches!(m.get("role").and_then(Value::as_str), Some("system" | "developer")) {
            break;
        }
    }
    any.then_some(h)
}

impl Peers {
    pub fn new(opts: &super::ServerOptions) -> anyhow::Result<Arc<Self>> {
        let mut list = Vec::new();
        for p in &opts.peers {
            let key = p.key.clone().or_else(|| opts.api_key.clone()).ok_or_else(|| {
                anyhow::anyhow!(
                    "peer {}: no key. Peers only take requests with their API key: give it as --peer {},key=KEY (or start both servers with the same --api-key)",
                    p.addr,
                    p.addr
                )
            })?;
            if list.iter().any(|q: &Arc<Peer>| q.addr == p.addr) {
                anyhow::bail!("peer {} is listed twice", p.addr);
            }
            list.push(Arc::new(Peer {
                addr: p.addr.clone(),
                key,
                st: Mutex::new(State::default()),
                inflight: Mutex::new(HashMap::new()),
                conns: Mutex::new(HashMap::new()),
                forwarded: AtomicU64::new(0),
            }));
        }
        Ok(Arc::new(Self {
            list,
            routes: Mutex::new(Routes::default()),
            next_conn: AtomicU64::new(1),
            me: opts.listen.clone(),
            log: opts.log,
        }))
    }

    pub fn any(&self) -> bool {
        !self.list.is_empty()
    }

    /// Asks every peer for its state (each within [`CONNECT`] + [`ANSWER`]), at once.
    pub fn refresh(&self) {
        std::thread::scope(|s| {
            for p in &self.list {
                s.spawn(move || match p.poll() {
                    Some(Some(why)) => eprintln!("[WARNING: peer {} {why}]", p.addr),
                    Some(None) if self.log => eprintln!("[peer {} is up]", p.addr),
                    _ => {}
                });
            }
        });
    }

    /// The peers that cannot be used now, one line each (`/health`).
    pub fn warnings(&self) -> Vec<String> {
        self.list
            .iter()
            .filter_map(|p| lock(&p.st).error.as_ref().map(|e| format!("peer {} {e}", p.addr)))
            .collect()
    }

    pub fn status_json(&self) -> Value {
        Value::Array(self.list.iter().map(|p| p.status_json()).collect())
    }

    /// The peers' models as `/v1/models` entries, with the peers that list each: `(id, entry,
    /// addresses)`, in the peers' order.
    pub fn models(&self) -> Vec<(String, Value, Vec<String>)> {
        let mut out: Vec<(String, Value, Vec<String>)> = Vec::new();
        for p in &self.list {
            let st = lock(&p.st);
            let Some(info) = st.info.as_ref().filter(|_| st.up) else { continue };
            for m in &info.models {
                match out.iter_mut().find(|(id, ..)| *id == m.id) {
                    Some((_, _, at)) => at.push(p.addr.clone()),
                    None => out.push((m.id.clone(), m.entry.clone(), vec![p.addr.clone()])),
                }
            }
        }
        out
    }

    /// Where the request may run (None: here, as without peers): see the module's notes.
    pub fn plan(&self, api: &Api, req: &ApiRequest, path: &str) -> Option<Plan> {
        if self.list.is_empty() || req.forwarded || !req.method.eq_ignore_ascii_case("POST") || path.starts_with("/v1/videos") {
            return None;
        }
        let name = super::requested_model(req)?;
        // (`MODEL@PROFILE`, unless a machine lists that very name.)
        let listed = |id: &str| api.models.find(id).is_ok() || self.list.iter().any(|p| p.model(id).is_some());
        let (model, profile) = match learn::base_model(&name) {
            base if base != name && !listed(&name) => (base.to_string(), Some(name[base.len() + 1..].to_string())),
            _ => (name.clone(), None),
        };
        let local = api.models.status().into_iter().find(|m| m.spec.id == model);
        let remote: Vec<Option<PeerModel>> = self.list.iter().map(|p| p.model(&model)).collect();
        if local.is_none() && remote.iter().all(Option::is_none) {
            // Only a peer that does not answer now has it: an error that says so, not a 404.
            if self.list.iter().any(|p| p.had(&model)) {
                return Some(Plan {
                    order: Vec::new(),
                    model,
                    key: None,
                    auto: false,
                });
            }
            return self.plan_auto(api, req, &model);
        }
        if remote.iter().all(Option::is_none) {
            return None;
        }
        // A profile lives in one machine's learn directory: always the same machine, the first
        // that has the profile, else the first that lists the model.
        if let Some(profile) = &profile {
            let has = |i: usize| match i {
                0 => local.is_some() && api.learn.profile_names(&model).contains(profile),
                i => remote[i - 1].as_ref().is_some_and(|m| m.profiles.contains(profile)),
            };
            let lists = |i: usize| if i == 0 { local.is_some() } else { remote[i - 1].is_some() };
            let all = 0..=self.list.len();
            let at = all.clone().find(|&i| has(i)).or_else(|| all.clone().find(|&i| lists(i)))?;
            return Some(Plan {
                order: vec![at],
                model,
                key: None,
                auto: false,
            });
        }
        // (class, requests, machine): a loaded model with a free session first, then one to
        // load, then one whose requests would wait; fewer requests first; this machine first.
        let mut cands: Vec<(u8, usize, usize)> = Vec::new();
        let class = |loaded: bool, n: usize, parallel: usize| match (loaded, n < parallel) {
            (true, true) => 0,
            (false, _) => 1,
            (true, false) => 2,
        };
        if let Some(m) = &local {
            let n = api
                .activity
                .requests()
                .iter()
                .filter(|r| r.peer.is_none() && r.model.as_deref().is_some_and(|x| x == model || learn::base_model(x) == model))
                .count();
            cands.push((class(m.loaded, n, parallel_of(api, m)), n, 0));
        }
        for (i, m) in remote.iter().enumerate() {
            if let Some(m) = m {
                let n = m.requests.max(self.list[i].inflight(&model));
                cands.push((class(m.loaded, n, m.parallel), n, i + 1));
            }
        }
        cands.sort();
        let key = conversation_key(req, path, &model);
        // The machine that holds the conversation's cache, while it can take it now and no
        // request of the same conversation is running (which would hold that cache).
        if let Some(k) = key {
            let r = lock(&self.routes);
            if let Some(&(at, _)) = r.last.get(&k)
                && !r.inflight.contains_key(&k)
                && let Some(pos) = cands.iter().position(|c| c.2 == at && c.0 == 0)
            {
                let c = cands.remove(pos);
                cands.insert(0, c);
            }
        }
        Some(Plan {
            order: cands.into_iter().map(|c| c.2).collect(),
            model,
            key,
            auto: false,
        })
    }

    /// A model no machine lists, which requests may list ([`super::ServerOptions::auto_models`]):
    /// this machine when the model fits in its free memory, else the peer with the most free
    /// memory that it fits in, else the machines that would have to unload idle models for it.
    /// A model whose size is not known here (its files are not on this machine's disk) is tried
    /// here first.
    fn plan_auto(&self, api: &Api, req: &ApiRequest, model: &str) -> Option<Plan> {
        if !api.may_admit(req) || !super::registry::auto_name(model) {
            return None;
        }
        let need = api.models.estimate_id(model);
        // (fits now, fits the budget, free memory, machine)
        let mut cands: Vec<(bool, bool, u64, usize)> = Vec::new();
        let fits = |free: Option<u64>, budget: Option<u64>| (free.is_none_or(|f| need <= f), budget.is_none_or(|b| need <= b));
        let (now, ever) = fits(api.memory_free(), api.models.budget());
        cands.push((now, ever, u64::MAX, 0));
        for (i, p) in self.list.iter().enumerate() {
            let st = lock(&p.st);
            if let Some(info) = st.info.as_ref().filter(|i| st.up && i.auto_models) {
                let (now, ever) = fits(info.free, info.budget);
                cands.push((now, ever, info.free.unwrap_or(0), i + 1));
            }
        }
        if cands.len() < 2 {
            return None;
        }
        cands.sort_by(|a, b| (b.0, b.1, b.2).cmp(&(a.0, a.1, a.2)).then(a.3.cmp(&b.3)));
        Some(Plan {
            order: cands.into_iter().map(|c| c.3).collect(),
            model: model.to_string(),
            key: None,
            auto: true,
        })
    }

    /// Remembers where the plan's conversation runs, and counts its request until the guard is
    /// dropped.
    pub fn begin(self: &Arc<Self>, plan: &Plan, at: usize) -> KeyGuard {
        if let Some(k) = plan.key {
            let mut r = lock(&self.routes);
            r.tick += 1;
            let tick = r.tick;
            r.last.insert(k, (at, tick));
            *r.inflight.entry(k).or_insert(0) += 1;
            if r.last.len() > MEMORY {
                // The older half goes.
                let mut ticks: Vec<u64> = r.last.values().map(|v| v.1).collect();
                ticks.sort_unstable();
                let cut = ticks[ticks.len() / 2];
                r.last.retain(|_, v| v.1 > cut);
            }
        }
        KeyGuard {
            peers: self.clone(),
            key: plan.key,
        }
    }

    /// Forwards the request to peer `at - 1`. `Err(None)`: the peer could not be reached (or
    /// refused the key) before anything of a response arrived, so another machine may take the
    /// request.
    pub fn forward(self: &Arc<Self>, plan: &Plan, at: usize, req: &ApiRequest, path: &str) -> Result<Output, Option<ApiError>> {
        let peer = self.list[at - 1].clone();
        let unreachable = |e: &dyn std::fmt::Display| {
            let msg = format!("unreachable ({e})");
            let mut st = lock(&peer.st);
            st.up = false;
            st.silent_since.get_or_insert_with(Instant::now);
            if st.error.as_deref() != Some(msg.as_str()) {
                eprintln!("[WARNING: peer {} {msg}]", peer.addr);
                st.error = Some(msg);
            }
        };
        let stream = connect(&peer.addr).map_err(|e| {
            unreachable(&e);
            None
        })?;
        let conn = self.next_conn.fetch_add(1, Ordering::Relaxed);
        if let Ok(c) = stream.try_clone() {
            lock(&peer.conns).insert(conn, c);
        }
        *lock(&peer.inflight).entry(plan.model.clone()).or_insert(0) += 1;
        let ours = stream
            .try_clone()
            .map_err(|e| Some(ApiError::new(500, format!("peer {}: {e}", peer.addr))))?;
        let flight = Flight {
            peer: peer.clone(),
            model: plan.model.clone(),
            conn,
            stream: ours,
        };
        // A cancel (the status page's) ends the connection, and with it the peer's work.
        let task = activity::current();
        if let Some(t) = &task {
            activity::set_peer(t, &peer.addr);
            if let Ok(c) = stream.try_clone() {
                t.on_cancel(Box::new(move || {
                    let _ = c.shutdown(Shutdown::Both);
                }));
            }
        }
        let target = if req.query.is_empty() {
            path.to_string()
        } else {
            format!("{path}?{}", req.query)
        };
        let auth = format!("Bearer {}", peer.key);
        let mut headers = vec![
            ("Host", peer.addr.as_str()),
            ("Authorization", auth.as_str()),
            (FORWARDED, self.me.as_str()),
        ];
        if let Some(ct) = &req.content_type {
            headers.push(("Content-Type", ct));
        }
        if let Some(s) = &req.session {
            headers.push(("X-Session-Id", s));
        }
        if let Some(ua) = &req.user_agent {
            headers.push(("User-Agent", ua));
        }
        let cancelled = || task.as_ref().is_some_and(|t| t.cancelled());
        let mut reply = match request(stream, &req.method, &target, &headers, &req.body) {
            Ok(r) => r,
            Err(_) if cancelled() => return Err(Some(activity::cancelled_error())),
            Err(e) => {
                // (The request may have started there: not one to run again elsewhere.)
                return Err(Some(
                    ApiError::new(502, format!("peer {}: no response ({e})", peer.addr)).code("peer_unavailable"),
                ));
            }
        };
        if reply.status == 401 || reply.status == 403 {
            let msg = "refuses this server's key (give the peer's --api-key as key=KEY)";
            let mut st = lock(&peer.st);
            st.up = false;
            if st.error.as_deref() != Some(msg) {
                eprintln!("[WARNING: peer {} {msg}]", peer.addr);
                st.error = Some(msg.into());
            }
            return Err(None);
        }
        peer.forwarded.fetch_add(1, Ordering::Relaxed);
        let streamed = reply.status == 200 && !matches!(reply.framing, Framing::Length(_));
        if !streamed {
            let body = match reply.read_all(1 << 30) {
                Ok(b) => b,
                Err(_) if cancelled() => return Err(Some(activity::cancelled_error())),
                Err(e) => {
                    return Err(Some(
                        ApiError::new(502, format!("peer {}: its response ended early ({e})", peer.addr)).code("peer_unavailable"),
                    ));
                }
            };
            drop(flight);
            return Ok(Output::Full(ApiResponse {
                status: reply.status,
                content_type: reply.content_type,
                body,
            }));
        }
        let events = reply.content_type.starts_with("text/event-stream");
        let anthropic = super::openai::is_anthropic_path(path);
        let kind = activity::kind_of(path).unwrap_or("chat");
        let addr = peer.addr.clone();
        Ok(Output::Stream {
            content_type: reply.content_type.clone(),
            producer: Box::new(move |sink| {
                let _flight = flight;
                let mut buf = Vec::with_capacity(16 << 10);
                loop {
                    buf.clear();
                    match reply.read(&mut buf) {
                        Ok(true) => {
                            if !sink(&buf) {
                                break;
                            }
                        }
                        Ok(false) => break,
                        Err(e) => {
                            if events {
                                let e = ApiError::new(502, format!("peer {addr}: its response ended early ({e})")).code("peer_unavailable");
                                sink(super::error_event(&e, kind, anthropic).as_bytes());
                            }
                            break;
                        }
                    }
                }
            }),
        })
    }
}

/// The requests a model runs at once: a chat model's sessions ([`super::ServerOptions::parallel`]),
/// one for the other kinds.
fn parallel_of(api: &Api, m: &super::registry::EntryStatus) -> usize {
    if m.kind.is_some_and(|k| k != Kind::Chat) {
        return 1;
    }
    m.spec
        .option("parallel")
        .and_then(|p| p.parse::<usize>().ok())
        .unwrap_or(api.opts.parallel)
        .max(1)
}

/// Polls the peers until the server is gone.
pub(crate) fn spawn_poller(api: &Arc<Api>) {
    if !api.peers.any() {
        return;
    }
    let weak = Arc::downgrade(api);
    let _ = std::thread::Builder::new().name("oxinfer-peers".into()).spawn(move || {
        loop {
            std::thread::sleep(POLL);
            let Some(api) = weak.upgrade() else { break };
            api.peers.refresh();
        }
    });
}

/// `GET /v1/cluster`: what this machine tells the servers that name it as a peer.
pub(crate) fn node_json(api: &Api) -> Value {
    let requests = api.activity.requests();
    let status = api.models.status();
    let mut entries = api.models_local_json();
    for m in &mut entries {
        let id = m["id"].as_str().unwrap_or_default().to_string();
        let st = status.iter().find(|s| s.spec.id == id);
        let n = requests
            .iter()
            .filter(|r| r.peer.is_none() && r.model.as_deref().is_some_and(|x| x == id || learn::base_model(x) == id))
            .count();
        m["requests"] = json!(n);
        m["parallel"] = json!(st.map_or(1, |s| parallel_of(api, s)));
        m["memory"] = json!(st.map_or(0, |s| s.bytes));
    }
    json!({
        "object": "cluster.node",
        "version": env!("CARGO_PKG_VERSION"),
        "auto_models": api.opts.auto_models && api.admin(),
        "memory": {
            "budget": api.models.budget(),
            "models": status.iter().map(|m| m.bytes).sum::<u64>(),
            "free": api.memory_free(),
        },
        "models": entries,
        "peers": api.peers.status_json(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peer_spec_syntax() {
        let p = PeerSpec::parse("10.100.80.2:8080,key=k1").unwrap();
        assert_eq!((p.addr.as_str(), p.key.as_deref()), ("10.100.80.2:8080", Some("k1")));
        assert_eq!(PeerSpec::parse("http://h:1/").unwrap().addr, "h:1");
        assert!(PeerSpec::parse("https://h:1").is_err());
        assert!(PeerSpec::parse("nohost").is_err());
        assert!(PeerSpec::parse("h:1,tls=1").is_err());
        let p = PeerSpec::from_json(&json!({"addr": "h:2", "key": "x"})).unwrap();
        assert_eq!((p.addr.as_str(), p.key.as_deref()), ("h:2", Some("x")));
        assert!(PeerSpec::from_json(&json!({"addr": "h:2", "port": 1})).is_err());
    }

    #[test]
    fn conversation_keys() {
        let chat = |body: &str, session: Option<&str>| {
            let mut r = ApiRequest::post_json("/v1/chat/completions", body);
            r.session = session.map(str::to_string);
            conversation_key(&r, "/v1/chat/completions", "m")
        };
        let first = r#"{"model":"m","messages":[{"role":"system","content":"s"},{"role":"user","content":"a"}]}"#;
        let later = r#"{"model":"m","messages":[{"role":"system","content":"s"},{"role":"user","content":"a"},{"role":"assistant","content":"b"},{"role":"user","content":"c"}]}"#;
        let other = r#"{"model":"m","messages":[{"role":"system","content":"s"},{"role":"user","content":"z"}]}"#;
        assert!(chat(first, None).is_some());
        assert_eq!(chat(first, None), chat(later, None));
        assert_ne!(chat(first, None), chat(other, None));
        // The client's own name for the conversation wins over its content.
        assert_eq!(chat(first, Some("ses_1")), chat(other, Some("ses_1")));
        assert_ne!(chat(first, Some("ses_1")), chat(first, Some("ses_2")));
        let r = ApiRequest::post_json("/v1/embeddings", r#"{"model":"m","input":"x"}"#);
        assert_eq!(conversation_key(&r, "/v1/embeddings", "m"), None);
    }
}
