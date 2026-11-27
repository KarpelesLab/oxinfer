//! One model across two machines (docs/cluster.md): the first layers run here, the rest on a
//! worker (`oxinfer split-worker`), the hidden states of each pass crossing the link.
//!
//! The front (the engine option `split=HOST:PORT`) loads the embeddings and layers `0..k` of a
//! Qwen-path model ([`crate::qwen::QwenModel`]), the worker layers `k..L`, the final norm and the
//! LM head. Each side keeps the caches of its own layers. A pass on the front ends with the
//! residual stream after layer `k - 1`, sent to the worker as raw `f32` rows (little-endian, as
//! the host holds them: nothing is rounded on the way, so both halves compute what one machine
//! would); the worker answers with the logits, or the final hidden states, the caller asked for.
//! A prompt chunk nothing reads the hidden states of is not waited for: the front starts the
//! next chunk while the worker runs this one (the answers are read, in order, before the next
//! one that is needed).
//!
//! The link is plain TCP with a shared key (no TLS: see docs/cluster.md), one connection per
//! session, which lives on the worker for as long as its connection. Every pass names the
//! position it starts at, and the worker cuts its session back to there first: a session
//! rewound on the front needs no message of its own.
//!
//! Wire format (integers little-endian). Hello: `OXSPLIT1`, the key (`u32` length, bytes), six
//! `u64`: hidden size, vocabulary, layers, checkpoint bytes, session id (0: none), the session
//! to copy (0: a new one). Answered with `u32` status, `u32` message length, the message, and
//! `u64` `k`. Then requests: `u32` op, `u32` flags, `u64` position, `u32` tokens, `u32` spans,
//! `u64` aux, the vision spans (three `u64` each), the rows (`tokens x hidden` `f32`); each
//! answered with `u32` status, `u32` message length, `u64` position, `u32` rows, `u32` logits,
//! the message, the rows and the logits.

use crate::qwen::{QwenModel, QwenSession, VisionSpan};
use anyhow::{Context, Result, anyhow, bail};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

const MAGIC: &[u8; 8] = b"OXSPLIT1";
const OP_RUN: u32 = 1;
const OP_LOGITS: u32 = 2;
const OP_CHECKPOINT: u32 = 3;
/// The longest a connection to the worker may take to open.
const CONNECT: Duration = Duration::from_secs(5);
/// The longest key and error message read.
const MAX_TEXT: usize = 4096;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// The longest the worker may take to answer a pass (`OXINFER_SPLIT_TIMEOUT`, seconds; default
/// 1800: a first pass pages its half of the weights in from disk).
fn answer_timeout() -> Duration {
    let secs = std::env::var("OXINFER_SPLIT_TIMEOUT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1800);
    Duration::from_secs(secs)
}

/// Which of the model's entry points a pass came through: the worker takes the same one, so
/// that both halves choose their kernels as one machine would.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Entry {
    /// [`QwenModel::forward`].
    Forward = 0,
    /// [`QwenModel::forward_hidden_mm`].
    Hidden = 1,
    /// [`QwenModel::prefill_hidden`].
    Prefill = 2,
}

/// The final hidden states a pass's caller wants back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Rows {
    None = 0,
    Last = 1,
    All = 2,
}

/// What tells two copies of a model apart, as far as a file's size and the configuration go.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Meta {
    pub hidden: u64,
    pub vocab: u64,
    pub layers: u64,
    pub bytes: u64,
}

fn f32_bytes(v: &[f32]) -> std::borrow::Cow<'_, [u8]> {
    #[cfg(target_endian = "little")]
    {
        // SAFETY: f32 has no padding and any alignment suits u8; the length is in bytes.
        std::borrow::Cow::Borrowed(unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) })
    }
    #[cfg(not(target_endian = "little"))]
    {
        std::borrow::Cow::Owned(v.iter().flat_map(|f| f.to_le_bytes()).collect())
    }
}

fn read_f32s(r: &mut impl Read, n: usize) -> std::io::Result<Vec<f32>> {
    let mut v = vec![0f32; n];
    #[cfg(target_endian = "little")]
    {
        // SAFETY: as in `f32_bytes`; every bit pattern is a valid f32.
        let b = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), n * 4) };
        r.read_exact(b)?;
    }
    #[cfg(not(target_endian = "little"))]
    {
        let mut b = vec![0u8; n * 4];
        r.read_exact(&mut b)?;
        for (f, c) in v.iter_mut().zip(b.chunks_exact(4)) {
            *f = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        }
    }
    Ok(v)
}

fn read_u32(r: &mut impl Read) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(r: &mut impl Read) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_text(r: &mut impl Read, n: usize) -> std::io::Result<String> {
    if n > MAX_TEXT {
        return Err(std::io::Error::other("text too long"));
    }
    let mut b = vec![0u8; n];
    r.read_exact(&mut b)?;
    Ok(String::from_utf8_lossy(&b).into_owned())
}

/// A header and its payload: one write when small (a decode step's row), so that the pass
/// leaves in one segment train.
fn send(s: &mut TcpStream, mut head: Vec<u8>, payload: &[&[u8]]) -> std::io::Result<()> {
    let n: usize = payload.iter().map(|p| p.len()).sum();
    if n <= 256 << 10 {
        for p in payload {
            head.extend_from_slice(p);
        }
        return s.write_all(&head);
    }
    s.write_all(&head)?;
    for p in payload {
        s.write_all(p)?;
    }
    Ok(())
}

// ---- the front ---------------------------------------------------------------------------------

/// What a pass was answered with.
pub(crate) struct Reply {
    pub rows: Vec<f32>,
    pub logits: Vec<f32>,
}

/// The worker as the front's model knows it.
pub struct Client {
    addr: String,
    key: String,
    meta: Meta,
    /// The first layer the worker runs: this side loads the layers before it.
    k: usize,
    next: AtomicU64,
    /// The connection the head-only requests take ([`Client::logits`]).
    control: Mutex<Option<Conn>>,
}

struct Conn {
    s: TcpStream,
    /// Answers not read yet (passes nothing waited for).
    pending: usize,
}

impl Conn {
    fn reply(&mut self) -> Result<Reply> {
        let s = &mut self.s;
        let status = read_u32(s)?;
        let msg_len = read_u32(s)? as usize;
        let _pos = read_u64(s)?;
        let n_rows = read_u32(s)? as usize;
        let n_logits = read_u32(s)? as usize;
        let msg = read_text(s, msg_len)?;
        if status != 0 {
            bail!("{msg}");
        }
        Ok(Reply {
            rows: read_f32s(s, n_rows)?,
            logits: read_f32s(s, n_logits)?,
        })
    }

    /// Reads the answers of the passes not waited for.
    fn drain(&mut self) -> Result<()> {
        while self.pending > 0 {
            self.pending -= 1;
            self.reply()?;
        }
        Ok(())
    }
}

impl Client {
    /// Connects to the worker at `addr` and checks that it holds the same model; the layers it
    /// runs are from [`Client::k`] on.
    pub(crate) fn connect(addr: &str, key: &str, meta: Meta) -> Result<Arc<Client>> {
        if key.is_empty() {
            bail!("split: the worker's key is needed (split_key=KEY, or OXINFER_SPLIT_KEY)");
        }
        // Session ids of this front: unlikely to meet another front's on the same worker.
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(1, |d| d.as_nanos() as u64)
            ^ ((std::process::id() as u64) << 32);
        let mut c = Client {
            addr: addr.to_string(),
            key: key.to_string(),
            meta,
            k: 0,
            next: AtomicU64::new(seed | 1),
            control: Mutex::new(None),
        };
        let (conn, k) = c.open(0, 0)?;
        if k == 0 || k as u64 >= meta.layers {
            bail!(
                "split: the worker at {addr} runs the layers from {k} on, of {}: nothing is left for one side",
                meta.layers
            );
        }
        c.k = k;
        c.control = Mutex::new(Some(conn));
        Ok(Arc::new(c))
    }

    /// The first layer the worker runs.
    pub fn k(&self) -> usize {
        self.k
    }

    pub fn addr(&self) -> &str {
        &self.addr
    }

    fn open(&self, session: u64, fork_of: u64) -> Result<(Conn, usize)> {
        let addr = &self.addr;
        let sa = addr
            .to_socket_addrs()
            .with_context(|| format!("split: worker address {addr:?}"))?
            .next()
            .with_context(|| format!("split: worker address {addr:?} resolves to nothing"))?;
        let mut s = TcpStream::connect_timeout(&sa, CONNECT).with_context(|| format!("split: cannot reach the worker at {addr}"))?;
        s.set_nodelay(true)?;
        s.set_read_timeout(Some(answer_timeout()))?;
        s.set_write_timeout(Some(answer_timeout()))?;
        let mut hello = MAGIC.to_vec();
        hello.extend_from_slice(&(self.key.len() as u32).to_le_bytes());
        hello.extend_from_slice(self.key.as_bytes());
        let m = self.meta;
        for v in [m.hidden, m.vocab, m.layers, m.bytes, session, fork_of] {
            hello.extend_from_slice(&v.to_le_bytes());
        }
        s.write_all(&hello)?;
        let io = |e: std::io::Error| anyhow!("split: the worker at {addr} did not answer the hello ({e}): is it an oxinfer split-worker?");
        let status = read_u32(&mut s).map_err(io)?;
        let n = read_u32(&mut s).map_err(io)? as usize;
        let msg = read_text(&mut s, n).map_err(io)?;
        let k = read_u64(&mut s).map_err(io)? as usize;
        if status != 0 {
            bail!("split: the worker at {addr} refused: {msg}");
        }
        Ok((Conn { s, pending: 0 }, k))
    }

    /// The LM head on the worker: `t` rows of final hidden states to `t` rows of logits.
    pub(crate) fn logits(&self, hidden: &[f32], t: usize) -> Result<Vec<f32>> {
        let mut g = lock(&self.control);
        if g.is_none() {
            *g = Some(self.open(0, 0)?.0);
        }
        let conn = g.as_mut().expect("opened above");
        let r = request(conn, OP_LOGITS, 0, 0, t, &[], 0, hidden).and_then(|()| conn.reply());
        if r.is_err() {
            *g = None;
        }
        Ok(r?.logits)
    }
}

fn request(conn: &mut Conn, op: u32, flags: u32, pos: usize, t: usize, spans: &[VisionSpan], aux: u64, rows: &[f32]) -> Result<()> {
    let mut head = Vec::with_capacity(32 + spans.len() * 24);
    head.extend_from_slice(&op.to_le_bytes());
    head.extend_from_slice(&flags.to_le_bytes());
    head.extend_from_slice(&(pos as u64).to_le_bytes());
    head.extend_from_slice(&(t as u32).to_le_bytes());
    head.extend_from_slice(&(spans.len() as u32).to_le_bytes());
    head.extend_from_slice(&aux.to_le_bytes());
    for sp in spans {
        for v in [sp.start, sp.gh, sp.gw] {
            head.extend_from_slice(&(v as u64).to_le_bytes());
        }
    }
    send(&mut conn.s, head, &[&f32_bytes(rows)])?;
    Ok(())
}

/// A session's other half: the worker's session, for as long as this lives. A copy
/// ([`Clone`]) is a copy of the worker's session as well.
pub(crate) struct Sess {
    client: Arc<Client>,
    id: u64,
    conn: Mutex<Option<Conn>>,
    /// Some layer of the model, on either side, keeps a recurrent state (so going back needs a
    /// checkpoint: [`QwenSession::cut_to`]).
    pub recurrent: bool,
}

impl Sess {
    pub fn new(client: &Arc<Client>, recurrent: bool) -> Self {
        Self {
            client: client.clone(),
            id: client.next.fetch_add(1, Ordering::Relaxed),
            conn: Mutex::new(None),
            recurrent,
        }
    }

    fn with<R>(&self, f: impl FnOnce(&mut Conn) -> Result<R>) -> Result<R> {
        let mut g = lock(&self.conn);
        if g.is_none() {
            *g = Some(self.client.open(self.id, 0)?.0);
        }
        f(g.as_mut().expect("opened above"))
    }

    /// Sends a pass of `t` rows starting at `pos` without waiting for its answer.
    #[allow(clippy::too_many_arguments)]
    pub fn send_run(&self, pos: usize, t: usize, rows: &[f32], spans: &[VisionSpan], entry: Entry, logits: bool, want: Rows) -> Result<()> {
        let flags = entry as u32 | (logits as u32) << 2 | (want as u32) << 3;
        self.with(|c| {
            request(c, OP_RUN, flags, pos, t, spans, 0, rows)?;
            c.pending += 1;
            Ok(())
        })
    }

    /// The answer of the last pass sent (after those not waited for, which are checked).
    pub fn recv(&self) -> Result<Reply> {
        self.with(|c| {
            if c.pending == 0 {
                bail!("no pass to wait for");
            }
            c.pending -= 1;
            while c.pending > 0 {
                c.pending -= 1;
                c.reply()?;
            }
            c.reply()
        })
    }

    /// Takes a checkpoint of position `pos` on the worker, as [`QwenSession::checkpoint`] does
    /// here (`max`: the most kept).
    pub fn checkpoint(&self, pos: usize, max: usize) -> Result<()> {
        self.with(|c| {
            c.drain()?;
            request(c, OP_CHECKPOINT, 0, pos, 0, &[], max as u64, &[])?;
            c.reply().map(|_| ())
        })
    }
}

impl Clone for Sess {
    /// A copy of the worker's session too, as it is once the passes sent so far are done. (A
    /// failure shows at the copy's first pass, whose position the worker then does not have.)
    fn clone(&self) -> Self {
        let id = self.client.next.fetch_add(1, Ordering::Relaxed);
        let mut g = lock(&self.conn);
        let conn = match g.as_mut() {
            // Nothing ran yet: a fresh session.
            None => None,
            Some(c) => c.drain().and_then(|()| self.client.open(id, self.id)).map(|(c, _)| c).ok(),
        };
        Self {
            client: self.client.clone(),
            id,
            conn: Mutex::new(conn),
            recurrent: self.recurrent,
        }
    }
}

/// Loads the front of the model at `dir` (a path the engine resolved; `files` its weights,
/// mapped): the layers before those the worker at `addr` runs, which must hold the same model.
/// What the engine option `split` does.
pub fn load_front<'a>(
    dir: &std::path::Path,
    files: &'a crate::safetensors::ModelFiles,
    quant: crate::model::Quant,
    addr: &str,
    key: &str,
) -> Result<QwenModel<'a>> {
    let cfg = crate::config::QwenConfig::load(dir)?;
    let meta = Meta {
        hidden: cfg.hidden_size as u64,
        vocab: cfg.vocab_size as u64,
        layers: cfg.num_hidden_layers as u64,
        bytes: crate::safetensors::checkpoint_bytes(dir),
    };
    let client = Client::connect(addr, key, meta)?;
    let k = client.k();
    QwenModel::load_part(cfg, files, quant, 0..k, Some(client))
}

// ---- the worker --------------------------------------------------------------------------------

/// Settings of a worker ([`Worker::start`]).
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct WorkerOptions {
    /// The address to listen on (`HOST:PORT`; the direct link's address, see docs/cluster.md).
    pub listen: String,
    /// The key fronts must give (not empty).
    pub key: String,
    /// The first layer this side runs (default: half the layers).
    pub from: Option<usize>,
    /// How the model is loaded: `quant`, `device`, `cache_dir` are read.
    pub engine: crate::engine::EngineOptions,
    /// One line per connection on stderr.
    pub log: bool,
}

impl Default for WorkerOptions {
    fn default() -> Self {
        Self {
            listen: "127.0.0.1:8090".into(),
            key: String::new(),
            from: None,
            engine: Default::default(),
            log: true,
        }
    }
}

struct Shared {
    model: &'static QwenModel<'static>,
    sessions: Mutex<HashMap<u64, Arc<Mutex<QwenSession>>>>,
    key: String,
    meta: Meta,
    k: usize,
    log: bool,
}

/// A running worker: the last layers of a model, served to a front.
pub struct Worker {
    addr: SocketAddr,
    k: usize,
    layers: usize,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Worker {
    /// Loads layers `from..` of `model` (a path or a repo id, as the engine takes them; a
    /// Qwen-path model of dense layers) and serves them on [`WorkerOptions::listen`]. The model
    /// stays loaded for the life of the process.
    pub fn start(model: &str, opts: &WorkerOptions) -> Result<Worker> {
        if opts.key.is_empty() {
            bail!("split-worker: a key is needed (--key KEY, or OXINFER_SPLIT_KEY): the front gives the same one");
        }
        let eo = &opts.engine;
        let dir = crate::engine::resolve_model(model, eo.cache_dir.as_deref())?;
        crate::gpu::ensure(eo.device)?;
        let cfg = crate::config::QwenConfig::load(&dir)?;
        let layers = cfg.num_hidden_layers;
        let k = opts.from.unwrap_or(layers / 2);
        if k == 0 || k >= layers {
            bail!("split-worker: --from {k} leaves nothing for one side (the model has {layers} layers)");
        }
        let meta = Meta {
            hidden: cfg.hidden_size as u64,
            vocab: cfg.vocab_size as u64,
            layers: layers as u64,
            bytes: crate::safetensors::checkpoint_bytes(&dir),
        };
        // Mapped: the first layers' weights are never read here.
        let files: &'static crate::safetensors::ModelFiles = Box::leak(Box::new(crate::safetensors::ModelFiles::open(&dir, false)?));
        let model: &'static QwenModel<'static> = Box::leak(Box::new(QwenModel::load_part(cfg, files, eo.quant, k..layers, None)?));
        let listener = TcpListener::bind(&opts.listen).with_context(|| format!("split-worker: bind {}", opts.listen))?;
        listener.set_nonblocking(true)?;
        let addr = listener.local_addr()?;
        let shared = Arc::new(Shared {
            model,
            sessions: Mutex::new(HashMap::new()),
            key: opts.key.clone(),
            meta,
            k,
            log: opts.log,
        });
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = stop.clone();
        let thread = std::thread::Builder::new().name("oxinfer-split".into()).spawn(move || {
            while !stopped.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((s, peer)) => {
                        let shared = shared.clone();
                        let _ = std::thread::Builder::new().name("oxinfer-split-conn".into()).spawn(move || {
                            if let Err(e) = serve(&shared, s)
                                && shared.log
                            {
                                eprintln!("[split-worker: {peer}: {e:#}]");
                            }
                        });
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => std::thread::sleep(Duration::from_millis(20)),
                    Err(_) => std::thread::sleep(Duration::from_millis(100)),
                }
            }
        })?;
        Ok(Worker {
            addr,
            k,
            layers,
            stop,
            thread: Some(thread),
        })
    }

    /// The address it listens on (with the port chosen when the configured one was 0).
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// The layers it runs: `from..total`.
    pub fn layers(&self) -> (usize, usize) {
        (self.k, self.layers)
    }

    /// Blocks until the worker stops.
    pub fn wait(mut self) {
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }

    /// Stops accepting connections (those open end with their fronts).
    pub fn stop(self) {
        self.stop.store(true, Ordering::Relaxed);
        self.wait();
    }
}

fn answer(s: &mut TcpStream, r: Result<(usize, Vec<f32>, Vec<f32>)>) -> std::io::Result<()> {
    let (status, msg, pos, rows, logits) = match r {
        Ok((pos, rows, logits)) => (0u32, String::new(), pos, rows, logits),
        Err(e) => {
            let mut m = format!("{e:#}");
            m.truncate(MAX_TEXT);
            (1, m, 0, Vec::new(), Vec::new())
        }
    };
    let mut head = Vec::with_capacity(24 + msg.len());
    head.extend_from_slice(&status.to_le_bytes());
    head.extend_from_slice(&(msg.len() as u32).to_le_bytes());
    head.extend_from_slice(&(pos as u64).to_le_bytes());
    head.extend_from_slice(&(rows.len() as u32).to_le_bytes());
    head.extend_from_slice(&(logits.len() as u32).to_le_bytes());
    head.extend_from_slice(msg.as_bytes());
    send(s, head, &[&f32_bytes(&rows), &f32_bytes(&logits)])
}

fn serve(sh: &Arc<Shared>, mut s: TcpStream) -> Result<()> {
    s.set_nonblocking(false)?;
    s.set_nodelay(true)?;
    // The hello within a few seconds; a front may then stay silent for as long as it lives.
    s.set_read_timeout(Some(Duration::from_secs(10)))?;
    let mut magic = [0u8; 8];
    s.read_exact(&mut magic)?;
    if &magic != MAGIC {
        bail!("not a split front (bad magic)");
    }
    let n = read_u32(&mut s)? as usize;
    let key = read_text(&mut s, n)?;
    let mut v = [0u64; 6];
    for x in &mut v {
        *x = read_u64(&mut s)?;
    }
    let [hidden, vocab, layers, bytes, session, fork_of] = v;
    let hello = |s: &mut TcpStream, status: u32, msg: &str| -> std::io::Result<()> {
        let mut out = status.to_le_bytes().to_vec();
        out.extend_from_slice(&(msg.len() as u32).to_le_bytes());
        out.extend_from_slice(msg.as_bytes());
        out.extend_from_slice(&(sh.k as u64).to_le_bytes());
        s.write_all(&out)
    };
    // Compared in constant time.
    let same = key.len() == sh.key.len() && key.bytes().zip(sh.key.bytes()).fold(0u8, |a, (x, y)| a | (x ^ y)) == 0;
    if !same {
        hello(&mut s, 1, "wrong key")?;
        bail!("wrong key");
    }
    let theirs = Meta {
        hidden,
        vocab,
        layers,
        bytes,
    };
    if theirs != sh.meta {
        let msg = format!(
            "not the same model: the front has hidden {hidden}, vocabulary {vocab}, {layers} layers, {bytes} bytes; the worker hidden {}, vocabulary {}, {} layers, {} bytes",
            sh.meta.hidden, sh.meta.vocab, sh.meta.layers, sh.meta.bytes
        );
        hello(&mut s, 1, &msg)?;
        bail!("{msg}");
    }
    let sess = if session == 0 {
        None
    } else {
        let copy = match fork_of {
            0 => None,
            f => match lock(&sh.sessions).get(&f).cloned() {
                Some(src) => Some(lock(&src).clone()),
                None => {
                    hello(&mut s, 1, "the session to copy is gone")?;
                    bail!("session {f:x} to copy is gone");
                }
            },
        };
        let sess = Arc::new(Mutex::new(copy.unwrap_or_else(|| sh.model.new_session())));
        lock(&sh.sessions).insert(session, sess.clone());
        Some(sess)
    };
    hello(&mut s, 0, "")?;
    s.set_read_timeout(None)?;
    let h = sh.meta.hidden as usize;
    let r = (|| -> Result<()> {
        loop {
            let op = match read_u32(&mut s) {
                Ok(op) => op,
                // The front closed the connection: its session ends.
                Err(_) => return Ok(()),
            };
            let flags = read_u32(&mut s)?;
            let pos = read_u64(&mut s)? as usize;
            let t = read_u32(&mut s)? as usize;
            let n_spans = read_u32(&mut s)? as usize;
            let aux = read_u64(&mut s)?;
            if t > 1 << 20 || n_spans > 1 << 16 {
                bail!("bad request ({t} tokens, {n_spans} spans)");
            }
            let mut spans = Vec::with_capacity(n_spans);
            for _ in 0..n_spans {
                let (start, gh, gw) = (read_u64(&mut s)? as usize, read_u64(&mut s)? as usize, read_u64(&mut s)? as usize);
                spans.push(VisionSpan { start, gh, gw });
            }
            let rows = read_f32s(&mut s, t * h)?;
            let out = match op {
                OP_LOGITS => Ok((0, Vec::new(), sh.model.logits(&rows, t))),
                OP_RUN | OP_CHECKPOINT => match &sess {
                    None => Err(anyhow!("no session on this connection")),
                    Some(sess) => {
                        let mut g = lock(sess);
                        if op == OP_CHECKPOINT {
                            if g.pos == pos {
                                g.checkpoint(aux as usize);
                                Ok((pos, Vec::new(), Vec::new()))
                            } else {
                                Err(anyhow!("checkpoint of position {pos}: the worker's session is at {}", g.pos))
                            }
                        } else {
                            let entry = match flags & 3 {
                                0 => Entry::Forward,
                                1 => Entry::Hidden,
                                _ => Entry::Prefill,
                            };
                            let want = match (flags >> 3) & 3 {
                                0 => Rows::None,
                                1 => Rows::Last,
                                _ => Rows::All,
                            };
                            sh.model
                                .tail_run(&mut g, pos, &rows, t, &spans, entry, flags & 4 != 0, want)
                                .map(|(rows, logits)| (g.pos, rows, logits))
                        }
                    }
                },
                other => Err(anyhow!("unknown op {other}")),
            };
            answer(&mut s, out)?;
        }
    })();
    if session != 0 {
        lock(&sh.sessions).remove(&session);
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::QwenConfig;
    use crate::engine::{Engine, EngineOptions, Flow, Message};
    use crate::model::Quant;
    use crate::safetensors::ModelFiles;
    use crate::sampler::argmax;

    const KEY: &str = "link-key";

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    fn toks(n: usize, seed: u32) -> Vec<u32> {
        (0..n as u32)
            .map(|i| (i * 37 + seed * 11 + 5) % crate::qwen::tiny::SPECIAL_BASE)
            .collect()
    }

    fn worker(dir: &std::path::Path, from: usize) -> Worker {
        let mut o = WorkerOptions {
            listen: "127.0.0.1:0".into(),
            key: KEY.into(),
            from: Some(from),
            log: false,
            ..Default::default()
        };
        // (Lib tests share one process-wide backend: the CPU.)
        o.engine.set("device", "cpu").unwrap();
        Worker::start(dir.to_str().unwrap(), &o).unwrap()
    }

    fn meta(dir: &std::path::Path, cfg: &QwenConfig) -> Meta {
        Meta {
            hidden: cfg.hidden_size as u64,
            vocab: cfg.vocab_size as u64,
            layers: cfg.num_hidden_layers as u64,
            bytes: crate::safetensors::checkpoint_bytes(dir),
        }
    }

    /// The whole model, and its front with a worker running the layers from `from`.
    fn pair(dir: &std::path::Path, from: usize) -> (&'static QwenModel<'static>, &'static QwenModel<'static>, Worker) {
        let files: &'static ModelFiles = Box::leak(Box::new(ModelFiles::open(dir, true).unwrap()));
        let cfg = QwenConfig::load(dir).unwrap();
        let whole: &'static QwenModel = Box::leak(Box::new(QwenModel::load(cfg.clone(), files, Quant::Bf16).unwrap()));
        let w = worker(dir, from);
        let client = Client::connect(&w.addr().to_string(), KEY, meta(dir, &cfg)).unwrap();
        assert_eq!(client.k(), from);
        let front: &'static QwenModel = Box::leak(Box::new(
            QwenModel::load_part(cfg, files, Quant::Bf16, 0..from, Some(client)).unwrap(),
        ));
        assert_eq!(front.cfg.num_hidden_layers, from);
        (whole, front, w)
    }

    /// A prompt in chunks, decode steps, a checkpoint and a cut back to it, a copy of the
    /// session and a batched step: the same bits from the split model as from the whole one.
    fn same_as_whole(dir: &std::path::Path, from: usize) {
        let (whole, front, w) = pair(dir, from);
        let run = |m: &'static QwenModel<'static>| -> Vec<Vec<u32>> {
            let mut out = Vec::new();
            let mut s = m.new_session();
            // A prompt of 45 tokens in chunks of 16 (the last chunk's hidden states are read).
            let hs = crate::generate::prefill(m, &mut s, &toks(45, 1), 16, &[]);
            let h = m.cfg.hidden_size;
            let l = m.logits(&hs[hs.len() - h..], 1);
            out.push(bits(&l));
            let mut t = argmax(&l);
            for _ in 0..6 {
                let l = m.forward(&mut s, &[t]);
                t = argmax(&l);
                out.push(bits(&l));
            }
            // A checkpoint, more tokens, back to it, other tokens.
            s.checkpoint(4);
            let at = s.pos;
            for &tk in &toks(5, 2) {
                out.push(bits(&m.forward(&mut s, &[tk])));
            }
            s.cut_to(at).unwrap();
            assert_eq!(s.pos, at);
            // Final hidden states of a few tokens at once.
            out.push(bits(&m.forward_hidden_mm(&mut s, &toks(3, 3), &[])));
            // A copy of the session, each going its own way.
            let mut s2 = s.clone();
            out.push(bits(&m.forward(&mut s2, &[7])));
            out.push(bits(&m.forward(&mut s, &[9])));
            out.push(bits(&m.forward(&mut s2, &[11])));
            // One step of both at once.
            for l in m.forward_batch(&mut [&mut s, &mut s2], &[21, 22]) {
                out.push(bits(&l));
            }
            out.push(bits(&m.forward(&mut s, &[1])));
            out.push(bits(&m.forward(&mut s2, &[2])));
            out
        };
        let (a, b) = (run(whole), run(front));
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            assert!(x == y, "output {i} of the split model differs from the whole model's");
        }
        w.stop();
    }

    #[test]
    fn split_matches_whole_qwen35() {
        let dir = std::env::temp_dir().join(format!("oxinfer-split-q35-{}", std::process::id()));
        crate::qwen::tiny::write(&dir, false, false).unwrap();
        // Recurrent layers on both sides (three Gated DeltaNet layers, then attention).
        same_as_whole(&dir, 2);
        // The last layer alone on the worker: attention there, the recurrent ones here.
        same_as_whole(&dir, 3);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn split_matches_whole_dense() {
        for llama in [false, true] {
            let dir = std::env::temp_dir().join(format!("oxinfer-split-dense-{llama}-{}", std::process::id()));
            crate::qwen::tiny::write_dense(&dir, llama).unwrap();
            same_as_whole(&dir, 1);
            std::fs::remove_dir_all(&dir).ok();
        }
    }

    /// The worker refuses another key and another model; a front says what it cannot do.
    #[test]
    fn refusals() {
        let dir = std::env::temp_dir().join(format!("oxinfer-split-refuse-{}", std::process::id()));
        crate::qwen::tiny::write(&dir, false, false).unwrap();
        let cfg = QwenConfig::load(&dir).unwrap();
        let w = worker(&dir, 2);
        let addr = w.addr().to_string();
        let m = meta(&dir, &cfg);
        let e = Client::connect(&addr, "other", m).err().unwrap().to_string();
        assert!(e.contains("wrong key"), "{e}");
        let e = Client::connect(&addr, "", m).err().unwrap().to_string();
        assert!(e.contains("key is needed"), "{e}");
        let e = Client::connect(&addr, KEY, Meta { bytes: m.bytes + 1, ..m })
            .err()
            .unwrap()
            .to_string();
        assert!(e.contains("not the same model"), "{e}");
        // No worker there: an error within the connect timeout, not a wait.
        let free = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap();
        let t0 = std::time::Instant::now();
        assert!(Client::connect(&free.to_string(), KEY, m).is_err());
        assert!(t0.elapsed() < CONNECT + Duration::from_secs(2));
        // A worker needs a key and a layer that leaves something on both sides.
        let mut o = WorkerOptions::default();
        o.engine.set("device", "cpu").unwrap();
        o.listen = "127.0.0.1:0".into();
        o.log = false;
        let err = |o: &WorkerOptions, dir: &std::path::Path| Worker::start(dir.to_str().unwrap(), o).err().unwrap().to_string();
        assert!(err(&o, &dir).contains("key is needed"));
        o.key = KEY.into();
        o.from = Some(4);
        assert!(err(&o, &dir).contains("leaves nothing"));
        // Options a split model does not take.
        let mut eo = EngineOptions::default();
        eo.set("device", "cpu").unwrap();
        eo.set("split", &addr).unwrap();
        eo.set("split_key", KEY).unwrap();
        eo.set("mtp", "1").unwrap();
        let e = Engine::load(dir.to_str().unwrap(), &eo).err().unwrap().to_string();
        assert!(e.contains("MTP") && e.contains("not supported"), "{e}");
        // Mixture-of-experts layers are not split.
        let moe = std::env::temp_dir().join(format!("oxinfer-split-refuse-moe-{}", std::process::id()));
        crate::qwen::tiny::write(&moe, true, false).unwrap();
        o.from = Some(2);
        let e = err(&o, &moe);
        assert!(e.contains("mixture-of-experts"), "{e}");
        w.stop();
        std::fs::remove_dir_all(&dir).ok();
        std::fs::remove_dir_all(&moe).ok();
    }

    /// Through the engine: a conversation of two turns gives the tokens of the whole model, the
    /// second turn reusing the first one's cache on both sides.
    #[test]
    fn engine_chat_matches_whole() {
        let dir = std::env::temp_dir().join(format!("oxinfer-split-engine-{}", std::process::id()));
        crate::qwen::tiny::write(&dir, false, false).unwrap();
        let w = worker(&dir, 2);
        let mut quiet = |_: &crate::engine::Delta| Flow::Continue;
        let mut run = |split: bool| {
            let mut o = EngineOptions::default();
            o.set("device", "cpu").unwrap();
            // (The whole model decodes plainly too: the comparison is of the same decoding.)
            o.set("draft", "none").unwrap();
            o.set("lookup", "0").unwrap();
            if split {
                o.set("split", &w.addr().to_string()).unwrap();
                o.set("split_key", KEY).unwrap();
            }
            let mut e = Engine::load(dir.to_str().unwrap(), &o).unwrap();
            e.config_mut().max_tokens = 24;
            let mut convo = vec![
                Message::system("You answer briefly. ".repeat(4)),
                Message::user("Hello there, what is the weather like today?"),
            ];
            let r1 = e.chat(&convo, &mut quiet).unwrap();
            convo.push(Message::assistant(r1.text.clone()));
            convo.push(Message::user("And tomorrow?"));
            let r2 = e.chat(&convo, &mut quiet).unwrap();
            // The same conversation again: back to the checkpoint before its last prompt token.
            let r3 = e.chat(&convo, &mut quiet).unwrap();
            (
                r1.tokens.clone(),
                r2.tokens.clone(),
                r3.tokens.clone(),
                [r1.stats.prompt_tokens, r2.stats.cached_tokens, r3.stats.cached_tokens],
            )
        };
        let (whole, split) = (run(false), run(true));
        assert_eq!(whole, split);
        assert!(whole.3[1] > 0 && whole.3[2] > 0, "{:?}", whole.3);
        w.stop();
        std::fs::remove_dir_all(&dir).ok();
    }
}
