//! The hub client's network layer over rsurl: where the hubs are, how long each step may take,
//! and which address of a host is dialled first.
//!
//! Nothing here waits without a bound. A name lookup, each connection attempt and each silence
//! on an open connection (the TLS handshake, the response headers, a body that stops) have their
//! own timeout, and the small requests (a repository listing, a configuration file) also have one
//! deadline for the whole exchange ([`send`]). The addresses of a host are raced as Happy
//! Eyeballs does ([`Eyeballs`]), so that an address that does not answer (an IPv6 route that
//! went away) costs the stagger between two attempts, not a connect timeout.

use anyhow::{Result, anyhow};
use rsurl::Request;
use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::{Arc, Mutex, OnceLock, mpsc};
use std::time::{Duration, Instant};

/// huggingface.co, unless `HF_ENDPOINT` names another server with its API.
const HF_ENDPOINT: &str = "https://huggingface.co";

/// Where the hubs are and how long the client waits for them.
#[derive(Clone)]
pub(super) struct Cfg {
    /// The HuggingFace endpoint, without a trailing slash.
    pub hf: String,
    /// The mirror's listing method.
    pub mirror_api: String,
    /// The hub for this configuration; None: [`super::hub`].
    pub hub: Option<super::Hub>,
    /// No request at all: the cache serves or the load fails.
    pub offline: bool,
    /// The whole listing of a repository whose cache can serve the load: past it the cache does.
    pub check: Duration,
    /// The whole of a small request that the load cannot do without.
    pub meta: Duration,
    /// One connection attempt.
    pub connect: Duration,
    /// The longest silence on an open connection.
    pub stall: Duration,
    /// Retries of a download that makes no progress, and the first wait between two.
    pub retries: u32,
    pub backoff: Duration,
    /// How long a hub that did not answer is not asked again when the cache can serve.
    pub down_for: Duration,
    pub resolver: Arc<Eyeballs>,
}

/// A number of seconds from the environment (`2`, `0.5`); None when unset or not a positive
/// number.
fn env_secs(name: &str) -> Option<Duration> {
    parse_secs(&std::env::var(name).ok()?)
}

fn parse_secs(v: &str) -> Option<Duration> {
    let s: f64 = v.trim().parse().ok()?;
    (s.is_finite() && s > 0.0).then(|| Duration::from_secs_f64(s.min(86_400.0)))
}

/// Whether an environment value switches something on, as huggingface_hub reads its own
/// (`1`, `true`, `yes`, `on`, in any case).
pub(super) fn truthy(v: &str) -> bool {
    matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on")
}

impl Cfg {
    fn from_env() -> Cfg {
        let connect = Duration::from_secs(5);
        Cfg {
            hf: std::env::var("HF_ENDPOINT")
                .ok()
                .map(|e| e.trim().trim_end_matches('/').to_string())
                .filter(|e| !e.is_empty())
                .unwrap_or_else(|| HF_ENDPOINT.to_string()),
            mirror_api: super::MIRROR_API.to_string(),
            hub: None,
            offline: false,
            check: env_secs("HF_HUB_ETAG_TIMEOUT").unwrap_or(Duration::from_secs(5)),
            meta: Duration::from_secs(30),
            connect,
            stall: env_secs("HF_HUB_DOWNLOAD_TIMEOUT").unwrap_or(Duration::from_secs(20)),
            retries: 3,
            backoff: Duration::from_millis(500),
            down_for: Duration::from_secs(60),
            resolver: Arc::new(Eyeballs::system(Duration::from_secs(5), connect, Duration::from_millis(250))),
        }
    }

    /// Whether `url` is on the HuggingFace endpoint (the only place the token goes).
    pub fn is_hf(&self, url: &str) -> bool {
        url.strip_prefix(self.hf.as_str()).is_some_and(|rest| rest.starts_with('/'))
    }
}

#[cfg(test)]
thread_local! {
    /// The configuration of this test's thread ([`with_cfg`]).
    static TEST_CFG: std::cell::RefCell<Option<Cfg>> = const { std::cell::RefCell::new(None) };
}

/// Runs `f` with `cfg` as this thread's configuration (tests: local servers, short timeouts).
#[cfg(test)]
pub(super) fn with_cfg<T>(cfg: Cfg, f: impl FnOnce() -> T) -> T {
    let prev = TEST_CFG.with(|c| c.replace(Some(cfg)));
    let r = f();
    TEST_CFG.with(|c| *c.borrow_mut() = prev);
    r
}

/// The configuration in use: the environment's (read once), with the offline switch as it is now.
pub(super) fn cfg() -> Cfg {
    #[cfg(test)]
    if let Some(c) = TEST_CFG.with(|c| c.borrow().clone()) {
        return c;
    }
    static BASE: OnceLock<Cfg> = OnceLock::new();
    let mut c = BASE.get_or_init(Cfg::from_env).clone();
    c.offline = super::offline();
    c
}

/// Marks an error as the hub not answering (as opposed to the hub answering "no"): the cause the
/// cache may stand in for.
#[derive(Debug)]
pub(super) struct Unreachable(pub String);

impl std::fmt::Display for Unreachable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Unreachable {}

pub(super) fn unreachable(msg: impl Into<String>) -> anyhow::Error {
    anyhow::Error::new(Unreachable(msg.into()))
}

pub(super) fn is_unreachable(e: &anyhow::Error) -> bool {
    e.downcast_ref::<Unreachable>().is_some()
}

fn secs(d: Duration) -> String {
    let s = d.as_secs_f64();
    if s < 1.0 {
        format!("{:.0} ms", s * 1e3)
    } else if (s - s.round()).abs() < 0.05 {
        format!("{s:.0} s")
    } else {
        format!("{s:.1} s")
    }
}

/// A GET request with the client's timeouts and address choice; `auth` adds the HuggingFace token
/// (never sent to other hosts).
pub(super) fn request(cfg: &Cfg, url: &str, auth: bool) -> Result<Request> {
    let mut r = Request::get(url)
        .map_err(|e| anyhow!("{url}: {e}"))?
        .follow_redirects(true)
        .max_redirs(10)
        .connect_timeout(cfg.connect)
        .read_timeout(Some(cfg.stall))
        .resolver(cfg.resolver.clone())
        .header("User-Agent", "oxinfer/0.1")
        // The resumable downloader stores raw bytes; refuse compressed transfer encodings.
        .header("Accept-Encoding", "identity");
    if auth && let Some(t) = super::token() {
        r = r.header("Authorization", &format!("Bearer {t}"));
    }
    Ok(r)
}

/// Sends a small request with `total` as the deadline of the whole exchange (lookup, connection,
/// TLS, headers and body): past it the connection is torn down and the error says so. Every
/// failure to get an answer is [`Unreachable`]; the status of an answer is the caller's to read.
pub(super) fn send(req: Request, total: Duration) -> Result<rsurl::Response> {
    if total.is_zero() {
        return Err(unreachable("no time left to ask"));
    }
    let cancel = rsurl::CancelToken::new();
    let req = req.cancel_token(cancel.clone());
    let (tx, rx) = mpsc::channel();
    // On its own thread: nothing in a blocking exchange can be given a deadline from inside.
    std::thread::Builder::new()
        .name("oxinfer-hub".into())
        .spawn(move || {
            let _ = tx.send(req.send());
        })
        .map_err(|e| unreachable(format!("cannot start the request: {e}")))?;
    match rx.recv_timeout(total) {
        Ok(Ok(resp)) => Ok(resp),
        Ok(Err(e)) => Err(unreachable(e.to_string())),
        Err(_) => {
            cancel.cancel();
            Err(unreachable(format!("no answer within {}", secs(total))))
        }
    }
}

type Lookup = dyn Fn(&str, u16) -> std::io::Result<Vec<SocketAddr>> + Send + Sync;
type Dial = dyn Fn(SocketAddr, Duration) -> std::io::Result<()> + Send + Sync;

/// How many addresses of a host are handed to the dialler, the one that answered first: the
/// dialler gives each the whole connect timeout, so the list is what a route lost between the
/// race and the dial can cost.
const MAX_ADDRS: usize = 3;
/// How long the address that answered stays the first choice without being raced again.
const KEEP_GOOD: Duration = Duration::from_secs(60);

/// Name resolution with a deadline and an address order found as Happy Eyeballs finds it
/// (RFC 8305): the families alternate, starting with the one the system resolver prefers, an
/// attempt starts every `stagger` (or as soon as the one before fails), and the address that
/// connects first goes first. rsurl dials the addresses it is given one after the other, each
/// with the whole connect timeout, so the race is run here on throwaway connections and its
/// winner (remembered for a minute per host) is what rsurl then dials.
pub(super) struct Eyeballs {
    dns: Duration,
    connect: Duration,
    stagger: Duration,
    lookup: Arc<Lookup>,
    dial: Arc<Dial>,
    good: Mutex<HashMap<(String, u16), (SocketAddr, Instant)>>,
}

impl std::fmt::Debug for Eyeballs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Eyeballs")
            .field("dns", &self.dns)
            .field("connect", &self.connect)
            .field("stagger", &self.stagger)
            .finish()
    }
}

impl Eyeballs {
    /// With the system resolver and real connections.
    pub fn system(dns: Duration, connect: Duration, stagger: Duration) -> Eyeballs {
        Eyeballs::new(
            dns,
            connect,
            stagger,
            Arc::new(|host, port| std::net::ToSocketAddrs::to_socket_addrs(&(host, port)).map(Iterator::collect)),
            Arc::new(|addr, timeout| std::net::TcpStream::connect_timeout(&addr, timeout).map(drop)),
        )
    }

    pub fn new(dns: Duration, connect: Duration, stagger: Duration, lookup: Arc<Lookup>, dial: Arc<Dial>) -> Eyeballs {
        Eyeballs {
            dns,
            connect,
            stagger,
            lookup,
            dial,
            good: Mutex::new(HashMap::new()),
        }
    }

    /// The addresses of `host`, within the lookup deadline (the system resolver cannot be
    /// interrupted: past the deadline its thread is left to finish on its own).
    fn lookup(&self, host: &str, port: u16) -> std::io::Result<Vec<SocketAddr>> {
        let (tx, rx) = mpsc::channel();
        let (lookup, name) = (self.lookup.clone(), host.to_string());
        std::thread::Builder::new().name("oxinfer-dns".into()).spawn(move || {
            let _ = tx.send(lookup(&name, port));
        })?;
        match rx.recv_timeout(self.dns) {
            Ok(r) => r,
            Err(_) => Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                format!("the lookup of {host} got no answer within {}", secs(self.dns)),
            )),
        }
    }

    /// The index of the address that connects first, the attempts staggered.
    fn race(&self, addrs: &[SocketAddr]) -> std::io::Result<usize> {
        let (tx, rx) = mpsc::channel::<(usize, std::io::Result<()>)>();
        let (mut started, mut pending) = (0, 0);
        let mut next_at = Instant::now();
        let mut last_err = None;
        loop {
            if started < addrs.len() && (pending == 0 || Instant::now() >= next_at) {
                let (tx, dial, addr, timeout, i) = (tx.clone(), self.dial.clone(), addrs[started], self.connect, started);
                std::thread::Builder::new().name("oxinfer-dial".into()).spawn(move || {
                    let _ = tx.send((i, dial(addr, timeout)));
                })?;
                started += 1;
                pending += 1;
                next_at = Instant::now() + self.stagger;
                continue;
            }
            if pending == 0 {
                break;
            }
            let wait = if started < addrs.len() {
                next_at.saturating_duration_since(Instant::now())
            } else {
                // Every attempt has its own timeout; the margin is for one that overruns it.
                self.connect + Duration::from_secs(1)
            };
            match rx.recv_timeout(wait) {
                Ok((i, Ok(()))) => return Ok(i),
                Ok((_, Err(e))) => {
                    pending -= 1;
                    last_err = Some(e);
                }
                Err(_) if started == addrs.len() => break,
                Err(_) => {}
            }
        }
        Err(last_err.unwrap_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                format!("none of {} addresses answered within {}", addrs.len(), secs(self.connect)),
            )
        }))
    }

    fn addresses(&self, host: &str, port: u16) -> std::io::Result<Vec<SocketAddr>> {
        let bare = host.trim_start_matches('[').trim_end_matches(']');
        if let Ok(ip) = bare.parse::<IpAddr>() {
            return Ok(vec![SocketAddr::new(ip, port)]);
        }
        let mut addrs = Vec::new();
        for a in self.lookup(bare, port)? {
            if !addrs.contains(&a) {
                addrs.push(a);
            }
        }
        if addrs.is_empty() {
            return Err(std::io::Error::new(std::io::ErrorKind::NotFound, format!("{host} has no address")));
        }
        if addrs.len() == 1 {
            return Ok(addrs);
        }
        let mut addrs = interleave(addrs);
        let key = (bare.to_ascii_lowercase(), port);
        let known = {
            let good = self.good.lock().unwrap_or_else(|e| e.into_inner());
            good.get(&key).filter(|(_, at)| at.elapsed() < KEEP_GOOD).map(|(a, _)| *a)
        };
        let first = match known.and_then(|a| addrs.iter().position(|b| *b == a)) {
            Some(i) => i,
            None => {
                let i = self.race(&addrs)?;
                self.good
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .insert(key, (addrs[i], Instant::now()));
                i
            }
        };
        // Behind it, the addresses of its family first: the route that carried it carries them.
        let winner = addrs.remove(first);
        addrs.sort_by_key(|a| a.is_ipv6() != winner.is_ipv6());
        addrs.insert(0, winner);
        addrs.truncate(MAX_ADDRS);
        Ok(addrs)
    }
}

/// The addresses with their families alternating, the first one's family first, each family in
/// the order given.
fn interleave(addrs: Vec<SocketAddr>) -> Vec<SocketAddr> {
    let Some(first) = addrs.first().map(SocketAddr::is_ipv6) else {
        return addrs;
    };
    let (a, b): (Vec<SocketAddr>, Vec<SocketAddr>) = addrs.into_iter().partition(|x| x.is_ipv6() == first);
    let (mut a, mut b) = (a.into_iter(), b.into_iter());
    let mut out = Vec::new();
    loop {
        match (a.next(), b.next()) {
            (None, None) => return out,
            (x, y) => out.extend(x.into_iter().chain(y)),
        }
    }
}

impl rsurl::net::Resolver for Eyeballs {
    fn resolve(&self, host: &str, port: u16) -> rsurl::Result<Vec<SocketAddr>> {
        self.addresses(host, port).map_err(rsurl::Error::Io)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn v(s: &str) -> SocketAddr {
        s.parse().unwrap()
    }

    #[test]
    fn families_alternate() {
        let got = interleave(vec![
            v("[2001:db8::1]:1"),
            v("[2001:db8::2]:1"),
            v("192.0.2.1:1"),
            v("[2001:db8::3]:1"),
        ]);
        assert_eq!(
            got,
            [v("[2001:db8::1]:1"), v("192.0.2.1:1"), v("[2001:db8::2]:1"), v("[2001:db8::3]:1")]
        );
        let got = interleave(vec![v("192.0.2.1:1"), v("192.0.2.2:1"), v("[2001:db8::1]:1")]);
        assert_eq!(got, [v("192.0.2.1:1"), v("[2001:db8::1]:1"), v("192.0.2.2:1")]);
        assert!(interleave(Vec::new()).is_empty());
    }

    #[test]
    fn seconds_and_switches() {
        assert_eq!(parse_secs("2"), Some(Duration::from_secs(2)));
        assert_eq!(parse_secs(" 0.5 "), Some(Duration::from_millis(500)));
        for bad in ["", "0", "-1", "soon", "nan", "inf"] {
            assert_eq!(parse_secs(bad), None, "{bad}");
        }
        for on in ["1", "true", "TRUE", "Yes", "on"] {
            assert!(truthy(on), "{on}");
        }
        for off in ["", "0", "false", "no", "off", "2"] {
            assert!(!truthy(off), "{off}");
        }
    }

    /// A resolver whose host has the addresses `addrs`, and the number of connection attempts it
    /// made; dialling an address in `dead` waits out the timeout as an address that drops packets
    /// does, the others connect at once. The bounds asserted below are far from both sides: a
    /// race takes tens of milliseconds, what it replaces takes whole connect timeouts.
    fn eyeballs(addrs: Vec<SocketAddr>, dead: Vec<SocketAddr>, connect: Duration) -> (Eyeballs, Arc<AtomicUsize>) {
        let dials = Arc::new(AtomicUsize::new(0));
        let count = dials.clone();
        let e = Eyeballs::new(
            Duration::from_secs(2),
            connect,
            Duration::from_millis(50),
            Arc::new(move |_, _| Ok(addrs.clone())),
            Arc::new(move |a, t| {
                count.fetch_add(1, Ordering::SeqCst);
                if dead.contains(&a) {
                    std::thread::sleep(t);
                    Err(std::io::Error::new(std::io::ErrorKind::TimedOut, "connection timed out"))
                } else {
                    Ok(())
                }
            }),
        );
        (e, dials)
    }

    #[test]
    fn dead_first_address_costs_the_stagger() {
        let (dead6, live4) = (v("[2001:db8::1]:443"), v("192.0.2.7:443"));
        let (e, dials) = eyeballs(vec![dead6, live4], vec![dead6], Duration::from_secs(10));
        let t = Instant::now();
        let got = e.addresses("hub.test", 443).unwrap();
        assert_eq!(got, [live4, dead6]);
        assert!(t.elapsed() < Duration::from_secs(3), "{:?}", t.elapsed());
        assert_eq!(dials.load(Ordering::SeqCst), 2);
        // The winner is remembered: no second race, and the dead address stays behind it.
        assert_eq!(e.addresses("hub.test", 443).unwrap()[0], live4);
        assert_eq!(dials.load(Ordering::SeqCst), 2);
        // Per host and port.
        assert_eq!(e.addresses("hub.test", 80).unwrap()[0], live4);
        assert_eq!(dials.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn at_most_three_addresses_are_handed_on() {
        let addrs: Vec<SocketAddr> = (1..=8)
            .map(|i| v(&format!("[2001:db8::{i}]:443")))
            .chain([v("192.0.2.1:443"), v("192.0.2.2:443")])
            .collect();
        let (e, _) = eyeballs(addrs.clone(), addrs[..8].to_vec(), Duration::from_secs(10));
        let t = Instant::now();
        // The first IPv4 address is the second attempt, one stagger after the first.
        // Behind it the other address of its family, then one of the family that did not answer.
        assert_eq!(e.addresses("hub.test", 443).unwrap(), [addrs[8], addrs[9], addrs[0]]);
        assert!(t.elapsed() < Duration::from_secs(3), "{:?}", t.elapsed());
    }

    #[test]
    fn all_addresses_dead_fails_within_the_connect_timeout() {
        let addrs = vec![v("[2001:db8::1]:443"), v("192.0.2.7:443"), v("[2001:db8::2]:443")];
        let (e, dials) = eyeballs(addrs.clone(), addrs, Duration::from_secs(1));
        let t = Instant::now();
        let err = e.addresses("hub.test", 443).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::TimedOut);
        assert_eq!(dials.load(Ordering::SeqCst), 3);
        // Two staggers and one connect timeout (1.1 s), not three connect timeouts (3 s).
        assert!(t.elapsed() < Duration::from_millis(2500), "{:?}", t.elapsed());
    }

    #[test]
    fn refused_addresses_do_not_wait_for_the_stagger() {
        let (a, b, c) = (v("192.0.2.1:443"), v("192.0.2.2:443"), v("192.0.2.3:443"));
        let e = Eyeballs::new(
            Duration::from_secs(2),
            Duration::from_secs(10),
            Duration::from_secs(10),
            Arc::new(move |_, _| Ok(vec![a, b, c])),
            Arc::new(move |x, _| {
                if x == c {
                    Ok(())
                } else {
                    Err(std::io::Error::new(std::io::ErrorKind::ConnectionRefused, "refused"))
                }
            }),
        );
        let t = Instant::now();
        assert_eq!(e.addresses("hub.test", 443).unwrap()[0], c);
        assert!(t.elapsed() < Duration::from_secs(5), "{:?}", t.elapsed());
    }

    #[test]
    fn lookup_has_a_deadline_and_literals_skip_it() {
        let e = Eyeballs::new(
            Duration::from_millis(100),
            Duration::from_secs(1),
            Duration::from_millis(50),
            Arc::new(|_, _| {
                std::thread::sleep(Duration::from_secs(10));
                Ok(Vec::new())
            }),
            Arc::new(|_, _| Ok(())),
        );
        let t = Instant::now();
        let err = e.addresses("hub.test", 443).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::TimedOut);
        assert!(t.elapsed() < Duration::from_secs(5), "{:?}", t.elapsed());
        assert_eq!(e.addresses("127.0.0.1", 80).unwrap(), [v("127.0.0.1:80")]);
        assert_eq!(e.addresses("[::1]", 80).unwrap(), [v("[::1]:80")]);
        assert_eq!(e.addresses("::1", 80).unwrap(), [v("[::1]:80")]);
    }

    #[test]
    fn a_small_request_has_one_deadline() {
        // A server that accepts and says nothing, with every other bound far away.
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/", l.local_addr().unwrap());
        let far = Duration::from_secs(30);
        let cfg = Cfg {
            hf: String::new(),
            mirror_api: String::new(),
            hub: None,
            offline: false,
            check: far,
            meta: far,
            connect: far,
            stall: far,
            retries: 0,
            backoff: Duration::ZERO,
            down_for: far,
            resolver: Arc::new(Eyeballs::system(far, far, far)),
        };
        let t = Instant::now();
        let e = send(request(&cfg, &url, false).unwrap(), Duration::from_millis(300)).unwrap_err();
        let took = t.elapsed();
        assert!(is_unreachable(&e), "{e:#}");
        assert!(format!("{e:#}").contains("no answer within 300 ms"), "{e:#}");
        assert!(took >= Duration::from_millis(250) && took < Duration::from_secs(10), "{took:?}");
        // Through a context, as the callers add one.
        assert!(is_unreachable(&e.context("the listing")));
        assert!(!is_unreachable(&anyhow!("HTTP 404")));
        drop(l);
    }
}
