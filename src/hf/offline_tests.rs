//! The hub client against local servers that misbehave the way a bad link does: no external
//! network, and timeouts short enough to keep each test near a second.
//!
//! A server that stalls does so for [`STALL`], far longer than any bound the tests configure, and
//! the tests allow [`SOON`] for what should take a fraction of a second: a bound that does not
//! hold shows as a test that takes [`STALL`], and a slow machine does not fail them.

use super::net::{self, Cfg, Eyeballs};
use super::*;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

const STALL: Duration = Duration::from_secs(30);
const SOON: Duration = Duration::from_secs(10);

/// A local HTTP server: `handler(request head, connection)` writes the answer, and the
/// connection closes when it returns. The count is of the connections accepted.
fn serve(handler: impl Fn(&str, &mut TcpStream) + Send + Sync + 'static) -> (SocketAddr, Arc<AtomicUsize>) {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap();
    let hits = Arc::new(AtomicUsize::new(0));
    let (count, handler) = (hits.clone(), Arc::new(handler));
    std::thread::spawn(move || {
        for mut s in l.incoming().flatten() {
            count.fetch_add(1, Ordering::SeqCst);
            let handler = handler.clone();
            std::thread::spawn(move || {
                let mut head = Vec::new();
                let mut b = [0u8; 1];
                while !head.ends_with(b"\r\n\r\n") {
                    match s.read(&mut b) {
                        Ok(1) => head.push(b[0]),
                        _ => return,
                    }
                }
                handler(&String::from_utf8_lossy(&head), &mut s);
            });
        }
    });
    (addr, hits)
}

fn reply(s: &mut TcpStream, status: &str, headers: &str, body: &[u8]) {
    let _ = write!(
        s,
        "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n{headers}\r\n",
        body.len()
    );
    let _ = s.write_all(body);
}

/// Keeps a connection open and silent.
fn stall() {
    std::thread::sleep(STALL);
}

/// A server that accepts and never answers.
fn silent() -> (SocketAddr, Arc<AtomicUsize>) {
    serve(|_, _| stall())
}

/// The path of a request head.
fn path_of(head: &str) -> &str {
    head.split_whitespace().nth(1).unwrap_or("")
}

/// huggingface.co's listing of a repository with these files.
fn hf_listing(files: &[(&str, usize)]) -> Vec<u8> {
    let siblings: Vec<serde_json::Value> = files.iter().map(|(n, s)| serde_json::json!({"rfilename": n, "size": s})).collect();
    serde_json::json!({ "siblings": siblings }).to_string().into_bytes()
}

/// A configuration for hubs at these addresses: the deadlines of the small requests and the
/// stall timeout in the hundreds of milliseconds, a connection attempt as long as [`STALL`].
fn cfg(hf: SocketAddr, mirror: SocketAddr, hub: Hub) -> Cfg {
    Cfg {
        hf: format!("http://{hf}"),
        mirror_api: format!("http://{mirror}/Model:get"),
        hub: Some(hub),
        offline: false,
        check: Duration::from_millis(400),
        meta: Duration::from_millis(1200),
        connect: STALL,
        stall: Duration::from_millis(300),
        retries: 2,
        backoff: Duration::from_millis(10),
        down_for: Duration::from_secs(600),
        resolver: Arc::new(Eyeballs::system(STALL, STALL, Duration::from_millis(50))),
    }
}

/// A new cache folder for one test, and the folder of `repo` in it.
fn cache(test: &str, repo: &str) -> (PathBuf, PathBuf) {
    let root = std::env::temp_dir().join(format!("oxinfer-hf-{test}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let dir = root.join(repo.replace('/', "--"));
    std::fs::create_dir_all(&dir).unwrap();
    (root, dir)
}

/// A whole safetensors model in `dir`, as far as the hub client can tell.
fn cache_model(dir: &Path) {
    std::fs::write(dir.join("config.json"), b"{}").unwrap();
    std::fs::write(dir.join("model.safetensors"), b"weights").unwrap();
}

#[test]
fn cached_model_loads_when_the_hub_never_answers() {
    let ((hf, hf_hits), (mirror, mirror_hits)) = (silent(), silent());
    let hits = move || (mirror_hits.load(Ordering::SeqCst), hf_hits.load(Ordering::SeqCst));
    let (root, dir) = cache("silent", "org/silent");
    cache_model(&dir);
    let mut c = cfg(hf, mirror, Hub::Auto);
    // Only the deadline of the whole listing can end the wait.
    c.stall = STALL;
    net::with_cfg(c.clone(), || {
        // The two hubs share the deadline: the mirror has half of it, huggingface.co the rest.
        let e = format!("{:#}", listing(&c, "org/silent", Duration::from_millis(400)).err().unwrap());
        assert!(e.contains("Model:get?key=org/silent: no answer within 200 ms; "), "{e}");
        assert!(e.contains("/api/models/org/silent?blobs=true: no answer within "), "{e}");
        assert_eq!(hits(), (1, 1));

        let t = Instant::now();
        assert_eq!(resolve("org/silent", Some(&root)).unwrap(), dir);
        let took = t.elapsed();
        assert_eq!(hits(), (2, 2));
        assert!(took >= Duration::from_millis(350) && took < SOON, "{took:?}");
        // While the hub is held to be down, a load the cache can serve does not ask at all.
        assert_eq!(resolve("org/silent", Some(&root)).unwrap(), dir);
        assert_eq!(plan("org/silent", Some(&root)).unwrap().missing(), 0);
        assert_eq!(resolve_files("org/silent", &["config.json"], Some(&root)).unwrap(), dir);
        assert_eq!(hits(), (2, 2));
    });
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn cached_model_loads_when_something_else_answers() {
    // A captive portal's page in place of the listing, and a server that is overloaded.
    let (portal, portal_hits) = serve(|_, s| reply(s, "200 OK", "Content-Type: text/html\r\n", b"<html>Sign in</html>"));
    let (busy, busy_hits) = serve(|_, s| reply(s, "503 Service Unavailable", "", b"later"));
    for (name, at, hits) in [("portal", portal, portal_hits), ("busy", busy, busy_hits)] {
        let repo = format!("org/{name}");
        let (root, dir) = cache(name, &repo);
        cache_model(&dir);
        net::with_cfg(cfg(at, at, Hub::Auto), || {
            assert_eq!(resolve(&repo, Some(&root)).unwrap(), dir);
        });
        // The mirror, then huggingface.co.
        assert_eq!(hits.load(Ordering::SeqCst), 2);
        std::fs::remove_dir_all(&root).unwrap();
    }
}

#[test]
fn offline_asks_nothing() {
    let (hub, hits) = serve(|_, s| reply(s, "200 OK", "", &hf_listing(&[("config.json", 2), ("model.safetensors", 7)])));
    let (root, dir) = cache("offline", "org/offline");
    let mut c = cfg(hub, hub, Hub::HuggingFace);
    c.offline = true;
    net::with_cfg(c, || {
        // Nothing of the repository in the cache.
        let e = format!("{:#}", resolve("org/offline", Some(&root)).unwrap_err());
        assert!(e.contains("offline") && e.contains("not in the cache"), "{e}");
        // A whole model: a folder with subfolders and a download that was interrupted beside it.
        cache_model(&dir);
        std::fs::create_dir_all(dir.join("1_Pooling")).unwrap();
        std::fs::write(dir.join("1_Pooling/config.json"), b"{}").unwrap();
        std::fs::write(dir.join("tokenizer.json.rsurlpart"), b"half").unwrap();
        std::fs::write(dir.join("empty.json"), b"").unwrap();
        let list = cached_listing(&dir);
        let mut names: Vec<&str> = list.keys().map(String::as_str).collect();
        names.sort_unstable();
        assert_eq!(names, ["1_Pooling/config.json", "config.json", "model.safetensors"]);
        assert_eq!(resolve("org/offline", Some(&root)).unwrap(), dir);
        let p = plan("org/offline", Some(&root)).unwrap();
        assert_eq!((p.missing(), p.total()), (0, 9));
        assert_eq!(
            resolve_files("org/offline", &["config.json", "absent.json"], Some(&root)).unwrap(),
            dir
        );
        assert_eq!(resolve_prefix("org/offline", "1_Pooling", Some(&root)).unwrap(), dir);
        assert_eq!(resolve_file("org/offline", Some(&root)).unwrap(), dir.join("model.safetensors"));
        // The weights gone: the load fails at once and says what the cache lacks.
        std::fs::remove_file(dir.join("model.safetensors")).unwrap();
        let e = format!("{:#}", resolve("org/offline", Some(&root)).unwrap_err());
        assert!(
            e.contains("offline") && e.contains("cannot serve this load") && e.contains("no safetensors"),
            "{e}"
        );
        assert!(plan("org/offline", Some(&root)).is_err());
    });
    assert_eq!(hits.load(Ordering::SeqCst), 0);
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn offline_gguf_with_parts_and_without_projector() {
    let (hub, hits) = silent();
    let (root, dir) = cache("offline-gguf", "org/gguf");
    let mut c = cfg(hub, hub, Hub::Auto);
    c.offline = true;
    net::with_cfg(c, || {
        std::fs::create_dir_all(dir.join("Q4")).unwrap();
        std::fs::write(dir.join("Q4/x-Q4-00001-of-00002.gguf"), b"GGUF1").unwrap();
        // A part missing.
        let e = format!("{:#}", resolve_gguf("org/gguf", "Q4", Some(&root)).unwrap_err());
        assert!(e.contains("offline") && e.contains("x-Q4-00002-of-00002.gguf"), "{e}");
        assert!(plan("org/gguf:Q4", Some(&root)).is_err());
        std::fs::write(dir.join("Q4/x-Q4-00002-of-00002.gguf"), b"GGUF2").unwrap();
        // Whole, and no projector in the cache: none is asked for.
        assert_eq!(
            resolve_gguf("org/gguf", "Q4", Some(&root)).unwrap(),
            dir.join("Q4/x-Q4-00001-of-00002.gguf")
        );
        assert_eq!(plan("org/gguf:Q4", Some(&root)).unwrap().files.len(), 2);
    });
    assert_eq!(hits.load(Ordering::SeqCst), 0);
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn a_listing_the_load_needs_fails_within_its_bounds() {
    let (hub, _) = silent();
    let (root, _) = cache("needed", "org/needed");
    let mut c = cfg(hub, hub, Hub::HuggingFace);
    net::with_cfg(c.clone(), || {
        // The silence of the server ends the wait (300 ms).
        let t = Instant::now();
        let e = format!("{:#}", resolve("org/needed", Some(&root)).unwrap_err());
        assert!(e.contains("timed out"), "{e}");
        assert!(t.elapsed() < SOON, "{:?}", t.elapsed());
    });
    // A server allowed to be silent for longer than the whole listing may take: the deadline of
    // the listing does (1.2 s).
    c.stall = STALL;
    net::with_cfg(c, || {
        let t = Instant::now();
        let e = format!("{:#}", resolve("org/needed", Some(&root)).unwrap_err());
        let took = t.elapsed();
        assert!(e.contains("no answer within 1.2 s"), "{e}");
        assert!(took >= Duration::from_millis(1100) && took < SOON, "{took:?}");
    });
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn an_incomplete_cache_and_a_silent_hub_say_both() {
    let (hub, hits) = silent();
    let (root, dir) = cache("incomplete", "org/incomplete");
    std::fs::write(dir.join("config.json"), b"{}").unwrap();
    net::with_cfg(cfg(hub, hub, Hub::HuggingFace), || {
        let t = Instant::now();
        let e = format!("{:#}", resolve("org/incomplete", Some(&root)).unwrap_err());
        // The short check, then the listing the load needs: one after the other, both bounded.
        assert!(t.elapsed() < SOON, "{:?}", t.elapsed());
        assert!(e.contains("the hub did not answer") && e.contains("no safetensors"), "{e}");
    });
    assert_eq!(hits.load(Ordering::SeqCst), 2);
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn a_body_that_stops_is_resumed() {
    let body: Arc<Vec<u8>> = Arc::new((0..200_000u32).map(|i| (i % 251) as u8).collect());
    let ranges = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let (data, seen) = (body.clone(), ranges.clone());
    let (hub, _) = serve(move |head, s| {
        let n = data.len();
        match path_of(head) {
            p if p.starts_with("/api/models/") => reply(s, "200 OK", "", &hf_listing(&[("config.json", 2), ("model.safetensors", n)])),
            p if p.ends_with("/config.json") => reply(s, "200 OK", "", b"{}"),
            _ => {
                let range = head
                    .lines()
                    .find_map(|l| l.to_ascii_lowercase().strip_prefix("range: bytes=").map(str::to_string));
                seen.lock().unwrap().push(range.clone().unwrap_or_default());
                match range.and_then(|r| r.trim().trim_end_matches('-').parse::<usize>().ok()) {
                    // The first answer stops after 60 000 bytes and says nothing more.
                    None => {
                        let _ = write!(
                            s,
                            "HTTP/1.1 200 OK\r\nContent-Length: {n}\r\nAccept-Ranges: bytes\r\nETag: \"v1\"\r\n\r\n"
                        );
                        let _ = s.write_all(&data[..60_000]);
                        let _ = s.flush();
                        stall();
                    }
                    Some(from) => reply(
                        s,
                        "206 Partial Content",
                        &format!("Content-Range: bytes {from}-{}/{n}\r\nETag: \"v1\"\r\n", n - 1),
                        &data[from..],
                    ),
                }
            }
        }
    });
    let (root, dir) = cache("resume", "org/resume");
    net::with_cfg(cfg(hub, hub, Hub::HuggingFace), || {
        let t = Instant::now();
        assert_eq!(resolve("org/resume", Some(&root)).unwrap(), dir);
        // One stall timeout (300 ms), then the rest from where the first answer stopped.
        assert!(t.elapsed() < SOON, "{:?}", t.elapsed());
    });
    assert_eq!(std::fs::read(dir.join("model.safetensors")).unwrap(), *body);
    assert!(!dir.join("model.safetensors.rsurlpart").exists());
    let ranges = ranges.lock().unwrap();
    assert_eq!(ranges.len(), 2, "{ranges:?}");
    assert_eq!(ranges[0], "");
    let from: usize = ranges[1].trim().trim_end_matches('-').parse().unwrap();
    assert!(from > 0 && from <= 60_000, "{ranges:?}");
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn a_body_that_never_comes_gives_up() {
    let (hub, hits) = serve(|head, s| match path_of(head) {
        p if p.starts_with("/api/models/") => reply(s, "200 OK", "", &hf_listing(&[("config.json", 2), ("model.safetensors", 1000)])),
        p if p.ends_with("/config.json") => reply(s, "200 OK", "", b"{}"),
        _ => stall(),
    });
    let (root, dir) = cache("stuck", "org/stuck");
    net::with_cfg(cfg(hub, hub, Hub::HuggingFace), || {
        let t = Instant::now();
        let e = format!("{:#}", resolve("org/stuck", Some(&root)).unwrap_err());
        assert!(e.contains("download org/stuck/model.safetensors"), "{e}");
        // The first attempt and two retries, a stall timeout (300 ms) each.
        assert!(t.elapsed() < SOON, "{:?}", t.elapsed());
    });
    // The listing, config.json and three attempts at the weights.
    assert_eq!(hits.load(Ordering::SeqCst), 5);
    assert!(!dir.join("model.safetensors").exists());
    std::fs::remove_dir_all(&root).unwrap();
}

/// A resolver for a hub whose name has the addresses `addrs`, of which those in `dead` drop
/// every packet (the attempt waits out its timeout) and the others are dialled for real.
fn name_with(addrs: Vec<SocketAddr>, dead: Vec<SocketAddr>) -> Arc<Eyeballs> {
    Arc::new(Eyeballs::new(
        STALL,
        STALL,
        Duration::from_millis(50),
        Arc::new(move |_, _| Ok(addrs.clone())),
        Arc::new(move |a, t| {
            if dead.contains(&a) {
                std::thread::sleep(t);
                return Err(std::io::Error::new(std::io::ErrorKind::TimedOut, "connection timed out"));
            }
            TcpStream::connect_timeout(&a, t).map(drop)
        }),
    ))
}

#[test]
fn a_dead_first_address_costs_a_fraction_of_a_second() {
    let (live, hits) = serve(|head, s| match path_of(head) {
        p if p.starts_with("/api/models/") => reply(s, "200 OK", "", &hf_listing(&[("config.json", 2), ("model.safetensors", 7)])),
        p if p.ends_with("/config.json") => reply(s, "200 OK", "", b"{}"),
        _ => reply(s, "200 OK", "Accept-Ranges: bytes\r\n", b"weights"),
    });
    // The IPv6 address the name also has, and that the resolver gives first: its route went away.
    let dead: SocketAddr = format!("[2001:db8::1]:{}", live.port()).parse().unwrap();
    let (root, dir) = cache("eyeballs", "org/eyeballs");
    let mut c = cfg(live, live, Hub::HuggingFace);
    c.hf = format!("http://hub.test:{}", live.port());
    c.resolver = name_with(vec![dead, live], vec![dead]);
    net::with_cfg(c, || {
        let t = Instant::now();
        // The listing and two files, each on a new connection, with nothing in the cache.
        assert_eq!(resolve("org/eyeballs", Some(&root)).unwrap(), dir);
        assert!(t.elapsed() < SOON, "{:?}", t.elapsed());
    });
    assert_eq!(std::fs::read(dir.join("model.safetensors")).unwrap(), b"weights");
    // One connection of the race, then the three requests.
    assert_eq!(hits.load(Ordering::SeqCst), 4);
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn cached_model_loads_when_no_address_connects() {
    let (unused, hits) = silent();
    let addrs: Vec<SocketAddr> = vec![
        format!("[2001:db8::1]:{}", unused.port()).parse().unwrap(),
        format!("192.0.2.7:{}", unused.port()).parse().unwrap(),
    ];
    let (root, dir) = cache("connect", "org/connect");
    cache_model(&dir);
    let mut c = cfg(unused, unused, Hub::Auto);
    c.hf = format!("http://hub.test:{}", unused.port());
    c.mirror_api = format!("http://mirror.test:{}/Model:get", unused.port());
    // A connection attempt takes far longer than the check: the deadline of the whole listing
    // is what ends the wait.
    c.resolver = name_with(addrs.clone(), addrs);
    net::with_cfg(c, || {
        let t = Instant::now();
        assert_eq!(resolve("org/connect", Some(&root)).unwrap(), dir);
        let took = t.elapsed();
        assert!(took >= Duration::from_millis(350) && took < SOON, "{took:?}");
    });
    assert_eq!(hits.load(Ordering::SeqCst), 0);
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn a_projector_that_cannot_be_fetched_does_not_fail_the_load() {
    let (hub, hits) = serve(|head, s| match path_of(head) {
        p if p.starts_with("/api/models/") => reply(s, "200 OK", "", &hf_listing(&[("x-Q4.gguf", 5), ("mmproj-F16.gguf", 900)])),
        p if p.ends_with("/x-Q4.gguf") => reply(s, "200 OK", "Accept-Ranges: bytes\r\n", b"GGUF!"),
        _ => stall(),
    });
    let (root, dir) = cache("projector", "org/projector");
    net::with_cfg(cfg(hub, hub, Hub::HuggingFace), || {
        let t = Instant::now();
        assert_eq!(resolve_gguf("org/projector", "Q4", Some(&root)).unwrap(), dir.join("x-Q4.gguf"));
        assert!(t.elapsed() < SOON, "{:?}", t.elapsed());
        assert!(!dir.join("mmproj-F16.gguf").exists());
        // The listing, the model, three attempts at the projector.
        assert_eq!(hits.load(Ordering::SeqCst), 5);
        // With `mmproj none` it is not asked for: the listing alone.
        resolve_gguf_with("org/projector", "Q4", Some(&root), &crate::pixtral::Mmproj::None).unwrap();
        assert_eq!(hits.load(Ordering::SeqCst), 6);
    });
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn the_hub_saying_no_is_not_passed_over() {
    let (hub, _) = serve(|_, s| reply(s, "404 Not Found", "", b"{}"));
    let (root, dir) = cache("gone", "org/gone");
    cache_model(&dir);
    net::with_cfg(cfg(hub, hub, Hub::HuggingFace), || {
        let e = format!("{:#}", resolve("org/gone", Some(&root)).unwrap_err());
        assert!(e.contains("repository not found"), "{e}");
    });
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn the_token_goes_to_the_endpoint_only() {
    let a: SocketAddr = "127.0.0.1:8080".parse().unwrap();
    let c = cfg(a, a, Hub::Auto);
    assert!(c.is_hf("http://127.0.0.1:8080/org/x/resolve/main/config.json"));
    assert!(!c.is_hf("http://127.0.0.1:80801/org/x/resolve/main/config.json"));
    assert!(!c.is_hf("http://127.0.0.1:8080.example.com/x"));
    assert!(!c.is_hf("https://example.com/x"));
}
