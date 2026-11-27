// A TCP link benchmark for docs/cluster.md: `rustc -O tools/cluster/linkbench.rs`, then
//   linkbench server ADDR:PORT                 on one machine
//   linkbench tput ADDR:PORT STREAMS SECONDS   on the other: what it sends, in Gbit/s
//   linkbench rr ADDR:PORT BYTES COUNT         a request of BYTES and its echo: p50 / p90 / p99
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::{Duration, Instant};
fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a[1].as_str() {
        "server" => {
            let l = TcpListener::bind(&a[2]).unwrap();
            for s in l.incoming() {
                let mut s = s.unwrap();
                std::thread::spawn(move || {
                    s.set_nodelay(true).ok();
                    let mut m = [0u8; 1];
                    if s.read_exact(&mut m).is_err() {
                        return;
                    }
                    if m[0] == b'T' {
                        let mut buf = vec![0u8; 1 << 20];
                        loop {
                            match s.read(&mut buf) {
                                Ok(0) | Err(_) => break,
                                _ => {}
                            }
                        }
                    } else {
                        let mut h = [0u8; 4];
                        let mut buf = vec![0u8; 1 << 26];
                        loop {
                            if s.read_exact(&mut h).is_err() {
                                break;
                            }
                            let n = u32::from_le_bytes(h) as usize;
                            if s.read_exact(&mut buf[..n]).is_err() {
                                break;
                            }
                            if s.write_all(&h).is_err() {
                                break;
                            }
                            if s.write_all(&buf[..n]).is_err() {
                                break;
                            }
                        }
                    }
                });
            }
        }
        "tput" => {
            let n: usize = a[3].parse().unwrap();
            let secs: u64 = a[4].parse().unwrap();
            let hs: Vec<_> = (0..n)
                .map(|_| {
                    let addr = a[2].clone();
                    std::thread::spawn(move || {
                        let mut s = TcpStream::connect(&addr).unwrap();
                        s.write_all(b"T").unwrap();
                        let buf = vec![7u8; 1 << 20];
                        let t = Instant::now();
                        let mut tot = 0u64;
                        while t.elapsed() < Duration::from_secs(secs) {
                            s.write_all(&buf).unwrap();
                            tot += buf.len() as u64;
                        }
                        (tot, t.elapsed().as_secs_f64())
                    })
                })
                .collect();
            let mut bps = 0.0;
            for h in hs {
                let (t, d) = h.join().unwrap();
                bps += t as f64 * 8.0 / d;
            }
            println!("tput {} streams={} {:.2} Gbit/s ({:.0} MB/s)", a[2], n, bps / 1e9, bps / 8e6);
        }
        "rr" => {
            let size: usize = a[3].parse().unwrap();
            let count: usize = a[4].parse().unwrap();
            let mut s = TcpStream::connect(&a[2]).unwrap();
            s.set_nodelay(true).unwrap();
            s.write_all(b"E").unwrap();
            let mut buf = vec![3u8; size + 4];
            buf[..4].copy_from_slice(&(size as u32).to_le_bytes());
            let mut back = vec![0u8; size + 4];
            let mut v = Vec::with_capacity(count);
            for _ in 0..count + 200 {
                let t = Instant::now();
                s.write_all(&buf).unwrap();
                s.read_exact(&mut back).unwrap();
                v.push(t.elapsed().as_secs_f64() * 1e6);
            }
            let mut v = v.split_off(200);
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p = |q: f64| v[((v.len() - 1) as f64 * q) as usize];
            println!(
                "rr {} size={} n={} p50={:.0}us p90={:.0}us p99={:.0}us max={:.0}us",
                a[2],
                size,
                count,
                p(0.5),
                p(0.9),
                p(0.99),
                p(1.0)
            );
        }
        _ => {}
    }
}
