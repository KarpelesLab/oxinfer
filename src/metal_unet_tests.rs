//! Tests of the whole-pass UNet session (`metal_unet.rs`) against a float32 reference.

use super::*;
use crate::gpu::unet::{Block, Graph, Item, Lin, Norm, Op, Transformer};
use crate::quant::PageVec;
use std::borrow::Cow;

fn rnd(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
}

fn bf(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

fn unbf(u: u16) -> f32 {
    f32::from_bits((u as u32) << 16)
}

/// A layer's weights (bf16, uploaded) and bias.
struct W {
    w: PageVec<u16>,
    dev: crate::gpu::DeviceWeight,
    b: Vec<f32>,
    n_out: usize,
    n_in: usize,
}

impl W {
    fn new(be: &MetalBackend, n_out: usize, n_in: usize, scale: f32, seed: &mut u64) -> W {
        let mut w = PageVec::<u16>::zeroed(n_out * n_in);
        for v in w.iter_mut() {
            *v = bf(rnd(seed) * scale);
        }
        let dev = be.upload(&WeightRef::Bf16(&w), n_out, n_in).unwrap();
        let b = (0..n_out).map(|_| rnd(seed) * 0.2).collect();
        W { w, dev, b, n_out, n_in }
    }

    fn lin(&self) -> Lin<'_> {
        Lin {
            w: WeightRef::Bf16(&self.w),
            dev: &self.dev,
            n_out: self.n_out,
            n_in: self.n_in,
            bias: Some(&self.b),
        }
    }

    /// `y[r] = W x[r] (+ b)` over rows of n_in.
    fn apply(&self, x: &[f32], bias: bool) -> Vec<f32> {
        let rows = x.len() / self.n_in;
        let mut y = vec![0f32; rows * self.n_out];
        for r in 0..rows {
            let xr = &x[r * self.n_in..(r + 1) * self.n_in];
            for o in 0..self.n_out {
                let wr = &self.w[o * self.n_in..(o + 1) * self.n_in];
                let s: f32 = wr.iter().zip(xr).map(|(&a, &b)| unbf(a) * b).sum();
                y[r * self.n_out + o] = s + if bias { self.b[o] } else { 0.0 };
            }
        }
        y
    }
}

struct NormW {
    g: Vec<f32>,
    b: Vec<f32>,
}

impl NormW {
    fn new(c: usize, seed: &mut u64) -> NormW {
        NormW {
            g: (0..c).map(|_| 1.0 + rnd(seed) * 0.4).collect(),
            b: (0..c).map(|_| rnd(seed) * 0.4).collect(),
        }
    }

    fn norm(&self, groups: usize) -> Norm<'_> {
        Norm {
            gamma: &self.g,
            beta: &self.b,
            eps: 1e-5,
            groups,
        }
    }

    /// Group norm (`groups` > 0, over all pixels of one image) or layer norm (per row), then SiLU.
    fn apply(&self, x: &[f32], c: usize, groups: usize, silu: bool) -> Vec<f32> {
        let n = x.len() / c;
        let mut y = x.to_vec();
        let stats = |v: Vec<f64>| {
            let m = v.iter().sum::<f64>() / v.len() as f64;
            let var = v.iter().map(|a| (a - m) * (a - m)).sum::<f64>() / v.len() as f64;
            (m as f32, (1.0 / (var + 1e-5).sqrt()) as f32)
        };
        let spans: Vec<(Vec<usize>, std::ops::Range<usize>)> = if let Some(cpg) = c.checked_div(groups) {
            (0..groups).map(|g| ((0..n).collect(), g * cpg..(g + 1) * cpg)).collect()
        } else {
            (0..n).map(|p| (vec![p], 0..c)).collect()
        };
        for (px, chs) in spans {
            let (m, inv) = stats(px.iter().flat_map(|&p| chs.clone().map(move |ch| x[p * c + ch] as f64)).collect());
            for &p in &px {
                for ch in chs.clone() {
                    y[p * c + ch] = (x[p * c + ch] - m) * inv * self.g[ch] + self.b[ch];
                }
            }
        }
        if silu {
            y.iter_mut().for_each(|v| *v /= 1.0 + (-*v).exp());
        }
        y
    }
}

/// k x k convolution of `[h, w, cin]` (zero padding `pad`), weights `[cout][ky][kx][cin]`.
#[allow(clippy::too_many_arguments)]
fn conv(x: &[f32], h: usize, w: usize, cin: usize, wt: &W, k: usize, stride: usize, pad: usize) -> Vec<f32> {
    let oh = (h + 2 * pad - k) / stride + 1;
    let ow = (w + 2 * pad - k) / stride + 1;
    let mut cols = vec![0f32; oh * ow * k * k * cin];
    for oy in 0..oh {
        for ox in 0..ow {
            for ky in 0..k {
                for kx in 0..k {
                    let iy = (oy * stride + ky) as isize - pad as isize;
                    let ix = (ox * stride + kx) as isize - pad as isize;
                    if iy < 0 || ix < 0 || iy >= h as isize || ix >= w as isize {
                        continue;
                    }
                    let d = ((oy * ow + ox) * k * k + ky * k + kx) * cin;
                    let s = (iy as usize * w + ix as usize) * cin;
                    cols[d..d + cin].copy_from_slice(&x[s..s + cin]);
                }
            }
        }
    }
    wt.apply(&cols, true)
}

fn attention(q: &[f32], k: &[f32], v: &[f32], heads: usize, c: usize) -> Vec<f32> {
    let (tq, tk, dh) = (q.len() / c, k.len() / c, c / heads);
    let mut o = vec![0f32; tq * c];
    for h in 0..heads {
        for i in 0..tq {
            let s: Vec<f32> = (0..tk)
                .map(|j| (0..dh).map(|d| q[i * c + h * dh + d] * k[j * c + h * dh + d]).sum::<f32>() / (dh as f32).sqrt())
                .collect();
            let m = s.iter().fold(f32::MIN, |a, &b| a.max(b));
            let e: Vec<f32> = s.iter().map(|v| (v - m).exp()).collect();
            let z: f32 = e.iter().sum();
            for d in 0..dh {
                o[i * c + h * dh + d] = (0..tk).map(|j| e[j] * v[j * c + h * dh + d]).sum::<f32>() / z;
            }
        }
    }
    o
}

fn add(a: &mut [f32], b: &[f32]) {
    a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
}

fn rel(y: &[f32], r: &[f32]) -> f64 {
    let (mut n, mut d) = (0f64, 0f64);
    for (a, b) in y.iter().zip(r) {
        n += ((a - b) as f64).powi(2);
        d += (*b as f64).powi(2);
    }
    (n / d.max(1e-30)).sqrt()
}

/// Convolutions (3x3 with group norm + SiLU and per-image biases, a residual add, strided,
/// upsampled), a concatenation and a transformer, for two images, against float32.
#[test]
fn pass_matches_reference() {
    let Ok(be) = MetalBackend::new() else { return };
    let be: &'static MetalBackend = Box::leak(Box::new(be));
    let Some(mut sess) = be.unet_session_impl() else { return };
    let mut seed = 7u64;
    let s = &mut seed;
    let (h, w, c0, c, cd, m, nb, inner) = (8, 6, 32, 64, 64, 5, 2, 256);
    let ct = 2 * c;
    let wc1 = W::new(be, c, 9 * c0, 0.1, s);
    let wc2 = W::new(be, c, 9 * c, 0.05, s);
    let wdn = W::new(be, c, 9 * c, 0.05, s);
    let wup = W::new(be, c, 9 * c, 0.05, s);
    let (pin, pout) = (W::new(be, ct, ct, 0.1, s), W::new(be, ct, ct, 0.1, s));
    let a1: Vec<W> = (0..4).map(|_| W::new(be, ct, ct, 0.1, s)).collect();
    let a2: Vec<W> = (0..4)
        .map(|i| W::new(be, ct, if i == 1 || i == 2 { cd } else { ct }, 0.1, s))
        .collect();
    let ff1 = W::new(be, 2 * inner, ct, 0.1, s);
    let ff2 = W::new(be, ct, inner, 0.05, s);
    let (n1, n2, nt) = (NormW::new(c0, s), NormW::new(c, s), NormW::new(ct, s));
    let lns: Vec<NormW> = (0..3).map(|_| NormW::new(ct, s)).collect();
    let bias_img: Vec<f32> = (0..nb * c).map(|_| rnd(s)).collect();
    let lat: Vec<Vec<f32>> = (0..nb).map(|_| (0..h * w * c0).map(|_| rnd(s) * 4.0).collect()).collect();
    let ctx: Vec<Vec<f32>> = (0..nb).map(|_| (0..m * cd).map(|_| rnd(s)).collect()).collect();
    // Self-attention: 4 heads of 32 (padded to 48 for the flash kernels, with biases);
    // cross-attention: 2 heads of 64 without.
    fn attn(ws: &[W], heads: usize, bias: bool) -> crate::gpu::unet::Attn<'_> {
        let l = |i: usize| if bias { ws[i].lin() } else { Lin { bias: None, ..ws[i].lin() } };
        crate::gpu::unet::Attn {
            q: l(0),
            k: l(1),
            v: l(2),
            o: ws[3].lin(),
            heads,
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn cv<'a>(x: usize, out: usize, w: &'a W, stride: usize, up: bool, pre: Option<(Norm<'a>, bool)>, add: Option<usize>) -> Op<'a> {
        Op::Conv {
            x,
            out,
            w: w.lin(),
            k: 3,
            stride,
            pad: 1,
            up,
            pre,
            bias: None,
            bias_per_image: false,
            add,
        }
    }
    let g = Graph {
        tensors: vec![
            [h, w, c0],
            [h, w, c],
            [h, w, c],
            [h / 2, w / 2, c],
            [h, w, c],
            [h, w, ct],
            [h, w, ct],
        ],
        ops: vec![
            Op::Conv {
                x: 0,
                out: 1,
                w: wc1.lin(),
                k: 3,
                stride: 1,
                pad: 1,
                up: false,
                pre: Some((n1.norm(8), true)),
                bias: Some(Cow::Borrowed(&bias_img)),
                bias_per_image: true,
                add: None,
            },
            cv(1, 2, &wc2, 1, false, Some((n2.norm(16), true)), Some(1)),
            cv(2, 3, &wdn, 2, false, None, None),
            cv(3, 4, &wup, 1, true, None, None),
            Op::Concat { a: 4, b: 2, out: 5 },
            Op::Transformer {
                x: 5,
                out: 6,
                t: Transformer {
                    norm: nt.norm(32),
                    proj_in: pin.lin(),
                    blocks: vec![Block {
                        n1: lns[0].norm(0),
                        a1: attn(&a1, 4, true),
                        n2: lns[1].norm(0),
                        a2: attn(&a2, 2, false),
                        n3: lns[2].norm(0),
                        ff1: ff1.lin(),
                        ff2: ff2.lin(),
                    }],
                    proj_out: pout.lin(),
                },
            },
        ],
        input: 0,
        output: 6,
        ctx_dim: cd,
    };
    let items: Vec<Item> = (0..nb)
        .map(|b| Item {
            latent: &lat[b],
            ctx: &ctx[b],
            ctx_len: m,
        })
        .collect();
    let mut out = vec![Vec::new(); nb];
    // Twice: the second pass reuses the session's buffers and the text keys / values the first
    // one computed, bit for bit the same output.
    assert!(sess.forward(&g, &items, &mut out), "device pass failed");
    let first = out.clone();
    assert!(sess.forward(&g, &items, &mut out), "device pass failed");
    assert!(first == out, "the cached cross-attention keys / values changed the output");
    // Other contexts recompute them (a different output), and the first ones again give the
    // first output.
    let ctx2: Vec<Vec<f32>> = ctx.iter().map(|c| c.iter().map(|v| v * 0.5).collect()).collect();
    let items2: Vec<Item> = (0..nb)
        .map(|b| Item {
            latent: &lat[b],
            ctx: &ctx2[b],
            ctx_len: m,
        })
        .collect();
    assert!(sess.forward(&g, &items2, &mut out), "device pass failed");
    assert!(first != out, "new contexts must not reuse the old keys / values");
    assert!(sess.forward(&g, &items, &mut out), "device pass failed");
    assert!(first == out, "back to the first contexts: not the first output");
    for b in 0..nb {
        let mut t1 = conv(&n1.apply(&lat[b], c0, 8, true), h, w, c0, &wc1, 3, 1, 1);
        for r in t1.chunks_mut(c) {
            for o in 0..c {
                r[o] += bias_img[b * c + o] - wc1.b[o];
            }
        }
        let mut t2 = conv(&n2.apply(&t1, c, 16, true), h, w, c, &wc2, 3, 1, 1);
        add(&mut t2, &t1);
        let t3 = conv(&t2, h, w, c, &wdn, 3, 2, 1);
        let mut upx = vec![0f32; h * w * c];
        for y in 0..h {
            for x in 0..w {
                let s = ((y / 2) * (w / 2) + x / 2) * c;
                upx[(y * w + x) * c..(y * w + x + 1) * c].copy_from_slice(&t3[s..s + c]);
            }
        }
        let t4 = conv(&upx, h, w, c, &wup, 3, 1, 1);
        let mut t5 = vec![0f32; h * w * ct];
        for p in 0..h * w {
            t5[p * ct..p * ct + c].copy_from_slice(&t4[p * c..(p + 1) * c]);
            t5[p * ct + c..(p + 1) * ct].copy_from_slice(&t2[p * c..(p + 1) * c]);
        }
        let mut hs = pin.apply(&nt.apply(&t5, ct, 32, false), true);
        let hn = lns[0].apply(&hs, ct, 0, false);
        let o = attention(&a1[0].apply(&hn, true), &a1[1].apply(&hn, true), &a1[2].apply(&hn, true), 4, ct);
        add(&mut hs, &a1[3].apply(&o, true));
        let hn = lns[1].apply(&hs, ct, 0, false);
        let o = attention(
            &a2[0].apply(&hn, false),
            &a2[1].apply(&ctx[b], false),
            &a2[2].apply(&ctx[b], false),
            2,
            ct,
        );
        add(&mut hs, &a2[3].apply(&o, true));
        let f = ff1.apply(&lns[2].apply(&hs, ct, 0, false), true);
        let gg: Vec<f32> = f
            .chunks(2 * inner)
            .flat_map(|r| {
                (0..inner)
                    .map(|j| r[j] * crate::sd::ops::gelu_erf(r[inner + j]))
                    .collect::<Vec<_>>()
            })
            .collect();
        add(&mut hs, &ff2.apply(&gg, true));
        let mut t6 = pout.apply(&hs, true);
        add(&mut t6, &t5);
        let e = rel(&out[b], &t6);
        assert!(e < 3e-3, "image {b}: rel L2 {e:.2e}");
    }
}

/// GEMM kernel timings over UNet shapes (`cargo test --release --lib un_gemm_bench -- --ignored --nocapture`;
/// `un_gemm` / `un_gemm_m64`: whole tiles only, `plan`: as the passes run them (split last
/// tiles included), other names: kernels of OXINFER_MMA_EXTRA on a grid of (outputs / 64, rows / tm)).
#[test]
#[ignore]
fn un_gemm_bench() {
    let Ok(b) = MetalBackend::new() else { return };
    if b.mma().is_none_or(|m| !m.units) {
        return;
    }
    let be: &'static MetalBackend = Box::leak(Box::new(b));
    let f = Fns::new(be);
    let pipe = |n: &str| be.named_pipeline(n).unwrap();
    let names = std::env::var("UN_BENCH_KERNELS").unwrap_or("un_gemm,un_gemm_m64,plan".into());
    // UN_BENCH_SHAPES=rows:cin:n_out:taps,...
    let shapes: Vec<(usize, usize, usize, usize)> = match std::env::var("UN_BENCH_SHAPES") {
        Ok(v) => v
            .split(',')
            .map(|s| {
                let n: Vec<usize> = s.split(':').map(|x| x.parse().unwrap()).collect();
                (n[0], n[1], n[2], n[3])
            })
            .collect(),
        Err(_) => vec![
            (8192, 320, 320, 9),
            (8192, 320, 2560, 1),
            (8192, 1280, 320, 1),
            (2048, 640, 640, 9),
            (2048, 640, 5120, 1),
            (512, 1280, 1280, 9),
            (200, 1280, 1280, 9),
            (648, 1280, 1280, 9),
            (32768, 320, 320, 9),
            (8192, 640, 640, 9),
        ],
    };
    for &(rows, cin, nout, taps) in &shapes {
        let mut bx = DBuf::empty();
        let mut bw = DBuf::empty();
        let mut by = DBuf::empty();
        let mut bz = DBuf::empty();
        let mut bp = DBuf::empty();
        let a_rows = rows + 256;
        assert!(ensure(be, &mut bx, a_rows * cin * 2));
        // Twice the outputs: GEMMs with a gate (GEGLU) read the rows past n_out.
        assert!(ensure(be, &mut bw, 2 * nout * cin * taps * 2 + 4096));
        assert!(ensure(be, &mut by, rows * nout * 4));
        assert!(ensure(be, &mut bz, 4096));
        assert!(ensure(be, &mut bp, 4 * gpu_cores() * 128 * 64 * 4));
        let mut off = [0i32; 9];
        for (t, o) in off.iter_mut().enumerate().take(taps) {
            *o = t as i32 * 3;
        }
        let p = Ug {
            n_out: nout as u32,
            ldw: (cin * taps) as u32,
            cin: cin as u32,
            taps: taps as u32,
            off,
            a_rows: a_rows as u32,
            r0: 0,
            r_end: rows as u32,
            omode: 0,
            ldy: nout as u32,
            has_bias: 0,
            bstride: 0,
            add: 0,
            nimg: rows as u32,
            g: Geo::default(),
            scaled: 0,
            full: 0,
            rs: 1,
            kper: 0,
            nct: nout.div_ceil(64) as u32,
            tm: 0,
            nrem: 0,
            _p: 0,
        };
        let flop = 2.0 * rows as f64 * cin as f64 * taps as f64 * nout as f64;
        for name in names.split(',') {
            let whole = |tm: usize| Plan {
                tm,
                nct: nout.div_ceil(64),
                full: rows.div_ceil(tm) * nout.div_ceil(64),
                nrem: 0,
                rs: 1,
                kper: cin * taps,
            };
            let pl = match name {
                "un_gemm" => Some(whole(128)),
                "un_gemm_m64" => Some(whole(64)),
                "plan" => Some(plan(rows, nout, cin * taps, 4 * gpu_cores(), true)),
                _ => None,
            };
            let pp = match pl {
                Some(pl) if pl.tm == 128 => pipe("un_gemm"),
                Some(_) => pipe("un_gemm_m64"),
                None => pipe(name),
            };
            let red = pipe("un_gemm_red");
            let mut p = p;
            if let Some(pl) = pl {
                (p.full, p.rs, p.kper, p.tm, p.nrem) = (pl.full as u32, pl.rs as u32, pl.kper as u32, pl.tm as u32, pl.nrem as u32);
            }
            let tn = if name.contains("_n128") {
                128
            } else if name.contains("_n32") {
                32
            } else {
                64
            };
            let tmr = if name.contains("_m64") {
                64
            } else if name.contains("_m32") {
                32
            } else {
                128
            };
            let reps = 20;
            let mut best = f64::MAX;
            for _ in 0..3 {
                let t = std::time::Instant::now();
                let parts = [std::ops::Range { start: 0, end: reps }];
                let (ok, _) = run_concurrent(be, &f, &parts, |e, r| {
                    for _ in r {
                        e.pipe(pp);
                        e.buf(bw.id, 0, 0);
                        e.buf(bx.id, 0, 1);
                        e.buf(by.id, 0, 2);
                        e.buf(bz.id, 0, 3);
                        e.bytes(&p, 4);
                        e.buf(by.id, 0, 5);
                        e.buf(bz.id, 0, 6);
                        e.buf(bp.id, 0, 7);
                        match pl {
                            Some(pl) => e.dispatch((pl.threadgroups(), 1), 128),
                            None => e.dispatch((nout.div_ceil(tn), rows.div_ceil(tmr)), 128),
                        }
                        e.barrier();
                        if let Some(pl) = pl.filter(|pl| pl.nrem > 0) {
                            e.pipe(red);
                            e.buf(by.id, 0, 2);
                            e.buf(bz.id, 0, 3);
                            e.bytes(&p, 4);
                            e.buf(by.id, 0, 5);
                            e.buf(bz.id, 0, 6);
                            e.buf(bp.id, 0, 7);
                            e.dispatch(((pl.nrem * pl.tm * 64).div_ceil(256), 1), 256);
                            e.barrier();
                        }
                    }
                });
                assert!(ok);
                best = best.min(t.elapsed().as_secs_f64() / reps as f64);
            }
            eprintln!(
                "{rows:6} x {cin:5} x{taps} -> {nout:5}: {name:14} {:7.3} ms {:6.1} TFLOP/s {}",
                best * 1e3,
                flop / best / 1e12,
                pl.map_or(String::new(), |p| format!("{p:?}"))
            );
        }
    }
}

/// Softmax attention of `q` (rows x dh) over `k` / `v` (keys x dh) in f64.
fn attn_ref(q: &[f64], k: &[f64], v: &[f64], dh: usize, scale: f64) -> Vec<f64> {
    let tk = k.len() / dh;
    let mut o = vec![0f64; q.len()];
    for (qr, orow) in q.chunks(dh).zip(o.chunks_mut(dh)) {
        let s: Vec<f64> = k
            .chunks(dh)
            .map(|kr| qr.iter().zip(kr).map(|(a, b)| a * b).sum::<f64>() * scale)
            .collect();
        let m = s.iter().cloned().fold(f64::MIN, f64::max);
        let e: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
        let l: f64 = e.iter().sum();
        for j in 0..tk {
            for c in 0..dh {
                orow[c] += e[j] / l * v[j * dh + c];
            }
        }
    }
    o
}

/// Relative L2 distance of `y` from `r`.
fn rel64(y: &[f64], r: &[f64]) -> f64 {
    let n: f64 = y.iter().zip(r).map(|(a, b)| (a - b).powi(2)).sum();
    let d: f64 = r.iter().map(|b| b * b).sum();
    (n / d.max(1e-300)).sqrt()
}

/// Flash attention kernel timings over UNet shapes, each kernel's output against the first's
/// (`cargo test --release --lib un_flash_bench -- --ignored --nocapture`; UN_FLASH_KERNELS=a,b
/// names kernels of one head width, `_m32` ones (and un_flash48 / 64) taking 128 queries a
/// threadgroup, others 64;
/// UN_FLASH_SHAPES=images:queries:keys:heads:dh,...).
#[test]
#[ignore]
fn un_flash_bench() {
    let Ok(b) = MetalBackend::new() else { return };
    if b.mma().is_none_or(|m| !m.units) {
        return;
    }
    let be: &'static MetalBackend = Box::leak(Box::new(b));
    let f = Fns::new(be);
    let names = std::env::var("UN_FLASH_KERNELS").unwrap_or("un_flash48".into());
    let shapes: Vec<[usize; 5]> = match std::env::var("UN_FLASH_SHAPES") {
        Ok(v) => v
            .split(',')
            .map(|s| {
                let n: Vec<usize> = s.split(':').map(|x| x.parse().unwrap()).collect();
                [n[0], n[1], n[2], n[3], n[4]]
            })
            .collect(),
        Err(_) => vec![[2, 4096, 4096, 8, 40], [2, 4096, 77, 8, 40]],
    };
    let mut seed = 7u64;
    for &[nb, tq, tk, heads, dh] in &shapes {
        let d = head_pad(dh).unwrap();
        let ks = tk.next_multiple_of(64);
        let qrows = nb * tq + 512;
        let krows = nb * ks + 512;
        let ld = heads * d;
        let mut bq = DBuf::empty();
        let mut bk = DBuf::empty();
        let mut bv = DBuf::empty();
        let mut bo = DBuf::empty();
        assert!(ensure(be, &mut bq, qrows * ld * 2));
        assert!(ensure(be, &mut bk, krows * ld * 2));
        assert!(ensure(be, &mut bv, krows * ld * 2));
        assert!(ensure(be, &mut bo, nb * tq * heads * dh * 2));
        for (buf, rows, amp) in [(&bq, qrows, 2.0f32), (&bk, krows, 2.0), (&bv, krows, 1.0)] {
            let p = f.contents(buf.id) as *mut u16;
            for r in 0..rows {
                for c in 0..ld {
                    let v = if c % d < dh { rnd(&mut seed) * amp } else { 0.0 };
                    // SAFETY: rows x ld halves were allocated above.
                    unsafe { *p.add(r * ld + c) = half::f16::from_f32(v).to_bits() };
                }
            }
        }
        let p = Ufa {
            tq: tq as u32,
            tk: tk as u32,
            qs: tq as u32,
            ks: ks as u32,
            heads: heads as u32,
            dh: dh as u32,
            o_ld: (heads * dh) as u32,
            scale: 1.0 / (dh as f32).sqrt(),
        };
        let n_out = nb * tq * heads * dh;
        let mut first: Option<Vec<f32>> = None;
        // The first 64 queries of image 0's last head in f64.
        let hh = heads - 1;
        let get = |b: &DBuf, r: usize, c: usize| -> f64 {
            let p = f.contents(b.id) as *const u16;
            // SAFETY: r < rows, c < dh <= d.
            half::f16::from_bits(unsafe { *p.add(r * ld + hh * d + c) }).to_f64()
        };
        let rq = 64.min(tq);
        let qr: Vec<f64> = (0..rq * dh).map(|i| get(&bq, i / dh, i % dh)).collect();
        let kr: Vec<f64> = (0..tk * dh).map(|i| get(&bk, i / dh, i % dh)).collect();
        let vr: Vec<f64> = (0..tk * dh).map(|i| get(&bv, i / dh, i % dh)).collect();
        let want = attn_ref(&qr, &kr, &vr, dh, 1.0 / (dh as f64).sqrt());
        for name in names.split(',') {
            let pp = be.named_pipeline(name).unwrap();
            let qtg = if name.contains("_m32") || name == "un_flash48" || name == "un_flash64" {
                128
            } else {
                64
            };
            let reps = 20;
            let mut best = f64::MAX;
            for _ in 0..3 {
                let t = std::time::Instant::now();
                let parts = [std::ops::Range { start: 0, end: reps }];
                let (ok, _) = run_concurrent(be, &f, &parts, |e, r| {
                    for _ in r {
                        e.pipe(pp);
                        e.buf(bq.id, 0, 0);
                        e.buf(bk.id, 0, 1);
                        e.buf(bv.id, 0, 2);
                        e.buf(bo.id, 0, 3);
                        e.bytes(&p, 4);
                        e.dispatch3((tq.div_ceil(qtg), heads * (d / value_chunk(d)), nb), 128);
                        e.barrier();
                    }
                });
                assert!(ok);
                best = best.min(t.elapsed().as_secs_f64() / reps as f64);
            }
            let op = f.contents(bo.id) as *const u16;
            // SAFETY: the output holds n_out halves.
            let out: Vec<f32> = (0..n_out).map(|i| half::f16::from_bits(unsafe { *op.add(i) }).to_f32()).collect();
            let err = first.as_ref().map_or(0.0, |r| rel(&out, r));
            let got: Vec<f64> = (0..rq * dh).map(|i| out[(i / dh) * heads * dh + hh * dh + i % dh] as f64).collect();
            let eref = rel64(&got, &want);
            eprintln!(
                "{nb} x {tq:5} q x {tk:5} k x {heads} heads of {dh:3}: {name:14} {:7.3} ms {:6.1} G scores/s, rel {err:.2e}, f64 {eref:.2e}",
                best * 1e3,
                (nb * tq * tk * heads) as f64 / best / 1e9
            );
            if first.is_none() {
                first = Some(out);
            }
        }
    }
}

/// Timings of the DiT attention kernels over 128-wide heads, each kernel's output against the
/// first's (`cargo test --release --lib fa128_bench -- --ignored --nocapture`;
/// FA_KERNELS=flash_attn128_mma,dit_flash128_hilo (`hilo` ones read the low halves too),
/// FA_SHAPES=tokens:heads,...).
#[test]
#[ignore]
fn fa128_bench() {
    let Ok(b) = MetalBackend::new() else { return };
    if b.mma().is_none_or(|m| !m.units) {
        return;
    }
    let be: &'static MetalBackend = Box::leak(Box::new(b));
    let f = Fns::new(be);
    let names = std::env::var("FA_KERNELS").unwrap_or("flash_attn128_mma".into());
    let shapes: Vec<[usize; 2]> = match std::env::var("FA_SHAPES") {
        Ok(v) => v
            .split(',')
            .map(|s| {
                let n: Vec<usize> = s.split(':').map(|x| x.parse().unwrap()).collect();
                [n[0], n[1]]
            })
            .collect(),
        Err(_) => vec![[4608, 24]],
    };
    let mut seed = 11u64;
    for &[n, heads] in &shapes {
        let d = heads * 128;
        let rows = n.next_multiple_of(64);
        let bufs: Vec<DBuf> = (0..5)
            .map(|i| {
                let mut b = DBuf::empty();
                assert!(ensure(be, &mut b, rows * d * 2));
                let p = f.contents(b.id) as *mut u16;
                for j in 0..rows * d {
                    let v = if j < n * d {
                        rnd(&mut seed) * if i >= 3 { 1e-3 } else { 3.0 }
                    } else {
                        0.0
                    };
                    // SAFETY: rows x d halves were allocated above.
                    unsafe { *p.add(j) = half::f16::from_f32(v).to_bits() };
                }
                b
            })
            .collect();
        let mut bo = DBuf::empty();
        assert!(ensure(be, &mut bo, n * d * 4));
        let p = U4([n as u32, n as u32, d as u32, (1.0f32 / (128f32).sqrt()).to_bits()]);
        let mut first: Option<Vec<f32>> = None;
        // The first 32 queries of head 0 in f64, from the high halves and from hi + lo.
        let get = |b: &DBuf, i: usize| -> f64 {
            let p = f.contents(b.id) as *const u16;
            // SAFETY: i < rows x d.
            half::f16::from_bits(unsafe { *p.add(i) }).to_f64()
        };
        let at = |i: usize| (i / 128) * d + i % 128;
        let refs: Vec<Vec<f64>> = [false, true]
            .iter()
            .map(|&lo| {
                let qv: Vec<f64> = (0..32 * 128)
                    .map(|i| get(&bufs[0], at(i)) + if lo { get(&bufs[3], at(i)) } else { 0.0 })
                    .collect();
                let kv: Vec<f64> = (0..n * 128)
                    .map(|i| get(&bufs[1], at(i)) + if lo { get(&bufs[4], at(i)) } else { 0.0 })
                    .collect();
                let vv: Vec<f64> = (0..n * 128).map(|i| get(&bufs[2], at(i))).collect();
                attn_ref(&qv, &kv, &vv, 128, 1.0 / (128f64).sqrt())
            })
            .collect();
        for name in names.split(',') {
            let pp = be.named_pipeline(name).unwrap();
            let reps = 10;
            let mut best = f64::MAX;
            for _ in 0..3 {
                let t = std::time::Instant::now();
                let parts = [std::ops::Range { start: 0, end: reps }];
                let (ok, _) = run_concurrent(be, &f, &parts, |e, r| {
                    for _ in r {
                        e.pipe(pp);
                        for i in 0..3 {
                            e.buf(bufs[i].id, 0, i);
                        }
                        e.buf(bo.id, 0, 3);
                        e.bytes(&p, 4);
                        if name.contains("hilo") {
                            e.buf(bufs[3].id, 0, 5);
                            e.buf(bufs[4].id, 0, 6);
                        }
                        e.dispatch((n.div_ceil(64), heads), 128);
                        e.barrier();
                    }
                });
                assert!(ok);
                best = best.min(t.elapsed().as_secs_f64() / reps as f64);
            }
            let op = f.contents(bo.id) as *const f32;
            // SAFETY: the output holds n x d floats.
            let out: Vec<f32> = unsafe { std::slice::from_raw_parts(op, n * d) }.to_vec();
            let err = first.as_ref().map_or(0.0, |r| rel(&out, r));
            let got: Vec<f64> = (0..32 * 128).map(|i| out[at(i)] as f64).collect();
            let eref = rel64(&got, &refs[name.contains("hilo") as usize]);
            eprintln!(
                "{n:5} tokens x {heads} heads of 128: {name:20} {:7.3} ms {:6.1} G scores/s, rel {err:.2e}, f64 {eref:.2e}",
                best * 1e3,
                (n * n * heads) as f64 / best / 1e9
            );
            if first.is_none() {
                first = Some(out);
            }
        }
    }
}
