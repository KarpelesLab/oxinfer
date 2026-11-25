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
    fn attn(ws: &[W], heads: usize) -> crate::gpu::unet::Attn<'_> {
        crate::gpu::unet::Attn {
            q: Lin { bias: None, ..ws[0].lin() },
            k: Lin { bias: None, ..ws[1].lin() },
            v: Lin { bias: None, ..ws[2].lin() },
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
                        a1: attn(&a1, 2),
                        n2: lns[1].norm(0),
                        a2: attn(&a2, 2),
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
    // Twice: the second pass reuses the session's buffers.
    for _ in 0..2 {
        assert!(sess.forward(&g, &items, &mut out), "device pass failed");
    }
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
        let o = attention(&a1[0].apply(&hn, false), &a1[1].apply(&hn, false), &a1[2].apply(&hn, false), 2, ct);
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
