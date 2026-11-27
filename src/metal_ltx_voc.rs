//! Whole vocoder generator passes on Metal ([`LtxVocSession`], `crate::ltx::audio`): the signals
//! stay in f32 device buffers as rows of channels after a guard of zero steps; every 1-D
//! convolution is `un_gemm`'s shifted GEMMs over half rows (its dilated taps, at most nine per
//! GEMM), the transposed convolutions one GEMM per output phase (strided stores), the
//! anti-aliased SnakeBeta activations `lb_act` (`metal_ltx.metal`), which write the next
//! convolution's half input directly. One command buffer per generator.

use super::super::decode::{FastEnc, Fns, run_concurrent};
use super::super::*;
use super::{DBuf, grow, release_all};
use crate::gpu::dit::{LtxVocAct, LtxVocGen, LtxVocLin, LtxVocSession};

struct Pipes {
    gemm: Id,
    act: Id,
    mean: Id,
    half: Id,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum B {
    /// The stage's input, the transposed convolution's output, a residual branch's output, the
    /// three AMP blocks' signals, a GEMM's half input.
    X = 0,
    Y,
    T,
    Z0,
    Z1,
    Z2,
    H,
    /// Weights, biases, activation constants (uploaded per pass).
    W,
    C,
}
const NBUF: usize = 9;

pub(crate) struct MetalLtxVoc {
    be: &'static MetalBackend,
    f: Fns,
    p: Pipes,
    bufs: Vec<DBuf>,
}

// SAFETY: as MetalDit.
unsafe impl Send for MetalLtxVoc {}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Geo {
    h: u32,
    w: u32,
    g: u32,
    p: u32,
}

/// `metal_unet.metal`'s `UG`.
#[repr(C)]
struct Ug {
    n_out: u32,
    ldw: u32,
    cin: u32,
    taps: u32,
    off: [i32; 9],
    a_rows: u32,
    r0: u32,
    r_end: u32,
    omode: u32,
    ldy: u32,
    has_bias: u32,
    bstride: u32,
    add: u32,
    nimg: u32,
    g: Geo,
    scaled: u32,
    full: u32,
    rs: u32,
    kper: u32,
    nct: u32,
    tm: u32,
    nrem: u32,
    _p: u32,
}

#[repr(C)]
struct Lba {
    t: u32,
    g: u32,
    c: u32,
    k: u32,
    rows: u32,
    _p: [u32; 3],
}

#[repr(C)]
struct Lbm {
    n: u32,
    g: u32,
    t: u32,
    scale: f32,
}

#[repr(C)]
struct Lbh {
    n: u32,
    g: u32,
    t: u32,
    rows: u32,
}

/// Rows per GEMM tile, and the guard of zero steps before the generator's input.
const TM: usize = 128;
const G0: usize = 32;

impl MetalBackend {
    /// [`crate::gpu::Backend::ltx_voc_session`].
    pub(crate) fn ltx_voc_session_impl(&'static self) -> Option<Box<dyn LtxVocSession>> {
        self.mma().filter(|m| m.units)?;
        let pipe = |n: &str| self.named_pipeline(n);
        let p = Pipes {
            gemm: pipe("un_gemm")?,
            act: pipe("lb_act")?,
            mean: pipe("lb_mean3")?,
            half: pipe("lb_half")?,
        };
        Some(Box::new(MetalLtxVoc {
            be: self,
            f: Fns::new(self),
            p,
            bufs: (0..NBUF)
                .map(|_| DBuf {
                    id: std::ptr::null_mut(),
                    cap: 0,
                })
                .collect(),
        }))
    }
}

impl Drop for MetalLtxVoc {
    fn drop(&mut self) {
        release_all(self.be, &self.bufs);
    }
}

/// A signal's layout: `t` steps of `c` channels after a guard of `g` rows.
#[derive(Clone, Copy)]
struct Sig {
    g: usize,
    t: usize,
}

impl Sig {
    fn rows(&self) -> usize {
        2 * self.g + self.t + 2 * TM
    }
}

/// Byte offsets of a layer's weights / bias, or an activation's constants, in `W` / `C`.
#[derive(Clone, Copy)]
struct LinAt {
    w: usize,
    b: usize,
}

/// Every weight and constant of a generator, gathered for one upload.
#[derive(Default)]
struct Pack {
    w: Vec<u8>,
    c: Vec<f32>,
}

impl Pack {
    fn lin(&mut self, l: &LtxVocLin) -> LinAt {
        let w = self.w.len();
        // SAFETY: plain bytes of the u16 weights.
        self.w
            .extend_from_slice(unsafe { std::slice::from_raw_parts(l.w.as_ptr() as *const u8, l.w.len() * 2) });
        self.w.resize(self.w.len().next_multiple_of(256), 0);
        LinAt { w, b: self.put(l.bias) }
    }

    fn put(&mut self, v: &[f32]) -> usize {
        let o = self.c.len() * 4;
        self.c.extend_from_slice(v);
        self.c.resize(self.c.len().next_multiple_of(64), 0.0);
        o
    }

    /// An activation's (alpha / inverse beta, filters) offsets.
    fn act(&mut self, a: &LtxVocAct) -> (usize, usize) {
        (self.put(a.ab), self.put(a.filt))
    }
}

fn lin_ok(l: &LtxVocLin) -> bool {
    let rows = l.cout.next_multiple_of(64);
    !l.offs.is_empty() && l.cin.is_multiple_of(32) && l.w.len() == rows * l.offs.len() * l.cin && l.bias.len() == rows
}

impl MetalLtxVoc {
    fn ensure(&mut self, b: B, bytes: usize) -> bool {
        grow(self.be, &mut self.bufs[b as usize], bytes)
    }

    fn host<T>(&self, b: B) -> *mut T {
        self.f.contents(self.bufs[b as usize].id) as *mut T
    }
}

impl LtxVocSession for MetalLtxVoc {
    fn generate(&mut self, gn: &LtxVocGen, x: &[f32], t: usize, out: &mut Vec<f32>) -> bool {
        let c0 = gn.pre.cin;
        if t == 0 || x.len() != t * c0 || !lin_ok(&gn.pre) || !lin_ok(&gn.post) {
            return false;
        }
        // Shapes, buffer sizes and the packed constants.
        let mut pack = Pack::default();
        let pre = pack.lin(&gn.pre);
        let (mut s, mut c) = (Sig { g: G0, t }, gn.pre.cout);
        let mut maxf = s.rows() * c0.max(c);
        let mut stages = Vec::new();
        for st in &gn.stages {
            let stride = st.up.len();
            if stride == 0 || st.amps.len() != 3 || st.up.iter().any(|l| !lin_ok(l) || l.cin != c) {
                return false;
            }
            maxf = maxf.max(s.rows() * c);
            let ups: Vec<LinAt> = st.up.iter().map(|l| pack.lin(l)).collect();
            let co = st.up[0].cout;
            s = Sig {
                g: s.g * stride,
                t: s.t * stride,
            };
            c = co;
            maxf = maxf.max(s.rows() * c);
            let mut amps = Vec::new();
            for amp in &st.amps {
                let mut rs = Vec::new();
                for r in amp {
                    if !lin_ok(&r.conv1) || !lin_ok(&r.conv2) || r.conv1.cin != c || r.conv2.cout != c || r.conv1.cout != r.conv2.cin {
                        return false;
                    }
                    let max_off = r
                        .conv1
                        .offs
                        .iter()
                        .chain(r.conv2.offs)
                        .map(|o| o.unsigned_abs() as usize)
                        .max()
                        .unwrap_or(0);
                    if max_off > s.g {
                        return false;
                    }
                    rs.push((pack.act(&r.act1), pack.lin(&r.conv1), pack.act(&r.act2), pack.lin(&r.conv2)));
                }
                amps.push(rs);
            }
            stages.push((ups, amps, stride, co));
        }
        if gn.post.cin != c {
            return false;
        }
        let act_post = pack.act(&gn.act_post);
        let post = pack.lin(&gn.post);
        let steps = s.t;
        let ok = [B::X, B::Y, B::T, B::Z0, B::Z1, B::Z2].iter().all(|&b| self.ensure(b, maxf * 4))
            && self.ensure(B::H, maxf * 2)
            && self.ensure(B::W, pack.w.len())
            && self.ensure(B::C, pack.c.len() * 4);
        if !ok {
            return false;
        }
        // SAFETY: the buffers hold these bytes; the device is idle.
        unsafe {
            std::ptr::copy_nonoverlapping(pack.w.as_ptr(), self.host::<u8>(B::W), pack.w.len());
            std::ptr::copy_nonoverlapping(pack.c.as_ptr(), self.host::<f32>(B::C), pack.c.len());
            let xp = self.host::<f32>(B::X);
            std::ptr::copy_nonoverlapping(x.as_ptr(), xp.add(G0 * c0), t * c0);
        }
        let ids: [Id; NBUF] = std::array::from_fn(|i| self.bufs[i].id);
        let id = |b: B| ids[b as usize];
        let p = &self.p;
        // `l` over the half rows of H (signal `s`) into y (+ y_off bytes, rows `ldy` floats
        // apart), plus `add` (y's layout) when given.
        let gemm = |e: &FastEnc, l: &LtxVocLin, at: LinAt, s: Sig, y: (B, usize), ldy: usize, add: Option<B>| {
            let (r0, r_end) = (s.g, s.g + s.t);
            let nct = l.cout.div_ceil(64);
            let tiles = (r_end - r0).div_ceil(TM) * nct;
            let taps = l.offs.len();
            for (ci, chunk) in l.offs.chunks(9).enumerate() {
                let mut off = [0i32; 9];
                off[..chunk.len()].copy_from_slice(chunk);
                let addb = if ci == 0 { add } else { Some(y.0) };
                e.pipe(p.gemm);
                e.buf(id(B::W), at.w + ci * 9 * l.cin * 2, 0);
                e.buf(id(B::H), 0, 1);
                e.buf(id(y.0), y.1, 2);
                e.buf(id(B::C), at.b, 3);
                e.bytes(
                    &Ug {
                        n_out: l.cout as u32,
                        ldw: (taps * l.cin) as u32,
                        cin: l.cin as u32,
                        taps: chunk.len() as u32,
                        off,
                        a_rows: s.rows() as u32,
                        r0: r0 as u32,
                        r_end: r_end as u32,
                        omode: 0,
                        ldy: ldy as u32,
                        has_bias: (ci == 0) as u32,
                        bstride: 0,
                        add: addb.is_some() as u32,
                        nimg: 1,
                        g: Geo::default(),
                        scaled: 0,
                        full: tiles as u32,
                        rs: 1,
                        kper: (chunk.len() * l.cin) as u32,
                        nct: nct as u32,
                        tm: TM as u32,
                        nrem: 0,
                        _p: 0,
                    },
                    4,
                );
                e.buf(id(addb.unwrap_or(y.0)), y.1, 5);
                e.buf(id(B::C), 0, 6);
                e.buf(id(B::T), 0, 7);
                e.dispatch((tiles, 1), 128);
                e.barrier();
            }
        };
        // `x` (c channels, signal s) through an activation into H.
        let act = |e: &FastEnc, x: B, s: Sig, c: usize, (ab, filt): (usize, usize), k: usize| {
            e.pipe(p.act);
            e.buf(id(x), 0, 0);
            e.buf(id(B::H), 0, 1);
            e.buf(id(B::C), ab, 2);
            e.buf(id(B::C), filt, 3);
            e.bytes(
                &Lba {
                    t: s.t as u32,
                    g: s.g as u32,
                    c: c as u32,
                    k: k as u32,
                    rows: s.rows() as u32,
                    _p: [0; 3],
                },
                4,
            );
            e.dispatch(((s.rows() * c).div_ceil(256), 1), 256);
            e.barrier();
        };
        let half = |e: &FastEnc, x: B, s: Sig, c: usize| {
            e.pipe(p.half);
            e.buf(id(x), 0, 0);
            e.buf(id(B::H), 0, 1);
            e.bytes(
                &Lbh {
                    n: c as u32,
                    g: s.g as u32,
                    t: s.t as u32,
                    rows: s.rows() as u32,
                },
                2,
            );
            e.dispatch(((s.rows() * c).div_ceil(256), 1), 256);
            e.barrier();
        };
        let (ok, _) = run_concurrent(self.be, &self.f, std::slice::from_ref(&(0..1)), |e, _| {
            // conv_pre: X (the input) into Y, then the stages from X.
            let s0 = Sig { g: G0, t };
            half(e, B::X, s0, c0);
            gemm(e, &gn.pre, pre, s0, (B::X, 0), gn.pre.cout, None);
            let (mut s, mut c) = (s0, gn.pre.cout);
            for (st, (ups, amps, stride, co)) in gn.stages.iter().zip(&stages) {
                // The transposed convolution: phase ph of input step i is output step i stride +
                // ph (Y's guard is X's times the stride).
                half(e, B::X, s, c);
                for (ph, (l, at)) in st.up.iter().zip(ups).enumerate() {
                    gemm(e, l, *at, s, (B::Y, ph * co * 4), stride * co, None);
                }
                s = Sig {
                    g: s.g * stride,
                    t: s.t * stride,
                };
                c = *co;
                // The AMP blocks, each from Y into its own Z.
                for (j, (amp, rs)) in st.amps.iter().zip(amps).enumerate() {
                    let z = [B::Z0, B::Z1, B::Z2][j];
                    for (i, (r, (a1, l1, a2, l2))) in amp.iter().zip(rs).enumerate() {
                        let src = if i == 0 { B::Y } else { z };
                        act(e, src, s, c, *a1, r.act1.k);
                        gemm(e, &r.conv1, *l1, s, (B::T, 0), r.conv1.cout, None);
                        act(e, B::T, s, r.conv1.cout, *a2, r.act2.k);
                        gemm(e, &r.conv2, *l2, s, (z, 0), c, Some(src));
                    }
                }
                e.pipe(p.mean);
                e.buf(id(B::Z0), 0, 0);
                e.buf(id(B::Z1), 0, 1);
                e.buf(id(B::Z2), 0, 2);
                e.buf(id(B::X), 0, 3);
                e.bytes(
                    &Lbm {
                        n: c as u32,
                        g: s.g as u32,
                        t: s.t as u32,
                        scale: 1.0 / 3.0,
                    },
                    4,
                );
                e.dispatch(((s.t * c).div_ceil(256), 1), 256);
                e.barrier();
            }
            act(e, B::X, s, c, act_post, gn.act_post.k);
            gemm(e, &gn.post, post, s, (B::T, 0), gn.post.cout, None);
        });
        if !ok {
            return false;
        }
        let n = gn.post.cout;
        out.resize(steps * n, 0.0);
        // SAFETY: T holds the rows of the last signal (its guard s.g, `steps` steps).
        unsafe {
            std::ptr::copy_nonoverlapping(self.host::<f32>(B::T).add(s.g * n), out.as_mut_ptr(), steps * n);
        }
        true
    }
}
