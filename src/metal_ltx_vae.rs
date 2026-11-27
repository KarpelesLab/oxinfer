//! Whole LTX-2.5 video VAE decodes on Metal ([`LtxVaeSession`], `crate::ltx::vae`): the
//! activations stay in f32 device buffers in `metal_unet.metal`'s padded layout (the frames as
//! its images, with images for the repeated first and last frames around them); every 3 x 3 x 3
//! convolution is three of `un_gemm`'s shifted GEMMs on the matrix units (one per time tap, nine
//! spatial taps each, bf16 weights, half inputs: `PixelNorm` and SiLU'd by `lv_prep`, or scaled
//! into half's range), accumulating in f32; the depth-to-space steps and the final patch layout
//! are `lv_d2s` / `lv_out` (`metal_ltx.metal`). One command buffer per op; the frames come back
//! once. The encoder runs the same way: its space-to-depth steps (the first frame repeated when
//! time halves: `lv_prep`'s frame shift, the input's channel-group means added) are `lv_s2d`.

use super::super::decode::{Fns, run_concurrent};
use super::super::*;
use super::{DBuf, grow, release_all};
use crate::gpu::dit::{LtxVaeConv, LtxVaeDecoder, LtxVaeEncOp, LtxVaeEncoder, LtxVaeOp, LtxVaeSession};

struct Pipes {
    gemm: Id,
    prep: Id,
    amax: Id,
    d2s: Id,
    out: Id,
    s2d: Id,
}

#[derive(Clone, Copy)]
enum B {
    /// Activations (ping-pong), a convolution's output, its half input.
    X = 0,
    Y,
    T,
    H,
    /// Every convolution's weights and biases (uploaded once per decoder).
    W,
    Bias,
    Amax,
    Out,
}
const NBUF: usize = 8;

pub(crate) struct MetalLtxVae {
    be: &'static MetalBackend,
    f: Fns,
    p: Pipes,
    bufs: Vec<DBuf>,
    /// The decoder whose weights `W` / `Bias` hold (its input convolution's weight address).
    weights_of: usize,
}

// SAFETY: as MetalDit.
unsafe impl Send for MetalLtxVae {}

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
struct Lva {
    g: Geo,
    c: u32,
    lo: u32,
    hi: u32,
    rows: u32,
}

#[repr(C)]
struct Lvp {
    g: Geo,
    c: u32,
    nimg: u32,
    lo: u32,
    hi: u32,
    mode: u32,
    rows: u32,
    eps: f32,
    shift: u32,
    _p: [u32; 3],
}

#[repr(C)]
struct Lvs {
    s: Geo,
    d: Geo,
    co: u32,
    xc: u32,
    p1: u32,
    p2: u32,
    p3: u32,
    group: u32,
    dup: u32,
    lo: u32,
    t: u32,
    _p: u32,
}

#[repr(C)]
struct Lvd {
    s: Geo,
    d: Geo,
    c: u32,
    p1: u32,
    p2: u32,
    p3: u32,
    drop: u32,
    lo: u32,
    t: u32,
    _p: u32,
}

#[repr(C)]
struct Lvo {
    s: Geo,
    c: u32,
    pq: u32,
    oc: u32,
    lo: u32,
    t: u32,
    _p: [u32; 3],
}

/// Images before the clip's frames (the causal time padding repeats the first frame twice) and
/// after them.
const LO: usize = 2;
const HI_PAD: usize = 1;
/// Rows per GEMM tile.
const TM: usize = 128;

/// The padded layout of `t` frames of `h x w`.
#[derive(Clone, Copy)]
struct Lay {
    t: usize,
    h: usize,
    w: usize,
}

impl Lay {
    fn g(&self) -> usize {
        self.w + 3
    }
    fn p(&self) -> usize {
        (self.h + 2) * (self.w + 2)
    }
    fn nimg(&self) -> usize {
        LO + self.t + HI_PAD
    }
    /// Rows to allocate: the guards, the images, and room for the last GEMM tile's taps.
    fn rows(&self) -> usize {
        2 * self.g() + self.nimg() * self.p() + 2 * TM
    }
    fn geo(&self) -> Geo {
        Geo {
            h: self.h as u32,
            w: self.w as u32,
            g: self.g() as u32,
            p: self.p() as u32,
        }
    }
    /// The rows of the clip's frames (what a convolution computes).
    fn span(&self) -> (usize, usize) {
        (self.g() + LO * self.p(), self.g() + (LO + self.t) * self.p())
    }
}

impl MetalBackend {
    /// [`crate::gpu::Backend::ltx_vae_session`].
    pub(crate) fn ltx_vae_session_impl(&'static self) -> Option<Box<dyn LtxVaeSession>> {
        self.mma().filter(|m| m.units)?;
        let pipe = |n: &str| self.named_pipeline(n);
        let p = Pipes {
            gemm: pipe("un_gemm")?,
            prep: pipe("lv_prep")?,
            amax: pipe("lv_amax")?,
            d2s: pipe("lv_d2s")?,
            out: pipe("lv_out")?,
            s2d: pipe("lv_s2d")?,
        };
        Some(Box::new(MetalLtxVae {
            be: self,
            f: Fns::new(self),
            p,
            bufs: (0..NBUF)
                .map(|_| DBuf {
                    id: std::ptr::null_mut(),
                    cap: 0,
                })
                .collect(),
            weights_of: 0,
        }))
    }
}

impl Drop for MetalLtxVae {
    fn drop(&mut self) {
        release_all(self.be, &self.bufs);
    }
}

/// Where a convolution's weights and bias are in `W` / `Bias` (bytes).
#[derive(Clone, Copy)]
struct ConvAt {
    w: usize,
    b: usize,
    cin: usize,
    cout: usize,
}

impl MetalLtxVae {
    fn ensure(&mut self, b: B, bytes: usize) -> bool {
        grow(self.be, &mut self.bufs[b as usize], bytes)
    }

    fn id(&self, b: B) -> Id {
        self.bufs[b as usize].id
    }

    fn host<T>(&self, b: B) -> *mut T {
        self.f.contents(self.id(b)) as *mut T
    }

    /// Runs `encode` in one command buffer and waits for it.
    fn run(&self, encode: impl Fn(&super::super::decode::FastEnc)) -> bool {
        run_concurrent(self.be, &self.f, std::slice::from_ref(&(0..1)), |e, _| encode(e)).0
    }
}

/// The convolutions of `m` in upload order.
fn convs<'m, 'a>(m: &'m LtxVaeDecoder<'a>) -> Vec<&'m LtxVaeConv<'a>> {
    let mut v = vec![&m.conv_in];
    for op in &m.ops {
        match op {
            LtxVaeOp::Res(a, b) => {
                v.push(a);
                v.push(b);
            }
            LtxVaeOp::D2s(c, _) => v.push(c),
        }
    }
    v.push(&m.conv_out);
    v
}

impl LtxVaeSession for MetalLtxVae {
    fn decode(
        &mut self,
        m: &LtxVaeDecoder,
        lat: &[f32],
        t: usize,
        h: usize,
        w: usize,
        out: &mut Vec<f32>,
        progress: &mut dyn FnMut(usize, usize),
    ) -> bool {
        let all = convs(m);
        let c0 = m.conv_in.cin;
        if t == 0 || h == 0 || w == 0 || lat.len() != t * h * w * c0 {
            return false;
        }
        for c in &all {
            let rows = c.cout.next_multiple_of(64);
            if !c.cin.is_multiple_of(32) || c.w.len() != rows * 27 * c.cin || c.bias.len() != rows {
                return false;
            }
        }
        // The shapes through the decoder: the largest activation, convolution output and half
        // input.
        let (mut lay, mut ch) = (Lay { t, h, w }, m.conv_in.cout);
        let (mut act, mut conv_out, mut half) = (lay.rows() * c0 * 4, lay.rows() * ch * 4, lay.rows() * c0 * 2);
        act = act.max(lay.rows() * ch * 4);
        for op in &m.ops {
            match op {
                LtxVaeOp::Res(a, b) => {
                    half = half.max(lay.rows() * a.cin.max(b.cin) * 2);
                    conv_out = conv_out.max(lay.rows() * a.cout * 4);
                    act = act.max(lay.rows() * b.cout * 4);
                    ch = b.cout;
                }
                LtxVaeOp::D2s(c, st) => {
                    half = half.max(lay.rows() * c.cin * 2);
                    conv_out = conv_out.max(lay.rows() * c.cout * 4);
                    let n = st[0] * st[1] * st[2];
                    if !c.cout.is_multiple_of(n) {
                        return false;
                    }
                    ch = c.cout / n;
                    lay = Lay {
                        t: lay.t * st[0] - (st[0] == 2) as usize,
                        h: lay.h * st[1],
                        w: lay.w * st[2],
                    };
                    act = act.max(lay.rows() * ch * 4);
                }
            }
        }
        let pq = m.patch;
        let oc = m.conv_out.cout / (pq * pq);
        if ch != m.conv_out.cin || oc * pq * pq != m.conv_out.cout {
            return false;
        }
        half = half.max(lay.rows() * ch * 2);
        conv_out = conv_out.max(lay.rows() * m.conv_out.cout * 4);
        let frames = lay.t * lay.h * pq * lay.w * pq * oc;
        let ok = self.ensure(B::X, act)
            && self.ensure(B::Y, act)
            && self.ensure(B::T, conv_out)
            && self.ensure(B::H, half)
            && self.ensure(B::Amax, 256)
            && self.ensure(B::Out, frames * 4);
        if !ok {
            return false;
        }
        // Weights and biases, once per decoder.
        let key = m.conv_in.w.as_ptr() as usize;
        let mut at = Vec::with_capacity(all.len());
        let (mut wb, mut bb) = (0usize, 0usize);
        for c in &all {
            at.push(ConvAt {
                w: wb,
                b: bb,
                cin: c.cin,
                cout: c.cout,
            });
            wb += (c.w.len() * 2).next_multiple_of(256);
            bb += (c.bias.len() * 4).next_multiple_of(256);
        }
        if self.weights_of != key {
            if !self.ensure(B::W, wb) || !self.ensure(B::Bias, bb) {
                return false;
            }
            let (wp, bp) = (self.host::<u8>(B::W), self.host::<u8>(B::Bias));
            for (c, a) in all.iter().zip(&at) {
                // SAFETY: the buffers hold wb / bb bytes, every convolution inside them; the
                // device is idle.
                unsafe {
                    std::ptr::copy_nonoverlapping(c.w.as_ptr() as *const u8, wp.add(a.w), c.w.len() * 2);
                    super::super::par_copy(bp.add(a.b) as *mut f32, c.bias.as_ptr(), c.bias.len());
                }
            }
            self.weights_of = key;
        }
        // The latents into X's frames (borders and guards are never read).
        let lay0 = Lay { t, h, w };
        let xp = self.host::<f32>(B::X);
        for f in 0..t {
            for y in 0..h {
                let r = lay0.g() + (LO + f) * lay0.p() + (y + 1) * (w + 2) + 1;
                // SAFETY: X holds lay0.rows() rows of c0 floats.
                unsafe {
                    std::ptr::copy_nonoverlapping(lat.as_ptr().add((f * h + y) * w * c0), xp.add(r * c0), w * c0);
                }
            }
        }
        let ids: [Id; NBUF] = std::array::from_fn(|i| self.bufs[i].id);
        let id = |b: B| ids[b as usize];
        let p = &self.p;
        let shift = if m.causal { 2 } else { 1 };
        // The half rows of `src` (c channels, layout l) into H.
        let prep = |e: &super::super::decode::FastEnc, src: B, l: Lay, c: usize, mode: u32| {
            if mode == 2 {
                e.pipe(p.amax);
                e.buf(id(src), 0, 0);
                e.buf(id(B::Amax), 0, 1);
                e.bytes(
                    &Lva {
                        g: l.geo(),
                        c: c as u32,
                        lo: LO as u32,
                        hi: (LO + l.t) as u32,
                        rows: l.rows() as u32,
                    },
                    2,
                );
                e.dispatch(((l.rows() * c / 4).div_ceil(256), 1), 256);
                e.barrier();
            }
            e.pipe(p.prep);
            e.buf(id(src), 0, 0);
            e.buf(id(B::H), 0, 1);
            e.buf(id(B::Amax), 0, 2);
            e.bytes(
                &Lvp {
                    g: l.geo(),
                    c: c as u32,
                    nimg: l.nimg() as u32,
                    lo: LO as u32,
                    hi: (LO + l.t) as u32,
                    mode,
                    rows: l.rows() as u32,
                    eps: 1e-8,
                    shift: 0,
                    _p: [0; 3],
                },
                3,
            );
            e.dispatch((l.rows().div_ceil(8), 1), 256);
            e.barrier();
        };
        // The convolution `a` of H (layout l) into `y`, plus `add` (same layout) when given.
        let conv = |e: &super::super::decode::FastEnc, a: ConvAt, l: Lay, y: B, add: Option<B>, scaled: bool| {
            let (r0, r_end) = l.span();
            let nct = a.cout.div_ceil(64);
            let tiles = (r_end - r0).div_ceil(TM) * nct;
            let (p_, w2) = (l.p() as i32, l.w as i32 + 2);
            for kt in 0..3 {
                let mut off = [0i32; 9];
                for (k, o) in off.iter_mut().enumerate() {
                    let (ky, kx) = ((k / 3) as i32, (k % 3) as i32);
                    *o = (kt as i32 - shift) * p_ + (ky - 1) * w2 + (kx - 1);
                }
                let addb = if kt == 0 { add } else { Some(y) };
                e.pipe(p.gemm);
                e.buf(id(B::W), a.w + kt * 9 * a.cin * 2, 0);
                e.buf(id(B::H), 0, 1);
                e.buf(id(y), 0, 2);
                e.buf(id(B::Bias), a.b, 3);
                e.bytes(
                    &Ug {
                        n_out: a.cout as u32,
                        ldw: (27 * a.cin) as u32,
                        cin: a.cin as u32,
                        taps: 9,
                        off,
                        a_rows: l.rows() as u32,
                        r0: r0 as u32,
                        r_end: r_end as u32,
                        omode: 2,
                        ldy: a.cout as u32,
                        has_bias: (kt == 0) as u32,
                        bstride: 0,
                        add: addb.is_some() as u32,
                        nimg: l.nimg() as u32,
                        g: l.geo(),
                        scaled: scaled as u32,
                        full: tiles as u32,
                        rs: 1,
                        kper: (9 * a.cin) as u32,
                        nct: nct as u32,
                        tm: TM as u32,
                        nrem: 0,
                        _p: 0,
                    },
                    4,
                );
                e.buf(id(addb.unwrap_or(y)), 0, 5);
                e.buf(id(B::Amax), 0, 6);
                e.buf(id(B::T), 0, 7);
                e.dispatch((tiles, 1), 128);
                e.barrier();
            }
        };
        let zero_amax = |s: &Self| {
            // SAFETY: Amax holds 256 bytes; the device is idle.
            unsafe { std::ptr::write_bytes(s.host::<u8>(B::Amax), 0, 256) };
        };
        let total = m.ops.len() + 2;
        let mut lay = lay0;
        // conv_in: the latents scaled into half's range.
        zero_amax(self);
        // The latents in X; conv_in's output (the activations from now on) in Y.
        let ok = self.run(|e| {
            prep(e, B::X, lay, c0, 2);
            conv(e, at[0], lay, B::Y, None, true);
        });
        if !ok {
            return false;
        }
        let (mut x, mut yb) = (B::Y, B::X);
        progress(1, total);
        let mut ci = 1;
        for (oi, op) in m.ops.iter().enumerate() {
            match op {
                LtxVaeOp::Res(a, b) => {
                    let (ca, cb) = (at[ci], at[ci + 1]);
                    ci += 2;
                    let ok = self.run(|e| {
                        prep(e, x, lay, a.cin, 1);
                        conv(e, ca, lay, B::T, None, false);
                        prep(e, B::T, lay, b.cin, 1);
                        conv(e, cb, lay, yb, Some(x), false);
                    });
                    if !ok {
                        return false;
                    }
                    std::mem::swap(&mut x, &mut yb);
                }
                LtxVaeOp::D2s(c, st) => {
                    let ca = at[ci];
                    ci += 1;
                    let drop = (st[0] == 2) as usize;
                    let nl = Lay {
                        t: lay.t * st[0] - drop,
                        h: lay.h * st[1],
                        w: lay.w * st[2],
                    };
                    let co = c.cout / (st[0] * st[1] * st[2]);
                    zero_amax(self);
                    let ok = self.run(|e| {
                        prep(e, x, lay, c.cin, 2);
                        conv(e, ca, lay, B::T, None, true);
                        e.pipe(p.d2s);
                        e.buf(id(B::T), 0, 0);
                        e.buf(id(yb), 0, 1);
                        e.bytes(
                            &Lvd {
                                s: lay.geo(),
                                d: nl.geo(),
                                c: co as u32,
                                p1: st[0] as u32,
                                p2: st[1] as u32,
                                p3: st[2] as u32,
                                drop: drop as u32,
                                lo: LO as u32,
                                t: nl.t as u32,
                                _p: 0,
                            },
                            2,
                        );
                        e.dispatch(((nl.t * nl.h * nl.w * co).div_ceil(256), 1), 256);
                    });
                    if !ok {
                        return false;
                    }
                    lay = nl;
                    std::mem::swap(&mut x, &mut yb);
                }
            }
            progress(oi + 2, total);
        }
        // Norm, SiLU, conv_out, the patches.
        let ca = at[ci];
        let ok = self.run(|e| {
            prep(e, x, lay, ca.cin, 1);
            conv(e, ca, lay, B::T, None, false);
            e.pipe(p.out);
            e.buf(id(B::T), 0, 0);
            e.buf(id(B::Out), 0, 1);
            e.bytes(
                &Lvo {
                    s: lay.geo(),
                    c: ca.cout as u32,
                    pq: pq as u32,
                    oc: oc as u32,
                    lo: LO as u32,
                    t: lay.t as u32,
                    _p: [0; 3],
                },
                2,
            );
            e.dispatch((frames.div_ceil(256), 1), 256);
        });
        if !ok {
            return false;
        }
        out.resize(frames, 0.0);
        // SAFETY: Out holds `frames` floats.
        unsafe { super::super::par_copy(out.as_mut_ptr(), self.host::<f32>(B::Out), frames) };
        progress(total, total);
        true
    }

    fn encode(
        &mut self,
        m: &LtxVaeEncoder,
        xin: &[f32],
        t: usize,
        h: usize,
        w: usize,
        out: &mut Vec<f32>,
        progress: &mut dyn FnMut(usize, usize),
    ) -> bool {
        let mut all = vec![&m.conv_in];
        for op in &m.ops {
            match op {
                LtxVaeEncOp::Res(a, b) => {
                    all.push(a);
                    all.push(b);
                }
                LtxVaeEncOp::S2d(c, _, _) => all.push(c),
            }
        }
        all.push(&m.conv_out);
        let c0 = m.conv_in.cin;
        if t == 0 || h == 0 || w == 0 || xin.len() != t * h * w * c0 || m.latent > m.conv_out.cout {
            return false;
        }
        for c in &all {
            let rows = c.cout.next_multiple_of(64);
            if !c.cin.is_multiple_of(32) || c.w.len() != rows * 27 * c.cin || c.bias.len() != rows {
                return false;
            }
        }
        // The shapes through the encoder: the largest activation, convolution output and half
        // input.
        let (mut lay, mut ch) = (Lay { t, h, w }, m.conv_in.cout);
        let (mut act, mut conv_out, mut half) = (lay.rows() * c0.max(ch) * 4, lay.rows() * ch * 4, lay.rows() * c0 * 2);
        for op in &m.ops {
            match op {
                LtxVaeEncOp::Res(a, b) => {
                    if a.cin != ch || b.cout != ch {
                        return false;
                    }
                    half = half.max(lay.rows() * a.cin.max(b.cin) * 2);
                    conv_out = conv_out.max(lay.rows() * a.cout * 4);
                }
                LtxVaeEncOp::S2d(c, st, group) => {
                    let dup = (st[0] == 2) as usize;
                    let n = st[0] * st[1] * st[2];
                    let lc = Lay { t: lay.t + dup, ..lay };
                    if c.cin != ch
                        || !lc.t.is_multiple_of(st[0])
                        || !lay.h.is_multiple_of(st[1])
                        || !lay.w.is_multiple_of(st[2])
                        || ch * n != c.cout * n * group
                    {
                        return false;
                    }
                    half = half.max(lc.rows() * c.cin * 2);
                    conv_out = conv_out.max(lc.rows() * c.cout * 4);
                    act = act.max(lc.rows() * ch * 4);
                    lay = Lay {
                        t: lc.t / st[0],
                        h: lay.h / st[1],
                        w: lay.w / st[2],
                    };
                    ch = c.cout * n;
                    act = act.max(lay.rows() * ch * 4);
                }
            }
        }
        if ch != m.conv_out.cin {
            return false;
        }
        half = half.max(lay.rows() * ch * 2);
        conv_out = conv_out.max(lay.rows() * m.conv_out.cout * 4);
        let ok = self.ensure(B::X, act)
            && self.ensure(B::Y, act)
            && self.ensure(B::T, conv_out)
            && self.ensure(B::H, half)
            && self.ensure(B::Amax, 256);
        if !ok {
            return false;
        }
        // Weights and biases, once per encoder.
        let key = m.conv_in.w.as_ptr() as usize;
        let mut at = Vec::with_capacity(all.len());
        let (mut wb, mut bb) = (0usize, 0usize);
        for c in &all {
            at.push(ConvAt {
                w: wb,
                b: bb,
                cin: c.cin,
                cout: c.cout,
            });
            wb += (c.w.len() * 2).next_multiple_of(256);
            bb += (c.bias.len() * 4).next_multiple_of(256);
        }
        if self.weights_of != key {
            if !self.ensure(B::W, wb) || !self.ensure(B::Bias, bb) {
                return false;
            }
            let (wp, bp) = (self.host::<u8>(B::W), self.host::<u8>(B::Bias));
            for (c, a) in all.iter().zip(&at) {
                // SAFETY: the buffers hold wb / bb bytes, every convolution inside them; the
                // device is idle.
                unsafe {
                    std::ptr::copy_nonoverlapping(c.w.as_ptr() as *const u8, wp.add(a.w), c.w.len() * 2);
                    super::super::par_copy(bp.add(a.b) as *mut f32, c.bias.as_ptr(), c.bias.len());
                }
            }
            self.weights_of = key;
        }
        // The frames into X (borders and guards are never read).
        let lay0 = Lay { t, h, w };
        let xp = self.host::<f32>(B::X);
        for f in 0..t {
            for y in 0..h {
                let r = lay0.g() + (LO + f) * lay0.p() + (y + 1) * (w + 2) + 1;
                // SAFETY: X holds lay0.rows() rows of c0 floats.
                unsafe {
                    std::ptr::copy_nonoverlapping(xin.as_ptr().add((f * h + y) * w * c0), xp.add(r * c0), w * c0);
                }
            }
        }
        let ids: [Id; NBUF] = std::array::from_fn(|i| self.bufs[i].id);
        let id = |b: B| ids[b as usize];
        let p = &self.p;
        // The half rows of `src` (c channels, layout sl) into H in layout dl (the same size, `shift`
        // more frames in front, each a repeat of the first).
        let prep = |e: &super::super::decode::FastEnc, src: B, sl: Lay, dl: Lay, c: usize, mode: u32, shift: usize| {
            if mode == 2 {
                e.pipe(p.amax);
                e.buf(id(src), 0, 0);
                e.buf(id(B::Amax), 0, 1);
                e.bytes(
                    &Lva {
                        g: sl.geo(),
                        c: c as u32,
                        lo: LO as u32,
                        hi: (LO + sl.t) as u32,
                        rows: sl.rows() as u32,
                    },
                    2,
                );
                e.dispatch(((sl.rows() * c / 4).div_ceil(256), 1), 256);
                e.barrier();
            }
            e.pipe(p.prep);
            e.buf(id(src), 0, 0);
            e.buf(id(B::H), 0, 1);
            e.buf(id(B::Amax), 0, 2);
            e.bytes(
                &Lvp {
                    g: dl.geo(),
                    c: c as u32,
                    nimg: dl.nimg() as u32,
                    lo: LO as u32,
                    hi: (LO + sl.t) as u32,
                    mode,
                    rows: dl.rows() as u32,
                    eps: 1e-8,
                    shift: shift as u32,
                    _p: [0; 3],
                },
                3,
            );
            e.dispatch((dl.rows().div_ceil(8), 1), 256);
            e.barrier();
        };
        // The causal convolution `a` of H (layout l) into `y`, plus `add` (same layout) when given.
        let conv = |e: &super::super::decode::FastEnc, a: ConvAt, l: Lay, y: B, add: Option<B>, scaled: bool| {
            let (r0, r_end) = l.span();
            let nct = a.cout.div_ceil(64);
            let tiles = (r_end - r0).div_ceil(TM) * nct;
            let (p_, w2) = (l.p() as i32, l.w as i32 + 2);
            for kt in 0..3 {
                let mut off = [0i32; 9];
                for (k, o) in off.iter_mut().enumerate() {
                    let (ky, kx) = ((k / 3) as i32, (k % 3) as i32);
                    *o = (kt as i32 - 2) * p_ + (ky - 1) * w2 + (kx - 1);
                }
                let addb = if kt == 0 { add } else { Some(y) };
                e.pipe(p.gemm);
                e.buf(id(B::W), a.w + kt * 9 * a.cin * 2, 0);
                e.buf(id(B::H), 0, 1);
                e.buf(id(y), 0, 2);
                e.buf(id(B::Bias), a.b, 3);
                e.bytes(
                    &Ug {
                        n_out: a.cout as u32,
                        ldw: (27 * a.cin) as u32,
                        cin: a.cin as u32,
                        taps: 9,
                        off,
                        a_rows: l.rows() as u32,
                        r0: r0 as u32,
                        r_end: r_end as u32,
                        omode: 2,
                        ldy: a.cout as u32,
                        has_bias: (kt == 0) as u32,
                        bstride: 0,
                        add: addb.is_some() as u32,
                        nimg: l.nimg() as u32,
                        g: l.geo(),
                        scaled: scaled as u32,
                        full: tiles as u32,
                        rs: 1,
                        kper: (9 * a.cin) as u32,
                        nct: nct as u32,
                        tm: TM as u32,
                        nrem: 0,
                        _p: 0,
                    },
                    4,
                );
                e.buf(id(addb.unwrap_or(y)), 0, 5);
                e.buf(id(B::Amax), 0, 6);
                e.buf(id(B::T), 0, 7);
                e.dispatch((tiles, 1), 128);
                e.barrier();
            }
        };
        let zero_amax = |s: &Self| {
            // SAFETY: Amax holds 256 bytes; the device is idle.
            unsafe { std::ptr::write_bytes(s.host::<u8>(B::Amax), 0, 256) };
        };
        let total = m.ops.len() + 2;
        let mut lay = lay0;
        zero_amax(self);
        let ok = self.run(|e| {
            prep(e, B::X, lay, lay, c0, 2, 0);
            conv(e, at[0], lay, B::Y, None, true);
        });
        if !ok {
            return false;
        }
        let (mut x, mut yb) = (B::Y, B::X);
        let mut ch = m.conv_in.cout;
        progress(1, total);
        let mut ci = 1;
        for (oi, op) in m.ops.iter().enumerate() {
            match op {
                LtxVaeEncOp::Res(a, b) => {
                    let (ca, cb) = (at[ci], at[ci + 1]);
                    ci += 2;
                    let ok = self.run(|e| {
                        prep(e, x, lay, lay, a.cin, 1, 0);
                        conv(e, ca, lay, B::T, None, false);
                        prep(e, B::T, lay, lay, b.cin, 1, 0);
                        conv(e, cb, lay, yb, Some(x), false);
                    });
                    if !ok {
                        return false;
                    }
                    std::mem::swap(&mut x, &mut yb);
                }
                LtxVaeEncOp::S2d(c, st, group) => {
                    let ca = at[ci];
                    ci += 1;
                    let dup = (st[0] == 2) as usize;
                    let lc = Lay { t: lay.t + dup, ..lay };
                    let nl = Lay {
                        t: lc.t / st[0],
                        h: lay.h / st[1],
                        w: lay.w / st[2],
                    };
                    let oc = c.cout * st[0] * st[1] * st[2];
                    zero_amax(self);
                    let ok = self.run(|e| {
                        prep(e, x, lay, lc, c.cin, 2, dup);
                        conv(e, ca, lc, B::T, None, true);
                        e.pipe(p.s2d);
                        e.buf(id(B::T), 0, 0);
                        e.buf(id(x), 0, 1);
                        e.buf(id(yb), 0, 2);
                        e.bytes(
                            &Lvs {
                                s: lay.geo(),
                                d: nl.geo(),
                                co: c.cout as u32,
                                xc: ch as u32,
                                p1: st[0] as u32,
                                p2: st[1] as u32,
                                p3: st[2] as u32,
                                group: *group as u32,
                                dup: dup as u32,
                                lo: LO as u32,
                                t: nl.t as u32,
                                _p: 0,
                            },
                            3,
                        );
                        e.dispatch(((nl.t * nl.h * nl.w * oc).div_ceil(256), 1), 256);
                    });
                    if !ok {
                        return false;
                    }
                    lay = nl;
                    ch = oc;
                    std::mem::swap(&mut x, &mut yb);
                }
            }
            progress(oi + 2, total);
        }
        // Norm, SiLU, conv_out; the means read back.
        let ca = at[ci];
        let ok = self.run(|e| {
            prep(e, x, lay, lay, ca.cin, 1, 0);
            conv(e, ca, lay, B::T, None, false);
        });
        if !ok {
            return false;
        }
        let (co, lc) = (m.conv_out.cout, m.latent);
        out.clear();
        out.reserve(lay.t * lay.h * lay.w * lc);
        let tp = self.host::<f32>(B::T);
        for f in 0..lay.t {
            for y in 0..lay.h {
                for xx in 0..lay.w {
                    let r = lay.g() + (LO + f) * lay.p() + (y + 1) * (lay.w + 2) + xx + 1;
                    // SAFETY: T holds lay.rows() rows of co floats.
                    out.extend_from_slice(unsafe { std::slice::from_raw_parts(tp.add(r * co), lc) });
                }
            }
        }
        progress(total, total);
        true
    }
}
