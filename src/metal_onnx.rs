//! The ONNX interpreter's operators on Metal ([`crate::gpu::onnx`]): constants in their own
//! buffers, a session's tensors in shared buffers (reused once released), a run's steps in one
//! command buffer of the `ox_*` kernels of `metal_onnx.metal`, a barrier after each.

use super::decode::{FastEnc, Fns, run_concurrent};
use super::*;
use crate::gpu::onnx::{Device, Post, Session, Step};

pub(crate) const SRC: &str = include_str!("metal_onnx.metal");

/// Room past a session buffer's values: 32 rows of up to 1024 floats (`Step::Pointwise` reads
/// whole 32-row tiles).
const SLACK: usize = 32 * 1024;

#[derive(Clone, Copy)]
struct Pipes {
    conv: Id,
    pw: Id,
    round: Id,
    add: Id,
    unary: Id,
    resize: Id,
    cat: Id,
}

fn new_buf(be: &MetalBackend, bytes: usize) -> Option<(Id, usize)> {
    let new_len: MsgNewBufLen = be.o.msg_send();
    let bytes = bytes.max(256).div_ceil(PAGE) * PAGE;
    let id = unsafe { new_len(be.device, be.sels.new_buffer_len, bytes, 0) };
    (!id.is_null()).then_some((id, bytes))
}

fn release(be: &MetalBackend, id: Id) {
    if !id.is_null() {
        let void: MsgVoid = be.o.msg_send();
        unsafe { void(id, be.sels.release) };
    }
}

pub(crate) struct MetalOnnx {
    be: &'static MetalBackend,
    p: Pipes,
    consts: std::sync::Mutex<Vec<Id>>,
}

// SAFETY: the buffers and pipelines are Metal objects usable from any thread; the list is
// behind a mutex.
unsafe impl Send for MetalOnnx {}
unsafe impl Sync for MetalOnnx {}

impl Drop for MetalOnnx {
    fn drop(&mut self) {
        for &id in self.consts.lock().unwrap_or_else(|e| e.into_inner()).iter() {
            release(self.be, id);
        }
    }
}

impl MetalBackend {
    /// [`crate::gpu::Backend::onnx_device`].
    pub(crate) fn onnx_device_impl(&'static self) -> Option<std::sync::Arc<dyn Device>> {
        let pipe = |n: &str| self.named_pipeline(n);
        let p = Pipes {
            conv: pipe("ox_conv")?,
            pw: pipe("ox_pw")?,
            round: pipe("ox_round")?,
            add: pipe("ox_add")?,
            unary: pipe("ox_unary")?,
            resize: pipe("ox_resize")?,
            cat: pipe("ox_cat")?,
        };
        Some(std::sync::Arc::new(MetalOnnx {
            be: self,
            p,
            consts: std::sync::Mutex::new(Vec::new()),
        }))
    }
}

impl Device for MetalOnnx {
    fn constant(&self, v: &[f32]) -> Option<usize> {
        let (id, _) = new_buf(self.be, v.len() * 4)?;
        let f = Fns::new(self.be);
        // SAFETY: the new buffer holds v.len() floats.
        unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), f.contents(id) as *mut f32, v.len()) };
        let mut c = self.consts.lock().unwrap_or_else(|e| e.into_inner());
        c.push(id);
        Some(c.len() - 1)
    }

    fn session(&self) -> Option<Box<dyn Session + '_>> {
        let consts = self.consts.lock().unwrap_or_else(|e| e.into_inner()).clone();
        Some(Box::new(MetalOnnxSession {
            dev: self,
            f: Fns::new(self.be),
            consts,
            bufs: Vec::new(),
        }))
    }
}

struct SBuf {
    id: Id,
    /// Floats it holds (without the slack).
    cap: usize,
    /// Floats asked for.
    len: usize,
    free: bool,
}

struct MetalOnnxSession<'a> {
    dev: &'a MetalOnnx,
    f: Fns,
    consts: Vec<Id>,
    bufs: Vec<SBuf>,
}

impl Drop for MetalOnnxSession<'_> {
    fn drop(&mut self) {
        for b in &self.bufs {
            release(self.dev.be, b.id);
        }
    }
}

#[repr(C)]
struct OxPost {
    kind: u32,
    s: f32,
    zp: f32,
    lo: f32,
    hi: f32,
    s2: f32,
    zp2: f32,
    _p: f32,
}

#[repr(C)]
struct OxConv {
    n: u32,
    h: u32,
    w: u32,
    cin: u32,
    oh: u32,
    ow: u32,
    cout: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    dh: u32,
    dw: u32,
    pt: u32,
    pl: u32,
    cpg: u32,
    copg: u32,
    s_in: f32,
    _p: [u32; 2],
}

#[repr(C)]
struct OxPw {
    npix: u32,
    cin: u32,
    cout: u32,
    s_in: f32,
}

#[repr(C)]
struct OxAdd {
    n: u32,
    exact: u32,
    sa: f32,
    sb: f32,
    rac: f32,
    rbc: f32,
    zc: f32,
    _p: f32,
}

#[repr(C)]
struct OxRs {
    n: u32,
    h: u32,
    w: u32,
    c: u32,
    oh: u32,
    ow: u32,
    _p: [u32; 2],
}

#[repr(C)]
struct OxCat {
    npix: u32,
    cs: u32,
    cd: u32,
    off: u32,
}

/// The kernels' form of a [`Post`] (the multipliers' buffer apart).
fn post_of(p: &Post) -> OxPost {
    let mut q = OxPost {
        kind: p.relu as u32
            | (p.fq.is_some() as u32) << 1
            | (p.requant.is_some() as u32) << 2
            | (p.sigmoid as u32) << 3
            | (p.round.is_some() as u32) << 4,
        s: 1.0,
        zp: 0.0,
        lo: 0.0,
        hi: 0.0,
        s2: 1.0,
        zp2: 0.0,
        _p: 0.0,
    };
    if let Some([s, zp, lo, hi, s2, zp2]) = p.fq {
        (q.s, q.zp, q.lo, q.hi, q.s2, q.zp2) = (s, zp, lo, hi, s2, zp2);
    }
    if let Some([zp, lo, hi, s2, zp2]) = p.requant.map(|r| r.1).or(p.round) {
        (q.zp, q.lo, q.hi, q.s2, q.zp2) = (zp, lo, hi, s2, zp2);
    }
    q
}

impl MetalOnnxSession<'_> {
    fn buf(&self, i: usize) -> Id {
        self.bufs[i].id
    }

    /// Encodes one step (`tmp`: buffers of the index tables of resizes, kept until the run ends).
    fn encode(&self, e: &FastEnc, s: &Step, tmp: &mut Vec<Id>) -> bool {
        let p = &self.dev.p;
        let groups = |n: usize, t: usize| (n.div_ceil(t), 1);
        // The multipliers of a requantization (any buffer when there are none).
        let mult = |post: &Post| {
            post.requant
                .map_or(self.consts.first().copied().unwrap_or(self.bufs[0].id), |(m, _)| self.consts[m])
        };
        match s {
            Step::Conv { x, y, w, b, g, s_in, post } => {
                e.pipe(p.conv);
                e.buf(self.buf(*x), 0, 0);
                e.buf(self.consts[*w], 0, 1);
                e.buf(self.consts[*b], 0, 2);
                e.buf(self.buf(*y), 0, 3);
                e.bytes(
                    &OxConv {
                        n: g.n as u32,
                        h: g.h as u32,
                        w: g.w as u32,
                        cin: g.cin as u32,
                        oh: g.oh as u32,
                        ow: g.ow as u32,
                        cout: g.cout as u32,
                        kh: g.kh as u32,
                        kw: g.kw as u32,
                        sh: g.stride[0] as u32,
                        sw: g.stride[1] as u32,
                        dh: g.dil[0] as u32,
                        dw: g.dil[1] as u32,
                        pt: g.pad[0] as u32,
                        pl: g.pad[1] as u32,
                        cpg: (g.cin / g.groups) as u32,
                        copg: (g.cout / g.groups) as u32,
                        s_in: *s_in,
                        _p: [0; 2],
                    },
                    4,
                );
                e.bytes(&post_of(post), 5);
                e.buf(mult(post), 0, 6);
                e.dispatch(groups(g.n * g.oh * g.ow * g.cout, 256), 256);
            }
            Step::Pointwise {
                x,
                y,
                w,
                b,
                npix,
                cin,
                cout,
                post,
            } => {
                e.pipe(p.pw);
                e.buf(self.buf(*x), 0, 0);
                e.buf(self.consts[*w], 0, 1);
                e.buf(self.consts[*b], 0, 2);
                e.buf(self.buf(*y), 0, 3);
                e.bytes(
                    &OxPw {
                        npix: *npix as u32,
                        cin: *cin as u32,
                        cout: *cout as u32,
                        s_in: 0.0,
                    },
                    4,
                );
                e.bytes(&post_of(post), 5);
                e.buf(mult(post), 0, 6);
                e.dispatch((cout.div_ceil(32), npix.div_ceil(32)), 128);
            }
            Step::Round { x, y, n, s } => {
                e.pipe(p.round);
                e.buf(self.buf(*x), 0, 0);
                e.buf(self.buf(*y), 0, 1);
                e.bytes(s, 2);
                e.bytes(&(*n as u32), 3);
                e.dispatch(groups(*n, 256), 256);
            }
            Step::Add { a, b, y, n, exact, post } => {
                let [sa, sb, rac, rbc, zc] = exact.unwrap_or([1.0, 1.0, 1.0, 1.0, 0.0]);
                e.pipe(p.add);
                e.buf(self.buf(*a), 0, 0);
                e.buf(self.buf(*b), 0, 1);
                e.buf(self.buf(*y), 0, 2);
                e.bytes(
                    &OxAdd {
                        n: *n as u32,
                        exact: exact.is_some() as u32,
                        sa,
                        sb,
                        rac,
                        rbc,
                        zc,
                        _p: 0.0,
                    },
                    3,
                );
                e.bytes(&post_of(post), 4);
                e.dispatch(groups(*n, 256), 256);
            }
            Step::Unary { x, y, n, post } => {
                e.pipe(p.unary);
                e.buf(self.buf(*x), 0, 0);
                e.buf(self.buf(*y), 0, 1);
                e.bytes(&(*n as u32), 2);
                e.bytes(&post_of(post), 3);
                e.dispatch(groups(*n, 256), 256);
            }
            Step::Resize { x, y, ys, xs, n, h, w, c } => {
                let (Some((yb, _)), Some((xb, _))) = (new_buf(self.dev.be, ys.len() * 4), new_buf(self.dev.be, xs.len() * 4)) else {
                    return false;
                };
                tmp.extend([yb, xb]);
                // SAFETY: the new buffers hold the tables.
                unsafe {
                    std::ptr::copy_nonoverlapping(ys.as_ptr(), self.f.contents(yb) as *mut u32, ys.len());
                    std::ptr::copy_nonoverlapping(xs.as_ptr(), self.f.contents(xb) as *mut u32, xs.len());
                }
                e.pipe(p.resize);
                e.buf(self.buf(*x), 0, 0);
                e.buf(self.buf(*y), 0, 1);
                e.buf(yb, 0, 2);
                e.buf(xb, 0, 3);
                e.bytes(
                    &OxRs {
                        n: *n as u32,
                        h: *h as u32,
                        w: *w as u32,
                        c: *c as u32,
                        oh: ys.len() as u32,
                        ow: xs.len() as u32,
                        _p: [0; 2],
                    },
                    4,
                );
                e.dispatch(groups(n * ys.len() * xs.len() * c, 256), 256);
            }
            Step::Cat {
                src,
                dst,
                npix,
                cs,
                cd,
                off,
            } => {
                e.pipe(p.cat);
                e.buf(self.buf(*src), 0, 0);
                e.buf(self.buf(*dst), 0, 1);
                e.bytes(
                    &OxCat {
                        npix: *npix as u32,
                        cs: *cs as u32,
                        cd: *cd as u32,
                        off: *off as u32,
                    },
                    2,
                );
                e.dispatch(groups(npix * cs, 256), 256);
            }
        }
        e.barrier();
        true
    }
}

impl Session for MetalOnnxSession<'_> {
    fn alloc(&mut self, floats: usize) -> Option<usize> {
        let floats = floats.max(1);
        // The smallest free buffer that holds it.
        let best = self
            .bufs
            .iter()
            .enumerate()
            .filter(|(_, b)| b.free && b.cap >= floats)
            .min_by_key(|(_, b)| b.cap)
            .map(|(i, _)| i);
        if let Some(i) = best {
            self.bufs[i].free = false;
            self.bufs[i].len = floats;
            return Some(i);
        }
        let (id, bytes) = new_buf(self.dev.be, (floats + SLACK) * 4)?;
        self.bufs.push(SBuf {
            id,
            cap: bytes / 4 - SLACK,
            len: floats,
            free: false,
        });
        Some(self.bufs.len() - 1)
    }

    fn release(&mut self, buf: usize) {
        if let Some(b) = self.bufs.get_mut(buf) {
            b.free = true;
        }
    }

    fn data(&mut self, buf: usize) -> &mut [f32] {
        let b = &self.bufs[buf];
        // SAFETY: the buffer holds at least `len` floats; no step is running.
        unsafe { std::slice::from_raw_parts_mut(self.f.contents(b.id) as *mut f32, b.len) }
    }

    fn run(&mut self, steps: &[Step]) -> bool {
        if steps.is_empty() {
            return true;
        }
        let mut tmp = Vec::new();
        let ok = std::cell::Cell::new(true);
        let this = &*self;
        let done = {
            let tmp_cell = std::cell::RefCell::new(&mut tmp);
            let all = 0..steps.len();
            run_concurrent(this.dev.be, &this.f, std::slice::from_ref(&all), |e, r| {
                for s in &steps[r] {
                    if !this.encode(e, s, &mut tmp_cell.borrow_mut()) {
                        ok.set(false);
                        return;
                    }
                }
            })
            .0
        };
        for id in tmp {
            release(self.dev.be, id);
        }
        done && ok.get()
    }
}
