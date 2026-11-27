//! The video VAE on CUDA with the frames kept on the device: the passes of [`super`] over device
//! buffers (float32, `[pixels][channels]`), one launch per convolution or norm instead of a round
//! trip through host memory per layer. A causal convolution reads its (up to three) input frames
//! where they are, the 2x upsampling in front of a convolution is part of its input addressing,
//! the residual sum part of its output. The mid block's attention (one head over the pixels of a
//! latent frame) runs on the host.
//!
//! The weights are float32 on the device (the bf16-rounded kernels this VAE runs with, exact),
//! the products float32 FMAs, so a pass differs from the host's by the order of the sums.
//! Activations of the size of a few frames per layer stay allocated during a pass and are freed
//! at its end (so are the weights on a discrete card, whose memory the transformer wants).
//!
//! `OXINFER_AVATAR_VAE_GPU=0` keeps the per-layer path; a pass that cannot run here (no CUDA
//! backend, out of device memory, a kernel error) returns `None` and the host path runs.

use super::{Attn, CConv, Frame, Layer, Res, Rms, Stack, VideoVae};
use crate::cuda::CudaBackend;
use crate::sd::ops::Conv;
use anyhow::{Result, anyhow, bail};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{CString, c_char, c_int, c_void};
use std::rc::Rc;
use std::sync::Mutex;

dlsym::library! {
    /// The driver entry points this engine calls.
    struct Cuh {
        unsafe extern "C" {
            fn cuMemAlloc_v2(ptr: *mut u64, bytes: usize) -> c_int;
            fn cuMemFree_v2(ptr: u64) -> c_int;
            fn cuMemcpyHtoD_v2(dst: u64, src: *const c_void, bytes: usize) -> c_int;
            fn cuMemcpyDtoH_v2(dst: *mut c_void, src: u64, bytes: usize) -> c_int;
            fn cuStreamCreate(stream: *mut *mut c_void, flags: u32) -> c_int;
            fn cuStreamSynchronize(stream: *mut c_void) -> c_int;
            fn cuStreamDestroy_v2(stream: *mut c_void) -> c_int;
            fn cuModuleGetFunction(f: *mut *mut c_void, module: *mut c_void, name: *const c_char) -> c_int;
            fn cuModuleUnload(module: *mut c_void) -> c_int;
            fn cuCtxSetCurrent(ctx: *mut c_void) -> c_int;
            fn cuLaunchKernel(
                f: *mut c_void, gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32,
                shmem: u32, stream: *mut c_void, params: *mut *mut c_void, extra: *mut *mut c_void,
            ) -> c_int;
        }
    }
}

type F = *mut c_void;

fn debug(why: &str) {
    if std::env::var("OXINFER_GPU_PLAN_DEBUG").is_ok_and(|v| v != "0") {
        eprintln!("[longcat-avatar vae cuda: {why}]");
    }
}

fn bytes_of(v: &[f32]) -> &[u8] {
    // SAFETY: plain numeric slices.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// `VConvP` of `dev.cu`.
#[repr(C)]
struct VConvP {
    x: [u64; 3],
    w: u64,
    bias: u64,
    add: u64,
    y: u64,
    h: u32,
    wd: u32,
    cin: u32,
    xstride: u32,
    xoff: u32,
    k: u32,
    kt: u32,
    stride: u32,
    pad: u32,
    up: u32,
    ow: u32,
    npix: u32,
    cout: u32,
    pad2: u32,
}

/// A convolution's weights on the device: rows of `taps * cin` floats (`cin`: the layer's
/// channels per frame rounded up to a multiple of 32), a multiple of 64 of them (zero rows after
/// the real ones, and between the two halves of a layer whose outputs are read as two frames),
/// and a bias per row.
#[derive(Clone, Copy)]
struct DConv {
    w: u64,
    bias: u64,
    cin: usize,
    /// Floats per pixel of the output: the outputs rounded up to a multiple of 32 (each half).
    cout: usize,
}

#[derive(Default)]
struct State {
    convs: HashMap<usize, DConv>,
    gains: HashMap<usize, u64>,
    owned: Vec<u64>,
}

pub(super) struct Engine {
    x: Cuh,
    ctx: *mut c_void,
    module: F,
    stream: F,
    conv: F,
    rms: F,
    state: Mutex<State>,
}

// SAFETY: the raw pointers are driver handles; every use binds the context and goes through
// `state`'s lock.
unsafe impl Send for Engine {}
unsafe impl Sync for Engine {}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe {
            if self.x.cuCtxSetCurrent(self.ctx) != 0 {
                return;
            }
            self.x.cuStreamSynchronize(self.stream);
            for p in self.state.get_mut().unwrap().owned.drain(..) {
                self.x.cuMemFree_v2(p);
            }
            self.x.cuStreamDestroy_v2(self.stream);
            self.x.cuModuleUnload(self.module);
        }
    }
}

/// The device buffers of one pass: freed ones are handed out again by size.
#[derive(Default)]
struct Pool {
    free: RefCell<Vec<(usize, u64)>>,
    all: RefCell<Vec<u64>>,
}

struct Buf<'p> {
    p: u64,
    bytes: usize,
    pool: &'p Pool,
}

impl Drop for Buf<'_> {
    fn drop(&mut self) {
        // Launches run in order on one stream: a kernel queued later may take the buffer.
        self.pool.free.borrow_mut().push((self.bytes, self.p));
    }
}

/// A frame on the device: `stride` floats per pixel, its channels from `off` on (zeros after
/// them up to a multiple of 32).
#[derive(Clone)]
struct DFrame<'p> {
    buf: Rc<Buf<'p>>,
    off: usize,
    stride: usize,
}

/// What a convolution's output is.
#[derive(Clone, Copy, PartialEq)]
enum Out {
    /// A frame for the next layer (zero channels up to a multiple of 32).
    Frame,
    /// Two frames' channels, each half as [`Out::Frame`].
    Halves,
    /// The convolution's channels alone.
    Dense,
}

enum DCache<'p> {
    None,
    Rep,
    Frames(Vec<DFrame<'p>>),
}

struct DState<'p> {
    caches: Vec<DCache<'p>>,
    at: usize,
}

impl<'p> DState<'p> {
    /// The index of the next layer's cache.
    fn next(&mut self) -> usize {
        if self.at == self.caches.len() {
            self.caches.push(DCache::None);
        }
        self.at += 1;
        self.at - 1
    }

    fn frames(&self, i: usize) -> Option<Vec<DFrame<'p>>> {
        match &self.caches[i] {
            DCache::Frames(f) => Some(f.clone()),
            _ => None,
        }
    }
}

/// The last two frames of `x`, with the last cached frame in front when `x` has only one.
fn tail2<'p>(x: &[DFrame<'p>], prev: Option<&[DFrame<'p>]>) -> Vec<DFrame<'p>> {
    let mut keep: Vec<DFrame> = x[x.len().saturating_sub(2)..].to_vec();
    if keep.len() < 2
        && let Some(last) = prev.and_then(|p| p.last())
    {
        keep.insert(0, last.clone());
    }
    keep
}

impl Engine {
    pub(super) fn new() -> Option<Engine> {
        if std::env::var("OXINFER_AVATAR_VAE_GPU").is_ok_and(|v| v == "0") {
            return None;
        }
        let be = crate::gpu::active()?.as_any()?.downcast_ref::<CudaBackend>()?;
        match Self::build(be) {
            Ok(e) => Some(e),
            Err(e) => {
                debug(&format!("{e:#}"));
                None
            }
        }
    }

    fn build(be: &CudaBackend) -> Result<Engine> {
        let x: Cuh = {
            use dlsym::Bindings;
            // SAFETY: libcuda is already initialised by the backend.
            unsafe { Cuh::open_first(crate::platform::gpu::cuda::CANDIDATES.iter().copied()) }.map_err(|e| anyhow!("libcuda: {e}"))?
        };
        let ctx = be.context().ok_or_else(|| anyhow!("no context"))?;
        let module = be.cached_module(include_str!("dev.cu"), "longcat-avatar-vae-{sm}", "longcat_avatar_vae.cu")?;
        let get = |name: &str| -> Result<F> {
            let cn = CString::new(name)?;
            let mut f = std::ptr::null_mut();
            if unsafe { x.cuModuleGetFunction(&mut f, module, cn.as_ptr()) } != 0 {
                bail!("kernel {name} missing");
            }
            Ok(f)
        };
        let (conv, rms) = (get("vconv")?, get("vrms")?);
        let mut stream = std::ptr::null_mut();
        if unsafe { x.cuStreamCreate(&mut stream, 1) } != 0 {
            bail!("cuStreamCreate failed");
        }
        Ok(Engine {
            x,
            ctx,
            module,
            stream,
            conv,
            rms,
            state: Mutex::new(State::default()),
        })
    }

    fn htod(&self, dst: u64, src: &[u8]) -> Result<()> {
        crate::cuda::touch_before_upload(src);
        let ok = src.is_empty()
            || unsafe {
                self.x.cuMemcpyHtoD_v2(dst, src.as_ptr() as *const c_void, src.len()) == 0
                    && self.x.cuStreamSynchronize(std::ptr::null_mut()) == 0
            };
        if !ok {
            bail!("a copy to the device failed");
        }
        Ok(())
    }

    /// `v` in a buffer of its own that the engine keeps.
    fn keep(&self, st: &mut State, v: &[f32]) -> Result<u64> {
        let mut d = 0u64;
        if unsafe { self.x.cuMemAlloc_v2(&mut d, (v.len() * 4).max(256)) } != 0 {
            bail!("out of device memory for the weights");
        }
        st.owned.push(d);
        self.htod(d, bytes_of(v))?;
        Ok(d)
    }

    fn free_weights(&self, st: &mut State) {
        unsafe { self.x.cuStreamSynchronize(self.stream) };
        for p in st.owned.drain(..) {
            unsafe { self.x.cuMemFree_v2(p) };
        }
        st.convs.clear();
        st.gains.clear();
    }

    /// `f` over a pass's buffers and the weights on the device.
    fn pass<T>(&self, f: impl FnOnce(&mut Pass) -> Result<T>) -> Option<T> {
        let mut st = self.state.lock().unwrap();
        if unsafe { self.x.cuCtxSetCurrent(self.ctx) } != 0 {
            return None;
        }
        let pool = Pool::default();
        let out = {
            let mut p = Pass {
                e: self,
                pool: &pool,
                st: &mut st,
            };
            f(&mut p)
        };
        unsafe { self.x.cuStreamSynchronize(self.stream) };
        for p in pool.all.borrow_mut().drain(..) {
            unsafe { self.x.cuMemFree_v2(p) };
        }
        if !crate::cuda::integrated() || out.is_err() {
            self.free_weights(&mut st);
        }
        match out {
            Ok(v) => Some(v),
            Err(e) => {
                debug(&format!("{e:#}; the host path runs"));
                None
            }
        }
    }
}

struct Pass<'p, 's> {
    e: &'p Engine,
    pool: &'p Pool,
    st: &'s mut State,
}

impl<'p> Pass<'p, '_> {
    fn alloc(&self, floats: usize, stride: usize) -> Result<DFrame<'p>> {
        let bytes = (floats * 4).next_multiple_of(256);
        let mut free = self.pool.free.borrow_mut();
        let p = match free.iter().position(|&(b, _)| b == bytes) {
            Some(i) => free.swap_remove(i).1,
            None => {
                let mut d = 0u64;
                if unsafe { self.e.x.cuMemAlloc_v2(&mut d, bytes) } != 0 {
                    bail!("out of device memory for the frames");
                }
                self.pool.all.borrow_mut().push(d);
                d
            }
        };
        Ok(DFrame {
            buf: Rc::new(Buf { p, bytes, pool: self.pool }),
            off: 0,
            stride,
        })
    }

    /// A host frame `[pixels][c]` on the device, with zero channels up to a multiple of 32.
    fn upload(&self, x: &[f32], c: usize) -> Result<DFrame<'p>> {
        let cp = c.next_multiple_of(32);
        let npix = x.len() / c;
        let f = self.alloc(npix * cp, cp)?;
        // The buffer may be one a queued kernel still reads: the copy is not in the stream's order.
        if unsafe { self.e.x.cuStreamSynchronize(self.e.stream) } != 0 {
            bail!("a kernel failed");
        }
        if cp == c {
            self.e.htod(f.buf.p, bytes_of(x))?;
        } else {
            let mut v = vec![0f32; npix * cp];
            for (o, s) in v.chunks_mut(cp).zip(x.chunks(c)) {
                o[..c].copy_from_slice(s);
            }
            self.e.htod(f.buf.p, bytes_of(&v))?;
        }
        Ok(f)
    }

    /// The first `c` channels of a frame's `npix` pixels.
    fn download(&self, f: &DFrame, npix: usize, c: usize) -> Result<Frame> {
        let mut v = vec![0f32; npix * f.stride];
        let ok = f.off == 0
            && unsafe {
                self.e.x.cuStreamSynchronize(self.e.stream) == 0
                    && self.e.x.cuMemcpyDtoH_v2(v.as_mut_ptr() as *mut c_void, f.buf.p, v.len() * 4) == 0
            };
        if !ok {
            bail!("a kernel or a copy from the device failed");
        }
        if f.stride != c {
            v = v.chunks(f.stride).flat_map(|p| p[..c].iter().copied()).collect();
        }
        Ok(v)
    }

    fn go(&self, f: F, grid: (u32, u32), args: &mut [*mut c_void]) -> Result<()> {
        let rc = unsafe {
            self.e.x.cuLaunchKernel(
                f,
                grid.0.max(1),
                grid.1.max(1),
                1,
                256,
                1,
                1,
                0,
                self.e.stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            bail!("a kernel launch failed ({rc})");
        }
        Ok(())
    }

    /// The weights of `c` (whose input is `kt` frames side by side) on the device; `halves`: its
    /// outputs are two frames' channels.
    fn weights(&mut self, c: &Conv, kt: usize, halves: bool) -> Result<DConv> {
        let key = c as *const Conv as usize;
        if let Some(d) = self.st.convs.get(&key) {
            return Ok(*d);
        }
        let (rows, bias) = c.rows_f32().ok_or_else(|| anyhow!("a kernel's rows cannot be read"))?;
        let cin = c.cin / kt;
        let (taps, cp) = (c.k * c.k * kt, cin.next_multiple_of(32));
        let n_in = taps * cin;
        // Where output `o` goes.
        let half = c.cout / 2;
        let row = |o: usize| {
            if halves && o >= half {
                half.next_multiple_of(32) + o - half
            } else {
                o
            }
        };
        let cout = if halves {
            2 * half.next_multiple_of(32)
        } else {
            c.cout.next_multiple_of(32)
        };
        let n_rows = cout.next_multiple_of(64);
        let mut w = vec![0f32; n_rows * taps * cp];
        let mut b = vec![0f32; n_rows];
        for (o, r) in rows.chunks(n_in).enumerate() {
            let dst = &mut w[row(o) * taps * cp..(row(o) + 1) * taps * cp];
            for (t, g) in r.chunks(cin).enumerate() {
                dst[t * cp..t * cp + cin].copy_from_slice(g);
            }
            b[row(o)] = bias.map_or(0.0, |b| b[o]);
        }
        let d = DConv {
            w: self.e.keep(self.st, &w)?,
            bias: self.e.keep(self.st, &b)?,
            cin: cp,
            cout,
        };
        self.st.convs.insert(key, d);
        Ok(d)
    }

    /// The convolution `c` of the frames `taps` (`None`: a zero frame) of `h` x `w`, upsampled
    /// 2x first with `up`; `add` is added to the output. `out`: the output as a frame for the
    /// next layer, as two frames' channels, or as the convolution's channels alone (to download).
    #[allow(clippy::too_many_arguments)]
    fn conv(
        &mut self,
        c: &Conv,
        taps: &[Option<&DFrame<'p>>],
        h: usize,
        w: usize,
        stride: usize,
        up: bool,
        add: Option<&DFrame<'p>>,
        out: Out,
    ) -> Result<(DFrame<'p>, usize, usize)> {
        let kt = taps.len();
        if kt == 0 || kt > 3 {
            bail!("a convolution over {kt} frames");
        }
        let d = self.weights(c, kt, out == Out::Halves)?;
        let ys = if out == Out::Dense { c.cout } else { d.cout };
        let Some(first) = taps.iter().flatten().next() else {
            bail!("a convolution without an input frame");
        };
        let (xstride, xoff) = (first.stride, first.off);
        if taps.iter().flatten().any(|f| f.stride != xstride || f.off != xoff) || xstride < xoff + d.cin {
            bail!("a convolution's frames differ in layout");
        }
        let (before, after) = c.pads();
        let (ih, iw) = if up { (2 * h, 2 * w) } else { (h, w) };
        let (oh, ow) = ((ih + before + after - c.k) / stride + 1, (iw + before + after - c.k) / stride + 1);
        let y = self.alloc(oh * ow * ys, ys)?;
        let x = |i: usize| taps.get(i).copied().flatten().map_or(0, |f| f.buf.p);
        let p = VConvP {
            x: [x(0), x(1), x(2)],
            w: d.w,
            bias: d.bias,
            add: add.map_or(0, |a| a.buf.p),
            y: y.buf.p,
            h: h as u32,
            wd: w as u32,
            cin: d.cin as u32,
            xstride: xstride as u32,
            xoff: xoff as u32,
            k: c.k as u32,
            kt: kt as u32,
            stride: stride as u32,
            pad: before as u32,
            up: up as u32,
            ow: ow as u32,
            npix: (oh * ow) as u32,
            cout: ys as u32,
            pad2: 0,
        };
        if add.is_some_and(|a| a.off != 0 || a.stride != ys) {
            bail!("a residual of another layout");
        }
        let grid = ((oh * ow).div_ceil(64) as u32, ys.div_ceil(64) as u32);
        self.go(self.e.conv, grid, &mut [&p as *const _ as *mut c_void])?;
        Ok((y, oh, ow))
    }

    fn rms(&mut self, n: &Rms, x: &DFrame<'p>, npix: usize, silu: bool) -> Result<DFrame<'p>> {
        let c = n.g.len().next_multiple_of(32);
        if x.off != 0 || x.stride != c {
            bail!("a norm over a frame of another layout");
        }
        let key = n as *const Rms as usize;
        let g = match self.st.gains.get(&key) {
            Some(g) => *g,
            None => {
                let mut g = n.g.clone();
                g.resize(c, 0.0);
                let g = self.e.keep(self.st, &g)?;
                self.st.gains.insert(key, g);
                g
            }
        };
        let y = self.alloc(npix * c, c)?;
        let (cu, nu, su) = (c as u32, npix as u32, silu as u32);
        let mut args = [
            &x.buf.p as *const _ as *mut c_void,
            &g as *const _ as *mut c_void,
            &y.buf.p as *const _ as *mut c_void,
            &cu as *const _ as *mut c_void,
            &nu as *const _ as *mut c_void,
            &su as *const _ as *mut c_void,
        ];
        self.go(self.e.rms, (npix.div_ceil(8) as u32, 1), &mut args)?;
        Ok(y)
    }

    fn rms_all(&mut self, n: &Rms, x: &[DFrame<'p>], npix: usize) -> Result<Vec<DFrame<'p>>> {
        x.iter().map(|f| self.rms(n, f, npix, true)).collect()
    }

    /// [`CConv::forward`]; `adds[i]` is added to output frame `i`.
    #[allow(clippy::too_many_arguments)]
    fn cconv(
        &mut self,
        cc: &CConv,
        x: &[DFrame<'p>],
        cache: Option<&[DFrame<'p>]>,
        h: usize,
        w: usize,
        adds: Option<&[DFrame<'p>]>,
        out: Out,
    ) -> Result<Vec<DFrame<'p>>> {
        if cc.kt == 1 {
            return x
                .iter()
                .enumerate()
                .map(|(i, f)| Ok(self.conv(&cc.conv, &[Some(f)], h, w, 1, false, adds.map(|a| &a[i]), out)?.0))
                .collect();
        }
        let cached = if cc.pad_t > 0 { cache.unwrap_or(&[]) } else { &[] };
        let pad = cc.pad_t.saturating_sub(cached.len());
        let seq: Vec<Option<&DFrame>> = (0..pad)
            .map(|_| None)
            .chain(cached.iter().map(Some))
            .chain(x.iter().map(Some))
            .collect();
        let n_out = if seq.len() >= cc.kt {
            (seq.len() - cc.kt) / cc.stride_t + 1
        } else {
            0
        };
        let mut ys = Vec::with_capacity(n_out);
        for o in 0..n_out {
            let taps = &seq[o * cc.stride_t..o * cc.stride_t + cc.kt];
            if taps.iter().all(|t| t.is_none()) {
                bail!("a convolution of zero frames only");
            }
            ys.push(self.conv(&cc.conv, taps, h, w, 1, false, adds.map(|a| &a[o]), out)?.0);
        }
        Ok(ys)
    }

    /// [`CConv::cached`].
    #[allow(clippy::too_many_arguments)]
    fn cached(
        &mut self,
        cc: &CConv,
        x: &[DFrame<'p>],
        st: &mut DState<'p>,
        h: usize,
        w: usize,
        adds: Option<&[DFrame<'p>]>,
        out: Out,
    ) -> Result<Vec<DFrame<'p>>> {
        let i = st.next();
        let prev = st.frames(i);
        let y = self.cconv(cc, x, prev.as_deref(), h, w, adds, out)?;
        st.caches[i] = DCache::Frames(tail2(x, prev.as_deref()));
        Ok(y)
    }

    fn res(&mut self, r: &Res, x: Vec<DFrame<'p>>, st: &mut DState<'p>, h: usize, w: usize) -> Result<Vec<DFrame<'p>>> {
        let sc = match &r.shortcut {
            Some(c) => Some(self.cconv(c, &x, None, h, w, None, Out::Frame)?),
            None => None,
        };
        let a = self.rms_all(&r.norm1, &x, h * w)?;
        let a = self.cached(&r.conv1, &a, st, h, w, None, Out::Frame)?;
        let a = self.rms_all(&r.norm2, &a, h * w)?;
        self.cached(&r.conv2, &a, st, h, w, Some(sc.as_deref().unwrap_or(&x)), Out::Frame)
    }

    /// The attention of a frame, on the host.
    fn attn(&mut self, a: &Attn, f: &DFrame<'p>, h: usize, w: usize) -> Result<DFrame<'p>> {
        let x = self.download(f, h * w, a.c)?;
        self.upload(&a.forward(&x, h, w), a.c)
    }

    fn layer(&mut self, l: &Layer, x: Vec<DFrame<'p>>, st: &mut DState<'p>, h: usize, w: usize) -> Result<(Vec<DFrame<'p>>, usize, usize)> {
        match l {
            Layer::Res(r) => Ok((self.res(r, x, st, h, w)?, h, w)),
            Layer::Attn(a) => {
                let y = x.iter().map(|f| self.attn(a, f, h, w)).collect::<Result<Vec<_>>>()?;
                Ok((y, h, w))
            }
            Layer::Down(c, time) => {
                let (mut oh, mut ow) = (h, w);
                let mut y = Vec::with_capacity(x.len());
                for f in &x {
                    let (g, a, b) = self.conv(c, &[Some(f)], h, w, 2, false, None, Out::Frame)?;
                    (oh, ow) = (a, b);
                    y.push(g);
                }
                let Some(tc) = time else { return Ok((y, oh, ow)) };
                let i = st.next();
                match st.frames(i) {
                    Some(prev) => {
                        let mut seq = vec![prev.last().unwrap().clone()];
                        seq.extend(y);
                        let keep = seq.last().unwrap().clone();
                        let out = self.cconv(tc, &seq, None, oh, ow, None, Out::Frame)?;
                        st.caches[i] = DCache::Frames(vec![keep]);
                        Ok((out, oh, ow))
                    }
                    None => {
                        st.caches[i] = DCache::Frames(y.clone());
                        Ok((y, oh, ow))
                    }
                }
            }
            Layer::Up(c, time) => {
                let mut x = x;
                if let Some(tc) = time {
                    let i = st.next();
                    if matches!(st.caches[i], DCache::None) {
                        st.caches[i] = DCache::Rep;
                    } else {
                        let prev = st.frames(i);
                        let mut keep = tail2(&x, prev.as_deref());
                        if keep.len() < 2 && prev.is_none() {
                            let ch = tc.cin;
                            keep.insert(0, self.upload(&vec![0f32; h * w * ch], ch)?);
                        }
                        let y = self.cconv(tc, &x, prev.as_deref(), h, w, None, Out::Halves)?;
                        st.caches[i] = DCache::Frames(keep);
                        // Channels `[0, C)` are the first of each pair of frames, `[C, 2C)` the second.
                        let ch = tc.cin.next_multiple_of(32);
                        x = y
                            .iter()
                            .flat_map(|f| {
                                (0..2).map(|half| DFrame {
                                    buf: f.buf.clone(),
                                    off: half * ch,
                                    stride: 2 * ch,
                                })
                            })
                            .collect();
                    }
                }
                let mut y = Vec::with_capacity(x.len());
                for f in &x {
                    y.push(self.conv(c, &[Some(f)], h, w, 1, true, None, Out::Frame)?.0);
                }
                Ok((y, 2 * h, 2 * w))
            }
        }
    }

    /// [`Stack::forward`].
    fn stack(&mut self, s: &Stack, x: &[DFrame<'p>], st: &mut DState<'p>, h: usize, w: usize) -> Result<(Vec<DFrame<'p>>, usize, usize)> {
        st.at = 0;
        let (mut x, mut h, mut w) = (self.cached(&s.conv_in, x, st, h, w, None, Out::Frame)?, h, w);
        for l in &s.layers {
            (x, h, w) = self.layer(l, x, st, h, w)?;
        }
        let a = self.rms_all(&s.norm_out, &x, h * w)?;
        drop(x);
        Ok((self.cached(&s.conv_out, &a, st, h, w, None, Out::Dense)?, h, w))
    }
}

/// [`VideoVae::encode`]'s chunks on the device: the encoder's output frames (`[lh * lw][2 z]`)
/// and the latent size.
pub(super) fn encode(e: &Engine, vae: &VideoVae, frames: &[Frame], h: usize, w: usize) -> Option<(Vec<Frame>, usize, usize)> {
    let enc = vae.enc.as_ref()?;
    let st_t = vae.scale_t();
    let cout = enc.conv_out.conv.cout;
    e.pass(|p| {
        let mut st = DState { caches: Vec::new(), at: 0 };
        let mut out = Vec::new();
        let (mut lh, mut lw) = (0, 0);
        let mut at = 0;
        while at < frames.len() {
            let n = if at == 0 { 1 } else { st_t };
            let x = frames[at..at + n].iter().map(|f| p.upload(f, 3)).collect::<Result<Vec<_>>>()?;
            let (y, a, b) = p.stack(enc, &x, &mut st, h, w)?;
            (lh, lw) = (a, b);
            for f in &y {
                out.push(p.download(f, a * b, cout)?);
            }
            at += n;
        }
        Ok((out, lh, lw))
    })
}

/// [`VideoVae::decode`]'s chunks on the device: `x` are the latent frames after `post_quant`
/// (`[h * w][z]`); the frames `[64 h w][3]`, not clamped.
pub(super) fn decode(
    e: &Engine,
    vae: &VideoVae,
    x: &[Frame],
    h: usize,
    w: usize,
    progress: &mut dyn FnMut(usize, usize),
) -> Option<Vec<Frame>> {
    let z = vae.cfg.z_dim;
    e.pass(|p| {
        let mut st = DState { caches: Vec::new(), at: 0 };
        let mut out = Vec::new();
        for (i, f) in x.iter().enumerate() {
            let up = p.upload(f, z)?;
            let (y, oh, ow) = p.stack(&vae.dec, &[up], &mut st, h, w)?;
            for g in &y {
                out.push(p.download(g, oh * ow, 3)?);
            }
            progress(i + 1, x.len());
        }
        Ok(out)
    })
}

#[cfg(test)]
mod tests {
    use super::super::State;
    use super::*;

    fn backend() -> Option<&'static CudaBackend> {
        static B: std::sync::OnceLock<Option<CudaBackend>> = std::sync::OnceLock::new();
        B.get_or_init(|| CudaBackend::new(0).ok()).as_ref()
    }

    fn close(what: &str, dev: &[Frame], host: &[Frame]) {
        assert_eq!(dev.len(), host.len(), "{what}: frames");
        for (i, (a, b)) in dev.iter().zip(host).enumerate() {
            assert_eq!(a.len(), b.len(), "{what} frame {i}: size");
            let scale = b.iter().map(|v| v.abs()).sum::<f32>() / b.len() as f32;
            let max = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
            assert!(max < 1e-4 * scale.max(1.0), "{what} frame {i}: {max:.3e} (mean |v| {scale:.3e})");
        }
    }

    /// The device passes against the host's on the tiny release's VAE (skipped without an NVIDIA
    /// GPU): 9 frames encoded (three chunks), 3 latent frames decoded (9 frames).
    #[test]
    fn passes_match_host() {
        let Some(be) = backend() else { return };
        let dir = std::env::temp_dir().join(format!("oxinfer-longcat-avatar-vae-cuda-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (_, base) = super::super::super::tiny::write_tiny_release(&dir).unwrap();
        let vae = VideoVae::load(&base.join("vae"), true).unwrap();
        let e = Engine::build(be).unwrap();
        let mut seed = 5u64;
        let mut rnd = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        };
        let (h, w) = (32, 48);
        let frames: Vec<Frame> = (0..9).map(|_| (0..h * w * 3).map(|_| rnd()).collect()).collect();
        let (dev, lh, lw) = encode(&e, &vae, &frames, h, w).expect("the engine declined");
        let enc = vae.enc.as_ref().unwrap();
        let mut st = State { caches: Vec::new(), at: 0 };
        let mut host = Vec::new();
        let mut at = 0;
        while at < frames.len() {
            let n = if at == 0 { 1 } else { vae.scale_t() };
            host.extend(enc.forward(&frames[at..at + n], &mut st, h, w).0);
            at += n;
        }
        assert_eq!((lh, lw), (h / vae.scale(), w / vae.scale()));
        close("encode", &dev, &host);

        let z = vae.cfg.z_dim;
        let lat: Vec<Frame> = (0..3).map(|_| (0..lh * lw * z).map(|_| rnd()).collect()).collect();
        let dev = decode(&e, &vae, &lat, lh, lw, &mut |_, _| {}).expect("the engine declined");
        let mut st = State { caches: Vec::new(), at: 0 };
        let mut host = Vec::new();
        for f in &lat {
            host.extend(vae.dec.forward(std::slice::from_ref(f), &mut st, lh, lw).0);
        }
        close("decode", &dev, &host);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
