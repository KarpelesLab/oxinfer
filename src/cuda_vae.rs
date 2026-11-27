//! Convolutional VAEs on CUDA with the activations kept on the device: float32 frames
//! (`[pixels][channels]`) in device buffers, one launch per convolution or norm instead of a
//! round trip through host memory per layer (`gpu::Backend::conv2d`). A model walks its own
//! layers over a [`Pass`]: [`Pass::conv`] (a 2D convolution, or a causal 3D one reading its up
//! to three input frames where they are; the 2x nearest upsampling in front of a convolution is
//! part of its input addressing, the bias and a residual part of its output) and [`Pass::rms`]
//! (channel RMS norm, then SiLU). What has no kernel here (a mid block's attention) goes through
//! the host with [`Pass::download`] / [`Pass::upload`].
//!
//! The weights are float32 on the device (a copy of the layer's rows, made at first use), the
//! products float32 FMAs, so a pass differs from the host's by the order of the sums. The
//! buffers of a pass are handed out again by size while it runs and freed at its end (so are the
//! weights on a discrete card, whose memory a transformer wants). A pass that cannot run (out
//! of device memory, a kernel error) returns `None`: the caller's host path runs.
//!
//! Used by [`crate::longcat_avatar::vae`] (the Wan 2.1 video VAE) and [`crate::qwen_image::vae`]
//! (the same network on one frame).

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
        eprintln!("[vae cuda: {why}]");
    }
}

fn bytes_of(v: &[f32]) -> &[u8] {
    // SAFETY: plain numeric slices.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// `VConvP` of `cuda_vae.cu`.
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

pub(crate) struct Engine {
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
pub(crate) struct DFrame<'p> {
    buf: Rc<Buf<'p>>,
    off: usize,
    stride: usize,
}

impl DFrame<'_> {
    /// Half `i` of the channels of an [`Out::Halves`] output as a frame (in the same buffer).
    pub(crate) fn half(&self, i: usize) -> Self {
        DFrame {
            buf: self.buf.clone(),
            off: i * self.stride / 2,
            stride: self.stride,
        }
    }
}

/// What a convolution's output is.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Out {
    /// A frame for the next layer (zero channels up to a multiple of 32).
    Frame,
    /// Two frames' channels, each half as [`Out::Frame`].
    Halves,
    /// The convolution's channels alone.
    Dense,
}

impl Engine {
    /// The engine on the active backend when that is CUDA, unless the variable `off` is `0`.
    pub(crate) fn new(off: &str) -> Option<Engine> {
        if std::env::var(off).is_ok_and(|v| v == "0") {
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

    pub(crate) fn build(be: &CudaBackend) -> Result<Engine> {
        let x: Cuh = {
            use dlsym::Bindings;
            // SAFETY: libcuda is already initialised by the backend.
            unsafe { Cuh::open_first(crate::platform::gpu::cuda::CANDIDATES.iter().copied()) }.map_err(|e| anyhow!("libcuda: {e}"))?
        };
        let ctx = be.context().ok_or_else(|| anyhow!("no context"))?;
        let module = be.cached_module(include_str!("cuda_vae.cu"), "vae-{sm}", "cuda_vae.cu")?;
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
    pub(crate) fn pass<T>(&self, f: impl FnOnce(&mut Pass) -> Result<T>) -> Option<T> {
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

pub(crate) struct Pass<'p, 's> {
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
    pub(crate) fn upload(&self, x: &[f32], c: usize) -> Result<DFrame<'p>> {
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
    pub(crate) fn download(&self, f: &DFrame, npix: usize, c: usize) -> Result<Vec<f32>> {
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
    pub(crate) fn conv(
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

    /// `x / max(|x|, 1e-12) * g` per pixel (`g`: a gain per channel), then SiLU with `silu`.
    pub(crate) fn rms(&mut self, g: &[f32], x: &DFrame<'p>, npix: usize, silu: bool) -> Result<DFrame<'p>> {
        let gains = g;
        let c = gains.len().next_multiple_of(32);
        if x.off != 0 || x.stride != c {
            bail!("a norm over a frame of another layout");
        }
        let key = gains.as_ptr() as usize;
        let g = match self.st.gains.get(&key) {
            Some(g) => *g,
            None => {
                let mut g = gains.to_vec();
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
}
