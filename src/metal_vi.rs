//! Realtime Inworld Voice Isolator stream steps on Metal ([`crate::gpu::vi`]): the weights in one
//! device buffer (f32, rows padded for the GEMM's tiles), each session's lanes (key / value
//! caches, Mamba conv tails and scan states) in shared buffers, a step (band encoders, every block
//! of each lane, decoder heads) in one command buffer of the `vi_*` kernels of `metal_vi.metal`,
//! all f32. The Mamba states are double-buffered (a step reads one copy and writes the other,
//! kept only when the step completed), the cache bookkeeping (growth, compaction) is the CPU
//! path's, done on the host between steps.

use super::decode::{FastEnc, Fns, run_concurrent};
use super::*;
use crate::gpu::vi::{Block, Dense, Device, LaneBlock, Model, Norm, Session, StreamSpec};

pub(crate) const SRC: &str = include_str!("metal_vi.metal");

/// Profiling (`OXINFER_VI_SKIP=gemm,attn`): ops left out of the steps.
fn skip_op(op: &str) -> bool {
    static S: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    S.get_or_init(|| std::env::var("OXINFER_VI_SKIP").unwrap_or_default())
        .split(',')
        .any(|x| x == op)
}

/// Longest attention a query may take (`VI_SCMAX` in the shader).
const SCMAX: usize = 4096;

#[derive(Clone, Copy)]
struct Pipes {
    gemm: Id,
    /// `vi_gemm_k{1,4}`.
    gemm_k: [Id; 2],
    ln: Id,
    enc_in: Id,
    dec_out: Id,
    rope_kv: Id,
    attn: Id,
    conv: Id,
    tail: Id,
    scan: Id,
    copy: Id,
}

/// A shared buffer.
struct Buf {
    id: Id,
    bytes: usize,
}

fn new_buf(be: &MetalBackend, bytes: usize) -> Option<Buf> {
    let new_len: MsgNewBufLen = be.o.msg_send();
    let bytes = bytes.max(256).div_ceil(PAGE) * PAGE;
    let id = unsafe { new_len(be.device, be.sels.new_buffer_len, bytes, 0) };
    (!id.is_null()).then_some(Buf { id, bytes })
}

fn release(be: &MetalBackend, id: Id) {
    if !id.is_null() {
        let void: MsgVoid = be.o.msg_send();
        unsafe { void(id, be.sels.release) };
    }
}

fn host(f: &Fns, b: &Buf) -> *mut f32 {
    f.contents(b.id) as *mut f32
}

/// A linear layer in the blob: weights (rows padded to 32, columns to `n_in`), bias.
#[derive(Clone, Copy)]
struct DOff {
    w: usize,
    b: Option<usize>,
    n_out: usize,
    n_in: usize,
}

#[derive(Clone, Copy)]
struct NOff {
    w: usize,
    b: usize,
}

#[derive(Clone, Copy)]
struct MOff {
    in_proj: DOff,
    conv_w: usize,
    conv_b: usize,
    x_proj: DOff,
    dt_proj: DOff,
    dt_bias: usize,
    a: usize,
    d: usize,
    out_proj: DOff,
    inner: usize,
    k: usize,
    dt_rank: usize,
}

struct BiOff {
    fwd: MOff,
    bwd: Option<MOff>,
    output_proj: DOff,
    norm: NOff,
}

struct AOff {
    norm1: NOff,
    qkv: DOff,
    proj: DOff,
    norm2: NOff,
    ffn1: DOff,
    ffn2: DOff,
    excl: bool,
}

enum BOff {
    Attn(Box<AOff>, Box<AOff>),
    Mamba(Box<BiOff>, Box<BiOff>),
}

/// The blob under construction.
struct Blob(Vec<f32>);

impl Blob {
    fn put(&mut self, v: &[f32]) -> usize {
        let o = self.0.len();
        self.0.extend_from_slice(v);
        self.0.resize(self.0.len().next_multiple_of(64), 0.0);
        o
    }
    /// `w` `[n_out][n_in]` with rows padded to a multiple of 32 and columns to `n_in_p`.
    fn mat(&mut self, w: &[f32], n_out: usize, n_in: usize, n_in_p: usize) -> usize {
        let o = self.0.len();
        let rows = n_out.next_multiple_of(32);
        self.0.resize(o + rows * n_in_p, 0.0);
        for r in 0..n_out {
            self.0[o + r * n_in_p..o + r * n_in_p + n_in].copy_from_slice(&w[r * n_in..(r + 1) * n_in]);
        }
        self.0.resize(self.0.len().next_multiple_of(64), 0.0);
        o
    }
    fn dense(&mut self, d: &Dense) -> DOff {
        DOff {
            w: self.mat(&d.w, d.n_out, d.n_in, d.n_in),
            b: d.b.map(|b| self.put(b)),
            n_out: d.n_out,
            n_in: d.n_in,
        }
    }
    fn norm(&mut self, n: &Norm) -> NOff {
        NOff {
            w: self.put(n.w),
            b: self.put(n.b),
        }
    }
    fn mamba(&mut self, m: &crate::gpu::vi::Mamba) -> MOff {
        MOff {
            in_proj: self.dense(&m.in_proj),
            conv_w: self.put(m.conv_w),
            conv_b: self.put(m.conv_b),
            x_proj: self.dense(&m.x_proj),
            dt_proj: self.dense(&m.dt_proj),
            dt_bias: self.put(m.dt_bias),
            a: self.put(m.a),
            d: self.put(m.d),
            out_proj: self.dense(&m.out_proj),
            inner: m.dt_bias.len(),
            k: m.conv_w.len() / m.dt_bias.len().max(1),
            dt_rank: m.dt_proj.n_in,
        }
    }
}

/// The uploaded weights.
pub(crate) struct MetalViDevice {
    be: &'static MetalBackend,
    p: Pipes,
    w: Buf,
    bands: Buf,
    width: usize,
    heads: usize,
    nb: usize,
    bins: usize,
    window: usize,
    rope_inv: Vec<f32>,
    enc: usize,
    enc_ld: usize,
    /// The encoder's bias plus the band embedding plus each look-ahead embedding (`[bands][w]`
    /// each; one entry without look-ahead embeddings).
    enc_bias: Vec<usize>,
    blocks: Vec<BOff>,
    dec_scale: usize,
    dec_shift: usize,
    dec_norm: NOff,
    l1: usize,
    l1_b: usize,
    l2: usize,
    l2_b: usize,
    dec_ld: usize,
    /// The widest inner size of the Mambas.
    inner: usize,
    xw: usize,
}

// SAFETY: the buffers and pipelines are immutable Metal objects, usable from any thread.
unsafe impl Send for MetalViDevice {}
unsafe impl Sync for MetalViDevice {}

impl Drop for MetalViDevice {
    fn drop(&mut self) {
        release(self.be, self.w.id);
        release(self.be, self.bands.id);
    }
}

impl MetalBackend {
    /// [`crate::gpu::Backend::vi_device`].
    pub(crate) fn vi_device_impl(&'static self, m: &Model) -> Option<std::sync::Arc<dyn Device>> {
        let pipe = |n: &str| self.named_pipeline(n);
        let p = Pipes {
            gemm: pipe("vi_gemm")?,
            gemm_k: [pipe("vi_gemm_k1")?, pipe("vi_gemm_k4")?],
            ln: pipe("vi_ln")?,
            enc_in: pipe("vi_enc_in")?,
            dec_out: pipe("vi_dec_out")?,
            rope_kv: pipe("vi_rope_kv")?,
            attn: pipe("vi_attn")?,
            conv: pipe("vi_conv")?,
            tail: pipe("vi_tail")?,
            scan: pipe("vi_scan")?,
            copy: pipe("vi_copy")?,
        };
        let (w, nh, nb) = (m.width, m.heads, m.edges.len());
        if nh == 0 || !w.is_multiple_of(nh) {
            return None;
        }
        let hd = w / nh;
        if !w.is_multiple_of(32) || !hd.is_multiple_of(32) || hd > 128 || nb == 0 || nb > SCMAX {
            return None;
        }
        if m.enc.len() != nb || m.enc_norm.len() != nb || m.heads_dec.len() != nb {
            return None;
        }
        let mut inner = 0;
        let mut xw = 0;
        let mut ok = true;
        for b in &m.blocks {
            match b {
                Block::Attn(t, a) => {
                    for x in [t, a] {
                        ok &= x.qkv.n_out == 3 * w && x.proj.n_in == w && x.ffn1.n_in == w && x.ffn2.n_out == w;
                        ok &= x.ffn1.n_out.is_multiple_of(8) && x.ffn1.n_out == x.ffn2.n_in;
                    }
                }
                Block::Mamba(t, f) => {
                    ok &= t.bwd.is_none() && f.bwd.is_some();
                    for bm in [t, f] {
                        for mm in std::iter::once(&bm.fwd).chain(bm.bwd.as_ref()) {
                            let i = mm.dt_bias.len();
                            let ns = mm.a.len() / i.max(1);
                            let k = mm.conv_w.len() / i.max(1);
                            ok &= ns == 16 && (2..=9).contains(&k) && i.is_multiple_of(8) && mm.dt_proj.n_in.is_multiple_of(8);
                            ok &= mm.in_proj.n_out == 2 * i && mm.x_proj.n_out == mm.dt_proj.n_in + 32 && mm.out_proj.n_out == w;
                            inner = inner.max(i);
                            xw = xw.max(mm.x_proj.n_out);
                        }
                    }
                }
            }
        }
        if !ok {
            return None;
        }
        let mut blob = Blob(Vec::new());
        // The encoders: inputs padded to a multiple of 8, the weights [band][w][enc_ld].
        let enc_ld = m.edges.iter().map(|&(s, e)| 2 * (e - s)).max()?.next_multiple_of(8);
        let enc = blob.0.len();
        for d in &m.enc {
            blob.mat(&d.w, d.n_out, d.n_in, enc_ld);
        }
        let mut btab: Vec<u32> = Vec::with_capacity(4 * nb);
        for (b, &(s, e)) in m.edges.iter().enumerate() {
            let n = m.enc_norm[b];
            let o = blob.0.len();
            blob.0.extend_from_slice(n.w);
            blob.0.extend_from_slice(n.b);
            blob.0.resize(blob.0.len().next_multiple_of(64), 0.0);
            btab.extend_from_slice(&[s as u32, e as u32, o as u32, 0]);
        }
        let looks = m.look_emb.map(|e| e.len() / w).unwrap_or(0).max(1);
        let enc_bias = (0..looks)
            .map(|l| {
                let mut v = vec![0f32; nb * w];
                for b in 0..nb {
                    for o in 0..w {
                        let mut x = m.enc[b].b.map(|bb| bb[o]).unwrap_or(0.0) + m.band_emb[b * w + o];
                        if let Some(e) = m.look_emb {
                            x += e[l * w + o];
                        }
                        v[b * w + o] = x;
                    }
                }
                blob.put(&v)
            })
            .collect();
        let blocks = m
            .blocks
            .iter()
            .map(|b| match b {
                Block::Attn(t, a) => {
                    let mut at = |x: &crate::gpu::vi::TimeAttn| AOff {
                        norm1: blob.norm(&x.norm1),
                        qkv: blob.dense(&x.qkv),
                        proj: blob.dense(&x.proj),
                        norm2: blob.norm(&x.norm2),
                        ffn1: blob.dense(&x.ffn1),
                        ffn2: blob.dense(&x.ffn2),
                        excl: x.exclusive,
                    };
                    BOff::Attn(Box::new(at(t)), Box::new(at(a)))
                }
                Block::Mamba(t, f) => {
                    let mut bi = |x: &crate::gpu::vi::BiMamba| BiOff {
                        fwd: blob.mamba(&x.fwd),
                        bwd: x.bwd.as_ref().map(|b| blob.mamba(b)),
                        output_proj: blob.dense(&x.output_proj),
                        norm: blob.norm(&x.norm),
                    };
                    BOff::Mamba(Box::new(bi(t)), Box::new(bi(f)))
                }
            })
            .collect();
        let dec_scale = blob.put(m.dec_scale);
        let dec_shift = blob.put(m.dec_shift);
        let hn_w: Vec<f32> = m.heads_dec.iter().flat_map(|h| h.norm.w.iter().copied()).collect();
        let hn_b: Vec<f32> = m.heads_dec.iter().flat_map(|h| h.norm.b.iter().copied()).collect();
        let dec_norm = NOff {
            w: blob.put(&hn_w),
            b: blob.put(&hn_b),
        };
        let l1 = blob.0.len();
        for h in &m.heads_dec {
            if h.l1.n_out != w || h.l1.n_in != w || h.l2.n_in != w {
                return None;
            }
            blob.mat(&h.l1.w, w, w, w);
        }
        let l1b: Vec<f32> = m
            .heads_dec
            .iter()
            .flat_map(|h| (0..w).map(|o| h.l1.b.map(|b| b[o]).unwrap_or(0.0)))
            .collect();
        let l1_b = blob.put(&l1b);
        let dec_ld = m.heads_dec.iter().map(|h| h.l2.n_out).max()?.next_multiple_of(32);
        let l2 = blob.0.len();
        for h in &m.heads_dec {
            let mut pw = vec![0f32; dec_ld * w];
            pw[..h.l2.n_out * w].copy_from_slice(&h.l2.w);
            blob.mat(&pw, dec_ld, w, w);
        }
        let l2b: Vec<f32> = m
            .heads_dec
            .iter()
            .flat_map(|h| (0..dec_ld).map(|o| if o < h.l2.n_out { h.l2.b.map(|b| b[o]).unwrap_or(0.0) } else { 0.0 }))
            .collect();
        let l2_b = blob.put(&l2b);
        let wbuf = new_buf(self, blob.0.len() * 4)?;
        let f = Fns::new(self);
        // SAFETY: the buffer holds the blob.
        unsafe { par_copy(host(&f, &wbuf), blob.0.as_ptr(), blob.0.len()) };
        let bands = new_buf(self, btab.len() * 4)?;
        // SAFETY: the buffer holds the table.
        unsafe { std::ptr::copy_nonoverlapping(btab.as_ptr(), f.contents(bands.id) as *mut u32, btab.len()) };
        Some(std::sync::Arc::new(MetalViDevice {
            be: self,
            p,
            w: wbuf,
            bands,
            width: w,
            heads: nh,
            nb,
            bins: m.bins,
            window: m.window,
            rope_inv: m.rope_inv.to_vec(),
            enc,
            enc_ld,
            enc_bias,
            blocks,
            dec_scale,
            dec_shift,
            dec_norm,
            l1,
            l1_b,
            l2,
            l2_b,
            dec_ld,
            inner,
            xw,
        }))
    }
}

/// A lane's block state on the device.
enum DevBlock {
    Attn {
        k: Option<Buf>,
        v: Option<Buf>,
        cap: usize,
        end: usize,
    },
    /// Two copies of (tail, state): `cur` is the live one.
    Mamba { tail: [Buf; 2], ssm: [Buf; 2] },
}

struct Lane {
    blocks: Vec<DevBlock>,
    cur: usize,
}

/// Activation buffers (grown with the frames of a step).
#[derive(Clone, Copy)]
enum A {
    Spec = 0,
    Ein,
    F,
    X,
    H,
    Qkv,
    Q,
    O,
    Ff,
    U,
    Xp,
    Dt,
    Y,
    Yc,
    P,
    Dy,
    Raw0,
    Raw1,
}
const NA: usize = 18;

pub(crate) struct MetalViSession {
    dev: std::sync::Arc<MetalViDevice>,
    f: Fns,
    spec: StreamSpec,
    acts: Vec<Option<Buf>>,
    lanes: [Lane; 2],
    cos: Option<Buf>,
    sin: Option<Buf>,
    rope_len: usize,
}

// SAFETY: one thread at a time (`&mut self`); Metal buffers are thread-safe objects.
unsafe impl Send for MetalViSession {}

impl Drop for MetalViSession {
    fn drop(&mut self) {
        let be = self.dev.be;
        let mut ids: Vec<Id> = self.acts.iter().flatten().map(|b| b.id).collect();
        ids.extend(self.cos.iter().chain(&self.sin).map(|b| b.id));
        for l in &self.lanes {
            for b in &l.blocks {
                match b {
                    DevBlock::Attn { k, v, .. } => ids.extend(k.iter().chain(v).map(|b| b.id)),
                    DevBlock::Mamba { tail, ssm } => ids.extend(tail.iter().chain(ssm).map(|b| b.id)),
                }
            }
        }
        for id in ids {
            release(be, id);
        }
    }
}

impl Device for MetalViDevice {
    fn session(self: std::sync::Arc<Self>, spec: &StreamSpec) -> Option<Box<dyn Session>> {
        let dev = self;
        if spec.look.unwrap_or(0) >= dev.enc_bias.len() || spec.max_frames == 0 {
            return None;
        }
        let longest = if dev.window > 0 {
            dev.window.min(spec.cache_frames + spec.max_frames)
        } else {
            spec.cache_frames + spec.max_frames
        };
        if longest > SCMAX {
            return None;
        }
        let f = Fns::new(dev.be);
        let mut lanes = Vec::with_capacity(2);
        for _ in 0..2 {
            let mut blocks = Vec::new();
            for b in &dev.blocks {
                match b {
                    BOff::Attn(..) => blocks.push(DevBlock::Attn {
                        k: None,
                        v: None,
                        cap: 0,
                        end: 0,
                    }),
                    BOff::Mamba(t, _) => {
                        let m = &t.fwd;
                        let tb = dev.nb * (m.k - 1) * m.inner * 4;
                        let sb = dev.nb * m.inner * 16 * 4;
                        let tail = [new_buf(dev.be, tb)?, new_buf(dev.be, tb)?];
                        let ssm = [new_buf(dev.be, sb)?, new_buf(dev.be, sb)?];
                        for x in tail.iter().chain(&ssm) {
                            // SAFETY: zeroing the new buffer.
                            unsafe { std::ptr::write_bytes(f.contents(x.id), 0, x.bytes) };
                        }
                        blocks.push(DevBlock::Mamba { tail, ssm });
                    }
                }
            }
            lanes.push(Lane { blocks, cur: 0 });
        }
        let l1 = lanes.pop().unwrap();
        let l0 = lanes.pop().unwrap();
        Some(Box::new(MetalViSession {
            dev,
            f,
            spec: *spec,
            acts: (0..NA).map(|_| None).collect(),
            lanes: [l0, l1],
            cos: None,
            sin: None,
            rope_len: 0,
        }))
    }
}

#[repr(C)]
struct ViG {
    n_out: u32,
    n_in: u32,
    t: u32,
    ldx: u32,
    ldy: u32,
    ldr: u32,
    act: u32,
    has_bias: u32,
    gw: u32,
    gx: u32,
    gy: u32,
    gb: u32,
    gr: u32,
    has_res: u32,
    _p: [u32; 2],
}
#[repr(C)]
struct ViLn {
    n: u32,
    ldx: u32,
    ldy: u32,
    mode: u32,
    rpg: u32,
    gw: u32,
    film: u32,
    _p: u32,
}
#[repr(C)]
struct ViEnc {
    nf: u32,
    bins: u32,
    ld: u32,
    _p: u32,
}
#[repr(C)]
struct ViRope {
    nf: u32,
    w: u32,
    nh: u32,
    hd: u32,
    cap: u32,
    first: u32,
    n: u32,
    _p: u32,
}
#[repr(C)]
struct ViAtt {
    mode: u32,
    nf: u32,
    nb: u32,
    w: u32,
    nh: u32,
    hd: u32,
    cap: u32,
    first: u32,
    past: u32,
    window: u32,
    excl: u32,
    scale: f32,
}
#[repr(C)]
struct ViMb {
    inner: u32,
    k: u32,
    len: u32,
    seqs: u32,
    ss: u32,
    ps: u32,
    reverse: u32,
    has_tail: u32,
    ns: u32,
    dt_rank: u32,
    xw: u32,
    _p: u32,
}

/// A buffer and a float offset in it.
type At = (Id, usize);

/// A GEMM: `y` = act(`x` W^T + bias) (+ `res`), `t` rows per group, `groups` groups.
struct Gemm {
    w: usize,
    n_out: usize,
    n_in: usize,
    bias: Option<usize>,
    act: u32,
    x: At,
    ldx: usize,
    y: At,
    ldy: usize,
    res: Option<(At, usize)>,
    t: usize,
    groups: usize,
    /// Group strides (floats): weights, x, y, bias, res.
    g: [usize; 5],
}

/// The rotation `Rope::rotate` applies at `pos` with sign -1: `(cos, -sin)` per pair.
fn back_table(inv: &[f32], pos: usize) -> Vec<(f32, f32)> {
    inv.iter()
        .map(|&f| {
            let a = pos as f32 * f;
            let (s, c) = (a as f64).sin_cos();
            (c as f32, -(s as f32))
        })
        .collect()
}

/// A cache key rotated back by the table's slots (`Rope::rotate` with sign -1).
fn rotate_back(tab: &[(f32, f32)], v: &mut [f32]) {
    let half = tab.len();
    for (i, &(c, s)) in tab.iter().enumerate() {
        let (x1, x2) = (v[i], v[i + half]);
        v[i] = x1 * c - x2 * s;
        v[i + half] = x1 * s + x2 * c;
    }
}

impl MetalViSession {
    fn act(&mut self, a: A, floats: usize) -> Option<Id> {
        let bytes = floats * 4;
        let slot = &mut self.acts[a as usize];
        if slot.as_ref().is_none_or(|b| b.bytes < bytes) {
            if let Some(b) = slot.take() {
                release(self.dev.be, b.id);
            }
            *slot = Some(new_buf(self.dev.be, bytes)?);
        }
        Some(slot.as_ref().unwrap().id)
    }

    /// The rotary tables up to slot `n` (`Rope::rotate`'s angles).
    fn rope(&mut self, n: usize) -> bool {
        if n <= self.rope_len {
            return true;
        }
        let n = n.next_multiple_of(256);
        let half = self.dev.rope_inv.len();
        let (Some(cb), Some(sb)) = (new_buf(self.dev.be, n * half * 4), new_buf(self.dev.be, n * half * 4)) else {
            return false;
        };
        let (cp, sp) = (host(&self.f, &cb), host(&self.f, &sb));
        for pos in 0..n {
            for i in 0..half {
                let a = pos as f32 * self.dev.rope_inv[i];
                let (s, c) = (a as f64).sin_cos();
                // SAFETY: within the n * half floats of each table.
                unsafe {
                    *cp.add(pos * half + i) = c as f32;
                    *sp.add(pos * half + i) = s as f32;
                }
            }
        }
        for b in [self.cos.take(), self.sin.take()].into_iter().flatten() {
            release(self.dev.be, b.id);
        }
        self.cos = Some(cb);
        self.sin = Some(sb);
        self.rope_len = n;
        true
    }

    /// The CPU path's cache bookkeeping before `nf` new slots: compaction past the capacity,
    /// then room for them.
    fn prepare_cache(&mut self, lane: usize, nf: usize) -> bool {
        let (nb, nh, hd) = (self.dev.nb, self.dev.heads, self.dev.width / self.dev.heads);
        let (cache_frames, capacity) = (self.spec.cache_frames, self.spec.capacity);
        let be = self.dev.be;
        let f = &self.f;
        let inv = &self.dev.rope_inv;
        for b in &mut self.lanes[lane].blocks {
            let DevBlock::Attn { k, v, cap, end } = b else {
                continue;
            };
            if *end + nf > capacity {
                let keep = cache_frames.min(*end);
                let shift = *end - keep;
                if shift > 0 {
                    let (kp, vp) = (host(f, k.as_ref().unwrap()), host(f, v.as_ref().unwrap()));
                    let c = *cap;
                    let tab = back_table(inv, shift);
                    for rh in 0..nb * nh {
                        let base = rh * c * hd;
                        // SAFETY: rows of the cap-slot buffers; forward copies (dst before src).
                        unsafe {
                            std::ptr::copy(kp.add(base + shift * hd), kp.add(base), keep * hd);
                            std::ptr::copy(vp.add(base + shift * hd), vp.add(base), keep * hd);
                            for s in 0..keep {
                                rotate_back(&tab, std::slice::from_raw_parts_mut(kp.add(base + s * hd), hd));
                            }
                        }
                    }
                    *end = keep;
                }
            }
            let slots = *end + nf;
            if slots > *cap {
                // The whole capacity at once (the pages are touched as the slots fill): growing
                // by doubling would copy hundreds of megabytes inside a push.
                let nc = slots.max(capacity + self.spec.max_frames);
                let bytes = nb * nh * nc * hd * 4;
                let (Some(nk), Some(nv)) = (new_buf(be, bytes), new_buf(be, bytes)) else {
                    return false;
                };
                if *end > 0 {
                    let (ok_, ov_) = (host(f, k.as_ref().unwrap()), host(f, v.as_ref().unwrap()));
                    let (nk_, nv_) = (host(f, &nk), host(f, &nv));
                    for rh in 0..nb * nh {
                        // SAFETY: the used slots of each row, old and new strides.
                        unsafe {
                            std::ptr::copy_nonoverlapping(ok_.add(rh * *cap * hd), nk_.add(rh * nc * hd), *end * hd);
                            std::ptr::copy_nonoverlapping(ov_.add(rh * *cap * hd), nv_.add(rh * nc * hd), *end * hd);
                        }
                    }
                }
                for o in [k.take(), v.take()].into_iter().flatten() {
                    release(be, o.id);
                }
                *k = Some(nk);
                *v = Some(nv);
                *cap = nc;
            }
        }
        true
    }

    fn gemm(&self, e: &FastEnc, g: &Gemm) {
        if skip_op("gemm") {
            return;
        }
        let d = &*self.dev;
        // Few rows: the K-split kernels (16 outputs a threadgroup); many: 32 x 32 tiles.
        let (pipe, tile_o, tile_t) = match g.t {
            0..=8 => (d.p.gemm_k[0], 16, 8),
            9..=128 => (d.p.gemm_k[1], 16, 32),
            _ => (d.p.gemm, 32, 32),
        };
        e.pipe(pipe);
        e.buf(d.w.id, g.w * 4, 0);
        e.buf(g.x.0, g.x.1 * 4, 1);
        e.buf(g.y.0, g.y.1 * 4, 2);
        e.buf(d.w.id, g.bias.unwrap_or(0) * 4, 3);
        let (r, ldr) = g.res.unwrap_or((g.y, 0));
        e.buf(r.0, r.1 * 4, 4);
        e.bytes(
            &ViG {
                n_out: g.n_out as u32,
                n_in: g.n_in as u32,
                t: g.t as u32,
                ldx: g.ldx as u32,
                ldy: g.ldy as u32,
                ldr: ldr as u32,
                act: g.act,
                has_bias: g.bias.is_some() as u32,
                gw: g.g[0] as u32,
                gx: g.g[1] as u32,
                gy: g.g[2] as u32,
                gb: g.g[3] as u32,
                gr: g.g[4] as u32,
                has_res: g.res.is_some() as u32,
                _p: [0; 2],
            },
            5,
        );
        e.dispatch3((g.n_out.div_ceil(tile_o), g.t.div_ceil(tile_t), g.groups), 128);
        e.barrier();
    }

    /// A plain linear layer over `n` rows.
    #[allow(clippy::too_many_arguments)]
    fn lin(&self, e: &FastEnc, l: &DOff, x: At, ldx: usize, y: At, ldy: usize, act: u32, res: Option<(At, usize)>, n: usize) {
        self.gemm(
            e,
            &Gemm {
                w: l.w,
                n_out: l.n_out,
                n_in: l.n_in,
                bias: l.b,
                act,
                x,
                ldx,
                y,
                ldy,
                res,
                t: n,
                groups: 1,
                g: [0; 5],
            },
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn ln(&self, e: &FastEnc, nm: &NOff, x: At, ldx: usize, y: At, ldy: usize, n: usize, rows: usize, add: bool) {
        let d = &*self.dev;
        e.pipe(d.p.ln);
        e.buf(x.0, x.1 * 4, 0);
        e.buf(y.0, y.1 * 4, 1);
        e.buf(d.w.id, nm.w * 4, 2);
        e.buf(d.w.id, nm.b * 4, 3);
        e.buf(d.w.id, 0, 4);
        e.buf(d.w.id, 0, 5);
        e.bytes(
            &ViLn {
                n: n as u32,
                ldx: ldx as u32,
                ldy: ldy as u32,
                mode: add as u32,
                rpg: rows.max(1) as u32,
                gw: 0,
                film: 0,
                _p: 0,
            },
            6,
        );
        e.dispatch((rows, 1), 128);
        e.barrier();
    }

    /// One Mamba direction over `seqs` sequences of `len` tokens (strides `ss`, `ps`), its
    /// `out_proj(y) + x` into `yc` (`ldyc` wide, at column `col`).
    #[allow(clippy::too_many_arguments)]
    fn mamba(
        &self,
        e: &FastEnc,
        m: &MOff,
        ids: &[Id; NA],
        n: usize,
        seqs: usize,
        len: usize,
        ss: usize,
        ps: usize,
        reverse: bool,
        carry: Option<(&[Buf; 2], &[Buf; 2], usize)>,
        ldyc: usize,
        col: usize,
    ) {
        let d = &*self.dev;
        let w = d.width;
        let at = |a: A| (ids[a as usize], 0usize);
        let inner = m.inner;
        self.lin(e, &m.in_proj, at(A::X), w, at(A::Ff), 2 * inner, 0, None, n);
        let mp = ViMb {
            inner: inner as u32,
            k: m.k as u32,
            len: len as u32,
            seqs: seqs as u32,
            ss: ss as u32,
            ps: ps as u32,
            reverse: reverse as u32,
            has_tail: carry.is_some() as u32,
            ns: 16,
            dt_rank: m.dt_rank as u32,
            xw: m.x_proj.n_out as u32,
            _p: 0,
        };
        e.pipe(d.p.conv);
        e.buf(ids[A::Ff as usize], 0, 0);
        e.buf(ids[A::U as usize], 0, 1);
        e.buf(d.w.id, m.conv_w * 4, 2);
        e.buf(d.w.id, m.conv_b * 4, 3);
        match carry {
            Some((tail, _, cur)) => e.buf(tail[cur].id, 0, 4),
            None => e.buf(d.w.id, 0, 4),
        }
        e.bytes(&mp, 5);
        e.dispatch(((seqs * len * inner).div_ceil(256), 1), 256);
        if let Some((tail, _, cur)) = carry {
            e.pipe(d.p.tail);
            e.buf(ids[A::Ff as usize], 0, 0);
            e.buf(tail[cur].id, 0, 1);
            e.bytes(&mp, 2);
            e.buf(tail[1 - cur].id, 0, 3);
            e.dispatch(((seqs * inner).div_ceil(256), 1), 256);
        }
        e.barrier();
        let xw = m.x_proj.n_out;
        self.lin(e, &m.x_proj, at(A::U), inner, at(A::Xp), xw, 0, None, n);
        // delta = softplus(dt_proj(xp[.., ..dt_rank]) + dt_bias)
        self.gemm(
            e,
            &Gemm {
                w: m.dt_proj.w,
                n_out: m.dt_proj.n_out,
                n_in: m.dt_proj.n_in,
                bias: Some(m.dt_bias),
                act: 2,
                x: at(A::Xp),
                ldx: xw,
                y: at(A::Dt),
                ldy: inner,
                res: None,
                t: n,
                groups: 1,
                g: [0; 5],
            },
        );
        e.pipe(d.p.scan);
        e.buf(ids[A::U as usize], 0, 0);
        e.buf(ids[A::Dt as usize], 0, 1);
        e.buf(ids[A::Xp as usize], 0, 2);
        e.buf(ids[A::Ff as usize], 0, 3);
        e.buf(d.w.id, m.a * 4, 4);
        e.buf(d.w.id, m.d * 4, 5);
        e.buf(ids[A::Y as usize], 0, 6);
        match carry {
            Some((_, ssm, cur)) => {
                e.buf(ssm[cur].id, 0, 7);
                e.buf(ssm[1 - cur].id, 0, 9);
            }
            None => {
                e.buf(d.w.id, 0, 7);
                e.buf(ids[A::Y as usize], 0, 9);
            }
        }
        e.bytes(&mp, 8);
        e.dispatch(((seqs * inner).div_ceil(64), 1), 64);
        e.barrier();
        self.lin(
            e,
            &m.out_proj,
            at(A::Y),
            inner,
            (ids[A::Yc as usize], col),
            ldyc,
            0,
            Some((at(A::X), w)),
            n,
        );
    }

    /// Encodes lane `lane`'s blocks and decoder over the features in `A::F` into `raw`.
    fn encode_lane(&self, e: &FastEnc, ids: &[Id; NA], lane: usize, nf: usize, raw: Id) {
        let d = &*self.dev;
        let (w, nb, nh) = (d.width, d.nb, d.heads);
        let hd = w / nh;
        let n = nb * nf;
        let at = |a: A| (ids[a as usize], 0usize);
        e.pipe(d.p.copy);
        e.buf(ids[A::F as usize], 0, 0);
        e.buf(ids[A::X as usize], 0, 1);
        e.bytes(&((n * w) as u32), 2);
        e.dispatch(((n * w).div_ceil(256), 1), 256);
        e.barrier();
        let l = &self.lanes[lane];
        let x = at(A::X);
        for (bi, blk) in d.blocks.iter().enumerate() {
            match blk {
                BOff::Attn(t, b) => {
                    let DevBlock::Attn { k, v, cap, end } = &l.blocks[bi] else {
                        unreachable!()
                    };
                    let (kc, vc) = (k.as_ref().unwrap().id, v.as_ref().unwrap().id);
                    let first = *end;
                    // Time.
                    self.ln(e, &t.norm1, x, w, at(A::H), w, w, n, false);
                    self.lin(e, &t.qkv, at(A::H), w, at(A::Qkv), 3 * w, 0, None, n);
                    e.pipe(d.p.rope_kv);
                    e.buf(ids[A::Qkv as usize], 0, 0);
                    e.buf(ids[A::Q as usize], 0, 1);
                    e.buf(kc, 0, 2);
                    e.buf(vc, 0, 3);
                    e.buf(self.cos.as_ref().unwrap().id, 0, 4);
                    e.buf(self.sin.as_ref().unwrap().id, 0, 5);
                    e.bytes(
                        &ViRope {
                            nf: nf as u32,
                            w: w as u32,
                            nh: nh as u32,
                            hd: hd as u32,
                            cap: *cap as u32,
                            first: first as u32,
                            n: n as u32,
                            _p: 0,
                        },
                        6,
                    );
                    e.dispatch(((n * nh * hd / 2).div_ceil(256), 1), 256);
                    e.barrier();
                    let att = |mode: u32, excl: bool, out: A| {
                        if skip_op("attn") {
                            return;
                        }
                        e.pipe(d.p.attn);
                        e.buf(ids[A::Qkv as usize], 0, 0);
                        e.buf(ids[A::Q as usize], 0, 1);
                        e.buf(kc, 0, 2);
                        e.buf(vc, 0, 3);
                        e.buf(ids[out as usize], 0, 4);
                        e.bytes(
                            &ViAtt {
                                mode,
                                nf: nf as u32,
                                nb: nb as u32,
                                w: w as u32,
                                nh: nh as u32,
                                hd: hd as u32,
                                cap: *cap as u32,
                                first: first as u32,
                                past: first.min(self.spec.cache_frames) as u32,
                                window: d.window as u32,
                                excl: excl as u32,
                                scale: 1.0 / (hd as f32).sqrt(),
                            },
                            5,
                        );
                        e.dispatch((n * nh, 1), 128);
                        e.barrier();
                    };
                    att(0, t.excl, A::O);
                    self.lin(e, &t.proj, at(A::O), w, x, w, 0, Some((x, w)), n);
                    self.ln(e, &t.norm2, x, w, at(A::H), w, w, n, false);
                    let fw = t.ffn1.n_out;
                    self.lin(e, &t.ffn1, at(A::H), w, at(A::Ff), fw, 1, None, n);
                    self.lin(e, &t.ffn2, at(A::Ff), fw, x, w, 0, Some((x, w)), n);
                    // Bands.
                    self.ln(e, &b.norm1, x, w, at(A::H), w, w, n, false);
                    self.lin(e, &b.qkv, at(A::H), w, at(A::Qkv), 3 * w, 0, None, n);
                    att(1, b.excl, A::O);
                    self.lin(e, &b.proj, at(A::O), w, x, w, 0, Some((x, w)), n);
                    self.ln(e, &b.norm2, x, w, at(A::H), w, w, n, false);
                    let fw = b.ffn1.n_out;
                    self.lin(e, &b.ffn1, at(A::H), w, at(A::Ff), fw, 1, None, n);
                    self.lin(e, &b.ffn2, at(A::Ff), fw, x, w, 0, Some((x, w)), n);
                }
                BOff::Mamba(tm, fm) => {
                    let DevBlock::Mamba { tail, ssm } = &l.blocks[bi] else {
                        unreachable!()
                    };
                    // Time, causal: rows are bands, positions frames.
                    self.mamba(e, &tm.fwd, ids, n, nb, nf, nf, 1, false, Some((tail, ssm, l.cur)), w, 0);
                    self.lin(e, &tm.output_proj, at(A::Yc), w, at(A::P), w, 0, None, n);
                    self.ln(e, &tm.norm, at(A::P), w, x, w, w, n, true);
                    // Bands, both ways: rows are frames, positions bands.
                    self.mamba(e, &fm.fwd, ids, n, nf, nb, 1, nf, false, None, 2 * w, 0);
                    self.mamba(e, fm.bwd.as_ref().unwrap(), ids, n, nf, nb, 1, nf, true, None, 2 * w, w);
                    self.lin(e, &fm.output_proj, at(A::Yc), 2 * w, at(A::P), w, 0, None, n);
                    self.ln(e, &fm.norm, at(A::P), w, x, w, w, n, true);
                }
            }
        }
        // The decoder heads: FiLM + each band's LayerNorm, linear + GELU, linear, to the bins.
        e.pipe(d.p.ln);
        e.buf(ids[A::X as usize], 0, 0);
        e.buf(ids[A::H as usize], 0, 1);
        e.buf(d.w.id, d.dec_norm.w * 4, 2);
        e.buf(d.w.id, d.dec_norm.b * 4, 3);
        e.buf(d.w.id, d.dec_scale * 4, 4);
        e.buf(d.w.id, d.dec_shift * 4, 5);
        e.bytes(
            &ViLn {
                n: w as u32,
                ldx: w as u32,
                ldy: w as u32,
                mode: 0,
                rpg: nf as u32,
                gw: w as u32,
                film: 1,
                _p: 0,
            },
            6,
        );
        e.dispatch((n, 1), 128);
        e.barrier();
        self.gemm(
            e,
            &Gemm {
                w: d.l1,
                n_out: w,
                n_in: w,
                bias: Some(d.l1_b),
                act: 1,
                x: at(A::H),
                ldx: w,
                y: at(A::P),
                ldy: w,
                res: None,
                t: nf,
                groups: nb,
                g: [w * w, nf * w, nf * w, w, 0],
            },
        );
        let dl = d.dec_ld;
        self.gemm(
            e,
            &Gemm {
                w: d.l2,
                n_out: dl,
                n_in: w,
                bias: Some(d.l2_b),
                act: 0,
                x: at(A::P),
                ldx: w,
                y: at(A::Dy),
                ldy: dl,
                res: None,
                t: nf,
                groups: nb,
                g: [dl * w, nf * w, nf * dl, dl, 0],
            },
        );
        e.pipe(d.p.dec_out);
        e.buf(ids[A::Dy as usize], 0, 0);
        e.buf(raw, 0, 1);
        e.buf(d.bands.id, 0, 2);
        e.bytes(
            &ViEnc {
                nf: nf as u32,
                bins: d.bins as u32,
                ld: dl as u32,
                _p: 0,
            },
            3,
        );
        e.dispatch((n, 1), 128);
        e.barrier();
    }
}

impl Session for MetalViSession {
    fn step(&mut self, spec: &[f32], nf: usize, lanes: &[usize], raw: &mut [Vec<f32>]) -> bool {
        let dev = self.dev.clone();
        let (w, nb, bins) = (dev.width, dev.nb, dev.bins);
        if nf == 0 || lanes.is_empty() || lanes.len() > 2 || raw.len() < lanes.len() || spec.len() != nf * bins * 2 {
            return false;
        }
        if lanes.iter().any(|&l| l > 1) || (lanes.len() == 2 && lanes[0] == lanes[1]) {
            return false;
        }
        let n = nb * nf;
        // Rows of slack past the last group's for the GEMM's 32-row tiles.
        let rows = n + 64;
        let inner = dev.inner.max(w);
        let ff = dev
            .blocks
            .iter()
            .map(|b| match b {
                BOff::Attn(t, a) => t.ffn1.n_out.max(a.ffn1.n_out),
                BOff::Mamba(..) => 0,
            })
            .max()
            .unwrap_or(0)
            .max(2 * inner);
        for &l in lanes {
            if !self.prepare_cache(l, nf) {
                return false;
            }
        }
        let need_pos = self.lanes[lanes[0]]
            .blocks
            .iter()
            .chain(lanes.get(1).map(|&l| self.lanes[l].blocks.iter()).into_iter().flatten())
            .filter_map(|b| match b {
                DevBlock::Attn { end, .. } => Some(end + nf),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        if !self.rope(need_pos) {
            return false;
        }
        let sizes: [(A, usize); NA] = [
            (A::Spec, spec.len()),
            (A::Ein, (rows) * dev.enc_ld),
            (A::F, rows * w),
            (A::X, rows * w),
            (A::H, rows * w),
            (A::Qkv, rows * 3 * w),
            (A::Q, rows * w),
            (A::O, rows * w),
            (A::Ff, rows * ff),
            (A::U, rows * inner),
            (A::Xp, rows * dev.xw.max(8)),
            (A::Dt, rows * inner),
            (A::Y, rows * inner),
            (A::Yc, rows * 2 * w),
            (A::P, rows * w),
            (A::Dy, rows * dev.dec_ld),
            (A::Raw0, nf * bins * 4),
            (A::Raw1, nf * bins * 4),
        ];
        let mut ids = [std::ptr::null_mut(); NA];
        for (a, fl) in sizes {
            match self.act(a, fl) {
                Some(id) => ids[a as usize] = id,
                None => return false,
            }
        }
        // SAFETY: the spectrum buffer holds `spec.len()` floats; the device is idle.
        unsafe { std::ptr::copy_nonoverlapping(spec.as_ptr(), self.f.contents(ids[A::Spec as usize]) as *mut f32, spec.len()) };
        let look = self.spec.look.unwrap_or(0);
        let this = &*self;
        let (ok, _) = run_concurrent(dev.be, &this.f, std::slice::from_ref(&(0..1)), |e, _| {
            let d = &*this.dev;
            e.pipe(d.p.enc_in);
            e.buf(ids[A::Spec as usize], 0, 0);
            e.buf(ids[A::Ein as usize], 0, 1);
            e.buf(d.w.id, 0, 2);
            e.buf(d.bands.id, 0, 3);
            e.bytes(
                &ViEnc {
                    nf: nf as u32,
                    bins: bins as u32,
                    ld: d.enc_ld as u32,
                    _p: 0,
                },
                4,
            );
            e.dispatch((n, 1), 128);
            e.barrier();
            this.gemm(
                e,
                &Gemm {
                    w: d.enc,
                    n_out: w,
                    n_in: d.enc_ld,
                    bias: Some(d.enc_bias[look]),
                    act: 0,
                    x: (ids[A::Ein as usize], 0),
                    ldx: d.enc_ld,
                    y: (ids[A::F as usize], 0),
                    ldy: w,
                    res: None,
                    t: nf,
                    groups: nb,
                    g: [w * d.enc_ld, nf * d.enc_ld, nf * w, w, 0],
                },
            );
            for (i, &l) in lanes.iter().enumerate() {
                this.encode_lane(e, &ids, l, nf, ids[if i == 0 { A::Raw0 } else { A::Raw1 } as usize]);
            }
        });
        if !ok {
            return false;
        }
        for (i, &l) in lanes.iter().enumerate() {
            let src = self.f.contents(ids[if i == 0 { A::Raw0 } else { A::Raw1 } as usize]) as *const f32;
            raw[i].resize(nf * bins * 4, 0.0);
            // SAFETY: the decoder output holds nf * bins * 4 floats.
            unsafe { std::ptr::copy_nonoverlapping(src, raw[i].as_mut_ptr(), nf * bins * 4) };
            let lane = &mut self.lanes[l];
            lane.cur = 1 - lane.cur;
            for b in &mut lane.blocks {
                if let DevBlock::Attn { end, .. } = b {
                    *end += nf;
                }
            }
        }
        true
    }

    fn fork(&mut self, from: usize, to: usize) {
        if from == to {
            return;
        }
        let f = &self.f;
        let (a, b) = self.lanes.split_at_mut(1);
        let (src, dst) = if from == 0 { (&a[0], &mut b[0]) } else { (&b[0], &mut a[0]) };
        for (s, d) in src.blocks.iter().zip(dst.blocks.iter_mut()) {
            match (s, d) {
                (DevBlock::Mamba { tail: st, ssm: ss }, DevBlock::Mamba { tail: dt, ssm: ds }) => {
                    for (x, y) in [(&st[src.cur], &dt[dst.cur]), (&ss[src.cur], &ds[dst.cur])] {
                        // SAFETY: two buffers of one size; the device is idle.
                        unsafe { std::ptr::copy_nonoverlapping(f.contents(x.id), f.contents(y.id), x.bytes.min(y.bytes)) };
                    }
                }
                (_, DevBlock::Attn { end, .. }) => *end = 0,
                _ => {}
            }
        }
    }

    fn export(&self, lane: usize) -> Vec<LaneBlock> {
        let d = &*self.dev;
        let (nb, nh, hd) = (d.nb, d.heads, d.width / d.heads);
        let l = &self.lanes[lane];
        l.blocks
            .iter()
            .zip(&d.blocks)
            .map(|(b, o)| match b {
                DevBlock::Attn { k, v, cap, end } => {
                    let len = nb * nh * cap * hd;
                    let read = |b: &Option<Buf>| match b {
                        // SAFETY: the buffer holds the cache's len floats.
                        Some(b) => unsafe { std::slice::from_raw_parts(host(&self.f, b), len) }.to_vec(),
                        None => Vec::new(),
                    };
                    LaneBlock::Attn {
                        k: read(k),
                        v: read(v),
                        cap: *cap,
                        end: *end,
                    }
                }
                DevBlock::Mamba { tail, ssm } => {
                    let BOff::Mamba(t, _) = o else { unreachable!() };
                    let m = &t.fwd;
                    let (tl, sl) = (nb * (m.k - 1) * m.inner, nb * m.inner * 16);
                    // SAFETY: the buffers hold tl and sl floats.
                    unsafe {
                        LaneBlock::Mamba {
                            tail: std::slice::from_raw_parts(host(&self.f, &tail[l.cur]), tl).to_vec(),
                            ssm: std::slice::from_raw_parts(host(&self.f, &ssm[l.cur]), sl).to_vec(),
                        }
                    }
                }
            })
            .collect()
    }
}
