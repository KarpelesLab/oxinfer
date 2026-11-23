//! Inkling-2-Mini's drafter ([`crate::inkling_draft`]) on Metal: a whole draft pass in one
//! submission, the context injection of committed positions in another. The matvecs run on the
//! decode library's kernels (the target's LM head over the block's rows included); the norms,
//! RoPE, the block's attention over the device copy of the draft's key / value cache and the
//! recurrent head's chain (with the greedy choice of each slot's token on the device) are the
//! kernels of `metal_inkling_draft.metal`.
//!
//! The host's cache (`DsparkSession`) stays authoritative: every injection writes the device
//! cache and copies the new rows back; rows the host gained otherwise are uploaded before use.

use super::decode::{FastEnc, Fns, MvFmt, MvMode, MvParams, MvWeight, MvkPipe, encode_mv, run_concurrent};
use super::{Id, IntoInnerUnsafe, MetalBackend, MetalWeight, MsgNewBufLen, MsgVoid};
use crate::ggml::GgmlType;
use crate::gpu::decode::Slot;
use crate::inkling_draft::{InkDraft, InkDraftDevice, KeysValues};
use crate::model::Lin;

const SRC: &str = include_str!("metal_inkling_draft.metal");
/// Positions of one injection submission at most (the multi-token matvecs take 16).
const INJ_T: usize = 16;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MvkP {
    m: MvParams,
    ntok: u32,
    sx: u32,
    sy: u32,
    sgate: u32,
}

/// A matrix and its pipelines: one token, and the multi-token kernels of every token class up to
/// 16.
struct Mw {
    w: MvWeight,
    one: (Id, usize, usize),
    multi: MvkPipe,
}

struct LayerW {
    attn_norm: u32,
    ffn_norm: u32,
    ctx_norm: u32,
    q_norm: u32,
    k_norm: u32,
    kc_norm: u32,
    window: usize,
    q: Mw,
    k: Mw,
    v: Mw,
    o: Mw,
    kc: Mw,
    vc: Mw,
    gate: Mw,
    up: Mw,
    down: Mw,
}

struct Pipes {
    norm: Id,
    mix: Id,
    qk: Id,
    store: Id,
    attn: Id,
    jin: Id,
    gate: Id,
    logit: Id,
    argmax: Id,
}

/// Arena offsets (floats).
#[derive(Default)]
struct Bufs {
    feats: usize,
    ctx: usize,
    kx: usize,
    vx: usize,
    rnn: usize,
    x: usize,
    hn: usize,
    q: usize,
    k: usize,
    v: usize,
    att: usize,
    act: usize,
    xf: usize,
    base: usize,
    jin: usize,
    j: usize,
    state: usize,
    lat: usize,
    bias: usize,
    logits: usize,
    tok: usize,
    ctxrow: usize,
    len: usize,
}

pub(crate) struct InkDraftMetal {
    be: &'static MetalBackend,
    f: Fns,
    p: Pipes,
    arena: Id,
    base: *mut f32,
    consts: Id,
    w1: Id,
    b: Bufs,
    layers: Vec<LayerW>,
    head: Mw,
    joint: Mw,
    proj: Mw,
    out_norm: u32,
    rnn_norm: u32,
    joint_b: u32,
    cvec: u32,
    fuse: Vec<f32>,
    h: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    inter: usize,
    rank: usize,
    nb: usize,
    nt: usize,
    vp: usize,
    u: usize,
    eps: f32,
    theta: f32,
    cap: f32,
    /// Wraps shared with other plans (`MetalBackend::wrap_shared`).
    shared: Vec<Id>,
    lock: std::sync::Mutex<()>,
}
unsafe impl Send for InkDraftMetal {}
unsafe impl Sync for InkDraftMetal {}

impl Drop for InkDraftMetal {
    fn drop(&mut self) {
        let release: MsgVoid = self.be.o.msg_send();
        let last: Vec<Id> = self.shared.iter().copied().filter(|&b| self.be.unwrap_shared(b)).collect();
        for &b in [self.arena, self.consts, self.w1].iter().chain(&last) {
            if !b.is_null() {
                unsafe { release(b, self.be.sels.release) };
            }
        }
    }
}

/// A session's device copy of the draft's cache: per layer keys and values, `cap` positions.
struct DState {
    k: Vec<(Id, *mut f32)>,
    v: Vec<(Id, *mut f32)>,
    cap: usize,
    kw: usize,
    release: (MsgVoid, super::Sel),
}
unsafe impl Send for DState {}
unsafe impl Sync for DState {}
impl Drop for DState {
    fn drop(&mut self) {
        for &(b, _) in self.k.iter().chain(&self.v) {
            unsafe { (self.release.0)(b, self.release.1) };
        }
    }
}

fn new_buffer(be: &MetalBackend, bytes: usize) -> Option<(Id, *mut u8)> {
    let new_len: MsgNewBufLen = be.o.msg_send();
    let id = unsafe { new_len(be.device, be.sels.new_buffer_len, bytes.max(16), 0) };
    if id.is_null() {
        return None;
    }
    let contents: super::MsgPtr = be.o.msg_send();
    let p = unsafe { contents(id, be.sels.contents) } as *mut u8;
    unsafe { std::ptr::write_bytes(p, 0, bytes.max(16)) };
    Some((id, p))
}

fn specialized(f: MvFmt) -> bool {
    !matches!(f, MvFmt::Gguf(k) if !matches!(k, GgmlType::F16 | GgmlType::Q4_0 | GgmlType::Q8_0 | GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K))
}

fn chunk(f: MvFmt) -> usize {
    match f {
        MvFmt::Bf16 | MvFmt::Q8 | MvFmt::Gguf(GgmlType::F16) => 16,
        _ => 32,
    }
}

/// The library of the drafter's kernels, compiled once per process.
fn library(be: &MetalBackend) -> Option<Id> {
    use super::{Msg0, MsgNewLib};
    static LIB: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    let l = LIB.get_or_init(|| {
        let o = &be.o;
        let get_class: unsafe extern "C" fn(*const std::ffi::c_char) -> Id =
            unsafe { o.objc.get(c"objc_getClass") }.ok()?.into_inner_unsafe();
        let msg_str: unsafe extern "C" fn(Id, super::Sel, *const std::ffi::c_char) -> Id = o.msg_send();
        let msg0: Msg0 = o.msg_send();
        let src = std::ffi::CString::new(SRC).ok()?;
        let ns = unsafe { msg_str(get_class(c"NSString".as_ptr()), o.sel(c"stringWithUTF8String:"), src.as_ptr()) };
        let new_lib: MsgNewLib = o.msg_send();
        let mut err: Id = std::ptr::null_mut();
        let lib = unsafe {
            new_lib(
                be.device,
                o.sel(c"newLibraryWithSource:options:error:"),
                ns,
                std::ptr::null_mut(),
                &mut err,
            )
        };
        if lib.is_null() {
            if std::env::var_os("OXINFER_DEBUG").is_some() && !err.is_null() {
                eprintln!("[inkling draft kernels: {}]", unsafe {
                    o.nsstring_to_string(msg0(err, o.sel(c"localizedDescription")))
                });
            }
            return None;
        }
        Some(lib as usize)
    });
    l.map(|p| p as Id)
}

fn pipeline(be: &MetalBackend, lib: Id, name: &str) -> Option<Id> {
    use super::{Msg1, MsgNewPipe};
    let o = &be.o;
    let get_class: unsafe extern "C" fn(*const std::ffi::c_char) -> Id = unsafe { o.objc.get(c"objc_getClass") }.ok()?.into_inner_unsafe();
    let msg_str: unsafe extern "C" fn(Id, super::Sel, *const std::ffi::c_char) -> Id = o.msg_send();
    let cname = std::ffi::CString::new(name).ok()?;
    let ns = unsafe { msg_str(get_class(c"NSString".as_ptr()), o.sel(c"stringWithUTF8String:"), cname.as_ptr()) };
    let msg_fn: Msg1 = o.msg_send();
    let f = unsafe { msg_fn(lib, o.sel(c"newFunctionWithName:"), ns) };
    if f.is_null() {
        return None;
    }
    let new_pipe: MsgNewPipe = o.msg_send();
    let mut err: Id = std::ptr::null_mut();
    let p = unsafe { new_pipe(be.device, o.sel(c"newComputePipelineStateWithFunction:error:"), f, &mut err) };
    let void: MsgVoid = o.msg_send();
    unsafe { void(f, be.sels.release) };
    (!p.is_null()).then_some(p)
}

/// The drafter on the active Metal backend, with the target's LM head `head` (`[vp, H]`) and
/// the head product `proj` (`[u, rank]`) and its bias `cvec`; None when something is not for
/// these kernels (the drafter then runs on the CPU).
pub(crate) fn build(d: &InkDraft, head: &Lin, vp: usize, proj: &Lin, cvec: &[f32]) -> Option<Box<dyn InkDraftDevice>> {
    let be: &'static MetalBackend = crate::gpu::active()?.as_any()?.downcast_ref::<MetalBackend>()?;
    let r = InkDraftMetal::new(be, d, head, vp, proj, cvec);
    if r.is_none() && std::env::var_os("OXINFER_DEBUG").is_some() {
        eprintln!("[inkling draft: runs on the CPU (not for the Metal kernels)]");
    }
    r.map(|e| Box::new(e) as Box<dyn InkDraftDevice>)
}

impl InkDraftMetal {
    fn new(be: &'static MetalBackend, d: &InkDraft, head: &Lin, vp: usize, proj: &Lin, cvec: &[f32]) -> Option<Self> {
        let c = &d.cfg;
        let (h, nh, nkv, hd, inter, r) = (c.hidden, c.n_heads, c.n_kv, c.head_dim, c.inter, d.rank());
        let u = cvec.len();
        if hd != 128 || !nh.is_multiple_of(nkv) || c.block_size > 8 || !(1..=8).contains(&c.target_layers.len()) || h % 32 != 0 {
            return None;
        }
        let lib = library(be)?;
        let pp = |n: &str| pipeline(be, lib, n);
        let p = Pipes {
            norm: pp("idr_norm")?,
            mix: pp("idr_mix")?,
            qk: pp("idr_qk")?,
            store: pp("idr_store")?,
            attn: pp("idr_attn")?,
            jin: pp("idr_jin")?,
            gate: pp("idr_gate")?,
            logit: pp("idr_logit")?,
            argmax: pp("idr_argmax")?,
        };
        let wraps = std::cell::RefCell::new(Vec::<Id>::new());
        let mw = |l: &Lin, n_out: usize, n_in: usize| -> Option<Mw> {
            if l.has_lora() {
                return None;
            }
            let w = match l.as_ref() {
                // GGUF bf16 (the Markov head of llama.cpp's conversion): the bf16 kernels' rows.
                crate::gpu::WeightRef::Gguf(g) if g.kind == GgmlType::BF16 && g.block_scales.is_none() => {
                    let (id, addr) = be.wrap_shared(g.data.as_ptr(), g.data.len())?;
                    wraps.borrow_mut().push(id);
                    MvWeight {
                        fmt: MvFmt::Bf16,
                        n_out,
                        n_in,
                        data: id,
                        data_off: addr - be.buffer_gpu_address(id),
                        scales: id,
                        scales_off: 0,
                        offs: id,
                        offs_off: 0,
                        gpr: (n_in * 2) as u32,
                        gk: 0,
                    }
                }
                wr => {
                    let dev = l.device_weight(n_out, n_in)?;
                    if let Some(m) = dev.downcast_ref::<MetalWeight>() {
                        m.keep_resident(&be.residency);
                    }
                    MvWeight::new(&wr, dev, n_out, n_in)?
                }
            };
            let ks = if specialized(w.fmt) {
                (n_in / chunk(w.fmt) / 64).clamp(1, 4).next_power_of_two().min(4)
            } else {
                4
            };
            let (nr1, ks1) = if specialized(w.fmt) { (1, ks) } else { (2, 4) };
            Some(Mw {
                one: (be.mv_pipe(w.fmt, nr1, ks1)?, nr1, ks1),
                multi: be.mvk_pipes(w.fmt, ks, 16)?,
                w,
            })
        };
        let mut consts: Vec<f32> = Vec::new();
        let mut add = |v: &[f32]| -> u32 {
            let o = consts.len();
            consts.extend_from_slice(v);
            consts.resize(consts.len().next_multiple_of(64), 0.0);
            o as u32
        };
        let (qw, kw) = (nh * hd, nkv * hd);
        let mut layers = Vec::with_capacity(d.layers.len());
        for (li, l) in d.layers.iter().enumerate() {
            layers.push(LayerW {
                attn_norm: add(&l.attn_norm),
                ffn_norm: add(&l.ffn_norm),
                ctx_norm: add(&l.ctx_norm),
                q_norm: add(&l.q_norm),
                k_norm: add(&l.k_norm),
                kc_norm: add(&l.k_ctx_norm),
                window: c.windows[li],
                q: mw(&l.q, qw, h)?,
                k: mw(&l.k, kw, h)?,
                v: mw(&l.v, kw, h)?,
                o: mw(&l.o, h, qw)?,
                kc: mw(&l.k_ctx, kw, h)?,
                vc: mw(&l.v_ctx, kw, h)?,
                gate: mw(&l.gate, inter, h)?,
                up: mw(&l.up, inter, h)?,
                down: mw(&l.down, h, inter)?,
            });
            let ly = layers.last().unwrap();
            if ly.gate.w.fmt != ly.up.w.fmt {
                return None;
            }
        }
        let out_norm = add(&d.out_norm);
        let rnn_norm = add(&d.rnn_norm);
        let joint_b = add(&d.joint_b);
        let cv = add(cvec);
        let fuse = d.fuse.clone();
        let (cid, cp) = new_buffer(be, consts.len() * 4)?;
        unsafe { std::ptr::copy_nonoverlapping(consts.as_ptr(), cp as *mut f32, consts.len()) };
        // `markov_w1` as bf16 rows (the chain gathers the previous token's).
        let vrows = d.w1_rows;
        let (w1, w1p) = new_buffer(be, vrows * r * 2)?;
        {
            let w1s = unsafe { std::slice::from_raw_parts_mut(w1p as *mut u16, vrows * r) };
            let mut row = vec![0f32; r];
            for t in 0..vrows {
                d.w1.row(t, &mut row);
                for (o, &x) in w1s[t * r..(t + 1) * r].iter_mut().zip(&row) {
                    *o = half::bf16::from_f32(x).to_bits();
                }
            }
        }
        let nt = c.target_layers.len();
        let nb = c.block_size;
        let mut b = Bufs::default();
        let mut at = 0usize;
        let mut take = |n: usize| {
            let o = at;
            at += n.next_multiple_of(64);
            o
        };
        b.feats = take(INJ_T * nt * h);
        b.ctx = take(INJ_T * h);
        b.kx = take(INJ_T * kw);
        b.vx = take(INJ_T * kw);
        b.rnn = take(INJ_T * h);
        b.x = take(nb * h);
        b.hn = take(nb * h);
        b.q = take(nb * qw);
        b.k = take(nb * kw);
        b.v = take(nb * kw);
        b.att = take(nb * qw);
        b.act = take(nb * inter);
        b.xf = take(nb * h);
        b.base = take(nb * vp);
        b.jin = take(2 * r + 2 * h);
        b.j = take(3 * r);
        b.state = take(r);
        b.lat = take(r);
        b.bias = take(u);
        b.logits = take(nb * u);
        b.tok = take(nb);
        b.ctxrow = take(h);
        b.len = at;
        let (arena, ap) = new_buffer(be, b.len * 4)?;
        let e = Self {
            be,
            f: Fns::new(be),
            p,
            arena,
            base: ap as *mut f32,
            consts: cid,
            w1,
            b,
            head: mw(head, vp, h)?,
            joint: mw(&d.joint, 3 * r, 2 * r + 2 * h)?,
            proj: mw(proj, u, r)?,
            layers,
            out_norm,
            rnn_norm,
            joint_b,
            cvec: cv,
            fuse,
            h,
            nh,
            nkv,
            hd,
            inter,
            rank: r,
            nb,
            nt,
            vp,
            u,
            eps: c.eps,
            theta: c.theta as f32,
            cap: c.softcap,
            shared: wraps.take(),
            lock: std::sync::Mutex::new(()),
        };
        Some(e)
    }

    #[allow(clippy::too_many_arguments)]
    fn mv(&self, e: &FastEnc, m: &Mw, w2: Option<&Mw>, x: usize, y: usize, t: usize, mode: MvMode) {
        let a = self.arena;
        if t == 1 && w2.is_none() {
            let (pipe, nr, ks) = m.one;
            encode_mv(e, pipe, nr, ks, &m.w, None, (a, x * 4), (a, y * 4), mode, (a, 0));
            return;
        }
        let (pipe, nr, threads) = m.multi.pick(t);
        let (w, w2) = (&m.w, w2.map_or(&m.w, |u| &u.w));
        let p = MvkP {
            m: MvParams {
                n_out: w.n_out as u32,
                n_in: w.n_in as u32,
                mode: mode as u32,
                rows_per_tg: nr as u32,
                w_off: w.data_off,
                s_off: w.scales_off,
                w2_off: w2.data_off,
                s2_off: w2.scales_off,
                o_off: w.offs_off,
                o2_off: w2.offs_off,
                gpr: w.gpr,
                gk: w.gk,
                ldy: 0,
                _p: 0,
            },
            ntok: t as u32,
            sx: w.n_in as u32,
            sy: w.n_out as u32,
            sgate: 0,
        };
        e.pipe(pipe);
        e.buf(w.data, 0, 0);
        e.buf(w.scales, 0, 1);
        e.buf(a, x * 4, 2);
        e.buf(a, y * 4, 3);
        e.bytes(&p, 4);
        e.buf(w2.data, 0, 5);
        e.buf(w2.scales, 0, 6);
        e.buf(w.offs, 0, 7);
        e.buf(w2.offs, 0, 8);
        e.buf(a, 0, 9);
        e.dispatch((w.n_out.div_ceil(nr), 1), threads);
    }

    #[allow(clippy::too_many_arguments)]
    fn norm(&self, e: &FastEnc, n: usize, x: usize, out: usize, w: u32, rows: usize) {
        #[repr(C)]
        struct P {
            n: u32,
            eps: f32,
            x: u32,
            out: u32,
            w: u32,
            xs: u32,
            os: u32,
            _p: u32,
        }
        e.pipe(self.p.norm);
        e.buf(self.arena, 0, 0);
        e.buf(self.consts, 0, 1);
        e.bytes(
            &P {
                n: n as u32,
                eps: self.eps,
                x: x as u32,
                out: out as u32,
                w,
                xs: n as u32,
                os: n as u32,
                _p: 0,
            },
            2,
        );
        e.dispatch((rows, 1), 256);
    }

    fn mix(&self, e: &FastEnc, row: usize, out: usize, nw: u32, t: usize) {
        #[repr(C)]
        struct P {
            h: u32,
            nt: u32,
            f: u32,
            out: u32,
            nw: u32,
            eps: f32,
            _p0: u32,
            _p1: u32,
            w: [f32; 8],
        }
        let mut w = [0f32; 8];
        w[..self.nt].copy_from_slice(&self.fuse[row * self.nt..(row + 1) * self.nt]);
        e.pipe(self.p.mix);
        e.buf(self.arena, 0, 0);
        e.buf(self.consts, 0, 1);
        e.bytes(
            &P {
                h: self.h as u32,
                nt: self.nt as u32,
                f: self.b.feats as u32,
                out: out as u32,
                nw,
                eps: self.eps,
                _p0: 0,
                _p1: 0,
                w,
            },
            2,
        );
        e.dispatch((t, 1), 256);
    }

    /// Per-head norm and RoPE of rows `x` (`heads` per row) at `pos..`; with `store`, also into
    /// cache `kc`.
    #[allow(clippy::too_many_arguments)]
    fn qk(&self, e: &FastEnc, x: usize, heads: usize, w: u32, pos: usize, rows: usize, store: Option<Id>) {
        #[repr(C)]
        struct P {
            heads: u32,
            hd: u32,
            x: u32,
            w: u32,
            pos: u32,
            eps: f32,
            theta: f32,
            store: u32,
        }
        e.pipe(self.p.qk);
        e.buf(self.arena, 0, 0);
        e.buf(self.consts, 0, 1);
        e.bytes(
            &P {
                heads: heads as u32,
                hd: self.hd as u32,
                x: x as u32,
                w,
                pos: pos as u32,
                eps: self.eps,
                theta: self.theta,
                store: store.is_some() as u32,
            },
            2,
        );
        e.buf(store.unwrap_or(self.arena), 0, 3);
        e.dispatch((heads, rows), self.hd / 2);
    }

    fn store(&self, e: &FastEnc, x: usize, n: usize, pos: usize, rows: usize, kc: Id) {
        e.pipe(self.p.store);
        e.buf(self.arena, 0, 0);
        e.bytes(&[n as u32, x as u32, pos as u32, 0u32], 1);
        e.buf(kc, 0, 2);
        e.dispatch((n.div_ceil(256), rows), 256);
    }

    /// The device state of a session with room for `rows` positions (rows kept).
    fn state<'s>(&self, slot: &'s mut Slot, rows: usize) -> Option<&'s mut DState> {
        let kw = self.nkv * self.hd;
        let fresh = !matches!(slot.0.as_ref().and_then(|s| s.downcast_ref::<DState>()), Some(s) if s.kw == kw);
        if fresh {
            slot.0 = Some(Box::new(DState {
                k: Vec::new(),
                v: Vec::new(),
                cap: 0,
                kw,
                release: (self.be.o.msg_send(), self.be.sels.release),
            }));
        }
        let st = slot.0.as_mut()?.downcast_mut::<DState>()?;
        if st.cap < rows {
            let cap = rows.next_power_of_two().max(1024);
            let mut nk = Vec::with_capacity(self.layers.len());
            let mut nv = Vec::with_capacity(self.layers.len());
            for li in 0..self.layers.len() {
                let (k, kp) = new_buffer(self.be, cap * kw * 4)?;
                let (v, vp) = new_buffer(self.be, cap * kw * 4)?;
                if st.cap > 0 {
                    unsafe {
                        std::ptr::copy_nonoverlapping(st.k[li].1, kp as *mut f32, st.cap * kw);
                        std::ptr::copy_nonoverlapping(st.v[li].1, vp as *mut f32, st.cap * kw);
                    }
                }
                nk.push((k, kp as *mut f32));
                nv.push((v, vp as *mut f32));
            }
            for &(b, _) in st.k.iter().chain(&st.v) {
                unsafe { (st.release.0)(b, st.release.1) };
            }
            (st.k, st.v, st.cap) = (nk, nv, cap);
        }
        Some(st)
    }

    fn run(&self, encode: impl Fn(&FastEnc)) -> bool {
        let indirect: &[Id] = &[];
        run_concurrent(self.be, &self.f, std::slice::from_ref(&(0..1)), |e, _| {
            e.use_resources(indirect);
            encode(e);
        })
        .0
    }
}

impl InkDraftDevice for InkDraftMetal {
    fn upload(&self, slot: &mut Slot, from: usize, to: usize, rows: &[(&[f32], &[f32])]) -> bool {
        let _g = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        let kw = self.nkv * self.hd;
        let Some(st) = self.state(slot, to) else { return false };
        for (li, (k, v)) in rows.iter().enumerate() {
            let n = (to - from) * kw;
            if k.len() < n || v.len() < n {
                return false;
            }
            unsafe {
                std::ptr::copy_nonoverlapping(k.as_ptr(), st.k[li].1.add(from * kw), n);
                std::ptr::copy_nonoverlapping(v.as_ptr(), st.v[li].1.add(from * kw), n);
            }
        }
        true
    }

    fn context(&self, slot: &mut Slot, feats: &[f32], t: usize, pos: usize) -> Option<(Vec<KeysValues>, Vec<f32>)> {
        let _g = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        let (h, kw, nt) = (self.h, self.nkv * self.hd, self.nt);
        if feats.len() != t * nt * h {
            return None;
        }
        let st = self.state(slot, pos + t)?;
        let ids: Vec<(Id, Id)> = st.k.iter().zip(&st.v).map(|(k, v)| (k.0, v.0)).collect();
        let b = &self.b;
        let mut rnn = Vec::with_capacity(t * h);
        for t0 in (0..t).step_by(INJ_T) {
            let n = (t - t0).min(INJ_T);
            unsafe { std::ptr::copy_nonoverlapping(feats[t0 * nt * h..].as_ptr(), self.base.add(b.feats), n * nt * h) };
            let p0 = pos + t0;
            let ok = self.run(|e| {
                for (li, l) in self.layers.iter().enumerate() {
                    self.mix(e, li, b.ctx, l.ctx_norm, n);
                    e.barrier();
                    self.mv(e, &l.kc, None, b.ctx, b.kx, n, MvMode::Store);
                    self.mv(e, &l.vc, None, b.ctx, b.vx, n, MvMode::Store);
                    e.barrier();
                    self.qk(e, b.kx, self.nkv, l.kc_norm, p0, n, Some(ids[li].0));
                    self.store(e, b.vx, kw, p0, n, ids[li].1);
                    e.barrier();
                }
                self.mix(e, self.layers.len(), b.rnn, self.rnn_norm, n);
            });
            if !ok {
                return None;
            }
            rnn.extend_from_slice(unsafe { std::slice::from_raw_parts(self.base.add(b.rnn), n * h) });
        }
        let kv =
            st.k.iter()
                .zip(&st.v)
                .map(|(k, v)| unsafe {
                    (
                        std::slice::from_raw_parts(k.1.add(pos * kw), t * kw).to_vec(),
                        std::slice::from_raw_parts(v.1.add(pos * kw), t * kw).to_vec(),
                    )
                })
                .collect();
        Some((kv, rnn))
    }

    fn draft(&self, slot: &mut Slot, x0: &[f32], context: &[f32], pos: usize, anchor: u32, k: usize) -> Option<(Vec<f32>, Vec<u32>)> {
        let _g = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        let (h, nb, kw, qw, r, u) = (self.h, self.nb, self.nkv * self.hd, self.nh * self.hd, self.rank, self.u);
        if x0.len() != nb * h || context.len() != h || k == 0 || k >= nb {
            return None;
        }
        let st = self.state(slot, pos.max(1))?;
        let ids: Vec<(Id, Id)> = st.k.iter().zip(&st.v).map(|(k, v)| (k.0, v.0)).collect();
        let b = &self.b;
        unsafe {
            std::ptr::copy_nonoverlapping(x0.as_ptr(), self.base.add(b.x), nb * h);
            std::ptr::copy_nonoverlapping(context.as_ptr(), self.base.add(b.ctxrow), h);
            std::ptr::write_bytes(self.base.add(b.state), 0, r);
        }
        let a = self.arena;
        let ok = self.run(|e| {
            for (li, l) in self.layers.iter().enumerate() {
                self.norm(e, h, b.x, b.hn, l.attn_norm, nb);
                e.barrier();
                self.mv(e, &l.q, None, b.hn, b.q, nb, MvMode::Store);
                self.mv(e, &l.k, None, b.hn, b.k, nb, MvMode::Store);
                self.mv(e, &l.v, None, b.hn, b.v, nb, MvMode::Store);
                e.barrier();
                self.qk(e, b.q, self.nh, l.q_norm, pos, nb, None);
                self.qk(e, b.k, self.nkv, l.k_norm, pos, nb, None);
                e.barrier();
                let lo = match l.window {
                    0 => 0,
                    w => pos.saturating_sub(w),
                };
                #[repr(C)]
                struct Ap {
                    nh: u32,
                    nkv: u32,
                    lo: u32,
                    hi: u32,
                    nb: u32,
                    q: u32,
                    k: u32,
                    v: u32,
                    out: u32,
                    scale: f32,
                    _p0: u32,
                    _p1: u32,
                }
                e.pipe(self.p.attn);
                e.buf(a, 0, 0);
                e.bytes(
                    &Ap {
                        nh: self.nh as u32,
                        nkv: self.nkv as u32,
                        lo: lo as u32,
                        hi: pos as u32,
                        nb: nb as u32,
                        q: b.q as u32,
                        k: b.k as u32,
                        v: b.v as u32,
                        out: b.att as u32,
                        scale: (self.hd as f32).powf(-0.5),
                        _p0: 0,
                        _p1: 0,
                    },
                    1,
                );
                e.buf(ids[li].0, 0, 2);
                e.buf(ids[li].1, 0, 3);
                e.dispatch((self.nh, nb), 128);
                e.barrier();
                self.mv(e, &l.o, None, b.att, b.x, nb, MvMode::Add);
                e.barrier();
                self.norm(e, h, b.x, b.hn, l.ffn_norm, nb);
                e.barrier();
                self.mv(e, &l.gate, Some(&l.up), b.hn, b.act, nb, MvMode::SiluPair);
                e.barrier();
                self.mv(e, &l.down, None, b.act, b.x, nb, MvMode::Add);
                e.barrier();
            }
            self.norm(e, h, b.x, b.xf, self.out_norm, nb);
            e.barrier();
            // The target's head over the draft rows (1..=k), raw.
            self.mv(e, &self.head, None, b.xf + h, b.base, k, MvMode::Store);
            e.barrier();
            for i in 0..k {
                #[repr(C)]
                struct Jp {
                    r: u32,
                    h: u32,
                    state: u32,
                    hid: u32,
                    ctx: u32,
                    jin: u32,
                    tok: u32,
                    i: u32,
                    anchor: u32,
                    _p: [u32; 3],
                }
                e.pipe(self.p.jin);
                e.buf(a, 0, 0);
                e.buf(self.w1, 0, 1);
                e.bytes(
                    &Jp {
                        r: r as u32,
                        h: h as u32,
                        state: b.state as u32,
                        hid: (b.xf + (1 + i) * h) as u32,
                        ctx: b.ctxrow as u32,
                        jin: b.jin as u32,
                        tok: b.tok as u32,
                        i: i as u32,
                        anchor,
                        _p: [0; 3],
                    },
                    2,
                );
                e.dispatch(((2 * r + 2 * h).div_ceil(256), 1), 256);
                e.barrier();
                self.mv(e, &self.joint, None, b.jin, b.j, 1, MvMode::Store);
                e.barrier();
                e.pipe(self.p.gate);
                e.buf(a, 0, 0);
                e.buf(self.consts, 0, 1);
                e.bytes(&[r as u32, b.j as u32, self.joint_b, b.state as u32, b.lat as u32, 0, 0, 0], 2);
                e.dispatch((r.div_ceil(256), 1), 256);
                e.barrier();
                self.mv(e, &self.proj, None, b.lat, b.bias, 1, MvMode::Store);
                e.barrier();
                #[repr(C)]
                struct Lp {
                    u: u32,
                    base: u32,
                    bias: u32,
                    c: u32,
                    out: u32,
                    cap: f32,
                    _p: [u32; 2],
                }
                e.pipe(self.p.logit);
                e.buf(a, 0, 0);
                e.buf(self.consts, 0, 1);
                e.bytes(
                    &Lp {
                        u: u as u32,
                        base: (b.base + i * self.vp) as u32,
                        bias: b.bias as u32,
                        c: self.cvec,
                        out: (b.logits + i * u) as u32,
                        cap: self.cap,
                        _p: [0; 2],
                    },
                    2,
                );
                e.dispatch((u.div_ceil(256), 1), 256);
                e.barrier();
                e.pipe(self.p.argmax);
                e.buf(a, 0, 0);
                e.bytes(&[u as u32, (b.logits + i * u) as u32, b.tok as u32, i as u32], 1);
                e.dispatch((1, 1), 1024);
                e.barrier();
            }
        });
        if !ok {
            return None;
        }
        let _ = (kw, qw, self.inter);
        let logits = unsafe { std::slice::from_raw_parts(self.base.add(b.logits), k * u).to_vec() };
        let toks = unsafe { std::slice::from_raw_parts(self.base.add(b.tok) as *const u32, k).to_vec() };
        Some((logits, toks))
    }
}
