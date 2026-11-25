//! Whole UNet passes on Metal ([`crate::gpu::unet`]): every op of a pass for every image of the
//! batch in a few command buffers, the activations in device buffers that live as long as the
//! session. Convolutions run as shifted GEMMs on the matrix units (`metal_unet.metal`: padded
//! activations, nine taps accumulated into one tile; strided ones and narrow inputs through an
//! unfolded copy), attention as fused flash kernels over heads padded to 64 / 96 / 128 / 160, the
//! norms and the GEGLU as row kernels writing the GEMMs' half inputs. Without matrix units there
//! is no session (the per-op path runs instead).

use super::decode::{FastEnc, Fns, run_concurrent};
use super::*;
use crate::gpu::unet::{Attn, Graph, Item, Lin, Norm, Op, Session, Transformer};
use std::collections::HashMap;

pub(crate) const SRC: &str = include_str!("metal_unet.metal");

/// Ops per command buffer (the GPU starts on the first while the rest is encoded).
const OPS_PER_CB: usize = 12;
/// Pixels per group-norm statistics chunk.
const GN_CHUNK: usize = 64;
/// GEMM rows per threadgroup.
const TM: usize = 128;

/// Profiling (`OXINFER_UNET_SKIP=conv3,conv1,unfold,qkv,lin,flash,gn,prep,ln,geglu,concat`): kinds
/// of dispatches left out of the passes (the results are then wrong; the time saved is theirs).
fn skip(kind: &str) -> bool {
    static S: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let s = S.get_or_init(|| std::env::var("OXINFER_UNET_SKIP").unwrap_or_default());
    !s.is_empty() && s.split(',').any(|x| x == kind)
}

/// A device buffer that grows on demand (released with the session).
struct DBuf {
    id: Id,
    cap: usize,
}

impl DBuf {
    const fn empty() -> Self {
        DBuf {
            id: std::ptr::null_mut(),
            cap: 0,
        }
    }
}

struct Pipes {
    stats: Id,
    fin: Id,
    prep: Id,
    im2col: Id,
    gemm64: Id,
    gemm128: Id,
    ln: Id,
    geglu: Id,
    concat: Id,
    half: Id,
    amax: Id,
    amax_clear: Id,
    flash: [(usize, Id); 5],
}

/// Scratch buffers of a session.
#[derive(Clone, Copy)]
enum S {
    /// Half GEMM inputs (padded or compact rows).
    Xh = 0,
    /// Unfolded rows (strided convolutions).
    Xc,
    Part,
    Stats,
    Consts,
    /// Transformer: hidden states (f32 rows), their normalized half copy, q / k / v heads,
    /// attention output, feed-forward rows and their GEGLU.
    H,
    Hn,
    Q,
    K,
    V,
    Ah,
    F,
    Gh,
    /// Text contexts as half rows (each padded to a multiple of 64 with zeros).
    Ctx,
    /// The largest magnitude of a convolution's raw input (float bits).
    Amax,
    /// Its keys and values.
    Ck,
    Cv,
}
const NS: usize = 17;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Geo {
    h: u32,
    w: u32,
    g: u32,
    p: u32,
}
#[repr(C)]
struct Ugs {
    g: Geo,
    c: u32,
    groups: u32,
    chunk: u32,
    nchunk: u32,
}
#[repr(C)]
struct Upr {
    d: Geo,
    s: Geo,
    c: u32,
    nimg: u32,
    norm: u32,
    silu: u32,
    up: u32,
    compact: u32,
    groups: u32,
    rows: u32,
}
#[repr(C)]
struct Uic {
    s: Geo,
    c: u32,
    k: u32,
    stride: u32,
    pad: u32,
    oh: u32,
    ow: u32,
    kp: u32,
    rows: u32,
}
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
    dh: u32,
    dhp: u32,
    g: Geo,
    scaled: u32,
    _p: [u32; 3],
}
#[repr(C)]
struct Uam {
    g: Geo,
    c: u32,
    nimg: u32,
    rows: u32,
    _p: u32,
}
#[repr(C)]
struct Uln {
    c: u32,
    rows: u32,
    eps: f32,
    _p: u32,
}
#[repr(C)]
struct U4([u32; 4]);
#[repr(C)]
struct Ufa {
    tq: u32,
    tk: u32,
    qs: u32,
    ks: u32,
    heads: u32,
    dh: u32,
    o_ld: u32,
    scale: f32,
}

pub(crate) struct MetalUnet {
    be: &'static MetalBackend,
    f: Fns,
    p: Pipes,
    scratch: Vec<DBuf>,
    /// Physical buffers of the graph's tensors (assigned by liveness each pass).
    slots: Vec<DBuf>,
}

// SAFETY: the session is used by one thread at a time (`&mut self`); the Objective-C objects
// it holds are thread-safe Metal buffers and pipelines.
unsafe impl Send for MetalUnet {}

impl MetalBackend {
    /// [`crate::gpu::Backend::unet_session`].
    pub(crate) fn unet_session_impl(&'static self) -> Option<Box<dyn Session>> {
        self.mma().filter(|m| m.units)?;
        let pipe = |n: &str| self.named_pipeline(n);
        let p = Pipes {
            stats: pipe("un_gn_stats")?,
            fin: pipe("un_gn_fin")?,
            prep: pipe("un_prep")?,
            im2col: pipe("un_im2col")?,
            gemm64: pipe("un_gemm64")?,
            gemm128: pipe("un_gemm128")?,
            ln: pipe("un_ln")?,
            geglu: pipe("un_geglu")?,
            concat: pipe("un_concat")?,
            half: pipe("un_half")?,
            amax: pipe("un_amax")?,
            amax_clear: pipe("un_amax_clear")?,
            flash: [
                (64, pipe("un_flash64")?),
                (96, pipe("un_flash96")?),
                (128, pipe("un_flash128")?),
                (160, pipe("un_flash160")?),
                (512, pipe("un_flash512")?),
            ],
        };
        Some(Box::new(MetalUnet {
            be: self,
            f: Fns::new(self),
            p,
            scratch: (0..NS).map(|_| DBuf::empty()).collect(),
            slots: Vec::new(),
        }))
    }
}

impl Drop for MetalUnet {
    fn drop(&mut self) {
        let void: MsgVoid = self.be.o.msg_send();
        for b in self.scratch.iter().chain(&self.slots) {
            if !b.id.is_null() {
                unsafe { void(b.id, self.be.sels.release) };
            }
        }
    }
}

/// Makes `s` hold at least `bytes` (new buffers are zero-filled; contents are lost when one grows).
fn ensure(be: &MetalBackend, s: &mut DBuf, bytes: usize) -> bool {
    if s.cap >= bytes && !s.id.is_null() {
        return true;
    }
    let new_len: MsgNewBufLen = be.o.msg_send();
    let void: MsgVoid = be.o.msg_send();
    let cap = bytes.max(256).div_ceil(PAGE) * PAGE;
    let id = unsafe { new_len(be.device, be.sels.new_buffer_len, cap, 0) };
    if id.is_null() {
        return false;
    }
    if !s.id.is_null() {
        unsafe { void(s.id, be.sels.release) };
    }
    s.id = id;
    s.cap = cap;
    true
}

/// The padded layout of an `h x w` tensor for `nimg` images.
#[derive(Clone, Copy)]
struct Lay {
    h: usize,
    w: usize,
    g: usize,
    p: usize,
    /// Rows to allocate (guards, the images, slack for the last GEMM tile and its taps).
    rows: usize,
}

fn lay(h: usize, w: usize, nimg: usize) -> Lay {
    let g = w + 3;
    let p = (h + 2) * (w + 2);
    Lay {
        h,
        w,
        g,
        p,
        rows: 2 * g + nimg * p + TM,
    }
}

impl Lay {
    fn geo(&self) -> Geo {
        Geo {
            h: self.h as u32,
            w: self.w as u32,
            g: self.g as u32,
            p: self.p as u32,
        }
    }
}

/// Rows of a compact buffer of `n` rows (the last GEMM tile, and flash query blocks, read past).
fn compact_rows(n: usize) -> usize {
    n.next_multiple_of(TM) + 64
}

/// The flash kernel width for heads of `dh`.
fn head_pad(dh: usize) -> Option<usize> {
    if dh == 512 {
        return Some(512);
    }
    [64, 96, 128, 160].into_iter().find(|&d| d >= dh && d - dh <= dh)
}

/// The value / output chunk of the flash kernel for heads padded to `d`.
fn value_chunk(d: usize) -> usize {
    if d > 160 { 128 } else { d }
}

fn metal_w<'a>(l: &'a Lin) -> Option<&'a MetalWeight> {
    if !matches!(l.w, WeightRef::Bf16(_)) || !l.n_out.is_multiple_of(64) || !l.n_in.is_multiple_of(32) {
        return None;
    }
    l.dev.downcast_ref::<MetalWeight>()
}

/// How a convolution runs: nine (or one) shifted GEMMs over padded rows, or unfolded rows.
fn shifted(k: usize, stride: usize, pad: usize, up: bool, cin: usize, n_in: usize) -> bool {
    let _ = up;
    stride == 1 && pad == k / 2 && (k == 1 || k == 3) && cin.is_multiple_of(32) && n_in == k * k * cin
}

/// Constants of a pass (norm weights, biases) and their offsets in bytes.
struct Consts {
    v: Vec<f32>,
    at: HashMap<usize, usize>,
}

impl Consts {
    fn put(&mut self, key: *const f32, parts: &[&[f32]]) {
        let k = key as usize;
        if self.at.contains_key(&k) {
            return;
        }
        self.at.insert(k, self.v.len() * 4);
        for p in parts {
            self.v.extend_from_slice(p);
        }
        self.v.resize(self.v.len().next_multiple_of(64), 0.0);
    }
    fn norm(&mut self, n: &Norm) {
        self.put(n.gamma.as_ptr(), &[n.gamma, n.beta]);
    }
    fn get(&self, key: *const f32) -> usize {
        self.at[&(key as usize)]
    }
}

/// Scratch sizes in bytes.
#[derive(Default)]
struct Need([usize; NS]);

impl Need {
    fn at_least(&mut self, s: S, bytes: usize) {
        let v = &mut self.0[s as usize];
        *v = (*v).max(bytes);
    }
}

impl Session for MetalUnet {
    fn forward(&mut self, g: &Graph, items: &[Item], out: &mut [Vec<f32>]) -> bool {
        let be = self.be;
        let nb = items.len();
        if nb == 0 || out.len() != nb {
            return false;
        }
        let [ih, iw, ic] = g.tensors[g.input];
        if items
            .iter()
            .any(|it| it.latent.len() != ih * iw * ic || it.ctx.len() != it.ctx_len * g.ctx_dim)
        {
            return false;
        }
        let m = items[0].ctx_len;
        let cross = g.ops.iter().any(|op| matches!(op, Op::Transformer { .. }));
        if cross && (items.iter().any(|it| it.ctx_len != m) || m == 0 || !g.ctx_dim.is_multiple_of(32)) {
            return false;
        }
        let mp = m.next_multiple_of(64);
        let mut need = Need::default();
        let mut consts = Consts {
            v: Vec::new(),
            at: HashMap::new(),
        };
        // Validate every op and size the scratch buffers.
        let lin_ok = |l: &Lin, consts: &mut Consts| -> bool {
            if let Some(b) = l.bias {
                if b.len() != l.n_out {
                    return false;
                }
                consts.put(b.as_ptr(), &[b]);
            }
            metal_w(l).is_some()
        };
        for op in &g.ops {
            match op {
                Op::Conv {
                    x,
                    out,
                    w,
                    k,
                    stride,
                    pad,
                    up,
                    pre,
                    bias,
                    bias_per_image,
                    add,
                } => {
                    let [h, wd, c] = g.tensors[*x];
                    let [oh, ow, oc] = g.tensors[*out];
                    let (sh, sw) = if *up { (h * 2, wd * 2) } else { (h, wd) };
                    if !c.is_multiple_of(4) || c > 2560 || oc > w.n_out || w.n_out - oc >= 64 || !lin_ok(w, &mut consts) {
                        return false;
                    }
                    if (oh, ow) != ((sh + 2 * pad - k) / stride + 1, (sw + 2 * pad - k) / stride + 1) {
                        return false;
                    }
                    if add.is_some_and(|a| g.tensors[a] != [oh, ow, oc]) {
                        return false;
                    }
                    if let Some((n, _)) = pre {
                        if n.groups == 0 || !c.is_multiple_of(n.groups) || n.gamma.len() != c || n.beta.len() != c {
                            return false;
                        }
                        consts.norm(n);
                        need.at_least(S::Part, nb * (sh * sw).div_ceil(GN_CHUNK) * n.groups * 8);
                        need.at_least(S::Stats, nb * n.groups * 8);
                    }
                    if let Some(b) = bias {
                        if b.len() != if *bias_per_image { nb * oc } else { oc } {
                            return false;
                        }
                        consts.put(b.as_ptr(), &[b]);
                    }
                    let li = lay(sh, sw, nb);
                    need.at_least(S::Xh, li.rows * c * 2);
                    need.at_least(S::Amax, 256);
                    if !shifted(*k, *stride, *pad, *up, c, w.n_in) {
                        if w.n_in < k * k * c {
                            return false;
                        }
                        need.at_least(S::Xc, compact_rows(nb * oh * ow) * w.n_in * 2);
                    }
                }
                Op::Concat { a, b, out } => {
                    let [h, wd, ca] = g.tensors[*a];
                    let [h2, w2, cb] = g.tensors[*b];
                    if (h, wd) != (h2, w2) || g.tensors[*out] != [h, wd, ca + cb] || !ca.is_multiple_of(4) || !cb.is_multiple_of(4) {
                        return false;
                    }
                }
                Op::Attention { x, out, norm, a } => {
                    let [h, wd, c] = g.tensors[*x];
                    if g.tensors[*out] != [h, wd, c] || !c.is_multiple_of(32) || norm.groups == 0 || !c.is_multiple_of(norm.groups) {
                        return false;
                    }
                    if norm.gamma.len() != c || norm.beta.len() != c {
                        return false;
                    }
                    consts.norm(norm);
                    let dh = c / a.heads;
                    let Some(dhp) = head_pad(dh) else { return false };
                    if a.heads * dh != c || [&a.q, &a.k, &a.v, &a.o].iter().any(|l| l.n_in != c || l.n_out != c) {
                        return false;
                    }
                    for l in [&a.q, &a.k, &a.v, &a.o] {
                        if !lin_ok(l, &mut consts) {
                            return false;
                        }
                    }
                    let n = h * wd;
                    let rows = compact_rows(nb * n);
                    need.at_least(S::Part, nb * n.div_ceil(GN_CHUNK) * norm.groups * 8);
                    need.at_least(S::Stats, nb * norm.groups * 8);
                    need.at_least(S::Xh, rows * c * 2);
                    need.at_least(S::Ah, rows * c * 2);
                    for b in [S::Q, S::K, S::V] {
                        need.at_least(b, rows * a.heads * dhp * 2);
                    }
                }
                Op::Transformer { x, out, t } => {
                    let [h, wd, c] = g.tensors[*x];
                    if g.tensors[*out] != [h, wd, c] || c > 2048 || !c.is_multiple_of(32) || t.norm.groups == 0 {
                        return false;
                    }
                    if !lin_ok(&t.proj_in, &mut consts) || !lin_ok(&t.proj_out, &mut consts) {
                        return false;
                    }
                    if t.proj_in.n_in != c || t.proj_in.n_out != c || t.proj_out.n_in != c || t.proj_out.n_out != c {
                        return false;
                    }
                    consts.norm(&t.norm);
                    let n = h * wd;
                    let rows = compact_rows(nb * n);
                    need.at_least(S::Part, nb * n.div_ceil(GN_CHUNK) * t.norm.groups * 8);
                    need.at_least(S::Stats, nb * t.norm.groups * 8);
                    need.at_least(S::Xh, rows * c * 2);
                    need.at_least(S::H, rows * c * 4);
                    need.at_least(S::Hn, rows * c * 2);
                    need.at_least(S::Ah, rows * c * 2);
                    need.at_least(S::Ctx, (nb * mp + 64) * g.ctx_dim * 2);
                    for b in &t.blocks {
                        for nm in [&b.n1, &b.n2, &b.n3] {
                            if nm.gamma.len() != c || nm.beta.len() != c {
                                return false;
                            }
                            consts.norm(nm);
                        }
                        let attn_ok = |a: &Attn, kv_in: usize, consts: &mut Consts| -> Option<usize> {
                            let dh = c / a.heads;
                            if a.heads * dh != c || a.q.n_out != c || a.k.n_out != c || a.v.n_out != c || a.o.n_in != c || a.o.n_out != c {
                                return None;
                            }
                            if a.q.n_in != c || a.k.n_in != kv_in || a.v.n_in != kv_in {
                                return None;
                            }
                            for l in [&a.q, &a.k, &a.v, &a.o] {
                                if !lin_ok(l, consts) {
                                    return None;
                                }
                            }
                            Some(a.heads * head_pad(dh)?)
                        };
                        let (Some(w1), Some(w2)) = (attn_ok(&b.a1, c, &mut consts), attn_ok(&b.a2, g.ctx_dim, &mut consts)) else {
                            return false;
                        };
                        need.at_least(S::Q, rows * w1.max(w2) * 2);
                        need.at_least(S::K, rows * w1 * 2);
                        need.at_least(S::V, rows * w1 * 2);
                        need.at_least(S::Ck, (nb * mp + 64) * w2 * 2);
                        need.at_least(S::Cv, (nb * mp + 64) * w2 * 2);
                        let inner = b.ff2.n_in;
                        if b.ff1.n_in != c || b.ff1.n_out != 2 * inner || b.ff2.n_out != c || !inner.is_multiple_of(64) {
                            return false;
                        }
                        if !lin_ok(&b.ff1, &mut consts) || !lin_ok(&b.ff2, &mut consts) {
                            return false;
                        }
                        need.at_least(S::F, rows * 2 * inner * 4);
                        need.at_least(S::Gh, rows * inner * 2);
                    }
                }
            }
        }
        // Tensor slots by liveness: a tensor's buffer is free after the last op reading it.
        let nt = g.tensors.len();
        let mut last = vec![0usize; nt];
        let mut first = vec![usize::MAX; nt];
        first[g.input] = 0;
        for (i, op) in g.ops.iter().enumerate() {
            let (ins, o): (Vec<usize>, usize) = match op {
                Op::Conv { x, out, add, .. } => (std::iter::once(*x).chain(*add).collect(), *out),
                Op::Concat { a, b, out } => (vec![*a, *b], *out),
                Op::Transformer { x, out, .. } | Op::Attention { x, out, .. } => (vec![*x], *out),
            };
            for t in ins {
                last[t] = last[t].max(i);
            }
            first[o] = first[o].min(i);
            last[o] = last[o].max(i);
        }
        last[g.output] = usize::MAX;
        let bytes_of = |t: usize| {
            let [h, w, c] = g.tensors[t];
            lay(h, w, nb).rows * c * 4
        };
        let mut slot_of = vec![usize::MAX; nt];
        let mut slot_size: Vec<usize> = Vec::new();
        let mut free: Vec<usize> = Vec::new();
        let mut order: Vec<usize> = (0..nt).filter(|&t| first[t] != usize::MAX).collect();
        order.sort_by_key(|&t| first[t]);
        // Walk ops in order: allocate outputs when first written, free after the last read.
        let mut by_first: HashMap<usize, Vec<usize>> = HashMap::new();
        for &t in &order {
            by_first.entry(first[t]).or_default().push(t);
        }
        let mut by_last: HashMap<usize, Vec<usize>> = HashMap::new();
        for &t in &order {
            if last[t] != usize::MAX {
                by_last.entry(last[t]).or_default().push(t);
            }
        }
        for i in 0..g.ops.len().max(1) {
            for &t in by_first.get(&i).into_iter().flatten() {
                if slot_of[t] != usize::MAX {
                    continue;
                }
                let sz = bytes_of(t);
                // The smallest free slot that fits, else the largest (grown), else a new one.
                let pick = free
                    .iter()
                    .enumerate()
                    .filter(|&(_, &s)| slot_size[s] >= sz)
                    .min_by_key(|&(_, &s)| slot_size[s])
                    .or_else(|| free.iter().enumerate().max_by_key(|&(_, &s)| slot_size[s]))
                    .map(|(j, _)| j);
                let s = match pick {
                    Some(j) => free.swap_remove(j),
                    None => {
                        slot_size.push(0);
                        slot_size.len() - 1
                    }
                };
                slot_size[s] = slot_size[s].max(sz);
                slot_of[t] = s;
            }
            for &t in by_last.get(&i).into_iter().flatten() {
                if slot_of[t] != usize::MAX {
                    free.push(slot_of[t]);
                }
            }
        }
        if slot_of[g.output] == usize::MAX || slot_of[g.input] == usize::MAX {
            return false;
        }
        while self.slots.len() < slot_size.len() {
            self.slots.push(DBuf::empty());
        }
        for (s, &sz) in slot_size.iter().enumerate() {
            if !ensure(be, &mut self.slots[s], sz) {
                return false;
            }
        }
        if !ensure(be, &mut self.scratch[S::Consts as usize], consts.v.len().max(64) * 4) {
            return false;
        }
        for (i, &bytes) in need.0.iter().enumerate() {
            if bytes > 0 && !ensure(be, &mut self.scratch[i], bytes) {
                return false;
            }
        }
        // Inputs: the latents into the input tensor's padded rows, the contexts as half rows.
        let li = lay(ih, iw, nb);
        // SAFETY: the buffers were sized above; the device is idle (the last pass was waited for).
        unsafe {
            let cp = self.f.contents(self.scratch[S::Consts as usize].id) as *mut f32;
            par_copy(cp, consts.v.as_ptr(), consts.v.len());
            let xp = self.f.contents(self.slots[slot_of[g.input]].id) as *mut f32;
            std::ptr::write_bytes(xp, 0, li.rows * ic);
            for (b, it) in items.iter().enumerate() {
                for y in 0..ih {
                    let r = li.g + b * li.p + (y + 1) * (iw + 2) + 1;
                    std::ptr::copy_nonoverlapping(it.latent.as_ptr().add(y * iw * ic), xp.add(r * ic), iw * ic);
                }
            }
            if need.0[S::Ctx as usize] > 0 {
                let ch = self.f.contents(self.scratch[S::Ctx as usize].id) as *mut u16;
                let cd = g.ctx_dim;
                std::ptr::write_bytes(ch, 0, (nb * mp + 64) * cd);
                for (b, it) in items.iter().enumerate() {
                    let dst = std::slice::from_raw_parts_mut(ch.add(b * mp * cd), m * cd);
                    for (d, &s) in dst.iter_mut().zip(it.ctx) {
                        *d = half::f16::from_f32(s).to_bits();
                    }
                }
            }
        }
        let sc: [Id; NS] = std::array::from_fn(|i| self.scratch[i].id);
        let tid: Vec<Id> = slot_of
            .iter()
            .map(|&s| if s == usize::MAX { std::ptr::null_mut() } else { self.slots[s].id })
            .collect();
        let cx = Cx {
            p: &self.p,
            sc,
            tid,
            g,
            nb,
            m,
            mp,
            consts: &consts,
        };
        let nops = g.ops.len();
        let parts: Vec<std::ops::Range<usize>> = (0..nops.div_ceil(OPS_PER_CB).max(1))
            .map(|c| c * OPS_PER_CB..((c + 1) * OPS_PER_CB).min(nops))
            .collect();
        let (ok, _) = run_concurrent(be, &self.f, &parts, |e, range| {
            for i in range {
                cx.op(e, &g.ops[i]);
            }
        });
        if !ok {
            return false;
        }
        let [oh, ow, oc] = g.tensors[g.output];
        let lo = lay(oh, ow, nb);
        let op = self.f.contents(self.slots[slot_of[g.output]].id) as *const f32;
        for (b, o) in out.iter_mut().enumerate() {
            o.resize(oh * ow * oc, 0.0);
            for y in 0..oh {
                let r = lo.g + b * lo.p + (y + 1) * (ow + 2) + 1;
                // SAFETY: the output tensor's buffer holds lo.rows rows of oc floats.
                unsafe { std::ptr::copy_nonoverlapping(op.add(r * oc), o.as_mut_ptr().add(y * ow * oc), ow * oc) };
            }
        }
        true
    }
}

/// What encoding a pass needs.
struct Cx<'c, 'g> {
    p: &'c Pipes,
    sc: [Id; NS],
    tid: Vec<Id>,
    g: &'c Graph<'g>,
    nb: usize,
    m: usize,
    mp: usize,
    consts: &'c Consts,
}

/// The store of a GEMM.
#[derive(Clone, Copy)]
enum Out {
    /// f32 rows `ld` wide, compact (`nimg` rows per image for per-image biases).
    Rows { y: Id, ld: usize, nimg: usize },
    /// f32 padded rows of `l` from compact rows.
    ToPadded { y: Id, l: Lay },
    /// f32 padded rows of `l` from padded rows.
    Padded { y: Id, l: Lay },
    /// Half rows of heads padded to `dhp`.
    Heads { y: Id, dh: usize, dhp: usize, ld: usize },
}

impl Cx<'_, '_> {
    fn sc(&self, s: S) -> Id {
        self.sc[s as usize]
    }

    fn cst(&self) -> Id {
        self.sc(S::Consts)
    }

    /// `y = sum over taps of A[r + off] W_tap^T (+ bias) (+ add)` over output rows `r0..r_end`.
    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &self,
        e: &FastEnc,
        w: &Lin,
        a: Id,
        a_rows: usize,
        cin: usize,
        offs: &[i32],
        rows: (usize, usize),
        out: Out,
        bias: Option<(usize, usize)>,
        add: Option<Id>,
        n_real: usize,
        scaled: bool,
    ) {
        let kind = match out {
            _ if offs.len() > 1 => "conv3",
            Out::Padded { .. } => "conv1",
            Out::Heads { .. } => "qkv",
            _ if cin > 2048 || w.n_in != cin => "unfold",
            _ => "lin",
        };
        if skip(kind) {
            return;
        }
        let mw = metal_w(w).unwrap();
        let (tn, pipe) = if w.n_out.is_multiple_of(128) {
            (128, self.p.gemm128)
        } else {
            (64, self.p.gemm64)
        };
        let mut off = [0i32; 9];
        off[..offs.len()].copy_from_slice(offs);
        let (y, omode, ldy, nimg, dh, dhp, geo) = match out {
            Out::Rows { y, ld, nimg } => (y, 0, ld, nimg, 0, 0, Geo::default()),
            Out::ToPadded { y, l } => (y, 1, n_real, l.h * l.w, 0, 0, l.geo()),
            Out::Padded { y, l } => (y, 2, n_real, self.nb, 0, 0, l.geo()),
            Out::Heads { y, dh, dhp, ld } => (y, 3, ld, usize::MAX >> 1, dh, dhp, Geo::default()),
        };
        let bias = bias.or_else(|| w.bias.map(|b| (self.consts.get(b.as_ptr()), 0)));
        let p = Ug {
            n_out: n_real as u32,
            ldw: w.n_in as u32,
            cin: cin as u32,
            taps: offs.len() as u32,
            off,
            a_rows: a_rows as u32,
            r0: rows.0 as u32,
            r_end: rows.1 as u32,
            omode,
            ldy: ldy as u32,
            has_bias: bias.is_some() as u32,
            bstride: bias.map_or(0, |b| b.1) as u32,
            add: add.is_some() as u32,
            nimg: nimg.min(u32::MAX as usize) as u32,
            dh: dh as u32,
            dhp: dhp as u32,
            g: geo,
            scaled: scaled as u32,
            _p: [0; 3],
        };
        e.pipe(pipe);
        e.buf(mw.data.id, mw.data.offset_bytes, 0);
        e.buf(a, 0, 1);
        e.buf(y, 0, 2);
        e.buf(self.cst(), bias.map_or(0, |b| b.0), 3);
        e.bytes(&p, 4);
        e.buf(add.unwrap_or(y), 0, 5);
        e.buf(self.sc(S::Amax), 0, 6);
        e.dispatch((w.n_out.div_ceil(tn), (rows.1 - rows.0).div_ceil(TM)), 128);
        e.barrier();
    }

    /// Group-norm statistics of tensor buffer `x` (layout `l`, `c` channels) into `S::Stats`.
    fn gn_stats(&self, e: &FastEnc, x: Id, l: Lay, c: usize, n: &Norm) {
        if skip("gn") {
            return;
        }
        let nchunk = (l.h * l.w).div_ceil(GN_CHUNK);
        let p = Ugs {
            g: l.geo(),
            c: c as u32,
            groups: n.groups as u32,
            chunk: GN_CHUNK as u32,
            nchunk: nchunk as u32,
        };
        e.pipe(self.p.stats);
        e.buf(x, 0, 0);
        e.buf(self.sc(S::Part), 0, 1);
        e.bytes(&p, 2);
        e.dispatch((nchunk, self.nb), 256);
        e.barrier();
        e.pipe(self.p.fin);
        e.buf(self.sc(S::Part), 0, 0);
        e.buf(self.sc(S::Stats), 0, 1);
        e.bytes(&p, 2);
        e.bytes(&n.eps, 3);
        e.dispatch((n.groups.div_ceil(64), self.nb), 64);
        e.barrier();
    }

    /// Half rows of `x` (layout `src`, `c` channels) into `out`: padded rows of `dst` (or compact),
    /// normalized with `norm` (stats in `S::Stats`) first, upsampled from `src` when `dst` is
    /// twice its size.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn prep(&self, e: &FastEnc, x: Id, src: Lay, dst: Lay, c: usize, norm: Option<(&Norm, bool)>, compact: bool, out: Id) {
        self.prep_scaled(e, x, src, dst, c, norm, compact, out, false)
    }

    /// [`Cx::prep`]; without a norm and with `scaled`, scaled to half's range by the largest
    /// magnitude of `x` first (`S::Amax`, for a GEMM with `scaled`).
    #[allow(clippy::too_many_arguments)]
    fn prep_scaled(
        &self,
        e: &FastEnc,
        x: Id,
        src: Lay,
        dst: Lay,
        c: usize,
        norm: Option<(&Norm, bool)>,
        compact: bool,
        out: Id,
        scaled: bool,
    ) {
        if scaled && norm.is_none() && !skip("prep") {
            e.pipe(self.p.amax_clear);
            e.buf(self.sc(S::Amax), 0, 0);
            e.dispatch((1, 1), 32);
            e.barrier();
            let rows = src.rows;
            e.pipe(self.p.amax);
            e.buf(x, 0, 0);
            e.buf(self.sc(S::Amax), 0, 1);
            e.bytes(
                &Uam {
                    g: src.geo(),
                    c: c as u32,
                    nimg: self.nb as u32,
                    rows: rows as u32,
                    _p: 0,
                },
                2,
            );
            e.dispatch(((rows * c / 4).div_ceil(256), 1), 256);
            e.barrier();
        }
        if skip("prep") {
            return;
        }
        let rows = if compact { self.nb * dst.h * dst.w } else { dst.rows };
        let p = Upr {
            d: dst.geo(),
            s: src.geo(),
            c: c as u32,
            nimg: self.nb as u32,
            norm: if norm.is_some() {
                1
            } else if scaled {
                2
            } else {
                0
            },
            silu: norm.is_some_and(|n| n.1) as u32,
            up: (dst.h != src.h) as u32,
            compact: compact as u32,
            groups: norm.map_or(1, |n| n.0.groups) as u32,
            rows: rows as u32,
        };
        e.pipe(self.p.prep);
        e.buf(x, 0, 0);
        e.buf(out, 0, 1);
        e.buf(
            if norm.is_none() && scaled {
                self.sc(S::Amax)
            } else {
                self.sc(S::Stats)
            },
            0,
            2,
        );
        e.buf(self.cst(), norm.map_or(0, |n| self.consts.get(n.0.gamma.as_ptr())), 3);
        e.bytes(&p, 4);
        e.dispatch(((rows * c / 4).div_ceil(256), 1), 256);
        e.barrier();
    }

    /// Layer norm of `rows` f32 rows of `c` (`S::H`) into half rows (`S::Hn`).
    fn ln(&self, e: &FastEnc, rows: usize, c: usize, n: &Norm) {
        if skip("ln") {
            return;
        }
        e.pipe(self.p.ln);
        e.buf(self.sc(S::H), 0, 0);
        e.buf(self.sc(S::Hn), 0, 1);
        e.buf(self.cst(), self.consts.get(n.gamma.as_ptr()), 2);
        e.bytes(
            &Uln {
                c: c as u32,
                rows: rows as u32,
                eps: n.eps,
                _p: 0,
            },
            3,
        );
        e.dispatch((rows.div_ceil(4), 1), 128);
        e.barrier();
    }

    #[allow(clippy::too_many_arguments)]
    fn flash(&self, e: &FastEnc, a: &Attn, c: usize, q: Id, k: Id, v: Id, tq: usize, tk: usize, ks: usize) {
        if skip("flash") {
            return;
        }
        let dh = c / a.heads;
        let d = head_pad(dh).unwrap();
        let pipe = self.p.flash.iter().find(|f| f.0 == d).unwrap().1;
        e.pipe(pipe);
        e.buf(q, 0, 0);
        e.buf(k, 0, 1);
        e.buf(v, 0, 2);
        e.buf(self.sc(S::Ah), 0, 3);
        e.bytes(
            &Ufa {
                tq: tq as u32,
                tk: tk as u32,
                qs: tq as u32,
                ks: ks as u32,
                heads: a.heads as u32,
                dh: dh as u32,
                o_ld: c as u32,
                scale: 1.0 / (dh as f32).sqrt(),
            },
            4,
        );
        e.dispatch3((tq.div_ceil(64), a.heads * (d / value_chunk(d)), self.nb), 128);
        e.barrier();
    }

    fn op(&self, e: &FastEnc, op: &Op) {
        let nb = self.nb;
        match op {
            Op::Conv {
                x,
                out,
                w,
                k,
                stride,
                pad,
                up,
                pre,
                bias,
                bias_per_image,
                add,
            } => {
                let [h, wd, c] = self.g.tensors[*x];
                let [oh, ow, oc] = self.g.tensors[*out];
                let ls = lay(h, wd, nb);
                let li = if *up { lay(h * 2, wd * 2, nb) } else { ls };
                let lo = lay(oh, ow, nb);
                let (xb, yb) = (self.tid[*x], self.tid[*out]);
                if let Some((n, _)) = pre {
                    self.gn_stats(e, xb, ls, c, n);
                }
                let xh = self.sc(S::Xh);
                self.prep_scaled(e, xb, ls, li, c, pre.as_ref().map(|(n, s)| (n, *s)), false, xh, pre.is_none());
                let bias = bias
                    .as_ref()
                    .map(|b| (self.consts.get(b.as_ptr()), if *bias_per_image { oc } else { 0 }));
                let addb = add.map(|a| self.tid[a]);
                if shifted(*k, *stride, *pad, *up, c, w.n_in) {
                    let r = (li.w + 2) as i32;
                    let offs: Vec<i32> = if *k == 1 {
                        vec![0]
                    } else {
                        (0..9).map(|t| (t / 3 - 1) * r + (t % 3 - 1)).collect()
                    };
                    let out = Out::Padded { y: yb, l: lo };
                    self.gemm(
                        e,
                        w,
                        xh,
                        li.rows,
                        c,
                        &offs,
                        (lo.g, lo.g + nb * lo.p),
                        out,
                        bias,
                        addb,
                        oc,
                        pre.is_none(),
                    );
                } else {
                    let rows = nb * oh * ow;
                    let kp = w.n_in;
                    e.pipe(self.p.im2col);
                    e.buf(xh, 0, 0);
                    e.buf(self.sc(S::Xc), 0, 1);
                    e.bytes(
                        &Uic {
                            s: li.geo(),
                            c: c as u32,
                            k: *k as u32,
                            stride: *stride as u32,
                            pad: *pad as u32,
                            oh: oh as u32,
                            ow: ow as u32,
                            kp: kp as u32,
                            rows: rows as u32,
                        },
                        2,
                    );
                    e.dispatch(((rows * kp).div_ceil(256), 1), 256);
                    e.barrier();
                    let out = Out::ToPadded { y: yb, l: lo };
                    self.gemm(
                        e,
                        w,
                        self.sc(S::Xc),
                        compact_rows(rows),
                        kp,
                        &[0],
                        (0, rows),
                        out,
                        bias,
                        addb,
                        oc,
                        pre.is_none(),
                    );
                }
            }
            Op::Concat { a, b, out } => {
                let [h, wd, ca] = self.g.tensors[*a];
                let cb = self.g.tensors[*b][2];
                let rows = lay(h, wd, nb).rows;
                e.pipe(self.p.concat);
                e.buf(self.tid[*a], 0, 0);
                e.buf(self.tid[*b], 0, 1);
                e.buf(self.tid[*out], 0, 2);
                e.bytes(&U4([ca as u32, cb as u32, rows as u32, 0]), 3);
                e.dispatch(((rows * (ca + cb) / 4).div_ceil(256), 1), 256);
                e.barrier();
            }
            Op::Transformer { x, out, t } => self.transformer(e, *x, *out, t),
            Op::Attention { x, out, norm, a } => {
                let [h, wd, c] = self.g.tensors[*x];
                let l = lay(h, wd, nb);
                let (n, rows) = (h * wd, nb * h * wd);
                let crow = compact_rows(rows);
                let xb = self.tid[*x];
                self.gn_stats(e, xb, l, c, norm);
                let xh = self.sc(S::Xh);
                self.prep(e, xb, l, l, c, Some((norm, false)), true, xh);
                let dh = c / a.heads;
                let dhp = head_pad(dh).unwrap();
                let ld = a.heads * dhp;
                let (qb, kb, vb) = (self.sc(S::Q), self.sc(S::K), self.sc(S::V));
                for (lin, dst) in [(&a.q, qb), (&a.k, kb), (&a.v, vb)] {
                    let o = Out::Heads { y: dst, dh, dhp, ld };
                    self.gemm(e, lin, xh, crow, c, &[0], (0, rows), o, None, None, lin.n_out, false);
                }
                self.flash(e, a, c, qb, kb, vb, n, n, n);
                let o = Out::ToPadded { y: self.tid[*out], l };
                self.gemm(
                    e,
                    &a.o,
                    self.sc(S::Ah),
                    crow,
                    c,
                    &[0],
                    (0, rows),
                    o,
                    None,
                    Some(xb),
                    a.o.n_out,
                    false,
                );
            }
        }
    }

    fn transformer(&self, e: &FastEnc, x: usize, out: usize, t: &Transformer) {
        let nb = self.nb;
        let [h, wd, c] = self.g.tensors[x];
        let l = lay(h, wd, nb);
        let n = h * wd;
        let rows = nb * n;
        let crow = compact_rows(rows);
        let xb = self.tid[x];
        self.gn_stats(e, xb, l, c, &t.norm);
        let xh = self.sc(S::Xh);
        self.prep(e, xb, l, l, c, Some((&t.norm, false)), true, xh);
        let hb = self.sc(S::H);
        let hrows = Out::Rows { y: hb, ld: c, nimg: n };
        self.gemm(
            e,
            &t.proj_in,
            xh,
            crow,
            c,
            &[0],
            (0, rows),
            hrows,
            None,
            None,
            t.proj_in.n_out,
            false,
        );
        let (hn, qb, kb, vb, ah) = (self.sc(S::Hn), self.sc(S::Q), self.sc(S::K), self.sc(S::V), self.sc(S::Ah));
        let cd = self.g.ctx_dim;
        let crows = nb * self.mp;
        for b in &t.blocks {
            // Self-attention.
            let dh = c / b.a1.heads;
            let dhp = head_pad(dh).unwrap();
            let ld = b.a1.heads * dhp;
            self.ln(e, rows, c, &b.n1);
            for (l, dst) in [(&b.a1.q, qb), (&b.a1.k, kb), (&b.a1.v, vb)] {
                let o = Out::Heads { y: dst, dh, dhp, ld };
                self.gemm(e, l, hn, crow, c, &[0], (0, rows), o, None, None, l.n_out, false);
            }
            self.flash(e, &b.a1, c, qb, kb, vb, n, n, n);
            self.gemm(e, &b.a1.o, ah, crow, c, &[0], (0, rows), hrows, None, Some(hb), b.a1.o.n_out, false);
            // Cross-attention.
            let dh = c / b.a2.heads;
            let dhp = head_pad(dh).unwrap();
            let ld = b.a2.heads * dhp;
            self.ln(e, rows, c, &b.n2);
            let o = Out::Heads { y: qb, dh, dhp, ld };
            self.gemm(e, &b.a2.q, hn, crow, c, &[0], (0, rows), o, None, None, b.a2.q.n_out, false);
            let ctx = self.sc(S::Ctx);
            for (l, dst) in [(&b.a2.k, self.sc(S::Ck)), (&b.a2.v, self.sc(S::Cv))] {
                let o = Out::Heads { y: dst, dh, dhp, ld };
                self.gemm(e, l, ctx, crows + 64, cd, &[0], (0, crows), o, None, None, l.n_out, false);
            }
            self.flash(e, &b.a2, c, qb, self.sc(S::Ck), self.sc(S::Cv), n, self.m, self.mp);
            self.gemm(e, &b.a2.o, ah, crow, c, &[0], (0, rows), hrows, None, Some(hb), b.a2.o.n_out, false);
            // GEGLU feed-forward.
            self.ln(e, rows, c, &b.n3);
            let inner = b.ff2.n_in;
            let fb = self.sc(S::F);
            let frows = Out::Rows {
                y: fb,
                ld: 2 * inner,
                nimg: n,
            };
            self.gemm(e, &b.ff1, hn, crow, c, &[0], (0, rows), frows, None, None, b.ff1.n_out, false);
            e.pipe(self.p.geglu);
            e.buf(fb, 0, 0);
            e.buf(self.sc(S::Gh), 0, 1);
            e.bytes(&U4([inner as u32, rows as u32, 0, 0]), 2);
            e.dispatch(((rows * inner / 4).div_ceil(256), 1), 256);
            e.barrier();
            self.gemm(
                e,
                &b.ff2,
                self.sc(S::Gh),
                crow,
                inner,
                &[0],
                (0, rows),
                hrows,
                None,
                Some(hb),
                b.ff2.n_out,
                false,
            );
        }
        // proj_out, added to the input, back into padded rows.
        let n4 = (rows * c / 4) as u32;
        e.pipe(self.p.half);
        e.buf(hb, 0, 0);
        e.buf(hn, 0, 1);
        e.bytes(&n4, 2);
        e.dispatch(((n4 as usize).div_ceil(256), 1), 256);
        e.barrier();
        let o = Out::ToPadded { y: self.tid[out], l };
        self.gemm(
            e,
            &t.proj_out,
            hn,
            crow,
            c,
            &[0],
            (0, rows),
            o,
            None,
            Some(xb),
            t.proj_out.n_out,
            false,
        );
    }
}

#[cfg(test)]
#[path = "metal_unet_tests.rs"]
mod tests;
