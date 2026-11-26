//! Whole UNet passes on Metal ([`crate::gpu::unet`]): every op of a pass for every image of the
//! batch in a few command buffers, the activations in device buffers that live as long as the
//! session. Convolutions run as shifted GEMMs on the matrix units (`metal_unet.metal`: padded
//! activations, nine taps accumulated into one tile; strided ones, narrow inputs and the small
//! levels through an unfolded copy; tiles past the last full wave split, [`plan`]), attention as
//! fused flash kernels over heads padded to 64 / 96 / 128 / 160, the norms and the GEGLU as row
//! kernels writing the GEMMs' half inputs. Without matrix units there is no session (the per-op
//! path runs instead).

use super::decode::{FastEnc, Fns, run_concurrent};
use super::*;
use crate::gpu::unet::{Attn, Graph, Item, Lin, Norm, Op, Session, Transformer};
use std::collections::HashMap;

pub(crate) const SRC: &str = include_str!("metal_unet.metal");

/// Ops per command buffer (the GPU starts on the first while the rest is encoded).
const OPS_PER_CB: usize = 12;
/// Pixels per group-norm statistics chunk for images of `n` pixels: 32, or more to keep to
/// 256 chunks an image (a VAE's full-size tensors).
fn gn_chunk(n: usize) -> usize {
    n.div_ceil(256).max(32)
}

/// Bytes of the group-norm partial sums of `nb` images of `n` pixels and `c` channels.
fn gn_part_bytes(nb: usize, n: usize, c: usize) -> usize {
    nb * n.div_ceil(gn_chunk(n)) * c * 8
}
/// GEMM rows per threadgroup.
const TM: usize = 128;

/// Profiling (`OXINFER_UNET_SKIP=conv3,conv1,unfold,im2col,qkv,lin,flash,gn,prep,amax,ln,concat`): kinds
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
    /// GEMMs over 128-row and 64-row tiles (64 outputs wide).
    gemm: Id,
    gemm_m64: Id,
    /// The split tiles' sums (see [`plan`]).
    gemm_red: Id,
    /// The transformers' 1x1 GEMMs ([`Cx::lean`]), by [half out][residual][64-row tiles].
    lean: [[[Id; 2]; 2]; 2],
    /// Its f32 padded-row stores, with a residual (64-row tiles).
    lean_pad: Id,
    ln: Id,
    /// The feed-forward's first GEMM with its GEGLU.
    gemm_geglu: Id,
    concat: Id,
    half: Id,
    amax: Id,
    /// Flash attention per head width: (width, the kernel over many keys, over few).
    flash: [(usize, Id, Id); 7],
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
    /// attention output, the feed-forward's GEGLU.
    H,
    Hn,
    Q,
    K,
    V,
    Ah,
    Gh,
    /// Text contexts as half rows (each padded to a multiple of 64 with zeros).
    Ctx,
    /// The largest magnitude of a convolution's raw input (float bits), a slot per op (zeroed
    /// by the host before each pass).
    Amax,
    /// The products of split GEMM tiles (f32 blocks of 128 x 64 at most, [`plan`]).
    Sk,
}
const NS: usize = 15;

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
#[derive(Clone, Copy)]
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
    /// The cross-attention keys and values of every transformer block (keys then values, in
    /// the graph's order), as half heads: they depend on the text contexts only, so they are
    /// computed by the first pass that sees a batch of contexts and read by the passes after it
    /// (the other steps of a generation) as long as the contexts stay the same.
    kv: Vec<DBuf>,
    /// The contexts (half bits, every image of the batch, `ctx_len` and the batch size
    /// first) that `kv` holds; empty when it holds nothing valid.
    kv_key: Vec<u16>,
    /// Attention weights whose heads are padded (`head_pad`): rows of each head then zero rows
    /// up to the padded width, so the GEMM writes the flash kernels' layout with whole-tile
    /// stores. Keyed by the bf16 weights' address.
    padw: HashMap<usize, DBuf>,
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
            gemm: pipe("un_gemm")?,
            gemm_m64: pipe("un_gemm_m64")?,
            gemm_red: pipe("un_gemm_red")?,
            lean: [
                [
                    [pipe("un_gemm_lf")?, pipe("un_gemm_lf_m64")?],
                    [pipe("un_gemm_lfa")?, pipe("un_gemm_lfa_m64")?],
                ],
                [
                    [pipe("un_gemm_lh")?, pipe("un_gemm_lh_m64")?],
                    [pipe("un_gemm_lha")?, pipe("un_gemm_lha_m64")?],
                ],
            ],
            lean_pad: pipe("un_gemm_lpa_m64")?,
            ln: pipe("un_ln")?,
            gemm_geglu: pipe("un_gemm_geglu")?,
            concat: pipe("un_concat")?,
            half: pipe("un_half")?,
            amax: pipe("un_amax")?,
            flash: [
                (48, pipe("un_flash48")?, pipe("un_flash48_m16")?),
                (64, pipe("un_flash64")?, pipe("un_flash64_m16")?),
                (80, pipe("un_flash80")?, pipe("un_flash80")?),
                (96, pipe("un_flash96")?, pipe("un_flash96")?),
                (128, pipe("un_flash128")?, pipe("un_flash128")?),
                (160, pipe("un_flash160")?, pipe("un_flash160")?),
                (512, pipe("un_flash512")?, pipe("un_flash512")?),
            ],
        };
        Some(Box::new(MetalUnet {
            be: self,
            f: Fns::new(self),
            p,
            scratch: (0..NS).map(|_| DBuf::empty()).collect(),
            slots: Vec::new(),
            kv: Vec::new(),
            kv_key: Vec::new(),
            padw: HashMap::new(),
        }))
    }
}

impl Drop for MetalUnet {
    fn drop(&mut self) {
        let void: MsgVoid = self.be.o.msg_send();
        for b in self.scratch.iter().chain(&self.slots).chain(&self.kv).chain(self.padw.values()) {
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

/// Rows of a compact buffer of `n` rows (the last GEMM tile, and flash query blocks, read past:
/// the last image's last block of 128 queries ends up to 127 rows past `n`).
fn compact_rows(n: usize) -> usize {
    n.next_multiple_of(TM) + 128
}

/// Whether the flash kernel for heads padded to `d` over `tk` keys takes 4 SIMD-groups of 32
/// queries (heads of 48 or 64 over more than 256 keys), else of 16.
fn flash_wide(d: usize, tk: usize) -> bool {
    d <= 64 && tk > 256
}

/// The flash kernel width for heads of `dh`.
fn head_pad(dh: usize) -> Option<usize> {
    if dh == 512 {
        return Some(512);
    }
    // The matrix units take products over multiples of 16.
    [48, 64, 80, 96, 128, 160].into_iter().find(|&d| d >= dh && d - dh <= dh)
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

/// The address of a Lin's bf16 weights (the key of [`MetalUnet::padw`]).
fn wkey(l: &Lin) -> usize {
    match l.w {
        WeightRef::Bf16(w) => w.as_ptr() as usize,
        _ => 0,
    }
}

/// The key of a padded bias in [`Consts`] (its address plus one byte: f32 addresses are aligned).
fn pad_bias_key(b: &[f32]) -> *const f32 {
    b.as_ptr().wrapping_byte_add(1)
}

/// Rows (or biases) of `x` for `heads` heads of `dh` values padded to `dhp`: each head's
/// rows, then zero rows; `n` values a row, `rows` rows in all (at least `heads dhp`).
fn pad_heads<T: Copy + Default>(x: &[T], n: usize, heads: usize, dh: usize, dhp: usize, rows: usize) -> Vec<T> {
    let mut out = vec![T::default(); rows * n];
    for h in 0..heads {
        let (src, dst) = (h * dh * n, h * dhp * n);
        out[dst..dst + dh * n].copy_from_slice(&x[src..src + dh * n]);
    }
    out
}

/// How a convolution over an `h x w` input runs: nine (or one) shifted GEMMs over padded rows,
/// or unfolded rows. 3x3 ones over inputs whose borders would be more than 30% of the padded
/// rows (8x8: 200 rows for 128 pixels of 2 images) are unfolded too: the copy costs less than
/// the products of the border rows (`OXINFER_UNET_UNFOLD=N`: the 3x3 ones over at most N pixels
/// instead).
fn shifted(k: usize, stride: usize, pad: usize, cin: usize, n_in: usize, h: usize, w: usize) -> bool {
    static U: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    let small = match *U.get_or_init(|| std::env::var("OXINFER_UNET_UNFOLD").ok().and_then(|v| v.parse().ok())) {
        Some(n) => h * w <= n,
        None => h * w * 5 < (h + 2) * (w + 2) * 4,
    };
    stride == 1 && pad == k / 2 && (k == 1 || k == 3 && !small) && cin.is_multiple_of(32) && n_in == k * k * cin
}

/// GPU cores (IOKit's `gpu-core-count` of the AGX accelerator; `OXINFER_GPU_CORES` overrides;
/// 40 when unknown).
fn gpu_cores() -> usize {
    static C: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *C.get_or_init(|| {
        if let Some(n) = std::env::var("OXINFER_GPU_CORES").ok().and_then(|v| v.parse().ok()) {
            return n;
        }
        io_gpu_cores().filter(|&n| (1..=1024).contains(&n)).unwrap_or(40)
    })
}

fn io_gpu_cores() -> Option<usize> {
    type CfRef = *const std::ffi::c_void;
    // SAFETY: C calls into IOKit / CoreFoundation with their documented signatures; every object
    // created here is released.
    unsafe {
        let iokit = dlsym::Library::open("/System/Library/Frameworks/IOKit.framework/IOKit").ok()?;
        let cf = dlsym::Library::open("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation").ok()?;
        let matching: unsafe extern "C" fn(*const std::ffi::c_char) -> CfRef = iokit.get(c"IOServiceMatching").ok()?.into_inner();
        let get_service: unsafe extern "C" fn(u32, CfRef) -> u32 = iokit.get(c"IOServiceGetMatchingService").ok()?.into_inner();
        let search: unsafe extern "C" fn(u32, *const std::ffi::c_char, CfRef, CfRef, u32) -> CfRef =
            iokit.get(c"IORegistryEntrySearchCFProperty").ok()?.into_inner();
        let release_obj: unsafe extern "C" fn(u32) -> i32 = iokit.get(c"IOObjectRelease").ok()?.into_inner();
        let cfstr: unsafe extern "C" fn(CfRef, *const std::ffi::c_char, u32) -> CfRef =
            cf.get(c"CFStringCreateWithCString").ok()?.into_inner();
        let type_id: unsafe extern "C" fn(CfRef) -> usize = cf.get(c"CFGetTypeID").ok()?.into_inner();
        let number_type: unsafe extern "C" fn() -> usize = cf.get(c"CFNumberGetTypeID").ok()?.into_inner();
        let number_value: unsafe extern "C" fn(CfRef, isize, *mut i64) -> u8 = cf.get(c"CFNumberGetValue").ok()?.into_inner();
        let release: unsafe extern "C" fn(CfRef) = cf.get(c"CFRelease").ok()?.into_inner();
        let m = matching(c"AGXAccelerator".as_ptr());
        if m.is_null() {
            return None;
        }
        // (Consumes the matching dictionary.)
        let svc = get_service(0, m);
        if svc == 0 {
            return None;
        }
        // kCFStringEncodingUTF8
        let key = cfstr(std::ptr::null(), c"gpu-core-count".as_ptr(), 0x0800_0100);
        let mut n = None;
        if !key.is_null() {
            // kIORegistryIterateRecursively
            let v = search(svc, c"IOService".as_ptr(), key, std::ptr::null(), 1);
            if !v.is_null() {
                let mut x = 0i64;
                // kCFNumberSInt64Type
                if type_id(v) == number_type() && number_value(v, 4, &mut x) != 0 {
                    n = usize::try_from(x).ok();
                }
                release(v);
            }
            release(key);
        }
        release_obj(svc);
        n
    }
}

/// How a GEMM's tiles run ([`plan`]): `tm` rows by 64 outputs a tile, `nct` tiles across, the
/// first `full` tiles whole, the `nrem` after them in `rs` parts of `kper` inputs each.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Plan {
    tm: usize,
    nct: usize,
    full: usize,
    nrem: usize,
    rs: usize,
    kper: usize,
}

impl Plan {
    fn threadgroups(&self) -> usize {
        self.full + self.nrem * self.rs
    }
}

/// The tiles of a GEMM of `n_rows` x `k` inputs into `n_out` on a GPU that runs `conc`
/// threadgroups at once (`split`: the last tiles may be split). A tile takes about as long
/// alone as beside others, so a last, partial wave of tiles costs a whole one: the tiles past
/// the last full wave are split along their inputs, to fill it at a fraction of the time (the
/// parts then summed by un_gemm_red). Of 128-row and 64-row tiles, the shorter estimate.
fn plan(n_rows: usize, n_out: usize, k: usize, conc: usize, split: bool) -> Plan {
    let nct = n_out.div_ceil(64);
    let mut best: Option<(f64, Plan)> = None;
    for tm in [128usize, 64] {
        let tiles = n_rows.div_ceil(tm) * nct;
        let (waves, rem) = (tiles / conc, tiles % conc);
        // A wave's time in us: a 128-row tile's products at ~0.38 TFLOP/s a threadgroup; a
        // 64-row tile takes 2/3 of its time (measured: 1280 outputs over 11520 inputs, 160
        // threadgroups of 64 rows in 0.25 ms, 120 of 128 in 0.37).
        let wave = (128 * 64 * k * 2) as f64 / 0.38e6 * if tm == 128 { 1.0 } else { 0.67 };
        let mut p = Plan {
            tm,
            nct,
            full: tiles,
            nrem: 0,
            rs: 1,
            kper: k,
        };
        let mut cost = (waves + (rem > 0) as usize) as f64 * wave;
        // Parts of at least 256 inputs.
        let rs = if rem > 0 && split { (conc / rem).min(k / 256).min(16) } else { 1 };
        if rs >= 2 {
            let kper = k.div_ceil(rs).next_multiple_of(64);
            let rs = k.div_ceil(kper);
            // The reduction: a dispatch, and the parts written and read at ~400 GB/s.
            let red = 8.0 + (rem * rs * tm * 64 * 8) as f64 / 400e3;
            let c2 = waves as f64 * wave + wave / rs as f64 + red;
            if c2 < cost {
                cost = c2;
                p = Plan {
                    full: tiles - rem,
                    nrem: rem,
                    rs,
                    kper,
                    ..p
                };
            }
        }
        if best.is_none_or(|b| cost < b.0) {
            best = Some((cost, p));
        }
    }
    best.unwrap().1
}

/// Whether GEMMs may split tiles ([`plan`]; `OXINFER_UNET_SPLIT=0`: never).
fn split_ok() -> bool {
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *S.get_or_init(|| std::env::var("OXINFER_UNET_SPLIT").map_or(true, |v| v != "0"))
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
        // Bytes of each transformer block's cross-attention keys (and of its values).
        let mut kv_bytes: Vec<usize> = Vec::new();
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
                    if *pad >= *k || (oh, ow) != ((sh - 1) / stride + 1, (sw - 1) / stride + 1) {
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
                        need.at_least(S::Part, gn_part_bytes(nb, h * wd, c));
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
                    // A slot (16 bytes) per convolution for its input's largest magnitude.
                    need.at_least(S::Amax, 16 * g.ops.len());
                    need.at_least(S::Sk, 4 * gpu_cores() * 128 * 64 * 4);
                    if !shifted(*k, *stride, *pad, c, w.n_in, sh, sw) {
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
                    need.at_least(S::Part, gn_part_bytes(nb, n, c));
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
                    need.at_least(S::Part, gn_part_bytes(nb, n, c));
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
                        kv_bytes.push((nb * mp + 64) * w2 * 2);
                        let inner = b.ff2.n_in;
                        if b.ff1.n_in != c || b.ff1.n_out != 2 * inner || b.ff2.n_out != c || !inner.is_multiple_of(64) {
                            return false;
                        }
                        if !lin_ok(&b.ff1, &mut consts) || !lin_ok(&b.ff2, &mut consts) {
                            return false;
                        }
                        need.at_least(S::Gh, rows * inner * 2);
                    }
                }
            }
        }
        // Attention weights over padded heads (made by the first pass that needs them).
        for op in &g.ops {
            let attns: Vec<&Attn> = match op {
                Op::Attention { a, .. } => vec![a],
                Op::Transformer { t, .. } => t.blocks.iter().flat_map(|b| [&b.a1, &b.a2]).collect(),
                Op::Conv { .. } | Op::Concat { .. } => continue,
            };
            for a in attns {
                let dh = a.q.n_out / a.heads;
                let dhp = head_pad(dh).unwrap();
                if dh == dhp {
                    continue;
                }
                for l in [&a.q, &a.k, &a.v] {
                    if let Some(b) = l.bias {
                        consts.put(pad_bias_key(b), &[&pad_heads(b, 1, a.heads, dh, dhp, a.heads * dhp)]);
                    }
                    let key = wkey(l);
                    if self.padw.contains_key(&key) {
                        continue;
                    }
                    let WeightRef::Bf16(w) = l.w else { return false };
                    let rows = pad_heads(w, l.n_in, a.heads, dh, dhp, (a.heads * dhp).next_multiple_of(64));
                    let mut d = DBuf::empty();
                    if !ensure(be, &mut d, rows.len() * 2) {
                        return false;
                    }
                    // SAFETY: the new buffer holds rows.len() halves (shared storage, unused by the GPU yet).
                    unsafe { std::ptr::copy_nonoverlapping(rows.as_ptr(), self.f.contents(d.id) as *mut u16, rows.len()) };
                    self.padw.insert(key, d);
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
        // The contexts as half rows, and whether the keys and values of the last pass are theirs.
        let cd = g.ctx_dim;
        let mut key: Vec<u16> = Vec::new();
        if cross {
            key.reserve(2 + nb * m * cd);
            key.extend([nb as u16, m as u16]);
            for it in items {
                key.extend(it.ctx.iter().map(|&s| half::f16::from_f32(s).to_bits()));
            }
        }
        while self.kv.len() < 2 * kv_bytes.len() {
            self.kv.push(DBuf::empty());
        }
        let mut kv_cached = cross && !kv_bytes.is_empty() && key == self.kv_key;
        for (i, &bytes) in kv_bytes.iter().enumerate() {
            for b in &mut self.kv[2 * i..2 * i + 2] {
                let had = b.cap >= bytes && !b.id.is_null();
                if !ensure(be, b, bytes) {
                    self.kv_key.clear();
                    return false;
                }
                kv_cached &= had;
            }
        }
        // Until this pass completes, nothing valid (it rewrites them when not cached).
        if !kv_cached {
            self.kv_key.clear();
        }
        // Inputs: the latents into the input tensor's padded rows, the contexts as half rows.
        let li = lay(ih, iw, nb);
        // SAFETY: the buffers were sized above; the device is idle (the last pass was waited for).
        unsafe {
            if need.0[S::Amax as usize] > 0 {
                std::ptr::write_bytes(self.f.contents(self.scratch[S::Amax as usize].id), 0, need.0[S::Amax as usize]);
            }
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
            if need.0[S::Ctx as usize] > 0 && !kv_cached {
                let ch = self.f.contents(self.scratch[S::Ctx as usize].id) as *mut u16;
                std::ptr::write_bytes(ch, 0, (nb * mp + 64) * cd);
                for b in 0..nb {
                    std::ptr::copy_nonoverlapping(key.as_ptr().add(2 + b * m * cd), ch.add(b * mp * cd), m * cd);
                }
            }
        }
        let sc: [Id; NS] = std::array::from_fn(|i| self.scratch[i].id);
        let tid: Vec<Id> = slot_of
            .iter()
            .map(|&s| if s == usize::MAX { std::ptr::null_mut() } else { self.slots[s].id })
            .collect();
        // The first cross-attention block of each op (its keys and values at kv[2 i], kv[2 i + 1]).
        let mut kv_base = Vec::with_capacity(g.ops.len());
        let mut nblk = 0;
        for op in &g.ops {
            kv_base.push(nblk);
            if let Op::Transformer { t, .. } = op {
                nblk += t.blocks.len();
            }
        }
        let cx = Cx {
            p: &self.p,
            sc,
            tid,
            g,
            nb,
            m,
            mp,
            consts: &consts,
            nobar: std::cell::Cell::new(false),
            amax_n: std::cell::Cell::new(0),
            amax_off: std::cell::Cell::new(0),
            kv: self.kv.iter().map(|b| b.id).collect(),
            kv_cached,
            padw: self.padw.iter().map(|(&k, b)| (k, b.id)).collect(),
        };
        let nops = g.ops.len();
        let parts: Vec<std::ops::Range<usize>> = (0..nops.div_ceil(OPS_PER_CB).max(1))
            .map(|c| c * OPS_PER_CB..((c + 1) * OPS_PER_CB).min(nops))
            .collect();
        let (ok, gpu_s) = run_concurrent(be, &self.f, &parts, |e, range| {
            for i in range {
                cx.op(e, &g.ops[i], kv_base[i]);
            }
        });
        if !ok {
            return false;
        }
        if gpu_s > 0.0 && std::env::var_os("OXINFER_UNET_TIME").is_some() {
            // Profiling (with OXINFER_PF_PROFILE, which times the command buffers): the pass's
            // GPU span, from the first command buffer's start to the last one's end.
            eprintln!("[unet pass: {nb} images, {} ops, GPU {:.2} ms]", g.ops.len(), gpu_s * 1e3);
        }
        if cross && !kv_cached && !skip("qkv") {
            self.kv_key = key;
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
    /// The next GEMM needs no barrier after it (independent GEMMs reading the same input).
    nobar: std::cell::Cell<bool>,
    /// `S::Amax` slots taken so far, and the byte offset of the last one (read by the
    /// convolution's GEMM).
    amax_n: std::cell::Cell<usize>,
    amax_off: std::cell::Cell<usize>,
    /// Every cross-attention block's keys and values ([`MetalUnet::kv`]), and whether they
    /// already hold this batch's (else the pass computes them).
    kv: Vec<Id>,
    kv_cached: bool,
    /// [`MetalUnet::padw`]'s buffers.
    padw: HashMap<usize, Id>,
}

/// A GEMM's weights: the buffer and byte offset of bf16 rows of `ldw`, `n_out` of them, and the
/// bias's offset in the constants.
#[derive(Clone, Copy)]
struct Wt {
    id: Id,
    off: usize,
    ldw: usize,
    n_out: usize,
    bias: Option<usize>,
}

/// The store of a GEMM ([`Cx::lean`] writes the compact rows).
#[derive(Clone, Copy)]
enum Out {
    /// f32 padded rows of `l` from compact rows.
    ToPadded { y: Id, l: Lay },
    /// f32 padded rows of `l` from padded rows.
    Padded { y: Id, l: Lay },
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
            _ if cin > 2048 || w.n_in != cin => "unfold",
            _ => "lin",
        };
        if skip(kind) {
            return;
        }
        let mw = metal_w(w).unwrap();
        let n_rows = rows.1 - rows.0;
        // Convolutions may split tiles (the scratch for their parts is sized by them), unless
        // independent GEMMs run beside this one (they would share it).
        let split = split_ok() && matches!(kind, "conv3" | "conv1" | "unfold") && !self.nobar.get();
        let pl = plan(n_rows, w.n_out, offs.len() * cin, 4 * gpu_cores(), split);
        let pipe = if pl.tm == 128 { self.p.gemm } else { self.p.gemm_m64 };
        let mut off = [0i32; 9];
        off[..offs.len()].copy_from_slice(offs);
        let (y, omode, ldy, nimg, geo) = match out {
            Out::ToPadded { y, l } => (y, 1, n_real, l.h * l.w, l.geo()),
            Out::Padded { y, l } => (y, 2, n_real, self.nb, l.geo()),
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
            g: geo,
            scaled: scaled as u32,
            full: pl.full as u32,
            rs: pl.rs as u32,
            kper: pl.kper as u32,
            nct: pl.nct as u32,
            tm: pl.tm as u32,
            nrem: pl.nrem as u32,
            _p: 0,
        };
        e.pipe(pipe);
        e.buf(mw.data.id, mw.data.offset_bytes, 0);
        e.buf(a, 0, 1);
        e.buf(y, 0, 2);
        e.buf(self.cst(), bias.map_or(0, |b| b.0), 3);
        e.bytes(&p, 4);
        e.buf(add.unwrap_or(y), 0, 5);
        e.buf(self.sc(S::Amax), self.amax_off.get(), 6);
        e.buf(self.sc(S::Sk), 0, 7);
        e.dispatch((pl.threadgroups(), 1), 128);
        if pl.nrem > 0 {
            e.barrier();
            e.pipe(self.p.gemm_red);
            e.buf(y, 0, 2);
            e.buf(self.cst(), bias.map_or(0, |b| b.0), 3);
            e.bytes(&p, 4);
            e.buf(add.unwrap_or(y), 0, 5);
            e.buf(self.sc(S::Amax), self.amax_off.get(), 6);
            e.buf(self.sc(S::Sk), 0, 7);
            e.dispatch(((pl.nrem * pl.tm * 64).div_ceil(256), 1), 256);
        }
        if !self.nobar.replace(false) {
            e.barrier();
        }
    }

    /// The weights of `l`; with `heads`, its outputs as heads padded for the flash kernels
    /// (`head_pad`: [`MetalUnet::padw`] when they need padding).
    fn wt(&self, l: &Lin, heads: usize) -> Wt {
        let mw = metal_w(l).unwrap();
        let bias = l.bias.map(|b| self.consts.get(b.as_ptr()));
        let w = Wt {
            id: mw.data.id,
            off: mw.data.offset_bytes,
            ldw: l.n_in,
            n_out: l.n_out,
            bias,
        };
        if heads == 0 {
            return w;
        }
        let dh = l.n_out / heads;
        let dhp = head_pad(dh).unwrap();
        if dh == dhp {
            return w;
        }
        Wt {
            id: self.padw[&wkey(l)],
            off: 0,
            n_out: heads * dhp,
            bias: l.bias.map(|b| self.consts.get(pad_bias_key(b))),
            ..w
        }
    }

    /// `y_i = A W_i^T (+ bias_i) (+ add)` for each `(W_i, y_i)` of `ws` (at most three, as many
    /// outputs each), one dispatch: rows `0..rows` of half rows of `cin` in `a` (`a_rows`
    /// allocated), into rows `ldy` wide of half (`half`) or f32. `add`: f32 rows `ldy` wide.
    #[allow(clippy::too_many_arguments)]
    fn lean(
        &self,
        e: &FastEnc,
        kind: &str,
        ws: &[(Wt, Id)],
        a: Id,
        a_rows: usize,
        cin: usize,
        rows: usize,
        half: bool,
        add: Option<Id>,
        ldy: usize,
    ) {
        self.lean_to(e, kind, ws, a, a_rows, cin, rows, half, add, ldy, None)
    }

    /// [`Cx::lean`]; with `pad`, into the f32 padded rows of that layout (`add` there too), its
    /// width a multiple of 64.
    #[allow(clippy::too_many_arguments)]
    fn lean_to(
        &self,
        e: &FastEnc,
        kind: &str,
        ws: &[(Wt, Id)],
        a: Id,
        a_rows: usize,
        cin: usize,
        rows: usize,
        half: bool,
        add: Option<Id>,
        ldy: usize,
        pad: Option<Lay>,
    ) {
        if skip(kind) {
            return;
        }
        let w0 = ws[0].0;
        debug_assert!(ws.len() <= 3 && ws.iter().all(|w| w.0.n_out == w0.n_out && w.0.bias.is_some() == w0.bias.is_some()));
        let nct = w0.n_out.div_ceil(64);
        // 64-row tiles over the short products (measured faster up to 1280 inputs).
        let tm = if cin <= 1280 || pad.is_some() {
            64
        } else {
            plan(rows, w0.n_out * ws.len(), cin, 4 * gpu_cores(), false).tm
        };
        let p = Ug {
            n_out: w0.n_out as u32,
            ldw: w0.ldw as u32,
            cin: cin as u32,
            taps: 1,
            off: [0; 9],
            a_rows: a_rows as u32,
            r0: 0,
            r_end: rows as u32,
            omode: 0,
            ldy: ldy as u32,
            has_bias: w0.bias.is_some() as u32,
            bstride: 0,
            add: add.is_some() as u32,
            nimg: 0,
            g: pad.map_or(Geo::default(), |l| l.geo()),
            scaled: 0,
            full: 0,
            rs: 1,
            kper: cin as u32,
            nct: nct as u32,
            tm: tm as u32,
            nrem: 0,
            _p: 0,
        };
        e.pipe(match pad {
            Some(_) => {
                debug_assert!(!half && add.is_some() && pad.is_some_and(|l| l.w.is_multiple_of(64)));
                self.p.lean_pad
            }
            None => self.p.lean[half as usize][add.is_some() as usize][(tm == 64) as usize],
        });
        let at = |i: usize| ws[i.min(ws.len() - 1)];
        for (i, (wslot, yslot, bslot)) in [(0, 2, 3), (8, 10, 12), (9, 11, 13)].into_iter().enumerate() {
            let (w, y) = at(i);
            e.buf(w.id, w.off, wslot);
            e.buf(y, 0, yslot);
            e.buf(self.cst(), w.bias.unwrap_or(0), bslot);
        }
        e.buf(a, 0, 1);
        e.bytes(&p, 4);
        e.buf(add.unwrap_or(ws[0].1), 0, 5);
        e.dispatch((nct * rows.div_ceil(tm), ws.len()), 128);
        if !self.nobar.replace(false) {
            e.barrier();
        }
    }

    /// `y = A W^T (+ bias) + add` into the f32 padded rows of `l` (`add` in that layout) from
    /// `rows` compact half rows of `cin` in `a` (an attention's output projection, proj_out).
    #[allow(clippy::too_many_arguments)]
    fn out_proj(&self, e: &FastEnc, w: &Lin, a: Id, cin: usize, rows: usize, y: Id, l: Lay, add: Id) {
        let a_rows = compact_rows(rows);
        if l.w.is_multiple_of(64) && cin <= 2048 {
            let ws = [(self.wt(w, 0), y)];
            return self.lean_to(e, "lin", &ws, a, a_rows, cin, rows, false, Some(add), w.n_out, Some(l));
        }
        let o = Out::ToPadded { y, l };
        self.gemm(e, w, a, a_rows, cin, &[0], (0, rows), o, None, Some(add), w.n_out, false);
    }

    /// `S::Gh = value * gelu(gate)` (half rows of `inner`) of `[value | gate] = A W^T + bias`
    /// over `rows` compact half rows of `cin` in `a` (`a_rows` allocated).
    #[allow(clippy::too_many_arguments)]
    fn gemm_geglu(&self, e: &FastEnc, w: &Lin, a: Id, a_rows: usize, cin: usize, rows: usize, inner: usize) {
        if skip("lin") {
            return;
        }
        let mw = metal_w(w).unwrap();
        let bias = w.bias.map(|b| self.consts.get(b.as_ptr()));
        let p = Ug {
            n_out: inner as u32,
            ldw: w.n_in as u32,
            cin: cin as u32,
            taps: 1,
            off: [0; 9],
            a_rows: a_rows as u32,
            r0: 0,
            r_end: rows as u32,
            omode: 0,
            ldy: inner as u32,
            has_bias: bias.is_some() as u32,
            bstride: 0,
            add: 0,
            nimg: 0,
            g: Geo::default(),
            scaled: 0,
            full: 0,
            rs: 0,
            kper: 0,
            nct: 0,
            tm: 0,
            nrem: 0,
            _p: 0,
        };
        e.pipe(self.p.gemm_geglu);
        e.buf(mw.data.id, mw.data.offset_bytes, 0);
        e.buf(a, 0, 1);
        e.buf(self.sc(S::Gh), 0, 2);
        e.buf(self.cst(), bias.unwrap_or(0), 3);
        e.bytes(&p, 4);
        e.dispatch((inner.div_ceil(64), rows.div_ceil(64)), 128);
        e.barrier();
    }

    /// Group-norm statistics of tensor buffer `x` (layout `l`, `c` channels) into `S::Stats`.
    fn gn_stats(&self, e: &FastEnc, x: Id, l: Lay, c: usize, n: &Norm) {
        if skip("gn") {
            return;
        }
        let chunk = gn_chunk(l.h * l.w);
        let nchunk = (l.h * l.w).div_ceil(chunk);
        let p = Ugs {
            g: l.geo(),
            c: c as u32,
            groups: n.groups as u32,
            chunk: chunk as u32,
            nchunk: nchunk as u32,
        };
        e.pipe(self.p.stats);
        e.buf(x, 0, 0);
        e.buf(self.sc(S::Part), 0, 1);
        e.bytes(&p, 2);
        e.dispatch3((nchunk, self.nb, c.div_ceil(256)), 256);
        e.barrier();
        e.pipe(self.p.fin);
        e.buf(self.sc(S::Part), 0, 0);
        e.buf(self.sc(S::Stats), 0, 1);
        e.bytes(&p, 2);
        e.bytes(&n.eps, 3);
        e.buf(x, 0, 4);
        e.dispatch((n.groups, self.nb), 64);
        e.barrier();
    }

    /// Half rows of `x` (layout `src`, `c` channels) into `out`: padded rows of `dst` (or compact),
    /// normalized with `norm` (stats in `S::Stats`) first, upsampled from `src` when `dst` is
    /// twice its size.
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
        if scaled && norm.is_none() {
            self.amax(e, x, src, c);
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
        self.prep_consts(e, norm, scaled, 2, 3);
        e.bytes(&p, 4);
        e.dispatch(((rows * c / 4).div_ceil(256), 1), 256);
        e.barrier();
    }

    /// Binds what a preparation reads at buffers `st` and `gb`: the statistics and the norm's
    /// weights, or the scale's slot of `S::Amax`.
    fn prep_consts(&self, e: &FastEnc, norm: Option<(&Norm, bool)>, scaled: bool, st: usize, gb: usize) {
        if norm.is_none() && scaled {
            e.buf(self.sc(S::Amax), self.amax_off.get(), st);
        } else {
            e.buf(self.sc(S::Stats), 0, st);
        }
        e.buf(self.cst(), norm.map_or(0, |n| self.consts.get(n.0.gamma.as_ptr())), gb);
    }

    /// The largest magnitude of the interior of `x` (layout `src`, `c` channels) into the next
    /// slot of `S::Amax` (then [`Cx::amax_off`]).
    fn amax(&self, e: &FastEnc, x: Id, src: Lay, c: usize) {
        let off = self.amax_n.replace(self.amax_n.get() + 1) * 16;
        self.amax_off.set(off);
        if !skip("amax") {
            let rows = src.rows;
            e.pipe(self.p.amax);
            e.buf(x, 0, 0);
            e.buf(self.sc(S::Amax), off, 1);
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
        let wide = flash_wide(d, tk);
        let f = self.p.flash.iter().find(|f| f.0 == d).unwrap();
        let pipe = if wide { f.1 } else { f.2 };
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
        e.dispatch3(
            (tq.div_ceil(if wide { 128 } else { 64 }), a.heads * (d / value_chunk(d)), self.nb),
            128,
        );
        e.barrier();
    }

    fn op(&self, e: &FastEnc, op: &Op, kv_base: usize) {
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
                let norm = pre.as_ref().map(|(n, s)| (n, *s));
                self.prep_scaled(e, xb, ls, li, c, norm, false, xh, pre.is_none());
                let bias = bias
                    .as_ref()
                    .map(|b| (self.consts.get(b.as_ptr()), if *bias_per_image { oc } else { 0 }));
                let addb = add.map(|a| self.tid[a]);
                if shifted(*k, *stride, *pad, c, w.n_in, li.h, li.w) {
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
                        // Not the first image's first row nor the last one's last (borders).
                        (lo.g + lo.w + 2, lo.g + nb * lo.p - (lo.w + 2)),
                        out,
                        bias,
                        addb,
                        oc,
                        pre.is_none(),
                    );
                } else {
                    let rows = nb * oh * ow;
                    let kp = w.n_in;
                    if !skip("im2col") {
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
                        e.dispatch(((rows * kp / 4).div_ceil(256), 1), 256);
                        e.barrier();
                    }
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
            Op::Transformer { x, out, t } => self.transformer(e, *x, *out, t, kv_base),
            Op::Attention { x, out, norm, a } => {
                let [h, wd, c] = self.g.tensors[*x];
                let l = lay(h, wd, nb);
                let (n, rows) = (h * wd, nb * h * wd);
                let crow = compact_rows(rows);
                let xb = self.tid[*x];
                self.gn_stats(e, xb, l, c, norm);
                let xh = self.sc(S::Xh);
                self.prep(e, xb, l, l, c, Some((norm, false)), true, xh);
                let (qb, kb, vb) = (self.sc(S::Q), self.sc(S::K), self.sc(S::V));
                let ws = [
                    (self.wt(&a.q, a.heads), qb),
                    (self.wt(&a.k, a.heads), kb),
                    (self.wt(&a.v, a.heads), vb),
                ];
                self.lean(e, "qkv", &ws, xh, crow, c, rows, true, None, ws[0].0.n_out);
                self.flash(e, a, c, qb, kb, vb, n, n, n);
                self.out_proj(e, &a.o, self.sc(S::Ah), c, rows, self.tid[*out], l, xb);
            }
        }
    }

    fn transformer(&self, e: &FastEnc, x: usize, out: usize, t: &Transformer, kv_base: usize) {
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
        self.lean(e, "lin", &[(self.wt(&t.proj_in, 0), hb)], xh, crow, c, rows, false, None, c);
        let (hn, qb, kb, vb, ah) = (self.sc(S::Hn), self.sc(S::Q), self.sc(S::K), self.sc(S::V), self.sc(S::Ah));
        let cd = self.g.ctx_dim;
        let crows = nb * self.mp;
        for (bi, b) in t.blocks.iter().enumerate() {
            let (ck, cv) = (self.kv[2 * (kv_base + bi)], self.kv[2 * (kv_base + bi) + 1]);
            // Self-attention: q, k and v in one dispatch.
            self.ln(e, rows, c, &b.n1);
            let h1 = b.a1.heads;
            let ws = [(self.wt(&b.a1.q, h1), qb), (self.wt(&b.a1.k, h1), kb), (self.wt(&b.a1.v, h1), vb)];
            self.lean(e, "qkv", &ws, hn, crow, c, rows, true, None, ws[0].0.n_out);
            self.flash(e, &b.a1, c, qb, kb, vb, n, n, n);
            self.lean(e, "lin", &[(self.wt(&b.a1.o, 0), hb)], ah, crow, c, rows, false, Some(hb), c);
            // Cross-attention.
            self.ln(e, rows, c, &b.n2);
            let h2 = b.a2.heads;
            let wq = self.wt(&b.a2.q, h2);
            // The keys and values of the text: computed by the first pass over these contexts
            // (beside the query, without a barrier between them), read as they are after it.
            self.nobar.set(!self.kv_cached);
            self.lean(e, "qkv", &[(wq, qb)], hn, crow, c, rows, true, None, wq.n_out);
            if !self.kv_cached {
                let ws = [(self.wt(&b.a2.k, h2), ck), (self.wt(&b.a2.v, h2), cv)];
                self.lean(e, "qkv", &ws, self.sc(S::Ctx), crows + 64, cd, crows, true, None, wq.n_out);
            }
            self.flash(e, &b.a2, c, qb, ck, cv, n, self.m, self.mp);
            self.lean(e, "lin", &[(self.wt(&b.a2.o, 0), hb)], ah, crow, c, rows, false, Some(hb), c);
            // GEGLU feed-forward; the last block's output as the half input of proj_out.
            self.ln(e, rows, c, &b.n3);
            let inner = b.ff2.n_in;
            self.gemm_geglu(e, &b.ff1, hn, crow, c, rows, inner);
            let last = bi + 1 == t.blocks.len();
            let dst = if last { hn } else { hb };
            self.lean(
                e,
                "lin",
                &[(self.wt(&b.ff2, 0), dst)],
                self.sc(S::Gh),
                crow,
                inner,
                rows,
                last,
                Some(hb),
                c,
            );
        }
        if t.blocks.is_empty() {
            let n4 = (rows * c / 4) as u32;
            e.pipe(self.p.half);
            e.buf(hb, 0, 0);
            e.buf(hn, 0, 1);
            e.bytes(&n4, 2);
            e.dispatch(((n4 as usize).div_ceil(256), 1), 256);
            e.barrier();
        }
        // proj_out, added to the input, back into padded rows.
        self.out_proj(e, &t.proj_out, hn, c, rows, self.tid[out], l, xb);
    }
}

#[cfg(test)]
#[path = "metal_unet_tests.rs"]
mod tests;
