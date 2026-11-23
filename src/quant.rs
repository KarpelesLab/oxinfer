//! Int8 block quantization (Q8: blocks of 32 weights sharing one f32 scale).

use crate::kernels::bf16_to_f32;
use crate::pool;
use std::ops::{Deref, DerefMut};

pub const QBLOCK: usize = 32;

/// Page size assumed for GPU zero-copy wrapping (16 KiB covers Apple Silicon; a multiple of 4 KiB).
pub const PAGE: usize = 16384;

/// A zero-initialised, page-aligned buffer whose allocation is a whole number of pages, so GPU
/// APIs can map it without copying.
pub struct PageVec<T: Copy> {
    ptr: *mut T,
    len: usize,
    bytes: usize,
}
unsafe impl<T: Copy + Send> Send for PageVec<T> {}
unsafe impl<T: Copy + Sync> Sync for PageVec<T> {}

impl<T: Copy> PageVec<T> {
    pub fn zeroed(len: usize) -> Self {
        let bytes = (len * std::mem::size_of::<T>()).div_ceil(PAGE).max(1) * PAGE;
        let layout = std::alloc::Layout::from_size_align(bytes, PAGE).expect("layout");
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) } as *mut T;
        assert!(!ptr.is_null(), "allocation of {bytes} bytes failed");
        Self { ptr, len, bytes }
    }
    /// Size of the underlying page-aligned allocation in bytes.
    pub fn alloc_bytes(&self) -> usize {
        self.bytes
    }
    /// Rebuilds a `PageVec` from a pointer obtained by forgetting one created with `zeroed(len)`.
    ///
    /// # Safety
    /// `ptr` must come from `PageVec::<T>::zeroed(len)` with the same `len`, forgotten with
    /// `std::mem::forget`, and must not be rebuilt twice.
    pub unsafe fn from_raw(ptr: *mut T, len: usize) -> Self {
        let bytes = (len * std::mem::size_of::<T>()).div_ceil(PAGE).max(1) * PAGE;
        Self { ptr, len, bytes }
    }
}
impl<T: Copy> Clone for PageVec<T> {
    fn clone(&self) -> Self {
        let mut v = Self::zeroed(self.len);
        v.copy_from_slice(self);
        v
    }
}
impl<T: Copy> Drop for PageVec<T> {
    fn drop(&mut self) {
        let layout = std::alloc::Layout::from_size_align(self.bytes, PAGE).expect("layout");
        unsafe { std::alloc::dealloc(self.ptr as *mut u8, layout) };
    }
}
impl<T: Copy> Deref for PageVec<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}
impl<T: Copy> DerefMut for PageVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

/// Storage for quantized data: either owned (quantized at load time) or borrowed from a
/// pre-quantized checkpoint that outlives the matrix (the model borrows its `ModelFiles`).
pub struct QBuf<T: Copy> {
    ptr: *const T,
    len: usize,
    _own: Option<PageVec<T>>,
}
unsafe impl<T: Copy + Send> Send for QBuf<T> {}
unsafe impl<T: Copy + Sync> Sync for QBuf<T> {}

impl<T: Copy> QBuf<T> {
    /// Borrows `s` without copying.
    ///
    /// # Safety
    /// `s` must stay valid and unmodified for as long as the returned buffer is used.
    pub unsafe fn borrowed(s: &[T]) -> Self {
        Self {
            ptr: s.as_ptr(),
            len: s.len(),
            _own: None,
        }
    }
    pub fn is_borrowed(&self) -> bool {
        self._own.is_none()
    }
}
impl<T: Copy> From<PageVec<T>> for QBuf<T> {
    fn from(v: PageVec<T>) -> Self {
        Self {
            ptr: v.as_ptr(),
            len: v.len(),
            _own: Some(v),
        }
    }
}
impl<T: Copy> Deref for QBuf<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

/// Quantized storage formats, as recorded in and read from pre-quantized checkpoints.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QKind {
    Q8,
    Q4,
}

impl QKind {
    pub fn suffix(self) -> &'static str {
        match self {
            QKind::Q8 => "q8",
            QKind::Q4 => "q4",
        }
    }
}

/// A quantized `[n_out, n_in]` matrix: int8 values plus one scale per 32-element block.
pub struct QMatrix {
    pub q: QBuf<i8>,
    pub scales: QBuf<f32>,
    pub n_out: usize,
    pub n_in: usize,
}

impl QMatrix {
    pub fn blocks_per_row(&self) -> usize {
        self.n_in / QBLOCK
    }
    pub fn bytes(&self) -> usize {
        self.q.len() + self.scales.len() * 4
    }
}

#[derive(Clone, Copy)]
struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
impl<T> SendPtr<T> {
    #[inline(always)]
    fn get(self) -> *mut T {
        self.0
    }
}

/// A weight element the quantizers read: bf16 bits or f32.
pub trait QSrc: Copy + Sync {
    fn f32(self) -> f32;
}
impl QSrc for u16 {
    #[inline(always)]
    fn f32(self) -> f32 {
        bf16_to_f32(self)
    }
}
impl QSrc for f32 {
    #[inline(always)]
    fn f32(self) -> f32 {
        self
    }
}

/// Quantizes a bf16 row-major matrix. Symmetric round-to-nearest, scale = absmax / 127 per block.
pub fn quantize_bf16(w: &[u16], n_out: usize, n_in: usize) -> QMatrix {
    quantize_q8_from(w, n_out, n_in)
}

/// [`quantize_bf16`] for f32 weights (e.g. a LoRA merged in f32, which rounding to bf16 first
/// would mostly erase: a step-distillation update is often below half a bf16 ulp of the weight).
pub fn quantize_f32(w: &[f32], n_out: usize, n_in: usize) -> QMatrix {
    quantize_q8_from(w, n_out, n_in)
}

fn quantize_q8_from<T: QSrc>(w: &[T], n_out: usize, n_in: usize) -> QMatrix {
    assert_eq!(w.len(), n_out * n_in);
    assert!(n_in.is_multiple_of(QBLOCK), "n_in must be a multiple of {QBLOCK}");
    let bpr = n_in / QBLOCK;
    let mut q = PageVec::<i8>::zeroed(n_out * n_in);
    let mut scales = PageVec::<f32>::zeroed(n_out * bpr);
    let qp = SendPtr(q.as_mut_ptr());
    let sp = SendPtr(scales.as_mut_ptr());
    // Rows in chunks so each task is a few hundred KiB.
    let rows_per_task = (256 * 1024 / (n_in * 2)).max(1);
    let ntasks = n_out.div_ceil(rows_per_task);
    pool::run(ntasks, &|task| {
        let r0 = task * rows_per_task;
        let r1 = (r0 + rows_per_task).min(n_out);
        for r in r0..r1 {
            let row = &w[r * n_in..(r + 1) * n_in];
            // SAFETY: rows are disjoint between tasks.
            let qrow = unsafe { std::slice::from_raw_parts_mut(qp.get().add(r * n_in), n_in) };
            let srow = unsafe { std::slice::from_raw_parts_mut(sp.get().add(r * bpr), bpr) };
            for b in 0..bpr {
                let blk = &row[b * QBLOCK..(b + 1) * QBLOCK];
                let mut amax = 0f32;
                for &v in blk {
                    amax = amax.max(v.f32().abs());
                }
                let scale = amax / 127.0;
                let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
                srow[b] = scale;
                for (i, &v) in blk.iter().enumerate() {
                    qrow[b * QBLOCK + i] = (v.f32() * inv).round().clamp(-127.0, 127.0) as i8;
                }
            }
        }
    });
    QMatrix {
        q: q.into(),
        scales: scales.into(),
        n_out,
        n_in,
    }
}

/// Quantized activations `[t, n_in]`: int8 values and one scale per block.
#[allow(dead_code)]
pub struct QAct {
    pub q: Vec<i8>,
    pub scales: Vec<f32>,
    /// Sum of the int8 values of each block (used by the Q4 kernel to fold the nibble offset).
    pub sums: Vec<i32>,
    /// Exact f32 sum of each block (for the per-group offsets of affine Q4 matrices).
    pub fsums: Vec<f32>,
    /// Exact f32 sum of each half block of 16 (for the per-16 mins of GGUF Q2_K weights).
    pub hsums: Vec<f32>,
    /// `-8` times each block's integer sum (the offset of centered nibbles, ready to start an
    /// integer dot product from: the interleaved Q4_0 and Q4_K kernels).
    pub neg8: Vec<i32>,
    /// `-32` times the integer sum of each half block of 16 (Q6_K's offset, likewise).
    pub neg32h: Vec<i32>,
    pub t: usize,
    pub n_in: usize,
}

/// One block of 32: int8 values (`v * 127 / amax`, rounded half away from zero), returning
/// `amax` and the integer sums of the two halves of 16.
#[inline(always)]
fn quantize_block(blk: &[f32], q: &mut [i8]) -> (f32, [i32; 2]) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let p = blk.as_ptr();
        let v: [float32x4_t; 8] = std::array::from_fn(|i| vld1q_f32(p.add(4 * i)));
        let mut m = vdupq_n_f32(0.0);
        for x in v {
            m = vmaxnmq_f32(m, vabsq_f32(x));
        }
        let amax = vmaxnmvq_f32(m);
        let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
        let mut s = [vdupq_n_s32(0); 2];
        for h in 0..4 {
            let (a, b) = (
                vcvtaq_s32_f32(vmulq_n_f32(v[2 * h], inv)),
                vcvtaq_s32_f32(vmulq_n_f32(v[2 * h + 1], inv)),
            );
            s[h / 2] = vaddq_s32(s[h / 2], vaddq_s32(a, b));
            let n = vqmovn_s16(vcombine_s16(vqmovn_s32(a), vqmovn_s32(b)));
            vst1_s8(q.as_mut_ptr().add(8 * h), n);
        }
        (amax, [vaddvq_s32(s[0]), vaddvq_s32(s[1])])
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
        let mut s = [0i32; 2];
        for (i, (o, &v)) in q.iter_mut().zip(blk).enumerate() {
            *o = (v * inv).round() as i8;
            s[i / 16] += *o as i32;
        }
        (amax, s)
    }
}

/// A row's slices of a `QAct`.
struct RowOut<'a> {
    q: &'a mut [i8],
    scales: &'a mut [f32],
    sums: &'a mut [i32],
    fsums: &'a mut [f32],
    hsums: &'a mut [f32],
    neg8: &'a mut [i32],
    neg32h: &'a mut [i32],
}

/// Quantizes one row of `x` (`n_in = q.len()` values) into its slices of a `QAct`.
fn quantize_row(row: &[f32], o: RowOut) {
    let RowOut {
        q,
        scales,
        sums,
        fsums,
        hsums,
        neg8,
        neg32h,
    } = o;
    let nb = row.len() / QBLOCK;
    for b in 0..nb {
        let (amax, [s0, s1]) = quantize_block(&row[b * QBLOCK..(b + 1) * QBLOCK], &mut q[b * QBLOCK..(b + 1) * QBLOCK]);
        scales[b] = amax / 127.0;
        sums[b] = s0 + s1;
        neg8[b] = -8 * (s0 + s1);
        neg32h[2 * b] = -32 * s0;
        neg32h[2 * b + 1] = -32 * s1;
    }
    // The exact sums, each a sequential left-to-right sum (`Iterator::sum`'s order, from -0.0):
    // on NEON four blocks at a time, one per lane (the sums of a block are serial chains, the
    // blocks independent).
    #[allow(unused_mut)]
    let mut b0 = 0;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        while b0 + 4 <= nb {
            let p = row.as_ptr().add(b0 * QBLOCK);
            let (mut h0, mut h1) = (vdupq_n_f32(-0.0), vdupq_n_f32(-0.0));
            let mut second = [vdupq_n_f32(0.0); 16];
            for i in (0..16).step_by(4) {
                // Elements i..i+4 of the four blocks, transposed: lane k of `c[j]` is element
                // i + j of block k (and `d` the same 16 further on).
                let ld = |o: usize| -> [float32x4_t; 4] {
                    let r: [float32x4_t; 4] = std::array::from_fn(|k| vld1q_f32(p.add(k * QBLOCK + o)));
                    let (t0, t1) = (vtrn1q_f32(r[0], r[1]), vtrn2q_f32(r[0], r[1]));
                    let (t2, t3) = (vtrn1q_f32(r[2], r[3]), vtrn2q_f32(r[2], r[3]));
                    let (a0, a1) = (vreinterpretq_f64_f32(t0), vreinterpretq_f64_f32(t1));
                    let (a2, a3) = (vreinterpretq_f64_f32(t2), vreinterpretq_f64_f32(t3));
                    [
                        vreinterpretq_f32_f64(vtrn1q_f64(a0, a2)),
                        vreinterpretq_f32_f64(vtrn1q_f64(a1, a3)),
                        vreinterpretq_f32_f64(vtrn2q_f64(a0, a2)),
                        vreinterpretq_f32_f64(vtrn2q_f64(a1, a3)),
                    ]
                };
                let (c, d) = (ld(i), ld(16 + i));
                for j in 0..4 {
                    h0 = vaddq_f32(h0, c[j]);
                    h1 = vaddq_f32(h1, d[j]);
                    second[i + j] = d[j];
                }
            }
            // The whole block's sum continues the first half's chain.
            let mut f = h0;
            for v in second {
                f = vaddq_f32(f, v);
            }
            let (mut a0, mut a1, mut af) = ([0f32; 4], [0f32; 4], [0f32; 4]);
            vst1q_f32(a0.as_mut_ptr(), h0);
            vst1q_f32(a1.as_mut_ptr(), h1);
            vst1q_f32(af.as_mut_ptr(), f);
            for k in 0..4 {
                hsums[2 * (b0 + k)] = a0[k];
                hsums[2 * (b0 + k) + 1] = a1[k];
                fsums[b0 + k] = af[k];
            }
            b0 += 4;
        }
    }
    for b in b0..nb {
        let blk = &row[b * QBLOCK..(b + 1) * QBLOCK];
        let (h0, h1): (f32, f32) = (blk[..16].iter().sum(), blk[16..].iter().sum());
        hsums[2 * b] = h0;
        hsums[2 * b + 1] = h1;
        fsums[b] = blk.iter().sum();
    }
}

/// Raw pointers to every array of a [`QAct`], for filling disjoint blocks from several threads.
struct QActPtrs {
    q: SendPtr<i8>,
    scales: SendPtr<f32>,
    sums: SendPtr<i32>,
    fsums: SendPtr<f32>,
    hsums: SendPtr<f32>,
    neg8: SendPtr<i32>,
    neg32h: SendPtr<i32>,
}

impl QActPtrs {
    fn new(xa: &mut QAct) -> Self {
        Self {
            q: SendPtr(xa.q.as_mut_ptr()),
            scales: SendPtr(xa.scales.as_mut_ptr()),
            sums: SendPtr(xa.sums.as_mut_ptr()),
            fsums: SendPtr(xa.fsums.as_mut_ptr()),
            hsums: SendPtr(xa.hsums.as_mut_ptr()),
            neg8: SendPtr(xa.neg8.as_mut_ptr()),
            neg32h: SendPtr(xa.neg32h.as_mut_ptr()),
        }
    }

    /// The slices of blocks `b0..b0 + nb` (counted over all rows).
    ///
    /// # Safety
    /// In bounds, and no other live `RowOut` covers these blocks.
    #[allow(clippy::mut_from_ref)]
    unsafe fn out(&self, b0: usize, nb: usize) -> RowOut<'_> {
        use std::slice::from_raw_parts_mut as sl;
        unsafe {
            RowOut {
                q: sl(self.q.get().add(b0 * QBLOCK), nb * QBLOCK),
                scales: sl(self.scales.get().add(b0), nb),
                sums: sl(self.sums.get().add(b0), nb),
                fsums: sl(self.fsums.get().add(b0), nb),
                hsums: sl(self.hsums.get().add(2 * b0), 2 * nb),
                neg8: sl(self.neg8.get().add(b0), nb),
                neg32h: sl(self.neg32h.get().add(2 * b0), 2 * nb),
            }
        }
    }
}

impl QAct {
    /// `t` rows of `n_in` (a multiple of [`QBLOCK`]) for a caller that writes every block: a
    /// buffer set recycled on this thread (its earlier contents) when one is large enough, else
    /// zeros. Decoding quantizes several inputs a layer; recycling them saves the allocations
    /// and zero fills of each.
    pub(crate) fn zeroed(t: usize, n_in: usize) -> Self {
        assert!(n_in.is_multiple_of(QBLOCK));
        let nb = t * n_in / QBLOCK;
        let reused = QACT_FREE.with(|f| {
            let mut f = f.borrow_mut();
            let i = f.iter().position(|v| v.q.len() >= t * n_in)?;
            Some(f.swap_remove(i))
        });
        if let Some(mut v) = reused {
            // (Every array's length is proportional to its `q`'s.)
            v.q.truncate(t * n_in);
            v.scales.truncate(nb);
            v.sums.truncate(nb);
            v.fsums.truncate(nb);
            v.hsums.truncate(2 * nb);
            v.neg8.truncate(nb);
            v.neg32h.truncate(2 * nb);
            v.t = t;
            v.n_in = n_in;
            return v;
        }
        QAct {
            q: vec![0i8; t * n_in],
            scales: vec![0f32; nb],
            sums: vec![0i32; nb],
            fsums: vec![0f32; nb],
            hsums: vec![0f32; 2 * nb],
            neg8: vec![0i32; nb],
            neg32h: vec![0i32; 2 * nb],
            t,
            n_in,
        }
    }
}

/// Activation buffers given back on this thread ([`QAct`]'s drop), at most [`QACT_KEEP`] of up
/// to [`QACT_MAX`] values each (decode-sized: prompts' are not kept).
const QACT_KEEP: usize = 8;
const QACT_MAX: usize = 1 << 16;

thread_local! {
    // Already const; clippy (1.100) still flags it on Android's thread-local implementation.
    #[allow(clippy::missing_const_for_thread_local)]
    static QACT_FREE: std::cell::RefCell<Vec<QAct>> = const { std::cell::RefCell::new(Vec::new()) };
}

impl Drop for QAct {
    fn drop(&mut self) {
        let n = self.q.len();
        if n == 0 || n > QACT_MAX || self.scales.len() * QBLOCK != n || self.neg32h.len() * 16 != n {
            return;
        }
        // (`try_with`: nothing is kept while the thread's storage is being torn down.)
        let _ = QACT_FREE.try_with(|f| {
            let Ok(mut f) = f.try_borrow_mut() else { return };
            // The slot to fill: a free one, else the smallest set if this one is larger.
            let slot = if f.len() < QACT_KEEP {
                None
            } else {
                match (0..f.len()).min_by_key(|&i| f[i].q.len()) {
                    Some(i) if f[i].q.len() < n => Some(i),
                    _ => return,
                }
            };
            let v = QAct {
                q: std::mem::take(&mut self.q),
                scales: std::mem::take(&mut self.scales),
                sums: std::mem::take(&mut self.sums),
                fsums: std::mem::take(&mut self.fsums),
                hsums: std::mem::take(&mut self.hsums),
                neg8: std::mem::take(&mut self.neg8),
                neg32h: std::mem::take(&mut self.neg32h),
                t: 0,
                n_in: 0,
            };
            // (The set replaced is dropped while `f` is borrowed, so it is not kept again.)
            match slot {
                None => f.push(v),
                Some(i) => f[i] = v,
            }
        });
    }
}

/// A [`QAct`] of `t` rows of `n_in` filled block by block from several threads at once
/// ([`QActFill::fill`]): each block gets exactly what [`quantize_act`] gives it, so filling every
/// block of every row makes the same activations.
pub(crate) struct QActFill {
    xa: QAct,
    p: QActPtrs,
}

impl QActFill {
    pub(crate) fn new(t: usize, n_in: usize) -> Self {
        let mut xa = QAct::zeroed(t, n_in);
        Self {
            p: QActPtrs::new(&mut xa),
            xa,
        }
    }

    /// Quantizes `src`, elements `e0..e0 + src.len()` of row `ti` (whole blocks).
    ///
    /// # Safety
    /// Calls running at the same time cover disjoint blocks.
    pub(crate) unsafe fn fill(&self, ti: usize, e0: usize, src: &[f32]) {
        let n_in = self.xa.n_in;
        assert!(ti < self.xa.t && e0.is_multiple_of(QBLOCK) && src.len().is_multiple_of(QBLOCK) && e0 + src.len() <= n_in);
        let (b0, nb) = ((ti * n_in + e0) / QBLOCK, src.len() / QBLOCK);
        // SAFETY: in bounds (checked above); disjoint from other calls' blocks (the caller's
        // contract).
        unsafe { quantize_row(src, self.p.out(b0, nb)) }
    }

    /// The activations (every block filled).
    pub(crate) fn finish(self) -> QAct {
        self.xa
    }
}

pub fn quantize_act(x: &[f32], t: usize, n_in: usize) -> QAct {
    assert_eq!(x.len(), t * n_in);
    let mut xa = QAct::zeroed(t, n_in);
    let bpr = n_in / QBLOCK;
    let p = QActPtrs::new(&mut xa);
    if t == 1 {
        // SAFETY: the whole of the one row.
        unsafe { quantize_row(x, p.out(0, bpr)) };
    } else {
        // Rows in parallel (several tokens: speculative verification, prefill).
        // SAFETY: each job takes its own row.
        crate::pool::run(t, &|ti| unsafe {
            quantize_row(&x[ti * n_in..(ti + 1) * n_in], p.out(ti * bpr, bpr))
        });
    }
    xa
}

impl QAct {
    /// Rows `rows` (in that order) as their own `QAct`: the same values as quantizing those rows
    /// of the input alone, since every row is quantized on its own.
    pub fn gather(&self, rows: &[usize]) -> QAct {
        let n = self.n_in;
        let bpr = n / QBLOCK;
        fn pick<T: Copy>(v: &[T], per: usize, rows: &[usize]) -> Vec<T> {
            rows.iter().flat_map(|&r| v[r * per..(r + 1) * per].iter().copied()).collect()
        }
        QAct {
            q: pick(&self.q, n, rows),
            scales: pick(&self.scales, bpr, rows),
            sums: pick(&self.sums, bpr, rows),
            fsums: pick(&self.fsums, bpr, rows),
            hsums: pick(&self.hsums, 2 * bpr, rows),
            neg8: pick(&self.neg8, bpr, rows),
            neg32h: pick(&self.neg32h, 2 * bpr, rows),
            t: rows.len(),
            n_in: n,
        }
    }
}

/// Quantized activations interleaved per token pair for 2x2 int8 matrix-multiply instructions
/// (SMMLA): pair `p` holds tokens `2p` and `2p + 1` (an odd count repeats the last token).
/// `q`: per 8 values, 8 of the first token then 8 of the second (`2 * n_in` bytes per pair);
/// `s` (per block of 32) and `h` (per half block) as `[first, second, first, second]`.
pub struct PairAct {
    pub q: Vec<i8>,
    pub s: Vec<f32>,
    pub h: Vec<f32>,
}

impl PairAct {
    pub fn new(xa: &QAct) -> Self {
        let (t, n) = (xa.t, xa.n_in);
        let bpr = n / QBLOCK;
        let np = t.div_ceil(2);
        let mut q = vec![0i8; np * 2 * n];
        let mut s = vec![0f32; np * bpr * 4];
        let mut h = vec![0f32; np * bpr * 8];
        for p in 0..np {
            let (t0, t1) = (2 * p, (2 * p + 1).min(t - 1));
            let qp = &mut q[p * 2 * n..(p + 1) * 2 * n];
            for (k, d) in qp.as_chunks_mut::<16>().0.iter_mut().enumerate() {
                d[..8].copy_from_slice(&xa.q[t0 * n + 8 * k..t0 * n + 8 * k + 8]);
                d[8..].copy_from_slice(&xa.q[t1 * n + 8 * k..t1 * n + 8 * k + 8]);
            }
            fn quad<T: Copy>(dst: &mut [T], src: &[T], per: usize, t0: usize, t1: usize) {
                for (b, d) in dst.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                    let (a, c) = (src[t0 * per + b], src[t1 * per + b]);
                    *d = [a, c, a, c];
                }
            }
            quad(&mut s[p * bpr * 4..(p + 1) * bpr * 4], &xa.scales, bpr, t0, t1);
            quad(&mut h[p * bpr * 8..(p + 1) * bpr * 8], &xa.hsums, 2 * bpr, t0, t1);
        }
        Self { q, s, h }
    }
}

/// Dequantizes rows `r0..r1` of `m` into `out` (f32, row-major, `(r1-r0) * n_in`).
pub fn dequantize_rows(m: &QMatrix, r0: usize, r1: usize, out: &mut [f32]) {
    dequantize_block(m, r0, r1, 0, m.n_in, out)
}

/// Columns `k0..k1` (multiples of [`QBLOCK`]) of rows `r0..r1` of `m` into `out` (f32,
/// row-major, `(r1-r0) * (k1-k0)`): the values [`dequantize_rows`] gives there.
pub fn dequantize_block(m: &QMatrix, r0: usize, r1: usize, k0: usize, k1: usize, out: &mut [f32]) {
    let n_in = m.n_in;
    let bpr = m.blocks_per_row();
    let (b0, b1, n) = (k0 / QBLOCK, k1 / QBLOCK, k1 - k0);
    debug_assert!(k0.is_multiple_of(QBLOCK) && k1.is_multiple_of(QBLOCK) && k1 <= n_in);
    for (i, r) in (r0..r1).enumerate() {
        let qrow = &m.q[r * n_in + k0..r * n_in + k1];
        let srow = &m.scales[r * bpr + b0..r * bpr + b1];
        let orow = &mut out[i * n..(i + 1) * n];
        for (b, &s) in srow.iter().enumerate() {
            for j in 0..QBLOCK {
                orow[b * QBLOCK + j] = qrow[b * QBLOCK + j] as f32 * s;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Q4: GGUF Q4_0 layout: 4-bit symmetric blocks of 32 (two nibbles per byte, f16 scale) = 4.5 bits/weight.
// Byte j of a block holds element j in its low nibble and element j+16 in its high nibble;
// stored value = clamp(round(v / d), -8, 7) + 8 with d = (signed absmax) / -8.
// ---------------------------------------------------------------------------

pub struct Q4Matrix {
    pub q: QBuf<u8>,       // [n_out, n_in/2]
    pub scales: QBuf<u16>, // f16, [n_out, n_in/32]
    pub n_out: usize,
    pub n_in: usize,
    /// Affine form (MLX checkpoints): a bf16 term added to every element of a group,
    /// `[n_out, n_in/group]`, with `group` a multiple of 32.
    pub offsets: Option<Offsets>,
}

pub struct Offsets {
    pub v: QBuf<u16>,
    pub group: usize,
}

impl Q4Matrix {
    pub fn bytes(&self) -> usize {
        self.q.len() + self.scales.len() * 2 + self.offsets.as_ref().map_or(0, |o| o.v.len() * 2)
    }
}

#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    half::f16::from_bits(h).to_f32()
}
#[inline]
fn f32_to_f16(x: f32) -> u16 {
    half::f16::from_f32(x).to_bits()
}

pub fn quantize_q4(w: &[u16], n_out: usize, n_in: usize) -> Q4Matrix {
    quantize_q4_from(w, n_out, n_in)
}

/// [`quantize_q4`] for f32 weights.
pub fn quantize_q4_f32(w: &[f32], n_out: usize, n_in: usize) -> Q4Matrix {
    quantize_q4_from(w, n_out, n_in)
}

fn quantize_q4_from<T: QSrc>(w: &[T], n_out: usize, n_in: usize) -> Q4Matrix {
    assert_eq!(w.len(), n_out * n_in);
    assert!(n_in.is_multiple_of(QBLOCK));
    let bpr = n_in / QBLOCK;
    let mut q = PageVec::<u8>::zeroed(n_out * n_in / 2);
    let mut scales = PageVec::<u16>::zeroed(n_out * bpr);
    let qp = SendPtr(q.as_mut_ptr());
    let sp = SendPtr(scales.as_mut_ptr());
    let rows_per_task = (256 * 1024 / (n_in * 2)).max(1);
    let ntasks = n_out.div_ceil(rows_per_task);
    pool::run(ntasks, &|task| {
        let r0 = task * rows_per_task;
        let r1 = (r0 + rows_per_task).min(n_out);
        let mut blk = [0f32; QBLOCK];
        for r in r0..r1 {
            let row = &w[r * n_in..(r + 1) * n_in];
            // SAFETY: rows are disjoint between tasks.
            let qrow = unsafe { std::slice::from_raw_parts_mut(qp.get().add(r * n_in / 2), n_in / 2) };
            let srow = unsafe { std::slice::from_raw_parts_mut(sp.get().add(r * bpr), bpr) };
            for b in 0..bpr {
                let mut vmax = 0f32;
                for (j, &v) in row[b * QBLOCK..(b + 1) * QBLOCK].iter().enumerate() {
                    let f = v.f32();
                    blk[j] = f;
                    if f.abs() > vmax.abs() {
                        vmax = f;
                    }
                }
                let d = f16_to_f32(f32_to_f16(vmax / -8.0));
                srow[b] = f32_to_f16(d);
                let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                let out = &mut qrow[b * 16..(b + 1) * 16];
                for j in 0..16 {
                    let lo = ((blk[j] * id).round().clamp(-8.0, 7.0) as i32 + 8) as u8;
                    let hi = ((blk[j + 16] * id).round().clamp(-8.0, 7.0) as i32 + 8) as u8;
                    out[j] = lo | (hi << 4);
                }
            }
        }
    });
    Q4Matrix {
        q: q.into(),
        scales: scales.into(),
        n_out,
        n_in,
        offsets: None,
    }
}

/// Repacks an MLX affine 4-bit matrix (`w = scale * q + bias` per group of `group` elements, eight
/// nibbles per little-endian u32, low nibble first) into Q4_0 layout plus per-group offsets:
/// `scale * q + bias = scale * (q - 8) + (bias + 8 * scale)`. The nibbles carry over unchanged.
pub fn from_mlx_q4(w: &[u32], scales: &[u16], biases: &[u16], n_out: usize, n_in: usize, group: usize) -> Q4Matrix {
    assert!(group.is_multiple_of(QBLOCK) && n_in.is_multiple_of(group));
    assert_eq!(w.len(), n_out * n_in / 8);
    let gpr = n_in / group;
    assert_eq!(scales.len(), n_out * gpr);
    assert_eq!(biases.len(), n_out * gpr);
    let bpr = n_in / QBLOCK;
    let mut q = PageVec::<u8>::zeroed(n_out * n_in / 2);
    let mut sc = PageVec::<u16>::zeroed(n_out * bpr);
    let mut off = PageVec::<u16>::zeroed(n_out * gpr);
    let (qp, sp, op) = (SendPtr(q.as_mut_ptr()), SendPtr(sc.as_mut_ptr()), SendPtr(off.as_mut_ptr()));
    let rows_per_task = (256 * 1024 / n_in).max(1);
    pool::run(n_out.div_ceil(rows_per_task), &|task| {
        let r0 = task * rows_per_task;
        for r in r0..(r0 + rows_per_task).min(n_out) {
            // SAFETY: rows are disjoint between tasks.
            let qrow = unsafe { std::slice::from_raw_parts_mut(qp.get().add(r * n_in / 2), n_in / 2) };
            let srow = unsafe { std::slice::from_raw_parts_mut(sp.get().add(r * bpr), bpr) };
            let orow = unsafe { std::slice::from_raw_parts_mut(op.get().add(r * gpr), gpr) };
            let wrow = &w[r * n_in / 8..(r + 1) * n_in / 8];
            for g in 0..gpr {
                let s = bf16_to_f32(scales[r * gpr + g]);
                let b = bf16_to_f32(biases[r * gpr + g]);
                orow[g] = f32_to_bf16(b + 8.0 * s);
                let d = f32_to_f16(s);
                for blk in g * group / QBLOCK..(g + 1) * group / QBLOCK {
                    srow[blk] = d;
                }
            }
            for blk in 0..bpr {
                let words = &wrow[blk * 4..blk * 4 + 4];
                let nib = |e: usize| ((words[e / 8] >> (4 * (e % 8))) & 15) as u8;
                for j in 0..16 {
                    qrow[blk * 16 + j] = nib(j) | (nib(j + 16) << 4);
                }
            }
        }
    });
    Q4Matrix {
        q: q.into(),
        scales: sc.into(),
        n_out,
        n_in,
        offsets: Some(Offsets { v: off.into(), group }),
    }
}

/// Dequantizes one row of an MLX affine matrix (any bit width dividing 32) into `out`.
pub fn mlx_dequant_row(w: &[u32], scales: &[u16], biases: &[u16], bits: usize, group: usize, out: &mut [f32]) {
    let per = 32 / bits;
    let mask = (1u32 << bits) - 1;
    for (g, chunk) in out.chunks_mut(group).enumerate() {
        let s = bf16_to_f32(scales[g]);
        let b = bf16_to_f32(biases[g]);
        for (i, o) in chunk.iter_mut().enumerate() {
            let e = g * group + i;
            let q = (w[e / per] >> (bits * (e % per))) & mask;
            *o = s * q as f32 + b;
        }
    }
}

#[inline]
fn f32_to_bf16(x: f32) -> u16 {
    half::bf16::from_f32(x).to_bits()
}

/// Dequantizes rows `r0..r1` of `m` into `out` (f32 row-major).
pub fn dequantize_rows_q4(m: &Q4Matrix, r0: usize, r1: usize, out: &mut [f32]) {
    dequantize_block_q4(m, r0, r1, 0, m.n_in, out)
}

/// Columns `k0..k1` (multiples of [`QBLOCK`]) of rows `r0..r1` of `m` into `out` (f32,
/// row-major, `(r1-r0) * (k1-k0)`): the values [`dequantize_rows_q4`] gives there.
pub fn dequantize_block_q4(m: &Q4Matrix, r0: usize, r1: usize, k0: usize, k1: usize, out: &mut [f32]) {
    let n_in = m.n_in;
    let bpr = n_in / QBLOCK;
    let (b0, b1, n) = (k0 / QBLOCK, k1 / QBLOCK, k1 - k0);
    debug_assert!(k0.is_multiple_of(QBLOCK) && k1.is_multiple_of(QBLOCK) && k1 <= n_in);
    for (i, r) in (r0..r1).enumerate() {
        let qrow = &m.q[r * n_in / 2..(r + 1) * n_in / 2];
        let srow = &m.scales[r * bpr..(r + 1) * bpr];
        let orow = &mut out[i * n..(i + 1) * n];
        for b in b0..b1 {
            let d = f16_to_f32(srow[b]);
            let m = m
                .offsets
                .as_ref()
                .map_or(0.0, |o| bf16_to_f32(o.v[r * (n_in / o.group) + b * QBLOCK / o.group]));
            let o = &mut orow[(b - b0) * QBLOCK..(b - b0 + 1) * QBLOCK];
            for j in 0..16 {
                let byte = qrow[b * 16 + j];
                o[j] = ((byte & 0x0F) as i32 - 8) as f32 * d + m;
                o[j + 16] = ((byte >> 4) as i32 - 8) as f32 * d + m;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The f32 quantizers give the bf16 ones' bytes on bf16-representable weights.
    #[test]
    fn f32_quantizers_match_bf16() {
        let (n_out, n_in) = (3usize, 2 * QBLOCK);
        let mut seed = 11u64;
        let bf: Vec<u16> = (0..n_out * n_in)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                crate::sd::ops::f32_to_bf16(((seed >> 33) % 2001) as f32 / 1000.0 - 1.0)
            })
            .collect();
        let f: Vec<f32> = bf.iter().map(|&v| bf16_to_f32(v)).collect();
        let (a, b) = (quantize_bf16(&bf, n_out, n_in), quantize_f32(&f, n_out, n_in));
        assert_eq!(&a.q[..], &b.q[..]);
        assert_eq!(&a.scales[..], &b.scales[..]);
        let (a, b) = (quantize_q4(&bf, n_out, n_in), quantize_q4_f32(&f, n_out, n_in));
        assert_eq!(&a.q[..], &b.q[..]);
        assert_eq!(&a.scales[..], &b.scales[..]);
    }

    /// The vector / parallel activation quantization equals the plain scalar definition.
    #[test]
    fn activation_quantization_matches_scalar() {
        let (t, n) = (5usize, 4 * QBLOCK);
        let mut seed = 7u64;
        let mut x: Vec<f32> = (0..t * n)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((seed >> 33) % 20001) as f32 / 1000.0 - 10.0
            })
            .collect();
        x[QBLOCK..2 * QBLOCK].fill(0.0); // an all-zero block
        x[2 * QBLOCK + 3] = 1e-30;
        // (The later passes take the earlier ones' recycled buffers, stale contents included.)
        for tt in [1, t, 1, 2, t] {
            let xa = quantize_act(&x[..tt * n], tt, n);
            assert_eq!((xa.q.len(), xa.neg32h.len(), xa.t), (tt * n, tt * n / 16, tt));
            for (bi, blk) in x[..tt * n].chunks(QBLOCK).enumerate() {
                let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
                let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
                let q: Vec<i8> = blk.iter().map(|&v| (v * inv).round() as i8).collect();
                assert_eq!(&xa.q[bi * QBLOCK..(bi + 1) * QBLOCK], &q[..], "block {bi}");
                assert_eq!(xa.scales[bi].to_bits(), (amax / 127.0).to_bits());
                assert_eq!(xa.sums[bi], q.iter().map(|&v| v as i32).sum::<i32>());
                assert_eq!(xa.fsums[bi].to_bits(), blk.iter().sum::<f32>().to_bits());
                assert_eq!(xa.hsums[2 * bi + 1].to_bits(), blk[16..].iter().sum::<f32>().to_bits());
                assert_eq!(xa.neg8[bi], -8 * xa.sums[bi]);
                for h in 0..2 {
                    let s16: i32 = q[16 * h..16 * h + 16].iter().map(|&v| v as i32).sum();
                    assert_eq!(xa.neg32h[2 * bi + h], -32 * s16);
                }
            }
        }
    }
}

#[cfg(test)]
mod act_sum_tests {
    /// The activation sums against `Iterator::sum` of each block and half block, bit for bit
    /// (rows of several lengths: the four-block NEON path and the remainder; signed zeros).
    #[test]
    fn act_sums_match_sequential_sums() {
        let mut seed = 3u64;
        for n in [32usize, 96, 128, 2048, 8192 + 96] {
            let x: Vec<f32> = (0..n)
                .map(|i| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let v = ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * [1e-3, 1.0, 300.0, 7e4][(i / 32) % 4];
                    if i % 61 == 0 { -0.0 } else { v }
                })
                .collect();
            let a = super::quantize_act(&x, 1, n);
            for (b, blk) in x.chunks(32).enumerate() {
                let f: f32 = blk.iter().sum();
                let (h0, h1): (f32, f32) = (blk[..16].iter().sum(), blk[16..].iter().sum());
                assert_eq!(a.fsums[b].to_bits(), f.to_bits(), "n {n} block {b}");
                assert_eq!(a.hsums[2 * b].to_bits(), h0.to_bits(), "n {n} block {b}");
                assert_eq!(a.hsums[2 * b + 1].to_bits(), h1.to_bits(), "n {n} block {b}");
            }
        }
        // A block of negative zeros sums to -0.0 (as `Iterator::sum` does).
        let a = super::quantize_act(&[-0.0f32; 128], 1, 128);
        assert!(a.fsums.iter().chain(&a.hsums).all(|v| v.to_bits() == (-0.0f32).to_bits()));
    }
}
