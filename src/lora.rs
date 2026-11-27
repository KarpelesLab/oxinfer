//! Runtime (unmerged) LoRA adapters for the language models: a PEFT adapter directory
//! (`adapter_config.json` + `adapter_model.safetensors`) is attached to the linear layers it
//! targets, and while its switch is on every targeted layer adds `scale * B (A x)` to its output.
//! The base weights stay untouched, in whatever format they were loaded (bf16, Q8 / Q4,
//! pre-quantized, MLX, GGUF), so one loaded model serves the base model (switch off: the code path
//! is exactly the one without an adapter, bit for bit) and the adapted one (switch on).
//!
//! The low-rank products are tiny next to the base matmuls (rank 16 against thousands of rows),
//! so they always run on the CPU, in f32, after the base matmul (CPU or GPU) has filled `y`.
//!
//! Keys are PEFT's: `base_model.model.<module path>.lora_A.weight` (`[r, in]`) and `.lora_B.weight`
//! (`[out, r]`). Module paths are matched from `layers.N.` on (the `model.` / `model.language_model.`
//! prefix differs between the text-only and the multimodal classes) or as `lm_head`. The token
//! embedding's `lora_embedding_A` (`[r, vocab]`) / `lora_embedding_B` (`[hidden, r]`) go on the
//! embedding table ([`EmbedDelta`]).
//!
//! llama.cpp's GGUF adapters (`convert_lora_to_gguf.py`) are read too ([`gguf`]): their tensor
//! names are mapped to the same module paths, their layout is the base GGUF file's.
//!
//! User adapters ([`attach_specs`], `--lora FILE[:SCALE]`, the engine's `lora` option) are on from
//! load with a strength (the `:SCALE`, a factor on `alpha / r` that can change at run time; 0 is
//! the base model's code path), several of them chain on the same layer, and with merging on
//! ([`LoraAdapter::set_merge`]) updates go into the weights instead where the weight allows it
//! ([`crate::model::Lin::merge_lora`]).
//!
//! Routed MoE experts (Qwen MoE: `layers.N.mlp.experts`, Gemma 4 26B-A4B: `layers.N.experts`) are
//! adapted per expert, in either layout PEFT produces for them ([`LoraAdapter::attach_experts`]):
//! - stacked parameters (`LoraConfig(target_parameters=["experts.gate_up_proj",
//!   "experts.down_proj"])` on the current transformers classes, whose experts are 3-D
//!   `[experts, out, in]` parameters): one wrapper per parameter, nested, saved as
//!   `...experts.lora_A.weight`, `...experts.base_layer.lora_A.weight`, ... with `lora_A`
//!   `[experts * r, in]` (expert `e`: rows `e * r ..`) and `lora_B` `[out, r * experts]` (expert
//!   `e`, component `k`: column `k * experts + e`). The keys do not name the parameter, so it is
//!   told from the shape (`gate_up_proj`: `[2 * inter, H]`, `down_proj`: `[H, inter]`);
//! - one module per expert (`...experts.E.gate_proj`, `.up_proj`, `.down_proj`, the layout of
//!   classes that keep each expert as an MLP of `nn.Linear`s), each an ordinary pair.
//!
//! Either layout goes on either storage of the base weights: a `gate_up_proj` update is split
//! by rows for checkpoints with separate gate and up matrices (GGUF), and `gate_proj` /
//! `up_proj` updates are stacked (rank `r_gate + r_up`) for fused ones. Not accepted (an error
//! at load, like any module that finds no layer): separate stacked `gate_proj` / `up_proj`
//! parameters (their two wrappers have the same shape, and the keys do not say which is which),
//! and both layouts on the same expert weight.

use crate::kernels::dot;
use crate::pool;
use crate::safetensors::SafeTensors;
use anyhow::{Context, Result, bail};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

mod gguf;

pub use crate::sd::lora::LoraSpec;

/// The low-rank update of one linear layer, attached to its `Lin`. Several adapters on the same
/// layer chain their updates ([`LoraDelta::push`]).
pub struct LoraDelta {
    /// `[r, n_in]`
    a: Vec<f32>,
    /// `[r, n_out]` (B transposed, so that the update is `r` contiguous axpys per token).
    bt: Vec<f32>,
    r: usize,
    n_in: usize,
    n_out: usize,
    scale: f32,
    active: Arc<AtomicBool>,
    /// The adapter's strength (`f32` bits): a factor on `scale` that can change at run time (0
    /// turns the update off).
    gain: Arc<AtomicU32>,
    /// The update of the next adapter on the same layer.
    next: Option<Box<LoraDelta>>,
}

/// The strength one update is applied with, read at run time (GPU plans copy it to the device
/// before every step, so a strength change needs no new plan).
#[derive(Clone)]
pub struct Strength {
    active: Arc<AtomicBool>,
    gain: Arc<AtomicU32>,
    scale: f32,
}

impl Strength {
    /// The strength now (0 while its adapter is off).
    pub fn get(&self) -> f32 {
        strength(&self.active, &self.gain, self.scale)
    }

    /// The adapter's factor now: its gain while on, 0 while off (the strength over the scale).
    pub fn factor(&self) -> f32 {
        strength(&self.active, &self.gain, 1.0)
    }

    /// Whether `o` always gives the same strength (the same adapter switch, strength and scale).
    pub fn same(&self, o: &Strength) -> bool {
        Arc::ptr_eq(&self.active, &o.active) && Arc::ptr_eq(&self.gain, &o.gain) && self.scale.to_bits() == o.scale.to_bits()
    }
}

/// A strength of 1 (the bits of `1.0f32`), for an adapter's [`LoraAdapter::gain`].
pub(crate) fn unit_gain() -> Arc<AtomicU32> {
    Arc::new(AtomicU32::new(1f32.to_bits()))
}

/// The gain of an adapter taken off a loaded model ([`retire`]): a NaN no strength is set to.
const RETIRED: u32 = 0x7fc0_dead;

/// Takes an adapter attached with `switch` and `gain` off its model for good: its updates stay
/// chained to their layers (and in memory) until the model is dropped, add nothing on the CPU
/// and are left out of the GPU plans built from then on ([`LoraDelta::retired`]).
pub(crate) fn retire(switch: &AtomicBool, gain: &AtomicU32) {
    switch.store(false, Ordering::SeqCst);
    gain.store(RETIRED, Ordering::SeqCst);
}

/// The strength `on` and `gain` give an update of `scale` now.
#[inline]
fn strength(on: &AtomicBool, gain: &AtomicU32, scale: f32) -> f32 {
    if !on.load(Ordering::Relaxed) {
        return 0.0;
    }
    scale * f32::from_bits(gain.load(Ordering::Relaxed))
}

impl LoraDelta {
    /// An update that is always on: `y += B (A x)` with `a` `[r, n_in]` and `b` `[n_out, r]`
    /// (strengths folded in). For LoRA files applied at load to models whose weights stay in
    /// their stored format (the FLUX transformer: mapped bf16, GGUF blocks, Q8 / Q4).
    pub fn fixed(a: Vec<f32>, b: &[f32], r: usize, n_in: usize, n_out: usize) -> Self {
        assert_eq!(a.len(), r * n_in, "LoRA down shape");
        assert_eq!(b.len(), n_out * r, "LoRA up shape");
        let mut bt = vec![0f32; r * n_out];
        for o in 0..n_out {
            for k in 0..r {
                bt[k * n_out + o] = b[o * r + k];
            }
        }
        Self {
            a,
            bt,
            r,
            n_in,
            n_out,
            scale: 1.0,
            active: Arc::new(AtomicBool::new(true)),
            gain: unit_gain(),
            next: None,
        }
    }

    /// This update switched on and off by `on` (shared by the updates of one adapter).
    pub fn with_switch(mut self, on: Arc<AtomicBool>) -> Self {
        self.active = on;
        self
    }

    /// This update's strength read from `gain` (`f32` bits, shared by the updates of one
    /// adapter) at run time.
    pub fn with_gain(mut self, gain: Arc<AtomicU32>) -> Self {
        self.gain = gain;
        self
    }

    /// The strength this update alone is applied with now (0 when its adapter is off).
    #[inline]
    pub(crate) fn strength(&self) -> f32 {
        strength(&self.active, &self.gain, self.scale)
    }

    /// This update's fixed scale (its strength is the scale times its adapter's gain while on).
    pub(crate) fn scale(&self) -> f32 {
        self.scale
    }

    /// Whether this update or one chained after it is switched on (with a nonzero strength).
    #[inline]
    pub(crate) fn is_active(&self) -> bool {
        self.strength() != 0.0 || self.next.as_ref().is_some_and(|n| n.is_active())
    }

    /// Whether this update's adapter was taken off the model ([`retire`]).
    pub(crate) fn retired(&self) -> bool {
        self.gain.load(Ordering::Relaxed) == RETIRED
    }

    /// This update and those chained after it.
    pub(crate) fn links(&self) -> impl Iterator<Item = &LoraDelta> {
        std::iter::successors(Some(self), |d| d.next.as_deref())
    }

    /// The run-time strength of this update alone.
    pub(crate) fn strength_handle(&self) -> Strength {
        Strength {
            active: self.active.clone(),
            gain: self.gain.clone(),
            scale: self.scale,
        }
    }

    /// `A` (`[r, n_in]`), `B` transposed (`[r, n_out]`) and the rank of this update alone.
    pub(crate) fn factors(&self) -> (&[f32], &[f32], usize) {
        (&self.a, &self.bt, self.r)
    }

    /// Chains `d` after this update (and those already chained).
    pub(crate) fn push(&mut self, d: Box<LoraDelta>) {
        match &mut self.next {
            Some(n) => n.push(d),
            None => self.next = Some(d),
        }
    }

    /// Parameter bytes held (f32), chained updates included.
    pub fn bytes(&self) -> usize {
        (self.a.len() + self.bt.len()) * 4 + self.next.as_ref().map_or(0, |n| n.bytes())
    }

    /// The layer's shape, `(n_out, n_in)`.
    pub(crate) fn shape(&self) -> (usize, usize) {
        (self.n_out, self.n_in)
    }

    /// `w += s * B A` (this update alone, `s` its scale times its adapter's strength) for the
    /// weight `w` (`[n_out, n_in]`, row-major): the update merged into a float copy of the weight.
    pub(crate) fn merge_into(&self, w: &mut [f32]) {
        let (r, n_in, n_out) = (self.r, self.n_in, self.n_out);
        assert_eq!(w.len(), n_out * n_in, "LoRA merge shape mismatch");
        let s = self.scale * f32::from_bits(self.gain.load(Ordering::Relaxed));
        let wp = SendPtr(w.as_mut_ptr());
        const ROWS: usize = 64;
        pool::run(n_out.div_ceil(ROWS), &|j| {
            for o in j * ROWS..((j + 1) * ROWS).min(n_out) {
                // SAFETY: each row of w is written by one item.
                let row = unsafe { std::slice::from_raw_parts_mut(wp.get().add(o * n_in), n_in) };
                for k in 0..r {
                    let c = s * self.bt[k * n_out + o];
                    for (v, &a) in row.iter_mut().zip(&self.a[k * n_in..(k + 1) * n_in]) {
                        *v += c * a;
                    }
                }
            }
        });
    }

    /// `y += scale * B (A x)` for `t` rows of `x` (`[t, n_in]`) into `y` (`[t, n_out]`), for each
    /// chained update that is switched on.
    pub(crate) fn apply(&self, x: &[f32], t: usize, y: &mut [f32]) {
        let s = self.strength();
        if s != 0.0 {
            self.apply_one(x, t, y, s);
        }
        if let Some(n) = &self.next {
            n.apply(x, t, y);
        }
    }

    /// `y += scale * B (A x)` for this update alone.
    fn apply_one(&self, x: &[f32], t: usize, y: &mut [f32], scale: f32) {
        let (r, n_in, n_out) = (self.r, self.n_in, self.n_out);
        assert_eq!(x.len(), t * n_in, "LoRA input shape mismatch");
        assert_eq!(y.len(), t * n_out, "LoRA output shape mismatch");
        if r == 0 || t == 0 {
            return;
        }
        let _timer = crate::prof::start(crate::prof::Cat::Matmul);
        // Work items cover groups of G tokens, computed in blocks of TB: each slice of A or B
        // brought into L1 serves the whole group, each register load of it a block.
        const G: usize = 16;
        let groups = t.div_ceil(G);
        let blocks = |g: usize| (g * G..((g + 1) * G).min(t)).step_by(TB).map(move |i0| (i0, (i0 + TB).min(t)));
        // u = scale * A x: [t, r], over slices of the input that keep the group's activations
        // in L1 while every rank component reads them.
        const KC: usize = 512;
        let mut u = vec![0f32; t * r];
        let up = SendPtr(u.as_mut_ptr());
        pool::run(groups, &|g| {
            let (g0, g1) = (g * G, ((g + 1) * G).min(t));
            // SAFETY: each token group of u is written by one item.
            let ug = unsafe { std::slice::from_raw_parts_mut(up.get().add(g0 * r), (g1 - g0) * r) };
            for c0 in (0..n_in).step_by(KC) {
                let c1 = (c0 + KC).min(n_in);
                for k in 0..r {
                    let ak = &self.a[k * n_in + c0..k * n_in + c1];
                    for (i0, i1) in blocks(g) {
                        let mut d = [0f32; TB];
                        dot_rows(ak, &x[i0 * n_in + c0..], n_in, i1 - i0, &mut d);
                        for (ii, v) in d[..i1 - i0].iter().enumerate() {
                            ug[(i0 - g0 + ii) * r + k] += v;
                        }
                    }
                }
            }
            for v in ug.iter_mut() {
                *v *= scale;
            }
        });
        // y += u B^T over (token group, column block) items.
        const BLK: usize = 256;
        let nb = n_out.div_ceil(BLK);
        let yp = SendPtr(y.as_mut_ptr());
        pool::run(groups * nb, &|j| {
            let (g, cb) = (j / nb, j % nb);
            let (c0, c1) = (cb * BLK, ((cb + 1) * BLK).min(n_out));
            for (i0, i1) in blocks(g) {
                let w = c1 - c0;
                // SAFETY: each (token, column block) of y is written by one item.
                let row = |i: usize| unsafe { std::slice::from_raw_parts_mut(yp.get().add(i * n_out + c0), w) };
                let bt = |k: usize, c: usize| -> &[f32; 8] { self.bt[k * n_out + c0 + c..].first_chunk::<8>().unwrap() };
                let w8 = w / 8 * 8;
                if i1 - i0 == TB {
                    // Four tokens share every 8-column slice of B.
                    let (y0, y1, y2, y3) = (row(i0), row(i0 + 1), row(i0 + 2), row(i0 + 3));
                    let us = &u[i0 * r..i1 * r];
                    update4([y0, y1, y2, y3], us, r, &self.bt[c0..], n_out, w8);
                } else {
                    for i in i0..i1 {
                        let y = row(i);
                        let ur = &u[i * r..(i + 1) * r];
                        for c in (0..w8).step_by(8) {
                            let mut acc = [0f32; 8];
                            acc.copy_from_slice(&y[c..c + 8]);
                            for (k, &uk) in ur.iter().enumerate() {
                                let b = bt(k, c);
                                for l in 0..8 {
                                    acc[l] += uk * b[l];
                                }
                            }
                            y[c..c + 8].copy_from_slice(&acc);
                        }
                    }
                }
                for i in i0..i1 {
                    let y = row(i);
                    for c in w8..w {
                        for k in 0..r {
                            y[c] += u[i * r + k] * self.bt[k * n_out + c0 + c];
                        }
                    }
                }
            }
        });
    }
}

/// The low-rank update of a token embedding table: row `t` gets `scale * B a_t`, `a_t` row `t`
/// of the adapter's `[vocab, r]` table (PEFT's `lora_embedding_A` transposed, llama.cpp's
/// `token_embd.weight.lora_a`) and `B` `[hidden, r]`. Attached to the table
/// ([`crate::model::Table::add_lora`]); always applied at run time (it costs `r * hidden` per
/// token looked up).
pub struct EmbedDelta {
    /// `[vocab, r]`
    a: Vec<f32>,
    /// `[r, hidden]`
    bt: Vec<f32>,
    r: usize,
    vocab: usize,
    hidden: usize,
    scale: f32,
    active: Arc<AtomicBool>,
    gain: Arc<AtomicU32>,
    next: Option<Box<EmbedDelta>>,
}

impl EmbedDelta {
    /// Adds the update of every chained adapter that is on to `out`, the row of token `tok`.
    pub(crate) fn add_row(&self, tok: usize, out: &mut [f32]) {
        let s = strength(&self.active, &self.gain, self.scale);
        if s != 0.0 && tok < self.vocab && out.len() == self.hidden {
            let at = &self.a[tok * self.r..(tok + 1) * self.r];
            for (k, &ak) in at.iter().enumerate() {
                let c = s * ak;
                for (o, &b) in out.iter_mut().zip(&self.bt[k * self.hidden..(k + 1) * self.hidden]) {
                    *o += c * b;
                }
            }
        }
        if let Some(n) = &self.next {
            n.add_row(tok, out);
        }
    }

    /// Chains `d` after this update.
    pub(crate) fn push(&mut self, d: Box<EmbedDelta>) {
        match &mut self.next {
            Some(n) => n.push(d),
            None => self.next = Some(d),
        }
    }

    /// Parameter bytes held (f32), chained updates included.
    pub fn bytes(&self) -> usize {
        (self.a.len() + self.bt.len()) * 4 + self.next.as_ref().map_or(0, |n| n.bytes())
    }
}

/// Tokens per block of the LoRA products.
const TB: usize = 4;

/// `d[i] = a . x[i * stride ..]` for `n <= TB` rows of `x`, reading `a` once.
#[inline]
fn dot_rows(a: &[f32], x: &[f32], stride: usize, n: usize, d: &mut [f32; TB]) {
    let m = a.len();
    let row = |i: usize| &x[i * stride..i * stride + m];
    if n < TB {
        for i in 0..n {
            d[i] = dot(a, row(i));
        }
        return;
    }
    let xs = [row(0), row(1), row(2), row(3)];
    let m8 = m / 8 * 8;
    *d = dot4(&a[..m8], xs.map(|r| &r[..m8]));
    for (i, xi) in xs.iter().enumerate() {
        for o in m8..m {
            d[i] += a[o] * xi[o];
        }
    }
}

/// `a . x[i]` for four rows as long as `a` (a multiple of 8).
#[cfg(target_arch = "aarch64")]
fn dot4(a: &[f32], x: [&[f32]; 4]) -> [f32; 4] {
    use std::arch::aarch64::*;
    let m = a.len();
    assert!(m.is_multiple_of(8) && x.iter().all(|r| r.len() == m));
    // SAFETY: every load is within `a` or a row of `x` (lengths checked); NEON is baseline on
    // aarch64. Two accumulators per row keep eight FMA chains in flight.
    unsafe {
        let z = vdupq_n_f32(0.0);
        let (mut s0, mut t0, mut s1, mut t1, mut s2, mut t2, mut s3, mut t3) = (z, z, z, z, z, z, z, z);
        let (pa, p0, p1, p2, p3) = (a.as_ptr(), x[0].as_ptr(), x[1].as_ptr(), x[2].as_ptr(), x[3].as_ptr());
        for o in (0..m).step_by(8) {
            let (a0, a1) = (vld1q_f32(pa.add(o)), vld1q_f32(pa.add(o + 4)));
            s0 = vfmaq_f32(s0, a0, vld1q_f32(p0.add(o)));
            t0 = vfmaq_f32(t0, a1, vld1q_f32(p0.add(o + 4)));
            s1 = vfmaq_f32(s1, a0, vld1q_f32(p1.add(o)));
            t1 = vfmaq_f32(t1, a1, vld1q_f32(p1.add(o + 4)));
            s2 = vfmaq_f32(s2, a0, vld1q_f32(p2.add(o)));
            t2 = vfmaq_f32(t2, a1, vld1q_f32(p2.add(o + 4)));
            s3 = vfmaq_f32(s3, a0, vld1q_f32(p3.add(o)));
            t3 = vfmaq_f32(t3, a1, vld1q_f32(p3.add(o + 4)));
        }
        [
            vaddvq_f32(vaddq_f32(s0, t0)),
            vaddvq_f32(vaddq_f32(s1, t1)),
            vaddvq_f32(vaddq_f32(s2, t2)),
            vaddvq_f32(vaddq_f32(s3, t3)),
        ]
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn dot4(a: &[f32], x: [&[f32]; 4]) -> [f32; 4] {
    // Named accumulators: arrays of them are not always kept in registers.
    let (mut s0, mut s1, mut s2, mut s3) = ([0f32; 8], [0f32; 8], [0f32; 8], [0f32; 8]);
    let it = a.as_chunks::<8>().0.iter().zip(x[0].as_chunks::<8>().0);
    let it = it
        .zip(x[1].as_chunks::<8>().0)
        .zip(x[2].as_chunks::<8>().0)
        .zip(x[3].as_chunks::<8>().0);
    for ((((a8, p0), p1), p2), p3) in it {
        for l in 0..8 {
            s0[l] += a8[l] * p0[l];
            s1[l] += a8[l] * p1[l];
            s2[l] += a8[l] * p2[l];
            s3[l] += a8[l] * p3[l];
        }
    }
    [s0, s1, s2, s3].map(|s| s.iter().sum())
}

/// `y[i][c] += sum_k u[i * r + k] * bt[k * stride + c]` for four rows `y` and the first `w8`
/// columns (a multiple of 8).
#[cfg(target_arch = "aarch64")]
fn update4(y: [&mut [f32]; 4], u: &[f32], r: usize, bt: &[f32], stride: usize, w8: usize) {
    use std::arch::aarch64::*;
    assert!(w8.is_multiple_of(8) && y.iter().all(|v| v.len() >= w8) && u.len() >= 4 * r);
    assert!(r == 0 || bt.len() >= (r - 1) * stride + w8);
    // SAFETY: every access is within the bounds checked above; NEON is baseline on aarch64.
    unsafe {
        let [y0, y1, y2, y3] = y.map(|v| v.as_mut_ptr());
        let (pu, pb) = (u.as_ptr(), bt.as_ptr());
        for c in (0..w8).step_by(8) {
            let (mut a0, mut b0) = (vld1q_f32(y0.add(c)), vld1q_f32(y0.add(c + 4)));
            let (mut a1, mut b1) = (vld1q_f32(y1.add(c)), vld1q_f32(y1.add(c + 4)));
            let (mut a2, mut b2) = (vld1q_f32(y2.add(c)), vld1q_f32(y2.add(c + 4)));
            let (mut a3, mut b3) = (vld1q_f32(y3.add(c)), vld1q_f32(y3.add(c + 4)));
            for k in 0..r {
                let (lo, hi) = (vld1q_f32(pb.add(k * stride + c)), vld1q_f32(pb.add(k * stride + c + 4)));
                let (u0, u1, u2, u3) = (*pu.add(k), *pu.add(r + k), *pu.add(2 * r + k), *pu.add(3 * r + k));
                a0 = vfmaq_n_f32(a0, lo, u0);
                b0 = vfmaq_n_f32(b0, hi, u0);
                a1 = vfmaq_n_f32(a1, lo, u1);
                b1 = vfmaq_n_f32(b1, hi, u1);
                a2 = vfmaq_n_f32(a2, lo, u2);
                b2 = vfmaq_n_f32(b2, hi, u2);
                a3 = vfmaq_n_f32(a3, lo, u3);
                b3 = vfmaq_n_f32(b3, hi, u3);
            }
            vst1q_f32(y0.add(c), a0);
            vst1q_f32(y0.add(c + 4), b0);
            vst1q_f32(y1.add(c), a1);
            vst1q_f32(y1.add(c + 4), b1);
            vst1q_f32(y2.add(c), a2);
            vst1q_f32(y2.add(c + 4), b2);
            vst1q_f32(y3.add(c), a3);
            vst1q_f32(y3.add(c + 4), b3);
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn update4(y: [&mut [f32]; 4], u: &[f32], r: usize, bt: &[f32], stride: usize, w8: usize) {
    let [y0, y1, y2, y3] = y;
    for c in (0..w8).step_by(8) {
        let ld = |y: &[f32]| -> [f32; 8] { *y[c..].first_chunk::<8>().unwrap() };
        let (mut a0, mut a1, mut a2, mut a3) = (ld(y0), ld(y1), ld(y2), ld(y3));
        for k in 0..r {
            let b = bt[k * stride + c..].first_chunk::<8>().unwrap();
            let (u0, u1, u2, u3) = (u[k], u[r + k], u[2 * r + k], u[3 * r + k]);
            for l in 0..8 {
                a0[l] += u0 * b[l];
                a1[l] += u1 * b[l];
                a2[l] += u2 * b[l];
                a3[l] += u3 * b[l];
            }
        }
        y0[c..c + 8].copy_from_slice(&a0);
        y1[c..c + 8].copy_from_slice(&a1);
        y2[c..c + 8].copy_from_slice(&a2);
        y3[c..c + 8].copy_from_slice(&a3);
    }
}

#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    #[inline(always)]
    fn get(self) -> *mut f32 {
        self.0
    }
}

/// One module of the adapter file, before it is attached.
#[derive(Clone)]
struct Pair {
    a: Vec<f32>, // [r, n_in]
    b: Vec<f32>, // [n_out, r]
    r: usize,
    n_in: usize,
    n_out: usize,
}

impl Pair {
    /// The pair without the components that cannot contribute (adapters zero-padded to a common
    /// rank; the halves of a split or stacked pair).
    fn pruned(self) -> Self {
        let (r0, n_in, n_out) = (self.r, self.n_in, self.n_out);
        let (a, b) = (&self.a, &self.b);
        let keep: Vec<usize> = (0..r0)
            .filter(|&k| a[k * n_in..(k + 1) * n_in].iter().any(|&v| v != 0.0) && (0..n_out).any(|o| b[o * r0 + k] != 0.0))
            .collect();
        if keep.len() == r0 {
            return self;
        }
        let r = keep.len();
        let mut ak = Vec::with_capacity(r * n_in);
        for &k in &keep {
            ak.extend_from_slice(&a[k * n_in..(k + 1) * n_in]);
        }
        let mut bk = Vec::with_capacity(n_out * r);
        for o in 0..n_out {
            bk.extend(keep.iter().map(|&k| b[o * r0 + k]));
        }
        Self {
            a: ak,
            b: bk,
            r,
            n_in,
            n_out,
        }
    }

    /// Expert `e` of a PEFT stacked-parameter pair over `n_exp` experts (see the module docs).
    fn expert(&self, e: usize, n_exp: usize) -> Self {
        let (r, n_in, n_out) = (self.r / n_exp, self.n_in, self.n_out);
        let a = self.a[e * r * n_in..(e + 1) * r * n_in].to_vec();
        let mut b = Vec::with_capacity(n_out * r);
        for o in 0..n_out {
            b.extend((0..r).map(|k| self.b[o * self.r + k * n_exp + e]));
        }
        Self { a, b, r, n_in, n_out }.pruned()
    }

    /// The pairs of output rows `..n` and `n..` (the same `A`).
    fn split_out(self, n: usize) -> (Self, Self) {
        let (r, n_in) = (self.r, self.n_in);
        let (b0, b1) = self.b.split_at(n * r);
        let half = |b: &[f32], n_out: usize| {
            Self {
                a: self.a.clone(),
                b: b.to_vec(),
                r,
                n_in,
                n_out,
            }
            .pruned()
        };
        (half(b0, n), half(b1, self.n_out - n))
    }

    /// The pair of two layers stacked by output rows (`first`'s `n_first` rows, then `second`'s
    /// `n_second`; a missing one updates nothing): rank `r_first + r_second`.
    fn stack_out(first: Option<Self>, second: Option<Self>, n_first: usize, n_second: usize, n_in: usize) -> Self {
        let (r0, r1) = (first.as_ref().map_or(0, |p| p.r), second.as_ref().map_or(0, |p| p.r));
        let r = r0 + r1;
        let mut a = Vec::with_capacity(r * n_in);
        let mut b = vec![0f32; (n_first + n_second) * r];
        for (p, row0, k0) in [(&first, 0, 0), (&second, n_first, r0)] {
            let Some(p) = p else { continue };
            a.extend_from_slice(&p.a);
            for o in 0..p.n_out {
                b[(row0 + o) * r + k0..(row0 + o) * r + k0 + p.r].copy_from_slice(&p.b[o * p.r..(o + 1) * p.r]);
            }
        }
        Self {
            a,
            b,
            r,
            n_in,
            n_out: n_first + n_second,
        }
    }
}

/// Whether the canonical module `key` is a PEFT wrapper of a stacked expert parameter
/// (`...experts`, `...experts.base_layer`, ...).
fn is_stacked(key: &str) -> bool {
    let mut k = key;
    while let Some(rest) = k.strip_suffix(".base_layer") {
        k = rest;
    }
    k.ends_with(".experts")
}

/// A PEFT LoRA adapter, loaded and waiting to be attached to a model ([`LoraAdapter::take`] by
/// module, then [`LoraAdapter::finish`]).
#[derive(Clone)]
pub struct LoraAdapter {
    /// The adapter folder (PEFT) or file (GGUF).
    pub dir: PathBuf,
    /// Rank from the config (zero-padded components are dropped per module on load); a GGUF
    /// adapter's largest rank.
    pub rank: usize,
    pub alpha: f32,
    /// `alpha / r` (or `alpha / sqrt(r)` with rsLoRA). A GGUF adapter whose tensors have
    /// different ranks has each one's `alpha / r` folded into its `B` at load and 1 here.
    pub scale: f32,
    pub target_modules: Vec<String>,
    /// `peft` or `gguf`.
    pub format: &'static str,
    /// GGUF adapters: the tensors that were skipped (norm vectors, which llama.cpp ignores too).
    pub skipped: Vec<String>,
    modules: BTreeMap<String, Pair>,
    /// The token embedding's update.
    embed: Option<EmbedPair>,
    /// The tensors are in the layout of the base model's GGUF file (a llama.cpp adapter): rows
    /// and columns a GGUF checkpoint reorders are already in its order.
    stored_order: bool,
    active: Arc<AtomicBool>,
    /// The adapter's strength (the user scale), shared by its attached updates.
    gain: Arc<AtomicU32>,
    /// Merge the updates into the weights at attach time where the weight allows it.
    merge: bool,
    attached: usize,
    merged: usize,
    bytes: usize,
}

/// An embedding update before it is attached: `a` `[vocab, r]`, `b` `[hidden, r]`.
#[derive(Clone)]
struct EmbedPair {
    a: Vec<f32>,
    b: Vec<f32>,
    r: usize,
    vocab: usize,
    hidden: usize,
}

/// Whether `dir` holds a PEFT adapter.
pub fn is_adapter_dir(dir: &Path) -> bool {
    dir.join("adapter_config.json").is_file() && dir.join("adapter_model.safetensors").is_file()
}

/// Whether PEFT module path `module` is the token embedding of the language model.
fn is_embedding(module: &str) -> bool {
    let m = module.strip_prefix("base_model.model.").unwrap_or(module);
    ["model.language_model.", "language_model.model.", "language_model.", "model.", ""]
        .iter()
        .any(|p| m.strip_prefix(p) == Some("embed_tokens"))
        || m == "model.llm.embed"
}

/// The module path of a PEFT key, from `layers.` on, or `lm_head`, or for the MTP layer of
/// Qwen3.5 / 3.8 `mtp.layers.0. ...` and `mtp.fc`; None for keys of other parts (vision towers,
/// embeddings).
fn canonical(module: &str) -> Option<String> {
    let m = module.strip_prefix("base_model.model.").unwrap_or(module);
    if m == "lm_head" || m.ends_with(".lm_head") && !m.contains("layers.") {
        return Some("lm_head".into());
    }
    for p in ["model.mtp.", "mtp."] {
        if let Some(rest) = m.strip_prefix(p)
            && (rest.starts_with("layers.") || rest == "fc")
        {
            return Some(format!("mtp.{rest}"));
        }
    }
    // (`model.llm.`: Inkling-2.)
    for p in [
        "model.language_model.",
        "language_model.model.",
        "language_model.",
        "model.llm.",
        "model.",
        "",
    ] {
        if let Some(rest) = m.strip_prefix(p)
            && rest.starts_with("layers.")
        {
            return Some(rest.to_string());
        }
    }
    None
}

/// A float tensor of the adapter file as f32.
fn tensor_f32(st: &SafeTensors, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let info = st.info(name).with_context(|| format!("missing tensor {name}"))?;
    let b = st.bytes(name)?;
    let v = match info.dtype.as_str() {
        "F32" => b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect(),
        "BF16" => b
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| crate::kernels::bf16_to_f32(u16::from_le_bytes(*c)))
            .collect(),
        "F16" => b.as_chunks::<2>().0.iter().map(|c| half::f16::from_le_bytes(*c).to_f32()).collect(),
        d => bail!("LoRA tensor {name} has dtype {d}, expected a float type"),
    };
    Ok((v, info.shape.clone()))
}

impl LoraAdapter {
    /// Loads the adapter in `dir` (all tensors as f32).
    pub fn load(dir: &Path) -> Result<Self> {
        let cfg_path = dir.join("adapter_config.json");
        let cfg: serde_json::Value =
            serde_json::from_slice(&crate::fsx::read(&cfg_path)?).with_context(|| format!("parse {}", cfg_path.display()))?;
        let get_str = |k: &str| cfg.get(k).and_then(|v| v.as_str()).unwrap_or_default().to_string();
        let peft_type = get_str("peft_type");
        if !peft_type.is_empty() && peft_type != "LORA" {
            bail!("{}: peft_type {peft_type} (only LORA adapters are supported)", cfg_path.display());
        }
        let flag = |k: &str| cfg.get(k).and_then(|v| v.as_bool()).unwrap_or(false);
        for k in ["use_dora", "fan_in_fan_out", "use_qalora", "lora_bias"] {
            if flag(k) {
                bail!("{}: {k} is not supported", cfg_path.display());
            }
        }
        if !matches!(get_str("bias").as_str(), "" | "none") {
            bail!("{}: bias {:?} is not supported", cfg_path.display(), get_str("bias"));
        }
        for k in ["rank_pattern", "alpha_pattern"] {
            if cfg.get(k).and_then(|v| v.as_object()).is_some_and(|o| !o.is_empty()) {
                bail!("{}: per-module {k} is not supported", cfg_path.display());
            }
        }
        if cfg
            .get("modules_to_save")
            .is_some_and(|v| v.as_array().is_some_and(|a| !a.is_empty()))
        {
            bail!("{}: modules_to_save (fully trained modules) is not supported", cfg_path.display());
        }
        let rank = cfg.get("r").and_then(|v| v.as_u64()).context("adapter_config.json: missing r")? as usize;
        let alpha = cfg.get("lora_alpha").and_then(|v| v.as_f64()).unwrap_or(rank as f64) as f32;
        let scale = if flag("use_rslora") {
            alpha / (rank as f32).sqrt()
        } else {
            alpha / rank as f32
        };
        let target_modules = match cfg.get("target_modules") {
            Some(serde_json::Value::Array(a)) => a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect(),
            Some(serde_json::Value::String(s)) => vec![s.clone()],
            _ => Vec::new(),
        };

        let path = dir.join("adapter_model.safetensors");
        let st = SafeTensors::open(&path, true)?;
        let mut halves: BTreeMap<String, (Option<String>, Option<String>)> = BTreeMap::new();
        // The embedding's `lora_embedding_A` (`[r, vocab]`) and `lora_embedding_B` (`[hidden, r]`).
        let mut emb: (Option<String>, Option<String>) = (None, None);
        for name in st.names() {
            let ea = name
                .strip_suffix(".lora_embedding_A")
                .or_else(|| name.strip_suffix(".lora_embedding_A.default"));
            let eb = name
                .strip_suffix(".lora_embedding_B")
                .or_else(|| name.strip_suffix(".lora_embedding_B.default"));
            if let Some(m) = ea.or(eb) {
                if !is_embedding(m) {
                    bail!("{}: {name} is not the language model's token embedding", path.display());
                }
                let slot = if ea.is_some() { &mut emb.0 } else { &mut emb.1 };
                if slot.replace(name.clone()).is_some() {
                    bail!("{}: two embedding updates", path.display());
                }
                continue;
            }
            let (module, is_a) = if let Some(m) = name
                .strip_suffix(".lora_A.weight")
                .or_else(|| name.strip_suffix(".lora_A.default.weight"))
            {
                (m, true)
            } else if let Some(m) = name
                .strip_suffix(".lora_B.weight")
                .or_else(|| name.strip_suffix(".lora_B.default.weight"))
            {
                (m, false)
            } else {
                bail!("{}: unsupported tensor {name} (expected lora_A / lora_B weights)", path.display());
            };
            let e = halves.entry(module.to_string()).or_default();
            if is_a {
                e.0 = Some(name.clone());
            } else {
                e.1 = Some(name.clone());
            }
        }
        let mut modules = BTreeMap::new();
        let mut bytes = 0;
        for (module, (an, bn)) in halves {
            let (Some(an), Some(bn)) = (an, bn) else {
                bail!("{}: {module} lacks its lora_A or lora_B weight", path.display());
            };
            let key =
                canonical(&module).with_context(|| format!("{}: module {module} is not a decoder layer or lm_head", path.display()))?;
            let (a, ash) = tensor_f32(&st, &an)?;
            let (b, bsh) = tensor_f32(&st, &bn)?;
            if ash.len() != 2 || bsh.len() != 2 || ash[0] != bsh[1] {
                bail!("{module}: lora_A {ash:?} and lora_B {bsh:?} do not form a low-rank pair");
            }
            let pair = Pair {
                a,
                b,
                r: ash[0],
                n_in: ash[1],
                n_out: bsh[0],
            };
            // Drop components that cannot contribute (adapters zero-padded to a common rank);
            // those of a stacked expert parameter keep their places until the experts are
            // taken apart.
            let pair = if is_stacked(&key) { pair } else { pair.pruned() };
            bytes += (pair.a.len() + pair.b.len()) * 4;
            if modules.insert(key.clone(), pair).is_some() {
                bail!("{}: module {key} appears twice", path.display());
            }
        }
        let embed = match emb {
            (None, None) => None,
            (Some(an), Some(bn)) => {
                let (at, ash) = tensor_f32(&st, &an)?;
                let (b, bsh) = tensor_f32(&st, &bn)?;
                if ash.len() != 2 || bsh.len() != 2 || ash[0] != bsh[1] {
                    bail!(
                        "{}: lora_embedding_A {ash:?} and lora_embedding_B {bsh:?} do not form a low-rank pair",
                        path.display()
                    );
                }
                let (r, vocab, hidden) = (ash[0], ash[1], bsh[0]);
                let mut a = vec![0f32; vocab * r];
                for k in 0..r {
                    for t in 0..vocab {
                        a[t * r + k] = at[k * vocab + t];
                    }
                }
                bytes += (a.len() + b.len()) * 4;
                Some(EmbedPair { a, b, r, vocab, hidden })
            }
            _ => bail!(
                "{}: the embedding update lacks its lora_embedding_A or lora_embedding_B",
                path.display()
            ),
        };
        if modules.is_empty() && embed.is_none() {
            bail!("{}: no LoRA modules", path.display());
        }
        let mut ad = Self::empty(dir, "peft", rank, alpha, scale);
        ad.target_modules = target_modules;
        ad.modules = modules;
        ad.embed = embed;
        ad.bytes = bytes;
        Ok(ad)
    }

    /// An adapter without modules, switched off.
    fn empty(dir: &Path, format: &'static str, rank: usize, alpha: f32, scale: f32) -> Self {
        Self {
            dir: dir.to_path_buf(),
            rank,
            alpha,
            scale,
            target_modules: Vec::new(),
            format,
            skipped: Vec::new(),
            modules: BTreeMap::new(),
            embed: None,
            stored_order: false,
            active: Arc::new(AtomicBool::new(false)),
            gain: unit_gain(),
            merge: false,
            attached: 0,
            merged: 0,
            bytes: 0,
        }
    }

    /// Loads a llama.cpp LoRA adapter (`general.type = adapter`, `adapter.type = lora`, as
    /// `convert_lora_to_gguf.py` writes them) for the base model whose GGUF header is `base`:
    /// see [`gguf`]. Its tensors are named after the base file's (`blk.N.attn_output.weight.lora_a`)
    /// and laid out as stored there, so it only goes on a GGUF base of the same architecture.
    pub fn load_gguf(path: &Path, base: &crate::gguf::Header) -> Result<Self> {
        gguf::load(path, base)
    }

    /// A PEFT adapter folder (or its `adapter_model.safetensors`), or a llama.cpp GGUF adapter
    /// file, for the base model at `base` (a directory or a `.gguf` file).
    pub fn open(path: &Path, base: &Path) -> Result<Self> {
        if crate::gguf::is_gguf(path) {
            if !crate::gguf::is_gguf(base) {
                bail!(
                    "{}: a llama.cpp (GGUF) LoRA adapter goes on a GGUF base model (its tensors are laid out as the base file's), \
                     not on {}; use the adapter's PEFT form (adapter_config.json + adapter_model.safetensors) for other checkpoints",
                    path.display(),
                    base.display()
                );
            }
            let h = crate::gguf::header(base)?;
            return Self::load_gguf(path, &h);
        }
        let dir = if path.is_file() && path.file_name().is_some_and(|n| n == "adapter_model.safetensors") {
            path.parent().unwrap_or(Path::new("."))
        } else {
            path
        };
        if !is_adapter_dir(dir) {
            bail!(
                "{}: not a language-model LoRA adapter (a PEFT folder with adapter_config.json and adapter_model.safetensors, \
                 or a llama.cpp GGUF adapter)",
                path.display()
            );
        }
        Self::load(dir)
    }

    /// Sets the strength (the user scale: the updates are `strength * scale * B A x`), and
    /// switches the adapter on for good (unlike the decision adapters, which are switched per
    /// request).
    pub fn set_strength(&mut self, s: f32) {
        self.gain.store(s.to_bits(), Ordering::Relaxed);
        self.active.store(true, Ordering::Relaxed);
    }

    /// Merge the updates into the weights as they are attached (see [`LoraAdapter::put`]).
    pub fn set_merge(&mut self, on: bool) {
        self.merge = on;
    }

    /// The strength handle shared by the attached updates (`f32` bits).
    pub fn gain(&self) -> Arc<AtomicU32> {
        self.gain.clone()
    }

    /// How many of the attached updates were merged into their weights.
    pub fn merged(&self) -> usize {
        self.merged
    }

    /// Attaches update `d` to `lin`: merged into the weight when merging is on and the weight
    /// allows it ([`crate::model::Lin::merge_lora`]), else chained as a runtime update.
    pub(crate) fn put(&mut self, lin: &mut crate::model::Lin, d: Box<LoraDelta>) -> Result<()> {
        if self.merge && lin.merge_lora(&d)? {
            self.merged += 1;
            return Ok(());
        }
        lin.add_lora(d);
        Ok(())
    }

    /// Takes the token embedding's update for a table of `vocab x hidden` (None when the
    /// adapter has none). Errors on another shape.
    pub fn take_embed(&mut self, vocab: usize, hidden: usize) -> Result<Option<Box<EmbedDelta>>> {
        let Some(p) = self.embed.take() else {
            return Ok(None);
        };
        if p.vocab != vocab || p.hidden != hidden {
            bail!(
                "LoRA embedding update is [{} x {}] but the model's embedding is [{vocab} x {hidden}]: the adapter is for another base model",
                p.vocab,
                p.hidden
            );
        }
        let mut bt = vec![0f32; p.r * hidden];
        for o in 0..hidden {
            for k in 0..p.r {
                bt[k * hidden + o] = p.b[o * p.r + k];
            }
        }
        self.attached += 1;
        Ok(Some(Box::new(EmbedDelta {
            a: p.a,
            bt,
            r: p.r,
            vocab,
            hidden,
            scale: self.scale,
            active: self.active.clone(),
            gain: self.gain.clone(),
            next: None,
        })))
    }

    /// The switch shared by every attached module (off at load).
    pub fn switch(&self) -> Arc<AtomicBool> {
        self.active.clone()
    }

    /// Moves the MTP layer's modules (`mtp.*`) into an adapter of their own, which shares this
    /// one's strength (None when there are none): the MTP layer loads after the model.
    pub(crate) fn split_mtp(&mut self) -> Option<Self> {
        let keys: Vec<String> = self.modules.keys().filter(|k| k.starts_with("mtp.")).cloned().collect();
        if keys.is_empty() {
            return None;
        }
        let mut m = Self::empty(&self.dir, self.format, self.rank, self.alpha, self.scale);
        (m.stored_order, m.merge) = (self.stored_order, self.merge);
        (m.active, m.gain) = (self.active.clone(), self.gain.clone());
        for k in keys {
            let p = self.modules.remove(&k).unwrap();
            m.bytes += (p.a.len() + p.b.len()) * 4;
            m.modules.insert(k, p);
        }
        Some(m)
    }

    /// The largest `layers.N` index the adapter touches, plus one.
    pub fn layer_count(&self) -> usize {
        self.modules
            .keys()
            .filter_map(|k| k.strip_prefix("layers.")?.split('.').next()?.parse::<usize>().ok())
            .max()
            .map_or(0, |l| l + 1)
    }

    /// Whether the adapter has a module `name` (`layers.N.mlp.down_proj`, `lm_head`).
    pub fn has(&self, name: &str) -> bool {
        self.modules.contains_key(name)
    }

    /// Takes the update of module `name` for a layer of `n_out x n_in`, or None when the adapter
    /// does not target it. `rows[p]` / `cols[p]`, when given, name the adapter's row / column
    /// that the model stores at `p` (checkpoints that reorder heads, e.g. GGUF); a GGUF adapter
    /// is already in the stored order and ignores them.
    pub fn take(
        &mut self,
        name: &str,
        n_out: usize,
        n_in: usize,
        rows: Option<&[usize]>,
        cols: Option<&[usize]>,
    ) -> Result<Option<Box<LoraDelta>>> {
        let Some(p) = self.take_pair(name, n_out, n_in)? else {
            return Ok(None);
        };
        Ok(Some(self.delta(p, rows, cols)))
    }

    /// [`LoraAdapter::take`], attaching the update to `lin` ([`LoraAdapter::put`]).
    pub(crate) fn attach(
        &mut self,
        lin: &mut crate::model::Lin,
        name: &str,
        n_out: usize,
        n_in: usize,
        rows: Option<&[usize]>,
        cols: Option<&[usize]>,
    ) -> Result<()> {
        if let Some(d) = self.take(name, n_out, n_in, rows, cols)? {
            self.put(lin, d)?;
        }
        Ok(())
    }

    /// The pair of module `name`, checked against a layer of `n_out x n_in`.
    fn take_pair(&mut self, name: &str, n_out: usize, n_in: usize) -> Result<Option<Pair>> {
        let Some(p) = self.modules.remove(name) else {
            return Ok(None);
        };
        if p.n_out != n_out || p.n_in != n_in {
            bail!(
                "LoRA module {name} is [{} x {}] but the model's layer is [{n_out} x {n_in}]: the adapter is for another base model",
                p.n_out,
                p.n_in
            );
        }
        Ok(Some(p))
    }

    /// The update of `p`, counted as attached (`rows` / `cols` as in [`LoraAdapter::take`]).
    fn delta(&mut self, p: Pair, rows: Option<&[usize]>, cols: Option<&[usize]>) -> Box<LoraDelta> {
        let (rows, cols) = if self.stored_order { (None, None) } else { (rows, cols) };
        let (r, n_in, n_out) = (p.r, p.n_in, p.n_out);
        let mut a = p.a;
        if let Some(cols) = cols {
            let src = a.clone();
            for k in 0..r {
                for (dst, &c) in cols.iter().enumerate() {
                    a[k * n_in + dst] = src[k * n_in + c];
                }
            }
        }
        let mut bt = vec![0f32; r * n_out];
        for o in 0..n_out {
            let src = rows.map_or(o, |rs| rs[o]);
            for k in 0..r {
                bt[k * n_out + o] = p.b[src * r + k];
            }
        }
        self.attached += 1;
        Box::new(LoraDelta {
            a,
            bt,
            r,
            n_in,
            n_out,
            scale: self.scale,
            active: self.active.clone(),
            gain: self.gain.clone(),
            next: None,
        })
    }

    /// Attaches the adapter's modules for the routed experts `ex` of a MoE layer, whose module
    /// path is `name` (`layers.N.mlp.experts`, `layers.N.experts`), in either PEFT layout (see
    /// the module docs) or one module per expert with fused gate/up rows (`name.E.gate_up_proj`,
    /// how a GGUF adapter's `ffn_gate_up_exps` stack is taken apart): each expert's weights get
    /// their own update, applied to the tokens routed to that expert. Errors on a module that
    /// fits no expert weight.
    pub(crate) fn attach_experts(&mut self, ex: &mut crate::moe::Experts, name: &str, h: usize) -> Result<()> {
        use crate::moe::GateUp;
        let (ne, mi) = (ex.n_experts(), ex.inter);
        // Per expert: gate, up (or `fused`: both as one), down.
        let mut pairs: Vec<[Option<Pair>; 3]> = (0..ne).map(|_| [None, None, None]).collect();
        let mut fused: Vec<Option<Pair>> = (0..ne).map(|_| None).collect();
        // Stacked parameters: `name`, `name.base_layer`, ... (one wrapper per parameter).
        let mut key = name.to_string();
        let mut seen = [false; 2];
        while let Some(p) = self.modules.remove(&key) {
            let which = match (p.n_out, p.n_in) {
                (o, i) if o == 2 * mi && i == h => 0,
                (o, i) if o == h && i == mi => 1,
                (o, i) if o == mi && i == h => bail!(
                    "LoRA module {key} is for a stacked [{ne}, {o}, {i}] parameter: separate stacked gate_proj / up_proj \
                     parameters are not supported (their adapters cannot be told apart)"
                ),
                (o, i) => bail!(
                    "LoRA module {key} is for a stacked [{ne}, {o}, {i}] parameter, but the model's experts are \
                     [{ne}, {}, {h}] (gate_up_proj) and [{ne}, {h}, {mi}] (down_proj): the adapter is for another base model",
                    2 * mi
                ),
            };
            if p.r != ne * self.rank {
                bail!(
                    "LoRA module {key} has {} components, expected {} (rank {} for each of the model's {ne} experts): \
                     the adapter is for another base model",
                    p.r,
                    ne * self.rank,
                    self.rank
                );
            }
            if std::mem::replace(&mut seen[which], true) {
                bail!("LoRA module {key}: two adapters for the same stacked expert parameter");
            }
            for e in 0..ne {
                let pe = p.expert(e, ne);
                if which == 0 {
                    fused[e] = Some(pe);
                } else {
                    pairs[e][2] = Some(pe);
                }
            }
            key.push_str(".base_layer");
        }
        // One module per expert.
        for (e, (pe, fe)) in pairs.iter_mut().zip(fused.iter_mut()).enumerate() {
            for (i, (m, n_out, n_in)) in [("gate_proj", mi, h), ("up_proj", mi, h), ("down_proj", h, mi)]
                .into_iter()
                .enumerate()
            {
                let key = format!("{name}.{e}.{m}");
                if let Some(p) = self.take_pair(&key, n_out, n_in)? {
                    if (i < 2 && fe.is_some()) || pe[i].is_some() {
                        bail!("LoRA module {key}: the adapter also targets the stacked parameter of this weight");
                    }
                    pe[i] = Some(p);
                }
            }
            let key = format!("{name}.{e}.gate_up_proj");
            if let Some(p) = self.take_pair(&key, 2 * mi, h)? {
                if fe.is_some() || pe[0].is_some() || pe[1].is_some() {
                    bail!("LoRA module {key}: the adapter targets this expert's gate / up weights twice");
                }
                *fe = Some(p);
            }
        }
        // Interleaved gate/up rows (Inkling-2's `w13_weight`): stored row `2j` is gate row `j`,
        // `2j + 1` up row `j`.
        let interleave: Option<Vec<usize>> = ex
            .interleaved
            .then(|| (0..2 * mi).map(|p| if p % 2 == 0 { p / 2 } else { mi + p / 2 }).collect());
        for (e, (([gate, up, down], fused), gu)) in pairs.into_iter().zip(fused).zip(ex.gate_up.iter_mut()).enumerate() {
            match gu {
                GateUp::Fused(l) => {
                    let p = match fused {
                        Some(p) => Some(p),
                        None if gate.is_some() || up.is_some() => Some(Pair::stack_out(gate, up, mi, mi, h)),
                        None => None,
                    };
                    if let Some(p) = p {
                        let d = self.delta(p, interleave.as_deref(), None);
                        self.put(l, d)?;
                    }
                }
                GateUp::Split(g, u) => {
                    let (gate, up) = match fused {
                        Some(p) => {
                            let (a, b) = p.split_out(mi);
                            (Some(a), Some(b))
                        }
                        None => (gate, up),
                    };
                    if let Some(p) = gate {
                        let d = self.delta(p, None, None);
                        self.put(g, d)?;
                    }
                    if let Some(p) = up {
                        let d = self.delta(p, None, None);
                        self.put(u, d)?;
                    }
                }
            }
            if let Some(p) = down {
                let d = self.delta(p, None, None);
                self.put(&mut ex.down[e], d)?;
            }
        }
        Ok(())
    }

    /// Checks that every module found a layer; returns the number attached.
    pub fn finish(&self) -> Result<usize> {
        if let Some(k) = self.modules.keys().next() {
            bail!(
                "LoRA adapter {}: {} module(s) match no layer of the model (first: {k})",
                self.dir.display(),
                self.modules.len()
            );
        }
        if self.embed.is_some() {
            bail!(
                "LoRA adapter {}: it updates the token embedding, which this model does not take an update on",
                self.dir.display()
            );
        }
        Ok(self.attached)
    }

    /// Parameter bytes (f32) of the loaded modules.
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

/// A language model that runtime LoRA adapters attach to (by Hugging Face module name, see the
/// module docs): Qwen / Llama ([`crate::qwen::QwenModel`]), Gemma ([`crate::model::Model`]),
/// Inkling-2 ([`crate::inkling::Inkling2Model`]).
pub trait Adaptable {
    /// Attaches the modules of `ad` to the layers they target and returns how many were attached
    /// (errors when one matches no layer or has another shape).
    fn attach_lora(&mut self, ad: &mut LoraAdapter) -> Result<usize>;
}

impl Adaptable for crate::qwen::QwenModel<'_> {
    fn attach_lora(&mut self, ad: &mut LoraAdapter) -> Result<usize> {
        crate::qwen::QwenModel::attach_lora(self, ad)
    }
}

impl Adaptable for crate::model::Model<'_> {
    fn attach_lora(&mut self, ad: &mut LoraAdapter) -> Result<usize> {
        crate::model::Model::attach_lora(self, ad)
    }
}

impl Adaptable for crate::inkling::Inkling2Model<'_> {
    fn attach_lora(&mut self, ad: &mut LoraAdapter) -> Result<usize> {
        crate::inkling::Inkling2Model::attach_lora(self, ad)
    }
}

/// An adapter attached by [`attach_specs`].
#[derive(Clone, Debug)]
pub struct Attached {
    /// The adapter folder (PEFT) or file (GGUF).
    pub path: PathBuf,
    /// `peft` or `gguf`.
    pub format: &'static str,
    pub rank: usize,
    pub alpha: f32,
    /// The strength it was loaded with (the `:SCALE` of `FILE[:SCALE]`).
    pub strength: f32,
    /// Updates attached (layers, experts, the embedding), and of those merged into the weights.
    pub modules: usize,
    pub merged: usize,
    /// Parameter bytes (f32) held for the runtime updates.
    pub bytes: usize,
    /// Tensors of the file that were skipped (norm vectors of GGUF adapters).
    pub skipped: Vec<String>,
    gain: Arc<AtomicU32>,
}

impl Attached {
    /// The strength now.
    pub fn strength_now(&self) -> f32 {
        f32::from_bits(self.gain.load(Ordering::Relaxed))
    }

    /// Changes the strength of the runtime updates (0: off, exactly the base model). Merged
    /// updates are in the weights and cannot change: an error when there are any.
    pub fn set_strength(&self, s: f32) -> Result<()> {
        if self.merged > 0 {
            bail!(
                "LoRA {}: {} of its updates are merged into the weights (lora_merge), so its strength is fixed",
                self.path.display(),
                self.merged
            );
        }
        if !s.is_finite() {
            bail!("LoRA strength {s} is not a finite number");
        }
        self.gain.store(s.to_bits(), Ordering::Relaxed);
        Ok(())
    }

    /// A one-line description for logs.
    pub fn summary(&self) -> String {
        let mut s = format!(
            "{} ({}, rank {}, alpha {}, strength {}): {} update(s)",
            self.path.display(),
            self.format,
            self.rank,
            self.alpha,
            self.strength,
            self.modules
        );
        if self.merged > 0 {
            s.push_str(&format!(", {} merged into the weights", self.merged));
        }
        s.push_str(&format!(", {:.1} MB", self.bytes as f64 / 1e6));
        if !self.skipped.is_empty() {
            s.push_str(&format!(", {} tensor(s) skipped", self.skipped.len()));
        }
        s
    }
}

/// Loads each adapter of `specs` (a PEFT folder or a llama.cpp GGUF adapter; see
/// [`LoraAdapter::open`]) for the base model at `base` and attaches it to `model` with its
/// strength, switched on; with `merge`, updates go into the weights where they allow it
/// ([`crate::model::Lin::merge_lora`]).
pub fn attach_specs<M: Adaptable + ?Sized>(model: &mut M, specs: &[LoraSpec], base: &Path, merge: bool) -> Result<Vec<Attached>> {
    let mut out = Vec::with_capacity(specs.len());
    for s in specs {
        let mut ad = LoraAdapter::open(&s.path, base).with_context(|| format!("LoRA {}", s.path.display()))?;
        ad.set_strength(s.scale);
        ad.set_merge(merge);
        let n = model
            .attach_lora(&mut ad)
            .with_context(|| format!("attach LoRA {}", s.path.display()))?;
        out.push(Attached {
            path: ad.dir.clone(),
            format: ad.format,
            rank: ad.rank,
            alpha: ad.alpha,
            strength: s.scale,
            modules: n,
            merged: ad.merged,
            bytes: ad.bytes,
            skipped: ad.skipped.clone(),
            gain: ad.gain(),
        });
    }
    Ok(out)
}

/// An adapter attached to a model after its load, behind a switch of its own
/// ([`attach_switched`]): off until [`Switched::set`] turns it on.
#[derive(Clone, Debug)]
pub struct Switched {
    pub adapter: Attached,
    switch: Arc<AtomicBool>,
}

impl Switched {
    /// Turns the adapter's updates on or off (off: exactly the base model on the CPU; the GPU
    /// plans read the switch before every step).
    pub fn set(&self, on: bool) {
        if !self.retired() {
            self.switch.store(on, Ordering::SeqCst);
        }
    }

    /// Whether the adapter is on.
    pub fn is_on(&self) -> bool {
        self.switch.load(Ordering::SeqCst)
    }

    /// Takes the adapter off the model for good (see [`retire`]).
    pub(crate) fn retire(&self) {
        retire(&self.switch, &self.adapter.gain);
    }

    fn retired(&self) -> bool {
        self.adapter.gain.load(Ordering::Relaxed) == RETIRED
    }
}

/// Loads the adapter at `path` (see [`LoraAdapter::open`]) for the base model at `base` and
/// attaches it to `model` as runtime updates of `strength`, switched off. When the adapter does
/// not fit the model, the updates attached before the failure are retired ([`retire`]): the
/// model computes what it did before.
pub fn attach_switched<M: Adaptable + ?Sized>(model: &mut M, path: &Path, base: &Path, strength: f32) -> Result<Switched> {
    if !strength.is_finite() {
        bail!("LoRA strength {strength} is not a finite number");
    }
    let mut ad = LoraAdapter::open(path, base).with_context(|| format!("LoRA {}", path.display()))?;
    ad.set_strength(strength);
    let switch = ad.switch();
    switch.store(false, Ordering::SeqCst);
    let n = match model.attach_lora(&mut ad) {
        Ok(n) => n,
        Err(e) => {
            retire(&switch, &ad.gain);
            return Err(e.context(format!("attach LoRA {}", path.display())));
        }
    };
    Ok(Switched {
        adapter: Attached {
            path: ad.dir.clone(),
            format: ad.format,
            rank: ad.rank,
            alpha: ad.alpha,
            strength,
            modules: n,
            merged: 0,
            bytes: ad.bytes,
            skipped: ad.skipped.clone(),
            gain: ad.gain(),
        },
        switch,
    })
}

/// Resolves the path of a language-model LoRA spec: a local file or folder as it is; with the
/// `hf` feature, `org/name/path/in/repo` (a file, e.g. a GGUF adapter) or a repo id (`org/name`,
/// `org/name/subfolder`) holding a PEFT adapter, fetched into the cache. Other repos fall back to
/// [`crate::engine::resolve_file`] (a repo holding a single safetensors file).
pub fn resolve(spec: &str, cache_dir: Option<&Path>) -> Result<PathBuf> {
    let p = Path::new(spec);
    if p.exists() {
        return Ok(p.to_path_buf());
    }
    #[cfg(feature = "hf")]
    {
        let parts: Vec<&str> = spec.split('/').collect();
        let file_like = parts.last().is_some_and(|l| l.contains('.'));
        if parts.len() >= 2 && !file_like && !spec.starts_with('/') && !spec.contains(':') {
            let repo = format!("{}/{}", parts[0], parts[1]);
            let sub = parts[2..].join("/");
            let names: Vec<String> = ["adapter_config.json", "adapter_model.safetensors"]
                .iter()
                .map(|n| if sub.is_empty() { n.to_string() } else { format!("{sub}/{n}") })
                .collect();
            let refs: Vec<&str> = names.iter().map(String::as_str).collect();
            if let Ok(dir) = crate::hf::resolve_files(&repo, &refs, cache_dir) {
                let dir = if sub.is_empty() { dir } else { dir.join(&sub) };
                if is_adapter_dir(&dir) {
                    return Ok(dir);
                }
            }
        }
    }
    crate::engine::resolve_file(spec, cache_dir)
}

#[cfg(test)]
impl LoraAdapter {
    /// An adapter of rank `rank` holding `modules` (`(name, lora_A [r, n_in], lora_B [n_out, r],
    /// n_in)`, as an adapter file would), switched on.
    pub(crate) fn for_test(rank: usize, scale: f32, modules: Vec<(String, Vec<f32>, Vec<f32>, usize)>) -> Self {
        let modules = modules
            .into_iter()
            .map(|(name, a, b, n_in)| {
                let r = a.len() / n_in;
                let n_out = b.len() / r;
                (name, Pair { a, b, r, n_in, n_out })
            })
            .collect();
        let mut ad = Self::empty(Path::new(""), "peft", rank, scale * rank as f32, scale);
        ad.modules = modules;
        ad.active.store(true, Ordering::Relaxed);
        ad
    }
}

/// Turns an adapter switch on for the guard's lifetime.
pub struct Active(Arc<AtomicBool>);

impl Active {
    pub fn new(switch: &Arc<AtomicBool>) -> Self {
        switch.store(true, Ordering::SeqCst);
        Self(switch.clone())
    }
}

impl Drop for Active {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

/// Row order of a Gated DeltaNet projection whose value heads a GGUF file stores tiled: entry `p`
/// is the Hugging Face row stored at row `p`. The first `head_rows` rows (query and key heads of
/// `in_proj_qkv`) keep their place; then `nv` value heads of `hd` rows follow, stored head `p`
/// being HF head `(p % nk) * (nv / nk) + p / nk`.
pub fn tiled_v_rows(head_rows: usize, nk: usize, nv: usize, hd: usize) -> Vec<usize> {
    let rep = nv / nk;
    let mut v: Vec<usize> = (0..head_rows).collect();
    for p in 0..nv {
        let hf = (p % nk) * rep + p / nk;
        v.extend((0..hd).map(|d| head_rows + hf * hd + d));
    }
    v
}

/// Row order of a Llama query / key projection in a GGUF file (llama.cpp's `LlamaModel.permute`):
/// entry `p` is the Hugging Face row stored at row `p`; stored row `2i + j` of a head of `hd` rows
/// is row `j * hd/2 + i` of that head.
pub fn rope_pair_rows(n_heads: usize, hd: usize) -> Vec<usize> {
    (0..n_heads * hd)
        .map(|p| {
            let (head, r) = (p / hd, p % hd);
            head * hd + (r % 2) * (hd / 2) + r / 2
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// llama.cpp's `permute` (`reshape(n_head, 2, hd/2).swapaxes(1, 2)`), and the equivalence the
    /// Llama GGUF path relies on: adjacent-pair RoPE on the stored rows is rotate-half RoPE on the
    /// Hugging Face rows, in the stored order.
    #[test]
    fn rope_pair_rows_match_llama_cpp_permute() {
        use crate::kernels::{apply_rope, apply_rope_interleaved};
        let (nh, hd) = (3, 8);
        let rows = rope_pair_rows(nh, hd);
        for h in 0..nh {
            for i in 0..hd / 2 {
                for j in 0..2 {
                    assert_eq!(rows[h * hd + i * 2 + j], h * hd + j * hd / 2 + i);
                }
            }
        }
        let x: Vec<f32> = (0..nh * hd).map(|i| (i as f32 * 0.37).sin()).collect();
        let cos: Vec<f32> = (0..hd / 2).map(|i| (1.0 + i as f32).cos()).collect();
        let sin: Vec<f32> = (0..hd / 2).map(|i| (1.0 + i as f32).sin()).collect();
        let mut hf = x.clone();
        let mut stored: Vec<f32> = rows.iter().map(|&r| x[r]).collect();
        for h in 0..nh {
            apply_rope(&mut hf[h * hd..(h + 1) * hd], &cos, &sin);
            apply_rope_interleaved(&mut stored[h * hd..(h + 1) * hd], &cos, &sin);
        }
        for (p, &r) in rows.iter().enumerate() {
            assert_eq!(stored[p], hf[r]);
        }
    }

    #[test]
    fn canonical_names() {
        assert_eq!(
            canonical("base_model.model.model.layers.3.mlp.down_proj").as_deref(),
            Some("layers.3.mlp.down_proj")
        );
        assert_eq!(
            canonical("base_model.model.model.language_model.layers.0.linear_attn.in_proj_qkv").as_deref(),
            Some("layers.0.linear_attn.in_proj_qkv")
        );
        assert_eq!(canonical("base_model.model.lm_head").as_deref(), Some("lm_head"));
        assert_eq!(
            canonical("base_model.model.mtp.layers.0.mlp.up_proj").as_deref(),
            Some("mtp.layers.0.mlp.up_proj")
        );
        assert_eq!(canonical("mtp.fc").as_deref(), Some("mtp.fc"));
        assert_eq!(canonical("base_model.model.model.visual.blocks.0.attn.qkv"), None);
    }

    #[test]
    fn delta_matches_reference() {
        let (r, n_in, n_out, t) = (3, 37, 1100, 21);
        let mut s = 1u32;
        let mut rnd = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        let a: Vec<f32> = (0..r * n_in).map(|_| rnd()).collect();
        let b: Vec<f32> = (0..n_out * r).map(|_| rnd()).collect();
        let x: Vec<f32> = (0..t * n_in).map(|_| rnd()).collect();
        let mut ad = LoraAdapter::for_test(r, 2.0, vec![("m".to_string(), a.clone(), b.clone(), n_in)]);
        let d = ad.take("m", n_out, n_in, None, None).unwrap().unwrap();
        assert!(ad.finish().is_ok());
        let mut y = vec![1f32; t * n_out];
        d.apply(&x, t, &mut y);
        for i in 0..t {
            for o in 0..n_out {
                let mut acc = 0f64;
                for k in 0..r {
                    let ax: f64 = (0..n_in).map(|c| a[k * n_in + c] as f64 * x[i * n_in + c] as f64).sum();
                    acc += b[o * r + k] as f64 * ax;
                }
                let want = 1.0 + 2.0 * acc;
                assert!((y[i * n_out + o] as f64 - want).abs() < 1e-4, "{i} {o}");
            }
        }
    }

    /// `cargo test --release --lib lora::tests::bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench() {
        for (r, n_in, n_out, t) in [
            (16, 5120, 17408, 1024),
            (16, 17408, 8, 1024),
            (16, 8, 17408, 1024),
            (16, 5120, 17408, 1),
        ] {
            let d = LoraDelta {
                a: vec![0.01; r * n_in],
                bt: vec![0.02; r * n_out],
                r,
                n_in,
                n_out,
                scale: 2.0,
                active: Arc::new(AtomicBool::new(true)),
                gain: unit_gain(),
                next: None,
            };
            let x = vec![0.5f32; t * n_in];
            let mut y = vec![0f32; t * n_out];
            d.apply(&x, t, &mut y);
            let t0 = std::time::Instant::now();
            for _ in 0..5 {
                d.apply(&x, t, &mut y);
            }
            let s = t0.elapsed().as_secs_f64() / 5.0;
            let flops = 2.0 * (r * (n_in + n_out) * t) as f64;
            eprintln!("{r} {n_in} {n_out} {t}: {:.3} ms, {:.0} GFLOP/s", s * 1e3, flops / s / 1e9);
        }
    }

    #[test]
    fn tiled_rows() {
        // 2 key heads, 4 value heads of 1 row: stored heads are HF heads 0, 2, 1, 3.
        assert_eq!(tiled_v_rows(2, 2, 4, 1), vec![0, 1, 2, 4, 3, 5]);
    }
}
