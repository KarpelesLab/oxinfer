//! LTX-2.5's transformer on an accelerator's whole-pass session ([`crate::gpu::dit::LtxSession`]):
//! the layers as the device sees them, the per-forward modulation tables, and the CPU output
//! heads after it.
//!
//! ComfyUI's int8 ConvRot weights (int8 rows times a row scale) get a twin in the GEMMs' Q8
//! layout: the same int8 bytes (borrowed from the mapped file) with the row scale repeated per
//! block of 32; the session rotates their inputs (`dit_rot_half`). bf16 layers go as stored.
//! LoRA updates become side paths (`up (down x)`, float16, the rank padded to 64; `down` in the
//! rotated basis of a ConvRot layer), made once with every update of the layer: each update's
//! run-time strength (a switched adapter's on / off and gain) scales its rank columns on the
//! device (`Lin::side_gains`), so one copy serves every strength. NVFP4 layers (ggml's blocks, repacked at load) go as they
//! are: the GEMMs dequantize their E2M1 codes, E4M3 scales and tensor-wide factor
//! (`mma_NVFP4`). FP8 E4M3 layers with one scale for the whole matrix (ComfyUI's "scaled" FP8)
//! get a twin with that scale repeated over the 128 x 128 blocks the FP8 GEMMs read (the same
//! bytes, borrowed). Anything else keeps the CPU path.

use super::dit::Dit;
use super::nn::Rope;
use crate::gpu::dit::{Lin as DevLin, LtxAttn, LtxBlock, LtxFf, LtxIn, LtxModel, LtxStream};
use crate::minimax_h3::dit::DLin;
use std::collections::HashMap;

/// Device twins of the int8 and scaled FP8 layers and float32 copies of the gates, made on the
/// first device pass.
#[derive(Default)]
pub(crate) struct Twins<'a> {
    /// By the address of the layer's `DLin`.
    twin: HashMap<usize, crate::model::Lin<'a>>,
    /// By the address of the gate's `DLin`: `[heads, q_dim]` weights and the bias.
    gates: HashMap<usize, (Vec<f32>, Vec<f32>)>,
    /// By the address of the gate's `DLin`: each LoRA update of a gate as a dense delta (its
    /// fixed scale in) and its run-time strength.
    gate_deltas: HashMap<usize, Vec<(Vec<f32>, crate::lora::Strength)>>,
    /// The gates' weights with their updates at each set of strengths they ran at (by the
    /// updates' factors, in the order of `gate_deltas`'s sorted keys), never removed.
    #[allow(clippy::type_complexity)]
    gate_views: std::sync::Mutex<Vec<(Vec<u32>, Box<HashMap<usize, Vec<f32>>>)>>,
    /// By the address of the layer's `DLin`: its LoRA updates as one side path (down, up).
    sides: HashMap<usize, Side<'a>>,
    /// Twins with unit scales, their layers' scales applied to the outputs instead (`[n_out]`
    /// in `out_scales`): the GEMMs then read the int8 / E4M3 values as they are, exact in half,
    /// rather than their products with the scales rounded to half.
    pub(crate) unit_scales: bool,
    out_scales: HashMap<usize, Vec<f32>>,
}

/// A layer's LoRA updates as one side path: `down` (each update's rank rows at its fixed scale)
/// and `up`, the padded rank, and where each update's rows start with its run-time strength.
struct Side<'a> {
    down: crate::model::Lin<'a>,
    up: crate::model::Lin<'a>,
    rank: usize,
    links: Vec<(usize, crate::lora::Strength)>,
}

/// An f32 matrix `[n_out, n_in]` as a float16 "GGUF" layer.
pub(crate) fn f16_lin(w: &[f32], n_out: usize, n_in: usize) -> crate::model::Lin<'static> {
    let mut v = crate::quant::PageVec::<u8>::zeroed(w.len() * 2);
    for (o, x) in v.as_chunks_mut::<2>().0.iter_mut().zip(w) {
        *o = half::f16::from_f32(*x).to_le_bytes();
    }
    crate::model::Lin::gguf(crate::ggml::GgufMatrix {
        kind: crate::ggml::GgmlType::F16,
        data: v.into(),
        n_out,
        n_in,
        block_scales: None,
        x4: false,
    })
}

/// The LoRA updates of `d` (all but retired adapters', whatever their strengths now) as one side
/// path, the rank padded to 64, `down` rotated when the layer's input is (`rot`); each update's
/// rows at its fixed scale, its run-time factor (on / off, gain) left to [`Twins::lin`].
fn side_of(d: &DLin, rot: Option<usize>) -> Option<Side<'static>> {
    let delta = d.w.lora_delta()?;
    let (n_in, n_out) = (d.n_in, d.n_out);
    let links: Vec<_> = delta.links().filter(|l| !l.retired() && l.scale() != 0.0).collect();
    let r: usize = links.iter().map(|l| l.factors().2).sum();
    if r == 0 {
        return None;
    }
    let rp = r.next_multiple_of(64);
    let mut down = vec![0f32; rp * n_in];
    let mut up = vec![0f32; n_out * rp];
    let mut k0 = 0;
    let mut starts = Vec::with_capacity(links.len());
    for l in links {
        let (a, bt, lr) = l.factors();
        let s = l.scale();
        starts.push((k0, l.strength_handle()));
        for k in 0..lr {
            let row = &mut down[(k0 + k) * n_in..(k0 + k + 1) * n_in];
            row.iter_mut().zip(&a[k * n_in..(k + 1) * n_in]).for_each(|(o, &v)| *o = v * s);
            for o in 0..n_out {
                up[o * rp + k0 + k] = bt[k * n_out + o];
            }
        }
        k0 += lr;
    }
    if let Some(g) = rot {
        // `W x = W_rot (H x)`: the update in the rotated basis is `(A H) (H x)`, H its own inverse.
        crate::kernels::for_chunks(&mut down, g * 16, true, |_, c| {
            c.chunks_exact_mut(g).for_each(crate::hadamard::fwht_regular)
        });
    }
    Some(Side {
        down: f16_lin(&down, rp, n_in),
        up: f16_lin(&up, n_out, rp),
        rank: rp,
        links: starts,
    })
}

fn key(d: &DLin) -> usize {
    d as *const DLin as usize
}

/// The Q8 twin of an int8 (row-scaled) layer (with `unit`, every scale 1 and the row scales
/// returned).
fn q8_twin<'a>(d: &DLin<'a>, unit: bool) -> Option<(crate::model::Lin<'a>, Option<Vec<f32>>)> {
    let crate::model::Weight::Gguf(m) = &d.w.w else { return None };
    if m.kind != crate::ggml::GgmlType::I8 || !d.n_in.is_multiple_of(crate::quant::QBLOCK) {
        return None;
    }
    let b = m.block_scales.as_ref()?;
    let bpr = d.n_in / crate::quant::QBLOCK;
    let row = |r: usize| if b.s.len() == 1 { b.s[0] } else { b.s[r] };
    let mut sc = crate::quant::PageVec::<f32>::zeroed(d.n_out * bpr);
    for (r, c) in sc.chunks_mut(bpr).enumerate() {
        c.fill(if unit { 1.0 } else { row(r) });
    }
    // SAFETY: the bytes are the mapped file's (or the layer's own buffer), which outlive the
    // transformer holding both the layer and its twin; i8 and u8 share a layout.
    let q: &'a [i8] = unsafe { std::slice::from_raw_parts(m.data.as_ptr() as *const i8, d.n_out * d.n_in) };
    let lin = crate::model::Lin::q8(crate::quant::QMatrix {
        q: unsafe { crate::quant::QBuf::borrowed(q) },
        scales: sc.into(),
        n_out: d.n_out,
        n_in: d.n_in,
    });
    Some((lin, unit.then(|| (0..d.n_out).map(row).collect())))
}

/// The FP8 twin of an FP8 layer with one scale for the whole matrix: the scale repeated over
/// the 128 x 128 blocks of the device's FP8 GEMMs (with `unit`, 1 there and the scale returned
/// per row).
fn fp8_twin<'a>(d: &DLin<'a>, unit: bool) -> Option<(crate::model::Lin<'a>, Option<Vec<f32>>)> {
    let crate::model::Weight::Gguf(m) = &d.w.w else { return None };
    let b = m.block_scales.as_ref()?;
    if m.kind != crate::ggml::GgmlType::F8E4M3 || b.e8m0.is_some() || b.s.len() != 1 || m.data.len() != d.n_out * d.n_in {
        return None;
    }
    const B: usize = 128;
    let mut sc = crate::quant::PageVec::<f32>::zeroed(d.n_out.div_ceil(B) * d.n_in.div_ceil(B));
    sc.fill(if unit { 1.0 } else { b.s[0] });
    // SAFETY: as in `q8_twin`.
    let q: &'a [u8] = unsafe { std::slice::from_raw_parts(m.data.as_ptr(), d.n_out * d.n_in) };
    let g = crate::ggml::GgufMatrix::fp8(unsafe { crate::quant::QBuf::borrowed(q) }, sc.into(), d.n_out, d.n_in, (B, B)).ok()?;
    Some((crate::model::Lin::gguf(g), unit.then(|| vec![b.s[0]; d.n_out])))
}

impl<'a> Twins<'a> {
    pub(crate) fn build(dit: &Dit<'a>) -> Self {
        let mut t = Self::default();
        dit.for_each_block_layer(&mut |d, gate| t.add(d, gate));
        t
    }

    /// Makes the device state of layer `d` (a gate: its float32 copy; otherwise an int8
    /// layer's Q8 twin or a scaled FP8 layer's twin, and its LoRA side path).
    pub(crate) fn add(&mut self, d: &DLin<'a>, gate: bool) {
        if gate {
            let mut w = vec![0f32; d.n_out * d.n_in];
            let rows: Vec<u32> = (0..d.n_out as u32).collect();
            if d.w.rows(&rows, d.n_in, &mut w) {
                self.gates.insert(key(d), (w, d.b.clone().unwrap_or_else(|| vec![0.0; d.n_out])));
                // The LoRA updates (`rows` reads the stored weights alone).
                let deltas: Vec<_> =
                    d.w.lora_delta()
                        .into_iter()
                        .flat_map(|l| l.links())
                        .filter(|l| !l.retired() && l.scale() != 0.0)
                        .map(|l| {
                            let (a, bt, r) = l.factors();
                            let mut dw = vec![0f32; d.n_out * d.n_in];
                            for k in 0..r {
                                let ak = &a[k * d.n_in..(k + 1) * d.n_in];
                                for o in 0..d.n_out {
                                    let b = bt[k * d.n_out + o] * l.scale();
                                    dw[o * d.n_in..(o + 1) * d.n_in].iter_mut().zip(ak).for_each(|(w, &x)| *w += b * x);
                                }
                            }
                            (dw, l.strength_handle())
                        })
                        .collect();
                if !deltas.is_empty() {
                    self.gate_deltas.insert(key(d), deltas);
                }
            }
        } else {
            if let Some((q, s)) = q8_twin(d, self.unit_scales).or_else(|| fp8_twin(d, self.unit_scales)) {
                self.twin.insert(key(d), q);
                if let Some(s) = s {
                    self.out_scales.insert(key(d), s);
                }
            }
            let rot = d.w.fold.as_ref().map(|f| f.block);
            if let Some(s) = side_of(d, rot) {
                self.sides.insert(key(d), s);
            }
        }
    }

    /// The scales of `d`'s outputs when its twin has unit scales.
    pub(crate) fn out_scale(&self, d: &DLin<'a>) -> Option<&[f32]> {
        self.out_scales.get(&key(d)).map(|v| &v[..])
    }

    /// `d` as the device sees it: None when the device has no state for it or its input
    /// transform is not ConvRot's.
    pub(crate) fn lin<'s>(&'s self, d: &'s DLin<'a>) -> Option<DevLin<'s>> {
        if d.w.active_lora().is_some() && !self.sides.contains_key(&key(d)) {
            return None;
        }
        let rot = match &d.w.fold {
            None => None,
            Some(f) if f.regular && f.signs.is_empty() && f.perm.is_none() => Some(f.block),
            Some(_) => return None,
        };
        let w = self.twin.get(&key(d)).unwrap_or(&d.w);
        if rot.is_some() && std::ptr::eq(w, &d.w) {
            // A rotated weight in a format without a twin (NVFP4 ConvRot is not one).
            return None;
        }
        // Each update's factor now (its adapter's on / off and gain); none on: no side path.
        let mut side_gains: Vec<(usize, f32)> = Vec::new();
        let side = match self.sides.get(&key(d)) {
            Some(sd) => {
                for (start, st) in &sd.links {
                    let g = st.factor();
                    if side_gains.last().is_none_or(|l| l.1 != g) {
                        side_gains.push((*start, g));
                    }
                }
                if side_gains.iter().all(|g| g.1 == 0.0) {
                    side_gains.clear();
                    None
                } else {
                    if side_gains.len() > 4 {
                        return None;
                    }
                    if side_gains.iter().all(|g| g.1 == 1.0) {
                        side_gains.clear();
                    }
                    let (rp, n_in, n_out) = (sd.rank, d.n_in, d.n_out);
                    let plain = |l: &'s crate::model::Lin<'a>, o: usize, i: usize| -> Option<Box<DevLin<'s>>> {
                        Some(Box::new(DevLin {
                            w: l.as_ref(),
                            dev: l.device_weight_unfolded(o, i)?,
                            n_out: o,
                            n_in: i,
                            bias: None,
                            side: None,
                            side_gains: Vec::new(),
                            rot: None,
                        }))
                    };
                    Some((plain(&sd.down, rp, n_in)?, plain(&sd.up, n_out, rp)?))
                }
            }
            None => None,
        };
        Some(DevLin {
            w: w.as_ref(),
            dev: w.device_weight_unfolded(d.n_out, d.n_in)?,
            n_out: d.n_out,
            n_in: d.n_in,
            bias: d.b.as_deref(),
            side,
            side_gains,
            rot,
        })
    }

    /// The gates' weights with their LoRA updates at the strengths now (those with updates).
    fn gate_view(&self) -> &HashMap<usize, Vec<f32>> {
        let mut keys: Vec<usize> = self.gate_deltas.keys().copied().collect();
        keys.sort_unstable();
        let factors: Vec<u32> = keys
            .iter()
            .flat_map(|k| self.gate_deltas[k].iter().map(|(_, s)| s.factor().to_bits()))
            .collect();
        let mut v = self.gate_views.lock().unwrap_or_else(|e| e.into_inner());
        let p: *const HashMap<usize, Vec<f32>> = match v.iter().find(|(f, _)| *f == factors) {
            Some((_, m)) => &**m,
            None => {
                let mut m = HashMap::new();
                for k in keys {
                    let mut w = self.gates[&k].0.clone();
                    for (dw, s) in &self.gate_deltas[&k] {
                        let f = s.factor();
                        if f != 0.0 {
                            w.iter_mut().zip(dw).for_each(|(w, &d)| *w += f * d);
                        }
                    }
                    m.insert(k, w);
                }
                v.push((factors, Box::new(m)));
                &*v.last().unwrap().1
            }
        };
        // SAFETY: the boxes are never removed or replaced while `self` lives.
        unsafe { &*p }
    }

    fn attn<'s>(&'s self, a: &'s super::nn::Attn<'a>, gv: &'s HashMap<usize, Vec<f32>>) -> Option<LtxAttn<'s>> {
        let g = a.gate.as_ref()?;
        let (gw, gb) = self.gates.get(&key(g))?;
        let gw = gv.get(&key(g)).unwrap_or(gw);
        Some(LtxAttn {
            q: self.lin(&a.q)?,
            k: self.lin(&a.k)?,
            v: self.lin(&a.v)?,
            o: self.lin(&a.o)?,
            gate_w: gw,
            gate_b: gb,
            qn: &a.qn,
            kn: &a.kn,
            heads: a.heads,
            hd: a.hd,
        })
    }

    fn ff<'s>(&'s self, f: &'s super::nn::Ff<'a>) -> Option<LtxFf<'s>> {
        Some(LtxFf {
            up: self.lin(&f.up)?,
            down: self.lin(&f.down)?,
        })
    }

    /// The device's view of a text connector's blocks.
    pub(crate) fn connector<'s>(&'s self, c: &'s super::dit::Connector<'a>) -> Option<Vec<crate::gpu::dit::LtxConnBlock<'s>>> {
        let gv = self.gate_view();
        c.blocks()
            .iter()
            .map(|(a, f)| {
                Some(crate::gpu::dit::LtxConnBlock {
                    attn: self.attn(a, gv)?,
                    ff: self.ff(f)?,
                })
            })
            .collect()
    }

    /// The device's view of `dit`'s blocks.
    pub(crate) fn model<'s>(&'s self, dit: &'s Dit<'a>, half_attention: bool) -> Option<LtxModel<'s>> {
        let mut blocks = Vec::with_capacity(dit.n_blocks());
        let gv = self.gate_view();
        for i in 0..dit.n_blocks() {
            let (v, a, a2v, v2a) = dit.block_parts(i);
            let stream = |s: (&'s super::nn::Attn<'a>, &'s super::nn::Attn<'a>, &'s super::nn::Ff<'a>)| -> Option<LtxStream<'s>> {
                Some(LtxStream {
                    attn1: self.attn(s.0, gv)?,
                    attn2: self.attn(s.1, gv)?,
                    ff: self.ff(s.2)?,
                })
            };
            blocks.push(LtxBlock {
                v: stream(v)?,
                a: stream(a)?,
                a2v: self.attn(a2v, gv)?,
                v2a: self.attn(v2a, gv)?,
            });
        }
        Some(LtxModel {
            dim: dit.cfg.dim(),
            audio_dim: dit.cfg.audio_dim(),
            eps: dit.cfg.eps,
            blocks,
            half_attention,
        })
    }
}

/// One stream's per-forward tables in the session's layout.
pub(crate) struct Tables {
    pub cls: Vec<u32>,
    pub classes: usize,
    pub mods: Vec<f32>,
    pub av: Vec<f32>,
    pub av_gate: Vec<f32>,
    pub prompt: Vec<f32>,
}

impl Tables {
    pub(crate) fn input<'s>(&'s self, x: &'s [f32], rows: usize, ctx: &'s [f32], n_ctx: usize, pe: &'s Rope, cpe: &'s Rope) -> LtxIn<'s> {
        LtxIn {
            x,
            rows,
            cls: &self.cls,
            classes: self.classes,
            mods: &self.mods,
            av: &self.av,
            av_gate: &self.av_gate,
            prompt: &self.prompt,
            ctx,
            n_ctx,
            cos: &pe.cos,
            sin: &pe.sin,
            ccos: &cpe.cos,
            csin: &cpe.sin,
        }
    }
}
