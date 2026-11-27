//! LTX-2.5's transformer on an accelerator's whole-pass session ([`crate::gpu::dit::LtxSession`]):
//! the layers as the device sees them, the per-forward modulation tables, and the CPU output
//! heads after it.
//!
//! ComfyUI's int8 ConvRot weights (int8 rows times a row scale) get a twin in the GEMMs' Q8
//! layout: the same int8 bytes (borrowed from the mapped file) with the row scale repeated per
//! block of 32; the session rotates their inputs (`dit_rot_half`). bf16 layers go as stored.
//! LoRA updates become side paths (`up (down x)`, float16, the rank padded to 64; `down` in the
//! rotated basis of a ConvRot layer). NVFP4 layers (ggml's blocks, repacked at load) go as they
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
    /// By the address of the layer's `DLin`: its LoRA updates as one side path (down, up).
    sides: HashMap<usize, (crate::model::Lin<'a>, crate::model::Lin<'a>, usize)>,
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

/// The LoRA updates of `d` (at their strengths now) as one side path, the rank padded to 64,
/// `down` rotated when the layer's input is (`rot`).
fn side_of(d: &DLin, rot: Option<usize>) -> Option<(crate::model::Lin<'static>, crate::model::Lin<'static>, usize)> {
    let delta = d.w.lora_delta()?;
    let (n_in, n_out) = (d.n_in, d.n_out);
    let links: Vec<_> = delta.links().filter(|l| l.strength() != 0.0).collect();
    let r: usize = links.iter().map(|l| l.factors().2).sum();
    if r == 0 {
        return None;
    }
    let rp = r.next_multiple_of(64);
    let mut down = vec![0f32; rp * n_in];
    let mut up = vec![0f32; n_out * rp];
    let mut k0 = 0;
    for l in links {
        let (a, bt, lr) = l.factors();
        let s = l.strength();
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
    Some((f16_lin(&down, rp, n_in), f16_lin(&up, n_out, rp), rp))
}

fn key(d: &DLin) -> usize {
    d as *const DLin as usize
}

/// The Q8 twin of an int8 (row-scaled) layer.
fn q8_twin<'a>(d: &DLin<'a>) -> Option<crate::model::Lin<'a>> {
    let crate::model::Weight::Gguf(m) = &d.w.w else { return None };
    if m.kind != crate::ggml::GgmlType::I8 || !d.n_in.is_multiple_of(crate::quant::QBLOCK) {
        return None;
    }
    let b = m.block_scales.as_ref()?;
    let bpr = d.n_in / crate::quant::QBLOCK;
    let mut sc = crate::quant::PageVec::<f32>::zeroed(d.n_out * bpr);
    for (r, c) in sc.chunks_mut(bpr).enumerate() {
        c.fill(if b.s.len() == 1 { b.s[0] } else { b.s[r] });
    }
    // SAFETY: the bytes are the mapped file's (or the layer's own buffer), which outlive the
    // transformer holding both the layer and its twin; i8 and u8 share a layout.
    let q: &'a [i8] = unsafe { std::slice::from_raw_parts(m.data.as_ptr() as *const i8, d.n_out * d.n_in) };
    Some(crate::model::Lin::q8(crate::quant::QMatrix {
        q: unsafe { crate::quant::QBuf::borrowed(q) },
        scales: sc.into(),
        n_out: d.n_out,
        n_in: d.n_in,
    }))
}

/// The FP8 twin of an FP8 layer with one scale for the whole matrix: the scale repeated over
/// the 128 x 128 blocks of the device's FP8 GEMMs.
fn fp8_twin<'a>(d: &DLin<'a>) -> Option<crate::model::Lin<'a>> {
    let crate::model::Weight::Gguf(m) = &d.w.w else { return None };
    let b = m.block_scales.as_ref()?;
    if m.kind != crate::ggml::GgmlType::F8E4M3 || b.e8m0.is_some() || b.s.len() != 1 || m.data.len() != d.n_out * d.n_in {
        return None;
    }
    const B: usize = 128;
    let mut sc = crate::quant::PageVec::<f32>::zeroed(d.n_out.div_ceil(B) * d.n_in.div_ceil(B));
    sc.fill(b.s[0]);
    // SAFETY: as in `q8_twin`.
    let q: &'a [u8] = unsafe { std::slice::from_raw_parts(m.data.as_ptr(), d.n_out * d.n_in) };
    let g = crate::ggml::GgufMatrix::fp8(unsafe { crate::quant::QBuf::borrowed(q) }, sc.into(), d.n_out, d.n_in, (B, B)).ok()?;
    Some(crate::model::Lin::gguf(g))
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
            }
        } else {
            if let Some(q) = q8_twin(d).or_else(|| fp8_twin(d)) {
                self.twin.insert(key(d), q);
            }
            let rot = d.w.fold.as_ref().map(|f| f.block);
            if let Some(s) = side_of(d, rot) {
                self.sides.insert(key(d), s);
            }
        }
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
        let side = match self.sides.get(&key(d)) {
            Some((dn, up, rp)) => {
                let (rp, n_in, n_out) = (*rp, d.n_in, d.n_out);
                let plain = |l: &'s crate::model::Lin<'a>, o: usize, i: usize| -> Option<Box<DevLin<'s>>> {
                    Some(Box::new(DevLin {
                        w: l.as_ref(),
                        dev: l.device_weight_unfolded(o, i)?,
                        n_out: o,
                        n_in: i,
                        bias: None,
                        side: None,
                        rot: None,
                    }))
                };
                Some((plain(dn, rp, n_in)?, plain(up, n_out, rp)?))
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
            rot,
        })
    }

    fn attn<'s>(&'s self, a: &'s super::nn::Attn<'a>) -> Option<LtxAttn<'s>> {
        let g = a.gate.as_ref()?;
        let (gw, gb) = self.gates.get(&key(g))?;
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
        c.blocks()
            .iter()
            .map(|(a, f)| {
                Some(crate::gpu::dit::LtxConnBlock {
                    attn: self.attn(a)?,
                    ff: self.ff(f)?,
                })
            })
            .collect()
    }

    /// The device's view of `dit`'s blocks.
    pub(crate) fn model<'s>(&'s self, dit: &'s Dit<'a>, half_attention: bool) -> Option<LtxModel<'s>> {
        let mut blocks = Vec::with_capacity(dit.n_blocks());
        for i in 0..dit.n_blocks() {
            let (v, a, a2v, v2a) = dit.block_parts(i);
            let stream = |s: (&'s super::nn::Attn<'a>, &'s super::nn::Attn<'a>, &'s super::nn::Ff<'a>)| -> Option<LtxStream<'s>> {
                Some(LtxStream {
                    attn1: self.attn(s.0)?,
                    attn2: self.attn(s.1)?,
                    ff: self.ff(s.2)?,
                })
            };
            blocks.push(LtxBlock {
                v: stream(v)?,
                a: stream(a)?,
                a2v: self.attn(a2v)?,
                v2a: self.attn(v2a)?,
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
