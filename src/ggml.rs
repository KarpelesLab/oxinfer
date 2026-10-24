//! llama.cpp / ggml weight formats, as stored in GGUF files: block layouts, exact dequantization
//! and the decode dot products, so GGUF checkpoints run without re-quantizing.
//!
//! A `[n_out, n_in]` matrix is `n_out` rows of `n_in / block_size` consecutive blocks (a GGUF
//! tensor with dims `[ne0, ne1, ..]` has `n_in = ne0` and `n_out` = the product of the rest).
//! Dequantization follows ggml's reference code (`ggml-quants.c`) operation by operation, so it
//! matches it (and gguf-py's numpy version) bit for bit. The decode kernels multiply the stored
//! integers with int8 activations (`QAct`, per-32 scales) and apply the block scales in f32.
//!
//! Every format is first expressed as int8 values with one f32 scale (and, for formats with a
//! min or offset, one f32 offset) per 16 elements (`decode_i8`); that representation serves the
//! portable dot product. The hot formats have NEON kernels in `kernels_ggml.rs` that unpack in
//! registers instead.
//!
//! The block layouts and the i-quant grid tables (`ggml_tables.rs`) come from ggml
//! (https://github.com/ggml-org/llama.cpp, MIT License, Copyright (c) 2023-2024 The ggml authors).

use crate::kernels::{QMat, bf16_to_f32};
use crate::quant::{PageVec, PairAct, QAct, QBLOCK, QBuf, f16_to_f32};
use anyhow::{Result, bail};

#[path = "ggml_tables.rs"]
pub(crate) mod tables;
use tables::*;

#[cfg(target_arch = "aarch64")]
#[path = "kernels_ggml.rs"]
mod neon;

/// Elements per K-quant / i-quant super-block.
pub const QK_K: usize = 256;

/// ggml tensor types (the numeric values are the GGUF type ids).
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    IQ1_M = 29,
    BF16 = 30,
}

impl GgmlType {
    pub const ALL: [GgmlType; 22] = [
        GgmlType::F32,
        GgmlType::F16,
        GgmlType::Q4_0,
        GgmlType::Q4_1,
        GgmlType::Q5_0,
        GgmlType::Q5_1,
        GgmlType::Q8_0,
        GgmlType::Q2_K,
        GgmlType::Q3_K,
        GgmlType::Q4_K,
        GgmlType::Q5_K,
        GgmlType::Q6_K,
        GgmlType::IQ2_XXS,
        GgmlType::IQ2_XS,
        GgmlType::IQ3_XXS,
        GgmlType::IQ1_S,
        GgmlType::IQ4_NL,
        GgmlType::IQ3_S,
        GgmlType::IQ2_S,
        GgmlType::IQ4_XS,
        GgmlType::IQ1_M,
        GgmlType::BF16,
    ];

    /// The type with GGUF id `id`, if supported.
    pub fn from_id(id: u32) -> Option<GgmlType> {
        Self::ALL.into_iter().find(|t| *t as u32 == id)
    }
    pub fn id(self) -> u32 {
        self as u32
    }
    pub fn name(self) -> &'static str {
        use GgmlType::*;
        match self {
            F32 => "F32",
            F16 => "F16",
            Q4_0 => "Q4_0",
            Q4_1 => "Q4_1",
            Q5_0 => "Q5_0",
            Q5_1 => "Q5_1",
            Q8_0 => "Q8_0",
            Q2_K => "Q2_K",
            Q3_K => "Q3_K",
            Q4_K => "Q4_K",
            Q5_K => "Q5_K",
            Q6_K => "Q6_K",
            IQ2_XXS => "IQ2_XXS",
            IQ2_XS => "IQ2_XS",
            IQ3_XXS => "IQ3_XXS",
            IQ1_S => "IQ1_S",
            IQ4_NL => "IQ4_NL",
            IQ3_S => "IQ3_S",
            IQ2_S => "IQ2_S",
            IQ4_XS => "IQ4_XS",
            IQ1_M => "IQ1_M",
            BF16 => "BF16",
        }
    }
    /// Elements per block.
    pub fn block_size(self) -> usize {
        use GgmlType::*;
        match self {
            F32 | F16 | BF16 => 1,
            Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | IQ4_NL => 32,
            _ => QK_K,
        }
    }
    /// Bytes per block.
    pub fn type_size(self) -> usize {
        use GgmlType::*;
        match self {
            F32 => 4,
            F16 | BF16 => 2,
            Q4_0 => 18,
            Q4_1 => 20,
            Q5_0 => 22,
            Q5_1 => 24,
            Q8_0 => 34,
            Q2_K => 84,
            Q3_K => 110,
            Q4_K => 144,
            Q5_K => 176,
            Q6_K => 210,
            IQ2_XXS => 66,
            IQ2_XS => 74,
            IQ3_XXS => 98,
            IQ1_S => 50,
            IQ4_NL => 18,
            IQ3_S => 110,
            IQ2_S => 82,
            IQ4_XS => 136,
            IQ1_M => 56,
        }
    }
    /// Plain float storage (no blocks): runs through the f32 path rather than int8 dot products.
    pub fn is_float(self) -> bool {
        matches!(self, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16)
    }
    /// Bytes of a row of `n` elements (`n` must be a multiple of the block size).
    pub fn row_bytes(self, n: usize) -> usize {
        n / self.block_size() * self.type_size()
    }
}

impl std::fmt::Display for GgmlType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// A `[n_out, n_in]` matrix in a ggml format: rows of `n_in` elements stored as consecutive
/// blocks, `row_bytes` apart. The bytes are owned or borrowed from a mapped GGUF file.
pub struct GgufMatrix {
    pub kind: GgmlType,
    pub data: QBuf<u8>,
    pub n_out: usize,
    pub n_in: usize,
}

impl GgufMatrix {
    /// Wraps `data` (`n_out` rows of `kind.row_bytes(n_in)` bytes). Rows must hold whole blocks,
    /// and quantized formats need `n_in` to be a multiple of 32 (the activation block).
    pub fn new(kind: GgmlType, data: QBuf<u8>, n_out: usize, n_in: usize) -> Result<Self> {
        if !n_in.is_multiple_of(kind.block_size()) {
            bail!("{kind} row length {n_in} is not a multiple of the block size {}", kind.block_size());
        }
        if !kind.is_float() && !n_in.is_multiple_of(QBLOCK) {
            bail!("{kind} row length {n_in} is not a multiple of {QBLOCK}");
        }
        let want = n_out * kind.row_bytes(n_in);
        if data.len() != want {
            bail!("{kind} [{n_out}, {n_in}] needs {want} bytes, got {}", data.len());
        }
        Ok(Self { kind, data, n_out, n_in })
    }

    /// Borrows the raw tensor bytes without copying (e.g. from a mapped GGUF file).
    ///
    /// # Safety
    /// `bytes` must stay valid and unmodified for as long as the matrix is used.
    pub unsafe fn borrowed(kind: GgmlType, bytes: &[u8], n_out: usize, n_in: usize) -> Result<Self> {
        Self::new(kind, unsafe { QBuf::borrowed(bytes) }, n_out, n_in)
    }

    /// Copies the raw tensor bytes into an owned, page-aligned buffer.
    pub fn from_bytes(kind: GgmlType, bytes: &[u8], n_out: usize, n_in: usize) -> Result<Self> {
        let mut v = PageVec::<u8>::zeroed(bytes.len());
        v.copy_from_slice(bytes);
        Self::new(kind, v.into(), n_out, n_in)
    }

    pub fn row_bytes(&self) -> usize {
        self.kind.row_bytes(self.n_in)
    }
    pub fn row(&self, r: usize) -> &[u8] {
        let rb = self.row_bytes();
        &self.data[r * rb..(r + 1) * rb]
    }
    pub fn bytes(&self) -> usize {
        self.data.len()
    }
    /// Dequantizes rows `r0..r1` into `out` (f32 row-major, `(r1 - r0) * n_in`); the same values
    /// as `dequantize_row`, through NEON where available.
    pub fn dequantize_rows(&self, r0: usize, r1: usize, out: &mut [f32]) {
        let n = self.n_in;
        for (i, r) in (r0..r1).enumerate() {
            let o = &mut out[i * n..(i + 1) * n];
            #[cfg(target_arch = "aarch64")]
            // SAFETY: the row holds `n` elements of `kind` and `o` has room for them.
            if unsafe { neon::dequant_row(self.kind, self.row(r).as_ptr(), n, o.as_mut_ptr()) } {
                continue;
            }
            dequantize_row(self.kind, self.row(r), o);
        }
    }
}

#[inline(always)]
fn f16at(b: &[u8], i: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[i], b[i + 1]]))
}
#[inline(always)]
fn u16at(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([b[i], b[i + 1]])
}
#[inline(always)]
fn u32at(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])
}

/// Scale and min `j` (0..8) of a Q4_K / Q5_K block (`get_scale_min_k4`).
#[inline(always)]
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// The 16 signed 6-bit scales of a Q3_K block, unpacked from its 12 bytes (still offset by 32).
#[inline(always)]
fn q3k_scales(s: &[u8]) -> [i8; 16] {
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;
    let a = [u32at(s, 0), u32at(s, 4), u32at(s, 8)];
    let tmp = a[2];
    let aux = [
        (a[0] & KMASK2) | ((tmp & KMASK1) << 4),
        (a[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4),
        ((a[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4),
        ((a[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4),
    ];
    let mut out = [0i8; 16];
    for (i, w) in aux.iter().enumerate() {
        for (j, b) in w.to_le_bytes().into_iter().enumerate() {
            out[i * 4 + j] = b as i8;
        }
    }
    out
}

#[inline(always)]
fn grid8(g: u64, j: usize) -> u8 {
    (g >> (8 * j)) as u8
}
#[inline(always)]
fn grid4(g: u32, j: usize) -> u8 {
    (g >> (8 * j)) as u8
}

/// The 6-bit scale of IQ4_XS sub-block `ib`, offset by 32.
#[inline(always)]
fn iq4xs_scale(b: &[u8], ib: usize) -> i32 {
    let sh = u16at(b, 2);
    (((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) as i32 | ((((sh >> (2 * ib)) & 3) as i32) << 4)) - 32
}

/// Dequantizes one row (`out.len()` elements) of `kind` from its raw bytes.
pub fn dequantize_row(kind: GgmlType, src: &[u8], out: &mut [f32]) {
    use GgmlType::*;
    let n = out.len();
    assert_eq!(src.len(), kind.row_bytes(n), "{kind}: {} bytes for {n} elements", src.len());
    match kind {
        F32 => {
            for (o, c) in out.iter_mut().zip(src.as_chunks::<4>().0) {
                *o = f32::from_le_bytes(*c);
            }
            return;
        }
        F16 => {
            for (o, c) in out.iter_mut().zip(src.as_chunks::<2>().0) {
                *o = f16_to_f32(u16::from_le_bytes(*c));
            }
            return;
        }
        BF16 => {
            for (o, c) in out.iter_mut().zip(src.as_chunks::<2>().0) {
                *o = bf16_to_f32(u16::from_le_bytes(*c));
            }
            return;
        }
        _ => {}
    }
    let (bs, ts) = (kind.block_size(), kind.type_size());
    for (b, y) in src.chunks_exact(ts).zip(out.chunks_exact_mut(bs)) {
        match kind {
            Q4_0 => {
                let d = f16at(b, 0);
                for j in 0..16 {
                    y[j] = ((b[2 + j] & 0xF) as i32 - 8) as f32 * d;
                    y[j + 16] = ((b[2 + j] >> 4) as i32 - 8) as f32 * d;
                }
            }
            Q4_1 => {
                let (d, m) = (f16at(b, 0), f16at(b, 2));
                for j in 0..16 {
                    y[j] = (b[4 + j] & 0xF) as f32 * d + m;
                    y[j + 16] = (b[4 + j] >> 4) as f32 * d + m;
                }
            }
            Q5_0 => {
                let d = f16at(b, 0);
                let qh = u32at(b, 2);
                for j in 0..16 {
                    let h0 = (((qh >> j) << 4) & 0x10) as u8;
                    let h1 = ((qh >> (j + 12)) & 0x10) as u8;
                    y[j] = (((b[6 + j] & 0xF) | h0) as i32 - 16) as f32 * d;
                    y[j + 16] = (((b[6 + j] >> 4) | h1) as i32 - 16) as f32 * d;
                }
            }
            Q5_1 => {
                let (d, m) = (f16at(b, 0), f16at(b, 2));
                let qh = u32at(b, 4);
                for j in 0..16 {
                    let h0 = (((qh >> j) << 4) & 0x10) as u8;
                    let h1 = ((qh >> (j + 12)) & 0x10) as u8;
                    y[j] = ((b[8 + j] & 0xF) | h0) as f32 * d + m;
                    y[j + 16] = ((b[8 + j] >> 4) | h1) as f32 * d + m;
                }
            }
            Q8_0 => {
                let d = f16at(b, 0);
                for j in 0..32 {
                    y[j] = (b[2 + j] as i8) as f32 * d;
                }
            }
            Q2_K => {
                let (d, min) = (f16at(b, 80), f16at(b, 82));
                let (sc, qs) = (&b[0..16], &b[16..80]);
                for n in 0..2 {
                    let q = &qs[32 * n..32 * n + 32];
                    for j in 0..4 {
                        for h in 0..2 {
                            let s = sc[8 * n + 2 * j + h];
                            let (dl, ml) = (d * (s & 0xF) as f32, min * (s >> 4) as f32);
                            let o = &mut y[128 * n + 32 * j + 16 * h..][..16];
                            for l in 0..16 {
                                o[l] = dl * ((q[16 * h + l] >> (2 * j)) & 3) as f32 - ml;
                            }
                        }
                    }
                }
            }
            Q3_K => {
                let d = f16at(b, 108);
                let (hm, qs) = (&b[0..32], &b[32..96]);
                let sc = q3k_scales(&b[96..108]);
                for n in 0..2 {
                    let q = &qs[32 * n..32 * n + 32];
                    for j in 0..4 {
                        let m = 1u8 << (4 * n + j);
                        for h in 0..2 {
                            let dl = d * (sc[8 * n + 2 * j + h] as i32 - 32) as f32;
                            let o = &mut y[128 * n + 32 * j + 16 * h..][..16];
                            for l in 0..16 {
                                let i = 16 * h + l;
                                let v = ((q[i] >> (2 * j)) & 3) as i32 - if hm[i] & m != 0 { 0 } else { 4 };
                                o[l] = dl * v as f32;
                            }
                        }
                    }
                }
            }
            Q4_K | Q5_K => {
                let (d, min) = (f16at(b, 0), f16at(b, 2));
                let sc = &b[4..16];
                let (qh, qs) = if kind == Q5_K {
                    (&b[16..48], &b[48..176])
                } else {
                    (&b[0..0], &b[16..144])
                };
                for j in 0..4 {
                    let q = &qs[32 * j..32 * j + 32];
                    for h in 0..2 {
                        let (s, m) = scale_min_k4(2 * j + h, sc);
                        let (d1, m1) = (d * s as f32, min * m as f32);
                        let o = &mut y[64 * j + 32 * h..][..32];
                        for l in 0..32 {
                            let mut v = (q[l] >> (4 * h)) & 0xF;
                            if kind == Q5_K && qh[l] & (1 << (2 * j + h)) != 0 {
                                v += 16;
                            }
                            o[l] = d1 * v as f32 - m1;
                        }
                    }
                }
            }
            Q6_K => {
                let d = f16at(b, 208);
                for n in 0..2 {
                    let ql = &b[64 * n..64 * n + 64];
                    let qh = &b[128 + 32 * n..128 + 32 * n + 32];
                    let sc = &b[192 + 8 * n..192 + 8 * n + 8];
                    let o = &mut y[128 * n..128 * n + 128];
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
                        let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                        let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                        let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                        o[l] = d * (sc[is] as i8) as f32 * q1 as f32;
                        o[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                        o[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                        o[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
                    }
                }
            }
            IQ2_XXS => {
                let d = f16at(b, 0);
                for ib in 0..8 {
                    let (a0, a1) = (u32at(b, 2 + 8 * ib), u32at(b, 6 + 8 * ib));
                    let db = d * (0.5 + (a1 >> 28) as f32) * 0.25;
                    for l in 0..4 {
                        let g = IQ2XXS_GRID[((a0 >> (8 * l)) & 255) as usize];
                        let signs = KSIGNS_IQ2XS[((a1 >> (7 * l)) & 127) as usize];
                        for j in 0..8 {
                            y[32 * ib + 8 * l + j] = db * grid8(g, j) as f32 * if signs & (1 << j) != 0 { -1.0 } else { 1.0 };
                        }
                    }
                }
            }
            IQ2_XS => {
                let d = f16at(b, 0);
                for ib in 0..8 {
                    let s = b[66 + ib];
                    let db = [d * (0.5 + (s & 0xF) as f32) * 0.25, d * (0.5 + (s >> 4) as f32) * 0.25];
                    for l in 0..4 {
                        let q = u16at(b, 2 + 2 * (4 * ib + l));
                        let g = IQ2XS_GRID[(q & 511) as usize];
                        let signs = KSIGNS_IQ2XS[(q >> 9) as usize];
                        for j in 0..8 {
                            y[32 * ib + 8 * l + j] = db[l / 2] * grid8(g, j) as f32 * if signs & (1 << j) != 0 { -1.0 } else { 1.0 };
                        }
                    }
                }
            }
            IQ2_S => {
                let d = f16at(b, 0);
                let (qs, signs, qh, scales) = (&b[2..34], &b[34..66], &b[66..74], &b[74..82]);
                for ib in 0..8 {
                    let s = scales[ib];
                    let db = [d * (0.5 + (s & 0xF) as f32) * 0.25, d * (0.5 + (s >> 4) as f32) * 0.25];
                    for l in 0..4 {
                        let idx = qs[4 * ib + l] as usize | (((qh[ib] as usize) << (8 - 2 * l)) & 0x300);
                        let g = IQ2S_GRID[idx];
                        let sg = signs[4 * ib + l];
                        for j in 0..8 {
                            y[32 * ib + 8 * l + j] = db[l / 2] * grid8(g, j) as f32 * if sg & (1 << j) != 0 { -1.0 } else { 1.0 };
                        }
                    }
                }
            }
            IQ3_XXS => {
                let d = f16at(b, 0);
                let (qs, ss) = (&b[2..66], &b[66..98]);
                for ib in 0..8 {
                    let aux = u32at(ss, 4 * ib);
                    let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
                    for l in 0..4 {
                        let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
                        let g1 = IQ3XXS_GRID[qs[8 * ib + 2 * l] as usize];
                        let g2 = IQ3XXS_GRID[qs[8 * ib + 2 * l + 1] as usize];
                        let o = &mut y[32 * ib + 8 * l..][..8];
                        for j in 0..4 {
                            o[j] = db * grid4(g1, j) as f32 * if signs & (1 << j) != 0 { -1.0 } else { 1.0 };
                            o[j + 4] = db * grid4(g2, j) as f32 * if signs & (1 << (j + 4)) != 0 { -1.0 } else { 1.0 };
                        }
                    }
                }
            }
            IQ3_S => {
                let d = f16at(b, 0);
                let (qs, qh, signs, scales) = (&b[2..66], &b[66..74], &b[74..106], &b[106..110]);
                for ib in 0..8 {
                    let s = scales[ib / 2] >> (4 * (ib % 2));
                    let db = d * (1 + 2 * (s & 0xF) as i32) as f32;
                    let h = qh[ib] as usize;
                    for l in 0..4 {
                        let g1 = IQ3S_GRID[qs[8 * ib + 2 * l] as usize | ((h << (8 - 2 * l)) & 256)];
                        let g2 = IQ3S_GRID[qs[8 * ib + 2 * l + 1] as usize | ((h << (7 - 2 * l)) & 256)];
                        let sg = signs[4 * ib + l];
                        let o = &mut y[32 * ib + 8 * l..][..8];
                        for j in 0..4 {
                            o[j] = db * grid4(g1, j) as f32 * if sg & (1 << j) != 0 { -1.0 } else { 1.0 };
                            o[j + 4] = db * grid4(g2, j) as f32 * if sg & (1 << (j + 4)) != 0 { -1.0 } else { 1.0 };
                        }
                    }
                }
            }
            IQ1_S => {
                let d = f16at(b, 0);
                for ib in 0..8 {
                    let qh = u16at(b, 34 + 2 * ib);
                    let dl = d * (2 * ((qh >> 12) & 7) + 1) as f32;
                    let delta = if qh & 0x8000 != 0 { -0.125 } else { 0.125 };
                    for l in 0..4 {
                        let g = IQ1S_GRID[b[2 + 4 * ib + l] as usize | ((((qh >> (3 * l)) & 7) as usize) << 8)];
                        for j in 0..8 {
                            y[32 * ib + 8 * l + j] = dl * (grid8(g, j) as i8 as f32 + delta);
                        }
                    }
                }
            }
            IQ1_M => {
                let (qs, qh) = (&b[0..32], &b[32..48]);
                let sc = [u16at(b, 48), u16at(b, 50), u16at(b, 52), u16at(b, 54)];
                let d = f16_to_f32((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
                for ib in 0..8 {
                    let s = sc[ib / 2] >> (6 * (ib % 2));
                    let dl = [d * (2 * (s & 7) + 1) as f32, d * (2 * ((s >> 3) & 7) + 1) as f32];
                    let (h0, h1) = (qh[2 * ib] as usize, qh[2 * ib + 1] as usize);
                    let q = &qs[4 * ib..4 * ib + 4];
                    let idx = [
                        q[0] as usize | ((h0 << 8) & 0x700),
                        q[1] as usize | ((h0 << 4) & 0x700),
                        q[2] as usize | ((h1 << 8) & 0x700),
                        q[3] as usize | ((h1 << 4) & 0x700),
                    ];
                    let dm = |h: usize, bit: usize| if h & bit != 0 { -0.125 } else { 0.125 };
                    let delta = [dm(h0, 0x08), dm(h0, 0x80), dm(h1, 0x08), dm(h1, 0x80)];
                    for l in 0..4 {
                        let g = IQ1S_GRID[idx[l]];
                        for j in 0..8 {
                            y[32 * ib + 8 * l + j] = dl[l / 2] * (grid8(g, j) as i8 as f32 + delta[l]);
                        }
                    }
                }
            }
            IQ4_NL => {
                let d = f16at(b, 0);
                for j in 0..16 {
                    y[j] = d * KVALUES_IQ4NL[(b[2 + j] & 0xF) as usize] as f32;
                    y[j + 16] = d * KVALUES_IQ4NL[(b[2 + j] >> 4) as usize] as f32;
                }
            }
            IQ4_XS => {
                let d = f16at(b, 0);
                for ib in 0..8 {
                    let dl = d * iq4xs_scale(b, ib) as f32;
                    let q = &b[8 + 16 * ib..8 + 16 * ib + 16];
                    for j in 0..16 {
                        y[32 * ib + j] = dl * KVALUES_IQ4NL[(q[j] & 0xF) as usize] as f32;
                        y[32 * ib + j + 16] = dl * KVALUES_IQ4NL[(q[j] >> 4) as usize] as f32;
                    }
                }
            }
            F32 | F16 | BF16 => unreachable!(),
        }
    }
}

/// Decodes one block of a quantized `kind` into int8 values `q` (`block_size` of them) with one
/// scale `s` and one offset `m` per 16 elements: element `i` is `s[i/16] * q[i] + m[i/16]` (up to
/// f32 rounding). Returns false when the format has no offsets (`m` is left untouched).
pub fn decode_i8(kind: GgmlType, b: &[u8], q: &mut [i8], s: &mut [f32], m: &mut [f32]) -> bool {
    use GgmlType::*;
    let sign = |signs: u8, j: usize, v: u8| if signs & (1 << j) != 0 { -(v as i8) } else { v as i8 };
    match kind {
        Q4_0 | Q4_1 | Q5_0 | Q5_1 | IQ4_NL => {
            let d = f16at(b, 0);
            let (qs, qh, off) = match kind {
                Q4_0 | IQ4_NL => (&b[2..18], 0, 8),
                Q4_1 => (&b[4..20], 0, 0),
                Q5_0 => (&b[6..22], u32at(b, 2), 16),
                _ => (&b[8..24], u32at(b, 4), 0),
            };
            for j in 0..16 {
                let h0 = (((qh >> j) << 4) & 0x10) as u8;
                let h1 = ((qh >> (j + 12)) & 0x10) as u8;
                let (lo, hi) = ((qs[j] & 0xF) | h0, (qs[j] >> 4) | h1);
                if kind == IQ4_NL {
                    q[j] = KVALUES_IQ4NL[lo as usize];
                    q[j + 16] = KVALUES_IQ4NL[hi as usize];
                } else {
                    q[j] = (lo as i32 - off) as i8;
                    q[j + 16] = (hi as i32 - off) as i8;
                }
            }
            s[..2].fill(d);
            if matches!(kind, Q4_1 | Q5_1) {
                // Centered like Q4_0 / Q5_0 (smaller products with the rounded activations).
                let c = if kind == Q4_1 { 8 } else { 16 };
                for v in q[..32].iter_mut() {
                    *v -= c as i8;
                }
                m[..2].fill(f16at(b, 2) + d * c as f32);
                return true;
            }
            false
        }
        Q8_0 => {
            for j in 0..32 {
                q[j] = b[2 + j] as i8;
            }
            s[..2].fill(f16at(b, 0));
            false
        }
        Q2_K => {
            let (d, min) = (f16at(b, 80), f16at(b, 82));
            for n in 0..2 {
                for j in 0..4 {
                    for h in 0..2 {
                        let g = 8 * n + 2 * j + h;
                        // Centered: q - 2, the offset 2 * s moves into m.
                        s[g] = d * (b[g] & 0xF) as f32;
                        m[g] = 2.0 * s[g] - min * (b[g] >> 4) as f32;
                        for l in 0..16 {
                            q[16 * g + l] = ((b[16 + 32 * n + 16 * h + l] >> (2 * j)) & 3) as i8 - 2;
                        }
                    }
                }
            }
            true
        }
        Q3_K => {
            let d = f16at(b, 108);
            let sc = q3k_scales(&b[96..108]);
            for n in 0..2 {
                for j in 0..4 {
                    for h in 0..2 {
                        let g = 8 * n + 2 * j + h;
                        s[g] = d * (sc[g] as i32 - 32) as f32;
                        for l in 0..16 {
                            let i = 16 * h + l;
                            let v = (b[32 + 32 * n + i] >> (2 * j)) & 3;
                            let hb = b[i] & (1 << (4 * n + j)) != 0;
                            q[16 * g + l] = v as i8 - if hb { 0 } else { 4 };
                        }
                    }
                }
            }
            false
        }
        Q4_K | Q5_K => {
            let (d, min) = (f16at(b, 0), f16at(b, 2));
            let (qh, qs) = if kind == Q5_K {
                (&b[16..48], &b[48..176])
            } else {
                (&b[0..0], &b[16..144])
            };
            for j in 0..4 {
                for h in 0..2 {
                    let (sc, mn) = scale_min_k4(2 * j + h, &b[4..16]);
                    let g = 2 * (2 * j + h);
                    // Centered: q - 8 (q - 16 for Q5_K), the offset moves into m.
                    let off = if kind == Q5_K { 16 } else { 8 };
                    s[g] = d * sc as f32;
                    s[g + 1] = s[g];
                    m[g] = off as f32 * s[g] - min * mn as f32;
                    m[g + 1] = m[g];
                    for l in 0..32 {
                        let mut v = (qs[32 * j + l] >> (4 * h)) & 0xF;
                        if kind == Q5_K && qh[l] & (1 << (2 * j + h)) != 0 {
                            v += 16;
                        }
                        q[64 * j + 32 * h + l] = v as i8 - off;
                    }
                }
            }
            true
        }
        Q6_K => {
            let d = f16at(b, 208);
            for g in 0..16 {
                s[g] = d * (b[192 + g] as i8) as f32;
            }
            for n in 0..2 {
                let ql = &b[64 * n..64 * n + 64];
                let qh = &b[128 + 32 * n..128 + 32 * n + 32];
                let o = &mut q[128 * n..128 * n + 128];
                for l in 0..32 {
                    o[l] = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i8 - 32;
                    o[l + 32] = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
                    o[l + 64] = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
                    o[l + 96] = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;
                }
            }
            false
        }
        IQ2_XXS => {
            let d = f16at(b, 0);
            for ib in 0..8 {
                let (a0, a1) = (u32at(b, 2 + 8 * ib), u32at(b, 6 + 8 * ib));
                s[2 * ib] = d * (0.5 + (a1 >> 28) as f32) * 0.25;
                s[2 * ib + 1] = s[2 * ib];
                for l in 0..4 {
                    let g = IQ2XXS_GRID[((a0 >> (8 * l)) & 255) as usize];
                    let signs = KSIGNS_IQ2XS[((a1 >> (7 * l)) & 127) as usize];
                    for j in 0..8 {
                        q[32 * ib + 8 * l + j] = sign(signs, j, grid8(g, j));
                    }
                }
            }
            false
        }
        IQ2_XS | IQ2_S => {
            let d = f16at(b, 0);
            for ib in 0..8 {
                let sc = b[if kind == IQ2_XS { 66 } else { 74 } + ib];
                s[2 * ib] = d * (0.5 + (sc & 0xF) as f32) * 0.25;
                s[2 * ib + 1] = d * (0.5 + (sc >> 4) as f32) * 0.25;
                for l in 0..4 {
                    let (g, signs) = if kind == IQ2_XS {
                        let v = u16at(b, 2 + 2 * (4 * ib + l));
                        (IQ2XS_GRID[(v & 511) as usize], KSIGNS_IQ2XS[(v >> 9) as usize])
                    } else {
                        let idx = b[2 + 4 * ib + l] as usize | (((b[66 + ib] as usize) << (8 - 2 * l)) & 0x300);
                        (IQ2S_GRID[idx], b[34 + 4 * ib + l])
                    };
                    for j in 0..8 {
                        q[32 * ib + 8 * l + j] = sign(signs, j, grid8(g, j));
                    }
                }
            }
            false
        }
        IQ3_XXS => {
            let d = f16at(b, 0);
            for ib in 0..8 {
                let aux = u32at(b, 66 + 4 * ib);
                s[2 * ib] = d * (0.5 + (aux >> 28) as f32) * 0.5;
                s[2 * ib + 1] = s[2 * ib];
                for l in 0..4 {
                    let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
                    let g1 = IQ3XXS_GRID[b[2 + 8 * ib + 2 * l] as usize];
                    let g2 = IQ3XXS_GRID[b[2 + 8 * ib + 2 * l + 1] as usize];
                    for j in 0..4 {
                        q[32 * ib + 8 * l + j] = sign(signs, j, grid4(g1, j));
                        q[32 * ib + 8 * l + j + 4] = sign(signs, j + 4, grid4(g2, j));
                    }
                }
            }
            false
        }
        IQ3_S => {
            let d = f16at(b, 0);
            for ib in 0..8 {
                let sc = b[106 + ib / 2] >> (4 * (ib % 2));
                s[2 * ib] = d * (1 + 2 * (sc & 0xF) as i32) as f32;
                s[2 * ib + 1] = s[2 * ib];
                let h = b[66 + ib] as usize;
                for l in 0..4 {
                    let g1 = IQ3S_GRID[b[2 + 8 * ib + 2 * l] as usize | ((h << (8 - 2 * l)) & 256)];
                    let g2 = IQ3S_GRID[b[2 + 8 * ib + 2 * l + 1] as usize | ((h << (7 - 2 * l)) & 256)];
                    let signs = b[74 + 4 * ib + l];
                    for j in 0..4 {
                        q[32 * ib + 8 * l + j] = sign(signs, j, grid4(g1, j));
                        q[32 * ib + 8 * l + j + 4] = sign(signs, j + 4, grid4(g2, j));
                    }
                }
            }
            false
        }
        // dl * (grid + delta) with delta = +-1/8: q = 8 * grid +- 1 and scale dl / 8.
        IQ1_S => {
            let d = f16at(b, 0);
            for ib in 0..8 {
                let qh = u16at(b, 34 + 2 * ib);
                s[2 * ib] = d * (2 * ((qh >> 12) & 7) + 1) as f32 * 0.125;
                s[2 * ib + 1] = s[2 * ib];
                let delta: i8 = if qh & 0x8000 != 0 { -1 } else { 1 };
                for l in 0..4 {
                    let g = IQ1S_GRID[b[2 + 4 * ib + l] as usize | ((((qh >> (3 * l)) & 7) as usize) << 8)];
                    for j in 0..8 {
                        q[32 * ib + 8 * l + j] = 8 * grid8(g, j) as i8 + delta;
                    }
                }
            }
            false
        }
        IQ1_M => {
            let sc = [u16at(b, 48), u16at(b, 50), u16at(b, 52), u16at(b, 54)];
            let d = f16_to_f32((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
            for ib in 0..8 {
                let sv = sc[ib / 2] >> (6 * (ib % 2));
                s[2 * ib] = d * (2 * (sv & 7) + 1) as f32 * 0.125;
                s[2 * ib + 1] = d * (2 * ((sv >> 3) & 7) + 1) as f32 * 0.125;
                let (h0, h1) = (b[32 + 2 * ib] as usize, b[32 + 2 * ib + 1] as usize);
                let qs = &b[4 * ib..4 * ib + 4];
                let idx = [
                    qs[0] as usize | ((h0 << 8) & 0x700),
                    qs[1] as usize | ((h0 << 4) & 0x700),
                    qs[2] as usize | ((h1 << 8) & 0x700),
                    qs[3] as usize | ((h1 << 4) & 0x700),
                ];
                let dm = |h: usize, bit: usize| if h & bit != 0 { -1i8 } else { 1 };
                let delta = [dm(h0, 0x08), dm(h0, 0x80), dm(h1, 0x08), dm(h1, 0x80)];
                for l in 0..4 {
                    let g = IQ1S_GRID[idx[l]];
                    for j in 0..8 {
                        q[32 * ib + 8 * l + j] = 8 * grid8(g, j) as i8 + delta[l];
                    }
                }
            }
            false
        }
        IQ4_XS => {
            let d = f16at(b, 0);
            for ib in 0..8 {
                s[2 * ib] = d * iq4xs_scale(b, ib) as f32;
                s[2 * ib + 1] = s[2 * ib];
                for j in 0..16 {
                    let v = b[8 + 16 * ib + j];
                    q[32 * ib + j] = KVALUES_IQ4NL[(v & 0xF) as usize];
                    q[32 * ib + j + 16] = KVALUES_IQ4NL[(v >> 4) as usize];
                }
            }
            false
        }
        F32 | F16 | BF16 => unreachable!("decode_i8 on float type {kind}"),
    }
}

/// Activation rows for the dot kernels: token `c` of the quantized input starts at `c * n_in`
/// values / `c * n_in / 32` scales and block sums / `c * n_in / 16` half-block sums.
#[derive(Clone, Copy)]
pub(crate) struct Act {
    pub q: *const i8,
    pub s: *const f32,
    /// Block sums (the NEON kernels' per-32 mins; the portable path uses the half-block sums).
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    pub f: *const f32,
    pub h: *const f32,
    /// Integer block sums of the int8 values (folds the offset of centered nibble formats).
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    pub i: *const i32,
    pub n_in: usize,
}

impl Act {
    fn new(xa: &QAct, c0: usize) -> Self {
        let n = xa.n_in;
        Self {
            q: unsafe { xa.q.as_ptr().add(c0 * n) },
            s: unsafe { xa.scales.as_ptr().add(c0 * n / QBLOCK) },
            f: unsafe { xa.fsums.as_ptr().add(c0 * n / QBLOCK) },
            h: unsafe { xa.hsums.as_ptr().add(c0 * n / 16) },
            i: unsafe { xa.sums.as_ptr().add(c0 * n / QBLOCK) },
            n_in: n,
        }
    }
}

/// Integer dot product of 16 int8 pairs.
#[inline(always)]
unsafe fn dot16(a: *const i8, b: *const i8) -> i32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        vaddvq_s32(crate::kernels::simd::sdot(vdupq_n_s32(0), vld1q_s8(a), vld1q_s8(b)))
    }
    #[cfg(not(target_arch = "aarch64"))]
    unsafe {
        let mut s = 0i32;
        for i in 0..16 {
            s += *a.add(i) as i32 * *b.add(i) as i32;
        }
        s
    }
}

/// Portable dot products of one row against TC activation rows through `decode_i8`.
unsafe fn dot_row_generic<const TC: usize>(kind: GgmlType, row: &[u8], x: &Act) -> [f32; TC] {
    let (bs, ts) = (kind.block_size(), kind.type_size());
    let mut q = [0i8; QK_K];
    let mut s = [0f32; QK_K / 16];
    let mut m = [0f32; QK_K / 16];
    let mut acc = [0f32; TC];
    for (bi, blk) in row.chunks_exact(ts).enumerate() {
        let has_min = decode_i8(kind, blk, &mut q[..bs], &mut s[..bs / 16], &mut m[..bs / 16]);
        let e0 = bi * bs;
        for (c, a) in acc.iter_mut().enumerate() {
            let xq = unsafe { x.q.add(c * x.n_in + e0) };
            let xs = unsafe { x.s.add(c * x.n_in / QBLOCK + e0 / QBLOCK) };
            let mut sum = 0f32;
            for g in 0..bs / 16 {
                let isum = unsafe { dot16(q.as_ptr().add(16 * g), xq.add(16 * g)) };
                sum += s[g] * unsafe { *xs.add(g / 2) } * isum as f32;
            }
            if has_min {
                let xh = unsafe { x.h.add(c * x.n_in / 16 + e0 / 16) };
                for g in 0..bs / 16 {
                    sum += m[g] * unsafe { *xh.add(g) };
                }
            }
            *a += sum;
        }
    }
    acc
}

impl GgufMatrix {
    /// Rows `r0..r1` against activation rows `c0..c0+TC`, into `y[c * ys + r]` (c from 0).
    unsafe fn rows_impl<const TC: usize>(&self, xa: &QAct, c0: usize, y: *mut f32, ys: usize, r0: usize, r1: usize) {
        let x = Act::new(xa, c0);
        if self.kind.is_float() {
            // Not reached through the matmul driver (float rows take the f32 path); kept correct.
            let mut w = vec![0f32; self.n_in];
            let bpr = self.n_in / QBLOCK;
            for r in r0..r1 {
                self.dequantize_rows(r, r + 1, &mut w);
                for c in 0..TC {
                    let mut acc = 0f32;
                    for (i, &wv) in w.iter().enumerate() {
                        acc += wv * unsafe { *x.q.add(c * self.n_in + i) as f32 * *x.s.add(c * bpr + i / QBLOCK) };
                    }
                    unsafe { *y.add(c * ys + r) = acc };
                }
            }
            return;
        }
        #[cfg(target_arch = "aarch64")]
        if unsafe { neon::rows::<TC>(self, &x, y, ys, r0, r1) } {
            return;
        }
        for r in r0..r1 {
            let o = unsafe { dot_row_generic::<TC>(self.kind, self.row(r), &x) };
            for c in 0..TC {
                unsafe { *y.add(c * ys + r) = o[c] };
            }
        }
    }
}

impl QMat for GgufMatrix {
    fn n_out(&self) -> usize {
        self.n_out
    }
    fn n_in(&self) -> usize {
        self.n_in
    }
    fn bytes(&self) -> usize {
        self.data.len()
    }
    fn dequantize_rows(&self, r0: usize, r1: usize, out: &mut [f32]) {
        GgufMatrix::dequantize_rows(self, r0, r1, out)
    }
    fn dense(&self) -> bool {
        self.kind.is_float()
    }
    unsafe fn rows_t1(&self, xa: &QAct, c: usize, y: *mut f32, r0: usize, r1: usize) {
        unsafe { self.rows_impl::<1>(xa, c, y, 0, r0, r1) }
    }
    fn multi(&self, t: usize) -> Option<bool> {
        #[cfg(target_arch = "aarch64")]
        {
            neon::multi_kind(self.kind, t)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let _ = t;
            None
        }
    }
    unsafe fn rows_multi(&self, xa: &QAct, pairs: Option<&PairAct>, t: usize, y: *mut f32, r0: usize, r1: usize) -> bool {
        #[cfg(target_arch = "aarch64")]
        {
            unsafe { neon::rows_multi(self, &Act::new(xa, 0), pairs, t, y, self.n_out, r0, r1) }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let _ = (xa, pairs, t, y, r0, r1);
            false
        }
    }
    unsafe fn rows_tc(&self, xa: &QAct, c0: usize, tc: usize, y: *mut f32, r0: usize, r1: usize) {
        let (n, y) = (self.n_out, unsafe { y.add(c0 * self.n_out) });
        unsafe {
            match tc {
                2 => self.rows_impl::<2>(xa, c0, y, n, r0, r1),
                3 => self.rows_impl::<3>(xa, c0, y, n, r0, r1),
                4 => self.rows_impl::<4>(xa, c0, y, n, r0, r1),
                _ => unreachable!("bad tc {tc}"),
            }
        }
    }
}

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *seed >> 33
}

fn put_f16(b: &mut [u8], i: usize, v: f32) {
    b[i..i + 2].copy_from_slice(&half::f16::from_f32(v).to_bits().to_le_bytes());
}

/// `nb` random blocks of `kind` with sane scales (test data for kernels and benchmarks): random bytes, then every f16 scale field
/// set to a random value of magnitude 2^-10..2^-4 (`signed`: either sign).
pub fn random_blocks(kind: GgmlType, nb: usize, seed: &mut u64) -> Vec<u8> {
    use GgmlType::*;
    let ts = kind.type_size();
    let mut v: Vec<u8> = (0..nb * ts).map(|_| lcg(seed) as u8).collect();
    let rnd = |seed: &mut u64, signed: bool| {
        // Built from bits (2^-10 .. 2^-4) so the fixtures are identical on every target.
        let mag = f32::from_bits(((117 + lcg(seed) % 6) << 23) as u32 | (lcg(seed) as u32 & 0x7f_ffff));
        if signed && lcg(seed) & 1 == 1 { -mag } else { mag }
    };
    for b in v.chunks_exact_mut(ts) {
        match kind {
            F32 => {
                let x = rnd(seed, true) * 64.0;
                b.copy_from_slice(&x.to_le_bytes());
            }
            F16 => put_f16(b, 0, rnd(seed, true) * 64.0),
            BF16 => b.copy_from_slice(&half::bf16::from_f32(rnd(seed, true) * 64.0).to_bits().to_le_bytes()),
            Q4_1 | Q5_1 => {
                put_f16(b, 0, rnd(seed, false));
                put_f16(b, 2, rnd(seed, true) * 8.0);
            }
            Q2_K => {
                put_f16(b, 80, rnd(seed, false));
                put_f16(b, 82, rnd(seed, false));
            }
            Q3_K => put_f16(b, 108, rnd(seed, true)),
            Q4_K | Q5_K => {
                put_f16(b, 0, rnd(seed, false));
                put_f16(b, 2, rnd(seed, false));
            }
            Q6_K => put_f16(b, 208, rnd(seed, true)),
            IQ1_M => {
                let d = half::f16::from_f32(rnd(seed, true)).to_bits();
                for k in 0..4 {
                    let i = 48 + 2 * k;
                    let s = (u16::from_le_bytes([b[i], b[i + 1]]) & 0x0fff) | (((d >> (4 * k)) & 0xF) << 12);
                    b[i..i + 2].copy_from_slice(&s.to_le_bytes());
                }
            }
            _ => put_f16(b, 0, rnd(seed, true)),
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::matmul_quant_multi;

    fn fnv(v: &[f32]) -> u64 {
        let mut h = 0xcbf29ce484222325u64;
        for x in v {
            for b in x.to_bits().to_le_bytes() {
                h = (h ^ b as u64).wrapping_mul(0x100000001b3);
            }
        }
        h
    }

    const FIXTURE_ELEMS: usize = 1024;

    /// The raw blocks behind `dequantization_matches_ggml` (seed depends on the type id).
    fn fixture(kind: GgmlType) -> Vec<u8> {
        let mut seed = 1000 + kind.id() as u64;
        random_blocks(kind, FIXTURE_ELEMS / kind.block_size(), &mut seed)
    }

    /// FNV-1a hashes of the f32 bits of gguf-py's `gguf.quants.dequantize` on `fixture(kind)`
    /// (computed with `OXINFER_GGML_DUMP=dir` + tools/ggml_ref.py).
    const REF_HASHES: &[(GgmlType, u64)] = &[
        (GgmlType::BF16, 0x767e77c423088e31),
        (GgmlType::F16, 0xd03c69ee7934955f),
        (GgmlType::F32, 0x8c4157552256808b),
        (GgmlType::IQ1_M, 0x195615d8aa814326),
        (GgmlType::IQ1_S, 0xd6aa790ae5e0991e),
        (GgmlType::IQ2_S, 0xc5aee549b0fd9b52),
        (GgmlType::IQ2_XS, 0xc35698d5139657fd),
        (GgmlType::IQ2_XXS, 0x06589829ff33d4e4),
        (GgmlType::IQ3_S, 0x9ac8f077e004fefd),
        (GgmlType::IQ3_XXS, 0x9a7e32aaa9e093ca),
        (GgmlType::IQ4_NL, 0x9de348f3e6e39671),
        (GgmlType::IQ4_XS, 0xf26fd3824913af96),
        (GgmlType::Q2_K, 0x44b0a081c4b9c71c),
        (GgmlType::Q3_K, 0x0c8fa80fd8589b8a),
        (GgmlType::Q4_0, 0xc2ac60f0d9d22bc8),
        (GgmlType::Q4_1, 0x8a49b60459c03fa2),
        (GgmlType::Q4_K, 0xde4da39c1d9096b2),
        (GgmlType::Q5_0, 0x46640e791feaf19f),
        (GgmlType::Q5_1, 0xb0fd98e61b74d61d),
        (GgmlType::Q5_K, 0x4be2f59b66cc7042),
        (GgmlType::Q6_K, 0x20a51cc0feaae638),
        (GgmlType::Q8_0, 0x3199be3deddf5c1a),
    ];

    #[test]
    fn dequantization_matches_ggml() {
        if let Ok(dir) = std::env::var("OXINFER_GGML_DUMP") {
            for kind in GgmlType::ALL {
                let raw = fixture(kind);
                let mut y = vec![0f32; FIXTURE_ELEMS];
                dequantize_row(kind, &raw, &mut y);
                std::fs::write(format!("{dir}/{}.bin", kind.name()), &raw).unwrap();
                let bytes: Vec<u8> = y.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(format!("{dir}/{}.ours", kind.name()), bytes).unwrap();
            }
        }
        assert_eq!(REF_HASHES.len(), GgmlType::ALL.len());
        for &(kind, want) in REF_HASHES {
            let mut y = vec![0f32; FIXTURE_ELEMS];
            dequantize_row(kind, &fixture(kind), &mut y);
            assert_eq!(fnv(&y), want, "{kind}: dequantized values differ from ggml");
        }
    }

    /// Real tensors (`OXINFER_GGML_REAL=dir` of NAME.bin raw rows, NAME.ref gguf-py values and
    /// NAME.meta "type rows n_in"): bit-exact rows, and matmuls against the f64 product.
    #[test]
    fn real_gguf_tensors() {
        let Ok(dir) = std::env::var("OXINFER_GGML_REAL") else { return };
        crate::pool::init(4);
        let mut seed = 3u64;
        for e in std::fs::read_dir(&dir).unwrap() {
            let p = e.unwrap().path();
            if p.extension().is_none_or(|x| x != "meta") {
                continue;
            }
            let meta = std::fs::read_to_string(&p).unwrap();
            let v: Vec<usize> = meta.split_whitespace().map(|x| x.parse().unwrap()).collect();
            let kind = GgmlType::from_id(v[0] as u32).unwrap();
            let (rows, n_in) = (v[1], v[2]);
            let raw = std::fs::read(p.with_extension("bin")).unwrap();
            let want: Vec<f32> = std::fs::read(p.with_extension("ref"))
                .unwrap()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            let m = GgufMatrix::from_bytes(kind, &raw, rows, n_in).unwrap();
            let mut wf = vec![0f32; rows * n_in];
            m.dequantize_rows(0, rows, &mut wf);
            assert!(
                wf.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()),
                "{p:?}: dequantization differs"
            );
            for t in [1usize, 3, 9] {
                let x: Vec<f32> = (0..t * n_in).map(|_| (lcg(&mut seed) % 2001) as f32 / 1000.0 - 1.0).collect();
                let mut y = vec![0f32; t * rows];
                matmul_quant_multi(&mut [(&m, &mut y[..])], &x, t);
                let yr = f32_reference(&wf, rows, n_in, &x, t);
                let rms = (yr.iter().map(|v| v * v).sum::<f32>() / yr.len() as f32).sqrt();
                let err = (y.iter().zip(&yr).map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / yr.len() as f32).sqrt();
                eprintln!(
                    "{} {kind} t={t}: rms error {:.2e} of rms {rms:.3e}",
                    p.file_stem().unwrap().to_string_lossy(),
                    err / rms
                );
                assert!(err <= 2e-2 * rms, "{p:?} t={t}");
            }
        }
    }

    /// `GgufMatrix::dequantize_rows` (NEON on aarch64) is bit-identical to `dequantize_row`.
    #[test]
    fn row_dequantization_paths_agree() {
        for kind in GgmlType::ALL {
            let raw = fixture(kind);
            let n = FIXTURE_ELEMS / 4;
            let m = GgufMatrix::from_bytes(kind, &raw, 4, n).unwrap();
            let mut fast = vec![0f32; FIXTURE_ELEMS];
            m.dequantize_rows(0, 4, &mut fast);
            let mut want = vec![0f32; FIXTURE_ELEMS];
            dequantize_row(kind, &raw, &mut want);
            assert!(fast.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "{kind}");
        }
    }

    /// `decode_i8` must describe the same values as `dequantize_row`.
    #[test]
    fn int8_decoding_matches_dequantization() {
        for kind in GgmlType::ALL.into_iter().filter(|k| !k.is_float()) {
            let raw = fixture(kind);
            let mut want = vec![0f32; FIXTURE_ELEMS];
            dequantize_row(kind, &raw, &mut want);
            let bs = kind.block_size();
            let (mut q, mut s, mut m) = ([0i8; QK_K], [0f32; 16], [0f32; 16]);
            for (bi, blk) in raw.chunks_exact(kind.type_size()).enumerate() {
                m.fill(0.0);
                decode_i8(kind, blk, &mut q[..bs], &mut s[..bs / 16], &mut m[..bs / 16]);
                for i in 0..bs {
                    let v = s[i / 16] * q[i] as f32 + m[i / 16];
                    let w = want[bi * bs + i];
                    assert!((v - w).abs() <= 1e-6 * w.abs().max(1e-3), "{kind} block {bi} elem {i}: {v} vs {w}");
                }
            }
        }
    }

    #[test]
    fn type_ids_round_trip() {
        for kind in GgmlType::ALL {
            assert_eq!(GgmlType::from_id(kind.id()), Some(kind));
        }
        assert_eq!(GgmlType::from_id(15), None); // Q8_K is not a weight format
        assert_eq!(GgmlType::Q4_K.row_bytes(5120), 20 * 144);
    }

    fn check(name: &str, y: &[f32], yr: &[f32], tol: f32) {
        let (mut max, mut scale) = (0f32, 0f32);
        for (a, b) in y.iter().zip(yr) {
            max = max.max((a - b).abs());
            scale = scale.max(b.abs());
        }
        assert!(max <= tol * scale.max(1e-3), "{name}: max abs diff {max} (scale {scale})");
    }

    /// What the int8 decode path computes, in f64: per 16 elements `s * sum(q * xq) * xs`, plus
    /// `m * sum(x)` over the exact activations for formats with offsets.
    fn int8_reference(m: &GgufMatrix, x: &[f32], t: usize) -> Vec<f32> {
        let (n_out, n_in, kind) = (m.n_out, m.n_in, m.kind);
        let xa = crate::quant::quantize_act(x, t, n_in);
        let bs = kind.block_size();
        let mut y = vec![0f32; t * n_out];
        let (mut q, mut s, mut mn) = ([0i8; QK_K], [0f32; 16], [0f32; 16]);
        for r in 0..n_out {
            for c in 0..t {
                let mut acc = 0f64;
                for (bi, blk) in m.row(r).chunks_exact(kind.type_size()).enumerate() {
                    let has_min = decode_i8(kind, blk, &mut q[..bs], &mut s[..bs / 16], &mut mn[..bs / 16]);
                    for g in 0..bs / 16 {
                        let e = c * n_in + bi * bs + 16 * g;
                        let isum: i32 = (0..16).map(|i| q[16 * g + i] as i32 * xa.q[e + i] as i32).sum();
                        acc += s[g] as f64 * isum as f64 * xa.scales[e / QBLOCK] as f64;
                        if has_min {
                            acc += mn[g] as f64 * x[e..e + 16].iter().map(|&v| v as f64).sum::<f64>();
                        }
                    }
                }
                y[c * n_out + r] = acc as f32;
            }
        }
        y
    }

    fn f32_reference(wf: &[f32], n_out: usize, n_in: usize, x: &[f32], t: usize) -> Vec<f32> {
        let mut y = vec![0f32; t * n_out];
        for c in 0..t {
            for r in 0..n_out {
                let w = &wf[r * n_in..(r + 1) * n_in];
                y[c * n_out + r] = w.iter().zip(&x[c * n_in..(c + 1) * n_in]).map(|(a, b)| (a * b) as f64).sum::<f64>() as f32;
            }
        }
        y
    }

    /// A fused batch mixing GGUF types with Q8 and borrowed BF16 (`Lin::from_gguf`) gives the
    /// same result as each layer alone.
    #[test]
    fn mixed_format_batches_split() {
        crate::pool::init(2);
        let (n_in, mut seed) = (2 * QK_K, 9u64);
        let raw: Vec<(GgmlType, usize, Vec<u8>)> = [
            (GgmlType::Q4_K, 24),
            (GgmlType::Q6_K, 8),
            (GgmlType::BF16, 16),
            (GgmlType::IQ3_S, 8),
        ]
        .into_iter()
        .map(|(k, n)| (k, n, random_blocks(k, n * n_in / k.block_size(), &mut seed)))
        .collect();
        let mut lins: Vec<(crate::model::Lin, usize)> = raw
            .iter()
            .map(|(k, n, b)| (crate::model::Lin::from_gguf(*k, b, *n, n_in).unwrap(), *n))
            .collect();
        assert!(matches!(lins[2].0.w, crate::model::Weight::Bf16(..)));
        let wb: Vec<u16> = (0..16 * n_in)
            .map(|i| half::bf16::from_f32((i % 7) as f32 * 0.01 - 0.03).to_bits())
            .collect();
        lins.push((crate::model::Lin::q8(crate::quant::quantize_bf16(&wb, 16, n_in)), 16));
        for t in [1usize, 3, 9] {
            let x: Vec<f32> = (0..t * n_in).map(|_| (lcg(&mut seed) % 2001) as f32 / 1000.0 - 1.0).collect();
            let mut alone: Vec<Vec<f32>> = Vec::new();
            for (l, n) in &lins {
                let mut y = vec![0f32; t * n];
                crate::model::linear(l, *n, n_in, &x, t, &mut y);
                alone.push(y);
            }
            let mut ys: Vec<Vec<f32>> = lins.iter().map(|(_, n)| vec![0f32; t * n]).collect();
            let mut jobs: Vec<(&crate::model::Lin, usize, &mut [f32])> =
                lins.iter().zip(ys.iter_mut()).map(|((l, n), y)| (l, *n, &mut y[..])).collect();
            crate::model::linear_multi(&mut jobs, n_in, &x, t);
            assert_eq!(ys, alone, "t={t}");
        }
    }

    /// The multi-token decode kernels (SDOT and, where the CPU has it, SMMLA) give every row
    /// bit-identical to the one-token kernel, for every token count.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn multi_token_kernels_are_batch_invariant() {
        let (n_out, n_in) = (21usize, 3 * QK_K);
        let mm = crate::kernels::cpu_has_i8mm();
        for kind in GgmlType::ALL.into_iter().filter(|k| neon::multi_kind(*k, 2).is_some()) {
            let mut seed = 31 + kind.id() as u64;
            let m = GgufMatrix::from_bytes(kind, &random_blocks(kind, n_out * n_in / kind.block_size(), &mut seed), n_out, n_in).unwrap();
            for t in 2..=crate::ggml::neon::MULTI_MAX {
                let x: Vec<f32> = (0..t * n_in).map(|_| (lcg(&mut seed) % 2001) as f32 / 1000.0 - 1.0).collect();
                let xa = crate::quant::quantize_act(&x, t, n_in);
                let pairs = crate::quant::PairAct::new(&xa);
                let mut y1 = vec![0f32; t * n_out];
                for c in 0..t {
                    unsafe { m.rows_t1(&xa, c, y1.as_mut_ptr().add(c * n_out), 0, n_out) };
                }
                let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
                for p in [None, Some(&pairs)] {
                    if p.is_some() && !(mm && neon::multi_kind(kind, t) == Some(true)) {
                        continue;
                    }
                    let mut y = vec![0f32; t * n_out];
                    // Two row ranges, the first odd-sized (a row-pair tail inside the matrix).
                    unsafe {
                        assert!(neon::rows_multi(&m, &Act::new(&xa, 0), p, t, y.as_mut_ptr(), n_out, 0, 7));
                        assert!(neon::rows_multi(&m, &Act::new(&xa, 0), p, t, y.as_mut_ptr(), n_out, 7, n_out));
                    }
                    assert_eq!(bits(&y), bits(&y1), "{kind} t={t} smmla={}", p.is_some());
                }
            }
        }
    }

    /// The portable dot product (used where no SIMD kernel exists) against the reference, also on
    /// targets where the matmul driver would pick a NEON kernel.
    #[test]
    fn portable_dot_matches_reference() {
        let (n_out, n_in, t) = (5usize, 2 * QK_K, 3usize);
        for kind in GgmlType::ALL.into_iter().filter(|k| !k.is_float()) {
            let mut seed = 5 + kind.id() as u64;
            let m = GgufMatrix::from_bytes(kind, &random_blocks(kind, n_out * n_in / kind.block_size(), &mut seed), n_out, n_in).unwrap();
            let x: Vec<f32> = (0..t * n_in).map(|_| (lcg(&mut seed) % 2001) as f32 / 1000.0 - 1.0).collect();
            let xa = crate::quant::quantize_act(&x, t, n_in);
            let act = Act::new(&xa, 0);
            let mut y = vec![0f32; t * n_out];
            for r in 0..n_out {
                let o = unsafe { dot_row_generic::<3>(kind, m.row(r), &act) };
                for c in 0..t {
                    y[c * n_out + r] = o[c];
                }
            }
            check(&format!("{kind} portable"), &y, &int8_reference(&m, &x, t), 1e-4);
        }
    }

    /// Decode (t = 1..4, int8 activations: NEON and portable kernels), prefill (t >= 8, f32 GEMM)
    /// and grouped matmuls against references, for every type.
    #[test]
    fn gguf_matmuls_match_reference() {
        crate::pool::init(3);
        let n_out = 21usize;
        for kind in GgmlType::ALL {
            // Block-of-32 formats get a block count that is not a multiple of 4 (tail loops).
            let n_in = if kind.block_size() == QK_K { 3 * QK_K } else { 3 * QK_K + 96 };
            let mut seed = 77 + kind.id() as u64;
            let raw = random_blocks(kind, n_out * n_in / kind.block_size(), &mut seed);
            let m = GgufMatrix::from_bytes(kind, &raw, n_out, n_in).unwrap();
            let mut wf = vec![0f32; n_out * n_in];
            m.dequantize_rows(0, n_out, &mut wf);
            for t in [1usize, 2, 3, 4, 5, 8, 11] {
                let x: Vec<f32> = (0..t * n_in).map(|_| (lcg(&mut seed) % 2001) as f32 / 1000.0 - 1.0).collect();
                let yr = if t >= 8 || kind.is_float() {
                    f32_reference(&wf, n_out, n_in, &x, t)
                } else {
                    int8_reference(&m, &x, t)
                };
                let mut y = vec![0f32; t * n_out];
                matmul_quant_multi(&mut [(&m, &mut y[..])], &x, t);
                check(&format!("{kind} t={t}"), &y, &yr, 1e-4);
                // Sanity check of the int8 path against the exact product.
                check(&format!("{kind} t={t} vs f32"), &y, &f32_reference(&wf, n_out, n_in, &x, t), 3e-2);
            }
            // Grouped (MoE) prefill: two experts of this type over 8 + 16 tokens.
            let m2 = GgufMatrix::from_bytes(kind, &random_blocks(kind, n_out * n_in / kind.block_size(), &mut seed), n_out, n_in).unwrap();
            let mut wf2 = vec![0f32; n_out * n_in];
            m2.dequantize_rows(0, n_out, &mut wf2);
            let x: Vec<f32> = (0..24 * n_in).map(|_| (lcg(&mut seed) % 2001) as f32 / 1000.0 - 1.0).collect();
            let mut y = vec![0f32; 24 * n_out];
            crate::kernels::matmul_quant_grouped(&[&m, &m2], &[(0, 8), (8, 16)], &x, &mut y);
            let mut yr = f32_reference(&wf, n_out, n_in, &x[..8 * n_in], 8);
            yr.extend(f32_reference(&wf2, n_out, n_in, &x[8 * n_in..], 16));
            check(&format!("{kind} grouped"), &y, &yr, 1e-4);
        }
    }
}
