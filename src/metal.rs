//! Metal compute backend (Apple Silicon), reached through the Objective-C runtime with `dlsym`:
//! no Metal framework is linked and nothing is needed at build time. Shaders are compiled from
//! source at start-up. Weights live in page-aligned host memory (quantized matrices and whole
//! safetensors blobs) and are wrapped as shared `MTLBuffer`s without copying (unified memory), so
//! "uploading" a model costs nothing.
//!
//! The `Backend` matmuls here take prefill-sized passes: with ~0.15 ms per Metal submission,
//! offloading single-token matmuls one by one is a loss. Single-token decode instead runs the
//! whole forward pass in one command buffer (`metal_decode.rs`, through `gpu::decode` graphs).
//! Prefill is compute-bound and the GEMM below reaches ~11 TFLOP/s against ~1.3 on the CPU.
//!
//! GEMM: a threadgroup of 128 threads (2x2 SIMD-groups) computes a 64 (out) x 64 (tokens) tile.
//! Per 32-wide K block the weights are dequantized to f32 in threadgroup memory; x fragments are
//! read straight from device memory with transposed 8x8 loads; `simdgroup_float8x8` FMAs do the
//! math. Accumulators must be named variables: arrays of `simdgroup_float8x8` are spilled to
//! memory by the shader compiler and run 12x slower. Results are bit-identical to the CPU path.
//! The weight formats: bf16, oxinfer's Q8 / Q4, the GGUF block types, and FP8 E4M3 with 128x128
//! block scales (the official Mistral Large 4 checkpoint), each read in place.
//!
//! Attention: unmasked heads of 64 (Whisper's encoder, SD 2.x / SD-Turbo) or 128 (FLUX) run as one
//! fused kernel (`flash_attn64` / `flash_attn128`, f32 simdgroup MMA with an online softmax; ~2x faster than the
//! GEMM + softmax + GEMM path, which the other shapes keep, and no per-head operand repacking).
//!
//! On GPUs whose cores have matrix units (M5, A19: the Apple10 family), the GEMMs and the MoE
//! expert layers run on them instead (`MMA_SHADER_SRC`, Metal 4 `matmul2d`), with half-precision
//! operands and f32 accumulation: ~3-5x faster, at a relative error of ~3e-4 per product. So does
//! attention with 128-wide heads (`flash_attn128_mma`, ~2x faster than `flash_attn128`), or with
//! f32 operands under [`crate::gpu::set_exact_attention`] (`flash_attn128_mma_f32`, as accurate as
//! `flash_attn128` and ~10% faster). GPUs without matrix units (M1-M4) get the same GEMM kernels
//! (names, operands and rounding contract: half inputs, f32 sums) on 8x8 SIMD-group matrices, so
//! the prefill plans and the MoE expert GEMMs run there too.
//! Environment: `OXINFER_METAL_MMA=0` keeps the f32 kernels (bit-identical to the CPU), `=1` tries
//! the matrix-unit kernels on any GPU; `OXINFER_METAL_MOE=0` keeps MoE experts on the CPU;
//! `OXINFER_METAL_TRACE=1` prints the copy and GPU time of every submission;
//! `OXINFER_METAL_MIN_T=N` takes matmuls from N tokens on (experiments).

use crate::ggml::GgmlType;
use crate::gpu::{Backend, DeviceWeight, Job, WeightRef};
use crate::platform::gpu::metal::{Id, Objc, Sel};
use crate::quant::PAGE;
use anyhow::{Result, anyhow, bail};
use std::ffi::{CStr, CString, c_void};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

#[path = "metal_decode.rs"]
pub(crate) mod decode;
#[path = "metal_dit.rs"]
mod dit;
#[path = "metal_ds4.rs"]
pub(crate) mod ds4;
#[path = "metal_ds41.rs"]
pub(crate) mod ds41;
#[path = "metal_inkling.rs"]
pub(crate) mod inkling;
#[path = "metal_inkling_draft.rs"]
pub(crate) mod inkling_draft;
#[path = "metal_prefill.rs"]
mod prefill;
#[path = "metal_unet.rs"]
mod unet;
#[path = "metal_whisper.rs"]
mod whisper;

const TILE: usize = 64; // out rows and tokens per threadgroup
const MIN_T: usize = 16; // smaller token counts stay on the CPU
const MIN_FLOPS: f64 = 3e8; // ~0.15 ms submission cost vs ~1 TFLOP/s on the CPU

const SHADER_SRC: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
struct P { uint n_out; uint n_in; uint w_off; uint s_off; };
#define LD 36   // threadgroup row stride (floats): 32 + padding against bank conflicts

// ---- Dequantizers: 32 consecutive k values of one row, as 8 float4s -------------------------
struct Q4W { device const uchar* q; device const half* s; uint bpr; uint half_row; };
struct Q8W { device const char* q; device const float* s; uint bpr; uint n_in; };
struct BfW { device const ushort* w; uint n_in; };

inline float4 nib_lo(uint u) { return float4(float(int(u & 15) - 8), float(int((u >> 8) & 15) - 8), float(int((u >> 16) & 15) - 8), float(int((u >> 24) & 15) - 8)); }
inline float4 nib_hi(uint u) { return float4(float(int((u >> 4) & 15) - 8), float(int((u >> 12) & 15) - 8), float(int((u >> 20) & 15) - 8), float(int((u >> 28) & 15) - 8)); }
inline void load(Q4W w, uint row, uint b, thread float4* out) {
    float d = float(w.s[(ulong)row * w.bpr + b]);
    uint4 v = *(device const uint4*)(w.q + (ulong)row * w.half_row + b * 16);
    out[0] = nib_lo(v.x) * d; out[1] = nib_lo(v.y) * d; out[2] = nib_lo(v.z) * d; out[3] = nib_lo(v.w) * d;
    out[4] = nib_hi(v.x) * d; out[5] = nib_hi(v.y) * d; out[6] = nib_hi(v.z) * d; out[7] = nib_hi(v.w) * d;
}
// Affine Q4 (MLX): a bf16 offset per group of `gk` blocks is added to every element.
struct Q4A { Q4W b; device const ushort* o; uint gpr; uint gk; };
inline void load(Q4A w, uint row, uint b, thread float4* out) {
    load(w.b, row, b, out);
    float m = as_type<float>(uint(w.o[(ulong)row * w.gpr + b / w.gk]) << 16);
    for (uint i = 0; i < 8; i++) out[i] += m;
}
inline void load(Q8W w, uint row, uint b, thread float4* out) {
    float d = w.s[(ulong)row * w.bpr + b];
    device const char4* p = (device const char4*)(w.q + (ulong)row * w.n_in + b * 32);
    for (uint i = 0; i < 8; i++) out[i] = float4(p[i]) * d;
}
inline float4 bf4(packed_ushort4 u4) {
    ushort4 u = ushort4(u4);
    return float4(as_type<float>(uint(u.x) << 16), as_type<float>(uint(u.y) << 16), as_type<float>(uint(u.z) << 16), as_type<float>(uint(u.w) << 16));
}
// f32 "weights": attention keys and transposed values.
struct F32W { device const float* w; uint n_in; };
inline void load(F32W w, uint row, uint b, thread float4* out) {
    device const float4* p = (device const float4*)(w.w + (ulong)row * w.n_in + b * 32);
    for (uint i = 0; i < 8; i++) out[i] = p[i];
}
inline void load(BfW w, uint row, uint b, thread float4* out) {
    // bf16 tensors inside a safetensors blob are only 2-byte aligned: packed loads.
    device const packed_ushort4* p = (device const packed_ushort4*)(w.w + (ulong)row * w.n_in + b * 32);
    for (uint i = 0; i < 8; i++) out[i] = bf4(p[i]);
}

// ---- GEMM ------------------------------------------------------------------------------------
#define MMA(I, J) simdgroup_multiply_accumulate(c##I##J, a##I, b##J, c##I##J)
#define ST(I, J) simdgroup_store(c##I##J, y + (ulong)(t0 + sc + J * 8) * p.n_out + o0 + sr + I * 8, p.n_out, ulong2(0, 0), true)
template <typename W>
inline void gemm_body(W w, device const float* x, device float* y, constant P& p, uint2 tg, uint tid, uint sg, threadgroup float* Ws) {
    uint o0 = tg.x * 64, t0 = tg.y * 64;
    uint nblk = p.n_in / 32;
    simdgroup_float8x8 c00(0.f), c01(0.f), c02(0.f), c03(0.f), c10(0.f), c11(0.f), c12(0.f), c13(0.f),
                       c20(0.f), c21(0.f), c22(0.f), c23(0.f), c30(0.f), c31(0.f), c32(0.f), c33(0.f);
    uint sr = (sg / 2) * 32, sc = (sg % 2) * 32;
    uint drow = tid / 2, dhalf = tid % 2;   // each thread dequantizes half a block of one row
    device const float* xs = x + (ulong)(t0 + sc) * p.n_in;
    for (uint b = 0; b < nblk; b++) {
        float4 wv[8];
        load(w, o0 + drow, b, wv);
        threadgroup float4* wd = (threadgroup float4*)(Ws + drow * LD + dhalf * 16);
        wd[0] = wv[dhalf * 4]; wd[1] = wv[dhalf * 4 + 1]; wd[2] = wv[dhalf * 4 + 2]; wd[3] = wv[dhalf * 4 + 3];
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
        for (uint k = 0; k < 32; k += 8) {
            simdgroup_float8x8 a0, a1, a2, a3, b0, b1, b2, b3;
            simdgroup_load(a0, Ws + (sr + 0) * LD + k, LD);
            simdgroup_load(a1, Ws + (sr + 8) * LD + k, LD);
            simdgroup_load(a2, Ws + (sr + 16) * LD + k, LD);
            simdgroup_load(a3, Ws + (sr + 24) * LD + k, LD);
            simdgroup_load(b0, xs + (ulong)0 * p.n_in + b * 32 + k, p.n_in, ulong2(0, 0), true);
            simdgroup_load(b1, xs + (ulong)8 * p.n_in + b * 32 + k, p.n_in, ulong2(0, 0), true);
            simdgroup_load(b2, xs + (ulong)16 * p.n_in + b * 32 + k, p.n_in, ulong2(0, 0), true);
            simdgroup_load(b3, xs + (ulong)24 * p.n_in + b * 32 + k, p.n_in, ulong2(0, 0), true);
            MMA(0, 0); MMA(0, 1); MMA(0, 2); MMA(0, 3); MMA(1, 0); MMA(1, 1); MMA(1, 2); MMA(1, 3);
            MMA(2, 0); MMA(2, 1); MMA(2, 2); MMA(2, 3); MMA(3, 0); MMA(3, 1); MMA(3, 2); MMA(3, 3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // c_ij is [out 8][tok 8]; y is [tok][n_out], so store transposed.
    ST(0, 0); ST(0, 1); ST(0, 2); ST(0, 3); ST(1, 0); ST(1, 1); ST(1, 2); ST(1, 3);
    ST(2, 0); ST(2, 1); ST(2, 2); ST(2, 3); ST(3, 0); ST(3, 1); ST(3, 2); ST(3, 3);
}
#define GEMM(NAME, WT, MAKE) \
kernel void NAME(device const void* qb [[buffer(0)]], device const void* sb [[buffer(1)]], \
                 device const float* x [[buffer(2)]], device float* y [[buffer(3)]], constant P& p [[buffer(4)]], \
                 uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                 uint sg [[simdgroup_index_in_threadgroup]]) { \
    threadgroup float Ws[64 * LD]; \
    WT w = MAKE; gemm_body<WT>(w, x, y, p, tg, tid, sg, Ws); }
// ---- Convolution as an implicit GEMM -------------------------------------------------------------
// K runs over (ky, kx, ci) with cin % 32 == 0, so each 32-wide block of K is one tap's contiguous
// channel slice. Every threadgroup gathers its 64 pixels x 32 channels into threadgroup memory
// (zeros outside the image), then runs the same simdgroup MMAs as the GEMM.
// mode: 0 plain, 1 group norm (stats per group of cpg channels), 2 layer norm (stats per input
// pixel); st holds (mean, 1/std); gb holds gamma[cin] then beta[cin].
struct CP { uint n_out; uint n_in; uint w_off; uint s_off; uint h; uint w; uint cin; uint k; uint stride; uint ow; uint npix; uint pad; uint mode; uint silu; uint cpg; uint _p; };
inline float4 norm4(float4 v, uint c, uint pin, constant CP& p, device const float2* st, device const float* gb) {
    float o[4] = { v.x, v.y, v.z, v.w };
    for (uint i = 0; i < 4; i++) {
        uint ch = c + i;
        float2 m = p.mode == 1 ? st[ch / p.cpg] : st[pin];
        float y = (o[i] - m.x) * m.y * gb[ch] + gb[p.cin + ch];
        o[i] = p.silu != 0 ? y / (1.0f + exp(-y)) : y;
    }
    return float4(o[0], o[1], o[2], o[3]);
}
template <typename W>
inline void conv_body(W wt, device const float* x, device float* y, constant CP& p, device const float2* st, device const float* gb, uint2 tg, uint tid, uint sg, threadgroup float* Ws, threadgroup float* Xs) {
    uint o0 = tg.x * 64, t0 = tg.y * 64;
    uint nblk = p.n_in / 32;
    simdgroup_float8x8 c00(0.f), c01(0.f), c02(0.f), c03(0.f), c10(0.f), c11(0.f), c12(0.f), c13(0.f),
                       c20(0.f), c21(0.f), c22(0.f), c23(0.f), c30(0.f), c31(0.f), c32(0.f), c33(0.f);
    uint sr = (sg / 2) * 32, sc = (sg % 2) * 32;
    uint drow = tid / 2, dhalf = tid % 2;
    uint pix = t0 + drow;
    int oy = int(pix / p.ow), ox = int(pix % p.ow);
    for (uint b = 0; b < nblk; b++) {
        float4 wv[8];
        load(wt, o0 + drow, b, wv);
        threadgroup float4* wd = (threadgroup float4*)(Ws + drow * LD + dhalf * 16);
        wd[0] = wv[dhalf * 4]; wd[1] = wv[dhalf * 4 + 1]; wd[2] = wv[dhalf * 4 + 2]; wd[3] = wv[dhalf * 4 + 3];
        uint kk = b * 32, tap = kk / p.cin, ci0 = kk - tap * p.cin;
        int ky = int(tap / p.k), kx = int(tap % p.k);
        int iy = oy * int(p.stride) + ky - int(p.pad), ix = ox * int(p.stride) + kx - int(p.pad);
        threadgroup float4* xd = (threadgroup float4*)(Xs + drow * LD + dhalf * 16);
        if (pix < p.npix && iy >= 0 && ix >= 0 && iy < int(p.h) && ix < int(p.w)) {
            device const float4* src = (device const float4*)(x + ((ulong)iy * p.w + ix) * p.cin + ci0 + dhalf * 16);
            float4 v0 = src[0], v1 = src[1], v2 = src[2], v3 = src[3];
            if (p.mode != 0) {
                uint c0 = ci0 + dhalf * 16, pin = uint(iy) * p.w + uint(ix);
                v0 = norm4(v0, c0, pin, p, st, gb); v1 = norm4(v1, c0 + 4, pin, p, st, gb);
                v2 = norm4(v2, c0 + 8, pin, p, st, gb); v3 = norm4(v3, c0 + 12, pin, p, st, gb);
            }
            xd[0] = v0; xd[1] = v1; xd[2] = v2; xd[3] = v3;
        } else {
            xd[0] = 0.f; xd[1] = 0.f; xd[2] = 0.f; xd[3] = 0.f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
        for (uint k = 0; k < 32; k += 8) {
            simdgroup_float8x8 a0, a1, a2, a3, b0, b1, b2, b3;
            simdgroup_load(a0, Ws + (sr + 0) * LD + k, LD);
            simdgroup_load(a1, Ws + (sr + 8) * LD + k, LD);
            simdgroup_load(a2, Ws + (sr + 16) * LD + k, LD);
            simdgroup_load(a3, Ws + (sr + 24) * LD + k, LD);
            simdgroup_load(b0, Xs + (sc + 0) * LD + k, LD, ulong2(0, 0), true);
            simdgroup_load(b1, Xs + (sc + 8) * LD + k, LD, ulong2(0, 0), true);
            simdgroup_load(b2, Xs + (sc + 16) * LD + k, LD, ulong2(0, 0), true);
            simdgroup_load(b3, Xs + (sc + 24) * LD + k, LD, ulong2(0, 0), true);
            MMA(0, 0); MMA(0, 1); MMA(0, 2); MMA(0, 3); MMA(1, 0); MMA(1, 1); MMA(1, 2); MMA(1, 3);
            MMA(2, 0); MMA(2, 1); MMA(2, 2); MMA(2, 3); MMA(3, 0); MMA(3, 1); MMA(3, 2); MMA(3, 3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    ST(0, 0); ST(0, 1); ST(0, 2); ST(0, 3); ST(1, 0); ST(1, 1); ST(1, 2); ST(1, 3);
    ST(2, 0); ST(2, 1); ST(2, 2); ST(2, 3); ST(3, 0); ST(3, 1); ST(3, 2); ST(3, 3);
}
#define CONV(NAME, WT, MAKE) \
kernel void NAME(device const void* qb [[buffer(0)]], device const void* sb [[buffer(1)]], \
                 device const float* x [[buffer(2)]], device float* y [[buffer(3)]], constant CP& p [[buffer(4)]], \
                 device const float2* st [[buffer(5)]], device const float* gb [[buffer(6)]], \
                 uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                 uint sg [[simdgroup_index_in_threadgroup]]) { \
    threadgroup float Ws[64 * LD]; threadgroup float Xs[64 * LD]; \
    WT w = MAKE; conv_body<WT>(w, x, y, p, st, gb, tg, tid, sg, Ws, Xs); }

#define MK_Q4 Q4W{ (device const uchar*)qb + p.w_off, (device const half*)sb + p.s_off, p.n_in / 32, p.n_in / 2 }
#define MK_Q8 Q8W{ (device const char*)qb + p.w_off, (device const float*)sb + p.s_off, p.n_in / 32, p.n_in }
#define MK_BF BfW{ (device const ushort*)qb + p.w_off, p.n_in }
#define MK_F32 F32W{ (device const float*)qb + p.w_off, p.n_in }
GEMM(gemm_bf, BfW, MK_BF) GEMM(gemm_q8, Q8W, MK_Q8) GEMM(gemm_q4, Q4W, MK_Q4) GEMM(gemm_f32, F32W, MK_F32)
// f32 weights for few outputs (MoE routers): 32 outputs x 32 tokens per threadgroup, 4
// SIMD-groups of 16 x 16 reading the weights straight from device memory; the same 8x8 products
// in the same order as gemm_f32, so the same results, with 8x the threadgroups.
kernel void gemm_f32_s(device const void* qb [[buffer(0)]], device const float* x [[buffer(2)]], device float* y [[buffer(3)]],
                       constant P& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                       uint sg [[simdgroup_index_in_threadgroup]]) {
    uint o0 = tg.x * 32 + (sg / 2) * 16, t0 = tg.y * 32 + (sg % 2) * 16;
    device const float* ws = (device const float*)qb + p.w_off + (ulong)o0 * p.n_in;
    device const float* xs = x + (ulong)t0 * p.n_in;
    simdgroup_float8x8 c00(0.f), c01(0.f), c10(0.f), c11(0.f);
    for (uint k = 0; k < p.n_in; k += 8) {
        simdgroup_float8x8 a0, a1, b0, b1;
        simdgroup_load(a0, ws + k, p.n_in);
        simdgroup_load(a1, ws + (ulong)8 * p.n_in + k, p.n_in);
        simdgroup_load(b0, xs + k, p.n_in, ulong2(0, 0), true);
        simdgroup_load(b1, xs + (ulong)8 * p.n_in + k, p.n_in, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(c00, a0, b0, c00); simdgroup_multiply_accumulate(c01, a0, b1, c01);
        simdgroup_multiply_accumulate(c10, a1, b0, c10); simdgroup_multiply_accumulate(c11, a1, b1, c11);
    }
    simdgroup_store(c00, y + (ulong)t0 * p.n_out + o0, p.n_out, ulong2(0, 0), true);
    simdgroup_store(c01, y + (ulong)(t0 + 8) * p.n_out + o0, p.n_out, ulong2(0, 0), true);
    simdgroup_store(c10, y + (ulong)t0 * p.n_out + o0 + 8, p.n_out, ulong2(0, 0), true);
    simdgroup_store(c11, y + (ulong)(t0 + 8) * p.n_out + o0 + 8, p.n_out, ulong2(0, 0), true);
}
// gemm_f32_s for n_in % 32 == 0 (prefill plans' MoE routers), each step's loads issued for 32
// columns of K before its products, which run in gemm_f32_s's order (the same results): the
// router's chain of dependent steps waits on memory 4x less often.
#define GFR_LD(A0, A1, B0, B1, K) \
    simdgroup_float8x8 A0, A1, B0, B1; \
    simdgroup_load(A0, ws + (K), p.n_in); simdgroup_load(A1, ws + (ulong)8 * p.n_in + (K), p.n_in); \
    simdgroup_load(B0, xs + (K), p.n_in, ulong2(0, 0), true); simdgroup_load(B1, xs + (ulong)8 * p.n_in + (K), p.n_in, ulong2(0, 0), true);
#define GFR_MMA(A0, A1, B0, B1) \
    simdgroup_multiply_accumulate(c00, A0, B0, c00); simdgroup_multiply_accumulate(c01, A0, B1, c01); \
    simdgroup_multiply_accumulate(c10, A1, B0, c10); simdgroup_multiply_accumulate(c11, A1, B1, c11);
kernel void gemm_f32_r(device const void* qb [[buffer(0)]], device const float* x [[buffer(2)]], device float* y [[buffer(3)]],
                       constant P& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                       uint sg [[simdgroup_index_in_threadgroup]]) {
    uint o0 = tg.x * 32 + (sg / 2) * 16, t0 = tg.y * 32 + (sg % 2) * 16;
    device const float* ws = (device const float*)qb + p.w_off + (ulong)o0 * p.n_in;
    device const float* xs = x + (ulong)t0 * p.n_in;
    simdgroup_float8x8 c00(0.f), c01(0.f), c10(0.f), c11(0.f);
    for (uint k = 0; k < p.n_in; k += 32) {
        GFR_LD(a0, a1, b0, b1, k) GFR_LD(d0, d1, e0, e1, k + 8)
        GFR_LD(f0, f1, g0, g1, k + 16) GFR_LD(h0, h1, i0, i1, k + 24)
        GFR_MMA(a0, a1, b0, b1) GFR_MMA(d0, d1, e0, e1) GFR_MMA(f0, f1, g0, g1) GFR_MMA(h0, h1, i0, i1)
    }
    simdgroup_store(c00, y + (ulong)t0 * p.n_out + o0, p.n_out, ulong2(0, 0), true);
    simdgroup_store(c01, y + (ulong)(t0 + 8) * p.n_out + o0, p.n_out, ulong2(0, 0), true);
    simdgroup_store(c10, y + (ulong)t0 * p.n_out + o0 + 8, p.n_out, ulong2(0, 0), true);
    simdgroup_store(c11, y + (ulong)(t0 + 8) * p.n_out + o0 + 8, p.n_out, ulong2(0, 0), true);
}
struct OP { uint o_off; uint gpr; uint gk; uint _p; };
kernel void gemm_q4a(device const void* qb [[buffer(0)]], device const void* sb [[buffer(1)]],
                     device const float* x [[buffer(2)]], device float* y [[buffer(3)]], constant P& p [[buffer(4)]],
                     device const ushort* ob [[buffer(5)]], constant OP& op [[buffer(6)]],
                     uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float Ws[64 * LD];
    Q4A w = { MK_Q4, ob + op.o_off, op.gpr, op.gk }; gemm_body<Q4A>(w, x, y, p, tg, tid, sg, Ws); }
CONV(conv_bf, BfW, MK_BF) CONV(conv_q8, Q8W, MK_Q8) CONV(conv_q4, Q4W, MK_Q4) CONV(conv_f32, F32W, MK_F32)

// ---- Normalization statistics and softmax --------------------------------------------------------
// Two passes (mean, then the mean squared deviation) for accuracy.
struct GS { uint n; uint c; uint groups; float eps; };
kernel void group_stats(device const float* x [[buffer(0)]], device float2* st [[buffer(1)]], constant GS& p [[buffer(2)]],
                        uint g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8];
    uint cpg = p.c / p.groups, total = p.n * cpg;
    float s = 0.f;
    for (uint i = tid; i < total; i += 256) s += x[(ulong)(i / cpg) * p.c + g * cpg + i % cpg];
    s = simd_sum(s);
    if (lane == 0) red[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = (red[0] + red[1] + red[2] + red[3] + red[4] + red[5] + red[6] + red[7]) / float(total);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float v = 0.f;
    for (uint i = tid; i < total; i += 256) { float d = x[(ulong)(i / cpg) * p.c + g * cpg + i % cpg] - mean; v += d * d; }
    v = simd_sum(v);
    if (lane == 0) red[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float t = red[0] + red[1] + red[2] + red[3] + red[4] + red[5] + red[6] + red[7];
        st[g] = float2(mean, rsqrt(t / float(total) + p.eps));
    }
}
struct RS { uint c; float eps; };
kernel void row_stats(device const float* x [[buffer(0)]], device float2* st [[buffer(1)]], constant RS& p [[buffer(2)]],
                      uint r [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    device const float* row = x + (ulong)r * p.c;
    float s = 0.f;
    for (uint i = lane; i < p.c; i += 32) s += row[i];
    float mean = simd_sum(s) / float(p.c);
    float v = 0.f;
    for (uint i = lane; i < p.c; i += 32) { float d = row[i] - mean; v += d * d; }
    v = simd_sum(v) / float(p.c);
    if (lane == 0) st[r] = float2(mean, rsqrt(v + p.eps));
}
// Row softmax of scaled scores in place; columns from `cols` (or after the row, if causal) to `ld`
// become 0 so the following product ignores the padding.
struct SM { uint cols; uint ld; float scale; uint causal; };
kernel void softmax_rows(device float* s [[buffer(0)]], constant SM& p [[buffer(1)]],
                         uint r [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    device float* row = s + (ulong)r * p.ld;
    uint lim = p.causal != 0 ? min(r + 1, p.cols) : p.cols;
    float m = -INFINITY;
    for (uint i = lane; i < lim; i += 32) m = max(m, row[i] * p.scale);
    m = simd_max(m);
    float sum = 0.f;
    for (uint i = lane; i < lim; i += 32) { float e = exp(row[i] * p.scale - m); row[i] = e; sum += e; }
    float inv = 1.0f / simd_sum(sum);
    for (uint i = lane; i < p.ld; i += 32) row[i] = i < lim ? row[i] * inv : 0.f;
}

// ---- Fused attention, head size 64, no mask ------------------------------------------------------
// q, k, v and o in the [rows][heads * 64] layout of the model (q and o rows padded to 32). One
// threadgroup per (32 queries, head), one SIMD-group per 8 queries; keys and values stream in
// blocks of 32 through threadgroup memory. Scores (f32 simdgroup MMA), the running max / sum of
// the online softmax and the output accumulators stay on chip: no score matrix in memory.
struct FA { uint tq; uint tk; uint d; float scale; };
#define FLD 68
#define SLD 36
#define FQK(C, QA) \
    simdgroup_load(kt, Ks + 0 * FLD + C * 8, FLD, ulong2(0, 0), true); simdgroup_multiply_accumulate(s0, QA, kt, s0); \
    simdgroup_load(kt, Ks + 8 * FLD + C * 8, FLD, ulong2(0, 0), true); simdgroup_multiply_accumulate(s1, QA, kt, s1); \
    simdgroup_load(kt, Ks + 16 * FLD + C * 8, FLD, ulong2(0, 0), true); simdgroup_multiply_accumulate(s2, QA, kt, s2); \
    simdgroup_load(kt, Ks + 24 * FLD + C * 8, FLD, ulong2(0, 0), true); simdgroup_multiply_accumulate(s3, QA, kt, s3);
#define FPV(J) \
    simdgroup_load(pa, S + J * 8, SLD); \
    simdgroup_load(vt, Vs + J * 8 * FLD + 0, FLD); simdgroup_multiply_accumulate(o0, pa, vt, o0); \
    simdgroup_load(vt, Vs + J * 8 * FLD + 8, FLD); simdgroup_multiply_accumulate(o1, pa, vt, o1); \
    simdgroup_load(vt, Vs + J * 8 * FLD + 16, FLD); simdgroup_multiply_accumulate(o2, pa, vt, o2); \
    simdgroup_load(vt, Vs + J * 8 * FLD + 24, FLD); simdgroup_multiply_accumulate(o3, pa, vt, o3); \
    simdgroup_load(vt, Vs + J * 8 * FLD + 32, FLD); simdgroup_multiply_accumulate(o4, pa, vt, o4); \
    simdgroup_load(vt, Vs + J * 8 * FLD + 40, FLD); simdgroup_multiply_accumulate(o5, pa, vt, o5); \
    simdgroup_load(vt, Vs + J * 8 * FLD + 48, FLD); simdgroup_multiply_accumulate(o6, pa, vt, o6); \
    simdgroup_load(vt, Vs + J * 8 * FLD + 56, FLD); simdgroup_multiply_accumulate(o7, pa, vt, o7);
#define FSCALE \
    simdgroup_load(dg, D, 8); \
    simdgroup_multiply(o0, dg, o0); simdgroup_multiply(o1, dg, o1); simdgroup_multiply(o2, dg, o2); simdgroup_multiply(o3, dg, o3); \
    simdgroup_multiply(o4, dg, o4); simdgroup_multiply(o5, dg, o5); simdgroup_multiply(o6, dg, o6); simdgroup_multiply(o7, dg, o7);
kernel void flash_attn64(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]],
                         device const float* v [[buffer(2)]], device float* o [[buffer(3)]], constant FA& p [[buffer(4)]],
                         uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float Ks[32 * FLD];
    threadgroup float Vs[32 * FLD];
    threadgroup float Ss[4 * 8 * SLD];
    threadgroup float Dg[4 * 64];
    uint q0 = tg.x * 32 + sg * 8;
    ulong hoff = (ulong)tg.y * 64;
    threadgroup float* S = Ss + sg * 8 * SLD;
    threadgroup float* D = Dg + sg * 64;
    for (uint i = lane; i < 64; i += 32) D[i] = 0.0f;
    device const float* qs = q + (ulong)q0 * p.d + hoff;
    simdgroup_float8x8 qa0, qa1, qa2, qa3, qa4, qa5, qa6, qa7;
    simdgroup_load(qa0, qs + 0, p.d); simdgroup_load(qa1, qs + 8, p.d); simdgroup_load(qa2, qs + 16, p.d); simdgroup_load(qa3, qs + 24, p.d);
    simdgroup_load(qa4, qs + 32, p.d); simdgroup_load(qa5, qs + 40, p.d); simdgroup_load(qa6, qs + 48, p.d); simdgroup_load(qa7, qs + 56, p.d);
    simdgroup_float8x8 o0(0.f), o1(0.f), o2(0.f), o3(0.f), o4(0.f), o5(0.f), o6(0.f), o7(0.f);
    simdgroup_float8x8 kt, vt, pa, dg;
    uint r = lane / 4, part = lane % 4;
    float m = -INFINITY, l = 0.0f;
    for (uint kb = 0; kb < p.tk; kb += 32) {
        for (uint i = tid; i < 32 * 16; i += 128) {
            uint row = i / 16, c4 = i % 16, key = kb + row;
            float4 kv = 0.0f, vv = 0.0f;
            if (key < p.tk) {
                kv = ((device const float4*)(k + (ulong)key * p.d + hoff))[c4];
                vv = ((device const float4*)(v + (ulong)key * p.d + hoff))[c4];
            }
            *(threadgroup float4*)(Ks + row * FLD + c4 * 4) = kv;
            *(threadgroup float4*)(Vs + row * FLD + c4 * 4) = vv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 s0(0.f), s1(0.f), s2(0.f), s3(0.f);
        FQK(0, qa0) FQK(1, qa1) FQK(2, qa2) FQK(3, qa3) FQK(4, qa4) FQK(5, qa5) FQK(6, qa6) FQK(7, qa7)
        simdgroup_store(s0, S + 0, SLD); simdgroup_store(s1, S + 8, SLD); simdgroup_store(s2, S + 16, SLD); simdgroup_store(s3, S + 24, SLD);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        // Online softmax: 4 lanes per query row, 8 keys each.
        float x[8];
        float bm = -INFINITY;
        for (uint j = 0; j < 8; j++) {
            uint col = part * 8 + j;
            x[j] = kb + col < p.tk ? S[r * SLD + col] * p.scale : -INFINITY;
            bm = max(bm, x[j]);
        }
        bm = max(bm, simd_shuffle_xor(bm, 1));
        bm = max(bm, simd_shuffle_xor(bm, 2));
        float mn = max(m, bm);
        float alpha = exp(m - mn);
        float ps = 0.0f;
        for (uint j = 0; j < 8; j++) { float e = exp(x[j] - mn); S[r * SLD + part * 8 + j] = e; ps += e; }
        ps += simd_shuffle_xor(ps, 1);
        ps += simd_shuffle_xor(ps, 2);
        l = l * alpha + ps;
        m = mn;
        if (part == 0) D[r * 9] = alpha;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        FSCALE
        FPV(0) FPV(1) FPV(2) FPV(3)
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (part == 0) D[r * 9] = 1.0f / l;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    FSCALE
    device float* os = o + (ulong)q0 * p.d + hoff;
    simdgroup_store(o0, os + 0, p.d); simdgroup_store(o1, os + 8, p.d); simdgroup_store(o2, os + 16, p.d); simdgroup_store(o3, os + 24, p.d);
    simdgroup_store(o4, os + 32, p.d); simdgroup_store(o5, os + 40, p.d); simdgroup_store(o6, os + 48, p.d); simdgroup_store(o7, os + 56, p.d);
}

// ---- Fused attention, head size 128, no mask (FLUX, Qwen-Image, ...) ---------------------------
// As flash_attn64 with twice the head width: keys and values stream in blocks of 16 (two 16 x 128
// f32 tiles fit the 32 KB of threadgroup memory), 4 lanes per query row hold 4 scores each.
#define FLD2 132
#define SLD2 20
#define FQK2(C, QA) \
    simdgroup_load(kt, Ks + 0 * FLD2 + C * 8, FLD2, ulong2(0, 0), true); simdgroup_multiply_accumulate(s0, QA, kt, s0); \
    simdgroup_load(kt, Ks + 8 * FLD2 + C * 8, FLD2, ulong2(0, 0), true); simdgroup_multiply_accumulate(s1, QA, kt, s1);
#define PV2(J, C, O) simdgroup_load(vt, Vs + J * 8 * FLD2 + C * 8, FLD2); simdgroup_multiply_accumulate(O, pa, vt, O);
#define FPV2(J) \
    simdgroup_load(pa, S + J * 8, SLD2); \
    PV2(J, 0, o0) PV2(J, 1, o1) PV2(J, 2, o2) PV2(J, 3, o3) PV2(J, 4, o4) PV2(J, 5, o5) PV2(J, 6, o6) PV2(J, 7, o7) \
    PV2(J, 8, o8) PV2(J, 9, o9) PV2(J, 10, o10) PV2(J, 11, o11) PV2(J, 12, o12) PV2(J, 13, o13) PV2(J, 14, o14) PV2(J, 15, o15)
#define FSCALE2 \
    simdgroup_load(dg, D, 8); \
    simdgroup_multiply(o0, dg, o0); simdgroup_multiply(o1, dg, o1); simdgroup_multiply(o2, dg, o2); simdgroup_multiply(o3, dg, o3); \
    simdgroup_multiply(o4, dg, o4); simdgroup_multiply(o5, dg, o5); simdgroup_multiply(o6, dg, o6); simdgroup_multiply(o7, dg, o7); \
    simdgroup_multiply(o8, dg, o8); simdgroup_multiply(o9, dg, o9); simdgroup_multiply(o10, dg, o10); simdgroup_multiply(o11, dg, o11); \
    simdgroup_multiply(o12, dg, o12); simdgroup_multiply(o13, dg, o13); simdgroup_multiply(o14, dg, o14); simdgroup_multiply(o15, dg, o15);
#define FST2(O, C) simdgroup_store(O, os + C * 8, p.d);
kernel void flash_attn128(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]],
                          device const float* v [[buffer(2)]], device float* o [[buffer(3)]], constant FA& p [[buffer(4)]],
                          uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float Ks[16 * FLD2];
    threadgroup float Vs[16 * FLD2];
    threadgroup float Ss[4 * 8 * SLD2];
    threadgroup float Dg[4 * 64];
    uint q0 = tg.x * 32 + sg * 8;
    ulong hoff = (ulong)tg.y * 128;
    threadgroup float* S = Ss + sg * 8 * SLD2;
    threadgroup float* D = Dg + sg * 64;
    for (uint i = lane; i < 64; i += 32) D[i] = 0.0f;
    device const float* qs = q + (ulong)q0 * p.d + hoff;
    simdgroup_float8x8 qa0, qa1, qa2, qa3, qa4, qa5, qa6, qa7, qa8, qa9, qa10, qa11, qa12, qa13, qa14, qa15;
    simdgroup_load(qa0, qs + 0, p.d); simdgroup_load(qa1, qs + 8, p.d); simdgroup_load(qa2, qs + 16, p.d); simdgroup_load(qa3, qs + 24, p.d);
    simdgroup_load(qa4, qs + 32, p.d); simdgroup_load(qa5, qs + 40, p.d); simdgroup_load(qa6, qs + 48, p.d); simdgroup_load(qa7, qs + 56, p.d);
    simdgroup_load(qa8, qs + 64, p.d); simdgroup_load(qa9, qs + 72, p.d); simdgroup_load(qa10, qs + 80, p.d); simdgroup_load(qa11, qs + 88, p.d);
    simdgroup_load(qa12, qs + 96, p.d); simdgroup_load(qa13, qs + 104, p.d); simdgroup_load(qa14, qs + 112, p.d); simdgroup_load(qa15, qs + 120, p.d);
    simdgroup_float8x8 o0(0.f), o1(0.f), o2(0.f), o3(0.f), o4(0.f), o5(0.f), o6(0.f), o7(0.f);
    simdgroup_float8x8 o8(0.f), o9(0.f), o10(0.f), o11(0.f), o12(0.f), o13(0.f), o14(0.f), o15(0.f);
    simdgroup_float8x8 kt, vt, pa, dg;
    uint r = lane / 4, part = lane % 4;
    float m = -INFINITY, l = 0.0f;
    for (uint kb = 0; kb < p.tk; kb += 16) {
        for (uint i = tid; i < 16 * 32; i += 128) {
            uint row = i / 32, c4 = i % 32, key = kb + row;
            float4 kv = 0.0f, vv = 0.0f;
            if (key < p.tk) {
                kv = ((device const float4*)(k + (ulong)key * p.d + hoff))[c4];
                vv = ((device const float4*)(v + (ulong)key * p.d + hoff))[c4];
            }
            *(threadgroup float4*)(Ks + row * FLD2 + c4 * 4) = kv;
            *(threadgroup float4*)(Vs + row * FLD2 + c4 * 4) = vv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 s0(0.f), s1(0.f);
        FQK2(0, qa0) FQK2(1, qa1) FQK2(2, qa2) FQK2(3, qa3) FQK2(4, qa4) FQK2(5, qa5) FQK2(6, qa6) FQK2(7, qa7)
        FQK2(8, qa8) FQK2(9, qa9) FQK2(10, qa10) FQK2(11, qa11) FQK2(12, qa12) FQK2(13, qa13) FQK2(14, qa14) FQK2(15, qa15)
        simdgroup_store(s0, S + 0, SLD2); simdgroup_store(s1, S + 8, SLD2);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        float x[4];
        float bm = -INFINITY;
        for (uint j = 0; j < 4; j++) {
            uint col = part * 4 + j;
            x[j] = kb + col < p.tk ? S[r * SLD2 + col] * p.scale : -INFINITY;
            bm = max(bm, x[j]);
        }
        bm = max(bm, simd_shuffle_xor(bm, 1));
        bm = max(bm, simd_shuffle_xor(bm, 2));
        float mn = max(m, bm);
        float alpha = exp(m - mn);
        float ps = 0.0f;
        for (uint j = 0; j < 4; j++) { float e = exp(x[j] - mn); S[r * SLD2 + part * 4 + j] = e; ps += e; }
        ps += simd_shuffle_xor(ps, 1);
        ps += simd_shuffle_xor(ps, 2);
        l = l * alpha + ps;
        m = mn;
        if (part == 0) D[r * 9] = alpha;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        FSCALE2
        FPV2(0) FPV2(1)
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (part == 0) D[r * 9] = 1.0f / l;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    FSCALE2
    device float* os = o + (ulong)q0 * p.d + hoff;
    FST2(o0, 0) FST2(o1, 1) FST2(o2, 2) FST2(o3, 3) FST2(o4, 4) FST2(o5, 5) FST2(o6, 6) FST2(o7, 7)
    FST2(o8, 8) FST2(o9, 9) FST2(o10, 10) FST2(o11, 11) FST2(o12, 12) FST2(o13, 13) FST2(o14, 14) FST2(o15, 15)
}
"#;

/// GGUF (ggml) block formats for the GEMM: one dequantizer per type producing the 32 values of
/// K block `b` of a row (a sub-block of a 256-element super-block for the K and i-quants). The
/// weights are the raw tensor bytes; `P::s_off` carries the row length in bytes. The i-quant grids
/// (`GGUF_TABLES`) are appended from `ggml_tables.rs` when the library is built.
const GGUF_SHADER_SRC: &str = r#"
struct GgW { device const uchar* q; uint rb; };
#define MK_GG(T) T{ { (device const uchar*)qb + p.w_off, p.s_off } }
// Block `b` (of `bs` bytes) of row `row`: rows of `rb` bytes one after another, or, with rb's top
// bit set (routed experts copied for the grouped GEMMs, `MetalBackend::tiled_experts`), each band
// of 64 rows stored block column by block column (block b of row r at block ((r / 64) * nb + b) *
// 64 + r % 64 of the matrix, nb blocks a row), so that a grouped tile's K step reads one run of
// memory rather than 64 pieces 64 rows apart.
#define GG_TILED 0x80000000u
inline device const uchar* gblk(GgW w, uint row, uint b, uint bs) {
    uint rb = w.rb & ~GG_TILED;
    if (w.rb & GG_TILED) return w.q + (ulong)(row & ~63u) * rb + (ulong)b * (64 * bs) + (row & 63u) * bs;
    return w.q + (ulong)row * rb + (ulong)b * bs;
}
// Tiled Q4_K, Q5_K and Q6_K go further: each band's column of 64 super-blocks of `tsz` bytes is
// stored part by part (what every 32-block of a super-block reads, then the parts later K steps
// read: see `kquant_parts`), the 64 rows' copies of a part one after another, so that a K step
// reads the parts it needs as runs. The part of `len` bytes that starts `off` bytes into a
// super-block (after the parts before it) of row `row`, super-block `b / 8`.
inline device const uchar* kpart(GgW w, uint row, uint b, uint tsz, uint off, uint len) {
    uint rb = w.rb & ~GG_TILED;
    return w.q + (ulong)(row & ~63u) * rb + (ulong)(b / 8) * (64 * tsz) + 64 * off + (row & 63u) * len;
}
inline float h2f(device const uchar* p) { return float(as_type<half>(ushort(ushort(p[0]) | (ushort(p[1]) << 8)))); }
inline uint rd16(device const uchar* p) { return uint(p[0]) | (uint(p[1]) << 8); }
inline uint rd32(device const uchar* p) { return uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) | (uint(p[3]) << 24); }
inline uchar4 ld4(device const uchar* p) { return uchar4(*(device const packed_uchar4*)p); }
inline float4 f4(uchar4 v) { return float4(v); }
inline float4 f4(char4 v) { return float4(v); }
inline device const uchar* sblk(GgW w, uint row, uint b, uint tsz) { return gblk(w, row, b / 8, tsz); }
// 8 values of an i-quant grid entry (bytes, lowest first) with sign bits `s` (bit j: value j negative).
inline void grid8(ulong g, uint s, float db, thread float4* o) {
    float v[8];
    for (uint j = 0; j < 8; j++) v[j] = db * float((g >> (8 * j)) & 255) * ((s >> j) & 1 ? -1.f : 1.f);
    o[0] = float4(v[0], v[1], v[2], v[3]); o[1] = float4(v[4], v[5], v[6], v[7]);
}
inline void grid4x2(uint g1, uint g2, uint s, float db, thread float4* o) {
    grid8(ulong(g1) | (ulong(g2) << 32), s, db, o);
}
inline uint ksign(uint i) { return i | ((popcount(i) & 1) << 7); }

struct Q4_0W { GgW g; };
inline void load(Q4_0W w, uint row, uint b, thread float4* out) {
    device const uchar* B = gblk(w.g, row, b, 18);
    float d = h2f(B);
    for (uint i = 0; i < 4; i++) {
        uchar4 v = ld4(B + 2 + 4 * i);
        out[i] = (f4(v & uchar(15)) - 8.f) * d;
        out[i + 4] = (f4(v >> uchar(4)) - 8.f) * d;
    }
}
struct IQ4_NLW { GgW g; };
inline float4 kv4(uchar4 v) { return float4(kvalues_iq4nl[v.x], kvalues_iq4nl[v.y], kvalues_iq4nl[v.z], kvalues_iq4nl[v.w]); }
inline void load(IQ4_NLW w, uint row, uint b, thread float4* out) {
    device const uchar* B = gblk(w.g, row, b, 18);
    float d = h2f(B);
    for (uint i = 0; i < 4; i++) {
        uchar4 v = ld4(B + 2 + 4 * i);
        out[i] = kv4(v & uchar(15)) * d;
        out[i + 4] = kv4(v >> uchar(4)) * d;
    }
}
struct MXFP4W { GgW g; };
inline float4 kvm4(uchar4 v) { return float4(kvalues_mxfp4[v.x], kvalues_mxfp4[v.y], kvalues_mxfp4[v.z], kvalues_mxfp4[v.w]); }
inline void load(MXFP4W w, uint row, uint b, thread float4* out) {
    device const uchar* B = gblk(w.g, row, b, 17);
    uint e = B[0];
    float d = e < 2 ? as_type<float>(0x00200000u << e) : as_type<float>((e - 1) << 23);
    for (uint i = 0; i < 4; i++) {
        uchar4 v = ld4(B + 1 + 4 * i);
        out[i] = kvm4(v & uchar(15)) * d;
        out[i + 4] = kvm4(v >> uchar(4)) * d;
    }
}
struct Q8_0W { GgW g; };
inline void load(Q8_0W w, uint row, uint b, thread float4* out) {
    device const uchar* B = gblk(w.g, row, b, 34);
    float d = h2f(B);
    for (uint i = 0; i < 8; i++) out[i] = f4(as_type<char4>(ld4(B + 2 + 4 * i))) * d;
}
inline void scale_min_k4(uint j, device const uchar* s, thread float& sc, thread float& mn) {
    if (j < 4) { sc = float(s[j] & 63); mn = float(s[j + 4] & 63); }
    else { sc = float((s[j + 4] & 15) | ((s[j - 4] >> 6) << 4)); mn = float((s[j + 4] >> 4) | ((s[j] >> 6) << 4)); }
}
// Q4_K and Q5_1 blocks are read as words (GGUF tensors are 8-byte aligned, and so are these
// blocks within them; same values as reading bytes): the byte reads were a large part of the
// matrix-unit GEMMs' time.
inline uint byte_of(uint w, uint i) { return (w >> (8 * i)) & 255u; }
// The scale and minimum of 32-block j of a Q4_K / Q5_K super-block from its 12 scale bytes
// (`scale_min_k4`, read as three words).
inline void scale_min_w(uint j, uint s0, uint s1, uint s2, thread float& sc, thread float& mn) {
    if (j < 4) { sc = float(byte_of(s0, j) & 63); mn = float(byte_of(s1, j) & 63); }
    else {
        uint a = byte_of(s2, j - 4);
        sc = float((a & 15) | ((byte_of(s0, j - 4) >> 6) << 4));
        mn = float((a >> 4) | ((byte_of(s1, j - 4) >> 6) << 4));
    }
}
struct Q4_KW { GgW g; };
inline void load(Q4_KW w, uint row, uint b, thread float4* out) {
    uint j = b % 8;
    // Tiled: the 16 bytes of scales, then each pair of 32-blocks' 32 quant bytes.
    bool t = w.g.rb & GG_TILED;
    device const uint* B = (device const uint*)(t ? kpart(w.g, row, b, 144, 0, 16) : sblk(w.g, row, b, 144));
    device const uint2* q = (device const uint2*)(t ? kpart(w.g, row, b, 144, 16 + 32 * (j / 2), 32) : (device const uchar*)(B + 4 + 8 * (j / 2)));
    half2 dm = as_type<half2>(B[0]);
    float sc, mn;
    scale_min_w(j, B[1], B[2], B[3], sc, mn);
    float d1 = float(dm.x) * sc, m1 = float(dm.y) * mn;
    uint sh = 4 * (j % 2);
    for (uint i = 0; i < 4; i++) {
        uint2 u = q[i];
        out[2 * i] = f4(as_type<uchar4>((u.x >> sh) & 0x0F0F0F0Fu)) * d1 - m1;
        out[2 * i + 1] = f4(as_type<uchar4>((u.y >> sh) & 0x0F0F0F0Fu)) * d1 - m1;
    }
}
// Q5_K: as words too (176-byte blocks).
struct Q5_KW { GgW g; };
inline void load(Q5_KW w, uint row, uint b, thread float4* out) {
    uint j = b % 8;
    // Tiled: the 48 bytes of scales and high bits, then each pair of 32-blocks' quant bytes.
    bool t = w.g.rb & GG_TILED;
    device const uint* B = (device const uint*)(t ? kpart(w.g, row, b, 176, 0, 48) : sblk(w.g, row, b, 176));
    device const uint2* q = (device const uint2*)(t ? kpart(w.g, row, b, 176, 48 + 32 * (j / 2), 32) : (device const uchar*)(B + 12 + 8 * (j / 2)));
    half2 dm = as_type<half2>(B[0]);
    float sc, mn;
    scale_min_w(j, B[1], B[2], B[3], sc, mn);
    float d1 = float(dm.x) * sc, m1 = float(dm.y) * mn;
    device const uint2* qh = (device const uint2*)(B + 4);
    uint sh = 4 * (j % 2);
    for (uint i = 0; i < 4; i++) {
        uint2 u = q[i], h = qh[i];
        uint2 v = ((u >> sh) & 0x0F0F0F0Fu) | (((h >> j) & 0x01010101u) << 4);
        out[2 * i] = f4(as_type<uchar4>(v.x)) * d1 - m1;
        out[2 * i + 1] = f4(as_type<uchar4>(v.y)) * d1 - m1;
    }
}
// Q6_K blocks (210 bytes) are 2-byte aligned: read as 16-bit words (same values as bytes; the
// byte reads made Q6_K the slowest of the matrix-unit GEMMs, 15-25% behind these).
struct Q6_KW { GgW g; };
inline void load(Q6_KW w, uint row, uint b, thread float4* out) {
    uint j = b % 8, n = j / 4, jj = j % 4;
    // Tiled: the 16 scales and d (18 bytes), then each half's 64 low and 32 high quant bytes.
    device const uchar *ql8, *qh8, *sc8;
    if (w.g.rb & GG_TILED) {
        sc8 = kpart(w.g, row, b, 210, 0, 18);
        device const uchar* P = kpart(w.g, row, b, 210, 18 + 96 * n, 96);
        ql8 = P;
        qh8 = P + 64;
    } else {
        device const uchar* B = sblk(w.g, row, b, 210);
        ql8 = B + 64 * n;
        qh8 = B + 128 + 32 * n;
        sc8 = B + 192;
    }
    device const ushort* ql = (device const ushort*)(ql8 + 32 * (jj & 1));
    device const ushort* qh = (device const ushort*)qh8;
    uint scs = *(device const ushort*)(sc8 + 8 * n + 2 * jj);
    float d = float(as_type<half>(*(device const ushort*)(sc8 + 16)));
    float dl0 = d * float(as_type<char>(uchar(scs & 255))), dl1 = d * float(as_type<char>(uchar(scs >> 8)));
    uint lsh = 4 * (jj >> 1), hsh = 2 * jj;
    for (uint i = 0; i < 8; i++) {
        uint l = uint(ql[2 * i]) | (uint(ql[2 * i + 1]) << 16);
        uint h = uint(qh[2 * i]) | (uint(qh[2 * i + 1]) << 16);
        uint q = ((l >> lsh) & 0x0F0F0F0Fu) | (((h >> hsh) & 0x03030303u) << 4);
        out[i] = (f4(as_type<uchar4>(q)) - 32.f) * (i < 4 ? dl0 : dl1);
    }
}
struct Q3_KW { GgW g; };
inline void load(Q3_KW w, uint row, uint b, thread float4* out) {
    device const uchar* B = sblk(w.g, row, b, 110);
    uint j = b % 8, n = j / 4, jj = j % 4;
    float d = h2f(B + 108);
    uchar sh = uchar(2 * jj), m = uchar(1 << (4 * n + jj));
    for (uint h = 0; h < 2; h++) {
        uint k = 8 * n + 2 * jj + h;
        uint lo = k < 8 ? (B[96 + k] & 15) : (B[96 + k - 8] >> 4);
        uint hi = (B[104 + k % 4] >> (2 * (k / 4))) & 3;
        float dl = d * float(int(lo | (hi << 4)) - 32);
        for (uint i = 0; i < 4; i++) {
            uint l = 16 * h + 4 * i;
            uchar4 q = (ld4(B + 32 + 32 * n + l) >> sh) & uchar(3);
            uchar4 hm = ld4(B + l) & m;
            out[4 * h + i] = (f4(q) - select(float4(4.f), float4(0.f), hm != uchar4(0))) * dl;
        }
    }
}
struct Q2_KW { GgW g; };
inline void load(Q2_KW w, uint row, uint b, thread float4* out) {
    device const uchar* B = sblk(w.g, row, b, 84);
    uint j = b % 8, n = j / 4, jj = j % 4;
    float d = h2f(B + 80), dmin = h2f(B + 82);
    uchar sh = uchar(2 * jj);
    for (uint h = 0; h < 2; h++) {
        uchar s = B[8 * n + 2 * jj + h];
        float dl = d * float(s & 15), ml = dmin * float(s >> 4);
        for (uint i = 0; i < 4; i++) out[4 * h + i] = f4((ld4(B + 16 + 32 * n + 16 * h + 4 * i) >> sh) & uchar(3)) * dl - ml;
    }
}
struct IQ4_XSW { GgW g; };
inline void load(IQ4_XSW w, uint row, uint b, thread float4* out) {
    device const uchar* B = sblk(w.g, row, b, 136);
    uint j = b % 8;
    int ls = int(((B[4 + j / 2] >> (4 * (j % 2))) & 15) | (((rd16(B + 2) >> (2 * j)) & 3) << 4)) - 32;
    float dl = h2f(B) * float(ls);
    for (uint i = 0; i < 4; i++) {
        uchar4 v = ld4(B + 8 + 16 * j + 4 * i);
        out[i] = kv4(v & uchar(15)) * dl;
        out[i + 4] = kv4(v >> uchar(4)) * dl;
    }
}
struct IQ2_XXSW { GgW g; };
inline void load(IQ2_XXSW w, uint row, uint b, thread float4* out) {
    device const uchar* B = sblk(w.g, row, b, 66);
    uint j = b % 8;
    uint a0 = rd32(B + 2 + 8 * j), a1 = rd32(B + 6 + 8 * j);
    float db = h2f(B) * (0.5f + float(a1 >> 28)) * 0.25f;
    for (uint l = 0; l < 4; l++) grid8(iq2xxs_grid[(a0 >> (8 * l)) & 255], ksign((a1 >> (7 * l)) & 127), db, out + 2 * l);
}
struct IQ2_XSW { GgW g; };
inline void load(IQ2_XSW w, uint row, uint b, thread float4* out) {
    device const uchar* B = sblk(w.g, row, b, 74);
    uint j = b % 8;
    float d = h2f(B);
    uchar s = B[66 + j];
    for (uint l = 0; l < 4; l++) {
        uint q = rd16(B + 2 + 2 * (4 * j + l));
        float db = d * (0.5f + float(l < 2 ? (s & 15) : (s >> 4))) * 0.25f;
        grid8(iq2xs_grid[q & 511], ksign(q >> 9), db, out + 2 * l);
    }
}
struct IQ2_SW { GgW g; };
inline void load(IQ2_SW w, uint row, uint b, thread float4* out) {
    device const uchar* B = sblk(w.g, row, b, 82);
    uint j = b % 8;
    float d = h2f(B);
    uchar s = B[74 + j];
    uint qh = B[66 + j];
    for (uint l = 0; l < 4; l++) {
        float db = d * (0.5f + float(l < 2 ? (s & 15) : (s >> 4))) * 0.25f;
        uint idx = uint(B[2 + 4 * j + l]) | ((qh << (8 - 2 * l)) & 0x300);
        grid8(iq2s_grid[idx], B[34 + 4 * j + l], db, out + 2 * l);
    }
}
struct IQ3_XXSW { GgW g; };
inline void load(IQ3_XXSW w, uint row, uint b, thread float4* out) {
    device const uchar* B = sblk(w.g, row, b, 98);
    uint j = b % 8;
    uint aux = rd32(B + 66 + 4 * j);
    float db = h2f(B) * (0.5f + float(aux >> 28)) * 0.5f;
    for (uint l = 0; l < 4; l++)
        grid4x2(iq3xxs_grid[B[2 + 8 * j + 2 * l]], iq3xxs_grid[B[3 + 8 * j + 2 * l]], ksign((aux >> (7 * l)) & 127), db, out + 2 * l);
}
struct IQ3_SW { GgW g; };
inline void load(IQ3_SW w, uint row, uint b, thread float4* out) {
    device const uchar* B = sblk(w.g, row, b, 110);
    uint j = b % 8;
    float db = h2f(B) * float(1 + 2 * ((B[106 + j / 2] >> (4 * (j % 2))) & 15));
    uint qh = B[66 + j];
    for (uint l = 0; l < 4; l++) {
        uint i1 = uint(B[2 + 8 * j + 2 * l]) | ((qh << (8 - 2 * l)) & 256);
        uint i2 = uint(B[3 + 8 * j + 2 * l]) | ((qh << (7 - 2 * l)) & 256);
        grid4x2(iq3s_grid[i1], iq3s_grid[i2], B[74 + 4 * j + l], db, out + 2 * l);
    }
}
// Ternary grids: values -1, 0, 1 as signed bytes, plus +-1/8.
inline void grid1(ulong g, float dl, float delta, thread float4* o) {
    float v[8];
    for (uint j = 0; j < 8; j++) v[j] = dl * (float(as_type<char>(uchar((g >> (8 * j)) & 255))) + delta);
    o[0] = float4(v[0], v[1], v[2], v[3]); o[1] = float4(v[4], v[5], v[6], v[7]);
}
struct IQ1_SW { GgW g; };
inline void load(IQ1_SW w, uint row, uint b, thread float4* out) {
    device const uchar* B = sblk(w.g, row, b, 50);
    uint j = b % 8;
    uint qh = rd16(B + 34 + 2 * j);
    float dl = h2f(B) * float(2 * ((qh >> 12) & 7) + 1);
    float delta = (qh & 0x8000) ? -0.125f : 0.125f;
    for (uint l = 0; l < 4; l++) grid1(iq1s_grid[uint(B[2 + 4 * j + l]) | (((qh >> (3 * l)) & 7) << 8)], dl, delta, out + 2 * l);
}
struct IQ1_MW { GgW g; };
inline void load(IQ1_MW w, uint row, uint b, thread float4* out) {
    device const uchar* B = sblk(w.g, row, b, 56);
    uint j = b % 8;
    uint s0 = rd16(B + 48), s1 = rd16(B + 50), s2 = rd16(B + 52), s3 = rd16(B + 54);
    float d = float(as_type<half>(ushort((s0 >> 12) | ((s1 >> 8) & 0x00f0) | ((s2 >> 4) & 0x0f00) | (s3 & 0xf000))));
    uint sv = rd16(B + 48 + 2 * (j / 2)) >> (6 * (j % 2));
    for (uint l = 0; l < 4; l++) {
        uint qh = B[32 + 2 * j + l / 2];
        uint idx = uint(B[4 * j + l]) | ((qh << (l % 2 ? 4 : 8)) & 0x700);
        float dl = d * float(2 * ((sv >> (l < 2 ? 0 : 3)) & 7) + 1);
        float delta = (qh & (l % 2 ? 0x80 : 0x08)) ? -0.125f : 0.125f;
        grid1(iq1s_grid[idx], dl, delta, out + 2 * l);
    }
}
// Q4_1 / Q5_0 / Q5_1: blocks of 32 with an f16 min (`MIN`) and/or a fifth bit per element (`HB`).
inline void load_q45(device const uchar* B, bool MIN, bool HB, thread float4* out) {
    float d = h2f(B), m = MIN ? h2f(B + 2) : 0.f;
    uint qh = HB ? rd32(B + (MIN ? 4 : 2)) : 0;
    device const uchar* q = B + 2 + (MIN ? 2 : 0) + (HB ? 4 : 0);
    float off = (HB && !MIN) ? 16.f : 0.f;
    for (uint i = 0; i < 4; i++) {
        uchar4 v = ld4(q + 4 * i);
        float4 lo = f4(v & uchar(15)), hi = f4(v >> uchar(4));
        if (HB) {
            uint e = 4 * i;
            lo += float4(float((qh >> e) & 1), float((qh >> (e + 1)) & 1), float((qh >> (e + 2)) & 1), float((qh >> (e + 3)) & 1)) * 16.f;
            hi += float4(float((qh >> (e + 16)) & 1), float((qh >> (e + 17)) & 1), float((qh >> (e + 18)) & 1), float((qh >> (e + 19)) & 1)) * 16.f;
        }
        out[i] = (lo - off) * d + m;
        out[i + 4] = (hi - off) * d + m;
    }
}
struct Q4_1W { GgW g; };
inline void load(Q4_1W w, uint row, uint b, thread float4* out) { load_q45(gblk(w.g, row, b, 20), true, false, out); }
struct Q5_0W { GgW g; };
inline void load(Q5_0W w, uint row, uint b, thread float4* out) { load_q45(gblk(w.g, row, b, 22), false, true, out); }
// Bits 0..3 of x moved to bit 4 of bytes 0..3.
inline uint spread4(uint x) { return ((x & 1u) | ((x & 2u) << 7) | ((x & 4u) << 14) | ((x & 8u) << 21)) << 4; }
struct Q5_1W { GgW g; };
inline void load(Q5_1W w, uint row, uint b, thread float4* out) {
    device const uint* B = (device const uint*)(gblk(w.g, row, b, 24));
    half2 dm = as_type<half2>(B[0]);
    float d = float(dm.x), m = float(dm.y);
    uint qh = B[1];
    for (uint i = 0; i < 4; i++) {
        uint v = B[2 + i], e = 4 * i;
        float4 lo = f4(as_type<uchar4>((v & 0x0F0F0F0Fu) | spread4((qh >> e) & 15u)));
        float4 hi = f4(as_type<uchar4>(((v >> 4) & 0x0F0F0F0Fu) | spread4((qh >> (e + 16)) & 15u)));
        out[i] = (lo - 0.f) * d + m;
        out[i + 4] = (hi - 0.f) * d + m;
    }
}
// Ternary blocks (Q2_0: 64 values, PrismML's PQ2_0 / PTQ1_0: 128) with one f16 scale. Q2_0 /
// PQ2_0: 2-bit codes c - 1, value j in bits 2 (j % 4) of byte j / 4.
inline void load_t2(device const uchar* q, float d, thread float4* out) {
    for (uint i = 0; i < 8; i++) {
        uint v = q[i];
        out[i] = (float4(float(v & 3u), float((v >> 2) & 3u), float((v >> 4) & 3u), float(v >> 6)) - 1.f) * d;
    }
}
struct Q2_0W { GgW g; };
inline void load(Q2_0W w, uint row, uint b, thread float4* out) {
    device const uchar* B = gblk(w.g, row, b / 2, 18);
    load_t2(B + 2 + 8 * (b % 2), h2f(B), out);
}
struct PQ2_0W { GgW g; };
inline void load(PQ2_0W w, uint row, uint b, thread float4* out) {
    device const uchar* B = gblk(w.g, row, b / 4, 34);
    load_t2(B + 2 + 8 * (b % 4), h2f(B), out);
}
// Base-3 digit n (3^n in p) of byte q minus one, as ggml reads TQ1_0 / PTQ1_0 bytes.
inline float trit(uint q, uint p) { return float((((q * p) & 255u) * 3u) >> 8) - 1.f; }
// PTQ1_0: 24 bytes of five trits in runs of 16 and 8 (digit n of byte m of a run of c bytes is
// value n c + m of the run's 5 c), 2 bytes of four trits (value 120 + 2 n + h), then the scale.
inline float ptq1_at(device const uchar* B, uint e) {
    const uint p3[5] = {1u, 3u, 9u, 27u, 81u};
    if (e < 80) return trit(B[e % 16], p3[e / 16]);
    if (e < 120) return trit(B[16 + (e - 80) % 8], p3[(e - 80) / 8]);
    return trit(B[24 + (e - 120) % 2], p3[(e - 120) / 2]);
}
struct PTQ1_0W { GgW g; };
inline void load(PTQ1_0W w, uint row, uint b, thread float4* out) {
    device const uchar* B = gblk(w.g, row, b / 4, 28);
    float d = h2f(B + 26);
    uint e = 32 * (b % 4);
    for (uint i = 0; i < 8; i++)
        out[i] = float4(ptq1_at(B, e + 4 * i), ptq1_at(B, e + 4 * i + 1), ptq1_at(B, e + 4 * i + 2), ptq1_at(B, e + 4 * i + 3)) * d;
}
struct F16W { GgW g; };
inline void load(F16W w, uint row, uint b, thread float4* out) {
    device const packed_half4* p = (device const packed_half4*)(gblk(w.g, row, b, 64));
    for (uint i = 0; i < 8; i++) out[i] = float4(half4(p[i]));
}
struct F32W2 { GgW g; };
inline void load(F32W2 w, uint row, uint b, thread float4* out) {
    device const packed_float4* p = (device const packed_float4*)(gblk(w.g, row, b, 128));
    for (uint i = 0; i < 8; i++) out[i] = float4(p[i]);
}
// FP8 E4M3 with one f32 scale per 128 x 128 block (DeepSeek / Mistral `weight_scale_inv`): rows
// of n_in bytes at w_off, the scale grid (n_in / 128 rounded up per row block) at s_off floats.
// Shifting the sign and the 7 exponent / mantissa bits into a half's places gives the value
// times 2^-8 exactly (subnormals included), folded into the scale.
struct F8W { device const uchar* q; device const float* s; uint n_in; uint ncb; };
#define MK_F8 F8W{ (device const uchar*)qb + p.w_off, (device const float*)sb + p.s_off, p.n_in, (p.n_in + 127) / 128 }
inline float4 e4m3x4(uchar4 v) { ushort4 u = ushort4(v); return float4(as_type<half4>(((u & ushort(0x80)) << 8) | ((u & ushort(0x7f)) << 7))); }
inline void load(F8W w, uint row, uint b, thread float4* out) {
    float d = w.s[(row / 128) * w.ncb + b / 4] * 256.f;
    device const uchar* B = w.q + (ulong)row * w.n_in + b * 32;
    for (uint i = 0; i < 8; i++) out[i] = e4m3x4(ld4(B + 4 * i)) * d;
}
GEMM(gemm_F8_E4M3, F8W, MK_F8)
// NVFP4 in ggml's blocks of 64 (four unsigned E4M3 scales, halved, then four runs of 8 bytes
// holding elements j and j + 8 of a sub-block of 16; the codes through MXFP4's doubled table),
// times the tensor-wide factor (the matrix's one block scale, at s_off floats): chunk b of 32
// values is half of block b / 2. Rows of n_in / 64 blocks of 36 bytes.
// Four E2M1 codes (the low nibbles of v's bytes) as floats over 2^-14: the code's exponent and
// mantissa bits are a half's (subnormal for exponent 0), its sign the half's sign (no lookups).
inline float4 e2m1x4(uint v) {
    uint lo = ((v & 0x00070007u) << 9) | ((v & 0x00080008u) << 12);
    uint hi = (((v >> 8) & 0x00070007u) << 9) | (((v >> 8) & 0x00080008u) << 12);
    half2 a = as_type<half2>(lo), b = as_type<half2>(hi);
    return float4(float(a.x), float(b.x), float(a.y), float(b.y));
}
struct NvfW { device const uchar* q; float f; uint rb; };
#define MK_NV NvfW{ (device const uchar*)qb + p.w_off, ((device const float*)sb)[p.s_off], (p.n_in / 64) * 36 }
inline void load(NvfW w, uint row, uint b, thread float4* out) {
    device const uchar* B = w.q + (ulong)row * w.rb + (b / 2) * 36;
    for (uint s2 = 0; s2 < 2; s2++) {
        uint s = 2 * (b % 2) + s2;
        // (The scale, halved, times the codes' 2^14, the table's doubling and the factor.)
        float d = float(as_type<half>(ushort(ushort(B[s] & 0x7f) << 7))) * (128.f * 32768.f) * w.f;
        // Byte i's low nibble is element i, its high one element i + 8 of the sub-block.
        uint v0 = rd32(B + 4 + 8 * s), v1 = rd32(B + 8 + 8 * s);
        out[4 * s2 + 0] = e2m1x4(v0) * d;
        out[4 * s2 + 1] = e2m1x4(v1) * d;
        out[4 * s2 + 2] = e2m1x4(v0 >> 4) * d;
        out[4 * s2 + 3] = e2m1x4(v1 >> 4) * d;
    }
}
GEMM(gemm_NVFP4, NvfW, MK_NV)
#define GG(T) GEMM(gemm_##T, T##W, MK_GG(T##W))
GEMM(gemm_F32, F32W2, MK_GG(F32W2)) GG(F16) GG(Q4_1) GG(Q5_0) GG(Q5_1)
GG(Q4_0) GG(IQ4_NL) GG(Q8_0) GG(Q4_K) GG(Q5_K) GG(Q6_K) GG(Q3_K) GG(Q2_K) GG(IQ4_XS)
GG(IQ2_XXS) GG(IQ2_XS) GG(IQ2_S) GG(IQ3_XXS) GG(IQ3_S) GG(IQ1_S) GG(IQ1_M) GG(MXFP4)
GG(Q2_0) GG(PQ2_0) GG(PTQ1_0)
"#;

/// GEMMs on the matrix units of M5 / A19 GPU cores (Metal 4 tensor operations, `matmul2d` from
/// MetalPerformancePrimitives), appended to the library when the device has them. Operands are
/// half precision with f32 accumulation: each input row is scaled by a power of two so its
/// largest value lands in [2^13, 2^14) before the conversion (no overflow, no subnormals), and the
/// outputs are scaled back. Quantized weights are dequantized per 64-wide K step into threadgroup
/// memory as half (64 rows x 64 k); bf16 weights are read straight from device memory as bfloat.
/// A threadgroup (4 SIMD-groups) computes 64 outputs x 128 tokens (32 tokens in the grouped
/// kernels, whose segments are short). The tensor bounds would clip partial token tiles, but
/// those take a ~1.6x slower path, so the inputs are padded to whole tiles instead.
/// `mmag_*` run many weight matrices in one dispatch (MoE experts): each threadgroup row looks up
/// its tile (expert, segment start and length, offset) and the expert's weight addresses.
const MMA_SHADER_SRC: &str = r#"
#if MMA_UNITS
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
#endif
// The first four fields as in P (the MK_* macros read them), then the row count; dense
// kernels: _p0 the output row stride (0: n_out), _p1 the first token row of the grid, _p2 nonzero
// to add the products to the outputs.
struct MP { uint n_out; uint n_in; uint w_off; uint s_off; uint rows; uint _p0; uint _p1; uint _p2; };
#define MKB 64
#if MMA_UNITS
typedef tensor<device half, dextents<int32_t, 2>, tensor_inline> XT;
typedef tensor<device float, dextents<int32_t, 2>, tensor_inline> YT;

template <typename CT, typename DT>
inline void scale_store(thread CT& c, device const float* rs, uint t0, DT d) {
#pragma unroll
    for (uint16_t i = 0; i < c.get_capacity(); ++i)
        if (c.is_valid_element(i)) { auto idx = c.get_multidimensional_index(i); c[i] *= rs[t0 + idx[1]]; }
    c.store(d);
}
// The same into rows ldy floats apart (prefill plans: a view of a wider activation row), or
// added to them (`add`: the residual adds of prefill plans).
template <typename CT>
inline void scale_store_ld(thread CT& c, device const float* rs, uint t0, device float* y, uint ldy, uint o0, bool add = false) {
#pragma unroll
    for (uint16_t i = 0; i < c.get_capacity(); ++i)
        if (c.is_valid_element(i)) {
            auto idx = c.get_multidimensional_index(i);
            uint r = t0 + idx[1];
            device float* o = y + (ulong)r * ldy + o0 + idx[0];
            float v = c[i] * rs[r];
            *o = add ? *o + v : v;
        }
}
template <typename CT>
inline void zero(thread CT& c) {
#pragma unroll
    for (uint16_t i = 0; i < c.get_capacity(); ++i) if (c.is_valid_element(i)) c[i] = 0;
}
#endif
// A 32-block of weights as half: dequantized in f32, then converted; Q8_0 and Q4_0 in half
// arithmetic from 16-bit loads instead (the same values: q d is exact in f32 and rounded once
// either way), which cuts the dequantization's share of the GEMMs.
template <typename W>
inline void loadh(W w, uint row, uint b, thread half4* out) {
    float4 t[8];
    load(w, row, b, t);
    for (uint i = 0; i < 8; i++) out[i] = half4(t[i]);
}
inline void loadh(Q8_0W w, uint row, uint b, thread half4* out) {
    device const ushort* B = (device const ushort*)(gblk(w.g, row, b, 34));
    half d = as_type<half>(B[0]);
    for (uint i = 0; i < 8; i++) out[i] = half4(as_type<char4>(ushort2(B[1 + 2 * i], B[2 + 2 * i]))) * d;
}
inline void loadh(Q4_0W w, uint row, uint b, thread half4* out) {
    device const ushort* B = (device const ushort*)(gblk(w.g, row, b, 18));
    half d = as_type<half>(B[0]);
    for (uint i = 0; i < 4; i++) {
        uchar4 v = as_type<uchar4>(ushort2(B[1 + 2 * i], B[2 + 2 * i]));
        out[i] = (half4(v & uchar(15)) - 8.h) * d;
        out[i + 4] = (half4(v >> uchar(4)) - 8.h) * d;
    }
}
#if MMA_UNITS
#define MMA_WS threadgroup half Ws[64 * MKB];
template <typename W, int TM>
inline void mma_tile(W w, device const half* xh, device const float* rs, device float* y, uint n_out, uint n_in, uint rows,
                     uint o0, uint t0, uint tid, threadgroup half* Ws, uint ldy = 0, uint add = 0) {
    XT X((device half*)xh, dextents<int32_t, 2>(n_in, rows));
    YT Y(y, dextents<int32_t, 2>(n_out, rows));
    constexpr auto desc = matmul2d_descriptor(TM, 64, MKB, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;
    typedef tensor<threadgroup half, extents<int32_t, MKB, 64>, tensor_inline> WT;
    WT Wt(Ws, extents<int32_t, MKB, 64>());
    auto a0 = X.template slice<MKB, TM>(0, t0);
    auto c = op.template get_destination_cooperative_tensor<decltype(a0), WT, float>();
    zero(c);
    uint row = tid / 2, blk = tid % 2;   // each thread dequantizes one 32-block of one row
    threadgroup half4* wd = (threadgroup half4*)(Ws + row * MKB + blk * 32);
    for (uint k0 = 0; k0 < n_in; k0 += MKB) {
        half4 wv[8];
        loadh(w, o0 + row, k0 / 32 + blk, wv);
        for (uint i = 0; i < 8; i++) wd[i] = wv[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto a = X.template slice<MKB, TM>(k0, t0);
        op.run(a, Wt, c);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (ldy != 0 || add != 0) scale_store_ld(c, rs, t0, y, ldy != 0 ? ldy : n_out, o0, add != 0);
    else scale_store(c, rs, t0, Y.slice(o0, t0));
}
template <int TM, int TN>
inline void mma_bf_tile(device const bfloat* w, device const half* xh, device const float* rs, device float* y, uint n_out, uint n_in, uint rows,
                        uint o0, uint t0, uint ldy = 0, uint add = 0) {
    XT X((device half*)xh, dextents<int32_t, 2>(n_in, rows));
    YT Y(y, dextents<int32_t, 2>(n_out, rows));
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Wt((device bfloat*)w, dextents<int32_t, 2>(n_in, n_out));
    constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto a = X.slice(0, t0);
    auto b = Wt.slice(0, o0);
    auto c = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
    zero(c);
    op.run(a, b, c);
    if (ldy != 0 || add != 0) scale_store_ld(c, rs, t0, y, ldy != 0 ? ldy : n_out, o0, add != 0);
    else scale_store(c, rs, t0, Y.slice(o0, t0));
}
#else
// GPUs without matrix units (M1-M4): the same kernels on 8x8 SIMD-group matrices, half operands
// and f32 accumulators, as on the matrix units. 4 SIMD-groups per threadgroup: 2 x 2 (32 outputs
// x 32 tokens each) for 64-token tiles, 4 along the outputs (16 outputs each) for 32- and 16-token
// ones, whose products are then bound by the weights' loads from threadgroup memory rather than
// by the tokens' (M3 Ultra: 25-40% faster than 2 x 2 there). Per 64-wide K step the 64 weight rows
// are dequantized to half in threadgroup memory and the token rows read straight from device
// memory with transposed 8x8 loads. Each lane scales the two elements of an 8x8 result it holds
// (one output row, two tokens: the lane layout of `metal_pfflash.metal`) by their token rows'
// inverse scales, adds the outputs to them for an add, and the results are stored transposed.
#define SG_LDW 72
#define MMA_WS threadgroup half Ws[64 * SG_LDW];
template <typename W, int TM>
inline void sg_tile(W w, device const half* xh, device const float* rs, device float* y, uint n_out, uint n_in, uint rows,
                    uint o0, uint t0, uint tid, threadgroup half* Ws, uint ldy, uint add) {
    // SGO SIMD-groups along the outputs (NA 8-row blocks each), 4 / SGO along the tokens (NB).
    constexpr int SGO = TM >= 64 ? 2 : 4, NA = 8 / SGO, NB = TM / (32 / SGO);
    uint sg = tid / 32;
    uint so = (sg % SGO) * (8 * NA), st = (sg / SGO) * (8 * NB);
    simdgroup_float8x8 c[NA * NB];
#pragma unroll
    for (int i = 0; i < NA * NB; i++) c[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    uint row = tid / 2, blk = tid % 2;   // each thread dequantizes one 32-block of one row
    threadgroup half4* wd = (threadgroup half4*)(Ws + row * SG_LDW + blk * 32);
    device const half* xs = xh + (ulong)(t0 + st) * n_in;
    for (uint k0 = 0; k0 < n_in; k0 += MKB) {
        half4 wv[8];
        loadh(w, o0 + row, k0 / 32 + blk, wv);
        for (uint i = 0; i < 8; i++) wd[i] = wv[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
        for (uint k = 0; k < MKB; k += 8) {
            simdgroup_half8x8 a[NA], b[NB];
#pragma unroll
            for (int i = 0; i < NA; i++) simdgroup_load(a[i], Ws + (so + 8 * i) * SG_LDW + k, SG_LDW);
#pragma unroll
            for (int j = 0; j < NB; j++) simdgroup_load(b[j], xs + (ulong)(8 * j) * n_in + k0 + k, n_in, ulong2(0, 0), true);
#pragma unroll
            for (int i = 0; i < NA; i++)
#pragma unroll
                for (int j = 0; j < NB; j++) simdgroup_multiply_accumulate(c[i * NB + j], a[i], b[j], c[i * NB + j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // This lane's elements of an 8x8 result: column (token) fc and fc + 1 of one row.
    uint lane = tid % 32;
    uint fc = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
    uint ld = ldy != 0 ? ldy : n_out;
#pragma unroll
    for (int j = 0; j < NB; j++) {
        uint tt = t0 + st + 8 * j;
        if (tt >= rows) break;   // rows are whole 8-row blocks
        float2 sc = float2(rs[tt + fc], rs[tt + fc + 1]);
#pragma unroll
        for (int i = 0; i < NA; i++) {
            device float* o = y + (ulong)tt * ld + o0 + so + 8 * i;
            thread float2& e = reinterpret_cast<thread float2&>(c[i * NB + j].thread_elements());
            e *= sc;
            if (add != 0) {
                simdgroup_float8x8 r0;
                simdgroup_load(r0, o, ld, ulong2(0, 0), true);
                e += reinterpret_cast<thread float2&>(r0.thread_elements());
            }
            simdgroup_store(c[i * NB + j], o, ld, ulong2(0, 0), true);
        }
    }
}
// 128-token tiles as two of 64 (32 accumulators per SIMD-group would spill).
template <typename W, int TM>
inline void mma_tile(W w, device const half* xh, device const float* rs, device float* y, uint n_out, uint n_in, uint rows,
                     uint o0, uint t0, uint tid, threadgroup half* Ws, uint ldy = 0, uint add = 0) {
    constexpr int T = TM > 64 ? 64 : TM;
    for (uint h = 0; h < TM / T; h++) sg_tile<W, T>(w, xh, rs, y, n_out, n_in, rows, o0, t0 + h * T, tid, Ws, ldy, add);
}
#endif
// Dense: 128-token tiles, 32-token tiles (`_s`) for short inputs and 64-token ones (`_m`) for
// narrow outputs (prefill plans). (MAKE expands to text with commas, so it is only ever used
// directly in a body, never forwarded to another macro.)
#define MMA_SIG(NAME) \
kernel void NAME(device const void* qb [[buffer(0)]], device const void* sb [[buffer(1)]], device const half* xh [[buffer(2)]], \
                 device float* y [[buffer(3)]], constant MP& p [[buffer(4)]], device const float* rs [[buffer(5)]], \
                 uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]])
#define MMAD(NAME, WT, MAKE) \
MMA_SIG(NAME) { MMA_WS \
    WT w = MAKE; mma_tile<WT, 128>(w, xh, rs, y, p.n_out, p.n_in, p.rows, tg.x * 64, p._p1 + tg.y * 128, tid, Ws, p._p0, p._p2); } \
MMA_SIG(NAME##_s) { MMA_WS \
    WT w = MAKE; mma_tile<WT, 32>(w, xh, rs, y, p.n_out, p.n_in, p.rows, tg.x * 64, p._p1 + tg.y * 32, tid, Ws, p._p0, p._p2); } \
MMA_SIG(NAME##_m) { MMA_WS \
    WT w = MAKE; mma_tile<WT, 64>(w, xh, rs, y, p.n_out, p.n_in, p.rows, tg.x * 64, p._p1 + tg.y * 64, tid, Ws, p._p0, p._p2); }
// Grouped: per-expert weight addresses, and one tile per threadgroup row. Each format comes with
// tiles of GTM rows (NAME), 128 rows (NAME_l: the bulk of an expert's rows when it has many, the
// weights dequantized once for 4x the rows) and 16 rows (NAME_s: an odd remainder, half the
// padding); a layer's tiles are dispatched one grid per size (see `group_tiles`).
struct EP { device const uchar* q; device const uchar* s; };
struct GT { uint e; uint r0; uint rows; uint t0; };
#define MMAG_SIG(NAME) \
kernel void NAME(device const EP* ep [[buffer(0)]], device const GT* tiles [[buffer(1)]], device const half* xh [[buffer(2)]], \
                 device float* y [[buffer(3)]], constant MP& p [[buffer(4)]], device const float* rs [[buffer(5)]], \
                 uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]])
#define MMAG_PRE MMA_WS \
    GT g = tiles[tg.y]; device const void* qb = ep[g.e].q; device const void* sb = ep[g.e].s;
#define MMAG_ARGS w, xh + (ulong)g.r0 * p.n_in, rs + g.r0, y + (ulong)g.r0 * p.n_out, p.n_out, p.n_in, g.rows, tg.x * 64, g.t0, tid, Ws
#define MMAG(NAME, WT, MAKE) \
MMAG_SIG(NAME) { MMAG_PRE WT w = MAKE; mma_tile<WT, GTM>(MMAG_ARGS); } \
MMAG_SIG(NAME##_l) { MMAG_PRE WT w = MAKE; mma_tile<WT, 128>(MMAG_ARGS); } \
MMAG_SIG(NAME##_s) { MMAG_PRE WT w = MAKE; mma_tile<WT, 16>(MMAG_ARGS); }
MMAD(mma_q8, Q8W, MK_Q8) MMAD(mma_q4, Q4W, MK_Q4)
MMAG(mmag_q8, Q8W, MK_Q8) MMAG(mmag_q4, Q4W, MK_Q4)
MMAD(mma_F8_E4M3, F8W, MK_F8) MMAG(mmag_F8_E4M3, F8W, MK_F8)
MMAD(mma_NVFP4, NvfW, MK_NV) MMAG(mmag_NVFP4, NvfW, MK_NV)
#define MMA_Q4A(NAME, TM) \
kernel void NAME(device const void* qb [[buffer(0)]], device const void* sb [[buffer(1)]], device const half* xh [[buffer(2)]], \
                 device float* y [[buffer(3)]], constant MP& p [[buffer(4)]], device const float* rs [[buffer(5)]], \
                 device const ushort* ob [[buffer(6)]], constant OP& op [[buffer(7)]], \
                 uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
    MMA_WS \
    Q4A w = { MK_Q4, ob + op.o_off, op.gpr, op.gk }; \
    mma_tile<Q4A, TM>(w, xh, rs, y, p.n_out, p.n_in, p.rows, tg.x * 64, p._p1 + tg.y * TM, tid, Ws, p._p0, p._p2); }
MMA_Q4A(mma_q4a, 128) MMA_Q4A(mma_q4a_s, 32) MMA_Q4A(mma_q4a_m, 64)
#if MMA_UNITS
#define MMA_BF(NAME, TN, TM) \
kernel void NAME(device const ushort* qb [[buffer(0)]], device const half* xh [[buffer(2)]], device float* y [[buffer(3)]], \
                 constant MP& p [[buffer(4)]], device const float* rs [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]]) { \
    mma_bf_tile<TM, TN>((device const bfloat*)(qb + p.w_off), xh, rs, y, p.n_out, p.n_in, p.rows, tg.x * TN, p._p1 + tg.y * TM, p._p0, p._p2); }
MMA_BF(mma_bf64, 64, 128) MMA_BF(mma_bf128, 128, 128) MMA_BF(mma_bf64_s, 64, 32) MMA_BF(mma_bf128_s, 128, 32)
MMA_BF(mma_bf64_m, 64, 64) MMA_BF(mma_bf128_m, 128, 64)
#define MMAG_BF(NAME, TM) \
kernel void NAME(device const EP* ep [[buffer(0)]], device const GT* tiles [[buffer(1)]], device const half* xh [[buffer(2)]], \
                 device float* y [[buffer(3)]], constant MP& p [[buffer(4)]], device const float* rs [[buffer(5)]], \
                 uint2 tg [[threadgroup_position_in_grid]]) { \
    GT g = tiles[tg.y]; \
    mma_bf_tile<TM, 64>((device const bfloat*)ep[g.e].q, xh + (ulong)g.r0 * p.n_in, rs + g.r0, y + (ulong)g.r0 * p.n_out, \
                        p.n_out, p.n_in, g.rows, tg.x * 64, g.t0); }
MMAG_BF(mmag_bf, GTM) MMAG_BF(mmag_bf_l, 128) MMAG_BF(mmag_bf_s, 16)
#else
// bf16 weights through threadgroup memory as the quantized formats (converted to half), 64
// outputs at a time.
#define MMA_BF(NAME, TN, TM) \
kernel void NAME(device const ushort* qb [[buffer(0)]], device const half* xh [[buffer(2)]], device float* y [[buffer(3)]], \
                 constant MP& p [[buffer(4)]], device const float* rs [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]], \
                 uint tid [[thread_index_in_threadgroup]]) { \
    MMA_WS BfW w = { qb + p.w_off, p.n_in }; \
    for (uint h = 0; h < TN / 64; h++) \
        mma_tile<BfW, TM>(w, xh, rs, y, p.n_out, p.n_in, p.rows, tg.x * TN + h * 64, p._p1 + tg.y * TM, tid, Ws, p._p0, p._p2); }
MMA_BF(mma_bf64, 64, 128) MMA_BF(mma_bf128, 128, 128) MMA_BF(mma_bf64_s, 64, 32) MMA_BF(mma_bf128_s, 128, 32)
MMA_BF(mma_bf64_m, 64, 64) MMA_BF(mma_bf128_m, 128, 64)
#define MMAG_BF(NAME, TM) \
kernel void NAME(device const EP* ep [[buffer(0)]], device const GT* tiles [[buffer(1)]], device const half* xh [[buffer(2)]], \
                 device float* y [[buffer(3)]], constant MP& p [[buffer(4)]], device const float* rs [[buffer(5)]], \
                 uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
    MMA_WS GT g = tiles[tg.y]; BfW w = { (device const ushort*)ep[g.e].q, p.n_in }; \
    mma_tile<BfW, TM>(w, xh + (ulong)g.r0 * p.n_in, rs + g.r0, y + (ulong)g.r0 * p.n_out, p.n_out, p.n_in, g.rows, tg.x * 64, g.t0, tid, Ws); }
MMAG_BF(mmag_bf, GTM) MMAG_BF(mmag_bf_l, 128) MMAG_BF(mmag_bf_s, 16)
#endif
#define GGM(T) MMAD(mma_##T, T##W, MK_GG(T##W)) MMAG(mmag_##T, T##W, MK_GG(T##W))
GGM(F16) GGM(Q4_1) GGM(Q5_0) GGM(Q5_1)
GGM(Q4_0) GGM(IQ4_NL) GGM(Q8_0) GGM(Q4_K) GGM(Q5_K) GGM(Q6_K) GGM(Q3_K) GGM(Q2_K) GGM(IQ4_XS)
GGM(IQ2_XXS) GGM(IQ2_XS) GGM(IQ2_S) GGM(IQ3_XXS) GGM(IQ3_S) GGM(IQ1_S) GGM(IQ1_M) GGM(MXFP4)
GGM(Q2_0) GGM(PQ2_0) GGM(PTQ1_0)
#if MMA_UNITS
// Prefill plans: a gate / up pair and its SwiGLU in one GEMM (`mmap_*`). A threadgroup's 64
// weight rows are 32 gate rows and the 32 matching up rows, so the products run as one dense GEMM
// tile; the up half of the results goes through threadgroup memory to the threads holding the
// gate half, which store silu(g) * u for 32 outputs. g and u are rounded to f32 with the token
// rows' scales first, as the two GEMMs' stores are, and silu as `dec_glu_t`'s: the same values
// bit for bit, without the gate and up rows' f32 stores and the GLU pass that read them. The
// plans pass gelu = 0 only (GeGLU's tanh in these stores rounded otherwise), but the branch stays:
// it keeps the expression in `pf_glu_half`'s shape, which fast math otherwise rearranges (other
// bits). pq: the up weight's w_off and s_off, gelu.
template <typename W, int TM>
inline void mma_pair_tile(W wg, W wu, device const half* xh, device const float* rs, device float* y, uint n_in, uint rows,
                          uint ob, uint t0, uint tid, threadgroup half* Ws, threadgroup float* U, uint ldy, uint gelu) {
    XT X((device half*)xh, dextents<int32_t, 2>(n_in, rows));
    constexpr auto desc = matmul2d_descriptor(TM, 64, MKB, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;
    typedef tensor<threadgroup half, extents<int32_t, MKB, 64>, tensor_inline> WT;
    WT Wt(Ws, extents<int32_t, MKB, 64>());
    auto a0 = X.template slice<MKB, TM>(0, t0);
    auto c = op.template get_destination_cooperative_tensor<decltype(a0), WT, float>();
    zero(c);
    uint row = tid / 2, blk = tid % 2;   // each thread dequantizes one 32-block of one row
    W w = row < 32 ? wg : wu;
    uint wrow = ob + row % 32;
    threadgroup half4* wd = (threadgroup half4*)(Ws + row * MKB + blk * 32);
    for (uint k0 = 0; k0 < n_in; k0 += MKB) {
        half4 wv[8];
        loadh(w, wrow, k0 / 32 + blk, wv);
        for (uint i = 0; i < 8; i++) wd[i] = wv[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto a = X.template slice<MKB, TM>(k0, t0);
        op.run(a, Wt, c);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
#pragma unroll
    for (uint16_t i = 0; i < c.get_capacity(); ++i)
        if (c.is_valid_element(i)) {
            auto idx = c.get_multidimensional_index(i);
            if (idx[0] >= 32) U[idx[1] * 33 + idx[0] - 32] = c[i] * rs[t0 + idx[1]];
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint16_t i = 0; i < c.get_capacity(); ++i)
        if (c.is_valid_element(i)) {
            auto idx = c.get_multidimensional_index(i);
            if (idx[0] < 32) {
                uint r = t0 + idx[1];
                float a = c[i] * rs[r];
                float act;
                if (gelu != 0) {
                    float tt = 0.7978845608f * (a + 0.044715f * a * a * a);
                    act = 0.5f * a * (1.0f + precise::tanh(tt));
                } else {
                    act = a / (1.0f + exp(-a));
                }
                y[(ulong)r * ldy + ob + idx[0]] = act * U[idx[1] * 33 + idx[0]];
            }
        }
}
#define MMAP_SIG(NAME) \
kernel void NAME(device const void* qb [[buffer(0)]], device const void* sb [[buffer(1)]], device const half* xh [[buffer(2)]], \
                 device float* y [[buffer(3)]], constant MP& p [[buffer(4)]], device const float* rs [[buffer(5)]], \
                 device const void* qb2 [[buffer(6)]], device const void* sb2 [[buffer(7)]], constant uint4& pq [[buffer(8)]], \
                 uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]])
#define MMAP_K(NAME, WT, TM, MK) \
MMAP_SIG(NAME) { threadgroup half Ws[64 * MKB]; threadgroup float U[TM * 33]; \
    MP p2 = p; p2.w_off = pq.x; p2.s_off = pq.y; \
    mma_pair_tile<WT, TM>(MK(qb, sb, p), MK(qb2, sb2, p2), xh, rs, y, p.n_in, p.rows, tg.x * 32, p._p1 + tg.y * TM, tid, Ws, U, p._p0, pq.z); }
// (MAKE is used directly in the helper's body: forwarded to another macro, its commas would
// split it.)
#define MMAP(NAME, WT, MAKE) \
inline WT NAME##_w(device const void* qb, device const void* sb, MP p) { return MAKE; } \
MMAP_K(NAME, WT, 128, NAME##_w) MMAP_K(NAME##_s, WT, 32, NAME##_w) MMAP_K(NAME##_m, WT, 64, NAME##_w)
MMAP(mmap_q8, Q8W, MK_Q8) MMAP(mmap_q4, Q4W, MK_Q4)
#define GGP(T) MMAP(mmap_##T, T##W, MK_GG(T##W))
GGP(Q4_0) GGP(Q4_1) GGP(Q5_0) GGP(Q5_1) GGP(Q8_0) GGP(IQ4_NL) GGP(IQ4_XS) GGP(MXFP4)
GGP(Q2_K) GGP(Q3_K) GGP(Q4_K) GGP(Q5_K) GGP(Q6_K)
// Prefill plans' routed experts with gate and up in one tensor (gate rows, then up rows): the
// grouped GEMM tile of `mmag_*` with 32 gate rows and their 32 up rows (`mmagp_*`, as `mmap_*`),
// storing act(g) * u for 32 outputs into rows of `p._p0` (the intermediate size). g and u are
// rounded to f32 with the rows' scales first, as the gate / up GEMM's stores are, and the
// activation is `act_mul4`'s, so the rows hold `silu_half`'s values before its half conversion;
// the gate / up rows are never stored. p._p1: the up rows' byte offset in the expert's tensor,
// p.rows: the up rows' offset in its scales (Q8 / Q4; the GGUF kernels read rows of p.s_off
// bytes, and the grouped kernels take their rows from the tiles), p._p2: GELU (else SiLU).
template <typename W, int TM>
inline void mmag_pair_tile(W wg, W wu, device const half* xh, device const float* rs, device float* y, uint n_in, uint rows,
                           uint ob, uint t0, uint tid, threadgroup half* Ws, threadgroup float* U, uint ldy, uint gelu) {
    XT X((device half*)xh, dextents<int32_t, 2>(n_in, rows));
    constexpr auto desc = matmul2d_descriptor(TM, 64, MKB, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;
    typedef tensor<threadgroup half, extents<int32_t, MKB, 64>, tensor_inline> WT;
    WT Wt(Ws, extents<int32_t, MKB, 64>());
    auto a0 = X.template slice<MKB, TM>(0, t0);
    auto c = op.template get_destination_cooperative_tensor<decltype(a0), WT, float>();
    zero(c);
    uint row = tid / 2, blk = tid % 2;
    W w = row < 32 ? wg : wu;
    uint wrow = ob + row % 32;
    threadgroup half4* wd = (threadgroup half4*)(Ws + row * MKB + blk * 32);
    for (uint k0 = 0; k0 < n_in; k0 += MKB) {
        half4 wv[8];
        loadh(w, wrow, k0 / 32 + blk, wv);
        for (uint i = 0; i < 8; i++) wd[i] = wv[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto a = X.template slice<MKB, TM>(k0, t0);
        op.run(a, Wt, c);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
#pragma unroll
    for (uint16_t i = 0; i < c.get_capacity(); ++i)
        if (c.is_valid_element(i)) {
            auto idx = c.get_multidimensional_index(i);
            if (idx[0] >= 32) U[idx[1] * 33 + idx[0] - 32] = c[i] * rs[t0 + idx[1]];
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint16_t i = 0; i < c.get_capacity(); ++i)
        if (c.is_valid_element(i)) {
            auto idx = c.get_multidimensional_index(i);
            if (idx[0] < 32) {
                uint r = t0 + idx[1];
                float g = c[i] * rs[r], u = U[idx[1] * 33 + idx[0]];
                y[(ulong)r * ldy + ob + idx[0]] = gelu != 0 ? g / (1.0f + exp(-1.5957691f * (g + 0.044715f * g * g * g))) * u
                                                             : g / (1.0f + exp(-g)) * u;
            }
        }
}
#define MMAGP_K(NAME, WT, TM, MK) \
MMAG_SIG(NAME) { threadgroup half Ws[64 * MKB]; threadgroup float U[TM * 33]; \
    GT g = tiles[tg.y]; device const void* qb = ep[g.e].q; device const void* sb = ep[g.e].s; \
    MP p2 = p; p2.w_off = p._p1; p2.s_off = p.s_off + p.rows; \
    mmag_pair_tile<WT, TM>(MK(qb, sb, p), MK(qb, sb, p2), xh + (ulong)g.r0 * p.n_in, rs + g.r0, y + (ulong)g.r0 * p._p0, p.n_in, \
                           g.rows, tg.x * 32, g.t0, tid, Ws, U, p._p0, p._p2); }
// The same with gate and up in tensors of their own (`mmagq_*`): the up experts' addresses in
// ep2, their row bytes (GGUF) in p._p1.
#define MMAGQ_K(NAME, WT, TM, MK) \
kernel void NAME(device const EP* ep [[buffer(0)]], device const GT* tiles [[buffer(1)]], device const half* xh [[buffer(2)]], \
                 device float* y [[buffer(3)]], constant MP& p [[buffer(4)]], device const float* rs [[buffer(5)]], \
                 device const EP* ep2 [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup half Ws[64 * MKB]; threadgroup float U[TM * 33]; \
    GT g = tiles[tg.y]; MP p2 = p; p2.s_off = p._p1; \
    mmag_pair_tile<WT, TM>(MK(ep[g.e].q, ep[g.e].s, p), MK(ep2[g.e].q, ep2[g.e].s, p2), xh + (ulong)g.r0 * p.n_in, rs + g.r0, \
                           y + (ulong)g.r0 * p._p0, p.n_in, g.rows, tg.x * 32, g.t0, tid, Ws, U, p._p0, p._p2); }
#define MMAGP(NAME, NAMEQ, WT, MAKE) \
inline WT NAME##_w(device const void* qb, device const void* sb, MP p) { return MAKE; } \
MMAGP_K(NAME, WT, GTM, NAME##_w) MMAGP_K(NAME##_l, WT, 128, NAME##_w) MMAGP_K(NAME##_s, WT, 16, NAME##_w) \
MMAGQ_K(NAMEQ, WT, GTM, NAME##_w) MMAGQ_K(NAMEQ##_l, WT, 128, NAME##_w) MMAGQ_K(NAMEQ##_s, WT, 16, NAME##_w)
MMAGP(mmagp_q8, mmagq_q8, Q8W, MK_Q8) MMAGP(mmagp_q4, mmagq_q4, Q4W, MK_Q4)
#define GGXP(T) MMAGP(mmagp_##T, mmagq_##T, T##W, MK_GG(T##W))
GGXP(Q4_0) GGXP(Q4_1) GGXP(Q5_0) GGXP(Q5_1) GGXP(Q8_0) GGXP(IQ4_NL) GGXP(IQ4_XS) GGXP(MXFP4)
GGXP(Q2_K) GGXP(Q3_K) GGXP(Q4_K) GGXP(Q5_K) GGXP(Q6_K)
#endif

// Input rows (optionally gathered: row r is source row src[r], zeros for ~0) to half with a
// power-of-two scale per row (see above); rs gets the inverse scale. Source rows are `stride`
// floats apart. One threadgroup of 256 per row; n % 4 == 0.
struct XH { uint n; uint gather; uint stride; uint _p; };
inline float row_scale(float m, thread float& inv) {
    int e = 0;
    if (m > 0.f && isfinite(m)) { frexp(m, e); e = clamp(14 - e, -100, 100); }
    inv = ldexp(1.0f, -e);
    return ldexp(1.0f, e);
}
inline float tg_max(float m, uint sg, uint lane, threadgroup float* red) {
    m = simd_max(m);
    if (lane == 0) red[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return max(max(max(red[0], red[1]), max(red[2], red[3])), max(max(red[4], red[5]), max(red[6], red[7])));
}
kernel void x_half(device const float* x [[buffer(0)]], device half* xh [[buffer(1)]], device float* rs [[buffer(2)]],
                   constant XH& p [[buffer(3)]], device const uint* src [[buffer(4)]],
                   uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8];
    uint sr = p.gather != 0 ? src[r] : r;
    uint n4 = p.n / 4;
    if (sr == 0xffffffffu) {
        device half4* out = (device half4*)(xh + (ulong)r * p.n);
        for (uint i = tid; i < n4; i += 256) out[i] = half4(0.f);
        if (tid == 0) rs[r] = 1.f;
        return;
    }
    device const float4* row = (device const float4*)(x + (ulong)sr * p.stride);
    float m = 0.f;
    for (uint i = tid; i < n4; i += 256) { float4 v = abs(row[i]); m = max(m, max(max(v.x, v.y), max(v.z, v.w))); }
    m = tg_max(m, sg, lane, red);
    float inv, s = row_scale(m, inv);
    device half4* out = (device half4*)(xh + (ulong)r * p.n);
    for (uint i = tid; i < n4; i += 256) out[i] = half4(row[i] * s);
    if (tid == 0) rs[r] = inv;
}
// act(gate) * up (SiLU, or GELU-tanh with `gelu` = 1, squared ReLU of the gate alone with 2; with `limit` > 0 gate clamped above and up
// both ways first, DeepSeek V4) of fused [gate | up] rows (n each) into scaled half rows, as x_half; with `split` = G nonzero, gate row r is at r * n and up row r at
// (G + r) * n (gate and up experts stored apart).
inline float4 silu_mul4(float4 g, float4 u) { return g / (1.0f + exp(-g)) * u; }
inline float4 gelu_mul4(float4 g, float4 u) { return g / (1.0f + exp(-1.5957691f * (g + 0.044715f * g * g * g))) * u; }
inline float4 act_mul4(float4 g, float4 u, uint gelu, float limit) { if (gelu == 2) { float4 r = max(g, 0.f); return r * r; } if (limit > 0.f) { g = min(g, limit); u = clamp(u, -limit, limit); } return gelu != 0 ? gelu_mul4(g, u) : silu_mul4(g, u); }
struct SH { uint n; uint gelu; uint split; float limit; };
kernel void silu_half(device const float* gu [[buffer(0)]], device half* ah [[buffer(1)]], device float* rs [[buffer(2)]],
                      constant SH& p [[buffer(3)]],
                      uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8];
    // (gelu == 2: squared ReLU of rows of n alone, Nemotron-H's ungated experts.)
    device const float4* g = (device const float4*)(gu + (ulong)r * (p.split != 0 || p.gelu == 2 ? 1 : 2) * p.n);
    device const float4* u = p.gelu == 2 ? g : p.split != 0 ? (device const float4*)(gu + (ulong)(p.split + r) * p.n) : g + p.n / 4;
    uint n4 = p.n / 4;
    float m = 0.f;
    for (uint i = tid; i < n4; i += 256) { float4 v = abs(act_mul4(g[i], u[i], p.gelu, p.limit)); m = max(m, max(max(v.x, v.y), max(v.z, v.w))); }
    m = tg_max(m, sg, lane, red);
    float inv, s = row_scale(m, inv);
    device half4* out = (device half4*)(ah + (ulong)r * p.n);
    for (uint i = tid; i < n4; i += 256) out[i] = half4(act_mul4(g[i], u[i], p.gelu, p.limit) * s);
    if (tid == 0) rs[r] = inv;
}
// Fused attention for 128-wide heads, no mask, on the matrix units: q / k / v of type T (the
// model's [rows][heads * 128] layout, q rows padded to 64, k / v rows to 64 with zeros), f32
// output. One threadgroup (4 SIMD-groups) per 64 queries of one head; keys stream in blocks of 64.
// Scores S = Q K^T (f32) go through threadgroup memory for the online softmax, probabilities come
// back as T in P for O += P V; O stays in a cooperative tensor, rescaled per query row as the max
// moves. T = float runs at f32 precision (flash_attn128_mma_f32; the half kernel is
// flash_attn128_reg below, which keeps S and P in registers). P reuses S's memory (each
// row of halves in the first half of its row of floats): each pair of lanes reads its row's
// scores into registers before either writes probabilities over them, and the pair is in one
// SIMD-group, so the reads complete first; the threadgroup holds 16 KB instead of 24.
#if MMA_UNITS
struct FAM { uint tq; uint tk; uint d; float scale; };
template <typename T>
inline void flash_attn128_body(device const T* q, device const T* k, device const T* v, device float* o, constant FAM& p,
                               threadgroup float* S, threadgroup T* P, threadgroup float* A, threadgroup float* L,
                               threadgroup uint* RS, uint2 tg, uint tid) {
    uint q0 = tg.x * 64, h0 = tg.y * 128;
    uint tqp = (p.tq + 63) / 64 * 64, tkp = (p.tk + 63) / 64 * 64;
    tensor<device T, dextents<int32_t, 2>, tensor_inline> Qt((device T*)q, dextents<int32_t, 2>(p.d, tqp));
    tensor<device T, dextents<int32_t, 2>, tensor_inline> Kt((device T*)k, dextents<int32_t, 2>(p.d, tkp));
    tensor<device T, dextents<int32_t, 2>, tensor_inline> Vt((device T*)v, dextents<int32_t, 2>(p.d, tkp));
    tensor<threadgroup float, extents<int32_t, 64, 64>, tensor_inline> St(S, extents<int32_t, 64, 64>());
    constexpr int PLD = 64 * sizeof(float) / sizeof(T);
    tensor<threadgroup T, dextents<int32_t, 2>, tensor_inline> Pt(P, dextents<int32_t, 2>(64, 64), array<int32_t, 2>{1, PLD});
    constexpr auto ds = matmul2d_descriptor(64, 64, 128, false, true, false);
    constexpr auto dov = matmul2d_descriptor(64, 128, 64, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<ds, execution_simdgroups<4>> ops;
    matmul2d<dov, execution_simdgroups<4>> opo;
    auto qa = Qt.template slice<128, 64>(h0, q0);
    auto vb0 = Vt.template slice<128, 64>(h0, 0);
    auto co = opo.template get_destination_cooperative_tensor<decltype(Pt), decltype(vb0), float>();
    zero(co);
    uint r = tid / 2, part = tid % 2;
    float m = -INFINITY, l = 0.0f;
    threadgroup uint& rescale = RS[0];
    if (tid == 0) rescale = 0;
    for (uint kb = 0; kb < p.tk; kb += 64) {
        auto kt = Kt.template slice<128, 64>(h0, kb);
        auto cs = ops.template get_destination_cooperative_tensor<decltype(qa), decltype(kt), float>();
        zero(cs);
        ops.run(qa, kt, cs);
        cs.store(St);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float x[32];
        float bm = -INFINITY;
        for (uint j = 0; j < 32; j++) {
            uint col = part * 32 + j;
            x[j] = kb + col < p.tk ? S[r * 64 + col] * p.scale : -INFINITY;
            bm = max(bm, x[j]);
        }
        bm = max(bm, simd_shuffle_xor(bm, 1));
        // Lazy rescaling: the running max moves only when a score exceeds it by more than 8
        // (probabilities then stay below e^8, well within half range), so most key blocks skip
        // rescaling the output.
        bool grow = bm > m + 8.0f;
        float mn = grow ? bm : m;
        float alpha = grow ? exp(m - mn) : 1.0f;
        if (grow) rescale = 1;
        float ps = 0.0f;
        for (uint j = 0; j < 32; j++) { float e = exp(x[j] - mn); P[r * PLD + part * 32 + j] = T(e); ps += e; }
        ps += simd_shuffle_xor(ps, 1);
        l = l * alpha + ps;
        m = mn;
        if (part == 0) A[r] = alpha;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (rescale != 0) {
#pragma unroll
            for (uint16_t i = 0; i < co.get_capacity(); ++i)
                if (co.is_valid_element(i)) { auto idx = co.get_multidimensional_index(i); co[i] *= A[idx[1]]; }
        }
        auto vt = Vt.template slice<128, 64>(h0, kb);
        opo.run(Pt, vt, co);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) rescale = 0;
    }
    if (part == 0) L[r] = 1.0f / l;
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint16_t i = 0; i < co.get_capacity(); ++i)
        if (co.is_valid_element(i)) {
            auto idx = co.get_multidimensional_index(i);
            uint row = q0 + idx[1];
            if (row < p.tq) o[(ulong)row * p.d + h0 + idx[0]] = co[i] * L[idx[1]];
        }
}
// The fast kernel's arithmetic with the scores, probabilities and output in the matrix-unit
// registers of each SIMD-group (flash_attn128_mma, and dit_flash128_hilo with HILO: queries and
// keys as hi + lo half pairs, scores Qh Kh^T + Qh Kl^T + Ql Kh^T): each of the 4 SIMD-groups owns
// MQ = 16 of the threadgroup's 64 queries and runs alone over key blocks of KB = 32 (the row
// maxima and sums in row-reduction tensors, the rescaling factors through MQ floats of
// threadgroup memory), as un_flash_body in metal_unet.metal: 1.7x the speed of
// flash_attn128_body over FLUX.1's 4608 tokens (`metal::unet::tests::fa128_bench`).
template <bool HILO, int MQ = 16, int KB = 32>
inline void flash_attn128_reg(device const half* q, device const half* k, device const half* v, device float* o, constant FAM& p,
                                  device const half* ql, device const half* kl, threadgroup float* T, uint2 tg, uint sg) {
    uint q0 = tg.x * 64 + sg * MQ, h0 = tg.y * 128;
    uint tqp = (p.tq + 63) / 64 * 64, tkp = (p.tk + 63) / 64 * 64;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> Qt((device half*)q, dextents<int32_t, 2>(p.d, tqp));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> Qlt((device half*)ql, dextents<int32_t, 2>(p.d, tqp));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> Kt((device half*)k, dextents<int32_t, 2>(p.d, tkp));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> Klt((device half*)kl, dextents<int32_t, 2>(p.d, tkp));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> Vt((device half*)v, dextents<int32_t, 2>(p.d, tkp));
    using OPS = matmul2d<matmul2d_descriptor(MQ, KB, 128, false, true, false, matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroup>;
    using OPO = matmul2d<matmul2d_descriptor(MQ, 128, KB, false, false, false, matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroup>;
    OPS ops;
    OPO opo;
    auto qa = Qt.template slice<128, MQ>(h0, q0);
    auto qla = Qlt.template slice<128, MQ>(h0, q0);
    auto kb0 = Kt.template slice<128, KB>(h0, 0);
    auto vb0 = Vt.template slice<128, KB>(h0, 0);
    using PL = typename OPO::template cooperative_tensor_left_input_t<half, half, float>;
    auto co = opo.template get_destination_cooperative_tensor<PL, decltype(vb0), float>();
    zero(co);
    auto mrun = ops.template get_row_reduction_destination_cooperative_tensor<decltype(qa), decltype(kb0), float>();
    auto lrun = ops.template get_row_reduction_destination_cooperative_tensor<decltype(qa), decltype(kb0), float>();
    auto rm = ops.template get_row_reduction_destination_cooperative_tensor<decltype(qa), decltype(kb0), float>();
    auto alpha = ops.template get_row_reduction_destination_cooperative_tensor<decltype(qa), decltype(kb0), float>();
    for (uint16_t i = 0; i < mrun.get_capacity(); ++i) { mrun[i] = -INFINITY; lrun[i] = 0.f; }
    threadgroup float* A = T + sg * MQ;
    for (uint kb = 0; kb < p.tk; kb += KB) {
        auto kt = Kt.template slice<128, KB>(h0, kb);
        auto cs = ops.template get_destination_cooperative_tensor<decltype(qa), decltype(kb0), float>();
        zero(cs);
        if (HILO) {
            auto klt = Klt.template slice<128, KB>(h0, kb);
            ops.run(qla, kt, cs);
            ops.run(qa, klt, cs);
        }
        ops.run(qa, kt, cs);
        bool tail = kb + KB > p.tk;
#pragma unroll
        for (uint16_t i = 0; i < cs.get_capacity(); ++i) {
            float x = cs[i] * p.scale;
            if (tail && kb + cs.get_multidimensional_index(i)[0] >= p.tk) x = -INFINITY;
            cs[i] = x;
        }
        reduce_rows(cs, rm, reduction_operation::max, -INFINITY);
        bool grew = false;
        for (uint16_t i = 0; i < rm.get_capacity(); ++i) {
            float bm = rm[i], mo = mrun[i];
            bool grow = bm > mo + 8.0f;
            float mn = grow ? bm : mo;
            alpha[i] = grow ? exp(mo - mn) : 1.0f;
            mrun[i] = mn;
            grew = grew || grow;
        }
        for (auto it = cs.begin(); it != cs.end(); it++) {
            auto mi = mrun.map_iterator(it);
            *it = exp(*it - *mi);
        }
        reduce_rows(cs, rm, reduction_operation::sum, 0.0f);
        for (uint16_t i = 0; i < rm.get_capacity(); ++i) lrun[i] = lrun[i] * alpha[i] + rm[i];
        auto ph = ops.template get_destination_cooperative_tensor<decltype(qa), decltype(kb0), half>();
#pragma unroll
        for (uint16_t i = 0; i < cs.get_capacity(); ++i) ph[i] = half(cs[i]);
        auto pl = opo.template get_left_input_cooperative_tensor<half, half, float>(ph);
        if (simd_any(grew)) {
            for (uint16_t i = 0; i < alpha.get_capacity(); ++i)
                if (alpha.is_valid_element(i)) A[alpha.get_multidimensional_index(i)[0]] = alpha[i];
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint16_t i = 0; i < co.get_capacity(); ++i)
                if (co.is_valid_element(i)) co[i] *= A[co.get_multidimensional_index(i)[1]];
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        auto vt = Vt.template slice<128, KB>(h0, kb);
        opo.run(pl, vt, co);
    }
    for (uint16_t i = 0; i < lrun.get_capacity(); ++i)
        if (lrun.is_valid_element(i)) A[lrun.get_multidimensional_index(i)[0]] = 1.0f / lrun[i];
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint16_t i = 0; i < co.get_capacity(); ++i)
        if (co.is_valid_element(i)) {
            auto idx = co.get_multidimensional_index(i);
            uint row = q0 + idx[1];
            if (row < p.tq) o[(ulong)row * p.d + h0 + idx[0]] = co[i] * A[idx[1]];
        }
}
kernel void flash_attn128_mma(device const half* q [[buffer(0)]], device const half* k [[buffer(1)]],
                              device const half* v [[buffer(2)]], device float* o [[buffer(3)]], constant FAM& p [[buffer(4)]],
                              uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float T[64];
    flash_attn128_reg<false>(q, k, v, o, p, q, k, T, tg, sg);
}
kernel void flash_attn128_mma_f32(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]],
                                  device const float* v [[buffer(2)]], device float* o [[buffer(3)]], constant FAM& p [[buffer(4)]],
                                  uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float S[64 * 64];
    threadgroup float A[64];
    threadgroup float L[64];
    threadgroup uint RS[1];
    flash_attn128_body<float>(q, k, v, o, p, S, S, A, L, RS, tg, tid);
}
#endif
// out[t] = sum over the token's k slots of weight * o[row] (`add`: out[t] += the sum).
struct CB { uint n; uint k; uint ldo; uint add; };
kernel void moe_combine(device const float* o [[buffer(0)]], device const float2* slots [[buffer(1)]], device float* out [[buffer(2)]],
                        constant CB& p [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    uint c = gid.x * 4, t = gid.y;
    if (c >= p.n) return;
    float4 acc = 0.f;
    for (uint j = 0; j < p.k; j++) {
        float2 s = slots[t * p.k + j];
        acc += s.y * *(device const float4*)(o + (ulong)as_type<uint>(s.x) * p.n + c);
    }
    device float4* d = (device float4*)(out + (ulong)t * p.ldo + c);
    *d = p.add != 0 ? *d + acc : acc;
}
// The routing of a prefill chunk's MoE layer on the device, as the host's `pf_route` lays it
// out (so the GPU need not stop for the host at each MoE layer): per token, `k` expert ids (as
// bits) then `k` weights in `sel`. One threadgroup of 256 per expert: each counts every expert's
// entries and scans the counts (the experts' segments, 16-row multiples, cut into tiles as
// `group_tiles`: class c from tile c * max_tiles), then writes its own tiles and places its own
// entries in token order, 256 at a time (each entry's rank among its expert's by a prefix sum):
// the gathered row's source token (`src`, ~0 for padding), its expert (`rexp`) and each slot's
// [row, weight]. Threadgroup 0 writes `args`: indirect dispatch sizes, (rows, 1, 1), then (gu_x,
// tiles of class c, 1) for c = 0..2, then (dn_x, tiles of class c, 1).
struct RT { uint t; uint k; uint n_exp; uint sel_stride; uint max_tiles; uint gu_x; uint dn_x; uint _p; };
#define RT_THREADS 256
#define RT_PER 4
inline uint rt_id(device const float* sel, constant RT& p, uint i) {
    return as_type<uint>(sel[(ulong)(i / p.k) * p.sel_stride + i % p.k]);
}
// Exclusive prefix sum of v over the threadgroup (RT_THREADS threads); `total` gets the sum.
inline uint rt_scan(uint v, uint tid, threadgroup uint* part, thread uint& total) {
    uint lane = tid % 32, sg = tid / 32;
    uint ex = simd_prefix_exclusive_sum(v);
    if (lane == 31) part[sg] = ex + v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint before = 0, all = 0;
    for (uint q = 0; q < RT_THREADS / 32; q++) {
        uint x = part[q];
        before += q < sg ? x : 0;
        all += x;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total = all;
    return before + ex;
}
kernel void pf_route(device const float* sel [[buffer(0)]], device uint* src [[buffer(1)]], device uint* rexp [[buffer(2)]],
                     device uint2* slots [[buffer(3)]], device uint4* tiles [[buffer(4)]], device uint* args [[buffer(5)]],
                     constant RT& p [[buffer(6)]], uint me [[threadgroup_position_in_grid]],
                     uint tid [[thread_position_in_threadgroup]]) {
    threadgroup atomic_uint cnt[RT_THREADS * RT_PER];
    threadgroup uint part[RT_THREADS / 32];
    threadgroup uint mine[5];
    uint n = p.t * p.k;
    for (uint x = tid; x < p.n_exp; x += RT_THREADS) atomic_store_explicit(&cnt[x], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < n; i += RT_THREADS) {
        uint e = rt_id(sel, p, i);
        if (e < p.n_exp) atomic_fetch_add_explicit(&cnt[e], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Experts RT_PER * tid ..: their padded lengths and tiles of each class, then the prefix sums.
    uint len[RT_PER], c0[RT_PER], c1[RT_PER], c2[RT_PER];
    uint sl = 0, s0 = 0, s1 = 0, s2 = 0;
    for (uint j = 0; j < RT_PER; j++) {
        uint x = tid * RT_PER + j;
        uint c = x < p.n_exp ? atomic_load_explicit(&cnt[x], memory_order_relaxed) : 0;
        uint l = (c + 15) / 16 * 16;
        uint a = l / 128, r = l - a * 128, b = r / GTM;
        len[j] = l; c0[j] = a; c1[j] = b; c2[j] = r - b * GTM > 0 ? 1 : 0;
        sl += l; s0 += a; s1 += b; s2 += c2[j];
    }
    uint rows, n0, n1, n2;
    uint bl = rt_scan(sl, tid, part, rows), b0 = rt_scan(s0, tid, part, n0);
    uint b1 = rt_scan(s1, tid, part, n1), b2 = rt_scan(s2, tid, part, n2);
    for (uint j = 0; j < RT_PER; j++) {
        if (tid * RT_PER + j == me) {
            mine[0] = bl; mine[1] = len[j]; mine[2] = b0; mine[3] = b1; mine[4] = b2;
        }
        bl += len[j]; b0 += c0[j]; b1 += c1[j]; b2 += c2[j];
    }
    if (me == 0 && tid == 0) {
        args[0] = rows; args[1] = 1; args[2] = 1;
        uint nc[3] = { n0, n1, n2 };
        for (uint c = 0; c < 3; c++) {
            args[3 + 3 * c] = p.gu_x; args[4 + 3 * c] = nc[c]; args[5 + 3 * c] = 1;
            args[12 + 3 * c] = p.dn_x; args[13 + 3 * c] = nc[c]; args[14 + 3 * c] = 1;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint start = mine[0], l = mine[1];
    if (tid == 0) {
        uint t0 = 0, i0 = mine[2], i1 = mine[3];
        for (; t0 + 128 <= l; t0 += 128) tiles[i0++] = uint4(me, start, l, t0);
        for (; t0 + GTM <= l; t0 += GTM) tiles[p.max_tiles + i1++] = uint4(me, start, l, t0);
        if (t0 < l) tiles[2 * p.max_tiles + mine[4]] = uint4(me, start, l, t0);
    }
    uint c = atomic_load_explicit(&cnt[me], memory_order_relaxed);
    for (uint r = c + tid; r < l; r += RT_THREADS) { src[start + r] = 0xffffffffu; rexp[start + r] = 0xffffffffu; }
    uint base = 0;
    for (uint c0i = 0; c0i < n && base < c; c0i += RT_THREADS) {
        uint i = c0i + tid;
        bool hit = i < n && rt_id(sel, p, i) == me;
        uint tot;
        uint rank = rt_scan(hit ? 1u : 0u, tid, part, tot);
        if (hit) {
            uint r = start + base + rank, j = i / p.k, s = i % p.k;
            src[r] = j;
            rexp[r] = me;
            slots[i] = uint2(r, as_type<uint>(sel[(ulong)j * p.sel_stride + p.k + s]));
        }
        base += tot;
    }
}
// Routed experts' matrices in the grouped GEMMs' tiled layout (see `gblk` and `kpart`;
// `tiled_experts`): expert e's rows (nb blocks of bs bytes each) from the address at src[e *
// stride] to dst + e * mb, one thread per block of the destination, in its order (block j: band
// j / (64 nb), column (j / 64) % nb, row j % 64 of the band).
// A block's bytes go as `nseg` segments: seg[i] = (offset in the block, length, offset in the
// band's 64-block column, part length: the row's copy at that offset plus row % 64 times it).
struct TB { uint rows; uint nb; uint bs; uint stride; ulong mb; uint nseg; uint even; uint4 seg[8]; };
struct TSrc { device const uchar* q; };
kernel void tile_blocks(device const TSrc* src [[buffer(0)]], device uchar* dst [[buffer(1)]], constant TB& p [[buffer(2)]],
                        uint2 gid [[thread_position_in_grid]]) {
    uint j = gid.x, e = gid.y;
    if (j >= p.rows * p.nb) return;
    uint band = j / (64 * p.nb), c = (j / 64) % p.nb, slot = j % 64, r = band * 64 + slot;
    device const uchar* s = src[e * p.stride].q + ((ulong)r * p.nb + c) * p.bs;
    device uchar* col = dst + e * p.mb + (ulong)(j - slot) * p.bs;
    for (uint k = 0; k < p.nseg; k++) {
        uint4 g = p.seg[k];
        device const uchar* a = s + g.x;
        device uchar* d = col + g.z + slot * g.w;
        if (p.even != 0) {
            for (uint i = 0; i < g.y / 2; i++) ((device ushort*)d)[i] = ((device const ushort*)a)[i];
        } else {
            for (uint i = 0; i < g.y; i++) d[i] = a[i];
        }
    }
}
"#;

/// GGUF types with a Metal GEMM (`gemm_<TYPE>` in `GGUF_SHADER_SRC`) and a decode matvec
/// (`metal_decode.rs`). FP8 E4M3 (128 x 128 block scales, [`fp8_ok`]) has the GEMMs only.
const GGUF_GPU_TYPES: [GgmlType; 24] = [
    GgmlType::F32,
    GgmlType::F16,
    GgmlType::Q4_1,
    GgmlType::Q5_0,
    GgmlType::Q5_1,
    GgmlType::Q4_0,
    GgmlType::IQ4_NL,
    GgmlType::Q8_0,
    GgmlType::Q4_K,
    GgmlType::Q5_K,
    GgmlType::Q6_K,
    GgmlType::Q3_K,
    GgmlType::Q2_K,
    GgmlType::IQ4_XS,
    GgmlType::IQ2_XXS,
    GgmlType::IQ2_XS,
    GgmlType::IQ2_S,
    GgmlType::IQ3_XXS,
    GgmlType::IQ3_S,
    GgmlType::IQ1_S,
    GgmlType::IQ1_M,
    GgmlType::Q2_0,
    GgmlType::PQ2_0,
    GgmlType::PTQ1_0,
];

/// GGUF types with a matrix-unit GEMM (`mma_<TYPE>` / `mmag_<TYPE>` in `MMA_SHADER_SRC`): all but
/// F32, whose values would lose precision as half, and MXFP4 / NVFP4 (which have the GEMMs but
/// no decode matvec, so they are not in [`GGUF_GPU_TYPES`]; NVFP4, with its tensor-wide factor,
/// no pair GEMMs either).
const GGUF_MMA_TYPES: [GgmlType; 26] = [
    GgmlType::MXFP4,
    GgmlType::NVFP4,
    GgmlType::F8E4M3,
    GgmlType::F16,
    GgmlType::Q4_1,
    GgmlType::Q5_0,
    GgmlType::Q5_1,
    GgmlType::Q4_0,
    GgmlType::IQ4_NL,
    GgmlType::Q8_0,
    GgmlType::Q4_K,
    GgmlType::Q5_K,
    GgmlType::Q6_K,
    GgmlType::Q3_K,
    GgmlType::Q2_K,
    GgmlType::IQ4_XS,
    GgmlType::IQ2_XXS,
    GgmlType::IQ2_XS,
    GgmlType::IQ2_S,
    GgmlType::IQ3_XXS,
    GgmlType::IQ3_S,
    GgmlType::IQ1_S,
    GgmlType::IQ1_M,
    GgmlType::Q2_0,
    GgmlType::PQ2_0,
    GgmlType::PTQ1_0,
];

/// The grouped GEMM tiles of a MoE layer with `count[e]` rows for expert e: the gathered input
/// holds one segment per expert, padded to a multiple of 16 rows, cut into tiles of 128 rows while
/// that many remain, then of `MMA_GTM` (32), then one of 16 for an odd remainder. Returns the
/// tiles ([expert, segment start, segment length, tile start]) listed by size (128, 32, 16: one
/// grid each), how many of each, every segment's start and the rows in all. A layer of t tokens
/// with k experts each has at most `t k + 15 n_exp` rows and a tile per 16 rows.
pub(crate) fn group_tiles(count: &[usize]) -> (Vec<[u32; 4]>, [usize; 3], Vec<usize>, usize) {
    let mut cls: [Vec<[u32; 4]>; 3] = Default::default();
    let mut start = Vec::with_capacity(count.len());
    let mut rows = 0usize;
    for (e, &c) in count.iter().enumerate() {
        start.push(rows);
        let len = c.next_multiple_of(16);
        let tile = |t0: usize| [e as u32, rows as u32, len as u32, t0 as u32];
        let mut t0 = 0;
        while t0 + 128 <= len {
            cls[0].push(tile(t0));
            t0 += 128;
        }
        while t0 + MMA_GTM <= len {
            cls[1].push(tile(t0));
            t0 += MMA_GTM;
        }
        if t0 < len {
            cls[2].push(tile(t0));
        }
        rows += len;
    }
    let n = [cls[0].len(), cls[1].len(), cls[2].len()];
    (cls.concat(), n, start, rows)
}

fn find_kind(v: &[(GgmlType, Id)], k: GgmlType) -> Option<Id> {
    v.iter().find(|(t, _)| *t == k).map(|(_, p)| *p)
}

/// Whether the device kernels take `m`: anything but FP8, FP8 with the 128 x 128 block scales
/// the shaders assume, or NVFP4 with its one tensor-wide factor (`MK_NV`); not other formats
/// with extra scales (row-wide factors the shaders would leave out).
fn fp8_ok(m: &crate::ggml::GgufMatrix) -> bool {
    match &m.block_scales {
        None => m.kind != GgmlType::F8E4M3,
        Some(b) => {
            (m.kind == GgmlType::F8E4M3 && b.rows == 128 && b.cols == 128)
                || (m.kind == GgmlType::NVFP4 && b.s.len() == 1 && b.e8m0.is_none() && !m.x4)
        }
    }
}

/// `P::s_off` of a GGUF matrix: the row length in bytes, or for FP8 the offset of its block
/// scales (in floats) inside their buffer.
fn gguf_s_off(m: &crate::ggml::GgufMatrix, mw: &MetalWeight) -> usize {
    match &mw.scales {
        Some(s) if m.block_scales.is_some() => s.offset_bytes / 4,
        _ => m.row_bytes(),
    }
}

/// The i-quant grids as MSL constant arrays.
fn gguf_shader_tables() -> String {
    use crate::ggml::tables::*;
    use std::fmt::Write;
    let mut s = String::new();
    let arr = |s: &mut String, ty: &str, name: &str, vals: &mut dyn Iterator<Item = String>| {
        let _ = write!(s, "constant {ty} {name}[] = {{");
        for v in vals {
            let _ = write!(s, "{v},");
        }
        s.push_str("};\n");
    };
    arr(
        &mut s,
        "float",
        "kvalues_iq4nl",
        &mut KVALUES_IQ4NL.iter().map(|v| format!("{v}.f")),
    );
    arr(
        &mut s,
        "float",
        "kvalues_mxfp4",
        &mut KVALUES_MXFP4.iter().map(|v| format!("{v}.f")),
    );
    for (name, t) in [
        ("iq2xxs_grid", &IQ2XXS_GRID[..]),
        ("iq2xs_grid", &IQ2XS_GRID[..]),
        ("iq2s_grid", &IQ2S_GRID[..]),
        ("iq1s_grid", &IQ1S_GRID[..]),
    ] {
        arr(&mut s, "ulong", name, &mut t.iter().map(|v| format!("{v:#x}ul")));
    }
    for (name, t) in [("iq3xxs_grid", &IQ3XXS_GRID[..]), ("iq3s_grid", &IQ3S_GRID[..])] {
        arr(&mut s, "uint", name, &mut t.iter().map(|v| format!("{v:#x}u")));
    }
    s
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ConvParams {
    n_out: u32,
    n_in: u32,
    w_off: u32,
    s_off: u32,
    h: u32,
    w: u32,
    cin: u32,
    k: u32,
    stride: u32,
    ow: u32,
    npix: u32,
    pad: u32,
    mode: u32,
    silu: u32,
    cpg: u32,
    _p: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    n_out: u32,
    n_in: u32,
    w_off: u32,
    s_off: u32,
}

/// Parameters of the matrix-unit GEMMs (`MP` in the shader).
#[repr(C)]
#[derive(Clone, Copy)]
struct MmaParams {
    n_out: u32,
    n_in: u32,
    w_off: u32,
    s_off: u32,
    rows: u32,
    _p: [u32; 3],
}

/// Tokens per threadgroup of the dense matrix-unit GEMM (`MMA_TM_S` for inputs of at most
/// `MMA_SMALL` rows, which would mostly be padding in a 128-row tile), and of the grouped one.
const MMA_TM: usize = 128;
const MMA_TM_S: usize = 32;
const MMA_SMALL: usize = 64;
const MMA_GTM: usize = 32;
/// K step of the matrix-unit GEMMs (`n_in` must be a multiple).
const MMA_KB: usize = 64;

/// Which matrix-unit GEMM: dense with 128-, 32- or 64-token tiles, or grouped (MoE).
#[derive(Clone, Copy, PartialEq)]
enum Tile {
    Dense,
    Small,
    Mid,
    Grouped,
}

/// Matrix-unit pipelines (GPU cores with neural accelerators: M5, A19), or the same kernels on
/// SIMD-group matrices (`units` false: earlier GPUs, which then have no `flash128*`).
struct Mma {
    /// Whether these are the matrix-unit (Metal 4 `matmul2d`) kernels.
    units: bool,
    x_half: Id,
    silu_half: Id,
    combine: Id,
    /// bf16 weights read from device memory, 64 and 128 outputs per threadgroup.
    bf: [Id; 2],
    q8: Id,
    q4: Id,
    q4a: Id,
    gguf: Vec<(GgmlType, Id)>,
    /// The same with 32-token tiles.
    bf_s: [Id; 2],
    q8_s: Id,
    q4_s: Id,
    q4a_s: Id,
    gguf_s: Vec<(GgmlType, Id)>,
    /// Grouped (MoE) variants.
    g_bf: Id,
    g_q8: Id,
    g_q4: Id,
    g_gguf: Vec<(GgmlType, Id)>,
    /// Fused attention for 128-wide heads on the matrix units: half operands, and f32 operands
    /// (as accurate as `flash_attn128`, ~15% faster).
    flash128: Id,
    flash128_f32: Id,
}

impl Mma {
    /// The fused 128-wide attention kernels (matrix units only).
    fn flash128(&self) -> Option<(Id, Id)> {
        self.units.then_some((self.flash128, self.flash128_f32))
    }
}

#[repr(C)]
struct MtlSize {
    w: usize,
    h: usize,
    d: usize,
}

/// A device buffer plus the byte offset of the wrapped data inside it.
struct Buf {
    id: Id,
    offset_bytes: usize,
}

struct MetalWeight {
    data: Buf,
    scales: Option<Buf>,
    /// Affine Q4: the offsets buffer, offsets per row, and blocks of 32 per offset group.
    offsets: Option<(Buf, u32, u32)>,
    /// `objc_msgSend` as `release`, and its selector.
    release: (MsgVoid, Sel),
    /// The queue's residency set, once the buffers are in it (see `MetalWeight::keep_resident`).
    resident: std::sync::OnceLock<std::sync::Arc<Residency>>,
}

impl MetalWeight {
    fn ids(&self) -> Vec<Id> {
        [Some(&self.data), self.scales.as_ref(), self.offsets.as_ref().map(|o| &o.0)]
            .into_iter()
            .flatten()
            .map(|b| b.id)
            .collect()
    }

    /// Puts the buffers in the queue's residency set (once). Bound per submission instead, a
    /// buffer wrapping mapped file pages is made resident again every time, which for the few
    /// rows of a draft model's matmuls costs several times the kernels (~30 ms per GB).
    fn keep_resident(&self, r: &Option<std::sync::Arc<Residency>>) {
        if let Some(r) = r
            && self.resident.get().is_none()
            && self.resident.set(r.clone()).is_ok()
        {
            r.update(&self.ids(), true);
        }
    }
}

impl Drop for MetalWeight {
    fn drop(&mut self) {
        if let Some(r) = self.resident.get() {
            r.update(&self.ids(), false);
        }
        let (f, sel) = self.release;
        let ids = [Some(&self.data), self.scales.as_ref(), self.offsets.as_ref().map(|o| &o.0)];
        for b in ids.into_iter().flatten() {
            // SAFETY: each buffer was created by this weight and is released once.
            unsafe { f(b.id, sel) };
        }
    }
}

/// An `MTLResidencySet` attached to the command queue, and its selectors.
pub(crate) struct Residency {
    /// Several sets, attached to the queue: buffers are spread over them and each set's
    /// residency is requested from a thread of its own (the wiring of mapped files is the
    /// bulk of a first pass's setup and goes faster in parallel).
    sets: Vec<Id>,
    add: Sel,
    remove: Sel,
    commit: Sel,
    request: Sel,
    /// `objc_msgSend` with one object argument, and with none.
    send_id: unsafe extern "C" fn(Id, Sel, Id),
    send: MsgVoid,
    /// Serializes changes to the set.
    lock: Mutex<()>,
}
unsafe impl Send for Residency {}
unsafe impl Sync for Residency {}

impl Residency {
    /// Adds (or removes) buffers, then commits the change (added buffers are made resident
    /// before it returns).
    pub(crate) fn update(&self, ids: &[Id], add: bool) {
        let _g = self.lock.lock().unwrap();
        let n = self.sets.len().min(ids.len()).max(1);
        // SAFETY: the sets are live residency sets and `ids` are live buffers.
        unsafe {
            for (i, &id) in ids.iter().enumerate() {
                if add {
                    (self.send_id)(self.sets[i % n], self.add, id);
                } else {
                    for &s in &self.sets {
                        (self.send_id)(s, self.remove, id);
                    }
                }
            }
            for &s in if add { &self.sets[..n] } else { &self.sets[..] } {
                (self.send)(s, self.commit);
            }
        }
        // Residency requested for the sets just added to, in parallel (78 GB of mapped expert
        // weights: ~1 s from one thread, ~0.3-0.5 s from eight).
        if add && n == 1 {
            // SAFETY: a live residency set.
            unsafe { (self.send)(self.sets[0], self.request) };
        } else if add {
            struct P(Id);
            unsafe impl Send for P {}
            unsafe impl Sync for P {}
            let sets: Vec<P> = self.sets[..n].iter().map(|&s| P(s)).collect();
            std::thread::scope(|sc| {
                for s in &sets {
                    // SAFETY: a live residency set.
                    sc.spawn(move || unsafe { (self.send)(s.0, self.request) });
                }
            });
        }
    }
}

/// Residency sets attached to the queue (see `Residency`).
const RESIDENCY_SETS: usize = 8;

/// Device state of an MoE layer's routed experts (`Backend::upload_moe`).
struct MoeDev {
    /// Grouped GEMM pipelines (tiles of 32, 128 and 16 rows) for the gate/up and the down
    /// projections.
    pipes: ([Id; 3], [Id; 3]),
    /// GGUF row bytes of each (the `s_off` parameter), 0 for other formats.
    row_bytes: (u32, u32),
    /// The wrapped weight memory (adjacent experts share a buffer).
    bufs: Vec<Buf>,
    /// Per-expert (data, scales) GPU addresses of each projection, for `n_exp` experts.
    ep: (Id, Id),
    /// Separate up projections (GGUF): their pipeline, row bytes and address table; `pipes.0`
    /// and `ep.0` are then the gate projections alone.
    up: Option<([Id; 3], u32, Id)>,
    n_exp: usize,
    /// The experts' activation.
    act: crate::gpu::MoeAct,
    /// The residency set holding `bufs`, if any (else they are made resident per submission).
    residency: Option<std::sync::Arc<Residency>>,
    release: (MsgVoid, Sel),
    /// Tiled copies shared with the model's plans (`tiled_experts`), and their keeper (the
    /// process's backend, which outlives the state).
    shared: Vec<Id>,
    backend: *const MetalBackend,
}

impl Drop for MoeDev {
    fn drop(&mut self) {
        // SAFETY: the backend outlives the states it makes (`backend`).
        let last: Vec<Id> = self
            .shared
            .iter()
            .copied()
            .filter(|&id| unsafe { (*self.backend).unwrap_shared(id) })
            .collect();
        if let Some(r) = &self.residency {
            r.update(
                &self.bufs.iter().map(|b| b.id).chain(last.iter().copied()).collect::<Vec<_>>(),
                false,
            );
        }
        let (f, sel) = self.release;
        let ids = self
            .bufs
            .iter()
            .map(|b| b.id)
            .chain(last)
            .chain([self.ep.0, self.ep.1, self.up.map_or(std::ptr::null_mut(), |u| u.2)]);
        for id in ids.filter(|id| !id.is_null()) {
            // SAFETY: each buffer was created for this state and is released once.
            unsafe { f(id, sel) };
        }
    }
}
unsafe impl Send for MoeDev {}
unsafe impl Sync for MoeDev {}

#[repr(C)]
struct OffsetParams {
    o_off: u32,
    gpr: u32,
    gk: u32,
    _p: u32,
}
unsafe impl Send for MetalWeight {}
unsafe impl Sync for MetalWeight {}

struct Staging {
    x: Id,
    x_cap: usize,
    y: Id,
    y_cap: usize,
    /// Normalization statistics (float2 per group or row) and gamma/beta for fused norms.
    stats: Grow,
    gb: Grow,
    /// Attention operands: per-head padded Q, K, transposed V, the scores of one head, outputs.
    aq: Grow,
    ak: Grow,
    av: Grow,
    asc: Grow,
    ao: Grow,
    /// Matrix-unit GEMMs: half input rows and their inverse scales.
    xh: Grow,
    rs: Grow,
    /// MoE: gathered row sources, tiles, combine slots, intermediates.
    m_src: Grow,
    m_tiles: Grow,
    m_slots: Grow,
    m_gu: Grow,
    m_ah: Grow,
    m_rs2: Grow,
    m_o: Grow,
}

/// A device buffer that only ever grows.
struct Grow {
    id: Id,
    cap: usize,
}

impl Grow {
    const fn new() -> Self {
        Self {
            id: std::ptr::null_mut(),
            cap: 0,
        }
    }
}

#[derive(Default)]
struct Stats {
    calls: AtomicU64,
    copy_in_ns: AtomicU64,
    gpu_ns: AtomicU64,
    copy_out_ns: AtomicU64,
}

/// `GgW::rb`'s flag (the row length the GGUF kernels get in `P::s_off`) for weights in the
/// grouped GEMMs' tiled layout ([`MetalBackend::tiled_experts`]).
pub(crate) const GG_TILED: u32 = 1 << 31;

/// The parts of a tiled K-quant super-block (`kpart` in the shaders), in the order they are
/// stored, each as byte ranges of the super-block: first what every 32-block reads (scales, Q5_K's
/// high bits, Q6_K's d), then what each K step of 64 values reads (Q4_K / Q5_K: a pair of
/// 32-blocks' quant bytes; Q6_K: a half's low and high bits).
fn kquant_parts(k: GgmlType) -> Option<&'static [&'static [(u32, u32)]]> {
    Some(match k {
        GgmlType::Q4_K => &[&[(0, 16)], &[(16, 32)], &[(48, 32)], &[(80, 32)], &[(112, 32)]],
        GgmlType::Q5_K => &[&[(0, 48)], &[(48, 32)], &[(80, 32)], &[(112, 32)], &[(144, 32)]],
        GgmlType::Q6_K => &[&[(192, 18)], &[(0, 64), (128, 32)], &[(64, 64), (160, 32)]],
        _ => return None,
    })
}

/// A shared wrap (`MetalBackend::shared_wraps`): buffer, GPU address, plans using it.
type SharedWrap = (usize, u64, usize);

pub struct MetalBackend {
    stats: Stats,
    o: Objc,
    _foundation: dlsym::Library,
    device: Id,
    queue: Id,
    /// GEMM pipelines indexed by format: bf16, q8, q4.
    pipes: [Id; 3],
    /// GEMM over affine Q4 weights.
    pipe_q4a: Id,
    /// Implicit-GEMM convolution pipelines, same order.
    conv_pipes: [Id; 3],
    gemm_f32: Id,
    /// GEMM pipelines for the GGUF block formats.
    gguf_pipes: Vec<(GgmlType, Id)>,
    group_stats: Id,
    row_stats: Id,
    softmax: Id,
    /// Fused attention for head size 64 without a mask (`flash_attn64`).
    flash64: Id,
    /// The same for head size 128 (`flash_attn128`).
    flash128: Id,
    /// Matrix-unit kernels, when the device has them (and `OXINFER_METAL_MMA` allows), or why
    /// they failed to compile.
    mma: Option<Mma>,
    #[cfg_attr(not(test), allow(dead_code))]
    mma_error: Option<String>,
    /// Keeps MoE expert weights resident (see `new`).
    residency: Option<std::sync::Arc<Residency>>,
    staging: Mutex<Staging>,
    sels: Sels,
    /// The compiled shader library, and pipelines made from it on demand (decode kernels).
    lib: Id,
    named: Mutex<std::collections::HashMap<String, Id>>,
    /// Expert weight spans wrapped by decode plans, shared by the plans of one model (a model's
    /// single- and multi-token plans would otherwise each wire every expert: 78 GB twice for
    /// Qwen3.8-Flash-Next at Q4): `(start, len) -> (buffer, plans using it)`.
    pub(crate) shared_wraps: Mutex<std::collections::HashMap<(usize, usize), SharedWrap>>,
    /// The bytes of the live tiled expert copies of plans (`tiled_experts`), by buffer.
    tiled: Mutex<std::collections::HashMap<usize, usize>>,
}
unsafe impl Send for MetalBackend {}
unsafe impl Sync for MetalBackend {}

struct Sels {
    command_buffer: Sel,
    compute_encoder: Sel,
    set_pipeline: Sel,
    set_buffer: Sel,
    set_bytes: Sel,
    dispatch: Sel,
    end_encoding: Sel,
    commit: Sel,
    wait: Sel,
    contents: Sel,
    new_buffer_nocopy: Sel,
    new_buffer_bytes: Sel,
    new_buffer_len: Sel,
    status: Sel,
    release: Sel,
    gpu_address: Sel,
    use_resources: Sel,
    pool_push: unsafe extern "C" fn() -> *mut c_void,
    pool_pop: unsafe extern "C" fn(*mut c_void),
}

type Msg0 = unsafe extern "C" fn(Id, Sel) -> Id;
type Msg1 = unsafe extern "C" fn(Id, Sel, Id) -> Id;
type MsgU = unsafe extern "C" fn(Id, Sel) -> u64;
type MsgPtr = unsafe extern "C" fn(Id, Sel) -> *mut c_void;
type MsgNewLib = unsafe extern "C" fn(Id, Sel, Id, Id, *mut Id) -> Id;
type MsgNewPipe = unsafe extern "C" fn(Id, Sel, Id, *mut Id) -> Id;
type MsgNewBufNoCopy = unsafe extern "C" fn(Id, Sel, *mut c_void, usize, usize, Id) -> Id;
type MsgNewBufBytes = unsafe extern "C" fn(Id, Sel, *const c_void, usize, usize) -> Id;
type MsgNewBufLen = unsafe extern "C" fn(Id, Sel, usize, usize) -> Id;
type MsgSetBuffer = unsafe extern "C" fn(Id, Sel, Id, usize, usize);
type MsgSetBytes = unsafe extern "C" fn(Id, Sel, *const c_void, usize, usize);
type MsgSetPipe = unsafe extern "C" fn(Id, Sel, Id);
type MsgDispatch = unsafe extern "C" fn(Id, Sel, *const MtlSize, *const MtlSize);
type MsgVoid = unsafe extern "C" fn(Id, Sel);

trait IntoInnerUnsafe<T> {
    fn into_inner_unsafe(self) -> T;
}
impl<T: dlsym::FnPtr> IntoInnerUnsafe<T> for dlsym::Symbol<'_, T> {
    fn into_inner_unsafe(self) -> T {
        unsafe { self.into_inner() }
    }
}

impl MetalBackend {
    pub fn new() -> Result<Self> {
        let o = Objc::open().map_err(|e| anyhow!(e))?;
        let foundation =
            unsafe { dlsym::Library::open("/System/Library/Frameworks/Foundation.framework/Foundation") }.map_err(|e| anyhow!("{e}"))?;
        let device = o.default_device();
        if device.is_null() {
            bail!("no Metal device");
        }
        let msg0: Msg0 = o.msg_send();
        let queue = unsafe { msg0(device, o.sel(c"newCommandQueue")) };
        if queue.is_null() {
            bail!("newCommandQueue failed");
        }
        let get_class: unsafe extern "C" fn(*const std::ffi::c_char) -> Id = unsafe { o.objc.get(c"objc_getClass") }
            .map_err(|e| anyhow!("{e}"))?
            .into_inner_unsafe();
        let nsstring_cls = unsafe { get_class(c"NSString".as_ptr()) };
        let msg_str: unsafe extern "C" fn(Id, Sel, *const std::ffi::c_char) -> Id = o.msg_send();
        let error_text = |err: Id| -> String {
            if err.is_null() {
                return String::new();
            }
            // SAFETY: `localizedDescription` returns an NSString owned by the NSError.
            unsafe { o.nsstring_to_string(msg0(err, o.sel(c"localizedDescription"))) }
        };
        // Compile the shader library from source: with the matrix-unit kernels (Metal 4) when the
        // GPU cores have neural accelerators (Apple10 family: M5, A19), else without.
        // OXINFER_METAL_MMA=0 leaves them out, =1 tries them on any device; `gpu::set_exact`
        // bypasses them at run time.
        let supports_family: unsafe extern "C" fn(Id, Sel, i64) -> u8 = o.msg_send();
        let apple10 = unsafe { supports_family(device, o.sel(c"supportsFamily:"), 1010) } != 0;
        let mma_env = std::env::var("OXINFER_METAL_MMA").ok();
        let want_mma = match mma_env.as_deref() {
            Some("0") => false,
            Some("1") => true,
            _ => apple10,
        };
        let base_src = format!(
            "{SHADER_SRC}{}{GGUF_SHADER_SRC}{}{}",
            gguf_shader_tables(),
            decode::shader_src(),
            dit::SRC
        ) + dit::LTX_SRC;
        let compile = |src: &str, metal4: bool| -> Result<Id> {
            let src = CString::new(src)?;
            let ns_src = unsafe { msg_str(nsstring_cls, o.sel(c"stringWithUTF8String:"), src.as_ptr()) };
            let mut opts: Id = std::ptr::null_mut();
            if metal4 {
                let cls = unsafe { get_class(c"MTLCompileOptions".as_ptr()) };
                if cls.is_null() {
                    bail!("no MTLCompileOptions");
                }
                opts = unsafe { msg0(msg0(cls, o.sel(c"alloc")), o.sel(c"init")) };
                let set_ver: unsafe extern "C" fn(Id, Sel, u64) = o.msg_send();
                unsafe { set_ver(opts, o.sel(c"setLanguageVersion:"), 0x4_0000) }; // MTLLanguageVersion4_0
            }
            let new_lib: MsgNewLib = o.msg_send();
            let mut err: Id = std::ptr::null_mut();
            let lib = unsafe { new_lib(device, o.sel(c"newLibraryWithSource:options:error:"), ns_src, opts, &mut err) };
            if !opts.is_null() {
                let void: MsgVoid = o.msg_send();
                unsafe { void(opts, o.sel(c"release")) };
            }
            if lib.is_null() {
                bail!("Metal shader compilation failed: {}", error_text(err));
            }
            Ok(lib)
        };
        let t_compile = Instant::now();
        // Without matrix units the same GEMMs (half operands, f32 accumulation) on SIMD-group
        // matrices; OXINFER_METAL_MMA=0 keeps neither.
        let extra = || {
            // Kernel experiments (`mma_gemm_bench`): more source for the library.
            std::env::var("OXINFER_MMA_EXTRA")
                .ok()
                .and_then(|f| std::fs::read_to_string(f).ok())
                .unwrap_or_default()
        };
        // The first lines of a compiler log (for a warning; the whole log stays in `mma_error`).
        let brief = |e: &str| -> String {
            // Its errors (the log starts with warnings), else its first lines.
            let errors: Vec<&str> = e.lines().filter(|l| l.contains("error")).take(3).collect();
            let lines = if errors.is_empty() {
                e.lines().filter(|l| !l.trim().is_empty()).take(4).collect()
            } else {
                errors
            };
            let mut b = lines.join(" / ");
            if b.len() > 600 {
                b.truncate(b.floor_char_boundary(600));
                b.push_str("...");
            }
            b
        };
        let sg_src = || format!("#define GTM {MMA_GTM}\n#define MMA_UNITS 0\n{base_src}{MMA_SHADER_SRC}{}", extra());
        let sg_lib = |e: Option<String>| -> Result<(Id, bool, bool, Option<String>)> {
            match compile(&sg_src(), false) {
                Ok(l) => Ok((l, true, false, e)),
                Err(e2) => {
                    crate::gpu::warn(format!(
                        "metal: the SIMD-group half GEMM kernels failed to compile, so the GPU runs its f32 kernels only: {}",
                        brief(&e2.to_string())
                    ));
                    Ok((compile(&base_src, false)?, false, false, Some(e.unwrap_or_else(|| e2.to_string()))))
                }
            }
        };
        let (lib, has_mma, units, mma_error) = match want_mma {
            true => match compile(
                &format!(
                    "#define GTM {MMA_GTM}\n#define MMA_UNITS 1\n{base_src}{MMA_SHADER_SRC}{}{}{}{}{}{}",
                    decode::PFM_SRC,
                    unet::SRC,
                    whisper::SRC,
                    dit::MMA_SRC,
                    extra(),
                    // OXINFER_METAL_MMA_BREAK=1: a compile error in that library, to see the
                    // degraded mode (`gpu::warnings`) without a broken kernel.
                    if std::env::var_os("OXINFER_METAL_MMA_BREAK").is_some() {
                        "\n#error OXINFER_METAL_MMA_BREAK\n"
                    } else {
                        ""
                    }
                ),
                true,
            ) {
                Ok(l) => (l, true, true, None),
                Err(e) => {
                    if apple10 || mma_env.is_some() {
                        // The GPU has matrix units but they go unused: everything (GEMMs,
                        // attention, the whole-pass image / audio plans) runs 2-3x slower.
                        crate::gpu::warn(format!(
                            "metal: the matrix-unit kernels failed to compile, so this GPU runs the slower \
                             SIMD-group kernels (2-3x slower GEMMs, no whole-pass image plans): {}",
                            brief(&e.to_string())
                        ));
                    } else if std::env::var_os("OXINFER_DEBUG").is_some() {
                        eprintln!("[metal: matrix-unit kernels unavailable: {e}]");
                    }
                    sg_lib(Some(e.to_string()))?
                }
            },
            false if mma_env.as_deref() == Some("0") => (compile(&base_src, false)?, false, false, None),
            false => sg_lib(None)?,
        };
        let pipeline = |name: &CStr| -> Result<Id> {
            let ns_name = unsafe { msg_str(nsstring_cls, o.sel(c"stringWithUTF8String:"), name.as_ptr()) };
            let msg_fn: Msg1 = o.msg_send();
            let f = unsafe { msg_fn(lib, o.sel(c"newFunctionWithName:"), ns_name) };
            if f.is_null() {
                bail!("kernel {name:?} not found");
            }
            let new_pipe: MsgNewPipe = o.msg_send();
            let mut err: Id = std::ptr::null_mut();
            let p = unsafe { new_pipe(device, o.sel(c"newComputePipelineStateWithFunction:error:"), f, &mut err) };
            if p.is_null() {
                bail!("pipeline for {name:?} failed: {}", error_text(err));
            }
            Ok(p)
        };
        let pipes = [pipeline(c"gemm_bf")?, pipeline(c"gemm_q8")?, pipeline(c"gemm_q4")?];
        let pipe_q4a = pipeline(c"gemm_q4a")?;
        let conv_pipes = [pipeline(c"conv_bf")?, pipeline(c"conv_q8")?, pipeline(c"conv_q4")?];
        let gemm_f32 = pipeline(c"gemm_f32")?;
        let gguf_pipes = GGUF_GPU_TYPES
            .iter()
            .chain(&[GgmlType::F8E4M3, GgmlType::MXFP4, GgmlType::NVFP4])
            .map(|&k| Ok((k, pipeline(&CString::new(format!("gemm_{}", k.name()))?)?)))
            .collect::<Result<Vec<_>>>()?;
        let group_stats = pipeline(c"group_stats")?;
        let row_stats = pipeline(c"row_stats")?;
        let softmax = pipeline(c"softmax_rows")?;
        let flash64 = pipeline(c"flash_attn64")?;
        let flash128 = pipeline(c"flash_attn128")?;
        let mma = if has_mma {
            let gg = |prefix: &str| {
                GGUF_MMA_TYPES
                    .iter()
                    .map(|&k| Ok((k, pipeline(&CString::new(format!("{prefix}_{}", k.name()))?)?)))
                    .collect::<Result<Vec<_>>>()
            };
            let (flash128, flash128_f32) = match units {
                true => (pipeline(c"flash_attn128_mma")?, pipeline(c"flash_attn128_mma_f32")?),
                false => (flash128, flash128),
            };
            Some(Mma {
                units,
                x_half: pipeline(c"x_half")?,
                flash128,
                flash128_f32,
                silu_half: pipeline(c"silu_half")?,
                combine: pipeline(c"moe_combine")?,
                bf: [pipeline(c"mma_bf64")?, pipeline(c"mma_bf128")?],
                q8: pipeline(c"mma_q8")?,
                q4: pipeline(c"mma_q4")?,
                q4a: pipeline(c"mma_q4a")?,
                gguf: gg("mma")?,
                bf_s: [pipeline(c"mma_bf64_s")?, pipeline(c"mma_bf128_s")?],
                q8_s: pipeline(c"mma_q8_s")?,
                q4_s: pipeline(c"mma_q4_s")?,
                q4a_s: pipeline(c"mma_q4a_s")?,
                gguf_s: GGUF_MMA_TYPES
                    .iter()
                    .map(|&k| Ok((k, pipeline(&CString::new(format!("mma_{}_s", k.name()))?)?)))
                    .collect::<Result<Vec<_>>>()?,
                g_bf: pipeline(c"mmag_bf")?,
                g_q8: pipeline(c"mmag_q8")?,
                g_q4: pipeline(c"mmag_q4")?,
                g_gguf: gg("mmag")?,
            })
        } else {
            None
        };
        let pool_push = unsafe { o.objc.get::<unsafe extern "C" fn() -> *mut c_void>(c"objc_autoreleasePoolPush") }
            .map_err(|e| anyhow!("{e}"))?
            .into_inner_unsafe();
        let pool_pop = unsafe { o.objc.get::<unsafe extern "C" fn(*mut c_void)>(c"objc_autoreleasePoolPop") }
            .map_err(|e| anyhow!("{e}"))?
            .into_inner_unsafe();
        let sels = Sels {
            command_buffer: o.sel(c"commandBuffer"),
            compute_encoder: o.sel(c"computeCommandEncoder"),
            set_pipeline: o.sel(c"setComputePipelineState:"),
            set_buffer: o.sel(c"setBuffer:offset:atIndex:"),
            set_bytes: o.sel(c"setBytes:length:atIndex:"),
            dispatch: o.sel(c"dispatchThreadgroups:threadsPerThreadgroup:"),
            end_encoding: o.sel(c"endEncoding"),
            commit: o.sel(c"commit"),
            wait: o.sel(c"waitUntilCompleted"),
            contents: o.sel(c"contents"),
            new_buffer_nocopy: o.sel(c"newBufferWithBytesNoCopy:length:options:deallocator:"),
            new_buffer_bytes: o.sel(c"newBufferWithBytes:length:options:"),
            new_buffer_len: o.sel(c"newBufferWithLength:options:"),
            status: o.sel(c"status"),
            release: o.sel(c"release"),
            gpu_address: o.sel(c"gpuAddress"),
            use_resources: o.sel(c"useResources:count:usage:"),
            pool_push,
            pool_pop,
        };
        // A residency set (macOS 15 / iOS 18) attached to the queue keeps the MoE expert weights
        // resident: reached through addresses, they would otherwise have to be made resident
        // (`useResources`) on every submission, which costs more than the GEMMs for mapped files.
        let residency = (|| {
            mma.as_ref()?;
            let responds: unsafe extern "C" fn(Id, Sel, Sel) -> u8 = o.msg_send();
            let new_sel = o.sel(c"newResidencySetWithDescriptor:error:");
            if unsafe { responds(device, o.sel(c"respondsToSelector:"), new_sel) } == 0 {
                return None;
            }
            let cls = unsafe { get_class(c"MTLResidencySetDescriptor".as_ptr()) };
            if cls.is_null() {
                return None;
            }
            let desc = unsafe { msg0(msg0(cls, o.sel(c"alloc")), o.sel(c"init")) };
            let new_set: unsafe extern "C" fn(Id, Sel, Id, *mut Id) -> Id = o.msg_send();
            let add: unsafe extern "C" fn(Id, Sel, Id) = o.msg_send();
            let mut sets = Vec::new();
            for _ in 0..RESIDENCY_SETS {
                let mut err: Id = std::ptr::null_mut();
                let set = unsafe { new_set(device, new_sel, desc, &mut err) };
                if set.is_null() {
                    break;
                }
                unsafe { add(queue, o.sel(c"addResidencySet:"), set) };
                sets.push(set);
            }
            let void: MsgVoid = o.msg_send();
            unsafe { void(desc, o.sel(c"release")) };
            if sets.is_empty() {
                return None;
            }
            Some(std::sync::Arc::new(Residency {
                sets,
                add: o.sel(c"addAllocation:"),
                remove: o.sel(c"removeAllocation:"),
                commit: o.sel(c"commit"),
                request: o.sel(c"requestResidency"),
                send_id: add,
                send: void,
                lock: Mutex::new(()),
            }))
        })();
        if std::env::var_os("OXINFER_METAL_TRACE").is_some() {
            eprintln!(
                "[metal: library and pipelines ready in {:.0} ms (matrix units: {units}, half GEMMs: {has_mma})]",
                t_compile.elapsed().as_secs_f64() * 1e3
            );
        }
        Ok(Self {
            residency,
            stats: Stats::default(),
            o,
            _foundation: foundation,
            device,
            queue,
            pipes,
            pipe_q4a,
            conv_pipes,
            gemm_f32,
            gguf_pipes,
            group_stats,
            row_stats,
            softmax,
            flash64,
            flash128,
            mma,
            mma_error,
            staging: Mutex::new(Staging {
                x: std::ptr::null_mut(),
                x_cap: 0,
                y: std::ptr::null_mut(),
                y_cap: 0,
                stats: Grow::new(),
                gb: Grow::new(),
                aq: Grow::new(),
                ak: Grow::new(),
                av: Grow::new(),
                asc: Grow::new(),
                ao: Grow::new(),
                xh: Grow::new(),
                rs: Grow::new(),
                m_src: Grow::new(),
                m_tiles: Grow::new(),
                m_slots: Grow::new(),
                m_gu: Grow::new(),
                m_ah: Grow::new(),
                m_rs2: Grow::new(),
                m_o: Grow::new(),
            }),
            sels,
            lib,
            named: Mutex::new(std::collections::HashMap::new()),
            shared_wraps: Mutex::new(std::collections::HashMap::new()),
            tiled: Mutex::new(std::collections::HashMap::new()),
        })
    }

    /// The pipeline of kernel `name` in the library (made once, then cached).
    pub(crate) fn named_pipeline(&self, name: &str) -> Option<Id> {
        let mut m = self.named.lock().unwrap();
        if let Some(&p) = m.get(name) {
            return Some(p);
        }
        let msg_str: unsafe extern "C" fn(Id, Sel, *const std::ffi::c_char) -> Id = self.o.msg_send();
        let get_class: unsafe extern "C" fn(*const std::ffi::c_char) -> Id =
            unsafe { self.o.objc.get(c"objc_getClass") }.ok()?.into_inner_unsafe();
        let cname = CString::new(name).ok()?;
        let ns_name = unsafe {
            msg_str(
                get_class(c"NSString".as_ptr()),
                self.o.sel(c"stringWithUTF8String:"),
                cname.as_ptr(),
            )
        };
        let msg_fn: Msg1 = self.o.msg_send();
        let f = unsafe { msg_fn(self.lib, self.o.sel(c"newFunctionWithName:"), ns_name) };
        if f.is_null() {
            return None;
        }
        let new_pipe: MsgNewPipe = self.o.msg_send();
        let mut err: Id = std::ptr::null_mut();
        let p = unsafe { new_pipe(self.device, self.o.sel(c"newComputePipelineStateWithFunction:error:"), f, &mut err) };
        let void: MsgVoid = self.o.msg_send();
        unsafe { void(f, self.sels.release) };
        if p.is_null() {
            let why = if err.is_null() {
                "no error given".to_string()
            } else {
                let msg0: Msg0 = self.o.msg_send();
                // SAFETY: `localizedDescription` returns an NSString owned by the NSError.
                unsafe { self.o.nsstring_to_string(msg0(err, self.o.sel(c"localizedDescription"))) }
            };
            // The kernel exists but cannot run: whatever needed it takes a slower path.
            crate::gpu::warn(format!(
                "metal: the pipeline of kernel {name} failed, its users fall back to slower paths: {why}"
            ));
            return None;
        }
        m.insert(name.to_string(), p);
        Some(p)
    }

    /// Wraps host memory as a shared buffer without copying (the enclosing pages are mapped; the
    /// caller guarantees they belong to one allocation); falls back to copying. The iOS
    /// Simulator's Metal driver aborts on wrapped host memory (it hands buffers to the host's GPU
    /// process as shared memory), so it always copies there.
    /// [`Self::wrap`] shared between decode plans (see `shared_wraps`); give it back with
    /// [`Self::unwrap_shared`].
    pub(crate) fn wrap_shared(&self, ptr: *const u8, len: usize) -> Option<(Id, u64)> {
        let mut m = self.shared_wraps.lock().unwrap();
        if let Some(e) = m.get_mut(&(ptr as usize, len)) {
            e.2 += 1;
            return Some((e.0 as Id, e.1));
        }
        let b = self.wrap(ptr, len)?;
        let addr = self.gpu_address(&b);
        m.insert((ptr as usize, len), (b.id as usize, addr, 1));
        Some((b.id, addr))
    }

    /// Drops a plan's use of a shared wrap; true when it was the last (the caller releases it).
    pub(crate) fn unwrap_shared(&self, id: Id) -> bool {
        let mut m = self.shared_wraps.lock().unwrap();
        let Some(k) = m.iter().find(|e| e.1.0 == id as usize).map(|e| *e.0) else {
            return true;
        };
        let e = m.get_mut(&k).unwrap();
        e.2 -= 1;
        if e.2 == 0 {
            m.remove(&k);
            self.tiled.lock().unwrap().remove(&(id as usize));
            return true;
        }
        false
    }

    /// The routed experts `ws` (GGUF matrices of one block format, `n_out` x `n_in` each) copied
    /// into one private buffer in the grouped GEMMs' tiled layout (`gblk` in the shaders: the
    /// kernels see it through [`GG_TILED`] in their row length), expert e at e times the bytes of
    /// a matrix, by the GPU from the addresses in `src` (rows of `stride` u64s, the data address
    /// first) in the buffers `uses`. `shared`: kept in `shared_wraps` for the other plans of the
    /// same model (each gives it back with `unwrap_shared`), else the caller's (forced copies
    /// only). `mode`: as [`crate::gpu::expert_tiles`]. None when the mode, the format or the
    /// shape rule it out, or (automatic) where the OS limits the app's memory or the live copies
    /// would pass a quarter of the machine's memory (copies that do not take the experts' place),
    /// or, with a null `src`, when no shared copy exists yet ([`Self::tiled_shared`]). Returns the
    /// buffer, its GPU address and the bytes of one expert.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn tiled_experts(
        &self,
        ws: &[WeightRef],
        n_out: usize,
        n_in: usize,
        src: Id,
        stride: usize,
        uses: &[Id],
        shared: bool,
        mode: Option<bool>,
    ) -> Option<(Id, u64, usize)> {
        let forced = match mode {
            Some(false) => return None,
            Some(true) => true,
            None => false,
        };
        if !forced && !self.mma().is_some_and(|m| m.units) {
            return None;
        }
        let ok = |w: &WeightRef, k0: Option<GgmlType>| match w {
            WeightRef::Gguf(g) => {
                g.n_out == n_out
                    && g.n_in == n_in
                    && g.block_scales.is_none()
                    && !g.x4
                    && k0.is_none_or(|k| k == g.kind)
                    && g.kind != GgmlType::F8E4M3
                    && GGUF_MMA_TYPES.contains(&g.kind)
                    && n_in.is_multiple_of(g.kind.block_size().max(32))
            }
            _ => false,
        };
        let WeightRef::Gguf(g0) = ws.first()? else { return None };
        if !n_out.is_multiple_of(64) || !ws.iter().all(|w| ok(w, Some(g0.kind))) {
            return None;
        }
        // The blocks the dequantizers address (`gblk`): the format's, or for F16 32 values.
        let (be, bs) = match g0.kind {
            GgmlType::F16 => (32, 64),
            k => (k.block_size(), k.type_size()),
        };
        let (nb, mb) = (n_in / be, g0.row_bytes() * n_out);
        let total = mb * ws.len();
        let key = (g0.data.as_ptr() as usize, total | 1 << 63);
        // (Plans build concurrently: one copy at a time, so that each is made once.)
        static TILING: Mutex<()> = Mutex::new(());
        let _one = TILING.lock().unwrap_or_else(|e| e.into_inner());
        if shared && let Some(e) = self.shared_wraps.lock().unwrap().get_mut(&key) {
            e.2 += 1;
            return Some((e.0 as Id, e.1, mb));
        }
        if src.is_null() {
            return None;
        }
        // Experts in a checkpoint file's memory ([`crate::safetensors::releasable`]: mapped, or
        // GGUF read into memory): the decode plans give their memory back once they have their
        // copies (`expert_tables`), so the copies take its place. Others stay beside their copies.
        let free = ws
            .iter()
            .all(|w| matches!(w, WeightRef::Gguf(g) if crate::safetensors::releasable(g.data.as_ptr() as usize, g.data.len())));
        // Automatic: not where the OS limits the app's memory (iOS), and, but for copies that
        // take the experts' place, all the live copies within a quarter of the machine's memory.
        if !forced {
            let live: usize = self.tiled.lock().unwrap().values().sum();
            let cap = crate::platform::total_memory().unwrap_or(0) as usize / 4;
            if !shared || crate::platform::process_available_memory().is_some() || (!free && live + total > cap) {
                return None;
            }
        }
        let pipe = self.named_pipeline("tile_blocks")?;
        let new_len: MsgNewBufLen = self.o.msg_send();
        // MTLResourceStorageModePrivate
        let id = unsafe { new_len(self.device, self.sels.new_buffer_len, total, 2 << 4) };
        if id.is_null() {
            return None;
        }
        // The block's bytes as segments (`tile_blocks`): one, or a K-quant's parts in turn.
        let whole: &[&[(u32, u32)]] = &[&[(0, bs as u32)]];
        let mut seg = [[0u32; 4]; 8];
        let mut nseg = 0;
        let mut off = 0;
        for part in kquant_parts(g0.kind).unwrap_or(whole) {
            let len: u32 = part.iter().map(|r| r.1).sum();
            let mut pos = 0;
            for &(src, n) in part.iter() {
                seg[nseg] = [src, n, 64 * off + pos, len];
                nseg += 1;
                pos += n;
            }
            off += len;
        }
        debug_assert_eq!(off as usize, bs);
        #[repr(C)]
        struct Tb {
            rows: u32,
            nb: u32,
            bs: u32,
            stride: u32,
            mb: u64,
            nseg: u32,
            even: u32,
            seg: [[u32; 4]; 8],
        }
        let p = Tb {
            rows: n_out as u32,
            nb: nb as u32,
            bs: bs as u32,
            stride: stride as u32,
            mb: mb as u64,
            nseg: nseg as u32,
            even: seg[..nseg].iter().flatten().all(|v| v % 2 == 0) as u32,
            seg,
        };
        let blocks = n_out * nb;
        let mut res: Vec<Id> = uses.to_vec();
        res.push(id);
        let done = self.submit(
            Instant::now(),
            &mut (),
            |e: &Enc, _: &()| {
                e.use_resources(&res);
                e.pipe(pipe);
                e.buf(src, 0, 0);
                e.buf(id, 0, 1);
                e.bytes(&p, 2);
                e.dispatch((blocks.div_ceil(256), ws.len()), 256);
            },
            |_| {},
        );
        if !done {
            let void: MsgVoid = self.o.msg_send();
            unsafe { void(id, self.sels.release) };
            return None;
        }
        let addr = self.buffer_gpu_address(id);
        if shared {
            self.shared_wraps.lock().unwrap().insert(key, (id as usize, addr, 1));
            if !free {
                self.tiled.lock().unwrap().insert(id as usize, total);
            }
        }
        Some((id, addr, mb))
    }

    /// The shared tiled copy of the experts `ws` ([`Self::tiled_experts`]) if another plan made
    /// it already (the caller gives it back with `unwrap_shared`).
    pub(crate) fn tiled_shared(&self, ws: &[WeightRef], n_out: usize, n_in: usize, mode: Option<bool>) -> Option<(Id, u64, usize)> {
        self.tiled_experts(ws, n_out, n_in, std::ptr::null_mut(), 0, &[], true, mode)
    }

    /// [`Self::wrap`] for a caller that releases the buffer itself: (buffer, offset of `ptr`).
    pub(crate) fn wrap_raw(&self, ptr: *const u8, len: usize) -> Option<(Id, usize)> {
        self.wrap(ptr, len).map(|b| (b.id, b.offset_bytes))
    }

    fn wrap(&self, ptr: *const u8, len: usize) -> Option<Buf> {
        let start = (ptr as usize) & !(PAGE - 1);
        let end = (ptr as usize + len).div_ceil(PAGE) * PAGE;
        let nocopy: MsgNewBufNoCopy = self.o.msg_send();
        let id = if cfg!(target_abi = "sim") {
            std::ptr::null_mut()
        } else {
            unsafe {
                nocopy(
                    self.device,
                    self.sels.new_buffer_nocopy,
                    start as *mut c_void,
                    end - start,
                    0,
                    std::ptr::null_mut(),
                )
            }
        };
        if !id.is_null() {
            return Some(Buf {
                id,
                offset_bytes: ptr as usize - start,
            });
        }
        let copy: MsgNewBufBytes = self.o.msg_send();
        let id = unsafe { copy(self.device, self.sels.new_buffer_bytes, ptr as *const c_void, len, 0) };
        if id.is_null() {
            return None;
        }
        Some(Buf { id, offset_bytes: 0 })
    }

    fn gguf_pipe(&self, kind: GgmlType) -> Option<Id> {
        self.gguf_pipes.iter().find(|(k, _)| *k == kind).map(|(_, p)| *p)
    }

    /// Makes `g` at least `bytes` long (replaced buffers are released).
    fn grow(&self, g: &mut Grow, bytes: usize) -> bool {
        if g.cap >= bytes && !g.id.is_null() {
            return true;
        }
        let new_len: MsgNewBufLen = self.o.msg_send();
        let void: MsgVoid = self.o.msg_send();
        let cap = bytes.max(16).div_ceil(PAGE) * PAGE * 2;
        let id = unsafe { new_len(self.device, self.sels.new_buffer_len, cap, 0) };
        if id.is_null() {
            return false;
        }
        if !g.id.is_null() {
            unsafe { void(g.id, self.sels.release) };
        }
        g.id = id;
        g.cap = cap;
        true
    }

    fn ensure_staging(&self, st: &mut Staging, x_bytes: usize, y_bytes: usize) -> bool {
        let new_len: MsgNewBufLen = self.o.msg_send();
        if st.x_cap < x_bytes {
            let cap = x_bytes.div_ceil(PAGE) * PAGE * 2;
            st.x = unsafe { new_len(self.device, self.sels.new_buffer_len, cap, 0) };
            st.x_cap = if st.x.is_null() { 0 } else { cap };
        }
        if st.y_cap < y_bytes {
            let cap = y_bytes.div_ceil(PAGE) * PAGE * 2;
            st.y = unsafe { new_len(self.device, self.sels.new_buffer_len, cap, 0) };
            st.y_cap = if st.y.is_null() { 0 } else { cap };
        }
        !st.x.is_null() && !st.y.is_null()
    }
}

/// Copies `n` floats, on the worker pool when large (staging copies of prefill activations run
/// at memory bandwidth rather than one core's).
///
/// # Safety
/// `src` and `dst` must be valid for `n` floats and not overlap.
/// `src[..n]` as half into `dst`, zero-filled up to `total` values.
///
/// # Safety
/// `dst` must hold `total` values and `src` `n`.
unsafe fn par_half(dst: *mut u16, src: *const f32, n: usize, total: usize) {
    const CHUNK: usize = 1 << 16;
    let d = unsafe { std::slice::from_raw_parts_mut(dst, total) };
    let s = unsafe { std::slice::from_raw_parts(src, n) };
    let chunks = total.div_ceil(CHUNK);
    let dp = d.as_mut_ptr() as usize;
    crate::pool::run(chunks, &|i| {
        let (a, b) = (i * CHUNK, ((i + 1) * CHUNK).min(total));
        // SAFETY: chunks are disjoint.
        let o = unsafe { std::slice::from_raw_parts_mut((dp as *mut u16).add(a), b - a) };
        for (j, x) in o.iter_mut().enumerate() {
            let k = a + j;
            *x = if k < n { half::f16::from_f32(s[k]).to_bits() } else { 0 };
        }
    });
}

unsafe fn par_copy(dst: *mut f32, src: *const f32, n: usize) {
    const CHUNK: usize = 1 << 16;
    if n < 8 * CHUNK {
        unsafe { std::ptr::copy_nonoverlapping(src, dst, n) };
        return;
    }
    let d = unsafe { std::slice::from_raw_parts_mut(dst, n) };
    let s = src as usize;
    crate::kernels::for_chunks(d, CHUNK, true, |i, c| {
        // SAFETY: chunk `i` of the source is as long as `c`.
        unsafe { std::ptr::copy_nonoverlapping((s as *const f32).add(i * CHUNK), c.as_mut_ptr(), c.len()) };
    });
}

/// A compute command encoder being filled (thin wrappers over the `objc_msgSend` calls).
struct Enc<'a> {
    b: &'a MetalBackend,
    enc: Id,
}

impl Enc<'_> {
    fn pipe(&self, p: Id) {
        let f: MsgSetPipe = self.b.o.msg_send();
        unsafe { f(self.enc, self.b.sels.set_pipeline, p) };
    }
    fn buf(&self, b: Id, offset: usize, index: usize) {
        let f: MsgSetBuffer = self.b.o.msg_send();
        unsafe { f(self.enc, self.b.sels.set_buffer, b, offset, index) };
    }
    fn bytes<T>(&self, v: &T, index: usize) {
        let f: MsgSetBytes = self.b.o.msg_send();
        unsafe {
            f(
                self.enc,
                self.b.sels.set_bytes,
                v as *const T as *const c_void,
                std::mem::size_of::<T>(),
                index,
            )
        };
    }
    fn dispatch(&self, groups: (usize, usize), threads: usize) {
        let f: MsgDispatch = self.b.o.msg_send();
        let g = MtlSize {
            w: groups.0,
            h: groups.1,
            d: 1,
        };
        let t = MtlSize { w: threads, h: 1, d: 1 };
        unsafe { f(self.enc, self.b.sels.dispatch, &g, &t) };
    }
    /// Makes buffers that are only reached through addresses stored in other buffers resident.
    fn use_resources(&self, ids: &[Id]) {
        if ids.is_empty() {
            return;
        }
        let f: unsafe extern "C" fn(Id, Sel, *const Id, usize, usize) = self.b.o.msg_send();
        unsafe { f(self.enc, self.b.sels.use_resources, ids.as_ptr(), ids.len(), 1) }; // MTLResourceUsageRead
    }
}

impl MetalBackend {
    /// Unmasked attention with 64- or 128-dimensional heads in one fused kernel (`flash_attn64` /
    /// `flash_attn128`): the operands keep the model's `[rows][heads * dh]` layout, so staging is
    /// three plain copies.
    #[allow(clippy::too_many_arguments)]
    fn flash_attention(&self, q: &[f32], k: &[f32], v: &[f32], tq: usize, tk: usize, heads: usize, dh: usize, out: &mut [f32]) -> bool {
        let d = heads * dh;
        let tqp = tq.next_multiple_of(32);
        let mut st = self.staging.lock().unwrap();
        let st = &mut *st;
        let ok = self.grow(&mut st.aq, tqp * d * 4)
            && self.grow(&mut st.ak, tk * d * 4)
            && self.grow(&mut st.av, tk * d * 4)
            && self.grow(&mut st.ao, tqp * d * 4);
        if !ok {
            return false;
        }
        let t0 = Instant::now();
        let (aq, ak, av, ao) = (st.aq.id, st.ak.id, st.av.id, st.ao.id);
        // SAFETY: the staging buffers hold at least the sizes grown above.
        unsafe {
            let qp = self.contents(aq) as *mut f32;
            par_copy(qp, q.as_ptr(), tq * d);
            std::ptr::write_bytes(qp.add(tq * d), 0, (tqp - tq) * d);
            par_copy(self.contents(ak) as *mut f32, k.as_ptr(), tk * d);
            par_copy(self.contents(av) as *mut f32, v.as_ptr(), tk * d);
        }
        #[repr(C)]
        struct Fa {
            tq: u32,
            tk: u32,
            d: u32,
            scale: f32,
        }
        let p = Fa {
            tq: tq as u32,
            tk: tk as u32,
            d: d as u32,
            scale: 1.0 / (dh as f32).sqrt(),
        };
        let pipe = if dh == 128 { self.flash128 } else { self.flash64 };
        let mut out = out;
        self.submit(
            t0,
            &mut out,
            |e, _| {
                e.pipe(pipe);
                e.buf(aq, 0, 0);
                e.buf(ak, 0, 1);
                e.buf(av, 0, 2);
                e.buf(ao, 0, 3);
                e.bytes(&p, 4);
                e.dispatch((tqp / 32, heads), 128);
            },
            // SAFETY: `ao` holds tqp >= tq rows of d floats.
            |out| unsafe { par_copy(out.as_mut_ptr(), self.contents(ao) as *const f32, tq * d) },
        )
    }

    /// [`MetalBackend::flash_attention`] for 128-wide heads on the matrix units: q, k and v are
    /// staged as half (`flash_attn128_mma`) when `half`, else as f32 (`flash_attn128_mma_f32`),
    /// rows padded to 64 with zeros.
    #[allow(clippy::too_many_arguments)]
    fn flash_attention_mma(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        tq: usize,
        tk: usize,
        heads: usize,
        half: bool,
        out: &mut [f32],
    ) -> bool {
        let Some((f_half, f_f32)) = self.mma().and_then(|m| m.flash128()) else {
            return false;
        };
        let d = heads * 128;
        let (tqp, tkp) = (tq.next_multiple_of(64), tk.next_multiple_of(64));
        let w = if half { 2 } else { 4 };
        let mut st = self.staging.lock().unwrap();
        let st = &mut *st;
        let ok = self.grow(&mut st.aq, tqp * d * w)
            && self.grow(&mut st.ak, tkp * d * w)
            && self.grow(&mut st.av, tkp * d * w)
            && self.grow(&mut st.ao, tq * d * 4);
        if !ok {
            return false;
        }
        let t0 = Instant::now();
        let (aq, ak, av, ao) = (st.aq.id, st.ak.id, st.av.id, st.ao.id);
        // SAFETY: the staging buffers hold at least the sizes grown above.
        unsafe {
            for (buf, x, n, total) in [(aq, q, tq * d, tqp * d), (ak, k, tk * d, tkp * d), (av, v, tk * d, tkp * d)] {
                if half {
                    par_half(self.contents(buf) as *mut u16, x.as_ptr(), n, total);
                } else {
                    let p = self.contents(buf) as *mut f32;
                    par_copy(p, x.as_ptr(), n);
                    std::ptr::write_bytes(p.add(n), 0, total - n);
                }
            }
        }
        #[repr(C)]
        struct Fa {
            tq: u32,
            tk: u32,
            d: u32,
            scale: f32,
        }
        let p = Fa {
            tq: tq as u32,
            tk: tk as u32,
            d: d as u32,
            scale: 1.0 / (128f32).sqrt(),
        };
        let pipe = if half { f_half } else { f_f32 };
        let mut out = out;
        self.submit(
            t0,
            &mut out,
            |e, _| {
                e.pipe(pipe);
                e.buf(aq, 0, 0);
                e.buf(ak, 0, 1);
                e.buf(av, 0, 2);
                e.buf(ao, 0, 3);
                e.bytes(&p, 4);
                e.dispatch((tqp / 64, heads), 128);
            },
            // SAFETY: `ao` holds tq rows of d floats.
            |out| unsafe { par_copy(out.as_mut_ptr(), self.contents(ao) as *const f32, tq * d) },
        )
    }

    /// The matrix-unit kernels, unless absent or bypassed by [`crate::gpu::set_exact`].
    fn mma(&self) -> Option<&Mma> {
        if crate::gpu::exact() { None } else { self.mma.as_ref() }
    }

    /// Encodes one compute pass with `encode`, submits it and waits; then, if it completed, calls
    /// `read_back`. Both get `state` (what they both borrow). `t_in` is when copying the inputs
    /// started (for the statistics).
    fn submit<S>(&self, t_in: Instant, state: &mut S, encode: impl FnOnce(&Enc, &S), read_back: impl FnOnce(&mut S)) -> bool {
        let msg0: Msg0 = self.o.msg_send();
        let void: MsgVoid = self.o.msg_send();
        let status: MsgU = self.o.msg_send();
        unsafe {
            let t1 = Instant::now();
            let pool = (self.sels.pool_push)();
            let cb = msg0(self.queue, self.sels.command_buffer);
            let enc = msg0(cb, self.sels.compute_encoder);
            encode(&Enc { b: self, enc }, state);
            void(enc, self.sels.end_encoding);
            void(cb, self.sels.commit);
            void(cb, self.sels.wait);
            let ok = status(cb, self.sels.status) == 4; // MTLCommandBufferStatusCompleted
            let t2 = Instant::now();
            if std::env::var_os("OXINFER_METAL_TRACE").is_some() {
                let time: unsafe extern "C" fn(Id, Sel) -> f64 = self.o.msg_send();
                let g = time(cb, self.o.sel(c"GPUEndTime")) - time(cb, self.o.sel(c"GPUStartTime"));
                eprintln!(
                    "[metal: copy-in {:.3} ms, gpu {:.3} ms, wall {:.3} ms]",
                    (t1 - t_in).as_secs_f64() * 1e3,
                    g * 1e3,
                    (t2 - t1).as_secs_f64() * 1e3
                );
            }
            if ok {
                read_back(state);
            }
            (self.sels.pool_pop)(pool);
            self.stats.calls.fetch_add(1, Ordering::Relaxed);
            self.stats.copy_in_ns.fetch_add((t1 - t_in).as_nanos() as u64, Ordering::Relaxed);
            self.stats.gpu_ns.fetch_add((t2 - t1).as_nanos() as u64, Ordering::Relaxed);
            self.stats.copy_out_ns.fetch_add(t2.elapsed().as_nanos() as u64, Ordering::Relaxed);
            ok
        }
    }

    /// [`crate::gpu::Backend::na3d`]: `ltx_na3d_k*` over buffers made for the call (released
    /// after it, so the decoder's large volumes are not kept).
    #[allow(clippy::too_many_arguments)]
    fn na3d_impl(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        qd: [usize; 3],
        kd: [usize; 3],
        off: [usize; 3],
        ker: [usize; 3],
        heads: usize,
        hd: usize,
        out: &mut [f32],
    ) -> bool {
        let (nq, nk, d) = (qd.iter().product::<usize>(), kd.iter().product::<usize>(), heads * hd);
        if hd != 64 || nq == 0 || q.len() < nq * d || k.len() < nk * d || v.len() < nk * d || out.len() < nq * d {
            return false;
        }
        if (0..3).any(|a| ker[a] == 0 || ker[a] > kd[a]) {
            return false;
        }
        let Some(pipe) = self.named_pipeline(&format!("ltx_na3d_k{}", ker[2])) else {
            return false;
        };
        let new_len: MsgNewBufLen = self.o.msg_send();
        let void: MsgVoid = self.o.msg_send();
        let mk = |bytes: usize| unsafe { new_len(self.device, self.sels.new_buffer_len, bytes.div_ceil(PAGE) * PAGE, 0) };
        let bufs = [mk(nq * d * 4), mk(nk * d * 4), mk(nk * d * 4), mk(nq * d * 4)];
        let release = || {
            for b in bufs {
                if !b.is_null() {
                    unsafe { void(b, self.sels.release) };
                }
            }
        };
        if bufs.iter().any(|b| b.is_null()) {
            release();
            return false;
        }
        let t0 = Instant::now();
        // SAFETY: the buffers hold the sizes made above.
        unsafe {
            par_copy(self.contents(bufs[0]) as *mut f32, q.as_ptr(), nq * d);
            par_copy(self.contents(bufs[1]) as *mut f32, k.as_ptr(), nk * d);
            par_copy(self.contents(bufs[2]) as *mut f32, v.as_ptr(), nk * d);
        }
        #[repr(C)]
        struct Na3 {
            q: [u32; 3],
            k: [u32; 3],
            off: [u32; 3],
            kt: u32,
            kh: u32,
            heads: u32,
            dim: u32,
            row0: u32,
            wtiles: u32,
            _p: u32,
        }
        let u = |a: [usize; 3]| a.map(|x| x as u32);
        let p = Na3 {
            q: u(qd),
            k: u(kd),
            off: u(off),
            kt: ker[0] as u32,
            kh: ker[1] as u32,
            heads: heads as u32,
            dim: d as u32,
            row0: 0,
            wtiles: qd[2].div_ceil(32) as u32,
            _p: 0,
        };
        // A few frames of queries per command buffer (long submissions stall the display).
        let per = (1usize << 22).div_ceil(qd[1] * qd[2] * heads).clamp(1, qd[0]);
        let mut ok = true;
        let mut t = 0;
        while t < qd[0] && ok {
            let t1 = (t + per).min(qd[0]);
            let mut none = ();
            let pc = Na3 {
                row0: (t * qd[1]) as u32,
                ..p
            };
            ok = self.submit(
                t0,
                &mut none,
                |e, _| {
                    e.pipe(pipe);
                    e.buf(bufs[0], 0, 0);
                    e.buf(bufs[1], 0, 1);
                    e.buf(bufs[2], 0, 2);
                    e.buf(bufs[3], 0, 3);
                    e.bytes(&pc, 4);
                    e.dispatch((qd[2].div_ceil(32) * heads, (t1 - t) * qd[1]), 32);
                },
                |_| {},
            );
            t = t1;
        }
        if ok {
            // SAFETY: the output buffer holds nq rows of d floats.
            unsafe { par_copy(out.as_mut_ptr(), self.contents(bufs[3]) as *const f32, nq * d) };
        }
        release();
        ok
    }

    fn contents(&self, b: Id) -> *mut u8 {
        let f: MsgPtr = self.o.msg_send();
        unsafe { f(b, self.sels.contents) as *mut u8 }
    }

    /// The GPU address of buffer `id`'s first byte.
    pub(crate) fn buffer_gpu_address(&self, id: Id) -> u64 {
        let f: MsgU = self.o.msg_send();
        unsafe { f(id, self.sels.gpu_address) }
    }

    fn gpu_address(&self, b: &Buf) -> u64 {
        let f: MsgU = self.o.msg_send();
        unsafe { f(b.id, self.sels.gpu_address) + b.offset_bytes as u64 }
    }

    /// The matrix-unit pipeline for a weight and the outputs per threadgroup it computes.
    fn mma_pipe(&self, w: &WeightRef, affine: bool, n_out: usize, n_in: usize, tile: Tile) -> Option<(Id, usize)> {
        let m = self.mma()?;
        if !n_in.is_multiple_of(MMA_KB) || !n_out.is_multiple_of(64) {
            return None;
        }
        if tile == Tile::Mid {
            // Built on first use: the 128-token kernel's name with `_m`.
            let (base, tn) = match w {
                // On the matrix units 128-wide output tiles run at 70-75% of the 64-wide ones'
                // speed (prefill and FLUX shapes, `metal::unet` `un_gemm_bench`).
                WeightRef::Bf16(_) if n_out.is_multiple_of(128) && !m.units => ("mma_bf128".to_string(), 128),
                WeightRef::Bf16(_) => ("mma_bf64".to_string(), 64),
                WeightRef::Q8(_) => ("mma_q8".to_string(), 64),
                WeightRef::Q4(_) if affine => ("mma_q4a".to_string(), 64),
                WeightRef::Q4(_) => ("mma_q4".to_string(), 64),
                WeightRef::Gguf(g) if !fp8_ok(g) => return None,
                WeightRef::Gguf(g) => {
                    find_kind(&m.gguf, g.kind)?;
                    (format!("mma_{}", g.kind.name()), 64)
                }
            };
            return Some((self.named_pipeline(&format!("{base}_m"))?, tn));
        }
        let find = find_kind;
        let small = tile == Tile::Small;
        let pick = |big: Id, s: Id| if small { s } else { big };
        Some(match (w, tile) {
            (WeightRef::Bf16(_), Tile::Grouped) => (m.g_bf, 64),
            // Matrix units: 64-wide output tiles for the 128-token GEMMs (128-wide ones are 25-30%
            // slower there); the 32-token ones keep 128 where they divide.
            (WeightRef::Bf16(_), _) if n_out.is_multiple_of(128) && (small || !m.units) => (pick(m.bf[1], m.bf_s[1]), 128),
            (WeightRef::Bf16(_), _) => (pick(m.bf[0], m.bf_s[0]), 64),
            (WeightRef::Q8(_), Tile::Grouped) => (m.g_q8, 64),
            (WeightRef::Q8(_), _) => (pick(m.q8, m.q8_s), 64),
            (WeightRef::Q4(_), Tile::Grouped) if !affine => (m.g_q4, 64),
            (WeightRef::Q4(_), Tile::Grouped) => return None,
            (WeightRef::Q4(_), _) if affine => (pick(m.q4a, m.q4a_s), 64),
            (WeightRef::Q4(_), _) => (pick(m.q4, m.q4_s), 64),
            (WeightRef::Gguf(g), _) if !fp8_ok(g) => return None,
            (WeightRef::Gguf(g), Tile::Grouped) => match self.gk_override(g.kind) {
                Some((p, _)) => (p, 64),
                None => (find(&m.g_gguf, g.kind)?, 64),
            },
            (WeightRef::Gguf(g), _) => (pick(find(&m.gguf, g.kind)?, find(&m.gguf_s, g.kind)?), 64),
        })
    }

    /// The gate / up pair GEMM pipelines with the GLU in their stores (`mmap_*`, matrix units
    /// only) for 128-, 32- and 64-token tiles, and the outputs per threadgroup (32); built on
    /// first use.
    pub(crate) fn mma_pair_pipes(&self, w: &WeightRef, n_out: usize, n_in: usize) -> Option<[(Id, usize); 3]> {
        let m = self.mma()?;
        if !m.units || !n_in.is_multiple_of(MMA_KB) || !n_out.is_multiple_of(32) {
            return None;
        }
        let base = match w {
            WeightRef::Q8(_) => "mmap_q8".to_string(),
            WeightRef::Q4(_) => "mmap_q4".to_string(),
            WeightRef::Gguf(g) if fp8_ok(g) && g.kind != GgmlType::NVFP4 => format!("mmap_{}", g.kind.name()),
            _ => return None,
        };
        Some([
            (self.named_pipeline(&base)?, 32),
            (self.named_pipeline(&format!("{base}_s"))?, 32),
            (self.named_pipeline(&format!("{base}_m"))?, 32),
        ])
    }

    /// Experiments: grouped kernels `{OXINFER_PF_GK}_{TYPE}` (with `_l` / `_s`) in place of
    /// `mmag_{TYPE}`, when the library has them.
    fn gk_override(&self, kind: GgmlType) -> Option<(Id, String)> {
        let b = format!("{}_{}", std::env::var("OXINFER_PF_GK").ok()?, kind.name());
        Some((self.named_pipeline(&b)?, b))
    }

    /// The grouped (MoE) pipelines of a weight for tiles of `MMA_GTM`, 128 and 16 rows (see
    /// [`group_tiles`]); the latter two are built on first use.
    fn grouped_pipes(&self, w: &WeightRef, affine: bool, n_out: usize, n_in: usize) -> Option<[Id; 3]> {
        let p = self.mma_pipe(w, affine, n_out, n_in, Tile::Grouped)?.0;
        let mut base = match w {
            WeightRef::Bf16(_) => "mmag_bf".to_string(),
            WeightRef::Q8(_) => "mmag_q8".to_string(),
            WeightRef::Q4(_) => "mmag_q4".to_string(),
            WeightRef::Gguf(g) => format!("mmag_{}", g.kind.name()),
        };
        if let WeightRef::Gguf(g) = w
            && let Some((_, b)) = self.gk_override(g.kind)
        {
            base = b;
        }
        Some([
            p,
            self.named_pipeline(&format!("{base}_l"))?,
            self.named_pipeline(&format!("{base}_s"))?,
        ])
    }

    /// Encodes the conversion of `rows` rows of `n` floats (row r taken from row `src[r]` if
    /// `src` is given) to scaled half rows plus their inverse scales.
    #[allow(clippy::too_many_arguments)]
    fn encode_x_half(&self, e: &Enc, x: Id, xh: Id, rs: Id, n: usize, rows: usize, src: Option<Id>) {
        #[repr(C)]
        struct Xh {
            n: u32,
            gather: u32,
            stride: u32,
            _p: u32,
        }
        let m = self.mma.as_ref().unwrap();
        e.pipe(m.x_half);
        e.buf(x, 0, 0);
        e.buf(xh, 0, 1);
        e.buf(rs, 0, 2);
        e.bytes(
            &Xh {
                n: n as u32,
                gather: src.is_some() as u32,
                stride: n as u32,
                _p: 0,
            },
            3,
        );
        e.buf(src.unwrap_or(x), 0, 4);
        e.dispatch((rows, 1), 256);
    }

    /// `Backend::matmul` on the matrix units; None when some job has no such kernel.
    fn matmul_mma(&self, x: &[f32], t: usize, n_in: usize, jobs: &mut [Job]) -> Option<bool> {
        // Rows padded to whole token tiles (zero rows; partial tiles take a slower path).
        // Without matrix units 64-row tiles (see `sg_tile`).
        let (tile, tm) = if t <= MMA_SMALL {
            (Tile::Small, MMA_TM_S)
        } else if self.mma.as_ref().is_some_and(|m| !m.units) {
            (Tile::Mid, MMA_TM_S * 2)
        } else {
            (Tile::Dense, MMA_TM)
        };
        let rows = t.next_multiple_of(tm);
        let mut plan = Vec::with_capacity(jobs.len());
        let mut y_total = 0;
        for j in jobs.iter() {
            let mw = j.dev.downcast_ref::<MetalWeight>()?;
            if j.y.len() < t * j.n_out {
                return None;
            }
            plan.push(self.mma_pipe(&j.w, mw.offsets.is_some(), j.n_out, n_in, tile)?);
            y_total += rows * j.n_out;
        }
        let mut st = self.staging.lock().unwrap();
        let st = &mut *st;
        if !self.ensure_staging(st, rows * n_in * 4, y_total * 4)
            || !self.grow(&mut st.xh, rows * n_in * 2)
            || !self.grow(&mut st.rs, rows * 4)
        {
            return Some(false);
        }
        let t0 = Instant::now();
        unsafe {
            let xp = self.contents(st.x) as *mut f32;
            par_copy(xp, x.as_ptr(), t * n_in);
            std::ptr::write_bytes(xp.add(t * n_in), 0, (rows - t) * n_in);
        }
        let encode = |e: &Enc, jobs: &&mut [Job]| {
            self.encode_x_half(e, st.x, st.xh.id, st.rs.id, n_in, rows, None);
            let mut y_off = 0usize;
            for (j, &(pipe, tn)) in jobs.iter().zip(&plan) {
                let mw = j.dev.downcast_ref::<MetalWeight>().unwrap();
                let (_, elem_bytes, scale_bytes) = format_index(&j.w);
                let s_off = match &j.w {
                    WeightRef::Gguf(m) => gguf_s_off(m, mw),
                    _ => mw.scales.as_ref().map(|s| s.offset_bytes / scale_bytes).unwrap_or(0),
                };
                e.pipe(pipe);
                e.buf(mw.data.id, 0, 0);
                e.buf(mw.scales.as_ref().map(|s| s.id).unwrap_or(mw.data.id), 0, 1);
                e.buf(st.xh.id, 0, 2);
                e.buf(st.y, y_off * 4, 3);
                e.bytes(
                    &MmaParams {
                        n_out: j.n_out as u32,
                        n_in: n_in as u32,
                        w_off: (mw.data.offset_bytes / elem_bytes) as u32,
                        s_off: s_off as u32,
                        rows: rows as u32,
                        _p: [0; 3],
                    },
                    4,
                );
                e.buf(st.rs.id, 0, 5);
                if let Some((ob, gpr, gk)) = &mw.offsets {
                    e.buf(ob.id, 0, 6);
                    e.bytes(
                        &OffsetParams {
                            o_off: (ob.offset_bytes / 2) as u32,
                            gpr: *gpr,
                            gk: *gk,
                            _p: 0,
                        },
                        7,
                    );
                }
                e.dispatch((j.n_out / tn, rows / tm), 128);
                y_off += rows * j.n_out;
            }
        };
        let yp = self.contents(st.y) as *const f32;
        let read_back = |jobs: &mut &mut [Job]| {
            let mut y_off = 0usize;
            for j in jobs.iter_mut() {
                unsafe { par_copy(j.y.as_mut_ptr(), yp.add(y_off), t * j.n_out) };
                y_off += rows * j.n_out;
            }
        };
        let mut jobs = jobs;
        Some(self.submit(t0, &mut jobs, encode, read_back))
    }
}

fn format_index(w: &WeightRef) -> (usize, usize, usize) {
    // (pipeline index, bytes per weight element, bytes per scale)
    match w {
        WeightRef::Bf16(_) => (0, 2, 1),
        WeightRef::Q8(_) => (1, 1, 4),
        WeightRef::Q4(_) => (2, 1, 2),
        WeightRef::Gguf(_) => (usize::MAX, 1, 1),
    }
}

impl MetalBackend {
    /// `Backend::upload_moe`; `resident` puts the wrapped experts in the queue's residency set
    /// (else each submission makes them resident: transient state of streamed experts); `tiles`
    /// (as [`crate::gpu::expert_tiles`]) runs the GEMMs on tiled copies of GGUF experts
    /// (`tiled_experts`; transient experts: only when forced on).
    #[allow(clippy::too_many_arguments)]
    fn moe_state(
        &self,
        gate_up: &[WeightRef],
        up: Option<&[WeightRef]>,
        down: &[WeightRef],
        h: usize,
        mi: usize,
        act: crate::gpu::MoeAct,
        resident: bool,
        tiles: Option<bool>,
    ) -> Option<DeviceWeight> {
        self.mma()?;
        if !crate::gpu::moe_offload() || gate_up.is_empty() || gate_up.len() != down.len() || up.is_some_and(|u| u.len() != down.len()) {
            return None;
        }
        let affine = |w: &WeightRef| matches!(w, WeightRef::Q4(m) if m.offsets.is_some());
        // Rows of the first projection: fused gate and up, or the gate alone.
        let g_rows = if up.is_some() { mi } else { 2 * mi };
        let pg3 = self.grouped_pipes(&gate_up[0], affine(&gate_up[0]), g_rows, h)?;
        let pu3 = match up {
            Some(u) => Some(self.grouped_pipes(&u[0], affine(&u[0]), mi, h)?),
            None => None,
        };
        let pd3 = self.grouped_pipes(&down[0], affine(&down[0]), h, mi)?;
        let (pg, pu, pd) = (pg3[0], pu3.map(|p| p[0]), pd3[0]);
        // GGUF row lengths (FP8: 0, its MK_F8 reads the scales at the expert's own address).
        let row_bytes = |w: &WeightRef| match w {
            WeightRef::Gguf(g) if g.block_scales.is_none() => g.row_bytes(),
            _ => 0,
        };
        // Every expert's (data, scales) memory regions; one format for all.
        let regions = |w: &WeightRef| -> ((usize, usize), Option<(usize, usize)>) {
            fn span<T>(v: &[T]) -> (usize, usize) {
                (v.as_ptr() as usize, std::mem::size_of_val(v))
            }
            match w {
                WeightRef::Bf16(v) => (span(v), None),
                WeightRef::Q8(m) => (span(&m.q[..]), Some(span(&m.scales[..]))),
                WeightRef::Q4(m) => (span(&m.q[..]), Some(span(&m.scales[..]))),
                WeightRef::Gguf(m) => (span(&m.data[..]), m.block_scales.as_ref().map(|b| span(&b.s[..]))),
            }
        };
        let mut all: Vec<(usize, usize)> = Vec::new();
        for (w, n_out, n_in, pipe) in gate_up
            .iter()
            .map(|w| (w, g_rows, h, pg))
            .chain(up.unwrap_or(&[]).iter().map(|w| (w, mi, h, pu.unwrap())))
            .chain(down.iter().map(|w| (w, h, mi, pd)))
        {
            if self.mma_pipe(w, affine(w), n_out, n_in, Tile::Grouped)?.0 != pipe {
                return None;
            }
            let (d, s) = regions(w);
            all.push(d);
            all.extend(s);
        }
        // Experts usually are slices of one tensor per projection: wrap adjacent regions as one
        // buffer, so a submission makes a few buffers resident rather than one per expert.
        all.sort_unstable();
        let mut spans: Vec<(usize, usize)> = Vec::new();
        for (p, n) in all {
            match spans.last_mut() {
                Some((s, e)) if p <= *e => *e = (*e).max(p + n),
                _ => spans.push((p, p + n)),
            }
        }
        let release: (MsgVoid, Sel) = (self.o.msg_send(), self.sels.release);
        let mut dev = MoeDev {
            pipes: (pg3, pd3),
            row_bytes: (row_bytes(&gate_up[0]) as u32, row_bytes(&down[0]) as u32),
            bufs: Vec::with_capacity(spans.len()),
            ep: (std::ptr::null_mut(), std::ptr::null_mut()),
            up: None,
            n_exp: gate_up.len(),
            act,
            residency: None,
            release,
            shared: Vec::new(),
            backend: self,
        };
        // Resident experts whose shared tiled copies the model's plans made already: the tables
        // point at those (the checkpoint's experts, given back by then, are not wrapped again).
        if resident && tiles != Some(false) {
            let mut projs: Vec<(&[WeightRef], usize, usize)> = vec![(gate_up, g_rows, h), (down, h, mi)];
            projs.extend(up.map(|u| (u, mi, h)));
            let hits: Vec<Option<(Id, u64, usize)>> = projs.iter().map(|p| self.tiled_shared(p.0, p.1, p.2, tiles)).collect();
            if hits.iter().all(|c| c.is_some()) {
                let copy: MsgNewBufBytes = self.o.msg_send();
                let mut tabs = Vec::new();
                for &(id, base, mb) in hits.iter().flatten() {
                    dev.shared.push(id);
                    let tab: Vec<[u64; 2]> = (0..gate_up.len()).map(|e| [base + (e * mb) as u64, 0]).collect();
                    let t = unsafe {
                        copy(
                            self.device,
                            self.sels.new_buffer_bytes,
                            tab.as_ptr() as *const c_void,
                            tab.len() * 16,
                            0,
                        )
                    };
                    if t.is_null() {
                        return None;
                    }
                    tabs.push(t);
                }
                dev.ep = (tabs[0], tabs[1]);
                dev.row_bytes = (dev.row_bytes.0 | GG_TILED, dev.row_bytes.1 | GG_TILED);
                if let (Some(u), Some(pu)) = (up, pu3) {
                    dev.up = Some((pu, row_bytes(&u[0]) as u32 | GG_TILED, tabs[2]));
                }
                if let Some(r) = self.residency.as_ref() {
                    r.update(&dev.shared, true);
                    dev.residency = Some(r.clone());
                }
                return Some(Box::new(dev));
            }
            for (id, _, _) in hits.into_iter().flatten() {
                if self.unwrap_shared(id) {
                    unsafe { release.0(id, release.1) };
                }
            }
        }
        let mut addr: Vec<u64> = Vec::with_capacity(spans.len());
        for &(s, e) in &spans {
            let b = self.wrap(s as *const u8, e - s)?;
            addr.push(self.gpu_address(&b));
            dev.bufs.push(b);
        }
        let at = |p: usize| {
            let i = spans.partition_point(|&(s, _)| s <= p) - 1;
            addr[i] + (p - spans[i].0) as u64
        };
        let table = |ws: &[WeightRef]| -> Vec<[u64; 2]> {
            ws.iter()
                .map(|w| {
                    let (d, s) = regions(w);
                    [at(d.0), s.map_or(at(d.0), |s| at(s.0))]
                })
                .collect()
        };
        let copy: MsgNewBufBytes = self.o.msg_send();
        for (slot, tab) in [(&mut dev.ep.0, table(gate_up)), (&mut dev.ep.1, table(down))] {
            *slot = unsafe {
                copy(
                    self.device,
                    self.sels.new_buffer_bytes,
                    tab.as_ptr() as *const c_void,
                    tab.len() * 16,
                    0,
                )
            };
            if slot.is_null() {
                return None;
            }
        }
        if let (Some(u), Some(pu)) = (up, pu3) {
            let tab = table(u);
            let id = unsafe {
                copy(
                    self.device,
                    self.sels.new_buffer_bytes,
                    tab.as_ptr() as *const c_void,
                    tab.len() * 16,
                    0,
                )
            };
            if id.is_null() {
                return None;
            }
            dev.up = Some((pu, row_bytes(&u[0]) as u32, id));
        }
        // Tiled copies of the experts (`tiled_experts`), made from the address tables, which then
        // point at them. Resident experts: the copies the model's plans share (made here when no
        // plan has them yet), in the checkpoint's place as there (`expert_tables`). Transient
        // ones (only when forced on): copies of their own, beside them.
        let mode = if resident || tiles == Some(true) { tiles } else { Some(false) };
        if mode != Some(false) {
            let wrapped: Vec<Id> = dev.bufs.iter().map(|b| b.id).collect();
            let mut projs = vec![
                (gate_up, g_rows, h, dev.ep.0, &mut dev.row_bytes.0),
                (down, h, mi, dev.ep.1, &mut dev.row_bytes.1),
            ];
            if let (Some(u), Some((_, rb, tab))) = (up, dev.up.as_mut()) {
                projs.push((u, mi, h, *tab, rb));
            }
            let copies: Vec<Option<(Id, u64, usize)>> = projs
                .iter()
                .map(|p| self.tiled_experts(p.0, p.1, p.2, p.3, 2, &wrapped, resident, mode))
                .collect();
            // (Resident: every projection or none, so that the checkpoint's experts can go.)
            let all = copies.iter().all(|c| c.is_some());
            for ((ws, _, _, tab, rb), c) in projs.into_iter().zip(&copies) {
                let Some((id, addr, mb)) = *c else { continue };
                if resident && !all {
                    if self.unwrap_shared(id) {
                        unsafe { release.0(id, release.1) };
                    }
                    continue;
                }
                // SAFETY: the table holds a [data, scales] pair of u64s per expert.
                let rows = unsafe { std::slice::from_raw_parts_mut(self.contents(tab) as *mut [u64; 2], ws.len()) };
                for (e, r) in rows.iter_mut().enumerate() {
                    *r = [addr + (e * mb) as u64, 0];
                }
                *rb |= GG_TILED;
                if resident {
                    dev.shared.push(id);
                } else {
                    dev.bufs.push(Buf { id, offset_bytes: 0 });
                }
            }
            if resident && all {
                for b in dev.bufs.drain(..) {
                    unsafe { release.0(b.id, release.1) };
                }
                std::thread::spawn(move || {
                    for (s, e) in spans {
                        crate::safetensors::release_copied(s, e - s);
                    }
                });
            }
        }
        if let Some(r) = self.residency.as_ref().filter(|_| resident) {
            r.update(
                &dev.bufs.iter().map(|b| b.id).chain(dev.shared.iter().copied()).collect::<Vec<_>>(),
                true,
            );
            dev.residency = Some(r.clone());
        }
        Some(Box::new(dev))
    }
}

impl Backend for MetalBackend {
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
    fn name(&self) -> &str {
        "metal"
    }

    fn stats(&self) -> Option<String> {
        let n = self.stats.calls.load(Ordering::Relaxed);
        if n == 0 {
            return None;
        }
        let ms = |a: &AtomicU64| a.load(Ordering::Relaxed) as f64 / 1e6;
        Some(format!(
            "{n} submissions: copy-in {:.1} ms, gpu {:.1} ms, copy-out {:.1} ms",
            ms(&self.stats.copy_in_ns),
            ms(&self.stats.gpu_ns),
            ms(&self.stats.copy_out_ns)
        ))
    }

    fn wants(&self, n_out: usize, n_in: usize, t: usize) -> bool {
        // OXINFER_METAL_MIN_T=N: take matmuls from N tokens on, whatever their size (experiments).
        static MIN_T_ENV: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
        let min_t = *MIN_T_ENV.get_or_init(|| std::env::var("OXINFER_METAL_MIN_T").ok().and_then(|v| v.parse().ok()));
        let big = match min_t {
            Some(m) => t >= m,
            None => t >= MIN_T && 2.0 * (t * n_out * n_in) as f64 >= MIN_FLOPS,
        };
        big && n_out.is_multiple_of(TILE) && n_in.is_multiple_of(32)
    }

    fn upload(&self, w: &WeightRef, _n_out: usize, _n_in: usize) -> Option<DeviceWeight> {
        let release: (MsgVoid, Sel) = (self.o.msg_send(), self.sels.release);
        let mw = match w {
            WeightRef::Gguf(m) => {
                self.gguf_pipe(m.kind)?;
                if !fp8_ok(m) {
                    return None;
                }
                MetalWeight {
                    data: self.wrap(m.data.as_ptr(), m.data.len())?,
                    scales: match &m.block_scales {
                        Some(b) => Some(self.wrap(b.s.as_ptr() as *const u8, b.s.len() * 4)?),
                        None => None,
                    },
                    offsets: None,
                    release,
                    resident: std::sync::OnceLock::new(),
                }
            }
            WeightRef::Bf16(w) => MetalWeight {
                data: self.wrap(w.as_ptr() as *const u8, w.len() * 2)?,
                scales: None,
                offsets: None,
                release,
                resident: std::sync::OnceLock::new(),
            },
            WeightRef::Q8(m) => MetalWeight {
                data: self.wrap(m.q.as_ptr() as *const u8, m.q.len())?,
                scales: Some(self.wrap(m.scales.as_ptr() as *const u8, m.scales.len() * 4)?),
                offsets: None,
                release,
                resident: std::sync::OnceLock::new(),
            },
            WeightRef::Q4(m) => MetalWeight {
                data: self.wrap(m.q.as_ptr(), m.q.len())?,
                scales: Some(self.wrap(m.scales.as_ptr() as *const u8, m.scales.len() * 2)?),
                offsets: match &m.offsets {
                    Some(o) => Some((
                        self.wrap(o.v.as_ptr() as *const u8, o.v.len() * 2)?,
                        (m.n_in / o.group) as u32,
                        (o.group / 32) as u32,
                    )),
                    None => None,
                },
                release,
                resident: std::sync::OnceLock::new(),
            },
        };
        Some(Box::new(mw))
    }

    fn conv2d(&self, x: &[f32], g: &crate::gpu::ConvGeom, norm: Option<&crate::gpu::NormSpec>, job: &mut Job) -> bool {
        use crate::gpu::NormKind;
        // Float32 weights (GGUF F32: exact float32 convolutions, `conv_f32`); other GGUF formats
        // have no convolution kernel.
        let f32_pipe = match &job.w {
            WeightRef::Gguf(m) if m.kind == GgmlType::F32 => match self.named_pipeline("conv_f32") {
                Some(p) => Some(p),
                None => return false,
            },
            WeightRef::Gguf(_) => return false,
            _ => None,
        };
        let n_in = g.k * g.k * g.cin;
        let t = g.oh * g.ow;
        let npx = g.h * g.w;
        if !g.cin.is_multiple_of(32) || !job.n_out.is_multiple_of(TILE) || x.len() < npx * g.cin || job.y.len() < t * job.n_out {
            return false;
        }
        if let Some(n) = norm
            && (n.gamma.len() != g.cin
                || n.beta.len() != g.cin
                || matches!(n.kind, NormKind::Group(gr) if gr == 0 || !g.cin.is_multiple_of(gr)))
        {
            return false;
        }
        let Some(mw) = job.dev.downcast_ref::<MetalWeight>() else {
            return false;
        };
        if mw.offsets.is_some() {
            return false; // no affine convolution kernel
        }
        let t_pad = t.div_ceil(TILE) * TILE;
        let mut st = self.staging.lock().unwrap();
        if !self.ensure_staging(&mut st, npx * g.cin * 4, t_pad * job.n_out * 4) {
            return false;
        }
        let n_stats = match norm.map(|n| n.kind) {
            Some(NormKind::Group(gr)) => gr,
            Some(NormKind::Layer) => npx,
            None => 1,
        };
        let st = &mut *st;
        if !self.grow(&mut st.stats, n_stats * 8) || !self.grow(&mut st.gb, 2 * g.cin * 4) {
            return false;
        }
        let contents: MsgPtr = self.o.msg_send();
        let t0 = Instant::now();
        unsafe {
            let xp = contents(st.x, self.sels.contents) as *mut f32;
            std::ptr::copy_nonoverlapping(x.as_ptr(), xp, npx * g.cin);
            if let Some(n) = norm {
                let gp = contents(st.gb.id, self.sels.contents) as *mut f32;
                std::ptr::copy_nonoverlapping(n.gamma.as_ptr(), gp, g.cin);
                std::ptr::copy_nonoverlapping(n.beta.as_ptr(), gp.add(g.cin), g.cin);
            }
        }
        let msg0: Msg0 = self.o.msg_send();
        let set_pipe: MsgSetPipe = self.o.msg_send();
        let set_buf: MsgSetBuffer = self.o.msg_send();
        let set_bytes: MsgSetBytes = self.o.msg_send();
        let dispatch: MsgDispatch = self.o.msg_send();
        let void: MsgVoid = self.o.msg_send();
        let status: MsgU = self.o.msg_send();
        unsafe {
            let t1 = Instant::now();
            let pool = (self.sels.pool_push)();
            let cb = msg0(self.queue, self.sels.command_buffer);
            let enc = msg0(cb, self.sels.compute_encoder);
            // Statistics first; dispatches in one encoder run in order.
            match norm {
                Some(
                    n @ crate::gpu::NormSpec {
                        kind: NormKind::Group(gr), ..
                    },
                ) => {
                    #[repr(C)]
                    struct Gs {
                        n: u32,
                        c: u32,
                        groups: u32,
                        eps: f32,
                    }
                    let p = Gs {
                        n: npx as u32,
                        c: g.cin as u32,
                        groups: *gr as u32,
                        eps: n.eps,
                    };
                    set_pipe(enc, self.sels.set_pipeline, self.group_stats);
                    set_buf(enc, self.sels.set_buffer, st.x, 0, 0);
                    set_buf(enc, self.sels.set_buffer, st.stats.id, 0, 1);
                    set_bytes(
                        enc,
                        self.sels.set_bytes,
                        &p as *const Gs as *const c_void,
                        std::mem::size_of::<Gs>(),
                        2,
                    );
                    dispatch(
                        enc,
                        self.sels.dispatch,
                        &MtlSize { w: *gr, h: 1, d: 1 },
                        &MtlSize { w: 256, h: 1, d: 1 },
                    );
                }
                Some(n) => {
                    #[repr(C)]
                    struct Rs {
                        c: u32,
                        eps: f32,
                    }
                    let p = Rs {
                        c: g.cin as u32,
                        eps: n.eps,
                    };
                    set_pipe(enc, self.sels.set_pipeline, self.row_stats);
                    set_buf(enc, self.sels.set_buffer, st.x, 0, 0);
                    set_buf(enc, self.sels.set_buffer, st.stats.id, 0, 1);
                    set_bytes(
                        enc,
                        self.sels.set_bytes,
                        &p as *const Rs as *const c_void,
                        std::mem::size_of::<Rs>(),
                        2,
                    );
                    dispatch(
                        enc,
                        self.sels.dispatch,
                        &MtlSize { w: npx, h: 1, d: 1 },
                        &MtlSize { w: 32, h: 1, d: 1 },
                    );
                }
                None => {}
            }
            let (fi, elem_bytes, scale_bytes) = match f32_pipe {
                Some(_) => (0, 4, 1),
                None => format_index(&job.w),
            };
            let params = ConvParams {
                n_out: job.n_out as u32,
                n_in: n_in as u32,
                w_off: (mw.data.offset_bytes / elem_bytes) as u32,
                s_off: mw.scales.as_ref().map(|s| s.offset_bytes / scale_bytes).unwrap_or(0) as u32,
                h: g.h as u32,
                w: g.w as u32,
                cin: g.cin as u32,
                k: g.k as u32,
                stride: g.stride as u32,
                ow: g.ow as u32,
                npix: t as u32,
                pad: g.pad as u32,
                mode: match norm.map(|n| n.kind) {
                    None => 0,
                    Some(NormKind::Group(_)) => 1,
                    Some(NormKind::Layer) => 2,
                },
                silu: norm.is_some_and(|n| n.silu) as u32,
                cpg: match norm.map(|n| n.kind) {
                    Some(NormKind::Group(gr)) => (g.cin / gr) as u32,
                    _ => 1,
                },
                _p: 0,
            };
            set_pipe(enc, self.sels.set_pipeline, f32_pipe.unwrap_or(self.conv_pipes[fi]));
            set_buf(enc, self.sels.set_buffer, mw.data.id, 0, 0);
            set_buf(
                enc,
                self.sels.set_buffer,
                mw.scales.as_ref().map(|s| s.id).unwrap_or(mw.data.id),
                0,
                1,
            );
            set_buf(enc, self.sels.set_buffer, st.x, 0, 2);
            set_buf(enc, self.sels.set_buffer, st.y, 0, 3);
            set_bytes(
                enc,
                self.sels.set_bytes,
                &params as *const ConvParams as *const c_void,
                std::mem::size_of::<ConvParams>(),
                4,
            );
            set_buf(enc, self.sels.set_buffer, st.stats.id, 0, 5);
            set_buf(enc, self.sels.set_buffer, st.gb.id, 0, 6);
            let groups = MtlSize {
                w: job.n_out / TILE,
                h: t_pad / TILE,
                d: 1,
            };
            let threads = MtlSize { w: 128, h: 1, d: 1 };
            dispatch(enc, self.sels.dispatch, &groups, &threads);
            void(enc, self.sels.end_encoding);
            void(cb, self.sels.commit);
            void(cb, self.sels.wait);
            let ok = status(cb, self.sels.status) == 4;
            let t2 = Instant::now();
            if ok {
                let yp = contents(st.y, self.sels.contents) as *const f32;
                std::ptr::copy_nonoverlapping(yp, job.y.as_mut_ptr(), t * job.n_out);
            }
            (self.sels.pool_pop)(pool);
            self.stats.calls.fetch_add(1, Ordering::Relaxed);
            self.stats.copy_in_ns.fetch_add((t1 - t0).as_nanos() as u64, Ordering::Relaxed);
            self.stats.gpu_ns.fetch_add((t2 - t1).as_nanos() as u64, Ordering::Relaxed);
            self.stats.copy_out_ns.fetch_add(t2.elapsed().as_nanos() as u64, Ordering::Relaxed);
            ok
        }
    }

    fn attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        tq: usize,
        tk: usize,
        heads: usize,
        dh: usize,
        causal: bool,
        out: &mut [f32],
    ) -> bool {
        let d = heads * dh;
        if q.len() < tq * d || k.len() < tk * d || v.len() < tk * d || out.len() < tq * d || tq == 0 || tk == 0 {
            return false;
        }
        if dh == 128 && !causal && self.mma().is_some_and(|m| m.units) {
            return self.flash_attention_mma(q, k, v, tq, tk, heads, !crate::gpu::exact_attention(), out);
        }
        if (dh == 64 || dh == 128) && !causal {
            return self.flash_attention(q, k, v, tq, tk, heads, dh, out);
        }
        // GEMM tiling: out rows (keys, value dims) in multiples of 64, K in multiples of 32.
        let (dhp, dho, tkp, tqp) = (
            dh.next_multiple_of(32),
            dh.next_multiple_of(64),
            tk.next_multiple_of(64),
            tq.next_multiple_of(64),
        );
        let mut st = self.staging.lock().unwrap();
        let st = &mut *st;
        let ok = self.grow(&mut st.aq, heads * tqp * dhp * 4)
            && self.grow(&mut st.ak, heads * tkp * dhp * 4)
            && self.grow(&mut st.av, heads * dho * tkp * 4)
            && self.grow(&mut st.asc, tqp * tkp * 4)
            && self.grow(&mut st.ao, heads * tqp * dho * 4);
        if !ok {
            return false;
        }
        let contents: MsgPtr = self.o.msg_send();
        let t0 = Instant::now();
        unsafe {
            // Per-head operands, zero padded (padding contributes nothing to any product).
            let qp = contents(st.aq.id, self.sels.contents) as *mut f32;
            let kp = contents(st.ak.id, self.sels.contents) as *mut f32;
            let vp = contents(st.av.id, self.sels.contents) as *mut f32;
            std::ptr::write_bytes(qp, 0, heads * tqp * dhp);
            std::ptr::write_bytes(kp, 0, heads * tkp * dhp);
            std::ptr::write_bytes(vp, 0, heads * dho * tkp);
            for h in 0..heads {
                for i in 0..tq {
                    std::ptr::copy_nonoverlapping(q.as_ptr().add(i * d + h * dh), qp.add((h * tqp + i) * dhp), dh);
                }
                for j in 0..tk {
                    std::ptr::copy_nonoverlapping(k.as_ptr().add(j * d + h * dh), kp.add((h * tkp + j) * dhp), dh);
                    for c in 0..dh {
                        *vp.add((h * dho + c) * tkp + j) = v[j * d + h * dh + c];
                    }
                }
            }
        }
        let msg0: Msg0 = self.o.msg_send();
        let set_pipe: MsgSetPipe = self.o.msg_send();
        let set_buf: MsgSetBuffer = self.o.msg_send();
        let set_bytes: MsgSetBytes = self.o.msg_send();
        let dispatch: MsgDispatch = self.o.msg_send();
        let void: MsgVoid = self.o.msg_send();
        let status: MsgU = self.o.msg_send();
        unsafe {
            let t1 = Instant::now();
            let pool = (self.sels.pool_push)();
            let cb = msg0(self.queue, self.sels.command_buffer);
            let enc = msg0(cb, self.sels.compute_encoder);
            #[repr(C)]
            struct Sm {
                cols: u32,
                ld: u32,
                scale: f32,
                causal: u32,
            }
            let sm = Sm {
                cols: tk as u32,
                ld: tkp as u32,
                scale: 1.0 / (dh as f32).sqrt(),
                causal: causal as u32,
            };
            for h in 0..heads {
                // Scores: keys as the "weights" ([tkp, dhp]), queries as the input.
                let p = Params {
                    n_out: tkp as u32,
                    n_in: dhp as u32,
                    w_off: 0,
                    s_off: 0,
                };
                set_pipe(enc, self.sels.set_pipeline, self.gemm_f32);
                set_buf(enc, self.sels.set_buffer, st.ak.id, h * tkp * dhp * 4, 0);
                set_buf(enc, self.sels.set_buffer, st.ak.id, 0, 1);
                set_buf(enc, self.sels.set_buffer, st.aq.id, h * tqp * dhp * 4, 2);
                set_buf(enc, self.sels.set_buffer, st.asc.id, 0, 3);
                set_bytes(
                    enc,
                    self.sels.set_bytes,
                    &p as *const Params as *const c_void,
                    std::mem::size_of::<Params>(),
                    4,
                );
                dispatch(
                    enc,
                    self.sels.dispatch,
                    &MtlSize {
                        w: tkp / TILE,
                        h: tqp / TILE,
                        d: 1,
                    },
                    &MtlSize { w: 128, h: 1, d: 1 },
                );
                set_pipe(enc, self.sels.set_pipeline, self.softmax);
                set_buf(enc, self.sels.set_buffer, st.asc.id, 0, 0);
                set_bytes(
                    enc,
                    self.sels.set_bytes,
                    &sm as *const Sm as *const c_void,
                    std::mem::size_of::<Sm>(),
                    1,
                );
                dispatch(
                    enc,
                    self.sels.dispatch,
                    &MtlSize { w: tqp, h: 1, d: 1 },
                    &MtlSize { w: 32, h: 1, d: 1 },
                );
                // Output: transposed values as the "weights" ([dho, tkp]), probabilities as the input.
                let p = Params {
                    n_out: dho as u32,
                    n_in: tkp as u32,
                    w_off: 0,
                    s_off: 0,
                };
                set_pipe(enc, self.sels.set_pipeline, self.gemm_f32);
                set_buf(enc, self.sels.set_buffer, st.av.id, h * dho * tkp * 4, 0);
                set_buf(enc, self.sels.set_buffer, st.av.id, 0, 1);
                set_buf(enc, self.sels.set_buffer, st.asc.id, 0, 2);
                set_buf(enc, self.sels.set_buffer, st.ao.id, h * tqp * dho * 4, 3);
                set_bytes(
                    enc,
                    self.sels.set_bytes,
                    &p as *const Params as *const c_void,
                    std::mem::size_of::<Params>(),
                    4,
                );
                dispatch(
                    enc,
                    self.sels.dispatch,
                    &MtlSize {
                        w: dho / TILE,
                        h: tqp / TILE,
                        d: 1,
                    },
                    &MtlSize { w: 128, h: 1, d: 1 },
                );
            }
            void(enc, self.sels.end_encoding);
            void(cb, self.sels.commit);
            void(cb, self.sels.wait);
            let ok = status(cb, self.sels.status) == 4;
            let t2 = Instant::now();
            if ok {
                let op = contents(st.ao.id, self.sels.contents) as *const f32;
                for h in 0..heads {
                    for i in 0..tq {
                        std::ptr::copy_nonoverlapping(op.add((h * tqp + i) * dho), out.as_mut_ptr().add(i * d + h * dh), dh);
                    }
                }
            }
            (self.sels.pool_pop)(pool);
            self.stats.calls.fetch_add(1, Ordering::Relaxed);
            self.stats.copy_in_ns.fetch_add((t1 - t0).as_nanos() as u64, Ordering::Relaxed);
            self.stats.gpu_ns.fetch_add((t2 - t1).as_nanos() as u64, Ordering::Relaxed);
            self.stats.copy_out_ns.fetch_add(t2.elapsed().as_nanos() as u64, Ordering::Relaxed);
            ok
        }
    }

    fn decodes(&self) -> bool {
        true
    }

    fn working_set(&self) -> Option<u64> {
        let f: MsgU = self.o.msg_send();
        let v = unsafe { f(self.device, self.o.sel(c"recommendedMaxWorkingSetSize")) };
        (v > 0).then_some(v)
    }

    fn allocated_memory(&self) -> Option<u64> {
        let f: MsgU = self.o.msg_send();
        Some(unsafe { f(self.device, self.o.sel(c"currentAllocatedSize")) })
    }

    fn dit_session(&'static self) -> Option<Box<dyn crate::gpu::dit::Session>> {
        self.dit_session_impl()
    }

    fn flux_session(&'static self) -> Option<Box<dyn crate::gpu::dit::FluxSession>> {
        self.flux_session_impl()
    }

    fn ltx_session(&'static self) -> Option<Box<dyn crate::gpu::dit::LtxSession>> {
        self.ltx_session_impl()
    }

    fn na_session(&'static self) -> Option<Box<dyn crate::gpu::dit::NaSession>> {
        self.na_session_impl()
    }

    fn na3d(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        qd: [usize; 3],
        kd: [usize; 3],
        off: [usize; 3],
        ker: [usize; 3],
        heads: usize,
        hd: usize,
        out: &mut [f32],
    ) -> bool {
        self.na3d_impl(q, k, v, qd, kd, off, ker, heads, hd, out)
    }

    fn ltx_vae_session(&'static self) -> Option<Box<dyn crate::gpu::dit::LtxVaeSession>> {
        self.ltx_vae_session_impl()
    }

    fn ltx_voc_session(&'static self) -> Option<Box<dyn crate::gpu::dit::LtxVocSession>> {
        self.ltx_voc_session_impl()
    }

    fn t5_session(&'static self) -> Option<Box<dyn crate::gpu::dit::T5Session>> {
        self.t5_session_impl()
    }

    fn clip_session(&'static self) -> Option<Box<dyn crate::gpu::dit::ClipSession>> {
        self.clip_session_impl()
    }

    fn unet_session(&'static self) -> Option<Box<dyn crate::gpu::unet::Session>> {
        self.unet_session_impl()
    }

    fn whisper_session(&'static self) -> Option<Box<dyn crate::gpu::whisper::Session>> {
        self.whisper_session_impl()
    }

    fn attention_cached(&self, j: &mut crate::gpu::AttnJob) -> bool {
        self.attention_cached_impl(j)
    }

    fn gdn_rule(&self, j: &mut crate::gpu::GdnJob) -> bool {
        self.gdn_rule_impl(j)
    }

    fn decode_plan<'s>(&'s self, g: &crate::gpu::decode::Graph) -> Option<Box<dyn crate::gpu::decode::DecodePlan + 's>> {
        self.build_plan(g)
            .map(|p| Box::new(p) as Box<dyn crate::gpu::decode::DecodePlan + 's>)
    }

    fn matvec_rows(&self, x: &[f32], t: usize, n_in: usize, jobs: &mut [crate::gpu::RowsJob]) -> bool {
        self.matvec_rows_impl(x, t, n_in, jobs)
    }

    fn markov_greedy(&self, j: &mut crate::gpu::MarkovJob) -> bool {
        self.markov_greedy_impl(j)
    }

    fn has_markov_greedy(&self) -> bool {
        true
    }

    fn decode_plan_multi<'s>(
        &'s self,
        g: &crate::gpu::decode::Graph,
        tokens: usize,
    ) -> Option<Box<dyn crate::gpu::decode::DecodePlan + 's>> {
        self.build_plan_t(g, tokens, false)
            .map(|p| Box::new(p) as Box<dyn crate::gpu::decode::DecodePlan + 's>)
    }

    fn decode_plan_prefill<'s>(
        &'s self,
        g: &crate::gpu::decode::Graph,
        tokens: usize,
    ) -> Option<Box<dyn crate::gpu::decode::DecodePlan + 's>> {
        self.build_plan_pf(g, tokens)
            .map(|p| Box::new(p) as Box<dyn crate::gpu::decode::DecodePlan + 's>)
    }

    fn decode_plan_multi_fast<'s>(
        &'s self,
        g: &crate::gpu::decode::Graph,
        tokens: usize,
    ) -> Option<Box<dyn crate::gpu::decode::DecodePlan + 's>> {
        self.build_plan_t(g, tokens, true)
            .map(|p| Box::new(p) as Box<dyn crate::gpu::decode::DecodePlan + 's>)
    }

    fn upload_moe(
        &self,
        gate_up: &[WeightRef],
        up: Option<&[WeightRef]>,
        down: &[WeightRef],
        h: usize,
        mi: usize,
        act: crate::gpu::MoeAct,
    ) -> Option<DeviceWeight> {
        self.moe_state(gate_up, up, down, h, mi, act, true, crate::gpu::expert_tiles())
    }

    fn upload_moe_transient(
        &self,
        gate_up: &[WeightRef],
        up: Option<&[WeightRef]>,
        down: &[WeightRef],
        h: usize,
        mi: usize,
        act: crate::gpu::MoeAct,
    ) -> Option<DeviceWeight> {
        self.moe_state(gate_up, up, down, h, mi, act, false, crate::gpu::expert_tiles())
    }

    fn moe(&self, x: &[f32], t: usize, h: usize, mi: usize, routes: &[Vec<(usize, f32)>], dev: &DeviceWeight, y: &mut [f32]) -> bool {
        if !crate::gpu::moe_offload() {
            return false;
        }
        let (Some(m), Some(md)) = (self.mma(), dev.downcast_ref::<MoeDev>()) else {
            return false;
        };
        let k = routes.first().map_or(0, |r| r.len());
        if t == 0 || k == 0 || routes.len() != t || routes.iter().any(|r| r.len() != k) || x.len() < t * h || y.len() < t * h {
            return false;
        }
        // Rows per expert, in expert order: the gathered input is one segment per expert, padded
        // to whole tiles (partial tiles take a slower path in the tensor ops; padding rows come
        // from a zero row after the input and are never read back).
        let mut count: Vec<usize> = Vec::new();
        for &(e, _) in routes.iter().flatten() {
            if e >= md.n_exp {
                return false;
            }
            if e >= count.len() {
                count.resize(e + 1, 0);
            }
            count[e] += 1;
        }
        let (tiles, ntiles, mut start, rows) = group_tiles(&count);
        // Gathered row sources, and each token's (row, weight) slots for the combination.
        let mut src = vec![t as u32; rows];
        let mut slots = vec![[0u32; 2]; t * k];
        for (ti, r) in routes.iter().enumerate() {
            for (j, &(e, w)) in r.iter().enumerate() {
                let i = start[e];
                start[e] += 1;
                src[i] = ti as u32;
                slots[ti * k + j] = [i as u32, w.to_bits()];
            }
        }
        let mut st = self.staging.lock().unwrap();
        let st = &mut *st;
        let pad = rows + 128;
        let ok = self.ensure_staging(st, (t + 1) * h * 4, t * h * 4)
            && self.grow(&mut st.xh, rows * h * 2)
            && self.grow(&mut st.rs, pad * 4)
            && self.grow(&mut st.m_gu, rows * 2 * mi * 4)
            && self.grow(&mut st.m_ah, rows * mi * 2)
            && self.grow(&mut st.m_rs2, pad * 4)
            && self.grow(&mut st.m_o, rows * h * 4)
            && self.grow(&mut st.m_src, rows * 4)
            && self.grow(&mut st.m_tiles, tiles.len() * 16)
            && self.grow(&mut st.m_slots, slots.len() * 8);
        if !ok {
            return false;
        }
        let t0 = Instant::now();
        // SAFETY: every buffer was just sized for what is copied into it.
        unsafe {
            let cp = |dst: Id, src: *const u8, n: usize| std::ptr::copy_nonoverlapping(src, self.contents(dst), n);
            par_copy(self.contents(st.x) as *mut f32, x.as_ptr(), t * h);
            std::ptr::write_bytes((self.contents(st.x) as *mut f32).add(t * h), 0, h);
            cp(st.m_src.id, src.as_ptr() as *const u8, rows * 4);
            cp(st.m_tiles.id, tiles.as_ptr() as *const u8, tiles.len() * 16);
            cp(st.m_slots.id, slots.as_ptr() as *const u8, slots.len() * 8);
        }
        let params = |n_out: usize, n_in: usize, s_off: u32| MmaParams {
            n_out: n_out as u32,
            n_in: n_in as u32,
            w_off: 0,
            s_off,
            rows: rows as u32,
            _p: [0; 3],
        };
        let resident: Vec<Id> = match md.residency {
            Some(_) => Vec::new(),
            None => md.bufs.iter().map(|b| b.id).collect(),
        };
        let encode = |e: &Enc, _: &()| {
            e.use_resources(&resident);
            self.encode_x_half(e, st.x, st.xh.id, st.rs.id, h, rows, Some(st.m_src.id));
            // One grid per tile size (disjoint rows: no barrier between them).
            let grouped = |pipes: [Id; 3], ep: Id, xh: Id, out: (Id, usize), rs: Id, p: &MmaParams| {
                let mut first = 0;
                for (pipe, n) in [pipes[1], pipes[0], pipes[2]].into_iter().zip(ntiles) {
                    if n > 0 {
                        e.pipe(pipe);
                        e.buf(ep, 0, 0);
                        e.buf(st.m_tiles.id, first * 16, 1);
                        e.buf(xh, 0, 2);
                        e.buf(out.0, out.1, 3);
                        e.bytes(p, 4);
                        e.buf(rs, 0, 5);
                        e.dispatch((p.n_out as usize / 64, n), 128);
                    }
                    first += n;
                }
            };
            match md.up {
                // Gate rows [rows, mi], then up rows [rows, mi].
                Some((pu, rb, ep)) => {
                    grouped(
                        md.pipes.0,
                        md.ep.0,
                        st.xh.id,
                        (st.m_gu.id, 0),
                        st.rs.id,
                        &params(mi, h, md.row_bytes.0),
                    );
                    grouped(pu, ep, st.xh.id, (st.m_gu.id, rows * mi * 4), st.rs.id, &params(mi, h, rb));
                }
                None => grouped(
                    md.pipes.0,
                    md.ep.0,
                    st.xh.id,
                    (st.m_gu.id, 0),
                    st.rs.id,
                    &params(2 * mi, h, md.row_bytes.0),
                ),
            }
            #[repr(C)]
            struct Sh {
                n: u32,
                gelu: u32,
                split: u32,
                limit: f32,
            }
            e.pipe(m.silu_half);
            e.buf(st.m_gu.id, 0, 0);
            e.buf(st.m_ah.id, 0, 1);
            e.buf(st.m_rs2.id, 0, 2);
            e.bytes(
                &Sh {
                    n: mi as u32,
                    gelu: (md.act == crate::gpu::MoeAct::GeluTanh) as u32,
                    split: if md.up.is_some() { rows as u32 } else { 0 },
                    limit: match md.act {
                        crate::gpu::MoeAct::SiluClamp(l) => l,
                        _ => 0.0,
                    },
                },
                3,
            );
            e.dispatch((rows, 1), 256);
            grouped(
                md.pipes.1,
                md.ep.1,
                st.m_ah.id,
                (st.m_o.id, 0),
                st.m_rs2.id,
                &params(h, mi, md.row_bytes.1),
            );
            #[repr(C)]
            struct Cb {
                n: u32,
                k: u32,
                ldo: u32,
                _p: u32,
            }
            e.pipe(m.combine);
            e.buf(st.m_o.id, 0, 0);
            e.buf(st.m_slots.id, 0, 1);
            e.buf(st.y, 0, 2);
            e.bytes(
                &Cb {
                    n: h as u32,
                    k: k as u32,
                    ldo: h as u32,
                    _p: 0,
                },
                3,
            );
            e.dispatch(((h / 4).div_ceil(64), t), 64);
        };
        let out = self.contents(st.y) as *const f32;
        let read_back = |_: &mut ()| {
            // SAFETY: the output buffer holds t * h floats.
            let out = unsafe { std::slice::from_raw_parts(out, t * h) };
            crate::kernels::for_chunks(&mut y[..t * h], h, t > 1, |ti, row| {
                for (a, b) in row.iter_mut().zip(&out[ti * h..(ti + 1) * h]) {
                    *a += b;
                }
            });
        };
        self.submit(t0, &mut (), encode, read_back)
    }

    fn matmul(&self, x: &[f32], t: usize, n_in: usize, jobs: &mut [Job]) -> bool {
        if jobs.is_empty() || !n_in.is_multiple_of(32) || x.len() < t * n_in {
            return false;
        }
        if self.mma().is_some()
            && let Some(ok) = self.matmul_mma(x, t, n_in, jobs)
        {
            return ok;
        }
        let t_pad = t.div_ceil(TILE) * TILE;
        let mut y_total = 0;
        for j in jobs.iter() {
            if j.n_out % TILE != 0 || j.y.len() < t * j.n_out || j.dev.downcast_ref::<MetalWeight>().is_none() {
                return false;
            }
            y_total += t_pad * j.n_out;
        }
        let mut st = self.staging.lock().unwrap();
        if !self.ensure_staging(&mut st, t_pad * n_in * 4, y_total * 4) {
            return false;
        }
        let contents: MsgPtr = self.o.msg_send();
        let t0 = Instant::now();
        unsafe {
            let xp = contents(st.x, self.sels.contents) as *mut f32;
            std::ptr::copy_nonoverlapping(x.as_ptr(), xp, t * n_in);
            if t_pad > t {
                std::ptr::write_bytes(xp.add(t * n_in), 0, (t_pad - t) * n_in);
            }
        }
        let msg0: Msg0 = self.o.msg_send();
        let set_pipe: MsgSetPipe = self.o.msg_send();
        let set_buf: MsgSetBuffer = self.o.msg_send();
        let set_bytes: MsgSetBytes = self.o.msg_send();
        let dispatch: MsgDispatch = self.o.msg_send();
        let void: MsgVoid = self.o.msg_send();
        let status: MsgU = self.o.msg_send();
        unsafe {
            let t1 = Instant::now();
            let pool = (self.sels.pool_push)();
            let cb = msg0(self.queue, self.sels.command_buffer);
            let enc = msg0(cb, self.sels.compute_encoder);
            set_buf(enc, self.sels.set_buffer, st.x, 0, 2);
            let mut y_off = 0usize;
            for j in jobs.iter() {
                let mw = j.dev.downcast_ref::<MetalWeight>().unwrap();
                let (fi, elem_bytes, scale_bytes) = format_index(&j.w);
                // GGUF: byte offset of the data, and the row length in bytes in place of s_off.
                let (s_off, pipe) = match &j.w {
                    WeightRef::Gguf(m) => (gguf_s_off(m, mw), self.gguf_pipe(m.kind).unwrap()),
                    _ => (
                        mw.scales.as_ref().map(|s| s.offset_bytes / scale_bytes).unwrap_or(0),
                        if mw.offsets.is_some() { self.pipe_q4a } else { self.pipes[fi] },
                    ),
                };
                let params = Params {
                    n_out: j.n_out as u32,
                    n_in: n_in as u32,
                    w_off: (mw.data.offset_bytes / elem_bytes) as u32,
                    s_off: s_off as u32,
                };
                set_pipe(enc, self.sels.set_pipeline, pipe);
                if let Some((ob, gpr, gk)) = &mw.offsets {
                    let op = OffsetParams {
                        o_off: (ob.offset_bytes / 2) as u32,
                        gpr: *gpr,
                        gk: *gk,
                        _p: 0,
                    };
                    set_buf(enc, self.sels.set_buffer, ob.id, 0, 5);
                    set_bytes(
                        enc,
                        self.sels.set_bytes,
                        &op as *const OffsetParams as *const c_void,
                        std::mem::size_of::<OffsetParams>(),
                        6,
                    );
                }
                set_buf(enc, self.sels.set_buffer, mw.data.id, 0, 0);
                set_buf(
                    enc,
                    self.sels.set_buffer,
                    mw.scales.as_ref().map(|s| s.id).unwrap_or(mw.data.id),
                    0,
                    1,
                );
                set_buf(enc, self.sels.set_buffer, st.y, y_off * 4, 3);
                set_bytes(
                    enc,
                    self.sels.set_bytes,
                    &params as *const Params as *const c_void,
                    std::mem::size_of::<Params>(),
                    4,
                );
                let groups = MtlSize {
                    w: j.n_out / TILE,
                    h: t_pad / TILE,
                    d: 1,
                };
                let threads = MtlSize { w: 128, h: 1, d: 1 };
                dispatch(enc, self.sels.dispatch, &groups, &threads);
                y_off += t_pad * j.n_out;
            }
            void(enc, self.sels.end_encoding);
            void(cb, self.sels.commit);
            void(cb, self.sels.wait);
            let ok = status(cb, self.sels.status) == 4; // MTLCommandBufferStatusCompleted
            let t2 = Instant::now();
            if ok {
                let yp = contents(st.y, self.sels.contents) as *const f32;
                let mut y_off = 0usize;
                for j in jobs.iter_mut() {
                    std::ptr::copy_nonoverlapping(yp.add(y_off), j.y.as_mut_ptr(), t * j.n_out);
                    y_off += t_pad * j.n_out;
                }
            }
            (self.sels.pool_pop)(pool);
            self.stats.calls.fetch_add(1, Ordering::Relaxed);
            self.stats.copy_in_ns.fetch_add((t1 - t0).as_nanos() as u64, Ordering::Relaxed);
            self.stats.gpu_ns.fetch_add((t2 - t1).as_nanos() as u64, Ordering::Relaxed);
            self.stats.copy_out_ns.fetch_add(t2.elapsed().as_nanos() as u64, Ordering::Relaxed);
            ok
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels;
    use crate::quant::{quantize_bf16, quantize_q4};

    /// A backend with the matrix-unit kernels, or None (other devices skip these tests).
    fn backend() -> Option<MetalBackend> {
        let b = MetalBackend::new().ok()?;
        if let Some(e) = &b.mma_error {
            panic!("matrix-unit kernels failed to compile: {e}");
        }
        b.mma.as_ref()?;
        Some(b)
    }

    fn rnd(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }

    fn bf16(n: usize, seed: &mut u64) -> Vec<u16> {
        (0..n).map(|_| (rnd(seed).to_bits() >> 16) as u16).collect()
    }

    /// Rows of very different magnitudes (the per-row half scaling must cover them, beyond
    /// half's range included).
    fn input(t: usize, n: usize, seed: &mut u64) -> Vec<f32> {
        (0..t * n).map(|i| rnd(seed) * [1e-3, 1.0, 300.0, 2e5][(i / n) % 4]).collect()
    }

    /// Worst error of any row relative to that row's largest reference value.
    fn row_err(y: &[f32], r: &[f32], n: usize) -> f32 {
        y.chunks(n)
            .zip(r.chunks(n))
            .map(|(a, b)| {
                let m = b.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30);
                a.iter().zip(b).fold(0f32, |e, (p, q)| e.max((p - q).abs())) / m
            })
            .fold(0f32, f32::max)
    }

    /// Timing of `Backend::moe` at Qwen3.8-Flash-Next's shapes (512 experts, Q4_K gate / up,
    /// Q5_1 down, top 10): `cargo test --release --lib moe_bench -- --ignored --nocapture`, or
    /// `OXINFER_MOE_SHAPE=experts,hidden,inter,k,GATE_TYPE,DOWN_TYPE[,gelu]` (Gemma 4 26B-A4B:
    /// `128,2816,704,8,Q4_0,Q4_0,gelu`).
    /// Routes are uniform at random, or with `OXINFER_MOE_ROUTES=FILE` the median layer of each
    /// size recorded by a prefill-plan run with `OXINFER_PF_ROUTES=FILE` (real routing is far from
    /// uniform: at 2048 tokens the busiest expert takes about half of them).
    #[test]
    #[ignore]
    fn moe_bench() {
        let Some(be) = backend() else { return };
        let shape = std::env::var("OXINFER_MOE_SHAPE").unwrap_or("512,2560,640,10,Q4_K,Q5_1".into());
        let f: Vec<&str> = shape.split(',').collect();
        let (ne, h, mi, k): (usize, usize, usize, usize) = (
            f[0].parse().unwrap(),
            f[1].parse().unwrap(),
            f[2].parse().unwrap(),
            f[3].parse().unwrap(),
        );
        let kind = |n: &str| GgmlType::ALL.into_iter().find(|t| t.name() == n).unwrap();
        let (gk, dk) = (kind(f[4]), kind(f[5]));
        let act = if f.get(6) == Some(&"gelu") {
            crate::gpu::MoeAct::GeluTanh
        } else {
            crate::gpu::MoeAct::Silu
        };
        let mut seed = 11;
        // Random blocks with sane f16 scales (bytes 0..4 of each block, Q4_K; 0..4 Q5_1).
        let make = |kind: GgmlType, n_out: usize, n_in: usize, seed: &mut u64| {
            let rb = kind.row_bytes(n_in);
            let mut q = crate::quant::PageVec::<u8>::zeroed(ne * n_out * rb);
            for b in q.iter_mut() {
                *b = (rnd(seed).to_bits() >> 9) as u8;
            }
            let bs = kind.type_size();
            for blk in q.chunks_mut(bs) {
                blk[0..2].copy_from_slice(&0x2000u16.to_le_bytes());
                blk[2..4].copy_from_slice(&0x2000u16.to_le_bytes());
            }
            q
        };
        let g = make(gk, mi, h, &mut seed);
        let u = make(gk, mi, h, &mut seed);
        let d = make(dk, h, mi, &mut seed);
        let mats = |q: &crate::quant::PageVec<u8>, kind: GgmlType, n_out: usize, n_in: usize| -> Vec<crate::ggml::GgufMatrix> {
            let sz = n_out * kind.row_bytes(n_in);
            (0..ne)
                .map(|e| crate::ggml::GgufMatrix {
                    kind,
                    data: unsafe { crate::quant::QBuf::borrowed(&q[e * sz..(e + 1) * sz]) },
                    n_out,
                    n_in,
                    block_scales: None,
                    x4: false,
                })
                .collect()
        };
        let (gm, um, dm) = (mats(&g, gk, mi, h), mats(&u, gk, mi, h), mats(&d, dk, h, mi));
        let gw: Vec<WeightRef> = gm.iter().map(WeightRef::Gguf).collect();
        let uw: Vec<WeightRef> = um.iter().map(WeightRef::Gguf).collect();
        let dw: Vec<WeightRef> = dm.iter().map(WeightRef::Gguf).collect();
        let dev = be.upload_moe(&gw, Some(&uw), &dw, h, mi, act).unwrap();
        // With an experimental kernel (OXINFER_PF_GK), the default kernels' results to compare.
        let gk = std::env::var("OXINFER_PF_GK").ok();
        let dref = gk.as_ref().map(|v| {
            unsafe { std::env::remove_var("OXINFER_PF_GK") };
            let d = be.upload_moe(&gw, Some(&uw), &dw, h, mi, act).unwrap();
            unsafe { std::env::set_var("OXINFER_PF_GK", v) };
            d
        });
        // Recorded layers: (tokens, rows per expert).
        let recorded: Option<Vec<(usize, Vec<usize>)>> = std::env::var_os("OXINFER_MOE_ROUTES").map(|p| {
            let b = std::fs::read(p).unwrap();
            let w: Vec<usize> = b.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c) as usize).collect();
            w.chunks_exact(ne + 1).map(|r| (r[0], r[1..].to_vec())).collect()
        });
        let sizes: Vec<usize> = match &recorded {
            Some(r) => {
                let mut v: Vec<usize> = r.iter().map(|l| l.0).collect();
                v.sort_unstable();
                v.dedup();
                v
            }
            None => vec![512, 1024, 2048, 4096],
        };
        for t in sizes {
            let x = input(t, h, &mut seed);
            let mut routes: Vec<Vec<(usize, f32)>> = (0..t)
                .map(|_| {
                    let mut v: Vec<usize> = Vec::new();
                    while v.len() < k {
                        let e = ((rnd(&mut seed) + 0.5) * ne as f32) as usize % ne;
                        if !v.contains(&e) {
                            v.push(e);
                        }
                    }
                    v.into_iter().map(|e| (e, 0.1)).collect()
                })
                .collect();
            if let Some(r) = &recorded {
                // The median layer of this size; a token's experts are consecutive slots of a
                // cursor running over the experts' rows (distinct, k per token).
                let layers: Vec<&Vec<usize>> = r.iter().filter(|l| l.0 == t).map(|l| &l.1).collect();
                let mut by_max = layers.clone();
                by_max.sort_by_key(|c| c.iter().max().copied());
                routes = vec![Vec::new(); t];
                let mut cur = 0;
                for (e, &c) in by_max[by_max.len() / 2].iter().enumerate() {
                    for _ in 0..c {
                        routes[cur % t].push((e, 0.1));
                        cur += 1;
                    }
                }
            }
            let mut y = vec![0f32; t * h];
            assert!(be.moe(&x, t, h, mi, &routes, &dev, &mut y));
            if let Some(d) = &dref {
                let mut r = vec![0f32; t * h];
                assert!(be.moe(&x, t, h, mi, &routes, d, &mut r));
                eprintln!("t={t}: vs default kernels {:.3e}", row_err(&y, &r, h));
            }
            // Results of another run (`OXINFER_MOE_SAVE=DIR` writes them, `OXINFER_MOE_CMP=DIR`
            // compares).
            let file = |d: std::ffi::OsString| std::path::Path::new(&d).join(format!("moe{t}.f32"));
            if let Some(d) = std::env::var_os("OXINFER_MOE_SAVE") {
                std::fs::write(file(d), y.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
            }
            if let Some(d) = std::env::var_os("OXINFER_MOE_CMP") {
                let b = std::fs::read(file(d)).unwrap();
                let r: Vec<f32> = b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
                eprintln!("t={t}: vs saved {:.3e}", row_err(&y, &r, h));
            }
            let n = 10;
            let g0 = be.stats.gpu_ns.load(Ordering::Relaxed);
            for _ in 0..n {
                assert!(be.moe(&x, t, h, mi, &routes, &dev, &mut y));
            }
            let dt = (be.stats.gpu_ns.load(Ordering::Relaxed) - g0) as f64 / 1e9 / n as f64;
            let flops = 2.0 * (t * k) as f64 * (3 * h * mi) as f64;
            eprintln!("moe t={t}: {:.2} ms GPU, {:.1} TFLOP/s useful", dt * 1e3, flops / dt / 1e12);
        }
    }

    /// A random FP8 E4M3 matrix (no NaN bytes) with 128 x 128 block scales.
    fn fp8(n_out: usize, n_in: usize, seed: &mut u64) -> crate::ggml::GgufMatrix {
        let mut q = crate::quant::PageVec::<u8>::zeroed(n_out * n_in);
        for b in q.iter_mut() {
            let v = (rnd(seed).to_bits() >> 7) as u8;
            *b = if v & 0x7f == 0x7f { v ^ 1 } else { v };
        }
        let mut s = crate::quant::PageVec::<f32>::zeroed(n_out.div_ceil(128) * n_in.div_ceil(128));
        for v in s.iter_mut() {
            *v = 0.01 + 0.02 * rnd(seed).abs();
        }
        crate::ggml::GgufMatrix::fp8(q.into(), s.into(), n_out, n_in, (128, 128)).unwrap()
    }

    fn cpu(w: &WeightRef, n_out: usize, n_in: usize, x: &[f32], t: usize) -> Vec<f32> {
        let mut y = vec![0f32; t * n_out];
        match w {
            WeightRef::Bf16(w) => kernels::matmul(w, n_out, n_in, x, t, &mut y),
            WeightRef::Q8(m) => kernels::matmul_quant_multi(&mut [(*m, &mut y[..])], x, t),
            WeightRef::Q4(m) => kernels::matmul_quant_multi(&mut [(*m, &mut y[..])], x, t),
            WeightRef::Gguf(m) => kernels::matmul_quant_multi(&mut [(*m, &mut y[..])], x, t),
        }
        y
    }

    #[test]
    fn mma_matmul_matches_cpu() {
        let Some(be) = backend() else { return };
        let (n_out, n_in) = (192, 512); // bf16 at 64 outputs per threadgroup
        let mut seed = 1;
        let w = bf16(n_out * n_in, &mut seed);
        let q8 = quantize_bf16(&w, n_out, n_in);
        let q4 = quantize_q4(&w, n_out, n_in);
        let gg: Vec<_> = [GgmlType::Q4_K, GgmlType::Q6_K, GgmlType::IQ4_XS, GgmlType::MXFP4]
            .into_iter()
            .map(|k| {
                let raw = crate::ggml::random_blocks(k, n_out * n_in / k.block_size(), &mut seed);
                crate::ggml::GgufMatrix::from_bytes(k, &raw, n_out, n_in).unwrap()
            })
            .collect();
        let f8 = fp8(n_out, n_in, &mut seed);
        // NVFP4 with its tensor-wide factor: random codes, E4M3 scales between 2^-3 and 2^3.
        let packed: Vec<u8> = (0..n_out * n_in / 2).map(|_| (rnd(&mut seed).to_bits() >> 9) as u8).collect();
        let sc: Vec<u8> = (0..n_out * n_in / 16)
            .map(|_| 32 + (rnd(&mut seed).to_bits() >> 9) as u8 % 48)
            .collect();
        let nv = crate::model::nvfp4_blocks(&packed, &sc, 0.013, n_out, n_in).unwrap();
        let mut ws = vec![
            WeightRef::Bf16(&w),
            WeightRef::Q8(&q8),
            WeightRef::Q4(&q4),
            WeightRef::Gguf(&f8),
            WeightRef::Gguf(&nv),
        ];
        ws.extend(gg.iter().map(WeightRef::Gguf));
        // Dense tiles (with a partial one) and 32-token tiles.
        for (w, t) in ws.iter().flat_map(|w| [(w, 150), (w, 37)]) {
            let x = input(t, n_in, &mut seed);
            let dev = be.upload(w, n_out, n_in).unwrap();
            let mut y = vec![0f32; t * n_out];
            let wr = match w {
                WeightRef::Bf16(v) => WeightRef::Bf16(v),
                WeightRef::Q8(m) => WeightRef::Q8(m),
                WeightRef::Q4(m) => WeightRef::Q4(m),
                WeightRef::Gguf(m) => WeightRef::Gguf(m),
            };
            let mut jobs = [Job {
                w: wr,
                dev: &dev,
                n_out,
                y: &mut y,
            }];
            assert!(be.matmul_mma(&x, t, n_in, &mut jobs) == Some(true));
            let e = row_err(&y, &cpu(w, n_out, n_in, &x, t), n_out);
            assert!(e < 3e-3, "relative error {e}");
        }
    }

    /// `w` as f32 rows, and an exact (f64) matrix-vector product with them.
    fn dense(w: &WeightRef, n_out: usize, n_in: usize) -> Vec<f32> {
        let mut f = vec![0f32; n_out * n_in];
        match w {
            WeightRef::Bf16(v) => f.iter_mut().zip(v.iter()).for_each(|(o, &b)| *o = f32::from_bits((b as u32) << 16)),
            WeightRef::Q4(m) => crate::quant::dequantize_rows_q4(m, 0, n_out, &mut f),
            WeightRef::Gguf(m) => m.dequantize_rows(0, n_out, &mut f),
            _ => unimplemented!(),
        }
        f
    }
    fn matvec(w: &[f32], x: &[f32]) -> Vec<f32> {
        w.chunks(x.len())
            .map(|r| r.iter().zip(x).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() as f32)
            .collect()
    }

    /// Grouped GEMMs on tiled copies of GGUF experts (`tiled_experts`) give the results of the
    /// checkpoint's layout bit for bit, for formats of 32- and 256-element blocks, fused and
    /// separate gate / up projections and tiles of every size (one expert past 128 rows); both
    /// match the CPU up to the half-precision operands' rounding.
    #[test]
    fn tiled_experts_match() {
        let Some(be) = backend() else { return };
        let (ne, h, mi, t) = (6, 256, 256, 160);
        let mut seed = 5;
        // Routes: expert 0 takes every token (tiles of 128 and 32 rows), the others uneven
        // shares (16-row tiles), expert 4 none.
        let routes: Vec<Vec<(usize, f32)>> = (0..t).map(|i| vec![(0, 0.6), ([1, 2, 3, 5, 1, 1][i % 6], 0.4)]).collect();
        let x = input(t, h, &mut seed);
        use crate::gpu::MoeAct;
        for (kind, act, split) in [
            (GgmlType::Q4_0, MoeAct::GeluTanh, false),
            (GgmlType::Q8_0, MoeAct::Silu, false),
            (GgmlType::Q4_K, MoeAct::Silu, true),
            (GgmlType::Q5_K, MoeAct::Silu, false),
            (GgmlType::Q6_K, MoeAct::GeluTanh, true),
            (GgmlType::IQ4_XS, MoeAct::Silu, false),
            (GgmlType::MXFP4, MoeAct::Silu, true),
            (GgmlType::F16, MoeAct::Silu, false),
        ] {
            // Random blocks with sane f16 scales (MXFP4: a power-of-two exponent byte; F16:
            // every value).
            let scales: &[usize] = match kind {
                GgmlType::Q4_K | GgmlType::Q5_K => &[0, 2],
                GgmlType::Q6_K => &[208],
                GgmlType::MXFP4 => &[],
                GgmlType::F16 => &[0, 2, 4, 6],
                _ => &[0],
            };
            let mut make = |n_out: usize, n_in: usize| -> Vec<crate::quant::PageVec<u8>> {
                (0..ne)
                    .map(|_| {
                        let mut q = crate::quant::PageVec::<u8>::zeroed(n_out * kind.row_bytes(n_in));
                        for b in q.iter_mut() {
                            *b = (rnd(&mut seed).to_bits() >> 9) as u8;
                        }
                        for blk in q.chunks_mut(if kind == GgmlType::F16 { 8 } else { kind.type_size() }) {
                            for &o in scales {
                                let v = match kind {
                                    GgmlType::F16 => half::f16::from_f32(rnd(&mut seed) * 0.1).to_bits(),
                                    _ => 0x2000,
                                };
                                blk[o..o + 2].copy_from_slice(&v.to_le_bytes());
                            }
                            if kind == GgmlType::MXFP4 {
                                blk[0] = 120 + blk[0] % 8;
                            }
                        }
                        q
                    })
                    .collect()
            };
            let (gq, uq, dq) = (make(if split { mi } else { 2 * mi }, h), make(mi, h), make(h, mi));
            let mats = |qs: &[crate::quant::PageVec<u8>], n_out: usize, n_in: usize| -> Vec<crate::ggml::GgufMatrix> {
                qs.iter()
                    .map(|q| crate::ggml::GgufMatrix {
                        kind,
                        data: unsafe { crate::quant::QBuf::borrowed(&q[..]) },
                        n_out,
                        n_in,
                        block_scales: None,
                        x4: false,
                    })
                    .collect()
            };
            let (gm, um, dm) = (mats(&gq, if split { mi } else { 2 * mi }, h), mats(&uq, mi, h), mats(&dq, h, mi));
            fn refs(m: &[crate::ggml::GgufMatrix]) -> Vec<WeightRef<'_>> {
                m.iter().map(WeightRef::Gguf).collect()
            }
            let (gw, uw, dw) = (refs(&gm), refs(&um), refs(&dm));
            let up = split.then_some(&uw[..]);
            let mut y = [vec![0f32; t * h], vec![0f32; t * h]];
            for (tiles, y) in [false, true].into_iter().zip(y.iter_mut()) {
                let dev = be.moe_state(&gw, up, &dw, h, mi, act, true, Some(tiles)).unwrap();
                let md = dev.downcast_ref::<MoeDev>().unwrap();
                assert_eq!(md.row_bytes.0 & GG_TILED != 0, tiles, "{kind:?}: tiled copies");
                assert!(be.moe(&x, t, h, mi, &routes, &dev, y));
            }
            let same = y[0].iter().zip(&y[1]).all(|(a, b)| a.to_bits() == b.to_bits());
            assert!(same, "{kind:?}: tiled experts differ from the checkpoint's layout");
            // Against the CPU.
            let fg: Vec<Vec<f32>> = gw.iter().map(|w| dense(w, if split { mi } else { 2 * mi }, h)).collect();
            let fu: Vec<Vec<f32>> = uw.iter().map(|w| dense(w, mi, h)).collect();
            let fd: Vec<Vec<f32>> = dw.iter().map(|w| dense(w, h, mi)).collect();
            let mut r = vec![0f32; t * h];
            for (i, rt) in routes.iter().enumerate() {
                let xi = &x[i * h..(i + 1) * h];
                for &(e, wt) in rt {
                    let mut g = matvec(&fg[e], xi);
                    if split {
                        g.extend(matvec(&fu[e], xi));
                    }
                    let a: Vec<f32> = (0..mi)
                        .map(|j| match act {
                            MoeAct::GeluTanh => crate::kernels::gelu_tanh(g[j]) * g[mi + j],
                            _ => g[j] / (1.0 + (-g[j]).exp()) * g[mi + j],
                        })
                        .collect();
                    let o = matvec(&fd[e], &a);
                    for (yv, ov) in r[i * h..(i + 1) * h].iter_mut().zip(&o) {
                        *yv += wt * ov;
                    }
                }
            }
            // (Random K-quant minimums cancel: Q5_K's half-rounded weights are ~6e-3 off.)
            let e = row_err(&y[1], &r, h);
            assert!(e < 1e-2, "{kind:?}: relative error {e} against the CPU");
            // Decoding (`MOp::Experts`' kernels) on the tiled copies, in every geometry: the
            // checkpoint layout's activations and outputs (the same products summed in another
            // order, so up to f32 rounding), for tokens of one expert and of two.
            let gelu = act == MoeAct::GeluTanh;
            for i in [0, 1, 5] {
                let rt = &routes[i][..if i == 5 { 1 } else { 2 }];
                let xi = &x[i * h..(i + 1) * h];
                let Some(plain) = be.decode_experts_once(&gw, up, &dw, (h, mi), gelu, xi, rt, None) else {
                    continue; // (no decode kernels: MXFP4)
                };
                if rt.len() == 2 {
                    let e = row_err(&plain.1, &r[i * h..(i + 1) * h], h);
                    assert!(e < 1e-2, "{kind:?}: decode relative error {e} against the CPU");
                }
                for geo in super::decode::XT_GEOS {
                    let t = be.decode_experts_once(&gw, up, &dw, (h, mi), gelu, xi, rt, Some(geo)).unwrap();
                    let (ea, ey) = (row_err(&t.0, &plain.0, mi), row_err(&t.1, &plain.1, h));
                    assert!(ea < 1e-5 && ey < 1e-5, "{kind:?}: tiled decode {geo:?} off by {ea} / {ey}");
                }
            }
            // Several tokens' experts in one dispatch (batched decode steps): each token's
            // activations and output those of its own dispatch, within rounding (bit for bit
            // where the compiler keeps the same arithmetic, which is printed, not required: that
            // depends on the chip), for 1 to 8 tokens sharing some experts.
            let mut same = true;
            for nt in [1usize, 3, 8] {
                let rts: Vec<Vec<(usize, f32)>> = (0..nt)
                    .map(|i| vec![([0, 2, 3, 0, 5, 1, 3, 2][i], 0.7), ([1, 1, 5, 2, 0, 3, 2, 5][i], 0.3)])
                    .collect();
                let rr: Vec<&[(usize, f32)]> = rts.iter().map(|r| &r[..]).collect();
                let xs = &x[..nt * h];
                for geo in std::iter::once(None).chain(super::decode::XT_GEOS.map(Some)) {
                    let Some(one) = be.decode_experts_rows(&gw, up, &dw, (h, mi), gelu, xs, &rr, geo, false) else {
                        continue;
                    };
                    let all = be.decode_experts_rows(&gw, up, &dw, (h, mi), gelu, xs, &rr, geo, true).unwrap();
                    let (ea, ey) = (row_err(&all.0, &one.0, mi), row_err(&all.1, &one.1, h));
                    assert!(ea < 1e-5 && ey < 1e-5, "{kind:?}: {nt} tokens' experts {geo:?} off by {ea} / {ey}");
                    same &= all
                        .0
                        .iter()
                        .zip(&one.0)
                        .chain(all.1.iter().zip(&one.1))
                        .all(|(a, b)| a.to_bits() == b.to_bits());
                }
            }
            eprintln!(
                "{kind:?}: several tokens' experts {}",
                if same { "bit-identical" } else { "within rounding" }
            );
        }
    }

    #[test]
    fn mma_moe_matches_cpu() {
        let Some(be) = backend() else { return };
        let (ne, h, mi, t) = (8, 128, 64, 75);
        let mut seed = 7;
        // bf16 experts as slices of one tensor per projection (one buffer each), Q4 separately.
        let gu = bf16(ne * 2 * mi * h, &mut seed);
        let dn = bf16(ne * h * mi, &mut seed);
        let gu4: Vec<_> = gu.chunks(2 * mi * h).map(|w| quantize_q4(w, 2 * mi, h)).collect();
        let dn4: Vec<_> = dn.chunks(h * mi).map(|w| quantize_q4(w, h, mi)).collect();
        // FP8 experts (the official Mistral Large 4 checkpoint's format).
        let gu8: Vec<_> = (0..ne).map(|_| fp8(2 * mi, h, &mut seed)).collect();
        let dn8: Vec<_> = (0..ne).map(|_| fp8(h, mi, &mut seed)).collect();
        let x = input(t, h, &mut seed);
        // Routes: expert 5 unused, the others with uneven loads.
        let routes: Vec<Vec<(usize, f32)>> = (0..t)
            .map(|i| vec![([0, 1, 2, 3, 4, 6, 7][i % 7], 0.7), ([1, 7][i % 2], 0.3)])
            .collect();
        // bf16 / Q4 / FP8 (format 0 / 1 / 2) fused, SiLU / GELU; bf16 with gate and up apart (GGUF
        // layout).
        use crate::gpu::MoeAct;
        for (fmt, act, split) in [
            (0, MoeAct::Silu, false),
            (1, MoeAct::Silu, false),
            (0, MoeAct::GeluTanh, false),
            (1, MoeAct::GeluTanh, false),
            (0, MoeAct::Silu, true),
            (0, MoeAct::SiluClamp(0.5), true),
            (2, MoeAct::Silu, false),
        ] {
            // The clamp is absolute: inputs of unit scale, so that some activations are clamped
            // and some not (rows of very large inputs would all clamp, and cancel out).
            let x1: Vec<f32> = x
                .iter()
                .enumerate()
                .map(|(i, v)| v / [1e-3, 1.0, 300.0, 2e5][(i / h) % 4])
                .collect();
            let x = if matches!(act, MoeAct::SiluClamp(_)) { &x1 } else { &x };
            let (wg, wd): (Vec<WeightRef>, Vec<WeightRef>) = (0..ne)
                .map(|e| match fmt {
                    0 => (
                        WeightRef::Bf16(&gu[e * 2 * mi * h..(e + 1) * 2 * mi * h]),
                        WeightRef::Bf16(&dn[e * h * mi..(e + 1) * h * mi]),
                    ),
                    1 => (WeightRef::Q4(&gu4[e]), WeightRef::Q4(&dn4[e])),
                    _ => (WeightRef::Gguf(&gu8[e]), WeightRef::Gguf(&dn8[e])),
                })
                .unzip();
            let half = |e: usize, i: usize| WeightRef::Bf16(&gu[e * 2 * mi * h + i * mi * h..e * 2 * mi * h + (i + 1) * mi * h]);
            let (sg, su): (Vec<WeightRef>, Vec<WeightRef>) = (0..ne).map(|e| (half(e, 0), half(e, 1))).unzip();
            let dev = match split {
                false => be.upload_moe(&wg, None, &wd, h, mi, act).unwrap(),
                true => be.upload_moe(&sg, Some(&su), &wd, h, mi, act).unwrap(),
            };
            let mut y = vec![0f32; t * h];
            assert!(be.moe(x, t, h, mi, &routes, &dev, &mut y));
            let fg: Vec<Vec<f32>> = wg.iter().map(|w| dense(w, 2 * mi, h)).collect();
            let fd: Vec<Vec<f32>> = wd.iter().map(|w| dense(w, h, mi)).collect();
            let mut r = vec![0f32; t * h];
            for (i, rt) in routes.iter().enumerate() {
                let xi = &x[i * h..(i + 1) * h];
                for &(e, wt) in rt {
                    let g = matvec(&fg[e], xi);
                    let a: Vec<f32> = (0..mi)
                        .map(|j| match act {
                            MoeAct::GeluTanh => crate::kernels::gelu_tanh(g[j]) * g[mi + j],
                            MoeAct::Silu => g[j] / (1.0 + (-g[j]).exp()) * g[mi + j],
                            MoeAct::SiluClamp(l) => {
                                let v = g[j].min(l);
                                v / (1.0 + (-v).exp()) * g[mi + j].clamp(-l, l)
                            }
                            MoeAct::SwigluOai { .. } | MoeAct::Situ { .. } => unreachable!("not a device activation"),
                        })
                        .collect();
                    let o = matvec(&fd[e], &a);
                    for (yv, ov) in r[i * h..(i + 1) * h].iter_mut().zip(&o) {
                        *yv += wt * ov;
                    }
                }
            }

            let e = row_err(&y, &r, h);
            assert!(e < 3e-3, "format {fmt} act {act:?} split {split}: relative error {e}");
        }
    }
}
