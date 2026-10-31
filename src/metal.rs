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
//!
//! Attention: unmasked heads of 64 (Whisper's encoder, SD 2.x / SD-Turbo) run as one fused
//! kernel (`flash_attn64`, f32 simdgroup MMA with an online softmax; ~2x faster than the
//! GEMM + softmax + GEMM path, which the other shapes keep, and no per-head operand repacking).
//!
//! On GPUs whose cores have matrix units (M5, A19: the Apple10 family), the GEMMs and the MoE
//! expert layers run on them instead (`MMA_SHADER_SRC`, Metal 4 `matmul2d`), with half-precision
//! operands and f32 accumulation: ~3-5x faster, at a relative error of ~3e-4 per product.
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
struct OP { uint o_off; uint gpr; uint gk; uint _p; };
kernel void gemm_q4a(device const void* qb [[buffer(0)]], device const void* sb [[buffer(1)]],
                     device const float* x [[buffer(2)]], device float* y [[buffer(3)]], constant P& p [[buffer(4)]],
                     device const ushort* ob [[buffer(5)]], constant OP& op [[buffer(6)]],
                     uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float Ws[64 * LD];
    Q4A w = { MK_Q4, ob + op.o_off, op.gpr, op.gk }; gemm_body<Q4A>(w, x, y, p, tg, tid, sg, Ws); }
CONV(conv_bf, BfW, MK_BF) CONV(conv_q8, Q8W, MK_Q8) CONV(conv_q4, Q4W, MK_Q4)

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
"#;

/// GGUF (ggml) block formats for the GEMM: one dequantizer per type producing the 32 values of
/// K block `b` of a row (a sub-block of a 256-element super-block for the K and i-quants). The
/// weights are the raw tensor bytes; `P::s_off` carries the row length in bytes. The i-quant grids
/// (`GGUF_TABLES`) are appended from `ggml_tables.rs` when the library is built.
const GGUF_SHADER_SRC: &str = r#"
struct GgW { device const uchar* q; uint rb; };
#define MK_GG(T) T{ { (device const uchar*)qb + p.w_off, p.s_off } }
inline float h2f(device const uchar* p) { return float(as_type<half>(ushort(ushort(p[0]) | (ushort(p[1]) << 8)))); }
inline uint rd16(device const uchar* p) { return uint(p[0]) | (uint(p[1]) << 8); }
inline uint rd32(device const uchar* p) { return uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) | (uint(p[3]) << 24); }
inline uchar4 ld4(device const uchar* p) { return uchar4(*(device const packed_uchar4*)p); }
inline float4 f4(uchar4 v) { return float4(v); }
inline float4 f4(char4 v) { return float4(v); }
inline device const uchar* sblk(GgW w, uint row, uint b, uint tsz) { return w.q + (ulong)row * w.rb + (ulong)(b / 8) * tsz; }
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
    device const uchar* B = w.g.q + (ulong)row * w.g.rb + b * 18;
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
    device const uchar* B = w.g.q + (ulong)row * w.g.rb + b * 18;
    float d = h2f(B);
    for (uint i = 0; i < 4; i++) {
        uchar4 v = ld4(B + 2 + 4 * i);
        out[i] = kv4(v & uchar(15)) * d;
        out[i + 4] = kv4(v >> uchar(4)) * d;
    }
}
struct Q8_0W { GgW g; };
inline void load(Q8_0W w, uint row, uint b, thread float4* out) {
    device const uchar* B = w.g.q + (ulong)row * w.g.rb + b * 34;
    float d = h2f(B);
    for (uint i = 0; i < 8; i++) out[i] = f4(as_type<char4>(ld4(B + 2 + 4 * i))) * d;
}
inline void scale_min_k4(uint j, device const uchar* s, thread float& sc, thread float& mn) {
    if (j < 4) { sc = float(s[j] & 63); mn = float(s[j + 4] & 63); }
    else { sc = float((s[j + 4] & 15) | ((s[j - 4] >> 6) << 4)); mn = float((s[j + 4] >> 4) | ((s[j] >> 6) << 4)); }
}
struct Q4_KW { GgW g; };
inline void load(Q4_KW w, uint row, uint b, thread float4* out) {
    device const uchar* B = sblk(w.g, row, b, 144);
    uint j = b % 8;
    float sc, mn; scale_min_k4(j, B + 4, sc, mn);
    float d1 = h2f(B) * sc, m1 = h2f(B + 2) * mn;
    device const uchar* q = B + 16 + 32 * (j / 2);
    uchar sh = uchar(4 * (j % 2));
    for (uint i = 0; i < 8; i++) out[i] = f4((ld4(q + 4 * i) >> sh) & uchar(15)) * d1 - m1;
}
struct Q5_KW { GgW g; };
inline void load(Q5_KW w, uint row, uint b, thread float4* out) {
    device const uchar* B = sblk(w.g, row, b, 176);
    uint j = b % 8;
    float sc, mn; scale_min_k4(j, B + 4, sc, mn);
    float d1 = h2f(B) * sc, m1 = h2f(B + 2) * mn;
    device const uchar* q = B + 48 + 32 * (j / 2);
    uchar sh = uchar(4 * (j % 2));
    for (uint i = 0; i < 8; i++) {
        uchar4 hb = ((ld4(B + 16 + 4 * i) >> uchar(j)) & uchar(1)) << uchar(4);
        out[i] = f4(((ld4(q + 4 * i) >> sh) & uchar(15)) | hb) * d1 - m1;
    }
}
struct Q6_KW { GgW g; };
inline void load(Q6_KW w, uint row, uint b, thread float4* out) {
    device const uchar* B = sblk(w.g, row, b, 210);
    uint j = b % 8, n = j / 4, jj = j % 4;
    float d = h2f(B + 208);
    device const uchar* ql = B + 64 * n + 32 * (jj & 1);
    device const uchar* qh = B + 128 + 32 * n;
    uchar lsh = uchar(4 * (jj >> 1)), hsh = uchar(2 * jj);
    for (uint i = 0; i < 8; i++) {
        float dl = d * float(as_type<char>(B[192 + 8 * n + 2 * jj + i / 4]));
        uchar4 q = ((ld4(ql + 4 * i) >> lsh) & uchar(15)) | (((ld4(qh + 4 * i) >> hsh) & uchar(3)) << uchar(4));
        out[i] = (f4(q) - 32.f) * dl;
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
inline void load(Q4_1W w, uint row, uint b, thread float4* out) { load_q45(w.g.q + (ulong)row * w.g.rb + b * 20, true, false, out); }
struct Q5_0W { GgW g; };
inline void load(Q5_0W w, uint row, uint b, thread float4* out) { load_q45(w.g.q + (ulong)row * w.g.rb + b * 22, false, true, out); }
struct Q5_1W { GgW g; };
inline void load(Q5_1W w, uint row, uint b, thread float4* out) { load_q45(w.g.q + (ulong)row * w.g.rb + b * 24, true, true, out); }
struct F16W { GgW g; };
inline void load(F16W w, uint row, uint b, thread float4* out) {
    device const packed_half4* p = (device const packed_half4*)(w.g.q + (ulong)row * w.g.rb + b * 64);
    for (uint i = 0; i < 8; i++) out[i] = float4(half4(p[i]));
}
struct F32W2 { GgW g; };
inline void load(F32W2 w, uint row, uint b, thread float4* out) {
    device const packed_float4* p = (device const packed_float4*)(w.g.q + (ulong)row * w.g.rb + b * 128);
    for (uint i = 0; i < 8; i++) out[i] = float4(p[i]);
}
#define GG(T) GEMM(gemm_##T, T##W, MK_GG(T##W))
GEMM(gemm_F32, F32W2, MK_GG(F32W2)) GG(F16) GG(Q4_1) GG(Q5_0) GG(Q5_1)
GG(Q4_0) GG(IQ4_NL) GG(Q8_0) GG(Q4_K) GG(Q5_K) GG(Q6_K) GG(Q3_K) GG(Q2_K) GG(IQ4_XS)
GG(IQ2_XXS) GG(IQ2_XS) GG(IQ2_S) GG(IQ3_XXS) GG(IQ3_S) GG(IQ1_S) GG(IQ1_M)
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
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
// The first four fields as in P (the MK_* macros read them), then the row count.
struct MP { uint n_out; uint n_in; uint w_off; uint s_off; uint rows; uint _p0; uint _p1; uint _p2; };
#define MKB 64
typedef tensor<device half, dextents<int32_t, 2>, tensor_inline> XT;
typedef tensor<device float, dextents<int32_t, 2>, tensor_inline> YT;

template <typename CT, typename DT>
inline void scale_store(thread CT& c, device const float* rs, uint t0, DT d) {
#pragma unroll
    for (uint16_t i = 0; i < c.get_capacity(); ++i)
        if (c.is_valid_element(i)) { auto idx = c.get_multidimensional_index(i); c[i] *= rs[t0 + idx[1]]; }
    c.store(d);
}
template <typename CT>
inline void zero(thread CT& c) {
#pragma unroll
    for (uint16_t i = 0; i < c.get_capacity(); ++i) if (c.is_valid_element(i)) c[i] = 0;
}
template <typename W, int TM>
inline void mma_tile(W w, device const half* xh, device const float* rs, device float* y, uint n_out, uint n_in, uint rows,
                     uint o0, uint t0, uint tid, threadgroup half* Ws) {
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
        float4 wv[8];
        load(w, o0 + row, k0 / 32 + blk, wv);
        for (uint i = 0; i < 8; i++) wd[i] = half4(wv[i]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto a = X.template slice<MKB, TM>(k0, t0);
        op.run(a, Wt, c);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    scale_store(c, rs, t0, Y.slice(o0, t0));
}
template <int TM, int TN>
inline void mma_bf_tile(device const bfloat* w, device const half* xh, device const float* rs, device float* y, uint n_out, uint n_in, uint rows,
                        uint o0, uint t0) {
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
    scale_store(c, rs, t0, Y.slice(o0, t0));
}
// Dense: 128-token tiles, and 32-token tiles (`_s`) for short inputs. (MAKE expands to text with
// commas, so it is only ever used directly in a body, never forwarded to another macro.)
#define MMA_SIG(NAME) \
kernel void NAME(device const void* qb [[buffer(0)]], device const void* sb [[buffer(1)]], device const half* xh [[buffer(2)]], \
                 device float* y [[buffer(3)]], constant MP& p [[buffer(4)]], device const float* rs [[buffer(5)]], \
                 uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]])
#define MMAD(NAME, WT, MAKE) \
MMA_SIG(NAME) { threadgroup half Ws[64 * MKB]; \
    WT w = MAKE; mma_tile<WT, 128>(w, xh, rs, y, p.n_out, p.n_in, p.rows, tg.x * 64, tg.y * 128, tid, Ws); } \
MMA_SIG(NAME##_s) { threadgroup half Ws[64 * MKB]; \
    WT w = MAKE; mma_tile<WT, 32>(w, xh, rs, y, p.n_out, p.n_in, p.rows, tg.x * 64, tg.y * 32, tid, Ws); }
// Grouped: per-expert weight addresses, and one tile per threadgroup row.
struct EP { device const uchar* q; device const uchar* s; };
struct GT { uint e; uint r0; uint rows; uint t0; };
#define MMAG(NAME, WT, MAKE) \
kernel void NAME(device const EP* ep [[buffer(0)]], device const GT* tiles [[buffer(1)]], device const half* xh [[buffer(2)]], \
                 device float* y [[buffer(3)]], constant MP& p [[buffer(4)]], device const float* rs [[buffer(5)]], \
                 uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup half Ws[64 * MKB]; \
    GT g = tiles[tg.y]; device const void* qb = ep[g.e].q; device const void* sb = ep[g.e].s; \
    WT w = MAKE; mma_tile<WT, 32>(w, xh + (ulong)g.r0 * p.n_in, rs + g.r0, y + (ulong)g.r0 * p.n_out, p.n_out, p.n_in, g.rows, tg.x * 64, g.t0, tid, Ws); }
MMAD(mma_q8, Q8W, MK_Q8) MMAD(mma_q4, Q4W, MK_Q4)
MMAG(mmag_q8, Q8W, MK_Q8) MMAG(mmag_q4, Q4W, MK_Q4)
#define MMA_Q4A(NAME, TM) \
kernel void NAME(device const void* qb [[buffer(0)]], device const void* sb [[buffer(1)]], device const half* xh [[buffer(2)]], \
                 device float* y [[buffer(3)]], constant MP& p [[buffer(4)]], device const float* rs [[buffer(5)]], \
                 device const ushort* ob [[buffer(6)]], constant OP& op [[buffer(7)]], \
                 uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup half Ws[64 * MKB]; \
    Q4A w = { MK_Q4, ob + op.o_off, op.gpr, op.gk }; \
    mma_tile<Q4A, TM>(w, xh, rs, y, p.n_out, p.n_in, p.rows, tg.x * 64, tg.y * TM, tid, Ws); }
MMA_Q4A(mma_q4a, 128) MMA_Q4A(mma_q4a_s, 32)
#define MMA_BF(NAME, TN, TM) \
kernel void NAME(device const ushort* qb [[buffer(0)]], device const half* xh [[buffer(2)]], device float* y [[buffer(3)]], \
                 constant MP& p [[buffer(4)]], device const float* rs [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]]) { \
    mma_bf_tile<TM, TN>((device const bfloat*)(qb + p.w_off), xh, rs, y, p.n_out, p.n_in, p.rows, tg.x * TN, tg.y * TM); }
MMA_BF(mma_bf64, 64, 128) MMA_BF(mma_bf128, 128, 128) MMA_BF(mma_bf64_s, 64, 32) MMA_BF(mma_bf128_s, 128, 32)
kernel void mmag_bf(device const EP* ep [[buffer(0)]], device const GT* tiles [[buffer(1)]], device const half* xh [[buffer(2)]],
                    device float* y [[buffer(3)]], constant MP& p [[buffer(4)]], device const float* rs [[buffer(5)]],
                    uint2 tg [[threadgroup_position_in_grid]]) {
    GT g = tiles[tg.y];
    mma_bf_tile<32, 64>((device const bfloat*)ep[g.e].q, xh + (ulong)g.r0 * p.n_in, rs + g.r0, y + (ulong)g.r0 * p.n_out,
                        p.n_out, p.n_in, g.rows, tg.x * 64, g.t0); }
#define GGM(T) MMAD(mma_##T, T##W, MK_GG(T##W)) MMAG(mmag_##T, T##W, MK_GG(T##W))
GGM(F16) GGM(Q4_1) GGM(Q5_0) GGM(Q5_1)
GGM(Q4_0) GGM(IQ4_NL) GGM(Q8_0) GGM(Q4_K) GGM(Q5_K) GGM(Q6_K) GGM(Q3_K) GGM(Q2_K) GGM(IQ4_XS)
GGM(IQ2_XXS) GGM(IQ2_XS) GGM(IQ2_S) GGM(IQ3_XXS) GGM(IQ3_S) GGM(IQ1_S) GGM(IQ1_M)

// Input rows (optionally gathered: row r is source row src[r]) to half with a power-of-two scale
// per row (see above); rs gets the inverse scale. One threadgroup of 256 per row; n % 4 == 0.
struct XH { uint n; uint gather; };
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
    device const float4* row = (device const float4*)(x + (ulong)(p.gather != 0 ? src[r] : r) * p.n);
    uint n4 = p.n / 4;
    float m = 0.f;
    for (uint i = tid; i < n4; i += 256) { float4 v = abs(row[i]); m = max(m, max(max(v.x, v.y), max(v.z, v.w))); }
    m = tg_max(m, sg, lane, red);
    float inv, s = row_scale(m, inv);
    device half4* out = (device half4*)(xh + (ulong)r * p.n);
    for (uint i = tid; i < n4; i += 256) out[i] = half4(row[i] * s);
    if (tid == 0) rs[r] = inv;
}
// act(gate) * up (SiLU, or GELU-tanh with `gelu`) of fused [gate | up] rows (n each) into scaled
// half rows, as x_half; with `split` = G nonzero, gate row r is at r * n and up row r at
// (G + r) * n (gate and up experts stored apart).
inline float4 silu_mul4(float4 g, float4 u) { return g / (1.0f + exp(-g)) * u; }
inline float4 gelu_mul4(float4 g, float4 u) { return g / (1.0f + exp(-1.5957691f * (g + 0.044715f * g * g * g))) * u; }
inline float4 act_mul4(float4 g, float4 u, uint gelu) { return gelu != 0 ? gelu_mul4(g, u) : silu_mul4(g, u); }
struct SH { uint n; uint gelu; uint split; uint _p; };
kernel void silu_half(device const float* gu [[buffer(0)]], device half* ah [[buffer(1)]], device float* rs [[buffer(2)]],
                      constant SH& p [[buffer(3)]],
                      uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8];
    device const float4* g = (device const float4*)(gu + (ulong)r * (p.split != 0 ? 1 : 2) * p.n);
    device const float4* u = p.split != 0 ? (device const float4*)(gu + (ulong)(p.split + r) * p.n) : g + p.n / 4;
    uint n4 = p.n / 4;
    float m = 0.f;
    for (uint i = tid; i < n4; i += 256) { float4 v = abs(act_mul4(g[i], u[i], p.gelu)); m = max(m, max(max(v.x, v.y), max(v.z, v.w))); }
    m = tg_max(m, sg, lane, red);
    float inv, s = row_scale(m, inv);
    device half4* out = (device half4*)(ah + (ulong)r * p.n);
    for (uint i = tid; i < n4; i += 256) out[i] = half4(act_mul4(g[i], u[i], p.gelu) * s);
    if (tid == 0) rs[r] = inv;
}
// out[t] = sum over the token's k slots of weight * o[row].
struct CB { uint n; uint k; };
kernel void moe_combine(device const float* o [[buffer(0)]], device const float2* slots [[buffer(1)]], device float* out [[buffer(2)]],
                        constant CB& p [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    uint c = gid.x * 4, t = gid.y;
    if (c >= p.n) return;
    float4 acc = 0.f;
    for (uint j = 0; j < p.k; j++) {
        float2 s = slots[t * p.k + j];
        acc += s.y * *(device const float4*)(o + (ulong)as_type<uint>(s.x) * p.n + c);
    }
    *(device float4*)(out + (ulong)t * p.n + c) = acc;
}
"#;

/// GGUF types with a Metal GEMM (`gemm_<TYPE>` in `GGUF_SHADER_SRC`).
const GGUF_GPU_TYPES: [GgmlType; 21] = [
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
];

/// GGUF types with a matrix-unit GEMM (`mma_<TYPE>` / `mmag_<TYPE>` in `MMA_SHADER_SRC`): all but
/// F32, whose values would lose precision as half.
const GGUF_MMA_TYPES: [GgmlType; 20] = [
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
];

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

/// Which matrix-unit GEMM: dense with 128- or 32-token tiles, or grouped (MoE).
#[derive(Clone, Copy, PartialEq)]
enum Tile {
    Dense,
    Small,
    Grouped,
}

/// Matrix-unit pipelines (GPU cores with neural accelerators: M5, A19).
struct Mma {
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
}

impl Drop for MetalWeight {
    fn drop(&mut self) {
        let (f, sel) = self.release;
        let ids = [Some(&self.data), self.scales.as_ref(), self.offsets.as_ref().map(|o| &o.0)];
        for b in ids.into_iter().flatten() {
            // SAFETY: each buffer was created by this weight and is released once.
            unsafe { f(b.id, sel) };
        }
    }
}

/// An `MTLResidencySet` attached to the command queue, and its selectors.
struct Residency {
    set: Id,
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
    /// Adds (or removes) buffers, then commits the change.
    fn update(&self, ids: &[Id], add: bool) {
        let _g = self.lock.lock().unwrap();
        // SAFETY: `set` is a live residency set and `ids` are live buffers.
        unsafe {
            for &id in ids {
                (self.send_id)(self.set, if add { self.add } else { self.remove }, id);
            }
            (self.send)(self.set, self.commit);
            if add {
                (self.send)(self.set, self.request);
            }
        }
    }
}

/// Device state of an MoE layer's routed experts (`Backend::upload_moe`).
struct MoeDev {
    /// Grouped GEMM pipelines for the gate/up and the down projections.
    pipes: (Id, Id),
    /// GGUF row bytes of each (the `s_off` parameter), 0 for other formats.
    row_bytes: (u32, u32),
    /// The wrapped weight memory (adjacent experts share a buffer).
    bufs: Vec<Buf>,
    /// Per-expert (data, scales) GPU addresses of each projection, for `n_exp` experts.
    ep: (Id, Id),
    /// Separate up projections (GGUF): their pipeline, row bytes and address table; `pipes.0`
    /// and `ep.0` are then the gate projections alone.
    up: Option<(Id, u32, Id)>,
    n_exp: usize,
    /// GELU-tanh experts (Gemma 4) rather than SiLU.
    gelu: bool,
    /// The residency set holding `bufs`, if any (else they are made resident per submission).
    residency: Option<std::sync::Arc<Residency>>,
    release: (MsgVoid, Sel),
}

impl Drop for MoeDev {
    fn drop(&mut self) {
        if let Some(r) = &self.residency {
            r.update(&self.bufs.iter().map(|b| b.id).collect::<Vec<_>>(), false);
        }
        let (f, sel) = self.release;
        let ids = self
            .bufs
            .iter()
            .map(|b| b.id)
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
        let base_src = format!("{SHADER_SRC}{}{GGUF_SHADER_SRC}{}", gguf_shader_tables(), decode::shader_src());
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
        let (lib, has_mma, mma_error) = match want_mma {
            true => match compile(&format!("{base_src}{MMA_SHADER_SRC}"), true) {
                Ok(l) => (l, true, None),
                Err(e) => {
                    if apple10 || mma_env.is_some() || std::env::var_os("OXINFER_DEBUG").is_some() {
                        eprintln!("[metal: matrix-unit kernels unavailable: {e}]");
                    }
                    (compile(&base_src, false)?, false, Some(e.to_string()))
                }
            },
            false => (compile(&base_src, false)?, false, None),
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
            .map(|&k| Ok((k, pipeline(&CString::new(format!("gemm_{}", k.name()))?)?)))
            .collect::<Result<Vec<_>>>()?;
        let group_stats = pipeline(c"group_stats")?;
        let row_stats = pipeline(c"row_stats")?;
        let softmax = pipeline(c"softmax_rows")?;
        let flash64 = pipeline(c"flash_attn64")?;
        let mma = if has_mma {
            let gg = |prefix: &str| {
                GGUF_MMA_TYPES
                    .iter()
                    .map(|&k| Ok((k, pipeline(&CString::new(format!("{prefix}_{}", k.name()))?)?)))
                    .collect::<Result<Vec<_>>>()
            };
            Some(Mma {
                x_half: pipeline(c"x_half")?,
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
            let mut err: Id = std::ptr::null_mut();
            let set = unsafe { new_set(device, new_sel, desc, &mut err) };
            let void: MsgVoid = o.msg_send();
            unsafe { void(desc, o.sel(c"release")) };
            if set.is_null() {
                return None;
            }
            let add: unsafe extern "C" fn(Id, Sel, Id) = o.msg_send();
            unsafe { add(queue, o.sel(c"addResidencySet:"), set) };
            Some(std::sync::Arc::new(Residency {
                set,
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
                "[metal: library and pipelines ready in {:.0} ms (matrix units: {has_mma})]",
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
            return None;
        }
        m.insert(name.to_string(), p);
        Some(p)
    }

    /// Wraps host memory as a shared buffer without copying (the enclosing pages are mapped; the
    /// caller guarantees they belong to one allocation); falls back to copying. The iOS
    /// Simulator's Metal driver aborts on wrapped host memory (it hands buffers to the host's GPU
    /// process as shared memory), so it always copies there.
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
    /// Unmasked attention with 64-dimensional heads in one fused kernel (`flash_attn64`): the
    /// operands keep the model's `[rows][heads * 64]` layout, so staging is three plain copies.
    #[allow(clippy::too_many_arguments)]
    fn flash_attention(&self, q: &[f32], k: &[f32], v: &[f32], tq: usize, tk: usize, heads: usize, out: &mut [f32]) -> bool {
        let d = heads * 64;
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
            scale: 0.125,
        };
        let pipe = self.flash64;
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

    fn contents(&self, b: Id) -> *mut u8 {
        let f: MsgPtr = self.o.msg_send();
        unsafe { f(b, self.sels.contents) as *mut u8 }
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
        let find = |v: &[(GgmlType, Id)], k: GgmlType| v.iter().find(|(t, _)| *t == k).map(|(_, p)| *p);
        let small = tile == Tile::Small;
        let pick = |big: Id, s: Id| if small { s } else { big };
        Some(match (w, tile) {
            (WeightRef::Bf16(_), Tile::Grouped) => (m.g_bf, 64),
            (WeightRef::Bf16(_), _) if n_out.is_multiple_of(128) => (pick(m.bf[1], m.bf_s[1]), 128),
            (WeightRef::Bf16(_), _) => (pick(m.bf[0], m.bf_s[0]), 64),
            (WeightRef::Q8(_), Tile::Grouped) => (m.g_q8, 64),
            (WeightRef::Q8(_), _) => (pick(m.q8, m.q8_s), 64),
            (WeightRef::Q4(_), Tile::Grouped) if !affine => (m.g_q4, 64),
            (WeightRef::Q4(_), Tile::Grouped) => return None,
            (WeightRef::Q4(_), _) if affine => (pick(m.q4a, m.q4a_s), 64),
            (WeightRef::Q4(_), _) => (pick(m.q4, m.q4_s), 64),
            (WeightRef::Gguf(g), Tile::Grouped) => (find(&m.g_gguf, g.kind)?, 64),
            (WeightRef::Gguf(g), _) => (pick(find(&m.gguf, g.kind)?, find(&m.gguf_s, g.kind)?), 64),
        })
    }

    /// Encodes the conversion of `rows` rows of `n` floats (row r taken from row `src[r]` if
    /// `src` is given) to scaled half rows plus their inverse scales.
    #[allow(clippy::too_many_arguments)]
    fn encode_x_half(&self, e: &Enc, x: Id, xh: Id, rs: Id, n: usize, rows: usize, src: Option<Id>) {
        #[repr(C)]
        struct Xh {
            n: u32,
            gather: u32,
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
            },
            3,
        );
        e.buf(src.unwrap_or(x), 0, 4);
        e.dispatch((rows, 1), 256);
    }

    /// `Backend::matmul` on the matrix units; None when some job has no such kernel.
    fn matmul_mma(&self, x: &[f32], t: usize, n_in: usize, jobs: &mut [Job]) -> Option<bool> {
        // Rows padded to whole token tiles (zero rows; partial tiles take a slower path).
        let (tile, tm) = if t <= MMA_SMALL {
            (Tile::Small, MMA_TM_S)
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
                    WeightRef::Gguf(m) => m.row_bytes(),
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

impl Backend for MetalBackend {
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
                MetalWeight {
                    data: self.wrap(m.data.as_ptr(), m.data.len())?,
                    scales: None,
                    offsets: None,
                    release,
                }
            }
            WeightRef::Bf16(w) => MetalWeight {
                data: self.wrap(w.as_ptr() as *const u8, w.len() * 2)?,
                scales: None,
                offsets: None,
                release,
            },
            WeightRef::Q8(m) => MetalWeight {
                data: self.wrap(m.q.as_ptr() as *const u8, m.q.len())?,
                scales: Some(self.wrap(m.scales.as_ptr() as *const u8, m.scales.len() * 4)?),
                offsets: None,
                release,
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
            },
        };
        Some(Box::new(mw))
    }

    fn conv2d(&self, x: &[f32], g: &crate::gpu::ConvGeom, norm: Option<&crate::gpu::NormSpec>, job: &mut Job) -> bool {
        use crate::gpu::NormKind;
        if matches!(job.w, WeightRef::Gguf(_)) {
            return false;
        }
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
            let (fi, elem_bytes, scale_bytes) = format_index(&job.w);
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
            set_pipe(enc, self.sels.set_pipeline, self.conv_pipes[fi]);
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
        if dh == 64 && !causal {
            return self.flash_attention(q, k, v, tq, tk, heads, out);
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

    fn decode_plan<'s>(&'s self, g: &crate::gpu::decode::Graph) -> Option<Box<dyn crate::gpu::decode::DecodePlan + 's>> {
        self.build_plan(g)
            .map(|p| Box::new(p) as Box<dyn crate::gpu::decode::DecodePlan + 's>)
    }

    fn upload_moe(
        &self,
        gate_up: &[WeightRef],
        up: Option<&[WeightRef]>,
        down: &[WeightRef],
        h: usize,
        mi: usize,
        gelu: bool,
    ) -> Option<DeviceWeight> {
        self.mma()?;
        if !crate::gpu::moe_offload() || gate_up.is_empty() || gate_up.len() != down.len() || up.is_some_and(|u| u.len() != down.len()) {
            return None;
        }
        let affine = |w: &WeightRef| matches!(w, WeightRef::Q4(m) if m.offsets.is_some());
        // Rows of the first projection: fused gate and up, or the gate alone.
        let g_rows = if up.is_some() { mi } else { 2 * mi };
        let (pg, _) = self.mma_pipe(&gate_up[0], affine(&gate_up[0]), g_rows, h, Tile::Grouped)?;
        let pu = match up {
            Some(u) => Some(self.mma_pipe(&u[0], affine(&u[0]), mi, h, Tile::Grouped)?.0),
            None => None,
        };
        let (pd, _) = self.mma_pipe(&down[0], affine(&down[0]), h, mi, Tile::Grouped)?;
        let row_bytes = |w: &WeightRef| match w {
            WeightRef::Gguf(g) => g.row_bytes(),
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
                WeightRef::Gguf(m) => (span(&m.data[..]), None),
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
            pipes: (pg, pd),
            row_bytes: (row_bytes(&gate_up[0]) as u32, row_bytes(&down[0]) as u32),
            bufs: Vec::with_capacity(spans.len()),
            ep: (std::ptr::null_mut(), std::ptr::null_mut()),
            up: None,
            n_exp: gate_up.len(),
            gelu,
            residency: None,
            release,
        };
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
        if let (Some(u), Some(pu)) = (up, pu) {
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
        if let Some(r) = &self.residency {
            r.update(&dev.bufs.iter().map(|b| b.id).collect::<Vec<_>>(), true);
            dev.residency = Some(r.clone());
        }
        Some(Box::new(dev))
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
        // to whole tiles of MMA_GTM rows (partial tiles take a slower path in the tensor ops;
        // padding rows come from a zero row after the input and are never read back).
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
        let mut start = vec![0usize; count.len()];
        let mut tiles: Vec<[u32; 4]> = Vec::new();
        let mut rows = 0usize;
        for (e, &c) in count.iter().enumerate() {
            start[e] = rows;
            let len = c.next_multiple_of(MMA_GTM);
            for t0 in (0..len).step_by(MMA_GTM) {
                tiles.push([e as u32, rows as u32, len as u32, t0 as u32]);
            }
            rows += len;
        }
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
        let pad = rows + MMA_GTM;
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
            let grouped = |pipe: Id, ep: Id, xh: Id, out: (Id, usize), rs: Id, p: &MmaParams| {
                e.pipe(pipe);
                e.buf(ep, 0, 0);
                e.buf(st.m_tiles.id, 0, 1);
                e.buf(xh, 0, 2);
                e.buf(out.0, out.1, 3);
                e.bytes(p, 4);
                e.buf(rs, 0, 5);
                e.dispatch((p.n_out as usize / 64, tiles.len()), 128);
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
                _p: u32,
            }
            e.pipe(m.silu_half);
            e.buf(st.m_gu.id, 0, 0);
            e.buf(st.m_ah.id, 0, 1);
            e.buf(st.m_rs2.id, 0, 2);
            e.bytes(
                &Sh {
                    n: mi as u32,
                    gelu: md.gelu as u32,
                    split: if md.up.is_some() { rows as u32 } else { 0 },
                    _p: 0,
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
            }
            e.pipe(m.combine);
            e.buf(st.m_o.id, 0, 0);
            e.buf(st.m_slots.id, 0, 1);
            e.buf(st.y, 0, 2);
            e.bytes(&Cb { n: h as u32, k: k as u32 }, 3);
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
                    WeightRef::Gguf(m) => (m.row_bytes(), self.gguf_pipe(m.kind).unwrap()),
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
        let gg: Vec<_> = [GgmlType::Q4_K, GgmlType::Q6_K, GgmlType::IQ4_XS]
            .into_iter()
            .map(|k| {
                let raw = crate::ggml::random_blocks(k, n_out * n_in / k.block_size(), &mut seed);
                crate::ggml::GgufMatrix::from_bytes(k, &raw, n_out, n_in).unwrap()
            })
            .collect();
        let mut ws = vec![WeightRef::Bf16(&w), WeightRef::Q8(&q8), WeightRef::Q4(&q4)];
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
            _ => unimplemented!(),
        }
        f
    }
    fn matvec(w: &[f32], x: &[f32]) -> Vec<f32> {
        w.chunks(x.len())
            .map(|r| r.iter().zip(x).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() as f32)
            .collect()
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
        let x = input(t, h, &mut seed);
        // Routes: expert 5 unused, the others with uneven loads.
        let routes: Vec<Vec<(usize, f32)>> = (0..t)
            .map(|i| vec![([0, 1, 2, 3, 4, 6, 7][i % 7], 0.7), ([1, 7][i % 2], 0.3)])
            .collect();
        // bf16 / Q4 fused, SiLU / GELU; bf16 with gate and up apart (GGUF layout).
        for (q4, gelu, split) in [
            (false, false, false),
            (true, false, false),
            (false, true, false),
            (true, true, false),
            (false, false, true),
        ] {
            let (wg, wd): (Vec<WeightRef>, Vec<WeightRef>) = (0..ne)
                .map(|e| match q4 {
                    false => (
                        WeightRef::Bf16(&gu[e * 2 * mi * h..(e + 1) * 2 * mi * h]),
                        WeightRef::Bf16(&dn[e * h * mi..(e + 1) * h * mi]),
                    ),
                    true => (WeightRef::Q4(&gu4[e]), WeightRef::Q4(&dn4[e])),
                })
                .unzip();
            let half = |e: usize, i: usize| WeightRef::Bf16(&gu[e * 2 * mi * h + i * mi * h..e * 2 * mi * h + (i + 1) * mi * h]);
            let (sg, su): (Vec<WeightRef>, Vec<WeightRef>) = (0..ne).map(|e| (half(e, 0), half(e, 1))).unzip();
            let dev = match split {
                false => be.upload_moe(&wg, None, &wd, h, mi, gelu).unwrap(),
                true => be.upload_moe(&sg, Some(&su), &wd, h, mi, gelu).unwrap(),
            };
            let mut y = vec![0f32; t * h];
            assert!(be.moe(&x, t, h, mi, &routes, &dev, &mut y));
            let fg: Vec<Vec<f32>> = wg.iter().map(|w| dense(w, 2 * mi, h)).collect();
            let fd: Vec<Vec<f32>> = wd.iter().map(|w| dense(w, h, mi)).collect();
            let mut r = vec![0f32; t * h];
            for (i, rt) in routes.iter().enumerate() {
                let xi = &x[i * h..(i + 1) * h];
                for &(e, wt) in rt {
                    let g = matvec(&fg[e], xi);
                    let act = |v: f32| {
                        if gelu {
                            crate::kernels::gelu_tanh(v)
                        } else {
                            v / (1.0 + (-v).exp())
                        }
                    };
                    let a: Vec<f32> = (0..mi).map(|j| act(g[j]) * g[mi + j]).collect();
                    let o = matvec(&fd[e], &a);
                    for (yv, ov) in r[i * h..(i + 1) * h].iter_mut().zip(&o) {
                        *yv += wt * ov;
                    }
                }
            }

            let e = row_err(&y, &r, h);
            assert!(e < 3e-3, "q4 {q4} gelu {gelu} split {split}: relative error {e}");
        }
    }
}
