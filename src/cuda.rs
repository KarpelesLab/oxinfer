//! CUDA backend (Linux / Windows, NVIDIA): the driver API and NVRTC are loaded at run time with
//! `dlsym`, so the binary has no CUDA dependency and runs unchanged on machines without a GPU.
//! The GEMM kernels are compiled from source by NVRTC for the detected compute capability.
//!
//! Only prefill-sized work is offloaded (see `wants`): matmuls, convolutions, attention, and the
//! routed experts of MoE layers (gathered per expert, one GEMM pair per expert, combined on the
//! device; a layer's experts go up all at once or not at all). Weights are copied to device memory
//! the first time a layer is used (skipped when VRAM is short, that layer then stays on the CPU),
//! activations are copied up and results down per batch. Discrete GPUs pay for those transfers,
//! so the threshold is higher than Metal's. Whole decode steps (single tokens, and the
//! multi-token passes of speculative verification) run as compiled plans (`cuda_decode.rs`).
//!
//! Every kernel is tested against references from the definitions (`cargo test --lib cuda::` on a
//! machine with an NVIDIA GPU) and has run on real models on an RTX 5080; see the README.

use crate::gpu::{Backend, DeviceWeight, Job, WeightRef};
use anyhow::{Result, anyhow, bail};
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

#[path = "cuda_decode.rs"]
mod decode;

const TILE: usize = 64;
const MIN_T: usize = 32;
const MIN_FLOPS: f64 = 1e9;
const VRAM_RESERVE: usize = 512 << 20;

dlsym::library! {
    pub struct Cu {
        unsafe extern "C" {
            pub fn cuInit(flags: u32) -> c_int;
            pub fn cuDeviceGet(dev: *mut c_int, ordinal: c_int) -> c_int;
            pub fn cuDeviceGetAttribute(v: *mut c_int, attr: c_int, dev: c_int) -> c_int;
            pub fn cuDevicePrimaryCtxRetain(ctx: *mut *mut c_void, dev: c_int) -> c_int;
            pub fn cuCtxSetCurrent(ctx: *mut c_void) -> c_int;
            pub fn cuModuleLoadData(module: *mut *mut c_void, image: *const c_void) -> c_int;
            pub fn cuModuleGetFunction(f: *mut *mut c_void, module: *mut c_void, name: *const c_char) -> c_int;
            pub fn cuMemAlloc_v2(ptr: *mut u64, bytes: usize) -> c_int;
            pub fn cuMemFree_v2(ptr: u64) -> c_int;
            pub fn cuMemcpyHtoD_v2(dst: u64, src: *const c_void, bytes: usize) -> c_int;
            pub fn cuMemcpyDtoH_v2(dst: *mut c_void, src: u64, bytes: usize) -> c_int;
            pub fn cuMemcpyDtoD_v2(dst: u64, src: u64, bytes: usize) -> c_int;
            pub fn cuMemsetD8_v2(dst: u64, value: u8, bytes: usize) -> c_int;
            pub fn cuMemAllocHost_v2(p: *mut *mut c_void, bytes: usize) -> c_int;
            pub fn cuMemFreeHost(p: *mut c_void) -> c_int;
            pub fn cuStreamCreate(stream: *mut *mut c_void, flags: u32) -> c_int;
            pub fn cuStreamSynchronize(stream: *mut c_void) -> c_int;
            pub fn cuStreamWaitEvent(stream: *mut c_void, event: *mut c_void, flags: u32) -> c_int;
            pub fn cuEventCreate(event: *mut *mut c_void, flags: u32) -> c_int;
            pub fn cuEventRecord(event: *mut c_void, stream: *mut c_void) -> c_int;
            pub fn cuMemcpyDtoHAsync_v2(dst: *mut c_void, src: u64, bytes: usize, stream: *mut c_void) -> c_int;
            pub fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> c_int;
            pub fn cuLaunchKernel(
                f: *mut c_void, gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32,
                shmem: u32, stream: *mut c_void, params: *mut *mut c_void, extra: *mut *mut c_void,
            ) -> c_int;
            pub fn cuCtxSynchronize() -> c_int;
            #[optional]
            pub fn cuGetErrorString(err: c_int, s: *mut *const c_char) -> c_int;
        }
    }
}

dlsym::library! {
    pub struct Nvrtc {
        unsafe extern "C" {
            pub fn nvrtcCreateProgram(
                prog: *mut *mut c_void, src: *const c_char, name: *const c_char,
                n_headers: c_int, headers: *const *const c_char, include_names: *const *const c_char,
            ) -> c_int;
            pub fn nvrtcCompileProgram(prog: *mut c_void, n_opts: c_int, opts: *const *const c_char) -> c_int;
            pub fn nvrtcGetPTXSize(prog: *mut c_void, size: *mut usize) -> c_int;
            pub fn nvrtcGetPTX(prog: *mut c_void, ptx: *mut c_char) -> c_int;
            #[optional]
            pub fn nvrtcGetCUBINSize(prog: *mut c_void, size: *mut usize) -> c_int;
            #[optional]
            pub fn nvrtcGetCUBIN(prog: *mut c_void, cubin: *mut c_char) -> c_int;
            pub fn nvrtcGetProgramLogSize(prog: *mut c_void, size: *mut usize) -> c_int;
            pub fn nvrtcGetProgramLog(prog: *mut c_void, log: *mut c_char) -> c_int;
            pub fn nvrtcDestroyProgram(prog: *mut *mut c_void) -> c_int;
        }
    }
}

const NVRTC_CANDIDATES: &[&str] = &[
    "libnvrtc.so",
    "libnvrtc.so.13",
    "libnvrtc.so.12",
    "libnvrtc.so.11.2",
    "nvrtc64_130_0.dll",
    "nvrtc64_120_0.dll",
    "nvrtc64_112_0.dll",
];

/// Tiled f32 GEMM: a 256-thread block computes 64 (out rows) x 64 (tokens); per 32-wide K block the
/// weights are dequantized into shared memory (transposed, `[k][row]`) next to the x tile
/// (`[k][tok]`), then each thread accumulates a 4x4 register tile with float4 shared loads.
/// The host pads the token count to a multiple of 64.
const KERNEL_SRC: &str = r#"
struct Q4W { const unsigned char* q; const unsigned short* s; };
struct Q8W { const signed char* q; const float* s; };
struct BfW { const unsigned short* w; };

__device__ __forceinline__ float bf16f(unsigned short u) { return __uint_as_float(((unsigned int)u) << 16); }
__device__ __forceinline__ float f16f(unsigned short h) {
    unsigned int e = (h >> 10) & 31, m = h & 1023, sign = ((unsigned int)(h & 0x8000)) << 16;
    if (e == 31) return __uint_as_float(sign | 0x7f800000u | (m << 13));
    if (e == 0) return __uint_as_float(sign | __float_as_uint((float)m * 5.9604644775390625e-8f));
    return __uint_as_float(sign | ((e + 112) << 23) | (m << 13));
}

// 8 consecutive values (k0 .. k0+8, k0 % 8 == 0) of block b of one row.
__device__ __forceinline__ void load8(Q4W w, unsigned int n_in, unsigned int row, unsigned int b, unsigned int k0, float* out) {
    float d = f16f(w.s[(size_t)row * (n_in / 32) + b]);
    const unsigned char* blk = w.q + (size_t)row * (n_in / 2) + b * 16;
    if (k0 < 16) { for (int i = 0; i < 8; i++) out[i] = (float)((int)(blk[k0 + i] & 15) - 8) * d; }
    else { for (int i = 0; i < 8; i++) out[i] = (float)((int)(blk[k0 - 16 + i] >> 4) - 8) * d; }
}
// Affine Q4 (MLX): a bf16 offset per group of `gk` blocks is added to every element.
struct Q4A { Q4W b; const unsigned short* o; unsigned int gpr; unsigned int gk; };
__device__ __forceinline__ void load8(Q4A w, unsigned int n_in, unsigned int row, unsigned int b, unsigned int k0, float* out) {
    load8(w.b, n_in, row, b, k0, out);
    float m = bf16f(w.o[(size_t)row * w.gpr + b / w.gk]);
    for (int i = 0; i < 8; i++) out[i] += m;
}
__device__ __forceinline__ void load8(Q8W w, unsigned int n_in, unsigned int row, unsigned int b, unsigned int k0, float* out) {
    float d = w.s[(size_t)row * (n_in / 32) + b];
    const signed char* p = w.q + (size_t)row * n_in + b * 32 + k0;
    for (int i = 0; i < 8; i++) out[i] = (float)p[i] * d;
}
struct F32W { const float* w; };
__device__ __forceinline__ void load8(F32W w, unsigned int n_in, unsigned int row, unsigned int b, unsigned int k0, float* out) {
    const float4* p = (const float4*)(w.w + (size_t)row * n_in + b * 32 + k0);
    float4 a = p[0], c = p[1];
    out[0] = a.x; out[1] = a.y; out[2] = a.z; out[3] = a.w; out[4] = c.x; out[5] = c.y; out[6] = c.z; out[7] = c.w;
}
__device__ __forceinline__ void load8(BfW w, unsigned int n_in, unsigned int row, unsigned int b, unsigned int k0, float* out) {
    const unsigned short* p = w.w + (size_t)row * n_in + b * 32 + k0;
    for (int i = 0; i < 8; i++) out[i] = bf16f(p[i]);
}

// Activation loaders: 8 consecutive k values of one token row of block b.
struct XDense {
    const float* x; unsigned int n_in;
    __device__ __forceinline__ void load8(unsigned int tok, unsigned int b, unsigned int k0, float* out) const {
        const float4* p = (const float4*)(x + (size_t)tok * n_in + b * 32 + k0);
        float4 a = p[0], c = p[1];
        out[0] = a.x; out[1] = a.y; out[2] = a.z; out[3] = a.w; out[4] = c.x; out[5] = c.y; out[6] = c.z; out[7] = c.w;
    }
};
// Convolution as an implicit GEMM: K runs over (ky, kx, ci) with cin % 32 == 0, so block b is one
// tap's contiguous channel slice of the input pixel (zeros outside the image).
// mode: 0 plain, 1 group norm (stats per group of cpg channels), 2 layer norm (per input pixel);
// st holds (mean, 1/std), gb holds gamma[cin] then beta[cin].
struct ConvP { unsigned int h, w, cin, k, stride, ow, npix, pad, mode, silu, cpg, _p; };
struct XConv {
    const float* x; ConvP p; const float2* st; const float* gb;
    __device__ __forceinline__ void load8(unsigned int tok, unsigned int b, unsigned int k0, float* out) const {
        unsigned int kk = b * 32, tap = kk / p.cin, ci0 = kk - tap * p.cin;
        int ky = (int)(tap / p.k), kx = (int)(tap % p.k);
        int oy = (int)(tok / p.ow), ox = (int)(tok % p.ow);
        int iy = oy * (int)p.stride + ky - (int)p.pad, ix = ox * (int)p.stride + kx - (int)p.pad;
        if (tok < p.npix && iy >= 0 && ix >= 0 && iy < (int)p.h && ix < (int)p.w) {
            const float4* s = (const float4*)(x + ((size_t)iy * p.w + ix) * p.cin + ci0 + k0);
            float4 a = s[0], c = s[1];
            out[0] = a.x; out[1] = a.y; out[2] = a.z; out[3] = a.w; out[4] = c.x; out[5] = c.y; out[6] = c.z; out[7] = c.w;
            if (p.mode != 0) {
                unsigned int pin = (unsigned int)iy * p.w + (unsigned int)ix;
                for (int i = 0; i < 8; i++) {
                    unsigned int ch = ci0 + k0 + i;
                    float2 m = p.mode == 1 ? st[ch / p.cpg] : st[pin];
                    float yv = (out[i] - m.x) * m.y * gb[ch] + gb[p.cin + ch];
                    out[i] = p.silu ? yv / (1.f + __expf(-yv)) : yv;
                }
            }
        } else {
            for (int i = 0; i < 8; i++) out[i] = 0.f;
        }
    }
};

template <typename W, typename X>
__device__ __forceinline__ void gemm_body(W w, X xl, float* __restrict__ y, unsigned int n_out, unsigned int n_in) {
    __shared__ __align__(16) float Ws[32][64];
    __shared__ __align__(16) float Xs[32][64];
    const unsigned int o0 = blockIdx.x * 64, t0 = blockIdx.y * 64;
    const unsigned int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
    const unsigned int lrow = tid / 4, lk = (tid % 4) * 8;
    float acc[4][4];
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) acc[i][j] = 0.f;
    for (unsigned int b = 0; b < n_in / 32; b++) {
        float wv[8];
        load8(w, n_in, o0 + lrow, b, lk, wv);
        for (int i = 0; i < 8; i++) Ws[lk + i][lrow] = wv[i];
        float xv[8];
        xl.load8(t0 + lrow, b, lk, xv);
        for (int i = 0; i < 8; i++) Xs[lk + i][lrow] = xv[i];
        __syncthreads();
#pragma unroll
        for (int k = 0; k < 32; k++) {
            float4 wr = *(const float4*)&Ws[k][ty * 4];
            float4 xr = *(const float4*)&Xs[k][tx * 4];
            float wa[4] = { wr.x, wr.y, wr.z, wr.w };
            float xa[4] = { xr.x, xr.y, xr.z, xr.w };
            for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) acc[i][j] += wa[i] * xa[j];
        }
        __syncthreads();
    }
    for (int j = 0; j < 4; j++) {
        float* yp = y + (size_t)(t0 + tx * 4 + j) * n_out + o0 + ty * 4;
        *(float4*)yp = make_float4(acc[0][j], acc[1][j], acc[2][j], acc[3][j]);
    }
}

extern "C" __global__ void gemm_q4(const unsigned char* q, const unsigned short* s, const float* x, float* y, unsigned int n_out, unsigned int n_in) {
    Q4W w = { q, s }; XDense xl = { x, n_in }; gemm_body(w, xl, y, n_out, n_in);
}
extern "C" __global__ void gemm_q4a(const unsigned char* q, const unsigned short* s, const float* x, float* y, unsigned int n_out, unsigned int n_in, const unsigned short* o, unsigned int gpr, unsigned int gk) {
    Q4A w = { { q, s }, o, gpr, gk }; XDense xl = { x, n_in }; gemm_body(w, xl, y, n_out, n_in);
}
extern "C" __global__ void gemm_q8(const signed char* q, const float* s, const float* x, float* y, unsigned int n_out, unsigned int n_in) {
    Q8W w = { q, s }; XDense xl = { x, n_in }; gemm_body(w, xl, y, n_out, n_in);
}
extern "C" __global__ void gemm_bf(const unsigned short* wp, const void* unused, const float* x, float* y, unsigned int n_out, unsigned int n_in) {
    BfW w = { wp }; XDense xl = { x, n_in }; gemm_body(w, xl, y, n_out, n_in);
}
extern "C" __global__ void gemm_f32(const float* wp, const void* unused, const float* x, float* y, unsigned int n_out, unsigned int n_in) {
    F32W w = { wp }; XDense xl = { x, n_in }; gemm_body(w, xl, y, n_out, n_in);
}
extern "C" __global__ void conv_q4(const unsigned char* q, const unsigned short* s, const float* x, float* y, unsigned int n_out, unsigned int n_in, ConvP p, const float2* st, const float* gb) {
    Q4W w = { q, s }; XConv xl = { x, p, st, gb }; gemm_body(w, xl, y, n_out, n_in);
}
extern "C" __global__ void conv_q8(const signed char* q, const float* s, const float* x, float* y, unsigned int n_out, unsigned int n_in, ConvP p, const float2* st, const float* gb) {
    Q8W w = { q, s }; XConv xl = { x, p, st, gb }; gemm_body(w, xl, y, n_out, n_in);
}
extern "C" __global__ void conv_bf(const unsigned short* wp, const void* unused, const float* x, float* y, unsigned int n_out, unsigned int n_in, ConvP p, const float2* st, const float* gb) {
    BfW w = { wp }; XConv xl = { x, p, st, gb }; gemm_body(w, xl, y, n_out, n_in);
}

// ---- normalization statistics and softmax (two passes: mean, then mean squared deviation) ----
__device__ __forceinline__ float warp_sum(float v) { for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o); return v; }
__device__ __forceinline__ float warp_max(float v) { for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o)); return v; }
__device__ __forceinline__ float block_sum256(float v, float* red) {
    red[threadIdx.x] = v;
    __syncthreads();
    for (unsigned int o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o]; __syncthreads(); }
    float t = red[0];
    __syncthreads();
    return t;
}
// One 256-thread block per group.
extern "C" __global__ void group_stats(const float* x, float2* st, unsigned int n, unsigned int c, unsigned int groups, float eps) {
    __shared__ float red[256];
    unsigned int g = blockIdx.x, cpg = c / groups;
    size_t total = (size_t)n * cpg;
    float s = 0.f;
    for (size_t i = threadIdx.x; i < total; i += 256) s += x[(i / cpg) * c + g * cpg + i % cpg];
    float mean = block_sum256(s, red) / (float)total;
    float v = 0.f;
    for (size_t i = threadIdx.x; i < total; i += 256) { float d = x[(i / cpg) * c + g * cpg + i % cpg] - mean; v += d * d; }
    float var = block_sum256(v, red) / (float)total;
    if (threadIdx.x == 0) st[g] = make_float2(mean, rsqrtf(var + eps));
}
// One warp per row.
extern "C" __global__ void row_stats(const float* x, float2* st, unsigned int c, float eps) {
    const float* row = x + (size_t)blockIdx.x * c;
    float s = 0.f;
    for (unsigned int i = threadIdx.x; i < c; i += 32) s += row[i];
    float mean = warp_sum(s) / (float)c;
    float v = 0.f;
    for (unsigned int i = threadIdx.x; i < c; i += 32) { float d = row[i] - mean; v += d * d; }
    float var = warp_sum(v) / (float)c;
    if (threadIdx.x == 0) st[blockIdx.x] = make_float2(mean, rsqrtf(var + eps));
}
// Row softmax of scaled scores in place, one warp per row; padding columns become 0.
extern "C" __global__ void softmax_rows(float* s, unsigned int cols, unsigned int ld, float scale, unsigned int causal) {
    unsigned int r = blockIdx.x;
    float* row = s + (size_t)r * ld;
    unsigned int lim = causal ? min(r + 1, cols) : cols;
    float m = -__uint_as_float(0x7f800000u);
    for (unsigned int i = threadIdx.x; i < lim; i += 32) m = fmaxf(m, row[i] * scale);
    m = warp_max(m);
    float sum = 0.f;
    for (unsigned int i = threadIdx.x; i < lim; i += 32) { float e = expf(row[i] * scale - m); row[i] = e; sum += e; }
    float inv = 1.f / warp_sum(sum);
    for (unsigned int i = threadIdx.x; i < ld; i += 32) row[i] = i < lim ? row[i] * inv : 0.f;
}

// MoE experts. Gather: row r of the per-expert segments is token src[r] (t: a padding row, zero).
extern "C" __global__ void moe_gather(const float* x, const unsigned int* src, float* xs, unsigned int h, unsigned int t) {
    unsigned int r = blockIdx.y, s = src[r];
    for (unsigned int c = blockIdx.x * blockDim.x + threadIdx.x; c < h; c += gridDim.x * blockDim.x)
        xs[(size_t)r * h + c] = s < t ? x[(size_t)s * h + c] : 0.f;
}
// a = act(gate) * up over [rows, mi]: gate and up side by side per row ([rows, 2 mi]), or with
// `split` gate rows [rows, mi] then up rows [rows, mi]. kind: 0 SiLU, 1 tanh GELU, 2 clamped SiLU.
extern "C" __global__ void moe_act(const float* gu, float* a, unsigned int rows, unsigned int mi, unsigned int split, unsigned int kind, float limit) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * mi) return;
    size_t r = i / mi, j = i % mi;
    float g, u;
    if (split) { g = gu[i]; u = gu[(size_t)rows * mi + i]; }
    else { g = gu[r * 2 * mi + j]; u = gu[r * 2 * mi + mi + j]; }
    float v;
    if (kind == 1) {
        float k = 1.5957692f * (g + 0.044715f * g * g * g);
        v = g / (1.f + expf(-k));
    } else {
        if (kind == 2) { g = fminf(g, limit); u = fminf(fmaxf(u, -limit), limit); }
        v = g / (1.f + expf(-g));
    }
    a[i] = v * u;
}
// y[t] += weight * o[row] for each of token t's k (row, weight bits) slots, in order, unfused
// (the CPU's axpy).
extern "C" __global__ void moe_combine(const float* o, const unsigned int* slots, float* y, unsigned int h, unsigned int k) {
    unsigned int t = blockIdx.y;
    for (unsigned int c = blockIdx.x * blockDim.x + threadIdx.x; c < h; c += gridDim.x * blockDim.x) {
        float acc = y[(size_t)t * h + c];
        for (unsigned int j = 0; j < k; j++) {
            unsigned int row = slots[2 * (t * k + j)];
            float w = __uint_as_float(slots[2 * (t * k + j) + 1]);
            acc = __fadd_rn(acc, __fmul_rn(w, o[(size_t)row * h + c]));
        }
        y[(size_t)t * h + c] = acc;
    }
}
"#;

#[derive(Clone, Copy)]
struct CudaWeight {
    data: u64,
    scales: u64,
    /// Affine Q4: device offsets (0 if none), offsets per row, blocks of 32 per offset group.
    offsets: u64,
    gpr: u32,
    gk: u32,
    /// A ggml block format (`WeightRef::Gguf`): its kernels, the bytes per row and the factor on
    /// every output; `data` holds the rows as stored.
    gg: Option<(GgKernels, u64, f32)>,
}

/// The kernels over one ggml block format: `inkling_cuda.cu` compiled for that format alone
/// (`GGML_ONLY`) the first time a weight of it is uploaded, and kept on disk
/// ([`CudaBackend::cached_module`]).
#[derive(Clone, Copy, PartialEq)]
struct GgKernels {
    /// `mv_K`: matrix-vector products, a warp per row, a grid row per token.
    mv: *mut c_void,
    /// `gemmi_K`: the tensor-core GEMM with the weights' exact values and split activations.
    gemmi: *mut c_void,
}

/// A weight matrix as the ggml kernels take it (`W` in `inkling_cuda.cu`).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Wg {
    q: u64,
    s: u64,
    ef: u64,
    rb: u64,
    es: u64,
    ss: u64,
    f: f32,
    sr: u32,
    sc: u32,
    ncb: u32,
}

impl Wg {
    fn of(cw: &CudaWeight) -> Option<(GgKernels, Wg)> {
        let (k, rb, f) = cw.gg?;
        Some((
            k,
            Wg {
                q: cw.data,
                rb,
                f,
                sr: 1,
                sc: 1,
                ncb: 1,
                ..Wg::default()
            },
        ))
    }
}

/// The GEMM kernels' parameters (`GemmP` in `inkling_cuda.cu`): `rows` rows of `x` (`xs` floats
/// apart) into rows of `y` (`ys` apart); mode 0 stores, 1 adds.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GemmP {
    x: u64,
    y: u64,
    gidx: u64,
    tiles: u64,
    ntiles: u64,
    wrow: u64,
    xh: u64,
    xl: u64,
    n_in: u32,
    n_out: u32,
    rows: u32,
    xs: u32,
    ys: u32,
    mode: u32,
    mi: u32,
    inter: u32,
    beta: f32,
    lbeta: f32,
}

/// Whether every value is finite and below 32768 in magnitude: what the ggml GEMMs' split into
/// f16 halves holds (`split_h2` in `inkling_cuda.cu`). Checked across the thread pool.
fn fits_halves(x: &[f32]) -> bool {
    const CHUNK: usize = 1 << 16;
    let bad = std::sync::atomic::AtomicBool::new(false);
    crate::pool::run(x.len().div_ceil(CHUNK), &|i| {
        let c = &x[i * CHUNK..((i + 1) * CHUNK).min(x.len())];
        if c.iter().fold(0u32, |m, v| m.max(v.to_bits() & 0x7fff_ffff)) >= 0x4700_0000 {
            bad.store(true, Ordering::Relaxed);
        }
    });
    !bad.load(Ordering::Relaxed)
}

/// The ggml formats with device kernels (`dq<K>` in `inkling_cuda.cu`): every block format but
/// the ternary and 1-bit ones, and the float ones.
fn gg_supported(kind: crate::ggml::GgmlType) -> bool {
    use crate::ggml::GgmlType::*;
    matches!(
        kind,
        F32 | F16
            | BF16
            | Q4_0
            | Q4_1
            | Q5_0
            | Q5_1
            | Q8_0
            | Q2_K
            | Q3_K
            | Q4_K
            | Q5_K
            | Q6_K
            | IQ1_S
            | IQ1_M
            | IQ2_XXS
            | IQ2_XS
            | IQ2_S
            | IQ3_XXS
            | IQ3_S
            | IQ4_NL
            | IQ4_XS
            | MXFP4
    )
}

/// The i-quant grids and value tables, as device arrays.
pub(crate) fn ggml_tables_src() -> String {
    use crate::ggml::tables::*;
    use std::fmt::Write;
    let mut s = String::new();
    let mut arr = |ty: &str, name: &str, vals: Vec<String>| {
        let _ = write!(s, "__device__ const {ty} {name}[{}] = {{", vals.len());
        s.push_str(&vals.join(","));
        s.push_str("};\n");
    };
    arr("float", "kvalues_iq4nl", KVALUES_IQ4NL.iter().map(|v| format!("{v}.f")).collect());
    arr("float", "kvalues_mxfp4", KVALUES_MXFP4.iter().map(|v| format!("{v}.f")).collect());
    for (name, t) in [
        ("iq2xxs_grid", &IQ2XXS_GRID[..]),
        ("iq2xs_grid", &IQ2XS_GRID[..]),
        ("iq2s_grid", &IQ2S_GRID[..]),
        ("iq1s_grid", &IQ1S_GRID[..]),
    ] {
        arr("unsigned long long", name, t.iter().map(|v| format!("{v:#x}ull")).collect());
    }
    for (name, t) in [("iq3xxs_grid", &IQ3XXS_GRID[..]), ("iq3s_grid", &IQ3S_GRID[..])] {
        arr("unsigned int", name, t.iter().map(|v| format!("{v:#x}u")).collect());
    }
    s
}

#[derive(Default)]
struct Staging {
    x: u64,
    x_cap: usize,
    y: u64,
    y_cap: usize,
    /// Fused-norm statistics and gamma/beta; attention operands (Q, K, transposed V, one head's
    /// scores, outputs). Each is (device pointer, capacity in bytes).
    stats: (u64, usize),
    gb: (u64, usize),
    aq: (u64, usize),
    ak: (u64, usize),
    av: (u64, usize),
    asc: (u64, usize),
    ao: (u64, usize),
    /// MoE: input, gather sources, gathered rows, gate/up outputs, activations, expert outputs,
    /// combination slots.
    mx: (u64, usize),
    msrc: (u64, usize),
    mxs: (u64, usize),
    mgu: (u64, usize),
    ma: (u64, usize),
    mo: (u64, usize),
    mslots: (u64, usize),
}

/// The routed experts of one MoE layer in device memory (`upload_moe`): per expert the gate/up
/// projection (fused `[2 mi, h]`, or the gate `[mi, h]` with `up` holding the up projections) and
/// the down projection, each projection's GEMM kernel (one format per projection).
struct MoeDev {
    gu: Vec<CudaWeight>,
    up: Option<Vec<CudaWeight>>,
    down: Vec<CudaWeight>,
    funcs: [*mut c_void; 3],
    act: crate::gpu::MoeAct,
    /// Transient copies (`upload_moe_transient`): freed with this state, with the context to
    /// make current for that. Resident ones live in the shared cache.
    owner: Option<(std::sync::Arc<Cu>, *mut c_void)>,
    /// Resident ones: this state's references to the shared cache (freed with the last one).
    _refs: Option<CacheRefs>,
}
unsafe impl Send for MoeDev {}
unsafe impl Sync for MoeDev {}

impl Drop for MoeDev {
    fn drop(&mut self) {
        let Some((cu, ctx)) = &self.owner else { return };
        unsafe {
            if cu.cuCtxSetCurrent(*ctx) != 0 {
                return;
            }
            for cw in self.gu.iter().chain(self.up.iter().flatten()).chain(&self.down) {
                for p in [cw.data, cw.scales, cw.offsets] {
                    if p != 0 {
                        cu.cuMemFree_v2(p);
                    }
                }
            }
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

type WKey = (usize, usize, u64);
/// The shared weights by key, with the number of references to each.
type WCache = std::sync::Arc<Mutex<std::collections::HashMap<WKey, (CudaWeight, usize)>>>;

/// References to weights of the backend's shared cache: released on drop, each weight freed with
/// its last reference.
struct CacheRefs {
    cache: WCache,
    cu: std::sync::Arc<Cu>,
    ctx: *mut c_void,
    keys: Vec<WKey>,
}
// SAFETY: the context pointer is only made current; the cache is behind its lock.
unsafe impl Send for CacheRefs {}
unsafe impl Sync for CacheRefs {}

impl Drop for CacheRefs {
    fn drop(&mut self) {
        if self.keys.is_empty() {
            return;
        }
        let mut c = self.cache.lock().unwrap_or_else(|p| p.into_inner());
        let current = unsafe { self.cu.cuCtxSetCurrent(self.ctx) } == 0;
        for k in &self.keys {
            let Some(e) = c.get_mut(k) else { continue };
            e.1 -= 1;
            if e.1 == 0 {
                let (cw, _) = c.remove(k).unwrap();
                if current {
                    for p in [cw.data, cw.scales, cw.offsets] {
                        if p != 0 {
                            unsafe { self.cu.cuMemFree_v2(p) };
                        }
                    }
                }
            }
        }
    }
}

pub struct CudaBackend {
    /// Shared with per-session decode state, which frees its buffers on drop.
    cu: std::sync::Arc<Cu>,
    /// The primary context; current per thread, so every entry point makes it current (`bind`).
    ctx: *mut c_void,
    module: *mut c_void,
    /// Expert weights by host address and length: uploaded once, shared by the prefill MoE path
    /// and decode plans, freed with the last [`CacheRefs`] holding them (the model's unload).
    wcache: WCache,
    /// The ggml kernels by format id (None: the module did not compile or load). Code, not
    /// weights: the modules stay loaded for the process, like the main one.
    gg: Mutex<std::collections::HashMap<u32, Option<GgKernels>>>,
    /// bf16, q8, q4
    funcs: [*mut c_void; 3],
    /// GEMM over affine Q4 weights.
    func_q4a: *mut c_void,
    /// Implicit-GEMM convolutions, same order.
    conv_funcs: [*mut c_void; 3],
    /// gemm_f32, group_stats, row_stats, softmax_rows.
    aux_funcs: [*mut c_void; 4],
    /// moe_gather, moe_act, moe_combine.
    moe_funcs: [*mut c_void; 3],
    staging: Mutex<Staging>,
    stats: Stats,
    name: String,
    /// Compute capability (major, minor).
    sm: (c_int, c_int),
}
unsafe impl Send for CudaBackend {}
unsafe impl Sync for CudaBackend {}

impl CudaBackend {
    /// Makes the context current on the calling thread (a host may load on one thread and generate
    /// on another). Cheap: a thread-local store in the driver.
    fn bind(&self) -> bool {
        unsafe { self.cu.cuCtxSetCurrent(self.ctx) == 0 }
    }

    fn check(&self, rc: c_int, what: &str) -> Result<()> {
        check(&self.cu, rc, what)
    }

    pub fn new(ordinal: usize) -> Result<Self> {
        let cu = {
            use dlsym::Bindings;
            // SAFETY: libcuda's initialisers have no preconditions.
            unsafe { Cu::open_first(crate::platform::gpu::cuda::CANDIDATES.iter().copied()) }.map_err(|e| anyhow!("libcuda: {e}"))?
        };
        check(&cu, unsafe { cu.cuInit(0) }, "cuInit")?;
        let mut dev: c_int = 0;
        check(&cu, unsafe { cu.cuDeviceGet(&mut dev, ordinal as c_int) }, "cuDeviceGet")?;
        let (mut major, mut minor) = (0, 0);
        unsafe {
            cu.cuDeviceGetAttribute(&mut major, 75, dev); // COMPUTE_CAPABILITY_MAJOR
            cu.cuDeviceGetAttribute(&mut minor, 76, dev);
        }
        let mut ctx: *mut c_void = std::ptr::null_mut();
        check(
            &cu,
            unsafe { cu.cuDevicePrimaryCtxRetain(&mut ctx, dev) },
            "cuDevicePrimaryCtxRetain",
        )?;
        check(&cu, unsafe { cu.cuCtxSetCurrent(ctx) }, "cuCtxSetCurrent")?;

        // Compile the kernels for this device with NVRTC.
        let image = nvrtc_image(
            &format!("{KERNEL_SRC}{}", include_str!("cuda_decode.cu")),
            "oxinfer.cu",
            (major, minor),
        )?;
        let mut module: *mut c_void = std::ptr::null_mut();
        check(
            &cu,
            unsafe { cu.cuModuleLoadData(&mut module, image.as_ptr() as *const c_void) },
            "cuModuleLoadData",
        )?;
        let mut funcs = [std::ptr::null_mut(); 3];
        for (i, fname) in [c"gemm_bf", c"gemm_q8", c"gemm_q4"].iter().enumerate() {
            check(
                &cu,
                unsafe { cu.cuModuleGetFunction(&mut funcs[i], module, fname.as_ptr()) },
                "cuModuleGetFunction",
            )?;
        }
        let mut func_q4a = std::ptr::null_mut();
        check(
            &cu,
            unsafe { cu.cuModuleGetFunction(&mut func_q4a, module, c"gemm_q4a".as_ptr()) },
            "cuModuleGetFunction",
        )?;
        let mut aux_funcs = [std::ptr::null_mut(); 4];
        for (i, fname) in [c"gemm_f32", c"group_stats", c"row_stats", c"softmax_rows"].iter().enumerate() {
            check(
                &cu,
                unsafe { cu.cuModuleGetFunction(&mut aux_funcs[i], module, fname.as_ptr()) },
                "cuModuleGetFunction",
            )?;
        }
        let mut moe_funcs = [std::ptr::null_mut(); 3];
        for (i, fname) in [c"moe_gather", c"moe_act", c"moe_combine"].iter().enumerate() {
            check(
                &cu,
                unsafe { cu.cuModuleGetFunction(&mut moe_funcs[i], module, fname.as_ptr()) },
                "cuModuleGetFunction",
            )?;
        }
        let mut conv_funcs = [std::ptr::null_mut(); 3];
        for (i, fname) in [c"conv_bf", c"conv_q8", c"conv_q4"].iter().enumerate() {
            check(
                &cu,
                unsafe { cu.cuModuleGetFunction(&mut conv_funcs[i], module, fname.as_ptr()) },
                "cuModuleGetFunction",
            )?;
        }
        Ok(Self {
            cu: std::sync::Arc::new(cu),
            ctx,
            module,
            wcache: Default::default(),
            gg: Mutex::new(std::collections::HashMap::new()),
            funcs,
            func_q4a,
            conv_funcs,
            aux_funcs,
            moe_funcs,
            staging: Mutex::new(Staging::default()),
            stats: Stats::default(),
            name: format!("cuda (sm_{major}{minor})"),
            sm: (major, minor),
        })
    }

    /// Device copy of `bytes`, or None when VRAM is short (keeps a reserve for activations).
    fn upload_bytes(&self, ptr: *const u8, bytes: usize) -> Option<u64> {
        let (mut free, mut total) = (0usize, 0usize);
        unsafe { self.cu.cuMemGetInfo_v2(&mut free, &mut total) };
        if free < bytes + VRAM_RESERVE {
            return None;
        }
        let mut d = 0u64;
        if unsafe { self.cu.cuMemAlloc_v2(&mut d, bytes) } != 0 {
            return None;
        }
        if unsafe { self.cu.cuMemcpyHtoD_v2(d, ptr as *const c_void, bytes) } != 0 {
            unsafe { self.cu.cuMemFree_v2(d) };
            return None;
        }
        Some(d)
    }

    /// Makes a device buffer at least `need` bytes long.
    fn grow(&self, b: &mut (u64, usize), need: usize) -> bool {
        if b.1 >= need && b.0 != 0 {
            return true;
        }
        if b.0 != 0 {
            unsafe { self.cu.cuMemFree_v2(b.0) };
            *b = (0, 0);
        }
        let cap = need.max(256) * 2;
        if unsafe { self.cu.cuMemAlloc_v2(&mut b.0, cap) } != 0 {
            b.0 = 0;
            return false;
        }
        b.1 = cap;
        true
    }

    /// Launches `f` on the default stream; `params` point at each kernel argument.
    fn launch(&self, f: *mut c_void, grid: (u32, u32), block: u32, params: &mut [*mut c_void]) -> bool {
        self.launch3(f, (grid.0, grid.1, 1), block, params)
    }

    /// [`Self::launch`] with a three-dimensional grid.
    fn launch3(&self, f: *mut c_void, grid: (u32, u32, u32), block: u32, params: &mut [*mut c_void]) -> bool {
        let rc = unsafe {
            self.cu.cuLaunchKernel(
                f,
                grid.0,
                grid.1,
                grid.2,
                block,
                1,
                1,
                0,
                std::ptr::null_mut(),
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            let _ = self.check(rc, "cuLaunchKernel").map_err(|e| eprintln!("[cuda: {e}]"));
        }
        rc == 0
    }

    fn htod<T>(&self, dst: u64, src: &[T]) -> bool {
        unsafe {
            self.cu
                .cuMemcpyHtoD_v2(dst, src.as_ptr() as *const c_void, std::mem::size_of_val(src))
                == 0
        }
    }

    /// The kernels for ggml format `kind`, compiled on first use.
    fn gg_kernels(&self, kind: crate::ggml::GgmlType) -> Option<GgKernels> {
        if !gg_supported(kind) {
            return None;
        }
        let id = kind.id();
        if let Some(k) = self.gg.lock().unwrap().get(&id) {
            return *k;
        }
        let body = include_str!("inkling_cuda.cu");
        // Tests may take the kernels from a file (`ggml_bench`: no rebuild per kernel edit).
        #[cfg(test)]
        let body = &std::env::var("OXINFER_GGML_CU")
            .ok()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .unwrap_or_else(|| body.to_string());
        let src = format!("#define GGML_ONLY 1\n#define HAS_{id} 1\n{}{body}", ggml_tables_src());
        let k = match self.cached_module(&src, &format!("ggml-{{sm}}-{id}"), "ggml.cu") {
            Ok(module) => {
                let get = |name: String| -> Option<*mut c_void> {
                    let c = CString::new(name).ok()?;
                    let mut f = std::ptr::null_mut();
                    (unsafe { self.cu.cuModuleGetFunction(&mut f, module, c.as_ptr()) } == 0).then_some(f)
                };
                get(format!("mv_{id}"))
                    .zip(get(format!("gemmi_{id}")))
                    .map(|(mv, gemmi)| GgKernels { mv, gemmi })
            }
            Err(e) => {
                eprintln!("[cuda: no kernels for {kind} weights ({e}): they stay on the CPU]");
                None
            }
        };
        self.gg.lock().unwrap().insert(id, k);
        k
    }

    /// Device copy of one weight matrix (None when VRAM is short, or for a GGUF format without
    /// device kernels: the ternary and 1-bit ones, FP8 with block scales).
    fn upload_weight(&self, w: &WeightRef) -> Option<CudaWeight> {
        Some(match w {
            WeightRef::Gguf(m) => {
                // A tensor-wide scale is a factor on the outputs; block scales have no kernel here.
                let f = match &m.block_scales {
                    None => 1.0,
                    Some(bs) if !m.kind.is_float() && bs.rows >= m.n_out && bs.cols >= m.n_in => bs.s[0],
                    Some(_) => return None,
                };
                if !m.n_in.is_multiple_of(32) {
                    return None;
                }
                let k = self.gg_kernels(m.kind)?;
                let rb = m.row_bytes();
                CudaWeight {
                    data: self.upload_bytes(m.data.as_ptr(), m.n_out * rb)?,
                    scales: 0,
                    offsets: 0,
                    gpr: 0,
                    gk: 0,
                    gg: Some((k, rb as u64, f)),
                }
            }
            WeightRef::Bf16(w) => CudaWeight {
                data: self.upload_bytes(w.as_ptr() as *const u8, w.len() * 2)?,
                scales: 0,
                offsets: 0,
                gpr: 0,
                gk: 0,
                gg: None,
            },
            WeightRef::Q8(m) => {
                let data = self.upload_bytes(m.q.as_ptr() as *const u8, m.q.len())?;
                let scales = match self.upload_bytes(m.scales.as_ptr() as *const u8, m.scales.len() * 4) {
                    Some(s) => s,
                    None => {
                        unsafe { self.cu.cuMemFree_v2(data) };
                        return None;
                    }
                };
                CudaWeight {
                    data,
                    scales,
                    offsets: 0,
                    gpr: 0,
                    gk: 0,
                    gg: None,
                }
            }
            WeightRef::Q4(m) => {
                let data = self.upload_bytes(m.q.as_ptr(), m.q.len())?;
                let scales = match self.upload_bytes(m.scales.as_ptr() as *const u8, m.scales.len() * 2) {
                    Some(s) => s,
                    None => {
                        unsafe { self.cu.cuMemFree_v2(data) };
                        return None;
                    }
                };
                let (mut offsets, mut gpr, mut gk) = (0, 0, 0);
                if let Some(o) = &m.offsets {
                    offsets = match self.upload_bytes(o.v.as_ptr() as *const u8, o.v.len() * 2) {
                        Some(p) => p,
                        None => {
                            unsafe {
                                self.cu.cuMemFree_v2(data);
                                self.cu.cuMemFree_v2(scales);
                            }
                            return None;
                        }
                    };
                    (gpr, gk) = ((m.n_in / o.group) as u32, (o.group / 32) as u32);
                }
                CudaWeight {
                    data,
                    scales,
                    offsets,
                    gpr,
                    gk,
                    gg: None,
                }
            }
        })
    }

    /// A kernel of the module by name.
    fn func(&self, name: &str) -> Option<*mut c_void> {
        let c = CString::new(name).ok()?;
        let mut f = std::ptr::null_mut();
        (unsafe { self.cu.cuModuleGetFunction(&mut f, self.module, c.as_ptr()) } == 0).then_some(f)
    }

    /// No references yet to the shared cache.
    fn cache_refs(&self) -> CacheRefs {
        CacheRefs {
            cache: self.wcache.clone(),
            cu: self.cu.clone(),
            ctx: self.ctx,
            keys: Vec::new(),
        }
    }

    /// [`Self::upload_weight`] through the cache of shared weights, the reference counted in `refs`.
    fn upload_cached(&self, w: &WeightRef, refs: &mut CacheRefs) -> Option<CudaWeight> {
        let key = weight_key(w)?;
        let mut c = self.wcache.lock().unwrap_or_else(|p| p.into_inner());
        let cw = match c.get_mut(&key) {
            Some(e) => {
                e.1 += 1;
                e.0
            }
            None => {
                let cw = self.upload_weight(w)?;
                c.insert(key, (cw, 1));
                cw
            }
        };
        refs.keys.push(key);
        Some(cw)
    }

    /// Every weight of `groups` through the cache, or none: what this call uploaded is freed again
    /// when one does not fit (the references are dropped).
    fn upload_all(&self, groups: &[&[WeightRef]]) -> Option<(Vec<Vec<CudaWeight>>, CacheRefs)> {
        let mut refs = self.cache_refs();
        let mut out: Vec<Vec<CudaWeight>> = Vec::with_capacity(groups.len());
        for ws in groups {
            let v = ws.iter().map(|w| self.upload_cached(w, &mut refs)).collect::<Option<Vec<_>>>()?;
            out.push(v);
        }
        Some((out, refs))
    }

    /// `upload_moe` (`resident`: through the shared cache, kept) or `upload_moe_transient` (owned
    /// copies, freed with the returned state).
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
    ) -> Option<DeviceWeight> {
        let n = gate_up.len();

        if !crate::gpu::moe_offload() || n == 0 || down.len() != n || up.is_some_and(|u| u.len() != n) || !self.bind() {
            return None;
        }
        // The GEMM tiles: whole tiles of outputs, whole blocks of 32 inputs.
        let g_rows = if up.is_some() { mi } else { 2 * mi };
        if !g_rows.is_multiple_of(TILE) || !h.is_multiple_of(TILE) || !h.is_multiple_of(32) || !mi.is_multiple_of(32) {
            return None;
        }
        // One kernel per projection: every expert in the same format.
        let func = |ws: &[WeightRef]| -> Option<*mut c_void> {
            let f = self.gemm_func(&ws[0])?;
            ws.iter().all(|w| self.gemm_func(w) == Some(f)).then_some(f)
        };
        let funcs = [func(gate_up)?, up.map_or(Some(std::ptr::null_mut()), func)?, func(down)?];
        // All experts or none (VRAM): a partial upload is freed again.
        let mut groups: Vec<&[WeightRef]> = vec![gate_up, down];
        groups.extend(up);
        let (all, refs) = if resident {
            let (a, r) = self.upload_all(&groups)?;
            (a, Some(r))
        } else {
            (self.upload_owned(&groups)?, None)
        };
        let mut all = all.into_iter();
        let (gu, dw) = (all.next()?, all.next()?);
        let upw = all.next();
        Some(Box::new(MoeDev {
            gu,
            up: upw,
            down: dw,
            funcs,
            act,
            owner: (!resident).then(|| (self.cu.clone(), self.ctx)),
            _refs: refs,
        }))
    }
    #[allow(clippy::too_many_arguments)]
    fn moe_run(&self, x: &[f32], t: usize, h: usize, mi: usize, routes: &[Vec<(usize, f32)>], dev: &DeviceWeight, y: &mut [f32]) -> bool {
        let Some(md) = dev.downcast_ref::<MoeDev>() else {
            return false;
        };
        let k = routes.first().map_or(0, |r| r.len());
        if !crate::gpu::moe_offload()
            || t == 0
            || k == 0
            || routes.len() != t
            || routes.iter().any(|r| r.len() != k || r.iter().any(|&(e, _)| e >= md.gu.len()))
            || x.len() < t * h
            || y.len() < t * h
            || !self.bind()
        {
            return false;
        }
        // One segment of rows per expert, in expert order, padded to whole GEMM tiles.
        let ne = md.gu.len();
        let mut count = vec![0usize; ne];
        for &(e, _) in routes.iter().flatten() {
            count[e] += 1;
        }
        let mut segs = Vec::with_capacity(ne);
        let mut rows = 0usize;
        for &c in &count {
            segs.push(rows);
            rows += c.next_multiple_of(TILE);
        }
        // Gathered row sources (t: padding), and each token's (row, weight bits) slots.
        let mut src = vec![t as u32; rows];
        let mut fill = segs.clone();
        let mut slots = vec![0u32; 2 * t * k];
        for (ti, r) in routes.iter().enumerate() {
            for (j, &(e, w)) in r.iter().enumerate() {
                let i = fill[e];
                fill[e] += 1;
                src[i] = ti as u32;
                slots[2 * (ti * k + j)] = i as u32;
                slots[2 * (ti * k + j) + 1] = w.to_bits();
            }
        }
        let split = md.up.is_some();
        let mut st = self.staging.lock().unwrap();
        let st = &mut *st;
        let ok = self.grow(&mut st.mx, t * h * 4)
            && self.grow(&mut st.msrc, rows * 4)
            && self.grow(&mut st.mxs, rows * h * 4)
            && self.grow(&mut st.mgu, rows * 2 * mi * 4)
            && self.grow(&mut st.ma, rows * mi * 4)
            && self.grow(&mut st.mo, rows * h * 4)
            && self.grow(&mut st.mslots, slots.len() * 4);
        if !ok {
            return false;
        }
        let t0 = Instant::now();
        // y goes up too: the combination adds to it in the CPU's order.
        if !(self.htod(st.mx.0, &x[..t * h]) && self.htod(st.msrc.0, &src) && self.htod(st.mslots.0, &slots)) {
            return false;
        }
        let t1 = Instant::now();
        let (mut p_x, mut p_src, mut p_xs, mut h_u, mut t_u) = (st.mx.0, st.msrc.0, st.mxs.0, h as u32, t as u32);
        let cols = (h as u32).div_ceil(256).min(64);
        if !self.launch(
            self.moe_funcs[0],
            (cols, rows as u32),
            256,
            &mut [
                &mut p_x as *mut u64 as *mut c_void,
                &mut p_src as *mut u64 as *mut c_void,
                &mut p_xs as *mut u64 as *mut c_void,
                &mut h_u as *mut u32 as *mut c_void,
                &mut t_u as *mut u32 as *mut c_void,
            ],
        ) {
            return false;
        }
        // Gate/up and down GEMMs per expert over its segment (experts with no rows are skipped).
        for e in 0..ne {
            let len = count[e].next_multiple_of(TILE);
            if len == 0 {
                continue;
            }
            let r0 = segs[e];
            let xs = st.mxs.0 + (r0 * h * 4) as u64;
            let ok = match &md.up {
                Some(up) => {
                    self.gemm(md.funcs[0], &md.gu[e], xs, st.mgu.0 + (r0 * mi * 4) as u64, mi, h, len)
                        && self.gemm(md.funcs[1], &up[e], xs, st.mgu.0 + ((rows + r0) * mi * 4) as u64, mi, h, len)
                }
                None => self.gemm(md.funcs[0], &md.gu[e], xs, st.mgu.0 + (r0 * 2 * mi * 4) as u64, 2 * mi, h, len),
            };
            if !ok {
                return false;
            }
        }
        let (kind, limit) = match md.act {
            crate::gpu::MoeAct::Silu => (0u32, 0f32),
            crate::gpu::MoeAct::GeluTanh => (1, 0.0),
            crate::gpu::MoeAct::SiluClamp(l) => (2, l),
            crate::gpu::MoeAct::SwigluOai { .. } | crate::gpu::MoeAct::Situ { .. } => return false,
        };
        let (mut p_gu, mut p_a, mut rows_u, mut mi_u, mut split_u, mut kind_u, mut limit_f) =
            (st.mgu.0, st.ma.0, rows as u32, mi as u32, split as u32, kind, limit);
        if !self.launch(
            self.moe_funcs[1],
            ((rows * mi).div_ceil(256) as u32, 1),
            256,
            &mut [
                &mut p_gu as *mut u64 as *mut c_void,
                &mut p_a as *mut u64 as *mut c_void,
                &mut rows_u as *mut u32 as *mut c_void,
                &mut mi_u as *mut u32 as *mut c_void,
                &mut split_u as *mut u32 as *mut c_void,
                &mut kind_u as *mut u32 as *mut c_void,
                &mut limit_f as *mut f32 as *mut c_void,
            ],
        ) {
            return false;
        }
        for e in 0..ne {
            let len = count[e].next_multiple_of(TILE);
            if len == 0 {
                continue;
            }
            let r0 = segs[e];
            if !self.gemm(
                md.funcs[2],
                &md.down[e],
                st.ma.0 + (r0 * mi * 4) as u64,
                st.mo.0 + (r0 * h * 4) as u64,
                h,
                mi,
                len,
            ) {
                return false;
            }
        }
        // The input buffer is free again: it takes y for the combination.
        if !self.htod(st.mx.0, &y[..t * h]) {
            return false;
        }
        let (mut p_o, mut p_sl, mut p_y, mut k_u) = (st.mo.0, st.mslots.0, st.mx.0, k as u32);
        if !self.launch(
            self.moe_funcs[2],
            (cols, t as u32),
            256,
            &mut [
                &mut p_o as *mut u64 as *mut c_void,
                &mut p_sl as *mut u64 as *mut c_void,
                &mut p_y as *mut u64 as *mut c_void,
                &mut h_u as *mut u32 as *mut c_void,
                &mut k_u as *mut u32 as *mut c_void,
            ],
        ) {
            return false;
        }
        if unsafe { self.cu.cuCtxSynchronize() } != 0 {
            return false;
        }
        let t2 = Instant::now();
        if unsafe { self.cu.cuMemcpyDtoH_v2(y.as_mut_ptr() as *mut c_void, st.mx.0, t * h * 4) } != 0 {
            return false;
        }
        self.stats.calls.fetch_add(1, Ordering::Relaxed);
        self.stats.copy_in_ns.fetch_add((t1 - t0).as_nanos() as u64, Ordering::Relaxed);
        self.stats.gpu_ns.fetch_add((t2 - t1).as_nanos() as u64, Ordering::Relaxed);
        self.stats.copy_out_ns.fetch_add(t2.elapsed().as_nanos() as u64, Ordering::Relaxed);
        true
    }

    /// Every weight of `groups` as owned copies (not cached), or none.
    fn upload_owned(&self, groups: &[&[WeightRef]]) -> Option<Vec<Vec<CudaWeight>>> {
        let mut out: Vec<Vec<CudaWeight>> = Vec::with_capacity(groups.len());
        for ws in groups {
            let mut v = Vec::with_capacity(ws.len());
            for w in ws.iter() {
                match self.upload_weight(w) {
                    Some(cw) => v.push(cw),
                    None => {
                        for cw in out.iter().flatten().chain(&v) {
                            self.free_weight(cw);
                        }
                        return None;
                    }
                }
            }
            out.push(v);
        }
        Some(out)
    }

    fn free_weight(&self, cw: &CudaWeight) {
        for p in [cw.data, cw.scales, cw.offsets] {
            if p != 0 {
                unsafe { self.cu.cuMemFree_v2(p) };
            }
        }
    }

    /// The GEMM kernel for a weight of this format.
    fn gemm_func(&self, w: &WeightRef) -> Option<*mut c_void> {
        Some(match w {
            WeightRef::Bf16(_) => self.funcs[0],
            WeightRef::Q8(_) => self.funcs[1],
            WeightRef::Q4(m) if m.offsets.is_some() => self.func_q4a,
            WeightRef::Q4(_) => self.funcs[2],
            WeightRef::Gguf(_) => return None,
        })
    }

    /// Queues `y[t_pad][n_out] = x[t_pad][n_in] * W^T` (t_pad a multiple of TILE) with `func`
    /// (from `gemm_func`): the plain kernels take the first six parameters, the affine one all
    /// nine.
    #[allow(clippy::too_many_arguments)]
    fn gemm(&self, func: *mut c_void, cw: &CudaWeight, x: u64, y: u64, n_out: usize, n_in: usize, t_pad: usize) -> bool {
        let (mut p_w, mut p_s, mut p_x, mut p_y) = (cw.data, cw.scales, x, y);
        let (mut n_out_u, mut n_in_u) = (n_out as u32, n_in as u32);
        let (mut p_o, mut gpr, mut gk) = (cw.offsets, cw.gpr, cw.gk);
        let mut params: [*mut c_void; 9] = [
            &mut p_w as *mut u64 as *mut c_void,
            &mut p_s as *mut u64 as *mut c_void,
            &mut p_x as *mut u64 as *mut c_void,
            &mut p_y as *mut u64 as *mut c_void,
            &mut n_out_u as *mut u32 as *mut c_void,
            &mut n_in_u as *mut u32 as *mut c_void,
            &mut p_o as *mut u64 as *mut c_void,
            &mut gpr as *mut u32 as *mut c_void,
            &mut gk as *mut u32 as *mut c_void,
        ];
        self.launch(func, ((n_out / TILE) as u32, (t_pad / TILE) as u32), 256, &mut params)
    }

    /// Queues `y[t][n_out] (+)= x[t][n_in] * W^T` over a ggml matrix (`gemmi_K`: the weights'
    /// values as stored, the activations as pairs of halves, f32 sums per 16 columns); rows of
    /// `x` are `n_in` floats apart and rows of `y` `n_out` (`mode` 1 adds).
    #[allow(clippy::too_many_arguments)]
    fn gemm_gg(&self, k: GgKernels, w: Wg, x: u64, n_in: usize, y: u64, n_out: usize, t: usize, mode: u32) -> bool {
        self.gemm_gg_p(
            k,
            w,
            GemmP {
                x,
                y,
                n_in: n_in as u32,
                n_out: n_out as u32,
                rows: t as u32,
                xs: n_in as u32,
                ys: n_out as u32,
                mode,
                ..GemmP::default()
            },
        )
    }

    /// Queues the GEMM `p` over a ggml matrix (`rows` rows of `x`, `xs` floats apart, into rows
    /// of `y`, `ys` apart).
    fn gemm_gg_p(&self, k: GgKernels, w: Wg, p: GemmP) -> bool {
        let mut params = [&w as *const Wg as *mut c_void, &p as *const GemmP as *mut c_void];
        self.launch(k.gemmi, (p.n_out.div_ceil(64), p.rows.div_ceil(64)), 128, &mut params)
    }

    fn ensure_staging(&self, st: &mut Staging, x_bytes: usize, y_bytes: usize) -> bool {
        let grow = |ptr: &mut u64, cap: &mut usize, need: usize| -> bool {
            if *cap >= need {
                return true;
            }
            if *ptr != 0 {
                unsafe { self.cu.cuMemFree_v2(*ptr) };
                *ptr = 0;
                *cap = 0;
            }
            let new_cap = need * 2;
            if unsafe { self.cu.cuMemAlloc_v2(ptr, new_cap) } != 0 {
                *ptr = 0;
                return false;
            }
            *cap = new_cap;
            true
        };
        grow(&mut st.x, &mut st.x_cap, x_bytes) && grow(&mut st.y, &mut st.y_cap, y_bytes)
    }
}

/// The upload cache's key of a weight: its address, length and a fingerprint of samples of its
/// bytes (memory freed and reused at the same address by other weights must not hit).
/// Compiles `src` with NVRTC for compute capability `sm` into a native binary (or PTX when NVRTC
/// cannot give one), NUL-terminated for `cuModuleLoadData`.
fn nvrtc_image(src: &str, name: &str, sm: (c_int, c_int)) -> Result<Vec<u8>> {
    let nvrtc = {
        use dlsym::Bindings;
        // SAFETY: libnvrtc's initialisers have no preconditions.
        unsafe { Nvrtc::open_first(NVRTC_CANDIDATES.iter().copied()) }
            .map_err(|e| anyhow!("libnvrtc (ships with the CUDA toolkit): {e}"))?
    };
    let src = CString::new(src)?;
    let name = CString::new(name)?;
    let mut prog: *mut c_void = std::ptr::null_mut();
    let rc = unsafe { nvrtc.nvrtcCreateProgram(&mut prog, src.as_ptr(), name.as_ptr(), 0, std::ptr::null(), std::ptr::null()) };
    if rc != 0 {
        bail!("nvrtcCreateProgram failed ({rc})");
    }
    let arch = CString::new(format!("--gpu-architecture=sm_{}{}", sm.0, sm.1))?;
    let opts = [arch.as_ptr(), c"--use_fast_math".as_ptr()];
    let rc = unsafe { nvrtc.nvrtcCompileProgram(prog, opts.len() as c_int, opts.as_ptr()) };
    if rc != 0 {
        let mut n = 0usize;
        unsafe { nvrtc.nvrtcGetProgramLogSize(prog, &mut n) };
        let mut log = vec![0u8; n.max(1)];
        unsafe { nvrtc.nvrtcGetProgramLog(prog, log.as_mut_ptr() as *mut c_char) };
        unsafe { nvrtc.nvrtcDestroyProgram(&mut prog) };
        bail!("NVRTC compilation failed ({rc}):\n{}", String::from_utf8_lossy(&log));
    }
    let mut n = 0usize;
    let cubin_rc = unsafe { nvrtc.nvrtcGetCUBINSize(prog, &mut n) };
    // Prefer a native binary: PTX carries the toolkit's version, which a driver older than the
    // toolkit refuses to load, whereas a binary for this exact core needs no driver JIT.
    let mut image;
    if cubin_rc == Some(0) && n > 0 {
        image = vec![0u8; n];
        unsafe { nvrtc.nvrtcGetCUBIN(prog, image.as_mut_ptr() as *mut c_char) };
    } else {
        unsafe { nvrtc.nvrtcGetPTXSize(prog, &mut n) };
        image = vec![0u8; n.max(1)];
        unsafe { nvrtc.nvrtcGetPTX(prog, image.as_mut_ptr() as *mut c_char) };
    }
    unsafe { nvrtc.nvrtcDestroyProgram(&mut prog) };
    if *image.last().unwrap_or(&1) != 0 {
        image.push(0);
    }
    Ok(image)
}

/// What model code that runs its own kernels needs of the backend ([`crate::inkling`]'s CUDA
/// path): the driver, the context, and modules compiled from its sources.
impl CudaBackend {
    pub(crate) fn driver(&self) -> &std::sync::Arc<Cu> {
        &self.cu
    }
    /// The primary context, made current on the calling thread.
    pub(crate) fn context(&self) -> Option<*mut c_void> {
        self.bind().then_some(self.ctx)
    }
    /// `src` compiled for this device (a native binary, or PTX), for [`Self::load_image`].
    pub(crate) fn compile(&self, src: &str, name: &str) -> Result<Vec<u8>> {
        nvrtc_image(src, name, self.sm)
    }
    /// The module for `src` (compiled as `unit`), compiled once per machine: kept in
    /// `$OXINFER_KERNEL_CACHE` (default `~/.cache/oxinfer/kernels`; `0` turns it off) as one file
    /// (`tag` with `{sm}` standing for the compute capability) holding the source and the binary,
    /// used again only for exactly that source (NVRTC takes seconds for a large module).
    pub(crate) fn cached_module(&self, src: &str, tag: &str, unit: &str) -> Result<*mut c_void> {
        let dir = match std::env::var("OXINFER_KERNEL_CACHE") {
            Ok(v) if v == "0" => None,
            Ok(v) => Some(std::path::PathBuf::from(v)),
            Err(_) => std::env::var("HOME")
                .ok()
                .map(|h| std::path::Path::new(&h).join(".cache").join("oxinfer").join("kernels")),
        };
        let (ma, mi) = self.sm;
        let name = format!("{}.bin", tag.replace("{sm}", &format!("sm{ma}{mi}")));
        // [source length, u64][source][image]
        if let Some(d) = &dir
            && let Ok(f) = std::fs::read(d.join(&name))
            && f.len() >= 8
        {
            let n = u64::from_le_bytes(f[..8].try_into().unwrap()) as usize;
            if f.len() > 8 + n
                && &f[8..8 + n] == src.as_bytes()
                && let Ok(m) = self.load_image(&f[8 + n..])
            {
                return Ok(m);
            }
        }
        let img = self.compile(src, unit)?;
        if let Some(d) = &dir
            && std::fs::create_dir_all(d).is_ok()
        {
            // Written under a temporary name, then renamed: a reader sees a whole file.
            let tmp = d.join(format!("{name}.{}", std::process::id()));
            let mut f = (src.len() as u64).to_le_bytes().to_vec();
            f.extend_from_slice(src.as_bytes());
            f.extend_from_slice(&img);
            if std::fs::write(&tmp, &f).is_ok() && std::fs::rename(&tmp, d.join(&name)).is_err() {
                let _ = std::fs::remove_file(&tmp);
            }
        }
        self.load_image(&img)
    }
    /// Loads a compiled module (from [`Self::compile`]).
    pub(crate) fn load_image(&self, image: &[u8]) -> Result<*mut c_void> {
        if !self.bind() {
            bail!("cuCtxSetCurrent failed");
        }
        let mut module: *mut c_void = std::ptr::null_mut();
        self.check(
            unsafe { self.cu.cuModuleLoadData(&mut module, image.as_ptr() as *const c_void) },
            "cuModuleLoadData",
        )?;
        Ok(module)
    }
}

fn weight_key(w: &WeightRef) -> Option<(usize, usize, u64)> {
    fn sample(b: &[u8], h: &mut u64) {
        let n = b.len();
        for start in [0, n / 2, n.saturating_sub(256)] {
            for &x in &b[start.min(n)..(start + 256).min(n)] {
                *h = (*h ^ x as u64).wrapping_mul(0x100000001b3);
            }
        }
    }
    fn bytes<T>(v: &[T]) -> &[u8] {
        // SAFETY: plain numeric data viewed as bytes.
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
    }
    let mut h = 0xcbf29ce484222325u64;
    let (p, n) = match w {
        WeightRef::Bf16(v) => {
            sample(bytes(v), &mut h);
            (v.as_ptr() as usize, v.len() * 2)
        }
        WeightRef::Q8(m) => {
            sample(bytes(&m.q[..]), &mut h);
            sample(bytes(&m.scales[..]), &mut h);
            (m.q.as_ptr() as usize, m.q.len())
        }
        WeightRef::Q4(m) => {
            sample(&m.q[..], &mut h);
            sample(bytes(&m.scales[..]), &mut h);
            if let Some(o) = &m.offsets {
                sample(bytes(&o.v[..]), &mut h);
            }
            (m.q.as_ptr() as usize, m.q.len())
        }
        WeightRef::Gguf(_) => return None,
    };
    Some((p, n, h))
}

fn check(cu: &Cu, rc: c_int, what: &str) -> Result<()> {
    if rc == 0 {
        return Ok(());
    }
    let mut s: *const c_char = std::ptr::null();
    let text = match unsafe { cu.cuGetErrorString(rc, &mut s) } {
        Some(0) if !s.is_null() => unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned(),
        _ => format!("error {rc}"),
    };
    Err(anyhow!("{what}: {text}"))
}

/// A device weight, freed when its model drops it (with the context to make current for that).
struct CudaWeightBox(CudaWeight, std::sync::Arc<Cu>, *mut c_void);
unsafe impl Send for CudaWeightBox {}
unsafe impl Sync for CudaWeightBox {}

impl Drop for CudaWeightBox {
    fn drop(&mut self) {
        if unsafe { self.1.cuCtxSetCurrent(self.2) } != 0 {
            return;
        }
        for p in [self.0.data, self.0.scales, self.0.offsets] {
            if p != 0 {
                unsafe { self.1.cuMemFree_v2(p) };
            }
        }
    }
}

impl Backend for CudaBackend {
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
    fn name(&self) -> &str {
        &self.name
    }

    fn device_memory(&self) -> Option<(u64, u64)> {
        if !self.bind() {
            return None;
        }
        let (mut free, mut total) = (0usize, 0usize);
        (unsafe { self.cu.cuMemGetInfo_v2(&mut free, &mut total) } == 0).then_some((free as u64, total as u64))
    }

    fn stats(&self) -> Option<String> {
        let n = self.stats.calls.load(Ordering::Relaxed);
        if n == 0 {
            return None;
        }
        let ms = |a: &AtomicU64| a.load(Ordering::Relaxed) as f64 / 1e6;
        Some(format!(
            "{n} launches: copy-in {:.1} ms, gpu {:.1} ms, copy-out {:.1} ms",
            ms(&self.stats.copy_in_ns),
            ms(&self.stats.gpu_ns),
            ms(&self.stats.copy_out_ns)
        ))
    }

    fn wants(&self, n_out: usize, n_in: usize, t: usize) -> bool {
        t >= MIN_T && n_out.is_multiple_of(TILE) && n_in.is_multiple_of(32) && 2.0 * (t * n_out * n_in) as f64 >= MIN_FLOPS
    }

    fn upload(&self, w: &WeightRef, _n_out: usize, _n_in: usize) -> Option<DeviceWeight> {
        if !self.bind() {
            return None;
        }
        Some(Box::new(CudaWeightBox(self.upload_weight(w)?, self.cu.clone(), self.ctx)))
    }

    fn conv2d(&self, x: &[f32], g: &crate::gpu::ConvGeom, norm: Option<&crate::gpu::NormSpec>, job: &mut Job) -> bool {
        use crate::gpu::NormKind;
        if !self.bind() {
            return false;
        }
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
        let Some(cw) = job.dev.downcast_ref::<CudaWeightBox>() else {
            return false;
        };
        if cw.0.offsets != 0 {
            return false; // no affine convolution kernel
        }
        let cw = &cw.0;
        let t_pad = t.div_ceil(TILE) * TILE;
        let mut st = self.staging.lock().unwrap();
        let st = &mut *st;
        if !self.ensure_staging(st, npx * g.cin * 4, t_pad * job.n_out * 4) {
            return false;
        }
        let n_stats = match norm.map(|n| n.kind) {
            Some(NormKind::Group(gr)) => gr,
            Some(NormKind::Layer) => npx,
            None => 1,
        };
        if !self.grow(&mut st.stats, n_stats * 8) || !self.grow(&mut st.gb, 2 * g.cin * 4) {
            return false;
        }
        let t0 = Instant::now();
        if !self.htod(st.x, &x[..npx * g.cin]) {
            return false;
        }
        if let Some(n) = norm {
            let gb: Vec<f32> = n.gamma.iter().chain(n.beta).copied().collect();
            if !self.htod(st.gb.0, &gb) {
                return false;
            }
        }
        let t1 = Instant::now();
        // Statistics first; launches on one stream run in order.
        if let Some(n) = norm {
            let (mut px, mut sp, mut eps) = (st.x, st.stats.0, n.eps);
            let ok = match n.kind {
                NormKind::Group(gr) => {
                    let (mut nn, mut cc, mut gg) = (npx as u32, g.cin as u32, gr as u32);
                    let mut p: [*mut c_void; 6] = [
                        &mut px as *mut u64 as *mut c_void,
                        &mut sp as *mut u64 as *mut c_void,
                        &mut nn as *mut u32 as *mut c_void,
                        &mut cc as *mut u32 as *mut c_void,
                        &mut gg as *mut u32 as *mut c_void,
                        &mut eps as *mut f32 as *mut c_void,
                    ];
                    self.launch(self.aux_funcs[1], (gr as u32, 1), 256, &mut p)
                }
                NormKind::Layer => {
                    let mut cc = g.cin as u32;
                    let mut p: [*mut c_void; 4] = [
                        &mut px as *mut u64 as *mut c_void,
                        &mut sp as *mut u64 as *mut c_void,
                        &mut cc as *mut u32 as *mut c_void,
                        &mut eps as *mut f32 as *mut c_void,
                    ];
                    self.launch(self.aux_funcs[2], (npx as u32, 1), 32, &mut p)
                }
            };
            if !ok {
                return false;
            }
        }
        let fi = match job.w {
            WeightRef::Bf16(_) => 0,
            WeightRef::Q8(_) => 1,
            WeightRef::Q4(_) => 2,
            WeightRef::Gguf(_) => unreachable!("GGUF weights are never uploaded"),
        };
        #[repr(C)]
        struct ConvP {
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
        let mut p = ConvP {
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
        let (mut p_w, mut p_s, mut p_x, mut p_y, mut p_st, mut p_gb) = (cw.data, cw.scales, st.x, st.y, st.stats.0, st.gb.0);
        let (mut n_out, mut n_in_u) = (job.n_out as u32, n_in as u32);
        let mut params: [*mut c_void; 9] = [
            &mut p_w as *mut u64 as *mut c_void,
            &mut p_s as *mut u64 as *mut c_void,
            &mut p_x as *mut u64 as *mut c_void,
            &mut p_y as *mut u64 as *mut c_void,
            &mut n_out as *mut u32 as *mut c_void,
            &mut n_in_u as *mut u32 as *mut c_void,
            &mut p as *mut ConvP as *mut c_void,
            &mut p_st as *mut u64 as *mut c_void,
            &mut p_gb as *mut u64 as *mut c_void,
        ];
        if !self.launch(
            self.conv_funcs[fi],
            ((job.n_out / TILE) as u32, (t_pad / TILE) as u32),
            256,
            &mut params,
        ) {
            return false;
        }
        if unsafe { self.cu.cuCtxSynchronize() } != 0 {
            return false;
        }
        let t2 = Instant::now();
        if unsafe { self.cu.cuMemcpyDtoH_v2(job.y.as_mut_ptr() as *mut c_void, st.y, t * job.n_out * 4) } != 0 {
            return false;
        }
        self.stats.calls.fetch_add(1, Ordering::Relaxed);
        self.stats.copy_in_ns.fetch_add((t1 - t0).as_nanos() as u64, Ordering::Relaxed);
        self.stats.gpu_ns.fetch_add((t2 - t1).as_nanos() as u64, Ordering::Relaxed);
        self.stats.copy_out_ns.fetch_add(t2.elapsed().as_nanos() as u64, Ordering::Relaxed);
        true
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
        if !self.bind() {
            return false;
        }
        if q.len() < tq * d || k.len() < tk * d || v.len() < tk * d || out.len() < tq * d || tq == 0 || tk == 0 {
            return false;
        }
        let (dhp, dho, tkp, tqp) = (
            dh.next_multiple_of(32),
            dh.next_multiple_of(64),
            tk.next_multiple_of(64),
            tq.next_multiple_of(64),
        );
        // Per-head operands, zero padded (padding contributes nothing to any product).
        let mut qh = vec![0f32; heads * tqp * dhp];
        let mut kh = vec![0f32; heads * tkp * dhp];
        let mut vt = vec![0f32; heads * dho * tkp];
        for h in 0..heads {
            for i in 0..tq {
                qh[(h * tqp + i) * dhp..][..dh].copy_from_slice(&q[i * d + h * dh..][..dh]);
            }
            for j in 0..tk {
                kh[(h * tkp + j) * dhp..][..dh].copy_from_slice(&k[j * d + h * dh..][..dh]);
                for c in 0..dh {
                    vt[(h * dho + c) * tkp + j] = v[j * d + h * dh + c];
                }
            }
        }
        let mut st = self.staging.lock().unwrap();
        let st = &mut *st;
        let ok = self.grow(&mut st.aq, qh.len() * 4)
            && self.grow(&mut st.ak, kh.len() * 4)
            && self.grow(&mut st.av, vt.len() * 4)
            && self.grow(&mut st.asc, tqp * tkp * 4)
            && self.grow(&mut st.ao, heads * tqp * dho * 4);
        if !ok {
            return false;
        }
        let t0 = Instant::now();
        if !self.htod(st.aq.0, &qh) || !self.htod(st.ak.0, &kh) || !self.htod(st.av.0, &vt) {
            return false;
        }
        let t1 = Instant::now();
        let (mut cols, mut ld, mut scale, mut causal_u) = (tk as u32, tkp as u32, 1.0 / (dh as f32).sqrt(), causal as u32);
        for h in 0..heads {
            // Scores: keys as the "weights" ([tkp, dhp]), queries as the input.
            let (mut w, mut un, mut x, mut y) = (
                st.ak.0 + (h * tkp * dhp * 4) as u64,
                0u64,
                st.aq.0 + (h * tqp * dhp * 4) as u64,
                st.asc.0,
            );
            let (mut no, mut ni) = (tkp as u32, dhp as u32);
            let mut p: [*mut c_void; 6] = [
                &mut w as *mut u64 as *mut c_void,
                &mut un as *mut u64 as *mut c_void,
                &mut x as *mut u64 as *mut c_void,
                &mut y as *mut u64 as *mut c_void,
                &mut no as *mut u32 as *mut c_void,
                &mut ni as *mut u32 as *mut c_void,
            ];
            if !self.launch(self.aux_funcs[0], ((tkp / TILE) as u32, (tqp / TILE) as u32), 256, &mut p) {
                return false;
            }
            let mut sp = st.asc.0;
            let mut ps: [*mut c_void; 5] = [
                &mut sp as *mut u64 as *mut c_void,
                &mut cols as *mut u32 as *mut c_void,
                &mut ld as *mut u32 as *mut c_void,
                &mut scale as *mut f32 as *mut c_void,
                &mut causal_u as *mut u32 as *mut c_void,
            ];
            if !self.launch(self.aux_funcs[3], (tqp as u32, 1), 32, &mut ps) {
                return false;
            }
            // Output: transposed values as the "weights" ([dho, tkp]), probabilities as the input.
            let (mut w, mut x, mut y) = (st.av.0 + (h * dho * tkp * 4) as u64, st.asc.0, st.ao.0 + (h * tqp * dho * 4) as u64);
            let (mut no, mut ni) = (dho as u32, tkp as u32);
            let mut p: [*mut c_void; 6] = [
                &mut w as *mut u64 as *mut c_void,
                &mut un as *mut u64 as *mut c_void,
                &mut x as *mut u64 as *mut c_void,
                &mut y as *mut u64 as *mut c_void,
                &mut no as *mut u32 as *mut c_void,
                &mut ni as *mut u32 as *mut c_void,
            ];
            if !self.launch(self.aux_funcs[0], ((dho / TILE) as u32, (tqp / TILE) as u32), 256, &mut p) {
                return false;
            }
        }
        if unsafe { self.cu.cuCtxSynchronize() } != 0 {
            return false;
        }
        let t2 = Instant::now();
        let mut o = vec![0f32; heads * tqp * dho];
        if unsafe { self.cu.cuMemcpyDtoH_v2(o.as_mut_ptr() as *mut c_void, st.ao.0, o.len() * 4) } != 0 {
            return false;
        }
        for h in 0..heads {
            for i in 0..tq {
                out[i * d + h * dh..][..dh].copy_from_slice(&o[(h * tqp + i) * dho..][..dh]);
            }
        }
        self.stats.calls.fetch_add(1, Ordering::Relaxed);
        self.stats.copy_in_ns.fetch_add((t1 - t0).as_nanos() as u64, Ordering::Relaxed);
        self.stats.gpu_ns.fetch_add((t2 - t1).as_nanos() as u64, Ordering::Relaxed);
        self.stats.copy_out_ns.fetch_add(t2.elapsed().as_nanos() as u64, Ordering::Relaxed);
        true
    }

    fn matmul(&self, x: &[f32], t: usize, n_in: usize, jobs: &mut [Job]) -> bool {
        if !self.bind() {
            return false;
        }
        if jobs.is_empty() || !n_in.is_multiple_of(32) || x.len() < t * n_in {
            return false;
        }
        // The ggml GEMM takes the activations as pairs of f16 halves.
        if jobs.iter().any(|j| matches!(j.w, WeightRef::Gguf(_))) && !fits_halves(&x[..t * n_in]) {
            return false;
        }
        let t_pad = t.div_ceil(TILE) * TILE;
        let mut y_total = 0;
        for j in jobs.iter() {
            if j.n_out % TILE != 0 || j.y.len() < t * j.n_out || j.dev.downcast_ref::<CudaWeightBox>().is_none() {
                return false;
            }
            y_total += t_pad * j.n_out;
        }
        let mut st = self.staging.lock().unwrap();
        if !self.ensure_staging(&mut st, t_pad * n_in * 4, y_total * 4) {
            return false;
        }
        let t0 = Instant::now();
        unsafe {
            if self.cu.cuMemcpyHtoD_v2(st.x, x.as_ptr() as *const c_void, t * n_in * 4) != 0 {
                return false;
            }
            if t_pad > t {
                let zeros = vec![0f32; (t_pad - t) * n_in];
                if self
                    .cu
                    .cuMemcpyHtoD_v2(st.x + (t * n_in * 4) as u64, zeros.as_ptr() as *const c_void, zeros.len() * 4)
                    != 0
                {
                    return false;
                }
            }
        }
        let t1 = Instant::now();
        let mut y_off = 0usize;
        for j in jobs.iter() {
            let cw = &j.dev.downcast_ref::<CudaWeightBox>().unwrap().0;
            let y = st.y + (y_off * 4) as u64;
            let ok = match Wg::of(cw) {
                // GGUF rows: the tensor-core GEMM over the stored blocks.
                Some((k, w)) => self.gemm_gg(k, w, st.x, n_in, y, j.n_out, t, 0),
                None => match self.gemm_func(&j.w) {
                    Some(func) => self.gemm(func, cw, st.x, y, j.n_out, n_in, t_pad),
                    None => false,
                },
            };
            if !ok {
                return false;
            }
            y_off += t_pad * j.n_out;
        }
        if unsafe { self.cu.cuCtxSynchronize() } != 0 {
            return false;
        }
        let t2 = Instant::now();
        let mut y_off = 0usize;
        for j in jobs.iter_mut() {
            if unsafe {
                self.cu
                    .cuMemcpyDtoH_v2(j.y.as_mut_ptr() as *mut c_void, st.y + (y_off * 4) as u64, t * j.n_out * 4)
            } != 0
            {
                return false;
            }
            y_off += t_pad * j.n_out;
        }
        self.stats.calls.fetch_add(1, Ordering::Relaxed);
        self.stats.copy_in_ns.fetch_add((t1 - t0).as_nanos() as u64, Ordering::Relaxed);
        self.stats.gpu_ns.fetch_add((t2 - t1).as_nanos() as u64, Ordering::Relaxed);
        self.stats.copy_out_ns.fetch_add(t2.elapsed().as_nanos() as u64, Ordering::Relaxed);
        true
    }

    fn decodes(&self) -> bool {
        true
    }

    /// Measured on an RTX 5080: Qwen3.5-9B Q4 decodes at 111 tok/s plainly, 32 with MTP (the MTP
    /// layer drafts on the CPU); Qwen3-4B 4-bit at 173 plainly, 140 with its DSpark draft.
    fn prefers_plain_decode(&self) -> bool {
        true
    }

    fn decode_plan<'s>(&'s self, g: &crate::gpu::decode::Graph) -> Option<Box<dyn crate::gpu::decode::DecodePlan + 's>> {
        self.build_plan(g, 1)
            .map(|p| Box::new(p) as Box<dyn crate::gpu::decode::DecodePlan + 's>)
    }

    fn decode_plan_multi<'s>(
        &'s self,
        g: &crate::gpu::decode::Graph,
        tokens: usize,
    ) -> Option<Box<dyn crate::gpu::decode::DecodePlan + 's>> {
        self.build_plan(g, tokens)
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

    fn upload_moe(
        &self,
        gate_up: &[WeightRef],
        up: Option<&[WeightRef]>,
        down: &[WeightRef],
        h: usize,
        mi: usize,
        act: crate::gpu::MoeAct,
    ) -> Option<DeviceWeight> {
        self.moe_state(gate_up, up, down, h, mi, act, true)
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
        self.moe_state(gate_up, up, down, h, mi, act, false)
    }

    fn moe(&self, x: &[f32], t: usize, h: usize, mi: usize, routes: &[Vec<(usize, f32)>], dev: &DeviceWeight, y: &mut [f32]) -> bool {
        self.moe_run(x, t, h, mi, routes, dev, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::{ConvGeom, NormKind, NormSpec};
    use crate::kernels;
    use crate::quant::{Q4Matrix, QMatrix, dequantize_rows_q4, f16_to_f32, from_mlx_q4, mlx_dequant_row, quantize_bf16, quantize_q4};

    /// The backend, or None without an NVIDIA GPU. With one, only a missing NVRTC (a driver without
    /// the toolkit) may skip: a kernel that fails to compile or load must fail the tests, since
    /// the backend would otherwise never run anywhere.
    fn backend() -> Option<CudaBackend> {
        let plat = crate::platform::detect();
        let g = plat.gpus.iter().find(|g| g.kind == crate::platform::GpuKind::Cuda)?;
        match CudaBackend::new(g.index) {
            Ok(b) => Some(b),
            Err(e) if e.to_string().starts_with("libnvrtc") => None,
            Err(e) => panic!("{e}"),
        }
    }

    fn rnd(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }
    fn bf16(n: usize, seed: &mut u64) -> Vec<u16> {
        (0..n).map(|_| (rnd(seed).to_bits() >> 16) as u16).collect()
    }
    fn f32s(n: usize, seed: &mut u64) -> Vec<f32> {
        (0..n).map(|_| rnd(seed) * 2.0).collect()
    }
    /// Token rows of very different magnitudes.
    fn input(t: usize, n: usize, seed: &mut u64) -> Vec<f32> {
        (0..t * n).map(|i| rnd(seed) * [1e-3, 1.0, 300.0, 2e5][(i / n) % 4]).collect()
    }
    /// Worst error of any row relative to that row's largest reference value.
    fn row_err(y: &[f32], r: &[f32], n: usize) -> f32 {
        assert_eq!(y.len(), r.len());
        y.chunks(n)
            .zip(r.chunks(n))
            .map(|(a, b)| {
                let m = b.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30);
                a.iter().zip(b).fold(0f32, |e, (p, q)| e.max((p - q).abs())) / m
            })
            .fold(0f32, f32::max)
    }
    /// `[t, n_out]` product with `[n_out, n_in]` f32 weights, exactly (f64).
    fn gemm_ref(w: &[f32], n_out: usize, n_in: usize, x: &[f32], t: usize) -> Vec<f32> {
        let mut y = vec![0f32; t * n_out];
        for i in 0..t {
            for o in 0..n_out {
                y[i * n_out + o] = (0..n_in).map(|k| w[o * n_in + k] as f64 * x[i * n_in + k] as f64).sum::<f64>() as f32;
            }
        }
        y
    }
    fn dense_bf(w: &[u16]) -> Vec<f32> {
        w.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect()
    }
    fn dense_q8(m: &QMatrix) -> Vec<f32> {
        let mut f = vec![0f32; m.n_out * m.n_in];
        for (i, o) in f.iter_mut().enumerate() {
            let (r, k) = (i / m.n_in, i % m.n_in);
            *o = m.q[i] as f32 * m.scales[r * (m.n_in / 32) + k / 32];
        }
        f
    }
    fn dense_q4(m: &Q4Matrix) -> Vec<f32> {
        let mut f = vec![0f32; m.n_out * m.n_in];
        dequantize_rows_q4(m, 0, m.n_out, &mut f);
        f
    }
    fn wref<'a>(w: &'a WeightRef<'a>) -> WeightRef<'a> {
        match w {
            WeightRef::Bf16(v) => WeightRef::Bf16(v),
            WeightRef::Q8(m) => WeightRef::Q8(m),
            WeightRef::Q4(m) => WeightRef::Q4(m),
            WeightRef::Gguf(m) => WeightRef::Gguf(m),
        }
    }
    fn run_matmul(be: &CudaBackend, w: &WeightRef, n_out: usize, n_in: usize, x: &[f32], t: usize) -> Vec<f32> {
        let dev = be.upload(w, n_out, n_in).expect("upload");
        let mut y = vec![f32::NAN; t * n_out];
        let mut jobs = [Job {
            w: wref(w),
            dev: &dev,
            n_out,
            y: &mut y,
        }];
        assert!(be.matmul(x, t, n_in, &mut jobs), "matmul declined");
        y
    }
    /// An MLX-style affine matrix and its values straight from `w = scale * q + bias`.
    fn affine(n_out: usize, n_in: usize, group: usize, seed: &mut u64) -> (Q4Matrix, Vec<f32>) {
        let gpr = n_in / group;
        let words: Vec<u32> = (0..n_out * n_in / 8)
            .map(|_| {
                *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (*seed >> 32) as u32
            })
            .collect();
        let to = |v: f32| half::bf16::from_f32(v).to_bits();
        let scales: Vec<u16> = (0..n_out * gpr).map(|_| to(0.002 + rnd(seed).abs() * 0.02)).collect();
        let biases: Vec<u16> = (0..n_out * gpr).map(|_| to(rnd(seed))).collect();
        let m = from_mlx_q4(&words, &scales, &biases, n_out, n_in, group);
        let mut f = vec![0f32; n_out * n_in];
        for r in 0..n_out {
            let wr = &words[r * n_in / 8..(r + 1) * n_in / 8];
            mlx_dequant_row(
                wr,
                &scales[r * gpr..(r + 1) * gpr],
                &biases[r * gpr..(r + 1) * gpr],
                4,
                group,
                &mut f[r * n_in..(r + 1) * n_in],
            );
        }
        (m, f)
    }

    #[test]
    fn matmul_matches_cpu_and_reference() {
        let Some(be) = backend() else { return };
        let (n_out, n_in) = (128, 512);
        let mut seed = 1;
        let w = bf16(n_out * n_in, &mut seed);
        let (q8, q4) = (quantize_bf16(&w, n_out, n_in), quantize_q4(&w, n_out, n_in));
        let formats = [
            (WeightRef::Bf16(&w), dense_bf(&w)),
            (WeightRef::Q8(&q8), dense_q8(&q8)),
            (WeightRef::Q4(&q4), dense_q4(&q4)),
        ];
        // Whole tiles, a partial tile, and fewer tokens than one tile.
        for t in [64, 150, 37, 1] {
            let x = input(t, n_in, &mut seed);
            for (w, dense) in &formats {
                let y = run_matmul(&be, w, n_out, n_in, &x, t);
                let e = row_err(&y, &gemm_ref(dense, n_out, n_in, &x, t), n_out);
                assert!(e < 1e-4, "t {t}: error {e} against the reference");
                // Below a tile the CPU's quantized kernels are the int8 decode ones, approximate by
                // design (3e-3 here): the exact reference above is what binds there.
                if t < 32 {
                    continue;
                }
                let mut c = vec![0f32; t * n_out];
                match w {
                    WeightRef::Bf16(v) => kernels::matmul(v, n_out, n_in, &x, t, &mut c),
                    WeightRef::Q8(m) => kernels::matmul_quant_multi(&mut [(*m, &mut c[..])], &x, t),
                    WeightRef::Q4(m) => kernels::matmul_quant_multi(&mut [(*m, &mut c[..])], &x, t),
                    WeightRef::Gguf(_) => unreachable!(),
                }
                let e = row_err(&y, &c, n_out);
                assert!(e < 1e-4, "t {t}: error {e} against the CPU kernels");
            }
        }
    }

    /// Several outputs per input in one submission, of different widths and formats, then a bigger
    /// batch (the staging buffers grow) and a smaller one again.
    #[test]
    fn matmul_batches_and_staging_growth() {
        let Some(be) = backend() else { return };
        let n_in = 256;
        let mut seed = 2;
        let (a, b) = (bf16(64 * n_in, &mut seed), bf16(192 * n_in, &mut seed));
        let q4 = quantize_q4(&b, 192, n_in);
        let (wa, wb) = (WeightRef::Bf16(&a), WeightRef::Q4(&q4));
        let (da, db) = (be.upload(&wa, 64, n_in).unwrap(), be.upload(&wb, 192, n_in).unwrap());
        let (ra, rb) = (dense_bf(&a), dense_q4(&q4));
        for t in [40, 700, 40] {
            let x = input(t, n_in, &mut seed);
            let (mut ya, mut yb) = (vec![0f32; t * 64], vec![0f32; t * 192]);
            let mut jobs = [
                Job {
                    w: wref(&wa),
                    dev: &da,
                    n_out: 64,
                    y: &mut ya,
                },
                Job {
                    w: wref(&wb),
                    dev: &db,
                    n_out: 192,
                    y: &mut yb,
                },
            ];
            assert!(be.matmul(&x, t, n_in, &mut jobs));
            assert!(row_err(&ya, &gemm_ref(&ra, 64, n_in, &x, t), 64) < 1e-4);
            assert!(row_err(&yb, &gemm_ref(&rb, 192, n_in, &x, t), 192) < 1e-4);
        }
    }

    #[test]
    fn affine_q4_matmul_matches_mlx_definition() {
        let Some(be) = backend() else { return };
        let (n_out, n_in) = (128, 512);
        let mut seed = 3;
        // MLX's group of 64, and groups of one block and of the whole row.
        for group in [64, 32, 512] {
            let (m, truth) = affine(n_out, n_in, group, &mut seed);
            assert!(m.offsets.is_some());
            for t in [64, 150, 37] {
                let x = input(t, n_in, &mut seed);
                let y = run_matmul(&be, &WeightRef::Q4(&m), n_out, n_in, &x, t);
                // The offset and the f16 scale are rounded from the definition's values.
                let e = row_err(&y, &gemm_ref(&dense_q4(&m), n_out, n_in, &x, t), n_out);
                assert!(e < 1e-4, "group {group} t {t}: error {e} against the dequantized matrix");
                let e = row_err(&y, &gemm_ref(&truth, n_out, n_in, &x, t), n_out);
                assert!(e < 2e-2, "group {group} t {t}: error {e} against w = scale * q + bias");
                let mut c = vec![0f32; t * n_out];
                kernels::matmul_quant_multi(&mut [(&m, &mut c[..])], &x, t);
                let e = row_err(&y, &c, n_out);
                assert!(e < 1e-4, "group {group} t {t}: error {e} against the CPU kernels");
            }
        }
    }

    fn qbuf<T: Copy>(v: &[T]) -> crate::quant::QBuf<T> {
        let mut p = crate::quant::PageVec::<T>::zeroed(v.len());
        // SAFETY: `p` holds `v.len()` elements.
        unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), p.as_mut_ptr(), v.len()) };
        p.into()
    }

    /// The kernels' hand-written f16 decoding, on every finite f16 scale (subnormals included).
    #[test]
    fn q4_scales_decode_every_finite_f16() {
        let Some(be) = backend() else { return };
        let (n_out, n_in) = (65536, 32);
        let f16s: Vec<u16> = (0..=u16::MAX).map(|h| if (h >> 10) & 31 == 31 { 0 } else { h }).collect();
        // Element 0 is +1 (low nibble 9), the rest 0 (nibble 8).
        let mut q = vec![0x88u8; n_out * n_in / 2];
        q.iter_mut().step_by(n_in / 2).for_each(|b| *b = 0x89);
        let m = Q4Matrix {
            q: qbuf(&q),
            scales: qbuf(&f16s),
            n_out,
            n_in,
            offsets: None,
        };
        let t = 64;
        let mut x = vec![0f32; t * n_in];
        x.chunks_mut(n_in).for_each(|r| r[0] = 1.0);
        let y = run_matmul(&be, &WeightRef::Q4(&m), n_out, n_in, &x, t);
        for (i, &h) in f16s.iter().enumerate() {
            assert_eq!(y[i], f16_to_f32(h), "f16 bits {h:#06x}");
        }
    }

    /// The input after an optional normalization, from its definition.
    fn normalized(x: &[f32], npx: usize, c: usize, n: Option<&NormSpec>) -> Vec<f32> {
        let Some(n) = n else { return x[..npx * c].to_vec() };
        let mut o = vec![0f32; npx * c];
        let stats = |vals: &mut dyn Iterator<Item = f32>| {
            let v: Vec<f64> = vals.map(|a| a as f64).collect();
            let mean = v.iter().sum::<f64>() / v.len() as f64;
            let var = v.iter().map(|a| (a - mean) * (a - mean)).sum::<f64>() / v.len() as f64;
            (mean, 1.0 / (var + n.eps as f64).sqrt())
        };
        let mut apply = |p: usize, ch: usize, (mean, inv): (f64, f64)| {
            let v = (x[p * c + ch] as f64 - mean) * inv * n.gamma[ch] as f64 + n.beta[ch] as f64;
            o[p * c + ch] = (if n.silu { v / (1.0 + (-v).exp()) } else { v }) as f32;
        };
        match n.kind {
            NormKind::Group(g) => {
                let cpg = c / g;
                for gi in 0..g {
                    let s = stats(&mut (0..npx).flat_map(|p| (gi * cpg..(gi + 1) * cpg).map(move |ch| x[p * c + ch])));
                    for p in 0..npx {
                        for ch in gi * cpg..(gi + 1) * cpg {
                            apply(p, ch, s);
                        }
                    }
                }
            }
            NormKind::Layer => {
                for p in 0..npx {
                    let s = stats(&mut (0..c).map(|ch| x[p * c + ch]));
                    for ch in 0..c {
                        apply(p, ch, s);
                    }
                }
            }
        }
        o
    }
    /// The convolution from its definition: weights `[n_out][ky][kx][cin]`, zeros outside the image.
    fn conv_ref(w: &[f32], n_out: usize, x: &[f32], g: &ConvGeom) -> Vec<f32> {
        let n_in = g.k * g.k * g.cin;
        let mut y = vec![0f32; g.oh * g.ow * n_out];
        for oy in 0..g.oh {
            for ox in 0..g.ow {
                for o in 0..n_out {
                    let mut s = 0f64;
                    for ky in 0..g.k {
                        for kx in 0..g.k {
                            let iy = (oy * g.stride + ky) as isize - g.pad as isize;
                            let ix = (ox * g.stride + kx) as isize - g.pad as isize;
                            if iy < 0 || ix < 0 || iy >= g.h as isize || ix >= g.w as isize {
                                continue;
                            }
                            for ci in 0..g.cin {
                                let wv = w[o * n_in + (ky * g.k + kx) * g.cin + ci];
                                s += wv as f64 * x[(iy as usize * g.w + ix as usize) * g.cin + ci] as f64;
                            }
                        }
                    }
                    y[(oy * g.ow + ox) * n_out + o] = s as f32;
                }
            }
        }
        y
    }

    #[test]
    fn conv2d_matches_definition() {
        let Some(be) = backend() else { return };
        let mut seed = 4;
        let geo = |h, w, cin, k, stride, pad, oh, ow| ConvGeom {
            h,
            w,
            cin,
            k,
            stride,
            pad,
            oh,
            ow,
        };
        let cases = [
            geo(20, 30, 32, 3, 1, 1, 20, 30), // "same"
            geo(20, 30, 64, 3, 2, 0, 10, 15), // stride 2, the zero padding after only (VAE encoder)
            geo(21, 31, 32, 3, 2, 1, 11, 16), // stride 2, padded on both sides, partial tile
            geo(16, 16, 64, 1, 1, 0, 16, 16), // 1x1
            geo(100, 1, 64, 1, 1, 0, 100, 1), // a linear layer as a convolution
            geo(9, 7, 32, 5, 1, 2, 9, 7),     // fewer pixels than one tile, wide kernel
        ];
        for g in cases {
            let (n_out, npx, n_in) = (64, g.h * g.w, g.k * g.k * g.cin);
            let x = f32s(npx * g.cin, &mut seed);
            let gamma: Vec<f32> = (0..g.cin).map(|_| 1.0 + rnd(&mut seed)).collect();
            let beta: Vec<f32> = (0..g.cin).map(|_| rnd(&mut seed)).collect();
            let gamma1 = vec![1f32; g.cin];
            let beta0 = vec![0f32; g.cin];
            let group = |silu| NormSpec {
                kind: NormKind::Group(8),
                eps: 1e-6,
                gamma: &gamma,
                beta: &beta,
                silu,
            };
            let layer = |silu| NormSpec {
                kind: NormKind::Layer,
                eps: 1e-6,
                gamma: &gamma,
                beta: &beta,
                silu,
            };
            let plain_scale = NormSpec {
                kind: NormKind::Group(1),
                eps: 1e-5,
                gamma: &gamma1,
                beta: &beta0,
                silu: false,
            };
            let norms = [
                None,
                Some(group(false)),
                Some(group(true)),
                Some(layer(false)),
                Some(layer(true)),
                Some(plain_scale),
            ];
            let w = bf16(n_out * n_in, &mut seed);
            let (q8, q4) = (quantize_bf16(&w, n_out, n_in), quantize_q4(&w, n_out, n_in));
            let formats = [
                (WeightRef::Bf16(&w), dense_bf(&w)),
                (WeightRef::Q8(&q8), dense_q8(&q8)),
                (WeightRef::Q4(&q4), dense_q4(&q4)),
            ];
            for (wr, dense) in &formats {
                let dev = be.upload(wr, n_out, n_in).unwrap();
                for n in &norms {
                    let mut y = vec![f32::NAN; g.oh * g.ow * n_out];
                    let mut job = Job {
                        w: wref(wr),
                        dev: &dev,
                        n_out,
                        y: &mut y,
                    };
                    assert!(be.conv2d(&x, &g, n.as_ref(), &mut job), "{g:?} declined");
                    let r = conv_ref(dense, n_out, &normalized(&x, npx, g.cin, n.as_ref()), &g);
                    let e = row_err(&y, &r, n_out);
                    assert!(e < 2e-3, "{g:?} norm {:?}: error {e}", n.map(|n| (n.kind, n.silu)));
                }
            }
        }
    }

    /// Cases the kernels cannot take are declined, so the caller runs the CPU path.
    #[test]
    fn conv2d_declines_what_it_cannot_run() {
        let Some(be) = backend() else { return };
        let mut seed = 5;
        let g = ConvGeom {
            h: 8,
            w: 8,
            cin: 32,
            k: 1,
            stride: 1,
            pad: 0,
            oh: 8,
            ow: 8,
        };
        let x = f32s(64 * 32, &mut seed);
        let (m, _) = affine(64, 32, 32, &mut seed);
        let wr = WeightRef::Q4(&m);
        let dev = be.upload(&wr, 64, 32).unwrap();
        let mut y = vec![0f32; 64 * 64];
        let mut job = Job {
            w: wref(&wr),
            dev: &dev,
            n_out: 64,
            y: &mut y,
        };
        assert!(!be.conv2d(&x, &g, None, &mut job), "no affine convolution kernel");
        // A group count that does not divide the channels.
        let w = bf16(64 * 32, &mut seed);
        let wr = WeightRef::Bf16(&w);
        let dev = be.upload(&wr, 64, 32).unwrap();
        let mut job = Job {
            w: wref(&wr),
            dev: &dev,
            n_out: 64,
            y: &mut y,
        };
        let (ga, be0) = (vec![1f32; 32], vec![0f32; 32]);
        let n = NormSpec {
            kind: NormKind::Group(5),
            eps: 1e-5,
            gamma: &ga,
            beta: &be0,
            silu: false,
        };
        assert!(!be.conv2d(&x, &g, Some(&n), &mut job));
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_ref(q: &[f32], k: &[f32], v: &[f32], tq: usize, tk: usize, heads: usize, dh: usize, causal: bool) -> Vec<f32> {
        let d = heads * dh;
        let mut o = vec![0f32; tq * d];
        let scale = 1.0 / (dh as f64).sqrt();
        for h in 0..heads {
            for i in 0..tq {
                let lim = if causal { (i + 1).min(tk) } else { tk };
                let s: Vec<f64> = (0..lim)
                    .map(|j| {
                        (0..dh)
                            .map(|c| q[i * d + h * dh + c] as f64 * k[j * d + h * dh + c] as f64)
                            .sum::<f64>()
                            * scale
                    })
                    .collect();
                let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = s.iter().map(|a| (a - m).exp()).collect();
                let sum: f64 = e.iter().sum();
                for c in 0..dh {
                    o[i * d + h * dh + c] = ((0..lim).map(|j| e[j] * v[j * d + h * dh + c] as f64).sum::<f64>() / sum) as f32;
                }
            }
        }
        o
    }

    #[test]
    fn attention_matches_definition() {
        let Some(be) = backend() else { return };
        let mut seed = 6;
        for (tq, tk, heads, dh) in [
            (100, 100, 8, 64),
            (64, 64, 4, 128),
            (70, 130, 3, 40), // dh not a multiple of 32, cross attention
            (130, 70, 2, 33), // more queries than keys
            (1, 5, 1, 32),
            (5, 1, 2, 16),
            (1024, 1024, 16, 64), // a prefill chunk
        ] {
            let d = heads * dh;
            let (q, k, v) = (f32s(tq * d, &mut seed), f32s(tk * d, &mut seed), f32s(tk * d, &mut seed));
            for causal in [false, true] {
                let mut out = vec![f32::NAN; tq * d];
                assert!(be.attention(&q, &k, &v, tq, tk, heads, dh, causal, &mut out));
                let r = attention_ref(&q, &k, &v, tq, tk, heads, dh, causal);
                let e = row_err(&out, &r, d);
                assert!(e < 1e-3, "{tq}x{tk} {heads}x{dh} causal {causal}: error {e}");
            }
        }
    }

    /// CUDA contexts are per thread: everything must work from a thread other than the one that
    /// created the backend (a host loading on one thread and generating on another).
    #[test]
    fn works_from_another_thread() {
        let Some(be) = backend() else { return };
        let be = std::sync::Arc::new(be);
        let (n_out, n_in, t) = (64, 64, 64);
        let mut seed = 7;
        let w = bf16(n_out * n_in, &mut seed);
        let x = input(t, n_in, &mut seed);
        let (b2, w2, x2) = (be.clone(), w.clone(), x.clone());
        let y = std::thread::spawn(move || run_matmul(&b2, &WeightRef::Bf16(&w2), n_out, n_in, &x2, t))
            .join()
            .expect("matmul from another thread");
        assert!(row_err(&y, &gemm_ref(&dense_bf(&w), n_out, n_in, &x, t), n_out) < 1e-4);
        // Weights uploaded here, used there.
        let wr = WeightRef::Bf16(&w);
        let dev = be.upload(&wr, n_out, n_in).unwrap();
        let (r, b3, x3) = (gemm_ref(&dense_bf(&w), n_out, n_in, &x, t), be.clone(), x.clone());
        std::thread::scope(|s| {
            s.spawn(|| {
                let mut y = vec![0f32; t * n_out];
                let mut jobs = [Job {
                    w: WeightRef::Bf16(&w),
                    dev: &dev,
                    n_out,
                    y: &mut y,
                }];
                assert!(b3.matmul(&x3, t, n_in, &mut jobs));
                assert!(row_err(&y, &r, n_out) < 1e-4);
            });
        });
    }

    /// One device shared by racing threads: the staging buffers must not mix their work up.
    /// The routed experts against their definition in f64: fused affine Q4 gate/up with SiLU,
    /// split bf16 gate and up with tanh GELU and with the clamped SiLU; experts that get no
    /// tokens, segments of whole and partial tiles, and `y` added to rather than overwritten.
    #[test]
    fn moe_matches_definition() {
        use crate::gpu::MoeAct;
        let Some(be) = backend() else { return };
        let (n_exp, h, mi, k) = (6, 128, 64, 2);
        let mut seed = 11;
        for (case, act) in [MoeAct::Silu, MoeAct::GeluTanh, MoeAct::SiluClamp(0.4)].into_iter().enumerate() {
            let split = case > 0;
            // Dense values of each expert's projections next to their storage.
            let (mut gq, mut gd, mut ub, mut ud, mut dq, mut dd) = (vec![], vec![], vec![], vec![], vec![], vec![]);
            for _ in 0..n_exp {
                if split {
                    let (g, u) = (bf16(mi * h, &mut seed), bf16(mi * h, &mut seed));
                    gd.push(dense_bf(&g));
                    ud.push(dense_bf(&u));
                    gq.push(None);
                    ub.push(Some((g, u)));
                } else {
                    // The repacked matrix (its offsets are rounded from the MLX biases).
                    let (m, _) = affine(2 * mi, h, 64, &mut seed);
                    gd.push(dense_q4(&m));
                    gq.push(Some(m));
                    ub.push(None);
                }
                let (m, _) = affine(h, mi, 32, &mut seed);
                dd.push(dense_q4(&m));
                dq.push(m);
            }
            let gate_up: Vec<WeightRef> = (0..n_exp)
                .map(|e| match (&gq[e], &ub[e]) {
                    (Some(m), _) => WeightRef::Q4(m),
                    (_, Some((g, _))) => WeightRef::Bf16(g),
                    _ => unreachable!(),
                })
                .collect();
            let up: Vec<WeightRef> = ub.iter().flatten().map(|(_, u)| WeightRef::Bf16(u)).collect();
            let down: Vec<WeightRef> = dq.iter().map(WeightRef::Q4).collect();
            let dev = be
                .upload_moe(&gate_up, split.then_some(&up[..]), &down, h, mi, act)
                .expect("upload_moe");
            // Expert 4 gets no tokens; expert 0 more than one tile.
            let t = 90;
            let routes: Vec<Vec<(usize, f32)>> = (0..t)
                .map(|i| {
                    let a = if i < 70 { 0 } else { [1, 2, 3, 5][i % 4] };
                    vec![
                        (a, 0.25 + rnd(&mut seed).abs()),
                        ([1, 2, 3, 5][(i / 3) % 4], 0.1 + rnd(&mut seed).abs()),
                    ]
                })
                .map(|mut r: Vec<(usize, f32)>| {
                    if r[0].0 == r[1].0 {
                        r[1].0 = if r[0].0 == 1 { 2 } else { 1 };
                    }
                    r
                })
                .collect();
            assert_eq!(routes[0].len(), k);
            let x = f32s(t * h, &mut seed);
            let y0 = f32s(t * h, &mut seed);
            let mut y = y0.clone();
            assert!(be.moe(&x, t, h, mi, &routes, &dev, &mut y), "moe declined");
            let mut r = vec![0f32; t * h];
            for ti in 0..t {
                let xr = &x[ti * h..(ti + 1) * h];
                let mut acc: Vec<f64> = y0[ti * h..(ti + 1) * h].iter().map(|&v| v as f64).collect();
                for &(e, w) in &routes[ti] {
                    let dot = |m: &[f32], row: usize, n: usize, v: &[f64]| -> f64 { (0..n).map(|c| m[row * n + c] as f64 * v[c]).sum() };
                    let xv: Vec<f64> = xr.iter().map(|&v| v as f64).collect();
                    let a: Vec<f64> = (0..mi)
                        .map(|j| {
                            let (g, u) = if split {
                                (dot(&gd[e], j, h, &xv), dot(&ud[e], j, h, &xv))
                            } else {
                                (dot(&gd[e], j, h, &xv), dot(&gd[e], mi + j, h, &xv))
                            };
                            match act {
                                MoeAct::Silu => g / (1.0 + (-g).exp()) * u,
                                MoeAct::GeluTanh => 0.5 * g * (1.0 + (0.7978845608 * (g + 0.044715 * g * g * g)).tanh()) * u,
                                MoeAct::SiluClamp(l) => {
                                    let (g, l) = (g.min(l as f64), l as f64);
                                    g / (1.0 + (-g).exp()) * u.clamp(-l, l)
                                }
                                MoeAct::SwigluOai { .. } | MoeAct::Situ { .. } => unreachable!("not a device activation"),
                            }
                        })
                        .collect();
                    for (c, o) in acc.iter_mut().enumerate() {
                        *o += w as f64 * dot(&dd[e], c, mi, &a);
                    }
                }
                for (c, o) in acc.iter().enumerate() {
                    r[ti * h + c] = *o as f32;
                }
            }
            let e = row_err(&y, &r, h);
            assert!(e < 1e-4, "{act:?}: error {e}");
            // The same experts as transient copies (streamed weights): the same result.
            let tdev = be
                .upload_moe_transient(&gate_up, split.then_some(&up[..]), &down, h, mi, act)
                .expect("upload_moe_transient");
            let mut y2 = y0.clone();
            assert!(be.moe(&x, t, h, mi, &routes, &tdev, &mut y2), "moe declined (transient)");
            assert!(y2 == y, "{act:?}: transient experts differ");
            drop(tdev);
        }
    }

    /// The decode matvecs (`cuda_decode.cu`): every format and output mode against the f64
    /// definition, and the multi-token kernel bit-identical to the single-token one per token
    /// (what makes a verification pass equal to one-token steps).
    #[test]
    fn decode_matvecs_match_definition_and_each_other() {
        let Some(be) = backend() else { return };
        assert!(be.bind());
        #[repr(C)]
        #[derive(Clone, Copy, Default)]
        struct Dw {
            q: u64,
            s: u64,
            o: u64,
            gpr: u32,
            gk: u32,
        }
        let dw = |cw: &CudaWeight| Dw {
            q: cw.data,
            s: cw.scales,
            o: cw.offsets,
            gpr: cw.gpr,
            gk: cw.gk,
        };
        // 100 rows: a partial group of 4 rows per warp, and of 16 per block.
        let (n_out, n_in, t) = (100, 2560, 5);
        let mut seed = 21;
        let w = bf16(n_out * n_in, &mut seed);
        let w2 = bf16(n_out * n_in, &mut seed);
        let (q8, q8b) = (quantize_bf16(&w, n_out, n_in), quantize_bf16(&w2, n_out, n_in));
        let (q4, q4b) = (quantize_q4(&w, n_out, n_in), quantize_q4(&w2, n_out, n_in));
        let (qa, _) = affine(n_out, n_in, 64, &mut seed);
        let (qa2, _) = affine(n_out, n_in, 64, &mut seed);
        type Case<'a> = (&'a str, WeightRef<'a>, WeightRef<'a>, Vec<f32>, Vec<f32>);
        let cases: Vec<Case> = vec![
            ("bf", WeightRef::Bf16(&w), WeightRef::Bf16(&w2), dense_bf(&w), dense_bf(&w2)),
            ("q8", WeightRef::Q8(&q8), WeightRef::Q8(&q8b), dense_q8(&q8), dense_q8(&q8b)),
            ("q4", WeightRef::Q4(&q4), WeightRef::Q4(&q4b), dense_q4(&q4), dense_q4(&q4b)),
            ("q4a", WeightRef::Q4(&qa), WeightRef::Q4(&qa2), dense_q4(&qa), dense_q4(&qa2)),
        ];
        // Token rows 800 floats apart (a multi-token plan's padded stride), one gate per token.
        let xs = 2624usize;
        let x: Vec<f32> = (0..t * xs).map(|_| rnd(&mut seed) * 2.0).collect();
        let gates: Vec<f32> = (0..t).map(|_| rnd(&mut seed) * 4.0).collect();
        let y0: Vec<f32> = (0..t * n_out).map(|_| rnd(&mut seed)).collect();
        let dx = be.upload_bytes(x.as_ptr() as *const u8, x.len() * 4).unwrap();
        let dg = be.upload_bytes(gates.as_ptr() as *const u8, gates.len() * 4).unwrap();
        let mut dy = 0u64;
        assert_eq!(unsafe { be.cu.cuMemAlloc_v2(&mut dy, t * n_out * 4) }, 0);
        let down = |n: usize| -> Vec<f32> {
            let mut v = vec![0f32; n];
            assert_eq!(unsafe { be.cu.cuMemcpyDtoH_v2(v.as_mut_ptr() as *mut c_void, dy, n * 4) }, 0);
            v
        };
        for (name, a, b, da, db) in &cases {
            let (ca, cb) = (dw(&be.upload_weight(a).unwrap()), dw(&be.upload_weight(b).unwrap()));
            let (f1, fk) = (be.func(&format!("dmv_{name}")).unwrap(), be.func(&format!("dmvk_{name}")).unwrap());
            for mode in 0u32..5 {
                // Single-token kernel, token by token.
                assert!(be.htod(dy, &y0));
                for j in 0..t {
                    let (xj, yj, gj) = (dx + (j * xs * 4) as u64, dy + (j * n_out * 4) as u64, dg + (j * 4) as u64);
                    let (no, ni) = (n_out as u32, n_in as u32);
                    let mut args = [
                        &ca as *const _ as *mut c_void,
                        &cb as *const _ as *mut c_void,
                        &xj as *const _ as *mut c_void,
                        &yj as *const _ as *mut c_void,
                        &no as *const _ as *mut c_void,
                        &ni as *const _ as *mut c_void,
                        &mode as *const _ as *mut c_void,
                        &gj as *const _ as *mut c_void,
                    ];
                    assert!(be.launch(f1, ((n_out as u32).div_ceil(16), 1), 128, &mut args));
                }
                let single = down(t * n_out);
                // Multi-token kernel, all tokens at once.
                assert!(be.htod(dy, &y0));
                let (no, ni, xsu, ysu, gs, nt) = (n_out as u32, n_in as u32, xs as u32, n_out as u32, 1u32, t as u32);
                let mut args = [
                    &ca as *const _ as *mut c_void,
                    &cb as *const _ as *mut c_void,
                    &dx as *const _ as *mut c_void,
                    &xsu as *const _ as *mut c_void,
                    &dy as *const _ as *mut c_void,
                    &ysu as *const _ as *mut c_void,
                    &no as *const _ as *mut c_void,
                    &ni as *const _ as *mut c_void,
                    &mode as *const _ as *mut c_void,
                    &dg as *const _ as *mut c_void,
                    &gs as *const _ as *mut c_void,
                    &nt as *const _ as *mut c_void,
                ];
                assert!(be.launch(fk, ((n_out as u32).div_ceil(32), 1), 256, &mut args));
                let multi = down(t * n_out);
                assert!(
                    single.iter().zip(&multi).all(|(p, q)| p.to_bits() == q.to_bits()),
                    "{name} mode {mode}: the multi-token kernel differs from the single-token one"
                );
                // The definition, in f64.
                let mut r = vec![0f32; t * n_out];
                for j in 0..t {
                    let xr = &x[j * xs..j * xs + n_in];
                    for o in 0..n_out {
                        let dot = |m: &[f32]| (0..n_in).map(|k| m[o * n_in + k] as f64 * xr[k] as f64).sum::<f64>();
                        let (av, bv, y) = (dot(da), dot(db), y0[j * n_out + o] as f64);
                        let g = gates[j] as f64;
                        r[j * n_out + o] = match mode {
                            0 => av,
                            1 => y + av,
                            2 => av / (1.0 + (-av).exp()) * bv,
                            3 => 0.5 * av * (1.0 + (0.7978845608 * (av + 0.044715 * av * av * av)).tanh()) * bv,
                            _ => y + av / (1.0 + (-g).exp()),
                        } as f32;
                    }
                }
                let e = row_err(&multi, &r, n_out);
                assert!(e < 1e-5, "{name} mode {mode}: error {e} against the definition");
            }
        }
    }

    /// The ggml kernels (`inkling_cuda.cu` compiled per format): the matvec and the GEMM of every
    /// format with device kernels against the dequantized rows' products in f64.
    #[test]
    fn ggml_matvec_and_gemm_match_definition() {
        use crate::ggml::{GgmlType, GgufMatrix, random_blocks};
        use GgmlType::*;
        let Some(be) = backend() else { return };
        assert!(be.bind());
        // 100 rows: a partial block of 8 warps and a partial 64-row tile; 70 tokens: a partial
        // token tile.
        let (n_out, n_in, t) = (100usize, 1024usize, 70usize);
        let xs = 1088usize;
        let mut seed = 77;
        let x: Vec<f32> = (0..t * xs).map(|_| rnd(&mut seed) * 2.0).collect();
        let y0: Vec<f32> = (0..t * n_out).map(|_| rnd(&mut seed)).collect();
        let dx = be.upload_bytes(x.as_ptr() as *const u8, x.len() * 4).unwrap();
        let mut dy = 0u64;
        assert_eq!(unsafe { be.cu.cuMemAlloc_v2(&mut dy, t * n_out * 4) }, 0);
        for kind in [
            F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
            IQ3_S, IQ4_NL, IQ4_XS, MXFP4,
        ] {
            let bytes = random_blocks(kind, n_out * n_in / kind.block_size(), &mut seed);
            let m = GgufMatrix::from_bytes(kind, &bytes, n_out, n_in).unwrap();
            let mut dense = vec![0f32; n_out * n_in];
            m.dequantize_rows(0, n_out, &mut dense);
            let cw = be
                .upload_weight(&WeightRef::Gguf(&m))
                .unwrap_or_else(|| panic!("{kind}: no device weight"));
            let (k, w) = Wg::of(&cw).unwrap();
            for mode in 0u32..2 {
                let mut r = vec![0f32; t * n_out];
                let mut mag = vec![0f64; t * n_out];
                for j in 0..t {
                    let xr = &x[j * xs..j * xs + n_in];
                    for o in 0..n_out {
                        let row = &dense[o * n_in..(o + 1) * n_in];
                        let dot: f64 = row.iter().zip(xr).map(|(&a, &b)| a as f64 * b as f64).sum();
                        mag[j * n_out + o] = row.iter().zip(xr).map(|(&a, &b)| (a as f64 * b as f64).abs()).sum();
                        r[j * n_out + o] = (dot + if mode == 1 { y0[j * n_out + o] as f64 } else { 0.0 }) as f32;
                    }
                }
                let check = |what: &str, tol: f64| {
                    let mut y = vec![0f32; t * n_out];
                    assert_eq!(unsafe { be.cu.cuMemcpyDtoH_v2(y.as_mut_ptr() as *mut c_void, dy, y.len() * 4) }, 0);
                    let worst = y
                        .iter()
                        .zip(&r)
                        .zip(&mag)
                        .map(|((&a, &b), &m)| (a as f64 - b as f64).abs() / (m + 1e-3))
                        .fold(0f64, f64::max);
                    assert!(worst < tol, "{kind} {what} mode {mode}: {worst:.2e} of the summed magnitudes");
                };
                // The matvec: a grid row per token.
                assert!(be.htod(dy, &y0));
                let (ni, no, xsu, ysu) = (n_in as u32, n_out as u32, xs as u32, n_out as u32);
                let mut args = [
                    &w as *const _ as *mut c_void,
                    &dx as *const _ as *mut c_void,
                    &dy as *const _ as *mut c_void,
                    &ni as *const _ as *mut c_void,
                    &no as *const _ as *mut c_void,
                    &xsu as *const _ as *mut c_void,
                    &ysu as *const _ as *mut c_void,
                    &mode as *const _ as *mut c_void,
                ];
                assert!(be.launch(k.mv, (no.div_ceil(8), t as u32), 256, &mut args));
                check("matvec", 2e-6);
                // The GEMM (rows of x `xs` apart).
                assert!(be.htod(dy, &y0));
                let p = GemmP {
                    x: dx,
                    y: dy,
                    n_in: ni,
                    n_out: no,
                    rows: t as u32,
                    xs: xsu,
                    ys: ysu,
                    mode,
                    ..GemmP::default()
                };
                let mut args = [&w as *const _ as *mut c_void, &p as *const _ as *mut c_void];
                assert!(be.launch(k.gemmi, (no.div_ceil(64), (t as u32).div_ceil(64)), 128, &mut args));
                check("gemm", 2e-6);
            }
            be.free_weight(&cw);
        }
        // Activations the GEMM's f16 halves cannot hold are declined too.
        {
            let bytes = random_blocks(Q4_K, 64 * n_in / Q4_K.block_size(), &mut seed);
            let m = GgufMatrix::from_bytes(Q4_K, &bytes, 64, n_in).unwrap();
            let dev = be.upload(&WeightRef::Gguf(&m), 64, n_in).unwrap();
            let mut xb: Vec<f32> = (0..64 * n_in).map(|_| rnd(&mut seed)).collect();
            let mut y = vec![0f32; 64 * 64];
            let mut go = |xb: &[f32]| {
                let mut jobs = [Job {
                    w: WeightRef::Gguf(&m),
                    dev: &dev,
                    n_out: 64,
                    y: &mut y,
                }];
                be.matmul(xb, 64, n_in, &mut jobs)
            };
            assert!(go(&xb));
            xb[7 * n_in + 3] = -4.0e4;
            assert!(!go(&xb));
            xb[7 * n_in + 3] = f32::NAN;
            assert!(!go(&xb));
        }
        // Formats without kernels are declined (the CPU path keeps them).
        let bytes = random_blocks(TQ2_0, n_out * n_in / TQ2_0.block_size(), &mut seed);
        let m = GgufMatrix::from_bytes(TQ2_0, &bytes, n_out, n_in).unwrap();
        assert!(be.upload_weight(&WeightRef::Gguf(&m)).is_none());
    }

    /// A decode graph over GGUF weights (an attention layer with gated heads and a gate / up
    /// pair, a Gated DeltaNet layer) as a single-token plan, a multi-token plan and a prefill
    /// plan: the multi-token plan's steps give the single-token steps' bits, the prefill plan's
    /// (GEMMs, the other ops over all tokens at once, steps that start at a later position and
    /// end inside a token tile) their values up to the GEMMs' rounding, for the hidden states,
    /// the keys and values and the recurrent states.
    #[test]
    fn ggml_plans_match_single_token_steps() {
        use crate::ggml::{GgmlType, GgufMatrix, random_blocks};
        use crate::gpu::decode::{DecodePlan, GdnRule, Graph, LinMode, Op, Slot, Step};
        use crate::model::LayerCache;
        use GgmlType::*;
        let Some(be) = backend() else { return };
        let (h, nh, nkv, hd, inter) = (256usize, 4usize, 2usize, 64usize, 512usize);
        let (nk, nv, hdk, hdv, kc) = (2usize, 4usize, 32usize, 32usize, 4usize);
        let (key_dim, value_dim) = (nk * hdk, nv * hdv);
        let conv_dim = 2 * key_dim + value_dim;
        let qw = 2 * nh * hd;
        let mut seed = 99u64;
        // (format, rows, columns, factor on the outputs: keeps the activations near 1).
        let specs = [
            (Q4_K, qw, h, 0.02f32),
            (Q8_0, nkv * hd, h, 0.02),
            (IQ4_XS, nkv * hd, h, 0.02),
            (Q6_K, h, nh * hd, 0.02),
            (IQ3_S, inter, h, 0.02),
            (IQ3_S, inter, h, 0.02),
            (Q5_K, h, inter, 0.02),
            (Q3_K, conv_dim, h, 0.02),
            (Q4_0, value_dim, h, 0.05),
            (F32, nv, h, 1.0),
            (Q8_0, nv, h, 0.05),
            (Q4_0, h, value_dim, 0.05),
        ];
        let ms: Vec<GgufMatrix> = specs
            .iter()
            .map(|&(kind, n_out, n_in, f)| {
                let bytes = random_blocks(kind, n_out * n_in / kind.block_size(), &mut seed);
                let mut m = GgufMatrix::from_bytes(kind, &bytes, n_out, n_in).unwrap();
                if !kind.is_float() {
                    m.set_factor(f).unwrap();
                }
                m
            })
            .collect();
        let devs: Vec<DeviceWeight> = ms
            .iter()
            .map(|m| be.upload(&WeightRef::Gguf(m), m.n_out, m.n_in).expect("device weight"))
            .collect();
        let ones = vec![1f32; h];
        let qn: Vec<f32> = (0..hd).map(|_| 1.0 + rnd(&mut seed) * 0.2).collect();
        let conv_w: Vec<f32> = (0..conv_dim * kc).map(|_| rnd(&mut seed)).collect();
        let decay: Vec<f32> = (0..nv).map(|_| -0.5 - rnd(&mut seed).abs()).collect();
        let dt_bias: Vec<f32> = (0..nv).map(|_| rnd(&mut seed)).collect();
        let norm_w: Vec<f32> = (0..hdv).map(|_| 1.0 + rnd(&mut seed) * 0.2).collect();
        let mut g = Graph::new();
        let x = g.buf(h);
        let hn = g.buf(h);
        let qkv = g.buf(qw + 2 * nkv * hd);
        let qo = g.buf(nh * hd);
        let att = g.buf(nh * hd);
        let act = g.buf(inter);
        let gproj = g.buf(conv_dim + value_dim + 2 * nv);
        let gconv = g.buf(conv_dim);
        let gout = g.buf(value_dim);
        let hidden = g.buf(h);
        (g.input, g.hidden, g.n_hidden) = (x, hidden, h);
        g.rope_halves = vec![hd / 2];
        g.kv = vec![(nkv, hd)];
        g.states = vec![nv * hdk * hdv, (kc - 1) * conv_dim];
        let lin = |i: usize, x, y, mode| Op::Linear {
            w: WeightRef::Gguf(&ms[i]),
            dev: &devs[i],
            w2: None,
            n_out: ms[i].n_out,
            n_in: ms[i].n_in,
            x,
            y,
            mode,
        };
        g.ops.push(Op::norm(x, hn, &ones, h, 1e-6));
        g.ops.push(lin(0, hn, qkv, LinMode::Store));
        g.ops.push(lin(1, hn, qkv.at(qw), LinMode::Store));
        g.ops.push(lin(2, hn, qkv.at(qw + nkv * hd), LinMode::Store));
        g.ops.push(Op::QkRope {
            qkv,
            q_out: qo,
            kv: 0,
            nh,
            nkv,
            hd,
            half_rot: hd / 2,
            q_stride: 2 * hd,
            k_off: qw,
            v_off: qw + nkv * hd,
            interleaved: false,
            q_norm: Some(&qn),
            k_norm: Some(&qn),
            eps: 1e-6,
            rope: 0,
            v_norm: false,
            v_from_k: false,
        });
        g.ops.push(Op::Attend {
            q: qo,
            out: att,
            kv: 0,
            nh,
            nkv,
            hd,
            scale: 0.05,
            window: None,
            gate: Some((qkv.at(hd), 2 * hd)),
            block: None,
        });
        g.ops.push(lin(3, att, x, LinMode::Add));
        g.ops.push(Op::norm(x, hn, &ones, h, 1e-6));
        g.ops.push(Op::Linear {
            w: WeightRef::Gguf(&ms[4]),
            dev: &devs[4],
            w2: Some((WeightRef::Gguf(&ms[5]), &devs[5])),
            n_out: inter,
            n_in: h,
            x: hn,
            y: act,
            mode: LinMode::SiluPair,
        });
        g.ops.push(lin(6, act, x, LinMode::Add));
        g.ops.push(Op::norm(x, hn, &ones, h, 1e-6));
        g.ops.push(lin(7, hn, gproj, LinMode::Store));
        g.ops.push(lin(8, hn, gproj.at(conv_dim), LinMode::Store));
        g.ops.push(lin(9, hn, gproj.at(conv_dim + value_dim), LinMode::Store));
        g.ops.push(lin(10, hn, gproj.at(conv_dim + value_dim + nv), LinMode::Store));
        g.ops.push(Op::GdnConv {
            x: gproj,
            out: gconv,
            state: 1,
            w: &conv_w,
            dim: conv_dim,
            kc,
        });
        g.ops.push(Op::GdnRule(GdnRule {
            c: gconv,
            z: gproj.at(conv_dim),
            b: gproj.at(conv_dim + value_dim),
            a: gproj.at(conv_dim + value_dim + nv),
            out: gout,
            state: 0,
            nk,
            nv,
            hdk,
            hdv,
            tiled: true,
            sigmoid_gate: false,
            decay_coef: &decay,
            dt_bias: &dt_bias,
            norm_w: &norm_w,
            eps: 1e-6,
        }));
        g.ops.push(lin(11, gout, x, LinMode::Add));
        g.ops.push(Op::norm(x, hidden, &ones, h, 1e-6));
        g.head_from = g.ops.len();

        // 107 tokens: more than three attention batches, and a partial GEMM tile.
        let n = 107usize;
        let xin: Vec<f32> = (0..n * h).map(|_| rnd(&mut seed) * 2.0).collect();
        let half = hd / 2;
        let cs: Vec<f32> = (0..n)
            .flat_map(|p| {
                let ang = move |j: usize| p as f32 * 0.37f32.powf(j as f32 / half as f32);
                (0..half).map(move |j| ang(j).cos()).chain((0..half).map(move |j| ang(j).sin()))
            })
            .collect();
        // Runs the tokens in steps of the given sizes: hidden states, keys, values, states.
        let run = |plan: &dyn DecodePlan, steps: &[usize]| -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
            let mut slot = Slot(None);
            let mut cache = LayerCache::new(nkv, hd);
            let mut s0 = vec![0f32; nv * hdk * hdv];
            let mut s1 = vec![0f32; (kc - 1) * conv_dim];
            let mut hs = vec![0f32; n * h];
            let mut pos = 0;
            for &t in steps {
                let mut kv = [&mut cache];
                let mut states: [&mut [f32]; 2] = [&mut s0, &mut s1];
                assert!(
                    plan.step(
                        &mut slot,
                        Step {
                            pos,
                            t,
                            rope_cs: &cs[pos * hd..(pos + t) * hd],
                            x: &xin[pos * h..(pos + t) * h],
                            x2: &[],
                            kv: &mut kv,
                            states: &mut states,
                            hidden: &mut hs[pos * h..(pos + t) * h],
                            logits: None,
                            params: None,
                            capture: None,
                            scratch: false,
                            states_kept: false,
                        },
                    ),
                    "step of {t} tokens at {pos}"
                );
                pos += t;
            }
            assert_eq!(pos, n);
            let (k, v) = cache.data();
            s0.extend_from_slice(&s1);
            (hs, k.to_vec(), v.to_vec(), s0)
        };
        let one = be.build_plan(&g, 1).expect("single-token plan");
        let multi = be.build_plan(&g, 8).expect("multi-token plan");
        let pf = be.build_plan_pf(&g, 2048).expect("prefill plan");
        let r1 = run(&one, &vec![1; n]);
        assert!(r1.0.iter().all(|v| v.is_finite()) && r1.0.iter().any(|&v| v.abs() > 0.1));
        // Multi-token steps: a grid row per token of the same kernels.
        let rm = run(&multi, &[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 3]);
        for (what, a, b) in [
            ("hidden", &r1.0, &rm.0),
            ("keys", &r1.1, &rm.1),
            ("values", &r1.2, &rm.2),
            ("states", &r1.3, &rm.3),
        ] {
            assert!(
                a.iter().zip(b).all(|(p, q)| p.to_bits() == q.to_bits()),
                "multi-token plan: {what} differ from single-token steps"
            );
        }
        // Prefill steps: 40 tokens, then 67 on top of them (GEMMs from 8 tokens on).
        let rp = run(&pf, &[40, 67]);
        for (what, a, b) in [
            ("hidden", &r1.0, &rp.0),
            ("keys", &r1.1, &rp.1),
            ("values", &r1.2, &rp.2),
            ("states", &r1.3, &rp.3),
        ] {
            let scale = a.iter().fold(0f32, |m, v| m.max(v.abs()));
            let worst = a.iter().zip(b).fold(0f32, |m, (p, q)| m.max((p - q).abs()));
            assert!(
                worst <= 2e-4 * scale,
                "prefill plan: {what} differ by {worst:.2e} (largest value {scale:.2e})"
            );
        }
        // An input the GEMMs cannot take (here not a number) declines the step and leaves the
        // host's caches and states alone.
        {
            let mut slot = Slot(None);
            let mut cache = LayerCache::new(nkv, hd);
            let mut s0 = vec![0f32; nv * hdk * hdv];
            let mut s1 = vec![0f32; (kc - 1) * conv_dim];
            let mut bad = xin[..16 * h].to_vec();
            bad[5 * h + 9] = f32::NAN;
            let mut hs = vec![0f32; 16 * h];
            let mut kv = [&mut cache];
            let mut states: [&mut [f32]; 2] = [&mut s0, &mut s1];
            let ok = pf.step(
                &mut slot,
                Step {
                    pos: 0,
                    t: 16,
                    rope_cs: &cs[..16 * hd],
                    x: &bad,
                    x2: &[],
                    kv: &mut kv,
                    states: &mut states,
                    hidden: &mut hs,
                    logits: None,
                    params: None,
                    capture: None,
                    scratch: false,
                    states_kept: false,
                },
            );
            assert!(!ok, "a step with a non-finite activation must be declined");
            assert!(cache.len == 0 && s0.iter().chain(&s1).all(|&v| v == 0.0));
        }
    }

    /// `cargo test --profile ci --lib ggml_bench -- --ignored --nocapture`: the ggml matvec (GB/s
    /// of weights read) and GEMM (TFLOP/s at 512 tokens) on matrices of Qwen3.8-27B's shapes.
    /// `OXINFER_GGML_CU=file` compiles the kernels from a file, `BENCH_KINDS=23,21` picks formats.
    #[test]
    #[ignore]
    fn ggml_bench() {
        use crate::ggml::{GgmlType, GgufMatrix, random_blocks};
        let Some(be) = backend() else { return };
        assert!(be.bind());
        let kinds: Vec<u32> = std::env::var("BENCH_KINDS")
            .map(|v| v.split(',').map(|k| k.parse().unwrap()).collect())
            .unwrap_or_else(|_| vec![23, 21, 12]);
        let t = 512usize;
        let mut seed = 5;
        for id in kinds {
            let kind = GgmlType::from_id(id).unwrap();
            for (n_out, n_in) in [(17408usize, 5120usize), (5120, 17408)] {
                let bytes = random_blocks(kind, n_out * n_in / kind.block_size(), &mut seed);
                let m = GgufMatrix::from_bytes(kind, &bytes, n_out, n_in).unwrap();
                let cw = be.upload_weight(&WeightRef::Gguf(&m)).unwrap();
                let (k, w) = Wg::of(&cw).unwrap();
                let x: Vec<f32> = (0..t * n_in).map(|_| rnd(&mut seed)).collect();
                let dx = be.upload_bytes(x.as_ptr() as *const u8, x.len() * 4).unwrap();
                let mut dy = 0u64;
                assert_eq!(unsafe { be.cu.cuMemAlloc_v2(&mut dy, t * n_out * 4) }, 0);
                let sync = || assert_eq!(unsafe { be.cu.cuCtxSynchronize() }, 0);
                let (ni, no, zero) = (n_in as u32, n_out as u32, 0u32);
                let mv = || {
                    let mut args = [
                        &w as *const _ as *mut c_void,
                        &dx as *const _ as *mut c_void,
                        &dy as *const _ as *mut c_void,
                        &ni as *const _ as *mut c_void,
                        &no as *const _ as *mut c_void,
                        &zero as *const _ as *mut c_void,
                        &zero as *const _ as *mut c_void,
                        &zero as *const _ as *mut c_void,
                    ];
                    assert!(be.launch(k.mv, (no.div_ceil(8), 1), 256, &mut args));
                };
                let p = GemmP {
                    x: dx,
                    y: dy,
                    n_in: ni,
                    n_out: no,
                    rows: t as u32,
                    xs: ni,
                    ys: no,
                    ..GemmP::default()
                };
                let gemm = || assert!(be.gemm_gg_p(k, w, p));
                mv();
                gemm();
                sync();
                let t0 = Instant::now();
                for _ in 0..50 {
                    mv();
                }
                sync();
                let mv_s = t0.elapsed().as_secs_f64() / 50.0;
                let t0 = Instant::now();
                for _ in 0..5 {
                    gemm();
                }
                sync();
                let g_s = t0.elapsed().as_secs_f64() / 5.0;
                eprintln!(
                    "{kind} {n_out} x {n_in}: matvec {:.0} us = {:.0} GB/s; gemm of {t} tokens {:.2} ms = {:.1} TFLOP/s",
                    mv_s * 1e6,
                    (n_out * m.row_bytes()) as f64 / mv_s / 1e9,
                    g_s * 1e3,
                    2.0 * (t * n_out * n_in) as f64 / g_s / 1e12
                );
                be.free_weight(&cw);
                unsafe {
                    be.cu.cuMemFree_v2(dx);
                    be.cu.cuMemFree_v2(dy);
                }
            }
        }
    }

    /// `cargo test --release --lib decode_matvec_bench -- --ignored --nocapture`: the decode
    /// matvecs on a 12288 x 4096 Q4 matrix, one token and four.
    #[test]
    #[ignore]
    fn decode_matvec_bench() {
        let Some(be) = backend() else { return };
        assert!(be.bind());
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct Dw {
            q: u64,
            s: u64,
            o: u64,
            gpr: u32,
            gk: u32,
        }
        let (n_out, n_in) = (12288usize, 4096usize);
        let mut seed = 5;
        for name in ["q4", "bf"] {
            let w = bf16(n_out * n_in, &mut seed);
            let q = quantize_q4(&w, n_out, n_in);
            let cw = if name == "q4" {
                be.upload_weight(&WeightRef::Q4(&q)).unwrap()
            } else {
                be.upload_weight(&WeightRef::Bf16(&w)).unwrap()
            };
            let d = Dw {
                q: cw.data,
                s: cw.scales,
                o: cw.offsets,
                gpr: cw.gpr,
                gk: cw.gk,
            };
            let x = vec![0.5f32; 8 * n_in];
            let dx = be.upload_bytes(x.as_ptr() as *const u8, x.len() * 4).unwrap();
            let mut dy = 0u64;
            unsafe { be.cu.cuMemAlloc_v2(&mut dy, 8 * n_out * 4) };
            let bytes = if name == "q4" {
                n_out * n_in / 2 + n_out * n_in / 16
            } else {
                n_out * n_in * 2
            };
            let f1 = be.func(&format!("dmv_{name}")).unwrap();
            let fk = be.func(&format!("dmvk_{name}")).unwrap();
            let (no, ni, mode, g0) = (n_out as u32, n_in as u32, 0u32, 0u64);
            let time = |f: &dyn Fn()| {
                f();
                unsafe { be.cu.cuCtxSynchronize() };
                let t0 = std::time::Instant::now();
                for _ in 0..50 {
                    f();
                }
                unsafe { be.cu.cuCtxSynchronize() };
                t0.elapsed().as_secs_f64() / 50.0
            };
            let s1 = time(&|| {
                let mut a = [
                    &d as *const _ as *mut c_void,
                    &d as *const _ as *mut c_void,
                    &dx as *const _ as *mut c_void,
                    &dy as *const _ as *mut c_void,
                    &no as *const _ as *mut c_void,
                    &ni as *const _ as *mut c_void,
                    &mode as *const _ as *mut c_void,
                    &g0 as *const _ as *mut c_void,
                ];
                be.launch(f1, (no.div_ceil(16), 1), 128, &mut a);
            });
            for nt in [1u32, 4, 8] {
                let (xs, ys, gs) = (ni, no, 0u32);
                let sk = time(&|| {
                    let mut a = [
                        &d as *const _ as *mut c_void,
                        &d as *const _ as *mut c_void,
                        &dx as *const _ as *mut c_void,
                        &xs as *const _ as *mut c_void,
                        &dy as *const _ as *mut c_void,
                        &ys as *const _ as *mut c_void,
                        &no as *const _ as *mut c_void,
                        &ni as *const _ as *mut c_void,
                        &mode as *const _ as *mut c_void,
                        &g0 as *const _ as *mut c_void,
                        &gs as *const _ as *mut c_void,
                        &nt as *const _ as *mut c_void,
                    ];
                    be.launch(fk, (no.div_ceil(32), 1), 256, &mut a);
                });
                eprintln!(
                    "{name}: single {:.0} us ({:.0} GB/s); multi t={nt} {:.0} us ({:.0} GB/s)",
                    s1 * 1e6,
                    bytes as f64 / s1 / 1e9,
                    sk * 1e6,
                    bytes as f64 / sk / 1e9
                );
            }
        }
    }

    #[test]
    fn concurrent_calls_do_not_mix() {
        let Some(be) = backend() else { return };
        let (n_out, n_in, t) = (64, 128, 128);
        let mut seed = 8;
        let w = bf16(n_out * n_in, &mut seed);
        let dense = dense_bf(&w);
        let wr = WeightRef::Bf16(&w);
        let dev = be.upload(&wr, n_out, n_in).unwrap();
        let inputs: Vec<Vec<f32>> = (0..8).map(|_| input(t, n_in, &mut seed)).collect();
        std::thread::scope(|s| {
            for x in &inputs {
                let (be, w, dense, dev) = (&be, &w, &dense, &dev);
                s.spawn(move || {
                    let r = gemm_ref(dense, n_out, n_in, x, t);
                    for _ in 0..8 {
                        let mut y = vec![0f32; t * n_out];
                        let mut jobs = [Job {
                            w: WeightRef::Bf16(w),
                            dev,
                            n_out,
                            y: &mut y,
                        }];
                        assert!(be.matmul(x, t, n_in, &mut jobs));
                        assert!(row_err(&y, &r, n_out) < 1e-4);
                    }
                });
            }
        });
    }
}
