//! CUDA backend (Linux / Windows, NVIDIA): the driver API and NVRTC are loaded at run time with
//! `dlsym`, so the binary has no CUDA dependency and runs unchanged on machines without a GPU.
//! The GEMM kernels are compiled from source by NVRTC for the detected compute capability.
//!
//! Only prefill-sized matmuls are offloaded (see `wants`): weights are copied to device memory
//! the first time a layer is used (skipped when VRAM is short, that layer then stays on the CPU),
//! activations are copied up and results down per batch. Discrete GPUs pay for those transfers,
//! so the threshold is higher than Metal's. Decode stays on the CPU.
//!
//! This backend is compile-checked but has not been run on NVIDIA hardware yet.

use crate::gpu::{Backend, DeviceWeight, Job, WeightRef};
use anyhow::{Result, anyhow, bail};
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

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
__device__ __forceinline__ float f16f(unsigned short h) { return __half2float(__ushort_as_half(h)); }

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
    float m = -INFINITY;
    for (unsigned int i = threadIdx.x; i < lim; i += 32) m = fmaxf(m, row[i] * scale);
    m = warp_max(m);
    float sum = 0.f;
    for (unsigned int i = threadIdx.x; i < lim; i += 32) { float e = expf(row[i] * scale - m); row[i] = e; sum += e; }
    float inv = 1.f / warp_sum(sum);
    for (unsigned int i = threadIdx.x; i < ld; i += 32) row[i] = i < lim ? row[i] * inv : 0.f;
}
"#;

struct CudaWeight {
    data: u64,
    scales: u64,
    /// Affine Q4: device offsets (0 if none), offsets per row, blocks of 32 per offset group.
    offsets: u64,
    gpr: u32,
    gk: u32,
}

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
}

#[derive(Default)]
struct Stats {
    calls: AtomicU64,
    copy_in_ns: AtomicU64,
    gpu_ns: AtomicU64,
    copy_out_ns: AtomicU64,
}

pub struct CudaBackend {
    cu: Cu,
    _module: *mut c_void,
    /// bf16, q8, q4
    funcs: [*mut c_void; 3],
    /// GEMM over affine Q4 weights.
    func_q4a: *mut c_void,
    /// Implicit-GEMM convolutions, same order.
    conv_funcs: [*mut c_void; 3],
    /// gemm_f32, group_stats, row_stats, softmax_rows.
    aux_funcs: [*mut c_void; 4],
    staging: Mutex<Staging>,
    stats: Stats,
    name: String,
}
unsafe impl Send for CudaBackend {}
unsafe impl Sync for CudaBackend {}

impl CudaBackend {
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
        let nvrtc = {
            use dlsym::Bindings;
            // SAFETY: as above.
            unsafe { Nvrtc::open_first(NVRTC_CANDIDATES.iter().copied()) }
                .map_err(|e| anyhow!("libnvrtc (ships with the CUDA toolkit): {e}"))?
        };
        let src = CString::new(KERNEL_SRC)?;
        let name = CString::new("oxinfer.cu")?;
        let mut prog: *mut c_void = std::ptr::null_mut();
        let rc = unsafe { nvrtc.nvrtcCreateProgram(&mut prog, src.as_ptr(), name.as_ptr(), 0, std::ptr::null(), std::ptr::null()) };
        if rc != 0 {
            bail!("nvrtcCreateProgram failed ({rc})");
        }
        let arch = CString::new(format!("--gpu-architecture=compute_{major}{minor}"))?;
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
        unsafe { nvrtc.nvrtcGetPTXSize(prog, &mut n) };
        let mut ptx = vec![0u8; n.max(1)];
        unsafe { nvrtc.nvrtcGetPTX(prog, ptx.as_mut_ptr() as *mut c_char) };
        unsafe { nvrtc.nvrtcDestroyProgram(&mut prog) };
        if *ptx.last().unwrap_or(&1) != 0 {
            ptx.push(0);
        }
        let mut module: *mut c_void = std::ptr::null_mut();
        check(
            &cu,
            unsafe { cu.cuModuleLoadData(&mut module, ptx.as_ptr() as *const c_void) },
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
        let mut conv_funcs = [std::ptr::null_mut(); 3];
        for (i, fname) in [c"conv_bf", c"conv_q8", c"conv_q4"].iter().enumerate() {
            check(
                &cu,
                unsafe { cu.cuModuleGetFunction(&mut conv_funcs[i], module, fname.as_ptr()) },
                "cuModuleGetFunction",
            )?;
        }
        Ok(Self {
            cu,
            _module: module,
            funcs,
            func_q4a,
            conv_funcs,
            aux_funcs,
            staging: Mutex::new(Staging {
                x: 0,
                x_cap: 0,
                y: 0,
                y_cap: 0,
                stats: (0, 0),
                gb: (0, 0),
                aq: (0, 0),
                ak: (0, 0),
                av: (0, 0),
                asc: (0, 0),
                ao: (0, 0),
            }),
            stats: Stats::default(),
            name: format!("cuda (sm_{major}{minor})"),
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
        let rc = unsafe {
            self.cu.cuLaunchKernel(
                f,
                grid.0,
                grid.1,
                1,
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

/// Device weights are never freed individually: a model's layers live as long as the process.
struct CudaWeightBox(CudaWeight);
unsafe impl Send for CudaWeightBox {}
unsafe impl Sync for CudaWeightBox {}

impl Backend for CudaBackend {
    fn name(&self) -> &str {
        &self.name
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
        let cw = match w {
            WeightRef::Gguf(_) => return None,
            WeightRef::Bf16(w) => CudaWeight {
                data: self.upload_bytes(w.as_ptr() as *const u8, w.len() * 2)?,
                scales: 0,
                offsets: 0,
                gpr: 0,
                gk: 0,
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
                }
            }
        };
        Some(Box::new(CudaWeightBox(cw)))
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
        if jobs.is_empty() || !n_in.is_multiple_of(32) || x.len() < t * n_in || jobs.iter().any(|j| matches!(j.w, WeightRef::Gguf(_))) {
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
            let fi = match j.w {
                WeightRef::Bf16(_) => 0,
                WeightRef::Q8(_) => 1,
                WeightRef::Q4(_) => 2,
                WeightRef::Gguf(_) => unreachable!("GGUF weights are never uploaded"),
            };
            let mut p_w = cw.data;
            let mut p_s = cw.scales;
            let mut p_x = st.x;
            let mut p_y = st.y + (y_off * 4) as u64;
            let mut n_out = j.n_out as u32;
            let mut n_in_u = n_in as u32;
            let (mut p_o, mut gpr, mut gk) = (cw.offsets, cw.gpr, cw.gk);
            let mut params: [*mut c_void; 9] = [
                &mut p_w as *mut u64 as *mut c_void,
                &mut p_s as *mut u64 as *mut c_void,
                &mut p_x as *mut u64 as *mut c_void,
                &mut p_y as *mut u64 as *mut c_void,
                &mut n_out as *mut u32 as *mut c_void,
                &mut n_in_u as *mut u32 as *mut c_void,
                &mut p_o as *mut u64 as *mut c_void,
                &mut gpr as *mut u32 as *mut c_void,
                &mut gk as *mut u32 as *mut c_void,
            ];
            // The plain kernels take the first six parameters; the affine one all nine.
            let func = if cw.offsets != 0 { self.func_q4a } else { self.funcs[fi] };
            let rc = unsafe {
                self.cu.cuLaunchKernel(
                    func,
                    (j.n_out / TILE) as u32,
                    (t_pad / TILE) as u32,
                    1,
                    256,
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
}
