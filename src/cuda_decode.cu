// Single-token decode kernels (`cuda_decode.rs`), appended to the GEMM source. Ports of the Metal
// decode kernels (`metal_decode.metal`): same arithmetic per op, warps for SIMD-groups.
// `sp` points at the step's parameters on the device: sp[0] = the token's position.

__device__ __forceinline__ float dsig(float v) { return 1.0f / (1.0f + expf(-v)); }
__device__ __forceinline__ float dgelu(float a) {
    float t = 0.7978845608f * (a + 0.044715f * a * a * a);
    return 0.5f * a * (1.0f + tanhf(t));
}
// Sum over the block (every thread calls it; blockDim a multiple of 32, at most 1024).
__device__ __forceinline__ float dblock_sum(float v, float* red) {
    v = warp_sum(v);
    unsigned int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = (blockDim.x + 31) >> 5;
    __syncthreads();
    if (l == 0) red[w] = v;
    __syncthreads();
    float t = 0.f;
    for (unsigned int i = 0; i < nw; i++) t += red[i];
    return t;
}

// ---- Matrix-vector products. Weight formats: 0 bf16, 1 Q8 (f32 scale per 32), 2 Q4 (f16 scale per
// 32, value (q - 8) * d), 3 affine Q4 (plus a bf16 offset per group of gk blocks), 4 f32. A row is
// read in 16-byte chunks (8, 16 or 32 values; f32: 32 bytes, 8 values); rows are 16-byte aligned
// (n_in % 32 == 0). Every kernel sums a chunk with `WC::dot` and adds the chunks of a lane in order
// with explicit roundings, so the single- and multi-token kernels give the same bits.
struct DW { const unsigned char* q; const unsigned char* s; const unsigned short* o; unsigned int gpr; unsigned int gk; };

__device__ __forceinline__ void x8(const float* x, unsigned int i, float* v) {
    if ((((size_t)(x + i)) & 15) == 0) {
        float4 a = *(const float4*)(x + i), b = *(const float4*)(x + i + 4);
        v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
    } else {
        for (int j = 0; j < 8; j++) v[j] = x[i + j];
    }
}
__device__ __forceinline__ float bfl(unsigned int u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bfh(unsigned int u) { return __uint_as_float(u & 0xffff0000u); }

// Values per chunk of a format.
template <int F> struct CH { static constexpr unsigned int N = F == 0 || F == 4 ? 8 : (F == 1 ? 16 : 32); };

// One chunk of one row, loaded (and for Q4 unpacked) once, dotted with any number of x chunks.
template <int F> struct WC;
template <> struct WC<0> {
    uint4 u;
    __device__ __forceinline__ void load(DW w, unsigned int row, unsigned int c, unsigned int n_in) {
        u = ((const uint4*)(w.q + (size_t)row * n_in * 2))[c];
    }
    __device__ __forceinline__ float dot(const float* v) const {
        return bfl(u.x) * v[0] + bfh(u.x) * v[1] + bfl(u.y) * v[2] + bfh(u.y) * v[3] + bfl(u.z) * v[4] + bfh(u.z) * v[5] +
               bfl(u.w) * v[6] + bfh(u.w) * v[7];
    }
};
template <> struct WC<4> {
    float4 a, b;
    __device__ __forceinline__ void load(DW w, unsigned int row, unsigned int c, unsigned int n_in) {
        const float4* p = (const float4*)(w.q + (size_t)row * n_in * 4);
        a = p[2 * c];
        b = p[2 * c + 1];
    }
    __device__ __forceinline__ float dot(const float* v) const {
        return a.x * v[0] + a.y * v[1] + a.z * v[2] + a.w * v[3] + b.x * v[4] + b.y * v[5] + b.z * v[6] + b.w * v[7];
    }
};
template <> struct WC<1> {
    uint4 u;
    float d;
    __device__ __forceinline__ void load(DW w, unsigned int row, unsigned int c, unsigned int n_in) {
        u = ((const uint4*)(w.q + (size_t)row * n_in))[c];
        d = ((const float*)w.s)[(size_t)row * (n_in / 32) + c / 2];
    }
    __device__ __forceinline__ float dot(const float* v) const {
        unsigned int ws[4] = { u.x, u.y, u.z, u.w };
        float s = 0.f;
        for (int k = 0; k < 4; k++) {
            unsigned int q = ws[k];
            s += (float)(signed char)(q & 255) * v[4 * k] + (float)(signed char)((q >> 8) & 255) * v[4 * k + 1] +
                 (float)(signed char)((q >> 16) & 255) * v[4 * k + 2] + (float)(signed char)(q >> 24) * v[4 * k + 3];
        }
        return __fmul_rn(s, d);
    }
};
// Unpacked once per chunk (a multi-token step dots it with every token's values).
template <int AFF> struct WCQ4 {
    float lo[16], hi[16];
    float d, m;
    __device__ __forceinline__ void load(DW w, unsigned int row, unsigned int b, unsigned int n_in) {
        uint4 u = ((const uint4*)(w.q + (size_t)row * (n_in / 2)))[b];
        unsigned int ws[4] = { u.x, u.y, u.z, u.w };
        for (int k = 0; k < 4; k++)
            for (int j = 0; j < 4; j++) {
                unsigned int byte = (ws[k] >> (8 * j)) & 255;
                lo[k * 4 + j] = (float)(byte & 15);
                hi[k * 4 + j] = (float)(byte >> 4);
            }
        d = f16f(((const unsigned short*)w.s)[(size_t)row * (n_in / 32) + b]);
        m = AFF ? bf16f(w.o[(size_t)row * w.gpr + b / w.gk]) : 0.f;
    }
    // v: the block's 32 values (low nibbles take the first 16).
    __device__ __forceinline__ float dot(const float* v) const {
        float dq = 0.f, sx = 0.f;
        for (int i = 0; i < 16; i++) { dq += lo[i] * v[i] + hi[i] * v[16 + i]; sx += v[i] + v[16 + i]; }
        float r = __fmul_rn(d, __fsub_rn(dq, __fmul_rn(8.f, sx)));
        return AFF ? __fadd_rn(r, __fmul_rn(m, sx)) : r;
    }
};
template <> struct WC<2> : WCQ4<0> {};
template <> struct WC<3> : WCQ4<1> {};

template <int F>
__device__ __forceinline__ void xload(const float* x, unsigned int c, float* v) {
    for (unsigned int i = 0; i < CH<F>::N; i += 8) x8(x, c * CH<F>::N + i, v + i);
}

// This lane's part of row . x (one row; the MoE kernels).
template <int F>
__device__ __forceinline__ float rowdot(DW w, unsigned int row, const float* x, unsigned int n_in, unsigned int lane) {
    float a = 0.f;
    for (unsigned int c = lane; c < n_in / CH<F>::N; c += 32) {
        float v[CH<F>::N];
        xload<F>(x, c, v);
        WC<F> wc;
        wc.load(w, row, c, n_in);
        a = __fadd_rn(a, wc.dot(v));
    }
    return a;
}

// mode: 0 store, 1 add, 2 silu(a) * b, 3 gelu(a) * b, 4 add a * sigmoid(gate[0]).
__device__ __forceinline__ void dmv_out(float* y, float a, float b, unsigned int mode, const float* gate) {
    if (mode == 0) *y = a;
    else if (mode == 1) *y += a;
    else if (mode == 2) *y = a / (1.0f + expf(-a)) * b;
    else if (mode == 3) *y = dgelu(a) * b;
    else *y += a * dsig(gate[0]);
}

// One token: a warp takes NR consecutive rows, reading each x chunk once for all of them.
#define DMV_NR 4
template <int F>
__device__ __forceinline__ void dmv(DW w, DW w2, const float* x, float* y, unsigned int n_out, unsigned int n_in, unsigned int mode,
                                    const float* gate) {
    unsigned int lane = threadIdx.x & 31, r0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * DMV_NR;
    if (r0 >= n_out) return;
    bool pair = mode == 2 || mode == 3;
    float a[DMV_NR], b[DMV_NR];
    for (int r = 0; r < DMV_NR; r++) { a[r] = 0.f; b[r] = 0.f; }
    for (unsigned int c = lane; c < n_in / CH<F>::N; c += 32) {
        float v[CH<F>::N];
        xload<F>(x, c, v);
#pragma unroll
        for (int r = 0; r < DMV_NR; r++) {
            unsigned int row = min(r0 + r, n_out - 1);
            WC<F> wc;
            wc.load(w, row, c, n_in);
            a[r] = __fadd_rn(a[r], wc.dot(v));
            if (pair) {
                WC<F> wc2;
                wc2.load(w2, row, c, n_in);
                b[r] = __fadd_rn(b[r], wc2.dot(v));
            }
        }
    }
#pragma unroll
    for (int r = 0; r < DMV_NR; r++) {
        float av = warp_sum(a[r]), bv = pair ? warp_sum(b[r]) : 0.f;
        if (lane == 0 && r0 + r < n_out) dmv_out(y + r0 + r, av, bv, mode, gate);
    }
}
#define DMV(NAME, F) \
extern "C" __global__ void NAME(DW w, DW w2, const float* x, float* y, unsigned int n_out, unsigned int n_in, unsigned int mode, \
                                const float* gate) { dmv<F>(w, w2, x, y, n_out, n_in, mode, gate); }
DMV(dmv_bf, 0) DMV(dmv_q8, 1) DMV(dmv_q4, 2) DMV(dmv_q4a, 3) DMV(dmv_f32, 4)

// Several tokens (up to 8 per launch, `xs` / `ys` / `gs` floats apart): 8 warps per block, a row
// each, sharing each 1024-column tile of the tokens' inputs through shared memory; every weight
// chunk is read once for all the tokens. A tile holds a multiple of 32 chunks of every format, so a
// lane visits its chunks in the single-token kernel's order: each token's sum is that kernel's.
#define DMVK_TILE 1024
#define DMVK_NR 4
template <int F>
__device__ __forceinline__ void dmvk(DW w, DW w2, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int n_out,
                                     unsigned int n_in, unsigned int mode, const float* gate, unsigned int gs, unsigned int nt) {
    // One float of padding per 32: lane l's chunk starts in bank l (no conflicts for any format).
    __shared__ float xt[8][DMVK_TILE + DMVK_TILE / 32];
    unsigned int lane = threadIdx.x & 31;
    unsigned int r0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * DMVK_NR;
    bool pair = mode == 2 || mode == 3;
    float a[DMVK_NR][8], b[DMVK_NR][8];
#pragma unroll
    for (int r = 0; r < DMVK_NR; r++)
#pragma unroll
        for (int t = 0; t < 8; t++) { a[r][t] = 0.f; b[r][t] = 0.f; }
    const unsigned int nch = n_in / CH<F>::N;
    for (unsigned int k0 = 0; k0 < n_in; k0 += DMVK_TILE) {
        __syncthreads();
        for (unsigned int i = threadIdx.x; i < nt * DMVK_TILE; i += blockDim.x) {
            unsigned int t = i / DMVK_TILE, k = i % DMVK_TILE;
            xt[t][k + k / 32] = k0 + k < n_in ? x[(size_t)t * xs + k0 + k] : 0.f;
        }
        __syncthreads();
        if (r0 >= n_out) continue;
        for (unsigned int cc = lane; cc < DMVK_TILE / CH<F>::N; cc += 32) {
            unsigned int c = k0 / CH<F>::N + cc;
            if (c >= nch) break;
#pragma unroll
            for (int r = 0; r < DMVK_NR; r++) {
                unsigned int row = min(r0 + r, n_out - 1);
                WC<F> wc;
                wc.load(w, row, c, n_in);
#pragma unroll
                for (unsigned int t = 0; t < 8; t++) {
                    if (t >= nt) break;
                    a[r][t] = __fadd_rn(a[r][t], wc.dot(&xt[t][cc * CH<F>::N + cc * CH<F>::N / 32]));
                }
                if (pair) {
                    WC<F> wc2;
                    wc2.load(w2, row, c, n_in);
#pragma unroll
                    for (unsigned int t = 0; t < 8; t++) {
                        if (t >= nt) break;
                        b[r][t] = __fadd_rn(b[r][t], wc2.dot(&xt[t][cc * CH<F>::N + cc * CH<F>::N / 32]));
                    }
                }
            }
        }
    }
    if (r0 >= n_out) return;
#pragma unroll
    for (int r = 0; r < DMVK_NR; r++)
#pragma unroll
        for (unsigned int t = 0; t < 8; t++) {
            if (t >= nt) break;
            float av = warp_sum(a[r][t]), bv = pair ? warp_sum(b[r][t]) : 0.f;
            if (lane == 0 && r0 + r < n_out) dmv_out(y + (size_t)t * ys + r0 + r, av, bv, mode, gate + (size_t)t * gs);
        }
}
#define DMVK(NAME, F) \
extern "C" __global__ void NAME(DW w, DW w2, const float* x, unsigned int xs, float* y, unsigned int ys, unsigned int n_out, \
                                unsigned int n_in, unsigned int mode, const float* gate, unsigned int gs, unsigned int nt) { \
    dmvk<F>(w, w2, x, xs, y, ys, n_out, n_in, mode, gate, gs, nt); }
DMVK(dmvk_bf, 0) DMVK(dmvk_q8, 1) DMVK(dmvk_q4, 2) DMVK(dmvk_q4a, 3)

// t rows of n floats, `ss` floats apart, into consecutive rows (inputs kept for a replay).
extern "C" __global__ void dcopy_rows(const float* src, unsigned int ss, float* dst, unsigned int n, unsigned int t) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (i < n && r < t) dst[(size_t)r * n + i] = src[(size_t)r * ss + i];
}

// ---- MoE experts of one token: `tab[e]` is expert e's weights; `sel` holds k ids (uint bits)
// then k weights. Pair: grid (rows, k) computes act[s] = act(gate_e x) * (up_e x) (fused: the up
// rows at r2off of the same matrix, tab2 == tab); sum: y (+)= sum_s weight_s * (down_e act[s]).
template <int F>
__device__ __forceinline__ void dmvx_pair(const DW* tab, const DW* tab2, const float* x, float* act, unsigned int inter,
                                          unsigned int h, unsigned int r2off, unsigned int gelu, const float* sel) {
    unsigned int lane = threadIdx.x & 31, row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), s = blockIdx.y;
    if (row >= inter) return;
    unsigned int e = __float_as_uint(sel[s]);
    float a = warp_sum(rowdot<F>(tab[e], row, x, h, lane));
    float b = warp_sum(rowdot<F>(tab2[e], row + r2off, x, h, lane));
    if (lane != 0) return;
    // gelu 2: the raw gate then up outputs (2 inter per slot), for LoRA updates (`dxlora_act`).
    if (gelu == 2) {
        act[(size_t)s * 2 * inter + row] = a;
        act[(size_t)s * 2 * inter + inter + row] = b;
    } else {
        act[(size_t)s * inter + row] = (gelu ? dgelu(a) : a / (1.0f + expf(-a))) * b;
    }
}
template <int F>
__device__ __forceinline__ void dmvx_sum(const DW* tab, const float* act, float* y, unsigned int h, unsigned int inter, unsigned int k,
                                         unsigned int add, const float* sel) {
    unsigned int lane = threadIdx.x & 31, row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= h) return;
    float acc = 0.f;
    for (unsigned int s = 0; s < k; s++) {
        unsigned int e = __float_as_uint(sel[s]);
        acc += sel[k + s] * rowdot<F>(tab[e], row, act + (size_t)s * inter, inter, lane);
    }
    acc = warp_sum(acc);
    if (lane == 0) { if (add) y[row] += acc; else y[row] = acc; }
}
#define DMVX(NAME, F) \
extern "C" __global__ void NAME##_pair(const DW* tab, const DW* tab2, const float* x, float* act, unsigned int inter, unsigned int h, \
                                       unsigned int r2off, unsigned int gelu, const float* sel) { \
    dmvx_pair<F>(tab, tab2, x, act, inter, h, r2off, gelu, sel); } \
extern "C" __global__ void NAME##_sum(const DW* tab, const float* act, float* y, unsigned int h, unsigned int inter, unsigned int k, \
                                      unsigned int add, const float* sel) { dmvx_sum<F>(tab, act, y, h, inter, k, add, sel); }
DMVX(dmvx_bf, 0) DMVX(dmvx_q8, 1) DMVX(dmvx_q4, 2) DMVX(dmvx_q4a, 3)

// ---- Top k of n router logits (ties to the lower index), softmax over the chosen, times
// scale[id] if `has_scale`: ids (uint bits) then weights into `sel`. One block, a thread per logit.
extern "C" __global__ void dtopk(const float* logits, float* sel, unsigned int n, unsigned int k, unsigned int has_scale,
                                 const float* scale) {
    __shared__ float bv[32];
    __shared__ unsigned int bi[32];
    __shared__ float chosen[64];
    unsigned int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, nw = (blockDim.x + 31) >> 5;
    float v = tid < n ? logits[tid] : -__uint_as_float(0x7f800000u);
    for (unsigned int j = 0; j < k; j++) {
        float m = warp_max(v);
        unsigned int idx = v == m ? tid : 0xffffffffu;
        for (int o = 16; o > 0; o >>= 1) idx = min(idx, __shfl_xor_sync(0xffffffffu, idx, o));
        if (lane == 0) { bv[w] = m; bi[w] = idx; }
        __syncthreads();
        float best = bv[0];
        unsigned int bidx = bi[0];
        for (unsigned int i = 1; i < nw; i++) if (bv[i] > best || (bv[i] == best && bi[i] < bidx)) { best = bv[i]; bidx = bi[i]; }
        if (tid == bidx) v = -__uint_as_float(0x7f800000u);
        if (tid == 0) { sel[j] = __uint_as_float(bidx); chosen[j] = best; }
        __syncthreads();
    }
    if (tid == 0) {
        float m = chosen[0], s = 0.f;
        for (unsigned int j = 0; j < k; j++) { float e = expf(chosen[j] - m); chosen[j] = e; s += e; }
        for (unsigned int j = 0; j < k; j++) sel[k + j] = chosen[j] / s * (has_scale ? scale[__float_as_uint(sel[j])] : 1.f);
    }
}

// ---- RMSNorm of rows of n: out (+)= x' * rsqrt(mean(x'^2) + eps) * w, x' = x * pre. A block per row.
struct DNP { unsigned int n; float eps; unsigned int has_w; unsigned int add; unsigned int x_stride; unsigned int o_stride; float pre; unsigned int _p; };
extern "C" __global__ void dnorm(const float* x, float* out, const float* w, DNP p) {
    __shared__ float red[32];
    const float* xr = x + (size_t)blockIdx.x * p.x_stride;
    float* o = out + (size_t)blockIdx.x * p.o_stride;
    float s = 0.f;
    for (unsigned int i = threadIdx.x; i < p.n; i += blockDim.x) { float v = xr[i] * p.pre; s += v * v; }
    float k = rsqrtf(dblock_sum(s, red) / (float)p.n + p.eps);
    for (unsigned int i = threadIdx.x; i < p.n; i += blockDim.x) {
        float v = xr[i] * p.pre * k;
        if (p.has_w) v *= w[i];
        if (p.add) o[i] += v; else o[i] = v;
    }
}

// `dnorm` (a row per block) that also writes its rows as f16 for the next GEMM (`xh`, rows of
// p.n halves, the values `dhalf_t` would make of them) and sets *flag as `dhalf_t` does.
extern "C" __global__ void dnorm_h(const float* x, float* out, const float* w, DNP p, unsigned short* xh, unsigned int* flag) {
    __shared__ float red[32];
    const float* xr = x + (size_t)blockIdx.x * p.x_stride;
    float* o = out + (size_t)blockIdx.x * p.o_stride;
    unsigned short* h = xh + (size_t)blockIdx.x * p.n;
    float s = 0.f;
    for (unsigned int i = threadIdx.x; i < p.n; i += blockDim.x) { float v = xr[i] * p.pre; s += v * v; }
    float k = rsqrtf(dblock_sum(s, red) / (float)p.n + p.eps);
    unsigned int top = 0u;
    for (unsigned int i = threadIdx.x; i < p.n; i += blockDim.x) {
        float v = xr[i] * p.pre * k;
        if (p.has_w) v *= w[i];
        if (p.add) v += o[i];
        o[i] = v;
        unsigned short hv;
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(hv) : "f"(v));
        h[i] = hv;
        top = max(top, __float_as_uint(v) & 0x7fffffffu);
    }
    if (top >= 0x47000000u) *flag = 1u;
}

// ---- x = (x + rmsnorm(y) * wy) * scale, then with has_next out = rmsnorm(x) * wn. One block.
extern "C" __global__ void dadd_norm(const float* y, float* x, float* out, const float* wy, const float* wn, unsigned int n, float eps,
                                     float scale, unsigned int has_next) {
    __shared__ float red[32];
    float s = 0.f;
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) s += y[i] * y[i];
    float k = rsqrtf(dblock_sum(s, red) / (float)n + eps);
    float s2 = 0.f;
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = (x[i] + y[i] * k * wy[i]) * scale;
        x[i] = v;
        s2 += v * v;
    }
    if (!has_next) return;
    float k2 = rsqrtf(dblock_sum(s2, red) / (float)n + eps);
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) out[i] = x[i] * k2 * wn[i];
}

// ---- Attention inputs of the token: blocks nh query heads, then nkv key heads, then nkv value
// heads, a thread per dimension. Keys and values go to the caches at sp[0] and to kout / vout.
struct DQP { unsigned int nh, nkv, hd, half_rot, q_stride, k_off, v_off, interleaved, has_qn, has_kn, v_norm, v_from_k; float eps; unsigned int _p[3]; };
extern "C" __global__ void dqk_rope(const float* qkv, float* qo, float* kc, float* vc, float* kout, float* vout, const float* qn,
                                    const float* kn, const float* cs, const unsigned int* sp, DQP p) {
    __shared__ float buf[512];
    __shared__ float red[32];
    unsigned int h = blockIdx.x, i = threadIdx.x, hd = p.hd, pos = sp[0];
    unsigned int kind = h < p.nh ? 0 : (h < p.nh + p.nkv ? 1 : 2);
    unsigned int hh = kind == 0 ? h : (kind == 1 ? h - p.nh : h - p.nh - p.nkv);
    float v = kind == 0 ? qkv[hh * p.q_stride + i]
            : (kind == 1 || p.v_from_k) ? qkv[p.k_off + hh * hd + i] : qkv[p.v_off + hh * hd + i];
    bool norm = kind == 0 ? p.has_qn != 0 : (kind == 1 ? p.has_kn != 0 : p.v_norm != 0);
    if (norm) {
        float s = dblock_sum(v * v, red);
        v = v * rsqrtf(s / (float)hd + p.eps);
        if (kind == 0) v *= qn[i];
        else if (kind == 1) v *= kn[i];
    }
    size_t row = (size_t)p.nkv * hd;
    if (kind == 2) {
        vc[(size_t)pos * row + hh * hd + i] = v;
        vout[hh * hd + i] = v;
        return;
    }
    buf[i] = v;
    __syncthreads();
    unsigned int hr = p.half_rot;
    if (i < 2 * hr) {
        if (p.interleaved) {
            unsigned int j = i / 2;
            float c = cs[j], s = cs[hr + j];
            float x1 = buf[2 * j], x2 = buf[2 * j + 1];
            v = (i % 2 == 0) ? x1 * c - x2 * s : x2 * c + x1 * s;
        } else {
            unsigned int j = i % hr;
            float c = cs[j], s = cs[hr + j];
            float x1 = buf[j], x2 = buf[j + hr];
            v = (i < hr) ? x1 * c - x2 * s : x2 * c + x1 * s;
        }
    }
    if (kind == 0) qo[hh * hd + i] = v;
    else {
        kc[(size_t)pos * row + hh * hd + i] = v;
        kout[hh * hd + i] = v;
    }
}

// ---- Attention of one query per head over cache positions lo .. len - 1 (len = sp[0] + 1, lo from
// the window). A block of 8 warps per (G query heads sharing a KV head, chunk of keys); warp w
// takes blocks of 32 keys, lane t scoring key t, each lane then owning D = hd / 32 dimensions.
// Long contexts split over up to 64 chunks (partials combined by dattn_combine); the split is
// derived from len on the device, so the launch does not depend on the position.
// sp = [position, first position of the step, tokens of the step, index in the step]. block: 1 a
// DSpark draft block (every token sees the cached positions before the step, the last `window` if
// set, and the step's tokens), 2 causal within the block.
struct DAP { unsigned int nh, nkv, window, has_gate, g_stride, block; float scale; unsigned int _q; };
__device__ __forceinline__ void attn_split(const unsigned int* sp, DAP p, unsigned int& len, unsigned int& lo,
                                           unsigned int& chunk, unsigned int& nsplit) {
    unsigned int window = p.window;
    if (p.block & 1) {
        unsigned int pos0 = sp[1];
        lo = window != 0 && pos0 > window ? pos0 - window : 0;
        len = (p.block & 2) ? pos0 + sp[3] + 1 : pos0 + sp[2];
    } else {
        len = sp[0] + 1;
        lo = window != 0 && len > window ? len - window : 0;
    }
    unsigned int n = len - lo;
    chunk = max(256u, (((n + 63) / 64) + 255) / 256 * 256);
    nsplit = (n + chunk - 1) / chunk;
}
template <int D, int G>
__device__ __forceinline__ void dattn_body(const float* q, const float* kc, const float* vc, float* out, const float* gate, float* part,
                                           const unsigned int* sp, DAP p) {
    constexpr unsigned int HD = D * 32;
    __shared__ float sm[16];
    __shared__ float so[8 * HD];
    __shared__ __align__(16) float qs[G * HD];
    unsigned int len, lo, chunk, nsplit;
    attn_split(sp, p, len, lo, chunk, nsplit);
    unsigned int split = blockIdx.y;
    if (split >= nsplit) return;
    unsigned int tid = threadIdx.x, sg = tid >> 5, lane = tid & 31;
    unsigned int h0 = blockIdx.x * G, kvh = h0 / (p.nh / p.nkv);
    const float NEG = -__uint_as_float(0x7f800000u);
    for (unsigned int i = tid; i < G * HD; i += 256) qs[i] = q[h0 * HD + i] * p.scale;
    __syncthreads();
    float o[G][D], m[G], l[G];
    for (int g = 0; g < G; g++) {
        m[g] = NEG;
        l[g] = 0.f;
        for (int d = 0; d < D; d++) o[g][d] = 0.f;
    }
    size_t rs = (size_t)p.nkv * HD;
    unsigned int j0 = lo + split * chunk, j1 = min(len, j0 + chunk);
    for (unsigned int base = j0 + sg * 32; base < j1; base += 8 * 32) {
        unsigned int j = base + lane;
        float s[G];
        for (int g = 0; g < G; g++) s[g] = NEG;
        if (j < j1) {
            const float4* kr = (const float4*)(kc + (size_t)j * rs + (size_t)kvh * HD);
            float4 acc[G];
            for (int g = 0; g < G; g++) acc[g] = make_float4(0.f, 0.f, 0.f, 0.f);
            for (unsigned int d = 0; d < HD / 4; d++) {
                float4 k4 = kr[d];
                for (int g = 0; g < G; g++) {
                    float4 q4 = ((const float4*)(qs + g * HD))[d];
                    acc[g].x = fmaf(k4.x, q4.x, acc[g].x); acc[g].y = fmaf(k4.y, q4.y, acc[g].y);
                    acc[g].z = fmaf(k4.z, q4.z, acc[g].z); acc[g].w = fmaf(k4.w, q4.w, acc[g].w);
                }
            }
            for (int g = 0; g < G; g++) s[g] = (acc[g].x + acc[g].y) + (acc[g].z + acc[g].w);
        }
        float e[G];
        for (int g = 0; g < G; g++) {
            float mn = fmaxf(m[g], warp_max(s[g]));
            float a = expf(m[g] - mn);
            e[g] = j < j1 ? expf(s[g] - mn) : 0.f;
            l[g] = l[g] * a + warp_sum(e[g]);
            for (int d = 0; d < D; d++) o[g][d] *= a;
            m[g] = mn;
        }
        unsigned int nb = min(32u, j1 - base);
        const float* vb = vc + (size_t)base * rs + (size_t)kvh * HD + lane * D;
        for (unsigned int t = 0; t < nb; t++) {
            const float* vr = vb + t * rs;
            float v[D];
            for (int d = 0; d < D; d++) v[d] = vr[d];
            for (int g = 0; g < G; g++) {
                float et = __shfl_sync(0xffffffffu, e[g], t);
                for (int d = 0; d < D; d++) o[g][d] = fmaf(et, v[d], o[g][d]);
            }
        }
    }
    for (int g = 0; g < G; g++) {
        unsigned int h = h0 + g;
        if (lane == 0) { sm[sg] = m[g]; sm[8 + sg] = l[g]; }
        for (int d = 0; d < D; d++) so[sg * HD + lane * D + d] = o[g][d];
        __syncthreads();
        float M = sm[0];
        for (unsigned int s = 1; s < 8; s++) M = fmaxf(M, sm[s]);
        float L = 0.f;
        for (unsigned int s = 0; s < 8; s++) L += sm[s] == NEG ? 0.f : sm[8 + s] * expf(sm[s] - M);
        for (unsigned int i = tid; i < HD; i += 256) {
            float acc = 0.f;
            for (unsigned int s = 0; s < 8; s++) if (sm[s] != NEG) acc += so[s * HD + i] * expf(sm[s] - M);
            if (nsplit > 1) {
                float* pp = part + ((size_t)h * 64 + split) * (HD + 2);
                pp[i] = acc;
                if (i == 0) { pp[HD] = M; pp[HD + 1] = L; }
                continue;
            }
            float r = acc / L;
            if (p.has_gate) r *= dsig(gate[h * p.g_stride + i]);
            out[h * HD + i] = r;
        }
        __syncthreads();
    }
}
#define DATTN(D, G) \
extern "C" __global__ void dattn_##D##_##G(const float* q, const float* kc, const float* vc, float* out, const float* gate, float* part, \
                                           const unsigned int* sp, DAP p) { dattn_body<D, G>(q, kc, vc, out, gate, part, sp, p); }
DATTN(2, 1) DATTN(2, 2) DATTN(2, 4) DATTN(2, 8) DATTN(4, 1) DATTN(4, 2) DATTN(4, 4) DATTN(8, 1) DATTN(8, 2) DATTN(16, 1)
extern "C" __global__ void dattn_combine(const float* part, float* out, const float* gate, const unsigned int* sp, DAP p, unsigned int hd) {
    unsigned int len, lo, chunk, nsplit;
    attn_split(sp, p, len, lo, chunk, nsplit);
    if (nsplit <= 1) return;
    const float NEG = -__uint_as_float(0x7f800000u);
    unsigned int h = blockIdx.x;
    const float* ph = part + (size_t)h * 64 * (hd + 2);
    float M = NEG;
    for (unsigned int s = 0; s < nsplit; s++) M = fmaxf(M, ph[s * (hd + 2) + hd]);
    float L = 0.f;
    for (unsigned int s = 0; s < nsplit; s++) {
        float ms = ph[s * (hd + 2) + hd];
        if (ms != NEG) L += ph[s * (hd + 2) + hd + 1] * expf(ms - M);
    }
    for (unsigned int i = threadIdx.x; i < hd; i += blockDim.x) {
        float acc = 0.f;
        for (unsigned int s = 0; s < nsplit; s++) {
            float ms = ph[s * (hd + 2) + hd];
            if (ms != NEG) acc += ph[s * (hd + 2) + i] * expf(ms - M);
        }
        float r = acc / L;
        if (p.has_gate) r *= dsig(gate[h * p.g_stride + i]);
        out[h * hd + i] = r;
    }
}

// ---- Gated DeltaNet: causal depthwise conv (kernel kc) with SiLU, a thread per channel; the state
// [kc - 1][dim] holds the earlier inputs (oldest first) and shifts in this one.
extern "C" __global__ void dgdn_conv(const float* x, float* out, float* st, const float* w, unsigned int dim, unsigned int kc) {
    unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= dim) return;
    float xv = x[ch];
    float s = 0.f;
    for (unsigned int j = 0; j + 1 < kc; j++) s += w[ch * kc + j] * st[j * dim + ch];
    s += w[ch * kc + kc - 1] * xv;
    out[ch] = s / (1.0f + expf(-s));
    for (unsigned int j = 0; j + 2 < kc; j++) st[j * dim + ch] = st[(j + 1) * dim + ch];
    st[(kc - 2) * dim + ch] = xv;
}
// The gated delta rule for one token, a block per value head; thread (part, j) keeps rows 32 part ..
// 32 part + 31 of column j of the head's [hdk, hdv] state in registers. Then the gated RMSNorm.
// cst: decay_coef[nv], dt_bias[nv], norm_w[hdv]. flags: 1 tiled heads, 2 sigmoid output gate.
struct DGR { unsigned int nk, nv, key_dim, flags; float eps, qscale; unsigned int hdk, hdv; };
__device__ __forceinline__ float dtg_sum(float v, unsigned int n, float* red) {
    // sum of v over threads 0 .. n - 1 (n a multiple of 32); every thread calls it
    unsigned int tid = threadIdx.x;
    if (tid < n) { float s = warp_sum(v); if ((tid & 31) == 0) red[tid >> 5] = s; }
    __syncthreads();
    float t = 0.f;
    for (unsigned int i = 0; i < n / 32; i++) t += red[i];
    __syncthreads();
    return t;
}
extern "C" __global__ void dgdn_rule(const float* c, const float* z, const float* b, const float* a, float* out, float* state,
                                     const float* cst, DGR p) {
    __shared__ float qs[256], ks[256], red[1024], red2[32];
    unsigned int hv = blockIdx.x, tid = threadIdx.x;
    unsigned int hdk = p.hdk, hdv = p.hdv, np = hdk / 32;
    unsigned int hk = (p.flags & 1u) ? hv % p.nk : hv / (p.nv / p.nk);
    unsigned int part = tid / hdv, j = tid % hdv;
    float q = 0.f, k = 0.f;
    if (tid < hdk) {
        q = c[hk * hdk + tid];
        k = c[p.key_dim + hk * hdk + tid];
        qs[tid] = q;
        ks[tid] = k;
    }
    float qinv = 1.0f / sqrtf(dtg_sum(q * q, hdk, red2) + 1e-6f) * p.qscale;
    float kinv = 1.0f / sqrtf(dtg_sum(k * k, hdk, red2) + 1e-6f);
    float beta = 1.0f / (1.0f + expf(-b[hv]));
    float ga = a[hv] + cst[p.nv + hv];
    float spl = ga > 20.f ? ga : logf(1.0f + expf(ga));
    float decay = expf(cst[hv] * spl);
    float* S = state + (size_t)hv * hdk * hdv + part * 32 * hdv + j;
    float s[32];
    float kv = 0.f;
    for (unsigned int d = 0; d < 32; d++) {
        s[d] = S[d * hdv] * decay;
        kv += s[d] * (ks[part * 32 + d] * kinv);
    }
    red[part * hdv + j] = kv;
    __syncthreads();
    float kvm = 0.f;
    for (unsigned int i = 0; i < np; i++) kvm += red[i * hdv + j];
    float delta = (c[2 * p.key_dim + hv * hdv + j] - kvm) * beta;
    float o = 0.f;
    for (unsigned int d = 0; d < 32; d++) {
        s[d] += (ks[part * 32 + d] * kinv) * delta;
        o += s[d] * (qs[part * 32 + d] * qinv);
        S[d * hdv] = s[d];
    }
    __syncthreads();
    red[part * hdv + j] = o;
    __syncthreads();
    float ov = 0.f;
    if (tid < hdv)
        for (unsigned int i = 0; i < np; i++) ov += red[i * hdv + j];
    float r = rsqrtf(dtg_sum(ov * ov, hdv, red2) / (float)hdv + p.eps);
    if (tid < hdv) {
        float zz = z[hv * hdv + j];
        out[hv * hdv + j] = ov * r * cst[2 * p.nv + j] * ((p.flags & 2u) ? dsig(zz) : zz / (1.0f + expf(-zz)));
    }
}

// ---- out = act(g) * u (SiLU, or tanh GELU).
extern "C" __global__ void dglu(const float* g, const float* u, float* out, unsigned int n, unsigned int gelu) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = g[i];
    out[i] = (gelu ? dgelu(a) : a / (1.0f + expf(-a))) * u[i];
}

// ---- Elementwise: kind 0 a * p0, 1 tanh(a / p0) * p0, 2 (a * p0 + b * p1) * p2, 3 silu(a * p0).
extern "C" __global__ void dew(const float* a, const float* b, float* out, unsigned int n, unsigned int kind, float p0, float p1, float p2) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = a[i];
    if (kind == 0) out[i] = v * p0;
    else if (kind == 1) out[i] = tanhf(v / p0) * p0;
    else if (kind == 2) out[i] = (v * p0 + b[i] * p1) * p2;
    else { float u = v * p0; out[i] = u / (1.0f + expf(-u)); }
}

// ---- Runtime LoRA updates (crate::lora, `Op::Lora`; one token per launch, as the Metal kernels).
// Partial sums of u = A x over chunks of 1024 inputs: block c * r + k writes u[c * r + k].
extern "C" __global__ void dlora_down(const float* a, const float* x, float* u, unsigned int n_in, unsigned int r) {
    __shared__ float red[32];
    unsigned int k = blockIdx.x % r, c = blockIdx.x / r;
    unsigned int i0 = c * 1024, i1 = min(i0 + 1024, n_in);
    const float* ar = a + (size_t)k * n_in;
    float s = 0.f;
    for (unsigned int i = i0 + threadIdx.x; i < i1; i += blockDim.x) s += ar[i] * x[i];
    s = dblock_sum(s, red);
    if (threadIdx.x == 0) u[blockIdx.x] = s;
}
// y[o] += g sum_k u_k S[sc[k]] bt[k, o], u_k the sum of component k's chunks in order (g: the
// sigmoid of *gate, or 1).
extern "C" __global__ void dlora_up(const float* bt, const unsigned int* sc, const float* str, const float* u, float* y,
                                    const float* gate, unsigned int r, unsigned int nch, unsigned int n_out, unsigned int has_gate) {
    __shared__ float us[256];
    for (unsigned int k = threadIdx.x; k < r && k < 256; k += blockDim.x) {
        float v = 0.f;
        for (unsigned int c = 0; c < nch; c++) v += u[c * r + k];
        us[k] = v * str[sc[k]];
    }
    __syncthreads();
    unsigned int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n_out || r > 256) return;
    float acc = 0.f;
    for (unsigned int k = 0; k < r; k++) acc += us[k] * bt[(size_t)k * n_out + o];
    if (has_gate) acc *= dsig(gate[0]);
    y[o] += acc;
}
// Routed experts (`Experts::lora`): stacks [n_exp, r, ...]; `sel` holds the k ids then weights.
// u[s, c] = A_e[c] . x_s (x_s = x + s * xs): block (c, s).
extern "C" __global__ void dxlora_down(const float* a, const float* x, float* u, const float* sel, unsigned int n_in, unsigned int r,
                                       unsigned int xs) {
    __shared__ float red[32];
    unsigned int c = blockIdx.x, s = blockIdx.y, e = __float_as_uint(sel[s]);
    const float* ar = a + ((size_t)e * r + c) * n_in;
    const float* xr = x + (size_t)s * xs;
    float v = 0.f;
    for (unsigned int i = threadIdx.x; i < n_in; i += blockDim.x) v += ar[i] * xr[i];
    v = dblock_sum(v, red);
    if (threadIdx.x == 0) u[s * r + c] = v;
}
// act[s, o] = act(g + dg) * (v + dv) from the raw pair outputs (`raw`: [k, 2 n]).
extern "C" __global__ void dxlora_act(const float* raw, const float* u, const float* bt, const unsigned int* sc, const float* str,
                                      const float* sel, float* act, unsigned int n, unsigned int r, unsigned int gelu) {
    unsigned int o = blockIdx.x * blockDim.x + threadIdx.x, s = blockIdx.y, w = 2 * n;
    if (o >= n) return;
    unsigned int e = __float_as_uint(sel[s]);
    const float* b = bt + (size_t)e * r * w;
    const unsigned int* sce = sc + (size_t)e * r;
    float g = raw[(size_t)s * w + o], v = raw[(size_t)s * w + n + o];
    for (unsigned int c = 0; c < r; c++) {
        float f = u[s * r + c] * str[sce[c]];
        g += f * b[(size_t)c * w + o];
        v += f * b[(size_t)c * w + n + o];
    }
    act[(size_t)s * n + o] = (gelu ? dgelu(g) : g / (1.0f + expf(-g))) * v;
}
// y[o] += sum_s weight_s sum_c u[s, c] S[sc] bt_e[c, o].
extern "C" __global__ void dxlora_up(const float* u, const float* bt, const unsigned int* sc, const float* str, const float* sel,
                                     float* y, unsigned int n_out, unsigned int r, unsigned int k) {
    unsigned int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n_out) return;
    float acc = 0.f;
    for (unsigned int s = 0; s < k; s++) {
        unsigned int e = __float_as_uint(sel[s]);
        const float* b = bt + (size_t)e * r * n_out;
        const unsigned int* sce = sc + (size_t)e * r;
        float a = 0.f;
        for (unsigned int c = 0; c < r; c++) a += u[s * r + c] * str[sce[c]] * b[(size_t)c * n_out + o];
        acc += sel[k + s] * a;
    }
    y[o] += acc;
}

// ---- Prefill plans (`decode_plan_prefill`): the ops above over every token of a step in one
// launch. Token `j` of the step reads and writes rows `j` strides further (in floats), the
// arithmetic per token being the single-token kernel's.

// out = act(g) * u; grid (blocks of n, tokens).
extern "C" __global__ void dglu_t(const float* g, const float* u, float* out, unsigned int n, unsigned int gelu, unsigned int gs,
                                  unsigned int us, unsigned int os) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t t = blockIdx.y;
    if (i >= n) return;
    float a = g[t * gs + i];
    out[t * os + i] = (gelu ? dgelu(a) : a / (1.0f + expf(-a))) * u[t * us + i];
}

// `dglu_t` that also writes its rows as f16 for the next GEMM (`xh`, rows of n halves) and sets
// *flag, as `dnorm_h`.
extern "C" __global__ void dglu_th(const float* g, const float* u, float* out, unsigned int n, unsigned int gelu, unsigned int gs,
                                   unsigned int us, unsigned int os, unsigned short* xh, unsigned int* flag) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t t = blockIdx.y;
    if (i >= n) return;
    float a = g[t * gs + i];
    float v = (gelu ? dgelu(a) : a / (1.0f + expf(-a))) * u[t * us + i];
    out[t * os + i] = v;
    unsigned short hv;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(hv) : "f"(v));
    xh[t * n + i] = hv;
    if ((__float_as_uint(v) & 0x7fffffffu) >= 0x47000000u) *flag = 1u;
}

// `dew` per token; grid (blocks of n, tokens).
extern "C" __global__ void dew_t(const float* a, const float* b, float* out, unsigned int n, unsigned int kind, float p0, float p1,
                                 float p2, unsigned int as, unsigned int bs, unsigned int os) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t t = blockIdx.y;
    if (i >= n) return;
    float v = a[t * as + i];
    float* o = out + t * os + i;
    if (kind == 0) *o = v * p0;
    else if (kind == 1) *o = tanhf(v / p0) * p0;
    else if (kind == 2) *o = (v * p0 + b[t * bs + i] * p1) * p2;
    else { float u = v * p0; *o = u / (1.0f + expf(-u)); }
}

// `dqk_rope` per token: grid (nh + 2 nkv, tokens). Token j is at position sp[4 j]; its RoPE row
// is 2 half_rot floats after the previous one's, its kout / vout rows nkv hd.
__device__ __forceinline__ void dqk_rope_t_body(const float* qkv, float* qo, float* kc, float* vc, float* kout, float* vout,
                                                const float* qn, const float* kn, const float* cs, const unsigned int* sp, DQP p,
                                                unsigned int qs, unsigned int os) {
    __shared__ float buf[512];
    __shared__ float red[32];
    size_t tok = blockIdx.y;
    qkv += tok * qs;
    qo += tok * os;
    cs += tok * 2 * p.half_rot;
    unsigned int h = blockIdx.x, i = threadIdx.x, hd = p.hd, pos = sp[4 * tok];
    size_t row = (size_t)p.nkv * hd;
    kout += tok * row;
    vout += tok * row;
    unsigned int kind = h < p.nh ? 0 : (h < p.nh + p.nkv ? 1 : 2);
    unsigned int hh = kind == 0 ? h : (kind == 1 ? h - p.nh : h - p.nh - p.nkv);
    float v = kind == 0 ? qkv[hh * p.q_stride + i]
            : (kind == 1 || p.v_from_k) ? qkv[p.k_off + hh * hd + i] : qkv[p.v_off + hh * hd + i];
    bool norm = kind == 0 ? p.has_qn != 0 : (kind == 1 ? p.has_kn != 0 : p.v_norm != 0);
    if (norm) {
        float s = dblock_sum(v * v, red);
        v = v * rsqrtf(s / (float)hd + p.eps);
        if (kind == 0) v *= qn[i];
        else if (kind == 1) v *= kn[i];
    }
    if (kind == 2) {
        vc[(size_t)pos * row + hh * hd + i] = v;
        vout[hh * hd + i] = v;
        return;
    }
    buf[i] = v;
    __syncthreads();
    unsigned int hr = p.half_rot;
    if (i < 2 * hr) {
        if (p.interleaved) {
            unsigned int j = i / 2;
            float c = cs[j], s = cs[hr + j];
            float x1 = buf[2 * j], x2 = buf[2 * j + 1];
            v = (i % 2 == 0) ? x1 * c - x2 * s : x2 * c + x1 * s;
        } else {
            unsigned int j = i % hr;
            float c = cs[j], s = cs[hr + j];
            float x1 = buf[j], x2 = buf[j + hr];
            v = (i < hr) ? x1 * c - x2 * s : x2 * c + x1 * s;
        }
    }
    if (kind == 0) qo[hh * hd + i] = v;
    else {
        kc[(size_t)pos * row + hh * hd + i] = v;
        kout[hh * hd + i] = v;
    }
}
extern "C" __global__ void dqk_rope_t(const float* qkv, float* qo, float* kc, float* vc, float* kout, float* vout, const float* qn,
                                      const float* kn, const float* cs, const unsigned int* sp, DQP p, unsigned int qs, unsigned int os) {
    dqk_rope_t_body(qkv, qo, kc, vc, kout, vout, qn, kn, cs, sp, p, qs, os);
}
// Batched rows (`step_rows`): token j is a sequence of its own, with its own cache (`kvt`: per
// token, the device addresses of its keys and of its values).
extern "C" __global__ void dqk_rope_r(const float* qkv, float* qo, const unsigned long long* kvt, float* kout, float* vout, const float* qn,
                                      const float* kn, const float* cs, const unsigned int* sp, DQP p, unsigned int qs, unsigned int os) {
    dqk_rope_t_body(qkv, qo, (float*)kvt[2 * blockIdx.y], (float*)kvt[2 * blockIdx.y + 1], kout, vout, qn, kn, cs, sp, p, qs, os);
}

// `dattn_*` for a batch of tokens: grid (head groups, 64 chunks, tokens of the batch); q, out,
// gate and sp point at the batch's first token, `part` holds nh * 64 * (hd + 2) floats per token
// of the batch.
#define DATTNT(D, G) \
extern "C" __global__ void dattnt_##D##_##G(const float* q, const float* kc, const float* vc, float* out, const float* gate, float* part, \
                                            const unsigned int* sp, DAP p, unsigned int qs, unsigned int os, unsigned int gs) { \
    size_t t = blockIdx.z; \
    dattn_body<D, G>(q + t * qs, kc, vc, out + t * os, gate + t * gs, part + t * p.nh * 64 * (D * 32 + 2), sp + 4 * t, p); }
DATTNT(2, 1) DATTNT(2, 2) DATTNT(2, 4) DATTNT(2, 8) DATTNT(4, 1) DATTNT(4, 2) DATTNT(4, 4) DATTNT(8, 1) DATTNT(8, 2) DATTNT(16, 1)
// The same over batched rows: token j attends over its own cache (`kvt` as dqk_rope_r's).
#define DATTNR(D, G) \
extern "C" __global__ void dattnr_##D##_##G(const float* q, const unsigned long long* kvt, float* out, const float* gate, float* part, \
                                            const unsigned int* sp, DAP p, unsigned int qs, unsigned int os, unsigned int gs) { \
    size_t t = blockIdx.z; \
    dattn_body<D, G>(q + t * qs, (const float*)kvt[2 * t], (const float*)kvt[2 * t + 1], out + t * os, gate + t * gs, \
                     part + t * p.nh * 64 * (D * 32 + 2), sp + 4 * t, p); }
DATTNR(2, 1) DATTNR(2, 2) DATTNR(2, 4) DATTNR(2, 8) DATTNR(4, 1) DATTNR(4, 2) DATTNR(4, 4) DATTNR(8, 1) DATTNR(8, 2) DATTNR(16, 1)
__device__ __forceinline__ void dattn_combine_body(const float* part, float* out, const float* gate, const unsigned int* sp, DAP p,
                                                   unsigned int hd) {
    unsigned int len, lo, chunk, nsplit;
    attn_split(sp, p, len, lo, chunk, nsplit);
    if (nsplit <= 1) return;
    const float NEG = -__uint_as_float(0x7f800000u);
    unsigned int h = blockIdx.x;
    const float* ph = part + (size_t)h * 64 * (hd + 2);
    float M = NEG;
    for (unsigned int s = 0; s < nsplit; s++) M = fmaxf(M, ph[s * (hd + 2) + hd]);
    float L = 0.f;
    for (unsigned int s = 0; s < nsplit; s++) {
        float ms = ph[s * (hd + 2) + hd];
        if (ms != NEG) L += ph[s * (hd + 2) + hd + 1] * expf(ms - M);
    }
    for (unsigned int i = threadIdx.x; i < hd; i += blockDim.x) {
        float acc = 0.f;
        for (unsigned int s = 0; s < nsplit; s++) {
            float ms = ph[s * (hd + 2) + hd];
            if (ms != NEG) acc += ph[s * (hd + 2) + i] * expf(ms - M);
        }
        float r = acc / L;
        if (p.has_gate) r *= dsig(gate[h * p.g_stride + i]);
        out[h * hd + i] = r;
    }
}
extern "C" __global__ void dattn_combine_t(const float* part, float* out, const float* gate, const unsigned int* sp, DAP p, unsigned int hd,
                                           unsigned int os, unsigned int gs) {
    size_t t = blockIdx.y;
    dattn_combine_body(part + t * p.nh * 64 * (hd + 2), out + t * os, gate + t * gs, sp + 4 * t, p, hd);
}

// `dgdn_conv` over nt tokens in order, a thread per channel keeping the channel's history
// (kc <= 9) in registers.
extern "C" __global__ void dgdn_conv_t(const float* x, float* out, float* st, const float* w, unsigned int dim, unsigned int kc,
                                       unsigned int nt, unsigned int xs, unsigned int os) {
    unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= dim) return;
    float h[8], wk[9];
    for (unsigned int j = 0; j + 1 < kc; j++) h[j] = st[j * dim + ch];
    for (unsigned int j = 0; j < kc; j++) wk[j] = w[ch * kc + j];
    for (size_t t = 0; t < nt; t++) {
        float xv = x[t * xs + ch];
        float s = 0.f;
        for (unsigned int j = 0; j + 1 < kc; j++) s += wk[j] * h[j];
        s += wk[kc - 1] * xv;
        out[t * os + ch] = s / (1.0f + expf(-s));
        for (unsigned int j = 0; j + 2 < kc; j++) h[j] = h[j + 1];
        h[kc - 2] = xv;
    }
    for (unsigned int j = 0; j + 1 < kc; j++) st[j * dim + ch] = h[j];
}

// `dgdn_conv_t` with a block row per token (grid (dim / 256, nt)): a token's output takes the kc - 1
// inputs before it (from the state for the step's first tokens) and its own, in dgdn_conv's
// order, so the tokens need not run in turn. `out` must not share memory with `x`; the state is
// left as it was (`dgdn_conv_st` then moves it on).
extern "C" __global__ void dgdn_conv_p(const float* x, float* out, const float* st, const float* w, unsigned int dim, unsigned int kc,
                                       unsigned int xs, unsigned int os) {
    unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= dim) return;
    size_t t = blockIdx.y;
    float s = 0.f;
    for (unsigned int j = 0; j + 1 < kc; j++) {
        float h = t + j >= kc - 1 ? x[(t + j - (kc - 1)) * xs + ch] : st[(t + j) * dim + ch];
        s += w[ch * kc + j] * h;
    }
    s += w[ch * kc + kc - 1] * x[t * xs + ch];
    out[t * os + ch] = s / (1.0f + expf(-s));
}
// The conv state after nt tokens: the last kc - 1 inputs (the state's own rows where nt < kc - 1).
extern "C" __global__ void dgdn_conv_st(const float* x, float* st, unsigned int dim, unsigned int kc, unsigned int nt, unsigned int xs) {
    unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= dim) return;
    for (unsigned int j = 0; j + 1 < kc; j++)
        st[j * dim + ch] = nt + j >= kc - 1 ? x[(size_t)(nt + j - (kc - 1)) * xs + ch] : st[(nt + j) * dim + ch];
}

// `dgdn_rule` over nt tokens in order: the state stays in the threads' registers between tokens
// (read once before the first, written once after the last).
__device__ __forceinline__ void dgdn_rule_t_body(const float* c, const float* z, const float* b, const float* a, float* out,
                                                 float* state, const float* cst, DGR p, unsigned int nt, unsigned int cs, unsigned int zs,
                                                 unsigned int bs, unsigned int as, unsigned int os) {
    __shared__ float qs[256], ks[256], red[1024], red2[32];
    unsigned int hv = blockIdx.x, tid = threadIdx.x;
    unsigned int hdk = p.hdk, hdv = p.hdv, np = hdk / 32;
    unsigned int hk = (p.flags & 1u) ? hv % p.nk : hv / (p.nv / p.nk);
    unsigned int part = tid / hdv, j = tid % hdv;
    float* S = state + (size_t)hv * hdk * hdv + part * 32 * hdv + j;
    float s[32];
    for (unsigned int d = 0; d < 32; d++) s[d] = S[d * hdv];
    for (size_t t = 0; t < nt; t++) {
        const float* ct = c + t * cs;
        float q = 0.f, k = 0.f;
        if (tid < hdk) {
            q = ct[hk * hdk + tid];
            k = ct[p.key_dim + hk * hdk + tid];
            qs[tid] = q;
            ks[tid] = k;
        }
        float qinv = 1.0f / sqrtf(dtg_sum(q * q, hdk, red2) + 1e-6f) * p.qscale;
        float kinv = 1.0f / sqrtf(dtg_sum(k * k, hdk, red2) + 1e-6f);
        float beta = 1.0f / (1.0f + expf(-b[t * bs + hv]));
        float ga = a[t * as + hv] + cst[p.nv + hv];
        float spl = ga > 20.f ? ga : logf(1.0f + expf(ga));
        float decay = expf(cst[hv] * spl);
        float kv = 0.f;
        for (unsigned int d = 0; d < 32; d++) {
            s[d] = s[d] * decay;
            kv += s[d] * (ks[part * 32 + d] * kinv);
        }
        red[part * hdv + j] = kv;
        __syncthreads();
        float kvm = 0.f;
        for (unsigned int i = 0; i < np; i++) kvm += red[i * hdv + j];
        float delta = (ct[2 * p.key_dim + hv * hdv + j] - kvm) * beta;
        float o = 0.f;
        for (unsigned int d = 0; d < 32; d++) {
            s[d] += (ks[part * 32 + d] * kinv) * delta;
            o += s[d] * (qs[part * 32 + d] * qinv);
        }
        __syncthreads();
        red[part * hdv + j] = o;
        __syncthreads();
        float ov = 0.f;
        if (tid < hdv)
            for (unsigned int i = 0; i < np; i++) ov += red[i * hdv + j];
        float r = rsqrtf(dtg_sum(ov * ov, hdv, red2) / (float)hdv + p.eps);
        if (tid < hdv) {
            float zz = z[t * zs + hv * hdv + j];
            out[t * os + hv * hdv + j] = ov * r * cst[2 * p.nv + j] * ((p.flags & 2u) ? dsig(zz) : zz / (1.0f + expf(-zz)));
        }
    }
    for (unsigned int d = 0; d < 32; d++) S[d * hdv] = s[d];
}
// Blocks of up to 512 threads (hdk / 32 * hdv; the registers a thread may take depend on it), or
// up to 1024.
extern "C" __global__ void __launch_bounds__(512) dgdn_rule_t(const float* c, const float* z, const float* b, const float* a, float* out,
                                                              float* state, const float* cst, DGR p, unsigned int nt, unsigned int cs,
                                                              unsigned int zs, unsigned int bs, unsigned int as, unsigned int os) {
    dgdn_rule_t_body(c, z, b, a, out, state, cst, p, nt, cs, zs, bs, as, os);
}
extern "C" __global__ void __launch_bounds__(1024) dgdn_rule_t1k(const float* c, const float* z, const float* b, const float* a, float* out,
                                                                 float* state, const float* cst, DGR p, unsigned int nt, unsigned int cs,
                                                                 unsigned int zs, unsigned int bs, unsigned int as, unsigned int os) {
    dgdn_rule_t_body(c, z, b, a, out, state, cst, p, nt, cs, zs, bs, as, os);
}

// ---- `dgdn_rule_t` in three launches, the same states bit for bit and the same outputs where
// the value head dimension is a power of two (`dgdn_rule_split_matches`; the kernels are built
// with fast math, and the output norm's mean over another dimension rounds apart in the last bit):
// what a token's update does not take from the token before it runs for every token at once,
// and the recurrence alone stays sequential, without a barrier per token.
//  - dgdn_qk_t: grid (nk, nt), hdk threads: the normed queries and keys of every token into
//    scratch rows (`ss` floats apart: nk * hdk queries, then as many keys).
//  - dgdn_bd_t: grid (nt), a thread per value head: beta and decay after them (nv each).
//  - dgdn_seq_HDK: grid (nv), hdv threads, thread j keeping column j of the head's state (HDK
//    values) in registers across the tokens; the sums over a column go in dgdn_rule's order (32
//    at a time, then the parts in turn). It leaves the raw outputs in `out`.
//  - dgdn_out_t: grid (nv, nt), hdv threads: the output norm and gate over `out` in place.
extern "C" __global__ void dgdn_qk_t(const float* c, float* sc, DGR p, unsigned int cs, unsigned int ss) {
    __shared__ float red2[32];
    unsigned int hk = blockIdx.x, tid = threadIdx.x, hdk = p.hdk;
    const float* ct = c + (size_t)blockIdx.y * cs;
    float* st = sc + (size_t)blockIdx.y * ss;
    float q = ct[hk * hdk + tid], k = ct[p.key_dim + hk * hdk + tid];
    float qinv = 1.0f / sqrtf(dtg_sum(q * q, hdk, red2) + 1e-6f) * p.qscale;
    float kinv = 1.0f / sqrtf(dtg_sum(k * k, hdk, red2) + 1e-6f);
    st[hk * hdk + tid] = q * qinv;
    st[p.key_dim + hk * hdk + tid] = k * kinv;
}
extern "C" __global__ void dgdn_bd_t(const float* b, const float* a, float* sc, const float* cst, DGR p, unsigned int bs, unsigned int as,
                                     unsigned int ss) {
    unsigned int hv = threadIdx.x;
    size_t t = blockIdx.x;
    if (hv >= p.nv) return;
    float beta = 1.0f / (1.0f + expf(-b[t * bs + hv]));
    float ga = a[t * as + hv] + cst[p.nv + hv];
    float spl = ga > 20.f ? ga : logf(1.0f + expf(ga));
    float decay = expf(cst[hv] * spl);
    float* st = sc + t * ss + 2 * p.key_dim;
    st[hv] = beta;
    st[p.nv + hv] = decay;
}
template <int HDK>
__device__ __forceinline__ void dgdn_seq_body(const float* c, const float* sc, float* out, float* state, DGR p, unsigned int nt,
                                              unsigned int cs, unsigned int ss, unsigned int os) {
    unsigned int hv = blockIdx.x, j = threadIdx.x, hdv = p.hdv;
    unsigned int hk = (p.flags & 1u) ? hv % p.nk : hv / (p.nv / p.nk);
    float* S = state + (size_t)hv * HDK * hdv + j;
    float s[HDK];
#pragma unroll
    for (int d = 0; d < HDK; d++) s[d] = S[(size_t)d * hdv];
    for (size_t t = 0; t < nt; t++) {
        const float* st = sc + t * ss;
        const float* qn = st + hk * HDK;
        const float* kn = st + p.key_dim + hk * HDK;
        float beta = st[2 * p.key_dim + hv], decay = st[2 * p.key_dim + p.nv + hv];
        float kvm = 0.f;
#pragma unroll
        for (int part = 0; part < HDK / 32; part++) {
            float kv = 0.f;
#pragma unroll
            for (int d = 0; d < 32; d++) {
                s[part * 32 + d] = s[part * 32 + d] * decay;
                kv = fmaf(s[part * 32 + d], kn[part * 32 + d], kv);
            }
            kvm += kv;
        }
        float delta = (c[t * cs + 2 * p.key_dim + hv * hdv + j] - kvm) * beta;
        float ov = 0.f;
#pragma unroll
        for (int part = 0; part < HDK / 32; part++) {
            float o = 0.f;
#pragma unroll
            for (int d = 0; d < 32; d++) {
                s[part * 32 + d] = fmaf(kn[part * 32 + d], delta, s[part * 32 + d]);
                o = fmaf(s[part * 32 + d], qn[part * 32 + d], o);
            }
            ov += o;
        }
        out[t * os + hv * hdv + j] = ov;
    }
#pragma unroll
    for (int d = 0; d < HDK; d++) S[(size_t)d * hdv] = s[d];
}
#define DGDN_SEQ(HDK) \
extern "C" __global__ void __launch_bounds__(256) dgdn_seq_##HDK(const float* c, const float* sc, float* out, float* state, DGR p, \
                                                                 unsigned int nt, unsigned int cs, unsigned int ss, unsigned int os) { \
    dgdn_seq_body<HDK>(c, sc, out, state, p, nt, cs, ss, os); }
DGDN_SEQ(32) DGDN_SEQ(64) DGDN_SEQ(128)
// `dgdn_seqw_HDK` (fast prefill plans): dgdn_seq with a warp per column of the state instead of
// a thread, lane l keeping the HDK / 32 values of key dimensions l, l + 32, ...; a column's sums
// are each lane's own values, then the lanes by shuffles, so a token takes ~50 instructions a
// thread and no barrier, and every SM has warps of it (grid (hdv / 4, nv), 128 threads). The
// sums go in another order than dgdn_rule's: its outputs and states differ in the last bits.
template <int HDK>
__device__ __forceinline__ void dgdn_seqw_body(const float* c, const float* sc, float* out, float* state, DGR p, unsigned int nt,
                                               unsigned int cs, unsigned int ss, unsigned int os) {
    constexpr int NP = HDK / 32;
    unsigned int hv = blockIdx.y, lane = threadIdx.x & 31, hdv = p.hdv;
    unsigned int j = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (j >= hdv) return;
    unsigned int hk = (p.flags & 1u) ? hv % p.nk : hv / (p.nv / p.nk);
    float* S = state + (size_t)hv * HDK * hdv + (size_t)lane * hdv + j;
    float s[NP];
#pragma unroll
    for (int i = 0; i < NP; i++) s[i] = S[(size_t)i * 32 * hdv];
    const float* qn = sc + hk * HDK + lane;
    const float* kn = qn + p.key_dim;
    const float* bd = sc + 2 * p.key_dim + hv;
    const float* v = c + 2 * p.key_dim + hv * hdv + j;
    float* o_ = out + hv * hdv + j;
    for (size_t t = 0; t < nt; t++) {
        float beta = bd[t * ss], decay = bd[t * ss + p.nv];
        float k[NP], kv = 0.f;
#pragma unroll
        for (int i = 0; i < NP; i++) {
            k[i] = kn[t * ss + i * 32];
            s[i] = s[i] * decay;
            kv = fmaf(s[i], k[i], kv);
        }
        float delta = (v[t * cs] - warp_sum(kv)) * beta;
        float o = 0.f;
#pragma unroll
        for (int i = 0; i < NP; i++) {
            s[i] = fmaf(k[i], delta, s[i]);
            o = fmaf(s[i], qn[t * ss + i * 32], o);
        }
        o = warp_sum(o);
        if (lane == 0) o_[t * os] = o;
    }
#pragma unroll
    for (int i = 0; i < NP; i++) S[(size_t)i * 32 * hdv] = s[i];
}
#define DGDN_SEQW(HDK) \
extern "C" __global__ void __launch_bounds__(128) dgdn_seqw_##HDK(const float* c, const float* sc, float* out, float* state, DGR p, \
                                                                  unsigned int nt, unsigned int cs, unsigned int ss, unsigned int os) { \
    dgdn_seqw_body<HDK>(c, sc, out, state, p, nt, cs, ss, os); }
DGDN_SEQW(32) DGDN_SEQW(64) DGDN_SEQW(128)
extern "C" __global__ void dgdn_out_t(const float* z, float* out, const float* cst, DGR p, unsigned int zs, unsigned int os) {
    __shared__ float red2[32];
    unsigned int hv = blockIdx.x, j = threadIdx.x, hdv = p.hdv;
    size_t t = blockIdx.y;
    float ov = out[t * os + hv * hdv + j];
    float r = rsqrtf(dtg_sum(ov * ov, hdv, red2) / (float)hdv + p.eps);
    float zz = z[t * zs + hv * hdv + j];
    out[t * os + hv * hdv + j] = ov * r * cst[2 * p.nv + j] * ((p.flags & 2u) ? dsig(zz) : zz / (1.0f + expf(-zz)));
}

// The fast prefill GEMM's input (`gemmh_K`): the step's rows (n floats each, xs apart; grid
// (blocks, tokens)) as f16 rows of n, with dguard_t's check (f16 holds up to 65504; the same
// bound as the split GEMM keeps the declined steps alike).
extern "C" __global__ void dhalf_t(const float* x, unsigned int n, unsigned int xs, unsigned short* xh, unsigned int* flag) {
    const float* r = x + (size_t)blockIdx.y * xs;
    unsigned int* o = (unsigned int*)(xh + (size_t)blockIdx.y * n);
    unsigned int top = 0u;
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; 2 * i < n; i += gridDim.x * blockDim.x) {
        float a = r[2 * i], b = r[2 * i + 1];
        top = max(top, max(__float_as_uint(a) & 0x7fffffffu, __float_as_uint(b) & 0x7fffffffu));
        unsigned int h;
        asm("cvt.rn.f16x2.f32 %0, %2, %1;" : "=r"(h) : "f"(a), "f"(b));
        o[i] = h;
    }
    if (top >= 0x47000000u) *flag = 1u;
}

// ---- The GEMMs of a prefill plan take their inputs as pairs of f16 halves: sets *flag when a
// value of the step's rows (n floats each, xs apart; grid (blocks, tokens)) is 32768 or more in
// magnitude or not finite, and the host then declines the step (the CPU path runs it).
extern "C" __global__ void dguard_t(const float* x, unsigned int n, unsigned int xs, unsigned int* flag) {
    const float* r = x + (size_t)blockIdx.y * xs;
    unsigned int top = 0u;
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        top = max(top, __float_as_uint(r[i]) & 0x7fffffffu);
    if (top >= 0x47000000u) *flag = 1u;
}

// ---- Prefill attention on the tensor cores (`dfattn_D`, prefill plans' fast path): the step's
// queries (tokens j at positions sp[4 j], consecutive) attend causally over a cache (rows of
// nkv * D floats, keys at their positions; the last `window` only, if set). A 128-thread block
// takes 64 queries of one head, a warp 16 of them; key tiles of 64 (32 at head dimension 256:
// the tiles' shared memory) go to shared memory as f16 (keys and values), the scores Q K^T and the products P V are f16 MMAs with f32 sums, and the
// softmax runs online in f32 (scores scaled by p.scale before the queries are rounded to f16).
// Out of the step's t tokens, query rows past t are skipped. `gate`: out *= sigmoid(gate[h
// g_stride + i]) as in dattn. Rows of q / out / gate are qs / os / gs floats apart.
__device__ __forceinline__ unsigned int fa_pack(float a, float b) {
    unsigned int r;
    asm("cvt.rn.f16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(a), "f"(b));
    return r;
}
__device__ __forceinline__ void fa_mma(float* c, const unsigned int* a, const unsigned int* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void fa_ldsm4(unsigned int* r, const void* p) {
    unsigned int a = (unsigned int)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void fa_ldsm4t(unsigned int* r, const void* p) {
    unsigned int a = (unsigned int)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
#define FA_Q 64
#define FA_K 64
template <int D, int KT>
__device__ __forceinline__ void dfattn_body(const float* q, const float* kc, const float* vc, float* out, const float* gate,
                                            const unsigned int* sp, DAP p, unsigned int t, unsigned int qs, unsigned int os, unsigned int gs) {
    constexpr unsigned int LD = D + 8; // halves per shared row (ldmatrix rows on distinct banks)
    __shared__ __align__(16) unsigned short Ks[KT * LD];
    __shared__ __align__(16) unsigned short Vs[KT * LD];
    unsigned int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, c4 = lane & 3;
    unsigned int h = blockIdx.y, kvh = h / (p.nh / p.nkv);
    unsigned int j0 = blockIdx.x * FA_Q;
    if (j0 >= t) return;
    unsigned int pos0 = sp[0];
    size_t rs = (size_t)p.nkv * D;
    // This warp's 16 queries as A fragments (D / 16 steps), scaled: rows g and g + 8.
    unsigned int qa[D / 16][4];
    unsigned int qr0 = j0 + 16 * warp + g, qr1 = qr0 + 8;
    {
        const float* q0 = q + (size_t)min(qr0, t - 1) * qs + h * D;
        const float* q1 = q + (size_t)min(qr1, t - 1) * qs + h * D;
#pragma unroll
        for (int kk = 0; kk < D / 16; kk++) {
            unsigned int c = 16 * kk + 2 * c4;
            qa[kk][0] = fa_pack(q0[c] * p.scale, q0[c + 1] * p.scale);
            qa[kk][1] = fa_pack(q1[c] * p.scale, q1[c + 1] * p.scale);
            qa[kk][2] = fa_pack(q0[c + 8] * p.scale, q0[c + 9] * p.scale);
            qa[kk][3] = fa_pack(q1[c + 8] * p.scale, q1[c + 9] * p.scale);
        }
    }
    // Key positions this warp's rows see: [lo, pos] per row.
    unsigned int P0 = pos0 + qr0, P1 = pos0 + qr1;
    unsigned int lo0 = p.window != 0 && P0 + 1 > p.window ? P0 + 1 - p.window : 0;
    unsigned int lo1 = p.window != 0 && P1 + 1 > p.window ? P1 + 1 - p.window : 0;
    // The block's key range.
    unsigned int plast = pos0 + min(j0 + FA_Q, t) - 1, pfirst = pos0 + j0;
    unsigned int klo = p.window != 0 && pfirst + 1 > p.window ? pfirst + 1 - p.window : 0;
    klo = klo / KT * KT;
    const float NEG = -__uint_as_float(0x7f800000u);
    float m0 = NEG, m1 = NEG, l0 = 0.f, l1 = 0.f;
    float o[D / 8][4];
#pragma unroll
    for (int i = 0; i < D / 8; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    for (unsigned int k0 = klo; k0 <= plast; k0 += KT) {
        __syncthreads();
        // The tile's keys and values (rows past plast as zeros) as f16.
        for (unsigned int i = tid; i < KT * D / 4; i += 128) {
            unsigned int r = i / (D / 4), c = 4 * (i % (D / 4));
            float4 kv = make_float4(0.f, 0.f, 0.f, 0.f), vv = kv;
            if (k0 + r <= plast) {
                kv = *(const float4*)(kc + (size_t)(k0 + r) * rs + kvh * D + c);
                vv = *(const float4*)(vc + (size_t)(k0 + r) * rs + kvh * D + c);
            }
            *(uint2*)&Ks[r * LD + c] = make_uint2(fa_pack(kv.x, kv.y), fa_pack(kv.z, kv.w));
            *(uint2*)&Vs[r * LD + c] = make_uint2(fa_pack(vv.x, vv.y), fa_pack(vv.z, vv.w));
        }
        __syncthreads();
        // Scores of the warp's 16 queries against the tile's KT keys, 8 at a time.
        float s[KT / 8][4];
#pragma unroll
        for (int nt = 0; nt < KT / 8; nt++) s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
#pragma unroll
        for (int kk = 0; kk < D / 16; kk++)
#pragma unroll
            for (int np = 0; np < KT / 16; np++) {
                unsigned int r = 16 * np + (lane & 7) + 8 * (lane >> 4), c = 16 * kk + 8 * ((lane >> 3) & 1);
                unsigned int b4[4];
                fa_ldsm4(b4, &Ks[r * LD + c]);
                fa_mma(s[2 * np], qa[kk], b4);
                fa_mma(s[2 * np + 1], qa[kk], b4 + 2);
            }
        // Masks, then the online softmax (rows g: s[..][0..1], g + 8: s[..][2..3]).
        float mx0 = m0, mx1 = m1;
#pragma unroll
        for (int nt = 0; nt < KT / 8; nt++)
#pragma unroll
            for (int e = 0; e < 2; e++) {
                unsigned int kp = k0 + 8 * nt + 2 * c4 + e;
                if (kp > P0 || kp < lo0) s[nt][e] = NEG;
                if (kp > P1 || kp < lo1) s[nt][2 + e] = NEG;
                mx0 = fmaxf(mx0, s[nt][e]);
                mx1 = fmaxf(mx1, s[nt][2 + e]);
            }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        // A row with no key yet (all masked) keeps m = -inf: its factors are taken as 0.
        float a0 = mx0 == NEG ? 1.f : expf(m0 - mx0), a1 = mx1 == NEG ? 1.f : expf(m1 - mx1);
        float rs0 = 0.f, rs1 = 0.f;
        unsigned int pa[KT / 16][4];
#pragma unroll
        for (int nt = 0; nt < KT / 8; nt++) {
            float e0 = mx0 == NEG ? 0.f : expf(s[nt][0] - mx0), e1 = mx0 == NEG ? 0.f : expf(s[nt][1] - mx0);
            float e2 = mx1 == NEG ? 0.f : expf(s[nt][2] - mx1), e3 = mx1 == NEG ? 0.f : expf(s[nt][3] - mx1);
            rs0 += e0 + e1;
            rs1 += e2 + e3;
            unsigned int kk = nt >> 1;
            if ((nt & 1) == 0) {
                pa[kk][0] = fa_pack(e0, e1);
                pa[kk][1] = fa_pack(e2, e3);
            } else {
                pa[kk][2] = fa_pack(e0, e1);
                pa[kk][3] = fa_pack(e2, e3);
            }
        }
        rs0 += __shfl_xor_sync(0xffffffffu, rs0, 1);
        rs0 += __shfl_xor_sync(0xffffffffu, rs0, 2);
        rs1 += __shfl_xor_sync(0xffffffffu, rs1, 1);
        rs1 += __shfl_xor_sync(0xffffffffu, rs1, 2);
        l0 = l0 * a0 + rs0;
        l1 = l1 * a1 + rs1;
        m0 = mx0;
        m1 = mx1;
#pragma unroll
        for (int i = 0; i < D / 8; i++) {
            o[i][0] *= a0;
            o[i][1] *= a0;
            o[i][2] *= a1;
            o[i][3] *= a1;
        }
        // O += P V: V as the B operand by transposed ldmatrix (keys are its K, dimensions its N).
#pragma unroll
        for (int kk = 0; kk < KT / 16; kk++)
#pragma unroll
            for (int dp = 0; dp < D / 16; dp++) {
                unsigned int r = 16 * kk + (lane & 7) + 8 * ((lane >> 3) & 1), c = 16 * dp + 8 * (lane >> 4);
                unsigned int b4[4];
                fa_ldsm4t(b4, &Vs[r * LD + c]);
                fa_mma(o[2 * dp], pa[kk], b4);
                fa_mma(o[2 * dp + 1], pa[kk], b4 + 2);
            }
    }
    float il0 = l0 > 0.f ? 1.f / l0 : 0.f, il1 = l1 > 0.f ? 1.f / l1 : 0.f;
#pragma unroll
    for (int i = 0; i < D / 8; i++)
#pragma unroll
        for (int e = 0; e < 2; e++) {
            unsigned int d = 8 * i + 2 * c4 + e;
            if (qr0 < t) {
                float v = o[i][e] * il0;
                if (p.has_gate) v *= dsig(gate[(size_t)qr0 * gs + h * p.g_stride + d]);
                out[(size_t)qr0 * os + h * D + d] = v;
            }
            if (qr1 < t) {
                float v = o[i][2 + e] * il1;
                if (p.has_gate) v *= dsig(gate[(size_t)qr1 * gs + h * p.g_stride + d]);
                out[(size_t)qr1 * os + h * D + d] = v;
            }
        }
}
#define DFATTN(D, KT) \
extern "C" __global__ void __launch_bounds__(128) dfattn_##D(const float* q, const float* kc, const float* vc, float* out, const float* gate, \
                                                             const unsigned int* sp, DAP p, unsigned int t, unsigned int qs, unsigned int os, \
                                                             unsigned int gs) { dfattn_body<D, KT>(q, kc, vc, out, gate, sp, p, t, qs, os, gs); }
DFATTN(64, 64) DFATTN(128, 64) DFATTN(256, 32)

// ---- `dfattn2_D`: dfattn with the heads of a GQA group packed into one block and the key /
// value tiles streamed. A 256-thread block (8 warps of 16 query rows) takes `hb` heads that
// share a key / value head (hb in 1, 2, 4, 8, dividing the group) times 128 / hb consecutive
// queries, so one tile of 64 keys serves 128 query rows (twice dfattn's). Each tile arrives in
// f32 by cp.async into a staging buffer while the previous one is in the MMAs, then is rounded
// to f16 shared to shared (the same f16 values as dfattn). Blocks are taken longest first
// (last query tile first) so the causal tail does not run alone at the end. A warp skips the
// MMAs of tiles wholly past its last query. Dynamic shared memory: DFA2_SHM(D) bytes.
#define DFA2_SHM(D) (2 * FA_K * (D) * 4 + 2 * FA_K * ((D) + 8) * 2)
__device__ __forceinline__ void fa_cpa16(void* dst, const void* src, unsigned int n) {
    unsigned int a = (unsigned int)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(a), "l"(src), "r"(n));
}
template <int D>
__device__ __forceinline__ void dfattn2_body(const float* q, const float* kc, const float* vc, float* out, const float* gate,
                                             const unsigned int* sp, DAP p, unsigned int t, unsigned int qs, unsigned int os,
                                             unsigned int gs, unsigned int hb) {
    constexpr unsigned int LD = D + 8;
    extern __shared__ __align__(16) unsigned char fa2_smem[];
    float* Kf = (float*)fa2_smem;
    float* Vf = Kf + FA_K * D;
    unsigned short* Ks = (unsigned short*)(Vf + FA_K * D);
    unsigned short* Vs = Ks + FA_K * LD;
    unsigned int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, c4 = lane & 3;
    unsigned int tq = 128 / hb, wph = 8 / hb; // queries per block, warps per head
    unsigned int h = blockIdx.y * hb + warp / wph, kvh = (blockIdx.y * hb) / (p.nh / p.nkv);
    unsigned int j0 = (gridDim.x - 1 - blockIdx.x) * tq;
    if (j0 >= t) return;
    unsigned int pos0 = sp[0];
    size_t rs = (size_t)p.nkv * D;
    unsigned int qr0 = j0 + 16 * (warp % wph) + g, qr1 = qr0 + 8;
    unsigned int qa[D / 16][4];
    {
        const float* q0 = q + (size_t)min(qr0, t - 1) * qs + h * D;
        const float* q1 = q + (size_t)min(qr1, t - 1) * qs + h * D;
#pragma unroll
        for (int kk = 0; kk < D / 16; kk++) {
            unsigned int c = 16 * kk + 2 * c4;
            qa[kk][0] = fa_pack(q0[c] * p.scale, q0[c + 1] * p.scale);
            qa[kk][1] = fa_pack(q1[c] * p.scale, q1[c + 1] * p.scale);
            qa[kk][2] = fa_pack(q0[c + 8] * p.scale, q0[c + 9] * p.scale);
            qa[kk][3] = fa_pack(q1[c + 8] * p.scale, q1[c + 9] * p.scale);
        }
    }
    unsigned int P0 = pos0 + qr0, P1 = pos0 + qr1;
    unsigned int lo0 = p.window != 0 && P0 + 1 > p.window ? P0 + 1 - p.window : 0;
    unsigned int lo1 = p.window != 0 && P1 + 1 > p.window ? P1 + 1 - p.window : 0;
    // The warp's last query position, and the first key any of its rows sees.
    unsigned int wlast = pos0 + min(j0 + 16 * (warp % wph) + 15, t - 1);
    unsigned int wfirst = pos0 + j0 + 16 * (warp % wph);
    unsigned int wlo = p.window != 0 && wfirst + 1 > p.window ? wfirst + 1 - p.window : 0;
    unsigned int plast = pos0 + min(j0 + tq, t) - 1, pfirst = pos0 + j0;
    unsigned int klo = p.window != 0 && pfirst + 1 > p.window ? pfirst + 1 - p.window : 0;
    klo = klo / FA_K * FA_K;
    // Tile k0's f32 keys and values into the staging buffers (rows past plast as zeros).
    auto fetch = [&](unsigned int k0) {
        for (unsigned int i = tid; i < FA_K * D / 4; i += 256) {
            unsigned int r = i / (D / 4), c = 4 * (i % (D / 4));
            bool in = k0 + r <= plast;
            size_t off = in ? (size_t)(k0 + r) * rs + kvh * D + c : 0;
            fa_cpa16(Kf + r * D + c, kc + off, in ? 16u : 0u);
            fa_cpa16(Vf + r * D + c, vc + off, in ? 16u : 0u);
        }
        asm volatile("cp.async.commit_group;");
    };
    auto convert = [&]() {
        for (unsigned int i = tid; i < FA_K * D / 4; i += 256) {
            unsigned int r = i / (D / 4), c = 4 * (i % (D / 4));
            float4 kv = *(const float4*)(Kf + r * D + c), vv = *(const float4*)(Vf + r * D + c);
            *(uint2*)&Ks[r * LD + c] = make_uint2(fa_pack(kv.x, kv.y), fa_pack(kv.z, kv.w));
            *(uint2*)&Vs[r * LD + c] = make_uint2(fa_pack(vv.x, vv.y), fa_pack(vv.z, vv.w));
        }
    };
    const float NEG = -__uint_as_float(0x7f800000u);
    float m0 = NEG, m1 = NEG, l0 = 0.f, l1 = 0.f;
    float o[D / 8][4];
#pragma unroll
    for (int i = 0; i < D / 8; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    fetch(klo);
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    convert();
    __syncthreads();
    if (klo + FA_K <= plast) fetch(klo + FA_K);
    for (unsigned int k0 = klo; k0 <= plast; k0 += FA_K) {
        if (k0 <= wlast && k0 + FA_K > wlo) {
            float s[FA_K / 8][4];
#pragma unroll
            for (int nt = 0; nt < FA_K / 8; nt++) s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
#pragma unroll
            for (int kk = 0; kk < D / 16; kk++)
#pragma unroll
                for (int np = 0; np < FA_K / 16; np++) {
                    unsigned int r = 16 * np + (lane & 7) + 8 * (lane >> 4), c = 16 * kk + 8 * ((lane >> 3) & 1);
                    unsigned int b4[4];
                    fa_ldsm4(b4, &Ks[r * LD + c]);
                    fa_mma(s[2 * np], qa[kk], b4);
                    fa_mma(s[2 * np + 1], qa[kk], b4 + 2);
                }
            float mx0 = m0, mx1 = m1;
#pragma unroll
            for (int nt = 0; nt < FA_K / 8; nt++)
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    unsigned int kp = k0 + 8 * nt + 2 * c4 + e;
                    if (kp > P0 || kp < lo0) s[nt][e] = NEG;
                    if (kp > P1 || kp < lo1) s[nt][2 + e] = NEG;
                    mx0 = fmaxf(mx0, s[nt][e]);
                    mx1 = fmaxf(mx1, s[nt][2 + e]);
                }
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
            float a0 = mx0 == NEG ? 1.f : expf(m0 - mx0), a1 = mx1 == NEG ? 1.f : expf(m1 - mx1);
            float rs0 = 0.f, rs1 = 0.f;
            unsigned int pa[FA_K / 16][4];
#pragma unroll
            for (int nt = 0; nt < FA_K / 8; nt++) {
                float e0 = mx0 == NEG ? 0.f : expf(s[nt][0] - mx0), e1 = mx0 == NEG ? 0.f : expf(s[nt][1] - mx0);
                float e2 = mx1 == NEG ? 0.f : expf(s[nt][2] - mx1), e3 = mx1 == NEG ? 0.f : expf(s[nt][3] - mx1);
                rs0 += e0 + e1;
                rs1 += e2 + e3;
                unsigned int kk = nt >> 1;
                if ((nt & 1) == 0) {
                    pa[kk][0] = fa_pack(e0, e1);
                    pa[kk][1] = fa_pack(e2, e3);
                } else {
                    pa[kk][2] = fa_pack(e0, e1);
                    pa[kk][3] = fa_pack(e2, e3);
                }
            }
            rs0 += __shfl_xor_sync(0xffffffffu, rs0, 1);
            rs0 += __shfl_xor_sync(0xffffffffu, rs0, 2);
            rs1 += __shfl_xor_sync(0xffffffffu, rs1, 1);
            rs1 += __shfl_xor_sync(0xffffffffu, rs1, 2);
            l0 = l0 * a0 + rs0;
            l1 = l1 * a1 + rs1;
            m0 = mx0;
            m1 = mx1;
#pragma unroll
            for (int i = 0; i < D / 8; i++) {
                o[i][0] *= a0;
                o[i][1] *= a0;
                o[i][2] *= a1;
                o[i][3] *= a1;
            }
#pragma unroll
            for (int kk = 0; kk < FA_K / 16; kk++)
#pragma unroll
                for (int dp = 0; dp < D / 16; dp++) {
                    unsigned int r = 16 * kk + (lane & 7) + 8 * ((lane >> 3) & 1), c = 16 * dp + 8 * (lane >> 4);
                    unsigned int b4[4];
                    fa_ldsm4t(b4, &Vs[r * LD + c]);
                    fa_mma(o[2 * dp], pa[kk], b4);
                    fa_mma(o[2 * dp + 1], pa[kk], b4 + 2);
                }
        }
        if (k0 + FA_K <= plast) {
            asm volatile("cp.async.wait_group 0;");
            __syncthreads(); // the staged tile is in, and every warp is done with the f16 one
            convert();
            __syncthreads();
            if (k0 + 2 * FA_K <= plast) fetch(k0 + 2 * FA_K);
        }
    }
    float il0 = l0 > 0.f ? 1.f / l0 : 0.f, il1 = l1 > 0.f ? 1.f / l1 : 0.f;
#pragma unroll
    for (int i = 0; i < D / 8; i++)
#pragma unroll
        for (int e = 0; e < 2; e++) {
            unsigned int d = 8 * i + 2 * c4 + e;
            if (qr0 < t) {
                float v = o[i][e] * il0;
                if (p.has_gate) v *= dsig(gate[(size_t)qr0 * gs + h * p.g_stride + d]);
                out[(size_t)qr0 * os + h * D + d] = v;
            }
            if (qr1 < t) {
                float v = o[i][2 + e] * il1;
                if (p.has_gate) v *= dsig(gate[(size_t)qr1 * gs + h * p.g_stride + d]);
                out[(size_t)qr1 * os + h * D + d] = v;
            }
        }
}
#define DFATTN2(D) \
extern "C" __global__ void __launch_bounds__(256) dfattn2_##D(const float* q, const float* kc, const float* vc, float* out, const float* gate, \
                                                              const unsigned int* sp, DAP p, unsigned int t, unsigned int qs, unsigned int os, \
                                                              unsigned int gs, unsigned int hb) { dfattn2_body<D>(q, kc, vc, out, gate, sp, p, t, qs, os, gs, hb); }
DFATTN2(64) DFATTN2(128)
