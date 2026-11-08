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
    if (lane == 0) act[(size_t)s * inter + row] = (gelu ? dgelu(a) : a / (1.0f + expf(-a))) * b;
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
