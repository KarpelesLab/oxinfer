// MiniMax-H3's transformer blocks on CUDA (`minimax_h3/cuda.rs`), compiled by NVRTC at first use.
//
// Activations are f32; the products run on the tensor cores with f16 operands and f32
// accumulation. An f32 operand goes in as a pair of halves, `hi = f16(v)` and
// `lo = f16((v - hi) 2^11)` (~22 bits), after a power of two that keeps it in f16's normal range
// (per row for activations, per matrix for weights); FP8 E4M3 weights are exact in one half.

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

__device__ __forceinline__ float h2f(u16 h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}
__device__ __forceinline__ u32 pack_h2(float a, float b) {
    u32 r;
    asm("cvt.rn.f16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(a), "f"(b));
    return r;
}
__device__ __forceinline__ void split_h2(float a, float b, u32& hi, u32& lo) {
    hi = pack_h2(a, b);
    float ha, hb;
    asm("{.reg .f16 l, h; mov.b32 {l, h}, %2; cvt.f32.f16 %0, l; cvt.f32.f16 %1, h;}" : "=f"(ha), "=f"(hb) : "r"(hi));
    lo = pack_h2((a - ha) * 2048.f, (b - hb) * 2048.f);
}
__device__ __forceinline__ void mma16816(float* c, const u32* a, const u32* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// An E4M3 byte (sign, 4 exponent bits, 3 mantissa bits) shifted into a half's places is the value
// times 2^-8, subnormals included.
__device__ __forceinline__ u32 f8h(u32 b) { return ((b & 0x80u) << 8) | ((b & 0x7fu) << 7); }
__device__ __forceinline__ u32 f8h2(u32 two) { return f8h(two & 255u) | (f8h((two >> 8) & 255u) << 16); }
// Two signed bytes as two halves (every int8 is exact in a half).
__device__ __forceinline__ u32 i8h2(u32 two) { return pack_h2((float)(signed char)(two & 255u), (float)(signed char)((two >> 8) & 255u)); }

__device__ __forceinline__ float block_sum256(float v, float* red) {
    red[threadIdx.x] = v;
    __syncthreads();
    for (u32 o = 128; o > 0; o >>= 1) {
        if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o];
        __syncthreads();
    }
    float t = red[0];
    __syncthreads();
    return t;
}
__device__ __forceinline__ float block_max256(float v, float* red) {
    red[threadIdx.x] = v;
    __syncthreads();
    for (u32 o = 128; o > 0; o >>= 1) {
        if (threadIdx.x < o) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + o]);
        __syncthreads();
    }
    float t = red[0];
    __syncthreads();
    return t;
}
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
// The power of two f with m f in [2^13, 2^14) (1 for 0, infinities and NaN), and 1 / f.
__device__ __forceinline__ void pow2_for(float m, float& f, float& inv) {
    f = 1.f;
    inv = 1.f;
    u32 e = (__float_as_uint(m) >> 23) & 255u;
    if (m > 0.f && e != 255u) {
        int s = 13 - ((int)e - 127);
        s = s > 100 ? 100 : (s < -100 ? -100 : s);
        f = __uint_as_float((u32)(127 + s) << 23);
        inv = __uint_as_float((u32)(127 - s) << 23);
    }
}

// ---- GEMM: Y[m][n] (= or +=) f rs[m] sum_k X[m][k] W[n][k] + bias[n] (`rs`, `bias`: or null).
// A 128-thread block computes 64
// rows x 64 outputs in 32-wide K steps (the grid: see `cg` below). X: half2 planes `xh` (and `xl`), rows of n_in / 2 words.
// W: FP8 bytes (WK 0; rows of n_in bytes), half2 planes `w` (and `wl`; WK 1) or signed bytes
// (WK 2; rows of n_in bytes, with `cs`: `longcat_avatar/cuda.cu`).
// NP: 1 (xh w), 2 (+ xl w), 3 (+ xh wl).

#define GBM 64
#define GLD 20 // words (half2) per shared row: 16 + 4 padding

struct GemmP {
    const u32* xh;
    const u32* xl;
    const u8* w;
    const u32* wl;
    const float* rs;
    const float* bias;
    float* y;
    u64 ys;
    u32 n_in, n_out, rows, add;
    float f;
    u32 cg;
};

// `cs`: a factor per output (int8 rows with a scale each), or null.
template <int WK, int NP>
__device__ __forceinline__ void gemm_body(const GemmP& p, const float* cs = nullptr) {
    __shared__ u32 smem[(NP == 3 ? 4 : NP == 2 ? 3 : 2) * GBM * GLD];
    u32* As = smem;
    u32* Bs = smem + GBM * GLD;
    u32* Al = smem + (NP >= 2 ? 2 : 0) * GBM * GLD;
    u32* Bl = smem + (NP == 3 ? 3 : 0) * GBM * GLD;
    u32 tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    // The grid: x, one of `cg` tiles of 64 outputs; y, every tile of 64 rows for one group of
    // `cg`, then for the next (a group's weights stay in the cache while the rows pass).
    u32 rt = (p.rows + GBM - 1) / GBM;
    u32 r0 = (blockIdx.y % rt) * GBM;
    u32 nr = min((u32)GBM, p.rows - r0);
    u32 n0 = ((blockIdx.y / rt) * p.cg + blockIdx.x) * 64;
    if (n0 >= p.n_out) return;
    u32 lr = tid >> 1, lh = tid & 1;
    u32 half_in = p.n_in / 2;
    const u32* xhr = p.xh + (u64)(r0 + lr) * half_in;
    const u32* xlr = NP >= 2 ? p.xl + (u64)(r0 + lr) * half_in : nullptr;
    bool xok = lr < nr;
    u32 wr = n0 + lr;
    bool wok = wr < p.n_out;
    const u8* wrow = p.w + (u64)wr * (WK == 1 ? (u64)half_in * 4 : (u64)p.n_in);
    const u32* wlrow = NP == 3 ? p.wl + (u64)wr * half_in : nullptr;
    float acc[2][4][4], accl[2][4][4];
#pragma unroll
    for (int a = 0; a < 2; a++)
#pragma unroll
        for (int b = 0; b < 4; b++)
#pragma unroll
            for (int c = 0; c < 4; c++) {
                acc[a][b][c] = 0.f;
                accl[a][b][c] = 0.f;
            }
    u32 wm = (warp >> 1) * 32, wn = (warp & 1) * 32;
    u32 g = lane >> 2, c4 = lane & 3;
    u32 nb = p.n_in / 32;
    u32 at = lr * GLD + 8 * lh;
    for (u32 b = 0; b < nb; b++) {
        if (xok) {
            const uint4* ph = (const uint4*)(xhr + 16 * b + 8 * lh);
            uint4 h0 = ph[0], h1 = ph[1];
            u32* ah = As + at;
            ah[0] = h0.x, ah[1] = h0.y, ah[2] = h0.z, ah[3] = h0.w, ah[4] = h1.x, ah[5] = h1.y, ah[6] = h1.z, ah[7] = h1.w;
            if constexpr (NP >= 2) {
                const uint4* pl = (const uint4*)(xlr + 16 * b + 8 * lh);
                uint4 l0 = pl[0], l1 = pl[1];
                u32* al = Al + at;
                al[0] = l0.x, al[1] = l0.y, al[2] = l0.z, al[3] = l0.w, al[4] = l1.x, al[5] = l1.y, al[6] = l1.z, al[7] = l1.w;
            }
        } else {
#pragma unroll
            for (int i = 0; i < 8; i++) {
                As[at + i] = 0u;
                if constexpr (NP >= 2) Al[at + i] = 0u;
            }
        }
        if (wok) {
            if constexpr (WK == 0) {
                uint4 v = *(const uint4*)(wrow + 32 * b + 16 * lh);
                u32* bh = Bs + at;
                bh[0] = f8h2(v.x), bh[1] = f8h2(v.x >> 16), bh[2] = f8h2(v.y), bh[3] = f8h2(v.y >> 16);
                bh[4] = f8h2(v.z), bh[5] = f8h2(v.z >> 16), bh[6] = f8h2(v.w), bh[7] = f8h2(v.w >> 16);
            } else if constexpr (WK == 2) {
                uint4 v = *(const uint4*)(wrow + 32 * b + 16 * lh);
                u32* bh = Bs + at;
                bh[0] = i8h2(v.x), bh[1] = i8h2(v.x >> 16), bh[2] = i8h2(v.y), bh[3] = i8h2(v.y >> 16);
                bh[4] = i8h2(v.z), bh[5] = i8h2(v.z >> 16), bh[6] = i8h2(v.w), bh[7] = i8h2(v.w >> 16);
            } else {
                const uint4* ph = (const uint4*)((const u32*)wrow + 16 * b + 8 * lh);
                uint4 h0 = ph[0], h1 = ph[1];
                u32* bh = Bs + at;
                bh[0] = h0.x, bh[1] = h0.y, bh[2] = h0.z, bh[3] = h0.w, bh[4] = h1.x, bh[5] = h1.y, bh[6] = h1.z, bh[7] = h1.w;
                if constexpr (NP == 3) {
                    const uint4* pl = (const uint4*)(wlrow + 16 * b + 8 * lh);
                    uint4 l0 = pl[0], l1 = pl[1];
                    u32* bl = Bl + at;
                    bl[0] = l0.x, bl[1] = l0.y, bl[2] = l0.z, bl[3] = l0.w, bl[4] = l1.x, bl[5] = l1.y, bl[6] = l1.z, bl[7] = l1.w;
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < 8; i++) {
                Bs[at + i] = 0u;
                if constexpr (NP == 3) Bl[at + i] = 0u;
            }
        }
        __syncthreads();
        // NP > 1: this K step's products in fresh accumulators, added to the running sums with
        // f32 adds (the tensor cores' accumulation truncates).
        float st[2][4][4], stl[2][4][4];
        if constexpr (NP >= 2) {
#pragma unroll
            for (int a = 0; a < 2; a++)
#pragma unroll
                for (int bb = 0; bb < 4; bb++)
#pragma unroll
                    for (int c = 0; c < 4; c++) st[a][bb][c] = stl[a][bb][c] = 0.f;
        }
#pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            u32 kb = 8 * kk; // in half2 units
            u32 af[2][4], bf[4][2];
#pragma unroll
            for (int mt = 0; mt < 2; mt++) {
                u32 rb = (wm + 16 * mt + g) * GLD + kb + c4;
                af[mt][0] = As[rb];
                af[mt][1] = As[rb + 8 * GLD];
                af[mt][2] = As[rb + 4];
                af[mt][3] = As[rb + 8 * GLD + 4];
            }
#pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                u32 cb = (wn + 8 * nt + g) * GLD + kb + c4;
                bf[nt][0] = Bs[cb];
                bf[nt][1] = Bs[cb + 4];
            }
#pragma unroll
            for (int mt = 0; mt < 2; mt++)
#pragma unroll
                for (int nt = 0; nt < 4; nt++) {
                    if constexpr (NP >= 2)
                        mma16816(st[mt][nt], af[mt], bf[nt]);
                    else
                        mma16816(acc[mt][nt], af[mt], bf[nt]);
                }
            if constexpr (NP >= 2) {
                u32 al[2][4];
#pragma unroll
                for (int mt = 0; mt < 2; mt++) {
                    u32 rb = (wm + 16 * mt + g) * GLD + kb + c4;
                    al[mt][0] = Al[rb];
                    al[mt][1] = Al[rb + 8 * GLD];
                    al[mt][2] = Al[rb + 4];
                    al[mt][3] = Al[rb + 8 * GLD + 4];
                }
#pragma unroll
                for (int mt = 0; mt < 2; mt++)
#pragma unroll
                    for (int nt = 0; nt < 4; nt++) mma16816(stl[mt][nt], al[mt], bf[nt]);
            }
            if constexpr (NP == 3) {
                u32 bl[4][2];
#pragma unroll
                for (int nt = 0; nt < 4; nt++) {
                    u32 cb = (wn + 8 * nt + g) * GLD + kb + c4;
                    bl[nt][0] = Bl[cb];
                    bl[nt][1] = Bl[cb + 4];
                }
#pragma unroll
                for (int mt = 0; mt < 2; mt++)
#pragma unroll
                    for (int nt = 0; nt < 4; nt++) mma16816(stl[mt][nt], af[mt], bl[nt]);
            }
        }
        if constexpr (NP >= 2) {
#pragma unroll
            for (int a = 0; a < 2; a++)
#pragma unroll
                for (int bb = 0; bb < 4; bb++)
#pragma unroll
                    for (int c = 0; c < 4; c++) {
                        acc[a][bb][c] += st[a][bb][c];
                        accl[a][bb][c] += stl[a][bb][c];
                    }
        }
        __syncthreads();
    }
    if constexpr (NP >= 2) {
#pragma unroll
        for (int a = 0; a < 2; a++)
#pragma unroll
            for (int b = 0; b < 4; b++)
#pragma unroll
                for (int c = 0; c < 4; c++) acc[a][b][c] += accl[a][b][c] * (1.f / 2048.f);
    }
#pragma unroll
    for (int mt = 0; mt < 2; mt++)
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
#pragma unroll
            for (int q = 0; q < 4; q++) {
                u32 rr = wm + 16 * mt + g + 8 * (q >> 1), cc = n0 + wn + 8 * nt + 2 * c4 + (q & 1);
                if (rr < nr && cc < p.n_out) {
                    float v = acc[mt][nt][q] * p.f;
                    if (cs) v *= cs[cc];
                    if (p.rs) v *= p.rs[r0 + rr];
                    if (p.bias) v += p.bias[cc];
                    float* o = p.y + (u64)(r0 + rr) * p.ys + cc;
                    if (p.add) v += *o;
                    *o = v;
                }
            }
}

extern "C" __global__ void __launch_bounds__(128) gemm_f8_1(GemmP p) { gemm_body<0, 1>(p); }
extern "C" __global__ void __launch_bounds__(128) gemm_f8_2(GemmP p) { gemm_body<0, 2>(p); }
extern "C" __global__ void __launch_bounds__(128) gemm_h_1(GemmP p) { gemm_body<1, 1>(p); }
extern "C" __global__ void __launch_bounds__(128) gemm_h_2(GemmP p) { gemm_body<1, 2>(p); }
extern "C" __global__ void __launch_bounds__(128) gemm_h_3(GemmP p) { gemm_body<1, 3>(p); }

// ---- the rows of a GEMM's input as half planes: row r of x (k values, rows `xs` floats apart)
// times the power of two that brings its largest value into [2^13, 2^14); rs[r] is its inverse.
extern "C" __global__ void __launch_bounds__(256) split_rows(const float* x, u64 xs, u32 k, u32* hi, u32* lo, float* rs) {
    __shared__ float red[256];
    u32 r = blockIdx.x, tid = threadIdx.x;
    const float* row = x + (u64)r * xs;
    float m = 0.f;
    for (u32 i = tid; i < k; i += 256) m = fmaxf(m, fabsf(row[i]));
    m = block_max256(m, red);
    float f, inv;
    pow2_for(m, f, inv);
    u32 per = k / 2;
    for (u32 i = tid; i < per; i += 256) {
        float2 v = *(const float2*)(row + 2 * i);
        split_h2(v.x * f, v.y * f, hi[(u64)r * per + i], lo[(u64)r * per + i]);
    }
    if (tid == 0) rs[r] = inv;
}

// ---- out[r] = rmsnorm(x[r]) w, then (1 + scale) and + shift of the row's AdaLN set
// (`tab`: sets of 6 h values, `key[r]` the row's set, vectors `which` and `which + 1`; null: no
// modulation).
extern "C" __global__ void __launch_bounds__(256) norm_mod(const float* x, const float* w, const float* tab, const u32* key, float* out, u32 h, float eps, u32 which) {
    __shared__ float red[256];
    u32 r = blockIdx.x, tid = threadIdx.x;
    const float* row = x + (u64)r * h;
    float s = 0.f;
    for (u32 i = tid; i < h; i += 256) s += row[i] * row[i];
    float ss = block_sum256(s, red) / (float)h + eps;
    float rr = 1.f / sqrtf(ss);
    float* o = out + (u64)r * h;
    if (!tab) {
        for (u32 i = tid; i < h; i += 256) o[i] = row[i] * (rr * w[i]);
        return;
    }
    const float* sh = tab + (u64)key[r] * 6 * h + (u64)which * h;
    const float* sc = sh + h;
    for (u32 i = tid; i < h; i += 256) {
        float v = row[i] * (rr * w[i]);
        o[i] = v * (1.f + sc[i]) + sh[i];
    }
}

// ---- x[r] += gate y[r], the gate vector `which` of the row's AdaLN set (`key` null: of set 0).
extern "C" __global__ void __launch_bounds__(256) resid(float* x, const float* y, const float* tab, const u32* key, u32 h, u32 which) {
    u32 r = blockIdx.y;
    u32 i = blockIdx.x * 256 + threadIdx.x;
    if (i >= h) return;
    const float* gt = tab + (u64)(key ? key[r] : 0u) * 6 * h + (u64)which * h;
    x[(u64)r * h + i] += gt[i] * y[(u64)r * h + i];
}

// ---- act = value * silu(gate) over [rows, ffn] of gu ([rows, 2 ffn]: gate rows first, or value).
extern "C" __global__ void __launch_bounds__(256) swiglu(const float* gu, float* act, u32 rows, u32 ffn, u32 gate_first) {
    u64 i = (u64)blockIdx.x * 256 + threadIdx.x;
    if (i >= (u64)rows * ffn) return;
    u64 r = i / ffn, j = i % ffn;
    const float* row = gu + r * 2 * ffn;
    float g = gate_first ? row[j] : row[ffn + j];
    float u = gate_first ? row[ffn + j] : row[j];
    act[i] = u * g / (1.f + expf(-g));
}

// ---- queries or keys as half planes: one warp per (row, head). The head's hd values (at
// `off0 + head * hstep` of the row of qkv) are RMS normed with weights `w` (null: none), their first 2 * half
// rotated half against half by the row's cos / sin; written at [head][row][hd / 2].
extern "C" __global__ void __launch_bounds__(32) qk_prep(const float* qkv, u64 xs, u32 off0, u32 hstep, const float* w, const float* cs, const float* sn, u32 half, u32 hd, u32 n, float eps, u32* hi, u32* lo) {
    u32 row = blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
    const float* v = qkv + (u64)row * xs + off0 + (u64)head * hstep;
    u32 per = hd / 32;
    float s = 0.f;
    for (u32 c = lane * per; c < (lane + 1) * per; c++) s += v[c] * v[c];
    float ss = warp_sum(s) / (float)hd + eps;
    float r = 1.f / sqrtf(ss);
    const float* cr = cs + (u64)row * half;
    const float* sr = sn + (u64)row * half;
    u64 base = ((u64)head * n + row) * (hd / 2);
    for (u32 c = lane * per; c < (lane + 1) * per; c += 2) {
        float o[2];
        for (u32 j = 0; j < 2; j++) {
            u32 cc = c + j;
            float a = v[cc] * (w ? r * w[cc] : r);
            if (cc < half) {
                float b = v[cc + half] * (w ? r * w[cc + half] : r);
                a = a * cr[cc] - b * sr[cc];
            } else if (cc < 2 * half) {
                float b = v[cc - half] * (w ? r * w[cc - half] : r);
                a = a * cr[cc - half] + b * sr[cc - half];
            }
            o[j] = a;
        }
        split_h2(o[0], o[1], hi[base + c / 2], lo[base + c / 2]);
    }
}

// ---- vs[head] = 1 / f, f the power of two that brings the head's largest |value| into
// [2^13, 2^14): one block per head.
extern "C" __global__ void __launch_bounds__(256) v_scale(const float* qkv, u64 xs, u32 off0, u32 hstep, u32 hd, u32 n, float* vs) {
    __shared__ float red[256];
    u32 head = blockIdx.x, tid = threadIdx.x;
    float m = 0.f;
    u64 total = (u64)n * hd;
    for (u64 i = tid; i < total; i += 256) m = fmaxf(m, fabsf(qkv[(i / hd) * xs + off0 + (u64)head * hstep + i % hd]));
    m = block_max256(m, red);
    float f, inv;
    pow2_for(m, f, inv);
    if (tid == 0) vs[head] = inv;
}

// ---- the values transposed as half planes [head][c][nk / 2] (keys past n are zero), scaled by
// 1 / vs[head].
extern "C" __global__ void __launch_bounds__(256) v_planes(const float* qkv, u64 xs, u32 off0, u32 hstep, u32 hd, u32 n, u32 nk, const float* vs, u32* hi, u32* lo) {
    u32 hc = blockIdx.y, head = hc / hd, c = hc % hd;
    u32 i = blockIdx.x * 256 + threadIdx.x;
    if (i >= nk / 2) return;
    float f = 1.f / vs[head];
    const float* col = qkv + off0 + (u64)head * hstep + c;
    u32 j = 2 * i;
    float a = j < n ? col[(u64)j * xs] * f : 0.f;
    float b = j + 1 < n ? col[(u64)(j + 1) * xs] * f : 0.f;
    split_h2(a, b, hi[(u64)hc * (nk / 2) + i], lo[(u64)hc * (nk / 2) + i]);
}

// ---- the softmax of each row of scores (`cols` of `ld` per row, times `scale`) as half planes of
// p 2^14 (columns past `cols` zero); prs[row] = 2^-14 vs[0] (the output factor of the row's
// product with the value planes). One block per row; the scores are overwritten.
extern "C" __global__ void __launch_bounds__(256) softmax_split(float* s, u32 cols, u32 ld, float scale, u32* ph, u32* pl, float* prs, const float* vs) {
    __shared__ float red[256];
    u32 r = blockIdx.x, tid = threadIdx.x;
    float* row = s + (u64)r * ld;
    float m = -__int_as_float(0x7f800000);
    for (u32 i = tid; i < cols; i += 256) m = fmaxf(m, row[i]);
    m = block_max256(m, red);
    float sum = 0.f;
    for (u32 i = tid; i < cols; i += 256) {
        float e = expf((row[i] - m) * scale);
        row[i] = e;
        sum += e;
    }
    float inv = 16384.f / block_sum256(sum, red);
    u32 per = ld / 2;
    for (u32 i = tid; i < per; i += 256) {
        u32 j = 2 * i;
        float a = j < cols ? row[j] * inv : 0.f;
        float b = j + 1 < cols ? row[j + 1] * inv : 0.f;
        split_h2(a, b, ph[(u64)r * per + i], pl[(u64)r * per + i]);
    }
    if (tid == 0) prs[r] = vs[0] * (1.f / 16384.f);
}

// ---- attention without the score matrix (heads of HD = 64 or 128): one 256-thread block per
// (64 queries, head) walks the keys 64 at a time. Per step: the scores of 64 x 64 (three
// tensor-core products of the query and key planes) stay in the accumulators; each row's running
// maximum m and sum l are updated, the sums so far rescaled by exp(m_old - m_new); exp(s - m) 2^14
// goes to shared memory as half planes and is multiplied with the 64 keys' value planes (three
// products). out[q][c] = O / l vs[head] 2^-14. Shared memory: dynamic, see `flash_smem` in
// cuda.rs. q, k: planes [head][n][HD / 2 words]; v: planes [head][HD][nk / 2 words].

#define FQ 64
#define FK 64
#define PLD 36 // words per shared row of 64 halves

struct AttP {
    const u32* qh;
    const u32* ql;
    const u32* kh;
    const u32* kl;
    const u32* vh;
    const u32* vl;
    const float* vs;
    float* out;
    u64 os;
    u32 n, nk;
    float scale;
    u32 pad;
};

__device__ __forceinline__ void put8(u32* d, const u32* s) {
    uint4 a = ((const uint4*)s)[0], b = ((const uint4*)s)[1];
    d[0] = a.x, d[1] = a.y, d[2] = a.z, d[3] = a.w, d[4] = b.x, d[5] = b.y, d[6] = b.z, d[7] = b.w;
}
__device__ __forceinline__ void zero8(u32* d) {
#pragma unroll
    for (int i = 0; i < 8; i++) d[i] = 0u;
}
__device__ __forceinline__ void store4(u32* d, uint4 v) { d[0] = v.x, d[1] = v.y, d[2] = v.z, d[3] = v.w; }

// One 32-wide K step (from word k0) of a warp's 32 x 8 NT products: A rows from `a0` on in `ah` /
// `al` (rows of `ald` words), B rows from `b0` on in `bh` / `bl`, three products; their sum
// `hi + lo / 2048` is added to `acc`.
template <int NT>
__device__ __forceinline__ void flash_step(float acc[2][NT][4], const u32* ah, const u32* al, u32 ald, const u32* bh, const u32* bl, u32 bld, u32 a0, u32 b0, u32 g, u32 c4, u32 k0) {
    float st[2][NT][4], stl[2][NT][4];
#pragma unroll
    for (int a = 0; a < 2; a++)
#pragma unroll
        for (int b = 0; b < NT; b++)
#pragma unroll
            for (int c = 0; c < 4; c++) st[a][b][c] = stl[a][b][c] = 0.f;
#pragma unroll
    for (int kk = 0; kk < 2; kk++) {
        u32 kb = k0 + 8 * kk;
        u32 af[2][4], alo[2][4], bf[NT][2], blo[NT][2];
#pragma unroll
        for (int mt = 0; mt < 2; mt++) {
            u32 rb = (a0 + 16 * mt + g) * ald + kb + c4;
            af[mt][0] = ah[rb], af[mt][1] = ah[rb + 8 * ald], af[mt][2] = ah[rb + 4], af[mt][3] = ah[rb + 8 * ald + 4];
            alo[mt][0] = al[rb], alo[mt][1] = al[rb + 8 * ald], alo[mt][2] = al[rb + 4], alo[mt][3] = al[rb + 8 * ald + 4];
        }
#pragma unroll
        for (int nt = 0; nt < NT; nt++) {
            u32 cb = (b0 + 8 * nt + g) * bld + kb + c4;
            bf[nt][0] = bh[cb], bf[nt][1] = bh[cb + 4];
            blo[nt][0] = bl[cb], blo[nt][1] = bl[cb + 4];
        }
#pragma unroll
        for (int mt = 0; mt < 2; mt++)
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                mma16816(st[mt][nt], af[mt], bf[nt]);
                mma16816(stl[mt][nt], alo[mt], bf[nt]);
            }
#pragma unroll
        for (int mt = 0; mt < 2; mt++)
#pragma unroll
            for (int nt = 0; nt < NT; nt++) mma16816(stl[mt][nt], af[mt], blo[nt]);
    }
#pragma unroll
    for (int a = 0; a < 2; a++)
#pragma unroll
        for (int b = 0; b < NT; b++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[a][b][c] += st[a][b][c] + stl[a][b][c] * (1.f / 2048.f);
}

// Queries `qlo..qhi` against keys `klo..khi` (rows of the same planes: `p.n` is their stride).
template <int HD>
__device__ __forceinline__ void flash_body(const AttP& p, u32 qlo, u32 qhi, u32 klo, u32 khi) {
    const u32 HW = HD / 2, QLD = HW + 4;
    const u32 BW = FK * QLD > HD * PLD ? FK * QLD : HD * PLD;
    const int NO = HD / 32; // 8-wide output tiles per warp (a quarter of the head's channels)
    extern __shared__ u32 sm[];
    u32* Qh = sm;
    u32* Ql = Qh + FQ * QLD;
    u32* Bh = Ql + FQ * QLD; // the step's keys (rows of QLD), then its values (HD rows of PLD)
    u32* Bl = Bh + BW;
    u32* Ph = Bl + BW;
    u32* Pl = Ph + FQ * PLD;
    float* rmax = (float*)(Pl + FQ * PLD); // [4][FQ]: per key quarter of the step
    float* rsum = rmax + 4 * FQ;
    u32 tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    u32 head = blockIdx.y, q0 = qlo + blockIdx.x * FQ;
    if (q0 >= qhi) return;
    u32 nq = min((u32)FQ, qhi - q0);
    // A thread's part of a 64-row tile of HD halves: a quarter of a row.
    u32 tr = tid >> 2, to = (HW / 4) * (tid & 3);
#pragma unroll
    for (u32 i = 0; i < HW / 4; i += 8) {
        if (tr < nq) {
            u64 at = ((u64)head * p.n + q0 + tr) * HW + to + i;
            put8(Qh + tr * QLD + to + i, p.qh + at);
            put8(Ql + tr * QLD + to + i, p.ql + at);
        } else {
            zero8(Qh + tr * QLD + to + i);
            zero8(Ql + tr * QLD + to + i);
        }
    }
    // Its part of the HD rows of 64 values: 256 / HD threads per row.
    const u32 TPR = 256 / HD;
    u32 vr = tid / TPR, vo = (32 / TPR) * (tid % TPR);
    // This warp's 32 queries, its 16 keys of a step, its HD / 4 output channels.
    u32 wm = (warp >> 2) * 32, kw = warp & 3, wn = kw * 16, cn = kw * (HD / 4);
    u32 g = lane >> 2, c4 = lane & 3;
    const float ninf = -__int_as_float(0x7f800000);
    // Per thread: 4 rows (mt, q >> 1).
    float o[2][NO][4], m[2][2], l[2][2];
#pragma unroll
    for (int a = 0; a < 2; a++) {
#pragma unroll
        for (int b = 0; b < NO; b++)
#pragma unroll
            for (int c = 0; c < 4; c++) o[a][b][c] = 0.f;
        m[a][0] = m[a][1] = ninf;
        l[a][0] = l[a][1] = 0.f;
    }
    u32 half_nk = p.nk / 2;
    // A step's keys and values are read from the device's memory into registers one phase ahead
    // (while the products before them run), then stored to shared memory.
    const int KN = HW / 16, VN = 8 / TPR;
    uint4 kr[2][KN], vq[2][VN];
    auto read_keys = [&](u32 k0) {
#pragma unroll
        for (int i = 0; i < KN; i++) {
            if (k0 + tr < p.n) {
                u64 at = ((u64)head * p.n + k0 + tr) * HW + to + 4 * i;
                kr[0][i] = *(const uint4*)(p.kh + at);
                kr[1][i] = *(const uint4*)(p.kl + at);
            } else {
                kr[0][i] = kr[1][i] = make_uint4(0u, 0u, 0u, 0u);
            }
        }
    };
    read_keys(klo / FK * FK);
    for (u32 k0 = klo / FK * FK; k0 < khi; k0 += FK) {
#pragma unroll
        for (int i = 0; i < KN; i++) {
            store4(Bh + tr * QLD + to + 4 * i, kr[0][i]);
            store4(Bl + tr * QLD + to + 4 * i, kr[1][i]);
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < VN; i++) {
            u32 w0 = k0 / 2 + vo + 4 * i;
            if (w0 < half_nk) {
                u64 at = ((u64)head * HD + vr) * half_nk + w0;
                vq[0][i] = *(const uint4*)(p.vh + at);
                vq[1][i] = *(const uint4*)(p.vl + at);
            } else {
                vq[0][i] = vq[1][i] = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        float s[2][2][4];
#pragma unroll
        for (int a = 0; a < 2; a++)
#pragma unroll
            for (int b = 0; b < 2; b++)
#pragma unroll
                for (int c = 0; c < 4; c++) s[a][b][c] = 0.f;
#pragma unroll
        for (u32 k = 0; k < HW; k += 16) flash_step<2>(s, Qh, Ql, QLD, Bh, Bl, QLD, wm, wn, g, c4, k);
        // The scaled scores (keys past n: -inf) and each row's maximum over this warp's keys.
#pragma unroll
        for (int mt = 0; mt < 2; mt++) {
            float mx[2] = {ninf, ninf};
#pragma unroll
            for (int nt = 0; nt < 2; nt++)
#pragma unroll
                for (int q = 0; q < 4; q++) {
                    u32 kc = k0 + wn + 8 * nt + 2 * c4 + (q & 1);
                    float v = kc >= klo && kc < khi ? s[mt][nt][q] * p.scale : ninf;
                    s[mt][nt][q] = v;
                    mx[q >> 1] = fmaxf(mx[q >> 1], v);
                }
#pragma unroll
            for (int hq = 0; hq < 2; hq++) {
                float v = mx[hq];
                v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
                v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
                if (c4 == 0) rmax[kw * FQ + wm + 16 * mt + g + 8 * hq] = v;
            }
        }
        __syncthreads();
        // Every warp has its scores: the keys' place takes the values of the step's keys.
#pragma unroll
        for (int i = 0; i < VN; i++) {
            store4(Bh + vr * PLD + vo + 4 * i, vq[0][i]);
            store4(Bl + vr * PLD + vo + 4 * i, vq[1][i]);
        }
        float alpha[2][2];
#pragma unroll
        for (int mt = 0; mt < 2; mt++) {
#pragma unroll
            for (int hq = 0; hq < 2; hq++) {
                u32 rr = wm + 16 * mt + g + 8 * hq;
                float mn = fmaxf(fmaxf(m[mt][hq], rmax[rr]), fmaxf(rmax[FQ + rr], fmaxf(rmax[2 * FQ + rr], rmax[3 * FQ + rr])));
                alpha[mt][hq] = expf(m[mt][hq] - mn);
                m[mt][hq] = mn;
            }
            float sum[2] = {0.f, 0.f};
#pragma unroll
            for (int nt = 0; nt < 2; nt++)
#pragma unroll
                for (int hq = 0; hq < 2; hq++) {
                    float e0 = expf(s[mt][nt][2 * hq] - m[mt][hq]);
                    float e1 = expf(s[mt][nt][2 * hq + 1] - m[mt][hq]);
                    sum[hq] += e0 + e1;
                    u32 at = (wm + 16 * mt + g + 8 * hq) * PLD + (wn >> 1) + 4 * nt + c4;
                    split_h2(e0 * 16384.f, e1 * 16384.f, Ph[at], Pl[at]);
                }
#pragma unroll
            for (int hq = 0; hq < 2; hq++) {
                float v = sum[hq];
                v += __shfl_xor_sync(0xffffffffu, v, 1);
                v += __shfl_xor_sync(0xffffffffu, v, 2);
                if (c4 == 0) rsum[kw * FQ + wm + 16 * mt + g + 8 * hq] = v;
            }
        }
        __syncthreads();
        read_keys(k0 + FK);
#pragma unroll
        for (int mt = 0; mt < 2; mt++) {
#pragma unroll
            for (int hq = 0; hq < 2; hq++) {
                u32 rr = wm + 16 * mt + g + 8 * hq;
                l[mt][hq] = l[mt][hq] * alpha[mt][hq] + ((rsum[rr] + rsum[FQ + rr]) + (rsum[2 * FQ + rr] + rsum[3 * FQ + rr]));
            }
#pragma unroll
            for (int nt = 0; nt < NO; nt++)
#pragma unroll
                for (int q = 0; q < 4; q++) o[mt][nt][q] *= alpha[mt][q >> 1];
        }
        flash_step<NO>(o, Ph, Pl, PLD, Bh, Bl, PLD, wm, cn, g, c4, 0);
        flash_step<NO>(o, Ph, Pl, PLD, Bh, Bl, PLD, wm, cn, g, c4, 16);
        __syncthreads();
    }
    float f = p.vs[head] * (1.f / 16384.f);
#pragma unroll
    for (int mt = 0; mt < 2; mt++)
#pragma unroll
        for (int nt = 0; nt < NO; nt++)
#pragma unroll
            for (int q = 0; q < 4; q++) {
                u32 rr = wm + 16 * mt + g + 8 * (q >> 1), cc = cn + 8 * nt + 2 * c4 + (q & 1);
                if (rr < nq) p.out[(u64)(q0 + rr) * p.os + (u64)head * HD + cc] = o[mt][nt][q] * (f / l[mt][q >> 1]);
            }
}

extern "C" __global__ void __launch_bounds__(256) attn_flash_64(AttP p) { flash_body<64>(p, 0, p.n, 0, p.n); }
extern "C" __global__ void __launch_bounds__(256) attn_flash_128(AttP p) { flash_body<128>(p, 0, p.n, 0, p.n); }
