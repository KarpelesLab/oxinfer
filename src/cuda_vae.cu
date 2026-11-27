// Convolutional VAEs on CUDA (`cuda_vae.rs`): float32 frames `[pixels][channels]` that stay on
// the device from one layer to the next.

typedef unsigned int u32;
typedef unsigned long long u64;

// ---- a causal 3D convolution as an implicit GEMM in float32. K runs over (ky, kx, frame, channel)
// with `cin` a multiple of 32, so a 32-wide K step is a contiguous channel slice of one input
// pixel of one of the `kt` frames (`x0`, `x1`, `x2`; a null frame and the pixels outside the
// image are zeros). A frame holds `xstride` floats per pixel, the convolution's channels from
// `xoff` on. `up`: the input is the frame upsampled 2x (nearest). `w`: `[rows][K]` and `bias`:
// `[rows]`, rows a multiple of 64 with zeros after the real ones.
// y[pixel][c] = sum + bias[c] (+ add[pixel][c]) for the first `cout` rows, `cout` floats per pixel.
// A 256-thread block computes 64 pixels x 64 outputs; the grid's x is the pixel tile (a tile of
// outputs keeps its weights in the cache while the pixels pass), y the output tile.
struct VConvP {
    const float* x0;
    const float* x1;
    const float* x2;
    const float* w;
    const float* bias;
    const float* add;
    float* y;
    u32 h, wd, cin, xstride, xoff, k, kt, stride, pad, up, ow, npix, cout, pad2;
};

extern "C" __global__ void __launch_bounds__(256) vconv(VConvP p) {
    __shared__ __align__(16) float Ws[32][64];
    __shared__ __align__(16) float Xs[32][64];
    const u32 t0 = blockIdx.x * 64, o0 = blockIdx.y * 64;
    const u32 tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
    const u32 lrow = tid / 4, lk = (tid % 4) * 8;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.f;
    const u32 kk = p.k * p.k * p.kt * p.cin;
    const u32 tok = t0 + lrow;
    const bool tok_ok = tok < p.npix;
    const int oy = (int)(tok / p.ow) * (int)p.stride - (int)p.pad, ox = (int)(tok % p.ow) * (int)p.stride - (int)p.pad;
    const int ih = (int)(p.up ? 2 * p.h : p.h), iw = (int)(p.up ? 2 * p.wd : p.wd);
    const float* wr = p.w + (u64)(o0 + lrow) * kk + lk;
    const u32 cb = p.cin / 32;
    u32 b = 0;
    for (u32 ky = 0; ky < p.k; ky++)
        for (u32 kx = 0; kx < p.k; kx++)
            for (u32 j = 0; j < p.kt; j++) {
                const float* xf = j == 0 ? p.x0 : (j == 1 ? p.x1 : p.x2);
                int iy = oy + (int)ky, ix = ox + (int)kx;
                bool ok = tok_ok && xf && iy >= 0 && ix >= 0 && iy < ih && ix < iw;
                if (p.up) {
                    iy >>= 1;
                    ix >>= 1;
                }
                const float* xr = ok ? xf + ((u64)iy * p.wd + (u64)ix) * p.xstride + p.xoff + lk : nullptr;
                for (u32 c = 0; c < cb; c++, b++) {
                    const float4* pw = (const float4*)(wr + (u64)b * 32);
                    float4 w0 = pw[0], w1 = pw[1];
                    float4 a0 = make_float4(0.f, 0.f, 0.f, 0.f), a1 = a0;
                    if (ok) {
                        const float4* px = (const float4*)(xr + c * 32);
                        a0 = px[0];
                        a1 = px[1];
                    }
                    Ws[lk][lrow] = w0.x, Ws[lk + 1][lrow] = w0.y, Ws[lk + 2][lrow] = w0.z, Ws[lk + 3][lrow] = w0.w;
                    Ws[lk + 4][lrow] = w1.x, Ws[lk + 5][lrow] = w1.y, Ws[lk + 6][lrow] = w1.z, Ws[lk + 7][lrow] = w1.w;
                    Xs[lk][lrow] = a0.x, Xs[lk + 1][lrow] = a0.y, Xs[lk + 2][lrow] = a0.z, Xs[lk + 3][lrow] = a0.w;
                    Xs[lk + 4][lrow] = a1.x, Xs[lk + 5][lrow] = a1.y, Xs[lk + 6][lrow] = a1.z, Xs[lk + 7][lrow] = a1.w;
                    __syncthreads();
#pragma unroll
                    for (int k = 0; k < 32; k++) {
                        float4 wv = *(const float4*)&Ws[k][ty * 4];
                        float4 xv = *(const float4*)&Xs[k][tx * 4];
                        float wa[4] = {wv.x, wv.y, wv.z, wv.w};
                        float xa[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
                        for (int i = 0; i < 4; i++)
#pragma unroll
                            for (int q = 0; q < 4; q++) acc[i][q] += wa[i] * xa[q];
                    }
                    __syncthreads();
                }
            }
#pragma unroll
    for (int q = 0; q < 4; q++) {
        u32 t = t0 + tx * 4 + q;
        if (t >= p.npix) continue;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            u32 cc = o0 + ty * 4 + i;
            if (cc >= p.cout) continue;
            float v = acc[i][q];
            v += p.bias[cc];
            u64 at = (u64)t * p.cout + cc;
            if (p.add) v += p.add[at];
            p.y[at] = v;
        }
    }
}

// ---- out[pixel] = x[pixel] / max(|x[pixel]|, 1e-12) * g (then SiLU): one warp per pixel.
extern "C" __global__ void __launch_bounds__(256) vrms(const float* x, const float* g, float* out, u32 c, u32 npix, u32 silu) {
    u32 pix = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (pix >= npix) return;
    const float* row = x + (u64)pix * c;
    float s = 0.f;
    for (u32 i = lane; i < c; i += 32) s += row[i] * row[i];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    float inv = 1.f / fmaxf(sqrtf(s), 1e-12f);
    float* o = out + (u64)pix * c;
    for (u32 i = lane; i < c; i += 32) {
        float a = row[i] * inv * g[i];
        o[i] = silu ? a / (1.f + expf(-a)) : a;
    }
}

// ---- the same convolution on the tensor cores: float16 operands, float32 accumulation. The
// weights are half planes `[rows][K / 2 words]` (the float32 ones times a power of two, exact:
// the host checks); an activation v goes in as two halves, `hi = f16(v f)` and
// `lo = f16((v f - hi) 2^11)`, f the power of two that brings its frame's largest |value|
// (`m0`, `m1`, `m2`) into [2^13, 2^14): two products per K step, summed in float32 per step
// (the tensor cores' accumulation truncates) and scaled back per frame.
// A 128-thread block computes 64 pixels x 64 outputs, each warp 32 x 32; the grid is `vconv`'s.

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
__device__ __forceinline__ u32 smem_addr(const void* p) {
    u32 a;
    asm("{ .reg .u64 s; cvta.to.shared.u64 s, %1; cvt.u32.u64 %0, s; }" : "=r"(a) : "l"(p));
    return a;
}
// Four 8 x 8 tiles of halves from shared memory: lane l gives the address of row l % 8 of tile
// l / 8, and receives of each tile the pair (row lane / 4, columns 2 (lane % 4) and the next).
__device__ __forceinline__ void ldmatrix4(u32* r, const u32* at) {
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_addr(at)));
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

// part[block] = the largest |x| of the block's `chunk` values of `n`.
extern "C" __global__ void __launch_bounds__(256) vmax(const float* x, u64 n, u64 chunk, float* part) {
    __shared__ float red[256];
    u32 tid = threadIdx.x;
    u64 lo = (u64)blockIdx.x * chunk, hi = lo + chunk < n ? lo + chunk : n;
    float m = 0.f;
    for (u64 i = lo + tid; i < hi; i += 256) m = fmaxf(m, fabsf(x[i]));
    red[tid] = m;
    __syncthreads();
    for (u32 o = 128; o > 0; o >>= 1) {
        if (tid < o) red[tid] = fmaxf(red[tid], red[tid + o]);
        __syncthreads();
    }
    if (tid == 0) part[blockIdx.x] = red[0];
}

#define HLD 20 // words (half2) per shared row: 16 + 4 padding

struct VConvHP {
    const float* x0;
    const float* x1;
    const float* x2;
    const float* m0;
    const float* m1;
    const float* m2;
    const u32* w;
    const float* bias;
    const float* add;
    float* y;
    float winv;
    u32 h, wd, cin, xstride, xoff, k, kt, stride, pad, up, ow, npix, cout;
};

extern "C" __global__ void __launch_bounds__(128) vconv_h(VConvHP p) {
    __shared__ __align__(16) u32 As[64 * HLD];
    __shared__ __align__(16) u32 Al[64 * HLD];
    __shared__ __align__(16) u32 Bs[64 * HLD];
    const u32 tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const u32 t0 = blockIdx.x * 64, o0 = blockIdx.y * 64;
    const u32 lr = tid >> 1, lh = tid & 1;
    const u32 tok = t0 + lr;
    const bool tok_ok = tok < p.npix;
    const int oy = (int)(tok / p.ow) * (int)p.stride - (int)p.pad, ox = (int)(tok % p.ow) * (int)p.stride - (int)p.pad;
    const int ih = (int)(p.up ? 2 * p.h : p.h), iw = (int)(p.up ? 2 * p.wd : p.wd);
    const u32 half_k = p.k * p.k * p.kt * p.cin / 2;
    const u32* wr = p.w + (u64)(o0 + lr) * half_k + 8 * lh;
    float tot[2][4][4];
#pragma unroll
    for (int a = 0; a < 2; a++)
#pragma unroll
        for (int b = 0; b < 4; b++)
#pragma unroll
            for (int c = 0; c < 4; c++) tot[a][b][c] = 0.f;
    const u32 wm = (warp >> 1) * 32, wn = (warp & 1) * 32;
    const u32 ra = (wm + (lane & 15)) * HLD + 4 * (lane >> 4);
    const u32* lb = Bs + (wn + (lane & 7) + 8 * (lane >> 4)) * HLD + 4 * ((lane >> 3) & 1);
    u32* ah = As + lr * HLD + 8 * lh;
    u32* al = Al + lr * HLD + 8 * lh;
    uint4* bs = (uint4*)(Bs + lr * HLD + 8 * lh);
    const u32 cb = p.cin / 32;
    u32 b = 0;
    for (u32 ky = 0; ky < p.k; ky++)
        for (u32 kx = 0; kx < p.k; kx++)
            for (u32 j = 0; j < p.kt; j++) {
                const float* xf = j == 0 ? p.x0 : (j == 1 ? p.x1 : p.x2);
                const float* mx = j == 0 ? p.m0 : (j == 1 ? p.m1 : p.m2);
                float f = 1.f, inv = 1.f;
                if (xf) pow2_for(*mx, f, inv);
                int iy = oy + (int)ky, ix = ox + (int)kx;
                bool ok = tok_ok && xf && iy >= 0 && ix >= 0 && iy < ih && ix < iw;
                if (p.up) {
                    iy >>= 1;
                    ix >>= 1;
                }
                const float* xr = ok ? xf + ((u64)iy * p.wd + (u64)ix) * p.xstride + p.xoff + 16 * lh : nullptr;
                for (u32 c = 0; c < cb; c++, b++) {
                    const uint4* pw = (const uint4*)(wr + (u64)b * 16);
                    uint4 w0 = pw[0], w1 = pw[1];
                    if (ok) {
                        const float4* px = (const float4*)(xr + c * 32);
#pragma unroll
                        for (int i = 0; i < 4; i++) {
                            float4 v = px[i];
                            split_h2(v.x * f, v.y * f, ah[2 * i], al[2 * i]);
                            split_h2(v.z * f, v.w * f, ah[2 * i + 1], al[2 * i + 1]);
                        }
                    } else {
#pragma unroll
                        for (int i = 0; i < 8; i++) ah[i] = al[i] = 0u;
                    }
                    bs[0] = w0;
                    bs[1] = w1;
                    __syncthreads();
                    float st[2][4][4], stl[2][4][4];
#pragma unroll
                    for (int a = 0; a < 2; a++)
#pragma unroll
                        for (int bb = 0; bb < 4; bb++)
#pragma unroll
                            for (int q = 0; q < 4; q++) st[a][bb][q] = stl[a][bb][q] = 0.f;
#pragma unroll
                    for (int kk = 0; kk < 2; kk++) {
                        u32 af[2][4], lf[2][4], bf[4][2];
#pragma unroll
                        for (int mt = 0; mt < 2; mt++) {
                            ldmatrix4(af[mt], As + ra + 16 * mt * HLD + 8 * kk);
                            ldmatrix4(lf[mt], Al + ra + 16 * mt * HLD + 8 * kk);
                        }
#pragma unroll
                        for (int np = 0; np < 2; np++) {
                            u32 r[4];
                            ldmatrix4(r, lb + 16 * np * HLD + 8 * kk);
                            bf[2 * np][0] = r[0], bf[2 * np][1] = r[1], bf[2 * np + 1][0] = r[2], bf[2 * np + 1][1] = r[3];
                        }
#pragma unroll
                        for (int mt = 0; mt < 2; mt++)
#pragma unroll
                            for (int nt = 0; nt < 4; nt++) {
                                mma16816(st[mt][nt], af[mt], bf[nt]);
                                mma16816(stl[mt][nt], lf[mt], bf[nt]);
                            }
                    }
#pragma unroll
                    for (int a = 0; a < 2; a++)
#pragma unroll
                        for (int bb = 0; bb < 4; bb++)
#pragma unroll
                            for (int q = 0; q < 4; q++) tot[a][bb][q] += (st[a][bb][q] + stl[a][bb][q] * (1.f / 2048.f)) * inv;
                    __syncthreads();
                }
            }
    const u32 g = lane >> 2, c4 = lane & 3;
#pragma unroll
    for (int mt = 0; mt < 2; mt++)
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
#pragma unroll
            for (int q = 0; q < 4; q++) {
                u32 t = t0 + wm + 16 * mt + g + 8 * (q >> 1), cc = o0 + wn + 8 * nt + 2 * c4 + (q & 1);
                if (t >= p.npix || cc >= p.cout) continue;
                float v = tot[mt][nt][q] * p.winv + p.bias[cc];
                u64 at = (u64)t * p.cout + cc;
                if (p.add) v += p.add[at];
                p.y[at] = v;
            }
}
