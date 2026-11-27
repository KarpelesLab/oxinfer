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
