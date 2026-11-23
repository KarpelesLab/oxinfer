// Inkling-2 on CUDA (`inkling_cuda.rs`): whole passes of the decoder on the device. Compiled by
// NVRTC at run time after the i-quant tables (generated from `ggml_tables.rs`) and a `#define` per
// weight kind the model uses (`HAS_<id>`), so only those kinds' kernels are built.
//
// Weights are read as stored: every GGUF block format with a CPU dequantizer (except the ternary
// and 1-bit ones), NVFP4 in ggml's layout, FP8 E4M3 with block scales (the release's MXFP8
// experts), oxinfer's own Q8 / Q4, bf16, f16 and f32. `dq<K>(row, b, j)` gives elements
// `32 b + 4 j .. + 3` of a row (block `b`, part `j` of 8), unscaled by the matrix's block scales.
// Activations are f32 everywhere except in the GEMMs (f16 operands, f32 accumulation).

#define INFINITY __int_as_float(0x7f800000)
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

// A weight matrix (or a stack of experts, `es` bytes apart): rows of `rb` bytes. `s`: block scales
// (`(row / sr) * ncb + col / sc`, `ncb = ceil(n_in / sc)`, `ss` floats apart per expert) or null;
// `f`: a factor on every output; `ef`: per-expert factors or null.
struct W {
    const u8* q;
    const float* s;
    const float* ef;
    u64 rb;
    u64 es;
    u64 ss;
    float f;
    u32 sr;
    u32 sc;
    u32 ncb;
};

#define K_F32 0
#define K_F16 1
#define K_Q4_0 2
#define K_Q4_1 3
#define K_Q5_0 6
#define K_Q5_1 7
#define K_Q8_0 8
#define K_Q2_K 10
#define K_Q3_K 11
#define K_Q4_K 12
#define K_Q5_K 13
#define K_Q6_K 14
#define K_IQ2_XXS 16
#define K_IQ2_XS 17
#define K_IQ3_XXS 18
#define K_IQ1_S 19
#define K_IQ4_NL 20
#define K_IQ3_S 21
#define K_IQ2_S 22
#define K_IQ4_XS 23
#define K_IQ1_M 29
#define K_BF16 30
#define K_MXFP4 39
#define K_NVFP4 40
#define K_OQ8 200
#define K_OQ4 201
#define K_F8 202

__device__ __forceinline__ float h2f(u16 h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}
__device__ __forceinline__ float bf2f(u32 h) { return __uint_as_float(h << 16); }
__device__ __forceinline__ u32 rd16(const u8* p) { return *(const u16*)p; }
// 2-byte aligned 32-bit read.
__device__ __forceinline__ u32 rd32h(const u8* p) { return (u32)((const u16*)p)[0] | ((u32)((const u16*)p)[1] << 16); }
__device__ __forceinline__ u32 rd32(const u8* p) { return *(const u32*)p; }
__device__ __forceinline__ u32 rd32b(const u8* p) { return (u32)p[0] | ((u32)p[1] << 8) | ((u32)p[2] << 16) | ((u32)p[3] << 24); }
__device__ __forceinline__ u32 byte(u32 w, u32 i) { return (w >> (8 * i)) & 255u; }
__device__ __forceinline__ float sbyte(u32 w, u32 i) { return (float)(signed char)(u8)(w >> (8 * i)); }
__device__ __forceinline__ u32 ksign(u32 i) { return i | ((__popc(i) & 1) << 7); }
// Half the value of an E8M0 byte (MXFP4's values are stored doubled).
__device__ __forceinline__ float e8m0_half(u32 e) { return e < 2 ? __uint_as_float(0x00200000u << e) : __uint_as_float((e - 1) << 23); }
// Half the value of an unsigned E4M3 byte (NVFP4's; 0 and 0x7f are 0).
__device__ __forceinline__ float ue4m3_half(u32 x) {
    if (x == 0 || x == 0x7f) return 0.f;
    u32 e = (x >> 3) & 15, m = x & 7;
    float raw = e == 0 ? (float)m * 0.001953125f : (1.f + (float)m * 0.125f) * __uint_as_float((e + 120) << 23);
    return raw * 0.5f;
}
// E4M3 (sign, 4 exponent bits, 3 mantissa bits): shifted into a half's places it is the value
// times 2^-8, subnormals included.
__device__ __forceinline__ float e4m3(u32 v) { return h2f((u16)(((v & 0x80) << 8) | ((v & 0x7f) << 7))) * 256.f; }

// Scale and min `j` of a Q4_K / Q5_K block (12 bytes at `s`).
__device__ __forceinline__ void scale_min_k4(u32 j, const u8* s, float& sc, float& mn) {
    if (j < 4) {
        sc = (float)(s[j] & 63);
        mn = (float)(s[j + 4] & 63);
    } else {
        sc = (float)((s[j + 4] & 15) | ((s[j - 4] >> 6) << 4));
        mn = (float)((s[j + 4] >> 4) | ((s[j] >> 6) << 4));
    }
}

template <int K>
__device__ __forceinline__ float4 dq(const u8* row, u32 b, u32 j) {
    float4 o;
    if constexpr (K == K_F32) {
        o = ((const float4*)row)[8 * b + j];
    } else if constexpr (K == K_F16 || K == K_BF16) {
        uint2 u = ((const uint2*)row)[8 * b + j];
        if constexpr (K == K_F16) {
            o = make_float4(h2f(u.x & 0xffff), h2f(u.x >> 16), h2f(u.y & 0xffff), h2f(u.y >> 16));
        } else {
            o = make_float4(bf2f(u.x & 0xffff), bf2f(u.x >> 16), bf2f(u.y & 0xffff), bf2f(u.y >> 16));
        }
    } else if constexpr (K == K_Q8_0) {
        const u8* B = row + 34 * b;
        float d = h2f(rd16(B));
        u32 u = rd32h(B + 2 + 4 * j);
        o = make_float4(sbyte(u, 0) * d, sbyte(u, 1) * d, sbyte(u, 2) * d, sbyte(u, 3) * d);
    } else if constexpr (K == K_OQ8 || K == K_F8) {
        u32 u = rd32(row + 32 * b + 4 * j);
        if constexpr (K == K_OQ8) {
            o = make_float4(sbyte(u, 0), sbyte(u, 1), sbyte(u, 2), sbyte(u, 3));
        } else {
            o = make_float4(e4m3(byte(u, 0)), e4m3(byte(u, 1)), e4m3(byte(u, 2)), e4m3(byte(u, 3)));
        }
    } else if constexpr (K == K_Q4_0 || K == K_IQ4_NL || K == K_OQ4) {
        const u8* B = K == K_OQ4 ? row + 16 * b : row + 18 * b + 2;
        float d = K == K_OQ4 ? 1.f : h2f(rd16(B - 2));
        u32 u = (K == K_OQ4 ? rd32(B + 4 * (j & 3)) : rd32h(B + 4 * (j & 3))) >> (4 * (j >> 2));
        if constexpr (K == K_IQ4_NL) {
            o = make_float4(kvalues_iq4nl[u & 15] * d, kvalues_iq4nl[(u >> 8) & 15] * d, kvalues_iq4nl[(u >> 16) & 15] * d,
                            kvalues_iq4nl[(u >> 24) & 15] * d);
        } else {
            o = make_float4(((float)(u & 15) - 8.f) * d, ((float)((u >> 8) & 15) - 8.f) * d, ((float)((u >> 16) & 15) - 8.f) * d,
                            ((float)((u >> 24) & 15) - 8.f) * d);
        }
    } else if constexpr (K == K_Q4_1) {
        const u8* B = row + 20 * b;
        float d = h2f(rd16(B)), m = h2f(rd16(B + 2));
        u32 u = rd32(B + 4 + 4 * (j & 3)) >> (4 * (j >> 2));
        o = make_float4((float)(u & 15) * d + m, (float)((u >> 8) & 15) * d + m, (float)((u >> 16) & 15) * d + m,
                        (float)((u >> 24) & 15) * d + m);
    } else if constexpr (K == K_Q5_0 || K == K_Q5_1) {
        const u8* B = row + (K == K_Q5_0 ? 22 : 24) * b;
        float d = h2f(rd16(B)), m = K == K_Q5_1 ? h2f(rd16(B + 2)) : 0.f;
        u32 qh = K == K_Q5_0 ? rd32h(B + 2) : rd32(B + 4);
        const u8* q = B + (K == K_Q5_0 ? 6 : 8);
        u32 u = (K == K_Q5_0 ? rd32h(q + 4 * (j & 3)) : rd32(q + 4 * (j & 3))) >> (4 * (j >> 2));
        u32 e = 4 * j;
        float off = K == K_Q5_0 ? 16.f : 0.f;
        float v0 = (float)((u & 15) | (((qh >> e) & 1) << 4)) - off;
        float v1 = (float)(((u >> 8) & 15) | (((qh >> (e + 1)) & 1) << 4)) - off;
        float v2 = (float)(((u >> 16) & 15) | (((qh >> (e + 2)) & 1) << 4)) - off;
        float v3 = (float)(((u >> 24) & 15) | (((qh >> (e + 3)) & 1) << 4)) - off;
        o = make_float4(v0 * d + m, v1 * d + m, v2 * d + m, v3 * d + m);
    } else if constexpr (K == K_MXFP4) {
        const u8* B = row + 17 * b;
        float d = e8m0_half(B[0]);
        const u8* q = B + 1 + 4 * (j & 3);
        u32 sh = 4 * (j >> 2);
        o = make_float4(kvalues_mxfp4[(q[0] >> sh) & 15] * d, kvalues_mxfp4[(q[1] >> sh) & 15] * d, kvalues_mxfp4[(q[2] >> sh) & 15] * d,
                        kvalues_mxfp4[(q[3] >> sh) & 15] * d);
    } else if constexpr (K == K_NVFP4) {
        const u8* B = row + 36 * (b >> 1);
        u32 e0 = 32 * (b & 1) + 4 * j, s = e0 >> 4, w = e0 & 15;
        float d = ue4m3_half(B[s]);
        u32 u = rd32(B + 4 + 8 * s + (w & 7)) >> (w >= 8 ? 4 : 0);
        o = make_float4(kvalues_mxfp4[u & 15] * d, kvalues_mxfp4[(u >> 8) & 15] * d, kvalues_mxfp4[(u >> 16) & 15] * d,
                        kvalues_mxfp4[(u >> 24) & 15] * d);
    } else if constexpr (K == K_Q4_K || K == K_Q5_K) {
        const u8* B = row + (K == K_Q4_K ? 144 : 176) * (b >> 3);
        u32 sub = b & 7;
        float sc, mn;
        scale_min_k4(sub, B + 4, sc, mn);
        float d1 = h2f(rd16(B)) * sc, m1 = h2f(rd16(B + 2)) * mn;
        u32 u = rd32(B + (K == K_Q4_K ? 16 : 48) + 32 * (sub >> 1) + 4 * j) >> (4 * (sub & 1));
        u32 hb = 0;
        if constexpr (K == K_Q5_K) hb = rd32(B + 16 + 4 * j) >> sub;
        o = make_float4((float)((u & 15) | ((hb & 1) << 4)) * d1 - m1, (float)(((u >> 8) & 15) | (((hb >> 8) & 1) << 4)) * d1 - m1,
                        (float)(((u >> 16) & 15) | (((hb >> 16) & 1) << 4)) * d1 - m1,
                        (float)(((u >> 24) & 15) | (((hb >> 24) & 1) << 4)) * d1 - m1);
    } else if constexpr (K == K_Q6_K) {
        const u8* B = row + 210 * (b >> 3);
        u32 sub = b & 7, n = sub >> 2, qq = sub & 3;
        float d = h2f(rd16(B + 208)) * (float)(signed char)B[192 + 8 * n + (j >> 2) + 2 * qq];
        u32 ul = rd32h(B + 64 * n + 32 * (qq & 1) + 4 * j) >> (4 * (qq >> 1));
        u32 uh = rd32h(B + 128 + 32 * n + 4 * j) >> (2 * qq);
        o = make_float4((float)(int)((ul & 15) | ((uh & 3) << 4)) - 32.f, (float)(int)(((ul >> 8) & 15) | (((uh >> 8) & 3) << 4)) - 32.f,
                        (float)(int)(((ul >> 16) & 15) | (((uh >> 16) & 3) << 4)) - 32.f,
                        (float)(int)(((ul >> 24) & 15) | (((uh >> 24) & 3) << 4)) - 32.f);
        o.x *= d;
        o.y *= d;
        o.z *= d;
        o.w *= d;
    } else if constexpr (K == K_Q3_K) {
        const u8* B = row + 110 * (b >> 3);
        u32 sub = b & 7, n = sub >> 2, jj = sub & 3;
        u32 k = 8 * n + 2 * jj + (j >> 2);
        u32 a0 = rd32h(B + 96), a1 = rd32h(B + 100), tmp = rd32h(B + 104);
        const u32 m1 = 0x03030303u, m2 = 0x0f0f0f0fu;
        u32 aux;
        switch (k >> 2) {
            case 0: aux = (a0 & m2) | ((tmp & m1) << 4); break;
            case 1: aux = (a1 & m2) | (((tmp >> 2) & m1) << 4); break;
            case 2: aux = ((a0 >> 4) & m2) | (((tmp >> 4) & m1) << 4); break;
            default: aux = ((a1 >> 4) & m2) | (((tmp >> 6) & m1) << 4); break;
        }
        float dl = h2f(rd16(B + 108)) * (float)((int)(signed char)(u8)(aux >> (8 * (k & 3))) - 32);
        u32 u = rd32h(B + 32 + 32 * n + 4 * j) >> (2 * jj);
        u32 h = rd32h(B + 4 * j) >> (4 * n + jj);
        o = make_float4(((float)(u & 3) - ((h & 1) ? 0.f : 4.f)) * dl, ((float)((u >> 8) & 3) - (((h >> 8) & 1) ? 0.f : 4.f)) * dl,
                        ((float)((u >> 16) & 3) - (((h >> 16) & 1) ? 0.f : 4.f)) * dl,
                        ((float)((u >> 24) & 3) - (((h >> 24) & 1) ? 0.f : 4.f)) * dl);
    } else if constexpr (K == K_Q2_K) {
        const u8* B = row + 84 * (b >> 3);
        u32 sub = b & 7, n = sub >> 2, jj = sub & 3;
        u32 s = B[8 * n + 2 * jj + (j >> 2)];
        float dl = h2f(rd16(B + 80)) * (float)(s & 15), ml = h2f(rd16(B + 82)) * (float)(s >> 4);
        u32 u = rd32(B + 16 + 32 * n + 4 * j) >> (2 * jj);
        o = make_float4((float)(u & 3) * dl - ml, (float)((u >> 8) & 3) * dl - ml, (float)((u >> 16) & 3) * dl - ml,
                        (float)((u >> 24) & 3) * dl - ml);
    } else if constexpr (K == K_IQ4_XS) {
        const u8* B = row + 136 * (b >> 3);
        u32 ib = b & 7;
        int ls = (int)(((B[4 + ib / 2] >> (4 * (ib % 2))) & 15) | (((rd16(B + 2) >> (2 * ib)) & 3) << 4)) - 32;
        float d = h2f(rd16(B)) * (float)ls;
        u32 u = rd32(B + 8 + 16 * ib + 4 * (j & 3)) >> (4 * (j >> 2));
        o = make_float4(kvalues_iq4nl[u & 15] * d, kvalues_iq4nl[(u >> 8) & 15] * d, kvalues_iq4nl[(u >> 16) & 15] * d,
                        kvalues_iq4nl[(u >> 24) & 15] * d);
    } else if constexpr (K == K_IQ2_XXS || K == K_IQ2_XS || K == K_IQ2_S) {
        u32 ib = b & 7, l = j >> 1, j0 = 4 * (j & 1);
        u64 g;
        u32 signs;
        float db;
        if constexpr (K == K_IQ2_XXS) {
            const u8* B = row + 66 * (b >> 3);
            u32 a0 = rd32h(B + 2 + 8 * ib), a1 = rd32h(B + 6 + 8 * ib);
            db = h2f(rd16(B)) * (0.5f + (float)(a1 >> 28)) * 0.25f;
            g = iq2xxs_grid[(a0 >> (8 * l)) & 255];
            signs = ksign((a1 >> (7 * l)) & 127);
        } else if constexpr (K == K_IQ2_XS) {
            const u8* B = row + 74 * (b >> 3);
            u32 s = B[66 + ib];
            db = h2f(rd16(B)) * (0.5f + (float)(l < 2 ? (s & 15) : (s >> 4))) * 0.25f;
            u32 q = rd16(B + 2 + 2 * (4 * ib + l));
            g = iq2xs_grid[q & 511];
            signs = ksign(q >> 9);
        } else {
            const u8* B = row + 82 * (b >> 3);
            u32 s = B[74 + ib];
            db = h2f(rd16(B)) * (0.5f + (float)(l < 2 ? (s & 15) : (s >> 4))) * 0.25f;
            u32 idx = (u32)B[2 + 4 * ib + l] | (((u32)B[66 + ib] << (8 - 2 * l)) & 0x300);
            g = iq2s_grid[idx];
            signs = B[34 + 4 * ib + l];
        }
        u32 gw = (u32)(g >> (8 * j0));
        o = make_float4(db * (float)byte(gw, 0) * ((signs >> j0) & 1 ? -1.f : 1.f), db * (float)byte(gw, 1) * ((signs >> (j0 + 1)) & 1 ? -1.f : 1.f),
                        db * (float)byte(gw, 2) * ((signs >> (j0 + 2)) & 1 ? -1.f : 1.f),
                        db * (float)byte(gw, 3) * ((signs >> (j0 + 3)) & 1 ? -1.f : 1.f));
    } else if constexpr (K == K_IQ3_XXS || K == K_IQ3_S) {
        u32 ib = b & 7, l = j >> 1, hf = j & 1;
        u32 gw, signs;
        float db;
        if constexpr (K == K_IQ3_XXS) {
            const u8* B = row + 98 * (b >> 3);
            u32 aux = rd32h(B + 66 + 4 * ib);
            db = h2f(rd16(B)) * (0.5f + (float)(aux >> 28)) * 0.5f;
            signs = ksign((aux >> (7 * l)) & 127);
            gw = iq3xxs_grid[B[2 + 8 * ib + 2 * l + hf]];
        } else {
            const u8* B = row + 110 * (b >> 3);
            u32 s = B[106 + ib / 2] >> (4 * (ib % 2));
            db = h2f(rd16(B)) * (float)(1 + 2 * (s & 15));
            u32 h = B[66 + ib];
            u32 idx = (u32)B[2 + 8 * ib + 2 * l + hf] | ((h << (8 - 2 * l - hf)) & 256);
            gw = iq3s_grid[idx];
            signs = B[74 + 4 * ib + l];
        }
        u32 s0 = 4 * hf;
        o = make_float4(db * (float)byte(gw, 0) * ((signs >> s0) & 1 ? -1.f : 1.f), db * (float)byte(gw, 1) * ((signs >> (s0 + 1)) & 1 ? -1.f : 1.f),
                        db * (float)byte(gw, 2) * ((signs >> (s0 + 2)) & 1 ? -1.f : 1.f),
                        db * (float)byte(gw, 3) * ((signs >> (s0 + 3)) & 1 ? -1.f : 1.f));
    } else if constexpr (K == K_IQ1_S || K == K_IQ1_M) {
        u32 ib = b & 7, l = j >> 1, j0 = 4 * (j & 1);
        float dl, delta;
        u64 g;
        if constexpr (K == K_IQ1_S) {
            const u8* B = row + 50 * (b >> 3);
            u32 qh = rd16(B + 34 + 2 * ib);
            dl = h2f(rd16(B)) * (float)(2 * ((qh >> 12) & 7) + 1);
            delta = (qh & 0x8000) ? -0.125f : 0.125f;
            g = iq1s_grid[(u32)B[2 + 4 * ib + l] | (((qh >> (3 * l)) & 7) << 8)];
        } else {
            const u8* B = row + 56 * (b >> 3);
            u32 s0 = rd16(B + 48), s1 = rd16(B + 50), s2 = rd16(B + 52), s3 = rd16(B + 54);
            float d = h2f((u16)((s0 >> 12) | ((s1 >> 8) & 0x00f0) | ((s2 >> 4) & 0x0f00) | (s3 & 0xf000)));
            u32 sv = rd16(B + 48 + 2 * (ib / 2)) >> (6 * (ib % 2));
            dl = d * (float)(2 * ((sv >> (l < 2 ? 0 : 3)) & 7) + 1);
            u32 qh = B[32 + 2 * ib + l / 2];
            u32 idx = (u32)B[4 * ib + l] | ((qh << (l % 2 ? 4 : 8)) & 0x700);
            delta = (qh & (l % 2 ? 0x80 : 0x08)) ? -0.125f : 0.125f;
            g = iq1s_grid[idx];
        }
        u32 gw = (u32)(g >> (8 * j0));
        o = make_float4(dl * (sbyte(gw, 0) + delta), dl * (sbyte(gw, 1) + delta), dl * (sbyte(gw, 2) + delta), dl * (sbyte(gw, 3) + delta));
    }
    return o;
}

// Kinds whose values are multiplied by per-block scales from `W::s`.
#define BLOCK_SCALED(K) ((K) == K_F8 || (K) == K_OQ8 || (K) == K_OQ4)

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// One row's dot product with `x` by a warp (8 lanes per block of 32, 4 values per lane), summed
// over the warp (every lane gets it). `s`: this matrix's (or expert's) block scales.
template <int K>
__device__ __forceinline__ float rowdot(const W& w, const u8* row, const float* s, u32 r, const float* x, u32 nb, u32 lane) {
    float acc = 0.f;
    u32 j = lane & 7, b = lane >> 3;
    // Four blocks in flight per lane (their loads issued together), then the rest.
    for (; b + 12 < nb; b += 16) {
        float4 v[4], xv[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            v[i] = dq<K>(row, b + 4 * i, j);
            xv[i] = *(const float4*)(x + 32 * (b + 4 * i) + 4 * j);
        }
#pragma unroll
        for (int i = 0; i < 4; i++) {
            float p = v[i].x * xv[i].x + v[i].y * xv[i].y + v[i].z * xv[i].z + v[i].w * xv[i].w;
            if constexpr (BLOCK_SCALED(K)) p *= s[(r / w.sr) * w.ncb + (32 * (b + 4 * i)) / w.sc];
            acc += p;
        }
    }
    for (; b < nb; b += 4) {
        float4 v = dq<K>(row, b, j);
        float4 xv = *(const float4*)(x + 32 * b + 4 * j);
        float p = v.x * xv.x + v.y * xv.y + v.z * xv.z + v.w * xv.w;
        if constexpr (BLOCK_SCALED(K)) p *= s[(r / w.sr) * w.ncb + (32 * b) / w.sc];
        acc += p;
    }
    return warp_sum(acc);
}

// Two rows of one matrix against the same x, interleaved (twice the loads in flight).
template <int K>
__device__ __forceinline__ float2 rowdot2(const W& w, const u8* ra, const u8* rb2, const float* s, u32 r1, u32 r2, const float* x, u32 nb, u32 lane) {
    float a1 = 0.f, a2 = 0.f;
    u32 j = lane & 7, b = lane >> 3;
    for (; b + 4 < nb; b += 8) {
        float4 v[4], xv[2];
#pragma unroll
        for (int i = 0; i < 2; i++) {
            v[2 * i] = dq<K>(ra, b + 4 * i, j);
            v[2 * i + 1] = dq<K>(rb2, b + 4 * i, j);
            xv[i] = *(const float4*)(x + 32 * (b + 4 * i) + 4 * j);
        }
#pragma unroll
        for (int i = 0; i < 2; i++) {
            float p1 = v[2 * i].x * xv[i].x + v[2 * i].y * xv[i].y + v[2 * i].z * xv[i].z + v[2 * i].w * xv[i].w;
            float p2 = v[2 * i + 1].x * xv[i].x + v[2 * i + 1].y * xv[i].y + v[2 * i + 1].z * xv[i].z + v[2 * i + 1].w * xv[i].w;
            if constexpr (BLOCK_SCALED(K)) {
                p1 *= s[(r1 / w.sr) * w.ncb + (32 * (b + 4 * i)) / w.sc];
                p2 *= s[(r2 / w.sr) * w.ncb + (32 * (b + 4 * i)) / w.sc];
            }
            a1 += p1;
            a2 += p2;
        }
    }
    for (; b < nb; b += 4) {
        float4 v1 = dq<K>(ra, b, j), v2 = dq<K>(rb2, b, j);
        float4 xv = *(const float4*)(x + 32 * b + 4 * j);
        float p1 = v1.x * xv.x + v1.y * xv.y + v1.z * xv.z + v1.w * xv.w;
        float p2 = v2.x * xv.x + v2.y * xv.y + v2.z * xv.z + v2.w * xv.w;
        if constexpr (BLOCK_SCALED(K)) {
            p1 *= s[(r1 / w.sr) * w.ncb + (32 * b) / w.sc];
            p2 *= s[(r2 / w.sr) * w.ncb + (32 * b) / w.sc];
        }
        a1 += p1;
        a2 += p2;
    }
    return make_float2(warp_sum(a1), warp_sum(a2));
}

// y[tok][r] (= or +=) W[r] . x[tok]: 8 rows per 256-thread block, grid.y tokens.
// `MV_ROWS` (N; the launch then takes N rows per warp of its block): a warp takes N consecutive rows and a
// lane whole blocks of 32 values, so each block of `x` is loaded once for the N rows and what a
// block's values share (its scale, the superblock's header) is decoded once per 32 values, not
// once per 4. For devices whose matvecs are bound by the loads and the arithmetic and not by the
// memory (the GB10 of a DGX Spark, `cuda.rs`); the sums are taken in another order.
template <int K>
__device__ __forceinline__ void mv_body(const W& w, const float* x, float* y, u32 n_in, u32 n_out, u32 xs, u32 ys, u32 mode) {
    u32 warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
#ifdef MV_ROWS
    u32 r0 = (blockIdx.x * (blockDim.x >> 5) + warp) * MV_ROWS;
    if (r0 >= n_out) return;
    const float* xt = x + (u64)blockIdx.y * xs;
    float a[MV_ROWS];
#pragma unroll
    for (int i = 0; i < MV_ROWS; i++) a[i] = 0.f;
    for (u32 b = lane; b < n_in / 32; b += 32) {
        float4 xv[8];
#pragma unroll
        for (int j = 0; j < 8; j++) xv[j] = *(const float4*)(xt + 32 * b + 4 * j);
#pragma unroll
        for (int i = 0; i < MV_ROWS; i++) {
            u32 row = min(r0 + i, n_out - 1);
            const u8* rp = w.q + (u64)row * w.rb;
            float p = 0.f;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                float4 v = dq<K>(rp, b, j);
                p += v.x * xv[j].x + v.y * xv[j].y + v.z * xv[j].z + v.w * xv[j].w;
            }
            if constexpr (BLOCK_SCALED(K)) p *= w.s[(row / w.sr) * w.ncb + (32 * b) / w.sc];
            a[i] += p;
        }
    }
#pragma unroll
    for (int i = 0; i < MV_ROWS; i++) {
        float v = warp_sum(a[i]) * w.f;
        if (lane == 0 && r0 + i < n_out) {
            float* o = y + (u64)blockIdx.y * ys + r0 + i;
            *o = mode ? *o + v : v;
        }
    }
    return;
#endif
    u32 r = blockIdx.x * (blockDim.x >> 5) + warp;
    if (r >= n_out) return;
    const float* xr = x + (u64)blockIdx.y * xs;
    float acc = rowdot<K>(w, w.q + (u64)r * w.rb, w.s, r, xr, n_in / 32, lane) * w.f;
    if (lane == 0) {
        float* o = y + (u64)blockIdx.y * ys + r;
        *o = mode ? *o + acc : acc;
    }
}

// A gate / up pair (the language models' plans): y[tok][r] = act(W[r] . x) * (W2[r] . x), act SiLU
// (gelu 0) or tanh GELU (1), the two matrices in the same format. As `mv_body` with `MV_ROWS`
// (a lane per block of 32), the warp taking MVP_ROWS rows of both: each block of x is loaded once
// for the 2 MVP_ROWS rows.
#ifndef MVP_ROWS
#ifdef MV_ROWS
#define MVP_ROWS MV_ROWS
#else
#define MVP_ROWS 1
#endif
#endif
template <int K>
__device__ __forceinline__ void mvp_body(const W& w, const W& w2, const float* x, float* y, u32 n_in, u32 n_out, u32 xs, u32 ys, u32 gelu) {
    u32 warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    u32 r0 = (blockIdx.x * (blockDim.x >> 5) + warp) * MVP_ROWS;
    if (r0 >= n_out) return;
    const float* xt = x + (u64)blockIdx.y * xs;
    float a[2][MVP_ROWS];
#pragma unroll
    for (int i = 0; i < MVP_ROWS; i++) a[0][i] = a[1][i] = 0.f;
    for (u32 b = lane; b < n_in / 32; b += 32) {
        float4 xv[8];
#pragma unroll
        for (int j = 0; j < 8; j++) xv[j] = *(const float4*)(xt + 32 * b + 4 * j);
#pragma unroll
        for (int m = 0; m < 2; m++) {
            const W& ww = m ? w2 : w;
#pragma unroll
            for (int i = 0; i < MVP_ROWS; i++) {
                u32 row = min(r0 + i, n_out - 1);
                const u8* rp = ww.q + (u64)row * ww.rb;
                float p = 0.f;
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    float4 v = dq<K>(rp, b, j);
                    p += v.x * xv[j].x + v.y * xv[j].y + v.z * xv[j].z + v.w * xv[j].w;
                }
                if constexpr (BLOCK_SCALED(K)) p *= ww.s[(row / ww.sr) * ww.ncb + (32 * b) / ww.sc];
                a[m][i] += p;
            }
        }
    }
#pragma unroll
    for (int i = 0; i < MVP_ROWS; i++) {
        float g = warp_sum(a[0][i]) * w.f, u = warp_sum(a[1][i]) * w2.f;
        if (lane == 0 && r0 + i < n_out) {
            float act;
            if (gelu) {
                float t = 0.7978845608f * (g + 0.044715f * g * g * g);
                act = 0.5f * g * (1.0f + tanhf(t));
            } else {
                act = g / (1.0f + expf(-g));
            }
            y[(u64)blockIdx.y * ys + r0 + i] = act * u;
        }
    }
}

__device__ __forceinline__ float block_sum2(float v, float* red) {
    v = warp_sum(v);
    u32 warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float s = 0.f;
    for (u32 i = 0; i < nw; i++) s += red[i];
    return s;
}

// A message gate's low-rank input, d[tok][row] = down[row] . [rms(x); rms(m)]: a block per
// (row, token), the norms recomputed by each.
template <int K>
__device__ __forceinline__ void gate_down_body(const W& w, const float* x, const float* m, float* d, u32 h, u32 r, float eps) {
    __shared__ float red[32];
    u32 row = blockIdx.x, tok = blockIdx.y;
    const float* xr = x + (u64)tok * h;
    const float* mr = m + (u64)tok * h;
    float sx = 0.f, sm = 0.f;
    for (u32 i = threadIdx.x; i < h; i += blockDim.x) {
        sx += xr[i] * xr[i];
        sm += mr[i] * mr[i];
    }
    float rx = rsqrtf(block_sum2(sx, red) / (float)h + eps);
    float rm = rsqrtf(block_sum2(sm, red) / (float)h + eps);
    const u8* wr = w.q + (u64)row * w.rb;
    float a = 0.f, bm = 0.f;
    for (u32 idx = threadIdx.x; idx < h / 2; idx += blockDim.x) {
        u32 b = idx >> 3, j = idx & 7, c = 32 * b + 4 * j;
        float4 v = dq<K>(wr, b, j);
        float4 xv = c < h ? *(const float4*)(xr + c) : *(const float4*)(mr + c - h);
        float p = v.x * xv.x + v.y * xv.y + v.z * xv.z + v.w * xv.w;
        if constexpr (BLOCK_SCALED(K)) p *= w.s[(row / w.sr) * w.ncb + c / w.sc];
        if (c < h)
            a += p;
        else
            bm += p;
    }
    float tot = block_sum2(a, red) * rx;
    tot += block_sum2(bm, red) * rm;
    if (threadIdx.x == 0) d[(u64)tok * r + row] = tot * w.f;
}

// The same for prefill: a block (8 warps) per token, the norms computed once, each warp taking
// every 8th row.
template <int K>
__device__ __forceinline__ void gate_down_t_body(const W& w, const float* x, const float* m, float* d, u32 h, u32 r, float eps) {
    __shared__ float red[32];
    __shared__ float cat[4608];
    u32 tok = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const float* xr = x + (u64)tok * h;
    const float* mr = m + (u64)tok * h;
    float sx = 0.f, sm = 0.f;
    for (u32 i = threadIdx.x; i < h; i += blockDim.x) {
        sx += xr[i] * xr[i];
        sm += mr[i] * mr[i];
    }
    float rx = rsqrtf(block_sum2(sx, red) / (float)h + eps);
    float rm = rsqrtf(block_sum2(sm, red) / (float)h + eps);
    for (u32 i = threadIdx.x; i < h; i += blockDim.x) {
        cat[i] = xr[i] * rx;
        cat[h + i] = mr[i] * rm;
    }
    __syncthreads();
    for (u32 row = warp; row < r; row += 8) {
        float v = rowdot<K>(w, w.q + (u64)row * w.rb, w.s, row, cat, 2 * h / 32, lane);
        if (lane == 0) d[(u64)tok * r + row] = v * w.f;
    }
}

// The gate itself: out = (1 - tanh(forget d)) x + tanh(gain d) m, a warp per output.
__device__ __forceinline__ float tanh_acc(float x);
template <int K>
__device__ __forceinline__ void gate_out_body(const W& wf, const W& wg, const float* d, const float* x, const float* m, float* out, u32 h, u32 r) {
    u32 warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    u32 j = blockIdx.x * 8 + warp, tok = blockIdx.y;
    if (j >= h) return;
    const float* dr = d + (u64)tok * r;
    float f = rowdot<K>(wf, wf.q + (u64)j * wf.rb, wf.s, j, dr, r / 32, lane) * wf.f;
    float g = rowdot<K>(wg, wg.q + (u64)j * wg.rb, wg.s, j, dr, r / 32, lane) * wg.f;
    u64 at = (u64)tok * h + j;
    if (lane == 0) out[at] = (1.f - tanh_acc(f)) * x[at] + tanh_acc(g) * m[at];
}

// Up to 8 matvecs over one input in one launch (decode: the projections of a layer's input).
// Job i covers blocks b0[i].. of grid.x.
struct MvJob {
    W w;
    float* y;
    u32 n_out, ys, b0, _p;
};
struct MvJobs {
    MvJob j[8];
    const float* x;
    u32 n, n_in, xs, _p;
};
template <int K>
__device__ __forceinline__ void mvm_body(const MvJobs& p) {
    u32 jb = 0;
    for (u32 i = 1; i < p.n; i++)
        if (blockIdx.x >= p.j[i].b0) jb = i;
    const MvJob& J = p.j[jb];
    u32 warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    u32 r = (blockIdx.x - J.b0) * 8 + warp;
    if (r >= J.n_out) return;
    const float* xr = p.x + (u64)blockIdx.y * p.xs;
    float acc = rowdot<K>(J.w, J.w.q + (u64)r * J.w.rb, J.w.s, r, xr, p.n_in / 32, lane) * J.w.f;
    if (lane == 0) J.y[(u64)blockIdx.y * J.ys + r] = acc;
}

// tanh to a few 1e-7 (with --use_fast_math, tanhf is the hardware approximation, ~5e-4).
__device__ __forceinline__ float tanh_acc(float x) {
    float a = fabsf(x);
    if (a < 0.02f) return x * (1.f - x * x * (1.f / 3.f));
    float e = __expf(-2.f * a);
    float t = (1.f - e) / (1.f + e);
    return copysignf(t, x);
}

__device__ __forceinline__ float situ(float g, float u, float beta, float lbeta) {
    float uu = lbeta > 0.f ? lbeta * tanh_acc(u / lbeta) : u;
    return beta * tanh_acc(g / beta) / (1.f + __expf(-g)) * uu;
}

// MoE gate/up for the chosen experts: grid (mi / 8, tokens * k); mid[slot][j] = act(gate, up).
template <int K>
__device__ __forceinline__ void moe_gu_body(const W& w, const int* ids, const float* x, float* mid, u32 n_in, u32 mi, u32 k, u32 xs,
                                            u32 inter, float beta, float lbeta) {
    u32 warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    u32 jr = blockIdx.x * 8 + warp;
    if (jr >= mi) return;
    u32 slot = blockIdx.y, tok = slot / k;
    u32 e = ids[slot];
    const u8* base = w.q + (u64)e * w.es;
    const float* s = w.s ? w.s + (u64)e * w.ss : w.s;
    float f = w.f * (w.ef ? w.ef[e] : 1.f);
    u32 rg = inter ? 2 * jr : jr, ru = inter ? 2 * jr + 1 : mi + jr;
    const float* xr = x + (u64)tok * xs;
    float2 gu = rowdot2<K>(w, base + (u64)rg * w.rb, base + (u64)ru * w.rb, s, rg, ru, xr, n_in / 32, lane);
    if (lane == 0) mid[(u64)slot * mi + jr] = situ(gu.x * f, gu.y * f, beta, lbeta);
}

// MoE down for the chosen experts, summed with their weights: grid (h / 2, tokens); a block's
// 8 warps take 2 rows, 4 warps per row each with every 4th expert, summed in a fixed order.
template <int K>
__device__ __forceinline__ void moe_dn_body(const W& w, const int* ids, const float* wts, const float* mid, float* y, u32 h, u32 mi, u32 k, u32 ys) {
    __shared__ float part[8];
    u32 warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    u32 r = blockIdx.x * 2 + (warp >> 2), q = warp & 3;
    u32 tok = blockIdx.y;
    float acc = 0.f;
    if (r < h) {
        for (u32 sl = q; sl < k; sl += 4) {
            u32 slot = tok * k + sl, e = ids[slot];
            const float* s = w.s ? w.s + (u64)e * w.ss : w.s;
            float f = w.f * (w.ef ? w.ef[e] : 1.f);
            acc += wts[slot] * f * rowdot<K>(w, w.q + (u64)e * w.es + (u64)r * w.rb, s, r, mid + (u64)slot * mi, mi / 32, lane);
        }
    }
    if (lane == 0) part[warp] = acc;
    __syncthreads();
    if (threadIdx.x < 2) {
        u32 rr = blockIdx.x * 2 + threadIdx.x;
        if (rr < h) {
            const float* pp = part + 4 * threadIdx.x;
            y[(u64)tok * ys + rr] = (pp[0] + pp[1]) + (pp[2] + pp[3]);
        }
    }
}

// ---- GEMM on the tensor cores (prefill): Y[m][n] = sum_k X[m][k] W[n][k], f16 operands, f32
// accumulation. A 128-thread block computes 64 rows (tokens) x 64 outputs; per 32-wide K step the
// weights are dequantized into shared memory as half. Rows come from a tile table (MoE segments:
// expert, first row, rows) or, without one, 64 consecutive rows per blockIdx.y. `gidx` maps a
// row to the row of X it reads (MoE: the slot's token), or null.
// Epilogues: 0 store, 1 add, 2 act pairs (the 64 columns are gate rows j0..j0+31 and their up rows;
// writes mid[row][j0 + c]), 3 store times wrow[row].

struct Tile {
    u32 e, r0, n;
};

__device__ __forceinline__ u32 pack_h2(float a, float b) {
    u32 r;
    asm("cvt.rn.f16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(a), "f"(b));
    return r;
}

// (hi, lo) f16 pairs of two floats: hi = f16(v), lo = f16((v - hi) 2^11), so hi + lo 2^-11
// carries ~22 bits; the scale keeps lo out of f16's subnormals (the products of lo terms are
// summed apart and scaled back).
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

#define GBM 64
#define GBN 64
#define GLD 20 // u32 (half2) per shared row: 16 + 4 padding

struct GemmP {
    const float* x;
    float* y;
    const int* gidx;
    const Tile* tiles;
    const u32* ntiles;
    const float* wrow;
    // X3: x already split (`split_rows`): hi and lo half2 rows of n_in / 2, or null.
    const u32* xh;
    const u32* xl;
    u32 n_in, n_out, rows, xs, ys, mode, mi, inter;
    float beta, lbeta;
};

// X3: each operand as a (hi, lo) pair of halves and three products (hi hi, lo hi, hi lo): the
// products to ~1e-6 relative instead of ~5e-4 (`gpu::exact`).
template <int K, bool X3>
__device__ __forceinline__ void gemm_body(const W& w, const GemmP& p) {
    // The operand tiles (hi, and lo for X3), and the 64 x 65 outputs of the act epilogue.
    __shared__ u32 smem[X3 ? 4 * GBM * GLD : GBM * 65];
    u32* As = smem;
    u32* Bs = smem + GBM * GLD;
    u32* Al = smem + 2 * GBM * GLD;
    u32* Bl = smem + 3 * GBM * GLD;
    u32 tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    u32 e = 0, r0, nr;
    if (p.tiles) {
        if (blockIdx.y >= *p.ntiles) return;
        Tile tl = p.tiles[blockIdx.y];
        e = tl.e;
        r0 = tl.r0;
        nr = tl.n;
    } else {
        r0 = blockIdx.y * GBM;
        if (r0 >= p.rows) return;
        nr = min(GBM, p.rows - r0);
    }
    const u8* base = w.q + (u64)e * w.es;
    const float* sbase = w.s ? w.s + (u64)e * w.ss : w.s;
    float f = w.f * (w.ef ? w.ef[e] : 1.f);
    u32 n0 = blockIdx.x * (p.mode == 2 ? 32 : GBN);
    // This thread's loads: X rows (tid / 2, 16 values from (tid % 2) * 16) and weight rows
    // (tid / 2, parts (tid % 2) * 4 .. + 3).
    u32 lr = tid >> 1, lh = tid & 1;
    const float* xrow = nullptr;
    const u32 *xhr = nullptr, *xlr = nullptr;
    if (lr < nr) {
        u32 gr = r0 + lr;
        u32 src = p.gidx ? (u32)p.gidx[gr] : gr;
        xrow = p.x + (u64)src * p.xs;
        if (X3 && p.xh) {
            xhr = p.xh + (u64)src * (p.n_in / 2);
            xlr = p.xl + (u64)src * (p.n_in / 2);
        }
    }
    u32 wr;
    bool wok;
    if (p.mode == 2) {
        u32 jj = n0 + (lr & 31);
        wok = jj < p.mi;
        wr = lr < 32 ? (p.inter ? 2 * jj : jj) : (p.inter ? 2 * jj + 1 : p.mi + jj);
    } else {
        wr = n0 + lr;
        wok = wr < p.n_out;
    }
    const u8* wrow = base + (u64)wr * w.rb;
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
    for (u32 b = 0; b < nb; b++) {
        // X: 16 floats -> 8 half2.
        if (X3 && xhr) {
            const uint4* ph = (const uint4*)(xhr + 16 * b + 8 * lh);
            const uint4* pl = (const uint4*)(xlr + 16 * b + 8 * lh);
            uint4 h0 = ph[0], h1 = ph[1], l0 = pl[0], l1 = pl[1];
            u32* ah = As + lr * GLD + 8 * lh;
            u32* al2 = Al + lr * GLD + 8 * lh;
            ah[0] = h0.x, ah[1] = h0.y, ah[2] = h0.z, ah[3] = h0.w, ah[4] = h1.x, ah[5] = h1.y, ah[6] = h1.z, ah[7] = h1.w;
            al2[0] = l0.x, al2[1] = l0.y, al2[2] = l0.z, al2[3] = l0.w, al2[4] = l1.x, al2[5] = l1.y, al2[6] = l1.z, al2[7] = l1.w;
        } else if (xrow) {
            const float4* xp = (const float4*)(xrow + 32 * b + 16 * lh);
#pragma unroll
            for (int i = 0; i < 4; i++) {
                float4 v = xp[i];
                u32 at = lr * GLD + 8 * lh + 2 * i;
                if constexpr (X3) {
                    split_h2(v.x, v.y, As[at], Al[at]);
                    split_h2(v.z, v.w, As[at + 1], Al[at + 1]);
                } else {
                    As[at] = pack_h2(v.x, v.y);
                    As[at + 1] = pack_h2(v.z, v.w);
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < 8; i++) {
                As[lr * GLD + 8 * lh + i] = 0u;
                if constexpr (X3) Al[lr * GLD + 8 * lh + i] = 0u;
            }
        }
        if (wok) {
            float sc = 1.f;
            if constexpr (BLOCK_SCALED(K)) sc = sbase[(wr / w.sr) * w.ncb + (32 * b) / w.sc];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                float4 v = dq<K>(wrow, b, 4 * lh + i);
                u32 at = lr * GLD + 8 * lh + 2 * i;
                if constexpr (X3) {
                    split_h2(v.x * sc, v.y * sc, Bs[at], Bl[at]);
                    split_h2(v.z * sc, v.w * sc, Bs[at + 1], Bl[at + 1]);
                } else {
                    Bs[at] = pack_h2(v.x * sc, v.y * sc);
                    Bs[at + 1] = pack_h2(v.z * sc, v.w * sc);
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < 8; i++) {
                Bs[lr * GLD + 8 * lh + i] = 0u;
                if constexpr (X3) Bl[lr * GLD + 8 * lh + i] = 0u;
            }
        }
        __syncthreads();
        // X3: this K step's products in fresh accumulators, added to the running sums with f32
        // adds (the tensor cores' accumulation truncates).
        float st[X3 ? 2 : 1][4][4], stl[X3 ? 2 : 1][4][4];
        if constexpr (X3) {
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
                    if constexpr (X3)
                        mma16816(st[mt][nt], af[mt], bf[nt]);
                    else
                        mma16816(acc[mt][nt], af[mt], bf[nt]);
                }
            if constexpr (X3) {
                u32 al[2][4], bl[4][2];
#pragma unroll
                for (int mt = 0; mt < 2; mt++) {
                    u32 rb = (wm + 16 * mt + g) * GLD + kb + c4;
                    al[mt][0] = Al[rb];
                    al[mt][1] = Al[rb + 8 * GLD];
                    al[mt][2] = Al[rb + 4];
                    al[mt][3] = Al[rb + 8 * GLD + 4];
                }
#pragma unroll
                for (int nt = 0; nt < 4; nt++) {
                    u32 cb = (wn + 8 * nt + g) * GLD + kb + c4;
                    bl[nt][0] = Bl[cb];
                    bl[nt][1] = Bl[cb + 4];
                }
#pragma unroll
                for (int mt = 0; mt < 2; mt++)
#pragma unroll
                    for (int nt = 0; nt < 4; nt++) {
                        mma16816(stl[mt][nt], al[mt], bf[nt]);
                        mma16816(stl[mt][nt], af[mt], bl[nt]);
                    }
            }
        }
        if constexpr (X3) {
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
    if constexpr (X3) {
#pragma unroll
        for (int a = 0; a < 2; a++)
#pragma unroll
            for (int b = 0; b < 4; b++)
#pragma unroll
                for (int c = 0; c < 4; c++) acc[a][b][c] += accl[a][b][c] * (1.f / 2048.f);
    }
    if (p.mode == 2) {
        // Gate columns in warps 0 / 2 (wn = 0), up columns in warps 1 / 3: through shared memory.
        float* Cs = (float*)smem;
        __syncthreads();
#pragma unroll
        for (int mt = 0; mt < 2; mt++)
#pragma unroll
            for (int nt = 0; nt < 4; nt++)
#pragma unroll
                for (int q = 0; q < 4; q++) {
                    u32 rr = wm + 16 * mt + g + 8 * (q >> 1), cc = wn + 8 * nt + 2 * c4 + (q & 1);
                    Cs[rr * 65 + cc] = acc[mt][nt][q] * f;
                }
        __syncthreads();
        for (u32 i = tid; i < GBM * 32; i += 128) {
            u32 rr = i >> 5, cc = i & 31;
            if (rr < nr && n0 + cc < p.mi) p.y[(u64)(r0 + rr) * p.ys + n0 + cc] = situ(Cs[rr * 65 + cc], Cs[rr * 65 + 32 + cc], p.beta, p.lbeta);
        }
        return;
    }
#pragma unroll
    for (int mt = 0; mt < 2; mt++)
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
#pragma unroll
            for (int q = 0; q < 4; q++) {
                u32 rr = wm + 16 * mt + g + 8 * (q >> 1), cc = n0 + wn + 8 * nt + 2 * c4 + (q & 1);
                if (rr < nr && cc < p.n_out) {
                    float v = acc[mt][nt][q] * f;
                    float* o = p.y + (u64)(r0 + rr) * p.ys + cc;
                    if (p.mode == 1) v += *o;
                    if (p.mode == 3) v *= p.wrow[r0 + rr];
                    *o = v;
                }
            }
}

// ---- The split GEMM with exact weight values (`gemmi_*`): every format but f32 / bf16 stores
// small integers or grid values (exact in f16) times a scale (and minus a minimum) per 16 or 32
// values. The values go to the tensor cores as they are, the activations as split halves (two
// products instead of three, no split of the weights), and each 16-wide step's sums are scaled
// (plus minimum times the step's activation sum) with f32 arithmetic.
template <int K>
__device__ __forceinline__ constexpr bool int_ok() {
    return K != K_F32 && K != K_BF16;
}
template <int K>
__device__ __forceinline__ constexpr bool has_min() {
    return K == K_Q4_1 || K == K_Q5_1 || K == K_Q4_K || K == K_Q5_K || K == K_Q2_K;
}
__device__ __forceinline__ float sgn(u32 signs, u32 bit) { return (signs >> bit) & 1 ? -1.f : 1.f; }

// Elements 32 b + 4 j .. + 3 of a row as the values (dq = scale * value + min).
template <int K>
__device__ __forceinline__ float4 dqi(const u8* row, u32 b, u32 j) {
    float4 o;
    if constexpr (K == K_F16) {
        uint2 u = ((const uint2*)row)[8 * b + j];
        o = make_float4(h2f(u.x & 0xffff), h2f(u.x >> 16), h2f(u.y & 0xffff), h2f(u.y >> 16));
    } else if constexpr (K == K_Q8_0) {
        u32 u = rd32h(row + 34 * b + 2 + 4 * j);
        o = make_float4(sbyte(u, 0), sbyte(u, 1), sbyte(u, 2), sbyte(u, 3));
    } else if constexpr (K == K_OQ8 || K == K_F8) {
        u32 u = rd32(row + 32 * b + 4 * j);
        if constexpr (K == K_OQ8)
            o = make_float4(sbyte(u, 0), sbyte(u, 1), sbyte(u, 2), sbyte(u, 3));
        else
            o = make_float4(e4m3(byte(u, 0)), e4m3(byte(u, 1)), e4m3(byte(u, 2)), e4m3(byte(u, 3)));
    } else if constexpr (K == K_Q4_0 || K == K_IQ4_NL || K == K_OQ4) {
        const u8* B = K == K_OQ4 ? row + 16 * b : row + 18 * b + 2;
        u32 u = (K == K_OQ4 ? rd32(B + 4 * (j & 3)) : rd32h(B + 4 * (j & 3))) >> (4 * (j >> 2));
        if constexpr (K == K_IQ4_NL)
            o = make_float4(kvalues_iq4nl[u & 15], kvalues_iq4nl[(u >> 8) & 15], kvalues_iq4nl[(u >> 16) & 15], kvalues_iq4nl[(u >> 24) & 15]);
        else
            o = make_float4((float)(u & 15) - 8.f, (float)((u >> 8) & 15) - 8.f, (float)((u >> 16) & 15) - 8.f, (float)((u >> 24) & 15) - 8.f);
    } else if constexpr (K == K_Q4_1) {
        u32 u = rd32(row + 20 * b + 4 + 4 * (j & 3)) >> (4 * (j >> 2));
        o = make_float4((float)(u & 15), (float)((u >> 8) & 15), (float)((u >> 16) & 15), (float)((u >> 24) & 15));
    } else if constexpr (K == K_Q5_0 || K == K_Q5_1) {
        const u8* B = row + (K == K_Q5_0 ? 22 : 24) * b;
        u32 qh = K == K_Q5_0 ? rd32h(B + 2) : rd32(B + 4);
        const u8* q = B + (K == K_Q5_0 ? 6 : 8);
        u32 u = (K == K_Q5_0 ? rd32h(q + 4 * (j & 3)) : rd32(q + 4 * (j & 3))) >> (4 * (j >> 2));
        u32 e = 4 * j;
        float off = K == K_Q5_0 ? 16.f : 0.f;
        o = make_float4((float)((u & 15) | (((qh >> e) & 1) << 4)) - off, (float)(((u >> 8) & 15) | (((qh >> (e + 1)) & 1) << 4)) - off,
                        (float)(((u >> 16) & 15) | (((qh >> (e + 2)) & 1) << 4)) - off,
                        (float)(((u >> 24) & 15) | (((qh >> (e + 3)) & 1) << 4)) - off);
    } else if constexpr (K == K_MXFP4) {
        const u8* q = row + 17 * b + 1 + 4 * (j & 3);
        u32 sh = 4 * (j >> 2);
        o = make_float4(kvalues_mxfp4[(q[0] >> sh) & 15], kvalues_mxfp4[(q[1] >> sh) & 15], kvalues_mxfp4[(q[2] >> sh) & 15], kvalues_mxfp4[(q[3] >> sh) & 15]);
    } else if constexpr (K == K_NVFP4) {
        const u8* B = row + 36 * (b >> 1);
        u32 e0 = 32 * (b & 1) + 4 * j, sb = e0 >> 4, w = e0 & 15;
        u32 u = rd32(B + 4 + 8 * sb + (w & 7)) >> (w >= 8 ? 4 : 0);
        o = make_float4(kvalues_mxfp4[u & 15], kvalues_mxfp4[(u >> 8) & 15], kvalues_mxfp4[(u >> 16) & 15], kvalues_mxfp4[(u >> 24) & 15]);
    } else if constexpr (K == K_Q4_K || K == K_Q5_K) {
        const u8* B = row + (K == K_Q4_K ? 144 : 176) * (b >> 3);
        u32 sub = b & 7;
        u32 u = rd32(B + (K == K_Q4_K ? 16 : 48) + 32 * (sub >> 1) + 4 * j) >> (4 * (sub & 1));
        u32 hb = 0;
        if constexpr (K == K_Q5_K) hb = rd32(B + 16 + 4 * j) >> sub;
        o = make_float4((float)((u & 15) | ((hb & 1) << 4)), (float)(((u >> 8) & 15) | (((hb >> 8) & 1) << 4)),
                        (float)(((u >> 16) & 15) | (((hb >> 16) & 1) << 4)), (float)(((u >> 24) & 15) | (((hb >> 24) & 1) << 4)));
    } else if constexpr (K == K_Q6_K) {
        const u8* B = row + 210 * (b >> 3);
        u32 sub = b & 7, n = sub >> 2, qq = sub & 3;
        u32 ul = rd32h(B + 64 * n + 32 * (qq & 1) + 4 * j) >> (4 * (qq >> 1));
        u32 uh = rd32h(B + 128 + 32 * n + 4 * j) >> (2 * qq);
        o = make_float4((float)(int)((ul & 15) | ((uh & 3) << 4)) - 32.f, (float)(int)(((ul >> 8) & 15) | (((uh >> 8) & 3) << 4)) - 32.f,
                        (float)(int)(((ul >> 16) & 15) | (((uh >> 16) & 3) << 4)) - 32.f,
                        (float)(int)(((ul >> 24) & 15) | (((uh >> 24) & 3) << 4)) - 32.f);
    } else if constexpr (K == K_Q3_K) {
        const u8* B = row + 110 * (b >> 3);
        u32 sub = b & 7, n = sub >> 2, jj = sub & 3;
        u32 u = rd32h(B + 32 + 32 * n + 4 * j) >> (2 * jj);
        u32 h = rd32h(B + 4 * j) >> (4 * n + jj);
        o = make_float4((float)(u & 3) - ((h & 1) ? 0.f : 4.f), (float)((u >> 8) & 3) - (((h >> 8) & 1) ? 0.f : 4.f),
                        (float)((u >> 16) & 3) - (((h >> 16) & 1) ? 0.f : 4.f), (float)((u >> 24) & 3) - (((h >> 24) & 1) ? 0.f : 4.f));
    } else if constexpr (K == K_Q2_K) {
        const u8* B = row + 84 * (b >> 3);
        u32 sub = b & 7, n = sub >> 2, jj = sub & 3;
        u32 u = rd32(B + 16 + 32 * n + 4 * j) >> (2 * jj);
        o = make_float4((float)(u & 3), (float)((u >> 8) & 3), (float)((u >> 16) & 3), (float)((u >> 24) & 3));
    } else if constexpr (K == K_IQ4_XS) {
        const u8* B = row + 136 * (b >> 3);
        u32 u = rd32(B + 8 + 16 * (b & 7) + 4 * (j & 3)) >> (4 * (j >> 2));
        o = make_float4(kvalues_iq4nl[u & 15], kvalues_iq4nl[(u >> 8) & 15], kvalues_iq4nl[(u >> 16) & 15], kvalues_iq4nl[(u >> 24) & 15]);
    } else if constexpr (K == K_IQ2_XXS || K == K_IQ2_XS || K == K_IQ2_S) {
        u32 ib = b & 7, l = j >> 1, j0 = 4 * (j & 1);
        u64 g;
        u32 signs;
        if constexpr (K == K_IQ2_XXS) {
            const u8* B = row + 66 * (b >> 3);
            u32 a0 = rd32h(B + 2 + 8 * ib), a1 = rd32h(B + 6 + 8 * ib);
            g = iq2xxs_grid[(a0 >> (8 * l)) & 255];
            signs = ksign((a1 >> (7 * l)) & 127);
        } else if constexpr (K == K_IQ2_XS) {
            const u8* B = row + 74 * (b >> 3);
            u32 q = rd16(B + 2 + 2 * (4 * ib + l));
            g = iq2xs_grid[q & 511];
            signs = ksign(q >> 9);
        } else {
            const u8* B = row + 82 * (b >> 3);
            u32 idx = (u32)B[2 + 4 * ib + l] | (((u32)B[66 + ib] << (8 - 2 * l)) & 0x300);
            g = iq2s_grid[idx];
            signs = B[34 + 4 * ib + l];
        }
        u32 gw = (u32)(g >> (8 * j0));
        o = make_float4((float)byte(gw, 0) * sgn(signs, j0), (float)byte(gw, 1) * sgn(signs, j0 + 1), (float)byte(gw, 2) * sgn(signs, j0 + 2),
                        (float)byte(gw, 3) * sgn(signs, j0 + 3));
    } else if constexpr (K == K_IQ3_XXS || K == K_IQ3_S) {
        u32 ib = b & 7, l = j >> 1, hf = j & 1;
        u32 gw, signs;
        if constexpr (K == K_IQ3_XXS) {
            const u8* B = row + 98 * (b >> 3);
            u32 aux = rd32h(B + 66 + 4 * ib);
            signs = ksign((aux >> (7 * l)) & 127);
            gw = iq3xxs_grid[B[2 + 8 * ib + 2 * l + hf]];
        } else {
            const u8* B = row + 110 * (b >> 3);
            u32 h = B[66 + ib];
            u32 idx = (u32)B[2 + 8 * ib + 2 * l + hf] | ((h << (8 - 2 * l - hf)) & 256);
            gw = iq3s_grid[idx];
            signs = B[74 + 4 * ib + l];
        }
        u32 s0 = 4 * hf;
        o = make_float4((float)byte(gw, 0) * sgn(signs, s0), (float)byte(gw, 1) * sgn(signs, s0 + 1), (float)byte(gw, 2) * sgn(signs, s0 + 2),
                        (float)byte(gw, 3) * sgn(signs, s0 + 3));
    } else if constexpr (K == K_IQ1_S || K == K_IQ1_M) {
        u32 ib = b & 7, l = j >> 1, j0 = 4 * (j & 1);
        float delta;
        u64 g;
        if constexpr (K == K_IQ1_S) {
            const u8* B = row + 50 * (b >> 3);
            u32 qh = rd16(B + 34 + 2 * ib);
            delta = (qh & 0x8000) ? -0.125f : 0.125f;
            g = iq1s_grid[(u32)B[2 + 4 * ib + l] | (((qh >> (3 * l)) & 7) << 8)];
        } else {
            const u8* B = row + 56 * (b >> 3);
            u32 qh = B[32 + 2 * ib + l / 2];
            u32 idx = (u32)B[4 * ib + l] | ((qh << (l % 2 ? 4 : 8)) & 0x700);
            delta = (qh & (l % 2 ? 0x80 : 0x08)) ? -0.125f : 0.125f;
            g = iq1s_grid[idx];
        }
        u32 gw = (u32)(g >> (8 * j0));
        o = make_float4(sbyte(gw, 0) + delta, sbyte(gw, 1) + delta, sbyte(gw, 2) + delta, sbyte(gw, 3) + delta);
    } else {
        o = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    return o;
}

// The (scale, minimum) of elements 32 b + 16 half .. + 15 of a row (before a matrix's block
// scales and factors).
template <int K>
__device__ __forceinline__ float2 dqs(const u8* row, u32 b, u32 half) {
    if constexpr (K == K_Q8_0) return make_float2(h2f(rd16(row + 34 * b)), 0.f);
    else if constexpr (K == K_Q4_0 || K == K_IQ4_NL) return make_float2(h2f(rd16(row + 18 * b)), 0.f);
    else if constexpr (K == K_Q4_1) return make_float2(h2f(rd16(row + 20 * b)), h2f(rd16(row + 20 * b + 2)));
    else if constexpr (K == K_Q5_0) return make_float2(h2f(rd16(row + 22 * b)), 0.f);
    else if constexpr (K == K_Q5_1) return make_float2(h2f(rd16(row + 24 * b)), h2f(rd16(row + 24 * b + 2)));
    else if constexpr (K == K_MXFP4) return make_float2(e8m0_half(row[17 * b]), 0.f);
    else if constexpr (K == K_NVFP4) return make_float2(ue4m3_half(row[36 * (b >> 1) + 2 * (b & 1) + half]), 0.f);
    else if constexpr (K == K_Q4_K || K == K_Q5_K) {
        const u8* B = row + (K == K_Q4_K ? 144 : 176) * (b >> 3);
        float sc, mn;
        scale_min_k4(b & 7, B + 4, sc, mn);
        return make_float2(h2f(rd16(B)) * sc, -h2f(rd16(B + 2)) * mn);
    } else if constexpr (K == K_Q6_K) {
        const u8* B = row + 210 * (b >> 3);
        u32 sub = b & 7, n = sub >> 2, qq = sub & 3;
        return make_float2(h2f(rd16(B + 208)) * (float)(signed char)B[192 + 8 * n + half + 2 * qq], 0.f);
    } else if constexpr (K == K_Q3_K) {
        const u8* B = row + 110 * (b >> 3);
        u32 sub = b & 7, n = sub >> 2, jj = sub & 3;
        u32 k = 8 * n + 2 * jj + half;
        u32 a0 = rd32h(B + 96), a1 = rd32h(B + 100), tmp = rd32h(B + 104);
        const u32 m1 = 0x03030303u, m2 = 0x0f0f0f0fu;
        u32 aux;
        switch (k >> 2) {
            case 0: aux = (a0 & m2) | ((tmp & m1) << 4); break;
            case 1: aux = (a1 & m2) | (((tmp >> 2) & m1) << 4); break;
            case 2: aux = ((a0 >> 4) & m2) | (((tmp >> 4) & m1) << 4); break;
            default: aux = ((a1 >> 4) & m2) | (((tmp >> 6) & m1) << 4); break;
        }
        return make_float2(h2f(rd16(B + 108)) * (float)((int)(signed char)(u8)(aux >> (8 * (k & 3))) - 32), 0.f);
    } else if constexpr (K == K_Q2_K) {
        const u8* B = row + 84 * (b >> 3);
        u32 sub = b & 7, n = sub >> 2, jj = sub & 3;
        u32 sv = B[8 * n + 2 * jj + half];
        return make_float2(h2f(rd16(B + 80)) * (float)(sv & 15), -h2f(rd16(B + 82)) * (float)(sv >> 4));
    } else if constexpr (K == K_IQ4_XS) {
        const u8* B = row + 136 * (b >> 3);
        u32 ib = b & 7;
        int ls = (int)(((B[4 + ib / 2] >> (4 * (ib % 2))) & 15) | (((rd16(B + 2) >> (2 * ib)) & 3) << 4)) - 32;
        return make_float2(h2f(rd16(B)) * (float)ls, 0.f);
    } else if constexpr (K == K_IQ2_XXS) {
        const u8* B = row + 66 * (b >> 3);
        u32 a1 = rd32h(B + 6 + 8 * (b & 7));
        return make_float2(h2f(rd16(B)) * (0.5f + (float)(a1 >> 28)) * 0.25f, 0.f);
    } else if constexpr (K == K_IQ2_XS || K == K_IQ2_S) {
        const u8* B = row + (K == K_IQ2_XS ? 74 : 82) * (b >> 3);
        u32 sv = B[(K == K_IQ2_XS ? 66 : 74) + (b & 7)];
        return make_float2(h2f(rd16(B)) * (0.5f + (float)(half ? (sv >> 4) : (sv & 15))) * 0.25f, 0.f);
    } else if constexpr (K == K_IQ3_XXS) {
        const u8* B = row + 98 * (b >> 3);
        u32 aux = rd32h(B + 66 + 4 * (b & 7));
        return make_float2(h2f(rd16(B)) * (0.5f + (float)(aux >> 28)) * 0.5f, 0.f);
    } else if constexpr (K == K_IQ3_S) {
        const u8* B = row + 110 * (b >> 3);
        u32 ib = b & 7;
        u32 sv = B[106 + ib / 2] >> (4 * (ib % 2));
        return make_float2(h2f(rd16(B)) * (float)(1 + 2 * (sv & 15)), 0.f);
    } else if constexpr (K == K_IQ1_S) {
        const u8* B = row + 50 * (b >> 3);
        u32 qh = rd16(B + 34 + 2 * (b & 7));
        return make_float2(h2f(rd16(B)) * (float)(2 * ((qh >> 12) & 7) + 1), 0.f);
    } else if constexpr (K == K_IQ1_M) {
        const u8* B = row + 56 * (b >> 3);
        u32 ib = b & 7;
        u32 s0 = rd16(B + 48), s1 = rd16(B + 50), s2 = rd16(B + 52), s3 = rd16(B + 54);
        float d = h2f((u16)((s0 >> 12) | ((s1 >> 8) & 0x00f0) | ((s2 >> 4) & 0x0f00) | (s3 & 0xf000)));
        u32 sv = rd16(B + 48 + 2 * (ib / 2)) >> (6 * (ib % 2));
        return make_float2(d * (float)(2 * ((sv >> (half ? 3 : 0)) & 7) + 1), 0.f);
    } else {
        return make_float2(1.f, 0.f);
    }
}

template <int K>
__device__ __forceinline__ void gemmi_body(const W& w, const GemmP& p) {
    // The activations' halves, the weights' values, per 16-wide step the scales and minima of the
    // weight rows and the activations' sums; the 64 x 65 outputs of the act epilogue after.
    __shared__ u32 smem[GBM * 65 > 3 * GBM * GLD + 6 * 64 ? GBM * 65 : 3 * GBM * GLD + 6 * 64];
    u32* As = smem;
    u32* Al = smem + GBM * GLD;
    u32* Bs = smem + 2 * GBM * GLD;
    float* Ssc = (float*)(smem + 3 * GBM * GLD); // [2][64]
    float* Smn = Ssc + 128;                      // [2][64]
    float* Xsm = Smn + 128;                      // [2][64]
    u32 tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    u32 e = 0, r0, nr;
    if (p.tiles) {
        if (blockIdx.y >= *p.ntiles) return;
        Tile tl = p.tiles[blockIdx.y];
        e = tl.e;
        r0 = tl.r0;
        nr = tl.n;
    } else {
        r0 = blockIdx.y * GBM;
        if (r0 >= p.rows) return;
        nr = min(GBM, p.rows - r0);
    }
    const u8* base = w.q + (u64)e * w.es;
    const float* sbase = w.s ? w.s + (u64)e * w.ss : w.s;
    float f = w.f * (w.ef ? w.ef[e] : 1.f);
    u32 n0 = blockIdx.x * (p.mode == 2 ? 32 : GBN);
    u32 lr = tid >> 1, lh = tid & 1;
    const float* xrow = nullptr;
    const u32 *xhr = nullptr, *xlr = nullptr;
    if (lr < nr) {
        u32 gr = r0 + lr;
        u32 src = p.gidx ? (u32)p.gidx[gr] : gr;
        xrow = p.x + (u64)src * p.xs;
        if (p.xh) {
            xhr = p.xh + (u64)src * (p.n_in / 2);
            xlr = p.xl + (u64)src * (p.n_in / 2);
        }
    }
    u32 wr;
    bool wok;
    if (p.mode == 2) {
        u32 jj = n0 + (lr & 31);
        wok = jj < p.mi;
        wr = lr < 32 ? (p.inter ? 2 * jj : jj) : (p.inter ? 2 * jj + 1 : p.mi + jj);
    } else {
        wr = n0 + lr;
        wok = wr < p.n_out;
    }
    const u8* wrow = base + (u64)wr * w.rb;
    float acc[2][4][4];
#pragma unroll
    for (int a = 0; a < 2; a++)
#pragma unroll
        for (int b = 0; b < 4; b++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[a][b][c] = 0.f;
    u32 wm = (warp >> 1) * 32, wn = (warp & 1) * 32;
    u32 g = lane >> 2, c4 = lane & 3;
    u32 nb = p.n_in / 32;
    for (u32 b = 0; b < nb; b++) {
        // X: this thread's 16 values (half lh of the step) as split halves, and their sum.
        float xs = 0.f;
        if (xhr) {
            const uint4* ph = (const uint4*)(xhr + 16 * b + 8 * lh);
            const uint4* pl = (const uint4*)(xlr + 16 * b + 8 * lh);
            uint4 h0 = ph[0], h1 = ph[1], l0 = pl[0], l1 = pl[1];
            u32* ah = As + lr * GLD + 8 * lh;
            u32* al2 = Al + lr * GLD + 8 * lh;
            ah[0] = h0.x, ah[1] = h0.y, ah[2] = h0.z, ah[3] = h0.w, ah[4] = h1.x, ah[5] = h1.y, ah[6] = h1.z, ah[7] = h1.w;
            al2[0] = l0.x, al2[1] = l0.y, al2[2] = l0.z, al2[3] = l0.w, al2[4] = l1.x, al2[5] = l1.y, al2[6] = l1.z, al2[7] = l1.w;
            if constexpr (has_min<K>()) {
                const float4* xp = (const float4*)(xrow + 32 * b + 16 * lh);
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    float4 v = xp[i];
                    xs += (v.x + v.y) + (v.z + v.w);
                }
            }
        } else if (xrow) {
            const float4* xp = (const float4*)(xrow + 32 * b + 16 * lh);
#pragma unroll
            for (int i = 0; i < 4; i++) {
                float4 v = xp[i];
                u32 at = lr * GLD + 8 * lh + 2 * i;
                split_h2(v.x, v.y, As[at], Al[at]);
                split_h2(v.z, v.w, As[at + 1], Al[at + 1]);
                xs += (v.x + v.y) + (v.z + v.w);
            }
        } else {
#pragma unroll
            for (int i = 0; i < 8; i++) As[lr * GLD + 8 * lh + i] = Al[lr * GLD + 8 * lh + i] = 0u;
        }
        Xsm[lh * 64 + lr] = xs;
        // W: the 16 values of half lh (exact in f16), its scale and minimum.
        if (wok) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
                float4 v = dqi<K>(wrow, b, 4 * lh + i);
                u32 at = lr * GLD + 8 * lh + 2 * i;
                Bs[at] = pack_h2(v.x, v.y);
                Bs[at + 1] = pack_h2(v.z, v.w);
            }
            float2 sm = dqs<K>(wrow, b, lh);
            if constexpr (BLOCK_SCALED(K)) {
                float bs = sbase[(wr / w.sr) * w.ncb + (32 * b) / w.sc];
                sm.x *= bs;
                sm.y *= bs;
            }
            Ssc[lh * 64 + lr] = sm.x;
            Smn[lh * 64 + lr] = sm.y;
        } else {
#pragma unroll
            for (int i = 0; i < 8; i++) Bs[lr * GLD + 8 * lh + i] = 0u;
            Ssc[lh * 64 + lr] = 0.f;
            Smn[lh * 64 + lr] = 0.f;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            u32 kb = 8 * kk;
            u32 af[2][4], al[2][4], bf[4][2];
#pragma unroll
            for (int mt = 0; mt < 2; mt++) {
                u32 rb = (wm + 16 * mt + g) * GLD + kb + c4;
                af[mt][0] = As[rb];
                af[mt][1] = As[rb + 8 * GLD];
                af[mt][2] = As[rb + 4];
                af[mt][3] = As[rb + 8 * GLD + 4];
                al[mt][0] = Al[rb];
                al[mt][1] = Al[rb + 8 * GLD];
                al[mt][2] = Al[rb + 4];
                al[mt][3] = Al[rb + 8 * GLD + 4];
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
                    float st[4] = {0.f, 0.f, 0.f, 0.f}, stl[4] = {0.f, 0.f, 0.f, 0.f};
                    mma16816(st, af[mt], bf[nt]);
                    mma16816(stl, al[mt], bf[nt]);
#pragma unroll
                    for (int q = 0; q < 4; q++) {
                        u32 rr = wm + 16 * mt + g + 8 * (q >> 1), cc = wn + 8 * nt + 2 * c4 + (q & 1);
                        float v = st[q] + stl[q] * (1.f / 2048.f);
                        acc[mt][nt][q] += Ssc[kk * 64 + cc] * v;
                        if constexpr (has_min<K>()) acc[mt][nt][q] += Smn[kk * 64 + cc] * Xsm[kk * 64 + rr];
                    }
                }
        }
        __syncthreads();
    }
    if (p.mode == 2) {
        float* Cs = (float*)smem;
#pragma unroll
        for (int mt = 0; mt < 2; mt++)
#pragma unroll
            for (int nt = 0; nt < 4; nt++)
#pragma unroll
                for (int q = 0; q < 4; q++) {
                    u32 rr = wm + 16 * mt + g + 8 * (q >> 1), cc = wn + 8 * nt + 2 * c4 + (q & 1);
                    Cs[rr * 65 + cc] = acc[mt][nt][q] * f;
                }
        __syncthreads();
        for (u32 i = tid; i < GBM * 32; i += 128) {
            u32 rr = i >> 5, cc = i & 31;
            if (rr < nr && n0 + cc < p.mi) p.y[(u64)(r0 + rr) * p.ys + n0 + cc] = situ(Cs[rr * 65 + cc], Cs[rr * 65 + 32 + cc], p.beta, p.lbeta);
        }
        return;
    }
#pragma unroll
    for (int mt = 0; mt < 2; mt++)
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
#pragma unroll
            for (int q = 0; q < 4; q++) {
                u32 rr = wm + 16 * mt + g + 8 * (q >> 1), cc = n0 + wn + 8 * nt + 2 * c4 + (q & 1);
                if (rr < nr && cc < p.n_out) {
                    float v = acc[mt][nt][q] * f;
                    float* o = p.y + (u64)(r0 + rr) * p.ys + cc;
                    if (p.mode == 1) v += *o;
                    if (p.mode == 3) v *= p.wrow[r0 + rr];
                    *o = v;
                }
            }
}

template <int K>
__device__ __forceinline__ void gemmi_any(const W& w, const GemmP& p) {
    if constexpr (int_ok<K>())
        gemmi_body<K>(w, p);
    else
        gemm_body<K, true>(w, p);
}

// `GGML_ONLY` (the language models' decode and prefill plans, `cuda_decode.rs`): only the matvec
// and the GEMMs of the kinds asked for.
#ifdef GGML_ONLY
#define KERNELS(K)                                                                                                                         \
    extern "C" __global__ void __launch_bounds__(256) mv_##K(W w, const float* x, float* y, u32 n_in, u32 n_out, u32 xs, u32 ys, u32 mode) { \
        mv_body<K>(w, x, y, n_in, n_out, xs, ys, mode);                                                                                    \
    }                                                                                                                                      \
    extern "C" __global__ void __launch_bounds__(256) mvp_##K(W w, W w2, const float* x, float* y, u32 n_in, u32 n_out, u32 xs, u32 ys, u32 gelu) { \
        mvp_body<K>(w, w2, x, y, n_in, n_out, xs, ys, gelu);                                                                               \
    }                                                                                                                                      \
    extern "C" __global__ void __launch_bounds__(128) gemm_##K(W w, GemmP p) { gemm_body<K, false>(w, p); }                  \
    extern "C" __global__ void __launch_bounds__(128) gemmi_##K(W w, GemmP p) { gemmi_any<K>(w, p); }
#else
#define KERNELS(K)                                                                                                                         \
    extern "C" __global__ void __launch_bounds__(128) mv_##K(W w, const float* x, float* y, u32 n_in, u32 n_out, u32 xs, u32 ys, u32 mode) { \
        mv_body<K>(w, x, y, n_in, n_out, xs, ys, mode);                                                                                    \
    }                                                                                                                                      \
    extern "C" __global__ void __launch_bounds__(256)                                                                                      \
        moe_gu_##K(W w, const int* ids, const float* x, float* mid, u32 n_in, u32 mi, u32 k, u32 xs, u32 inter, float beta, float lbeta) { \
        moe_gu_body<K>(w, ids, x, mid, n_in, mi, k, xs, inter, beta, lbeta);                                                               \
    }                                                                                                                                      \
    extern "C" __global__ void __launch_bounds__(256)                                                                                      \
        moe_dn_##K(W w, const int* ids, const float* wts, const float* mid, float* y, u32 h, u32 mi, u32 k, u32 ys) {                      \
        moe_dn_body<K>(w, ids, wts, mid, y, h, mi, k, ys);                                                                                 \
    }                                                                                                                                      \
    extern "C" __global__ void __launch_bounds__(128) gemm_##K(W w, GemmP p) { gemm_body<K, false>(w, p); }                  \
    extern "C" __global__ void __launch_bounds__(128) gemmi_##K(W w, GemmP p) { gemmi_any<K>(w, p); }                         \
    extern "C" __global__ void __launch_bounds__(256) mvm_##K(MvJobs p) { mvm_body<K>(p); }                                  \
    extern "C" __global__ void __launch_bounds__(256)                                                                                      \
        gate_down_##K(W w, const float* x, const float* m, float* d, u32 h, u32 r, float eps) {                                            \
        gate_down_body<K>(w, x, m, d, h, r, eps);                                                                                          \
    }                                                                                                                                      \
    extern "C" __global__ void __launch_bounds__(256)                                                                                      \
        gate_down_t_##K(W w, const float* x, const float* m, float* d, u32 h, u32 r, float eps) {                                          \
        gate_down_t_body<K>(w, x, m, d, h, r, eps);                                                                                        \
    }                                                                                                                                      \
    extern "C" __global__ void __launch_bounds__(256)                                                                                      \
        gate_out_##K(W wf, W wg, const float* d, const float* x, const float* m, float* out, u32 h, u32 r) {                               \
        gate_out_body<K>(wf, wg, d, x, m, out, h, r);                                                                                      \
    }

#endif

#ifdef HAS_0
KERNELS(0)
#endif
#ifdef HAS_1
KERNELS(1)
#endif
#ifdef HAS_2
KERNELS(2)
#endif
#ifdef HAS_3
KERNELS(3)
#endif
#ifdef HAS_6
KERNELS(6)
#endif
#ifdef HAS_7
KERNELS(7)
#endif
#ifdef HAS_8
KERNELS(8)
#endif
#ifdef HAS_10
KERNELS(10)
#endif
#ifdef HAS_11
KERNELS(11)
#endif
#ifdef HAS_12
KERNELS(12)
#endif
#ifdef HAS_13
KERNELS(13)
#endif
#ifdef HAS_14
KERNELS(14)
#endif
#ifdef HAS_16
KERNELS(16)
#endif
#ifdef HAS_17
KERNELS(17)
#endif
#ifdef HAS_18
KERNELS(18)
#endif
#ifdef HAS_19
KERNELS(19)
#endif
#ifdef HAS_20
KERNELS(20)
#endif
#ifdef HAS_21
KERNELS(21)
#endif
#ifdef HAS_22
KERNELS(22)
#endif
#ifdef HAS_23
KERNELS(23)
#endif
#ifdef HAS_29
KERNELS(29)
#endif
#ifdef HAS_30
KERNELS(30)
#endif
#ifdef HAS_39
KERNELS(39)
#endif
#ifdef HAS_40
KERNELS(40)
#endif
#ifdef HAS_200
KERNELS(200)
#endif
#ifdef HAS_201
KERNELS(201)
#endif
#ifdef HAS_202
KERNELS(202)
#endif

#ifndef GGML_ONLY

// ---- Small ops. `info[0]` is the position of the pass's first token (read on the device, so a
// captured decode step replays at any position).

__device__ __forceinline__ float block_sum(float v, float* red) {
    v = warp_sum(v);
    u32 warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float s = 0.f;
    for (u32 i = 0; i < nw; i++) s += red[i];
    return s;
}

// y[row] = x[row] * rsqrt(mean(x^2) + eps) (* w): a block per row.
extern "C" __global__ void rms_rows(const float* x, float* y, const float* w, u32 n, float eps, u32 xs, u32 ys) {
    __shared__ float red[32];
    const float* xr = x + (u64)blockIdx.x * xs;
    float* yr = y + (u64)blockIdx.x * ys;
    float ss = 0.f;
    for (u32 i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    float r = rsqrtf(block_sum(ss, red) / (float)n + eps);
    for (u32 i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * r * (w ? w[i] : 1.f);
}

// The attention residual's mixes: for each token (a block) and query, the softmax (query . state
// * h^-0.5) mix of the block states (RMS-normed already) and of the RMS-normed x.
#define MAXS 11
struct MixP {
    const float* st[MAXS];
    float* out[4];
    // Optional RMS norm weights of the outputs (all or none), and their eps.
    const float* nw[4];
    const float* x;
    const float* q;
    u32 ns, nq, h;
    float eps;
    float neps;
    u32 _p;
};
template <int NQ>
__device__ __forceinline__ void mix_body(const MixP& p) {
    // One pass for every dot product (each thread keeps a partial per state and query, and of
    // x . x; the loops are unrolled to MAXS so they stay in registers), one reduction, then the
    // outputs (with norms: their sums of squares first).
    __shared__ float red[32][NQ * (MAXS + 1) + 1];
    __shared__ float wsh[NQ][MAXS + 1];
    __shared__ float rn_s, on[NQ];
    u32 tok = blockIdx.x, h = p.h, ns = p.ns;
    u32 warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    const float* xr = p.x + (u64)tok * h;
    constexpr u32 NP = NQ * (MAXS + 1) + 1;
    float part[NP];
#pragma unroll
    for (u32 i = 0; i < NP; i++) part[i] = 0.f;
    for (u32 i = threadIdx.x; i < h; i += blockDim.x) {
        float xv = xr[i];
        part[NP - 1] += xv * xv;
        float qv[NQ];
#pragma unroll
        for (u32 qi = 0; qi < NQ; qi++) qv[qi] = p.q[(u64)qi * h + i];
#pragma unroll
        for (u32 s = 0; s <= MAXS; s++) {
            if (s > ns) break;
            float v = s < ns ? p.st[s][(u64)tok * h + i] : xv;
#pragma unroll
            for (u32 qi = 0; qi < NQ; qi++) part[qi * (MAXS + 1) + s] += qv[qi] * v;
        }
    }
#pragma unroll
    for (u32 i = 0; i < NP; i++) {
        float v = warp_sum(part[i]);
        if (lane == 0) red[warp][i] = v;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot[NP];
        for (u32 i = 0; i < NP; i++) {
            float v = 0.f;
            for (u32 w = 0; w < nw; w++) v += red[w][i];
            tot[i] = v;
        }
        float rn = 1.f / sqrtf(tot[NP - 1] / (float)h + p.eps);
        float scale = rsqrtf((float)h);
        rn_s = rn;
        for (u32 qi = 0; qi < NQ; qi++) {
            float m = -INFINITY, sc[MAXS + 1];
            for (u32 s = 0; s <= ns; s++) {
                sc[s] = tot[qi * (MAXS + 1) + s] * scale * (s < ns ? 1.f : rn);
                m = fmaxf(m, sc[s]);
            }
            float l = 0.f;
            for (u32 s = 0; s <= ns; s++) {
                sc[s] = __expf(sc[s] - m);
                l += sc[s];
            }
            for (u32 s = 0; s <= ns; s++) wsh[qi][s] = sc[s] / l;
        }
    }
    __syncthreads();
    float rn = rn_s;
    if (p.nw[0]) {
        float ss[NQ];
#pragma unroll
        for (u32 qi = 0; qi < NQ; qi++) ss[qi] = 0.f;
        for (u32 i = threadIdx.x; i < h; i += blockDim.x) {
            float xv = xr[i] * rn;
            float a[NQ];
#pragma unroll
            for (u32 qi = 0; qi < NQ; qi++) a[qi] = wsh[qi][ns] * xv;
            for (u32 s = 0; s < ns; s++) {
                float v = p.st[s][(u64)tok * h + i];
#pragma unroll
                for (u32 qi = 0; qi < NQ; qi++) a[qi] += wsh[qi][s] * v;
            }
#pragma unroll
            for (u32 qi = 0; qi < NQ; qi++) ss[qi] += a[qi] * a[qi];
        }
#pragma unroll
        for (u32 qi = 0; qi < NQ; qi++) {
            float v = warp_sum(ss[qi]);
            if (lane == 0) red[warp][qi] = v;
        }
        __syncthreads();
        if (threadIdx.x < NQ) {
            float t = 0.f;
            for (u32 w = 0; w < nw; w++) t += red[w][threadIdx.x];
            on[threadIdx.x] = rsqrtf(t / (float)h + p.neps);
        }
        __syncthreads();
    }
    for (u32 i = threadIdx.x; i < h; i += blockDim.x) {
        float xv = xr[i] * rn;
        float a[NQ];
#pragma unroll
        for (u32 qi = 0; qi < NQ; qi++) a[qi] = wsh[qi][ns] * xv;
        for (u32 s = 0; s < ns; s++) {
            float v = p.st[s][(u64)tok * h + i];
#pragma unroll
            for (u32 qi = 0; qi < NQ; qi++) a[qi] += wsh[qi][s] * v;
        }
#pragma unroll
        for (u32 qi = 0; qi < NQ; qi++) p.out[qi][(u64)tok * h + i] = p.nw[0] ? a[qi] * on[qi] * p.nw[qi][i] : a[qi];
    }
}
extern "C" __global__ void __launch_bounds__(1024) mix_1(MixP p) { mix_body<1>(p); }
extern "C" __global__ void __launch_bounds__(512) mix_3(MixP p) { mix_body<3>(p); }
extern "C" __global__ void __launch_bounds__(512) mix_4(MixP p) { mix_body<4>(p); }

// A message gate's input: cat[tok] = [rms(x); rms(m)].
extern "C" __global__ void gate_cat(const float* x, const float* m, float* cat, u32 h, float eps) {
    __shared__ float red[32];
    u32 tok = blockIdx.x;
    const float* xr = x + (u64)tok * h;
    const float* mr = m + (u64)tok * h;
    float a = 0.f, b = 0.f;
    for (u32 i = threadIdx.x; i < h; i += blockDim.x) {
        a += xr[i] * xr[i];
        b += mr[i] * mr[i];
    }
    float ra = rsqrtf(block_sum(a, red) / (float)h + eps);
    float rb = rsqrtf(block_sum(b, red) / (float)h + eps);
    float* o = cat + (u64)tok * 2 * h;
    for (u32 i = threadIdx.x; i < h; i += blockDim.x) {
        o[i] = xr[i] * ra;
        o[h + i] = mr[i] * rb;
    }
}

// The split halves of `rows` rows of k values (x rows `xs` apart) for the GEMMs: hi[r][i] and
// lo[r][i] the half2 of elements 2i, 2i + 1.
extern "C" __global__ void split_rows(const float* x, u32 xs, u32 rows, u32 k, u32* hi, u32* lo) {
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 per = k / 2;
    if (i >= (u64)rows * per) return;
    u64 r = i / per, c = i % per;
    float2 v = *(const float2*)(x + r * xs + 2 * c);
    split_h2(v.x, v.y, hi[i], lo[i]);
}

// out = (1 - tanh f) x + tanh(g) m (elementwise over n values).
extern "C" __global__ void gate_mix(const float* x, const float* m, const float* f, const float* g, float* out, u32 n) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (1.f - tanh_acc(f[i])) * x[i] + tanh_acc(g[i]) * m[i];
}

extern "C" __global__ void add_k(float* x, const float* a, u32 n) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += a[i];
}

// The causal depthwise convolution with a skip, in place over x ([t, ch]) for the positions
// pos .. pos + t - 1 (pos = info[0]). `hist` is a ring of `ring` rows by position (the host's
// ConvRing: position p in row (p + k - 1) % ring, `ring` = k - 1 + the positions a session can go
// back), so the inputs before a pass are found whatever position it starts at; every pass writes
// its inputs to it. `dyn` ([t][groups * k]): per-token taps added to channel c's (group
// c / (ch / groups)).
struct ConvJob {
    float* x;
    float* hist;
    const float* taps;
    const float* dyn;
};
struct ConvJobs {
    ConvJob j[2];
    const int* info;
    u32 ch, k, t, groups, ring;
    float norm;
};
__device__ __forceinline__ u64 ring_row(int p, u32 k, u32 ring) { return (u64)((p + (int)k - 1 + 1024 * (int)ring) % (int)ring); }

// Decode (a few tokens): a thread per channel walking the tokens, the last k - 1 inputs in
// registers.
extern "C" __global__ void sconv(ConvJobs P) {
    const ConvJob& J = P.j[blockIdx.y];
    float* x = J.x;
    float* hist = J.hist;
    const float* taps = J.taps;
    const float* dyn = J.dyn;
    u32 ch = P.ch, k = P.k, t = P.t, groups = P.groups;
    int pos = P.info[0];
    float norm = P.norm;
    u32 c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    float win[8];
    for (u32 i = 0; i + 1 < k; i++) win[i] = hist[ring_row(pos - (int)(k - 1) + (int)i, k, P.ring) * ch + c];
    float tp[8];
    for (u32 l = 0; l < k; l++) tp[l] = taps[(u64)l * ch + c];
    u32 grp = dyn ? c / (ch / groups) : 0;
    for (u32 ti = 0; ti < t; ti++) {
        float cur = x[(u64)ti * ch + c];
        const float* dr = dyn ? dyn + ((u64)ti * groups + grp) * k : nullptr;
        float acc = cur + (tp[0] + (dr ? dr[0] : 0.f)) * cur;
        for (u32 l = 1; l < k; l++) acc += (tp[l] + (dr ? dr[l] : 0.f)) * win[k - 1 - l];
        x[(u64)ti * ch + c] = acc * norm;
        hist[ring_row(pos + (int)ti, k, P.ring) * ch + c] = cur;
        for (u32 i = 0; i + 2 < k; i++) win[i] = win[i + 1];
        if (k > 1) win[k - 2] = cur;
    }
}

// Prefill convolutions, a thread per (channel, token), out of place into y; `sconv_hist` then
// writes the pass's inputs to the ring (before y is copied over x).
extern "C" __global__ void sconv_par(ConvJobs P, float* y0, float* y1) {
    const ConvJob& J = P.j[blockIdx.z];
    float* y = blockIdx.z ? y1 : y0;
    u32 c = blockIdx.x * blockDim.x + threadIdx.x, ti = blockIdx.y;
    u32 ch = P.ch, k = P.k;
    int pos = P.info[0];
    if (c >= ch) return;
    u32 grp = J.dyn ? c / (ch / P.groups) : 0;
    const float* dr = J.dyn ? J.dyn + ((u64)ti * P.groups + grp) * k : nullptr;
    float acc = J.x[(u64)ti * ch + c];
    for (u32 l = 0; l < k; l++) {
        // Input ti - l: x, or the ring before x.
        int src = (int)ti - (int)l;
        float v = src >= 0 ? J.x[(u64)src * ch + c] : J.hist[ring_row(pos + src, k, P.ring) * ch + c];
        acc += (J.taps[(u64)l * ch + c] + (dr ? dr[l] : 0.f)) * v;
    }
    y[(u64)ti * ch + c] = acc * P.norm;
}
// The last min(t, ring) inputs of the pass into the ring.
extern "C" __global__ void sconv_hist(ConvJobs P) {
    const ConvJob& J = P.j[blockIdx.z];
    u32 ch = P.ch, t = P.t;
    int pos = P.info[0];
    u32 n = min(t, P.ring);
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)n * ch) return;
    u32 r = (u32)(i / ch), c = (u32)(i % ch);
    u32 ti = t - n + r;
    J.hist[ring_row(pos + (int)ti, P.k, P.ring) * ch + c] = J.x[(u64)ti * ch + c];
}

// Per-vector post-processing of n-value vectors (a warp each; nv per token, rows `stride` apart):
// mode 0: times `scale` (per head, [nv][n]; may be null), clipped to L2 norm `clip`, times tau of
// the position when `alpha` > 0; mode 1: RMS-normed (no gain). Up to 4 jobs per launch (job i
// from block b0).
struct VecJob {
    float* x;
    const float* scale;
    double alpha;
    u32 n, nv, stride, total, mode, b0;
    float clip, _p;
};
struct VecJobs {
    VecJob j[4];
    const int* info;
    double floor_;
    float eps;
    u32 n;
};
extern "C" __global__ void vec_post(VecJobs P) {
    u32 jb = 0;
    for (u32 i = 1; i < P.n; i++)
        if (blockIdx.x >= P.j[i].b0) jb = i;
    const VecJob& J = P.j[jb];
    u32 warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    u32 v = (blockIdx.x - J.b0) * 8 + warp;
    if (v >= J.total) return;
    u32 n = J.n, tok = v / J.nv, hh = v % J.nv;
    float* p = J.x + (u64)tok * J.stride + (u64)hh * n;
    float ss = 0.f;
    for (u32 i = lane; i < n; i += 32) {
        float a = p[i];
        if (J.mode == 0 && J.scale) a *= J.scale[hh * n + i];
        p[i] = a;
        ss += a * a;
    }
    ss = warp_sum(ss);
    float f;
    if (J.mode == 0) {
        f = J.clip / (sqrtf(ss) + 1e-6f);
        if (f >= 1.f) f = 1.f;
        if (J.alpha > 0.0) {
            double nn = (double)(P.info[0] + tok + 1) / P.floor_;
            if (nn < 1.0) nn = 1.0;
            f *= (float)(1.0 + J.alpha * log(nn));
        }
    } else {
        f = 1.f / sqrtf(ss / (float)n + P.eps);
    }
    if (f != 1.f)
        for (u32 i = lane; i < n; i += 32) p[i] *= f;
}

// Appends t rows of keys and values ([t][kw]) to a cache at rows (pos % cap). Formats: 0 f32,
// 1 f16, 2 int8 with an f32 scale per 32 (amax / 127, rounded half away from zero).
extern "C" __global__ void kv_store(const float* k, const float* v, void* kc, void* vc, float* ksc, float* vsc, u32 kw, u32 t, u32 cap, u32 fmt,
                                    const int* info) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x; // a group of 32 values
    u32 groups = kw / 32;
    if (i >= 2 * t * groups) return;
    u32 which = i / (t * groups), rem = i % (t * groups);
    u32 tok = rem / groups, gi = rem % groups;
    const float* src = (which ? v : k) + (u64)tok * kw + 32 * gi;
    u64 row = (u64)((info[0] + tok) % cap);
    u64 at = row * kw + 32 * gi;
    if (fmt == 0) {
        float* d = (float*)(which ? vc : kc) + at;
        for (int j = 0; j < 32; j++) d[j] = src[j];
    } else if (fmt == 1) {
        u16* d = (u16*)(which ? vc : kc) + at;
        for (int j = 0; j < 32; j++) {
            u16 hb;
            asm("cvt.rn.f16.f32 %0, %1;" : "=h"(hb) : "f"(src[j]));
            d[j] = hb;
        }
    } else {
        float amax = 0.f;
        for (int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(src[j]));
        float dd = amax / 127.f, inv = dd > 0.f ? 1.f / dd : 0.f;
        signed char* d = (signed char*)(which ? vc : kc) + at;
        for (int j = 0; j < 32; j++) d[j] = (signed char)roundf(src[j] * inv);
        (which ? vsc : ksc)[row * groups + gi] = dd;
    }
}

template <int F>
__device__ __forceinline__ float4 kv_load(const void* c, const float* sc, u64 row, u32 kw, u32 col) {
    if constexpr (F == 0) {
        return *(const float4*)((const float*)c + row * kw + col);
    } else if constexpr (F == 1) {
        uint2 u = *(const uint2*)((const u16*)c + row * kw + col);
        return make_float4(h2f(u.x & 0xffff), h2f(u.x >> 16), h2f(u.y & 0xffff), h2f(u.y >> 16));
    } else {
        u32 u = *(const u32*)((const signed char*)c + row * kw + col);
        float d = sc[row * (kw / 32) + col / 32];
        return make_float4(sbyte(u, 0) * d, sbyte(u, 1) * d, sbyte(u, 2) * d, sbyte(u, 3) * d);
    }
}

// Attention with relative bias over a cache: a block (4 warps) per (key/value head, token,
// split of the keys); each warp takes every 4th key, a lane 4 dims of each of the G <= 4 query
// heads of the group (head_dim 128). Writes per (token, head, split) the running max, the sum and
// the 128 accumulators. Keys of query p: [p + 1 - window, p] (window 0: all), at cache rows j % cap.
struct AttP {
    const float* q;
    const float* r;
    const void* kc;
    const void* vc;
    const float* ks;
    const float* vs;
    const float* rel;
    float* part;
    const int* info;
    u32 nh, nkv, dr, extent, window, cap, ns, qs, rs;
    float scaling;
};

template <int F, int G>
__device__ __forceinline__ void attn_body(const AttP& p) {
    __shared__ float sm[G][4][130];
    u32 g = blockIdx.x, tok = blockIdx.y, sp = blockIdx.z;
    u32 warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    u32 kw = p.nkv * 128;
    u32 pos = (u32)p.info[0] + tok;
    u32 lo = p.window && pos + 1 > p.window ? pos + 1 - p.window : 0, hi = pos + 1;
    u32 n = hi - lo;
    u32 j0 = lo + (u32)((u64)n * sp / p.ns), j1 = lo + (u32)((u64)n * (sp + 1) / p.ns);
    float4 qv[G];
    float rv[G], m[G], l[G];
    float4 acc[G];
#pragma unroll
    for (u32 h = 0; h < G; h++) {
        u32 hh = g * G + h;
        qv[h] = *(const float4*)(p.q + (u64)tok * p.qs + hh * 128 + 4 * lane);
        qv[h].x *= p.scaling;
        qv[h].y *= p.scaling;
        qv[h].z *= p.scaling;
        qv[h].w *= p.scaling;
        rv[h] = lane < p.dr ? p.r[(u64)tok * p.rs + hh * p.dr + lane] : 0.f;
        m[h] = -INFINITY;
        l[h] = 0.f;
        acc[h] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    for (u32 j = j0 + warp; j < j1; j += 4) {
        u64 row = j % p.cap;
        float4 kv = kv_load<F>(p.kc, p.ks, row, kw, g * 128 + 4 * lane);
        float4 vv = kv_load<F>(p.vc, p.vs, row, kw, g * 128 + 4 * lane);
        u32 dist = pos - j;
        float rel = lane < p.dr && dist < p.extent ? p.rel[(u64)lane * p.extent + dist] : 0.f;
        float sc[G];
#pragma unroll
        for (u32 h = 0; h < G; h++) sc[h] = qv[h].x * kv.x + qv[h].y * kv.y + qv[h].z * kv.z + qv[h].w * kv.w + rv[h] * rel;
        // The G reductions interleaved.
#pragma unroll
        for (int o = 16; o > 0; o >>= 1)
#pragma unroll
            for (u32 h = 0; h < G; h++) sc[h] += __shfl_xor_sync(0xffffffffu, sc[h], o);
#pragma unroll
        for (u32 h = 0; h < G; h++) {
            float mn = fmaxf(m[h], sc[h]);
            float a = __expf(m[h] - mn), e = __expf(sc[h] - mn);
            l[h] = l[h] * a + e;
            acc[h].x = acc[h].x * a + e * vv.x;
            acc[h].y = acc[h].y * a + e * vv.y;
            acc[h].z = acc[h].z * a + e * vv.z;
            acc[h].w = acc[h].w * a + e * vv.w;
            m[h] = mn;
        }
    }
#pragma unroll
    for (u32 h = 0; h < G; h++) {
        sm[h][warp][4 * lane] = acc[h].x;
        sm[h][warp][4 * lane + 1] = acc[h].y;
        sm[h][warp][4 * lane + 2] = acc[h].z;
        sm[h][warp][4 * lane + 3] = acc[h].w;
        if (lane == 0) {
            sm[h][warp][128] = m[h];
            sm[h][warp][129] = l[h];
        }
    }
    __syncthreads();
    // Warp h (< G) merges head h's four warps.
    if (warp < G) {
        u32 h = warp;
        float M = -INFINITY;
        for (u32 w = 0; w < 4; w++) M = fmaxf(M, sm[h][w][128]);
        float* out = p.part + (((u64)tok * p.nh + g * G + h) * p.ns + sp) * 130;
        if (M == -INFINITY) {
            if (lane == 0) {
                out[128] = -INFINITY;
                out[129] = 0.f;
            }
            return;
        }
        float L = 0.f, c[4];
        for (u32 w = 0; w < 4; w++) {
            c[w] = sm[h][w][128] == -INFINITY ? 0.f : __expf(sm[h][w][128] - M);
            L += c[w] * sm[h][w][129];
        }
        for (u32 i = lane; i < 128; i += 32) {
            float a = 0.f;
            for (u32 w = 0; w < 4; w++) a += c[w] * sm[h][w][i];
            out[i] = a;
        }
        if (lane == 0) {
            out[128] = M;
            out[129] = L;
        }
    }
}
#define ATTN(G)                                                                                     \
    extern "C" __global__ void __launch_bounds__(128) attn_f32_##G(AttP p) { attn_body<0, G>(p); } \
    extern "C" __global__ void __launch_bounds__(128) attn_f16_##G(AttP p) { attn_body<1, G>(p); } \
    extern "C" __global__ void __launch_bounds__(128) attn_q8_##G(AttP p) { attn_body<2, G>(p); }
ATTN(1)
ATTN(2)
ATTN(3)
ATTN(4)

// Prefill attention, tiled: a block (4 warps) takes QT = 8 consecutive queries of one key/value
// head (each warp 2 of them, with the G query heads of the group) and walks their keys 32 at a
// time, the keys and values staged in shared memory once for all the queries. Scores with a
// lane per key (the queries' rows broadcast from shared memory), an online softmax per row, the
// values with a lane per 4 dimensions. Writes the same partial results as `attn_*` (split
// `blockIdx.z` of the keys from the first query's lowest key to the last query's position).
#define AQT 8
#define AKT 32
template <int F, int G>
__device__ __forceinline__ void attnt_body(const AttP& p, u32 t) {
    // Keys transposed (a lane per key reads consecutive words), values, the queries' rows and
    // relative vectors: 46 KB for G = 3 (static shared memory stops at 48).
    __shared__ float Kt[128][AKT];
    __shared__ __align__(16) float Vs[AKT][128];
    __shared__ float Qs[AQT * G][128];
    __shared__ float Rs[AQT * G][16];
    u32 g = blockIdx.x, q0 = blockIdx.y * AQT, sp = blockIdx.z;
    u32 warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    u32 kw = p.nkv * 128, nq = min((u32)AQT, t - q0);
    u32 base = (u32)p.info[0];
    // This block's key range: [lowest key of its first query, position of its last query].
    u32 plast = base + q0 + nq - 1, pfirst = base + q0;
    u32 lo_all = p.window && pfirst + 1 > p.window ? pfirst + 1 - p.window : 0;
    u32 n = plast + 1 - lo_all;
    u32 j0 = lo_all + (u32)((u64)n * sp / p.ns), j1 = lo_all + (u32)((u64)n * (sp + 1) / p.ns);
    // The queries' rows (pre-scaled) and relative vectors.
    for (u32 i = threadIdx.x; i < AQT * G * 128; i += blockDim.x) {
        u32 row = i / 128, d = i % 128, qi = row / G, h = row % G;
        Qs[row][d] = qi < nq ? p.q[(u64)(q0 + qi) * p.qs + (g * G + h) * 128 + d] * p.scaling : 0.f;
    }
    for (u32 i = threadIdx.x; i < AQT * G * 16; i += blockDim.x) {
        u32 row = i / 16, z = i % 16, qi = row / G, h = row % G;
        Rs[row][z] = qi < nq && z < p.dr ? p.r[(u64)(q0 + qi) * p.rs + (g * G + h) * p.dr + z] : 0.f;
    }
    // Warp w: queries 2w, 2w + 1, rows (query, head).
    constexpr u32 RW = 2 * G;
    float m[RW], l[RW];
    float4 acc[RW];
#pragma unroll
    for (u32 r = 0; r < RW; r++) {
        m[r] = -INFINITY;
        l[r] = 0.f;
        acc[r] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    __syncthreads();
    for (u32 k0 = j0; k0 < j1; k0 += AKT) {
        // Stage keys k0 .. k0 + 31 (decoded to f32).
        for (u32 i = threadIdx.x; i < AKT * 32; i += blockDim.x) {
            u32 kk = i / 32, c4 = i % 32, j = k0 + kk;
            float4 kv = make_float4(0.f, 0.f, 0.f, 0.f), vv = kv;
            if (j < j1) {
                u64 row = j % p.cap;
                kv = kv_load<F>(p.kc, p.ks, row, kw, g * 128 + 4 * c4);
                vv = kv_load<F>(p.vc, p.vs, row, kw, g * 128 + 4 * c4);
            }
            Kt[4 * c4][kk] = kv.x, Kt[4 * c4 + 1][kk] = kv.y, Kt[4 * c4 + 2][kk] = kv.z, Kt[4 * c4 + 3][kk] = kv.w;
            *(float4*)&Vs[kk][4 * c4] = vv;
        }
        __syncthreads();
        u32 j = k0 + lane;
        // Scores: lane = key, eight partial sums per score added as a tree (one f32 chain of 128
        // products is 10x less accurate, enough to move near-tied expert choices downstream).
        float sc[RW];
        {
            float ps[RW][8];
#pragma unroll
            for (u32 r = 0; r < RW; r++)
#pragma unroll
                for (u32 u = 0; u < 8; u++) ps[r][u] = 0.f;
            for (u32 d = 0; d < 128; d += 8) {
#pragma unroll
                for (u32 u = 0; u < 8; u++) {
                    float kd = Kt[d + u][lane];
#pragma unroll
                    for (u32 r = 0; r < RW; r++) ps[r][u] += Qs[warp * RW + r][d + u] * kd;
                }
            }
#pragma unroll
            for (u32 r = 0; r < RW; r++) sc[r] = ((ps[r][0] + ps[r][1]) + (ps[r][2] + ps[r][3])) + ((ps[r][4] + ps[r][5]) + (ps[r][6] + ps[r][7]));
        }
#pragma unroll
        for (u32 r = 0; r < RW; r++) {
            u32 qi = warp * 2 + r / G;
            u32 pos = base + q0 + qi;
            u32 lo = p.window && pos + 1 > p.window ? pos + 1 - p.window : 0;
            bool ok = qi < nq && j < j1 && j <= pos && j >= lo;
            float s = -INFINITY;
            if (ok) {
                u32 dist = pos - j;
                float b = 0.f;
                if (dist < p.extent)
                    for (u32 z = 0; z < p.dr; z++) b += Rs[warp * RW + r][z] * p.rel[(u64)z * p.extent + dist];
                s = sc[r] + b;
            }
            float bm = warp_max(s);
            float mn = fmaxf(m[r], bm);
            float e = s == -INFINITY ? 0.f : __expf(s - mn);
            float a = m[r] == -INFINITY ? 0.f : __expf(m[r] - mn);
            float es = warp_sum(e);
            if (mn == -INFINITY) {
                sc[r] = 0.f;
                continue;
            }
            l[r] = l[r] * a + es;
            m[r] = mn;
            acc[r].x *= a, acc[r].y *= a, acc[r].z *= a, acc[r].w *= a;
            sc[r] = e;
        }
        // Values: lane = dims 4 lane .. 4 lane + 3; this tile's sum apart, then added to the running
        // one (a single chain over all the keys is several times less accurate).
        float4 ta[RW];
#pragma unroll
        for (u32 r = 0; r < RW; r++) ta[r] = make_float4(0.f, 0.f, 0.f, 0.f);
        for (u32 kk = 0; kk < AKT; kk++) {
            float4 vv = *(const float4*)&Vs[kk][4 * lane];
#pragma unroll
            for (u32 r = 0; r < RW; r++) {
                float pr = __shfl_sync(0xffffffffu, sc[r], kk);
                ta[r].x += pr * vv.x, ta[r].y += pr * vv.y, ta[r].z += pr * vv.z, ta[r].w += pr * vv.w;
            }
        }
#pragma unroll
        for (u32 r = 0; r < RW; r++) acc[r].x += ta[r].x, acc[r].y += ta[r].y, acc[r].z += ta[r].z, acc[r].w += ta[r].w;
        __syncthreads();
    }
#pragma unroll
    for (u32 r = 0; r < RW; r++) {
        u32 qi = warp * 2 + r / G, h = r % G;
        if (qi >= nq) continue;
        float* out = p.part + (((u64)(q0 + qi) * p.nh + g * G + h) * p.ns + sp) * 130;
        out[4 * lane] = acc[r].x, out[4 * lane + 1] = acc[r].y, out[4 * lane + 2] = acc[r].z, out[4 * lane + 3] = acc[r].w;
        if (lane == 0) {
            out[128] = m[r];
            out[129] = m[r] == -INFINITY ? 0.f : l[r];
        }
    }
}
#define ATTNT(G)                                                                                               \
    extern "C" __global__ void __launch_bounds__(128) attnt_f32_##G(AttP p, u32 t) { attnt_body<0, G>(p, t); } \
    extern "C" __global__ void __launch_bounds__(128) attnt_f16_##G(AttP p, u32 t) { attnt_body<1, G>(p, t); } \
    extern "C" __global__ void __launch_bounds__(128) attnt_q8_##G(AttP p, u32 t) { attnt_body<2, G>(p, t); }
ATTNT(1)
ATTNT(2)
ATTNT(3)

// Merges the splits with the sink (a logit with a zero value) and gates the result:
// out[tok][hh * 128 + c] = o * sigmoid(gate). A block of 128 threads per (token, head).
extern "C" __global__ void attn_merge(const float* part, const float* sink, const float* gate, float* out, u32 nh, u32 ns, u32 sink_s, u32 gate_s) {
    u32 tok = blockIdx.y, hh = blockIdx.x, c = threadIdx.x;
    const float* pp = part + ((u64)tok * nh + hh) * ns * 130;
    float s0 = sink[(u64)tok * sink_s + hh];
    float m = s0;
    for (u32 s = 0; s < ns; s++) m = fmaxf(m, pp[s * 130 + 128]);
    float l = __expf(s0 - m), o = 0.f;
    for (u32 s = 0; s < ns; s++) {
        float ms = pp[s * 130 + 128];
        if (ms == -INFINITY) continue;
        float w = __expf(ms - m);
        l += w * pp[s * 130 + 129];
        o += w * pp[s * 130 + c];
    }
    float gv = gate[(u64)tok * gate_s + hh * 128 + c];
    out[(u64)tok * nh * 128 + hh * 128 + c] = o / l / (1.f + __expf(-gv));
}

// A float as an unsigned integer in the same order (NaN below everything, -inf above it).
__device__ __forceinline__ u32 ord_key(float f) {
    u32 b = __float_as_uint(f);
    if ((b & 0x7fffffffu) > 0x7f800000u) return 0;
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// Sigmoid routing: per token (a warp), the k experts with the largest sigmoid(logit) + bias (ties
// to the lower index), in that order, weighted by their sigmoid scores renormalized, times scale.
// The choice compares integer keys (score order, then the lower index), so NaN scores (the
// lowest) cannot leave a slot empty.
extern "C" __global__ void route_k(const float* logits, const float* bias, int* ids, float* wts, u32 ne, u32 k, float scale, u32 t) {
    u32 warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    u32 tok = blockIdx.x * (blockDim.x >> 5) + warp;
    if (tok >= t) return;
    float sv[8];
    u64 key[8];
    u32 per = (ne + 31) / 32;
    for (u32 i = 0; i < per; i++) {
        u32 e = lane + 32 * i;
        float lg = e < ne ? logits[(u64)tok * ne + e] : 0.f;
        sv[i] = 1.f / (1.f + expf(-lg));
        key[i] = e < ne ? ((u64)(ord_key(sv[i] + bias[e]) + 1) << 32) | (0xffffffffu - e) : 0;
    }
    float sum = 0.f;
    for (u32 kk = 0; kk < k; kk++) {
        u64 best = 0;
        for (u32 i = 0; i < per; i++) best = key[i] > best ? key[i] : best;
        for (int o = 16; o > 0; o >>= 1) {
            u64 ob = __shfl_xor_sync(0xffffffffu, best, o);
            best = ob > best ? ob : best;
        }
        u32 bi = 0xffffffffu - (u32)best;
        float s = 0.f;
        if (bi % 32 == lane) {
            s = sv[bi / 32];
            key[bi / 32] = 0;
        }
        s = __shfl_sync(0xffffffffu, s, bi % 32);
        s = s == s ? s : 0.f;
        sum += s;
        if (lane == 0) {
            ids[tok * k + kk] = (int)bi;
            wts[tok * k + kk] = s;
        }
    }
    __syncwarp();
    if (lane < k) wts[tok * k + lane] = sum > 0.f ? wts[tok * k + lane] / sum * scale : 0.f;
}

// The MoE segments of a prefill pass (one block of 1024 threads): per expert its routed slots
// (tok * k + i), contiguous from offs[e] (the order inside a segment is not fixed; nothing depends
// on it), inv[slot] its row; tiles of up to 64 rows per expert.
extern "C" __global__ void moe_plan(const int* ids, u32 n, u32 ne, int* perm, int* inv, Tile* tiles, u32* ntiles, u32* cnt) {
    __shared__ u32 c[256], off[257], fill[256], toff[257];
    for (u32 e = threadIdx.x; e < ne; e += blockDim.x) {
        c[e] = 0;
        fill[e] = 0;
    }
    __syncthreads();
    for (u32 i = threadIdx.x; i < n; i += blockDim.x) atomicAdd(&c[ids[i]], 1u);
    __syncthreads();
    if (threadIdx.x == 0) {
        off[0] = 0;
        toff[0] = 0;
        for (u32 e = 0; e < ne; e++) {
            off[e + 1] = off[e] + c[e];
            toff[e + 1] = toff[e] + (c[e] + 63) / 64;
        }
        *ntiles = toff[ne];
    }
    __syncthreads();
    for (u32 i = threadIdx.x; i < n; i += blockDim.x) {
        u32 e = ids[i];
        u32 at = off[e] + atomicAdd(&fill[e], 1u);
        perm[at] = (int)i;
        inv[i] = (int)at;
    }
    __syncthreads();
    for (u32 e = threadIdx.x; e < ne; e += blockDim.x) {
        cnt[e] = c[e];
        for (u32 tt = 0; tt < (c[e] + 63) / 64; tt++) {
            Tile tl;
            tl.e = e;
            tl.r0 = off[e] + 64 * tt;
            tl.n = min(64u, c[e] - 64 * tt);
            tiles[toff[e] + tt] = tl;
        }
    }
}

// perm (slot ids) to the tokens the rows read, and their routing weights.
extern "C" __global__ void moe_rows(const int* perm, const float* wts, int* tok, float* wrow, u32 n, u32 k) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        tok[i] = perm[i] / (int)k;
        wrow[i] = wts[perm[i]];
    }
}

// y[tok] = sum over its k slots (in order) of d[inv[slot]].
extern "C" __global__ void moe_combine(const float* d, const int* inv, float* y, u32 h, u32 k, u32 t) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= t * h) return;
    u32 tok = i / h, c = i % h;
    float a = 0.f;
    for (u32 s = 0; s < k; s++) a += d[(u64)inv[tok * k + s] * h + c];
    y[i] = a;
}

// logits[tok][0..u] = cap tanh(l / cap), compacted from rows of v.
extern "C" __global__ void softcap_k(const float* l, float* out, u32 v, u32 u, u32 t, float cap) {
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)t * u) return;
    u64 tok = i / u, c = i % u;
    float x = l[tok * v + c];
    out[i] = cap > 0.f ? cap * tanh_acc(x / cap) : x;
}
#endif // GGML_ONLY
