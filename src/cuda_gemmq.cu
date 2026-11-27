// The int8 prefill GEMM of the language models' plans (`gemmq*_K`, and `gemmqp*_K` over a gate /
// up pair; appended to `inkling_cuda.cu` in the `GGML_ONLY` modules, for Q4_K, Q5_K, Q6_K and
// Q8_0).
//
// Y[m][n] (+)= sum_k X[m][k] W[n][k] with X rounded to int8 per block of 32 as the decode plans'
// int8 matvecs round it (`dq8_x`: q = round(x 127 / amax), per block d = amax / 127 and d sum q)
// and the weights as the integers the format stores. A 32-wide K step is one
// `mma.m16n8k32.s8` per 16 x 8 outputs (two `m16n8k16` for Q6_K, whose scales are per 16), and
// the step's integer sums are scaled in f32: sum += (float)C * d_x[m] * s_w[n], plus for the
// formats with a minimum d_x sum_q[m] * min_w[n]. Against the f16 GEMM (`gemmh_K`) a step is half
// the MMAs, an integer MMA with nothing carried in issues faster than an f16 one summing in f32,
// the weights' unpacking is shifts and masks, and the activations' tile is half the bytes; the
// price is the rounding of x to 8 bits a block, the decode kernels' and llama.cpp's.
//
// Stream-K as `gemmhk_K` (the grid's blocks take equal shares of every tile's steps, tiles in
// order, rows fastest), without the second pass: a block that takes a tile from a later step on
// (only its first tile) leaves its sums in `fix` and then writes `seq` to its word of `st`; the
// block that began the tile (it ends its share there) waits for the words of the blocks after
// it up to the tile's end and adds their sums in the blocks' order, so the results do not
// depend on timing, and then applies whatever follows the sums (the pair's activation). The
// blocks of a launch are all resident (one per SM: `gemmq_launch` in `cuda.rs`); a block that
// waited 2^24 polls sets `*flag` and the host declines the step.

#define QLD 48 // bytes per shared row of 32 int8: ldmatrix rows on distinct banks
struct GemmQ {
    const u8* xq;     // rows of n_in int8, n_in apart
    const float2* xd; // rows of n_in / 32: (d, d sum q)
    float* y;
    u32* st;
    u32* flag;
    u32 n_in, n_out, rows, ys, mode, seq;
};

template <int K>
__device__ __forceinline__ constexpr bool has_gemmq() {
    return K == K_Q4_K || K == K_Q5_K || K == K_Q6_K || K == K_Q8_0;
}
// Formats with a scale per 16 values: two 16-wide MMAs a step.
template <int K>
__device__ __forceinline__ constexpr bool gq_k16() {
    return K == K_Q6_K;
}
// Formats with a minimum per block.
template <int K>
__device__ __forceinline__ constexpr bool gq_min() {
    return K == K_Q4_K || K == K_Q5_K;
}

// The bytes of half `h` (16 values) of block `b` of a row, and for h = 0 the block's header.
struct GqRaw {
    uint4 hd;
    u32 a[5], c[5];
    u32 fs;
};
// 16 bytes at a 2-byte aligned p as 4 aligned words and the last two bytes: nothing past
// p + 16 is read (`ld32x8_h`'s reason: a matrix may end with its allocation's page).
__device__ __forceinline__ void gq_ld16(const u8* p, u32* r) {
    u32 odd = (u32)(u64)p & 2;
    const u32* a = (const u32*)((u64)p & ~(u64)3);
#pragma unroll
    for (int j = 0; j < 4; j++) r[j] = a[j];
    r[4] = *(const u16*)((const u8*)(a + 4) - (2 - odd));
}
template <int K>
__device__ __forceinline__ void gq_load(const u8* row, u32 b, u32 h, GqRaw& r) {
    u32 sub = b & 7;
    if constexpr (K == K_Q4_K) {
        const u8* B = row + 144 * (b >> 3);
        uint4 u = *(const uint4*)(B + 16 + 32 * (sub >> 1) + 16 * h);
        r.a[0] = u.x, r.a[1] = u.y, r.a[2] = u.z, r.a[3] = u.w;
        if (h == 0) r.hd = *(const uint4*)B;
    } else if constexpr (K == K_Q5_K) {
        const u8* B = row + 176 * (b >> 3);
        uint4 u = *(const uint4*)(B + 48 + 32 * (sub >> 1) + 16 * h);
        uint4 v = *(const uint4*)(B + 16 + 16 * h);
        r.a[0] = u.x, r.a[1] = u.y, r.a[2] = u.z, r.a[3] = u.w;
        r.c[0] = v.x, r.c[1] = v.y, r.c[2] = v.z, r.c[3] = v.w;
        if (h == 0) r.hd = *(const uint4*)B;
    } else if constexpr (K == K_Q6_K) {
        const u8* B = row + 210 * (b >> 3);
        u32 n = sub >> 2, qq = sub & 3;
        const u8* pl = B + 64 * n + 32 * (qq & 1) + 16 * h;
        r.fs = 8 * ((u32)(u64)pl & 2);
        gq_ld16(pl, r.a);
        gq_ld16(B + 128 + 32 * n + 16 * h, r.c);
        // The two halves' scale bytes and d.
        if (h == 0) r.hd.x = rd16(B + 192 + 8 * n + 2 * qq) | (rd16(B + 208) << 16);
    } else {
        const u8* B = row + 34 * b;
        const u8* p = B + 2 + 16 * h;
        r.fs = 8 * ((u32)(u64)p & 2);
        gq_ld16(p, r.a);
        if (h == 0) r.hd.x = rd16(B);
    }
}
// The 16 values as int8 (`q`), and for h = 0 the block's scales (`s`): (scale, minimum), or the
// scales of its two halves (Q6_K).
template <int K>
__device__ __forceinline__ void gq_unpack(u32 b, u32 h, const GqRaw& r, uint4& q, float2& s) {
    u32 sub = b & 7;
    if constexpr (K == K_Q4_K || K == K_Q5_K) {
        u32 sh = 4 * (sub & 1);
        u32 v[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            v[i] = (r.a[i] >> sh) & 0x0f0f0f0fu;
            if constexpr (K == K_Q5_K) v[i] |= ((r.c[i] >> sub) & 0x01010101u) << 4;
        }
        q = make_uint4(v[0], v[1], v[2], v[3]);
        if (h == 0) {
            const uint4 hd = r.hd;
            u32 sh8 = 8 * (sub & 3), sc, mn;
            if (sub < 4) {
                sc = (hd.y >> sh8) & 63;
                mn = (hd.z >> sh8) & 63;
            } else {
                u32 c = hd.w >> sh8;
                sc = (c & 15) | (((hd.y >> sh8) >> 6 & 3) << 4);
                mn = ((c >> 4) & 15) | (((hd.z >> sh8) >> 6 & 3) << 4);
            }
            s = make_float2(h2f(hd.x & 0xffff) * (float)sc, -h2f(hd.x >> 16) * (float)mn);
        }
    } else if constexpr (K == K_Q6_K) {
        u32 qq = sub & 3;
        u32 v[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            u32 ul = __funnelshift_r(r.a[i], r.a[i + 1], r.fs) >> (4 * (qq >> 1));
            u32 uh = __funnelshift_r(r.c[i], r.c[i + 1], r.fs) >> (2 * qq);
            v[i] = __vsub4((ul & 0x0f0f0f0fu) | ((uh & 0x03030303u) << 4), 0x20202020u);
        }
        q = make_uint4(v[0], v[1], v[2], v[3]);
        if (h == 0) {
            float d = h2f((u16)(r.hd.x >> 16));
            s = make_float2(d * (float)(signed char)(r.hd.x & 255), d * (float)(signed char)((r.hd.x >> 8) & 255));
        }
    } else {
        u32 v[4];
#pragma unroll
        for (int i = 0; i < 4; i++) v[i] = __funnelshift_r(r.a[i], r.a[i + 1], r.fs);
        q = make_uint4(v[0], v[1], v[2], v[3]);
        if (h == 0) s = make_float2(h2f((u16)r.hd.x), 0.f);
    }
}

__device__ __forceinline__ void gq_mma32(int* c, const u32* a, u32 b0, u32 b1) {
    int z = 0;
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(z));
}
__device__ __forceinline__ void gq_mma16(int* c, u32 a0, u32 a1, u32 b) {
    int z = 0;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a0), "r"(a1), "r"(b), "r"(z));
}
__device__ __forceinline__ void gq_cp(void* dst, const void* src, bool ok) {
    u32 a = (u32)__cvta_generic_to_shared(dst);
    int n = ok ? 8 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" ::"r"(a), "l"(src), "r"(n));
}

// The blocks' shares: block g of G takes units [k0, k1) of tiles * ceil(n_in / 32 / ku), a unit
// being `ku` 32-wide steps, over tiles of bm rows x bn outputs.
__device__ __forceinline__ void gq_range(const GemmQ& p, u32 g, u32 G, u32 bm, u32 bn, u32 ku, u64& k0, u64& k1) {
    u64 tiles = (u64)((p.n_out + bn - 1) / bn) * ((p.rows + bm - 1) / bm), W = tiles * ((p.n_in / 32 + ku - 1) / ku);
    k0 = W * g / G;
    k1 = W * (g + 1) / G;
}

// A tile is BM rows x BN weight rows on NTH / 32 warps, WNW of them along the weight rows: each
// warp (BM / (NTH / 32 / WNW)) rows x (BN / WNW) weight rows, MT x NT MMA tiles of 16 x 8.
// PAIR (a gate / up pair, `gemmqp*_K`): the first half of a warp's weight rows are rows of the
// gate matrix and the second half the same rows of the up matrix, so a thread holds both sums
// of an output and writes act(gate) * up; the tile then covers BN / 2 outputs.

// Weight row `pos` of the tile whose outputs start at n0: its bytes, and whether it exists.
template <int BN, int WNW, bool PAIR>
__device__ __forceinline__ const u8* gq_wrow(const W& w, const W& w2, const GemmQ& p, u32 n0, u32 pos, bool& ok) {
    if constexpr (PAIR) {
        constexpr u32 CW = BN / WNW, H = CW / 2;
        u32 c = pos % CW, r = n0 + (pos / CW) * H + c % H;
        ok = r < p.n_out;
        const W& m = c >= H ? w2 : w;
        return m.q + (u64)min(r, p.n_out - 1) * m.rb;
    } else {
        u32 r = n0 + pos;
        ok = r < p.n_out;
        return w.q + (u64)min(r, p.n_out - 1) * w.rb;
    }
}

// A 32-wide step of a warp: `A` / `B` the tile's activation / weight rows at the step's 32
// bytes (rows `qs` bytes apart), `Ax` / `Bx` the step's scales (the activations' with rows g and
// g + 8 of a group of 16 side by side).
template <int K, int MT, int NT>
__device__ __forceinline__ void gq_step(float (&acc)[MT][NT][4], const u8* A, const float2* Ax, const u8* B, const float2* Bx, u32 qs,
                                        u32 wm, u32 wn, u32 lane) {
    u32 g8 = lane >> 2, c4 = lane & 3;
    u32 af[MT][4];
    float4 xs[MT];
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
        u32 r = wm + 16 * mt + (lane & 7) + 8 * ((lane >> 3) & 1);
        ldsm_x4(af[mt], A + r * qs + 16 * (lane >> 4));
        xs[mt] = *(const float4*)(Ax + wm + 16 * mt + 2 * g8);
    }
#pragma unroll
    for (int np = 0; np < NT / 2; np++) {
        u32 r = wn + 16 * np + (lane & 7) + 8 * (lane >> 4);
        u32 t4[4];
        ldsm_x4(t4, B + r * qs + 16 * ((lane >> 3) & 1));
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {
            int nt = 2 * np + hh;
            // The scales of weight rows 2 c4 and 2 c4 + 1 of the 8.
            float4 ws = *(const float4*)(Bx + wn + 8 * nt + 2 * c4);
#pragma unroll
            for (int mt = 0; mt < MT; mt++) {
                float* a = acc[mt][nt];
                float d0 = xs[mt].x, d1 = xs[mt].z;
                if constexpr (gq_k16<K>()) {
                    int c0[4], c1[4];
                    gq_mma16(c0, af[mt][0], af[mt][1], t4[2 * hh]);
                    gq_mma16(c1, af[mt][2], af[mt][3], t4[2 * hh + 1]);
                    a[0] = fmaf((float)c1[0] * d0, ws.y, fmaf((float)c0[0] * d0, ws.x, a[0]));
                    a[1] = fmaf((float)c1[1] * d0, ws.w, fmaf((float)c0[1] * d0, ws.z, a[1]));
                    a[2] = fmaf((float)c1[2] * d1, ws.y, fmaf((float)c0[2] * d1, ws.x, a[2]));
                    a[3] = fmaf((float)c1[3] * d1, ws.w, fmaf((float)c0[3] * d1, ws.z, a[3]));
                } else {
                    int c[4];
                    gq_mma32(c, af[mt], t4[2 * hh], t4[2 * hh + 1]);
                    a[0] = fmaf((float)c[0] * d0, ws.x, a[0]);
                    a[1] = fmaf((float)c[1] * d0, ws.z, a[1]);
                    a[2] = fmaf((float)c[2] * d1, ws.x, a[2]);
                    a[3] = fmaf((float)c[3] * d1, ws.z, a[3]);
                    if constexpr (gq_min<K>()) {
                        float s0 = xs[mt].y, s1 = xs[mt].w;
                        a[0] = fmaf(s0, ws.y, a[0]);
                        a[1] = fmaf(s0, ws.w, a[1]);
                        a[2] = fmaf(s1, ws.y, a[2]);
                        a[3] = fmaf(s1, ws.w, a[3]);
                    }
                }
            }
        }
    }
}

// The end of a block's units of a tile. `part`: the block took the tile from a later unit on,
// and its sums go to its scratch tile, then its word; else it writes the outputs, after adding
// (`open`: the tile's units end in later blocks' shares, `kend` the tile's last unit + 1) the
// parts of those blocks, in order.
template <int BM, int BN, int NTH, int WNW, bool PAIR, int KU>
__device__ __forceinline__ void gq_finish(const W& w, const W& w2, const GemmQ& p, float (&acc)[BM / (NTH / 32 / WNW) / 16][BN / (8 * WNW)][4],
                                          u32 m0, u32 n0, bool part, bool open, float* fix, u64 kend) {
    constexpr int WMW = NTH / 32 / WNW, RPW = BM / WMW, MT = RPW / 16, NT = BN / (8 * WNW);
    constexpr int BNE = PAIR ? BN / 2 : BN;
    u32 tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, c4 = lane & 3;
    u32 wm = (warp / WNW) * RPW, wn = (warp % WNW) * (BN / WNW);
    u32 g = blockIdx.x;
    if (part) {
        float* f = fix + (u64)g * BM * BN;
#pragma unroll
        for (int mt = 0; mt < MT; mt++)
#pragma unroll
            for (int nt = 0; nt < NT; nt++)
#pragma unroll
                for (int q = 0; q < 4; q++) {
                    u32 lr = wm + 16 * mt + g8 + 8 * (q >> 1), lc = wn + 8 * nt + 2 * c4 + (q & 1);
                    f[lr * BN + lc] = acc[mt][nt][q];
                }
        __syncthreads();
        if (tid == 0) {
            __threadfence();
            *(volatile u32*)(p.st + g) = p.seq;
        }
        return;
    }
    if (open) {
        for (u32 h = g + 1; h < gridDim.x; h++) {
            u64 h0, h1;
            gq_range(p, h, gridDim.x, BM, BNE, KU, h0, h1);
            if (h0 >= kend) break;
            if (h1 <= h0) continue;
            if (tid == 0) {
                u32 n = 0;
                while (*(volatile u32*)(p.st + h) != p.seq)
                    if (++n >> 24) {
                        *p.flag = 1u;
                        break;
                    }
            }
            __syncthreads();
            const float* f = fix + (u64)h * BM * BN;
#pragma unroll
            for (int mt = 0; mt < MT; mt++)
#pragma unroll
                for (int nt = 0; nt < NT; nt++)
#pragma unroll
                    for (int q = 0; q < 4; q++) {
                        u32 lr = wm + 16 * mt + g8 + 8 * (q >> 1), lc = wn + 8 * nt + 2 * c4 + (q & 1);
                        acc[mt][nt][q] += f[lr * BN + lc];
                    }
            if (h1 >= kend) break;
        }
    }
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
#pragma unroll
        for (int nt = 0; nt < (PAIR ? NT / 2 : NT); nt++)
#pragma unroll
            for (int q = 0; q < 4; q++) {
                u32 rr = m0 + wm + 16 * mt + g8 + 8 * (q >> 1);
                u32 cc = n0 + (PAIR ? wn / 2 : wn) + 8 * nt + 2 * c4 + (q & 1);
                if (rr < p.rows && cc < p.n_out) {
                    float v = acc[mt][nt][q] * w.f;
                    float* o = p.y + (u64)rr * p.ys + cc;
                    if constexpr (PAIR) {
                        // dglu's: SiLU, or the tanh GELU (mode 1).
                        float act = p.mode ? 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v))) : v / (1.0f + expf(-v));
                        *o = act * (acc[mt][nt + NT / 2][q] * w2.f);
                    } else {
                        *o = p.mode ? *o + v : v;
                    }
                }
            }
}

// ---- `gemmq*_K` with a step at a time: steps [kb0, kb1) of the tile at rows m0 and outputs n0.
// Two stages of shared memory: the next step's activations come in by cp.async and its weight
// bytes are loaded into registers before the current step's MMAs, and unpacked after them.
template <int K, int BM, int BN, int NTH, int WNW, bool PAIR>
__device__ __forceinline__ void gemmq_seg(const W& w, const W& w2, const GemmQ& p, u32 m0, u32 n0, u32 kb0, u32 kb1, float* fix, u64 kend) {
    constexpr int WMW = NTH / 32 / WNW, RPW = BM / WMW;
    constexpr int MT = RPW / 16, NT = BN / (8 * WNW);
    constexpr int AR = BM > NTH ? BM / NTH : 1, BU = 2 * BN > NTH ? 2 * BN / NTH : 1;
    extern __shared__ __align__(16) u8 gq_dyn[];
    u8* const Aq = gq_dyn;
    float2* const Ax = (float2*)(Aq + 2 * BM * QLD);
    u8* const Bq = (u8*)(Ax + 2 * BM);
    float2* const Bx = (float2*)(Bq + 2 * BN * QLD);
    u32 tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    u32 nb = p.n_in / 32;
    auto load_a = [&](u32 b, u32 st) {
#pragma unroll
        for (int i = 0; i < AR; i++) {
            u32 r = tid + NTH * i;
            if (BM < NTH && r >= BM) break;
            bool ok = m0 + r < p.rows;
            u64 row = min(m0 + r, p.rows - 1);
            const u8* s = p.xq + row * p.n_in + 32 * b;
            u8* d = Aq + (st * BM + r) * QLD;
            cp16(d, s, ok);
            cp16(d + 16, s + 16, ok);
            gq_cp(Ax + st * BM + ((r & ~15u) | ((r & 7) << 1) | ((r >> 3) & 1)), p.xd + row * nb + b, ok);
        }
        asm volatile("cp.async.commit_group;");
    };
    GqRaw raw[BU];
    const u8* wrow[BU];
    bool bok[BU];
#pragma unroll
    for (int i = 0; i < BU; i++) {
        u32 u = tid + NTH * i;
        wrow[i] = gq_wrow<BN, WNW, PAIR>(w, w2, p, n0, min(u >> 1, (u32)BN - 1), bok[i]);
        bok[i] = bok[i] && u < 2 * BN;
    }
    auto raw_b = [&](u32 b) {
#pragma unroll
        for (int i = 0; i < BU; i++)
            if (bok[i]) gq_load<K>(wrow[i], b, (tid + NTH * i) & 1, raw[i]);
    };
    auto store_b = [&](u32 b, u32 st) {
#pragma unroll
        for (int i = 0; i < BU; i++) {
            u32 u = tid + NTH * i, br = u >> 1, h = u & 1;
            if (u >= 2 * BN) break;
            uint4 q = make_uint4(0u, 0u, 0u, 0u);
            float2 s = make_float2(0.f, 0.f);
            if (bok[i]) gq_unpack<K>(b, h, raw[i], q, s);
            *(uint4*)(Bq + (st * BN + br) * QLD + 16 * h) = q;
            if (h == 0) Bx[st * BN + br] = s;
        }
    };
    float acc[MT][NT][4];
#pragma unroll
    for (int a = 0; a < MT; a++)
#pragma unroll
        for (int b = 0; b < NT; b++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[a][b][c] = 0.f;
    u32 wm = (warp / WNW) * RPW, wn = (warp % WNW) * (BN / WNW);
    load_a(kb0, kb0 & 1);
    raw_b(kb0);
    store_b(kb0, kb0 & 1);
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    for (u32 b = kb0; b < kb1; b++) {
        u32 st = b & 1;
        bool more = b + 1 < kb1;
        if (more) {
            raw_b(b + 1);
            load_a(b + 1, st ^ 1);
        }
        gq_step<K, MT, NT>(acc, Aq + st * BM * QLD, Ax + st * BM, Bq + st * BN * QLD, Bx + st * BN, QLD, wm, wn, lane);
        if (more) store_b(b + 1, st ^ 1);
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
    }
    gq_finish<BM, BN, NTH, WNW, PAIR, 1>(w, w2, p, acc, m0, n0, kb0 != 0, kb1 < nb, fix, kend);
}
template <int K, int BM, int BN, int NTH, int WNW, bool PAIR>
__device__ __forceinline__ void gemmq_body(const W& w, const W& w2, const GemmQ& p, float* fix) {
    constexpr int BNE = PAIR ? BN / 2 : BN;
    u32 nb = p.n_in / 32, ntm = (p.rows + BM - 1) / BM;
    u64 k0, k1;
    gq_range(p, blockIdx.x, gridDim.x, BM, BNE, 1, k0, k1);
    while (k0 < k1) {
        u32 tile = (u32)(k0 / nb), kb0 = (u32)(k0 % nb);
        u32 kb1 = (u32)min((u64)nb, kb0 + (k1 - k0));
        gemmq_seg<K, BM, BN, NTH, WNW, PAIR>(w, w2, p, (tile % ntm) * BM, (tile / ntm) * BNE, kb0, kb1, fix, (u64)(tile + 1) * nb);
        k0 += kb1 - kb0;
    }
}

// ---- `gemmq*_K` over groups of KS steps (a 256-wide superblock at KS 8): one stage of shared
// memory holds a group of the tile's activations and weights, every load of the group is issued
// at once (their latencies overlap: a step with its own loads and barrier is bound by them when
// its MMAs are few, as in a tile of 64 rows), then the group's steps run back to back. The
// blocks' shares are of groups [kg0, kg1). Two threads a weight row.
template <int K, int BM, int BN, int NTH, int WNW, bool PAIR, int KS>
__device__ __forceinline__ void gemmg_seg(const W& w, const W& w2, const GemmQ& p, u32 m0, u32 n0, u32 kg0, u32 kg1, float* fix, u64 kend) {
    constexpr int WMW = NTH / 32 / WNW, RPW = BM / WMW;
    constexpr int MT = RPW / 16, NT = BN / (8 * WNW);
    constexpr int QS = 32 * KS + 16; // bytes per shared row: ldmatrix rows on distinct banks
    constexpr int TPA = NTH / BM;    // threads per row of activations
    constexpr int UB = 4;            // weight blocks loaded before they are unpacked
    static_assert(NTH == 2 * BN && NTH % BM == 0 && KS % TPA == 0 && KS % UB == 0, "tile");
    extern __shared__ __align__(16) u8 gq_dyn[];
    u8* const Aq = gq_dyn;
    float2* const Ax = (float2*)(Aq + BM * QS); // [KS][BM]
    u8* const Bq = (u8*)(Ax + KS * BM);
    float2* const Bx = (float2*)(Bq + BN * QS); // [KS][BN]
    u32 tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    u32 nb = p.n_in / 32, ng = (nb + KS - 1) / KS;
    // Activations: row tid / TPA, the blocks of the group that are tid % TPA modulo TPA.
    u32 ar = tid / TPA, ap = tid % TPA;
    bool aok = m0 + ar < p.rows;
    u64 arow = min(m0 + ar, p.rows - 1);
    const u8* xq = p.xq + arow * p.n_in;
    const float2* xd = p.xd + arow * nb;
    u32 apos = (ar & ~15u) | ((ar & 7) << 1) | ((ar >> 3) & 1);
    // Weights: row tid / 2, half tid % 2 of each block of the group.
    u32 br = tid >> 1, bh = tid & 1;
    bool bok;
    const u8* wrow = gq_wrow<BN, WNW, PAIR>(w, w2, p, n0, br, bok);
    float acc[MT][NT][4];
#pragma unroll
    for (int a = 0; a < MT; a++)
#pragma unroll
        for (int b = 0; b < NT; b++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[a][b][c] = 0.f;
    u32 wm = (warp / WNW) * RPW, wn = (warp % WNW) * (BN / WNW);
    for (u32 kg = kg0; kg < kg1; kg++) {
        u32 b0 = kg * KS, jn = min((u32)KS, nb - b0);
#pragma unroll
        for (int i = 0; i < KS / TPA; i++) {
            u32 j = ap + TPA * i;
            bool ok = aok && j < jn;
            cp16(Aq + ar * QS + 32 * j, xq + 32 * (b0 + j), ok);
            cp16(Aq + ar * QS + 32 * j + 16, xq + 32 * (b0 + j) + 16, ok);
            gq_cp(Ax + j * BM + apos, xd + min(b0 + j, nb - 1), ok);
        }
        asm volatile("cp.async.commit_group;");
#pragma unroll
        for (int i0 = 0; i0 < KS; i0 += UB) {
            GqRaw raw[UB];
#pragma unroll
            for (int i = 0; i < UB; i++)
                if (bok && i0 + i < jn) gq_load<K>(wrow, b0 + i0 + i, bh, raw[i]);
#pragma unroll
            for (int i = 0; i < UB; i++) {
                u32 j = i0 + i;
                uint4 q = make_uint4(0u, 0u, 0u, 0u);
                float2 s = make_float2(0.f, 0.f);
                if (bok && j < jn) gq_unpack<K>(b0 + j, bh, raw[i], q, s);
                *(uint4*)(Bq + br * QS + 32 * j + 16 * bh) = q;
                if (bh == 0) Bx[j * BN + br] = s;
            }
        }
        if (kg + 1 < kg1 && (kg + 2) * KS <= nb && bok) {
            // The next group's weight bytes on their way (a whole group, so nothing past the
            // row): they come from memory, not from a cache, when the matrix is first read in
            // the step.
            constexpr u32 GB = qgbytes<K>() * (KS / qgroup<K>());
            for (u32 o = 64 * bh; o < GB + 64; o += 128) prefetch_l1(wrow + (u64)GB * (kg + 1) + min(o, GB - 1));
        }
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        for (u32 j = 0; j < jn; j++) gq_step<K, MT, NT>(acc, Aq + 32 * j, Ax + j * BM, Bq + 32 * j, Bx + j * BN, QS, wm, wn, lane);
        __syncthreads();
    }
    gq_finish<BM, BN, NTH, WNW, PAIR, KS>(w, w2, p, acc, m0, n0, kg0 != 0, kg1 < ng, fix, kend);
}
template <int K, int BM, int BN, int NTH, int WNW, bool PAIR, int KS>
__device__ __forceinline__ void gemmg_body(const W& w, const W& w2, const GemmQ& p, float* fix) {
    constexpr int BNE = PAIR ? BN / 2 : BN;
    u32 ng = (p.n_in / 32 + KS - 1) / KS, ntm = (p.rows + BM - 1) / BM;
    u64 k0, k1;
    gq_range(p, blockIdx.x, gridDim.x, BM, BNE, KS, k0, k1);
    while (k0 < k1) {
        u32 tile = (u32)(k0 / ng), kg0 = (u32)(k0 % ng);
        u32 kg1 = (u32)min((u64)ng, kg0 + (k1 - k0));
        gemmg_seg<K, BM, BN, NTH, WNW, PAIR, KS>(w, w2, p, (tile % ntm) * BM, (tile / ntm) * BNE, kg0, kg1, fix, (u64)(tile + 1) * ng);
        k0 += kg1 - kg0;
    }
}

// The tiles (`GEMMQ_TILES` in `cuda.rs`): rows x weight rows, threads, warps along the weight
// rows (and steps a group); `gemmqp*_K` their gate / up pairs.
#define GEMMQ1(K, SFX, BM, BN, NTH, PER, WNW)                                                                                    \
    extern "C" __global__ void __launch_bounds__(NTH, PER) gemmq##SFX##_##K(W w, GemmQ p, float* fix) {                          \
        gemmq_body<K, BM, BN, NTH, WNW, false>(w, w, p, fix);                                                                    \
    }                                                                                                                             \
    extern "C" __global__ void __launch_bounds__(NTH, PER) gemmqp##SFX##_##K(W w, W w2, GemmQ p, float* fix) {                   \
        gemmq_body<K, BM, BN, NTH, WNW, true>(w, w2, p, fix);                                                                    \
    }
#define GEMMQG(K, SFX, BM, BN, NTH, PER, WNW, KS)                                                                                \
    extern "C" __global__ void __launch_bounds__(NTH, PER) gemmq##SFX##_##K(W w, GemmQ p, float* fix) {                          \
        gemmg_body<K, BM, BN, NTH, WNW, false, KS>(w, w, p, fix);                                                                \
    }                                                                                                                             \
    extern "C" __global__ void __launch_bounds__(NTH, PER) gemmqp##SFX##_##K(W w, W w2, GemmQ p, float* fix) {                   \
        gemmg_body<K, BM, BN, NTH, WNW, true, KS>(w, w2, p, fix);                                                                \
    }
#define GEMMQ(K)                      \
    GEMMQ1(K, , 256, 128, 256, 1, 2)  \
    GEMMQ1(K, s, 128, 128, 256, 1, 2) \
    GEMMQG(K, g, 64, 128, 256, 1, 4, 8)
#ifdef HAS_8
GEMMQ(8)
#endif
#ifdef HAS_12
GEMMQ(12)
#endif
#ifdef HAS_13
GEMMQ(13)
#endif
#ifdef HAS_14
GEMMQ(14)
#endif

// `dq8_x` (the int8 GEMM's rows: grid (blocks, rows), a warp per block of 32, the same values)
// that also sets *flag when a value is not finite: the host then declines the step.
extern "C" __global__ void __launch_bounds__(256) dq8_t(const float* x, u32 n, u32 xs, u8* xq, float2* xd, u32* flag) {
    const float* r = x + (u64)blockIdx.y * xs;
    u32 lane = threadIdx.x & 31, nb = n / 32;
    u32 b = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (b >= nb) return;
    float v = r[32 * b + lane];
    float amax = warp_max(fabsf(v));
    u32 top = __reduce_max_sync(0xffffffffu, __float_as_uint(v) & 0x7fffffffu);
    float inv = amax > 0.f ? __fdiv_rn(127.f, amax) : 0.f;
    int q = (int)roundf(__fmul_rn(v, inv));
    int s = __reduce_add_sync(0xffffffffu, q);
    u32 by = (u32)(q & 255) << (8 * (lane & 3));
    by |= __shfl_xor_sync(0xffffffffu, by, 1);
    by |= __shfl_xor_sync(0xffffffffu, by, 2);
    if ((lane & 3) == 0) ((u32*)(xq + (u64)blockIdx.y * n + 32 * b))[lane >> 2] = by;
    if (lane == 0) {
        float d = __fdiv_rn(amax, 127.f);
        xd[(u64)blockIdx.y * nb + b] = make_float2(d, __fmul_rn(d, (float)s));
        if (top >= 0x7f800000u) *flag = 1u;
    }
}
