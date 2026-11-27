// LongCat-Video-Avatar's transformer blocks on CUDA (`longcat_avatar/cuda.rs`): compiled after
// `minimax_h3/cuda.cu`, whose GEMM, row split, value planes, SwiGLU and flash attention it uses.

// ---- GEMM with int8 weights and a scale per output row: `GemmP` plus the scales.
struct LcGemmP {
    GemmP p;
    const float* cs;
};
extern "C" __global__ void __launch_bounds__(128) lc_gemm_i8_1(LcGemmP a) { gemm_body<2, 1>(a.p, a.cs); }
extern "C" __global__ void __launch_bounds__(128) lc_gemm_i8_2(LcGemmP a) { gemm_body<2, 2>(a.p, a.cs); }

// ---- the GEMMs of the default path (first halves of the operands only), in larger tiles and
// with what follows a matrix in the block as part of its output.
// A 256-thread block computes 128 rows x 128 outputs, each of its eight warps 64 x 32 (the
// operands of its 16 tensor-core products per 16-wide K step read from shared memory by six
// `ldmatrix`). Per 32-wide K step a thread copies half a row of X and half a row of W (32 bytes
// each, or 16 int8) to shared memory; it reads them from the device's memory one step ahead,
// while the products of the step before run. Twice the tile of `gemm_body` is half its reads per
// product, which is what bounds these GEMMs on a GPU that shares the machine's memory.
// The grid is `gemm_body`'s with tiles of 128 (`cg` groups of outputs).
//
// Side paths (LoRAs, up to two): after the main product, `u B^T` for the planes `uh` of
// `u = A x` (rows `us` words apart, `ur` halves, scales `urs`) and the planes `bw` of the
// outputs' rows of B, times `uf`, added to the result.
// The output (`mode`): 0, y = v; 1, y += gate v, the gate of the row's frame at `gate_at` of
// `tab` (`stride` floats per frame; row r of the call is row `row0 + r` of frame
// `frame0 + (row0 + r) / s`; `tab` null: 1), the residual stream as y; 2, y = v silu(y), y
// holding the gate rows of a SwiGLU.
#define BM 128
#define BLD 20 // words (half2) per shared row: 16 + 4 padding

struct LcFastP {
    GemmP p;
    const float* cs;
    const u32* uh[2];
    const u32* bw[2];
    const float* urs[2];
    const float* tab;
    float uf[2];
    u32 ur, us, mode, s, frame0, stride, gate_at, row0;
};

// SIDES: with the side paths (their sums wait in registers of their own, which a GEMM without
// them does not carry).
template <int WK, bool SIDES>
__device__ __forceinline__ void gemm_fast(const LcFastP& a) {
    const GemmP& p = a.p;
    __shared__ __align__(16) u32 As[BM * BLD];
    __shared__ __align__(16) u32 Bs[BM * BLD];
    u32 tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    u32 rt = (p.rows + BM - 1) / BM;
    u32 r0 = (blockIdx.y % rt) * BM;
    u32 nr = min((u32)BM, p.rows - r0);
    u32 n0 = ((blockIdx.y / rt) * p.cg + blockIdx.x) * BM;
    if (n0 >= p.n_out) return;
    u32 half_in = p.n_in / 2;
    u32 lr = tid >> 1, lh = tid & 1;
    bool xok = lr < nr, wok = n0 + lr < p.n_out;
    const u32* xr = p.xh + (u64)(r0 + lr) * half_in + 8 * lh;
    const u8* wr = p.w + (u64)(n0 + lr) * (WK == 1 ? (u64)half_in * 4 : (u64)p.n_in) + (WK == 1 ? 32 : 16) * lh;
    float acc[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int b = 0; b < 4; b++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[i][b][c] = 0.f;
    u32 wm = (warp >> 2) * 64, wn = (warp & 3) * 32;
    const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
    uint4* as = (uint4*)(As + lr * BLD + 8 * lh);
    uint4* bs = (uint4*)(Bs + lr * BLD + 8 * lh);
    // The rows the lane names in an `ldmatrix`: of A (two tiles of rows, two of columns), of B
    // (per tile of 8 outputs its two tiles of columns).
    const u32* la = As + (wm + (lane & 15)) * BLD + 4 * (lane >> 4);
    const u32* lb = Bs + (wn + (lane & 7) + 8 * (lane >> 4)) * BLD + 4 * ((lane >> 3) & 1);
    // One 32-wide K step of the warp's products, from shared memory.
    auto products = [&]() {
#pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            u32 af[4][4], bf[4][2];
#pragma unroll
            for (int mt = 0; mt < 4; mt++) ldmatrix4(af[mt], la + 16 * mt * BLD + 8 * kk);
#pragma unroll
            for (int np = 0; np < 2; np++) {
                u32 r[4];
                ldmatrix4(r, lb + 16 * np * BLD + 8 * kk);
                bf[2 * np][0] = r[0], bf[2 * np][1] = r[1], bf[2 * np + 1][0] = r[2], bf[2 * np + 1][1] = r[3];
            }
#pragma unroll
            for (int mt = 0; mt < 4; mt++)
#pragma unroll
                for (int nt = 0; nt < 4; nt++) mma16816(acc[mt][nt], af[mt], bf[nt]);
        }
    };
    {
        u32 nb = p.n_in / 32;
        const int WN = WK == 1 ? 2 : 1;
        uint4 xa[2], wb[WN];
        auto fetch = [&](u32 b) {
#pragma unroll
            for (int i = 0; i < 2; i++) xa[i] = xok ? ((const uint4*)(xr + 16 * b))[i] : zero;
#pragma unroll
            for (int i = 0; i < WN; i++) wb[i] = wok ? ((const uint4*)(wr + (WK == 1 ? 64 : 32) * (u64)b))[i] : zero;
        };
        fetch(0);
        for (u32 b = 0; b < nb; b++) {
            as[0] = xa[0], as[1] = xa[1];
            if constexpr (WK == 1) {
                bs[0] = wb[0], bs[1] = wb[1];
            } else {
                uint4 v = wb[0];
                bs[0] = make_uint4(i8h2(v.x), i8h2(v.x >> 16), i8h2(v.y), i8h2(v.y >> 16));
                bs[1] = make_uint4(i8h2(v.z), i8h2(v.z >> 16), i8h2(v.w), i8h2(v.w >> 16));
            }
            __syncthreads();
            if (b + 1 < nb) fetch(b + 1);
            products();
            __syncthreads();
        }
    }
    // This thread's 64 results: (row, output) of accumulator [mt][nt][q].
    u32 g = lane >> 2, c4 = lane & 3;
    auto result = [&](int mt, int nt, int q, u32 rr, u32 cc) {
        float t = acc[mt][nt][q] * p.f;
        if (a.cs) t *= a.cs[cc];
        if (p.rs) t *= p.rs[r0 + rr];
        if (p.bias) t += p.bias[cc];
        return t;
    };
    auto store = [&](float t, u32 rr, u32 cc) {
        float* o = p.y + (u64)(r0 + rr) * p.ys + cc;
        if (a.mode == 1) {
            float gate = a.tab ? a.tab[(u64)(a.frame0 + (a.row0 + r0 + rr) / a.s) * a.stride + a.gate_at + cc] : 1.f;
            *o += gate * t;
        } else if (a.mode == 2) {
            float gt = *o;
            *o = t * gt / (1.f + expf(-gt));
        } else {
            *o = t;
        }
    };
    if constexpr (!SIDES) {
#pragma unroll
        for (int mt = 0; mt < 4; mt++)
#pragma unroll
            for (int nt = 0; nt < 4; nt++)
#pragma unroll
                for (int q = 0; q < 4; q++) {
                    u32 rr = wm + 16 * mt + g + 8 * (q >> 1), cc = n0 + wn + 8 * nt + 2 * c4 + (q & 1);
                    if (rr < nr && cc < p.n_out) store(result(mt, nt, q, rr, cc), rr, cc);
                }
        return;
    }
    float v[4][4][4];
#pragma unroll
    for (int mt = 0; mt < 4; mt++)
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
#pragma unroll
            for (int q = 0; q < 4; q++) {
                u32 rr = wm + 16 * mt + g + 8 * (q >> 1), cc = n0 + wn + 8 * nt + 2 * c4 + (q & 1);
                v[mt][nt][q] = rr < nr && cc < p.n_out ? result(mt, nt, q, rr, cc) : 0.f;
                acc[mt][nt][q] = 0.f;
            }
#pragma unroll
    for (int i = 0; i < 2; i++) {
        if (!a.uh[i]) continue;
        const u32* ux = a.uh[i] + (u64)(r0 + lr) * a.us + 8 * lh;
        const u32* bx = a.bw[i] + (u64)(n0 + lr) * (a.ur / 2) + 8 * lh;
        for (u32 b = 0; b < a.ur / 32; b++) {
            const uint4* pu = (const uint4*)(ux + 16 * b);
            const uint4* pb = (const uint4*)(bx + 16 * b);
            as[0] = xok ? pu[0] : zero, as[1] = xok ? pu[1] : zero;
            bs[0] = wok ? pb[0] : zero, bs[1] = wok ? pb[1] : zero;
            __syncthreads();
            products();
            __syncthreads();
        }
#pragma unroll
        for (int mt = 0; mt < 4; mt++)
#pragma unroll
            for (int nt = 0; nt < 4; nt++)
#pragma unroll
                for (int q = 0; q < 4; q++) {
                    u32 rr = wm + 16 * mt + g + 8 * (q >> 1), cc = n0 + wn + 8 * nt + 2 * c4 + (q & 1);
                    if (rr < nr && cc < p.n_out) {
                        float t = acc[mt][nt][q] * a.uf[i];
                        t *= a.urs[i][r0 + rr];
                        v[mt][nt][q] += t;
                    }
                    acc[mt][nt][q] = 0.f;
                }
    }
#pragma unroll
    for (int mt = 0; mt < 4; mt++)
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
#pragma unroll
            for (int q = 0; q < 4; q++) {
                u32 rr = wm + 16 * mt + g + 8 * (q >> 1), cc = n0 + wn + 8 * nt + 2 * c4 + (q & 1);
                if (rr < nr && cc < p.n_out) store(v[mt][nt][q], rr, cc);
            }
}
extern "C" __global__ void __launch_bounds__(256) lc_gemm_fast_h(LcFastP a) { gemm_fast<1, false>(a); }
extern "C" __global__ void __launch_bounds__(256) lc_gemm_fast_i8(LcFastP a) { gemm_fast<2, false>(a); }
extern "C" __global__ void __launch_bounds__(256) lc_gemm_sides_h(LcFastP a) { gemm_fast<1, true>(a); }
extern "C" __global__ void __launch_bounds__(256) lc_gemm_sides_i8(LcFastP a) { gemm_fast<2, true>(a); }

// ---- attention of queries `qlo..qhi` over keys `klo..khi` of the same planes.
struct LcAttP {
    AttP p;
    u32 qlo, qhi, klo, khi;
};
extern "C" __global__ void __launch_bounds__(256) lc_attn_flash_64(LcAttP a) { flash_body<64>(a.p, a.qlo, a.qhi, a.klo, a.khi); }
extern "C" __global__ void __launch_bounds__(256) lc_attn_flash_128(LcAttP a) { flash_body<128>(a.p, a.qlo, a.qhi, a.klo, a.khi); }
// The same with the first halves of the operands only (float16 queries, keys, probabilities and
// values: a third of the products), in blocks of 64 queries and, what the engine runs, of 128.
extern "C" __global__ void __launch_bounds__(256) lc_attn_flash_64_fast(LcAttP a) { flash_body<64, 1>(a.p, a.qlo, a.qhi, a.klo, a.khi); }
extern "C" __global__ void __launch_bounds__(256) lc_attn_flash_128_fast(LcAttP a) { flash_body<128, 1>(a.p, a.qlo, a.qhi, a.klo, a.khi); }
extern "C" __global__ void __launch_bounds__(512) lc_attn_flash_64_wide(LcAttP a) { flash_body<64, 1, 128>(a.p, a.qlo, a.qhi, a.klo, a.khi); }
extern "C" __global__ void __launch_bounds__(512) lc_attn_flash_128_wide(LcAttP a) { flash_body<128, 1, 128>(a.p, a.qlo, a.qhi, a.klo, a.khi); }

// ---- out[r] = layernorm(x[r]) (no weights), then (1 + scale) and + shift of the row's frame:
// `tab` holds `stride` floats per frame, the shift at `shift_at` and the scale at `scale_at`;
// row r belongs to frame `frame0 + r / s`. One block per row.
extern "C" __global__ void __launch_bounds__(256) lc_ln_mod(const float* x, const float* tab, float* out, u32 h, u32 s, u32 frame0, u32 stride, u32 shift_at, u32 scale_at) {
    __shared__ float red[256];
    u32 r = blockIdx.x, tid = threadIdx.x;
    const float* row = x + (u64)r * h;
    float a = 0.f;
    for (u32 i = tid; i < h; i += 256) a += row[i];
    float mean = block_sum256(a, red) / (float)h;
    float v = 0.f;
    for (u32 i = tid; i < h; i += 256) {
        float d = row[i] - mean;
        v += d * d;
    }
    float inv = 1.f / sqrtf(block_sum256(v, red) / (float)h + 1e-6f);
    const float* m = tab + (u64)(frame0 + r / s) * stride;
    const float* sh = m + shift_at;
    const float* sc = m + scale_at;
    float* o = out + (u64)r * h;
    for (u32 i = tid; i < h; i += 256) o[i] = (row[i] - mean) * inv * (1.f + sc[i]) + sh[i];
}

// ---- out[r] = layernorm(x[r]) w + b. One block per row.
extern "C" __global__ void __launch_bounds__(256) lc_ln_affine(const float* x, const float* w, const float* b, float* out, u32 h) {
    __shared__ float red[256];
    u32 r = blockIdx.x, tid = threadIdx.x;
    const float* row = x + (u64)r * h;
    float a = 0.f;
    for (u32 i = tid; i < h; i += 256) a += row[i];
    float mean = block_sum256(a, red) / (float)h;
    float v = 0.f;
    for (u32 i = tid; i < h; i += 256) {
        float d = row[i] - mean;
        v += d * d;
    }
    float inv = 1.f / sqrtf(block_sum256(v, red) / (float)h + 1e-6f);
    float* o = out + (u64)r * h;
    for (u32 i = tid; i < h; i += 256) o[i] = (row[i] - mean) * inv * w[i] + b[i];
}

// ---- x[r] += gate y[r], the gate at `gate_at` of the row's frame (`tab` null: x[r] += y[r]).
extern "C" __global__ void __launch_bounds__(256) lc_resid(float* x, const float* y, const float* tab, u32 h, u32 s, u32 frame0, u32 stride, u32 gate_at) {
    u32 r = blockIdx.y;
    u32 i = blockIdx.x * 256 + threadIdx.x;
    if (i >= h) return;
    float g = tab ? tab[(u64)(frame0 + r / s) * stride + gate_at + i] : 1.f;
    x[(u64)r * h + i] += g * y[(u64)r * h + i];
}

// ---- queries or keys as half planes: one warp per (row, head). The head's hd values (at
// `off0 + head * hd` of the row of `src`, rows `xs` floats apart) are RMS normed with weights `w`
// and their adjacent pairs rotated by the row's cos / sin (`hd / 2` each; null: no rotation);
// written at [head][row][hd / 2] of planes whose heads are `n` rows apart (`lo` null: the first
// halves only).
extern "C" __global__ void __launch_bounds__(32) lc_qk_prep(const float* src, u64 xs, u32 off0, const float* w, const float* cs, const float* sn, u32 hd, u32 n, u32* hi, u32* lo) {
    u32 row = blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
    const float* v = src + (u64)row * xs + off0 + (u64)head * hd;
    u32 per = hd / 32;
    float s = 0.f;
    for (u32 c = lane * per; c < (lane + 1) * per; c++) s += v[c] * v[c];
    float r = 1.f / sqrtf(warp_sum(s) / (float)hd + 1e-6f);
    u64 base = ((u64)head * n + row) * (hd / 2);
    for (u32 c = lane * per; c < (lane + 1) * per; c += 2) {
        float a = v[c] * r * w[c], b = v[c + 1] * r * w[c + 1];
        if (cs) {
            float co = cs[(u64)row * (hd / 2) + c / 2], si = sn[(u64)row * (hd / 2) + c / 2];
            float ra = a * co - b * si;
            b = b * co + a * si;
            a = ra;
        }
        if (lo)
            split_h2(a, b, hi[base + c / 2], lo[base + c / 2]);
        else
            hi[base + c / 2] = pack_h2(a, b);
    }
}
