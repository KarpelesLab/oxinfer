// LongCat-Video-Avatar's transformer blocks on CUDA (`longcat_avatar/cuda.rs`): compiled after
// `minimax_h3/cuda.cu`, whose GEMM, row split, value planes, SwiGLU and flash attention it uses.

// ---- GEMM with int8 weights and a scale per output row: `GemmP` plus the scales.
struct LcGemmP {
    GemmP p;
    const float* cs;
};
extern "C" __global__ void __launch_bounds__(128) lc_gemm_i8_1(LcGemmP a) { gemm_body<2, 1>(a.p, a.cs); }
extern "C" __global__ void __launch_bounds__(128) lc_gemm_i8_2(LcGemmP a) { gemm_body<2, 2>(a.p, a.cs); }

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
