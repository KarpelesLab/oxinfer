// ---- MiniMax-H3 transformer passes (metal_h3.rs, crate::gpu::dit::H3Session) ----
// Rows carry an AdaLN set (`key`: the row's (timestep, modality)); a block's table holds the six
// vectors of every set (`ld_tab` floats apart).

// y = rms(x) * w * (1 + tab[key][sc]) + tab[key][sh] (`has_mod`; else rms(x) * w) over rows of n,
// into f32 rows of n or scaled half rows (`out_half`, `rs` the inverse scales). 256 threads per
// row.
struct HRM { uint n; uint out_half; float eps; uint ld_tab; uint sc_off; uint sh_off; uint has_mod; uint _p1; };
kernel void h3_rms_mod(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]], device const float* tab [[buffer(2)]],
                       device const uint* key [[buffer(3)]], device void* out [[buffer(4)]], device float* rs [[buffer(5)]],
                       constant HRM& p [[buffer(6)]],
                       uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8];
    threadgroup float red2[8];
    device const float4* row = (device const float4*)(x + (ulong)r * p.n);
    device const float4* w4 = (device const float4*)w;
    ulong c = p.has_mod != 0 ? (ulong)key[r] * p.ld_tab : 0;
    device const float4* sc4 = (device const float4*)(tab + c + p.sc_off);
    device const float4* sh4 = (device const float4*)(tab + c + p.sh_off);
    uint n4 = p.n / 4;
    float ss = 0.f;
    for (uint i = tid; i < n4; i += 256) { float4 v = row[i]; ss += dot(v, v); }
    ss = dit_tg_sum(ss, sg, lane, red);
    float inv = 1.0f / sqrt(ss / float(p.n) + p.eps);
#define H3_RM(i) (p.has_mod != 0 ? row[i] * inv * w4[i] * (1.0f + sc4[i]) + sh4[i] : row[i] * inv * w4[i])
    if (p.out_half == 0) {
        device float4* o = (device float4*)((device float*)out + (ulong)r * p.n);
        for (uint i = tid; i < n4; i += 256) o[i] = H3_RM(i);
        return;
    }
    float m = 0.f;
    for (uint i = tid; i < n4; i += 256) { float4 v = abs(H3_RM(i)); m = max(m, max(max(v.x, v.y), max(v.z, v.w))); }
    m = dit_tg_max(m, sg, lane, red2);
    float sinv, s = dit_row_scale(m, sinv);
    device half4* o = (device half4*)((device half*)out + (ulong)r * p.n);
    for (uint i = tid; i < n4; i += 256) o[i] = half4(H3_RM(i) * s);
    if (tid == 0) rs[r] = sinv;
#undef H3_RM
}

// Attention operands of heads 64 or 128 wide (hd) from the fused projection's f32 rows (ld_src
// apart; head h of this operand at `base + h * hstride`) plus its bias (`has_bias`, indexed as
// the source row): RMS-normed with weight w (`norm`), the first 2 * half channels rotated half
// against half (channel i with i + half, by the row's `half` angles in cs / sn, `rope`), into dst
// rows of heads * hd: f32 (mode 0), half (1), or hi / lo halves (2, lo into dst_lo). Rows from
// n_valid on are zeros. One SIMD-group per (row, head).
struct HHD { uint ld_src; uint base; uint hstride; uint heads; uint norm; uint rope; uint half_; uint mode; float eps; uint n_valid; uint hd; uint has_bias; };
kernel void h3_heads(device const float* src [[buffer(0)]], device void* dst [[buffer(1)]], device const float* w [[buffer(2)]],
                     device const float* cs [[buffer(3)]], device const float* sn [[buffer(4)]], constant HHD& p [[buffer(5)]],
                     device half* dst_lo [[buffer(6)]], device const float* bias [[buffer(7)]],
                     uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float t[128];
    uint r = tg.x, h = tg.y;
    uint nj = p.hd / 32;
    ulong o = (ulong)r * p.heads * p.hd + h * p.hd;
    if (r >= p.n_valid) {
        for (uint i = lane; i < p.hd; i += 32) {
            if (p.mode == 0) ((device float*)dst)[o + i] = 0.f;
            else ((device half*)dst)[o + i] = 0.0h;
            if (p.mode == 2) dst_lo[o + i] = 0.0h;
        }
        return;
    }
    uint col = p.base + h * p.hstride;
    device const float* s = src + (ulong)r * p.ld_src + col;
    float v[4] = {0.f, 0.f, 0.f, 0.f};
    for (uint j = 0; j < nj; j++) v[j] = s[lane + 32 * j] + (p.has_bias != 0 ? bias[col + lane + 32 * j] : 0.f);
    if (p.norm != 0) {
        float ss = simd_sum(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]);
        float inv = 1.0f / sqrt(ss / float(p.hd) + p.eps);
        for (uint j = 0; j < nj; j++) v[j] *= inv * w[lane + 32 * j];
    }
    if (p.rope != 0) {
        for (uint j = 0; j < nj; j++) t[lane + 32 * j] = v[j];
        simdgroup_barrier(mem_flags::mem_threadgroup);
        device const float* c = cs + (ulong)r * p.half_;
        device const float* si = sn + (ulong)r * p.half_;
        for (uint j = 0; j < nj; j++) {
            uint i = lane + 32 * j;
            if (i < p.half_) v[j] = t[i] * c[i] - t[i + p.half_] * si[i];
            else if (i < 2 * p.half_) { uint k = i - p.half_; v[j] = t[i] * c[k] + t[k] * si[k]; }
        }
    }
    for (uint j = 0; j < nj; j++) {
        uint i = lane + 32 * j;
        if (p.mode == 0) ((device float*)dst)[o + i] = v[j];
        else {
            half hi = half(v[j]);
            ((device half*)dst)[o + i] = hi;
            if (p.mode == 2) dst_lo[o + i] = half(v[j] - float(hi));
        }
    }
}

// SwiGLU of the feed-forward's first layer's rows (2 f: gate then value with `gate_first`, else
// value then gate; plus the layer's bias with `has_bias`): value * silu(gate), into f32 rows of f
// or scaled half rows. 256 threads a row.
struct HSW { uint f; uint gate_first; uint out_half; uint has_bias; };
inline float4 h3_sw4(float4 u, float4 g) { return u * g / (1.0f + exp(-g)); }
kernel void h3_swiglu(device const float* gu [[buffer(0)]], device void* out [[buffer(1)]], device float* rs [[buffer(2)]],
                      constant HSW& p [[buffer(3)]], device const float* bias [[buffer(4)]],
                      uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8];
    device const float4* a = (device const float4*)(gu + (ulong)r * 2 * p.f);
    device const float4* b = a + p.f / 4;
    device const float4* g = p.gate_first != 0 ? a : b;
    device const float4* u = p.gate_first != 0 ? b : a;
    device const float4* ba = (device const float4*)bias;
    device const float4* bb = ba + p.f / 4;
    device const float4* bg = p.gate_first != 0 ? ba : bb;
    device const float4* bu = p.gate_first != 0 ? bb : ba;
    uint n4 = p.f / 4;
#define H3_SW(i) (p.has_bias != 0 ? h3_sw4(u[i] + bu[i], g[i] + bg[i]) : h3_sw4(u[i], g[i]))
    if (p.out_half == 0) {
        device float4* o = (device float4*)((device float*)out + (ulong)r * p.f);
        for (uint i = tid; i < n4; i += 256) o[i] = H3_SW(i);
        return;
    }
    float m = 0.f;
    for (uint i = tid; i < n4; i += 256) { float4 v = abs(H3_SW(i)); m = max(m, max(max(v.x, v.y), max(v.z, v.w))); }
    m = dit_tg_max(m, sg, lane, red);
    float sinv, s = dit_row_scale(m, sinv);
    device half4* o = (device half4*)((device half*)out + (ulong)r * p.f);
    for (uint i = tid; i < n4; i += 256) o[i] = half4(H3_SW(i) * s);
    if (tid == 0) rs[r] = sinv;
#undef H3_SW
}

// ---- The H3 corrector (metal_h3.rs, crate::gpu::dit::CorrSession) ----
// LayerNorm (weight w, bias b) of rows of n, then `* (1 + sc) + sh` with `has_mod` (one vector
// for every row), into f32 rows of n or scaled half rows. 256 threads per row.
struct CLM { uint n; uint out_half; float eps; uint has_mod; };
kernel void cor_ln_mod(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]], device const float* b [[buffer(2)]],
                       device const float* sc [[buffer(3)]], device const float* sh [[buffer(4)]],
                       device void* out [[buffer(5)]], device float* rs [[buffer(6)]], constant CLM& p [[buffer(7)]],
                       uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8];
    threadgroup float red2[8];
    threadgroup float red3[8];
    device const float4* row = (device const float4*)(x + (ulong)r * p.n);
    device const float4* w4 = (device const float4*)w;
    device const float4* b4 = (device const float4*)b;
    device const float4* sc4 = (device const float4*)sc;
    device const float4* sh4 = (device const float4*)sh;
    uint n4 = p.n / 4;
    float s = 0.f;
    for (uint i = tid; i < n4; i += 256) { float4 v = row[i]; s += v.x + v.y + v.z + v.w; }
    float mean = dit_tg_sum(s, sg, lane, red) / float(p.n);
    float ss = 0.f;
    for (uint i = tid; i < n4; i += 256) { float4 v = row[i] - mean; ss += dot(v, v); }
    float inv = 1.0f / sqrt(dit_tg_sum(ss, sg, lane, red2) / float(p.n) + p.eps);
#define COR_LN(i) (p.has_mod != 0 ? ((row[i] - mean) * inv * w4[i] + b4[i]) * (1.0f + sc4[i]) + sh4[i] : (row[i] - mean) * inv * w4[i] + b4[i])
    if (p.out_half == 0) {
        device float4* o = (device float4*)((device float*)out + (ulong)r * p.n);
        for (uint i = tid; i < n4; i += 256) o[i] = COR_LN(i);
        return;
    }
    float m = 0.f;
    for (uint i = tid; i < n4; i += 256) { float4 v = abs(COR_LN(i)); m = max(m, max(max(v.x, v.y), max(v.z, v.w))); }
    m = dit_tg_max(m, sg, lane, red3);
    float sinv, sc_ = dit_row_scale(m, sinv);
    device half4* o = (device half4*)((device half*)out + (ulong)r * p.n);
    for (uint i = tid; i < n4; i += 256) o[i] = half4(COR_LN(i) * sc_);
    if (tid == 0) rs[r] = sinv;
#undef COR_LN
}

// 64-wide attention operands: head h of row r from src (rows ld_src floats apart, the operand
// at `base`) plus its bias, consecutive channel pairs rotated by the row's angles (`rope`: hd / 2
// cosines and sines per row), into f32 rows of heads * 64. Rows from n_valid on are zeros. One
// SIMD-group per (row, head), lane l holding pair l.
struct CHD { uint ld_src; uint base; uint heads; uint rope; uint n_valid; uint _p0; uint _p1; uint _p2; };
kernel void cor_heads64(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]], device const float* bias [[buffer(2)]],
                        device const float* cs [[buffer(3)]], device const float* sn [[buffer(4)]], constant CHD& p [[buffer(5)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint r = tg.x, h = tg.y;
    device float2* d = (device float2*)(dst + (ulong)r * p.heads * 64 + h * 64) + lane;
    if (r >= p.n_valid) { *d = float2(0.f); return; }
    float2 v = *((device const float2*)(src + (ulong)r * p.ld_src + p.base + h * 64) + lane) + *((device const float2*)(bias + h * 64) + lane);
    if (p.rope != 0) {
        float c = cs[(ulong)r * 32 + lane], s = sn[(ulong)r * 32 + lane];
        v = float2(v.x * c - v.y * s, v.x * s + v.y * c);
    }
    *d = v;
}
