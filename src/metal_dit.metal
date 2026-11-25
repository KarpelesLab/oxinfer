// ---- Diffusion transformer passes (metal_dit.rs, crate::gpu::dit) ----
// Row kernels feeding the matrix-unit GEMMs: their half outputs carry a power-of-two scale per
// row (`rs` gets its inverse), as `x_half` produces them.
inline float dit_row_scale(float m, thread float& inv) {
    int e = 0;
    if (m > 0.f && isfinite(m)) { frexp(m, e); e = clamp(14 - e, -100, 100); }
    inv = ldexp(1.0f, -e);
    return ldexp(1.0f, e);
}
inline float dit_tg_sum(float v, uint sg, uint lane, threadgroup float* red) {
    v = simd_sum(v);
    if (lane == 0) red[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0.f;
    for (uint i = 0; i < 8; i++) s += red[i];
    return s;
}
inline float dit_tg_max(float v, uint sg, uint lane, threadgroup float* red) {
    v = simd_max(v);
    if (lane == 0) red[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0.f;
    for (uint i = 0; i < 8; i++) s = max(s, red[i]);
    return s;
}
inline float4 dit_amax4(float4 v) { return abs(v); }

// RMS norm with weight of rows of n (x rows ld floats apart), then `* (1 + scale) + shift` with
// `has_mod`; into f32 rows of n, or with `out_half` scaled half rows. 256 threads per row.
struct DRM { uint n; uint ld; uint has_mod; uint out_half; float eps; uint _p0; uint _p1; uint _p2; };
kernel void dit_rms_mod(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                        device const float* scale [[buffer(2)]], device const float* shift [[buffer(3)]],
                        device void* out [[buffer(4)]], device float* rs [[buffer(5)]], constant DRM& p [[buffer(6)]],
                        uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8];
    threadgroup float red2[8];
    device const float4* row = (device const float4*)(x + (ulong)r * p.ld);
    device const float4* w4 = (device const float4*)w;
    device const float4* sc4 = (device const float4*)scale;
    device const float4* sh4 = (device const float4*)shift;
    uint n4 = p.n / 4;
    float ss = 0.f;
    for (uint i = tid; i < n4; i += 256) { float4 v = row[i]; ss += dot(v, v); }
    ss = dit_tg_sum(ss, sg, lane, red);
    float inv = 1.0f / sqrt(ss / float(p.n) + p.eps);
#define DIT_RM(i) (p.has_mod != 0 ? row[i] * inv * w4[i] * (1.0f + sc4[i]) + sh4[i] : row[i] * inv * w4[i])
    if (p.out_half == 0) {
        device float4* o = (device float4*)((device float*)out + (ulong)r * p.n);
        for (uint i = tid; i < n4; i += 256) o[i] = DIT_RM(i);
        return;
    }
    float m = 0.f;
    for (uint i = tid; i < n4; i += 256) { float4 v = abs(DIT_RM(i)); m = max(m, max(max(v.x, v.y), max(v.z, v.w))); }
    m = dit_tg_max(m, sg, lane, red2);
    float sinv, s = dit_row_scale(m, sinv);
    device half4* o = (device half4*)((device half*)out + (ulong)r * p.n);
    for (uint i = tid; i < n4; i += 256) o[i] = half4(DIT_RM(i) * s);
    if (tid == 0) rs[r] = sinv;
#undef DIT_RM
}

// SiTU over rows of [value | gate] (f each): (4 tanh(g / 4) sigmoid(g)) (25 tanh(h / 25)), into
// f32 rows of f or scaled half rows. 256 threads per row.
struct DSI { uint f; uint out_half; uint _p0; uint _p1; };
inline float4 dit_situ4(float4 h, float4 g) {
    float4 tg = 2.0f / (1.0f + exp(-0.5f * g)) - 1.0f;
    float4 sg = 1.0f / (1.0f + exp(-g));
    float4 th = 2.0f / (1.0f + exp(-0.08f * h)) - 1.0f;
    return 100.0f * tg * sg * th;
}
kernel void dit_situ(device const float* hg [[buffer(0)]], device void* out [[buffer(1)]], device float* rs [[buffer(2)]],
                     constant DSI& p [[buffer(3)]],
                     uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8];
    device const float4* h = (device const float4*)(hg + (ulong)r * 2 * p.f);
    device const float4* g = h + p.f / 4;
    uint n4 = p.f / 4;
    if (p.out_half == 0) {
        device float4* o = (device float4*)((device float*)out + (ulong)r * p.f);
        for (uint i = tid; i < n4; i += 256) o[i] = dit_situ4(h[i], g[i]);
        return;
    }
    float m = 0.f;
    for (uint i = tid; i < n4; i += 256) { float4 v = abs(dit_situ4(h[i], g[i])); m = max(m, max(max(v.x, v.y), max(v.z, v.w))); }
    m = dit_tg_max(m, sg, lane, red);
    float sinv, s = dit_row_scale(m, sinv);
    device half4* o = (device half4*)((device half*)out + (ulong)r * p.f);
    for (uint i = tid; i < n4; i += 256) o[i] = half4(dit_situ4(h[i], g[i]) * s);
    if (tid == 0) rs[r] = sinv;
}

// Attention operands of 128-wide heads: head h of row r from src (rows ld_src floats apart),
// optionally RMS-normed with weight w (eps), rotated (rotate-half: dims i and i + 64 by the
// row's angle i, tables cos / sin of 64 per row), into dst rows of heads * 128 (half or f32).
// One SIMD-group per (row, head), each lane holding dims lane + 32 j.
struct DHD { uint ld_src; uint norm; uint rope; uint out_half; float eps; uint heads; uint _p0; uint _p1; };
kernel void dit_heads(device const float* src [[buffer(0)]], device void* dst [[buffer(1)]], device const float* w [[buffer(2)]],
                      device const float* cs [[buffer(3)]], device const float* sn [[buffer(4)]], constant DHD& p [[buffer(5)]],
                      uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint r = tg.x, h = tg.y;
    device const float* s = src + (ulong)r * p.ld_src + h * 128;
    float a0 = s[lane], a1 = s[lane + 32], b0 = s[lane + 64], b1 = s[lane + 96];
    if (p.norm != 0) {
        float ss = simd_sum(a0 * a0 + a1 * a1 + b0 * b0 + b1 * b1);
        float inv = 1.0f / sqrt(ss / 128.0f + p.eps);
        a0 = a0 * inv * w[lane]; a1 = a1 * inv * w[lane + 32];
        b0 = b0 * inv * w[lane + 64]; b1 = b1 * inv * w[lane + 96];
    }
    if (p.rope != 0) {
        device const float* c = cs + (ulong)r * 64;
        device const float* si = sn + (ulong)r * 64;
        float c0 = c[lane], c1 = c[lane + 32], s0 = si[lane], s1 = si[lane + 32];
        float x0 = a0 * c0 - b0 * s0, y0 = b0 * c0 + a0 * s0;
        float x1 = a1 * c1 - b1 * s1, y1 = b1 * c1 + a1 * s1;
        a0 = x0; b0 = y0; a1 = x1; b1 = y1;
    }
    ulong o = (ulong)r * p.heads * 128 + h * 128;
    if (p.out_half != 0) {
        device half* d = (device half*)dst + o;
        d[lane] = half(a0); d[lane + 32] = half(a1); d[lane + 64] = half(b0); d[lane + 96] = half(b1);
    } else {
        device float* d = (device float*)dst + o;
        d[lane] = a0; d[lane + 32] = a1; d[lane + 64] = b0; d[lane + 96] = b1;
    }
}

// h[r][c] += gate[c] * y[r][c] over rows of n (y rows ld_y floats apart).
struct DGA { uint n; uint ld_y; uint _p0; uint _p1; };
kernel void dit_gated_add(device float* h [[buffer(0)]], device const float* y [[buffer(1)]], device const float* g [[buffer(2)]],
                          constant DGA& p [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    uint c = (tg.x * 256 + tid) * 4, r = tg.y;
    if (c >= p.n) return;
    device float4* hp = (device float4*)(h + (ulong)r * p.n + c);
    *hp += *(device const float4*)(g + c) * *(device const float4*)(y + (ulong)r * p.ld_y + c);
}

// dst[r][c] = src[r][c] (+ bias[c]) over rows of n.
struct DBR { uint n; uint lds; uint ldd; uint has_bias; };
kernel void dit_bias_rows(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]], device const float* b [[buffer(2)]],
                          constant DBR& p [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    uint c = tg.x * 256 + tid, r = tg.y;
    if (c >= p.n) return;
    dst[(ulong)r * p.ldd + c] = src[(ulong)r * p.lds + c] + (p.has_bias != 0 ? b[c] : 0.f);
}

// ComfyUI's ConvRot: f32 rows of n (in place) through the normalized regular Hadamard
// transform (Kronecker powers of H4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]], as
// crate::hadamard::fwht_regular) over groups of 256, then to scaled half rows. 256 threads
// (8 SIMD-groups) per row; a SIMD-group takes a group at a time, lane l holding its values
// 8 l .. 8 l + 7. Index digits (base 4): d0 = j % 4, d1 = j / 4 + 2 (l % 2), d2 = (l / 2) % 4,
// d3 = l / 8; each stage maps a quad to t - 2 x[3 - k].
struct DRT { uint n; uint _p0; uint _p1; uint _p2; };
inline float dit_h4_lanes(float x, uint b) {
    float s = x + simd_shuffle_xor(x, 1u << b);
    s += simd_shuffle_xor(s, 2u << b);
    return s - 2.0f * simd_shuffle_xor(x, 3u << b);
}
kernel void dit_rot_half(device float* x [[buffer(0)]], device half* xh [[buffer(1)]], device float* rs [[buffer(2)]],
                         constant DRT& p [[buffer(3)]],
                         uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8];
    device float* row = x + (ulong)r * p.n;
    float m = 0.f;
    for (uint g0 = sg * 256; g0 < p.n; g0 += 8 * 256) {
        device float* gp = row + g0 + lane * 8;
        float v[8];
        for (uint j = 0; j < 8; j++) v[j] = gp[j];
        // d0: within the lane, quads j = 0..3 and 4..7.
        for (uint q = 0; q < 8; q += 4) {
            float t = v[q] + v[q + 1] + v[q + 2] + v[q + 3];
            float a = v[q], b = v[q + 1], c = v[q + 2], d = v[q + 3];
            v[q] = t - 2.f * d; v[q + 1] = t - 2.f * c; v[q + 2] = t - 2.f * b; v[q + 3] = t - 2.f * a;
        }
        // d1 = j / 4 + 2 (lane % 2): the quad is {v[j], v[j ^ 4]} of this lane and lane ^ 1.
        float u[8];
        for (uint j = 0; j < 8; j++) {
            float t = v[j] + v[j ^ 4];
            t += simd_shuffle_xor(t, 1u);
            u[j] = t - 2.f * simd_shuffle_xor(v[j ^ 4], 1u);
        }
        for (uint j = 0; j < 8; j++) {
            // d2 on lane bits 1-2, d3 on lane bits 3-4; then the 1 / 16 normalization.
            float y = dit_h4_lanes(u[j], 1);
            y = dit_h4_lanes(y, 3) * 0.0625f;
            gp[j] = y;
            m = max(m, abs(y));
        }
    }
    m = dit_tg_max(m, sg, lane, red);
    threadgroup_barrier(mem_flags::mem_device);
    float sinv, s = dit_row_scale(m, sinv);
    device const float4* row4 = (device const float4*)row;
    device half4* o = (device half4*)(xh + (ulong)r * p.n);
    for (uint i = tid; i < p.n / 4; i += 256) o[i] = half4(row4[i] * s);
    if (tid == 0) rs[r] = sinv;
}

// ---- FLUX.1 passes (metal_dit.rs, crate::gpu::dit::FluxSession) ----
// Layer norm without weights of rows of n (eps), then `* (1 + scale) + shift`, into scaled half
// rows. 256 threads per row.
struct FLN { uint n; float eps; uint _p0; uint _p1; };
kernel void flux_ln_mod(device const float* x [[buffer(0)]], device const float* scale [[buffer(1)]],
                        device const float* shift [[buffer(2)]], device half* out [[buffer(3)]], device float* rs [[buffer(4)]],
                        constant FLN& p [[buffer(5)]],
                        uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8];
    threadgroup float red2[8];
    threadgroup float red3[8];
    device const float4* row = (device const float4*)(x + (ulong)r * p.n);
    device const float4* sc4 = (device const float4*)scale;
    device const float4* sh4 = (device const float4*)shift;
    uint n4 = p.n / 4;
    float s = 0.f;
    for (uint i = tid; i < n4; i += 256) { float4 v = row[i]; s += v.x + v.y + v.z + v.w; }
    float mean = dit_tg_sum(s, sg, lane, red) / float(p.n);
    float ss = 0.f;
    for (uint i = tid; i < n4; i += 256) { float4 v = row[i] - mean; ss += dot(v, v); }
    float inv = 1.0f / sqrt(dit_tg_sum(ss, sg, lane, red2) / float(p.n) + p.eps);
#define FLUX_LN(i) ((row[i] - mean) * inv * (1.0f + sc4[i]) + sh4[i])
    float m = 0.f;
    for (uint i = tid; i < n4; i += 256) { float4 v = abs(FLUX_LN(i)); m = max(m, max(max(v.x, v.y), max(v.z, v.w))); }
    m = dit_tg_max(m, sg, lane, red3);
    float sinv, sc = dit_row_scale(m, sinv);
    device half4* o = (device half4*)(out + (ulong)r * p.n);
    for (uint i = tid; i < n4; i += 256) o[i] = half4(FLUX_LN(i) * sc);
    if (tid == 0) rs[r] = sinv;
#undef FLUX_LN
}

// Attention operands of 128-wide heads: head h of row r from src (rows ld_src floats apart) plus
// a bias, RMS-normed with weight w (with `norm`), rotated by interleaved pairs (dims 2i, 2i + 1 by
// the row's angle i, tables of 64 per row; with `rope`), into dst rows of heads * 128 (half or
// f32, or with out_half 2 as hi / lo halves into dst and dst_lo). Rows below `split` take weights / biases A, the others B and come from source row
// r + gap (the joint rows are compact, the source's text rows padded to a GEMM tile). One
// SIMD-group per (row, head), lane l holding dims 4 l .. 4 l + 3.
struct FHD { uint ld_src; uint norm; uint rope; uint out_half; float eps; uint heads; uint split; uint has_bias;
             uint gap; uint _p0; uint _p1; uint _p2; };
kernel void flux_heads(device const float* src [[buffer(0)]], device void* dst [[buffer(1)]],
                       device const float* wa [[buffer(2)]], device const float* wb [[buffer(3)]],
                       device const float* ba [[buffer(4)]], device const float* bb [[buffer(5)]],
                       device const float* cs [[buffer(6)]], device const float* sn [[buffer(7)]], constant FHD& p [[buffer(8)]],
                       device half* dst_lo [[buffer(9)]], uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint r = tg.x, h = tg.y;
    bool a = r < p.split;
    float4 x = *(device const float4*)(src + (ulong)(a ? r : r + p.gap) * p.ld_src + h * 128 + lane * 4);
    if (p.has_bias != 0) x += *(device const float4*)((a ? ba : bb) + h * 128 + lane * 4);
    if (p.norm != 0) {
        float ss = simd_sum(dot(x, x));
        float inv = 1.0f / sqrt(ss / 128.0f + p.eps);
        x = x * inv * *(device const float4*)((a ? wa : wb) + lane * 4);
    }
    if (p.rope != 0) {
        float2 c = *(device const float2*)(cs + (ulong)r * 64 + lane * 2);
        float2 s = *(device const float2*)(sn + (ulong)r * 64 + lane * 2);
        x = float4(x.x * c.x - x.y * s.x, x.y * c.x + x.x * s.x, x.z * c.y - x.w * s.y, x.w * c.y + x.z * s.y);
    }
    ulong o = (ulong)r * p.heads * 128 + h * 128 + lane * 4;
    if (p.out_half == 2) {
        // hi / lo halves (dit_flash128_hilo).
        half4 hi = half4(x);
        *(device half4*)((device half*)dst + o) = hi;
        *(device half4*)(dst_lo + o) = half4(x - float4(hi));
    } else if (p.out_half != 0) *(device half4*)((device half*)dst + o) = half4(x);
    else *(device float4*)((device float*)dst + o) = x;
}

// GELU-tanh of rows of n (+ bias) into scaled half rows. 256 threads per row.
inline float4 flux_gelu4(float4 x) { return x / (1.0f + exp(-1.5957691f * (x + 0.044715f * x * x * x))); }
struct FGE { uint n; uint has_bias; uint _p0; uint _p1; };
kernel void flux_gelu_half(device const float* x [[buffer(0)]], device const float* b [[buffer(1)]], device half* out [[buffer(2)]],
                           device float* rs [[buffer(3)]], constant FGE& p [[buffer(4)]],
                           uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8];
    device const float4* row = (device const float4*)(x + (ulong)r * p.n);
    device const float4* b4 = (device const float4*)b;
    uint n4 = p.n / 4;
#define FLUX_GE(i) flux_gelu4(p.has_bias != 0 ? row[i] + b4[i] : row[i])
    float m = 0.f;
    for (uint i = tid; i < n4; i += 256) { float4 v = abs(FLUX_GE(i)); m = max(m, max(max(v.x, v.y), max(v.z, v.w))); }
    m = dit_tg_max(m, sg, lane, red);
    float sinv, s = dit_row_scale(m, sinv);
    device half4* o = (device half4*)(out + (ulong)r * p.n);
    for (uint i = tid; i < n4; i += 256) o[i] = half4(FLUX_GE(i) * s);
    if (tid == 0) rs[r] = sinv;
#undef FLUX_GE
}

// Single-block projection input: row r of [a (rows of na) | gelu_tanh(g + bias) (rows of ng)]
// into scaled half rows of na + ng. 256 threads per row.
struct FCT { uint na; uint ng; uint has_bias; uint _p0; };
kernel void flux_cat_half(device const float* a [[buffer(0)]], device const float* g [[buffer(1)]], device const float* b [[buffer(2)]],
                          device half* out [[buffer(3)]], device float* rs [[buffer(4)]], constant FCT& p [[buffer(5)]],
                          uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8];
    device const float4* ar = (device const float4*)(a + (ulong)r * p.na);
    device const float4* gr = (device const float4*)(g + (ulong)r * p.ng);
    device const float4* b4 = (device const float4*)b;
    uint na4 = p.na / 4, ng4 = p.ng / 4;
#define FLUX_CG(i) flux_gelu4(p.has_bias != 0 ? gr[i] + b4[i] : gr[i])
    float m = 0.f;
    for (uint i = tid; i < na4; i += 256) { float4 v = abs(ar[i]); m = max(m, max(max(v.x, v.y), max(v.z, v.w))); }
    for (uint i = tid; i < ng4; i += 256) { float4 v = abs(FLUX_CG(i)); m = max(m, max(max(v.x, v.y), max(v.z, v.w))); }
    m = dit_tg_max(m, sg, lane, red);
    float sinv, s = dit_row_scale(m, sinv);
    device half4* o = (device half4*)(out + (ulong)r * (p.na + p.ng));
    for (uint i = tid; i < na4; i += 256) o[i] = half4(ar[i] * s);
    for (uint i = tid; i < ng4; i += 256) o[na4 + i] = half4(FLUX_CG(i) * s);
    if (tid == 0) rs[r] = sinv;
#undef FLUX_CG
}

// h[r][c] += gate[c] * (y[r][c] + bias[c]) over rows of n.
struct FGA { uint n; uint has_bias; uint _p0; uint _p1; };
kernel void flux_gated_add(device float* h [[buffer(0)]], device const float* y [[buffer(1)]], device const float* g [[buffer(2)]],
                           device const float* b [[buffer(3)]], constant FGA& p [[buffer(4)]],
                           uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    uint c = (tg.x * 256 + tid) * 4, r = tg.y;
    if (c >= p.n) return;
    device float4* hp = (device float4*)(h + (ulong)r * p.n + c);
    float4 v = *(device const float4*)(y + (ulong)r * p.n + c);
    if (p.has_bias != 0) v += *(device const float4*)(b + c);
    *hp += *(device const float4*)(g + c) * v;
}
