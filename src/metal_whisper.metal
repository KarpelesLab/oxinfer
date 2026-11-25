// ---- Whisper encoder passes (metal_whisper.rs, crate::gpu::whisper) ----
// Row kernels around the matrix-unit GEMMs: their half outputs carry a power-of-two scale per row
// (`rs` gets its inverse), as `x_half` produces them. 256 threads per row.
#if MMA_UNITS
inline float wh_row_scale(float m, thread float& inv) {
    int e = 0;
    if (m > 0.f && isfinite(m)) { frexp(m, e); e = clamp(14 - e, -100, 100); }
    inv = ldexp(1.0f, -e);
    return ldexp(1.0f, e);
}
inline float wh_tg_sum(float v, uint sg, uint lane, threadgroup float* red) {
    v = simd_sum(v);
    if (lane == 0) red[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0.f;
    for (uint i = 0; i < 8; i++) s += red[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return s;
}
inline float wh_tg_max(float v, uint sg, uint lane, threadgroup float* red) {
    v = simd_max(v);
    if (lane == 0) red[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0.f;
    for (uint i = 0; i < 8; i++) s = max(s, red[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return s;
}
// Exact GELU with erf from Abramowitz & Stegun 7.1.26, as the CPU's `gelu_erf`.
inline float wh_gelu(float x) {
    float z = x * 0.70710678f;
    float a = fabs(z);
    float t = 1.0f / (1.0f + 0.3275911f * a);
    float poly = t * (0.2548296f + t * (-0.28449674f + t * (1.4214137f + t * (-1.4531521f + t * 1.0614054f))));
    float e = 1.0f - poly * exp(-a * a);
    return 0.5f * x * (1.0f + copysign(e, z));
}
inline float4 wh_gelu4(float4 v) { return float4(wh_gelu(v.x), wh_gelu(v.y), wh_gelu(v.z), wh_gelu(v.w)); }
inline float wh_amax4(float4 v) { v = abs(v); return max(max(v.x, v.y), max(v.z, v.w)); }

// conv1's unfolded rows: row (window b, frame i) holds mel frames i - 1, i, i + 1 (zeros outside
// the window) then zeros up to k1, as scaled half. Rows past the window's frames are zeros.
struct WC1 { uint frames; uint nm; uint k1; uint rows_per_win; };
kernel void wh_cols1(device const float* mel [[buffer(0)]], device half* xh [[buffer(1)]], device float* rs [[buffer(2)]],
                     constant WC1& p [[buffer(3)]],
                     uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8];
    uint b = r / p.rows_per_win, i = r % p.rows_per_win;
    device half* out = xh + (ulong)r * p.k1;
    device const float* m = mel + (ulong)b * p.frames * p.nm;
    float mx = 0.f;
    for (uint c = tid; c < p.k1; c += 256) {
        uint j = c / p.nm, ch = c % p.nm;
        int src = int(i) + int(j) - 1;
        float v = (i < p.frames && j < 3 && src >= 0 && uint(src) < p.frames) ? m[(ulong)src * p.nm + ch] : 0.f;
        mx = max(mx, fabs(v));
    }
    mx = wh_tg_max(mx, sg, lane, red);
    float inv, s = wh_row_scale(mx, inv);
    for (uint c = tid; c < p.k1; c += 256) {
        uint j = c / p.nm, ch = c % p.nm;
        int src = int(i) + int(j) - 1;
        float v = (i < p.frames && j < 3 && src >= 0 && uint(src) < p.frames) ? m[(ulong)src * p.nm + ch] : 0.f;
        out[c] = half(v * s);
    }
    if (tid == 0) rs[r] = inv;
}

// conv2's unfolded rows (kernel 3, stride 2): row (window b, position i) holds gelu(y1 + bias) of
// conv1 frames 2i - 1, 2i, 2i + 1 (zeros outside the window), as scaled half rows of 3 d.
struct WC2 { uint d; uint frames; uint n; uint rows1; uint rows; uint _p0; uint _p1; uint _p2; };
kernel void wh_cols2(device const float* y1 [[buffer(0)]], device const float* bias [[buffer(1)]],
                     device half* xh [[buffer(2)]], device float* rs [[buffer(3)]], constant WC2& p [[buffer(4)]],
                     uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8];
    uint b = r / p.rows, i = r % p.rows;
    uint d4 = p.d / 4;
    device half4* out = (device half4*)(xh + (ulong)r * 3 * p.d);
    device const float4* b4 = (device const float4*)bias;
    float mx = 0.f;
    for (uint c = tid; c < 3 * d4; c += 256) {
        uint j = c / d4, ch = c % d4;
        int src = 2 * int(i) + int(j) - 1;
        float4 v = 0.f;
        if (i < p.n && src >= 0 && uint(src) < p.frames)
            v = wh_gelu4(((device const float4*)(y1 + ((ulong)b * p.rows1 + src) * p.d))[ch] + b4[ch]);
        mx = max(mx, wh_amax4(v));
    }
    mx = wh_tg_max(mx, sg, lane, red);
    float inv, s = wh_row_scale(mx, inv);
    for (uint c = tid; c < 3 * d4; c += 256) {
        uint j = c / d4, ch = c % d4;
        int src = 2 * int(i) + int(j) - 1;
        float4 v = 0.f;
        if (i < p.n && src >= 0 && uint(src) < p.frames)
            v = wh_gelu4(((device const float4*)(y1 + ((ulong)b * p.rows1 + src) * p.d))[ch] + b4[ch]);
        out[c] = half4(v * s);
    }
    if (tid == 0) rs[r] = inv;
}

// Residual step and LayerNorm of row r of x (d wide): `mode` 0: x += bias; 1: x = gelu(x + bias)
// + pos[i] (i the row's position in its window, rows past n set to 0). Then with `ln` the row's
// LayerNorm (weight w, bias lb) into scaled half rows (`out_half`) or f32 rows of `out`.
struct WRL { uint d; uint mode; uint ln; uint out_half; uint rows; uint n; float eps; uint _p; };
kernel void wh_res_ln(device float* x [[buffer(0)]], device const float* bias [[buffer(1)]],
                      device const float* pos [[buffer(2)]], device const float* w [[buffer(3)]],
                      device const float* lb [[buffer(4)]], device void* out [[buffer(5)]], device float* rs [[buffer(6)]],
                      constant WRL& p [[buffer(7)]],
                      uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8];
    uint d4 = p.d / 4;
    device float4* xr = (device float4*)(x + (ulong)r * p.d);
    device const float4* b4 = (device const float4*)bias;
    uint i = r % p.rows;
    float sum = 0.f;
    for (uint c = tid; c < d4; c += 256) {
        float4 v;
        if (p.mode == 1) {
            v = i < p.n ? wh_gelu4(xr[c] + b4[c]) + ((device const float4*)(pos + (ulong)i * p.d))[c] : float4(0.f);
        } else {
            v = xr[c] + b4[c];
        }
        xr[c] = v;
        sum += v.x + v.y + v.z + v.w;
    }
    if (p.ln == 0) return;
    float mean = wh_tg_sum(sum, sg, lane, red) / float(p.d);
    float sq = 0.f;
    for (uint c = tid; c < d4; c += 256) { float4 v = xr[c] - mean; sq += dot(v, v); }
    float inv = 1.0f / sqrt(wh_tg_sum(sq, sg, lane, red) / float(p.d) + p.eps);
    device const float4* w4 = (device const float4*)w;
    device const float4* lb4 = (device const float4*)lb;
    if (p.out_half == 0) {
        device float4* o = (device float4*)((device float*)out + (ulong)r * p.d);
        for (uint c = tid; c < d4; c += 256) o[c] = (xr[c] - mean) * inv * w4[c] + lb4[c];
        return;
    }
    float mx = 0.f;
    for (uint c = tid; c < d4; c += 256) mx = max(mx, wh_amax4((xr[c] - mean) * inv * w4[c] + lb4[c]));
    mx = wh_tg_max(mx, sg, lane, red);
    float sinv, s = wh_row_scale(mx, sinv);
    device half4* o = (device half4*)((device half*)out + (ulong)r * p.d);
    for (uint c = tid; c < d4; c += 256) o[c] = half4(((xr[c] - mean) * inv * w4[c] + lb4[c]) * s);
    if (tid == 0) rs[r] = sinv;
}

// gelu(y + bias) of rows of n into scaled half rows (the feed-forward's hidden layer).
struct WGH { uint n; uint _p0; uint _p1; uint _p2; };
kernel void wh_gelu_half(device const float* y [[buffer(0)]], device const float* bias [[buffer(1)]],
                         device half* xh [[buffer(2)]], device float* rs [[buffer(3)]], constant WGH& p [[buffer(4)]],
                         uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8];
    uint n4 = p.n / 4;
    device const float4* yr = (device const float4*)(y + (ulong)r * p.n);
    device const float4* b4 = (device const float4*)bias;
    float mx = 0.f;
    for (uint c = tid; c < n4; c += 256) mx = max(mx, wh_amax4(wh_gelu4(yr[c] + b4[c])));
    mx = wh_tg_max(mx, sg, lane, red);
    float inv, s = wh_row_scale(mx, inv);
    device half4* o = (device half4*)(xh + (ulong)r * p.n);
    for (uint c = tid; c < n4; c += 256) o[c] = half4(wh_gelu4(yr[c] + b4[c]) * s);
    if (tid == 0) rs[r] = inv;
}

// y + bias (has_bias) of rows of n into half (out_half) or f32 values, for the attention inputs.
struct WBH { uint n; uint has_bias; uint out_half; uint _p; };
kernel void wh_bias(device const float* y [[buffer(0)]], device const float* bias [[buffer(1)]],
                    device void* out [[buffer(2)]], constant WBH& p [[buffer(3)]],
                    uint2 g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    uint c = g.x * 256 + tid, n4 = p.n / 4;
    if (c >= n4) return;
    ulong r = g.y;
    float4 v = ((device const float4*)(y + r * p.n))[c];
    if (p.has_bias != 0) v += ((device const float4*)bias)[c];
    if (p.out_half != 0) ((device half4*)((device half*)out + r * p.n))[c] = half4(v);
    else ((device float4*)((device float*)out + r * p.n))[c] = v;
}

// Fused attention for 64-wide heads, no mask, on the matrix units: flash_attn128_body's algorithm
// with half the head width (q / k / v of type T in the model's [rows][heads * 64] layout, rows
// padded to 64 with finite values, f32 output).
struct WFA { uint tq; uint tk; uint d; float scale; };
template <typename T>
inline void wh_flash64_body(device const T* q, device const T* k, device const T* v, device float* o, constant WFA& p,
                            threadgroup float* S, threadgroup T* P, threadgroup float* A, threadgroup float* L,
                            threadgroup uint* RS, uint2 tg, uint tid) {
    uint q0 = tg.x * 64, h0 = tg.y * 64;
    uint tqp = (p.tq + 63) / 64 * 64, tkp = (p.tk + 63) / 64 * 64;
    tensor<device T, dextents<int32_t, 2>, tensor_inline> Qt((device T*)q, dextents<int32_t, 2>(p.d, tqp));
    tensor<device T, dextents<int32_t, 2>, tensor_inline> Kt((device T*)k, dextents<int32_t, 2>(p.d, tkp));
    tensor<device T, dextents<int32_t, 2>, tensor_inline> Vt((device T*)v, dextents<int32_t, 2>(p.d, tkp));
    tensor<threadgroup float, extents<int32_t, 64, 64>, tensor_inline> St(S, extents<int32_t, 64, 64>());
    tensor<threadgroup T, extents<int32_t, 64, 64>, tensor_inline> Pt(P, extents<int32_t, 64, 64>());
    constexpr auto ds = matmul2d_descriptor(64, 64, 64, false, true, false);
    constexpr auto dov = matmul2d_descriptor(64, 64, 64, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<ds, execution_simdgroups<4>> ops;
    matmul2d<dov, execution_simdgroups<4>> opo;
    auto qa = Qt.template slice<64, 64>(h0, q0);
    auto vb0 = Vt.template slice<64, 64>(h0, 0);
    auto co = opo.template get_destination_cooperative_tensor<decltype(Pt), decltype(vb0), float>();
    zero(co);
    uint r = tid / 2, part = tid % 2;
    float m = -INFINITY, l = 0.0f;
    threadgroup uint& rescale = RS[0];
    if (tid == 0) rescale = 0;
    for (uint kb = 0; kb < p.tk; kb += 64) {
        auto kt = Kt.template slice<64, 64>(h0, kb);
        auto cs = ops.template get_destination_cooperative_tensor<decltype(qa), decltype(kt), float>();
        zero(cs);
        ops.run(qa, kt, cs);
        cs.store(St);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float x[32];
        float bm = -INFINITY;
        for (uint j = 0; j < 32; j++) {
            uint col = part * 32 + j;
            x[j] = kb + col < p.tk ? S[r * 64 + col] * p.scale : -INFINITY;
            bm = max(bm, x[j]);
        }
        bm = max(bm, simd_shuffle_xor(bm, 1));
        // Lazy rescaling (as flash_attn128_body): the running max moves only past +8, so most
        // key blocks skip rescaling the output.
        bool grow = bm > m + 8.0f;
        float mn = grow ? bm : m;
        float alpha = grow ? exp(m - mn) : 1.0f;
        if (grow) rescale = 1;
        float ps = 0.0f;
        for (uint j = 0; j < 32; j++) { float e = exp(x[j] - mn); P[r * 64 + part * 32 + j] = T(e); ps += e; }
        ps += simd_shuffle_xor(ps, 1);
        l = l * alpha + ps;
        m = mn;
        if (part == 0) A[r] = alpha;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (rescale != 0) {
#pragma unroll
            for (uint16_t i = 0; i < co.get_capacity(); ++i)
                if (co.is_valid_element(i)) { auto idx = co.get_multidimensional_index(i); co[i] *= A[idx[1]]; }
        }
        auto vt = Vt.template slice<64, 64>(h0, kb);
        opo.run(Pt, vt, co);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) rescale = 0;
    }
    if (part == 0) L[r] = 1.0f / l;
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint16_t i = 0; i < co.get_capacity(); ++i)
        if (co.is_valid_element(i)) {
            auto idx = co.get_multidimensional_index(i);
            uint row = q0 + idx[1];
            if (row < p.tq) o[(ulong)row * p.d + h0 + idx[0]] = co[i] * L[idx[1]];
        }
}
kernel void wh_flash64(device const half* q [[buffer(0)]], device const half* k [[buffer(1)]],
                       device const half* v [[buffer(2)]], device float* o [[buffer(3)]], constant WFA& p [[buffer(4)]],
                       uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float S[64 * 64];
    threadgroup half P[64 * 64];
    threadgroup float A[64];
    threadgroup float L[64];
    threadgroup uint RS[1];
    wh_flash64_body<half>(q, k, v, o, p, S, P, A, L, RS, tg, tid);
}
kernel void wh_flash64_f32(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]],
                           device const float* v [[buffer(2)]], device float* o [[buffer(3)]], constant WFA& p [[buffer(4)]],
                           uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float S[64 * 64];
    threadgroup float A[64];
    threadgroup float L[64];
    threadgroup uint RS[1];
    wh_flash64_body<float>(q, k, v, o, p, S, S, A, L, RS, tg, tid);
}

// ---- Decoder steps ----
// gelu(y + bias) in place over rows of n.
struct WDG { uint n; uint _p0; uint _p1; uint _p2; };
kernel void whd_gelu(device float* y [[buffer(0)]], device const float* bias [[buffer(1)]], constant WDG& p [[buffer(2)]],
                     uint2 g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    uint c = g.x * 256 + tid, n4 = p.n / 4;
    if (c >= n4) return;
    device float4* r = (device float4*)(y + (ulong)g.y * p.n);
    r[c] = wh_gelu4(r[c] + ((device const float4*)bias)[c]);
}

// One query row's attention over n cached positions in chunks of 128, one threadgroup (128
// threads) per chunk and 64-wide head: q = (q + bias) * scale, keys and values of type T in rows
// d apart. Each writes its chunk's max score, exponential sum and weighted values (68 floats at
// part + (head * chunks + chunk) * 68); `whd_attn_merge` combines the chunks.
// With `fuse`, position n - 1 is the row's own: its key and value come from `nk` / `nv` (plus
// `bk` / `bv`) and are stored into the cache by the chunk holding it (`k` / `v` then f32).
// With one chunk the attention output goes straight to `o`.
struct WDP { uint n; uint d; float scale; uint chunks; uint fuse; uint _p0; uint _p1; uint _p2; };
template <typename T>
inline void whd_part_body(device const float* q, device const float* qb, device T* k, device T* v, device float* part,
                          constant WDP& p, uint2 g, uint tid, uint sg, uint lane, threadgroup float* qs, threadgroup float* S,
                          threadgroup float* red, threadgroup float4* acc, device const float* nk, device const float* nv,
                          device const float* bk, device const float* bv, device float* o, threadgroup float* kn,
                          threadgroup float* vn) {
    uint h = g.y, c0 = g.x * 128;
    uint own = p.fuse != 0 ? p.n - 1 : 0xffffffffu;
    bool mine = own >= c0 && own < c0 + 128;
    if (tid < 64) {
        qs[tid] = (q[h * 64 + tid] + qb[h * 64 + tid]) * p.scale;
        if (mine) {
            float kv = nk[h * 64 + tid] + bk[h * 64 + tid], vv = nv[h * 64 + tid] + bv[h * 64 + tid];
            kn[tid] = kv;
            vn[tid] = vv;
            k[(ulong)own * p.d + h * 64 + tid] = T(kv);
            v[(ulong)own * p.d + h * 64 + tid] = T(vv);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint i = c0 + tid;
    float s = -INFINITY;
    if (i == own) {
        float4 s4 = 0.f;
        for (uint c = 0; c < 16; c++) s4 = fma(((threadgroup const float4*)kn)[c], ((threadgroup const float4*)qs)[c], s4);
        s = s4.x + s4.y + s4.z + s4.w;
    } else if (i < p.n) {
        device const vec<T, 4>* kr = (device const vec<T, 4>*)(k + (ulong)i * p.d + h * 64);
        float4 s4 = 0.f;
        for (uint c = 0; c < 16; c++) s4 = fma(float4(kr[c]), ((threadgroup const float4*)qs)[c], s4);
        s = s4.x + s4.y + s4.z + s4.w;
    }
    float m = simd_max(s);
    if (lane == 0) red[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = max(max(red[0], red[1]), max(red[2], red[3]));
    float e = i < p.n ? exp(s - m) : 0.f;
    S[tid] = e;
    float l = simd_sum(e);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) red[4 + sg] = l;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    l = red[4] + red[5] + red[6] + red[7];
    uint gr = tid / 16, c = tid % 16;
    uint end = min(128u, p.n - c0);
    float4 a = 0.f;
    for (uint j = gr; j < end; j += 8) {
        float4 vv = c0 + j == own ? ((threadgroup const float4*)vn)[c]
                                  : float4(((device const vec<T, 4>*)(v + (ulong)(c0 + j) * p.d + h * 64))[c]);
        a = fma(S[j], vv, a);
    }
    acc[tid] = a;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (p.chunks == 1) {
        // The merge of one chunk: (1 * a) / (1 * l).
        if (tid < 16) {
            float4 t = acc[tid];
            for (uint j = 1; j < 8; j++) t += acc[j * 16 + tid];
            ((device float4*)(o + h * 64))[tid] = t / l;
        }
        return;
    }
    device float* out = part + ((ulong)h * p.chunks + g.x) * 68;
    if (tid < 16) {
        float4 t = acc[tid];
        for (uint j = 1; j < 8; j++) t += acc[j * 16 + tid];
        ((device float4*)(out + 4))[tid] = t;
    }
    if (tid == 0) { out[0] = m; out[1] = l; }
}
kernel void whd_part_f32(device const float* q [[buffer(0)]], device const float* qb [[buffer(1)]], device float* k [[buffer(2)]],
                         device float* v [[buffer(3)]], device float* part [[buffer(4)]], constant WDP& p [[buffer(5)]],
                         device const float* nk [[buffer(6)]], device const float* nv [[buffer(7)]], device const float* bk [[buffer(8)]],
                         device const float* bv [[buffer(9)]], device float* o [[buffer(10)]],
                         uint2 g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float qs[64], S[128], red[8], kn[64], vn[64];
    threadgroup float4 acc[128];
    whd_part_body<float>(q, qb, k, v, part, p, g, tid, sg, lane, qs, S, red, acc, nk, nv, bk, bv, o, kn, vn);
}
kernel void whd_part_h(device const float* q [[buffer(0)]], device const float* qb [[buffer(1)]], device half* k [[buffer(2)]],
                       device half* v [[buffer(3)]], device float* part [[buffer(4)]], constant WDP& p [[buffer(5)]],
                       device float* o [[buffer(10)]],
                       uint2 g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float qs[64], S[128], red[8], kn[64], vn[64];
    threadgroup float4 acc[128];
    whd_part_body<half>(q, qb, k, v, part, p, g, tid, sg, lane, qs, S, red, acc, q, q, q, q, o, kn, vn);
}
// x += bias over rows of d (the residual step), then each row's LayerNorm (weight w, bias lb)
// into f32 rows of `out`: one SIMD-group per row.
struct WDL { uint d; float eps; uint _p0; uint _p1; };
kernel void whd_ln(device float* x [[buffer(0)]], device const float* bias [[buffer(1)]], device const float* w [[buffer(2)]],
                   device const float* lb [[buffer(3)]], device float* out [[buffer(4)]], constant WDL& p [[buffer(5)]],
                   uint r [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    uint d4 = p.d / 4;
    device float4* xr = (device float4*)(x + (ulong)r * p.d);
    device const float4* b4 = (device const float4*)bias;
    float sum = 0.f;
    for (uint c = lane; c < d4; c += 32) {
        float4 v = xr[c] + b4[c];
        xr[c] = v;
        sum += v.x + v.y + v.z + v.w;
    }
    float mean = simd_sum(sum) / float(p.d);
    float sq = 0.f;
    for (uint c = lane; c < d4; c += 32) { float4 v = xr[c] - mean; sq += dot(v, v); }
    float inv = 1.0f / sqrt(simd_sum(sq) / float(p.d) + p.eps);
    device const float4* w4 = (device const float4*)w;
    device const float4* l4 = (device const float4*)lb;
    device float4* o = (device float4*)(out + (ulong)r * p.d);
    for (uint c = lane; c < d4; c += 32) o[c] = (xr[c] - mean) * inv * w4[c] + l4[c];
}
// The chunks of `whd_part_*` of one head (threadgroup, 64 threads: one per value) into o.
kernel void whd_attn_merge(device const float* part [[buffer(0)]], device float* o [[buffer(1)]], constant WDP& p [[buffer(2)]],
                           uint h [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    device const float* ph = part + (ulong)h * p.chunks * 68;
    float m = -INFINITY;
    for (uint c = 0; c < p.chunks; c++) m = max(m, ph[c * 68]);
    float l = 0.f, a = 0.f;
    for (uint c = 0; c < p.chunks; c++) {
        float w = exp(ph[c * 68] - m);
        l += w * ph[c * 68 + 1];
        a += w * ph[c * 68 + 4 + tid];
    }
    o[h * 64 + tid] = a / l;
}

// ---- Greedy decoding chained on the device ----
// Per row of a chain: the token to decode next, the one before it, the last timestamp generated
// (0xffffffff: none), the tokens generated so far, and whether the row applies the timestamp
// rules.
struct WCS { uint last; uint penult; uint last_ts; uint n; uint timestamps; uint _p0; uint _p1; uint _p2; };
// Rows' input vectors: the bf16 embedding of the row's next token (`emb`, rows of d) times `es`
// plus the position's (`pos` rows of d at the row's position `p[r]`).
struct WCE { uint d; float es; uint _p0; uint _p1; };
kernel void whc_embed(device const ushort* emb [[buffer(0)]], device const float* pos [[buffer(1)]],
                      device const WCS* st [[buffer(2)]], device float* x [[buffer(3)]], constant WCE& p [[buffer(4)]],
                      constant uint* rp [[buffer(5)]],
                      uint2 g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    uint c = g.x * 256 + tid, r = g.y;
    if (c >= p.d) return;
    float e = float(as_type<bfloat>(emb[(ulong)st[r].last * p.d + c]));
    x[(ulong)r * p.d + c] = e * p.es + pos[(ulong)rp[r] * p.d + c];
}

// The next token of each row (one threadgroup of 1024 threads per row) from its logits: the
// suppression masks (`mask` bit 0: always, bit 1: as the first generated token) and, for rows with
// timestamps, the timestamp rules of `crate::whisper::process`; then the most likely token (the
// lowest index among equals) and its log-probability. The token goes to `out` and the row's state
// moves on.
struct WCP { uint vocab; uint tb; uint eot; uint no_ts; uint max_init; uint step; uint steps; uint _p; };
inline float whc_rule(device const float* l, device const uchar* mask, WCS s, constant WCP& p, uint i, bool last_ts, bool penult_ts, uint ts_last) {
    float v = l[i];
    uchar mk = mask[i];
    if ((mk & 1) != 0 || ((mk & 2) != 0 && s.n == 0)) return -INFINITY;
    if (s.timestamps == 0) return v;
    if (i == p.no_ts) return -INFINITY;
    if (last_ts) {
        if (penult_ts) { if (i >= p.tb) return -INFINITY; }
        else if (i < p.eot) return -INFINITY;
    }
    if (i >= p.tb && i < ts_last) return -INFINITY;
    if (s.n == 0) {
        if (i < p.tb) return -INFINITY;
        if (p.max_init != 0xffffffffu && i > p.tb + p.max_init) return -INFINITY;
    }
    return v;
}
kernel void whc_select(device const float* logits [[buffer(0)]], device const uchar* mask [[buffer(1)]],
                       device WCS* st [[buffer(2)]], device float2* out [[buffer(3)]], constant WCP& p [[buffer(4)]],
                       uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float rf[32];
    threadgroup float rg[32];
    threadgroup uint ri[32];
    device const float* l = logits + (ulong)r * p.vocab;
    WCS s = st[r];
    bool last_ts = s.n >= 1 && s.last >= p.tb;
    bool penult_ts = s.n < 2 || s.penult >= p.tb;
    uint ts_last = p.tb;
    if (s.last_ts != 0xffffffffu) ts_last = (last_ts && !penult_ts) ? s.last_ts : s.last_ts + 1;
    // Pass 1 (timestamps): the timestamps' log-sum-exp against the best text token.
    bool kill_text = false;
    if (s.timestamps != 0) {
        float tm = -INFINITY, xm = -INFINITY;
        for (uint i = tid; i < p.vocab; i += 1024) {
            float v = whc_rule(l, mask, s, p, i, last_ts, penult_ts, ts_last);
            if (i >= p.tb) tm = max(tm, v); else xm = max(xm, v);
        }
        tm = simd_max(tm); xm = simd_max(xm);
        if (lane == 0) { rf[sg] = tm; rg[sg] = xm; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tm = rf[0]; xm = rg[0];
        for (uint i = 1; i < 32; i++) { tm = max(tm, rf[i]); xm = max(xm, rg[i]); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float ts = 0.f;
        if (tm != -INFINITY)
            for (uint i = p.tb + tid; i < p.vocab; i += 1024) ts += exp(whc_rule(l, mask, s, p, i, last_ts, penult_ts, ts_last) - tm);
        ts = simd_sum(ts);
        if (lane == 0) rf[sg] = ts;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ts = 0.f;
        for (uint i = 0; i < 32; i++) ts += rf[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float ts_lse = tm == -INFINITY ? -INFINITY : tm + log(ts);
        kill_text = ts_lse > xm;
    }
    // Pass 2: argmax (lowest index among equals) and the maximum, then the log-sum-exp.
    float bm = -INFINITY;
    uint bi = 0xffffffffu;
    for (uint i = tid; i < p.vocab; i += 1024) {
        float v = (kill_text && i < p.tb) ? -INFINITY : whc_rule(l, mask, s, p, i, last_ts, penult_ts, ts_last);
        if (v > bm || (v == bm && i < bi)) { bm = v; bi = i; }
    }
    for (uint o = 16; o > 0; o /= 2) {
        float om = simd_shuffle_xor(bm, o);
        uint oi = simd_shuffle_xor(bi, o);
        if (om > bm || (om == bm && oi < bi)) { bm = om; bi = oi; }
    }
    if (lane == 0) { rf[sg] = bm; ri[sg] = bi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bm = rf[0]; bi = ri[0];
    for (uint i = 1; i < 32; i++) if (rf[i] > bm || (rf[i] == bm && ri[i] < bi)) { bm = rf[i]; bi = ri[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float se = 0.f;
    for (uint i = tid; i < p.vocab; i += 1024) {
        float v = (kill_text && i < p.tb) ? -INFINITY : whc_rule(l, mask, s, p, i, last_ts, penult_ts, ts_last);
        se += exp(v - bm);
    }
    se = simd_sum(se);
    if (lane == 0) rg[sg] = se;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float t = 0.f;
        for (uint i = 0; i < 32; i++) t += rg[i];
        float lp = -log(t);
        out[(ulong)r * p.steps + p.step] = float2(as_type<float>(bi), lp);
        WCS n = s;
        n.penult = s.last;
        n.last = bi;
        if (bi >= p.tb) n.last_ts = bi;
        n.n = s.n + 1;
        st[r] = n;
    }
}
#endif
