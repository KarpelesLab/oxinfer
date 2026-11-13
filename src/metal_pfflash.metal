// ---- Prefill attention on the SIMD-group matrices (flash attention; metal_decode.rs `MOp::PfFlash`) ----
// Causal attention of a chunk of `t` queries (positions start .. start + t - 1) over the device KV
// cache (f32 rows of nkv * HD floats per position), optionally windowed (keys > position - window).
// Grid (blocks of 32 queries, query heads), 4 SIMD-groups of 8 queries each: S = Q K^T and O += P V
// on 8x8 matrices, keys 8 at a time with an online softmax held per row (each lane owns two
// elements of a row of the 8x8 tiles: row (lane / 4 & 4) + (lane / 2 % 4), columns
// (lane / 4 & 2) * 2 + (lane % 2) * 2 and the next one, so a row's lanes differ in bits 0 and 3).
// The SIMD-groups of a threadgroup read the same keys and values (cache hits). Rounding differs
// from the one-token kernels (prefill plans already do: half GEMM inputs).
struct PFF { uint start; uint t; uint nh; uint nkv; float scale; uint window; uint sq; uint so; uint sgate; uint g_off; uint g_stride; uint has_gate; };

template <int HD, bool QREG>
inline void pff_body(device const float* q, device const float* kc, device const float* vc, device float* out,
                     device const float* gate, constant PFF& a, uint2 tg, ushort sg, ushort lane) {
    constexpr int NK = HD / 8;
    uint h = tg.y, kvh = h / (a.nh / a.nkv);
    uint i0 = tg.x * 32 + sg * 8;
    if (i0 >= a.t) return;
    ulong rs = (ulong)a.nkv * HD;
    ushort qid = lane / 4;
    ushort fr = (qid & 4) + ((lane / 2) % 4);
    ushort fc = (qid & 2) * 2 + (lane % 2) * 2;
    uint pi = a.start + i0 + fr;
    uint lo = a.window != 0 && pi + 1 > a.window ? pi + 1 - a.window : 0;
    uint pfirst = a.start + i0;
    uint kbeg = a.window != 0 && pfirst + 1 > a.window ? (pfirst + 1 - a.window) & ~7u : 0;
    uint kend = a.start + min(i0 + 8, a.t);
    device const float* qb = q + (ulong)i0 * a.sq + (ulong)h * HD;
    simdgroup_float8x8 Qm[QREG ? NK : 1];
    if (QREG) {
#pragma unroll
        for (int k = 0; k < NK; k++) simdgroup_load(Qm[k], qb + k * 8, a.sq);
    }
    simdgroup_float8x8 O[NK];
#pragma unroll
    for (int k = 0; k < NK; k++) O[k] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    float m = -INFINITY, l = 0.f;
    const float sc = a.scale * M_LOG2E_F;
    device const float* kb = kc + (ulong)kvh * HD;
    device const float* vb = vc + (ulong)kvh * HD;
    for (uint j = kbeg; j < kend; j += 8) {
        // A last partial block is read as the 8 keys ending at kend (the cache may end there);
        // the keys before j were done already and are masked.
        uint jl = (j + 8 > kend && kend >= 8) ? kend - 8 : j;
        simdgroup_float8x8 S = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
#pragma unroll
        for (int k = 0; k < NK; k++) {
            simdgroup_float8x8 Kt;
            simdgroup_load(Kt, kb + (ulong)jl * rs + k * 8, rs, ulong2(0, 0), true);
            if (QREG) {
                simdgroup_multiply_accumulate(S, Qm[k], Kt, S);
            } else {
                simdgroup_float8x8 Qk;
                simdgroup_load(Qk, qb + k * 8, a.sq);
                simdgroup_multiply_accumulate(S, Qk, Kt, S);
            }
        }
        thread float2& se = reinterpret_cast<thread float2&>(S.thread_elements());
        float2 s = se * sc;
        uint k0 = jl + fc, k1 = k0 + 1;
        s.x = (k0 >= j && k0 <= pi && k0 >= lo) ? s.x : -INFINITY;
        s.y = (k1 >= j && k1 <= pi && k1 >= lo) ? s.y : -INFINITY;
        float rmax = max(s.x, s.y);
        rmax = max(rmax, simd_shuffle_xor(rmax, 1));
        rmax = max(rmax, simd_shuffle_xor(rmax, 8));
        float mn = max(m, rmax);
        float alpha = mn == -INFINITY ? 1.f : exp2(m - mn);
        float2 p = mn == -INFINITY ? float2(0.f) : exp2(s - mn);
        float rsum = p.x + p.y;
        rsum += simd_shuffle_xor(rsum, 1);
        rsum += simd_shuffle_xor(rsum, 8);
        l = l * alpha + rsum;
        m = mn;
        se = p;
#pragma unroll
        for (int k = 0; k < NK; k++) {
            thread float2& oe = reinterpret_cast<thread float2&>(O[k].thread_elements());
            oe *= alpha;
            simdgroup_float8x8 V;
            simdgroup_load(V, vb + (ulong)jl * rs + k * 8, rs);
            simdgroup_multiply_accumulate(O[k], S, V, O[k]);
        }
    }
    uint i = i0 + fr;
    if (i >= a.t) return;
    float inv = 1.f / l;
    device float* oi = out + (ulong)i * a.so + (ulong)h * HD;
    device const float* gi = gate + (ulong)i * a.sgate + a.g_off + (ulong)h * a.g_stride;
#pragma unroll
    for (int k = 0; k < NK; k++) {
        float2 o = reinterpret_cast<thread float2&>(O[k].thread_elements()) * inv;
        uint d = k * 8 + fc;
        if (a.has_gate != 0) {
            o.x *= 1.0f / (1.0f + exp(-gi[d]));
            o.y *= 1.0f / (1.0f + exp(-gi[d + 1]));
        }
        oi[d] = o.x;
        oi[d + 1] = o.y;
    }
}
#define PFF_INST(HD, QREG) \
kernel void pff_##HD(device const float* q [[buffer(0)]], device const float* kc [[buffer(1)]], device const float* vc [[buffer(2)]], \
                     device float* out [[buffer(3)]], device const float* gate [[buffer(4)]], constant PFF& a [[buffer(5)]], \
                     uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], \
                     ushort lane [[thread_index_in_simdgroup]]) { \
    pff_body<HD, QREG>(q, kc, vc, out, gate, a, tg, sg, lane); }
PFF_INST(64, true) PFF_INST(128, true)

// Large heads (256, 512): NS SIMD-groups share 8 queries, each holding HD / NS dimensions of the
// queries and outputs (registers of the HD <= 128 kernel). Per block of 8 keys each writes its
// partial scores to threadgroup memory (double-buffered: one barrier a block), every one sums the
// NS partials in the same order and runs the same online softmax, then accumulates its
// dimensions. QG groups of 8 queries per threadgroup (QG * NS SIMD-groups) share the key reads.
template <int HD, int NS, int QG>
inline void pffs_body(device const float* q, device const float* kc, device const float* vc, device float* out,
                      device const float* gate, constant PFF& a, uint2 tg, ushort sgi, ushort lane, threadgroup float* sb) {
    constexpr int DS = HD / NS;
    constexpr int NK = DS / 8;
    ushort qg = sgi / NS, ds = sgi % NS;
    uint h = tg.y, kvh = h / (a.nh / a.nkv);
    uint tq0 = tg.x * (8 * QG);
    uint i0 = tq0 + qg * 8;
    ulong rs = (ulong)a.nkv * HD;
    ushort qid = lane / 4;
    ushort fr = (qid & 4) + ((lane / 2) % 4);
    ushort fc = (qid & 2) * 2 + (lane % 2) * 2;
    uint pi = a.start + i0 + fr;
    uint lo = a.window != 0 && pi + 1 > a.window ? pi + 1 - a.window : 0;
    uint pfirst = a.start + tq0;
    uint kbeg = a.window != 0 && pfirst + 1 > a.window ? (pfirst + 1 - a.window) & ~7u : 0;
    uint kend = a.start + min(tq0 + 8 * QG, a.t);
    device const float* qb = q + (ulong)i0 * a.sq + (ulong)h * HD + ds * DS;
    simdgroup_float8x8 Qm[NK];
#pragma unroll
    for (int k = 0; k < NK; k++) simdgroup_load(Qm[k], qb + k * 8, a.sq);
    simdgroup_float8x8 O[NK];
#pragma unroll
    for (int k = 0; k < NK; k++) O[k] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    float m = -INFINITY, l = 0.f;
    const float sc = a.scale * M_LOG2E_F;
    device const float* kb = kc + (ulong)kvh * HD + ds * DS;
    device const float* vb = vc + (ulong)kvh * HD + ds * DS;
    uint it = 0;
    for (uint j = kbeg; j < kend; j += 8, it++) {
        uint jl = (j + 8 > kend && kend >= 8) ? kend - 8 : j;
        simdgroup_float8x8 S = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
#pragma unroll
        for (int k = 0; k < NK; k++) {
            simdgroup_float8x8 Kt;
            simdgroup_load(Kt, kb + (ulong)jl * rs + k * 8, rs, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(S, Qm[k], Kt, S);
        }
        thread float2& se = reinterpret_cast<thread float2&>(S.thread_elements());
        threadgroup float* slot = sb + (it & 1) * (QG * NS * 64) + qg * NS * 64;
        *(threadgroup float2*)(slot + ds * 64 + fr * 8 + fc) = se;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float2 s = 0.f;
        for (int d = 0; d < NS; d++) s += *(threadgroup const float2*)(slot + d * 64 + fr * 8 + fc);
        s *= sc;
        uint k0 = jl + fc, k1 = k0 + 1;
        s.x = (k0 >= j && k0 <= pi && k0 >= lo) ? s.x : -INFINITY;
        s.y = (k1 >= j && k1 <= pi && k1 >= lo) ? s.y : -INFINITY;
        float rmax = max(s.x, s.y);
        rmax = max(rmax, simd_shuffle_xor(rmax, 1));
        rmax = max(rmax, simd_shuffle_xor(rmax, 8));
        float mn = max(m, rmax);
        float alpha = mn == -INFINITY ? 1.f : exp2(m - mn);
        float2 p = mn == -INFINITY ? float2(0.f) : exp2(s - mn);
        float rsum = p.x + p.y;
        rsum += simd_shuffle_xor(rsum, 1);
        rsum += simd_shuffle_xor(rsum, 8);
        l = l * alpha + rsum;
        m = mn;
        se = p;
#pragma unroll
        for (int k = 0; k < NK; k++) {
            thread float2& oe = reinterpret_cast<thread float2&>(O[k].thread_elements());
            oe *= alpha;
            simdgroup_float8x8 V;
            simdgroup_load(V, vb + (ulong)jl * rs + k * 8, rs);
            simdgroup_multiply_accumulate(O[k], S, V, O[k]);
        }
    }
    uint i = i0 + fr;
    if (i >= a.t) return;
    float inv = 1.f / l;
    device float* oi = out + (ulong)i * a.so + (ulong)h * HD + ds * DS;
    device const float* gi = gate + (ulong)i * a.sgate + a.g_off + (ulong)h * a.g_stride + ds * DS;
#pragma unroll
    for (int k = 0; k < NK; k++) {
        float2 o = reinterpret_cast<thread float2&>(O[k].thread_elements()) * inv;
        uint d = k * 8 + fc;
        if (a.has_gate != 0) {
            o.x *= 1.0f / (1.0f + exp(-gi[d]));
            o.y *= 1.0f / (1.0f + exp(-gi[d + 1]));
        }
        oi[d] = o.x;
        oi[d + 1] = o.y;
    }
}
#define PFFS_INST(HD, NS, QG) \
kernel void pffs_##HD(device const float* q [[buffer(0)]], device const float* kc [[buffer(1)]], device const float* vc [[buffer(2)]], \
                      device float* out [[buffer(3)]], device const float* gate [[buffer(4)]], constant PFF& a [[buffer(5)]], \
                      uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], \
                      ushort lane [[thread_index_in_simdgroup]]) { \
    threadgroup float sb[2 * QG * NS * 64]; \
    pffs_body<HD, NS, QG>(q, kc, vc, out, gate, a, tg, sg, lane, sb); }
PFFS_INST(256, 2, 4) PFFS_INST(512, 4, 2)
