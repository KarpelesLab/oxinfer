// ---- Prefill attention on the matrix units (Metal 4 `matmul2d`; metal_decode.rs `MOp::PfFlash`) ----
// The causal (optionally windowed) attention of `pff_*` (same `PFF` arguments), as llama.cpp's
// `kernel_flash_attn_ext_tensor`, over half copies of the cache's rows (`kh` / `vh`, rows of
// nkv * HD halves by position: the chunk's written by `dec_qk_rope_t` next to the f32 cache, the
// earlier ones converted by `pf_kv_half`; or, with `gpu_kv=f16`, the half caches themselves, at
// any length). A threadgroup of NSG SIMD-groups takes Q queries of one
// head and walks the keys C at a time: S = Q K^T (f32 sums) goes to threadgroup memory for an
// online softmax (Q / NSG rows per SIMD-group, C / 32 columns per lane), the probabilities come
// back as half for O += P V, O held in a cooperative tensor. The queries are converted to QT
// with the scale (times log2 e) folded in. The running maximum moves lazily, only when a row's
// grows by more than 8 (2^8 fits half), as llama.cpp's.
// QT = half: the keys from the half rows. QT = float (`pfmq_*`, for unscaled scores, as Gemma's,
// whose rounding half Q and K would scale up): S = Q K^T in f32 from the f32 cache, only P V on
// half operands.
template <int HD, int Q, int C, int NSG, typename QT>
inline void pfm_body(device const float* q, device const float* kc, device const half* kh, device const half* vh, device float* out,
                     device const float* gate, constant PFF& a, uint2 tg, ushort sg, ushort lane,
                     threadgroup QT* sq, threadgroup float* ss, threadgroup half* sp, threadgroup float* sr, threadgroup int* sf) {
    typedef vec<QT, 4> QT4;
    constexpr int NT = NSG * 32;
    constexpr int NQ = Q / NSG;
    constexpr int NC = C / 32;
    constexpr int H4 = HD / 4;
    const ushort tid = sg * 32 + lane;
    const uint h = tg.y, kvh = h / (a.nh / a.nkv);
    const uint i0 = tg.x * Q;
    const int rs = a.nkv * HD;
    const float sc = a.scale * M_LOG2E_F;
    for (uint i = tid; i < Q * H4; i += NT) {
        uint j = i / H4;
        float4 v = 0.f;
        if (i0 + j < a.t) v = ((device const float4*)(q + (ulong)(i0 + j) * a.sq + (ulong)h * HD))[i % H4];
        ((threadgroup QT4*)sq)[i] = QT4(v * sc);
    }
    const uint pfirst = a.start + i0;
    const uint kend = a.start + min(i0 + Q, a.t);
    const uint kbeg = a.window != 0 && pfirst + 1 > a.window ? pfirst + 1 - a.window : 0;
    float M[NQ], L[NQ];
    uint pi[NQ], lo[NQ];
#pragma unroll
    for (int jj = 0; jj < NQ; jj++) {
        M[jj] = -FLT_MAX / 2;
        L[jj] = 0.f;
        pi[jj] = pfirst + jj * NSG + sg;
        lo[jj] = a.window != 0 && pi[jj] + 1 > a.window ? pi[jj] + 1 - a.window : 0;
    }
    auto tq = tensor<threadgroup QT, extents<int32_t, HD, Q>, tensor_inline>(sq, extents<int32_t, HD, Q>());
    auto ts = tensor<threadgroup float, extents<int32_t, C, Q>, tensor_inline>(ss, extents<int32_t, C, Q>());
    auto tp = tensor<threadgroup half, extents<int32_t, C, Q>, tensor_inline>(sp, extents<int32_t, C, Q>());
    matmul2d<matmul2d_descriptor(Q, C, HD, false, true, false, matmul2d_descriptor::mode::multiply), execution_simdgroups<NSG>> mm_qk;
    matmul2d<matmul2d_descriptor(Q, HD, C, false, false, false, matmul2d_descriptor::mode::multiply_accumulate),
             execution_simdgroups<NSG>> mm_pv;
    device const QT* kq = (device const QT*)(is_same<QT, half>::value ? (device const void*)kh : (device const void*)kc) + kvh * HD;
    device const half* vq = vh + kvh * HD;
    auto tv0 = tensor((device half*)vq, dextents<int32_t, 2>(HD, C), array<int, 2>({1, rs}));
    auto co = mm_pv.template get_destination_cooperative_tensor<decltype(tp), decltype(tv0), float>();
#pragma unroll
    for (uint16_t i = 0; i < co.get_capacity(); ++i)
        if (co.is_valid_element(i)) co[i] = 0.f;
    if (tid == 0) sf[0] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint it = 0;
    for (uint kb = kbeg; kb < kend; kb += C, it++) {
        // A last partial block: rows past kend are clipped by the tensors' bounds (their scores
        // masked, their values zeros).
        int nk = min((int)C, (int)(kend - kb));
        auto tk = tensor((device QT*)(kq + (ulong)kb * rs), dextents<int32_t, 2>(HD, nk), array<int, 2>({1, rs}));
        mm_qk.run(tq, tk, ts);
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
        for (int jj = 0; jj < NQ; jj++) {
            const int j = jj * NSG + sg;
            float s[NC];
            float m = M[jj];
#pragma unroll
            for (int ii = 0; ii < NC; ii++) {
                uint c = ii * 32 + lane;
                uint key = kb + c;
                float x = ss[j * C + c];
                s[ii] = (int)c < nk && key <= pi[jj] && key >= lo[jj] ? x : -INFINITY;
                m = max(m, s[ii]);
            }
            m = simd_max(m);
            float ms = 1.f;
            if (m > M[jj] + 8.f) {
                ms = exp2(M[jj] - m);
                M[jj] = m;
                if (lane == 0) sf[0] = it + 1;
            }
            float sum = 0.f;
#pragma unroll
            for (int ii = 0; ii < NC; ii++) {
                half p = half(exp2(s[ii] - M[jj]));
                sp[j * C + ii * 32 + lane] = p;
                sum += float(p);
            }
            L[jj] = L[jj] * ms + simd_sum(sum);
            if (lane == 0) sr[j] = ms;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sf[0] == int(it + 1)) {
#pragma unroll
            for (uint16_t i = 0; i < co.get_capacity(); ++i)
                if (co.is_valid_element(i)) co[i] *= sr[co.get_multidimensional_index(i)[1]];
        }
        auto tv = tensor((device half*)(vq + (ulong)kb * rs), dextents<int32_t, 2>(HD, nk), array<int, 2>({1, rs}));
        mm_pv.run(tp, tv, co);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
#pragma unroll
    for (int jj = 0; jj < NQ; jj++)
        if (lane == 0) sr[jj * NSG + sg] = L[jj] == 0.f ? 0.f : 1.f / L[jj];
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint16_t i = 0; i < co.get_capacity(); ++i)
        if (co.is_valid_element(i)) {
            auto idx = co.get_multidimensional_index(i);
            uint r = i0 + idx[1];
            if (r < a.t) {
                uint d = idx[0];
                float o = co[i] * sr[idx[1]];
                if (a.has_gate != 0) o *= 1.0f / (1.0f + exp(-gate[(ulong)r * a.sgate + a.g_off + (ulong)h * a.g_stride + d]));
                out[(ulong)r * a.so + (ulong)h * HD + d] = o;
            }
        }
}
#define PFM_INST(NAME, HD, Q, C, NSG, QT) \
kernel void NAME(device const float* q [[buffer(0)]], device const float* kc [[buffer(1)]], device const float* vc [[buffer(2)]], \
                 device float* out [[buffer(3)]], device const float* gate [[buffer(4)]], constant PFF& a [[buffer(5)]], \
                 device const half* kh [[buffer(6)]], device const half* vh [[buffer(7)]], \
                 uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], \
                 ushort lane [[thread_index_in_simdgroup]]) { \
    threadgroup QT sq[Q * HD]; \
    threadgroup float ss[Q * C]; \
    threadgroup half sp[Q * C]; \
    threadgroup float sr[Q]; \
    threadgroup int sf[1]; \
    pfm_body<HD, Q, C, NSG, QT>(q, kc, kh, vh, out, gate, a, tg, sg, lane, sq, ss, sp, sr, sf); }
PFM_INST(pfm_64, 64, 32, 64, 4, half)
PFM_INST(pfm_128, 128, 16, 128, 4, half)
// Heads of 256 and 512 walk 128 keys at a time (`pf_attn_bench`, 2048 queries after 30000
// positions: 8 x 256 heads 15.8 -> 13.7 ms, 8 x 512 48.1 -> 30.5 ms; 512 queries from 0: level or
// faster). 64 keys at a time, 4 or 16 SIMD-groups for 512, 8 for 256, and for heads of 64 16
// queries or 128 keys were slower at long contexts.
PFM_INST(pfm_256, 256, 16, 128, 4, half)
PFM_INST(pfm_512, 512, 16, 128, 8, half)
PFM_INST(pfmq_64, 64, 32, 64, 4, float)
PFM_INST(pfmq_128, 128, 16, 128, 4, float)
PFM_INST(pfmq_256, 256, 16, 64, 4, float)
