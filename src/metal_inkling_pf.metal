// Inkling-2 prefill kernels (metal_inkling.rs), after the matrix-unit GEMMs' source (MMA_SHADER_SRC: tg_max,
// row_scale, the grouped GEMMs) and metal_inkling.metal (the weight loaders of NVFP4, MXFP8 and Q2_0).
#pragma clang fp reassociate(on)

// ---- Prefill passes (more tokens than the rings hold) ---------------------------------------------
// The raw inputs of the last `ring_cap` tokens into a convolution ring (grid: channel blocks,
// token), after the convolutions of the pass have read the ring.
struct IRsP { uint ch; uint x; uint pos; uint ntok; uint ring_cap; uint kn; uint _p0; uint _p1; };
kernel void ink_ring_store(device float* a [[buffer(0)]], constant IRsP& p [[buffer(1)]], device float* ring [[buffer(2)]],
                           uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x, ti = gid.y;
    if (i >= p.ch || ti + p.ring_cap < p.ntok) return;
    ring[(ulong)((p.pos + ti + p.kn - 1) % p.ring_cap) * p.ch + i] = a[p.x + (ulong)ti * p.ch + i];
}
// Rows of a pass's keys and values (written by position from `base` into `src`) to a sliding
// window's ring of `cap` rows: the last `cap` positions (grid: row bytes / 4, token).
struct IKrP { uint rb; uint sb; uint base; uint ntok; uint cap; uint _p0; uint _p1; uint _p2; };
kernel void ink_kv_ring(constant IKrP& p [[buffer(0)]], device const uchar* sk [[buffer(1)]], device const uchar* sv [[buffer(2)]],
                        device const uchar* sks [[buffer(3)]], device const uchar* svs [[buffer(4)]], device uchar* dk [[buffer(5)]],
                        device uchar* dv [[buffer(6)]], device uchar* dks [[buffer(7)]], device uchar* dvs [[buffer(8)]],
                        uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x, ti = gid.y;
    if (ti + p.cap < p.ntok) return;
    ulong r = (p.base + ti) % p.cap;
    if (i < p.rb / 4) {
        ((device uint*)(dk + r * p.rb))[i] = ((device const uint*)(sk + (ulong)ti * p.rb))[i];
        ((device uint*)(dv + r * p.rb))[i] = ((device const uint*)(sv + (ulong)ti * p.rb))[i];
    }
    if (i < p.sb / 4) {
        ((device uint*)(dks + r * p.sb))[i] = ((device const uint*)(sks + (ulong)ti * p.sb))[i];
        ((device uint*)(dvs + r * p.sb))[i] = ((device const uint*)(svs + (ulong)ti * p.sb))[i];
    }
}
// SiTU of grouped gate / up rows (`[rows][2 inter]`, interleaved with `il` 2) times each row's
// factor, as scaled half rows for the down GEMM (as `silu_half`).
struct ISiP { uint n; uint il; float cap_g; float cap_u; };
inline float situ(float g, float u, float cg, float cu) {
    float uu = cu > 0.f ? cu * precise::tanh(u / cu) : u;
    return cg * precise::tanh(g / cg) / (1.f + exp(-g)) * uu;
}
kernel void ink_situ_half(device const float* gu [[buffer(0)]], device half* ah [[buffer(1)]], device float* rs [[buffer(2)]],
                          constant ISiP& p [[buffer(3)]], device const float* fac [[buffer(4)]],
                          uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8];
    device const float* row = gu + (ulong)r * 2 * p.n;
    float f = fac[r];
    float m = 0.f;
    for (uint i = tid; i < p.n; i += 256) {
        float g = (p.il == 2 ? row[2 * i] : row[i]) * f, u = (p.il == 2 ? row[2 * i + 1] : row[p.n + i]) * f;
        m = max(m, fabs(situ(g, u, p.cap_g, p.cap_u)));
    }
    m = tg_max(m, sg, lane, red);
    float inv, s = row_scale(m, inv);
    device half* out = ah + (ulong)r * p.n;
    for (uint i = tid; i < p.n; i += 256) {
        float g = (p.il == 2 ? row[2 * i] : row[i]) * f, u = (p.il == 2 ? row[2 * i + 1] : row[p.n + i]) * f;
        out[i] = half(situ(g, u, p.cap_g, p.cap_u) * s);
    }
    if (tid == 0) rs[r] = inv;
}
// The experts' rows of each token summed with its routing weights times the down factors:
// `y[t] = sum_s w[t][s] fd[row] o[row]`, row = map[t k + s] (grid: blocks of h, token).
struct ICbP { uint h; uint k; uint sel; uint y; };
kernel void ink_combine(device float* a [[buffer(0)]], constant ICbP& p [[buffer(1)]], device const float* o [[buffer(2)]],
                        device const uint* map [[buffer(3)]], device const float* fd [[buffer(4)]],
                        uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x, ti = gid.y;
    if (i >= p.h) return;
    device const float* sel = a + p.sel + (ulong)ti * 2 * p.k;
    float s = 0.f;
    for (uint j = 0; j < p.k; j++) {
        uint r = map[ti * p.k + j];
        s += sel[p.k + j] * fd[r] * o[(ulong)r * p.h + i];
    }
    a[p.y + (ulong)ti * p.h + i] = s;
}

// Grouped expert GEMMs of the formats the matrix-unit library leaves out.
#define MK_MXF8 MXF8W{ (device const uchar*)qb, (device const float*)sb, p.n_in }
MMAG(mmag_nvfp4, NVFP4W, MK_GG(NVFP4W)) MMAG(mmag_Q2_0, Q2_0W, MK_GG(Q2_0W)) MMAG(mmag_mxf8, MXF8W, MK_MXF8)
#define MK_MXF8B MXF8B{ (device const uchar*)qb, (device const uchar*)sb, p.n_in }
MMAG(mmag_mxf8b, MXF8B, MK_MXF8B)

// ---- Prefill attention of a full-attention layer on the matrix units (flash attention) ---------
// `ink_attn` with its splits and `ink_attn_merge` in one kernel for a pass's queries over a cache
// of rows by position, as `pfm_body` (metal_pfmma.metal): a threadgroup of NSG SIMD-groups takes
// Q tokens of one query head and walks the keys C at a time, S = Q K^T (f32 sums) to threadgroup
// memory for an online softmax (the maximum moving lazily, by more than 8 in log2 units), the
// probabilities back as ET for O += P V in a cooperative tensor. The keys within `extent` of a
// query get their relative bias (`ink_attn`'s), the sink joins the sum at the end and the output
// is gated. `IAP` as `ink_attn`'s, `_m1` the pass's token count. The keys and values come from
// SRC: f32 or f16 rows read in place (`InkFaDev`), or int8 rows with their scales dequantized
// to half tiles in threadgroup memory (`InkFaQ8`).
// (Rounding: ET operands, f32 sums; f16 caches are half already.)
template <int HD, int C, typename ET>
struct InkFaDev {
    device const ET* kq;
    device const ET* vq;
    int rs;
    inline void load_k(uint, int, ushort, int) const {}
    inline void load_v(uint, int, ushort, int) const {}
    inline auto kt(uint kb, int nk) const {
        return tensor((device ET*)(kq + (ulong)kb * rs), dextents<int32_t, 2>(HD, nk), array<int, 2>({1, rs}));
    }
    inline auto vt(uint kb, int nk) const {
        return tensor((device ET*)(vq + (ulong)kb * rs), dextents<int32_t, 2>(HD, nk), array<int, 2>({1, rs}));
    }
};
template <int HD, int C>
struct InkFaQ8 {
    device const char* kq;
    device const char* vq;
    device const float* ks;
    device const float* vs;
    uint off;
    uint kw;
    threadgroup half* skv;
    // Keys kb .. kb + nk - 1 as half rows [C][HD] (zeros past nk).
    inline void load(device const char* q, device const float* s, uint kb, int nk, ushort tid, int nt) const {
        for (int i = tid; i < C * HD / 4; i += nt) {
            int key = i / (HD / 4), d = i % (HD / 4) * 4;
            half4 v = 0.h;
            if (key < nk) {
                ulong idx = (ulong)(kb + key) * kw + off + d;
                v = half4(float4(*(device const char4*)(q + idx)) * s[idx / 32]);
            }
            *(threadgroup half4*)(skv + key * HD + d) = v;
        }
    }
    inline void load_k(uint kb, int nk, ushort tid, int nt) const { load(kq, ks, kb, nk, tid, nt); }
    inline void load_v(uint kb, int nk, ushort tid, int nt) const { load(vq, vs, kb, nk, tid, nt); }
    inline auto kt(uint, int) const { return tensor<threadgroup half, extents<int32_t, HD, C>, tensor_inline>(skv, extents<int32_t, HD, C>()); }
    inline auto vt(uint, int) const { return kt(0, 0); }
};
template <int HD, int Q, int C, int NSG, typename ET, bool STAGED, typename SRC>
inline void ink_fa_body(device float* a, device const float* c, constant IAP& p, thread const SRC& src, uint2 tg, ushort sg, ushort lane,
                        threadgroup ET* sq, threadgroup float* ss, threadgroup ET* sp, threadgroup float* sr, threadgroup float* srr,
                        threadgroup int* sf) {
    typedef vec<ET, 4> ET4;
    constexpr int NT = NSG * 32;
    constexpr int NQ = Q / NSG;
    constexpr int NC = C / 32;
    constexpr int H4 = HD / 4;
    const ushort tid = sg * 32 + lane;
    const uint h = tg.y, nt = p._m1;
    const uint i0 = tg.x * Q;
    const float sc = p.scale * M_LOG2E_F;
    for (uint i = tid; i < Q * H4; i += NT) {
        uint j = i / H4;
        float4 v = 0.f;
        // (Scalar loads: the arena's offsets need not be 16-byte aligned.)
        if (i0 + j < nt) {
            device const float* qr = a + p.q + (ulong)(i0 + j) * p.nh * HD + (ulong)h * HD + (i % H4) * 4;
            v = float4(qr[0], qr[1], qr[2], qr[3]);
        }
        ((threadgroup ET4*)sq)[i] = ET4(v * sc);
    }
    // The relative vectors (times log2 e), 16 per query.
    for (uint i = tid; i < Q * 16; i += NT) {
        uint j = i / 16, d = i % 16;
        srr[i] = i0 + j < nt && d < p.dr ? a[p.r + ((ulong)(i0 + j) * p.nh + h) * p.dr + d] * M_LOG2E_F : 0.f;
    }
    const uint pfirst = p.pos + i0;
    const uint kend = p.pos + min(i0 + Q, nt);
    float M[NQ], L[NQ];
    uint pi[NQ];
#pragma unroll
    for (int jj = 0; jj < NQ; jj++) {
        M[jj] = -FLT_MAX / 2;
        L[jj] = 0.f;
        pi[jj] = pfirst + jj * NSG + sg;
    }
    auto tq = tensor<threadgroup ET, extents<int32_t, HD, Q>, tensor_inline>(sq, extents<int32_t, HD, Q>());
    auto ts = tensor<threadgroup float, extents<int32_t, C, Q>, tensor_inline>(ss, extents<int32_t, C, Q>());
    auto tp = tensor<threadgroup ET, extents<int32_t, C, Q>, tensor_inline>(sp, extents<int32_t, C, Q>());
    matmul2d<matmul2d_descriptor(Q, C, HD, false, true, false, matmul2d_descriptor::mode::multiply), execution_simdgroups<NSG>> mm_qk;
    matmul2d<matmul2d_descriptor(Q, HD, C, false, false, false, matmul2d_descriptor::mode::multiply_accumulate),
             execution_simdgroups<NSG>> mm_pv;
    auto tv0 = src.vt(0, C);
    auto co = mm_pv.template get_destination_cooperative_tensor<decltype(tp), decltype(tv0), float>();
#pragma unroll
    for (uint16_t i = 0; i < co.get_capacity(); ++i)
        if (co.is_valid_element(i)) co[i] = 0.f;
    if (tid == 0) sf[0] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // (Keys from distance `extent` before the first query on get a bias.)
    const uint kbias = pfirst + 1 > p.extent ? pfirst + 1 - p.extent : 0;
    uint it = 0;
    for (uint kb = 0; kb < kend; kb += C, it++) {
        int nk = min((int)C, (int)(kend - kb));
        if (STAGED) {
            src.load_k(kb, nk, tid, NT);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        auto tk = src.kt(kb, nk);
        mm_qk.run(tq, tk, ts);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // (A staged tile's values replace its keys while the scores are made.)
        if (STAGED) src.load_v(kb, nk, tid, NT);
        const bool near = kb + C > kbias;
#pragma unroll
        for (int jj = 0; jj < NQ; jj++) {
            const int j = jj * NSG + sg;
            float s[NC];
            float m = M[jj];
#pragma unroll
            for (int ii = 0; ii < NC; ii++) {
                uint cc = ii * 32 + lane;
                uint key = kb + cc;
                float x = ss[j * C + cc];
                if (near) {
                    uint dist = pi[jj] - key;
                    if (key <= pi[jj] && dist < p.extent) {
                        device const float* rel = c + p.rel + dist;
                        threadgroup const float* rr = srr + j * 16;
                        float b = 0.f;
                        for (uint d = 0; d < p.dr; d++) b += rr[d] * rel[d * p.extent];
                        x += b;
                    }
                }
                s[ii] = (int)cc < nk && key <= pi[jj] ? x : -INFINITY;
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
                ET pe = ET(exp2(s[ii] - M[jj]));
                sp[j * C + ii * 32 + lane] = pe;
                sum += float(pe);
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
        auto tv = src.vt(kb, nk);
        mm_pv.run(tp, tv, co);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // The sink (a logit of a zero value) into each row's sum: the output factor per row.
#pragma unroll
    for (int jj = 0; jj < NQ; jj++) {
        const int j = jj * NSG + sg;
        if (lane == 0) {
            float f = 0.f;
            if (i0 + j < nt) {
                float sl = a[p.sink + (ulong)(i0 + j) * p.nh + h] * M_LOG2E_F;
                float mf = max(M[jj], sl);
                float e = exp2(M[jj] - mf);
                f = e / (L[jj] * e + exp2(sl - mf));
            }
            sr[j] = f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint16_t i = 0; i < co.get_capacity(); ++i)
        if (co.is_valid_element(i)) {
            auto idx = co.get_multidimensional_index(i);
            uint r = i0 + idx[1];
            if (r < nt) {
                ulong o = (ulong)r * p.nh * HD + (ulong)h * HD + idx[0];
                a[p.out + o] = co[i] * sr[idx[1]] * (1.0f / (1.0f + exp(-a[p.gate + o])));
            }
        }
}
#define INK_FA_TG(Q, C, ET) \
    threadgroup ET sq[Q * HD]; \
    threadgroup float ss[Q * C]; \
    threadgroup ET sp[Q * C]; \
    threadgroup float sr[Q]; \
    threadgroup float srr[Q * 16]; \
    threadgroup int sf[1];
#define INK_FA(NAME, HD_, Q, C, NSG, ET) \
kernel void NAME(device float* a [[buffer(0)]], device const float* c [[buffer(1)]], constant IAP& p [[buffer(2)]], \
                 device const ET* kc [[buffer(3)]], device const ET* vc [[buffer(4)]], \
                 uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], \
                 ushort lane [[thread_index_in_simdgroup]]) { \
    constexpr int HD = HD_; \
    INK_FA_TG(Q, C, ET) \
    uint kvh = tg.y / (p.nh / p.nkv); \
    InkFaDev<HD, C, ET> src{kc + kvh * HD, vc + kvh * HD, (int)p.kw}; \
    ink_fa_body<HD, Q, C, NSG, ET, false>(a, c, p, src, tg, sg, lane, sq, ss, sp, sr, srr, sf); }
INK_FA(ink_fa_h, 128, 16, 128, 4, half)
INK_FA(ink_fa_f, 128, 16, 128, 4, float)
// (Measured at 64k positions, f16 caches: 32 queries and 64 keys, 64 keys, two SIMD-groups: no
// faster; 32 queries and 128 keys exceed the threadgroup memory.)
// int8 caches: keys and values staged as half tiles of 64 rows (16 KB, one tile at a time).
kernel void ink_fa_q(device float* a [[buffer(0)]], device const float* c [[buffer(1)]], constant IAP& p [[buffer(2)]],
                     device const char* kc [[buffer(3)]], device const char* vc [[buffer(4)]], device const float* ks [[buffer(5)]],
                     device const float* vs [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                     ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int HD = 128;
    INK_FA_TG(16, 64, half)
    threadgroup half skv[64 * HD];
    uint kvh = tg.y / (p.nh / p.nkv);
    InkFaQ8<HD, 64> src{kc, vc, ks, vs, kvh * HD, p.kw, skv};
    ink_fa_body<HD, 16, 64, 4, half, true>(a, c, p, src, tg, sg, lane, sq, ss, sp, sr, srr, sf);
}
