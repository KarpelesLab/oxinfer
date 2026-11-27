// Inkling-2 prefill kernels (metal_inkling.rs), after the matrix-unit GEMMs' source (MMA_SHADER_SRC: tg_max,
// row_scale, the grouped GEMMs) and metal_inkling.metal (the weight loaders of NVFP4 and MXFP8).
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
MMAG(mmag_nvfp4, NVFP4W, MK_GG(NVFP4W)) MMAG(mmag_mxf8, MXF8W, MK_MXF8)
#define MK_MXF8B MXF8B{ (device const uchar*)qb, (device const uchar*)sb, p.n_in }
MMAG(mmag_mxf8b, MXF8B, MK_MXF8B)

// ---- Prefill attention on the matrix units (flash attention) ------------------------------------
// `ink_attn` with its splits and `ink_attn_merge` in one kernel for a pass's queries, as `pfm_body`
// (metal_pfmma.metal): a threadgroup of NSG SIMD-groups takes Q tokens of one query head and walks
// their keys a tile of up to C at a time, S = Q K^T (f32 sums) to threadgroup memory for an online
// softmax (the maximum moving lazily, by more than 8 in log2 units), the probabilities back as ET
// over the first half of their scores' rows for O += P V in a cooperative tensor. Keys within
// `extent` of a query get their relative bias (`ink_attn`'s), the sink joins the sum at the end
// and the output is gated. `IAP` as `ink_attn`'s, `_m1` the pass's token count; a sliding window
// (`window`) reads its keys from the window's ring (`cap` rows, position j at row j % cap) and,
// with `tmp`, the pass's own from their rows apart (from position `tbase`), a tile never
// straddling the two or the ring's end. The queries are converted (times the scale and log2 e) to
// ET rows in the arena's `part`, a region per threadgroup, which the products read from there
// (threadgroup memory holds the scores only, so that two threadgroups fit a core). The keys and
// values come from SRC: f32 or f16 rows read in place (`InkFaDev`), or int8 rows with their scales
// (`InkFaQ8`), each thread loading its share of the next tile's raw rows into registers while the
// threadgroup works on the current one, dequantized into a half tile in threadgroup memory (keys,
// then values over them).
// (Rounding: ET operands, f32 sums; f16 caches are half already.)
template <int HD, int C, typename ET>
struct InkFaDev {
    device const ET* kq;
    device const ET* vq;
    device const ET* tkq;
    device const ET* tvq;
    int rs;
    // The tile's first row and whether it is in the pass's rows.
    uint row;
    bool tmp;
    inline void at(uint r, bool t) { row = r; tmp = t; }
    inline void fetch(uint, uint, int, ushort, constant IAP&) {}
    inline void put_k(ushort) const {}
    inline void put_v(ushort) const {}
    inline auto kt(int nk) const {
        return tensor((device ET*)((tmp ? tkq : kq) + (ulong)row * rs), dextents<int32_t, 2>(HD, nk), array<int, 2>({1, rs}));
    }
    inline auto vt(int nk) const {
        return tensor((device ET*)((tmp ? tvq : vq) + (ulong)row * rs), dextents<int32_t, 2>(HD, nk), array<int, 2>({1, rs}));
    }
};
template <int HD, int C, int NT>
struct InkFaQ8 {
    device const char* kq;
    device const char* vq;
    device const float* ks;
    device const float* vs;
    device const char* tkq;
    device const char* tvq;
    device const float* tks;
    device const float* tvs;
    uint off;
    uint kw;
    threadgroup half* skv;
    // (Each thread: 16 values of a key row a load, with their scale.)
    enum { NL = C * HD / 16 / NT };
    uint4 rk[NL], rv[NL];
    float sk[NL], sv[NL];
    inline void at(uint, bool) {}
    // Keys `kb .. kb + nk - 1`, the first at `row` (of the pass's rows with `tmp`).
    inline void fetch(uint row, bool tmp, int nk, ushort tid, constant IAP&) {
#pragma unroll
        for (int l = 0; l < NL; l++) {
            int i = tid + l * NT, key = i / (HD / 16), d = i % (HD / 16) * 16;
            if (key < nk) {
                ulong idx = (ulong)(row + key) * kw + off + d;
                rk[l] = *(device const uint4*)((tmp ? tkq : kq) + idx);
                rv[l] = *(device const uint4*)((tmp ? tvq : vq) + idx);
                sk[l] = (tmp ? tks : ks)[idx / 32];
                sv[l] = (tmp ? tvs : vs)[idx / 32];
            } else {
                rk[l] = 0;
                rv[l] = 0;
                sk[l] = 0.f;
                sv[l] = 0.f;
            }
        }
    }
    inline void put(thread const uint4* r, thread const float* sc, ushort tid) const {
#pragma unroll
        for (int l = 0; l < NL; l++) {
            int i = tid + l * NT, key = i / (HD / 16), d = i % (HD / 16) * 16;
            threadgroup half4* o = (threadgroup half4*)(skv + key * HD + d);
            o[0] = half4(float4(as_type<char4>(r[l].x)) * sc[l]);
            o[1] = half4(float4(as_type<char4>(r[l].y)) * sc[l]);
            o[2] = half4(float4(as_type<char4>(r[l].z)) * sc[l]);
            o[3] = half4(float4(as_type<char4>(r[l].w)) * sc[l]);
        }
    }
    inline void put_k(ushort tid) const { put(rk, sk, tid); }
    inline void put_v(ushort tid) const { put(rv, sv, tid); }
    inline auto kt(int) const { return tensor<threadgroup half, extents<int32_t, HD, C>, tensor_inline>(skv, extents<int32_t, HD, C>()); }
    inline auto vt(int) const { return kt(0); }
};
// The tile of keys from `kb` (below `kend`): its first row, whether in the pass's rows, its length.
template <int C>
inline int ink_fa_tile(constant IAP& p, uint kb, uint kend, thread uint& row, thread bool& tmp) {
    uint lim = kend;
    tmp = p.tmp != 0 && kb >= p.tbase;
    if (tmp) {
        row = kb - p.tbase;
    } else {
        if (p.tmp != 0) lim = min(lim, p.tbase);
        row = kb;
        if (p.cap != 0) {
            row = kb % p.cap;
            lim = min(lim, kb + (p.cap - row));
        }
    }
    return min((int)C, (int)(lim - kb));
}
template <int HD, int Q, int C, int NSG, typename ET, bool STAGED, typename SRC>
inline void ink_fa_body(device float* a, device const float* c, constant IAP& p, thread SRC& src, uint2 tg, uint2 ntg, ushort sg,
                        ushort lane, threadgroup float* ss, threadgroup float* sr, threadgroup int* sf) {
    constexpr int NT = NSG * 32;
    constexpr int NQ = Q / NSG;
    constexpr int NC = C / 32;
    constexpr int H4 = HD / 4;
    const ushort tid = sg * 32 + lane;
    const uint h = tg.y, nt = p._m1;
    const uint i0 = tg.x * Q;
    const float sc = p.scale * M_LOG2E_F;
    device ET* sq = (device ET*)(a + p.part) + ((ulong)tg.y * ntg.x + tg.x) * Q * HD;
    for (uint i = tid; i < Q * H4; i += NT) {
        uint j = i / H4;
        float4 v = 0.f;
        // (Scalar loads: the arena's offsets need not be 16-byte aligned.)
        if (i0 + j < nt) {
            device const float* qr = a + p.q + (ulong)(i0 + j) * p.nh * HD + (ulong)h * HD + (i % H4) * 4;
            v = float4(qr[0], qr[1], qr[2], qr[3]) * sc;
        }
        sq[4 * i] = ET(v.x);
        sq[4 * i + 1] = ET(v.y);
        sq[4 * i + 2] = ET(v.z);
        sq[4 * i + 3] = ET(v.w);
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
    auto tq = tensor<device ET, extents<int32_t, HD, Q>, tensor_inline>(sq, extents<int32_t, HD, Q>());
    auto ts = tensor<threadgroup float, extents<int32_t, C, Q>, tensor_inline>(ss, extents<int32_t, C, Q>());
    threadgroup ET* sp = (threadgroup ET*)ss;
    constexpr int PS = C * 4 / sizeof(ET);
    auto tp = tensor<threadgroup ET, extents<int32_t, C, Q>, tensor_inline>(sp, extents<int32_t, C, Q>(), array<int, 2>({1, PS}));
    matmul2d<matmul2d_descriptor(Q, C, HD, false, true, false, matmul2d_descriptor::mode::multiply), execution_simdgroups<NSG>> mm_qk;
    matmul2d<matmul2d_descriptor(Q, HD, C, false, false, false, matmul2d_descriptor::mode::multiply_accumulate),
             execution_simdgroups<NSG>> mm_pv;
    auto tv0 = src.vt(C);
    auto co = mm_pv.template get_destination_cooperative_tensor<decltype(tp), decltype(tv0), float>();
#pragma unroll
    for (uint16_t i = 0; i < co.get_capacity(); ++i)
        if (co.is_valid_element(i)) co[i] = 0.f;
    if (tid == 0) sf[0] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    // (Keys from distance `extent` before the first query on get a bias; a window's first keys
    // are those of the first query.)
    const uint kbias = pfirst + 1 > p.extent ? pfirst + 1 - p.extent : 0;
    const uint klo = p.window != 0 && pfirst + 1 > p.window ? pfirst + 1 - p.window : 0;
    uint row;
    bool tmp;
    int nk = ink_fa_tile<C>(p, klo, kend, row, tmp);
    if (STAGED) src.fetch(row, tmp, nk, tid, p);
    uint it = 0;
    for (uint kb = klo; kb < kend; it++) {
        src.at(row, tmp);
        if (STAGED) {
            src.put_k(tid);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        auto tk = src.kt(nk);
        mm_qk.run(tq, tk, ts);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint kn = kb + nk;
        uint nrow;
        bool ntmp;
        const int nnk = kn < kend ? ink_fa_tile<C>(p, kn, kend, nrow, ntmp) : 0;
        // (A staged tile's values replace its keys while the scores are made, and the next
        // tile's rows are on their way.)
        if (STAGED) {
            src.put_v(tid);
            if (nnk > 0) src.fetch(nrow, ntmp, nnk, tid, p);
        }
        const bool fast = p.window == 0 && kb + C <= kbias && nk == C;
#pragma unroll
        for (int jj = 0; jj < NQ; jj++) {
            const int j = jj * NSG + sg;
            float s[NC];
            float m = -INFINITY;
            if (fast) {
#pragma unroll
                for (int ii = 0; ii < NC; ii++) {
                    s[ii] = ss[j * C + ii * 32 + lane];
                    m = max(m, s[ii]);
                }
            } else {
#pragma unroll
                for (int ii = 0; ii < NC; ii++) {
                    uint cc = ii * 32 + lane;
                    uint key = kb + cc;
                    float x = ss[j * C + cc];
                    uint dist = pi[jj] - key;
                    if (key <= pi[jj] && dist < p.extent && i0 + j < nt) {
                        device const float* rel = c + p.rel + dist;
                        device const float* rr = a + p.r + ((ulong)(i0 + j) * p.nh + h) * p.dr;
                        float b = 0.f;
                        for (uint d = 0; d < p.dr; d++) b += rr[d] * rel[d * p.extent];
                        x += b * M_LOG2E_F;
                    }
                    bool vis = (int)cc < nk && key <= pi[jj] && (p.window == 0 || dist < p.window);
                    s[ii] = vis ? x : -INFINITY;
                    m = max(m, s[ii]);
                }
            }
            m = simd_max(m);
            float ms = 1.f;
            if (m > M[jj] + 8.f) {
                ms = exp2(M[jj] - m);
                M[jj] = m;
                if (lane == 0) sf[0] = it + 1;
            }
            // (Each lane's share of the sums, added up at the end.)
            float sum = 0.f;
#pragma unroll
            for (int ii = 0; ii < NC; ii++) {
                ET pe = ET(exp2(s[ii] - M[jj]));
                sp[j * PS + ii * 32 + lane] = pe;
                sum += float(pe);
            }
            L[jj] = L[jj] * ms + sum;
            if (lane == 0) sr[j] = ms;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sf[0] == int(it + 1)) {
#pragma unroll
            for (uint16_t i = 0; i < co.get_capacity(); ++i)
                if (co.is_valid_element(i)) co[i] *= sr[co.get_multidimensional_index(i)[1]];
        }
        auto tv = src.vt(nk);
        mm_pv.run(tp, tv, co);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        kb = kn;
        row = nrow;
        tmp = ntmp;
        nk = nnk;
    }
    // The sink (a logit of a zero value) into each row's sum: the output factor per row.
#pragma unroll
    for (int jj = 0; jj < NQ; jj++) {
        const int j = jj * NSG + sg;
        float l = simd_sum(L[jj]);
        if (lane == 0) {
            float f = 0.f;
            if (i0 + j < nt) {
                float sl = a[p.sink + (ulong)(i0 + j) * p.nh + h] * M_LOG2E_F;
                float mf = max(M[jj], sl);
                float e = exp2(M[jj] - mf);
                f = e / (l * e + exp2(sl - mf));
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
// Queries Q and keys C a step, NSG SIMD-groups (`metal_inkling.rs`: FA_TILES; measured on M5 at
// 64k positions: f16 rows 32 / 128 / 4 (16 / 128 / 2 and 4, 16 and 32 / 64, two SIMD-groups, a
// key head's 3 query heads as 48 rows: all slower), int8 16 / 64 / 4 (32 queries, 32 keys, 2 or 8
// SIMD-groups, 3 heads as 48 rows: slower; dequantizing in half arithmetic: no faster).
#define INK_FA_ARGS(T) \
    device float* a [[buffer(0)]], device const float* c [[buffer(1)]], constant IAP& p [[buffer(2)]], \
    device const T* kc [[buffer(3)]], device const T* vc [[buffer(4)]], device const float* ks [[buffer(5)]], \
    device const float* vs [[buffer(6)]], device const T* tk [[buffer(7)]], device const T* tv [[buffer(8)]], \
    device const float* tks [[buffer(9)]], device const float* tvs [[buffer(10)]], uint2 tg [[threadgroup_position_in_grid]], \
    uint2 ntg [[threadgroups_per_grid]], ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]
#define INK_FA(NAME, Q, C, NSG, ET) \
kernel void NAME(INK_FA_ARGS(ET)) { \
    constexpr int HD = 128; \
    threadgroup float ss[Q * C]; \
    threadgroup float sr[Q]; \
    threadgroup int sf[1]; \
    uint kvh = tg.y / (p.nh / p.nkv); \
    InkFaDev<HD, C, ET> src{kc + kvh * HD, vc + kvh * HD, tk + kvh * HD, tv + kvh * HD, (int)p.kw, 0, false}; \
    ink_fa_body<HD, Q, C, NSG, ET, false>(a, c, p, src, tg, ntg, sg, lane, ss, sr, sf); }
INK_FA(ink_fa_h, 32, 128, 4, half)
INK_FA(ink_fa_f, 16, 128, 4, float)
kernel void ink_fa_q(INK_FA_ARGS(char)) {
    constexpr int HD = 128;
    threadgroup float ss[16 * 64];
    threadgroup float sr[16];
    threadgroup int sf[1];
    threadgroup half skv[64 * HD];
    uint kvh = tg.y / (p.nh / p.nkv);
    InkFaQ8<HD, 64, 128> src{kc, vc, ks, vs, tk, tv, tks, tvs, kvh * HD, p.kw, skv};
    ink_fa_body<HD, 16, 64, 4, half, true>(a, c, p, src, tg, ntg, sg, lane, ss, sr, sf);
}
