// ---- QSA (Qwen4-Exp sparse attention, see qwen4exp.rs and gpu_decode.rs `Op::QsaKey` / `QsaSelect`
// / `AttendSel`). Appended to the decode kernels (metal_decode.metal).
//
// Every float op of the indexer below is written in the CPU path's order without contraction, so
// given the same inputs the pooled keys, the normed queries and the block scores match it but for
// the RMSNorm factors: rsqrt here, powf(x, -0.5) there (Apple's is not correctly rounded), at
// most an ulp apart. The selection breaks ties exactly as the CPU path does (the lower block).
// OXINFER_QSA_CHECK compares each decode step's selections with the CPU path's (qwen4exp.rs).
#pragma clang fp reassociate(off)

// Attention over the positions listed in idx (the dec_attn_D_G kernels otherwise).
#define ATTN_SEL(D, G) \
kernel void dec_attn_sel_##D##_##G(device const float* q [[buffer(0)]], device const float* kc [[buffer(1)]], device const float* vc [[buffer(2)]], \
                         device float* out [[buffer(3)]], device const float* gate [[buffer(4)]], constant AP& p [[buffer(5)]], \
                         device float* part [[buffer(6)]], device const uint* idx [[buffer(7)]], \
                         uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float sm[16]; threadgroup float so[8 * D * 32]; threadgroup float qs[G * D * 32]; \
    attn_body<D, G, true>(q, kc, vc, out, gate, part, p, tg.x, tg.y, tid, sg, lane, sm, so, qs, idx); }
ATTN_SEL(2, 1) ATTN_SEL(2, 2) ATTN_SEL(2, 4) ATTN_SEL(2, 8) ATTN_SEL(4, 1) ATTN_SEL(4, 2) ATTN_SEL(4, 4) ATTN_SEL(8, 1) ATTN_SEL(8, 2) ATTN_SEL(16, 1)

struct QKP { uint d; uint r; uint half_rot; uint pos; uint t; float eps; float inv_r; uint k_stride; uint cs_stride; uint _p0; uint _p1; uint _p2; };
// Indexer keys of t tokens (one threadgroup, thread i owns dimension i, tokens in order): token j's
// raw key goes to key row pos + j; when it completes a block, value row pos + j gets the block's
// pooled key (mean, RMSNorm, RoPE with the cos / sin of the block's first position at
// cs + j * cs_stride), else zeros.
kernel void dec_qsa_key(device const float* k [[buffer(0)]], device float* kc [[buffer(1)]], device float* vc [[buffer(2)]],
                        device const float* w [[buffer(3)]], constant QKP& p [[buffer(4)]], constant float* cs [[buffer(5)]],
                        uint i [[thread_index_in_threadgroup]]) {
#pragma clang fp contract(off)
    threadgroup float m[512];
    threadgroup float rs[1];
    uint d = p.d;
    for (uint j = 0; j < p.t; j++) {
        uint pos = p.pos + j;
        float kv = k[(ulong)j * p.k_stride + i];
        kc[(ulong)pos * d + i] = kv;
        float out = 0.f;
        if ((pos + 1) % p.r == 0) {
            // The block's earlier raw keys: this thread wrote dimension i of each.
            float s = 0.f;
            for (uint q = pos + 1 - p.r; q < pos; q++) s += kc[(ulong)q * d + i];
            s += kv;
            s *= p.inv_r;
            m[i] = s;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (i == 0) {
                float ss = 0.f;
                for (uint a = 0; a < d; a++) ss += m[a] * m[a];
                rs[0] = precise::rsqrt(precise::divide(ss, float(d)) + p.eps);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float v = (m[i] * rs[0]) * w[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            m[i] = v;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            out = v;
            uint hr = p.half_rot;
            if (i < 2 * hr) {
                uint c = i % hr;
                float co = cs[j * p.cs_stride + c], si = cs[j * p.cs_stride + hr + c];
                float x1 = m[c], x2 = m[c + hr];
                out = i < hr ? x1 * co - x2 * si : x2 * co + x1 * si;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        vc[(ulong)pos * d + i] = out;
    }
}

struct QSP { uint nh; uint d; uint half_rot; uint nb; uint r; float eps; float scale; uint _p; };
// Block scores of one token: every threadgroup norms and rotates the queries (thread h < nh sums
// head h's squares), then thread b scores block b against its pooled key (value row b r + r - 1):
// per head the 8-accumulator dot of `crate::kernels::dot`, relu, summed over the heads, scaled.
kernel void dec_qsa_score(device const float* q [[buffer(0)]], device const float* vc [[buffer(1)]], device const float* w [[buffer(2)]],
                          device float* sc [[buffer(3)]], constant QSP& p [[buffer(4)]], constant float* cs [[buffer(5)]],
                          uint g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                          uint nt [[threads_per_threadgroup]]) {
#pragma clang fp contract(off)
    threadgroup float tmp[1024];
    threadgroup float qn[1024];
    threadgroup float qr[8];
    uint d = p.d, n = p.nh * d, hr = p.half_rot;
    if (tid < p.nh) {
        float ss = 0.f;
        for (uint a = 0; a < d; a++) {
            float v = q[tid * d + a];
            ss += v * v;
        }
        qr[tid] = precise::rsqrt(precise::divide(ss, float(d)) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < n; i += nt) tmp[i] = (q[i] * qr[i / d]) * w[i % d];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < n; i += nt) {
        uint h = i / d, c = i % d;
        float v = tmp[i];
        if (c < 2 * hr) {
            uint e = c % hr;
            float co = cs[e], si = cs[hr + e];
            float x1 = tmp[h * d + e], x2 = tmp[h * d + e + hr];
            v = c < hr ? x1 * co - x2 * si : x2 * co + x1 * si;
        }
        qn[i] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint b = g * nt + tid;
    if (b >= p.nb) return;
    device const float* kb = vc + ((ulong)b * p.r + p.r - 1) * d;
    float s = -0.0f;
    for (uint h = 0; h < p.nh; h++) {
        float acc[8] = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
        for (uint a = 0; a < d; a += 8) {
            for (uint l = 0; l < 8; l++) acc[l] += qn[h * d + a + l] * kb[a + l];
        }
        float dt = -0.0f;
        for (uint l = 0; l < 8; l++) dt += acc[l];
        s += max(dt, 0.f);
    }
    sc[b] = s * p.scale;
}

// Order-preserving key of a score (-0 as +0: the CPU path's comparison has them equal).
inline uint qsa_order(float s) {
    uint b = as_type<uint>(s);
    if (b == 0x80000000u) b = 0;
    return (b & 0x80000000u) != 0 ? ~b : (b | 0x80000000u);
}
struct QTP { uint nb; uint top; uint r; uint pos; };
// The top blocks of nb scores, ties to the lower block (the CPU path's order), one threadgroup: a
// radix select (8 bits a pass) finds T, the key of the top-th best block and how many blocks keyed T
// are taken (`need`, the lowest ones); thread t then takes from its chunk of blocks those keyed above
// T, and those keyed T while fewer than `need` blocks before took one. The positions go out in
// block order, `r` per block, then the incomplete tail up to pos.
kernel void dec_qsa_topk(device const float* sc [[buffer(0)]], device uint* sel [[buffer(1)]], constant QTP& p [[buffer(2)]],
                         uint tid [[thread_index_in_threadgroup]], uint nt [[threads_per_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup atomic_uint hist[256];
    threadgroup uint sh[2];
    threadgroup uint tot[2][32];
    uint nb = p.nb;
    uint prefix = 0, mask = 0, need = p.top;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (uint i = tid; i < 256; i += nt) atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint b = tid; b < nb; b += nt) {
            uint k = qsa_order(sc[b]);
            if ((k & mask) == prefix) atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            uint acc = 0;
            for (int dg = 255; dg >= 0; dg--) {
                uint c = atomic_load_explicit(&hist[dg], memory_order_relaxed);
                if (acc + c >= need) {
                    sh[0] = prefix | (uint(dg) << shift);
                    sh[1] = need - acc;
                    break;
                }
                acc += c;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix = sh[0];
        need = sh[1];
        mask |= 255u << shift;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint T = prefix;
    uint ch = (nb + nt - 1) / nt;
    uint lo = min(nb, tid * ch), hi = min(nb, lo + ch);
    uint ngt = 0, neq = 0;
    for (uint b = lo; b < hi; b++) {
        uint k = qsa_order(sc[b]);
        ngt += k > T ? 1u : 0u;
        neq += k == T ? 1u : 0u;
    }
    uint gx = simd_prefix_exclusive_sum(ngt), ex = simd_prefix_exclusive_sum(neq);
    if (lane == 31) {
        tot[0][sg] = gx + ngt;
        tot[1][sg] = ex + neq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 0; s < sg; s++) {
        gx += tot[0][s];
        ex += tot[1][s];
    }
    uint o = gx + min(ex, need);
    uint er = ex;
    uint r = p.r;
    for (uint b = lo; b < hi; b++) {
        uint k = qsa_order(sc[b]);
        bool take = k > T || (k == T && er < need);
        if (k == T) er++;
        if (take) {
            for (uint j = 0; j < r; j++) sel[o * r + j] = b * r + j;
            o++;
        }
    }
    uint tail = nb * r;
    for (uint q = tail + tid; q <= p.pos; q += nt) sel[p.top * r + q - tail] = q;
}
