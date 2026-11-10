// ---- QSA (Qwen4-Exp sparse attention, see qwen4exp.rs and gpu_decode.rs `Op::QsaKey` / `QsaSelect`
// / `AttendSel`), and the Qwen4-Exp prefill kernels (`pf_*`, metal_prefill.rs). Appended to the decode
// kernels (metal_decode.metal).
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

// Prefill attention over a host KV cache (gpu.rs `AttnJob`, QSA-aware): grid (head groups, tokens);
// token i (position start + i) attends to the meta[2 i + 1] positions listed at idx + meta[2 i], or
// with meta[2 i] == ~0 to positions 0 .. start + i. q (scaled) and out: [t, nh, hd]. One threadgroup
// takes all of a token's keys (the tokens give the parallelism).
#define ATTN_PF(D, G) \
kernel void pf_attn_##D##_##G(device const float* q [[buffer(0)]], device const float* kc [[buffer(1)]], device const float* vc [[buffer(2)]], \
                         device float* out [[buffer(3)]], constant AP& p0 [[buffer(4)]], device const uint* idx [[buffer(5)]], \
                         device const uint* meta [[buffer(6)]], constant uint& start [[buffer(7)]], \
                         uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float sm[16]; threadgroup float so[8 * D * 32]; threadgroup float qs[G * D * 32]; \
    uint i = tg.y; \
    AP p = p0; \
    p.lo = 0; p.nsplit = 1; p.has_gate = 0; \
    ulong qo = (ulong)i * p.nh * D * 32; \
    uint off = meta[2 * i]; \
    if (off == 0xffffffffu) { \
        p.len = start + i + 1; p.chunk = p.len; \
        attn_body<D, G, false, thread AP&>(q + qo, kc, vc, out + qo, q, (device float*)nullptr, p, tg.x, 0, tid, sg, lane, sm, so, qs); \
    } else { \
        p.len = meta[2 * i + 1]; p.chunk = p.len; \
        attn_body<D, G, true, thread AP&>(q + qo, kc, vc, out + qo, q, (device float*)nullptr, p, tg.x, 0, tid, sg, lane, sm, so, qs, idx + off); \
    } }
ATTN_PF(2, 1) ATTN_PF(2, 2) ATTN_PF(2, 4) ATTN_PF(2, 8) ATTN_PF(4, 1) ATTN_PF(4, 2) ATTN_PF(4, 4) ATTN_PF(8, 1) ATTN_PF(8, 2) ATTN_PF(16, 1)

// Prefill Gated DeltaNet rule (gpu.rs `GdnJob`) without the gated norm: threadgroup (value head hv,
// group g of 32 state columns), thread (part, jj) keeps rows 32 part .. 32 part + 31 of column
// 32 g + jj in registers over all the tokens (the columns evolve independently), writing each
// token's raw output column; the math is gdn_rule_body's. hdk / 32 * 32 = hdk threads.
kernel void pf_gdn_rule(device const float* c [[buffer(0)]], device const float* ba [[buffer(1)]], device float* out [[buffer(2)]],
                        device float* state [[buffer(3)]], device const float* cst [[buffer(4)]], constant GR& p [[buffer(5)]],
                        constant uint4& ts [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float qs[256], ks[256], red[256], red2[32];
    uint hv = tg.x, hdk = p.hdk, hdv = p.hdv, np = hdk / 32;
    uint hk = (p.tiled & 1u) != 0 ? hv % p.nk : hv / (p.nv / p.nk);
    uint part = tid / 32, jj = tid % 32, j = tg.y * 32 + jj;
    device float* S = state + (ulong)hv * hdk * hdv + part * 32 * hdv + j;
    float s[32];
    for (uint d = 0; d < 32; d++) s[d] = S[d * hdv];
    float dc = cst[hv], dtb = cst[p.nv + hv];
    for (uint t = 0; t < ts.x; t++) {
        device const float* ct = c + (ulong)t * ts.y;
        device const float* zt = ba + (ulong)t * ts.z;
        float q = ct[hk * hdk + tid], k = ct[p.key_dim + hk * hdk + tid];
        qs[tid] = q;
        ks[tid] = k;
        float qinv = 1.0f / sqrt(tg_sum(q * q, hdk, tid, sg, lane, red2) + 1e-6f) * p.qscale;
        float kinv = 1.0f / sqrt(tg_sum(k * k, hdk, tid, sg, lane, red2) + 1e-6f);
        float beta = 1.0f / (1.0f + exp(-zt[hv]));
        float ga = zt[p.nv + hv] + dtb;
        float sp = ga > 20.f ? ga : log(1.0f + exp(ga));
        float decay = exp(dc * sp);
        float kv = 0.f;
        for (uint d = 0; d < 32; d++) {
            s[d] *= decay;
            kv += s[d] * (ks[part * 32 + d] * kinv);
        }
        red[part * 32 + jj] = kv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float kvm = 0.f;
        for (uint i = 0; i < np; i++) kvm += red[i * 32 + jj];
        float delta = (ct[2 * p.key_dim + hv * hdv + j] - kvm) * beta;
        float o = 0.f;
        for (uint d = 0; d < 32; d++) {
            s[d] += (ks[part * 32 + d] * kinv) * delta;
            o += s[d] * (qs[part * 32 + d] * qinv);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        red[part * 32 + jj] = o;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (part == 0) {
            float ov = 0.f;
            for (uint i = 0; i < np; i++) ov += red[i * 32 + jj];
            out[(ulong)t * ts.w + hv * hdv + j] = ov;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint d = 0; d < 32; d++) S[d * hdv] = s[d];
}

// ---- Whole-layer prefill plans (metal_decode.rs, `MetalBackend::build_plan_pf`) ----------------
// Router top-k of several tokens (grid: one threadgroup per token): dec_topk's selection with the
// logits and selection rows ts.x and ts.y floats apart.
kernel void pf_topk(device const float* logits [[buffer(0)]], device float* sel [[buffer(1)]], constant uint4& p [[buffer(2)]],
                    device const float* scale [[buffer(3)]], constant uint4& ts [[buffer(4)]],
                    uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint nt [[threads_per_threadgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float bv[32];
    threadgroup uint bi[32];
    threadgroup float chosen[64];
    logits += (ulong)t * ts.x;
    sel += (ulong)t * ts.y;
    uint n = p.x, k = p.y;
    float v = tid < n ? logits[tid] : -INFINITY;
    uint nsg = (nt + 31) / 32;
    for (uint j = 0; j < k; j++) {
        float m = simd_max(v);
        uint idx = v == m ? tid : 0xffffffffu;
        idx = simd_min(idx);
        if (lane == 0) { bv[sg] = m; bi[sg] = idx; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float best = bv[0]; uint bidx = bi[0];
        for (uint i = 1; i < nsg; i++) if (bv[i] > best || (bv[i] == best && bi[i] < bidx)) { best = bv[i]; bidx = bi[i]; }
        if (tid == bidx) v = -INFINITY;
        if (tid == 0) { sel[j] = as_type<float>(bidx); chosen[j] = best; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float m = chosen[0], s = 0.f;
        for (uint j = 0; j < k; j++) { float e = exp(chosen[j] - m); chosen[j] = e; s += e; }
        for (uint j = 0; j < k; j++) sel[k + j] = chosen[j] / s * (p.z != 0 ? scale[as_type<uint>(sel[j])] : 1.f);
    }
}

// Gated DeltaNet's gated RMSNorm in place after pf_gdn_rule: grid (value head, token), hdv threads;
// o = o / sqrt(mean(o^2) + eps) * w * gate(z), gate sigmoid (p.y != 0) or SiLU. ts: strides of out
// and z.
kernel void pf_gated_norm(device float* out [[buffer(0)]], device const float* z [[buffer(1)]], device const float* w [[buffer(2)]],
                          constant float2& pe [[buffer(3)]], constant uint4& ts [[buffer(4)]],
                          uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[32];
    uint hdv = ts.z;
    device float* o = out + (ulong)tg.y * ts.x + tg.x * hdv;
    device const float* zz = z + (ulong)tg.y * ts.y + tg.x * hdv;
    float v = o[tid];
    float r = 1.0f / sqrt(tg_sum(v * v, hdv, tid, sg, lane, red) / float(hdv) + pe.x);
    float g = zz[tid];
    float gate = ts.w != 0 ? 1.0f / (1.0f + precise::exp(-g)) : g / (1.0f + exp(-g));
    o[tid] = v * r * w[tid] * gate;
}

// y (rows ts.y floats apart) += x (rows of n, ts.x apart) * (ts.w != 0 ? sigmoid(gate[t ts.z]) : 1):
// a GEMM's output added (gated: the shared expert).
kernel void pf_add_rows(device float* y [[buffer(0)]], device const float* x [[buffer(1)]], device const float* gate [[buffer(2)]],
                        constant uint& n [[buffer(3)]], constant uint4& ts [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x, t = gid.y;
    if (i >= n) return;
    float s = ts.w != 0 ? 1.0f / (1.0f + exp(-gate[(ulong)t * ts.z])) : 1.0f;
    y[(ulong)t * ts.y + i] += x[(ulong)t * ts.x + i] * s;
}

// Prefill attention of a chunk over the device KV cache: grid (head groups, tokens); token i
// (position a.x + i) attends to positions 0 .. a.x + i, or with QSA (a.y = block size, a.z = blocks
// kept) past a.z blocks to the positions its selection row lists (sel + i * a.w; the top blocks
// then the incomplete tail). q, out and gate rows qs.x, qs.y, qs.z floats apart; p0 as dec_attn's.
struct PFA { uint start; uint r; uint top; uint ss; uint sq; uint so; uint sgate; uint _p; };
#define ATTN_PFX(D, G) \
kernel void pfa_##D##_##G(device const float* q [[buffer(0)]], device const float* kc [[buffer(1)]], device const float* vc [[buffer(2)]], \
                          device float* out [[buffer(3)]], device const float* gate [[buffer(4)]], constant AP& p0 [[buffer(5)]], \
                          device const uint* sel [[buffer(6)]], constant PFA& a [[buffer(7)]], \
                          uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float sm[16]; threadgroup float so[8 * D * 32]; threadgroup float qs[G * D * 32]; \
    uint i = tg.y, pos = a.start + i; \
    AP p = p0; \
    p.lo = 0; p.nsplit = 1; \
    device const float* qi = q + (ulong)i * a.sq; device float* oi = out + (ulong)i * a.so; \
    device const float* gi = gate + (ulong)i * a.sgate; \
    uint nb = a.r != 0 ? (pos + 1) / a.r : 0; \
    if (nb <= a.top) { \
        p.len = pos + 1; p.chunk = p.len; \
        attn_body<D, G, false, thread AP&>(qi, kc, vc, oi, gi, (device float*)nullptr, p, tg.x, 0, tid, sg, lane, sm, so, qs); \
    } else { \
        p.len = a.top * a.r + pos + 1 - nb * a.r; p.chunk = p.len; \
        attn_body<D, G, true, thread AP&>(qi, kc, vc, oi, gi, (device float*)nullptr, p, tg.x, 0, tid, sg, lane, sm, so, qs, sel + (ulong)i * a.ss); \
    } }
ATTN_PFX(2, 1) ATTN_PFX(2, 2) ATTN_PFX(2, 4) ATTN_PFX(2, 8) ATTN_PFX(4, 1) ATTN_PFX(4, 2) ATTN_PFX(4, 4) ATTN_PFX(8, 1) ATTN_PFX(8, 2) ATTN_PFX(16, 1) ATTN_PFX(8, 4) ATTN_PFX(4, 8)

// Causal depthwise conv of a chunk in parallel (dec_gdn_conv_t's results; its sums in the same
// order): grid (channels, tokens); input i of the window is state row i, then x row i - (kc - 1).
// ts: x and out strides (y, z). pf_conv_state then shifts the chunk into the state.
kernel void pf_conv(device const float* x [[buffer(0)]], device float* out [[buffer(1)]], device const float* st [[buffer(2)]],
                    device const float* w [[buffer(3)]], constant GC& p [[buffer(4)]], constant uint4& ts [[buffer(5)]],
                    uint2 gid [[thread_position_in_grid]]) {
    uint ch = gid.x, t = gid.y;
    if (ch >= p.dim) return;
    uint kc = p.kc, hs = kc - 1;
    float s = 0.f;
    for (uint j = 0; j < kc; j++) {
        uint i = t + j;
        float v = i < hs ? st[i * p.dim + ch] : x[(ulong)(i - hs) * ts.y + ch];
        s += w[ch * kc + j] * v;
    }
    out[(ulong)t * ts.z + ch] = s / (1.0f + exp(-s));
}
kernel void pf_conv_state(device const float* x [[buffer(0)]], device float* st [[buffer(2)]], constant GC& p [[buffer(4)]],
                          constant uint4& ts [[buffer(5)]], uint ch [[thread_position_in_grid]]) {
    if (ch >= p.dim) return;
    uint hs = p.kc - 1, t = ts.x;
    for (uint j = 0; j < hs; j++) {
        uint i = t + j;
        st[j * p.dim + ch] = i < hs ? st[i * p.dim + ch] : x[(ulong)(i - hs) * ts.y + ch];
    }
}

// dec_qsa_key over a chunk with a threadgroup per token: the earlier raw keys of a block come
// from the chunk's inputs when inside it (other threadgroups write them to the cache), else from
// the cache; the same sums and rounding.
kernel void pf_qsa_key(device const float* k [[buffer(0)]], device float* kc [[buffer(1)]], device float* vc [[buffer(2)]],
                       device const float* w [[buffer(3)]], constant QKP& p [[buffer(4)]], device const float* cs [[buffer(5)]],
                       uint j [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]]) {
#pragma clang fp contract(off)
    threadgroup float m[512];
    threadgroup float rs[1];
    uint d = p.d;
    uint pos = p.pos + j;
    float kv = k[(ulong)j * p.k_stride + i];
    kc[(ulong)pos * d + i] = kv;
    float out = 0.f;
    if ((pos + 1) % p.r == 0) {
        float s = 0.f;
        for (uint q = pos + 1 - p.r; q < pos; q++) s += q >= p.pos ? k[(ulong)(q - p.pos) * p.k_stride + i] : kc[(ulong)q * d + i];
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
    }
    vc[(ulong)pos * d + i] = out;
}

// The Gated DeltaNet rule in two steps. pf_gdn_prep, grid (key heads, tokens) of hdk threads:
// each token's normalized (and for q scaled) q and k per key head, and per value head beta and the
// decay, into rows of g: [q: key_dim][k: key_dim][beta: nv][decay: nv] (ts.w floats apart).
kernel void pf_gdn_prep(device const float* c [[buffer(0)]], device const float* ba [[buffer(1)]], device float* g [[buffer(2)]],
                        device const float* cst [[buffer(4)]], constant GR& p [[buffer(5)]], constant uint4& ts [[buffer(6)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red2[32];
    uint hk = tg.x, t = tg.y, hdk = p.hdk;
    device const float* ct = c + (ulong)t * ts.y;
    device float* gt = g + (ulong)t * ts.w;
    float q = ct[hk * hdk + tid], k = ct[p.key_dim + hk * hdk + tid];
    float qinv = 1.0f / sqrt(tg_sum(q * q, hdk, tid, sg, lane, red2) + 1e-6f) * p.qscale;
    float kinv = 1.0f / sqrt(tg_sum(k * k, hdk, tid, sg, lane, red2) + 1e-6f);
    gt[hk * hdk + tid] = q * qinv;
    gt[p.key_dim + hk * hdk + tid] = k * kinv;
    if (hk == 0 && tid < p.nv) {
        device const float* zt = ba + (ulong)t * ts.z;
        float beta = 1.0f / (1.0f + exp(-zt[tid]));
        float ga = zt[p.nv + tid] + cst[p.nv + tid];
        float sp = ga > 20.f ? ga : log(1.0f + exp(ga));
        gt[2 * p.key_dim + tid] = beta;
        gt[2 * p.key_dim + p.nv + tid] = exp(cst[tid] * sp);
    }
}
// pf_gdn_seq, grid (value heads, hdv / 32) of 4 * 32 threads: lane = 4 column + part, thread keeps
// rows 32 part .. 32 part + 31 of its column of the state in registers; the four parts of a column
// are lanes of one SIMD-group, so the sums over the rows need no barrier. ts: (t, c stride, out
// stride, g stride).
kernel void pf_gdn_seq(device const float* c [[buffer(0)]], device const float* g [[buffer(1)]], device float* out [[buffer(2)]],
                       device float* state [[buffer(3)]], constant GR& p [[buffer(5)]], constant uint4& ts [[buffer(6)]],
                       uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    uint hv = tg.x, hdk = p.hdk, hdv = p.hdv;
    uint hk = (p.tiled & 1u) != 0 ? hv % p.nk : hv / (p.nv / p.nk);
    uint part = lane % 4, j = tg.y * 32 + sg * 8 + lane / 4;
    uint r0 = part * 32;
    device float* S = state + (ulong)hv * hdk * hdv + (ulong)r0 * hdv + j;
    float s[32];
    for (uint d = 0; d < 32; d++) s[d] = S[d * hdv];
    for (uint t = 0; t < ts.x; t++) {
        device const float* gt = g + (ulong)t * ts.w;
        device const float* qv = gt + hk * hdk + r0;
        device const float* kv4 = gt + p.key_dim + hk * hdk + r0;
        float beta = gt[2 * p.key_dim + hv], decay = gt[2 * p.key_dim + p.nv + hv];
        // pf_gdn_rule's sums: in row order within a part, then the parts in order.
        float kv = 0.f;
        for (uint d = 0; d < 32; d++) {
            s[d] *= decay;
            kv += s[d] * kv4[d];
        }
        uint b = lane & ~3u;
        float kvm = ((simd_shuffle(kv, b) + simd_shuffle(kv, b + 1)) + simd_shuffle(kv, b + 2)) + simd_shuffle(kv, b + 3);
        float delta = (c[(ulong)t * ts.y + 2 * p.key_dim + hv * hdv + j] - kvm) * beta;
        float o = 0.f;
        for (uint d = 0; d < 32; d++) {
            s[d] += kv4[d] * delta;
            o += s[d] * qv[d];
        }
        float ov = ((simd_shuffle(o, b) + simd_shuffle(o, b + 1)) + simd_shuffle(o, b + 2)) + simd_shuffle(o, b + 3);
        if (part == 0) out[(ulong)t * ts.z + hv * hdv + j] = ov;
    }
    for (uint d = 0; d < 32; d++) S[d * hdv] = s[d];
}
