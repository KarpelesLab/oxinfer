// DeepSeek V4.1 single-token decode on Metal (metal_ds41.rs): the ops of a layer that are not
// matvecs (those run on the decode matvec kernels of metal_decode.metal). f32 throughout; the
// transcendental functions are the precise ones, as the CPU path's.
#include <metal_stdlib>
using namespace metal;

inline float d41_sigmoid(float x) { return 1.0f / (1.0f + precise::exp(-x)); }

// Sum of v over a threadgroup of nt threads (<= 1024); every thread gets the total. `red` holds
// 32 floats.
inline float d41_tg_sum(float v, threadgroup float* red, uint tid, uint nt, uint sg, uint lane) {
    v = simd_sum(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) red[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = 0.f;
    uint nsg = (nt + 31) / 32;
    for (uint i = 0; i < nsg; i++) t += red[i];
    return t;
}

// Sum over the lanes that differ from this one in the bits from `lo` up to (excluding) `hi`.
inline float d41_xsum(float x, uint lo, uint hi) {
    for (uint o = lo; o < hi; o <<= 1) x += simd_shuffle_xor(x, o);
    return x;
}

// ---- Hyper-connection mixing weights --------------------------------------------------------
// Dot products of `rows` rows of w (each `len` floats) with x, plus x . x as row `rows`, in
// `split` parts: grid (rows + 1, split), 256 threads; part[r * split + s].
struct DotsP { uint len; uint rows; uint split; uint _p; };
kernel void d41_dots(device const float* w [[buffer(0)]], device const float* x [[buffer(1)]], device float* part [[buffer(2)]],
                     constant DotsP& p [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[32];
    uint r = tg.x, s = tg.y;
    uint chunk = p.len / p.split;
    uint b = s * chunk;
    device const float4* xv = (device const float4*)(x + b);
    device const float4* wv = (device const float4*)((r < p.rows ? w + (ulong)r * p.len : x) + b);
    float4 a = 0.f;
    for (uint i = tid; i < chunk / 4; i += 256) a = fma(wv[i], xv[i], a);
    float t = d41_tg_sum((a.x + a.y) + (a.z + a.w), red, tid, 256, sg, lane);
    if (tid == 0) part[r * p.split + s] = t;
}

// The mixing weights from the dot products (DeepSeek's hc_split_sinkhorn): out = pre [n], post
// [n], comb [n * n] (row j = source copy j). One thread.
struct MixP { uint n; uint split; uint len; uint iters; float eps; float heps; uint _p0; uint _p1; };
kernel void d41_mix_fin(device const float* part [[buffer(0)]], device const float* base [[buffer(1)]],
                        device const float* scale [[buffer(2)]], device float* out [[buffer(3)]], constant MixP& p [[buffer(4)]],
                        uint tid [[thread_index_in_threadgroup]]) {
    uint n = p.n, m = (2 + n) * n;
    if (m <= 32 && n * n <= 32 && (n & (n - 1)) == 0) {
        // One SIMD-group: lane i holds mixing value i; the Sinkhorn sums over a row (lanes
        // j n .. j n + n - 1) and over a column (lanes k, k + n, ...) by shuffles.
        uint lane = tid;
        float ss = 0.f;
        for (uint s = 0; s < p.split; s++) ss += part[m * p.split + s];
        float r = 1.0f / precise::sqrt(ss / float(p.len) + p.eps);
        float v = 0.f;
        if (lane < m)
            for (uint s = 0; s < p.split; s++) v += part[lane * p.split + s];
        float mm = v * r;
        if (lane < n) out[lane] = d41_sigmoid(mm * scale[0] + base[lane]) + p.heps;
        else if (lane < 2 * n) out[lane] = 2.0f * d41_sigmoid(mm * scale[1] + base[lane]);
        // Lane l of the comb takes mixing value 2 n + l.
        float c = simd_shuffle_down(mm, 2 * n);
        bool in = lane < n * n;
        c = in ? c * scale[2] + base[min(2 * n + lane, m - 1)] : 0.f;
        float mx = in ? c : -INFINITY;
        for (uint o = 1; o < n; o <<= 1) mx = max(mx, simd_shuffle_xor(mx, o));
        float e = in ? precise::exp(c - mx) : 0.f;
        c = in ? e / d41_xsum(e, 1, n) + p.heps : 0.f;
        for (uint it = 0; it < p.iters; it++) {
            if (it > 0) c = in ? c / (d41_xsum(c, 1, n) + p.heps) : 0.f;
            c = in ? c / (d41_xsum(c, n, n * n) + p.heps) : 0.f;
        }
        if (in) out[2 * n + lane] = c;
        return;
    }
    if (tid != 0) return;
    float ss = 0.f;
    for (uint s = 0; s < p.split; s++) ss += part[m * p.split + s];
    float r = 1.0f / precise::sqrt(ss / float(p.len) + p.eps);
    float mm[64];
    for (uint i = 0; i < m; i++) {
        float v = 0.f;
        for (uint s = 0; s < p.split; s++) v += part[i * p.split + s];
        mm[i] = v * r;
    }
    float c[16];
    for (uint i = 0; i < n; i++) {
        out[i] = d41_sigmoid(mm[i] * scale[0] + base[i]) + p.heps;
        out[n + i] = 2.0f * d41_sigmoid(mm[n + i] * scale[1] + base[n + i]);
    }
    for (uint i = 0; i < n * n; i++) c[i] = mm[2 * n + i] * scale[2] + base[2 * n + i];
    for (uint j = 0; j < n; j++) {
        float mx = -INFINITY;
        for (uint k = 0; k < n; k++) mx = max(mx, c[j * n + k]);
        float s = 0.f;
        for (uint k = 0; k < n; k++) { c[j * n + k] = precise::exp(c[j * n + k] - mx); s += c[j * n + k]; }
        for (uint k = 0; k < n; k++) c[j * n + k] = c[j * n + k] / s + p.heps;
    }
    for (uint it = 0; it < p.iters; it++) {
        if (it > 0)
            for (uint j = 0; j < n; j++) {
                float s = 0.f;
                for (uint k = 0; k < n; k++) s += c[j * n + k];
                s += p.heps;
                for (uint k = 0; k < n; k++) c[j * n + k] /= s;
            }
        for (uint k = 0; k < n; k++) {
            float s = 0.f;
            for (uint j = 0; j < n; j++) s += c[j * n + k];
            s += p.heps;
            for (uint j = 0; j < n; j++) c[j * n + k] /= s;
        }
    }
    for (uint i = 0; i < n * n; i++) out[2 * n + i] = c[i];
}

// out = rmsnorm(sum_k pre[k] x[k]) * w over rows of h (one threadgroup of 1024, h <= 8192).
struct CollP { uint h; uint n; float eps; uint _p; };
kernel void d41_collapse(device const float* x [[buffer(0)]], device const float* pre [[buffer(1)]], device const float* w [[buffer(2)]],
                         device float* out [[buffer(3)]], constant CollP& p [[buffer(4)]],
                         uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[32];
    float y[8];
    float ss = 0.f;
    for (uint j = 0; j < 8; j++) {
        uint i = tid + 1024 * j;
        float v = 0.f;
        if (i < p.h) {
            for (uint k = 0; k < p.n; k++) v = fma(pre[k], x[k * p.h + i], v);
        }
        y[j] = v;
        ss += v * v;
    }
    float t = d41_tg_sum(ss, red, tid, 1024, sg, lane);
    float sc = 1.0f / precise::sqrt(t / float(p.h) + p.eps);
    for (uint j = 0; j < 8; j++) {
        uint i = tid + 1024 * j;
        if (i < p.h) out[i] = y[j] * sc * w[i];
    }
}

// Rows of d (grid = rows, 256 threads, d <= 4096): optionally RMS-normed (flags bit 0; weighted
// with bit 1), then the last 2 half dimensions rotated as interleaved pairs by cs (cos [half]
// then sin [half]; backwards with bit 2); src row r at r * ss, dst row at r * ds.
struct NrP { uint d; uint flags; uint half_; uint ss; uint ds; float eps; uint _p0; uint _p1; };
kernel void d41_norm_rope(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]], device const float* w [[buffer(2)]],
                          device const float* cs [[buffer(3)]], constant NrP& p [[buffer(4)]],
                          uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[32];
    device const float* x = src + (ulong)row * p.ss;
    device float* y = dst + (ulong)row * p.ds;
    float sc = 1.f;
    if (p.flags & 1u) {
        float ss = 0.f;
        for (uint i = tid; i < p.d; i += 256) ss += x[i] * x[i];
        float t = d41_tg_sum(ss, red, tid, 256, sg, lane);
        sc = 1.0f / precise::sqrt(t / float(p.d) + p.eps);
    }
    uint t0 = p.d - 2 * p.half_;
    for (uint q = tid; q < p.d / 2; q += 256) {
        uint i0 = 2 * q;
        float a = x[i0], b = x[i0 + 1];
        if (p.flags & 1u) {
            a = a * sc; b = b * sc;
            if (p.flags & 2u) { a = a * w[i0]; b = b * w[i0 + 1]; }
        }
        if (i0 >= t0) {
            uint k = (i0 - t0) / 2;
            float c = cs[k], s = (p.flags & 4u) ? -cs[p.half_ + k] : cs[p.half_ + k];
            float a2 = a * c - b * s, b2 = b * c + a * s;
            a = a2; b = b2;
        }
        y[i0] = a;
        y[i0 + 1] = b;
    }
}

// A key source's latent: per channel the softmax over the r gate rows of the group pooling the
// r kv rows (slots in position order; r = 1: the row itself), then RMS-normed with w. One
// threadgroup of 1024, d <= 1024.
struct PoolP { uint d; uint r; float eps; uint _p; };
kernel void d41_pool(device const float* kv [[buffer(0)]], device const float* sc [[buffer(1)]], device const float* w [[buffer(2)]],
                     device float* out [[buffer(3)]], constant PoolP& p [[buffer(4)]],
                     uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[32];
    float v = 0.f;
    if (tid < p.d) {
        float mx = -INFINITY;
        for (uint j = 0; j < p.r; j++) mx = max(mx, sc[j * p.d + tid]);
        float tot = 0.f, acc = 0.f;
        for (uint j = 0; j < p.r; j++) {
            float e = precise::exp(sc[j * p.d + tid] - mx);
            tot += e;
            acc += e * kv[j * p.d + tid];
        }
        v = acc / tot;
    }
    float t = d41_tg_sum(v * v, red, tid, 1024, sg, lane);
    float s = 1.0f / precise::sqrt(t / float(p.d) + p.eps);
    if (tid < p.d) out[tid] = v * s * w[tid];
}

// ---- Attention: the window (ring of W rows) then compressed keys (all, or the chosen ones) --
// Keys are the values (one 512-wide head shared by every query head), with a sink per head.
// Partials: grid (nh / 8, splits), 256 threads; SIMD-group sg takes head 8 tg.x + sg over keys
// [chunk tg.y, chunk (tg.y + 1)). Lane l holds dimensions l + 32 m of the query (rotated) and
// of the running output. part[(h * splits + s) * (512 + 2)] = (max, sum, acc[512]).
struct AttnP { uint nh; uint nw; uint lo; uint win; uint nc; uint use_sel; uint chunk; uint half_; float scale; uint splits; uint _p0; uint _p1; };
kernel void d41_attn(device const float* q [[buffer(0)]], device const float* ring [[buffer(1)]], device const float* comp [[buffer(2)]],
                     device const uint* sel [[buffer(3)]], device const float* cs [[buffer(4)]], device float* part [[buffer(5)]],
                     constant AttnP& p [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint HD = 512;
    uint h = tg.x * 8 + sg;
    if (h >= p.nh) return;
    device const float* qh = q + (ulong)h * HD;
    float qv[16];
    uint t0 = HD - 2 * p.half_;
    for (uint m = 0; m < 16; m++) {
        uint d = lane + 32 * m;
        float v = qh[d];
        float o = simd_shuffle_xor(v, 1);
        if (d >= t0) {
            uint k = (d - t0) / 2;
            float c = cs[k], s = cs[p.half_ + k];
            v = (d & 1u) ? v * c + o * s : v * c - o * s;
        }
        qv[m] = v;
    }
    float mx = -INFINITY, l = 0.f;
    float acc[16];
    for (uint m = 0; m < 16; m++) acc[m] = 0.f;
    uint total = p.nw + p.nc;
    uint j0 = tg.y * p.chunk, j1 = min(total, j0 + p.chunk);
    for (uint j = j0; j < j1; j++) {
        device const float* kr;
        if (j < p.nw) kr = ring + (ulong)((p.lo + j) % p.win) * HD;
        else {
            uint c = j - p.nw;
            kr = comp + (ulong)(p.use_sel != 0 ? sel[c] : c) * HD;
        }
        float kv[16];
        float s = 0.f;
        for (uint m = 0; m < 16; m++) { kv[m] = kr[lane + 32 * m]; s = fma(qv[m], kv[m], s); }
        s = simd_sum(s) * p.scale;
        float nm = max(mx, s);
        float corr = precise::exp(mx - nm), e = precise::exp(s - nm);
        l = l * corr + e;
        for (uint m = 0; m < 16; m++) acc[m] = acc[m] * corr + e * kv[m];
        mx = nm;
    }
    device float* o = part + (ulong)(h * p.splits + tg.y) * (HD + 2);
    if (lane == 0) { o[0] = mx; o[1] = l; }
    for (uint m = 0; m < 16; m++) o[2 + lane + 32 * m] = acc[m];
}
// Combines the partials of each head with its sink, then undoes the rotation of the output's tail
// (grid nh, 256 threads: thread t takes dimensions 2t, 2t + 1).
kernel void d41_attn_comb(device const float* part [[buffer(0)]], device const float* sink [[buffer(1)]], device const float* cs [[buffer(2)]],
                          device float* out [[buffer(3)]], constant AttnP& p [[buffer(4)]],
                          uint h [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint HD = 512;
    device const float* ph = part + (ulong)h * p.splits * (HD + 2);
    float M = sink[h];
    for (uint s = 0; s < p.splits; s++) M = max(M, ph[s * (HD + 2)]);
    float L = precise::exp(sink[h] - M);
    float a = 0.f, b = 0.f;
    for (uint s = 0; s < p.splits; s++) {
        device const float* r = ph + s * (HD + 2);
        float f = r[0] == -INFINITY ? 0.f : precise::exp(r[0] - M);
        L += r[1] * f;
        a += r[2 + 2 * tid] * f;
        b += r[3 + 2 * tid] * f;
    }
    a /= L; b /= L;
    uint t0 = HD - 2 * p.half_, d = 2 * tid;
    if (d >= t0) {
        uint k = (d - t0) / 2;
        float c = cs[k], s = -cs[p.half_ + k];
        float a2 = a * c - b * s, b2 = b * c + a * s;
        a = a2; b = b2;
    }
    out[(ulong)h * HD + d] = a;
    out[(ulong)h * HD + d + 1] = b;
}

// ---- After a block: x_out[k] = post[k] (o1 + o2) + sum_j comb[j][k] x_in[j] -----------------
struct PostP { uint h; uint n; uint has_o2; uint _p; };
kernel void d41_hc_post(device const float* xin [[buffer(0)]], device const float* o1 [[buffer(1)]], device const float* o2 [[buffer(2)]],
                        device const float* mix [[buffer(3)]], device float* xout [[buffer(4)]], constant PostP& p [[buffer(5)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= p.h) return;
    uint n = p.n;
    float f = o1[i];
    if (p.has_o2 != 0) f += o2[i];
    float res[8];
    for (uint j = 0; j < n; j++) res[j] = xin[j * p.h + i];
    for (uint k = 0; k < n; k++) {
        float v = mix[n + k] * f;
        for (uint j = 0; j < n; j++) v = fma(mix[2 * n + j * n + k], res[j], v);
        xout[k * p.h + i] = v;
    }
}

// ---- Engram: each copy k gets value * sigmoid(sign(d) sqrt(|d|)), d the normalized product of
// copy and key k (kv: n keys then the value, h each). Grid n, 1024 threads.
struct EngP { uint h; uint n; float eps; uint _p; };
kernel void d41_engram(device const float* xin [[buffer(0)]], device const float* kv [[buffer(1)]], device const float* w [[buffer(2)]],
                       device float* xout [[buffer(3)]], constant EngP& p [[buffer(4)]],
                       uint k [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[32];
    uint h = p.h;
    device const float* xc = xin + (ulong)k * h;
    device const float* key = kv + (ulong)k * h;
    device const float* wk = w + (ulong)k * h;
    device const float* val = kv + (ulong)p.n * h;
    float d = 0.f, xx = 0.f, kk = 0.f;
    for (uint i = tid; i < h; i += 1024) {
        d += xc[i] * wk[i] * key[i];
        xx += xc[i] * xc[i];
        kk += key[i] * key[i];
    }
    d = d41_tg_sum(d, red, tid, 1024, sg, lane);
    xx = d41_tg_sum(xx, red, tid, 1024, sg, lane);
    kk = d41_tg_sum(kk, red, tid, 1024, sg, lane);
    float rstd = 1.0f / (precise::sqrt(xx / float(h) + p.eps) * precise::sqrt(kk / float(h) + p.eps));
    d = d * rstd * precise::rsqrt(float(h));
    float g = d41_sigmoid(copysign(precise::sqrt(max(abs(d), 1e-6f)), d));
    for (uint i = tid; i < h; i += 1024) xout[(ulong)k * h + i] = fma(g, val[i], xc[i]);
}

// ---- MoE routing -----------------------------------------------------------------------------
// Scores of n_exp logits (0 softmax, 1 sigmoid, 2 sqrt(softplus)), the top k of score + bias
// (ties to the lower index), weights renormalized (norm != 0) times rsf. A chosen expert whose
// gate address in `tab` (32-byte entries) is null is a miss: it is recorded in `miss` (count,
// then layer << 16 | expert) and replaced by expert n_exp (zeros) with weight 0. sel: k ids (uint
// bits), k weights, the SwiGLU limit. One SIMD-group (n_exp <= 1024): lane l holds experts
// l + 32 i.
struct RouteP { uint n; uint k; float rsf; float limit; uint scoring; uint norm; uint layer; uint has_bias; };
inline float d41_score(float lg, uint scoring) {
    if (scoring == 1) return d41_sigmoid(lg);
    float u = precise::exp(lg);
    float y = 1.0f + u;
    float l1 = y == 1.0f ? u : precise::log(y) - ((y - 1.0f) - u) / y;
    return precise::sqrt(lg > 20.f ? lg : l1);
}
kernel void d41_route(device const float* logits [[buffer(0)]], device const float* bias [[buffer(1)]], device const ulong* tab [[buffer(2)]],
                      device float* sel [[buffer(3)]], device atomic_uint* miss [[buffer(4)]], constant RouteP& p [[buffer(5)]],
                      uint lane [[thread_index_in_simdgroup]]) {
    const uint PER = 32;
    float sc[PER], v[PER];
    uint cnt = (p.n + 31) / 32;
    float mx = -INFINITY, tot = 0.f;
    for (uint i = 0; i < PER; i++) {
        uint e = lane + 32 * i;
        float lg = (i < cnt && e < p.n) ? logits[e] : -INFINITY;
        sc[i] = lg;
        mx = max(mx, lg);
    }
    if (p.scoring == 0) {
        mx = simd_max(mx);
        for (uint i = 0; i < PER; i++) {
            sc[i] = sc[i] == -INFINITY ? 0.f : precise::exp(sc[i] - mx);
            tot += sc[i];
        }
        tot = simd_sum(tot);
    }
    for (uint i = 0; i < PER; i++) {
        uint e = lane + 32 * i;
        bool ok = i < cnt && e < p.n;
        if (p.scoring == 0) sc[i] = sc[i] / tot;
        else if (ok) sc[i] = d41_score(sc[i], p.scoring);
        v[i] = ok ? sc[i] + (p.has_bias != 0 ? bias[e] : 0.f) : -INFINITY;
    }
    uint chosen[16];
    float cs[16];
    for (uint j = 0; j < p.k; j++) {
        // This lane's best (the lowest index among equals), then the SIMD-group's.
        float b = -INFINITY;
        uint bi = 0xffffffffu, bs = 0;
        for (uint i = 0; i < PER; i++)
            if (v[i] > b) { b = v[i]; bi = lane + 32 * i; bs = i; }
        float m = simd_max(b);
        uint idx = simd_min(b == m ? bi : 0xffffffffu);
        if (bi == idx) v[bs] = -INFINITY;
        chosen[j] = idx;
        // The winner's score.
        float s = 0.f;
        for (uint i = 0; i < PER; i++) s = (lane + 32 * i == idx) ? sc[i] : s;
        cs[j] = simd_sum(s);
    }
    if (lane != 0) return;
    float t = 0.f;
    for (uint j = 0; j < p.k; j++) t += cs[j];
    t += 1e-20f;
    for (uint j = 0; j < p.k; j++) {
        uint e = chosen[j];
        float w = (p.norm != 0 && p.k > 1 ? cs[j] / t : cs[j]) * p.rsf;
        if (tab[4 * e] == 0) {
            uint at = atomic_fetch_add_explicit(&miss[0], 1u, memory_order_relaxed);
            atomic_store_explicit(&miss[1 + at], (p.layer << 16) | e, memory_order_relaxed);
            e = p.n;
            w = 0.f;
        }
        sel[j] = as_type<float>(e);
        sel[p.k + j] = w;
    }
    sel[2 * p.k] = p.limit;
}

// ---- Lightning indexer ----------------------------------------------------------------------
// Scores of nv index keys (di = 128 wide) for 32 heads of queries iq (rotated here) with weights
// iw * ws: s[w] = sum_h iw_h ws relu(q_h . k_w); keys of blocks (bs positions) with mask[b] == 0
// score -inf when has_mask. Grid ceil(nv / 64), 256 threads: lane = head, 8 keys per SIMD-group.
struct IdxP { uint nv; uint half_; float ws; uint bs; uint has_mask; uint _p0; uint _p1; uint _p2; };
kernel void d41_idx_score(device const float* iq [[buffer(0)]], device const float* iw [[buffer(1)]], device const float* keys [[buffer(2)]],
                          device const float* cs [[buffer(3)]], device const uint* mask [[buffer(4)]], device float* scores [[buffer(5)]],
                          constant IdxP& p [[buffer(6)]], uint tg [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
    const uint NI = 32, DI = 128;
    threadgroup float qt[DI * NI];
    uint t0 = DI - 2 * p.half_;
    for (uint q = tid; q < NI * DI / 2; q += 256) {
        uint hh = q / (DI / 2), d0 = 2 * (q % (DI / 2));
        float a = iq[hh * DI + d0], b = iq[hh * DI + d0 + 1];
        if (d0 >= t0) {
            uint k = (d0 - t0) / 2;
            float c = cs[k], s = cs[p.half_ + k];
            float a2 = a * c - b * s, b2 = b * c + a * s;
            a = a2; b = b2;
        }
        qt[d0 * NI + hh] = a;
        qt[(d0 + 1) * NI + hh] = b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float wh = iw[lane] * p.ws;
    for (uint j = 0; j < 8; j++) {
        uint w = tg * 64 + sg * 8 + j;
        if (w >= p.nv) break;
        if (p.has_mask != 0 && mask[w / p.bs] == 0) {
            if (lane == 0) scores[w] = -INFINITY;
            continue;
        }
        device const float4* kr = (device const float4*)(keys + (ulong)w * DI);
        float d = 0.f;
        for (uint i = 0; i < DI / 4; i++) {
            float4 k4 = kr[i];
            d = fma(qt[(4 * i) * NI + lane], k4.x, d);
            d = fma(qt[(4 * i + 1) * NI + lane], k4.y, d);
            d = fma(qt[(4 * i + 2) * NI + lane], k4.z, d);
            d = fma(qt[(4 * i + 3) * NI + lane], k4.w, d);
        }
        float s = simd_sum(wh * max(d, 0.f));
        if (lane == 0) scores[w] = s;
    }
}

// The block scores of the candidate source: each block of bs positions scores as its best
// position, the newest block +inf.
struct BlkP { uint nv; uint bs; uint nb; uint _p; };
kernel void d41_block_max(device const float* s [[buffer(0)]], device float* out [[buffer(1)]], constant BlkP& p [[buffer(2)]],
                          uint b [[thread_position_in_grid]]) {
    if (b >= p.nb) return;
    float m = -INFINITY;
    for (uint i = b * p.bs; i < min(p.nv, (b + 1) * p.bs); i++) m = max(m, s[i]);
    out[b] = b == p.nb - 1 ? INFINITY : m;
}

// mask[b] = 1 for the `k` blocks listed in idx, 0 for the other of nb blocks. One threadgroup.
struct MaskP { uint nb; uint k; uint _p0; uint _p1; };
kernel void d41_mask(device const uint* idx [[buffer(0)]], device uint* mask [[buffer(1)]], constant MaskP& p [[buffer(2)]],
                     uint tid [[thread_index_in_threadgroup]]) {
    for (uint i = tid; i < p.nb; i += 1024) mask[i] = 0;
    threadgroup_barrier(mem_flags::mem_device);
    for (uint i = tid; i < p.k; i += 1024) mask[idx[i]] = 1;
}

// The k largest of n scores (ties to the lower position), as positions in ascending order, into
// out[0..k] (k < n). One threadgroup of 1024: a radix select of the k-th largest key (the
// floats in total order as unsigned keys), then the positions above it and the first of those
// equal to it, in order.
inline uint d41_key(float f) {
    uint u = as_type<uint>(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
struct TopkP { uint n; uint k; uint _p0; uint _p1; };
kernel void d41_topk(device const float* s [[buffer(0)]], device uint* out [[buffer(1)]], constant TopkP& p [[buffer(2)]],
                     uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
    threadgroup atomic_uint hist[256];
    threadgroup uint st[4]; // prefix, need (still to take within the prefix)
    threadgroup uint sgc[2][32];
    if (tid == 0) { st[0] = 0; st[1] = p.k; }
    uint mask = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (uint i = tid; i < 256; i += 1024) atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint prefix = st[0];
        for (uint i = tid; i < p.n; i += 1024) {
            uint u = d41_key(s[i]);
            if ((u & mask) == prefix) atomic_fetch_add_explicit(&hist[(u >> shift) & 255u], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            uint need = st[1], cum = 0;
            for (int b = 255; b >= 0; b--) {
                uint c = atomic_load_explicit(&hist[b], memory_order_relaxed);
                if (cum + c >= need) {
                    st[0] = prefix | (uint(b) << shift);
                    st[1] = need - cum;
                    break;
                }
                cum += c;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mask |= 255u << shift;
    }
    uint T = st[0], need_eq = st[1];
    uint taken = 0, eq_seen = 0;
    for (uint b0 = 0; b0 < p.n; b0 += 1024) {
        uint i = b0 + tid;
        uint u = i < p.n ? d41_key(s[i]) : 0u;
        uint gt = (i < p.n && u > T) ? 1u : 0u, eq = (i < p.n && u == T) ? 1u : 0u;
        uint ex_eq = simd_prefix_exclusive_sum(eq);
        if (lane == 31) sgc[0][sg] = ex_eq + eq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint base_eq = 0, tot_eq = 0;
        for (uint j = 0; j < 32; j++) { if (j < sg) base_eq += sgc[0][j]; tot_eq += sgc[0][j]; }
        uint take = (gt != 0 || (eq != 0 && eq_seen + base_eq + ex_eq < need_eq)) ? 1u : 0u;
        uint ex_t = simd_prefix_exclusive_sum(take);
        if (lane == 31) sgc[1][sg] = ex_t + take;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint base_t = 0, tot_t = 0;
        for (uint j = 0; j < 32; j++) { if (j < sg) base_t += sgc[1][j]; tot_t += sgc[1][j]; }
        if (take != 0) out[taken + base_t + ex_t] = i;
        taken += tot_t;
        eq_seen += tot_eq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
