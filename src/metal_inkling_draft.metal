// Inkling-2-Mini's drafter on Metal (metal_inkling_draft.rs): the small kernels between its
// matvecs (which run on the decode library's). Activations are float rows of the arena `a`
// (offsets in floats), the drafter's small tables are in the constants `c`.
#include <metal_stdlib>
using namespace metal;

inline float idr_sum(float v, threadgroup float* red, uint tid, uint nt) {
    v = simd_sum(v);
    uint nsg = (nt + 31) / 32;
    if (tid % 32 == 0) red[tid / 32] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0.f;
    for (uint i = 0; i < nsg; i++) s += red[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return s;
}

// ---- RMS norm of rows with a weight (grid: row): `out = x * rsqrt(mean(x^2) + eps) * w`.
struct DNormP { uint n; float eps; uint x; uint out; uint w; uint xs; uint os; uint _p; };
kernel void idr_norm(device float* a [[buffer(0)]], device const float* c [[buffer(1)]], constant DNormP& p [[buffer(2)]],
                     uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                     uint nt [[threads_per_threadgroup]]) {
    threadgroup float red[32];
    device const float* x = a + p.x + (ulong)r * p.xs;
    device float* o = a + p.out + (ulong)r * p.os;
    float ss = 0.f;
    for (uint i = tid; i < p.n; i += nt) ss += x[i] * x[i];
    ss = idr_sum(ss, red, tid, nt);
    float s = rsqrt(ss / float(p.n) + p.eps);
    for (uint i = tid; i < p.n; i += nt) o[i] = x[i] * s * c[p.w + i];
}

// ---- The fuse mix of a position's target streams (grid: row): `out = rms(sum_j w_j f_j) * nw`,
// the `nt` streams of a row `h` apart, rows `nt * h` apart.
struct DMixP { uint h; uint nt; uint f; uint out; uint nw; float eps; uint _p0; uint _p1; float w[8]; };
kernel void idr_mix(device float* a [[buffer(0)]], device const float* c [[buffer(1)]], constant DMixP& p [[buffer(2)]],
                    uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                    uint nt [[threads_per_threadgroup]]) {
    threadgroup float red[32];
    device const float* f = a + p.f + (ulong)r * p.nt * p.h;
    device float* o = a + p.out + (ulong)r * p.h;
    float ss = 0.f;
    for (uint i = tid; i < p.h; i += nt) {
        float v = 0.f;
        for (uint j = 0; j < p.nt; j++) v += p.w[j] * f[j * p.h + i];
        o[i] = v;
        ss += v * v;
    }
    ss = idr_sum(ss, red, tid, nt);
    float s = rsqrt(ss / float(p.h) + p.eps);
    for (uint i = tid; i < p.h; i += nt) o[i] = o[i] * s * c[p.nw + i];
}

// ---- Per-head RMS norm (weight `w`, `hd` values) and rotate-half RoPE at `pos + row` (grid:
// head, row; one thread per pair), in place; with `store`, the head also goes to cache row
// `pos + row` of `kc` (`heads * hd` floats per row).
struct DQkP { uint heads; uint hd; uint x; uint w; uint pos; float eps; float theta; uint store; };
kernel void idr_qk(device float* a [[buffer(0)]], device const float* c [[buffer(1)]], constant DQkP& p [[buffer(2)]],
                   device float* kc [[buffer(3)]],
                   uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float red[32];
    uint hh = tg.x, r = tg.y, half_ = p.hd / 2, nt = half_;
    device float* v = a + p.x + ((ulong)r * p.heads + hh) * p.hd;
    float x1 = tid < half_ ? v[tid] : 0.f, x2 = tid < half_ ? v[tid + half_] : 0.f;
    float ss = idr_sum(x1 * x1 + x2 * x2, red, tid, nt);
    float s = rsqrt(ss / float(p.hd) + p.eps);
    if (tid >= half_) return;
    x1 *= s * c[p.w + tid];
    x2 *= s * c[p.w + tid + half_];
    float ang = float(p.pos + r) * (1.f / pow(p.theta, float(2 * tid) / float(p.hd)));
    float cs = cos(ang), sn = sin(ang);
    float o1 = x1 * cs - x2 * sn, o2 = x2 * cs + x1 * sn;
    v[tid] = o1;
    v[tid + half_] = o2;
    if (p.store != 0) {
        device float* k = kc + ((ulong)(p.pos + r) * p.heads + hh) * p.hd;
        k[tid] = o1;
        k[tid + half_] = o2;
    }
}

// ---- Rows of `n` floats to cache rows `pos..` (grid: blocks, row).
struct DStoreP { uint n; uint x; uint pos; uint _p; };
kernel void idr_store(device const float* a [[buffer(0)]], constant DStoreP& p [[buffer(1)]], device float* kc [[buffer(2)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.n) return;
    kc[(ulong)(p.pos + gid.y) * p.n + gid.x] = a[p.x + (ulong)gid.y * p.n + gid.x];
}

// ---- The block's attention (grid: query head, row; 4 SIMD-groups of 32 lanes, a lane holding 4
// of the 128 dims): over the cached keys of positions `lo..hi` (`kc`, `vc`) and the block's own
// `nb` keys (`k`, `v` in the arena), non-causal, scores times `scale`.
struct DAttnP { uint nh; uint nkv; uint lo; uint hi; uint nb; uint q; uint k; uint v; uint out; float scale; uint _p0; uint _p1; };
kernel void idr_attn(device float* a [[buffer(0)]], constant DAttnP& p [[buffer(1)]], device const float* kc [[buffer(2)]],
                     device const float* vc [[buffer(3)]],
                     uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float sm[8];
    threadgroup float4 so[4 * 32];
    uint hh = tg.x, r = tg.y, g = hh / (p.nh / p.nkv);
    uint kw = p.nkv * 128;
    float4 q = ((device const float4*)(a + p.q + ((ulong)r * p.nh + hh) * 128))[lane] * p.scale;
    float m = -INFINITY, l = 0.f;
    float4 o = 0.f;
    uint nc = p.hi - p.lo, n = nc + p.nb;
    for (uint j = sg; j < n; j += 4) {
        device const float4* kr;
        device const float4* vr;
        if (j < nc) {
            ulong row = (ulong)(p.lo + j) * kw + g * 128;
            kr = (device const float4*)(kc + row);
            vr = (device const float4*)(vc + row);
        } else {
            ulong row = (ulong)(j - nc) * kw + g * 128;
            kr = (device const float4*)(a + p.k + row);
            vr = (device const float4*)(a + p.v + row);
        }
        float s = simd_sum(dot(q, kr[lane]));
        float mn = max(m, s);
        float al = exp(m - mn), e = exp(s - mn);
        l = l * al + e;
        o = o * al + e * vr[lane];
        m = mn;
    }
    if (lane == 0) { sm[sg] = m; sm[4 + sg] = l; }
    so[sg * 32 + lane] = o;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0) return;
    float M = max(max(sm[0], sm[1]), max(sm[2], sm[3]));
    float L = 0.f;
    float4 acc = 0.f;
    for (uint s2 = 0; s2 < 4; s2++) {
        if (sm[s2] == -INFINITY) continue;
        float f = exp(sm[s2] - M);
        L += sm[4 + s2] * f;
        acc += so[s2 * 32 + lane] * f;
    }
    ((device float4*)(a + p.out + ((ulong)r * p.nh + hh) * 128))[lane] = acc / L;
}

// ---- The head's chain, slot `i`. Joint input `[state, w1[prev], hidden, context]` (grid: blocks
// of 256): `prev` is the anchor for slot 0, else the token chosen for slot i - 1.
struct DJinP { uint r; uint h; uint state; uint hid; uint ctx; uint jin; uint tok; uint i; uint anchor; uint _p0; uint _p1; uint _p2; };
kernel void idr_jin(device float* a [[buffer(0)]], device const ushort* w1 [[buffer(1)]], constant DJinP& p [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
    uint r = p.r, h = p.h;
    uint n = 2 * r + 2 * h;
    if (gid >= n) return;
    float v;
    if (gid < r) v = a[p.state + gid];
    else if (gid < 2 * r) {
        uint prev = p.i == 0 ? p.anchor : as_type<uint>(a[p.tok + p.i - 1]);
        v = as_type<float>(uint(w1[(ulong)prev * r + gid - r]) << 16);
    } else if (gid < 2 * r + h) v = a[p.hid + gid - 2 * r];
    else v = a[p.ctx + gid - 2 * r - h];
    a[p.jin + gid] = v;
}

// The gated state update and the latent (one threadgroup of `r` threads): `j` (`[3 r]`) plus the
// bias; `state = cand + sigmoid(j0) (state - cand)`, `cand = tanh(j1)`, `latent = tanh(j2)`.
struct DGateP { uint r; uint j; uint b; uint state; uint lat; uint _p0; uint _p1; uint _p2; };
kernel void idr_gate(device float* a [[buffer(0)]], device const float* c [[buffer(1)]], constant DGateP& p [[buffer(2)]],
                     uint e [[thread_position_in_grid]]) {
    if (e >= p.r) return;
    uint r = p.r;
    float j0 = a[p.j + e] + c[p.b + e], j1 = a[p.j + r + e] + c[p.b + r + e], j2 = a[p.j + 2 * r + e] + c[p.b + 2 * r + e];
    float gate = 1.f / (1.f + exp(-j0));
    float cand = precise::tanh(j1);
    float st = a[p.state + e];
    a[p.state + e] = cand + gate * (st - cand);
    a[p.lat + e] = precise::tanh(j2);
}

// The slot's logits `cap tanh((base + bias + c) / cap)` over the `u` tokens (grid: blocks).
struct DLogitP { uint u; uint base; uint bias; uint c; uint out; float cap; uint _p0; uint _p1; };
kernel void idr_logit(device float* a [[buffer(0)]], device const float* cv [[buffer(1)]], constant DLogitP& p [[buffer(2)]],
                      uint t [[thread_position_in_grid]]) {
    if (t >= p.u) return;
    float x = a[p.base + t] + a[p.bias + t] + cv[p.c + t];
    a[p.out + t] = precise::tanh(x / p.cap) * p.cap;
}

// The first maximum of `n` logits (one threadgroup of 1024): its index (uint bits) to `tok + i`.
struct DArgP { uint n; uint x; uint tok; uint i; };
kernel void idr_argmax(device float* a [[buffer(0)]], constant DArgP& p [[buffer(1)]],
                       uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float bv[32];
    threadgroup uint bi[32];
    float m = -INFINITY;
    uint mi = 0xffffffffu;
    for (uint t = tid; t < p.n; t += 1024) {
        float v = a[p.x + t];
        if (v > m) { m = v; mi = t; }
    }
    float sm = simd_max(m);
    uint si = simd_min(m == sm ? mi : 0xffffffffu);
    if (lane == 0) { bv[sg] = sm; bi[sg] = si; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid != 0) return;
    float best = bv[0];
    uint bidx = bi[0];
    for (uint i = 1; i < 32; i++) if (bv[i] > best || (bv[i] == best && bi[i] < bidx)) { best = bv[i]; bidx = bi[i]; }
    a[p.tok + p.i] = as_type<float>(bidx);
}
