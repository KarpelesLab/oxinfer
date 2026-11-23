// Inkling-2 kernels (metal_inkling.rs): appended to the decode library, whose matvec templates
// (`mv_acc`, the weight loaders) the expert kernels use. Activations are float rows of the plan's
// arena (`a`, offsets in floats) for the tokens of a step (one row per token, `h` floats apart
// unless said otherwise); the model's small tables are in the constants (`c`).
//
// Every kernel computes a token's results the same way whatever the number of tokens of the step
// (the token is a grid dimension), so that a speculative verification pass gives exactly the
// logits one-token steps give.
#pragma clang fp reassociate(off)

// The sum of `v` over a threadgroup of `nt` threads (all get it). `red` holds 32 floats.
inline float ink_sum(float v, threadgroup float* red, uint tid, uint nt) {
    v = simd_sum(v);
    uint nsg = (nt + 31) / 32;
    if (tid % 32 == 0) red[tid / 32] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0.f;
    for (uint i = 0; i < nsg; i++) s += red[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return s;
}

// ---- The attention residual: per query (grid x) and token (grid y), the softmax mix
// (`q . state / sqrt(h)`) of the block states `s[0..ns]` (already RMS-normed, rows h apart per
// token) and of the RMS-normed x; with `norm`, the result is RMS-normed with the weights at
// `nw + query * h`. Output row of (query, token): `out + query * out_qs + token * h`.
// `cat` (nonzero): the output also normed without a weight into `cat + qi cat_qs + ti cat_ts` (the
// message gates' inputs); the query `cat_last` into `cat_n` consecutive ones.
struct IMixP { uint h; uint ns; uint norm; uint x; uint q; uint out; uint out_qs; uint nw; float scale; float eps; float mix_eps; uint _p; uint s[16];
               uint cat; uint cat_qs; uint cat_ts; uint cat_last; uint cat_n; uint _p1; uint _p2; uint _p3; };
// (One threadgroup of 1024 per (query, token); at most NSM block states. A thread's channels, up
// to 3 (h <= 3072), keep their x and states in registers from the first pass to the output.)
template <uint NSM>
inline void ink_mix_body(device float* a, device const float* c, constant IMixP& p, uint qi, uint ti, uint tid, uint sg, uint lane,
                         threadgroup float* red) {
    constexpr uint nt = 1024;
    uint h = p.h, ns = p.ns;
    device const float* x = a + p.x + (ulong)ti * h;
    device const float* q = c + p.q + (ulong)qi * h;
    // j < ns: q . state j; NSM: q . x; NSM + 1: x . x (sums over the threadgroup in `red` from 64:
    // 32 SIMD-groups x 18).
    float d[NSM + 2];
    for (uint j = 0; j < NSM + 2; j++) d[j] = 0.f;
    float xv[3], sv[3][NSM], wv[3];
    for (uint k = 0; k < 3; k++) {
        uint i = tid + k * nt;
        bool in = i < h;
        xv[k] = in ? x[i] : 0.f;
        float qv = in ? q[i] : 0.f;
        wv[k] = in && p.norm != 0 ? c[p.nw + (ulong)qi * h + i] : 0.f;
        for (uint j = 0; j < NSM; j++) {
            sv[k][j] = in && j < ns ? a[p.s[j] + (ulong)ti * h + i] : 0.f;
            d[j] = fma(qv, sv[k][j], d[j]);
        }
        d[NSM] = fma(qv, xv[k], d[NSM]);
        d[NSM + 1] = fma(xv[k], xv[k], d[NSM + 1]);
    }
    for (uint j = 0; j < NSM + 2; j++) {
        float v = simd_sum(d[j]);
        if (lane == 0) red[64 + sg * 18 + j] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < NSM + 2) {
        float t = 0.f;
        for (uint g = 0; g < nt / 32; g++) t += red[64 + g * 18 + tid];
        red[tid] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = rsqrt(red[NSM + 1] / float(h) + p.mix_eps);
    float sc[NSM + 1];
    float m = -INFINITY;
    for (uint j = 0; j <= NSM; j++) {
        sc[j] = j == NSM ? red[NSM] * r * p.scale : (j < ns ? red[j] * p.scale : -INFINITY);
        m = max(m, sc[j]);
    }
    float l = 0.f;
    for (uint j = 0; j <= NSM; j++) { sc[j] = sc[j] == -INFINITY ? 0.f : exp(sc[j] - m); l += sc[j]; }
    for (uint j = 0; j <= NSM; j++) sc[j] /= l;
    device float* out = a + p.out + (ulong)qi * p.out_qs + (ulong)ti * h;
    float oss = 0.f, ov[3];
    for (uint k = 0; k < 3; k++) {
        float v = sc[NSM] * (xv[k] * r);
        for (uint j = 0; j < NSM; j++) if (j < ns) v += sc[j] * sv[k][j];
        ov[k] = v;
        oss += v * v;
    }
    if (p.norm == 0 && p.cat == 0) {
        for (uint k = 0; k < 3; k++) if (tid + k * nt < h) out[tid + k * nt] = ov[k];
        return;
    }
    oss = simd_sum(oss);
    if (lane == 0) red[32 + sg] = oss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.f;
    for (uint g = 0; g < nt / 32; g++) tot += red[32 + g];
    float rn = rsqrt(tot / float(h) + p.eps);
    uint copies = p.cat == 0 ? 0 : (qi == p.cat_last ? p.cat_n : 1);
    for (uint k = 0; k < 3; k++) {
        uint i = tid + k * nt;
        if (i >= h) continue;
        for (uint cc = 0; cc < copies; cc++) a[p.cat + (ulong)(qi + cc) * p.cat_qs + (ulong)ti * p.cat_ts + i] = ov[k] * rn;
        out[i] = p.norm != 0 ? ov[k] * rn * wv[k] : ov[k];
    }
}
kernel void ink_mix(device float* a [[buffer(0)]], device const float* c [[buffer(1)]], constant IMixP& p [[buffer(2)]],
                    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[64 + 32 * 18];
    ink_mix_body<16>(a, c, p, tg.x, tg.y, tid, sg, lane, red);
}
kernel void ink_mix8(device float* a [[buffer(0)]], device const float* c [[buffer(1)]], constant IMixP& p [[buffer(2)]],
                     uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[64 + 32 * 18];
    ink_mix_body<8>(a, c, p, tg.x, tg.y, tid, sg, lane, red);
}

// ---- RMS norm of rows (grid x: token): `out = x * rsqrt(mean(x^2) + eps) (* w)`; rows `xs` /
// `os` floats apart. In place allowed.
struct INormP { uint h; float eps; uint x; uint out; uint w; uint has_w; uint xs; uint os; };
kernel void ink_norm(device float* a [[buffer(0)]], device const float* c [[buffer(1)]], constant INormP& p [[buffer(2)]],
                     uint ti [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                     uint nt [[threads_per_threadgroup]]) {
    threadgroup float red[32];
    device const float* x = a + p.x + (ulong)ti * p.xs;
    device float* out = a + p.out + (ulong)ti * p.os;
    // (A thread's first 12 values loaded at once; longer rows loop.)
    float xv[12];
    for (uint k = 0; k < 12; k++) xv[k] = tid + k * nt < p.h ? x[tid + k * nt] : 0.f;
    float ss = 0.f;
    for (uint k = 0; k < 12; k++) ss += xv[k] * xv[k];
    for (uint i = tid + 12 * nt; i < p.h; i += nt) ss += x[i] * x[i];
    ss = ink_sum(ss, red, tid, nt);
    float r = rsqrt(ss / float(p.h) + p.eps);
    for (uint k = 0; k < 12; k++) {
        uint i = tid + k * nt;
        if (i < p.h) out[i] = xv[k] * r * (p.has_w != 0 ? c[p.w + i] : 1.f);
    }
    for (uint i = tid + 12 * nt; i < p.h; i += nt) out[i] = x[i] * r * (p.has_w != 0 ? c[p.w + i] : 1.f);
}

// ---- A message gate's mix (grid: channel blocks, token): `out = (1 - tanh f) x + tanh(g) m`.
struct IGmP { uint h; uint x; uint m; uint f; uint g; uint out; };
kernel void ink_gmix(device float* a [[buffer(0)]], constant IGmP& p [[buffer(1)]], uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x, ti = gid.y;
    if (i >= p.h) return;
    ulong o = (ulong)ti * p.h + i;
    float x = a[p.x + o], m = a[p.m + o];
    a[p.out + o] = (1.f - precise::tanh(a[p.f + o])) * x + precise::tanh(a[p.g + o]) * m;
}

// The same and the RMS norm of the result with the weights at `w` into `nout` (one threadgroup of
// 1024 per token).
struct IGnP { uint h; uint x; uint m; uint f; uint g; uint nout; uint w; float eps; };
kernel void ink_gmix_norm(device float* a [[buffer(0)]], device const float* c [[buffer(1)]], constant IGnP& p [[buffer(2)]],
                          uint ti [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[32];
    float ss = 0.f;
    for (uint i = tid; i < p.h; i += 1024) {
        ulong o = (ulong)ti * p.h + i;
        float v = (1.f - precise::tanh(a[p.f + o])) * a[p.x + o] + precise::tanh(a[p.g + o]) * a[p.m + o];
        a[p.nout + o] = v;
        ss += v * v;
    }
    ss = simd_sum(ss);
    if (lane == 0) red[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.f;
    for (uint g = 0; g < 32; g++) tot += red[g];
    float r = rsqrt(tot / float(p.h) + p.eps);
    for (uint i = tid; i < p.h; i += 1024) {
        ulong o = (ulong)ti * p.h + i;
        a[p.nout + o] = a[p.nout + o] * r * c[p.w + i];
    }
}

// ---- A message gate's forget and gain matvecs over its rank inputs `d` (float32 or bf16 rows of
// `rank` values, a multiple of 16) and its mix, per channel: `out = (1 - tanh f) x + tanh(g) m`
// (grid: blocks of 64 channels, token; 8 SIMD-groups of 8 channels, 4 lanes per channel: rows this
// short leave most lanes of a SIMD-group per row idle). The norm follows (`ink_norm`), or with `ss`
// the output is weighted by the norm's weights at `w` and the threadgroup's sum of the squares of
// the unweighted outputs goes to `ss + token * 64 + block`: the kernels that read the output scale
// their sums by the norm (`ink_gscale`) instead.
struct IGfP { uint h; uint rank; uint d; uint x; uint m; uint out; uint bf; uint _p; uint w; uint ss; uint _q0; uint _q1; };
inline float4 gf_w(device const uchar* w, uint bf, ulong i) {
    if (bf != 0) {
        device const ushort* u = (device const ushort*)w + i;
        return float4(as_type<float>(uint(u[0]) << 16), as_type<float>(uint(u[1]) << 16), as_type<float>(uint(u[2]) << 16),
                      as_type<float>(uint(u[3]) << 16));
    }
    return float4(*((device const packed_float4*)((device const float*)w + i)));
}
kernel void ink_gfg(device float* a [[buffer(0)]], device const uchar* fw [[buffer(1)]], device const uchar* gw [[buffer(2)]],
                    constant IGfP& p [[buffer(3)]], device const float* c [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8];
    uint row = tg.x * 64 + sg * 8 + lane / 4, part = lane % 4, ti = tg.y;
    uint r = min(row, p.h - 1), per = p.rank / 4;
    device const float* d = a + p.d + (ulong)ti * p.rank + part * per;
    ulong w0 = (ulong)r * p.rank + part * per;
    float f = 0.f, g = 0.f;
    for (uint i = 0; i < per; i += 4) {
        float4 dv = float4(*((device const packed_float4*)(d + i)));
        float4 fv = gf_w(fw, p.bf, w0 + i), gv = gf_w(gw, p.bf, w0 + i);
        f = fma(fv.x, dv.x, f); f = fma(fv.y, dv.y, f); f = fma(fv.z, dv.z, f); f = fma(fv.w, dv.w, f);
        g = fma(gv.x, dv.x, g); g = fma(gv.y, dv.y, g); g = fma(gv.z, dv.z, g); g = fma(gv.w, dv.w, g);
    }
    f += simd_shuffle_xor(f, 1);
    f += simd_shuffle_xor(f, 2);
    g += simd_shuffle_xor(g, 1);
    g += simd_shuffle_xor(g, 2);
    bool mine = part == 0 && row < p.h;
    ulong o = (ulong)ti * p.h + row;
    float v = mine ? (1.f - precise::tanh(f)) * a[p.x + o] + precise::tanh(g) * a[p.m + o] : 0.f;
    if (p.ss == 0) {
        if (mine) a[p.out + o] = v;
        return;
    }
    if (mine) a[p.out + o] = v * c[p.w + row];
    float s2 = simd_sum(v * v);
    if (lane == 0) red[sg] = s2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane == 0) {
        float t = 0.f;
        for (uint i = 0; i < 8; i++) t += red[i];
        a[p.ss + (ulong)ti * 64 + tg.x] = t;
    }
}
// The RMS norm's factor of a row whose sums of squares `ink_gfg` left in `np` (<= 64) partials at
// `ss`, in a SIMD-group (all lanes get it; the same sum in every kernel).
inline float ink_gscale(device const float* a, uint ss, uint np, uint h, float eps, uint lane) {
    float v = (lane < np ? a[ss + lane] : 0.f) + (lane + 32 < np ? a[ss + lane + 32] : 0.f);
    return rsqrt(simd_sum(v) / float(h) + eps);
}

// ---- Copies (grid: channel blocks, token): `n` floats per token, rows `n` apart.
// Rows of `n` floats; destination rows `ds` floats apart.
struct ICopyP { uint n; uint src; uint dst; uint ds; };
kernel void ink_copy(device float* a [[buffer(0)]], constant ICopyP& p [[buffer(1)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.n) return;
    a[p.dst + (ulong)gid.y * p.ds + gid.x] = a[p.src + (ulong)gid.y * p.n + gid.x];
}

// ---- KV cache rows: float32, half or int8 with a float scale per 32 values (separate buffer).
// KFMT 0: f32, 1: f16, 2: q8.
template <int KFMT>
inline float kv_get(device const uchar* kb, device const float* ks, ulong idx) {
    if (KFMT == 0) return ((device const float*)kb)[idx];
    if (KFMT == 1) return float(((device const half*)kb)[idx]);
    return float(((device const char*)kb)[idx]) * ks[idx / 32];
}
// Four values from `idx` (a multiple of 4).
template <int KFMT>
inline float4 kv_get4(device const uchar* kb, device const float* ks, ulong idx) {
    if (KFMT == 0) return *(device const float4*)((device const float*)kb + idx);
    if (KFMT == 1) return float4(*(device const half4*)((device const half*)kb + idx));
    return float4(*(device const char4*)((device const char*)kb + idx)) * ks[idx / 32];
}

// ---- Keys and values of the tokens (grid: kv head, token; one thread per channel of a head):
// the short convolutions (learned taps `[lag][kw]`, plus with `dynamic` a tap per (token, kv
// head, lag) at `kd` / `vd`), the inputs of earlier positions from the rings (row of position q
// at `(q + kn - 1) % ring_cap`; the step's own tokens from their rows); then the key's L2 clip
// and the value's RMS norm, into the caches at the token's position (row `pos % cap` when
// `cap` is set: a sliding window's ring; else the position). The raw inputs go to the rings.
struct IKvP { uint kw; uint hd; uint kn; uint nkv; uint dynamic; float norm; float clip; float eps; uint k; uint v; uint kd; uint vd;
              uint tk; uint tv; uint pos; uint ring_cap; uint cap; float clip_eps; uint ring_write; uint row_base; };
template <int KFMT>
inline void kv_store(device uchar* cb, device float* cs, ulong idx, float v, uint lane) {
    if (KFMT == 0) ((device float*)cb)[idx] = v;
    else if (KFMT == 1) ((device half*)cb)[idx] = half(v);
    else {
        float amax = simd_max(fabs(v));
        float d = amax / 127.f;
        float inv = d > 0.f ? 1.f / d : 0.f;
        ((device char*)cb)[idx] = char(clamp(round(v * inv), -128.f, 127.f));
        if (lane == 0) cs[idx / 32] = d;
    }
}
template <int KFMT>
inline void ink_kv_body(device float* a, device const float* c, constant IKvP& p, device float* rk, device float* rv,
                        device uchar* kc, device uchar* vc, device float* ksc, device float* vsc, uint g, uint ti, uint tid,
                        threadgroup float* red) {
    uint ch = g * p.hd + tid, kw = p.kw, kn = p.kn;
    uint pos = p.pos + ti;
    float rk_in = a[p.k + (ulong)ti * kw + ch], rv_in = a[p.v + (ulong)ti * kw + ch];
    float sk = rk_in, sv = rv_in;
    for (uint lag = 0; lag < kn; lag++) {
        float xk, xv;
        if (lag <= ti) {
            xk = a[p.k + (ulong)(ti - lag) * kw + ch];
            xv = a[p.v + (ulong)(ti - lag) * kw + ch];
        } else {
            int q = int(pos) - int(lag);
            ulong r = (ulong)((q + int(kn) - 1 + int(p.ring_cap)) % int(p.ring_cap)) * kw + ch;
            xk = rk[r];
            xv = rv[r];
        }
        float wk = c[p.tk + lag * kw + ch], wv = c[p.tv + lag * kw + ch];
        if (p.dynamic != 0) {
            wk += a[p.kd + ((ulong)ti * p.nkv + g) * kn + lag];
            wv += a[p.vd + ((ulong)ti * p.nkv + g) * kn + lag];
        }
        sk += wk * xk;
        sv += wv * xv;
    }
    sk *= p.norm;
    sv *= p.norm;
    float nk = ink_sum(sk * sk, red, tid, p.hd);
    float f = p.clip / (sqrt(nk) + p.clip_eps);
    if (f < 1.f) sk *= f;
    float nv = ink_sum(sv * sv, red, tid, p.hd);
    sv *= rsqrt(nv / float(p.hd) + p.eps);
    ulong row = p.cap != 0 ? pos % p.cap : pos - p.row_base;
    kv_store<KFMT>(kc, ksc, row * kw + ch, sk, tid % 32);
    kv_store<KFMT>(vc, vsc, row * kw + ch, sv, tid % 32);
    if (p.ring_write != 0) {
        ulong wr = (ulong)((pos + kn - 1) % p.ring_cap) * kw + ch;
        rk[wr] = rk_in;
        rv[wr] = rv_in;
    }
}
#define INK_KV(F) \
kernel void ink_kv_##F(device float* a [[buffer(0)]], device const float* c [[buffer(1)]], constant IKvP& p [[buffer(2)]], \
                      device float* rk [[buffer(3)]], device float* rv [[buffer(4)]], device uchar* kc [[buffer(5)]], \
                      device uchar* vc [[buffer(6)]], device float* ksc [[buffer(7)]], device float* vsc [[buffer(8)]], \
                      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup float red[32]; ink_kv_body<F>(a, c, p, rk, rv, kc, vc, ksc, vsc, tg.x, tg.y, tid, red); }
INK_KV(0) INK_KV(1) INK_KV(2)

// ---- Queries and relative vectors (grid: query head, token; one thread per channel): scaled
// per channel, clipped to an L2 norm, times `1 + alpha ln(max(1, (pos + 1) / floor))` when
// `alpha` is set (full layers). In place.
struct IQP { uint w; uint hd; uint dr; uint nh; uint q; uint r; uint qs; uint rs; float qclip; float rclip; float alpha; float floor_;
             uint pos; float clip_eps; uint _p0; uint _p1; };
kernel void ink_q(device float* a [[buffer(0)]], device const float* c [[buffer(1)]], constant IQP& p [[buffer(2)]],
                  uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float red[32];
    uint hh = tg.x, ti = tg.y;
    float tau = 1.f;
    if (p.alpha != 0.f) tau = 1.f + p.alpha * log(max(1.f, float(p.pos + ti + 1) / p.floor_));
    device float* q = a + p.q + (ulong)ti * p.w + hh * p.hd;
    float v = q[tid] * c[p.qs + hh * p.hd + tid];
    float n = ink_sum(v * v, red, tid, p.hd);
    float f = p.qclip / (sqrt(n) + p.clip_eps);
    if (f < 1.f) v *= f;
    q[tid] = v * tau;
    device float* r = a + p.r + ((ulong)ti * p.nh + hh) * p.dr;
    float rv = tid < p.dr ? r[tid] * c[p.rs + hh * p.dr + tid] : 0.f;
    float rn = ink_sum(rv * rv, red, tid, p.hd);
    float rf = p.rclip / (sqrt(rn) + p.clip_eps);
    if (rf < 1.f) rv *= rf;
    if (tid < p.dr) r[tid] = rv * tau;
}

// ---- Attention of the tokens' queries (grid: kv head, key split, token): the G query heads of a
// key/value head over a slice of `chunk` keys of the token's range, with the relative bias of
// each backward distance below `extent` (the token's relative vector against the bank `rel`,
// `[dr][extent]` in the constants); partial softmax results to `part` (per token, head and
// split: D * 32 accumulators, then the max and the sum). The token at position `pos + t` sees
// positions `max(first, pos + t + 1 - window)..pos + t` (`window` 0: all); key q is cache row
// `q % cap` (`cap` 0: q).
// (`mcnt`: with `fin`, the arena offset of a counter per (token, key head) by which the last of a
// token's splits to finish merges them, `ink_attn_merge`'s work.)
struct IAP { uint nh; uint nkv; uint hd; uint dr; uint pos; uint window; uint cap; uint extent; uint cmin; uint maxsplit;
             float scale; uint q; uint r; uint part; uint rel; uint kw; uint tbase; uint tmp; uint fin; uint sink; uint gate; uint out;
             uint mcnt; uint _m1; };
// Output `i` of head `hh` of token `ti` from its `ns` splits' partials (`part`) and its sink, gated:
// `ink_attn_merge`'s arithmetic.
inline void ink_merge_elem(device float* a, uint part, uint nh, uint maxsplit, uint sinko, uint gate, uint out, uint hh, uint ti, uint ns,
                           uint hd, uint i) {
    device const float* ph = a + part + ((ulong)ti * nh + hh) * maxsplit * (hd + 2);
    float sink = a[sinko + (ulong)ti * nh + hh];
    float M = sink;
    for (uint s = 0; s < ns; s++) M = max(M, ph[s * (hd + 2) + hd]);
    float L = exp(sink - M);
    for (uint s = 0; s < ns; s++) {
        float ms = ph[s * (hd + 2) + hd];
        if (ms != -INFINITY) L += ph[s * (hd + 2) + hd + 1] * exp(ms - M);
    }
    float acc = 0.f;
    for (uint s = 0; s < ns; s++) {
        float ms = ph[s * (hd + 2) + hd];
        if (ms != -INFINITY) acc += ph[s * (hd + 2) + i] * exp(ms - M);
    }
    ulong o = (ulong)ti * nh * hd + hh * hd + i;
    float gt = a[gate + o];
    a[out + o] = acc / L * (1.0f / (1.0f + exp(-gt)));
}
inline void ink_range(constant IAP& p, uint ti, thread uint& lo, thread uint& hi) {
    uint pp = p.pos + ti;
    hi = pp + 1;
    lo = p.window != 0 && hi > p.window ? hi - p.window : 0;
}
// Keys per split of a token's range of `len` keys, and the splits (at most `ms`): the same rule for
// every token, its own length. Splits of about a fifth of the range, of 32 to 256 keys and at least
// `cmin` (64 in passes of up to 16 tokens, whose few threadgroups would otherwise leave most cores
// idle on ranges of a few hundred keys; 256 in prefill passes).
inline uint ink_chunk(uint len, uint ms, uint cmin) {
    uint fifth = min(256u, ((len + 4) / 5 + 31) / 32 * 32);
    return max(max(cmin, fifth), ((len + ms - 1) / ms + 31) / 32 * 32);
}
inline uint ink_nsplit(uint len, uint ms, uint cmin) {
    uint ch = ink_chunk(len, ms, cmin);
    return max(1u, (len + ch - 1) / ch);
}
// Where key j is: the rows of the pass's own tokens (`tmp`: positions from `tbase`, a prefill pass
// longer than a window's ring keeps them apart), else the cache (ring row `j % cap`).
#define INK_ROW(j) ((p.tmp != 0 && (j) >= p.tbase) ? (ulong)((j) - p.tbase) : (p.cap != 0 ? (ulong)((j) % p.cap) : (ulong)(j)))
#define INK_KB(j) (p.tmp != 0 && (j) >= p.tbase ? tk : kc)
#define INK_VB(j) (p.tmp != 0 && (j) >= p.tbase ? tv : vc)
#define INK_KS(j) (p.tmp != 0 && (j) >= p.tbase ? tks : ksc)
#define INK_VS(j) (p.tmp != 0 && (j) >= p.tbase ? tvs : vsc)
template <int D, int G, int KFMT>
inline void ink_attn_body(device float* a, device const float* c, constant IAP& p, device const uchar* kc, device const uchar* vc,
                          device const float* ksc, device const float* vsc, device const uchar* tk, device const uchar* tv,
                          device const float* tks, device const float* tvs, uint g, uint split, uint ti, uint tid, uint sg, uint lane,
                          threadgroup float* sm, threadgroup float* so, threadgroup float* qs, threadgroup float* bias) {
    constexpr uint HD = D * 32;
    uint lo, hi;
    ink_range(p, ti, lo, hi);
    uint ns = ink_nsplit(hi - lo, p.maxsplit, p.cmin), chunk = ink_chunk(hi - lo, p.maxsplit, p.cmin);
    if (split >= ns) return;
    uint j0 = lo + split * chunk, j1 = min(hi, j0 + chunk);
    uint h0 = g * G, pp = p.pos + ti;
    for (uint i = tid; i < G * HD; i += 256) qs[i] = a[p.q + (ulong)ti * p.nh * HD + h0 * HD + i] * p.scale;
    // The bias of the distances this slice covers: pp - j for j in j0..j1, below the extent.
    uint dmin = pp + 1 - j1, dmax = pp - j0;
    uint nb = dmin < p.extent ? min(dmax + 1, p.extent) - dmin : 0;
    // (Item i: head i / nb, distance dmin + i % nb; the threads on the items there are rather than
    // on G * 1024 slots, most of them past the distances of a short range.)
    for (uint i = tid; i < G * nb; i += 256) {
        uint gg = i / nb, k = i % nb;
        float b = 0.f;
        device const float* rq = a + p.r + ((ulong)ti * p.nh + h0 + gg) * p.dr;
        for (uint d = 0; d < p.dr; d++) b += rq[d] * c[p.rel + d * p.extent + dmin + k];
        bias[gg * 1024 + k] = b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float o[G][D], m[G], l[G];
    for (int gg = 0; gg < G; gg++) {
        m[gg] = -INFINITY;
        l[gg] = 0.f;
        for (int d = 0; d < D; d++) o[gg][d] = 0.f;
    }
    ulong koff = (ulong)g * HD;
    // (Keys interleaved over the SIMD-groups, SIMD-group sg taking keys j0 + sg + 8 k, so that a
    // short range keeps all of them busy.)
    for (uint base = j0 + sg; base < j1; base += 8 * 32) {
        uint j = base + 8 * lane;
        float s[G];
        for (int gg = 0; gg < G; gg++) s[gg] = -INFINITY;
        if (j < j1) {
            ulong row = INK_ROW(j) * p.kw + koff;
            device const uchar* kcb = INK_KB(j);
            device const float* ksb = INK_KS(j);
            float acc[G];
            for (int gg = 0; gg < G; gg++) acc[gg] = 0.f;
            // (The key's loads 8 at a time: in flight together rather than one per step.)
            for (uint d0 = 0; d0 < HD; d0 += 32) {
                float4 k8[8];
                for (uint u = 0; u < 8; u++) k8[u] = kv_get4<KFMT>(kcb, ksb, row + d0 + 4 * u);
                for (uint u = 0; u < 8; u++) {
                    float4 k4 = k8[u];
                    uint d = d0 + 4 * u;
                    for (int gg = 0; gg < G; gg++) {
                        float4 q4 = ((threadgroup const float4*)(qs + gg * HD))[d / 4];
                        acc[gg] = fma(k4.x, q4.x, acc[gg]);
                        acc[gg] = fma(k4.y, q4.y, acc[gg]);
                        acc[gg] = fma(k4.z, q4.z, acc[gg]);
                        acc[gg] = fma(k4.w, q4.w, acc[gg]);
                    }
                }
            }
            uint dist = pp - j;
            for (int gg = 0; gg < G; gg++) s[gg] = acc[gg] + (dist >= dmin && dist - dmin < nb ? bias[gg * 1024 + dist - dmin] : 0.f);
        }
        float e[G];
        for (int gg = 0; gg < G; gg++) {
            float mn = max(m[gg], simd_max(s[gg]));
            float al = mn == -INFINITY ? 1.f : exp(m[gg] - mn);
            e[gg] = j < j1 ? exp(s[gg] - mn) : 0.f;
            l[gg] = l[gg] * al + simd_sum(e[gg]);
            for (int d = 0; d < D; d++) o[gg][d] *= al;
            m[gg] = mn;
        }
        uint nbk = min(32u, (j1 - base + 7) / 8);
        // (The values of 8 keys loaded together, then accumulated in key order.)
        for (uint t0 = 0; t0 < nbk; t0 += 8) {
            float v8[8][D];
            for (uint u = 0; u < 8; u++) {
                uint jj = min(base + 8 * (t0 + u), j1 - 1);
                ulong row = INK_ROW(jj) * p.kw + koff + lane * D;
                device const uchar* vcb = INK_VB(jj);
                device const float* vsb = INK_VS(jj);
                for (int d = 0; d < D; d += 4) {
                    float4 v4 = kv_get4<KFMT>(vcb, vsb, row + d);
                    v8[u][d] = v4.x; v8[u][d + 1] = v4.y; v8[u][d + 2] = v4.z; v8[u][d + 3] = v4.w;
                }
            }
            for (uint u = 0; u < 8 && t0 + u < nbk; u++) {
                for (int gg = 0; gg < G; gg++) {
                    float et = simd_shuffle(e[gg], t0 + u);
                    for (int d = 0; d < D; d++) o[gg][d] = fma(et, v8[u][d], o[gg][d]);
                }
            }
        }
    }
    for (int gg = 0; gg < G; gg++) {
        uint hh = h0 + gg;
        if (lane == 0) { sm[sg] = m[gg]; sm[8 + sg] = l[gg]; }
        for (int d = 0; d < D; d++) so[sg * HD + lane * D + d] = o[gg][d];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float M = sm[0];
        for (uint s2 = 1; s2 < 8; s2++) M = max(M, sm[s2]);
        float L = 0.f;
        for (uint s2 = 0; s2 < 8; s2++) L += sm[s2] == -INFINITY ? 0.f : sm[8 + s2] * exp(sm[s2] - M);
        device float* pr = a + p.part + (((ulong)ti * p.nh + hh) * p.maxsplit + split) * (HD + 2);
        for (uint i = tid; i < HD; i += 256) {
            float acc = 0.f;
            for (uint s2 = 0; s2 < 8; s2++) if (sm[s2] != -INFINITY) acc += so[s2 * HD + i] * exp(sm[s2] - M);
            pr[i] = acc;
        }
        if (tid == 0) { pr[HD] = M; pr[HD + 1] = L; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
#define INK_ATTN(D, G, F) \
kernel void ink_attn_##D##_##G##_##F(device float* a [[buffer(0)]], device const float* c [[buffer(1)]], constant IAP& p [[buffer(2)]], \
                         device const uchar* kc [[buffer(3)]], device const uchar* vc [[buffer(4)]], device const float* ksc [[buffer(5)]], \
                         device const float* vsc [[buffer(6)]], device const uchar* tk [[buffer(7)]], device const uchar* tv [[buffer(8)]], \
                         device const float* tks [[buffer(9)]], device const float* tvs [[buffer(10)]], \
                         uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float sm[16]; threadgroup float so[8 * D * 32]; threadgroup float qs[G * D * 32]; threadgroup float bias[G * 1024]; \
    ink_attn_body<D, G, F>(a, c, p, kc, vc, ksc, vsc, tk, tv, tks, tvs, tg.x, tg.y, tg.z, tid, sg, lane, sm, so, qs, bias); }
INK_ATTN(4, 3, 0) INK_ATTN(4, 3, 1) INK_ATTN(4, 3, 2)
INK_ATTN(4, 1, 0) INK_ATTN(4, 1, 1) INK_ATTN(4, 1, 2)
INK_ATTN(4, 2, 0) INK_ATTN(4, 2, 1) INK_ATTN(4, 2, 2)
INK_ATTN(4, 4, 0) INK_ATTN(4, 4, 1) INK_ATTN(4, 4, 2)

// ---- The same for passes of up to 16 tokens (same grid, parameters and partial results): the
// lanes of a SIMD-group across a head's channels (D each) and the SIMD-groups across the keys
// (SIMD-group sg: keys j0 + sg + 8 m), KB keys at a time with their key and value loads issued
// together, each dot product a SIMD-group sum; at the short ranges of decoding each key is one
// load per lane rather than a lane's walk over its row.
#define INK_AKB 2
// The SIMD-groups' partial results (maxima and sums in `sm`, accumulators in `so`) combined:
// the merge's work for a token of one split, else the split's partial result.
template <int D, int G>
inline void ink_attv_fin(device float* a, constant IAP& p, uint ns, uint split, uint ti, uint h0, uint tid, threadgroup float* sm,
                         threadgroup float* so) {
    constexpr uint HD = D * 32;
    for (uint i = tid; i < G * HD; i += 256) {
        uint gg = i / HD, d = i % HD;
        threadgroup const float* smg = sm + gg * 16;
        float M = smg[0];
        for (uint s2 = 1; s2 < 8; s2++) M = max(M, smg[s2]);
        float L = 0.f, acc = 0.f;
        for (uint s2 = 0; s2 < 8; s2++) {
            if (smg[s2] == -INFINITY) continue;
            float f = exp(smg[s2] - M);
            L += smg[8 + s2] * f;
            acc += so[(gg * 8 + s2) * HD + d] * f;
        }
        if (p.fin != 0 && ns == 1) {
            // (A token of one split: the merge's work here, `ink_attn_merge` leaves it out. Per
            // token, so that it is done the same way whatever the pass's other tokens.)
            uint hh = h0 + gg;
            float sink = a[p.sink + (ulong)ti * p.nh + hh];
            float M2 = max(sink, M);
            float L2 = exp(sink - M2);
            float acc2 = 0.f;
            if (M != -INFINITY) {
                L2 += L * exp(M - M2);
                acc2 += acc * exp(M - M2);
            }
            ulong o = (ulong)ti * p.nh * HD + hh * HD + d;
            float gt = a[p.gate + o];
            a[p.out + o] = acc2 / L2 * (1.0f / (1.0f + exp(-gt)));
            continue;
        }
        device float* pr = a + p.part + (((ulong)ti * p.nh + h0 + gg) * p.maxsplit + split) * (HD + 2);
        pr[d] = acc;
        if (d == 0) { pr[HD] = M; pr[HD + 1] = L; }
    }
    // The last of the token's splits to finish merges them (the others' partials made visible by
    // the fences around the counter, which it resets).
    if (p.fin == 0 || ns == 1 || p.mcnt == 0) return;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device atomic_uint* cnt = (device atomic_uint*)(a + p.mcnt) + (ulong)ti * p.nkv + h0 / G;
    if (tid == 0) sm[0] = atomic_fetch_add_explicit(cnt, 1u, memory_order_relaxed) == ns - 1 ? 1.f : 0.f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sm[0] == 0.f) return;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    // `ink_merge_elem`'s arithmetic with its loads spread: the splits' maxima and sums (one
    // (head, split) a thread), then per head the maximum, the weights exp(m - M) and the sum, then
    // the outputs, every split's partial loaded at once.
    threadgroup float* wm = so;                 // [G][64] maxima, then the weights
    threadgroup float* wl = so + G * 64;        // [G][64] sums
    threadgroup float* wh = so + 2 * G * 64;    // [G] the sums L
    device const float* ph0 = a + p.part + ((ulong)ti * p.nh + h0) * p.maxsplit * (HD + 2);
    for (uint i = tid; i < G * ns; i += 256) {
        uint gg = i / ns, s2 = i % ns;
        device const float* q = ph0 + ((ulong)gg * p.maxsplit + s2) * (HD + 2);
        wm[gg * 64 + s2] = q[HD];
        wl[gg * 64 + s2] = q[HD + 1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < uint(G)) {
        float sink = a[p.sink + (ulong)ti * p.nh + h0 + tid];
        float M = sink;
        for (uint s2 = 0; s2 < ns; s2++) M = max(M, wm[tid * 64 + s2]);
        float L = exp(sink - M);
        for (uint s2 = 0; s2 < ns; s2++) {
            float ms = wm[tid * 64 + s2];
            float e = ms != -INFINITY ? exp(ms - M) : 0.f;
            if (ms != -INFINITY) L += wl[tid * 64 + s2] * e;
            wm[tid * 64 + s2] = ms != -INFINITY ? e : -INFINITY;
        }
        wh[tid] = L;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < G * HD; i += 256) {
        uint gg = i / HD, d = i % HD;
        device const float* q = ph0 + (ulong)gg * p.maxsplit * (HD + 2) + d;
        ulong o = (ulong)ti * p.nh * HD + (h0 + gg) * HD + d;
        float gt = a[p.gate + o];
        float acc = 0.f;
        for (uint s0 = 0; s0 < ns; s0 += 8) {
            float v[8];
            for (uint u = 0; u < 8; u++) v[u] = s0 + u < ns ? q[(ulong)(s0 + u) * (HD + 2)] : 0.f;
            for (uint u = 0; u < 8; u++) {
                float e = s0 + u < ns ? wm[gg * 64 + s0 + u] : -INFINITY;
                if (e != -INFINITY) acc += v[u] * e;
            }
        }
        a[p.out + o] = acc / wh[gg] * (1.0f / (1.0f + exp(-gt)));
    }
    if (tid == 0) atomic_store_explicit(cnt, 0u, memory_order_relaxed);
}
template <int D, int G, int KFMT>
inline void ink_attv_body(device float* a, device const float* c, constant IAP& p, device const uchar* kc, device const uchar* vc,
                          device const float* ksc, device const float* vsc, device const uchar* tk, device const uchar* tv,
                          device const float* tks, device const float* tvs, uint g, uint split, uint ti, uint tid, uint sg, uint lane,
                          threadgroup float* sm, threadgroup float* so, threadgroup float* bias) {
    constexpr uint HD = D * 32;
    uint lo, hi;
    ink_range(p, ti, lo, hi);
    uint ns = ink_nsplit(hi - lo, p.maxsplit, p.cmin), chunk = ink_chunk(hi - lo, p.maxsplit, p.cmin);
    if (split >= ns) return;
    uint j0 = lo + split * chunk, j1 = min(hi, j0 + chunk);
    uint h0 = g * G, pp = p.pos + ti;
    float4 q4[G];
    for (int gg = 0; gg < G; gg++)
        q4[gg] = *(device const float4*)(a + p.q + (ulong)ti * p.nh * HD + (h0 + gg) * HD + lane * D) * p.scale;
    uint dmin = pp + 1 - j1, dmax = pp - j0;
    uint nb = dmin < p.extent ? min(dmax + 1, p.extent) - dmin : 0;
    for (uint i = tid; i < G * nb; i += 256) {
        uint gg = i / nb, k = i % nb;
        device const float* rq = a + p.r + ((ulong)ti * p.nh + h0 + gg) * p.dr;
        float rv[16], cv[16];
        for (uint d = 0; d < 16; d++) {
            rv[d] = d < p.dr ? rq[d] : 0.f;
            cv[d] = d < p.dr ? c[p.rel + d * p.extent + dmin + k] : 0.f;
        }
        float b = 0.f;
        for (uint d = 0; d < 16; d++) if (d < p.dr) b += rv[d] * cv[d];
        bias[gg * 1024 + k] = b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float4 o[G];
    float m[G], l[G];
    for (int gg = 0; gg < G; gg++) {
        m[gg] = -INFINITY;
        l[gg] = 0.f;
        o[gg] = 0.f;
    }
    ulong koff = (ulong)g * HD + lane * D;
    for (uint base = j0 + sg; base < j1; base += 8 * INK_AKB) {
        float4 k4[INK_AKB], v4[INK_AKB];
        for (uint u = 0; u < INK_AKB; u++) {
            uint j = min(base + 8 * u, j1 - 1);
            ulong row = INK_ROW(j) * p.kw + koff;
            k4[u] = kv_get4<KFMT>(INK_KB(j), INK_KS(j), row);
            v4[u] = kv_get4<KFMT>(INK_VB(j), INK_VS(j), row);
        }
        for (int gg = 0; gg < G; gg++) {
            float s[INK_AKB];
            float mx = m[gg];
            for (uint u = 0; u < INK_AKB; u++) {
                uint j = base + 8 * u;
                float4 pr = k4[u] * q4[gg];
                float dot = simd_sum((pr.x + pr.y) + (pr.z + pr.w));
                uint dist = pp - j;
                s[u] = j < j1 ? dot + (dist - dmin < nb ? bias[gg * 1024 + dist - dmin] : 0.f) : -INFINITY;
                mx = max(mx, s[u]);
            }
            float al = mx == -INFINITY ? 1.f : exp(m[gg] - mx);
            float4 acc = o[gg] * al;
            float ls = l[gg] * al;
            for (uint u = 0; u < INK_AKB; u++) {
                float e = s[u] == -INFINITY ? 0.f : exp(s[u] - mx);
                ls += e;
                acc = fma(e, v4[u], acc);
            }
            o[gg] = acc;
            l[gg] = ls;
            m[gg] = mx;
        }
    }
    // The SIMD-groups' results combined: sm[gg * 16 + sg] the maxima, + 8 the sums.
    for (int gg = 0; gg < G; gg++) {
        if (lane == 0) { sm[gg * 16 + sg] = m[gg]; sm[gg * 16 + 8 + sg] = l[gg]; }
        *(threadgroup float4*)(so + (gg * 8 + sg) * HD + lane * D) = o[gg];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ink_attv_fin<D, G>(a, p, ns, split, ti, h0, tid, sm, so);
}
#define INK_ATTV(D, G, F) \
kernel void ink_attv_##D##_##G##_##F(device float* a [[buffer(0)]], device const float* c [[buffer(1)]], constant IAP& p [[buffer(2)]], \
                         device const uchar* kc [[buffer(3)]], device const uchar* vc [[buffer(4)]], device const float* ksc [[buffer(5)]], \
                         device const float* vsc [[buffer(6)]], device const uchar* tk [[buffer(7)]], device const uchar* tv [[buffer(8)]], \
                         device const float* tks [[buffer(9)]], device const float* tvs [[buffer(10)]], \
                         uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float sm[G * 16]; threadgroup float so[G * 8 * D * 32]; threadgroup float bias[G * 1024]; \
    ink_attv_body<D, G, F>(a, c, p, kc, vc, ksc, vsc, tk, tv, tks, tvs, tg.x, tg.y, tg.z, tid, sg, lane, sm, so, bias); }
INK_ATTV(4, 3, 0) INK_ATTV(4, 3, 1) INK_ATTV(4, 3, 2)
INK_ATTV(4, 1, 0) INK_ATTV(4, 1, 1) INK_ATTV(4, 1, 2)
INK_ATTV(4, 2, 0) INK_ATTV(4, 2, 1) INK_ATTV(4, 2, 2)
INK_ATTV(4, 4, 0) INK_ATTV(4, 4, 1) INK_ATTV(4, 4, 2)

// ---- Passes of up to 8 tokens: `ink_q` and `ink_kv` in the attention kernel (`ink_attf`; no stage
// between the projections and the attention). A threadgroup (kv head g, split, token ti) makes its
// query heads' scaled and clipped queries and relative vectors as `ink_q` does, and the keys and
// values of the pass's tokens in its split's range as `ink_kv` does (the same sums in the same
// order: a channel a thread, SIMD-group sums then the head's 4 in order), rounded as the cache
// stores them; the threadgroup whose split ends at its token's own position writes that token's
// key and value to the cache and its inputs to the rings. (Keys of earlier positions come from the
// cache, written by earlier passes.)
template <int D, int G, int KFMT>
inline void ink_attf_body(device float* a, device const float* c, constant IAP& p, constant IKvP& kp, constant IQP& qp,
                          device uchar* kc, device uchar* vc, device float* ksc, device float* vsc, device float* rk, device float* rv,
                          uint g, uint split, uint ti, uint tid, uint sg, uint lane, threadgroup float* sm, threadgroup float* so,
                          threadgroup float* bias, threadgroup float* red, threadgroup float* rr) {
    constexpr uint HD = D * 32;
    uint lo, hi;
    ink_range(p, ti, lo, hi);
    uint ns = ink_nsplit(hi - lo, p.maxsplit, p.cmin), chunk = ink_chunk(hi - lo, p.maxsplit, p.cmin);
    if (split >= ns) return;
    uint j0 = lo + split * chunk, j1 = min(hi, j0 + chunk);
    uint h0 = g * G, pp = p.pos + ti;
    // The pass's tokens whose keys are in [j0, j1): kt0 .. kt0 + nloc - 1 (the last is ti at most).
    uint kt0 = max(j0, p.pos) - p.pos;
    uint nloc = j1 > p.pos ? j1 - max(j0, p.pos) : 0;
    bool writer = split == ns - 1;
    threadgroup float* qv = bias;
    threadgroup float* kt = so;
    threadgroup float* vt = so + 8 * HD;
    uint kw = kp.kw, kn = kp.kn;
    float tau = 1.f;
    if (qp.alpha != 0.f) tau = 1.f + qp.alpha * log(max(1.f, float(qp.pos + ti + 1) / qp.floor_));
    // Queries (`ink_q`): red[gg * 4 + s] the head's SIMD-group sums.
    for (uint w = tid; w < G * HD; w += 256) {
        uint gg = w / HD, ch = w % HD;
        float v = a[qp.q + (ulong)ti * qp.w + (h0 + gg) * HD + ch] * c[qp.qs + (h0 + gg) * HD + ch];
        qv[w] = v;
        float s2 = simd_sum(v * v);
        if (lane == 0) red[w / 32] = s2;
    }
    // Relative vectors (`ink_q`): SIMD-group gg, lane d.
    if (sg < uint(G)) {
        uint hh = h0 + sg;
        float r = lane < qp.dr ? a[qp.r + ((ulong)ti * qp.nh + hh) * qp.dr + lane] * c[qp.rs + hh * qp.dr + lane] : 0.f;
        float rn = simd_sum(r * r);
        float rf = qp.rclip / (sqrt(rn) + qp.clip_eps);
        if (rf < 1.f) r *= rf;
        if (lane < 16) rr[sg * 16 + lane] = r * tau;
    }
    // Keys and values (`ink_kv`): red[16 + kk * 4 + s] and red[48 + kk * 4 + s].
    for (uint w = tid; w < nloc * HD; w += 256) {
        uint kk = w / HD, ch = w % HD, tj = kt0 + kk, chg = g * HD + ch;
        uint pos = kp.pos + tj;
        float sk = a[kp.k + (ulong)tj * kw + chg], sv = a[kp.v + (ulong)tj * kw + chg];
        for (uint lag = 0; lag < kn; lag++) {
            float xk, xv;
            if (lag <= tj) {
                xk = a[kp.k + (ulong)(tj - lag) * kw + chg];
                xv = a[kp.v + (ulong)(tj - lag) * kw + chg];
            } else {
                int q = int(pos) - int(lag);
                ulong r = (ulong)((q + int(kn) - 1 + int(kp.ring_cap)) % int(kp.ring_cap)) * kw + chg;
                xk = rk[r];
                xv = rv[r];
            }
            float wk = c[kp.tk + lag * kw + chg], wv = c[kp.tv + lag * kw + chg];
            if (kp.dynamic != 0) {
                wk += a[kp.kd + ((ulong)tj * kp.nkv + g) * kn + lag];
                wv += a[kp.vd + ((ulong)tj * kp.nkv + g) * kn + lag];
            }
            sk += wk * xk;
            sv += wv * xv;
        }
        sk *= kp.norm;
        sv *= kp.norm;
        kt[w] = sk;
        vt[w] = sv;
        float s2k = simd_sum(sk * sk), s2v = simd_sum(sv * sv);
        if (lane == 0) {
            red[16 + w / 32] = s2k;
            red[48 + w / 32] = s2v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float4 q4[G];
    for (int gg = 0; gg < G; gg++) {
        float n = 0.f;
        for (uint s = 0; s < 4; s++) n += red[gg * 4 + s];
        float f = qp.qclip / (sqrt(n) + qp.clip_eps);
        float4 v = *(threadgroup float4*)(qv + gg * HD + lane * D);
        if (f < 1.f) v *= f;
        q4[gg] = (v * tau) * p.scale;
    }
    for (uint w = tid; w < nloc * HD; w += 256) {
        uint kk = w / HD, ch = w % HD, tj = kt0 + kk, chg = g * HD + ch;
        float nk = 0.f, nv = 0.f;
        for (uint s = 0; s < 4; s++) nk += red[16 + kk * 4 + s];
        for (uint s = 0; s < 4; s++) nv += red[48 + kk * 4 + s];
        float sk = kt[w], sv = vt[w];
        float f = kp.clip / (sqrt(nk) + kp.clip_eps);
        if (f < 1.f) sk *= f;
        sv *= rsqrt(nv / float(HD) + kp.eps);
        // The values as the cache holds them.
        float dk = sk, dv = sv;
        if (KFMT == 1) {
            dk = float(half(sk));
            dv = float(half(sv));
        } else if (KFMT == 2) {
            float ak = simd_max(fabs(sk)) / 127.f, av = simd_max(fabs(sv)) / 127.f;
            dk = float(char(clamp(round(sk * (ak > 0.f ? 1.f / ak : 0.f)), -128.f, 127.f))) * ak;
            dv = float(char(clamp(round(sv * (av > 0.f ? 1.f / av : 0.f)), -128.f, 127.f))) * av;
        }
        kt[w] = dk;
        vt[w] = dv;
        if (writer && tj == ti) {
            uint pos = kp.pos + tj;
            ulong row = kp.cap != 0 ? pos % kp.cap : pos - kp.row_base;
            kv_store<KFMT>(kc, ksc, row * kw + chg, sk, lane);
            kv_store<KFMT>(vc, vsc, row * kw + chg, sv, lane);
            if (kp.ring_write != 0) {
                ulong wr = (ulong)((pos + kn - 1) % kp.ring_cap) * kw + chg;
                rk[wr] = a[kp.k + (ulong)tj * kw + chg];
                rv[wr] = a[kp.v + (ulong)tj * kw + chg];
            }
        }
    }
    uint dmin = pp + 1 - j1, dmax = pp - j0;
    uint nb = dmin < p.extent ? min(dmax + 1, p.extent) - dmin : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < G * nb; i += 256) {
        uint gg = i / nb, k = i % nb;
        float rv2[16], cv[16];
        for (uint d = 0; d < 16; d++) {
            rv2[d] = d < p.dr ? rr[gg * 16 + d] : 0.f;
            cv[d] = d < p.dr ? c[p.rel + d * p.extent + dmin + k] : 0.f;
        }
        float b = 0.f;
        for (uint d = 0; d < 16; d++) if (d < p.dr) b += rv2[d] * cv[d];
        bias[gg * 1024 + k] = b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float4 o[G];
    float m[G], l[G];
    for (int gg = 0; gg < G; gg++) {
        m[gg] = -INFINITY;
        l[gg] = 0.f;
        o[gg] = 0.f;
    }
    ulong koff = (ulong)g * HD + lane * D;
    for (uint base = j0 + sg; base < j1; base += 8 * INK_AKB) {
        float4 k4[INK_AKB], v4[INK_AKB];
        for (uint u = 0; u < INK_AKB; u++) {
            uint j = min(base + 8 * u, j1 - 1);
            if (j >= p.pos) {
                k4[u] = *(threadgroup const float4*)(kt + (j - p.pos - kt0) * HD + lane * D);
                v4[u] = *(threadgroup const float4*)(vt + (j - p.pos - kt0) * HD + lane * D);
            } else {
                ulong row = (p.cap != 0 ? (ulong)(j % p.cap) : (ulong)j) * p.kw + koff;
                k4[u] = kv_get4<KFMT>(kc, ksc, row);
                v4[u] = kv_get4<KFMT>(vc, vsc, row);
            }
        }
        for (int gg = 0; gg < G; gg++) {
            float s[INK_AKB];
            float mx = m[gg];
            for (uint u = 0; u < INK_AKB; u++) {
                uint j = base + 8 * u;
                float4 pr = k4[u] * q4[gg];
                float dot = simd_sum((pr.x + pr.y) + (pr.z + pr.w));
                uint dist = pp - j;
                s[u] = j < j1 ? dot + (dist - dmin < nb ? bias[gg * 1024 + dist - dmin] : 0.f) : -INFINITY;
                mx = max(mx, s[u]);
            }
            float al = mx == -INFINITY ? 1.f : exp(m[gg] - mx);
            float4 acc = o[gg] * al;
            float ls = l[gg] * al;
            for (uint u = 0; u < INK_AKB; u++) {
                float e = s[u] == -INFINITY ? 0.f : exp(s[u] - mx);
                ls += e;
                acc = fma(e, v4[u], acc);
            }
            o[gg] = acc;
            l[gg] = ls;
            m[gg] = mx;
        }
    }
    // (`so` held the pass's keys: every SIMD-group is done with them first.)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int gg = 0; gg < G; gg++) {
        if (lane == 0) { sm[gg * 16 + sg] = m[gg]; sm[gg * 16 + 8 + sg] = l[gg]; }
        *(threadgroup float4*)(so + (gg * 8 + sg) * HD + lane * D) = o[gg];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ink_attv_fin<D, G>(a, p, ns, split, ti, h0, tid, sm, so);
}
#define INK_ATTF(D, G, F) \
kernel void ink_attf_##D##_##G##_##F(device float* a [[buffer(0)]], device const float* c [[buffer(1)]], constant IAP& p [[buffer(2)]], \
                         device uchar* kc [[buffer(3)]], device uchar* vc [[buffer(4)]], device float* ksc [[buffer(5)]], \
                         device float* vsc [[buffer(6)]], constant IKvP& kp [[buffer(11)]], constant IQP& qp [[buffer(12)]], \
                         device float* rk [[buffer(13)]], device float* rv [[buffer(14)]], \
                         uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float sm[G * 16]; threadgroup float so[G * 8 * D * 32]; threadgroup float bias[G * 1024]; \
    threadgroup float red[128]; threadgroup float rr[G * 16]; \
    ink_attf_body<D, G, F>(a, c, p, kp, qp, kc, vc, ksc, vsc, rk, rv, tg.x, tg.y, tg.z, tid, sg, lane, sm, so, bias, red, rr); }
INK_ATTF(4, 3, 0) INK_ATTF(4, 3, 1) INK_ATTF(4, 3, 2)
INK_ATTF(4, 1, 0) INK_ATTF(4, 1, 1) INK_ATTF(4, 1, 2)
INK_ATTF(4, 2, 0) INK_ATTF(4, 2, 1) INK_ATTF(4, 2, 2)
INK_ATTF(4, 4, 0) INK_ATTF(4, 4, 1) INK_ATTF(4, 4, 2)

// ---- The splits of each (token, head) merged with the sink (one more logit, `sink`, of a zero
// value), then gated: `out = softmax-weighted values * sigmoid(gate)` (grid: head, token).
struct IACP { uint nh; uint hd; uint maxsplit; uint part; uint sink; uint gate; uint out; uint pos; uint window; uint cmin; uint fin; uint _p2; };
kernel void ink_attn_merge(device float* a [[buffer(0)]], constant IACP& p [[buffer(1)]],
                           uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    uint hh = tg.x, ti = tg.y, hd = p.hd;
    uint hi = p.pos + ti + 1;
    uint lo = p.window != 0 && hi > p.window ? hi - p.window : 0;
    uint ns = ink_nsplit(hi - lo, p.maxsplit, p.cmin);
    // (`fin`: tokens of one split were merged by `ink_attv`.)
    if (p.fin != 0 && ns == 1) return;
    for (uint i = tid; i < hd; i += 128) ink_merge_elem(a, p.part, p.nh, p.maxsplit, p.sink, p.gate, p.out, hh, ti, ns, hd, i);
}

// ---- The convolutions after the attention and the MLP (grid: channel blocks, token): `y = (x +
// sum_lag taps[lag] x[t - lag]) * norm` over the inputs `x` (earlier positions from the ring),
// then `out = y` (mode 0) or `out += y` (mode 1). The inputs go to the ring.
struct ICrP { uint h; uint kn; uint x; uint out; uint taps; uint pos; uint mode; float norm; uint ring_cap; uint ntok; uint ring_write; uint _p1;
              uint od; uint ok; uint _q0; uint _q1; };
// The input of token `tj` at channel `i`: its row of `x`, or with `ok` the sum of its `ok` slots of
// `od` (the routed experts' rows, in slot order).
inline float conv_in(device const float* a, constant ICrP& p, uint tj, uint i) {
    if (p.ok == 0) return a[p.x + (ulong)tj * p.h + i];
    float v = 0.f;
    for (uint s = 0; s < p.ok; s++) v += a[p.od + ((ulong)tj * p.ok + s) * p.h + i];
    return v;
}
// (`xin`: token ti's input, given.)
inline void ink_conv_in(device float* a, device const float* c, constant ICrP& p, device float* ring, uint i, uint ti, float xin) {
    uint h = p.h, kn = p.kn;
    uint pos = p.pos + ti;
    device float* o = a + p.out + (ulong)ti * h + i;
    float old = p.mode != 0 ? *o : 0.f;
    float s = xin;
    for (uint lag = 0; lag < kn; lag++) {
        float xv;
        if (lag <= ti) xv = lag == 0 ? xin : conv_in(a, p, ti - lag, i);
        else {
            int q = int(pos) - int(lag);
            xv = ring[(ulong)((q + int(kn) - 1 + int(p.ring_cap)) % int(p.ring_cap)) * h + i];
        }
        s += c[p.taps + lag * h + i] * xv;
    }
    s *= p.norm;
    // The ring takes the raw inputs of the last tokens (positions of earlier tokens are read above
    // from their rows, never from the ring, so the writes do not race with the reads).
    if (p.ring_write != 0) ring[(ulong)((pos + kn - 1) % p.ring_cap) * h + i] = xin;
    *o = p.mode == 0 ? s : old + s;
}
inline void ink_conv_one(device float* a, device const float* c, constant ICrP& p, device float* ring, uint i, uint ti) {
    ink_conv_in(a, c, p, ring, i, ti, conv_in(a, p, ti, i));
}
kernel void ink_conv(device float* a [[buffer(0)]], device const float* c [[buffer(1)]], constant ICrP& p [[buffer(2)]],
                     device float* ring [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.h) return;
    ink_conv_one(a, c, p, ring, gid.x, gid.y);
}

// ---- Matrix-vector products with float32 weights in the constants (the MoE routers; grid:
// blocks of 8 rows, token; one SIMD-group per row).
// (`ink_mvr`: with `ss`, x is the weighted output of `ink_gfg` and the logits are scaled by its norm.)
struct IMvfP { uint n_out; uint n_in; uint w; uint x; uint y; uint ss; uint np; float eps; };
kernel void ink_mvf(device float* a [[buffer(0)]], device const float* c [[buffer(1)]], constant IMvfP& p [[buffer(2)]],
                    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
    uint r = tg.x * 8 + sg, ti = tg.y;
    if (r >= p.n_out) return;
    device const float4* w = (device const float4*)(c + p.w + (ulong)r * p.n_in);
    device const float4* x = (device const float4*)(a + p.x + (ulong)ti * p.n_in);
    float4 acc = 0.f;
    for (uint i = lane; i < p.n_in / 4; i += 32) acc = fma(w[i], x[i], acc);
    float s = simd_sum((acc.x + acc.y) + (acc.z + acc.w));
    if (lane == 0) a[p.y + (ulong)ti * p.n_out + r] = s;
}

// ---- Sigmoid routing (grid: token; one thread per expert, n <= 1024): the k experts of the
// largest `sigmoid(logit) + bias` (ties to the lower index), weighted by their sigmoid scores
// renormalized over the k, times `scale`: ids (uint bits) then weights into `sel` (2 k per token).
// (`ntok`: the tokens of `ink_route_group`.)
struct IRouteP { uint n; uint k; uint logits; uint sel; uint bias; float scale; uint single; uint grp; uint asg; uint ntok; uint _p1; uint _p2; };
// The routing of token ti in one SIMD-group (n <= 128, k <= 32; 4 experts a lane, expert u * 32 +
// lane in slot u, no threadgroup barriers): lane j < k gets the j-th pick and its sigmoid score; all
// lanes the sum of the k scores.
inline void ink_route_sg(device const float* a, device const float* c, constant IRouteP& p, uint ti, uint lane, thread uint& myid,
                         thread float& mych, thread float& sum) {
    uint n = p.n, k = p.k;
    float scv[4], v[4];
    for (uint u = 0; u < 4; u++) {
        uint e = u * 32 + lane;
        float lg = e < n ? a[p.logits + (ulong)ti * n + e] : 0.f;
        float bs = e < n ? c[p.bias + e] : 0.f;
        scv[u] = e < n ? 1.f / (1.f + exp(-lg)) : 0.f;
        v[u] = e < n ? scv[u] + bs : -INFINITY;
    }
    myid = 0;
    mych = 0.f;
    sum = 0.f;
    for (uint j = 0; j < k; j++) {
        float bm = v[0], bsc = scv[0];
        uint bu = 0;
        for (uint u = 1; u < 4; u++) if (v[u] > bm) { bm = v[u]; bu = u; bsc = scv[u]; }
        float m = simd_max(bm);
        uint idx = simd_min(bm == m ? bu * 32 + lane : 0xffffffffu);
        float cs = simd_shuffle(bsc, idx % 32);
        if (lane == idx % 32) for (uint u = 0; u < 4; u++) if (u == idx / 32) v[u] = -INFINITY;
        if (lane == j) { myid = idx; mych = cs; }
        sum += cs;
    }
}
kernel void ink_route(device float* a [[buffer(0)]], device const float* c [[buffer(1)]], constant IRouteP& p [[buffer(2)]],
                      uint ti [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint nt [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float bv[32];
    threadgroup uint bi[32];
    threadgroup float chosen[64];
    uint n = p.n, k = p.k;
    device float* sel = a + p.sel + (ulong)ti * 2 * k;
    if (n <= 128 && k <= 32) {
        // (One SIMD-group, 4 experts a lane, expert u * 32 + lane in slot u: no threadgroup
        // barriers. Lane j keeps the j-th pick.)
        if (sg != 0) return;
        uint myid;
        float mych, sum;
        ink_route_sg(a, c, p, ti, lane, myid, mych, sum);
        if (lane < k) {
            sel[lane] = as_type<float>(myid);
            sel[k + lane] = mych / sum * p.scale;
            if (p.single != 0) {
                device uint* g = (device uint*)(a + p.grp);
                device uint* asg = (device uint*)(a + p.asg);
                if (lane == 0) g[0] = k;
                g[1 + 3 * lane] = myid;
                g[2 + 3 * lane] = 1;
                g[3 + 3 * lane] = lane;
                asg[lane] = lane;
            }
        }
        return;
    }
    float sc = tid < n ? 1.f / (1.f + exp(-a[p.logits + (ulong)ti * n + tid])) : 0.f;
    float v = tid < n ? sc + c[p.bias + tid] : -INFINITY;
    uint nsg = (nt + 31) / 32;
    for (uint j = 0; j < k; j++) {
        float m = simd_max(v);
        uint idx = v == m ? tid : 0xffffffffu;
        idx = simd_min(idx);
        if (lane == 0) { bv[sg] = m; bi[sg] = idx; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float best = bv[0]; uint bidx = bi[0];
        for (uint i = 1; i < nsg; i++) if (bv[i] > best || (bv[i] == best && bi[i] < bidx)) { best = bv[i]; bidx = bi[i]; }
        if (tid == bidx) { v = -INFINITY; chosen[j] = sc; }
        if (tid == 0) sel[j] = as_type<float>(bidx);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float s = 0.f;
        for (uint j = 0; j < k; j++) s += chosen[j];
        for (uint j = 0; j < k; j++) sel[k + j] = chosen[j] / s * p.scale;
        if (p.single != 0) {
            device uint* g = (device uint*)(a + p.grp);
            device uint* asg = (device uint*)(a + p.asg);
            g[0] = k;
            for (uint j = 0; j < k; j++) {
                g[1 + 3 * j] = as_type<uint>(sel[j]);
                g[2 + 3 * j] = 1;
                g[3 + 3 * j] = j;
                asg[j] = j;
            }
        }
    }
}

// ---- The router's logits of passes of up to 16 tokens (grid: expert, token; a threadgroup of 128
// per row, float rows of `n_in`, a multiple of 4, in the constants), as `ink_mvf`.
kernel void ink_mvr(device float* a [[buffer(0)]], device const float* c [[buffer(1)]], constant IMvfP& p [[buffer(2)]],
                    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[4];
    uint r = tg.x, ti = tg.y, n4 = p.n_in / 4;
    device const float4* w = (device const float4*)(c + p.w + (ulong)r * p.n_in);
    device const float4* x = (device const float4*)(a + p.x + (ulong)ti * p.n_in);
    // (Up to 8 float4 per thread, loaded together: n_in <= 4096.)
    float4 wv[8], xv[8];
    for (uint u = 0; u < 8; u++) {
        uint k = tid + 128 * u;
        wv[u] = k < n4 ? w[k] : 0.f;
        xv[u] = k < n4 ? x[k] : 0.f;
    }
    float4 acc = 0.f;
    for (uint u = 0; u < 8; u++) acc = fma(wv[u], xv[u], acc);
    float s = simd_sum((acc.x + acc.y) + (acc.z + acc.w));
    if (lane == 0) red[sg] = s;
    float sc = sg == 0 && p.ss != 0 ? ink_gscale(a, p.ss + ti * 64, p.np, p.n_in, p.eps, lane) : 1.f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) a[p.y + (ulong)ti * p.n_out + r] = ((red[0] + red[1]) + (red[2] + red[3])) * sc;
}

// ---- The soft-capped logits of the tokens (`cap tanh(x / cap)`), cut to the first `u` of each
// row of `v` (grid: blocks, token).
// With `am` (greedy decoding on the device, one row), each threadgroup's argmax as (value, index
// bits) into `am + 2 tg` (`ink_amax` then takes the first index of the largest).
struct ICapP { uint v; uint u; uint x; uint out; float cap; uint am; uint _p1; uint _p2; };
// (v, i) for (bv, bi): the larger value, the lower index on a tie (indices of no candidate:
// 0xffffffff, value -inf).
inline void amax_pick(thread float& bv, thread uint& bi, float v, uint i) {
    if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
}
inline void amax_simd(thread float& bv, thread uint& bi) {
    for (ushort m = 16; m > 0; m >>= 1) amax_pick(bv, bi, simd_shuffle_xor(bv, m), simd_shuffle_xor(bi, m));
}
kernel void ink_softcap(device float* a [[buffer(0)]], constant ICapP& p [[buffer(1)]], uint2 gid [[thread_position_in_grid]],
                        uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float tv[8];
    threadgroup uint ti[8];
    bool in = gid.x < p.u;
    float y = -INFINITY;
    if (in) {
        float x = a[p.x + (ulong)gid.y * p.v + gid.x];
        y = precise::tanh(x / p.cap) * p.cap;
        a[p.out + (ulong)gid.y * p.u + gid.x] = y;
    }
    if (p.am == 0) return;
    // (The host's argmax: the first index of the largest value above -inf; NaNs never win.)
    bool c = in && y > -INFINITY;
    float bv = c ? y : -INFINITY;
    uint bi = c ? gid.x : 0xffffffffu;
    amax_simd(bv, bi);
    if (lane == 0) { tv[sg] = bv; ti[sg] = bi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        for (uint k = 1; k < 8; k++) amax_pick(bv, bi, tv[k], ti[k]);
        a[p.am + 2 * tg.x] = bv;
        a[p.am + 2 * tg.x + 1] = as_type<float>(bi);
    }
}
// The argmax over `n` partials (one threadgroup of 1024): the token id (uint bits) into `tok`.
struct IAmP { uint n; uint am; uint tok; uint _p; };
kernel void ink_amax(device float* a [[buffer(0)]], constant IAmP& p [[buffer(1)]], uint tid [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float tv[32];
    threadgroup uint ti[32];
    float bv = -INFINITY;
    uint bi = 0xffffffffu;
    for (uint k = tid; k < p.n; k += 1024) amax_pick(bv, bi, a[p.am + 2 * k], as_type<uint>(a[p.am + 2 * k + 1]));
    amax_simd(bv, bi);
    if (lane == 0) { tv[sg] = bv; ti[sg] = bi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        for (uint k = 1; k < 32; k++) amax_pick(bv, bi, tv[k], ti[k]);
        ((device uint*)a)[p.tok] = bi == 0xffffffffu ? 0u : bi;
    }
}

// ---- The input row of a pass whose token the device chose (`ink_amax` of the pass before): the
// embedding table's row of the token (uint at `tok`) dequantized as the host does, then the
// embedding norm (weights at `nw`) into `x` (one threadgroup of 1024; h <= 4096). Tables: MLX
// affine rows (`bits` 2, 4 or 8, a bf16 scale and offset per `group`; `q`, `s`, `b` addresses),
// GGUF rows of `rb` bytes (`ink_embed_<type>`), bf16 rows.
struct IEmbP { ulong q; ulong s; ulong b; uint h; uint tok; uint x; uint nw; float eps; uint rb; uint bits; uint group; };
inline void ink_embed_norm(device float* a, device const float* c, constant IEmbP& p, threadgroup float* xs, threadgroup float* red,
                           uint tid) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ss = 0.f;
    for (uint i = tid; i < p.h; i += 1024) ss += xs[i] * xs[i];
    ss = ink_sum(ss, red, tid, 1024);
    float r = 1.f / sqrt(ss / float(p.h) + p.eps);
    for (uint i = tid; i < p.h; i += 1024) a[p.x + i] = xs[i] * (r * c[p.nw + i]);
}
inline float bf16f(ushort v) { return as_type<float>(uint(v) << 16); }
kernel void ink_embed_mlx(device float* a [[buffer(0)]], device const float* c [[buffer(1)]], constant IEmbP& p [[buffer(2)]],
                          uint tid [[thread_index_in_threadgroup]]) {
#pragma clang fp contract(off)
    threadgroup float xs[4096];
    threadgroup float red[32];
    uint row = ((device const uint*)a)[p.tok];
    uint per = 32 / p.bits, mask = (1u << p.bits) - 1, ng = p.h / p.group;
    device const uint* w = (device const uint*)p.q + (ulong)row * (p.h / per);
    device const ushort* sc = (device const ushort*)p.s + (ulong)row * ng;
    device const ushort* bs = (device const ushort*)p.b + (ulong)row * ng;
    for (uint i = tid; i < p.h; i += 1024) {
        uint q = (w[i / per] >> (p.bits * (i % per))) & mask;
        uint g = i / p.group;
        float t = bf16f(sc[g]) * float(q);
        xs[i] = t + bf16f(bs[g]);
    }
    ink_embed_norm(a, c, p, xs, red, tid);
}
kernel void ink_embed_bf(device float* a [[buffer(0)]], device const float* c [[buffer(1)]], constant IEmbP& p [[buffer(2)]],
                         uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float xs[4096];
    threadgroup float red[32];
    uint row = ((device const uint*)a)[p.tok];
    device const ushort* w = (device const ushort*)p.q + (ulong)row * p.h;
    for (uint i = tid; i < p.h; i += 1024) xs[i] = bf16f(w[i]);
    ink_embed_norm(a, c, p, xs, red, tid);
}
#define INK_EMB(T) \
kernel void ink_embed_##T(device float* a [[buffer(0)]], device const float* c [[buffer(1)]], constant IEmbP& p [[buffer(2)]], \
                          uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup float xs[4096]; \
    threadgroup float red[32]; \
    uint row = ((device const uint*)a)[p.tok]; \
    T##W w = { { (device const uchar*)p.q, p.rb } }; \
    for (uint b = tid; b < p.h / 32; b += 1024) { \
        float4 v[8]; \
        load(w, row, b, v); \
        for (uint i = 0; i < 8; i++) *(threadgroup float4*)(xs + b * 32 + 4 * i) = v[i]; \
    } \
    ink_embed_norm(a, c, p, xs, red, tid); }

// ---- Weight formats of the experts beyond the decode library's. NVFP4 (ggml's blocks of 64:
// four unsigned E4M3 scales, halved, then four runs of 8 bytes holding elements j and j + 8 of
// a sub-block of 16; the codes through MXFP4's doubled table): chunk b of 32 values is half of
// block b / 2.
struct NVFP4W { GgW g; };
// Four E2M1 codes (the low nibbles of v's bytes) as floats over 2^-14: the code's exponent and mantissa
// bits are a half's (subnormal for exponent 0), its sign the half's sign (no table lookups).
inline float4 e2m1x4(uint v) {
    uint lo = ((v & 0x00070007u) << 9) | ((v & 0x00080008u) << 12);
    uint hi = (((v >> 8) & 0x00070007u) << 9) | (((v >> 8) & 0x00080008u) << 12);
    half2 a = as_type<half2>(lo), b = as_type<half2>(hi);
    return float4(float(a.x), float(b.x), float(a.y), float(b.y));
}
inline void load(NVFP4W w, uint row, uint b, thread float4* out) {
    device const uchar* B = w.g.q + (ulong)row * w.g.rb + (b / 2) * 36;
    for (uint s2 = 0; s2 < 2; s2++) {
        uint s = 2 * (b % 2) + s2;
        // (The scale, halved, times the codes' 2^14 and the table's doubling.)
        float d = float(as_type<half>(ushort(ushort(B[s] & 0x7f) << 7))) * (128.f * 32768.f);
        // (Blocks of 36 bytes in rows of whole blocks: the runs are 4-byte aligned.) Byte i's
        // low nibble is element i, its high one element i + 8 of the sub-block.
        uint v0 = *(device const uint*)(B + 4 + 8 * s), v1 = *(device const uint*)(B + 8 + 8 * s);
        out[4 * s2 + 0] = e2m1x4(v0) * d;
        out[4 * s2 + 1] = e2m1x4(v1) * d;
        out[4 * s2 + 2] = e2m1x4(v0 >> 4) * d;
        out[4 * s2 + 3] = e2m1x4(v1 >> 4) * d;
    }
}
// MXFP8: E4M3 bytes with a float scale per 32 values of a row (the E8M0 exponents expanded).
struct MXF8W { device const uchar* q; device const float* s; uint n_in; };
inline void load(MXF8W w, uint row, uint b, thread float4* out) {
    float d = w.s[(ulong)row * (w.n_in / 32) + b] * 256.f;
    device const uchar* B = w.q + (ulong)row * w.n_in + b * 32;
    for (uint i = 0; i < 8; i++) out[i] = e4m3x4(ld4(B + 4 * i)) * d;
}
// MXFP8 with the E8M0 exponent bytes as stored (borrowed from the mapped file): `2^(e - 127)`
// decoded from the bits as the CPU does (0: 2^-127, 255: NaN), the same floats as the expanded
// scales, so the same outputs bit for bit.
struct MXF8B { device const uchar* q; device const uchar* e; uint n_in; };
inline float e8m0f(uint e) { return as_type<float>(e == 255 ? 0x7fc00000u : e == 0 ? 0x00400000u : e << 23); }
inline void load(MXF8B w, uint row, uint b, thread float4* out) {
    float d = e8m0f(w.e[(ulong)row * (w.n_in / 32) + b]) * 256.f;
    device const uchar* B = w.q + (ulong)row * w.n_in + b * 32;
    for (uint i = 0; i < 8; i++) out[i] = e4m3x4(ld4(B + 4 * i)) * d;
}

// ---- Routed experts (threadgroups of 4 SIMD-groups, each computing two outputs). Expert e's
// projection is at tab[e]: data, scales (address, 0 if none), the factor of its outputs, the row
// bytes. Pair kernels (grid: blocks of 8 outputs j, slot, token): the gate and up rows of the
// token's slot-th expert (`il` 2: interleaved rows 2j, 2j + 1; else gate rows j and up rows
// inter + j), `act = cap_g tanh(g / cap_g) sigmoid(g) * cap_u tanh(u / cap_u)` (SiTU) into `act`
// (`[token][slot][inter]`). Sum kernels (grid: blocks of 8 rows, token): `y = sum over slots of
// weight * factor * (down_e act)`.
struct IXT { ulong q; ulong s; float factor; uint rb; ulong _p; };
// An expert's LoRA update (`ink_xlora_*`): A and Bt (addresses), its rank and strength.
struct IXL { ulong a; ulong bt; uint r; float s; ulong _p; };
// The gate / up rows' LoRA updates of the pair kernels: `on`, the arena offset of the slots' `A x`
// (`[t k][rmax]`, from `ink_xlora_a`) and rmax.
struct IXQ { uint on; uint u; uint rmax; uint _p; };
// The update of stored gate / up row `row` of expert `ex`'s pair for slot `slot`: `s Bt[.][row] . u`.
inline float ink_gu_lora(device const float* a, device const IXL* lt, constant IXQ& lq, uint ex, uint slot, uint row, uint n_out) {
    IXL x = lt[ex];
    device const float* bt = (device const float*)x.bt;
    device const float* u = a + lq.u + (ulong)slot * lq.rmax;
    float v = 0.f;
    for (uint q = 0; q < x.r; q++) v = fma(bt[(ulong)q * n_out + row], u[q], v);
    return x.s * v;
}
// (`ss`: x is the weighted output of `ink_gfg`, the gate and up sums scaled by its norm, `np`, `eps`.)
struct IXP { uint inter; uint h; uint k; uint il; uint off; uint x; uint sel; uint act; uint y; float cap_g; float cap_u; uint add;
             uint ss; uint np; float eps; uint _p; };
#define INK_XP(NAME, T, X, CH, MAKE) \
kernel void NAME##_pair(device float* a [[buffer(0)]], device const IXT* tab [[buffer(1)]], constant IXP& p [[buffer(2)]], \
                        device const IXL* lt [[buffer(3)]], constant IXQ& lq [[buffer(4)]], \
                        uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], \
                        uint lane [[thread_index_in_simdgroup]]) { \
    uint j = (tg.x * 4 + sg) * 2, s = tg.y, ti = tg.z; \
    if (j >= p.inter) return; \
    device const float* sel = a + p.sel + (ulong)ti * 2 * p.k; \
    uint ex = as_type<uint>(sel[s]); \
    IXT e = tab[ex]; \
    T f = MAKE(e, p.h); \
    device const float* x = a + p.x + (ulong)ti * p.h; \
    float g[2], u[2]; \
    if (p.il == 2) { \
        float acc[4] = { 0.f, 0.f, 0.f, 0.f }, acc2[4] = { 0.f, 0.f, 0.f, 0.f }; \
        mv_acc<T, X, CH, 4, 1>(f, f, false, 0, x, p.h, 2 * p.inter, 2 * j, 0, lane, acc, acc2, 1.f); \
        g[0] = simd_sum(acc[0]); u[0] = simd_sum(acc[1]); g[1] = simd_sum(acc[2]); u[1] = simd_sum(acc[3]); \
    } else { \
        float acc[2] = { 0.f, 0.f }, acc2[2] = { 0.f, 0.f }; \
        mv_acc<T, X, CH, 2, 1>(f, f, true, p.off, x, p.h, 2 * p.inter, j, 0, lane, acc, acc2, 1.f); \
        g[0] = simd_sum(acc[0]); g[1] = simd_sum(acc[1]); u[0] = simd_sum(acc2[0]); u[1] = simd_sum(acc2[1]); \
    } \
    if (lane == 0) \
        for (uint i = 0; i < 2 && j + i < p.inter; i++) { \
            float gg = g[i] * e.factor, uu = u[i] * e.factor; \
            if (lq.on != 0) { \
                uint rg = p.il == 2 ? 2 * (j + i) : j + i, ru = rg + p.off; \
                gg += ink_gu_lora(a, lt, lq, ex, ti * p.k + s, rg, 2 * p.inter); \
                uu += ink_gu_lora(a, lt, lq, ex, ti * p.k + s, ru, 2 * p.inter); \
            } \
            float cu = p.cap_u > 0.f ? p.cap_u * precise::tanh(uu / p.cap_u) : uu; \
            a[p.act + ((ulong)ti * p.k + s) * p.inter + j + i] = p.cap_g * precise::tanh(gg / p.cap_g) / (1.f + exp(-gg)) * cu; \
        } } \
kernel void NAME##_sum(device float* a [[buffer(0)]], device const IXT* tab [[buffer(1)]], constant IXP& p [[buffer(2)]], \
                       uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], \
                       uint lane [[thread_index_in_simdgroup]]) { \
    uint r = (tg.x * 4 + sg) * 2, ti = tg.y; \
    if (r >= p.h) return; \
    device const float* sel = a + p.sel + (ulong)ti * 2 * p.k; \
    float acc[2] = { 0.f, 0.f }, acc2[2] = { 0.f, 0.f }; \
    for (uint s = 0; s < p.k; s++) { \
        IXT e = tab[as_type<uint>(sel[s])]; T f = MAKE(e, p.inter); \
        mv_acc<T, X, CH, 2, 1>(f, f, false, 0, a + p.act + ((ulong)ti * p.k + s) * p.inter, p.inter, p.h, r, 0, lane, acc, acc2, \
                               sel[p.k + s] * e.factor); \
    } \
    float v0 = simd_sum(acc[0]), v1 = simd_sum(acc[1]); \
    if (lane == 0) { \
        device float* y = a + p.y + (ulong)ti * p.h + r; \
        if (p.add != 0) { y[0] += v0; if (r + 1 < p.h) y[1] += v1; } else { y[0] = v0; if (r + 1 < p.h) y[1] = v1; } \
    } }
// Q2_0 (ggml type 42): an f16 scale, then 16 bytes of 2-bit codes `c - 1` (element j: bits
// 2 (j % 4) of byte j / 4) per block of 64; chunk b of 32 is half of block b / 2.
struct Q2_0W { GgW g; };
inline void load(Q2_0W w, uint row, uint b, thread float4* out) {
    device const uchar* B = w.g.q + (ulong)row * w.g.rb + (b / 2) * 18;
    float d = h2f(B);
    device const uchar* q = B + 2 + 8 * (b % 2);
    for (uint i = 0; i < 8; i++) {
        uint c = q[i];
        out[i] = (float4(float(c & 3), float((c >> 2) & 3), float((c >> 4) & 3), float(c >> 6)) - 1.f) * d;
    }
}

// ---- Runtime LoRA updates (`crate::lora`): `y += s B (A x)` beside a matvec or GEMM. The down
// half (grid: rank, token; one SIMD-group each): `u[t][r] = A[r] . x[t]`; the up half (grid: blocks
// of outputs, token): `y[t][o] += s sum_r Bt[r][o] u[t][r]`. A and Bt are float32 tables.
struct ILaP { uint n_in; uint r; uint x; uint u; };
kernel void ink_lora_a(device float* a [[buffer(0)]], device const float* A [[buffer(1)]], constant ILaP& p [[buffer(2)]],
                       uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint r = tg.x, ti = tg.y;
    device const float* w = A + (ulong)r * p.n_in;
    device const float* x = a + p.x + (ulong)ti * p.n_in;
    float s = 0.f;
    for (uint i = lane; i < p.n_in; i += 32) s = fma(w[i], x[i], s);
    s = simd_sum(s);
    if (lane == 0) a[p.u + (ulong)ti * p.r + r] = s;
}
struct ILbP { uint n_out; uint r; uint u; uint y; float s; uint _p0; uint _p1; uint _p2; };
kernel void ink_lora_b(device float* a [[buffer(0)]], device const float* Bt [[buffer(1)]], constant ILbP& p [[buffer(2)]],
                       uint2 gid [[thread_position_in_grid]]) {
    uint o = gid.x, ti = gid.y;
    if (o >= p.n_out) return;
    float v = 0.f;
    for (uint k = 0; k < p.r; k++) v = fma(Bt[(ulong)k * p.n_out + o], a[p.u + (ulong)ti * p.r + k], v);
    a[p.y + (ulong)ti * p.n_out + o] += p.s * v;
}
// The same for the routed experts' down projections, per expert: entry e of the table holds its
// A and Bt (addresses), its rank and its strength. The down half (grid: rank, slot, token; or with
// `rows`, grid: rank, row: the grouped rows of a prefill, each of its expert `rexp[row]`, inputs
// half rows scaled by `rs[row]`): `u[slot] = A_e . act[slot]`; the up half (grid: blocks of
// outputs, token): `y[t] += sum_s w_s s_e Bt_e u[map[t k + s]]` (map: identity without `rows`).
struct IXlaP { uint n_in; uint k; uint rmax; uint act; uint u; uint sel; uint rows; uint tok; };
kernel void ink_xlora_a(device float* a [[buffer(0)]], device const IXL* tab [[buffer(1)]], constant IXlaP& p [[buffer(2)]],
                        device const half* ah [[buffer(3)]], device const float* rs [[buffer(4)]], device const uint* rexp [[buffer(5)]],
                        uint3 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint r = tg.x;
    uint e, slot;
    if (p.rows != 0) {
        slot = tg.y;
        if (slot >= p.rows) return;
        e = rexp[slot];
        if (e == 0xffffffffu) return;
    } else {
        slot = tg.z * p.k + tg.y;
        e = as_type<uint>(a[p.sel + (ulong)tg.z * 2 * p.k + tg.y]);
    }
    IXL x = tab[e];
    if (r >= x.r) return;
    device const float* w = (device const float*)x.a + (ulong)r * p.n_in;
    float s = 0.f;
    if (p.rows != 0) {
        device const half* v = ah + (ulong)slot * p.n_in;
        for (uint i = lane; i < p.n_in; i += 32) s = fma(w[i], float(v[i]), s);
        s *= rs[slot];
    } else {
        // (`tok`: every slot of a token reads the token's row, the gate / up updates' input.)
        device const float* v = a + p.act + (ulong)(p.tok != 0 ? tg.z : slot) * p.n_in;
        for (uint i = lane; i < p.n_in; i += 32) s = fma(w[i], v[i], s);
    }
    s = simd_sum(s);
    if (lane == 0) a[p.u + (ulong)slot * p.rmax + r] = s;
}
struct IXlbP { uint n_out; uint k; uint rmax; uint u; uint sel; uint y; uint mapped; uint _p; };
kernel void ink_xlora_b(device float* a [[buffer(0)]], device const IXL* tab [[buffer(1)]], constant IXlbP& p [[buffer(2)]],
                        device const uint* map [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    uint o = gid.x, ti = gid.y;
    if (o >= p.n_out) return;
    device const float* sel = a + p.sel + (ulong)ti * 2 * p.k;
    float acc = 0.f;
    for (uint s = 0; s < p.k; s++) {
        IXL x = tab[as_type<uint>(sel[s])];
        uint slot = p.mapped != 0 ? map[ti * p.k + s] : ti * p.k + s;
        device const float* bt = (device const float*)x.bt;
        float v = 0.f;
        for (uint q = 0; q < x.r; q++) v = fma(bt[(ulong)q * p.n_out + o], a[p.u + (ulong)slot * p.rmax + q], v);
        acc += sel[p.k + s] * x.s * v;
    }
    a[p.y + (ulong)ti * p.n_out + o] += acc;
}

// ---- The routed experts of a pass grouped by expert (decoding and verification passes), so
// that tokens choosing the same expert read its weights once (from the cache for all but the
// first). `ink_xgroup` (one threadgroup of t k <= 1024 threads, one per (token, slot)
// assignment): the experts in order of first appearance, each with its assignments, as `grp`
// (`[count of groups] [expert, count, start]...`) and `asg` (assignment ids `token * k + slot`,
// grouped). A token's results do not depend on the grouping.
struct IXgP { uint k; uint n; uint sel; uint grp; uint asg; uint _p0; uint _p1; uint _p2; };
kernel void ink_xgroup(device float* a [[buffer(0)]], constant IXgP& p [[buffer(1)]], uint i [[thread_index_in_threadgroup]]) {
    threadgroup uint ex[1024];
    threadgroup uint fs[1024];
    uint k = p.k, n = p.n;
    if (i < n) ex[i] = as_type<uint>(a[p.sel + (ulong)(i / k) * 2 * k + i % k]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // The first assignment of each one's expert.
    uint e = i < n ? ex[i] : 0, first = i;
    if (i < n)
        for (uint j = 0; j < i; j++) if (ex[j] == e) { first = j; break; }
    fs[i] = first;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (i >= n) return;
    // Its group (the leaders before `first`), the assignments of the earlier groups, its rank in
    // its group and the group's size.
    uint gid = 0, before = 0, rank = 0, cnt = 0, ng = 0;
    for (uint j = 0; j < n; j++) {
        uint fj = fs[j];
        if (fj == j) { ng++; if (j < first) gid++; }
        if (fj < first) before++;
        if (fj == first) { cnt++; if (j < i) rank++; }
    }
    device uint* g = (device uint*)(a + p.grp);
    device uint* asg = (device uint*)(a + p.asg);
    asg[before + rank] = i;
    if (i == first) {
        g[1 + 3 * gid] = e;
        g[2 + 3 * gid] = cnt;
        g[3 + 3 * gid] = before;
    }
    if (i == 0) g[0] = ng;
}
// The routing of a pass's tokens (`ink_route`'s single SIMD-group routing, SIMD-group s taking
// tokens s, s + groups, ...) and `ink_xgroup` in one threadgroup of at least t k threads (n <= 128
// experts, k <= 32): the same `sel`, `grp` and `asg`.
kernel void ink_route_group(device float* a [[buffer(0)]], device const float* c [[buffer(1)]], constant IRouteP& rp [[buffer(2)]],
                            uint i [[thread_index_in_threadgroup]], uint nth [[threads_per_threadgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint ex[1024];
    threadgroup uint fs[1024];
    uint k = rp.k, nt = rp.ntok, n = nt * k;
    for (uint ti = sg; ti < nt; ti += nth / 32) {
        uint myid;
        float mych, sum;
        ink_route_sg(a, c, rp, ti, lane, myid, mych, sum);
        if (lane < k) {
            a[rp.sel + (ulong)ti * 2 * k + lane] = as_type<float>(myid);
            a[rp.sel + (ulong)ti * 2 * k + k + lane] = mych / sum * rp.scale;
            ex[ti * k + lane] = myid;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint e = i < n ? ex[i] : 0, first = i;
    if (i < n)
        for (uint j = 0; j < i; j++) if (ex[j] == e) { first = j; break; }
    fs[i] = first;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (i >= n) return;
    uint gid = 0, before = 0, rank = 0, cnt = 0, ng = 0;
    for (uint j = 0; j < n; j++) {
        uint fj = fs[j];
        if (fj == j) { ng++; if (j < first) gid++; }
        if (fj < first) before++;
        if (fj == first) { cnt++; if (j < i) rank++; }
    }
    device uint* g = (device uint*)(a + rp.grp);
    device uint* asg = (device uint*)(a + rp.asg);
    asg[before + rank] = i;
    if (i == first) {
        g[1 + 3 * gid] = e;
        g[2 + 3 * gid] = cnt;
        g[3 + 3 * gid] = before;
    }
    if (i == 0) g[0] = ng;
}
// Per (token, slot): the down projection's rows times the routing weight and the expert's factor,
// into `od` (`[t k][h]`); `ink_xcombine` sums a token's slots in order into `y`.
struct IXcP { uint h; uint k; uint od; uint y; uint add; uint _p0; uint _p1; uint _p2; };
kernel void ink_xcombine(device float* a [[buffer(0)]], constant IXcP& p [[buffer(1)]], uint2 gid [[thread_position_in_grid]]) {
    uint r = gid.x, ti = gid.y;
    if (r >= p.h) return;
    float v = 0.f;
    for (uint s = 0; s < p.k; s++) v += a[p.od + ((ulong)ti * p.k + s) * p.h + r];
    device float* y = a + p.y + (ulong)ti * p.h + r;
    if (p.add != 0) *y += v; else *y = v;
}
// `_rpair` (passes of one token): `_gpair` with the routing (`ink_route`'s) in every SIMD-group first,
// group y being the y-th pick; one threadgroup writes the picks and their weights to `sel`.
// `_dconv` (passes of one token; grid: blocks of 32 / lpr rows, k SIMD-groups): SIMD-group s computes
// slot s's down rows as `_gsum` does, then the convolution after the experts takes their sum in
// slot order (`ink_conv`'s input) and writes the stream and the ring: no stage between them.
// The pair and down kernels over the groups (grid: blocks of outputs, group): as `_pair` for every
// assignment of the group; the down rows of every assignment into `od` (`p.y` is `od`). The lanes
// of a SIMD-group are split into row groups of `lpr` lanes (`gq.z`, a power of two dividing the
// rows' chunk count, at most 16 for the pairs), each walking one row (chunk `jj + lpr i` for lane
// jj): every lane has the same number of chunks, and a row's sum is a reduction over its group.
// A pair kernel's row groups alternate the gate and up rows of consecutive outputs (16 / lpr
// outputs per SIMD-group); a down kernel's are consecutive rows (32 / lpr per SIMD-group).
inline float ink_lsum(float v, uint lpr) {
    for (uint m = 1; m < lpr; m <<= 1) v += simd_shuffle_xor(v, m);
    return v;
}
template <typename F, typename X, uint CH>
inline float ink_row_acc(F f, uint row, device const float* x, uint nch, uint jj, uint lpr) {
    float a = 0.f;
    for (uint c = jj; c < nch; c += lpr) {
        X xr;
        xload(x, c, xr);
        a += mdot(f, row, c, xr);
    }
    return a;
}
// NVFP4 (decode-path experts): the scales applied to each sub-block's dot product; the 16 code
// bytes of a chunk loaded together.
inline float mdot(MGen<NVFP4W> f, uint row, uint c, thread const X32& xr) {
    device const uchar* B = f.w.g.q + (ulong)row * f.w.g.rb + (c / 2) * 36;
    uint s = 2 * (c % 2);
    device const uint* cw = (device const uint*)(B + 4 + 8 * s);
    uint v0 = cw[0], v1 = cw[1], v2 = cw[2], v3 = cw[3];
    float d0 = float(as_type<half>(ushort(ushort(B[s] & 0x7f) << 7))) * (128.f * 32768.f);
    float d1 = float(as_type<half>(ushort(ushort(B[s + 1] & 0x7f) << 7))) * (128.f * 32768.f);
    float4 s0 = e2m1x4(v0) * xr.v[0];
    s0 = fma(e2m1x4(v1), xr.v[1], s0);
    s0 = fma(e2m1x4(v0 >> 4), xr.v[2], s0);
    s0 = fma(e2m1x4(v1 >> 4), xr.v[3], s0);
    float4 s1 = e2m1x4(v2) * xr.v[4];
    s1 = fma(e2m1x4(v3), xr.v[5], s1);
    s1 = fma(e2m1x4(v2 >> 4), xr.v[6], s1);
    s1 = fma(e2m1x4(v3 >> 4), xr.v[7], s1);
    return fma(d0, hsum(s0), d1 * hsum(s1));
}
#define INK_XG(NAME, T, X, CH, MAKE) \
kernel void NAME##_gpair(device float* a [[buffer(0)]], device const IXT* tab [[buffer(1)]], constant IXP& p [[buffer(2)]], \
                         constant uint4& gq [[buffer(3)]], device const IXL* lt [[buffer(4)]], constant IXQ& lq [[buffer(5)]], \
                         uint2 tg [[threadgroup_position_in_grid]], \
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    uint lpr = gq.z, rg = lane / lpr, jj = lane % lpr; \
    uint j = (tg.x * 4 + sg) * (16 / lpr) + rg / 2, up = rg % 2; \
    device const uint* g = (device const uint*)(a + gq.x); \
    device const uint* asg = (device const uint*)(a + gq.y); \
    if (tg.y >= g[0]) return; \
    uint ge = g[1 + 3 * tg.y], cnt = g[2 + 3 * tg.y], st = g[3 + 3 * tg.y]; \
    IXT e = tab[ge]; \
    T f = MAKE(e, p.h); \
    uint jc = min(j, p.inter - 1); \
    uint row = p.il == 2 ? 2 * jc + up : jc + up * p.off; \
    for (uint q = 0; q < cnt; q++) { \
        uint asn = asg[st + q], ti = asn / p.k, s = asn % p.k; \
        float v = ink_lsum(ink_row_acc<T, X, CH>(f, row, a + p.x + (ulong)ti * p.h, p.h / CH, jj, lpr), lpr); \
        if (p.ss != 0) v *= ink_gscale(a, p.ss + ti * 64, p.np, p.h, p.eps, lane); \
        float uv = simd_shuffle_down(v, lpr); \
        if (jj == 0 && up == 0 && j < p.inter) { \
            float gg = v * e.factor, uu = uv * e.factor; \
            if (lq.on != 0) { \
                uint rgt = p.il == 2 ? 2 * j : j, ru = rgt + p.off; \
                gg += ink_gu_lora(a, lt, lq, ge, asn, rgt, 2 * p.inter); \
                uu += ink_gu_lora(a, lt, lq, ge, asn, ru, 2 * p.inter); \
            } \
            float cu = p.cap_u > 0.f ? p.cap_u * precise::tanh(uu / p.cap_u) : uu; \
            a[p.act + (ulong)asn * p.inter + j] = p.cap_g * precise::tanh(gg / p.cap_g) / (1.f + exp(-gg)) * cu; \
        } \
    } } \
kernel void NAME##_rpair(device float* a [[buffer(0)]], device const IXT* tab [[buffer(1)]], constant IXP& p [[buffer(2)]], \
                         constant uint4& gq [[buffer(3)]], device const float* c [[buffer(6)]], constant IRouteP& rp [[buffer(7)]], \
                         uint2 tg [[threadgroup_position_in_grid]], \
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    uint myid; float mych, sum; \
    ink_route_sg(a, c, rp, 0, lane, myid, mych, sum); \
    if (tg.x == 0 && tg.y == 0 && sg == 0 && lane < p.k) { \
        a[p.sel + lane] = as_type<float>(myid); \
        a[p.sel + p.k + lane] = mych / sum * rp.scale; \
    } \
    uint lpr = gq.z, rg = lane / lpr, jj = lane % lpr; \
    uint j = (tg.x * 4 + sg) * (16 / lpr) + rg / 2, up = rg % 2; \
    uint ge = simd_shuffle(myid, tg.y); \
    IXT e = tab[ge]; \
    T f = MAKE(e, p.h); \
    uint jc = min(j, p.inter - 1); \
    uint row = p.il == 2 ? 2 * jc + up : jc + up * p.off; \
    float v = ink_lsum(ink_row_acc<T, X, CH>(f, row, a + p.x, p.h / CH, jj, lpr), lpr); \
    if (p.ss != 0) v *= ink_gscale(a, p.ss, p.np, p.h, p.eps, lane); \
    float uv = simd_shuffle_down(v, lpr); \
    if (jj == 0 && up == 0 && j < p.inter) { \
        float gg = v * e.factor, uu = uv * e.factor; \
        float cu = p.cap_u > 0.f ? p.cap_u * precise::tanh(uu / p.cap_u) : uu; \
        a[p.act + (ulong)tg.y * p.inter + j] = p.cap_g * precise::tanh(gg / p.cap_g) / (1.f + exp(-gg)) * cu; \
    } } \
kernel void NAME##_gsum(device float* a [[buffer(0)]], device const IXT* tab [[buffer(1)]], constant IXP& p [[buffer(2)]], \
                        constant uint4& gq [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]], \
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    uint lpr = gq.w, rg = lane / lpr, jj = lane % lpr; \
    uint r = (tg.x * 4 + sg) * (32 / lpr) + rg; \
    device const uint* g = (device const uint*)(a + gq.x); \
    device const uint* asg = (device const uint*)(a + gq.y); \
    if (tg.y >= g[0]) return; \
    uint ge = g[1 + 3 * tg.y], cnt = g[2 + 3 * tg.y], st = g[3 + 3 * tg.y]; \
    IXT e = tab[ge]; \
    T f = MAKE(e, p.inter); \
    uint rc = min(r, p.h - 1); \
    for (uint q = 0; q < cnt; q++) { \
        uint asn = asg[st + q], ti = asn / p.k, s = asn % p.k; \
        float v = ink_lsum(ink_row_acc<T, X, CH>(f, rc, a + p.act + (ulong)asn * p.inter, p.inter / CH, jj, lpr), lpr); \
        if (jj == 0 && r < p.h) \
            a[p.y + (ulong)asn * p.h + r] = v * (a[p.sel + (ulong)ti * 2 * p.k + p.k + s] * e.factor); \
    } } \
kernel void NAME##_dconv(device float* a [[buffer(0)]], device const IXT* tab [[buffer(1)]], constant IXP& p [[buffer(2)]], \
                         constant uint4& gq [[buffer(3)]], device const float* c [[buffer(4)]], constant ICrP& cp [[buffer(5)]], \
                         device float* ring [[buffer(6)]], uint tg [[threadgroup_position_in_grid]], \
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float part[16 * 32]; \
    uint lpr = gq.w, rg = lane / lpr, jj = lane % lpr, nr = 32 / lpr; \
    uint r = tg * nr + rg; \
    if (sg < p.k) { \
        IXT e = tab[as_type<uint>(a[p.sel + sg])]; \
        T f = MAKE(e, p.inter); \
        float v = ink_lsum(ink_row_acc<T, X, CH>(f, min(r, p.h - 1), a + p.act + (ulong)sg * p.inter, p.inter / CH, jj, lpr), lpr); \
        if (jj == 0) part[sg * nr + rg] = v * (a[p.sel + p.k + sg] * e.factor); \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    uint i = tg * nr + lane; \
    if (sg == 0 && lane < nr && i < p.h) { \
        float xin = 0.f; \
        for (uint s = 0; s < p.k; s++) xin += part[s * nr + lane]; \
        ink_conv_in(a, c, cp, ring, i, 0, xin); \
    } }

// ---- Passes of one token: the attention's output projection and the convolution after it in one
// kernel (grid x: blocks of 4 rows; 4 SIMD-groups of 32 lanes, a row each, its chunks c = lane,
// lane + 32, ... summed per lane then over the SIMD-group as the decode library's one-token matvec
// does with one SIMD-group a row, so the same rows); the row is the convolution's input of that
// channel (`ink_conv`'s, parameters `cp`), written to the ring and added to the stream.
struct IMvE { device const uchar* q; device const uchar* s; device const uchar* o; };
#define INK_MVC(NAME, T, X, CH, MAKE) \
kernel void NAME(device const uchar* wb [[buffer(0)]], device const uchar* sb [[buffer(1)]], device float* a [[buffer(2)]], \
                 constant MV& p [[buffer(4)]], device const uchar* ob [[buffer(7)]], device const float* c [[buffer(10)]], \
                 constant ICrP& cp [[buffer(11)]], device float* ring [[buffer(12)]], \
                 uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], \
                 uint lane [[thread_index_in_simdgroup]]) { \
    IMvE e = { wb + p.w_off, sb + p.s_off, ob + p.o_off }; \
    T f = MAKE(e); \
    uint r = tg * 4 + sg; \
    float acc = 0.f; \
    acc += ink_row_acc<T, X, CH>(f, min(r, p.n_out - 1), a + p.ldy, p.n_in / CH, lane, 32); \
    float v = simd_sum(acc); \
    if (lane == 0 && r < p.n_out) ink_conv_in(a, c, cp, ring, r, 0, v); }

// ---- Passes of one token: the decode library's one-token matvec (`mv_body` with NR rows and KS
// SIMD-groups a threadgroup: the same sums) of a message gate's weighted output (`ink_gfg` with
// `ss`), scaled by the gate's norm (`sp`: its partial sums, their count, the output's arena offset,
// eps bits; x at `p.ldy`).
#define INK_MVS(NAME, T, X, CH, MAKE, NR, KS) \
kernel void NAME(device const uchar* wb [[buffer(0)]], device const uchar* sb [[buffer(1)]], device float* a [[buffer(2)]], \
                 constant MV& p [[buffer(4)]], device const uchar* ob [[buffer(7)]], constant uint4& sp [[buffer(11)]], \
                 uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], \
                 uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float red[2 * NR * KS]; \
    IMvE e = { wb + p.w_off, sb + p.s_off, ob + p.o_off }; \
    T f = MAKE(e); \
    float acc[NR], acc2[NR]; \
    for (int r = 0; r < NR; r++) { acc[r] = 0.f; acc2[r] = 0.f; } \
    mv_acc<T, X, CH, NR, KS>(f, f, false, 0, a + p.ldy, p.n_in, p.n_out, tg * NR, sg, lane, acc, acc2, 1.f); \
    for (int r = 0; r < NR; r++) acc[r] = simd_sum(acc[r]); \
    if (KS > 1) { \
        if (lane == 0) \
            for (int r = 0; r < NR; r++) red[sg * 2 * NR + r] = acc[r]; \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        if (sg != 0) return; \
        for (int r = 0; r < NR; r++) { \
            float v = 0.f; \
            for (int s2 = 0; s2 < KS; s2++) v += red[s2 * 2 * NR + r]; \
            acc[r] = v; \
        } \
    } \
    float sc = ink_gscale(a, sp.x, sp.y, p.n_in, as_type<float>(sp.w), lane); \
    uint r0 = tg * NR; \
    if (lane < uint(NR) && r0 + lane < p.n_out) { \
        float v = acc[0]; \
        for (int r = 1; r < NR; r++) if (lane == uint(r)) v = acc[r]; \
        a[sp.z + r0 + lane] = v * sc; \
    } }

// ---- Passes of several tokens: rows of `n` floats at `y` (one a token) scaled by their message
// gate's norm (partials `ss + token * 64`, `np` of them; `ink_mvs_*`'s product). Grid: blocks of 256,
// token.
struct IGsP { uint y; uint n; uint ss; uint np; uint h; float eps; uint _p0; uint _p1; };
kernel void ink_gsc(device float* a [[buffer(0)]], constant IGsP& p [[buffer(1)]], uint2 tg [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint ti = tg.y, i = tg.x * 256 + tid;
    float sc = ink_gscale(a, p.ss + ti * 64, p.np, p.h, p.eps, lane);
    if (i < p.n) a[p.y + (ulong)ti * p.n + i] *= sc;
}
