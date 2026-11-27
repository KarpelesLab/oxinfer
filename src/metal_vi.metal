// Inworld Voice Isolator stream steps (metal_vi.rs): every kernel in f32, the exponentials and
// erf as the CPU computes them (`fast_exp`, the Numerical Recipes erf), so a step stays within
// rounding of the CPU path.

inline float vi_fast_exp(float x) {
    x = clamp(x, -87.0f, 88.0f);
    float fx = rint(x * 1.44269504088896341f);
    float r = (x - fx * 0.693359375f) - fx * -2.12194440e-4f;
    float p = 1.0f + r * (1.0f + r * (0.5f + r * (0.16666667f + r * (0.041666668f + r * (0.008333334f + r * 0.0013888889f)))));
    return as_type<float>(uint((int(fx) + 127) << 23)) * p;
}
inline float vi_silu(float x) { return x / (1.0f + vi_fast_exp(-x)); }
inline float vi_erf(float x) {
    float t = 1.0f / (1.0f + 0.5f * fabs(x));
    float y = t * precise::exp(-x * x - 1.26551223f
        + t * (1.00002368f + t * (0.37409196f + t * (0.09678418f + t * (-0.18628806f
        + t * (0.27886807f + t * (-1.13520398f + t * (1.48851587f + t * (-0.82215223f + t * 0.17087277f)))))))));
    return x >= 0.0f ? 1.0f - y : y - 1.0f;
}
inline float vi_gelu(float x) { return 0.5f * x * (1.0f + vi_erf(x * 0.70710678118654752f)); }
inline float vi_softplus(float x) { return x > 20.0f ? x : precise::log(1.0f + precise::exp(x)); }

// y[g][t][o] = act(sum_k w[g][o][k] x[g][t][k] + bias[g][o]) + res[g][t][o] for t < p.t,
// o < p.n_out. 32 outputs x 32 rows a threadgroup (4 SIMD-groups of 16 x 16), the weights read
// straight from device memory (rows padded to 32, n_in a multiple of 8; rows of x past p.t are
// read, never written).
struct ViG {
    uint n_out; uint n_in; uint t; uint ldx;
    uint ldy; uint ldr; uint act; uint has_bias;
    uint gw; uint gx; uint gy; uint gb;
    uint gr; uint has_res; uint _p0; uint _p1;
};
kernel void vi_gemm(device const float* w [[buffer(0)]], device const float* x [[buffer(1)]], device float* y [[buffer(2)]],
                    device const float* bias [[buffer(3)]], device const float* res [[buffer(4)]], constant ViG& p [[buffer(5)]],
                    uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float tile[32 * 36];
    uint g = tg.z;
    uint lo = (sg / 2) * 16, lt = (sg % 2) * 16;
    device const float* ws = w + (ulong)g * p.gw + (ulong)(tg.x * 32 + lo) * p.n_in;
    device const float* xs = x + (ulong)g * p.gx + (ulong)(tg.y * 32 + lt) * p.ldx;
    simdgroup_float8x8 c00(0.f), c01(0.f), c10(0.f), c11(0.f);
    for (uint k = 0; k < p.n_in; k += 8) {
        simdgroup_float8x8 a0, a1, b0, b1;
        simdgroup_load(a0, ws + k, p.n_in);
        simdgroup_load(a1, ws + (ulong)8 * p.n_in + k, p.n_in);
        simdgroup_load(b0, xs + k, p.ldx, ulong2(0, 0), true);
        simdgroup_load(b1, xs + (ulong)8 * p.ldx + k, p.ldx, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(c00, a0, b0, c00); simdgroup_multiply_accumulate(c01, a0, b1, c01);
        simdgroup_multiply_accumulate(c10, a1, b0, c10); simdgroup_multiply_accumulate(c11, a1, b1, c11);
    }
    // tile[row][out]
    simdgroup_store(c00, tile + lt * 36 + lo, 36, ulong2(0, 0), true);
    simdgroup_store(c01, tile + (lt + 8) * 36 + lo, 36, ulong2(0, 0), true);
    simdgroup_store(c10, tile + lt * 36 + lo + 8, 36, ulong2(0, 0), true);
    simdgroup_store(c11, tile + (lt + 8) * 36 + lo + 8, 36, ulong2(0, 0), true);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < 32 * 32; e += 128) {
        uint r = e / 32, c = e % 32;
        uint t = tg.y * 32 + r, o = tg.x * 32 + c;
        if (t >= p.t || o >= p.n_out) continue;
        float v = tile[r * 36 + c];
        if (p.has_bias != 0) v += bias[(ulong)g * p.gb + o];
        if (p.act == 1) v = vi_gelu(v);
        else if (p.act == 2) v = vi_softplus(v);
        if (p.has_res != 0) v += res[(ulong)g * p.gr + (ulong)t * p.ldr + o];
        y[(ulong)g * p.gy + (ulong)t * p.ldy + o] = v;
    }
}

// The same product for few rows: 16 outputs x 8 * NT rows a threadgroup, the 4 SIMD-groups
// splitting K (8-wide steps s, s + 4, ...), their partial sums added in a fixed order. More
// threadgroups than vi_gemm (the matrices of a 20 ms step have 92 rows), each weight read
// ceil(rows / (8 NT)) times. n_in a multiple of 8. (On an M5 Max, 92 rows: NT = 4 beats NT = 8,
// vi_gemm and 64 x 64 tiles staging x in threadgroup memory: 3.6-4.0, 5.6-6.0, 3.9-4.3 and
// 10.2-10.4 ms a step.)
#define VI_PLD 20
template <int NT>
inline void vi_gemm_k(device const float* w, device const float* x, device float* y, device const float* bias,
                      device const float* res, constant ViG& p, uint3 tg, uint sg, uint tid, threadgroup float* part) {
    uint g = tg.z;
    uint o0 = tg.x * 16, t0 = tg.y * (8 * NT);
    device const float* ws = w + (ulong)g * p.gw + (ulong)o0 * p.n_in;
    device const float* xs = x + (ulong)g * p.gx + (ulong)t0 * p.ldx;
    simdgroup_float8x8 c0[NT], c1[NT];
    for (int t = 0; t < NT; t++) { c0[t] = simdgroup_float8x8(0.f); c1[t] = simdgroup_float8x8(0.f); }
    for (uint k = sg * 8; k < p.n_in; k += 32) {
        simdgroup_float8x8 a0, a1;
        simdgroup_load(a0, ws + k, p.n_in);
        simdgroup_load(a1, ws + (ulong)8 * p.n_in + k, p.n_in);
        for (int t = 0; t < NT; t++) {
            simdgroup_float8x8 b;
            simdgroup_load(b, xs + (ulong)(8 * t) * p.ldx + k, p.ldx, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(c0[t], a0, b, c0[t]);
            simdgroup_multiply_accumulate(c1[t], a1, b, c1[t]);
        }
    }
    // part[sg][row][out]
    threadgroup float* ps = part + sg * (8 * NT) * VI_PLD;
    for (int t = 0; t < NT; t++) {
        simdgroup_store(c0[t], ps + (8 * t) * VI_PLD, VI_PLD, ulong2(0, 0), true);
        simdgroup_store(c1[t], ps + (8 * t) * VI_PLD + 8, VI_PLD, ulong2(0, 0), true);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < 16 * 8 * NT; e += 128) {
        uint r = e / 16, c = e % 16;
        uint t = t0 + r, o = o0 + c;
        if (t >= p.t || o >= p.n_out) continue;
        uint at = r * VI_PLD + c, st = (8 * NT) * VI_PLD;
        float v = ((part[at] + part[at + st]) + part[at + 2 * st]) + part[at + 3 * st];
        if (p.has_bias != 0) v += bias[(ulong)g * p.gb + o];
        if (p.act == 1) v = vi_gelu(v);
        else if (p.act == 2) v = vi_softplus(v);
        if (p.has_res != 0) v += res[(ulong)g * p.gr + (ulong)t * p.ldr + o];
        y[(ulong)g * p.gy + (ulong)t * p.ldy + o] = v;
    }
}
#define VI_GEMM_K(NAME, NT) \
kernel void NAME(device const float* w [[buffer(0)]], device const float* x [[buffer(1)]], device float* y [[buffer(2)]], \
                 device const float* bias [[buffer(3)]], device const float* res [[buffer(4)]], constant ViG& p [[buffer(5)]], \
                 uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], \
                 uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup float part[4 * 8 * NT * VI_PLD]; \
    vi_gemm_k<NT>(w, x, y, bias, res, p, tg, sg, tid, part); }
VI_GEMM_K(vi_gemm_k1, 1) VI_GEMM_K(vi_gemm_k4, 4)

// Reduction of one value over a threadgroup of 128 threads (4 SIMD-groups).
inline float vi_sum(float v, threadgroup float* red, uint tid) {
    v = simd_sum(v);
    if (tid % 32 == 0) red[tid / 32] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = red[0] + red[1] + red[2] + red[3];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return s;
}
inline float vi_max(float v, threadgroup float* red, uint tid) {
    v = simd_max(v);
    if (tid % 32 == 0) red[tid / 32] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = max(max(red[0], red[1]), max(red[2], red[3]));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return s;
}

// LayerNorm (eps 1e-5) of row r of x (n wide), weights of group r / rpg (gw apart), optionally
// after a FiLM (x * scale + shift); mode 0 stores, 1 adds into y. 128 threads a row.
struct ViLn { uint n; uint ldx; uint ldy; uint mode; uint rpg; uint gw; uint film; uint _p; };
kernel void vi_ln(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], device const float* wt [[buffer(2)]],
                  device const float* bs [[buffer(3)]], device const float* sc [[buffer(4)]], device const float* sh [[buffer(5)]],
                  constant ViLn& p [[buffer(6)]], uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float red[4];
    device const float* xr = x + (ulong)r * p.ldx;
    device float* yr = y + (ulong)r * p.ldy;
    uint g = r / p.rpg;
    device const float* ww = wt + (ulong)g * p.gw;
    device const float* bb = bs + (ulong)g * p.gw;
    float s = 0.f;
    for (uint i = tid; i < p.n; i += 128) s += p.film != 0 ? xr[i] * sc[i] + sh[i] : xr[i];
    float mean = vi_sum(s, red, tid) / float(p.n);
    float q = 0.f;
    for (uint i = tid; i < p.n; i += 128) {
        float v = (p.film != 0 ? xr[i] * sc[i] + sh[i] : xr[i]) - mean;
        q += v * v;
    }
    float inv = 1.0f / sqrt(vi_sum(q, red, tid) / float(p.n) + 1e-5f);
    for (uint i = tid; i < p.n; i += 128) {
        float v = p.film != 0 ? xr[i] * sc[i] + sh[i] : xr[i];
        float o = (v - mean) * inv * ww[i] + bb[i];
        if (p.mode == 1) yr[i] += o; else yr[i] = o;
    }
}

// The encoder's inputs: row (band b, frame f) = the LayerNorm of band b's bins of frame f's
// spectrum (re, im pairs), zero-padded to ld. bands[b] = (first bin, end bin, norm offset).
struct ViEnc { uint nf; uint bins; uint ld; uint _p; };
kernel void vi_enc_in(device const float* spec [[buffer(0)]], device float* y [[buffer(1)]], device const float* wt [[buffer(2)]],
                      device const uint4* bands [[buffer(3)]], constant ViEnc& p [[buffer(4)]],
                      uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float red[4];
    uint b = r / p.nf, f = r % p.nf;
    uint4 bd = bands[b];
    uint n = 2 * (bd.y - bd.x);
    device const float* xr = spec + ((ulong)f * p.bins + bd.x) * 2;
    device const float* ww = wt + bd.z;
    device const float* bb = wt + bd.z + n;
    device float* yr = y + (ulong)r * p.ld;
    float s = 0.f;
    for (uint i = tid; i < n; i += 128) s += xr[i];
    float mean = vi_sum(s, red, tid) / float(n);
    float q = 0.f;
    for (uint i = tid; i < n; i += 128) { float v = xr[i] - mean; q += v * v; }
    float inv = 1.0f / sqrt(vi_sum(q, red, tid) / float(n) + 1e-5f);
    for (uint i = tid; i < p.ld; i += 128) yr[i] = i < n ? (xr[i] - mean) * inv * ww[i] + bb[i] : 0.f;
}

// The decoder's last layer to the spectrum layout: raw[f][s + j / 4][j % 4] = y[b][f][j].
kernel void vi_dec_out(device const float* y [[buffer(0)]], device float* raw [[buffer(1)]], device const uint4* bands [[buffer(2)]],
                       constant ViEnc& p [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    uint r = tg.x, b = r / p.nf, f = r % p.nf;
    uint4 bd = bands[b];
    uint n = 4 * (bd.y - bd.x);
    for (uint j = tid; j < n; j += 128) raw[((ulong)f * p.bins + bd.x) * 4 + j] = y[(ulong)r * p.ld + j];
}

// The time attention's queries rotated to their slots (into q), keys rotated and values stored
// in the cache ([rows][heads][cap][hd]) at slot first + f. One thread per (token, head, i < hd / 2).
struct ViRope { uint nf; uint w; uint nh; uint hd; uint cap; uint first; uint n; uint _p; };
kernel void vi_rope_kv(device const float* qkv [[buffer(0)]], device float* q [[buffer(1)]], device float* kc [[buffer(2)]],
                       device float* vc [[buffer(3)]], device const float* cs [[buffer(4)]], device const float* sn [[buffer(5)]],
                       constant ViRope& p [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
    uint half_ = p.hd / 2;
    if (gid >= p.n * p.nh * half_) return;
    uint i = gid % half_, h = (gid / half_) % p.nh, r = gid / (half_ * p.nh);
    uint b = r / p.nf, f = r % p.nf, slot = p.first + f;
    device const float* src = qkv + (ulong)r * 3 * p.w + h * p.hd;
    float c = cs[(ulong)slot * half_ + i], s = sn[(ulong)slot * half_ + i];
    float x1 = src[i], x2 = src[i + half_];
    q[(ulong)r * p.w + h * p.hd + i] = x1 * c - x2 * s;
    q[(ulong)r * p.w + h * p.hd + i + half_] = x1 * s + x2 * c;
    ulong at = (((ulong)b * p.nh + h) * p.cap + slot) * p.hd;
    x1 = src[p.w + i]; x2 = src[p.w + i + half_];
    kc[at + i] = x1 * c - x2 * s;
    kc[at + i + half_] = x1 * s + x2 * c;
    vc[at + i] = src[2 * p.w + i];
    vc[at + i + half_] = src[2 * p.w + i + half_];
}

// Softmax attention of one query a threadgroup (128 threads), its output's component along its
// own value removed when p.excl. mode 0: the time attention (query (row b, head h, frame f)
// against cache slots lo..first + f); mode 1: across the bands of a frame (query (band b, head
// h, frame f) against the keys of every band in qkv).
struct ViAtt {
    uint mode; uint nf; uint nb; uint w;
    uint nh; uint hd; uint cap; uint first;
    uint past; uint window; uint excl; float scale;
};
#define VI_SCMAX 4096
kernel void vi_attn(device const float* qkv [[buffer(0)]], device const float* qr [[buffer(1)]], device const float* kc [[buffer(2)]],
                    device const float* vc [[buffer(3)]], device float* out [[buffer(4)]], constant ViAtt& p [[buffer(5)]],
                    uint qi [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sc[VI_SCMAX];
    threadgroup float red[4];
    threadgroup float part[512];
    uint hd = p.hd;
    uint f = qi % p.nf, h = (qi / p.nf) % p.nh, b = qi / (p.nf * p.nh);
    uint r = b * p.nf + f;
    device const float* q;
    device const float* keys;
    device const float* vals;
    ulong stride;
    uint cnt;
    if (p.mode == 0) {
        q = qr + (ulong)r * p.w + h * hd;
        uint qs = p.first + f;
        uint lo = p.first - p.past;
        if (p.window > 0 && qs + 1 > p.window) lo = max(lo, qs + 1 - p.window);
        cnt = qs + 1 - lo;
        ulong base = (((ulong)b * p.nh + h) * p.cap + lo) * hd;
        keys = kc + base;
        vals = vc + base;
        stride = hd;
    } else {
        q = qkv + (ulong)r * 3 * p.w + h * hd;
        keys = qkv + (ulong)f * 3 * p.w + p.w + h * hd;
        vals = keys + p.w;
        stride = (ulong)p.nf * 3 * p.w;
        cnt = p.nb;
    }
    float m = -INFINITY;
    for (uint j = tid; j < cnt; j += 128) {
        device const float4* k4 = (device const float4*)(keys + j * stride);
        device const float4* q4 = (device const float4*)q;
        float4 a = 0.f;
        for (uint d = 0; d < hd / 4; d++) a += q4[d] * k4[d];
        float s = (a.x + a.y + a.z + a.w) * p.scale;
        sc[j] = s;
        m = max(m, s);
    }
    m = vi_max(m, red, tid);
    float sum = 0.f;
    for (uint j = tid; j < cnt; j += 128) {
        float e = vi_fast_exp(sc[j] - m);
        sc[j] = e;
        sum += e;
    }
    float inv = 1.0f / vi_sum(sum, red, tid);
    // SIMD-group s takes keys s, s + 4, ..., its lane `hd / 32` dimensions; the four partial
    // sums added in order.
    uint dpl = hd / 32, sgi = tid / 32, lane = tid % 32;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (uint j = sgi; j < cnt; j += 4) {
        float pj = sc[j] * inv;
        device const float* vr = vals + j * stride + lane * dpl;
        for (uint q = 0; q < dpl; q++) acc[q] += pj * vr[q];
    }
    for (uint q = 0; q < dpl; q++) part[sgi * hd + lane * dpl + q] = acc[q];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float o_own = 0.f;
    if (tid < hd) o_own = ((part[tid] + part[hd + tid]) + part[2 * hd + tid]) + part[3 * hd + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < hd) part[tid] = o_own;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float* dst = out + (ulong)r * p.w + h * hd;
    if (p.excl != 0) {
        device const float* v = qkv + (ulong)r * 3 * p.w + 2 * p.w + h * hd;
        float ov = 0.f, vv = 0.f;
        for (uint i = tid; i < hd; i += 128) { ov += part[i] * v[i]; vv += v[i] * v[i]; }
        ov = vi_sum(ov, red, tid);
        vv = vi_sum(vv, red, tid);
        float c = ov / max(vv, 1e-6f);
        for (uint i = tid; i < hd; i += 128) dst[i] = part[i] - c * v[i];
    } else {
        for (uint i = tid; i < hd; i += 128) dst[i] = part[i];
    }
}

// Mamba. Token (sequence s, position t) is row s * ss + t * ps.
struct ViMb {
    uint inner; uint k; uint len; uint seqs;
    uint ss; uint ps; uint reverse; uint has_tail;
    uint ns; uint dt_rank; uint xw; uint _p;
};
// u = silu(conv_b + causal depthwise conv of the first half of xz, in scan order), the carried
// tail (the last k - 1 inputs) before the first. One thread per (sequence, position, channel).
kernel void vi_conv(device const float* xz [[buffer(0)]], device float* u [[buffer(1)]], device const float* cw [[buffer(2)]],
                    device const float* cb [[buffer(3)]], device const float* tail [[buffer(4)]], constant ViMb& p [[buffer(5)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.seqs * p.len * p.inner) return;
    uint c = gid % p.inner, step = (gid / p.inner) % p.len, s = gid / (p.inner * p.len);
    uint t = p.reverse != 0 ? p.len - 1 - step : step;
    float acc = cb[c];
    for (uint j = 0; j < p.k; j++) {
        int si = int(step) + int(j) - int(p.k - 1);
        float v;
        if (si >= 0) {
            uint ts = p.reverse != 0 ? p.len - 1 - uint(si) : uint(si);
            v = xz[(ulong)(s * p.ss + ts * p.ps) * 2 * p.inner + c];
        } else if (p.has_tail != 0) {
            v = tail[((ulong)s * (p.k - 1) + uint(int(p.k - 1) + si)) * p.inner + c];
        } else {
            continue;
        }
        acc += cw[j * p.inner + c] * v;
    }
    u[(ulong)(s * p.ss + t * p.ps) * p.inner + c] = vi_silu(acc);
}
// The carried tail after this call into tail_out: the last k - 1 inputs of (old tail, this
// call's inputs). One thread per (sequence, channel).
kernel void vi_tail(device const float* xz [[buffer(0)]], device const float* tail [[buffer(1)]], constant ViMb& p [[buffer(2)]],
                    device float* tail_out [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= p.seqs * p.inner) return;
    uint c = gid % p.inner, s = gid / p.inner;
    uint km = p.k - 1;
    float nt[8];
    for (uint ti = 0; ti < km; ti++) {
        uint from_end = km - ti;
        if (from_end <= p.len) {
            uint t = p.len - from_end;
            nt[ti] = xz[(ulong)(s * p.ss + t * p.ps) * 2 * p.inner + c];
        } else {
            uint oi = km - (from_end - p.len);
            nt[ti] = tail[((ulong)s * km + oi) * p.inner + c];
        }
    }
    for (uint ti = 0; ti < km; ti++) tail_out[((ulong)s * km + ti) * p.inner + c] = nt[ti];
}
// The selective scan (16 states): h = exp(delta a) h + delta b u, y = (c . h + d u) silu(z), from
// the carried state into state_out when given. One thread per (sequence, channel).
kernel void vi_scan(device const float* u [[buffer(0)]], device const float* delta [[buffer(1)]], device const float* xp [[buffer(2)]],
                    device const float* xz [[buffer(3)]], device const float* a [[buffer(4)]], device const float* dd [[buffer(5)]],
                    device float* y [[buffer(6)]], device const float* state [[buffer(7)]], constant ViMb& p [[buffer(8)]],
                    device float* state_out [[buffer(9)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= p.seqs * p.inner) return;
    uint c = gid % p.inner, s = gid / p.inner;
    float h[16];
    float av[16];
    for (uint n = 0; n < 16; n++) {
        h[n] = p.has_tail != 0 ? state[((ulong)s * p.inner + c) * 16 + n] : 0.f;
        av[n] = a[c * 16 + n];
    }
    float dc = dd[c];
    for (uint step = 0; step < p.len; step++) {
        uint t = p.reverse != 0 ? p.len - 1 - step : step;
        ulong tok = s * p.ss + t * p.ps;
        float dl = delta[tok * p.inner + c];
        float x = u[tok * p.inner + c];
        float drive = dl * x;
        device const float* bv = xp + tok * p.xw + p.dt_rank;
        device const float* cv = bv + 16;
        float acc = 0.f;
        for (uint n = 0; n < 16; n++) {
            h[n] = vi_fast_exp(dl * av[n]) * h[n] + drive * bv[n];
            acc += h[n] * cv[n];
        }
        float o = acc + x * dc;
        float g = xz[tok * 2 * p.inner + p.inner + c];
        y[tok * p.inner + c] = o * vi_silu(g);
    }
    if (p.has_tail != 0)
        for (uint n = 0; n < 16; n++) state_out[((ulong)s * p.inner + c) * 16 + n] = h[n];
}

// dst = src (n floats).
kernel void vi_copy(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]], constant uint& n [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid < n) dst[gid] = src[gid];
}
