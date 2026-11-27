// ---- LTX-2.5 transformer passes (metal_ltx.rs, crate::gpu::dit::LtxSession) ----
// Rows of one stream carry a timestep class (`cls`): the modulation tables hold one set per class
// (`ld_tab` floats apart), a row reads its class's.

// y = (norm ? rms(x) : x) * (1 + tab[cls][sc]) + tab[cls][sh] over rows of n (x rows ld_x apart;
// `uniform`: class 0 for every row), into f32 rows of n or scaled half rows (`out_half`, `rs` the
// inverse scales). 256 threads per row.
struct LRM { uint n; uint ld_x; uint norm; uint out_half; float eps; uint ld_tab; uint sc_off; uint sh_off; uint uniform_cls; uint _p0; uint _p1; uint _p2; };
kernel void ltx_rms_mod(device const float* x [[buffer(0)]], device const float* tab [[buffer(1)]], device const uint* cls [[buffer(2)]],
                        device void* out [[buffer(3)]], device float* rs [[buffer(4)]], constant LRM& p [[buffer(5)]],
                        uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8];
    threadgroup float red2[8];
    device const float4* row = (device const float4*)(x + (ulong)r * p.ld_x);
    uint c = p.uniform_cls != 0 ? 0 : cls[r];
    device const float4* sc4 = (device const float4*)(tab + (ulong)c * p.ld_tab + p.sc_off);
    device const float4* sh4 = (device const float4*)(tab + (ulong)c * p.ld_tab + p.sh_off);
    uint n4 = p.n / 4;
    float inv = 1.0f;
    if (p.norm != 0) {
        float ss = 0.f;
        for (uint i = tid; i < n4; i += 256) { float4 v = row[i]; ss += dot(v, v); }
        ss = dit_tg_sum(ss, sg, lane, red);
        inv = 1.0f / sqrt(ss / float(p.n) + p.eps);
    }
#define LTX_RM(i) (row[i] * inv * (1.0f + sc4[i]) + sh4[i])
    if (p.out_half == 0) {
        device float4* o = (device float4*)((device float*)out + (ulong)r * p.n);
        for (uint i = tid; i < n4; i += 256) o[i] = LTX_RM(i);
        return;
    }
    float m = 0.f;
    for (uint i = tid; i < n4; i += 256) { float4 v = abs(LTX_RM(i)); m = max(m, max(max(v.x, v.y), max(v.z, v.w))); }
    m = dit_tg_max(m, sg, lane, red2);
    float sinv, s = dit_row_scale(m, sinv);
    device half4* o = (device half4*)((device half*)out + (ulong)r * p.n);
    for (uint i = tid; i < n4; i += 256) o[i] = half4(LTX_RM(i) * s);
    if (tid == 0) rs[r] = sinv;
#undef LTX_RM
}

// Attention operands: row r of src (ld_src apart) plus bias, RMS-normed over all n values with
// weight w (`norm`), each head's halves rotated by the row's angles (`rope`: cos / sin tables of
// n / 2 per row, head h at h * hd / 2), into dst rows of n (half or f32). Rows from n_valid on are
// zeros (padding keys and values). 256 threads per row.
struct LHD { uint n; uint ld_src; uint hd; uint norm; uint rope; uint out_half; float eps; uint n_valid; };
kernel void ltx_heads(device const float* src [[buffer(0)]], device void* dst [[buffer(1)]], device const float* bias [[buffer(2)]],
                      device const float* w [[buffer(3)]], device const float* cs [[buffer(4)]], device const float* sn [[buffer(5)]],
                      constant LHD& p [[buffer(6)]],
                      uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8];
    device const float* row = src + (ulong)r * p.ld_src;
    device half* dh = (device half*)dst + (ulong)r * p.n;
    device float* df = (device float*)dst + (ulong)r * p.n;
    if (r >= p.n_valid) {
        for (uint i = tid; i < p.n; i += 256) {
            if (p.out_half != 0) dh[i] = 0.0h; else df[i] = 0.f;
        }
        return;
    }
    float inv = 1.0f;
    if (p.norm != 0) {
        float ss = 0.f;
        for (uint i = tid; i < p.n; i += 256) { float v = row[i] + bias[i]; ss += v * v; }
        ss = dit_tg_sum(ss, sg, lane, red);
        inv = 1.0f / sqrt(ss / float(p.n) + p.eps);
    }
    uint h2 = p.hd / 2;
    device const float* c = cs + (ulong)r * (p.n / 2);
    device const float* s = sn + (ulong)r * (p.n / 2);
    for (uint j = tid; j < p.n / 2; j += 256) {
        uint h = j / h2, i = j % h2;
        uint a = h * p.hd + i, b = a + h2;
        float va = row[a] + bias[a], vb = row[b] + bias[b];
        if (p.norm != 0) { va *= inv * w[a]; vb *= inv * w[b]; }
        if (p.rope != 0) {
            float cc = c[j], ss2 = s[j];
            float x = va * cc - ss2 * vb, y = vb * cc + ss2 * va;
            va = x; vb = y;
        }
        if (p.out_half != 0) { dh[a] = half(va); dh[b] = half(vb); } else { df[a] = va; df[b] = vb; }
    }
}

// Gate logits of a gated attention: g[r][h] = 2 sigmoid(dot(x[r], W[h]) + b[h]) for the `heads`
// rows of W (n wide; x rows ld_x apart). One threadgroup (8 SIMD-groups) per row.
struct LGL { uint n; uint ld_x; uint heads; uint _p; };
kernel void ltx_gate_logits(device const float* x [[buffer(0)]], device const float* wt [[buffer(1)]], device const float* b [[buffer(2)]],
                            device float* g [[buffer(3)]], constant LGL& p [[buffer(4)]],
                            uint r [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    device const float4* xr = (device const float4*)(x + (ulong)r * p.ld_x);
    uint n4 = p.n / 4;
    for (uint h = sg; h < p.heads; h += 8) {
        device const float4* wr = (device const float4*)(wt + (ulong)h * p.n);
        float s = 0.f;
        for (uint i = lane; i < n4; i += 32) s += dot(xr[i], wr[i]);
        s = simd_sum(s);
        if (lane == 0) g[(ulong)r * p.heads + h] = 2.0f / (1.0f + exp(-(s + b[h])));
    }
}

// att[r][h * hd + i] *= g[r][h] over rows of inner (= heads * hd).
struct LGA { uint inner; uint hd; uint heads; uint _p; };
kernel void ltx_gate_apply(device float* att [[buffer(0)]], device const float* g [[buffer(1)]], constant LGA& p [[buffer(2)]],
                           uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    uint c = tg.x * 256 + tid, r = tg.y;
    if (c >= p.inner) return;
    att[(ulong)r * p.inner + c] *= g[(ulong)r * p.heads + c / p.hd];
}

// h[r][c] += (y[r][c] + b[c]) * gate, gate = tab[cls[r]][off + c] (or tab[off + c] with
// `uniform_cls`), over rows of n (y rows ld_y apart).
struct LGD { uint n; uint ld_y; uint ld_tab; uint off; uint uniform_cls; uint has_bias; uint _p0; uint _p1; };
kernel void ltx_gated_add(device float* h [[buffer(0)]], device const float* y [[buffer(1)]], device const float* b [[buffer(2)]],
                          device const float* tab [[buffer(3)]], device const uint* cls [[buffer(4)]], constant LGD& p [[buffer(5)]],
                          uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    uint c = (tg.x * 256 + tid) * 4, r = tg.y;
    if (c >= p.n) return;
    uint k = p.uniform_cls != 0 ? 0 : cls[r];
    float4 g = *(device const float4*)(tab + (ulong)k * p.ld_tab + p.off + c);
    float4 yv = *(device const float4*)(y + (ulong)r * p.ld_y + c);
    if (p.has_bias != 0) yv += *(device const float4*)(b + c);
    *(device float4*)(h + (ulong)r * p.n + c) += yv * g;
}

// x[r][c] = gelu_tanh(x[r][c] + b[c]) in place over rows of n.
struct LGE { uint n; uint has_bias; uint _p0; uint _p1; };
kernel void ltx_gelu(device float* x [[buffer(0)]], device const float* b [[buffer(1)]], constant LGE& p [[buffer(2)]],
                     uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    uint c = (tg.x * 256 + tid) * 4, r = tg.y;
    if (c >= p.n) return;
    device float4* xp = (device float4*)(x + (ulong)r * p.n + c);
    float4 v = *xp;
    if (p.has_bias != 0) v += *(device const float4*)(b + c);
    float4 u = 1.5957691216057308f * (v + 0.044715f * v * v * v);
    *xp = v / (1.0f + exp(-u));
}

// ---- LTX-2.5 DiffVAE: 3D neighborhood attention (natten.na3d), 64-wide heads ----
// Queries over a [qt, qh, qw] grid attend to the [kt_, kh_, KW] window of keys / values over a
// [kt, kh, kw] grid centered at the query's index plus (ot, oh, ow), shifted inward at the
// borders (queries already scaled). Rows of `dim` floats, head h at h * 64. One SIMD-group per
// threadgroup: 32 consecutive queries of one (t, h) row and one head; each key row segment they
// need is staged in threadgroup memory.
struct NA3 { uint qt; uint qh; uint qw; uint kt; uint kh; uint kw; uint ot; uint oh; uint ow; uint kt_; uint kh_; uint heads; uint dim; uint row0; uint wtiles; uint _p2; };
static inline uint na_start(uint i, uint off, uint k, uint len) {
    int s = int(i + off) - int(k / 2);
    return uint(clamp(s, 0, int(len - k)));
}
template <uint KW>
static inline void ltx_na3d_impl(device const float* q, device const float* k, device const float* v, device float* o,
                                 constant NA3& p, uint3 tg, uint lane, threadgroup float4* ks, threadgroup float4* vs) {
    uint row = tg.y + p.row0, head = tg.x / p.wtiles;
    uint w0 = (tg.x % p.wtiles) * 32, t = row / p.qh, h = row % p.qh;
    uint w = w0 + lane;
    bool valid = w < p.qw;
    uint wc = min(w, p.qw - 1), wl = min(w0 + 31, p.qw - 1);
    uint st = na_start(t, p.ot, p.kt_, p.kt), sh = na_start(h, p.oh, p.kh_, p.kh);
    uint sw = na_start(wc, p.ow, KW, p.kw);
    uint u0 = na_start(w0, p.ow, KW, p.kw), u1 = na_start(wl, p.ow, KW, p.kw) + KW;
    uint nu = u1 - u0;
    ulong qi = ((ulong)t * p.qh + h) * p.qw + wc;
    device const float4* qr = (device const float4*)(q + qi * p.dim + head * 64);
    float4 qv[16];
    float4 acc[16];
    for (uint i = 0; i < 16; i++) { qv[i] = qr[i]; acc[i] = float4(0.f); }
    float m = -INFINITY, l = 0.f;
    for (uint a = 0; a < p.kt_; a++) {
        for (uint b = 0; b < p.kh_; b++) {
            ulong base = (((ulong)(st + a) * p.kh + sh + b) * p.kw + u0) * p.dim + head * 64;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = lane; i < nu * 16; i += 32) {
                uint j = i / 16, c = i % 16;
                ks[i] = *((device const float4*)(k + base + (ulong)j * p.dim) + c);
                vs[i] = *((device const float4*)(v + base + (ulong)j * p.dim) + c);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float s[KW];
            float mx = m;
            uint j0 = sw - u0;
            for (uint c = 0; c < KW; c++) {
                threadgroup const float4* kr = ks + (j0 + c) * 16;
                float4 d4 = qv[0] * kr[0];
                for (uint i = 1; i < 16; i++) d4 = fma(qv[i], kr[i], d4);
                s[c] = d4.x + d4.y + d4.z + d4.w;
                mx = max(mx, s[c]);
            }
            float sc = exp(m - mx);
            l *= sc;
            for (uint i = 0; i < 16; i++) acc[i] *= sc;
            for (uint c = 0; c < KW; c++) {
                float pc = exp(s[c] - mx);
                l += pc;
                threadgroup const float4* vr = vs + (j0 + c) * 16;
                for (uint i = 0; i < 16; i++) acc[i] = fma(float4(pc), vr[i], acc[i]);
            }
            m = mx;
        }
    }
    if (!valid) return;
    device float4* orow = (device float4*)(o + qi * p.dim + head * 64);
    float inv = 1.f / l;
    for (uint i = 0; i < 16; i++) orow[i] = acc[i] * inv;
}
#define LTX_NA3D(KW)                                                                                                     \
    kernel void ltx_na3d_k##KW(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]],                \
                               device const float* v [[buffer(2)]], device float* o [[buffer(3)]],                      \
                               constant NA3& p [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],                \
                               uint lane [[thread_index_in_threadgroup]]) {                                             \
        threadgroup float4 ks[(32 + KW - 1) * 16];                                                                       \
        threadgroup float4 vs[(32 + KW - 1) * 16];                                                                       \
        ltx_na3d_impl<KW>(q, k, v, o, p, tg, lane, ks, vs);                                                              \
    }
LTX_NA3D(3)
LTX_NA3D(5)
LTX_NA3D(7)
LTX_NA3D(11)

// ---- LTX-2.5 DiffVAE block session (metal_diffvae.rs) ----
// Fused q / k / v rows (f32, `3 dim` wide, rows r0.. of the grid, biases not yet added) to the
// attention's half operands: per 64-wide head the RMS-normed queries (weight qn, times `scale`)
// and keys (weight kn), rotated by absolute RoPE (adjacent pairs: the first pt pairs by time, the
// next ph by height, the rest by width; tables at tab + o* of cos then sin per position), and the
// values. Keys and values go to a grid `kw` wide (the query's column plus `halo`); the edge columns
// are also written to the halo columns past their edge, rotated at those positions. One SIMD-group
// per head (lane = pair), 256 threads per row.
struct DQH { uint dim; uint heads; uint qh; uint qw; uint kw; uint halo; uint r0; uint pt; uint ph;
             uint ot; uint oh; uint ow; float scale; float eps; uint nt; uint nh; uint nw; uint _p0; uint _p1; uint _p2; };
static inline float2 dv_rot(float2 x, device const float* tab, uint n, uint idx) {
    float c = tab[idx], s = tab[n + idx];
    return float2(x.x * c - x.y * s, x.x * s + x.y * c);
}
kernel void dv_qkv_heads(device const float* src [[buffer(0)]], device const float* bias [[buffer(1)]],
                         device const float* qn [[buffer(2)]], device const float* kn [[buffer(3)]],
                         device const float* tab [[buffer(4)]], device half* qo [[buffer(5)]],
                         device half* ko [[buffer(6)]], device half* vo [[buffer(7)]], constant DQH& p [[buffer(8)]],
                         uint ri [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    uint r = p.r0 + ri;
    uint w = r % p.qw, h = (r / p.qw) % p.qh, t = r / (p.qw * p.qh);
    device const float* row = src + (ulong)ri * 3 * p.dim;
    uint pw = 32 - p.pt - p.ph;
    // The pair's axis: its table (n cosines, then n sines) and position index.
    device const float* tb;
    uint n, base, pi;
    bool wax = false;
    if (lane < p.pt) { tb = tab + p.ot; n = p.nt * p.pt; base = t * p.pt; pi = lane; }
    else if (lane < p.pt + p.ph) { tb = tab + p.oh; n = p.nh * p.ph; base = h * p.ph; pi = lane - p.pt; }
    else { tb = tab + p.ow; n = p.nw * pw; base = (w + p.halo) * pw; pi = lane - p.pt - p.ph; wax = true; }
    ulong kr0 = ((ulong)t * p.qh + h) * p.kw;
    ulong kr = kr0 + w + p.halo;
    for (uint hd = sg; hd < p.heads; hd += 8) {
        uint c = hd * 64 + 2 * lane;
        float2 q = float2(row[c] + bias[c], row[c + 1] + bias[c + 1]);
        float2 k = float2(row[p.dim + c] + bias[p.dim + c], row[p.dim + c + 1] + bias[p.dim + c + 1]);
        float2 v = float2(row[2 * p.dim + c] + bias[2 * p.dim + c], row[2 * p.dim + c + 1] + bias[2 * p.dim + c + 1]);
        float sq = simd_sum(q.x * q.x + q.y * q.y), sk = simd_sum(k.x * k.x + k.y * k.y);
        q *= rsqrt(sq / 64.f + p.eps) * p.scale * float2(qn[2 * lane], qn[2 * lane + 1]);
        k *= rsqrt(sk / 64.f + p.eps) * float2(kn[2 * lane], kn[2 * lane + 1]);
        q = dv_rot(q, tb, n, base + pi);
        float2 kq = dv_rot(k, tb, n, base + pi);
        ulong o = (ulong)r * p.dim + c;
        qo[o] = half(q.x); qo[o + 1] = half(q.y);
        ulong ok = kr * p.dim + c;
        ko[ok] = half(kq.x); ko[ok + 1] = half(kq.y);
        vo[ok] = half(v.x); vo[ok + 1] = half(v.y);
        if (p.halo > 0 && (w == 0 || w == p.qw - 1)) {
            for (uint j = 0; j < p.halo; j++) {
                uint col = w == 0 ? j : w + p.halo + 1 + j;
                float2 kk = wax ? dv_rot(k, tb, n, col * pw + pi) : kq;
                ulong oc = (kr0 + col) * p.dim + c;
                ko[oc] = half(kk.x); ko[oc + 1] = half(kk.y);
                vo[oc] = half(v.x); vo[oc + 1] = half(v.y);
            }
        }
    }
}

// SwiGLU's middle: rows of silu(g) * u (f32 rows of f) to scaled half rows. 256 threads per row.
kernel void dv_silu_mul(device const float* g [[buffer(0)]], device const float* u [[buffer(1)]], device half* out [[buffer(2)]],
                        device float* rs [[buffer(3)]], constant uint& f [[buffer(4)]],
                        uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8];
    device const float4* g4 = (device const float4*)(g + (ulong)r * f);
    device const float4* u4 = (device const float4*)(u + (ulong)r * f);
    uint n4 = f / 4;
#define DV_SM(i) (g4[i] / (1.0f + exp(-g4[i])) * u4[i])
    float m = 0.f;
    for (uint i = tid; i < n4; i += 256) { float4 v = abs(DV_SM(i)); m = max(m, max(max(v.x, v.y), max(v.z, v.w))); }
    m = dit_tg_max(m, sg, lane, red);
    float sinv, s = dit_row_scale(m, sinv);
    device half4* o = (device half4*)(out + (ulong)r * f);
    for (uint i = tid; i < n4; i += 256) o[i] = half4(DV_SM(i) * s);
    if (tid == 0) rs[r] = sinv;
#undef DV_SM
}

// h[r][c] += b[c] over rows of n.
kernel void dv_add_bias(device float* h [[buffer(0)]], device const float* b [[buffer(1)]], constant uint& n [[buffer(2)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    uint c = tg.x * 256 + tid;
    if (c >= n) return;
    h[(ulong)tg.y * n + c] += b[c];
}

// 3D neighborhood attention over half operands (see ltx_na3d_k*; keys over the same frames and
// rows, `ow` extra columns before the queries'), the output as half rows (the next GEMM's input,
// unscaled). Threadgroups of 4 SIMD-groups: 32 consecutive queries of 4 consecutive rows (one
// frame) and one head; each key row any of them needs is staged once.
struct DNA { uint qt; uint qh; uint qw; uint kt; uint kh; uint kw; uint ow; uint kt_; uint kh_; uint heads; uint dim; uint hblocks;
             uint row0; uint wtiles; uint _p0; uint _p1; };
template <uint KW>
static inline void dv_na3d_impl(device const half* q, device const half* k, device const half* v, device half* o, constant DNA& p,
                                uint2 tg, uint sg, uint lane, uint tid, threadgroup half4* ks, threadgroup half4* vs) {
    uint head = tg.x / p.wtiles, w0 = (tg.x % p.wtiles) * 32;
    uint g = tg.y + p.row0, t = g / p.hblocks, h0 = (g % p.hblocks) * 4;
    uint h = h0 + sg;
    bool hv = h < p.qh;
    uint hc = min(h, p.qh - 1), hl = min(h0 + 3, p.qh - 1);
    uint w = w0 + lane;
    bool valid = hv && w < p.qw;
    uint wc = min(w, p.qw - 1), wl = min(w0 + 31, p.qw - 1);
    uint st = na_start(t, 0, p.kt_, p.kt), sh = na_start(hc, 0, p.kh_, p.kh);
    uint sh0 = na_start(h0, 0, p.kh_, p.kh), sh1 = na_start(hl, 0, p.kh_, p.kh) + p.kh_;
    uint sw = na_start(wc, p.ow, KW, p.kw);
    uint u0 = na_start(w0, p.ow, KW, p.kw), u1 = na_start(wl, p.ow, KW, p.kw) + KW;
    uint nu = u1 - u0;
    ulong qi = ((ulong)t * p.qh + hc) * p.qw + wc;
    device const half4* qr = (device const half4*)(q + qi * p.dim + head * 64);
    float4 qv[16];
    float4 acc[16];
    for (uint i = 0; i < 16; i++) { qv[i] = float4(qr[i]); acc[i] = float4(0.f); }
    float m = -INFINITY, l = 0.f;
    uint j0 = sw - u0;
    for (uint a = 0; a < p.kt_; a++) {
        for (uint b = sh0; b < sh1; b++) {
            ulong base = (((ulong)(st + a) * p.kh + b) * p.kw + u0) * p.dim + head * 64;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = tid; i < nu * 16; i += 128) {
                uint j = i / 16, c = i % 16;
                ks[i] = *((device const half4*)(k + base + (ulong)j * p.dim) + c);
                vs[i] = *((device const half4*)(v + base + (ulong)j * p.dim) + c);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (b < sh || b >= sh + p.kh_) continue;
            float s[KW];
            float mx = m;
            for (uint c = 0; c < KW; c++) {
                threadgroup const half4* kr = ks + (j0 + c) * 16;
                float4 d4 = qv[0] * float4(kr[0]);
                for (uint i = 1; i < 16; i++) d4 = fma(qv[i], float4(kr[i]), d4);
                s[c] = d4.x + d4.y + d4.z + d4.w;
                mx = max(mx, s[c]);
            }
            float sc = exp(m - mx);
            l *= sc;
            for (uint i = 0; i < 16; i++) acc[i] *= sc;
            for (uint c = 0; c < KW; c++) {
                float pc = exp(s[c] - mx);
                l += pc;
                threadgroup const half4* vr = vs + (j0 + c) * 16;
                for (uint i = 0; i < 16; i++) acc[i] = fma(float4(pc), float4(vr[i]), acc[i]);
            }
            m = mx;
        }
    }
    if (!valid) return;
    device half4* orow = (device half4*)(o + qi * p.dim + head * 64);
    float inv = 1.f / l;
    for (uint i = 0; i < 16; i++) orow[i] = half4(acc[i] * inv);
}
#define DV_NA3D(KW)                                                                                                      \
    kernel void dv_na3d_k##KW(device const half* q [[buffer(0)]], device const half* k [[buffer(1)]],                  \
                              device const half* v [[buffer(2)]], device half* o [[buffer(3)]], constant DNA& p [[buffer(4)]], \
                              uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],    \
                              uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {      \
        threadgroup half4 ks[(32 + KW - 1) * 16];                                                                         \
        threadgroup half4 vs[(32 + KW - 1) * 16];                                                                         \
        dv_na3d_impl<KW>(q, k, v, o, p, tg, sg, lane, tid, ks, vs);                                                       \
    }
DV_NA3D(3)
DV_NA3D(5)
DV_NA3D(7)
DV_NA3D(11)

// ---- LTX-2.5's video VAE decoder (metal_ltx_vae.rs, crate::gpu::dit::LtxVaeSession) ----
// Activations are f32 rows in metal_unet.metal's padded layout, the frames as its images
// ((h + 2) x (w + 2) grids of rows, zero borders, after a guard of G = w + 3 rows), with images
// before and after the clip's frames (lo of them before): a 3x3x3 convolution's time taps are
// then rows P apart, and lv_prep fills those images with the repeated first / last frame the
// time padding takes. The convolutions are un_gemm's shifted GEMMs over lv_prep's half rows.
struct LVG { uint h; uint w; uint G; uint P; };
inline uint lv_row(LVG g, uint b, uint y, uint x) { return g.G + b * g.P + (y + 1) * (g.w + 2) + x + 1; }
inline bool lv_interior(LVG g, uint nimg, uint r, thread uint& b, thread uint& y, thread uint& x) {
    if (r < g.G) return false;
    uint rr = r - g.G;
    b = rr / g.P;
    if (b >= nimg) return false;
    uint q = rr - b * g.P, py = q / (g.w + 2), px = q - py * (g.w + 2);
    if (py == 0 || px == 0 || py > g.h || px > g.w) return false;
    y = py - 1; x = px - 1;
    return true;
}
// metal_unet.metal's un_scale (the GEMM divides its products by the same factor).
inline float lv_scale(uint bits) {
    float m = as_type<float>(bits);
    if (!(m > 16384.0f)) return 1.0f;
    int e = 0;
    frexp(min(m, 3.0e38f), e);
    return ldexp(1.0f, 14 - e);
}
// The largest magnitude over the interior pixels of frames (images) lo .. hi of a padded f32
// tensor, as float bits (an atomic max into a slot the host zeroed; only past 16384, where the
// half rows need scaling).
struct LVA { LVG g; uint c; uint lo; uint hi; uint rows; };
kernel void lv_amax(device const float* x [[buffer(0)]], device atomic_uint* m [[buffer(1)]], constant LVA& p [[buffer(2)]],
                    uint i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint c4 = p.c / 4, row = i / c4, j = i - row * c4;
    float a = 0.f;
    uint b, y, xx;
    if (row < p.rows && lv_interior(p.g, p.hi, row, b, y, xx) && b >= p.lo) {
        float4 v = abs(*((device const float4*)(x + (ulong)row * p.c) + j));
        a = max(max(v.x, v.y), max(v.z, v.w));
        if (isnan(a)) a = INFINITY;
    }
    a = simd_max(a);
    if (lane == 0 && a > 16384.0f) atomic_fetch_max_explicit(m, as_type<uint>(a), memory_order_relaxed);
}
// The half input rows of a convolution: every row of the layout (`rows`; zeros on borders and
// guards), image b reading frame clamp(b, lo, hi - 1) of x: as it is (mode 0), RMS-normed over
// its c channels and SiLU'd (mode 1, `PixelNorm`), or scaled by lv_scale of the largest magnitude
// (mode 2). A SIMD-group per row, 8 rows per threadgroup.
struct LVP { LVG g; uint c; uint nimg; uint lo; uint hi; uint mode; uint rows; float eps; };
kernel void lv_prep(device const float* x [[buffer(0)]], device half* out [[buffer(1)]], device const uint* amax [[buffer(2)]],
                    constant LVP& p [[buffer(3)]], uint tg [[threadgroup_position_in_grid]],
                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint row = tg * 8 + sg;
    if (row >= p.rows) return;
    uint c4 = p.c / 4;
    device half4* o = (device half4*)(out + (ulong)row * p.c);
    uint b, y, xx;
    if (!lv_interior(p.g, p.nimg, row, b, y, xx)) {
        for (uint j = lane; j < c4; j += 32) o[j] = half4(0.h);
        return;
    }
    uint sb = clamp(b, p.lo, p.hi - 1);
    device const float4* s = (device const float4*)(x + (ulong)lv_row(p.g, sb, y, xx) * p.c);
    float k = 1.0f;
    if (p.mode == 1) {
        float ss = 0.f;
        for (uint j = lane; j < c4; j += 32) { float4 v = s[j]; ss += dot(v, v); }
        ss = simd_sum(ss);
        k = 1.0f / sqrt(ss / float(p.c) + p.eps);
    } else if (p.mode == 2) {
        k = lv_scale(amax[0]);
    }
    for (uint j = lane; j < c4; j += 32) {
        float4 v = s[j] * k;
        if (p.mode == 1) v = v / (1.0f + exp(-v));
        o[j] = half4(v);
    }
}
// Depth-to-space: output frame F (frames lo + F of the layout d, c channels) pixel (oy, ox)
// channel ch from input frame f = (F + drop) / p1 (layout s, c p1 p2 p3 channels) pixel
// (oy / p2, ox / p3), channel ((ch p1 + i1) p2 + i2) p3 + i3. A thread per output value.
struct LVD { LVG s; LVG d; uint c; uint p1; uint p2; uint p3; uint drop; uint lo; uint t; uint _p; };
kernel void lv_d2s(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], constant LVD& p [[buffer(2)]],
                   uint i [[thread_position_in_grid]]) {
    uint n = p.d.h * p.d.w * p.c, F = i / n;
    if (F >= p.t) return;
    uint q = i - F * n, pix = q / p.c, ch = q - pix * p.c, oy = pix / p.d.w, ox = pix - oy * p.d.w;
    uint ff = F + p.drop, f = ff / p.p1, i1 = ff - f * p.p1;
    uint sy = oy / p.p2, i2 = oy - sy * p.p2, sx = ox / p.p3, i3 = ox - sx * p.p3;
    uint sc = p.c * p.p1 * p.p2 * p.p3;
    y[(ulong)lv_row(p.d, p.lo + F, oy, ox) * p.c + ch] =
        x[(ulong)lv_row(p.s, p.lo + f, sy, sx) * sc + ((ch * p.p1 + i1) * p.p2 + i2) * p.p3 + i3];
}
// The decoder's output: frames lo .. lo + t of conv_out's rows (c = oc pq pq channels ordered
// (channel, x, y) per patch) as compact frames [t, h pq, w pq, oc]. A thread per output value.
struct LVO { LVG s; uint c; uint pq; uint oc; uint lo; uint t; uint _p0; uint _p1; uint _p2; };
kernel void lv_out(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], constant LVO& p [[buffer(2)]],
                   uint i [[thread_position_in_grid]]) {
    uint W = p.s.w * p.pq, n = p.s.h * p.pq * W * p.oc, F = i / n;
    if (F >= p.t) return;
    uint q = i - F * n, pix = q / p.oc, ch = q - pix * p.oc, oy = pix / W, ox = pix - oy * W;
    uint yy = oy / p.pq, qq = oy - yy * p.pq, xx = ox / p.pq, r = ox - xx * p.pq;
    y[i] = x[(ulong)lv_row(p.s, p.lo + F, yy, xx) * p.c + (ch * p.pq + r) * p.pq + qq];
}

// ---- LTX-2.5's vocoder (BigVGAN v2 generators: metal_ltx_voc.rs, crate::gpu::dit::LtxVocSession) ----
// Signals are rows of channels, `t` steps after a guard of g rows (zeros in the half GEMM inputs:
// a 1-D convolution's taps are rows dilation apart, un_gemm's shifted GEMMs; its zero padding).
// The anti-aliased SnakeBeta (`Activation1d`) of f32 rows x (steps g .. g + t) into half rows of
// the same layout (every row of `rows`, zeros outside the steps): per step i, the 2x upsampled
// signal (filter `up` of k taps, already times 2, the input's ends repeated), x + inv sin(alpha
// x)^2 of it, its 2x downsampling (filter `down`, the upsampled ends repeated). A thread per value.
// The value goes out as two halves, `out` and the remainder `lo` (the GEMMs read both: about 21
// bits of the input's mantissa instead of 11).
struct LBA { uint t; uint g; uint c; uint k; uint rows; uint _p0; uint _p1; uint _p2; };
kernel void lb_act(device const float* x [[buffer(0)]], device half* out [[buffer(1)]], device const float* ab [[buffer(2)]],
                   device const float* filt [[buffer(3)]], constant LBA& p [[buffer(4)]], device half* lo [[buffer(5)]],
                   uint i [[thread_position_in_grid]]) {
    uint row = i / p.c, ch = i - row * p.c;
    if (row >= p.rows) return;
    if (row < p.g || row >= p.g + p.t) { out[i] = 0.0h; lo[i] = 0.0h; return; }
    uint s = row - p.g;
    int k = int(p.k), pad = k / 2 - 1, pad_left = pad * 2 + (k - 2) / 2, t = int(p.t);
    float alpha = ab[ch], inv = ab[p.c + ch];
    device const float* up = filt;
    device const float* down = filt + p.k;
    device const float* xc = x + (ulong)p.g * p.c + ch;
    float acc = 0.f;
    for (int j = 0; j < k; j++) {
        int m = clamp(2 * int(s) + j - pad, 0, 2 * t - 1);
        // Upsampled value m: taps of the parity of m + pad_left.
        int n = m + pad_left;
        float u = 0.f;
        for (int q = n % 2; q < k; q += 2) {
            int src = clamp((n - q) / 2 - pad, 0, t - 1);
            u += up[q] * xc[(ulong)src * p.c];
        }
        float sv = sin(alpha * u);
        acc += down[j] * (u + inv * sv * sv);
    }
    half hi = half(acc);
    out[i] = hi;
    lo[i] = half(acc - float(hi));
}
// y[r] = (a[r] + b[r] + c[r]) * scale over rows g .. g + t of n channels (the mean of a stage's
// AMP blocks).
struct LBM { uint n; uint g; uint t; float scale; };
kernel void lb_mean3(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device const float* c [[buffer(2)]],
                     device float* y [[buffer(3)]], constant LBM& p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.t * p.n) return;
    ulong j = (ulong)p.g * p.n + i;
    y[j] = (a[j] + b[j] + c[j]) * p.scale;
}
// f32 rows to half rows and their remainders, every row of `rows` (zeros outside steps g .. g + t).
struct LBH { uint n; uint g; uint t; uint rows; };
kernel void lb_half(device const float* x [[buffer(0)]], device half* out [[buffer(1)]], constant LBH& p [[buffer(2)]],
                    device half* lo [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    uint row = i / p.n;
    if (row >= p.rows) return;
    float v = (row < p.g || row >= p.g + p.t) ? 0.f : x[i];
    half hi = half(v);
    out[i] = hi;
    lo[i] = half(v - float(hi));
}
