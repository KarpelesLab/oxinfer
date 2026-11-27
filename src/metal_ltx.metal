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
