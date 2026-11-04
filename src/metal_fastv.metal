// Fast multi-token matmuls for speculative verification in "fast verify" mode (see
// metal_decode.rs, `MetalBackend::build_plan_t` with `fast`). Appended after the decode kernels,
// with the default (reassociating) math: unlike the `mvk_*` kernels, these do NOT reproduce the
// single-token kernels' arithmetic, so a token's results depend (in rounding) on the batch.
//
// y[t] = W x[t] for up to TT tokens (x and y rows `q.sx` / `q.sy` floats apart) on 8x8 simdgroup
// matrices (f32). A threadgroup takes FV_R = 32 rows; its 4 SIMD-groups split the rows' 32-value
// blocks (SIMD-group s takes blocks s, s + 4, ...) and multiply (4 row tiles) x (TT / 8 token
// tiles) per 8-value K-tile, the weights dequantized straight into the lanes' matrix fragments
// (no threadgroup staging), each weight read once for all the tokens. The four partial sums are
// reduced through threadgroup memory and written by `mode` as the single-token kernels do (store,
// add, SiLU / GELU pair, gated add). An 8-token pass costs ~1.1-1.4 single-token passes.
#pragma clang fp reassociate(on)
#define FV_R 32
#define FV_MA(C, A, B) simdgroup_multiply_accumulate(C, A, B, C)
// Partial sums [token][row] of accumulators C<row tile><token tile> at ps.
#define FV_ST(C, ps)                                                                                        \
    simdgroup_store(C##00, ps + 0, FV_R, ulong2(0, 0), true);                                                \
    simdgroup_store(C##10, ps + 8, FV_R, ulong2(0, 0), true);                                                \
    simdgroup_store(C##20, ps + 16, FV_R, ulong2(0, 0), true);                                               \
    simdgroup_store(C##30, ps + 24, FV_R, ulong2(0, 0), true);                                               \
    if (TT > 8 && two) {                                                                                     \
        simdgroup_store(C##01, ps + 8 * FV_R + 0, FV_R, ulong2(0, 0), true);                                 \
        simdgroup_store(C##11, ps + 8 * FV_R + 8, FV_R, ulong2(0, 0), true);                                 \
        simdgroup_store(C##21, ps + 8 * FV_R + 16, FV_R, ulong2(0, 0), true);                                \
        simdgroup_store(C##31, ps + 8 * FV_R + 24, FV_R, ulong2(0, 0), true);                                \
    }

// `fv_load8`: values 8 p .. 8 p + 7 of 32-value block b of a row, dequantized (the formats of
// `FV_FORMATS` in metal_decode.rs; going through the GEMMs' 32-value `load` for one chunk would
// dequantize four times the values, slower than the exact kernels).
inline float4 fv_c4(uint u) { return float4(as_type<char4>(u)); }
inline void fv_load8(Q8_0W w, uint row, uint b, uint p, thread float4& lo, thread float4& hi) {
    device const ushort* B = (device const ushort*)(w.g.q + (ulong)row * w.g.rb + b * 34);
    float d = float(as_type<half>(B[0]));
    device const ushort* q = B + 1 + 4 * p;
    lo = fv_c4(uint(q[0]) | (uint(q[1]) << 16)) * d;
    hi = fv_c4(uint(q[2]) | (uint(q[3]) << 16)) * d;
}
inline void fv_load8(Q8W w, uint row, uint b, uint p, thread float4& lo, thread float4& hi) {
    float d = w.s[(ulong)row * w.bpr + b];
    uint2 u = *(device const uint2*)(w.q + (ulong)row * w.n_in + b * 32 + 8 * p);
    lo = fv_c4(u.x) * d;
    hi = fv_c4(u.y) * d;
}
inline void fv_load8(BfW w, uint row, uint b, uint p, thread float4& lo, thread float4& hi) {
    device const packed_ushort4* q = (device const packed_ushort4*)(w.w + (ulong)row * w.n_in + b * 32 + 8 * p);
    lo = bf4(q[0]);
    hi = bf4(q[1]);
}
inline void fv_load8(F16W w, uint row, uint b, uint p, thread float4& lo, thread float4& hi) {
    device const packed_half4* q = (device const packed_half4*)(w.g.q + (ulong)row * w.g.rb + b * 64 + 16 * p);
    lo = float4(half4(q[0]));
    hi = float4(half4(q[1]));
}
// Q4_0 nibbles: byte i holds value i (low) and i + 16 (high): values 8p.. are the low (p < 2) or
// high nibbles of bytes 8 (p % 2) ..
inline void fv_nib8(uint a, uint c, uint sh, float d, float m, thread float4& lo, thread float4& hi) {
    a >>= sh; c >>= sh;
    lo = (float4(float(a & 15), float((a >> 8) & 15), float((a >> 16) & 15), float((a >> 24) & 15)) - 8.f) * d + m;
    hi = (float4(float(c & 15), float((c >> 8) & 15), float((c >> 16) & 15), float((c >> 24) & 15)) - 8.f) * d + m;
}
inline void fv_load8(Q4W w, uint row, uint b, uint p, thread float4& lo, thread float4& hi) {
    float d = float(w.s[(ulong)row * w.bpr + b]);
    uint2 u = *(device const uint2*)(w.q + (ulong)row * w.half_row + b * 16 + 8 * (p % 2));
    fv_nib8(u.x, u.y, 4 * (p / 2), d, 0.f, lo, hi);
}
inline void fv_load8(Q4A w, uint row, uint b, uint p, thread float4& lo, thread float4& hi) {
    float d = float(w.b.s[(ulong)row * w.b.bpr + b]);
    float m = as_type<float>(uint(w.o[(ulong)row * w.gpr + b / w.gk]) << 16);
    uint2 u = *(device const uint2*)(w.b.q + (ulong)row * w.b.half_row + b * 16 + 8 * (p % 2));
    fv_nib8(u.x, u.y, 4 * (p / 2), d, m, lo, hi);
}
inline void fv_load8(Q4_0W w, uint row, uint b, uint p, thread float4& lo, thread float4& hi) {
    device const ushort* B = (device const ushort*)(w.g.q + (ulong)row * w.g.rb + b * 18);
    float d = float(as_type<half>(B[0]));
    device const ushort* q = B + 1 + 4 * (p % 2);
    fv_nib8(uint(q[0]) | (uint(q[1]) << 16), uint(q[2]) | (uint(q[3]) << 16), 4 * (p / 2), d, 0.f, lo, hi);
}
inline void fv_load8(Q4_KW w, uint row, uint b, uint p, thread float4& lo, thread float4& hi) {
    device const uchar* B = sblk(w.g, row, b, 144);
    uint j = b % 8;
    float sc, mn; scale_min_k4(j, B + 4, sc, mn);
    float d1 = h2f(B) * sc, m1 = h2f(B + 2) * mn;
    uint2 u = *(device const uint2*)(B + 16 + 32 * (j / 2) + 8 * p);
    uint sh = 4 * (j % 2);
    uint a = u.x >> sh, c = u.y >> sh;
    lo = float4(float(a & 15), float((a >> 8) & 15), float((a >> 16) & 15), float((a >> 24) & 15)) * d1 - m1;
    hi = float4(float(c & 15), float((c >> 8) & 15), float((c >> 16) & 15), float((c >> 24) & 15)) * d1 - m1;
}

inline void fv_load8(Q5_KW w, uint row, uint b, uint p, thread float4& lo, thread float4& hi) {
    device const uchar* B = sblk(w.g, row, b, 176);
    uint j = b % 8;
    float sc, mn; scale_min_k4(j, B + 4, sc, mn);
    float d1 = h2f(B) * sc, m1 = h2f(B + 2) * mn;
    device const uchar* q = B + 48 + 32 * (j / 2) + 8 * p;
    uchar sh = uchar(4 * (j % 2));
    uchar4 h0 = ((ld4(B + 16 + 8 * p) >> uchar(j)) & uchar(1)) << uchar(4);
    uchar4 h1 = ((ld4(B + 20 + 8 * p) >> uchar(j)) & uchar(1)) << uchar(4);
    lo = f4(((ld4(q) >> sh) & uchar(15)) | h0) * d1 - m1;
    hi = f4(((ld4(q + 4) >> sh) & uchar(15)) | h1) * d1 - m1;
}
inline void fv_load8(Q6_KW w, uint row, uint b, uint p, thread float4& lo, thread float4& hi) {
    device const uchar* B = sblk(w.g, row, b, 210);
    uint j = b % 8, n = j / 4, jj = j % 4;
    float dl = h2f(B + 208) * float(as_type<char>(B[192 + 8 * n + 2 * jj + p / 2]));
    device const uchar* ql = B + 64 * n + 32 * (jj & 1) + 8 * p;
    device const uchar* qh = B + 128 + 32 * n + 8 * p;
    uchar lsh = uchar(4 * (jj >> 1)), hsh = uchar(2 * jj);
    uchar4 q0 = ((ld4(ql) >> lsh) & uchar(15)) | (((ld4(qh) >> hsh) & uchar(3)) << uchar(4));
    uchar4 q1 = ((ld4(ql + 4) >> lsh) & uchar(15)) | (((ld4(qh + 4) >> hsh) & uchar(3)) << uchar(4));
    lo = (f4(q0) - 32.f) * dl;
    hi = (f4(q1) - 32.f) * dl;
}

// Fills fragment `a` (K-tile kt) from a lane's 8 values (lo, hi): elements 2 kt, 2 kt + 1.
inline void fv_frag(thread simdgroup_float8x8& a, float4 lo, float4 hi, uint kt) {
    thread auto& e = a.thread_elements();
    float4 s = kt < 2 ? lo : hi;
    e[0] = (kt % 2) == 0 ? s.x : s.z;
    e[1] = (kt % 2) == 0 ? s.y : s.w;
}
#define FV3_A(kt)                                                                                            \
    fv_frag(a0, w0l, w0h, kt); fv_frag(a1, w1l, w1h, kt); fv_frag(a2, w2l, w2h, kt); fv_frag(a3, w3l, w3h, kt);
#define FV3_B(kt)                                                                                            \
    { thread auto& e = b0.thread_elements(); e[0] = xa0[kt]; e[1] = xa1[kt]; }                                    \
    if (TT > 8 && two) { thread auto& e = b1.thread_elements(); e[0] = xb0[kt]; e[1] = xb1[kt]; }
#define FV3_MMA(C)                                                                                           \
    FV_MA(C##00, a0, b0); FV_MA(C##10, a1, b0); FV_MA(C##20, a2, b0); FV_MA(C##30, a3, b0);                  \
    if (TT > 8 && two) { FV_MA(C##01, a0, b1); FV_MA(C##11, a1, b1); FV_MA(C##21, a2, b1); FV_MA(C##31, a3, b1); }
// A lane's 4 x values (one per K-tile) of token t at chunk base `xo` (K positions xo + 2 kt).
inline float4 fv_x4(device const float* x, uint t, uint ntok, uint sx, uint xo) {
    if (t >= ntok) return float4(0.f);
    device const float* p = x + (ulong)t * sx + xo;
    return float4(p[0], p[2], p[4], p[6]);
}
template <typename W, int TT>
inline void fv_body(W w, W w2, device const float* x, device float* y, constant MVK& q, uint tg, uint sg, uint lane,
                     threadgroup float* red, device const float* gate) {
    constant MV& p = q.m;
    const bool pair = p.mode == 2 || p.mode == 3;
    const bool two = q.ntok > 8;
    const uint r0 = tg * FV_R;
    const uint nblk = p.n_in / 32;
    const uint fr = 4 * (lane / 16) + (lane % 8) / 2, fc = 4 * ((lane / 8) % 2) + 2 * (lane % 2);
    const uint pa = fc / 2;                  // the lane's W chunk of each block
    const uint xo0 = 8 * (fr / 2) + fr % 2;  // its x positions in a block: xo0 + 2 kt
    uint rows[4];
    for (uint i = 0; i < 4; i++) rows[i] = min(r0 + fr + 8 * i, p.n_out - 1);
    simdgroup_float8x8 c00(0.f), c10(0.f), c20(0.f), c30(0.f), c01(0.f), c11(0.f), c21(0.f), c31(0.f);
    simdgroup_float8x8 d00(0.f), d10(0.f), d20(0.f), d30(0.f), d01(0.f), d11(0.f), d21(0.f), d31(0.f);
    simdgroup_float8x8 a0, a1, a2, a3, b0, b1;
    for (uint b = sg; b < nblk; b += 4) {
        float4 w0l, w0h, w1l, w1h, w2l, w2h, w3l, w3h;
        fv_load8(w, rows[0], b, pa, w0l, w0h);
        fv_load8(w, rows[1], b, pa, w1l, w1h);
        fv_load8(w, rows[2], b, pa, w2l, w2h);
        fv_load8(w, rows[3], b, pa, w3l, w3h);
        const uint xo = b * 32 + xo0;
        float4 xa0 = fv_x4(x, fc, q.ntok, q.sx, xo), xa1 = fv_x4(x, fc + 1, q.ntok, q.sx, xo);
        float4 xb0 = 0.f, xb1 = 0.f;
        if (TT > 8 && two) { xb0 = fv_x4(x, 8 + fc, q.ntok, q.sx, xo); xb1 = fv_x4(x, 9 + fc, q.ntok, q.sx, xo); }
        FV3_A(0) FV3_B(0) FV3_MMA(c)
        FV3_A(1) FV3_B(1) FV3_MMA(c)
        FV3_A(2) FV3_B(2) FV3_MMA(c)
        FV3_A(3) FV3_B(3) FV3_MMA(c)
        if (pair) {
            fv_load8(w2, rows[0], b, pa, w0l, w0h);
            fv_load8(w2, rows[1], b, pa, w1l, w1h);
            fv_load8(w2, rows[2], b, pa, w2l, w2h);
            fv_load8(w2, rows[3], b, pa, w3l, w3h);
            FV3_A(0) FV3_B(0) FV3_MMA(d)
            FV3_A(1) FV3_B(1) FV3_MMA(d)
            FV3_A(2) FV3_B(2) FV3_MMA(d)
            FV3_A(3) FV3_B(3) FV3_MMA(d)
        }
    }
    threadgroup float* ps = red + sg * (2 * TT * FV_R);
    FV_ST(c, ps)
    if (pair) { FV_ST(d, (ps + TT * FV_R)) }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint tid = sg * 32 + lane;
    uint r = tid % FV_R, o = r0 + r;
    if (o >= p.n_out) return;
    for (uint t = tid / FV_R; t < q.ntok; t += 128 / FV_R) {
        float a = 0.f, bb = 0.f;
        for (uint s = 0; s < 4; s++) a += red[s * (2 * TT * FV_R) + t * FV_R + r];
        if (pair)
            for (uint s = 0; s < 4; s++) bb += red[s * (2 * TT * FV_R) + TT * FV_R + t * FV_R + r];
        device float* yo = y + (ulong)t * q.sy + o;
        if (p.mode == 0) *yo = a;
        else if (p.mode == 1) *yo += a;
        else if (p.mode == 2) *yo = a / (1.0f + exp(-a)) * bb;
        else if (p.mode == 3) {
            float u = 0.7978845608f * (a + 0.044715f * a * a * a);
            *yo = 0.5f * a * (1.0f + precise::tanh(u)) * bb;
        } else *yo += a / (1.0f + exp(-gate[(ulong)t * q.sgate]));
    }
}
#define FV_SIG(NAME) \
kernel void NAME(device const uchar* wb [[buffer(0)]], device const uchar* sb [[buffer(1)]], device const float* x [[buffer(2)]], \
                 device float* y [[buffer(3)]], constant MVK& q [[buffer(4)]], device const uchar* wb2 [[buffer(5)]], \
                 device const uchar* sb2 [[buffer(6)]], device const uchar* ob [[buffer(7)]], device const uchar* ob2 [[buffer(8)]], \
                 device const float* gate [[buffer(9)]], \
                 uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], \
                 uint lane [[thread_index_in_simdgroup]])
// NAME, weight type, its constructor from (data, scales, offsets, row bytes) of one matrix, the
// tile element type (float or half) and token capacity.
// NAME, weight type, its constructor from (data, scales, offsets, row bytes) of one matrix, and
// token capacity. Threadgroup memory: the partial sums of 4 SIMD-groups, two matrices for a pair.
#define FV_INST(NAME, W, MAKE, TT) \
FV_SIG(NAME) { threadgroup float red[4 * 2 * TT * FV_R]; constant MV& p = q.m; \
    W w = MAKE(wb, sb, ob, p.w_off, p.s_off, p.o_off); W w2 = MAKE(wb2, sb2, ob2, p.w2_off, p.s2_off, p.o2_off); \
    fv_body<W, TT>(w, w2, x, y, q, tg, sg, lane, red, gate); }
#define FV_MK_BF(W, S, O, WO, SO, OO) BfW{ (device const ushort*)((W) + (WO)), p.n_in }
#define FV_MK_Q8(W, S, O, WO, SO, OO) Q8W{ (device const char*)((W) + (WO)), (device const float*)((S) + (SO)), p.n_in / 32, p.n_in }
#define FV_MK_Q4(W, S, O, WO, SO, OO) Q4W{ (W) + (WO), (device const half*)((S) + (SO)), p.n_in / 32, p.n_in / 2 }
#define FV_MK_Q4A(W, S, O, WO, SO, OO) Q4A{ FV_MK_Q4(W, S, O, WO, SO, OO), (device const ushort*)((O) + (OO)), p.gpr, p.gk }
