// Single-token decode kernels (see metal_decode.rs). Appended to the library after SHADER_SRC and
// GGUF_SHADER_SRC, whose dequantizers (`load`, `h2f`, ...) the generic matvec reuses.

// ---- Matrix-vector products ---------------------------------------------------------------------
// y[r] = W[r] . x for one input row. A threadgroup computes NR consecutive rows with KS SIMD-groups
// splitting the row between them (KS > 1 for short rows, so enough lanes stream weights); every
// lane takes whole "chunks" of a row (a format-specific run of values, e.g. 32 of a 256-value
// K-quant super-block), consecutive lanes consecutive chunks, so a SIMD-group reads contiguous
// memory. The x values of a chunk are loaded once and reused for the NR rows.
// Modes: 0 y = W x, 1 y += W x (residual), 2 y = silu(W x) * (W2 x) (gate / up pair), 3 the same
// with GELU (tanh approximation).
struct MV { uint n_out; uint n_in; uint mode; uint rows_per_tg; ulong w_off; ulong s_off; ulong w2_off; ulong s2_off; ulong o_off; ulong o2_off; uint gpr; uint gk; uint ldy; uint _p; };

struct X16 { float4 v[4]; };
struct X32 { float4 v[8]; };
inline void xload(device const float* x, uint c, thread X16& xr) {
    device const float4* p = (device const float4*)(x + c * 16);
    xr.v[0] = p[0]; xr.v[1] = p[1]; xr.v[2] = p[2]; xr.v[3] = p[3];
}
inline void xload(device const float* x, uint c, thread X32& xr) {
    device const float4* p = (device const float4*)(x + c * 32);
    for (uint i = 0; i < 8; i++) xr.v[i] = p[i];
}
inline float hsum(float4 v) { return (v.x + v.y) + (v.z + v.w); }

// Nibble products without shifts: the masked nibble of byte j of a word, converted to float, is
// the value times 2^(8j) (low nibbles) or 2^(8j+4) (high nibbles); the powers are divided out once
// per word group. `lo` gets x[4k+j] * low nibble j, `hi` x2[4k+j] * high nibble j.
inline void nib_acc(uint w, float4 xa, float4 xb, thread float4& lo, thread float4& hi) {
    lo += xa * float4(float(w & 0xFu), float(w & 0xF00u), float(w & 0xF0000u), float(w & 0xF000000u));
    hi += xb * float4(float(w & 0xF0u), float(w & 0xF000u), float(w & 0xF00000u), float(w & 0xF0000000u));
}
constant float4 NIB_LO = float4(1.f, 1.f / 256.f, 1.f / 65536.f, 1.f / 16777216.f);
constant float4 NIB_HI = float4(1.f / 16.f, 1.f / 4096.f, 1.f / 1048576.f, 1.f / 268435456.f);

// ---- bf16: chunk of 16 (32 bytes; tensors inside safetensors blobs are only 2-byte aligned).
struct MBf { device const uchar* w; ulong rb; };
inline float mdot(MBf f, uint row, uint c, thread const X16& xr) {
    device const packed_ushort4* p = (device const packed_ushort4*)(f.w + (ulong)row * f.rb + c * 32);
    float4 a = bf4(p[0]) * xr.v[0];
    a = fma(bf4(p[1]), xr.v[1], a);
    a = fma(bf4(p[2]), xr.v[2], a);
    a = fma(bf4(p[3]), xr.v[3], a);
    return hsum(a);
}
// ---- f16 (GGUF): chunk of 16.
struct MF16 { device const uchar* w; ulong rb; };
inline float mdot(MF16 f, uint row, uint c, thread const X16& xr) {
    device const packed_half4* p = (device const packed_half4*)(f.w + (ulong)row * f.rb + c * 32);
    float4 a = float4(half4(p[0])) * xr.v[0];
    a = fma(float4(half4(p[1])), xr.v[1], a);
    a = fma(float4(half4(p[2])), xr.v[2], a);
    a = fma(float4(half4(p[3])), xr.v[3], a);
    return hsum(a);
}
// ---- oxinfer Q8: int8 rows plus an f32 scale per 32; chunk of 16.
struct MQ8 { device const char* q; device const float* s; uint n_in; };
inline float mdot(MQ8 f, uint row, uint c, thread const X16& xr) {
    uint4 u = *(device const uint4*)(f.q + (ulong)row * f.n_in + c * 16);
    float d = f.s[(ulong)row * (f.n_in / 32) + c / 2];
    float4 a = float4(as_type<char4>(u.x)) * xr.v[0];
    a = fma(float4(as_type<char4>(u.y)), xr.v[1], a);
    a = fma(float4(as_type<char4>(u.z)), xr.v[2], a);
    a = fma(float4(as_type<char4>(u.w)), xr.v[3], a);
    return d * hsum(a);
}
// ---- oxinfer Q4 (Q4_0 nibbles, f16 scales in a separate array; optional MLX affine offsets):
// chunk of 32 = one block of 16 bytes (low nibble = value i, high = value i + 16).
struct MQ4 { device const uchar* q; device const half* s; uint n_in; device const ushort* o; uint gpr; uint gk; };
inline float q4_block(uint4 u, thread const X32& xr) {
    float4 lo = 0.f, hi = 0.f;
    nib_acc(u.x, xr.v[0], xr.v[4], lo, hi);
    nib_acc(u.y, xr.v[1], xr.v[5], lo, hi);
    nib_acc(u.z, xr.v[2], xr.v[6], lo, hi);
    nib_acc(u.w, xr.v[3], xr.v[7], lo, hi);
    return dot(lo, NIB_LO) + dot(hi, NIB_HI);
}
inline float xsum32(thread const X32& xr) {
    float4 s = ((xr.v[0] + xr.v[1]) + (xr.v[2] + xr.v[3])) + ((xr.v[4] + xr.v[5]) + (xr.v[6] + xr.v[7]));
    return hsum(s);
}
inline float mdot(MQ4 f, uint row, uint c, thread const X32& xr) {
    uint4 u = *(device const uint4*)(f.q + (ulong)row * (f.n_in / 2) + c * 16);
    float d = float(f.s[(ulong)row * (f.n_in / 32) + c]);
    float r = d * (q4_block(u, xr) - 8.f * xsum32(xr));
    if (f.gk != 0) r += as_type<float>(uint(f.o[(ulong)row * f.gpr + c / f.gk]) << 16) * xsum32(xr);
    return r;
}
// ---- GGUF Q4_0: 18-byte blocks (2-byte aligned), chunk of 32.
struct MQ4_0 { device const uchar* w; ulong rb; };
inline uint ld32u(device const ushort* p) { return uint(p[0]) | (uint(p[1]) << 16); }
inline float mdot(MQ4_0 f, uint row, uint c, thread const X32& xr) {
    device const ushort* b = (device const ushort*)(f.w + (ulong)row * f.rb + c * 18);
    float d = float(as_type<half>(b[0]));
    uint4 u = uint4(ld32u(b + 1), ld32u(b + 3), ld32u(b + 5), ld32u(b + 7));
    return d * (q4_block(u, xr) - 8.f * xsum32(xr));
}
// ---- GGUF Q8_0: 34-byte blocks (2-byte aligned), chunk of 32.
struct MQ8_0 { device const uchar* w; ulong rb; };
inline float4 c4(ushort a, ushort b) { return float4(float(as_type<char2>(a).x), float(as_type<char2>(a).y), float(as_type<char2>(b).x), float(as_type<char2>(b).y)); }
inline float mdot(MQ8_0 f, uint row, uint c, thread const X32& xr) {
    device const ushort* b = (device const ushort*)(f.w + (ulong)row * f.rb + c * 34);
    float d = float(as_type<half>(b[0]));
    float4 a = c4(b[1], b[2]) * xr.v[0];
    a = fma(c4(b[3], b[4]), xr.v[1], a);
    a = fma(c4(b[5], b[6]), xr.v[2], a);
    a = fma(c4(b[7], b[8]), xr.v[3], a);
    a = fma(c4(b[9], b[10]), xr.v[4], a);
    a = fma(c4(b[11], b[12]), xr.v[5], a);
    a = fma(c4(b[13], b[14]), xr.v[6], a);
    a = fma(c4(b[15], b[16]), xr.v[7], a);
    return d * hsum(a);
}
// ---- K-quants: 256-value super-blocks, 8 chunks of 32 each. Chunk k of a super-block covers 16
// bytes of quants: group g = k / 2 (64 values), half h = k % 2; its low nibbles are values
// 64g + 16h + i (sub-block 2g), the high nibbles 64g + 32 + 16h + i (sub-block 2g + 1).
struct XK { float4 l[4]; float4 h[4]; float sl; float sh; };
inline void xload_k(device const float* x, uint c, thread XK& xr) {
    uint sb = c / 8, k = c % 8, g = k / 2, h = k % 2;
    device const float4* pl = (device const float4*)(x + sb * 256 + 64 * g + 16 * h);
    device const float4* ph = pl + 8;
    for (uint i = 0; i < 4; i++) { xr.l[i] = pl[i]; xr.h[i] = ph[i]; }
    xr.sl = hsum((xr.l[0] + xr.l[1]) + (xr.l[2] + xr.l[3]));
    xr.sh = hsum((xr.h[0] + xr.h[1]) + (xr.h[2] + xr.h[3]));
}
struct XKQ { XK k; };
inline void xload(device const float* x, uint c, thread XKQ& xr) { xload_k(x, c, xr.k); }
// Scale and min of sub-block j of a Q4_K / Q5_K super-block (6 bits each, packed in 12 bytes).
inline float2 kscale(device const uchar* s, uint j) {
    if (j < 4) return float2(float(s[j] & 63), float(s[j + 4] & 63));
    return float2(float((s[j + 4] & 15) | ((s[j - 4] >> 6) << 4)), float((s[j + 4] >> 4) | ((s[j] >> 6) << 4)));
}
// Both sub-blocks of a chunk (2g, 2g + 1) from the 12 scale bytes read as three words, without
// branches: (scale, min) of sub-block 2g in .xy, of 2g + 1 in .zw.
inline float4 kscale2(device const uchar* s, uint g) {
    device const uint* w = (device const uint*)s;
    uint u0 = w[0], u1 = w[1], u2 = w[2];
    uint i = (2 * g) & 3, sh0 = 8 * i, sh1 = sh0 + 8;
    uint a0 = (u0 >> sh0) & 255, b0 = (u1 >> sh0) & 255, c0 = (u2 >> sh0) & 255;
    uint a1 = (u0 >> sh1) & 255, b1 = (u1 >> sh1) & 255, c1 = (u2 >> sh1) & 255;
    bool hi = g >= 2;
    uint s0 = hi ? ((c0 & 15) | ((a0 >> 6) << 4)) : (a0 & 63);
    uint m0 = hi ? ((c0 >> 4) | ((b0 >> 6) << 4)) : (b0 & 63);
    uint s1 = hi ? ((c1 & 15) | ((a1 >> 6) << 4)) : (a1 & 63);
    uint m1 = hi ? ((c1 >> 4) | ((b1 >> 6) << 4)) : (b1 & 63);
    return float4(float(s0), float(m0), float(s1), float(m1));
}
struct MQ4_K { device const uchar* w; ulong rb; };
inline float mdot(MQ4_K f, uint row, uint c, thread const XKQ& x) {
    uint k = c % 8, g = k / 2, h = k % 2;
    device const uchar* B = f.w + (ulong)row * f.rb + (c / 8) * 144;
    float2 dm = float2(*(device const half2*)B);
    float4 sc = kscale2(B + 4, g);
    float2 s0 = sc.xy, s1 = sc.zw;
    uint4 u = *(device const uint4*)(B + 16 + 32 * g + 16 * h);
    float4 lo = 0.f, hi = 0.f;
    nib_acc(u.x, x.k.l[0], x.k.h[0], lo, hi);
    nib_acc(u.y, x.k.l[1], x.k.h[1], lo, hi);
    nib_acc(u.z, x.k.l[2], x.k.h[2], lo, hi);
    nib_acc(u.w, x.k.l[3], x.k.h[3], lo, hi);
    return dm.x * (s0.x * dot(lo, NIB_LO) + s1.x * dot(hi, NIB_HI)) - dm.y * (s0.y * x.k.sl + s1.y * x.k.sh);
}
// Q5_K: the fifth bit of value 64g + 16h + i (and of 64g + 32 + 16h + i) is bit 2g (2g + 1) of
// qh[16h + i].
struct MQ5_K { device const uchar* w; ulong rb; };
inline void bit_acc(uint w, uint sh, float4 xa, float4 xb, thread float4& lo, thread float4& hi) {
    uint a = (w >> sh) & 0x01010101u, b = (w >> (sh + 1)) & 0x01010101u;
    lo += xa * float4(float(a & 1u), float(a & 0x100u), float(a & 0x10000u), float(a & 0x1000000u));
    hi += xb * float4(float(b & 1u), float(b & 0x100u), float(b & 0x10000u), float(b & 0x1000000u));
}
inline float mdot(MQ5_K f, uint row, uint c, thread const XKQ& x) {
    uint k = c % 8, g = k / 2, h = k % 2;
    device const uchar* B = f.w + (ulong)row * f.rb + (c / 8) * 176;
    float2 dm = float2(*(device const half2*)B);
    float4 sc = kscale2(B + 4, g);
    float2 s0 = sc.xy, s1 = sc.zw;
    uint4 u = *(device const uint4*)(B + 48 + 32 * g + 16 * h);
    uint4 q = *(device const uint4*)(B + 16 + 16 * h);
    float4 lo = 0.f, hi = 0.f, bl = 0.f, bh = 0.f;
    nib_acc(u.x, x.k.l[0], x.k.h[0], lo, hi);
    nib_acc(u.y, x.k.l[1], x.k.h[1], lo, hi);
    nib_acc(u.z, x.k.l[2], x.k.h[2], lo, hi);
    nib_acc(u.w, x.k.l[3], x.k.h[3], lo, hi);
    bit_acc(q.x, 2 * g, x.k.l[0], x.k.h[0], bl, bh);
    bit_acc(q.y, 2 * g, x.k.l[1], x.k.h[1], bl, bh);
    bit_acc(q.z, 2 * g, x.k.l[2], x.k.h[2], bl, bh);
    bit_acc(q.w, 2 * g, x.k.l[3], x.k.h[3], bl, bh);
    float vl = dot(lo, NIB_LO) + 16.f * dot(bl, NIB_LO), vh = dot(hi, NIB_HI) + 16.f * dot(bh, NIB_LO);
    return dm.x * (s0.x * vl + s1.x * vh) - dm.y * (s0.y * x.k.sl + s1.y * x.k.sh);
}
// Q6_K (210-byte super-blocks, 2-byte aligned): chunk k covers half n = k / 4 and positions
// l = 8 (k % 4) + i of it; its values are 128n + l + {0, 32, 64, 96} (sub-blocks of 16 with
// scales sc[8n + l / 16 + {0, 2, 4, 6}]).
struct X6 { float4 v[8]; };
inline void xload(device const float* x, uint c, thread X6& xr) {
    uint n = (c % 8) / 4, l0 = 8 * (c % 4);
    device const float* p = x + (c / 8) * 256 + 128 * n + l0;
    for (uint j = 0; j < 4; j++) {
        xr.v[2 * j] = *(device const float4*)(p + 32 * j);
        xr.v[2 * j + 1] = *(device const float4*)(p + 32 * j + 4);
    }
}
struct MQ6_K { device const uchar* w; ulong rb; };
// Four 6-bit values (minus 32) of one quarter: low bits from `ql` nibbles, high bits from `qh`.
inline float4 q6v(uint ql, uint qh) {
    uint q = (ql & 0x0F0F0F0Fu) | ((qh & 0x03030303u) << 4);
    return float4(as_type<uchar4>(q)) - 32.f;
}
inline float mdot(MQ6_K f, uint row, uint c, thread const X6& x) {
    uint n = (c % 8) / 4, l0 = 8 * (c % 4);
    device const uchar* B = f.w + (ulong)row * f.rb + (c / 8) * 210;
    device const ushort* q1 = (device const ushort*)(B + 64 * n + l0);
    device const ushort* q2 = (device const ushort*)(B + 64 * n + 32 + l0);
    device const ushort* qh = (device const ushort*)(B + 128 + 32 * n + l0);
    uint a0 = ld32u(q1), a1 = ld32u(q1 + 2), b0 = ld32u(q2), b1 = ld32u(q2 + 2), h0 = ld32u(qh), h1 = ld32u(qh + 2);
    device const char* sc = (device const char*)(B + 192 + 8 * n + l0 / 16);
    float d = float(*(device const half*)(B + 208));
    float s0 = dot(q6v(a0, h0), x.v[0]) + dot(q6v(a1, h1), x.v[1]);
    float s1 = dot(q6v(b0, h0 >> 2), x.v[2]) + dot(q6v(b1, h1 >> 2), x.v[3]);
    float s2 = dot(q6v(a0 >> 4, h0 >> 4), x.v[4]) + dot(q6v(a1 >> 4, h1 >> 4), x.v[5]);
    float s3 = dot(q6v(b0 >> 4, h0 >> 6), x.v[6]) + dot(q6v(b1 >> 4, h1 >> 6), x.v[7]);
    return d * (s0 * float(sc[0]) + s1 * float(sc[2]) + s2 * float(sc[4]) + s3 * float(sc[6]));
}
// ---- Any other GGUF block format through its GEMM dequantizer: chunk of 32.
template <typename W>
struct MGen { W w; };
template <typename W>
inline float mdot(MGen<W> f, uint row, uint c, thread const X32& xr) {
    float4 v[8];
    load(f.w, row, c, v);
    float4 a = v[0] * xr.v[0];
    for (uint i = 1; i < 8; i++) a = fma(v[i], xr.v[i], a);
    return hsum(a);
}

// Lane partial sums of rows r0 .. r0 + NR - 1 of f (into acc) and, for a pair, of f2 (rows
// shifted by r2off, into acc2), times `scale`.
template <typename F, typename X, uint CH, int NR, int KS>
inline void mv_acc(F f, F f2, bool pair, uint r2off, device const float* x, uint n_in, uint n_out, uint r0, uint sg, uint lane,
                   thread float* acc, thread float* acc2, float scale) {
    uint rows[NR];
    for (int r = 0; r < NR; r++) rows[r] = min(r0 + r, n_out - 1);
    uint nch = n_in / CH;
    float a[NR], b[NR];
    for (int r = 0; r < NR; r++) { a[r] = 0.f; b[r] = 0.f; }
    for (uint c = sg * 32 + lane; c < nch; c += KS * 32) {
        X xr;
        xload(x, c, xr);
        for (int r = 0; r < NR; r++) a[r] += mdot(f, rows[r], c, xr);
        if (pair)
            for (int r = 0; r < NR; r++) b[r] += mdot(f2, rows[r] + r2off, c, xr);
    }
    for (int r = 0; r < NR; r++) { acc[r] += scale * a[r]; acc2[r] += scale * b[r]; }
}
// Reduces the lane sums over the threadgroup and writes rows r0 .. of y by `mode`: 0 store, 1 add,
// 2 silu(acc) * acc2, 3 gelu(acc) * acc2, 4 add acc * sigmoid(*gate).
template <int NR, int KS>
inline void mv_finish(thread float* acc, thread float* acc2, device float* y, uint mode, uint r0, uint n_out, uint sg, uint lane,
                      threadgroup float* red, device const float* gate) {
    for (int r = 0; r < NR; r++) { acc[r] = simd_sum(acc[r]); acc2[r] = simd_sum(acc2[r]); }
    if (KS > 1) {
        if (lane == 0)
            for (int r = 0; r < NR; r++) { red[sg * 2 * NR + r] = acc[r]; red[sg * 2 * NR + NR + r] = acc2[r]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg != 0) return;
        for (int r = 0; r < NR; r++) {
            float a = 0.f, b = 0.f;
            for (int s = 0; s < KS; s++) { a += red[s * 2 * NR + r]; b += red[s * 2 * NR + NR + r]; }
            acc[r] = a; acc2[r] = b;
        }
    }
    if (lane < NR && r0 + lane < n_out) {
        float a = acc[0], b = acc2[0];
        for (int r = 1; r < NR; r++) if (lane == uint(r)) { a = acc[r]; b = acc2[r]; }
        uint o = r0 + lane;
        if (mode == 0) y[o] = a;
        else if (mode == 1) y[o] += a;
        else if (mode == 2) y[o] = a / (1.0f + exp(-a)) * b;
        else if (mode == 3) {
            float t = 0.7978845608f * (a + 0.044715f * a * a * a);
            y[o] = 0.5f * a * (1.0f + precise::tanh(t)) * b;
        } else y[o] += a / (1.0f + exp(-gate[0]));
    }
}
template <typename F, typename X, uint CH, int NR, int KS>
inline void mv_body(F f, F f2, device const float* x, device float* y, constant MV& p, uint tg, uint sg, uint lane,
                    threadgroup float* red, device const float* gate) {
    float acc[NR], acc2[NR];
    for (int r = 0; r < NR; r++) { acc[r] = 0.f; acc2[r] = 0.f; }
    mv_acc<F, X, CH, NR, KS>(f, f2, p.mode == 2 || p.mode == 3, 0, x, p.n_in, p.n_out, tg * NR, sg, lane, acc, acc2, 1.f);
    mv_finish<NR, KS>(acc, acc2, y, p.mode, tg * NR, p.n_out, sg, lane, red, gate);
}

#define MV_SIG(NAME) \
kernel void NAME(device const uchar* wb [[buffer(0)]], device const uchar* sb [[buffer(1)]], device const float* x [[buffer(2)]], \
                 device float* y [[buffer(3)]], constant MV& p [[buffer(4)]], device const uchar* wb2 [[buffer(5)]], \
                 device const uchar* sb2 [[buffer(6)]], device const uchar* ob [[buffer(7)]], device const uchar* ob2 [[buffer(8)]], \
                 device const float* gate [[buffer(9)]], \
                 uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], \
                 uint lane [[thread_index_in_simdgroup]])
// Formats whose rows are raw bytes: the row length in bytes is passed in `gpr`.
#define MV_RAW(NAME, T, X, CH, NR, KS) \
MV_SIG(NAME) { threadgroup float red[2 * NR * KS]; \
    T f = { wb + p.w_off, p.gpr }; T f2 = { wb2 + p.w2_off, p.gpr }; \
    mv_body<T, X, CH, NR, KS>(f, f2, x, y, p, tg, sg, lane, red, gate); }
#define MV_Q8(NAME, NR, KS) \
MV_SIG(NAME) { threadgroup float red[2 * NR * KS]; \
    MQ8 f = { (device const char*)(wb + p.w_off), (device const float*)(sb + p.s_off), p.n_in }; \
    MQ8 f2 = { (device const char*)(wb2 + p.w2_off), (device const float*)(sb2 + p.s2_off), p.n_in }; \
    mv_body<MQ8, X16, 16, NR, KS>(f, f2, x, y, p, tg, sg, lane, red, gate); }
#define MV_Q4(NAME, NR, KS) \
MV_SIG(NAME) { threadgroup float red[2 * NR * KS]; \
    MQ4 f = { wb + p.w_off, (device const half*)(sb + p.s_off), p.n_in, (device const ushort*)(ob + p.o_off), p.gpr, p.gk }; \
    MQ4 f2 = { wb2 + p.w2_off, (device const half*)(sb2 + p.s2_off), p.n_in, (device const ushort*)(ob2 + p.o2_off), p.gpr, p.gk }; \
    mv_body<MQ4, X32, 32, NR, KS>(f, f2, x, y, p, tg, sg, lane, red, gate); }
#define MV_GEN(NAME, W, NR, KS) \
MV_SIG(NAME) { threadgroup float red[2 * NR * KS]; \
    MGen<W> f = { W{ { wb + p.w_off, p.gpr } } }; MGen<W> f2 = { W{ { wb2 + p.w2_off, p.gpr } } }; \
    mv_body<MGen<W>, X32, 32, NR, KS>(f, f2, x, y, p, tg, sg, lane, red, gate); }

// ---- RMSNorm over rows of n: out = x * rsqrt(mean(x^2) + eps) * w (w optional; `add`: out += ...).
// One threadgroup per row.
struct NP { uint n; float eps; uint has_w; uint add; uint x_stride; uint o_stride; float pre; uint _p; };
kernel void dec_rms_norm(device const float* x [[buffer(0)]], device float* out [[buffer(1)]], device const float* w [[buffer(2)]],
                         constant NP& p [[buffer(3)]], uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                         uint nt [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[32];
    device const float* xr = x + (ulong)r * p.x_stride;
    device float* o = out + (ulong)r * p.o_stride;
    // Up to 8 values per thread are loaded at once and kept in registers (one memory latency
    // instead of a chain); longer rows loop.
    float xv[8];
    bool fits = p.n <= 8 * nt;
    float s = 0.f;
    if (fits) {
        for (uint j = 0; j < 8; j++) { uint i = tid + j * nt; xv[j] = i < p.n ? xr[i] * p.pre : 0.f; }
        for (uint j = 0; j < 8; j++) s += xv[j] * xv[j];
    } else {
        for (uint i = tid; i < p.n; i += nt) { float v = xr[i] * p.pre; s += v * v; }
    }
    s = simd_sum(s);
    uint nsg = (nt + 31) / 32;
    if (nsg > 1) {
        if (lane == 0) red[sg] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s = 0.f;
        for (uint i = 0; i < nsg; i++) s += red[i];
    }
    float k = rsqrt(s / float(p.n) + p.eps);
    if (fits) {
        for (uint j = 0; j < 8; j++) {
            uint i = tid + j * nt;
            if (i < p.n) {
                float v = xv[j] * k;
                if (p.has_w != 0) v *= w[i];
                if (p.add != 0) o[i] += v; else o[i] = v;
            }
        }
    } else {
        for (uint i = tid; i < p.n; i += nt) {
            float v = xr[i] * p.pre * k;
            if (p.has_w != 0) v *= w[i];
            if (p.add != 0) o[i] += v; else o[i] = v;
        }
    }
}

// ---- Attention inputs for one token: per-head RMSNorm (optional) and RoPE of the queries (into
// `qo`, contiguous heads) and keys (into the key cache at `pos`), values copied into the value
// cache. Threadgroups: nh query heads, then nkv key heads, then nkv value heads; one thread per
// dimension (hd threads). cos / sin of the position come from the host (`cs`: half cos, half sin).
struct QP { uint nh; uint nkv; uint hd; uint half_rot; uint q_stride; uint k_off; uint v_off; uint pos;
            uint interleaved; uint has_qn; uint has_kn; float eps; uint v_norm; uint v_from_k; uint _p0; uint _p1; };
kernel void dec_qk_rope(device const float* qkv [[buffer(0)]], device float* qo [[buffer(1)]], device float* kc [[buffer(2)]],
                        device float* vc [[buffer(3)]], device const float* qn [[buffer(4)]], device const float* kn [[buffer(5)]],
                        constant QP& p [[buffer(6)]], constant float* cs [[buffer(7)]],
                        uint h [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float buf[512];
    threadgroup float red[16];
    uint hd = p.hd;
    uint kind = h < p.nh ? 0 : (h < p.nh + p.nkv ? 1 : 2); // query, key, value head
    uint hh = kind == 0 ? h : (kind == 1 ? h - p.nh : h - p.nh - p.nkv);
    float v = kind == 0 ? qkv[hh * p.q_stride + i]
            : (kind == 1 || p.v_from_k != 0) ? qkv[p.k_off + hh * hd + i] : qkv[p.v_off + hh * hd + i];
    bool norm = kind == 0 ? p.has_qn != 0 : (kind == 1 ? p.has_kn != 0 : p.v_norm != 0);
    if (norm) {
        float s = simd_sum(v * v);
        uint nsg = (hd + 31) / 32;
        if (lane == 0) red[sg] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s = 0.f;
        for (uint j = 0; j < nsg; j++) s += red[j];
        v = v * rsqrt(s / float(hd) + p.eps);
        if (kind == 0) v *= qn[i];
        else if (kind == 1) v *= kn[i];
    }
    if (kind == 2) {
        vc[((ulong)p.pos * p.nkv + hh) * hd + i] = v;
        return;
    }
    buf[i] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint hr = p.half_rot;
    if (i < 2 * hr) {
        if (p.interleaved != 0) {
            uint j = i / 2;
            float c = cs[j], s = cs[hr + j];
            float x1 = buf[2 * j], x2 = buf[2 * j + 1];
            v = (i % 2 == 0) ? x1 * c - x2 * s : x2 * c + x1 * s;
        } else {
            uint j = i % hr;
            float c = cs[j], s = cs[hr + j];
            float x1 = buf[j], x2 = buf[j + hr];
            v = (i < hr) ? x1 * c - x2 * s : x2 * c + x1 * s;
        }
    }
    if (kind == 0) qo[hh * hd + i] = v;
    else kc[((ulong)p.pos * p.nkv + hh) * hd + i] = v;
}

// ---- Attention of one query per head over the cache (positions lo..len-1). One threadgroup of
// 8 SIMD-groups per query head; SIMD-group s takes keys lo + s, lo + s + 8, ...; each lane holds
// D = hd / 32 dimensions. Online softmax per SIMD-group, combined at the end. `gate`: out *=
// sigmoid(gate[h * g_stride + d]) (Qwen3.5 gated attention).
// Long contexts split the keys over `nsplit` threadgroups per head (chunks of `chunk` keys from
// lo), each writing a partial (unnormalized output, max, sum) to `part`, combined by
// dec_attn_combine; with nsplit = 1 the threadgroup writes the result itself.
struct AP { uint nh; uint nkv; uint len; uint lo; float scale; uint has_gate; uint g_stride; uint g_off; uint nsplit; uint chunk; uint _p0; uint _p1; };
// One threadgroup per (G query heads sharing a key / value head, chunk of keys): every key and
// value row is read once for the G heads.
template <int D, int G>
inline void attn_body(device const float* q, device const float* kc, device const float* vc, device float* out, device const float* gate,
                      device float* part, constant AP& p, uint hg, uint split, uint tid, uint sg, uint lane, threadgroup float* sm,
                      threadgroup float* so, threadgroup float* qs) {
    constexpr uint HD = D * 32;
    uint h0 = hg * G;
    uint kvh = h0 / (p.nh / p.nkv);
    // The scaled queries in threadgroup memory (every lane reads all of them).
    for (uint i = tid; i < G * HD; i += 256) qs[i] = q[h0 * HD + i] * p.scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float o[G][D], m[G], l[G];
    for (int g = 0; g < G; g++) {
        m[g] = -INFINITY;
        l[g] = 0.f;
        for (int d = 0; d < D; d++) o[g][d] = 0.f;
    }
    ulong rs = (ulong)p.nkv * HD; // floats per cache position
    // SIMD-group sg takes blocks of 32 keys (j0 + 32 (sg + 8 i)): lane t scores key t of the block
    // (its whole row, so the 32 rows load in parallel), the block's softmax update is a few SIMD
    // reductions, then the values are accumulated with each lane owning D dimensions.
    uint j0 = p.lo + split * p.chunk, j1 = min(p.len, j0 + p.chunk);
    for (uint base = j0 + sg * 32; base < j1; base += 8 * 32) {
        uint j = base + lane;
        float s[G];
        for (int g = 0; g < G; g++) s[g] = -INFINITY;
        if (j < j1) {
            device const float4* kr = (device const float4*)(kc + (ulong)j * rs + (ulong)kvh * HD);
            float4 acc[G];
            for (int g = 0; g < G; g++) acc[g] = 0.f;
            for (uint d = 0; d < HD / 4; d++) {
                float4 k4 = kr[d];
                for (int g = 0; g < G; g++) acc[g] = fma(k4, ((threadgroup const float4*)(qs + g * HD))[d], acc[g]);
            }
            for (int g = 0; g < G; g++) s[g] = (acc[g].x + acc[g].y) + (acc[g].z + acc[g].w);
        }
        float e[G];
        for (int g = 0; g < G; g++) {
            float mn = max(m[g], simd_max(s[g]));
            float a = exp(m[g] - mn);
            e[g] = j < j1 ? exp(s[g] - mn) : 0.f;
            l[g] = l[g] * a + simd_sum(e[g]);
            for (int d = 0; d < D; d++) o[g][d] *= a;
            m[g] = mn;
        }
        uint nb = min(32u, j1 - base);
        device const float* vb = vc + (ulong)base * rs + (ulong)kvh * HD + lane * D;
        for (uint t = 0; t < nb; t++) {
            device const float* vr = vb + t * rs;
            float v[D];
            for (int d = 0; d < D; d++) v[d] = vr[d];
            for (int g = 0; g < G; g++) {
                float et = simd_shuffle(e[g], t);
                for (int d = 0; d < D; d++) o[g][d] = fma(et, v[d], o[g][d]);
            }
        }
    }
    // Combine the SIMD-groups, one head at a time.
    for (int g = 0; g < G; g++) {
        uint h = h0 + g;
        if (lane == 0) { sm[sg] = m[g]; sm[8 + sg] = l[g]; }
        for (int d = 0; d < D; d++) so[sg * HD + lane * D + d] = o[g][d];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float M = sm[0];
        for (uint s = 1; s < 8; s++) M = max(M, sm[s]);
        float L = 0.f;
        for (uint s = 0; s < 8; s++) L += sm[s] == -INFINITY ? 0.f : sm[8 + s] * exp(sm[s] - M);
        for (uint i = tid; i < HD; i += 256) {
            float acc = 0.f;
            for (uint s = 0; s < 8; s++) if (sm[s] != -INFINITY) acc += so[s * HD + i] * exp(sm[s] - M);
            if (p.nsplit > 1) {
                device float* pp = part + ((ulong)h * p.nsplit + split) * (HD + 2);
                pp[i] = acc;
                if (i == 0) { pp[HD] = M; pp[HD + 1] = L; }
                continue;
            }
            float r = acc / L;
            if (p.has_gate != 0) { float gt = gate[p.g_off + h * p.g_stride + i]; r *= 1.0f / (1.0f + exp(-gt)); }
            out[h * HD + i] = r;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
#define ATTN(D, G) \
kernel void dec_attn_##D##_##G(device const float* q [[buffer(0)]], device const float* kc [[buffer(1)]], device const float* vc [[buffer(2)]], \
                         device float* out [[buffer(3)]], device const float* gate [[buffer(4)]], constant AP& p [[buffer(5)]], \
                         device float* part [[buffer(6)]], \
                         uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float sm[16]; threadgroup float so[8 * D * 32]; threadgroup float qs[G * D * 32]; \
    attn_body<D, G>(q, kc, vc, out, gate, part, p, tg.x, tg.y, tid, sg, lane, sm, so, qs); }
ATTN(2, 1) ATTN(2, 2) ATTN(2, 4) ATTN(2, 8) ATTN(4, 1) ATTN(4, 2) ATTN(4, 4) ATTN(8, 1) ATTN(8, 2) ATTN(16, 1)
kernel void dec_attn_combine(device const float* part [[buffer(0)]], device float* out [[buffer(1)]], device const float* gate [[buffer(2)]],
                             constant AP& p [[buffer(3)]], constant uint& hd [[buffer(4)]],
                             uint h [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    device const float* ph = part + (ulong)h * p.nsplit * (hd + 2);
    float M = -INFINITY;
    for (uint s = 0; s < p.nsplit; s++) M = max(M, ph[s * (hd + 2) + hd]);
    float L = 0.f;
    for (uint s = 0; s < p.nsplit; s++) {
        float ms = ph[s * (hd + 2) + hd];
        if (ms != -INFINITY) L += ph[s * (hd + 2) + hd + 1] * exp(ms - M);
    }
    for (uint i = tid; i < hd; i += 256) {
        float acc = 0.f;
        for (uint s = 0; s < p.nsplit; s++) {
            float ms = ph[s * (hd + 2) + hd];
            if (ms != -INFINITY) acc += ph[s * (hd + 2) + i] * exp(ms - M);
        }
        float r = acc / L;
        if (p.has_gate != 0) { float g = gate[p.g_off + h * p.g_stride + i]; r *= 1.0f / (1.0f + exp(-g)); }
        out[h * hd + i] = r;
    }
}

// ---- Gated DeltaNet ------------------------------------------------------------------------------
// Causal depthwise conv (kernel kc) with SiLU, one thread per channel; the state [kc-1][dim] holds
// the earlier inputs (oldest first) and shifts in this one.
struct GC { uint dim; uint kc; };
kernel void dec_gdn_conv(device const float* x [[buffer(0)]], device float* out [[buffer(1)]], device float* st [[buffer(2)]],
                         device const float* w [[buffer(3)]], constant GC& p [[buffer(4)]], uint ch [[thread_position_in_grid]]) {
    if (ch >= p.dim) return;
    uint kc = p.kc;
    float xv = x[ch];
    float s = 0.f;
    for (uint j = 0; j + 1 < kc; j++) s += w[ch * kc + j] * st[j * p.dim + ch];
    s += w[ch * kc + kc - 1] * xv;
    out[ch] = s / (1.0f + exp(-s));
    for (uint j = 0; j + 2 < kc; j++) st[j * p.dim + ch] = st[(j + 1) * p.dim + ch];
    st[(kc - 2) * p.dim + ch] = xv;
}

// The gated delta rule for one token, one threadgroup per value head: thread (part, j) keeps rows
// 32 part .. 32 part + 31 of column j of the head's [hdk, hdv] state in registers (hdk / 32 parts,
// hdk and hdv multiples of 32, hdk / 32 * hdv <= 1024 threads), so the state is read and written
// once. Then the gated RMSNorm: out * rsqrt(mean(out^2) + eps) * w * silu(z).
struct GR { uint nk; uint nv; uint key_dim; uint tiled; float eps; float qscale; uint hdk; uint hdv; };
inline float tg_sum(float v, uint n, uint tid, uint sg, uint lane, threadgroup float* red) {
    // sum of v over threads 0 .. n - 1 (n a multiple of 32); every thread must call it
    if (tid < n) { float s = simd_sum(v); if (lane == 0) red[sg] = s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = 0.f;
    for (uint i = 0; i < n / 32; i++) t += red[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return t;
}
kernel void dec_gdn_rule(device const float* c [[buffer(0)]], device const float* z [[buffer(1)]], device const float* b [[buffer(2)]],
                         device const float* a [[buffer(3)]], device float* out [[buffer(4)]], device float* state [[buffer(5)]],
                         device const float* cst [[buffer(6)]], constant GR& p [[buffer(7)]],
                         uint hv [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    // cst: decay_coef[nv], dt_bias[nv], norm_w[hdv]
    threadgroup float qs[256], ks[256], red[1024], red2[32];
    uint hdk = p.hdk, hdv = p.hdv, np = hdk / 32;
    uint hk = p.tiled != 0 ? hv % p.nk : hv / (p.nv / p.nk);
    uint part = tid / hdv, j = tid % hdv;
    float q = 0.f, k = 0.f;
    if (tid < hdk) {
        q = c[hk * hdk + tid];
        k = c[p.key_dim + hk * hdk + tid];
        qs[tid] = q;
        ks[tid] = k;
    }
    float qinv = 1.0f / sqrt(tg_sum(q * q, hdk, tid, sg, lane, red2) + 1e-6f) * p.qscale;
    float kinv = 1.0f / sqrt(tg_sum(k * k, hdk, tid, sg, lane, red2) + 1e-6f);
    float beta = 1.0f / (1.0f + exp(-b[hv]));
    float ga = a[hv] + cst[p.nv + hv];
    float sp = ga > 20.f ? ga : log(1.0f + exp(ga));
    float decay = exp(cst[hv] * sp);
    device float* S = state + (ulong)hv * hdk * hdv + part * 32 * hdv + j;
    float s[32];
    float kv = 0.f;
    for (uint d = 0; d < 32; d++) {
        s[d] = S[d * hdv] * decay;
        kv += s[d] * (ks[part * 32 + d] * kinv);
    }
    red[part * hdv + j] = kv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float kvm = 0.f;
    for (uint i = 0; i < np; i++) kvm += red[i * hdv + j];
    float delta = (c[2 * p.key_dim + hv * hdv + j] - kvm) * beta;
    float o = 0.f;
    for (uint d = 0; d < 32; d++) {
        s[d] += (ks[part * 32 + d] * kinv) * delta;
        o += s[d] * (qs[part * 32 + d] * qinv);
        S[d * hdv] = s[d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    red[part * hdv + j] = o;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ov = 0.f;
    if (tid < hdv)
        for (uint i = 0; i < np; i++) ov += red[i * hdv + j];
    float r = rsqrt(tg_sum(ov * ov, hdv, tid, sg, lane, red2) / float(hdv) + p.eps);
    if (tid < hdv) {
        float zz = z[hv * hdv + j];
        out[hv * hdv + j] = ov * r * cst[2 * p.nv + j] * (zz / (1.0f + exp(-zz)));
    }
}

// ---- out = act(g) * u (SiLU, or GELU with the tanh approximation).
kernel void dec_glu(device const float* g [[buffer(0)]], device const float* u [[buffer(1)]], device float* out [[buffer(2)]],
                    constant uint2& p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.x) return;
    float a = g[i];
    float act;
    if (p.y != 0) {
        float t = 0.7978845608f * (a + 0.044715f * a * a * a);
        act = 0.5f * a * (1.0f + precise::tanh(t));
    } else {
        act = a / (1.0f + exp(-a));
    }
    out[i] = act * u[i];
}

// ---- MoE experts of one token -------------------------------------------------------------------
// Expert e's weights are at the addresses tab[e] (data, scales, affine offsets). `sel` holds the k
// chosen expert ids (as uint bits) then their weights. Pair kernels: threadgroup (row, slot s)
// computes act[s] = silu(gate_e x) * (up_e x) (fused gate / up tables: up rows after r2off rows of
// the same matrix); sum kernels: y += sum over slots of weight_s * (down_e act[s]).
struct EPX { device const uchar* q; device const uchar* s; device const uchar* o; device const uchar* _p; };
#define MVX_SIG(NAME) \
kernel void NAME(device const EPX* tab [[buffer(0)]], device const EPX* tab2 [[buffer(1)]], device const float* x [[buffer(2)]], \
                 device float* y [[buffer(3)]], constant MV& p [[buffer(4)]], device const float* sel [[buffer(5)]], \
                 uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], \
                 uint lane [[thread_index_in_simdgroup]])
#define MVX(NAME, T, X, CH, MAKE) \
MVX_SIG(NAME##_pair) { threadgroup float red[2]; \
    uint s = tg.y; EPX e = tab[as_type<uint>(sel[s])], e2 = tab2[as_type<uint>(sel[s])]; \
    float acc[1] = { 0.f }, acc2[1] = { 0.f }; \
    T f = MAKE(e); T f2 = MAKE(e2); \
    mv_acc<T, X, CH, 1, 1>(f, f2, true, p._p, x, p.n_in, p.n_out, tg.x, sg, lane, acc, acc2, 1.f); \
    mv_finish<1, 1>(acc, acc2, y + (ulong)s * p.n_out, p.mode, tg.x, p.n_out, sg, lane, red, sel); } \
MVX_SIG(NAME##_sum) { threadgroup float red[2]; \
    float acc[1] = { 0.f }, acc2[1] = { 0.f }; \
    for (uint s = 0; s < p.ldy; s++) { \
        EPX e = tab[as_type<uint>(sel[s])]; T f = MAKE(e); \
        mv_acc<T, X, CH, 1, 1>(f, f, false, 0, x + (ulong)s * p.n_in, p.n_in, p.n_out, tg.x, sg, lane, acc, acc2, sel[p.ldy + s]); \
    } \
    mv_finish<1, 1>(acc, acc2, y, p.mode, tg.x, p.n_out, sg, lane, red, sel); }
#define MK_RAWX(T) T{ e.q, p.gpr }
#define MKX_Q8(e) MQ8{ (device const char*)(e).q, (device const float*)(e).s, p.n_in }
#define MKX_Q4(e) MQ4{ (e).q, (device const half*)(e).s, p.n_in, (device const ushort*)(e).o, p.gpr, p.gk }

// Top k of n router logits (ties to the lower index), softmax over the chosen: ids (as uint bits)
// then weights into `sel`. One threadgroup, one logit per thread (n <= 1024).
// p.z != 0: the weights are then multiplied by scale[id].
kernel void dec_topk(device const float* logits [[buffer(0)]], device float* sel [[buffer(1)]], constant uint4& p [[buffer(2)]],
                     device const float* scale [[buffer(3)]],
                     uint tid [[thread_index_in_threadgroup]], uint nt [[threads_per_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float bv[32];
    threadgroup uint bi[32];
    threadgroup float chosen[64];
    uint n = p.x, k = p.y;
    float v = tid < n ? logits[tid] : -INFINITY;
    uint nsg = (nt + 31) / 32;
    for (uint j = 0; j < k; j++) {
        // max value, then the lowest index holding it
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

// ---- Elementwise: p.x = 0 scale (a * p.y), 1 softcap (tanh(a / p.y) * p.y), 2 combine
// ((a * p.y + b * p.z) * p.w).
kernel void dec_ew(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* out [[buffer(2)]],
                   constant float4& p [[buffer(3)]], constant uint& n [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float v = a[i];
    if (p.x == 0.f) out[i] = v * p.y;
    else if (p.x == 1.f) out[i] = precise::tanh(v / p.y) * p.y;
    else out[i] = (v * p.y + b[i] * p.z) * p.w;
}

// ---- x = (x + rmsnorm(y) * wy) * scale, then optionally out = rmsnorm(x) * wn; one threadgroup,
// up to 8 values per thread kept in registers (n <= 8 * threads).
struct AN { uint n; float eps; float scale; uint has_next; };
inline float tg_total(float s, uint nt, uint sg, uint lane, threadgroup float* red) {
    s = simd_sum(s);
    uint nsg = (nt + 31) / 32;
    if (nsg > 1) {
        if (lane == 0) red[sg] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s = 0.f;
        for (uint i = 0; i < nsg; i++) s += red[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return s;
}
kernel void dec_add_norm(device const float* y [[buffer(0)]], device float* x [[buffer(1)]], device float* out [[buffer(2)]],
                         device const float* wy [[buffer(3)]], device const float* wn [[buffer(4)]], constant AN& p [[buffer(5)]],
                         uint tid [[thread_index_in_threadgroup]], uint nt [[threads_per_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[32];
    float yv[8], xv[8];
    float s = 0.f;
    for (uint j = 0; j < 8; j++) {
        uint i = tid + j * nt;
        yv[j] = i < p.n ? y[i] : 0.f;
        xv[j] = i < p.n ? x[i] : 0.f;
        s += yv[j] * yv[j];
    }
    float k = rsqrt(tg_total(s, nt, sg, lane, red) / float(p.n) + p.eps);
    float s2 = 0.f;
    for (uint j = 0; j < 8; j++) {
        uint i = tid + j * nt;
        if (i < p.n) {
            xv[j] = (xv[j] + yv[j] * k * wy[i]) * p.scale;
            x[i] = xv[j];
            s2 += xv[j] * xv[j];
        }
    }
    if (p.has_next == 0) return;
    float k2 = rsqrt(tg_total(s2, nt, sg, lane, red) / float(p.n) + p.eps);
    for (uint j = 0; j < 8; j++) {
        uint i = tid + j * nt;
        if (i < p.n) out[i] = xv[j] * k2 * wn[i];
    }
}

// ---- Interventions (crate::intervene) -------------------------------------------------------
// Residual hooks of one row x[n] (one threadgroup): `hot` holds the n_add add coefficients (padded
// to 4) then 8 floats per item (bias, threshold, act: 0 linear / 1 relu / 2 jumprelu, op: 0 set /
// 1 scale, value, enabled); `cold` the n_add add vectors, then the items' encoder rows, then their
// decoder rows. x += sum c_a add_a first, then every item reads that x: a_j = act(enc_j . x +
// bias_j), x += sum (a'_j - a_j) dec_j.
kernel void dec_steer(device float* x [[buffer(0)]], device const float* hot [[buffer(1)]], device const float* cold [[buffer(2)]],
                      constant uint4& p [[buffer(3)]], uint tid [[thread_index_in_threadgroup]], uint nt [[threads_per_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[32];
    threadgroup float coef[64];
    uint n = p.x, n_add = p.y, n_items = p.z;
    device const float* items = hot + ((n_add + 3) & ~3u);
    // The added vectors first (the edits then read the result).
    if (n_add > 0) {
        for (uint i = tid; i < n; i += nt) {
            float v = x[i];
            for (uint a = 0; a < n_add; a++) v += hot[a] * cold[a * n + i];
            x[i] = v;
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
    for (uint j = 0; j < n_items; j++) {
        device const float* e = cold + (n_add + j) * n;
        float s = 0.f;
        for (uint i = tid; i < n; i += nt) s += e[i] * x[i];
        s = tg_total(s, nt, sg, lane, red);
        if (tid == 0) {
            device const float* q = items + j * 8;
            float pre = s + q[0];
            float a = q[2] == 0.f ? pre : (q[2] == 1.f ? max(pre, 0.f) : (pre > q[1] ? pre : 0.f));
            float a2 = q[3] == 0.f ? q[4] : a * q[4];
            coef[j] = q[5] != 0.f ? a2 - a : 0.f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const float* dec = cold + (n_add + n_items) * n;
    if (n_items == 0) return;
    for (uint i = tid; i < n; i += nt) {
        float v = x[i];
        for (uint j = 0; j < n_items; j++) v += coef[j] * dec[j * n + i];
        x[i] = v;
    }
}

// Neuron edits: entry i (8 floats: expert id or -1, neuron, op: 0 scale / 1 add / 2 set / 3 clamp,
// a, b, enabled) on act; with experts (p.w = 1), act is [k, inter] for the k experts in sel.
kernel void dec_neuron_edit(device float* act [[buffer(0)]], device const float* hot [[buffer(1)]], device const float* sel [[buffer(2)]],
                            constant uint4& p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.x) return;
    device const float* q = hot + i * 8;
    if (q[5] == 0.f) return;
    uint idx = uint(q[1]);
    uint slots = p.w != 0 ? p.z : 1;
    for (uint s = 0; s < slots; s++) {
        if (p.w != 0 && float(as_type<uint>(sel[s])) != q[0]) continue;
        device float* a = act + s * p.y + idx;
        float v = *a;
        if (q[2] == 0.f) v = v * q[3];
        else if (q[2] == 1.f) v = v + q[3];
        else if (q[2] == 2.f) v = q[3];
        else v = min(max(v, q[3]), q[4]);
        *a = v;
    }
}
