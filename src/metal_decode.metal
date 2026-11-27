// Single-token decode kernels (see metal_decode.rs). Appended to the library after SHADER_SRC and
// GGUF_SHADER_SRC, whose dequantizers (`load`, `h2f`, ...) the generic matvec reuses.
//
// No reassociation in the kernels decoding plans use: the multi-token kernels (speculative
// verification) must compute every token exactly as the single-token ones, whatever the loop
// structure around the same expressions. The few-rows matvecs and the MoE expert kernels, outside
// the plans' multi-token steps, keep the default (fast) math.
#pragma clang fp reassociate(off)

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

// Nibble products: `lo` gets x[4k+j] * low nibble j, `hi` x2[4k+j] * high nibble j, the four
// nibbles of a word unpacked as bytes. (These kernels once multiplied by the masked nibbles in
// place, the value times 2^(8j) or 2^(8j+4), with the powers divided out by the NIB_* weights of
// the final dot: the same products and sums scaled by exact powers of two, so the results are
// bit for bit those of the byte unpacking, which takes fewer instructions.)
inline void nib_acc(uint w, float4 xa, float4 xb, thread float4& lo, thread float4& hi) {
    lo += xa * float4(as_type<uchar4>(w & 0x0F0F0F0Fu));
    hi += xb * float4(as_type<uchar4>((w >> 4) & 0x0F0F0F0Fu));
}
constant float4 NIB_LO = float4(1.f);
constant float4 NIB_HI = float4(1.f);

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
// Routed experts in the grouped GEMMs' tiled layout (`gblk` / `kpart` in GGUF_SHADER_SRC,
// `MetalBackend::tiled_experts`): the `<type>T` twins of the raw types below read the same bytes
// at their tiled places (`rb`: the row length without `GG_TILED`) and compute the same products
// in the same order. Block `blk` (of `bs` bytes) of row `row`, and the part of `len` bytes `off`
// bytes into super-block `sb` (of `tsz` bytes) of a K-quant row.
inline device const uchar* tblk(device const uchar* w, ulong rb, uint row, uint blk, uint bs) {
    return w + (ulong)(row & ~63u) * rb + (ulong)blk * (64 * bs) + (row & 63u) * bs;
}
inline device const uchar* tpart(device const uchar* w, ulong rb, uint row, uint sb, uint tsz, uint off, uint len) {
    return w + (ulong)(row & ~63u) * rb + (ulong)sb * (64 * tsz) + 64 * off + (row & 63u) * len;
}
inline float f16_dot(device const packed_half4* p, thread const X16& xr) {
    float4 a = float4(half4(p[0])) * xr.v[0];
    a = fma(float4(half4(p[1])), xr.v[1], a);
    a = fma(float4(half4(p[2])), xr.v[2], a);
    a = fma(float4(half4(p[3])), xr.v[3], a);
    return hsum(a);
}
inline float mdot(MF16 f, uint row, uint c, thread const X16& xr) {
    return f16_dot((device const packed_half4*)(f.w + (ulong)row * f.rb + c * 32), xr);
}
// (Tiled F16 blocks are of 32 values: two chunks.)
struct MF16T { device const uchar* w; ulong rb; };
inline float mdot(MF16T f, uint row, uint c, thread const X16& xr) {
    return f16_dot((device const packed_half4*)(tblk(f.w, f.rb, row, c / 2, 64) + (c % 2) * 32), xr);
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
// The block's value d (q - 8) . x plus the affine offset o times the sum of x, as one fused
// multiply-add on d (q - 8) . x, written out with contraction off: left to the compiler, the
// multi-token kernels fused the other product in some contexts (verification passes differed
// from one-token passes in the last bits on MLX 4-bit weights).
inline float q4_affine(float d, float qd, float xs, float o) {
#pragma clang fp contract(off)
    return fma(o, xs, d * fma(-8.f, xs, qd));
}
inline float mdot(MQ4 f, uint row, uint c, thread const X32& xr) {
    uint4 u = *(device const uint4*)(f.q + (ulong)row * (f.n_in / 2) + c * 16);
    float d = float(f.s[(ulong)row * (f.n_in / 32) + c]);
    if (f.gk == 0) return d * (q4_block(u, xr) - 8.f * xsum32(xr));
    return q4_affine(d, q4_block(u, xr), xsum32(xr), as_type<float>(uint(f.o[(ulong)row * f.gpr + c / f.gk]) << 16));
}
// ---- GGUF Q4_0: 18-byte blocks (2-byte aligned), chunk of 32.
struct MQ4_0 { device const uchar* w; ulong rb; };
inline uint ld32u(device const ushort* p) { return uint(p[0]) | (uint(p[1]) << 16); }
inline float q4_0_dot(device const ushort* b, thread const X32& xr) {
    float d = float(as_type<half>(b[0]));
    uint4 u = uint4(ld32u(b + 1), ld32u(b + 3), ld32u(b + 5), ld32u(b + 7));
    return d * (q4_block(u, xr) - 8.f * xsum32(xr));
}
inline float mdot(MQ4_0 f, uint row, uint c, thread const X32& xr) { return q4_0_dot((device const ushort*)(f.w + (ulong)row * f.rb + c * 18), xr); }
struct MQ4_0T { device const uchar* w; ulong rb; };
inline float mdot(MQ4_0T f, uint row, uint c, thread const X32& xr) { return q4_0_dot((device const ushort*)tblk(f.w, f.rb, row, c, 18), xr); }
// ---- GGUF Q8_0: 34-byte blocks (2-byte aligned), chunk of 32.
struct MQ8_0 { device const uchar* w; ulong rb; };
inline float4 c4(ushort a, ushort b) { return float4(float(as_type<char2>(a).x), float(as_type<char2>(a).y), float(as_type<char2>(b).x), float(as_type<char2>(b).y)); }
inline float q8_0_dot(device const ushort* b, thread const X32& xr) {
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
inline float mdot(MQ8_0 f, uint row, uint c, thread const X32& xr) { return q8_0_dot((device const ushort*)(f.w + (ulong)row * f.rb + c * 34), xr); }
struct MQ8_0T { device const uchar* w; ulong rb; };
inline float mdot(MQ8_0T f, uint row, uint c, thread const X32& xr) { return q8_0_dot((device const ushort*)tblk(f.w, f.rb, row, c, 34), xr); }
// ---- Ternary blocks with one f16 scale, chunk of 32: Q2_0 (18-byte blocks of 64), PrismML's
// PQ2_0 (34 bytes of 128: Q2_0's codes) and PTQ1_0 (28 bytes of 128: five trits a byte). The
// codes c of a chunk are summed against x and the scale applied once: d (sum c x - sum x).
inline float4 t2x4(uint v) { return float4(uint4(v, v >> 2, v >> 4, v >> 6) & 3u); }
// The 8 code bytes of a Q2_0 / PQ2_0 chunk (4 ushorts) against x.
inline float t2_dot(device const ushort* q, thread const X32& xr) {
    float4 a = float4(0.f);
    for (uint i = 0; i < 4; i++) {
        uint u = q[i];
        a = fma(t2x4(u & 255u), xr.v[2 * i], a);
        a = fma(t2x4(u >> 8), xr.v[2 * i + 1], a);
    }
    return hsum(a) - xsum32(xr);
}
struct MQ2_0 { device const uchar* w; ulong rb; };
inline float mdot(MQ2_0 f, uint row, uint c, thread const X32& xr) {
    device const ushort* b = (device const ushort*)(f.w + (ulong)row * f.rb + (c / 2) * 18);
    return float(as_type<half>(b[0])) * t2_dot(b + 1 + 4 * (c % 2), xr);
}
struct MPQ2_0 { device const uchar* w; ulong rb; };
inline float mdot(MPQ2_0 f, uint row, uint c, thread const X32& xr) {
    device const ushort* b = (device const ushort*)(f.w + (ulong)row * f.rb + (c / 4) * 34);
    return float(as_type<half>(b[0])) * t2_dot(b + 1 + 4 * (c % 4), xr);
}
// Base-3 digits (3^n in p) of the four bytes of w, as ggml reads TQ1_0 / PTQ1_0 bytes (not minus one).
inline float4 trit4(uint w, uint4 p) { return float4(((((uint4(as_type<uchar4>(w)) * p) & 255u) * 3u) >> 8)); }
inline float4 trit4(uint w, uint p) { return trit4(w, uint4(p)); }
// PTQ1_0 chunk j (values 32 j .. 32 j + 31, four at a time as x.v[k]) reads the block's word
// wi (6: the two qh bytes, repeated) and from each byte the base-3 digit e: chunks 0 and 1 digits
// 2 j and 2 j + 1 of the first run's 16 bytes (words 0..3); chunk 2 digit 4 of the first run, then
// digits 0 and 1 of the second run's 8 bytes (words 4, 5); chunk 3 digits 2, 3, 4 of the second
// run, then digit n of qh byte h as value 120 + 2 n + h. Computed with selects rather than branches
// or tables: the lanes of a SIMD-group read different chunks.
inline uint pow3(uint e) { return e == 0 ? 1u : e == 1 ? 3u : e == 2 ? 9u : e == 3 ? 27u : 81u; }
struct MPTQ1_0 { device const uchar* w; ulong rb; };
inline float mdot(MPTQ1_0 f, uint row, uint c, thread const X32& xr) {
    device const uint* b = (device const uint*)(f.w + (ulong)row * f.rb + (c / 4) * 28);
    uint j = c % 4;
    bool lo = j < 2, j2 = j == 2;
    float4 a = float4(0.f);
    for (uint k = 0; k < 4; k++) {
        uint w = b[j == 3 ? 4 + (k & 1) : k];
        a = fma(trit4(w, pow3(lo ? 2 * j : j2 ? 4u : 2 + k / 2)), xr.v[k], a);
    }
    for (uint k = 4; k < 6; k++) {
        uint w = b[lo ? k - 4 : 4 + (k & 1)];
        a = fma(trit4(w, pow3(lo ? 2 * j + 1 : j2 ? 0u : 4u)), xr.v[k], a);
    }
    for (uint k = 6; k < 8; k++) {
        uint w = b[lo ? k - 4 : j2 ? 4 + (k & 1) : 6u];
        uint h = (w & 0xFFFFu) * 0x10001u;
        uint4 ph = k == 6 ? uint4(1u, 1u, 3u, 3u) : uint4(9u, 9u, 27u, 27u);
        uint4 p = j == 3 ? ph : uint4(pow3(lo ? 2 * j + 1 : 1u));
        a = fma(trit4(j == 3 ? h : w, p), xr.v[k], a);
    }
    float d = float(as_type<half>(ushort(b[6] >> 16)));
    return d * (hsum(a) - xsum32(xr));
}
// ---- IQ1_S (50-byte blocks of 256, 2-byte aligned), chunk of 32 = one sub-block: four grid
// entries of 8 ternary values (signed bytes) plus a shared delta, one scale. The delta is applied
// to the sum of the x values rather than to each weight. The chunk is decoded (`iq1s_dec`) and its
// sums taken (`iq1s_fin`, contraction off) by the same functions in the one-token and the
// multi-token kernels, so that a verification pass gives the one-token bits.
struct MIQ1_S { device const uchar* w; ulong rb; };
struct WIQ1S { float4 g[8]; float dl; float delta; };
inline void iq1s_dec(MIQ1_S f, uint row, uint c, thread WIQ1S& w) {
    device const ushort* B = (device const ushort*)(f.w + (ulong)row * f.rb + (c / 8) * 50);
    uint j = c % 8;
    uint qh = B[17 + j], q01 = B[1 + 2 * j], q23 = B[2 + 2 * j];
    w.dl = float(as_type<half>(B[0])) * float(2 * ((qh >> 12) & 7) + 1);
    w.delta = (qh & 0x8000) ? -0.125f : 0.125f;
    uint idx[4] = { (q01 & 255) | ((qh << 8) & 0x700), (q01 >> 8) | ((qh << 5) & 0x700),
                    (q23 & 255) | ((qh << 2) & 0x700), (q23 >> 8) | ((qh >> 1) & 0x700) };
    for (uint l = 0; l < 4; l++) {
        ulong g = iq1s_grid[idx[l]];
        w.g[2 * l] = float4(as_type<char4>(uint(g)));
        w.g[2 * l + 1] = float4(as_type<char4>(uint(g >> 32)));
    }
}
// The sums of one group of 8 (two float4 of grid values), then of the chunk; every product and
// sum explicit (contraction off), so any kernel shape computes the same bits.
inline void iq1s_acc(float4 g0, float4 g1, float4 x0, float4 x1, thread float4& s, thread float4& sx) {
#pragma clang fp contract(off)
    s = fma(g0, x0, s);
    s = fma(g1, x1, s);
    sx = sx + (x0 + x1);
}
inline float iq1s_end(float dl, float delta, float4 s, float4 sx) {
#pragma clang fp contract(off)
    return dl * fma(delta, hsum(sx), hsum(s));
}
inline float iq1s_fin(thread const WIQ1S& w, thread const X32& x) {
    float4 s = 0.f, sx = 0.f;
    for (uint l = 0; l < 4; l++) iq1s_acc(w.g[2 * l], w.g[2 * l + 1], x.v[2 * l], x.v[2 * l + 1], s, sx);
    return iq1s_end(w.dl, w.delta, s, sx);
}
// (The one-token dot decodes each grid entry as it goes: the same operations as `iq1s_fin` on
// `iq1s_dec`'s values, with fewer registers live.)
inline float iq1s_dot(device const ushort* B, uint c, thread const X32& x) {
    uint j = c % 8;
    uint qh = B[17 + j], q01 = B[1 + 2 * j], q23 = B[2 + 2 * j];
    float dl = float(as_type<half>(B[0])) * float(2 * ((qh >> 12) & 7) + 1);
    float delta = (qh & 0x8000) ? -0.125f : 0.125f;
    uint idx[4] = { (q01 & 255) | ((qh << 8) & 0x700), (q01 >> 8) | ((qh << 5) & 0x700),
                    (q23 & 255) | ((qh << 2) & 0x700), (q23 >> 8) | ((qh >> 1) & 0x700) };
    float4 s = 0.f, sx = 0.f;
    for (uint l = 0; l < 4; l++) {
        ulong g = iq1s_grid[idx[l]];
        iq1s_acc(float4(as_type<char4>(uint(g))), float4(as_type<char4>(uint(g >> 32))), x.v[2 * l], x.v[2 * l + 1], s, sx);
    }
    return iq1s_end(dl, delta, s, sx);
}
inline float mdot(MIQ1_S f, uint row, uint c, thread const X32& x) {
    return iq1s_dot((device const ushort*)(f.w + (ulong)row * f.rb + (c / 8) * 50), c, x);
}
struct MIQ1_ST { device const uchar* w; ulong rb; };
inline float mdot(MIQ1_ST f, uint row, uint c, thread const X32& x) { return iq1s_dot((device const ushort*)tblk(f.w, f.rb, row, c / 8, 50), c, x); }
// ---- IQ1_M (56-byte blocks of 256): per group of 8 its own scale (3 bits a half sub-block) and
// delta sign; the f16 scale in the top nibbles of the four scale words. Decoded and summed as
// IQ1_S (`iq1m_dec`, `iq1m_fin`).
struct MIQ1_M { device const uchar* w; ulong rb; };
struct WIQ1M { float4 g[8]; float4 dl; float4 delta; };
inline void iq1m_dec(MIQ1_M f, uint row, uint c, thread WIQ1M& w) {
    device const uchar* B = f.w + (ulong)row * f.rb + (c / 8) * 56;
    uint j = c % 8;
    device const ushort* S = (device const ushort*)(B + 48);
    uint s0 = S[0], s1 = S[1], s2 = S[2], s3 = S[3];
    float d = float(as_type<half>(ushort((s0 >> 12) | ((s1 >> 8) & 0x00f0) | ((s2 >> 4) & 0x0f00) | (s3 & 0xf000))));
    uint sv = S[j / 2] >> (6 * (j % 2));
    device const ushort* Q = (device const ushort*)B;
    uint q01 = Q[2 * j], q23 = Q[2 * j + 1];
    uint qh = Q[16 + j];
    for (uint l = 0; l < 4; l++) {
        uint hb = (qh >> (8 * (l / 2))) & 255;
        uint q = ((l < 2 ? q01 : q23) >> (8 * (l % 2))) & 255;
        uint idx = q | ((hb << (l % 2 ? 4 : 8)) & 0x700);
        w.dl[l] = d * float(2 * ((sv >> (l < 2 ? 0 : 3)) & 7) + 1);
        w.delta[l] = (hb & (l % 2 ? 0x80u : 0x08u)) ? -0.125f : 0.125f;
        ulong g = iq1s_grid[idx];
        w.g[2 * l] = float4(as_type<char4>(uint(g)));
        w.g[2 * l + 1] = float4(as_type<char4>(uint(g >> 32)));
    }
}
// One group of 8 added to the chunk's sum `r`, every operation explicit (contraction off).
inline float iq1m_grp(float r, float4 g0, float4 g1, float4 x0, float4 x1, float dl, float delta) {
#pragma clang fp contract(off)
    float4 a = g0 * x0;
    a = fma(g1, x1, a);
    return fma(dl, fma(delta, hsum(x0 + x1), hsum(a)), r);
}
inline float iq1m_fin(thread const WIQ1M& w, thread const X32& x) {
    float r = 0.f;
    for (uint l = 0; l < 4; l++) r = iq1m_grp(r, w.g[2 * l], w.g[2 * l + 1], x.v[2 * l], x.v[2 * l + 1], w.dl[l], w.delta[l]);
    return r;
}
// (As for IQ1_S: `iq1m_fin`'s operations on `iq1m_dec`'s values, decoded as they are used.)
inline float iq1m_dot(device const uchar* B, uint c, thread const X32& x) {
    uint j = c % 8;
    device const ushort* S = (device const ushort*)(B + 48);
    uint s0 = S[0], s1 = S[1], s2 = S[2], s3 = S[3];
    float d = float(as_type<half>(ushort((s0 >> 12) | ((s1 >> 8) & 0x00f0) | ((s2 >> 4) & 0x0f00) | (s3 & 0xf000))));
    uint sv = S[j / 2] >> (6 * (j % 2));
    device const ushort* Q = (device const ushort*)B;
    uint q01 = Q[2 * j], q23 = Q[2 * j + 1];
    uint qh = Q[16 + j];
    float r = 0.f;
    for (uint l = 0; l < 4; l++) {
        uint hb = (qh >> (8 * (l / 2))) & 255;
        uint q = ((l < 2 ? q01 : q23) >> (8 * (l % 2))) & 255;
        uint idx = q | ((hb << (l % 2 ? 4 : 8)) & 0x700);
        float dl = d * float(2 * ((sv >> (l < 2 ? 0 : 3)) & 7) + 1);
        float delta = (hb & (l % 2 ? 0x80u : 0x08u)) ? -0.125f : 0.125f;
        ulong g = iq1s_grid[idx];
        r = iq1m_grp(r, float4(as_type<char4>(uint(g))), float4(as_type<char4>(uint(g >> 32))), x.v[2 * l], x.v[2 * l + 1], dl, delta);
    }
    return r;
}
inline float mdot(MIQ1_M f, uint row, uint c, thread const X32& x) { return iq1m_dot(f.w + (ulong)row * f.rb + (c / 8) * 56, c, x); }
struct MIQ1_MT { device const uchar* w; ulong rb; };
inline float mdot(MIQ1_MT f, uint row, uint c, thread const X32& x) { return iq1m_dot(tblk(f.w, f.rb, row, c / 8, 56), c, x); }
// ---- The routed experts' IQ1_S / IQ1_M dots (`mvxd_*`; one-token only, so free of the sharing
// above): the grid entries from `iq1s_grid_h`, the same ternary values as halves (+-1 = 0x3c00 /
// 0xbc00, a 16-byte row per entry, prepended by `decode_src`), which the multiply-adds take
// without a conversion instruction (the kernels are bound by instructions, not memory: 2304 x
// 5120 gate / up pairs 344 -> ~470 GB/s); x's sums over each group of 8 (for the deltas) formed
// when x is loaded (`X32S`), once for the rows that share the load.
struct X32S { float4 v[8]; float4 s; float t; };
inline void xload(device const float* x, uint c, thread X32S& xr) {
    device const float4* p = (device const float4*)(x + c * 32);
    for (uint i = 0; i < 8; i++) xr.v[i] = p[i];
    for (uint l = 0; l < 4; l++) xr.s[l] = hsum(xr.v[2 * l] + xr.v[2 * l + 1]);
    xr.t = (xr.s.x + xr.s.y) + (xr.s.z + xr.s.w);
}
inline float4 gh_lo(uint4 g) { return float4(as_type<half4>(g.xy)); }
inline float4 gh_hi(uint4 g) { return float4(as_type<half4>(g.zw)); }
struct MXIQ1_S { device const uchar* w; ulong rb; };
inline float mdot(MXIQ1_S f, uint row, uint c, thread const X32S& x) {
    device const ushort* B = (device const ushort*)(f.w + (ulong)row * f.rb + (c / 8) * 50);
    uint j = c % 8;
    uint qh = B[17 + j], q01 = B[1 + 2 * j], q23 = B[2 + 2 * j];
    float dl = float(as_type<half>(B[0])) * float(2 * ((qh >> 12) & 7) + 1);
    float delta = (qh & 0x8000) ? -0.125f : 0.125f;
    uint idx[4] = { (q01 & 255) | ((qh << 8) & 0x700), (q01 >> 8) | ((qh << 5) & 0x700),
                    (q23 & 255) | ((qh << 2) & 0x700), (q23 >> 8) | ((qh >> 1) & 0x700) };
    float4 s = 0.f;
    for (uint l = 0; l < 4; l++) {
        uint4 g = iq1s_grid_h[idx[l]];
        s = fma(gh_lo(g), x.v[2 * l], s);
        s = fma(gh_hi(g), x.v[2 * l + 1], s);
    }
    return dl * fma(delta, x.t, hsum(s));
}
// IQ1_M: groups 0-1 and 2-3 of the chunk share a scale; group l's delta sign is bit 4l + 3 of qh.
struct MXIQ1_M { device const uchar* w; ulong rb; };
inline float mdot(MXIQ1_M f, uint row, uint c, thread const X32S& x) {
    device const uchar* B = f.w + (ulong)row * f.rb + (c / 8) * 56;
    uint j = c % 8;
    device const ushort* S = (device const ushort*)(B + 48);
    uint s0 = S[0], s1 = S[1], s2 = S[2], s3 = S[3];
    float d = float(as_type<half>(ushort((s0 >> 12) | ((s1 >> 8) & 0x00f0) | ((s2 >> 4) & 0x0f00) | (s3 & 0xf000))));
    uint sv = S[j / 2] >> (6 * (j % 2));
    device const ushort* Q = (device const ushort*)B;
    uint q01 = Q[2 * j], q23 = Q[2 * j + 1];
    uint qh = Q[16 + j];
    uint4 g0 = iq1s_grid_h[(q01 & 255) | ((qh << 8) & 0x700)];
    uint4 g1 = iq1s_grid_h[(q01 >> 8) | ((qh << 4) & 0x700)];
    uint4 g2 = iq1s_grid_h[(q23 & 255) | (qh & 0x700)];
    uint4 g3 = iq1s_grid_h[(q23 >> 8) | ((qh >> 4) & 0x700)];
    float4 a = gh_lo(g0) * x.v[0];
    a = fma(gh_hi(g0), x.v[1], a);
    a = fma(gh_lo(g1), x.v[2], a);
    a = fma(gh_hi(g1), x.v[3], a);
    float4 b = gh_lo(g2) * x.v[4];
    b = fma(gh_hi(g2), x.v[5], b);
    b = fma(gh_lo(g3), x.v[6], b);
    b = fma(gh_hi(g3), x.v[7], b);
    float4 ds = select(float4(0.125f), float4(-0.125f), (uint4(qh) & uint4(0x8, 0x80, 0x800, 0x8000)) != 0u) * x.s;
    float r0 = hsum(a) + (ds.x + ds.y), r1 = hsum(b) + (ds.z + ds.w);
    return d * fma(float(2 * (sv & 7) + 1), r0, float(2 * ((sv >> 3) & 7) + 1) * r1);
}
// ---- The routed experts' IQ2_XXS (66-byte blocks of 256, 2-byte aligned) and IQ2_S (82 bytes)
// dots, chunk of 32 = one sub-block: four grid entries of 8 magnitudes (8, 25 or 43) with a sign
// bit each, one scale (two for IQ2_S). The block read as 16-bit words (the GEMM loader reads
// bytes), the signs applied to the magnitude bytes (`sgn4`), the scale to the chunk's sum. (Grids
// of halves as for IQ1, with the signs as half sign masks from a table: no faster.)
// Four magnitude bytes b, negated where bits 0-3 of s are set, as floats: m = 0xff in the bytes
// to negate, (b ^ m) + 1 = 256 - b there (b > 0: no carry into the next byte).
inline float4 sgn4(uint b, uint s) {
    uint m = ((s & 15u) * 0x00204081u) & 0x01010101u;
    return float4(as_type<char4>((b ^ (m * 255u)) + m));
}
struct MXIQ2_XXS { device const uchar* w; ulong rb; };
inline float mdot(MXIQ2_XXS f, uint row, uint c, thread const X32& x) {
    device const ushort* B = (device const ushort*)(f.w + (ulong)row * f.rb + (c / 8) * 66);
    uint j = c % 8;
    uint a0 = uint(B[1 + 4 * j]) | (uint(B[2 + 4 * j]) << 16), a1 = uint(B[3 + 4 * j]) | (uint(B[4 + 4 * j]) << 16);
    float db = float(as_type<half>(B[0])) * (0.5f + float(a1 >> 28)) * 0.25f;
    float4 s = 0.f;
    for (uint l = 0; l < 4; l++) {
        ulong g = iq2xxs_grid[(a0 >> (8 * l)) & 255];
        uint sg = (a1 >> (7 * l)) & 127;
        sg |= (popcount(sg) & 1) << 7;
        s = fma(sgn4(uint(g), sg), x.v[2 * l], s);
        s = fma(sgn4(uint(g >> 32), sg >> 4), x.v[2 * l + 1], s);
    }
    return db * hsum(s);
}
struct MXIQ2_S { device const uchar* w; ulong rb; };
inline float mdot(MXIQ2_S f, uint row, uint c, thread const X32& x) {
    device const uchar* B = f.w + (ulong)row * f.rb + (c / 8) * 82;
    device const ushort* H = (device const ushort*)B;
    uint j = c % 8;
    uint qs = uint(H[1 + 2 * j]) | (uint(H[2 + 2 * j]) << 16);
    uint sn = uint(H[17 + 2 * j]) | (uint(H[18 + 2 * j]) << 16);
    uint qh = B[66 + j], sc = B[74 + j];
    float4 a = 0.f, b = 0.f;
    for (uint l = 0; l < 4; l++) {
        ulong g = iq2s_grid[((qs >> (8 * l)) & 255) | ((qh << (8 - 2 * l)) & 0x300)];
        uint sg = sn >> (8 * l);
        float4 v = fma(sgn4(uint(g), sg), x.v[2 * l], sgn4(uint(g >> 32), sg >> 4) * x.v[2 * l + 1]);
        if (l < 2) a += v; else b += v;
    }
    float d = float(as_type<half>(H[0])) * 0.25f;
    return d * fma(0.5f + float(sc & 15), hsum(a), (0.5f + float(sc >> 4)) * hsum(b));
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
// A Q4_K / Q5_K chunk's value from its sums: d (sc0 . lo + sc1 . hi) - m (mn0 sl + mn1 sh), the
// multiply-adds written out (contraction off) as the compiler had fused them, so that the
// multi-token kernels compute the same bits in any context.
inline float kq_fin(float2 dm, float2 s0, float2 s1, float dl, float dh, float sl, float sh) {
#pragma clang fp contract(off)
    return fma(dm.x, fma(s0.x, dl, s1.x * dh), -(dm.y * fma(s0.y, sl, s1.y * sh)));
}
struct MQ4_K { device const uchar* w; ulong rb; };
// (B: the super-block's d, dmin and scales; Q: the chunk's 16 bytes of quants.)
inline float q4_k_dot(device const uchar* B, device const uchar* Q, uint g, thread const XKQ& x) {
    float2 dm = float2(*(device const half2*)B);
    float4 sc = kscale2(B + 4, g);
    float2 s0 = sc.xy, s1 = sc.zw;
    uint4 u = *(device const uint4*)Q;
    float4 lo = 0.f, hi = 0.f;
    nib_acc(u.x, x.k.l[0], x.k.h[0], lo, hi);
    nib_acc(u.y, x.k.l[1], x.k.h[1], lo, hi);
    nib_acc(u.z, x.k.l[2], x.k.h[2], lo, hi);
    nib_acc(u.w, x.k.l[3], x.k.h[3], lo, hi);
    return kq_fin(dm, s0, s1, dot(lo, NIB_LO), dot(hi, NIB_HI), x.k.sl, x.k.sh);
}
inline float mdot(MQ4_K f, uint row, uint c, thread const XKQ& x) {
    uint k = c % 8, g = k / 2, h = k % 2;
    device const uchar* B = f.w + (ulong)row * f.rb + (c / 8) * 144;
    return q4_k_dot(B, B + 16 + 32 * g + 16 * h, g, x);
}
struct MQ4_KT { device const uchar* w; ulong rb; };
inline float mdot(MQ4_KT f, uint row, uint c, thread const XKQ& x) {
    uint k = c % 8, g = k / 2, h = k % 2;
    return q4_k_dot(tpart(f.w, f.rb, row, c / 8, 144, 0, 16), tpart(f.w, f.rb, row, c / 8, 144, 16 + 32 * g, 32) + 16 * h, g, x);
}
// Q5_K: the fifth bit of value 64g + 16h + i (and of 64g + 32 + 16h + i) is bit 2g (2g + 1) of
// qh[16h + i]. Merged into each value's byte before the float conversion (one conversion and
// product per value, as Q4_K: 550-560 GB/s instead of 425-440 with the bits summed apart).
struct MQ5_K { device const uchar* w; ulong rb; };
inline void nib5_acc(uint w, uint q, uint sh, float4 xa, float4 xb, thread float4& lo, thread float4& hi) {
    uint a = (w & 0x0F0F0F0Fu) | (((q >> sh) & 0x01010101u) << 4);
    uint b = ((w >> 4) & 0x0F0F0F0Fu) | (((q >> (sh + 1)) & 0x01010101u) << 4);
    lo += xa * float4(as_type<uchar4>(a));
    hi += xb * float4(as_type<uchar4>(b));
}
// (B: the super-block's d, dmin, scales and high bits; Q: the chunk's 16 bytes of quants.)
inline float q5_k_dot(device const uchar* B, device const uchar* Q, uint g, uint h, thread const XKQ& x) {
    float2 dm = float2(*(device const half2*)B);
    float4 sc = kscale2(B + 4, g);
    float2 s0 = sc.xy, s1 = sc.zw;
    uint4 u = *(device const uint4*)Q;
    uint4 q = *(device const uint4*)(B + 16 + 16 * h);
    float4 lo = 0.f, hi = 0.f;
    nib5_acc(u.x, q.x, 2 * g, x.k.l[0], x.k.h[0], lo, hi);
    nib5_acc(u.y, q.y, 2 * g, x.k.l[1], x.k.h[1], lo, hi);
    nib5_acc(u.z, q.z, 2 * g, x.k.l[2], x.k.h[2], lo, hi);
    nib5_acc(u.w, q.w, 2 * g, x.k.l[3], x.k.h[3], lo, hi);
    return kq_fin(dm, s0, s1, dot(lo, NIB_LO), dot(hi, NIB_HI), x.k.sl, x.k.sh);
}
inline float mdot(MQ5_K f, uint row, uint c, thread const XKQ& x) {
    uint k = c % 8, g = k / 2, h = k % 2;
    device const uchar* B = f.w + (ulong)row * f.rb + (c / 8) * 176;
    return q5_k_dot(B, B + 48 + 32 * g + 16 * h, g, h, x);
}
struct MQ5_KT { device const uchar* w; ulong rb; };
inline float mdot(MQ5_KT f, uint row, uint c, thread const XKQ& x) {
    uint k = c % 8, g = k / 2, h = k % 2;
    return q5_k_dot(tpart(f.w, f.rb, row, c / 8, 176, 0, 48), tpart(f.w, f.rb, row, c / 8, 176, 48 + 32 * g, 32) + 16 * h, g, h, x);
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
// Q6_K's sums written out (contraction off) as the compiler had fused them: each pair of float4
// dot products as two chains of multiply-adds, then the four sub-blocks' scales (the product by
// d is left to fuse with the lane's running sum, as it did).
inline float q6_pair(float4 a, float4 b, float4 c, float4 e) {
#pragma clang fp contract(off)
    return fma(a.w, b.w, fma(a.z, b.z, fma(a.y, b.y, a.x * b.x))) + fma(c.w, e.w, fma(c.z, e.z, fma(c.y, e.y, c.x * e.x)));
}
inline float q6_sum(float s0, float s1, float s2, float s3, float c0, float c2, float c4, float c6) {
#pragma clang fp contract(off)
    return fma(s3, c6, fma(s2, c4, fma(s0, c0, s1 * c2)));
}
struct MQ6_K { device const uchar* w; ulong rb; };
// Four 6-bit values (minus 32) of one quarter: low bits from `ql` nibbles, high bits from `qh`.
inline float4 q6v(uint ql, uint qh) {
    uint q = (ql & 0x0F0F0F0Fu) | ((qh & 0x03030303u) << 4);
    return float4(as_type<uchar4>(q)) - 32.f;
}
// (L: the half's 64 low quant bytes, H: its 32 high ones, S: the 16 scales and d.)
inline float q6_k_dot(device const uchar* L, device const uchar* H, device const uchar* S, uint n, uint l0, thread const X6& x) {
    device const ushort* q1 = (device const ushort*)(L + l0);
    device const ushort* q2 = (device const ushort*)(L + 32 + l0);
    device const ushort* qh = (device const ushort*)(H + l0);
    uint a0 = ld32u(q1), a1 = ld32u(q1 + 2), b0 = ld32u(q2), b1 = ld32u(q2 + 2), h0 = ld32u(qh), h1 = ld32u(qh + 2);
    device const char* sc = (device const char*)(S + 8 * n + l0 / 16);
    float d = float(*(device const half*)(S + 16));
    float s0 = q6_pair(q6v(a0, h0), x.v[0], q6v(a1, h1), x.v[1]);
    float s1 = q6_pair(q6v(b0, h0 >> 2), x.v[2], q6v(b1, h1 >> 2), x.v[3]);
    float s2 = q6_pair(q6v(a0 >> 4, h0 >> 4), x.v[4], q6v(a1 >> 4, h1 >> 4), x.v[5]);
    float s3 = q6_pair(q6v(b0 >> 4, h0 >> 6), x.v[6], q6v(b1 >> 4, h1 >> 6), x.v[7]);
    return d * q6_sum(s0, s1, s2, s3, float(sc[0]), float(sc[2]), float(sc[4]), float(sc[6]));
}
inline float mdot(MQ6_K f, uint row, uint c, thread const X6& x) {
    uint n = (c % 8) / 4, l0 = 8 * (c % 4);
    device const uchar* B = f.w + (ulong)row * f.rb + (c / 8) * 210;
    return q6_k_dot(B + 64 * n, B + 128 + 32 * n, B + 192, n, l0, x);
}
struct MQ6_KT { device const uchar* w; ulong rb; };
inline float mdot(MQ6_KT f, uint row, uint c, thread const X6& x) {
    uint n = (c % 8) / 4, l0 = 8 * (c % 4);
    device const uchar* P = tpart(f.w, f.rb, row, c / 8, 210, 18 + 96 * n, 96);
    return q6_k_dot(P, P + 64, tpart(f.w, f.rb, row, c / 8, 210, 0, 18), n, l0, x);
}
// ---- IQ4_XS (136-byte super-blocks of 256, 8-byte aligned): chunk k is sub-block k % 8, its 16
// bytes of nibbles (low = value i, high = value i + 16) looked up in the kernel's threadgroup copy
// of the non-linear values (`kv`, filled by MV_IQ4 / MVK_IQ4: a lookup in constant memory
// serializes over the lanes' distinct indices), dotted with x, then scaled by the sub-block's
// d (ls - 32). The products and sums written out (contraction off), the same in any kernel.
struct MIQ4_XS { device const uchar* w; ulong rb; threadgroup const float* kv; };
inline float4 iq4v(threadgroup const float* kv, uint w) {
    return float4(kv[w & 15u], kv[(w >> 8) & 15u], kv[(w >> 16) & 15u], kv[(w >> 24) & 15u]);
}
inline float mdot(MIQ4_XS f, uint row, uint c, thread const X32& xr) {
#pragma clang fp contract(off)
    device const uchar* B = f.w + (ulong)row * f.rb + (c / 8) * 136;
    uint j = c % 8;
    uint hs = *(device const uint*)B, sl = *(device const uint*)(B + 4);
    int ls = int(((sl >> (4 * j)) & 15u) | (((hs >> (16 + 2 * j)) & 3u) << 4)) - 32;
    float dl = float(as_type<half>(ushort(hs & 0xFFFFu))) * float(ls);
    device const uint2* q = (device const uint2*)(B + 8 + 16 * j);
    uint2 a = q[0], b = q[1];
    uint u[4] = { a.x, a.y, b.x, b.y };
    float4 lo = iq4v(f.kv, u[0]) * xr.v[0], hi = iq4v(f.kv, u[0] >> 4) * xr.v[4];
    for (int i = 1; i < 4; i++) {
        lo = fma(iq4v(f.kv, u[i]), xr.v[i], lo);
        hi = fma(iq4v(f.kv, u[i] >> 4), xr.v[i + 4], hi);
    }
    return dl * hsum(lo + hi);
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
// The chunks' dot products (`mdot`) are added without contraction: the compiler would fuse the
// last product of a chunk with the sum where a loop's shape lets it, and the multi-token kernels
// (`mvk_*`), which loop otherwise, must add the same values the same way.
template <typename F, typename X, uint CH, int NR, int KS>
inline void mv_acc(F f, F f2, bool pair, uint r2off, device const float* x, uint n_in, uint n_out, uint r0, uint sg, uint lane,
                   thread float* acc, thread float* acc2, float scale) {
    // (Each chunk's dot product rounded, then added: not fused with the sum, whatever the loop
    // around it, so that every kernel adds the chunks alike; see `mv_acc`.)
#pragma clang fp contract(off)
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
// 2 silu(acc) * acc2, 3 gelu(acc) * acc2, 4 add acc * sigmoid(*gate), 5 raw pair, 6 clamped SwiGLU
// pair (limit *gate; the expert pair kernels read it after the k ids and weights of `sel`).
// Row o's sums a (and b, of a pair) written to y by `mode` (see `mv_finish`).
inline void mv_store(float a, float b, device float* y, uint mode, uint o, uint n_out, device const float* gate) {
    if (mode == 0) y[o] = a;
    else if (mode == 1) y[o] += a;
    else if (mode == 2) y[o] = a / (1.0f + exp(-a)) * b;
    else if (mode == 3) {
        float t = 0.7978845608f * (a + 0.044715f * a * a * a);
        y[o] = 0.5f * a * (1.0f + precise::tanh(t)) * b;
    } else if (mode == 5) {
        // Raw pair (experts with LoRA updates): gate then up outputs, 2 n_out per slot.
        y[o] = a;
        y[n_out + o] = b;
    } else if (mode == 6) {
        // DeepSeek's clamped SwiGLU, the limit at *gate: silu(min(a, l)) * clamp(b, -l, l).
        float l = gate[0];
        float g = min(a, l);
        y[o] = g / (1.0f + precise::exp(-g)) * clamp(b, -l, l);
    } else if (mode == 7) {
        // The GLU after a lone matvec (`dec_glu`'s arithmetic): gelu(a) * u, u at gate.
        float t = 0.7978845608f * (a + 0.044715f * a * a * a);
        float act = 0.5f * a * (1.0f + precise::tanh(t));
        y[o] = act * gate[o];
    } else if (mode == 8) {
        float act = a / (1.0f + exp(-a));
        y[o] = act * gate[o];
    } else y[o] += a / (1.0f + exp(-gate[0]));
}
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
        mv_store(a, b, y, mode, r0 + lane, n_out, gate);
    }
}
template <typename F, typename X, uint CH, int NR, int KS>
inline void mv_body(F f, F f2, device const float* x, device float* y, constant MV& p, uint tg, uint sg, uint lane,
                    threadgroup float* red, device const float* gate) {
    float acc[NR], acc2[NR];
    for (int r = 0; r < NR; r++) { acc[r] = 0.f; acc2[r] = 0.f; }
    mv_acc<F, X, CH, NR, KS>(f, f2, p.mode == 2 || p.mode == 3 || p.mode == 6, 0, x, p.n_in, p.n_out, tg * NR, sg, lane, acc, acc2, 1.f);
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
#define IQ4_TABLE threadgroup float kvt[16]; \
    if (sg == 0 && lane < 16) kvt[lane] = kvalues_iq4nl[lane]; \
    threadgroup_barrier(mem_flags::mem_threadgroup);
#define MV_IQ4(NAME, NR, KS) \
MV_SIG(NAME) { threadgroup float red[2 * NR * KS]; IQ4_TABLE \
    MIQ4_XS f = { wb + p.w_off, p.gpr, kvt }; MIQ4_XS f2 = { wb2 + p.w2_off, p.gpr, kvt }; \
    mv_body<MIQ4_XS, X32, 32, NR, KS>(f, f2, x, y, p, tg, sg, lane, red, gate); }
#define MV_GEN(NAME, W, NR, KS) \
MV_SIG(NAME) { threadgroup float red[2 * NR * KS]; \
    MGen<W> f = { W{ { wb + p.w_off, p.gpr } } }; MGen<W> f2 = { W{ { wb2 + p.w2_off, p.gpr } } }; \
    mv_body<MGen<W>, X32, 32, NR, KS>(f, f2, x, y, p, tg, sg, lane, red, gate); }

#pragma clang fp reassociate(on)
// ---- The same for a few input rows (1 to 4, `p.ldy`; rows of x and y are n_in / n_out apart), a
// DSpark draft's block of 3 or a speculative round's committed tokens. Each weight chunk is read
// from memory once for all rows: GGUF formats dequantize it once into registers (the GEMMs'
// `load`) and dot it with every row, the other formats repeat their fused dot product per row
// (the weights then come from the cache). Modes 0 (store), 1 (add) and 2 (SiLU pair). Token loops
// run to 4 with a guard, so the accumulators stay in registers.
template <int NR, int KS>
inline void mvt_finish(thread float (&a)[4][NR], thread float (&b)[4][NR], device float* y, constant MV& p, uint r0, uint sg,
                       uint lane, threadgroup float* red) {
    const uint nt = p.ldy, n_out = p.n_out;
    for (int t = 0; t < 4; t++)
        for (int r = 0; r < NR; r++) { a[t][r] = simd_sum(a[t][r]); b[t][r] = simd_sum(b[t][r]); }
    if (KS > 1) {
        if (lane == 0)
            for (int t = 0; t < 4; t++)
                for (int r = 0; r < NR; r++) {
                    red[((sg * 4 + t) * NR + r) * 2] = a[t][r];
                    red[((sg * 4 + t) * NR + r) * 2 + 1] = b[t][r];
                }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg != 0) return;
        for (int t = 0; t < 4; t++)
            for (int r = 0; r < NR; r++) {
                float sa = 0.f, sb = 0.f;
                for (int s = 0; s < KS; s++) { sa += red[((s * 4 + t) * NR + r) * 2]; sb += red[((s * 4 + t) * NR + r) * 2 + 1]; }
                a[t][r] = sa; b[t][r] = sb;
            }
    }
    // Lane t * NR + r writes row r of token t.
    uint lt = lane / NR, lr = lane % NR;
    if (lt < nt && r0 + lr < n_out) {
        float va = 0.f, vb = 0.f;
        for (int t = 0; t < 4; t++)
            for (int r = 0; r < NR; r++)
                if (lt == uint(t) && lr == uint(r)) { va = a[t][r]; vb = b[t][r]; }
        device float* o = y + (ulong)lt * n_out + r0 + lr;
        if (p.mode == 0) *o = va;
        else if (p.mode == 1) *o += va;
        else *o = va / (1.0f + exp(-va)) * vb;
    }
}
template <typename F, typename X, uint CH, int NR, int KS>
inline void mvt_body(F f, F f2, device const float* x, device float* y, constant MV& p, uint tg, uint sg, uint lane,
                     threadgroup float* red) {
    const uint nt = p.ldy, r0 = tg * NR;
    const bool pair = p.mode == 2;
    uint rows[NR];
    for (int r = 0; r < NR; r++) rows[r] = min(r0 + r, p.n_out - 1);
    float a[4][NR], b[4][NR];
    for (int t = 0; t < 4; t++)
        for (int r = 0; r < NR; r++) { a[t][r] = 0.f; b[t][r] = 0.f; }
    uint nch = p.n_in / CH;
    for (uint c = sg * 32 + lane; c < nch; c += KS * 32) {
        for (int t = 0; t < 4; t++) {
            if (uint(t) >= nt) break;
            X xr;
            xload(x + (ulong)t * p.n_in, c, xr);
            for (int r = 0; r < NR; r++) a[t][r] += mdot(f, rows[r], c, xr);
            if (pair)
                for (int r = 0; r < NR; r++) b[t][r] += mdot(f2, rows[r], c, xr);
        }
    }
    mvt_finish<NR, KS>(a, b, y, p, r0, sg, lane, red);
}
inline float dot8(thread const float4* v, thread const X32& xr) {
    float4 s = v[0] * xr.v[0];
    for (uint i = 1; i < 8; i++) s = fma(v[i], xr.v[i], s);
    return hsum(s);
}
template <typename W, int NR, int KS>
inline void mvtl_body(W w, W w2, device const float* x, device float* y, constant MV& p, uint tg, uint sg, uint lane,
                      threadgroup float* red) {
    const uint nt = p.ldy, r0 = tg * NR;
    const bool pair = p.mode == 2;
    float a[4][NR], b[4][NR];
    for (int t = 0; t < 4; t++)
        for (int r = 0; r < NR; r++) { a[t][r] = 0.f; b[t][r] = 0.f; }
    uint nch = p.n_in / 32;
    for (uint c = sg * 32 + lane; c < nch; c += KS * 32) {
        for (int r = 0; r < NR; r++) {
            uint row = min(r0 + r, p.n_out - 1);
            float4 v[8], u[8];
            load(w, row, c, v);
            if (pair) load(w2, row, c, u);
            for (int t = 0; t < 4; t++) {
                if (uint(t) >= nt) break;
                X32 xr;
                xload(x + (ulong)t * p.n_in, c, xr);
                a[t][r] += dot8(v, xr);
                if (pair) b[t][r] += dot8(u, xr);
            }
        }
    }
    mvt_finish<NR, KS>(a, b, y, p, r0, sg, lane, red);
}
#define MVT_RAW(NAME, T, X, CH, NR, KS) \
MV_SIG(NAME) { threadgroup float red[8 * NR * KS]; \
    T f = { wb + p.w_off, p.gpr }; T f2 = { wb2 + p.w2_off, p.gpr }; \
    mvt_body<T, X, CH, NR, KS>(f, f2, x, y, p, tg, sg, lane, red); }
#define MVT_Q8(NAME, NR, KS) \
MV_SIG(NAME) { threadgroup float red[8 * NR * KS]; \
    MQ8 f = { (device const char*)(wb + p.w_off), (device const float*)(sb + p.s_off), p.n_in }; \
    MQ8 f2 = { (device const char*)(wb2 + p.w2_off), (device const float*)(sb2 + p.s2_off), p.n_in }; \
    mvt_body<MQ8, X16, 16, NR, KS>(f, f2, x, y, p, tg, sg, lane, red); }
#define MVT_Q4(NAME, NR, KS) \
MV_SIG(NAME) { threadgroup float red[8 * NR * KS]; \
    MQ4 f = { wb + p.w_off, (device const half*)(sb + p.s_off), p.n_in, (device const ushort*)(ob + p.o_off), p.gpr, p.gk }; \
    MQ4 f2 = { wb2 + p.w2_off, (device const half*)(sb2 + p.s2_off), p.n_in, (device const ushort*)(ob2 + p.o2_off), p.gpr, p.gk }; \
    mvt_body<MQ4, X32, 32, NR, KS>(f, f2, x, y, p, tg, sg, lane, red); }
#define MVT_LOAD(NAME, W, NR, KS) \
MV_SIG(NAME) { threadgroup float red[8 * NR * KS]; \
    W f = W{ { wb + p.w_off, p.gpr } }; W f2 = W{ { wb2 + p.w2_off, p.gpr } }; \
    mvtl_body<W, NR, KS>(f, f2, x, y, p, tg, sg, lane, red); }

#pragma clang fp reassociate(off)
// ---- Matvecs of several tokens (speculative verification in a decoding plan): the weights are read once for up to
// TT tokens (x and y rows `sx` / `sy` floats apart, the AddGated gate `sgate` apart). Every
// (token, row) is computed with exactly the single-token kernel's arithmetic (same chunks per
// lane, same reduction order), so a verification pass reproduces one-token decoding bit for bit.
struct MVK { MV m; uint ntok; uint sx; uint sy; uint sgate; };
// A chunk of one row decoded once for every token: `wdec` converts the quants to floats (they do
// not depend on x), `wdot` is then `mdot`'s arithmetic on them, product for product. Formats
// without a decoded form keep the row and call `mdot`.
struct WRow { uint row; };
template <typename F, int H> struct WDec { typedef WRow T; };
template <typename F> inline void wdec(F f, uint row, uint c, thread WRow& w) { w.row = row; }
template <typename F, typename X> inline float wdot(F f, thread const WRow& w, uint c, thread const X& xr) { return mdot(f, w.row, c, xr); }
// Any other GGUF format: the chunk dequantized once (its GEMM loader), then mdot's dot product.
template <int H> struct WDec<MIQ1_S, H> { typedef WIQ1S T; };
inline void wdec(MIQ1_S f, uint row, uint c, thread WIQ1S& w) { iq1s_dec(f, row, c, w); }
inline float wdot(MIQ1_S f, thread const WIQ1S& w, uint c, thread const X32& xr) { return iq1s_fin(w, xr); }
template <int H> struct WDec<MIQ1_M, H> { typedef WIQ1M T; };
inline void wdec(MIQ1_M f, uint row, uint c, thread WIQ1M& w) { iq1m_dec(f, row, c, w); }
inline float wdot(MIQ1_M f, thread const WIQ1M& w, uint c, thread const X32& xr) { return iq1m_fin(w, xr); }
struct WGen { float4 v[8]; };
template <typename W, int H> struct WDec<MGen<W>, H> { typedef WGen T; };
template <typename W>
inline void wdec(MGen<W> f, uint row, uint c, thread WGen& w) { load(f.w, row, c, w.v); }
template <typename W>
inline float wdot(MGen<W> f, thread const WGen& w, uint c, thread const X32& xr) {
    float4 a = w.v[0] * xr.v[0];
    for (uint i = 1; i < 8; i++) a = fma(w.v[i], xr.v[i], a);
    return hsum(a);
}
// The masked nibbles of a word as floats (see nib_acc): low nibbles in `l`, high in `h`.
template <typename V>
inline void nib_dec(uint w, thread V& l, thread V& h) {
    l = V(as_type<uchar4>(w & 0x0F0F0F0Fu));
    h = V(as_type<uchar4>((w >> 4) & 0x0F0F0F0Fu));
}
template <typename V>
inline float q4_dot(thread const V* nl, thread const V* nh, thread const X32& xr) {
    float4 lo = 0.f, hi = 0.f;
    for (int i = 0; i < 4; i++) { lo += xr.v[i] * float4(nl[i]); hi += xr.v[4 + i] * float4(nh[i]); }
    return dot(lo, NIB_LO) + dot(hi, NIB_HI);
}
// Decoded quants are float4 or half4 (exact either way; `H`, chosen by the host per format,
// token count and row length, see `mvk_variant`): half takes half the registers, float no
// conversion at each use.
template <int H> struct HV { typedef float4 T; };
template <> struct HV<1> { typedef half4 T; };
template <typename V> struct WQ4 { V nl[4]; V nh[4]; float d; float o; };
template <int H> struct WDec<MQ4, H> { typedef WQ4<typename HV<H>::T> T; };
template <typename V>
inline void wdec(MQ4 f, uint row, uint c, thread WQ4<V>& w) {
    uint4 u = *(device const uint4*)(f.q + (ulong)row * (f.n_in / 2) + c * 16);
    w.d = float(f.s[(ulong)row * (f.n_in / 32) + c]);
    w.o = f.gk != 0 ? as_type<float>(uint(f.o[(ulong)row * f.gpr + c / f.gk]) << 16) : 0.f;
    nib_dec(u.x, w.nl[0], w.nh[0]);
    nib_dec(u.y, w.nl[1], w.nh[1]);
    nib_dec(u.z, w.nl[2], w.nh[2]);
    nib_dec(u.w, w.nl[3], w.nh[3]);
}
template <typename V>
inline float wdot(MQ4 f, thread const WQ4<V>& w, uint c, thread const X32& xr) {
    // (Both forms computed, the affine one selected: as the previous kernels' shape, 4-7% faster
    // than two returns for 8-16 tokens of 4096 x 12288.)
    float qd = q4_dot(w.nl, w.nh, xr), xs = xsum32(xr);
    float r = w.d * (qd - 8.f * xs);
    if (f.gk != 0) r = q4_affine(w.d, qd, xs, w.o);
    return r;
}
template <int H> struct WDec<MQ4_0, H> { typedef WQ4<typename HV<H>::T> T; };
template <typename V>
inline void wdec(MQ4_0 f, uint row, uint c, thread WQ4<V>& w) {
    device const ushort* b = (device const ushort*)(f.w + (ulong)row * f.rb + c * 18);
    w.d = float(as_type<half>(b[0]));
    w.o = 0.f;
    nib_dec(ld32u(b + 1), w.nl[0], w.nh[0]);
    nib_dec(ld32u(b + 3), w.nl[1], w.nh[1]);
    nib_dec(ld32u(b + 5), w.nl[2], w.nh[2]);
    nib_dec(ld32u(b + 7), w.nl[3], w.nh[3]);
}
template <typename V>
inline float wdot(MQ4_0 f, thread const WQ4<V>& w, uint c, thread const X32& xr) { return w.d * (q4_dot(w.nl, w.nh, xr) - 8.f * xsum32(xr)); }
struct WF16 { float4 w[4]; };
template <int H> struct WDec<MBf, H> { typedef WF16 T; };
inline void wdec(MBf f, uint row, uint c, thread WF16& w) {
    device const packed_ushort4* p = (device const packed_ushort4*)(f.w + (ulong)row * f.rb + c * 32);
    for (int i = 0; i < 4; i++) w.w[i] = bf4(p[i]);
}
template <int H> struct WDec<MF16, H> { typedef WF16 T; };
inline void wdec(MF16 f, uint row, uint c, thread WF16& w) {
    device const packed_half4* p = (device const packed_half4*)(f.w + (ulong)row * f.rb + c * 32);
    for (int i = 0; i < 4; i++) w.w[i] = float4(half4(p[i]));
}
template <typename F>
inline float wdot(F f, thread const WF16& w, uint c, thread const X16& xr) {
    float4 a = w.w[0] * xr.v[0];
    a = fma(w.w[1], xr.v[1], a);
    a = fma(w.w[2], xr.v[2], a);
    a = fma(w.w[3], xr.v[3], a);
    return hsum(a);
}
template <typename V> struct WQ8 { V w[4]; float d; };
template <int H> struct WDec<MQ8, H> { typedef WQ8<typename HV<H>::T> T; };
template <typename V>
inline void wdec(MQ8 f, uint row, uint c, thread WQ8<V>& w) {
    uint4 u = *(device const uint4*)(f.q + (ulong)row * f.n_in + c * 16);
    w.d = f.s[(ulong)row * (f.n_in / 32) + c / 2];
    w.w[0] = V(as_type<char4>(u.x)); w.w[1] = V(as_type<char4>(u.y));
    w.w[2] = V(as_type<char4>(u.z)); w.w[3] = V(as_type<char4>(u.w));
}
template <typename V>
inline float wdot(MQ8 f, thread const WQ8<V>& w, uint c, thread const X16& xr) {
    float4 a = float4(w.w[0]) * xr.v[0];
    a = fma(float4(w.w[1]), xr.v[1], a);
    a = fma(float4(w.w[2]), xr.v[2], a);
    a = fma(float4(w.w[3]), xr.v[3], a);
    return w.d * hsum(a);
}
template <typename V> struct WQ80 { V w[8]; float d; };
template <int H> struct WDec<MQ8_0, H> { typedef WQ80<typename HV<H>::T> T; };
template <typename V>
inline void wdec(MQ8_0 f, uint row, uint c, thread WQ80<V>& w) {
    device const ushort* b = (device const ushort*)(f.w + (ulong)row * f.rb + c * 34);
    w.d = float(as_type<half>(b[0]));
    for (int i = 0; i < 8; i++) w.w[i] = V(c4(b[1 + 2 * i], b[2 + 2 * i]));
}
template <typename V>
inline float wdot(MQ8_0 f, thread const WQ80<V>& w, uint c, thread const X32& xr) {
    float4 a = float4(w.w[0]) * xr.v[0];
    for (int i = 1; i < 8; i++) a = fma(float4(w.w[i]), xr.v[i], a);
    return w.d * hsum(a);
}
// The output of row o of one token by `mode` from its sums a (and b, a pair's second matrix):
// mv_finish's arithmetic.
inline void mvk_out(device float* y, uint o, uint mode, float a, float b, device const float* gate) {
    if (mode == 0) y[o] = a;
    else if (mode == 1) y[o] += a;
    else if (mode == 2) y[o] = a / (1.0f + exp(-a)) * b;
    else if (mode == 3) {
        float t = 0.7978845608f * (a + 0.044715f * a * a * a);
        y[o] = 0.5f * a * (1.0f + precise::tanh(t)) * b;
    } else if (mode == 6) {
        float l = gate[0];
        float g = min(a, l);
        y[o] = g / (1.0f + precise::exp(-g)) * clamp(b, -l, l);
    } else if (mode == 7) {
        float t = 0.7978845608f * (a + 0.044715f * a * a * a);
        float act = 0.5f * a * (1.0f + precise::tanh(t));
        y[o] = act * gate[o];
    } else if (mode == 8) {
        float act = a / (1.0f + exp(-a));
        y[o] = act * gate[o];
    } else y[o] += a / (1.0f + exp(-gate[0]));
}
// K-quants (Q4_K, Q5_K): the chunk's 32 quants as half (exact) and its super-block scale, min and the
// two sub-blocks' scales and mins.
template <typename V> struct WQK { V nl[4]; V nh[4]; float2 dm; float4 sc; };
template <int H> struct WDec<MQ4_K, H> { typedef WQK<typename HV<H>::T> T; };
template <typename V>
inline void wdec(MQ4_K f, uint row, uint c, thread WQK<V>& w) {
    uint k = c % 8, g = k / 2, h = k % 2;
    device const uchar* B = f.w + (ulong)row * f.rb + (c / 8) * 144;
    w.dm = float2(*(device const half2*)B);
    w.sc = kscale2(B + 4, g);
    uint4 u = *(device const uint4*)(B + 16 + 32 * g + 16 * h);
    nib_dec(u.x, w.nl[0], w.nh[0]);
    nib_dec(u.y, w.nl[1], w.nh[1]);
    nib_dec(u.z, w.nl[2], w.nh[2]);
    nib_dec(u.w, w.nl[3], w.nh[3]);
}
template <int H> struct WDec<MQ5_K, H> { typedef WQK<typename HV<H>::T> T; };
template <typename V>
inline void wdec(MQ5_K f, uint row, uint c, thread WQK<V>& w) {
    uint k = c % 8, g = k / 2, h = k % 2;
    device const uchar* B = f.w + (ulong)row * f.rb + (c / 8) * 176;
    w.dm = float2(*(device const half2*)B);
    w.sc = kscale2(B + 4, g);
    uint4 u = *(device const uint4*)(B + 48 + 32 * g + 16 * h);
    uint4 q = *(device const uint4*)(B + 16 + 16 * h);
    uint sh = 2 * g;
    for (int i = 0; i < 4; i++) {
        uint wd = u[i], qd = q[i];
        w.nl[i] = V(as_type<uchar4>((wd & 0x0F0F0F0Fu) | (((qd >> sh) & 0x01010101u) << 4)));
        w.nh[i] = V(as_type<uchar4>(((wd >> 4) & 0x0F0F0F0Fu) | (((qd >> (sh + 1)) & 0x01010101u) << 4)));
    }
}
template <typename F, typename V>
inline float wdot(F f, thread const WQK<V>& w, uint c, thread const XKQ& x) {
    float4 lo = 0.f, hi = 0.f;
    for (int i = 0; i < 4; i++) { lo += x.k.l[i] * float4(w.nl[i]); hi += x.k.h[i] * float4(w.nh[i]); }
    return kq_fin(w.dm, w.sc.xy, w.sc.zw, dot(lo, NIB_LO), dot(hi, NIB_HI), x.k.sl, x.k.sh);
}
// Packed chunks (`H` 3): the quant words as stored, unpacked at each use into the same values
// (a quarter of the registers of float quants, so that more rows share each x load).
struct WQ4KP { uint4 u; float2 dm; float4 sc; };
template <> struct WDec<MQ4_K, 3> { typedef WQ4KP T; };
inline void wdec(MQ4_K f, uint row, uint c, thread WQ4KP& w) {
    uint k = c % 8, g = k / 2, h = k % 2;
    device const uchar* B = f.w + (ulong)row * f.rb + (c / 8) * 144;
    w.dm = float2(*(device const half2*)B);
    w.sc = kscale2(B + 4, g);
    w.u = *(device const uint4*)(B + 16 + 32 * g + 16 * h);
}
template <typename F>
inline float wdot(F f, thread const WQ4KP& w, uint c, thread const XKQ& x) {
    float4 lo = 0.f, hi = 0.f;
    for (int i = 0; i < 4; i++) {
        lo += x.k.l[i] * float4(as_type<uchar4>(w.u[i] & 0x0F0F0F0Fu));
        hi += x.k.h[i] * float4(as_type<uchar4>((w.u[i] >> 4) & 0x0F0F0F0Fu));
    }
    return kq_fin(w.dm, w.sc.xy, w.sc.zw, dot(lo, NIB_LO), dot(hi, NIB_HI), x.k.sl, x.k.sh);
}
struct WQ80P { uint w[8]; float d; };
template <> struct WDec<MQ8_0, 3> { typedef WQ80P T; };
inline void wdec(MQ8_0 f, uint row, uint c, thread WQ80P& w) {
    device const ushort* b = (device const ushort*)(f.w + (ulong)row * f.rb + c * 34);
    w.d = float(as_type<half>(b[0]));
    for (int i = 0; i < 8; i++) w.w[i] = uint(b[1 + 2 * i]) | (uint(b[2 + 2 * i]) << 16);
}
inline float wdot(MQ8_0 f, thread const WQ80P& w, uint c, thread const X32& xr) {
    float4 a = float4(as_type<char4>(w.w[0])) * xr.v[0];
    for (int i = 1; i < 8; i++) a = fma(float4(as_type<char4>(w.w[i])), xr.v[i], a);
    return w.d * hsum(a);
}
// Q6_K: the chunk's 32 values (minus 32) as half (exact), its scale and the four sub-blocks' scales.
template <typename V> struct WQ6 { V v[8]; float d; float4 c; };
template <int H> struct WDec<MQ6_K, H> { typedef WQ6<typename HV<H>::T> T; };
template <typename V>
inline void wdec(MQ6_K f, uint row, uint c, thread WQ6<V>& w) {
    uint n = (c % 8) / 4, l0 = 8 * (c % 4);
    device const uchar* B = f.w + (ulong)row * f.rb + (c / 8) * 210;
    device const ushort* q1 = (device const ushort*)(B + 64 * n + l0);
    device const ushort* q2 = (device const ushort*)(B + 64 * n + 32 + l0);
    device const ushort* qh = (device const ushort*)(B + 128 + 32 * n + l0);
    uint a0 = ld32u(q1), a1 = ld32u(q1 + 2), b0 = ld32u(q2), b1 = ld32u(q2 + 2), h0 = ld32u(qh), h1 = ld32u(qh + 2);
    device const char* sc = (device const char*)(B + 192 + 8 * n + l0 / 16);
    w.d = float(*(device const half*)(B + 208));
    w.c = float4(float(sc[0]), float(sc[2]), float(sc[4]), float(sc[6]));
    w.v[0] = V(q6v(a0, h0)); w.v[1] = V(q6v(a1, h1));
    w.v[2] = V(q6v(b0, h0 >> 2)); w.v[3] = V(q6v(b1, h1 >> 2));
    w.v[4] = V(q6v(a0 >> 4, h0 >> 4)); w.v[5] = V(q6v(a1 >> 4, h1 >> 4));
    w.v[6] = V(q6v(b0 >> 4, h0 >> 6)); w.v[7] = V(q6v(b1 >> 4, h1 >> 6));
}
template <typename V>
inline float wdot(MQ6_K f, thread const WQ6<V>& w, uint c, thread const X6& x) {
    float s0 = q6_pair(float4(w.v[0]), x.v[0], float4(w.v[1]), x.v[1]);
    float s1 = q6_pair(float4(w.v[2]), x.v[2], float4(w.v[3]), x.v[3]);
    float s2 = q6_pair(float4(w.v[4]), x.v[4], float4(w.v[5]), x.v[5]);
    float s3 = q6_pair(float4(w.v[6]), x.v[6], float4(w.v[7]), x.v[7]);
    return w.d * q6_sum(s0, s1, s2, s3, w.c.x, w.c.y, w.c.z, w.c.w);
}
// Every token's lane sums reduced as mv_finish reduces one token's: the SIMD-group sum, then the
// KS SIMD-groups' sums in order (through threadgroup memory, `red`: 2 T NR KS floats), then the
// mode's output, one (token, row) a lane of the first SIMD-group. With one SIMD-group a row,
// every lane holds every sum after the SIMD-group sums and lane t NR + r writes row r of token t.
template <int NR, int KS, int T>
inline void mvk_finish(thread float (&a)[T][NR], thread float (&b)[T][NR], bool pair, uint nt, device float* y, constant MVK& q,
                       uint r0, uint sg, uint lane, threadgroup float* red, device const float* gate) {
    constant MV& p = q.m;
    for (int t = 0; t < T; t++) {
        if (uint(t) >= nt) break;
        for (int r = 0; r < NR; r++) {
            a[t][r] = simd_sum(a[t][r]);
            if (pair) b[t][r] = simd_sum(b[t][r]);
        }
    }
    if (KS == 1) {
        for (uint i0 = 0; i0 < nt * NR; i0 += 32) {
            uint i = i0 + lane, tl = i / NR, rl = i % NR;
            float va = 0.f, vb = 0.f;
            for (int t = 0; t < T; t++)
                for (int r = 0; r < NR; r++)
                    if (tl == uint(t) && rl == uint(r)) { va = a[t][r]; vb = b[t][r]; }
            if (i < nt * NR && r0 + rl < p.n_out) mvk_out(y + (ulong)tl * q.sy, r0 + rl, p.mode, va, vb, gate + (ulong)tl * q.sgate);
        }
        return;
    }
    if (lane == 0)
        for (int t = 0; t < T; t++) {
            if (uint(t) >= nt) break;
            for (int r = 0; r < NR; r++) {
                red[((sg * T + t) * NR + r) * 2] = a[t][r];
                red[((sg * T + t) * NR + r) * 2 + 1] = b[t][r];
            }
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0) return;
    for (uint i = lane; i < nt * NR; i += 32) {
        uint t = i / NR, r = i % NR;
        if (r0 + r >= p.n_out) continue;
        float va = 0.f, vb = 0.f;
        for (int s = 0; s < KS; s++) {
            va += red[((s * T + t) * NR + r) * 2];
            vb += red[((s * T + t) * NR + r) * 2 + 1];
        }
        mvk_out(y + (ulong)t * q.sy, r0 + r, p.mode, va, vb, gate + (ulong)t * q.sgate);
    }
}
// nt <= T tokens: per chunk, the NR rows' quants are decoded once (`wdec`; with a pair, the
// second matrix's too), then each token's x chunk is loaded once and dotted with every row
// (`wdot`, the single-token kernel's arithmetic): the weights are read once for all the tokens.
// (Decoding a pair's second matrix into the first's registers after the first's tokens, x loaded
// again, measured 1.5x slower in passes, for the other modes too: Inkling-2-Mini Q8_0 4-token
// pass, its 2304 x 3072 matvecs 26.4 -> 41.9 us a stage.)
template <typename F, typename X, uint CH, int NR, int KS, int T, int H>
inline void mvk_part(F f, F f2, device const float* x, device float* y, device const float* gate, uint nt, constant MVK& q, uint tg,
                     uint sg, uint lane, threadgroup float* red) {
    // (Each chunk's dot product rounded, then added: not fused with the sum, whatever the loop
    // around it, so that every kernel adds the chunks alike; see `mv_acc`.)
#pragma clang fp contract(off)
    constant MV& p = q.m;
    const bool pair = p.mode == 2 || p.mode == 3 || p.mode == 6;
    const uint r0 = tg * NR, nch = p.n_in / CH;
    uint rows[NR];
    for (int r = 0; r < NR; r++) rows[r] = min(r0 + r, p.n_out - 1);
    float a[T][NR], b[T][NR];
    for (int t = 0; t < T; t++)
        for (int r = 0; r < NR; r++) { a[t][r] = 0.f; b[t][r] = 0.f; }
    for (uint c = sg * 32 + lane; c < nch; c += KS * 32) {
        typename WDec<F, H>::T w[NR];
        for (int r = 0; r < NR; r++) wdec(f, rows[r], c, w[r]);
        if (pair) {
            typename WDec<F, H>::T w2[NR];
            for (int r = 0; r < NR; r++) wdec(f2, rows[r], c, w2[r]);
            for (int t = 0; t < T; t++) {
                if (uint(t) >= nt) break;
                X xr;
                xload(x + (ulong)t * q.sx, c, xr);
                for (int r = 0; r < NR; r++) {
                    a[t][r] += wdot(f, w[r], c, xr);
                    b[t][r] += wdot(f2, w2[r], c, xr);
                }
            }
        } else {
            for (int t = 0; t < T; t++) {
                if (uint(t) >= nt) break;
                X xr;
                xload(x + (ulong)t * q.sx, c, xr);
                for (int r = 0; r < NR; r++) a[t][r] += wdot(f, w[r], c, xr);
            }
        }
    }
    mvk_finish<NR, KS, T>(a, b, pair, nt, y, q, r0, sg, lane, red, gate);
}
// Rows of a few chunks a lane (`H` 6, 7, 8: mvk_part's kinds 0, 1, 3, for n_in <= C KS 32 CH):
// every chunk of the lane's rows decoded once and kept, then each token's lane sums (the lane's
// chunks in order, as the single-token kernel adds them) reduced over the SIMD-group as soon as
// they are computed, rather than kept for every token until the end (T NR fewer registers), then
// over the KS SIMD-groups as mvk_finish does. The single-token kernel's arithmetic: the lane's
// chunks added to zero in order, the SIMD-group sum, the KS sums in order.
template <typename F, typename X, uint CH, int NR, int KS, int T, int H, int C>
inline void mvk_one(F f, F f2, device const float* x, device float* y, device const float* gate, constant MVK& q, uint tg,
                    uint sg, uint lane, threadgroup float* red) {
    // (Each chunk's dot product rounded, then added: not fused with the sum, whatever the loop
    // around it, so that every kernel adds the chunks alike; see `mv_acc`.)
#pragma clang fp contract(off)
    constant MV& p = q.m;
    const bool pair = p.mode == 2 || p.mode == 3 || p.mode == 6;
    const uint r0 = tg * NR, nch = p.n_in / CH;
    uint cs[C];
    bool on[C];
    typename WDec<F, H == 8 ? 3 : H - 6>::T w[C][NR], w2[C][NR];
    for (int k = 0; k < C; k++) {
        uint c = sg * 32 + lane + uint(k) * KS * 32;
        on[k] = c < nch;
        cs[k] = on[k] ? c : nch - 1;
        for (int r = 0; r < NR; r++) wdec(f, min(r0 + r, p.n_out - 1), cs[k], w[k][r]);
        if (pair)
            for (int r = 0; r < NR; r++) wdec(f2, min(r0 + r, p.n_out - 1), cs[k], w2[k][r]);
    }
    for (int t = 0; t < T; t++) {
        float sa[NR], sb[NR];
        for (int r = 0; r < NR; r++) { sa[r] = 0.f; sb[r] = 0.f; }
        for (int k = 0; k < C; k++) {
            X xr;
            xload(x + (ulong)t * q.sx, cs[k], xr);
            for (int r = 0; r < NR; r++) {
                if (on[k]) sa[r] += wdot(f, w[k][r], cs[k], xr);
                if (pair && on[k]) sb[r] += wdot(f2, w2[k][r], cs[k], xr);
            }
        }
        for (int r = 0; r < NR; r++) {
            float va = simd_sum(0.f + sa[r]);
            float vb = pair ? simd_sum(0.f + sb[r]) : 0.f;
            if (KS == 1) {
                if (lane == 0 && r0 + r < p.n_out) mvk_out(y + (ulong)t * q.sy, r0 + r, p.mode, va, vb, gate + (ulong)t * q.sgate);
            } else if (lane == 0) {
                red[((sg * T + t) * NR + r) * 2] = va;
                red[((sg * T + t) * NR + r) * 2 + 1] = vb;
            }
        }
    }
    if (KS == 1) return;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0) return;
    for (uint i = lane; i < T * NR; i += 32) {
        uint t = i / NR, r = i % NR;
        if (r0 + r >= p.n_out) continue;
        float va = 0.f, vb = 0.f;
        for (int s = 0; s < KS; s++) {
            va += red[((s * T + t) * NR + r) * 2];
            vb += red[((s * T + t) * NR + r) * 2 + 1];
        }
        mvk_out(y + (ulong)t * q.sy, r0 + r, p.mode, va, vb, gate + (ulong)t * q.sgate);
    }
}
// Sweeps of 4 tokens, each bounded by the step's count and finished token by token (the
// previous kernels' structure, float rows): measured faster for 9-16 tokens of the oxinfer Q4 /
// Q4_0 rows and of Q8_0 rows on one SIMD-group.
template <int NR, int KS>
inline void mvk_tfinish(thread float* acc, thread float* acc2, device float* y, uint mode, uint r0, uint n_out, uint sg, uint lane,
                        threadgroup float* red, device const float* gate) {
    for (int r = 0; r < NR; r++) { acc[r] = simd_sum(acc[r]); acc2[r] = simd_sum(acc2[r]); }
    bool write = true;
    if (KS > 1) {
        if (lane == 0)
            for (int r = 0; r < NR; r++) { red[sg * 2 * NR + r] = acc[r]; red[sg * 2 * NR + NR + r] = acc2[r]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        write = sg == 0;
        if (write)
            for (int r = 0; r < NR; r++) {
                float a = 0.f, b = 0.f;
                for (int s = 0; s < KS; s++) { a += red[s * 2 * NR + r]; b += red[s * 2 * NR + NR + r]; }
                acc[r] = a; acc2[r] = b;
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (write && lane < NR && r0 + lane < n_out) {
        float a = acc[0], b = acc2[0];
        for (int r = 1; r < NR; r++) if (lane == uint(r)) { a = acc[r]; b = acc2[r]; }
        mvk_out(y, r0 + lane, mode, a, b, gate);
    }
}
template <typename F, typename X, uint CH, int NR, int KS>
inline void mvk_sweeps(F f, F f2, device const float* x, device float* y, constant MVK& q, uint tg, uint sg, uint lane,
                       threadgroup float* red, device const float* gate) {
    // (Each chunk's dot product rounded, then added: not fused with the sum, whatever the loop
    // around it, so that every kernel adds the chunks alike; see `mv_acc`.)
#pragma clang fp contract(off)
    constant MV& p = q.m;
    const bool pair = p.mode == 2 || p.mode == 3 || p.mode == 6;
    const uint r0 = tg * NR, nch = p.n_in / CH;
    uint rows[NR];
    for (int r = 0; r < NR; r++) rows[r] = min(r0 + r, p.n_out - 1);
    for (uint t0 = 0; t0 < q.ntok; t0 += 4) {
        const uint nt = min(q.ntok - t0, 4u);
        device const float* xg = x + (ulong)t0 * q.sx;
        float a[4][NR], b[4][NR];
        for (int t = 0; t < 4; t++)
            for (int r = 0; r < NR; r++) { a[t][r] = 0.f; b[t][r] = 0.f; }
        for (uint c = sg * 32 + lane; c < nch; c += KS * 32) {
            typename WDec<F, 0>::T w[NR];
            for (int r = 0; r < NR; r++) wdec(f, rows[r], c, w[r]);
            for (int t = 0; t < 4; t++) {
                if (uint(t) >= nt) continue;
                X xr;
                xload(xg + (ulong)t * q.sx, c, xr);
                for (int r = 0; r < NR; r++) a[t][r] += wdot(f, w[r], c, xr);
            }
            if (pair) {
                for (int r = 0; r < NR; r++) wdec(f2, rows[r], c, w[r]);
                for (int t = 0; t < 4; t++) {
                    if (uint(t) >= nt) continue;
                    X xr;
                    xload(xg + (ulong)t * q.sx, c, xr);
                    for (int r = 0; r < NR; r++) b[t][r] += wdot(f2, w[r], c, xr);
                }
            }
        }
        for (int t = 0; t < 4; t++) {
            if (uint(t) >= nt) break;
            uint tt = t0 + t;
            mvk_tfinish<NR, KS>(a[t], b[t], y + (ulong)tt * q.sy, p.mode, r0, p.n_out, sg, lane, red, gate + (ulong)tt * q.sgate);
        }
    }
}
// Kernels of up to 8 tokens run exactly T (no bounds on the token loops); of up to 16, as many
// as the step has in groups of 8, a sweep over the rows each (the rows' weights from the cache
// after the first): accumulators for 16 tokens measured 2-3 times slower than two sweeps.
template <typename F, typename X, uint CH, int NR, int KS, int T, int H, int C>
inline void mvk_body(F f, F f2, device const float* x, device float* y, constant MVK& q, uint tg, uint sg, uint lane,
                     threadgroup float* red, device const float* gate) {
    if (H >= 6 && T <= 8) {
        mvk_one<F, X, CH, NR, KS, T, H, C>(f, f2, x, y, gate, q, tg, sg, lane, red);
        return;
    }
    if (H == 2) {
        mvk_sweeps<F, X, CH, NR, KS>(f, f2, x, y, q, tg, sg, lane, red, gate);
        return;
    }
    if (T <= 8) {
        mvk_part<F, X, CH, NR, KS, T, H>(f, f2, x, y, gate, T, q, tg, sg, lane, red);
        return;
    }
    for (uint t0 = 0; t0 < q.ntok; t0 += 8) {
        if (t0 != 0) threadgroup_barrier(mem_flags::mem_threadgroup);
        mvk_part<F, X, CH, NR, KS, 8, H>(f, f2, x + (ulong)t0 * q.sx, y + (ulong)t0 * q.sy, gate + (ulong)t0 * q.sgate,
                                      min(q.ntok - t0, 8u), q, tg, sg, lane, red);
    }
}
#define MVK_SIG(NAME) \
kernel void NAME(device const uchar* wb [[buffer(0)]], device const uchar* sb [[buffer(1)]], device const float* x [[buffer(2)]], \
                 device float* y [[buffer(3)]], constant MVK& q [[buffer(4)]], device const uchar* wb2 [[buffer(5)]], \
                 device const uchar* sb2 [[buffer(6)]], device const uchar* ob [[buffer(7)]], device const uchar* ob2 [[buffer(8)]], \
                 device const float* gate [[buffer(9)]], \
                 uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], \
                 uint lane [[thread_index_in_simdgroup]])
#define MVK_RED(NR, KS, TT, H) (2 * NR * KS * (H == 2 ? 1 : (TT > 8 ? 8 : TT)))
// G groups of KS SIMD-groups a threadgroup, each its own NR rows as a threadgroup of the one-group
// kernel would take them (the same chunks per lane and reductions); the groups read the same x
// chunks at about the same time.
#define MVK_G(NR, KS, TT, H, G) threadgroup float redg[MVK_RED(NR, KS, TT, H) * G]; \
    threadgroup float* red = redg + (sg / KS) * MVK_RED(NR, KS, TT, H); \
    tg = tg * G + sg / KS; sg = sg % KS; constant MV& p = q.m;
#define MVK_RAW(NAME, T, X, CH, NR, KS, TT, H, G, C) \
MVK_SIG(NAME) { MVK_G(NR, KS, TT, H, G) \
    T f = { wb + p.w_off, p.gpr }; T f2 = { wb2 + p.w2_off, p.gpr }; \
    mvk_body<T, X, CH, NR, KS, TT, H, C>(f, f2, x, y, q, tg, sg, lane, red, gate); }
#define MVK_Q8(NAME, NR, KS, TT, H, G, C) \
MVK_SIG(NAME) { MVK_G(NR, KS, TT, H, G) \
    MQ8 f = { (device const char*)(wb + p.w_off), (device const float*)(sb + p.s_off), p.n_in }; \
    MQ8 f2 = { (device const char*)(wb2 + p.w2_off), (device const float*)(sb2 + p.s2_off), p.n_in }; \
    mvk_body<MQ8, X16, 16, NR, KS, TT, H, C>(f, f2, x, y, q, tg, sg, lane, red, gate); }
#define MVK_IQ4(NAME, NR, KS, TT, H, G, C) \
MVK_SIG(NAME) { threadgroup float kvt[16]; if (sg == 0 && lane < 16) kvt[lane] = kvalues_iq4nl[lane]; \
    threadgroup_barrier(mem_flags::mem_threadgroup); MVK_G(NR, KS, TT, H, G) \
    MIQ4_XS f = { wb + p.w_off, p.gpr, kvt }; MIQ4_XS f2 = { wb2 + p.w2_off, p.gpr, kvt }; \
    mvk_body<MIQ4_XS, X32, 32, NR, KS, TT, H, C>(f, f2, x, y, q, tg, sg, lane, red, gate); }
#define MVK_Q4(NAME, NR, KS, TT, H, G, C) \
MVK_SIG(NAME) { MVK_G(NR, KS, TT, H, G) \
    MQ4 f = { wb + p.w_off, (device const half*)(sb + p.s_off), p.n_in, (device const ushort*)(ob + p.o_off), p.gpr, p.gk }; \
    MQ4 f2 = { wb2 + p.w2_off, (device const half*)(sb2 + p.s2_off), p.n_in, (device const ushort*)(ob2 + p.o2_off), p.gpr, p.gk }; \
    mvk_body<MQ4, X32, 32, NR, KS, TT, H, C>(f, f2, x, y, q, tg, sg, lane, red, gate); }
#define MVK_GEN(NAME, W, NR, KS, TT, H, G, C) \
MVK_SIG(NAME) { MVK_G(NR, KS, TT, H, G) \
    MGen<W> f = { W{ { wb + p.w_off, p.gpr } } }; MGen<W> f2 = { W{ { wb2 + p.w2_off, p.gpr } } }; \
    mvk_body<MGen<W>, X32, 32, NR, KS, TT, H, C>(f, f2, x, y, q, tg, sg, lane, red, gate); }

// ---- RMSNorm over rows of n: out = x * rsqrt(mean(x^2) + eps) * w (w optional; `add`: out += ...).
// One threadgroup per row.
struct NP { uint n; float eps; uint has_w; uint add; uint x_stride; uint o_stride; float pre; uint _p; };
inline void rms_norm_body(device const float* x, device float* out, device const float* w, constant NP& p, uint r, uint tid, uint nt,
                          uint sg, uint lane, threadgroup float* red) {
    device const float* xr = x + (ulong)r * p.x_stride;
    device float* o = out + (ulong)r * p.o_stride;
    // Up to 8 values per thread (and their weights) are loaded at once and kept in registers (one
    // memory latency instead of a chain); longer rows loop.
    float xv[8], wv[8];
    bool fits = p.n <= 8 * nt;
    float s = 0.f;
    if (fits) {
        for (uint j = 0; j < 8; j++) { uint i = tid + j * nt; xv[j] = i < p.n ? xr[i] * p.pre : 0.f; }
        for (uint j = 0; j < 8; j++) { uint i = tid + j * nt; wv[j] = p.has_w != 0 && i < p.n ? w[i] : 1.f; }
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
                if (p.has_w != 0) v *= wv[j];
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

kernel void dec_rms_norm(device const float* x [[buffer(0)]], device float* out [[buffer(1)]], device const float* w [[buffer(2)]],
                         constant NP& p [[buffer(3)]], uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                         uint nt [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[32];
    rms_norm_body(x, out, w, p, r, tid, nt, sg, lane, red);
}

// ---- Attention inputs for one token: per-head RMSNorm (optional) and RoPE of the queries (into
// `qo`, contiguous heads) and keys (into the key cache at `pos`), values copied into the value
// cache. Threadgroups: nh query heads, then nkv key heads, then nkv value heads; one thread per
// dimension (hd threads). cos / sin of the position come from the host (`cs`: half cos, half sin).
struct QP { uint nh; uint nkv; uint hd; uint half_rot; uint q_stride; uint k_off; uint v_off; uint pos;
            uint interleaved; uint has_qn; uint has_kn; float eps; uint v_norm; uint v_from_k; uint _p0; uint _p1; };
// A cache entry of type KT (the general plans' caches are f32, or half rows with `gpu_kv=f16`:
// clamped to half's range).
template <typename KT> inline KT kv_store(float v) { return KT(v); }
template <> inline half kv_store<half>(float v) { return half(clamp(v, -65504.f, 65504.f)); }
template <typename CS, typename KT = float>
inline void qk_rope_body(device const float* qkv, device float* qo, device KT* kc, device KT* vc, device const float* qn,
                         device const float* kn, constant QP& p, uint pos, CS cs, uint h, uint i, uint sg, uint lane,
                         threadgroup float* buf, threadgroup float* red, device half* kh = nullptr, device half* vh = nullptr);
kernel void dec_qk_rope(device const float* qkv [[buffer(0)]], device float* qo [[buffer(1)]], device float* kc [[buffer(2)]],
                        device float* vc [[buffer(3)]], device const float* qn [[buffer(4)]], device const float* kn [[buffer(5)]],
                        constant QP& p [[buffer(6)]], constant float* cs [[buffer(7)]],
                        uint h [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float buf[512];
    threadgroup float red[16];
    qk_rope_body(qkv, qo, kc, vc, qn, kn, p, p.pos, cs, h, i, sg, lane, buf, red);
}
// Several tokens (grid y): token t at position pos + t, inputs and query outputs `ts.x` / `ts.y`
// floats apart, its cos / sin `ts.z` floats apart in `cs`. With `_p0` (prefill plans), the keys
// and values also go to `kh` / `vh` as half rows (`pfm_*` reads them), clamped to its range.
kernel void dec_qk_rope_t(device const float* qkv [[buffer(0)]], device float* qo [[buffer(1)]], device float* kc [[buffer(2)]],
                          device float* vc [[buffer(3)]], device const float* qn [[buffer(4)]], device const float* kn [[buffer(5)]],
                          constant QP& p [[buffer(6)]], device const float* cs [[buffer(7)]], constant uint4& ts [[buffer(8)]],
                          device half* kh [[buffer(9)]], device half* vh [[buffer(10)]],
                          uint2 tg [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float buf[512];
    threadgroup float red[16];
    uint t = tg.y;
    qk_rope_body(qkv + (ulong)t * ts.x, qo + (ulong)t * ts.y, kc, vc, qn, kn, p, p.pos + t, cs + (ulong)t * ts.z, tg.x, i, sg, lane, buf, red,
                 kh, vh);
}
// The same over half caches (`gpu_kv=f16`).
kernel void dec_qk_rope_h(device const float* qkv [[buffer(0)]], device float* qo [[buffer(1)]], device half* kc [[buffer(2)]],
                          device half* vc [[buffer(3)]], device const float* qn [[buffer(4)]], device const float* kn [[buffer(5)]],
                          constant QP& p [[buffer(6)]], constant float* cs [[buffer(7)]],
                          uint h [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float buf[512];
    threadgroup float red[16];
    qk_rope_body(qkv, qo, kc, vc, qn, kn, p, p.pos, cs, h, i, sg, lane, buf, red);
}
kernel void dec_qk_rope_t_h(device const float* qkv [[buffer(0)]], device float* qo [[buffer(1)]], device half* kc [[buffer(2)]],
                            device half* vc [[buffer(3)]], device const float* qn [[buffer(4)]], device const float* kn [[buffer(5)]],
                            constant QP& p [[buffer(6)]], device const float* cs [[buffer(7)]], constant uint4& ts [[buffer(8)]],
                            uint2 tg [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float buf[512];
    threadgroup float red[16];
    uint t = tg.y;
    qk_rope_body(qkv + (ulong)t * ts.x, qo + (ulong)t * ts.y, kc, vc, qn, kn, p, p.pos + t, cs + (ulong)t * ts.z, tg.x, i, sg, lane, buf, red);
}
template <typename CS, typename KT>
inline void qk_rope_body(device const float* qkv, device float* qo, device KT* kc, device KT* vc, device const float* qn,
                         device const float* kn, constant QP& p, uint pos, CS cs, uint h, uint i, uint sg, uint lane,
                         threadgroup float* buf, threadgroup float* red, device half* kh, device half* vh) {
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
        vc[((ulong)pos * p.nkv + hh) * hd + i] = kv_store<KT>(v);
        if (p._p0 != 0) vh[((ulong)pos * p.nkv + hh) * hd + i] = half(clamp(v, -65504.f, 65504.f));
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
    else {
        kc[((ulong)pos * p.nkv + hh) * hd + i] = kv_store<KT>(v);
        if (p._p0 != 0) kh[((ulong)pos * p.nkv + hh) * hd + i] = half(clamp(v, -65504.f, 65504.f));
    }
}

// ---- Prefill plans (`pfm_*`): cache rows p.y .. p.y + grid y - 1 (p.x float4s each) of keys
// and values as half rows, clamped to its range.
kernel void pf_kv_half(device const float4* kc [[buffer(0)]], device const float4* vc [[buffer(1)]], device half4* kh [[buffer(2)]],
                       device half4* vh [[buffer(3)]], constant uint2& p [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.x) return;
    ulong at = (ulong)(p.y + gid.y) * p.x + gid.x;
    kh[at] = half4(clamp(kc[at], -65504.f, 65504.f));
    vh[at] = half4(clamp(vc[at], -65504.f, 65504.f));
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
// SEL (QSA sparse attention, metal_qsa.metal): key j is cache position idx[j] rather than j. PT: how
// the parameters arrive (per-token copies in the prefill kernels).
// R (`dec_attnr_*`, the token's RoPE fused): the caller has put the scaled queries in `qs`, and
// position `jn`'s key and value rows (as the cache holds them) in `kn` / `vn`, read instead of the
// cache's row there.
template <int D, int G, bool SEL = false, typename PT = constant AP&, typename KT = float, bool R = false>
inline void attn_body(device const float* q, device const KT* kc, device const KT* vc, device float* out, device const float* gate,
                      device float* part, PT p, uint hg, uint split, uint tid, uint sg, uint lane, threadgroup float* sm,
                      threadgroup float* so, threadgroup float* qs, device const uint* idx = nullptr,
                      threadgroup const float* kn = nullptr, threadgroup const float* vn = nullptr, uint jn = 0xFFFFFFFFu) {
    constexpr uint HD = D * 32;
    uint h0 = hg * G;
    uint kvh = h0 / (p.nh / p.nkv);
    if (!R) {
        // The scaled queries in threadgroup memory (every lane reads all of them).
        for (uint i = tid; i < G * HD; i += 256) qs[i] = q[h0 * HD + i] * p.scale;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
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
        if (R && j == jn && j < j1) {
            threadgroup const float4* kr = (threadgroup const float4*)kn;
            float4 acc[G];
            for (int g = 0; g < G; g++) acc[g] = 0.f;
            for (uint d = 0; d < HD / 4; d++) {
                float4 k4 = kr[d];
                for (int g = 0; g < G; g++) acc[g] = fma(k4, ((threadgroup const float4*)(qs + g * HD))[d], acc[g]);
            }
            for (int g = 0; g < G; g++) s[g] = (acc[g].x + acc[g].y) + (acc[g].z + acc[g].w);
        } else if (j < j1) {
            device const vec<KT, 4>* kr = (device const vec<KT, 4>*)(kc + (ulong)(SEL ? idx[j] : j) * rs + (ulong)kvh * HD);
            float4 acc[G];
            for (int g = 0; g < G; g++) acc[g] = 0.f;
            for (uint d = 0; d < HD / 4; d++) {
                float4 k4 = float4(kr[d]);
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
        device const KT* vb = vc + (ulong)base * rs + (ulong)kvh * HD + lane * D;
        uint t0 = 0;
        if (!SEL) {
            // Whole groups of VB keys: their value rows are loaded together (one memory latency
            // instead of a chain), then accumulated in the same order as below (VB x D floats of
            // registers: fewer keys for wider heads, which would spill).
            constexpr int VB = D <= 4 ? 8 : D == 8 ? 4 : 2;
            for (; t0 + VB <= nb; t0 += VB) {
                float v8[VB][D];
                for (int u = 0; u < VB; u++)
                    for (int d = 0; d < D; d++) v8[u][d] = float(vb[(t0 + u) * rs + d]);
                if (R && jn - base - t0 < uint(VB))
                    for (int u = 0; u < VB; u++)
                        if (base + t0 + u == jn)
                            for (int d = 0; d < D; d++) v8[u][d] = vn[lane * D + d];
                for (int u = 0; u < VB; u++)
                    for (int g = 0; g < G; g++) {
                        float et = simd_shuffle(e[g], t0 + u);
                        for (int d = 0; d < D; d++) o[g][d] = fma(et, v8[u][d], o[g][d]);
                    }
            }
        }
        for (uint t = t0; t < nb; t++) {
            device const KT* vr = SEL ? vc + (ulong)idx[base + t] * rs + (ulong)kvh * HD + lane * D : vb + t * rs;
            float v[D];
            for (int d = 0; d < D; d++) v[d] = float(vr[d]);
            if (R && base + t == jn)
                for (int d = 0; d < D; d++) v[d] = vn[lane * D + d];
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
// Over half caches (`gpu_kv=f16`; more query heads per threadgroup for heads of 256 and 512).
// (Lanes splitting each key's dimensions, a step's loads then whole rows, were no faster:
// `dec_attn_bench` level for heads of 128 and 256, 13% slower for heads of 64 at 32k keys.)
#define ATTN_H(D, G) \
kernel void dec_attn_h_##D##_##G(device const float* q [[buffer(0)]], device const half* kc [[buffer(1)]], device const half* vc [[buffer(2)]], \
                         device float* out [[buffer(3)]], device const float* gate [[buffer(4)]], constant AP& p [[buffer(5)]], \
                         device float* part [[buffer(6)]], \
                         uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float sm[16]; threadgroup float so[8 * D * 32]; threadgroup float qs[G * D * 32]; \
    attn_body<D, G, false, constant AP&, half>(q, kc, vc, out, gate, part, p, tg.x, tg.y, tid, sg, lane, sm, so, qs); }
ATTN_H(2, 1) ATTN_H(2, 2) ATTN_H(2, 4) ATTN_H(2, 8) ATTN_H(4, 1) ATTN_H(4, 2) ATTN_H(4, 4) ATTN_H(8, 1) ATTN_H(8, 2) ATTN_H(8, 4)
ATTN_H(16, 1) ATTN_H(16, 2) ATTN_H(16, 4)
// ---- One-token attention with the token's RoPE fused (`dec_attnr_*`: dec_qk_rope's op folded in,
// a dispatch and a barrier fewer per layer). Every threadgroup prepares its G query heads and,
// when the layer writes keys and values (`rp.nkv` > 0), its key head's new key and value row,
// with qk_rope_body's arithmetic (the same bits: the norms' SIMD-group sums over the same 32
// dimensions, added in the same order); the new rows are used from threadgroup memory as the
// cache holds them (`kv_store`), and one threadgroup per key head (the first of its query heads,
// split 0) stores them in the caches, as split 0 stores the roped queries in `qo`. Nothing in the
// dispatch reads the cache's row at the token's position.
template <int D, int G, typename KT>
inline void attnr_prep(device const float* qkv, device float* qo, device KT* kc, device KT* vc, device const float* qn,
                       device const float* kn, constant QP& rp, constant float* cs, constant AP& p, uint hg, uint split, uint tid,
                       uint sg, uint lane, threadgroup float* buf, threadgroup float* red, threadgroup float* qs,
                       threadgroup float* knew, threadgroup float* vnew) {
    constexpr uint HD = D * 32;
    constexpr uint NV = ((G + 2) * HD + 255) / 256;
    uint h0 = hg * G, kvh = h0 / (p.nh / p.nkv);
    uint nv = (rp.nkv != 0 ? G + 2 : G) * HD;
    float v[NV];
    for (uint k = 0; k < NV; k++) {
        uint x = tid + 256 * k;
        v[k] = 0.f;
        if (x >= nv) continue;
        uint r = x / HD, i = x % HD;
        uint kind = r < G ? 0 : (r == G ? 1 : 2);
        float a = kind == 0 ? qkv[(h0 + r) * rp.q_stride + i]
                : (kind == 1 || rp.v_from_k != 0) ? qkv[rp.k_off + kvh * HD + i] : qkv[rp.v_off + kvh * HD + i];
        v[k] = a;
        float s = simd_sum(a * a);
        if (lane == 0) red[r * 16 + i / 32] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0; k < NV; k++) {
        uint x = tid + 256 * k;
        if (x >= nv) continue;
        uint r = x / HD, i = x % HD;
        uint kind = r < G ? 0 : (r == G ? 1 : 2);
        bool norm = kind == 0 ? rp.has_qn != 0 : (kind == 1 ? rp.has_kn != 0 : rp.v_norm != 0);
        if (norm) {
            float s = 0.f;
            for (uint j = 0; j < HD / 32; j++) s += red[r * 16 + j];
            v[k] = v[k] * rsqrt(s / float(HD) + rp.eps);
            if (kind == 0) v[k] *= qn[i];
            else if (kind == 1) v[k] *= kn[i];
        }
        buf[x] = v[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool writer = split == 0 && h0 % (p.nh / p.nkv) == 0;
    uint hr = rp.half_rot;
    for (uint k = 0; k < NV; k++) {
        uint x = tid + 256 * k;
        if (x >= nv) continue;
        uint r = x / HD, i = x % HD;
        uint kind = r < G ? 0 : (r == G ? 1 : 2);
        float o = v[k];
        if (kind == 2) {
            KT st = kv_store<KT>(o);
            vnew[i] = float(st);
            if (writer) vc[((ulong)rp.pos * rp.nkv + kvh) * HD + i] = st;
            continue;
        }
        threadgroup const float* b = buf + r * HD;
        if (i < 2 * hr) {
            if (rp.interleaved != 0) {
                uint j = i / 2;
                float c = cs[j], sn = cs[hr + j];
                float x1 = b[2 * j], x2 = b[2 * j + 1];
                o = (i % 2 == 0) ? x1 * c - x2 * sn : x2 * c + x1 * sn;
            } else {
                uint j = i % hr;
                float c = cs[j], sn = cs[hr + j];
                float x1 = b[j], x2 = b[j + hr];
                o = (i < hr) ? x1 * c - x2 * sn : x2 * c + x1 * sn;
            }
        }
        if (kind == 0) {
            qs[r * HD + i] = o * p.scale;
            if (split == 0) qo[(h0 + r) * HD + i] = o;
        } else {
            KT st = kv_store<KT>(o);
            knew[i] = float(st);
            if (writer) kc[((ulong)rp.pos * rp.nkv + kvh) * HD + i] = st;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
#define ATTNR(NAME, D, G, KT) \
kernel void NAME(device const float* qkv [[buffer(0)]], device KT* kc [[buffer(1)]], device KT* vc [[buffer(2)]], \
                 device float* out [[buffer(3)]], device const float* gate [[buffer(4)]], constant AP& p [[buffer(5)]], \
                 device float* part [[buffer(6)]], constant float* cs [[buffer(7)]], constant QP& rp [[buffer(8)]], \
                 device const float* qn [[buffer(9)]], device const float* kn [[buffer(10)]], device float* qo [[buffer(11)]], \
                 uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float sm[16]; threadgroup float so[(G + 2 > 8 ? G + 2 : 8) * D * 32]; threadgroup float qs[G * D * 32]; \
    threadgroup float nkv[2 * D * 32]; threadgroup float red[(G + 2) * 16]; \
    attnr_prep<D, G, KT>(qkv, qo, kc, vc, qn, kn, rp, cs, p, tg.x, tg.y, tid, sg, lane, so, red, qs, nkv, nkv + D * 32); \
    attn_body<D, G, false, constant AP&, KT, true>(qkv, kc, vc, out, gate, part, p, tg.x, tg.y, tid, sg, lane, sm, so, qs, nullptr, \
                                                   nkv, nkv + D * 32, rp.nkv != 0 ? rp.pos : 0xFFFFFFFFu); }
ATTNR(dec_attnr_2_1, 2, 1, float) ATTNR(dec_attnr_2_2, 2, 2, float) ATTNR(dec_attnr_2_4, 2, 4, float) ATTNR(dec_attnr_2_8, 2, 8, float)
ATTNR(dec_attnr_4_1, 4, 1, float) ATTNR(dec_attnr_4_2, 4, 2, float) ATTNR(dec_attnr_4_4, 4, 4, float) ATTNR(dec_attnr_8_1, 8, 1, float)
ATTNR(dec_attnr_8_2, 8, 2, float) ATTNR(dec_attnr_16_1, 16, 1, float)
ATTNR(dec_attnr_h_2_1, 2, 1, half) ATTNR(dec_attnr_h_2_2, 2, 2, half) ATTNR(dec_attnr_h_2_4, 2, 4, half) ATTNR(dec_attnr_h_2_8, 2, 8, half)
ATTNR(dec_attnr_h_4_1, 4, 1, half) ATTNR(dec_attnr_h_4_2, 4, 2, half) ATTNR(dec_attnr_h_4_4, 4, 4, half)
ATTNR(dec_attnr_h_8_1, 8, 1, half) ATTNR(dec_attnr_h_8_2, 8, 2, half) ATTNR(dec_attnr_h_8_4, 8, 4, half)
ATTNR(dec_attnr_h_16_1, 16, 1, half) ATTNR(dec_attnr_h_16_2, 16, 2, half) ATTNR(dec_attnr_h_16_4, 16, 4, half)
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

// ---- One-token attention split finely over the keys (`dec_attv_*`, `dec_attvr_*` with the
// token's RoPE fused as `dec_attnr_*`): a threadgroup per (G query heads, chunk of keys) as
// `dec_attn_*`, but the lanes of a SIMD-group across a head's dimensions (D each) and the
// SIMD-groups across the keys (SIMD-group sg: keys j0 + sg + 8 m, KB at a time with their key
// and value loads issued together, the first ones before the queries are made), each score a
// SIMD-group sum: a key is one coalesced row load, chunks of 32 keys and more fill the GPU at
// short ranges, and the last of a head's threadgroups to finish merges the chunks (in their
// order), no combine stage after it. `cnt`: a counter per head group (zero between dispatches:
// the merging threadgroup resets it).
template <int D> struct AvRows {
    static constexpr constant int KB = D <= 4 ? 4 : D == 8 ? 2 : 1;
    float4 k[KB][D / 4], v[KB][D / 4];
};
template <int D, typename KT>
inline void attv_load(thread AvRows<D>& r, device const KT* kb, device const KT* vb, ulong rs, uint base, uint j1) {
    for (int u = 0; u < AvRows<D>::KB; u++) {
        uint j = min(base + 8 * u, j1 - 1);
        device const vec<KT, 4>* kr = (device const vec<KT, 4>*)(kb + (ulong)j * rs);
        device const vec<KT, 4>* vr = (device const vec<KT, 4>*)(vb + (ulong)j * rs);
        for (int d = 0; d < D / 4; d++) {
            r.k[u][d] = float4(kr[d]);
            r.v[u][d] = float4(vr[d]);
        }
    }
}
// The rows of SIMD-group sg's first keys (loaded before the queries are made: their latency
// passes during the queries' barriers).
template <int D, typename KT>
inline void attv_first(thread AvRows<D>& r, device const KT* kc, device const KT* vc, constant AP& p, uint hg, uint G, uint split,
                       uint sg, uint lane) {
    constexpr uint HD = D * 32;
    uint kvh = hg * G / (p.nh / p.nkv);
    ulong rs = (ulong)p.nkv * HD;
    uint j0 = p.lo + split * p.chunk, j1 = min(p.len, j0 + p.chunk);
    attv_load<D, KT>(r, kc + (ulong)kvh * HD + lane * D, vc + (ulong)kvh * HD + lane * D, rs, j0 + sg, j1);
}
template <int D, int G, typename KT, bool R>
inline void attv_body(thread AvRows<D>& rw, device const float* q, device const KT* kc, device const KT* vc, device float* out,
                      device const float* gate, device float* part, device atomic_uint* cnt, constant AP& p, uint hg, uint split,
                      uint tid, uint sg, uint lane, threadgroup float* sm, threadgroup float* so, threadgroup float* qs,
                      threadgroup const float* kn, threadgroup const float* vn, uint jn) {
    constexpr uint HD = D * 32;
    constexpr int KB = AvRows<D>::KB;
    constexpr int D4 = D / 4;
    uint h0 = hg * G, kvh = h0 / (p.nh / p.nkv);
    if (!R) {
        for (uint i = tid; i < G * HD; i += 256) qs[i] = q[h0 * HD + i] * p.scale;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float4 qv[G][D4], o[G][D4];
    float m[G], l[G];
    for (int g = 0; g < G; g++) {
        m[g] = -INFINITY;
        l[g] = 0.f;
        for (int d = 0; d < D4; d++) {
            qv[g][d] = ((threadgroup const float4*)(qs + g * HD + lane * D))[d];
            o[g][d] = 0.f;
        }
    }
    ulong rs = (ulong)p.nkv * HD;
    device const KT* kb = kc + (ulong)kvh * HD + lane * D;
    device const KT* vb = vc + (ulong)kvh * HD + lane * D;
    uint j0 = p.lo + split * p.chunk, j1 = min(p.len, j0 + p.chunk);
    for (uint base = j0 + sg; base < j1; base += 8 * KB) {
        // (The token's own row as the threadgroup made it.)
        if (R && jn - base < uint(8 * KB))
            for (int u = 0; u < KB; u++)
                if (base + 8 * u == jn)
                    for (int d = 0; d < D4; d++) {
                        rw.k[u][d] = ((threadgroup const float4*)(kn + lane * D))[d];
                        rw.v[u][d] = ((threadgroup const float4*)(vn + lane * D))[d];
                    }
        for (int g = 0; g < G; g++) {
            float s[KB];
            float mx = m[g];
            for (int u = 0; u < KB; u++) {
                float4 acc = 0.f;
                for (int d = 0; d < D4; d++) acc = fma(rw.k[u][d], qv[g][d], acc);
                float dot = simd_sum((acc.x + acc.y) + (acc.z + acc.w));
                s[u] = base + 8 * u < j1 ? dot : -INFINITY;
                mx = max(mx, s[u]);
            }
            float al = mx == -INFINITY ? 1.f : exp(m[g] - mx);
            float ls = l[g] * al;
            for (int d = 0; d < D4; d++) o[g][d] *= al;
            for (int u = 0; u < KB; u++) {
                float e = s[u] == -INFINITY ? 0.f : exp(s[u] - mx);
                ls += e;
                for (int d = 0; d < D4; d++) o[g][d] = fma(e, rw.v[u][d], o[g][d]);
            }
            l[g] = ls;
            m[g] = mx;
        }
        if (base + 8 * KB < j1) attv_load<D, KT>(rw, kb, vb, rs, base + 8 * KB, j1);
    }
    // The SIMD-groups combined, one head at a time: the output, or the chunk's partial result.
    for (int g = 0; g < G; g++) {
        uint h = h0 + g;
        if (lane == 0) { sm[sg] = m[g]; sm[8 + sg] = l[g]; }
        for (int d = 0; d < D4; d++) ((threadgroup float4*)(so + sg * HD + lane * D))[d] = o[g][d];
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
    if (p.nsplit == 1) return;
    // The last of the head group's threadgroups merges the chunks (`dec_attn_combine`'s
    // arithmetic; the others' partials made visible by the fences around the counter).
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) sm[0] = atomic_fetch_add_explicit(cnt + hg, 1u, memory_order_relaxed) == p.nsplit - 1 ? 1.f : 0.f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sm[0] == 0.f) return;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    uint ns = p.nsplit;
    for (uint i = tid; i < G * HD; i += 256) {
        uint h = h0 + i / HD, d = i % HD;
        device const float* ph = part + (ulong)h * ns * (HD + 2);
        float M = -INFINITY;
        for (uint s = 0; s < ns; s++) M = max(M, ph[s * (HD + 2) + HD]);
        float L = 0.f, acc = 0.f;
        for (uint s = 0; s < ns; s++) {
            float ms = ph[s * (HD + 2) + HD];
            if (ms == -INFINITY) continue;
            float f = exp(ms - M);
            L += ph[s * (HD + 2) + HD + 1] * f;
            acc += ph[s * (HD + 2) + d] * f;
        }
        float r = acc / L;
        if (p.has_gate != 0) { float gt = gate[p.g_off + h * p.g_stride + d]; r *= 1.0f / (1.0f + exp(-gt)); }
        out[h * HD + d] = r;
    }
    if (tid == 0) atomic_store_explicit(cnt + hg, 0u, memory_order_relaxed);
}
#define ATTV(NAME, D, G, KT) \
kernel void NAME(device const float* q [[buffer(0)]], device const KT* kc [[buffer(1)]], device const KT* vc [[buffer(2)]], \
                 device float* out [[buffer(3)]], device const float* gate [[buffer(4)]], constant AP& p [[buffer(5)]], \
                 device float* part [[buffer(6)]], device atomic_uint* cnt [[buffer(7)]], \
                 uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float sm[16]; threadgroup float so[8 * D * 32]; threadgroup float qs[G * D * 32]; \
    AvRows<D> rw; attv_first<D, KT>(rw, kc, vc, p, tg.x, G, tg.y, sg, lane); \
    attv_body<D, G, KT, false>(rw, q, kc, vc, out, gate, part, cnt, p, tg.x, tg.y, tid, sg, lane, sm, so, qs, nullptr, nullptr, 0xFFFFFFFFu); }
#define ATTVR(NAME, D, G, KT) \
kernel void NAME(device const float* qkv [[buffer(0)]], device KT* kc [[buffer(1)]], device KT* vc [[buffer(2)]], \
                 device float* out [[buffer(3)]], device const float* gate [[buffer(4)]], constant AP& p [[buffer(5)]], \
                 device float* part [[buffer(6)]], constant float* cs [[buffer(7)]], constant QP& rp [[buffer(8)]], \
                 device const float* qn [[buffer(9)]], device const float* kn [[buffer(10)]], device float* qo [[buffer(11)]], \
                 device atomic_uint* cnt [[buffer(12)]], \
                 uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float sm[16]; threadgroup float so[(G + 2 > 8 ? G + 2 : 8) * D * 32]; threadgroup float qs[G * D * 32]; \
    threadgroup float nkv[2 * D * 32]; threadgroup float red[(G + 2) * 16]; \
    AvRows<D> rw; attv_first<D, KT>(rw, kc, vc, p, tg.x, G, tg.y, sg, lane); \
    attnr_prep<D, G, KT>(qkv, qo, kc, vc, qn, kn, rp, cs, p, tg.x, tg.y, tid, sg, lane, so, red, qs, nkv, nkv + D * 32); \
    attv_body<D, G, KT, true>(rw, qkv, kc, vc, out, gate, part, cnt, p, tg.x, tg.y, tid, sg, lane, sm, so, qs, nkv, nkv + D * 32, \
                              rp.nkv != 0 ? rp.pos : 0xFFFFFFFFu); }
#define ATTV_ALL(D, G) ATTV(dec_attv_##D##_##G, D, G, float) ATTV(dec_attv_h_##D##_##G, D, G, half) \
    ATTVR(dec_attvr_##D##_##G, D, G, float) ATTVR(dec_attvr_h_##D##_##G, D, G, half)
// (Heads of 128 and more: a lane's dimensions are whole float4s.)
ATTV_ALL(4, 1) ATTV_ALL(4, 2) ATTV_ALL(4, 4)
ATTV_ALL(8, 1) ATTV_ALL(8, 2) ATTV_ALL(8, 4) ATTV_ALL(16, 1) ATTV_ALL(16, 2)

// ---- Gated DeltaNet ------------------------------------------------------------------------------
// Causal depthwise conv (kernel kc) with SiLU, one thread per channel; the state [kc-1][dim] holds
// the earlier inputs (oldest first) and shifts in this one. The state is read from `si` and
// written to `st` (the same buffer but in a greedy chain's passes, which alternate between two
// copies of the recurrent states, see `MetalPlan::greedy`).
struct GC { uint dim; uint kc; };
inline void gdn_conv_body(device const float* x, device float* out, device const float* si, device float* st, device const float* w,
                          constant GC& p, uint ch) {
    uint kc = p.kc;
    float xv = x[ch];
    float s = 0.f;
    for (uint j = 0; j + 1 < kc; j++) s += w[ch * kc + j] * si[j * p.dim + ch];
    s += w[ch * kc + kc - 1] * xv;
    out[ch] = s / (1.0f + exp(-s));
    for (uint j = 0; j + 2 < kc; j++) st[j * p.dim + ch] = si[(j + 1) * p.dim + ch];
    st[(kc - 2) * p.dim + ch] = xv;
}
kernel void dec_gdn_conv(device const float* x [[buffer(0)]], device float* out [[buffer(1)]], device float* st [[buffer(2)]],
                         device const float* w [[buffer(3)]], constant GC& p [[buffer(4)]], device const float* si [[buffer(5)]],
                         uint ch [[thread_position_in_grid]]) {
    if (ch >= p.dim) return;
    gdn_conv_body(x, out, si, st, w, p, ch);
}
// ts: tokens, input and output strides.
kernel void dec_gdn_conv_t(device const float* x [[buffer(0)]], device float* out [[buffer(1)]], device float* st [[buffer(2)]],
                           device const float* w [[buffer(3)]], constant GC& p [[buffer(4)]], constant uint4& ts [[buffer(5)]],
                           uint ch [[thread_position_in_grid]]) {
    if (ch >= p.dim) return;
    for (uint t = 0; t < ts.x; t++) gdn_conv_body(x + (ulong)t * ts.y, out + (ulong)t * ts.z, st, st, w, p, ch);
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
inline void gdn_rule_body(device const float* c, device const float* z, device const float* b, device const float* a, device float* out,
                          device const float* si, device float* state, device const float* cst, constant GR& p, uint hv, uint tid, uint sg, uint lane,
                          threadgroup float* qs, threadgroup float* ks, threadgroup float* red, threadgroup float* red2) {
    // cst: decay_coef[nv], dt_bias[nv], norm_w[hdv]
    uint hdk = p.hdk, hdv = p.hdv, np = hdk / 32;
    uint hk = (p.tiled & 1u) != 0 ? hv % p.nk : hv / (p.nv / p.nk);
    uint part = tid / hdv, j = tid % hdv;
    ulong so = (ulong)hv * hdk * hdv + part * 32 * hdv + j;
    device const float* Si = si + so;
    device float* S = state + so;
    // The state, the value and the gate are loaded first, so that their latency passes during the
    // q / k norms' barriers (the decay is applied below: the same products).
    float s[32];
    for (uint d = 0; d < 32; d++) s[d] = Si[d * hdv];
    float vj = c[2 * p.key_dim + hv * hdv + j];
    float zz = tid < hdv ? z[hv * hdv + j] : 0.f;
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
    float kv = 0.f;
    for (uint d = 0; d < 32; d++) {
        s[d] = s[d] * decay;
        kv += s[d] * (ks[part * 32 + d] * kinv);
    }
    red[part * hdv + j] = kv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float kvm = 0.f;
    for (uint i = 0; i < np; i++) kvm += red[i * hdv + j];
    float delta = (vj - kvm) * beta;
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
        out[hv * hdv + j] = ov * r * cst[2 * p.nv + j] * ((p.tiled & 2u) != 0 ? 1.0f / (1.0f + precise::exp(-zz)) : zz / (1.0f + exp(-zz)));
    }
}
kernel void dec_gdn_rule(device const float* c [[buffer(0)]], device const float* z [[buffer(1)]], device const float* b [[buffer(2)]],
                         device const float* a [[buffer(3)]], device float* out [[buffer(4)]], device float* state [[buffer(5)]],
                         device const float* cst [[buffer(6)]], constant GR& p [[buffer(7)]], device const float* si [[buffer(8)]],
                         uint hv [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float qs[256], ks[256], red[1024], red2[32];
    gdn_rule_body(c, z, b, a, out, si, state, cst, p, hv, tid, sg, lane, qs, ks, red, red2);
}
// ts: tokens, strides of c, of z / b / a (one projection buffer) and of out; the tokens run in
// order, each exactly as the single-token kernel.
kernel void dec_gdn_rule_t(device const float* c [[buffer(0)]], device const float* z [[buffer(1)]], device const float* b [[buffer(2)]],
                           device const float* a [[buffer(3)]], device float* out [[buffer(4)]], device float* state [[buffer(5)]],
                           device const float* cst [[buffer(6)]], constant GR& p [[buffer(7)]], constant uint4& ts [[buffer(8)]],
                           uint hv [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float qs[256], ks[256], red[1024], red2[32];
    for (uint t = 0; t < ts.x; t++) {
        ulong pp = (ulong)t * ts.z;
        gdn_rule_body(c + (ulong)t * ts.y, z + pp, b + pp, a + pp, out + (ulong)t * ts.w, state, state, cst, p, hv, tid, sg, lane, qs, ks, red, red2);
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
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

// Several tokens (grid y): ts = strides of g, u, out.
kernel void dec_glu_t(device const float* g [[buffer(0)]], device const float* u [[buffer(1)]], device float* out [[buffer(2)]],
                      constant uint2& p [[buffer(3)]], constant uint4& ts [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x, t = gid.y;
    if (i >= p.x) return;
    float a = g[(ulong)t * ts.x + i];
    float act;
    if (p.y != 0) {
        float tt = 0.7978845608f * (a + 0.044715f * a * a * a);
        act = 0.5f * a * (1.0f + precise::tanh(tt));
    } else {
        act = a / (1.0f + exp(-a));
    }
    out[(ulong)t * ts.z + i] = act * u[(ulong)t * ts.y + i];
}

// ---- Prefill plans: a row's power-of-two scale for the matrix-unit GEMMs' half rows (the MMA
// source's `row_scale`, for kernels of this one).
inline float pf_row_scale(float m, thread float& inv) {
    int e = 0;
    if (m > 0.f && isfinite(m)) { frexp(m, e); e = clamp(14 - e, -100, 100); }
    inv = ldexp(1.0f, -e);
    return ldexp(1.0f, e);
}
inline float pf_tg_max(float m, uint sg, uint lane, threadgroup float* red) {
    m = simd_max(m);
    if (lane == 0) red[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return max(max(max(red[0], red[1]), max(red[2], red[3])), max(max(red[4], red[5]), max(red[6], red[7])));
}
// dec_glu_t's rows (the same values), one threadgroup of 256 per token, also as the scaled half
// rows `x_half` makes of them (`xh`, `rs`) for the next GEMM (only those with ts.w: nothing reads
// the f32 rows).
#define PF_GLU_V 64
kernel void pf_glu_half(device const float* g [[buffer(0)]], device const float* u [[buffer(1)]], device float* out [[buffer(2)]],
                        constant uint2& p [[buffer(3)]], constant uint4& ts [[buffer(4)]], device half* xh [[buffer(5)]],
                        device float* rs [[buffer(6)]], uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8];
    float v[PF_GLU_V];
    float m = 0.f;
    for (uint j = 0; j < PF_GLU_V; j++) {
        uint i = tid + j * 256;
        if (i >= p.x) break;
        float a = g[(ulong)t * ts.x + i];
        float act;
        if (p.y != 0) {
            float tt = 0.7978845608f * (a + 0.044715f * a * a * a);
            act = 0.5f * a * (1.0f + precise::tanh(tt));
        } else {
            act = a / (1.0f + exp(-a));
        }
        v[j] = act * u[(ulong)t * ts.y + i];
        if (ts.w == 0) out[(ulong)t * ts.z + i] = v[j];
        m = max(m, abs(v[j]));
    }
    m = pf_tg_max(m, sg, lane, red);
    float inv, s = pf_row_scale(m, inv);
    device half* o = xh + (ulong)t * p.x;
    for (uint j = 0; j < PF_GLU_V; j++) {
        uint i = tid + j * 256;
        if (i >= p.x) break;
        o[i] = half(v[j] * s);
    }
    if (tid == 0) rs[t] = inv;
}
// RMSNorm of token rows (dec_rms_norm's, summed in another order), one threadgroup of 256 per
// token, n % 4 == 0 and n <= 256 * 4 * PF_NORM_V: the f32 rows (unless p._p: nothing reads them)
// and their scaled half rows for the next GEMM (x_half's).
#define PF_NORM_V 8
kernel void pf_norm_half(device const float* x [[buffer(0)]], device float* out [[buffer(1)]], device const float* w [[buffer(2)]],
                         constant NP& p [[buffer(3)]], device half* xh [[buffer(4)]], device float* rs [[buffer(5)]],
                         uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8];
    threadgroup float red2[8];
    device const float4* xr = (device const float4*)(x + (ulong)r * p.x_stride);
    device float4* o = (device float4*)(out + (ulong)r * p.o_stride);
    device const float4* w4 = (device const float4*)w;
    uint n4 = p.n / 4;
    float4 v[PF_NORM_V];
    float s = 0.f;
    for (uint j = 0; j < PF_NORM_V; j++) {
        uint i = tid + j * 256;
        v[j] = i < n4 ? xr[i] * p.pre : 0.f;
        s += dot(v[j], v[j]);
    }
    s = simd_sum(s);
    if (lane == 0) red[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    s = ((red[0] + red[1]) + (red[2] + red[3])) + ((red[4] + red[5]) + (red[6] + red[7]));
    float k = rsqrt(s / float(p.n) + p.eps);
    float m = 0.f;
    for (uint j = 0; j < PF_NORM_V; j++) {
        uint i = tid + j * 256;
        if (i < n4) {
            float4 y = v[j] * k;
            if (p.has_w != 0) y *= w4[i];
            if (p._p == 0) o[i] = y;
            v[j] = y;
            float4 a = abs(y);
            m = max(m, max(max(a.x, a.y), max(a.z, a.w)));
        }
    }
    m = pf_tg_max(m, sg, lane, red2);
    float inv, sc = pf_row_scale(m, inv);
    device half4* oh = (device half4*)(xh + (ulong)r * p.n);
    for (uint j = 0; j < PF_NORM_V; j++) {
        uint i = tid + j * 256;
        if (i < n4) oh[i] = half4(v[j] * sc);
    }
    if (tid == 0) rs[r] = inv;
}

#pragma clang fp reassociate(on)
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
    mv_finish<1, 1>(acc, acc2, y + (ulong)s * p.n_out * (p.mode == 5u ? 2u : 1u), p.mode, tg.x, p.n_out, sg, lane, red, sel + 2 * p.ldy); } \
MVX_SIG(NAME##_sum) { threadgroup float red[2]; \
    float acc[1] = { 0.f }, acc2[1] = { 0.f }; \
    for (uint s = 0; s < p.ldy; s++) { \
        EPX e = tab[as_type<uint>(sel[s])]; T f = MAKE(e); \
        mv_acc<T, X, CH, 1, 1>(f, f, false, 0, x + (ulong)s * p.n_in, p.n_in, p.n_out, tg.x, sg, lane, acc, acc2, sel[p.ldy + s]); \
    } \
    mv_finish<1, 1>(acc, acc2, y, p.mode, tg.x, p.n_out, sg, lane, red, sel); }
// DeepSeek V4.1's routed experts on their own dots (`mvxd_<fmt>`, see `MetalBackend::expert_pipe`),
// one SIMD-group a threadgroup. The pair kernel takes NRP rows of gate and up a SIMD-group (each x
// chunk loaded once for them). The sum kernel takes NRS rows; its lanes go over the (slot, chunk)
// tasks of all slots in turn instead of each slot's chunks (2304 values are 72 chunks of 32: a
// slot at a time left 8 lanes with 3 chunks and 24 with 2), so every lane has 13 or 14 tasks.
#define MVXD(NAME, T, X, CH, MAKE, NRP, NRS) \
MVX_SIG(NAME##_pair) { threadgroup float red[2]; \
    uint s = tg.y, row = tg.x * NRP; \
    EPX e = tab[as_type<uint>(sel[s])], e2 = tab2[as_type<uint>(sel[s])]; \
    float acc[NRP], acc2[NRP]; \
    for (int r = 0; r < NRP; r++) { acc[r] = 0.f; acc2[r] = 0.f; } \
    T f = MAKE(e); T f2 = MAKE(e2); \
    mv_acc<T, X, CH, NRP, 1>(f, f2, true, p._p, x, p.n_in, p.n_out, row, 0, lane, acc, acc2, 1.f); \
    mv_finish<NRP, 1>(acc, acc2, y + (ulong)s * p.n_out * (p.mode == 5u ? 2u : 1u), p.mode, row, p.n_out, 0, lane, red, sel + 2 * p.ldy); } \
MVX_SIG(NAME##_sum) { threadgroup float red[2]; \
    uint row = tg.x * NRS, rows[NRS]; \
    for (int r = 0; r < NRS; r++) rows[r] = min(row + r, p.n_out - 1); \
    uint nch = p.n_in / CH, s = 0, c = lane; \
    while (c >= nch) { c -= nch; s++; } \
    float a[NRS], b[NRS]; \
    for (int r = 0; r < NRS; r++) { a[r] = 0.f; b[r] = 0.f; } \
    while (s < p.ldy) { \
        EPX e = tab[as_type<uint>(sel[s])]; T f = MAKE(e); \
        X xr; \
        xload(x + (ulong)s * p.n_in, c, xr); \
        float w = sel[p.ldy + s]; \
        for (int r = 0; r < NRS; r++) a[r] += w * mdot(f, rows[r], c, xr); \
        c += 32; \
        while (c >= nch) { c -= nch; s++; } \
    } \
    mv_finish<NRS, 1>(a, b, y, p.mode, row, p.n_out, 0, lane, red, sel); }
// Experts in the tiled layout (`<type>T` weights, `p.gpr` with `GG_TILED`, see `gblk`): a lane a
// row. Consecutive lanes take consecutive rows of a band (a threadgroup RPS of them, 32 / RPS
// lanes a row), so that each read of the SIMD-group is one run of memory (the rows' copies of a
// block, one after another), where a lane a chunk (`MVX`) would read RPS pieces a band's column
// apart. The row's chunks are cut into 32 / RPS x KS runs, one per lane of the row in each of the
// KS SIMD-groups, summed in order, then the runs' sums over the lanes (shuffles) and the
// SIMD-groups (threadgroup memory). Other sums than `MVX`'s (the same products), the same in every
// kernel of a plan (multi-token steps run these per token).
template <typename T, typename X, uint CH, uint RPS, uint KS>
inline float xt_run(T f, uint row, device const float* x, uint n_in, uint sg, uint lane) {
    uint nch = n_in / CH, np = KS * (32 / RPS), q = sg * (32 / RPS) + lane / RPS;
    uint c0 = nch * q / np, c1 = nch * (q + 1) / np;
    float a = 0.f;
    for (uint c = c0; c < c1; c++) {
        X xr;
        xload(x, c, xr);
        a += mdot(f, row, c, xr);
    }
    return a;
}
template <typename T, typename X, uint CH, uint RPS, uint KS>
inline void xt_run2(T f, T f2, uint row, uint r2, device const float* x, uint n_in, uint sg, uint lane, thread float& a, thread float& b) {
    uint nch = n_in / CH, np = KS * (32 / RPS), q = sg * (32 / RPS) + lane / RPS;
    uint c0 = nch * q / np, c1 = nch * (q + 1) / np;
    for (uint c = c0; c < c1; c++) {
        X xr;
        xload(x, c, xr);
        a += mdot(f, row, c, xr);
        b += mdot(f2, row + r2, c, xr);
    }
}
// The sums of a row's runs into lanes 0 .. RPS - 1 of SIMD-group 0; false for the other lanes.
template <uint RPS, uint KS>
inline bool xt_reduce(thread float& a, thread float& b, uint sg, uint lane, threadgroup float* red) {
    for (uint o = 16; o >= RPS; o /= 2) {
        a += simd_shuffle_down(a, o);
        b += simd_shuffle_down(b, o);
    }
    if (KS > 1) {
        if (lane < RPS) { red[sg * 2 * RPS + lane] = a; red[sg * 2 * RPS + RPS + lane] = b; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg != 0 || lane >= RPS) return false;
        a = red[lane]; b = red[RPS + lane];
        for (uint k = 1; k < KS; k++) { a += red[k * 2 * RPS + lane]; b += red[k * 2 * RPS + RPS + lane]; }
        return true;
    }
    return lane < RPS;
}
#define MVXT(NAME, T, X, CH, MAKE, RPS, KS) \
MVX_SIG(NAME##_pair) { threadgroup float red[2 * RPS * KS]; \
    uint s = tg.y, row = tg.x * RPS + lane % RPS; EPX e = tab[as_type<uint>(sel[s])], e2 = tab2[as_type<uint>(sel[s])]; \
    T f = MAKE(e); T f2 = MAKE(e2); \
    float a = 0.f, b = 0.f; \
    xt_run2<T, X, CH, RPS, KS>(f, f2, row, p._p, x, p.n_in, sg, lane, a, b); \
    if (xt_reduce<RPS, KS>(a, b, sg, lane, red)) \
        mv_store(a, b, y + (ulong)s * p.n_out * (p.mode == 5u ? 2u : 1u), p.mode, row, p.n_out, sel + 2 * p.ldy); } \
MVX_SIG(NAME##_sum) { threadgroup float red[2 * RPS * KS]; \
    uint row = tg.x * RPS + lane % RPS; \
    float a = 0.f, b = 0.f; \
    for (uint s = 0; s < p.ldy; s++) { \
        EPX e = tab[as_type<uint>(sel[s])]; T f = MAKE(e); \
        a += sel[p.ldy + s] * xt_run<T, X, CH, RPS, KS>(f, row, x + (ulong)s * p.n_in, p.n_in, sg, lane); \
    } \
    if (xt_reduce<RPS, KS>(a, b, sg, lane, red)) mv_store(a, b, y, p.mode, row, p.n_out, sel); }
// ---- MoE expert sums of several tokens in one dispatch (the rows of a batched decode step, a
// verification pass): threadgroup (token t, row block) computes token t's weighted sum over its
// slots with exactly the one-token sum kernel's arithmetic (the same inline bodies), so each token
// gets the bits of its one-token step; the tokens of a row block are neighbours in the grid, so the
// expert rows they share come from memory once, then from the caches. (Gate / up pairs of several
// tokens in one dispatch, and an expert's tokens in one threadgroup, measured slower than a
// dispatch per token: Gemma 4 26B-A4B, 8 rows, 31-38 ms a step against 29.) `XB`: the tokens and
// the x, selection, activation and y strides in floats.
struct XB { uint ntok; uint sx; uint ssel; uint sact; uint sy; uint _p0; uint _p1; uint _p2; };
#define MVXB_SIG(NAME) \
kernel void NAME(device const EPX* tab [[buffer(0)]], device const EPX* tab2 [[buffer(1)]], device const float* x [[buffer(2)]], \
                 device float* y [[buffer(3)]], constant MV& p [[buffer(4)]], device const float* sel [[buffer(5)]], \
                 constant XB& b [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], \
                 uint lane [[thread_index_in_simdgroup]])
#define MVXB(NAME, T, X, CH, MAKE) \
MVXB_SIG(NAME##_bsum) { threadgroup float red[2]; \
    uint t = tg.x, row = tg.y; \
    device const float* st = sel + (ulong)t * b.ssel; \
    device const float* xt = x + (ulong)t * b.sact; \
    float acc[1] = { 0.f }, acc2[1] = { 0.f }; \
    for (uint s = 0; s < p.ldy; s++) { \
        EPX e = tab[as_type<uint>(st[s])]; T f = MAKE(e); \
        mv_acc<T, X, CH, 1, 1>(f, f, false, 0, xt + (ulong)s * p.n_in, p.n_in, p.n_out, row, sg, lane, acc, acc2, st[p.ldy + s]); \
    } \
    mv_finish<1, 1>(acc, acc2, y + (ulong)t * b.sy, p.mode, row, p.n_out, sg, lane, red, st); }
#define MVXTB(NAME, T, X, CH, MAKE, RPS, KS) \
MVXB_SIG(NAME##_bsum) { threadgroup float red[2 * RPS * KS]; \
    uint t = tg.x, row = tg.y * RPS + lane % RPS; \
    device const float* st = sel + (ulong)t * b.ssel; \
    device const float* xt = x + (ulong)t * b.sact; \
    float a = 0.f, bb = 0.f; \
    for (uint s = 0; s < p.ldy; s++) { \
        EPX e = tab[as_type<uint>(st[s])]; T f = MAKE(e); \
        a += st[p.ldy + s] * xt_run<T, X, CH, RPS, KS>(f, row, xt + (ulong)s * p.n_in, p.n_in, sg, lane); \
    } \
    if (xt_reduce<RPS, KS>(a, bb, sg, lane, red)) mv_store(a, bb, y + (ulong)t * b.sy, p.mode, row, p.n_out, st); }
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
    if (p.w == 0 && k <= 32 && nsg * k <= 64) {
        // Each SIMD-group's own top k (no threadgroup barrier), then SIMD-group 0 picks the k
        // best of those candidates: the same picks in the same order (of finite logits: the
        // global top k are among their SIMD-groups' top k), one barrier instead of 2 k.
        threadgroup float cv[64];
        threadgroup uint ci[64];
        float lv = -INFINITY;
        uint li = 0xffffffffu;
        for (uint j = 0; j < k; j++) {
            float m = simd_max(v);
            uint idx = simd_min(v == m ? tid : 0xffffffffu);
            if (tid == idx) v = -INFINITY;
            if (lane == j) { lv = m; li = idx; }
        }
        if (lane < k) { cv[sg * k + lane] = lv; ci[sg * k + lane] = li; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint nc = nsg * k;
        float a0 = lane < nc ? cv[lane] : -INFINITY, a1 = lane + 32 < nc ? cv[lane + 32] : -INFINITY;
        uint i0 = lane < nc ? ci[lane] : 0xffffffffu, i1 = lane + 32 < nc ? ci[lane + 32] : 0xffffffffu;
        for (uint j = 0; j < k && sg == 0; j++) {
            float m = simd_max(max(a0, a1));
            uint idx = simd_min(min(a0 == m ? i0 : 0xffffffffu, a1 == m ? i1 : 0xffffffffu));
            if (i0 == idx) a0 = -INFINITY;
            if (i1 == idx) a1 = -INFINITY;
            if (tid == 0) { sel[j] = as_type<float>(idx); chosen[j] = m; }
        }
    } else for (uint j = 0; j < k; j++) {
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

#pragma clang fp reassociate(off)
// ---- Elementwise: p.x = 0 scale (a * p.y), 1 softcap (tanh(a / p.y) * p.y), 2 combine
// ((a * p.y + b * p.z) * p.w), 3 silu(a * p.y), 4 hyper-connection read (out = p.w * sum over
// p.y streams of sigmoid(a) * b, streams n apart), 5 hyper-connection write (out[s m + i] +=
// a[i] * 2 sigmoid(b[s] * p.z), m = p.y).
inline void ew_body(device const float* a, device const float* b, device float* out, constant float4& p, uint n, uint i) {
    if (p.x == 4.f) {
        uint hc = (uint)p.y;
        float s = 0.f;
        for (uint j = 0; j < hc; j++) {
            float g = a[j * n + i];
            s += b[j * n + i] / (1.0f + precise::exp(-g));
        }
        out[i] = s * p.w;
        return;
    }
    if (p.x == 5.f) {
        uint m = (uint)p.y;
        uint s = i / m;
        float w = 2.0f / (1.0f + precise::exp(-b[s] * p.z));
        out[i] += a[i - s * m] * w;
        return;
    }
    float v = a[i];
    if (p.x == 0.f) out[i] = v * p.y;
    else if (p.x == 1.f) out[i] = precise::tanh(v / p.y) * p.y;
    else if (p.x == 3.f) { float u = v * p.y; out[i] = u / (1.0f + precise::exp(-u)); }
    else out[i] = (v * p.y + b[i] * p.z) * p.w;
}
kernel void dec_ew(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* out [[buffer(2)]],
                   constant float4& p [[buffer(3)]], constant uint& n [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    ew_body(a, b, out, p, n, i);
}

// Several tokens (grid y): ts = strides of a, b, out.
kernel void dec_ew_t(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* out [[buffer(2)]],
                     constant float4& p [[buffer(3)]], constant uint& n [[buffer(4)]], constant uint4& ts [[buffer(5)]],
                     uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x, t = gid.y;
    if (i >= n) return;
    ew_body(a + (ulong)t * ts.x, b + (ulong)t * ts.y, out + (ulong)t * ts.z, p, n, i);
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
inline void add_norm_body(device const float* y, device float* x, device float* out, device const float* wy, device const float* wn,
                          constant AN& p, uint tid, uint nt, uint sg, uint lane, threadgroup float* red) {
    float yv[8], xv[8], wyv[8], wnv[8];
    float s = 0.f;
    for (uint j = 0; j < 8; j++) {
        uint i = tid + j * nt;
        yv[j] = i < p.n ? y[i] : 0.f;
        xv[j] = i < p.n ? x[i] : 0.f;
        s += yv[j] * yv[j];
    }
    // The weights loaded while the sums are taken.
    for (uint j = 0; j < 8; j++) {
        uint i = tid + j * nt;
        wyv[j] = i < p.n ? wy[i] : 0.f;
        wnv[j] = i < p.n && p.has_next != 0 ? wn[i] : 0.f;
    }
    float k = rsqrt(tg_total(s, nt, sg, lane, red) / float(p.n) + p.eps);
    float s2 = 0.f;
    for (uint j = 0; j < 8; j++) {
        uint i = tid + j * nt;
        if (i < p.n) {
            xv[j] = (xv[j] + yv[j] * k * wyv[j]) * p.scale;
            x[i] = xv[j];
            s2 += xv[j] * xv[j];
        }
    }
    if (p.has_next == 0) return;
    float k2 = rsqrt(tg_total(s2, nt, sg, lane, red) / float(p.n) + p.eps);
    for (uint j = 0; j < 8; j++) {
        uint i = tid + j * nt;
        if (i < p.n) out[i] = xv[j] * k2 * wnv[j];
    }
}
kernel void dec_add_norm(device const float* y [[buffer(0)]], device float* x [[buffer(1)]], device float* out [[buffer(2)]],
                         device const float* wy [[buffer(3)]], device const float* wn [[buffer(4)]], constant AN& p [[buffer(5)]],
                         uint tid [[thread_index_in_threadgroup]], uint nt [[threads_per_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[32];
    add_norm_body(y, x, out, wy, wn, p, tid, nt, sg, lane, red);
}
// A one-row RMSNorm added into y first (y += rmsnorm(e) * w0: `dec_rms_norm`'s body and
// parameters `np`, on the same threads), then the AddNorm: Gemma 4 26B-A4B's routed experts' post norm and its
// residual step in one dispatch. Each thread reads back the y values it wrote itself.
kernel void dec_norm_add_norm(device float* y [[buffer(0)]], device float* x [[buffer(1)]], device float* out [[buffer(2)]],
                              device const float* wy [[buffer(3)]], device const float* wn [[buffer(4)]], constant AN& p [[buffer(5)]],
                              device const float* e [[buffer(6)]], device const float* w0 [[buffer(7)]], constant NP& np [[buffer(8)]],
                              uint tid [[thread_index_in_threadgroup]], uint nt [[threads_per_threadgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[32];
    rms_norm_body(e, y, w0, np, 0, tid, nt, sg, lane, red);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    add_norm_body(y, x, out, wy, wn, p, tid, nt, sg, lane, red);
}
// Several tokens (threadgroup y): ts = strides of y, x, out.
kernel void dec_add_norm_t(device const float* y [[buffer(0)]], device float* x [[buffer(1)]], device float* out [[buffer(2)]],
                           device const float* wy [[buffer(3)]], device const float* wn [[buffer(4)]], constant AN& p [[buffer(5)]],
                           constant uint4& ts [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]], uint2 nt2 [[threads_per_threadgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[32];
    ulong t = tg.y;
    add_norm_body(y + t * ts.x, x + t * ts.y, out + t * ts.z, wy, wn, p, tid, nt2.x, sg, lane, red);
}

// AddNorm with a next norm, of token rows (add_norm_body's, summed in another order), one
// threadgroup of 256 per token, n % 4 == 0 and n <= 256 * 4 * PF_NORM_V: x = (x + norm(y) wy)
// scale, out = norm(x) wn (unless ts.w: nothing reads it) and out's scaled half rows for the next
// GEMM (x_half's).
kernel void pf_add_norm_half(device const float* y [[buffer(0)]], device float* x [[buffer(1)]], device float* out [[buffer(2)]],
                             device const float* wy [[buffer(3)]], device const float* wn [[buffer(4)]], constant AN& p [[buffer(5)]],
                             constant uint4& ts [[buffer(6)]], device half* xh [[buffer(7)]], device float* rs [[buffer(8)]],
                             constant uint4& ex [[buffer(9)]], device float* o2 [[buffer(10)]], device const float* w2 [[buffer(11)]],
                             device float* o3 [[buffer(12)]], device const float* w3 [[buffer(13)]],
                             uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8];
    threadgroup float red2[8];
    threadgroup float red3[8];
    device const float4* yr = (device const float4*)(y + (ulong)r * ts.x);
    device float4* xr = (device float4*)(x + (ulong)r * ts.y);
    device float4* o = (device float4*)(out + (ulong)r * ts.z);
    device const float4* wy4 = (device const float4*)wy;
    device const float4* wn4 = (device const float4*)wn;
    uint n4 = p.n / 4;
    float4 yv[PF_NORM_V], xv[PF_NORM_V];
    float s = 0.f;
    for (uint j = 0; j < PF_NORM_V; j++) {
        uint i = tid + j * 256;
        yv[j] = i < n4 ? yr[i] : 0.f;
        xv[j] = i < n4 ? xr[i] : 0.f;
        s += dot(yv[j], yv[j]);
    }
    s = simd_sum(s);
    if (lane == 0) red[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    s = ((red[0] + red[1]) + (red[2] + red[3])) + ((red[4] + red[5]) + (red[6] + red[7]));
    float k = rsqrt(s / float(p.n) + p.eps);
    float s2 = 0.f;
    for (uint j = 0; j < PF_NORM_V; j++) {
        uint i = tid + j * 256;
        if (i < n4) {
            xv[j] = (xv[j] + yv[j] * k * wy4[i]) * p.scale;
            xr[i] = xv[j];
            s2 += dot(xv[j], xv[j]);
        }
    }
    s2 = simd_sum(s2);
    if (lane == 0) red2[sg] = s2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    s2 = ((red2[0] + red2[1]) + (red2[2] + red2[3])) + ((red2[4] + red2[5]) + (red2[6] + red2[7]));
    float k2 = rsqrt(s2 / float(p.n) + p.eps);
    float m = 0.f;
    // ex.x more norms of x (the new residual): rows ex.y / ex.z floats apart, with weights where
    // bit j of ex.w is set.
    device float4* e2 = (device float4*)(o2 + (ulong)r * ex.y);
    device float4* e3 = (device float4*)(o3 + (ulong)r * ex.z);
    for (uint j = 0; j < PF_NORM_V; j++) {
        uint i = tid + j * 256;
        if (i < n4) {
            float4 v = xv[j] * k2 * wn4[i];
            if (ts.w == 0) o[i] = v;
            yv[j] = v;
            float4 a = abs(v);
            m = max(m, max(max(a.x, a.y), max(a.z, a.w)));
            if (ex.x > 0) e2[i] = (ex.w & 1) != 0 ? xv[j] * k2 * ((device const float4*)w2)[i] : xv[j] * k2;
            if (ex.x > 1) e3[i] = (ex.w & 2) != 0 ? xv[j] * k2 * ((device const float4*)w3)[i] : xv[j] * k2;
        }
    }
    m = pf_tg_max(m, sg, lane, red3);
    float inv, sc = pf_row_scale(m, inv);
    device half4* oh = (device half4*)(xh + (ulong)r * p.n);
    for (uint j = 0; j < PF_NORM_V; j++) {
        uint i = tid + j * 256;
        if (i < n4) oh[i] = half4(yv[j] * sc);
    }
    if (tid == 0) rs[r] = inv;
}

// Gemma 4's MoE layer tail of token rows, one threadgroup of 256 per token, n % 4 == 0 and
// n <= 256 * 4 * PF_TAIL_V: the routed experts' rows combined as moe_combine does (e = the sum
// over the token's k slots of weight * eo[row]), y = norm(d) wd + norm(e) we for the dense output
// d (y's rows; norms as dec_rms_norm's, summed in another order), then pf_add_norm_half's step on
// y: x = (x + norm(y) wy) scale, out = norm(x) wn (unless ts.w) and its scaled half rows.
#define PF_TAIL_V 4
inline float pf_tg_sum(float s, uint sg, uint lane, threadgroup float* red) {
    s = simd_sum(s);
    if (lane == 0) red[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return ((red[0] + red[1]) + (red[2] + red[3])) + ((red[4] + red[5]) + (red[6] + red[7]));
}
kernel void pf_moe_tail(device const float* y [[buffer(0)]], device float* x [[buffer(1)]], device float* out [[buffer(2)]],
                        device const float* wy [[buffer(3)]], device const float* wn [[buffer(4)]], constant AN& p [[buffer(5)]],
                        constant uint4& ts [[buffer(6)]], device half* xh [[buffer(7)]], device float* rs [[buffer(8)]],
                        device const float* eo [[buffer(9)]], device const float2* slots [[buffer(10)]],
                        device const float* wd [[buffer(11)]], device const float* we [[buffer(12)]], constant uint& k [[buffer(13)]],
                        uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[5][8];
    device const float4* yr = (device const float4*)(y + (ulong)r * ts.x);
    device float4* xr = (device float4*)(x + (ulong)r * ts.y);
    device float4* o = (device float4*)(out + (ulong)r * ts.z);
    device const float4* wd4 = (device const float4*)wd;
    device const float4* we4 = (device const float4*)we;
    device const float4* wy4 = (device const float4*)wy;
    device const float4* wn4 = (device const float4*)wn;
    uint n4 = p.n / 4;
    float4 dv[PF_TAIL_V], ev[PF_TAIL_V];
    for (uint j = 0; j < PF_TAIL_V; j++) { dv[j] = 0.f; ev[j] = 0.f; }
    for (uint s = 0; s < k; s++) {
        float2 sl = slots[(ulong)r * k + s];
        device const float4* er = (device const float4*)(eo + (ulong)as_type<uint>(sl.x) * p.n);
        for (uint j = 0; j < PF_TAIL_V; j++) {
            uint i = tid + j * 256;
            if (i < n4) ev[j] += sl.y * er[i];
        }
    }
    float sd = 0.f, se = 0.f;
    for (uint j = 0; j < PF_TAIL_V; j++) {
        uint i = tid + j * 256;
        if (i < n4) dv[j] = yr[i];
        sd += dot(dv[j], dv[j]);
        se += dot(ev[j], ev[j]);
    }
    float kd = rsqrt(pf_tg_sum(sd, sg, lane, red[0]) / float(p.n) + p.eps);
    float ke = rsqrt(pf_tg_sum(se, sg, lane, red[1]) / float(p.n) + p.eps);
    float s = 0.f;
    for (uint j = 0; j < PF_TAIL_V; j++) {
        uint i = tid + j * 256;
        if (i < n4) dv[j] = dv[j] * kd * wd4[i] + ev[j] * ke * we4[i];
        s += dot(dv[j], dv[j]);
    }
    float k1 = rsqrt(pf_tg_sum(s, sg, lane, red[2]) / float(p.n) + p.eps);
    float s2 = 0.f;
    for (uint j = 0; j < PF_TAIL_V; j++) {
        uint i = tid + j * 256;
        if (i < n4) {
            ev[j] = (xr[i] + dv[j] * k1 * wy4[i]) * p.scale;
            xr[i] = ev[j];
            s2 += dot(ev[j], ev[j]);
        }
    }
    float k2 = rsqrt(pf_tg_sum(s2, sg, lane, red[3]) / float(p.n) + p.eps);
    float m = 0.f;
    for (uint j = 0; j < PF_TAIL_V; j++) {
        uint i = tid + j * 256;
        if (i < n4) {
            float4 v = ev[j] * k2 * wn4[i];
            if (ts.w == 0) o[i] = v;
            dv[j] = v;
            float4 a = abs(v);
            m = max(m, max(max(a.x, a.y), max(a.z, a.w)));
        }
    }
    m = pf_tg_max(m, sg, lane, red[4]);
    float inv, sc = pf_row_scale(m, inv);
    device half4* oh = (device half4*)(xh + (ulong)r * p.n);
    for (uint j = 0; j < PF_TAIL_V; j++) {
        uint i = tid + j * 256;
        if (i < n4) oh[i] = half4(dv[j] * sc);
    }
    if (tid == 0) rs[r] = inv;
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

// ---- Qwen4-Exp n-gram gate: one threadgroup per stream s: a = k_s . q_s / sqrt(n), then
// out[s n + i] = sigmoid(sign(a) sqrt(max(|a|, 1e-6))) * v[i].
kernel void dec_ple_gate(device const float* k [[buffer(0)]], device const float* q [[buffer(1)]], device const float* v [[buffer(2)]],
                         device float* out [[buffer(3)]], constant uint2& p [[buffer(4)]], uint s [[threadgroup_position_in_grid]],
                         uint tid [[thread_position_in_threadgroup]], uint nt [[threads_per_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[32];
    uint n = p.x;
    device const float* ks = k + (ulong)s * n;
    device const float* qs = q + (ulong)s * n;
    float acc = 0.f;
    for (uint i = tid; i < n; i += nt) acc += ks[i] * qs[i];
    float a = tg_total(acc, nt, sg, lane, red) / sqrt((float)n);
    float m = sqrt(max(fabs(a), 1e-6f));
    float g = a > 0.f ? m : (a < 0.f ? -m : 0.f);
    float sig = 1.0f / (1.0f + precise::exp(-g));
    device float* o = out + (ulong)s * n;
    for (uint i = tid; i < n; i += nt) o[i] = sig * v[i];
}
// Several tokens (grid y): ts = strides of k, q, v, out.
kernel void dec_ple_gate_t(device const float* k [[buffer(0)]], device const float* q [[buffer(1)]], device const float* v [[buffer(2)]],
                           device float* out [[buffer(3)]], constant uint2& p [[buffer(4)]], constant uint4& ts [[buffer(5)]],
                           uint2 tg [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]],
                           uint2 nt2 [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[32];
    uint tid = tid2.x, nt = nt2.x;
    uint n = p.x, s = tg.x, t = tg.y;
    device const float* ks = k + (ulong)t * ts.x + (ulong)s * n;
    device const float* qs = q + (ulong)t * ts.y + (ulong)s * n;
    device const float* vs = v + (ulong)t * ts.z;
    float acc = 0.f;
    for (uint i = tid; i < n; i += nt) acc += ks[i] * qs[i];
    float a = tg_total(acc, nt, sg, lane, red) / sqrt((float)n);
    float m = sqrt(max(fabs(a), 1e-6f));
    float g = a > 0.f ? m : (a < 0.f ? -m : 0.f);
    float sig = 1.0f / (1.0f + precise::exp(-g));
    device float* o = out + (ulong)t * ts.w + (ulong)s * n;
    for (uint i = tid; i < n; i += nt) o[i] = sig * vs[i];
}

// ---- Runtime LoRA updates (crate::lora, `Op::Lora`) -----------------------------------------------
// p: n_in, r, n_out, then the token strides of x, u, y and the gate, and whether there is a gate.
struct LoraP { uint n_in, r, n_out, xs, us, ys, gs, has_gate; };
// Partial sums of u = A x over chunks of 1024 inputs: threadgroup (c * r + k, t) writes
// u[t, c * r + k] = A[k, chunk c] . x[t, chunk c] (p.n_out: the chunks).
kernel void lora_down(device const float* a [[buffer(0)]], device const float* x [[buffer(1)]], device float* u [[buffer(2)]],
                      constant LoraP& p [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[32];
    uint k = tg.x % p.r, c = tg.x / p.r, t = tg.y;
    uint i0 = c * 1024, i1 = min(i0 + 1024, p.n_in);
    device const float* ar = a + (ulong)k * p.n_in;
    device const float* xr = x + (ulong)t * p.xs;
    float s = 0.f;
    if (p.has_gate != 0u) {
        device const float4* a4 = (device const float4*)(ar + i0);
        device const float4* x4 = (device const float4*)(xr + i0);
        uint i = tid;
        if (i < (i1 - i0) / 4) s = dot(a4[i], x4[i]);
    } else {
        for (uint i = i0 + tid; i < i1; i += 256) s += ar[i] * xr[i];
    }
    s = tg_total(s, 256, sg, lane, red);
    if (tid == 0) u[(ulong)t * p.us + tg.x] = s;
}
// y[t, o] += g_t sum_k u_k S[sc[k]] bt[k, o], u_k the sum of component k's chunks in order (p.n_in:
// the chunks), g_t the sigmoid of the gate or 1: threadgroup (256 outputs, t).
kernel void lora_up(device const float* bt [[buffer(0)]], device const uint* sc [[buffer(1)]], device const float* str [[buffer(2)]],
                    device const float* u [[buffer(3)]], device float* y [[buffer(4)]], device const float* gate [[buffer(5)]],
                    constant LoraP& p [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float us[256];
    uint t = tg.y;
    device const float* ut = u + (ulong)t * p.us;
    for (uint k = tid; k < p.r; k += 256) {
        float v = 0.f;
        for (uint c = 0; c < p.n_in; c++) v += ut[c * p.r + k];
        us[k] = v * str[sc[k]];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float g = p.has_gate != 0u ? 1.0f / (1.0f + exp(-gate[(ulong)t * p.gs])) : 1.0f;
    uint o = tg.x * 256 + tid;
    if (o >= p.n_out) return;
    // Four independent sums keep several loads of B in flight.
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    device const float* b = bt + o;
    ulong n = p.n_out;
    uint k = 0;
    for (; k + 4 <= p.r; k += 4) {
        a0 += us[k] * b[k * n];
        a1 += us[k + 1] * b[(k + 1) * n];
        a2 += us[k + 2] * b[(k + 2) * n];
        a3 += us[k + 3] * b[(k + 3) * n];
    }
    for (; k < p.r; k++) a0 += us[k] * b[k * n];
    y[(ulong)t * p.ys + o] += g * ((a0 + a1) + (a2 + a3));
}
// Routed experts (`Experts::lora`), one token: the stacks are [n_exp, r, ...], `sel` holds the k
// expert ids (as uint bits) then their weights. p: n_in, r, n_out, the x stride between slots (0:
// one x for all), k, gelu.
struct XLoraP { uint n_in, r, n_out, xs, k, gelu, _p0, _p1; };
// u[s, c] = A_e[c] . x_s for slot s's expert e: threadgroup (c, s).
kernel void xlora_down(device const float* a [[buffer(0)]], device const float* x [[buffer(1)]], device float* u [[buffer(2)]],
                       device const float* sel [[buffer(3)]], constant XLoraP& p [[buffer(4)]],
                       uint2 tg [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]],
                       uint2 nt2 [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[32];
    uint tid = tid2.x, nt = nt2.x, c = tg.x, s = tg.y;
    uint e = as_type<uint>(sel[s]);
    device const float* ar = a + ((ulong)e * p.r + c) * p.n_in;
    device const float* xr = x + (ulong)s * p.xs;
    float acc = 0.f;
    if ((p.n_in & 3u) == 0u && (p.xs & 3u) == 0u) {
        device const float4* a4 = (device const float4*)ar;
        device const float4* x4 = (device const float4*)xr;
        for (uint i = tid; i < p.n_in / 4; i += nt) acc += dot(a4[i], x4[i]);
    } else {
        for (uint i = tid; i < p.n_in; i += nt) acc += ar[i] * xr[i];
    }
    acc = tg_total(acc, nt, sg, lane, red);
    if (tid == 0) u[s * p.r + c] = acc;
}
// The activation of slot s from its raw gate / up outputs (`raw`: [k, 2 n_out]) plus their
// updates: act[s, o] = act(g + dg) * (v + dv). Thread (o, s).
kernel void xlora_act(device const float* raw [[buffer(0)]], device const float* u [[buffer(1)]], device const float* bt [[buffer(2)]],
                      device const uint* sc [[buffer(3)]], device const float* str [[buffer(4)]], device const float* sel [[buffer(5)]],
                      device float* act [[buffer(6)]], constant XLoraP& p [[buffer(7)]], uint2 gid [[thread_position_in_grid]]) {
    uint o = gid.x, s = gid.y, n = p.n_out, w = 2 * n;
    if (o >= n) return;
    uint e = as_type<uint>(sel[s]);
    device const float* b = bt + (ulong)e * p.r * w;
    device const uint* sce = sc + (ulong)e * p.r;
    device const float* us = u + s * p.r;
    float g = raw[(ulong)s * w + o], v = raw[(ulong)s * w + n + o];
    for (uint c = 0; c < p.r; c++) {
        float f = us[c] * str[sce[c]];
        g += f * b[(ulong)c * w + o];
        v += f * b[(ulong)c * w + n + o];
    }
    float a;
    if (p.gelu != 0u) {
        float t = 0.7978845608f * (g + 0.044715f * g * g * g);
        a = 0.5f * g * (1.0f + precise::tanh(t));
    } else {
        a = g / (1.0f + exp(-g));
    }
    act[(ulong)s * n + o] = a * v;
}
// y[o] += sum_s weight_s sum_c u[s, c] S[sc] bt_e[c, o] (the down projections' updates). Thread o.
kernel void xlora_up(device const float* u [[buffer(0)]], device const float* bt [[buffer(1)]], device const uint* sc [[buffer(2)]],
                     device const float* str [[buffer(3)]], device const float* sel [[buffer(4)]], device float* y [[buffer(5)]],
                     constant XLoraP& p [[buffer(6)]], uint o [[thread_position_in_grid]]) {
    if (o >= p.n_out) return;
    float acc = 0.f;
    for (uint s = 0; s < p.k; s++) {
        uint e = as_type<uint>(sel[s]);
        device const float* b = bt + (ulong)e * p.r * p.n_out;
        device const uint* sce = sc + (ulong)e * p.r;
        float a = 0.f;
        for (uint c = 0; c < p.r; c++) a += u[s * p.r + c] * str[sce[c]] * b[(ulong)c * p.n_out + o];
        acc += sel[p.k + s] * a;
    }
    y[o] += acc;
}
// Prefill plans' routed experts with LoRA updates: rows gathered by expert (`MOp::PfExperts`),
// `src[j]` the token of row j (all ones for padding), `rexp[j]` its expert. p: n_in, r, n (the
// output width per gate or up half, or of the down projections), the x stride, `split` (gate and
// up rows stored apart: up row j at split + j) and gelu.
struct PxLoraP { uint n_in, r, n, xs, split, gelu, _p0, _p1; };
inline float pfx_act(float g, float u, uint gelu) {
    return gelu != 0u ? g / (1.0f + exp(-1.5957691f * (g + 0.044715f * g * g * g))) * u : g / (1.0f + exp(-g)) * u;
}
// u[j, c] = A_e[c] . x[src[j]] (`gu` null) or A_e[c] . act(gate_j, up_j) from the gate / up rows
// `gu` (the down projections' input): threadgroup (c, j).
kernel void pfx_lora_down(device const float* a [[buffer(0)]], device const float* x [[buffer(1)]], device const uint* src [[buffer(2)]],
                          device const uint* rexp [[buffer(3)]], device float* u [[buffer(4)]], constant PxLoraP& p [[buffer(5)]],
                          uint2 tg [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]],
                          uint2 nt2 [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[32];
    uint tid = tid2.x, nt = nt2.x, c = tg.x, j = tg.y;
    uint e = rexp[j];
    if (src[j] == 0xffffffffu || e == 0xffffffffu) {
        if (tid == 0) u[(ulong)j * p.r + c] = 0.f;
        return;
    }
    device const float* ar = a + ((ulong)e * p.r + c) * p.n_in;
    float acc = 0.f;
    if (p.xs != 0u) {
        device const float* xr = x + (ulong)src[j] * p.xs;
        for (uint i = tid; i < p.n_in; i += nt) acc += ar[i] * xr[i];
    } else {
        device const float* g = x + (ulong)j * (p.split != 0u ? 1u : 2u) * p.n_in;
        device const float* v = p.split != 0u ? x + (ulong)(p.split + j) * p.n_in : g + p.n_in;
        for (uint i = tid; i < p.n_in; i += nt) acc += ar[i] * pfx_act(g[i], v[i], p.gelu);
    }
    acc = tg_total(acc, nt, sg, lane, red);
    if (tid == 0) u[(ulong)j * p.r + c] = acc;
}
// Gate / up rows (`two` != 0: 2 n outputs per row, stored as the GEMMs wrote them) or down rows
// (n outputs) of row j: out += sum_c u[j, c] S[sc] bt_e[c, o]. Thread (o, j).
kernel void pfx_lora_add(device const float* u [[buffer(0)]], device const float* bt [[buffer(1)]], device const uint* sc [[buffer(2)]],
                         device const float* str [[buffer(3)]], device const uint* src [[buffer(4)]], device const uint* rexp [[buffer(5)]],
                         device float* out [[buffer(6)]], constant PxLoraP& p [[buffer(7)]], constant uint& two [[buffer(8)]],
                         uint2 gid [[thread_position_in_grid]]) {
    uint o = gid.x, j = gid.y, w = two != 0u ? 2 * p.n : p.n;
    if (o >= w) return;
    uint e = rexp[j];
    if (src[j] == 0xffffffffu || e == 0xffffffffu) return;
    device const float* b = bt + (ulong)e * p.r * w;
    device const uint* sce = sc + (ulong)e * p.r;
    float acc = 0.f;
    for (uint c = 0; c < p.r; c++) acc += u[(ulong)j * p.r + c] * str[sce[c]] * b[(ulong)c * w + o];
    ulong at;
    if (two == 0u) at = (ulong)j * p.n + o;
    else if (p.split != 0u) at = o < p.n ? (ulong)j * p.n + o : (ulong)(p.split + j) * p.n + (o - p.n);
    else at = (ulong)j * 2 * p.n + o;
    out[at] += acc;
}

// ---- A DSpark draft's Markov chain on the device (greedy, `Backend::markov_greedy`): slot i
// takes the bf16 row of `markov_w1` for the token at tok[i] (`mk_gather`), adds `markov_w2`
// times it to its logits (a few-rows matvec in add mode), and writes its argmax to tok[i + 1]
// (`mk_argmax`: the first index of the largest value; NaNs never win, 0 when nothing does).
struct MkP { uint rank; uint vocab; uint slot; uint _p; };
kernel void mk_gather(device const ushort* w1 [[buffer(0)]], device const uint* tok [[buffer(1)]], device float* x [[buffer(2)]],
                      constant MkP& p [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
    if (tid >= p.rank) return;
    x[tid] = as_type<float>(uint(w1[(ulong)tok[p.slot] * p.rank + tid]) << 16);
}
inline void mk_pick(thread float& bv, thread uint& bi, float v, uint i) {
    if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
}
kernel void mk_argmax(device const float* y [[buffer(0)]], device uint* tok [[buffer(1)]], constant MkP& p [[buffer(2)]],
                      uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float tv[32];
    threadgroup uint ti[32];
    device const float* row = y + (ulong)p.slot * p.vocab;
    float bv = -INFINITY;
    uint bi = 0xffffffffu;
    for (uint k = tid; k < p.vocab; k += 1024) {
        float v = row[k];
        if (v > bv) { bv = v; bi = k; }
    }
    for (ushort m = 16; m > 0; m >>= 1) mk_pick(bv, bi, simd_shuffle_xor(bv, m), simd_shuffle_xor(bi, m));
    if (lane == 0) { tv[sg] = bv; ti[sg] = bi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        for (uint k = 1; k < 32; k++) mk_pick(bv, bi, tv[k], ti[k]);
        tok[p.slot + 1] = bi == 0xffffffffu ? 0u : bi;
    }
}

// ---- Greedy chains on the device (`MetalPlan::greedy`) ------------------------------------------
// The argmax of a pass's logits (`n` floats at `x` of the arena): each threadgroup of 256 takes
// 1024 values (4 consecutive a thread) and writes its (value, index bits) to `part + 2 tg`;
// `dec_amax_fin` (one threadgroup of 1024) then writes the token (uint bits) to `tok`. The
// host's rule (`crate::sampler::argmax`): the first index of the largest value above -inf, NaNs
// never win, 0 when nothing does.
struct AmP { uint n; uint x; uint part; uint tok; };
inline void am_pick(thread float& bv, thread uint& bi, float v, uint i) {
    if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
}
inline void am_simd(thread float& bv, thread uint& bi) {
    for (ushort m = 16; m > 0; m >>= 1) am_pick(bv, bi, simd_shuffle_xor(bv, m), simd_shuffle_xor(bi, m));
}
kernel void dec_amax_part(device float* a [[buffer(0)]], constant AmP& p [[buffer(1)]], uint tg [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float tv[8];
    threadgroup uint ti[8];
    float bv = -INFINITY;
    uint bi = 0xffffffffu;
    uint i0 = tg * 1024 + tid * 4;
    for (uint j = 0; j < 4; j++) {
        uint i = i0 + j;
        if (i < p.n) {
            float v = a[p.x + i];
            if (v > bv) { bv = v; bi = i; }
        }
    }
    am_simd(bv, bi);
    if (lane == 0) { tv[sg] = bv; ti[sg] = bi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        for (uint k = 1; k < 8; k++) am_pick(bv, bi, tv[k], ti[k]);
        a[p.part + 2 * tg] = bv;
        a[p.part + 2 * tg + 1] = as_type<float>(bi);
    }
}
kernel void dec_amax_fin(device float* a [[buffer(0)]], constant AmP& p [[buffer(1)]], uint tid [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float tv[32];
    threadgroup uint ti[32];
    float bv = -INFINITY;
    uint bi = 0xffffffffu;
    uint np = (p.n + 1023) / 1024;
    for (uint k = tid; k < np; k += 1024) am_pick(bv, bi, a[p.part + 2 * k], as_type<uint>(a[p.part + 2 * k + 1]));
    am_simd(bv, bi);
    if (lane == 0) { tv[sg] = bv; ti[sg] = bi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        for (uint k = 1; k < 32; k++) am_pick(bv, bi, tv[k], ti[k]);
        ((device uint*)a)[p.tok] = bi == 0xffffffffu ? 0u : bi;
    }
}

// The input row of a pass whose token the device chose: row `tok` (a uint of the arena) of a
// token table, dequantized as the host's `Table::row` (the same operations in the same order, no
// contraction), times `scale`, into `out` (`n` floats). A thread per 32 values (a block, or a
// 32-value run of a 256-value K-quant super-block). Kinds (`EK_*` in metal_decode.rs): GGUF rows
// of `rb` bytes at `q` (F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q2_K, Q4_K, Q5_K, Q6_K);
// oxinfer's int8 rows (`q`, f32 scales per 32 at `s`); its Q4_0-layout rows (`q`, f16 scales at
// `s`, bf16 affine offsets per `group` at `o`, or none when `o` is 0); MLX affine rows (`bits`,
// bf16 scales and biases per `group` at `s` and `o`).
struct EmP { ulong q; ulong s; ulong o; uint n; uint tok; uint out; uint rb; float scale; uint group; uint bits; uint kind; };
inline float emb_h(device const uchar* b) { return float(as_type<half>(ushort(uint(b[0]) | (uint(b[1]) << 8)))); }
inline float emb_bf(ushort v) { return as_type<float>(uint(v) << 16); }
kernel void dec_embed(device float* a [[buffer(0)]], constant EmP& p [[buffer(1)]], uint c [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    if (c >= p.n / 32) return;
    uint row = ((device const uint*)a)[p.tok];
    device float* y = a + p.out + c * 32;
    float v[32];
    device const uchar* R = (device const uchar*)p.q + (ulong)row * p.rb;
    switch (p.kind) {
    case 0: { // F32
        device const uchar* b = R + c * 128;
        for (uint l = 0; l < 32; l++)
            v[l] = as_type<float>(uint(b[4 * l]) | (uint(b[4 * l + 1]) << 8) | (uint(b[4 * l + 2]) << 16) | (uint(b[4 * l + 3]) << 24));
        break;
    }
    case 1: { // F16
        device const uchar* b = R + c * 64;
        for (uint l = 0; l < 32; l++) v[l] = emb_h(b + 2 * l);
        break;
    }
    case 2: { // BF16
        device const uchar* b = R + c * 64;
        for (uint l = 0; l < 32; l++) v[l] = emb_bf(ushort(uint(b[2 * l]) | (uint(b[2 * l + 1]) << 8)));
        break;
    }
    case 3: { // Q8_0
        device const uchar* b = R + c * 34;
        float d = emb_h(b);
        for (uint l = 0; l < 32; l++) v[l] = float(as_type<char>(b[2 + l])) * d;
        break;
    }
    case 4: { // Q4_0
        device const uchar* b = R + c * 18;
        float d = emb_h(b);
        for (uint j = 0; j < 16; j++) {
            v[j] = float(int(b[2 + j] & 15) - 8) * d;
            v[j + 16] = float(int(b[2 + j] >> 4) - 8) * d;
        }
        break;
    }
    case 5: { // Q4_1
        device const uchar* b = R + c * 20;
        float d = emb_h(b), m = emb_h(b + 2);
        for (uint j = 0; j < 16; j++) {
            float t0 = float(b[4 + j] & 15) * d;
            float t1 = float(b[4 + j] >> 4) * d;
            v[j] = t0 + m;
            v[j + 16] = t1 + m;
        }
        break;
    }
    case 6: { // Q5_0
        device const uchar* b = R + c * 22;
        float d = emb_h(b);
        uint qh = uint(b[2]) | (uint(b[3]) << 8) | (uint(b[4]) << 16) | (uint(b[5]) << 24);
        for (uint j = 0; j < 16; j++) {
            uint h0 = ((qh >> j) << 4) & 0x10, h1 = (qh >> (j + 12)) & 0x10;
            v[j] = float(int((b[6 + j] & 15) | h0) - 16) * d;
            v[j + 16] = float(int((b[6 + j] >> 4) | h1) - 16) * d;
        }
        break;
    }
    case 7: { // Q5_1
        device const uchar* b = R + c * 24;
        float d = emb_h(b), m = emb_h(b + 2);
        uint qh = uint(b[4]) | (uint(b[5]) << 8) | (uint(b[6]) << 16) | (uint(b[7]) << 24);
        for (uint j = 0; j < 16; j++) {
            uint h0 = ((qh >> j) << 4) & 0x10, h1 = (qh >> (j + 12)) & 0x10;
            float t0 = float((b[8 + j] & 15) | h0) * d;
            float t1 = float((b[8 + j] >> 4) | h1) * d;
            v[j] = t0 + m;
            v[j + 16] = t1 + m;
        }
        break;
    }
    case 8: { // Q2_K: 32 values = the two 16-value sub-blocks (h = 0, 1) of (n, j)
        device const uchar* b = R + (c / 8) * 84;
        uint cc = c % 8, n = cc / 4, j = cc % 4;
        float d = emb_h(b + 80), mn = emb_h(b + 82);
        for (uint h = 0; h < 2; h++) {
            uchar s = b[8 * n + 2 * j + h];
            float dl = d * float(s & 15), ml = mn * float(s >> 4);
            for (uint l = 0; l < 16; l++) {
                float t = dl * float((b[16 + 32 * n + 16 * h + l] >> (2 * j)) & 3);
                v[16 * h + l] = t - ml;
            }
        }
        break;
    }
    case 9:    // Q4_K
    case 10: { // Q5_K
        bool five = p.kind == 10;
        device const uchar* b = R + (c / 8) * (five ? 176 : 144);
        uint cc = c % 8, j = cc / 2, h = cc % 2;
        device const uchar* sc = b + 4;
        float d = emb_h(b), mn = emb_h(b + 2);
        uint s, m;
        if (cc < 4) {
            s = sc[cc] & 63;
            m = sc[cc + 4] & 63;
        } else {
            s = (sc[cc + 4] & 15) | ((sc[cc - 4] >> 6) << 4);
            m = (sc[cc + 4] >> 4) | ((sc[cc] >> 6) << 4);
        }
        float d1 = d * float(s), m1 = mn * float(m);
        device const uchar* qs = b + (five ? 48 : 16) + 32 * j;
        for (uint l = 0; l < 32; l++) {
            uint q = (qs[l] >> (4 * h)) & 15;
            if (five && (b[16 + l] & (1u << cc)) != 0) q += 16;
            float t = d1 * float(q);
            v[l] = t - m1;
        }
        break;
    }
    case 11: { // Q6_K: run `part` of the 128-value half `n`
        device const uchar* b = R + (c / 8) * 210;
        uint cc = c % 8, n = cc / 4, part = cc % 4;
        device const uchar* ql = b + 64 * n;
        device const uchar* qh = b + 128 + 32 * n;
        device const uchar* sc = b + 192 + 8 * n;
        float d = emb_h(b + 208);
        for (uint l = 0; l < 32; l++) {
            uint lo = (part & 1) != 0 ? ql[l + 32] : ql[l];
            lo = (part & 2) != 0 ? lo >> 4 : lo & 15;
            int q = int(lo | (((qh[l] >> (2 * part)) & 3) << 4)) - 32;
            float ds = d * float(as_type<char>(sc[l / 16 + 2 * part]));
            v[l] = ds * float(q);
        }
        break;
    }
    case 20: { // oxinfer int8
        device const char* q = (device const char*)p.q + (ulong)row * p.n + c * 32;
        float s = ((device const float*)p.s)[(ulong)row * (p.n / 32) + c];
        for (uint l = 0; l < 32; l++) v[l] = float(q[l]) * s;
        break;
    }
    case 21: { // oxinfer Q4 (Q4_0 layout, optional affine offsets)
        device const uchar* q = (device const uchar*)p.q + (ulong)row * (p.n / 2) + c * 16;
        float d = float(((device const half*)p.s)[(ulong)row * (p.n / 32) + c]);
        float m = p.o != 0 ? emb_bf(((device const ushort*)p.o)[(ulong)row * (p.n / p.group) + c * 32 / p.group]) : 0.f;
        for (uint j = 0; j < 16; j++) {
            float t0 = float(int(q[j] & 15) - 8) * d;
            float t1 = float(int(q[j] >> 4) - 8) * d;
            v[j] = t0 + m;
            v[j + 16] = t1 + m;
        }
        break;
    }
    case 22: { // MLX affine
        uint per = 32 / p.bits, mask = (1u << p.bits) - 1, ng = p.n / p.group;
        device const uint* w = (device const uint*)p.q + (ulong)row * (p.n / per);
        uint g = c * 32 / p.group;
        float s = emb_bf(((device const ushort*)p.s)[(ulong)row * ng + g]);
        float bb = emb_bf(((device const ushort*)p.o)[(ulong)row * ng + g]);
        for (uint l = 0; l < 32; l++) {
            uint e = c * 32 + l;
            uint q = (w[e / per] >> (p.bits * (e % per))) & mask;
            float t = s * float(q);
            v[l] = t + bb;
        }
        break;
    }
    default:
        for (uint l = 0; l < 32; l++) v[l] = 0.f;
    }
    for (uint l = 0; l < 32; l++) y[l] = v[l] * p.scale;
}

// Copies `n` floats from `src` to `dst` (the recurrent states of a chain, see
// `MetalPlan::greedy`).
kernel void dec_copy(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]], constant uint& n [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
    if (i < n) dst[i] = src[i];
}

// ---- The input transform of a Hadamard-folded weight (`Op::Fold`, `hadamard::Fold::apply`):
// out = H (D P x) over blocks of `block` values (one threadgroup per block and token): P the
// value-head permutation (value head k + nk r of x read as head r + rep k), D the signs, H the
// normalized Sylvester Walsh-Hadamard matrix, butterflies in the CPU's order (the same values).
struct FoldP { uint n; uint block; uint hd; uint nk; uint rep; uint xs; uint os; uint has_signs; float scale; uint _p0; uint _p1; uint _p2; };
kernel void dec_fold(device const float* x [[buffer(0)]], device float* out [[buffer(1)]], device const float* signs [[buffer(2)]],
                     constant FoldP& p [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]], uint2 ntg [[threads_per_threadgroup]]) {
    uint nt = ntg.x;
    threadgroup float v[1024];
    uint b0 = tg.x * p.block;
    device const float* xr = x + (ulong)tg.y * p.xs;
    for (uint i = tid; i < p.block; i += nt) {
        uint d = b0 + i, s = d;
        if (p.hd != 0) {
            uint h = d / p.hd, j = d % p.hd;
            s = p.hd * (h / p.rep + p.nk * (h % p.rep)) + j;
        }
        float a = xr[s];
        v[i] = p.has_signs != 0 ? a * signs[d] : a;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint h = 1; h < p.block; h *= 2) {
        for (uint i = tid; i < p.block / 2; i += nt) {
            uint a = (i / h) * 2 * h + i % h;
            float u = v[a], w = v[a + h];
            v[a] = u + w;
            v[a + h] = u - w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device float* o = out + (ulong)tg.y * p.os + b0;
    for (uint i = tid; i < p.block; i += nt) o[i] = v[i] * p.scale;
}
