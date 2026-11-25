// ---- Whole UNet passes (metal_unet.rs, crate::gpu::unet) ----
// Activations are f32, channels-last, in a padded layout: each image of the batch is its
// (h + 2) x (w + 2) grid with a border of zero pixels, the images one after the other, after a
// guard of G = w + 3 rows (and more after the last), so that the input row of any tap of a 3x3
// convolution of row r is r + (ky - 1) (w + 2) + (kx - 1): a 3x3 convolution is nine shifted
// GEMMs over the same half rows, computed for every padded row (the border outputs are stored
// as zeros). The GEMM inputs are half rows without scales (normalized, or activations whose
// range half holds, as in diffusers' fp16 inference); every product accumulates in f32.
#if MMA_UNITS
struct UGEO { uint h; uint w; uint G; uint P; };
inline uint un_prow(UGEO g, uint b, uint y, uint x) { return g.G + b * g.P + (y + 1) * (g.w + 2) + x + 1; }
// Padded row r -> (image, y, x); false on a border or guard row.
inline bool un_interior(UGEO g, uint nimg, uint r, thread uint& b, thread uint& y, thread uint& x) {
    if (r < g.G) return false;
    uint rr = r - g.G;
    b = rr / g.P;
    if (b >= nimg) return false;
    uint q = rr - b * g.P, py = q / (g.w + 2), px = q - py * (g.w + 2);
    if (py == 0 || px == 0 || py > g.h || px > g.w) return false;
    y = py - 1; x = px - 1;
    return true;
}

// Group-norm statistics, first stage: per image and chunk of `chunk` pixels, the sum and sum of
// squares of each group. Grid (chunks, images), 256 threads.
struct UGS { UGEO g; uint c; uint groups; uint chunk; uint nchunk; };
kernel void un_gn_stats(device const float* x [[buffer(0)]], device float2* part [[buffer(1)]], constant UGS& p [[buffer(2)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float2 cs[2560];
    uint b = tg.y, n = p.g.h * p.g.w;
    uint p0 = tg.x * p.chunk, p1 = min(p0 + p.chunk, n);
    uint cpg = p.c / p.groups;
    device const float* x0 = x + (ulong)un_prow(p.g, b, 0, 0) * p.c;
    for (uint ch = tid; ch < p.c; ch += 256) {
        // Sums of the values minus the group's first value (no cancellation when the mean is
        // large against the spread).
        float s = 0.f, s2 = 0.f, k = x0[(ch / cpg) * cpg];
        uint y = p0 / p.g.w, xx = p0 - y * p.g.w;
        for (uint q = p0; q < p1; q++) {
            float v = x[(ulong)un_prow(p.g, b, y, xx) * p.c + ch] - k;
            s += v; s2 += v * v;
            if (++xx == p.g.w) { xx = 0; y++; }
        }
        cs[ch] = float2(s, s2);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint gi = tid; gi < p.groups; gi += 256) {
        float2 a = float2(0.f);
        for (uint j = 0; j < cpg; j++) a += cs[gi * cpg + j];
        part[((ulong)b * p.nchunk + tg.x) * p.groups + gi] = a;
    }
}
// Second stage: (mean, 1 / std) per image and group. One thread per (group, image).
kernel void un_gn_fin(device const float2* part [[buffer(0)]], device float2* st [[buffer(1)]], constant UGS& p [[buffer(2)]],
                      constant float& eps [[buffer(3)]], device const float* x [[buffer(4)]],
                      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    uint gi = tg.x * 64 + tid, b = tg.y;
    if (gi >= p.groups) return;
    float s = 0.f, s2 = 0.f;
    for (uint k = 0; k < p.nchunk; k++) { float2 a = part[((ulong)b * p.nchunk + k) * p.groups + gi]; s += a.x; s2 += a.y; }
    float cnt = float(p.g.h * p.g.w * (p.c / p.groups));
    float m = s / cnt;
    float var = max(s2 / cnt - m * m, 0.f);
    float k = x[(ulong)un_prow(p.g, b, 0, 0) * p.c + gi * (p.c / p.groups)];
    st[b * p.groups + gi] = float2(k + m, 1.0f / sqrt(var + eps));
}

// The half input of a GEMM from f32 padded rows: optionally group-normalized (stats from
// un_gn_fin, gamma then beta in gb) and SiLU'd, optionally upsampled 2x (nearest: the source is
// at half the size), into padded half rows (zeros on borders and guards; `rows` of them) or
// compact half rows (`compact`: image b's pixel q at row b h w + q). One thread per 4 channels
// of a row. norm 2: not normalized but scaled by un_scale of the largest magnitude (st holds its
// bits, from un_amax), so that activations beyond half's range (an SDXL VAE's) fit; the GEMM
// divides the products by the same factor.
inline float un_scale(uint bits) {
    float m = as_type<float>(bits);
    if (!(m > 16384.0f)) return 1.0f;
    int e = 0;
    frexp(min(m, 3.0e38f), e);
    return ldexp(1.0f, 14 - e);
}
kernel void un_amax_clear(device uint* m [[buffer(0)]], uint i [[thread_position_in_grid]]) { if (i == 0) m[0] = 0; }
// The largest magnitude over the interior pixels of a padded f32 tensor, as float bits (atomic max).
struct UAM { UGEO g; uint c; uint nimg; uint rows; uint _p; };
kernel void un_amax(device const float* x [[buffer(0)]], device atomic_uint* m [[buffer(1)]], constant UAM& p [[buffer(2)]],
                    uint i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint c4 = p.c / 4, row = i / c4, j = i - row * c4;
    float a = 0.f;
    uint b, y, xx;
    if (row < p.rows && un_interior(p.g, p.nimg, row, b, y, xx)) {
        float4 v = abs(*((device const float4*)(x + (ulong)row * p.c) + j));
        a = max(max(v.x, v.y), max(v.z, v.w));
        if (isnan(a)) a = INFINITY;
    }
    a = simd_max(a);
    if (lane == 0 && a > 16384.0f) atomic_fetch_max_explicit(m, as_type<uint>(a), memory_order_relaxed);
}
struct UPR { UGEO d; UGEO s; uint c; uint nimg; uint norm; uint silu; uint up; uint compact; uint groups; uint rows; };
kernel void un_prep(device const float* x [[buffer(0)]], device half* out [[buffer(1)]], device const float2* st [[buffer(2)]],
                    device const float* gb [[buffer(3)]], constant UPR& p [[buffer(4)]],
                    uint i [[thread_position_in_grid]]) {
    uint c4 = p.c / 4;
    uint row = i / c4, j = i - row * c4;
    if (row >= p.rows) return;
    uint b, y, xx;
    bool in;
    if (p.compact != 0) {
        uint n = p.d.h * p.d.w;
        b = row / n; uint q = row - b * n; y = q / p.d.w; xx = q - y * p.d.w;
        in = b < p.nimg;
    } else {
        in = un_interior(p.d, p.nimg, row, b, y, xx);
    }
    device half4* o = (device half4*)(out + (ulong)row * p.c) + j;
    if (!in) { *o = half4(0.h); return; }
    uint sr = un_prow(p.s, b, y >> p.up, xx >> p.up);
    float4 v = *((device const float4*)(x + (ulong)sr * p.c) + j);
    if (p.norm == 2) {
        v *= un_scale(((device const uint*)st)[0]);
    } else if (p.norm != 0) {
        uint cpg = p.c / p.groups;
        float r[4] = { v.x, v.y, v.z, v.w };
        for (uint k = 0; k < 4; k++) {
            uint ch = j * 4 + k;
            float2 m = st[b * p.groups + ch / cpg];
            float t = (r[k] - m.x) * m.y * gb[ch] + gb[p.c + ch];
            r[k] = p.silu != 0 ? t / (1.0f + exp(-t)) : t;
        }
        v = float4(r[0], r[1], r[2], r[3]);
    }
    *o = half4(v);
}

// Unfolded 3x3 (or k x k) neighbourhoods of padded half rows, for strided convolutions (and
// inputs too narrow for the shifted GEMMs): compact rows of `kp` halves (taps of c channels
// then zeros), output pixel (oy, ox) of image b reading input (oy s + ky - pad, ox s + kx - pad).
struct UIC { UGEO s; uint c; uint k; uint stride; uint pad; uint oh; uint ow; uint kp; uint rows; };
kernel void un_im2col(device const half* x [[buffer(0)]], device half* out [[buffer(1)]], constant UIC& p [[buffer(2)]],
                      uint i [[thread_position_in_grid]]) {
    uint row = i / p.kp, kk = i - row * p.kp;
    if (row >= p.rows) return;
    uint n = p.oh * p.ow, b = row / n, q = row - b * n, oy = q / p.ow, ox = q - oy * p.ow;
    half v = 0.h;
    if (kk < p.k * p.k * p.c) {
        uint tap = kk / p.c, ci = kk - tap * p.c, ky = tap / p.k, kx = tap - ky * p.k;
        // Padded coordinates: input y = oy s + ky - pad sits at padded y + 1.
        int py = int(oy * p.stride + ky) - int(p.pad) + 1, px = int(ox * p.stride + kx) - int(p.pad) + 1;
        if (py >= 0 && px >= 0 && py <= int(p.s.h) + 1 && px <= int(p.s.w) + 1)
            v = x[((ulong)p.s.G + b * p.s.P + uint(py) * (p.s.w + 2) + uint(px)) * p.c + ci];
    }
    out[(ulong)row * p.kp + kk] = v;
}

// y = sum over taps of A[r + off[tap]] W_tap^T: half rows of `cin` (A), bf16 weights [n_out][ldw]
// whose tap t is columns t cin .. (t + 1) cin. TMR rows x 64 outputs per threadgroup (4
// SIMD-groups; 128 x 128 tiles ran at 60-75% of this speed on every UNet shape measured, 64-row
// ones gain where 128-row tiles leave the GPU's cores short of threadgroups), rows r0 + ... <
// r_end. Stores (omode):
//   0 f32 rows ldy wide, row r (compact: image r / nimg);
//   1 f32 padded rows of geometry g from compact rows (image b's pixel q);
//   2 f32 padded rows from padded rows (borders stored as zeros);
//   3 half rows of heads: output o to column (o / dh) dhp + o % dh of rows ldy wide.
// With bias: + bias[b * bstride + o]; with add: + addb at the destination's index (addb may be y).
struct UG { uint n_out; uint ldw; uint cin; uint taps; int off[9]; uint a_rows; uint r0; uint r_end; uint omode; uint ldy;
            uint has_bias; uint bstride; uint add; uint nimg; uint dh; uint dhp; UGEO g; uint scaled; uint _p0; uint _p1; uint _p2; };
template <int TMR, int TN = 64>
inline void un_gemm_body(device const bfloat* w, device const half* xh, device float* y, device const float* bias, device const float* addb,
                         device const uint* amax, constant UG& p, uint2 tg) {
    float inv = p.scaled != 0 ? 1.0f / un_scale(amax[0]) : 1.0f;
    uint o0 = tg.x * TN, t0 = p.r0 + tg.y * TMR;
    int nop = int(p.n_out + TN - 1) / TN * TN;
    XT X((device half*)xh, dextents<int32_t, 2>(p.cin, p.a_rows));
    constexpr auto desc = matmul2d_descriptor(TMR, TN, static_cast<int>(dynamic_extent), false, true, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;
    array<int32_t, 2> ws = { 1, int32_t(p.ldw) };
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> W0((device bfloat*)w, dextents<int32_t, 2>(p.cin, nop), ws);
    auto a0 = X.slice(0, int(t0) + p.off[0]);
    auto b0 = W0.slice(0, o0);
    auto c = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
    zero(c);
    op.run(a0, b0, c);
    for (uint t = 1; t < p.taps; t++) {
        tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Wt((device bfloat*)w + t * p.cin, dextents<int32_t, 2>(p.cin, nop), ws);
        auto a = X.slice(0, int(t0) + p.off[t]);
        auto bt = Wt.slice(0, o0);
        op.run(a, bt, c);
    }
#pragma unroll
    for (uint16_t i = 0; i < c.get_capacity(); ++i) {
        if (!c.is_valid_element(i)) continue;
        auto idx = c.get_multidimensional_index(i);
        uint o = o0 + idx[0], r = t0 + idx[1];
        if (o >= p.n_out || r >= p.r_end) continue;
        float v = c[i] * inv;
        uint b = 0, dst = r;
        bool border = false;
        if (p.omode == 2) {
            uint yy, xx;
            border = !un_interior(p.g, p.nimg, r, b, yy, xx);
        } else {
            b = r / p.nimg;
            if (p.omode == 1) { uint q = r - b * p.nimg; dst = un_prow(p.g, b, q / p.g.w, q % p.g.w); }
        }
        if (p.omode == 3) {
            if (p.has_bias != 0) v += bias[o];
            device half* hr = (device half*)y + (ulong)dst * p.ldy + (o / p.dh) * p.dhp;
            uint d = o % p.dh;
            hr[d] = half(v);
            if (d < p.dhp - p.dh) hr[p.dh + d] = 0.h;   // the head's padding: zeros
            continue;
        }
        device float* d = y + (ulong)dst * p.ldy + o;
        if (border) { *d = 0.f; continue; }
        if (p.has_bias != 0) v += bias[b * p.bstride + o];
        if (p.add != 0) v += addb[(ulong)dst * p.ldy + o];
        *d = v;
    }
}
#define UN_GEMM(NAME, TMR) \
kernel void NAME(device const bfloat* w [[buffer(0)]], device const half* xh [[buffer(1)]], device float* y [[buffer(2)]], \
                 device const float* bias [[buffer(3)]], constant UG& p [[buffer(4)]], device const float* addb [[buffer(5)]], \
                 device const uint* amax [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]]) { \
    un_gemm_body<TMR>(w, xh, y, bias, addb, amax, p, tg); }
UN_GEMM(un_gemm, 128) UN_GEMM(un_gemm_m64, 64)

// Layer norm of f32 rows of c (c % 4 == 0, c <= 2048) into half rows. One SIMD-group per row.
struct ULN { uint c; uint rows; float eps; uint _p; };
kernel void un_ln(device const float* x [[buffer(0)]], device half* out [[buffer(1)]], device const float* gb [[buffer(2)]],
                  constant ULN& p [[buffer(3)]], uint r [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                  uint sg [[simdgroup_index_in_threadgroup]]) {
    uint row = r * 4 + sg;
    if (row >= p.rows) return;
    device const float4* xr = (device const float4*)(x + (ulong)row * p.c);
    uint c4 = p.c / 4;
    float4 v[16];
    float s = 0.f;
    for (uint k = 0; k < 16; k++) {
        uint j = lane + 32 * k;
        v[k] = j < c4 ? xr[j] : float4(0.f);
        s += v[k].x + v[k].y + v[k].z + v[k].w;
    }
    float mean = simd_sum(s) / float(p.c);
    float s2 = 0.f;
    for (uint k = 0; k < 16; k++) {
        uint j = lane + 32 * k;
        if (j < c4) { float4 d = v[k] - mean; s2 += dot(d, d); }
    }
    float inv = 1.0f / sqrt(simd_sum(s2) / float(p.c) + p.eps);
    device const float4* g4 = (device const float4*)gb;
    device const float4* b4 = (device const float4*)(gb + p.c);
    device half4* o = (device half4*)(out + (ulong)row * p.c);
    for (uint k = 0; k < 16; k++) {
        uint j = lane + 32 * k;
        if (j < c4) o[j] = half4((v[k] - mean) * inv * g4[j] + b4[j]);
    }
}

// GEGLU: rows of [value | gate] (inner each, f32) into half rows of value * gelu(gate) (erf).
inline float un_erf(float x) {
    // Abramowitz and Stegun 7.1.26 (|error| < 1.5e-7).
    float t = 1.0f / (1.0f + 0.3275911f * abs(x));
    float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * exp(-x * x);
    return copysign(y, x);
}
struct UGG { uint inner; uint rows; uint _p0; uint _p1; };
kernel void un_geglu(device const float* g [[buffer(0)]], device half* out [[buffer(1)]], constant UGG& p [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
    uint n4 = p.inner / 4, row = i / n4, j = i - row * n4;
    if (row >= p.rows) return;
    device const float4* r = (device const float4*)(g + (ulong)row * 2 * p.inner);
    float4 v = r[j], ga = r[n4 + j];
    float4 ge = 0.5f * ga * (1.0f + float4(un_erf(ga.x * M_SQRT1_2_F), un_erf(ga.y * M_SQRT1_2_F), un_erf(ga.z * M_SQRT1_2_F),
                                           un_erf(ga.w * M_SQRT1_2_F)));
    ((device half4*)(out + (ulong)row * p.inner))[j] = half4(v * ge);
}

// Concatenation of the channels of two padded f32 tensors of one geometry.
struct UCC { uint ca; uint cb; uint rows; uint _p; };
kernel void un_concat(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* out [[buffer(2)]],
                      constant UCC& p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    uint c4 = (p.ca + p.cb) / 4, row = i / c4, j = i - row * c4;
    if (row >= p.rows) return;
    float4 v = j * 4 < p.ca ? ((device const float4*)(a + (ulong)row * p.ca))[j]
                            : ((device const float4*)(b + (ulong)row * p.cb))[j - p.ca / 4];
    ((device float4*)(out + (ulong)row * (p.ca + p.cb)))[j] = v;
}

// Fused attention over heads padded to D (half q / k / v rows of heads * D, zeros past the
// real head width), into half rows of heads * dh (o_ld wide). Image z's queries are rows
// z qs .. z qs + tq, its keys rows z ks .. z ks + tk (key rows up to the next multiple of 64
// must be finite: they are masked). One threadgroup (4 SIMD-groups) per 64 queries of one head
// of one image; keys stream in blocks of 64 with an online softmax (flash_attn128_body's).
// Heads wider than 160 (a VAE's single head of 512) split their values and outputs into DV-wide
// chunks, a threadgroup each (the scores computed once per chunk): grid y is heads x D / DV.
#define UN_SLD 68
#define UN_PLD (2 * UN_SLD)
// Scores are in base 2 (the scale times log2(e), exp2). The running row maximum moves only
// when a block's scores pass it by more than UN_LAZY (probabilities then stay below 2^UN_LAZY),
// and the output's rescaling is skipped for key blocks where no row of the threadgroup moved.
// Probabilities go to half scaled by 2^10: with thousands of keys most are far below half's
// smallest normal (6e-5), where they would lose their precision; the sums divide it out
// (2^UN_LAZY * UN_PSCALE stays inside half's range).
#define UN_LAZY 5.0f
#define UN_PSCALE 1024.0f
struct UFA { uint tq; uint tk; uint qs; uint ks; uint heads; uint dh; uint o_ld; float scale; };
template <int D, int DV>
inline void un_flash_body(device const half* q, device const half* k, device const half* v, device half* o, constant UFA& p,
                          threadgroup float* S, threadgroup half* P, threadgroup float* A, threadgroup float* L,
                          threadgroup uint* RS, uint3 tg, uint tid) {
    uint q0 = tg.x * 64, h = tg.y / (D / DV), vc = (tg.y % (D / DV)) * DV, z = tg.z;
    int ld = int(p.heads) * D;
    int qrows = int(p.qs) * int(z) + int(q0) + 64, krows = int(p.ks) * int(z) + int(p.tk + 63) / 64 * 64;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> Qt((device half*)q, dextents<int32_t, 2>(ld, qrows));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> Kt((device half*)k, dextents<int32_t, 2>(ld, krows));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> Vt((device half*)v, dextents<int32_t, 2>(ld, krows));
    // Score and probability rows padded (UN_SLD floats, UN_PLD halves) against bank conflicts:
    // each thread reads half a row.
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> St(S, dextents<int32_t, 2>(64, 64), array<int32_t, 2>{1, UN_SLD});
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> Pt(P, dextents<int32_t, 2>(64, 64), array<int32_t, 2>{1, UN_PLD});
    constexpr auto ds = matmul2d_descriptor(64, 64, D, false, true, false);
    constexpr auto dov = matmul2d_descriptor(64, DV, 64, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<ds, execution_simdgroups<4>> ops;
    matmul2d<dov, execution_simdgroups<4>> opo;
    uint qr0 = p.qs * z + q0, kr0 = p.ks * z;
    auto qa = Qt.template slice<D, 64>(h * D, qr0);
    auto vb0 = Vt.template slice<DV, 64>(h * D + vc, kr0);
    auto co = opo.template get_destination_cooperative_tensor<decltype(Pt), decltype(vb0), float>();
    zero(co);
    uint r = tid / 2, part = tid % 2;
    float m = -INFINITY, l = 0.0f;
    float sl2 = p.scale * 1.44269504f;
    threadgroup uint& rescale = RS[0];
    if (tid == 0) rescale = 0;
    for (uint kb = 0; kb < p.tk; kb += 64) {
        auto kt = Kt.template slice<D, 64>(h * D, kr0 + kb);
        auto cs = ops.template get_destination_cooperative_tensor<decltype(qa), decltype(kt), float>();
        zero(cs);
        ops.run(qa, kt, cs);
        cs.store(St);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float x[32];
        float bm = -INFINITY;
        for (uint j = 0; j < 32; j++) {
            uint col = part * 32 + j;
            x[j] = kb + col < p.tk ? S[r * UN_SLD + col] * sl2 : -INFINITY;
            bm = max(bm, x[j]);
        }
        bm = max(bm, simd_shuffle_xor(bm, 1));
        bool grow = bm > m + UN_LAZY;
        float mn = grow ? bm : m;
        float alpha = grow ? exp2(m - mn) : 1.0f;
        if (grow) rescale = 1;
        float ps = 0.0f;
        for (uint j = 0; j < 32; j++) { float e = exp2(x[j] - mn); P[r * UN_PLD + part * 32 + j] = half(e * UN_PSCALE); ps += e; }
        ps += simd_shuffle_xor(ps, 1);
        l = l * alpha + ps;
        m = mn;
        if (part == 0) A[r] = alpha;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (rescale != 0) {
#pragma unroll
            for (uint16_t i = 0; i < co.get_capacity(); ++i)
                if (co.is_valid_element(i)) { auto idx = co.get_multidimensional_index(i); co[i] *= A[idx[1]]; }
        }
        auto vt = Vt.template slice<DV, 64>(h * D + vc, kr0 + kb);
        opo.run(Pt, vt, co);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) rescale = 0;
    }
    if (part == 0) L[r] = 1.0f / (l * UN_PSCALE);
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint16_t i = 0; i < co.get_capacity(); ++i)
        if (co.is_valid_element(i)) {
            auto idx = co.get_multidimensional_index(i);
            uint row = q0 + idx[1], d = vc + idx[0];
            if (row < p.tq && d < p.dh) o[(ulong)(p.qs * z + row) * p.o_ld + h * p.dh + d] = half(co[i] * L[idx[1]]);
        }
}
#define UN_FLASH(NAME, D, DV) \
kernel void NAME(device const half* q [[buffer(0)]], device const half* k [[buffer(1)]], device const half* v [[buffer(2)]], \
                        device half* o [[buffer(3)]], constant UFA& p [[buffer(4)]], \
                        uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup float S[64 * UN_SLD]; threadgroup half* P = (threadgroup half*)S; threadgroup float A[64]; threadgroup float L[64]; \
    threadgroup uint RS[1]; \
    un_flash_body<D, DV>(q, k, v, o, p, S, P, A, L, RS, tg, tid); }
UN_FLASH(un_flash48, 48, 48) UN_FLASH(un_flash64, 64, 64) UN_FLASH(un_flash80, 80, 80) UN_FLASH(un_flash96, 96, 96) UN_FLASH(un_flash128, 128, 128) UN_FLASH(un_flash160, 160, 160)
UN_FLASH(un_flash512, 512, 128)

// f32 rows to half rows (n4 float4s each).
kernel void un_half(device const float* x [[buffer(0)]], device half* out [[buffer(1)]], constant uint& n [[buffer(2)]],
                    uint i [[thread_position_in_grid]]) {
    if (i < n) ((device half4*)out)[i] = half4(((device const float4*)x)[i]);
}
#endif
