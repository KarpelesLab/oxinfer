// The ONNX interpreter's operators on Metal (metal_onnx.rs, `crate::onnx`): channels-last f32
// tensors, one thread per output value. The steps a tensor goes through after an operator
// (`Post` in onnx.rs) are the CPU's: ReLU, fake quantization (`(clamp(round(v / s) + zp, lo, hi)
// - zp2) * s2`), and the end of an integer step (`(clamp(round(v * mult[c]) + zp, lo, hi) - zp2)
// * s2`, `mult` per channel).

struct OxPost {
    uint kind;   // bit 0: ReLU first; bit 1: fake quantization; bit 2: integer requantization; bit 3: logistic; bit 4: rounding
    float s; float zp; float lo; float hi; float s2; float zp2; float _p;
};

inline float ox_post(float v, constant OxPost& q, device const float* mult, uint c) {
    if (q.kind & 1) v = max(v, 0.0f);
    if (q.kind & 2) v = (clamp(rint(precise::divide(v, q.s)) + q.zp, q.lo, q.hi) - q.zp2) * q.s2;
    if (q.kind & 4) v = (clamp(rint(v * mult[c]) + q.zp, q.lo, q.hi) - q.zp2) * q.s2;
    if (q.kind & 8) v = 1.0f / (1.0f + precise::exp(-v));
    if (q.kind & 16) v = (clamp(rint(v) + q.zp, q.lo, q.hi) - q.zp2) * q.s2;
    return v;
}

// A convolution (1D as 2D with h = 1), channels-last: y[n][oy][ox][co] = bias[co] + sum over the
// taps inside the image of x * w, w in ONNX's [co][ci of the group][ky][kx] layout. With
// s_in > 0 the input is the integer behind a fake-quantized tensor: round(x / s_in).
struct OxConv {
    uint n; uint h; uint w; uint cin;
    uint oh; uint ow; uint cout; uint kh;
    uint kw; uint sh; uint sw; uint dh;
    uint dw; uint pt; uint pl; uint cpg;
    uint copg; float s_in; uint _p0; uint _p1;
};
kernel void ox_conv(device const float* x [[buffer(0)]], device const float* wt [[buffer(1)]], device const float* bias [[buffer(2)]],
                    device float* y [[buffer(3)]], constant OxConv& p [[buffer(4)]], constant OxPost& q [[buffer(5)]],
                    device const float* mult [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
    uint total = p.n * p.oh * p.ow * p.cout;
    if (gid >= total) return;
    uint co = gid % p.cout;
    uint pix = gid / p.cout;
    uint ox = pix % p.ow, oy = (pix / p.ow) % p.oh, b = pix / (p.ow * p.oh);
    uint g = co / p.copg;
    float acc = bias[co];
    device const float* wc = wt + (ulong)co * p.cpg * p.kh * p.kw;
    for (uint ky = 0; ky < p.kh; ky++) {
        int iy = int(oy * p.sh + ky * p.dh) - int(p.pt);
        if (iy < 0 || iy >= int(p.h)) continue;
        for (uint kx = 0; kx < p.kw; kx++) {
            int ix = int(ox * p.sw + kx * p.dw) - int(p.pl);
            if (ix < 0 || ix >= int(p.w)) continue;
            device const float* xs = x + (((ulong)b * p.h + iy) * p.w + ix) * p.cin + g * p.cpg;
            uint kk = ky * p.kw + kx;
            if (p.s_in > 0.0f) {
                for (uint ci = 0; ci < p.cpg; ci++) acc += rint(precise::divide(xs[ci], p.s_in)) * wc[ci * p.kh * p.kw + kk];
            } else {
                for (uint ci = 0; ci < p.cpg; ci++) acc += xs[ci] * wc[ci * p.kh * p.kw + kk];
            }
        }
    }
    y[gid] = ox_post(acc, q, mult, co);
}

// A pointwise convolution with every input channel (a GEMM over the pixels): 32 output channels
// x 32 pixels a threadgroup, the 8x8 SIMD-group products over cin in steps of 8 (cin % 8 == 0,
// the weights padded to 32 rows), then the bias and the steps. Rows of x past the last pixel
// are read (the buffer has room), never written.
struct OxPw { uint npix; uint cin; uint cout; float s_in; };
kernel void ox_pw(device const float* x [[buffer(0)]], device const float* wt [[buffer(1)]], device const float* bias [[buffer(2)]],
                  device float* y [[buffer(3)]], constant OxPw& p [[buffer(4)]], constant OxPost& q [[buffer(5)]],
                  device const float* mult [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                  uint sg [[simdgroup_index_in_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float tile[32 * 36];
    uint lo = (sg / 2) * 16, lt = (sg % 2) * 16;
    device const float* ws = wt + (ulong)(tg.x * 32 + lo) * p.cin;
    device const float* xs = x + (ulong)(tg.y * 32 + lt) * p.cin;
    simdgroup_float8x8 c00(0.f), c01(0.f), c10(0.f), c11(0.f);
    for (uint k = 0; k < p.cin; k += 8) {
        simdgroup_float8x8 a0, a1, b0, b1;
        simdgroup_load(a0, ws + k, p.cin);
        simdgroup_load(a1, ws + (ulong)8 * p.cin + k, p.cin);
        simdgroup_load(b0, xs + k, p.cin, ulong2(0, 0), true);
        simdgroup_load(b1, xs + (ulong)8 * p.cin + k, p.cin, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(c00, a0, b0, c00); simdgroup_multiply_accumulate(c01, a0, b1, c01);
        simdgroup_multiply_accumulate(c10, a1, b0, c10); simdgroup_multiply_accumulate(c11, a1, b1, c11);
    }
    simdgroup_store(c00, tile + lt * 36 + lo, 36, ulong2(0, 0), true);
    simdgroup_store(c01, tile + (lt + 8) * 36 + lo, 36, ulong2(0, 0), true);
    simdgroup_store(c10, tile + lt * 36 + lo + 8, 36, ulong2(0, 0), true);
    simdgroup_store(c11, tile + (lt + 8) * 36 + lo + 8, 36, ulong2(0, 0), true);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < 32 * 32; e += 128) {
        uint r = e / 32, c = e % 32;
        uint t = tg.y * 32 + r, o = tg.x * 32 + c;
        if (t >= p.npix || o >= p.cout) continue;
        y[(ulong)t * p.cout + o] = ox_post(tile[r * 36 + c] + bias[o], q, mult, o);
    }
}

// The integers behind a fake-quantized tensor: y = round(x / s).
kernel void ox_round(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], constant float& s [[buffer(2)]],
                     constant uint& n [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    if (gid < n) y[gid] = rint(precise::divide(x[gid], s));
}

// y = post(a + b) element-wise; with exact (MLAS's QLinearAdd) the sum is
// rac round(a / sa) + rbc round(b / sb) + zc, each product and sum rounded to f32.
struct OxAdd { uint n; uint exact; float sa; float sb; float rac; float rbc; float zc; float _p; };
kernel void ox_add(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* y [[buffer(2)]],
                   constant OxAdd& p [[buffer(3)]], constant OxPost& q [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) return;
    float v;
    if (p.exact != 0) {
#pragma clang fp contract(off)
        float u = p.rac * rint(precise::divide(a[gid], p.sa));
        float w = p.rbc * rint(precise::divide(b[gid], p.sb));
        v = (u + w) + p.zc;
    } else {
        v = a[gid] + b[gid];
    }
    y[gid] = ox_post(v, q, y, 0);
}

// y = post(x) element-wise (a fake-quantization or ReLU node of its own).
kernel void ox_unary(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], constant uint& n [[buffer(2)]],
                     constant OxPost& q [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    if (gid < n) y[gid] = ox_post(x[gid], q, y, 0);
}

// Nearest resize, channels-last: y[i][oy][ox][c] = x[i][ys[oy]][xs[ox]][c].
struct OxRs { uint n; uint h; uint w; uint c; uint oh; uint ow; uint _p0; uint _p1; };
kernel void ox_resize(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], device const uint* ys [[buffer(2)]],
                      device const uint* xs [[buffer(3)]], constant OxRs& p [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    uint total = p.n * p.oh * p.ow * p.c;
    if (gid >= total) return;
    uint c = gid % p.c, pix = gid / p.c;
    uint ox = pix % p.ow, oy = (pix / p.ow) % p.oh, i = pix / (p.ow * p.oh);
    y[gid] = x[(((ulong)i * p.h + ys[oy]) * p.w + xs[ox]) * p.c + c];
}

// One part of a channel concatenation: dst[px][off + c] = src[px][c] for c < cs.
struct OxCat { uint npix; uint cs; uint cd; uint off; };
kernel void ox_cat(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]], constant OxCat& p [[buffer(2)]],
                   uint gid [[thread_position_in_grid]]) {
    if (gid >= p.npix * p.cs) return;
    uint c = gid % p.cs, px = gid / p.cs;
    dst[(ulong)px * p.cd + p.off + c] = src[gid];
}
