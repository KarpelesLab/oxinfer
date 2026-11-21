// Inkling-2 prefill kernels (metal_inkling.rs), after the matrix-unit GEMMs' source (MMA_SHADER_SRC: tg_max,
// row_scale, the grouped GEMMs) and metal_inkling.metal (the weight loaders of NVFP4, MXFP8 and Q2_0).
#pragma clang fp reassociate(on)

// ---- Prefill passes (more tokens than the rings hold) ---------------------------------------------
// The raw inputs of the last `ring_cap` tokens into a convolution ring (grid: channel blocks,
// token), after the convolutions of the pass have read the ring.
struct IRsP { uint ch; uint x; uint pos; uint ntok; uint ring_cap; uint kn; uint _p0; uint _p1; };
kernel void ink_ring_store(device float* a [[buffer(0)]], constant IRsP& p [[buffer(1)]], device float* ring [[buffer(2)]],
                           uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x, ti = gid.y;
    if (i >= p.ch || ti + p.ring_cap < p.ntok) return;
    ring[(ulong)((p.pos + ti + p.kn - 1) % p.ring_cap) * p.ch + i] = a[p.x + (ulong)ti * p.ch + i];
}
// Rows of a pass's keys and values (written by position from `base` into `src`) to a sliding
// window's ring of `cap` rows: the last `cap` positions (grid: row bytes / 4, token).
struct IKrP { uint rb; uint sb; uint base; uint ntok; uint cap; uint _p0; uint _p1; uint _p2; };
kernel void ink_kv_ring(constant IKrP& p [[buffer(0)]], device const uchar* sk [[buffer(1)]], device const uchar* sv [[buffer(2)]],
                        device const uchar* sks [[buffer(3)]], device const uchar* svs [[buffer(4)]], device uchar* dk [[buffer(5)]],
                        device uchar* dv [[buffer(6)]], device uchar* dks [[buffer(7)]], device uchar* dvs [[buffer(8)]],
                        uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x, ti = gid.y;
    if (ti + p.cap < p.ntok) return;
    ulong r = (p.base + ti) % p.cap;
    if (i < p.rb / 4) {
        ((device uint*)(dk + r * p.rb))[i] = ((device const uint*)(sk + (ulong)ti * p.rb))[i];
        ((device uint*)(dv + r * p.rb))[i] = ((device const uint*)(sv + (ulong)ti * p.rb))[i];
    }
    if (i < p.sb / 4) {
        ((device uint*)(dks + r * p.sb))[i] = ((device const uint*)(sks + (ulong)ti * p.sb))[i];
        ((device uint*)(dvs + r * p.sb))[i] = ((device const uint*)(svs + (ulong)ti * p.sb))[i];
    }
}
// SiTU of grouped gate / up rows (`[rows][2 inter]`, interleaved with `il` 2) times each row's
// factor, as scaled half rows for the down GEMM (as `silu_half`).
struct ISiP { uint n; uint il; float cap_g; float cap_u; };
inline float situ(float g, float u, float cg, float cu) {
    float uu = cu > 0.f ? cu * precise::tanh(u / cu) : u;
    return cg * precise::tanh(g / cg) / (1.f + exp(-g)) * uu;
}
kernel void ink_situ_half(device const float* gu [[buffer(0)]], device half* ah [[buffer(1)]], device float* rs [[buffer(2)]],
                          constant ISiP& p [[buffer(3)]], device const float* fac [[buffer(4)]],
                          uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8];
    device const float* row = gu + (ulong)r * 2 * p.n;
    float f = fac[r];
    float m = 0.f;
    for (uint i = tid; i < p.n; i += 256) {
        float g = (p.il == 2 ? row[2 * i] : row[i]) * f, u = (p.il == 2 ? row[2 * i + 1] : row[p.n + i]) * f;
        m = max(m, fabs(situ(g, u, p.cap_g, p.cap_u)));
    }
    m = tg_max(m, sg, lane, red);
    float inv, s = row_scale(m, inv);
    device half* out = ah + (ulong)r * p.n;
    for (uint i = tid; i < p.n; i += 256) {
        float g = (p.il == 2 ? row[2 * i] : row[i]) * f, u = (p.il == 2 ? row[2 * i + 1] : row[p.n + i]) * f;
        out[i] = half(situ(g, u, p.cap_g, p.cap_u) * s);
    }
    if (tid == 0) rs[r] = inv;
}
// The experts' rows of each token summed with its routing weights times the down factors:
// `y[t] = sum_s w[t][s] fd[row] o[row]`, row = map[t k + s] (grid: blocks of h, token).
struct ICbP { uint h; uint k; uint sel; uint y; };
kernel void ink_combine(device float* a [[buffer(0)]], constant ICbP& p [[buffer(1)]], device const float* o [[buffer(2)]],
                        device const uint* map [[buffer(3)]], device const float* fd [[buffer(4)]],
                        uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x, ti = gid.y;
    if (i >= p.h) return;
    device const float* sel = a + p.sel + (ulong)ti * 2 * p.k;
    float s = 0.f;
    for (uint j = 0; j < p.k; j++) {
        uint r = map[ti * p.k + j];
        s += sel[p.k + j] * fd[r] * o[(ulong)r * p.h + i];
    }
    a[p.y + (ulong)ti * p.h + i] = s;
}

// Grouped expert GEMMs of the formats the matrix-unit library leaves out.
#define MK_MXF8 MXF8W{ (device const uchar*)qb, (device const float*)sb, p.n_in }
MMAG(mmag_nvfp4, NVFP4W, MK_GG(NVFP4W)) MMAG(mmag_Q2_0, Q2_0W, MK_GG(Q2_0W)) MMAG(mmag_mxf8, MXF8W, MK_MXF8)
