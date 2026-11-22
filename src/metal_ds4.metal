// DeepSeek V4 / V4.1 prefill attention (metal_ds4.rs): every query head attends to its own list
// of rows of a key table (its sliding window, then its chosen compressed keys); the keys are the
// values (one shared 512-wide key-value head), and each head has a learned sink logit.
#include <metal_stdlib>
using namespace metal;

struct Ds4AttnP {
    uint nh;
    float scale;
    uint _p0;
    uint _p1;
};

// Grid (nh / 8, t), 256 threads: SIMD-group sg takes head 8 tg.x + sg of query tg.y. Query i's keys
// are rows idx[meta[2 i] .. meta[2 i] + meta[2 i + 1]] of kt ([nk, 512]); q and out are
// [t, nh, 512]. Lane l holds dimensions l + 32 m (m < 16) of the query and of the running output.
// Online softmax that starts from the sink (score sink[h], value 0), in key order.
kernel void ds4_attn(device const float* q [[buffer(0)]], device const float* kt [[buffer(1)]],
                     device const uint* idx [[buffer(2)]], device const uint* meta [[buffer(3)]],
                     device const float* sink [[buffer(4)]], device float* out [[buffer(5)]],
                     constant Ds4AttnP& p [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint h = tg.x * 8 + sg, i = tg.y;
    if (h >= p.nh) {
        return;
    }
    device const float* qp = q + ((ulong)i * p.nh + h) * 512;
    float qv[16], o[16];
    for (uint m = 0; m < 16; m++) {
        qv[m] = qp[lane + 32 * m];
        o[m] = 0.0f;
    }
    float mx = sink[h], l = 1.0f;
    uint off = meta[2 * i], n = meta[2 * i + 1];
    for (uint j = 0; j < n; j++) {
        device const float* kp = kt + (ulong)idx[off + j] * 512;
        float kv[16];
        float d = 0.0f;
        for (uint m = 0; m < 16; m++) {
            kv[m] = kp[lane + 32 * m];
            d = fma(qv[m], kv[m], d);
        }
        float s = simd_sum(d) * p.scale;
        float mn = max(mx, s);
        float a = exp(mx - mn), e = exp(s - mn);
        l = fma(l, a, e);
        for (uint m = 0; m < 16; m++) {
            o[m] = fma(o[m], a, e * kv[m]);
        }
        mx = mn;
    }
    device float* op = out + ((ulong)i * p.nh + h) * 512;
    float inv = 1.0f / l;
    for (uint m = 0; m < 16; m++) {
        op[lane + 32 * m] = o[m] * inv;
    }
}
