// ---- Split-precision attention (metal_dit.rs, appended after MMA_SHADER_SRC) ----
// flash_attn128_mma's arithmetic with queries and keys as half pairs hi + lo (hi = half(x),
// lo = half(x - hi): 22 bits of x) and scores Qh Kh^T + Qh Kl^T + Ql Kh^T in f32 (the dropped
// Ql Kl^T is 2^-22 of a score): logits as exact as float32 operands give them, on half-precision
// matrix units (Qwen-Image's logits run into the thousands, where half queries and keys alone
// move the velocity ~10%). Probabilities and values half, as in the fast kernel.
#if MMA_UNITS
kernel void dit_flash128_hilo(device const half* q [[buffer(0)]], device const half* k [[buffer(1)]],
                              device const half* v [[buffer(2)]], device float* o [[buffer(3)]], constant FAM& p [[buffer(4)]],
                              device const half* ql [[buffer(5)]], device const half* kl [[buffer(6)]],
                              uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float S[64 * 64];
    threadgroup half P[64 * 64];
    threadgroup float A[64];
    threadgroup float L[64];
    threadgroup uint rescale;
    uint q0 = tg.x * 64, h0 = tg.y * 128;
    uint tqp = (p.tq + 63) / 64 * 64, tkp = (p.tk + 63) / 64 * 64;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> Qt((device half*)q, dextents<int32_t, 2>(p.d, tqp));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> Qlt((device half*)ql, dextents<int32_t, 2>(p.d, tqp));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> Kt((device half*)k, dextents<int32_t, 2>(p.d, tkp));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> Klt((device half*)kl, dextents<int32_t, 2>(p.d, tkp));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> Vt((device half*)v, dextents<int32_t, 2>(p.d, tkp));
    tensor<threadgroup float, extents<int32_t, 64, 64>, tensor_inline> St(S, extents<int32_t, 64, 64>());
    tensor<threadgroup half, extents<int32_t, 64, 64>, tensor_inline> Pt(P, extents<int32_t, 64, 64>());
    constexpr auto ds = matmul2d_descriptor(64, 64, 128, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto dov = matmul2d_descriptor(64, 128, 64, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<ds, execution_simdgroups<4>> ops;
    matmul2d<dov, execution_simdgroups<4>> opo;
    auto qa = Qt.slice<128, 64>(h0, q0);
    auto qla = Qlt.slice<128, 64>(h0, q0);
    auto vb0 = Vt.slice<128, 64>(h0, 0);
    auto co = opo.get_destination_cooperative_tensor<decltype(Pt), decltype(vb0), float>();
    zero(co);
    uint r = tid / 2, part = tid % 2;
    float m = -INFINITY, l = 0.0f;
    if (tid == 0) rescale = 0;
    for (uint kb = 0; kb < p.tk; kb += 64) {
        auto kt = Kt.slice<128, 64>(h0, kb);
        auto klt = Klt.slice<128, 64>(h0, kb);
        auto cs = ops.get_destination_cooperative_tensor<decltype(qa), decltype(kt), float>();
        zero(cs);
        ops.run(qla, kt, cs);
        ops.run(qa, klt, cs);
        ops.run(qa, kt, cs);
        cs.store(St);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float x[32];
        float bm = -INFINITY;
        for (uint j = 0; j < 32; j++) {
            uint col = part * 32 + j;
            x[j] = kb + col < p.tk ? S[r * 64 + col] * p.scale : -INFINITY;
            bm = max(bm, x[j]);
        }
        bm = max(bm, simd_shuffle_xor(bm, 1));
        // Lazy rescaling, as flash_attn128_body: the running max moves only past +8.
        bool grow = bm > m + 8.0f;
        float mn = grow ? bm : m;
        float alpha = grow ? exp(m - mn) : 1.0f;
        if (grow) rescale = 1;
        float ps = 0.0f;
        for (uint j = 0; j < 32; j++) { float e = exp(x[j] - mn); P[r * 64 + part * 32 + j] = half(e); ps += e; }
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
        auto vt = Vt.slice<128, 64>(h0, kb);
        opo.run(Pt, vt, co);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) rescale = 0;
    }
    if (part == 0) L[r] = 1.0f / l;
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint16_t i = 0; i < co.get_capacity(); ++i)
        if (co.is_valid_element(i)) {
            auto idx = co.get_multidimensional_index(i);
            uint row = q0 + idx[1];
            if (row < p.tq) o[(ulong)row * p.d + h0 + idx[0]] = co[i] * L[idx[1]];
        }
}
#endif
