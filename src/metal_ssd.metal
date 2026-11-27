// ---- Mamba-2's chunked scan for prefill plans (SSD; metal_decode.rs `MOp::Mamba2::ssd`) ----------
// The recurrence of `dec_mamba2_t` (S_t = exp(dt_t a) S_{t-1} + dt_t x_t B_t^T, y_t = S_t C_t) over
// chunks of SSD_L tokens as products of 8x8 SIMD-group matrices (Mamba-2's "state space duality"):
// with cs_t the sum of dt_u a over the chunk's tokens u <= t,
//   y_t = sum_{s <= t} (C_t . B_s) exp(cs_t - cs_s) dt_s x_s   (within the chunk)
//       + exp(cs_t) S_prev C_t                                 (the state before the chunk)
//   S_end = exp(cs_last) S_prev + sum_s exp(cs_last - cs_s) dt_s x_s B_s^T.
// `pf_ssd_chunk` (grid: chunks, heads) writes each chunk's first term into `out` and its own part
// of the state (the sum) into `ds`; `pf_ssd_state` (grid: heads, blocks of SSD_RB state rows)
// walks the chunks in order, adds the second term, the D skip and the gate, and carries the state.
// Head width 64, state width a multiple of 64 up to 256. Rounding differs from the token-by-token
// kernels (sums in another order).
constant constexpr uint SSD_L = 64;
constant constexpr uint SSD_RB = 16;

// The chunk's dt (0 past the step's tokens) and the running sum of dt a, in `dts` / `cs`.
inline void ssd_dt(device const float* dtr, constant M2P& p, uint t0, uint n, uint sz, uint h, float a, float db,
                   threadgroup float* dts, threadgroup float* cs, uint tid) {
    if (tid < SSD_L) {
        float dt = tid < n ? mamba2_dt(dtr[(ulong)(t0 + tid) * sz + h], db, p) : 0.f;
        dts[tid] = dt;
        cs[tid] = dt * a;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        float v0 = simd_prefix_inclusive_sum(cs[tid]);
        float v1 = simd_prefix_inclusive_sum(cs[32 + tid]) + simd_broadcast(v0, 31);
        cs[tid] = v0;
        cs[32 + tid] = v1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ts: tokens, the strides of xbc, of z / dt and of out (floats). 128 threads.
kernel void pf_ssd_chunk(device const float* xbc [[buffer(0)]], device const float* dtr [[buffer(1)]],
                         device float* out [[buffer(2)]], device float* ds [[buffer(3)]],
                         device const float* cst [[buffer(4)]], constant M2P& p [[buffer(5)]],
                         constant uint4& ts [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float m[SSD_L * SSD_L];
    threadgroup float dts[SSD_L], cs[SSD_L];
    uint c = tg.x, h = tg.y, t0 = c * SSD_L, n = min(SSD_L, ts.x - t0);
    uint g = h / (p.nh / p.ng), inner = p.nh * p.hd, sx = ts.y;
    ssd_dt(dtr, p, t0, n, ts.z, h, cst[h], cst[2 * p.nh + h], dts, cs, tid);
    device const float* xr = xbc + (ulong)t0 * sx;
    device const float* X = xr + h * 64;
    device const float* B = xr + inner + g * p.ns;
    device const float* C = xr + inner + (p.ng + g) * p.ns;
    // G = C B^T, tile rows 2 sg and 2 sg + 1.
    simdgroup_float8x8 acc[2][8];
    for (uint i = 0; i < 2; i++)
        for (uint j = 0; j < 8; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    for (uint k = 0; k < p.ns; k += 8) {
        simdgroup_float8x8 a0, a1, b;
        simdgroup_load(a0, C + (ulong)(16 * sg) * sx + k, sx);
        simdgroup_load(a1, C + (ulong)(16 * sg + 8) * sx + k, sx);
        for (uint j = 0; j < 8; j++) {
            simdgroup_load(b, B + (ulong)(8 * j) * sx + k, sx, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc[0][j], a0, b, acc[0][j]);
            simdgroup_multiply_accumulate(acc[1][j], a1, b, acc[1][j]);
        }
    }
    for (uint i = 0; i < 2; i++)
        for (uint j = 0; j < 8; j++) simdgroup_store(acc[i][j], m + (16 * sg + 8 * i) * SSD_L + 8 * j, SSD_L);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // The decays and dt: M[t][s] = G[t][s] exp(cs_t - cs_s) dt_s for s <= t.
    for (uint e = tid; e < SSD_L * SSD_L; e += 128) {
        uint t = e / SSD_L, s = e % SSD_L;
        m[e] = s <= t && s < n ? m[e] * exp(cs[t] - cs[s]) * dts[s] : 0.f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Y = M X (64 tokens x 64 rows of the head) into `out`.
    for (uint i = 0; i < 2; i++)
        for (uint j = 0; j < 8; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    for (uint k = 0; k < SSD_L; k += 8) {
        simdgroup_float8x8 a0, a1, b;
        simdgroup_load(a0, m + (16 * sg) * SSD_L + k, SSD_L);
        simdgroup_load(a1, m + (16 * sg + 8) * SSD_L + k, SSD_L);
        for (uint j = 0; j < 8; j++) {
            simdgroup_load(b, X + (ulong)k * sx + 8 * j, sx);
            simdgroup_multiply_accumulate(acc[0][j], a0, b, acc[0][j]);
            simdgroup_multiply_accumulate(acc[1][j], a1, b, acc[1][j]);
        }
    }
    device float* o = out + (ulong)t0 * ts.w + h * 64;
    for (uint i = 0; i < 2; i++)
        for (uint j = 0; j < 8; j++) simdgroup_store(acc[i][j], o + (ulong)(16 * sg + 8 * i) * ts.w + 8 * j, ts.w);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // (X w)^T, w_s = exp(cs_last - cs_s) dt_s, over M's memory: row p, column s.
    float last = cs[SSD_L - 1];
    for (uint e = tid; e < 64 * SSD_L; e += 128) {
        uint r = e / SSD_L, s = e % SSD_L;
        m[e] = s < n ? X[(ulong)s * sx + r] * (exp(last - cs[s]) * dts[s]) : 0.f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // The chunk's part of the state: (X w)^T B, rows 2 sg and 2 sg + 1 of tiles, 64 columns a pass.
    device float* d = ds + ((ulong)c * p.nh + h) * 64 * p.ns;
    for (uint n0 = 0; n0 < p.ns; n0 += 64) {
        for (uint i = 0; i < 2; i++)
            for (uint j = 0; j < 8; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
        for (uint k = 0; k < SSD_L; k += 8) {
            simdgroup_float8x8 a0, a1, b;
            simdgroup_load(a0, m + (16 * sg) * SSD_L + k, SSD_L);
            simdgroup_load(a1, m + (16 * sg + 8) * SSD_L + k, SSD_L);
            for (uint j = 0; j < 8; j++) {
                simdgroup_load(b, B + (ulong)k * sx + n0 + 8 * j, sx);
                simdgroup_multiply_accumulate(acc[0][j], a0, b, acc[0][j]);
                simdgroup_multiply_accumulate(acc[1][j], a1, b, acc[1][j]);
            }
        }
        for (uint i = 0; i < 2; i++)
            for (uint j = 0; j < 8; j++) simdgroup_store(acc[i][j], d + (ulong)(16 * sg + 8 * i) * p.ns + n0 + 8 * j, p.ns);
    }
}

// The chunks in order for SSD_RB rows of a head's state (in threadgroup memory): the state's
// term, the D skip and the gate into `out` (which holds the chunk's own term), then the state
// carried over. 128 threads; `st` the state, read and written; `dtr` the raw dt (stride ts.z).
kernel void pf_ssd_state(device const float* xbc [[buffer(0)]], device const float* z [[buffer(1)]],
                         device float* out [[buffer(2)]], device const float* ds [[buffer(3)]],
                         device const float* cst [[buffer(4)]], constant M2P& p [[buffer(5)]],
                         constant uint4& ts [[buffer(6)]], device float* st [[buffer(7)]],
                         device const float* dtr [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sm[SSD_RB * 256];
    threadgroup float yt[SSD_L * SSD_RB];
    threadgroup float dts[SSD_L], cs[SSD_L];
    uint h = tg.x, r0 = tg.y * SSD_RB;
    uint g = h / (p.nh / p.ng), inner = p.nh * p.hd, sx = ts.y, ns = p.ns;
    float a = cst[h], dd = cst[p.nh + h], db = cst[2 * p.nh + h];
    device float* sp = st + ((ulong)h * p.hd + r0) * ns;
    for (uint e = tid; e < SSD_RB * ns; e += 128) sm[e] = sp[e];
    uint nc = (ts.x + SSD_L - 1) / SSD_L;
    for (uint c = 0; c < nc; c++) {
        uint t0 = c * SSD_L, n = min(SSD_L, ts.x - t0);
        ssd_dt(dtr, p, t0, n, ts.z, h, a, db, dts, cs, tid);
        // Y (64 tokens x SSD_RB rows) = C S^T: tile rows 2 sg and 2 sg + 1, both tile columns.
        device const float* C = xbc + (ulong)t0 * sx + inner + (p.ng + g) * ns;
        simdgroup_float8x8 acc[2][2];
        for (uint i = 0; i < 2; i++)
            for (uint j = 0; j < 2; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
        for (uint k = 0; k < ns; k += 8) {
            simdgroup_float8x8 a0, a1, b;
            simdgroup_load(a0, C + (ulong)(16 * sg) * sx + k, sx);
            simdgroup_load(a1, C + (ulong)(16 * sg + 8) * sx + k, sx);
            for (uint j = 0; j < 2; j++) {
                simdgroup_load(b, sm + 8 * j * ns + k, ns, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc[0][j], a0, b, acc[0][j]);
                simdgroup_multiply_accumulate(acc[1][j], a1, b, acc[1][j]);
            }
        }
        for (uint i = 0; i < 2; i++)
            for (uint j = 0; j < 2; j++) simdgroup_store(acc[i][j], yt + (16 * sg + 8 * i) * SSD_RB + 8 * j, SSD_RB);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < n * SSD_RB; e += 128) {
            uint t = e / SSD_RB, r = h * p.hd + r0 + e % SSD_RB;
            ulong row = (ulong)(t0 + t);
            float y = out[row * ts.w + r] + exp(cs[t]) * yt[e];
            out[row * ts.w + r] = mamba2_out(y, dd, xbc[row * sx + r], z[row * ts.z + r]);
        }
        float decay = exp(cs[SSD_L - 1]);
        device const float* d = ds + (((ulong)c * p.nh + h) * p.hd + r0) * ns;
        for (uint e = tid; e < SSD_RB * ns; e += 128) sm[e] = fma(decay, sm[e], d[e]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint e = tid; e < SSD_RB * ns; e += 128) sp[e] = sm[e];
}
