# llama-modchip: CPU runtime, scheduler and operator changes (dirty-room analysis)

Scope: `ggml/src/ggml-cpu/ggml-cpu.c`, `ggml/src/ggml-cpu/ops.cpp` and `ggml/src/ggml.c` in
`patches/deepseek-v4-flash-fractal-256k.patch` (called **DSV4** below), plus the matching hunks of
`patches/zen5-nemotron-256k-optimizations.patch` (called **Z5**). The model-side glue that drives
these hooks (llama-graph / kv-cache / deepseek4.cpp / llama-mmap / llama-model) gets brief notes too.
Quant block formats and dot kernels are covered by someone else. Here they are treated as black boxes:
"1xN multi-column dot = one weight row against N activation columns".

This document describes behaviour only. It contains no source text. Line numbers are positions in the
patch files.

Notes:
- `dsv4-64gb-cpu.patch` and `llama-modchip-dsv4-10tps.patch` are **byte-identical** copies of the DSV4
  patch (same MD5). There are really only two patches.
- Z5 contains a subset of the DSV4 runtime changes (dual-CCD, profiling, blck_1 cap, FA prefetch/SWA,
  THP). It adds topology discovery and safer shm/affinity handling, and has none of the Q8 cache,
  MoE pruning, SwiGLU skipping, MQA8 FA, HC or FWHT work.

---

## 0. Summary table

| # | Feature (marketing name) | Where | Default | Class | Real? |
|---|---|---|---|---|---|
| 1 | "4-level Register+L1D+L2+L3 tiling" (prefill) | mul_mat / mul_mat_id one_chunk | on, for 6 quant types only | numerically equivalent (depends on the 1xN kernels) | **2 levels are real** (128-row weight tile × 8/4/3/2 columns). There is **no L3 level** |
| 2 | Prefill static row partition + 16-row alignment | mul_mat / mul_mat_id chunking | on | lossless | real |
| 3 | Prefill MMID work-stealing (expert × row-slice tasks) | mul_mat_id | on when ≥4 tokens and ≥256 rows | lossless | real |
| 4 | 8-slot Q8 activation cache | graph loop + mul_mat(_id) | on | lossless (has a correctness hazard, see §4) | real |
| 5 | MMID row-group (ids) reuse | graph loop + mul_mat_id | on | lossless | real |
| 6 | Barrier elision | mul_mat, mul_mat_id, flash-attn | on | lossless | real, narrow conditions |
| 7 | Row-parallel from_float | mul_mat(_id) | on | lossless | real |
| 8 | L2-quadrature MoE tail pruning | graph loop, SCALE hook | **off** (cfg file) | **LOSSY** | real, gated by a /tmp mmap file |
| 9 | Gate-first SwiGLU row skipping | graph loop + one_chunk | **off** (cfg file) | **LOSSY** | real, gated by the same file |
| 10 | mmap runtime control (`/tmp/dsv4_moe_cfg.bin`) | graph loop | file auto-created | n/a (control plane) | real. **Security and footgun problems** |
| 11 | CSA/LID compressor skip | graph loop + llama-graph + deepseek4 | on | lossless (it drops dummy writes) | real |
| 12 | LID top-k bypass when ≤ top_k candidates | deepseek4.cpp | on | exact | real |
| 13 | MQA8 register-stationary MLA flash attention | ops.cpp FA | on (AVX-512+F16C, DSV4 MLA shape) | numerically different (more precise) | real |
| 14 | FA chunking / split-KV changes | ops.cpp FA | on | lossless | real |
| 15 | FA sparse-mask lookahead + NTA prefetch | ops.cpp FA | on (AVX2/512) | lossless (hints only) | real |
| 16 | FA SWA / interleaved-SWA / sinks via env or shm | ops.cpp FA | **off** | **LOSSY** | real, buggy |
| 17 | In-register FWHT | ops.cpp fwht | on (AVX-512) | numerically different (scale applied first) | 4 stages in-register, rest in memory. **AVX-512 only, not AVX2** |
| 18 | Vectorized HC_PRE / HC_POST (hc=4) | ops.cpp | on | equivalent | real, but just a scalar loop the compiler can auto-vectorize |
| 19 | HC_COMB single-thread for <4 tokens | n_tasks | on | lossless | real |
| 20 | RMS_NORM AVX-512 + flattened row parallelism | ops.cpp | on | numerically different (summation order) | real |
| 21 | Concat contiguous fast path | ops.cpp | on | lossless | real |
| 22 | top_k uses nth_element for k≥32 | ops.cpp | on | exact as a set (order unspecified) | real |
| 23 | Lightning-indexer key-stationary loop + skipping masked keys | ops.cpp | on | exact | real |
| 24 | Dual-CCD "vertical pipeline" | ggml_graph_compute (OpenMP only) | **off** (`LLAMA_DUAL_CCD_SPLIT`) | lossless | real but **sequential: no pipelining**. Drops the other optimizations |
| 25 | Thread pinning | OpenMP path | on for n_threads 16/32 (DSV4) | n/a | real, Linux-only API |
| 26 | L3/CCD topology discovery | Z5 only | used only by #24 | n/a | real, assumes 2 L3 domains |
| 27 | THP for weights and compute buffers | ggml.c, llama-mmap.cpp | on (Linux) | n/a | real. 2 MB alignment only in Z5 |
| 28 | LLAMA_PROFILE_OPS | graph loop | off | n/a | real |
| 29 | `LLAMA_MOE_TOP_K` (global n_expert_used override) | llama-model.cpp | off | **LOSSY** | real |

---

## 1. Matmul prefill tiling ("4-level Register + L1D + L2 + L3")

**Hooks.** `mul_mat_one_chunk` (DSV4 4400–4486) and `mul_mat_id_one_chunk` (4638–4722). A type→kernel
lookup (4309–4361) returns four optional function pointers, "1x8", "1x4", "1x3" and "1x2" (one weight
row against 8/4/3/2 activation columns, each writing that many float results). They are populated only
for **Q2_0, IQ1_S, IQ2_XXS, IQ1_M, Q5_K, Q6_K**. Every other type falls through to the stock upstream
16×16 blocking loop.

**Activation conditions (dense mul_mat).** The weight type has a 1x4 kernel, the chunk covers more
than one src1 column, the vec_dot is single-row (no mmla 2-row kernels), and SwiGLU skipping is not
active for this chunk.

**Algorithm (dense).**
- Outer loop: the chunk's weight rows in tiles of **128 rows** (`blck_0_l2 = 128`). The comment's
  claim: 128 quantized rows is about 64–276 KB, so the tile stays in a 1 MB L2.
- Inside a tile, walk the chunk's columns left to right, taking a group of k columns at a time:
  - k = 8 if a 1x8 kernel exists and at least 8 columns remain, unless exactly 9 remain;
  - else k = 4 if at least 4 remain, unless exactly 5 remain;
  - else k = 3 if exactly 3 or exactly 5 remain;
  - else k = 2 if at least 2 remain;
  - else k = 1 (plain vec_dot).

  So 9 splits as 4+3+2 and 5 splits as 3+2. A 1x1 call only happens when a single column is left.
- A group must not straddle two different src0 matrices (broadcast/batched 3D weights, e.g. `wo_a`).
  The column range is cut into segments of width `ne11 * r2` (columns per distinct src0 matrix, where
  r2 is the broadcast ratio ne12/ne02). The src0 base pointer is computed once per group from the
  group's first column.
- For each row in the tile, call the 1xk kernel with the k column pointers (Q8 activations) and
  scatter the k results to the k destination columns at that row index.
- Loop order: **rows-tile → column-groups → rows-in-tile**. The weight tile is re-streamed from L2
  for each column group. The k activation columns (k × Q8 row size, about 5.7–22 KB claimed) stay in
  L1 across the 128 rows.

**MMID variant.** Identical, except all columns of a chunk belong to the same expert (`cur_a`), so
there are no segments. Columns are the token rows mapped to that expert, taken from the
row-mapping table. The destination is (expert slot, token).

**Small upstream tweak (also in Z5).** The upstream 16-column block size `blck_1` is capped at
`ne11*r2` when that is below 16 (DSV4 4377–4380, Z5 ~215). Purely a locality change for 3D weights.

**Assessment.**
- Real register blocking (one weight-block unpack shared by k columns) plus L2 row tiling: 2 levels.
- **There is no L3-level code anywhere.** "L3 (64 MB)" exists only in the comments.
- Decode (1 column) never takes this path. All gains are prefill / multi-token MMID.
- Exactness: same math as k separate vec_dots. Bit-exactness depends on whether the 1xN kernels keep
  the same per-block accumulation order as the 1x1 kernel (other analyst).
- The claimed 1.7× prefill gain is plausible for the IQ1/Q2 types, where weight unpacking dominates.

## 2. Prefill chunk scheduling

**Dense mul_mat** (4609–4616). New rule placed before the upstream "too few chunks" rule: if there is
more than one column and `nr0 ≥ 8·nth`:
- 2D weight: nchunk0 = nth, nchunk1 = 1. Each thread takes one contiguous row slice across all
  columns, which is what makes the 128-row L2 tiling reuse weights across all N tokens.
- Batched weight with ne1 > 1: nchunk0 = nth, nchunk1 = ne2·ne3. One chunk per (row slice × batch
  matrix), consumed through the atomic chunk counter.

**MMID per-expert path** (4902–4911). If `nr0 ≥ 8·nth`, force nchunk0 = nth, nchunk1 = 1. When
nchunk0 = nth and `nr0 ≥ 16·nth`, round the row slice up to a multiple of 16 rows: 16 floats = one 64 B
output line, which avoids false sharing on dst. Skip empty slices.

**MMID batched path** (4855–4881). Taken when ≥4 tokens and ≥256 rows per expert (`ne01`):
- Task space = n_experts × n_sub, with n_sub = 8 if ne01 ≥ 2048, else 4. Slice height = ceil(ne01/n_sub)
  rounded up to 16.
- Each thread starts at task = ith, then pulls tasks from a single shared atomic counter (the
  per-expert counter slot 0, reset to nth before the barrier).
- A task processes all tokens routed to that expert over its row slice, using the §1 tiled kernel.
- Experts with zero tokens still cost an atomic increment.
- Lossless. Better load balance than upstream's sequential per-expert loop, which needs a separate
  chunk distribution per expert.

## 3. Row-parallel activation quantization (from_float)

mul_mat 4545–4580, mul_mat_id 4775–4810. When the total number of src1 rows is at least nth, thread
ith quantizes whole rows ith, ith+nth, … Otherwise it uses upstream's split of each row into nth
block ranges (the decode case). Lossless: Q8 quantization is per-block. This is the doc's "row-parallel
from_float / token-to-core affinity".

## 4. 8-slot Q8 activation cache

**Problem.** In decode, the same activation vector feeds several matmuls (q/kv projections, gate+up of
the shared expert, gate+up MMIDs, router). Upstream requantizes src1 to the vec_dot type (Q8_0 / Q8_K /
F16) for every matmul, followed by a barrier.

**Data structures** (4228–4236, 5040–5048).
- A global static buffer of **8 slots × 64 KiB**, 64 B aligned, shared by all threads of the process.
- Per worker thread, a local 8-entry table built at the start of each graph run. Each entry holds:
  the src1 data address, the base-tensor address range (view_src or itself), the vec_dot type, the
  4 dims, and strides 1..3.
- Thread-local flags passed to the op: "wdata override pointer" and "already valid".
- All threads build identical tables deterministically, so no sharing is needed.

**Lookup** (graph loop 5157–5195). Runs for MUL_MAT / MUL_MAT_ID when: not the reference path, the
matmul is not tagged as a Hadamard-src0 matmul, src1 is F32 and differs from the vec_dot type, and the
quantized size of all of src1 is ≤ 64 KiB (in practice decode only).
- Hit (same address, type, dims, strides): point the op at that slot and mark it valid. The op then
  skips quantization.
- Miss: claim slot `next++ mod 8` (round robin), record the key, point the op at the slot, and mark
  it not valid. The op quantizes into the slot.

**Invalidation.** After every executed node (including each node of a fused group), compute the
node's base-tensor byte range. Drop every slot whose recorded src1 base range overlaps it. This
handles in-place writes, views, and allocator buffer reuse within the graph. The table is rebuilt
empty for each graph run.

**In the op.** Quantize only if not valid. Barrier elision (§6).

**Class.** Lossless when it works.

**Bugs / risks.**
1. **Slot claimed but never filled.** A slot is reserved on a miss before the op runs. If the op then
   never reaches the quantization code, the next matmul with the same src1 gets a "hit" on a stale or
   garbage buffer. This happens when:
   - an extra buffer type (CPU repack, AMX, KleidiAI) handles the node first in
     `ggml_compute_forward`; or
   - the llamafile sgemm path handles it at the first attempt, which happens for F16 weights with
     F32 B on ARM-fp16 builds.

   Models that mix repacked and non-repacked weights reading the same activation would compute
   garbage. The DSV4 custom types are probably not repacked, so it may never trigger on their target.
2. The static global buffer is shared by all threadpools and contexts in the process. Two concurrent
   graph computes (draft and target model on different threadpools, or two contexts) race.
3. The dual-CCD path (§24) does not use the cache at all.

## 5. MMID row-group (expert routing table) reuse

Graph loop 5196–5214, op 4815–4832. Upstream rebuilds the expert → token-rows mapping (counts and
row list in `params->wdata`, after the Q8 area) on thread 0 for every MUL_MAT_ID, then hits a barrier.

DSV4 remembers the last MMID's ids tensor: address, ne0, ne1, number of experts, and the quantized-src1
byte offset (which locates the table in wdata). If the next MMID matches all of these, the "ids valid"
flag is set and thread 0 skips the rebuild.

The memory is forgotten after any node that is not a mul_mat / MMID, after any MUL_MAT that uses
`params->wdata` directly (no Q8-cache pointer), after a fused group, or when a written range overlaps
the ids base tensor.

In practice this reuses the table from gate → up of the same layer (down has a different src1 size and
therefore a different offset). Lossless.

The table is built with the MoE skip mask in effect (§8). The mask is constant between gate and up, so
reuse stays consistent.

## 6. Barrier elision

- **mul_mat** (4587–4596): skip the post-quantization barrier and the chunk-counter reset when
  (a) src1 came from the Q8 cache (so it was written by an earlier node, already behind that node's
  end barrier), (b) dst is a single column, and (c) ceil(ne0/64) < 4·nth. Under (c), the upstream rule
  rechunks to exactly nth static chunks, so the atomic counter is never read. With nth = 16 this means
  ne0 ≤ 4096, which matches the doc's "nr0 < 4096". Safe.
- **mul_mat_id** (4835–4843): reset the per-expert counters only if there is more than one token or
  ceil(ne01/64) ≥ 4·nth. Barrier only if Q8 was not cached, the ids were not reused, there is more than
  one token, or the same row test fails. In decode with both caches hit there is no barrier. Safe for
  the same static-chunk reason.
- **Flash attention** (6101–6178): the chunk-counter barrier is removed whenever the static
  nchunk = nth layout is used (all decode). It is kept only for the dynamic MLA-prefill chunking.
- The SCALE hook (§8) **adds** one barrier after every SCALE node shaped [1,6,…].

## 7. The OpenMP barrier change

`ggml_barrier` (4205–4220) uses the custom spin barrier instead of the OpenMP barrier when
`dual_ccd_active` is set. Twice as many OpenMP threads exist then, and only half participate, so the
OpenMP barrier would deadlock. The non-OpenMP path is unchanged. Also in Z5.

## 8. "L2-quadrature adaptive MoE tail pruning" (LOSSY, off by default)

**Hook** (5292–5357). After the barrier that ends any **GGML_OP_SCALE** node with `ne0 = 1` and
`ne1 = 6`. This shape-matches the DSV4 routed-expert weight tensor: 6 experts per token, after
routed-scale. Thread 0 does the work, then an extra barrier follows.

**Criterion.** Only if the node has `ne2 == 1` (single-token ubatch, so decode only). Steps:
1. Parse the layer index from the decimal suffix after the last '-' in the node name. If there is no
   dash, the layer is -1.
2. Read the config (§10):

   | slot | meaning | default |
   |---|---|---|
   | cfg0 | top-k outside the band | `DSV4_MOE_TOPK` or 6 |
   | cfg1 | top-k inside the band | same |
   | cfg2 | min weight × 10⁴ | `DSV4_MOE_MIN_W` or 0 |
   | cfg3 / cfg4 | band [start, end) | 3, 43 |
   | cfg5 | rescale mode | 1 |
   | cfg6 / cfg7 | SwiGLU keep for MMID / dense | 0, 0 |

3. k = cfg1 if the layer is inside the band, else cfg0, clamped to ≥1.
4. Prune only if **layer ≥ 3** (hard-coded) and (k < 6 or min_w > 0).
5. Sort the 6 slot weights in descending order (6-element selection sort). Drop the slot at rank r if
   r ≥ k, or if r > 0 and its weight < min_w. The top-1 is never dropped. Dropped weights are set to 0
   in place and a 6-bit "skip mask" over slot indices is built.
6. Rescale every surviving weight by a factor depending on the mode:
   - mode 1 (default), "L2 quadrature": sqrt(Σ all w² / Σ kept w²);
   - mode 2: Σ all w / Σ kept w;
   - other values: no rescale.
7. Publish the skip mask in a process-static atomic. Otherwise publish 0.

**Effect on compute.** Each MMID node loads the current mask (5197). When there is exactly one token:
- the routing-table build ignores skipped slots, so those experts' rows are never computed;
- thread 0 zeroes the skipped slots' dst rows (4846–4852).

Gate, up and down all skip, saving bandwidth for (6−k)/6 of the routed-expert weights.

**Class.** Lossy: it drops experts and renormalizes the rest. Mode 1 inflates the remaining weights,
so the output norm is preserved only in expectation, assuming orthogonal expert outputs.

**Default.** No pruning (k = 6, min_w = 0), unless the cfg file already holds other values. Docs
recommend [4,4,0,3,43,1,…] (top-4), quoting about 10.1 tok/s vs 9.06 lossless, i.e. about 12%.

**Risks.**
- The mask stays in force until the next matching SCALE node. Correctness relies on the SCALE (expert
  weights) node being ordered **before** that layer's gate/up MMIDs in the graph. Upstream's
  build_moe_ffn expands `weights` early for its topk-moe fusion, which probably guarantees this. If the
  ordering changed, layer L's MMIDs would use layer L−1's mask and skip **non-zero-weight** experts.
- It only triggers for exactly 6 experts. Setting `LLAMA_MOE_TOP_K` silently disables it.
- Any other model with a SCALE node of shape [1,6,1] (6 active experts) gets pruned whenever the cfg
  file says so.

## 9. "Gate-first SwiGLU" row skipping (LOSSY, off by default)

**Setup** (graph loop 5225–5265). For MUL_MAT / MMID nodes with ne2 = ne3 = 1 and a cfg mapping
present:
- keep = cfg6 for MMID or cfg7 for dense, in units of 1/128;
- the layer comes from the node-name suffix, defaulting to 3 if absent;
- requires 0 < keep < 128 and the layer inside [cfg3, cfg4);
- the node must be the **up** projection: the next node is a SWIGLU GLU whose second operand is this
  node, or the next node is a CLAMP of this node followed by such a GLU;
- the GLU's first operand (the gate) must be F32, same shape, already allocated.

If all hold, the gate output pointer, its row stride and `keep` are put in thread-locals.

It relies on graph order putting the gate matmul before the up matmul (true for upstream DFS:
GLU src0 = gate is visited first). Nothing checks that the gate was actually computed already.

**Per chunk** (mask function 4239–4304, used in 4390–4399 / 4627–4637 and the inner loops
4500–4509 / 4724–4733). Requirements: a single column (decode), ≤128 rows in the chunk, single-row
vec_dot. For MMID the gate row is the one for the expert slot of the chunk's first mapped row.
1. For the chunk's n rows compute s_j = SiLU(g_j)².
2. k_keep = floor(keep·n/128). If k_keep ≤ 0 or k_keep ≥ n, keep everything.
3. Quickselect (Hoare partition, middle pivot) on a copy to find the value t at ascending position
   n − k_keep. Keep rows with s > t, then fill ties (s ≥ t) in index order until exactly k_keep rows
   are kept.
4. α = sqrt(Σ s over the chunk / Σ s over kept rows), when the kept sum is > 1e-20 and smaller than
   the total. Otherwise α = 1.
5. In the matmul, rows not kept get output 0 and no dot product (their weights are never read). Kept
   rows get dot × α.

**Effect.** Only the **up** projection is skipped (down and gate are unchanged). The zeroed up value
makes h_j = 0. Savings are (128 − keep)/128 of the up-proj weight bytes, i.e. ≤ 1/3 of the FFN.

**Class.** Lossy. α ≥ 1 inflates the kept terms (heuristic norm preservation).

**Limits.**
- Engages only when each thread's row chunk is ≤ 128 rows: decode with nchunk0 = nth, e.g. 2048 rows
  over 16 threads gives 128.
- With fewer threads (larger chunks) it silently does nothing.
- The selection is per thread-chunk ("stratified"), not global.
- Docs claim 0.98–0.996 cosine vs exact for 50–75% keep.

## 10. Runtime control file ("mmap runtime control")

Graph loop 5049–5112. On the first graph, thread 0:
- reads `DSV4_MOE_TOPK` and `DSV4_MOE_MIN_W`;
- opens `DSV4_MOE_CFG_PATH`, else `/tmp/dsv4_moe_cfg.bin` (Windows: %TEMP%), with create, no-follow,
  **mode 0666 plus an explicit fchmod to 0666**;
- writes the 8 int32 defaults only if the file is shorter than 32 bytes;
- maps 32 bytes MAP_SHARED read/write. If the open fails, it falls back to a read-only map.

Every subsequent SCALE hook and matmul reads the slots live, so external writes take effect on the
next token with no restart.

**Problems.**
- **Security.** The file is world-writable in a shared /tmp. Any local user can pre-create it or edit
  it to turn on lossy pruning/skipping in someone else's inference. O_NOFOLLOW blocks symlink attacks
  but not a pre-created regular file.
- **Stale state.** The file persists across runs. A previous experiment's top-4 setting silently
  applies to the next run. Env vars only seed a *new* file and are ignored once it exists, because the
  mmap values take precedence.
- **Startup race.** Other threads read the mapping pointer without synchronization. On the very first
  graph, threads can disagree about whether SwiGLU skipping is active. This is a transient
  inconsistency, not a crash.
- Threads read cfg slots independently per node, so a write in mid-token can be seen by some threads
  and not others.
- Z5 has no such file. Its FA shm config (§16) is the only runtime file there.

## 11. CSA / LID compressor skip

- KV-cache plan (DSV4 7024–7033): for tokens that don't complete a compression window, the dummy state
  write position is now -1 instead of 0. The write still targets scratch slot kv_size − 1.
- llama-graph (7016–7020): when setting inputs for the "csa" plan, a process-global int is set to 1
  iff the plan has exactly one write position and it is -1, i.e. a single-token step with no real
  compressed row produced.
- deepseek4.cpp (7380–7405, 7470–7493): before building the CSA (and likewise LID) state-compression
  subgraph, all its true inputs are force-expanded into the graph first. The node count is recorded,
  the compression subgraph and its cpy into the cache are built, and every node added since is tagged
  with tensor flag bit **0x100**.
- Graph loop (5117–5124): if the global is set, any node with flag 0x100 is treated as a no-op. The
  graph is not rebuilt, hence "zero-rebuild".

**Class.** Lossless: the skipped work only produced a write into the dummy slot.

**Risks.**
- Process-global flag, so multiple contexts race.
- The decision is based on the CSA plan but also gates LID nodes (both use the CSA ratio, so it is
  probably fine).
- The -1 position does flow into the compression subgraph when it is *not* skipped (multi-token
  batches containing such tokens). That is harmless only if it is used purely as a dummy rope/position
  value and never as an index (not verified).
- The restore-index change (7040–7050) makes the graph shape depend on whether a rollback is pending,
  which slightly contradicts "zero-rebuild".

**Related exact change** (7314–7322): the LID top-k selection is skipped when the compressed-KV count
is ≤ indexer_top_k (top-k of everything is everything). The plain CSA mask is used instead.

## 12. MQA8 register-stationary MLA flash attention

ops.cpp 5831–6010.

**Conditions:** AVX-512F and F16C, no split-KV partials, K and V both F16, DK = DV = 512, one KV head,
neq3 = 1, no ALiBi, mask broadcast over heads. This is exactly the DSV4 MLA latent attention
(64 query heads sharing one 512-d latent KV).

**Algorithm.**
- Group up to 8 consecutive query heads of the same query position (the group never crosses a
  position).
- Per group, keep FP32 state: a 512-float accumulator per head, a running max M and a running sum S.
- For each KV row:
  - skip it if its mask is exactly −inf (raw F16 pattern);
  - prefetch the next row (4 cache lines);
  - convert each 16-wide F16 slice of K to FP32 once and FMA it against the 8 FP32 query vectors,
    giving 8 dot products;
  - per head: scale, optional tanh softcap, add mask, online-softmax update;
  - if the max increased, the old accumulator is multiplied by exp(Mold − s) and the new row gets
    weight 1; otherwise the row gets weight exp(s − M);
  - convert V (here physically the same latent) slice by slice once and FMA it into all 8
    accumulators.
- Finish: fold in attention sinks (when present, KV range starting at 0), divide by S (0 if S = 0),
  write each head's output row.
- When fewer than 8 heads remain, a per-head loop with the same math is used.

**Class.** Numerically different from upstream but equivalent:
- Q stays FP32 for the K dot (upstream converts Q to F16 first);
- the V accumulation is FP32 (upstream accumulates in F16 for F16 V).

So it is arguably *more* accurate. Plausibly about 2× on FA (claimed 415 → 208 ms): K/V rows are read
once per 8 heads instead of once per head.

**Scheduling** (6088–6178).
- Split-KV (parallel over KV with partials) is now used only when the number of heads is below nth.
  With DSV4's 64 heads, parallelization is over heads.
- Chunking is disabled, i.e. static nth chunks with no counter and no barrier, except for MLA prefill.
  There, chunks are 4·nth-based, rounded up to a multiple of 8 rows so MQA8 groups stay whole, and
  pulled dynamically through the counter (barrier kept).
- The tiled FA path is excluded for MLA.
- Side effect: non-MLA prefill FA now always uses a static partition (upstream: dynamic unless NUMA).
  Z5 forces static too (Z5 ~787). This hurts causal load balance.

## 13. FA prefetch (generic path)

ops.cpp 6036–6085. AVX2/512 builds, in the generic one_chunk KV loop:
- **MLA** (detected heuristically as K and V sharing a data pointer, or K being a CONCAT): scan ahead
  to the next row not masked −inf, which helps the sparse top-k CSA/LID masks, and prefetch its
  4 lines (1 KB latent row, K = V).
- **Others:** prefetch row ic+4, K and V, 2 lines each.
- Hint selection in decode (N = 1):
  - MLA: NTA if the next row index is ≥ 512, else T0;
  - non-MLA: NTA if the row lies in a configured prefix (`LLAMA_KV_L3_NTA_PREFIX`), or is farther than
    `LLAMA_KV_L3_WINDOW` (default 2048) rows from the end;
  - otherwise T0.

Hints only, so lossless. The gain is plausible but small. Z5 has the same code.

## 14. FA sliding-window / interleaved SWA (LOSSY, off by default)

ops.cpp 6013–6027 plus the config struct 5728–5813.

**Config.** Read once from env: `LLAMA_ATTN_WINDOW` (default 0, off), `LLAMA_ATTN_SINKS` (16),
`LLAMA_SWA_STRIDE` (0). It is overridden live by a read-only mapping of `/dev/shm/llama_modchip.bin`
when that file exists with magic "MODC". Z5 prefers `$XDG_RUNTIME_DIR/llama_modchip.bin`, uses
O_NOFOLLOW, and requires a regular file owned by the current uid. DSV4 has none of these checks.

**Criterion.** Never applied to MLA (DSV4). For other models, when window W > 0 and the "current
position" exceeds W:
- if stride > 1, layers whose number parsed from the K tensor name (last decimal run) satisfies
  (layer + 1) mod stride = 0 stay global;
- otherwise KV rows with index in [sinks, cur_pos − W) are skipped.

**Bugs.**
- cur_pos is derived from the KV loop end (`ic_end − 1` for decode, `ic_end − N + iq1` for prefill),
  not from real positions or the mask. With a padded KV cache the window is offset by the padding.
- In the split-KV decode path each thread's chunk has its own ic_end, so every chunk keeps its own
  last-W rows. Not a real window.
- The tiled FA path (prefill) never applies SWA, so prefill and decode see different contexts.
- The shm fields `disable_chunking`, `rc_*` and `moe_top_k` are never read by this code (dead).

## 15. FWHT ("9-stage in-register AVX2 FWHT")

ops.cpp 6286–6320. **AVX-512 only. There is no AVX2 version**, contrary to the docs.

For row length n ≥ 16:
1. Per 16-float vector, multiply by the normalization scale **first**.
2. Do the 4 butterfly stages with stride 1, 2, 4, 8 inside the register. Each stage negates the
   "upper" lanes of each pair with a mask, adds the lane-permuted copy, and stores.
3. Do the remaining log2(n) − 4 stages as ordinary in-memory butterflies over 16-wide vectors.

For a 512-d row that is 4 stages in registers and 5 in memory, i.e. "9 stages" total but not all in
registers. Row-index bookkeeping is now incremental instead of divisions.

**Class.** Numerically different (scale applied before the transform instead of wherever upstream
applies it). Mathematically identical.

## 16. Hyper-connection ops (DSV4_HC_PRE / HC_POST / HC_COMB)

**HC_PRE** (6192–6214). Fast path for hc = 4 with contiguous rows: dst[i] = Σ_{h<4} w_h · x_h[i] with
the 4 weights per token hoisted. Parallelization: by token (token it → thread it mod nth) when tokens
≥ nth; otherwise each thread takes a slice of the embedding dimension for all tokens.

**HC_POST** (6222–6280). Same partitioning. The 4 post weights p_h and the 4×4 comb matrix c[h][g]
per token are hoisted. For each i: d_h[i] = x[i]·p_h + Σ_g r_g[i]·c[h][g], for all 4 output streams in
one pass (reads x and the 4 residual streams once).

**HC_COMB** task count (4937–4940): 1 thread when fewer than 4 tokens, else all threads. Previously it
shared the HC_PRE/POST case. The doc's "vectorized Sinkhorn-20" is just this thread-count change;
there is no new Sinkhorn code.

**Class.** Equivalent (plain FP32 arithmetic). The "AVX2 HC" in the docs is ordinary scalar code with
restrict pointers that the compiler may auto-vectorize. Gain comes from loop fusion and partitioning.

## 17. Other operator changes in ops.cpp

- **RMS_NORM** (5612–5680):
  - rows are flattened over dims 1–3 and distributed round-robin, so all threads are used even when
    ne01 < nth;
  - AVX-512 sum of squares: the square is computed in FP32, widened to double, and accumulated in two
    8-lane double vectors, then horizontally reduced (upstream accumulates scalar doubles of FP32
    products);
  - the fused multiply-by-weight is 16-wide.

  Numerically different only by summation order.
- **CONCAT** F16/F32 (5557–5610): when all innermost strides are contiguous, parallelize over flattened
  rows with memcpy (dim 0 concat copies both halves; other dims copy from whichever source covers the
  row). Lossless.
- **TOP_K** (5712–5720): for k ≥ 32 and k < n, use a selection algorithm instead of a partial sort. The
  result is the same set, in unspecified order. The upstream CPU op already treats order as
  unspecified.
- **Lightning indexer** (6343–6389): loop interchange to key-outer / token-inner. Each key is
  dequantized once and reused for all tokens (upstream dequantizes it per token). Keys masked −inf
  for a token are written as −inf without computing. Exact.
- **gated_delta_net** (6184): chunking is always disabled (static partition). Upstream disabled it only
  for NUMA. Also in Z5.

## 18. Dual-CCD "vertical layer partitioning" (off by default, OpenMP builds only)

DSV4 5408–5503, Z5 ~350–495.

**Trigger.** `LLAMA_DUAL_CCD_SPLIT = L > 0` and more than 1 thread. The split point is the node right
after the node named `l_out-(L−1)` or `nemotron_h_block_out-(L−1)`.

**Mechanism.**
1. Launch **2·n_threads** OpenMP threads and pin each one:
   - first half → CCD0 CPU list, second half → CCD1 list;
   - DSV4 hard-codes the 16C/32T Zen layout {0–7, 16–23} / {8–15, 24–31};
   - Z5 reads cpu0's L3 `shared_cpu_list` from sysfs as CCD0 and puts every other online CPU in CCD1,
     assuming exactly 2 L3 domains.
2. Group A runs nodes [0, split) with the normal per-node barrier, then raises phase 1.
3. Group B spins until phase 1, runs [split, end), then raises phase 2. Group A spins until phase 2.
4. If no split node is found, group A runs everything and group B spins.

**Assessment.**
- No overlap at all within a graph (one token's layers are strictly serial), so it is **not a
  pipeline**.
- The only possible benefit is that each CCD's L3 retains state for "its" layers across tokens, e.g.
  recurrent/SSM state for Nemotron. Weights of a large model don't fit anyway.
- n_threads idle spinners burn CPU.
- The range runner (4948–5006) is a stripped copy of the graph loop **without** the Q8 cache, MMID
  reuse, MoE pruning, SwiGLU, CSA skip or profiling.
- It uses pthread affinity APIs inside an OpenMP region, which only compile on Linux/glibc.

Gain: doubtful.

## 19. Thread affinity

In the standard OpenMP path, when the threadpool has no explicit cpumask:
- **DSV4** (5520–5531): if n_threads is 16 or 32, pin thread ith to logical CPU ith, overriding any
  external taskset.
- **Z5**: only if n_threads = 16, 32 CPUs are online, and the current mask already allows all 32,
  i.e. it respects taskset.

On Linux Zen, CPUs 0–15 are distinct physical cores, so 16 threads get one per core.

Linux-only API in an OpenMP block: breaks MSVC/OpenMP Windows builds and macOS+OpenMP builds. The
non-OpenMP ggml threadpool is unchanged. Gain is plausible for 16C parts (avoids SMT co-scheduling
and migration).

## 20. Transparent huge pages

- **DSV4 ggml.c** (6999–7006): after a ≥ 2 MB aligned allocation, madvise(HUGEPAGE) from the
  allocation start rounded down to 4 KiB. No 2 MB alignment, so THP coverage of the head and tail is
  partial.
- **Z5 ggml.c** (Z5 1174–1200): also raises the posix_memalign alignment to 2 MB for ≥ 2 MB
  allocations. This is the real "THP alignment".
- **llama-mmap.cpp** (both): madvise(HUGEPAGE) on the whole file mapping. For page-cache file mappings
  this does nothing unless the kernel has read-only file THP (khugepaged collapse) support.
- These blocks compile only if `MADV_HUGEPAGE` is visible in ggml.c. The hunk adds no `sys/mman.h`
  include there, so if upstream ggml.c doesn't already pull it in, the code silently compiles out
  (not verifiable from the patch alone).
- Lossless. The gain (TLB) is plausible for the compute arenas and uncertain for mmapped weights.

## 21. LLAMA_PROFILE_OPS

DSV4 5015–5032 and 5358–5398. Z5 has the same idea with stack arrays and no gap reporting.

**Enable:** env value G > 0. Only thread 0 profiles, and only graphs with more than 500 nodes, which
skips small or auxiliary graphs.

**Per node:** wall time from before compute to after the trailing barrier (and the SCALE hook), so
barrier waits are included. Accumulated per (op, src0 type). A fused group is attributed to its first
op.

**Output:** a table for graph G and G+1, with total and implied tok/s. DSV4 also prints wall time and
the gap between graphs for graphs 45–55.

Arrays are process-static (DSV4), so concurrent graph computes would corrupt them. The dual-CCD path
is not profiled.

## 22. Model-side knobs relevant to the runtime

- `LLAMA_MOE_TOP_K` (llama-model.cpp 7163–7174, also Z5): globally overrides n_expert_used for any MoE
  model at load. **Lossy.** Changes the routing-weight shape, so it disables §8.
- deepseek4.cpp / kv-cache: see §11.

---

## 23. Claimed vs plausible gains (decode, 16C Zen 5, DSV4-Flash)

The docs claim 169 → 110 ms/token "lossless" (+53%) and 98.8 ms with top-4 pruning.

From the runtime code alone:
- Q8 cache, ids reuse and barrier elision save a few hundred µs per token: one requantization plus one
  or two barriers per matmul, over about 600 matmuls.
- MQA8 FA, the HC fusion, and the CSA skip (which removes real compressor work on non-boundary tokens)
  are credible multi-ms wins.
- Much of the claimed 169 → 122 history predates or lies outside this code: kernel work, graph-rebuild
  fixes.

The lossy modes (top-4/5, SwiGLU 50–75%) are where the remaining ~10% decode comes from, and they are
**not lossless** despite marketing that mixes them into "v2.4.0" numbers. The prefill claims (+68%)
rest mainly on the 1xN kernels with the 128-row tiling and the batched MMID scheduling; the "L3 level"
is fictitious.
