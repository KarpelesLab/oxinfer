# llama-modchip: model, graph, server and launcher changes (dirty-room spec)

Scope: the model-level, graph-level, KV/state, loader, server and drafting changes in
`patches/deepseek-v4-flash-fractal-256k.patch` (abbreviated **DSV4**; it is byte-identical to
`dsv4-64gb-cpu.patch` and `llama-modchip-dsv4-10tps.patch`, same md5) and
`patches/zen5-nemotron-256k-optimizations.patch` (abbreviated **Z5**). Also covered:
`speculative/cpp/ngram-map.cpp` (abbreviated **SPEC-NGM**), `kv_cache/llama_kv_cell_rc.h`, the
DSV4 runtime expert-pruning hook (it sits in ggml-cpu.c but is a model-level idea), the launchers
(`systemd/*`, `install.sh`-generated `start_server.sh`, `scripts/llama_modchip.py`, `modchip`,
`apple-silicon/run-server.sh`) and `scripts/prewarm_agent_prefix.py`.

Everything below is my own description. No source is reproduced. "L123" means a line number in the
named patch file.

Classification key:
- **EXACT**: output bits unchanged (pure bookkeeping, skipping work whose result is unused or a no-op).
- **REORDER**: mathematically equal but float rounding may differ.
- **LOSSY**: changes model output. Each one lists exactly when it triggers and whether it is on by default.

---

## 0. Summary table

| # | Change | Where | Class | Default | Wired? |
|---|---|---|---|---|---|
| 1 | Env `LLAMA_MOE_TOP_K` overrides the number of routed experts | DSV4 L7164-7175, Z5 L1220-1237 | LOSSY | Off in the C++; **ON (=4) in the shipped systemd unit and in the generated start script for any MoE with >=6 active experts** | Yes |
| 2 | DSV4 decode-time expert tail pruning + L2/L1 rescale, controlled by a /tmp file | DSV4 L5046-5110, L5198, L5299-5360, mul_mat_id L4810-4855 | LOSSY | Off (top-k 6, min_w 0), **but the setting persists in a file between runs** | Yes (CPU, decode only) |
| 3 | SwiGLU "gate-first" up-row skipping with L2 rescale | DSV4 L4238-4300, L4385-4400, L4505-4510, L5224-5262 | LOSSY | Off (cfg slots 6/7 = 0) | Yes |
| 4 | DSV4 "skip CSA/LID compressor when no block completes" | DSV4 L7018-7019, L7034, L7371-7495, ggml-cpu L5114-5123 | EXACT (CPU) | On | Yes |
| 5 | DSV4 skip indexer top-k when compressed cache <= top_k | DSV4 L7313-7325 | EXACT | On | Yes |
| 6 | DSV4 compressor-state restore/snapshot pruning (rollback planes) | DSV4 L7024-7085, L7371-7530 | intended EXACT; see risks | On | Yes |
| 7 | DSV4 `seq_rm` also truncates compressed caches | DSV4 L7080-7088 | correctness fix (EXACT intent) | On | Yes |
| 8 | `need_n_rs_seq` reserves rollback planes for every speculative type | DSV4 L8-17 | EXACT (memory cost) | On when any spec type set | Yes |
| 9 | RouteCrack/"Stockfish" n-gram drafter replacing `ngram-simple` | DSV4 L21-1427; SPEC-NGM | EXACT (drafts are verified) | On when `--spec-type ngram-simple` (shipped launchers use it) | Yes |
| 10 | Generic ngram-map reverted to older algorithm + RouteCrack fallback | DSV4 L1072-1427 | EXACT | On for ngram-map types | Yes |
| 11 | Server: pin first context checkpoint | DSV4 L7616-7630, Z5 L1243-1262 | EXACT | On | Yes |
| 12 | Server: save/restore hybrid checkpoints next to slot files (`.ckpt`) | DSV4 L7638-7716, Z5 L1263-1337 | EXACT | On whenever slot save/restore API is used | Yes |
| 13 | `MADV_HUGEPAGE` on model mmap and big aligned mallocs | DSV4 L7091-7103, L6991-7008; Z5 L1174-1214 | EXACT | On (Linux) | Yes |
| 14 | GGUF array-length tolerance (zero-fill short arrays) | DSV4 L7105-7118 | potentially LOSSY / unsafe | On | Yes |
| 15 | Qwen3.5-MoE per-layer head dims + alt `ssm_dt` tensor name | DSV4 L7535-7608 | EXACT (loader compat) | On | Yes |
| 16 | `llama-model.cpp` collateral reverts (arches removed, `get_tok_embd` rewrite) | DSV4 L7120-7303 | regressions | n/a | Yes |
| 17 | `llama_kv_cell_rc.h` mixed FP16/Q4 KV cell | kv_cache/ | n/a | n/a | **Dead** (self-declared, not included anywhere) |
| 18 | Slot prewarm via /dev/shm | scripts/prewarm_agent_prefix.py | EXACT | Manual | Client-side only |
| 19 | Launcher-tuned runtime flags | systemd, start_server.sh, llama_modchip.py | mix | see §8 | Yes |

The DSV4 hash-routed MoE layers (token-id to expert table for layers 0..2) are **not implemented by
this patch**; they already exist upstream. The patch only protects them from pruning with a
hard-coded "layer index >= 3" test (see §2).

---

## 1. `LLAMA_MOE_TOP_K` (routed expert count override) — LOSSY

**Where:** inside hparam loading, right after the expert-group keys are read
(DSV4 L7164-7175, Z5 L1224-1235). Identical in both patches.

**Algorithm:** if env `LLAMA_MOE_TOP_K` parses to a positive integer `v`, and `v <= n_expert`, and the
model has `n_expert_used > 0`, then set `n_expert_used = v` and log it. Applied once at model load to
every architecture and every MoE layer. It can also *raise* the count (no `v <= original` check).

**Effect on math:** the router's top-k selection now keeps `v` experts. Whether the remaining
weights are renormalized depends entirely on the existing per-arch flag (`expert_weights_norm`):
- norm on: kept weights are divided by their sum, so they still sum to 1 (or to the scale factor).
- norm off: the dropped mass is simply lost.

**Classification:** LOSSY whenever `v != original n_expert_used`.

**Default:** the C++ default is off. But:
- `systemd/nemotron-256k.service` sets `LLAMA_MOE_TOP_K=4` in both "Mode A" and "Mode B"
  (Nemotron 6 -> 4 experts).
- `install.sh` generates `start_server.sh`, which calls `scripts/llama_modchip.py`; its profile sets
  `moe_top_k = 4` for **any** MoE GGUF with `expert_used_count >= 6` (llama_modchip.py ~L520-540)
  and exports `LLAMA_MOE_TOP_K=4`. So on the standard Linux install path, pruning is **on by default**.
  It is labeled "TRADE-OFF", or "EXPERIMENTAL" when norm is off.
- The docs call it "opt-in" (NEMOTRON_OPTIMIZATIONS row 10). That is wrong for the shipped launchers.

**Overstated claims / bugs:**
- Docs say it can be "switchable live via TUI" through `/dev/shm/llama_modchip.bin`. The C++ never
  reads the `moe_top_k` field of that block. The value is read from env only, once, at load time.
- On DSV4, changing `n_expert_used` changes the routing-weight tensor from [1,6,...] to [1,v,...].
  The DSV4 pruning hook (§2) matches only a 6-wide tensor, so it silently turns off. Also, the
  upstream hash-routed layers (fixed 6-column token-to-expert table) may not tolerate v != 6. I could
  not verify this from the patch (the upstream code is not in the diff). Treat it as a risk.
- Expected gain: routed-expert bytes per token scale with `v / original` (6 -> 4 = -33% of routed
  traffic). Shared-expert, attention and SSM traffic are unchanged.

---

## 2. DSV4 runtime expert tail pruning (graph-executor hook) — LOSSY, off by default, persistent

**Where:** `ggml_graph_compute_thread` in ggml-cpu.c (DSV4 only). Init at L5046-5110. The hook runs
after the barrier that follows a node (L5299-5360). The skip mask is consumed in `mul_mat_id`
(L4810-4855).

**Config source (8 x int32 "cfg" array):**
1. Thread 0, first graph only: read env `DSV4_MOE_TOPK` (default 6) and `DSV4_MOE_MIN_W` (default 0).
2. Open or create a 32-byte file: `DSV4_MOE_CFG_PATH`, else `/tmp/dsv4_moe_cfg.bin` (Windows:
   `%TEMP%\dsv4_moe_cfg.bin`). Flags are read-write plus create, mode **0666** (and an explicit
   fchmod to 0666), with no-follow-symlink.
3. If the file is shorter than 32 bytes, write `[tk, tk, round(min_w*10000), 3, 43, 1, 0, 0]`.
4. Map it shared. If read-write open fails, fall back to a read-only map. The executor reads the
   mapping on every graph, so edits take effect "live".

Meaning of the cfg slots:
`[0] top-k for "early/late" layers, [1] top-k for "mid" layers, [2] min weight x 1e4,
[3] mid start layer, [4] mid end layer (exclusive), [5] rescale mode (1 = L2, 2 = L1, other = none),
[6] SwiGLU keep-count out of 128 for MUL_MAT_ID, [7] the same for plain MUL_MAT (see §3)]`.

**Trigger:** after any node with op SCALE and shape `[1, 6, 1]`, i.e. the DSV4 routing-weight tensor
after the routed scaling factor, for a single token (decode only; prefill has ne[2] > 1 and is never
pruned). The layer index is parsed from the text after the last '-' in the node name. If there is no
'-', the index is -1 and nothing is pruned.

**Algorithm (thread 0, then barrier):**
1. Let `w[0..5]` be the six routing weights. Sort the slot indices by weight, descending.
2. `K = cfg[1]` if `mid_start <= il < mid_end`, else `cfg[0]`. Clamp K to at least 1.
3. Only if `il >= 3` (protects the hash-routed layers 0..2) **and** (`K < 6` or `min_w > 0`):
   - Rank r (0-based) is dropped if `r >= K`, or if `r > 0` and `w < min_w`. A dropped slot gets its
     weight zeroed in place and its bit set in a 6-bit skip mask (bit = slot position in the top-k
     list, not the expert id).
   - Rescale every weight by one factor:
     - mode 1 (L2): `sqrt(sum_all w^2 / sum_kept w^2)`
     - mode 2 (L1): `sum_all w / sum_kept w`
     Rescale only if kept < all.
   - Publish the mask to a global atomic.
4. Otherwise publish mask 0.
5. Every following MUL_MAT_ID with a single token (ids->ne[1] == 1) picks up the mask. When grouping
   rows per expert it skips masked slots, and it zero-fills the output rows of masked slots. The
   expert weights for those slots are never read.

**Classification:** LOSSY. It triggers only when `il >= 3`, decode with n_tokens == 1, and the file
or env values give `K < 6` or `min_w > 0`. Default values give K = 6 and min_w = 0, so it is **off by
default**.

**Bugs and risks:**
- **Persistence footgun.** The env vars only initialize the file when it is missing. After the first
  run, the file is the source of truth, and changing `DSV4_MOE_TOPK` has no effect. A lossy setting
  written once survives restarts.
- **Security.** A world-writable control file in /tmp lets any local user change another user's
  model outputs. If another user created the file first, the read-write open fails and the read-only
  fallback maps the attacker's file. No owner check is done (unlike the Z5 shm reader).
- **Graph-order dependency (unverified, important).** The mask is set when the SCALE node executes
  and applies to whatever MUL_MAT_ID runs next. If the routing-weight SCALE is scheduled after that
  layer's expert matmuls (plausible with depth-first graph build, because the expert chain only
  depends on the ids, not on the weights), the zeroed weights still apply to the right layer, but the
  compute skip lands on the *next* layer's experts at the same slot positions. That would be a silent
  extra-lossy bug. A reimplementation should bind the mask to the layer explicitly.
- Everything is hard-coded to DSV4: 6 experts, layers 3 and 43, layer index from the node name.
- Docs give the L2-vs-L1 math (DEEPSEEK_V4_OPTIMIZATIONS §3.2). The code matches it. The claimed
  numbers: top-6 9.06 tok/s, top-5 L2 9.68 tok/s (+0.42% local PPL), top-4 L2 10.12 tok/s (+2.85%).
  So the gain is about 7-12% of decode for a small quality loss. "Top-1/Top-3 match 95.8%" means
  about 1 in 24 tokens changes, which is not negligible.

---

## 3. SwiGLU up-projection row skipping ("gate-first") — LOSSY, off by default

**Where:** helper at DSV4 L4238-4300, the dot-product loop in mul_mat_one_chunk at L4385-4400 and
L4505-4510, and the arming logic in the executor at L5224-5262.

**Arming (per MUL_MAT / MUL_MAT_ID node):** all of these must hold:
- the cfg file is mapped, and the node has ne[2] == 1 and ne[3] == 1;
- `keep = cfg[6]` for MUL_MAT_ID or `cfg[7]` for MUL_MAT, with 0 < keep < 128;
- the layer index from the node name is in [cfg[3], cfg[4]);
- the next node is a SWIGLU GLU whose *second* operand (the "up" side) is this node, optionally
  through one CLAMP;
- the GLU's first operand (the gate) is already computed, is F32, and has the same shape.

**Kernel:** applies only when the thread's chunk is one column (one token) and at most 128 rows long.
1. Let `n` = rows in the chunk, `k_keep = floor(keep * n / 128)`.
2. For each row j compute `s_j = silu(g_j)^2` from the gate values.
3. Quickselect the threshold so that the top k_keep rows by s_j are kept (ties filled in index
   order). `alpha = sqrt(sum_all s / sum_kept s)`.
4. Skipped rows output 0. Kept rows compute the normal dot product times alpha.
The skipped up-weight rows are never fetched.

**Classification:** LOSSY. It changes the FFN output whenever it is armed. The default cfg has 0 in
both slots, so it is off. Note that the selection is per thread chunk ("stratified"), so results
depend on the thread count and chunking.

**Claims:** docs show combined profiles giving 9.7-10.7 tok/s with probability cosine 0.98-0.994.
"100% exact match" on one tool-call prompt is not a quality metric.

---

## 4. DSV4: skip the CSA/LID compressor subgraph when no compressed block completes — EXACT (CPU)

**Background:** DSV4 attention keeps three compressed side caches. CSA and LID (the indexer) use
ratio 4; HCA uses a larger ratio. Each decode step builds a "compress from state" subgraph that pools
the last `ratio` rows of state into one compressed KV row, then scatter-writes it into the compressed
cache at `state_write_idxs`. Upstream always builds and runs this subgraph. When the current token
does not complete a block, upstream writes a dummy row to the last cache slot with position 0.

**Changes:**
1. In the compressor plan (DSV4 L7034), the "no completed block" case now records write-position
   **-1** instead of 0, as a sentinel.
2. When the "csa" inputs are set (llama-graph, DSV4 L7018-7019), a process-global int flag is set to
   1 iff the plan has exactly one write entry and its position is -1. That means a single-token
   ubatch whose token does not complete a 4-block.
3. Graph build (deepseek4.cpp, DSV4 L7371-7410 for CSA, L7459-7495 for LID): the state projection
   tensors and any restored state are force-expanded *first*. Then the node count is recorded, the
   compression subgraph and the cache write are built, and every node added after that point gets
   flag bit **0x100**.
4. Executor (ggml-cpu, L5114-5123): it reads the global once per graph. If it is set, every node
   carrying bit 0x100 is skipped like a NOP.

**Classification:** EXACT on CPU. The skipped work only produced a dummy row that nobody reads.

**Gain:** 3 of every 4 decode tokens skip the CSA and LID compression for all layers. The graph
topology stays fixed, so the graph can still be reused. Docs claim the "graph rebuild and compressor
overhead" goes from 1.54 ms to 0 ms per token.

**Risks:**
- The flag is a process-global, not per graph or per context. Two contexts, or a draft and a target
  model, computing concurrently can race.
- Bit 0x100 may collide with future ggml tensor flags.
- Only the CPU backend honors the flag. Any other backend will execute the compressor with position
  **-1** as an input. Upstream passed 0 here, so this may index out of range or corrupt data.
- One flag set from the CSA plan also gates the LID nodes. That is fine only because both use ratio 4.

## 5. DSV4: skip indexer top-k when everything would be selected — EXACT

**Where:** `build_csa_lid_attention` (DSV4 L7313-7325).

If the number of compressed CSA entries visible in the mask (mask width) is <= `indexer_top_k`, the
lightning-indexer scoring, its top-k and the top-k mask are not built. The plain CSA mask is used
instead. Every valid entry would have been selected anyway, so this is EXACT. Gain: the whole indexer
attention is skipped while context is below about `4 * indexer_top_k` tokens.

## 6. DSV4: compressor-state rollback planes (restore/snapshot) — intended EXACT, see risks

**Background:** for speculative decoding, each sequence keeps `1 + n_rs_seq` "planes" of compressor
state. Plane 0 is the live state. Plane d is the state d tokens ago. A rollback copies plane r into
plane 0 before the next batch. Each batch also writes snapshot planes.

**Changes (llama-kv-cache-dsv4.cpp L7024-7085, deepseek4.cpp L7371-7530):**
- **Restore:** upstream always emitted a full-width identity gather (plane 0 to plane 0) to keep the
  graph shape fixed. Now gather indices are emitted only when a rollback 1 <= r <= n_rs_seq is pending.
  When there are none, the restore inputs are absent and the restore nodes are not built. The graph
  shape now varies with rollback state, which costs graph reuse on those steps.
- **Snapshot:** skipped entirely for sequences with <= 1 token in the ubatch. Otherwise only planes
  1 .. min(n_rs_seq, n_seq_tokens - 1) are written. The source plane for "history" rows becomes
  `clamp(r_eff + d - n_seq_tokens, 0, n_rs_seq)`, where `r_eff` is the pending rollback or 0.
  Upstream used `(d - n_seq_tokens)` without a clamp, which can go negative.
- In the graph, the CSA, LID and HCA restore and snapshot concats are built only if their index
  inputs exist. Restored tensors are force-expanded when a restore exists.

**Classification:** meant to be EXACT. It is correct as long as rollback depth never exceeds
(previous batch tokens - 1). That holds for verify-then-rollback speculative decoding.

**Risks (only partly verifiable; the unchanged branch condition is outside the diff):**
- With max_d capped at n_seq_tokens - 1, the "shift older plane" branch appears unreachable, so
  planes >= n_seq_tokens are never refreshed and keep stale data. A deeper rollback, or a rollback
  after a single-token step (where snapshots are skipped), would restore stale state.
- In the HCA path, the fallback that built `hca_source_kv` / `hca_source_score` when they were still
  null was **deleted**. If any later code uses them while they are null, this is a crash or
  regression. The surrounding code is not in the diff.

## 7. DSV4 `seq_rm` propagates to compressed caches — correctness fix

**Where:** DSV4 L7080-7088. After the raw KV `seq_rm(seq, p0, p1)` succeeds, the patch also removes
entries from CSA and LID starting at `p0 / 4`, and from HCA starting at `p0 / HCA_ratio`, always to
the end (p1 is ignored and treated as -1). The partial block is regenerated on re-decode from the
restored state rows.

Bug: a middle-range removal with p1 != end wrongly deletes everything after p0. The server only does
tail truncation, so in practice this is fine.

## 8. Launcher and systemd tuned settings

| Setting | systemd unit / start_server.sh | Notes |
|---|---|---|
| threads `-t` / `-tb` | 16 / 16 (unit). start_server: all physical cores, or on >2 CCDs with >24 cores, `max(16, 4*CCDs)` capped at core count, decode = batch | pinned with `CPUAffinity=0-15` or `taskset` to one SMT sibling per core; `OMP_PROC_BIND=close`, `OMP_PLACES=cores` |
| `-b` / `-ub` | 2048 / 2048 | Mac script: ub 1024 by default (2048 OOMs Metal at 128K) |
| ctx | 262144 (unit); start_server picks by RAM (falls back to 65536) | `-np 1` |
| KV types | K q8_0, V q4_0 (Mac: both q8_0) | LOSSY vs f16 KV (standard llama.cpp option) |
| `--repack`, `--flash-attn auto`, `--load-mode none` | yes | `--load-mode none` replaces mlock/mmap flags in this llama.cpp version; **no `--mlock`** anywhere |
| prompt cache | `--cache-reuse 256`, `--ctx-checkpoints 64`, `--checkpoint-min-step 128`, `--slot-save-path /dev/shm` | Mac: `--cache-ram 1024`, checkpoints 8 |
| speculative | `--spec-type ngram-simple`, n=4, m=7, min-hits=1 | hijacked by RouteCrack (§9); n and min-hits are ignored by it |
| reasoning | `--reasoning-format deepseek --reasoning-budget 0` | budget 0 = thinking disabled: changes outputs vs model default |
| env | `LLAMA_MOE_TOP_K=4` (LOSSY), `LLAMA_KV_L3_WINDOW=2048`, `LLAMA_ATTN_WINDOW=0` (Mode A exact; Mode B 4096 = LOSSY SWA), `LLAMA_ATTN_SINKS=16`, `LLAMA_SWA_STRIDE=1`, `LLAMA_KV_DISABLE_CHUNKING=1`, `LLAMA_RC_*` gates 84/92/96, `RC_TABLES_DIR` | the attention env knobs belong to the ops.cpp analysis |
| THP | `systemd/thp-llm.conf` sets THP `enabled=madvise`, `defrag=defer+madvise` | pairs with §13 |

`llama_modchip.py` profile rules (these generate the env above):
- L3 window = 25% of L3 divided by KV bytes per token, rounded to 64 tokens, clamped to
  [512, 2048] for hybrid SSM models or [256, 4096] otherwise.
- SWA is off by default. With `--swa`, pure-GQA models are forced to stride 4.
- RC gates: 84/92/96 for hybrid, 80/90/95 for MoE, 78/88/94 for dense.
- `moe_top_k = 4` for any MoE with at least 6 active experts.
- It writes a 48-byte control block (magic "MODC", version 1, then 10 int32 fields) to
  `$XDG_RUNTIME_DIR` and `/dev/shm`, mode 0600, by **atomic rename**.

Bug: the C++ readers map the file once and keep the mapping. An atomic rename creates a new inode,
so later writes by the "live TUI" are **not seen** by a running server. The "live-tunable" claim is
false for edits made through the provided tool. It works only for in-place writes.

`apple-silicon/run-server.sh` and `fedora/nemotron.sh` use **stock** llama.cpp with
`--spec-type ngram-mod` or `draft-mtp`. None of the patches apply there.

---

## 9. RouteCrack + "Stockfish" n-gram drafter (`common_ngram_simple_draft` replacement) — EXACT

Drafts are verified by the target model, so output is unchanged under greedy decoding and under
llama.cpp's sample-and-match verification. Only speed changes.

**Wiring:** DSV4 replaces upstream `common/ngram-map.cpp` (L21-1427). Z5 does not patch this file,
but `scripts/build_zen5_llama.sh` copies `speculative/cpp/ngram-map.cpp` plus `rc_predictor.h`,
`rc_trie_v2.h` and `rc_stockfish.h` over `common/`. Active when `--spec-type ngram-simple` is used
(the shipped unit and script use it). It is also the fallback inside ngram-map drafting (§10).

**Global engine:** one process-wide singleton behind one mutex. A static initializer loads the
tables when the library loads, in **every** binary that links libcommon. Tables come from
`RC_TABLES_DIR`, default `./workspace/routecrack` (relative to the CWD):
- 8 static cascade tables, one each for k = 10, 8, 6, 5, 4, 3, 2, 1 (`hc_top4_k{k}.bin`). Each entry
  is a 64-bit FNV-1a hash of the last k tokens, with up to 4 predicted tokens and a 0-100 confidence
  each.
- `hot_rules.bin`: previous token -> prediction plus confidence, sorted for binary search.
- `macro_codebook.bin`: hash of a 5- or 6-token context -> continuation of up to 4 tokens plus
  confidence.
- `corpus_nemotron_warm.bin`: int32 tokens; the last 80,000 are replayed into the online learners.

**None of these tables ship in the repo.** Without them, `n_cascade = 0` and the engine runs in
"online_only" mode.

**Online learners** (updated from a 12-token window ending at each newly observed token):
- PPM, orders k = 2..6: open-addressed tables of 2^18 slots per order, 8 linear probes, keyed by
  FNV64 of the k-token context. Each entry keeps up to 4 predictions with counts, kept sorted by
  bubbling up. Predict: for k from 6 down to 2, take the first matching entry with total >= 2 and add
  `2^k * count/total` to each of its candidates. Return candidates sorted by score.
- Skipgram: key = (token at t-3, token at t-1), value = the first token seen at t, plus total and
  hit counters. Predict needs total >= 2.
- Trie: inserts all suffixes of depth 3..10. Predict takes the deepest matching suffix (up to the max
  depth) that has children, returns its most frequent child, `conf = best/sum` of child counts,
  `obs = sum`, and the depth.
- Zobrist TT ("Stockfish"): three direct-mapped tables of 4096 x 16-byte buckets, for the last 5, 4
  and 3 tokens. Hash = XOR of a SplitMix64 mix of `(token << 4) | position`. Each bucket holds the
  upper 32 hash bits, best_move, refuted_move, history in [-512, 512], wins and total.
  - Observe(actual, drafted):
    - new key: best = actual; refuted = drafted if the draft was wrong, else none;
      history = -128 if refuted, else 96; wins = 1.
    - same best: wins += 1, clear refuted if it equals actual, history gravity +128.
    - different best: refuted = drafted-if-wrong, else the old best; best = actual; wins = 1;
      history gravity -192.
  - Gravity update: `h += bonus - h*|bonus|/512`, then clamp.

**Ensemble query `Q(ctx)` -> (token, certainty) or verify-mode `V(ctx, cand)` -> agree count:**
1. Cascade: the first k in table order whose hash hits supplies top1 and its confidence.
2. Disambiguation: if top1 is refuted in TT5 or TT4 (TT5: refuted and best differs and history < 0;
   TT4: same with history <= -64), or its confidence is below 80, try alternatives 2..4 with
   confidence >= 20. Switch to an alternative (certainty 84) if a TT confirms it (TT5 best with
   wins >= 3 and history >= 160, or TT4 best with wins >= 4 and history >= 200) **and** the trie
   (depth >= 4, conf >= 0.8) **and** PPM top1 agree.
3. Verify mode: agree = count of {cascade top1 or cand in cascade list, trie, PPM, skipgram, hot rule}
   equal to the candidate.
4. Predict mode:
   - With a cascade hit: `cert = conf + 4·[k >= 6] + (6 if agree >= 3, 3 if agree == 2)`, where agree
     counts trie (depth >= 2), PPM, skipgram and hot rule. A k = 1 hit with conf < 90 gives cert 0.
   - Else trie with depth >= 4, obs >= 3, conf >= 0.88: cert = 100·conf.
   - Else hot rule with conf >= 90 that PPM or skipgram also predicts.
5. Trie rescue: if cert < 84 and the trie has depth >= 4, obs >= 4, conf >= 0.92 and is not refuted
   in TT5, TT4 or TT3, take the trie token when either:
   - a TT confirms it (TT5 wins >= 2 or TT4 wins >= 3) and at least 2 sources agree, or
   - depth >= 6, obs >= 5 and at least 3 sources agree.
   Then `cert = min(95, 92·conf + 4·[TT-confirmed])`.

**Draft procedure per call (inputs: history `inp`, last sampled token `s`, max length `M`):**
1. Return nothing if M == 0, history < 12, or RC is disabled. Disable sources, in priority order:
   shm field, then env `LLAMA_RC_ENABLED` (default 1). Gates come from shm, else env
   `LLAMA_RC_CERT_D0/D1/D2` (defaults 84/92/96).
2. Window = last 11 history tokens + s (buffer of 28).
3. Online update: if the history grew by 1..8 tokens since the last call, feed exactly those
   positions. Otherwise feed the last min(len, 4096) positions (DSV4) or just 1 (SPEC-NGM). Each
   feed records the previous draft token at that step so the TT learns refutations.
4. **Stage 1, in-context match.** Anchor = (inp[-2], inp[-1], s).
   - Hot zone: scan backwards over the last 16384 positions, at most 5 hits.
   - Each hit gets match length k: extend backwards while tokens agree, up to 12.
   - Rejections: (DSV4, with tables) prefix-zone hits with k < 5; hot-zone hits more than 256 back
     with k < 4. Online-only mode lowers the prefix threshold to 3 and drops the hot-zone rule.
     SPEC-NGM always uses the stricter rules.
   - Keep the best hit (largest k; later hits must agree on the following token to replace it). Track
     a "consensus length": how many following tokens the hits agree on, starting at 7. Remember the
     first conflicting second-best hit.
   - If the hot zone has no hit, scan the first 24576 tokens (the "prefix zone") backwards, at most
     3 hits.
   - Conflict (consensus 0): use V and TT4 to pick between the two continuations. One side needs
     agree >= 3 while the other has 0, plus TT4 best with wins >= 2. Then consensus = 1.
   - Base length from (k, hits, recency):
     - with tables: k >= 8 -> 5; k = 7 -> 5 or 4; k 5-6 -> 4 or 3; k = 4 -> 2 or 1;
       k = 3 -> 1 if hits >= 2 or (within 128 tokens and TT5-confirmed).
     - online-only: smaller, 1-4.
   - Extension cap: 7 if k >= 10 or (k >= 8 and at least 2 agreeing hits); 6 if k >= 7 with 2+
     agreeing hits; else the base length. Final `len = min(cap, consensus, M)`.
   - Copy continuation tokens one by one. Stop on:
     - a TT3 refutation for a single-hit match with k <= 5;
     - (with tables) single-hit k <= 5 tokens that lack verify agreement and TT backing;
     - positions beyond the base length that lack at least 2 agreeing sources (1 if k >= 11) or a TT
       confirmation, or that are TT-refuted.
5. **Stage 2, ensemble rollout** (only if the draft is shorter than the limit):
   - Look up a macro entry on the last 6 then 5 tokens (needs conf >= 92 and continuation >= 3).
   - Limit = min(M, macro ? min(4, start + 4) : 3).
   - Per step d: take Q.
     - Macro boost: if Q's token equals the macro continuation, cert = max(cert, macro conf + 4·[step >= 1]);
       otherwise drop the macro.
     - TT adjustments (TT5 checked first, otherwise TT4):
       - TT5: refuted -> switch to its best (cert 88 + min(8, wins)) if wins >= 3 and history >= 128,
         else -12; confirmed with wins >= 2 -> +4; no token but a strong TT5 best that the verify
         query backs -> adopt it (cert 86+).
       - TT4: similar, with -10 / +4 / adopt (cert 86).
       - TT3, when the cascade k <= 3: -10 if refuted, +3 if confirmed.
     - Gate = D0, D1, D2 for d = 0, 1, >= 2. Quiescence: if cert is within 2 below the gate, look one
       token ahead; promote to the gate if the next prediction has k >= 6, cert >= 95, agree >= 3 and
       a matching TT5 best with wins >= 2.
     - Accept if cert >= gate, else stop.
6. Remember up to 8 draft tokens for the next call's refutation learning.

**Gain claims:** "48-60 tok/s average/peak vs 40.9 baseline", "96.4% acceptance". That is plausible
only for repetitive agentic or tool-call text with the offline tables (not shipped). Online-only mode
is close to a plain prompt-lookup decoder with extra gating.

**Bugs and issues:**
- One global engine for all slots and sequences. `last_inp_size` tracking breaks with more than one
  slot. Learned statistics leak between users and requests (privacy). A new request triggers a
  4096-token re-feed (DSV4), which double-counts.
- `size_n` and `min_hits` from the CLI are ignored. Only m (max draft) is used.
- DSV4 reads the shm block with no owner check and no symlink protection. SPEC-NGM adds no-follow,
  a regular-file check, uid == self and a magic check. Effect is limited to draft speed and enabling.
- The static initializer does disk I/O from a CWD-relative path in every linked tool.
- Labels like "Stockfish-PVS", "Zobrist" and "quiescence" are marketing names. There is no search;
  it is a 1-ply lookahead plus hash tables.

## 10. Generic ngram-map (`ngram-map-k` / `-k4v` types) — EXACT, regression

DSV4 replaces upstream's newer ngram-map (LCG hash, map refresh and statistics) with an older,
simpler design:
- Hash = shift-xor of the key tokens modulo a fixed size. On a bucket collision with equal key
  tokens, the key list is searched **linearly** (O(number of keys) per new position).
- Up to `MAX_VALUES` continuations per key, each with a count and an n_accepted field (-1 initially).
- Best value = highest n_accepted, then highest count; values with n_accepted == 0 are skipped
  permanently.
- The map resets if the history shrinks or its first 100 tokens change.
- Draft: find a key matching (last size_key - 1 tokens + sampled). Return early with **no fallback**
  if key_num < min_hits. Otherwise emit the best value's tokens.
- If nothing was emitted, fall back to the RouteCrack simple-draft with m = size_value.

Output stays exact (drafting only). It is likely slower than upstream on long histories because of
the linear key search.

## 11. Server: pinned first checkpoint — EXACT

DSV4 L7616-7630 / Z5 L1243-1262. When the per-slot checkpoint list is full, upstream evicts the
oldest. The patch evicts the **second**-oldest when there are at least 2, so the first checkpoint
(assumed to be the system prompt + tools prefix) survives.

Caveat: the first checkpoint is not necessarily the shared prefix. If it belongs to an unrelated
earlier conversation, it is pinned until prefix invalidation removes it. Gain: avoids a cold
re-prefill of about 22K tokens (minutes on CPU) for hybrid/recurrent models, which cannot roll back
without a checkpoint.

## 12. Server: `.ckpt` companion file for slot save/restore — EXACT

DSV4 L7638-7716 / Z5 L1263-1337.

**Save:** after the normal sequence-state save to `<slot-save-path>/<filename>`, write
`<file>.ckpt` (truncate) with:
- u32 magic "CKPT" (0x434B5054), u32 count;
- then per checkpoint: n_tokens, pos_min, pos_max (native widths), three u64 sizes (target, draft,
  spec blobs), then the three blobs.

**Restore:** after the normal restore, if `<file>.ckpt` opens, the magic matches and count <= 256,
read entries and append them to the slot's checkpoint list. Stop at the first short read.

Purpose: a saved hybrid/Mamba slot normally keeps only the final recurrent state, so any divergence
after restore forces a full re-prefill. With checkpoints, partial reuse works.

**Bugs and security:**
- There are no sanity limits on blob sizes, so a crafted file can force huge allocations.
- There is no binding to the model or the token list.
- Write failures are silent.
- The launchers set `--slot-save-path /dev/shm`, which is world-writable with the sticky bit and has
  predictable names (`claude_40tools_slot0.bin`). Another local user can pre-create
  `<name>.bin.ckpt` (or the .bin). Then the service's save silently fails to overwrite it, and every
  restore loads attacker-controlled state into the model.
- Whether restore clears old checkpoints first depends on upstream `prompt.clear()`.

## 13. Hugepages — EXACT

- Model mmap (both patches): after mapping the GGUF, `madvise(MADV_HUGEPAGE)` on the whole range;
  the return value is ignored. For file-backed page cache on ext4/xfs this does little unless the
  kernel supports file THP or large folios. The "-99.6% dTLB misses on weights" claim is overstated
  for mmap'd weights. It is effective for anonymous buffers.
- `ggml_aligned_malloc`:
  - Z5: allocations >= 2 MiB are 2 MiB-aligned (posix_memalign) and then madvised.
  - DSV4: only madvises from the page-aligned start, without raising the alignment.
  - With `--repack` the weights live in anonymous repack buffers, so this is where hugepages really
    apply.
- Expected gain: fewer TLB misses on GEMV over multi-GB buffers. Real but modest.

## 14. GGUF array-length tolerance — unsafe

DSV4 L7105-7118. In the "fixed-size array from GGUF" path, if the stored array is *shorter* than
expected, the code zero-fills the destination and retries with the variable-length reader instead of
throwing. Per-layer parameters missing from the file silently become 0. That is LOSSY or crashy for
any model where 0 is not a valid value. Purpose: load GGUFs with truncated per-layer arrays (pruned
models).

## 15. Qwen3.5-MoE loader — EXACT

DSV4 L7535-7608:
- Attention tensor shapes and the attention graph use per-layer `n_embd_head_k(il)`, `n_head(il)`
  and `n_head_kv(il)` instead of the global values, including for the MTP block.
- The `ssm_dt` tensor is first tried without the ".bias" suffix, then with it.

This allows heterogeneous or pruned Qwen3.5/3.6-MoE GGUFs. It does not change output for standard
files.

## 16. Collateral damage in `llama-model.cpp` (DSV4 only)

The diff was clearly made against a base newer than the author's tree, so applying it **removes**
upstream features:
- arch mappings CLIP, QWEN3TTS, MUSE_GLIMMER and GRANITE_SWITCH (and their rope and print entries);
- MTP memory handling for Nemotron-H (breaks `draft-mtp` on Nemotron in the patched build);
- the TENSOR_SKIP path in qkv creation.

`llama_model_get_tok_embd` is rewritten. It now returns the **token count** instead of the element
count, and dequantizes straight from `tensor->data`, which assumes host memory. A caller that sizes
its buffer from the "query size" call (pass null, then allocate) gets a buffer n_embd times too small
and a **heap overflow** on the second call. It also crashes for device-resident embeddings. Callers
are speculative (EAGLE3/DFlash-style); this needs checking.

## 17. `kv_cache/llama_kv_cell_rc.h` — DEAD

The header says itself that it is a prototype that is not included anywhere. Even as a design it is
broken:
- the token id is 16-bit (Nemotron's vocab is 131K);
- the Q4 variant is padded to the FP16 size, so it saves no memory;
- there is one scale per 128-dim head (not per 32-element block);
- the /dev/shm "HARP page" descriptor has no implementation.

## 18. Prefix prewarm (`scripts/prewarm_agent_prefix.py`) — EXACT, client side

Uses stock server endpoints:
1. Wait for `/health`.
2. If `/dev/shm/claude_40tools_slot0.bin` is missing but a disk copy exists (`LLAMA_SLOT_SAVE_DIR`,
   default `./workspace/cache`), copy it into /dev/shm.
3. Call slot 0 restore with that filename.
4. Load the captured agent request (`claude_warmup_payload.json`), replace the messages with a single
   "ok", set max_tokens 1, and POST it to the gateway (`/v1/messages`, prewarm headers). This
   prefills the system prompt and tool schemas.
5. If the restore failed or the warmup took more than 5 s, save slot 0 and copy it back to disk.

The quoted "18.9 ms restore" is a memcpy of state from tmpfs, so it is plausible. Its value depends
on §11-12 for hybrid models. The /dev/shm security caveat from §12 applies.
