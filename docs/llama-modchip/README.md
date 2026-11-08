# llama-modchip: cleanroom specs

Dirty-room analysis of [synchopate/llama-modchip](https://github.com/synchopate/llama-modchip)
(Apache-2.0; analyzed at commit `a9635a5`, 2026-11-09). The project is a set of patches to llama.cpp/ggml
(pinned at `d2f83055`), plus a speculative drafter, launch scripts and Python tooling, targeting CPU
inference of DeepSeek-V4-Flash and Nemotron-3.5-30B-A3B.

These specs describe algorithms, bit layouts and parameters in prose, tables and pseudocode only. They
contain **no copied source code**. Upstream line references are there so a reviewer can check claims;
cleanroom implementers should read only these documents, never the upstream files.

All findings come from reading the code. Nothing was built or benchmarked.

| Spec | Scope |
|---|---|
| [quant-formats-and-kernels.md](quant-formats-and-kernels.md) | "Fractal-64" Q2_0 block format, AVX-512/AVX2 dot kernels, multi-column (weight-reuse) kernels, conversion pipeline |
| [cpu-runtime.md](cpu-runtime.md) | ggml-cpu scheduling: prefill tiling, Q8 activation cache, MMID reuse, barrier elision, MQA flash-attention, lossy MoE/SwiGLU pruning, control file |
| [model-and-server.md](model-and-server.md) | DeepSeek-V4/Qwen graph changes, compressor skip, checkpoints and slot restore, hugepages, shipped launch settings |
| [speculative-and-kv-cache.md](speculative-and-kv-cache.md) | The "RouteCrack / Stockfish-PVS" drafter, HARP-Cache, EAKV |

## Verdict

The README and paper overstate the project a lot. Some of the work is real, and a few ideas are worth
reimplementing. Two things hold in every spec:

- **Headline numbers mix lossless and lossy modes.** The shipped systemd unit and the generated start
  script set `LLAMA_MOE_TOP_K=4`, which cuts active experts on every MoE model even though the docs call
  it opt-in. Expert-tail pruning and SwiGLU row skipping are switched by a world-writable, persistent
  `/tmp/dsv4_moe_cfg.bin`.
- **Much of the speedup is just a smaller format.** The DeepSeek gains come mostly from re-quantizing
  dense tensors from Q8_0 to a 3.75 bpw format. The bit tricks in the kernels matter much less.

### Worth reimplementing (all lossless)

1. **Multi-column weight-reuse matmul kernels.** Each quantized weight block is decoded once and applied
   to 2, 3, 4 or 8 activation columns. Weight rows are tiled in groups of 128. Column counts are split
   into kernel widths, e.g. 5 = 3+2 and 9 = 4+3+2. This helps prefill and speculative or MTP batch
   verification, where N is small (2–16) and the matmul is bandwidth-bound. Directly relevant to our
   verify path.
2. **Activation-quantization cache.** When several matmuls share an input (Q/K/V, gate/up, MoE
   gate/up), quantize the activation to Q8 once and reuse it. Invalidate the cached copy whenever a write
   overlaps its source. Upstream's version has a bug when a different backend handles the matmul; see
   cpu-runtime.md.
3. **Reuse of the MoE routing table.** The expert to token-row grouping built for the gate matmul is
   reused for the up matmul.
4. **Barrier elision** between graph nodes that have no data dependency (conditions are in
   cpu-runtime.md).
5. **DeepSeek-V4-specific skips.** Skip the compressor (CSA/LID) on the decode tokens that don't finish
   a compression block (3 of 4). Skip the indexer top-k when the cache has fewer entries than top-k.
6. **Single-KV-head (MQA/MLA) decode attention.** One pass over each shared K/V row serves all query
   heads, with an online softmax and FP32 accumulators.
7. **Prompt-lookup drafting with a verification-outcome memory.** This is everything the "Stockfish"
   drafter really does: a single-chain suffix match over the context, plus a small table of recently
   rejected continuations that can veto drafts. It is lossless because every draft is verified. Our
   n-gram drafter already does most of this. The rejection-memory veto is the only new idea.
8. **Persisting slot and prefix-cache state** to tmpfs for fast restore. The good idea is the persisted
   snapshot; don't use a world-writable location as upstream does.

### Not worth reimplementing

- **The Q2_0 "Fractal-64" format.** A 30-byte / 64-weight codebook format ({±1, ±4, ±8, ±14} with 4-bit
  sub-scales), quantized without an imatrix. It is a new incompatible format that breaks every non-x86
  backend, and existing K-quants or IQ formats at similar bpw are a better trade.
- **Lossy pruning modes** (expert-tail pruning, SwiGLU row skipping, the top-k override, sliding-window
  attention). Their quality is unvalidated (the "bit-exact" evals only compare the converted model with
  itself). If we ever offer them, they must be explicit opt-in flags.
- **HARP-Cache, EAKV, logit/linear probes, `rc_engine` tree search.** These are telemetry, simulators or
  dead code.
- **The "4-level cache tiling".** There are actually two levels and no L3 tiling. "In-register FWHT" is
  partly in-register on AVX-512 only. The "dual-CCD pipeline" runs serially.
