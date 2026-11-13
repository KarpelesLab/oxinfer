# Streaming weights from flash (models larger than RAM)

`--stream` (Qwen3.5 / Qwen3.8, Gemma 4 / Gemma 3 including 26B-A4B's experts, Mistral Large 4, on by default when its checkpoint does not fit in the device's memory, `--no-stream` / `stream=off` turning that off; implies `--mmap`) runs a model whose weights do not fit in memory
by reading them in place from the mapped file while a reader thread fetches the next layers. It
is aimed at phones: HauhauCS's Qwen3.8-27B GGUFs (Q2_K_P 10.7 GB, Q4_K_P 17.9 GB) on a 12 GB
iPhone. Every forward pass reads the units (decoder layers, LM head, MTP layer, draft head)
cyclically, so an LRU page cache smaller than the model never hits: each page is gone before its
next use. With `--cache-budget-gb F` oxinfer manages the cache itself instead: a fixed resident
set of `F` minus the read-ahead window stays cached (draft units first, since a round runs them
up to 8 times, then evenly spaced layers so the reader can work ahead while they compute) and
every other unit is dropped from the cache right after use. The page cache then holds the
resident set plus the window and nothing else, so each pass reads exactly the streamed bytes.

Darwin details that decided the design (measured on an M5 Max): a clean file page only leaves
the cache through `msync(MS_INVALIDATE)` (`madvise(MADV_DONTNEED / MADV_FREE)` leave it cached);
`fcntl(F_RDADVISE)` and `pread` read at 13-14 GB/s while demand faults from compute threads
manage 2-3 GB/s and `MADV_WILLNEED` ~5 GB/s, so the reader issues `F_RDADVISE` per 8 MB and then
touches the pages (mapping a 16 KB page costs ~1 us, so above ~7 GB/s the touching, not the SSD,
limits the reader); a full `mincore` costs ~15 ms per GB, so cached chunks are recognized from
three probe pages; `msync(MS_INVALIDATE)` itself costs ~50 ms per GB, so an evictor thread drops
used units instead of the compute thread (160 ms per pass on Q2_K_P at an 8 GB budget). Metal
wraps the mapped weights without copying for GPU prefill, but a live no-copy buffer keeps its
pages from being evicted, so streamed units get per-call buffers that are released after each
matmul (`Lin::set_transient_device`). A pass's embedding rows are read through the streamer
before the lookup (with priority over the read-ahead) rather than faulting in page by page.

| Option | Meaning |
|---|---|
| `--stream` | stream weights from the mapped file, reading ahead in execution order |
| `--cache-budget-gb F` | page cache the model may use (resident set = F - prefetch window; streamed units evicted after use) |
| `--resident-mb N` | resident set size instead of the budget's |
| `--prefetch-mb N` | read-ahead window (default 768) |
| `--resident-mode M` | `cache` (clean file pages, default), `lock` (`mlock`), `copy` (anonymous memory over the same addresses) |
| `--flash-mbps N` / `OXINFER_FLASH_MBPS` | emulate storage that reads N MB/s (the reader sleeps per uncached byte) |
| `--stream-io M`, `--stream-threads N` | `advise` (default on Apple), `pread` or `touch`; reader threads (default 2) |
| `--stream-trace` | one line per token / speculative round, and one per prefill: bytes read vs cache hits, stall time |
| `--stream-gpu M` | `auto` (default: dense models decode on the GPU, MoE models measure both on their first tokens and keep the faster), `on`, `off` (Qwen3.5 / 3.8, Gemma 4 / 3; engine / C API / server key `stream_gpu`) |
| `--expert-cache-mb N` | MoE: page cache kept for recently used routed experts (default: what the budget leaves after the resident set) |
| `--expert-predict` | MoE: guess the next layer's experts and read them ahead (off by default, see below) |
| `--threads N`, `--qos C`, `--efficiency-threads N` | compute threads; their QoS class on Apple platforms (`interactive` default, `initiated`, `utility`, `background` = efficiency cores only); how many of them run at background QoS |

With `--mtp`, streaming defaults to up to 8 drafts with a cost-aware draft length
(`--mtp-cost`): each round picks the length with the most expected tokens per second from the
measured round time at each length and the measured acceptance at each draft position. When a
pass streams 5-12 GB, verifying 9 tokens costs about what verifying 1 does, so long drafts pay
off; in RAM the same policy settles on 2-3. Long drafts stay exact: `--mtp` verifies up to
`draft + 1` tokens on the batch-invariant decode kernels (`MtpOptions::exact`,
`kernels::set_decode_max_tokens`; greedy output checked token for token against plain greedy
over 160 tokens with 8 drafts, embedded layer and FastMTP sidecar). The rewind after a partial acceptance replays
the delta rule instead of keeping per-token states, so an 8-draft round needs one extra copy of
the recurrent state (151 MB on Qwen3.8-27B) rather than eight.

Verification passes of 3-16 tokens on GGUF weights run multi-token kernels (`kernels_ggml.rs`)
that unpack each weight super-block once for all tokens of the pass; before, the decode kernels
ran in token pairs and unpacked every block once per pair, which on HauhauCS's files (Q2_K_P is
mostly IQ3_XXS and IQ3_S grid formats, Q4_K_P mostly Q5_K) made unpacking the main cost. There
are two variants, both repeating the one-token kernels' f32 operations in the same order (the
integer sums are exact either way), so every row stays bit-identical whatever the token count
and the speculative greedy output equals plain greedy; no exactness switch is needed:

- SDOT, one weight row at a time against all tokens: Q4_K, Q5_K, IQ4_XS, and every format
  without i8mm;
- SMMLA (Armv8.6 i8mm, detected at run time: A15 / M2 and later, so the A19 Pro and the M5) on
  2x2 tiles of rows and tokens, the activations interleaved per token pair, from 4 tokens on:
  the formats with integer scales per 16 (Q2_K, Q3_K, Q6_K, IQ2/IQ3/IQ1), where SDOT needs many
  extra instructions to apply the scales. On the M5's cores SMMLA issues at half SDOT's rate
  for twice the multiply-adds (5 vs 10.5 G instructions/s per core, ~9-cycle latency) and does
  not overlap other vector work, so it buys nothing for the per-32 f32-scale formats.

The blocks-of-32 formats (Q8_0, Q4_0, IQ4_NL, MXFP4, Q4_1, Q5_0, Q5_1) and oxinfer's Q4 run
their one-token kernels on column groups of up to 8 tokens instead (see
[Multi-token passes and MoE experts on the CPU](speculative-decoding.md#multi-token-passes-and-moe-experts-on-the-cpu));
`Pairs` turns that off too.

`kernels::set_multi_token_kernels(MultiTokenKernels::{Auto, Dot, Pairs})` (or
`OXINFER_MULTI=auto|dot|pairs`) picks them (`Pairs` is the previous behaviour, for comparisons);
`kernels::cpu_has_i8mm()` reports the feature. One core, a 4096 x 5120 matrix, ns per 256
weights per token (`examples/kernel_bench.rs`):

| Format | 1 token | 4 tokens: pairs / SDOT / auto | 9 tokens: pairs / SDOT / auto | 9 tokens speed-up |
|---|---|---|---|---|
| IQ3_XXS | 20.6 | 11.3 / 6.9 / 6.0 | 12.2 / 5.2 / 4.1 | 2.9x |
| IQ3_S | 23.1 | 12.5 / 7.7 / 7.0 | 13.6 / 5.4 / 4.8 | 2.8x |
| Q2_K | 10.3 | 7.3 / 5.5 / 5.5 | 7.6 / 4.9 / 4.5 | 1.7x |
| Q3_K | 10.7 | 7.6 / 5.4 / 5.0 | 8.0 / 4.3 / 4.0 | 2.0x |
| Q6_K | 10.3 | 6.8 / 5.0 / 4.7 | 7.2 / 4.2 / 3.8 | 1.9x |
| Q4_K | 6.2 | 4.5 / 3.7 / 3.7 | 4.8 / 3.3 / 3.3 | 1.5x |
| Q5_K | 9.7 | 6.0 / 4.6 / 4.5 | 6.6 / 3.7 / 3.7 | 1.8x |
| IQ4_XS | 6.8 | 4.5 / 3.5 / 3.5 | 4.8 / 3.0 / 3.0 | 1.6x |

Whole verification passes (recording forward plus the LM head over every token,
`examples/pass_bench.rs`, the old and new kernels alternating pass by pass, best of 5):

| Model | Threads | 1 token | 2 tokens | 4 tokens: old / new | 9 tokens: old / new |
|---|---|---|---|---|---|
| Q2_K_P | 3 | 0.54 s | 0.65 s | 1.77 / 1.00 s | 3.04 / 1.14 s (2.7x) |
| Q2_K_P | 6 | 0.50 s | 0.54 s | 0.75 / 0.43 s | 1.74 / 0.67 s (2.6x) |
| Q4_K_P | 3 | 0.30 s | 0.46 s | 0.99 / 0.76 s | 1.71 / 1.03 s (1.7x) |
| Q4_K_P | 6 | 0.16 s | 0.21 s | 0.40 / 0.31 s | 0.91 / 0.59 s (1.55x) |

(Best of 5; the machine was shared, so single passes varied by up to 40%. One- and two-token
passes run the same kernels as before.)

The activation quantization for several tokens now runs vectorized and in parallel, with the
same values as before.

MoE models (Zawen-v0.1-35B-A3B and other Qwen3.5/3.6 MoE checkpoints) stream only the experts a
pass routes to: a MoE layer's unit holds its attention / DeltaNet weights, the router and the
shared expert, and each routed expert is an on-demand part of it (`UnitSpec::parts`) that the
layer requests once routing is known (`StreamHook::need_parts`), read with priority over the
read-ahead and then kept in a part cache (least recently used out) sized by what the budget
leaves (`StreamOptions::part_cache_bytes`). Zawen in oxinfer Q4 (19.5 GB), 6 GB budget at
2 GB/s, 3 threads, plain greedy: 0.13 -> 7.3 tok/s (14.2 -> 0.12 GB read per token: the dense
part, 1.1 GB, is resident and 78% of the requested experts come from the 4.1 GB part cache),
same tokens as without streaming. Guessing the next layer's experts from the residual stream
(`--expert-predict`) hides part of the remaining stall (77% -> 44% of the time) but the wrong
guesses read 27% more bytes, which the flash cannot afford: 6.9 tok/s, so it is off by default.

Measured on the M5 Max with the phone's limits emulated (`--cache-budget-gb`, `--flash-mbps`;
greedy, a 40-token answer to a short prompt, CPU decoding; "3 threads" approximates the
iPhone's 2 performance + 4 efficiency cores), before the multi-token kernels. Pass times were
within a few percent between repeated runs even with other jobs on the machine, since the
emulated storage dominates:

| Model | Budget | Flash | Resident / streamed per pass | Plain | `--mtp` (cost-aware, up to 8) |
|---|---|---|---|---|---|
| Q2_K_P | 4 GB | 1 GB/s | 3.1 / 7.1 GB | - | 0.35 tok/s (2.60 tok/pass) |
| Q2_K_P | 6 GB | 2 GB/s | 5.2 / 4.9 GB | 0.39 tok/s | 0.90 tok/s; 3 threads 0.91; FastMTP sidecar 0.73 |
| Q2_K_P | 8 GB | 3 GB/s | 7.1 / 2.9 GB | 0.91 tok/s; 3 threads 0.89 | 2.07 tok/s; 3 threads 1.57 (compute-bound: picks 1-4 drafts) |
| Q4_K_P | 4 GB | 1 GB/s | 3.0 / 13.9 GB | 0.069 tok/s | - |
| Q4_K_P | 6 GB | 2 GB/s | 5.0 / 12.0 GB | 0.15 tok/s | 0.39 tok/s; 3 threads 0.39; sidecar 0.31; fixed 1-3 drafts 0.37 |
| Q4_K_P | 8 GB | 3 GB/s | 7.0 / 10.2 GB | - | 0.66 tok/s |

A pass takes 1.04-1.14x the time the emulated storage needs for its bytes (88-97% of the
bandwidth used), and the page cache held by the model stays at or under the budget (sampled
with `mincore` at exit). Tokens per pass depend on the text, not on the IO: with 8 drafts every
round, 2.4-2.5 on prose, 2.8-3.3 on an explanation and 4.3-4.5 on code (embedded MTP layer a
little ahead of the FastMTP sidecar everywhere), so long drafts multiply the IO-bound speed by
2.3-4x. Sampling (`--temp 0.7`) streams the same way (1.03 tok/s at 6 GB / 2 GB/s on Q2_K_P).
Without a budget (`--stream` alone, the model cached) streaming costs nothing: 10.5 vs 10.9 tok/s
for `--mmap` on Q4_K_P.

With no throttle the Mac's SSD streams a 10 GB pass in ~1.3 s (7-8 GB/s, bounded by mapping the
pages). Prefill streams the weights once per chunk: on the GPU (`--device metal`, mapped weights
wrapped without copying) a 1138-token prompt on Q4_K_P at 6 GB / 2 GB/s took 13.9 s, near the
12 s the two passes' IO needs, against 53.7 s on the CPU.

## GPU decoding with streamed weights (mixed GPU / CPU)

With Metal, streamed Qwen3.5 / 3.8 models (dense, MoE and Qwen3.8-Flash-Next) and Gemma 4 / Gemma 3
(dense, E2B / E4B with their per-layer embedding projection as a unit, 26B-A4B with its routed
experts) decode on the GPU too, single tokens and multi-token MTP / DSpark verification alike (`--stream-gpu auto|on|off`, engine and
C API key `stream_gpu`, `gpu::set_stream_decode`):

- The resident set is chosen as for the CPU (the units a round uses most first: the MTP layer,
  the LM head when drafts share it; then whole decoder layers evenly spaced, so the reader works
  ahead while they compute) and its weights are wrapped once, as no-copy Metal buffers over the
  mapping, the same wraps for the single- and multi-token plans. They stay wired.
- Every other layer is a streamed unit of the plan (`gpu::decode::Graph::units`, `UnitGate`): the
  step enters the unit (waiting for the reader if it is not there yet), wraps the unit's memory
  (a few buffers per layer, 0.05-0.2 ms each), encodes the layer into a command buffer of its own
  and commits it, then enters the next unit while the GPU runs this one. Once the GPU finished a
  unit its wraps are released and the unit is left, so the evictor can drop its pages; at most two
  streamed units are in flight. Runs of resident layers share one command buffer.
- Routed experts (MoE) stay on-demand parts of their layer. The command buffer ends at the expert
  op: the routing is read back, the chosen experts are read (or found in the part cache) and
  wrapped, and their addresses go into the op's tables. Wraps of the experts the part cache keeps
  are kept too (in the queue's residency set), and released when the cache lets the expert go.
- On the CPU: the MTP drafts (the MTP layer is resident), the embedding lookup, sampling; reading
  is the reader thread's.

Darwin / Metal facts this rests on (M5 Max, measured with small Swift programs): creating a no-copy
buffer touches nothing; the first command buffer that uses it wires the whole buffer (about 20 ms
per GB when the pages are cached; uncached pages are read in at 4-6 GB/s on the way), so units are
wrapped exactly, never as one buffer over the file. Wired pages survive `msync(MS_INVALIDATE)`:
after the buffer is released they stay wired for another 50-260 ms (a live buffer keeps them about
a second after its last use), so the evictor checks that a drop worked and tries again every 100 ms
(`msync` calls on still-wired pages slowed the reader). The reader no longer touches the pages of
units the GPU reads, since the GPU maps them itself (touching cost the reader ~60 ms per GB: 0.69
to 0.89 tok/s on Q2_K_P at 8 GB / 3 GB/s). Because of the release lag, the page cache briefly holds
up to ~0.25 s of reads (750 MB at 3 GB/s) beyond the budget.

Which device runs a step: dense models decode on the GPU. A MoE step waits for its routing once per
layer (40 round trips per token on Qwen3.5-35B-A3B, ~1 ms each with the GPU work they wait for),
which on the M5 Max costs about what decoding on the CPU does, so with `stream_gpu` unset these
models time their first steps on both (after two warm-up steps each, four measured) and keep the
faster (`gpu::decode::DeviceChoice`; it prints its decision). Starting a layer that is still being
read on the CPU was not done: a 9-token verification pass spends ~0.1 s computing on the GPU against
~1 s of reading at 3 GB/s, so there is nothing for the CPU to hide.

Exactness: a streamed plan runs the same kernels on the same weights as the resident one, so its
tokens are the resident GPU's. Over 200 greedy tokens of two prompts (code and prose) on Q2_K_P,
streamed GPU decoding at 8 and 4 GB budgets gave exactly the resident GPU's tokens, and so did MTP
with fast verification (the default) and with `--exact-verify` (streamed and resident). Qwen3.5-35B-A3B
streamed on the GPU at 6 GB gave the resident GPU's 200 tokens, and Qwen3.8-Flash-Next UD-Q4_K_XL at
8 GB / 3 GB/s the resident GPU's 12 tokens. CPU decoding, streamed or not, gives the CPU's tokens
(identical between the two); those can part from the GPU's at a near-tie, as without streaming (the
prose prompt did after some tokens on both models), and so can the first steps of a MoE model whose
device is being measured.

Measured on the M5 Max with the phone's limits emulated (`--cache-budget-gb`, `--flash-mbps`; not
on a phone), greedy, a short prompt, plain = 24 tokens (12 for Q4_K_P at 4 GB), MTP = 40 tokens
(cost-aware, up to 8 drafts, 3.25-3.55 tokens per pass), prompt lookup off, CPU decoding on 6
threads unless noted; load average 6-30 from other jobs. Resident / streamed: what the budget keeps
resident and what each pass reads (with MTP the MTP layer joins the resident set):

| Model | Budget / flash | Resident / streamed | Plain GPU / CPU | MTP GPU / CPU | MTP, 3 threads: GPU / CPU |
|---|---|---|---|---|---|
| Q2_K_P | 4 GB / 1 GB/s | 3.1 / 6.9 GB | 0.14 / 0.14 tok/s | 0.44 / 0.43 tok/s | |
| Q2_K_P | 6 GB / 2 GB/s | 5.2 / 4.9 GB | 0.37 / 0.36 | 1.13 / 1.15 | |
| Q2_K_P | 8 GB / 3 GB/s | 7.1 / 2.9 GB | 0.95 / 0.93 (3 threads 0.94) | 2.96 / 3.10 | 3.03 / 2.87 |
| Q4_K_P | 4 GB / 1 GB/s | 3.0 / 13.9 GB | 0.071 / 0.069 | 0.25 / 0.24 | |
| Q4_K_P | 6 GB / 2 GB/s | 5.0 / 12.0 GB | 0.154 / 0.147 | 0.55 / 0.51 | |
| Q4_K_P | 8 GB / 3 GB/s | 7.0 / 10.0 GB | 0.27 / 0.24 (3 threads 0.24) | 0.87 / 0.75 | 0.88 / 0.82 |

Every pass read what the plan says (6.9-7.0 / 4.5-5.1 / 2.9-3.1 GB for Q2_K_P, 13.7-13.9 / 11.9 /
9.9-10.0 GB for Q4_K_P) and took 1.0-1.15x the emulated reading time: on the Mac both devices are
flash-bound, the GPU slightly ahead on Q4_K_P. What differs is the compute left over: at 8 GB / 3
GB/s with MTP the GPU computed 0.10 s (Q2_K_P) / 0.18 s (Q4_K_P) per 9-token pass and waited for
the flash 89-98% of the time, the CPU 0.46 / 0.83 s on 6 / 3 threads for Q2_K_P (waiting 59 / 27%
of the time) and 1.5 s on 3 threads for Q4_K_P.
A phone's CPU, slower than three M5 Max cores, crosses the reading time there; its GPU stays under it
(see the projection below).

MoE: Qwen3.5-35B-A3B in oxinfer Q4 (20.9 GB; 1.1 GB of dense weights resident, 18.1 GB of routed
experts read on demand, the rest of the budget a part cache of recently used experts), plain
decoding, 64 tokens; Qwen3.8-Flash-Next UD-Q4_K_XL (4.5 GB resident: attention, Gated DeltaNet,
shared experts, routers, hyper-connections, LM head; 78 GB of routed experts on demand; its 29 GB
n-gram table is read a few rows per token from the mapping, outside the budget), 32 tokens. The page
cache at exit was within the budget in every run:

| Model | Budget / flash | Expert cache | Experts from the cache | GB read per token | GPU | CPU (6 / 3 threads) |
|---|---|---|---|---|---|---|
| 35B-A3B | 4 GB / 1 GB/s | 2.1 GB | 63% | 0.21 | 2.8 tok/s | 3.0 / 2.6 tok/s |
| 35B-A3B | 6 GB / 2 GB/s | 4.1 GB | 76% | 0.13-0.14 | 5.1 | 5.8 / 5.3 |
| 35B-A3B | 8 GB / 3 GB/s | 6.1 GB | 84% | 0.09 | 10.0 | 12.3 / 12.1 |
| Flash-Next | 6 GB / 2 GB/s | 0.65 GB | 0% | 1.2-1.5 | 0.8 | 1.0 / 1.0 |
| Flash-Next | 8 GB / 3 GB/s | 2.65 GB | 47% | 0.77-0.80 | 1.8 | 1.8 / 2.2 |

Here the CPU wins on the Mac: the GPU spends ~40 ms per 35B-A3B token beyond waiting for the flash
(40 routing round trips, wrapping newly read experts), the CPU ~20 ms on 6 threads and ~27 on 3,
which is why `stream_gpu` left unset measures both (it chose the CPU in every run). On a phone,
with a slower CPU, the measurement may go the other way. Kept expert wraps are released as soon as
the part cache lets the expert go (checked after every step): an earlier version released them
only when their layer came round again, which kept a token's worth of experts wired beyond the
budget (and looked faster: Flash-Next at 6 GB read 0.9 instead of 1.2-1.5 GB per token).

Gemma 4, same emulation, plain greedy decoding of a code prompt (12-48 tokens), prompt lookup off,
two interleaved rounds (each cell: round 1 / round 2; load average 2-15 from other jobs; CPU on all
18 threads, in parentheses on 6): gemma-4-31B-it in oxinfer Q4 (17.3 GB), ggml-org's
gemma-4-12B-it Q8_0 (12.7 GB), ggml-org's 26B-A4B Q4_0 (14.5 GB: 1.7 GB of dense weights, 13 GB of
routed experts read on demand, 30 layers):

| Model | Budget / flash | Resident / streamed | GPU | CPU | `auto` |
|---|---|---|---|---|---|
| 31B Q4 | 6 GB / 2 GB/s | 5.2 / 12.1 GB | 0.162 / 0.159 tok/s | 0.158 / 0.153 (0.154 / 0.153) | GPU |
| 31B Q4 | 8 GB / 3 GB/s | 7.1 / 10.1 GB | 0.279 / 0.290 | 0.270 / 0.278 (0.275 / 0.280) | GPU |
| 12B Q8_0 | 4 GB / 2 GB/s | 3.1 / 9.6 GB | 0.206 / 0.206 | 0.197 / 0.197 (0.200 / 0.201) | 0.208 / 0.207 (GPU) |
| 26B-A4B | 4 GB / 1 GB/s | 1.7 GB + 1.5 GB expert cache | 2.1 / 2.0 | 2.1 / 2.0 (2.1 / 2.1) | 2.1 / 2.1 (CPU) |
| 26B-A4B | 6 GB / 2 GB/s | 1.7 GB + 3.5 GB expert cache | 5.6 / 5.6 | 6.4 / 6.5 (6.4 / 6.4) | 6.8 / 6.7 (CPU) |
| 26B-A4B | 8 GB / 3 GB/s | 1.7 GB + 5.5 GB expert cache | 10.0 / 10.0 | 12.9 / 12.9 (12.7 / 12.6) | 12.6 / 12.6 (CPU) |

The dense models are flash-bound on both devices (stalled on the reader 97-100% of the time), the
GPU a few percent ahead; on a phone the CPU's share of a pass grows (as for Qwen above) while the
GPU's stays small. The 26B-A4B waits for its routing 30 times per GPU step (~50 ms per token
beyond the CPU's at 8 GB), so `auto` timed both on the first tokens and kept the CPU in every run
(GPU 125 / CPU 73 ms per token at 8 GB / 3 GB/s, 219 / 122 at 6 GB / 2 GB/s).

Gemma exactness, checked step by step with `OXINFER_GPU_DECODE_CHECK=1` (every GPU step also run
on the CPU) and a hash of both outputs per step: E4B (1.5 GB budget, its PLE projection a
streamed unit), 12B Q8_0 (4 GB), 31B Q4 (6 GB) and 26B-A4B (4 GB with every dense layer resident;
1.5 GB with 9 of its 31 units streamed), 32-48 steps each: the streamed GPU's output was
bit-identical to the resident GPU's at every step, the streamed CPU's to the resident CPU's, and
the greedy tokens matched. GPU and CPU differ as without streaming (KL up to 1e-2; on E4B the 19th
token differed). The 31B with a DSpark draft (`--draft` RedHatAI speculator) verifying on the GPU
while streamed, with fast and with `--exact-verify` verification, gave the plain resident GPU's
tokens.

With the CLI defaults for a streamed dense model (MTP, prompt lookup on top) a code prompt on Q2_K_P
at 8 GB / 3 GB/s decoded 63 tokens at 5.7 tok/s, 6.3 tokens per pass (lookup found nothing to copy
from a fresh prompt; the same without it).

Defaults from the device (`stream::DeviceMemory`, `stream::auto_stream`): a Qwen or Gemma
checkpoint streams without being asked when it is larger than the device holds, and the budget is
picked from the device. Where the OS limits apps (iOS: `os_proc_available_memory`, which the
increased-memory-limit entitlement raises), the budget is what the app may still use minus 1 GB
(KV cache, recurrent states, activations, the app itself), at most 60% of the device's memory; on
other systems streaming starts above 60% of the memory and the budget is 70% of it. With GPU
decoding the budget is also capped at 85% of Metal's `recommendedMaxWorkingSetSize`, since the
resident layers stay wired for the GPU. A streamed dense Qwen with an MTP layer drafts with it by
default (up to 8 drafts, cost-aware; `--no-draft` / `draft=none` for plain decoding), MoE models do
not (their verification reads the experts of every drafted token). So on a 12 GB iPhone with 6 GB
left to the app and an 8 GB working set, Qwen3.8-27B Q2_K_P loads with a 4.9 GB budget (4.0 GB of
it resident), GPU decoding and MTP, without options; `OXINFER_DEVICE_MEM_GB`, `OXINFER_APP_MEM_GB`
and `OXINFER_GPU_WORKING_SET_GB` override what is detected (to try a phone's defaults elsewhere), and
`oxi_engine_info_json` reports both the device figures and the streaming plan. In the iOS Simulator
(the test app in `ios/`, those three variables set to 12 / 6 / 8 GB through `SIMCTL_CHILD_`), the
27B loaded that way and answered correctly ("Hello there, how are you?"; a functional check only:
the Simulator's GPU copies the wrapped buffers and runs at 1 token/s).

Projected for an iPhone 17 Pro Max (not measured on a device: no iPhone was available), assuming
sustained sequential reads of 1-3 GB/s (iPhone NVMe reads are usually quoted around 1.5-2.5 GB/s),
a page-cache budget of 4-8 GB (the 12 GB device minus the OS, the app's own memory and other apps),
the measured 90% bandwidth efficiency and the bytes per pass and 3.25 tokens per MTP pass of the
emulated runs above (tokens per pass depend on the text: 2.4-4.5). A pass then takes the larger of
its reading time and its compute time. Compute on the phone is an estimate: GPU decoding ~0.4 s per
9-token pass (the A19 Pro's GPU at a fifth to a sixth of the M5 Max's, which needs 0.10-0.18 s, and
LPDDR5X bandwidth for the resident weights), CPU decoding ~1.5 s (Q2_K_P) / 2.5 s (Q4_K_P) per
9-token pass (the M5 Max's 3 cores need 0.83 / 1.5 s; 2 performance + 4 efficiency cores assumed
1.7x slower):

| Budget | Q2_K_P plain / `--mtp`, 1 GB/s | 2 GB/s | 3 GB/s | Q4_K_P plain / `--mtp`, 1 GB/s | 2 GB/s | 3 GB/s |
|---|---|---|---|---|---|---|
| 4 GB | 0.13 / 0.41 | 0.26 / 0.82 | 0.39 / 1.24 | 0.065 / 0.21 | 0.13 / 0.41 | 0.19 / 0.62 |
| 6 GB | 0.19 / 0.57 | 0.37 / 1.15 | 0.56 / 1.72 | 0.075 / 0.24 | 0.15 / 0.48 | 0.22 / 0.72 |
| 8 GB | 0.31 / 0.94 | 0.62 / 1.89 | 0.93 / 2.83 (CPU: 2.17*) | 0.09 / 0.29 | 0.18 / 0.57 | 0.27 / 0.86 |

(tokens/s with GPU decoding; * where CPU decoding would be compute-bound: at 8 GB / 3 GB/s a pass
reads in 1.15 s, which the GPU's ~0.4 s keeps up with and an estimated 1.5 s of CPU verification
does not. Everywhere else both are flash-bound and equal.) Q2_K_P is the practical choice: about 1
token/s with 6 GB at 2 GB/s and 2-3 with 8 GB at 3 GB/s, with MTP (the default when streaming).

With the multi-token kernels (same prompt, 40 tokens, base = the commit before them; best of
two interleaved rounds on a machine shared with other jobs, which also read the same model files
and so sometimes lent cache hits to the IO emulation, mostly on Q4_K_P):

| Model | Budget / flash | MTP | base tok/s (s per pass) | new tok/s (s per pass) | GB read per pass |
|---|---|---|---|---|---|
| Q2_K_P | 6 GB / 2 GB/s | embedded | 0.82 (2.98) | 0.91 (2.84) | 5.1 |
| Q2_K_P | 6 GB / 2 GB/s | FastMTP sidecar | 0.74 (3.11) | 0.84 (2.90) | 5.2 |
| Q2_K_P | 8 GB / 3 GB/s | embedded | 1.39 (1.76) | 1.71 (1.43) | 3.1 |
| Q2_K_P | 8 GB / 3 GB/s | FastMTP sidecar | 1.19 (1.92) | 1.93 (1.27) | 3.2 |
| Q2_K_P, 6 threads | 8 GB / 3 GB/s | embedded | 1.75 | 2.00 | 3.1 |
| Q4_K_P | 6 GB / 2 GB/s | embedded | 0.46 | 0.40 | 10.0-11.9 |
| Q4_K_P | 6 GB / 2 GB/s | FastMTP sidecar | 0.34 | 0.31 | 9.6-12.1 |
| Q4_K_P | 8 GB / 3 GB/s | embedded | 0.77 | 0.67 | 7.3-9.3 |
| Q4_K_P | 8 GB / 3 GB/s | FastMTP sidecar | 0.54 | 0.54 | 8.6-9.0 |

(The Q4_K_P differences follow the bytes read per pass, which varied with the cache hits the
other jobs' reads of the same file lent: base 7.3 GB per pass against new 8.5 GB in the 8 GB
embedded runs, for example.)

Where the flash is the bottleneck (6 GB at 2 GB/s, and every Q4_K_P setting) nothing changes:
a pass already took about the time its bytes need. Where the CPU was (Q2_K_P at 8 GB / 3 GB/s)
the faster verification shows up directly, and the cost-aware policy drafts longer.
Greedy output was the same 40 tokens in every run (plain, embedded MTP layer, FastMTP sidecar,
old and new kernels), and `OXINFER_CHECK_SPEC` (every verification compared with one-token
forwards) found no difference over a 77-token answer with 8 drafts every round (embedded layer
and sidecar, Q2_K_P, which runs both kernel variants).

Phones have performance and efficiency cores (A19 Pro: 2 + 4). The pool hands out work block by
block, so a pool spanning both kinds lets the slower cores take fewer blocks; `--threads 6`
(the default `interactive` QoS, which iOS schedules on both kinds) is the setting to start from.
`pool::PoolOptions { threads, qos, efficiency_threads }` (`pool::init_with`, engine keys
`threads`, `qos`, `efficiency_threads`) sets the QoS class and keeps chosen workers on the
efficiency cores; `pool::cpu_cores()` reports the core counts. On the Mac (6 + 12 cores, no
efficiency cores in the phone's sense, loaded by other jobs) background-QoS threads were starved
by the other work (a 9-token Q2_K_P pass: 0.63 s on 6 interactive threads, 1.2 s with 4 of them
at background QoS, 6 s with all six), so the phone's balance has to be measured on the phone.

From the library (everything above is reachable without the CLI):

```rust
use oxinfer::qwen_mtp::{MtpDecoder, MtpOptions};
use oxinfer::stream::{StreamOptions, attach_qwen};

let files = ModelFiles::open(&path, false)?; // false: memory-mapped
let mut model = QwenModel::load(QwenConfig::load(&path)?, &files, Quant::Q4)?;
let mut mtp = model.load_mtp(&files)?;
let opts = StreamOptions { cache_budget: Some(6_000_000_000), ..Default::default() };
let streamer = attach_qwen(&mut model, Some(&mut mtp), &[&files], 8, opts)?;
let mut dec = MtpDecoder::new(&model, &mtp, MtpOptions { draft_tokens: 8, cost_aware: true, exact: true, stop, ..Default::default() });
// ... prefill / push, then per step:
let before = streamer.stats();
let tokens = dec.step(&mut sampler, max_new);
let io = streamer.stats() - before; // bytes_read, bytes_cached, stall, io_busy, ...
```

`Streamer::plan()` reports the resident set and the bytes streamed per pass (and any weight that
is a RAM copy instead of a view of the file, and the on-demand expert bytes and their cache),
`cached_bytes()` the model's current page-cache footprint, `set_flash_bytes_per_sec` changes the
emulated bandwidth while running, `StreamStats` also counts expert requests and part-cache hits
and the eviction time, and `MtpStats::costs` holds what the draft policy measured. The engine
API takes the same settings as keys (`expert_cache_mb`, `expert_predict`, `qos`,
`efficiency_threads`).
