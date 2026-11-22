# Performance

Decode is memory-bandwidth bound: every generated token streams all dense weights once.
On an M5 Max (18 cores, ~280 GB/s measured CPU read bandwidth):

| Model | Precision | Weights per token | Decode | Decode + drafter (prose / code) | Prefill (3k prompt) |
|---|---|---|---|---|---|
| E2B-it | bf16 | 4.6 GB | ~50 tok/s (≈230 GB/s) | n/a (no E2B drafter tested) | ~280 tok/s |
| E2B-it | `--quant q8` | 2.6 GB | ~86 tok/s | | ~255 tok/s |
| E2B-it | `--quant q4` | 1.3 GB | ~130 tok/s | | |
| E4B-it | bf16 | 9.3 GB | ~28 tok/s (≈245 GB/s) | 41 / 47 tok/s | ~135 tok/s |
| E4B-it | `--quant q8` | 5.2 GB | ~45 tok/s | 66 / 82 tok/s | ~135 tok/s |
| E4B-it | `--quant q4` | 2.6 GB | ~76 tok/s | 75 / 99 tok/s | ~135 tok/s |
| Qwen3.8-27B | bf16 | 51 GB | ~5.3 tok/s (≈270 GB/s) | | ~19 tok/s |
| Qwen3.8-27B | `--quant q8` | 29 GB | ~8.8 tok/s | `--mtp`: 1.5x / 2.0x | ~19 tok/s |
| Qwen3.8-27B | `--quant q4` | 14 GB | ~16 tok/s | `--mtp`: 1.1-1.2x / 1.6x | ~20 tok/s |
| Gemma 4 26B-A4B (MoE) | bf16 | 7.6 GB | ~30 tok/s (≈230 GB/s) | | ~150 tok/s (1.8k prompt) |
| Gemma 4 26B-A4B (MoE) | `--quant q8` | 4.3 GB | ~48 tok/s | | ~140 tok/s |
| Gemma 4 26B-A4B (MoE) | `--quant q4` | 2.1 GB | ~69 tok/s | | ~136 tok/s |

For the 27B model `--mmap` is recommended: quantization then reads the checkpoint straight from
the page cache (about 3 s to quantize 27.8B parameters on 18 threads) instead of copying 55 GB
into the process first.

With the Metal backend (default on Apple Silicon) prefill runs on the GPU:

| Model | Precision | Prefill, CPU only | Prefill, Metal |
|---|---|---|---|
| E4B-it | `--quant q4` | ~97 tok/s | ~580 tok/s |
| Qwen3.8-27B | `--quant q4` | ~18 tok/s | ~96 tok/s (256-token chunks) |
| Gemma 4 26B-A4B (MoE) | bf16 / `--quant q4` | ~150 / ~136 tok/s | ~544 / ~576 tok/s (1758-token prompt) |

(616-token prompt.)

CPU decode attention (`attn.rs`). When a decode step has fewer (token, KV head) pairs than threads
(single-token steps of GQA models, one shared key-value head as in DeepSeek V4, Mistral Large 4's
MLA latent cache, short speculative verification passes), attention runs in three parallel phases
that each read a shared key / value row once for all the query heads sharing it: scores over
blocks of keys (one key load serves 8 heads), the softmax per head, then the weighted values over
column slices of the head width (a 4-head x 16-column register tile on NEON). Every (query, head)
keeps the arithmetic of the per-head loop it replaces: the same 8-lane dot products summed in the
same order, the same softmax, and outputs accumulated from zero in key order element by element.
Results are therefore bit-identical to it. That was checked on the real models: last-token logits
after prompts fed 1 or 4 tokens at a time (3067 tokens on DeepSeek V4 Flash, 3172 on Qwen3.8-Flash-Next,
855 on Qwen3-8B) compare equal byte for byte with the previous kernels. The online softmax of
llama.cpp-style MQA kernels would change the summation order, so it is not used. Interleaved
runs, greedy tokens identical to the previous build in every run (load average 18-40 from other
jobs, so +-15%):

| model, device | context | decode before | decode after |
|---|---|---|---|
| DeepSeek V4 Flash UD-Q2_K_XL, Metal (CPU attention) | 857 | 8.1-9.3 tok/s | 9.9-10.0 tok/s |
| | 4191 | 6.5-7.9 | 8.3-8.8 |
| | 16076 | 5.4-6.5 | 8.2-10.3 |
| Qwen3.8-Flash-Next UD-Q4_K_XL, CPU | 863 | 17.6-18.3 | 18.6-19.0 |
| | 4338 (QSA sparse) | 13.6 | 17.7 |
| Qwen3.8-Flash-Next, Metal (CPU decode past 2051 tokens) | 4338 | 11.0-11.2 | 11.0-15.5 |
| Qwen3-8B Q8_0, CPU (8 KV heads) | 855 | 20.1-20.3 | 19.3-20.3 |
| | 4291 | 9.0-10.5 | 9.6-12.6 |

## Decoding on the GPU

With the Metal backend, single-token decode steps also run on the GPU, the whole forward pass in
one command buffer per token (see [Platforms](platforms.md#platforms-and-accelerators) for what runs where).
Measured on an M5 Max, 256 greedy tokens after a short prompt, against the same build with
`--gpu-decode off` (GPU prefill, CPU decode; interleaved runs, load average ~5-11):

| Model | Weights | CPU decode | GPU decode | llama.cpp Metal |
|---|---|---|---|---|
| Llama-3.2-1B-Instruct | GGUF Q4_K_M | 194-205 tok/s | 391 tok/s | 390-407 tok/s |
| Llama-3.2-1B-Instruct | GGUF Q8_0 | 148-156 tok/s | 284 tok/s | 269-285 tok/s |
| Orpheus 3B (text) | GGUF Q4_K_M | 87-94 tok/s | 161-168 tok/s | 176-189 tok/s |
| Orpheus 3B (`--speak`) | GGUF Q4_K_M | 99 tok/s | 191 tok/s (RTF 0.46) | |
| Qwen3.5-0.8B | GGUF Q8_0 | 179-182 tok/s | 356-361 tok/s | 282-284 tok/s |
| Qwen3.5-0.8B | `--quant q8` | 196 tok/s | 351 tok/s | |
| Qwen3.8-27B (HauhauCS) | GGUF Q4_K_P | 11.9-12.0 tok/s | 20.7-20.9 tok/s | 22.9-23.7 tok/s |
| Zawen-v0.1-35B-A3B (MoE) | GGUF Q4_K_M | 60 tok/s | 137-140 tok/s | 105-116 tok/s |
| Zawen-v0.1-35B-A3B (MoE) | GGUF Q8_0 | 47-48 tok/s | 111-114 tok/s | 85-89 tok/s |
| Zawen-v0.1-35B-A3B (MoE) | Q4 (`--save-quantized`) | 70 tok/s | 152 tok/s | |
| Gemma 4 E2B-it | Q4 (`--save-quantized`) | 134 tok/s | 201 tok/s | |
| Gemma 4 E4B-it | Q4 (`--save-quantized`) | 61 tok/s | 115 tok/s | |
| Gemma 4 26B-A4B (MoE) | GGUF Q4_0 | 63 tok/s | 130 tok/s | 115-116 tok/s |
| Inkling-2-Mini (MoE, looped; own engine, docs/inkling.md) | GGUF Q8_0 | 17 tok/s | 59 tok/s | 38 tok/s |
| Inkling-2-Mini | GGUF AD-Q4_K | | 62 tok/s | 41 tok/s |
| Inkling-2-Mini | release (NVFP4 experts) | | 56 tok/s | |
| Inkling-2-Mini | mlx-community 4bit | | 70 tok/s | |

Long contexts cost attention time (the KV cache stays f32): Llama-3.2-1B Q4_K_M decodes at 297
tok/s after a 2.5k-token prompt (CPU decode 139). The first step of a model compiles its plan
(~0.1-0.2 s); a multi-token pass (the prompt) builds it ahead.

Agreement with the CPU path (`OXINFER_GPU_DECODE_CHECK=1` runs every GPU step on the CPU too and
prints the KL divergence of the two next-token distributions): on bf16 weights the two match to a
KL of ~1e-12 (Llama 3.2 1B, Qwen3 0.6B, Qwen3.5-0.8B, Gemma 3 1B, Gemma 4 E2B / E4B / 26B-A4B, a
small Qwen3.5 MoE), with long contexts and split attention included. On quantized weights the CPU
kernels quantize the activations to int8 while the GPU keeps them f32, so the paths differ by a
mean KL of 1e-4 to 6e-4 (dense) and up to 3e-3 (MoE, whose routing amplifies it); with 7 prompts
of 128 tokens on Qwen3.8-27B Q4_K_P, 6 greedy texts were identical to the CPU's and one parted at
token 26. Against llama.cpp Metal the first 64 greedy tokens were identical on every model and
prompt above but Zawen Q4_K_M prose (both parted at token 25, as they did before). Orpheus GGUF
speech made on the GPU transcribes back exactly with whisper-small.

On GPUs whose cores have matrix units ("neural accelerators": M5, A19) the prefill GEMMs and the
MoE expert layers run on them (Metal 4 tensor operations, see
[Platforms](platforms.md#platforms-and-accelerators)). Measured on an M5 Max with a ~3k-token prompt (default
1024-token chunks; best of two interleaved runs on a shared machine), against the f32 GEMM of the
same build (`OXINFER_METAL_MMA=0`):

| Model | Precision | Metal, f32 GEMM | Metal, matrix units |
|---|---|---|---|
| E2B-it | bf16 | 631 tok/s | 724 tok/s |
| E4B-it | Q4 (`--save-quantized`) | 420 tok/s | 520 tok/s |
| Qwen3.8-27B | Q4 (`--save-quantized`) | 109 tok/s | 165 tok/s |
| Qwen3.8-27B (HauhauCS GGUF) | Q4_K_P | 93 tok/s | 158 tok/s |
| Zawen-v0.1-35B-A3B (MoE) | `--quant q4 --mmap` | 165 tok/s | 353 tok/s |
| Zawen-v0.1-35B-A3B (MoE) | `--quant q8 --mmap` | 161 tok/s | 255 tok/s |
| Zawen-v0.1-35B-A3B (MoE) | Q4 (`--save-quantized`) | 168 tok/s | 379 tok/s |

The GEMMs themselves run 3-4x faster (table under Platforms); what is left of prefill time is
mostly the CPU part of the layers (Gated DeltaNet recurrence, attention, routing). Against the CPU
path the next-token distributions differ by a KL divergence of 1e-7 to 2e-5 nats on the dense
models and 1e-4 to 1e-3 on Zawen (whose top-8 routing amplifies small differences), with the same
top 10 tokens in every case; a 128-token greedy generation of the 27B GGUF with every pass on the
matrix units matched the CPU's token for token.

The matmul kernels reach the machine's bandwidth ceiling with 12 threads
(`cargo run --release --bin membench` prints the raw numbers for your machine), so decode
speed is set by bytes per token: Q8 cuts them by 1.8x and Q4 by 3.5x. Prefill is compute-bound
on an 8x8 outer-product GEMM at roughly 1.3 TFLOP/s (the kernel itself runs at ~90% of FMA peak
when operands are in L1; a 12-token tile was tried and lost to register pressure). Quantized
prefill dequantizes to f32 and uses the same kernel, so prompt processing speed is independent
of precision.

## Comparison with llama.cpp

(Speculative decoding with DSpark drafts against llama.cpp master: see
[DSpark drafts for Qwen and Gemma 4](qwen.md#dspark-drafts-for-qwen-and-gemma-4).)

#### llama-bench matrix (November 2026)

`llama-bench` (llama.cpp master `fe14cad`, 2026-11-13, Metal build; `-p 512,2048 -n 128 -r 3`,
CPU rows `-ngl 0 -dev none` with its default 6 threads) against `examples/parity_bench.rs` on the
same GGUF files (the engine API, warm like llama-bench: a warm-up pass of each size, then the
median of 3 runs, 1 for the CPU rows of the larger models; raw text cut to exactly N tokens with
the BOS, as llama-bench counts them; plain decoding: `lookup=0 draft=none mtp=0`; CPU rows
`device=cpu` with its default 18 threads and `prefill_gemm=sme`, the default when measured (now
`exact`; the default is `fast`'s rounding, the column below): see
[CPU prefill GEMMs](#cpu-prefill-gemms-november-2026)). M5 Max shared with other jobs; the engines
interleaved model by model. The Metal columns were measured again on 2026-11-15 (load averages
5-17; the GPU's clock followed the other jobs, so the same pass varied by up to 2x over an hour and
only interleaved runs compare); the CPU columns were re-measured after the SME GEMM at load
averages 11-31 (Llama 1B Q4_K_M at 2-4), so they vary by +-20%. tok/s, llama.cpp / oxinfer:

| model | Metal pp512 | pp2048 | tg128 | CPU pp512 | pp2048 | tg128 |
|---|---|---|---|---|---|---|
| Llama-3.2-1B Q4_K_M (bartowski) | 17677 / 15039 | 16290 / 14476 | 419 / 388 | 642 / 953 | 534 / 746 | 167 / 240 |
| Llama-3.2-1B Q8_0 | 17580 / 15587 | 14935 / 14172 | 305 / 298 | 736 / 852 | 517 / 599 | 164 / 148 |
| Orpheus 3B Q4_K_M | 5625 / 4957 | 4990 / 4638 | 191 / 179 (22 tok) | 208 / 308 | 179 / 216 | 56 / 95 (24 tok) |
| Qwen3-8B Q8_0 | 2345 / 2110 | 2225 / 1761 | 58.1 / 55.6 | 107 / 115 | 97 / 91 | 28.9 / 26.6 |
| Qwen3-14B Q8_0 | 870 / 940 | 828 / 813 | 27.8 / 27.6 | 57 / 61 | 48 / 51 | 15.8 / 16.1 |
| Qwen3.5-9B BF16 (unsloth) | 2126 / 2064 | 1466 / 1807 | 27.7 / 31.5 | 97 / 135 | 103 / 125 | 1.4 / 16.4 |
| Gemma 4 12B Q8_0 (ggml-org) | 1292 / 1176 | 1189 / 1116 | 34.4 / 33.6 | 67 / 80 | 63 / 81 | 17.6 / 17.2 |
| Gemma 4 26B-A4B Q4_0 (ggml-org) | 3760 / 3128 | 2597 / 2475 | 98 / 116 | 150 / 162 | 128 / 148 | 58.6 / 53.4 |
| Qwen3.8-27B Q4_K_P (HauhauCS) | 484 / 519 | 492 / 497 | 19.1 / 17.5 | 20.8 / 31.4 | 20.4 / 31.2 | 7.1 / 10.5 |

CPU prefill with llama.cpp on more threads (its best is 12-16 on this machine; 18 drops sharply)
and oxinfer's `prefill_gemm=fast`, now the default for language models (same runs as the CPU columns above; tok/s,
pp512 / pp2048):

| model | llama.cpp 6 thr | 12 thr | 16 thr | oxinfer `auto` (SME) | `fast` | `f32` (before) |
|---|---|---|---|---|---|---|
| Llama-3.2-1B Q4_K_M | 642 / 534 | 1072 / 900 | 1335 / 1173 | 953 / 746 | 1289 / 920 | 405 / 327 |
| Llama-3.2-1B Q8_0 | 736 / 517 | 1313 / 1014 | (noisy) | 852 / 599 | 1066 / 827 | 416 / 323 |
| Orpheus 3B Q4_K_M | 208 / 179 | 320 / 269 | 211 / 306 | 308 / 216 | 403 / 215 | 124 / 87 |
| Qwen3-8B Q8_0 | 107 / 97 | 179 / 154 | 214 / 188 | 115 / 91 | 194 / 135 | 55 / 44 |
| Qwen3-14B Q8_0 | 57 / 48 | 69 / 68 | 105 / 83 | 61 / 51 | 79 / 78 | 29 / 35 |
| Qwen3.5-9B BF16 | 97 / 103 | 106 / 104 | 104 / 88 | 135 / 125 | 121 / 117 | 71 / 64 |
| Gemma 4 12B Q8_0 | 67 / 63 | 124 / 105 | 151 / 128 | 80 / 81 | 134 / 124 | 48 / 40 |
| Gemma 4 26B-A4B Q4_0 | 150 / 128 | 255 / 221 | 348 / 267 | 162 / 148 | 160 / 145 | 101 / 93 |
| Qwen3.8-27B Q4_K_P | 20.8 / 20.4 | 31.4 / 30.6 | 32.7 / 19.8 | 31.4 / 31.2 | 53.0 / 47.0 | 18 / 19 |

(`f32`: the previous default, measured in an earlier pass at load averages 8-45. Llama 1B
Q4_K_M from a pass at load 2-4. `fast` changes nothing for float-stored weights and runs the
MoE experts' grouped GEMMs on SME alone.)

Before this round (master `779396c`), Metal prefill ran only the matmuls on the GPU, with
attention and norms on the CPU and a host round trip per layer: pp512 / pp2048 were 1628 / 1017
(Llama 1B Q4_K_M), 392 / 202 (Orpheus), 282 / 174 (Qwen3-8B), 205 / 129 (Qwen3-14B), 646 / 337
(Qwen3.5-9B), 501 / 345 (Gemma 4 12B), 601 / 432 (26B-A4B), 154 / 116 (27B), measured under a
heavier load (18-38). Now every model with a GPU decode graph prefills on whole-pass prefill
plans with flash attention ([Decoding on the GPU](#decoding-on-the-gpu)): 2-12x faster, at or
above llama.cpp on Qwen3-8B / 14B / 3.5-9B pp512 and the 27B pp2048, 0.6-0.9x on the rest
(the 2026-11-13 run; see below for the second round). Decoding is level on Metal (the 27B's runs
varied with the page cache, other jobs held 100+ GB models) and on the CPU within noise (ahead on
Llama 1B Q4_K_M, Orpheus and the 27B, 5-10% behind on Qwen3-8B, Llama 1B Q8_0 and 26B-A4B in the
loaded CPU pass); llama.cpp's CPU decode of the BF16 Qwen3.5-9B (Gated DeltaNet) runs at 1.4 tok/s.

The second round of Metal prefill work (the run above, against master `9b48923` interleaved in
it: pp512 / pp2048 were 12361 / 12148, 12318 / 11893, 4944 / 4326, 1598 / 1573, 771 / 721,
2162 / 1346, 972 / 1041 (Gemma at 513 / 2049 tokens), 2624 / 2327 and 513 / 494 in the table's
order): a short last token tile runs as 32-row tiles instead of a padded 128-row one (Gemma's
513 tokens took 640 rows, 25% more GEMM work), narrow outputs take 64-row tiles where they had
32-row ones (Llama 1B pp512 +13%), Q8_0 and Q4_0 weights are dequantized in half arithmetic (the
same values; GEMM kernels +5-15%), residual adds happen in the GEMM stores, the MoE routers'
f32 GEMM has 8x the threadgroups, and each layer's new keys and values are copied to the host
caches while the GPU runs the later layers (that copy had been 3-9% of a pass). Logits are
bit-identical to master's and greedy output identical on all nine models (2 prompts x 64
tokens). Alternating llama-bench and oxinfer twice more on four models (pp512 / pp2048 ratios):
Qwen3-8B 0.79-0.85 / 0.92-0.94, Gemma 4 12B 0.90-0.93 / 0.95-0.96, Llama 1B Q4_K_M 0.80-0.88 /
0.70-0.88, 26B-A4B 0.78-0.79 / 0.96-1.01; when the GPU ran at full clock, Gemma 4 12B pp512
was 1600 / 1760-1780.

The third round (November 22; llama.cpp master of 2026-11-21, `llama-bench -p 512,2048 -n 0 -r 3`
interleaved model by model with `parity_bench`; tok/s pp512 / pp2048, llama.cpp / oxinfer before
/ oxinfer after; the before and after columns are runs half an hour apart at load averages 5-17
and 8-16 with the GPU shared, so they compare only roughly; the per-pass timings below are the
firmer measure):

| model | llama.cpp | before | after |
|---|---|---|---|
| Llama-3.2-1B Q4_K_M | 17698 / 16467 | 14143 / 14979 | 15575 / 13275 |
| Llama-3.2-1B Q8_0 | 18302 / 16993 | 16080 / 15354 | 15040 / 9806 |
| Orpheus 3B Q4_K_M | 6193 / 5241 | 5541 / 5115 | 4854 / 3668 |
| Qwen3-8B Q4_K_M | 2690 / 2359 | 2410 / 2258 | 1826 / 1705 |
| Qwen3-8B Q8_0 | 2769 / 2368 | 2599 / 2180 | 2140 / 1647 |
| Gemma 4 12B Q8_0 | 1800 / 1553 | 1644 / 1371 | 1261 / 1173 |
| Gemma 4 26B-A4B Q4_0 | 4095 / 3316 | 3345 / 3119 | 3430 / 2261 |

(the after run caught heavier GPU sharing: llama.cpp's own Orpheus pp512 moved from 6193 to 3678
between them). Alternating the two oxinfer builds pass by pass instead (shortest 512-token
plan step of six, GPU time where the step is one command buffer): Llama 1B Q4_K_M 31.3 -> 30.1 ms
GPU, the step 35.3 -> 34.1 ms; Orpheus 87.1 -> 85.3 ms; Qwen3-8B Q4_K_M 186.2 -> 182.8 ms;
26B-A4B 161.4 -> 149.1 ms (a 64-token prompt 98-110 -> 79-81 ms); Q8_0 / Q4_0 dense models
unchanged. What changed:

- Q6_K blocks (210 bytes, 2-byte aligned) are read as 16-bit words, Q5_K blocks as words (as
  Q4_K's were): the same values, so logits are bit-identical; `mma_gemm_bench` Q6_K 33 -> 38-43
  TFLOP/s (64-row tiles 27 -> 34), Q5_K +10-30%. Q4_K_M checkpoints keep their FFN down
  projections and some value projections in Q6_K, a third of Llama 1B's GEMM work.
- MoE layers of prefill plans are routed on the GPU (`pf_route`: one threadgroup counts each
  expert's rows, cuts the tiles as `group_tiles` does and places the rows in token order; the
  grouped GEMMs, gathers and activations take their grid sizes from it by indirect dispatch), so a
  pass is no longer cut into a command buffer per MoE layer with the host routing in between
  (~0.7 ms each). Same layout as the host's, so the same logits bit for bit
  (`OXINFER_PF_GPU_ROUTE=0` keeps the host routing).
- The engine computes a prompt's first logits with the GPU decode steps' matrix-vector kernels
  (`logits_device`) instead of the CPU's (1-4 ms for Llama's 128k vocabulary, more under CPU load;
  llama.cpp's prompt timing includes its GPU head).

Measured and dropped: K-quant dequantization in half arithmetic (Q4_K, Q5_K, Q6_K: no faster than
f32 with word reads, within 1%, and relative differences of ~4e-4, so the exact path stays the
only one); `relaxed_precision` matmul descriptors (no change); f32 activations read straight
into the matmul with relaxed precision, as llama.cpp does (8-10% slower than the half rows, which
cost an `x_half` pass of ~3% of a pass); double-buffered weight tiles with one barrier per K step
(mixed, no gain); 128-row tiles for narrow Q6_K outputs (slower in the pass than the 64-row ones);
fusing the half conversion into the RMSNorm before the GEMMs (+0.7 ms per Llama 1B pass) and into
the GLU (no change). Same-session `test-backend-ops` puts llama.cpp's own Q8_0 GEMM at ~35
TFLOP/s on 4096 x 512 x 14336 against oxinfer's ~45 for the same shape.

The fourth round (November 22) first accounted for a pass's time, per kind of op, in both engines
on the same 512-token prompt. llama.cpp (master of 2026-11-21) was patched for it in a scratch
copy: `GGML_METAL_OP_PROFILE` runs every (fused) graph node in a command buffer of its own and
adds up the GPU times by op and weight shape, as oxinfer's `OXINFER_PF_PROFILE` does, and
`GGML_METAL_CB_TRACE` prints each graph's command-buffer GPU span and busy time. oxinfer's
`OXINFER_METAL_TRACE` now reports the same, plus the host work done during the pass, the delay
from the first commit to the GPU's start and the delay from the GPU's end to the host's
(`OXINFER_METAL_TRACE_CBS` adds each buffer's time). The per-op times are sums of isolated
dispatches, each paying its own start-up, so they add up to more than a whole pass in which
independent dispatches overlap. ms per 512-token pass, the two engines alternated (load averages
~8 and ~7):

| | Llama-3.2-1B Q4_K_M: llama.cpp | oxinfer | Qwen3-8B Q4_K_M: llama.cpp | oxinfer |
|---|---|---|---|---|
| GEMMs (all projections) | 25.85 | 24.6 | 168.3 | 159.4 |
| LM head (last row) | 0.40 (in the graph) | 0.43 (own command buffer) | 0.89 | 0.95 |
| attention | 1.58 | 1.8 | 5.74 | 8.5 |
| norms | 0.37 | 1.1 | 2.20 | 3.0 |
| half rows for the GEMMs | — (the GEMM reads f32) | 1.0 | — | 4.1 |
| RoPE + KV write | 0.31 + 0.25 | 0.3 (one kernel) | 0.92 + 0.91 | 2.1 |
| GLU / residual adds | 1.35 / 0.42 | 1.5 / in the GEMMs | 5.04 / 2.06 | 5.3 / in the GEMMs |
| sum of the isolated ops | 30.5 | 30.5 | 186.0 | 182.3 |
| whole pass, GPU | 27.65 (2 command buffers) | 30.3 (8) + head 0.43 | 177.8 (2) | 180.3 (8) + head 0.95 |
| idle between command buffers | 0 | 0.02 | 0 | 0.01 |
| host side (wall - GPU) | 1.1 | 2.1 | 1.6 | 4.9 |
| wall (tok/s) | 28.8 (17769) | 32.4 (15790) | 179.4 (2854) | 185.2 (2765) |

So neither the GEMMs (oxinfer's are 5% faster summed) nor idle gaps between command buffers are
the gap. These are:

1. Overlap: llama.cpp's pass runs 2.9 ms (Llama) / 8 ms (Qwen) faster than the sum of its
   isolated ops, while oxinfer's runs at its sum. Removing every barrier from an oxinfer pass
   (wrong results, timing only: `OXINFER_PF_SKIP=barrier`) took 2.2 ms off Llama's 30.1, about
   12 µs per barrier.
2. Norms plus the half conversion: 2.1 / 7.1 ms, against llama.cpp's 0.4 / 2.2.
3. Attention with 128-wide heads: Qwen +2.8 ms at 512 tokens.
4. The host side, 1-3 ms more than llama.cpp's. It went to:
   - a fresh device KV cache allocated for each prompt: 0.5-0.7 ms before the pass, and the GPU
     starting ~1.2 ms after the first commit, not ~0.2;
   - the last part's host KV copy, after the GPU had finished (0.6 ms);
   - the head's own command buffer (0.2 ms beyond its GPU time);
   - the hidden rows read back (0.5 ms).

Methodology. `llama-bench -p N -n 0` times `llama_decode` of N random tokens:
- in ubatches of 512, so pp2048 is four passes over a growing cache;
- with the head on the last row only;
- after a warm-up run;
- keeping the KV cache allocated between repetitions.

`parity_bench` times the engine's prompt call on real text cut to N tokens: one pass of up to
2048 tokens, the head on the last row (on the GPU since round 3), tokenization excluded.

Its GPU times are like-for-like with llama.cpp's (the same pass, the head aside). Its tok/s also
include host work llama-bench does not pay: copying every layer's new keys and values to the host
caches (overlapped with the GPU), reading back the hidden rows, and, until this round, the KV
allocation. So they are a slightly conservative comparison. The default drafts must be off
(`draft=none lookup=0 mtp=0`); otherwise Qwen3-8B's DSpark draft adds a context pass of ~4 ms to
its prompt time.

What changed (host side; logits bit-identical):

- Device KV copies of dropped sessions are pooled (up to 1 GiB, 256 MiB on iOS), and new sessions
  take them instead of fresh buffers. The host's pre-pass time went from 0.5-0.7 ms to 10 µs on
  Qwen3-8B, and the GPU starts sooner after the first commit.
- A pass also cuts a command buffer after its last op that uses a KV cache, so every layer's
  host KV copy overlaps the GPU; the host now finishes 0.1 ms after the GPU instead of 0.6.
  Contiguous hidden rows are read back in parallel.
- New tools: `examples/pf_check` (greedy tokens and top log-probabilities through the engine,
  and a comparison of two runs) and `examples/pf_ab` (one engine per configuration in one
  process, their passes alternated one by one). With other jobs on the GPU, `pf_ab` was the only
  A/B that resolved 1-2% differences.

Measured in a quiet stretch (load average ~7), interleaved run by run (`parity_bench` reps=6
min-of-3 rounds; llama-bench `-r 6`), tok/s with the GPU ms of the pass:

| | pp512 llama.cpp | before | after | pp2048 llama.cpp | before | after |
|---|---|---|---|---|---|---|
| Llama-3.2-1B Q4_K_M | 17840 (27.6) | 16016 (30.0) | 16410 (30.0) | 16489 (4 x 27.4) | 15465 (126.4) | 15889 (126.3) |
| Qwen3-8B Q4_K_M | 2837 (178.1) | 2769 (180.1) | 2786 (180.3) | 2335 (4 x 221) | 2659 (755) | 2658 (762) |

That is 0.92 / 0.96 of llama.cpp on Llama 1B, and 0.98 / 1.14 on Qwen3-8B (whose single
2048-token pass beats llama.cpp's four ubatches).

Correctness. `pf_check`: 64 greedy tokens and the first step's top-20 log-probabilities, at 512
and 1500 prompt tokens. The prompt-pass changes below were checked against the previous plans
(`OXINFER_PF_FUSE_HALF=0 OXINFER_PF_ROWS_OVERLAP=0`) on Llama 1B Q4_K_M and Q8_0, Qwen3-8B
Q4_K_M, Orpheus 3B, Gemma 4 12B Q8_0 and 26B-A4B Q4_0: bit-identical in every case.
- Against the CPU path (its default `fast` GEMMs), the first steps differ by 0.04-0.1 in
  log-probability, and the greedy tokens agree on the Llama, Qwen and Orpheus models.
- On Gemma, raw-text continuations are near-uniform: the CPU's own `fast` and `exact` GEMMs
  disagree there as much.

Measured and dropped (`pf_ab`, 30 alternated passes per configuration unless noted):

- Folding the `x_half` pass into the RMSNorm or GLU before it (`pf_norm_half` / `pf_glu_half`).
  These give the same f32 values and half rows, so bit-identical logits, with a dispatch and a
  barrier fewer each. On Llama 1B at 512 tokens, the fused norm was 0.7 ms slower (best 31.63
  against 30.87 ms; medians 39.41 against 38.44), and the fused GLU made no difference (30.80).
  On Qwen3-8B, neither changed anything beyond noise (10 rounds). The norm kernel has to keep
  `dec_rms_norm`'s 1024 threads per row to sum in the same order, and adding the row's max and
  the half stores to it costs more than the separate pass.
- Telling the barrier analysis that views on disjoint columns of the same token rows do not
  overlap, so that the query, key and value GEMMs run concurrently: 12 barriers per Llama layer
  went to 8. No gain: Llama 1B 31.25 against 31.41 ms at 512 tokens and level at 2048; Qwen3-8B
  184.2 against 183.0 ms at 512 (medians 184.7 / 183.4) and noise at 2048. The narrow key and
  value GEMMs do not fill the GPU alone, but run beside the query GEMM they take as long.
- Flash attention on half tiles (`pfh_*`, for heads of 64 and 128). The four SIMD-groups shared
  32-key blocks of K and V converted to half in threadgroup memory, used half Q and P with f32
  accumulation, and took one online-softmax step per 32 keys, rounding as llama.cpp's half KV
  cache. KL was 1.4e-4, but it was no faster than the f32 kernel: Qwen3-8B 11.3 against 10.8 ms
  per 512-token pass and 128 against 123 ms at 2048; Llama 2.0 against 2.1. The SIMD-group
  matrices run no faster in half than in f32, so attention needs the matrix units (`matmul2d`)
  to gain.

Whole-pass A/B runs across processes moved by ±5-15% with other jobs' GPU work, even as minimums
of alternated runs (load averages were 40-65 for most of the round). The per-op table, the host
timings and `pf_ab` are the firmer measures.

Speculative decoding with a DSpark draft llama.cpp master loads (Anbeeld's Qwen3-14B-DSpark
Q8_0 beside Qwen3-14B Q8_0; `--spec-type draft-dspark --spec-draft-n-max 7`, all layers on the
GPU; oxinfer `--draft` with its default policy and fast verification; 200 tokens, greedy, one
run each, loaded machine): code 94 / 99 tok/s, math 146 / 131, chat 74 / 58, prose 58 / 31-64
(oxinfer's prose run varied 31-64 between two runs; plain decoding 33-35 in both engines). With
`--spec-type draft-dflash` llama.cpp accepts none of this file's drafts.

The matrix runs with `draft=none`. Until November 2026, the default drafts (DSpark for Qwen3
4B / 8B / 14B and Gemma 4 12B, MTP for Qwen3.5-4B / 9B, fast-verification drafts for the 27B and
Gemma 4 31B) kept the prompt off the prefill plans, since the drafts read captured layers: Qwen3-8B
Q8_0 prefilled a 4.3k-token prompt at 87-140 tok/s with its default draft against ~2000 without.
The plans now capture those layers on the device and the drafts' own context passes run on the
GPU, so prefill with a default draft is within 0-25% of `--no-draft` (per-model numbers in
[Prompts with a draft](speculative-decoding.md#prompts-with-a-draft)).

Where oxinfer is still behind:

- **Metal prefill** (0.92 / 0.96x on Llama 1B Q4_K_M at 512 / 2048 tokens, 0.98x on Qwen3-8B at
  512 and 1.14x at 2048 in the fourth round's quiet run; 0.8-0.95x on Orpheus and Gemma 4 12B in
  the earlier ones): not the GEMMs, which sum 5% under llama.cpp's on the same shapes (see the
  fourth round's per-op table), but what surrounds them: llama.cpp's dispatches overlap enough
  that its pass runs 10% under the sum of its ops, while oxinfer's runs at its sum (each of its
  ~190 barriers per Llama 1B pass costs ~12 µs); the norms and the half conversion for the GEMMs
  (2.1 ms of a 30 ms Llama pass, llama.cpp's norms 0.4 with f32 GEMM inputs); attention with
  128-wide heads (1.5x llama.cpp's at 512 tokens; the f32 and half SIMD-group kernels run alike,
  so the next step is the matrix units); and 1-2 ms of host time per prompt. In the kernels alone
  (`mma_gemm_bench`, 512 tokens) the dense GEMMs reach 40-50 TFLOP/s on Q8_0 / Q4_0 / Q4_K / Q5_K
  / Q6_K and wide outputs, less on 2048-wide outputs (35-40) and the 512-wide key / value
  projections; with this tiling the matrix units reach ~51-56 TFLOP/s with the weights'
  dequantization taken out. Half-precision K-quant dequantization, 128x128 tiles, 8 SIMD-groups,
  32-wide K steps, a token-major grid, software pipelining of the next block's loads,
  double-buffered weight tiles, split K, fusing the half conversion into the norms or GLUs and
  running the query / key / value GEMMs concurrently were measured slower or no better.
- **Gemma 4 26B-A4B prefill** (~0.85x at 512 tokens, level at 2048): the routed experts' grouped
  GEMMs are ~45% of the pass at ~17 TFLOP/s (about 32 rows per expert at 512 tokens); covering
  each expert with fewer, fuller tiles (64- and 128-row) was slower than the 128 / 32 / 16-row
  tiling. The dense parts have narrow outputs (2816). Routing now stays on the GPU (no host
  round trip per MoE layer); fusing gate and up is moot for this checkpoint (one fused tensor).
- **CPU prefill against llama.cpp on 12-16 threads**: level or ahead on the models measured in
  [CPU round 3](#cpu-round-3-november-2026) (the 26B-A4B's routed experts now take the int8
  GEMM too, and Qwen / Llama prompts no longer attend with a per-head loop), on a machine too
  loaded for clean ratios; llama.cpp's 16-thread runs collapsed under the load while
  oxinfer's work-claiming pool held up. Per-core throughput is not ahead: both engines' int8
  kernels issue at the SDOT rate (~340 GOP/s on one core).
- **CPU decode of small Q8_0 models** (Llama 3.2 1B Q8_0: 160-175 tok/s against llama.cpp's
  186-192 at 8 threads): the one-row Q8_0 kernel reads 67 GB/s on one core where plain streaming
  does 91 and llama.cpp's repacked `Q8_0_4x8` kernel ~80; see round 3 for what was tried.
  Qwen3-8B (Q8_0, Q4_K_M), Gemma 4 12B, 26B-A4B, Orpheus and Llama 1B Q4_K_M decode level with
  or ahead of llama.cpp's best thread count.

#### CPU prefill GEMMs (November 2026)

CPU prefill matmuls (more than `decode_max_tokens` tokens; decoding is unchanged) pick their
kernel with the engine option `prefill_gemm` (`--prefill-gemm`, `OXINFER_PREFILL_GEMM`):

- `auto` (default): `fast` while a language model processes a prompt (`generate::prefill`, the
  DSpark and MTP decoders' prefill, and the pool jobs they start: `kernels::lm_prefill`), i.e.
  llama.cpp's int8 activation rounding, with the SME tiles beside it where present (`int8` on
  aarch64 without SME and on x86-64 with AVX-512 VNNI, `f32` elsewhere); `exact` everywhere else
  (vision and audio towers, diffusion models and their text encoders, embeddings), whose results
  some features rely on (batched Whisper equal to single windows) or amplify (diffusion).
- `exact`: the opt-out: `sme` where the CPU has the Scalable Matrix Extension (Apple M4 and
  later), else `f32`; the same results as the f32 GEMM.
- `sme`: f32 GEMM on the SME tiles. Weights are dequantized to f32 32 rows at a time as before,
  both operands are transposed to `[k][32]` panels through ZA, and each `FMOPA` adds a 16x16
  outer product into one of ZA's four f32 tiles. Every output is the same fused-multiply-add
  chain in input order as the NEON GEMM's, so the logits are bit-identical to `f32` (Llama 3.2
  1B, Qwen3-8B, Qwen3.5-9B BF16; KL 4e-7 on Gemma 4 12B, where some products round
  differently).
- `hybrid`: `sme` on 7/16 of the row blocks and `int8x2` on the cores' NEON units for the rest,
  at the same time. Which kernel computes a block depends on the block only, not on timing.
- `fast`: the same with `int8` on the NEON units (5/16 on SME).
- `int8x2`: int8 `SDOT` GEMM (aarch64). Weight rows are unpacked four at a time into int8 tiles
  interleaved by four values, with their scales per 32 (integer scales per 16, mins per 32); an
  `SDOT` by element multiplies four activation bytes of one token against four rows. The
  activations are an int8 value plus an int8 rounding residual per block of 32 (about 16-bit).
- `int8`: the same with the value plane only (the decode kernels' rounding, and llama.cpp's for
  its repacked formats). Also on x86-64 with AVX-512 VNNI (`vpdpbusd`; see
  [x86-64 CPU prefill](#x86-64-cpu-prefill-november-2026)); `hybrid` and `int8x2` are the f32
  GEMM there.
- `f32`: dequantized weights times f32 activations on the NEON / AVX2 FMA GEMM (the previous
  path).

Last-token KL against `f32` after a 2.5k-token raw prompt, and greedy text (64 tokens, two
prompts, `--no-system`) identical to `f32`'s:

| | Llama 3.2 1B Q8_0 | Qwen3-8B Q8_0 | Gemma 4 12B Q8_0 | greedy texts identical |
|---|---|---|---|---|
| `sme` | 0 | 0 | 3.8e-7 | 6 / 6 |
| `hybrid` | 7.5e-7 | 4.6e-7 | 3.7e-6 | 4 / 6 |
| `int8x2` | 2.3e-6 | 7.4e-7 | 1.4e-5 | |
| `fast` | 1.1e-3 | 1.4e-3 | 2.2e-3 | 5 / 6 |
| `int8` | 9.5e-4 | 6.6e-4 | 5.9e-3 | |

The texts that parted did so at near ties after 30-50 tokens; llama.cpp's own CPU prefill has the
same rounding, which is why `fast` became the default for language models' prompts (with
`exact` as the opt-out). GEMM throughput (`examples/gemm_bench.rs`, 4096 x 4096 Q8_0, 512
tokens, M5 Max shared with other jobs): `f32` ~1.4 TFLOP/s on 18 threads, `sme` ~2.4 (1.1-1.3 on
one thread; four threads already saturate the SME units, which the cores of a cluster share),
`int8x2` ~2.1, `int8` ~3.9, `hybrid` ~2.7. On Qwen3-8B Q8_0 pp512 at a light load: `sme` 118-128
tok/s, `hybrid` 127-145, `fast` 216-226, against llama.cpp's 133 / 183 / 219 on 8 / 12 / 16
threads: llama.cpp's repacked int8 kernels keep scaling with free cores while `sme` is bound by
the SME units. Splitting the rows between the SME GEMM and the NEON f32 GEMM (which give the
same bits, so the result would not change) gained ~12% on an idle machine with a fixed share,
nothing under load, and lost with blocks handed out at run time (the slow NEON blocks finish
last), so `auto` keeps the SME GEMM alone.

#### x86-64 CPU kernels (November 2026)

Measured on a Threadripper 9970X (32 cores / 64 threads, AVX-512, 1 MiB of L2 per core, 125 GB)
with an RTX 5080, Linux, while two other sessions ran builds and model jobs on the same machine
(load average 45-75 from them throughout: the absolute numbers are low for this CPU, and only
interleaved runs compare). `membench` read 67-68 GB/s from memory at 12-18 threads under that
load. `examples/parity_bench.rs` as in the matrix above (`lookup=0 draft=none mtp=0`, a warm-up
pass per size, then the median of 3 runs); "before / after" are master `ce587c7` and this
work, the two builds alternated twice (both medians given). llama.cpp is master `225a534`
(2026-11-22, CUDA build): `llama-bench -ngl 0 -t 32 -r 2` for the CPU row, with no CUDA device
visible (`CUDA_VISIBLE_DEVICES=`), since with one it runs the prompt's matmuls on the GPU even
at `-ngl 0` (pp512 534 tok/s), and `-ngl 99 -r 3` for the CUDA row. tok/s:

| model, device | pp512 before | after | llama.cpp | decode before | after | llama.cpp |
|---|---|---|---|---|---|---|
| Qwen3.8-27B UD-Q3_K_XL (unsloth GGUF), CPU, 64 threads | 39.9 / 39.2 | 69.8 / 69.4 | 168 | 2.8 / 2.6 | 4.5 / 4.2 | 4.1 |
| the same, 32 threads (the default now) | | 68.0 / 65.5 / 67.8 | | | 5.0 / 4.7 / 4.9 | |
| Qwen3.5-9B `--quant q4`, CPU, 64 threads | 131 / 132 | 201 / 202 | | 11.4 / 11.3 | 11.2 / 11.3 | |
| Qwen3.5-9B `--quant q8`, CPU, 64 threads | 134 / 137 | 204 / 198 | | 6.7 / 6.7 | 6.8 / 6.5 | |
| Qwen3.5-9B `--quant q4`, CUDA (pp512 / pp2048) | 300 / 251, 300 / 312 | 296 / 248, 306 / 306 | | 108.5 / 105.2 | 105.7 / 103.2 | |
| Qwen3.5-9B `--quant q8`, CUDA (one round) | 317 / 272 | 320 / 298 | | 70.6 | 72.1 | |
| Qwen3.8-27B UD-Q3_K_XL, `device=cuda` (GGUF weights stayed on the CPU then; on the device since: 539 / 31.2, see [CUDA](platforms.md#cuda)) | | 73.8 | 2497 (`-ngl 99`; pp2048 2488) | | 4.3 | 58.1 |

(Decode: 24-64 tokens after a 16-token prompt on the CPU, 128 on CUDA. The "after" columns are
the two kernel changes below, on 64 threads; the zero-filling and the thread default came after
and have their own interleaved figures.) What changed:

- **Prefill GEMM.** The 8-row x 8-token GEMM dequantized 8 whole rows and passed them over every
  token panel, so every 8 rows streamed the whole packed prompt (35 MB at 512 tokens x 17408
  inputs) from L3 or memory: on 64 threads it ran at 1.4-1.5 TFLOP/s on a 5120 x 17408 matrix
  (`examples/gemm_bench.rs`, 512 tokens) and 2.5-2.7 on 17408 x 5120, against ~210 GFLOP/s for
  one thread alone. With AVX-512F (chosen at run time) the GEMM now takes 16 rows x 16 tokens
  per step and walks the columns in blocks of 512 (`gemm_blocked` in `kernels.rs`): a pool job
  of 64 rows x 256 tokens dequantizes 16 rows of one column block (32 KiB, in L1), multiplies
  them with its 16 panels of that block (512 KiB, in L2 for all the job's rows) and keeps its
  partial sums between blocks: 4.0-4.7 TFLOP/s on both shapes. Each output is still one fused
  multiply-add chain over the columns in order, so the logits are bit-identical to the AVX2
  GEMM's (`OXINFER_SIMD=avx2` keeps that one). 16 x 16 tiles over whole rows, without the
  column blocks, gave 1.5-2.0 TFLOP/s: the kernel was not what held the GEMM back. Other block
  sizes were measured and not kept: 256 columns 3.6-3.8, 1024 columns x 128 tokens 2.6, 128-row
  jobs 3.8-3.9, 32-row jobs 3.7-4.0.
- **Grid-table i-quants** (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S: 46% of the weights of the
  27B file above) went through the scalar `decode_i8`; they unpack with AVX2 now (four 8-value
  grid entries, or eight gathered 4-value ones, negated by 32 sign bits in one register), for
  decoding and for the prefill dequantization: 109-139 -> 28-38 ns per 256 weights and token
  on one thread (`examples/kernel_bench.rs`; IQ4_XS 18, Q4_K 26). That alone took the 27B from
  2.7 to 4.1-4.2 tok/s decoding and from 37.9 to 41.1 tok/s at pp512.
- **Zero-filled temporaries.** A layer's outputs for a whole prompt are megabytes each (200 MB
  per layer of the 27B at 641 tokens), every one made with `vec![0f32; n]` and then fully
  written by its matmul; once the allocator reuses freed blocks, that zeroing runs on the
  calling thread alone, about 24 ms per layer whatever the device. `kernels::zeros` fills them
  across the pool (the Qwen-family forward, `qwen.rs`; allocations under 1 MiB as before).
  Interleaved with the build before it, 64 threads, two rounds: Qwen3.5-9B q4 on CUDA pp512
  309 / 317 -> 391 / 397 and pp2048 274 / 285 -> 364 / 368 tok/s; on the CPU pp512 203 / 194 ->
  226 / 206 and pp2048 192 / 189 -> 204 / 192; the 27B GGUF on the CPU pp512 67.5 / 68.8 -> 75.6 / 74.8.
- **Threads.** The pool takes one thread per physical core by default where the logical CPUs are
  SMT siblings (Linux, from sysfs; `--threads` / `threads=` as before). Interleaved, 32 against
  64 threads, three rounds each: the 27B decodes at 5.0 / 4.7 / 4.9 against 4.2 / 4.3 / 4.5
  tok/s, Qwen3.5-9B q4 at 11.9 / 14.0 / 13.9 against 11.5 / 11.9 / 12.1 and q8 at 7.3 / 7.3 / 7.5
  against 6.5 / 6.7 / 6.6. pp512 is level on the 27B (68.0 / 65.5 / 67.8 against 66.6 / 68.2 /
  68.5) and 1-5% lower on the 9B (q4 191 / 198 / 203 against 194 / 202 / 206, q8 191 / 201 / 215
  against 199 / 202 / 226) on this loaded machine, where 64 threads also take a larger share of
  the CPU from the other jobs.

Results are the same: 24 greedy tokens after a 2,600-character prompt fed as one batch
(`examples/qwen_greedy_dump.rs`, the logits of the 25 steps and the tokens compared byte for
byte): the 27B GGUF against the previous build (two prompts), and Qwen3.5-9B q4 and q8 against
the AVX2 GEMM of the same build; then the build with the zero-filling change and 32 threads
against the one before it on 64 (the 27B GGUF and Qwen3.5-9B q4 on the CPU, Qwen3.5-9B q4 on
CUDA). The thread count does not enter any result.

Where the time goes now (`--profile`, 27B GGUF on the CPU, 641-token prompt then 24 tokens, 64
threads):

- Prefill, 15.0 ms per token before the zero-filling change: matmuls 73% (the GEMM, at 55-65%
  of what 32 cores' FMA units give alone: 32 x ~210 GFLOP/s), attention and the Gated DeltaNet
  recurrence 9%, elementwise and norms 3%, 16% outside the timers, which was the zero-filling
  above. Running malloc with huge pages (`glibc.malloc.hugetlb=1`) or without returning freed
  blocks to the system had changed nothing (Qwen3.5-9B q4 pp512 196-204, pp2048 188-195 tok/s
  in every variant): the cost was the fill itself, not the page faults.
- Decode, 229 ms per token: matmuls 89%, streaming 12.2 GB per token at 60 GB/s of the 67-68 the
  machine gave; the Gated DeltaNet states and attention 8% (48 states of 3.1 MB read and written
  twice per token); the LM head 5%. Qwen3.5-9B decodes at the same wall: q4 5.6 GB x 11.3 and
  q8 9.6 GB x 6.7 are both 63-64 GB/s. So CPU decoding here is bound by memory bandwidth, and
  equal to llama.cpp's on the same file (4.1 tok/s).

Where oxinfer is still behind on this machine:

- **CPU prefill against llama.cpp** (closed since, see
  [x86-64 CPU prefill](#x86-64-cpu-prefill-november-2026); as measured in this pass: 0.42x on
  the 27B GGUF, 70 against 168 tok/s). llama.cpp
  multiplies int8 activations with the quantized weights (AVX-512 VNNI, 64 products per
  instruction against 16 for an FMA), which is `prefill_gemm=fast`'s rounding; oxinfer has that
  GEMM on aarch64 only, and on x86-64 `fast` is the f32 GEMM, whose ceiling (32 cores x 64
  FLOP per cycle) is about where llama.cpp already is. An int8 GEMM for x86-64 changes the
  logits by the rounding documented under [CPU prefill GEMMs](#cpu-prefill-gemms-november-2026),
  so it was left out of this exact pass.
- **CUDA with GGUF weights.** When this was measured the CUDA backend had no kernels for the
  GGUF block formats, so `--device cuda` ran a GGUF file on the CPU: 74 tok/s at pp512 and 4.3
  decoding on the 27B, against llama.cpp's 2497 and 58.1 with every layer on the GPU (12.2 GiB
  of the card's 16). GGUF weights now decode and prefill on the device (539 tok/s at pp512,
  31.2 decoding, against 2488 and 58.1 in the same session; every block format but the ternary
  and 1-bit ones): [CUDA](platforms.md#cuda) has the table, the agreement with the CPU path and
  where the remaining time goes (the prefill GEMM, at 22-38 TFLOP/s).
- **CUDA prefill** (safetensors weights; no GGUF of the 9B on disk for llama.cpp). `--profile`,
  Qwen3.5-9B q4, 641-token prompt, 2.48 s before the zero-filling change: the GEMM kernels
  0.57 s (128 launches), the host-device copies of their activations 0.61 s (copy-in 0.22,
  copy-out 0.38, pageable memory), attention and the Gated DeltaNet on the CPU 0.39 s, 0.79 s
  outside the timers (the zero-filling). The layers still run op by op with a host round trip
  per group of matmuls, as Metal's prefill did before its whole-pass plans. CUDA decoding (108
  tok/s q4, 71-72 q8) is one plan per token and was not touched.
- **Decode kernels**: bound by memory bandwidth here, so nothing was gained there beyond the
  i-quant unpackers; the Gated DeltaNet update (two passes over each state per token, 8% of
  the 27B's decode) could read its states once less.

#### CPU round 3 (November 2026)

Three changes to CPU prompt processing (`device=cpu`), all inside `fast` (the default for
language models' prompts) except the attention, which applies in every mode:

- **Mixture-of-experts experts on the int8 GEMM.** The grouped matmuls of routed experts
  (`matmul_quant_grouped`) ran on the SME units alone, which saturate with a few threads. Under
  `fast` they now run the int8 SDOT GEMM on every core, skipping each expert segment's zero
  padding rows (`OXINFER_GROUPED_SME_SHARE=N` puts N/16 of the blocks back on SME; any share
  measured slower: 0 335, 4 285, 16 159 tok/s). `hybrid` keeps 7/16 on SME.
- **Qwen-family prompt attention as GEMMs.** Qwen3, Llama 3, Orpheus and the attention layers of
  Qwen3.5 attended over a prompt chunk with the per-(token, head) loop of decoding: at 2.3k
  tokens that was 44% of a Qwen3-8B prefill (5.2 of 11.7 ms per token). Chunks of 32 or more
  tokens now take `attention_prefill` (Gemma's already did): score and value GEMMs per block of
  queries and KV head. Each KV head's keys and transposed values are copied once per call into
  buffers reused by all query blocks (the old code built them per block, through freshly
  allocated vectors whose page faults cost as much as the GEMMs), and the softmax's
  exponentials run in their own vectorized loop (same bits). Verification passes (16 tokens or
  fewer) are unchanged, so exact speculative decoding still equals plain decoding. On x86-64
  with the AVX2 kernels the Qwen-family forward keeps the loop, which measured faster there
  (see [x86-64 CPU prefill](#x86-64-cpu-prefill-november-2026)).
- **MXFP8 scales in place** (CPU only): FP8 E4M3 matrices with E8M0 block exponents (the
  Inkling-2 MXFP8 experts, `fp8_matrix` checkpoints) borrow the exponent bytes from the mapped
  file and decode `2^(e - 127)` from the bits as they dequantize, instead of an f32 copy made at
  load: a quarter of the memory (Inkling-2-Small-MXFP8: ~18 GB of mapped bytes instead of ~73 GB
  of f32 copies). Bit-identical to the expansion (`ggml::e8m0_tests`). With a GPU backend active
  the expansion stays, because the Metal and CUDA MXFP8 kernels read f32 scales.

Correctness: Qwen3-8B Q8_0, last-token KL after a 2.8k-token raw prompt: `exact` before / after
9.1e-8 (the attention's summation order), `fast` before / after 8.8e-4 (fast against exact:
5.9e-4 before, 8.7e-4 after); greedy 64-token continuations of two raw prompts: the prose one
identical in all four runs, the code one flips one near-tie (`usize` / `u32`) between
binaries in each mode. Gemma 4 26B-A4B Q4_0 (whose routing amplifies any rounding): KL against
`exact` 3.3e-3 with the int8 experts, 1.1e-2 with the previous SME experts; top-1 unchanged.

Throughput, M5 Max shared with other agents' builds and GPU jobs (load averages in brackets:
the 30-70 runs are interleaved llama.cpp / this branch / the previous master `ab2da10`, best of
3, and say more about relative speed than absolute), tok/s:

| model | llama.cpp 12 thr | 16 thr | oxinfer before | after |
|---|---|---|---|---|
| 26B-A4B Q4_0 pp512 [7-15] | 240 | 274 | 180 | 357-373 |
| 26B-A4B Q4_0 pp512 / pp2048 [38] | 241 / 203 | 73 / 74 | 111 / 149 | 194 / 219 |
| Qwen3-8B Q8_0 pp512 / pp2048 [38] | 187 / 134 | 78 / 50 | 132 / 92 | 153 / 147 |
| Qwen3-8B Q4_K_M pp512 / pp2048 [38-68] | 83 / 56 | 38 / 36 | 89 / 65 | 87 / 92 |
| Llama 3.2 1B Q4_K_M pp512 / pp2048 [60] | 481 / 270 | 79 / 56 | 595 / 282 | 527 / 375 |
| Llama 3.2 1B Q8_0 pp512 / pp2048 [60] | 232 / 397 | 101 / 115 | 619 / 335 | 592 / 740 |
| Gemma 4 12B Q8_0 pp512 / pp2048 [48] | 49 / 44 | 19 / 34 | 96 / 57 | 70 / 58 |
| Gemma 4 12B Q8_0 pp512, two more rounds [18-28] | | | 113 / 115 | 125 / 101 |
| Orpheus 3B Q4_K_M pp512 / pp2048 [19-48] | 71 / 131 | 80 / 79 | 339 / 201 | 405 / 344 |
| Qwen3-8B Q8_0, 2.3k-token CLI prompt [8-11], ms per token | | | 11.7 (attention 5.2) | 7.1 (attention 1.7) |

The MTP prefill overhead of the GPU (18-20%) does not exist on the CPU: Qwen3.5-9B (`quant=q8`)
pp512 / pp2048 with its MTP layer against `mtp=0`, three interleaved rounds: 158 / 142 vs 172 /
154, 163 / 148 vs 158 / 150, 171 / 156 vs 151 / 151 (the layer is ~3% of the compute).

Decode (tg128, best thread count of each, interleaved, load 6-30): Qwen3-8B Q8_0 31.6 vs
llama.cpp 31.4 (10 threads), Q4_K_M 42.1 vs 41.8; Gemma 4 12B Q8_0 17.3 vs 17.6; 26B-A4B 61.5 vs
58.5; Orpheus 86.8 vs 63.6; Llama 3.2 1B Q4_K_M 213.9 vs 175.8; Llama 3.2 1B Q8_0 174.7 vs 186-192
(the one gap). On one thread llama.cpp is ahead (Qwen3-8B Q4_K_M 8.2 vs 6.7, Q8_0 9.2 vs 8.4):
its repacked kernels need fewer instructions per weight. `examples/matvec_bench` measured where
the Q8_0 decode time goes:

- one core streams 91 GB/s of plain reads, the Q8_0 kernel 67 GB/s (about 48 instructions per 4
  blocks: compute-bound), a prototype of llama.cpp's 4-row interleaved `Q8_0_4x8` layout 80 GB/s;
  at 8-18 threads both kernels land at 210-230 GB/s, so the repack (a second copy of the weights,
  and every kernel taught the layout) would not move multi-threaded decode;
- two rows per call sharing the activation loads, software prefetch at 0.5-8 KB ahead, and
  other row-block sizes (1-32 blocks per thread, 16-512 KB) measured no better;
- plain streaming in 128 KB blocks loses ~15% against 0.25-0.5 MB blocks, but the matvec did not
  follow that trend within the noise;
- running decode attention as one parallel region instead of three at short contexts was within
  noise.

The int8 prefill kernel already issues at the SDOT rate (~340 GOP/s on one core); SMMLA
(i8mm) issues at half that rate for twice the multiply-adds, so llama.cpp's SMMLA kernels are no
faster per core. SME's int8 outer products (SMOPA) would be ~2x the SME f32 rate, but applying a
per-block scale needs the int32 tiles read out every 32 values, which costs more than it gains.

#### x86-64 CPU prefill (November 2026)

The same machine as above (Threadripper 9970X, 32 threads by default, AVX-512 with VNNI), still
shared: load averages 25-37 from other sessions' builds and model jobs during these runs, so
only interleaved runs compare. `examples/parity_bench.rs` as in the matrix (`device=cpu lookup=0
draft=none mtp=0`, a warm-up pass per size, the median of 3 runs); "before" is master `2894417`
(which has [CPU round 3](#cpu-round-3-november-2026)), "exact" this work with
`prefill_gemm=exact`; both built with `--profile ci` (the release optimization level without
LTO); llama.cpp is master `8d60873` (2026-11-22) built without CUDA, `llama-bench -t 32 -r 3`.
The builds and llama.cpp alternated, three rounds for the 27B and two for the others; every
round's median is given. tok/s:

| model | pp512 before | after | after, `exact` | llama.cpp | decode before | after | llama.cpp |
|---|---|---|---|---|---|---|---|
| Qwen3.8-27B UD-Q3_K_XL (unsloth GGUF) | 75.5 / 76.9 / 75.6 | 195 / 201 / 199 | 85.2 / 82.2 / 82.9 | 175 / 174 / 173 | 5.0 / 4.9 / 5.1 | 5.1 / 5.1 / 5.1 | 4.3 / 4.2 / 4.2 |
| Qwen3.5-9B `--quant q4` | 225 / 223 | 638 / 659 | 267 / 240 | | 14.4 / 14.2 | 14.4 / 14.6 | |
| Qwen3.5-9B `--quant q8` | 217 / 220 | 610 / 627 | 248 / 254 | | 7.6 / 7.8 | 7.8 / 7.8 | |
| Llama-3.2-1B Q4_K_M (bartowski) | 822 / 850 | 2538 / 2518 | 1274 / 1246 | 2570 / 2591 | 76.3 / 75.1 | 77.2 / 74.8 | 82.4 / 84.0 |
| Llama-3.2-1B Q8_0 | 845 / 834 | 2444 / 2443 | 1274 / 1282 | 1286 / 1330 | 48.2 / 46.9 | 46.8 / 50.2 | 52.2 / 53.8 |

(Decode: 24 tokens after a 16-token prompt on the 27B, 32 on the 9B, 64 on Llama 1B. "Before"
with `prefill_gemm=exact` is the same as before within 6%, `fast` being the f32 GEMM there.)
Longer prompts:

| model | pp2048 before | after | after, `exact` | llama.cpp |
|---|---|---|---|---|
| Qwen3.8-27B UD-Q3_K_XL | 75.2 / 66.7 | 194 / 176 | 79.0 / 73.8 | 163 / 155 |
| Llama-3.2-1B Q4_K_M | 584 / 591 | 2052 / 2055 | 1147 / 1133 | 2411 / 2407 |
| Llama-3.2-1B Q8_0 | 604 / 603 | 2036 / 1994 | 1168 / 1170 | 1312 / 1459 |
| Llama-3.2-1B Q4_K_M, 2538-token prompt (command line, three rounds) | 546 / 547 / 526 | 1815 / 1851 / 1844 | 1076 / 1090 / 1095 | |

So a prompt of the 27B runs 2.6x faster than before and 1.11-1.19x faster than llama.cpp, the
9B 2.8-3.0x faster than before, Llama 1B 2.9-3.5x faster than before: level with llama.cpp on
Q4_K_M at 512 tokens (0.97-0.99x), 0.85x at 2048, and ahead on Q8_0 (llama.cpp's Q8_0 figure
varied between 1290 and 1960 tok/s across the day's runs and two of its commits). With `prefill_gemm=exact` the 27B and
the 9B gain 7-19% and Llama 1B 1.5x at 512 tokens and 1.9-2.0x at 2048. What changed:

- **Int8 prefill GEMM** (`src/kernels_i8gemm_x86.rs`; `prefill_gemm=int8`, which is what `fast`
  and so the default of a language model's prompt select where the CPU has AVX-512 VNNI, as on
  aarch64; `prefill_gemm=exact` keeps the f32 GEMM). The activations are rounded to int8 per
  block of 32 exactly as the decode kernels round them (the same values as `quantize_act`) and
  stored as unsigned bytes interleaved by groups of 16 tokens, so that one 512-bit load holds
  four consecutive inputs of 16 tokens. Weight rows are unpacked 8 at a time and one block of
  columns at a time to their int8 values (the AVX2 unpackers of the decode kernels) with an f32
  scale per 16 or 32 inputs and, for the formats that have them, an offset per 32. One
  `vpdpbusd` multiplies four weight bytes of a row, broadcast, with those of 16 tokens: 64
  products per instruction against 16 for an FMA; the `+ 128` of the unsigned operand is taken
  out by starting each integer sum at `-128 sum(w)`. Columns go in blocks as in the f32 GEMM:
  a pool job (32 rows, up to 512 tokens) keeps its activations of one block of up to 1024
  columns in L2 (512 KiB) and 8 unpacked rows of it in L1, with the partial f32 sums kept
  between blocks; a row is unpacked once per job. `examples/gemm_bench.rs`, 512 tokens, 32
  threads, two rounds: 10.6-12.2 TFLOP/s on 5120 x 17408 (Q3_K, Q4_K, Q6_K, IQ3_XXS, Q8_0)
  against 3.2-3.6 for the f32 GEMM, 12.7-13.3 against 4.5-4.7 on 17408 x 5120, and 0.58-0.65
  against 0.25 on one thread; the kernel alone, with its activations in L1, does 350-415
  G products/s on one core (0.70-0.83 TFLOP/s in the units above). The activations are written
  16 tokens at a time into memory that is not zeroed first (0.5-0.8 ms for a 512 x 17408 input
  beside a 9.5 ms GEMM; 1.0-1.4 ms when they were zero-filled on the calling thread and written
  through 4-byte scattered stores). Covered formats: Q8 and Q4 (`--quant`), GGUF Q8_0, Q4_0,
  Q4_1, Q5_0, Q5_1, Q3_K, Q4_K, Q5_K, Q6_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
  IQ3_S, IQ4_NL, IQ4_XS, MXFP4, NVFP4 and the low-bit kinds; Q2_K (offsets per 16 inputs) and
  float-stored weights stay on the f32 GEMM, and so do a mixture of experts' grouped GEMMs:
  round 3's int8 grouped GEMM is the aarch64 one, and `matmul_quant_grouped` on x86-64 is the
  f32 GEMM in every mode (no MoE GGUF was measured here). Every output is within 2e-5 of the
  one-token decode kernels on the same int8 activations (unit tests; 7e-6 over every output of
  every matmul of a Qwen3.5-9B q4 prompt fed 12 tokens at a time), and results do not depend
  on the blocks, the jobs or the prefill chunk (bit-identical logits with one column block,
  8-row and 64-token jobs, and chunks of 8, 64 and 736 tokens).
- **Attention of a prompt: the per-head loop, not round 3's GEMMs.** Round 3 moved the
  Qwen-family prompt attention (which GGUF Llama files also take) to score and value GEMMs
  (`attention_prefill`). On this CPU that is slower than the loop it replaced once the loop
  scores eight keys at a time per head (`kernels::dots`), keeps the output sums of up to four
  heads that share a value row in registers over 64 columns while the keys pass
  (`kernels::weighted_values`) and takes its softmax exponentials at the vector width. The
  same build with the GEMMs against the loop, interleaved: Llama-3.2-1B Q4_K_M on 2538 tokens
  712 / 699 / 713 against 1887 / 1879 / 1880 tok/s (attention 1.0 against 0.2 ms per token),
  on 744 tokens 1160 / 1153 / 1156 against 2428 / 2445 / 2406; the 27B on 748 tokens 182 / 184
  / 185 against 199 / 201 / 197 and on 2602 tokens 159 / 155 against 184 / 168; Qwen3.5-9B q4
  on 2602 tokens 513 / 506 against 604 / 593. So on x86-64 with the AVX2 kernels the
  Qwen-family forward keeps the loop for prompt chunks (`model::prefill_attention_gemm`);
  other architectures, and the models whose attention already was `attention_prefill` (Gemma,
  not measured here), are as in round 3. The loop gives a prompt chunk the bits of one-token
  attention, where the GEMMs sum in another order: against master the last prompt position
  differs by a KL of 9e-12 to 1.4e-6 (five models below). Giving the GEMMs' matrices column
  blocks for the cache-blocked f32 GEMM changed nothing (712 / 699 / 713 with, 704 / 712 / 715
  without).
- **Gated DeltaNet rule.** Per token the rule decays a head's state and reads `k^T S`, then
  adds `k (x) delta` and reads `q^T S`: two passes over the state (64 KiB per head on the 9B).
  Over a prompt, a token's second pass and the next token's first are one pass now (each
  element takes the same operations in the same order), compiled for AVX2 / AVX-512.
- **Element-wise passes** (SiLU of the feed-forward, the DeltaNet convolution and its gated
  norm, the attention output gate, which also ran on one thread): compiled for AVX2 / AVX-512
  where the CPU has them (`kernels::wide`). At baseline SSE2 they took 4 values per step and a
  libm call per value to round inside `fast_exp`.
- **Temporaries.** A layer's megabyte temporaries were zero-filled across the pool and then
  fully overwritten by the matmuls (393 ms of a 4.5 s pass over a 748-token prompt of the 27B
  when measured). Within a forward pass a layer now takes the previous layer's buffers back
  (`kernels::scratch` / `recycle`, freed when the pass ends).

Exact-path results (`prefill_gemm=exact`, 16 greedy tokens after the prompt fed as one batch,
`examples/qwen_greedy_dump.rs`, the 17 logits rows compared byte for byte): identical to
master built with the per-head attention loop in place of `attention_prefill`, on the 27B GGUF
and Qwen3.5-9B q4 and q8 (748 and 736 tokens) and Llama-3.2-1B Q4_K_M and Q8_0 (2538 tokens).
Against master as it is (GEMM attention) the last prompt position's KL is 1.2e-11 (27B),
4.3e-8 (9B q4), 1.4e-6 (9B q8), 8.8e-12 and 1.9e-7 (Llama 1B), and the 16 greedy tokens are
the same on four of the five (the 27B's part at the fourth token).

What the int8 rounding changes (KL divergence of the last prompt position's distribution
against the f32 prefill, raw corpus text; beside it the same for the decode kernels run over
the prompt 7 tokens at a time, `qwen_greedy_dump --chunk 7`, which is the rounding every
generated token has always had). The rows marked * were measured again on the merged tree and
gave the same figures; the others were measured before round 3 was merged, on the arithmetic
the identity check above confirms:

| | int8 GEMM vs f32 | decode kernels vs f32 |
|---|---|---|
| 27B UD-Q3_K_XL, four 1500-character prompts | 5.9e-6, 1.9e-4 *, 5.3e-4, 3.2e-3 * | 9.5e-7, 3.8e-4 *, 1.9e-4, 2.9e-3 * |
| 27B, 748 tokens * | 3.2e-3 | |
| 9B q4, four 1500-character prompts | 2.3e-8, 6.8e-4 *, 2.9e-4, 7.8e-4 * | 4.2e-8, 1.2e-3 *, 5.9e-4, 6.6e-4 * |
| 9B q4, two 6000-character prompts | 3.4e-3, 3.8e-3 | 2.7e-3, 4.2e-3 |
| 9B q4, 736 tokens * | 4.4e-2 | 1.7e-3 |
| 9B q8, four 1500-character prompts | 9.6e-9, 1.7e-4, 5.2e-4, 1.7e-3 | 3.2e-8, 2.8e-4, 3.1e-5, 6.8e-4 |
| 9B q8, two 6000-character prompts | 1.8e-3, 1.3e-3 | 8.6e-4, 1.8e-3 |
| 9B q8, 736 tokens * | 1.1e-3 | 1.0e-3 |
| Llama-3.2-1B Q4_K_M, 2538 tokens * | 1.9e-3 | |
| Llama-3.2-1B Q8_0, 2538 tokens * | 2.4e-3 | |

The top token is the same in all of these but one: on the 736-token prompt of the 9B q4 the
f32 pass gives its top token 0.19 and the int8 GEMM picks another (then the same tokens as
the f32 run, one position later), with the largest KL of the set. The 16 greedy tokens after
the prompt are the same as the f32 run's on the 27B and on both Llama 1B files; on the 9B q8
they part at the second token. The int8 GEMM and the decode kernels are as far from each
other (up to 4e-3 on the same prompts) as either is from f32, although each of their matmuls
agrees to 7e-6: once activations are rounded at every matmul, a last-bit difference moves an
activation across a rounding step somewhere, which is the mechanism
[docs/platforms.md](platforms.md) describes for the AVX2 and scalar decode kernels. Greedy
text, `prefill_gemm=exact` against the default, three chat prompts with a 5000-character
context, 48 tokens (64 on Llama 1B with a 6000-character context), before round 3 was merged:
identical in 2 / 3 on the 27B, 2 / 3 on the 9B q4, 1 / 3 on the 9B q8 (both late: after 190 of
240-290 bytes), 3 / 3 on Llama 1B Q4_K_M and 2 / 3 on Q8_0 (10 of 15).

Where the time goes now (`--profile`, 27B GGUF, 748-token prompt, 32 threads, load average
27-30, two rounds against master): 5.0-5.1 ms per token (196 / 200 tok/s) with the matmuls
3.8-3.9 ms and attention and the Gated DeltaNet 0.6 ms; master 13.5 ms (73.9 tok/s) with 11.4
and 1.3-1.4. Llama-3.2-1B Q4_K_M on 2538 tokens: 0.54 ms per token, the matmuls and the
attention 0.2 each and 0.1 outside the timers (the query / key norms, RoPE and cache copies).
Decode is unchanged within the noise and bound by memory bandwidth on the 27B.

Measured and not kept (before round 3 was merged, except the first):

- Round 3's GEMM attention on this CPU, and column blocks for it: above.
- 16 rows per load of a block's activations in the int8 kernel (two groups of 8, the f32 sums
  in memory): Q3_K 10.0 -> 10.5-11.2 TFLOP/s on 32 threads, but IQ3_XXS 11.8 -> 10.3-10.8 and
  one thread 471-515 -> 428-455 GFLOP/s.
- Jobs of 64, 128 or 256 rows instead of 32: 9.3-9.4, 8.9-9.1 and 5.8-8.0 against 9.9-10.3
  TFLOP/s (16 and 24 rows the same as 32); column blocks of 256, 512 or 2048 instead of 1024
  and jobs of 256 tokens: within the noise.
- A CPU prefill chunk of 512 or 1024 tokens instead of 256 (a row is unpacked once per chunk):
  177 / 185 / 181 -> 182 / 191 / 189 and 195 / 191 / 189 tok/s on the 27B over three
  interleaved rounds, 3-5%, at twice or four times the temporaries: the default stays 256
  (`prefill_chunk`).
- An AVX2 `maddubs` version of the int8 GEMM was not built: without VNNI a product of 32 pairs
  takes a sign, a `maddubs`, a `madd` and an add where the FMA GEMM takes one instruction per 8,
  so the x86-64 CPUs without AVX-512 VNNI keep the f32 GEMM.
- Reading the Gated DeltaNet states once per decoded token was not done: merging a token's
  second pass with the next token's first across decode steps would defer the state update to
  the next step and compute the output another way, which changes the bits.

Where oxinfer is still behind on this machine: **long prompts of Llama-3.2-1B Q4_K_M** (0.85x
of llama.cpp at 2048 tokens; level at 512), **decoding Llama-3.2-1B** (0.89-0.94x on Q4_K_M,
0.90-0.93x on Q8_0), a mixture of experts' grouped GEMMs (f32 on x86-64), and the CUDA items
of the [list above](#x86-64-cpu-kernels-november-2026). The int8 GEMM reaches 11-13 TFLOP/s on
32 threads of this shared machine, against 32 x 0.70-0.83 for its kernel alone on one core
with the data in L1.

#### Earlier comparisons (llama-server, op-by-op Metal prefill)

Same GGUF file in both engines, same prompt tokens (llama.cpp gets the text oxinfer's chat
template produces), greedy, 256 generated tokens (fewer when the model ends its turn) after a
short code prompt and a short prose prompt, plus prompt processing of 512 and 2048 tokens.
llama.cpp is commit 4df29be4 with HauhauCS's FastMTP patch, driven through `llama-server`
(one warm-up request per server start); oxinfer runs as a fresh CLI process per measurement.
M5 Max, 18 cores, 128 GB, with other jobs running: runs alternate between the engines, each
configuration ran 3 times, cells are the median with [min-max], and the load average is
recorded per run (medians 9-15 here; tok/s of the CPU-only rows varied by 2x during a heavier
spell, so treat small differences as noise). "Metal" is llama.cpp's default (every layer on the
GPU) and, in the tables of this section, oxinfer with prefill GEMMs on the GPU and everything else
on the CPU, which was its default before GPU decoding (see
[Decoding on the GPU](#decoding-on-the-gpu) for the current default); "CPU" is
`--device none -ngl 0` / `--device cpu` with each engine's default thread count (llama.cpp 6,
oxinfer 18). The oxinfer default and speculative rows were rerun after the Metal matrix-unit
prefill and the multi-token GGUF decode kernels landed (load ~11-12); the oxinfer CPU rows and
the llama.cpp rows are from the earlier runs.

Qwen3.8-27B (HauhauCS Aggressive, `Q4_K_P`, 17.9 GB; speculative rows use the embedded MTP
head, or the 32k-vocabulary FastMTP sidecar with `--mtp-from`):

| Engine | pp512 tok/s | pp2048 tok/s | Decode, code | Decode, prose | Drafts accepted (code / prose) |
|---|---|---|---|---|---|
| llama.cpp Metal | 650 [620-703] | 577 [538-720] | 24.2 [23.3-26.2] | 24.9 [24.1-25.2] | |
| llama.cpp Metal, MTP depth 2 | | | 29.5 [22.5-30.4] | 22.0 [19.9-22.7] | 84% / 61% |
| llama.cpp Metal, MTP depth 3 | | | 34.2 [26.0-35.3] | 24.0 [19.2-28.5] | 76% / 49% |
| llama.cpp Metal, FastMTP depth 3 | | | 37.7 [32.1-39.0] | 25.3 [22.5-26.9] | 78% / 42% |
| llama.cpp CPU | 20.8 [18.9-21.2] | 19.0 [18.9-20.4] | 6.7 [6.2-6.9] | 6.8 [6.3-6.9] | |
| oxinfer (Metal prefill) | 234 [233-252] | 209 [204-233] | 11.6 [11.1-12.0] | 11.7 [10.5-12.3] | |
| oxinfer `--mtp` | | | 21.0 [19.6-21.4] | 15.2 [14.7-15.2] | 80% / 50% |
| oxinfer `--mtp-from` FastMTP | | | 21.5 [21.2-22.3] | 15.0 [14.4-15.1] | 78% / 46% |
| oxinfer CPU | 24.2 [21.4-27.8] | 23.3 [20.8-27.2] | 10.6 [10.3-11.9] | 11.5 [10.8-11.8] | |

(`llama-bench` on the same file in a quiet moment: pp512 744, pp2048 642, tg256 25.5 tok/s.)

Llama-3.2-1B-Instruct (bartowski) and Orpheus 3B (text generation through the chat template, so
the code / prose replies are short: 29-33 tokens for code on Orpheus Q4_K_M; Orpheus Q8_0 replies
only 2-31 tokens and is left out), llama.cpp built from source (ggml 0.20.0), load average 6-8,
3 runs each, CPU rows with each engine's default thread count as above. llama.cpp gets token
ids (no second BOS):

| Model | Engine | pp512 tok/s | Decode, code | Decode, prose |
|---|---|---|---|---|
| 1B Q4_K_M | llama.cpp Metal | 16323 [16266-16451] | 404 [400-423] | 406 [404-408] |
| | llama.cpp CPU | 582 [529-613] | 162 [72-165] | 161 [71-166] |
| | oxinfer (Metal prefill) | 2048 [1164-3012] | 223 [139-227] | 225 [225-229] |
| | oxinfer CPU | 617 [582-665] | 231 [221-235] | 225 [221-235] |
| 1B Q8_0 | llama.cpp Metal | 16956 [16880-17214] | 305 [301-305] | 304 [303-305] |
| | llama.cpp CPU | 748 [677-749] | 159 [158-160] | 159 [158-160] |
| | oxinfer (Metal prefill) | 3012 [1969-3012] | 163 [163-166] | 161 [159-165] |
| | oxinfer CPU | 610 [533-617] | 164 [150-171] | 166 [152-170] |
| Orpheus 3B Q4_K_M | llama.cpp Metal | 5320 [4066-6098] | 190 [189-191] | 191 [189-192] |
| | llama.cpp CPU | 204 [187-209] | 68 [67-69] | 65 [65-66] |
| | oxinfer (Metal prefill) | 1313 [914-1384] | 94 [91-94] | 88 [87-89] |
| | oxinfer CPU | 226 [204-229] | 94 [94-100] | 89 [89-92] |

On the CPU oxinfer decodes 1.4x faster than llama.cpp on Q4_K_M at the default thread counts
(6 vs 18), but with 12 threads each (1B Q4_K_M, code) the gap is 5% (214-215 vs 221-227 tok/s);
Q8_0 is on par. llama.cpp runs the whole pass on the GPU and decodes 1.8-2.2x faster than
oxinfer's GPU-prefill-only path; with GPU decoding oxinfer is on par (table in
[Decoding on the GPU](#decoding-on-the-gpu)).
Greedy outputs: oxinfer's Metal and CPU runs agree with each other on every workload but
one (Orpheus prose, from token 3), and every CPU divergence between the engines within 64
tokens (at tokens 0, 26, 46 and 62) is at the token where llama.cpp's own Metal and CPU runs
diverge too (near ties); 7 of the 12 workload / backend pairs above are identical over the 64
tokens compared.

Zawen-v0.1-35B-A3B (Qwen3.5-MoE, 256 experts, converted with llama.cpp's `convert_hf_to_gguf.py`
+ `llama-quantize`; the checkpoint has no MTP weights, but the converter still writes
`nextn_predict_layers = 1` from `config.json`, which llama.cpp then refuses to load, so
`gguf_set_metadata.py` set it to 0 and `block_count` to 40; oxinfer loads either file):

| Engine | Q4_K_M pp512 | pp2048 | Decode code / prose | Q8_0 pp512 | pp2048 | Decode code / prose |
|---|---|---|---|---|---|---|
| llama.cpp Metal | 3001 [2998-3007] | 3065 [2381-3091] | 110 / 111 | 2700 [2603-2798] | 2476 [1732-2730] | 80.7 / 81.0 |
| llama.cpp CPU | 124 [122-125] | 117 [116-119] | 47.5 / 46.5 | 156 [150-164] | 153 [141-157] | 43.2 / 43.8 |
| oxinfer (Metal prefill) | 277 [274-280] | 294 [290-296] | 63.9 / 64.0 | 210 [210-230] | 236 [234-237] | 45.2 / 45.0 |
| oxinfer CPU | 157 [154-161] | 146 [143-151] | 54.8 / 55.2 | 122 [112-146] | 111 [106-136] | 46.4 / 43.2 |

Qwen3.5-0.8B (`Q8_0` converted with `convert_hf_to_gguf.py`, keeps the MTP layer):

| Engine | pp512 | pp2048 | Decode code / prose | MTP decode code / prose (accepted) |
|---|---|---|---|---|
| llama.cpp Metal | 14930 [14924-15158] | 16034 [15939-16088] | 280 / 281 | depth 2: 332 / 276 (67% / 47%); depth 3: 335 / 250 (59% / 35%) |
| llama.cpp CPU | 809 [797-811] | 748 [739-768] | 161 / 165 | |
| oxinfer (Metal prefill) | 1969 [1896-1969] | 1829 [1812-1845] | 183 / 184 | `--mtp`: 198 / 152 (61% / 36%) |
| oxinfer CPU | 826 [813-826] | 747 [739-779] | 173 / 172 | |

With 12 threads for both engines on the CPU (a separate interleaved run, load ~15), llama.cpp
pulls ahead on prompts: 27B pp512 27.4 vs 18.4 tok/s, decode 9.4 / 8.4 vs 8.0 / 9.7; Zawen Q4_K_M
pp512 197 vs 129, decode 60.5 / 60.1 vs 58.9 / 56.9.

Where the differences come from:

- **Decode with a GPU**: llama.cpp runs the whole forward pass on the GPU; the rows above have
  oxinfer decode on the CPU (`--profile` on the 27B: matmuls stream 16.9 GB/token at ~200 GB/s, 85
  of 95 ms per token), hence the 2.2x on the 27B and 2x on the MoE. oxinfer now runs the whole
  decode pass on the GPU too ([Decoding on the GPU](#decoding-on-the-gpu)): on par with llama.cpp on
  the 1B models, ahead on Qwen3.5-0.8B (1.27x), Zawen (1.2-1.3x) and Gemma 4 26B-A4B (1.13x),
  behind on Orpheus 3B and the 27B (~0.9x; its Q4_K / Q5_K matvecs read 450-530 GB/s against
  550-590 for Q6_K / Q8_0, the main remaining kernel gap).
- **Prefill with a GPU** (2.8x on the 27B, 10x on the MoE, 8x on 0.8B): only the dense GEMMs
  run on the GPU. They now use the M5 matrix units (half inputs, f32 accumulation; 144 -> 234
  tok/s at pp512 on the 27B, before that f32 simdgroup tiles at ~8.4 TFLOP/s against llama.cpp's
  ~35). Attention and the Gated DeltaNet scan stay on the CPU (2.6 ms/token, now the larger part
  of oxinfer's 27B prefill), and each layer's activations make a round trip through host memory.
  On the MoE, the GPU expert path takes fused `gate_up` weights (safetensors checkpoints: Zawen Q4
  prefill 165 -> 353-379 tok/s) and, since GPU decoding landed, GGUF's separate gate and up
  tensors as well (Zawen Q4_K_M, 695-token prompt: 292 -> 548 tok/s; the GGUF rows above predate
  it).
- **CPU prefill**: llama.cpp repacks K-quants at load (`CPU_REPACK`, interleaved rows for the
  i8mm `SMMLA` instructions) and multiplies in int8; oxinfer dequantizes to f32 and runs an
  FMA GEMM. It is ahead on K-quants only while llama.cpp uses its default 6 threads, and behind
  on Q8_0 even then.
- **MoE decode** on the CPU is not bandwidth-bound (Zawen Q4_K_M: 521 matmul calls per token at
  ~21 us, 170 GB/s over only 1.9 GB): dispatch and per-expert overhead dominate. On the GPU the
  routed experts of a layer are two dispatches (gate/up pairs, weighted down) reading through
  per-expert address tables.
- **Speculative decoding**: acceptance is the same in both engines (78-80% code, 46-50% prose
  on the 27B). The multi-token GGUF kernels (each weight block unpacked once per verification
  pass instead of once per token pair) made oxinfer's verification passes cheap enough that MTP
  now gains 1.8x on code and 1.3x on prose on the 27B (11.6 -> 21.0 / 11.7 -> 15.2 tok/s; it was
  1.35x / 1.04x), more than llama.cpp's 1.4-1.55x / ~1x, though llama.cpp's GPU decode still
  leads in absolute terms (34-38 tok/s on code). On 0.8B every draft step reads the 248k-row LM
  head, so prose loses in both engines.
- **Load**: oxinfer's spin-waiting pool on all 18 cores loses the most when other processes
  compete for cores (27B decode fell to 3-6 tok/s at load averages of 30-90, while llama.cpp's
  GPU decode stayed at ~20); `--threads 12` was faster than 18 in that state.

Output agreement (first 64 greedy tokens, same in all three repetitions): oxinfer matches
llama.cpp Metal token for token on the code prompt for every model and on the prose prompt for
the 27B; on prose the texts part at token 48 (0.8B), 25 (Zawen Q4_K_M) and 21 (Zawen Q8_0) after
an identical prefix. llama.cpp's own CPU and Metal backends part as early or earlier on the same
prompts (27B: tokens 53 / 20, 0.8B prose: 9, Zawen: 25 / 18-30), and oxinfer's CPU and Metal
paths always agree with each other, so these are rounding-order differences at near-ties rather
than a model error. Speculative decoding (embedded MTP or FastMTP) gives exactly the plain greedy
tokens in both engines.

To rerun (`tools/bench/bench.py --help` lists the configurations; `--max-load N` waits for a
quieter machine before each run):

```sh
cargo build --release
tools/bench/bench.py --model M.gguf --sidecar FastMTP.gguf --llama-bin ~/llama.cpp/build/bin \
    --reps 3 --tag mymodel --out results.jsonl
tools/bench/summarize.py results.jsonl          # medians, acceptance, greedy agreement
```

## Quantization quality

Q8 and Q4 are both measured against the bf16 output distribution (itself within 1e-4 of the
transformers float32 reference). KL divergence of the next-token distribution on a test prompt is
about 1e-5 nats for Q8 and 3e-5 nats for Q4 on E4B, with the same top-5 tokens. Google also
publishes `gemma-4-*-it-qat-q4_0-unquantized` checkpoints trained for Q4_0; they load the same
way (`--quant q4` on that directory) but did not measure better than Q4 of the regular weights
here, and they are a different fine-tune (KL 0.03 from the regular model).

## Pre-quantized checkpoints

`--quant` normally quantizes the bf16 weights at every start. `--save-quantized DIR` writes the
result instead, so later runs (and browsers) load the int8 / int4 blocks as they are:

```sh
./target/release/oxinfer --model google/gemma-4-E4B-it --quant q4 --save-quantized models/gemma-4-E4B-it-q4
./target/release/oxinfer --model models/gemma-4-E4B-it-q4 -p "Hello"     # precision comes from the file
```

The output is ordinary safetensors: each quantized matrix `X` becomes `X.q8` (I8) + `X.q8s` (F32
scales) or `X.q4` (U8, two nibbles per byte) + `X.q4s` (F16 scales), in exactly the in-memory
layout of the kernels, so the loader borrows them without copying (`QBuf`). Only the tensors the
model reads are written: Qwen keeps its MTP head and its vision tower (in bf16), Gemma's vision
and audio towers are left out; `__metadata__` records the precision. Shards are at most 1 GiB (one allocation must stay under 2 GiB in wasm32), and large
tensors start on 16 KiB boundaries so Metal wraps them in place. The embedding tables are stored
quantized too and dequantized one row per token; tables that are only ever looked up (Gemma's
per-layer embeddings, Qwen 27B's untied input embedding) go to separate `model-tables-*` shards,
which the browser host keeps outside wasm memory. `OXINFER_TABLE_QUANT=bf16|q8|q4` overrides
their precision (default: the model's).

| Model | bf16 checkpoint | Q4 file | Load (in RAM) | KL vs bf16: Q4 at load / Q4 file |
|---|---|---|---|---|
| E2B-it | 10.2 GB | 2.6 GB | 0.40 s | 6.5e-4 / 5.9e-4 |
| E4B-it | 16.0 GB | 4.2 GB | 0.48 s (was 1.51 s) | 0.167 / 0.181 |
| Qwen3.8-27B | 55.6 GB | 15.1 GB | 2.55 s (4.6 s with `--mmap` before) | 0.130 / 0.133 (vs Q8) |

The KL column uses one fixed prompt whose last-token distribution is unusually sensitive; what it
shows is that the file adds nothing on top of Q4 itself (the per-layer embedding table at Q4 is
within noise of keeping it at higher precision). Q8 files match Q8-at-load the same way (E4B
7.1e-5 vs 6.7e-5). On Qwen3.5-0.8B, Q4 is too lossy in either form (KL ~1.3); use Q8 there.
