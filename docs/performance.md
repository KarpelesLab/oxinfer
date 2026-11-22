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

- **Metal prefill** (0.8-0.95x on Llama 1B, Orpheus, Qwen3-8B, Gemma 4 12B; level or ahead on
  Qwen3-14B, Qwen3.5-9B pp2048 and the 27B): still the GEMMs, 85-90% of a pass (Llama 1B Q4_K_M:
  ~25 of 30 GPU ms at 512 tokens, the norms, RoPE, attention and GLU the rest). In the kernels
  alone (`mma_gemm_bench`, 512 tokens) the dense GEMMs reach 40-50 TFLOP/s on Q8_0 / Q4_0 /
  Q4_K / Q5_K / Q6_K and wide outputs (Q6_K and Q5_K since their word reads), less on 2048-wide
  outputs (35-40) and the 512-wide key / value projections; with this tiling the matrix units
  reach ~51-56 TFLOP/s with the weights' dequantization taken out. Half-precision K-quant
  dequantization, 128x128 tiles, 8 SIMD-groups, 32-wide K steps, a token-major grid, software
  pipelining of the next block's loads, double-buffered weight tiles, split K and fusing the half
  conversion into the norms were measured slower or no better (split K helped only the narrowest
  outputs, before the reduction it would need).
- **Gemma 4 26B-A4B prefill** (~0.85x at 512 tokens, level at 2048): the routed experts' grouped
  GEMMs are ~45% of the pass at ~17 TFLOP/s (about 32 rows per expert at 512 tokens); covering
  each expert with fewer, fuller tiles (64- and 128-row) was slower than the 128 / 32 / 16-row
  tiling. The dense parts have narrow outputs (2816). Routing now stays on the GPU (no host
  round trip per MoE layer); fusing gate and up is moot for this checkpoint (one fused tensor).
- **CPU prefill against llama.cpp on 12-16 threads** (0.45-0.7x on the Q8_0 / Q4 models, level on
  Orpheus and the 27B, ahead on the BF16 Qwen3.5-9B): oxinfer's default GEMM now runs on the SME
  units (see [CPU prefill GEMMs](#cpu-prefill-gemms-november-2026)), 1.6-2.5x the previous CPU
  prefill and ahead of llama.cpp's default 6 threads everywhere but Qwen3-8B pp2048 (0.94x), with
  the same results as before. But the SME units saturate with a few threads, while llama.cpp's
  repacked int8 `SMMLA` kernels (int8 activations) keep scaling with free cores. `fast` (SME
  plus int8 on the NEON units, llama.cpp-like rounding), the default for language models' prompts
  since 2026-11-16 (`prefill_gemm=exact` opts out), closes most of the gap (0.75-1.6x of
  llama.cpp's best; 0.46x on the 26B-A4B, whose experts stay on SME).

#### CPU prefill GEMMs (November 2026)

CPU prefill matmuls (more than `decode_max_tokens` tokens; decoding is unchanged) pick their
kernel with the engine option `prefill_gemm` (`--prefill-gemm`, `OXINFER_PREFILL_GEMM`):

- `auto` (default): `fast` while a language model processes a prompt (`generate::prefill`, the
  DSpark and MTP decoders' prefill, and the pool jobs they start: `kernels::lm_prefill`), i.e.
  llama.cpp's int8 activation rounding, with the SME tiles beside it where present (`int8` on
  aarch64 without SME, `f32` elsewhere); `exact` everywhere else (vision and audio towers,
  diffusion models and their text encoders, embeddings), whose results some features rely on
  (batched Whisper equal to single windows) or amplify (diffusion).
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
  its repacked formats).
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
