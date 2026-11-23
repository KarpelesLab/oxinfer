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

A checkpoint larger than memory streams its experts (DeepSeek V4.1 Flash IQ1_M, 231 GB, on the
M5 Max's 137 GB): a prefill chunk routes to nearly every expert of every layer, so the chunk size
and the expert cache decide how much is read from the SSD. Streamed DeepSeek V4 / V4.1 prefills
in chunks of at least 2048 tokens, and a pass that uses most of a layer's experts no longer
evicts the cache's oldest ones (a cycle over more experts than the cache holds is the worst case
of least-recently-used). A 32k-token prompt: 41.6 tok/s before the sparse attention moved to
Metal, 73-75 after, 79.4 with the cache change (300 GB read instead of 717); 126k tokens at 65.6
tok/s. See [DeepSeek V4.1](deepseek-v4.md#on-the-real-weights). A later round (engram rows
requested all at once, the router as one GEMM, and streaming keeping what an earlier run left in
the page cache): 7,666 tokens at 108 tok/s from a cold cache (85 before), 134 on a second run;
decode 8.9 -> 12.2 tok/s cold, 15.5 warm, on the CPU (a GPU that idles between short bursts runs
them at a third of its speed, so the hybrid tried lost); llama.cpp on the same file: 6.0 tok/s
decode and 22.5 tok/s prefill with its dense weights on Metal, 0.4 tok/s on the CPU alone. See
[the round](deepseek-v4.md#decode-and-prefill-against-llamacpp).

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
`--gpu-decode off` (GPU prefill, CPU decode; interleaved runs, load average ~5-11); the Llama,
Orpheus and 26B-A4B GPU rows are tg128 from the
[Metal decode round](#metal-decode-round-november-2026), the others from earlier builds:

| Model | Weights | CPU decode | GPU decode | llama.cpp Metal |
|---|---|---|---|---|
| Llama-3.2-1B-Instruct | GGUF Q4_K_M | 194-205 tok/s | 478 tok/s | 438 tok/s |
| Llama-3.2-1B-Instruct | GGUF Q8_0 | 148-156 tok/s | 347 tok/s | 320 tok/s |
| Orpheus 3B (text) | GGUF Q4_K_M | 87-94 tok/s | 211 tok/s | 198 tok/s |
| Orpheus 3B (`--speak`) | GGUF Q4_K_M | 99 tok/s | 191 tok/s (RTF 0.46) | |
| Qwen3.5-0.8B | GGUF Q8_0 | 179-182 tok/s | 356-361 tok/s | 282-284 tok/s |
| Qwen3.5-0.8B | `--quant q8` | 196 tok/s | 351 tok/s | |
| Qwen3.8-27B (HauhauCS) | GGUF Q4_K_P | 11.9-12.0 tok/s | 20.7-20.9 tok/s | 22.9-23.7 tok/s |
| Zawen-v0.1-35B-A3B (MoE) | GGUF Q4_K_M | 60 tok/s | 137-140 tok/s | 105-116 tok/s |
| Zawen-v0.1-35B-A3B (MoE) | GGUF Q8_0 | 47-48 tok/s | 111-114 tok/s | 85-89 tok/s |
| Zawen-v0.1-35B-A3B (MoE) | Q4 (`--save-quantized`) | 70 tok/s | 152 tok/s | |
| Gemma 4 E2B-it | Q4 (`--save-quantized`) | 134 tok/s | 201 tok/s | |
| Gemma 4 E4B-it | Q4 (`--save-quantized`) | 61 tok/s | 115 tok/s | |
| Gemma 4 26B-A4B (MoE) | GGUF Q4_0 | 63 tok/s | 149 tok/s | 123 tok/s |
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

#### Metal prefill round 5 (November 2026)

Measured first: per-op GPU times (`OXINFER_PF_PROFILE`, isolated dispatches) at 512 and 2048
tokens on master `a3284a2`. At 2048 tokens attention was the largest item after the GEMMs: 17.5%
of the summed op time on Llama-3.2-1B Q4_K_M (22 of 127 ms per pass), 14% on Qwen3-8B Q4_K_M (106
of 759), 15% on Gemma 4 12B Q8_0 (197 of 1332) and 21% on 26B-A4B Q4_0 (112 of 534), with
`pff_*` / `pffs_*` at 8.5-12.5 TFLOP/s. The GEMMs were not behind: the same 512-token passes
profiled op by op in both engines (llama.cpp patched as in round 4; ms per pass, oxinfer /
llama.cpp, the first numbers at load ~15, llama.cpp's at load 10-39) give Qwen3-8B Q8_0
12288x4096 78.8 / 76.3, 4096x12288 41.8 / 43.5, 4096x4096 29.5 / 29.8, 1024x4096 9.6 / 12.7;
Gemma 4 12B Q8_0 15360x3840 115.7 / 141.9 (50 / 41 TFLOP/s), 3840x15360 68.5 / 86.7, the
attention projections 46 / 57; 26B-A4B Q4_0 dense projections 47 / 66 and the routed experts
70.9 (with the routing, gathers and activation) / 66.3 (`MUL_MAT_ID`) plus ~5 of routing ops. So
the Q8_0 / Q4_0 GEMMs were left as they are.

What changed:

- **Attention on the matrix units** (`metal_pfmma.metal`, `pfm_<hd>`), as llama.cpp's
  `kernel_flash_attn_ext_tensor`: a threadgroup of 4 SIMD-groups takes 16 or 32 queries of a
  head; per block of 64 or 128 keys `matmul2d` computes S = Q K^T into threadgroup memory, an
  online softmax (lazy rescaling, as llama.cpp's) writes half probabilities, and `matmul2d`
  accumulates O += P V in a cooperative tensor. The keys and values come from half copies of
  the cache rows in the plan's arena: the RoPE kernel (`dec_qk_rope_t`) writes the chunk's next to
  the f32 cache, `pf_kv_half` converts up to 6144 earlier positions (a later chunk keeps
  `pff_*`), and caches whose lives do not overlap share a slot (one for Llama / Qwen). Kernel
  alone (`pf_attn_bench`, 2048 queries over 8 KV heads, quiet GPU): 32 heads of 128 2.90 ->
  0.99-1.05 ms (11.8 -> 33-35 TFLOP/s), 32 x 64 1.37 -> 0.70, 16 x 256 4.02 -> 1.14, 16 x 512
  (2 KV heads) 7.2 -> 2.7. Per 512-token pass: Llama 1B 1.75 -> 1.2 ms, Qwen3-8B 8.5 -> 3.5,
  Gemma 4 12B 16.9 -> 6.5, 26B-A4B 11.1 -> 4.3 (llama.cpp's FA: 1.6, 5.7, 10.5, 8.2).
- **Gemma's scores in f32.** Gemma 3 / 4 normalize q and k and attend with a scale of 1, so the
  scores are ~sqrt(hd) larger than with 1 / sqrt(hd), and half Q and K round them by up to 1e-2
  (`pf_attn_bench` at scale 1: outputs 1.3e-2 off). Attentions whose scale exceeds
  2 / sqrt(hd) take `pfmq_*` (f32 queries and the f32 cache's keys for S, half only for P V;
  18-20 TFLOP/s, 5e-4 off) and heads of 512 keep `pffs_512`. On Gemma 4 12B this took the last
  position's KL against the previous build from 1.5e-3 to 5.5e-6 (the CPU against the previous
  build: 1.3e-3); `OXINFER_PF_MMA_ATTN=2` keeps half scores everywhere (llama.cpp's rounding, its
  cache being f16), `=0` the old kernels.
- **Norms and GLUs write the GEMMs' half rows.** `pf_glu_half` computes the SwiGLU / GeGLU
  (`dec_glu_t`'s values, bit for bit) and the scaled half rows of the down projection's input in
  the same pass; `pf_norm_half` (RMSNorm) and `pf_add_norm_half` (Gemma's residual add and next
  norm) do the same for a norm whose next op's GEMM reads it, with 256 threads per token and
  float4 loads (another summation order than `dec_rms_norm`, so not bit-identical). Their f32
  rows are not stored when no later op and not the host reads them (every GLU output and the
  norms before the projections), and a GEMM fed by cached half rows no longer counts its f32
  input as read. Isolated per 512-token pass, norms, GLUs and half conversions: Llama 1B
  3.85 -> 2.35 ms, Qwen3-8B 12.4 -> 8.5; one x_half pass and barrier fewer per norm and GLU (3 to
  5 per layer). `OXINFER_PF_GLU_HALF=0` / `OXINFER_PF_NORM_HALF=0` turn these off; the switches are read
  as each plan is built, so `pf_ab` compares them in one process.

Measured and dropped:

- The cache's f32 keys and values straight into `matmul2d` (half Q and P): 12 TFLOP/s at 2048
  tokens, as `pff_*`; converting them to half in threadgroup memory per block (32 keys): 9.6
  TFLOP/s. Only half rows in device memory let the matrix units run at speed. 8 SIMD-groups per
  threadgroup (llama.cpp's choice) were slower than 4 here (1.82 against 1.25 ms).
- For Gemma's scores: splitting Q into half high and low parts halves the error only (7e-3, 20
  TFLOP/s); an f32 `pfmq_512` fits only 8 queries per threadgroup, no faster than `pffs_512`.
- Swizzling the dense GEMMs' threadgroups so that groups of 8 or 32 output tiles run for every
  token tile in turn (weights read from memory once): Qwen3-8B Q8_0 pp512 best of 5, three
  alternations, 2964 / 2953 / 2810 tok/s unswizzled against 2941 / 2913 / 2807 (8) and 2898 /
  2774 / 2281 (32).
- What barriers still cost: removing all of them from a Llama 1B pass (wrong results,
  `OXINFER_PF_SKIP=barrier`) takes 29.0 ms to 26.5; the remaining ones separate dependent ops
  (the query, key and value GEMMs run concurrently with no gain, round 4).

Measured on 2026-11-23, one lock window per model (load averages 15 falling to 2.5),
`llama-bench -p 512,2048 -n 0 -r 4` (llama.cpp master of 2026-11-21) and `parity_bench` of the
previous build (`a3284a2`) and this one, alternated three times; best of each, tok/s pp512 /
pp2048:

| model | llama.cpp | before | after | after / llama.cpp |
|---|---|---|---|---|
| Llama-3.2-1B Q4_K_M | 17820 / 16475 | 16370 / 15840 | 17751 / 18039 | 1.00 / 1.09 |
| Llama-3.2-1B Q8_0 | 18282 / 16899 | 16643 / 16028 | 18047 / 18924 | 0.99 / 1.12 |
| Orpheus 3B Q4_K_M | 6477 / 5643 | 6034 / 5618 | 6519 / 6746 | 1.01 / 1.20 |
| Qwen3-8B Q4_K_M | 2825 / 2321 | 2729 / 2319 | 2890 / 2643 | 1.02 / 1.14 |
| Qwen3-8B Q8_0 | 2939 / 2554 | 2794 / 2474 | 2969 / 2845 | 1.01 / 1.11 |
| Gemma 4 12B Q8_0 | 1800 / 1520 | 1756 / 1371 | 1857 / 1487 | 1.03 / 0.98 |
| Gemma 4 26B-A4B Q4_0 | 4165 / 3815 | 3624 / 3740 | 3840 / 4169 | 0.92 / 1.09 |

The four closest, measured again an hour later the same way (load averages 5-11; Gemma 4 12B
also with `OXINFER_PF_MMA_ATTN=2`, half scores: 1897 / 1730):

| model | llama.cpp | before | after | after / llama.cpp |
|---|---|---|---|---|
| Llama-3.2-1B Q4_K_M | 17827 / 16525 | 16398 / 15912 | 17832 / 18794 | 1.00 / 1.14 |
| Llama-3.2-1B Q8_0 | 18296 / 17002 | 16667 / 16041 | 18100 / 19094 | 0.99 / 1.12 |
| Gemma 4 12B Q8_0 | 1795 / 1585 | 1764 / 1709 | 1870 / 1856 | 1.04 / 1.17 |
| Gemma 4 26B-A4B Q4_0 | 4158 / 3718 | 3619 / 3690 | 3840 / 4067 | 0.92 / 1.09 |

So Gemma 4 12B's 0.98 at 2048 tokens above was the GPU's load (its own previous build moved
from 1371 to 1709 between the runs), and f32 scores cost it nothing measurable.

Correctness (`pf_check`, `draft=none lookup=0 mtp=0`, raw text through the engine, so no system
prompt):
- Against the previous build, the greedy tokens (64, or until the model stops) of a
  chat-formatted request to copy 5000 characters (the model's own template, 1100-1300 tokens):
  identical on all seven models, and on the CPU's 24 too; of the corpus cut to 512 / 1500 / 4000
  tokens (4000: a second chunk, through `pf_kv_half`): identical on the Llama, Orpheus and Qwen
  models.
- KL of the last position over the whole vocabulary, against the previous build / the CPU
  (`device=cpu`, its default `fast` GEMMs) against the previous build / the CPU against this one,
  chat prompt: Llama 1B Q4_K_M 2.1e-5 / 1.24e-2 / 1.19e-2, Q8_0 7.2e-5 / 4.4e-3 / 4.1e-3, Orpheus
  6.1e-7 / 9.0e-4 / 9.0e-4, Qwen3-8B Q4_K_M 1.2e-8 / 1.7e-5 / 1.7e-5, Q8_0 2.3e-7 / 5.4e-4 /
  5.5e-4, Gemma 4 12B 5.5e-6 / 1.3e-3 / 1.4e-3, 26B-A4B 4.3e-4 / 1.8e-3 / 3.0e-3 (its expert
  routing has near ties, see [Gemma 4 26B-A4B](gemma-4-26b.md)); the corpus at 2000 tokens:
  Llama 1B 1.6e-6, Qwen3-8B 1.6e-6 against the previous build.
- On raw corpus text Gemma's continuations are near-uniform and the greedy tokens part at near
  ties within a few steps, between the previous build and the CPU as much as with this one.

#### Metal prefill round 6 (November 2026)

Measured first (master `90f446a`, `OXINFER_PF_PROFILE`, ms per 512-token pass of Gemma 4
26B-A4B Q4_0, isolated dispatches): the routed experts took 69 of the 134 ms (gate / up 39.6,
down 20.2, the device routing `pf_route` 5.1, the activation 2.0, the gather 1.8, the combine
3.1), the routers' f32 GEMM 3.4 and the 120 plain norms 4.5. The whole pass took as long on the
GPU as the sum of its isolated ops (132-134 ms): nothing overlapped. Its MoE layers ran as one
chain of 15 stages, each behind a barrier: the shared (dense) feed-forward's gate / up, GLU and
down, then three norms, the router, its top-k, the routing, the gather, the experts' gate / up,
activation and down, the combine, a norm and the residual step. `pf_route` ran on one
threadgroup (170 µs a layer), alone on the GPU. At 512 tokens the experts get about 14-18 rows
each (median; up to ~220), so their grouped GEMMs run at ~12 TFLOP/s, about as fast as
llama.cpp's `MUL_MAT_ID`, while llama.cpp's whole pass runs well below the sum of its ops.

What changed:

- **Ops scheduled by dependence** (`pf_schedule`). A prefill plan used to keep the graph's order
  and put a barrier wherever an op touched what an op since the last barrier wrote (or wrote what
  it read). Now each graph op's compiled ops, cut at their own barriers, are placed in stages:
  first as early as their dependences allow, then as late as the ops that depend on them allow.
  On a MoE layer that puts the dense feed-forward's gate / up GEMMs beside the experts' gate / up,
  its GLU beside their activation and its down projection beside theirs, while the routing's
  small ops (router, top-k, routing, gather) form the start of the chain. The routed experts'
  encoding is cut into its six phases for this, and the routers' logits get a scratch of their
  own instead of sharing the dense GEMMs'. The dependences also compare the columns of token
  rows rather than spans of the buffer, so the query, key and value GEMMs no longer wait for
  each other (Llama layers 12 -> 10 stages). Scheduling only reorders: the logits are
  bit-identical to the previous build's. `OXINFER_PF_REORDER=0` (graph order), `=early` (as
  early as possible only), `OXINFER_PF_COLS=0` (buffer spans).
- **`pf_route` on a threadgroup per expert**: each counts every expert's entries and scans the
  counts (the same tiles and segments as before), writes its own tiles, and places its own
  entries in token order by a prefix sum. 10.2 -> 1.7 ms per two passes, isolated.
- **The routers' f32 GEMM** (`gemm_f32_r`) issues the loads of 32 columns of K before their
  products, which run in `gemm_f32_s`'s order (the same results); the router sits on the MoE
  layer's critical path. `OXINFER_PF_ROUTER=s` keeps the old kernel.
- **Gemma 4's MoE layer tail in one kernel** (`pf_moe_tail`): the experts' combine, the post
  norms of the dense and the experts' outputs, their sum and the residual step with the next
  layer's input norm (and its half rows for the next GEMMs) run as one pass per token instead of
  four ops in three stages; neither output is stored, as nothing reads them (`pf_moe_tails`
  checks the graph). `OXINFER_PF_MOE_TAIL=0`.
- **Fused AddNorms make the plain norms of their new residual** (`pf_norm_absorb`): the
  router's input and the experts' input, normed from the residual after the attention, come out
  of the attention's residual step (one stage and two dispatches fewer per MoE layer, and the
  router starts right after it). `OXINFER_PF_ABSORB=0`.

A 26B-A4B MoE layer is now 9 stages: residual step, router, top-k, routing, gather, then the two
chains' gate / up, activation and down side by side, then the tail. `pf_ab` (512 tokens, the
switches read as each plan is built, configurations alternated pass by pass; best of 30-40, the
GPU shared with other jobs, so each line compares only within itself):

| change (the new `pf_route` in every configuration) | before | after |
|---|---|---|
| scheduling, column-precise dependences and `gemm_f32_r` | 137.70 ms | 132.99 |
| column-precise dependences alone (same window) | 134.57 | 132.99 |
| `gemm_f32_r` alone | 128.64 | 126.87 |
| `pf_moe_tail` | 123.55 | 121.38 |
| `pf_norm_absorb` | 121.22 | 118.97 |

On Llama 1B Q8_0 the scheduling took the best of 40 passes from 28.78 to 28.40 ms in one window
and made no difference in another (29.18 / 29.28 ms, load average 10); on Qwen3-8B Q4_K_M at 2048
tokens it was level within the noise (best of 20: 901 / 909 ms, medians 959 / 949). A dense
model's layer is a chain: past the query / key / value GEMMs there is nothing to run side by
side, and the barriers left separate dependent ops.

Measured and dropped:

- Loading the next K step's weights before the current step's products in the grouped GEMMs'
  short tiles (32 and 16 rows, the same values): `moe_bench` at the 26B-A4B's shapes with its
  recorded 512-token routing, 2.74 / 2.66 / 2.78 ms a layer without against 2.85 / 2.78 / 2.79
  with, alternated; whole passes too noisy to tell.
- Only moving ops early: the dense feed-forward then runs beside the routing's small ops, which
  were not what cost: 131.83 ms against 131.53 in graph order (the early schedule with
  `gemm_f32_r`, graph order without), while late placement, beside the experts' GEMMs, took
  127.56 in the same window.
- An empty command buffer committed as a prefill step starts, to wake the GPU before the pass:
  the first part's start after its commit went from 0.5-0.9 ms to 0.27-0.74 on Gemma 4 12B, and
  the prompt rate did not move (1893 against 1891 tok/s).

Measured on 2026-11-23, one lock window per model, `llama-bench -p 512,2048 -n 0 -r 4` (llama.cpp
master of 2026-11-21) and `parity_bench` of round 5's build (`90f446a`) and this one, alternated
three times; best of each, tok/s pp512 / pp2048 (load averages 3.7-8.8, other jobs on the GPU):

| model | llama.cpp | before | after | after / llama.cpp |
|---|---|---|---|---|
| Llama-3.2-1B Q4_K_M | 17703 / 16398 | 17748 / 18678 | 18033 / 18999 | 1.02 / 1.16 |
| Llama-3.2-1B Q8_0 | 18285 / 16967 | 18053 / 18764 | 18295 / 19109 | 1.00 / 1.13 |
| Orpheus 3B Q4_K_M | 6478 / 4935 | 6554 / 7018 | 6629 / 6789 | 1.02 / 1.38 |
| Qwen3-8B Q4_K_M | 2822 / 2134 | 2960 / 2693 | 2985 / 2667 | 1.06 / 1.25 |
| Qwen3-8B Q8_0 | 2928 / 2374 | 2948 / 2564 | 2994 / 2674 | 1.02 / 1.13 |
| Gemma 4 12B Q8_0 | 1743 / 1231 | 1668 / 1321 | 1756 / 1325 | 1.01 / 1.08 |
| Gemma 4 26B-A4B Q4_0 | 4120 / 3954 | 3824 / 4222 | 4255 / 4580 | 1.03 / 1.16 |

An earlier run the same way an hour before (without `pf_norm_absorb`; load 4-7) gave the 26B-A4B
4144 / 3322 (llama.cpp), 3882 / 4257 (before), 4235 / 4486; Llama 1B Q8_0 18342 / 16910, 18020 /
19125, 18323 / 19231; Gemma 4 12B 1523 / 1178, 1874 / 1265, 1884 / 1243. The dense models' 2048-token
rates moved by up to 3% either way between the builds from run to run (Orpheus 7035 -> 7138 in
one, 7018 -> 6789 in the other), as the scheduling changes nothing on their chains past the
query / key / value GEMMs.

Correctness (`pf_check`, `draft=none lookup=0 mtp=0`, raw text through the engine, no system
prompt):
- Every model but the 26B-A4B gives the previous build's logits bit for bit (only the order of
  independent ops changed): Llama 1B Q4_K_M / Q8_0, Orpheus, Qwen3-8B Q4_K_M / Q8_0, Gemma 4 12B
  on the chat-formatted copy request (1100-1300 tokens), Qwen3.5 0.8B and 9B BF16 and Gemma 4 E2B
  at 1500 raw tokens, Llama 1B Q8_0 and Qwen3-8B Q4_K_M at 2000 and 4000 (a second chunk).
- The 26B-A4B's tail and absorbed norms sum in another order. KL of the last position over the
  whole vocabulary, against the previous build / the CPU (`device=cpu`) against the previous
  build / the CPU against this one: chat prompt 2.4e-4 / 3.0e-3 / 2.2e-3, the corpus at 2000
  tokens 1.5e-2 / 1.4e-2 / 1.8e-3 (near-tied expert routing moves either way; see
  [Gemma 4 26B-A4B](gemma-4-26b.md)), at 4000 tokens 9.0e-3 against the previous build with the
  16 greedy tokens the same. The 64 greedy tokens of the chat copy request are the previous
  build's, and the routing on the host (`OXINFER_PF_GPU_ROUTE=0`) gives the device routing's
  logits bit for bit. `oxinfer --no-system --temp 0` summarizes 916 tokens of the docs coherently.

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

- **Metal prefill** (after [round 6](#metal-prefill-round-6-november-2026): 1.00-1.06x
  llama.cpp at 512 tokens and 1.08-1.38x at 2048 on all seven models measured, Llama 1B Q8_0 at
  512 tokens level, 1.00): the GEMMs are level with or faster than llama.cpp's on the same
  shapes in isolation (Q8_0 / Q4_0 included), attention runs on the matrix units at 25-35
  TFLOP/s, the norms and GLUs write the GEMMs' half inputs, and ops run by dependence rather
  than in graph order. A dense model's layer is a chain of dependent ops, though: barriers still
  cost ~2.5 ms of a 29 ms Llama 1B pass, where a GEMM's last wave drains before the next op,
  and only fusing ops across those points (the RoPE into the projection GEMMs, the GLU into the
  gate / up GEMM) or putting the head into the pass's last command buffer (the prompt's logits
  take a command buffer of their own, ~0.25 ms beyond their GPU time on Llama 1B) would remove
  more. In the kernels alone (`mma_gemm_bench`, 512 tokens) the
  dense GEMMs reach 40-50 TFLOP/s on Q8_0 / Q4_0 / Q4_K / Q5_K / Q6_K and wide outputs, less on
  2048-wide outputs (35-40) and the 512-wide key / value projections; with this tiling the matrix
  units reach ~51-56 TFLOP/s with the weights' dequantization taken out. Half-precision K-quant
  dequantization, 128x128 tiles, 8 SIMD-groups, 32-wide K steps, a token-major grid or a swizzled
  one, software pipelining of the next block's loads, double-buffered weight tiles, split K and
  running the query / key / value GEMMs concurrently were measured slower or no better.
- **Metal decode** is no longer behind (1.04-1.21x llama.cpp at tg128 after the
  [Metal decode round](#metal-decode-round-november-2026), 5-7% less GPU time per token); what
  is left over the weights' streaming time (1.07-1.48x of it) is the drain and ramp of each
  dependent step, ~8 per layer: fusing the RoPE into the attention kernel would remove one.
- **Gemma 4 26B-A4B prefill** (1.03x at 512 tokens, 1.16x at 2048 after round 6, from 0.92 /
  1.09): the routed experts' grouped GEMMs are still about half of the pass at ~12 TFLOP/s
  (14-18 rows per expert at 512 tokens, median), as slow as llama.cpp's `MUL_MAT_ID`; the dense
  feed-forward now runs beside them, and the routing's small steps (router, top-k, routing,
  gather) are the rest of the layer's critical path. Covering each expert with fewer, fuller
  tiles (64- and 128-row) was slower than the 128 / 32 / 16-row tiling, and loading the next
  K step's weights early gained nothing. Fusing gate and up is moot for this checkpoint (one fused
  tensor).
- **CPU prefill against llama.cpp on 12-16 threads**: level or ahead on the models measured in
  [CPU round 3](#cpu-round-3-november-2026) (the 26B-A4B's routed experts now take the int8
  GEMM too, and Qwen / Llama prompts no longer attend with a per-head loop), on a machine too
  loaded for clean ratios; llama.cpp's 16-thread runs collapsed under the load while
  oxinfer's work-claiming pool held up. Per-core throughput is not ahead: both engines' int8
  kernels issue at the SDOT rate (~340 GOP/s on one core).
- **CPU decode per core** (round 6: Q8_0, Q4_0, Q4_K and Q6_K matrices interleaved four rows at
  a time in place, llama.cpp's repacked layout, bit-identical): level with llama.cpp on 1 and 4
  threads for Q8_0, ahead for Q4_0, 5-10% behind for Q4_K_M; on all threads level or ahead, and
  under load far ahead (llama.cpp's spinning threads collapse at 12 threads). Left: 8 threads on
  the dense Q8_0 models, 10-15% behind llama.cpp's 8 threads in quiet windows. See
  [CPU decode round 6](#cpu-decode-round-6-november-2026), and rounds 5, 4 and 3 before it.
  Qwen3-8B (Q8_0, Q4_K_M), Gemma 4 12B, 26B-A4B, Orpheus and Llama 1B Q4_K_M decode level with
  or ahead of llama.cpp's best thread count.

#### Metal decode round (November 2026)

One-token decode steps on Metal (the whole pass in one plan, see
[Decoding on the GPU](#decoding-on-the-gpu)), against llama.cpp's Metal decode. M5 Max shared with
other jobs (load averages 3-5 in the runs below; another session's GPU checks ran now and then,
so every comparison is alternated). llama.cpp is the master of 2026-11-21 built in the prefill
rounds (`llama-bench -n 128 -p 0 -r 3`, its average); oxinfer `parity_bench MODEL pp= tg=128
reps=3 lookup=0 draft=none mtp=0` (the best of 3; medians within 1% except where noted), the
engines run model by model back to back. "Before" is master `a1c3bbc`. tok/s:

| model | llama.cpp | before | after | after / llama.cpp |
|---|---|---|---|---|
| Llama-3.2-1B Q4_K_M | 438 | 422 | 478 | 1.09 |
| Llama-3.2-1B Q8_0 | 320 | 318 | 347 | 1.08 |
| Llama-3.2-1B Q5_K_M (`llama-quantize` from the f16 GGUF) | 327-394 | 360 | 435 | 1.10-1.33 |
| Orpheus 3B Q4_K_M | 198 | 192 | 211 | 1.06 |
| Qwen3-8B Q4_K_M | 98.9 | 97.0 | 104.0 | 1.05 |
| Qwen3-8B Q8_0 | 63.6 | 62.8 | 66.4 | 1.04 |
| Qwen3-14B Q8_0 | 32.4-34.2 | 36.5-36.8 | 37.2-37.4 | 1.09-1.15 |
| Qwen3.5-9B BF16 | 31.6 | 35.5 | 35.7 | 1.13 |
| Gemma 4 12B Q8_0 | 36.7-39.7 | 39.6-40.6 | 40.1-42.2 | 1.06-1.09 |
| Gemma 4 26B-A4B Q4_0 | 123 | 135 | 149 | 1.21 |

(The 12-15 GB Q8_0 models' runs varied by 5-10% with the page cache, other jobs holding large
models: two windows each. The 27B GGUF and Zawen were not in the cache.) Before this round
oxinfer was 0.96-0.99x llama.cpp on the Llama 1B Q4_K_M, Orpheus and Qwen3-8B rows and level on
the Q8_0 ones; it is now ahead on every model measured. Greedy text (`--no-system`, 200 tokens) is
the previous build's on Llama 1B Q4_K_M, Qwen3-8B Q4_K_M, Gemma 4 12B and 26B-A4B, and `pf_check`
(128 greedy tokens after two raw-text prompts) agreed on nine of them (all but Qwen3-14B, not run);
Qwen3-8B with its DSpark draft and `--exact-verify` still gives the plain greedy text token for
token.

**Matvec bandwidth.** The note that Q4_K / Q5_K matvecs read 450-530 GB/s against 550-590 for
Q6_K / Q8_0 was stale for Q4_K: streaming large matrices (no barrier between them, a ring of
weight copies larger than the caches), every format but Q5_K was already at 555-595 GB/s
(Q4_K 556-591, Q6_K 558-594, Q8_0 564-587, Q4_0 571-595). Q5_K was at 425-440, its fifth bits
converted and multiplied apart from the nibbles (llama.cpp's kernel sums them apart too); they
are now merged into each value's byte first (one conversion and product per value): 541-561 GB/s
(2048x2048 534 -> 584, 4096x4096 439 -> 559, 12288x4096 432 -> 551, 4096x12288 425 -> 561,
alternated in one process). The multi-token kernels call the same function, so speculative
verification stays exact.

**Where the time goes.** A decode pass is a chain of dependent ops, and with the matvecs at the
memory's rate the rest is the fixed cost of each step of the chain. Taking kinds of ops out of
the pass (`OXINFER_DEC_ONLY`, wrong results, timing only), GPU ms per token, before -> after:

| | Llama-3.2-1B Q4_K_M | Qwen3-8B Q4_K_M | Orpheus 3B Q4_K_M |
|---|---|---|---|
| matvecs alone (no barriers) | 1.42 -> 1.40 | 7.97 -> 7.89 | 3.59 -> 3.54 |
| + the barriers between them | +0.24 -> +0.26 | +0.69 -> +0.71 | +0.61 -> +0.63 |
| + norms | +0.16 -> +0.12 | +0.47 -> +0.31 | +0.32 -> +0.21 |
| + RoPE / KV append | +0.05 -> +0.04 | +0.08 -> +0.12 | +0.04 -> +0.08 |
| + attention (16-150 keys) | +0.31 -> +0.11 | +0.80 -> +0.32 | +0.39 -> +0.22 |
| whole pass | 2.17 -> 1.88 | 10.0 -> 9.25 | 4.94 -> 4.49 |

(After: the `+` rows from the build before the SIMD-group change below, which shortened the
barrier and norm stages again; the whole-pass rows are the final build.) llama.cpp's passes take
2.01 / 9.79 / 4.78 ms of GPU time (its command buffers' spans), so oxinfer's GPU time per token is
now 5-7% below llama.cpp's.

Against the time to stream each model's weights once (every tensor but the token embedding,
the head included, routed experts at k / n; 590 GB/s, the matvecs' rate), GPU ms per token
(`OXINFER_METAL_TRACE`, medians of 64-token runs, the two builds alternated twice):

| model | GB per token | streaming | before | after | after / streaming | step (host included) |
|---|---|---|---|---|---|---|
| Llama-3.2-1B Q4_K_M | 0.80 | 1.36 | 2.16-2.17 | 1.88 | 1.38 | 2.05-2.06 |
| Llama-3.2-1B Q8_0 | 1.31 | 2.23 | 2.93-2.97 | 2.66-2.70 | 1.21 | 2.84-2.90 |
| Orpheus 3B Q4_K_M | 2.08 | 3.53 | 4.89-4.94 | 4.47-4.48 | 1.27 | 4.66 |
| Qwen3-8B Q4_K_M | 4.67 | 7.92 | 9.96-9.97 | 9.26-9.28 | 1.17 | 9.47-9.49 |
| Qwen3-8B Q8_0 | 8.04 | 13.63 | 15.34 | 14.63 | 1.07 | 14.86-14.87 |
| Gemma 4 12B Q8_0 | 12.65 | 21.45 | 24.36-24.41 | 23.28-23.32 | 1.09 | 23.54-23.60 |
| Gemma 4 26B-A4B Q4_0 | 2.56 | 4.34 | 7.12-7.16 | 6.40-6.49 | 1.48 | 6.60-6.73 |

What separates a pass from its weights' streaming time is the ~8 dependent steps per layer
(more in Gemma's MoE layers), each paying the GPU's drain and ramp, 2.5-3 us per barrier stage:
the small models, with the least work per stage, lose the most.

Host time per token: the GPU starts a command buffer ~95 us after its commit and the host wakes
~55-70 us after it ends, plus 15-45 us to copy the logits out: ~180-200 us of a step besides its
GPU time (llama.cpp ~290 us on Llama 1B: 2.30 ms per token for 2.01 ms of GPU time).

What changed (logits bit-identical unless noted):

- Attention: up to 512 keys one query head per threadgroup, up to 1024 two (each head's
  arithmetic is the same whatever the grouping), so short contexts spread over 32 threadgroups
  instead of 8; the value rows of 8 keys (4 for heads of 256, 2 for 512, to stay in registers)
  are loaded together instead of as a chain. In isolation, 36 layers with barriers, 32 heads of
  128 over 80 keys: 15.3 -> 6.1 us per layer; heads of 256 at 150 keys 19.8 -> 13.3; 2048 keys
  54.5 -> 45.4.
- RMSNorm / AddNorm kernels on n/8 threads with their weights loaded up front: 4.6-5.3 ->
  3.2-4.0 us per norm (sums in another order: greedy unchanged, first-token logits within 4e-6).
- Matvecs: 4 SIMD-groups per row from rows of 2048 values (a SIMD-group per 16 chunks, at most
  4; it was per 64). Streaming, a SIMD-group per row is as fast or faster, but in a pass, where
  each matvec waits for the op before it, the wider split finishes sooner: Orpheus 4.70 -> 4.49
  ms, Llama 1B Q4_K_M 1.94 -> 1.88, Gemma 4 26B-A4B 6.62 -> 6.44, Qwen3-8B Q8_0 14.84 -> 14.73
  (the multi-token kernels take the same split, so verification stays exact).
- Q5_K matvecs as above.
- Gemma 4 26B-A4B: the routed experts' post norm (`ob += norm(eo) w`) runs in the residual
  step's AddNorm kernel (`dec_norm_add_norm`, `dec_rms_norm`'s body on the same threads, so the
  same bits; `OXINFER_DEC_NORM_PRE=0` keeps the separate norm): a dispatch and a barrier fewer
  per layer, 6.41 -> 6.31-6.38 ms.

Measured and dropped:

- Folding each RMSNorm into the matvecs that read it (the kernel takes x and the norm weight,
  dots x * w and scales by rsqrt of the sum of squares its lanes take of x, so the norm's
  dispatch and barrier go): slower everywhere, as the L1 traffic of x and w per chunk (256 bytes
  per 16 of Q4_K) became the limit. Qwen3-8B Q4_K_M 9.68 -> 10.2-10.4 ms with 1-8 rows per
  threadgroup, Llama 1B 2.03 -> 2.06-2.13, Orpheus 4.78 -> 5.04-5.12.
- The norm taken by the producing matvec's last threadgroup (an atomic count, device-scope
  fences, `dec_rms_norm`'s sums replayed exactly): bit-identical, but the fences in every
  threadgroup cost more than the dispatch they save (36 x a 4096 x 4096 Q4_K matvec and norm:
  43 -> 105 us per layer; 2048 x 2048: 11.5 -> 36.8).
- Prefetching the next matvec's weights into the caches during the light stages (norm, RoPE):
  no gain (a 4096 x 4096 Q4_K matvec after a norm 25.4 us, 24.5-31.4 with 10-50% prefetched).
- Matvec splits by matrix size (the old 1-2 SIMD-groups for matrices of over 1M chunk-rows) and
  2-16 rows per threadgroup: no better in passes (Qwen3-8B 9.25 vs 9.27-9.34 ms) though some
  stream faster alone.
- Attention with 64-key splits at short contexts (more threadgroups, a combine pass): slower
  (80 keys 15.3 -> 18.1 us); fewer SIMD-groups per threadgroup: no change.
- Host latency: a command buffer committed ahead and released by a shared event starts ~80 us
  after the signal instead of ~105 after a commit (not worth restructuring the step for); a
  GPU-side spinner holding the queue never sees the host's flag; a flag the last kernel writes is
  seen ~20 us after the GPU ends instead of ~60, but nothing guarantees the logits written
  before it are visible to the CPU then, so the step still waits for completion.

Tools: `examples/parity_bench` with `pp=` for decode only; `OXINFER_METAL_TRACE=1` per-step
GPU / host timing; `OXINFER_DEC_ONLY=barrier,mv,norm,qk,attn,...` (timing only);
`cargo test --release --lib decode_mv_bench -- --ignored` (matvecs; `OXINFER_MV_BARRIER=1` for
a barrier after each).

#### M3 Ultra (November 2026)

A Mac Studio with an M3 Ultra (24 performance and 8 efficiency cores, an 80-core GPU, 96 GB,
macOS 15, Darwin 24.6), shared with about ten long-running CPU jobs at nice 5 (load averages 12-25 from them
and from the builds of this round, more in the CPU runs noted below). Unlike the M5 Max of the
sections above it has neither GPU matrix units (Apple9 family: no Metal 4 `matmul2d`) nor SME,
so before this round every prompt on Metal took the op-by-op path (matmuls on the f32
SIMD-group GEMM, attention and norms on the CPU, a host round trip per layer): the prefill plans
need the half-row GEMMs, which existed only for the matrix units.

Methodology. llama.cpp is the Homebrew build 5780 (`caf5681f`, July 2025), which loads neither
Qwen3.5 (`qwen35`) nor Gemma 4 GGUFs, so those rows have no llama.cpp column; oxinfer does not
load `qwen3moe` GGUFs (Qwen3-30B-A3B), so there is no MoE row with both. `llama-bench -p 512,2048
-n 128 -r 3` and `parity_bench` (`draft=none lookup=0 mtp=0`, best of 3 after a warm-up pass of
each size) on the same files, the engines alternated run by run, both under `nice -n 19` and
**both under `caffeinate -d -i -u`**: with the display asleep (this machine runs headless) macOS
holds the GPU at its lowest clock (338 MHz, `powermetrics --samplers gpu_power`) after a second
or two of work, whatever the process's priority (`taskpolicy -t 0 -l 0` changed nothing), and
both engines then ran 2-5x slower and drifted within a run (llama.cpp Llama 1B pp512 7339 then
pp2048 1493, `± 430`); keeping the display awake lets the GPU clock up (pp512 ~8000 steady, both
engines). CPU rows: `llama-bench -ngl 0 -nopo 1 -t 8 / 16` (the CPU backend; `-nopo 1` keeps its
large matmuls off Metal), oxinfer `device=cpu threads=16 / 24`, the best of each.

tok/s, llama.cpp / oxinfer before this round (master `422ea32`) / after (master with `8ca1574`
and `c828f9a`, which also has the other sessions' work since `422ea32`, Metal
prefill round 6 among it: the per-change figures below are the alternated ones); Metal pp512 /
pp2048 / tg128:

| model (Metal) | llama.cpp | before | after |
|---|---|---|---|
| Llama-3.2-1B Q4_K_M (bartowski) | 8402-8419 / 7778-7781 / 288 | 1543 / 1858 / 359 | 8524 / 9769 / 359 |
| Llama-3.2-1B Q8_0 | 8833-8844 / 8141-8151 / 239-242 | 1560 / 1857 / 301 | 8749 / 9973 / 301 |
| Qwen3-8B Q4_K_M (Qwen) | 1354-1360 / 1291-1296 / 81-82 | 327 / 385 / 94 | 1381 / 1440 / 94 |
| Qwen3.5-9B Q4_K_M (unsloth) | (not loaded) | 490 / 619 / 92 | 1342 / 1436 / 93 |
| Gemma 4 26B-A4B Q4_0 (ggml-org) | (not loaded) | 229 / 274 / 113 | 2425 / 2695 / 113 |

CPU (oxinfer master `422ea32`: this round changed no CPU code), pp512 / tg64, load averages
14-23:

| model (CPU) | llama.cpp 8 threads | 16 threads | oxinfer 16 threads | 24 threads |
|---|---|---|---|---|
| Llama-3.2-1B Q4_K_M | 746-756 / 125-144 | 565-577 / 119-131 | 924 / 197 | 945 / 199 |
| Llama-3.2-1B Q8_0 | 789-817 / 113 | 582-595 / 81 | 943 / 132 | 1011 / 135 |
| Qwen3-8B Q4_K_M | 141-142 / 30 | 129 / 22 | 155 / 37 | 108 / 17 |

(An earlier pass at load averages 20-45, during builds: Llama 1B Q4_K_M llama.cpp 301 / 54 on 8
threads, oxinfer 420 / 127 on 24, and Qwen3-8B Q4_K_M oxinfer 80 at pp512; every figure here
moves with the other jobs.)

What changed:

- **The half-row GEMMs on SIMD-group matrices** (`8ca1574`). `MMA_SHADER_SRC` compiles with
  `MMA_UNITS 0` on GPUs without matrix units: every `mma_*` / `mmag_*` kernel keeps its name,
  buffers, tiles and rounding (half input rows with a power-of-two scale, weights dequantized to
  half per 64-wide K step, f32 sums), with the products on `simdgroup_half8x8` operands and
  `simdgroup_float8x8` accumulators, so the prefill plans, the grouped expert GEMMs and the
  op-by-op half-row GEMMs all run there (the matrix-unit attention, diffusion sessions and
  Inkling-2's engine stay M5 / A19 only; attention takes the f32 SIMD-group `pff_*`). A
  threadgroup's 4 SIMD-groups take 64 outputs: 2 x 2 of 32 x 32 for 64-token tiles, 128-token
  tiles as two of 64 (32 accumulators a SIMD-group spill: the first build ran gate / up at 0.4
  TFLOP/s), and the plans pick 64-token tiles where the matrix units take 128 (Llama 1B pp2048
  9196 -> 9666). The token rows come straight from device memory (transposed 8x8 loads), the
  weights through threadgroup memory. Against the f32 GEMMs (`OXINFER_METAL_MMA=0`, the op-by-op
  path): the first step's top-20 log-probabilities within 3e-3 (Llama 1B, Qwen3-8B) to 1e-2
  (Qwen3.5-9B) of theirs and the same 64 greedy tokens at 512 and 1500 corpus tokens (Qwen3-8B
  and Qwen3.5-9B at 1500); Gemma 4 26B-A4B within KL 4.5e-3 of them, as the CPU is (3.2e-3; the
  raw corpus's near-uniform continuations part after a few tokens in every pair).
- **SIMD-group layout for 32- and 16-token tiles** (`c828f9a`). `mma_gemm_bench` (isolated
  kernels, Q4_0 / Q4_K / Q6_K, TFLOP/s) showed the 32-token tiles (short prompts, narrow outputs, the remainder
  of a chunk and the routed experts' 32-row tiles) bound by the weight operands' loads from
  threadgroup memory: 2 x 2 SIMD-groups load 4 weight and 2 token fragments for 8 products, 4
  SIMD-groups along the outputs (16 outputs x 32 tokens each) 2 and 4:

  | layout (outputs x tokens per SIMD-group) | 1408x2816 Q4_0 | 2816x704 Q4_0 | 8192x2048 Q4_K | 2048x2048 Q6_K |
  |---|---|---|---|---|
  | 64-token tile, 32 x 32 (kept) | 14.0 | 16.9 | 20.9 / 22.3 (t=2048) | 14.7 / 20.1 |
  | 64-token tile, 16 x 64 | 12.9 | | 18.6 / 19.6 | 13.4 / 18.1 |
  | 32-token tile, 32 x 16 (before) | 11.6 | 11.3 | 14.0 / 15.1 | 11.3 / 13.0 |
  | 32-token tile, 16 x 32 (now) | 14.9 | 16.2 | 17.8 / 18.6 | 13.9 / 16.4 |
  | 16-token tile, 16 x 16 (now) | 13.8 | | 15.8 / 16.4 | 12.5 / 13.4 |

  Each lane now also scales the two elements of a result it holds by their tokens' inverse
  scales (the lane layout of `metal_pfflash.metal`) instead of a product with a diagonal matrix
  of them. The same products in the same order: the first-step log-probabilities and 32 greedy
  tokens are bit for bit the previous build's on Llama 1B Q4_K_M and Qwen3-8B Q4_K_M at 64 and
  1500 tokens; Gemma 4 26B-A4B within KL 4.5e-6. Alternated with the previous build twice (tok/s,
  pp64 / pp512 / pp2048): Llama 1B Q4_K_M 2905-2915 / 8476-8505 / 9742-9779 -> 4017-4048 /
  8505-8524 / 9741-9769, Qwen3-8B Q4_K_M 670-671 / 1364 / 1441 -> 851 / 1378-1381 / 1439-1440,
  Gemma 4 26B-A4B 1016-1018 / 2306 / 2620-2628 -> 1077-1087 / 2421-2425 / 2692-2695, Llama 1B
  Q8_0 8652 / 9911 -> 8749 / 9973, Qwen3.5-9B 1337 / 1438 -> 1342 / 1436.

Measured and dropped (`mma_gemm_bench` with the kernels in `OXINFER_MMA_EXTRA`, which the
SIMD-group library now takes too, unless noted):

- the token rows staged in threadgroup memory beside the weights (as llama.cpp's `mul_mm`):
  0.2-1.5 TFLOP/s slower at 64-token tiles, 0.3-0.5 faster at 32 (before the layout above);
- 8 SIMD-groups (256 threads) on 64 x 128 tiles, half of them dequantizing: 19.6 / 22.3 against
  21.0 / 22.3 at 512 / 2048 tokens; 128-token tiles in 4 SIMD-groups along the tokens: same;
- the next K step's weights loaded into registers before this step's products, double-buffered
  weight tiles, and threadgroup row strides of 64 / 68 / 80 halves instead of 72: all within
  0.3 TFLOP/s;
- 128 outputs x 32 tokens per threadgroup (4 SIMD-groups of 32 x 32): 13.1 / 15.5, below the
  16 x 32 layout's 14.9 / 16.2;
- prefill attention (`pff_*`, f32 SIMD-group matrices) taking 16 or 32 keys per online-softmax
  step instead of 8 (one row maximum, rescale and sum per step; `pf_attn_bench`, 32 heads over 8
  KV heads): heads of 128 at 512 tokens 0.488 -> 0.395 ms, but at 2048 2.68 -> 2.64 (16 keys) and
  2.82 (32), heads of 64 0.97 -> 1.00 / 1.15 ms at 2048; attention is ~7% of a 2048-token
  Qwen3-8B pass (97 of 1420 ms);
- decode matvecs with 2 or 4 rows per SIMD-group and 1 / 2 / 4 SIMD-groups per row for Q4_K
  (Qwen3-8B Q4_K_M tg64 93.7 -> 90.4-94.4), and timing probes without the K-quant activation
  sums or without the nibble conversions (wrong results; tg unchanged at 95.5-95.9): the Q4_K
  matvec (390-590 GB/s in `decode_mv_bench`, against 700-760 for bf16) is not bound by its ALU
  work. Gemma 4 26B-A4B's 8.4 ms decode step, timed by kind with `OXINFER_DEC_ONLY`: dense
  matvecs and the head 3.7 ms, the routed experts 2.5, attention 1.1, norms / GLUs / RoPE 1.6,
  barriers alone ~0.7: the small ops cost more here than on the M5 Max;
- **CPU prefill through Accelerate** (`cblas_sgemm` opened with dlsym, which runs on the AMX
  units): 2.4-3.2 TFLOP/s in f32 alone (512-2048 x 4096 x 4096-12288), but the weights must be
  dequantized to f32 first, and a GEMM of dequantized blocks (8 MB at a time, dequantized on
  the pool, with or without the next block's dequantization overlapping the multiply) reached
  1.0-1.24 TOP/s on 4096 x 4096 Q4_K at 512 tokens against the int8 GEMM's 0.97-1.62 (8-24
  threads, loaded machine), and 44.5 against 79.6 tok/s on Qwen3-8B Q4_K_M pp512.

Where oxinfer is still behind on this machine:

- **Prefill at 512 tokens** is level with llama.cpp b5780 (1.01-1.02x on Llama 1B Q4_K_M and
  Qwen3-8B, 0.99x on Llama 1B Q8_0), ahead at 2048 (1.11-1.26x); the
  SIMD-group GEMMs run at 19-22 TFLOP/s, about 75% of the GPU's f32 rate, and llama.cpp's
  `mul_mm` is no faster. A newer llama.cpp (its own non-tensor kernels, flash attention on by
  default) was not available on the machine.
- **Decoding small and MoE models**: ahead of llama.cpp b5780 (Llama 1B 1.25x, Qwen3-8B 1.15x),
  but a decode step pays ~3-5 µs per dispatch and barrier, which is most of a small model's
  token (Llama 1B: 2.8 ms for 0.8 GB of weights) and of Gemma 4 26B-A4B's (113 tok/s, ~280 GB/s
  of weights); the Q4_K matvec reads 390-590 GB/s where bf16 reads 700-760.
- **CPU**: ahead of llama.cpp b5780's CPU backend at its best thread count here (prefill
  1.1-1.25x, decoding 1.2-1.4x), at about the int8 SDOT rate per core in both. With ~10 cores
  held by other jobs, more threads than free cores lose (Qwen3-8B pp512 108 on 24 threads
  against 155 on 16, tg64 17 against 37; the default is 32, a thread per core, efficiency cores
  included); no fixed count suits every load, so the default was left alone.

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
  its repacked formats). On aarch64 Linux it is cache-blocked by columns and runs on `SMMLA`
  where the CPU has i8mm, with the same bits (see
  [DGX Spark CPU](#dgx-spark-cpu-november-2026)). Also on x86-64 with AVX-512 VNNI (`vpdpbusd`; see
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
| Qwen3.8-27B UD-Q3_K_XL, `device=cuda` (GGUF weights stayed on the CPU then; on the device since: 537 / 31.1, see [CUDA](platforms.md#cuda)) | | 73.8 | 2497 (`-ngl 99`; pp2048 2488) | | 4.3 | 58.1 |

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
  of the card's 16). GGUF weights now decode and prefill on the device (537 tok/s at pp512,
  31.1 decoding, against 2488 and 58.1 the same day; every block format but the ternary
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
  of f32 copies). Bit-identical to the expansion (`ggml::e8m0_tests`). The Metal expert kernels
  now read the bytes too ([CPU decode round 4](#cpu-decode-round-4-november-2026)); under CUDA
  and Vulkan the expansion stays, because their MXFP8 kernels read f32 scales.

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

#### CPU decode round 4 (November 2026)

The last CPU decode gap, Llama 3.2 1B Q8_0, profiled per parallel region (a timestamped trace of
every `pool::run` and of marks between them, one token at 18 threads, medians over 60 tokens; not
kept in the tree). Of ~6.4 ms a token, 5.8 ms were in parallel regions: per layer gate + up 154
us, down 78, query / key / value 31, output 21, attention 15-24; the LM head 1.19 ms. The rest
was serial work on the calling thread between regions, ~450 us a token:

- 77 us between tokens, almost all of it `sampler::argmax`, a compare and branch per logit over
  the 128k vocabulary, and 14 us zero-filling the logits on one thread;
- 3-7 us before each matmul: the residual add and RMSNorm (its sum of squares is one serial
  chain, kept for the bits), and the activations' quantization, whose exact block sums
  (`fsums` / `hsums`, for the formats with mins) were three scalar left-to-right sums per block
  (4.7 us for the 8192-wide down projection's input);
- 3.4 us of query / key preparation before the attention, whose shared-row kernel ran three
  parallel regions for a few hundred keys.

Kept (all bit-identical; greedy outputs of Llama 3.2 1B Q8_0 and Q4_K_M, Qwen3-8B Q8_0 and
Gemma 4 12B Q8_0 unchanged against the previous build):

- **Few-token attention per head up to 1024 keys** (`model::SHARED_MIN_KEYS`): one region of one
  task per query head instead of `attn::attend_shared`'s three (1.4-1.8x faster at 64-1024
  keys: 64 keys 6 against 11 us, 512 keys 22 against 30, the shared kernel ahead from about
  4096 keys); in decode 15 us against 24 a call.
- **Activation sums on NEON**: four blocks a vector, one per lane, each still a left-to-right sum
  from -0.0 (`quant::act_sum_tests`).
- **NEON argmax** (`sampler::argmax_matches_scalar`: first index on ties, NaNs never chosen) and
  the logits zeroed across the pool (`kernels::zeros_from`): 77 us between tokens down to 34.

Measured and dropped, with the same trace or interleaved runs:

- software prefetch in the Q8_0 kernel (0.5-16 KB ahead, L1 / L2 / streaming hints): within the
  noise; a leaner Q8_0 loop (multi-register loads, scales gathered by lane: ~25% fewer
  instructions, same bits): no faster at 12-18 threads; a timing probe of llama.cpp's 4-row
  interleaved layout: 21% faster on one core (55 against 67 us for 2048 x 2048), level at 18
  threads, as round 3 found;
- decreasing ("guided") block sizes toward the end of each region: 3.5-4% slower a token (tokens
  alternating between the two schedules in one process, so drift cancels: 1% between two
  identical schedules); 1, 2, 4 or 16 blocks per thread instead of 8: within 1%, 16 3% slower;
- idle workers prefetching the first 256 KB-4 MB of the matmul that followed last time while the
  calling thread runs the serial work in between: 2-3% slower;
- longer spinning before a worker yields (65536 or 10^6 spins instead of 4096, ~32 us): 15-70%
  slower on the shared machine (the spinning threads take the cores the calling thread needs);
- quantizing a one-token matmul's activations in the region's first tasks (the matmul tasks
  waiting on them) instead of on the calling thread: within the noise.

The kernels are not the limit: a replay of one decode step's matmuls alone (a `matvec_bench`
variant, not kept: same shapes, same order, 18 threads) takes 4.7-5.3 ms at its best and the same per-region medians as
in the model (gate + up 157-163 us, down 79-83), and plain streaming of the same sizes in the
same blocks reads 250-280 GB/s against the kernel's ~225 in place. In some windows the replay's
best tenth of passes ran ~15% faster (gate + up 127 against 149 us) when each matmul followed
the last within 2 us, and never when a 5-20 us pause separated them; the effect came and went
with the other jobs on the machine and was not reproduced on demand.

Decode tg128 (`parity_bench`, best of each run's 3; interleaved rounds of llama.cpp, the
previous build `a3284a2` and this one; llama.cpp's mean ± deviation of 3 at its best thread
count; load averages in brackets), tok/s:

| model | llama.cpp | before | after (18 threads) | after, 12 threads |
|---|---|---|---|---|
| Llama 3.2 1B Q8_0, 5 rounds [4-14] | 168-183 (12 thr) | 156.0-161.5 | 158.9-170.3 | 157.0-168.4 |
| Llama 3.2 1B Q4_K_M, 5 rounds [4-14] | 205-218 (12) | 210.2-224.5 | 223.5-231.2 | |
| Qwen3-8B Q8_0, 3 rounds [12-18] | 29.3-33.4 (10) | 30.4-30.9 | 30.9-31.3 | |
| Gemma 4 12B Q8_0, 3 rounds [8-19] | 18.0-18.2 (10) | 18.8-19.8 | 18.7-19.0 | |

Llama 3.2 1B Q8_0 stays the one gap, now ~7% (best against best 170.3 against 183.1; it was
12%). In one quiet window (load 2.6-9) a no-LTO build of this tree, with the read-ahead above
switched off, ran 176-188 tok/s; llama.cpp measured 174-183 a few minutes later and the release
build 157-165, so on this shared machine (other agents' builds and GPU jobs, which also take
memory bandwidth) a run's speed moved by 10% between minutes, more than the changes measured.
The other models are level or ahead, as before.

**MXFP8 byte scales on Metal.** The Metal expert kernels of Inkling-2 (`src/metal_inkling.metal`,
the decode path's `ink_x_*`; `src/metal_inkling_pf.metal`, the prefill's grouped matrix-unit
GEMMs `mmag_*`) have a byte-scale MXFP8 format, `mxf8b`, beside the f32-scale `mxf8`: its loader
decodes `2^(e - 127)` from the E8M0 bits as the CPU does (0: 2^-127, 255: NaN) and multiplies as
the f32 one does, so both give the same floats. With Metal active MXFP8 matrices now keep their
exponent bytes mapped from the file (`ggml::e8m0_in_place`); CUDA and Vulkan keep the f32
expansion, as does `OXINFER_MXFP8_SCALES=f32` (comparisons). The Metal backend's other paths
take no MXFP8 matrix with runs of 32 (the dense FP8 kernels want 128 x 128 blocks), so those fall
back to the CPU as before. `metal::inkling::mxf8_tests` runs the decode kernel (`_gsum`, 5
tokens) and the prefill GEMMs (150 and 37 rows of two experts: every tile size) both ways on
random E4M3 values and exponents (zero included) and finds the outputs bit-identical.

On `thinkingmachines/Inkling-2-Mini`, whose layers 20-23 have MXFP8 experts (4.1 GB of E4M3
values, 127 MB of exponent bytes, 510 MB as f32), on Metal (CLI, 672-token prompt, 128 tokens):

| scales, 3 interleaved rounds [5-7] | prefill tok/s | decode tok/s | peak footprint | max RSS |
|---|---|---|---|---|
| E8M0 bytes, mapped (now) | 1430-1436 | 63.0-64.3 | 21.32 GB | 32.68 GB |
| f32 expansion (`OXINFER_MXFP8_SCALES=f32`, before) | 1418-1427 | 63.6-64.2 | 21.70 GB | 33.18 GB |

The same speed, 375 MB less memory of the process's own (the f32 copies), and the same greedy
output in all six runs. On Inkling-2-Small-MXFP8 (48 MXFP8 layers) the same change keeps ~18 GB
of exponent bytes mapped instead of ~73 GB of f32 copies, with Metal active as on the CPU.

#### CPU decode round 5 (November 2026)

Llama 3.2 1B Q8_0 again, with a trace of every region's tasks (which thread ran which task,
when; not kept) and `examples/decode_ab.rs`, which times decode steps under several settings in
one process, switching settings every token (every 64 when they change the thread count, so
that the threads left out have parked) and restarting from the prompt every 128 tokens, as
tg128 does. Same-process alternation was the only way to see effects of a few percent: the
machine's speed moved by 10-20% between minutes (per-step medians of 5.3 to 6.7 ms for the same
build), and llama.cpp's own runs by as much.

Where a token goes, 18 threads, in a quiet window (load 2-3): the 1.31 GB of weights a token
(64.6 MB a layer, 279 MB of LM head) in 4.7-5.4 ms inside regions and ~0.3 ms between them.
Inside the regions the threads were busy 85-92% of the time: the workers start a region 0.8-1.2
us after it is published, the calling thread then waits 3-9 us at its end for the last tasks
(half a task of 8-17 us, more for the LM head's ~2 MB, 130 us tasks), and about one worker in
six regions does not take part at all (its core is busy elsewhere). Every thread ran its tasks
at about the same speed (0.92-1.00x the median task): this M5 Max has 6 "super" and 12
"performance" cores and no efficiency cores, and the work-claiming pool spreads the tasks
evenly; the `Background` QoS, the only way to place threads, halves a thread's kernel speed
(36 against 68 GB/s). Bandwidth (`matvec_bench`): plain reads stream 262-287 GB/s at 10-18
threads (238 at 6), the Q8_0 kernel alone 220-231 GB/s at 6-18 threads in most windows and
260-280 in quiet ones, the regions of a decode step 243-279 GB/s, the whole step 240-257 GB/s,
which is what llama.cpp's 183-196 tok/s amounts to as well.

Regions per token, Llama 3.2 1B (`pool::regions()`): 98 before (per layer the query / key /
value matmuls, the attention, the output projection, gate and up, the SiLU, the down
projection; then the logits' zero-fill and the LM head), 82 after (the SiLU's region gone into
gate and up's). The query, key and value projections were already one region (`linear_multi`);
the RMSNorms stay on the calling thread, their sum of squares a 2048-long serial chain (~1.4 us,
33 a token) kept for the bits.

Kept:

- **Gated MLP in one region** (`kernels::matmul_quant_glu`): each task takes the same rows of
  gate and up (whole blocks of 32), applies SiLU or GELU to them and quantizes them into the
  down projection's int8 input, instead of a region for gate and up, a region for the
  activation and the down projection's 8192-wide quantization on the calling thread.
  Bit-identical (`kernels::tests::glu_matches_separate_steps_bitwise`; `OXINFER_DECODE_GLU=0`
  for the old steps), for Llama / Qwen / Orpheus and Gemma's dense MLP. Gaps between regions
  389 to 311 us a token; in one process, alternating, the median step 0.7% faster (5.848 to
  5.806 ms over 1500 steps each) in a quiet window, 2.4% in a loaded one.
- **Pool synchronization**: each worker announces itself in a region on a flag on its own cache
  line, instead of incrementing and decrementing one shared counter (17 read-modify-writes
  queued on one line a region), and the claim counter and the waiting workers' generation have
  lines of their own: workers start 0.73-0.94 us after a region is published instead of
  0.83-1.0; 0.5-0.9% a step (three alternating runs, one at -0.1%).
- **Decode steps on all but two threads** from 12 threads on (`pool::decode_scope` around the
  steps of `generate::decode`; `OXINFER_DECODE_THREADS=N` sets the count, prompts keep every
  thread): the two workers left out park, leaving cores to the rest of the system instead of
  one of 18 threads losing its core in the middle of a region. 16 against 18 threads: in five
  one-process runs -1.4%, +0.2%, +0.9%, +2.4% and +4.9% a step; in two sets of four interleaved
  tg128 rounds +4.1% and +0.3% on average (single rounds from -2.6% to +6.7%). Which count is
  fastest moves with the load: 16 in quiet windows, 12-14 in the busiest (where 18 lost 7%), 8
  always slower (147 against 159 tok/s).

Measured and dropped:

- splitting a region's last blocks in four, so that the last tasks claimed are short: the
  calling thread's wait at the end of the gated MLP's region fell from 9.3 to 6.4 us, but steps
  were 1.0-1.4% slower;
- blocks of at most 512 KB in decode regions (the LM head's are ~2 MB): from -0.5% to +0.6%
  in four comparisons (+4-5% in one, not reproduced);
- choosing the decode thread count among 12-18 by measured step times (a few steps on each in
  turn, then the fastest for 500 steps): the choice followed the noise (16, 18, 18, 12 in four
  explorations of one run) and tg128 was no faster than the fixed 16 (157-169 against 162-173
  tok/s);
- the argmax fused into the LM head: after round 4's NEON argmax a token's argmax and the
  logits' zero-fill take a few microseconds, under 0.2% of a step, not worth carrying the
  greedy case through the model interface.

Not tried: llama.cpp's 4-row interleaved layout again. Its 21% faster single-core kernel did not
help at 18 threads (rounds 3 and 4), where the step streams as fast as plain reads in quiet
windows, but it is what lets llama.cpp run 8 threads: on 8 threads oxinfer decodes 147 tok/s
against 159 on 12-14, while llama.cpp's 8 threads held 186-189 tok/s through load averages of
12-16 that cost oxinfer's 18 threads 10%. It would take either a second copy of the weights or
every kernel (prompt GEMMs included) taught the interleaved layout.

Decode tg128 (`parity_bench`, best of 3 of each run; rounds interleave llama.cpp, master
`422ea32` and this branch; llama.cpp's mean of 3 at 8 and 12 threads, or 10 for the larger
models; load averages in brackets), tok/s:

| model | llama.cpp 8 thr | 12 thr | before | after |
|---|---|---|---|---|
| Llama 3.2 1B Q8_0, round 1 [5] | 185.2 | 177.6 | 161.3 | 167.8 |
| round 2 [8] | 158.6 | 168.2 | 158.7 | 161.0 |
| round 3 [10] | 156.6 | 182.9 | 158.4 | 166.9 |
| round 4 [11] | 156.1 | 179.9 | 159.7 | 175.4 |
| Llama 3.2 1B Q4_K_M, 2 rounds [12] | 190.1 / 196.9 | 220.1 / 216.8 | 231.8 / 230.0 | 251.3 / 245.9 |
| Qwen3-8B Q8_0, 2 rounds [10-15] | | 30.6 / 32.8 (10 thr) | 27.7 / 27.8 | 29.2 / 28.9 |
| Gemma 4 12B Q8_0, 2 rounds [17-19] | | 12.9 / 20.4 (10 thr) | 18.5 / 18.7 | 19.7 / 20.2 |
| Gemma 4 26B-A4B Q4_0, 2 rounds [17] | | 63.7 / 66.2 (10 thr) | 67.2 / 66.7 | 68.2 / 68.2 |

In two quieter windows earlier the same day (load 3-10) an intermediate build of this branch
(the gated MLP and the pool flags, with the tail split above) ran Llama 3.2 1B Q8_0 at 179-194
tok/s on 18 threads and 191-196 on 16, against llama.cpp's 185-196 at 8 threads and 174-189 at
12 and the previous master's 183-192: level with llama.cpp's best when the machine is quiet,
3-10% behind it under load (where llama.cpp's 8 threads hold up best), and 2-10% ahead of the
previous build.
The other models gained 1.5-7%; Qwen3-8B stays behind llama.cpp under this load (it was level in
round 3's quieter runs). Greedy 128-token outputs (`--device cpu --no-system`, two prompts) are
identical to the previous build's on Llama 3.2 1B Q8_0 and Q4_K_M, Qwen3-8B Q8_0, Gemma 4 12B
Q8_0 and Gemma 4 26B-A4B Q4_0.

#### CPU decode round 6 (November 2026)

Round 5 left llama.cpp ahead on loaded machines, and the cause was per-core speed: its
repacked kernels need fewer threads for the same bandwidth. This round interleaves the weights
the way llama.cpp's ggml-cpu "repack" buffers do (`block_q8_0x4` and its kin with 4-byte
pieces), in place at load, so that one SDOT by element computes four rows.

**The layout** (`src/ggml_repack.rs`). A Q8_0, Q4_0, Q4_K or Q6_K matrix whose row count is a
multiple of four is stored as groups of four rows, each group as its blocks in order with the
four rows' block `b` together: the scale fields first (the four `d`; Q4_K's four `dmin` too;
Q6_K's int8 scales by sub-block), then the values' bytes in 4-byte pieces, `[row 0 bytes 0..4,
row 1 bytes 0..4, row 2, row 3, row 0 bytes 4..8, ...]` (Q4_K's twelve bytes of 6-bit scales as
pieces as well, so the four rows' scales unpack in three vector words). A piece times four
activations (`sdot v.4s, w.16b, x.4b[lane]`) leaves each row's sum in its own lane, and a block
of 32 needs 8 SDOTs and two activation loads for four rows, where the plain kernel needs 2 SDOTs,
2 loads and a pairwise-add tree per row; the scales come as one vector per block. The same bytes
move, so the layout takes no memory of its own: the loader interleaves a tensor in place
(`ModelFiles::ggml_cpu`) in the copy of the checkpoint it read into memory, which is the default
outside iOS. Mapped checkpoints (`mmap=1`, too big for memory, streamed) keep the file's layout,
and so does every model while a GPU backend is active (its kernels read the file's layout).
Reading an interleaved tensor's bytes other than as a matrix is an error rather than garbage.

**Bit-identical.** The decode kernels (`kernels_ggml.rs`: `x4_kern` for Q8_0 / Q4_0, `q4k_x4`,
`q6k_x4`) repeat the plain kernels' f32 operations row by row in their order: the plain kernels
keep, per row, one accumulator lane per block of a step of four (Q8_0, Q4_0) or per sub-block
pair (Q4_K, Q6_K) and add the lanes as `(0 + 1) + (2 + 3)` at the end; the interleaved ones keep
one accumulator per such lane, with the four rows in its lanes, and add them in the same order.
The integer sums are exact either way. So every output is bit-identical to the plain matrix's
(`ggml::tests::interleaved_rows_match_plain`: one token, 2-4 token tiles, multi-token passes of
2-16 tokens, the driver's decode and prefill paths, row ranges that cut groups, dequantization;
`kernels::i8gemm::tests::interleaved_gguf_matches_decode_kernels`: the SDOT, SMMLA and
column-blocked prompt GEMMs), multi-token verification still equals plain decoding, and greedy
128-token outputs (`--device cpu --no-system`, two prompts) are identical to the previous
build's on Llama 3.2 1B Q8_0 and Q4_K_M, Qwen3-8B Q8_0, Gemma 4 12B Q8_0 and Gemma 4 26B-A4B
Q4_0.

**Prompts.** The prompt GEMMs read the interleaved groups: the int8 GEMM's four-row tile is the
Q8_0 group's layout, so a Q8_0 group is copied into it and a Q4_0 group's nibbles split
(`QMat::fill_tile4`), the SMMLA GEMM's row pairs are zipped from two pieces
(`QMat::fill_pairs4`), whole groups dequantize with de-interleaving loads for the SME and f32
GEMMs, and Q4_K / Q6_K rows are unpacked one super-block at a time back in the file's layout. A
first version that unpacked every row through that last path left Gemma 4 26B-A4B Q4_0 prompts
28% slower (pp512 289-334 against 208-244 tok/s); with the tiles filled from the groups, pp512,
`cpu_repack=0` against `=1`, alternated: 26B-A4B 354 / 334 against 355 / 357, Qwen3-8B Q8_0
223 / 205 against 213 / 201, Llama 3.2 1B Q8_0 1235 / 1288 against 1298 / 1189 (load 3-16).

**Opt-out**: `cpu_repack=0` (`EngineOptions::cpu_repack`, the C API's and the server's option
key), `--cpu-repack off`, or `OXINFER_REPACK=0`; process-wide, applied on load. The CLI notes
what it interleaved (`[113 matrices (1.31 GB) interleaved for the CPU decode kernels]`).

**One core.** `matvec_bench X4=1` (a 512 MB ring of random matrices, both layouts timed in
turn), 2048 x 2048 / 8192 x 2048 / 2048 x 8192, GB/s of weights read:

| format | 1 thread plain / interleaved | 4 threads | 8 threads | 16 threads |
|---|---|---|---|---|
| Q8_0 [3] | 69.9-71.0 / 84.8-85.7 (+21%) | 174-207 / 194-236 (+11-14%) | 210-242 / 215-244 (+1-3%) | 239-283 / 251-283 (level) |
| Q4_0, Gemma 26B-A4B's expert shapes [4] | 36.3-38.7 / 49.1-51.9 (+34-42%) | 92-116 / 107-125 | 112-152 / 118-170 | |
| Q4_K [5] | 32.5-33.3 / 34.5-35.9 (+4-10%) | 99-110 / 103-120 | 145-185 / 155-186 | |
| Q6_K, incl. the 128256 x 2048 LM head [3] | 30.5-30.7 / 32.3-32.5 (+6%) | 101-112 / 106-118 | 153-193 / 159-204 | |

The Q8_0 and Q4_0 kernels gain what the shared activation loads, scale conversions and add
trees saved; Q4_K and Q6_K are bound by unpacking the nibbles and 6-bit values, which
interleaving does not change (Q4_K gained 0-5% until its scales were unpacked four rows at a
time). For comparison, llama.cpp's own repack (`llama-bench --repack 0` / `1`, Llama 3.2 1B,
tg64, load 4) is worth +30% on one core for Q8_0 (43.4 to 56.5 tok/s) and +12% for Q4_K_M
(40.6 to 45.5), +8% and +2% on 8 threads.

**Decode against llama.cpp.** `parity_bench` (tg64, tg32 for the 8B and larger; median of 3)
with `cpu_repack=0` ("before": the previous build's kernels, bit for bit) and `=1` ("after"),
separate processes, alternated with `llama-bench -ngl 0 -dev none` (mean of 3) at the same
thread count, two rounds each (`/`); "all": oxinfer's default (18 threads, decode steps on 16),
llama.cpp on 12. tok/s.

A quieter window (load averages 6-12; the machine is shared):

| model | threads | llama.cpp | before | after |
|---|---|---|---|---|
| Llama 3.2 1B Q8_0 | 1 | 56.0 / 54.9 | 47.0 / 46.7 | 56.8 / 56.0 |
| | 4 | 154.7 / 156.9 | 123.2 / 118.0 | 156.5 / 157.1 |
| | 8 | 191.5 / 156.3 | 160.6 / 161.0 | 164.3 / 161.5 |
| | all | 169.2 / 188.7 | 162.7 / 163.9 | 183.7 / 183.9 |
| Llama 3.2 1B Q4_K_M | 1 | 41.7 / 44.1 | 35.0 / 37.1 | 38.7 / 39.5 |
| | 4 | 139.0 / 142.4 | 124.2 / 123.7 | 129.7 / 128.0 |
| | 8 | 206.3 / 201.4 | 207.1 / 205.8 | 216.1 / 213.1 |
| | all | 226.2 / 222.4 | 241.0 / 240.3 | 250.6 / 246.0 |
| Qwen3-8B Q8_0 | 1 | 9.14 / 9.48 | 8.0 / 8.1 | 9.8 / 9.8 |
| | 4 | 28.0 / 27.9 | 22.8 / 22.5 | 27.7 / 28.1 |
| | 8 | 28.6 / 33.2 | 28.6 / 28.9 | 29.0 / 28.8 |
| | all | 33.0 / 32.1 | 30.0 / 29.8 | 32.1 / 31.9 |
| Gemma 4 12B Q8_0 | 1 | 5.83 / 6.09 | 5.1 / 5.1 | 6.3 / 6.3 |
| | 4 | 17.5 / 17.2 | 14.0 / 14.1 | 17.4 / 17.3 |
| | 8 | 20.7 / 20.5 | 18.2 / 18.2 | 18.0 / 17.7 |
| | all | 20.2 / 20.3 | 19.4 / 19.6 | 20.9 / 20.8 |
| Gemma 4 26B-A4B Q4_0 | 1 | 16.8 / 17.9 | 16.2 / 16.7 | 21.0 / 21.2 |
| | 4 | 48.1 / 47.6 | 44.5 / 45.2 | 55.9 / 55.9 |
| | 8 | 64.9 / 64.5 | 64.8 / 66.8 | 67.8 / 66.3 |
| | all | 61.0 / 62.0 | 68.6 / 68.9 | 69.8 / 68.4 |

Under load: the same with nine busy loops (`perl -e 'while(1){}'`) on the 18 cores while
measuring, on top of other sessions' jobs (load averages 3-37):

| model | threads | llama.cpp | before | after |
|---|---|---|---|---|
| Llama 3.2 1B Q8_0 | 4 | 141.5 / 138.8 | 128.1 / 125.4 | 141.1 / 141.2 |
| | 8 | 130.7 / 106.8 (± 23-28) | 148.3 / 151.6 | 132.3 / 120.3 |
| | all | 7.0 / 4.7 | 132.7 / 149.1 | 153.9 / 156.0 |
| Llama 3.2 1B Q4_K_M | 4 | 115.0 / 113.4 | 106.0 / 106.6 | 112.2 / 108.7 |
| | 8 | 147.4 / 173.4 | 173.0 / 142.9 | 179.5 / 180.3 |
| | all | 6.4 / 6.2 | 172.4 / 173.5 | 174.4 / 162.8 |
| Qwen3-8B Q8_0 | 4 | 22.9 / 22.3 | 19.3 / 19.5 | 22.8 / 23.0 |
| | 8 | 14.3 / 15.2 | 23.8 / 25.9 | 24.1 / 27.0 |
| | all | 0.97 / 1.78 | 23.8 / 25.0 | 26.2 / 27.4 |
| Gemma 4 12B Q8_0 | 4 | 14.3 / 14.1 | 12.5 / 12.4 | 14.5 / 14.6 |
| | 8 | 16.5 / 16.7 | 16.7 / 16.5 | 17.4 / 17.3 |
| | all | 1.23 / 1.17 | 16.4 / 15.8 | 17.4 / 17.3 |
| Gemma 4 26B-A4B Q4_0 | 4 | 43.7 / 44.1 | 41.8 / 41.3 | 49.5 / 48.0 |
| | 8 | 58.3 / 60.7 | 56.5 / 57.7 | 60.0 / 61.6 |
| | all | 3.81 / 1.99 | 51.4 / 51.2 | 56.6 / 54.9 |

So per core oxinfer now matches llama.cpp on Q8_0 (1 and 4 threads: Llama 1B level, Qwen3-8B
and Gemma 4 12B level to 8% ahead; 15-33% faster than before) and is ahead on Q4_0 (26B-A4B
16-25% ahead of llama.cpp on 1-4 threads, 24-30% ahead of before); Q4_K_M per core stays 5-10%
behind llama.cpp (+4-11% over before) and ahead from 8 threads on. On every thread it is given
oxinfer is now level with or ahead of llama.cpp's best count in quiet windows (Llama 1B Q8_0
184 against llama.cpp's 156-192 at 8 and 169-189 at 12), and under load it holds where llama.cpp
does not: llama.cpp's spinning threads collapse to 1-7 tok/s at 12 threads and lose 20-55% at 8
(Llama 1B Q8_0 107-131, Qwen3-8B 14-15), while oxinfer's work-claiming pool on all threads
keeps 70-85% of its quiet speed, and the interleaved kernels add 4-16% there on the Q8_0 and
Q4_0 models (Qwen3-8B 26-27 against 24-25 before; Q4_K_M within the noise). One gap is left: 8 threads on the dense Q8_0 models, where llama.cpp
reaches 192-194 tok/s on Llama 1B and 20.6 on Gemma 4 12B in quiet windows against oxinfer's
161-167 and 18; the interleaved kernels do not change it (the kernels stream at 236-285 GB/s
on 8 threads in `matvec_bench`, so the time goes elsewhere in the step), and in one quiet
minute the same 8-thread setting ran 196-200 tok/s in `decode_ab`, so it is at least partly the
machine's minute-to-minute swings.

**Prompts, after.** pp512 with `cpu_repack=0` against `=1`, two alternations each (median of 3,
load 15-25): Gemma 4 26B-A4B Q4_0 358 / 358 against 382 / 394 (the expert tiles now come
straight from the groups), Llama 3.2 1B Q4_K_M 1322 / 1313 against 1250 / 1301, Qwen3-8B Q8_0
252 / 238 against 238 / 231; with the SMMLA GEMM forced (`OXINFER_I8_MMLA=1`, the aarch64 Linux
default, so that path is exercised here) Qwen3-8B 246 / 235 against 233 / 228 and the 26B-A4B
329 / 337 against 352 / 346. Taking the f32 (SME) and int8 GEMMs apart on Qwen3-8B gave 129 /
136 against 136 / 136 and 236 / 233 against 233 / 214: the Q8_0 prompts are level to a few
percent slower, within this machine's swings, which could not be narrowed further today; the
Q4_K and Q6_K rows are still unpacked one super-block at a time from the groups (a tile fill
like Q8_0's would remove that).

**Memory and load time** (`/usr/bin/time -l`, CLI, `--device cpu`, three alternations):
the peak footprint is the same within 10 MB (Gemma 4 12B Q8_0 13.137-13.145 GB without,
13.140-13.149 GB with; Qwen3-8B 13.77 both; 26B-A4B 15.13-15.14 both), since the tensors are
interleaved in the memory they were read into. Loading takes 0.2-0.35 s longer for the pass
over the weights on the thread pool (12B: 0.95-0.97 s to 1.29-1.31; Qwen3-8B 0.52-0.54 to
0.73-0.76; 26B-A4B 1.01-1.02 to 1.34; Llama 1B Q4_K_M 0.16 to 0.19), with the file in the page
cache. With `--mmap`, a checkpoint too big for memory or a streamed one, nothing is interleaved:
a copy would double the memory for the time of the load and turn clean, reclaimable file pages
into the process's own (as on iOS, where mapping is the default for that reason).

Dropped or not done:

- 8-row interleaving (llama.cpp's `q4_K_8x8`, `q6_K_8x8` with SMMLA): SMMLA issues at half
  SDOT's rate on these cores (round 3), and the four-row SDOT kernels were bound by unpacking,
  not by the dot products;
- other decode block sizes on 8 threads (1, 2, 4 or 16 blocks a thread, 32 KB minimum, in one
  `decode_ab` process, Llama 1B Q8_0, load 3): the default 8 a thread 200 tok/s, 4 198, 16
  198, a 32 KB minimum 199, 2 183, 1 with 32 KB 183, so the 8-thread gap above is not the
  blocks' size;
- interleaving on x86-64: the AVX2 / AVX-512 kernels keep the file's layout (the layout is only
  enabled on aarch64; their portable fallbacks read interleaved rows back if one is ever given).

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

#### Inkling-2-Mini decoding on Metal (November 2026)

The goal of this round was the 125 tok/s of decoding (and 1750 of prefill) reported for
Inkling-2-Mini on a 48 GB M5 Max with EXO (see [Inkling-2 Mini](inkling.md#speed), where that figure
could not be traced). The starting point, best of interleaved runs of `inkling2_bench -n 64 -v 1,4`:
mlx-community 4bit 82.5 tok/s (11.85 ms per one-token pass), unsloth UD-IQ3_XXS 80.2, ggml-org
Q8_0 65.7; prefill was already at about 1750 tok/s. The rule for keeping a change: the f32
reference agreement unchanged and, with `--no-system`, greedy text identical to the previous build
(9 replies of 160 tokens: 3 prompts on mlx-community 4bit, UD-IQ3_XXS and Q8_0; the previous build
gives the same text run after run).

**Before / after** (best of 4-5 interleaved runs of each build, load 7-16; dispatches and barriers per
one-token pass from `OXINFER_INKLING_PROFILE`):

| Checkpoint | tg64 ms / tok/s | one-token pass ms | 4-token pass ms | dispatches / barriers |
|---|---:|---:|---:|---:|
| mlx-community 4bit | 11.88 / 84.1 -> 11.80 / 84.7 | 11.59 -> 11.41 | 21.60 -> 21.34 | 759 / 751 -> 759 / 751 |
| unsloth UD-IQ3_XXS | 12.30 / 81.3 -> 12.30 / 81.3 | 12.02 -> 11.96 | 23.82 -> 23.76 | 759 / 751 -> 759 / 751 |
| ggml-org Q8_0 | 15.08 / 66.3 -> 15.04 / 66.4 | 14.82 -> 14.68 | 24.57 -> 24.55 | 759 / 751 -> 759 / 751 |

The kept changes are worth 0.5-1% (within the noise of this shared machine, whose other jobs made
single runs 10-100% slower); the fusions below, which cut 240 dispatches and 192 barriers, did
not change the time.

**Kept:**

- A decoding pass's first command buffer holds 2 of its 49 steps instead of 8: the GPU starts
  after about 20 us of encoding instead of 90 (the other command buffers are encoded while it runs).
  0.07 ms a token (11.93 -> 11.87 ms, best of 4 interleaved, load 5-12). The same logits.
- The attention's relative bias puts its threads on the distances the key range has
  (`G x nb` items) instead of on `G x 1024` slots of which a short range uses a few: the 33
  threads that had the 3 heads one after the other now take one each. The attention kernel at
  33 keys: 8.5 -> 7.9 us in isolation. The same logits.

**Measured and dropped** (each against the build before it; "mb" is a Swift harness that runs
one kernel 48-200 times in a command buffer with a barrier after each, weights rotated over 48
copies so they come from memory, best of 5):

- **Overlapping the host with the GPU.** With `OXINFER_METAL_TRACE` a one-token pass keeps the GPU
  busy 11.30 ms of the 11.97 between tokens: the rest is encoding the first command buffer (90
  us), commit to GPU start (190 us), completion to the host waking (70-85 us) and host work
  between passes (300 us: the 0.8 MB logits copy, the argmax, the embedding row). A prelaunch was
  built: the next one-token pass (position + 1, same session) encoded and committed while the
  current one runs, its first command buffer waiting on an `MTLSharedEvent` that the next call
  signals once it has written the embedding. 11.80-12.20 -> 11.73-11.83 ms (best of 3, 0.1 ms):
  a CPU-signalled event takes 330-440 us to start the GPU (mb), no less than a commit (40-190
  us after a gap of 150-300 us, 0.4-2 ms after the GPU went idle). A kernel spinning on a word
  the host writes never saw the write (it ran to its cap of 200M polls), and a keep-alive kernel
  queued after a pass only delays the next one (it waits for the spin; start 70-180 us either
  way). Dropped: 1% against a pass that runs whenever the next call is not the predicted one
  (garbage writes at position + 1, the rollback margin one row shorter, sessions that must flush it
  before they are copied, read or dropped).
- **Fusing the small kernels into their neighbours** (all of it bit for bit: the same logits
  digest with each fusion on or off): the attention's merge in its kernel, the routing in the
  router's matvec, the convolution after the experts in the down projections' kernel, the
  message gates' norms in their mix kernel, each by the last threadgroup to finish (a counter per
  token / head / row block in device memory, fences around the atomic, reset by the last). 759 ->
  519 dispatches and 751 -> 559 barriers per token; 12.10 -> 12.15 ms (best of 4, noise 0.1 ms).
  The stages that went were never the cost: an empty dispatch and barrier take 0.7 us on an idle
  GPU (mb), and leaving a kind of stage out altogether (`OXINFER_INKLING_SKIP`, busy machine)
  takes off its kernel's latency, not a barrier's: the merge 0-0.25 ms a token, the convolution
  after the experts 0.18-0.35, the q / kv stage 0.3-0.4, the attention kernel 0.85-1.5, the MLP's
  conv / mix kernel 0.7-0.9, the routed experts 4.3-4.9, the dense projections 1.9-2.5.
- **A faster attention for short ranges** (keys interleaved over the 8 SIMD-groups, so that 33
  keys keep all of them busy instead of one; splits of a fifth of the range, 64 to 256 keys, in
  passes of up to 16 tokens): 33 keys 8.5 -> 6.9 us, 200 keys 10.7 -> 7.2, 500 keys 16.0 -> 11.2
  (mb). Splits of 64 keys whatever the range are slower past 1000 keys (1000: 12.2 -> 15.1 us,
  2000: 22.8 -> 34.8: those read 8-16 MB of f32 cache and are bound by it), hence the fifth. The keys are summed in another order, and the
  greedy text then parted from the previous build's on 3 of the 9 replies (mlx-community 4bit at
  token 25, "accurate" against "careful"; UD-IQ3_XXS twice). Dropped by the equality rule; it is
  the change to make if that rule is relaxed (a few percent at contexts of a few hundred tokens).
- **Cheaper nibble decoding in the dense matvecs.** The MLX 4-bit 3072 x 2304 matvec takes 16.2 us
  on an idle GPU where a stream of its bytes takes 9.4 (mb; 412 GB/s for those 4 MB, 520 GB/s for
  64 MB). Its loads alone with its access pattern (weights, f16 scales, bf16 offsets and 128 B of
  x per lane and chunk) took as long as the whole kernel (32 against 32 us on a busy GPU): the
  decode is not where the time goes. Decoding with half bit patterns (`0x6400 | nibble`, minus
  1024) 35 against 32 us, x staged in threadgroup memory 43, 2 to 8 rows per SIMD-group with every
  row's loads issued first 16.9-30 us against 16.2, 1 to 8 SIMD-groups per threadgroup no change.
  MLX's `qmv_fast` decodes a value with a mask, a conversion and an FMA, as these kernels do, and
  its `fp_quantized` E2M1 decoding is the same bit trick as `e2m1x4`: nothing to take from it.
  (mlx-lm itself decodes this checkpoint at 29-35 tok/s here.)
- **A mixed 3/4-bit layout**: UD-IQ3_XXS already is one (IQ3_XXS experts, 4-6 bit dense) and decodes
  at the 4-bit file's speed (81 against 85 tok/s): at 3 bits the expert kernels are bound by the
  IQ3 grid decoding, so fewer bytes do not turn into speed. Not pursued.
- **The LM head** (0.2-0.7 ms a token): a shortlist cannot promise the full head's argmax without
  a bound on the other rows, and the only cheap one (|w . h| <= |w| |h|) is far too loose for
  logits within 0.1 of each other. Not built.

**Speculative decoding** (the CLI, 256 tokens, greedy, `--no-system`; every output identical to
the plain one; one run each, load 5-9):

| Checkpoint | prompt | plain | prompt lookup (default) | DSpark draft (`--draft`) |
|---|---|---:|---:|---:|
| mlx-community 4bit | code | 81.6 | 86.1 (1.09 tok/forward, 39% of drafts) | 79.2 (1.20, 61%; unsloth's Q8_0 draft by path) |
| | chat | 82.6 | 93.7 (1.18, 78%) | 82.7 (1.03, 28%) |
| unsloth UD-IQ3_XXS | code | 81.5 | 85.0 (1.08, 71%) | 81.5 (1.50, 65%) |
| | chat | 81.8 | 80.4 (1.00, 0%) | 79.4 (1.05, 38%) |
| ggml-org Q8_0 | code | 66.1 | 68.6 (1.08, 42%) | 75.1 (1.62, 57%) |
| | chat | 64.9 | 65.1 (1.05, 35%) | 63.3 (1.05, 35%) |

A verification pass of 4 tokens costs 21.3-24.6 ms against 11.4-14.7 for one (the tokens read
their own experts: 4 tokens touch 26-29 of the 128), so a round of 3 drafts accepted at a rate a
gives `1 + a + a^2 + a^3` tokens for about 25 ms: 125 tok/s needs 3.2 tokens a round, an
acceptance of about 87% at every slot. Prompt lookup gets 1.00-1.18 tokens per forward on these
prompts, the DSpark draft 1.20-1.62 on code (57-65% of drafts) and 1.03-1.05 on chat; where
drafting does not pay the cost-aware rounds stay within 3% of plain speed, and the best gain is
+13% (Q8_0 code with DSpark, chat with lookup on the 4-bit file).
Neither better acceptance nor cheaper verification is in reach on these prompts: the experts of
the drafted tokens are the floor of a verification pass.

**What is left to 125 tok/s.** A one-token pass is 11.3 ms of GPU time for 2.7-2.9 GB of
weights (mlx-community 4bit), which streams in 5.2-5.6 ms at the 520 GB/s this GPU reads; the
other 6 ms are the latency of a chain of about 15 dependent kernels per layer application, 48
times. 125 tok/s is 8 ms a token including 0.5-0.7 ms of host and launch gaps, so the chain would
have to lose about 4 ms; this round found that its stages cost little each (0.7 us of dispatch
and barrier) and that the kernels that run as one threadgroup per token (the mixes, the norms,
the routing) or as a few threadgroups (the attention at short ranges: 8 of 40 cores) are where
the time is. Making those spread over the GPU changes their sums' order, which the equality rule
excludes; even all of it (the attention change above, the mixes at 512 threads instead of 1024:
25% faster in mb on a busy GPU) is worth perhaps 0.5-1 ms. A realistic ceiling for plain decoding on this design is 90-95 tok/s
for the 4-bit files, and the published 125 is not reproducible here (but see round 2 below:
98.5 tok/s on the 4-bit file once the order of the sums was allowed to change).

**Round 2 (November 2026): the latency chain.** The rule changed for this round: a change is kept
when its agreement with the float32 reference (`inkling2_check`: decode KL, top-1 agreement) is no
worse than the previous build's within the noise, and greedy text stays coherent; summation
order may change. The previous build gives the same text run after run, this one parts from it
on 3 of the 9 replies of 160 tokens (at bytes 155, 240 and 431 of 504-653: "Need be careful" for
"Need be accurate", "simple decorations" for "simple jewelry"), all coherent.

Before / after (best of 4 interleaved runs of `inkling2_bench -n 64 -r 2 -v 1,4`, load 2.4-4.6):

| Checkpoint | tg64 ms / tok/s | one-token pass ms | 4-token pass ms |
|---|---:|---:|---:|
| mlx-community 4bit | 11.69 / 85.5 -> **10.15 / 98.5** | 11.28 -> 9.97 | 21.23 -> 20.11 |
| unsloth UD-IQ3_XXS | 12.18 / 82.1 -> **10.61 / 94.3** | 11.77 -> 10.42 | 23.57 -> 22.26 |
| ggml-org Q8_0 | 15.00 / 66.7 -> **13.58 / 73.6** | 14.63 -> 13.36 | 24.31 -> 23.22 |

The master this round is rebased onto (8ca1574) decodes at the previous round's speed (mlx-community
4bit: 11.68 ms, the previous round 11.68, this round 10.12; best of 3 interleaved, load 2.5-4.3).
At longer contexts (`inkling2_ctx_bench`, mlx-community 4bit, 48 tokens): 10.4 ms a token at 300
tokens, 11.3 at 2000, 11.9 at 8000.

**Accuracy.** The reference agreement is measured with the prompt fed in passes of 16 tokens
(`inkling2_check --chunk 16`), which run the decoding kernels in float32 throughout: fed as one
prefill pass, the prompt's activations are rounded to half precision for the GEMMs, and a change
of the last bit anywhere moves the following decode KL by one to two orders of magnitude
(the attention change below took the release's 16-step KL from 2.6e-6 to 3.0e-4 that way, with
the prefill KL going from 1.8e-4 to 6.3e-7). Mean decode KL of the release against its float32
reference on three prompts (24 tokens + 16 decoded, 620 + 96 (code), 357 + 128 (prose)); the
top-1 agreement is 16/16, 96/96 and 128/128 for every build:

| Build | 24 + 16 | 620 + 96 | 357 + 128 | tg64 ms (mlx-community 4bit) |
|---|---:|---:|---:|---:|
| previous round (644f76c) | 3.3e-12 | 2.1e-11 | 9.1e-13 | 11.71 |
| attention across the channels, splits of a fifth, router per threadgroup (691be1d) | 4.4e-12 | 2.1e-11 | 8.3e-13 | 11.05 |
| convolution apart from the mix, mix8, routing in a SIMD-group, norm loads (62adac6) | 4.3e-12 | 2.7e-11 | 9.1e-13 | 10.55 (11.19 before, same batch) |
| merge in the attention kernel (62832aa, bb4c533) | 3.3e-12 | 2.4e-11 | 8.2e-13 | 10.40 (10.47) |
| 2 keys a SIMD-group at a time (feee754) | 5.7e-12 | 2.3e-11 | 7.9e-13 | 10.24 (10.30) |

The builds measured on the way (variants not kept included) span 1.8e-11 to 2.8e-11 on the
620-token prompt: that is the noise. On the quantized checkpoints the decode KL is the
quantization's and the same to four digits before and after (mlx-community 4bit 2.352e-3 /
5.530e-2 / 1.509e-2, UD-IQ3_XXS 2.843e-3 / 3.623e-2 / 1.050e-2, Q8_0 7.137e-4 / 1.155e-2 /
8.766e-4; top-1 16/16, 83/96, 126/128; 16/16, 87/96, 124/128; 16/16, 94/96, 128/128).

**How it was measured.** `OXINFER_INKLING_PROFILE` ran each stage in a command buffer of its own
and waited for it, so the GPU idled between stages and ran them at a lower clock: its per-kernel
times summed to 39.6 ms for an 11.4 ms pass. It now commits the stages back to back and reads
their times at the end (10.4-11.7 ms of stage time for passes of 10.2-11.5): a stage's time there
includes about 2 us of command buffer start, but kernels can be compared in place, and a
variant of the Metal source (`OXINFER_INKLING_METAL`) needs no rebuild. Two other methods misled:
leaving a kind of stage out (`OXINFER_INKLING_SKIP`) leaves stale or garbage inputs to the routing,
whose experts then come from the cache (leaving the routing out took 5.2 ms off a 12.5 ms pass), and
a kernel timed alone in a harness runs on a GPU whose clock follows the load (an 8 x 64-FMA kernel
with a barrier: 1.35 to 3.6 us depending on what else ran). In a pass, an extra stage of an empty
kernel costs 1.1-1.4 us (passes padded with 192 or 384 of them).

What the stages cost (profile, mlx-community 4bit, us a stage, 48 layer applications):

| Kernel | before | after |
|---|---:|---:|
| attention (`ink_attn` -> `ink_attv`) | 21.3 | 11.7 |
| convolution + MLP mix (`ink_conv_mix`, one threadgroup) | 17.5-20.9 | -> `ink_conv` 6.5 + `ink_mix8` 7.9 |
| attention input mix (`ink_mix` -> `ink_mix8`) | 11.5 | 7.9 |
| router logits (`ink_mvf` -> `ink_mvr`) | 7.9 | 4.9 |
| routing (`ink_route`) | 6.8 | 4.5 |
| norms (`ink_norm`) | 7.4 | 6.3 |
| attention merge (`ink_attn_merge`) | 3.9 | 0 at fewer than 64 keys |

**Kept:**

- **Attention with the lanes across a head's channels** (`ink_attv`, passes of up to 16 tokens):
  a SIMD-group takes a key at a time with its 32 lanes on the head's 128 channels, so that a key
  is one 16-byte load per lane and a SIMD-group sum per query head, the keys interleaved over the
  8 SIMD-groups; the key and value loads of 2 keys are issued together. The previous kernel had a
  lane per key walk its 128 channels in 4 rounds of 8 loads, then the values in rounds of 8 keys.
  Splits of a fifth of the range, 64 to 256 keys (the change of the previous round dropped for the
  equality rule). 21.3 -> 11.7 us a layer at the ranges of tg64; 33 -> 28 us at 2000 tokens of
  context, 49 -> 44 at 8000 (the earlier kernel with the same splits). 1, 4 and 8 keys at a time:
  10.7, 13.0 and 16.9 us at tg64, 32, 30 and - at 2000 (more registers: fewer SIMD-groups in
  flight); splits of at most 128 or 64 keys: slower at 8000 (11.9 -> 12.1 / 12.3 ms a token).
- **The tokens of one split merge in the attention kernel** (the sink and the gate), the merge
  kernel leaving them out: one stage fewer a layer below 64 keys of range. 10.47 -> 10.40 ms.
- **The router's logits a threadgroup per expert** (`ink_mvr`: 128 threadgroups of 128 threads,
  every thread's 4-5 float4 loads issued together) instead of a SIMD-group per expert walking
  2304 values: 7.9 -> 4.9 us; 11.19 -> 11.05 ms a token.
- **The convolution after the attention apart from the MLP's mix.** `ink_conv_mix` (the
  convolution, then the mix, in one threadgroup of 1024 per token) walked each thread's 2-3
  channels three times, every walk a round trip to memory; the convolution as `ink_conv` (9
  threadgroups, a channel a thread) and then the mix is one stage more and 3-5 us less.
- **`ink_mix8`**: the mix of at most 8 block states (Inkling-2-Mini has at most 5), whose thread
  keeps its 3 channels' x and states in registers from the first walk to the output (8 + 2
  accumulators instead of 16 + 2): 11.5 -> 8.8 us. The same with 16 states: 15.8 us (spills).
- **The routing in one SIMD-group** (`ink_route`, at most 128 experts: 4 a lane, no threadgroup
  barriers; the k picks kept by lanes 0..k): 6.8 -> 4.5 us. **The norms' loads** (12 a thread)
  issued together: 7.4 -> 6.3 us. Together with the two above: 11.19 -> 10.55 ms a token.

**Measured and dropped:**

- **Mixes over the whole GPU in two stages** (`ink_mixp`: per block of 128 or 256 channels, the dot
  products of the queries with the states and of every pair of states, from which the output's
  norm follows without a pass over it; `ink_mixo`: the weights and the block's outputs): 9.7 + 8.4
  us against 11.5 for the one-threadgroup mix, 11.46-11.53 ms a token against 11.61-11.87.
- **One threadgroup of 768 threads with all of a thread's loads issued first** (`ink_mix3`, the
  convolution in it): 16.1 us against 11.4 / 18.0; the routing with 8 experts a lane and its picks
  in 16-entry arrays: 10.9 us against 6.8; the keys' and values' convolution with every lag's loads
  first: 6.2 -> 9.4 us; the convolution with its 16 expert slots and 8 lags in arrays: 8.4 -> 9.1-9.4
  us. In
  each the arrays took the registers that let more SIMD-groups hide their loads' latency; the kept
  versions issue their loads together with arrays of at most 4-12 values.
- **The relative bias's 16 loads issued together** in the earlier attention kernel: no change
  alone (kept in `ink_attv`).
- **The one-token matvecs in other shapes** (2 or 4 SIMD-groups a row, 2 rows a threadgroup): the
  attention's projections 29.3 -> 32-33 us, the LM head 485 -> 498-564 us; the output projection
  13.0 -> 11.8 with 4 SIMD-groups a row, not enough to pay for the rest.

**Speculative decoding** (the CLI, 256 tokens, greedy, `--no-system`, one run each, load 2.4-3.3;
DSpark on the 4-bit file with unsloth's Q8_0 draft by path):

| Checkpoint | prompt | plain | prompt lookup (default) | DSpark draft |
|---|---|---:|---:|---:|
| mlx-community 4bit | code | 98.5 | 97.9 (1.08 tok/forward, 33% of drafts) | 97.2 (1.17, 59%) |
| | chat | 98.5 | 111.9 (1.27, 79%) | 94.9 (1.03, 32%) |
| unsloth UD-IQ3_XXS | code | 94.1 | 95.5 (1.05, 55%) | 91.5 (1.16, 55%) |
| | chat | 93.9 | 92.3 (1.01, 7%) | 89.8 (1.04, 35%) |
| ggml-org Q8_0 | code | 73.5 | 75.9 (1.08, 43%) | 78.5 (1.38, 55%) |
| | chat | 73.9 | 74.1 (1.03, 27%) | 72.7 (1.05, 35%) |

Plain decoding gained more than verification passes (one-token passes 12% faster, 4-token passes
5%: the experts of 4 tokens, about 27 of the 128, are most of a verification pass), so drafting
pays less than before: +14% at best (chat with lookup on the 4-bit file, 1.27 tokens a forward),
-4% to +7% elsewhere. Eleven of the twelve speculative outputs are the plain ones; the 4-bit chat
with DSpark parts from it at byte 364 of 753. That is not new: a verification pass and one-token
passes give logits that differ by up to 1.3e-5 (8 tokens after 56 or 200, a third to all of the
logits differing in the last bits) on the previous round's build as on this one, so a near tie can go either way;
the previous build happens to give the same text on that prompt.

**What is left to 125 tok/s** (8 ms a token). A one-token pass is now 10.4-10.8 ms of stage time:
about 7.0 ms of the bandwidth kernels (routed experts 54.6 + 32.6 us a layer, 440 and 370 GB/s;
the attention's projections 29.7 us, 380 GB/s; the output projection 13.2 us, 300 GB/s; the LM
head 0.5 ms, 500 GB/s; the 4 MXFP8 layers' experts 0.6 ms) and 3.5 ms of the others (the mixes
7.9 us x 89, the attention 11.7 x 48, the convolutions 6.5 x 96, the message gates 7.8 + 6.3 + 6.0 x
48, the keys and queries 6.3, the router 4.9 and routing 4.5 x 48), plus 0.3-0.5 ms of host and
launch. Even every small kernel at a stage's floor (about 2 us: 1.1-1.4 of stage and one round
trip) would leave about 9 ms, and the matvecs at 480 GB/s another 0.8 ms less: 125 tok/s needs
both and more. The next steps in reach: the attention's key / query stage folded into the
attention kernel (the queries' clipping is a SIMD-group sum there), the convolution after the
experts into the next layer's mix (a stage a layer, but a verification pass's tokens would then
sum each other's expert slots), and the output projection and routed experts' down projection,
the two matvecs furthest from the GPU's bandwidth.

**Round 3 (November 2026): the host gap, and the stages beside the bandwidth kernels.** The rule
of round 2 (decode KL against the float32 references within the noise of the previous build,
measured with `--chunk 16`; top-1 agreement on the three references; coherent greedy text with
`--no-system`). Every GPU kernel change of this round computes the logits of the build before it
bit for bit (the logits digest of `INKLING2_BENCH_SUM` unchanged after each, the tokens of a
64-token greedy run the same): the release's decode KL is the previous round's to four digits,
5.716e-12 / 2.272e-11 / 7.911e-13 (24 + 16, 620 + 96, 357 + 128), top-1 16/16, 96/96 and 128/128.
The one thing that rounds differently is the embedding norm of the chained greedy decoding
(below, a sum over the threadgroup rather than the host's sequential sum): the 9 greedy replies
of 160 tokens (3 prompts on mlx-community 4bit, UD-IQ3_XXS and Q8_0) are those of the previous
round's build, byte for byte.

Before / after (best of 4 interleaved runs of `inkling2_bench -n 64 -r 2 -v 1,4`, load 1.9-4.5;
"host loop" is `tg64` as measured before, a `forward` and the host's argmax per token, "chained"
is `--chain`, greedy decoding as the CLI now runs it):

| Checkpoint | tg64 ms / tok/s before | host loop after | chained after | one-token pass ms | 4-token pass ms |
|---|---:|---:|---:|---:|---:|
| mlx-community 4bit | 10.13 / 98.7 | 9.61 / 104.1 | **9.05 / 110.5** | 9.95 -> 9.32 | 20.03 -> 19.86 |
| unsloth UD-IQ3_XXS | 10.68 / 93.6 | 10.24 / 97.7 | **9.69 / 103.2** | 10.45 -> 9.91 | 22.42 -> 22.25 |
| ggml-org Q8_0 | 13.47 / 74.2 | 12.94 / 77.3 | **12.38 / 80.8** | 13.29 -> 12.65 | 23.31 -> 23.05 |

**Kept** (mlx-community 4bit; profile times are `OXINFER_INKLING_PROFILE` stage times, us a stage):

- **Greedy decoding chained on the device** (`Inkling2Model::decode_greedy`, see
  [Inkling-2 Mini on the GPU](inkling.md#on-the-gpu)). With `OXINFER_METAL_TRACE` a one-token
  pass kept the GPU busy 9.8 of the 10.18 ms between tokens: 135-200 us from the first commit to
  the GPU starting (it had gone idle), 60-90 us from completion to the host waking, and the host's
  copy of 0.8 MB of logits, its argmax and the next embedding. A pass now ends with the argmax of
  its capped logits into a token slot (the softcap kernel's threadgroups reduce to partials, one
  threadgroup to the token: the host's rule, the first index of the largest) and the next pass
  starts from that token's embedding row, dequantized on the device (MLX affine rows, bf16, any
  GGUF type of the decode kernels) and normed; the host encodes and queues pass k + 1 while pass k
  runs and only reads a token after each. 10.15 -> 9.60 ms a token. `generate::decode` takes it
  for plain greedy sampling; prompt lookup takes it for its rounds without a draft (below).
- **The convolutions in the kernels before them** (passes of one token): the experts' down
  projections and the convolution after them in one kernel (`_dconv`: a SIMD-group per slot
  computes `_gsum`'s rows, the slots are summed in order in threadgroup memory, then the
  convolution per channel), and the attention's output projection and its convolution in one
  (`ink_mvc_*`: a row per SIMD-group with the decode library's one-token sums, the convolution
  by lane 0): two stages fewer a layer. 33.8 + 6.6 -> 34.3 us and 12.9 + 6.6 -> 14.7 us; 9.66 ->
  9.39 ms a token (chained).
- **Queries, keys and values in the attention kernel** (`ink_attf`, passes of up to 8 tokens):
  each threadgroup scales and clips its query heads' queries and relative vectors as `ink_q`
  did, and computes the keys and values of the pass's tokens in its split's range as `ink_kv`
  did, with the same sums in the same order (a channel a thread, SIMD-group sums, then the head's
  four in order), rounded as the cache holds them; the threadgroup whose split ends at its token
  writes that token's row and rings. 11.8 + 6.1 -> 16.2 us a layer; 9.29 -> 9.18 ms a token,
  4-token passes 20.10 -> 19.85 ms.
- **The routing in the gate / up kernel** (`_rpair`, passes of one token): every SIMD-group runs
  `ink_route`'s single-SIMD-group routing and takes the expert of its group's pick; one
  threadgroup writes the picks for `_dconv`. 52.8 + 4.5 -> 53.7 us; 9.30 -> 9.02 ms a token.
- **Prompt lookup on the chain**: a lookup round without a draft worth verifying runs chained
  passes, committing and streaming each token as it comes (`RoundDecoder::step_emit`), until the
  lookup finds a draft worth verifying; the pass already queued then still runs, and its token
  starts the verification round (`generate::Chain::Last`), so no queued pass is thrown away.

**Measured and dropped:**

- **The experts' kernels' memory access.** Rows of 2 or 4 outputs a row group, the x chunk loaded
  once for them: down projection 31.8 -> 31.1 / 35.6 us. The weight loads of 3 or 9 chunks issued
  before their dot products: 52.1 / 31.8 -> 54.2 / 32.2 and 66 / 39.7 us. What bounds them: with
  only the weight loads (the dot products replaced by an xor of the bits) the down projection
  takes 27.9 us of its 32 and the gate / up 48 of 52; the x loads alone 10.8 and 17.4 (they
  overlap the weights'); a flat stream of the same 8 experts' 11.9 MB 25.4 us (500 GB/s with
  the stage's start). The expert kernels are within 10% of what this GPU streams in a stage of
  that size: about 3.7 us of a stage's time goes to its start, ramp and tail (the LM head's 260 MB
  stream at 550 GB/s).
- **Dense one-token matvecs with several lanes a row** (`ink_mv1_*`, as the expert kernels: 4 to
  32 lanes a row, 4 SIMD-groups a threadgroup): the attention's projections 28.8 -> 29.7-29.9 us,
  the output projection 12.8 -> 12.4 (16 or 32 lanes), the gates' down projections 5.9 -> 7.1-27.
  And the attention's projections read twice in a row: 28.9 then 23.7 us, so the system cache
  keeps little of 11.7 MB, and prefetching the next stage's weights during small kernels was not
  tried.
- **The convolutions in the next mix** (the mix's one threadgroup a token computing its stream
  from the convolution, writing the stream and the ring, the block state pushed from the mix's own
  norm of x): mix 8.75 + convolution 6.41 -> 18.55 us, and the mix alone got slower with the code
  in it (7.54 -> 8.75 us: registers); 10.17 -> 10.59 ms a token. The convolution belongs with the
  kernels spread over the GPU before it (`_dconv`, `ink_mvc_*` above).
- **The convolution's loads issued first** in `_dconv` and `ink_mvc_*` (the ring, the taps and the
  stream loaded before the dot products): 34.1 -> 35.0 and 14.7 -> 16.5 us (registers held over
  the main loop); 9.04 -> 9.23 ms a token.

**Speculative decoding** (the CLI, 256 tokens, greedy, `--no-system`, one run each, the two builds
interleaved, load 1.8-3.5; DSpark with unsloth's Q8_0 draft by path; "same": the output is the
same build's plain one):

| Checkpoint | prompt | plain | prompt lookup (default) | DSpark draft |
|---|---|---:|---:|---:|
| mlx-community 4bit | code | 98.8 -> **108.9** | 99.1 -> 108.1 (1.02 tok/forward, 19% of drafts) | 99.4 -> 106.7 (1.17, 67%) |
| | chat | 99.0 -> **109.6** | 113.4 -> **124.8** (1.30, 85%) | 96.2 -> 102.2 (1.02, 26%; parts from plain) |
| unsloth UD-IQ3_XXS | code | 94.3 -> **102.2** | 95.8 -> 101.7 (1.02, 29%) | 93.2 -> 96.5 (1.12, 54%) |
| | chat | 94.4 -> **102.0** | 93.1 -> 100.7 (1.00, 0%) | 90.6 -> 95.1 (1.04, 35%) |
| ggml-org Q8_0 | code | 74.4 -> **80.1** | 76.5 -> 79.5 (1.01, 15%) | 78.9 -> **83.2** (1.42, 58%) |
| | chat | 74.7 -> **80.0** | 74.4 -> **81.9** (1.04, 89%) | 73.3 -> 76.6 (1.06, 41%) |

Plain decoding gained 7-11%, prompt lookup 4-10%. Prompt lookup now verifies fewer drafts (its
cost-aware rounds compare a draft's expected tokens with the time of the chained passes, which
got cheaper than a verification pass by more than the drafts' worth): within 1% of plain decoding
where drafts rarely hold, +14% on the chat prompt with the 4-bit file (124.8 tok/s, 1.30 tokens
a forward). A verification pass of 4 tokens still costs about twice a one-token pass (19.9
against 9.3 ms): its tokens touch about 27 of the 128 experts, 3.4 times the bytes of one
token's 8, and the kernels fused this round take passes of one token (the attention's up to 8).
The DSpark draft pays only on Q8_0 code (+4% over plain, 1.42 tokens a forward). As in round 2,
the 4-bit chat output with DSpark parts from the plain one (a near tie between a verification
pass's logits and a one-token pass's); all the other outputs are the plain ones.

**What is left to 125 tok/s** (8 ms a token). A chained one-token pass is 9.0 ms (mlx-community
4bit), all of it GPU time: 9.7 ms of stage time in the profile, of which about 6.9 ms in the
bandwidth kernels (gate / up with the routing 53.5 us x 44, down with the convolution 34.1 x 44,
the attention's projections 28.9 x 40 and 30.0 x 8, the output projection with its convolution
14.7 x 48, the LM head 0.47 ms, the MXFP8 layers' experts 0.55 ms, the router 4.8 x 48, the
gates' down projections 5.8 x 48) and 2.8 ms in the others (the attention with its queries and
keys 16.4 x 48, the mixes 7.8 x 89, the gates' forget / gain 7.6 x 48 and norms 5.9 x 49). The
bandwidth kernels stream at 430-500 GB/s against about 550 for a long stream; the stage's start
and ramp (about 3.7 us) is a quarter of the short ones' time. Reaching 8 ms needs about 1 ms
more: the message gates (three stages, 19 us, on 24 of the 48 layer applications: their norm
could go into the kernels that read their output, the router and the experts for the MLP's gate),
the kernels fused this round for passes of several tokens (verification would gain the same 2-3
stages a layer), and the dense projections' and experts' kernels at the stream's floor.

**Round 4 (November 2026): the message gates' norms, the attention's merge, verification
passes.** The rule of rounds 2-3 (decode KL against the float32 references within the noise of
the previous build, measured with `--chunk 16`; top-1 agreement on the three references; coherent
greedy text with `--no-system`). The 9 greedy replies of 160 tokens (3 prompts on mlx-community
4bit, UD-IQ3_XXS and Q8_0) are the previous round's byte for byte, and so are the plain outputs of
the speculative runs below.

Before / after (best of 4 interleaved runs of `inkling2_bench -n 64 -r 2 -v 1,4 --chain`, load
3.1-5.4; "one-token pass" and "4-token pass" are `verify1` / `verify4`):

| Checkpoint | tg64 chained ms / tok/s before | after | one-token pass ms | 4-token pass ms |
|---|---:|---:|---:|---:|
| mlx-community 4bit | 9.04 / 110.6 | **8.86 / 112.9** | 9.27 -> 9.13 | 19.81 -> 19.68 |
| unsloth UD-IQ3_XXS | 9.64 / 103.7 | **9.44 / 105.9** | 9.88 -> 9.69 | 22.18 -> 22.21 |
| ggml-org Q8_0 | 12.41 / 80.6 | **12.17 / 82.2** | 12.61 -> 12.37 | 23.02 -> 22.91 |

At longer contexts the in-kernel merge below counts too (`inkling2_ctx_bench`, mlx-community
4bit, host loop, best of 3 interleaved): 300 tokens 103.6 -> 106.3 tok/s, 2000 tokens 94.1 -> 96.6.
Profile stage time of a one-token pass (mlx-community 4bit, tg32): 9.67 -> 9.35 ms; of a
4-token pass 20.36 -> 20.11 ms.

**Accuracy** (mean decode KL against the float32 references, `inkling2_check --chunk 16`, the
release; top-1 16/16, 96/96 and 128/128 for every build):

| Build | 24 + 16 | 620 + 96 | 357 + 128 | tg64 chained ms (mlx-community 4bit) |
|---|---:|---:|---:|---:|
| previous round (ef5f3f8) | 5.716e-12 | 2.272e-11 | 7.911e-13 | 9.05 |
| the MLP's gate norm in the router and experts, the merge in the attention kernel (e2922fd) | 6.276e-12 | 2.478e-11 | 7.878e-13 | 8.97 |
| the attention gates' norms in the projections, rows of one SIMD-group (67dded8; the release's bf16 rows not yet) | the same | | | 8.83 |
| routing and grouping in one kernel (2f4d064; bit for bit) | the same | | | |
| the attention gates' norms on every format (5c7a3bf) | 5.845e-12 | 2.373e-11 | 7.588e-13 | 8.86 |

All within the 1.8e-11 to 2.8e-11 that round 2's builds spanned on the 620-token prompt. The
quantized checkpoints' decode KL is the previous round's to four digits after every change
(mlx-community 4bit 2.352e-3 / 5.530e-2 / 1.509e-2, top-1 16/16, 83/96, 126/128; UD-IQ3_XXS
2.843e-3 / 3.623e-2 / 1.050e-2, 16/16, 87/96, 124/128; Q8_0 7.137e-4 / 1.155e-2 / 8.766e-4, 16/16,
94/96, 128/128).

**Kept** (mlx-community 4bit; profile times are `OXINFER_INKLING_PROFILE` stage times, us a stage):

- **The MLP's message gate leaves its norm to the kernels after it.** `ink_gfg` writes the gated
  input times the norm's weights and, per block of 64 channels, the sum of the squares of the
  unweighted values (one threadgroup sum); the router's matvec (`ink_mvr`) and the experts' gate /
  up kernels (`_rpair`, `_gpair`) sum those 36 partials in a SIMD-group (`ink_gscale`, the same
  sum in each) and scale their dot products by the norm. The norm stage (5.9 us) goes on the 24
  gated layer applications; `ink_gfg` 7.5 -> 7.8 us, the router 4.75 -> 4.9, the gate / up kernel
  +0.4.
- **The attention's three gates the same way**, the scale in the projections after them:
  passes of one token run `ink_mvs_*`, the decode library's one-token matvec (`mv_acc` and
  `mv_finish`'s sums for the matrix's rows and SIMD-groups a threadgroup) times the gate's norm;
  passes of several tokens the decode library's multi-token matvecs and then `ink_gsc`, the same
  product, so that a verification pass still gives the one-token logits. One stage fewer on 24
  layer applications; profile 9.68 -> 9.42 ms; tg64 chained 8.97 -> 8.83 ms; UD-IQ3_XXS 10.12 ->
  9.99 ms of stages.
- **A token's key splits merge in the attention kernel**: past 64 keys (two or more splits) the
  last split of a (token, key head) to finish, by a counter in the arena (fences around the atomic,
  reset by the last), merges the splits as `ink_attn_merge` did, with its arithmetic (the same
  logits bit for bit). Done as that kernel does it, the merge took 6.5 us in the last threadgroup;
  with its loads spread over the threadgroup (the splits' maxima and sums a thread each, the
  weights per head, then every split's partial of an output loaded together) 4.4 us. At 300
  tokens of context: attention 16.7 + merge 6.9 -> 21.7 us a layer; 10.10-10.17 -> 9.98-10.00 ms of
  stages.
- **Passes of 2 to 16 tokens route and group their experts in one kernel** (`ink_route_group`:
  `ink_route`'s single SIMD-group routing, a SIMD-group a token, then `ink_xgroup`'s grouping in
  the same threadgroup; the same `sel`, `grp` and `asg`): 4.2 + 5.6 -> 7.0 us a layer of a 4-token
  pass.
- **`inkling2_bench --exact N`**: N tokens as one verification pass against the same N one at a
  time after prompts of 64 and 200 tokens. Q8_0, UD-IQ3_XXS and the release: 0 of the 600174
  logits of 3 tokens differ, before and after this round. mlx-community 4bit: up to 1.4e-5, a third
  to most of the logits, before this round too: the decode library's multi-token and one-token
  matvecs of MLX's affine 4-bit rows (`wdec` / `wdot` against `mdot`) round differently
  (`decode_mvk_matches_single_token` checks Q4 without the affine offsets). That is the near tie
  that made the 4-bit chat output with DSpark part from the plain one in rounds 2 and 3 (it no
  longer does on that prompt, by chance).

**Measured and dropped:**

- **Multi-token dense matvecs of Inkling's own** (`ink_mvk_*`: `ink_mvc_*`'s one-token sums for
  every token, 4 SIMD-groups a threadgroup of 1, 2 or 4 rows each, each token's x chunk loaded once
  for the rows): the attention's projections in a 4-token pass 66 us a layer with the decode
  library's kernel against 85 / 82 / 93; the weights decoded once a chunk (`wdec`) 76 / 103 / 382
  (registers); the x chunks loaded coalesced (lane l the l-th 16 bytes of every 512, a wrong
  result, for timing) 86 / 81 / 94: the x loads are not what bounds it. The decode library's
  kernel costs about 9 us a stage a token (1 token 27.5 us, 2 47.8, 3 56.8, 4 65.9, 8 122): a
  4-token pass's dense matvecs take 2.3 times a one-token pass's, the experts of its tokens
  (about 27 of the 128) 2.2 to 2.4 times theirs.
- **The convolution after the output projection in the multi-token matvec** (`mvk_body`, then the
  convolution by the lane that wrote each row, token by token: the same logits): 26.8 + 7.1 ->
  37.0 us; a lane per (channel, token) 33.2. The convolution stage is spread over the GPU and
  cheap; the matvec's tail is not.
- **Routing and grouping inside the experts' gate / up kernel** for passes of 2 to 8 tokens (every
  SIMD-group routing the pass's tokens and numbering the experts by a vote per token, threadgroup
  (0, 0) writing the tables; the same logits): 127 + 4.2 + 5.6 -> 305 us a layer. Every
  SIMD-group's weight loads wait for its routing of all the tokens.
- **Prefetching the next stage's weights beside a small kernel** (a kernel reading a word of
  every 128 bytes, dispatched in the same stage): all of the attention's projections (11.7 MB)
  read beside the attention's input mix take the projections' stage 28.3 -> 25.4 us, and the mix's
  stage 7.8 -> 16.6 (it streams at the full rate beside it); a tenth to half of them 8.85 -> 8.86 to
  8.95 ms a token. The output projection read beside the attention kernel: 14.5 -> 12.5 us, the
  attention 15.8 -> 18.8. The system cache gives back about a quarter of a second read's time,
  and the small stages are too short to hide the reads.

**Speculative decoding** (the CLI, 256 tokens, greedy, `--no-system`, one run each, the two builds
interleaved, load 3.0-4.2; DSpark with unsloth's Q8_0 draft by path; every output is the same
build's plain one, and the plain outputs are the previous build's):

| Checkpoint | prompt | plain | prompt lookup (default) | DSpark draft |
|---|---|---:|---:|---:|
| mlx-community 4bit | code | 109.1 -> **111.4** | 108.0 -> 110.7 (1.02 tok/forward, 19% of drafts) | 106.2 -> 108.0 (1.17, 67%) |
| | chat | 109.1 -> **110.9** | 123.4 -> **126.5** (1.30, 84%) | 101.6 -> 103.8 (1.02, 28%; before: parts from plain) |
| unsloth UD-IQ3_XXS | code | 102.1 -> **104.0** | 101.5 -> 103.2 (1.02, 29%) | 96.2 -> 98.9 (1.16, 64%) |
| | chat | 102.0 -> **103.9** | 100.5 -> 102.0 (1.00, 0%) | 94.9 -> 97.2 (1.04, 36%) |
| ggml-org Q8_0 | code | 79.8 -> **81.2** | 79.3 -> 80.6 (1.01, 15%) | 82.7 -> **84.2** (1.42, 60%) |
| | chat | 79.8 -> **81.2** | 81.4 -> **83.5** (1.04, 89%) | 76.4 -> 77.8 (1.08, 44%) |

Plain and speculative decoding gained alike (1.5-2.5%): the verification passes lost a stage a
layer (the routing) and the gates' norms as one-token passes did, but their cost is the dense
matvecs and the experts of their tokens. Prompt lookup on the chat prompt with the 4-bit file:
126.5 tok/s.

**What is left to 125 tok/s plain** (8.0 ms). A chained one-token pass is 8.86 ms (mlx-community
4bit), 9.35 ms of stage time in the profile: 7.5 ms in the bandwidth kernels with what was fused
into them (gate / up with the routing 53.8 us x 44, down with the convolution 34.0 x 44, the
attention's projections 28.4 x 40 and 29.0 x 8, the output projection with its convolution 14.6 x
48, the LM head 0.47 ms, the MXFP8 layers' experts 0.55 ms, the gates' down projections 5.7 x 48,
the router 4.9 x 48) and 1.9 ms in the others (the attention 15.9 x 48, 21.7 past 64 keys; the
mixes 7.5 x 89; `ink_gfg` 7.9 x 48). The 2.7-2.9 GB a token reads stream in 4.9-5.3 ms at 550
GB/s, so the bandwidth kernels hold about 2 ms of their stages' start, ramp and tail (3-4 us over
about 290 stages) and of the small work fused into them; this round found that the stream's start
does not move into the small stages before it (the prefetch above). The small kernels are chains
of a few dependent round trips in one or a few threadgroups (the mixes, the attention at short
ranges, the gates), and this round and the last folded them into a neighbour wherever one could
take them; what is left of them is their latency. A 4-token verification pass is 2.2 times a
one-token pass (19.7 against 9.1 ms), so drafts pay only above about 2.2 tokens a round: its
floor is the decode library's multi-token dense matvec (2.3 times the one-token time at 4 tokens,
about 9 us a stage a token) and the experts of the drafted tokens (2.2-2.4 times).

#### DGX Spark, CUDA (November 2026)

On a DGX Spark (GB10, 121 GB shared between the CPU and the GPU) against llama.cpp's CUDA backend
built there (master `94f3af3`, `llama-bench -ngl 99 -p 512,2048 -n 128 -r 3`), `parity_bench`
with `device=cuda lookup=0 draft=none mtp=0`, tok/s over two to five interleaved runs:
Llama-3.2-1B Q4_K_M 5173-5243 / 4137-4188 / 177-179 (pp512 / pp2048 / decode) against
20368-21213 / 21107-22577 / 224-249, Qwen3.8-27B UD-Q3_K_XL 258 / 231-239 / 12.4 against
994-1001 / 995 / 15.3-15.9. What the device's shared memory changes (the driver's free-memory
figure, weights held twice, copies of pages not yet mapped, memory the GPU reads in place), what
was measured and kept or not, and where the remaining time goes:
[DGX Spark](platforms.md#cuda-speed-and-memory-on-the-gb10).

#### DGX Spark CPU (November 2026)

An NVIDIA DGX Spark: 10 Cortex-X925 (3.9 GHz, 2 MiB of L2 each) and 10 Cortex-A725 (2.8 GHz,
512 KiB of L2 each), aarch64 Linux (Ubuntu 24.04, kernel 6.17), 121 GiB of unified LPDDR5x.
`--devices` reports `neon dotprod i8mm bf16 fp16 sve sve2`; there is no SME, and the SVE vectors
are 128 bits on both kinds of core. CPU only (`device=cpu`, 20 threads by default; the machine's
GPU is under [CUDA](platforms.md#cuda)). Two other sessions built and ran models on the same
machine throughout (load averages 8-21 from them; a lock kept the timed runs themselves apart),
so the figures are low for this CPU and only interleaved runs compare.
`examples/parity_bench.rs` as in the matrix (`lookup=0 draft=none mtp=0`, a warm-up pass per
size, then one run), the configurations alternated over three rounds: the median, with the
lowest and highest of the three in brackets. "Before" is master `a3284a2`; llama.cpp is master
`94f3af3` (2026-11-22) built on the machine without CUDA, `llama-bench -r 1` on 10 and on 20
threads in the same rounds. Prompt processing, tok/s:

| model | pp512 before | after | llama.cpp 10 thr | 20 thr |
|---|---|---|---|---|
| Llama-3.2-1B Q4_K_M (bartowski) | 857 [821-859] | 1370 [1356-1377] | 270 [221-277] | 342 [317-355] |
| Llama-3.2-1B Q8_0 | 866 [849-879] | 1546 [1522-1578] | 283 [278-288] | 352 [348-358] |
| Qwen3.5-9B `--quant q4` | 184 [182-184] | 302 [299-304] | | |
| Qwen3.5-9B `--quant q8` | 182 [179-185] | 289 [288-292] | | |
| Qwen3.8-27B UD-Q3_K_XL (unsloth GGUF) | 48.3 [43.9-48.3] | 76.2 [71.5-76.5] | 6.2, 6.3 | 8.4 [8.4-8.4] |
| Qwen3.8-27B UD-Q4_K_M (one round) | 47.8 | 82.8 | | 12.9 |

| model | pp2048 before | after | llama.cpp 10 thr | 20 thr |
|---|---|---|---|---|
| Llama-3.2-1B Q4_K_M | 630 [589-643] | 1029 [1027-1063] | 253 [208-259] | 323 [300-330] |
| Llama-3.2-1B Q8_0 | 646 [627-648] | 1158 [1147-1180] | 269 [265-269] | 335 [332-336] |
| Qwen3.5-9B `--quant q4` | 176 [176-177] | 279 [278-282] | | |
| Qwen3.5-9B `--quant q8` | 173 [172-175] | 274 [270-274] | | |
| Qwen3.8-27B UD-Q3_K_XL | 47.1 [46.1-47.4] | 73.0 [69.3-73.4] | 6.2 (one round) | |
| Qwen3.8-27B UD-Q4_K_M (one round) | 47.1 | 79.5 | | |

Decoding (64 tokens after a 16-token prompt for Llama 1B, 32 for the 9B, 24 for the 27B), tok/s:

| model | before | after | llama.cpp 10 thr | 20 thr |
|---|---|---|---|---|
| Llama-3.2-1B Q4_K_M | 118 [112-119] | 120 [118-122] | 103 [69-111] | 40.9 [13.0-41.9] |
| Llama-3.2-1B Q8_0 | 81.9 [79.2-82.0] | 81.0 [79.7-81.5] | 71.6 [71.3-72.8] | 32.7 [31.7-36.3] |
| Qwen3.5-9B `--quant q4` | 25.0 [24.9-25.4] | 25.2 [25.1-25.2] | | |
| Qwen3.5-9B `--quant q8` | 13.4 [13.0-13.6] | 13.4 [13.4-13.5] | | |
| Qwen3.8-27B UD-Q3_K_XL | 7.1 [6.9-7.2] | 7.2 [7.0-7.2] | 4.7, 4.5 | 4.9 [4.8-4.9] |
| Qwen3.8-27B UD-Q4_K_M (one round) | 7.3 | 7.3 | | 5.6 |

(The 27B's llama.cpp runs take minutes each at these speeds, so its 10-thread column has two
rounds at 512 tokens and one at 2048. The "after" column of Llama-3.2-1B Q4_K_M is the same
binary as the others; its three rounds ran in the group of the per-change runs below. The
"after" columns were measured before the f32 GEMM change below, which leaves this path alone:
the two builds alternated three more rounds at 1411 / 1082 against 1389 / 1073 on Llama 1B
Q4_K_M, pp512 78.0 against 76.5 on the 27B and 305 against 302 on the 9B q4.) So a prompt
runs 1.55-1.8x faster than before on every model, and decoding is unchanged, as it should be:
nothing in it was touched, and it is bound by memory (below). Against llama.cpp at its better
thread count the prompt is 3.2-4.4x faster on Llama 1B and 6.4-9.1x on the 27B files, and
decoding 1.1-1.5x faster. llama.cpp's prompt processing gains 24-35% from the second ten
threads and its decoding falls to 0.4-0.5x with them on Llama 1B (level on the 27B).

What changed. All of it applies on aarch64 Linux only: Apple's cores, Android and Windows keep
the kernels and defaults they had, since none of this could be measured on them, and x86-64 has
its own kernels.

- **Int8 prefill GEMM** (`src/kernels_i8gemm.rs`; `fast`, the default of a language model's
  prompt, and `int8`): two changes, neither of which moves a bit of the result.
  *Column blocks.* Each tile of 4 rows used to pass over every token of its pool job, so every
  4 rows read the job's int8 activations again: 4.5 MB for 256 tokens of the 27B's 17408-wide
  matrices, more than a core's L2 here. A job is now a group of 32 rows, unpacked once, and the
  columns go in blocks of 512: the group's weights of one block (16 KiB) stay in L1 while the
  job's activations of that block (128 KiB) pass once from L2, the partial f32 sums kept
  between blocks. *SMMLA.* Where the CPU has i8mm the kernel is `SMMLA` in place of `SDOT` by
  element: weight rows and tokens go two by two, 8 bytes of each per register, and one
  instruction adds 32 products into a 2 x 2 tile of int32 sums (8 rows x 4 tokens per kernel
  call: 8 instructions per 8 inputs, from 4 weight and 2 activation loads). The integer sums
  per block of 32 (per 16 for the formats with a scale per 16) and the f32 operations after
  them are the SDOT kernel's, in the same order. `examples/gemm_bench.rs`, one core, 512 x
  2048, 256 tokens, SDOT with column blocks against SMMLA: Q4_K 310 -> 394, Q6_K 272 -> 370,
  Q8_0 334 -> 458 GFLOP/s on a Cortex-X925 and 99 -> 129, 91 -> 112, 109 -> 149 on a
  Cortex-A725. On 20 threads at 256 tokens (Q4_K, two passes at load averages 17-18, both
  given), master against this: 5120 x 17408 1.50 / 1.59 -> 3.91 / 3.89 TFLOP/s, 17408 x 5120
  3.15 / 2.98 -> 4.63 / 4.62, 4096 x 4096 2.68 / 2.68 -> 3.96 / 3.96, 2048 x 8192 1.71 / 1.74
  -> 3.29 / 3.26, 2048 x 2048 2.92 / 3.01 -> 3.10 / 3.29.
- **Prompt attention as the per-head loop.** [CPU round 3](#cpu-round-3-november-2026) moved
  the Qwen-family prompt attention (which GGUF Llama files also take) to score and value GEMMs.
  As on x86-64, the loop it replaced is faster here once it has kernels: `kernels::dots` scores
  eight keys at a time per head and `kernels::weighted_values` keeps the output sums of up to
  four heads that share a value row in registers over 16 columns while the keys pass (NEON,
  the scalar loops' bits), and a prompt chunk goes through `model::attention_blocked`: the
  query heads of one KV head for 8 tokens take the keys, then the values, in chunks of about
  32 KiB, instead of every token streaming all the keys and values it sees. The loop gives a
  prompt chunk the bits of one-token attention; the GEMMs summed in another order. One round
  on Llama-3.2-1B Q4_K_M with an earlier build of the int8 GEMM, pp512 / pp2048: the GEMMs
  971 / 600, the loop with its kernels one token at a time 966 / 659, in blocks 1345 / 1048.
- **f32 prefill GEMM** (`prefill_gemm=exact`, and float-stored weights in any mode). The same
  fault: every 8 rows streamed the whole packed prompt (18 MB at 256 tokens x 17408 inputs)
  from memory, so 20 threads ran at what the memory delivers. It is cache-blocked by columns
  now as the AVX-512 one is (`gemm_blocked` in `kernels.rs`, 8 rows x 8 tokens per step): a
  pool job of 32 rows x 256 tokens dequantizes 8 rows of a 256-column block (8 KiB), multiplies
  them with the job's token panels of that block (256 KiB, in L2) and keeps its partial sums
  between blocks; bf16 weights take the same path. Each output is still one fused multiply-add
  chain over the columns in order, so the results are bit-identical (unit test against the 8x8
  GEMM over whole rows, and the logits below). `gemm_bench`, 20 threads, 256 tokens, two
  passes at load averages 17-18, master against this: Q4_K 5120 x 17408 0.67 / 0.68 -> 1.54 /
  1.54 TFLOP/s, 17408 x 5120 0.93 / 0.94 -> 1.62 / 1.64, 4096 x 4096 0.92 / 0.91 -> 1.62 /
  1.62, 2048 x 8192 0.85 / 0.85 -> 1.15 / 1.15, 2048 x 2048 1.26 / 1.28 -> 1.21 / 1.21 (the
  one shape whose activations already fit a Cortex-X925's L2); BF16 0.65 / 0.65 -> 1.56 /
  1.55, 0.91 / 0.99 -> 1.63 / 1.62, 0.96 / 0.98 -> 1.27 / 1.28, 0.78 / 0.77 -> 1.16 / 1.16,
  1.10 / 1.14 -> 1.17 / 1.18.
- **`--devices`** prints the kernels a prompt takes here (`kernels: prompt matmuls: int8 GEMM on
  SMMLA (i8mm), column-blocked; prompt attention: per-head NEON loop`).

Each change of the default path alone, in the same binary with one switch
(`OXINFER_I8_MMLA=0 OXINFER_I8_KB=0 OXINFER_I8_JOB_ROWS=0`: the previous int8 GEMM;
`OXINFER_I8_MMLA=0`: the blocked SDOT GEMM; `OXINFER_ATTN_NEON=0`: the GEMM attention;
`threads=10`), in the rounds of the tables above, tok/s (medians of three, the 27B's of two):

| model | all on | previous int8 GEMM | blocked SDOT GEMM | GEMM attention | 10 threads |
|---|---|---|---|---|---|
| Llama-3.2-1B Q4_K_M pp512 / pp2048 | 1378 / 1068 | 1068 / 863 | 1200 / 973 | 1110 / 785 | 1270 / 946 |
| Llama-3.2-1B Q8_0 pp512 / pp2048 | 1546 / 1158 | 1068 / 871 | 1332 / 1038 | 1178 / 804 | 1202 / 1026 |
| Qwen3.5-9B `--quant q4` pp512 | 302 | 186 | | 298 | 253 |
| Qwen3.5-9B `--quant q8` pp512 | 289 | 185 | | 289 | 231 |
| Qwen3.8-27B UD-Q3_K_XL pp512 | 76.2 | 49.2 | | 76.0 | 65.0 |

(The Q4_K_M row was measured with the pool variant of "not kept" below switched on, in all its
columns.) The 9B has attention in a quarter of its layers and the 27B's prompt is nearly all
matmuls, so the attention change is Llama's. Decoding on 10 threads against 20: 117 / 120
(Llama 1B Q4_K_M), 80.7 / 81.0 (Q8_0), 23.3 / 25.2 and 12.9 / 13.4 (the 9B), 6.0 / 7.2 (the
27B): every core is worth having, and 20 stays the default.

`prefill_gemm=exact`, which the f32 GEMM serves, tok/s (three rounds; master, then this):

| model | pp512 before | after | pp2048 before | after |
|---|---|---|---|---|
| Llama-3.2-1B Q4_K_M | 488 [484-490] | 661 [659-665] | 405 [403-409] | 579 [579-583] |
| Qwen3.5-9B `--quant q4` | 66.5 [66.3-68.3] | 106 [105-106] | | |
| Qwen3.8-27B UD-Q3_K_XL | 17.3 [16.3-17.6] | 30.8 [28.2-31.1] | | |

(Llama 1B with the per-head attention and the whole-row GEMM, `OXINFER_GEMM_KB=0`: 536 / 483,
so the blocks are 1.2x there and the attention 1.1-1.2x.) Float-stored weights take the f32
GEMM in every mode, so their default prompt gains the same: Qwen3.5-9B without `--quant`
(bf16), two rounds, pp512 67.8 / 67.3 -> 98.2 / 97.0 tok/s, decoding 8.1 either way.

Results. `examples/qwen_greedy_dump.rs`, a 2,600-character prompt (709 tokens for Llama, 736
for the Qwen models) fed as one batch and 16 greedy tokens, the 17 logits rows compared byte
for byte against master:

- `prefill_gemm=exact` with the GEMM attention (`OXINFER_ATTN_NEON=0`): identical on
  Llama-3.2-1B Q4_K_M, Qwen3.5-9B q4 and bf16 and the 27B UD-Q3_K_XL (and Llama-3.2-1B Q8_0 on
  the build before the f32 GEMM change), and with `OXINFER_GEMM_KB=0` beside the blocks. So
  the blocked f32 GEMM changes nothing, and neither does the thread count or the affinity
  (the same bytes from 10 threads pinned to the Cortex-X925s).
- `prefill_gemm=int8` with the GEMM attention: identical on the same models, with the SMMLA GEMM
  and with the blocked SDOT one.
- The per-head loop in blocks against the same loop one token at a time: identical.
- The per-head loop against the GEMM attention (what a default run changes), exact GEMMs: the
  last prompt position's distribution differs by a KL of 4.6e-12 (Llama 1B Q4_K_M), 1.1e-12
  (Q8_0), 4.3e-8 (the 9B q4) and 1.6e-11 (the 27B), the same figures as on x86-64; the 16 greedy
  tokens are the same on Llama 1B Q4_K_M and the 9B, and part at a near tie on Q8_0 (the
  sixth token, 0.138 against 0.136) and on the 27B (the fifth, 0.128 against 0.127).
- With the int8 GEMM the same change of summation order is amplified by the rounding of the
  activations at every matmul, as [x86-64 CPU prefill](#x86-64-cpu-prefill-november-2026)
  describes: KL 4.6e-3, 1.1e-3, 3.5e-3 and 9.5e-4 against master's int8 prefill, where the
  int8 prefill is 3.7e-3, 1.5e-4, 3.1e-3 and 2.4e-3 from the exact one of the same build; the
  top token of the last prompt position is the same in all of them.

Where the time goes (`--profile`, one run of master and one of this under the lock, 20
threads, load averages 8-16):

- Llama-3.2-1B Q4_K_M, 2503-token prompt: 1.8 -> 1.1 ms per token (542 -> 928 tok/s), the
  matmuls 1.5 -> 0.6 and attention 0.9 -> 0.4 (the two timers overlap in master's GEMM
  attention). Then 24 tokens at that context: 13.4 -> 12.2 ms each, the matmuls 8.7 -> 8.2
  (0.80 GB per token at 92-97 GB/s), attention 4.1 -> 3.5, the LM head 1.9.
- Qwen3.5-9B q4, 736 tokens: 5.5 -> 3.4 ms per token, the matmuls 4.9 -> 2.7, attention and
  the Gated DeltaNet 0.5. Decoding 40-42 ms per token, the matmuls 36-37 of it: 4.46 GB per
  token at 120-123 GB/s.
- The 27B UD-Q3_K_XL, 736 tokens: 21.1 -> 14.7 ms per token, the matmuls 19.4 -> 12.8,
  attention and the Gated DeltaNet 1.5 -> 1.3. Decoding 143-150 ms per token, the matmuls
  133-140 of it (12.2 GB per token at 87-92 GB/s), the LM head 7, attention and the states 7.

So a prompt is the int8 GEMM (87% of the 27B's time, 79% of the 9B's, about half of Llama
1B's), which now runs at 3.3-4.6 TFLOP/s on 20 threads against 10 x 0.39-0.46 + 10 x 0.13-0.15
for its kernel on one core of each kind; and decoding is the reading of the weights.

What the machine gives:

- `membench`: 33.5 GB/s of plain reads on one thread, 63 on 2, 99-107 on 4, 128 on 8 and
  126-131 on 12 to 18; the bf16 matvec 127-134 GB/s on 12 to 18 threads. `matvec_bench` (a 2 GB
  ring of Q4_K matrices, so every call reads from memory): 5120 x 17408 at 65 / 107 / 102 /
  108 / 116 / 113 GB/s on 4 / 8 / 10 / 12 / 16 / 20 threads against 123-131 of plain
  streaming. The 9B q4 decodes at that wall (4.46 GB per token at 120-123 GB/s in the
  matmuls); the 27B UD-Q3_K_XL at 87-92 GB/s is short of it because its formats (Q3_K and the
  i-quants) cost more per byte: `kernel_bench`, 5120 x 17408, one token, ns per 256 weights on 20 threads / one
  Cortex-X925 / one Cortex-A725: Q4_K 1.01 / 5.1 / 13.4, IQ4_XS 1.01 / 4.8 / 13.8, Q3_K 1.02 /
  8.2 / 26.6, IQ3_XXS 1.04 / 12.6 / 31.4 (Q3_K's 110 bytes in 1.02 ns are 108 GB/s, IQ3_XXS's
  98 bytes 94 GB/s).
- The multiply-add instructions alone, operands in L1, one core (a C probe, GCC 13 `-O3`,
  eight accumulators), instructions per ns and products per ns on a Cortex-X925 / a
  Cortex-A725: `SDOT` with both operands loaded 7.3 and 116 / 3.4 and 55; `SDOT` by element,
  one load per four 14.6 and 233 / 5.5 and 88; `SMMLA` with both operands loaded 7.5 and 240 /
  2.9 and 94; SVE `SDOT` 7.0 and 112 / 2.9 and 46; `FMLA` 7.3 and 29 / 2.8 and 11; `BFDOT` 7.3
  and 58 / 3.3 and 27; `BFMMLA` 7.3 and 116 / 2.8 and 45. So `SMMLA` issues at `SDOT`'s rate
  here for twice the products (on Apple's cores at half the rate,
  [CPU round 3](#cpu-round-3-november-2026)), SVE at 128 bits gives what NEON gives, and the
  SMMLA GEMM's 458 GFLOP/s on Q8_0 is 229 G products/s, at the instruction's rate.
- Cores: a Cortex-A725 runs the int8 GEMM at a third of a Cortex-X925's speed (129 against
  394 GFLOP/s on Q4_K) and the f32 GEMM at 0.28x (41 against 146), where the kernel's
  `cpu_capacity` says 0.70.

Measured and not kept:

- **Keeping the slow cores off the end of a parallel region.** The pool hands out blocks one
  at a time, so a slower core takes fewer; what remains is a block taken by a slow core near
  the end, which the others wait for. A variant in which workers that find themselves on a
  Cortex-A725 (`sched_getcpu` against `cpu_capacity`) leave the last 30 blocks of a region to
  the others: pp512 / pp2048 1378 / 1068 against 1370 / 1029 (Llama 1B Q4_K_M), 1581 / 1202
  against 1546 / 1158 (Q8_0), pp512 299 against 302 and 296 against 289 (the 9B), 78.8 against
  76.2 (the 27B); decoding 119 / 120, 80.1 / 81.0, 23.9 / 25.2, 13.3 / 13.4, 7.2 / 7.2. From
  -1% to +4% on prompts and 0 to -5% on decoding, inside the spread of a shared machine: the
  pool is as it was.
- **10 threads**, one per Cortex-X925: the table above (0.78-0.92x on prompts, 0.83-1.0x
  decoding); pinned to those cores (`taskset`) master ran Llama 1B Q4_K_M at 612 / 542 tok/s
  and 108 decoding against 735 / 593 and 114 on 20 threads.
- **Transparent huge pages** for the weights (the system has them on `madvise`):
  `GLIBC_TUNABLES=glibc.malloc.hugetlb=1`, which makes malloc ask for them, changed neither
  the plain reads (128-131 GB/s on 12 threads and 34 on one, either way) nor the bf16 matvec
  (127 -> 134 on 12 threads, 133 -> 132 on 18): decoding is at the memory's rate, not the
  TLB's. Nothing was added to the loader.
- **SVE / SVE2 kernels**: 128-bit vectors that issue no faster than NEON's (the probe above).
- **bf16 instructions**: `BFMMLA` multiplies four times `FMLA`'s products per instruction at
  the same rate, but on operands rounded to bf16 (8 bits of mantissa), which is neither the
  exact GEMM nor the documented int8 rounding; no kernel was built on it.
- **Two tiles per call in the SDOT kernel** (8 chains instead of 4): +10% on a Cortex-X925 and
  +4% on a Cortex-A725 for Q4_K and Q8_0, 0.66x / 0.76x for Q6_K, whose sums per 16 inputs no
  longer fit the registers. Kept for the formats without them, where the CPU has no i8mm.
- **Other SMMLA tiles** (row pairs x token pairs per call, one Cortex-X925, Q4_K / Q6_K /
  Q8_0 GFLOP/s): 4 x 2, the one kept, 394 / 370 / 458; 4 x 3 399 / 328 / 433; 4 x 4 398 / 195
  / 428; 2 x 2 342 / 322 / 402; 2 x 3 366 / 360 / 434; 2 x 4 369 / 338 / 440.
- **Other blocks for the int8 GEMM** (20 threads, 256 tokens, Q4_K, 5120 x 17408, second of
  two passes): 32-row jobs with 512 columns 3.95 TFLOP/s, with 1024 columns 3.57, with 2048
  3.46; 16-row jobs 2.60; 64-row jobs 3.57-3.73. The 2048-wide shapes were within the noise
  of each other.

Where oxinfer is still behind or bound on this machine:

- **Decoding** is where a model's weights leave memory at 113-130 GB/s, and the low-bit
  formats do not reach that: the 27B UD-Q3_K_XL streams 87-92 GB/s, where the Q3_K and
  IQ3_XXS decode kernels take 1.6x and 2.5x Q4_K's time per weight on a Cortex-X925. Their
  unpackers are the place to look for that file; the Q4 and Q8 models are at the memory's
  rate.
- **The exact GEMM** stays 2.1-2.9x slower than the int8 one (pp512 661 against 1370 on Llama
  1B, 106 against 302 on the 9B, 30.8 against 76.2 on the 27B): an `FMLA` multiplies 4
  products where an `SMMLA` multiplies 32, and there is no SME here to make up for it.
- **Prompts** are at the SMMLA instruction's rate on the Cortex-X925s, so what is left is
  outside the GEMM: on Llama 1B the attention (0.4 of 1.1 ms per token at 2.5k tokens) and
  the dequantization of activations and rows around the kernel; and a Cortex-A725 adds a
  third of a big core where its two vector pipelines allow no more.
- Not measured here: mixture-of-experts files (their grouped GEMM takes the same int8 kernel),
  Gemma (whose prompt attention was `attention_prefill` already and still is), and anything
  with the machine to itself.

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
  decode pass on the GPU too ([Decoding on the GPU](#decoding-on-the-gpu)): at the time on par
  with llama.cpp on the 1B models, ahead on Qwen3.5-0.8B (1.27x), Zawen (1.2-1.3x) and Gemma 4
  26B-A4B (1.13x), behind on Orpheus 3B and the 27B (~0.9x). Since the
  [Metal decode round](#metal-decode-round-november-2026) it is ahead on every model measured
  there (1.04-1.21x; Q5_K matvecs at the other formats' 550-590 GB/s; the 27B not re-measured).
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
