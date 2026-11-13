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

## Speculative decoding

`--draft models/gemma-4-E4B-it-assistant` enables Google's multi-token-prediction drafter: a
4-layer, 256-wide decoder that cross-attends to the main model's shared KV caches and predicts
from the concatenation of the last token's embedding and the previous hidden state. Each round
drafts 2–6 tokens (adaptive) and verifies them in one multi-token forward pass; greedy
verification makes the output identical to plain greedy decoding. Acceptance is about 40% on prose
and 55–65% on code, giving 2–4 tokens per forward. The drafter's own drafts were checked against
the transformers implementation token for token. It pays off most where the verification pass
is bandwidth-bound (bf16, Q8); at Q4 the multi-token int8 kernels are instruction-bound, so it
only helps on predictable text.

`--mtp` does the same for Qwen3.5 / Qwen3.8 with the checkpoint's own multi-token-prediction
layer (`mtp.*`, `qwen_mtp.rs`): one gated-attention decoder layer that takes the main model's
final-normed hidden state at position i and the embedding of token i+1 (each RMS-normed,
concatenated embedding first, projected by `mtp.fc`), runs at position i with its own KV cache,
and predicts token i+2 through the shared LM head; further drafts feed it its own output (the
vLLM `qwen3_5_mtp` scheme). The default is 3 drafts per round, adapting between 1 and 3. With GPU
decoding the verification is one GPU pass (the same multi-token kernels as DSpark's, so greedy output
is plain GPU decoding's) and the draft head runs on the GPU; MTP is on by default for Qwen3.5-4B and
Qwen3.5-9B, where it measured faster on every kind of prompt (numbers and the comparison with DSpark
in [DSpark drafts for Qwen and Gemma 4](qwen.md#dspark-drafts-for-qwen-and-gemma-4); `--no-draft` turns it
off). Both
greedy and sampled decoding are supported: greedy output is identical to plain greedy (the
decode kernels give bit-identical rows for up to 7 tokens per pass, so the verification logits
equal one-token logits; checked on long outputs with and without `--think`, bf16 / Q8 / Q4,
dense and MoE), and sampling uses the standard speculative-sampling acceptance rule, so every
token is drawn from exactly the distribution `--temp` / `--top-k` / `--top-p` define (the
random stream differs from plain sampling, so the same seed gives a different text). Rejected
drafts are undone exactly: the verification forward keeps the Gated DeltaNet states from
before the pass plus each token's delta-rule inputs, and a rewind replays the rule up to the
last accepted token (the same arithmetic, so bit-identical; one state copy instead of one per
token, which is 151 MB per drafted token on Qwen3.8-27B); KV caches are truncated. On Qwen3.8-27B (M5 Max, under load from other
jobs):

| Precision | Plain | `--mtp` (code) | `--mtp` (prose) | Acceptance (code / prose) |
|---|---|---|---|---|
| `--quant q8` | 7.5 tok/s | 15.2 tok/s (3.75 tok/forward) | 10.5 tok/s (2.55 tok/forward) | 94% / 59% |
| `--quant q4` | 14.6 tok/s | 24.0 tok/s | 14.7-16.1 tok/s | 94% / 57% |

Q4 gains less because a 4-token Q4 pass costs about twice a one-token pass (SDOT-bound). (The
multi-token CPU passes have since become cheaper: see
[Multi-token passes and MoE experts on the CPU](#multi-token-passes-and-moe-experts-on-the-cpu).) On
Qwen3.5-0.8B every draft step reads the 248k-row LM head (a third of the model), so `--mtp`
gains ~20-25% on code and loses a little on prose; a trimmed draft head (`DraftHead`: a
`[V_draft, H]` matrix plus a draft-to-target id map, FastMTP style) can replace it through the
library. The MoE models (Qwen3.6-35B-A3B) work the same way, with an MoE MTP layer. A converted
checkpoint (`--save-quantized`) keeps the MTP weights; older converted copies can take them from
the original checkpoint with `--mtp-from DIR`.

From the library: `QwenModel::load_mtp(&files)` (or `QwenMtp::load(cfg, sidecar_files, quant)`),
then `MtpDecoder::new(&model, &mtp, MtpOptions { .. })`: `prefill` the prompt, `push` the first
token, call `step` until a stop token (`examples/qwen_mtp.rs`).

FastMTP sidecars (a GGUF holding only an MTP layer, a trimmed output head and a `d2t` map, such as
HauhauCS's `...-FastMTP-32K.gguf`) load with `--mtp-from FILE` (or `org/repo:FastMTP-32K`): the
sidecar's layer, token embedding and 32k-row head replace the target's, as in HauhauCS's
llama.cpp patch (`QwenMtp::attach_sidecar` in the library). With HauhauCS's Qwen3.8-27B Q4_K_P
GGUF, greedy output with the embedded MTP layer or the FastMTP sidecar is token-for-token the
plain greedy output.

## Multi-token passes and MoE experts on the CPU

Speculative verification (and prompts shorter than 8 tokens) runs on the CPU's decode kernels,
whose rows are bit-identical whatever the token count, so verification logits equal one-token
logits. Two lossless changes (after the analysis of llama-modchip in
[docs/llama-modchip](llama-modchip/README.md)) made those passes cheaper:

- **Column groups for the blocks-of-32 formats.** GGUF Q8_0, Q4_0, IQ4_NL, MXFP4, Q4_1, Q5_0,
  Q5_1 and oxinfer's Q4 used to take tokens in pairs (each block unpacked once per pair, plus a
  one-token pass for an odd count). They now run the one-token kernel instantiated for up to 8
  tokens on column groups (9 = 5 + 4, 16 = 8 + 8; narrower when the row is long enough that a
  group's int8 activations would not stay in L1), with rows in 64 KB tiles. Each token's
  arithmetic is the one-token kernel's, so rows stay bit-identical. The K-quants and i-quants
  already had such kernels; oxinfer's int8 format gained nothing (its blocks need no unpacking)
  and keeps the pairs. Single thread, 4096 x 4096, ns per 256 weights per token, token pairs ->
  column groups (two interleaved runs, a loaded machine):

  | Format | 3 tokens | 4 | 5 | 8 | 15 |
  |---|---|---|---|---|---|
  | Q8_0 | 8.6 -> 6.6 | 7.2 -> 5.6 | 8.0 -> 5.0 | 7.7 -> 5.6 | 8.2 -> 6.0 |
  | Q4_0 | 7.6 -> 6.1 | 6.4 -> 5.5 | 7.1 -> 6.3 | 6.4 -> 6.0 | 7.2 -> 7.0 |
  | MXFP4 | 9.1 -> 6.8 | 7.4 -> 5.9 | 9.1 -> 5.5 | 7.8 -> 6.0 | 8.6 -> 6.0 |
  | Q5_0 | 13.7 -> 9.1 | 11.5 -> 8.1 | 12.0 -> 7.5 | 12.3 -> 6.7 | 12.8 -> 7.3 |
  | Q4 (`--quant q4`) | 7.1 -> 6.1 | 6.4 -> 5.8 | 6.3 -> 5.7 | 6.6 -> 6.1 | 6.4 -> 6.7 |

  A whole verification pass of Qwen3-8B Q8_0 (18 threads, recording forward plus LM head,
  `examples/pass_bench.rs`, best of 7 interleaved): 3 tokens 49 -> 41 ms, 4 tokens 49 -> 41 ms,
  8 tokens 83 -> 65 ms (one token: 34 ms). With its DSpark draft on the CPU the decoding gained
  12-23% on prose (40 -> 44-49 tok/s) and was within noise on code, where the cost-aware drafting
  now drafts longer blocks; greedy output identical. On other CPUs the portable GGUF kernel takes
  the same column groups (x86-64 under Rosetta, which runs the portable path: Q8_0 / Q4_K at
  8 tokens 2.2-2.3x faster per token).
- **MoE experts in one batch per stage.** Decode and verification passes of the MoE models
  (Qwen3.5 / 3.6 MoE, Gemma 4 26B-A4B, Mistral Large 4, DeepSeek V4) quantize the input once,
  build the expert -> tokens grouping once for gate, up and down, and run the gate/up matmuls of
  every chosen expert, and of the shared expert, as one parallel region, then all down matmuls as
  another (they ran one region per matmul: Qwen3.5-35B-A3B decode went from 521 to 161 regions
  per token). This applies only when every matmul would run on the CPU decode kernels anyway (no
  accelerator, LoRA or float weights), so no matmul run elsewhere ever reads the shared quantized
  input. Qwen3.5-35B-A3B Q4, passes of 1 / 2 / 4 / 7 tokens, per-expert regions -> batched (12
  threads, best of 9 interleaved): 35 -> 29, 38 -> 32, 52 -> 45, 81 -> 66 ms. End to end (CPU
  decode, 128 tokens, alternating binaries on a loaded machine) Gemma 4 26B-A4B Q4_0 went from
  55.3-55.8 to 56.2-59.8 tok/s; Qwen3.5-35B-A3B's runs (60-69 tok/s) were within the noise.

`examples/kernel_bench.rs` (formats, threads, shape, token counts) and `examples/pass_bench.rs`
(`pairs` / `auto` / `seq` / `old` modes; `PASS_COMPARE=1` asserts that all modes give bit-identical
logits) measure these. Prefill (8 tokens and more) keeps its f32 GEMM: switching it to the int8
kernels would change its results.

## Prompt lookup

`lookup.rs` drafts without a draft model: when the last tokens of the conversation (prompt and
reply so far) occurred before, the tokens that followed that occurrence are the draft, verified
like any other in one multi-token pass. Replies that quote, edit or summarize their prompt
(code edits, refactors, extraction, rewrites, repeated structure) accept long runs of such drafts;
on free prose there is rarely a match and the round is a plain one-token step.

- Matching: an index from every bigram of the context to its positions; the occurrences of the
  last two tokens (the 48 most recent) are extended backwards to their full match length, the
  longest match wins (the most recent among equals), and the draft is what followed it (up to
  `--lookup-tokens`, default 10; when the verification runs on the CPU, at most 6 unless
  `--exact-verify` raises the process-wide limit of the batch-invariant decode kernels, as DSpark
  and MTP do). Other occurrences that agree on the next token count as consensus.
- Verification-outcome memory (the one idea taken from llama-modchip's drafter, see
  `docs/llama-modchip`): a 4096-entry direct-mapped table keyed by a hash of the four tokens before
  a position remembers the draft token the model last rejected there and the token it chose
  instead; a draft token refuted in the same context, and not accepted there since, ends the draft
  (`--no-lookup-memory` turns it off).
- Cost-aware length: each draft token's chance of being accepted (when those before it were) is
  estimated per match class (match length, consensus, depth), calibrated online from the
  verification outcomes; with the measured time of verifying each number of drafts (a plain step
  for none), the round verifies the length with the most expected tokens per second, which is 0
  (a plain step) when nothing is worth it. Until a verification has been measured it counts as
  two plain steps (the first multi-token pass also prepares its GPU plan), so only a good match
  starts one; a match that is not worth it is still tried now and then (after 8 rounds, doubling
  to 128) so that a poor early estimate does not stick. In DSpark and MTP decoding every
  round's verification time feeds the same measurements.
- Exact like the other drafts: greedy output is the plain greedy output (bit for bit with
  `--exact-verify`; with fast verification the logits may differ in rounding, as for DSpark and
  MTP), and sampling uses speculative sampling with a one-token draft distribution (accept with
  the model's probability of the draft, else draw from the rest), which keeps the model's
  distribution; a repetition penalty sees the accepted tokens as one-token decoding would.
- With DSpark or MTP, a long lookup match (8 tokens or more: on shorter ones the draft models
  did better) replaces the draft model's round when its expected tokens per second (calibrated
  chances over the measured verification time) beat the recent rate of that decoder's rounds (or
  of a plain step when its policy chose one).

On by default (`--no-lookup`, engine / C API / server key `lookup=0`), for every family the
engine decodes plainly (Gemma 4 / 3, Qwen, Llama 3, Mistral Large 4, DeepSeek V4) and inside
DSpark and MTP decoding; not with the Gemma 4 assistant drafter or Core ML models. From the
library: `lookup::Lookup` (the drafter) and `lookup::LookupRounds` (a `RoundDecoder` over any
`dspark::Target`), or the `lookup` field of `DsparkDecoder` / `MtpDecoder`.

Measured on the M5 Max (Metal, GPU decoding; 300 tokens, greedy, `--no-system`, default fast
verification; tok/s means of 2 interleaved runs per prompt; the machine was shared with other
jobs, load 8-27, so single cells are noisy). Prompts: `code` (rename a function in a 60-line
Python module and print it back), `summary` (quote the sentences of a text that mention a date),
`json` (25 generated records), `prose` (a short story), `chat` (three tips). Tokens per forward
and drafts accepted in parentheses:

| model | plain | prompt lookup | (no memory) | code | summary | json | prose | chat |
|---|---|---|---|---|---|---|---|---|
| Qwen3-8B (Q8_0 GGUF) | 57.8 | **98.1** | 98.3 | 59 -> 138 (3.0, 74%) | 59 -> 160 (3.9, 74%) | 56 -> 78 (1.5, 38%) | 56.5 -> 57.0 | 58.9 -> 57.4 |
| gemma-4-12B-it (Q8_0 GGUF) | 32.8 | **58.7** | 58.5 | 32 -> 75 (3.1, 79%) | 31 -> 97 (4.2, 75%) | 33 -> 51 (1.9, 57%) | 33.4 -> 34.2 | 34.1 -> 36.5 |
| Qwen3.5-4B (bf16) | 57.3 | **108.9** | 109.9 | 57 -> 155 (3.4, 70%) | 59 -> 194 (4.5, 72%) | 57 -> 80 (1.6, 37%) | 56.6 -> 57.0 | 56.5 -> 59.1 |
| Qwen3.5-35B-A3B (q4, MoE) | 139.2 | **153.6** | 145.8 | 139 -> 169 (1.8, 92%) | 147 -> 183 (2.2, 92%) | 133 -> 137 | 134 -> 139 | 144 -> 141 |

Copying prompts gain 2-3x on the dense models; prose and chat stay at plain speed (a few drafts
per reply). On the MoE model a verified token costs its own routed experts (a 9-token pass is
4-6 plain steps), so the policy verifies only the likeliest few drafts (1.8-2.2 tokens per
forward at 92% acceptance) and the gain is 15-25% where the reply copies. The verification-outcome
memory vetoed nothing in any of these runs: with the most recent of the longest matches winning,
the target's own correction is already in the context the next time the same suffix comes up, so
a refuted continuation is rarely proposed again (llama-modchip's drafter also predicts from
n-gram statistics, which repeat). It costs a hash per draft token and stays on.

With the default draft models (a second run, load 20-27; DSpark for Qwen3-8B and Gemma 4 12B,
MTP for Qwen3.5-4B), lookup takes over only where a long match beats the draft model's recent
rounds:

| model | draft | draft + lookup (default) | code | summary | json | prose | chat |
|---|---|---|---|---|---|---|---|
| Qwen3-8B, DSpark | 116.9 | 115.8 | 129 -> 134 | 160 -> 149 | 127 -> 129 | 51.7 -> 51.6 | 117 -> 115 |
| gemma-4-12B-it, DSpark | 76.0 | **81.0** | 94 -> 99 | 106 -> 119 | 91 -> 94 | 37.9 -> 41.5 | 51 -> 53 |
| Qwen3.5-4B, MTP | 78.3 | **102.9** | 86 -> 138 | 102 -> 174 | 84.5 -> 85.8 | 49 -> 50 | 70 -> 67 |

(DSpark already accepts 80-97% of its 7-token blocks on the copying prompts, so lookup adds little
there; MTP drafts 3 tokens, and the lookup's longer runs lift it most. Without the long-match rule,
short prose matches displaced better DSpark rounds on Gemma 4 12B: prose 43.5 -> 35.7.)

Exactness: with `--exact-verify`, greedy output with lookup alone and with the default draft plus
lookup is the plain greedy output token for token on Qwen3-8B, gemma-4-12B-it, Qwen3.5-4B (Gated
DeltaNet rewinds) and Qwen3.5-35B-A3B (4 prompts x 300 tokens each); with the default fast
verification, all 130 outputs of these runs (lookup, without memory, DSpark, MTP, combined) are
the plain greedy outputs too.

## Streaming weights from flash (models larger than RAM)

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
[Multi-token passes and MoE experts on the CPU](#multi-token-passes-and-moe-experts-on-the-cpu));
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

### GPU decoding with streamed weights (mixed GPU / CPU)

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
