# Platforms and accelerators

The CPU path is the reference and works everywhere; everything else is detected at run time and
loaded with [`dlsym`](https://crates.io/crates/dlsym), so one binary runs with or without the
libraries present and nothing GPU-related is needed at build time. `--devices` prints what was found:

```
CPU: aarch64 on macos, 18 logical cores
  features: neon dotprod i8mm bf16 fp16
  memory: 137.4 GB
GPU (metal #0): Apple M5 Max, memory 115.4 GB (unified with CPU)
```

| Platform | Compute | Status |
|---|---|---|
| Apple Silicon (macOS, iOS) | NEON kernels (CPU prefill GEMMs on the SME tiles on M4 and later, `prefill_gemm`) + Metal GEMM for prefill; on M5 / A19 the GPU matrix units (prefill GEMMs and MoE experts); single-token decode entirely on the GPU (Llama 3, Qwen3 / 3.5 / 3.8 incl. MoE, Gemma 3 / 4 incl. 26B-A4B) | measured; the f32 GEMM gives bit-identical logits to the CPU path, the matrix units (default where present) ~3e-4 relative per product (`OXINFER_METAL_MMA=0` for bit-identical); iOS: runs in the Simulator (CPU and Metal; `ios/`), Inkling-2 Mini there greedy-identical to the Mac on CPU, Metal and streamed ([iOS / iPadOS](inkling.md#ios--ipados)) |
| Apple Neural Engine (macOS 15 / iOS 18) | Core ML, for ANEMLL-converted models | measured on macOS; builds for iOS; see [Core ML](coreml.md#core-ml--apple-neural-engine) |
| x86-64 (Linux, Windows, macOS) | AVX2 + FMA + F16C kernels chosen at run time: bf16 / f32 dot tiles and prefill GEMM, Q8 via `maddubs`, Q4, every GGUF format, decode attention; scalar fallback below AVX2 or with `OXINFER_SIMD=scalar` (`--devices` prints the set in use). With AVX-512F the prefill GEMM takes 16 rows x 16 tokens per step and is cache-blocked by columns, bit-identical to the AVX2 GEMM (`OXINFER_SIMD=avx2` keeps the latter; see [x86-64 CPU kernels](performance.md#x86-64-cpu-kernels-november-2026)); the grid-table i-quants unpack with AVX2. With AVX-512 VNNI a language model's prompt takes the int8 prefill GEMM (`vpdpbusd`; `prefill_gemm=fast`, the default there, with the decode kernels' activation rounding; `prefill_gemm=exact` keeps the f32 GEMM; see [x86-64 CPU prefill](performance.md#x86-64-cpu-prefill-november-2026)), and without it `fast` is the f32 GEMM; a Qwen-family prompt's attention is the per-head loop with AVX2 / AVX-512 kernels (the bits of one-token attention; faster here than the score and value GEMMs other architectures take), and the Gated DeltaNet rule and element-wise passes use AVX2 / AVX-512 with the scalar loops' bits; Q4's multi-token column kernel and LoRA's dot products are NEON-only (x86 runs the token-pair and portable loops) | executed on Linux on one Zen 5 CPU (Threadripper 9970X); Windows and macOS compile only. Every AVX2 kernel is tested directly against its scalar fallback (bit for bit for the integer parts, GGUF rows and attention; 1e-6 of the summed magnitudes for the f32 sums). Qwen3.5-0.8B on 5 prompts, AVX2 against scalar: last-prompt-token logits within 4.8e-5 (bf16, Q8, Q4), same top 10; 64 greedy tokens identical in 14/15 runs, and bf16 within 5e-5 of transformers float32 with the same argmax (5/5). Q8 decode steps differ between the two by 0.4-0.6 in the largest logit (same top 1 on 317/320 teacher-forced steps), as far as either is from bf16: the integer block sums are identical (a build with the AVX2 integer path and the scalar order of f32 additions gives the scalar logits bit for bit over 169 steps), the f32 additions are ordered differently, and re-quantizing activations to int8 at every matmul amplifies that. Llama-3.2-1B GGUF (Q4_K_M, Q8_0): decode-kernel passes (`--prefill-chunk 1`) give bit-identical logits and greedy text to the portable path (10/10); with the f32 prefill GEMM the last hidden state differs by rounding (up to 1.8e-4 on values up to 17) and the logits by either ~2e-5 or 1.6e-3 to 5.7e-3: the LM head of the last token is a one-token pass on int8 activations, and the larger figure appears exactly when that rounding moves one to three of the 2048 int8 activations by one step (7/16 pseudo-random prompts on Q4_K_M, 5/16 on Q8_0; an f32 LM head pass on the same states stays within 1e-4 on all 32); on the 5 chat prompts 4.8e-3 / 3.5e-5 at most, 48 greedy tokens identical in 3/5 and 5/5 |
| Linux / Windows + NVIDIA | CUDA driver API + NVRTC-compiled kernels (`cuda.rs`): prefill GEMMs (bf16, Q8, Q4, affine Q4, the GGUF block formats), MoE experts (resident or streamed), convolutions, group / layer norms, attention; whole decode steps, single-token and multi-token (speculative verification), over the same formats; whole-pass prefill plans for GGUF weights ([below](#cuda)); Inkling-2 / Mini whole prefill and decode passes on kernels of its own (every GGUF format but the ternary / 1-bit ones, NVFP4, MXFP8); loaded as a native binary, so a toolkit newer than the driver works | measured on Linux on one RTX 5080 (NVRTC 13.4, driver CUDA 13.2) and one RTX PRO 6000 Blackwell (Inkling-2), see [CUDA](#cuda), and on aarch64 on one DGX Spark (GB10, compute capability 12.1, NVRTC 13.0, unified memory), see [DGX Spark](#dgx-spark-aarch64-linux); Windows and other GPUs not run |
| aarch64 Linux | NEON kernels (dotprod, i8mm); a language model's prompt on the int8 prefill GEMM (`prefill_gemm=fast`, the default), `exact` on the NEON f32 GEMM where the cores have no SME; with an NVIDIA GPU, CUDA and Vulkan as on x86-64 | measured on one DGX Spark (10 Cortex-X925 + 10 Cortex-A725, GB10): the test suite, Qwen3.5-0.8B against transformers float32 and the three devices against each other, see [DGX Spark](#dgx-spark-aarch64-linux); no other aarch64 Linux machine run |
| Android (aarch64) | NEON kernels | cross-compiles (`aarch64-linux-android`) |
| Vulkan (any GPU with a Vulkan 1.0 driver) | compute shaders through the Vulkan loader (`vulkan.rs`, SPIR-V embedded in the binary): prefill GEMMs only (bf16, Q8, Q4, affine Q4); MoE experts, convolutions, norms, attention and decoding stay on the CPU | measured on Linux on one RTX 5080 (NVIDIA driver, Vulkan 1.4): logits bit-identical to the CPU path and the same greedy text for the four weight formats, see [Vulkan](#vulkan); the same on one integrated device, a DGX Spark's GB10 ([DGX Spark](#dgx-spark-aarch64-linux)); other vendors, Windows, Android and MoltenVK not run |
| wasm32 (WASI) | simd128 kernels (bf16, Q8, Q4), threads via wasi-threads | measured: Qwen3.5-0.8B Q8 at 123 tok/s decode with 16 threads under wasmtime 30, identical output to native; see below |
| Browser (Chrome 137+) | simd128 kernels + WebGPU GEMM for prefill (WGSL, via JSPI) | measured in headless Chrome 155: Qwen3.5-0.8B Q8 prefill 34 -> 231 tok/s, identical output; Firefox/Safari run the CPU path (no JSPI) |

What the GPU does and does not do: every backend only takes prefill-sized matmuls (`t >= 16`
tokens and enough FLOPs to amortize the ~0.15 ms it costs to submit and wait for a Metal command
buffer). Weights are not copied on Apple Silicon: quantized matrices and the safetensors blob
live in page-aligned memory and are wrapped as shared `MTLBuffer`s. The Metal GEMM (64x64 tile per
threadgroup, `simdgroup_float8x8`, weights dequantized to f32 in threadgroup memory) runs at
~10 TFLOP/s for bf16 and ~7.5 for Q4, against ~1.3 TFLOP/s for the NEON GEMM.

On M5 / A19 (Apple10-family GPUs, whose cores have matrix units) the GEMMs run on those instead,
through Metal 4 tensor operations (`matmul2d` from MetalPerformancePrimitives, compiled at run
time with the rest of the shaders; nothing is needed at build time). Input rows are converted to
half with a power-of-two scale per row (no overflow, no subnormals), quantized weights are
dequantized per 64-wide K step into threadgroup memory as half (Q8_0 and Q4_0 in half arithmetic,
which gives the same values: their products are exact in f32 and rounded once either way), bf16
weights are read as bfloat straight from device memory, and accumulation is f32: every format
with a GEMM (bf16, Q8, Q4, the GGUF types, FP8 E4M3 with 128x128 block scales) has a matrix-unit
version. Kernel throughput at 2048 tokens is ~60 TFLOP/s for
bf16 and ~50 for Q4 (against ~11 for the f32 GEMM); per call, copies and submission included
(`membench gpu`, 17408 x 5120):

| Tokens | bf16 f32 GEMM / matrix units | Q4 | Q4_K | Q8_0 |
|---|---|---|---|---|
| 256 | 9.7 / 10.5 TFLOP/s | 11.0 / 19.8 | 10.7 / 18.1 | 10.9 / 18.0 |
| 512 | 11.7 / 24.4 | 11.6 / 28.9 | 11.5 / 28.2 | 11.5 / 28.3 |
| 1024 | 12.0 / 41.4 | 11.6 / 42.1 | 11.7 / 38.7 | 11.7 / 37.1 |
| 2048 | 12.2 / 43.8 | 11.8 / 42.6 | 11.5 / 39.6 | 12.1 / 40.6 |

The products are accurate to ~3e-4 relative (half operands) instead of bit-identical to the CPU;
`OXINFER_METAL_MMA=0`, the engine option `gpu_exact=1` or `oxinfer::gpu::set_exact(true)` keep
the f32 GEMM (`gpu_moe=0` / `set_moe_offload(false)` keep the experts on the CPU). MoE layers (Qwen3.5/3.6-35B-A3B) run their routed
experts in one submission per layer: the tokens are gathered into one segment per expert, a
grouped GEMM walks per-expert tiles for the gate/up projection, then SiLU, the grouped down
projection and the weighted combination, all on the GPU (the expert weights are kept resident
with an `MTLResidencySet`; making them resident per submission cost more than the GEMMs).

Single-token decode runs entirely on the GPU where the model and its weight formats are supported
(`--gpu-decode on`, the default; `off`, the engine / speaker option `gpu_decode=0` or
`oxinfer::gpu::set_decode(Some(false))` keep it on the CPU): the model describes one token's
forward pass as a graph of device ops (`gpu::decode`), the Metal backend compiles it once into
kernels and buffer bindings, and each token is one command buffer (plus a small first one, so the
GPU starts while the rest is encoded) with a single wait. The host only looks up the input
embedding (and Gemma's per-layer-embedding row) and samples from the logits. What runs on the GPU:

| Model family | On the GPU per token |
|---|---|
| Llama 3.x (safetensors, GGUF), Orpheus TTS | RMSNorm, Q/K/V matvecs, RoPE (rotate-half or GGUF's interleaved pairs), KV append, attention, output projection + residual, fused SiLU gate/up, down + residual, final norm, LM head (skipped when only the hidden state is wanted, as Orpheus's restricted audio head runs on the CPU) |
| Qwen3 (dense) | the same plus per-head Q/K RMSNorm |
| Qwen3.5 / 3.6 / 3.8 (dense) | full attention layers with the output gate and partial RoPE; Gated DeltaNet layers: in/z/b/a projections, causal conv + SiLU over the conv state, the gated delta rule (state rows in registers, read and written once) with the gated RMSNorm; the recurrent states are the host's own page-aligned memory, updated in place |
| Qwen3.5 MoE (Zawen-35B-A3B, ...) | router and the shared expert's gate as one f32 matvec, top-k + softmax, the chosen experts' gate/up and weighted down matvecs read through per-expert address tables (fused or GGUF split gate/up), the shared expert scaled by its sigmoid gate |
| Gemma 4 E2B / E4B / 26B-A4B, Gemma 3 | per-layer embeddings, sliding and full attention (two RoPE tables, head size up to 512), KV sharing across layers, value norms, K = V layers, sandwich norms (each fused with the next norm), GELU-tanh MLPs, 26B-A4B's routed GELU experts beside the dense MLP (router scale, per-expert scales), layer scalars, the softcapped head |

Matvec kernels exist for bf16, oxinfer's Q8 and Q4 (also MLX affine Q4) and every GGUF format
with a GEMM (specialized kernels for F16, Q4_0, Q8_0, Q4_K, Q5_K and Q6_K; the others through
their GEMM dequantizers): one SIMD-group or more per row, lanes taking consecutive chunks of a
row so a SIMD-group reads contiguous memory, the nibbles (and Q5_K's high bits) of a word
unpacked four at a time as bytes, K-quant scales decoded without branches; 450-590 GB/s on
large matrices of an M5 Max (Q5_K ~360-390: twice the float work per value of Q4_K, as the high
bits are summed apart to keep single- and multi-token results identical; `cargo test --release
--lib decode_mv_bench -- --ignored`).
Attention scores 32 keys per SIMD-group step (one lane per key), reads each key and value row
once for the query heads sharing it, and splits long contexts over up to 64 threadgroups per head
group. Activations stay f32 end to end (the CPU kernels quantize them to int8 for the quantized
formats), and so do the KV caches: the host caches stay authoritative and the device keeps a
copy per session, appending what CPU passes added (prefill chunks, speculative verification) and
forgetting what a rewind dropped. Not on the GPU: models with LoRA adapters attached, streamed
weights (`--stream`), bidirectional configurations, the vision / audio towers, and WebGPU (no
decode graph backend yet). Multi-token passes run on multi-token plans (speculative
verification, bit-identical to one-token steps) and prefill plans (below).

Prefill passes of 32 tokens and more run whole on the GPU too, for every model with a decode
graph (Llama 3, Qwen3, Qwen3.5 / 3.6 / 3.8 dense and MoE, Gemma 3 / 4 incl. 26B-A4B,
Qwen3.8-Flash-Next): the graph without its LM head compiled as a prefill plan
(`MetalBackend::build_plan_pf`, chunks of up to 2048 tokens), the activations on the device for
the whole chunk, the matmuls as matrix-unit GEMMs, the routed experts as grouped GEMMs, and plain
causal or sliding-window attention as flash attention on the matrix units (`metal_pfmma.metal`,
as llama.cpp's `kernel_flash_attn_ext_tensor`: S = Q K^T and O += P V with `matmul2d`, an online
softmax in threadgroup memory, over half copies of the cache rows that the RoPE kernel writes for
the chunk and `pf_kv_half` converts for up to 6144 earlier positions; Gemma's unscaled scores in
f32, see [Metal prefill round 5](performance.md#metal-prefill-round-5-november-2026)), or else on
8x8 SIMD-group matrices (`metal_pfflash.metal`: 32 queries per threadgroup sharing the key reads;
heads of 256 and 512 split their dimensions over SIMD-groups; `OXINFER_PF_MMA_ATTN=0`). RMSNorms,
Gemma's add-norms and the SwiGLU / GeGLU before a GEMM write its half input rows themselves.
The compiled ops run in stages by dependence rather than in graph order (`pf_schedule`, see
[Metal prefill round 6](performance.md#metal-prefill-round-6-november-2026)): a MoE layer's dense
feed-forward runs beside its routed experts' GEMMs, and Gemma 4's MoE layer tail (the experts'
combine, the post norms and the residual step) is one kernel.
Token tiles are 128 rows, 64 for narrow outputs (fewer than 160 threadgroups of 128 rows: e.g.
Llama-3.2-1B's 2048-wide projections at 512 tokens), 32 below that, and a last partial tile of
up to 64 rows runs as 32-row tiles rather than one padded tile (513 tokens: 544 rows instead of
640); residual adds happen in the GEMMs' stores, and the MoE routers run as an f32 GEMM of
32x32 tiles. Only the new keys and values come back to the host caches, each layer's while the
GPU runs the later layers (a pass is 8 command buffers; MoE passes are cut at every MoE layer
anyway, for the routing). Interventions, captured layers,
streamed weights, M-RoPE position lists and Gemma image prompts keep the op-by-op path
(`OXINFER_PREFILL_GPU=0` forces it, `OXINFER_PF_FLASH=0` the per-token attention kernels). The
last-token distribution is as close to the CPU path's as before or closer (Qwen3-8B Q8_0, 2.5k
tokens: KL 2.2e-5 -> 6.4e-6; Gemma 4 12B Q8_0: 1.0e-4 -> 5.5e-5; Llama-3.2-1B Q4_K_M 2.5e-5 ->
2.3e-5), and greedy output is unchanged on the models checked (Llama 1B, Orpheus, Qwen3-8B / 14B,
Qwen3.5-9B, Qwen3.8-27B, Gemma 4 12B / 26B-A4B; 2 prompts x 64 tokens each). CUDA runs the same graphs, see [CUDA](#cuda).

Before that, forwarding only the matmuls of decode passes to the GPU was measured and lost: on
the 27B GGUF (`OXINFER_METAL_MIN_T=1`) a pass cost ~257 submissions with CPU work in between
(~50 ms of submit-and-wait alone), so plain decode dropped from 10.7 to 5.5 tok/s.


## CUDA

Measured on an RTX 5080 (16 GB, compute capability 12.0) against a 64-core AVX-512 CPU, with a
3,600-token prompt for the language models; "same" is the CPU path's result compared with `--device
cuda`.

| Model | Exercises | Agreement with the CPU | CPU | CUDA |
|---|---|---|---|---|
| Qwen3.5-0.8B bf16 | GEMM | logits bit-identical, same text | 609 tok/s | 915 tok/s |
| Qwen3.5-0.8B `--quant q8` / `q4` | GEMM | same text | 646 / 632 tok/s | 923 / 927 tok/s |
| mlx-community/Qwen3-4B-4bit | affine Q4 | logits bit-identical, same text | 144 tok/s | 220 tok/s |
| mlx-community/gemma-4-26b-a4b-it-4bit (MoE) | affine Q4 dense layers, attention, routed experts (the layers whose experts fit in VRAM; the rest on the CPU) | logits rel L2 1.3e-3, same top 3: the device attention sums in another order (with it off the dense path is bit-identical; the experts add 1.4e-6) | 112 tok/s | 169 tok/s (149 with the experts on the CPU) |
| Qwen3.5-9B (15.9 GB of streamed weights, more than fits) | layers past VRAM stay on the CPU | logits bit-identical, same text | 94 tok/s | 261 tok/s |
| SD-Turbo, 512x512 | convolutions, norms, attention | 83.4 dB; `OXINFER_GPU_CHECK=1` worst rel L2: attention 3.5e-6, conv+norm 1.5e-6 | UNet 2.56 s, VAE 9.16 s | UNet 0.88 s, VAE 1.83 s |
| whisper-small `--precision bf16` | encoder | same transcript | encoder 1.44 s | encoder 0.50 s |
| FLUX.1-Kontext-dev `--quant q4`, 512x512, 4 steps | GEMM, attention, norms, VAE | 71.3 dB (max pixel diff 1); `OXINFER_GPU_CHECK=1` worst rel L2: attention 4.1e-6, conv+norm 4.1e-7 | 516 s whole run | 47 s whole run |
| Qwen-Image-2512 `--quant q4 --mmap`, 512x512, 4 steps | GEMM, attention, VAE | 56.0 dB (max pixel diff 33) | 718 s whole run | 69 s whole run |
| Step1X-Edit-v1p2 `--quant q4 --mmap`, 4 steps | GEMM, attention, VAE | 70.5 dB (max pixel diff 1) | 3,201 s whole run | 938 s whole run |
| OmniGen2 `--quant q4`, edit at 512x512, 4 steps | GEMM, attention, VAE | 82.4 dB (max pixel diff 2) | 552 s whole run | 73 s whole run |
| BAGEL-7B-MoT `--mmap --max-size 512`, edit, 4 steps | GEMM, attention, VAE | 78.7 dB (max pixel diff 1) | 517 s whole run | 85 s whole run |
| Janus-Pro-1B `--generate-image` | GEMM, image decoder | 95.6 dB (max pixel diff 1) | 84 s whole run | 38 s whole run |

Short inputs stay on the CPU (see `wants`), but opening the device costs about 0.1 s at start-up.

Prompt speed with `--quant` weights that fit, measured again in November 2026 with llama-bench
style passes (Qwen3.5-9B: q4 391-397 tok/s at 512 tokens and 364-368 at 2048, and where the time
goes): [x86-64 CPU kernels](performance.md#x86-64-cpu-kernels-november-2026).

**Decoding.** Whole decode steps run on the GPU as compiled plans of the decode graph
(`cuda_decode.rs`, the kernels ported from Metal's): matvecs (bf16, Q8, Q4, affine Q4, the GGUF
block formats, see below), norms,
RoPE, attention (split over up to 64 chunks for long contexts), Gated DeltaNet, MoE routing and
experts. 128 greedy tokens after a short chat prompt (fixed system prompt; the default one
carries the time), same machine:

| Model | CPU decode | CUDA decode | Text |
|---|---|---|---|
| Qwen3.5-0.8B bf16 | 35 tok/s | 304 tok/s | identical |
| Qwen3.5-0.8B `--quant q8` | 53 tok/s | 369 tok/s | identical |
| Qwen3.5-0.8B `--quant q4` | 74 tok/s | 421 tok/s | parts (int8 activations on the CPU, below) |
| mlx-community/Qwen3-4B-4bit | 23 tok/s | 172 tok/s | identical |
| Qwen3.5-9B `--quant q4` | 12 tok/s | 107 tok/s | identical |

`OXINFER_GPU_DECODE_CHECK=1` (every GPU step also on the CPU): KL 1e-11 on bf16, also after a
3,700-token prompt (split attention); 1.5e-3 to 3.6e-3 on the quantized formats, whose CPU
kernels quantize the activations to int8 while the GPU keeps them f32.

Speculative verification runs as multi-token plans: their matvecs read each weight once for all
the tokens, each token's sum taken exactly as the single-token kernel takes it (the test
`decode_matvecs_match_definition_and_each_other` checks the bits), and the recurrent states are
replayed on a rewind. Greedy output with MTP (Qwen3.5-9B) or a DSpark draft (Qwen3-4B with
`deepseek-ai/dspark_qwen3_4b_block7`) is plain decoding's, token for token. They are slower than
plain GPU decoding here (40 and 154 tok/s against 107 and 172): the drafts run on the CPU, and a
Q4 verification pass is compute-bound on the CUDA cores. So with CUDA decoding they are not the
default (`Backend::prefers_plain_decode`); `--mtp` / `--draft` still use them.

Declined (decoding stays on the CPU): interventions, Qwen4-Exp, GGUF weights in the ternary /
1-bit formats or in MoE experts (below), and models whose
weights do not all fit in VRAM: Gemma 4 26B-A4B's experts do not on 16 GB
(`OXINFER_GPU_PLAN_DEBUG=1` names the op). Recurrent states go up and down every step (copied back
on a second stream under the remaining layers; the upload is skipped when nothing on the host
wrote them).
The image-model timings include loading and quantizing the weights.

`cargo test --lib cuda::` compares every kernel with a reference computed in f64 from its
definition; it fails if a kernel does not compile or load, and skips without an NVIDIA GPU or NVRTC.

**GGUF weights.** A GGUF file's matrices are copied to the device as stored and read there by
the kernels Inkling-2's engine already had for them (`inkling_cuda.cu`, compiled once per ggml
format the first time a weight of it is uploaded and kept in `~/.cache/oxinfer/kernels`): F32,
F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ1_S, IQ1_M, IQ2_XXS,
IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS and MXFP4. Three things run on them:

- **Decode steps**: a matvec per format, a warp per row, activations f32 (a gate / up pair is
  two matvecs and the activation). Multi-token plans (speculative verification) launch the same
  kernel with a grid row per token, so their results are the single-token steps' bit for bit.
- **Prompts, whole pass** (`decode_plan_prefill`, steps of up to 512 tokens,
  `OXINFER_CUDA_PREFILL_TOKENS`; `OXINFER_PREFILL_GPU=0` keeps the op-by-op path): the matmuls as
  tensor-core GEMMs with the weights' integer values as stored (exact in f16), the activations as
  pairs of halves and each 16-column step's sums scaled in f32 (`gemmi_*`); RMSNorm, RoPE, the
  activations, the Gated DeltaNet convolution and delta rule (the state in registers across the
  step's tokens) over all tokens in one launch; attention with the decode kernel, 32 tokens per
  launch; the hidden states, the new keys and values and the recurrent states come back to the
  host. The halves hold activations below 32768 in magnitude: a kernel checks every GEMM input
  on the device, and a step with a larger or non-finite value is dropped whole and runs on the
  CPU path (the op-by-op GEMM checks its input on the host). Run on Qwen3.8-27B, Llama 3.2 1B and Orpheus 3B; Qwen3 and Qwen3.5 GGUF files build the
  same graph and were not run.
- **Prompts, op by op** (interventions, image prompts' position lists, or when the plan cannot
  be built): the same GEMM for the prefill-sized matmuls, the rest on the CPU (the 27B: 213
  tok/s at 512 tokens, against 199 on the CPU alone).

Still on the CPU, automatically: the ternary and 1-bit formats (TQ1_0, TQ2_0, Q1_0, Q2_0, PQ2_0,
PTQ1_0) and FP8 matrices with block scales (no kernels; a model with one such matrix decodes on
the CPU), models with GGUF MoE experts (`qwen35moe`, Gemma 4 26B-A4B: the expert kernels read
bf16 / Q8 / Q4 only; their dense matmuls take the GEMM in prompts), Hadamard-folded matrices,
and models whose weights do not all fit in VRAM. Gemma GGUF files go through the same kernels
(their sandwich norm one token at a time in a prefill plan) and were not run: none is on this
machine. Prefill plans exist for GGUF weights only: bf16 / Q8 / Q4 prompts keep the op-by-op
path.

Measured on the RTX 5080 against the CPU path (32 threads) and llama.cpp's CUDA backend (master
`e300919`, 2026-11-22, `llama-bench -ngl 99 -p 512,2048 -n 128 -r 3`), `examples/parity_bench.rs`
as in [the llama-bench matrix](performance.md#llama-bench-matrix-november-2026) (`lookup=0
draft=none mtp=0`, median of 3; release build). The oxinfer rows are from the tree with the
[x86-64 CPU prefill](performance.md#x86-64-cpu-prefill-november-2026) work (the CPU's prompts on
its int8 GEMM, the default; `exact`: `prefill_gemm=exact`, the f32 GEMM), at load averages 5-24;
the llama.cpp rows from earlier the same day, at load averages 15-21, beside the
same CUDA code (which measured 539 / 509 / 31.2 on the 27B then). tok/s:

| Model | | pp512 | pp2048 | decode |
|---|---|---:|---:|---:|
| Qwen3.8-27B UD-Q3_K_XL (unsloth, 13.1 GB: IQ4_XS, IQ3_S, IQ3_XXS, IQ2_*, Q2_K-Q6_K, Q8_0, IQ4_NL) | CPU (what `device=cuda` ran before) | 199 (`exact` 81.5) | | 5.1 |
| | CUDA | 537 | 516 | 31.1 |
| | llama.cpp | 2488 | 2476 | 58.1 |
| Llama-3.2-1B Q4_K_M (Q4_K, Q6_K) | CPU | 2405 (`exact` 1235) | | 74 |
| | CUDA | 8485 | 7440 | 498 |
| | llama.cpp | 43978 | 46597 | 755 |
| Llama-3.2-1B Q8_0 | CPU | 2288 (`exact` 1240) | | 48 |
| | CUDA | 8971 | 7617 | 451 |
| | llama.cpp | 46176 | 48477 | 533 |
| Orpheus 3B Q4_K_M (the CPU's and CUDA's decode over 22 tokens) | CPU | 1154 (`exact` 557) | | 32 |
| | CUDA | 2352 | 1605 | 230 |
| | llama.cpp | 18239 | 17757 | 319 |

Agreement with the CPU path (`examples/qwen_greedy_dump.rs`: a raw prompt of 1,500 characters fed
as one batch, then greedy tokens one at a time, the logits of every step compared; four prompts
on the 27B, five on Llama 1B Q4_K_M). The plan's GEMMs keep f32 activations, so the prompt
matches the CPU's f32 GEMM (`OXINFER_PREFILL_GEMM=exact`): last-token distribution within KL
4e-11 to 7e-7 on the 27B (largest logit difference 0.012) and 2e-12 to 1.1e-7 on Llama 1B
(0.005). Against the CPU's default for prompts, the int8 GEMM (`fast`), it differs by that
GEMM's rounding: KL 6e-6 to 8e-3 on the 27B (logits up to 0.62) and 1.3e-4 to 1.5e-3 on Llama
1B (0.28). Decode steps differ as on the other quantized formats (the CPU kernels quantize
activations to int8, the GPU keeps them f32): mean KL 1e-4 to 7e-4 per step (1.2e-3 to 1.6e-3
on one 27B prompt), and with `OXINFER_GPU_DECODE_CHECK=1` on the 27B 1.5e-4 with the same top
token on 40 of 40 steps. Greedy tokens identical to the CPU's:

| | CPU prompt on `exact` | CPU prompt on `fast` (default) |
|---|---|---|
| Qwen3.8-27B, 32 tokens | 3 of 4 (the fourth parts at token 20, 0.337 against 0.335) | 2 of 4 (one parts at token 20, one at the first token, 0.147 against 0.144) |
| Llama-3.2-1B Q4_K_M, 48 tokens | 5 of 5 | 4 of 5 (the fifth parts at token 6, 0.059 against 0.056) |
| Llama-3.2-1B Q8_0, 48 tokens (before the CPU work, when its prompt GEMM was the f32 one) | 3 of 5 (the others part at tokens 26 and 28, 0.097 against 0.091 and 0.054 against 0.053) | |

`cargo test --lib cuda::tests::ggml` checks the matvec and the GEMM of every
format against the dequantized rows' products in f64 (2e-6 of the summed magnitudes), a
graph with an attention layer and a Gated DeltaNet layer over GGUF weights as a single-token, a
multi-token (bit-identical) and a prefill plan (2e-4 of the largest value), and that a load,
run, drop and reload leaves the device's free memory where it was.

**Memory.** GGUF device weights are owned like the other formats': a matrix's copy by its layer,
freed when the model drops it; the plans' arenas, constants and pinned buffers by the plans,
which the model owns; the KV copies and recurrent states by the session. So an unload (the
server's idle unload, a model dropped by the host) returns the device memory: three load / run /
drop rounds leave 14142 MB free after each drop of Llama-3.2-1B Q4_K_M (13116 MB while loaded)
and 14140, 14140 and 14138 MB after the 27B's (1476 MB while loaded), with the same text each
round. Only the compiled kernels of the formats seen stay for the process, as the main module
does.

Where the time goes, and what is left against llama.cpp (0.1-0.2x on prompts, 0.5-0.9x
decoding):

- **Prompts**: the GEMMs, 807 of the 910 ms of a 512-token step of the 27B (attention 28 ms, the
  delta rule 51, the convolutions 11; `OXINFER_CUDA_DECODE_TRACE=2`, which waits for the device
  after every kernel) and 37 of 49 ms on Llama 1B. `gemmi_*` reaches 22-38 TFLOP/s on the 27B's
  shapes (IQ4_XS and Q4_K 36-38, IQ3_S 22-29, IQ2_S 26;
  `cargo test --profile ci --lib ggml_bench -- --ignored --nocapture`), and llama.cpp processes
  4.6x the tokens per second on the same card. In the kernel, by taking each part out in turn:
  unpacking the weights is 20% of the time on IQ4_XS and 40% on IQ3_S, loading and splitting the
  activations 15-25%, the second (low halves) tensor-core product 17%, the f32 scaling of the
  sums under 3%. Measured and not kept: tiles of 256 tokens x 128 outputs with 16-column steps,
  so that a weight is unpacked once per 256 tokens (26-27 TFLOP/s on IQ4_XS against 34-38);
  weights and activations both split into halves, three products and no f32 scaling (25-27);
  plain f16 operands, which are inexact (59 on IQ4_XS, 28 on IQ3_S).
- **Decoding**: of the 27B's 32 ms per token, about 19 are the 497 matvecs by the kernels' own
  rates (12.3 GB of weights: 710-830 GB/s on IQ4_XS, Q4_K and Q5_K, near what the card's memory
  gives, 570-610 on IQ3_S and IQ3_XXS, 360-490 on the IQ2 formats and Q3_K), 6.6 the Gated
  DeltaNet states' copy into host memory after every step (149 MB) and about 3 the other
  kernels.

**Inkling-2 / Inkling-2-Mini** run whole passes on the GPU with an engine of their own
(`src/inkling_cuda.rs`, `inkling_cuda.cu`: the attention residual's mixes, the loop's message
gates, the short convolutions with rings by position, relative-bias attention with sinks, L2 clips
and output gate, the softcapped sigmoid routing and the SiTU experts in their stored formats,
GGUF files included; see [Inkling-2 Mini](inkling.md#the-cuda-engine)). Prefill matches the CPU
path to KL ~4e-11 at every position (a position where two experts tie within f32 rounding now and
then aside), decode steps the CPU's f32 prefill to 1e-10 - 7e-9, greedy text identical; sessions go
back up to 64 positions (prompt lookup, exact verification passes of up to 16 tokens). On an RTX
PRO 6000 Blackwell against llama.cpp's CUDA backend (built from source, `llama-bench`), tok/s:

| Mini checkpoint | engine | pp512 | pp2048 | tg128 | tg128 at 8k / 32k |
|---|---|---:|---:|---:|---:|
| ggml-org Q8_0 | llama.cpp | 3236 | 2747 | 87.8 | 85.7 / 79.7 |
| | oxinfer | 3558 | 4261 | 120.3 | 110.2 / 98.0 |
| unsloth UD-IQ3_XXS | llama.cpp | 3284 | 2784 | 98.2 | 96.0 / 88.3 |
| | oxinfer | 3321 | 3817 | 131.5 | 120.9 / 106.8 |
| AtomicChat AD-Q4_K | llama.cpp | 3243 | 2752 | 96.6 | 94.4 / 87.0 |
| | oxinfer | 3503 | 4033 | 131.1 | 120.2 / 106.4 |
| release (NVFP4) | oxinfer | 3481 | 4227 | 116.4 | 107.5 / 95.8 |

Prefill of 2048 tokens after 32k of context: 2333 tok/s against llama.cpp's 1094 (Q8_0).

## Vulkan

`--device vulkan` runs the prefill-sized matmuls (bf16, Q8, Q4 and affine Q4 weights, `t >= 16`
tokens, output rows a multiple of 64) as compute shaders on any Vulkan device. Nothing else runs
there yet: MoE experts, convolutions, norms, attention and decode steps stay on the CPU, so this
backend helps long prompts on machines whose GPU has neither Metal nor CUDA. `auto` keeps
preferring Metal and CUDA, and takes Vulkan when it is the only backend that starts (never a
software rasterizer such as llvmpipe, which `--device vulkan` still accepts). `--devices` lists
the Vulkan devices next to the others, the same physical GPU included:

```
GPU (cuda #0): NVIDIA GeForce RTX 5080
  memory: 16.6 GB
  driver: CUDA driver 13.2, compute capability 12.0
GPU (vulkan #0): NVIDIA GeForce RTX 5080
  memory: 17.1 GB
  driver: Vulkan 1.4.329 (discrete)
```

With several Vulkan devices, the first discrete one is used, then integrated ones (the larger
memory first within a type); `OXINFER_VULKAN_DEVICE=N` picks the device listed as `vulkan #N`.

- **Nothing at build time, no extension at run time.** `libvulkan` (`vulkan-1.dll`, MoltenVK) is
  opened with `dlsym` when the device is asked for. The shaders are GLSL (`src/vulkan/gemm.comp`,
  one source for the four formats) and their compiled SPIR-V is checked in next to it and embedded
  in the binary; `tools/vulkan_shaders.sh` regenerates it with `glslc` after an edit. They need
  only Vulkan 1.0 core: the 8- and 16-bit weights are read as 32-bit words, so neither 16-bit
  storage nor `shaderInt8` is required; the device must allow compute workgroups of 256
  invocations with 16 KiB of shared memory (checked at start-up, an error otherwise).
- **Memory.** Unified memory is not assumed. On discrete devices the weights are copied once into
  device-local buffers through a host-visible staging buffer (a layer the first time it is used),
  and every batch copies its activations up and its results down within its one submission; on
  integrated devices the buffers are host-visible and the shaders work in them directly
  (`OXINFER_VULKAN_STAGED=0|1` forces either). A weight matrix larger than the device's
  `maxStorageBufferRange` or `maxMemoryAllocationSize` is split by rows over several buffers, and
  a batch whose activations exceed them runs as several token ranges. A matrix that does not fit
  in device memory stays on the CPU: with `VK_EXT_memory_budget` the heap's budget is checked
  (512 MB kept free), otherwise the allocation's failure decides. Device weights are freed with
  their layer.
- **Arithmetic.** The kernel is the tiled GEMM of the other backends (a 64 x 64 tile of outputs
  per workgroup, weights dequantized to f32 in shared memory), each output one accumulator taking
  one `fma` per input in order, as the CPU prefill GEMM does. Vulkan lets a driver compile `fma`
  as a separate multiply and add, so bit-identity with the CPU is a property of the driver, not
  of the backend.

Measured on an RTX 5080 (NVIDIA driver, Vulkan 1.4.329) against the AVX-512 CPU path: the logits
after an 1,823-token prompt are bit-identical (`--dump-logits`, files
equal byte for byte) and greedy decoding (up to 64 tokens) gives the same text, for

| Model | Weights |
|---|---|
| Qwen3.5-0.8B | bf16 |
| Qwen3.5-0.8B `--quant q8` | Q8 |
| Qwen3.5-0.8B `--quant q4` | Q4 (also with `OXINFER_VULKAN_STAGED=0`) |
| mlx-community/Qwen3-4B-4bit | affine Q4 |

On a DGX Spark's GB10 (an integrated device) the same holds for the first three, the quantized
ones against the CPU's f32 prompt GEMM: [DGX Spark](#dgx-spark-aarch64-linux). No other device
or driver was run, and no speed was measured.

`cargo test --lib vulkan::` runs the kernels in both memory modes against the exact (f64)
product of the dequantized weights, the same product in the kernel's order of operations and the
CPU kernels, for each format; the f16 scale decoding on every finite value; batches of several
outputs; weights split over several buffers and batches split over token ranges (the limits
lowered, results bit-identical to the unsplit path); declined shapes; freed memory; concurrent
calls. `bit_identity_report` prints how many outputs differ from the fma-ordered reference on the
machine at hand (none here). The tests skip without a Vulkan device.

Instances and devices are created and destroyed one at a time in a process
(`platform::gpu::vulkan::lifecycle`): with NVIDIA's 580.173 driver on a DGX Spark, threads doing
so at once could leave a driver thread spinning in its exit handler, `vkDestroyDevice` waiting
for it and every later call waiting for the loader.

## DGX Spark (aarch64 Linux)

One NVIDIA DGX Spark, November 2026: 10 Cortex-X925 and 10 Cortex-A725 cores (`neon dotprod i8mm
bf16 fp16 sve sve2`, no SME), a GB10 GPU sharing the system's 131 GB, Ubuntu 24.04, kernel
6.17, driver 580.173.02 (CUDA 13.0, compute capability 12.1), NVRTC 13.0.88 from the toolkit's
`sbsa-linux` libraries, Vulkan 1.4.312, Rust 1.100. It is the first aarch64 Linux machine and
the first CUDA device outside x86-64 that oxinfer ran on; the aarch64 kernels had run on Apple
Silicon only. Two other jobs measured speed on the machine at the same time, so nothing here is
a timing.

```
CPU: aarch64 on linux, 20 logical cores
  features: neon dotprod i8mm bf16 fp16 sve sve2
  memory: 130.7 GB
GPU (cuda #0): NVIDIA GB10
  memory: 130.7 GB (unified with CPU)
  driver: CUDA driver 13.0, compute capability 12.1
GPU (vulkan #0): NVIDIA GB10
  memory: 98.0 GB (unified with CPU)
  driver: Vulkan 1.4.312 (integrated)
GPU (vulkan #1): llvmpipe (LLVM 20.1.2, 128 bits)
  memory: 130.7 GB
  driver: Vulkan 1.4.318 (cpu)
```

Before that listing the Vulkan loader prints `TU: error: ... failed to open device
/dev/dri/renderD128 (VK_ERROR_INCOMPATIBLE_DRIVER)`: Mesa's Turnip driver (one of the eleven
ICDs Ubuntu installs) trying NVIDIA's render node when the loader enumerates devices. It is not
oxinfer's and changes nothing; `VK_LOADER_DRIVERS_DISABLE='*freedreno*'` removes the line.

**What the platform changed.**

- *Unified memory under CUDA.* The device reports `CU_DEVICE_ATTRIBUTE_INTEGRATED`, and is
  listed as unified. Its `cuMemGetInfo` gives the system's `MemFree` (108.61 GB of 130.66 with
  `MemAvailable` at 118.28): the page cache, which the kernel gives back on demand and which
  holds the model files just read, counts as used. What a load may put on the device is
  therefore taken from the system's available memory less the reserve every load leaves
  (`cuda::usable_free`), for the GEMM weights, the Inkling-2 and MiniMax-H3 engines and the
  server's eviction. The device's copy of the weights is a second one in the same memory:
  Qwen3.5-0.8B bf16 holds 1.76 GB resident and 1.75 GB on the device (1.75 GB resident on the
  CPU alone; with `--mmap` the host's copy is page cache). `--models` counts both, so models up
  to 60 GB decode on the GPU here. The test that a dropped model returns its device memory
  prints the figures on such a device instead of asserting them (every process moves them by
  more than its 164 MB). Pinned memory (`cuMemAllocHost`) works as on a discrete card.
- *Host-to-device copies.* `cuMemcpyHtoD` may return with a copy of 64 KB or less from pageable
  memory still queued on the default stream. The Inkling-2 and MiniMax-H3 engines launch their
  kernels on a stream that does not wait for that one, so they now wait for each copy. On the
  Spark the race showed as `inkling::cuda::tests::matvecs_and_gemms_match_definition` failing
  2 times in 14 runs beside the other CUDA tests (a matvec reading its input before it
  arrived); none in 48 runs since. `inkling::cuda::tests::small_ops_match_cpu`, reported flaky
  on x86-64 (about 1 run in 16), uploads its inputs the same way; it did not fail in any run
  here, before or after.
- *Vulkan devices one at a time*, see [Vulkan](#vulkan): the suite hung in 2 of 9 runs before,
  in none of 22 since.
- Nothing else needed a change: `libcuda.so.1` and `libnvrtc.so` are found by name through
  `ld.so.conf`, NVRTC compiles every kernel for `sm_121` with the same options, feature
  detection is the standard library's (SME from `AT_HWCAP2`: absent, so its tests return at
  once), pages are 4 KB against Apple's 16 KB, and the thread pool takes all 20 cores.

**Test suite.** `cargo fmt --check`, `cargo clippy --all-targets --all-features` and
`--all-targets --no-default-features` (both with `-D warnings`) are clean. `cargo test --profile
ci --all-features`: 463 unit tests pass and 11 are ignored (benchmarks and tests that need files
not in the repository), the server's 10 offline tests pass and its 9 that fetch models are
ignored, as are the 2 Inkling-2 rollback tests (weights) and the 4 media round trips; 9 doc
tests pass (`tests/gemma_moe_gpu.rs` is for Apple Silicon and has no test here). The 23 CUDA tests (main backend, GGUF kernels and plans, Inkling-2 and MiniMax-H3
engines) and the 9 Vulkan tests run on the GB10 in both memory modes, none skipped. Compiled
out on this target: the x86-64 kernels' tests and the Metal and Core ML ones.
`bit_identity_report` (CUDA and Vulkan): no output of the bf16, Q8 and Q4 prefill GEMMs differs
from the CPU kernels.

**Agreement.** Qwen3.5-0.8B on five chat prompts of 56 to 87 tokens and one of 2,064 (7,000
characters of documentation and a question), `--dump-logits` for the last prompt token and 64
greedy tokens without a system prompt. The reference is transformers 5.24 with PyTorch 2.15 in
float32 on the CPU (`tools/ref_logits.py`'s comparison, the model loaded once). "exact" is
`OXINFER_PREFILL_GEMM=exact`, the f32 prompt GEMM.

| Qwen3.5-0.8B, CPU against transformers float32 | Largest logit difference | KL | Same top token | Same 64 tokens |
|---|---|---|---|---|
| bf16 | 4.2e-5 to 1.1e-4 | 2e-11 to 1.7e-10 | 6 of 6 (same top 10 in order) | 6 of 6 |
| `--quant q8` | 0.34 to 0.65 | 4.3e-4 to 9.6e-3 | 6 of 6 | 1 of 6 |
| `--quant q8`, exact | 0.24 to 0.40 | 2.4e-4 to 3.3e-3 | 6 of 6 | 2 of 6 |
| `--quant q4` | 4.3 to 6.1 | 0.33 to 1.35 | 3 of 6 | 0 of 6 |
| `--quant q4`, exact | 4.5 to 6.0 | 0.33 to 1.26 | 3 of 6 | 0 of 6 |

The Q4 row is the quantization of a 0.8B model, not the kernels: the same weights give the same
logits on the three devices below, and the kernels match their f64 definitions in the tests.

| Against the CPU path | Device | Prompt logits | Same 64 tokens |
|---|---|---|---|
| Qwen3.5-0.8B bf16 | CUDA | bit-identical on the five short prompts (the two of 85 and 87 tokens have matmuls on the device, the others stay under its thresholds); 4.1e-5 at most on the long one (KL 1.2e-11), the same bits in three runs | 6 of 6 |
| | Vulkan | bit-identical, 6 of 6 | 6 of 6 |
| Qwen3.5-0.8B `--quant q8`, exact on both | CUDA | bit-identical on the five; 1.2e-5 on the long one | 3 of 6 |
| | Vulkan | bit-identical, 6 of 6 | 6 of 6 |
| Qwen3.5-0.8B `--quant q4`, exact on both | CUDA | bit-identical on the five; 1.0e-2 on the long one (KL 5.2e-7) | 3 of 6 |
| | Vulkan | bit-identical, 6 of 6 | 6 of 6 |
| Qwen3.5-0.8B `--quant q8` / `q4`, defaults | CUDA | bit-identical on the three shortest; on the others 0.34 to 0.58 / 0.53 to 0.58 (KL 3.4e-4 to 4.9e-3), same top token | 2 / 1 of 6 |
| | Vulkan | 0.30 to 0.48 / 0.39 to 0.52 (KL 3e-4 to 6.8e-3), same top token 6 of 6 | 3 / 2 of 6 |
| Qwen3.5-9B `--quant q4`, exact on both | CUDA | bit-identical on 3 of 6, 2.3e-5 to 3.8e-3 on the others (KL 3e-13 to 3e-9), same top 10 | 5 of 6 |
| Qwen3.5-9B `--quant q4`, defaults | CUDA | 0.25 to 0.76 (KL 2.8e-5 to 1.5e-2), same top token 5 of 6 (the other prompt has two tokens 0.2 apart, and the exact CPU run picks CUDA's) | 3 of 6 |
| Llama-3.2-1B Q4_K_M (GGUF), chat prompts | CUDA, prefill and decode plans | 0.25 to 0.45 (KL 3.8e-5 to 4.1e-3), same top token 6 of 6 | 4 of 6 |
| Qwen3.5-0.8B bf16, one image (448 x 336, 140 image tokens) | CUDA / Vulkan | 6.4e-5 / 8.2e-5 (KL 2e-12) | yes / yes |
| SD-Turbo, 512 x 512, one step | CUDA (convolutions, norms, attention) | image 79.8 dB against the CPU's (535 pixels differ, by 1); `OXINFER_GPU_CHECK=1` worst rel L2: attention 1.6e-6, conv+norm 1.4e-6, linear+norm 2.6e-7 | |

How to read the rows that are not bit-identical:

- *The long bf16 prompt on CUDA.* CUDA also takes the first chunk's attention, which sums in
  another order than the CPU's. With the attention kept on the CPU (a build made for this
  check) the logits of the 2,064-token prompt are the CPU's bit for bit, as Vulkan's are, which
  takes the matmuls only. The 9B's three prompts that differ are those of 85 tokens and more,
  where its 16 heads pass the device attention's threshold.
- *Defaults on quantized weights.* On the CPU a language model's prompt takes the int8 GEMM
  (activations rounded to int8, as in decoding) and the GPUs keep f32, so a default run compares
  two roundings: the differences are those between `fast` and `exact` on the CPU alone (0.29 to
  0.57 on these prompts). With `exact` on both sides the GEMMs agree to the bit.
- *Greedy text on quantized weights under CUDA.* Decode steps run on the GPU with f32
  activations and on the CPU with int8 ones. `OXINFER_GPU_DECODE_CHECK=1` over 64 tokens of the
  five prompts (every GPU step also on the CPU): bf16 KL 1.6e-12 on average with the same token
  on 271 of 271 steps; Q8 4.2e-4, 267 of 267; Q4 5.0e-4, 305 of 309; Llama-3.2-1B Q4_K_M
  2.3e-4, 251 of 252. Vulkan does not decode, so its text follows from the prompt.

Llama-3.2-1B Q4_K_M with `examples/qwen_greedy_dump.rs` as in [CUDA](#cuda) (five raw prompts of
1,500 characters fed as one batch, 48 greedy tokens): against the CPU's f32 prompt GEMM the
last-position distribution is within KL 1.3e-12 to 6.4e-7 (largest logit difference 1.8e-5 to
5.1e-3) and the greedy tokens are identical on 3 of 5 (the others part at the second and third
generated token, 0.109 against 0.108 and 0.058 against 0.056); against the CPU's default int8
GEMM, KL 2.2e-4 to 2.9e-3 (0.16 to 0.48) and 2 of 5.

The image prompt against transformers (`tools/ref_vl.py`, float32): pixel values identical,
image features within 1.4e-4 (1.6e-5 relative), logits within 1.0e-4 (KL 1.0e-7) with the same
top 10, and the same 64 greedy tokens.

**Not run on this machine:** speed (measured separately), the affine-Q4 MLX checkpoints, MoE
models, GGUF files other than Llama-3.2-1B Q4_K_M, Inkling-2 and MiniMax-H3 with real weights
(their CUDA engines ran in the unit tests only), Whisper, the image and video pipelines other
than SD-Turbo, speculative decoding, streamed weights and the server with a model.
