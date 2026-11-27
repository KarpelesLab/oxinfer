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
| Apple Silicon (macOS, iOS) | NEON kernels (GGUF Q8_0 / Q4_0 / Q4_K / Q6_K weights interleaved four rows at a time for CPU decoding when read into memory, `cpu_repack`; CPU prefill GEMMs on the SME tiles on M4 and later, `prefill_gemm`) + Metal GEMM for prefill; on M5 / A19 the GPU matrix units (prefill GEMMs and MoE experts), on M1-M4 the same half-input GEMMs on SIMD-group matrices; single-token decode entirely on the GPU (Llama 3, Qwen3 / 3.5 / 3.8 incl. MoE, Gemma 3 / 4 incl. 26B-A4B) | measured; the f32 GEMM gives bit-identical logits to the CPU path, the matrix units and their SIMD-group versions (default) ~3e-4 relative per product (`OXINFER_METAL_MMA=0` for bit-identical); iOS: runs in the Simulator (CPU and Metal; `ios/`), Inkling-2 Mini there greedy-identical to the Mac on CPU, Metal and streamed ([iOS / iPadOS](inkling.md#ios--ipados)) |
| Apple Neural Engine (macOS 15 / iOS 18) | Core ML, for ANEMLL-converted models | measured on macOS; builds for iOS; see [Core ML](coreml.md#core-ml--apple-neural-engine) |
| x86-64 (Linux, Windows, macOS) | AVX2 + FMA + F16C kernels chosen at run time: bf16 / f32 dot tiles and prefill GEMM, Q8 via `maddubs`, Q4, every GGUF format, decode attention; scalar fallback below AVX2 or with `OXINFER_SIMD=scalar` (`--devices` prints the set in use). With AVX-512F the prefill GEMM takes 16 rows x 16 tokens per step and is cache-blocked by columns, bit-identical to the AVX2 GEMM (`OXINFER_SIMD=avx2` keeps the latter; see [x86-64 CPU kernels](performance.md#x86-64-cpu-kernels-november-2026)); the grid-table i-quants unpack with AVX2. With AVX-512 VNNI a language model's prompt takes the int8 prefill GEMM (`vpdpbusd`; `prefill_gemm=fast`, the default there, with the decode kernels' activation rounding; `prefill_gemm=exact` keeps the f32 GEMM; see [x86-64 CPU prefill](performance.md#x86-64-cpu-prefill-november-2026)), and without it `fast` is the f32 GEMM; a Qwen-family prompt's attention is the per-head loop with AVX2 / AVX-512 kernels (the bits of one-token attention; faster here than the score and value GEMMs other architectures take), and the Gated DeltaNet rule and element-wise passes use AVX2 / AVX-512 with the scalar loops' bits; Q4's multi-token column kernel and LoRA's dot products are NEON-only (x86 runs the token-pair and portable loops) | executed on Linux on one Zen 5 CPU (Threadripper 9970X); Windows and macOS compile only. Every AVX2 kernel is tested directly against its scalar fallback (bit for bit for the integer parts, GGUF rows and attention; 1e-6 of the summed magnitudes for the f32 sums). Qwen3.5-0.8B on 5 prompts, AVX2 against scalar: last-prompt-token logits within 4.8e-5 (bf16, Q8, Q4), same top 10; 64 greedy tokens identical in 14/15 runs, and bf16 within 5e-5 of transformers float32 with the same argmax (5/5). Q8 decode steps differ between the two by 0.4-0.6 in the largest logit (same top 1 on 317/320 teacher-forced steps), as far as either is from bf16: the integer block sums are identical (a build with the AVX2 integer path and the scalar order of f32 additions gives the scalar logits bit for bit over 169 steps), the f32 additions are ordered differently, and re-quantizing activations to int8 at every matmul amplifies that. Llama-3.2-1B GGUF (Q4_K_M, Q8_0): decode-kernel passes (`--prefill-chunk 1`) give bit-identical logits and greedy text to the portable path (10/10); with the f32 prefill GEMM the last hidden state differs by rounding (up to 1.8e-4 on values up to 17) and the logits by either ~2e-5 or 1.6e-3 to 5.7e-3: the LM head of the last token is a one-token pass on int8 activations, and the larger figure appears exactly when that rounding moves one to three of the 2048 int8 activations by one step (7/16 pseudo-random prompts on Q4_K_M, 5/16 on Q8_0; an f32 LM head pass on the same states stays within 1e-4 on all 32); on the 5 chat prompts 4.8e-3 / 3.5e-5 at most, 48 greedy tokens identical in 3/5 and 5/5 |
| Linux / Windows + NVIDIA | CUDA driver API + NVRTC-compiled kernels (`cuda.rs`): prefill GEMMs (bf16, Q8, Q4, affine Q4, the GGUF block formats), MoE experts (resident or streamed), convolutions, group / layer norms, attention; whole decode steps, single-token and multi-token (speculative verification), over the same formats; whole-pass prefill plans for GGUF weights ([below](#cuda)); Inkling-2 / Mini whole prefill and decode passes on kernels of its own (every GGUF format but the ternary / 1-bit ones, NVFP4, MXFP8); loaded as a native binary, so a toolkit newer than the driver works | measured on Linux on one RTX 5080 (NVRTC 13.4, driver CUDA 13.2) and one RTX PRO 6000 Blackwell (Inkling-2), see [CUDA](#cuda), and on aarch64 on one DGX Spark (GB10, compute capability 12.1, NVRTC 13.0, unified memory), see [DGX Spark](#dgx-spark-aarch64-linux); Windows and other GPUs not run |
| Linux on aarch64 (Arm's Cortex / Neoverse cores) | the NEON kernels, with the choices that measured faster on a DGX Spark: a language model's prompt takes the int8 prefill GEMM on `SMMLA` where the CPU has i8mm (`SDOT` without), cache-blocked by columns, with the SDOT GEMM's bits; the f32 prefill GEMM (`prefill_gemm=exact`, float-stored weights) is cache-blocked by columns too, bit-identical to the whole-row one; a Qwen-family prompt's attention is the per-head loop on NEON kernels over blocks of queries (the bits of one-token attention) in place of the score and value GEMMs. SVE / SVE2 are not used (128-bit vectors on these cores issue no faster than NEON) and the bf16 instructions neither. `--devices` prints the kernels chosen; `OXINFER_I8_MMLA=0`, `OXINFER_I8_KB=0`, `OXINFER_GEMM_KB=0` and `OXINFER_ATTN_NEON=0` give the previous ones back | measured on one DGX Spark (10 Cortex-X925 + 10 Cortex-A725, no SME), see [DGX Spark CPU](performance.md#dgx-spark-cpu-november-2026): each new kernel is tested against the one it replaces bit for bit, `prefill_gemm=exact` logits with the GEMM attention are byte-identical to the previous build's (Llama-3.2-1B Q4_K_M and Q8_0, Qwen3.5-9B q4, Qwen3.8-27B GGUF), and the per-head loop moves the last prompt position by a KL of 1e-12 to 4e-8; the test suite, Qwen3.5-0.8B against transformers float32 and the three devices against each other are under [DGX Spark](#dgx-spark-aarch64-linux); with an NVIDIA GPU, CUDA and Vulkan as on x86-64; other Arm servers and SME machines running Linux not run |
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
Prefill and decode plans run the routed experts of GGUF checkpoints on tiled copies (each band
of 64 rows stored block column by block column, so that a tile's K step reads one run of memory:
the same results, 3-7% faster prefill passes on an M5 Max); the checkpoint's experts are then let
go (pages of a GGUF file read into memory mapped back from the file, of a mapped one dropped from
the page cache; CPU paths read them from the file again), so the copies take no extra memory. On
GPUs with matrix units, not on iOS; engine option `gpu_expert_tiles=0|1`, `--gpu-expert-tiles`,
`OXINFER_GPU_EXPERT_TILES=0|1`.

If the matrix-unit library fails to compile on such a GPU (a driver or SDK that rejects one of
its kernels), oxinfer falls back to the SIMD-group kernels below, which still work but are 2-3x
slower and have no whole-pass image plans. That is no longer silent: a `[WARNING: ...]` line
says so at once and again after the CLI's `[device: ...]` line, the server's `GET /health` and
`/api/web/activity` report `"degraded": true` with the `warnings`, the library has
`gpu::warnings()` / `gpu::degraded()` and the C API `oxi_warnings()` (a JSON array). A kernel
whose pipeline fails, and a GPU backend that fails to start (`--device auto` then runs on the
CPU), are reported the same way. `OXINFER_METAL_MMA_BREAK=1` injects a compile error into that
library to see it.

GPUs without matrix units (M1-M4) get the same GEMM kernels, under the same names and with the
same operands and rounding (half input rows with a power-of-two scale, weights dequantized to
half, f32 sums), on 8x8 SIMD-group matrices (`MMA_UNITS 0` in `MMA_SHADER_SRC`): the prefill
plans, the grouped expert GEMMs and the op-by-op path take them, as on M5. On an M3 Ultra they
run at 19-22 TFLOP/s at 512-2048 tokens (Q4_K / Q6_K / Q8_0 / Q4_0, 64-token tiles), 15-19 at
32-token tiles. Attention keeps the f32 SIMD-group kernels there, and the matrix-unit attention,
diffusion sessions and Inkling-2's engine stay M5 / A19 only. See
[M3 Ultra](performance.md#m3-ultra-november-2026).

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
their GEMM dequantizers): one SIMD-group or more per row (4 from rows of 2048 values), lanes
taking consecutive chunks of a row so a SIMD-group reads contiguous memory, the nibbles (Q5_K's
fifth bits merged in) of a word unpacked four at a time as bytes, K-quant scales decoded without
branches; 540-595 GB/s streaming large matrices of an M5 Max in each of these formats (Q4_K,
Q5_K, Q6_K, Q8_0, Q4_0 alike; `cargo test --release --lib decode_mv_bench -- --ignored`), less in
a decode pass, where each matvec waits for the op before it (see
[Metal decode round](performance.md#metal-decode-round-november-2026)).
Attention scores 32 keys per SIMD-group step (one lane per key), reads each key and value row
once for the query heads sharing it (up to 512 keys one query head per threadgroup, so short
contexts spread over more of the GPU; each head's arithmetic is the same either way), loads
value rows a few at a time, and splits long contexts over up to 64 threadgroups per head group.
Activations stay f32 end to end (the CPU kernels quantize them to int8 for the quantized
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
the whole chunk, the matmuls as matrix-unit GEMMs (their SIMD-group versions on M1-M4), the
routed experts as grouped GEMMs, and plain
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
(`OXINFER_PREFILL_GPU=0` forces it, `OXINFER_PF_FLASH=0` the per-token attention kernels,
`OXINFER_PF_SSD=0` the token-by-token Mamba-2 scan). The
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

The Qwen3.5-0.8B bf16 row is the op-by-op prompt path, which `OXINFER_PREFILL_GPU=0` still runs
(logits bit-identical to the CPU's). By default the prompts of a model loaded as bf16 now take
whole-pass plans (below): 22,000-25,000 tok/s on this model from 512 tokens on, the first step
within the fast GEMM's rounding of the CPU's ([RTX 5080,
CUDA](performance.md#rtx-5080-cuda-november-2026)).

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

- **Decode steps**: a matvec per format, activations f32; on discrete cards a warp per row and a
  lane per block of 32 in 128-thread blocks, and a gate / up pair of one format in one launch
  with the activation. Multi-token plans (speculative verification) launch the same kernel with
  a grid row per token, so their results are the single-token steps' bit for bit. On an RTX PRO
  6000 Blackwell (`parity_bench`, tg128): Llama-3.2-1B Q4_K_M 768 tok/s and Qwen3-8B Q4_K_M 169
  (666 / 161 with 4 values per lane and the pairs unfused; llama.cpp `e6d624b` 1067 / 254).
  Q4_K and Q6_K blocks then go through a dot of their own (`blockdot`): the stored integers
  become floats by a byte permute into 2^23's mantissa and one subtraction (no conversion
  instructions), Q4_K applies its scale and min once per block (d sc sum q x - m sum x, sum x
  once per lane and block for every row), Q6_K reads its 2-byte aligned blocks as aligned words
  funnel-shifted (two 16-bit loads per word before) and Q4_K its header in one 16-byte load. The
  matvecs were bound by their arithmetic per value (Q4_K, 1.46 times Q6_K's values per byte, ran
  at 0.7 times its bandwidth); on the shapes alone (`ggml_bench`) Q4_K 12288 x 4096 875 -> 1036
  GB/s, 4096 x 12288 809 -> 970, the 151936-row LM head 895 -> 1128; Q6_K 1158 -> 1369, 1063 ->
  1244, 1154 -> 1318. Decode (tg128 median of 3, a GPU another process was using): Qwen3-8B
  Q4_K_M 161 -> 169 tok/s, Llama-3.2-1B Q4_K_M 866 -> 917.
  An int8 matvec for Q4_K, Q6_K and Q8_0 (x rounded per block of 32 as the CPU's decode kernels
  round it, `dp4a` against the stored integers and each block's sums scaled in f32, as
  llama.cpp's `mmvq`) first measured slower, 662 / 142 tok/s with a row per warp, 638 / 154
  with two, 618 / 152 with four: every block of rows rounded the whole of x into 40 KB of shared
  memory, so two blocks fitted an SM. Rounded once per matvec instead (`dq8_x`, into the plan's
  arena; the bytes and scales of the CPU's `quantize_act`, divisions rounded as its; the q / k /
  v matvecs of a layer round their input once), `mvq_K` / `mvqp_K` run a warp per row and a lane
  per block of 32 as the f32 matvec: from L2 (`ggml_bench`, a matrix that fits it) 12288 x 4096
  Q4_K in 6.9 us against 27.5 (7.8 with the rounding), Q6_K 9.8 against 30.3, so the f32
  matvec was bound by its arithmetic and the int8 one reads at what memory gives. Decode (tg128
  median of 3, the GPU 60-70% busy with another process, so against the f32 matvec in the same
  minutes): Qwen3-8B Q4_K_M 113 -> 133 tok/s, Llama-3.2-1B Q4_K_M 492-544 -> 702-704. It is the
  default; `gpu_exact=1` (or `OXINFER_CUDA_MVQ=0`) keeps f32 activations. Greedy decoding (64
  tokens of four `pf_check` prompts) agrees with the CPU's at least as far as the f32 matvec's:
  all four the same with both on three, and on the fourth (Qwen3-8B, 40 tokens) the int8 one
  agrees with the CPU for all 64 tokens where the f32 one parts at the 43rd. Single-token steps
  run as a CUDA graph (one `cuGraphLaunch` for the step's few hundred kernels, recorded on the
  first step and again when the session's device buffers or the ops run change; steps with
  copies interleaved and the per-op trace launch as before; `OXINFER_CUDA_GRAPH=0` off): the
  host queues a step in 14 us instead of 415 (Llama-3.2-1B) / 727 (Qwen3-8B), and the GPU's
  time per token goes 969 -> 934 us / 4346 -> 4263 (idle RTX PRO 6000, `--no-lookup`), the
  kernels' own gaps being small already. Consecutive int8 matvecs of one format over one x (a
  layer's q / k / v) run as one launch (`mvq3_K`, each block one matrix's `mvq_K` block, the
  same sums bit for bit, `cuda::tests`), so the small k / v ones do not take the GPU alone:
  934 -> 898 us / 4263 -> 4180; plain decode 844 -> 917 tok/s and 209.8 -> 218.9.
  Multi-token steps of the int8 matvecs (speculative verification, batched rows) sweep the
  weights once for up to 8 tokens (`mvqt_K`, `mvqpt_K`, `mvq3t_K`: a lane's block of a weight row
  loaded and unpacked once, then `mvq_K`'s dot with each token's block of x in turn), each
  token's sums its own launch's bit for bit (`cuda::tests`; `OXINFER_CUDA_MVQT=0`: a grid row
  per token as before).
- **Prompts, whole pass** (`decode_plan_prefill`, steps of up to 512 tokens,
  `OXINFER_CUDA_PREFILL_TOKENS`; `OXINFER_PREFILL_GPU=0` keeps the op-by-op path): the matmuls as
  tensor-core GEMMs. By default (`gemmh_*`) the activations go in as f16 rows (converted once per
  GEMM) and the weights as f16 values (Q4_K and Q6_K unpacked in half arithmetic, the other
  formats through the per-format dequantizer), with f32 sums over the whole input in the tensor
  cores, 128 x 128 tiles, the next step's activations by `cp.async` and its weight bytes
  prefetched into L1 during the current step's products. With `gpu_exact=1` (or
  `OXINFER_METAL_MMA=0`) the weights' integer values go in as stored (exact in f16), the
  activations as pairs of halves and each 16-column step's sums are scaled in f32 (`gemmi_*`,
  within ~1e-6 of f32). On an RTX PRO 6000 Blackwell (`parity_bench`, median of 3), exact /
  default: Qwen3-8B Q4_K_M pp512 3301 / 5249 tok/s, pp2048 2776 / 4122; Llama-3.2-1B Q4_K_M
  16027 / 21645 and 13109 / 16829 (llama.cpp `e6d624b`: 15602 / 15529 and 74052 / 82355). Against
  the CPU's f32 prefill (`prefill_gemm=exact`; `examples/pf_check.rs`, raw corpus text, 512 and
  1500 tokens, 32 greedy tokens), the default's first step is within 0.006-0.02 in
  log-probability over the top 20 (the exact GEMM's 1e-5-2e-3; the CPU's own default int8 GEMM's
  0.06-0.09) and its greedy tokens are the exact GEMM's in all four runs. The GEMM time of a
  512-token Qwen3-8B pass: 106 ms with the per-format dequantizer for every format, 98 with it
  reading cached bytes only (so its arithmetic, not memory, was the cost), 55 with the half
  arithmetic unpacking; 64-row tiles 159; the split GEMM 114. Attention in the default mode is
  a tensor-core kernel over the whole step (`dfattn_*`, head dimensions 64 and 128: a block per
  64 queries of a head, key tiles of 64 as f16 in shared memory, f16 products with f32 sums and
  the softmax online in f32; within ~1e-4 of the f64 definition in `cuda::tests`): a 512-token
  Qwen3-8B pass spends 1.5 ms there instead of 18.1 on the decode kernel in batches of 32
  (`gpu_exact=1` keeps that), so its prompts run at 6165 tok/s at 512 tokens and 5432 at 2048,
  Llama-3.2-1B's at 26871 and 21460, with the same greedy tokens as before on the four
  `pf_check` runs. Longer steps take `dfattn2_*` (from 1024 positions at head dimension 128,
  always at 64; `OXINFER_CUDA_FATTN2=0` / `1` forces one): a 256-thread block holds the heads of
  a GQA group (up to 8) times 128 / heads queries, so a key / value tile serves 128 query rows,
  the tiles come in by cp.async while the previous one is in the MMAs, and the longest blocks go
  first. Per layer on an RTX PRO 6000 (32 / 8 heads): Qwen3-8B 512 tokens after 512 cached 56 ->
  46 us, after 1536 110 -> 91, after 3584 218 -> 164, one 2048-token step 226 -> 182;
  Llama-3.2-1B 512 after 1536 58 -> 49, one 2048-token step 121 -> 100 (a 512-token step after
  none at 128: 30 vs 32 us, so `dfattn` stays there). Its outputs are `dfattn`'s bit for bit
  (the same f16 values and tile order; a warp skips only tiles masked out for all its rows):
  `pf_check` over 2048 tokens identical. Qwen3-8B Q4_K_M prompts of 4096 tokens 11773 -> 12038
  tok/s, 2048 12516 -> 12669 (idle GPU). The tiles are still read as f32: with every SM busy a
  tile of 128 rows takes 4.7 us against 3.2 alone, the L2 bandwidth of the f32 keys and values.
  Host time of a step: the new keys and values go into the host caches in one
  parallel copy for every layer (a parallel region per layer and cache took 11-19 ms of a
  512-token Qwen3-8B step, now 3.4), reserved for the whole prompt and the decode steps after it
  (a power of two of positions, on transparent huge pages where Linux gives them on request; the
  first decode step after a 512-token prompt took 25 ms moving 144 MB, now 6.6) and the device
  copies of every layer's cache are one allocation (a growth allocated, zeroed and freed two
  buffers per layer: ~6 ms per later chunk of a long prompt, now 0.5-1.2); the prompt's LM head
  runs on the GPU (`matvec_rows`, f32 activations; the CPU's head rounds them to int8, which moves
  Qwen3-8B's first-step log-probabilities by up to 0.08). Qwen3-8B Q4_K_M prompts then run at
  7588 tok/s at 512 tokens and 7119 at 2048, Llama-3.2-1B Q4_K_M at 32718 and 33050. The fast
  GEMM's tile follows the launch (128 x 128, 64 x 128 or 64 x 64 rows x outputs, the largest
  giving half the SMs a block: the 1024-wide key and value projections take 64 x 64) and two
  128 x 128 blocks share an SM; the logits do not change (every output sums its steps in the
  same order): Qwen3-8B 7800 / 7608 tok/s, Llama-3.2-1B 36839 / 36425. Where those tiles do not
  fill the GPU's blocks evenly (fewer than eight waves and not whole ones: every GEMM of these two
  models at 512 tokens, e.g. Qwen3-8B's 4096-wide projections give 128 tiles to 188 SMs and its
  gate / up 384 to 376 block slots), the GEMM goes by stream-K (`gemmhk_*`; `OXINFER_CUDA_STREAM_K=0`
  / `1` never / always): two blocks per SM, each taking an equal share of all 128 x 128 tiles'
  32-wide steps; a block writes the tiles it began, and the parts of tiles it took from a later
  step go to a scratch tile per block that a second pass (`gemmhk_fix`, sixteen blocks per tile)
  adds in the blocks' order, so the results are the same from run to run. On the shapes alone
  (`ggml_bench`, `BENCH_SHAPES`, best of 7 on a GPU another process was using): Q4_K 12288 x 4096
  329 -> 266 us, 4096 x 12288 352 -> 271, 1024 x 4096 62 -> 41, 4096 x 4096 137 -> 123; the
  second pass, one block per tile summing up to a dozen parts in turn, had cost 117 us of the
  1024 x 4096 GEMM. The GEMM time of a 512-token Qwen3-8B pass falls from 67-79 to 49-69 ms
  (per-op trace), and prompts (best of 5, a shared GPU): Qwen3-8B 7278 -> 8182 tok/s at 512
  tokens and 6590 -> 7151 at 2048, Llama-3.2-1B 35960 -> 43064 and 34802 -> 40273; greedy
  tokens unchanged on both (`pf_check`, 1500 tokens, 32 greedy: the first step within 0.006 of
  the exact GEMM's log-probabilities). An int8 GEMM in the tensor cores (x rounded to int8 per
  block of 32, `mma.m16n8k32.s8`, each block's sums scaled in f32 as llama.cpp's `mmq`) measured
  even with these tiles on Q4_K and up to 8% faster on Q6_K: the scaling after every 32-wide step
  took ~45% of its time (225 against 401 us without it), so it is not kept. The new keys and
  values (151 MB for a 512-token Qwen3-8B step) came down after the step's kernels and into the
  host caches after that, ~7 ms outside them; they now go down per layer as soon as the op
  writing them has run, on a stream of their own into the pinned buffer, and a host thread
  copies each layer's into the room `LayerCache::stage` gave it while the later layers run,
  taken into the caches (`commit`) only once the step has succeeded (a declined step leaves them
  as they were). `OXINFER_CUDA_KV_ASYNC=0` gives the copy after the kernels back; the logits are
  byte for byte the same (`pf_check`, 1500 tokens). On an idle GPU: Qwen3-8B Q4_K_M 8885 ->
  10440 tok/s at 512 tokens and 9011 -> 10099 at 2048, Llama-3.2-1B Q4_K_M 42667 -> 49734 and
  43800 -> 49713. Then (`parity_bench` / llama-bench `e6d624b`, 5 runs each, idle GPU): Qwen3-8B
  pp512 10490 / 15887, pp2048 10099 / 15402, tg128 220 / 252; Llama-3.2-1B 48674 / 76353,
  48751 / 82510, tg128 1300 / 1062. The weights' unpacking was then most of the GEMM's time
  (a build that skips it, `ggml_bench`: Q4_K 12288 x 4096 261 -> 160 us, Q6_K 355 -> 160). Q6_K
  reads its 2-byte aligned blocks as aligned words funnel-shifted and Q4_K its header in one
  16-byte load there too (Q6_K 355 -> 329 us, Q4_K 261 -> 247); and from 256 rows on the
  stream-K GEMM takes 256 x 128 tiles (`gemmhk2_*`: then 16 warps of 64 x 32 outputs, one 512-thread
  block per SM, 60 KB of dynamic shared memory), so each weight value unpacked serves twice the
  rows: Q4_K 12288 x 4096 240 -> 207 us, 4096 x 12288 257 -> 231, 4096 x 4096 90 -> 79; Q6_K
  329 -> 222, 311 -> 214, 120 -> 84 (`OXINFER_CUDA_GEMM_256=0` keeps the 128-row tiles).
  Prompts on an idle GPU (median of 5), 128-row / 256-row tiles: Qwen3-8B Q4_K_M 11242 / 12315
  tok/s at 512 tokens and 10777 / 12043 at 2048, Llama-3.2-1B Q4_K_M 50230 / 56789 and 50713 /
  55369; the first step within 0.005-0.0075 of the exact GEMM's log-probabilities and the
  greedy tokens the same (`pf_check`, 1500 tokens, 32 greedy). Every thread of a block unpacks
  weights (four a row in the 512-thread 256-row kernel and the 64 x 64 one, eight values each:
  half the threads did before; Q4_K 4096 x 12288 231 -> 197 us, Q6_K 222 -> 215 and 214 -> 206),
  and GEMMs of one input (q / k / v, gate / up) convert it to f16 once (the logits do not
  change). Timed in the kernel (`clock64` per phase, a 256-row step of 32 values): the MMAs took
  ~1870 cycles a warp against ~1020 at the tensor cores' rate, the next step's weight loads
  ~490 (L2 latency: the L1 prefetch did not land in time) and their issue ~370, the barrier and
  wait ~20, every warp in step. So the 256-row tile now runs on 8 warps of 64 x 64 outputs (256
  threads, up to 255 registers: a third fewer fragment loads per MMA) and Q4_K (and Q6_K with
  four threads a row) loads the next step's weight bytes into registers before the current
  step's MMAs, unpacking them after: Q4_K 4096 x 12288 196 -> 170 us, 12288 x 4096 210 -> 186,
  4096 x 4096 82 -> 71, the 128-row kernel 222 -> 204; Q6_K unchanged (205 / 216 -> 199 /
  209: its loads early measured no faster with two threads a row); the same sums in the same
  order (`pf_check` logits identical). Qwen3-8B Q4_K_M prompts 13240 -> 13470 tok/s at 512
  tokens and 12669 -> 13234 at 2048, Llama-3.2-1B 58828 -> 61276 at 512 (idle GPU). Tried and
  dropped: three stages of activations for the 256-row tile (92 KB of shared memory, less L1
  for the weights: Q4_K 196 -> 202 us, Q6_K 205 -> 224); the next step's unpacking between
  the two halves of a step's MMAs (no change). RMSNorm, RoPE, the
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
machine. Prefill plans exist for GGUF weights and for bf16 matrices (a safetensors checkpoint
loaded without `--quant`: its rows are the ggml BF16 format's, read by that format's GEMM and
matvec, the prompt's LM head included). oxinfer's Q8 / Q4 (`--quant`) take the fast plans
through the same kernels with their block scales read in place (`K_OQ8`, `K_OQ4`), and the
prompt's LM head with them; with `gpu_exact=1`, with experts, a LoRA or a sandwich norm in the
graph, and for MLX affine Q4, those prompts keep the op-by-op path (measured on a DGX Spark:
[Prompts on the GB10](#prompts-on-the-gb10)).

**Prompts of bf16 models, Gated DeltaNet layers, head dimension 256** (RTX 5080; the numbers and
the commands are under [RTX 5080, CUDA](performance.md#rtx-5080-cuda-november-2026)). A prefill
plan's ops beyond the GEMMs, as they are now:

- **Attention at head dimension 256** (Qwen3.5) on the tensor cores as at 64 and 128
  (`dfattn_256`: key tiles of 32, so that the f16 keys and values of a tile fit a block's 48 KB
  of shared memory); before, those prompts took the decode kernel in batches of 32 tokens.
- **The Gated DeltaNet rule** in four launches: the normed queries and keys (`dgdn_qk_t`) and
  beta / decay (`dgdn_bd_t`) of every token at once, the recurrence alone in order, then the
  output norm and gate of every token (`dgdn_out_t`). The recurrence runs a warp per column of a
  head's state (`dgdn_seqw_HDK`: the lanes across the key dimensions, a column's sums by
  shuffles, no barrier, every SM busy) on the fast plans, whose sums go in another order than the
  decode kernel's (4e-7 of the largest value on random inputs); with `gpu_exact=1` a thread per
  column in the decode kernel's order (`dgdn_seq_HDK`: the same states and outputs bit for bit,
  `cuda::tests::dgdn_rule_split_matches`). `OXINFER_CUDA_GDN_SPLIT=0` gives the one kernel back
  (a block per value head, nine barriers a token), `=exact` the exact order on fast plans.
- **The Gated DeltaNet convolution** with a block row per token (`dgdn_conv_p`: a token's output
  takes the kc - 1 inputs before it and its own; the same products in the same order, logits
  byte for byte; `OXINFER_CUDA_GDN_CONV=0`).
- **Recurrent states** go up for a pass's first chunk only (the later chunks find what the chunk
  before left on the device), and the **KV caches**, host and device, take room for the whole
  prompt at its first chunk (`generate::prompt_ahead`) instead of doubling every few chunks.

**Concurrent requests** (`oxinfer serve --parallel N`). The plain decode steps of the requests
decoding at once are one batched step (`DecodePlan::step_rows`, as on Metal): a multi-token plan's
launches over the rows, the matvecs with every row in one launch (the int8 GGUF ones in one sweep
of the weights, above), the norms, GLUs, element-wise ops, attention inputs and attention in one
launch for all rows too (`dqk_rope_r` / `dattnr_D_G` read each row's cache from a table of
addresses; `OXINFER_CUDA_ROWS_FUSED=0`: a launch per row), and the Gated DeltaNet ops per row on
the row's own device states. Each row gets its one-token step's bits
(`cuda::tests::ggml_plans_match_single_token_steps`: eight sequences at different positions, f32
and int8 matvecs). A Gated DeltaNet model's rows are bound by their recurrent states, which come
down to the host after every step (19 MB a row for Qwen3.5-0.8B).

Of these, the bf16 plans were run on Qwen3.5-0.8B only (other families loaded as bf16 build the
graphs their GGUF files build), and the batched rows on Llama-3.2-1B Q4_K_M, Qwen3.5-0.8B bf16
and mlx-community/Qwen3-4B-4bit.

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
Silicon only. Two other jobs measured speed on the machine at the same time, so nothing in the
first part is a timing; the CUDA backend's speed and memory are
[below](#cuda-speed-and-memory-on-the-gb10), and the CPU's under
[DGX Spark CPU](performance.md#dgx-spark-cpu-november-2026).

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
  Qwen3.5-0.8B bf16 read into memory holds 1.76 GB resident and 1.75 GB on the device (1.75 GB
  resident on the CPU alone). A checkpoint is therefore mapped on such a device, so that the
  host's copy is page cache the system takes back ([below](#cuda-speed-and-memory-on-the-gb10), with the
  figures for larger models); weights that `--quant` makes at load are still held twice.
  `--models` counts two copies, so models up to 60 GB are listed as decoding on the GPU here. The test that a dropped model returns its device memory
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

(The CUDA rows for bf16, `--quant q8` and `--quant q4` are those of the op-by-op prompt path,
which `OXINFER_PREFILL_GPU=0` still runs: by default their prompts of 32 tokens and more now
take the prefill plans, f16 operands as the GGUF plans do. What that moves here is measured in
[Prompts on the GB10](#prompts-on-the-gb10).)

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

### CUDA speed and memory on the GB10

Measured on the same machine while two other jobs took turns on it (load averages 3 to 18): each
number is the median of the three passes of one `examples/parity_bench.rs` run, and the runs of
the builds compared alternate. "Before" is the tree at v0.0.17, without the free-memory figure
above and without what follows. The discrete cards of [CUDA](#cuda) were not measured again:
each change applies only to a device that reports `CU_DEVICE_ATTRIBUTE_INTEGRATED`, and a
discrete card runs the code it ran before. Memory here is in units of 2^30 bytes, as
`/proc/meminfo` counts it: the machine's 131 GB are 121 of them.

**What the driver calls free memory.** `cuMemGetInfo` reports the system's free pages
(`MemFree`), without the page cache: with 87 GB of files cached it reported 37 GB free, and
fourteen 4 GB allocations still succeeded, the cache shrinking to 53 GB as the last ones came
(the driver's figure was 1.4 to 3.6 GB while 73 to 90 GB were available). The backend refused
every weight that did not fit in that figure, so a machine that had read its model files decoded
on the CPU without saying so. With `MemFree` at 13.4 to 21.4 GB of 121 (97 to 99 GB cached),
before / after taking what the system counts as available less its reserve (`cuda::usable_free`,
above):

| | decode tok/s | memory taken | whole run |
|---|---|---|---|
| Qwen3.5-9B `--quant q8` | 13.0 (no decode plan: CPU) / 22.5 | 38.5 / 22.4 GB | 40.3 / 27.6 s |
| Qwen3.8-27B UD-Q3_K_XL | 6.4 / 11.5 | 26.9 / 14.7 GB | 53.0 / 34.6 s |

(the memory and the time include the next change).

**Every weight was held twice.** A checkpoint read into memory and then copied to the device
occupies the same physical memory twice here: the 13.1 GB Qwen3.8-27B GGUF took 25.1 to 26.8 GB
(`MemAvailable` before the run less its minimum during it; 14.5 GB of it the process's own
pages), Qwen3.5-9B `--quant q8` 34.6 to 38.5 GB. On a unified-memory device the checkpoint is
now mapped instead (`Backend::prefers_mapped_weights`; `OXINFER_CUDA_MAP=0` reads it as
before): the file's pages are page cache the system takes back, and the device's copy is the
one that counts. The same runs then take 13.3 to 14.7 GB and 19.1 to 22.4 GB. What `--quant`
makes at load is still held twice, once for the CPU's kernels and once on the device, and
`--models` still counts two copies of every weight on such a device, which is now the upper
bound and not the rule for GGUF and bf16 files.

Copying from a mapping needed one more thing: the driver copies host memory that is not in the
process's page table yet, in pieces that do not start on a page boundary (a tensor of a GGUF
file, a buffer just allocated), at 0.32 GB/s (1 GB of a mapped file in 32 MB pieces: 3.4 s, all
of it user time in the driver; 5.3 GB/s for pieces that start on one), and at 17 GB/s once the
pages are there. A mapped 27B took 28.1 s to build its decode plan that way. Uploads on such a
device now read a byte of every page first, on the thread pool, and results coming down have
their pages written first (the callers' output buffers are fresh allocations). Loading the 27B
GGUF and answering a 568-token prompt with 32 tokens, whole run: 12.4 to 13.8 s read into
memory, 35.4 to 36.1 s mapped without the page pass, 9.1 to 9.9 s mapped with it. The prompts of weights without a prefill plan (bf16, Q8, Q4:
the op-by-op GEMMs, whose outputs come down after every launch) spent 10.7 to 11.8 s of a
benchmark run's 31 to 33 s of device calls in those copies, 2.9 s now:

| Qwen3.5-9B, tok/s | pp512 | pp2048 | decode | memory taken |
|---|---:|---:|---:|---:|
| `--quant q4`, before | 246-274 | 256-300 | 40.2-41.5 | 26.9-34.7 GB |
| `--quant q4` | 353-355 | 345-349 | 42.3 | 8.8-13.7 GB |
| `--quant q8`, before | 277 | 322-347 | 13.1 (CPU) or 22.9 | 34.6-37.5 GB |
| `--quant q8` | 359 | 351 | 23.1 | 19.1-22.0 GB |
| bf16, before | 277 | 357 | 13.5 | 36.0 GB |
| bf16 | 363 | 347 | 13.7 | 17.5-18.2 GB |

(two or three runs each where a range is given.) Qwen/Qwen3.8-27B in bf16 (56 GB of
safetensors, which no 16 GB card holds) with `--quant q4`, before / now (one run / two): 84.6 /
108.5-108.7 tok/s at 512 prompt tokens, 101.7 / 101.1-104.7 at 2048, 13.2 / 13.0-13.4 decoding;
84.0 / 31.4-31.6 GB taken; the benchmark with its load and quantization 241 / 173-186 s.

**Reading the host's memory in place** was measured and not kept. The driver registers host
memory (anonymous or a mapped file, `cuMemHostRegister` with a device mapping) at the host's
own address, so a kernel can read a checkpoint where it is. A kernel reading 2 GB once, a
64 KB window 32 times over, and 256 scattered words per thread:

| memory | read | written | 64 KB window, again and again | scattered reads |
|---|---:|---:|---:|---:|
| `cuMemAlloc` | 220 GB/s | 185 GB/s | 1333 GB/s | 3499 M/s |
| `cuMemAllocHost` (page-locked) | 201-214 | 167-169 | 228 | 16 |
| `cuMemHostAlloc`, write-combined | 219-226 | 184-191 | 230 | 3173 |
| registered, 4 KiB pages | 168-172 | 113 | 329 | 24 |
| registered, transparent huge pages | 154-166 | 110-115 | 374 | 2596 |
| registered mapping of a file | 168 | | | |

Only a device allocation is read again from the GPU's cache, and a GEMM reads each weight tile
once per tile of tokens. Llama-3.2-1B Q4_K_M with its weights registered in place: 1111 tok/s at
512 prompt tokens and 117 decoding, against 5295 and 169 from device allocations; copied into
page-locked host allocations 2309 and 130, write-combined ones 2484 and 148. The Gated DeltaNet
states (149 MB on the 27B, down to the host after every token) in page-locked memory that the
kernels write directly: 2.3 ms a token for the host's copy, against 2.1 ms through the device
copy and its pinned buffer; not kept either.

**GGUF matvecs.** Unlike the formats above, decoding GGUF weights was bound by the kernel: the
27B's 12.2 GB at 11.4 tokens a second are 139 GB/s. On a unified-memory device a lane of the
matvec now takes whole blocks of 32 values and a warp two rows (eight for the IQ2 formats), so
that what a block's values share is decoded once per block and each block of the input is
loaded once for the rows (`MV_ROWS` in `inkling_cuda.cu`; `OXINFER_CUDA_MV_ROWS=N` for every
format, 0 the warp-per-row kernel; discrete cards take one row per warp in 128-thread blocks,
`OXINFER_CUDA_MV_THREADS`, and fuse a gate / up pair: see [GGUF weights](#cuda)). GB/s of weights by format, 17408 x 5120 and 5120 x 17408,
two passes each, before / after: IQ4_XS 156 / 201, IQ4_NL 154 / 204, Q4_0 180 / 222, Q8_0 184 /
207, Q5_K 154 / 180, Q6_K 153 / 172, IQ3_XXS 148 / 167, Q4_K 146 / 161, Q2_K 208 / 221, Q3_K
129 / 136, IQ3_S 148 / 154, IQ2_XXS 155 / 222, IQ2_XS 187 / 226, IQ2_S 186 / 197 (formats not
in the list keep the other kernel: not measured). The sums are taken in another order; the test
`ggml_matvec_and_gemm_match_definition` holds every variant to the same 2e-6 of the summed
magnitudes.

Against llama.cpp's CUDA backend built on the machine (master `94f3af3`,
`llama-bench -ngl 99 -p 512,2048 -n 128 -r 3`) on the same files, `examples/parity_bench.rs`
with `device=cuda lookup=0 draft=none mtp=0`; "before" is the tree these changes started from,
with memory free (it decodes the 27B at 6.4 tok/s with the page cache full, above); tok/s, two
to five runs each:

| | | pp512 | pp2048 | decode | memory taken |
|---|---|---:|---:|---:|---:|
| Llama-3.2-1B Q4_K_M | before | 5029-5484 | 3993-4214 | 153-171 | 1.6-3.6 GB |
| | now | 5173-5243 | 4137-4188 | 177-179 | 1.5-2.5 GB |
| | llama.cpp | 20368-21213 | 21107-22577 | 224-249 | 1.7-2.0 GB |
| Qwen3.8-27B UD-Q3_K_XL | before | 251-260 | 238-241 | 11.0-11.4 | 25.4-26.2 GB |
| | now | 258 | 231-239 | 12.4 | 14.0-17.8 GB |
| | llama.cpp | 994-1001 | 995 | 15.3-15.9 | 13.9-14.2 GB |
| Qwen3.8-27B UD-Q4_K_M (16.6 GB: IQ4_XS, Q5_K, Q4_K, Q6_K; one run, two of now) | before | 279 | 257 | 9.1 | 34.0 GB |
| | now | 278 | 256-257 | 11.1-11.2 | 17.9 GB |
| | llama.cpp | 1012 | 1026 | 13.0 | 17.2 GB |

What is left against llama.cpp is in the kernels, 0.25x on prompts and 0.75 to 0.86x decoding.
A 512-token step of the 27B is 1.92 s, 1.73 s of it the GEMMs (attention 53 ms, the delta rule
79, the activations 27); on Llama 1B 68 of 91 ms (attention 17). `gemmi_*` reaches 12 to 18
TFLOP/s on the 27B's shapes here (IQ4_XS 17-18, Q4_K 16-17, IQ3_XXS 18, Q5_K, Q6_K and Q3_K
12.6 to 14.6, IQ3_S 12), about half of what it does on the RTX 5080. A token of the 27B is 80
ms, 76 of them the matvecs (160 GB/s over the file's mix of formats, IQ3_S and Q3_K the
slowest) and 2.2 the recurrent states' copy.

MiniMax-H3's engine copies its blocks with bindings of its own and reads their pages first in
the same way: one 448x256 clip of 124 frames in 67.5 to 68.7 s with every block on the device,
against up to 160.7 s before ([On CUDA](video.md#on-cuda)).

Agreement with the CPU path, measured here (`examples/qwen_greedy_dump.rs`: three raw prompts
of 1,500 characters fed as one batch, then greedy tokens one at a time, every step's logits
compared; the CPU on its f32 prompt GEMM):

| | last prompt token, KL (largest logit difference) | decode steps, mean KL | greedy tokens identical |
|---|---|---|---|
| Llama-3.2-1B Q4_K_M, 48 tokens | 5.0e-12 to 3.0e-7 (0.005) | 3.1e-4 to 5.2e-4 | 3 of 3 |
| Qwen3.8-27B UD-Q3_K_XL, 32 tokens | 1.4e-12 to 4.2e-7 (0.0045) | 2.3e-4 to 4.3e-4 | 2 of 3 (the third parts at token 10, 0.196 against 0.195) |
| Qwen3.5-9B `--quant q4`, 32 tokens | 3.1e-12 to 2.4e-7 (0.0032) | 5.1e-4 to 2.5e-3 | 3 of 3 |
| Qwen3.5-9B bf16, 32 tokens | 7.4e-12 to 1.7e-11 | 2.8e-12 to 7.9e-12 | 3 of 3 |

`OXINFER_GPU_DECODE_CHECK=1` (every GPU step on the CPU too) over 42 to 48 tokens: mean KL
1.0e-12 on Qwen3.5-9B bf16, 2.0e-4 with `--quant q4` and 2.5e-4 with `--quant q8` (the same top
token at every step of the three), 3.2e-4 on Llama 1B Q4_K_M (42 of 42), 1.0e-4 on the 27B
GGUF (47 of 48) and 1.0e-4 on the bf16 27B with `--quant q4` (48 of 48). The prompts of bf16
and `--quant` weights are not bit-identical to the CPU's here because these are long enough for
the device's attention, which sums in another order (see the rows above).

Against the tree these changes started from, both on CUDA: every logit of the prompt and of 32
greedy steps is bit-identical on Qwen3.5-9B bf16 and `--quant q4` (one prompt each); on the
GGUF files (three prompts of Llama 1B, one of the 27B) the prompt's last-token logits show no
difference (KL 0) and the decode steps differ by the matvec's order of sums, mean KL 1.7e-10 to
1.1e-8, with the same 48 and 32 tokens.

**Inkling-2 Mini** has an engine of its own on CUDA, which copies its weights and results with
driver bindings of its own. On this machine it now reads its weights' pages before the uploads
(Q8_0's 27.3 GB on the device in 1.5 s instead of 88 to 98), writes its outputs' pages before
the downloads (2048-token prompts 5 to 7% faster) and takes whole blocks of 32 per lane in its
decode kernels (one-token steps 2 to 10% faster): Q8_0 at 815-819 / 947-949 / 32.9-33.7 tok/s
(pp512 / pp2048 / tg128) against llama.cpp's 756 / 598 / 30.4, in 28.0 GiB; on the CPU 164 at
512 prompt tokens (`prefill_gemm=fast`) and 19.0 decoding against llama.cpp's 48-51 and 9.4. The four checkpoints measured, the profile, the agreement with
master and with the CPU, and what was not kept are in
[inkling.md](inkling.md#dgx-spark-gb10-unified-memory).

**Not run on this machine:** the affine-Q4 MLX checkpoints, MoE models other than Inkling-2
Mini, GGUF files other than Llama-3.2-1B Q4_K_M / Q8_0, Qwen3.8-27B UD-Q3_K_XL / UD-Q4_K_M and
Inkling-2 Mini's, Inkling-2 Small (MiniMax-H3 ran one clip, above), Whisper, the image and
video pipelines other than SD-Turbo and that clip, speculative decoding, streamed weights and
the server with a model.

### Prompts on the GB10

A second round on the same machine (November 27 and 28), on prompts. It ran beside the round on
the RTX 5080 ([RTX 5080, CUDA](performance.md#rtx-5080-cuda-november-2026)), which gave the
plans bf16 weights, the attention at head dimension 256, the Gated DeltaNet rule in separate
launches and the keys' and values' room for a whole prompt; "master" below is the tree with
that round (`f742f89`), measured here, and "first" the tree both rounds started from
(`b36e8a1`). One GPU job ran at a time (the machine's lock; other agents' CPU jobs ran beside,
load averages 1 to 20). Commands:

```sh
# tok/s at 64 / 512 / 2048 / 8192 prompt tokens and 128 decoded, median of 3 after a warm-up
cargo run --release --example parity_bench -- MODEL device=cuda lookup=0 draft=none mtp=0 \
    pp=64,512,2048,8192 tg=128            # device=cpu for the CPU rows; quant=q4 / q8 as said
llama-bench -m FILE.gguf -ngl 99 -p 64,512,2048,8192 -n 128 -r 3      # llama.cpp master 94f3af3
# where a 512-token step goes: every op synchronized, GPU time per kind of op, host time around
OXINFER_CUDA_DECODE_TRACE=2 cargo run --release --example parity_bench -- MODEL device=cuda pp=512 tg=0 reps=1
```

`MODEL` was the file or folder under `~/.cache/oxinfer/hub` (a repository name asks the hub at
every load, which the site's link did not always answer).

**What this round changed**, each on master's tree:

- *A read past the last Q8_0 block.* Qwen3.8-27B UD-Q4_K_M ran on the CPU with `device=cuda`,
  without a message: 68 / 77 / 65 tok/s at 64 / 512 / 2048 prompt tokens on master (the GPU at
  0%; the CPU alone 68 / 80 / 77, and 7.3 decoding). The int8 matvec of the decode plans loaded
  nine aligned words around a block's 32 bytes, two to four bytes past a Q8_0 block's end; a
  1024 x 5120 Q8_0 matrix is 1360 pages of 4 KiB, the page after it was not mapped, the first
  decode step faulted (`CUDA_ERROR_ILLEGAL_ADDRESS`, which stays with the context) and every
  later step was declined. The ninth word is now the block's last two bytes; a failed copy to
  the device is a recorded warning, and `OXINFER_GPU_PLAN_DEBUG=1` names the first op after
  which the device reports an error. The file then runs at 315 / 784 / 790 / 763 and 11.2.
  (UD-Q3_K_XL has Q8_0 matrices of that size too and did not fault: where an allocation ends
  is the driver's.)
- *A plan for Q8 and Q4 weights* (`--quant`; master's plans take GGUF and bf16). Their prompts
  ran op by op, an f32 GEMM without the tensor cores per matmul and every output through the
  host. Fast plans now take them through the kernels the GGUF formats have (`K_OQ8`, `K_OQ4`:
  the stored values with their block scales read in place, Q4's as the checkpoint's f16).
  `gpu_exact=1`, affine Q4, and graphs with experts, a LoRA or a sandwich norm keep the
  op-by-op path. Qwen3.5-9B `--quant q4` at 64 / 512 / 2048 / 8192 tokens: 344 / 415 / 391 /
  306 -> 910 / 2290 / 2340 / 2246; `--quant q8` 343 / 424 / 401 / 311 -> 716 / 1970 / 2015 /
  1966.
- *The LM head after a Q8 or Q4 prompt* on the device (master's takes bf16 and GGUF): unless
  `gpu_exact`, a matrix of 32768 outputs and more alone; a draft's layers in those formats stay
  on the CPU. `--quant q4` 910 / 2290 / 2340 / 2246 -> 952 / 2355 / 2383 / 2265; `--quant q8`
  716 / 1970 / 2015 / 1966 -> 767 / 2011 / 2065 / 1980.
- *Weights read twice per step* (unified memory only). The stream-K GEMM's 256 x 128 tiles
  read each weight once per 256 rows. For bf16 that read is the bound here (`ggml_bench`, 512
  rows: BF16 17408 x 5120 at 36.6 TFLOP/s where Q4_K reaches 69.2), so matrices of 8 bits a
  weight and more, of 16 MB and more, take 512 x 64 tiles (`gemmhk5_K`: a weight read once per
  512 rows, the activations' tile for every 64 outputs; `OXINFER_CUDA_GEMM_512=0` / `1` never /
  always). `ggml_bench`, 256 x 128 -> 512 x 64, TFLOP/s: BF16 17408 x 5120 36.6 -> 51.1, 5120
  x 17408 38.9 -> 50.6, 8192 x 2048 31.6 -> 43.4; Q6_K 57.6 -> 57.9, 52.8 -> 53.9, 46.0 ->
  49.1; Q4_K 69.2 -> 60.6, 52.7 -> 60.4, 56.2 -> 50.4; IQ4_XS 62.4 -> 51.9; IQ3_S 58.2 -> 49.2
  (the tree before the rebase on master; the kernel is the same). Prompts with it off / by
  default at 512 / 2048 / 8192 tokens: Qwen3.5-9B bf16 1665 / 1710 / 1693 -> 2014 / 2032 /
  2001; `--quant q8` 1992 / 2004 / 1956 -> 2055 / 2095 / 2039; Llama-3.2-1B Q8_0 11917 / 13087
  / 12385 -> 12081 / 13212 / 12457. On every format (`=1`, the earlier tree) the block formats
  lost 5 to 10% (Llama-3.2-1B Q4_K_M 14529 -> 13826 at 512 tokens, Qwen3.5-9B `--quant q4`
  2447 -> 2283, the 27B's UD-Q4_K_M 830 -> 769 and UD-Q3_K_XL 840 -> 759) and Qwen3.5-0.8B's
  bf16 matrices, 13 MB and less, 2%. On an RTX 5080 (shared at the time) `ggml_bench` has the
  tile 4 to 6% behind on Q4_K's six shapes: discrete cards keep 256 x 128.
- *Steps of 512 tokens for every model* (unified memory only). A graph of at most 100,000
  activation floats a token now takes steps of 1024 (`OXINFER_GPU_PLAN_DEBUG=1` prints the
  figure: Llama-3.2-1B 29706, Qwen3.5-0.8B 35879, Qwen3.5-9B 79943, Qwen3.8-27B 109671).
  `OXINFER_CUDA_PREFILL_TOKENS=512` / default, one build, 512 / 1024 / 2048 / 8192 tokens:
  Qwen3.5-0.8B 11335 / 11888 / 12067 / 11513 -> 11467 / 13139 / 13557 / 12754; Llama-3.2-1B
  Q4_K_M 13745 / 13986 / 13895 / 13131 -> 13647 / 14734 / 14900 / 14017; Qwen3.5-9B `--quant
  q4` 2433 / 2466 / 2455 / 2354 -> 2429 / 2462 / 2453 / 2386. The 27B with steps of 512 / 1024
  / 2048 (UD-Q3_K_XL at 2048 tokens, the earlier tree): 847 / 829 / 816, so it keeps 512.
- *The delta rule's recurrence with many value heads* (unified memory only). Master's
  recurrence runs a warp per column of a head's state, 32 x 48 blocks of 128 threads for the
  27B on this GPU's 48 SMs. The rule of a 512-token step, by recurrence kernel, Qwen3.5-0.8B
  (16 value heads, 18 layers) / Qwen3.5-9B (32, 24) / Qwen3.8-27B (48, 48), ms: a warp per
  column 10.8 / 25.7 / 75, a thread per column (`OXINFER_CUDA_GDN_SPLIT=exact`) 19.2 / 28.6 /
  66.8, the one kernel (`=0`) 23.6 / 36.6 / 80. `dgdn_seqb` keeps a block per value head, the
  head's state over 512 threads' registers, three barriers a token and the token's normed
  query and key in shared memory, the next token's rows prefetched, in the thread-per-column
  kernel's order (its states and outputs bit for bit, `dgdn_rule_split_matches`): 15.4 ms on the 9B and 34 on the 27B.
  It is taken here from 32 value heads on (`OXINFER_CUDA_GDN_SPLIT=warp` / `block` takes one
  always; discrete cards keep the warps). With the warps / by default, one build, 512 / 2048 /
  8192 tokens: Qwen3.5-9B `--quant q4` 2358 / 2391 / 2312 -> 2435 / 2491 / 2415; bf16 1988 /
  1993 / 1987 -> 2038 / 2056 / 2009; the 27B's UD-Q3_K_XL 787 / 791 / 765 -> 830 / 844 / 817;
  UD-Q4_K_M 776 / 781 / 762 -> 830 / 836 / 811.

**Where it stands.** tok/s at 64 / 512 / 2048 / 8192 prompt tokens, then decoding 128 tokens;
the CPU rows are oxinfer's (20 threads; `--quant q4` and the 27B: one pass, 32 tokens decoded).

| | | pp64 | pp512 | pp2048 | pp8192 | decode |
|---|---|---:|---:|---:|---:|---:|
| Llama-3.2-1B Q4_K_M | first | 4934 | 14636 | 14412 | 10256 | 220.9 |
| | master | 4957 | 14607 | 14670 | 13412 | 220.7 |
| | now | 5084 | 13834 | 15080 | 14204 | 215.6 |
| | llama.cpp | 8203 | 21039 | 22293 | 20223 | 242.5 |
| | CPU | 1208 | 1384 | 1073 | 526 | 126.6 |
| Qwen3.5-0.8B bf16 | first | 794 | 2168 | 2108 | 1399 | 116.0 |
| | master | 3102 | 11048 | 11806 | 11240 | 119.4 |
| | now | 3123 | 11005 | 12720 | 12144 | 115.6 |
| | CPU | 759 | 849 | 807 | 719 | 73.3 |
| Qwen3.5-9B bf16 | first | 192 | 443 | 414 | 320 | 13.7 |
| | master | 569 | 1650 | 1710 | 1653 | 13.8 |
| | now | 588 | 2062 | 2080 | 2056 | 13.8 |
| Qwen3.5-9B `--quant q4` | first | 342 | 412 | 390 | 303 | 43.2 |
| | master | 344 | 415 | 391 | 306 | 41.9 |
| | now | 965 | 2435 | 2509 | 2416 | 42.3 |
| | CPU | 259 | 295 | 283 | | 25.2 |
| Qwen3.5-9B `--quant q8` | master | 343 | 424 | 401 | 311 | 22.9 |
| | now | 755 | 2142 | 2218 | 2155 | 23.0 |
| Qwen3.8-27B UD-Q4_K_M | master (on the CPU, above) | 68 | 77 | 65 | | |
| | now | 320 | 821 | 835 | 810 | 11.2 |
| | llama.cpp | 552 | 1013 | 1012 | 984 | 13.2 |
| | CPU | 68 | 80 | 77 | | 7.3 |
| Qwen3.8-27B UD-Q3_K_XL | master | 284 | 791 | 794 | 772 | 12.7 |
| | now | 285 | 832 | 845 | 817 | 12.8 |
| | llama.cpp | 552 | 996 | 998 | 968 | 16.1 |
| Inkling-2 Mini AD-Q4_K (its own engine, unchanged; the first tree) | | 238 | 836 | 949 | 879 | 42.2 |
| | llama.cpp (`-fa 1`) | 512 | 812 | 634 | 511 | 36.0 |
| | CPU (`prefill_gemm=fast`) | 84 | 142 | 143 | | 19.8 |

(Master's 27B UD-Q4_K_M run was stopped during the 8192-token prompts. The "now" rows are one
run of the tree that carries all of the above.) Against llama.cpp on the same
files that is 0.62 / 0.66 / 0.68 / 0.70 on Llama-3.2-1B's prompts and 0.89 decoding, 0.58 /
0.81 / 0.83 / 0.82 and 0.85 on the 27B's UD-Q4_K_M, 0.52 / 0.84 / 0.85 / 0.84 and 0.80 on its
UD-Q3_K_XL. Master and this tree run in turn (three rounds
on Llama-3.2-1B, two on Qwen3.5-0.8B, medians of 3): Llama-3.2-1B 13653-13925 / 13888-14012 at
512 tokens, 14284-14538 / 15092-15342 at 2048, 13229-13361 / 14105-14357 at 8192;
Qwen3.5-0.8B 11094-11129 / 10950-10994 at 512, 11730-11930 / 12765-12909 at 2048, 11264-11289
/ 12093-12154 at 8192; at 64 tokens 4841-5282 / 4853-5211 and 3151-3175 / 3089-3094. The
1024-token steps are what moves these two, from 1024 tokens on.

**A 512-token step now** (ms of GPU time per kind of op, each op synchronized; then all of a
prompt of that length in the same run):

| | GEMMs | delta rule | its convolution | attention | activations | norms | RoPE | step | prompt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.2-1B Q4_K_M | 24.0-24.7 | | | 0.8-0.9 | 3.4-3.5 | 1.1 | 0.8-0.9 | 32.0-33.0 | 36.2 |
| Qwen3.5-0.8B bf16 | 22.2-22.3 | 10.3-10.5 | 2.2 | 0.7-0.8 | 1.2-1.3 | 1.0-1.1 | 0.3 | 38.5-38.9 | 44.8 |
| Qwen3.5-9B `--quant q4` | 149-151 | 15.2-15.6 | 3.7-3.8 | 1.9 | 11.1-11.2 | 4.1 | 0.9 | 187-191 | 201 |
| Qwen3.5-9B bf16 | 173-174 | 15.5 | 3.6 | 2.0-2.3 | 11.0 | 4.4-4.5 | 0.9 | 213-217 | 232 |
| Qwen3.8-27B UD-Q4_K_M | 475-478 | 34.1-34.4 | 8.9-9.1 | 5.7-6.0 | 32.1-32.2 | 10.9 | 2.4-2.5 | 572-578 | 600 |
| Qwen3.8-27B UD-Q3_K_XL | 473-474 | 33.8-34.1 | 8.9-9.0 | 5.8-6.1 | 32.1 | 10.9-11.3 | 2.4-2.7 | 565-571 | 592 |

**Agreement** (`examples/pf_check.rs`: raw corpus text cut to 512 and to 1500 tokens, 32 greedy
tokens, the top 20 log-probabilities of the first step, which is the prompt's; against the CPU
with `prefill_gemm=exact`, its f32 prompt GEMM, and the same for `gpu_exact=1`, the exact plans
for GGUF and bf16 weights and the op-by-op path for Q8 / Q4):

| | default against the CPU's f32 prompt: first step, largest difference | 32 tokens | `gpu_exact=1` against the CPU's | 32 tokens |
|---|---|---|---|---|
| Qwen3.5-0.8B bf16 | 4.4e-3 / 3.6e-3 | same / same | 2.8e-5 / 2.5e-5 | same / same |
| Qwen3.5-9B bf16 | 6.7e-3 / 2.9e-3 | same / same | 3.6e-5 / 3.5e-5 | same / same |
| Qwen3.5-9B `--quant q4` | 3.8e-2 / 1.9e-2 | 2 of 32 / same | 5.3e-2 / 2.6e-2 | same / 1 of 32 |
| Llama-3.2-1B Q4_K_M | 2.0e-2 / 3.7e-2 | same / 14 of 32 | 1.6e-2 / 3.5e-2 | same / 5 of 32 |
| Qwen3.8-27B UD-Q4_K_M | 2.9e-2 / 1.9e-2 | same / same | 2.7e-2 / 2.0e-2 | 2 of 32 / same |

Where the 32 tokens part, both paths do so on one prompt or the other: the decode steps after
the prompt round differently on the two devices (f32 activations here, int8 on the CPU for
quantized weights). The 29 CUDA tests of the library (`cargo test --profile ci --lib
cuda::`) pass on the GB10 on this tree, among them the Q8 / Q4 plans against single-token steps
(`quant_weights_prefill_plan_matches_single_token_steps`), their LM head's rows, the 512 x 64
tile in every ggml format and the recurrence kernels against each other.

**Measured and not kept, or left as it was** (the earlier tree, before the rebase).
`OXINFER_CUDA_STREAM_K=0` (Llama-3.2-1B 14442 -> 13805 tok/s at 512 tokens, the 27B UD-Q3_K_XL
844 -> 551), `OXINFER_CUDA_GEMM_256=0` (13381 and 564), `OXINFER_CUDA_KV_ASYNC=0` (13601) and
steps of 256 tokens (12640): the defaults stay. Steps of 2048 tokens against 1024, at 2048 /
8192 prompt tokens: Llama-3.2-1B 16172 against 15208 and 10602 against 10582, Qwen3.5-0.8B
14004 against 13606 and 11141 against 11646; 1024 was taken, with half the arena. The 512 x 64
tile for the 4-bit and 3-bit formats (above). This round's own attention at head dimension
256, delta rule in three launches, convolution per token and bf16 plans were the same work as
master's and gave way to it; of them only the recurrence kernel above came back, where it
measured ahead.

**What is left**, by size:

- *The GEMM kernel.* It is 57 to 83% of a step (the table), at 40 to 69 TFLOP/s on Q4_K by
  shape (`ggml_bench`: 2048 x 2048 40.6, 8192 x 2048 56.2, 2048 x 8192 64.3, 17408 x 5120 69.2;
  the 512 x 2048 key and value projections 25.4). llama.cpp takes 24.3 ms for all of a
  512-token Llama-3.2-1B prompt, these GEMMs alone 24 ms. That kernel is the same on a
  discrete card and was not changed here.
- *Short prompts.* A 64-token step reads and unpacks every weight as a 512-token one does:
  Llama-3.2-1B 9.5 of its 10.7 ms in the GEMMs (0.6 of llama.cpp at 64 tokens), and Inkling-2
  Mini 0.26 s for 64 tokens against 0.61 s for 512, its experts' `gemmi_12` 86 ms of a 64-token
  pass and 205 ms of a 512-token one (238 tok/s against llama.cpp's 512, where it leads from
  512 tokens on).
- *The activations' passes.* SiLU(gate) x up reads two f32 rows and writes one and its f16
  copy: 3.4 ms of Llama-3.2-1B's 32 (0.2 ms a layer over rows of 16.8 MB), 11 of the 9B's
  189, 32 of the 27B's 575; the f32 row is written whether or not anything reads
  it again.
- *The host around a step*: the table's last two columns; the 27B's recurrent states take 6 ms
  of a step up and down.
- *Decoding* is where it was: 0.79 to 0.91 of llama.cpp, the matvecs at what the section above
  measured.
