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
| Apple Silicon (macOS, iOS) | NEON kernels + Metal GEMM for prefill; on M5 / A19 the GPU matrix units (prefill GEMMs and MoE experts); single-token decode entirely on the GPU (Llama 3, Qwen3 / 3.5 / 3.8 incl. MoE, Gemma 3 / 4 incl. 26B-A4B) | measured; the f32 GEMM gives bit-identical logits to the CPU path, the matrix units (default where present) ~3e-4 relative per product (`OXINFER_METAL_MMA=0` for bit-identical); iOS: runs in the Simulator (CPU and Metal; `ios/`) |
| Apple Neural Engine (macOS 15 / iOS 18) | Core ML, for ANEMLL-converted models | measured on macOS; builds for iOS; see [Core ML](#core-ml--apple-neural-engine) |
| x86-64 (Linux, Windows, macOS) | AVX2 + FMA + F16C kernels chosen at run time: bf16 / f32 dot tiles and prefill GEMM, Q8 via `maddubs`, Q4, every GGUF format, decode attention; scalar fallback below AVX2 or with `OXINFER_SIMD=scalar` (`--devices` prints the set in use). No AVX-512 / VNNI kernels; Q4's multi-token column kernel and LoRA's dot products are NEON-only (x86 runs the token-pair and portable loops) | executed on Linux on one Zen 5 CPU (Threadripper 9970X); Windows and macOS compile only. Every AVX2 kernel is tested directly against its scalar fallback (bit for bit for the integer parts, GGUF rows and attention; 1e-6 of the summed magnitudes for the f32 sums). Qwen3.5-0.8B on 5 prompts, AVX2 against scalar: last-prompt-token logits within 4.8e-5 (bf16, Q8, Q4), same top 10; 64 greedy tokens identical in 14/15 runs, and bf16 within 5e-5 of transformers float32 with the same argmax (5/5). Q8 decode steps differ between the two by 0.4-0.6 in the largest logit (same top 1 on 317/320 teacher-forced steps), as far as either is from bf16: the integer block sums are identical (a build with the AVX2 integer path and the scalar order of f32 additions gives the scalar logits bit for bit over 169 steps), the f32 additions are ordered differently, and re-quantizing activations to int8 at every matmul amplifies that. Llama-3.2-1B GGUF (Q4_K_M, Q8_0): decode-kernel passes (`--prefill-chunk 1`) give bit-identical logits and greedy text to the portable path (10/10); with the f32 prefill GEMM the last hidden state differs by rounding (up to 1.8e-4 on values up to 17) and the logits by either ~2e-5 or 1.6e-3 to 5.7e-3: the LM head of the last token is a one-token pass on int8 activations, and the larger figure appears exactly when that rounding moves one to three of the 2048 int8 activations by one step (7/16 pseudo-random prompts on Q4_K_M, 5/16 on Q8_0; an f32 LM head pass on the same states stays within 1e-4 on all 32); on the 5 chat prompts 4.8e-3 / 3.5e-5 at most, 48 greedy tokens identical in 3/5 and 5/5 |
| Linux / Windows + NVIDIA | CUDA driver API + NVRTC-compiled kernels (`cuda.rs`): prefill GEMMs (bf16, Q8, Q4, affine Q4), MoE experts (resident or streamed), convolutions, group / layer norms, attention; whole decode steps, single-token and multi-token (speculative verification); loaded as a native binary, so a toolkit newer than the driver works | measured on Linux on one RTX 5080 (NVRTC 13.4, driver CUDA 13.2), see [CUDA](#cuda); Windows and other GPUs not run |
| Android (aarch64) | NEON kernels | cross-compiles (`aarch64-linux-android`) |
| Vulkan (any GPU with a Vulkan 1.0 driver) | compute shaders through the Vulkan loader (`vulkan.rs`, SPIR-V embedded in the binary): prefill GEMMs only (bf16, Q8, Q4, affine Q4); MoE experts, convolutions, norms, attention and decoding stay on the CPU | measured on Linux on one RTX 5080 (NVIDIA driver, Vulkan 1.4): logits bit-identical to the CPU path and the same greedy text for the four weight formats, see [Vulkan](#vulkan); integrated GPUs, other vendors, Windows, Android and MoltenVK not run |
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
dequantized per 64-wide K step into threadgroup memory as half, bf16 weights are read as bfloat
straight from device memory, and accumulation is f32: every format with a GEMM (bf16, Q8, Q4,
the GGUF types, FP8 E4M3 with 128x128 block scales) has a matrix-unit version. Kernel throughput at 2048 tokens is ~60 TFLOP/s for
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
row so a SIMD-group reads contiguous memory, nibbles masked rather than shifted (the powers of
two are divided out once per word), K-quant scales decoded without branches; 450-590 GB/s on
large matrices of an M5 Max (`cargo test --release --lib decode_mv_bench -- --ignored`).
Attention scores 32 keys per SIMD-group step (one lane per key), reads each key and value row
once for the query heads sharing it, and splits long contexts over up to 64 threadgroups per head
group. Activations stay f32 end to end (the CPU kernels quantize them to int8 for the quantized
formats), and so do the KV caches: the host caches stay authoritative and the device keeps a
copy per session, appending what CPU passes added (prefill chunks, speculative verification) and
forgetting what a rewind dropped. Not on the GPU: models with LoRA adapters attached, streamed
weights (`--stream`), bidirectional configurations, the vision / audio towers,
multi-token passes (prefill and speculative verification stay on the existing paths), and
WebGPU (no decode graph backend yet). CUDA runs the same graphs, see [CUDA](#cuda).

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

**Decoding.** Whole decode steps run on the GPU as compiled plans of the decode graph
(`cuda_decode.rs`, the kernels ported from Metal's): matvecs (bf16, Q8, Q4, affine Q4), norms,
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

Declined (decoding stays on the CPU): interventions, Qwen4-Exp, GGUF weights, and models whose
weights do not all fit in VRAM: Gemma 4 26B-A4B's experts do not on 16 GB
(`OXINFER_GPU_PLAN_DEBUG=1` names the op). Recurrent states go up and down every step (copied back
on a second stream under the remaining layers; the upload is skipped when nothing on the host
wrote them).
The image-model timings include loading and quantizing the weights.

`cargo test --lib cuda::` compares every kernel with a reference computed in f64 from its
definition; it fails if a kernel does not compile or load, and skips without an NVIDIA GPU or NVRTC.

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

No other device or driver was run, and no speed was measured.

`cargo test --lib vulkan::` runs the kernels in both memory modes against the exact (f64)
product of the dequantized weights, the same product in the kernel's order of operations and the
CPU kernels, for each format; the f16 scale decoding on every finite value; batches of several
outputs; weights split over several buffers and batches split over token ranges (the limits
lowered, results bit-identical to the unsplit path); declined shapes; freed memory; concurrent
calls. `bit_identity_report` prints how many outputs differ from the fma-ordered reference on the
machine at hand (none here). The tests skip without a Vulkan device.
## WebAssembly

`.cargo/config.toml` turns on `simd128` for the wasm targets (every current engine supports it;
without it the scalar kernels are used, ~8x slower on Q8 decode) and raises the shared-memory
maximum for the threads target to 4 GiB. The model directory is passed in as a preopened directory:

```sh
# single-threaded, runs under Node's built-in WASI (node --experimental-wasi ... on older versions)
cargo build --release --target wasm32-wasip1 --no-default-features
# threads (wasi-threads): wasmtime up to v30 runs it; v31+ dropped the wasi-threads proposal
cargo build --release --target wasm32-wasip1-threads --no-default-features
wasmtime run -W threads=y -S threads=y --dir ~/.cache/oxinfer/hub/Qwen--Qwen3.5-0.8B::/model \
  target/wasm32-wasip1-threads/release/oxinfer.wasm --model /model --quant q8 -j 16 -p "Hello"
```

Qwen3.5-0.8B Q8, 18-core M5 Max, 17-token prompt:

| Build | Threads | Prefill | Decode |
|---|---|---|---|
| wasm32-wasip1, scalar | 1 | 9 tok/s | 4.6 tok/s |
| wasm32-wasip1, simd128 | 1 | 29 tok/s | 40 tok/s |
| wasm32-wasip1-threads, simd128 | 16 | 327 tok/s | 123 tok/s |
| native (NEON) | 18 | 373 tok/s | 184 tok/s |

Memory is 32-bit, so a model plus its quantized copy must fit in 4 GiB (`--quant q4` on a 2B
model is fine; the 27B models are not). No file mapping exists in WASI, so the checkpoint is
always read into memory.

## Browser (WebGPU)

The site has a Chat mode and an Image mode. Image mode loads SD 1.5 with the LCM-LoRA by default
(2.1 GB of fp16 weights plus 135 MB from Hugging Face; 4 Euler ancestral steps without guidance,
about 8 s per 512x512 image in Chrome on the M5 Max with the machine busy, 63 dB from the native
Metal image); SD-Turbo (2.6 GB, one step) is the other preset, and a custom model can take LoRA
files too (`?lora=org/name/file.safetensors:0.8`, comma separated). With the LCM-LoRA, plain Euler
leaves speckle noise in the image: LCM is trained to be re-noised between steps, as the ancestral
sampler does. A third preset, InstructPix2Pix (with the same LCM-LoRA, 4 steps), edits an
uploaded picture by instruction, and every model takes an input image and a mask (image-to-image,
inpainting; see [Editing images](stable-diffusion.md#editing-images)); headless Chrome produced the same edit as the
native build. Diffusion weights stay in JavaScript memory and the module converts
them tensor by tensor, so only the bf16 copy counts against the 4 GiB of wasm memory.

The `web` feature (`src/web/`) turns the library into a plain C-ABI wasm module, no bindgen;
`cargo build-web` builds it as a `cdylib` for `wasm32-unknown-unknown` (the crate type is given on
the command line, so native builds never link a shared library). `web/` is the site deployed to
[GitHub Pages](https://karpeleslab.github.io/oxinfer/) by `.github/workflows/pages.yml`: a Vue 3 +
Vite app around the JavaScript host `web/src/lib/oxinfer.js`; it is not a crate. Downloads go to
the browser's Cache Storage, so a model is fetched once per browser. MLX checkpoints load too
(Gemma 4 E2B and E4B, Gemma 3 4B and Qwen3-4B from the mirror were checked in headless Chrome):
the host splits each one into shards of at most 1 GiB kept in JavaScript memory as it streams in,
since a 5 GB file fits neither one allocation nor a 32-bit module, and the module converts the
matrices into its own memory, then lets the host drop every shard but the embedding tables. The host registers the model files in the module's in-memory filesystem (streamed straight from
`fetch` or a local folder into module memory) and drives a chat session token by token.

GPU work goes the other way round: the module imports `gpu_upload` / `gpu_matmul`, which the
host implements with WebGPU (WGSL GEMM, 64x64 tile per workgroup, weights dequantized to f32 in
workgroup memory; ~3.9 TFLOP/s at 4096^3 on the M5 Max, ~0.7 ms per round trip). The module's
matmul call is synchronous while the buffer readback is asynchronous, so `gpu_matmul` is wrapped
in `WebAssembly.Suspending` and the generating exports in `WebAssembly.promising` (JavaScript
Promise Integration, Chrome 137+). Without JSPI or WebGPU the same page runs on the CPU.
WebGL is not used: it has no compute shaders.

```sh
cd web
npm install
npm run wasm                                       # cargo build-web, copied to public/oxinfer.wasm
npm run smoke                                      # the JS <-> wasm bridge, headless (Node)
OXINFER_MODELS=~/.cache/oxinfer/hub npm run dev    # local folders load as "./models/<folder>"
# or from Deno directly (Deno has WebGPU, but its readback polls every ~14 ms, so --cpu is as fast there):
deno run --allow-read --allow-env scripts/run.mjs --model ../models/gemma-4-E4B-it-q4 --cpu -p "Hello"
```

Point the page (or `run.mjs`) at a pre-quantized checkpoint (`--save-quantized`, see [Pre-quantized checkpoints](performance.md#pre-quantized-checkpoints)): it
downloads only the quantized bytes and uses them in place, and its `model-tables-*` shards stay in
JavaScript memory, read one row per token. That is what makes the larger models fit: E4B at Q4
puts 2.6 GB in wasm memory and 1.6 GB of per-layer embeddings in JavaScript. Plain bf16
checkpoints still work for small models (they are quantized in the module, which then holds both
copies).

Headless Chrome 155, Qwen3.5-0.8B Q8, 123-token prompt, single CPU thread (the browser build has
no threads):

| | Prefill | Decode |
|---|---|---|
| CPU (simd128) | 34 tok/s | 36 tok/s |
| WebGPU prefill, first turn (includes the weight upload) | 165 tok/s | 38 tok/s |
| WebGPU prefill, steady state | 231 tok/s | 38 tok/s |

Gemma 4 from pre-quantized Q4 files, same page, same 115-token prompt (second turn, weights
already on the GPU); the output text is identical with and without WebGPU, and 40 greedy tokens
from the wasm build match the native binary on the same file:

| Model | CPU prefill | WebGPU prefill | Decode |
|---|---|---|---|
| E2B-it Q4 | 10 tok/s | 124 tok/s | 7.7 tok/s |
| E4B-it Q4 | 5 tok/s | 125 tok/s | 4.0 tok/s |

Limits: decode is single-threaded CPU and bandwidth-bound (one core streams ~10 GB/s of Q4
weights), and stays off the GPU for the same reasons as on Metal. Weights in module memory must
fit in 4 GiB, and no single file can exceed 2 GiB. `web/scripts/test_gemm.mjs` checks the WGSL kernels
against a JavaScript reference.

Feature flags: `media` (oxideav image/audio/video file decoding), `hf` (HuggingFace fetching),
`gpu` (dlsym-loaded backends), all on by default; `oxideav` (oxideav-core types in the API, implied
by `media`; see [Using oxinfer from oxideav](cli.md#using-oxinfer-from-oxideav-and-other-crates)), `capi`,
and `web` (the browser module, off by default and only meaningful for `wasm32-unknown-unknown`);
`--no-default-features` gives the plain CPU build.
Cross-compiling needs no C toolchain: the tokenizer uses its pure-Rust regex backend.

## Core ML / Apple Neural Engine

Models converted by [ANEMLL](https://github.com/Anemll/Anemll) (the `anemll/*` repos on Hugging
Face: Qwen3, Qwen2.5, Gemma 3, Llama 3.2, VibeThinker, ...) are pre-compiled Core ML packages with
fixed shapes and a stateful KV cache, laid out for the Neural Engine. oxinfer runs them through
Core ML itself (`coreml.rs`: `CoreML.framework` via the Objective-C runtime and `dlsym`, nothing
linked), following ANEMLL's runtime contract (`anemll.rs`: `meta.yaml`, chunked / monolithic /
split-rotate layouts, batch prefill, causal masks, Gemma 3 sliding-window rotate functions, split
LM head or argmax in the model), and the extensions of oxinfer's own converter (below): Qwen3.5
recurrent states, Gemma 4 per-layer inputs and two caches. Needs macOS 15 / iOS 18 on Apple
Silicon (stateful and multifunction models); the module also builds for `aarch64-apple-ios`, and
`oxinfer::engine::Engine` loads these models like any other.

```sh
oxinfer --model anemll/anemll-Qwen-Qwen3-0.6B-ctx512_0.3.4 --prompt "Why is the sky blue?"
oxinfer --model anemll/anemll-google-gemma-3-1b-it-ctx4096_0.3.5 --chat
oxinfer --model DIR --coreml-units cpu      # ane (default: CPU + Neural Engine) | all | gpu | cpu
oxinfer --model DIR --coreml-prefill pad    # auto (default) | batch | pad | single
```

The whole repository is fetched (the `.mlmodelc` directories, tokenizer, `meta.yaml`); the first
load of a model compiles it for the device (30 s for Qwen3-0.6B, 2 min for Gemma-3-1B with its four
functions), later loads take 0.3-1 s. Prompt formatting, sampling, `--chat`, `--think`, `--system`,
`--raw`, `--max-tokens`, `--dump-logits` and `--print-prompt` work as with the native paths, but
models converted with `argmax_in_model` (the Gemma 3 and Llama repos above) return only their top
token: generation is greedy and there are no logits to dump. The context is fixed at conversion
(`ctx512` etc.); a longer prompt is an error and generation stops when the cache is full. The same
API is in the library (`oxinfer::anemll::AnemllModel::{load, prefill, step}`, see
`examples/anemll.rs`).

**Verified** against ANEMLL's own runtime (`tests/chat.py` with coremltools 9): same prompt tokens,
same greedy tokens, and bit-identical logits where the model has them:

| Repo (anemll/...) | Layout | Checked |
|---|---|---|
| anemll-Qwen-Qwen3-0.6B-ctx512_0.3.4 | 1 chunk | 60 + 100 tokens, logits identical, both prefill modes |
| anemll-Qwen-Qwen3-1.7B-ctx2048_0.3.5 | 2 chunks | 300 tokens, logits identical |
| anemll-WeiboAI-VibeThinker-1.5B-ctx2048_0.3.5 | 4 chunks (Qwen2.5) | 300 tokens, logits identical |
| anemll-google-gemma-3-1b-it-ctx4096_0.3.5 | 1 chunk, sliding window 512, argmax | 901-token prompt + 3150 tokens (to 4051 of 4096, across the window) |
| anemll-meta-llama-Llama-3.2-1B-Instruct-ctx1024_0.3.5 | 1 chunk, argmax | 300 tokens |
| anemll-Qwen-Qwen2.5-0.5B-Instruct-ctx2048-monolithic_0.3.5 | monolithic | 300 tokens, logits identical, both prefill modes |
| anemll-google-gemma-3-270m-it-ctx512-monolithic_0.3.5 | monolithic, argmax | 250 tokens, both prefill modes |

Prefill modes: full batches (64 tokens) go through the `prefill` functions and the rest through
`infer`, token by token, as in ANEMLL. `auto` does this only when `meta.yaml` declares
`prefill_dynamic_slice` or `update_mask_prefill` and otherwise feeds every token through `infer`, as
`chat.py` does: the batch prefill of older conversions can be wrong (the Gemma 3 270m monolithic
repo above produces garbage with it, in `chat.py` too, while Qwen3-0.6B 0.3.4 gives bit-identical
logits either way). `pad` also runs the last partial batch through `prefill`, padded: the
prediction is not bit-identical any more (greedy outputs diverged after 70 and 162 tokens on Llama
and Qwen2.5, stayed identical on Qwen3 and Gemma 3) but short prompts no longer pay one Neural
Engine call per token.

Compute plans (`MLComputePlan`) put 99.7-100% of the estimated cost of the `infer` functions on
the Neural Engine (2333 of 2340 ops for Qwen3-0.6B; the embedding lookups run on the CPU, and the
Gemma 3 argmax head is half CPU).

Measured on the M5 Max (macOS 26), 227-250-token prompt, 200 generated tokens, best of 2-3 runs
(the machine was shared):

| Model (Neural Engine) | Prefill (auto) | Prefill (pad) | Decode | Decode, back to back with `chat.py` | RSS | CPU time |
|---|---|---|---|---|---|---|
| Qwen3-0.6B ctx512 | 128 tok/s (715 batched) | 3630 tok/s | 106 tok/s | 114 vs 104 tok/s | 205 MB | 0.5 s |
| Gemma-3-1B ctx4096 | 483 tok/s (3250 at 901 tokens) | 2720 tok/s | 69 tok/s | 70 vs 64 tok/s | 630 MB | 1.5 s |
| Llama-3.2-1B ctx1024 | 284 tok/s | 2270 tok/s | 60 tok/s | 52 vs 51 tok/s | 170 MB | 0.5 s |
| Qwen2.5-0.5B monolithic ctx2048 | 358 tok/s | 2120 tok/s | 90 tok/s | 77 vs 75 tok/s | 170 MB | 0.5 s |
| Gemma-3-270m monolithic ctx512 | 261 tok/s | (wrong output) | 263 tok/s | 233 vs 221 tok/s | 560 MB | 0.8 s |

Decode slows down as the context fills (Gemma-3-1B: 67 tok/s averaged over 4050 tokens). Other
compute units on the same Core ML models, and oxinfer's own kernels on the original checkpoints
(`Qwen/Qwen3-0.6B`, `unsloth/gemma-3-1b-it` = google/gemma-3-1b-it):

| | Qwen3-0.6B prefill | decode | RSS | Gemma-3-1B prefill | decode | RSS |
|---|---|---|---|---|---|---|
| Core ML `ane` (CPU + Neural Engine) | 715 tok/s | 106 tok/s | 0.2 GB | 483 tok/s | 70 tok/s | 0.6 GB |
| Core ML `cpu` | 329 tok/s | 48 tok/s | 2.2 GB | 207 tok/s | 22 tok/s | 3.9 GB |
| Core ML `gpu` (CPU + GPU) | 28 tok/s | 21 tok/s | 8.6 GB | 18 tok/s | 14 tok/s | 14.2 GB |
| Core ML `all` | 26 tok/s | 21 tok/s | 8.6 GB | 15 tok/s | 17 tok/s | 14.2 GB |
| oxinfer CPU bf16 / q8 / q4 | 817 / 783 / 751 tok/s | 135 / 174 / 265 tok/s | 1.7-2.4 GB | 515 / 532 / 477 tok/s | 79 / 146 / 216 tok/s | 2.4-3.5 GB |
| oxinfer Metal prefill, bf16 / q4 | 891 / 901 tok/s | 137 / 259 tok/s | 1.7-2.1 GB | 782 / 760 tok/s | 82 / 228 tok/s | 2.4-3.0 GB |

On this chip the CPU cores are faster than the Neural Engine for these small models (and the
ANEMLL conversions use 6-bit palettized weights, so quality is not equal either). What the Neural
Engine buys is everything else: the generation above takes 0.5 s of CPU time against 13.6 s for
oxinfer's Q4 CPU path (1.3 s wall on 18 threads), the CPU and GPU stay free, and the process stays
at 0.2-0.7 GB resident, less than the weights themselves (0.8 GB for Qwen3-0.6B with its fp16
embedding table): the Neural Engine's copy of the compiled model is not counted in the process.
The `gpu` and `all` units expand the palettized weights (8-14 GB resident) and are slow; use the
native Metal path for the GPU instead. Over a 4050-token Gemma-3-1B generation the RSS stays flat
(680 MB at the start, 570 MB at the end): every prediction runs in its own autorelease pool and the
input / output arrays are allocated once per model.

Speed comes from wiring the parts together once at load: each part's output (an IOSurface-backed
fp16 `MLMultiArray`, what the Neural Engine writes directly) is bound as the next part's input in a
prepared feature provider, so a token only writes its id, position and mask row into existing
arrays and runs the chain; decode runs 2-10% faster than `chat.py` on the same models.

### Converting models

`tools/coreml_convert.py` converts Hugging Face checkpoints into Core ML packages for the
Neural Engine of iPhones and Macs, in the layout of [ANEMLL](https://github.com/Anemll/Anemll)
(meta.yaml, an embeddings model, decoder chunks with `infer` and `prefill` functions sharing
weights and a Core ML state, a 16-way split LM head, the tokenizer files). Qwen3 packages run
unchanged under ANEMLL's own `chat.py`; Qwen3.5 / 3.6 / 3.8 models (Gated DeltaNet recurrent
state kept as Core ML states, chunked delta rule in the prefill function; MoE variants evaluated
densely, for small models) and Gemma 4 E2B / E4B (per-layer embeddings, KV sharing, local and
global caches) extend the format as described in [docs/coreml-format.md](coreml-format.md);
oxinfer's Core ML runtime above runs them. `tools/coreml_check.py` runs a package with coremltools,
compares it with transformers in float32 over a few hundred teacher-forced tokens, benchmarks it
and lists which ops land on the ANE.

```sh
python3 -m venv venv && venv/bin/pip install coremltools==9.0 torch transformers safetensors \
    scikit-learn pyyaml "numpy<2.3"
venv/bin/python tools/coreml_convert.py --model ~/.cache/oxinfer/hub/Qwen--Qwen3.5-0.8B --output out/qwen35
venv/bin/python tools/coreml_convert.py --model ~/.cache/oxinfer/hub/Qwen--Qwen3.5-0.8B --output out/qwen35-small \
    --ffn-quant lut4 --lut-mix 4.5        # 25% smaller decoder
venv/bin/python tools/coreml_check.py out/qwen35 --hf ~/.cache/oxinfer/hub/Qwen--Qwen3.5-0.8B --bench --plan
```

**Weights.** Only LUT palettization runs correctly on the ANE (Core ML has LUTs of 1, 2, 3, 4, 6
and 8 bits, no 5), 8 output channels per LUT. Core ML's own palettizer fits each LUT with plain
k-means, which costs 15-20 points of top-1 agreement at 4 bits on sub-1B models. The converter
instead computes the LUTs from a calibration set by default (`--lut-method fisher`,
`tools/coreml_quant.py`): the float32 transformers model answers 32 built-in chat prompts
(under a minute on MPS), the squared gradients of the next-token loss give the Fisher information of
every weight, and each LUT is a k-means weighted by it (SqueezeLLM's sensitivity), so the levels
go where the loss is sensitive. `--lut-mix B` adds mixed precision: the weights whose 4-bit
palettization costs the most (estimated 0.5 Σ F Δw²) get 6-bit LUTs until the decoder averages B
bits. Things that did not pay off: GPTQ-style error feedback over the LUT grid (the per-layer
error drops 4x on the calibration data but rises on other text: it overfits a few thousand
tokens), per-channel scales on top of the LUT, 3x more calibration data. One LUT per output
channel is more accurate (Qwen3-0.6B LUT4: 85% top-1) but halves the ANE's decode speed.

Measured on an M5 Max with coremltools (context 512, batch 64; quality = the Core ML logits
against transformers float32 over 300 teacher-forced tokens of a 135-token chat prompt; speed =
best of 5, 256-token prefill and 64-token decode including the Python calls, on a shared
machine; package sizes are embeddings + decoder chunks + LM head):

| Model | Weights (decoder, LUT fit) | Size (MB) | Logit cosine | Top-1 agreement | KL | Prefill ANE / CPU | Decode ANE / CPU |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | fp16 | 311 + 884 + 311 | 0.99999 | 98.7% | | 2680 tok/s | 79 tok/s |
| Qwen3-0.6B | LUT6, k-means | 311 + 338 + 119 | 0.9972 | 93.0% | | 5131 / 1949 tok/s | 114 / 56 tok/s |
| Qwen3-0.6B | LUT6, Fisher (default) | 311 + 338 + 119 | 0.9984 | 93.7% | 0.016 | 5170 tok/s | 116 tok/s |
| Qwen3-0.6B, ANEMLL's release | LUT6 | 311 + 339 + 119 | 0.9971 | 92.3% | | 4409 / 1824 tok/s | 114 / 59 tok/s |
| Qwen3-0.6B | LUT4, k-means | 311 + 224 + 119 | 0.954 | 72.0% | | 4836 tok/s | 103 tok/s |
| Qwen3-0.6B | LUT4, Fisher | 311 + 224 + 119 | 0.9665 | 79.7% | 0.224 | 5219 tok/s | 116 tok/s |
| Qwen3-0.6B | LUT4, Fisher, mix 4.5 bits | 311 + 253 + 119 | 0.9813 | 84.7% | 0.153 | 5218 tok/s | 116 tok/s |
| Qwen3-0.6B | LUT4, Fisher, mix 5 bits | 311 + 281 + 119 | 0.9876 | 86.3% | 0.085 | 5229 tok/s | 116 tok/s |
| Qwen3-0.6B | LUT4, Fisher, 1 channel per LUT | 311 + 234 + 119 | 0.9850 | 85.3% | 0.141 | 2565 tok/s | 47 tok/s |
| Qwen3.5-0.8B | fp16 | 509 + 1000 + 509 | 0.9998 | 97.0% | | 4002 tok/s | 62 tok/s |
| Qwen3.5-0.8B | LUT6, k-means | 509 + 385 + 195 | 0.9974 | 93.0% | | 5176 / 1608 tok/s | 118 / 37 tok/s |
| Qwen3.5-0.8B | LUT6, Fisher (default) | 509 + 384 + 195 | 0.9974 | 94.7% | 0.011 | 5319 tok/s | 136 tok/s |
| Qwen3.5-0.8B | LUT4, k-means | 509 + 254 + 195 | 0.959 | 78.3% | | 5035 tok/s | 113 tok/s |
| Qwen3.5-0.8B | LUT4, Fisher | 509 + 255 + 195 | 0.9654 | 83.0% | 0.154 | 5450 tok/s | 136 tok/s |
| Qwen3.5-0.8B | LUT4, Fisher, mix 4.5 bits | 509 + 286 + 195 | 0.9823 | 85.3% | 0.083 | 5406 tok/s | 136 tok/s |
| Qwen3.5-0.8B | LUT4, Fisher, mix 5 bits | 509 + 319 + 195 | 0.9894 | 89.0% | 0.053 | 5356 tok/s | 137 tok/s |
| Gemma 4 E2B | fp16 | 5530 + 3740 + 805 | 1.00000 | 99.3% | | | |
| Gemma 4 E2B | LUT6, k-means | 5530 + 1419 + 306 | 0.99986 | 97.0% | | 1905 / 623 tok/s | 45 / 14 tok/s |
| Gemma 4 E2B | LUT6, Fisher (default) | 5530 + 1419 + 306 | 0.99987 | 97.7% | 0.010 | 1976 tok/s | 46 tok/s |
| Gemma 4 E2B | LUT4, k-means | 5530 + 942 + 306 | 0.99771 | 83.7% | 0.282 | 2010 tok/s | 47 tok/s |
| Gemma 4 E2B | LUT4, Fisher | 5530 + 942 + 306 | 0.99799 | 84.7% | 0.226 | 2023 tok/s | 47 tok/s |
| Gemma 4 E2B | LUT4, Fisher, mix 4.5 bits | 5530 + 1061 + 306 | 0.99888 | 91.0% | 0.068 | 2010 tok/s | 47 tok/s |
| Gemma 4 E4B | LUT6, k-means | 7034 + 2992 + 508 | 0.99945 | 96.7% | | 1060 / 333 tok/s | 25 / 6 tok/s |

The LM head stays LUT6 in every row (LUT4 there costs another 3-6 points). The Fisher packages of
Qwen3.5 decode faster than the k-means ones because the small depthwise convolution of the Gated
DeltaNet stays fp16 instead of being palettized. Everything
but the embedding lookups and the RoPE-table gathers runs on the ANE (MLComputePlan), including
the Gated DeltaNet recurrence and its chunked prefill. The Gemma embeddings model is large
because the per-layer embedding table stays fp16: Core ML memory-maps it and pages in only the
rows it reads, while a palettized table would be decompressed at load time (16.7 GB for E2B).
"CPU" is Core ML's CPU_ONLY path, whose outputs are NaN for these stateful fp16 graphs (ANEMLL's
packages too), so it only shows what the ANE saves. On a Max-class Mac, oxinfer's own CPU path
decodes faster (Qwen3.5-0.8B Q8: 184 tok/s); what the Neural Engine brings is prefill throughput
and power, with the CPU and GPU left free.

The same packages through oxinfer (`oxinfer --model out/qwen35`, or `Engine::load`): the logits
are bit-identical to coremltools' predictions of the package with `coreml_check.py`'s call
sequence (300 teacher-forced positions of the 135-token prompt, also with the prompt cut to 128
tokens so it ends on a batch boundary, and token-by-token prefill), greedy generation is identical
token for token, and so are three-turn chats through `Engine` replayed by the Python driver (the
Qwen3.5 state carried across turns: logit cosine 0.9975 against transformers over the whole
conversation, as for one prompt). The runtime follows the format's rules: every token fed exactly
once, the prediction from the call that processed the last one (`prefill`'s last-token output on a
batch boundary), no padded batches into the recurrent state, a reset makes a new state. For the
transformer packages `auto` pads the last partial batch, which here gives the same logits as
feeding those tokens one by one. M5 Max, Neural Engine, 227-231-token prompt + 200 tokens,
best of 4 with another job using 13 CPU cores (runs varied by 5-15%; the coremltools driver
measured under the same load is within that noise of oxinfer):

| Package | Load (cached / first) | Prefill | Decode | RSS | CPU time |
|---|---|---|---|---|---|
| Qwen3-0.6B LUT6 | 0.3 s / 46 s | 3070 tok/s | 95 tok/s | 215 MB | 0.6 s |
| Qwen3.5-0.8B LUT6 | 0.4 s / 90 s | 650 tok/s (3430 on a 256-token prompt) | 95 tok/s | 235 MB | 0.8 s |
| Gemma 4 E2B LUT6 | 0.5 s / 100 s | 1030 tok/s | 35 tok/s | 595 MB | 1.6 s |
| Gemma 4 E4B LUT6 | 0.9 s / 180 s | 546 tok/s | 19.6 tok/s | 620 MB | 2.3 s |

Qwen3.5 prefill is slower on prompts that do not fill whole batches: the remainder (39 of 231
tokens here, and all of a short chat turn) goes through `infer` at decode speed, since padding
would enter the recurrent state. The first load compiles the model for the Neural Engine
(cached by Core ML until another process loads the package with other settings).

**Long contexts.** Gemma 4's sliding-window layers keep a block-shifting cache of window + 2 x batch
= 640 positions once the context is longer than that (`--full-local-cache` restores
context-length caches); only the global layers' cache grows with the context. Keys keep their
slot within a block of `batch` positions, so `prefill` and `infer` give bit-identical logits
(oxinfer `--coreml-prefill auto` and `single`: identical last-prompt logits and replies), and
packages that differ only in context length too. Gemma 4 E2B LUT6, 1939-token prompt (a README
excerpt) + 100-200 teacher-forced tokens:

| Context | Local cache | Logit cosine / top-1 | Prefill | Decode (at the end of the prompt) |
|---|---|---|---|---|
| 2048, full-length local cache | 25 MB | 0.99980 / 98.0% | 1215 tok/s (1920 tokens) | 26.6 tok/s |
| 2048, block-shifting local cache | 8 MB | 0.99980 / 98.0% | 1530 tok/s | 33.4 tok/s |
| 8192, block-shifting local cache | 8 MB (vs 100 MB) | 0.99969 / 97.0% | 1080 tok/s (7936 tokens) | 22.6 tok/s |

**Memory on the Neural Engine.** Loading a chunk adds wired memory outside the process
footprint. Each function of the multifunction chunk gets its own copy of the weights even though
the package stores them once, and 3-bit LUTs take as much ANE memory as 4-bit ones. For four
layers of Qwen3.8-27B (1.52 B parameters): 0.38 / 0.57 / 0.76 GB on disk at LUT2 / 3 / 4, and
0.47 / 0.81 / 0.84 GB of ANE memory with only `infer` loaded (1.2 / 1.7-1.9 / 1.3-1.9 GB with
`prefill` too); `infer` takes 14.5 ms per four layers whatever the width. A whole Qwen3.8-27B
(24.3 B decoder parameters) would need about 13 GB of ANE memory at LUT3 or LUT4 with `infer`
alone (twice that with `prefill`), 7.5 GB at LUT2, plus ~0.7 GB for a LUT4 LM head: beyond what an
app can use on a 12 GB iPhone except at 2 bits, which costs far too much quality. On this Mac it
would decode at ~4 tok/s. Dense models up to ~8 B parameters (about 0.55 GB of ANE memory per
billion parameters at LUT4, `infer` only) are the practical range for 12 GB phones.
