# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.9](https://github.com/KarpelesLab/oxinfer/compare/v0.0.8...v0.0.9) - 2026-11-13

### Other

- oxinfer --check-update / --update from the GitHub releases (compcol, fstool)
- capabilities (CAN column, --can filter), every model of the README table
- clippy (large enum variants boxed, redundant slicing) and rustfmt in the MiniMax code
- --models, what runs on this machine with an estimated decoding speed
- --control-image reaches both FLUX and Qwen-Image; README: transcription endpoint fields
- rustfmt
- Qwen-Image image to image, inpainting, ControlNet, Layered and FP8 files (flags, agreement)
- image to image, inpainting, ControlNet, Layered, FP8 scaled checkpoints
- response_format (json_object, json_schema), tool_choice required / function, logprobs
- Constrained decoding: JSON / JSON Schema grammars, forced tool calls and logprobs in the sampler
- reference checks for the new modes (tiny pipeline, real-weight steps) and README
- LoRAs, several Kontext references, image to image, inpainting, Fill, Canny / Depth, Redux, BFL single files
- OmniGen2 / BAGEL / Step1X-Edit step caches, DPM-Solver++ and posterior sampling as measured
- OmniGen2, BAGEL, Step1X-Edit: step caches, DPM-Solver++, VAE posterior sampling (all opt-in)
- JanusFlow, Janus image sizes and conversations, measured agreement
- JanusFlow (rectified flow, SDXL VAE), conversations, output size
- Stable Diffusion 3 / 3.5: tools/ref_sd3.py reference, README section and measured agreement
- Stable Diffusion 3 / 3.5: MMDiT transformer, CLIP-L / CLIP-G / T5 conditioning, text to image and image to image
- Stable Diffusion XL: diffusers reference script and measured agreement
- Stable Diffusion XL: CLI, C API and server defaults, library example
- Stable Diffusion XL: two text encoders, added conditioning, kohya LoRA names
- batched utterances
- batched decoding of several clips
- decode steps of independent sequences in one pass
- adapters on routed MoE experts, per selected expert
- document beam search and batches, reference check against openai-whisper
- beam search and batched transcription
- x86-64 AVX2 kernels executed and verified on real hardware
- x86-64 kernels: f16 scales through F16C inside the Q4 and GGUF kernels
- AVX2 score lanes and value accumulation on x86-64
- AVX2 decode kernels and dequantization, bit-identical to the portable path
- x86-64 kernels: direct AVX2 tests against the scalar fallbacks, OXINFER_SIMD=scalar
- README (platform row, section with the measured agreement, --device)
- prefill GEMMs on a Vulkan compute device (bf16, Q8, Q4, affine Q4)
- DeepSeek V4: runtime interventions (residual hooks on the hyper-connection copies, shared / routed expert neurons)
- residual hooks on a stream of several copies per token
- Qwen2-based embedders (gte-Qwen2, stella) with bidirectional attention and per-size heads
- fix the pool deadlock in the VAE decode; README row
- skip text encoder folders without weights
- text to image (DiT, PDD sampler, ViT VAE decoder)
- Qwen3-VL text models (`qwen3_vl`) on the Qwen path
- NVFP4 in ModelOpt's packaging (MiniMax-H3-Image-PE)
- MiniMax-VL-01 image input; MiniMax-01 reference fixes
- MiniMax M3 / M3.1 image input
- Qwen2 / Qwen2.5 text models (MiniMax SynLogic), GGUF `qwen2` too
- Always leave the system 20% of its memory (at most 2 GB)
- MiniMax M3 / M3.1 and Text-01 / M1 rows
- MiniMax-Text-01 / VL-01 (text) / M1: lightning attention, post-norm residuals
- MXFP8 and NVFP4 checkpoints (MiniMax-M3.1-Flash, M3-MXFP8)
- MiniMax M3 / M3.1 text: MSA sparse attention, swigluoai MoE, chat format
- MiniMax M2 / M2.1 / M2.5 / M2.7 on the Qwen path; M3 text groundwork
- the whole tokenizer.json format and encode API
- 2048-token prefill plan steps
- whole prefill passes on Metal as a prefill plan
- GPU decoding with images in the context
- Gated DeltaNet rule, attention and MoE router of prefill on Metal
- QSA sparse attention in the GPU decode plan past 2051 tokens of context
- Qwen3.8-Flash-Next image input: Qwen3-VL mmproj GGUF loading (qwen3vl_merger) for Qwen3.5-family towers
- GPU decoding with streamed weights for Gemma 4 / Gemma 3 (dense, E2B/E4B, 26B-A4B)
- CPU decode attention reading shared key/value rows once; DeepSeek V4 / Mistral Large 4 indexer skips
- Streamed MoE on the GPU: release kept expert wraps after every step; README measurements
- stream_gpu=auto in the engine options; README option row
- Streaming GPU decode: no page touching for device-read units, slower eviction retries, measured GPU/CPU choice for streamed MoE
- GPU decoding with streamed weights (dense and MoE Qwen), device memory defaults
- :ops: f32 matmuls are float-stored, so they never reach the decode kernels
- Prompt lookup and prompt cache: measured defaults, README results
- Prompt-lookup drafting with a verification-outcome memory; session snapshots and a prompt cache
- CPU multi-token column-group kernels and batched MoE experts (measurements)
- `old` mode (token pairs and per-expert MoE matmuls)
- multi-column decode kernels for blocks-of-32 formats, batched MoE experts
- Tokenizer in the crate: drop the tokenizers dependency
- cleanroom specs of llama-modchip's performance changes (quant kernels, CPU runtime, model/server, speculative)
- CUDA decoding (speeds, agreement, speculative verification, defaults)
- whole decode steps on the GPU (single- and multi-token plans), streamed MoE experts
- CUDA measurements for OmniGen2, BAGEL, Janus-Pro and device MoE experts
- routed MoE experts on the device (gather, per-expert GEMMs, activation, combination)

## [0.0.8](https://github.com/KarpelesLab/oxinfer/compare/v0.0.7...v0.0.8) - 2026-11-05

### Other

- Qwen3.8-Flash-Next (model table, section with agreement and speed numbers)
- multi-token GPU verification for MTP (shared expert wraps between plans), MTP from unsloth's GGUF heads
- GPU decode on the real model (per-pass expert wrapping for huge MoE), auto-streaming excludes the n-gram table, split-GGUF metadata fixes, tools
- Qwen4-Exp (Qwen3.8-Flash-Next): hyper-connections, n-gram embedding, QSA, MTP; CPU + Metal decode
- Fast verification by default; --exact-verify opts out; Q4 Gemma 4 31B / Qwen3.8-27B drafts default with it
- Gemma 4 26B-A4B (MoE) measured on CUDA
- Fast verification: README speed / drift tables, C header key, CLI banner, specialized Q5_K / Q6_K loaders
- Fast verification (opt-in): batched simdgroup-matrix matmuls for speculative verification
- FLUX Kontext, Qwen-Image and Step1X-Edit measured on CUDA
- httpsd 0.2.0; drop the 20 s late-response workaround, --max-body-mb sets httpsd's limit
- Build the binary without the `media` feature again (wasi)
- Start-up banner: only claim GPU decoding for a backend that has it
- Speculative decoding defaults: DSpark for Qwen3 4B / 8B / 14B and Gemma 4 12B, MTP for Qwen3.5-4B / 9B; README
- DSpark / MTP: faster exact GPU verification, MoE multi-token passes, cost-aware rounds, GGUF drafts, draft discovery
- OpenAI-compatible server: oxinfer serve, oxinfer::server (httpsd), Engine::chat, C API
- no-reassociation only where decoding plans need it
- DSpark drafts for Qwen and Gemma 4 (agreement, acceptance, speed, CLI)
- Qwen3 / Qwen3.5 / 3.8 and Gemma 4 drafts; multi-token GPU verification
- DeepSeek V4: README section (architecture, checkpoints, agreement, speed, DSpark) and engine docs
- DeepSeek V4: NEON MXFP4 kernel, draft mapped when the target streams, 3 drafts per round by default
- DeepSeek V4: DSpark draft (DeepSeek V4 stages behind dspark::Target), per-group wo_a on the decode kernels
- DeepSeek V4: GGUF tokenizer specials in the BPE vocabulary; streamed MoE prefill wraps only the routed experts
- DeepSeek V4: text model (mHC, CSA / HCA compressed attention, lightning indexer, hash and clamped-SwiGLU MoE), GGUF and DeepSeek-layout checkpoints, MXFP4
- Mistral Large 4: short-prompt prefill on Metal, bounded DSpark context, draft on the GPU
- Mistral Large 4: DSpark speculative decoding, on by default with the draft beside the GGUF
- record what was measured on the RTX 5080
- list the weight formats in the module docs
- FP8 E4M3 GEMMs (f32 and matrix units, dense and grouped MoE)
- Mistral Large 4: explicit mmproj option, tri-state weight streaming
- Mistral Large 4: runtime interventions (residual hooks, dense / shared / routed expert neurons)
- make the context current per call; direct tests of every kernel
- A device asked for by name no longer falls back to the CPU
- fix NVRTC kernel compilation, load native binaries; first run on NVIDIA

## [0.0.7](https://github.com/KarpelesLab/oxinfer/compare/v0.0.6...v0.0.7) - 2026-11-01

### Other

- Mistral Large 4: document streaming and image options in the library and C API
- Mistral Large 4: README (architecture, speed, agreement), reference tools
- Mistral Large 4: streaming by default when too big for RAM, GPU experts in prefill, Pixtral vision
- Mistral Large 4: MLA + DSA indexer + MoE text model, split GGUF, Mistral-native and FP8 checkpoints
- exact attention on the matrix units with f32 operands; Step1X half-attention cost was noise
- quantized Lightning runs lose far less (merge LoRA in f32, exact modulation channels)
- v1.1 template, padding-token masking and reference offsets; README
- Step1X-Edit (v1.2, v1.1): connector + FLUX MMDiT, Qwen2.5-VL prompt encoding, CLI, C API
- instruction-based image editing, text to image and image understanding
- README numbers, calcuis GGUF transformers, mapped text encoder; VAE mid-attention in query chunks
- instruction editing, in-context generation and text to image
- Qwen-Image-Edit (2511 / 2509 / Edit) and Qwen-Image text to image; Qwen2.5-VL
- FLUX CLI: --size is the output size, and the reference image is resized to it
- README section and model table rows; CLI shares --init-image with SD editing
- 128-wide-head attention on the matrix units; examples/flux_edit.rs
- public DoubleBlock / SingleBlock (load + forward) for reuse by other FLUX-style DiTs
- exact modulation vectors with quantized weights, F16 weights as stored, T5 GGUF encoders
- FLUX.1 Kontext: instruction-based image editing, FLUX.1 text to image
- Image editing: README section, sd_edit example, PyTorch .bin diffusers weights (MagicBrush)
- Stable Diffusion image editing: img2img, inpainting, InstructPix2Pix

## [0.0.6](https://github.com/KarpelesLab/oxinfer/compare/v0.0.5...v0.0.6) - 2026-10-28

### Other

- runtime interventions (spec, formats, CLI, library, C API, agreement, results, GPU coverage, limits)
- Runtime interventions: neuron edits, steering, directional ablation, SAE features
- decoding on the GPU (what runs where, speed vs CPU and llama.cpp, agreement), --gpu-decode
- GPU decode: Gemma 4 26B-A4B MoE layers
- whole single-token decode pass on the GPU
- Gemma 4 26B-A4B: MLX experts, GEMM prefill attention, GPU vision MLP, README
- Gemma 4 26B-A4B (MoE): routed experts, vision, GGUF, decision bundles
- voice cloning, SNAC encoder and 32/44 kHz codecs, multilingual info, loop guard
- Llama 3 GGUF and Orpheus GGUF, agreement and speed vs llama.cpp
- Llama GGUF: Llama 3.x and Orpheus TTS GGUF files

## [0.0.5](https://github.com/KarpelesLab/oxinfer/compare/v0.0.4...v0.0.5) - 2026-10-25

### Other

- streaming context 2/2 by default, faster SNAC, README section
- Orpheus text to speech: Llama 3 on the native engine, SNAC decoder, oxinfer::tts
- README section, model table rows, tools/ref_janus.py
- text to image and image understanding
- oxideav-core types in the API; `oxideav` feature; media without oxideav-meta
- Typed decisions over images (JEV-27B-VL, JEV-9B vl/)
- image, video and audio input (Qwen and Gemma 4), agreement and speed
- Vision towers: torchvision-exact resize, GPU matmuls, bf16 Qwen tower
- Qwen3.5/3.6/3.8 image and video input, Gemma 4 video input
- Whisper speed table (final measurements, 6-thread rows)
- --threads
- bf16 cross-attention keys and values below f16 decoder precision
- Whisper agreement for base, medium, large-v3 and tiny.en
- share the engine's run lock, en for English-only models, rule and segment tests
- Whisper speech to text (models, usage, agreement with transformers, speed); tools/ref_whisper.py
- fused attention kernel for 64-wide heads (Whisper encoder, SD 2.x)
- Whisper speech to text: transformers-exact features, encoder/decoder, long-form decoding, timestamps
- project goals (run everything everywhere, then make it fast)
- JEV-35B (merged, LM-head readout) measurements
- typed decisions (Jev models), measured agreement and latency
- Jev typed decisions: runtime LoRA, Engine::decide, C API oxi_decide, CLI --decide
- Core ML converter: block-shifting Gemma local cache, prefill and infer bit-identical
- Core ML runtime: shifting local caches (local_cache_length) are fed exactly once, never padded
- Core ML converter: Fisher-weighted LUTs, mixed precision, shifting Gemma local cache, Qwen3.5 MoE
- llama.cpp comparison rerun for oxinfer after the matrix-unit and multi-token kernel merges
- interleaved oxinfer vs llama.cpp benchmark; README comparison tables
- SD 1.5 with the LCM-LoRA as the default image model, LoRA files in the browser build

## [0.0.4](https://github.com/KarpelesLab/oxinfer/compare/v0.0.3...v0.0.4) - 2026-10-25

### Other

- multi-token GGUF kernels, MoE expert streaming, thread policy, measurements
- Multi-token kernels: SMMLA tiles only for per-16 integer-scale formats, pair kernels for 2 tokens
- optional read-ahead of guessed MoE experts; vector, parallel activation quantization
- evict on a background thread; fix the miss count of a file's last page
- Thread QoS policy, embedding rows read ahead when streaming
- read the routed MoE experts on demand instead of whole layers
- GGUF multi-token decode kernels: weights unpacked once per pass, SMMLA (i8mm) tiles
- Engine options gpu_exact and gpu_moe (process-wide switches in oxinfer::gpu)
- Metal matrix units (M5/A19): prefill and GEMM measurements, MoE on the GPU, decode assessment
- 32-token matrix-unit tiles for short inputs; fail loudly on a broken MMA library
- pad matrix-unit GEMM rows to whole tiles, 32-row tiles for MoE experts
- prefill GEMMs and MoE experts on the M5/A19 matrix units
- Core ML runtime: Qwen3.5 and Gemma 4 packages, Core ML models in the Engine API
- cargo fmt (stream trace lines)
- Core ML converter: docs for Gemma 4 and ANE findings, README section with measurements
- Core ML (ANE) converter in ANEMLL format for Qwen3 and Qwen3.5
- release Metal buffers of streamed weights, resident copies, measurements
- Weight streaming for models larger than RAM (--stream), cost-aware MTP drafts
- Fix the merge of the Core ML backend onto the engine API (Path import; clippy drop_non_drop)
- Core ML: library helpers, example, README section with measurements
- Core ML: auto prefill mode as chat.py, padded prefill option, more layouts
- Core ML backend: run ANEMLL models on the Apple Neural Engine
- Library engine API, C API (capi feature) and an iOS test app
- Qwen3.5/3.8 MTP speculative decoding (--mtp), greedy and sampled
- batch-invariant multi-token decode tiles, quantized passes in token pairs
- load ggml-quantized matrices, tables and experts in their own format
- GGUF NEON: centered nibbles through the integer activation sums; vector IQ4_XS scales
- GGUF decode: center the weights of formats with mins (Q4_K, Q5_K, Q2_K, Q4_1, Q5_1)
- NEON dequantization for the prefill path (K-quants, IQ2/IQ3, IQ4_XS)
- GGUF F32/F16/Q4_1/Q5_0/Q5_1 GEMM as well
- prefill GEMM over GGUF weights (K-quants, i-quants, Q8_0, Q4_0)
- GGUF weight formats: llama.cpp block types as a Weight / Table kind
- Zawen-v0.1-35B-A3B checked on its real weights; GGUF: expose FastMTP d2t maps
- GGUF reader: qwen35 / qwen35moe / qwen3 files with their tokenizer, config and chat template
- Qwen3.5 MoE (Qwen3.6-35B-A3B, Zawen): routed + shared experts, grouped prefill matmuls
- Library API: hf::set_hub, Embedding::truncate, Embedder::sparse_tokens; embed and sd_lora examples
- Stable Diffusion: LoRA / LoCon merging (--lora FILE[:SCALE]), checked against diffusers
- ModernBERT (modernbert-embed, gte-modernbert)
- NomicBERT (nomic-embed-text v1 and v1.5)
- MPNet (all-mpnet-base-v2)
- MiniLM, bge-small, e5 and mxbai embedders checked against sentence-transformers
- EmbeddingGemma, Qwen3-Embedding, BGE-M3 (dense, sparse, ColBERT)

## [0.0.3](https://github.com/KarpelesLab/oxinfer/compare/v0.0.2...v0.0.3) - 2026-10-24

### Other

- MLX checkpoints, the mirror, Gemma 3 and Qwen3; Qwen3 tool-call test
- convolution weights are channels-last; permute them for the Gemma 4 audio tower
- MLX models and the ATOnline mirror
- CUDA and WebGPU: affine Q4 GEMMs for MLX weights
- affine Q4 GEMM for MLX weights; vectorized bf16 dot fallback
- MLX checkpoints, the ATOnline mirror, Qwen3 and Gemma 3

## [0.0.2](https://github.com/KarpelesLab/oxinfer/compare/v0.0.1...v0.0.2) - 2026-10-24

### Other

- reasoning-effort instruction only for templates that have it
- attention and fused group/layer norms; GPU check summary
- attention and fused group/layer norms
- Diffusion on the GPU: attention, and group/layer norms fused into their consumers (Metal)

## [0.0.1](https://github.com/KarpelesLab/oxinfer/compare/v0.0.0...v0.0.1) - 2026-10-24

### Other

- rustfmt src/web/mod.rs
- Browser chat: tools (date/time, sandboxed JavaScript) and an environment-aware system prompt
- Tool calling and a default system prompt (CLI)
- Image generation in the browser: WebGPU convolution, SD exports, Image mode on the site
- implicit-GEMM convolution kernels (conv2d backend hook)
- Diffusion speed: token-split matmul driver, cache-blocked prefill tiles, Metal implicit-GEMM convolution
- Image generation: Stable Diffusion 1.x / 2.x and SD-Turbo from diffusers folders
