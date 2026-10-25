# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
