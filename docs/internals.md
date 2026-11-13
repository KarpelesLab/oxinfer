# How it works

- `safetensors.rs`: parses the header, exposes bf16 tensors as `&[u16]` (single file or sharded).
- `config.rs`: reads `config.json` and derives per-layer facts (sliding vs global, head dims, KV sharing source, k=v).
- `advisor.rs`: `--models`, the tested models against the machine's disk, memory and GPU, with an
  estimated decoding speed (kernel rates timed on the spot)
- `platform.rs`: CPU feature and GPU detection (sysctl / procfs for memory, Metal via `objc_msgSend`,
  CUDA driver API, Vulkan loader), all through `dlsym`; `gpu.rs`: the `Backend` trait the linear layers
  dispatch through and the registry; `metal.rs`, `cuda.rs`, `vulkan.rs`: the backends with their shader sources
  (`src/vulkan/`: GLSL and the SPIR-V compiled from it).
- `fsx.rs`, `clock.rs`: file access and monotonic time behind one interface, so the loaders run on a
  real filesystem or on the browser's in-memory table; `src/web/` (feature `web`): the browser module's
  C-ABI exports and the WebGPU backend calling back into the host; `web/` at the root: the
  JavaScript host and page.
- `whisper/`: speech to text: `mel.rs` (Whisper's log-mel features, mixed-radix FFT), `model.rs`
  (encoder, decoder, cross-attention cache), `tokens.rs` (special tokens, languages, generation
  config), `align.rs` (word timestamps: DTW), `compress.rs` (the fallback's compression ratio),
  `mod.rs` (`Transcriber`: decoding rules, long-form windows, fallback); `whisper_cli.rs`: the CLI.
- `flux/`: FLUX.1 / Kontext: `transformer.rs` (double / single stream blocks), `vae.rs` (FLUX VAE
  encode / decode, latent packing), `sched.rs` (flow-matching Euler), `rope.rs` (multi-axis RoPE),
  `t5.rs` (T5 encoder), `weights.rs` (layers from safetensors / GGUF), `mod.rs` (the pipelines);
  `flux_cli.rs`: its CLI.
- `sd/`: Stable Diffusion: `clip.rs` (BPE tokenizer, text encoder), `unet.rs`, `vae.rs`, `sched.rs`
  (Euler samplers), `xl.rs` (SDXL: second text encoder, size conditioning, kohya LoRA names), `ops.rs` (channels-last convolution, group/layer norm, multi-head attention on
  the f32 GEMM tiles); `png.rs`: a dependency-free PNG writer.
- `kernels.rs`: NEON kernels (`kernels_x86.rs`, `kernels_wasm.rs`: the AVX2 and simd128 equivalents, selected by `build.rs` through the `has_simd` cfg). Decode tiles stream 1–2 weight rows with 8 accumulators (the prefetcher
  prefers few long streams); prefill packs activations into 8-token panels and runs an 8-row x 8-token
  outer-product kernel that broadcasts bf16 weights by lane; `kernels_ggml.rs`: the GGUF formats'
  one-token and multi-token (SDOT / SMMLA) decode kernels and dequantization (`kernels_ggml_x86.rs`: the AVX2 ones). Also RMSNorm, GELU (tanh approximation via
  a fast exp), RoPE tables, softmax.
- `quant.rs`: Q8 (32 weights per f32 scale) and Q4_0 (two nibbles per byte, f16 scale) block
  quantization of weights at load time, plus per-matmul activation quantization; decode kernels use
  int8 SDOT over four blocks at a time, with multi-token variants for speculative verification.
- `drafter.rs`: the Gemma 4 assistant model (drafter) and its centroid-masked LM head.
- `dspark.rs`: DSpark drafts (llama.cpp `dflash` GGUFs: target-feature context cache, block of
  mask embeddings, Markov head; the Mistral Large 4 MLA backbone) and their speculative decoder,
  over any target implementing `dspark::Target`.
- `qwen_mtp.rs`: the Qwen3.5 MTP layer and the speculative decoder (drafting, verification with
  recorded recurrent states, speculative sampling).
- `vision.rs`, `audio.rs`: Gemma 4's two encoders plus preprocessing (resize/patching, mel
  features) and the embedders; `qwen_vision.rs`: the Qwen3.5-family image / video processors and
  vision tower; `resample.rs`: torchvision- and Pillow-exact bicubic resizing; `video.rs`: frame
  sampling and video decoding; `media.rs`: file decoding through oxideav; `av.rs`: decoded oxideav
  frames to RGB / mono PCM and back; `mm.rs`: attachments to soft tokens and prompt placeholders; `mm_config.rs`: Gemma's tower configs and placeholder
  token ids.
- `pool.rs`: a spin-waiting thread pool with an atomic work counter. Dispatch costs about a microsecond,
  which matters because a decode step issues ~200 small parallel regions.
- `model.rs`: the Gemma 4 decoder: scaled embeddings, per-layer embeddings (PLE), QK-norm, sliding /
  global attention with proportional RoPE on 512-dim global heads, KV-cache sharing for the top layers,
  gated-GELU MLP, post-norms, per-layer scalars, tied softcapped LM head.
- `chat.rs`, `sampler.rs`, `main.rs`: chat template, greedy / top-k / top-p sampling, streaming CLI.
- `coreml.rs`, `anemll.rs`: Core ML through the Objective-C runtime (models, functions, states,
  multi-arrays, IOSurface buffers) and the ANEMLL model runner (`meta.yaml`, prefill / infer
  protocol, sliding-window rotation, split or argmax LM head); `anemll_cli.rs`: its CLI driver.
