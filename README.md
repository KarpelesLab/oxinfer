# oxinfer

[![CI](https://github.com/KarpelesLab/oxinfer/actions/workflows/ci.yml/badge.svg)](https://github.com/KarpelesLab/oxinfer/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/oxinfer.svg)](https://crates.io/crates/oxinfer)
[![docs.rs](https://img.shields.io/docsrs/oxinfer)](https://docs.rs/oxinfer)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**[Try it in your browser](https://karpeleslab.github.io/oxinfer/)**: Qwen 3.5 0.8B downloads from
Hugging Face and runs in the tab, with WebGPU when the browser has it; MLX 4-bit Gemma 4, Gemma 3
and Qwen3 models load from the mirror too. Early days: the crate is at 0.0.x and the API and file
formats may still change.

Local LLM inference in pure Rust: Gemma 4 (text, images, video, audio), Qwen 3.5 / 3.6 / 3.8
(text, images, video), Mistral Large 4 (text, images; 1T parameters streamed from storage),
DeepSeek V4 Flash (text; 284B parameters), Qwen3.8-Flash-Next (text; 125B + a 51B n-gram table read
from storage), Gemma 3 and Qwen 3 today, more model families as they come; also
embeddings, Stable Diffusion, FLUX.1 Kontext, Qwen-Image-Edit, OmniGen2, BAGEL and Step1X-Edit image editing, Whisper speech to text and Orpheus text to speech. No BLAS, no ML framework: weights are read straight out
of the safetensors files (HuggingFace checkpoints or MLX 4-bit ones) and every operation
(matmuls, attention, norms, RoPE, sampling) is implemented here, with hand-written SIMD kernels
(NEON, AVX2, wasm simd128) and its own thread pool. The CPU path runs everywhere; GPUs (Metal,
CUDA, WebGPU in the browser) are detected at run time and used for prefill when present, with
nothing linked at build time.

Output is validated against the `transformers` float32 reference: on Gemma 4 E2B-it the
last-token logits agree to a max absolute difference of about 1e-4 and greedy decoding produces
identical tokens.

## Goals

1. **Run everything, everywhere.** Any model on any machine: a phone, a browser tab, a laptop with
   no GPU, a server. It may be slow, but it works. A missing accelerator, too little memory or an
   unusual platform should never be the reason a model cannot run. The CPU path is the reference
   and is always there; a 1 TB model on a machine with 1 GB of RAM and no GPU should still produce
   tokens, by reading the weights from storage as it goes (see
   [Streaming weights from flash](#streaming-weights-from-flash-models-larger-than-ram)), very slowly.
2. **Then make it fast.** Anything that improves speed on a given machine is in scope: a GPU
   when there is one (Metal, CUDA, WebGPU, detected at run time), the Neural Engine through Core
   ML, quantization, SIMD kernels per CPU (NEON, SDOT / i8mm, AVX2, wasm simd128), speculative
   decoding (MTP, drafters, prompt lookup), prompt caches restored from disk, prefetching and
   caching for weights that do not fit in memory. When
   something is missing, oxinfer falls back to a slower path rather than failing.

Correctness comes first in both: every model and every fast path is checked against the reference
implementation (the [verification](#verifying-against-the-reference) section and the per-model
notes give the measured agreement), and a speed-up that changes the output must say so and stay
optional.

## Supported models

| Model | Params (total) | bf16 size | Status |
|---|---|---|---|
| google/gemma-4-E2B-it | 5.1B | 10.2 GB | text, images, video, audio; matches reference (max logit diff 1e-4) |
| google/gemma-4-E4B-it | 8.0B | 16.0 GB | text, images, video, audio; matches reference (max logit diff 8e-5) |
| google/gemma-4-12B-it | 12B | 24 GB | supported (the 26B-A4B code without the experts), untested |
| google/gemma-4-31B-it | 31B | 62 GB | supported, sharded loading, untested |
| google/gemma-4-26B-A4B-it | 25.8B MoE, 3.8B active | 52 GB | text, images; matches transformers (max logit diff 3e-4, same greedy tokens); GGUF and MLX 4-bit copies too; see [Gemma 4 26B-A4B](#gemma-4-26b-a4b-moe) |
| autotrust/GEV-26B-Decide, JEV-Gemma4-26B-A4B | 25.8B MoE + LoRA | 52 GB + 0.15 GB | typed decisions over text and images, on the bundled or Google's base; match transformers + peft to 2e-6 (see [Typed decisions](#typed-decisions-jev-models)) |
| Qwen/Qwen3.8-27B (and Qwen3.5/3.6 dense) | 27.8B | 55.6 GB | text, images, video; matches the bf16 reference; see below |
| Qwen/Qwen3.6-35B-A3B and fine-tunes (PixilabAI/Zawen-v0.1-35B-A3B) | 35B MoE, 3B active | 69 GB | text (the vision code applies to MoE checkpoints too but is untested there: Zawen ships no vision tower); Zawen matches the reference (see below) |
| autotrust/JEV-9B, JEV-27B (Jev decision models) | 9B / 27.8B + LoRA | 18 / 54 GB + 0.16 / 0.44 GB | typed decisions, on the bundled or any matching base; match transformers + peft (see [Typed decisions](#typed-decisions-jev-models)); image decisions with JEV-9B's `vl/` on Qwen/Qwen3.5-9B |
| autotrust/JEV-27B-VL | 27.8B + vision + LoRA | 55.6 GB + 0.45 GB | typed decisions over text and images; matches the reference server's math on transformers + peft (see [Image decisions](#image-decisions)) |
| Qwen/Qwen3.5-0.8B | 0.8B | 1.7 GB | text, images, video; matches transformers float32 (see [Image, video and audio input](#image-video-and-audio-input)); the small one used for the wasm measurements |
| mlx-community/gemma-4-e2b-it-4bit, gemma-4-e4b-it-4bit | 5.1B / 8.0B | 3.6 / 5.2 GB (MLX 4-bit) | text, images and audio; see [MLX checkpoints](#mlx-checkpoints) |
| mlx-community/gemma-4-26b-a4b-it-4bit | 25.8B MoE | 15.4 GB (MLX 4-bit) | text and images (4-bit `switch_glu` experts) |
| Gemma 4 GGUF (ggml-org/gemma-4-26B-A4B-it-GGUF, unsloth, bartowski, ...) | 25.8B MoE tested | 14.6 GB (Q4_0) - 27 GB (Q8_0) | text; same next token as transformers and llama.cpp (see [Gemma 4 26B-A4B](#gemma-4-26b-a4b-moe)) |
| mlx-community/Qwen3-4B-4bit (Qwen3 dense) | 4.0B | 2.3 GB (MLX 4-bit) | text; matches the reference on the same weights (same top 10) |
| mlx-community/gemma-3-4b-it-4bit (Gemma 3) | 4.3B | 3.4 GB (MLX 4-bit) | text only; matches the reference, also past the 1024-token window |
| google/embeddinggemma-300m | 0.3B | 1.2 GB (fp32) | embeddings; matches sentence-transformers (see [Embeddings](#embeddings)) |
| Qwen/Qwen3-Embedding-0.6B | 0.6B | 1.2 GB | embeddings; matches sentence-transformers |
| Alibaba-NLP/gte-Qwen2-1.5B-instruct, NovaSearch/stella_en_1.5B_v5 | 1.5B | 7.1 / 6.2 GB (fp32) | embeddings (bidirectional Qwen2; stella with its per-size heads); match sentence-transformers |
| BAAI/bge-m3 | 0.6B | 2.3 GB (fp32, `pytorch_model.bin`) | dense, sparse and ColBERT embeddings; matches the reference |
| sentence-transformers/all-MiniLM-L6-v2, paraphrase-multilingual-MiniLM-L12-v2 | 23M / 118M | 90 / 470 MB | embeddings (BERT); match sentence-transformers |
| BAAI/bge-small-en-v1.5, intfloat/multilingual-e5-small, mixedbread-ai/mxbai-embed-large-v1 | 33M-335M | 0.13-1.3 GB | embeddings (BERT / XLM-R); match sentence-transformers |
| stabilityai/sd-turbo | 1.3B | 2.6 GB (fp16) | text-to-image in one step, image-to-image in one or two; matches diffusers (see Image generation) |
| stable-diffusion-v1-5/stable-diffusion-v1-5 (and SD 1.x / 2.x fine-tunes) | 1.1B | 2.0 GB (fp16) | text-to-image, image-to-image, inpainting by mask blending; matches diffusers |
| timbrooks/instruct-pix2pix | 1.1B | 2.0 GB (fp16) | instruction edits of an image ("make it snowy"), also in 4 steps with the SD 1.5 LCM-LoRA; matches diffusers (see [Editing images](#editing-images)) |
| stable-diffusion-v1-5/stable-diffusion-inpainting (and other 9-channel SD inpainting UNets) | 1.1B | 2.0 GB (fp16) | inpainting with a mask; matches diffusers |
| vinesmsuic/magicbrush-jul7 (MagicBrush: InstructPix2Pix fine-tuned for local edits) | 1.1B | 4.3 GB (fp32 `.bin`) | instruction edits ("add a red car"); matches diffusers |
| deepseek-ai/Janus-Pro-1B | 2.1B (1.5B LM) | 4.2 GB (`pytorch_model.bin`) | text-to-image (384x384) and image understanding; matches the official `janus` package (all 576 greedy image tokens identical; see [Janus-Pro](#janus-pro-text-to-image-and-image-understanding)) |
| deepseek-ai/Janus-Pro-7B | 7.4B | 14.8 GB (sharded `.bin`) | same; matches the reference (logits, first 64 greedy image tokens) |
| deepseek-ai/JanusFlow-1.3B | 2.0B (1.3B LM) | 4.1 GB + the SDXL VAE (335 MB) | text-to-image by rectified flow (384x384) and image understanding; matches the official code (velocities, latents, image within 1 of 255; see [Janus-Pro](#janus-pro-text-to-image-and-image-understanding)) |
| black-forest-labs/FLUX.1-Kontext-dev (ungated copy: unsloth/FLUX.1-Kontext-dev) | 12B + T5-XXL 4.7B + CLIP-L | 23.8 GB + 9.5 GB | instruction-based image editing (and text to image); matches diffusers' FluxKontextPipeline (transformer blocks 3e-6, 55 dB end to end on a Q4_0 GGUF; see [FLUX.1 Kontext](#flux1-kontext-image-editing-from-instructions-and-flux1-text-to-image)); FLUX.1 [dev] Non-Commercial License |
| QuantStack/FLUX.1-Kontext-dev-GGUF | 12B | 4.0 GB (Q2_K) - 12.7 GB (Q8_0) | the same with a GGUF transformer (any block format, zero-copy); T5 / CLIP / VAE from unsloth/FLUX.1-Kontext-dev |
| city96/FLUX.1-schnell-gguf, city96/FLUX.1-dev-gguf | 12B | 6.8 GB (Q4_0) | text to image (schnell: 4 steps, Apache 2.0); schnell Q4_0 matches diffusers at 61.6 dB |
| Stable Diffusion 3 medium, Stable Diffusion 3.5 medium (diffusers layout; Stability AI's repos are gated) and city96/stable-diffusion-3.5-medium-gguf | 2.1B / 2.5B MMDiT + T5-XXL 4.7B + CLIP-L + CLIP-G | 4.2 / 4.9 GB (fp16 / bf16) + 9.5 + 1.6 GB; 1.7 GB (Q4_0) | text to image and image to image; match diffusers' StableDiffusion3Pipeline on real weights (transformer pass 1e-6 to 4e-6, 62-67 dB after 4 guided steps; see [Stable Diffusion 3 / 3.5](#stable-diffusion-3--35-text-to-image-and-image-to-image)); SD 3.5 large / large-turbo use the same code, not run |
| Qwen/Qwen-Image-Edit-2511 (also -2509, -Edit) | 20B MMDiT + Qwen2.5-VL-7B (8.3B) + VAE | 41 + 16.6 + 0.25 GB | instruction-based image editing with one or more images; matches diffusers' QwenImageEditPlusPipeline (68.5 dB end to end on the CPU; see [Qwen-Image-Edit](#qwen-image-edit-image-editing-from-instructions-and-qwen-image-text-to-image)); Lightning LoRAs (4 / 8 steps), GGUF transformers (unsloth, QuantStack); Apache 2.0 |
| Qwen/Qwen-Image, Qwen/Qwen-Image-2512 | same | same | text to image with the same transformer code (2512 tested from unsloth's Q4_0 GGUF + Lightning) |
| OmniGen2/OmniGen2 | 4.0B DiT + Qwen2.5-VL-3B + FLUX VAE | 15.9 + 15.0 + 0.3 GB (float32) | instruction-based image editing, in-context generation with up to 5 images, text to image; matches the official OmniGen2Pipeline (60 dB end to end with the same bf16 weights; see [OmniGen2](#omnigen2-image-editing-in-context-generation-and-text-to-image)); 2.3 GB of transformer weights at `--quant q4`; Apache 2.0 |
| stepfun-ai/Step1X-Edit-v1p2 (also -v1p2-preview) | 12.4B MMDiT + Qwen2.5-VL-7B (8.3B) + FLUX VAE | 24.9 + 16.6 + 0.34 GB | instruction-based image editing (and thinking mode); matches StepFun's diffusers Step1XEditPipelineV1P2 (62 dB after a step on the CPU, 53 dB after 8 steps on Metal; see [Step1X-Edit](#step1x-edit-image-editing-from-instructions)); q8 / q4 at load; Apache 2.0 |
| stepfun-ai/Step1X-Edit-v1p1-diffusers | same (stock Qwen2.5-VL-7B) | same | v1.1 editing; matches Step1XEditPipeline (49.5 dB after a step at 1024x1024) |
| ByteDance-Seed/BAGEL-7B-MoT | 14B (7B x 2 experts) + SigLIP 0.4B + VAE | 29.2 GB + 0.34 GB | one unified model: instruction-based image editing, text to image, image understanding, optional thinking; matches the official inference code (truncated f32 runs 1e-5; 41-44 dB end to end vs bf16; see [BAGEL](#bagel-image-editing-text-to-image-and-image-understanding-in-one-model)); Apache 2.0 |
| Qwen/Qwen2.5-VL-3B-Instruct, -7B-Instruct | 3.8B / 8.3B | 7.5 / 16.6 GB | text and images (no video yet); matches transformers float32 (logits max diff 1.1e-4, same greedy answer; see [Qwen2.5-VL](#qwen25-vl)); also the conditioning encoder `oxinfer::qwen_vl` |
| city96/t5-v1_1-xxl-encoder-gguf | 4.7B | 2.9 GB (Q4_K_M) - 5.1 GB (Q8_0) | FLUX's T5 encoder as GGUF (`--text-encoder`) |
| anemll/* (ANEMLL Core ML: Qwen3, Qwen2.5, Gemma 3, Llama 3.2, ...) | 0.27B-1.7B tested | 0.3-1.5 GB (LUT6) | Apple Neural Engine via Core ML; same tokens as ANEMLL's runtime (see [Core ML](#core-ml--apple-neural-engine)) |
| openai/whisper-tiny, -base, -small, -medium (and `.en`) | 39M-769M | 0.15-3.1 GB (fp32) | speech to text; transcripts match transformers token for token (see [Speech to text](#speech-to-text-whisper)) |
| openai/whisper-large-v3, -large-v3-turbo, distil-whisper/distil-large-v3 | 1.55B / 809M / 756M | 3.1 / 1.6 / 1.5 GB (fp16) | speech to text; match transformers token for token |
| canopylabs/orpheus-3b-0.1-ft (ungated copy: unsloth/orpheus-3b-0.1-ft) + hubertsiuzdak/snac_24khz | 3.3B + 20M | 6.6 GB + 80 MB (fp32) | text to speech; greedy tokens and waveform match transformers + snac (see [Text to speech](#text-to-speech-orpheus)); also any Llama 3.x as a text model |
| Orpheus GGUF (isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF, lex-au/Orpheus-3b-FT-Q8_0.gguf) | 3.3B | 2.4 / 4.0 GB | text to speech; whisper-small WER 0-1% like the safetensors model |
| mistralai/Mistral-Large-4-1T-A52B; unsloth/Mistral-Large-4-1T-A52B-GGUF | 1T MoE, 52B active (+ 2B vision) | 1.06 TB (FP8) / 378 GB (GGUF UD-Q2_K_XL) | text and images; streamed from storage when larger than RAM (1.3-1.5 tok/s on a 128 GB M5 Max; DSpark speculative decoding); matches transformers (tiny random checkpoints with every feature, KL < 1e-9) and llama.cpp on the real GGUF (same next tokens, same greedy text); see [Mistral Large 4](#mistral-large-4-1t-a52b) |
| deepseek-ai/DeepSeek-V4-Flash as GGUF (unsloth/DeepSeek-V4-Flash-GGUF) | 284B MoE, 13B active | 97 GB (UD-Q2_K_XL) | text, tool calls; streamed from storage when larger than 60% of RAM (5-10 tok/s on a 128 GB M5 Max); matches DeepSeek's own inference code (tiny random checkpoints, max logit diff 8e-5, KL 3e-11) and llama.cpp on the real GGUF (same greedy text); DSpark speculative decoding (DeepSeek-V4-Flash-DSpark); see [DeepSeek V4](#deepseek-v4-flash) |
| Qwen/Qwen3.8-Flash-Next as GGUF (unsloth/Qwen3.8-Flash-Next-GGUF) | 125B MoE, 6B active, + 51B n-gram embedding + 4B MTP | 111 GB (UD-Q4_K_XL, 29 GB of it the n-gram table) | text, thinking on / off, tool calls, MTP speculative decoding (unsloth's MTP GGUF); hyper-connections, n-gram embedding (read a few rows per token from storage) and QSA sparse attention; matches transformers (tiny random checkpoints, max logit diff 6.6e-7) and llama.cpp on the real GGUF (same greedy tokens, KL 2e-4 to 3e-3); 45-59 tok/s Metal decode (47 at 4.5k context), 21 CPU; see [Qwen3.8-Flash-Next](#qwen38-flash-next) |
| MiniMaxAI/MiniMax-M2, -M2.1, -M2.5, -M2.7 (FP8) and their GGUFs (unsloth/MiniMax-M2.7-GGUF, ...) | 229B MoE, 10B active | 230 GB (FP8) / 141 GB (UD-Q4_K_XL) | text, thinking, tool calls (`<minimax:tool_call>`); matches transformers (tiny random checkpoints, bf16 and FP8 block-quantized, max logit diff 2e-6) and llama.cpp on the real UD-Q4_K_XL GGUF (same greedy tokens 17/17, mean KL 9e-5); CPU decode for now (5.9 tok/s on 192 cores) |
| MiniMaxAI/MiniMax-M3, -M3-MXFP8, -M3.1-Flash and their GGUFs (unsloth/MiniMax-M3.1-Flash-GGUF, ...) | 427B / 220B MoE (text tower) | 854 GB (bf16) / 144 GB (UD-Q2_K_XL) | text, images, thinking (enabled / disabled / adaptive per the template), tool calls (nested XML); MiniMax Sparse Attention (lightning indexer, top 16 blocks of 128 per KV head, exact dense below 2048 positions), swigluoai experts; MXFP8 and NVFP4 (compressed-tensors) checkpoints; matches transformers (tiny random checkpoints in transformers' and the release's layouts, bf16 and MXFP8 + NVFP4, max logit diff 3.4e-6) and llama.cpp on the real UD-Q2_K_XL GGUF (17/17 greedy tokens, mean KL 1.6e-4); images (CLIP tower with 3D RoPE and patch merge, from the checkpoint or llama.cpp's mmproj file: pixels identical to transformers', embeddings within 9e-4 of transformers float32 on the real M3.1-Flash weights; tiny random models end to end within 6e-6); CPU decode for now (2.9 tok/s; 3.7 with M3.1-Flash's DSpark draft: 3.8 tokens per target forward); video not yet |
| MiniMaxAI/MiniMax-Text-01, -M1-40k, -M1-80k (and -hf), MiniMax-VL-01 | 456B MoE, 46B active (+ 0.3B CLIP) | 912 GB (bf16) | text, tool calls, images (VL-01: anyres tiles on CLIP ViT-L/14-336, unpadded with row newlines); lightning attention (7 of 8 layers) + softmax attention, post-norm residuals; matches MiniMax's own modeling code (tiny random checkpoints: max logit diff 1e-6 at 40 tokens, 2.5e-5 at 3000; VL-01 pixels identical, image features 2.5e-6, a 3043-token image prompt within 2.4e-5 of float64); untested on the real weights (no GGUF exists, the bf16 release did not fit here) |
| MiniMaxAI/MiniMax-H3-Image (+ MiniMax-H3's Qwen3-VL text encoder, `FL2VA/text_encoder`) | 31B DiT + 32B text encoder + ViT VAE decoder | 62 + 63 + 2.2 GB (bf16) | text to image in 6 PDD steps on a canvas of at least 2048x2048 (`OXINFER_H3_MIN_AREA` lowers it; much below 1024x1024 the model falls apart); DiT and tiled VAE decoder match diffusers on tiny checkpoints (7e-7 / 4e-7, both weight layouts); 1024x1024 in 2.3 min on the CPU |
| MiniMaxAI/MiniMax-H3-Image-PE (H3-Image's prompt enhancer: Qwen3.5-27B, ModelOpt NVFP4) | 27.4B | 18 GB (NVFP4) | text, thinking; its prompts (`prompt/t2i`, `prompt/edit`) give the expanded caption and aspect-ratio JSON; NVFP4 as ggml's blocks (both ModelOpt's and compressed-tensors' packagings match transformers exactly on tiny checkpoints) |
| MiniMaxAI/SynLogic-7B, -32B, -Mix-3-32B (Qwen2.5 fine-tunes; any `qwen2` model, GGUF too) | 7.6B / 32.8B | 15 / 66 GB | text; SynLogic-7B matches transformers float32 (max logit diff 1.6e-4, same top 10) |
| Llama 3.x GGUF (bartowski/Llama-3.2-1B-Instruct-GGUF, ...) | 1.2B tested | 0.8 GB (Q4_K_M) - 2.5 GB (F16) | text; F16 file bit-identical to the safetensors model, quantized files match llama.cpp (see [Llama 3 GGUF](#llama-3-gguf)) |
| canopylabs/orpheus-3b-0.1-pretrained (ungated copy: unsloth/orpheus-3b-0.1-pretrained) | 3.3B | 6.6 GB | voice cloning from a reference clip (SNAC encoder); prompt and greedy tokens match the canopylabs notebook + transformers |
| canopylabs/3b-{fr,de,ko,hi,zh,es_it}-ft-research_release (ungated copies: Our-Org-2/...) | 3.3B | 13 GB (fp32) | multilingual text to speech (French and German tested; French greedy tokens match transformers) |
| hubertsiuzdak/snac_24khz, snac_32khz, snac_44khz | 20M / 55M / 55M | 80 / 220 / 220 MB | the SNAC codec alone, encode and decode (32 / 44 kHz with local attention); codes identical to the `snac` package |

### Gemma 4 26B-A4B (MoE)

`google/gemma-4-26B-A4B-it` is the dense Gemma 4 layer (30 layers, 25 sliding with a 1024-token
window and 8 KV heads of 256, 5 global with 2 KV heads of 512 and keys reused as values) with a
mixture of experts next to the MLP. Every layer runs both, on the same residual stream:

- the dense GELU MLP (2112 wide) as usual, its output RMS-normed (`post_feedforward_layernorm_1`);
- a router reading the residual RMS-normed without weight, times a learned scale / sqrt(hidden):
  softmax over 128 experts, the top 8 renormalized, each weight times a learned per-expert scale;
- the 8 chosen GELU experts (704 wide, fused `gate_up_proj`) on the residual through their own
  norm (`pre_feedforward_layernorm_2`), their weighted sum RMS-normed (`post_feedforward_layernorm_2`);
- the two outputs added, normed again (`post_feedforward_layernorm`) and added to the residual.

There are no per-layer embeddings and no KV sharing (unlike E2B / E4B). The vision tower (27
layers of 1152) standardizes its pooled output (`std_bias`, `std_scale`), and with
`use_bidirectional_attention: "vision"` (also 12B / 31B) the soft tokens of one image attend to
each other both ways on the sliding layers (global layers stay causal; a prefill chunk never
splits an image). Its chat template closes an empty thought channel in the generation prompt
when thinking is off (`<|turn>model\n<|channel>thought\n<channel|>`), which oxinfer reproduces.

The routed experts share the Qwen MoE machinery (`moe.rs`): decode batches the 8 experts' input
projections into one matmul, prefill runs all experts as one grouped matmul (on the matrix units
on M5 / A19), `--quant q8|q4` and `--save-quantized` quantize them (a Q4 copy is 13 GB, text only
like the E2B / E4B copies), and `--stream` reads them on demand from storage (with
`--cache-budget-gb 12`: 4.8 GB of dense weights resident, 46 GB of experts read as routed, 5.4
tok/s with the experts mostly in the page cache). The same model loads from llama.cpp GGUF files (`gemma4` architecture, text
only: ggml-org, unsloth and bartowski publish them) and from `mlx-community/gemma-4-26b-a4b-it-4bit`
(4-bit `switch_glu` experts, the router in 8 bits).

```sh
./target/release/oxinfer --model google/gemma-4-26B-A4B-it --mmap -p "Write a haiku about the ocean."
./target/release/oxinfer --model google/gemma-4-26B-A4B-it --mmap --image photo.jpg -p "What is in this picture?"
./target/release/oxinfer --model ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_0 -p "Hello"
./target/release/oxinfer --model mlx-community/gemma-4-26b-a4b-it-4bit -p "Hello"
```

Agreement with transformers (float32 compute on the bf16 weights; five chat prompts of 21-1758
tokens, the longest well past the sliding window, and one image question; last-token logits, then
the 2-32 greedy tokens oxinfer generated fed back to the reference, which must pick each of them):

| oxinfer | max logit diff | KL | same top 1 | other |
|---|---|---|---|---|
| bf16, CPU | 1e-4 - 3e-4 (image: 0.03) | < 4e-9 (image: 6e-7) | 6/6 | same top 10 in the same order; 147/147 greedy tokens |
| bf16, Metal (half-precision matrix units) | 0.004 - 0.44 | < 1.3e-4 | 6/6 | |
| `--quant q8` | 0.38 - 1.1 | 7e-11 - 3.8e-3 | 6/6 | |
| `--quant q4` | 4.8 - 15 | 2e-8 - 0.04 | 6/6 | |
| GGUF Q8_0 (ggml-org) | 0.37 - 1.1 | 1e-11 - 4e-4 | 5/5 | text only |
| GGUF Q4_0 (ggml-org) | 11 - 26 | 2e-6 - 0.07 | 5/5 | text only |
| MLX 4-bit (mlx-community) | 5.5 - 14 | 3e-10 - 0.11 | 6/6 | |

(Every row against the same bf16-weight reference, so the quantized rows measure the quantization;
the 1758-token prompt is the hardest.) One trap when building that reference: transformers loaded
in bf16 rounds the embedding scale buffer, sqrt(2816) = 53.07, to 53.0, which on its own moves the
router logits by 3e-3 and flips expert choices; the float32 reference restores it. With it
restored, every one of the 750-4170 routing decisions per prompt is the same in both.

On the Q8_0 GGUF, oxinfer and llama.cpp (CPU) pick the same next token on all five prompts; their
top-20 log-probabilities differ by 0.2-1.9, but against the float32 forward pass on the bf16
weights oxinfer is the closer of the two on every prompt (0.05-0.66 against 0.25-1.90), so the gap
is llama.cpp's int8 activations rather than a different reading of the file. The GGUF tokenizer,
rebuilt from the vocabulary and merges, gives the same ids as `tokenizer.json` (this README, ten
languages, emoji, special tokens).

Speed on an M5 Max (18 threads, `--mmap`, the page cache warm; prefill of the 1758-token prompt,
decode of 256 tokens of code; median of 2 runs while other jobs kept the load average near 4.5, so
lower bounds):

| weights | device | prefill tok/s | decode tok/s |
|---|---|---|---|
| bf16 (7.6 GB read per token) | CPU | 152 | 30 |
| bf16 | Metal (default) | 544 | 30 |
| `--quant q8` | CPU | 139 | 48 |
| `--quant q8` | Metal | 571 | 48 |
| `--quant q4` | CPU | 136 | 68 |
| `--quant q4` | Metal | 576 | 70 |
| GGUF Q8_0 (2046-token prompt) | CPU / Metal | 133 / 605 | 44 / 45 |
| GGUF Q4_0 (2046-token prompt) | CPU / Metal | 130 / 603 | 61 / 60 |
| MLX 4-bit | CPU / Metal | 133 / 225 | 65 / 64 |

Decode runs on the CPU on either device (one token per pass reads the weights once, and the CPU
reaches the memory bandwidth). Prefill attention runs as GEMMs (on the GPU for a chunk that starts
the context), which took the Metal prefill from 325 to 540 tok/s and the CPU one from 117 to 150.
The MLX experts stay on the CPU in prefill (the GPU's grouped expert kernels take plain Q4, not
MLX's affine form), hence its lower Metal number. An image costs about 0.8 s in the vision tower
on Metal (its 4304-wide MLP is zero-padded to whole GPU tiles), 5 s on the CPU.

Against llama.cpp on the same GGUF files (`tools/bench/bench.py`: same prompt tokens, greedy,
llama.cpp through `llama-server`, oxinfer as a fresh process per run; M5 Max at load 4-5; median of
2 runs for Q4_0, 1 for Q8_0):

| file | engine, device | prefill 512 | prefill 2046 | decode (256 tokens of code) |
|---|---|---|---|---|
| Q4_0 | llama.cpp, Metal | 3230 | 3006 | 122 |
| Q4_0 | oxinfer, Metal | 564 | 603 | 60 |
| Q4_0 | llama.cpp, CPU | 144 | 127 | 57 |
| Q4_0 | oxinfer, CPU | 138 | 130 | 61 |
| Q8_0 | llama.cpp, Metal | 2912 | 2888 | 94 |
| Q8_0 | oxinfer, Metal | 512 | 605 | 45 |
| Q8_0 | llama.cpp, CPU | 148 | 130 | 45 |
| Q8_0 | oxinfer, CPU | 148 | 133 | 44 |

On the CPU the two are level; with the GPU, llama.cpp runs the whole pass there (5x the prefill,
2x the decode) while oxinfer keeps attention, routing and decode on the CPU. The greedy outputs
agree: Q4_0 identical for 64 tokens on both devices, Q8_0 splitting at a near tie after 31-48.

### Qwen3.5 / Qwen3.8

The Qwen3.5-family dense models are a different architecture (`qwen.rs`): 48 Gated DeltaNet layers
(a causal depthwise conv, L2-normalised keys/queries and a 128x128 delta-rule recurrent state per
value head) interleaved with 16 gated full-attention layers (24 query / 4 KV heads of 256, partial
RoPE on 64 dims, a sigmoid output gate), SiLU MLPs and zero-centred RMSNorm. The recurrent layers
keep a fixed-size state instead of a KV cache, so memory does not grow with context for 48 of the
64 layers. The vision tower takes images and videos (see
[Image, video and audio input](#image-video-and-audio-input)); the MTP head drives speculative
decoding with `--mtp` (see Speculative decoding; the default for Qwen3.5-4B and Qwen3.5-9B, see
[DSpark drafts for Qwen and Gemma 4](#dspark-drafts-for-qwen-and-gemma-4) for how it compares).

The MoE variants (`qwen3_5_moe`, Qwen3.5/3.6-35B-A3B) replace each MLP with 256 routed SiLU
experts of width 512 (softmax router, top 8, weights renormalised) plus a shared expert scaled by
a sigmoid gate. Experts load from the fused `experts.gate_up_proj` / `down_proj` tensors or from
per-expert `experts.N.*_proj` weights (as transformers saves them), and quantize like any other
matrix (`--save-quantized` included). Decode batches the 8 chosen experts' input projections into
one matmul. Prefill gathers each expert's tokens and runs every expert as one grouped matmul on
the CPU, or on M5 / A19 as one GPU submission per layer on the matrix units. With `--quant q4`
(about 19 GB) on an M5 Max, that is about 220 tok/s of prefill with the experts on the CPU,
2.1-2.3x that with the matrix units (see [Performance](#performance)), and 65 tok/s of decode.
On randomly initialised weights at the real width (2048 hidden, 256 experts, 2
layers, the real vocabulary), oxinfer matches transformers in float32 to a max logit difference of
7e-5, with the same 20 greedy tokens. On Zawen-v0.1-35B-A3B's real weights, cut to its first 4
layers, the float32 match is 3e-5. The full 40 layers against transformers in bf16 give a mean
logit difference of 0.035, with the same top 10 in the same order (Q8: same top 10; Q4: same top
token). MLX-quantized Qwen experts are not read yet (Gemma 4's are).

Thinking is on by default in the Qwen template; `--think` enables it here (the trace is dimmed),
otherwise the generation prompt closes the think block.

Text, images, video and audio are supported (E2B/E4B have both towers).

#### DSpark drafts for Qwen and Gemma 4

`--draft` also takes a DSpark draft for Qwen3, Qwen3.5 / 3.8 and Gemma 4 targets: a Hugging Face
draft folder or repo id (DeepSpec `Qwen3DSparkModel` / `Gemma4DSparkModel`, SpecForge / RadixArk
`DSparkDraftModel`, vLLM speculators, NVIDIA Model Optimizer, z-lab `DFlashDraftModel` with a
Markov head) or a GGUF: llama.cpp's `dflash` files (as `convert_hf_to_gguf.py` writes them, or
BeeLlama's, e.g. `Anbeeld/Qwen3-14B-DSpark-GGUF`) and the `dspark` files of llama.cpp's DSpark
branch (`ankk98/dspark-qwen3-8b-block7-Q4_K_M-GGUF`, `ankk98/dspark-gemma4-12b-block7-Q4_0-GGUF`,
which llama.cpp master does not load). A repository holding several GGUF quantizations gives its
`Q8_0` file. The draft is a few grouped-query attention layers (Gemma 4: K = V attention, value
norm, sandwich norms, layer scalars, proportional RoPE, GELU) that read the target's residual
streams at a few layers, plus the Markov head and the optional confidence head
(`--spec-draft-conf-min C` cuts a block where its estimate falls below C). A draft as wide as its
target uses the target's embeddings and LM head (the drafts ship frozen copies; a GGUF's copy may
be quantized further), and the draft's weights are quantized like the target's (a Q4 draft for a
Q4 target: as many drafts accepted as with the bf16 draft on Qwen3.8-27B, and its passes 23-27 ms
instead of 31-34). Greedy output is the plain greedy output token for token; sampling keeps the
target's distribution.

**How a draft is found.** `--draft PATH` (or a repo id, `org/name:Q8_0` for one file of a GGUF
repository) uses that draft. Without `--draft`, a draft is used when one is on disk and the pair
is a default (the table below): a `dspark*` draft folder or GGUF inside the model folder (or a
`dspark*.gguf` beside a GGUF model), or a known draft of the model (`dspark::KNOWN`) in the
download cache, i.e. an `org--name` folder next to the model's (the model is recognized by its
folder name, a converted copy's `name-oxinfer-q4`, a GGUF repository's `name-GGUF`, or a GGUF file
name up to its quantization). Nothing is downloaded implicitly; when a default draft is known but
missing, the CLI says so. `--draft auto` takes any draft found that way (default or not) and,
when there is none, fetches the model's known draft (its Hugging Face original) into the cache:

```sh
./target/release/oxinfer --model Qwen/Qwen3-8B-GGUF:Q8_0 --draft auto -p "..."  # fetches deepseek-ai/dspark_qwen3_8b_block7 once
./target/release/oxinfer --model Qwen/Qwen3-8B-GGUF:Q8_0 -p "..."               # then uses it by default
./target/release/oxinfer --model Qwen/Qwen3-8B-GGUF:Q8_0 --no-draft -p "..."    # plain
```

Library: `EngineOptions::draft` (`auto`, a path or repo id, `none`; unset for the default) and the
C API key `draft`; `dspark::draft_for`, `default_draft`, `known_drafts`, `auto_repo`, `draft_in`
and `engine::resolve_auto_draft` do the search.

**Each round** (unless `--spec-draft-fixed`, or when the model streams) is a draft pass or a plain
step, whichever has given more tokens per second lately (measured, the first two samples of each
kind discarded as warm-up). While plain steps win, a draft round is still tried after 6 plain
rounds, then twice as long apart after each such probe that loses (up to 96). A draft pass costs
the same for any number of drafts used, so it always drafts the whole block; then it verifies the
part of the block with the most expected tokens per second: the cost of verifying each length is
measured (interpolated where not yet measured), and the chance that each draft is accepted (when
the ones before it were) is estimated from its probability under the draft model, calibrated
online by tenths (decayed counts of verified drafts and accepted ones, with the draft's own
probability as a prior). The cut depends on the drafts only, so sampling keeps the target's
distribution. The draft's context (keys and values of the committed positions) is computed
lazily, in one pass before the next draft: rounds without a draft add no draft work.

**On the GPU** (Metal, GPU decoding on), the verification is one decode pass for all its tokens,
each token's arithmetic exactly the single-token kernels' (the output is the plain GPU greedy
output bit for bit). The multi-token matrix-vector kernels take the tokens 4 at a time over 2
rows per threadgroup: each row chunk's quants are decoded to floats once (Q4, Q4_0, Q8, Q8_0,
bf16, f16; the products and sums are then the single-token kernel's, in its order), each token's
inputs loaded once for both rows. For 8 tokens of a 17408 x 5120 matrix that is ~400 us at Q8 and
~500 us at Q4 (one token: 180 / 100 us; before: 620-1000 us), an 8-token verification of
Qwen3.8-27B at Q4 ~150 ms instead of 205-280 (a plain step: 30 ms). MoE targets verify in one
pass too (Qwen3.5-35B-A3B: 9 tokens in 30-45 ms, a plain step 6-8 ms; Gemma 4 26B-A4B's graph
takes the same path, untried for lack of a draft): the attention, recurrent and shared parts over
all the tokens, each token's routing and experts with the single-token kernels (tokens route to
different experts). A round without drafts runs a one-token plan (the plain decoding kernels plus the
feature copies), and the plans of a model share their device copies of the caches. The draft runs
as two such passes (its context and its block).

Defaults (Metal, M5 Max): a pair is on by default where it was faster than plain decoding on
average over 8-12 prompts (code, prose, chat, math, 200 tokens each). The numbers below are exact
verification; "fast" pairs are on only with [fast verification](#fast-verification) (the
default), where they gain (see its table):

| target | draft | default | plain -> DSpark, mean tok/s (code / prose / chat / math) | load |
|---|---|---|---|---|
| Qwen/Qwen3-4B (bf16) | deepseek-ai/dspark_qwen3_4b_block7 | yes | 52.2 -> 107.2 (48 -> 115 / 53 -> 71 / 53 -> 99 / 55 -> 145) | 13-15 |
| Qwen/Qwen3-8B (Q8_0 GGUF) | deepseek-ai/dspark_qwen3_8b_block7 | yes | 60.0 -> 78.7 (61 -> 95 / 59 -> 58 / 60 -> 60 / 60 -> 101) | 6-9 |
| Qwen/Qwen3-8B (Q8_0 GGUF) | ankk98/dspark-qwen3-8b-block7-Q4_K_M-GGUF | yes | 60.0 -> 77.9 | 6-9 |
| Qwen/Qwen3-14B (Q8_0 GGUF) | deepseek-ai/dspark_qwen3_14b_block7 | yes | 31.5 -> 44.4 (34 -> 56 / 34 -> 33 / 34 -> 39 / 25 -> 50) | 5-14 |
| Qwen/Qwen3-14B (Q8_0 GGUF) | Anbeeld/Qwen3-14B-DSpark-GGUF (Q8_0) | yes | 31.5 -> 43.5 | 5-14 |
| google/gemma-4-12B-it (q8) | deepseek-ai/dspark_gemma4_12b_block7 | yes | 34.6 -> 43.1 (34 -> 48 / 34 -> 36 / 35 -> 37 / 36 -> 52) | 11-16 |
| google/gemma-4-12B-it | ankk98/dspark-gemma4-12b-block7-Q4_0-GGUF | no | Q4_0 throughout: fewer drafts accepted (4.3 tokens per round on a code prompt vs 5.2) | |
| Qwen3.8-27B (q4) | RadixArk/Qwen3.8-27B-DSpark | fast | 30.6 -> 29.6 (31 -> 31 / 31 -> 29 / 30 -> 29 / 30 -> 30) | 6-12 |
| Qwen3.8-27B (q4) | jialinyyzz/Qwen3.8-27B-DSpark-drafter | no | 30.6 -> 29.5 | 6-12 |
| Qwen3.5-35B-A3B (q4) | jinzex/Qwen3.5-35B-A3B-DSpark | no | 126.9 -> 85.6 (125 -> 89 / 128 -> 84 / 136 -> 85 / 119 -> 84) | 13-24 |
| google/gemma-4-31B-it (q4) | RedHatAI/gemma-4-31B-it-speculator.dspark | fast | 23.2 -> 22.4 (21 -> 21 / 24 -> 23 / 24 -> 22 / 25 -> 23) | 12-24 |
| google/gemma-4-31B-it (q4) | Hikari07jp/DSpark-Gemma-4-31B-draft | no | 23.2 -> 21.6 | 12-24 |

All the outputs above are the plain greedy outputs, token for token (both GGUF and Hugging Face
drafts). Where the draft is not a default, the policy decodes mostly plainly and the loss is its
probes: verifying 8 tokens costs ~5 plain steps at Q4 (the multi-token kernels must reproduce the
single-token arithmetic, see the llama.cpp comparison below) against ~2-3 at Q8 / bf16, so the Q4
targets only gain on code and math. Qwen3.5-35B-A3B decodes a token in 6-8 ms; a pass over 9
tokens costs 30-45 ms (each token's 8 routed experts are its own) and a draft pass 15 ms, so even
its code rounds (5 tokens, 65% of the drafts accepted) lose, and on prose the plain rounds run at
plain speed while 6 probes (75 ms each) cost the 30%.

Agreement:

| target | draft | agreement with the float32 reference |
|---|---|---|
| Qwen/Qwen3-4B (bf16) | deepseek-ai/dspark_qwen3_4b_block7 | rel. logit error 2-2.5e-6 (CPU), 1.2e-4 (Metal); same drafts |
| Qwen3.8-27B | RadixArk/Qwen3.8-27B-DSpark | 2-4e-6; same drafts |
| Qwen3.8-27B | jialinyyzz/Qwen3.8-27B-DSpark-drafter | 1.5-2.3e-6; same drafts |
| google/gemma-4-12B-it | deepseek-ai/dspark_gemma4_12b_block7 | 1.8-3.4e-6; same drafts |
| google/gemma-4-31B-it | RedHatAI/gemma-4-31B-it-speculator.dspark | 2.6-9.1e-6; same 8 drafts |
| google/gemma-4-31B-it | Hikari07jp/DSpark-Gemma-4-31B-draft | 1.0-2.3e-6; same 15 drafts |
| Qwen/Qwen3.5-35B-A3B | jinzex/Qwen3.5-35B-A3B-DSpark | 1.6-2.7e-6; same 8 drafts |

(bf16 targets on the CPU; the reference takes the target's embeddings and head in bf16, so with
a Q4 target, whose head is Q8, the draft logits differ by 4-13% while the drafts stay the same.)

GGUF drafts against the original draft (same target features, all 7 slots, chained on each
draft's own tokens): llama.cpp master's conversion of the Gemma 4 12B draft at F16 matches to
2-3e-7 relative (KL < 1e-12; the converter needed two local fixes: transformers 5.23 drops
`global_head_dim`, and it refuses the draft's logit soft-cap, which the `dflash` graph supports),
at Q8_0 to 0.6-1.4e-3 (KL < 1e-5); the Qwen3-8B conversion at Q8_0 to 4-7e-3, Anbeeld's Qwen3-14B
Q8_0 to 4-6e-3, ankk98's Qwen3-8B Q4_K_M to 5-9e-2 (KL < 6e-3), all with the same drafts; ankk98's
Gemma 4 12B "pure Q4_0" to 0.12-0.17 (same drafts on the target's head; with its own Q4_0 head the
first draft differed).

The reference is `tools/ref_dspark_hf.py` (DeepSpec's modeling code in float32, fed the target
features oxinfer computes); `examples/dspark_check.rs` writes the features and the draft logits
and compares greedy output with and without the draft.

**DSpark or MTP.** Qwen3.5 / 3.8 models also have their own multi-token-prediction layer
(`--mtp`, see [Speculative decoding](#speculative-decoding)). Its verification now runs as the
same GPU passes (it used the CPU's multi-token kernels, so its greedy output followed CPU decoding
rather than plain GPU decoding; it is now plain GPU decoding token for token), and its draft head on
the GPU. Same prompts, 200 tokens, greedy, tok/s (tokens per verification, drafts accepted; the
Qwen3.8-27B copy converted to Q4 takes its MTP layer from the original with `--mtp-from`):

| model | plain | DSpark (default policy) | MTP (3 drafts, adaptive) | default |
|---|---|---|---|---|
| Qwen3.8-27B (q4; load 6-12) | code 30.8, prose 31.1, chat 30.3, math 30.3 | 31.2, 29.1, 28.8, 29.6 (RadixArk) | 35.6 (3.3, 82%), 25.7 (2.3, 53%), 26.3, 34.3 | plain |
| Qwen3.5-35B-A3B (q4; load 13-24) | 125, 128, 136, 119 | 89, 84, 85, 84 | 109 (3.2), 82 (2.2), 81, 104 | plain |
| Qwen3.5-4B (bf16; load 10-13) | 55.8, 50.8, 61.2, 61.1 | (no draft) | 76.0 (3.1), 65.2 (2.2), 71.5, 92.3 | MTP |
| Qwen3.5-9B (q8; load 8-11) | 57.2, 56.5, 56.8, 57.8 | (no draft) | 81.3 (3.3), 58.9 (2.2), 63.2, 88.3 | MTP |

MTP drafts one token per step (each step reads its decoder layer and the LM head), DSpark a
whole block per pass; on code and math MTP accepts as much as DSpark at a lower cost per round, but
it has no plain fallback, so it loses on prose and chat, where its acceptance drops to ~50%. MTP is therefore
the default for Qwen3.5-4B and Qwen3.5-9B (`qwen_mtp::BY_DEFAULT`; `--no-draft`, engine
`draft=none`, decodes plainly), and neither for Qwen3.8-27B nor Qwen3.5-35B-A3B at Q4. With CUDA
decoding, neither MTP nor DSpark is a default: the drafts run on the CPU, and plain GPU decoding
is faster (see [CUDA](#cuda)); `--mtp` / `--draft` still use them.

Against llama.cpp (master of 2026-11-02, `29c720eaf7`, `llama-server`, `--spec-type draft-dspark`,
every layer on the GPU; the same GGUF target and draft files, the drafts converted by
`convert_hf_to_gguf.py`; 12 prompts, 200 tokens, greedy, thinking off; tok/s means, drafts
accepted in parentheses, load 5-13):

| target, draft | plain llama.cpp / oxinfer | llama.cpp, 7 drafts | llama.cpp, 3 drafts | oxinfer (default policy) |
|---|---|---|---|---|
| Qwen3-8B Q8_0, dspark_qwen3_8b Q8_0 | 61.6 / 60.9 | 168.3 (code 213, prose 110; 31-75%) | 119.5 (50-90%) | 77.5 (code 97, prose 61; 59-89%) |
| gemma-4-12B-it Q8_0, dspark_gemma4_12b Q8_0 | 36.0 / 36.5 | 81.9 (code 126, prose 58; 27-64%) | 69.5 (51-84%) | 43.6 (code 54, prose 37; 53-81%) |

Plain decoding runs at the same speed, and the drafts agree (oxinfer accepts more of them since it
cuts unlikely ones), but llama.cpp verifies 8 tokens for little more than a plain step (its prose
runs at 110 tok/s with 31% of the drafts accepted, against 62 plain): its batched kernels reorder
the arithmetic (a token's logits depend on the batch), where oxinfer's verification must
reproduce the single-token kernels bit for bit to keep greedy output identical to plain decoding
(an 8-token verification costs ~2 plain steps at Q8, ~5 at Q4). Fast verification (below, the
default) lifts that constraint.

#### Fast verification

On by default; `--exact-verify` (engine / C API / server key `fast_verify=0`;
`DsparkOptions::fast_verify`, `MtpOptions::fast_verify` false) turns it off, so that greedy
output is the plain greedy output token for token (the Q4 drafts marked "fast" above are then
not used by default either). The multi-token GPU
passes (the verification, and the DSpark draft's own passes) then run their matmuls on batched
kernels (`metal_fastv.metal`): 8x8 f32 simdgroup-matrix products over all the tokens, the
weights dequantized straight into the lanes' matrix fragments (each 32-value block's K order
permuted alike for the weights and the inputs, so nothing goes through threadgroup memory), each
weight read once for the whole pass. An 8-token pass of a 17408 x 5120 matrix then costs
1.15x one token at Q8_0 (189 vs 164 us; the exact kernels ~400 us), 1.1-1.2x at bf16, 1.4x at
Q4 (16 tokens: 1.2-2.2x). Formats: bf16, Q8, Q4 (affine too), F16, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K;
others keep the exact kernels. Everything else (attention, norms, the MoE experts) is unchanged.

The sums are taken in another order, so the logits differ from one-token decoding in rounding,
and greedy output may drift from plain greedy output at a near-tie; sampling still draws from
the model's distribution up to that rounding. In practice the difference is f32 rounding noise
(`examples/dspark_check.rs --gen 200 --fast --drift`: the plain greedy output fed through
8-token passes, fast vs exact, each with its own caches):

| target | max \|logit diff\| (mean / max over 200 positions) | KL (mean / max) | top-1 changes |
|---|---|---|---|
| Qwen3-8B Q8_0 | 3.0e-5 / 6.5e-5 | 4e-12 / 1.4e-10 | 0 |
| gemma-4-12B-it (q8; prose / math) | 6.1e-5 / 3.4e-4, 6.1e-5 / 1.8e-4 | 1e-11 / 1.3e-10 | 0 |
| gemma-4-31B-it (q4) | 1.1e-4 / 3.9e-3 | 3e-12 / 2.2e-10 | 0 |
| Qwen3.8-27B (q4) | 1.7e-5 / 4.7e-5 | 6e-12 / 6.2e-11 | 0 |

Every greedy output of the table below (8 prompts x 7 models, 200 tokens) is the plain greedy
output token for token, and so are two 1000-token Qwen3-8B outputs (587 and 992 tokens).

Speed (M5 Max, 8 prompts: code, prose, chat, math, 200 tokens, greedy, `--no-system`; tok/s
means; plain / exact / fast interleaved prompt by prompt, 2 repetitions, 1 for the Q4 models;
llama.cpp from the table above, Q8_0 GGUF target and draft, 7 drafts):

| target, draft | plain | exact | fast (default) | llama.cpp | load |
|---|---|---|---|---|---|
| Qwen/Qwen3-4B (bf16), dspark_qwen3_4b | 63.9 | 112.9 | 145.5 (code 176, prose 111) | | 1.3-3.2 |
| Qwen/Qwen3-8B (Q8_0 GGUF), dspark_qwen3_8b | 63.1 | 84.2 | 125.5 (code 131, prose 95) | 168.3 | 1.7-3.8 |
| google/gemma-4-12B-it (q8), dspark_gemma4_12b | 38.8 | 52.6 | 92.6 (code 107, prose 59) | 81.9 | 2.1-8.3 |
| Qwen/Qwen3.5-4B (bf16), MTP | 62.9 | 86.7 | 85.9 | | 2.4-6.6 |
| Qwen3.8-27B (q4), RadixArk DSpark | 34.5 | 32.9 | 44.5 (code 57, prose 35) | | 1.4-4.3 |
| google/gemma-4-31B-it (q4), RedHatAI speculator | 27.4 | 25.9 | 46.2 (code 55, prose 29) | | 2.8-6.3 |
| Qwen3.5-35B-A3B (q4), jinzex DSpark (not a default) | 169.3 | 122.7 | 126.3 | | 3.6-7.2 |

Fast verification makes the Q4 dense drafts pay (Gemma 4 31B +69%, Qwen3.8-27B +29% over plain;
both drafts are therefore defaults with it),
and closes most of the gap to llama.cpp on Gemma 4 12B; on Qwen3-8B the rest is the draft's own
passes (~9 ms a round against ~19 ms for an 8-token verification, plain step 16 ms: the Markov
head's 7 chained submissions alone take 2 ms). MTP on Qwen3.5-4B gains nothing (its 4-token bf16
verification already cost about one step), nor does Qwen3.5-35B-A3B (each token's 8 routed
experts are still read per token).

### Mistral Large 4 (1T-A52B)

`mistralai/Mistral-Large-4-1T-A52B` (`mistral4.rs`) is a DeepSeek-V3.2-style MoE with Mistral's
own twists, 72 layers of 6144:

- MLA attention without a query LoRA: 64 heads of 128 (64 rotary dims), keys and values
  expanded from a 512-wide latent; only the latent and the shared rotary key are cached (2.3 KB
  per token and layer). Decoding scores against the latent (`kv_b_proj` absorbed into the query
  and the output, as llama.cpp does); prefill expands the keys and values and runs ordinary
  attention as GEMMs, on the GPU for a chunk that starts the context.
- Every fourth layer has no RoPE (the whole head is non-rotary), and layers 39, 43, ..., 71
  have no KV projection of their own: they read layer 35's latent cache through their own
  `kv_b_proj` (no cache stored for them).
- Queries scaled per head and position by `s_h ln(pos + 1) + b_h` ("SSMax"); RoPE on
  interleaved pairs with YaRN frequencies (factor 2048 over 512) times learned factors
  ("LeRoPE", `exp(alpha)`), without YaRN's magnitude scaling.
- DSA sparse attention: an indexer (32 heads of 128, LayerNorm'd, rotary on its first 64 dims)
  scores every earlier token as `sum_h w_h relu(q_h . k)`, and each query attends only to its
  2048 best keys; "shared" layers reuse the previous indexer's choice. The indexer keeps its own
  key cache, and the choice is made exactly as the reference (contexts up to 2048 tokens are
  dense attention).
- A dense first layer, then 128 softmax-routed SiLU experts of 6144 (top 4, renormalized) plus a
  shared expert; untied LM head with logits soft-capped at 30.

Checkpoints: the official one (Mistral's `consolidated-*.safetensors` layout, FP8 E4M3 weights
with 128x128 block scales, read as stored, zero-copy, through a `GgmlType::F8E4M3` with block
scales; prefill GEMMs and the grouped MoE experts run on the GPU, see
[FP8 on the GPU](#fp8-on-the-gpu)), transformers-saved copies, and llama.cpp GGUF files (`mistral-large4`, split into parts
or not; unsloth's are tested). The Pixtral vision tower (30 layers of 2048, 2D RoPE, 2x2 patch
merger and GELU projector) comes from the checkpoint, or for a GGUF model from the `mmproj` file
beside it (in its folder or the one above). `--mmproj FILE` (engine / C option `mmproj=FILE`, a
path or `org/repo/file`; `EngineOptions::mmproj`) names the projector instead, for a GGUF model
kept elsewhere or to replace a checkpoint's own tower, and `--mmproj none` (`mmproj=none`)
turns image input off. The chat format is the model's template:
`[SYSTEM_PROMPT]`, `[MODEL_SETTINGS]{"reasoning_effort": ...}` (`high` with `--think`, `low`
otherwise; the model thinks in `[THINK]...[/THINK]` either way, shown dimmed), `[INST]` turns,
`[AVAILABLE_TOOLS]` / `[TOOL_CALLS]` tool calling, images as rows of `[IMG]` closed by
`[IMG_BREAK]` / `[IMG_END]`. Speculative decoding uses the checkpoint's DSpark draft head (see
[below](#dspark-speculative-decoding)); with unsloth's GGUF it is on by default.
[Runtime interventions](#runtime-interventions-neuron-edits-steering-abliteration-sae-features)
work as for the other families: residual hooks around every layer, and neuron edits of layer 0's
dense MLP, the shared expert or any routed expert.

```sh
# unsloth's 378 GB UD-Q2_K_XL on a 128 GB Mac: the folder of parts (or any part, or the repo id)
./target/release/oxinfer --model unsloth/Mistral-Large-4-1T-A52B-GGUF:UD-Q2_K_XL -p "Why is the sky blue?"
./target/release/oxinfer --model ~/.cache/oxinfer/hub/unsloth--Mistral-Large-4-1T-A52B-GGUF/UD-Q2_K_XL \
    --image photo.jpg -p "Describe this image in two sentences."
# the official FP8 checkpoint (1.06 TB)
./target/release/oxinfer --model mistralai/Mistral-Large-4-1T-A52B -p "Hello"
```

A checkpoint larger than 60% of RAM is streamed (`--stream`, see
[Streaming weights from flash](#streaming-weights-from-flash-models-larger-than-ram)) with a page
cache budget of 70% of RAM unless told otherwise (`--stream`, `--cache-budget-gb`; `--no-stream`,
or `--mmap` alone, asks for plain demand paging). The engine / C option is tri-state:
`stream=auto` (the default, as above; setting a tuning key such as `cache_budget_gb` also turns it
on), `stream=on` / `1`, and `stream=off` / `0`, which really means a plain memory map
(`EngineOptions::stream_mode`, `StreamMode::{Auto, On, Off}`). Streamed, the 20 GB of dense weights stay resident, the routed experts
(357 GB) are read as they are routed and the rest of the budget caches the recent ones (about 60%
of the routed experts are then hits). In prefill, a layer's routed experts run on the GPU's
matrix units through per-call device state (`Backend::upload_moe_transient`, nothing kept
resident) that wraps only the experts the chunk routes to: a wrapped page must be in memory
while the submission runs, so wrapping all 128 made every short prompt read the whole 357 GB
(through page faults at 2-3 GB/s, 70-120 s for 10-30 tokens). A chunk of a few hundred tokens
still routes to nearly all experts, so longer chunks amortize the sweep. Nothing pins the model
in memory: the process footprint stays under 1 GB while decoding (3-5 GB in prefill), the page
cache holds the rest.

Prefill by prompt length, from a cold expert cache (each run starts with only the dense weights
cached; `--no-system`, plain decoding; load average 9-17 from other jobs, another agent's
llama.cpp holding a DeepSeek model in memory part of the time). Before: the experts wrapped
whole; the GPU is now faster than the CPU at every length, and the output is unchanged (the last
position's logits are bit-identical before and after; against the CPU path the usual matrix-unit
difference, KL 5e-9 at 24 tokens and 5e-8 at 103, same top token):

| prompt tokens | Metal before | Metal now | storage read | `--device cpu` |
|---|---|---|---|---|
| 10 | 117 s (0.1 tok/s) | 8.9 s (1.1 tok/s) | 67 GB | 12.4 s (0.8 tok/s) |
| 24 | 83 s (0.3) | 14.4-14.8 s (1.7) | 120 GB | 22.6 s (1.1) |
| 103 | 72 s (1.4) | 34-35 s (3.0) | 255 GB | 52.9 s (1.9) |
| 329 | 68 s (4.9) | 40-45 s (7.4-8.2) | 317 GB | 143 s (2.3) |
| 951 | 56 s (17.1) | 53 s (18.0) | 343 GB | 3.0 tok/s at 895 (earlier) |

Prefill now waits on storage 73-84% of the time at 7-9 GB/s. More reader threads help it
(`--stream-threads 8`: 103 tokens in 26 s instead of 34, one measurement under the same load;
their effect on decoding was lost in the noise of that load, so the default stays 2). Guessing
the next layer's experts during prefill (as `--expert-predict` does for decoding) gained nothing
measurable (24 tokens 15.9 s vs 14.8, 103: 31.5 vs 35.4, 329: the same), so it was left out.

Speed on the M5 Max (128 GB, 18 cores, SSD at ~13 GB/s; other jobs used 3-4 cores, so lower
bounds), unsloth's UD-Q2_K_XL:

| setting | prefill | decode |
|---|---|---|
| default with DSpark (`--spec-draft-p-min 0.85`), measured 2026-11-02 | 1.5-2.1 tok/s (27-50 tokens, 2026-11-03; was 0.4-0.7, see above) | 1.34-1.36 tok/s (plain: 1.26), see [DSpark](#dspark-speculative-decoding) |
| same, `--device cpu` | 1.7 tok/s (50 tokens) | 1.48 tok/s (plain: 1.35) |
| plain decoding (streaming, 96 GB budget, Metal), earlier measurement | 17.4 tok/s (895-token prompt), 7.5 tok/s (1406, two chunks), 1-2 tok/s (15-90 tokens) | 1.3-1.5 tok/s |
| same, `--device cpu` | 3.0 tok/s (895 tokens) | 1.1-1.2 tok/s |
| `--mmap` (demand paging, no streaming) | 0.4 tok/s (15 tokens) | 0.3-0.4 tok/s |
| `--expert-predict` (next layer's experts guessed and read ahead, 73% right) | | 1.4 tok/s: IO stalls fall from 60% to 27% of the time, but the reads slow the compute as much |
| llama.cpp (same build of its master, CPU, `-ngl 0`, mmap) | 0.2-0.27 tok/s | 0.25 tok/s |

A decode step reads 31 GB of weights: the resident 20 GB at ~120 GB/s and ~4 GB of routed experts
from storage (60% of the time is waiting on those reads).

Agreement. transformers (main, `modeling_mistral_large4.py`) in float32 on tiny random
checkpoints (8 layers of 256, 4 heads, 16 experts, a dense first layer, NoPE layers, a KV-sharing
layer, full and shared indexer layers, `index_topk` 16 so most queries are sparse, a 2-layer
Pixtral tower), against every position of a prompt and 8-16 greedy decode steps fed back
(`tools/mistral4_tiny.py` builds them, `tools/ref_mistral4.py` runs the reference,
`examples/mistral4_check.rs` compares):

| checkpoint | prefill (max logit diff / mean KL) | decode | other |
|---|---|---|---|
| Mistral layout, bf16, 60 tokens | 3.8e-4 / 4.5e-10 | 4.9e-4 / 1.3e-9 | the same in chunks of 1, 7, 33, 60 |
| same, 300 tokens | 1.0e-2 / 4.0e-8 | 3.3e-3 / 5.9e-8 | every indexer choice (1500 rows) identical; 300/300 and 16/16 top-1 |
| transformers-saved layout | 3.8e-4 / 4.5e-10 | 4.9e-4 / 1.3e-9 | |
| FP8 blocks (Mistral layout, `qscale_weight`), vs the reference on the dequantized weights, 60 tokens | 7.5e-4 / 7.5e-10 | 3.5e-4 / 1.0e-9 | the same with the GEMMs and experts on Metal's f32 kernels (bit-identical to the CPU, also at 300 tokens) |
| GGUF F32 (llama.cpp's converter), 300 tokens | 3.6e-3 / 9.3e-9 | 1.2e-3 / 1.0e-8 | |
| GGUF with unsloth's types (Q8_0, Q6_K, IQ2_S, IQ3_XXS, IQ3_S), vs the reference on its dequantized weights, 300 tokens | 5.9e-3 / 1.1e-8 | see below | |
| image features (Pixtral + projector), 384x384 and 1700x900 images | rel. 1.4e-6 / 1.9e-6 | | prompt ids identical; last-token logits 5e-4 |
| same from the converted `mmproj` GGUF (f16) | rel. 1.2e-6 | | |

On longer contexts a random tiny model is chaotic: near-ties in the indexer (gaps of 2e-6
relative between the 16th and 17th key) or the router flip with float32 rounding and move later
logits by up to 8. Quantized GGUF decoding multiplies with int8 activations (as llama.cpp), 0.7%
relative per layer here, which such a model amplifies; with float activations (passes of 8+
tokens) every quantized type matches as above.

On the real UD-Q2_K_XL, against llama.cpp (its master, CPU) on the same file: same next token
on both prompts tried, top-20 overlap 20/20 and 19/20, log-probabilities of the top 5 within
0.08 (0.37 over the top 20; llama.cpp quantizes activations differently and oxinfer's prefill
here ran on the GPU's half-precision matrix units), and the same 24 greedy tokens ("the founding
of Rome in 753 BC to the fall of Constantinople in 1453 AD. It is one of"). The rebuilt GGUF
tokenizer (tekken, numbers in groups of three) matches `tokenizer.json` on this README. Chat is
coherent on both devices, across turns, and about images.

#### FP8 on the GPU

The FP8 weights keep their bytes on the GPU too: Metal kernels read the E4M3 bytes and the
128x128 block scales in place (no copy, no conversion at load) and dequantize per K step, an
E4M3 byte becoming a half exactly by moving its sign and its 7 exponent / mantissa bits into a
half's places (the 2^8 offset of the exponent bias goes into the block scale). The f32 GEMM (any
Apple GPU, or `gpu_exact=1`) dequantizes to f32 in threadgroup memory and gives the CPU's results
bit for bit; on M5 / A19 the matrix units (`matmul2d`) take the weights as half in threadgroup
memory with f32 accumulation, as for the other formats, in a dense and a grouped (MoE experts)
kernel. Other block shapes than 128x128 stay on the CPU. Checks: the tiny FP8 checkpoint gives
the CPU's logits exactly on the f32 kernels (60 and 300 tokens), and on the matrix units exactly
the logits of its dequantized copy (the same values stored as float32; so FP8 adds no error beyond the half operands, which on such a tiny random model already
move logits a lot, the bf16 checkpoint included); `cargo test` compares the dense and grouped kernels with the CPU (FP8 among the other
formats). Throughput (`KINDS=Q8_0,F8_E4M3 membench gpu 17408 5120`, copies and submission
included, measured while another job kept 9 CPU cores busy):

| Tokens | FP8 f32 GEMM | FP8 matrix units | Q8_0 matrix units (same run) | CPU (FP8) |
|---|---|---|---|---|
| 256 | 9.3 TFLOP/s | 15.6 | 17.3 | 1.2 |
| 512 | | 18.7 | 20.9 | 1.1 |
| 1024 | 10.9 | 20.5 | 21.9 | 1.3 |
| 2048 | 11.3 | 41.7 | 40.8 | 1.1 |

Decoding stays on the CPU (this family has no whole-pass GPU decode).

#### DSpark speculative decoding

The checkpoint's DSpark draft head (`dspark.rs`; llama.cpp's `--spec-type draft-dspark`; unsloth
ships it as `dspark-Mistral-Large-4-1T-A52B-Q8_0.gguf`, 2.2 GB, beside the GGUF folder) proposes
a block of 3 tokens per pass:

- Context: for every position the target has run, its residual stream entering layers 2, 19, 37
  and 55 and after the last layer, concatenated, projected by `fc` and normed; each of the 3
  dense MLA draft layers (6144 wide, 64 heads, latent K = V of 512 + 64 rotary dims, plain YaRN)
  caches its own latent of that, and the block attends to the last 2048 positions. The cache
  keeps only those plus 256 to 512 more (`dspark::SLACK`; 6.9 KB per position, 17.7 MB at
  most) instead of every position: rolling back up to 256 positions stays exact, and a deeper
  `DsparkSession::truncate` leaves the block what is left until new positions refill the window
  (drafts change, the output does not).
- Block: the last committed token (not yet run by the target) and two learned mask embeddings at
  the next positions, attending non-causally to the context and to each other, then the target's
  LM head, soft-capped at 30. Slot `i` predicts the token `i + 1` positions after the anchor.
- A recurrent Markov head adds a low-rank logit bias to each slot keyed on the previous token
  (`markov_w2 tanh(.)` of a gated state over `[state, markov_w1[prev], hidden]`), chained over the
  block on the tokens actually drafted.

The target verifies `[last, d1, d2, d3]` in one pass on the batch-invariant decode kernels, the
features of the committed positions join the draft's cache, and rollback truncates the latent
and indexer key caches (the KV-sharing layers keep none). Greedy output is the plain greedy
output token for token; sampling keeps the target's distribution (standard speculative
sampling). The same `dspark::Target` trait serves the DeepSeek V4 draft and the Qwen / Gemma 4
ones (see [DSpark drafts for Qwen and Gemma 4](#dspark-drafts-for-qwen-and-gemma-4)).

A GGUF model uses the `dspark*.gguf` beside it (its folder or the one above) by default:

```sh
M=~/.cache/oxinfer/hub/unsloth--Mistral-Large-4-1T-A52B-GGUF
./target/release/oxinfer --model $M/UD-Q2_K_XL -p "Why is the sky blue?"            # DSpark on
./target/release/oxinfer --model $M/UD-Q2_K_XL --no-draft -p "Why is the sky blue?" # plain
./target/release/oxinfer --model $M/UD-Q2_K_XL --draft $M/dspark-Mistral-Large-4-1T-A52B-Q8_0.gguf \
    --spec-draft-n-max 3 --spec-draft-p-min 0.85 -p "..."
```

`--spec-draft-n-max` (`--draft-tokens`) is at most the block, 3. `--spec-draft-p-min P` cuts a
block before the first draft the draft model gives less than probability P, 0.85 by default when
the model streams (0 otherwise). `--draft-device auto|gpu|cpu` says where the draft's matmuls
run: by default on the GPU when there is one (Metal: `Backend::matvec_rows`, matrix-vector
kernels that read each weight block once for the block's 3 rows, the GGUF formats dequantized
once into registers; the draft weights and the target's LM head then stay in the queue's
residency set, 3.2 GB, since binding mapped file pages per submission made them resident again
every time at ~30 ms per GB); the attention, the per-head `k_b` / `v_b` and the Markov head's
joint projection stay on the CPU. Library: `EngineOptions::draft` (`auto`, a path, `none`),
`draft_tokens`, `draft_p_min`, `draft_device` (`DsparkOptions::device`, `Dspark::set_device`);
the C API takes the same keys. Tool calls, `--dump-logits`,
`--print-prompt` and interventions use plain decoding.

Agreement: `tools/ref_dspark.py` is an independent float32 torch reference written from
llama.cpp's `graph_mistral_large4` / `build_dspark_markov_head` and ggml's YaRN, run on the same
target features (`examples/dspark_check.rs` dumps them with the draft's logits). A tiny random
draft with bf16 weights matches to 8e-7 relative on the logits of all 3 slots; the real Q8_0
draft on the real target's features to 2-6e-3 relative (int8 activations, as llama.cpp), KL
under 9e-5, the same drafts. Greedy output with DSpark (cutoff 0.85 or none) is identical to
plain greedy on the real model (128 tokens, a chat answer and a Python function, Metal and CPU;
compare runs with `--no-system`, the default system prompt holds the time). With the draft on
the GPU (float activations): the tiny draft to 8e-7 as on the CPU, the real draft to 2.2e-4,
6.6e-4 and 2.4e-3 relative on its 3 slots (KL up to 1.1e-5, the same drafts), and the greedy
output again identical to plain greedy (the same two prompts, 128 tokens, and 96 tokens through
`dspark_check --gen`). The bounded context cache: with the tiny draft's window of 8 and a
951-token prompt (so the cache was trimmed), the draft matches the reference (which keeps every
position) to 4.5e-6, and 700 greedy tokens equal plain decoding; `cargo test` checks the
bounded cache bit for bit against an unbounded one through injections and rollbacks.

Acceptance (greedy, 127 tokens): without a cutoff 2.9-3.0 tokens per verification pass (64-66%
of the drafts accepted; 84% of first drafts, 75-79% of later ones); with the 0.85 cutoff
2.2-2.7 tokens per pass, 92-93% accepted.

Speed, 128 tokens, streaming with the default 96 GB budget, `--no-system` (other agents'
builds ran meanwhile, load average 4-12; tokens per pass and acceptance in parentheses):

| prompt, device | plain | 3 drafts, no cutoff | 3 drafts, cutoff 0.85 (default) |
|---|---|---|---|
| "why is the sky blue", Metal | 1.26 tok/s | 1.17 (2.95, 66%) | 1.36 (2.23, 92%) |
| Python function, Metal | 1.26 | 1.13 (2.89, 64%) | 1.34 (2.65, 93%) |
| Python function, `--device cpu` | 1.35 | | 1.48 (2.65, 93%) |
| sky, `--temp 0.7` | 1.25 | | 1.32 (2.05, 84%) |

Other settings on the sky prompt (with the default system prompt; plain 1.24): 1 draft 1.26, 2
drafts 1.07, cutoff 0.5 1.25, 0.75 1.33, 0.95 1.38; `--expert-predict` with 3 drafts 1.06. The
draft pass itself took 30-40 ms on the CPU; on the GPU it takes 17-21 ms against 28-38 ms on the
CPU, alternating pass by pass in one process (`dspark_check --bench 20`, load average 9-20; per
layer the GPU spends 0.13 ms on `q` + `kv_a`, 0.12 on `o`, 0.64 on gate / up and 0.61 on down,
3.4 ms on the LM head, 0.23 per slot on `markov_w2`). The first pass adds ~0.65 s once (the
residency set). End to end the difference is lost in the storage-bound verification (37 vs 36-60
ms per round over 128 tokens under a load of 16-27, the same tokens). The gain is small because decoding waits on storage 60-70% of
the time, and a verification pass reads the routed experts of every token it holds (11-12
distinct of the 16 routes per layer): in plain decoding the experts consecutive tokens share
were page-cache hits anyway, so a fully accepted round reads about what plain decoding of its
tokens reads, and every rejected draft adds the experts of a token (16 GB read per pass without
a cutoff, 4.3 GB per token in plain decoding). What is amortized is the dense part and the
per-pass work (0.26 s of compute per token vs 0.33). The cutoff keeps the drafts that are nearly
always accepted (98% of those the draft gives at least 0.95, 75% at 0.7-0.95, 40% at 0.3-0.7),
which brings the storage reads per token back to the plain level. With the model in memory the
verification is bound by compute and memory bandwidth and full blocks should pay off more (not
measurable on a 128 GB machine).

### DeepSeek V4 Flash

`deepseek-ai/DeepSeek-V4-Flash` (`deepseek4.rs`; 284B parameters, 13B active, 43 layers of 4096)
follows DeepSeek's own inference code (`inference/model.py` of the checkpoint):

- Manifold-constrained hyper-connections: the residual stream is 4 copies of the hidden state.
  Before the attention and before the MoE, a projection of the RMS-scaled copies gives per-token
  weights that collapse the copies into the block's input (`pre`), spread its output back over
  them (`post`) and mix them (`comb`, a softmax made doubly stochastic by 20 Sinkhorn steps); a
  last collapse (`hc_head`) feeds the final norm.
- Attention with one shared 512-wide key-value head (the keys are the values), queries through a
  LoRA (1024) and a weightless RMS norm per head (64 heads), RoPE on the last 64 dimensions undone
  on the output at the query's position, a learned sink logit per head, and a grouped output
  projection (8 groups of 8 heads to 1024 each, then mixed). Every layer sees a window of the last
  128 raw keys; most layers add pooled keys: every 4 tokens (overlapping windows of 8, "CSA",
  with a lightning indexer of 64 heads choosing the 512 best per query) or every 128 tokens
  ("HCA", all of them), pooled by a per-channel softmax over learned gates, RMS-normed and
  rotated (YaRN, theta 160000). A pooled key becomes visible once its window is complete.
- MoE on every layer: 256 SiLU experts with clamped inputs (`swiglu_limit` 10) scored by
  `sqrt(softplus(logit))`, top 6 by score plus a selection bias, renormalized and scaled by 1.5;
  the first 3 layers take their experts from a token-id table instead (hash routing); one
  shared expert.

The reference's quantization-aware-training simulations (FP8 rounding of the cached keys, FP4 of
the indexer, the Hadamard rotation in front of them) are left out, as transformers and llama.cpp
do. Decoding keeps per layer the last 128 + 64 raw keys (the 64 for speculative rollback), the
pooled keys and the rows of the windows still pooling: about 14 KB per token of context (f32)
instead of the 88 KB a full key cache would take.

Checkpoints: llama.cpp GGUF files (`deepseek4`, split or not; unsloth's are tested, including
their two MXFP4 tensors: MXFP4 is a GGUF type here now, with NEON and Metal kernels) and float
safetensors in DeepSeek's naming (`layers.N.attn.wq_a.weight`, as the tiny test checkpoints).
The official release's FP8 / FP4 safetensors are not read (no zero-copy layout for their packed
FP4 experts yet): use a GGUF. The chat format is DeepSeek's `encoding_dsv4.py` (byte-identical
prompts, checked with tools and in both modes): `<｜begin▁of▁sentence｜>` and the system prompt,
`<｜User｜>...<｜Assistant｜>` then `</think>` (chat) or `<think>` (`--think`, the reasoning shown
dimmed), DSML tool calls (`<｜DSML｜tool_calls>` blocks, `--tools`) with results returned as
`<tool_result>` blocks of a user turn.

[Runtime interventions](#runtime-interventions-neuron-edits-steering-abliteration-sae-features)
work on this family: neuron edits of the shared and routed experts, steering, ablation, SAE
features, capture and `--make-direction`. The residual stream is the 4 hyper-connection copies,
and "the residual at layer N" is defined as their mean (what the DSpark draft reads too): hooks
read the mean and write to every copy. That section gives the exact hook points and the
agreement with PyTorch hooks, measured on the tiny random checkpoint (the real model was not
run with interventions).

```sh
# unsloth's 97 GB UD-Q2_K_XL on a 128 GB Mac (streamed: more than 60% of RAM)
./target/release/oxinfer --model unsloth/DeepSeek-V4-Flash-GGUF:UD-Q2_K_XL -p "Why is the sky blue?"
./target/release/oxinfer --model ~/.cache/oxinfer/hub/unsloth--DeepSeek-V4-Flash-GGUF/UD-Q2_K_XL --think \
    -p "How many prime numbers are there below 100?"
./target/release/oxinfer --model ~/.cache/oxinfer/hub/unsloth--DeepSeek-V4-Flash-GGUF/UD-Q2_K_XL --tools \
    -p "Use the calculator: what is 1234.5 * 6789 - 2^20?"
```

As Mistral Large 4, a checkpoint larger than 60% of RAM streams by default with a page-cache
budget of 70% of RAM (`--no-stream` / `stream=off` for plain demand paging). For UD-Q2_K_XL the
6.5 GB of dense weights stay resident and the 90 GB of routed experts are read as they are
routed; a decoded token reads 8.6 GB (5.6 GB of it attention weights, Q8_0 in this quant). Prefill
runs the expert GEMMs on the GPU's matrix units over the experts the chunk routes to (wrapped per
call, nothing pinned).

Agreement. DeepSeek's `inference/model.py`, run in float32 with its TileLang kernels replaced by
PyTorch (`tools/ref_deepseek4.py`), on tiny random checkpoints in DeepSeek's layout with every
feature (hash layers, window-only, CSA with a top-8 indexer so most queries are sparse, HCA with
a rate of 16 so a 300-token prompt has many pooled keys, a 16-token window, YaRN, clamped experts;
`tools/deepseek4_tiny.py`, compared by `examples/deepseek4_check.rs`):

| check (bf16 weights) | prefill: max logit diff / mean KL | 16 greedy decode steps |
|---|---|---|
| 60 tokens | 5.7e-5 / 2.9e-11 (60/60 top-1) | 3.8e-5 / 1.6e-11 |
| 300 tokens | 7.7e-5 / 3.0e-11 (300/300) | 3.0e-5 / 1.2e-11 |
| 300 tokens in chunks of 7, of 64 | 6.0e-5, 7.7e-5 | 3.1e-5, 3.0e-5 |

(The reference's own prefill and token-by-token decode agree only when the indexer has no exact
ties: with 4 indexer heads, keys whose every head score is negative tie at 0 and `topk` breaks
the tie differently for different lengths, so the tiny checkpoints use 16.) transformers'
`DeepseekV4ForCausalLM` gives the same prefill logits as the reference (5e-5).

On the real UD-Q2_K_XL against llama.cpp (its master, CPU) on the same file: the same next token
on every prompt tried, top-20 overlap 18/20, 17/20 and 20/20 (a 2550-token prompt, where the
indexer chooses), log-probabilities of the top 5 within 0.31-0.39; greedy continuations identical
for 48/48 tokens (a chat answer with code), 32/32 (a 1978-token prompt) and 31/32 (2550 tokens),
and for 30 tokens of a 7-token raw prompt before a near-tie ("twin brothers who were raised" /
"twin brothers raised"). Layer 0's attention matches llama.cpp to 4 digits per head; the two
quantize the activations to int8 differently, which moves the results a little more than on
Mistral Large 4.

Speed on the M5 Max (128 GB; other jobs kept the load average at 9-20 throughout, so these vary
by +-20%), UD-Q2_K_XL, streaming with the default budget:

| | prefill | decode |
|---|---|---|
| oxinfer, Metal | 64 tok/s (2550-token prompt), 5.5 tok/s (15 tokens) | 6-10 tok/s plain, 8-10 with DSpark; 8-10 after a 16k-token prompt |
| llama.cpp master, CPU (`-ngl 0`, mmap), same file | 47 tok/s (2550 tokens, warm) | 1.6-2.2 tok/s |

Decoding skips work whose result is known (lossless, bit-identical logits): a pooled key is
computed only when a token completes its window (3 tokens in 4 add only their projected rows, as
DeepSeek's code does), and the lightning indexer's query projection and scoring run only once a
query sees more than 512 pooled keys (about 2048 tokens; before that every key is selected). The
indexer's scoring of one query is spread over all threads (it was one task per query, on one
thread), and attention takes the shared-row kernel described under
[Performance](#performance): the 64 heads share one 512-wide key-value head, so each key is read
once per phase instead of once per head. Before / after on the same file: 8.1-9.3 / 9.9-10.0
tok/s after an 857-token prompt, 6.5-7.9 / 8.3-8.8 after 4191, 5.4-6.5 / 8.2-10.3 after 16076.

#### DSpark speculative decoding for DeepSeek V4

`deepseek-ai/DeepSeek-V4-Flash-DSpark` is the same model with three DSpark stages (`mtp.0..2`,
shards 46-48 of the repository, 10.9 GB). llama.cpp's converter turns them into a `dflash` GGUF
(`convert_hf_to_gguf.py DIR --dspark --target-model-dir DIR`, with a folder holding only those
shards; MXFP4 experts, Q8_0 for the FP8 weights, 10.9 GB); named `dspark*.gguf` beside the model's
folder it is used by default, as for Mistral Large 4. The draft (`dspark.rs` with
`deepseek4::Ds4Draft`):

- Context: the target's residual (mean of its 4 copies) after layers 40, 41 and 42, concatenated,
  projected and RMS-normed; each stage turns it into a key of its own 128-position window.
- Block: the last committed token and 4 noise tokens (`dspark_noise_token_id`) at the next
  positions, through three full DeepSeek V4 blocks (hyper-connections, MoE with 256 experts,
  window-only attention to the context window and to the whole block, without a mask), the
  draft's own `hc_head` and norm, the target's head; slot `i` predicts the token `i + 1` after
  the anchor. The block always holds its 5 slots (the attention has no mask), whatever the number
  of drafts used.
- A Markov head adds `markov_w2 markov_w1[previous token]` to each slot's logits, chained on the
  drafted tokens.

Agreement: `tools/ref_dspark_ds4.py` runs the stages of DeepSeek's DSpark `inference/model.py`
in float32 (FP8 / FP4 weights dequantized as used) on the target features oxinfer dumps
(`examples/dspark_check.rs`, which takes DeepSeek V4 too). A tiny random draft
(`deepseek4_tiny.py --dspark`, written as GGUF by `tools/dspark_ds4_gguf.py`) matches to 8e-7
relative on all 5 slots (KL 1e-11); the real converted draft on the real target's features to
6-10e-3 relative (Q8_0 requantization of the FP8 weights, int8 activations), KL under 2e-3, the
same 5 drafts. Greedy output with DSpark is the plain greedy output (128 tokens, checked;
verification passes are bit-identical to one-token forwards, `OXINFER_CHECK_SPEC=1`).

Acceptance and speed (greedy, 128 tokens, `--no-system --threads 12`, three alternated rounds
each, under the load above; tokens per verification pass and drafts accepted in parentheses):

| prompt | plain | 3 drafts, cutoff 0.85 (default) | 3 drafts, no cutoff | 5 drafts, no cutoff |
|---|---|---|---|---|
| "why is the sky blue" | 8.0 tok/s | 8.6 (2.35, 87%) | 8.7 (2.89, 64%) | 7.7 (3.34, 49%) |
| a Python function | 7.1 | 8.3 (3.02, 96%) | 7.7 (3.53, 88%) | 8.3 (4.54, 75%) |

A draft pass takes 15-35 ms. A verification pass of 4 tokens costs about 2.5 times a plain step
(320 ms against 125): the dense attention weights are read once, but every token adds its routed
experts, and with this quant's experts only partly in the page cache (25-30% of the time is spent
waiting on storage) those reads dominate, as on Mistral Large 4. Full blocks of 5 accept more per
pass but lose. `--spec-draft-n-max` (at most 5), `--spec-draft-p-min`, `--no-draft` and the engine /
C options `draft`, `draft_tokens`, `draft_p_min` work as for Mistral Large 4.

Not done: the official FP8 / FP4 safetensors (above), the model's own MTP layer
(`num_nextn_predict_layers`; DSpark replaces it), GPU decoding (no whole-pass Metal graph for this
family; prefill only), the vision variant (`DeepSeek-V4-Flash-Vision-Exp`) and DeepSeek V4.1
(`deepseek_v41`, a different architecture). DeepSeek-V4-Pro (1.6T, 49B active) has the same
architecture and config keys and should load as GGUF, but was not tried (its smallest GGUFs are
far beyond this machine's free disk).

### Qwen3.8-Flash-Next

`Qwen/Qwen3.8-Flash-Next` (transformers `qwen4_exp`, "a preview of the Qwen4 architecture";
`qwen4exp.rs` on top of the Qwen3.5 code) is 125B parameters with 6B active, plus a 51B n-gram
embedding table and a 4B MTP layer: 48 layers of 2560, 3 Gated DeltaNet layers (sigmoid output
gate instead of SiLU) then one gated attention layer, MoE with 512 experts (10 routed + 1 shared,
640 wide) everywhere. What is new, following transformers' `modeling_qwen4_exp.py` (and
llama.cpp's `qwen4exp`):

- Hyper-connections ("gated residual") replace every norm: the residual stream is 4 streams.
  Before each block they are RMS-normed per stream, a low-rank bottleneck (10240 -> 320, SiLU of
  a quarter, -> 10240, sigmoid) gives an element-wise read gate and the block input is the mean
  of the gated streams; the output is added to every stream with its own weight
  `2 sigmoid(w_s . normed / 4)`. The embedding starts as 4 copies; a final mixer (no write
  weights) feeds the LM head (there is no output norm).
- An n-gram hash embedding before layer 1: each token hashes its bigram and trigram (the previous
  tokens up to an EOS) into 8 heads each, `(t0 m0 ^ t1 m1 [^ t2 m2]) mod p_h + offset_h` with
  primes around 20 million, gathers 16 rows of 160 from a 320M-row table and concatenates them; a
  value projection gated per stream by `sigmoid(signed_sqrt(norm(key_s) . norm(stream_s) /
  sqrt(2560)))`, plus a causal depthwise conv (kernel 4, dilation 3) of its normed copy, is added
  to the streams. The table (29 GB at IQ4_NL in unsloth's quants, 102 GB in bf16) is never loaded:
  each token reads its 16 rows (1.4 KB) from the mapped file.
- QSA sparse attention: each attention layer's indexer (4 query heads of 128 and one key head,
  partial RoPE) scores blocks of 4 cached tokens by their mean raw key (normed, rotated to the
  block's first position): `sum_h relu(q_h . k) / sqrt(128)`; a token attends to the best 512
  blocks plus the incomplete tail. Up to 2051 visible tokens that is every token.

Checkpoints: llama.cpp GGUF (unsloth's quants, split; the n-gram table is the last part) and the
safetensors layout (tested on tiny random checkpoints: the real bf16 release is 360 GB). The
merged weights llama.cpp's converter can write (`attn_qkv_gate`, `hc_*_down_inject`) are refused
with a clear error; unsloth's files have none. The chat format is Qwen3.8's (`--think` or the server's `enable_thinking` /
`reasoning_effort` turn thinking on, with the template's reasoning-effort system line), tools as
for Qwen3.5.

```sh
# 111 GB: 82 GB of weights read through the page cache, the 29 GB n-gram table a few rows a token
./target/release/oxinfer --model unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL -p "Why is the sky blue?"
# MTP speculative decoding with unsloth's draft head (1.3x on code, slower on prose: opt-in)
./target/release/oxinfer --model unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL \
    --mtp-from ~/.cache/oxinfer/hub/unsloth--Qwen3.8-Flash-Next-GGUF/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
    -p "Write a Python function that checks whether a number is prime."
# little memory: stream with a page-cache budget (GPU or CPU decoding, measured on the first tokens)
./target/release/oxinfer --model unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL --stream --cache-budget-gb 40 -p "Hi"
```

A checkpoint whose weights other than the n-gram table exceed 75% of RAM streams by default (as
Mistral Large 4); on a 128 GB Mac UD-Q4_K_XL does not, and is read in place through the page cache
with whole-token GPU decoding. The GPU decode plan wraps the 78 GB of routed experts once and its
single- and multi-token plans share them; prefill then wraps only the experts each pass routes
to (`Experts::wrap_per_pass`), since wiring them twice exhausts the GPU's memory. Past 2051 tokens
of context (where QSA starts dropping blocks) the decode plan scores the blocks, picks the top 512
(a radix select, ties to the lower block as on the CPU) and attends to the picked positions on
the GPU too: 47.6 tok/s after a 4556-token prompt (CPU decoding: 20.4). The indexer keys live in a
one-head KV cache per layer that the plan mirrors like the attention's; the device pools a block's
key when a token ends it, and the host pools it again (its RMSNorm takes `powf(-0.5)`, which the
GPU's `rsqrt` can miss by an ulp) so the cache always holds the CPU path's keys.
`OXINFER_QSA_CHECK=1` checks every step's selections against the CPU path's scoring of the same
queries (no difference over 160 steps x 12 layers at 4.5k context); multi-token verification
(prompt lookup, MTP) runs the same ops and gives the plain GPU decode's tokens with
`--exact-verify`. On the CPU a token's block scores are computed over all threads and its
attention (24 query heads on 2 KV heads) takes the shared-row CPU kernel (see
[Performance](#performance)): 13.6 -> 17.7 tok/s after a 4338-token prompt, same tokens.
The MTP layer (`mtp_from` in the engine, C API and server options) is the trunk's 4 residual
streams plus the next token's embedding, per stream `fc([enorm(e) ; hnorm_s(h_s)])`, through one
QSA + MoE layer with hyper-connections and its own final mixer, the LM head shared.

Agreement. transformers' `Qwen4ExpForCausalLM` (main branch) in float32 on tiny random checkpoints
with every feature (`tools/qwen4exp_tiny.py`: QSA with a budget of 16 tokens so most queries are
sparse, EOS resets in the n-gram contexts, MoE top-3 of 8, bf16-representable weights; compared by
`examples/qwen4exp_check.rs`): max logit diff 6.6e-7, KL 1e-14, over 40-token prompts and 12 to
30 cached decode steps, on the CPU and with GPU decoding (GPU against CPU: KL 1e-15), including a
switch from GPU decoding to the CPU's sparse attention mid-generation. One seed differed (3e-3):
an indexer score tie at exactly 0 (every head's dot product negative), which `torch.topk` breaks
in no fixed order.

On the real UD-Q4_K_XL against llama.cpp master on the same files (`tools/llama_logits_dump.cpp`,
greedy, every decode step compared):

| prompt | oxinfer CPU vs llama.cpp CPU | oxinfer Metal vs llama.cpp CPU |
|---|---|---|
| chat, 23 tokens + 33 steps | same greedy 32/33, mean KL 1.4e-3 | 32/33, KL 1.3e-3 |
| code, 24 + 49 | 49/49, KL 3.3e-4 | 49/49, KL 1.9e-4 |
| math (thinking), 65 + 49 | 49/49, KL 1.4e-3 | 49/49, KL 2.5e-3 |
| 3189-token prompt (QSA sparse) + 25 | 25/25, KL 3.3e-3 | 25/25, KL 3.0e-3 |

The one disagreement is a near-tie where llama.cpp's own Metal build picks oxinfer's token too
(llama.cpp CPU against llama.cpp Metal: KL 1.5e-3 at the first step, the same flip). MTP drafts
from unsloth's `mtp-...-shared-Q8_0.gguf` are accepted at 92-97% on code and math, and the output
is the plain greedy output.

Speed on the M5 Max (128 GB; load average 3-9 from other jobs), UD-Q4_K_XL, greedy:

| | prefill | decode |
|---|---|---|
| oxinfer, Metal (default) | 1008-1064 tok/s (4545 tokens), 737-1007 (563; second pass, under load) | 54-59 tok/s (45-49 under load), 47.6 after 4.5k tokens |
| oxinfer, Metal + MTP (3 drafts) | | 72 tok/s on code (3.9 tokens per pass), 46 on prose |
| oxinfer, CPU (`--device cpu`) | | 21.5 tok/s |
| oxinfer, streamed (`--stream --cache-budget-gb 40`) | | 12 tok/s (47% of the time waiting on storage) |
| llama.cpp master, Metal (`llama-bench`) | 1030 tok/s (pp512) | 50 tok/s (tg128) |
| llama.cpp master, CPU (12 threads) | 61 tok/s (pp512) | 21.4 tok/s |

Memory: the GPU run's footprint is 1.5 GB besides the page cache (the weights are file pages,
82 GB of them in use); streamed with a 40 GB budget, 0.7 GB plus 22 GB of page cache.

Image input. The vision tower is Qwen3.5's (a Qwen3-VL ViT without DeepStack: 27 layers of
1152, 16x16 patches over 2 frames, learned 48x48 position grid resampled per image, axial 2D
RoPE, 2x2 merger into 2560-wide tokens placed with interleaved M-RoPE; image positions hash as
`<|image_pad|>` in the n-gram embedding, as transformers does). From safetensors it is the
checkpoint's `model.visual.*`; for a GGUF model it comes from llama.cpp's `mmproj` file
(`clip.projector_type` `qwen3vl_merger`; unsloth publishes `mmproj-F16.gguf` / `-BF16` in the
repo root), found beside the model (its folder or the one above) or named with `--mmproj FILE`
(engine / C / server option `mmproj=FILE`; `mmproj=none` turns images off). The conv kernel
llama.cpp splits per frame (`v.patch_embd.weight`, `.weight.1`) is joined back; F16 weights load
as bf16, which restores the released bf16 values exactly. Images go through `--image`, the
engine's attachments, the C API and the server's `image_url` content parts.

```sh
# the mmproj file in the repo root is picked up automatically
curl -L -o ~/.cache/oxinfer/hub/unsloth--Qwen3.8-Flash-Next-GGUF/mmproj-F16.gguf \
    https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/mmproj-F16.gguf
./target/release/oxinfer --model unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL --image painting.png -p "Describe this image."
```

Checked against transformers' `Qwen4ExpForConditionalGeneration` in float32 on a tiny random
checkpoint with the real tokenizer and processor (`tools/qwen4exp_vl_tiny.py`, compared with
`tools/ref_vl.py`): a 681x975 PNG (630 image tokens) gives image features at 1.9e-4 relative
(pixels differ by 1/255 on a few rounding ties of the bicubic resize), last-position logits within
1.7e-4 (KL 3e-8), and the same 12 greedy tokens on the CPU and Metal; the tower loaded from the
llama.cpp-converted mmproj of that checkpoint matches the checkpoint's own to 4.5e-7. On the real
model the 27-layer tower from `mmproj-F16.gguf` matches transformers' tower with the same weights
to 2.5e-5 relative on the CPU (and with `OXINFER_METAL_MMA=0`), 0.9% with Metal's default
half-precision matmul operands (0.8 s instead of 1.4 s for 630 tokens); the model names Vermeer's
"Girl with a Pearl Earring" from the PNG and describes it (prefill 721 tokens in 3.4 s).

Prefill on Metal runs whole passes on the GPU: the decode graph compiled as a prefill plan
(`MetalBackend::build_plan_pf`, chunks of up to 2048 tokens, `OXINFER_PREFILL_TOKENS`; prefill
chunks of 1024 and up are stretched to it, ~3 GB of device buffers) keeps the
activations on the device and runs the matmuls as matrix-unit GEMMs, the MoE routers as f32 GEMMs,
the routed experts as grouped GEMMs over each expert's tokens (the routing is read back before each
MoE layer: the only host round trips), the Gated DeltaNet rule (per-token norms and gates in
parallel, then the recurrence with each column's four parts in one SIMD-group), the attention
(dense, or over each token's QSA positions, four query heads per threadgroup), the indexer and the
convolutions over the whole chunk, and the expert weights wrapped once for both plans.
`OXINFER_PREFILL_GPU=0` keeps everything but the matmuls on the CPU; `attn,gdn,router` gives the
previous op-by-op offload. On raw text (`examples/prefill_kl.rs`, first pass of a process / a
second pass, under load from other processes): 563 tokens 160-190 -> 215-335 / 737-1007 tok/s,
1323 tokens 178-231 -> 490-632 / 1106-1257, 4545 tokens 214-251 -> 653-871 / 1008-1064 (the first
pass includes building the GPU plans, ~1 s; unloaded the op-by-op offload measured 261 / 324 /
374). llama.cpp master's `llama-bench` under the same load: pp512 817, pp1024 802, pp2048 837,
pp4096 821 tok/s. The results do not depend on the chunk size (every GEMM row is computed alone).
Against the CPU path the last position's KL is 1.6e-2, 3.9e-4 and 9.1e-4 (the
op-by-op offload 2.9e-2, 1.2e-3, 2.4e-4; the older Metal prefill 3.8e-2, 1.8e-4, 1.1e-3): this
model amplifies any rounding (a flipped expert among 512), and the half-precision matrix units
dominate. Image prompts (and past 2051 tokens, QSA) run on the plan too. `OXINFER_PF_PROFILE=1`
runs each op of the plan in a command buffer of its own and prints the GPU time per kind of op
(with `prefill_kl` and `OXINFER_PROFILE`): at 4545 tokens the routed experts take ~40% (grouped
32-row tiles, much of them padding at ~40 tokens per expert, and the dequantization not overlapped
with the matrix units), the attention ~11%, the GEMMs ~25%.

Not done: video input from GGUF is wired the same way but untested, the merged-weight GGUF layout,
and faster routed experts in prefill (less tile padding, or larger chunks).

## Embeddings

sentence-transformers models are recognised by their `modules.json` and run in embedding mode:
the backbone (Gemma 3 with bidirectional attention, Qwen3, Qwen2, or a BERT / XLM-RoBERTa encoder in
`bert.rs`), then the listed modules: pooling (CLS, mean, max, last token), dense projections,
normalization. Named prompts come from `config_sentence_transformers.json`. Each input line
gives one JSON array (or `-p TEXT` one):

```sh
./target/release/oxinfer --model Qwen/Qwen3-Embedding-0.6B --prompt-name query -p "What is the capital of France?"
printf 'Which planet is the Red Planet?\nMars looks red.\nVenus is the hottest planet.\n' |
  ./target/release/oxinfer --model google/embeddinggemma-300m --prompt-name query --similarity
#  1.0000  0.8014  0.5470
#  ...
./target/release/oxinfer --model google/embeddinggemma-300m --dim 256 -p "..."   # Matryoshka
./target/release/oxinfer --model BAAI/bge-m3 --sparse -p "..."     # lexical weights {token: weight}
./target/release/oxinfer --model BAAI/bge-m3 --colbert -p "..."    # one vector per token
```

`google/embeddinggemma-300m` is gated on Hugging Face (set `HF_TOKEN`); `unsloth/embeddinggemma-300m`
is an ungated copy. BGE-M3 ships only a `pytorch_model.bin`: `torch.rs` reads `torch.save`
archives directly (a zip of raw storages plus a pickled state dict; only the handful of pickle
opcodes and constructors that state dicts use are understood, and nothing is executed), so the
same goes for its `sparse_linear.pt` / `colbert_linear.pt` heads.

Checked against sentence-transformers float32 (`tools/ref_embed.py`) on short texts in five
languages and a ~2000-token one: cosine between our vector and the reference at least 0.99999 for
EmbeddingGemma (query and document prompts, and at 256 dimensions) and 1 - 1e-6 for
Qwen3-Embedding; BGE-M3's dense vectors 0.99991 or better. fp32 checkpoints are computed with
bf16-rounded matrices; against a reference using the same rounding the vectors agree to about
1e-5, so that is the whole difference. One caveat: on the long text, BGE-M3's ColBERT vectors
have a mean cosine of 0.9995 to the fp32 reference, but one token drops to 0.40. That text is
unusually sensitive: the reference itself in fp16 already drops a token to 0.89. The sparse
weights select the same tokens, with weights within 0.013.

Any BERT, RoBERTa or XLM-RoBERTa sentence-transformers model should work the same way; besides
the three above, all-MiniLM-L6-v2, paraphrase-multilingual-MiniLM-L12-v2, bge-small-en-v1.5,
multilingual-e5-small and mxbai-embed-large-v1 were checked (cosine >= 0.99999 to the reference).
E5 models expect a `query: ` / `passage: ` prefix in the text itself (they define no prompts).
MPNet (all-mpnet-base-v2: BERT plus a shared T5-style relative position bias) is supported too,
cosine >= 0.999991 to the reference (0.999996 against bf16-rounded reference weights).

NomicBERT (`nomic_bert.rs`: rotary positions, fused Wqkv, SwiGLU, dynamic NTK scaling past 2048
tokens) runs nomic-embed-text-v1 and v1.5: cosine >= 0.99998 to the reference (`--trust-remote-code`)
with and without the `search_query: ` / `search_document: ` prefixes, which go in the text (the
model defines no prompts), and on a 4000-token text; 0.999999 against bf16-rounded weights.
v1.5's Matryoshka recipe layer-norms the pooled vector before truncating, which is not a
sentence-transformers module, so `--dim` alone truncates without it.

ModernBERT (alternating global and sliding-window attention) is supported too:
modernbert-embed-base and gte-modernbert-base give cosines of 0.99993 and 0.99998 against the
fp32 reference, and 1 - 1e-6 against one with the same bf16-rounded matrices. modernbert-embed
expects a `search_query: ` / `search_document: ` prefix in the text itself.

Qwen2-based embedders run on the Qwen2 decoder of `qwen.rs` (biased q / k / v projections, no
query / key norms) without the causal mask: the modeling code these repositories ship makes
`Qwen2Model` bidirectional unless asked otherwise, and sentence-transformers never asks.

- Alibaba-NLP/gte-Qwen2-1.5B-instruct (and gte-Qwen2-7B-instruct, the same code and layout; not
  run here): last-token pooling, normalized, 1536 dimensions. Queries take the `query` prompt
  (`Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: `);
  for another task write that format into the text and name no prompt. Documents take none. The
  pooled token is the `<|endoftext|>` that the model's tokenizer class appends
  (`add_eos_token`; its `tokenizer.json` has no template for it, so it is appended here).
- NovaSearch/stella_en_1.5B_v5 (dunzhang's, renamed): mean pooling over every token including
  the appended `<|endoftext|>`, then a linear head, not normalized (as its `modules.json` says).
  Prompts `s2p_query` (retrieval queries) and `s2s_query` (similarity), none for documents;
  512 tokens. The repository has one head per output size, `2_Dense_256` ... `2_Dense_8192`,
  and `modules.json` names the 1024 one: `--dim N` (`dimensions` in the API) uses the head of
  that size rather than truncating, and any other size is an error. Downloading fetches all of
  them (140 MB).

```sh
./target/release/oxinfer --model Alibaba-NLP/gte-Qwen2-1.5B-instruct --prompt-name query -p "how much protein should a female eat"
./target/release/oxinfer --model NovaSearch/stella_en_1.5B_v5 --prompt-name s2p_query --dim 256 -p "..."
```

Against sentence-transformers float32 running that modeling code (`tools/ref_embed.py
--trust-remote-code`), on nine short texts in five languages: gte-Qwen2-1.5B cosine >= 0.99999
with and without the query prompt (largest element difference 6e-4 on unit vectors), the same on
a 1560-word text, and 1 - 1e-6 (1.5e-6 per element) against the reference with the same
bf16-rounded matrices; the reference encoding the nine texts as one left-padded batch gives the
same numbers. stella_en_1.5B_v5: cosine >= 0.999997 for documents and both prompts, in a
right-aligned (left-padded) batch as well, 0.999996 on a text cut at 512 tokens, and >= 0.999997
with each of the other six heads (`--dense 2_Dense_N` on the reference side). Inputs are embedded
one at a time, so there is no padding to mask. stella_en_400M_v5 is not a Qwen2 model (its base
is Alibaba's `new` encoder) and is not supported.

Two details that are easy to get wrong: transformers halves EmbeddingGemma's sliding window for
bidirectional attention (512 in the config means keys less than 257 positions away on either
side), and Qwen3-Embedding pools the `<|endoftext|>` token its tokenizer appends.

From the library (`examples/embed.rs`):

```rust
let dir = oxinfer::hf::resolve("BAAI/bge-m3", None)?;
let files = oxinfer::safetensors::ModelFiles::open(&dir, false)?;
let emb = oxinfer::embed::Embedder::load(&dir, &files, oxinfer::model::Quant::Bf16)?;
let mut e = emb.embed("what is a panda?", None)?;          // or Some("query") with named prompts
e.truncate(256)?;                                         // Matryoshka (--dim)
let e256 = emb.embed_dim("what is a panda?", None, Some(256))?;  // --dim: stella's head of that size, else truncation
let lexical = emb.sparse_tokens(&e);                      // BGE-M3 heads: e.sparse / e.colbert
```

## Tool calling

Both model families can call tools, in the format each was trained on: Gemma 4 declares them in
its system turn (`<|tool>declaration:...<tool|>`) and calls `<|tool_call>call:NAME{...}<tool_call|>`;
Qwen 3.5 / 3.8 lists JSON schemas in `<tools>` and answers with `<tool_call><function=...>`.
`src/tools.rs` formats the declarations, parses the calls and feeds results back the way each
template does, and the rendered prompts are byte-identical to `transformers`'
`apply_chat_template(..., tools=...)` for both families.

On the command line `--tools` enables two built-in tools, the local date and time and an exact
calculator:

```sh
./target/release/oxinfer --model google/gemma-4-E4B-it --quant q4 --tools -p "What is 1234.5 * 6789 - 2^20?"
# [tool calculate({"expression":"1234.5 * 6789 - 2^20"}) -> 7332444.5]
# The result of 1234.5 x 6789 - 2^20 is **7332444.5**.
```

In the browser the model gets `get_datetime` and `run_javascript`, which runs the code it writes in
a Web Worker with the network and storage APIs removed and a 5 s limit (hardening, not a security
boundary). Its system prompt describes where it runs: browser, operating system, language, time
zone, local time, screen, CPU cores and GPU. Both can be changed under Settings.

## OpenAI-compatible server

`oxinfer serve` exposes the OpenAI API over a whitelist of models: only the models given with
`--model` can be requested, by their id. Each is `[ID=]SOURCE[,KEY=VALUE...]`: a local folder or
file, a repo id (fetched on first use, or at start with `--download`) or `org/name:QUANT` for a
GGUF repo, an optional id (default: the source) and the model's options. HTTP and HTTPS are
served by [httpsd](https://crates.io/crates/httpsd) (its thread-pool runtime, no async runtime).

```sh
./target/release/oxinfer serve --listen 127.0.0.1:8080 \
  --model qwen=Qwen/Qwen3.5-0.8B,quant=q8 \
  --model gemma=mlx-community/gemma-4-e2b-it-4bit,think=1 \
  --model bge=BAAI/bge-small-en-v1.5 \
  --model sd=stabilityai/sd-turbo \
  --model whisper=openai/whisper-small \
  --model tts=unsloth/orpheus-3b-0.1-ft,quant=q8
# [serving qwen (chat), gemma (chat), bge (embedding), sd (image), whisper (transcription), tts (speech) on http://127.0.0.1:8080/v1]

curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen", "messages": [{"role": "user", "content": "Why is the sky blue?"}], "stream": true}'
curl -s localhost:8080/v1/audio/transcriptions -F model=whisper -F file=@speech.mp3
curl -s localhost:8080/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model": "tts", "input": "Hello there!", "voice": "leah"}' -o hello.wav
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
r = client.chat.completions.create(model="qwen", messages=[
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
        {"type": "text", "text": "What is in this picture?"}]}])
print(r.choices[0].message.content)
img = client.images.generate(model="sd", prompt="a lighthouse at dusk", size="512x512",
                             response_format="b64_json", extra_body={"seed": 7})
```

| Endpoint | Models | Notes |
|---|---|---|
| `GET /v1/models`, `/v1/models/{id}` | all | the whitelist, with `kind` and `loaded` |
| `POST /v1/chat/completions` | chat (Gemma 4 / 3, Qwen 3.5 / 3.8 / 3, Llama 3, Mistral Large 4, DeepSeek V4, Core ML) | system / developer, multi-turn, `image_url` (data: or http(s) URLs) and `video_url` parts (vision checkpoints), `input_audio` (Gemma 4), `tools` / `tool_choice` (`auto`, `none`, `required` or a function: forced calls) with `tool_calls` out, `response_format` (`json_object`, `json_schema`: constrained decoding), `logprobs` / `top_logprobs`, `temperature`, `top_p`, `top_k`, `max_tokens` / `max_completion_tokens`, `stop`, `seed`, `n`, `stream` (SSE, `stream_options.include_usage`), thinking with `reasoning_effort` (`none` off, anything else on), `enable_thinking` or `chat_template_kwargs.enable_thinking`, returned as `reasoning_content` |
| `POST /v1/completions` | chat | raw continuations (no template): `prompt` as text, texts or token ids, `echo`, `stop`, `stream` |
| `POST /v1/embeddings` | embedding | `encoding_format` float / base64, `dimensions` (Matryoshka); `prompt_name` (extension) |
| `POST /v1/images/generations` | image (Stable Diffusion, FLUX.1, Qwen-Image, OmniGen2, Step1X-Edit, BAGEL, Janus-Pro) | `size`, `n`, `response_format` (`b64_json`, or `url` as a PNG data URL); extensions `seed`, `steps`, `guidance`, `negative_prompt` |
| `POST /v1/images/edits` | image | multipart: `image` (or `image[]`, several for Qwen-Image-Edit-2509+ and OmniGen2), `mask` (Stable Diffusion), `prompt`, the same fields, `strength` |
| `POST /v1/audio/transcriptions`, `/v1/audio/translations` | Whisper | multipart: `file` (any format oxideav decodes), `language`, `prompt`, `temperature`, `response_format` json / text / verbose_json / srt / vtt, `timestamp_granularities[]` word / segment, `beam_size`, `patience`, `length_penalty`, `best_of`; several `file` parts are transcribed as a batch and answered as a JSON array (json / verbose_json) |
| `POST /v1/audio/speech` | Orpheus | `input`, `voice` (the model's voices; OpenAI's names take the default one), `response_format` wav or pcm (16-bit 24 kHz mono, streamed as it is generated), `stream_format` sse; extensions `seed`, `temperature`, `top_p`, `repetition_penalty`; an array as `input` (1 to 64 texts, `voice` one name or an array) is generated as one batch and answered as JSON, `data[i].audio` the base64 WAV (or PCM) of `input[i]` |

Errors have OpenAI's shape (`{"error": {"message", "type", "param", "code"}}`) and status codes:
400 for a bad request (also a model of the wrong kind for the endpoint), 401 without the API key,
404 for an unknown model (`model_not_found`) or endpoint, 405, 413 over the body limit, 500 when
loading or running fails. The paths also work without `/v1`.

The kind of each model (chat, embedding, image, transcription, speech) comes from its files
(fetching only the small configuration files of a repo id that is not in the cache yet), or the
`kind` option. Models load on their first request, or all at start with `--preload`. Each model
runs one request at a time; the others wait for it. When a model to load would not fit in the
memory budget (`--memory-gb`, default 80% of RAM; each model counts the size of its weight files,
scaled for `quant`, or its `memory_gb` option) next to the loaded ones, the least recently used
idle models are unloaded first. A chat request that extends the previous conversation of a model
(the same messages, the reply as it was returned and new messages, as chat clients send them)
reuses its cache: only the new tokens are fed (`usage.prompt_tokens_details.cached_tokens`); this
works for Qwen3.5's recurrent layers too, and a conversation whose earlier messages carry media is
fed again. Models without recurrent state also keep the prefix a new conversation shares with the
previous one. With `--prompt-cache DIR` (or `auto`; per model `prompt_cache=`), conversations are
also kept on disk across requests, conversations and restarts and restored when a prompt starts
with one (see [Session snapshots and the prompt cache](#session-snapshots-and-the-prompt-cache)).

Model options: those of the engine for chat models (`quant`, `device`, `mmap`, `stream`, `mmproj`,
`draft`, `mtp`, `lookup`, `prompt_cache`, `max_context`, `prefill_chunk`, `gpu_decode`, ...; see
[As a library](#as-a-library)),
of the loader for the other kinds (Whisper `encoder` / `decoder` precision, Orpheus `snac` and
`quant`, FLUX `transformer` / `base`, Stable Diffusion `lora`, ...), and:

| Key | Meaning |
|---|---|
| `kind` | `chat`, `embedding`, `image`, `transcription` or `speech` (default: detected) |
| `system` | system prompt when a request has none (default none; `default` for the CLI's prompt with the date and time) |
| `think` | thinking by default (`1`); requests override it |
| `max_tokens` | reply length when a request sets none (default 4096) |
| `temperature`, `top_p`, `top_k`, `repetition_penalty` | sampling defaults (default OpenAI's: temperature 1, top-p 1) |
| `prompt_name` | embeddings: the named prompt used by default (`query`, ...) |
| `memory_gb` | the memory the model is counted for |

Server options: `--listen ADDR` (port 0 picks one), `--api-key KEY` (or `$OXINFER_API_KEY`;
requests need `Authorization: Bearer KEY`), `--cors ORIGIN`, `--max-body-mb N` (default 16),
`--preload`, `--download`, `--memory-gb F`, `--cache-dir DIR`, `--threads N`,
`--prompt-cache DIR|auto` and `--prompt-cache-gb F` (the chat models' prompt cache),
`--tls-cert FILE --tls-key FILE` or `--self-signed` for HTTPS, `--workers N` (connections served
at once, default 32), `--quiet`. Media URLs in requests may be `data:` URLs or `http(s)://` URLs,
which the server fetches (`--no-remote-media` turns that off); local paths and `file://` URLs
only with `--local-media`, for trusted clients. `--config FILE` reads the same options as JSON:

```json
{"listen": "0.0.0.0:8080", "api_key": "...", "memory_gb": 64,
 "models": ["qwen=Qwen/Qwen3.8-27B,quant=q4,mtp=1",
            {"id": "embed", "model": "Qwen/Qwen3-Embedding-0.6B", "options": {"prompt_name": "query"}}]}
```

In the library, `oxinfer::server::Api` handles a request without any transport (method, path,
content type, body in; a complete body or a streamed one out), and `oxinfer::server::http::start`
serves it; `Engine::chat` runs a whole conversation (a list of `Message`s) with the cache reuse
above. The C API has `oxi_server_new` (the options as `key=value` pairs or JSON),
`oxi_server_start` / `oxi_server_stop`, `oxi_server_handle` (one request, streamed bodies through a
callback) and `oxi_engine_chat`. `tools/openai_check.py` checks every endpoint with the official
`openai` Python client (chat, multi-turn with cache reuse, streaming, stop sequences, tool calls
and their results, streamed tool calls, thinking, structured outputs, forced tool calls, log
probabilities, completions, errors, vision, embeddings, image
generation and editing, speech as WAV and streamed PCM, transcriptions as JSON, verbose JSON and
SRT): all pass with Qwen/Qwen3-0.6B (chat), Qwen/Qwen3.5-0.8B (vision), BAAI/bge-small-en-v1.5,
stabilityai/sd-turbo, openai/whisper-tiny and the Orpheus Q4_K_M GGUF, and the chat and vision
checks with mlx-community/gemma-4-e2b-it-4bit, which also transcribes `input_audio` parts. The chat
checks with structured outputs, forced tool calls and log probabilities were run with
mlx-community/Qwen3-4B-4bit on the CPU: all pass.

### Structured outputs, forced tool calls and log probabilities

`response_format: {"type": "json_object"}` and `{"type": "json_schema", "json_schema": {"schema":
{...}}}` constrain the reply while it is decoded (`src/constrain/`, no dependency): the schema is
compiled to a grammar over bytes, and at every step the logits of the tokens that would leave the
grammar are masked before sampling, so the answer is always a prefix of a valid document, and a
valid document when the reply ends by itself. A token that spans several grammar symbols (`","`,
`"}]`) is followed through all of them; a multi-byte character split across tokens is tracked byte
by byte (only bytes that keep the text valid UTF-8 pass); special tokens never pass as text; the
end-of-turn token passes only once the document is complete. A thinking block is not constrained
(`reasoning_content` stays free text, but the reply cannot end in it); the grammar starts when it
closes. A reply cut by `max_tokens` has `finish_reason: "length"` and holds the start of a
document, as with OpenAI. The schema is always enforced, with or without `strict`.

The supported schema subset is the one of OpenAI's strict structured outputs:

| Keyword | Notes |
|---|---|
| `type` | `object`, `array`, `string`, `number`, `integer`, `boolean`, `null`, or a list of them |
| `properties`, `required`, `additionalProperties` | keys are written in the schema's order; properties that are not required may be left out; `additionalProperties: false` (also assumed next to `properties`); an object without `properties` takes any keys, with values of `additionalProperties` when that is a schema |
| `items`, `minItems`, `maxItems` | bounds up to 10 000 |
| `enum`, `const` | any JSON values |
| `anyOf` | |
| `$ref`, `$defs` / `definitions` | references inside the schema (`#`, `#/$defs/...`), recursion included |
| `title`, `description`, `default`, `examples`, `format`, `$schema`, `$id`, `$comment`, `deprecated`, `readOnly`, `writeOnly` | constrain nothing (string formats are not enforced) |

Anything else (`pattern`, `minLength`, `minimum`, `multipleOf`, `oneOf`, `allOf`, `not`,
`prefixItems`, `patternProperties`, `additionalProperties` next to `properties`, remote `$ref`,
a reference cycle that never reaches a value, ...) is a 400 naming the keyword and where it is
(`#/properties/n: schema keyword "minimum" is not supported`): a schema is never partly applied.
The root may be any type. Whitespace between JSON tokens is none, one space, or a newline and
up to 16 spaces or tabs (compact or indented output, never an endless run); numbers have at most
15 integer digits, 16 fraction digits and 2 exponent digits; `\u` escapes are whole code points
(no lone surrogates), so every reply parses with a strict parser.

`tool_choice: "required"` and `tool_choice: {"type": "function", "function": {"name": ...}}` force
the reply (after any thinking) to be one tool call, with the same machinery: the grammar is the
model family's call syntax (Qwen3's JSON in `<tool_call>`, Qwen 3.5 / 3.8's `<function=...>` with
`<parameter=...>` elements, Gemma 4's `<|tool_call>call:NAME{...}` with `<|"|>` strings, Mistral's
`[TOOL_CALLS]NAME[ARGS]{...}`, DeepSeek V4's DSML block; the markers as their special tokens) with
the arguments of the tool's `parameters` schema (the subset above; a schema outside it is a 400).
In the formats that write string arguments bare (Qwen 3.5 / 3.8, DeepSeek V4) a string cannot hold
`</`, and a parameter declared as a string now stays one when its text reads as JSON (`"42"`).
A forced call cut by `max_tokens` is not reported as a call. With `response_format` and tools but
no forced call, the model chooses between the document and a call.

`logprobs: true` with `top_logprobs: N` (0 to 20) returns, for every token of the answer (not the
thinking, the delimiters, a tool call or the stop token), `{token, logprob, bytes, top_logprobs}`
in `choices[].logprobs.content`, streamed with the chunk that carries the token's text;
`/v1/completions` takes `logprobs: N` and returns the legacy `tokens` / `token_logprobs` /
`top_logprobs` / `text_offset`. The log probabilities are of the distribution the model produced:
the softmax of its logits at temperature 1, before the repetition penalty, `temperature`, `top_p`,
`top_k` and before any constraint (as OpenAI's, which do not move with the sampling settings; a
token that a constraint forced can therefore be unlikely, and a probability of zero is shown as
-9999). `bytes` are the token's own bytes, so a character split across tokens can be put together.

The mask and the log probabilities are applied in the sampler, which every decoding loop goes
through with the model's logits, on the CPU and with GPU decoding alike. Speculative decoding
keeps them exact: verification masks the target's distribution at each position (a draft token
outside the grammar is rejected and replaced from the masked distribution), and DSpark / MTP
drafts are themselves drawn under the mask of a forked grammar state, so they stay acceptable;
prompt-lookup drafts are checked the same way. Gemma 4's greedy drafter loop, which does not
sample, is bypassed for such requests (plain decoding), and a Core ML model that returns only its
argmax refuses them. In the library these are `ChatConfig::response_format`, `tool_choice` and
`logprobs` (`Reply::logprobs`, `Delta::logprob`); in the C API `oxi_engine_set(e,
"response_format", ...)`.

Checked on real weights with `cargo test --release --test server -- --include-ignored structured
logprobs` (`OXINFER_E2E_CHAT` picks the model, `OXINFER_E2E_OPTS` adds model options): a nested
schema (enum, `anyOf`, array of objects) on four prompts that never describe it, greedy and at
temperature 1, and forced calls (`required` and a named function) on three prompts, of which two
ask for no tool. Every reply that ended by itself parsed and validated, and every forced call was
one call with arguments of its schema: with Qwen/Qwen3.5-0.8B on the CPU (6 of 8 schema replies
complete in 300 tokens, the others cut by the limit; 12 of 12 calls), the same with `lookup=1`
and with `mtp=1`, and with GPU decoding (CUDA; 4 and 5 of 8 complete in two runs, 12 of 12 calls); with
mlx-community/Qwen3-4B-4bit on the CPU, alone and with the DSpark draft
`deepseek-ai/dspark_qwen3_4b_block7` (8 of 8 and 12 of 12 each). Unit tests cover the rest:
accept / reject sets of the grammar and of every schema keyword, masks compared with a direct
check of every token along whole documents on a vocabulary with tokens that span symbols and
split characters, replies sampled from random logits (all valid documents), each family's call
syntax against its own renderer and parser, and prompt-lookup and DSpark rounds under a filter
giving exactly the tokens and log probabilities of plain filtered decoding.

Log probabilities do not depend on streaming: on a newly loaded model the entries of a streamed
and of a plain request are identical (difference 0 on both models). They do follow the cache: a
request that repeats the previous conversation feeds only the tokens the cache lacks, so on
models that keep a prefix (Qwen3, Gemma, Llama) the last prompt token then runs alone on the
decode kernels instead of inside the prompt's pass on the prefill kernels, which is a slightly
different computation (as it always was for the text). With 4-bit weights that is visible: on
Qwen3-4B-4bit the same request repeated (15 of 16 prompt tokens cached) gave the same tokens with
log probabilities up to 0.42 apart, where the logits of the two pass shapes themselves differ by
up to 0.63 (KL 1.3e-4, `examples/prefill_kl` with `--chunk 15` against one pass; 2.4e-5 and KL
2.6e-12 on the bf16 Qwen3.5-0.8B). Qwen3.5's recurrent layers cannot keep a prefix of a repeated
prompt, so there a repeat recomputes it and gives identical values.

Not supported: schema keywords outside the table above, several forced calls in one reply
(`tool_choice: "required"` gives exactly one), audio
output from chat models, the Responses and Assistants APIs, image `url` responses other than data
URLs, and speech formats other than WAV and PCM. Requests that take over 20 s are answered as 200
before they finish (httpsd drops a connection whose handler runs past its 30 s inactivity window),
so an error after that point arrives in a 200 body.

## Session snapshots and the prompt cache

A session (the KV caches, the Gated DeltaNet recurrent and convolution states, positions and
vision spans, the DSpark draft's context cache and pending features, the MTP layer's cache and
pending rows, and the tokens) can be written to a file and read back bit for bit (`snapshot.rs`):

```sh
oxinfer --model Qwen/Qwen3-8B-GGUF:Q8_0 -p "$(cat long_prompt.txt)" --save-session s.oxsnap
oxinfer --model Qwen/Qwen3-8B-GGUF:Q8_0 --load-session s.oxsnap -p "And then?"   # continues it
oxinfer --model Qwen/Qwen3-8B-GGUF:Q8_0 --prompt-cache auto -p "..."            # keyed cache
```

A **prompt cache** (`--prompt-cache DIR|auto`, engine / C API / per-model server key
`prompt_cache`, server-wide `prompt_cache=`) is a directory of such snapshots keyed by model
fingerprint (checkpoint, precision, format, device, speculative decoding and its options) and
token prefix, with a size cap (`prompt_cache_gb`, default 20; least recently used first).
`Engine::chat` restores the snapshot that saves the most of the prompt: for models without
recurrent state (Gemma, Qwen3, Llama, with their drafts) any common prefix (the restored caches
are cut back), for Qwen3.5 / 3.6 / 3.8 (recurrent state cannot be cut back) a whole snapshot the
prompt starts with. It stores each conversation after its reply (left pending so the reply is not
delayed: the server writes it on another thread once the response is sent; the next request, an
explicit `Engine::flush_prompt_cache` or the engine's drop also write it) and, for the recurrent
models, the prefix every conversation with the same system prompt and tools shares (the first
turn rendered with an empty message, tokenized, compared with the prompt) and the prompt but its
last token (a request sent again, a reply regenerated), the first time each is computed (the
prompt is then fed in two or three parts). For the models without recurrent state, a stored
conversation replaces the snapshots of its own prefixes. In memory, `Engine::chat` now also keeps the common prefix with the previous
conversation for these models (it used to start over unless the new conversation extended the
old one).

The directory is created private (0700; an existing one is set to 0700, which fails unless it is
yours, so `/tmp` or `/dev/shm` themselves are refused; a directory of your own on a tmpfs gives
memory-speed restores on Linux), files are written 0600 under a temporary name and renamed into
place, symbolic links are not followed, every section length is checked against the model before
anything is allocated, and a checksum is checked before the state is used. A snapshot of another
model or other options is refused; a corrupted one is removed from the cache and the prompt is
computed.

Families: Gemma 4 / 3, Qwen3, Qwen3.5 / 3.6 / 3.8 (dense, MoE, Qwen2.5-VL), Llama 3, with DSpark
or MTP drafts and the Gemma 4 assistant drafter. Not (yet): Mistral Large 4, DeepSeek V4
(compressed-attention state), Qwen3.8-Flash-Next (Qwen4-Exp: n-gram and indexer state), Core ML
models.

Checks (M5 Max, Metal; the machine was shared with other jobs at load 30-95, which slowed the
prefills): `examples/session_check.rs` prefills a ~3000-token prompt, writes the session, reads it
into a fresh session and decodes 32 greedy steps from the original, the restored and a recomputed
session: the restored logits are the original's bit for bit at every step (GPU decoding, whose
device copies of the caches are rebuilt from the restored ones), for Qwen3-8B (2780 tokens, 820 MB
snapshot, written in 0.35 s, read in 0.22 s against a 23 s prefill), Qwen3.5-4B (2909 tokens,
243 MB, 0.06 / 0.07 s against 11.6 s; recurrent states and the attention layers' KV) and
gemma-4-12B-it (3057 tokens, 2.1 GB, 0.53 / 0.31 s against 11.2 s). Through the engine,
`--save-session` then `--load-session` with a second message gives the same reply as one process
holding the conversation (Qwen3-8B with its DSpark draft, Qwen3.5-4B with MTP, gemma-4-12B-it with
DSpark, `--exact-verify`), and `tests/server.rs` (`prompt_cache_across_restarts`) restarts a server
and gets the same reply from a restored prompt. With `--prompt-cache` and the same ~3000-token
prompt twice:

| model | first request: prefill | second request: restore | snapshot |
|---|---|---|---|
| Qwen3.5-4B (bf16, MTP) | 2909 tokens in 9.2 s | 2908 tokens in 0.08 s | 267 MB |
| Qwen3-8B (Q8_0 GGUF, plain) | 2780 tokens in 87 s (load 40) | 2779 tokens in 0.50 s | 824 MB |
| gemma-4-12B-it (Q8_0 GGUF, plain) | 3057 tokens in 60 s (load 37) | 3056 tokens in 0.57-1.9 s | 2.1 GB |

and the same reply both times; a different question after the same document restores the common
prefix on the models without recurrent state (2771 of 2786 tokens for Qwen3-8B). The first decode
step after a restore copies the caches to the GPU, as the first step after a prefill does.
Snapshots hold f32 caches: about 0.3 MB per token for Qwen3-8B, 0.09 MB for Qwen3.5-4B (a quarter
of its layers have a KV cache), 0.7 MB for Gemma 4 12B, whose sliding-window layers keep every
position.

## Typed decisions (Jev models)

The Jev models (`autotrust/JEV-9B`, `autotrust/JEV-27B`, ...) answer a typed question about a
piece of text (or images: [Image decisions](#image-decisions)) with calibrated probabilities, in one
prefill pass and with no generation ("System 1"): `noul` (yes / no, probabilities for `["false", "true"]`), `score` (a 0-5 scale) or
`choice` (2-16 options labelled A-P, up to 256 with the extra single-token labels Q-Z, AA, AB,
...). The prompt is the `bare-v1` template:

```text
[kind] choice
[state] SKU AX-330 stock at 8% of safety level; supplier late twice this quarter.
[question] Supplier response for this scenario.
[options]
A) issue_warning
B) renegotiate
C) dual_source
D) maintain
[decision]:
```

The last token's final-norm hidden state goes through a small head (`head.safetensors`, 24 slots),
the active slots are divided by a per-kind temperature (`calibration.json`) and soft-maxed. The
backbone of JEV-9B / JEV-27B is the unmodified Qwen3.5-9B / Qwen3.8-27B; the decision skill is a
LoRA adapter (`adapter/`, r=16, about 160 / 435 MB) that oxinfer applies **unmerged, at run time,
on top of whatever base weights are loaded** (bf16, `--quant q8|q4`, pre-quantized, MLX or GGUF)
and only during a decision. One loaded model therefore serves chat (the base model, bit for bit:
the adapter's layers are skipped when it is off) and decisions. The bundles that ship merged
weights and read the decision from the LM head (`adapter_vllm/decision_head.json`, e.g.
`autotrust/JEV-35B`) work too.

```sh
# a Jev bundle (backbone, adapter and head; the repo id downloads all of it)
oxinfer --model autotrust/JEV-9B --decide noul \
  --state "Customer says the parcel arrived damaged and wants their money back." \
  --question "Is the customer asking for a refund?"
#   false  0.0092
# * true   0.9908  ########################################

# any matching base you already have, plus only the decision files of the bundle (~420 MB for
# JEV-27B: adapter, head and configs, not the 54 GB backbone)
oxinfer --model ~/.cache/oxinfer/hub/Qwen3.8-27B-oxinfer-q4 --decision-from autotrust/JEV-27B \
  --decide choice --state "SKU AX-330 stock at 8% of safety level; supplier late twice this quarter." \
  --question "Supplier response for this scenario." \
  --option issue_warning --option renegotiate --option dual_source --option maintain

# requests in the /v1/decide JSON format (an object, an array or one per line): one response per line
oxinfer --model autotrust/JEV-9B --decide-json requests.json
```

`--decision-from` (engine option `decision_from`) takes a directory or a repo id; the base must
match the bundle (hidden size, layer count, vocabulary, verbalizer tokens, and the shape of every
adapted layer), otherwise loading fails with a message saying which. As a library:

```rust
use oxinfer::decide::DecisionRequest;
use oxinfer::engine::{Engine, EngineOptions};
let mut opts = EngineOptions::default();
opts.set("quant", "q4")?;
opts.set("decision_from", "autotrust/JEV-9B")?;
let mut engine = Engine::load("Qwen/Qwen3.5-9B", &opts)?;
let d = engine.decide(&DecisionRequest::choice(
    "I was charged twice for my subscription this month.",
    "Which team should handle this ticket?",
    ["billing", "technical support", "sales", "account security"],
))?;
println!("{} ({:.3})", d.best, d.probabilities[d.best_index]);
```

A decision runs in a scratch context: the conversation of the engine is left as it was, so
`send` and `decide` can be interleaved (`examples/decide.rs` checks that the chat reply is the same
before and after a decision, and the same as without any adapter). `Engine::decide_json` and the C
API's `oxi_decide` take and return the JSON of the reference server's `POST /v1/decide` (`kind`,
`state`, `question`, `options` in; `options`, `probabilities`, `choice_index`, `choice`,
`adaptation` (`native` or `wide-labels`), `usage`, ... out); `oxi_decide_info_json` describes the
loaded assets.

Agreement with the reference (`transformers` + `peft`, unmerged adapter, the card's slot-head
readout; 10 questions: 3 noul, 2 score, 4- and 12-option choices and a 20-option wide-label choice
read through `adapter_vllm`'s LM-head LoRA), JEV-9B:

| oxinfer | vs reference f32 | | vs reference bf16 |
|---|---|---|---|
| | max \|Δp\| | max KL | max \|Δp\| |
| bf16 weights, CPU | 4.8e-4 | 5.3e-7 | 3.9e-3 |
| bf16 weights, Metal | 5.0e-4 | 5.8e-7 | |
| BF16 GGUF base (unsloth) + JEV-9B adapter, Metal | 4.6e-4 | 5.2e-7 | |
| `--quant q8` | 6.5e-3 | 1.2e-4 | |

(The bf16 reference itself differs from the f32 one by 4.0e-3.) Qwen3.5-9B with
`--decision-from autotrust/JEV-9B` gives the same probabilities as the JEV-9B folder, and JEV-27B on
the bf16 Qwen3.8-27B reproduces the model card (refund `true` 0.978, AX-330 `dual_source` 0.62);
on the Q4 conversion the largest difference to bf16 is 0.044 (KL 6.6e-3, from the 4-bit weights).
`autotrust/JEV-35B` (merged Qwen3.5-35B-A3B MoE, LM-head readout with 256 trained choice slots)
against `transformers` in bf16 (an f32 reference does not fit in 128 GB): median max |Δp| 4e-4,
worst 0.042 (KL 6.6e-3) on a near-tie score question; MoE routing flips near ties, so the CPU and
Metal runs of oxinfer differ from each other by up to 0.014 too. It decides in 0.25-0.45 s on
Metal for 45-160 tokens.

Latency (M5 Max, 18 threads; a decision is a prefill, so it scales with the prompt):

| model | device | 105-token prompt | 1461-token prompt | plain prefill of the same 1461 tokens |
|---|---|---|---|---|
| JEV-9B (bf16) | CPU | 1.31 s | 18.3 s | 17.8 s |
| JEV-9B (bf16) | Metal | 0.32 s | 2.63 s | 2.44 s |
| JEV-27B on Qwen3.8-27B Q4 (`decision_from`) | CPU | 5.2 s | 70 s | |
| JEV-27B on Qwen3.8-27B Q4 (`decision_from`) | Metal | 0.72 s | 6.2 s | 5.9 s |

The adapter costs 3-7 % of the prefill (low-rank products in f32 NEON on the CPU, after the base
matmul, also when the base matmul runs on the GPU). Not supported: Core ML models, and the
adaptive thinking of the reference server (`"thinking": "auto"`: System 2 reasoning folded into the
probabilities; the base model itself chats as usual in the same engine).

**LoRA on routed MoE experts.** An adapter may also target the routed experts of a MoE base
(Qwen3.5 / 3.6 MoE: `layers.N.mlp.experts`; Gemma 4 26B-A4B: `layers.N.experts`): each expert
gets its own low-rank update, applied to the tokens routed to it, on every CPU path of the
experts (decode, the batched kernels of short passes, grouped prefill) and with streamed experts.
Two adapter layouts are accepted, on either storage of the base experts (fused `gate_up_proj`, or
separate gate / up matrices as in GGUF):

- stacked parameters, what peft writes for the current transformers classes, whose experts are 3-D
  `[experts, out, in]` parameters: `LoraConfig(target_parameters=["experts.gate_up_proj",
  "experts.down_proj"])` saves nested wrappers `...experts.lora_A/B` and
  `...experts.base_layer.lora_A/B`, with `lora_A` `[experts * r, in]` and `lora_B`
  `[out, r * experts]`; the keys do not name the parameter, so it is told from the shape;
- one module per expert (`...experts.E.gate_proj`, `.up_proj`, `.down_proj`), ordinary pairs.

A prefill pass whose experts would run on the GPU runs them on the CPU while the adapter is on
(the device computes an expert in one submission, with no place for the side path; the dense
matmuls stay on the GPU, with theirs), and a model with an adapter attached decodes without the
GPU decode plan, as with any runtime LoRA. Not accepted, as an error at load (an expert module
that cannot be applied is never skipped): separate stacked `gate_proj` / `up_proj` parameters
(their wrappers have the same shape and cannot be told apart), a rank or expert count other than
the model's, both layouts on one weight, and DoRA / per-module ranks as for dense layers.

No published bundle adapts routed experts, so this was verified on synthetic adapters only:
`tools/moe_lora_tiny.py` makes a tiny random-weight Qwen3.5-MoE or Gemma 4 MoE checkpoint
(4 layers, 8 experts, top 3; hidden 128) and a random rank-4 peft adapter on the experts'
`gate_up_proj` / `down_proj` and on attention and MLP projections, and
`examples/moe_lora_check.rs` compares every position's logits of a 40-token prompt with peft's
unmerged forward (float32). bf16 weights, CPU, the adapter moving the logits by up to 9.6 (Qwen) /
4.4 (Gemma):

| | Qwen3.5-MoE: max \|Δlogit\| | max KL | Gemma 4 MoE: max \|Δlogit\| | max KL |
|---|---|---|---|---|
| one prefill pass | 7.1e-5 | 4.1e-10 | 6.0e-5 | 1.6e-10 |
| passes of 5 tokens, and token by token | 9.8e-5 | 3.6e-10 | 3.3e-5 | 5.5e-11 |
| adapter off, vs the base model | 1.6e-4 | 1.3e-9 | 4.3e-5 | 8.3e-11 |

The per-expert layout of the same adapter and the same checkpoint saved with one module per
expert give the same numbers in all four combinations. With larger random models (hidden and
expert width 1024, 4 experts, 160 tokens) on CUDA, where the base experts' prefill does run on the
GPU when the adapter is off: 5.1e-5 (Qwen) and 2.0e-4 (Gemma) with the adapter on, as on the CPU
(5.1e-5 / 2.0e-4). On Q8 / Q4 weights the batched expert kernels with the adapter match the
one-expert-at-a-time path bit for bit (logits of random weights quantized this coarsely are too
far from float32 to compare with the reference), and after a pass with the adapter the base
model's logits are bit for bit what they were. Not run: an adapter of this kind on real weights,
Metal, and streamed experts end to end (they take the same CPU paths).

### Gemma 4 26B-A4B bundles

`autotrust/GEV-26B-Decide` and `autotrust/JEV-Gemma4-26B-A4B` (the same weights published twice)
put System 1 on the unmodified `google/gemma-4-26B-A4B-it`: a LoRA (r=32) on the attention and
dense-MLP projections of all 30 layers (205 modules; the routed experts are not adapted), the
24-slot head and the temperatures. What differs from the Qwen bundles: the prompt starts with
`<bos>`, the head's logits are soft-capped like Gemma's own (`30 * tanh(z / 30)`, from
`judge_config.json`), and a choice with more than 16 options is a **tournament** by default, as in
the bundle's `serve_decide.py`: groups of at most 16 options are read natively, then a final pass
over each group's winner plus the best of the rest (16 in all), and the group shares are combined
as the reference does. `"strategy": "single"` (JSON) or `DecisionRequest::strategy` reads them in
one pass through the wide labels instead (GEV only: it needs `adapter_vllm/`'s LM-head readout;
JEV-Gemma4 has none). Images go where they appear, each as `<|image>` + its soft tokens +
`<image|>` with the processor's default budget, attending both ways within the image.

```sh
# only the decision files of the bundle (150 MB), on the Google base already on disk
oxinfer --model google/gemma-4-26B-A4B-it --mmap --decision-from autotrust/GEV-26B-Decide \
  --decide choice --state "Set an alarm for 6:30 tomorrow morning please." --question "Which intent?" \
  --option weather_query --option alarm_set --option alarm_remove --option play_music
oxinfer --model google/gemma-4-26B-A4B-it --mmap --decision-from autotrust/GEV-26B-Decide --decide-json requests.json
```

Agreement with transformers + peft (the card's readout, unmerged adapter, float32 compute on the
bf16 weights): the 10 text decisions of the JEV-9B check (3 noul, 2 score, 4- and 12-option
choices, a 20-option tournament) and 5 image decisions (fox photo and shapes, 1-2 images of 256-266
soft tokens, noul / score / 5- and 6-option choices):

| oxinfer | text: max \|Δp\| | max KL | images: max \|Δp\| | max KL |
|---|---|---|---|---|
| GEV-26B-Decide, bf16, CPU | 1.8e-6 | 7e-12 | 5.9e-3 (shapes; the others 5e-5) | 1.4e-4 |
| JEV-Gemma4-26B-A4B, bf16, Metal | 3.0e-3 | 2.6e-5 | | |
| 20 options with `"strategy": "single"` vs the reference through `adapter_vllm/` | 2.0e-3 | 2.5e-5 | | |

Prompt token counts match the processor's. The image difference comes from the vision tower's
input resize, as for chat (logits of image prompts differ by 0.03). On Metal (bf16, M5 Max) a text
decision takes 0.18-0.32 s for 46-139 prompt tokens, the 20-option tournament (3 passes, 364
tokens) 0.87 s, and an image decision 1.4 s (0.8 s of it in the vision tower; 2.5 s with two
images); on the CPU 0.4-0.9 s for text and 9-17 s with images.

### Image decisions

`autotrust/JEV-27B-VL` (Qwen3.8-27B with its vision tower, unchanged, plus JEV-27B's System 1 as
an unmerged `adapter_vllm/` LoRA with an `lm_head` readout) and JEV-9B's `vl/` bundle (the same
adapter as `adapter_vllm/` with the layer names of the multimodal class, for the unmodified
`Qwen/Qwen3.5-9B`) decide over images too. As in the reference server (`serve_decide.py`), the
state is a list mixing text and images, and each image goes where it appears:

```text
[kind] noul
[state] Photo: <|vision_start|><|image_pad|> x 256<|vision_end|>
[question] Is this scenario one where: the image shows a wild animal?
[options]
false
true
[decision]:
```

No chat template, no system prompt: text parts as they are, an image as its vision tokens (M-RoPE
positions as in chat), then the readout, adapter, bias and temperatures of the text decisions.
The head never saw an image in training, so these decisions are zero-shot. The reference server
takes images only (no video); so does oxinfer.

```sh
# JEV-9B on the multimodal Qwen3.5-9B: only vl/ is fetched from autotrust/JEV-9B (org/name/subdir)
oxinfer --model Qwen/Qwen3.5-9B --decision-from autotrust/JEV-9B/vl --decide choice \
  --image scene.png --state "Task: pick up the red cube." \
  --question "Where is the red object relative to the blue one?" --option left --option right

# JEV-27B-VL brings its own assets; JSON requests place images anywhere in the state
oxinfer --model autotrust/JEV-27B-VL --decide-json requests.json
```

`--image` (repeatable) puts images before the state text; `--max-soft-tokens N` caps the tokens of
an image (default: the processor's limits; each token covers 32x32 pixels). In JSON (`--decide-json`,
`Engine::decide_json`, `oxi_decide`) an image part is `{"image": REF}` or `{"type": "image_url",
"image_url": {"url": REF}}`, with REF a base64 `data:` URL, an `http(s)://` URL (fetched with the
`hf` feature's client), a `file://` URL or a path; `"attachments": {"images": [...],
"max_soft_tokens": N}` is the JSON form of `DecisionRequest::attachments`. In Rust:

```rust
let d = engine.decide(
    &DecisionRequest::noul(
        serde_json::json!(["Photo: ", {"image": "photo.jpg"}]),
        "Is this scenario one where: the image shows food or cooking?",
    ),
)?;
// or: DecisionRequest::noul("caption text", "...").with_images(["photo.jpg"])
// ImageSource::Bytes(png).to_part() makes a state part from encoded bytes
```

A text-only checkpoint (the `autotrust/JEV-9B` folder itself has no vision tower) refuses images
with a message naming the multimodal base to load instead.

Agreement with the reference server's math on `transformers` + `peft` (HF processor, unmerged
`adapter_vllm`, LM-head logprobs over the verbalizer tokens + bias, per-kind temperature; f32
compute on the bf16 weights), 11 image decisions (2-3 noul, 2 score, 2- and 4-option, two
12-option and a 20-option wide-label choice; one or two images of 140-396 vision tokens, as data
URLs, `image_url` parts and paths) and the 5 text decisions of the text check:

| oxinfer (bf16 weights) | images: max \|Δp\| | max KL | text: max \|Δp\| |
|---|---|---|---|
| Qwen3.5-9B + JEV-9B `vl/`, CPU | 3.7e-6 | 1.6e-8 | |
| Qwen3.5-9B + JEV-9B `vl/`, Metal | 1.1e-3 | 5.7e-6 | 1.3e-4 |
| JEV-27B-VL, CPU | 8.3e-6 | 4.5e-10 | |
| JEV-27B-VL, Metal | 6.0e-4 | 9.4e-7 | 8.7e-5 |

Prompt token counts match the processor's exactly; the Metal differences come from its half-precision
matrix units (`gpu_exact` removes them). Text decisions and chat give the same output as before
images were supported, bit for bit.

Latency of one image decision (512x512 photo = 256 vision tokens, 298 prompt tokens; M5 Max, while
other jobs loaded the machine, so upper bounds): Qwen3.5-9B + `vl/` 1.0-1.5 s on Metal (vision tower
0.4-0.6 s, prefill 0.6-0.9 s) and 7.4 s on the CPU; JEV-27B-VL (bf16) 3.3 s on Metal (tower 0.6 s)
and 22 s on the CPU.

## Runtime interventions: neuron edits, steering, abliteration, SAE features

oxinfer can change what a model computes while it runs, without touching the weights: edit
chosen MLP neurons, add steering vectors to the residual stream, project a direction out of it
("abliteration" at run time), and read or steer sparse-autoencoder (SAE) features. It can also
make steering vectors itself, from two sets of prompts. Everything is reachable from the CLI,
the library (`Engine::set_interventions`, `oxinfer::intervene`) and the C API
(`oxi_engine_set_interventions`). Interventions are set per request: changing a strength
between two messages rebuilds a small set of hooks, not the model, and GPU decoding keeps its
compiled plan. The shared code is in `src/intervene.rs` and `src/sae.rs`, wired into Qwen3.5 /
3.8 (dense and MoE), Qwen3, Llama 3, Gemma 4 / 3, Mistral Large 4 and DeepSeek V4.

**Where the hooks are.** Layers are 0-based, as in the checkpoint's module names
(`model.layers.14`).

| Hook | Meaning |
|---|---|
| neuron `L:n` | element `n` of the input of `layers.L.mlp.down_proj`, that is `act(gate) * up`. MoE layers: per routed expert (`"expert": 3`; the expert's activation before its routing weight scales the output) or the shared expert (`"expert": "shared"`). Mistral Large 4: layer 0's dense MLP (24576 neurons) takes no `expert`; the MoE layers have the shared expert's 6144 (`mlp.shared_experts.down_proj`) and each routed expert's 6144 (`mlp.experts.down_proj[e]`). |
| `post` of layer L | the layer's output, `hidden_states[L + 1]` in transformers. This is where llama.cpp control vectors and Qwen-Scope SAEs (`resid_post`) act. |
| `mid` of layer L | the residual after the attention (or Gated DeltaNet) block was added |
| `pre` of layer L | the layer's input: the output of layer L - 1, or the embeddings for layer 0 |

**DeepSeek V4** has no single residual stream: each token carries `hc_mult` (4) copies of the
hidden state (`[tokens, 4, hidden]`), which every block reads through a learned weighted sum and
writes back through learned weights and a mixing matrix. The hooks are defined on that stream
(`State::point_copies` in `src/intervene.rs`):

- **The points.** `pre` of layer L is the 4-copy tensor entering `layers.L` (`Block.forward`'s
  input in DeepSeek's `inference/model.py`; the repeated embedding for layer 0). `mid` is the
  tensor after the attention half (the output of its `hc_post`, the input of the MoE half's
  `hc_pre`). `post` is the layer's output, which is the next layer's `pre`.
- **The residual at a point** is the mean of the 4 copies. Capture, `--make-direction`, the
  `resid_median` calibration, SAE encoding and inspection read that mean, so vectors and SAEs
  have `hidden` values as for every other family.
- **Edits write to every copy**, so that the mean changes exactly as a single stream would:
  steering adds `alpha * v` to each copy; ablation projects `r` out of each copy (it is then
  absent from every weighted sum a block reads, not only from the mean, and stays absent
  through the mixing, whose weights are scalars per copy; hooking Arditi's block outputs before
  they are spread over the copies gives the same logits, which the reference checks); an SAE
  feature edit is computed on the mean and its change `(a' - a) * W_dec[f]` is added to each
  copy.
- **Neurons.** Every layer is a MoE (the first 3 hash-routed), so an edit always names an
  `expert`: `"shared"` is the input of `ffn.shared_experts.w2` (2048 neurons), an id the input of
  `ffn.experts.E.w2` (2048 each, 256 experts), both `silu(min(gate, 10)) * clamp(up, -10, 10)`,
  for a routed expert before its routing weight. There is no dense MLP.

**Where they apply.** `positions` is `all` (the default), `prompt` or `generated`. `generated`
covers the last prompt token and every generated token, so every position whose output is
sampled. `prompt` covers the prompt tokens before that.

### The spec

The spec is a JSON object. It goes to `--interventions FILE`, to the engine option
`interventions`, or to `oxi_engine_set_interventions`. The library type is
`oxinfer::intervene::Spec`.

```json
{
  "neurons": [
    {"layer": 14, "neuron": 8949, "dose": 1.0, "unit": "resid_median"},
    {"layer": 23, "neuron": 4790, "op": "scale", "value": 2.0},
    {"layer": 18, "neuron": 6052, "op": "zero"},
    {"layer": 5, "neuron": 7, "expert": "shared", "op": "clamp", "min": -1, "max": 1, "positions": "generated"}
  ],
  "steer": [{"file": "tone.safetensors", "layer": 12, "alpha": 1.5, "point": "post"}],
  "ablate": [{"file": "refusal.safetensors", "direction_layer": 12}],
  "sae": [{"sae": "Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50/layer12.sae.pt",
           "features": [{"id": 10155, "op": "clamp", "value": 4.0}]}],
  "sae_inspect": [{"sae": "Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50/layer12.sae.pt", "top_k": 5}]
}
```

- **Neuron edits.** `op` is one of:
  - `scale` multiplies the activation by `value`;
  - `zero`;
  - `add` adds `value`; `dose` is shorthand for this;
  - `set` replaces it with `value`;
  - `clamp` limits it to `[min, max]`.

  `"unit": "resid_median"` multiplies `value` by `median_t |h_L,t| / |W_down[:, n]|`. Here `h_L`
  is the output of layer L, measured on the unedited model over positions 1 onward (position 0,
  an attention sink with a huge norm, is skipped). With this unit, a dose of 1 moves the
  residual by the typical norm of that layer's output.

  The median comes from one unedited pass over the first prompt the edits see, and is then kept.
  You can instead measure it on a text of your choice (`--calibrate-text T`,
  `Engine::calibrate(text)`) or give it outright with `"resid_median": F`.

  The edit is added at every position (`add`, not `set`). Even-length medians average the two
  middle values (numpy's convention). The reference script applies the same edit to the same
  tensor through a PyTorch pre-hook on `down_proj`.
- **Steering.** Adds `alpha * v`. `v` is either an inline `vector` (with a `layer`) or comes from a
  `file`. With a file you get every layer it holds, or only `layer` / `layers`. `normalize: true`
  scales `v` to unit norm first, so `alpha` becomes the norm of what is added.
- **Ablation.** Applies `x -= (x . r) r` with `r` normalized. It runs on the embeddings and after
  every attention and MLP block, in all layers (or `layers`). This is Arditi et al.'s directional
  ablation; FailSpy's and mlabonne's tools hook the same places. Several directions are
  orthonormalized together.

  At one point, added vectors go in first and then directions are projected out. So an ablated
  direction also removes its part of a steering vector, just as a hook on the next layer's input
  would.
- **SAE features.** `op` is one of:
  - `clamp` sets the activation to `value` (Neuronpedia's clamp);
  - `scale` multiplies it;
  - `zero`;
  - `add` adds `value * W_dec[f]`, whatever the activation (SAELens / Neuronpedia / Qwen-Scope
    steering).

  The SAE's error term is kept: only `(a' - a) * W_dec[f]` is added to the residual. All edits of
  one SAE read the same encoding of the incoming residual. `sae_inspect` records the `top_k` most
  active features of every position: `Engine::sae_features()`, `oxi_engine_sae_features_json`,
  printed by the CLI.

**Vector files.** These are read by `steer` / `ablate` and written by `--make-direction`:
- **safetensors** with 1-D tensors `layers.N` (f32/bf16/f16). `layers.N` is the output of layer N.
  A single `[layers, hidden]` or `[layers + 1, hidden]` matrix is also accepted; the second form
  follows transformers' `hidden_states` and its row 0 is dropped. So is a lone 1-D vector.
- **GGUF control vectors**, as llama.cpp `--control-vector` and repeng's `export_gguf` write them:
  `direction.N` is added to the output of layer N, exactly as llama.cpp applies it. The same
  vectors as GGUF and as safetensors give bit-identical logits.
- **PyTorch `torch.save`** state dicts with the same names.

**SAEs.** Two formats load:
- **Qwen-Scope** (`Qwen/SAE-Res-*`, one `layerN.sae.pt` per layer plus the repo's `config.json`;
  TopK, `topk(relu(x W_enc^T + b_enc), k)`, `resid_post`).
- **SAELens directories** (`cfg.json` + `sae_weights.safetensors`; ReLU, JumpReLU or TopK,
  `apply_b_dec_to_input`; only `normalize_activations: none`).

Qwen-Scope has SAEs for Qwen3.5-2B-Base (32K features, k 50 or 100, 537 MB per layer), 9B-Base,
27B (80K features, the one Neuronpedia lists as `qwen3.5-27b / 31-qwenscope-res-80k`) and
35B-A3B-Base. There are none for 0.8B.

### Command line

```sh
# The user-reported "evil persona" edit on Qwen3.5-4B (layers 0-based, as in the module names)
oxinfer -m Qwen/Qwen3.5-4B -p "Write a short poem about the sea." \
  --neuron-edit 14:8949,dose=1,unit=resid_median --neuron-edit 13:7956,dose=-1,unit=resid_median

# A refusal direction from two prompt sets (one per line, or .json / .jsonl / .csv), then
# runtime abliteration with the layer-12 direction, and the opposite: activation addition
oxinfer -m Qwen/Qwen3.5-2B --make-direction harmful.json,harmless.json -o refusal.safetensors
oxinfer -m Qwen/Qwen3.5-2B --ablate refusal.safetensors:12 -p "..."
oxinfer -m Qwen/Qwen3.5-2B --steer refusal.safetensors:1.0 --steer-layers 12 -p "..."

# A llama.cpp / repeng control vector, generated tokens only
oxinfer -m Qwen/Qwen3.5-2B --steer happy.gguf:2 --positions generated -p "Describe a bus stop."

# SAE: inspect, then steer (the SAE file is fetched from Hugging Face)
oxinfer -m Qwen/Qwen3.5-2B-Base --raw -p "I adopted a cat." \
  --sae Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50/layer12.sae.pt --sae-inspect 5
oxinfer -m Qwen/Qwen3.5-2B-Base --raw -p "My favorite thing to do is" \
  --sae Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50/layer12.sae.pt --sae-feature 10155:add=4

# Pooled layer outputs of one prompt (layers.N, for your own analysis)
oxinfer -m Qwen/Qwen3.5-0.8B -p "..." --capture-activations acts.safetensors --capture-pool mean
```

Other flags:
- `--interventions FILE|JSON` takes the full spec.
- `--steer-point pre|mid|post` and `--ablate-layers A-B` place the hooks.
- `--sae-layer N` overrides the SAE's layer.
- `--capture-pool last|mean` pools the captured layer outputs. `last` is the last prompt token; `mean`
  averages positions 1 onward.

`--make-direction` wraps every prompt in the chat template unless `--raw` is given. It writes
`mean(first set) - mean(second set)` per layer, not normalized. With `alpha` 1, steering then
moves the second set's mean onto the first's.

### Library and C API

```rust
use oxinfer::engine::{Engine, EngineOptions, Flow};
use oxinfer::intervene::{Pool, Spec};
let mut engine = Engine::load("Qwen/Qwen3.5-2B", &EngineOptions::default())?;
// A direction from two prompt sets, saved for later use.
let d = engine.make_direction(&harmful, &harmless, Pool::Last, true)?;
oxinfer::intervene::save_vectors("refusal.safetensors".as_ref(), &d, &[])?;
// Change the interventions between requests: no reload.
for alpha in [0.5, 1.0, 2.0] {
    engine.set_interventions(Some(Spec::from_json(&format!(
        r#"{{"steer":[{{"file":"refusal.safetensors","layer":12,"alpha":{alpha}}}]}}"#
    ))?))?;
    engine.reset();
    let r = engine.send("Write a haiku about tea.", &mut |_| Flow::Continue)?;
}
engine.set_interventions(None)?;
```

`Engine::capture` returns the pooled layer outputs of a prompt list, and `Engine::calibrate`
measures the `resid_median` unit on a text. `examples/steer.rs` runs a prompt file under several
specs and strengths and writes the replies as JSONL; it produced the refusal and tone numbers
below.

The C API mirrors the library:
- `oxi_engine_set_interventions(engine, json)` (NULL clears) and
  `oxi_engine_interventions_json`;
- `oxi_engine_calibrate(engine, text)`;
- `oxi_engine_sae_features_json`;
- `oxi_engine_make_direction(engine, {"positive": [...] | "file", "negative": ..., "pool", "chat", "output"})`;
- the load option `interventions=JSON|@FILE`.

### Agreement with PyTorch hooks

`tools/ref_intervene.py` applies the same JSON spec to a transformers model with forward hooks
and compares logits. The hooks are its own implementation:
- a pre-hook on `down_proj` for neuron edits;
- output hooks on the layer and on the attention block for steering;
- Arditi's three hook sites for ablation;
- the release's own `topk(relu(...))` for Qwen-Scope.

Every row below compares the last prompt position, with oxinfer on the CPU against float32
PyTorch:

| Model | Interventions | max \|logit diff\| | KL | without interventions |
|---|---|---|---|---|
| Qwen3.5-0.8B | 4 neuron edits (scale, zero, dose resid_median, clamp), steering at 2 layers + a `mid` hook on generated positions, ablation in all layers | 2e-5 | 5e-8 | 4e-5 |
| Qwen3.5-4B | the two-neuron dose edit above | 4e-5 | 1e-8 | 6e-5 |
| Qwen3.5-2B-Base | Qwen-Scope layer 12: clamp, add, zero (TopK, error term kept) | 4e-5 | 3e-8 | |
| Qwen3-0.6B | neuron edits, steering (post, mid), ablation | 7e-5 | 5e-8 | |
| Gemma 4 E2B | neuron edits incl. resid_median, steering (post, mid), ablation | 2.2e-4 | ~0 | 1.3e-4 |
| Mistral Large 4, tiny random checkpoint (`tools/mistral4_tiny.py`: dense layer 0, 16 experts + shared) | 8 neuron edits (dense scale and resid_median dose; shared add and clamp; routed set, zero, resid_median dose on generated positions, scale on prompt positions), steering (post, mid on generated, pre normalized), ablation in all layers | 2e-5 | 1e-7 | 1.7e-4 |
| DeepSeek V4, tiny random checkpoint (`tools/deepseek4_tiny.py`: 6 layers, hash-routed layer 0, window-only, CSA and HCA layers, 8 experts + shared, 4 hyper-connection copies) | 32 neuron edits (shared add, clamp, resid_median dose; routed set, zero, scale on prompt positions, resid_median dose on generated positions, scale; one neuron of all 8 experts of 3 layers), steering (post at 2 layers, mid on generated, pre normalized, post on prompt positions), a TopK SAE (clamp, add, zero, scale) and a JumpReLU SAE (clamp, scale, add on generated positions), ablation in layers 0-1; a 106-token prompt | 2e-5 | 2e-8 | 3e-5 |

The `resid_median` values matched to 4 digits (for example L13 9.1125, L14 10.1972 on the
4B prompt).

DeepSeek V4 was measured on the tiny checkpoint only (the real model is a 97 GB download and
was not run with interventions). The reference (`ref_intervene.py`, transformers'
`DeepseekV4ForCausalLM`) hooks the same tensors: the 4-copy input and output of `layers[L]`,
the input of `ffn_hc` for `mid`, the shared expert's `down_proj` input, and the experts' loop
with the clamped SwiGLU for routed experts. Each kind alone agreed to 3e-5 or better (KL below
2e-7): the neuron edits, the steering entries, ablation of one direction in all layers and of
two (the second in layers 1-3 on generated positions), the TopK SAE, the JumpReLU SAE. With
`--block-hooks` the reference ablates on Arditi's sites instead (layer input, attention and
MoE outputs before they are spread over the copies) and gives the same logits (3e-5). The
combined spec gave 1e-5 with the prompt prefilled in passes of 7 tokens and of 1, and 2e-5 with
the weights streamed (`--stream`). Calibration medians matched to the 4 digits printed (L3
30.7286, L4 33.1161). Every spec gave the same 16 of 16 greedy tokens as `generate` under the
hooks. Captured layer outputs (mean pool) matched the mean of the copies in PyTorch to 1e-6
relative, and SAE inspection gave the same top-3 features and activations. Through the engine,
decoding with a (tiny random) DSpark draft gave the same replies as plain decoding under the
combined spec and under ablation (2 prompts, 24 tokens each).

Greedy generation on Qwen3.5-4B with the dose edits produced the same 40 of 40 tokens as
transformers' `generate` under the same hooks, with oxinfer decoding on the GPU. On the tiny
Mistral Large 4, every spec (all of the row above, and neurons, steering and ablation alone) gave
the same 8 of 8 greedy tokens as `generate` under the hooks; the reference edits the routed
experts by running transformers' fused-expert loop with the edit in place.

Qwen-Scope inspection agreed exactly: the top-5 features at all 25 positions of a sentence had
the same ids in the same order as the release's encoder, with activations equal to the 2
printed decimals.

GPU decode against the CPU (`OXINFER_GPU_DECODE_CHECK=1`, every step) gave the following KL:

| Model and interventions | KL |
|---|---|
| Qwen3.5-0.8B, all of the above | ≤ 2e-12 |
| Qwen3.5-0.8B, ReLU and JumpReLU SAE features | ≤ 2e-12 |
| a Qwen3.5 MoE test model, routed and shared-expert neuron edits | ≤ 1e-11 |
| Qwen3-0.6B | ≤ 3e-11 |
| Gemma 4 E2B | ≤ 1.3e-10 (1.2e-11 without interventions) |

### Results

**The Qwen3.5-4B persona edit:**
- **Edit:** +1 dose on layer 14 neuron 8949 and -1 on layer 13 neuron 7956.
- **Effect on a poem:** a gentle sea poem turns into a menacing one.
- **Effect on 12 neutral prompts** (introductions, advice, a birthday party plan, recipes; 120
  tokens each):
  - Replies containing hostile vocabulary went from 2 to 6.
  - Hostile words in total went from 2 to 20.
  - One reply proposed a "hostile takeover" as a party plan.
- **Calibration:** the medians come from the first prompt, on the unedited model (for the poem
  prompt: L13 9.11, L14 10.20).

**Runtime abliteration** (Qwen3.5-2B):
- **Direction:** the difference of means between 128 harmful and 128 harmless training
  instructions (Arditi et al.'s public splits), taken at the last token of the chat-templated
  prompt. Making it took 13 s.
- **Choosing a layer** (32 validation prompts, layer-L direction ablated everywhere):

  | Direction from | Refusal rate |
  |---|---|
  | no ablation | 81% |
  | layer 6 | 56% |
  | layer 8 | 38% |
  | layers 10 to 18 | 0% |

- **Held-out test** (100 harmful prompts, 64 tokens each, layer-12 direction): refusals fell from
  89% to 3%.
- **Held-out harmless prompts** (100):
  - ablation left refusals at 2% → 1%, and the replies stayed fluent and on topic;
  - adding the same direction at layer 12 (activation addition, alpha 1) made the model refuse
    92% of them.
- **How refusals were counted:** refusal phrases in the first 300 characters. The completions are
  not reproduced here.

**Tone steering** (Qwen3.5-2B, a happy-minus-sad direction from 24 + 24 prompts, layer 12, 10
neutral "two sentences about X" prompts):
- **alpha +1:** enthusiastic replies with exclamation marks and emoji ("the ultimate two-wheeled
  joyride").
- **alpha -1:** bleak replies ("its metal frame and chain eventually rust and become useless").
- **alpha ±2:** the text degenerates.

**SAE features:**
- **Inspection** finds concept features. For example, feature 10155 at layer 12 of the 2B-Base
  SAE fires on "cat", "kitten" and "Cats".
- **Single-feature steering** on the 2B base model nudged completions mildly (adding 3 to 7 times
  the unit decoder vector at generated positions). Larger strengths (10 and up, against residual
  norms of a few units at that layer) broke the text. Concept steering with one Qwen-Scope
  feature is weak at this size; the 9B / 27B SAEs are the ones Qwen's demo uses.

**The papers:**
- **H-Neurons** (arXiv 2512.01797) scale the `down_proj` input of selected neurons by
  alpha in [0, 3] (`op: scale`). They publish neither neuron lists nor Qwen results, so there is
  nothing to reproduce directly.
- **Perturbation Probing** (arXiv 2604.27401) scales the `W_down` columns of its Qwen3.5-2B
  neurons, for example L23:n4790, L18:n6052 and L14:n200 (2x amplification, zeroing for
  ablation). Scaling a column is the same as `op: scale` on that neuron. Their full 2B lists are
  not published.

### What runs where, and what it costs

| | CPU | Metal prefill | Metal whole-pass decode |
|---|---|---|---|
| neuron edits (dense, shared, routed experts) | yes | yes (hooks run between the GPU matmuls; MoE layers with routed-expert edits keep their experts on the CPU) | yes |
| steering, ablation | yes | yes | yes |
| SAE ReLU / JumpReLU features, SAE `add` | yes | yes | yes (up to 64 directional items per hook point) |
| TopK SAE clamp / scale / zero, SAE inspection, capture | yes | yes | no: those decode steps run on the CPU |

Every combination runs. When the GPU graph cannot take a hook, decoding falls back to the CPU on
its own, and the CLI says so when it loads the spec.

Decode speed on Qwen3.5-2B (bf16, M5 Max, 200 tokens, best of 3):

| Spec | Metal | CPU |
|---|---|---|
| none | 130.4 tok/s | 66.9 tok/s |
| steering at 1 layer | 130.4 | 66.6 |
| ablation at 49 hook points | 125.8 (-3.5%) | 66.1 |
| 3 neuron edits | 129.9 | 66.5 |

A TopK SAE clamp on 2B-Base moves decoding to the CPU: 124.6 → 63.0 tok/s, which is about the
CPU's own speed.

### Limits

- **Core ML (ANEMLL) models** refuse interventions.
- **Mistral Large 4** has no whole-pass GPU decode, so its hooks always run on the CPU between
  the (GPU or CPU) matmuls; a prefill pass with routed-expert edits keeps that layer's experts on
  the CPU. Typed decisions (`decision_from`) are not offered for it: a decision bundle is a LoRA
  adapter and head trained on one backbone (Qwen3.5 / 3.8, Gemma 4), none exists for this model,
  and its layers take no LoRA.
- **DeepSeek V4** is in the same position: no whole-pass GPU decode exists for this family, so
  on every device the hooks run on the CPU between the matmuls, in prefill and in decoding, and
  the CLI's "GPU decode keeps running on the GPU" line does not apply. A pass with routed-expert
  edits keeps that layer's experts on the CPU. Streamed weights work unchanged (the hooks touch
  activations only; the `resid_median` unit reads one `down_proj` column, for a routed expert
  through the streamer). The CLI decodes without the DSpark draft when interventions are
  given; the engine keeps the draft, whose stages are not hooked: they read the edited model's
  residual and the edited model verifies every draft, so the output follows the edited model.
  An SAE must have been trained on the mean of the copies (none is published for this model),
  and agreement was measured on a tiny random checkpoint only.
- **Speculative decoding.**
  - Qwen MTP drafts come from the unedited MTP head, but the edited main model verifies every
    draft, so the output follows the edited model.
  - The Gemma drafter path of the CLI is not covered.
- **SAEs.**
  - Only residual-stream outputs (resid_post) and `normalize_activations: none` are supported.
  - There is no GemmaScope `.npz` loader.
  - Weights are loaded as f32: 537 MB per Qwen-Scope 2B layer.
- **Making directions.**
  - Capture runs on the CPU paths (prefill matmuls still use the GPU).
  - Pooling is last-token or mean only. Arditi's per-position candidates and his
    bypass/induce/KL layer selection are not built in; the layer sweep above was done with
    `examples/steer.rs`.
- **Calibration.** The `resid_median` unit is measured once, on the first prompt (or the
  calibration text), and kept until `Engine::clear_calibration`.

## Image generation

A diffusers folder (one with `model_index.json`) switches oxinfer to text-to-image with Stable
Diffusion 1.x / 2.x, which includes SD 1.5, SD 2.1 and the one-step SD-Turbo, and with Stable
Diffusion XL (see [below](#stable-diffusion-xl)) (`src/sd/`):

```sh
./target/release/oxinfer --model models/sd-turbo -p "a red fox in a snowy forest" -o fox.png
./target/release/oxinfer --model models/sd-v1-5 -p "a watercolor lighthouse" --negative "blurry" \
    --steps 25 --guidance 7.5 --sampler euler_a --size 512x512 --seed 7 -o lighthouse.png
```

With `--init-image FILE` the same models edit an image instead (image-to-image, inpainting with
`--mask`, instruction edits with InstructPix2Pix): see [Editing images](#editing-images).

Turbo models (recognised by name) default to 1 step without guidance, others to 25 Euler-ancestral
steps at guidance 7.5. The CLIP tokenizer and text encoder, the UNet (ResNets, spatial transformers
with GEGLU, skip connections), the VAE decoder and the Euler samplers are implemented here.
Activations are channels-last, so every convolution unfolds into the same matmul the language models
use: the SIMD kernels apply on every CPU, and GPU backends run convolutions directly as an implicit
GEMM (`Backend::conv2d`: the kernel gathers each pixel's neighbourhood itself; Metal and CUDA
measured, WebGPU in the browser). Weights are converted from fp16 to bf16 at load.

`tools/ref_sd.py` runs diffusers on the same noise and compares every stage (`OXINFER_SD_DUMP=dir`).
With the weights rounded to bf16 on both sides the outputs agree to about 1e-6 (79 dB PSNR on
the SD-Turbo image, 75 dB on SD 1.5 after 4 guided steps), so the only difference from the fp32
reference is the bf16 weight storage (47 dB).

512x512 on the M5 Max:

| | Text | UNet pass | VAE decode | SD-Turbo image | SD 1.5, 25 steps + guidance |
|---|---|---|---|---|---|
| CPU only (18 threads) | 0.08 s | 0.91 s | 2.87 s | 3.9 s | |
| Metal | 0.08 s | 0.37 s | 0.66 s | 1.1 s | 16 s |
| Browser (Chrome, WebGPU, one CPU thread) | 0.2 s | 1.1 s | 1.9 s | 3.2 s | |

With a GPU, convolutions, attention and the group and layer norms run on it: each norm is fused
into the convolution or projection that consumes it (a statistics kernel, then normalization and
SiLU applied while the input is loaded), and attention runs its two products as GPU matmuls with a
softmax kernel between them. What stays on the CPU is element-wise: residual adds, the GEGLU gate,
upsampling and skip concatenation. `OXINFER_GPU_CHECK=1` recomputes every device op on the CPU and
prints the worst relative error per kind (about 4e-6 for attention, 1e-6 for the fused norms, on
Metal and WebGPU). SDXL runs through this pipeline (see [Stable Diffusion XL](#stable-diffusion-xl));
SD 3 / 3.5 and FLUX.1 have their own pipelines (see
[Stable Diffusion 3 / 3.5](#stable-diffusion-3--35-text-to-image-and-image-to-image) and
[FLUX.1 Kontext](#flux1-kontext-image-editing-from-instructions-and-flux1-text-to-image)).

**LoRA.** `--lora FILE[:SCALE]` (repeatable, scale 1 by default) merges a LoRA into the UNet and
text-encoder weights at load, `W += scale * alpha / rank * up @ down` in f32 before the bf16
conversion, so generation runs at the base model's speed (`src/sd/lora.rs`). Convolution LoRAs
(LoCon: a `[rank, in, 3, 3]` down and a `[out, rank, 1, 1]` up) are merged the same way. Both
common layouts are read: kohya / A1111 (`lora_unet_down_blocks_0_..._to_q.lora_down.weight`,
`.alpha`, `lora_te_text_model_...`) and diffusers / PEFT (`unet.<module>.lora_A.weight`, `.lora.down`,
`text_encoder.`, with the alpha from the `lora_adapter_metadata` header). Keys that match no weight
of the model, and LyCORIS kinds other than plain LoRA / LoCon (LoHa, LoKr, DoRA), are skipped and
reported; a file that matches nothing, or whose shapes belong to another base model, is an error.
`FILE` may also be a Hugging Face path, `org/name/file.safetensors` (or `org/name` when the repo
holds one safetensors file), fetched into the cache like models; or download it yourself:

```sh
curl -L -o lcm-lora.safetensors \
    https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/resolve/main/pytorch_lora_weights.safetensors
./target/release/oxinfer --model models/sd-v1-5 --lora lcm-lora.safetensors \
    --steps 4 --guidance 1 -p "a red fox in the snow" -o fox.png  # Euler ancestral (default)
./target/release/oxinfer --model models/sd-v1-5 --lora KoRo8888/ufotable_artstyle_locon_v1:0.8 \
    -p "a girl standing in a forest, anime screencap" -o forest.png
```

Against diffusers' `load_lora_weights` + `fuse_lora` (`tools/ref_sd.py --lora`, bf16 weights on
both sides), on SD 1.5: LCM-LoRA (kohya keys, 278 UNet modules including convolutions), 4 Euler
steps, 79.6 dB; the ufotable LoCon (kohya, 278 UNet + 72 text-encoder modules) at 0.8, 20 steps
at guidance 7.5, 78.4 dB, and the same file re-saved by diffusers in the PEFT layout gives the
identical image; both together (LCM 1.0 + ufotable 0.6), 82.3 dB. Scale 0 reproduces the base
model bit for bit. In the library, `sd::Pipeline::load_with_loras(dir, &[LoraSpec { path, scale }],
report)` does the same (`examples/sd_lora.rs`); in the browser build, `web_sd_load` takes the
LoRA files (`sdLoad(dir, [{ path, scale }])` in `oxinfer.js`).

### Stable Diffusion XL

An SDXL diffusers folder (`StableDiffusionXLPipeline` in `model_index.json`, with `text_encoder_2/`
and `tokenizer_2/`) loads through the same pipeline, CLI, library, C API and server entry points;
the model type is read from the folder. SDXL base 1.0 and SDXL-Turbo were run:

```sh
./target/release/oxinfer --model stabilityai/stable-diffusion-xl-base-1.0 \
    -p "a watercolor painting of a lighthouse on a cliff" --negative "blurry" -o lighthouse.png
./target/release/oxinfer --model stabilityai/sdxl-turbo -p "a red fox in a snowy forest" -o fox.png
# image-to-image and inpainting (mask blending), as for SD 1.x
./target/release/oxinfer --model stabilityai/stable-diffusion-xl-base-1.0 --init-image in.png \
    --strength 0.6 -p "an oil painting of a lighthouse at sunset" -o painting.png
# SDXL-Lightning: a few-step LoRA on the base model, over "trailing" timesteps
./target/release/oxinfer --model stabilityai/stable-diffusion-xl-base-1.0 \
    --lora ByteDance/SDXL-Lightning/sdxl_lightning_4step_lora.safetensors \
    --sampler euler_trailing --steps 4 --guidance 0 -p "a red fox in a snowy forest" -o fox.png
```

What SDXL adds is in `src/sd/xl.rs`; the rest is the SD 1.x code, which takes its structure from
the weights and configs:

- **Two text encoders**, CLIP ViT-L and OpenCLIP bigG with its text projection. The penultimate
  hidden states of both, concatenated per token (768 + 1280), are the cross-attention context;
  the projected end-of-text state of the second is the pooled embedding. An empty negative prompt
  conditions on zeros when the pipeline says so (`force_zeros_for_empty_prompt`: the base model,
  not Turbo).
- **Added conditioning.** The pooled embedding and six size ids (original size, crop offset,
  target size; each a 256-wide sinusoidal embedding) go through the UNet's `add_embedding` into
  the time embedding. As in diffusers the ids default to the output size and no crop;
  `Pipeline::size_cond` sets them in the library (`examples/sd_xl.rs`).
- **UNet.** Transformer depth per block (1, 2, 10), no attention at the top level, two
  downsamplings (so sizes are multiples of 32), linear projections: all read from the weights.
- **VAE.** The scaling factor (0.13025) comes from the VAE config. Everything here is computed
  in f32, so the original SDXL VAE's fp16 weights decode without the overflow that the "fp16-fix"
  VAE works around in half-precision runtimes (that VAE has the same architecture; not run).
- **Samplers and defaults.** Without `--sampler` an SDXL model uses the scheduler its config names
  (Euler for the base model, Euler ancestral for Turbo), with guidance 5 and 25 steps at
  1024x1024 for the base model and 1 step without guidance at 512x512 for Turbo.
  `--sampler euler_trailing` / `euler_a_trailing` run over "trailing" timesteps whatever the
  config says, which few-step LoRAs such as SDXL-Lightning need on the base model.
- **LoRA.** `--lora` reads kohya SDXL files, whose UNet keys follow the original (SGM) layout
  (`lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q`, `lora_unet_output_blocks_...`,
  `lora_unet_middle_block_...`) with `lora_te1_` / `lora_te2_` for the two text encoders, the
  diffusers-named kohya form (`lora_unet_down_blocks_...`) and the diffusers / PEFT layout
  (`unet.`, `text_encoder.`, `text_encoder_2.`).

`tools/ref_sdxl.py` runs diffusers' `StableDiffusionXLPipeline`, `...Img2ImgPipeline` and
`...InpaintPipeline` (Euler scheduler) on the same random draws, as `tools/ref_sd.py` does for
SD 1.x. On the CPU (x86-64, AVX-512), 512x512 to keep the float32 reference small, weights rounded
to bf16 on both sides (relative L2 error; image also as 8-bit PSNR):

| Run | Text context | Pooled | First UNet output | Final latents | Image |
|---|---|---|---|---|---|
| SDXL-Turbo, 1 step | 1.9e-6 | 2.0e-6 | 4.5e-7 | 9.4e-6 | 2.8e-5, 79.8 dB |
| SDXL-Turbo, 4 steps | 1.9e-6 | 2.0e-6 | 4.5e-7 | 1.0e-5 | 4.2e-5, 78.8 dB |
| SDXL base, 2 steps, guidance 5 | 1.9e-6 | 1.7e-6 | 1.1e-5 | 1.6e-5 | 3.4e-5, 76.2 dB |
| SDXL base, 6 steps, guidance 5 | 1.9e-6 | 1.7e-6 | 2.8e-6 | 1.5e-4 | 2.6e-4, 68.7 dB |
| SDXL base, 6 steps, guidance 7, negative prompt | 1.9e-6 | 1.7e-6 | 3.5e-6 | 1.3e-4 | 3.2e-4, 67.8 dB |
| SDXL base img2img, strength 0.6, 10 steps, guidance 5 (encoder mean 2.6e-5) | 1.9e-6 | 1.8e-6 | 2.3e-5 | 2.3e-4 | 3.2e-4, 68.5 dB |
| SDXL-Turbo img2img, strength 0.5, 2 steps (encoder mean 2.6e-5) | 1.9e-6 | 1.8e-6 | 2.8e-5 | 4.4e-5 | 7.2e-5, 75.8 dB |
| SDXL base inpainting (mask blending), strength 1, 8 steps, guidance 5 | 1.9e-6 | 2.1e-6 | 2.2e-6 | 4.8e-5 | 1.0e-4, 74.7 dB |
| SDXL base inpainting (mask blending), strength 0.7, 10 steps, guidance 5 | 1.9e-6 | 2.1e-6 | 1.5e-5 | 6.9e-5 | 6.9e-5, 75.8 dB |
| + kohya LoRA at 0.9 (SGM names: 722 UNet, 72 + 192 text-encoder modules), 6 steps, guidance 5 | 9.7e-7 | 1.8e-6 | 2.4e-6 | 1.1e-4 | 3.8e-4, 68.5 dB |
| + kohya LoRA with convolutions (SGM names, 788 UNet modules), 6 steps, guidance 5 | 2.0e-6 | 2.4e-6 | 1.7e-6 | 8.1e-4 | 2.6e-3, 62.0 dB |
| + SDXL-Lightning 4-step LoRA (788 UNet modules), `euler_trailing`, 4 steps | 1.9e-6 | 2.0e-6 | 2.4e-7 | 5.7e-5 | 8.5e-5, 75.9 dB |

Each stage agrees to float rounding; over several guided steps the differences grow along the
trajectory (compare the 2-step and 6-step rows of the same prompt and noise). Against the
float32 reference without the bf16 rounding, the weight storage is what shows: 42.7 dB for the
SDXL-Turbo image, and 30.1 dB after 6 guided steps of the base model (text context 1.2e-3, first
UNet output 2.6e-3, the same picture to the eye). On CUDA (RTX 5080) the 6-step base run gives
69.5 dB and the img2img run 69.4 dB against the same references, and `OXINFER_GPU_CHECK=1` reports
at worst 2.5e-6 for attention, 6.9e-7 for the fused convolution norms and 3.5e-7 for the fused
linear norms; a 1024x1024 image at the defaults (25 steps, guidance 5) was generated there too.

Not supported: the SDXL refiner (its own conditioning: one text encoder and an aesthetic score),
SDXL InstructPix2Pix, and single-file checkpoints (`sd_xl_base_1.0.safetensors`: as for SD 1.x,
only the diffusers folder layout is read). A dedicated 9-channel SDXL inpainting UNet takes the
same code path as the SD 1.x one but was not run. Repo ids download the full-precision
safetensors when a repo has both (about 14 GB for SDXL base); a folder holding only the `.fp16`
files (7 GB) loads the same, and the weights are kept in bf16 either way.

## Editing images

Give a Stable Diffusion pipeline an image and it edits it instead of starting from noise. What
the edit does follows from the model (the UNet's input channels):

| Model | UNet input | `--init-image` (+ prompt) | `--mask` |
|---|---|---|---|
| timbrooks/instruct-pix2pix, vinesmsuic/magicbrush-jul7 | 8 (latents + image latents) | **instruction edit**: the prompt says what to change | not used |
| stable-diffusion-v1-5/stable-diffusion-inpainting (and other SD inpainting models) | 9 (latents + mask + masked-image latents) | needs a mask | repaints the white part |
| any text-to-image model (SD 1.x / 2.x, SD-Turbo, SDXL, SDXL-Turbo, + LoRAs) | 4 | image-to-image at `--strength` | inpainting by blending |

**Editing with a prompt: InstructPix2Pix.** For "change this picture like so", use
InstructPix2Pix. With the SD 1.5 LCM-LoRA (InstructPix2Pix is a fine-tuned SD 1.5, so the LoRA
fits it) an edit takes 4 UNet passes instead of 60, and the edits look as good:

```sh
./target/release/oxinfer --model timbrooks/instruct-pix2pix --lora latent-consistency/lcm-lora-sdv1-5 \
    --init-image cabin.png -p "make it winter, covered in snow" --steps 4 --guidance 1 -o snowy.png
# Without the LoRA: 20 Euler-ancestral steps, three guided UNet passes each
./target/release/oxinfer --model timbrooks/instruct-pix2pix --init-image cat.png \
    -p "turn the cat into a tiger" --image-guidance 1.5 --guidance 7.5 -o tiger.png
```

`--guidance` (default 7.5) is how strongly the result follows the instruction and
`--image-guidance` (default 1.5) how closely it keeps to the input: each step runs the UNet on
(instruction, image), (no text, image) and (no text, no image) and combines them as diffusers'
`StableDiffusionInstructPix2PixPipeline` does, `uncond + g * (text - image) + ig * (image -
uncond)`. Guidance at or below 1 (as with the LCM-LoRA) turns both off: one pass per step. The
image conditioning is the mode of the VAE encoder's latent distribution, unscaled, concatenated
to the latents.

InstructPix2Pix tends to restyle the whole picture ("add a red car" painted the cabin red in our
test); MagicBrush (`vinesmsuic/magicbrush-jul7`, InstructPix2Pix fine-tuned on human-annotated
local edits, same architecture, CreativeML OpenRAIL-M, PyTorch `.bin` weights which load directly)
added the car and left the rest alone. It is a drop-in alternative at the same 20 guided steps; the
LCM-LoRA does not carry over to it (its scheduler config uses "leading" timesteps, and the 4-step
result was unusable).

**Image-to-image.** With a text-to-image model the image is encoded, noised to `--strength`
(default 0.8) and denoised towards the prompt: as diffusers' `StableDiffusionImg2ImgPipeline`,
the last `int(steps * strength)` steps of the schedule run, starting from `latents + sigma *
noise` at the first of them. Turbo models default to just enough steps for one to run (strength
0.5: 2 steps, 1 run):

```sh
./target/release/oxinfer --model models/sd-turbo --init-image cabin.png --strength 0.5 \
    -p "a cabin in a meadow at sunset, golden hour" -o sunset.png
./target/release/oxinfer --model models/sd-v1-5 --init-image cabin.png --strength 0.6 \
    -p "an oil painting of a cabin in a meadow, van gogh style" -o vangogh.png
```

**Inpainting.** `--mask FILE` marks what to repaint in white (Pillow's luma, binarized at 0.5;
any size, scaled like the image). An inpainting UNet sees the mask and the latents of the image
with the masked part zeroed at every step (diffusers' `StableDiffusionInpaintPipeline`); with a
text-to-image UNet the unmasked area is reset after each step to the input's latents noised to the
next level (diffusers' legacy approach: works, but the seams are weaker). `--strength` defaults to
1 (start from noise in the masked area); below 1 it starts from the noised image as in img2img.

```sh
./target/release/oxinfer --model stable-diffusion-v1-5/stable-diffusion-inpainting \
    --init-image cabin.png --mask cabin_mask.png -p "a red barn" -o barn.png
```

**Sizes.** Without `--size`, an edit runs at the model's area (512x512 for SD 1.5, 768x768 for
SD 2.x) with the input's aspect ratio, each side a multiple of 64 (a 768x512 photo runs at
640x448); with `--size WxH`, at that size. The image and the mask are scaled to cover it (Pillow
bicubic) and centre-cropped; the output has the working size. `--image FILE` is accepted in place
of `--init-image` (the flag the language models use); repo ids of diffusers pipelines download
like other models (`model_index.json`, configs, tokenizer and the fp16 safetensors of each
component; the safety checker and the single-file checkpoints are skipped).

**Library.** `sd::Pipeline::generate_from(&params, &EditInput, progress)` covers all three;
`img2img`, `inpaint` and `edit` (InstructPix2Pix) are shorthands, `edit_kind` says what an input
does with a given model, `edit_size` which size it runs at (`params.width` / `height` 0 pick it),
and `cancel_flag` stops a running generation from another thread. Images are
`video::RgbImage`; with the `oxideav` feature, `EditInput::from_picture` /
`with_mask_picture` take oxideav `Picture`s (any pixel format `Picture::to_rgb` converts), and the
result converts back with `Image::to_video_frame`. `examples/sd_edit.rs`:

```rust
use oxinfer::sd::{EditInput, Params, Pipeline, sched::Sampler};
oxinfer::gpu::init(&oxinfer::platform::detect(), oxinfer::platform::DevicePref::Auto);
let pipe = Pipeline::load(&oxinfer::hf::resolve("timbrooks/instruct-pix2pix", None)?)?;
let image = oxinfer::media::Media::new().load_image("cabin.png".as_ref())?;      // a video::RgbImage
let edit = EditInput::new(image).with_image_guidance(1.5);    // or .with_mask(m), .with_strength(s)
let params = Params {
    prompt: "make it snowy".into(), negative: String::new(), steps: 20, guidance: 7.5, seed: 1,
    width: 0, height: 0, sampler: Sampler::EulerAncestral,           // 0 x 0: Pipeline::edit_size
};
let (img, timings) = pipe.generate_from(&params, &edit, &mut |i, n| eprint!("\r{i}/{n}"))?;
```

**C API.** `oxi_sd_load(model, "device=auto;lora=FILE:SCALE", progress, user)`, then
`oxi_sd_generate(s, prompt, options_json, on_image, on_step, user)` with `options_json` such as
`{"init_image": "in.png", "mask": "mask.png", "strength": 0.8, "image_guidance": 1.5, "steps": 4,
"guidance": 1, "seed": 1}` (text to image without `init_image`), or `oxi_sd_edit_rgb(s, prompt,
rgb, width, height, mask_rgb_or_NULL, options_json, ...)` for pixels already in memory;
`oxi_sd_cancel` and returning non-zero from `on_step` stop it (see `include/oxinfer.h`).

**Browser.** Image mode has an InstructPix2Pix preset (with the LCM-LoRA, 4 steps) and, for every
model, an image and a mask upload with the strength or image-guidance setting
(`web_sd_set_input`, `sdSetInput(image, mask, { strength, imageGuidance })` in `oxinfer.js`).

**Agreement.** `tools/ref_sd.py --mode img2img | inpaint | pix2pix` runs diffusers' own
pipelines (Euler scheduler) and records their random draws by role, the initial noise and each
VAE sample, which oxinfer then reads (`OXINFER_SD_DUMP`). Weights rounded to bf16 on both sides,
Metal with `OXINFER_METAL_MMA=0` (the default matrix-unit path rounds activations to half:
about 3e-4 relative, 62-67 dB on the same runs), 512x512, generated test images:

| Edit | Encoder mean | First UNet output | Final latents | Image (rel L2, 8-bit PSNR) |
|---|---|---|---|---|
| InstructPix2Pix, 6 steps, guidance 7.5, image guidance 1.5 | 9.1e-7 | 7.5e-6 | 6.1e-6 | 4.2e-6, 85.8 dB |
| InstructPix2Pix + LCM-LoRA, 4 steps, guidance 1 | 2.1e-6 | 8.3e-7 | 3.1e-6 | 5.8e-6, 86.6 dB |
| SD 1.5 img2img, strength 0.6, 10 steps, guidance 7.5 | 9.1e-7 | 8.0e-6 | 5.7e-6 | 1.5e-5, 79.0 dB |
| SD-Turbo img2img, strength 0.5, 2 steps | 9.1e-7 | 1.7e-6 | 2.8e-6 | 1.7e-5, 77.9 dB |
| SD 1.5 inpainting UNet, strength 1, 10 steps, guidance 7.5 | 1.2e-5 (masked image) | 8.1e-6 | 2.3e-5 | 2.2e-5, 80.0 dB |
| SD 1.5 inpainting UNet, strength 0.75 | 9.1e-7 | 9.9e-6 | 1.3e-5 | 8.8e-6, 81.3 dB |
| SD 1.5 (text-to-image UNet) mask blending, 10 steps | 9.1e-7 | 7.9e-6 | 9.1e-6 | 5.7e-6, 86.1 dB |
| MagicBrush (fp32 `.bin`, "leading" timesteps), 6 steps, guidance 7.5 | 8.1e-7 | 7.5e-6 | 1.0e-5 | 1.1e-5, 84.3 dB |

The encoder's stride-2 convolutions pad one zero row and column after the input only
(`Downsample2D(padding=0)`); `ConvGeom::pad` carries that to the GPU kernels, so the encoder's
convolutions, attention and group norms take the same Metal / CUDA / WebGPU paths as the decoder
(`OXINFER_GPU_CHECK=1`: worst 1.5e-6 relative for the fused norms with `OXINFER_METAL_MMA=0`).

**Timings**, 512x512 on the M5 Max (encode = VAE encoder; the rest as for text to image):

| | Encode | UNet pass | Decode | Whole edit |
|---|---|---|---|---|
| Metal: InstructPix2Pix + LCM-LoRA, 4 steps | 0.43 s | 0.35 s | 0.65 s | 2.5 s |
| Metal: InstructPix2Pix, 20 steps x 3 passes | 0.42 s | 0.33 s | 0.64 s | 21 s |
| Metal: SD-Turbo img2img, strength 0.5 (1 pass) | 0.43 s | 0.26 s | 0.59 s | 1.4 s |
| Metal: SD 1.5 img2img, strength 0.6 (15 steps x 2) | 0.43 s | 0.33 s | 0.60 s | 11 s |
| Metal: inpainting UNet, 25 steps x 2 | 0.47 s | 0.36 s | 0.72 s | 19 s |
| CPU only (18 threads; measured with another 10-core job running, so upper bounds) | 4.2-5.9 s | 3.1-5.5 s | 7.5-9.9 s | LCM edit 38 s, turbo img2img 16 s |
| Browser (Chrome, WebGPU; machine busy): InstructPix2Pix + LCM-LoRA | 3.3 s | 3.0 s | 4.8 s | 20.5 s |

On an idle machine the CPU-only UNet pass is about 0.9 s and the decode 2.9-3.4 s (see Image
generation); the encoder costs about half the decoder. The browser numbers were taken with the
machine loaded too (SD-Turbo text to image took 7.7 s there against 3.2 s quiet), and the
browser output matched the native one.

On a phone the same code runs (iOS and Android cross builds are checked; not measured on a device
for editing): iOS takes the Metal backend like macOS, Android the CPU path. At a few seconds per
CPU UNet pass, the 4-step LCM InstructPix2Pix edit is the realistic choice there; the model needs
about 2 GB of bf16 weights in memory.

**Limits.** SD 1.x / 2.x and SDXL families only (no SD 3 or Flux editing here; no SDXL
InstructPix2Pix); one image per call;
the output has the working size (multiples of 64), not the input's; no paste-back of the
unmasked pixels after inpainting (as diffusers by default: the VAE round trip slightly changes
them); the mask is binary (no feathering); InstructPix2Pix with `--strength` is not a thing
(diffusers has none: it always starts from noise); image guidance needs text guidance above 1, so
the 4-step LCM-LoRA path has no image-guidance knob. Bigger instruction editors
(Qwen-Image-Edit, FLUX.1 Kontext, Step1X-Edit, OmniGen2, BAGEL) need their own architectures;
FLUX.1 Kontext, Qwen-Image-Edit, BAGEL and Step1X-Edit have one, [below](#flux1-kontext-image-editing-from-instructions-and-flux1-text-to-image),
[after it](#qwen-image-edit-image-editing-from-instructions-and-qwen-image-text-to-image),
[further on](#bagel-image-editing-text-to-image-and-image-understanding-in-one-model) and
[after BAGEL](#step1x-edit-image-editing-from-instructions).

## FLUX.1 Kontext: image editing from instructions (and FLUX.1 text to image)

FLUX.1 Kontext [dev] edits an image the way an instruction says: "make it night", "put a red
knitted hat on the dog", "change the text on the sign to ...". The same transformer family does
text to image as FLUX.1 [schnell] and [dev]. `src/flux/` follows diffusers'
`FluxKontextPipeline` and `FluxPipeline`. A diffusers FLUX folder, or a FLUX GGUF transformer,
switches oxinfer to it:

```sh
# editing: --init-image is the picture, --prompt the instruction (34 GB download, bf16)
./target/release/oxinfer --model unsloth/FLUX.1-Kontext-dev --init-image photo.png \
    -p "Make it night, with the porch lamps lit" --guidance 2.5 -o night.png
# the transformer as a GGUF quant (6.8 GB at Q4_0); T5, CLIP, VAE and tokenizers then come from
# unsloth/FLUX.1-Kontext-dev (--base), without its 24 GB transformer
./target/release/oxinfer --model QuantStack/FLUX.1-Kontext-dev-GGUF:Q4_0 --text-quant q8 \
    --init-image sign.png -p 'Change the text on the sign to "OXINFER CAFE"' -o cafe.png
# quantize the bf16 checkpoint at load instead (q8 / q4), T5 included
./target/release/oxinfer --model unsloth/FLUX.1-Kontext-dev --quant q4 --init-image in.png -p "..." -o out.png
# text to image with FLUX.1 [schnell] (4 steps, no guidance)
./target/release/oxinfer --model city96/FLUX.1-schnell-gguf:Q4_0 \
    -p "a photograph of a golden retriever on a farmhouse porch" --size 1024x768 -o dog.png
```

| Flag | Meaning |
|---|---|
| `--init-image FILE` | the image to edit (PNG, JPEG, WebP, ...); without it, text to image |
| `-p TEXT` | the instruction (editing) or the prompt |
| `--steps N`, `--guidance F`, `--seed N` | 28 steps (4 for schnell); distilled guidance 3.5 as diffusers (BFL suggests 2.5 for edits); seed 42 |
| `--size WxH` | output size, rounded to multiples of 16 (sets `--max-area` to W x H unless given; default: the input's aspect ratio at 1024^2 pixels; diffusers would rescale any size to 1024^2) |
| `--max-area N` | editing: output pixels, default 1048576 (1024^2), rounded to multiples of 16 |
| `--no-auto-resize` | editing: keep the input at its own size (rounded down to 16) instead of the closest of Kontext's 17 preferred ~1 MP resolutions |
| `--quant bf16\|q8\|q4` | transformer precision when stored as float; `--text-quant` sets T5's (default: `--quant`) |
| `--transformer SPEC` | a transformer replacing the folder's: GGUF file or `org/name:QUANT` |
| `--text-encoder SPEC` | a T5 encoder GGUF (`city96/t5-v1_1-xxl-encoder-gguf:Q8_0`) replacing `text_encoder_2/` |
| `--base SPEC` | with a GGUF `--model`: where T5, CLIP, the VAE and the tokenizers come from (default `unsloth/FLUX.1-Kontext-dev`) |

**What runs.** The prompt goes through CLIP-L (the pooled embedding, `sd::clip`) and the T5 v1.1
XXL encoder (512 tokens padded with `<pad>`, attended like any token as diffusers does; 256 for
schnell). For an edit, the input is resized with Pillow-exact Lanczos (diffusers'
`VaeImageProcessor`) to the preferred resolution closest to its aspect ratio, encoded by the VAE
(posterior mean, `(z - 0.1159) * 0.3611`), packed in 2x2 patches and appended to the noise tokens
with RoPE index 1 on the first axis (the noise is at index 0, text tokens at 0, 0, 0). The output
size is the requested one rescaled to `max_area` (Python's round-half-even, then multiples of 16).
Sigmas are `linspace(1, 1/steps)` shifted by `mu`, linear in the number of noise tokens between
(256, 0.5) and (4096, 1.15) (`FlowMatchEulerDiscreteScheduler`, dynamic shifting); FLUX.1 [schnell]
uses no shift. Each step runs the transformer on `[text; noise; reference]` with the guidance scale
as an input (guidance distillation: one pass per step, no unconditional pass; `true_cfg_scale` with
a negative prompt adds one, as in diffusers), drops the reference tokens from its output, and takes
an Euler step `x += (sigma_next - sigma) * v`. The latents are unpacked and decoded by the VAE.

**Weights.** A diffusers folder or repo (`model_index.json`) loads every component; repo ids fetch
only what the pipeline reads (component folders, not the top-level single-file checkpoints), so
`unsloth/FLUX.1-Kontext-dev` is about 34 GB. The transformer loads from `transformer/` (bf16, borrowed
from the mapped shards; `--quant q8|q4` quantizes at load) or from a GGUF file in the original
black-forest-labs layout (city96 / QuantStack quants, any GGUF block format, zero-copy; fused
`qkv` / `linear1` rows are split into separate layers without copying). The modulation vectors are
computed with float activations even for quantized weights (a single row would otherwise take the
int8 decode kernels, a 1e-3 error steering every block).

| Repository | What | Size | License |
|---|---|---|---|
| unsloth/FLUX.1-Kontext-dev (ungated copy of the gated black-forest-labs/FLUX.1-Kontext-dev) | the whole pipeline, diffusers layout | 23.8 GB transformer + 9.5 GB T5 + 0.4 GB CLIP / VAE (bf16) | FLUX.1 [dev] Non-Commercial License |
| QuantStack/FLUX.1-Kontext-dev-GGUF | Kontext transformer as GGUF, Q2_K to Q8_0 | 4.0 GB (Q2_K), 6.8 GB (Q4_0), 12.7 GB (Q8_0) | FLUX.1 [dev] Non-Commercial License |
| city96/FLUX.1-schnell-gguf | FLUX.1 [schnell] transformer (text to image) | 6.8 GB (Q4_0) - 23.8 GB (F16) | Apache 2.0 |
| city96/FLUX.1-dev-gguf | FLUX.1 [dev] transformer (text to image) | same | FLUX.1 [dev] Non-Commercial License |
| city96/t5-v1_1-xxl-encoder-gguf | the T5 encoder as GGUF (`t5encoder`) | 2.9 GB (Q4_K_M), 5.1 GB (Q8_0) | Apache 2.0 |

The FLUX.1 [dev] license (which covers Kontext [dev]) allows non-commercial use of the weights;
it claims no rights over outputs, which may be used commercially except to train a competing
model, and asks for content filtering or review of outputs and AI-generation disclosure where the
law requires it. Read `LICENSE.md` in the repository before use. FLUX.1 [schnell] and the T5
encoder are Apache 2.0.

In the library (`Pipeline::load` takes the same specs as `--model`; `LoadOptions` has `quant`,
`text_quant`, `transformer`, `text_encoder`, `base`, `device`, `cache_dir`, `vae_encoder`, `loras`, `vae`, `redux`;
`Options` are diffusers' call arguments, see `examples/flux_edit.rs`):

```rust
use oxinfer::flux::{LoadOptions, Options, Pipeline};
let pipe = Pipeline::load("QuantStack/FLUX.1-Kontext-dev-GGUF:Q4_0", &LoadOptions::default())?;
let photo = oxinfer::media::Media::new().load_image("photo.png".as_ref())?; // any video::RgbImage
let opts = Options { guidance: Some(2.5), width: Some(photo.width), height: Some(photo.height), ..Default::default() };
let (img, timings) = pipe.edit(&photo, "Make it night", &opts, &mut |step, steps| { eprint!("\r{step}/{steps}"); true })?;
std::fs::write("night.png", oxinfer::png::encode_rgb(img.width, img.height, &img.rgb))?;
let (img, _) = pipe.generate("a red fox in the snow", &Options::default(), &mut |_, _| true)?; // text to image
```

`edit_picture` takes an oxideav `Picture` and `Image::to_picture` returns one (feature
`oxideav`); `cancel_handle()` stops a run from another thread. The C API has `oxi_flux_load`
(`key=value` options), `oxi_flux_edit_rgb`, `oxi_flux_edit_file`, `oxi_flux_generate` (JSON
options, the image through the `oxi_image_fn` callback, steps through `oxi_step_fn`, which can
cancel), `oxi_flux_cancel` and `oxi_flux_free` (`include/oxinfer.h`).

**Reusable parts** (Step1X-Edit, OmniGen2 and BAGEL use the same VAE; Qwen-Image and others the
same scheduler):

| Module | What | API |
|---|---|---|
| `flux::vae` | the 16-channel FLUX `AutoencoderKL`: encoder (new; diffusers' asymmetric-padding downsampler) and `sd::vae::VaeDecoder`, normalization, 2x2 packing | `Vae::load(dir, encoder)`, `Vae::load_bfl(ae.safetensors, encoder)` (BFL layout), `encode(rgb, h, w) -> (latents, lh, lw)`, `decode(latents, lh, lw)`, `pack` / `unpack`, `rgb8_to_unit` / `unit_to_rgb8`, `chw_to_hwc` |
| `flux::sched` | `FlowMatchEulerDiscreteScheduler`: shift, dynamic shifting (exponential / linear), shift_terminal, inverted sigmas | `FlowMatchEuler::load(json)`, `calculate_shift`, `plan(base_sigmas, mu)`, `flux_plan(steps, seq)`, `step`, `scale_noise` |
| `flux::rope` | rotary embeddings over n-axis token ids (FLUX's 16/56/56 split, any split) | `Rope::new(ids, axes, theta)`, `Rope::flux_ids`, `apply(x, heads)` |
| `flux::transformer` | `DoubleBlock` / `SingleBlock` (load from diffusers or BFL names, forward), `FluxTransformer`, `timestep_proj`, `ln_mod` | `FluxTransformer::load(path, quant)`, `forward(img, n_img, txt, n_txt, pooled, t, guidance, rope)` |
| `flux::t5` | T5 v1.1 encoder (relative position buckets, gated GELU), safetensors or GGUF | `T5Encoder::load(dir, tokenizer_dir, quant)`, `encode(text, len)` |
| `flux::weights` | linear layers from safetensors / GGUF at bf16, q8, q4 or any GGUF type, row splitting | `Loader { files, quant }`, `dense`, `split`, `rows`; `Dense::forward`, `forward_vec` |

**Agreement with diffusers** (`tools/ref_flux.py` runs diffusers 0.42 / transformers 5.22 stage by
stage on the same inputs and noise, `OXINFER_FLUX_DUMP=dir` writes oxinfer's stages; float32 on
both sides, oxinfer with `OXINFER_METAL_MMA=0` unless noted):

| Stage | rel L2 | |
|---|---|---|
| T5-XXL, 512 tokens (bf16 weights as stored) | 2.0e-5 | |
| CLIP-L pooled embedding | 1.7e-6 | |
| VAE encode (512x512 reference image) | 4.8e-5 | weights fp16 -> bf16 |
| VAE decode | 5.6e-7 | 97.5 dB |
| transformer, 2 double + 2 single blocks of the bf16 checkpoint, 2560 tokens | 2.9e-6 | `OXINFER_FLUX_LAYERS=2,2` vs the truncated diffusers model |
| transformer, 1 + 1 blocks of a Q4_0 GGUF, weights dequantized by gguf-py | 2.8e-6 | |
| whole transformer (19 + 38 blocks), Q4_0 GGUF, one step | 5.6e-6 | vs diffusers' GGUF loader, 2560 tokens |
| Kontext edit, 512x512, 8 steps, Q4_0 GGUF, CPU | latents 3.1e-3 | 55.1 dB vs diffusers f32 |
| FLUX.1 [schnell] Q4_0, 512x512, 4 steps | latents 9.4e-4 | 61.6 dB |
| same, Metal matrix units (half-precision operands, the default) | latents 5.2e-2 | 37.8 dB (half-precision attention too); the same image to the eye |
| Kontext edit, bf16, 28 steps, Metal default vs diffusers bf16 on MPS | | 43.4 dB |

T5 at `--text-quant q8` is 1.9e-2 from bf16; city96's Q8_0 T5 GGUF is 11% off, almost all of it
from its quantized token-embedding table (rows reach 231 in magnitude; the same weights in
transformers give the same 11%), so the T5 GGUF files are for memory, not fidelity.

**Speed and memory** on the M5 Max (18 cores, 128 GB, shared with other jobs: load average 12-27
during these runs, so take the numbers as upper bounds). Per denoising step, steady state (the
first step also uploads the weights); an edit at 1024x1024 has 8704 tokens per pass (512 text +
4096 noise + 4096 reference), 512x512 with `--no-auto-resize --max-area 262144` 2560:

| Transformer | Metal, 1024^2 | Metal, 512^2 | CPU, 1024^2 | CPU, 512^2 | peak footprint (1024^2, Metal) | max RSS |
|---|---|---|---|---|---|---|
| bf16 (mapped) | 9.4 s | 1.9 s | | 41 s | 12.7 GB | 15.6 GB |
| `--quant q8` | 12.4 s | | | | 31.5 GB | 52.7 GB |
| `--quant q4` | 10.7 s | | | | 22.3 GB | 50.5 GB |
| Q8_0 GGUF (`--text-quant q8`) | 10.9 s | | | | 18.2 GB | 24.4 GB |
| Q4_0 GGUF (`--text-quant q4`) | 13.8 s | 1.6 s | 208 s | 43 s | 15.6 GB | 23.1 GB |

A whole 28-step 1024^2 edit takes 4.5-6.5 minutes on Metal (text encoding 1-2 s, VAE encode
2 s, decode 3-4 s) and about 1.6 hours on the CPU alone; at 512^2 it is about a minute on Metal.
The peak footprint counts the process's own memory (quantized copies, activations, Metal
buffers), max RSS adds the mapped weight pages, which the OS can drop and re-read: bf16 and GGUF
weights are used in place from the mapped files, so a machine with less RAM than the weights
streams them from storage (slowly) instead of failing; `--quant q8|q4` makes in-memory copies
and reads the bf16 files once at load (mapped, so the RSS shows them; the footprint is what
stays). For small machines the GGUF files are the way: a Q4_0 transformer with a Q4_K_M T5
(`--text-encoder city96/t5-v1_1-xxl-encoder-gguf:Q4_K_M`) is 9.7 GB on disk, all of it mapped.
On Metal, every matmul runs on the GPU (M5's matrix units: half-precision operands) and
attention uses a fused kernel on the matrix units (`flash_attn128_mma`: 7.2 -> 3.6 s per 1024^2
step against the f32 kernel); norms, RoPE and the gates stay on the CPU. `OXINFER_METAL_MMA=0`
gives the f32 kernels (the exact rows of the agreement table, 2-2.5x slower).

**Sample edits** (FLUX.1 Kontext [dev] Q8_0, guidance 2.5, 28 steps, 1168x880 from 1024x768 inputs
that FLUX.1 [schnell] generated): "Put a red knitted hat on the dog" keeps the dog, porch and
lighting and adds a knitted beanie with a pom-pom; "Make it night time, with the porch lamps lit"
darkens the scene with both wall lamps glowing; 'Change the text on the sign from "FRESH BREAD" to
"OXINFER CAFE"' rewrites the carved shop sign in the same typeface and wood, leaving the rest of the
storefront unchanged; "Turn this photo into a watercolor painting" restyles the whole picture.

**More FLUX.1 modes.** `Pipeline::render(prompt, &Inputs, &Options, progress)` runs all of them
(`Inputs { references, init, mask, control, image_prompts }`; `generate` and `edit` are shortcuts),
and the CLI picks the mode from the flags and the transformer's input width:

```sh
# several reference images for Kontext ("the woman from the first image wearing the hat from the second")
./target/release/oxinfer --model QuantStack/FLUX.1-Kontext-dev-GGUF:Q4_0 --init-image woman.png \
    --init-image hat.png -p "The woman wears the hat" -o out.png
# a LoRA (here a few-step one: 8 steps instead of 28) on a GGUF transformer
./target/release/oxinfer --model city96/FLUX.1-dev-gguf:Q4_0 --steps 8 \
    --lora alimama-creative/FLUX.1-Turbo-Alpha/diffusion_pytorch_model.safetensors -p "a red fox in the snow" -o fox.png
# image to image (FluxImg2ImgPipeline) and inpainting by mask blending (FluxInpaintPipeline)
./target/release/oxinfer --model city96/FLUX.1-dev-gguf:Q4_0 --init-image photo.png --strength 0.6 -p "an oil painting" -o oil.png
./target/release/oxinfer --model city96/FLUX.1-dev-gguf:Q4_0 --init-image photo.png --mask mask.png --strength 0.85 -p "a red sofa" -o sofa.png
# FLUX.1 Fill (a 384-channel transformer: FluxFillPipeline; guidance 30 is BFL's setting)
./target/release/oxinfer --model flux1-fill-dev-Q4_0.gguf --init-image photo.png --mask mask.png --guidance 30 -p "a red sofa" -o sofa.png
# FLUX.1 Canny / Depth: BFL's control LoRA on FLUX.1 [dev] (or a 128-channel Canny / Depth transformer)
./target/release/oxinfer --model city96/FLUX.1-dev-gguf:Q4_0 --lora flux1-canny-dev-lora.safetensors \
    --control-image edges.png --guidance 30 -p "a glass house in a forest" -o house.png
# FLUX.1 Redux: an image as the prompt (variations; -p adds text)
./target/release/oxinfer --model city96/FLUX.1-dev-gguf:Q4_0 --redux Runware/FLUX.1-Redux-dev --image-prompt photo.png -p "" -o variation.png
# BFL single-file checkpoints: the transformer and the autoencoder
./target/release/oxinfer --model flux1-dev.safetensors --vae ae.safetensors -p "a red fox in the snow" -o fox.png
```

| Flag | Meaning |
|---|---|
| `--init-image FILE` (repeated) | Kontext: several references. Each is VAE-encoded on its own (resized to its own preferred resolution) and appended as its own token sequence |
| `--ref-layout offset\|index\|diffusers` | where the references' tokens sit (RoPE ids `[index, row, column]`; one reference is `[1, y, x]` in all three). `offset` (default; ComfyUI's default for FLUX Kontext): index 1 for all, each reference beside or below the previous ones, whichever keeps the canvas more square. `index` (ComfyUI's `index` method): reference i at index i + 1, all at the origin. `diffusers`: the ids of diffusers' community `pipeline_flux_kontext_multiple_images.py` (every reference resized to the first one's size, index 1, reference i shifted by i x the grid height along the columns). BFL's own code and diffusers' `FluxKontextPipeline` take one image only |
| `--strength F` | `--init-image` becomes the start of an image-to-image run instead of a Kontext reference: the image is noised to step `int(steps * (1 - F))` of the schedule and denoised from there (default 0.6 as diffusers; 1 for Fill) |
| `--mask FILE` | inpaint `--init-image`: white is repainted (resized like the image, luma thresholded at 50%). FLUX.1 Fill reads the masked image's latents and the mask as 320 more input channels per token; any other FLUX.1 blends in latent space as diffusers' `FluxInpaintPipeline` (after each step the kept region is reset to the image at that step's noise level, mask sampled per latent pixel) |
| `--control-image FILE` | the edge / depth map (as an image; computing Canny edges or depth is up to you) for FLUX.1 Canny / Depth: its latents ride along every noise token (128 input channels) |
| `--image-prompt FILE`, `--redux SPEC`, `--image-prompt-scale F` | FLUX.1 Redux: the image through SigLIP so400m/14 at 384 px and the Redux projector into 729 tokens appended to the T5 tokens (repeatable); the scale multiplies them |
| `--lora FILE[:SCALE]` | LoRA files on the transformer (repeatable; a local file or `org/name/file.safetensors`) |
| `--vae SPEC` | the autoencoder from BFL's `ae.safetensors` (or a diffusers `vae` folder) |

**LoRAs** are applied at load as a low-rank side path on each targeted layer
(`y = W x + scale * alpha / rank * up (down x)`, f32, after the base matmul wherever that ran): the
same function as the merged weights `W + scale * alpha / rank * up @ down`, with `W` left as it is
stored (mapped bf16, any GGUF block format, q8 / q4). Rewriting a quantized matrix would mean
dequantizing and requantizing it, and the update of a few-step LoRA is smaller than a Q4 block's
rounding step, so the side path is both exact and free of a 24 GB copy; it costs two small matmuls
per layer and step. Key namings read: diffusers / PEFT (`transformer.<module>.lora_A.weight`,
alpha from the `lora_adapter_metadata` header), kohya / ai-toolkit (`lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight`
with `.alpha`, `lora_transformer_*`), BFL / ComfyUI (`[diffusion_model.]double_blocks.0.img_attn.qkv.lora_A.weight`,
with `lora_B.bias` and the QK-norm scales of BFL's control LoRAs, whose `img_in` update widens the
input to 128 channels) and XLabs (`double_blocks.0.processor.qkv_lora1.down.weight`). Modules on
BFL's fused matrices (`qkv`, `linear1`, the final modulation) are split by rows over the layers
oxinfer keeps. A key that is not a low-rank pair (DoRA, LoKr, LoHa) or a module that matches no
layer is an error; text-encoder modules (`lora_te1_*`, `text_encoder.*`) are not applied and the
load line says how many there were (`[lora FILE x1: 504 modules applied, rank 64]`). Few-step
LoRAs keep the model's scheduler; give `--steps` yourself.

**Single files.** `--model` (or `--transformer`) takes a BFL-layout safetensors file
(`flux1-dev.safetensors`, `flux1-kontext-dev.safetensors`, also with ComfyUI's
`model.diffusion_model.` prefix; bf16 / fp16 / f32, not fp8) the way it takes a GGUF: names are
remapped to the same layers (fused `qkv` / `linear1` split by rows, zero-copy), the configuration
comes from the tensor shapes, the other parts from `--base`. `--vae ae.safetensors` loads the
autoencoder from BFL's layout.

Measured against diffusers 0.42 with `tools/ref_flux_ext.py` (it runs both sides on the same
noise). On a tiny random-weight pipeline from `tools/flux_tiny.py` (2 + 2 blocks, the real
tokenizers; float32 on both sides, CPU), end to end:

| Check (`--mode all`) | final latents, rel L2 | image |
|---|---|---|
| image to image (strength 0.6 of 6 steps; 0.35 of 10 with a resize) vs `FluxImg2ImgPipeline` | 2.3e-6, 2.8e-6 | 86.9, 84.6 dB |
| inpainting (strength 0.8; 1.0) vs `FluxInpaintPipeline` | 2.2e-6, 1.9e-6 | 86.9, 84.3 dB |
| Fill (384 channels; strength 1 and 0.7) vs `FluxFillPipeline` | 2.7e-6, 2.7e-6 | 85.3, 85.6 dB |
| control (128 channels) vs `FluxControlPipeline` | 1.7e-6 | 88.9 dB |
| 3 Kontext references, `offset` / `index` / `diffusers` ids vs a diffusers Kontext loop over ComfyUI's / the community pipeline's ids | 1.6e-6, 1.6e-6, 1.5e-6 | 87.0, 87.0, 86.5 dB |
| LoRA, PEFT keys, vs `load_lora_weights` | 3.3e-6 | 84.3 dB |
| LoRA, kohya keys (fused `qkv` / `linear1`, per-module alpha); XLabs keys; PEFT x0.7 + kohya x-0.5 | 2.3e-6, 2.2e-6, 2.3e-6 | 89.0, 85.2, 85.6 dB |
| BFL control LoRA (`lora_B.bias`, norm scales, 128-channel `img_in`) on a 64-channel model vs `FluxControlPipeline` | 3.6e-6 | 83.9 dB |
| BFL single-file transformer + `ae.safetensors` vs the diffusers folder (and with the kohya LoRA) | 1.9e-6 (2.3e-6) | 88.2 dB (89.0) |
| Redux (tiny SigLIP + projector) vs `FluxPriorReduxPipeline` | 1.9e-6, tokens 6.4e-7 | 84.6 dB |

diffusers' converters for kohya / XLabs / BFL LoRA keys and for single files assume FLUX.1's
3072-wide layers and cannot read a tiny model, so for those rows the reference merges
`W + scale * alpha / rank * up @ down` into the BFL-layout weights in Python and runs the diffusers
pipeline on the result; where diffusers samples the VAE posterior the reference takes its mode,
as oxinfer does. On real weights (FLUX.1 [dev] Q4_0 GGUF, one transformer call at 256x256 against
diffusers' GGUF loader with `load_lora_weights`, on oxinfer's own text embeddings; f32, CPU):

| Check (`--mode step`, `--mode redux-tokens`) | rel L2 |
|---|---|
| alimama-creative/FLUX.1-Turbo-Alpha (PEFT keys, 504 modules, rank 64); the LoRA itself moves the prediction by 0.20 | 2.7e-6 |
| XLabs-AI/flux-RealismLora (XLabs keys, 76 modules, rank 16); moves the prediction by 0.12 | 3.8e-6 |
| BFL's flux1-canny-dev-lora (control LoRA, 314 low-rank modules of rank 64-128 with biases + 152 norm scales), 256x192 with a control image, vs `FluxControlPipeline`'s loader with `--fix-norm-out` | 1.7e-6 |
| the same against diffusers 0.42 as it is | 3.9e-2 |
| Redux tokens (SigLIP so400m + projector of FLUX.1-Redux-dev, bf16 weights) vs transformers | 9.2e-6 |

The last row is diffusers: its converter for BFL control LoRAs swaps the (shift, scale) halves of
`final_layer.adaLN_modulation.1`'s `lora_A` rows as well as `lora_B`'s, but `lora_A`'s rows are
rank components, not output channels, so the product changes; `--fix-norm-out` swaps them back in
the reference (with only that change the two agree, also on 1 + 1 blocks in f32: 1.8e-6). oxinfer
applies the update as BFL's own code does.

Not run on real weights: FLUX.1 Fill and the 128-channel Canny / Depth transformers (plumbing
checked on the tiny models only), BFL's 24 GB single files (the same tensors as the GGUF files,
which load through the same BFL-layout path), whole few-step generations (no picture was
looked at: the LoRA rows above compare one transformer call), and multi-reference Kontext on the
real model (the ids are checked, the picture quality of each layout is not). Redux' SigLIP resize
is Pillow's bicubic (transformers' PIL processor; its torchvision processor antialiases
differently).

Not supported: IP-Adapter (XLabs, InstantX), ControlNets, DoRA / LoKr / LoHa LoRAs and LoRAs on
the text encoders, fp8 single-file checkpoints, masks or `--strength` together with Kontext
references (diffusers' `FluxKontextInpaintPipeline`), the C API and server for the modes above
other than LoRA / VAE loading and `strength` (use the library's `Pipeline::render`), and the
browser build (the weights exceed its 4 GB).

## Stable Diffusion 3 / 3.5: text to image and image to image

Stable Diffusion 3 medium and Stable Diffusion 3.5 (medium, large, large-turbo) are rectified-flow
models like FLUX.1, with their own transformer (MMDiT) and three text encoders. `src/sd3/` follows
diffusers' `StableDiffusion3Pipeline` and `StableDiffusion3Img2ImgPipeline`. A diffusers folder
whose `model_index.json` names a `StableDiffusion3*` class, or an SD 3 GGUF transformer, switches
oxinfer to it:

```sh
# text to image (28 steps, guidance 7, 1024x1024 by default)
./target/release/oxinfer --model DIR -p "a photograph of a red fox sitting in a snowy forest" \
    --negative "blurry, low quality" --size 1024x768 -o fox.png
# without T5 (9.5 GB less to load; prompts are then read by the two CLIP encoders only)
./target/release/oxinfer --model DIR --no-t5 -p "a watercolor lighthouse" -o lighthouse.png
# image to image: --strength of the schedule is redone from the input (default 0.6)
./target/release/oxinfer --model DIR --init-image photo.png --strength 0.6 -p "a watercolor painting" -o wc.png
# the transformer as a GGUF quant; the text encoders, VAE and tokenizers come from --base
./target/release/oxinfer --model city96/stable-diffusion-3.5-medium-gguf:Q4_0 --base DIR -p "..." -o out.png
```

`DIR` is a local folder or a repo id in the diffusers layout. Stability AI's own repositories
(`stabilityai/stable-diffusion-3-medium-diffusers`, `stabilityai/stable-diffusion-3.5-medium`,
`-3.5-large`, `-3.5-large-turbo`) are gated: set `HF_TOKEN` after accepting their license, or
point `--model` at a folder you already have.

| Flag | Meaning |
|---|---|
| `-p TEXT`, `--negative TEXT` | the prompt; the negative prompt of guided runs (default empty) |
| `--steps N`, `--guidance F`, `--seed N` | 28 steps (50 image to image) at classifier-free guidance 7, as diffusers; models with "turbo" in their name default to 4 steps without guidance; seed 42 |
| `--size WxH` | output size, rounded down to multiples of 16 (default 1024x1024; image to image: the input's size, where diffusers would resize it to 1024x1024) |
| `--init-image FILE`, `--strength F` | image to image: the starting image and how much of the schedule is run (0..1, default 0.6) |
| `--no-t5` | do not load T5-XXL: its tokens are zeros, as diffusers does with `text_encoder_3=None` (also when the folder has no `text_encoder_3` weights) |
| `--max-sequence-length N` | T5 tokens, default 256, at most 512 |
| `--quant bf16\|q8\|q4` | transformer precision when stored as float; `--text-quant` sets T5's (default: `--quant`) |
| `--transformer SPEC` | a transformer replacing the folder's: GGUF file or `org/name:QUANT` |
| `--text-encoder SPEC` | a T5 encoder GGUF (`city96/t5-v1_1-xxl-encoder-gguf:Q8_0`) replacing `text_encoder_3/` |
| `--base SPEC` | with a GGUF `--model`: the diffusers SD 3 folder the text encoders, VAE and tokenizers come from (required: there is no ungated default) |

**What runs.** The prompt goes through three encoders. CLIP-L and OpenCLIP-bigG
(`CLIPTextModelWithProjection`, `sd3::clip`) each give their penultimate hidden states (77
tokens, 768 and 1280 wide) and a projected pooled embedding; the states are concatenated to 2048
channels and zero-padded to 4096, then followed by the T5-XXL states (256 tokens, `flux::t5`), so
the transformer reads 333 text tokens of 4096; the two pooled embeddings (2048 together) join the
timestep embedding as the conditioning vector. The latent (16 channels, 1/8 of the image) is cut
into 2x2 patches by a convolution, and each patch gets a row of the checkpoint's 2-D sinusoidal
position table, cropped around its centre (192x192 patches stored in SD 3 medium and 3.5 large,
384x384 in 3.5 medium). Every joint block (`sd3::transformer`) normalizes the two streams with
scale / shift / gate vectors computed from the conditioning vector, runs one attention over
`[text; image]` with separate projections per stream and a feed-forward per stream; SD 3.5 adds RMS
norms on queries and keys, SD 3.5 medium (MMDiT-X) a second attention over the image tokens
alone in its first 13 blocks, and the last block does not update the text stream. Guidance is
classifier-free: with a scale above 1 each step runs a second pass on the negative prompt and
combines `uncond + scale * (cond - uncond)`. The schedule is `FlowMatchEulerDiscreteScheduler`
with the config's constant shift (3 for every SD 3 model) or, when the config sets
`use_dynamic_shifting`, the resolution-dependent `mu` from the number of image tokens; steps are
Euler, `x += (sigma_next - sigma) * v`. Image to image encodes the input with the VAE (a sample of
the posterior, `(z - 0.0609) * 1.5305`), mixes it with noise at the starting sigma
(`sigma * noise + (1 - sigma) * z`) and runs the last `steps * strength` steps. The 16-channel VAE
is the `AutoencoderKL` code FLUX uses (`flux::vae`), with SD 3's scale and shift from its config.

**Weights.** The transformer loads from `transformer/` (float weights run as stored from the
mapped file, `--quant q8|q4` quantizes at load) or from a GGUF file in Stability AI's original
layout (`joint_blocks.N.x_block.attn.qkv`; city96's SD 3 / SD 3.5 quants, any GGUF block format,
zero-copy, fused `qkv` rows split into separate layers), whose configuration (width, depth,
QK norm, dual-attention blocks, position table size) is read from the tensor shapes. The CLIP
encoders run their fp16 weights as stored.

In the library (`Pipeline::load` takes the same specs as `--model`; `LoadOptions` has `quant`,
`text_quant`, `transformer`, `text_encoder`, `base`, `t5`, `device`, `cache_dir`, `vae_encoder`;
see `examples/sd3_generate.rs`):

```rust
use oxinfer::sd3::{LoadOptions, Options, Pipeline};
let pipe = Pipeline::load("models/stable-diffusion-3.5-medium", &LoadOptions::default())?;
let opts = Options { negative_prompt: Some("blurry".into()), width: Some(1024), height: Some(768), ..Default::default() };
let (img, timings) = pipe.generate("a red fox in the snow", &opts, &mut |step, steps| { eprint!("\r{step}/{steps}"); true })?;
std::fs::write("fox.png", oxinfer::png::encode_rgb(img.width, img.height, &img.rgb))?;
let photo = oxinfer::media::Media::new().load_image("photo.png".as_ref())?; // any video::RgbImage
let (img, _) = pipe.img2img(&photo, "a watercolor painting", &Options { strength: 0.6, ..Default::default() }, &mut |_, _| true)?;
```

`encode_prompt` returns the conditioning alone, `sd3::sd3_plan` the schedule, and
`cancel_handle()` stops a run from another thread.

**Agreement with diffusers** (real weights: `v2ray/stable-diffusion-3-medium-diffusers` and
`adamo1139/stable-diffusion-3.5-medium-ungated`, ungated copies of Stability AI's SD 3 medium and
SD 3.5 medium repositories, and `city96/stable-diffusion-3.5-medium-gguf`. `tools/ref_sd3.py` runs
diffusers 0.42 / transformers 5.22 in float32 stage by stage on the same noise,
`OXINFER_SD3_DUMP=dir` writes oxinfer's stages; 512x512, 4 steps at guidance 7 with a negative
prompt, so 8 transformer passes of 1357 tokens, `--device cpu`):

| Stage | rel L2 | |
|---|---|---|
| CLIP-L penultimate states / pooled embedding | 1.6e-6 / 1.6e-6 | |
| OpenCLIP-bigG penultimate states / pooled embedding | 5.1e-6 / 1.9e-6 | |
| T5-XXL, 256 tokens | 1.3e-5 | |
| schedule (sigmas, shift 3) | 4.1e-8 | also unit-tested against the scheduler with dynamic shifting |
| SD 3 medium transformer, one pass (24 blocks) | 3.6e-6 | |
| SD 3.5 medium transformer, one pass (24 blocks, QK norm, 13 dual-attention blocks) | 1.3e-6 | |
| SD 3.5 medium transformer, Q4_0 GGUF, one pass | 5.9e-5 | vs diffusers' GGUF loader on the same file |
| VAE decode, SD 3.5 medium's VAE | 1.7e-6 | 89.4 dB |
| VAE decode, SD 3 medium's VAE | 4.9e-4 | 62.3 dB |
| SD 3 medium, 4 guided steps end to end | latents 4.8e-5 | 62.3 dB (the VAE decode above) |
| same with `--no-t5` | latents 6.0e-5 | 62.0 dB |
| SD 3.5 medium, 4 guided steps end to end | latents 2.2e-4 | 67.3 dB |
| SD 3.5 medium Q4_0 GGUF, 4 guided steps end to end | latents 1.3e-3 | 58.1 dB |
| SD 3 medium image to image, strength 0.6 of 5 steps (3 run) | latents 4.4e-3 | 54.2 dB; the posterior sample from the VAE encoder is 2.6e-3 from diffusers' |
| SD 3.5 medium on CUDA (RTX 5080), 4 guided steps end to end | latents 2.1e-4 | 67.4 dB; `OXINFER_GPU_CHECK=1`: attention 1.9e-6, VAE convolutions with fused norms 4.5e-7 against the CPU ops |
| SD 3.5 medium Q4_0 GGUF as `--model` with `--base`, on CUDA | latents 1.3e-3 | 58.1 dB |

With a GPU, the joint attention and the VAE's convolutions, norms and attention run on it through
the same backend operations as FLUX and Stable Diffusion 1.x / 2.x; the rest is on the CPU.

Not run: SD 3.5 large and large-turbo. They use the same code path as SD 3 medium with QK norms
(38 blocks of 38 heads, no dual attention), both pieces verified above on the medium models, but
their 16 GB transformers were not downloaded, so no agreement is claimed for them. The "turbo"
defaults (4 steps, no guidance) are untested for the same reason. `--quant q8|q4` on the
safetensors transformer and resolution-dependent shifting in a full run (no SD 3 config sets it)
were not measured either.

Not supported: ControlNet, IP-Adapter, LoRAs on SD 3, inpainting, skip-layer guidance, `clip_skip`,
prompts longer than 77 CLIP tokens (truncated, as in diffusers), separate prompts per encoder
(`prompt_2`, `prompt_3`), single-file checkpoints with the text encoders inside
(`sd3.5_medium.safetensors`; the diffusers folder and the GGUF files cover the same weights),
the C API and the browser build.

## Qwen-Image-Edit: image editing from instructions (and Qwen-Image text to image)

Qwen-Image-Edit-2511 (and -2509, and the original Qwen-Image-Edit) edits images the way an
instruction says, combining several input images; Qwen-Image / Qwen-Image-2512 generate from text
with the same transformer. `src/qwen_image/` follows diffusers' `QwenImageEditPlusPipeline`,
`QwenImageEditPipeline` and `QwenImagePipeline`; the pipeline kind comes from `model_index.json`.

```sh
# full quality: 40 steps, true CFG 4 with a " " negative prompt (Qwen's recommended settings)
./target/release/oxinfer --model Qwen/Qwen-Image-Edit-2511 --mmap --init-image fox.png \
    -p "Add a red knitted scarf around the fox's neck." -o scarf.png
# 4 steps with the Lightning LoRA (its steps, scheduler and guidance are picked from the file name)
./target/release/oxinfer --model Qwen/Qwen-Image-Edit-2511 --mmap --init-image fox.png \
    --lora lightx2v/Qwen-Image-Edit-2511-Lightning/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors \
    -p "Make it winter: snow on the ground, the bench and the branches." -o winter.png
# several images ("Picture 1", "Picture 2" in the instruction), a 12 GB GGUF transformer
./target/release/oxinfer --model Qwen/Qwen-Image-Edit-2511 --mmap \
    --transformer unsloth/Qwen-Image-Edit-2511-GGUF/qwen-image-edit-2511-Q4_0.gguf \
    --init-image fox.png --init-image shop.png \
    -p "The fox from Picture 1 sits on the sidewalk in front of the bookshop from Picture 2." -o both.png
# text to image (a Qwen-Image-2512 folder; fetching the repo id downloads its bf16 transformer too)
./target/release/oxinfer --model Qwen/Qwen-Image-2512 --mmap \
    --transformer unsloth/Qwen-Image-2512-GGUF/qwen-image-2512-Q4_0.gguf \
    --lora lightx2v/Qwen-Image-2512-Lightning/Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors \
    -p "a cozy bookshop on a rainy street at night, a sign that reads OXINFER BOOKS" -o shop.png
# image to image and inpainting (white = repaint) with a text-to-image or an edit model
./target/release/oxinfer --model Qwen/Qwen-Image-2512 --mmap --init-image shop.png --strength 0.6 \
    -p "the same street in heavy snow" -o snow.png
./target/release/oxinfer --model Qwen/Qwen-Image-2512 --mmap --init-image shop.png --mask door.png \
    --strength 1 -p "a red wooden door" -o door.png
# ControlNet (canny / soft edge / depth / pose image; one Union model for all of them)
./target/release/oxinfer --model Qwen/Qwen-Image-2512 --mmap --controlnet InstantX/Qwen-Image-ControlNet-Union \
    --control-image canny.png --control-scale 1.0 -p "a stone cottage in a pine forest" -o cottage.png
# an FP8 "scaled" single-file transformer (ComfyUI / lightx2v), used as stored
./target/release/oxinfer --model Qwen/Qwen-Image-2512 --mmap \
    --transformer lightx2v/Qwen-Image-2512-Lightning/qwen_image_2512_fp8_e4m3fn_scaled_comfyui_4steps_v1.0.safetensors \
    -p "a lighthouse at dusk" -o lighthouse.png
# Qwen-Image-Layered: an image into RGBA layers (layers_0.png ...; without -p the text encoder captions it)
./target/release/oxinfer --model Qwen/Qwen-Image-Layered --mmap --init-image poster.png --layers 4 -o layers.png
```

| Flag | Meaning |
|---|---|
| `--init-image FILE` | image to edit (repeatable: Edit-2509 / 2511 take several; the output size follows the last one) |
| `--strength F` | start from the input image noised to this part of the schedule instead of from noise: image to image with Qwen-Image / -2512 (default 0.6, the output takes the image's size), and with the edit models once given (or with `--mask`, default 1) |
| `--mask FILE` | inpainting: the white region is repainted, the rest returns to the input image after every step |
| `--controlnet DIR`, `--control-image FILE` | a ControlNet folder or repo id and its condition image (repeatable: the images' residuals add up); with `--mask` and the inpainting ControlNet, the image to inpaint |
| `--control-scale F`, `--control-start F`, `--control-end F` | conditioning scale (1; one per image when repeated) and the part of the steps the ControlNet acts on (0 and 1) |
| `--layers N`, `--resolution 640\|1024` | Qwen-Image-Layered: the number of RGBA layers (4) and the working resolution |
| `--steps N`, `--guidance F`, `--negative TEXT` | 40, true CFG 4, `" "` (with a Lightning LoRA: its 4 / 8 steps, CFG 1) |
| `--size WxH` | output size (default: the input's aspect ratio at about 1024x1024, multiples of 16) |
| `--lora FILE[:SCALE]` | LoRA files (lightx2v Lightning, kohya-style or PEFT keys) |
| `--transformer FILE` | the transformer from a GGUF file (unsloth / QuantStack conversions; BF16 outer layers kept) or a single safetensors file (FP8 scaled checkpoints, bf16) |
| `--quant q8\|q4` | transformer and text-encoder precision (the text encoder stays Q8 with `q4`) |
| `--mmap`, `--stream` | map the weights; stream the transformer's 60 blocks from storage (`--cache-budget-gb`) |

What runs: the instruction goes through the pipeline's chat template with Qwen2.5-VL-7B
(`crate::qwen_vl`, below): a system turn, `Picture N: <|vision_start|>...<|vision_end|>` per input
image (resized to ~384x384 with Pillow's Lanczos, then the Qwen2-VL processor), the instruction;
the last hidden states minus the template's first 64 tokens condition the transformer. The input
images are also resized to ~1024x1024 and VAE-encoded (the Wan-style causal 3D VAE, run on one
frame, latents normalized by its per-channel mean / std) into 2x2 patches that follow the noisy
target's as extra tokens. The 60-block MMDiT (20B) keeps text and image streams with their own
AdaLN modulation (reference-image tokens modulated at timestep 0, Edit-2511's `zero_cond_t`), joint
attention with per-head RMSNorm and the 3-axis MSRoPE (frame, row, column centred on each image,
text after the largest image). Flow matching with Euler steps and the resolution-dependent shift
(`crate::flux::sched`), true CFG with the norm-preserving rescale of the combined velocity. The
Lightning LoRA uses its distillation's fixed shift of 3.

Components load when first needed and the 16 GB text encoder is dropped after encoding (the
library keeps it with `keep_text_encoder`). A LoRA is merged into the weights when they are
quantized at load anyway (`--quant`; merged in f32 and quantized from that, since a Lightning
update is below half a bf16 ulp of most weights and rounding the merged weights to bf16 would erase
it), and applied at run time otherwise (bf16 weights borrowed from the mapped checkpoint, GGUF
blocks), as two small extra matmuls, so no 40 GB copy is made. With `--quant`, the layers outside
the blocks stay bf16 (as GGUF conversions keep them), and the modulation layers keep the ~17 input
channels of `silu(timestep embedding)` that carry 99.9% of its energy exact (f32 columns beside the
quantized matrix). On Metal
the text encoder runs with exact (f32) matmuls (its hidden states have outlier channels near 200,
which the half-precision matrix units round by 1%), and the transformer's attention keeps f32
operands (its logits run into the thousands; half-precision queries and keys move the first
velocity ~10%); the transformer's matmuls and its attention both run on the matrix units.

**Agreement** with diffusers 0.42 / transformers 5.20 (`tools/ref_qwen_image.py`: same input image,
prompt and initial noise; bf16 weights on both sides with every reference matmul in f32;
Edit-2511, a 512x512 fox photo, "make it winter", 512x512 output with 4096 reference tokens):

| oxinfer | prompt embeddings | first velocity | final latents | image PSNR |
|---|---|---|---|---|
| bf16, CPU, 2 steps, true CFG 4 | 4.1e-4 | 1.5e-4 | 1.7e-4 | 68.5 dB |
| bf16, Metal (matrix units), 2 steps, CFG 4 | 1.1e-2 (before the exact text encoder) | 2.8e-3 | 3.1e-3 | 55.4 dB |
| bf16 + Lightning LoRA, CPU, 4 steps | 4.1e-4 | 1.8e-3 | 7.6e-3 | 45.7 dB |
| bf16 + Lightning, Metal with `OXINFER_METAL_MMA=0` | 4.4e-4 | 1.7e-3 | 7.9e-3 | 45.7 dB |
| bf16 + Lightning, Metal (default) | 4.4e-4 | 7.3e-3 | 8.3e-2 | 28.9 dB |
| `--quant q8` + Lightning, Metal | 1.7e-2 | 4.7e-2 (was 0.16) | 0.11 (was 0.35) | 26.8 dB (was 17.8) |
| `--quant q4` + Lightning, Metal | 1.7e-2 | 0.14 (was 0.20) | 0.39 (was 0.57) | 15.6 dB (was 12.2) |
| GGUF Q4_0 (unsloth) + Lightning, Metal | 4.4e-4 | 0.14 | 0.22 | 21.3 dB |
| reference itself with oxinfer's Q8 weights (dequantized, f32 matmuls), Lightning | | 3.4e-2 | 0.12 | 25.7 dB |

(Relative L2 errors.) The VAE encoder matches to 1.2e-6. The "was" figures came from two load-time
flaws, both fixed: the Lightning LoRA was merged into bf16-rounded weights before quantizing (its
update is 0.03-0.14% of each weight matrix's norm and the rounding erased 60-90% of it: bf16
merged weights alone moved the first velocity by 16.6%), and the modulation layers' few dominant
input channels shared Q8 blocks with the rest (the first block's modulation alone: 5.6%). What
remains is the model: the reference run with oxinfer's Q8 weights gives 3.4% and 25.7 dB (last
row), and oxinfer's Q8 run with the reference's prompt embeddings 3.5% and 25.3 dB. With the
Lightning LoRA, Qwen-Image amplifies weight noise about 6x into the first velocity (without it,
Q8 moves the first velocity by 0.21% and Q4 by 2.3% against oxinfer's bf16; they were 0.51% and
5.9%), then the four large steps amplify the difference again; the first-velocity error scales
linearly with the weight noise, so 1% would take ~10-bit weights. Quantized Lightning runs give a
visibly equivalent edit with different fine detail rather than the same pixels (as does the
default Metal bf16 run, whose half-precision matmul inputs give 28.9 dB). The text encoder at Q4
moves the conditioning by 24%, so `--quant q4` keeps it at Q8. diffusers 0.42 calls the text encoder without `mm_token_type_ids`, and
transformers 5 then gives the image tokens plain 1D positions; the reference script passes them
(M-RoPE, as the model was trained, and as oxinfer does).

**Image to image, inpainting, ControlNet, layers, FP8 files.** These follow diffusers'
`QwenImageImg2ImgPipeline`, `QwenImageInpaintPipeline`, `QwenImageEditInpaintPipeline`,
`QwenImageControlNetPipeline`, `QwenImageControlNetInpaintPipeline` and `QwenImageLayeredPipeline`:

- *Image to image*: the input is VAE-encoded (a sample of the posterior, as those pipelines take
  it), noised to the first of the last `steps * strength` noise levels, and denoised from there.
- *Inpainting*: the mask is resized to the output, binarized at one half and taken to the latent
  grid by nearest neighbour; after every step the kept region is set back to the input's latents
  noised to the next level (the clean latents after the last step). With Qwen-Image-Edit the
  input is also the reference image, as in `QwenImageEditInpaintPipeline`; Edit-2509 / 2511 have
  no diffusers inpainting pipeline, and oxinfer applies the same blending to their last image
  (no reference to check that combination against).
- *ControlNet* (`src/qwen_image/controlnet.rs`): InstantX's Union model is a copy of the first 5
  blocks with its own input layers; it runs on the noisy latents plus the embedded latents of the
  control image, each block's image stream goes through one linear layer, and the transformer
  adds residual `i / ceil(60 / 5)` after its block `i`. The residuals are computed once per step
  (with the positive prompt) and shared by the guided and unguided passes. The inpainting
  ControlNet (6 blocks) takes the latents of the image with the masked region blanked plus the
  kept-region mask as a 17th latent channel. ControlNets run with the text-to-image models.
- *Qwen-Image-Layered*: the composite and every layer are denoised together as frames 0..N of
  one token sequence, the input image's latents (through the RGBA VAE, opaque) follow at frame
  -1, the timestep embedding gets the "RGBA" embedding added, the sigmas are shifted linearly
  with `mu = sqrt(tokens / 256)`, and the layers (not the composite) are decoded to RGBA PNGs.
  Without a prompt the text encoder writes the caption (the pipeline's captioning prompt, greedy
  with its repetition penalty of 1.05, at most 512 tokens).
- *FP8 scaled checkpoints*: `float8_e4m3fn` weights times a scale, per tensor (ComfyUI's
  `scale_weight` with the `scaled_fp8` marker) or per output row (lightx2v's `weight_scale`),
  borrowed from the mapped file and run by the FP8 GEMMs; a `..._4steps_...` / `..._8steps_...`
  file name selects the Lightning settings like a Lightning LoRA does.

Checked against diffusers 0.42 on tiny random-weight pipelines (`tools/qwen_image_tiny.py make`
builds them: 3 blocks of width 64, a 2-block ControlNet, an RGBA VAE, bf16-exact weights; `ref`
runs the diffusers pipeline in float32 and hands oxinfer its noise), CPU, relative L2 of the final
latents and the 8-bit image:

| pipeline (tiny model) | settings | first residuals / velocity | final latents | image |
|---|---|---|---|---|
| image to image | 128x96, strength 0.6 of 5 steps | 2.1e-7 | 1.1e-7 | max pixel difference 1 |
| image to image, resized input, true CFG 2.5 | strength 0.75 of 6 steps | | 1.4e-7 | identical |
| inpainting | strength 0.8 of 5 steps | 2.8e-7 | 1.3e-7 | identical |
| inpainting, resized input and mask, true CFG 3 | strength 1, 4 steps | | 1.7e-7 | max difference 1 |
| Qwen-Image-Edit inpainting (1024x1024, 4096 + 4096 tokens) | strength 0.7 of 3 steps | 2.8e-7 | 2.0e-7 | max difference 1 |
| ControlNet, scale 0.8, true CFG 3 | 4 steps | 1.8e-7 (residuals) | 2.0e-7 | max difference 1 |
| ControlNet, scale 1.3, steps 0.2..0.8 of 5 | | 2.2e-7 | 9.3e-8 | max difference 1 |
| ControlNet, two control images | 3 steps | 2.5e-7 | 1.1e-7 | max difference 1 |
| inpainting ControlNet, true CFG 3 | 4 steps | 1.9e-7 (residuals) | 2.0e-7 | identical |
| Layered, 3 layers at 736x544 (6256 + 1564 tokens), true CFG 3 | 3 steps | | 4.3e-7 | max difference 1 |
| FP8, ComfyUI layout (reference: the decoded weights) | 3 steps | 2.3e-7 | 1.1e-7 | max difference 1 |
| FP8, per-row scales | 3 steps | 2.2e-7 | 1.1e-7 | max difference 1 |

These runs take the reference's prompt embeddings, so they measure the pipelines; with oxinfer's
own text encoding the tiny inpainting, ControlNet and layered runs give the same figures (prompt
embeddings 1.7e-7). The 512-token caption of the tiny layered model is the same text as
transformers' `generate` gives. On the real FP8 files, `fp8 * scale` of three layers read out of
`qwen_image_2512_fp8_e4m3fn_scaled.safetensors` (per-row scales) and of the ComfyUI-layout
`..._comfyui_4steps_...` file differs from Qwen-Image-2512's bf16 weights by 2.6-2.7% (FP8's
rounding; the latter has the Lightning LoRA merged); no generation was run from a 20 GB FP8 file.

On real weights (Qwen-Image-2512 with unsloth's Q3_K_M GGUF transformer, 512x512, CPU): the
residuals of InstantX's Union ControlNet at the first step, on oxinfer's own latents, control
tokens and prompt embeddings, match diffusers' `QwenImageControlNetModel` (float32) to 3.3e-6.
With 12 steps and true CFG 3 an edge map of a frontal stone cottage turns the three-quarter view
the prompt alone gives ("a wooden log cabin in a snowy pine forest at night") into a frontal
cabin with the door and windows where the edges are; with the 4-step Lightning LoRA the same
control image changes the picture but does not set its layout (the ControlNet was trained on
Qwen-Image, not on -2512 or the distilled model), so use it without the LoRA. Inpainting a
rectangle over the door of a generated cottage (Lightning, strength 1, "a bright blue door")
paints a blue door inside the mask and returns the rest at 32.7 dB of the input (the VAE round
trip of the kept region, as in diffusers); image to image at strength 0.6 ("in deep winter
snow") keeps the cottage and its framing. The inpainting ControlNet, Qwen-Image-Edit inpainting
and Qwen-Image-Layered were checked on the tiny models only (their weights were not downloaded).



**Speed and memory** (M5 Max, 1024x1024 output from a 512x512 photo: 4096 target + 4096 reference
+ ~210 text tokens per pass; Lightning: 4 passes, full quality: 80):

| weights | device | per transformer pass | Lightning edit (total) | 40 steps + CFG | peak memory (footprint) |
|---|---|---|---|---|---|
| bf16 (`--mmap`, 41 GB mapped) | Metal | 18 s | 84 s | 23 min (17 s / pass) | 10 GB |
| `--quant q8` | Metal | 16 s | 89 s | | 34 GB |
| `--quant q4` | Metal | 17 s | 89 s | | 22 GB |
| GGUF Q4_0 (12 GB, `--mmap`) | Metal | 18 s | 84 s | | 11 GB |
| GGUF Q4_0, `--stream --cache-budget-gb 4`, 512x512 | Metal | 13 s | 59 s | | 9.5 GB |
| bf16, 512x512 output (5300 tokens) | CPU, 18 threads | 185 s | 13 min | | |

The text encoder takes 1-2 s, the VAE 3 s to encode and 5 s to decode at 1024x1024; two input
images double the reference tokens (37 s per pass). Qwen-Image-2512 text to image (GGUF Q4_0 +
Lightning, 1024x1024): 58 s, 9.4 GB.

**Sample edits** (Edit-2511 bf16 + Lightning 4 steps, from a 512x512 fox photo and a bookshop that
Qwen-Image-2512 generated): "Make it winter" covers the ground, bench and branches in snow with
falling flakes; "Add a red knitted scarf around the fox's neck" adds the scarf and keeps the rest of
the photo; "Replace the text on the large shop sign with 'QWEN EDIT'" rewrites the gilded sign in
the same lettering and leaves the hanging sign and the window alone; with both images, "The fox
from Picture 1 sits on the sidewalk in front of the bookshop from Picture 2" places the fox there.

**Library and C API.**

```rust
use oxinfer::qwen_image::{Options, Params, Pipeline};
let mut pipe = Pipeline::load("Qwen/Qwen-Image-Edit-2511", Options { mmap: true, ..Default::default() })?;
let img = oxinfer::media::Media::new().load_image("fox.png".as_ref())?;
let (out, timings) = pipe.edit(&[img], &Params::new("make it winter"), &mut |i, n| eprintln!("{i}/{n}"))?;
// out: width, height, rgb; pipe.generate(&params, ..) for text to image; Params::lightning(prompt, 4)
// with a Lightning LoRA in Options::loras; edit_pictures(&[oxideav Picture]) with `oxideav`
// Params { strength: Some(0.6), mask: Some(mask), .. }: image to image / inpainting;
// Options { controlnet: Some(dir), .. } + Params { control: vec![Control { image, scale: 1.0 }], .. };
// pipe.decompose(&img, &Params::layered("", 4), ..) -> Vec<RgbaImage> (Qwen-Image-Layered)
```

The C API has `oxi_qwen_image_load` (`key=value` options: quant, text_quant, mmap, transformer,
controlnet, lora, keep_text_encoder, stream, device), `oxi_qwen_image_edit` (a JSON request:
prompt, images, negative, true_cfg, steps, seed, width, height, lightning, strength, mask,
control_images, control_scale, control_start, control_end), `oxi_qwen_image_edit_rgb` (pixels),
`oxi_qwen_image_cancel` and `oxi_qwen_image_free` (see `include/oxinfer.h`).

**Weights and licenses.** Qwen/Qwen-Image-Edit-2511, -2509, -Edit, Qwen/Qwen-Image, -2512:
Apache 2.0, ungated (57 GB each in bf16: 41 GB transformer, 16 GB text encoder, 0.25 GB VAE; the
text encoder and VAE files are identical across them). unsloth/Qwen-Image-Edit-2511-GGUF and
unsloth/Qwen-Image-2512-GGUF (Q2_K 7.7 GB to Q8_0 22 GB, BF16 41 GB), lightx2v/Qwen-Image-Edit-2511-Lightning
and lightx2v/Qwen-Image-2512-Lightning (0.85 GB LoRAs, 20 GB FP8 scaled transformers): Apache 2.0.
InstantX/Qwen-Image-ControlNet-Union (3.5 GB) and -Inpainting (4.2 GB), Qwen/Qwen-Image-Layered
(57 GB; unsloth/Qwen-Image-Layered-GGUF for its transformer): Apache 2.0.

Not supported: batches (one image per call), ComfyUI's mixed-precision (`comfy_quant`) and int8
single-file checkpoints, ControlNets with the edit models, several different ControlNets at once,
`padding_mask_crop` of the inpainting pipelines, layers through the C API and the server (the
command line and the library return them), the browser build (the weights exceed its 4 GB).
The transformer's activations go between CPU and GPU at every
matmul (about 1 s per pass at 1024x1024), the next lever for speed.

## OmniGen2: image editing, in-context generation and text to image

OmniGen2 edits an image from an instruction, composes several input images ("the cat from the
first image on the porch of the cabin in the second"), and generates from text, with one 4B
diffusion transformer conditioned by Qwen2.5-VL-3B. `src/omnigen2/` follows the official
`OmniGen2Pipeline` (github.com/VectorSpaceLab/OmniGen2; diffusers has no OmniGen2 pipeline); an
`OmniGen2/OmniGen2` folder is recognised from `model_index.json`.

```sh
# edit (50 steps, text guidance 5, image guidance 2, the official negative prompt)
./target/release/oxinfer --model OmniGen2/OmniGen2 --init-image cat.png \
    -p "Make the cat wear a small red wizard hat." -o hat.png
# in-context generation (up to 5 images), quantized transformer: 2.3 GB of weights
./target/release/oxinfer --model OmniGen2/OmniGen2 --quant q4 --init-image cat.png --init-image cabin.png \
    --size 1024x768 -p "The cat from the first image sits on the porch of the cabin in the second image." -o both.png
# text to image; the bf16 Qwen2.5-VL-3B-Instruct (7.5 GB) instead of the folder's float32 copy (15 GB)
./target/release/oxinfer --model OmniGen2/OmniGen2 --text-encoder Qwen/Qwen2.5-VL-3B-Instruct \
    --guidance 4 -p "A red fox sitting in a snowy forest at sunrise" -o fox.png
# understanding: without -o, --image asks the model's Qwen2.5-VL (the mllm/ folder as a chat model)
./target/release/oxinfer --model OmniGen2/OmniGen2 --image hat.png -p "What is the cat wearing?"
```

| Flag | Meaning |
|---|---|
| `--init-image FILE` | input image (repeatable, up to 5; none: text to image) |
| `--steps N`, `--guidance F`, `--image-guidance F`, `--negative TEXT` | 50, text guidance 5, image guidance 2, the official negative prompt |
| `--size WxH` | output size; default: the single input image's size (within 1024x1024 and 1024 per side, multiples of 16), else 1024x1024. With one input image the image is resized to it, so both token grids match |
| `--quant q8\|q4` | transformer precision (the text encoder follows, at Q8 for `q4`; `--text-quant` overrides) |
| `--text-encoder SPEC` | the Qwen2.5-VL-3B weights from another folder or repo id |
| `--transformer FILE`, `--mmap` | the transformer from a GGUF file (calcuis/omnigen2-gguf); map the weights |
| `--scheduler euler\|dpmsolver++` | the sampler: Euler (default) or the official DPM-Solver++ option |
| `--teacache F`, `--taylorseer` | a step cache (off by default; both skip transformer passes and change the image): TeaCache with relative-L1 threshold `F` (the official default is 0.05), or TaylorSeer |
| `--sample-posterior` | sample the VAE posterior of the input images instead of taking its mean (off by default) |

What runs: the instruction, in the pipeline's chat template (a system turn "You are a helpful
assistant that generates high-quality images based on user instructions." and the user turn), goes
through Qwen2.5-VL-3B as text only (the input images are not shown to it), and its final hidden
states condition the transformer. The input images are resized with Lanczos within 1024x1024
pixels and 1024 per side (never up; sides floored to 16), encoded by the FLUX VAE
(`crate::flux::vae`, posterior mean) and cut into 2x2 patches. The transformer
(`OmniGen2Transformer2DModel`, 2520 wide, 21 query / 7 key-value heads of 120) runs refiner blocks
per stream (2 unmodulated over the text, 2 over the noisy target, 2 over each input image on its
own after a learned per-image embedding), then 32 joint blocks over `[text, images..., target]`:
sandwich RMSNorms, tanh-gated `1 + scale` modulation, QK RMSNorm, SwiGLU, and Omni RoPE (text at
`(i, i, i)`, each image on its own grid at axis-0 offset text length + the larger sides of the
images before it). Euler steps over `t` from 0 (noise) to 1 with `t / (m - m t + t)`,
`m = sqrt(latent pixels) / 40`, and dual guidance `uncond + image_scale * (ref - uncond) +
text_scale * (cond - ref)` (three passes per step; `ref` has the images and the negative prompt,
`uncond` neither; two passes without input images). The noise and image refiners do not depend on
the prompt, so they run once per step for the three passes.

The 2520-wide layers fit neither Q8 / Q4 blocks of 32 nor the GPU's 64-row matmul tiles: they are
stored zero-padded (inputs to 2528, outputs to a multiple of 64 on a GPU) and their activations
padded and cropped around each matmul, an exact product. Heads of 120 are padded to 128 (queries
pre-scaled) for the GPU's flash-attention kernels; key-value heads are expanded. The published
checkpoint is float32: it is mapped and converted at load (to bf16, or quantized), so its 31 GB never
sit in memory.

**Agreement** with the official pipeline (torch 2.15, transformers 5.22, diffusers 0.42; float32
weights; same prompt, input image and initial noise; the VAE posterior mean instead of a sample on
both sides; relative L2 errors, PSNR on the 8-bit output):

| oxinfer | against | first velocity (3 passes, CFG) | image PSNR |
|---|---|---|---|
| prompt embeddings (Qwen2.5-VL-3B, bf16 = the float32 values) | float32 | 8.8e-6 | |
| bf16, CPU | reference with the same bf16-rounded weights | 2.4e-6 | |
| bf16, Metal | same | 2.4e-4 | 60.0 dB (edit, 512x512, 50 steps) |
| bf16, Metal, 2 input images | same | 4.2e-4 | 52.0 dB (in-context, 512x512, 50 steps) |
| bf16, Metal, text to image | same | 3.1e-4 | |
| bf16 | float32 weights | 5.4e-2 | 24.0 dB |
| `--quant q8` (CPU / Metal) | reference with our Q8 weights dequantized | 3.4e-4 / 4.0e-4 | |
| `--quant q8` | bf16-weight reference | 5.9e-3 | 40.8 dB (text encoder Q8 too) |
| `--quant q4` (CPU / Metal) | reference with our Q4 weights dequantized | 3.2e-3 / 2.9e-3 | |
| `--quant q4` | bf16-weight reference | 8.5e-2 | 20.6 dB (text encoder Q4 too; visually the same edit) |

Quantization error comes from the weights, not from the int8 activations of the CPU kernels: with
the same dequantized weights in the reference, Q8 agrees to 3e-4 and Q4 to 3e-3. bf16 weight
rounding alone moves the guided velocity by 5% (the official pipeline runs in bf16), and the
images then differ in fine detail only. Qwen2.5-VL-3B at Q4 moves the conditioning by 43% (Q8: 5%),
so `--quant q4` keeps the text encoder at Q8. OmniGen2's `mllm/` is exactly
Qwen/Qwen2.5-VL-3B-Instruct (identical hidden states), stored as float32.

**Speed and memory** (M5 Max; 50 steps, three transformer passes per step with input images, two
without; a 512x512 edit is 1024 target + 1024 image + ~40 text tokens per pass, a 1024x1024 edit
8.2k; rows marked * shared the GPU with another job):

| run | weights | device | per pass | total | peak memory (footprint) |
|---|---|---|---|---|---|
| edit, 512x512 | bf16 (8.0 GB) | Metal | 1.6-2.4 s | 4-6 min | 11.5 GB |
| edit, 512x512 | `--quant q8` | Metal | 3.0 s | 7.5 min | 10.1 GB |
| edit, 512x512 | `--quant q4` (2.3 GB) | Metal | 2.1 s | 5.2 min | 8.4 GB; 5.8 GB with `--text-encoder Qwen/Qwen2.5-VL-3B-Instruct` |
| edit, 512x512 | bf16 | CPU, 18 threads (shared) | 32-39 s | | 10.6 GB |
| edit, 512x512 | calcuis GGUF "Q4_0" (mostly F16) | CPU matmuls | 10.5 s | | |
| edit, 1024x1024 | bf16 | Metal | 6.9 s | 17 min | 20 GB |
| edit, 1024x1024 * | `--quant q4` | Metal | 13.5 s | 34 min | 13 GB |
| text to image, 1024x1024 * | `--quant q4` | Metal | 7.8 s | 13 min | 14 GB |
| 2 images (512² + 1024²) to 1024x768 * | `--quant q8` | Metal | 12.7 s | 32 min | 15 GB |

The text encoder takes 0.5 s; the VAE 0.4 / 1.7 s to encode and 0.7 / 2.8 s to decode at 512 /
1024. Memory beyond the weights is mostly the VAE at full resolution (its mid-block attention now
runs in query chunks, which took 3 GB off a 1024x1024 encode) and, from the float32 `mllm/`, the
text encoder's bf16 copy; for small devices use `--quant q4`, `--text-encoder
Qwen/Qwen2.5-VL-3B-Instruct` (mapped bf16, quantized to Q8) and 512x512 images.

**Sample edits** (bf16, 50 steps): a 512x512 cat photo with "Make the cat wear a small red wizard
hat" gets the hat and keeps the room; a flat 1024x1024 drawing of a house with "Make it night, with
warm light in the windows and stars in the sky" turns the sky dark blue with stars, the sun dark
and the window orange; a 512x512 green meadow cabin with "Change the season to winter" is covered
in snow (the same as the reference, 60 dB); in-context, "The cat from the first image sits on the
doorstep of the house in the second image" (Q8) puts the photographed cat in the drawn doorway;
text to image (Q4), "A photo of a red fox sitting in a snowy forest at sunrise" is a sharp,
detailed fox. The model's Qwen2.5-VL answers "What is the cat wearing?" with "The cat is wearing a
red witch hat." (600 tokens/s prefill, 26 tokens/s decode, bf16 on Metal).

**Library and C API.**

```rust
use oxinfer::omnigen2::{Options, Params, Pipeline};
let mut pipe = Pipeline::load("OmniGen2/OmniGen2", Options { quant: oxinfer::model::Quant::Q4, ..Default::default() })?;
let img = oxinfer::media::Media::new().load_image("cat.png".as_ref())?;
let (out, timings) = pipe.edit(&[img], &Params::new("add a red hat"), &mut |i, n| eprintln!("{i}/{n}"))?;
// pipe.generate(&params, ..) for text to image; edit_pictures(&[oxideav Picture]) with `oxideav`
// Params::new(..).with_teacache(0.05) / .with_taylorseer(); params.scheduler = Scheduler::DpmSolver;
// params.sample_posterior = true
```

The C API has `oxi_omnigen2_load` (`key=value` options: quant, text_quant, text_encoder, mmap,
transformer, keep_text_encoder, device), `oxi_omnigen2_edit` (a JSON request: prompt, images,
negative, steps, text_guidance, image_guidance, cfg_range, seed, width, height, max_pixels,
max_side, scheduler, teacache, taylorseer, sample_posterior; plus RGB8 pixel images), `oxi_omnigen2_cancel` and `oxi_omnigen2_free` (see
`include/oxinfer.h`). For understanding, load `mllm/` as an engine.

**Weights and license.** OmniGen2/OmniGen2: Apache 2.0, ungated; 31 GB, all float32 (15.9 GB
transformer, 15.0 GB Qwen2.5-VL-3B, 0.3 GB VAE). Qwen/Qwen2.5-VL-3B-Instruct (bf16, 7.5 GB) can
stand in for `mllm/` (`--text-encoder`). calcuis/omnigen2-gguf has GGUF transformers, but its
"Q4_0" file is 6 GB: the 2520-wide layers cannot be block-quantized, so most stay F16 (they load,
on the CPU); `--quant q4` (padded blocks) is smaller and runs on the GPU.

**Scheduler, step caches, posterior sampling.** All opt-in: each changes the image. Without them
the output is bit-identical to before they existed (same PNG and final latents with one input
image, two, and none, against a build of the previous commit, on the tiny model below).

- `--scheduler dpmsolver++` is the official script's option: the repository's
  `DPMSolverMultistepScheduler(algorithm_type="dpmsolver++", solver_type="midpoint",
  solver_order=2, prediction_type="flow_prediction")` on the same timesteps, working on
  `sigma = 1 - t` with the data prediction `x + sigma * v`. Its first and last steps are first
  order (and the second in effect, its difference term being divided by an infinite log-SNR
  ratio). `src/omnigen2/sched.rs` is checked step by step against that scheduler without a
  model: `tools/ref_omnigen2_sched.py` prints its sigmas and the sample after each of 6 steps,
  which the unit test holds (sigmas within 1e-7, samples within 2e-6).
- `--teacache F` is the official transformer's `enable_teacache`: per guidance branch
  (conditional, reference, unconditional: each its own state), the relative L1 change of the
  first joint block's modulated input since the previous step goes through the fitted polynomial
  (`-5.4826 x^4 + 11.4877 x^3 - 4.4741 x^2 + 2.4773 x - 0.0332`) and accumulates; while the sum
  is under `F` the 32 joint blocks are skipped and the last computed residual (blocks' output
  minus input) is added to the input instead. The first and last steps always run. The refiners
  run on every step (the decision needs their output).
- `--taylorseer` is the official pipeline's `enable_taylorseer` (order 4, interval 3, first
  enhance 5): per branch, the blocks run on its first 5 passes and then every 3rd; the passes
  between take the Taylor expansion of the hidden states, from finite differences between the
  last runs. The official code keeps the features of all 32 layers over the whole sequence; a
  skipped pass replaces every layer's output by its own prediction, so only the last layer's
  target rows reach the output, and those are what oxinfer keeps (the same arithmetic on the
  same values, megabytes per branch instead of gigabytes). A step whose passes are all
  predictions skips the refiners too. With both flags TaylorSeer wins, as in the official
  script.
- `--sample-posterior` draws each input image's latents as `mean + std * noise` like the official
  `latent_dist.sample()`, instead of the mean. The noise comes from oxinfer's generator
  (xoshiro256** seeded from `--seed`, one stream per input image, apart from the initial
  latents'), in `[c, h, w]` order: the reference's distribution, not its draw. oxinfer does not
  reproduce torch's generator (the initial latents are compared by passing the reference's noise
  in, and so is this), and the official pipeline draws this noise from torch's global generator
  rather than the run's seeded one, so there is no draw to match bit for bit: the mean stays the
  default. The posterior is narrow: for a 256x256 cat photo its standard deviations are 1.4e-4
  of the latents (L2), and sampling moved the 30-step edit below by 66.6 dB PSNR.

Checked against the official `OmniGen2Pipeline` itself (float32, CPU) on a tiny random-weight
model (`tools/omnigen2_tiny.py`: a 4-block 96-wide transformer and a small 16-channel VAE; the
reference's prompt embeddings, initial noise and posterior noise passed to oxinfer; 64x64 output;
relative L2 error of the final latents, and whether every pass made the same run / skip
decision):

| options | input images | steps | passes that ran the blocks (of all) | same decisions | final latents |
|---|---|---|---|---|---|
| none | 1 | 12 | 36 (36) | | 9.8e-7 |
| `--teacache 0.5` | 1 | 30 | 85 (90) | yes | 7.3e-7 |
| `--teacache 1.0` | 1 | 30 | 73 (90) | yes | 7.0e-7 |
| `--teacache 2.0` | 1 | 30 | 43 (90) | yes | 9.6e-7 |
| `--teacache 1.0` | 0 (text to image) | 30 | 28 (60) | yes | 7.5e-7 |
| `--taylorseer` | 1 | 30 | 39 (90) | yes | 9.9e-7 |
| `--taylorseer` | 2 | 30 | 39 (90) | yes | 9.3e-7 |
| `--taylorseer` | 0 (text to image) | 30 | 26 (60) | yes | 9.4e-7 |
| `--scheduler dpmsolver++` | 1 | 20 | 60 (60) | | 9.4e-7 |
| `--taylorseer --scheduler dpmsolver++` | 1 | 50 | 60 (150) | yes | 9.5e-7 |
| `--sample-posterior` | 1 | 12 | 36 (36) | | 9.6e-7 (sampled reference latents: 7.8e-7) |
| `--teacache 2.0 --scheduler dpmsolver++ --sample-posterior` | 2 | 30 | 39 (90) | yes | 8.3e-7 |

The tiny model needs thresholds far above 0.05 to skip anything (its random blocks move the
modulated input a lot between steps). On the real model (`--quant q4`, CUDA, a 256x256 cat photo,
"Make the cat wear a small red wizard hat.", seed 3; PSNR of the output against the same run
without the option, which is the size of the change the option makes, not an error):

| run | steps | passes that ran the blocks (of all) | image against the plain run |
|---|---|---|---|
| `--teacache 0.05` | 50 | 105 (150) | 46.4 dB |
| `--teacache 0.1` | 50 | 71 (150) | 40.5 dB |
| `--teacache 0.05` | 30 | 90 (90): nothing skipped | identical |
| `--teacache 0.2` | 30 | 44 (90) | 37.3 dB |
| `--taylorseer` | 30 | 39 (90) | 32.3 dB |
| `--scheduler dpmsolver++` | 30 | 90 (90) | 38.8 dB |
| `--sample-posterior` | 30 | 90 (90) | 66.6 dB |

All of them give the same edit to the eye (the same hat on the same cat). The step caches were
compared with the official pipeline on the tiny model only; the real model was not run in the
Python reference for them.

Not supported: OmniGen2 LoRAs, batches, `--save-quantized` for the padded transformer layers
(quantization happens at each load, a few seconds), and the reflection / RL variants
(OmniGen2-RL, EditScore).

## BAGEL: image editing, text to image and image understanding in one model

ByteDance's BAGEL-7B-MoT is a single unified model: it edits images from instructions, generates
them from text, answers questions about them, and can write a plan ("thinking") before an image or
an answer. `src/bagel/` follows the official inference code (`github.com/ByteDance-Seed/Bagel`:
`InterleaveInferencer`, `modeling/bagel`); there is no diffusers pipeline. The model is picked
from the folder (`llm_config.json` + `vit_config.json` + `ae.safetensors`).

```sh
# editing: --init-image is the picture, --prompt the instruction (29.5 GB download, bf16)
./target/release/oxinfer --model ByteDance-Seed/BAGEL-7B-MoT --mmap --init-image dog.png \
    -p "Put a red knitted hat on the dog" -o hat.png
# ~4x cheaper: inputs resized to at most 512 pixels per side (the output follows the input's size)
./target/release/oxinfer --model ByteDance-Seed/BAGEL-7B-MoT --mmap --max-size 512 --init-image dog.png \
    -p "Turn this photo into a watercolor painting" -o watercolor.png
# text to image, with a written plan first (printed)
./target/release/oxinfer --model ByteDance-Seed/BAGEL-7B-MoT --mmap --think --size 512x512 \
    -p "a car made of small cars" -o cars.png
# image understanding: without -o the answer is text
./target/release/oxinfer --model ByteDance-Seed/BAGEL-7B-MoT --mmap --init-image cat.png -p "What is in this image?"
```

| Flag | Meaning |
|---|---|
| `--init-image FILE` | with `-o`: the image to edit; without: the image to ask about (repeatable: the instruction follows all images) |
| `--steps N` | timesteps (default 50: 49 model evaluations per guidance branch) |
| `--cfg-text F`, `--cfg-img F` | text guidance (4) and image guidance (2 editing, 1 = off for text to image) |
| `--cfg-interval LO,HI` | timesteps with guidance (0,1 editing; 0.4,1 text to image) |
| `--cfg-renorm T`, `--cfg-renorm-min F` | CFG-renorm: `text_channel` (editing) / `global` / `channel`; lower bound 0 |
| `--timestep-shift F` | 3 |
| `--size WxH` | text-to-image size (default 1024x1024; at most 1024 per side) |
| `--max-size N`, `--min-size N` | input resize bounds (1024 and 512, the reference's `ImageTransform(1024, 512, 16)`) |
| `--think` | plan before the image, or reason before the answer (greedy, at most 1000 tokens) |
| `--taylorseer` | TaylorSeer step cache (off by default: fewer language-model passes, a different image) |
| `--sample-posterior` | sample the VAE posterior of input images instead of taking its mean (off by default) |
| `--quant q8\|q4`, `--mmap` | weight precision (quantized at load); map the checkpoint |

What runs: the language model is Qwen2.5-7B (with q/k RMSNorm added) as a Mixture-of-Transformers:
every layer has an understanding expert (text and SigLIP tokens) and a generation expert (VAE
latent tokens) with their own norms, projections and MLPs, meeting in one self-attention over the
packed sequence. Each text is wrapped as `<|im_start|>...<|im_end|>` and goes through causally. Each
input image is resized (sides multiples of 16, 512..1024), VAE-encoded with the FLUX autoencoder
(`flux::vae`, BFL file layout; posterior mean), cut into 2x2 latent patches through `vae2llm` plus a
timestep embedding of 0 and a 2-D sin-cos position embedding (generation expert), then read again
by SigLIP so400m/14 at its own size up to 980 pixels (NaViT style: one token per patch, learned
70x70 position table, 26 layers, connector MLP; understanding expert); every token of an image has
one RoPE position and attends bidirectionally. The output image's noise latents go after the
context at the next position and are denoised by rectified flow: `steps - 1` Euler steps over
`t = 1 -> 0` shifted by `3t / (1 + 2t)`, the generation expert predicting the velocity through
`llm2vae`. Guidance has two terms, evaluated as extra branches over shared KV caches (no copies):
text (the context before the instruction) and image (the texts without the images, a separate
small context); BAGEL's CFG-renorm caps the guided velocity's norm (per token on the text term for
editing). The three branches of a step run as one batch through every matmul. Thinking and answers
come from the understanding expert's LM head, decoding after `<|im_start|>`; the plan is then
appended to the context as text before the image is generated.

**Agreement** with the official code (the repository's modules with flash-attn replaced by SDPA, same inputs and initial noise; the VAE switched to its posterior mean
as oxinfer encodes):

| check | reference | oxinfer | relative error |
|---|---|---|---|
| SigLIP features, 512x512 cat (1369 patches) | float32, CPU | CPU | 1.6e-5 (connector output 1.3e-5) |
| same | float32 | Metal (matrix units) | 9.3e-4 |
| LM logits after an image question, 2 layers | float32 | CPU | 1.9e-6 (KL 8e-8) |
| first denoising step, editing, 2 layers: conditional / text-free / image-free / guided velocity | float32 | CPU | 8.7e-6 / 7.9e-6 / 7.1e-7 / 2.5e-5 |
| LM logits after an image question, full model | bf16 (CPU autocast) | CPU | 6.6e-2, KL 3.2e-4, same top 5, same 40 greedy tokens |
| first step velocity, full model, editing (cond / guided) | bf16 (MPS) | CPU | 1.3e-2 / 3.0e-2 |
| same | bf16 | Metal | 1.3e-2 / 3.1e-2 (Metal vs CPU: 8.9e-4 / 3.3e-3) |
| same, `--quant q8` / `q4` | bf16 | Metal | 1.3e-2 / 3.9e-2 (vs oxinfer bf16: 2.8e-3 / 3.7e-2) |
| edited image, 50 steps, cat 512x512 "Put sunglasses on the cat" | bf16 | Metal | 41.4 dB PSNR |
| text to image, 512x512 fox | bf16 | Metal | 44.0 dB PSNR |

The truncated float32 runs check the implementation (the f32 reference needs no 58 GB); the
full-model numbers are bf16-activation (reference) against f32-activation (oxinfer) rounding, and
the images still agree to 41-44 dB. Greedy thinking text follows the reference for its first ~30
tokens and then picks another near-tie wording (different plan, different image). Quantized
weights: q8 moves the first velocity by 0.28% against oxinfer's bf16 (the 1024-row matmuls run on
dequantized weights, so activations are never rounded to int8 here), q4 by 3.7%.

**Speed and memory** (M5 Max, machine shared with other jobs; 50 timesteps; an edit runs three
branches per step):

| task | weights | device | per branch pass | total | peak footprint |
|---|---|---|---|---|---|
| edit, 512x512 input (1024 latent + 2404 context tokens) | bf16 (`--mmap`) | Metal | 1.07 s | 2.9 min (147 passes) | 5.5 GB |
| same | `--quant q8` | Metal | 0.99 s | 2.8 min | 20 GB |
| same | `--quant q4` | Metal | 1.01 s | 2.8 min | 13 GB |
| same | bf16 (`--mmap`) | CPU, 18 threads | 18 s | ~45 min | 3.1 GB |
| same | `--quant q4` | CPU, 18 threads | 20 s | ~50 min | 11.6 GB |
| edit, 1024x768 input (3072 latent + 6727 context tokens) | bf16 (`--mmap`) | Metal | 3.4-3.6 s | 9.3 min | 10.4 GB |
| text to image, 512x512 | bf16 (`--mmap`) | Metal | 1.08 s | 1.7 min (90 passes) | 3.9 GB |
| text to image, 1024x1024 | bf16 (`--mmap`) | Metal | 4.1 s | ~6 min (~90 passes) | 15.4 GB |

The official code in PyTorch 2.15 on the same GPU (MPS, bf16) takes 98 s for the 512x512 edit and
109 s for the 512x512 text to image. The context (VAE + SigLIP + text) takes 5-6 s at 512x512 and
17-29 s at 1024x768 on Metal, 46 s at 512x512 on the CPU; the VAE decodes in 0.8 s (512) to 3 s (1024x768).
A 100-token answer about a 512x512 image takes 11.7 s on Metal (bf16), 6.9 s with `--quant q4`,
33 s on the CPU (single-token passes run on the CPU; about 10 tokens/s in bf16). Without `--mmap` the
whole 29 GB checkpoint is read into memory first (quantizing then keeps both: 41 GB at q4), so
`--mmap` is the way to run on smaller machines. At 1024x1024 the activations of the three
batched guidance branches (3 x 4098 tokens) dominate the footprint.

**Sample edits** (bf16, Metal): a golden retriever on a porch, "Put a red knitted hat on the dog",
gets a red beanie and nothing else changes; a green meadow with a cabin, "Make it winter: snow on
the roof and the ground", snows over roof and field; a bakery front, "Change the text on the sign from 'FRESH BREAD' to
'OXINFER CAFE'", rewrites the carved sign in the same style but spells it "OXIVFR CAFE" (BAGEL's
text rendering is its weak spot); "Turn this photo into a watercolor painting" repaints the whole
porch scene. `--quant q8` reproduces the bf16 edit at 51 dB (41.3 dB against the reference), q4 at
36 dB.

**Library and C API.**

```rust
use oxinfer::bagel::{Bagel, ImageOptions, LoadOptions, TextOptions};
let m = Bagel::load("ByteDance-Seed/BAGEL-7B-MoT", &LoadOptions { mmap: true, ..Default::default() })?;
let img = oxinfer::media::Media::new().load_image("dog.png".as_ref())?;
let (out, stats) = m.edit(&img, "put a red hat on the dog", &ImageOptions::default(), &mut |i, n| { eprintln!("{i}/{n}"); true })?;
let (fox, _) = m.generate("a red fox in the snow", &ImageOptions { width: Some(512), height: Some(512), ..Default::default() }, &mut |_, _| true)?;
let answer = m.understand(Some(&img), "What breed is this dog?", &TextOptions::default(), &mut |_| oxinfer::generate::Flow::Continue)?;
// generate_interleaved / understand_interleaved take any sequence of texts and images;
// edit_picture / understand_picture take oxideav pictures (`oxideav` feature)
```

`examples/bagel_edit.rs` edits an image and then describes the result with the same model. The C
API has `oxi_bagel_load` (`key=value` options: quant, mmap, device, cache_dir, generation,
understanding, text), `oxi_bagel_edit_rgb` / `oxi_bagel_edit_file` / `oxi_bagel_generate` (JSON
options as `ImageOptions`), `oxi_bagel_understand_rgb` / `_file` (streamed text),
`oxi_bagel_last_thinking`, `oxi_bagel_cancel` and `oxi_bagel_free` (see `include/oxinfer.h`).
Loading only what a task needs (`generation=0` for questions, `understanding=0` for text to image)
leaves the other expert's weights untouched in the mapped file.

**Weights and license.** ByteDance-Seed/BAGEL-7B-MoT: Apache 2.0, ungated; `ema.safetensors`
29.2 GB (bf16: both experts of the 7B language model, SigLIP, connectors) and `ae.safetensors`
0.34 GB (the FLUX.1 autoencoder, f32). No GGUF conversion of BAGEL exists; the community NF4 /
INT8 / FP8 / DFloat11 checkpoints are not supported (quantize at load with `--quant`).

**TaylorSeer and posterior sampling.** Both opt-in: they change the image. Without them the
output is bit-identical to before they existed (same PNG for an edit and a text-to-image run
against a build of the previous commit; real weights, the language model cut to 2 layers).

`--taylorseer` (`"taylorseer": true`) is the reference's `enable_taylorseer` (order 6, interval
3, first enhance 5): each guidance branch has its own state; it runs the language model on its
first 5 passes and then every 3rd, and the passes between take the Taylor expansion of the
hidden states, from finite differences between the last runs. The reference keeps the features
of every layer; on a skipped pass each layer returns its own prediction, so only the last
layer's latent-token states (before the final norm) reach `llm2vae`, and those are what oxinfer
keeps. Branches that run on the same step still share one batch. Against the official code
(`tools/ref_bagel_taylorseer.py`: the repository's model with the real weights, cut to 2 layers
and run in float32 on the CPU, its initial noise passed to oxinfer run with the same cut;
text to image, 256x256):

| run | steps | passes that ran the model (of all) | same decisions | final latents (relative L2) |
|---|---|---|---|---|
| no cache | 12 | 20 (20) | | 7.8e-6 |
| `--taylorseer` | 30 | 24 (53) | yes | 5.0e-5 |
| `--taylorseer --cfg-img 1.5` (three branches) | 50 | 53 (131) | yes | 9.6e-5 |

The larger error with the cache is float32 rounding through the order-6 differences, on the
same schedule. The full 28-layer model was not run with the cache here, and the editing path
was not compared with the reference (it differs from text to image only in the contexts the
branches attend to).

`--sample-posterior` (`"sample_posterior": true`) draws each input image's VAE latents as
`mean + exp(0.5 logvar) * noise`, as the reference's `DiagonalGaussian`, instead of the mean.
The noise comes from oxinfer's generator (xoshiro256** seeded from `--seed`, one stream per
input image, apart from the initial latents'): the reference's distribution, not its draw, which
comes from torch's global generator and cannot be matched, so the mean stays the default. The
sampling code (`src/vae_posterior.rs`) is the one checked against diffusers on OmniGen2's tiny
model above, here without diffusers' log-variance clamp, which the BFL autoencoder does not
have; BAGEL's own sampler was not compared with the reference. For a 256x256 cat photo the
posterior's standard deviations are 1.4e-4 of the latents (L2).

Not supported: batches, interleaved outputs past one image or one answer, the browser build (the
weights exceed 4 GB). Single-token text decoding runs on the CPU (about 10 tokens/s in bf16).

## Step1X-Edit: image editing from instructions

StepFun's Step1X-Edit edits an image the way an instruction says. `src/step1x/` follows the
`Step1XEditPipelineV1P2` / `Step1XEditPipeline` of StepFun's diffusers fork (branch
`step1xedit_v1p2` of Peyton-Chen/diffusers); v1.2 (`stepfun-ai/Step1X-Edit-v1p2`, "ReasonEdit-S",
and `-v1p2-preview`) and v1.1 (`stepfun-ai/Step1X-Edit-v1p1-diffusers`) are told apart by the
transformer's `enable_text_token_mapping`.

```sh
# 28 steps, true CFG 6 against an empty instruction (the pipeline's defaults; StepFun's examples use 50)
./target/release/oxinfer --model stepfun-ai/Step1X-Edit-v1p2 --mmap --init-image dog.png \
    -p "Put a red knitted hat on the dog" -o hat.png
# quoted text is rendered character by character
./target/release/oxinfer --model stepfun-ai/Step1X-Edit-v1p2 --mmap --init-image bakery.png \
    -p 'Change the text on the sign from "FRESH BREAD" to "OXINFER CAFE"' -o sign.png
# thinking mode: the VL model rewrites the instruction from the image first
./target/release/oxinfer --model stepfun-ai/Step1X-Edit-v1p2 --mmap --init-image cabin.png --think \
    -p "Show what this place would look like after 100 years of abandonment" -o old.png
# a 512x512 output worked at 512x512 (4x fewer tokens), q4 transformer
./target/release/oxinfer --model stepfun-ai/Step1X-Edit-v1p2 --quant q4 --init-image cat.png \
    --size 512x512 -p "Put sunglasses on the cat" -o cat.png
```

| Flag | Meaning |
|---|---|
| `--init-image FILE` | image to edit (without one: text to image, the vision tower seeing a black square) |
| `--steps N`, `--guidance F`, `--negative TEXT` | 28, true CFG 6, `""` |
| `--size WxH` | output size and working area; the input is center-cropped to that aspect ratio. Default: the input's size, worked at about 1024x1024 (multiples of 16) at its aspect ratio |
| `--think` | v1.2 thinking mode (sampled rewrite, temperature 0.7, top-p 0.9, up to 128 tokens) |
| `--sample-posterior` | sample the VAE posterior of the input image instead of taking its mean (off by default) |
| `--quant q8\|q4` | transformer precision (the text encoder: q8 with either) |
| `--mmap`, `--stream` | map the weights; stream the transformer's 57 blocks from storage (`--cache-budget-gb`) |

What runs: the input is resized (Lanczos) to the working size and given to Qwen2.5-VL-7B (v1.2
ships a fine-tuned copy) inside the pipeline's template: a system turn, a fixed editing prefix, the
instruction, then `Picture 1: <|vision_start|>...<|vision_end|>` with 256 to 400 vision tokens.
The template is split at special tokens and quotes and each piece tokenized alone, so quoted text
becomes one token per character; the text encoder gets explicit M-RoPE positions in which, as the
pipeline's `attention_mask = input_ids > 0`, a token with id 0 (a lone `!`) sits at position 1
and the rest close the gap. The last hidden states minus the first 90 tokens condition the DiT. A
connector (HunyuanVideo's token refiner: two self-attention blocks gated by the timestep embedding
plus a projection of the states' mean) maps them to the 4096-wide text stream and pools the
768-wide vector FLUX takes from CLIP; v1.2 adds the raw token embeddings of the quoted characters
through `text_token_mapping`. The DiT is FLUX.1's (19 double + 38 single blocks, the
`crate::flux::transformer` blocks as they are) over `[text; target; reference]`: the reference is
the input resized with torch's antialiased bilinear filter and VAE-encoded (the 16-channel FLUX
VAE), on the target's grid at RoPE index 1 (v1.1: shifted by one row and column). v1.2 gives the
text tokens negative positions on the first RoPE axis and the vision tokens the latent-patch
coordinates of the pixels they cover. Flow matching with FLUX's resolution-dependent shift; true
CFG where the guidance difference is divided by its per-token norm^0.4 when that norm exceeds 1.
The decoded image is resized back to the input's size (Pillow bicubic). v1.1 pads the text stream
to 640 tokens (masked in the connector, not in the DiT) with all text positions at 0.

Components load when first needed; the 16 GB text encoder is dropped after encoding. On Metal the
text encoder and the VAE encoder run with exact (f32) matmuls, and the VAE's f32 weights keep their
precision as a bf16 `hi + lo` pair (Step1X ships the FLUX VAE in f32: rounding its weights to bf16
moves the reference latents by 1.5%, the final image by 20 dB). The transformer uses the matrix
units, attention included (`exact_attention=1` / `OXINFER_EXACT_ATTENTION=1` gives it f32
operands, still on the matrix units).

**Agreement** with the StepFun diffusers fork + transformers 4.55 (bf16 weights on both sides,
every reference matmul in f32, VAE posterior mean, same initial noise; v1.2, a 512x512 cat photo,
`Add a red hat to the cat and write "MEOW" on a sign!` (quotes and a lone `!`), `--size 512x512`,
8 steps, CFG 6; relative L2 errors):

| oxinfer | first velocity (cond / uncond) | final latents | image PSNR |
|---|---|---|---|
| bf16, CPU (1 step) | 9.7e-5 / 3.5e-5 | 4.1e-4 | 62.0 dB |
| bf16, Metal, f32 attention | 3.4e-4 / 2.4e-4 | 1.1e-2 | 48.7 dB |
| bf16, Metal (default) | 3.3e-4 / 2.6e-4 | 1.3e-2 | 47.3 dB |
| `--quant q8`, Metal | 3.8e-3 / 2.5e-3 | 6.6e-2 | 33.9 dB |
| `--quant q4`, Metal | 4.6e-2 / 4.3e-2 | 0.25 | 21.7 dB |
| v1.1 (`-v1p1-diffusers`), Metal, 1024x1024, 1 step | 1.4e-2 / 2.7e-4 | 1.7e-2 | 49.5 dB |

Token ids are identical (both prompts), the conditioning hidden states agree to 2.1e-4, the RoPE
ids of the text to 2e-8, the reference latents to 3.5e-5. The quantized rows were also compared
with a reference running oxinfer's Q8 / Q4_0 weights dequantized: the weights alone move the first
velocity by 2.8e-3 (Q8) and 4.5e-2 (Q4); oxinfer's quantized matmuls add 2.6e-3 / 3.5e-3 on top,
so the Q4 error is the 4-bit weights, not the activations. CFG 6 then amplifies differences over
the steps: quantized edits look the same (same hat, same lettering, same composition) with
different fine detail. Below ~1e-2 the final latents are chaotic: multiplying the queries of the
f32 attention by 1 +- 1e-7 (random signs, about one f32 rounding step) moves them anywhere between
6e-3 and 1.4e-2 against the reference (52 to 47 dB), so the bf16 rows' final numbers are draws from
that band, and their first velocities are what measures precision: half-precision attention
leaves those unchanged. v1.1's text encoder also masks the `!` token as a key; oxinfer reproduces
that for every other token, while the `!` row itself still sees its own key (that one row is 36%
off, hence the 1.4e-2 on the conditioned velocity; prompts without a lone `!` match like v1.2).

**Speed and memory** (M5 Max, 1024x768 input worked at 1168x880: 4015 target + 4015 reference +
~410 text tokens per pass, two passes per step):

| weights | device | per transformer pass | 28-step edit | peak memory (footprint) |
|---|---|---|---|---|
| bf16 (`--mmap`, 24.9 GB mapped) | Metal | 9.3 s (f32 attention: 34% slower) | 9 min | 12.6 GB |
| bf16, read into memory | Metal | 17.5 s | | 40 GB |
| `--quant q8` | Metal | 14.0 s | | 28 GB |
| `--quant q4` | Metal | 14.2 s | | 20 GB |
| bf16, `--stream --cache-budget-gb 8` | Metal | 13.6 s | | 12.7 GB |
| bf16, 512x512 (2400 tokens) | Metal | 3-4 s | 3.5 min | 3.7 GB |
| bf16, 512x512 | CPU, 18 threads | 34 s | | |
| bf16, 1024x1024 (8832 tokens, v1.1) | CPU, 18 threads | 213 s | | |

The text encoder takes 3-6 s, the VAE 2.5 s each way at this size; the thinking mode adds ~9 s.

**Sample edits** (v1.2 bf16, 28 steps, default sizes): "Put a red knitted hat on the dog" (a 1024x768
porch photo) adds the hat and leaves every other pixel's content in place, with no shift or zoom
(`--quant q4` gives the same hat); "Change the text on the sign from "FRESH BREAD" to "OXINFER
CAFE"" rewrites the carved sign in the same lettering; "Put sunglasses on the cat", "Make it
winter, with snow on the roof and the ground" (a snowstorm over the same cabin) and "Turn this
photo into a watercolor painting" do what they say. Thinking mode turned "Show what this place
would look like after 100 years of abandonment" into "Change the wooden shed to a dilapidated
and overgrown log cabin." and weathered the cabin only lightly.

**Library and C API.**

```rust
use oxinfer::step1x::{Options, Params, Pipeline};
let mut pipe = Pipeline::load("stepfun-ai/Step1X-Edit-v1p2", Options { mmap: true, ..Default::default() })?;
let img = oxinfer::media::Media::new().load_image("dog.png".as_ref())?;
let (out, timings) = pipe.edit(&img, &Params::new("put a red hat on the dog"), &mut |i, n| eprintln!("{i}/{n}"))?;
// out: width, height, rgb (the input's size); Params { size_level, width, height, think, true_cfg,
// steps, seed, negative, .. }; pipe.generate(&params, ..) without an image; edit_picture with `oxideav`
```

The C API has `oxi_step1x_load` (`key=value` options: quant, text_quant, mmap, transformer,
keep_text_encoder, stream, exact_attention, device), `oxi_step1x_edit` (a JSON request: prompt,
images, negative, true_cfg, steps, seed, size_level, width, height, think, sample_posterior; or
RGB pixels),
`oxi_step1x_cancel` and `oxi_step1x_free` (see `include/oxinfer.h`). `examples/step1x_edit.rs`
edits one image.

**Weights and licenses.** stepfun-ai/Step1X-Edit-v1p2, -v1p2-preview and -v1p1-diffusers: Apache
2.0, ungated; 42 GB each (24.9 GB transformer in bf16, 16.6 GB text encoder, 0.34 GB VAE in f32).
v1.1's text encoder is stock Qwen2.5-VL-7B-Instruct (the same files as Qwen-Image's, which can be
linked instead of downloaded twice); v1.2's is fine-tuned. No GGUF conversions are published:
`--quant q8 / q4` quantizes at load (2-3 s from the mapped bf16 files).

Not supported: the reflection mode of v1.2 (a multi-round loop where the VL model scores the edit
and retries), the original `stepfun-ai/Step1X-Edit` single-file checkpoints (v1.0 / v1.1 in the
original repository's layout; the diffusers folders cover the same models) and its fp8 copies,
LoRAs, the RegionE acceleration, batches, the browser build (the weights exceed its 4 GB).

The reference image is VAE-encoded at the posterior mean by default (the pipeline samples it).
`--sample-posterior` (`"sample_posterior": true`, `Params::sample_posterior`) draws it as
`mean + std * noise` like the pipeline's `retrieve_latents(..., sample_mode="sample")`, with the
log-variance clamped to [-30, 20] as diffusers does. The noise comes from oxinfer's generator
(xoshiro256** seeded from `--seed`, a stream apart from the initial latents'): the pipeline's
distribution, not its draw, since oxinfer does not reproduce torch's generator (checks pass the
reference's noise in, as for the initial latents). So the mean stays the default, and the
agreement table above is measured with it. The sampling code is the one checked against
diffusers' `latent_dist.sample()` on OmniGen2's tiny model (sampled latents within 7.8e-7 with
the same noise), where the same kind of VAE measured a posterior 1.4e-4 of the latents wide.
Step1X-Edit itself was not run with the flag, nor re-run after this change (its 42 GB of weights
are more than this check could load on the shared machine); its default path calls the same
`Vae::encode` as before.

## Janus-Pro: text to image and image understanding

DeepSeek's Janus-Pro is one autoregressive model that both draws and reads images (`src/janus/`);
JanusFlow, its rectified-flow sibling, is described further down.
A Janus checkpoint (`model_type` `multi_modality`) is recognised from its config:

```sh
./target/release/oxinfer --model deepseek-ai/Janus-Pro-1B --generate-image "a red fox in the snow" -o fox.png
./target/release/oxinfer --model deepseek-ai/Janus-Pro-7B --quant q8 -p "a watercolor lighthouse" \
    -o lighthouse.png --num-images 4 --cfg 5 --temp 1 --seed 7          # lighthouse-0.png .. -3.png
./target/release/oxinfer --model deepseek-ai/Janus-Pro-1B --image chart.png -p "Which slice is the largest?"
```

**Text to image.** The prompt goes in the reference's SFT format (`<|User|>: PROMPT\n\n<|Assistant|>:`
plus `<begin_of_image>`), then 576 image tokens (a 24x24 grid) are sampled from `gen_head` with
classifier-free guidance, `uncond + cfg * (cond - uncond)`, the unconditional sequence being the
prompt with everything between BOS and `<begin_of_image>` replaced by padding. Each sampled token
is fed back through `gen_embed` and `gen_aligner`. The conditional and unconditional sequences of
every image run as one batch through the LLaMA decoder (DeepSeek-LLM, own KV cache per sequence),
so each of the 576 steps reads the weights once whatever the number of images; with `--quant
q8|q4` the int8 decode kernels take up to 15 rows (7 images) per step. The LlamaGen VQ-16 decoder
(l2-normalized 16384 x 8 codebook, taming-style ResNet / attention decoder, 16x upsampling) is built
from the Stable Diffusion blocks, so its convolutions run on the GPU when one is active. Options
are the reference's: CFG weight 5, temperature 1 (`--temp 0` is greedy), plus top-k / top-p and
`--num-images` (`parallel_size`). `--cfg` is an alias of SD's `--guidance`; `-p`, `-o` and `--seed`
are shared with Stable Diffusion.

**Image understanding.** Each image is resized (Pillow bicubic, longer side 384), padded to a
square with the processor's grey and encoded by the SigLIP-L/16 tower plus the aligner into 576
embeddings placed at `<image_placeholder>` (one per image, or before the question); the answer is
greedy by default, as in the reference's `inference.py`, with its system prompt.

In the library:

```rust
use oxinfer::janus::{ChatOptions, ImageOptions, Janus, JanusOptions};
let jm = Janus::load("deepseek-ai/Janus-Pro-1B", &JanusOptions::default())?;
let img = jm.generate_image("a red fox in the snow", &ImageOptions { seed: 7, ..Default::default() })?;
std::fs::write("fox.png", oxinfer::png::encode_rgb(img.width, img.height, &img.rgb))?;
// several images, a progress callback per image-token step (Flow::Stop cancels), timings:
let (imgs, stats) = jm.generate_images("a lighthouse", &ImageOptions { parallel_size: 4, ..Default::default() },
    &mut |p| { eprint!("\r{}/{}", p.step, p.steps); oxinfer::generate::Flow::Continue })?;
let chart = oxinfer::media::Media::new().load_image("chart.png".as_ref())?; // or any video::RgbImage
let answer = jm.understand("Which slice is the largest?", &[chart], &ChatOptions::default(), &mut |t| {
    print!("{t}"); oxinfer::generate::Flow::Continue })?;
```

`Image::to_video_frame()` gives an oxideav `VideoFrame` + `CodecParameters` (feature `oxideav`);
`Janus::cancel_handle()` stops a generation from another thread; `generate_tokens`,
`decode_tokens`, `image_logits` (teacher-forced guided logits) and `encode_image` expose the
stages. The C API has `oxi_janus_load`, `oxi_janus_generate` (images through a callback, a step
callback that can cancel), `oxi_janus_understand` / `_rgb` and `oxi_janus_cancel`
(`include/oxinfer.h`). Weights load from `pytorch_model.bin`, sharded `pytorch_model-*.bin` with
their index, or safetensors.

Agreement with the official `janus` package in float32 on CPU (`tools/ref_janus.py` vs
`examples/janus_check.rs`, oxinfer with bf16 weights as stored, `OXINFER_METAL_MMA=0`):

| Janus-Pro-1B | max abs diff | rel L2 | |
|---|---|---|---|
| text logits after a 56-token prompt | 1.2e-4 | 7.4e-7 | same argmax |
| guided image-token logits, all 576 steps teacher-forced | 7.0e-4 | 9.9e-6 | argmax identical at every step |
| greedy generation (`--temp 0`) | | | all 576 image tokens identical |
| VQ decoder, same 576 tokens (Metal, half-precision matrix units) | 1.2e-3 | 6.8e-5 | 94.5 dB PSNR |
| SigLIP features, 1200x799 image (padding path) | 3.0e-3 | 1.9e-5 | |
| aligned image embeddings | 1.6e-3 | 1.6e-5 | |
| logits after the 628-token image prompt | 2.7e-4 | 4.2e-6 | 64 greedy answer tokens identical |

Janus-Pro-7B: text logits max diff 8.4e-5 (rel 5.5e-6), guided logits over 64 teacher-forced
steps 1.1e-3 (rel 1.5e-5), the first 64 greedy image tokens identical (the length of the reference
run). With Metal's default half-precision matrix units the 1B text logits differ by up to 0.03
(rel 5.6e-4), still with the same argmax; generation then runs on the CPU except the prompt.

One 384x384 image on the M5 Max (18 cores; 576 steps of 2 sequences, then the VQ decoder):

| | 1B CPU | 1B Metal | 7B CPU | 7B Metal |
|---|---|---|---|---|
| bf16 | 9.8 s (15.0 ms/step) | 9.4 s | 40.5 s (67.8 ms/step) | 36.1 s |
| `--quant q8` | 6.0 s (8.5 ms/step) | | 19.4 s (31.2 ms/step) | |
| `--quant q4` | 5.0 s (6.5 ms/step) | | 14.1 s (22.0 ms/step) | |
| 4 images in one batch | 25.1 s bf16, 15 s q8 | 23.2 s bf16 | 43.2 s q8 | |

Decoding steps are memory-bound (about 190 GB/s of weights), so the batch of two CFG sequences
costs the same as one, and several images share the weight reads: four images take 2.2-2.6x the
time of one. Metal runs the prompt and the VQ decoder (1.0 s CPU, 0.3 s Metal per image); the
per-step matmuls of a few rows stay on the CPU, where they are faster than one GPU submission
each.

**JanusFlow** (`deepseek-ai/JanusFlow-1.3B`, recognised from its config; `src/janus/flow.rs`)
uses the same entry points:

```sh
./target/release/oxinfer --model deepseek-ai/JanusFlow-1.3B --generate-image "a red fox in the snow" -o fox.png
./target/release/oxinfer --model deepseek-ai/JanusFlow-1.3B -p "a watercolor lighthouse" -o l.png \
    --steps 30 --cfg 2 --seed 7 --num-images 2 --size 768x768
./target/release/oxinfer --model deepseek-ai/JanusFlow-1.3B --image chart.png -p "Which slice is the largest?" --chat
```

It generates by rectified flow in the latent space of the SDXL VAE, with the language model as
the velocity network. Each of the `--steps` Euler steps (default 30, t from 0 to 1) embeds the
48x48x4 latent with the `ShallowUViTEncoder` (sinusoidal time embedding through two linears, a
2x2 patch convolution, two ConvNeXt blocks with global response normalization modulated by the
time embedding) into 576 tokens, runs the LLaMA decoder over the prompt, the time embedding (in
place of the prompt's final `<｜begin▁of▁generation｜>`) and the tokens, and turns its hidden
states at the tokens back into a velocity with the `ShallowUViTDecoder` (RMSNorm and linear to
768 channels, concatenation with the encoder's tokens, the long skip connection, two more ConvNeXt
blocks, 1x1 convolution and pixel shuffle). Guidance is `cfg * v_cond - (cfg - 1) * v_uncond`
(default 2, the reference demo's), the unconditional sequence attending to BOS only as in the
reference's attention mask; both sequences of every image run as one batch, and the prompt's keys
and values are computed once and kept across the steps (`--cfg 1` runs the conditional sequence
alone). The latents are decoded by the Stable Diffusion VAE decoder (`src/sd/vae.rs`) with the
weights of `stabilityai/sdxl-vae`, fetched on first use (`JanusOptions::vae`, the C option `vae=`:
another folder or repo; a `vae/` folder inside the checkpoint is used when present), kept in full
f32 precision: rounded to bf16 they move the image by 0.6%. For understanding, JanusFlow has the
same SigLIP-L/16 tower with a single linear aligner and a learned embedding at `<image_beg>`,
and the older `User:` / `Assistant:` template; all of it is read from the checkpoint's configs.

Agreement with the official `janus.janusflow` code and diffusers' `AutoencoderKL` in float32 on
CPU (`tools/ref_janus.py` vs `examples/janus_check.rs`, oxinfer on the CPU with bf16 weights as
stored; generation from the same starting noise, 30 steps, CFG 2):

| JanusFlow-1.3B | max abs diff | rel L2 | |
|---|---|---|---|
| velocity of the first step (conditional and unconditional) | 9.1e-6 | 1.3e-6 | |
| velocities of all 30 steps | 3.0e-3 | 2.3e-5 | last step 2.2e-3, rel 6.9e-5 |
| final latents | 4.6e-4 | 1.1e-5 | |
| VAE output (384x384x3, in [-1, 1]) | 1.4e-3 | 4.0e-5 | PNG: 78.3 dB, max pixel diff 1, 428 of 442368 bytes differ |
| text logits after a 51-token prompt | 4.9e-5 | 2.6e-6 | same argmax |
| SigLIP features, 1280x960 image | 8.3e-3 | 2.4e-5 | |
| aligned image embeddings | 3.8e-3 | 2.2e-5 | |
| logits after the 630-token image prompt | 7.8e-5 | 3.9e-6 | greedy answer identical |
| the same for a 150x260 image (upscaled, padded left and right) | 5.3e-5 | 2.8e-6 | 64 greedy answer tokens identical |

**Image sizes.** Generation is 384x384 for every model of the family, a limit of the models
rather than of oxinfer: Janus-Pro was trained to write a 24x24 grid of image tokens and JanusFlow
a 48x48 latent (24x24 tokens), in raster order with one-dimensional positions, and the
reference generates nothing else. `--size WxH` (`ImageOptions::width` / `height`, the API's
`size`) therefore resamples the 384x384 image with a Lanczos filter, as the reference's demos do
for display; it adds no detail. For understanding, images of any size and aspect ratio go
through the reference processor's steps: longer side resized to 384 (antialiased bicubic, up or
down), centered on a square of the processor's background colour.

**Conversations.** `--chat` continues after the first answer with one question per line of
standard input, the images belonging to the first question; in the library `Janus::ask` takes a
`Conversation` and each turn may add images, and the C API has `oxi_janus_chat` /
`oxi_janus_chat_reset`. The turns follow the reference's template (`... Assistant: ANSWER<eos>User:
QUESTION\n\nAssistant:`). The language model's state is kept between turns; the previous answer is
read again as text, because the reference formats the whole conversation anew at every turn and
its tokens are not always the ones the model wrote (the first word loses or gains its leading
space). Checked on a second question after an image question and its greedy answer, against the
reference run on the whole two-turn conversation: 664 identical token ids for both models,
logits max diff 9.7e-5 (rel 4.7e-6) for JanusFlow-1.3B and 1.4e-4 (rel 3.0e-6) for Janus-Pro-1B.
Janus-Pro-1B checked again after these changes: guided logits over 48 teacher-forced steps
3.1e-4 (rel 6.4e-6), logits after a 150x260 image 1.2e-4 (rel 1.3e-6), same greedy answer.

Not supported: image generation at a native size other than 384x384 (a limit of the models, see
above) and the browser build. JanusFlow was checked on the CPU only; its ConvNeXt blocks always
run there.

## Image, video and audio input

```sh
./target/release/oxinfer --model models/gemma-4-E4B-it --quant q4 \
    --image photo.jpg --audio clip.wav --prompt "Describe the image and transcribe the audio."
./target/release/oxinfer --model Qwen/Qwen3.5-0.8B --video clip.mp4 --prompt "What happens in this video?"
./target/release/oxinfer --model models/Qwen3.8-27B --quant q8 --image a.png --image b.png -p "Compare them."
```

`--image`, `--video` and `--audio` can be repeated; attachments are placed before the text of the
user turn (or of the first turn in `--chat`): images, then videos, then audio. Gemma 4 takes all
three; Qwen3.5 / 3.6 / 3.8 checkpoints that include the vision tower (`model.visual.*`, as the
official ones do) take images and videos. Files are decoded with the pure-Rust
[oxideav](https://crates.io/crates/oxideav-core) crates: PNG, JPEG, WebP, GIF, BMP; WAV, MP3,
FLAC, AAC/M4A, Ogg Vorbis/Opus, MKV audio (resampled to 16 kHz mono); and video (below).

- **Gemma 4 vision**: aspect-preserving bicubic resize to at most 280 soft tokens
  (`--max-soft-tokens` accepts 70/140/280/560/1120), 16x16 patches, a 16-layer ViT (hidden 768)
  with axial 2D RoPE and clipped linears, 3x3 patch pooling, then the vision embedder. A 640x480
  image becomes 57x42 patches and 266 soft tokens.
- **Gemma 4 video** (as `Gemma4VideoProcessor`): 32 frames spread uniformly over the clip, each
  through the image tower at 70 soft tokens (frames are batched through the tower), written as
  `mm:ss <|image>` + soft tokens + `<image|>` per frame.
- **Gemma 4 audio**: 128-bin log-mel features (20 ms frames, 10 ms hop), two stride-2 conv layers,
  a 12-layer conformer with chunked local attention and relative position bias, then the audio
  embedder: 40 ms of audio per soft token (750 max, i.e. 30 s).
- **Qwen vision** (`qwen_vision.rs`, as `Qwen2VLImageProcessorFast` and `Qwen3_5VisionModel`):
  each side resized to a multiple of 32 pixels within the checkpoint's pixel budget (65536 to
  16.7M pixels; `--max-soft-tokens N` caps it at N tokens of 32x32 pixels), 16x16 patches with
  two temporal frames each, emitted in 2x2 merge blocks; a Conv3d patch embedding, learned
  position embeddings bilinearly resampled to the patch grid, axial 2D RoPE, 12 (0.8B) or 27
  (9B, 27B) pre-LayerNorm blocks with bidirectional attention, and a merger folding each 2x2
  block into one token. A 1024x1024 image is 4096 patches and 1024 tokens. The text model places
  the tokens with interleaved M-RoPE (time / height / width positions per rotary frequency; the
  text after an image continues from its largest side). The tower stays bf16 with `--quant`
  (int8 blocks moved the 27-layer tower's outputs by 4%), and `--save-quantized` keeps it.
- **Qwen video** (as `Qwen3VLVideoProcessor`): 2 frames per second (`--video-fps`), at least 4
  and at most 768 (`--video-frames`), resized together within a 25M-pixel budget for the clip;
  every two frames form one temporal patch, prefixed with its time (`<1.2 seconds>`) in the
  prompt.

Soft tokens replace the placeholder embeddings inside the text model (for Gemma, the
per-layer-embedding lookup uses the pad token there, as in the reference). All attention stays
causal, matching the configurations. The resize reproduces torchvision's uint8 antialiased
bicubic exactly (what the reference processors run by default; `resample.rs` also has Pillow's,
the PIL backend's), so the pixels fed to the towers are bit-identical to transformers'.

**Video files.** oxideav demuxes MP4, MOV, Matroska / WebM and AVI and decodes H.264 (Baseline
to High, CABAC, B-frames), H.265, AV1, VP8, VP9, MPEG-4 Part 2 and Motion JPEG, all in pure Rust;
every one of these was checked on ffmpeg-made clips (first frames within one level of
ffmpeg's, three for MJPEG). A directory of image files (sorted by name) is read as a video at
`--frames-fps` (default 24, the rate the processors assume). The decoder runs on one thread: a
10-s 1080p clip decodes in 8.4 s (H.264 High), 6.0 s (VP9) or 9.9 s (AV1), against 0.1 s for
ffmpeg, and frames up to the last sampled one are all decoded. Not available: other codecs
(ProRes, Theora, ...), the audio track of a video, and hardware decoding. The same attachments
go through the library (`Engine::send_with_media` with `Attachments`) and the C API
(`oxi_engine_send_media` with a JSON list); frames already decoded by oxideav go through the
library too (see [Using oxinfer from oxideav](#using-oxinfer-from-oxideav-and-other-crates)).

Agreement with transformers 5.20 (float32, CPU; `tools/ref_vl.py` runs the reference processor
and model on the same files and compares with oxinfer's `--dump-mm` / `--dump-logits`):

| Model, input | Tower output vs reference | Last-token logits | 40 greedy tokens |
|---|---|---|---|
| Qwen3.5-0.8B, 500x333 image (160 tokens) | max 2.5e-4, rel 2.4e-5 | max diff 1.9e-4 | identical |
| Qwen3.5-0.8B, 6-s 480x320 video (12 frames, 900 tokens) | max 8.9e-4, rel 2.3e-5 | max diff 2.8e-4 | identical |
| Qwen3.5-0.8B, 500x332 video (8 frames resized to 512x320) | max 6.1e-4, rel 2.8e-5 | max diff 2.2e-4 | identical |
| Qwen3.5-0.8B, 2 images + a video (1390 tokens, 9 M-RoPE spans) | | max diff 2.2e-4 | |
| Qwen3.5-9B tower (the 27-layer tower of 3.8-27B), CPU | max 6.0e-4, rel 1.3e-5 | | |
| same, Metal with the matrix units (half-precision inputs) | max 0.17, rel 2.4e-3 | | |
| Gemma 4 E2B, 500x333 image (260 soft tokens) | max 4.2e-5, rel 3.6e-6 | max diff 1.4e-4 | |
| Gemma 4 E2B, 6-s video (32 frames, 1920 soft tokens) | max 6.7e-5, rel 3.2e-6 | max diff 1.2e-4 | identical |

The pixels match bit for bit in every case, and the expanded prompts are identical. On
Qwen3.8-27B (`--quant q8`), the same video gives: "The video begins with a serene winter scene
featuring a red fox perched on a snow-covered tree branch. [...] As the video progresses, there is
a sudden transition to a completely different scene. The new scene is a watercolor painting of a
lighthouse situated on a rocky outcrop by the sea."

Tower time (M5 Max, 18 threads; load average 4-15 from other work on the machine; the Metal
column has the linear layers and attention on the GPU):

| Tower, input | CPU | Metal |
|---|---|---|
| Qwen3.5-0.8B, 1024x1024 image (4096 patches) | 1.9 s | 0.46 s |
| Qwen3.5-9B / Qwen3.8-27B (27 layers, hidden 1152), 1024x1024 image | 6.5 s | 1.6 s |
| Gemma 4 E2B, 1024x1024 image at 1120 soft tokens (9801 patches) | 12.4 s | 2.3 s |
| Gemma 4 E2B, 32 video frames of 480x320 (1920 soft tokens) | 8.0 s | 2.5 s |

### Qwen2.5-VL

Qwen/Qwen2.5-VL-3B-Instruct, -7B-Instruct (and the other Qwen2.5-VL / Qwen2-VL-style checkpoints,
`model_type` `qwen2_5_vl`) run through the normal engine, with images:

```sh
./target/release/oxinfer --model Qwen/Qwen2.5-VL-3B-Instruct --image fox.png -p "Describe this image."
```

The language model is the Qwen2 transformer (biased query / key / value projections, no query /
key norms, contiguous M-RoPE sections 16 / 24 / 24 over time, height and width) on the Qwen code
path; the vision tower (`src/qwen25_vision.rs`) has 14-pixel patches, axial 2D RoPE, RMSNorm
blocks with a SwiGLU MLP, window attention over 8x8-patch windows in all blocks but
`fullatt_block_indexes` (whole-image attention), and the 2x2 patch merger. The chat format is the
checkpoint's (default system prompt "You are a helpful assistant."). Against transformers 5.20 in
float32 on the 3B model with a 512x512 image (324 image tokens): pixels identical, image features
1.0e-5 relative, last-position logits max diff 1.1e-4 (KL 1.3e-9) on the CPU; on Metal (half-
precision matrix units) 1.9e-3 and KL 1.5e-7; greedy answers identical. Video input and single-
token GPU decoding (biased projections) are not supported for these checkpoints yet (decoding runs
on the CPU: 12 tok/s for the 3B in bf16).

`oxinfer::qwen_vl::QwenVl` is the same model as a conditioning encoder for diffusion pipelines
(Qwen-Image uses it; Step1X-Edit and OmniGen2 use Qwen2.5-VL-7B / 3B the same way):
`QwenVl::load(dir, &QwenVlOptions { quant, mmap, tokenizer_dir, processor_dir, vision })`,
`encode_image(rgb, w, h, max_pixels) -> VlImage` (embeddings, merged grid, `placeholder()` text),
and `hidden_states(text, &images, &layers) -> Hidden` with the final-normed states (transformers'
`hidden_states[-1]`) and the residual stream after any requested layers (`hidden_states[l + 1]`),
M-RoPE positions for the images included. `examples/qwen_vl_hidden.rs` dumps them.

## Speech to text (Whisper)

OpenAI's Whisper checkpoints in the Hugging Face layout (`model_type: whisper`) transcribe or
translate audio of any length (`src/whisper/`): openai/whisper-{tiny, base, small, medium,
large-v3, large-v3-turbo}, the English-only `.en` models, and distil-whisper (the same network
with fewer decoder layers, e.g. distil-whisper/distil-large-v3). Sizes come from `config.json`
(80 mel bins, 128 for v3), special tokens and decoding defaults from `generation_config.json`.

```sh
./target/release/oxinfer --model openai/whisper-small --transcribe talk.mp3              # text
./target/release/oxinfer --model openai/whisper-large-v3-turbo --transcribe talk.mp3 --timestamps
./target/release/oxinfer --model openai/whisper-small --transcribe interview.m4a --language fr --translate
./target/release/oxinfer --model openai/whisper-small --transcribe talk.wav --format srt > talk.srt
./target/release/oxinfer --model openai/whisper-small --transcribe talk.wav --word-timestamps --json
./target/release/oxinfer --model openai/whisper-small --transcribe talk.wav --beam 5       # beam search
./target/release/oxinfer --model openai/whisper-small --transcribe a.mp3 --transcribe b.mp3 --transcribe c.wav  # a batch
```

| Flag | Meaning |
|---|---|
| `--transcribe FILE` | audio to transcribe: WAV, MP3, FLAC, AAC/M4A, Ogg Vorbis/Opus, MKV audio (oxideav), resampled to 16 kHz mono. Repeat it for several files: they are decoded together as a batch, with the result each file gets alone |
| `--batch N` | files decoded together at most (default 8); the others take the places of those that finish |
| `--language XX` | spoken language, as a code or a name (default: detected from the first 30 s) |
| `--translate` | translate to English text |
| `--timestamps` | timed segments (always on past 30 s) |
| `--word-timestamps` | also word-level times |
| `--format F` | `text` (default), `json`, `srt`, `vtt` (`--json` is `--format json`) |
| `--initial-prompt TEXT`, `--prefix TEXT` | context for the decoder (names, spelling, style); forced start of the transcript |
| `--fallback` | openai-whisper's temperature fallback: temperatures 0, 0.2, ..., 1.0 while a window's tokens compress better than 2.4 or average below -1 log-probability; a window with a no-speech probability over 0.6 and a low log-probability is skipped as silence |
| `--condition-on-prev` | feed each 30 s window the previous text |
| `--beam N` | beam search with N beams (openai-whisper's `beam_size`) at temperature 0; with `--fallback`, the higher temperatures sample |
| `--patience P` | beam search ends once `round(N * P)` hypotheses have ended (default 1) |
| `--length-penalty A` | rank the ended hypotheses by log-probability over `((5 + length) / 6)^A`, A from 0 to 1 (default: over the length) |
| `--best-of N` | at temperatures above 0 (`--temp T`, `--fallback`), sample N sequences and keep the best ranked |
| `--precision f32\|f16\|bf16` | weight storage (default bf16); `--quant q8\|q4` quantizes the decoder |

What runs, following transformers' `WhisperFeatureExtractor` and
`WhisperForConditionalGeneration.generate`:

- **Features**: 400-sample periodic Hann window, hop 160, reflect padding, power spectrum,
  Slaney mel filterbank, `log10`, clamp to `max - 8`, `(x + 4) / 4`. Clips up to 30 s are
  zero-padded to 30 s first; longer audio is featurized whole (the clamp sees all of it). The DFT
  (a mixed-radix FFT of 400) and the filterbank run in f64.
- **Encoder**: two convolutions (kernel 3, the second with stride 2) with GELU, sinusoidal
  positions, pre-LayerNorm transformer layers, all through the shared matmul (SIMD kernels, GPU
  backends for the big products) and a fused attention kernel on Metal.
- **Decoder**: learned positions, causal self-attention over a KV cache, cross-attention over the
  window's encoder output (keys and values computed once per window, kept head-major and attended
  with the keys split across threads), the output projection tied to the token embedding.
- **Decoding**: greedy by default (`TranscribeOptions::temperatures` for sampling and fallback);
  language detection from the first-token distribution over the language tokens (`language` then
  holds the code and its probability); `suppress_tokens` and `begin_suppress_tokens`; the rules of
  `WhisperTimeStampLogitsProcessor` (no `<|notimestamps|>`, timestamps in pairs and increasing, a
  timestamp first, at most 1 s in, and a timestamp whenever all timestamps together outweigh every
  text token). Audio over 30 s is decoded in sequential 30 s windows, each moved to the last
  complete segment's end timestamp, as transformers' long-form generation does; `initial_prompt`
  becomes `<|startofprev|>` context; `condition_on_prev_tokens` feeds the last 223 tokens.
- **Beam search** (`beam_size`, `patience`, `length_penalty`; `src/whisper/decode.rs`), as
  openai-whisper's `BeamSearchDecoder` and `MaximumLikelihoodRanker`: every beam proposes its
  `beam_size + 1` most likely tokens after the logit rules above (applied per beam); the
  candidates are walked by decreasing cumulative log-probability until `beam_size` of them that
  go on are kept, those ending in `<|endoftext|>` met on the way are set aside, and the search
  ends once `round(beam_size * patience)` hypotheses have ended or after 224 tokens. The best is
  the one with the highest log-probability over its length (or over `((5 + length) / 6)^a` with
  a length penalty). Each beam has its own self-attention cache: a beam continued once hands its
  cache on, a beam continued several times is copied. Beam search runs at temperature 0; the
  fallback temperatures above it sample, `best_of` sequences each, ranked the same way.
- **Batches**: several inputs are transcribed together (`transcribe_batch`): the current 30 s
  window of each goes through the encoder in one pass (the windows stacked in every matmul, one
  GPU submission per product for all of them; attention per window) and every hypothesis of
  every input is one row of each decoder step (rows share the matmuls seven at a time; attention
  per row over its own caches). Inputs that need another temperature retry together, an input
  that ends leaves its place to the next. The windows of one input stay sequential, since each
  starts at the previous one's last timestamp. A row's values do not depend on the rows computed
  with it (the CPU kernels compute up to seven rows one at a time, and the encoder is stacked on
  a GPU only when the backend takes the same products for one window as for the stack), so each
  transcript is the one the input gets alone: same tokens, times and probabilities, bit for bit.
- **Word timestamps**: openai-whisper's alignment: the cross-attention of the checkpoint's
  `alignment_heads` over the text tokens, normalized and median-filtered, aligned to the audio
  frames by dynamic time warping, tokens grouped into words, punctuation merged. Against
  transformers' pipeline (`return_timestamps="word"`) on a 29 s LibriSpeech clip, the 60 shared
  words start within 16 ms of it on average.

In Rust, `oxinfer::whisper::Transcriber` does all of it (`examples/transcribe.rs`):

```rust
use oxinfer::whisper::{Transcriber, TranscriberOptions, TranscribeOptions};
let asr = Transcriber::load("openai/whisper-small", &TranscriberOptions::default())?;
let mut opts = TranscribeOptions::default();
opts.language = Some("fr".into());
opts.timestamps = true;
let pcm = Transcriber::load_audio(std::path::Path::new("talk.mp3"))?; // or your own 16 kHz samples
let t = asr.transcribe_with(&pcm, &opts, &mut |seg| {
    println!("[{:.2} -> {:.2}] {}", seg.start, seg.end, seg.text); // as each 30 s window finishes
    oxinfer::engine::Flow::Continue                               // Flow::Stop ends it there
})?;
println!("{} ({}, {:.1}x real time)", t.text, t.language, 1.0 / t.stats.rtf);
```

Beam search and batches are options and one more call:

```rust
opts.beam_size = Some(5);          // openai-whisper's beam_size; patience, length_penalty, best_of
opts.batch_size = 8;               // inputs decoded together
let clips: Vec<Vec<f32>> = load_clips();
let refs: Vec<&[f32]> = clips.iter().map(|c| &c[..]).collect();
let transcripts = asr.transcribe_batch(&refs, &opts)?;   // one Transcript per input, in order
// transcribe_batch_with(&refs, &opts, &mut |input, seg| ...) reports each segment with its input;
// transcribe_files(&paths, &opts) decodes the files first.
```

`asr.cancel_handle()` stops a transcription from another thread (it returns the finished
segments with `stopped` set); `transcribe_file` decodes a file first; `detect_language` returns
the language distribution. `TranscribeOptions` deserializes from JSON, which is how the C API
takes it: `oxi_transcriber_load`, `oxi_transcribe_file` / `oxi_transcribe_pcm` (any sample rate)
with a per-segment callback, `oxi_transcriber_cancel`, results as JSON (`include/oxinfer.h`);
`beam_size`, `patience`, `length_penalty` and `best_of` are keys of that JSON (the C API
transcribes one input per call). The server's `/v1/audio/transcriptions` takes the same four as
form fields, and several `file` parts in one request: they are transcribed as a batch and
answered with a JSON array of the per-file results, in order (`response_format` `json` or
`verbose_json`). With several `--transcribe` files the CLI prints each file's text under a
`== file ==` line (or a JSON array with a `file` key each), and writes subtitles next to each
input (`talk.mp3` gives `talk.srt`).

**Agreement with transformers** (float32 `WhisperForConditionalGeneration` + `WhisperProcessor`,
torch 2.14, transformers 5.20; `tools/ref_whisper.py` on the samples oxinfer decoded; oxinfer with
`--precision f32 --device cpu`). Clips: LibriSpeech test samples (5.9 s; 29.4 s with timestamps),
25 s of French from a LibriVox recording (language detection), and the first 100 s of a LibriVox
reading of Woolf's *Kew Gardens* (long-form: four or five windows):

| Model | Clip | Log-mel max abs diff | Encoder max abs diff (rel L2) | First-step logits max abs diff | Greedy transcript |
|---|---|---|---|---|---|
| whisper-tiny | 5.9 s | 1.2e-5 | 5.4e-4 (6.7e-6) | 1.2e-4 | identical, 21 tokens |
| whisper-tiny | 29.4 s, timestamps | 1.6e-5 | 6.3e-3 (1.3e-5) | 9.3e-5 | identical, 99 tokens with timestamps, same 5 segment times |
| whisper-tiny | 25 s French | 3.1e-5 | 1.6e-3 (6.8e-6) | 1.5e-4 | identical, 77 tokens; `fr` detected |
| whisper-tiny | 100 s long-form | 3.0e-5 | 4.7e-3 (1.2e-5) | 1.2e-4 | identical, 348 tokens, same 21 segment times |
| whisper-small | 5.9 s | 1.2e-5 | 3.1e-3 (1.1e-5) | 3.3e-5 | identical, 22 tokens |
| whisper-small | 29.4 s, timestamps | 1.6e-5 | 2.4e-3 (5.8e-6) | 2.6e-5 | identical, 110 tokens, same 7 segment times |
| whisper-small | 25 s French | 3.1e-5 | 1.8e-3 (5.5e-6) | 3.0e-5 | identical; `fr` detected; translated: identical, 60 tokens |
| whisper-small | 100 s long-form | 3.0e-5 | 5.2e-3 (1.3e-5) | 2.7e-5 | identical, 285 tokens, same 15 segment times |
| whisper-large-v3-turbo | 100 s long-form | 2.7e-5 | 2.5e-2 (1.4e-4) | 2.5e-4 | identical, 300 tokens, same 16 segment times |
| whisper-large-v3-turbo | 25 s French | 3.2e-5 | 1.2e-2 (9.3e-5) | 5.8e-5 | identical, 29 tokens |
| distil-large-v3 | 100 s long-form | 2.7e-5 | 3.8e-2 (1.4e-4) | 7.2e-5 | identical, 350 tokens, same 21 segment times |
| whisper-base | 100 s long-form | 3.0e-5 | 2.9e-3 (7.2e-6) | 4.3e-5 | identical, 361 tokens, same 22 segment times |
| whisper-medium | 29.4 s, timestamps | 1.6e-5 | 4.7e-3 (8.7e-6) | 4.1e-5 | identical, 102 tokens, same 6 segment times |
| whisper-large-v3 | 25 s French | 3.2e-5 | 1.1e-3 (7.1e-6) | 2.7e-5 | identical, 25 tokens; `fr` detected |
| whisper-tiny.en | 100 s long-form | 3.0e-5 | 7.7e-5 (3.3e-6) | 7.7e-5 | identical, 354 tokens, same 24 segment times |

The options agree too (whisper-small): an initial prompt (88 tokens), `condition_on_prev_tokens`
over the 100 s clip (213 tokens, 11 segments), both together, and the fallback settings (no window
falls back on that clip). The log-mel difference is torch's own: its f32 STFT is off from an f64
evaluation of the same pipeline by up to 1.2e-5, where oxinfer's features are within 7e-8 (f32
rounding), so bit-identical features would mean reproducing torch's FFT rounding. Encoder
differences grow with depth (32 layers for large-v3-turbo) but stay at a relative 1e-4.

**Beam search against openai-whisper** (openai-whisper 20250625, torch 2.15 on the CPU, float32,
the `tiny` and `base` checkpoints; `tools/ref_whisper_beam.py`; oxinfer with `--precision f32
--device cpu` on the Hugging Face copies of the same weights, language given as English). Twelve
LibriSpeech clips of 5.9 to 29.4 s (hf-internal-testing/librispeech_asr_dummy), each decoded by
`whisper.decode` with the same options; the tokens compared include the timestamp tokens where
timestamps are on:

| Model | `beam_size` | `patience` | `length_penalty` | Timestamps | Clips with identical tokens | Max abs diff of the average log-probability |
|---|---|---|---|---|---|---|
| whisper-tiny | 5 | | | no | 12 of 12 (492 tokens) | 2.4e-4 |
| whisper-tiny | 5 | | | yes | 12 of 12 (527 tokens) | 2.3e-4 |
| whisper-tiny | 2 | | | no | 12 of 12 (497 tokens) | 2.4e-4 |
| whisper-tiny | 8 | 2 | | yes | 12 of 12 (524 tokens) | 2.4e-4 |
| whisper-tiny | 5 | | 1.0 | yes | 12 of 12 (527 tokens) | 2.3e-4 |
| whisper-tiny | 4 | 1.5 | 0.6 | no | 12 of 12 (493 tokens) | 2.4e-4 |
| whisper-base | 5 | | | no | 12 of 12 (493 tokens) | 2.2e-4 |
| whisper-base | 5 | | | yes | 12 of 12 (547 tokens) | 2.5e-4 |
| whisper-base | 3 | 2 | 0.8 | yes | 12 of 12 (548 tokens) | 2.5e-4 |

The no-speech probabilities are within 5e-7 of the reference. One 29.4 s clip ends its first
window on a timestamp pair at 21.72 s (whisper-tiny, with timestamps): `whisper.decode` stops there, oxinfer
decodes the rest in a second window, and the first window is what the table compares. Long-form,
against whisper's `transcribe()` with `beam_size=5`, `temperature=0` and
`condition_on_previous_text=False`, on two clips of 90.6 and 75.6 s made of those samples end to
end: the text tokens and the segment times are identical for both clips with both models (tiny:
503 text tokens, 16 and 12 segments; base: 496 text tokens, 15 and 12 segments). Only token-level
bookkeeping differs there: oxinfer keeps both timestamps of the pair that closes a window in the
segment's `tokens`, as transformers does, and openai-whisper keeps one. Not compared: sampled
output (`best_of`, the fallback temperatures), which depends on the random number generator.

**Batches against single inputs**. The same 14 inputs (the twelve clips and the two long ones,
19 to 21 windows in all) as one batch and each alone, comparing every segment's tokens, times,
average log-probability, compression ratio and no-speech probability, the words and the detected
language as printed in the JSON output: identical for all 14 with whisper-tiny greedy (bf16, CPU),
whisper-tiny with `--beam 5 --timestamps --batch 5` (bf16, CPU), whisper-base with `--beam 5
--best-of 3 --fallback` and word timestamps (f32, CPU), and whisper-small with `--beam 3` on CUDA
(bf16, RTX 5080: the encoder windows stacked on the GPU). The unit tests check the same on a tiny
random-weight model without downloaded weights (`src/whisper/batch_tests.rs`: batched encoder and
decoder steps bit for bit in f32, bf16, q8 and q4, whole transcripts with beam search, sampling,
fallback and word timestamps, and the beam search against a slow transcription of
openai-whisper's Python). Not run: batches on Metal and WebGPU (no such device here).

With the default bf16 weights, transcripts stay close but are not always token-identical. On the
100 s clip, whisper-large-v3-turbo in bf16 on Metal gives the f32 transcript token for token;
whisper-small differs in one word of 206, because of the bf16 decoder (bf16 encoder with an f32
decoder, or f16 weights throughout, are token-identical). A q8 decoder: 0 or 1 word of 206 on
whisper-small, 1 of 210 on large-v3-turbo; q4 decoders drift (7 words of 210 on large-v3-turbo,
50 of 206 on whisper-small): not recommended for Whisper.

**Speed** on the M5 Max, 100 s of speech (long-form), wall time including features, language
detection and every window (real-time factor = time / audio duration; lower is faster). Best of
three runs, load average 4-8 from other jobs (one core busy with a compile during the first
rows); `--threads 6` approximates a phone's core count:

| Model | Device | Encoder | Decoder | Total | RTF | Speed |
|---|---|---|---|---|---|---|
| whisper-small | CPU, 18 threads | 2.17 s | 0.66 s (299 tokens) | 2.84 s | 0.028 | 35x real time |
| whisper-small | CPU, 6 threads | 4.58 s | 0.82 s | 5.43 s | 0.054 | 18x |
| whisper-small | Metal | 0.66 s | 0.65 s | 1.32 s | 0.013 | 76x |
| whisper-small, q8 decoder | Metal | 0.54 s | 0.54 s | 1.09 s | 0.011 | 92x |
| whisper-large-v3-turbo | CPU, 18 threads | 9.66 s | 0.57 s (304 tokens) | 10.24 s | 0.102 | 10x |
| whisper-large-v3-turbo | CPU, 6 threads | 19.58 s | 0.70 s | 20.32 s | 0.203 | 5x |
| whisper-large-v3-turbo | Metal | 1.39 s | 0.58 s | 1.99 s | 0.020 | 50x |
| whisper-large-v3-turbo, q8 decoder | Metal | 1.34 s | 0.39 s | 1.76 s | 0.018 | 57x |

On Metal the encoder's attention is one fused kernel (`flash_attn64`: about 3.6 ms per
large-v3-turbo layer, against 10 ms for the GEMM + softmax + GEMM path it replaced, which other
head sizes still use); a large-v3-turbo window then splits into ~160 ms of matmuls, ~115 ms of
attention and ~100 ms of CPU-side layer norms, GELU and residual adds.

The encoder (30 s per window, 1500 positions) is the heavy part: its matmuls and attention go to
the GPU when one is active (Metal, CUDA, WebGPU); decoding is one token at a time on the CPU
(one row per beam and per input of a batch),
where the bf16 / q8 / q4 kernels of the language models apply. Without a GPU (iOS and Android
builds without one, `--device cpu`), everything runs on the SIMD CPU kernels. f32 and f16 weights
run on the CPU only (the GPU backends take bf16, q8 and q4).

Not supported: batching the windows of one long file (they are sequential: each starts at the
previous one's last timestamp), transformers' own beam search (`num_beams`, a different
algorithm from openai-whisper's, which is the one implemented),
voice activity detection or speaker diarization, openai-whisper's word-timing heuristics
(the punctuation merge is there, not the long-word truncation) and `hallucination_silence_threshold`;
the compression ratio of the fallback is computed with a built-in DEFLATE size estimate (greedy
LZ77, fixed Huffman codes) instead of zlib, as no deflate implementation is in the dependency tree.

## Text to speech (Orpheus)

[Orpheus TTS](https://github.com/canopyai/Orpheus-TTS) is a Llama 3.2 3B fine-tune that writes
audio as extra vocabulary tokens: 7 tokens per frame, codes of the three codebooks of the
[SNAC](https://github.com/hubertsiuzdak/snac) codec, which decodes them to 24 kHz audio (a frame is
2048 samples, 85 ms). `canopylabs/orpheus-3b-0.1-ft` is gated (accept the licence and set
`HF_TOKEN`); `unsloth/orpheus-3b-0.1-ft` is an ungated bf16 copy of it. The SNAC decoder
(`hubertsiuzdak/snac_24khz`, 80 MB, `pytorch_model.bin`) is fetched on first use.

```sh
./target/release/oxinfer --model unsloth/orpheus-3b-0.1-ft --speak "Hey there, I'm Leo. <chuckle> Nice to meet you." --voice leo -o hi.wav
./target/release/oxinfer --model unsloth/orpheus-3b-0.1-ft --speak - --quant q4 --speak-stream < text.txt   # faster than real time
./target/release/oxinfer --model hubertsiuzdak/snac_24khz --decode-tokens tokens.json -o out.wav          # SNAC alone
# a batch: one utterance per line (or several --speak), generated together -> out-1.wav, out-2.wav, ...
./target/release/oxinfer --model unsloth/orpheus-3b-0.1-ft --quant q4 --speak-file lines.txt --voice tara,leo,mia -o out.wav
# a long text: its sentences as one batch, joined in one file
./target/release/oxinfer --model unsloth/orpheus-3b-0.1-ft --quant q4 --speak - --speak-split -o story.wav < story.txt
# voice cloning: the pretrained model, a reference clip and what it says (repeatable)
./target/release/oxinfer --model unsloth/orpheus-3b-0.1-pretrained --ref-audio me.wav --ref-text "What I say in the clip." --speak "Something new." -o clone.wav
# other languages
./target/release/oxinfer --model Our-Org-2/3b-fr-ft-research_release --speak "Bonjour, comment allez-vous ?" --voice pierre -o fr.wav
# the codec alone, any SNAC model: audio -> codes (JSON array per codebook) -> audio
./target/release/oxinfer --model hubertsiuzdak/snac_44khz --snac-encode music.wav -o codes.json
./target/release/oxinfer --model hubertsiuzdak/snac_44khz --decode-codes codes.json -o back.wav
```

Voices of the English model: tara (default), leah, jess, leo, dan, mia, zac, zoe; emotion tags
(`<laugh>`, `<chuckle>`, `<sigh>`, `<cough>`, `<sniffle>`, `<groan>`, `<yawn>`, `<gasp>`) go in the
text. Sampling follows the official package: temperature 0.6, top-p 0.8, repetition penalty 1.1
(over the prompt and the reply, as transformers and vLLM apply it), at most 1200 tokens (about
14 s; `--max-tokens` for more). The prompt is the one `orpheus_tts` builds (start of human,
`<|begin_of_text|>voice: text`, end of text / human, start of AI, start of speech) and generation
stops at end of speech. By default sampling is constrained to the tokens valid at each frame
position (plus the stop tokens), which also computes only those 4096 rows of the 157k-row output
layer; `--unconstrained` samples the whole vocabulary like the reference.

In Rust (`examples/speak.rs`):

```rust
use oxinfer::tts::{Speaker, SpeakerOptions, SpeakOptions};
let mut lo = SpeakerOptions::default();
lo.quant = oxinfer::model::Quant::Q4;
let speaker = Speaker::load("unsloth/orpheus-3b-0.1-ft", &lo)?;
let mut opts = SpeakOptions::default(); // voice, temperature, top_p, top_k, repetition_penalty, max_tokens, seed, ...
opts.voice = Some("leah".into());
let audio = speaker.speak("Hello there!", &opts)?; // Audio { samples: Vec<f32>, sample_rate: 24000 }
audio.write_wav("hello.wav")?;
// Streaming: chunks of audio while tokens are generated; Flow::Stop (or speaker.cancel_handle()) stops.
let speech = speaker.speak_stream("A longer sentence...", &opts, &mut |chunk| {
    play(chunk.samples); // chunk.start: index of its first sample
    oxinfer::engine::Flow::Continue
})?;
println!("RTF {:.2}", speech.stats.rtf());
```

### Batched utterances

`Speaker::speak_batch` generates several utterances together, each with its own text and
`SpeakOptions` (voice, sampling settings, seed, limits, references):

```rust
use oxinfer::tts::Utterance;
let (leo, mia) = (SpeakOptions { voice: Some("leo".into()), ..Default::default() },
                  SpeakOptions { voice: Some("mia".into()), seed: 7, ..Default::default() });
let speeches = speaker.speak_batch(&[Utterance::new("Hello there!", &leo), Utterance::new("And a longer one.", &mia)])?;
// speeches[i]: the Speech (audio, tokens, frames, finish, stats) of utterance i
let (audio, parts, stats) = speaker.speak_long(&long_text, &opts, 0)?; // the sentences as a batch, joined
```

- **Language model.** The prompts are prefilled one after another; then every decode step feeds
  the next token of each running utterance as one column of a single pass
  (`QwenModel::forward_hidden_batch`, `src/qwen_batch.rs`): the layer matmuls run on the
  multi-token decode kernels of
  [Multi-token passes](#multi-token-passes-and-moe-experts-on-the-cpu), which read the weights
  once for all columns, while each utterance keeps its own session (KV caches, position), and its
  attention is the one-token attention over its own cache. Up to 15 utterances share a pass with
  quantized weights (q8, q4, GGUF), 7 with bf16; a wider batch takes several passes per step.
- **Per utterance**: its sampler state and seed, repetition-penalty set, loop guard, length caps
  and stop. An utterance that ends leaves the batch and the others go on (ragged lengths).
  Unconstrained utterances share one pass through the whole output layer.
- **SNAC.** `Snac::decode_batch` decodes all the utterances in one pass per layer, the clips laid
  end to end with each one padded, and given its noise, on its own (groups of about 22 s of
  audio per pass).
- **Sentences.** `--speak-split` / `Speaker::speak_long` / `tts::split_sentences` cut a text at
  sentence ends (`.`, `!`, `?`, `…`, `。`, `！`, `？` and line breaks), join short neighbours up to
  240 bytes, and speak the pieces as one batch. Each piece is its own utterance (so a long text
  is no longer cut at `max_tokens`), and its prosody starts afresh.
- **CLI**: several `--speak`, or `--speak-file FILE` (one utterance per line, `-` for stdin);
  `-o out.wav` writes `out-1.wav`, `out-2.wav`, ...; `--voice a,b,c` gives utterance i the i-th
  voice, cycling. **Server**: an array as `input` of `/v1/audio/speech` (see
  [the server](#openai-compatible-server)). `examples/speak.rs` takes `--also TEXT`. The C API has
  no batch call yet, and a batch is not streamed (its audio is decoded when all of it is
  generated).

Every utterance of a batch is the utterance generated alone with the same options: the decode
kernels give each column the bits of a one-token pass, so on the CPU the tokens and the samples
are identical, greedy or sampled. Measured on the
Orpheus 3B Q4_K_M GGUF (CPU, 12 threads): a batch of 9 utterances in 8 voices (5 greedy without
SNAC noise, 4 sampled at 0.6 / 0.8 with it; 2 of the 9 unconstrained; 99 to 400 tokens each, 7
ending on a stop token at different steps and 2 at the 400-token limit; 2242 tokens in 399 steps)
against each utterance alone: 9 of 9 token sequences identical, and all 653312 samples
bit-identical (`real_batch_matches_alone`, an ignored test taking `OXINFER_TTS_MODEL`). The
tests that need no weights check the same on a random 2-layer Orpheus-shaped model and a small
random SNAC at bf16, q8 and q4 (9 utterances with mixed voices, seeds, samplers and limits:
`batch_matches_alone`), the batched language-model steps against one-token steps with sequences
leaving the batch (`batch_steps_match_one_token_steps`), and the batched SNAC decode against
single decodes with f32 and bf16 weights (`batch_decode_matches_alone`). The bf16 safetensors
checkpoint was not run in a batch on real weights (it was not on this machine).

A batch decodes on the CPU kernels (the GPU decode plans take one sequence), so with GPU
decoding on (`gpu_decode`, the default where a backend has it) a lone utterance and the same
utterance in a batch come from different kernels and can part where two tokens are close.

### Voice cloning

The pretrained base (`canopylabs/orpheus-3b-0.1-pretrained`, ungated copy
`unsloth/orpheus-3b-0.1-pretrained`) has no named voices; it continues in the voice of
reference utterances given in the prompt. The prompt is the one of canopylabs' notebook for
that model: per reference a finished turn (start of human, `<|begin_of_text|>transcript`, end
of text / human, start of AI, start of speech, the clip's SNAC codes as audio tokens, end of
speech, end of AI), then the turn for the new text (no voice name). The clip is resampled to
24 kHz and encoded by the SNAC encoder (also in Rust, see below). The model card notes that it
was not trained for cloning, so more reference pairs give a steadier voice; 5-10 s of clean
speech per reference works. Finetuned checkpoints accept references too, but follow them less.

```rust
use oxinfer::tts::{Speaker, SpeakerOptions, SpeakOptions, VoiceRef};
let speaker = Speaker::load("unsloth/orpheus-3b-0.1-pretrained", &SpeakerOptions::default())?;
let mut opts = SpeakOptions::default();
// Encoded once, reusable (serializable): from samples at any rate, a file, or an av::AudioClip.
opts.references.push(speaker.encode_reference(&samples, 16000, "What the clip says.")?);
opts.references.push(speaker.encode_reference_file("other.wav".as_ref(), "Its transcript.")?);
// or encoded at each call: VoiceRef::from_file("What it says.", "ref.wav")
let audio = speaker.speak("Something new.", &opts)?;
```

In JSON (C API, `--speak-json`): `"references": [{"text": "...", "audio": "ref.wav"}]`, or the
objects `oxi_speaker_encode_reference()` / `oxi_speaker_encode_reference_file()` return.

**Agreement** (one 6.5 s reference, a second sentence): the SNAC codes of the clip are identical
to the `snac` package's, and the prompt (586 tokens) identical to the notebook's; greedy
generation from it matches transformers float32 for 120 of 120 tokens.

**Intelligibility and voice**: two references (a male and a female voice, 6.5 s each, made by
the finetuned model's leo and tara), the four sentences of the WER test below, sampled with the
default settings: WER 5.7% for both (3 errors in 53 words each: "riverbank", "rainforest", as for
the finetuned voices). The median pitch follows the reference: 144-167 Hz for the leo clip
(151 Hz), 183-197 Hz for the tara clip (180 Hz), against 93-132 Hz for the pretrained model
without a reference.

### Other languages

The multilingual research release (`canopylabs/3b-{fr,de,ko,hi,zh,es_it}-ft-research_release`,
gated; ungated copies `Our-Org-2/3b-*-ft-research_release`, fp32 safetensors read as bf16) uses the
same prompt format with its own voices: French pierre, amelie, marie; German jana, thomas, max;
Korean 유나, 준서; Hindi ऋतिका; Mandarin 长乐, 白芷; Spanish javi, sergio, maria; Italian pietro,
giulia, carlo (`oxinfer::tts::voices_for`). `Speaker::info()` reports the languages (from the
model card), the known voices and whether the checkpoint is a pretrained base. Greedy French
tokens match transformers float32 (150 of 150).

WER of four sentences per voice (default sampling), transcribed with `--language`:

| Model | Voices | whisper-small | whisper-large-v3-turbo |
|---|---|---|---|
| French (`3b-fr-ft`) | pierre, amelie | 8% (24 utterances, 3 seeds) | 0% (8, seed 42) |
| French | marie | 58% (12, 3 seeds) | 81% (4) |
| German (`3b-de-ft`) | jana, thomas, max | 16% (12) | 1.7% (12) |

The voice "marie" of the French research model mostly produces wrong words (the same with
transformers' greedy tokens, so it is the checkpoint); the others are intelligible, and most of
whisper-small's errors are its own (whisper-large-v3-turbo hears them correctly).

### The SNAC codec

`oxinfer::snac::Snac` loads any SNAC checkpoint: `encode(&samples)` (the reference's `encode`,
input zero-padded to a multiple of the hop times `lcm(vq_strides[0], attn_window_size)`) and
`decode(&codes)`. The 32 and 44 kHz models (four codebooks at 8 / 4 / 2 / 1 times the frame
rate, music and general audio, not used by any Orpheus model) add local windowed attention after
the encoder and before the decoder: LayerNorm, 64-dim heads attending within windows of 32 steps
with rotary positions restarting in every window. Against `snac` 1.2.1 (float32, noise off), on a
4.9 s clip (resampled for 32 / 44 kHz) and 64 frames of random codes:

| Model | Encoder codes | Decode of the clip's codes | Decode of random codes | bf16 weights |
|---|---|---|---|---|
| snac_24khz | identical (57 / 114 / 228) | 1.6e-7 (127 dB) | 2.2e-6 (118 dB) | 58 / 46 dB |
| snac_32khz | identical (52 / 104 / 208 / 416) | 2.2e-7 (126 dB) | 5.6e-6 (118 dB) | 65 / 52 dB |
| snac_44khz | identical (72 / 144 / 288 / 576) | 3.3e-7 (126 dB) | 2.3e-6 (121 dB) | 64 / 54 dB |

(max abs difference and SNR.) Encoding 5 s takes 0.08-0.15 s. The reference requires lengths that
are a multiple of the attention window; here a shorter last window attends within itself.

With the `oxideav` feature, `Audio::to_frame()` / `AudioChunk::to_frame()` give an F32 mono
`oxideav_core::AudioFrame` with its `CodecParameters` (pts in samples), and `Audio::to_clip()` an
`av::AudioClip`. `SpeakOptions` deserializes from JSON; the C API takes it that way:
`oxi_speaker_load`, `oxi_speak` (PCM, optional chunk callback), `oxi_speak_wav`,
`oxi_speaker_cancel`, `oxi_speaker_stats_json` (`include/oxinfer.h`).

Streaming decodes overlapping windows of frames as they arrive, as the official decoder does (it
decodes the last 4 frames and keeps the second: 1 frame of left context, 2 of lookahead, and it
drops the first and last frames). Here every frame is emitted and the context is an option: the
decoder's receptive field is under 3 frames each way, so `context_left` / `context_right` 3 / 3
streams exactly the full decode (max abs diff 3e-7), the default 2 / 2 differs from it by at most
1.7e-3 (59 dB SNR), and the official 1 / 2 by 2.4e-2 (27 dB). The SNAC noise blocks draw their noise per (seed, layer, absolute position), so
overlapping windows see the same noise as a whole-clip decode.

Llama 3 itself runs on the same code as Qwen3 (full attention without query / key norms, llama3
RoPE frequency scaling), so any Llama 3.x checkpoint also works as a text model (`--model` with
the Llama 3 chat template), safetensors or GGUF (see [Llama 3 GGUF](#llama-3-gguf)).

GGUF conversions of Orpheus load the same way, as a path or `org/repo:QUANT`; the audio tokens
are in the GGUF vocabulary (156940 tokens, the 28683 added ones as user-defined tokens), and the
constrained audio head reads its rows of the output layer in place, in the file's own format:

```sh
./target/release/oxinfer --model isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF:Q4_K_M --speak "Hello there!" -o hi.wav
./target/release/oxinfer --model lex-au/Orpheus-3b-FT-Q8_0.gguf:Q8_0 --speak "Hello there!" -o hi.wav
```

On the backbone's first speech-token distribution (tara, a short prompt, CPU) the Q8_0 / Q4_K_M
files match the bf16 safetensors model with the same top 10 and KL 3.4e-4 / 2.6e-3. Eight
sentences (102 words, tara, default sampling, one seed each) transcribed by whisper-small: WER
0% for Q4_K_M, 1.0% for Q8_0 ("rainforests" -> "rainforest"), 0% for the safetensors model at
q4 and q8. Decode speed for a 7 s utterance (~600 tokens, not streamed, 3 interleaved runs, load
7-18):

| Weights | CPU | Default (Metal prefill) |
|---|---|---|
| GGUF Q4_K_M (2.4 GB; Q4_K with Q6_K for some tensors) | 99-102 tok/s | 96-99 tok/s |
| safetensors `--quant q4` | 112-116 tok/s | 113-118 tok/s |
| GGUF Q8_0 | 68-69 tok/s | 69-72 tok/s |
| safetensors `--quant q8` | 68-69 tok/s | 70-72 tok/s |

Q4_K_M decodes faster than real time (82 tokens/s) but ~13% slower than oxinfer's own q4: it keeps
the output layer and half the value and down projections at 6 bits (Q6_K).

**Agreement** with the reference: transformers 5.20 float32 `LlamaForCausalLM` + the `snac` 1.2.1
package, with the token handling of `orpheus_tts` (decoder.py) and the SNAC noise disabled on both
sides (it is random in the reference):

| Check | Result |
|---|---|
| Backbone last-token logits (Orpheus 3B, 14-token text prompt) | max abs diff 3e-5, same top 10 |
| Greedy speech tokens, tara, "The weather today is sunny with a light breeze from the west." (rep. penalty 1.1), CPU | identical, 379 of 379 tokens (54 frames + stop), constrained and unconstrained |
| Same, leo, "Could you send me the report by Friday? ...", Metal prefill | identical, 316 of 316 tokens |
| Waveform of those tokens after SNAC | max abs diff 3.3e-7 / 9.5e-7 (123 dB SNR) |
| SNAC alone, 400 random frames (34 s), f32 weights | max abs diff 6.6e-6 (117 dB); bf16 weights (`snac_f32=0`) 1.4e-2 (47 dB) |
| Streaming windows 1 / 2 vs the official decoder's windows | max abs diff 3e-7 |

q8 and q4 weights change the greedy path within a few tokens (greedy decoding of audio codes is
not a stable regime anyway); their speech is as intelligible as bf16 by the measure below.

**Greedy decoding and quantized weights.** Greedy (`--temp 0`, not the default) usually ends
normally at every precision: 183 of 184 greedy runs (20 texts, 2-8 voices, bf16 / q8 / q4) did.
The failure is a fixed point: on "The weather today is sunny with a light breeze from the
west." in tara's voice the q4 path says "The weather today", then repeats one frame (all 7
codes) forever, a held sound, with end of speech at probability ~1e-8, until `max_tokens`
(171 frames). The repetition penalty cannot break it (it divides any seen token's logit by 1.1,
however often it was seen); it is not a slow drift of the end-of-speech probability, which stays
far from the top as long as speech goes on and wins in one step when the text is done. Normal
speech never repeats a whole frame (0 repeats in all other runs, sampled q4 included), so two
guards are on by default:

- `loop_guard` (3): when a frame repeats the one before it 3 times in a row, the rest of the
  utterance is sampled at temperature >= 0.6, top-p <= 0.8 (a fallback, like Whisper's); after 8
  loops it ends. On the failing case, 4 of 5 fallback seeds then speak the whole sentence and
  stop (single-frame escapes rarely recovered: 2 of 5);
- `max_secs_per_byte` (0.25): at most 3 s plus 0.25 s per UTF-8 byte of text, about 3x the
  slowest speech in any of the supported languages, so a short text cannot run for 14 s.

Sampling (the default) never looped in these tests.

**Intelligibility**: four sentences in four voices each, sampled with the default settings,
transcribed by oxinfer's whisper-small and scored with jiwer (lower case, punctuation removed):
bf16 (tara, leah, leo, zac) WER 4.7%, q4 streamed (tara, jess, dan, mia) WER 4.7% (10 errors in
212 words each). 8 of the 10 come from "river bank" transcribed as "riverbank" (a substitution
and a deletion, in all four voices); the other two are "rainforests" -> "rainforest" and
"frog" -> "fraud" (bf16), "rainforest" twice (q4): about 1% of words actually misheard.

**Speed** on the M5 Max (18 threads, other jobs running, so the best case is a little higher),
an 8.5-10 s utterance, streamed (sliding windows, 2 / 2 context). Real-time output needs
7 x 24000 / 2048 = 82 tokens/s; the real-time factor (RTF) is total time / audio duration:

| Weights | Device | Decode | SNAC | RTF | First audio | Real time? |
|---|---|---|---|---|---|---|
| bf16 | CPU | 43 tok/s | 0.7 s | 2.0 | 0.62 s | no |
| q8 | CPU | 68 tok/s | 0.9 s | 1.33 | 0.50 s | no |
| q4 | CPU | 108-128 tok/s | 0.75 s | 0.76-0.90 | 0.38 s | yes |
| bf16 | Metal | 44 tok/s | 0.9 s | 2.0 | 0.59 s | no |
| q8 | Metal | 68 tok/s | 1.0 s | 1.32 | 0.41 s | no |
| q4 | Metal | 112-128 tok/s | 0.8 s | 0.69-0.85 | 0.28 s | yes |

Decoding is one token at a time on the CPU and memory-bound (2.8B weights per token once the
output layer is cut to the audio rows), so the GPU only speeds up the prompt; q4 is the setting
that keeps up with playback here. SNAC costs about 3 ms per frame decoded once (f32 weights;
streaming decodes each frame in 5 windows, ~14 ms per 85 ms frame).

**Speculative decoding** was measured and not added: no draft gives a speed-up. Acceptance per
generated token over 2976 tokens (8 utterances; "greedy" = the draft matches the greedy token,
"sampled" = the expected accepted mass `sum min(p, q)` at temperature 0.6, top-p 0.8):

| Draft | Greedy | Sampled | Tokens per verify pass (2 drafted) |
|---|---|---|---|
| n-gram prompt lookup (n = 1-4) | 0-0.1% | 0-0.1% | 1.00 |
| the previous frame's token at the same position | 0% | 1.4% | 1.01 |
| edwindn/orpheus-1b-0.1-weightDecay-chkpt78201 (Llama 3.2 1B, same tokens) | 20% | 31% | 1.24 / 1.41 |
| edwindn/orpheus-1b-0.1-finetuned-v3 | 20% | 31% | 1.24 / 1.41 |
| Gapeleon/Orpheus-1B-pt | 26% | 38% | 1.33 / 1.52 |

Audio codes do not repeat, so lookup drafts find nothing, and the community 1B models (canopylabs
never released its smaller sizes) cost about 0.35 of the 3B per token: at best 1.5 tokens per
pass for 1.7-2x the work of one target token, a slowdown. Medusa-style heads do not exist for
Orpheus.

Not supported: batched utterances on the GPU decode plans (a batch decodes on the CPU kernels),
streaming a batch, and a batch call in the C API. Korean, Hindi, Mandarin and the
Spanish / Italian model use the same code but were not tested.

## Getting weights

`--model` accepts a HuggingFace repo id and fetches it on first use (resumable, pure-Rust HTTP via
[rsurl](https://crates.io/crates/rsurl)) into `~/.cache/oxinfer/hub` (or `$OXINFER_CACHE`,
or `--cache-dir`):

```sh
./target/release/oxinfer --model google/gemma-4-E4B-it --quant q4 -p "Hello"
./target/release/oxinfer --model Qwen/Qwen3.8-27B --download-only     # just fetch
```

Set `HF_TOKEN` for gated repos. The Gemma 4 checkpoints are Apache 2.0 and not gated; manual
downloads with plain `curl` work too:

```sh
M=gemma-4-E4B-it
mkdir -p models/$M
for f in config.json generation_config.json tokenizer.json tokenizer_config.json model.safetensors; do
  curl -L -o models/$M/$f https://huggingface.co/google/$M/resolve/main/$f
done
```

For 31B, fetch `model.safetensors.index.json` and the `model-0000N-of-0000M.safetensors` shards instead of `model.safetensors`.
For speculative decoding also fetch `google/gemma-4-E4B-it-assistant` (160 MB) the same way.

### The ATOnline mirror

Repo ids are looked up on the [ATOnline](https://www.atonline.com/) mirror first: its
`AI/Model:get` method (`https://ws.atonline.com/_special/rest/AI/Model:get?key=org/name`) lists a
repo's files with download URLs, small ones inlined. Repos it does not carry come from
huggingface.co. `OXINFER_HUB=mirror` or `OXINFER_HUB=hf` (`oxinfer::hf::set_hub` in the library) pins one source; the HuggingFace token is
only ever sent to huggingface.co. The browser build uses the same lookup (the mirror allows
cross-origin requests).

### GGUF files

`--model` also takes llama.cpp GGUF files, as a path or as `org/repo:QUANT` (the one file of a
GGUF repository whose name ends in `QUANT.gguf`, e.g.
`HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF:Q4_K_P`). Models split into parts
(`name-00001-of-00009.gguf`, ...) load from any part or from their folder, and `org/repo:QUANT`
fetches every part. Architectures: `qwen35`
(Qwen3.5 / 3.6 / 3.8), `qwen35moe`, `qwen3`, `llama` (Llama 3.x, and Orpheus TTS, see
[below](#llama-3-gguf)), `gemma4` (Gemma 4 text models, dense and 26B-A4B; see
[Gemma 4 26B-A4B](#gemma-4-26b-a4b-moe)), `mistral-large4` (see
[Mistral Large 4](#mistral-large-4-1t-a52b), whose Pixtral `mmproj` file is read too) and `deepseek4`
(see [DeepSeek V4](#deepseek-v4-flash)). The tokenizer and chat template come from the
GGUF metadata (the rebuilt tokenizer matches Qwen3.8's `tokenizer.json` on 305 test texts, and
Gemma 4's, a SentencePiece-style BPE with byte fallback, on this README and multilingual text), and
llama.cpp's conversions are undone at load (`gguf.rs`): norms with the 1 folded in, `ssm_a`, the
tiled Gated DeltaNet value-head order, separate 3D expert tensors, Llama's permuted query / key
rows, Gemma 4's proportional RoPE stored as per-frequency divisors (`rope_freqs`). The `mmproj`
vision files are read for Mistral Large 4 (`pixtral`) and Qwen3.5-family towers
(`qwen3vl_merger`, see [Qwen3.8-Flash-Next](#qwen38-flash-next)); other models' image input needs
the safetensors or MLX checkpoint. Weights stay in their GGUF
format: F32, F16, BF16, Q4_0/Q4_1/Q5_0/Q5_1/Q8_0, Q2_K-Q6_K, IQ1_S/IQ1_M, IQ2_XXS/XS/S,
IQ3_XXS/S, IQ4_NL/XS and MXFP4 (`ggml.rs`, dequantization bit-exact with gguf-py), with NEON decode
kernels for every type and Metal prefill kernels. On x86-64 with AVX2 every type's int8 dot
products run on `maddubs` / `madd`, and the block-of-32 formats, the K-quants and IQ4_XS are
unpacked in registers (`kernels_ggml_x86.rs`; the grid-table i-quants IQ1-IQ3 still unpack through
the portable decoder); every row is bit-identical to the portable kernel's. The i-quants (IQ1-IQ3) decode several times
slower than Q4_K on the CPU (table lookups); Q4_K, Q5_K, Q6_K, IQ4_XS and Q8_0 are close to
oxinfer's own Q4/Q8. Embedded MTP layers (`blk.N.nextn.*`) work with `--mtp`.

On HauhauCS's Qwen3.8-27B Q4_K_P the next-token distribution matches llama.cpp on the same file
(same top 10, log-probabilities within 0.03, mean 0.013). F32/BF16 GGUFs from llama.cpp's
converter match transformers to 5e-5. `--save-quantized` does not take GGUF input.

#### Llama 3 GGUF

```sh
./target/release/oxinfer --model bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M -p "Why is the sky blue?"
```

llama.cpp's converter permutes the rows of `q_proj` / `k_proj` within each head for ggml's
adjacent-pair RoPE (`LlamaModel.permute` in `convert_hf_to_gguf.py`: stored row `2i + j` of a head
of `d` rows is Hugging Face row `j * d/2 + i`). Rotating adjacent pairs of the stored rows is the
same arithmetic as rotate-half RoPE on the original rows, and attention scores do not depend on
the order within a head, so the weights stay mapped as they are (no copy) and the Llama GGUF path
uses the interleaved RoPE instead; LoRA adapters on `q_proj` / `k_proj` get their rows reordered
to match. The llama3 RoPE scaling (factor, low / high frequency factors, original context) is not
in the GGUF metadata: the converter stores the resulting divisor of each frequency as the
`rope_freqs.weight` tensor, which divides the inverse frequencies. Base, heads, KV heads, head
size and eps come from the `llama.*` keys. The tokenizer is Llama 3's byte-level BPE with the
`llama-bpe` pre-tokenizer regex, no normalizer and `ignore_merges`, plus the special and
user-defined tokens (Orpheus's 28683 audio tokens): token for token identical to the checkpoints'
own `tokenizer.json` (Llama 3.2 1B and Orpheus 3B) on 2953 texts (this README, source code,
ten languages, emoji, whitespace runs, special tokens; 117k tokens). Chat uses the Llama 3
format when `tokenizer.chat_template` is Llama 3's (header tokens); other Llama-architecture
GGUFs (SmolLM, Mistral, ...) load as text models with `--raw` but their chat formats are refused.

Agreement on Llama-3.2-1B-Instruct (bartowski's GGUFs; seven raw prompts of 5-24 tokens in
English, German, French and Python, each starting with `<|begin_of_text|>`; the last-token
distribution and 64 greedy tokens; CPU):

| Check | Result |
|---|---|
| F16 GGUF vs the bf16 safetensors (unsloth/Llama-3.2-1B-Instruct), oxinfer both | logits bit-identical (max abs diff 0), same greedy text |
| Q8_0 GGUF vs bf16 safetensors | same top 1 in 7/7, top 10 overlap 9-10/10, KL 1e-5..3.5e-3 |
| Q4_K_M GGUF vs bf16 safetensors | same top 1 in 6/7, top 10 overlap 8-10/10, KL 3e-3..0.17 |
| F16 GGUF, oxinfer vs llama.cpp (same file, top 20 log-probabilities) | same top 1 7/7, within 0.027; 64/64 greedy tokens identical in 7/7 |
| Q8_0 GGUF, oxinfer vs llama.cpp | same top 1 7/7, within 0.10-0.20; 64/64 greedy tokens in 6/7, the 7th splits at a near tie after 8 |
| Q4_K_M GGUF, oxinfer vs llama.cpp | same top 1 7/7, within 0.15-0.34; 64/64 greedy tokens in 2/7, the others split at near ties after 1, 8, 19, 32 and 55 |
| Q8_0 / Q4_K_M, transformers float32 on the GGUF's dequantized weights (Q/K rows un-permuted) vs oxinfer | within 0.02-0.12 / 0.02-0.16 (vs llama.cpp: 0.11-0.21 / 0.13-0.36) |

On the F16 file that float32 reference matches oxinfer to 1e-4. The quantized files differ from
llama.cpp by more than the F16 file, but against the exact float32 forward pass on the same
dequantized weights oxinfer is the closer of the two, so the gap comes from how the engines round
intermediate values (both quantize activations to int8 for the integer dot products), not from a
different reading of the weights.

### MLX checkpoints

MLX repos (mlx-community and the like) store each matrix as packed 4-bit values (`.weight`, eight
per uint32) plus a bf16 scale and bias per group of 64: `w = scale * q + bias`. Since
`scale * q + bias = scale * (q - 8) + (bias + 8 * scale)`, the nibbles carry over unchanged into
the Q4 kernels' layout, and the extra term becomes one bf16 offset per group. Decoding adds
`offset * sum(x over the group)` per row after the int8 dot products, and prefill dequantizes it
into the f32 tiles (on the CPU, Metal, CUDA and WebGPU alike). Other bit widths, and quantized
tensors that are read as plain bf16 (Gemma 4's vision and audio projections), are dequantized on
load; embedding tables are dequantized one row at a time straight from the checkpoint. The MLX
names (`language_model.model.*`, `vision_tower.*`) map to the transformers ones, and channels-last
convolution weights are permuted back.

```sh
./target/release/oxinfer --model mlx-community/gemma-4-e2b-it-4bit -p "Hello"
./target/release/oxinfer --model mlx-community/gemma-4-e2b-it-4bit --audio clip.wav -p "Transcribe this."
```

Against transformers float32 on the dequantized MLX weights, last-token logits agree to a mean
absolute difference of 0.016 (Gemma 4 E4B), 0.026 (Qwen3-4B) and 0.055 (Gemma 3 4B), with the
same top 10; the difference to the original bf16 models is MLX's own quantization.

M5 Max, 200-token replies: Gemma 4 E2B 110 tok/s (120 for oxinfer's own Q4 file, which has no
offsets), Gemma 4 E4B 71 tok/s, Gemma 3 4B 77 tok/s, Qwen3-4B 70 tok/s. Prefill of 2153 tokens on
Gemma 3 4B takes 4.0 s with Metal (12.9 s on the CPU).

`tools/mlx_to_hf.py` dequantizes an MLX checkpoint into a float32 transformers folder, which is
the exact reference for those weights; `tools/ref_logits.py` then compares logits. LiteRT-LM
files (`.litertlm`, TFLite graphs) are not supported.

## Build and run

```sh
cargo build --release
./target/release/oxinfer --model models/gemma-4-E4B-it --prompt "Explain RMSNorm in two sentences."
./target/release/oxinfer --model models/gemma-4-E4B-it --chat          # interactive, multi-turn
./target/release/oxinfer --model models/gemma-4-E4B-it --think -p "..."  # enable thinking mode
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--temp F --top-k N --top-p F --seed N` | sampling (default is greedy, `--temp 0`) |
| `--system TEXT` | system prompt (default: a short one saying the model runs locally, with the local date, time and zone) |
| `--no-system` | no system prompt (also implied by `--raw` and `--dump-logits`) |
| `--tools` | let the model call built-in tools: `get_current_datetime` and `calculate` (see Tool calling) |
| `--print-prompt` | print the fully formatted prompt and exit |
| `--raw` | feed the prompt as-is after `<bos>`, no chat template |
| `--max-tokens N` | cap on generated tokens |
| `--threads N` | worker threads (default: all cores) |
| `--quant q8` | quantize streamed weights to int8 at load (blocks of 32, f32 scales); ~1.8x less memory traffic |
| `--quant q4` | GGUF-style Q4_0 (blocks of 32, 4-bit, f16 scale) for all streamed weights; ~3.5x less traffic |
| `--draft DIR` | speculative decoding with Google's `gemma-4-E4B-it-assistant` drafter (greedy only; output identical to plain greedy), or a DSpark draft (greedy and sampled): Mistral Large 4 / DeepSeek V4 a draft GGUF (default: the `dspark*.gguf` beside the model, which `auto` requires), Qwen / Gemma 4 a Hugging Face draft or dflash GGUF, see [DSpark](#dspark-speculative-decoding) and [Qwen and Gemma 4 drafts](#dspark-drafts-for-qwen-and-gemma-4) |
| `--no-draft` | no DSpark draft (plain decoding) |
| `--draft-tokens N`, `--spec-draft-n-max N` | max draft tokens per round (default 6 with `--draft`, 3 with `--mtp`, 8 with `--mtp --stream`; the length adapts; DSpark: Mistral Large 4 / DeepSeek V4 3, Qwen / Gemma 4 the whole block) |
| `--spec-draft-fixed` | DSpark (Qwen, Gemma 4): always draft `--draft-tokens` instead of choosing each round's length by measured cost |
| `--spec-draft-conf-min C` | DSpark: cut a block before the first draft whose confidence-head estimate is below C (default 0: off) |
| `--spec-draft-p-min P` | DSpark: cut a block before the first draft the draft model gives less than probability P (default 0.85 when the model streams, else 0) |
| `--exact-verify` | DSpark / MTP: verify with the one-token kernels, so greedy output is the plain greedy output token for token (default: fast batched matrix kernels on the GPU, whose logits differ from plain decoding in rounding; `--fast-verify` asks for them explicitly), see [fast verification](#fast-verification) |
| `--mtp` | Qwen3.5/3.8: speculative decoding with the checkpoint's multi-token-prediction layer (greedy and sampled) |
| `--mtp-fixed` | with `--mtp`, always draft `--draft-tokens` instead of adapting between 1 and N |
| `--mtp-from DIR` | `--mtp` with the MTP weights taken from another checkpoint (e.g. the original of a converted copy) |
| `--mmap` | map weights from disk instead of copying them into RAM |
| `--save-quantized DIR` | write a pre-quantized, text-only checkpoint at `--quant` precision (see below) and exit |
| `--devices` | list detected CPU features and GPUs (Metal, CUDA, Vulkan) and exit |
| `--device D` | `auto` (default: the first GPU backend that starts, Metal or CUDA before Vulkan, else the CPU), `cpu`, `metal`, `cuda`, `vulkan`; a device asked for by name that is missing or cannot start is an error, not a fallback |
| `--gpu-decode on\|off` | generate a whole token per GPU submission (default `on` where supported; see [Decoding on the GPU](#decoding-on-the-gpu)); engine / speaker option `gpu_decode` |
| `--prefill-chunk N` | tokens per prefill chunk (default 256 on the CPU, 1024 when a GPU backend is active) |
| `--profile` | per-operation timing and bandwidth breakdown |
| `--dump-logits FILE` | write the logits for the last prompt token (for reference checks) |
| `--show-tokens` | print token ids |

By default the whole safetensors file is read into RAM at startup (a few seconds), which
gives steady decode speed from the first token. `--mmap` starts faster but pays page
faults on the first pass.

### As a library

`oxinfer::engine` is the CLI's text generation as an API: load a directory, GGUF file or repo
id with `EngineOptions` (every flag above that applies to text, also settable as `key=value`
pairs), then hold a conversation with streamed output, cancellation from another thread
(`CancelHandle`), tool calls, attachments (`Engine::send_with_media` with images, videos and
audio clips in an `Attachments`) and per-turn statistics:

```rust
use oxinfer::engine::{Engine, EngineOptions, Flow};
let mut opts = EngineOptions::default();
opts.set("quant", "q8")?;
opts.set("mtp", "1")?;
let mut engine = Engine::load("Qwen/Qwen3.5-0.8B", &opts)?;
let reply = engine.send("Why is the sky blue?", &mut |d| {
    print!("{}", d.text); // d.thinking marks thinking text
    Flow::Continue        // Flow::Stop ends the reply
})?;
println!("\n{:.1} tok/s, {:.0}% drafts accepted", reply.stats.decode_tok_s(), 100.0 * reply.stats.acceptance());
```

`examples/chat.rs` is a complete chat. Wrap `Engine::load` in `oxinfer::hf::with_progress` for
download progress and cancellation. Core ML packages (below) load through the same
`Engine::load` on macOS / iOS, with the options `coreml_units` (`ane`, `all`, `gpu`, `cpu`) and
`coreml_prefill` (`auto`, `batch`, `pad`, `single`); `engine.info().coreml` describes them. The lower-level pieces the CLI is built from are public too:
`oxinfer::generate` (decoding loops with per-token callbacks, MTP and drafter speculative
decoding, streaming detokenization) and the model, session and sampler types. A C API over the
engine (`capi` feature, `include/oxinfer.h`) and a SwiftUI test app for iOS are in `ios/`
(see `ios/README.md`). Neuron edits, steering vectors, directional ablation and SAE features are
`Engine::set_interventions` (see [Runtime interventions](#runtime-interventions-neuron-edits-steering-abliteration-sae-features)).
`Engine::save_session` / `Engine::load_session` write and restore a conversation's caches, and the
`prompt_cache` option keeps them in a directory for `Engine::chat` (see
[Session snapshots and the prompt cache](#session-snapshots-and-the-prompt-cache)); prompt-lookup
drafting is the `lookup` options ([Prompt lookup](#prompt-lookup)).

Tokenization is in the crate too: `oxinfer::tokenizer::Tokenizer` reads HuggingFace
`tokenizer.json` files (BPE, Unigram, WordPiece and WordLevel models with every normalizer,
pre-tokenizer, post-processor and decoder of the format, BPE dropout, script splitting,
truncation with stride and overflowing parts, padding, added-token flags) and has the API of
the `tokenizers` crate, which oxinfer no longer depends on: `encode` of single texts, pairs
(with type ids) and pre-split words, `encode_batch`, `decode`, `decode_batch`, `decode_stream`,
`token_to_id`, `get_vocab`. Regex splits run on an in-crate backtracking engine with
`fancy-regex`'s syntax and semantics (lookbehind, atomic groups, backreferences, conditionals,
`\K`, `\G`). It was compared with `tokenizers` 0.23.2 on 41 `tokenizer.json`
files (Qwen 2.5 to 3.5, Llama 3, Mistral Small 3.2, Mistral Large 4's tekken, Gemma 3 and 4,
DeepSeek V4, Janus, Whisper, CLIP, T5, the WordPiece encoders of BGE, MiniLM, MPNet, nomic and
mxbai, the XLM-R Unigram of BGE-M3 and multilingual-E5, ModernBERT): ids, special-token masks
and decoding with and without special tokens agree on 30,037 texts each (a multilingual corpus
and random strings of added tokens, combining marks, emoji, control characters and whitespace
runs). Loading is faster (Gemma 4's 32 MB file: 280 ms instead of 430 ms), and so is encoding
(Qwen 2.5 on that corpus: 225 ms instead of 336 ms). Unusual files were covered by rewriting ten
of them (BERT, MPNet, ModernBERT, Qwen, Gemma, Mistral, T5, BGE-M3, CLIP, Whisper) about 200
ways each: every pre-tokenizer, two dozen split regexes with every behavior, normalizer chains,
model options, post-processors with each truncation strategy and padding, decoders and
added-token flags. Single texts, pairs, pre-split words, batches, decoding and streaming
decoding agree on 40 texts per rewrite, except where `tokenizers` itself is undefined: several
added tokens that normalize to the same string (it then picks one by hash-map order).

### Using oxinfer from oxideav (and other crates)

Images, audio and video in the API are [oxideav](https://crates.io/crates/oxideav-core) types:
decoded `VideoFrame`s and `AudioFrame`s with the `CodecParameters` that describe them, as an
oxideav pipeline hands them from stage to stage. Two features split what a dependent pulls in:

| Feature | Brings | Gives |
|---|---|---|
| `oxideav` | `oxideav-core` only | `oxinfer::av` and the in-memory inputs and outputs below; `oxinfer::oxideav_core` re-exports the crate |
| `media` (default, implies `oxideav`) | oxideav-pipeline, oxideav-image and the codec and container crates (PNG, JPEG, WebP, GIF, BMP, MP3, FLAC, AAC, Vorbis, Opus, WAV, MP4, MOV, Matroska / WebM, Ogg, AVI, H.264, H.265, VP8, VP9, AV1, MPEG-4 Part 2) | the path-based inputs (`--image`, `--video`, `--audio`, `transcribe_file`, `Media`) |

`hf` (HuggingFace fetching), `gpu` (run-time GPU backends) and `server` (default: the
[OpenAI-compatible server](#openai-compatible-server), with httpsd) are independent of both. A crate
that already has decoded frames, such as an oxideav filter, needs only the types:

```toml
[dependencies]
oxinfer = { version = "0.0.4", default-features = false, features = ["oxideav", "hf", "gpu"] }
oxideav-core = "0.1.41" # the same 0.1 release oxinfer uses, so the types are the same
```

Chat and multimodal prompts take frames next to file paths (each kind after the files of the
same kind):

```rust
use oxinfer::engine::{Attachments, Engine, EngineOptions, Flow};
let mut engine = Engine::load("Qwen/Qwen3.5-0.8B", &EngineOptions::default())?;
// A picture: one VideoFrame and its parameters (width, height, pixel_format).
let media = Attachments::default().with_image_frame(frame, params);
// A clip: frames in presentation order, their parameters and the stream time base; frames are
// sampled as the model's video processor does, and the prompt's timestamps come from the pts.
let media = media.with_video_frames(frames, video_params, Some(time_base));
// Audio (Gemma 4): frames and their parameters (sample_rate, channels, sample_format).
let media = media.with_audio_frames(audio_frames, audio_params);
let reply = engine.send_with_media("What happens here?", &media, &mut |_| Flow::Continue)?;
```

The other consumers and producers of media:

```rust
// Speech to text from decoded audio (any sample format, rate and channel count).
let t = transcriber.transcribe_frames(&audio_frames, &audio_params, &TranscribeOptions::default())?;
// Typed decisions over a picture (placed before the state text).
let req = DecisionRequest::noul("Backyard camera.", "Is there an animal?").with_image_frames([Picture::new(frame, params)]);
// Stable Diffusion output as an Rgb24 frame for an encoder or a filter graph.
let (img, _) = pipeline.generate(&params, &mut |_, _| {})?;
let (frame, frame_params) = img.into_video_frame();
```

Underneath, `av::video_frame_to_rgb` and `av::audio_frames_to_mono` do the conversions, and
`mm::prepare_frames` (with `mm::MmFrames`) turns frames into soft tokens for the lower-level
`generate` API. Pixel formats: packed RGB / BGR in every byte order (with or without alpha or
padding, 8 and 16 bits, 5-6-5 and 5-5-5), gray (8 to 16 bits, with alpha), 1-bit mono, `Pal8`,
planar YUV 4:2:0, 4:2:2, 4:4:4, 4:4:0 and 4:1:1 at 8 to 16 bits (with alpha, `YuvJ*`), NV12 /
NV21, YUYV / UYVY and planar GBR(A); others (CMYK, float) are an error naming the format. YCbCr
follows the colour signal of the frame or the stream (BT.601 limited range when unspecified;
BT.709, BT.2020, FCC, SMPTE 240M, identity, full range), in oxideav-pixfmt's fixed-point
arithmetic: the RGB is byte-identical to `oxideav_image::Image::to_rgb8` for every format both
convert, so frames and files give the same soft tokens, replies and transcripts
(`tests/oxideav_e2e.rs`). A missing `width`, `height` or `pixel_format` is inferred from the
plane layout, as for decoded files; audio of every `SampleFormat`, interleaved or planar, is
averaged to mono and resampled to the model's rate.

**Dependency cycles.** `media` depends on the codec and container crates directly, not on
`oxideav-meta`, so a transform crate that depends on oxinfer can itself be registered by
`oxideav-meta` without a cycle, whatever features it enables. (Cargo refuses any cycle among the
activated dependencies and accepts one through an optional dependency that is not enabled. With
a `media` that depended on `oxideav-meta`, oxinfer -> oxideav-meta -> transform -> oxinfer would
fail in any build that registers the transform in oxideav-meta and enables `media` anywhere, even
in another crate, since features are unified across the whole graph.) The rule for the transform crate and anything oxideav-meta pulls in:

- depend on `oxinfer` with `default-features = false` and the features you use (`oxideav`, and
  `hf` / `gpu` as wanted); take `media` only if you need oxinfer to open files itself;
- never make oxinfer depend on a crate that depends on the transform (oxideav-meta, an
  "everything" facade): its `media` feature lists the codec crates one by one for that reason.

## Performance

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

### Decoding on the GPU

With the Metal backend, single-token decode steps also run on the GPU, the whole forward pass in
one command buffer per token (see [Platforms](#platforms-and-accelerators) for what runs where).
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
[Platforms](#platforms-and-accelerators)). Measured on an M5 Max with a ~3k-token prompt (default
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

### Comparison with llama.cpp

(Speculative decoding with DSpark drafts against llama.cpp master: see
[DSpark drafts for Qwen and Gemma 4](#dspark-drafts-for-qwen-and-gemma-4).)

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

### Quantization quality

Q8 and Q4 are both measured against the bf16 output distribution (itself within 1e-4 of the
transformers float32 reference). KL divergence of the next-token distribution on a test prompt is
about 1e-5 nats for Q8 and 3e-5 nats for Q4 on E4B, with the same top-5 tokens. Google also
publishes `gemma-4-*-it-qat-q4_0-unquantized` checkpoints trained for Q4_0; they load the same
way (`--quant q4` on that directory) but did not measure better than Q4 of the regular weights
here, and they are a different fine-tune (KL 0.03 from the regular model).

### Pre-quantized checkpoints

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

### Speculative decoding

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
in [DSpark drafts for Qwen and Gemma 4](#dspark-drafts-for-qwen-and-gemma-4); `--no-draft` turns it
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

### Multi-token passes and MoE experts on the CPU

Speculative verification (and prompts shorter than 8 tokens) runs on the CPU's decode kernels,
whose rows are bit-identical whatever the token count, so verification logits equal one-token
logits. Two lossless changes (after the analysis of llama-modchip in
[docs/llama-modchip](docs/llama-modchip/README.md)) made those passes cheaper:

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

### Prompt lookup

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

### Streaming weights from flash (models larger than RAM)

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

#### GPU decoding with streamed weights (mixed GPU / CPU)

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

## Platforms and accelerators

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


### CUDA

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

### Vulkan

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
### WebAssembly

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

### Browser (WebGPU)

The site has a Chat mode and an Image mode. Image mode loads SD 1.5 with the LCM-LoRA by default
(2.1 GB of fp16 weights plus 135 MB from Hugging Face; 4 Euler ancestral steps without guidance,
about 8 s per 512x512 image in Chrome on the M5 Max with the machine busy, 63 dB from the native
Metal image); SD-Turbo (2.6 GB, one step) is the other preset, and a custom model can take LoRA
files too (`?lora=org/name/file.safetensors:0.8`, comma separated). With the LCM-LoRA, plain Euler
leaves speckle noise in the image: LCM is trained to be re-noised between steps, as the ancestral
sampler does. A third preset, InstructPix2Pix (with the same LCM-LoRA, 4 steps), edits an
uploaded picture by instruction, and every model takes an input image and a mask (image-to-image,
inpainting; see [Editing images](#editing-images)); headless Chrome produced the same edit as the
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

Point the page (or `run.mjs`) at a pre-quantized checkpoint (`--save-quantized`, see above): it
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
by `media`; see [Using oxinfer from oxideav](#using-oxinfer-from-oxideav-and-other-crates)), `capi`,
and `web` (the browser module, off by default and only meaningful for `wasm32-unknown-unknown`);
`--no-default-features` gives the plain CPU build.
Cross-compiling needs no C toolchain: the tokenizer uses its pure-Rust regex backend.

### Core ML / Apple Neural Engine

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

#### Converting models

`tools/coreml_convert.py` converts Hugging Face checkpoints into Core ML packages for the
Neural Engine of iPhones and Macs, in the layout of [ANEMLL](https://github.com/Anemll/Anemll)
(meta.yaml, an embeddings model, decoder chunks with `infer` and `prefill` functions sharing
weights and a Core ML state, a 16-way split LM head, the tokenizer files). Qwen3 packages run
unchanged under ANEMLL's own `chat.py`; Qwen3.5 / 3.6 / 3.8 models (Gated DeltaNet recurrent
state kept as Core ML states, chunked delta rule in the prefill function; MoE variants evaluated
densely, for small models) and Gemma 4 E2B / E4B (per-layer embeddings, KV sharing, local and
global caches) extend the format as described in [docs/coreml-format.md](docs/coreml-format.md);
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

## How it works

- `safetensors.rs`: parses the header, exposes bf16 tensors as `&[u16]` (single file or sharded).
- `config.rs`: reads `config.json` and derives per-layer facts (sliding vs global, head dims, KV sharing source, k=v).
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

## Verifying against the reference

```sh
python3 -m venv venv && ./venv/bin/pip install torch transformers safetensors numpy
printf '<bos><|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n' > prompt.txt
./target/release/oxinfer --model models/gemma-4-E2B-it --prompt "What is the capital of France?" --dump-logits ours.bin
./venv/bin/python tools/ref_logits.py models/gemma-4-E2B-it prompt.txt ours.bin --generate 12
```

For images and videos (Qwen3.5-family, and Gemma 4 videos; the reference needs `torchvision`),
dump oxinfer's side without a system prompt and let `tools/ref_vl.py` run the processor and model:

```sh
M=~/.cache/oxinfer/hub/Qwen--Qwen3.5-0.8B
./target/release/oxinfer --model $M --image fox.png -p "Describe this image." --no-system --print-prompt > prompt.txt
./target/release/oxinfer --model $M --image fox.png -p "Describe this image." --device cpu --dump-mm ours --dump-logits ours.bin
./venv/bin/python tools/ref_vl.py $M --image fox.png --prompt "Describe this image." \
    --ours ours --ours-logits ours.bin --ours-prompt prompt.txt --generate 40
```

Videos are compared as directories of frames (`ffmpeg -i clip.mp4 frames/%05d.png`, then
`--video frames --frames-fps 24` on both sides), so both read the same pixels.

Runtime interventions take the same JSON spec on both sides (`tools/ref_intervene.py` applies it
with PyTorch forward hooks; `--generate N --ours-ids IDS.json` also compares greedy text):

```sh
M=~/.cache/oxinfer/hub/Qwen--Qwen3.5-4B
./target/release/oxinfer --model $M --no-system -p "Write a short poem about the sea." --print-prompt > prompt.txt
./target/release/oxinfer --model $M --no-system -p "Write a short poem about the sea." --device cpu \
    --interventions spec.json --dump-logits ours.bin
./venv/bin/python tools/ref_intervene.py $M prompt.txt spec.json ours.bin
```
