# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.12](https://github.com/KarpelesLab/oxinfer/compare/v0.0.11...v0.0.12) - 2026-11-14

### Other

- MiniMax H3-Image: its VAE is no longer taken for the video VAE.
- MiniMax-H3 video generation, what was checked and what is not supported.
- The model catalog has a video kind with text-to-video and image-to-video capabilities, and MiniMax-H3 in it.
- MiniMax H3 video: text to video and first / last frame to video from the command line (--frames, --last-image), written as Motion JPEG AVI, animated WebP or PNG frames.
- MiniMax H3 video: the DiT's audio stream and the packed text / keyframe / audio / video sequence with its two schedules, checked against diffusers on tiny checkpoints.
- MiniMax H3 video: the video VAE, a causal 3D CNN encoder and the ViT decoder with 3-axis RoPE, temporal clips and tiling.
- LFM3-Omni in the model catalog, docs/models.md, docs/lfm3.md and the README index, with what was checked against the reference.
- tools/ref_lfm3_omni.py compares features, tokens, logits, greedy text, speech codes and the codec's waveform with the checkpoint's own code.
- LiquidAI/LFM3-Omni-2B reads audio and images through its own encoders, answers in text on the chat path, and speaks through the depth decoder and the Liquid codec (--speak, -o reply.wav).
- Streamed chat and completion responses say what they wait for before the first token.

## [0.0.11](https://github.com/KarpelesLab/oxinfer/compare/v0.0.10...v0.0.11) - 2026-11-14

### Other

- a newer release replaces the executable before any command runs, or is reported when it cannot be.
- The prompt-lookup test no longer asserts draft counts that depend on timing.
- Granite, LFM3 and EmbeddingGemma 2 notes say only what was checked.
- EmbeddingGemma 2: google/embeddinggemma-2 as an embedding backbone, with image and audio embeddings through Gemma 4's vision and audio towers.
- chat templates without a thinking block (taobao-mnn/MNN-Pocket-0.3B) get no <think> prefill after the assistant header.
- Liquid LFM3-1.7B runs on the Qwen path, its gated short convolutions as a mixer beside full and sliding-window attention, with its chat template and tool calls.
- Granite 4: IBM Granite 4.1 / 4.2 and fine-tunes such as TwIL-8B run on the Llama path, from safetensors and GGUF, with their chat template and tool calls.
- the catalog entry, the model table row, docs/haru.md with the measured agreement and its line in the README index.
- the comparison with the PyTorch reference (tools/ref_haru.py records a greedy conversation and every normal value it draws; examples/haru_check.rs replays them: the reference clip, the codec both ways, prompt tokens, and per frame the text token, the hidden state, the latent and the audio).
- the command line, in the shape of the reference's offline tool (--audio USER.wav -o REPLY.wav, --system for the card, --ref-audio for the voice; the model's text on standard output).
- Haru-4B (haru, higurashi-ai/Haru-4B), full-duplex speech to speech, from the PyTorch reference (oxinfer::haru): a frame's three streams summed into one position of the Qwen3.5-4B text model, the MeanFlow audio head and the projectors, VibeVoice's causal codec with a state per layer, the prompt (character card, voice reference), the text stream's sampling with the DRY penalty, and a session stepped one frame at a time.
- Audio8 TTS: the catalog entry, the model table row, docs/audio8-tts.md with the measured agreement, its line in the README index and the server's speech row.
- Audio8 TTS: the comparison with the checkpoint's own code (tools/ref_arktts.py, examples/arktts_check.rs: token ids, the RoPE table, the codes of a reference clip, prompt rows, greedy frames and logits, the decoder's waveform).
- Audio8 TTS: --speak on the command line (--ref-audio with --ref-text or a named clip of the voices folder, --speak-stream) and the server's /v1/audio/speech, streamed as it is generated.
- Audio8 TTS (arktts, Edge0/Audio8-TTS-Realistic-Air): the model from the checkpoint's own code (oxinfer::arktts): the chat prompt in rows of a token and ten codes, the slow and fast transformers with their bf16-rounded RoPE tables, the sampler with its redraw of repeated semantic tokens, and the codec read from codec.pth (causal DAC decoder and encoder, window transformers, residual quantizers), decoded in windows and piece by piece for streaming.
- documentation (docs/longcat.md with the sources of the architecture, the agreement on tiny checkpoints and the runs on LongCat-Flash-Lite-Sparse's real weights; the models table, the README index) and the --models catalog entries.
- longcat_check --nll (the likelihood a checkpoint gives a token sequence, for runs on real weights), and the n-gram mean taken as a division like the reference.
- the tiny checkpoint builder in the released layout (tools/longcat_tiny.py: LongCat-2.5's architecture, or LongCat-Flash-Lite's / LongCat-Flash-Chat's), the reference on transformers' LongcatFlash model and Meituan's NgramEmbedding with the indexer transcribed from SGLang-FluentLLM (tools/ref_longcat.py) and the comparison (examples/longcat_check.rs).
- LongCat from the CLI and the engine: streamed from storage when larger than memory, the chat format chosen from the checkpoint's template, prompt-lookup rounds.
- LongCat-2.5 (pro_v, LongcatCausalLM; also LongCat-2.0, LongCat-Flash-Lite-Sparse and the LongCat-Flash models they extend): the text model (two sublayers and a shortcut MoE per layer, MLA with LoRA scales and DeepSeek's YaRN, the sparse-attention indexer with keys always attended and a choice shared by the second attention, a softmax router with identity experts, the n-gram embedding read a few rows per token with the tokens left out of it) and the chat format with its tool calls; a config.json without model_type is recognized by its architecture.
- The GGUF matmul test no longer depends on which tests run beside it.
- Fish Audio Drama 3: the prompt cases of the reference check.
- Fish Audio Drama 3: the catalog entry, docs/fish-drama.md with the measured agreement (fish-speech's PyTorch reference on small checkpoints and on the released weights), speed and memory, the models table, the README index and the server's speech row.
- Fish Audio Drama 3: the server's memory estimate takes the checkpoint as stored whatever quant says (its FP8 experts are not requantized).
- Fish Audio Drama 3: the checks against fish-speech's PyTorch reference (examples/fish_check.rs dumps prompt rows, frames, logits and codec output; tools/ref_fish_drama.py compares; tools/fish_drama_tiny.py writes a small random checkpoint with bf16 or FP8 experts).
- Fish Audio Drama 3: --speak on the command line (--ref-audio with --ref-text, --voice as the speaker tag, --quant, --mmap; on the CPU) and the server's /v1/audio/speech.
- Fish Audio Drama 3: the model (fish_omni): the prompt rows, the backbone on the Qwen3.5 MoE code with its FP8 experts as stored, the codebook tower with its caches over the sequence, the reference's sampler and chunking of long texts, and the codec (the causal DAC decoder, and the encoder for reference clips).
- FP8 E4M3 rows are converted from their bits instead of through the table, in a form that vectorizes (AVX2 / AVX-512 where the CPU has them): the same values bit for bit, twice as fast on the conversion.
- catalog entries, the model table row, docs/ripple.md and its line in the README index.
- the model (sliding and full attention layers on Gemma's attention, gated attention, value residuals, a clamped MLP, an embedding scale) from Ricursive's modeling code, bf16 and NVFP4 checkpoints, CLI / engine dispatch, the check example.
- its chat format (untrimmed ChatML turns, JSON tool calls), held against the template's jinja rendering, and replies whose reasoning and tool calls are marked in plain text filtered in the text stream.
- catalog entry, the model table row, docs/raon.md and its line in the README index.
- the model (delta attention with a write gate, gated MLA, four Raon_mHC residual streams, per-layer embeddings, tied head) from KRAFTON's modeling code, CLI / engine dispatch and a check example.
- its chat format (Qwen3.5's ChatML template with its own no-thinking prompt), held against the template's jinja rendering.
- Kimi's KDA and gated MLA layers as functions other families can call, with what Raon-1's delta attention adds (a write gate, the decay's own rank, queries and keys normed as F.normalize does).
- the catalog entry, the model table row, docs/ocr.md and its line in the README index.
- the comparison with the checkpoint's own code (tools/ref_unlimited_ocr.py: token ids, pixel values, encoder features, packed embeddings, logits, the greedy reply).
- the command line (--image per page, the reply with its markup, the decoding rules as flags), the engine family and with it the OpenAI server's image input.
- the model from Baidu's own code: page views (global view and 640 tiles), the SAM ViT-B into CLIP ViT-L encoder and its projector, the MoE decoder with DASH block-sparse attention (per-layer router, entmax prior, page candidates), two-axis page RoPE, the page count guard and the no-repeat n-gram rule.
- Chat formats can ask for replies shown with their special tokens (ChatFormat::keeps_special_tokens), for models whose markup is made of them.
- documentation (docs/inkling.md, the models table, the README index) and the --models catalog entries.
- the tiny checkpoint builder with transformers' reference outputs (tools/inkling2_tiny.py: floats or packed NVFP4 / MXFP8 experts, chat prompts, an image), the reference for the vision tower alone (tools/ref_inkling2_vision.py) and the comparison (examples/inkling2_check.rs).
- Inkling-2 from the CLI and the engine: its chat format (typed messages, the thinking effort, tool declarations, calls and results, images as messages of their own), message headers left out of the text of a reply, and images through the vision tower.
- Inkling-2 (inkling2, thinkingmachines/Inkling-2-Small): the text decoder (attention with QK L2 clips, a relative-position bias, sinks and an output gate, short convolutions, sliding-window and full layers, the attention residual over block states, a soft-capped MoE and logits) and the vision tower (40-pixel tiles without resizing); experts read as the releases store them, interleaved gate / up rows in NVFP4, MXFP8 or floats.
- Nemotron 3.5: catalog entries, docs/nemotron.md with the measured agreement (transformers on tiny checkpoints and on the real Transcribe weights, llama.cpp on the real Lightning Q4_0), the models table and the README index.
- Nemotron 3.5 Transcribe: the audio side of the speech LM (NeMo's features, the speech recognition encoder, the Sortformer speaker encoder with its speaker cache, the projection), --transcribe and the server's transcription endpoints through whisper::Transcriber, with reference checks on a tiny checkpoint and on the real weights.
- Nemotron 3.5: the Mamba-2 time step follows the config's time_step_limit (no clamp, as NVIDIA's training and serving code and llama.cpp; transformers' scan floors it at time_step_min, which a session can reproduce); a GGUF block is MoE when it has a router; no prompt-lookup rounds on the command line; a tiny GGUF writer.
- Nemotron 3.5: the chat template (always a system turn, XML tool declarations, <think>, Qwen-style tool calls) and the command line and engine wiring.
- Nemotron 3.5: the Nemotron-H text model (nemotron_h: Mamba-2, attention without positions, MoE with relu^2 experts), the release's, NeMo's and GGUF tensor layouts, with a tiny-checkpoint reference check.
- Kimi K3.1: catalog entries, the model table row, docs/kimi.md and its line in the README index.
- Kimi K3.1: the text model (KDA, gated MLA with the K3.1 indexers, cross-layer residuals, latent MoE) from Moonshot's modeling code, safetensors and GGUF layouts, the tiktoken tokenizer, CLI / engine dispatch and a check example.
- Kimi K3.1: its chat format (encoding_k3.py: tagged messages, think / response / tools blocks, tool declarations and calls) and replies filtered by tag in the text stream.
- Kimi K3.1: the SiTU expert activation, compressed-tensors MXFP4 weights repacked into ggml blocks, tiktoken-ranked BPE merges, Mistral Large 4's MLA helpers shared.
- Fish Audio Drama 3: what the checkpoint holds and what each stage would need (docs/fish-drama.md); the model is not supported.
- the catalog entry and the documentation (docs/zonos.md).
- an example and a script that compare the tokenizer, the prompt, greedy generation, the decoder's waveform and the speaker embedding with Zyphra's PyTorch reference.
- --speak on the command line and the server's /v1/audio/speech.
- the language model, the ZAC 1 codec decoder and the speaker encoder of Zyphra/ZONOS3-Mini from its GGUF files (oxinfer::zonos).
- ERNIE 5.5: documentation (docs/ernie.md, the models table, the README index) and the --models catalog entries.
- ERNIE 5.5 from the CLI and the engine: the base model's prompt format (a text to continue, no system role), checkpoints whose shards have no index, and the tokenizer and the MTP layer in deepseek4_check --load.
- Tokenizer regex: a plus after an interval repeats it (Oniguruma's reading, which the Python tokenizers library has), so that ERNIE 5.5's tokenizer groups digits by three from the right.
- ERNIE 5.5: the reference in PyTorch (tools/ref_ernie55.py, from Baidu's vLLM plugin on DeepSeek's inference code), the tiny checkpoint builder, and the MTP layer in deepseek4_check.
- ERNIE 5.5 (ernie5_5_moe, baidu/ERNIE-5.5-Flash-Base) on the DeepSeek V4 model: HCA layers with the head mix and the output gate, MLA layers with the DSA indexer, the SiTU dense layer and latent MoE, MultiMax, the MTP layer and a greedy decoder with its drafts; bf16 and FP8 checkpoints in Baidu's naming.
- residency requested from eight sets in parallel; Flash-Next plans build concurrently
- Flash-Next prefill plans: QSA score rows in a 32 MB ring instead of 512 MB
- Metal MoE: grouped GEMM tiles of 128 / 32 / 16 rows; Q4_K and Q5_1 read as words
- allow clippy's const thread-local false positive on Android
- the fine-tunes run as smoke tests, and the count of types in the GSQ-RCO file.
- Catalog and docs: the GSQ-RCO and Ternary-Bonsai 2 GGUFs, what was checked on the trending Qwen and Gemma 4 derivatives, and what the Edge0 and E384-FP4 layouts would need.
- An example that dumps a Qwen model's greedy run in llama_logits_dump's format and compares two such runs.
- The AVX2 group products stop at the block for ggml blocks of 64 and 128 elements.
- AVX2 unpackers for the 1-bit and ternary GGUF types: 4 to 6 times fewer nanoseconds per weight than the portable decoder.
- DSpine drafts (tencent/Qwen3.8-27B-DSpine-b16) draft through their code bridge: a write after every draft layer, the last one redone per slot with the token drafted before it.
- A .gguf file without the GGUF magic is reported as damaged instead of as an unknown repo id.
- Qwen4-Exp GGUF files load again: the MiniMax lightning-indexer loader no longer claims their QSA indexer weights.
- Hadamard-folded GGUF checkpoints (prism.hadamard metadata, Ternary-Bonsai 2) run: folded weights get their input sign-flipped and rotated, the latent token embedding is rotated back.
- GGUF files with the 1-bit and ternary ggml types load: Q1_0, Q2_0, TQ1_0, TQ2_0 and the PrismML fork's PQ2_0 and PTQ1_0.
- catalog entries, docs/glm.md with the measured agreement (transformers on tiny checkpoints, llama.cpp on the real UD-Q2_K_XL), the models table and the README index; glm5_check --dump.
- no prompt-lookup rounds for GLM-5.3-Flash on the command line, whose recurrent state cannot roll a refuted draft back; the tiny GGUF writer copies a published file's tokenizer.
- the chat template (reasoning effort line, thinking, tool calls) and the command line, engine and streaming wiring.
- GGUF files keep -exp(A_log) in ssm_a, GLM-5.3's latent norms have their own epsilon; a tiny GGUF writer, a per-layer check and unit tests.
- the text model of GLM-5.3-Flash (glm5_next) and GLM-5.3 (glm_moe_dsa), with a tiny-checkpoint reference check.
- the transparent-picture run and the RGBA rule in the docs, a note in the C header.
- documentation (docs/qwen-image.md, models, the conditioning encoder) and the --models catalog entries.
- the session's DeepStack rows behind a box (the session stays under clippy's variant-size bound), an RGBA PNG only for a transparent region.
- editing with one or more input images (through the text encoder's vision tower and, as latent tokens, the transformer's sequence); an FP8 single file and edits in the tiny-checkpoint tool.
- Qwen3-VL DeepStack vision features for the conditioning encoder (qwen_vl): the tower's intermediate features through their mergers, added to the image tokens after the language model's first layers.
- Qwen-Image-2.1 and -2.1-Turbo: text to image through the Qwen-Image pipeline (Qwen3-VL text encoder, saved Turbo sigmas, GGUF transformers with or without metadata, LoRAs, RGBA output).
- the single-stream block-causal transformer, the prompt's keys and values computed once and reused by every step.
- the residual RGBA VAE (AutoencoderKLQwenImage21: 64 latent channels, 16x, a wider decoder, group shortcuts).
- DeepSeek V4.1 Flash: an engram table per layer (blk.N.engram_embd) is read too; where the two GGUF layouts' correspondence comes from.
- DeepSeek V4.1 Flash: catalog entries, the model table row and its section of the DeepSeek V4 page.
- DeepSeek V4.1 Flash: deepseek4_check --load opens a checkpoint without running it (the released GGUF layouts, headers only) and prints its tokenizer's ids.
- DeepSeek V4.1 Flash: a tiny GGUF writer in the layouts of the released conversions, integer GGUF tensors for the engram hash, image spans in the check example.
- DeepSeek V4.1 Flash: its chat format (a system token, the reasoning effort line, DSML tags with a leading space) and the CLI / engine dispatch.
- DeepSeek V4.1 Flash: the text model on the DeepSeek V4 code, from DeepSeek's naming and from deepseek41 GGUF files.

## [0.0.10](https://github.com/KarpelesLab/oxinfer/compare/v0.0.9...v0.0.10) - 2026-11-13

### Other

- httpsd 0.2.1; the static fullrust binary with the server again

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
