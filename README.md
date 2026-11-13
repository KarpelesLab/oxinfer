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
embeddings, Stable Diffusion, FLUX.1 Kontext, Qwen-Image-Edit, OmniGen2, BAGEL and Step1X-Edit image editing, Whisper speech to text, Orpheus and ZONOS3-Mini text to speech. No BLAS, no ML framework: weights are read straight out
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
   [Streaming weights from flash](docs/streaming.md#streaming-weights-from-flash-models-larger-than-ram)), very slowly.
2. **Then make it fast.** Anything that improves speed on a given machine is in scope: a GPU
   when there is one (Metal, CUDA, WebGPU, detected at run time), the Neural Engine through Core
   ML, quantization, SIMD kernels per CPU (NEON, SDOT / i8mm, AVX2, wasm simd128), speculative
   decoding (MTP, drafters, prompt lookup), prompt caches restored from disk, prefetching and
   caching for weights that do not fit in memory. When
   something is missing, oxinfer falls back to a slower path rather than failing.

Correctness comes first in both: every model and every fast path is checked against the reference
implementation (the [verification](docs/verification.md#verifying-against-the-reference) section and the per-model
notes give the measured agreement), and a speed-up that changes the output must say so and stay
optional.

## Quick start

```sh
cargo build --release          # or take a binary from the GitHub releases
./target/release/oxinfer --models                                   # what this machine can run, with estimated speeds
./target/release/oxinfer --model google/gemma-4-E4B-it -p "Explain RMSNorm in two sentences."
./target/release/oxinfer --model google/gemma-4-E4B-it --chat       # interactive, multi-turn
./target/release/oxinfer --model google/gemma-4-E4B-it --image photo.jpg -p "What is this?"
./target/release/oxinfer --model openai/whisper-small --transcribe talk.mp3
./target/release/oxinfer serve --model google/gemma-4-E4B-it        # OpenAI-compatible HTTP server
```

A Hugging Face repo id is downloaded to the cache on first use; a local folder or GGUF file works
too. The device is chosen at run time (`--device auto`: a GPU when one starts, else the CPU).
[docs/cli.md](docs/cli.md) has the flags, the release binaries, `--update` and `--models`;
[docs/library.md](docs/library.md) the Rust and C API.

## Documentation

**Using it**

| | |
|---|---|
| [Command line](docs/cli.md) | building, flags, release binaries and self-update, `--models` |
| [Library use](docs/library.md) | the Rust and C API, features, use from other crates |
| [Supported models](docs/models.md) | every tested checkpoint with its size and measured agreement with the reference |
| [Getting weights](docs/weights.md) | Hugging Face downloads, the mirror, GGUF files, MLX checkpoints |
| [OpenAI-compatible server](docs/server.md) | chat, embeddings, images, audio; structured outputs, forced tool calls, log probabilities |
| [Tool calling](docs/tool-calling.md) | built-in and caller-defined tools |
| [Session snapshots and the prompt cache](docs/sessions.md) | saving and restoring a conversation's state |
| [Embeddings](docs/embeddings.md) | dense, sparse and ColBERT embedding models |
| [Typed decisions](docs/decisions.md) | the Jev decision models, over text and images |
| [Runtime interventions](docs/interventions.md) | neuron edits, steering, abliteration, SAE features |

**Text models**

| | |
|---|---|
| [Gemma 4 26B-A4B](docs/gemma-4-26b.md) | the mixture-of-experts Gemma 4 |
| [Qwen3.5 / Qwen3.8](docs/qwen.md) | dense and MoE Qwen, MTP and DSpark speculative decoding |
| [Mistral Large 4](docs/mistral-large-4.md) | 1T parameters streamed from storage, FP8 on the GPU |
| [DeepSeek V4 Flash, V4.1 Flash](docs/deepseek-v4.md) | 284B MoE as GGUF; V4.1 (748B with its engram tables) on tiny checkpoints only |
| [GLM-5.3 and GLM-5.3-Flash](docs/glm.md) | hyper-connections, KDA linear attention, MLA with the DSA indexer; text only |
| [ERNIE 5.5 Flash Base](docs/ernie.md) | 280B MoE base model (HCA and MLA-DSA attention); on tiny checkpoints only |
| [Kimi K3.1 Pro, Kimi K3](docs/kimi.md) | 2.8T MoE with linear and sparse attention, on tiny checkpoints only |
| [LongCat-2.5, LongCat-2.0, LongCat-Flash-Lite-Sparse](docs/longcat.md) | 1.6T MoE with sparse attention and a hashed n-gram embedding; the 69B sibling on its real weights, LongCat-2.5 on tiny checkpoints only |
| [Nemotron 3.5 Lightning and Transcribe](docs/nemotron.md) | Mamba-2 / attention / MoE hybrid; a 34B speech LM for speech to text |
| [Inkling-2 Small](docs/inkling.md) | 594B MoE with images (relative-bias attention, short convolutions, attention residual); the decoder on tiny checkpoints only |
| [Raon-1 nano](docs/raon.md) | delta-rule linear attention and gated MLA on four residual streams, per-layer embeddings |
| [Ripple-65B Preview](docs/ripple.md) | dense 65B with sliding and full attention layers, gated attention, value residuals; run as NVFP4 |
| [Qwen3.8-Flash-Next](docs/qwen-flash-next.md) | 125B MoE with an n-gram embedding table |
| [Unlimited-OCR-2](docs/ocr.md) | document OCR: page images to text with layout, block-sparse attention |
| [IBM Granite 4 and TwIL-8B](docs/granite.md) | Granite 4.1 / 4.2 and their fine-tunes, safetensors and GGUF |
| [Image, video and audio input](docs/multimodal-input.md) | the vision and audio towers of Gemma 4 and Qwen, Qwen2.5-VL |

**Images and audio**

| | |
|---|---|
| [Stable Diffusion 1.x / 2.x / XL](docs/stable-diffusion.md) | text to image, image to image, inpainting, InstructPix2Pix |
| [Stable Diffusion 3 / 3.5](docs/sd3.md) | text to image, image to image |
| [FLUX.1](docs/flux.md) | Kontext editing, schnell / dev, Fill, Canny / Depth, Redux |
| [Qwen-Image, Qwen-Image-Edit and Qwen-Image-2.1](docs/qwen-image.md) | editing, text to image, ControlNet, layers; Qwen-Image-2.1 / 2.1-Turbo |
| [OmniGen2](docs/omnigen2.md), [BAGEL](docs/bagel.md), [Step1X-Edit](docs/step1x-edit.md), [Janus-Pro](docs/janus.md) | instruction editing, unified generation and understanding |
| [Whisper](docs/whisper.md) | speech to text |
| [Nemotron 3.5 Transcribe](docs/nemotron.md#speech-to-text-nemotron-35-transcribe) | speech to text and speech translation with a speech LM |
| [Orpheus](docs/orpheus.md) | text to speech, batches, voice cloning, the SNAC codec |
| [ZONOS3-Mini](docs/zonos.md) | text to speech at 44.1 kHz in 43 languages, default voices, cloning from a clip |
| [Audio8 TTS](docs/audio8-tts.md) | text to speech at 44.1 kHz in 23 languages, cloning from a clip and its transcript, its codec |
| [Haru-4B](docs/haru.md) | full-duplex speech to speech at 24 kHz: a recording of the user in, what the model says meanwhile out |
| [Fish Audio Drama 3](docs/fish-drama.md) | text to speech at 44.1 kHz with a 35B MoE, several speakers per text, cloning from a clip and its transcript |

**Under the hood**

| | |
|---|---|
| [Performance](docs/performance.md) | measurements, GPU decoding, comparison with llama.cpp, quantization |
| [Speculative decoding](docs/speculative-decoding.md) | drafters, MTP, multi-token passes, prompt lookup |
| [Streaming weights from flash](docs/streaming.md) | models larger than RAM, mixed GPU / CPU decoding |
| [Platforms and accelerators](docs/platforms.md) | what runs where; CUDA, Vulkan |
| [WebAssembly and the browser](docs/wasm.md) | WASI, WebGPU |
| [Core ML / Apple Neural Engine](docs/coreml.md) | running and converting Core ML models |
| [How it works](docs/internals.md) | the layout of the code |
| [Verifying against the reference](docs/verification.md) | how outputs are compared with `transformers` and others |
| [Core ML model format](docs/coreml-format.md), [llama.cpp notes](docs/llama-modchip/README.md) | reference notes |

## License

MIT, see [LICENSE](LICENSE). Model weights carry their own licenses.
