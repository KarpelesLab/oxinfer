# Build and run

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
| `--tools` | let the model call built-in tools: `get_current_datetime` and `calculate` (see [Tool calling](tool-calling.md)) |
| `--print-prompt` | print the fully formatted prompt and exit |
| `--raw` | feed the prompt as-is after `<bos>`, no chat template |
| `--max-tokens N` | cap on generated tokens |
| `--threads N` | worker threads (default: all cores) |
| `--quant q8` | quantize streamed weights to int8 at load (blocks of 32, f32 scales); ~1.8x less memory traffic |
| `--quant q4` | GGUF-style Q4_0 (blocks of 32, 4-bit, f16 scale) for all streamed weights; ~3.5x less traffic |
| `--draft DIR` | speculative decoding with Google's `gemma-4-E4B-it-assistant` drafter (greedy only; output identical to plain greedy), or a DSpark draft (greedy and sampled): Mistral Large 4 / DeepSeek V4 a draft GGUF (default: the `dspark*.gguf` beside the model, which `auto` requires), Qwen / Gemma 4 a Hugging Face draft or dflash GGUF, see [DSpark](mistral-large-4.md#dspark-speculative-decoding) and [Qwen and Gemma 4 drafts](qwen.md#dspark-drafts-for-qwen-and-gemma-4) |
| `--no-draft` | no DSpark draft (plain decoding) |
| `--draft-tokens N`, `--spec-draft-n-max N` | max draft tokens per round (default 6 with `--draft`, 3 with `--mtp`, 8 with `--mtp --stream`; the length adapts; DSpark: Mistral Large 4 / DeepSeek V4 3, Qwen / Gemma 4 the whole block) |
| `--spec-draft-fixed` | DSpark (Qwen, Gemma 4): always draft `--draft-tokens` instead of choosing each round's length by measured cost |
| `--spec-draft-conf-min C` | DSpark: cut a block before the first draft whose confidence-head estimate is below C (default 0: off) |
| `--spec-draft-p-min P` | DSpark: cut a block before the first draft the draft model gives less than probability P (default 0.85 when the model streams, else 0) |
| `--exact-verify` | DSpark / MTP: verify with the one-token kernels, so greedy output is the plain greedy output token for token (default: fast batched matrix kernels on the GPU, whose logits differ from plain decoding in rounding; `--fast-verify` asks for them explicitly), see [fast verification](qwen.md#fast-verification) |
| `--mtp` | Qwen3.5/3.8: speculative decoding with the checkpoint's multi-token-prediction layer (greedy and sampled) |
| `--mtp-fixed` | with `--mtp`, always draft `--draft-tokens` instead of adapting between 1 and N |
| `--mtp-from DIR` | `--mtp` with the MTP weights taken from another checkpoint (e.g. the original of a converted copy) |
| `--mmap` | map weights from disk instead of copying them into RAM |
| `--save-quantized DIR` | write a pre-quantized, text-only checkpoint at `--quant` precision (see [Pre-quantized checkpoints](performance.md#pre-quantized-checkpoints)) and exit |
| `--devices` | list detected CPU features and GPUs (Metal, CUDA, Vulkan) and exit |
| `--models` | list the tested models with what this machine can run and an estimated decoding speed (see [What runs on this machine](#what-runs-on-this-machine)) and exit |
| `--device D` | `auto` (default: the first GPU backend that starts, Metal or CUDA before Vulkan, else the CPU), `cpu`, `metal`, `cuda`, `vulkan`; a device asked for by name that is missing or cannot start is an error, not a fallback |
| `--gpu-decode on\|off` | generate a whole token per GPU submission (default `on` where supported; see [Decoding on the GPU](performance.md#decoding-on-the-gpu)); engine / speaker option `gpu_decode` |
| `--prefill-chunk N` | tokens per prefill chunk (default 256 on the CPU, 1024 when a GPU backend is active) |
| `--profile` | per-operation timing and bandwidth breakdown |
| `--dump-logits FILE` | write the logits for the last prompt token (for reference checks) |
| `--show-tokens` | print token ids |

By default the whole safetensors file is read into RAM at startup (a few seconds), which
gives steady decode speed from the first token. `--mmap` starts faster but pays page
faults on the first pass.

## Release binaries and self-update

Each GitHub release carries the `oxinfer` binary (`.github/workflows/release-binaries.yml`,
dispatched by release-plz on the new tag):

| archive | built with | notes |
|---|---|---|
| `oxinfer-<tag>-linux-x86_64-static.tar.gz` | the [fullrust](https://github.com/KarpelesLab/fullrust) toolchain | a static executable with no libc and no interpreter: it needs only a Linux kernel. CPU only (it cannot load the CUDA / Vulkan libraries) |
| `oxinfer-<tag>-linux-x86_64.tar.gz`, `-linux-aarch64.tar.gz` | stable Rust, glibc | all default features |
| `oxinfer-<tag>-windows-x86_64.zip`, `-windows-aarch64.zip` | stable Rust, MSVC | |
| `oxinfer-<tag>-macos-universal.tar.gz` | stable Rust, both Apple architectures in one binary | |

```sh
oxinfer --check-update    # compares this binary with the latest release
oxinfer --update          # downloads this platform's archive and replaces the executable
```

`--update` (`src/update.rs`, the `update` feature, on by default) reads the latest release from
the GitHub API, downloads the archive named for the platform the binary was built for, unpacks
it in memory (gzip with [compcol](https://crates.io/crates/compcol), the tar or zip container
with [fstool](https://crates.io/crates/fstool)), writes the new executable beside the running
one and swaps them by renames (on Windows the previous one stays as `oxinfer.exe.old` while it
runs). `--force` reinstalls the latest release. What vouches for the download is GitHub over TLS
and the size the API lists: the releases are not signed and no digest is checked. A binary built
from source for a platform without an archive (Android, the BSDs, musl) reports that instead.

## What runs on this machine

```sh
./target/release/oxinfer --models                 # the tested models, one line each
./target/release/oxinfer --models --all --kind chat   # every precision of the chat models
./target/release/oxinfer --models --model gemma --json  # filtered by name, as JSON
./target/release/oxinfer --models --device cpu     # as if there were no GPU
./target/release/oxinfer --models --can image-input,tools   # by capability
./target/release/oxinfer --models --kind image --can inpainting
```

`--models` checks the models of the [model table](models.md) against the machine and exits: the download
against the free space of the cache directory (`on disk` when the repo is already there, `N GB
short` when it does not fit), the memory each precision needs against the RAM and the GPU's
memory, where it would decode, an estimated decoding speed, and what the model can do. On a
Threadripper 9970X with an RTX 5080 (16 GB), 134 GB of RAM (the `CAN` column cut here):

```
CHAT                                                    DISK     MEMORY  RUNS ON                       DECODE
Qwen/Qwen3.5-0.8B                          bf16         on disk  1.7 GB  GPU                           ~312 tok/s
bartowski/Llama-3.2-1B-Instruct-GGUF:Q8_0  GGUF Q8_0    on disk  1.3 GB  CPU                           ~39 tok/s
google/gemma-4-E4B-it                      --quant q8   16 GB    12 GB   GPU                           ~104 tok/s
Qwen/Qwen3.5-9B                            --quant q8   on disk  12 GB   GPU                           ~71 tok/s
mlx-community/gemma-4-26b-a4b-it-4bit      MLX 4-bit    on disk  15 GB   CPU                           ~16 tok/s
Qwen/Qwen3.8-27B                           --quant q4   56 GB    19 GB   CPU                           ~4.6 tok/s
unsloth/DeepSeek-V4-Flash-GGUF:UD-Q2_K_XL  GGUF ...     97 GB    97 GB   storage (over 60% of memory)  storage-bound
```

One line per model shows the precision to pick: the most precise one that decodes in memory at
10 tokens/s or more, else the fastest in memory, else the smallest; `--all` lists `bf16`,
`--quant q8` and `--quant q4` of every safetensors checkpoint. The rules are the engine's: weights
over 60% of the RAM are streamed from storage (they run, at a speed set by the storage, which is
not estimated), and whole decode steps run on the GPU when the backend has decode plans for the
weight format and the weights fit in 92% of its memory with 1 GB to spare. Embedding, image,
Whisper and TTS models get the disk and memory verdicts only, and so do the chat models whose
bytes per token were not measured (the MiniMax family, Qwen2.5-VL, the Jev decision models, the
untested Gemma 4 12B / 31B, ...).

The `CAN` column (`"capabilities"` in the JSON) lists what the README states for each model, and
`--can A,B` keeps the models that have all of them: `text`, `image-input`, `video-input`,
`audio-input`, `thinking`, `tools`, `decisions`, `embeddings`, `text-to-image`, `image-to-image`,
`image-edit`, `inpainting`, `control`, `speech-to-text`, `translation`, `text-to-speech`,
`voice-cloning`. The catalog is the [model table](models.md) with one entry per repository or file (79 entries),
the MiniMax models included; it is a list kept by hand in `src/advisor.rs`, not read from the
checkpoints, so a fine-tune or another quantization of a listed model is not in it.

The speed is the cost model the measurements follow (`src/advisor.rs`, `oxinfer::advisor`): a
decoded token reads every dense weight once, so

```
seconds per token = bytes per token / rate(weight format) + layers x cost per layer
```

- **CPU.** The rates are measured when the command runs: the single-token kernel of each format
  on a random matrix of 268M weights (larger than the caches), about 0.1 s each, 2 to 3 s and up
  to 0.9 GB in all; they are printed (`CPU kernels, GB/s: ...`). Two constants were fitted on the
  9970X against 200 greedy tokens of ten checkpoints: 0.35 ms per layer (1.1 ms for a
  mixture-of-experts layer), and the ratio between a kernel's rate inside a model and on the
  synthetic matrix (1.2 for bf16, 1.1 for q8, 0.8 for q4, 0.78 for GGUF). Measured against
  predicted there, tokens/s on the CPU:

  | checkpoint | measured | predicted |
  |---|---|---|
  | Qwen3.5-0.8B bf16 / q8 / q4 | 33.0 / 49.1 / 73.2 | 39.5 / 55 / 71.5 |
  | Qwen3.5-9B bf16 / q8 / q4 | 5.9 / 9.7 / 14.6 | 5.3 / 8.9 / 14.2 |
  | Qwen3-4B MLX 4-bit | 19.5 | 22.5 |
  | Llama 3.2 1B GGUF Q4_K_M / Q8_0 | 60.7 / 39.4 | 60.3 / 39.5 |
  | Gemma 4 26B-A4B MLX 4-bit | 15.9 | 15.9 (the MoE constant is fitted on it) |
  | Orpheus 3B GGUF Q4_K_M as a text model (not part of the fit) | 25.8 | 26.1 |

  The machine was shared while these were measured (load average 10 to 60), and the kernel
  rates move with the load by 10 to 30%, so does the estimate. No other CPU was measured: on an
  Apple or a two-channel desktop CPU the kernel rates are measured the same way, but the two
  fitted constants are this machine's.
- **GPU.** A GPU cannot be timed without loading a model, so the rates are those of one device
  per backend. CUDA, on the RTX 5080: 850 / 740 / 540 GB/s for bf16 / q8 / q4 and 0.06 ms per
  layer, which gives 312 / 387 / 450 tokens/s for Qwen3.5-0.8B (measured 304 / 368 / 421), 71 /
  98 for Qwen3.5-9B q8 / q4 (measured 71.6 / 101.4) and 152 for Qwen3-4B 4-bit (measured 171).
  GGUF files and the 26B-A4B mixture of experts decoded at the CPU's speed with `--device cuda`
  (53.7 and 17.1 tokens/s against 57.0 and 16.1), so they are estimated on the CPU. Metal: 400
  GB/s and 0.03 ms per layer, from the M5 Max figures of
  [Decoding on the GPU](performance.md#decoding-on-the-gpu) (not measured again). On another device of the
  backend the same rates are used and the estimate is marked `?` (`"basis":
  "gpu-reference-other-device"` in the JSON): it is the reference device's speed, not this
  one's. Vulkan has no decode plans, so its models are estimated on the CPU.
- **What is not modelled:** speculative decoding (MTP, DSpark, prompt lookup), long contexts
  (attention grows with the context), prefill speed, thermal limits, and a GPU shared with other
  programs. Qwen3.5-9B in bf16 (15.9 GB per token) with `--device cuda` on the 16 GB card ran at
  3.7 tokens/s, slower than the CPU's 5.9: the table sends it to the CPU, the engine's `auto`
  does not yet.

The bytes per token, layers and sizes of the catalog come from the loader's own report for the
checkpoints loaded here (Qwen3.5-0.8B and 9B, Qwen3-4B 4-bit, Gemma 4 26B-A4B 4-bit, Llama 3.2
1B), from the tables of this README for Gemma 4 and Qwen3.8-27B, and from the parameter counts
for Qwen3.6-35B-A3B, Qwen3.8-Flash-Next, DeepSeek V4 Flash and Mistral Large 4 (noted in the
JSON). In the library: `advisor::Hardware::detect`, `advisor::survey`, `advisor::render`,
`advisor::to_json`; free disk space is read with `statvfs` (Linux, macOS, iOS, Android; unknown
elsewhere).

## As a library

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
`Engine::set_interventions` (see [Runtime interventions](interventions.md#runtime-interventions-neuron-edits-steering-abliteration-sae-features)).
`Engine::save_session` / `Engine::load_session` write and restore a conversation's caches, and the
`prompt_cache` option keeps them in a directory for `Engine::chat` (see
[Session snapshots and the prompt cache](sessions.md#session-snapshots-and-the-prompt-cache)); prompt-lookup
drafting is the `lookup` options ([Prompt lookup](performance.md#prompt-lookup)).

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

## Using oxinfer from oxideav (and other crates)

Images, audio and video in the API are [oxideav](https://crates.io/crates/oxideav-core) types:
decoded `VideoFrame`s and `AudioFrame`s with the `CodecParameters` that describe them, as an
oxideav pipeline hands them from stage to stage. Two features split what a dependent pulls in:

| Feature | Brings | Gives |
|---|---|---|
| `oxideav` | `oxideav-core` only | `oxinfer::av` and the in-memory inputs and outputs below; `oxinfer::oxideav_core` re-exports the crate |
| `media` (default, implies `oxideav`) | oxideav-pipeline, oxideav-image and the codec and container crates (PNG, JPEG, WebP, GIF, BMP, MP3, FLAC, AAC, Vorbis, Opus, WAV, MP4, MOV, Matroska / WebM, Ogg, AVI, H.264, H.265, VP8, VP9, AV1, MPEG-4 Part 2) | the path-based inputs (`--image`, `--video`, `--audio`, `transcribe_file`, `Media`) |

`hf` (HuggingFace fetching), `gpu` (run-time GPU backends) and `server` (default: the
[OpenAI-compatible server](server.md#openai-compatible-server), with httpsd) are independent of both. A crate
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
