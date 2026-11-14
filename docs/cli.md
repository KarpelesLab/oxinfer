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
oxinfer --auto-update --model google/gemma-4-E4B-it -p "Hi"   # the same check before a normal run
```

`--auto-update` goes with any other command (`serve` included). Before the command runs it asks
GitHub for the latest release, at most once an hour (a stamp file in the cache directory). A
newer release is installed over the executable when that is possible, and the run in progress
continues as the old version: the next one is the new. When it is not possible (the directory
of the executable is not writable, no binary is published for the platform, the download
fails) it prints one line saying which version is available and why it was not installed. A
check that fails (no network, GitHub's rate limit) is reported the same way and never stops the
command.

`--update` (`src/update.rs`, the `update` feature, on by default) reads the latest release from
the GitHub API, downloads the archive named for the platform the binary was built for, unpacks
it in memory (gzip with [compcol](https://crates.io/crates/compcol), the tar or zip container
with [fstool](https://crates.io/crates/fstool)), writes the new executable beside the running
one and swaps them by renames. Windows cannot delete a program that is running: there the
previous one is renamed to `oxinfer.exe.old` and marked hidden, and the next start of oxinfer
deletes it. `--force` reinstalls the latest release. What vouches for the download is GitHub over TLS
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
