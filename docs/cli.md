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
| `--threads N` | worker threads (default: one per physical core; every logical CPU where SMT siblings cannot be told apart) |
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
| `--lora FILE[:SCALE]` | Qwen, Llama, Gemma, Inkling-2: a LoRA adapter applied at run time (repeatable; scale 1, as llama.cpp's `--lora-scaled`): a PEFT folder or a llama.cpp GGUF adapter; image models: a LoRA merged into the pipeline, as before (see [LoRA adapters](lora.md)) |
| `--lora-merge` | merge the language-model LoRAs into the weights at load (float weights exact, quantized ones re-quantized) |
| `--mmproj FILE\|none` | the vision projector of a GGUF model (Qwen3.5-family, Mistral Large 4, MiniMax M3, Inkling-2): a llama.cpp `mmproj` GGUF, a path or `org/repo/file`; default: the `mmproj*.gguf` beside the model. `--model org/repo:QUANT` fetches the repository's top-level `mmproj*.gguf` with the model, with a note (BF16, else F16, else F32, else another; never files in subfolders such as `Old_Support/`), unless one is beside the model already; `none` turns images off and fetches none, and a given file is used instead (see [Getting weights](weights.md)) |
| `--mmap` | map weights from disk instead of copying them into RAM |
| `--save-quantized DIR` | write a pre-quantized, text-only checkpoint at `--quant` precision (see [Pre-quantized checkpoints](performance.md#pre-quantized-checkpoints)) and exit |
| `--devices` | list detected CPU features and GPUs (Metal, CUDA, Vulkan) and exit |
| `--models` | list the tested models with what this machine can run and an estimated decoding speed (see [What runs on this machine](#what-runs-on-this-machine)) and exit |
| `--device D` | `auto` (default: the first GPU backend that starts, Metal or CUDA before Vulkan, else the CPU), `cpu`, `metal`, `cuda`, `vulkan`; a device asked for by name that is missing or cannot start is an error, not a fallback |
| `--gpu-decode on\|off` | generate a whole token per GPU submission (default `on` where supported; see [Decoding on the GPU](performance.md#decoding-on-the-gpu)); engine / speaker option `gpu_decode` |
| `--gpu-chain on\|off` | greedy decoding chained on the device where tokens decode whole on the GPU: the argmax and the next token's embedding on the GPU, the next pass queued before the host reads a token (default `on`; the same tokens; see [Greedy decoding chained on the device](performance.md#greedy-decoding-chained-on-the-device-november-2026)); engine option `gpu_chain`, `OXINFER_GPU_CHAIN=0` |
| `--gpu-kv f16\|f32` | how the device holds its copies of the attention caches where whole-pass GPU plans run (Metal with matrix units): half-precision rows (default; half the attention's memory traffic at long contexts, as llama.cpp's f16 cache) or f32 ones; QSA caches stay f32 (see [Long contexts](performance.md#long-contexts-november-2026)); engine / server / C API option `gpu_kv`, `OXINFER_GPU_KV=f32` |
| `--prefill-chunk N` | tokens per prefill chunk (default 512 on the CPU, 1024 when a GPU backend is active) |
| `--profile` | per-operation timing and bandwidth breakdown |
| `--dump-logits FILE` | write the logits for the last prompt token (for reference checks) |
| `--show-tokens` | print token ids |

By default the whole safetensors file is read into RAM at startup (a few seconds), which
gives steady decode speed from the first token. `--mmap` starts faster but pays page
faults on the first pass.

## Release binaries and self-update

The installers download the latest release's binary for this machine:

On macOS and Linux:

```sh
curl -fsSL https://karpeleslab.github.io/oxinfer/install.sh | sh
```

On Windows (PowerShell):

```powershell
irm https://karpeleslab.github.io/oxinfer/install.ps1 | iex
```

- **`install.sh`** puts `oxinfer` in `~/.local/bin`, or `$OXINFER_INSTALL_DIR`, and says how to add that folder to `PATH` when it isn't there. It takes options after `sh -s --`:
  - `--dir DIR`: install into DIR.
  - `--version vX.Y.Z` (or `$OXINFER_VERSION`): install that release instead of the latest.
  - `--static`: on x86_64 Linux, take the static build. This build is also chosen when the system has no glibc.
  - `--web`: start `oxinfer web` once installed.
- **`install.ps1`** puts `oxinfer.exe` in `%LOCALAPPDATA%\oxinfer\bin`, or `$env:OXINFER_INSTALL_DIR`, and adds that folder to the user's `PATH`. `$env:OXINFER_VERSION` picks a release instead of the latest.

Both scripts live in `web/public/` and are published with the browser demo on GitHub Pages. Later updates use `oxinfer --update` (below).

Each GitHub release carries the `oxinfer` binary (`.github/workflows/release-binaries.yml`,
dispatched by release-plz on the new tag):

| archive | built with | notes |
|---|---|---|
| `oxinfer-<tag>-linux-x86_64-static.tar.gz` | the [fullrust](https://github.com/KarpelesLab/fullrust) toolchain | a static executable with no libc and no interpreter: it needs only a Linux kernel. It still uses the GPU where a driver is installed: dlsym boots the system's own `ld.so` (dldota) to load libcuda / libvulkan |
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

`RUNS ON` (`"accelerator"` per variant in the JSON, with `"accel": {"prefill", "decode",
"decode_measured", "streamed"}` naming the unit of each stage: `cpu`, `metal`, `cuda`, `vulkan`
or `neural-engine`) says which accelerator the engine uses with the defaults: `GPU (Metal) prefill
+ decode` for the families with a whole-token decode graph (Llama 3 and Orpheus, Qwen3 / 3.5 / 3.6
/ 3.8 dense and MoE, Granite 4, Gemma 3 / 4; Qwen3.8-Flash-Next on Metal only), `GPU (CUDA)
prefill, CPU decode` where the GPU takes only the prompt's matmuls (every other family, Qwen2.5,
LFM3 and MiniMax included, and a graph model whose weights do not fit a discrete GPU), `CPU` where
no backend takes the weight format (GGUF and FP8 on CUDA and Vulkan, NVFP4 and MXFP8 everywhere,
the folded ternary GGUFs) or the pipeline is CPU-only (video, ZONOS, Fish, Audio8, Haru), `GPU
(Metal) encoder, CPU decode` for Whisper, plain `GPU (Metal)` for image and embedding pipelines,
`Neural Engine (Core ML)` for ANEMLL packages, and `, streamed from disk` when the weights are
over 60% of the RAM (on unified memory a streamed mixture of experts times GPU and CPU decoding
on its first tokens: `GPU or CPU decode (the faster)`). The header lists each accelerator of the
machine on its own line (also `"hardware": {"accelerators": [...]}` in the JSON): the CPU's SIMD
features and how much of the RAM models keep in memory, each GPU with what it runs and its
memory (a unified GPU shares the RAM), or `No GPU detected: models run on the CPU`, and on Apple
systems whether the Neural Engine is available.

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

The [web GUI](web.md) (`oxinfer web`, Models view) shows the same catalog, grouped by kind, with
the same verdicts, and sets a model up from it: size and disk check, download with progress, load.

## Subcommands

| Command | |
|---|---|
| `oxinfer serve [--host ADDR] [--port N] [--model ...] [--api-key KEY]` | the server: the [browser GUI](web.md) with this machine as its backend, the [OpenAI-compatible API](server.md) under `/v1` and Anthropic's `/v1/messages`, all on one port (default `http://127.0.0.1:8080/`, or the next free port) |
| `oxinfer web ...` | the same server, and it opens the GUI in the browser (`serve --open` does the same, `web --no-open` does not) |
| `oxinfer models list [--json]` | the folders of the model cache (`$OXINFER_CACHE`, default `~/.cache/oxinfer/hub`) with their size, when oxinfer last loaded them and the specs that load them (`org/name`, `org/name:QUANT` for each GGUF file), and the free space of its disk |
| `oxinfer models rm NAME... [--yes]` | deletes cache folders, named by folder (`org--name`) or repo id (`org/name`), after a confirmation (`--yes` to skip it; required when stdin is not a terminal); only folders directly inside the cache directory are removed, never a path outside it |

`--cache-dir DIR` works with both `models` commands. The same operations are
`oxinfer::model_cache::{list, remove, disk_space}` in Rust and `oxi_cache_list`,
`oxi_cache_remove` in C.
