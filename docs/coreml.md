# Core ML / Apple Neural Engine

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

## Converting models

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
