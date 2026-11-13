# Getting weights

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

## The ATOnline mirror

Repo ids are looked up on the [ATOnline](https://www.atonline.com/) mirror first: its
`AI/Model:get` method (`https://ws.atonline.com/_special/rest/AI/Model:get?key=org/name`) lists a
repo's files with download URLs, small ones inlined. Repos it does not carry come from
huggingface.co. `OXINFER_HUB=mirror` or `OXINFER_HUB=hf` (`oxinfer::hf::set_hub` in the library) pins one source; the HuggingFace token is
only ever sent to huggingface.co. The browser build uses the same lookup (the mirror allows
cross-origin requests).

## GGUF files

`--model` also takes llama.cpp GGUF files, as a path or as `org/repo:QUANT` (the one file of a
GGUF repository whose name ends in `QUANT.gguf`, e.g.
`HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF:Q4_K_P`). Models split into parts
(`name-00001-of-00009.gguf`, ...) load from any part or from their folder, and `org/repo:QUANT`
fetches every part. Architectures: `qwen35`
(Qwen3.5 / 3.6 / 3.8), `qwen35moe`, `qwen3`, `llama` (Llama 3.x, and Orpheus TTS, see
[below](#llama-3-gguf)), `gemma4` (Gemma 4 text models, dense and 26B-A4B; see
[Gemma 4 26B-A4B](gemma-4-26b.md#gemma-4-26b-a4b-moe)), `mistral-large4` (see
[Mistral Large 4](mistral-large-4.md#mistral-large-4-1t-a52b), whose Pixtral `mmproj` file is read too), `deepseek4` and `deepseek41`
(see [DeepSeek V4](deepseek-v4.md#deepseek-v4-flash)), `glm5-next` and `glm-dsa` (see
[GLM-5.3](glm.md#glm-53-and-glm-53-flash); text only), and `kimi-k3` and `kimi-k31` (see
[Kimi K3.1](kimi.md)). The tokenizer and chat template come from the
GGUF metadata (the rebuilt tokenizer matches Qwen3.8's `tokenizer.json` on 305 test texts, and
Gemma 4's, a SentencePiece-style BPE with byte fallback, on this README and multilingual text), and
llama.cpp's conversions are undone at load (`gguf.rs`): norms with the 1 folded in, `ssm_a`, the
tiled Gated DeltaNet value-head order, separate 3D expert tensors, Llama's permuted query / key
rows, Gemma 4's proportional RoPE stored as per-frequency divisors (`rope_freqs`). The `mmproj`
vision files are read for Mistral Large 4 (`pixtral`) and Qwen3.5-family towers
(`qwen3vl_merger`, see [Qwen3.8-Flash-Next](qwen-flash-next.md#qwen38-flash-next)); other models' image input needs
the safetensors or MLX checkpoint. Weights stay in their GGUF
format: F32, F16, BF16, Q4_0/Q4_1/Q5_0/Q5_1/Q8_0, Q2_K-Q6_K, IQ1_S/IQ1_M, IQ2_XXS/XS/S,
IQ3_XXS/S, IQ4_NL/XS, MXFP4 and the 1-bit and ternary types Q1_0, Q2_0, TQ1_0, TQ2_0 and
PrismML's PQ2_0 / PTQ1_0 (`ggml.rs`, dequantization bit-exact with gguf-py or, for the 1-bit and
ternary types, ggml's C code; see [below](#mixed-low-bit-and-ternary-ggufs)), with NEON decode
kernels for every type but those six (which take the portable decoder on ARM) and Metal prefill
kernels. On x86-64 with AVX2 every type's int8 dot
products run on `maddubs` / `madd`, and the block-of-32 formats, the K-quants and IQ4_XS are
unpacked in registers (`kernels_ggml_x86.rs`; the grid-table i-quants IQ1-IQ3 still unpack through
the portable decoder); every row is bit-identical to the portable kernel's. The i-quants (IQ1-IQ3) decode several times
slower than Q4_K on the CPU (table lookups); Q4_K, Q5_K, Q6_K, IQ4_XS and Q8_0 are close to
oxinfer's own Q4/Q8. Embedded MTP layers (`blk.N.nextn.*`) work with `--mtp`.

On HauhauCS's Qwen3.8-27B Q4_K_P the next-token distribution matches llama.cpp on the same file
(same top 10, log-probabilities within 0.03, mean 0.013). F32/BF16 GGUFs from llama.cpp's
converter match transformers to 5e-5. `--save-quantized` does not take GGUF input.

### Mixed, low-bit and ternary GGUFs

Files checked in November 2026 (raw prompt of 21 tokens, greedy, CPU; "llama.cpp" is mainline
at 9daa0aa, 13 November 2026, built for the CPU, run on the same file through
`tools/llama_logits_dump.cpp`, oxinfer through `examples/qwen_greedy_dump.rs`, whose `--compare`
gives the numbers below: greedy tokens identical until the first difference, and the largest
log-probability difference over the top 10 of those steps):

| File | What it is | Result |
|---|---|---|
| `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF:IQ2_XS-mtp` (8.8 GB) | GSQ-RCO: every tensor quantized at each standard ggml type with GSQ (grid assignments and scales learned per tensor), then one type chosen per tensor under a size budget (RCO). The file is ordinary GGUF with thirteen types side by side (F32, BF16, Q2_K, Q4_K, Q6_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS); nothing new to decode | 32/32 tokens as llama.cpp, within 0.31; `--mtp` from the embedded layer: 3.58 tokens per forward, 88% of drafts accepted |
| `DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF:MTP-IQ3_M` (5.7 GB) | Qwen3.5-9B fine-tune with the MTP layer (`blk.32.nextn.*`), IQ3_S / Q4_K | 26 tokens as llama.cpp (within 0.65), then a near tie (0.334 against 0.316); `--mtp`: 2.33 tokens per forward. The repository's `MTP-IQ2_M` file is damaged on the Hub (its first 10.97 MB, the whole header, are zeros) |
| `HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive` `Q4_K_M`, `ccore/Qwen3.5-4B-3D-Coder` `Q4_K_M` (2.7 GB each) | Qwen3.5-4B fine-tunes in the usual layout (the second with the newer `nextn_predict_layers = 0` and `attention.recurrent_layers` keys) | 16 tokens as llama.cpp (within 0.28) then a near tie (0.202 against 0.199); 32/32 (within 0.25) |
| `prism-ml/Ternary-Bonsai-2-2B-gguf` `Q2_0` / `PQ2_0` / `PTQ1_0` (0.4-0.5 GB) | ternary weights, Hadamard-folded, tied output head (below) | `Q2_0`: 48/48 tokens as llama.cpp, within 0.22; the other two packings give the same 48 tokens |
| `prism-ml/Ternary-Bonsai-2-27B-gguf` `Q2_0` (7.6 GB) / `PTQ1_0` (5.9 GB) | the same for Qwen3.8-27B, with its own output head and the Gated DeltaNet permutation | `Q2_0`: 20 tokens as llama.cpp (within 0.07), then a near tie (0.328 against 0.323); `PTQ1_0`: the 32 tokens of `Q2_0`, within 0.06 |

`maritaca-ai/S5-nano` (`qwen3_5` safetensors of Qwen3.5-4B's shape, 9.3 GB) ran as stored and
with `--quant q4`; `mradermacher/gemma-4-31B-it-ARA-heretic-GGUF:Q2_K` was only loaded from its
header and tensor table (a file of the real size with zero weights: the 60 layers load and a
forward pass runs).

**Ternary-Bonsai 2** (PrismML) stores every matrix of the language model as ternary values with
one f16 scale per group, in one of three packings: `Q2_0` (mainline llama.cpp's type 42: 2-bit
codes, a scale per 64 weights, 2.25 bits per weight), `PQ2_0` (type 142 of PrismML's llama.cpp
fork: the same codes, a scale per 128, 2.125 bits) and `PTQ1_0` (type 143: five trits per byte, a
scale per 128, 1.75 bits). Mainline llama.cpp reads only the first; oxinfer reads the three (and
mainline's Q1_0, TQ1_0 and TQ2_0, which share the decoder).

Those weights are also *Hadamard-folded* (`prism.hadamard.*` metadata): each folded matrix is
`W D H`, the original times a diagonal of signs and a block Walsh-Hadamard rotation, so the
runtime has to flip and rotate the input of every such matmul (`H (D x)`, `hadamard.rs`), read
the token embedding back through the inverse (`D (H z)`), and feed the Gated DeltaNet output
projection its value heads in the checkpoint's order (`gdn_v_grouped`). A runtime that ignores the
metadata loads the file and prints noise (llama.cpp before b11556 did; oxinfer before this
support refused the files for their tensor type). Version 1 (27B: a separate output head) and
version 2 (2B: the folded token embedding is also the output head) are read; anything else in
that metadata is an error rather than a guess. Folded matmuls run on the CPU kernels (the GPU
decode and prefill plans do not take them), so on a machine with a GPU these models decode at the
CPU's speed. The six 1-bit / ternary types are unpacked in registers on x86-64 (AVX2, rows
bit-identical to the portable decoder's: 19-42 ns per 256 weights and token on one thread, as
Q2_K's 21 and Q4_K's 32, against 140-185 through the portable decoder) and take the portable
decoder on ARM. On the test machine, shared and under load, so indicative only: the 2B on 8
threads at 41 tok/s (`Q2_0`), 33 (`PQ2_0`) and 23 (`PTQ1_0`) where llama.cpp's CPU build gave 46
on `Q2_0`; the 27B `PTQ1_0` at 6.9 tok/s on a chat prompt (2.1 before the AVX2 unpackers; a
GSQ-RCO IQ2_XS file of the same model, whose i-quants unpack through the portable decoder: 2.3).

### Llama 3 GGUF

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

## MLX checkpoints

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
