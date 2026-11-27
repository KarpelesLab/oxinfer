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
fetches every part. When the repository has a vision projector at its top level
(`mmproj*.gguf`, as unsloth's and ggml-org's repositories of vision models do), `org/repo:QUANT`
fetches one beside the model too, with a note (BF16, else F16, else F32, else another; files in
subfolders such as `Old_Support/` are left alone), so that images work without naming it.
Nothing more is fetched when an `mmproj*.gguf` is beside the model already (whatever its name),
or when the `mmproj` option (`--mmproj`, engine / C / server `mmproj=`) is `none` or names a
file. Architectures: `qwen35`
(Qwen3.5 / 3.6 / 3.8), `qwen35moe`, `qwen3`, `llama` (Llama 3.x, and Orpheus TTS, see
[below](#llama-3-gguf)), `gemma4` (Gemma 4 text models, dense and 26B-A4B; see
[Gemma 4 26B-A4B](gemma-4-26b.md#gemma-4-26b-a4b-moe)), `mistral-large4` (see
[Mistral Large 4](mistral-large-4.md#mistral-large-4-1t-a52b), whose Pixtral `mmproj` file is read too), `deepseek4` and `deepseek41`
(see [DeepSeek V4](deepseek-v4.md#deepseek-v4-flash)), `glm5-next` and `glm-dsa` (see
[GLM-5.3](glm.md#glm-53-and-glm-53-flash); text only), `kimi-k3` and `kimi-k31` (see
[Kimi K3.1](kimi.md)), and `nemotron_h` and `nemotron_h_moe` (see
[Nemotron 3.5](nemotron.md#the-language-model-nemotron-h)). The tokenizer and chat template come from the
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
kernels for every type but Q1_0, TQ1_0 and TQ2_0 (which take the portable decoder on ARM), Metal
prefill kernels (and Metal decode kernels for every type but those three), and CUDA decode and
prefill kernels for every type but the six 1-bit / ternary ones
([CUDA](platforms.md#cuda)). On x86-64 with AVX2 every type's int8 dot
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
| `prism-ml/Ternary-Bonsai-2.5-27B-GGUF` `Q2_0` / `PQ2_0` / `PTQ1_0`, each plain and `-mtp` (5.6-7.9 GB), `mmproj-Q8_0` (0.63 GB) | Bonsai 2.5: the same format and model shape as the 27B below (version 1 folding, block 1024), retrained for agents; the `-mtp` files add the Q4_0 MTP layer (`blk.64`, not folded) | `Q2_0` on mainline llama.cpp (`de30697`, Metal and CPU) and oxinfer (Metal and CPU): 49/49 greedy steps identical, top-10 log-probabilities within 0.031 (Metal against Metal) / 0.103 (CPU against CPU); `PQ2_0` and `PTQ1_0` (which mainline cannot read) give the same 49 tokens, within 0.031 of llama.cpp's `Q2_0`; images through the `mmproj` file (Metal and CPU) |
| `prism-ml/Ternary-Bonsai-2-27B-gguf` `Q2_0` (7.6 GB) / `PTQ1_0` (5.9 GB) | the same for Qwen3.8-27B, with its own output head and the Gated DeltaNet permutation | `Q2_0`: 20 tokens as llama.cpp (within 0.07), then a near tie (0.328 against 0.323); `PTQ1_0`: the 32 tokens of `Q2_0`, within 0.06 |

`maritaca-ai/S5-nano` (`qwen3_5` safetensors of Qwen3.5-4B's shape, 9.3 GB) ran as stored and
with `--quant q4`; `mradermacher/gemma-4-31B-it-ARA-heretic-GGUF:Q2_K` was only loaded from its
header and tensor table (a file of the real size with zero weights: the 60 layers load and a
forward pass runs). `darrellbest/Qwen3.8-27B-Heretic-GGUF:Q4_K_M` (16.8 GB, with its embedded
MTP layer and the repository's `mmproj` file) ran on text and an image, against oxinfer's bf16
run of the safetensors release and not against llama.cpp; the release's compressed-tensors NVFP4
and FP8 repositories load from their safetensors (see
[Qwen3.8-27B-Heretic](qwen.md#qwen38-27b-heretic-and-its-packagings)).

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
that metadata is an error rather than a guess. On Metal the whole-token decode plans and the
prefill plans take the folded matmuls: a graph op (`Op::Fold`, kernel `dec_fold`, one
threadgroup per rotation block and token) applies `H (D P x)` to the input with the CPU's
butterflies in the CPU's order (the same values), once per input shared by the matmuls that read
it, and `Q2_0` / `PQ2_0` / `PTQ1_0` have matvec dots of their own (`MQ2_0`, `MPQ2_0`,
`MPTQ1_0`; the trits of a `PTQ1_0` chunk picked by selects, not branches, so the lanes of a
SIMD-group, which read different chunks of a block, stay on the same instructions: 9.5 -> 16.6
tok/s) and GEMM loaders (`load` / `loadh` in `GGUF_SHADER_SRC`, the matrix-unit GEMMs too;
`PTQ1_0`'s reads a 32-value quarter of a block as eight runs of one base-3 digit of the four
bytes of a word, two bytes at a time in the 16-bit halves of 32-bit words, `q 3^n mod 256` then
`3 x` giving the digit in its high byte and the next digit's `x` in its low one, the digits made
halves from their bits (`1024 + t`, minus 1025), and the matrix-unit tiles read a block's 7 words
once for its two K steps; the same values as before: pp512 ~270 -> 480-640 -> 740-826 tok/s,
as fast as `PQ2_0`, see below). CUDA keeps
these models on the CPU (`Op::Fold` is declined). The folded token embedding is still read on
the host (the greedy chain is off for these models). On ARM `Q2_0`, `PQ2_0` and `PTQ1_0` have
NEON dots (`q2_0`, `pq2_0`, `ptq1_0` in `kernels_ggml.rs`, the trits unpacked with byte
multiplies, `TRITS`' formula) and int8 GEMM rows: Bonsai 2.5 `PTQ1_0` on the CPU 1.2 -> 11.0
tok/s decoding, 12.6 -> 24.2 prefilling a 23-token prompt.

Ternary-Bonsai 2.5 27B on the M5 Max (`with-lock`, two alternated rounds while other agents'
jobs loaded the machine, so ranges; tok/s; the prompts as `tools/bench`'s: pp512 / pp2048 of
the corpus raw, 128 tokens of the prose and code prompts; llama.cpp mainline `de30697`
`llama-bench -fa 1`, `Q2_0` only, as it cannot read the other two):

| file | Metal pp512 | pp2048 | decode | CPU pp512 | decode |
|---|---|---|---|---|---|
| `PQ2_0` / `PQ2_0-mtp` | 508-728 | 442-637 | 29-37 | 51-61 | 9-12 |
| `Q2_0` / `Q2_0-mtp` | 616-710 | 544-605 | 32-35 | 49-62 | 11 (4 in one round under load) |
| `PTQ1_0` / `PTQ1_0-mtp` | 736-826 (544 in one round under load; was 480-642, before that 245-334) | 631-688 (was 487-615, 255-304) | 22-26 | 55-61 | 9-11 |
| llama.cpp `Q2_0` | 417 | 523 | 36.8 | 23 (12 threads) | 8.4 |

The `PTQ1_0` decode is ALU-bound (five trits a byte unpacked with multiplies) where the 2-bit
files are bandwidth-bound. Its prompts ran at half the 2-bit files' speed while the GEMM loader
decoded trit by trit (a division and a branch per value), then at ~0.9x with the runs of four
bytes digit by digit; the prompt GEMMs are bound by the dequantization's instructions as much as
by the matrix units (on random weights, 17408 x 5120 x 2048 tokens: no dequantization at all 6.45
ms, `PQ2_0` 6.79, `PTQ1_0` 7.32 with the run loader, 7.00 with the digits in f32 by
`floor(3 fract(q 3^n / 256))`, ~7.0 with every per-byte variant tried, 6.78 with the digits of
two bytes per 32-bit operation). `PTQ1_0` now prefills as fast as `PQ2_0` (alternated rounds,
best per round, the run loader / this one / `PQ2_0`: 2,047 tokens 615-629 / 666-682 / 661-671
tok/s and 512 tokens 740-776 / 823-826 / 786-799 on a quiet GPU; under other jobs' load 2,047
tokens 577-657 / 631-688 / 676-704; GPU time per 2,047-token pass, best of three: GEMMs 2,251 ms
against `PQ2_0`'s 2,219, the rest the same), logits identical to the old loader's (2,590-token
prompt: all 248,320 logits to the bit; against the CPU KL 2.0e-4, the same top 10, as before). What did not pay:
the next block's words loaded ahead (+1%: registers), two weight buffers with one barrier a K
step (+2%), the trits through a threadgroup-memory table of digit pairs (no faster), 256-token
tiles of 32 outputs or of 8 SIMD-groups (the matrix units run those 1-9% slower), the gate / up
pair GEMM for the ternary types (within noise).

The 2-bit files' GEMM loaders then got the same treatment: `Q2_0` / `PQ2_0` codes are made halves
two bytes at a time in the 16-bit halves of a 32-bit word (`(w >> 2k) & 0x00030003` ORed into
`1024 + c`, minus 1025, times the scale in half: the f32 path's `(c - 1) d`, exact in half),
where the generic path shifted, converted and scaled each value in f32 and rounded it to half,
and the matrix-unit tile of `PQ2_0` reads a block's scale once and its row's two quarters for its
two K steps together. On random weights (17408 x 5120 x 2048 tokens, a Swift harness on the
library's source): `PQ2_0` 6.81 -> 6.61 ms against 6.45 with no dequantization at all (the
loader alone 6.69; bytes as `uchar4` lanes converted to half 6.78). In the 27B's 2,047-token
pass (`OXINFER_PF_PROFILE`, two runs each, the same in both): `PQ2_0` GEMMs 1,359 -> 1,317 ms
(-3.1%), `Q2_0` (whose 64-value blocks give a K step one block, so only the loader) 1,417 / 1,469 ->
1,414 / 1,448 ms (-0.2 / -1.4%); whole passes under other jobs' load +2-5% at 511 tokens (`PQ2_0` 660-820 ->
696-839 tok/s, alternated rounds) and within the load's noise at 2,047. Logits on the
2,590-token prompt identical to the old loaders' for both files (all 248,320, matrix units and
SIMD-group path). With
`--mtp` (the files' Q4_0 MTP layer; PrismML's Metal preset drafts one token at p >= 0.6) the
drafts were accepted 42-57% of the time on Metal, and `--mtp-cost` then drafts in few rounds: 30-33 tok/s,
no faster than plain decoding (a verification pass costs about a plain step plus the drafts'
share here), so MTP stays opt-in for these files. Images: the `mmproj-Q8_0` tower (Q8_0
weights of the padded MLP layers read through a bf16 copy) described a 512x384 photo (mountains
at sunset above an inset of a Japanese room) correctly on Metal and the CPU, 192 image tokens,
prefill of the 213-token prompt 404 tok/s on Metal. Before this, the repository's
`Ternary-Bonsai-2.5-27B-mmproj-Q8_0.gguf` was not found beside the model (only `mmproj*` names
were: `gguf::is_mmproj_name` also takes a name with an `-mmproj` part) and its Q8_0 weights were
refused by the bf16 loaders of the tower. The six 1-bit / ternary types are unpacked in registers on x86-64 (AVX2, rows
bit-identical to the portable decoder's: 19-42 ns per 256 weights and token on one thread, as
Q2_K's 21 and Q4_K's 32, against 140-185 through the portable decoder). On the test machine, shared and under load, so indicative only: the 2B on 8
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

### GGUF LoRA adapters

llama.cpp's adapter files (`general.type = adapter`, `adapter.type = lora`, from
`convert_lora_to_gguf.py`) load with `--lora FILE[:SCALE]` on a GGUF base of the same
architecture, as in `llama-cli --lora` / `--lora-scaled`. They are applied at run time, unmerged,
or merged with `--lora-merge`:

```sh
./target/release/oxinfer --model AtomicChat/Inkling-2-Mini-GGUF:AD-IQ2_M \
  --lora AtomicChat/Inkling-2-Mini-Uncensored-GGUF/Inkling-2-Mini-Uncensored-LoRA.gguf -p "Hello"
```

Their tensors (`blk.N.attn_output.weight.lora_a` / `.lora_b`, expert stacks `ffn_*_exps` with one
pair per expert, `token_embd`) are named and laid out as the base file's, the Llama query / key
permutation and Qwen3.5's tiled value heads included. So they go on a GGUF base as they are, and
a safetensors base takes the PEFT form of the adapter instead. The layouts, the scale
(`adapter.lora.alpha / r * SCALE`), the families, merging and the checks against PEFT, llama.cpp
and AtomicChat's baked files are in [LoRA adapters](lora.md).

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
