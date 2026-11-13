# Mistral Large 4 (1T-A52B)

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
[Runtime interventions](interventions.md#runtime-interventions-neuron-edits-steering-abliteration-sae-features)
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
[Streaming weights from flash](performance.md#streaming-weights-from-flash-models-larger-than-ram)) with a page
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

## FP8 on the GPU

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

## DSpark speculative decoding

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
ones (see [DSpark drafts for Qwen and Gemma 4](qwen.md#dspark-drafts-for-qwen-and-gemma-4)).

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
