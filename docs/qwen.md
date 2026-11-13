# Qwen3.5 / Qwen3.8

The Qwen3.5-family dense models are a different architecture (`qwen.rs`): 48 Gated DeltaNet layers
(a causal depthwise conv, L2-normalised keys/queries and a 128x128 delta-rule recurrent state per
value head) interleaved with 16 gated full-attention layers (24 query / 4 KV heads of 256, partial
RoPE on 64 dims, a sigmoid output gate), SiLU MLPs and zero-centred RMSNorm. The recurrent layers
keep a fixed-size state instead of a KV cache, so memory does not grow with context for 48 of the
64 layers. The vision tower takes images and videos (see
[Image, video and audio input](multimodal-input.md#image-video-and-audio-input)); the MTP head drives speculative
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
2.1-2.3x that with the matrix units (see [Performance](performance.md#performance)), and 65 tok/s of decode.
On randomly initialised weights at the real width (2048 hidden, 256 experts, 2
layers, the real vocabulary), oxinfer matches transformers in float32 to a max logit difference of
7e-5, with the same 20 greedy tokens. On Zawen-v0.1-35B-A3B's real weights, cut to its first 4
layers, the float32 match is 3e-5. The full 40 layers against transformers in bf16 give a mean
logit difference of 0.035, with the same top 10 in the same order (Q8: same top 10; Q4: same top
token). MLX-quantized Qwen experts are not read yet (Gemma 4's are).

Thinking is on by default in the Qwen template; `--think` enables it here (the trace is dimmed),
otherwise the generation prompt closes the think block.

Text, images, video and audio are supported (E2B/E4B have both towers).

## DSpark drafts for Qwen and Gemma 4

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
(`--mtp`, see [Speculative decoding](performance.md#speculative-decoding)). Its verification now runs as the
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
is faster (see [CUDA](platforms.md#cuda)); `--mtp` / `--draft` still use them.

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

## Fast verification

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
