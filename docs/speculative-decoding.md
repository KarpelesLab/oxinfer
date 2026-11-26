# Speculative decoding and prompt lookup

## Speculative decoding

`--draft models/gemma-4-E4B-it-assistant` enables Google's multi-token-prediction drafter: a
4-layer, 256-wide decoder that cross-attends to the main model's shared KV caches and predicts
from the concatenation of the last token's embedding and the previous hidden state. Each round
drafts the number of tokens, 0 (a plain step) to 6, with the most expected tokens per second
(`spec_cost::ChainPolicy`: the measured cost of a draft step and of verifying each length, the
running acceptance of each draft position; `--spec-draft-fixed` always drafts `--draft-tokens`)
and verifies them in one multi-token pass of the GPU decode plan (each token's arithmetic the
one-token step's); greedy verification makes the output identical to plain greedy decoding. Acceptance is about 40% on prose
and 55–65% on code, giving 2–4 tokens per forward. The drafter's own drafts were checked against
the transformers implementation token for token. It pays off most where the verification pass
is bandwidth-bound (bf16, Q8); at Q4 the multi-token int8 kernels are instruction-bound, so it
only helps on predictable text.

`--mtp` does the same for Qwen3.5 / Qwen3.8 with the checkpoint's own multi-token-prediction
layer (`mtp.*`, `qwen_mtp.rs`): one gated-attention decoder layer that takes the main model's
final-normed hidden state at position i and the embedding of token i+1 (each RMS-normed,
concatenated embedding first, projected by `mtp.fc`), runs at position i with its own KV cache,
and predicts token i+2 through the shared LM head; further drafts feed it its own output (the
vLLM `qwen3_5_mtp` scheme). The default is up to 3 drafts per round, each round the length (0 to
3) with the most expected tokens per second from the measured round times and the acceptance by
draft position (`--no-mtp-cost` / `mtp_cost=0`: grow after a fully accepted round, shrink after a
fully rejected one, between 1 and 3). With GPU
decoding the verification is one GPU pass (the same multi-token kernels as DSpark's, so greedy output
is plain GPU decoding's) and the draft head runs on the GPU; MTP is on by default for Qwen3.5-4B and
Qwen3.5-9B, where it measured faster on every kind of prompt (numbers and the comparison with DSpark
in [DSpark drafts for Qwen and Gemma 4](qwen.md#dspark-drafts-for-qwen-and-gemma-4); `--no-draft` turns it
off). Both
greedy and sampled decoding are supported: greedy output is identical to plain greedy (the
decode kernels give bit-identical rows for up to 7 tokens per pass, so the verification logits
equal one-token logits; checked on long outputs with and without `--think`, bf16 / Q8 / Q4,
dense and MoE), and sampling uses the standard speculative-sampling acceptance rule, so every
token is drawn from exactly the distribution `--temp` / `--top-k` / `--top-p` define (the
random stream differs from plain sampling, so the same seed gives a different text). Rejected
drafts are undone exactly: the verification forward keeps the Gated DeltaNet states from
before the pass plus each token's delta-rule inputs, and a rewind replays the rule up to the
last accepted token (the same arithmetic, so bit-identical; one state copy instead of one per
token, which is 151 MB per drafted token on Qwen3.8-27B); KV caches are truncated. On Qwen3.8-27B (M5 Max, under load from other
jobs):

| Precision | Plain | `--mtp` (code) | `--mtp` (prose) | Acceptance (code / prose) |
|---|---|---|---|---|
| `--quant q8` | 7.5 tok/s | 15.2 tok/s (3.75 tok/forward) | 10.5 tok/s (2.55 tok/forward) | 94% / 59% |
| `--quant q4` | 14.6 tok/s | 24.0 tok/s | 14.7-16.1 tok/s | 94% / 57% |

Q4 gains less because a 4-token Q4 pass costs about twice a one-token pass (SDOT-bound). (The
multi-token CPU passes have since become cheaper: see
[Multi-token passes and MoE experts on the CPU](#multi-token-passes-and-moe-experts-on-the-cpu).) On
Qwen3.5-0.8B every draft step reads the 248k-row LM head (a third of the model), so `--mtp`
gains ~20-25% on code and loses a little on prose; a trimmed draft head (`DraftHead`: a
`[V_draft, H]` matrix plus a draft-to-target id map, FastMTP style) can replace it through the
library. The MoE models (Qwen3.6-35B-A3B) work the same way, with an MoE MTP layer. A converted
checkpoint (`--save-quantized`) keeps the MTP weights; older converted copies can take them from
the original checkpoint with `--mtp-from DIR`.

From the library: `QwenModel::load_mtp(&files)` (or `QwenMtp::load(cfg, sidecar_files, quant)`),
then `MtpDecoder::new(&model, &mtp, MtpOptions { .. })`: `prefill` the prompt, `push` the first
token, call `step` until a stop token (`examples/qwen_mtp.rs`).

FastMTP sidecars (a GGUF holding only an MTP layer, a trimmed output head and a `d2t` map, such as
HauhauCS's `...-FastMTP-32K.gguf`) load with `--mtp-from FILE` (or `org/repo:FastMTP-32K`): the
sidecar's layer, token embedding and 32k-row head replace the target's, as in HauhauCS's
llama.cpp patch (`QwenMtp::attach_sidecar` in the library). With HauhauCS's Qwen3.8-27B Q4_K_P
GGUF, greedy output with the embedded MTP layer or the FastMTP sidecar is token-for-token the
plain greedy output.

### Prompts with a draft

A DSpark draft reads the target's residual stream at a few layers for every position, the prompt's
included, and an MTP head runs its layer over the prompt's hidden states. On Metal the prompt still
runs on the whole-pass prefill plan: the plan copies the residual after the draft's layers into a
capture buffer on the device, read back once per 2048-token chunk (`gpu_prefill` with `cap` in
`qwen.rs` / `model_gpu.rs`); the draft's context pass over the prompt (feature projection, keys and
values of every draft layer) runs as a prefill plan of its own instead of 16 positions at a time or
on the CPU; and the MTP layer's attention over a prompt runs on the GPU (`attn_forward_draft`).
Until November 2026 a draft sent the prompt through the op-by-op path (GPU GEMMs, attention on the
CPU: 8.7 ms per token for Qwen3-8B at 4.3k tokens, most of the pass) and the draft's context pass
ran on the CPU, so default users prefilled 5-20x slower than with `--no-draft`. Prefill tok/s on a
4.3k-4.8k-token prompt (default drafts and verification; M5 Max shared with other jobs, load
averages 4-27, so +-15%):

| Model | Draft | Before | Now | `--no-draft` |
|---|---|---|---|---|
| Qwen3-4B bf16 | DSpark | 124 | 2109 | 2356 |
| Qwen3-8B Q8_0 | DSpark | 87-128 | 1915-1981 | 1980-2097 |
| Qwen3-14B Q8_0 | DSpark | 105 | 957 | 928 |
| Gemma 4 12B Q8_0 | DSpark | 327 | 1122 | 1246 |
| Qwen3.5-4B bf16 | MTP | 2141 | 3195 | 3986 |
| Qwen3.5-9B bf16 | MTP | 1203 | 1392 | 1693 |
| Qwen3.8-27B q4 | RadixArk DSpark | 143 | 690 | 617 |
| Gemma 4 31B q4 | RedHatAI DSpark | 98 | 274-416 | 366 |

The captured rows are the plan's residuals, so they differ from the op-by-op path's in rounding
only: with fixed 7-token drafts and exact verification, the drafts accepted were the same on every
pair (e.g. Qwen3-8B 41 of 146, Gemma 4 12B 45 of 120, the 27B 35 of 78) but Gemma 4 31B (31 of 116
before, 30 of 124 now). Greedy output is unchanged (identical to master's and, with exact
verification, to `--no-draft`'s). Prompt lookup needs no captures and always prefilled on the
plan.

### Inkling-2-Mini's drafter

Inkling-2-Mini ships its own 3-layer block drafter (`mtp.safetensors`, `--mtp`; unsloth's GGUF
conversion `dspark-Inkling-2-Mini-Q8_0.gguf`, `--draft`), which runs as a DSpark draft
(`src/inkling_draft.rs`, after llama.cpp's `graph_inkling2`): per-layer softmax mixes of the
target's streams at 4 depths as its context, a block of 8, and a recurrent head whose correction
goes through the target's LM head. Sessions go back up to 64 positions (the short convolutions
keep their inputs in rings by position), rounds verify at most 6 drafts (7-token passes stay on
the bit-exact decode kernels), and greedy output is plain greedy output. Its acceptance matches
llama.cpp's own implementation on the same prompts (77-78% of 3 drafts on code, 47-48% on prose,
35-38% on chat); on the CPU a verification pass costs about its token count in steps (every
token reads its own 8 experts), so the cost-aware rounds fall back to plain steps there. On the
GPU the draft pass is one Metal submission (3-4 ms for 3 slots) and a 4-token verification costs
1.6-2 steps: drafts gain 45-65% on code (Q8_0, 58 -> 80-95 tok/s), are about even on prose and
chat, and the cost-aware rounds (which also choose how many to draft) stay near plain speed where
drafting does not pay. On Metal the drafter found beside a checkpoint is used by default: its
decoder also takes prompt lookup's drafts in the rounds it does not draft, and probes with short
blocks, so it stays within a few percent of prompt lookup alone where drafting does not pay and
gains on code and code edits. Numbers: [Inkling-2 Mini](inkling.md#speculative-decoding).

## Multi-token passes and MoE experts on the CPU

Speculative verification (and prompts shorter than 8 tokens) runs on the CPU's decode kernels,
whose rows are bit-identical whatever the token count, so verification logits equal one-token
logits. Two lossless changes (after the analysis of llama-modchip in
[docs/llama-modchip](llama-modchip/README.md)) made those passes cheaper:

- **Column groups for the blocks-of-32 formats.** GGUF Q8_0, Q4_0, IQ4_NL, MXFP4, Q4_1, Q5_0,
  Q5_1 and oxinfer's Q4 used to take tokens in pairs (each block unpacked once per pair, plus a
  one-token pass for an odd count). They now run the one-token kernel instantiated for up to 8
  tokens on column groups (9 = 5 + 4, 16 = 8 + 8; narrower when the row is long enough that a
  group's int8 activations would not stay in L1), with rows in 64 KB tiles. Each token's
  arithmetic is the one-token kernel's, so rows stay bit-identical. The K-quants and i-quants
  already had such kernels; oxinfer's int8 format gained nothing (its blocks need no unpacking)
  and keeps the pairs. Single thread, 4096 x 4096, ns per 256 weights per token, token pairs ->
  column groups (two interleaved runs, a loaded machine):

  | Format | 3 tokens | 4 | 5 | 8 | 15 |
  |---|---|---|---|---|---|
  | Q8_0 | 8.6 -> 6.6 | 7.2 -> 5.6 | 8.0 -> 5.0 | 7.7 -> 5.6 | 8.2 -> 6.0 |
  | Q4_0 | 7.6 -> 6.1 | 6.4 -> 5.5 | 7.1 -> 6.3 | 6.4 -> 6.0 | 7.2 -> 7.0 |
  | MXFP4 | 9.1 -> 6.8 | 7.4 -> 5.9 | 9.1 -> 5.5 | 7.8 -> 6.0 | 8.6 -> 6.0 |
  | Q5_0 | 13.7 -> 9.1 | 11.5 -> 8.1 | 12.0 -> 7.5 | 12.3 -> 6.7 | 12.8 -> 7.3 |
  | Q4 (`--quant q4`) | 7.1 -> 6.1 | 6.4 -> 5.8 | 6.3 -> 5.7 | 6.6 -> 6.1 | 6.4 -> 6.7 |

  A whole verification pass of Qwen3-8B Q8_0 (18 threads, recording forward plus LM head,
  `examples/pass_bench.rs`, best of 7 interleaved): 3 tokens 49 -> 41 ms, 4 tokens 49 -> 41 ms,
  8 tokens 83 -> 65 ms (one token: 34 ms). With its DSpark draft on the CPU the decoding gained
  12-23% on prose (40 -> 44-49 tok/s) and was within noise on code, where the cost-aware drafting
  now drafts longer blocks; greedy output identical. On other CPUs the portable GGUF kernel takes
  the same column groups (x86-64 under Rosetta, which runs the portable path: Q8_0 / Q4_K at
  8 tokens 2.2-2.3x faster per token).
- **MoE experts in one batch per stage.** Decode and verification passes of the MoE models
  (Qwen3.5 / 3.6 MoE, Gemma 4 26B-A4B, Mistral Large 4, DeepSeek V4) quantize the input once,
  build the expert -> tokens grouping once for gate, up and down, and run the gate/up matmuls of
  every chosen expert, and of the shared expert, as one parallel region, then all down matmuls as
  another (they ran one region per matmul: Qwen3.5-35B-A3B decode went from 521 to 161 regions
  per token). This applies only when every matmul would run on the CPU decode kernels anyway (no
  accelerator, LoRA or float weights), so no matmul run elsewhere ever reads the shared quantized
  input. Qwen3.5-35B-A3B Q4, passes of 1 / 2 / 4 / 7 tokens, per-expert regions -> batched (12
  threads, best of 9 interleaved): 35 -> 29, 38 -> 32, 52 -> 45, 81 -> 66 ms. End to end (CPU
  decode, 128 tokens, alternating binaries on a loaded machine) Gemma 4 26B-A4B Q4_0 went from
  55.3-55.8 to 56.2-59.8 tok/s; Qwen3.5-35B-A3B's runs (60-69 tok/s) were within the noise.

`examples/kernel_bench.rs` (formats, threads, shape, token counts) and `examples/pass_bench.rs`
(`pairs` / `auto` / `seq` / `old` modes; `PASS_COMPARE=1` asserts that all modes give bit-identical
logits) measure these. Prefill (8 tokens and more) keeps its f32 GEMM: switching it to the int8
kernels would change its results.

## Prompt lookup

`lookup.rs` drafts without a draft model: when the last tokens of the conversation (prompt and
reply so far) occurred before, the tokens that followed that occurrence are the draft, verified
like any other in one multi-token pass. Replies that quote, edit or summarize their prompt
(code edits, refactors, extraction, rewrites, repeated structure) accept long runs of such drafts;
on free prose there is rarely a match and the round is a plain one-token step.

- Matching: an index from every bigram of the context to its positions; the occurrences of the
  last two tokens (the 48 most recent) are extended backwards to their full match length, the
  longest match wins (the most recent among equals), and the draft is what followed it (up to
  `--lookup-tokens`, default 10; when the verification runs on the CPU, at most 6 unless
  `--exact-verify` raises the process-wide limit of the batch-invariant decode kernels, as DSpark
  and MTP do). Other occurrences that agree on the next token count as consensus.
- Verification-outcome memory (the one idea taken from llama-modchip's drafter, see
  `docs/llama-modchip`): a 4096-entry direct-mapped table keyed by a hash of the four tokens before
  a position remembers the draft token the model last rejected there and the token it chose
  instead; a draft token refuted in the same context, and not accepted there since, ends the draft
  (`--no-lookup-memory` turns it off).
- Cost-aware length: each draft token's chance of being accepted (when those before it were) is
  estimated per match class (match length, consensus, depth), calibrated online from the
  verification outcomes; with the measured time of verifying each number of drafts (a plain step
  for none), the round verifies the length with the most expected tokens per second, which is 0
  (a plain step) when nothing is worth it. A length not yet measured is estimated from the
  measured ones through the backend's table of verification costs by token count
  (`spec_cost::CostPrior`); a match that is not worth it is still tried now and then (as rarely
  as keeps the expected loss of such probes within 2% of the plain steps' time) so that a poor
  early estimate does not stick. In DSpark and MTP decoding every round's verification time
  feeds the same measurements.
- Exact like the other drafts: greedy output is the plain greedy output (bit for bit with
  `--exact-verify`; with fast verification the logits may differ in rounding, as for DSpark and
  MTP), and sampling uses speculative sampling with a one-token draft distribution (accept with
  the model's probability of the draft, else draw from the rest), which keeps the model's
  distribution; a repetition penalty sees the accepted tokens as one-token decoding would.
- With DSpark or MTP, a long lookup match (8 tokens or more: on shorter ones the draft models
  did better) replaces the draft model's round when its expected tokens per second (calibrated
  chances over the measured verification time) beat the recent rate of that decoder's rounds (or
  of a plain step when its policy chose one).

On by default (`--no-lookup`, engine / C API / server key `lookup=0`), for every family the
engine decodes plainly (Gemma 4 / 3, Qwen, Llama 3, Mistral Large 4, DeepSeek V4) and inside
DSpark and MTP decoding; not with the Gemma 4 assistant drafter or Core ML models. From the
library: `lookup::Lookup` (the drafter) and `lookup::LookupRounds` (a `RoundDecoder` over any
`dspark::Target`), or the `lookup` field of `DsparkDecoder` / `MtpDecoder`.

Measured on the M5 Max (Metal, GPU decoding; 300 tokens, greedy, `--no-system`, default fast
verification; tok/s means of 2 interleaved runs per prompt; the machine was shared with other
jobs, load 8-27, so single cells are noisy). Prompts: `code` (rename a function in a 60-line
Python module and print it back), `summary` (quote the sentences of a text that mention a date),
`json` (25 generated records), `prose` (a short story), `chat` (three tips). Tokens per forward
and drafts accepted in parentheses:

| model | plain | prompt lookup | (no memory) | code | summary | json | prose | chat |
|---|---|---|---|---|---|---|---|---|
| Qwen3-8B (Q8_0 GGUF) | 57.8 | **98.1** | 98.3 | 59 -> 138 (3.0, 74%) | 59 -> 160 (3.9, 74%) | 56 -> 78 (1.5, 38%) | 56.5 -> 57.0 | 58.9 -> 57.4 |
| gemma-4-12B-it (Q8_0 GGUF) | 32.8 | **58.7** | 58.5 | 32 -> 75 (3.1, 79%) | 31 -> 97 (4.2, 75%) | 33 -> 51 (1.9, 57%) | 33.4 -> 34.2 | 34.1 -> 36.5 |
| Qwen3.5-4B (bf16) | 57.3 | **108.9** | 109.9 | 57 -> 155 (3.4, 70%) | 59 -> 194 (4.5, 72%) | 57 -> 80 (1.6, 37%) | 56.6 -> 57.0 | 56.5 -> 59.1 |
| Qwen3.5-35B-A3B (q4, MoE) | 139.2 | **153.6** | 145.8 | 139 -> 169 (1.8, 92%) | 147 -> 183 (2.2, 92%) | 133 -> 137 | 134 -> 139 | 144 -> 141 |

Copying prompts gain 2-3x on the dense models; prose and chat stay at plain speed (a few drafts
per reply). On the MoE model a verified token costs its own routed experts (a 9-token pass is
4-6 plain steps), so the policy verifies only the likeliest few drafts (1.8-2.2 tokens per
forward at 92% acceptance) and the gain is 15-25% where the reply copies. The verification-outcome
memory vetoed nothing in any of these runs: with the most recent of the longest matches winning,
the target's own correction is already in the context the next time the same suffix comes up, so
a refuted continuation is rarely proposed again (llama-modchip's drafter also predicts from
n-gram statistics, which repeat). It costs a hash per draft token and stays on.

With the default draft models (a second run, load 20-27; DSpark for Qwen3-8B and Gemma 4 12B,
MTP for Qwen3.5-4B), lookup takes over only where a long match beats the draft model's recent
rounds:

| model | draft | draft + lookup (default) | code | summary | json | prose | chat |
|---|---|---|---|---|---|---|---|
| Qwen3-8B, DSpark | 116.9 | 115.8 | 129 -> 134 | 160 -> 149 | 127 -> 129 | 51.7 -> 51.6 | 117 -> 115 |
| gemma-4-12B-it, DSpark | 76.0 | **81.0** | 94 -> 99 | 106 -> 119 | 91 -> 94 | 37.9 -> 41.5 | 51 -> 53 |
| Qwen3.5-4B, MTP | 78.3 | **102.9** | 86 -> 138 | 102 -> 174 | 84.5 -> 85.8 | 49 -> 50 | 70 -> 67 |

(DSpark already accepts 80-97% of its 7-token blocks on the copying prompts, so lookup adds little
there; MTP drafts 3 tokens, and the lookup's longer runs lift it most. Without the long-match rule,
short prose matches displaced better DSpark rounds on Gemma 4 12B: prose 43.5 -> 35.7.)

Exactness: with `--exact-verify`, greedy output with lookup alone and with the default draft plus
lookup is the plain greedy output token for token on Qwen3-8B, gemma-4-12B-it, Qwen3.5-4B (Gated
DeltaNet rewinds) and Qwen3.5-35B-A3B (4 prompts x 300 tokens each); with the default fast
verification, all 130 outputs of these runs (lookup, without memory, DSpark, MTP, combined) are
the plain greedy outputs too.
