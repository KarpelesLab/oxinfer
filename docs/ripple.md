# Ripple-65B Preview

`ricursiveAI/Ripple-65B-Preview` (`ricursive`, `RicursiveForCausalLM`; 80 layers of 5120, 65B
parameters, dense, 130 GB of bf16 safetensors) and `ricursiveAI/Ripple-65B-Preview-NVFP4` (the
same model with its MLPs in NVFP4, 57 GB) run on `ripple.rs`. Text, the model's own reasoning
blocks, tool calls.

What was run: tiny random checkpoints against Ricursive's modeling code (every feature on), the
first 10 layers of the real NVFP4 weights against that code, and the whole NVFP4 model for text.
**The bf16 checkpoint was not run**: its 130 GB are more than this work could download (100 GB),
and there is no float reference for the whole NVFP4 model on this machine (it would need the
bf16 weights in memory).

## The architecture

The repository ships its modeling code (`modeling_ricursive.py`), and `ripple.rs` follows it: a
dense transformer in the shape of Gemma 3 ("thin and deep", in the model card's words) with a few
additions.

- The embeddings are multiplied by `embed_scale` (71.55, the square root of the width).
- Blocks with four norms: `x += norm(attn(norm(x)))`, then `x += norm(mlp(norm(x)))`.
- More heads than the width holds: 80 query heads of 128 (a 10240-wide attention on a 5120-wide
  stream), 8 key / value heads. Queries and keys are RMS-normed per head; scores are scaled by
  `128^-0.5`.
- Two kinds of layers. `sliding_attention` (four out of five) rotates queries and keys (RoPE,
  absolute positions) and sees the last 1024 positions, the query's own included.
  `full_attention` (every fifth layer) sees everything and has no positions at all.
- Gated attention: the heads' outputs are multiplied by `sigmoid(g_proj(x))` before `o_proj`.
- Value residuals: in every layer the values are projected from
  `lerp(e, x, sigmoid(value_res_lambda))`, `e` being the token's scaled embedding and `x` the
  layer's normed input.
- A SwiGLU MLP eight times as wide as the stream (40960) with clamped activations: the gate
  branch at most 20, the other within plus and minus 20.
- An untied LM head.

The attention over the cache, its window and the rotary tables are Gemma's code
(`model::attention`); the model itself is small (`ripple.rs`). A config without the optional
parts (no gate, no value residuals, no scale, no clamp, no window, a tied head), which the
modeling code allows, is supported and checked too.

## Chat format

`chat::RippleChat` follows `chat_template.jinja`: ChatML turns written as they are (nothing is
trimmed), tools declared as JSON in the system turn, calls as `<tool_call>` around one JSON object
(`{"name": .., "arguments": {..}}`), results as `<tool_response>` in a user turn. The rendered
prompts are byte-identical to the template's (transformers' jinja rendering) with a system
prompt, with tools, with a past reply and through rounds of tool calling, nested arguments
included (`chat::tests::ripple_matches_template`).

The template has no thinking switch: the generation prompt is the bare assistant header and the
model decides to open `<think>` itself, so `--think` changes nothing here. `<think>`,
`</think>` and `<tool_call>` are not tokens of this vocabulary but plain text of several
tokens. The streamed text is therefore filtered on the decoded text (`chat::TextMarks`,
`generate::TextStream`): the marks are never shown, the reasoning is reported as thinking,
the line breaks that belong to the marks are dropped as Ricursive's own parser
(`ripple_format.py`) drops them, and the text stops at the first tool call. A unit test
(`ripple::tests::reply_marks_are_filtered`) holds that, token by token.

Not the template's: a past reply's reasoning is not kept (the template would write it back as
`<think>`, the reasoning, `</think>`); a reply counts as reasoning wherever `<think>` appears,
not only at its start; constrained replies (`response_format`, a forced `tool_choice`) are
refused, the grammars knowing no reply whose blocks are plain text; the per-token `content`
flag of log-probabilities does not see the reasoning.

The tokenizer (`tokenizer.json`, a cl100k-style BPE with a few added tokens) gives the ids of
the `tokenizers` library on 6 texts (36,759 tokens: Ricursive's own probes of digit runs,
contractions, whitespace runs and non-ASCII text, multilingual text, code, this README and the
model card).

## Agreement

The reference is `modeling_ricursive.py`, unchanged, run in float32 on the CPU with the cache it
builds by default (`tools/ref_ripple.py`). `examples/lm_check.rs` compares the logits at every
prompt position and after each of the reference's greedy tokens. The tiny checkpoint
(`tools/ripple_tiny.py`: 10 layers of 64, 8 heads of 16 over 2 key / value heads, layers 5 and
10 full, a window of 8, a clamp at 1.5, value residuals, the gate, an embedding scale of 8, an
untied head) has every feature on, and both bounds act: the window on every query past the
eighth token, the clamp on 20% of the gate branch's activations and 40% of the other's (without
it the reference's logits move by 5.9).

| check (bf16 weights) | prefill: max logit diff / mean KL | 16 greedy decode steps |
|---|---|---|
| tiny, 96 tokens | 2.2e-5 / 3.6e-12 (96/96 top-1) | 1.0e-5 / 2.1e-12 (16/16) |
| tiny, passes of 7 (both sides), then of 40, 7 and 1 (oxinfer) | 1.6e-5 to 2.2e-5 (96/96) | 1.1e-5 to 1.3e-5 (16/16) |
| tiny, 300 tokens in passes of 64 (both sides), then of 300 and 1 (oxinfer) | 1.9e-5 to 2.2e-5 / 3.4e-12 (300/300) | 9.4e-6 to 1.0e-5 (16/16) |
| tiny without the optional parts (`--plain`), 96 tokens, also in passes of 7 | 3.6e-5 / 9.6e-12 (96/96) | 2.3e-5 / 3.4e-12 (16/16) |

Logits reach 8 to 10 (33 for the plain variant). The same top token at every step means the
greedy continuations are identical.

On the real weights (the NVFP4 checkpoint), the model cut to its first 10 layers
(`--layers 10`: layers 1 to 10, two of them full attention, with the real final norm and head),
against the reference built the same way (`tools/ref_ripple.py --layers 10` reads the tensors
from the shards and decodes the NVFP4 matrices itself, 37 GB in float32). The prompt is 1300
tokens of the model card, so the window of 1024 acts on the last 276 positions:

| check (first 10 layers, real NVFP4 weights) | prefill: max logit diff / mean KL | 16 greedy decode steps |
|---|---|---|
| 1300 tokens in one pass | 1.4e-4 / 4.3e-11 (1300/1300 top-1) | 5.1e-5 / 3.4e-11 (16/16) |
| 1300 tokens in passes of 256 (oxinfer) | 1.4e-4 / 4.3e-11 (1300/1300) | 6.4e-5 / 3.6e-11 (16/16) |

Logits reach 17.6. A model cut at layer 10 does not write language (the reference's greedy
continuation settles on one token), but the weights, the scale, the norms and the window are the
released ones. Layers 11 to 80 were not compared with anything.

The whole NVFP4 model (80 layers, mapped, on 64 x86 cores, CPU): "What is the capital of
France? Answer in one sentence." gives "The capital of France is Paris."; with `--tools`, "What
is 1234 * 5678? Use the calculator." gives a reasoning block ("The user wants me to calculate
1234 * 5678 using the calculator tool."), the call `calculate({"expression": "1234 * 5678"})`,
and after the result "1234 × 5678 = **7,006,652**". That shows the layers beyond the tenth
working together, the marks filtered on real output and a tool round; it measures nothing.
Through the engine (`examples/chat.rs`), two turns: "A train leaves at 9:40 and arrives at
11:05. How long is the trip? Think it through first." gives "From 9:40 to 11:05 is 1 hour 25
minutes.", and "And if it left 20 minutes later?" gives "Then it would arrive at 11:25, so the
trip would still be 1 hour 25 minutes." (the model opened no reasoning block for these).

Speed there: a token reads 56 GB of weights; 0.6 to 0.9 tokens/s decoding (prompt-lookup rounds
included), 6 to 11 tokens/s in the prefill; 14 s to load (the 4-bit matrices are repacked).

A unit test without a reference (`ripple::tests::tiny_model`) holds that the logits do not
depend on how the tokens are split into passes, that a session cut back leaves no trace, and
that the window, the layer kinds, the gate, the value residuals, the clamp and the embedding
scale each change the logits.

```sh
./target/release/oxinfer --model ricursiveAI/Ripple-65B-Preview-NVFP4 -p "Why is the sky blue?"
# the checks above (CODE_DIR: the two .py files of ricursiveAI/Ripple-65B-Preview)
python tools/ripple_tiny.py tiny CODE_DIR && python tools/ref_ripple.py tiny CODE_DIR ref.safetensors 96 16
cargo run --release --example lm_check -- tiny ref.safetensors --chunk 7
python tools/ref_ripple.py MODEL_DIR CODE_DIR ref.safetensors 1300 16 0 --text prompt.txt --layers 10
cargo run --release --example lm_check -- MODEL_DIR ref.safetensors --layers 10 --mmap
```

## Checkpoints and memory

- bf16 (`ricursiveAI/Ripple-65B-Preview`, 130 GB): a token reads all of it but the embedding
  table. On a machine with less memory than that it is streamed from storage with a page-cache
  budget, like Mistral Large 4 (see [streaming](streaming.md)); `--quant q4` or `q8` quantizes
  at load. None of this was run on the real file: the streamed path and the quantization at
  load only ran on the tiny checkpoint.
- NVFP4 (`ricursiveAI/Ripple-65B-Preview-NVFP4`, 57 GB; ModelOpt's layout): the MLPs (50B of
  the 65B parameters) are 4-bit, the attention, the embeddings and the head bf16. The 4-bit
  matrices are repacked into ggml's NVFP4 blocks in memory at load (28 GB that cannot be
  streamed; the bf16 rest stays in the mapped files), so the model needs some 60 GB of memory.
  The checkpoint also carries an `input_scale` per MLP matrix and its config declares 4-bit
  input activations: Ricursive's local runtime (`inference/local`) has kernels that quantize
  what enters the MLPs' `down_proj`. oxinfer does not: the weights are the checkpoint's, the
  activations stay float.
- The key / value cache keeps every position of every layer (8 KB per token and layer, 0.66 MB
  per token in all): the sliding layers do not drop what leaves their window, so a session can
  be cut back to any position. A very long context costs memory accordingly (21 GB at 32k
  tokens).

## Not done

- A run of the bf16 checkpoint, and any comparison of the whole model with a float reference
  (above).
- Trimming the caches of the sliding layers to their window.
- The DFlash2 draft (`ricursiveAI/Ripple-65B-Preview-DFlash2`): not read, `--draft` is refused.
  Prompt-lookup rounds run (the caches are cut back on a rejected round).
- Quantized activations for the NVFP4 checkpoint (above), GGUF conversions, GPU decoding (no
  whole-pass graph for this family; the GPU, when present, runs the matmuls of long passes).
- The base model (`Ripple-65B-Base-Preview`) has the same architecture and should load; it was
  not tried.
- Padded batches and caller-given 4D masks of the modeling code: there are none here.
- Session snapshots.
- Runtime interventions use the points around each sublayer and the MLP's neurons; they were
  not measured on this architecture.
