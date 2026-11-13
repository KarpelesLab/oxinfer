# IBM Granite 4 and TwIL-8B

[Granite 4.1 / 4.2](https://huggingface.co/ibm-granite/granite-4.2-8b) (`model_type: granite`,
`GraniteForCausalLM`; 3B, 8B, 30B) and their fine-tunes, such as
[webAI-Official/TwIL-8B](https://huggingface.co/webAI-Official/TwIL-8B) (Granite 4.2 8B trained for
reasoning), run as safetensors folders and as GGUF files (`general.architecture = granite`).

```sh
./target/release/oxinfer --model ibm-granite/granite-4.2-3b -p "Why is the sky blue?"
./target/release/oxinfer --model ibm-granite/granite-4.2-3b-GGUF:Q4_K_M --chat --tools
./target/release/oxinfer --model webAI-Official/TwIL-8B:Q4_K_M-balanced --think -p "What is 17 * 23?"
./target/release/oxinfer --model webAI-Official/TwIL-8B --quant q8 --think --max-tokens 8192 -p "..."   # model-balanced.safetensors
./target/release/oxinfer serve --model twil=webAI-Official/TwIL-8B:Q4_K_M-balanced,think=1
```

TwIL-8B's repository holds two variants. The root is `balanced` (`model-balanced.safetensors`,
found through `model.safetensors.index.json`, and `TwIL-8B-Q4_K_M-balanced.gguf`); `verbose` has
GGUF files at the root (`:Q4_K_M-verbose`, `:Q8_0-verbose`, `:F16-verbose`) and its safetensors in
the `verbose/` folder (download that folder and pass its path). `:Q4_K_M` alone names two files,
so the variant is part of the name. The model is trained to think before it answers and its
traces are long (its card counts about 19.5k tokens per AIME problem): use `--think` and a large
`--max-tokens`.

## The model

Granite is a Llama-style decoder (RMSNorm, grouped-query attention with RoPE, SiLU-gated MLP, no
query / key norms) with four scales in its config, and runs on the Llama code path
(`crate::qwen`):

| Config key | Where it applies | Granite 4.2 |
|---|---|---|
| `attention_multiplier` | the attention scores, in place of `1 / sqrt(head_dim)` | 1/64 (3B), 1/128 (8B): `1 / head_dim` |
| `embedding_multiplier` | the token embeddings | 1 |
| `residual_multiplier` | each sublayer's output before it joins the residual | 1 |
| `logits_scaling` | divides the logits | 1 |

In GGUF files they are `granite.attention.scale`, `granite.embedding_scale`,
`granite.residual_scale` and `granite.logit_scale`. With the last three at 1 (Granite 4.1 / 4.2)
the model decodes on the GPU like Llama 3; other values (as older `granite` checkpoints have them)
are applied on the CPU path, checked on a random checkpoint only. A chat template that is not
ChatML (Granite 3's `<|start_of_role|>` turns) is not written: such a model is refused by the
engine and the server, and runs from the command line with `--raw` prompts. Other Granite
classes are not supported: `granitemoehybrid` (ibm-granite/granite-4.0-350m is one, although its
layers are all attention) and `granitemoe`.

GGUF files follow llama.cpp's Llama conversion (query / key rows permuted for adjacent-pair RoPE,
see [Llama 3 GGUF](weights.md#llama-3-gguf)). Their tokenizer is the `granite-docling`
pre-tokenizer: GPT-2's byte-level split without a normalizer, as the checkpoints' own
`tokenizer.json`.

## Chat format and tools

The template is ChatML with a system turn always present (empty without a system prompt) and no
BOS token. The reply opens with `<think>\n` when thinking is on (`--think`, the server's
`think=1`) and with an empty `<think></think>` otherwise. Tools are listed
after the system prompt as bare JSON functions (`{"name": ..., "description": ..., "parameters":
...}`) with the instructions of Qwen 3.5's template, and calls use the same format:

```text
<tool_call>
<function=get_weather>
<parameter=city>
Paris
</parameter>
</function>
</tool_call>
```

Results go back as `<tool_response>` blocks of one user turn. `--tools`, the server's `tools`,
`tool_choice` (forced calls) and `response_format` work as for Qwen 3.5 (see
[Tool calling](tool-calling.md)). The template's `{reasoning effort: low}` suffix is not written.

## Agreement with the reference

Against `transformers` 5.23 in float32 on the CPU (`tools/ref_logits.py`; the prompt as the chat
template renders it, logits of its last token, then greedy decoding):

| Check | Result |
|---|---|
| ibm-granite/granite-4.2-3b (bf16), "What is the capital of France? Answer in one sentence." | max logit difference 9e-5, same greedy tokens to the end of the reply |
| the same with thinking on (prompt ends in `<think>\n`) | max logit difference 1.3e-4, same 24 greedy tokens |
| webAI-Official/TwIL-8B (`model-balanced.safetensors`, bf16), "What is 17 * 23? Answer with the number only.", thinking on | max logit difference 2.0e-4, same 24 greedy tokens |
| a random 3-layer `granite` checkpoint with `embedding_multiplier` 12, `residual_multiplier` 0.22, `logits_scaling` 8, `attention_multiplier` 0.05 | max logit difference below 1e-6, same greedy tokens |
| granite-4.2-3b bf16 GGUF vs the bf16 safetensors, oxinfer both | max logit difference 1e-4, same greedy text |
| granite-4.2-3b Q8_0 / Q4_K_M GGUF vs the bf16 safetensors | same top 1, max logit difference 0.41 / 4.4 (mean 0.07 / 0.59), same greedy text |
| TwIL-8B Q4_K_M GGUF (`balanced`) vs its bf16 safetensors | same top 1, max logit difference 3.8 (mean 0.63) |
| GGUF tokenizer vs `tokenizer.json` | identical ids on 8811 texts (this repository's README, docs and some source files, the chat template, ten scripts, emoji, whitespace runs, special tokens), each decoding back to its text |
| GPU decoding (CUDA) vs the CPU path, granite-4.2-3b | same tokens, max logit difference below 1e-4 (`OXINFER_GPU_DECODE_CHECK=1`) |
| the chat template's renderings (system or not, tools, a call and its result, thinking on / off) | identical text (unit test `granite_matches_template`) |
| `tools/openai_check.py --chat` on granite-4.2-3b | all checks pass (tool calls, streamed calls, forced calls, JSON schema, reasoning, log probabilities) |

TwIL-8B's Q4_K_M GGUF (`balanced`) was run on the CPU and with CUDA: it answers `391` to
"What is 17 * 23? Answer with the number only." and reasons to the same answer with `--think`.
llama.cpp was not run on these files.
