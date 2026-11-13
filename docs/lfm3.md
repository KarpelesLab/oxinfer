# Liquid LFM3

[LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) (`model_type: lfm3`,
`Lfm3ForCausalLM`; 1.7B parameters, 3.5 GB of bf16 safetensors) is a hybrid of gated short
convolutions and attention, made for phones and laptops.

```sh
./target/release/oxinfer --model LiquidAI/LFM3-1.7B -p "Why is the sky blue?"
./target/release/oxinfer --model LiquidAI/LFM3-1.7B --chat --think --tools
./target/release/oxinfer --model LiquidAI/LFM3-1.7B --quant q8 -p "..."     # 8-bit weights at load
./target/release/oxinfer serve --model lfm=LiquidAI/LFM3-1.7B
```

## The model

24 layers, each named by `layer_types`: 14 `conv`, 5 `full_attention`, 5 `sliding_attention`
(full attention on every fourth layer from the third, sliding attention between them). It runs on
the Qwen code path (`crate::qwen`), the conv as one more mixer (`crate::lfm3`):

```text
layer:  x += post_operator_norm(op(operator_norm(x)))        # op: conv or attention
        x += post_ffn_norm(w2(silu(w1(n)) * w3(n))),  n = ffn_norm(x)
conv:   [B | C | u] = in_proj(n);  y = out_proj(C * conv7(B * u))
attn:   q, k RMS-normed per head, RoPE, softmax over the causal keys;
        each head's output times sigmoid(gate_proj(n)[head]); o_proj
```

- **Conv blocks**: a depthwise causal convolution of width 7 (`conv_L_cache`) over the product of
  two of the three projections, gated by the third. Its state is the last 6 inputs per channel
  (12288 floats per conv layer): recurrent state, like Qwen 3.5's Gated DeltaNet. A conversation
  whose earlier turns change starts over instead of being cut back (the prompt cache stores
  snapshots, as for Qwen 3.5).
- **Sandwich norm** (`use_sandwich_norm`): an RMSNorm before each operator and MLP, and another
  on its output before it joins the residual.
- **Attention**: 16 heads of 128 over 4 key / value heads, query / key norms, and a sigmoid gate
  per head (`gate_proj`, one row per head). `sliding_attention` layers see the last 256 keys
  (`sliding_window`), the query's own included; the whole KV cache is kept.
- **RoPE per layer type** (`rope_parameters`): base 1e6 with `partial_rotary_factor` 0.5 for full
  attention, base 1e4 for sliding attention. The partial rotation keeps the frequency progression
  of the whole head dimension and zeroes the second half of the frequencies: of the rotate-half
  pairs `(i, i + 64)`, those with `i < 32` rotate.
- The final norm is `embedding_norm`; the output head is tied to the embeddings.

Decoding runs on the CPU kernels (the GPU decode plans do not cover the conv and the gated heads;
the prefill matmuls still use a GPU when there is one). Speculative decoding (prompt lookup) is
on as for other models; MTP and DSpark do not apply.

`transformers` 5.23 has an `lfm3` class of its own whose attention gate is named `g_proj`: loaded
without `trust_remote_code`, it leaves the checkpoint's `gate_proj` weights unused and the gate
randomly initialized, and it does not apply the partial rotation. The reference here is the
modeling code the repository ships (`modeling_lfm3.py`, `tools/ref_logits.py --trust-remote-code`).

## Chat format and tools

`<|startoftext|>`, then ChatML turns. The system turn holds the system prompt and, after it, the
tools as bare JSON functions between `<tools>` and `</tools>` (no instructions; without a system
prompt or tools there is no system turn). The reply opens with `<think>` when thinking is on and
with an empty `<think></think>` otherwise. Tool calls are Qwen 3.5's
`<tool_call><function=NAME><parameter=KEY>` blocks; each result goes back in a `tool` turn as
`<tool_response>`. `--tools`, the server's `tools`, forced calls and `response_format` work as for
Qwen 3.5 (see [Tool calling](tool-calling.md)).

## Agreement with the reference

Against the repository's modeling code on `transformers` 5.23, float32, CPU
(`tools/ref_logits.py --trust-remote-code`; the prompt as the chat template renders it, logits
of its last token, then 24 greedy tokens):

| Check | Result |
|---|---|
| "What is the capital of France? Answer in one sentence." (23 tokens) | max logit difference 6e-5, same greedy tokens to the end of the reply |
| the same with thinking on (prompt ends in `<think>`) | max logit difference 5e-5, same 24 greedy tokens |
| a 1301-token prompt (notes to summarise; past the 256-key window of the sliding layers) | max logit difference 1.1e-4, same 24 greedy tokens |
| the 1301-token prompt fed one token at a time, or in chunks of 100, vs in one pass (conv state, windows, both RoPEs) | max logit difference 1e-4, same top 5 |
| 300 greedy tokens after that prompt: prompt lookup off, on (128 of 210 drafts rejected: the conv states rewound), and on CUDA | identical tokens |
| the chat template's renderings (system or not, tools, a call and its result, thinking on / off) | identical text (unit test `lfm3_matches_template`) |
| `tools/openai_check.py --chat` | all checks pass (tool calls, streamed and forced calls, JSON schema, reasoning, log probabilities) |

The greedy comparison runs the reference without a repetition penalty. The checkpoint's
`generation_config.json` asks for temperature 0.1, top-k 50 and a repetition penalty of 1.1, which
`transformers` applies to the prompt's tokens too, and that changes what the model does with
thinking off: on the 1301-token prompt above the raw logits put `<think>` first (20.2, against
19.6 for the first word of the answer) although the prompt ends in an empty `<think></think>`, so
plain greedy decoding thinks anyway, in the reference as here; the penalty demotes `<think>`
(it is in the prompt) and the reference then answers directly. oxinfer's default is greedy without
a penalty, and `--repetition-penalty` covers the generated tokens only, so with thinking off the
model may still open a thinking block on such prompts (the command line shows it dimmed, as
thinking). Short prompts answer directly.
