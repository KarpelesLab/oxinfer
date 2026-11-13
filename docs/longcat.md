# LongCat-2.5, LongCat-2.0, LongCat-Flash-Lite-Sparse

`meituan-longcat/LongCat-2.5` (`pro_v`, `LongcatCausalLM`; 38 layers of 8192, 768 experts and 128
identity experts of which 12 per token, 1.6T parameters, 48B active, 3.56 TB of bf16 safetensors),
`meituan-longcat/LongCat-2.0` (the same text model, 3.55 TB) and
`meituan-longcat/LongCat-Flash-Lite-Sparse` (the same architecture at 14 layers of 3072, 69B
parameters, 138 GB) run on `longcat.rs`, text only. So do the two releases they extend, as far as
the model goes: LongCat-Flash-Lite (no sparse attention) and LongCat-Flash-Chat (neither sparse
attention nor n-gram embedding), see [the siblings](#the-siblings).

**LongCat-2.5 itself was never run**: a token reads 112 GB of weights and the checkpoint is 35
times the 100 GB this work could download. What follows was measured on tiny random checkpoints
against the published code, on the headers of the released files, and on the real weights of
LongCat-Flash-Lite-Sparse (below).

## Where the architecture comes from

The repository has no modeling code: its `auto_map` names `modeling_longcat_oe.py`, which is not
among its files, and no GGUF conversion or llama.cpp support exists. The architecture was put
together from three published sources, none of which runs LongCat-2.5 on its own:

1. **transformers' `LongcatFlashForCausalLM`** (`models/longcat_flash`, in transformers 5; the
   model of LongCat-Flash-Chat): the layer, MLA attention with its LoRA scales, the MoE with
   identity experts, YaRN. LongCat-2.5's tensors under `model.layers.*` have the names and shapes
   of that model's (the experts one module each, as LongCat-Flash-Chat's own checkpoint).
2. **Meituan's `modeling_longcat_ngram.py`** (in `meituan-longcat/LongCat-Flash-Lite`): the n-gram
   embedding. LongCat-2.5 stores its tables as `model.oe_embed_tokens{i}` / `model.oe_embed_proj{i}`
   where that file has `model.ngram_embeddings.embedders.{i}` / `post_projs.{i}`; the sizes are
   those the file computes (16 tables of `int(163840 * 100.567) + 2 i + 1` rows of 512).
3. **Meituan's SGLang-FluentLLM**, branch `longcat-2.5-npu` (commit 790693d, 2026-11-09), which
   LongCat-2.5's model card gives as its deployment: `models/longcat_flash.py` (the model:
   the same layer and MoE as 1), `models/deepseek_v32.py` and
   `layers/attention/dsa/nsa_indexer.py` (the attention with its indexer),
   `layers/embedding/over_embedding.py` with the kernels `compute_n_gram_ids` and
   `update_oe_token_table` of `flash-npu-kernel` (the n-gram ids, the tokens left out of them).
   It is a serving engine on CUDA and Ascend kernels and was read, not run (its CUDA kernels
   `routing_flash` and `compute_n_gram_ids_v2` are in the `feature/longcat_main` branch of
   meituan-longcat/flashinfer, the second in a version older than the one it calls: without the
   stop at an end-of-sequence token that the Ascend kernel and Meituan's Python have). What
   LongCat-2.x has beyond 1 and 2 comes from it alone: the sparse attention, `oe_ignore_tokens`,
   and an epsilon where it differs from transformers (below).

The model:

- A layer is two sublayers and one MoE ("shortcut-connected MoE"): attention, dense MLP (12288),
  attention, dense MLP on the residual stream, each behind an RMS norm; the MoE reads the normed
  stream the first dense MLP reads and its output is added at the end of the layer. LongCat-2.5
  has 38 layers: 76 attentions.
- MLA attention, as DeepSeek V3: queries through a LoRA (1536), keys and values expanded per head
  (64 heads of 128 + 64) from a 512-wide latent plus one rotary key shared by the heads
  (interleaved pairs). The query and the latent are scaled by `sqrt(hidden / rank)`
  (`mla_scale_q_lora`, `mla_scale_kv_lora`). Only the latent and the 64 rotated key lanes are
  cached (2.3 KB per token and attention); `kv_b` is absorbed into the query and the output, as
  for Mistral Large 4.
- RoPE with DeepSeek's YaRN (`rope_type: deepseek_yarn`, factor 120 over 8192 positions): the
  frequencies are transformers' `yarn` ones, the cos / sin magnitude is 1, and the softmax scale
  is multiplied by `(0.1 ln(120) + 1)^2 = 2.19`.
- Sparse attention ("LSA", `index_topk` 2048): the first attention of each layer
  (`cli_factor` 2) has a lightning indexer (32 heads of 128),
  `I(t, s) = d^-0.5 H^-0.5 sum_h weights_proj(x_t)_h relu(wq_b(c_t)_h . k_s)`, with `c_t` the
  normed query latent before its LoRA scale and `k_s = RMSNorm(wk(x_s))`, the first 64 lanes of
  both rotated like the attention's. A query attends to its 2048 best keys, among which the first
  16 (`index_init_tokens`) and its last 1024 (`index_local_tokens`) always are; to every key
  while it has no more than 2048. The second attention of the layer reuses the first one's choice.
  Up to 2048 tokens of context the model is plain causal attention.
- MoE: a softmax router over 896 experts, the 12 largest of probability plus a selection bias
  (`e_score_correction_bias`), each weighted by its probability (not renormalized) times 9
  (`routed_scaling_factor`). 768 experts are SwiGLU MLPs (2048); the last 128 are identity
  ("zero-computation") experts, which return their input.
- The n-gram embedding ("over-encoding", `oe_*`): for each token, 16 hashed tables (n-grams of 2
  to 5 tokens, 4 tables each) are read at
  `sum_i token[t - i] 163840^i mod (m + 2 j + 1)` over the last `g` tokens, the row of each goes
  through its own projection (512 to 8192), and the token's embedding is the mean of those 16
  and of its word embedding. The window stops at the start of the sequence, after an
  end-of-sequence token, and at a token of `oe_ignore_tokens` (199 to 223: the image, video and
  audio markers), which itself keeps its word embedding alone. The tables are 270 GB; a token
  reads 16 rows of 1 KB, straight from storage.

Where the sources disagree or say nothing:

- The RMS norms of the query LoRA and of the latent have epsilon `rms_norm_eps` (1e-5) in
  SGLang-FluentLLM and the module default 1e-6 in transformers. oxinfer follows SGLang-FluentLLM
  for `LongcatCausalLM` and transformers for the LongCat-Flash architectures. On trained weights
  the difference is below rounding.
- An identity expert's weight is its probability times `routed_scaling_factor` in transformers,
  in Meituan's `modeling_longcat_flash.py` and in SGLang-FluentLLM, and its probability alone in
  upstream SGLang's CUDA path. oxinfer scales it; the real weights of LongCat-Flash-Lite-Sparse
  agree ([below](#on-real-weights-longcat-flash-lite-sparse)).
- Equal indexer scores (a key for which every index head's product is negative scores exactly 0):
  SGLang-FluentLLM's top-k is a CUDA kernel whose order among equals is not stated; here the
  earlier key wins.
- `moe_impl: mix` and `moe_switch_token_num` choose between two kernels of the serving engine by
  batch size; `index_dtype` is the precision of its index-key cache. They do not change the
  arithmetic and are not read. The model card's "Hierarchical Indexing" (a coarse recall of
  blocks before the indexer scores tokens) is an optional speed-up of serving, off in the
  numbers the card reports as best; it is not implemented.
- The MTP layers (`model.mtp.*`, three speculative-decoding heads) are not read.

## Checkpoints

The released bf16 safetensors as they are: one module per expert
(`model.layers.{l}.mlp.experts.{e}.{gate,up,down}_proj`), the router and the indexer's
`weights_proj` in float32. Larger than memory, the checkpoint is streamed from storage like
Mistral Large 4 ([streaming](streaming.md)): the dense weights of each layer as a unit, the
routed experts on demand, the rows of the n-gram tables as the tokens need them.
`examples/longcat_check MODEL --load` opens a checkpoint without reading its data: on sparse
local files holding the real headers of all 210 shards of LongCat-2.5 and all 194 of LongCat-2.0
(fetched by range request), the loader finds every tensor it wants with the shape the config
implies and reports 112.2 GB read per token (63.5 GB of dense layers, 46 GB of experts when all
12 are routed ones, 2.7 GB of LM head).

`meituan-longcat/LongCat-2.5-W4A4` (and the INT8 / FP8 releases of LongCat-2.0) are refused at
load: their experts are 4-bit integers used with 4-bit activations behind learned transforms
(`w1_trans.matrix_left`, `act_quantizer_w1.clip_factor_a`, per-channel `weight_scale`), and the
dense MLPs 8-bit with `smooth_scale`. That arithmetic belongs to the checkpoint and was not
implemented.

The tokenizer is a `tokenizer.json` (byte-level BPE behind five script-wise splits). The ids
equal those of the `tokenizers` library on 295 texts (7914 tokens: several scripts, code,
whitespace runs, random strings, glued special tokens), for LongCat-2.5's vocabulary (163840)
and for LongCat-Flash-Lite-Sparse's (131072; 8578 tokens).

## Chat format

`longcat_chat.rs` follows the checkpoint's `chat_template.jinja`. A conversation has no opening
token. Tools are declared first, in a `<longcat_tool_declare>` block, then `<longcat_system>`,
`<longcat_user>`, `<longcat_assistant>` and `<longcat_observation>` turns follow each other with
nothing between them. The turn before a reply ends with ` /think_on ` or ` /think_off ` and the
reply opens with `<longcat_think>`, left open in thinking mode (`--think`) and closed at once
(`\n\n</longcat_think>\n`) otherwise; a reply ends with `</longcat_s>`. A tool call is
`<longcat_tool_call>NAME` and one `<longcat_arg_key>` / `<longcat_arg_value>` pair per line
(strings bare, other values as JSON); results come in one `<longcat_observation>` turn as
`<longcat_tool_response>NAME\nCONTENT</longcat_tool_response>` blocks. Calls are parsed as
SGLang-FluentLLM's `longcat_xml` detector does (the name is the first line, values are trimmed,
a parameter declared as a string keeps its text, others are JSON when they parse).

The rendered prompts are byte-identical to the template's (transformers' jinja environment) with
and without thinking, with a system prompt, with two tools, through a round of two tool calls
with their results, and over a following turn; a unit test holds them
(`longcat_chat::tests::matches_template`). LongCat-2.0's template renders the same text.

Not rendered: `thinking_budget`, `files`, the `code_interpreter` tool type, a past reply's
reasoning (a past reply is rendered with an empty think block, as the template does for a message
without `reasoning_content`), images and video. Constrained replies were not tried: the format
declares no grammar for its call syntax, so a forced `tool_choice` cannot be held to it.

## Agreement

The reference (`tools/ref_longcat.py`, float32 on the CPU) is transformers 5.23's
`LongcatFlashForCausalLM`, unmodified, with the released tensors renamed onto it, fed the
embeddings of Meituan's `NgramEmbedding`. For the LongCat-Flash architectures that is all of it.
For `LongcatCausalLM` three things are added to it, **transcribed by hand from
SGLang-FluentLLM**: the indexer and its top-k as a mask on the attention (so two readings of the
same source are being compared there, not two implementations), the ignored tokens as extra
n-gram boundaries, and the epsilon above. `deepseek_yarn` is run as transformers' `yarn`, with
SGLang-FluentLLM's frequency formula asserted equal.

The tiny checkpoint (`tools/longcat_tiny.py`) is in the released layout with every feature of the
text model: 3 layers (6 attentions) of 64, MLA with both LoRA scales and YaRN, indexers on
attentions 0, 2 and 4 (4 heads of 16, top 12 with 2 first and 3 last keys always attended, so
most queries are sparse), 6 experts and 3 identity experts (top 3), 16 n-gram tables with
`oe_ignore_tokens`, the prompts holding end-of-sequence and ignored tokens. Weights are bf16 as
released, the arithmetic float32. `examples/longcat_check.rs` compares.

| check | prefill: max logit diff / mean KL | greedy decode steps |
|---|---|---|
| LongCat-2.5's architecture, 96 tokens | 1.4e-5 / 1.7e-12 (96/96 top-1) | 9.5e-6 / 1.8e-12 (16/16) |
| the same in passes of 7 and of 1 | 1.4e-5 / 1.7e-12 (96/96) | 9.3e-6 / 1.6e-12 (16/16) |
| 160 tokens, the reference in passes of 16; ours in one pass and in passes of 5 | 1.3e-5, 1.2e-5 / 1.3e-12 (160/160) | 8.6e-6, 1.1e-5 / 1.2e-12 (24/24) |
| LongCat-Flash-Lite's (`--ngram`: no indexer; published code only), 96 tokens in passes of 96, 7, 1 | 9.8e-6, 9.4e-6, 9.4e-6 / 1.2e-12 (96/96) | 6.9e-6, 7.2e-6, 7.2e-6 / 8e-13 (16/16) |
| LongCat-Flash-Chat's (`--flash`: published code only), 96 tokens in passes of 96, 7, 1 | 4.8e-6, 5.0e-6, 5.0e-6 / 3.4e-13 (96/96) | 3.8e-6, 2.9e-6, 2.9e-6 / 2.4e-13 (16/16) |

Logits reach 9. The same top token at every step means identical greedy continuations. As a
check that the comparison sees the mechanisms: the same reference against the same weights with
`index_topk` raised above the context gives a max logit difference of 10.9 (16/96 top tokens),
and with `oe_ignore_tokens` emptied 12.4 (38/96).

From the command line, on a tiny checkpoint with the real vocabulary and tokenizer, the prompt
`oxinfer --print-prompt` shows is the template's and the 12 greedy tokens that follow are the
reference's for those prompt ids; they are the same with `--no-lookup` and with `--stream`.

A unit test without a reference (`longcat::tests::tiny_model`) holds that the logits do not
depend on how the tokens are split into passes, that a rollback leaves no trace, and that the
indexer's choice, the keys always attended, the identity experts, the ignored tokens, the end of
a sequence, each LoRA scale and YaRN change the logits.

```sh
# every shard in one folder (3.56 TB); streamed from storage, 112 GB read per token
./target/release/oxinfer --model meituan-longcat/LongCat-2.5 --think -p "Why is the sky blue?"
# the checks above
python tools/longcat_tiny.py tiny && python tools/ref_longcat.py tiny CODE_DIR ref.safetensors 96 16
cargo run --release --example longcat_check -- tiny ref.safetensors --chunk 7
```

`CODE_DIR` holds `modeling_longcat_ngram.py` and `configuration_longcat_ngram.py` of
meituan-longcat/LongCat-Flash-Lite; the reference needs transformers 5 (`models/longcat_flash`).

## On real weights: LongCat-Flash-Lite-Sparse

The smallest release of the architecture is 138.3 GB: 74.5 GB of weights in 14 shards, and 12
shards that are each one n-gram table (5.2 GB) plus 84 MB of other tensors. To stay under the
download budget the 14 shards were downloaded whole (their sha256 checked against the
repository's) and the other 12 as sparse files holding the real header, the small tensors, and
only the table rows the tokens of the test read, fetched by range request (512 bytes a row): 76
GB on disk in all. The model is then exact for those token sequences and for no other, so there
was no free chat; what was run:

- **A prompt from the command line.** `oxinfer --model DIR --no-system -p "What is the capital
  of France?"` answers `The capital of France is **Paris**.` and stops on `</longcat_s>` (9
  tokens, greedy; the rows of each new token fetched before the run that feeds it). The
  checkpoint is opened memory-mapped and streamed (9.1 GB read per token); on this machine (64
  cores, 134 GB of RAM, CPU only) the prompt runs at 2.6 to 3.0 tokens/s and decoding at 3.6 to
  4.0 tokens/s. That is one short reply, not a benchmark.
- **A round of tool calling.** With `--tools` (the built-in calculator and clock, declared as the
  template declares tools; a prompt of 247 tokens, 28 tokens/s) and `-p "What is 1234 * 5678?
  Use the calculator."`, the model writes
  `<longcat_tool_call>calculate`, `<longcat_arg_key>expression</longcat_arg_key>`,
  `<longcat_arg_value>1234 * 5678</longcat_arg_value>`, `</longcat_tool_call>` on a line each
  and stops; the call is parsed and run, its result goes back as a `<longcat_observation>` turn,
  and the reply is `The result of 1234 * 5678 is **7,006,652**.` (decoding at 4.9 to 7.4
  tokens/s). The declaration, the call syntax, the parser and the results turn are the ones
  LongCat-2.5's template has.
- **The likelihood of a text.** The first 1024 tokens of *Alice's Adventures in Wonderland*
  (Project Gutenberg's text, which the model has plainly seen) get a mean negative
  log-likelihood of 0.169 nats per token (perplexity 1.18; the top token is the text's 980 times
  of 1023), the same in passes of 1024, 256 and 37 tokens
  (`longcat_check DIR --nll IDS`). The same measure with one thing changed at a time tells which
  readings of the sources the weights agree with:

| changed | NLL (nats / token) | top-1 of 1023 |
|---|---|---|
| nothing | 0.169 | 980 |
| no n-gram embedding (the word embedding alone) | 4.303 | 246 |
| the latent without its LoRA scale | 4.116 | 249 |
| the softmax scale without YaRN's `mscale^2` | 2.020 | 610 |
| plain RoPE (no YaRN frequencies, no `mscale^2`) | 1.365 | 779 |
| identity experts off | 0.253 | 957 |
| identity experts weighted by their probability alone (not times `routed_scaling_factor`) | 0.221 | 963 |
| the query without its LoRA scale | 0.173 | 979 |
| the LoRA norms' epsilon 1e-6 instead of 1e-5 | 0.169 | 980 |

  The identity experts are where the published code disagrees: transformers, Meituan's
  `modeling_longcat_flash.py` and both paths of SGLang-FluentLLM (its `routing_flash` kernel, in
  the `feature/longcat_main` branch of meituan-longcat/flashinfer, scales every chosen weight)
  multiply an identity expert's weight by `routed_scaling_factor`; upstream SGLang's CUDA path
  (`models/longcat_flash.py`, PR 32918 for this very checkpoint) adds the identity experts after
  scaling the routed ones only. The weights prefer the former, which is what oxinfer does.
  The softmax scale as a whole (`192^-0.5` times `mscale^2` times the query's LoRA scale) sits
  at the bottom of the curve: times 0.6, 0.8, 1.25 and 1.6 it gives 0.301, 0.177, 0.188 and
  0.258. (The rows that are not a config key were measured with temporary switches in the code,
  which are not in the tree; the others by editing config.json.)
- **The indexer.** 1024 tokens are below `index_topk` (2048), so the rows above are dense
  attention. With `index_topk` lowered in config.json the indexer chooses on the real weights:

| attention | NLL | top-1 of 1023 |
|---|---|---|
| every key (as released, up to 2048 tokens) | 0.169 | 980 |
| the 2 first and 16 last keys, and nothing chosen | 1.201 | 762 |
| those and the indexer's best 14 (`index_topk` 32) | 0.197 | 971 |
| the same, the index lanes rotated as two halves instead of pairs | 0.456 | 907 |
| the 2 first, 16 last and the indexer's best 110 (`index_topk` 128) | 0.157 | 987 |
| the same, rotated as two halves | 0.255 | 963 |
| the indexer's best 128, no key always attended | 0.159 | 986 |

  Fourteen keys chosen among up to a thousand recover nearly all of what dense attention gives,
  and the choice is clearly worse with the index lanes rotated the other way: the indexer as
  read from SGLang-FluentLLM is the one these weights were trained with.
- **Beyond `index_topk`.** On 3072 tokens of the same text (a prompt pass of 512 tokens at a
  time, 79 tokens/s), over the 1023 tokens predicted from more than 2048 keys, which is where the
  released settings choose (16 first keys, 1024 last, 1008 by score):

| attention | NLL of tokens 2049 to 3071 |
|---|---|
| as released | 0.255 |
| every key (`index_topk` above the context) | 0.254 |
| as released, the index lanes rotated as two halves | 0.258 |
| the best 2048, no key always attended | 0.256 |
| `index_topk` 256: the 2 first, 128 last and the best 126 | 0.225 |
| the same, rotated as two halves | 0.236 |
| only the 2 first and 128 last keys | 0.351 |

  At the released settings sparse attention costs nothing measurable against attending to
  everything, as its authors report; a context of 3072 is too short for the released top 2048 to
  tell a right indexer from a wrong one, which the smaller budgets above do.

What this does not show: nothing compares oxinfer's logits with another implementation on real
weights (SGLang, the only engine that serves this checkpoint, was not run: its model card asks
for a 141 GB GPU and this machine has one of 17 GB), LongCat-2.5's own weights were not touched,
and `oe_ignore_tokens` (which LongCat-Flash-Lite-Sparse does not have) is only verified on tiny
checkpoints.

## The siblings

- LongCat-2.0: the text model of LongCat-2.5 (the same config keys but the vision ones, the same
  tensor names; the loader maps the released headers). Its template renders as LongCat-2.5's.
- LongCat-Flash-Lite-Sparse (`LongcatCausalLM` without `model_type`; 14 layers of 3072, 256
  experts and 128 identity ones, 12 n-gram tables of 10.2 M rows): above. Its template writes the
  thinking marks only when asked and only on the last turn, which a conversation extended turn by
  turn cannot follow; it is not a thinking model, so replies open with `<longcat_assistant>`
  alone (the template's default) and `--think` is ignored.
- LongCat-Flash-Lite (`LongcatFlashNgramForCausalLM`): the model without the indexer, the n-gram
  tables under `model.ngram_embeddings.*`. Agreement on a tiny checkpoint with the published code
  alone (table above); its template (tool results in a `<longcat_user>` turn, without the tools'
  names) is rendered. Not run on its real weights (138 GB).
- LongCat-Flash-Chat (`LongcatFlashForCausalLM`): the model loads and agrees with transformers on
  a tiny checkpoint, but its chat format (`[Round n] USER:` ... `ASSISTANT:`, calls as JSON) is
  not implemented: the checkpoint is refused with that message. LongCat-Flash-Thinking, -Omni and
  -Prover were not looked at.

A config.json without `model_type` (all of them but LongCat-2.5's) is recognized by its
`architectures`.

## Not done

- A run of LongCat-2.5 or LongCat-2.0 on the real weights: no speed, no text quality. At 112 GB
  read per token, decoding needs most of that in the page cache.
- Image and video input: the vision tower (a Qwen2.5-style ViT, `model.visual_model.*`), its
  residual quantizer (8 levels over a 16384-entry codebook) and the bridge that embeds the codes
  (`model.visual_bridge_model.*`) are not read. Their code is in SGLang-FluentLLM
  (`models/longcat_flash_vl.py`, `multimodal/processors/longcat_flash_vl.py`).
- The quantized releases (W4A4, INT8, FP8), MTP speculative decoding, Hierarchical Indexing.
- An expanded-key attention for long prefill passes (Mistral Large 4 has one): every pass scores
  against the latent cache, which is the slower way for prompts of thousands of tokens.
- GPU decoding (no whole-pass graph for this family), `--quant q8` / `q4` on checkpoints of this
  size (the quantization pass holds the weights in memory), session snapshots.
- Runtime interventions: the residual points exist (after the first sublayer and at the end of
  each layer) but no neurons are offered, and nothing was measured.
- Prompt lookup runs (every pass can be rolled back, unit-tested); it was only exercised on
  random weights, where it drafts nothing.
