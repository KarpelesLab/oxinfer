# Inkling-2 Small

`thinkingmachines/Inkling-2-Small` and `thinkingmachines/Inkling-2-Small-MXFP8` (`inkling2`,
`Inkling2ForConditionalGeneration`; 594B parameters of which 17B per token, 48 layers of 4608,
384 experts of which 8 per token, a 1M-token context, text and images in, text out; 393.5 GB with
NVFP4 experts, 622.8 GB with MXFP8 experts) run on `src/inkling.rs` (the decoder) and
`src/inkling_vision.rs` (the vision tower).

**The decoder was never run on the real weights.** Neither release fits the 100 GB this work
could download, and neither does any GGUF conversion (the smallest, UD-IQ1_S of
`unsloth/Inkling-2-Small-GGUF`, is 172 GB; GGUF files of this model are not read, see below).
The decoder was measured on tiny random checkpoints against transformers, the loader on the
headers of both releases, the reading of the NVFP4 experts on a few real ones, the tokenizer and
the chat template on the released files. The vision tower (0.66 GB) was run on the real weights.

Whether the released model can be run at all depends on the machine: the MXFP8 release keeps
73 GB of scale tables in memory and streams the rest from storage, the NVFP4 release needs
290 GB of memory (see [Checkpoints](#checkpoints)).

```sh
oxinfer --model thinkingmachines/Inkling-2-Small-MXFP8 --prompt "Why is the sky blue?" --think
oxinfer --model thinkingmachines/Inkling-2-Small-MXFP8 --image photo.png --prompt "Describe this image."
```

## Architecture

As transformers' `modeling_inkling2.py` (below). Nothing in it is a standard pre-norm decoder
layer:

- **No positions in the stream.** The token embedding goes through an RMS norm; there is no
  position embedding and no rotary embedding. Attention gets its positions from a relative bias
  and from short causal convolutions.
- **Attention** is grouped-query (48 heads of 128 on 8 key/value heads) with a softmax, and:
  queries are scaled per channel (a learned `[heads, 128]` table) and then clipped to an L2 norm
  of `qk_l2_clip_threshold` (22.6 = 4 sqrt(32)); keys are clipped to the same norm; values go
  through an RMS norm without a gain. A fourth projection gives each query head a vector of
  `d_rel` (16), scaled and clipped (8) like a query, which mixes a learned bank of 16
  bias-versus-distance profiles (`rel_logits_proj`, `[16, 1024]`) into one bias per backward
  distance below 1024; further back the bias is 0. Each query also has a **sink**: one more
  logit, a projection of the layer input, whose value is zero, so it only takes probability away
  from the keys. The attended values are **gated** per channel by `sigmoid(gate_proj x)` before
  the output projection.
- **Short convolutions** (kernel 4, depthwise, causal, with a skip): on the keys and on the
  values before the clip and the norm, on the attention output and on the MLP output:
  `(conv(x) + x) / sqrt(5)`.
- **Local and global layers.** 40 of the 48 layers (`local_layer_ids`: all but every sixth) use
  sliding-window attention over the last 1024 tokens, and could have their own head counts
  (`swa_*`; the same as the others in this release). The 8 global layers see the whole context.
  There, queries and relative vectors are multiplied by `1 + 0.1 ln(max(1, (pos + 1) / 128000))`
  (nothing below 128000 tokens), and the key / value convolutions add to their learned taps a tap
  per token, key/value head and lag (a projection of the layer input), over `sqrt(9)`.
- **Attention residual.** There is no residual stream across blocks of 6 layers. The input of
  the first layer of each block is kept as a block state. Before its attention, before its MLP,
  and before the final norm, a layer reads a softmax mix of the block states so far and of the
  stream inside the current block: scores are a learned query against the RMS-normed states over
  `sqrt(hidden)`, and the mix is of the normed states. A global layer reads three mixes, one each
  for what its queries, keys and values are projected from, each with its own RMS norm. Inside a
  block the stream is the sum of the layers' outputs; the first layer of a block starts it again
  from its attention output alone.
- **MoE** on every layer, without a shared expert: sigmoid scores, the 8 best of score plus a
  selection bias, their scores renormalized and multiplied by `route_scale` (8). An expert is a
  SwiGLU with soft caps, `4 tanh(g / 4) sigmoid(g) * 25 tanh(u / 25)`
  ([ERNIE 5.5](ernie.md)'s SiTU with other constants).
- **Logits**: `30 tanh(l / 30)`, and only the first 200058 of the 201024 rows of the unembedding
  are tokens (`unpadded_vocab_size`). Embedding and unembedding are separate tensors.
- **Vision tower**: a 23-layer pre-norm ViT of 1024 with LayerScale and a class token. An image
  is not resized: it is padded on the right and at the bottom to whole tiles of 40 x 40 pixels,
  and each tile is one token of the language model (a 1024 x 768 picture is 520 tokens). A tile
  is 2 x 2 native patches of 20 pixels, embedded by a linear layer plus a learned position (a
  32 x 32 table resampled bilinearly to the picture's patch grid); axial rotary positions in the
  attention (half of each head turns with the column, half with the row); the four patches of a
  tile, concatenated, go through a two-layer GELU MLP to 4608. The features replace the
  placeholder tokens after the embedding norm.

Not implemented: the **MTP head** (`mtp.safetensors`, 3.4 GB: 5 layers that read the main
model's states at 8 depths, a block size of 8 and a Markov head of rank 256). transformers does
not run it either, so there is nothing to compare it with yet. The config's
`temporal_patch_size` (video) is unused by transformers too. The reference processor composites
transparent images onto a checkerboard; oxinfer decodes images to RGB first.

The convolutions keep only their last three inputs, so a session cannot go back to an earlier
position: there is no prompt lookup and no draft model for this family. The cache of a local
layer holds at most 2048 positions whatever the context length.

## Checkpoints

The releases' tensor names (`model.llm.*`, `model.visual.*`), which is also what transformers'
`save_pretrained` writes. The routed experts are two tensors per layer, `w13_weight`
(`[384, 4608, 4608]`: gate and up rows interleaved, gate first) and `w2_weight`
(`[384, 4608, 2304]`), in one of three forms:

- **MXFP8** (all 48 layers of the `-MXFP8` release, layers 40-47 of the other): E4M3 bytes with
  an E8M0 exponent byte per 32 values along a row (`*_scale`). Read in place; the exponents are
  expanded to float32 at load, 73 GB of memory for the whole model.
- **NVFP4** (layers 0-39 of `Inkling-2-Small`): two E2M1 codes per byte (low nibble first), an
  E4M3 scale per 16 values, and per expert `amax / outer_scale_denominator` (the denominator is
  2688 = 6 x 448). They are repacked at load into ggml's NVFP4 blocks, which takes the experts'
  size in memory (290 GB): this release needs a machine that holds it. On a smaller one the
  MXFP8 release is the one to use (its experts are streamed, see [streaming](streaming.md)).
- bf16 or float32 (what `save_pretrained` writes).

A decoded token reads 23.5 GB of the MXFP8 release (9.4 GB of attention weights, 1.9 GB of
unembedding, 12.2 GB of experts) and 19.0 GB of the NVFP4 one.

`unsloth/Inkling-2-Small-GGUF` (llama.cpp's `inkling2` architecture, 172 GB for UD-IQ1_S to
484 GB) is **not read**. Its headers were: the tensors map one to one onto the release's
(`blk.N.attn_q_clip`, `attn_rel_proj`, `shortconv_k`, `shortconv_k_dyn`, `attn_res_score`,
`ffn_res_score`, `exp_probs_b`, ...), with the experts split into gate / up / down and a
`.scale` of 384 values beside each, whose meaning has to come from llama.cpp's code. No
quantization of it fits the download budget of this work, so support for it could not have been
checked on real weights either.

## Chat format

`chat_template.jinja`: a conversation is a sequence of messages, each `<|message_ROLE|>`, an
optional name, a content kind, the content and `<|end_message|>`:

```
<|message_system|><|content_text|>Be brief.<|end_message|>
<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>
<|message_user|><|content_text|>What is the capital of France?<|end_message|>
```

(one line here per message; there is nothing between them). There is no generation prompt: the
model opens its own messages, `<|message_model|><|content_thinking|>...<|end_message|>` then
`<|message_model|><|content_text|>...<|end_message|>`, and ends its turn with
`<|content_model_end_sampling|>`.

- The thinking effort is a system message in front of the first message that is not a system
  one: 0.9 with `--think` (the template's default, "high"), 0 without.
- Tools are declared in a first system message, `<|message_system|>tool_declare<|content_xml|>`
  plus one JSON array with sorted keys and no spaces. A call is a model message named after the
  function, `<|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}`, and a
  result a tool message, `<|message_tool|>NAME<|content_text|>...`. Forced tool calls
  (`tool_choice`) are not available for this format.
- An image is a user message of its own before the text,
  `<|message_user|><|content_image|>` and one `<|unused_200054|>` per tile.
- In a reply, the name in a message header is not shown as text.

The prompts are byte-identical to the template's rendering (system prompt, thinking on and off,
tools, a reply with reasoning, text and two calls, tool results by name and by call id, a
further turn, an image: `chat::tests::inkling_matches_template` and the runs below), and the
tokenizer gives the token ids of the `tokenizers` library on them.

## Reference

The repository has no model code. transformers has the model since 5.24 (unreleased at the time:
`src/transformers/models/inkling2/` on its main branch, with the release's names mapped in
`conversion_mapping.py`); the model card names it among the ways to run the model, next to
SGLang, vLLM and TokenSpeed. `tools/inkling2_tiny.py` runs it in float32 on the CPU with its
eager attention (the only one it has for this model), uncached for the prompt and through its
cache token by token.

Two points were read in a second source. The router's bias is added to the sigmoid scores for
the choice of the experts only, not to the logits: SGLang's `inkling_common/moe.py` does the
same. And transformers cannot load the NVFP4 experts ("Loading pre-quantized NVFP4 checkpoints
is not supported yet"): the layout comes from the headers and from TokenSpeed's description of
it (`runtime/layers/quantization/inkling2.py`), and was checked on the real files, a few
megabytes of them read by HTTP range request:

- for the first two experts of layer 0, the largest `|code * scale|` is exactly 2688, so that
  `amax / 2688` gives back `amax`, and every decoded row has an L2 norm of 1.0000 (the experts'
  rows are unit vectors);
- expert 3 of layer 0, decoded from the NVFP4 release, has a cosine of 0.995 with the same
  expert decoded from the MXFP8 release, for `w13_weight` and for `w2_weight` (the rest is the
  error of 4 bits); with the two codes of a byte in the other order the cosine is 0.000.

So the NVFP4 and the MXFP8 readings of the experts are checked against each other on real
weights, though not against a model output.

## Verification

`tools/inkling2_tiny.py` builds a tiny random checkpoint in the release's layout with every text
feature on: 12 layers of 64 in two blocks of 6 (ten local layers of 4 heads of 8 on 2 key/value
heads with a window of 16, two global layers of 8 heads of 16), a relative extent of 24, log
scaling from position 32 with alpha 0.3, clip thresholds that bite on part of the vectors
(41% of the local queries, 88% of the global ones, 73% of the global keys, 36-40% of the
relative vectors), attention sinks that take 4-17% of the probability, 16 experts of which 4,
soft caps of 0.8 and 1.5 (they change the gate by 47% and `up` by 30% on average), a logit cap
of 3 (14%), an unembedding of 512 rows for 500 tokens, and random values in everything
transformers initializes to 0 or 1 (the residual queries, the clip scales, the learned taps, the
selection bias). `--packed` stores the experts as the releases do (NVFP4 in the first six
layers, MXFP8 in the others) and gives the reference the dequantized values. `--tokenizer` takes
the release's vocabulary (201024 rows) and `--chat` a conversation rendered by the release's
template.

Against transformers (float32; logits of standard deviation 1.7 under the cap of 3):

| Checkpoint, prompt | Pass | Max logit diff | Mean KL | Same top token |
|---|---|---:|---:|---:|
| floats, 150 random tokens | prefill in one pass | 2.2e-4 | 4.6e-11 | 150/150 |
| | prefill in chunks of 7, and one token at a time | 2.5e-4 | 5.4e-11 | 150/150 |
| | 16 greedy steps | 9.6e-5 | 3.8e-11 | 16/16 |
| floats, 700 random tokens | prefill in one pass | 2.4e-4 | 4.8e-11 | 700/700 |
| | prefill in chunks of 7, and one token at a time | 2.7e-4 | 4.6e-11 | 700/700 |
| | 24 greedy steps | 9.0e-5 | 2.6e-11 | 24/24 |
| blocks of 4, kernel 3, window 40, extent 9; 300 tokens | prefill in one pass | 9.4e-4 | 4.6e-10 | 300/300 |
| | prefill in chunks of 7, and one token at a time | 7.1e-4 | 3.2e-10 | 300/300 |
| | 24 greedy steps | 2.7e-4 | 2.5e-10 | 24/24 |
| `--packed` (NVFP4 and MXFP8 experts), 200 tokens | prefill in one pass, in chunks of 7, one token at a time, streamed | 1.3e-4 | 3.5e-11 | 200/200 |
| | 16 greedy steps | 1.3e-4 | 5.7e-11 | 16/16 |
| chat: system prompt, thinking (28 tokens) | prefill in one pass | 1.0e-4 | 2.5e-11 | 28/28 |
| | 16 greedy steps | 1.2e-4 | 2.3e-11 | 16/16 |
| chat: no thinking (14 tokens) | prefill in one pass | 1.7e-4 | 3.5e-11 | 14/14 |
| | 16 greedy steps | 1.3e-4 | 2.0e-11 | 16/16 |
| chat: tools, a reply with two calls, results, a further turn (339 tokens) | prefill in one pass | 3.9e-4 | 4.8e-11 | 339/339 |
| | prefill in chunks of 7, and one token at a time | 3.5e-4 | 4.9e-11 | 339/339 |
| | 16 greedy steps | 3.0e-4 | 1.9e-10 | 16/16 |
| an image of 12 tokens and a text (35 tokens), oxinfer's vision tower | prefill in one pass | 2.6e-4 | 1.2e-10 | 35/35 |
| | 16 greedy steps | 2.8e-4 | 8.6e-11 | 16/16 |

"Greedy steps" feeds the reference's greedy tokens one at a time after the prompt and compares
the logits after each. Decoding greedily on oxinfer's own logits gives the reference's tokens in
every row (16 of 16, or 24 of 24).

```sh
python tools/inkling2_tiny.py tiny [--packed] [--tokenizer REPO --chat CONVERSATION.json | --image FILE.png]
cargo run --release --example inkling2_check -- tiny/model tiny/ref.safetensors [--chunk N] [--image FILE.png] [--stream]
```

The image row runs the prompt the processor builds (12 image tokens for a picture of 130 x 95,
which needs padding on both sides): the tiles are the processor's to 6e-8, the tower's features
transformers' to 3e-8 (for values up to 0.06), and the CLI's prompt, token ids and 16 greedy
tokens are the reference's.

The CLI on the three chat checkpoints gives the template's text (`--print-prompt`), the
`tokenizers` library's ids, and the reference's 16 greedy tokens.

**The real vision tower.** `model.visual.*` is 0.66 GB at the end of the last shard of
`Inkling-2-Small` and was fetched by range request into a checkpoint of its own.
`tools/ref_inkling2_vision.py` runs transformers' `Inkling2VisionModel` and image processor on
it (float32):

| Picture | Tokens | Max feature diff | RMS diff | RMS of the features | Largest feature |
|---|---:|---:|---:|---:|---:|
| 333 x 250 (9 x 7 tiles, padded on both sides) | 63 | 1.0e-5 | 5.7e-7 | 0.151 | 6.44 |

```sh
python tools/ref_inkling2_vision.py DIR picture.png ref.safetensors
cargo run --release --example inkling2_check -- DIR ref.safetensors --image picture.png --vision-only
```

**The releases' headers.** The names and shapes the loaders read, written out from the config,
are exactly the tensors of the tiny checkpoints (338, or 386 packed), of `Inkling-2-Small`
(1770 tensors in 24 shards: 40 layers of NVFP4 experts and 8 of MXFP8) and of
`Inkling-2-Small-MXFP8` (1610 in 48 shards), with none missing and none unread, `mtp.safetensors`
aside. The two configs differ only by `quantization_config`.

What this does not show: that the released weights give sensible text here. A misreading of the
release that transformers shares would not show either, since the reference is transformers'
model and not the SGLang implementation the weights were trained with.
