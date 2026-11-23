# Inkling-2 Small

`thinkingmachines/Inkling-2-Small` and `thinkingmachines/Inkling-2-Small-MXFP8` (`inkling2`,
`Inkling2ForConditionalGeneration`; 594B parameters of which 17B per token, 48 layers of 4608,
384 experts of which 8 per token, a 1M-token context, text and images in, text out; 393.5 GB with
NVFP4 experts, 622.8 GB with MXFP8 experts) run on `src/inkling.rs` (the decoder) and
`src/inkling_vision.rs` (the vision tower).

**Small's own weights were never run.** Neither release fits the 100 GB this work could
download, and neither does any GGUF conversion (the smallest, UD-IQ1_S of
`unsloth/Inkling-2-Small-GGUF`, is 172 GB). The decoder was measured on tiny random checkpoints
against transformers, the loader on the headers of both releases, the reading of the NVFP4
experts on a few real ones, the tokenizer and the chat template on the released files. The
vision tower (0.66 GB) was run on the real weights. Since then [Inkling-2 Mini](#inkling-2-mini),
the same decoder with a loop around its middle layers, its experts in the same NVFP4 and MXFP8
forms and the same tokenizer, ran on its real weights and matches transformers there (max logit
diff 1.4e-4): no fault of the shared code showed. GGUF files are now read (llama.cpp's
`inkling2` and `inkling2_loop`, unsloth's layouts), though no Small one was tried.

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

The **MTP head** (`mtp.safetensors`) runs for Mini as a DSpark draft, see
[Speculative decoding](#speculative-decoding) (Small's head, 3.4 GB, was not tried). The config's
`temporal_patch_size` (video) is unused by transformers too. The reference processor composites
transparent images onto a checkerboard; oxinfer decodes images to RGB first.

The convolutions keep their recent inputs in rings indexed by position and the local layers
keep 64 positions beyond their window, so a session can go back up to 64 positions
(`inkling::ROLLBACK`, what speculative decoding needs). The cache of a local layer holds at most
2112 positions whatever the context length.

## Checkpoints

The releases' tensor names (`model.llm.*`, `model.visual.*`), which is also what transformers'
`save_pretrained` writes. The routed experts are two tensors per layer, `w13_weight`
(`[384, 4608, 4608]`: gate and up rows interleaved, gate first) and `w2_weight`
(`[384, 4608, 2304]`), in one of three forms:

- **MXFP8** (all 48 layers of the `-MXFP8` release, layers 40-47 of the other): E4M3 bytes with
  an E8M0 exponent byte per 32 values along a row (`*_scale`). Read in place, exponents
  included: the CPU and Metal kernels decode `2^(e - 127)` from the bytes (18 GB for the whole
  model). Under CUDA they are expanded to float32 at load (73 GB), which its kernels read.
- **NVFP4** (layers 0-39 of `Inkling-2-Small`): two E2M1 codes per byte (low nibble first), an
  E4M3 scale per 16 values, and per expert `amax / outer_scale_denominator` (the denominator is
  2688 = 6 x 448). They are repacked at load into ggml's NVFP4 blocks, which takes the experts'
  size in memory (290 GB): this release needs a machine that holds it. On a smaller one the
  MXFP8 release is the one to use (its experts are streamed, see [streaming](streaming.md)).
- bf16 or float32 (what `save_pretrained` writes).

A decoded token reads 23.5 GB of the MXFP8 release (9.4 GB of attention weights, 1.9 GB of
unembedding, 12.2 GB of experts) and 19.0 GB of the NVFP4 one.

`unsloth/Inkling-2-Small-GGUF` (172 GB for UD-IQ1_S to 484 GB) was not tried. GGUF files of the
family are read since Mini (llama.cpp's `inkling2` / `inkling2_loop` names and unsloth's own
layout, its `.scale` tensors being per-expert output factors; see
[Inkling-2 Mini](#inkling-2-mini)); the header seen earlier for Small (`attn_res_score`,
`shortconv_k`, ...) is of an older converter that the current names replaced, and no Small
quantization fits the download budget of this work.

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

# Inkling-2 Mini

`thinkingmachines/Inkling-2-Mini` (`inkling2_loop`, `Inkling2LoopForConditionalGeneration`;
26.4B parameters of which 3.4B per layer application, 24 layers of 2304, 128 experts of which 8
per token, text and images in, text out; 19.5 GB with NVFP4 / MXFP8 experts) runs on the same
code as Small (`src/inkling.rs`, `src/inkling_vision.rs`), from the release, from llama.cpp's
GGUF files (`ggml-org/Inkling-2-Mini-GGUF`) and from unsloth's (`unsloth/Inkling-2-Mini-GGUF`).

```sh
oxinfer --model thinkingmachines/Inkling-2-Mini --prompt "Why is the sky blue?"
oxinfer --model thinkingmachines/Inkling-2-Mini --image photo.png --prompt "Describe this image."
oxinfer --model ggml-org/Inkling-2-Mini-GGUF:Q8_0 --image photo.png --prompt "Describe this image."
oxinfer --model unsloth/Inkling-2-Mini-GGUF:UD-IQ3_XXS --kv-cache q8 --prompt "..."
```

A GGUF model takes its images from the `mmproj*.gguf` beside it (`mmproj` option). The model
always thinks a little, even at the lowest effort (its card says so); `--think` asks for the
template's default effort (0.9).

## The loop

Layers 6 to 17 (two blocks of the attention residual) run three times, with the same weights:
48 layer applications per token. Each pass after the first starts again from the stream and the
block states that entered layer 6, and each of its layers mixes *messages* from the previous
pass into its inputs:

- the messages of a layer are four reads of the previous pass, made like the attention
  residual's mixes: a learned query (one each for the query, key, value and MLP inputs) against
  the RMS-normed block states the pass made after its first one, and its RMS-normed output;
- a message gate mixes a message `m` into an input `x`: `g = down [rms(x); rms(m)]` (rank 64),
  then `(1 - tanh(forget g)) x + tanh(gain g) m`; on the query, key and value inputs (each from
  the layer's mixes, so a sliding layer's three inputs then differ) before their norms, and on
  the MLP's input before its norm.

Every application has its own attention cache and convolution history: a looped layer keeps
three, one per pass (as in transformers and llama.cpp). The release stores the queries as two
tables (`middle_cycle_cross_pass_queries`: the query and MLP queries of each looped layer,
`..._kv_queries`: the key and value ones) and the gates as `controllers.{attn,k,v,mlp}`.

## Checkpoints

- **The release**: the experts of layers 0-19 in NVFP4 (repacked at load into ggml's NVFP4
  blocks, 4.5 bits a weight; the file's own bytes of them are then given back to the system), of
  layers 20-23 in MXFP8 (read in place), the rest in bf16.
  `mtp.safetensors` (the MTP head) is the drafter of `--mtp`
  ([Speculative decoding](#speculative-decoding)).
- **llama.cpp's `inkling2_loop`** (ggml-org's Q8_0, IQ4_NL, BF16): the tensors renamed back to
  the release's names (`gguf_name`), the relative-bias tables transposed, the experts' gate and up
  rows not interleaved, four message queries per looped layer.
- **unsloth's own layout** (the files at the root of `unsloth/Inkling-2-Mini-GGUF`, architecture
  `inkling2` with `num_loops`): its block count and per-layer arrays cover the 48 applications,
  the message queries are the release's two tables (`loop_res_query`, `loop_kv_res_query`), the
  gates `loop_{attn,k,v,mlp}_{down,forget,gain}`, and the experts of the NVFP4 layers keep their
  per-expert factor beside them (`ffn_*_exps.scale`), applied to the experts' outputs. Its
  `Mainline_Support/` files are the same tensors under llama.cpp's names (read as above; unsloth
  warns that mainline llama.cpp gets them wrong until its PR 32704).
- **mmproj** (`inkling2v`, ggml-org's and unsloth's): the tower under the release's names, its
  patch embedding (a convolution there) laid out again as the release's linear layer.
- **AtomicChat's GGUF files** (`AtomicChat/Inkling-2-Mini-GGUF`, AD-* and plain quants, and
  `AtomicChat/Inkling-2-Mini-Uncensored-GGUF`): llama.cpp's `inkling2_loop` layout, read as
  ggml-org's. Their low-bit experts mix types per layer: AD-IQ2_M has IQ2_XXS gate/up experts
  and IQ4_NL or Q2_0 (ggml type 42, 2-bit codes `-1..2` with an f16 scale per 64) down experts,
  AD-Q4_K Q4_K gate/up and Q5_0 down. Q2_0 has its own SDOT decode kernel and int8 prefill rows
  (it went through the portable per-block decoding before: AD-IQ2_M decoded at 4.8-8.8 tok/s on
  the CPU, 17-21 after, as fast as Q8_0 measured alongside, at a load of 21-35).
- **mlx-lm's MLX checkpoints** (`mlx-community/Inkling-2-Mini-4bit`, mlx-lm's `inkling2`,
  17.3 GB): mlx-lm's names (`model.language_model.*`, `lm_head`, `model.vision_tower.*`) mapped
  back to the release's (`mlx_name`); the attention, embedding and LM head matrices in affine
  4-bit groups of 64 (read as oxinfer's Q4 with MLX offsets), the experts the release's bytes as
  32-bit words (NVFP4 with `global_scale = amax * 6 * 448 / denominator`, MXFP8), read as the
  release's; the message queries of the looped layers in one table. Against mlx-lm itself
  (PR 2086, on the GPU) on four raw-text prompts: mean KL of the next token 2.6e-5 to 1.2e-3,
  the same top 5 (CPU, int8 activations); against the release, as UD-IQ3_XXS is (KL 5e-4 to
  5e-2, same top token). The vision tower (bf16) is read too. oMLX's `oQ4e` files are another
  layout (fused projections) and are not read.

NVFP4 matrices (the release's experts, unsloth's NVFP4 layers) run on the int8 kernels: E2M1
codes through MXFP4's doubled table and the per-16 E4M3 scales halved, so a 16-value group is
one SDOT; the tensor-wide factor (or unsloth's per-expert one) multiplies the sums. Prefill
decodes them for the f32 GEMM.

## Verification

**Real weights against transformers.** `tools/ref_inkling2_mini.py` runs transformers'
`Inkling2LoopForConditionalGeneration` (main, 5.24.0.dev0) in float32 on the release, its experts
kept exactly as stored: transformers dequantizes them on load into the model's dtype; the script
keeps them in bfloat16 without their per-expert factor (E2M1 code times E4M3 scale, and E4M3
value times a power of two, are exact in bfloat16) and applies the factor, read from the file,
to the upcast weights. On a 24-token chat prompt (thinking effort 0):

| Pass | Max logit diff | Mean KL | Same top token |
|---|---:|---:|---:|
| prefill in one pass | 1.4e-4 | 3.4e-11 | 24/24 |
| 16 greedy steps, one-token kernels of the time (float NVFP4 path) | 7.8e-5 | 1.2e-12 | 16/16 |
| 16 greedy steps, NVFP4 on the int8 kernels (activations rounded to int8) | 0.74 | 2.8e-4 | 16/16 |

and the 16 greedy tokens of each are the reference's. Each of the 48 layer applications'
outputs is within 2e-6 (relative L2) of transformers' hidden states. The vision tower on a
320 x 228 picture (48 tokens): features within 5.3e-6 (RMS 0.137) of transformers', from the
release and from ggml-org's BF16 mmproj alike.

**A fault in transformers' loading** (not in its model): the loader hands its NVFP4 conversion
`amax` already cast to the model's dtype, so in bfloat16 every NVFP4 expert is off by up to 0.4%
(of its factor). Run that way, the reference differs from the stored model by up to 1.6 in the
logits (KL 5.6e-3) on the prompt above; that is how this was found.

**GGUF files** against the same reference (the requantized experts are the difference):

| File | Prefill max diff / KL | Decode KL | Greedy (16) | Tokenizer |
|---|---|---:|---:|---|
| ggml-org Q8_0 | 2.4 / 1.0e-2 | 1.1e-3 | 16/16 | the reference's ids |
| unsloth UD-IQ3_XXS | 18 / 0.16 | 1.4e-2 | 16/16 | the reference's ids |

The GGUF vocabulary (`tokenizer.ggml.pre` = `inkling2`: the GPT-4o split, merges ignored for
whole words) gives the `tokenizers` library's ids.

**Tiny random checkpoints** (`tools/inkling2_tiny.py --loop`: blocks of 3, layers 3-8 looped
three times, a full layer in each looped block, rank-8 gates). A looped tiny model is far more
sensitive to rounding than the plain ones: transformers' own float32 differs from its float64 by
up to 5 in the logits with the default settings, through near-tied expert choices made 36 times.
With `QSCALE=1` (attention-residual queries of 1 rather than 2), `SPREAD` (spaced selection
biases) and `F64=1` (the reference in float64), transformers' float32 is within 9.5e-3 of its
float64 and oxinfer within 1.2e-2 (prefill in chunks, KL 1.8e-7) to 2.9e-2 (one pass, KL
3.9e-7), with the same top token at all 150 positions and the reference's 16 greedy tokens; the
per-application hidden states drift away from the float64 ones smoothly (no step jumps), which a
misread loop would not do. The plain tiny checkpoints still match as before (1.9e-4).

## Long contexts and memory

The sliding-window layers keep at most twice their window and 64 positions more on the CPU
(2112 positions), a ring of `window + 64` (1088) on the GPU; the full-attention
layers keep everything, and past `log_scaling_n_floor` (8192 here, 128000 for Small) their
queries and relative vectors grow with `1 + 0.1 ln(pos / 8192)`. Of the 48 applications, 8 are
full attention (layers 5 and 23, and layers 11 and 17 once per pass), so a token of context
costs 8 x 2 x 1024 values:

| `--kv-cache` (`kv_cache` key) | Per token of context | Sliding windows (fixed) | 128k context |
|---|---:|---:|---:|
| `f32` (default, exact) | 64 KB | 692 MB | 8.6 GB |
| `f16` | 32 KB | 346 MB | 4.6 GB |
| `q8` (int8, a float32 scale per 32) | 18 KB | 195 MB | 2.6 GB |

Measured with `examples/inkling2_ctx_bench` (the caches filled with placeholder keys and values,
so that decoding at a context can be timed without running the prompt; peak memory from
`/usr/bin/time -l`), UD-IQ3_XXS (11.7 GB), CPU, on a busy 18-core M5 Max:

| Context | KV | Caches | Peak memory | Decode |
|---:|---|---:|---:|---:|
| 1k | q8 | 0.11 GB | 12.0 GB | 19.9 tok/s |
| 8k | q8 | 0.25 GB | 12.2 GB | 13.2 tok/s |
| 32k | q8 | 0.70 GB | 12.7 GB | 5.7 tok/s |
| 32k | f16 | 1.24 GB | 13.4 GB | 8.7 tok/s |
| 128k | q8 | 2.51 GB | 15.0 GB | 3.7 tok/s |
| 128k | f16 | 4.47 GB | 17.2 GB | 2.8 tok/s |
| 512k | q8 | 9.76 GB | 23.2 GB | 0.9 tok/s |

A real prompt of 9816 tokens with a passphrase hidden in its first seventh (`--kv-cache q8`,
UD-IQ3_XXS) was answered right; prefill 84 tok/s, decode 13 tok/s, peak memory 15.3 GB with
prefill chunks of 2048 (their activations are most of the difference: oxinfer uses chunks of 512
below 32 GB of memory, `--prefill-chunk` / `prefill_chunk` sets them).

**The quantized experts in memory.** The experts are never dequantized: NVFP4 runs from ggml's
NVFP4 blocks (the same 4.5 bits a weight; the CPU's int8 kernels and the GPU's kernels read them),
MXFP8 from the file's E4M3 bytes in place. The NVFP4 repacking held
the file's own codes and scales beside the blocks for the model's life (the file is read into
memory unless `--mmap`): the release on Metal had a footprint of 31 GB for an 18.6 GB file. The
loader now gives those pages back once the blocks are made (`ModelFiles::discard`, reading them
again is an error): 20 GB while decoding (`footprint`, 1.2 GB of it GPU memory: caches and
activations); mlx-community 4bit (16 GB) 19 GB, its dense matrices being converted too. And
MXFP8's E8M0 exponents are expanded to float32 for the kernels (one per 32 values, so an eighth
of the values' size rather than a thirty-second): 0.5 GB instead of 0.13 for Mini's four MXFP8
layers, 73 GB instead of 18 for `Inkling-2-Small-MXFP8`, whose values stream from storage
either way. Small's NVFP4 release repacks its 40 NVFP4 layers as Mini's (290 GB; the file's
bytes given back the same way when it is read into memory). transformers' quantized path (PR
51062) holds one copy of each too (9.9 GiB allocated on MPS at load, 21 GiB in the driver, a peak
footprint of 31 GB in the run of the speed table).

What fits, from these sizes: weights + 1.3 GB (runtime and prefill chunks of 512, measured on
the 9816-token prompt: 13.3 GB peak) + the windows + the context, in the memory a process can
use, taken here as RAM - 4 GB (macOS and the rest). With `q8`:

| Memory | UD-Q2_K_XL (9.1 GB) | UD-IQ3_XXS (11.7 GB) | UD-IQ4_XS (15.7 GB) | UD-Q4_K_XL (17.6 GB) | Q8_0 (27.8 GB) |
|---|---|---|---|---|---|
| 16 GB | about 75k | no (1.2 GB short; see --mmap below) | no | no | no |
| 24 GB | about 500k | about 370k | about 150k | about 50k | no |
| 32 GB | about 940k | about 800k | about 580k | about 480k | no |
| 64 GB | 1M (the model's limit) | 1M | 1M | 1M | 1M |

A 16 GB Mac can still run UD-IQ3_XXS memory-mapped (`--mmap`): the weights are then page cache
that macOS may drop and read again, slower but working.

## On the GPU

With the Metal backend (the default on Apple Silicon), every pass runs on the GPU
(`src/metal_inkling.rs`, `metal_inkling.metal`, `metal_inkling_pf.metal`), the whole model in one
submission per pass, with the activations, the attention caches and the convolution rings
resident on the device:

- **Passes of up to 16 tokens** (decoding, speculative verification): the attention residual's
  mixes (one threadgroup per query and token), the message gates of the loop's later passes,
  the projections on the decode library's matvec kernels (its exact multi-token kernels for
  several tokens), the key / value convolutions with their L2 clip and norm written straight to
  the caches, the scaled and clipped queries and relative vectors (with the log scaling of the
  full layers), attention in splits of keys with the relative bias of each distance, the sliding
  window, the sink and the output gate, the post-attention convolution fused with the MLP's mix
  and the sigmoid routing, the routed experts grouped by expert (tokens of a pass that choose the
  same expert read it once) in their stored format (NVFP4, MXFP8, Q2_0, every GGUF type of the
  decode kernels, the per-expert factors of unsloth's files), the post-MLP convolution summing the
  experts' slots, the final mix, the LM head and the logit cap. Every token of a pass is computed
  as it would be alone (the token is a grid dimension of every kernel; the splits of a token's
  keys depend on its own range), so a verification pass gives exactly the logits of one-token
  steps and speculative decoding stays exact.
- **Prefill passes** of up to 512 tokens (longer prompts in pieces): the same, with the dense
  projections on the matrix-unit GEMMs (half operands, f32 accumulation; float32 loop gates on
  the f32 GEMM) and the routed experts as grouped matrix-unit GEMMs over each expert's tokens
  (the routing read back at each MoE layer to build the tiles), SiTU and the routing-weighted sum
  on the device. A pass longer than a sliding window's spare rows keeps its keys and values in the
  arena and moves its last rows to the window's ring after; the convolution rings are written
  once the convolutions have read them.
- **KV caches** in f32, f16 or q8 rows (`--kv-cache`), the sliding windows as rings of
  `window + 64` rows by position, the full-attention caches growing; a session's state moves to
  the device on its first GPU pass and back to the host when a pass runs on the CPU (cloning a
  session reads it back), and `truncate` works on either side.
- **LoRA adapters** (`--lora`) run on the device too: the dense updates as a down / up pair of
  kernels beside each matmul, the experts' down-projection updates per expert from an address
  table whose strengths are written at each pass (switching an adapter needs no rebuild); the
  embedding's update stays with the host's lookup. The experts' gate / up updates too: `A x` per
  slot first, then the pair kernels add `s Bt u` to the gate and up rows before SiTU (prefill
  passes then take the per-slot pair kernels rather than the grouped GEMMs: 935 against 1125
  tok/s on a 600-token Q8_0 prompt, decode unchanged). Checked with a synthetic PEFT adapter
  (rank 2, gate / up updates of all 128 experts on layers 3 and 10, down updates on layer 3,
  moving the logits by 1.0-1.3) on ggml-org Q8_0 and the release: prefill against the CPU with
  the adapter KL 3.8e-10 / 6.6e-10, every decode step the CPU's top token. An update's `A x` runs
  in its matvec's own stage (both read only x), so an adapter adds one stage per updated matrix
  (before: three). OpenUncensored's PEFT adapter (rank 2, the 20 attention output projections of
  layers 4-23, so 44 of the 48 applications) costs 1-2% of decoding (release, greedy, best of 3:
  53.1 tok/s with it, 54.1 without, 54.7 merged into the release; load 12-61), AtomicChat's GGUF
  adapter on AD-IQ2_M (3097 updates of rank 1, every projection and expert) 10% (62.7 against
  69.6, its merged file 71.6), with the merged model's greedy text in both. (A report of half
  speed with the PEFT adapter did not reproduce here: 51-57 tok/s before this change too.)

The Metal engine and the CUDA one (`src/inkling_cuda.rs`) are two implementations of the same
device interface in `src/inkling.rs` (`DevEngine` for the passes, `DevSession` for a session's
state on the device): the host side (embedding, interventions and captured layers on the CPU,
LoRA strengths, moving a session between host and device, going back for speculative decoding)
is shared. A device says how far its sessions go back (`Inkling2Model::rollback_limit`): 64
positions on Metal, CUDA and the CPU.

On CUDA the convolutions' histories are rings of the recent inputs by position (`kernel - 1 + 64`
rows, the host's `ConvRing` layout: a session moves between host and device by copying them), the
sliding layers' caches rings of `window + 1024` rows and the full layers' caches rows by position,
so going back is setting the position: the rows from there on are written again by the next
passes. Passes of up to 16 tokens run on the decode kernels with each token's attention split as
in a pass of its own, so a 16-token pass gives the logits of 16 one-token steps bit for bit
(`DevEngine::exact_tokens`; longer passes take the prefill GEMMs, which round differently), and
such passes are captured as CUDA graphs per token count. The engine returns the streams draft
models capture (`DevWant::capture`), so prompt lookup, `--mtp` and DSpark drafts all run with
the target on the GPU (the drafter itself on the CPU).

Checked on an RTX PRO 6000: `tests/inkling_cuda_rollback.rs` (`cargo test --release --test
inkling_cuda_rollback -- --ignored --nocapture --test-threads 1`, ggml-org Q8_0) passes (going back
1-64 positions against an uninterrupted run; greedy with prompt lookup and with a draft against
plain greedy); `inkling2_cuda_check --rollback 16`: going back 16 positions and running them again,
one at a time or as one pass, gives the first run's logits exactly (f32, f16 and q8 caches, past
the sliding window); `small_ops_match_cpu` checks the rings against the host's. Greedy on a code
prompt (Q8_0, 256 tokens), the same text each way: plain 120.7 tok/s, prompt lookup 128.8
(1.15 tokens per forward), unsloth's DSpark draft 103-105 (its CPU drafting costs about as much as
the tokens it saves).

### The CUDA engine

With an NVIDIA GPU (`--device auto` or `cuda`), `src/inkling_cuda.rs` runs every pass on the
GPU with its own kernels (`src/inkling_cuda.cu`, compiled by NVRTC at the first pass for the
weight formats the model has, a few seconds; kept in `~/.cache/oxinfer/kernels`,
`OXINFER_KERNEL_CACHE=DIR` or `0`). The host looks up the embeddings and samples.

- **Passes of up to 16 tokens** (decoding, verification) on decode kernels: matvecs (the
  projections of one input in one launch), the chosen experts' rows read in place, the mixes and
  convolutions per token, attention split over the keys and merged with the sink; each such pass
  is a CUDA graph (one per token count and captured steps), so a decoded token is one launch.
- **Prefill** in chunks of up to 1024 tokens: tensor-core GEMMs, the routed experts as segments
  per expert built on the device and grouped GEMMs (SiTU in the epilogue), attention in tiles of
  8 queries sharing their keys and values in shared memory, convolutions a thread per (channel,
  token), the message gates' low-rank projection a block per token.
- **Weights as stored**: every GGUF block format but the ternary and 1-bit ones (the 2-bit Q2_0
  of AtomicChat's files included), the release's
  NVFP4 (with its per-expert factors) and MXFP8 experts, unsloth's per-expert output factors,
  oxinfer's Q8 / Q4, bf16, f16, f32. Caches in f32, f16 or q8 (`--kv-cache`); the full layers'
  grow by a quarter, the sliding ones are rings of `window + 1024` rows.
- **Precision**: activations in f32 throughout. The prefill GEMMs take the activations as pairs
  of halves (the low half scaled by 2^11, out of f16's subnormals) and the weights as their exact
  values (every format but f32 / bf16 stores small integers or grid values times a scale per 16
  or 32), two tensor-core products per 16-wide step whose sums are scaled, minima added, with
  f32 adds: 5e-8 of the summed magnitudes at worst, as an f32 GEMM (the matvecs: 2e-8). Plain
  f16 operands (`gpu_exact=0`, ~10% faster prefill) move the logits by KL ~2e-3: drifts of even
  3e-6 tip some of this model's near-tied expert choices.
- A session the device cannot hold moves to the host and continues on the CPU; LoRA adapters,
  Hadamard-folded or streamed weights, the ternary / 1-bit GGUF formats and MLX affine Q4 keep
  the CPU path (`OXINFER_GPU_PLAN_DEBUG=1` says why), and so does a model whose weights, 0.5 GB
  of activations and 1 GB to spare do not fit the device's free memory (Q8_0 on a 16 GB card:
  [below](#threadripper-9970x-and-rtx-5080-16-gb)); `OXINFER_INKLING_PROFILE=1` times each
  kernel (CUDA events) and prints the table at exit.

**Against the CPU path** (`examples/inkling2_cuda_check`: the same pseudo-random prompt on the
CPU, saved, then on the GPU; `--all` every position, `--tail N` the last N prompt tokens as decode
steps, compared with the CPU's f32 prefill since its decode kernels round activations to int8):

| Checkpoint, prompt | Prefill (every position) | Decode steps (32) | Greedy (16) |
|---|---|---|---|
| release (NVFP4 / MXFP8 experts), 300 tokens | KL 2.5e-10, max diff 3e-4 | KL 9.6e-11 | 17/17 |
| release, 8100 tokens (last 64 positions) | KL 7.6e-9 | KL 7.1e-9 | |
| ggml-org Q8_0, 1400 tokens (GPU chunks of 512) | KL 4.3e-11 (per-query attention) | KL 7.3e-9 | 17/17 |
| unsloth UD-IQ3_XXS, 300 tokens | KL 4.3e-11 | KL 4.2e-11 | 17/17 |
| AtomicChat AD-Q4_K, 300 tokens | KL 4.3e-11 | KL 4.1e-11 | 17/17 |

The GPU agrees with itself to KL 3e-12 (decode steps against its prefill) and its prefill is
deterministic. Some prompts have a position where two experts tie to within f32 rounding:
any two f32 implementations then choose differently there now and then. Of six 600-token Q8_0 prompts, three match at every position (KL
4e-11), three have one such position (KL up to 1e-2 there, fading over the next tokens), with the
tiled attention as with the per-query one (on different prompts); at 8400 tokens the release's
last 64 positions are at 2e-8 with three such spikes. `cargo test --lib cuda::` checks every
kernel against an f64 definition (every weight format through the matvec and both GEMMs, the
experts on both paths, attention in the three cache formats with windows, rings and splits, the
routing with ties and NaNs, the convolutions, mixes, clips and gates against the CPU's functions).

`OXINFER_GPU_DECODE=0` (or `--gpu-decode off`) keeps the passes on the CPU,
`OXINFER_INKLING_PREFILL=0` the prefill passes; `OXINFER_GPU_DECODE_CHECK=1` runs each decode
step on the CPU too and prints the difference; `OXINFER_INKLING_SKIP=dense,attn,exp,...` leaves
kinds of ops out (measurements only: the routing then sees stale inputs, so its experts' reads
are not those of a real pass); `OXINFER_INKLING_PROFILE=N` runs the passes of up to 16 tokens
stage by stage, a command buffer each committed back to back, and prints the GPU time per kind
of stage and the dispatch and barrier counts every N passes (profiling only);
`OXINFER_INKLING_NARROW=1` puts the router's logits back on `ink_mvf` (measurements). The kernels compile on first use (about 1.5 s, cached by
the system after).

**Correctness.** Against transformers float32 on the release (the 24-token chat prompt and 16
greedy steps of [Verification](#verification-1)):

| Pass | Max logit diff | Mean KL | Same top token | Greedy (16) |
|---|---:|---:|---:|---:|
| prompt in passes of 7 on the GPU (decode kernels) | 6.7e-4 | 2.2e-10 | 24/24 | |
| 16 greedy steps on the GPU | 1.1e-4 | 3.3e-12 | 16/16 | 16/16 |
| the same after rolled-back junk passes | 1.2e-4 | 3.8e-12 | 16/16 | |
| prompt as one prefill pass (matrix-unit GEMMs) | 0.69 | 1.0e-4 | 24/24 | 16/16 |

ggml-org Q8_0 (against the same reference): decode KL 7.1e-4 (the CPU path: 1.1e-3, whose int8
activations round more), 4.0e-4 with q8 caches; unsloth UD-IQ3_XXS with f16 caches 2.8e-3 (CPU
9.2e-3); greedy 16/16 everywhere. GPU prefill against the CPU path on the last logits: Q8_0 after
117, 207, 903 and 2447 tokens KL 2e-9, 4e-10, 4e-11, 2e-10 (same top token);
AtomicChat AD-IQ2_M with the Uncensored LoRA after 600 tokens KL 1.1e-8 (the adapter itself moves
the logits by up to 4.2), and its decode steps the CPU's top token at every step (KL 1e-9 to
3e-3).

## Speed

18-core M5 Max (128 GB, Apple10 GPU with matrix units), shared with other jobs (load averages
2-4 for the table below, no other GPU job; at load 6-12 with another session's image jobs on the
GPU the same runs were 10-25% slower). oxinfer: `examples/inkling2_bench` on the engine API (as
`llama-bench`: a prompt of N random tokens into a fresh session, N tokens generated after a
1-token prompt, after a warm-up pass of each, the median of 3 runs; f32 KV caches); llama.cpp:
`llama-bench -p 512,2048 -n 128 -r 3 -fa 1` (master of 2026-11-21, Metal, its default f16
caches), interleaved model by model. tok/s:

| Checkpoint | engine | pp512 | pp2048 | tg128 |
|---|---|---:|---:|---:|
| ggml-org Q8_0 (27.8 GB) | llama.cpp, Metal | 1002 | 860 | 38.3 |
| | **oxinfer, Metal** | **1603** | **1408** | **58.7** |
| | oxinfer, CPU (a 2456-token prompt, measured before) | 116 | | 17.1 |
| AtomicChat AD-Q4_K (16.5 GB) | llama.cpp, Metal | 1010 | 793 | 40.6 |
| | **oxinfer, Metal** | **1751** | **1418** | **62.4** |
| release (NVFP4 / MXFP8 experts, bf16 dense; 19.5 GB) | oxinfer, Metal | 1856 | 1359 | 55.6 |
| unsloth UD-IQ3_XXS (11.7 GB) | oxinfer, Metal | 1751 | 1333 | 65.6 |
| mlx-community 4bit (17.3 GB) | **oxinfer, Metal** | **1808** | **1521** | **70.4** |
| | mlx-lm (PR 2086), MLX 0.32.4 | 1387-1490 | 1056-1440 | 29-35 |
| mlx-community oQ4e-mtp (16.9 GB) | oMLX 0.8.2, its MTP off / on | 1164-1452 | 912-1127 | 30-35 / 25-42 |
| release | transformers (PR 51062, NVFP4 / MXFP8 experts kept quantized on MPS), bf16 | 165-248 | | 15.9 (13.4 with `generate`) |

(llama.cpp's mainline reads unsloth's root files wrongly and has no NVFP4 GGUF path for the
release; the MLX rows are mlx-lm's and oMLX's own measurements on this machine, load 9-31, see
below. The transformers row is huggingface/transformers PR 51062 (branch `inkling2-metal-fp4`,
not merged at the time), whose `ArthurZ/metal-moe-decode` kernels run the experts from the
checkpoint's banks; a 24-token prompt and 128 greedy tokens in a token loop, best of 3, idle GPU,
load 7; oxinfer on the same release in the same conditions: pp512 1613, tg128 57-63.)

**transformers' quantized experts against oxinfer's.** Their kernels read the release's layout in
place: per row, the E2M1 codes (16 bytes per 32 values, one `uint4` per lane, so a SIMD-group reads
512 contiguous bytes) and the E4M3 scales as `ushort` pairs; oxinfer repacks the NVFP4 experts into
ggml's 36-byte blocks (which its CPU kernels read too) and gives the file's bytes back, so both
hold one copy. The numerics are the same: E2M1 codes and E4M3 scales decoded by moving their bits
into a half (the same trick, the same exact values), the scales applied to each 16-value partial
dot product (oxinfer's grouped kernels now do this too), the per-expert factor
`amax / outer_scale_denominator` in float32 from the file's float32 `amax` (their factors equal the
file's to the bit: the bfloat16 `amax` of transformers' dequantizing loader, below, is not on this
path), and MXFP8's E8M0 exponents as `2^(e - 127)` (theirs from the byte at run time, oxinfer's
expanded to float32 at load, see [Long contexts and memory](#long-contexts-and-memory)). Their
activations are bfloat16, so against the float32 reference: prefill KL 1.2e-2 (max logit diff 2.9),
decode KL 8.2e-4 (3.4), greedy 16/16; oxinfer's 2.1e-10 and 3.3e-12. Their gate / up kernel flattens
a pair's two rows over the lanes (4.5 chunks a lane) and runs each (token, route) pair on its own;
oxinfer's group a pass's tokens by expert. Nothing in their dequantization is cheaper than ours. Decode at a context of 8k / 32k (`examples/inkling2_ctx_bench`, the caches filled with
placeholder rows; llama.cpp `-d 8192,32768 -n 32`), Q8_0:

| | 8k | 32k |
|---|---:|---:|
| llama.cpp (f16 caches) | 34.9 | 34.2 |
| oxinfer, f32 caches | 54.5 | 46.3 |
| oxinfer, f16 caches | 56.1 | 51.0 |

**Where a step's time goes.** A decoded token reads 5.1 GB of Q8_0 (48 layer applications, 8
routed experts each, the LM head), 2.7-2.9 GB at 4 bits. Leaving kinds of ops out
(`OXINFER_INKLING_SKIP`; the best ms per token of 4 decode runs of 32 tokens, load 5-8) gives:

| | step | routed experts | dense matmuls | attention | LM head | loop gates | mixes | convolutions | routing | barriers alone |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q8_0 | 14.97 | 6.33 | 2.75 | 1.57 | 0.93 | 0.73 | 0.25 | 0.25 | 0.22 | 1.48 |
| mlx-community 4bit | 12.10 | 4.73 | 2.12 | 1.58 | 0.68 | 0.91 | 0.53 | 0.41 | 0.39 | 1.58 |

At Q8_0 the experts and the dense matmuls stream at about 500 GB/s, near what this GPU reads,
so a Q8_0 step cannot go much below 11 ms. The NVFP4 experts read at about 400 GB/s; with their
arithmetic left out (the same loads) they save only 0.7 ms a token (460 GB/s). A step has 663
dispatches in 751 dependent stages (`OXINFER_INKLING_PROFILE=N` counts them and times each kind
of stage in a command buffer of its own; independent ops, such as a layer's six to eight
projections, its mixes or the three attention gates, share a stage on the concurrent encoder),
and with every kernel left out the barriers alone take 1.5-1.6 ms.

**Lanes per row** (the grouped expert kernels): a row of 2304 values is 72 chunks of 32, and with
one SIMD-group walking it chunk `lane + 32 i` the last round of every row had 8 of 32 lanes busy (75%
of the lanes over the row; 36 chunks of a down row, 56%), and these kernels are bound by their
arithmetic, not their loads (an NVFP4 gate / up pass of 8 experts in isolation: 62.5 us, the same
loads without the arithmetic 44 us, a plain stream of the bytes 40 us). The lanes are now split
into row groups of the largest power of two dividing the chunk count (8 lanes a gate / up row, 4 a
down row), each group walking one row with every lane on the same number of chunks, the row's sum a
reduction over its group; NVFP4 also applies its scales to each sub-block's dot product and loads a
chunk's 16 code bytes together. The same pass: 47.5 us. Best of 4 alternated runs (`inkling2_bench
-n 32 -v 1,4`, idle GPU, load 9-21), before -> after:

| Checkpoint | tg32 tok/s | 1-token pass ms | 4-token pass ms |
|---|---:|---:|---:|
| mlx-community 4bit | 81.2 -> 82.5 | 12.15 -> 11.85 | 24.41 -> 21.71 |
| unsloth UD-IQ3_XXS | 69.1 -> **80.2** | 13.11 -> 12.32 | 28.97 -> 23.97 |
| release | 62.5 -> 62.8 | 15.94 -> 16.07 | 26.23 -> 25.52 |
| ggml-org Q8_0 | 65.7 -> 65.7 | 15.05 -> 15.15 | 24.78 -> 25.07 |

The 3-bit file (IQ3_XXS experts, whose grid decoding is the most arithmetic) gains most and now
decodes about as fast as the 4-bit one (decode KL against the release's float32 reference, prompt
in passes of 7: UD-IQ3_XXS 2.8e-3, mlx-community 4bit 2.4e-3). Verification passes gain more than
one-token steps (a group's tokens reread the rows from the cache, which is arithmetic again).

Measured and not kept: the same row groups for the dense matvecs (`ink_mvs`, every pass of 1 to 16
tokens on one kernel family): 2-4% slower for one token (a SIMD-group then reads four rows' 128-byte
pieces instead of one row's 512 contiguous bytes, and a 3072-row matrix has a quarter of the
SIMD-groups), even for 16; three SIMD-groups splitting a row group's chunks, no better; a lane
striping over four concatenated rows (all lanes busy, consecutive lanes on consecutive chunks), the
same as the current kernel in isolation (an MLX 4-bit 3072 x 2304 matvec: 17.2 us, 18.8 striped;
its loads alone 10.5 us: the dense matvecs are bound by the nibble decoding too). Reading the next
projections' weights ahead into the system cache (32-48 MB, read back at 950 GB/s against 570 from
memory) by a dispatch beside the latency-bound kernels of the stage before: 5% slower (the reads
outlast those stages and they wait for them). Repacking the NVFP4 experts into tiles that a
SIMD-group reads as 512-byte runs: the same loads cost (44 us either way), so no gain over the row
groups on ggml's blocks.

**Barriers.** A dispatch and barrier of an empty kernel costs 1.5 us (one threadgroup) to 2.5 us
(1152), and a single-threadgroup kernel that reads and writes 9 KB 3.6 us; a grid-wide barrier
inside a persistent kernel (an atomic counter in device memory, a spin with a timeout) 0.5 us with
40 threadgroups of 256 or 1024 (one per core), 0.9 us with 160 (all co-resident here: no
timeouts). The 751 stages of a step would then cost about 0.4 ms instead of 1.6: that is the most a
persistent kernel could take off the barriers, against a kernel that would hold every op of a layer
(at the register count of the largest) and fall back to the dispatched path whenever its threadgroups
cannot all be resident.

What moved decoding (MLX 4-bit 70 -> 83 tok/s, Q8_0 59 -> 67): the kernels of one threadgroup
per token (the attention residual's mixes, the norms) spread their dot products over all their
threads instead of one SIMD-group per dot product walking the 2304 values (+11%); the message
gates' norms written by the mixes that make their inputs, and their forget / gain matvecs (rows
of the rank, 64) with the gate mix in one kernel of 4 lanes per row (+3%); the expert kernels at
one output per SIMD-group and NVFP4 decoded with half-precision bit patterns (+3% each); the
attention's key and value loads batched (+1-1.5% at 8k / 32k). What did not: a serial encoder
without barriers (each dispatch after the last: 12% slower than the concurrent one with its 751
barriers), the experts' input row staged in threadgroup memory (4% slower), two to eight
outputs per SIMD-group or an output split over SIMD-groups (1-10% slower), NVFP4 dots in half
precision (+0.8%), the routing folded into the experts' first kernel (7% slower: each of its
1152 threadgroups ranks the 128 scores again) or done as one ranking pass (no change), and one
stage less for the gates' norms by itself (no change: 805 -> 757 barriers). The cost is in the
latency of kernels that run as one threadgroup per token, not in the number of barriers.

**Verification passes** (speculative decoding) do not cost one step: a pass of n tokens reads the
experts of every token, and n tokens choosing 8 of 128 experts each share few of them (4 tokens
read about 26-29 distinct experts). mlx-community 4bit, ms per pass:

| tokens | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pass | 14.0 | 18.9 | 23.1 | 27.3 | 37.2 | 46.0 | 65.3 | 85.4 |
| without the routed experts | 9.4 | 11.2 | 12.4 | 13.7 | 17.5 | 20.4 | 27.2 | 34.6 |

(Measured before the changes above; now 10.0 ms for one token and 20.1 for four.) A pass of 4
tokens adds 9.1 ms of experts to one of 1 token, 2.2 ms of dense matvecs, 0.6 of LM head, 0.3 of
gates and nothing of attention. The experts add about 3.1 ms a token (4.6 ms for the first); the
rest grows by 0.4-1.6 ms a token,
mostly the multi-token matvecs (dense, head and gates: 2.1, 0.5 and 1.3 ms for one token, 14.6,
5.4 and 7.6 for 16), which decode each weight chunk once for four tokens but load every token's
input chunk once per output row pair (more rows per SIMD-group spill registers: twice slower).
A round of 3 drafts accepted at 50% (2.5 tokens) therefore costs about 32 ms (27 ms of verification,
4-5 of drafting) against 35 ms of plain steps: drafting pays on code, about breaks even on prose
and chat (see below).

**The 125 tok/s and 1750 tok/s figures** reported for Inkling-2-Mini on a 48 GB M5 Max with EXO
could not be traced or reproduced: EXO's model cards (`exo-explore/exo`, main of 2026-11-06)
have no Inkling-2-Mini (only an older "Inkling-Small" card, whose MTP layers EXO drops), EXO has
no speculative decoding, and its pinned mlx-lm predates Inkling 2 (only the unmerged mlx-lm PR
2086 runs it). The only public trace of "125" is a skipped entry of a benchmark collection
citing a partner video without quant or protocol. On this machine mlx-lm decodes the
`mlx-community/Inkling-2-Mini-4bit` checkpoint at 29-35 tok/s and prefills at 1056-1490 tok/s;
oMLX (the only MLX engine with the release's drafter, on by default for `-mtp` checkpoints:
block 8, depth 7) reaches 37-42 tok/s on code (63-65% of drafts accepted) and 25-31 on prose
(45-51%). 125 tok/s of plain decoding is not plausible on this chip (the best published
single-token figures for the model are about 46 tok/s on an M5 Max); 1750 tok/s of prefill is
(mlx-swift-lm reports 1834-1852 on an M5 Max; oxinfer: 1603-1856 at pp512 at load 3, 1542-1701 at load 6-12).
Two rounds aimed at that figure (host and GPU overlap, kernel fusion, the attention at short ranges,
the dense matvecs' decoding, speculative decoding; then the small kernels of the latency chain) are
in [performance.md](performance.md#inkling-2-mini-decoding-on-metal-november-2026): one-token
passes went from 11.4-11.6 ms on the 4-bit files to 10.0-10.4 (tg64: mlx-community 4bit 85.5 ->
98.5 tok/s, UD-IQ3_XXS 82.1 -> 94.3, Q8_0 66.7 -> 73.6), mostly by an attention kernel with its
lanes across a head's channels and by keeping the small kernels' loads together and their
registers few; about 7 ms of a pass is now the weights' streaming and 3.5 the chain.

### CUDA

RTX PRO 6000 Blackwell Workstation Edition (96 GB, sm_120), Threadripper PRO 9995WX, nothing else
running. oxinfer: `examples/inkling2_cuda_check` (a prompt of N random tokens into a fresh session,
128 tokens decoded after it; the second of two runs; `--depth D` runs D tokens first, untimed);
llama.cpp: `llama-bench -p 512,2048 -n 128 -fa 1 -ngl 99` and `-p 2048 -n 128 -d 8192,32768` (master
of 2026-11-22, built from source with CUDA 13.4, its default f16 caches; mainline reads unsloth's
`Mainline_Support` copy of UD-IQ3_XXS, not the root file). tok/s:

| Checkpoint | engine | pp512 | pp2048 | tg128 | pp2048 / tg128 at 8k | pp2048 / tg128 at 32k |
|---|---|---:|---:|---:|---:|---:|
| ggml-org Q8_0 | llama.cpp | 3236 | 2747 | 87.8 | 1943 / 85.7 | 1094 / 79.7 |
| | **oxinfer** | **3558** | **4261** | **120.3** | **3504 / 110.2** | **2333 / 98.0** |
| unsloth UD-IQ3_XXS | llama.cpp | 3284 | 2784 | 98.2 | 1963 / 96.0 | 1098 / 88.3 |
| | **oxinfer** | **3321** | **3817** | **131.5** | **3202 / 120.9** | **2181 / 106.8** |
| AtomicChat AD-Q4_K | llama.cpp | 3243 | 2752 | 96.6 | 1948 / 94.4 | 1093 / 87.0 |
| | **oxinfer** | **3503** | **4033** | **131.1** | **3338 / 120.2** | **2239 / 106.4** |
| release (NVFP4) | oxinfer | 3481 | 4227 | 116.4 | 3484 / 107.5 | 2317 / 95.8 |

oxinfer's caches are f32 by default; with `--kv-cache f16` decoding at 32k is 102.1 (Q8_0), 110.8,
110.3 and 99.5 tok/s. At 32k of context llama.cpp's prefill drops to 40% of its 2048-token rate,
oxinfer's to 55%. llama.cpp has no reader for the release's NVFP4 safetensors.

### Threadripper 9970X and RTX 5080 (16 GB)

Threadripper 9970X (32 cores / 64 threads, AVX-512 with VNNI, 125 GB; 32 threads, the default)
and an RTX 5080 (16 GB, 2.0 of them held by the desktop: 14.2 GB free), Linux, shared with other
sessions' builds and model jobs (load averages 11-44 during these runs), so only rows of the same
round compare. oxinfer: `examples/inkling2_bench` (a warm-up pass, then the median of 3; `--cpu`
for the CPU rows), "before" master `422ea32`, "after" this work, both `--release`; "f32" is the
bench's own prompt pass (the f32 GEMM, which is `prefill_gemm=exact`), "fast"
`OXINFER_PREFILL_GEMM=fast`: the int8 activations a prompt gets by default from the command line
and the engine. llama.cpp: master `f4f6f2e` (2026-11-23) built here, `llama-bench -p 512,2048 -n
128 -r 3`, `-t 32` on a build without CUDA for the CPU rows, `-ngl 99 -fa 1` on a CUDA build
(13.4) for the GPU rows. Two rounds, each engine and build in turn; tok/s, round 1 / round 2:

| CPU, checkpoint | engine | pp512 | pp2048 | tg128 |
|---|---|---:|---:|---:|
| ggml-org Q8_0 (27.8 GB) | llama.cpp | 115 / 116 | 109 / 108 | 10.0 / 9.4 |
| | oxinfer before, f32 | 107 / 115 | 118 / 126 | 8.6 / 9.7 |
| | oxinfer before, fast | 124 / 123 | 144 / 137 | |
| | oxinfer after, f32 | 179 / 183 | 195 / 195 | 9.5 / 10.2 |
| | **oxinfer after, fast** | **221 / 216** | **259 / 239** | |
| unsloth UD-IQ3_XXS (11.7 GB) | oxinfer before, f32 | 79 / 81 | 95 / 94 | 11.3 / 12.5 |
| | oxinfer before, fast | 84 / 85 | 102 / 104 | |
| | oxinfer after, f32 | 107 / 110 | 132 / 130 | 12.8 / 13.4 |
| | **oxinfer after, fast** | **237 / 244** | **267 / 259** | |

(Decoding has one path: the tg128 of a build is on its f32 row. Nothing changed in Q8_0's decode
path: its tg128 rows differ by what two rounds of one build differ by. Mainline llama.cpp does
not load unsloth's root files, `check_tensor_dims: tensor 'blk.24.attn_q.weight' not found`, and
the repository no longer has the `Mainline_Support/` copies.)

| RTX 5080, checkpoint | engine | pp512 | pp2048 | tg128 |
|---|---|---:|---:|---:|
| unsloth UD-IQ3_XXS (11.4 GB of weights on the device) | oxinfer | 1695 / 1702 | 1865 / 1895 | 97.9 / 99.6 |
| AtomicChat AD-IQ2_M (8.7 GB; 8.4 GB on the device) | llama.cpp | 2049 / 2051 | 1760 / 1747 | 81.5 / 82.0 |
| | oxinfer before (every pass on the CPU) | 128 | 137 | 13.4 |
| | **oxinfer after** | **1824 / 1846** | **2092 / 2066** | **104.2 / 103.7** |
| ggml-org Q8_0 (27.3 GB of weights: does not fit) | llama.cpp, `-ngl 0 -b 512` | 233 | | 7.7 (tg64) |
| | oxinfer (every pass on the CPU, fast) | 217 | 250 | 9.5 |

**What changed.**

- **A prompt chunk's temporaries are kept for the pass** (CPU, every `prefill_gemm`). A layer
  application took about forty temporaries of megabytes (a chunk of 2048 tokens: 19 MB each, the
  experts' gathered rows and outputs 151 MB) from the allocator and gave them back, 48 times a
  pass, and the process spent its time in the system mapping them again: a pp512 + pp2048 run of
  the int8 expert GEMM took 13 minutes of user time and 45 of system time, and with the outputs
  of one grouped GEMM written once on one thread beforehand (so that its 32 threads no longer
  touched fresh pages) that GEMM went from 55 to 22 ms. The pass now takes them from and gives
  them back to `kernels::Scratch` (the buffers the Qwen-family forward already kept; those that
  are added to are zero-filled across the pool, the others written whole), the int8 activations
  and the f32 GEMM's packed tiles included. Same bits: the logits of the last 64 of 600 prompt
  positions and of 16 greedy steps are byte-identical to master's on Q8_0 and UD-IQ3_XXS
  (`inkling2_cuda_check --cpu --save --all --keep 64`, `cmp`; the final build, with the changes
  below). The prompt columns of the f32 rows above are this change alone (Q8_0 1.6x,
  UD-IQ3_XXS 1.35x).
- **The experts' grouped GEMMs on the int8 VNNI GEMM** (x86-64 with AVX-512 VNNI, under `fast`;
  aarch64 already had its own). Each expert's tokens are rounded to int8 in groups of 16 of their
  own, a pool job is 32 rows of one expert over up to 512 of its tokens, a segment's padding rows
  are skipped. Matrices with a factor as wide as their rows (unsloth's per-expert scales, NVFP4's
  tensor-wide one: 20 of UD-IQ3_XXS's 24 layers have them on both expert matrices) take the int8
  GEMM too, the factor multiplied into each row's sums as the decode kernels do; before, one such
  matrix kept a whole grouped matmul on the f32 GEMM. With the first change in both builds (the
  `ci` profile, medians of 3): Q8_0 pp512 189 -> 212 and pp2048 214 -> 246, UD-IQ3_XXS 117 -> 222
  and 137 -> 255. Against the f32 prefill (the last 64 of 600 prompt positions): mean KL 2.8e-3
  (Q8_0) and 2.6e-3 (UD-IQ3_XXS), the same top token at 64 / 64 and the same 17 greedy tokens;
  with f32 experts and int8 dense matmuls, as before, 2.4e-3 and 3.0e-3 (63 / 64 on UD-IQ3_XXS).
  Without the first change this GEMM was slower than the f32 one (Q8_0 pp512 53-60 against
  104-115): its 32-row jobs write 128 bytes per token each, so the 32 threads wrote the same
  fresh pages at once.
- **NVFP4 blocks unpacked with AVX2.** NVFP4 was the one format of the x86-64 decode kernels
  still on the scalar unpacker. UD-IQ3_XXS (NVFP4 down experts on 20 layers), tg64, two
  alternated rounds: 12.4 / 11.7 -> 13.6 / 12.8 tok/s, the matmuls 67.9 / 67.8 -> 57.8 / 59.8 ms
  a token; the same logits digest and tokens (the unit tests compare the unpacked values and the
  rows with the portable kernel's bit for bit).
- **Q2_0 on the CUDA engine.** The engine had no kernel for ggml's 2-bit Q2_0 (type 42), so
  AD-IQ2_M, whose down experts are Q2_0 on 16 of 24 layers, ran every pass on the CPU, on any
  card. It is now a kind of the engine's kernels like the others. Against the CPU path on the
  RTX 5080 (300 tokens): AD-IQ2_M prefill over every position mean KL 2.2e-6 (6.4e-4 at the
  worst), the same top token at 300 / 300, 32 decode steps against the CPU's f32 prefill KL
  6.2e-10, 17 / 17 greedy tokens; UD-IQ3_XXS prefill KL 4.3e-11 (300 / 300), decode steps
  4.2e-11, 17 / 17, as on the RTX PRO 6000.

**A checkpoint that does not fit the card** keeps every pass on the CPU: the engine wants the
weights, 0.5 GB of activations and 1 GB to spare in the device's free memory and declines
otherwise (`OXINFER_GPU_PLAN_DEBUG=1`: `27.3 GB of weights and 0.5 GB of activations do not fit
in 14.2 GB of free device memory`); nothing is streamed to the device and there is no error.
Q8_0 then runs at the CPU table's speed (the row above, measured with the device present).
llama.cpp fails to load it with `-ngl 99` and fails to create its context with 4 to 11 layers
on the device (13 GB of compute buffers, with `-b 512` too); with `-ngl 0` it runs the prompt's
matmuls on the GPU from host weights (pp512 233-248) and aborts on a 2048-token batch (out of
device memory).

**Where the time goes on the CPU now** (`INKLING2_BENCH_PROF=1`, per parallel region, the `ci`
profile). A prompt, pp2048, fast: UD-IQ3_XXS 3.9 ms a token, of which the matmuls 1.6 (the
experts' grouped GEMMs 1.2, the dense projections 0.5), attention 1.0, the mixes, gates and norms
0.6 and 0.75 outside the timers (convolutions, clips, the caches' rows, copies); Q8_0 4.1 ms
with the same split. A decoded token reads 3.1 GB of UD-IQ3_XXS and 5.3 GB of Q8_0, and the large
matmuls stream at what the machine gives (67-68 GB/s measured with `membench`, see
[performance.md](performance.md#x86-64-cpu-kernels-november-2026)):

| ms a token (GB read, GB/s) | experts | dense projections | LM head | gates' float matmuls | step |
|---|---:|---:|---:|---:|---:|
| UD-IQ3_XXS | 23.1 (1.30, 56) | 22.7 (1.29, 57) | 4.6 (0.32, 69) | 9.3 (0.23, 25) | 80 |
| Q8_0 | 48.6 (3.25, 67) | 23.3 (1.37, 59) | 7.0 (0.49, 70) | 10.0 (0.23, 23) | 110 |

So decoding is bound by memory bandwidth here (5.3 GB at 68 GB/s is 78 ms), and level with
llama.cpp's on Q8_0 (9.5 / 10.2 against 10.0 / 9.4 tok/s). What is not: the 96 message gates of
a token, whose float matmuls (64 x 4608, then two of 2304 x 64) are 194 parallel regions of
1.2 MB at a third of the bandwidth, on the 16-token f32 GEMM tiles with one token in them.

**Where the time goes on the RTX 5080** (`OXINFER_INKLING_PROFILE=1`, UD-IQ3_XXS, CUDA events
around each launch: 12.9 ms a token that way, 10.0 without): a decoded token is about 1290 kernel
launches in one graph, the routed experts 23% of the kernel time (22-40 us a launch), the dense
matvecs and the router 25% (6-16 us), the LM head 3.5%, and the rest in kernels of 4-17 us each:
the mixes 12%, the gates 10%, the convolutions 8%, the norms 5.5%, the residual adds, cache
stores, routing, vector clips and attention 2-3% each. No kernel stands out: the step is a chain
of many short ones, as on Metal.

**Measured and not kept** (this machine): the gates' two tanh per value across the pool for a
decode step (no change: 10.1 / 9.6 against 10.4 / 9.4 ms of mixes and gates a token); transparent
huge pages for the heap (`glibc.malloc.hugetlb=1`: tg64 13.5 against 12.5-13.1 tok/s on the
weights, and pp512 28 against 68, pp2048 87 against 65 tok/s on the temporaries before they were
kept); whole rows instead of 1024-column blocks in the int8 GEMM for the experts' short segments
(no change, measured before the temporaries were kept); the
softmax exponentials of a prompt's attention through `fast_exp` under `fast` (attention 1.0-1.1
against 1.0-1.2 ms a token: the exponentials are not what it costs). The grouped int8 jobs' row
ranges start where every token's outputs cross a cache line (a large `Vec` starts 16 bytes into
one): 3-4% of the grouped GEMM over two rounds, kept.

## Speculative decoding

A session goes back up to 64 positions (`Inkling2Session::truncate`: the convolutions read
their last inputs from rings of `kernel - 1 + 64` rows indexed by position, the sliding windows
keep 64 positions more), so a rejected draft is undone exactly. Three drafters:

- **The release's MTP head** (`--mtp`, `mtp=1`; `mtp.safetensors` beside the release) and
  **unsloth's conversion of it** (`--draft PATH` or `--draft auto`, `draft=...`:
  `dspark-Inkling-2-Mini-Q8_0.gguf` at the root of `unsloth/Inkling-2-Mini-GGUF`, found beside
  unsloth's files and, for the release and ggml-org's files, in the download cache). They are the
  same 3-layer block drafter (`src/inkling_draft.rs`), run by the DSpark decoder
  (`crate::dspark`), as llama.cpp's `graph_inkling2` (ggml-org/llama.cpp PR 32218, `src/models/
  dspark.cpp`; transformers has no definition of it): per draft layer a softmax mix (one row of
  `fuse_weight`) of the target's streams entering applications 5, 23 and 41 and of its
  final-normed output, normed (`dflare_norm`) and projected by the layer's own context
  projections into its key / value cache; a block of the anchor and 7 mask rows through 3 Qwen3-
  style layers (RoPE theta 1e6, non-causal, windows of 1024 on layers 0 and 2) and the target's LM
  head; and a recurrent head (rank 256, `rnn_proj` over `[state, markov_w1[prev], hidden,
  context]`) whose correction also goes through the target's LM head (run here as one product with
  `head * hidden_out`, `[vocab, 256]`, made once from the target's head). The GGUF and the
  safetensors tensors were checked to be the same values (the fuse weights, norms and mask
  byte for byte after the upcast).
- **Prompt lookup** (`--lookup`, `lookup=1`, on by default as for the other families).

On the CPU a round verifies at most 6 drafts (`inkling::MAX_DRAFT`): a pass of up to 7 tokens
stays on the CPU's decode kernels, routed experts included, whose rows are those of one-token
forwards bit for bit; an 8-token pass takes the prefill path, which rounds differently (greedy
output then drifted at near-ties) and costs several times more. When the passes run on the GPU,
whose passes of up to 16 tokens give one-token results, the bound is 15
(`Inkling2Model::max_draft`; the drafter's block gives 7).

Greedy output with each drafter is the plain greedy output token for token: 45 runs (3
checkpoints x code / prose / chat x lookup, cost-aware draft, 3 and 6 fixed drafts; 96 tokens,
`--no-system`) all gave the plain text. Acceptance (drafts accepted / drafted, tokens per
target forward; CPU, `--draft-tokens N --mtp-fixed`):

| Checkpoint, drafter | code, 3 drafts | prose, 3 | chat, 3 | code, 6 | prose, 6 | chat, 6 |
|---|---|---|---|---|---|---|
| ggml-org Q8_0, unsloth's Q8_0 draft | 51%, 2.50 | 43%, 2.26 | 35%, 2.02 | 35%, 2.97 | 26%, 2.44 | 20%, 2.16 |
| release (NVFP4), `--mtp` (bf16) | 47%, 2.38 | 51%, 2.50 | 35%, 2.02 | 31%, 2.79 | 31%, 2.79 | 20%, 2.16 |
| unsloth UD-IQ3_XXS, unsloth's Q8_0 draft | 49%, 2.44 | 35%, 2.02 | 45%, 2.32 | 27%, 2.57 | 18%, 2.07 | 25%, 2.44 |

On the CPU none of it pays: the experts make a verification pass cost nearly its token count in
one-token steps (each token reads 8 experts of 128 of its own; a pass of 4 tokens took 140-290
ms against 50-70 for one token, of 7 tokens 160-380 ms), so 2-3 tokens per forward decode slower
than plain steps. The default cost-aware rounds therefore fall back to plain steps after a few
probes (1.1-1.6 tokens per forward, within 10-30% of plain decoding on a machine at load 15-33;
the draft's first pass also builds its head product, about 0.5 s). Prompt lookup found few matches
on these prompts (1.01-1.13 tokens per forward).

On the GPU the whole draft pass is one Metal submission (`src/metal_inkling_draft.rs`): the 3
layers over the block on the decode library's multi-token matvecs, RoPE, the block's attention
over a device copy of the draft's cache, the target's LM head over the draft rows, and the
recurrent head's chain with each slot's greedy token chosen on the device (its logits come back
for the acceptance rule; a sampled choice other than the slot's first maximum ends the block
there). The context of committed positions is injected in a second submission (0.3 ms), and the
device cache follows the host's (which stays authoritative: rows it gained otherwise are uploaded
first). A draft pass of 3 slots takes 3.2-3.4 ms on the GPU, 4-5 ms in all (12-18 ms before, with
one submission per matmul of 4 rows). The drafts are the CPU drafter's: the same acceptance, token
for token, on the runs checked.

**Against the reference.** llama.cpp's implementation (PR 32218 at 0a9105de, built here:
`llama-speculative-simple --spec-type draft-dspark --spec-draft-n-max 3`, greedy) and oxinfer
on the same raw prompts (the chat template's rendering of one user turn, thinking effort 0.9)
with ggml-org Q8_0 and unsloth's draft accept the same share of 3 drafts: 78% / 77% on code
(89 / 88 of 114), 47% / 48% on prose, 38% / 35% on chat. A line-by-line reading of the two found
no difference either: the head's state starts at zero each round and is carried slot to slot,
`prev` is the anchor then each slot's first maximum; the head's context is the mix of position
`n - 1` (the last position the target ran); the features are the streams entering visits 5, 23
and 41 of the 48 (a later pass of the loop restarts from the prelude's output, as here) and the
final-normed output; the block is always 8 rows (later mask rows change earlier predictions
through the non-causal attention), drafts read rows 1..7 (`sample_from_anchor` false). The lower
acceptance in the table above is that of its prompts (rendered without the effort line, the
model answers without thinking; a thinking preamble restates the request, which the drafter
predicts well). oMLX's 63-65% on code are on its own prompts.

**One-time costs.** The head's product with the target's LM head (`[vocab, 256]`) took 250-320
ms in the first draft pass, a quarter of a 128-token reply's decoding: 44 ms of LM head and
200+ ms of a single-threaded transpose. Transposed in parallel tiles and made at the end of the
prefill with the device drafter (51 ms in all), it no longer counts in decoding.

**Draft length.** The cost-aware rounds drafted the whole block (7) whenever they drafted, then
verified the most profitable part; but a draft pass on the GPU costs more for more drafts (its
head runs once per slot, reading the 103 MB product each time). They now also choose how many to
draft: the length with the most expected tokens per second, from each slot's recent acceptance
rate given the slots before it were accepted (decayed over about 30 rounds) and the measured
cost of drafting and verifying each length; one round in 8 drafts one more, so the next slot
stays measured (`dspark::Adaptive::best_k`). A/B against the whole block (256 tokens, Q8_0,
load 5): code 79.1 / 78.6 -> 80.1 / 79.9 tok/s, prose 60.9 / 60.3 -> 68.2 / 67.8, chat 53.3 /
49.5 -> 54.7 / 47.1 (both near plain: few draft rounds).

Decode tok/s on the GPU (same raw prompts, 256 tokens, load 4-8; every output the plain greedy
text; "3 drafts" is `--mtp-fixed --draft-tokens 3`, "auto" the default cost-aware rounds):

| Checkpoint, drafter | | plain | 3 drafts | auto |
|---|---|---:|---:|---:|
| ggml-org Q8_0, unsloth's draft | code | 55-58 | 70-85 | 65-81 |
| | prose | 55-58 | 51-53 | 60-65 |
| | chat | 56-57 | 45-46 | 54-55 |
| release, `--mtp` | code | 55.8 | 84.0 | 80.4 |
| | prose | 51.6 | 56.3 | 53.5 |
| | chat | 49.9 | 53.1 | 48.3 |
| unsloth UD-IQ3_XXS, its draft | code | 62.5 | 57.7 | 58.0 |
| | prose | 54.1 | 44.4 | 54.2 |
| | chat | 59.3 | 43.7 | 58.3 |

(At 128 tokens, Q8_0: code 58.5 -> 93.8-95.8 with 3 drafts, prose 57.9 -> 71-77, chat even.)
Drafting pays where the drafts are accepted (code: +45-65%); on prose and chat (35-48% of 3
drafts) it is about even, and the cost-aware rounds stay near plain speed where it does not pay.
At 4 bits (UD-IQ3_XXS) a verification pass of 4 tokens costs two steps (32 against 15 ms: the
experts the tokens do not share), so even code loses with fixed drafts; the cost-aware rounds
keep plain speed there.

**Tree verification** (several candidates per slot, verified in one pass) is not worth it on
this model: every node of a tree is a token of the verification pass and reads its own 8
experts, 3.1-3.6 ms each (a fifth of a step), and a sibling only helps when the first choice
was rejected (35-65% of rounds on prose and chat) and is then right itself; it pays only if the
draft's second choice is right more than about a third of those times. The drafter also gives
one chain (each slot's head is conditioned on the previous slot's token), so a branch needs its
own draft pass. Shorter chains chosen per round (above) are the better use of the expert floor.

```sh
oxinfer --model thinkingmachines/Inkling-2-Mini --mtp --prompt "..."
oxinfer --model unsloth/Inkling-2-Mini-GGUF:UD-IQ3_XXS --draft auto --prompt "..."
```

## iOS / iPadOS

Mini runs on iPhones and iPads through the C API (`ios/`, the `OxinferTest` app): the same
engine, the weights memory-mapped (the default on iOS: mapped file pages are clean memory, not
charged to the app the way copied ones are). What the app may use decides how it runs:

- **The budget**: what `os_proc_available_memory` leaves the app less 1 GB, at most 60% of the
  device's memory and 85% of Metal's working set (`stream::DeviceMemory`, as for Qwen and
  Gemma). Mini needs beside its weights 1.3 GB of runtime, the sliding windows (195 MB with
  `kv_cache=q8`) and 18 KB a token of context (q8; `inkling::runtime_bytes`, which assumes 4096
  tokens unless `max_context` says otherwise); prefill chunks are 512 tokens.
- **Weights within the budget** (less that runtime): every pass on the GPU, as on a Mac
  ([On the GPU](#on-the-gpu)).
- **Larger**: streamed from flash (`crate::stream`): the dense weights (0.83 GB) stay resident,
  the routed experts are read as they are routed and the most recent kept in what is left of the
  budget; the passes run on the CPU (the whole-pass GPU engine does not stream; the Metal backend
  still takes the prefill matmuls). The output is the same as without streaming.
- **A GPU engine never maps more than the budget**: weights over it keep the passes on the CPU
  (`[inkling: the 8.5 GB of weights exceed the GPU budget ...]`), so `stream=off` on a phone
  demand-pages the mapped file on the CPU instead of wiring it for the GPU.
- `memory_limit_gb` (C API, library, server; `--memory-limit-gb` on the command line) caps the
  app's memory: the same decisions on a Mac with a phone's 6-8 GB, or a smaller share on purpose.
  `OXINFER_DEVICE_MEM_GB` / `OXINFER_APP_MEM_GB` / `OXINFER_GPU_WORKING_SET_GB` set the device's
  figures instead (`SIMCTL_CHILD_` prefix in the Simulator).

What fits, with `kv_cache=q8` and 4096 tokens of context (the weights' share is the budget less
0.5 GB; each 1k tokens more takes 19 MB off it):

| Device | Memory | Budget (60%) | In memory, GPU passes | Streamed (CPU passes) |
|---|---:|---:|---|---|
| iPhone 15 Pro / Pro Max, 16 / 16 Plus / 16 Pro / 16e, 17, iPad Pro M4 256-512 GB | 8 GiB | 5.2 GB | none | `UD-IQ2_XXS` (8.5 GB): 0.8 GB of dense weights resident, about 3 GB of experts cached |
| iPhone 17 Pro / Pro Max / Air, iPad Pro M5 256-512 GB | 12 GiB | 7.7 GB | none (`UD-IQ1_S`, 7.8 GB, is 0.5 GB over) | `UD-IQ2_XXS`: about 5.5 GB of experts cached; `UD-IQ3_XXS` (11.7 GB) |
| iPad Pro M4 / M5 1-2 TB | 16 GiB | 10.3 GB | `UD-IQ2_XXS` (8.5 GB) up to about 75k tokens, `AD-IQ2_M` (8.7 GB) up to about 65k, `UD-Q2_K_XL` (9.1 GB) up to about 40k | `UD-IQ3_XXS` (11.7 GB) |

The budget assumes the app is allowed at least 60% of the memory plus 1 GB, which takes the
increased-memory-limit entitlement (below); with a lower limit it is that limit less 1 GB, and
`oxi_engine_info_json` reports what was decided (`device_memory`, `stream`). On the 8 and 12 GB
phones `UD-IQ2_XXS` is the file: the smallest that keeps the model's quality (unsloth's dynamic
2-bit; `UD-IQ1_S` saves 0.7 GB, not enough to fit anywhere it would not), and streaming it
reads, measured on the Mac with the phone's budgets emulated (200 tokens of a story, after the
prompt), 0.09 GB from flash per token with a 4.3 GB budget (stalled 22% of the time at the
Mac's SSD speed) and 0.01 GB with 6.7 GB (6%). `UD-IQ3_XXS` streams on the 12 GB phone too, at
a higher quality and more reading. The app's own (dirty) memory stays small either way: peak
footprint 0.57-0.71 GB streaming, 1.46 GB with the whole passes on the GPU over the mapped file
(macOS's accounting, UD-IQ2_XXS, 200 tokens; Metal's mapping of the file is not charged
there).

**Options** (one `key=value` per line in OxinferTest): `kv_cache=q8` (a quarter of the f32
caches, KL against f32 within the GGUF quantization's), `max_context=8192` (or more on the iPad;
only the context actually used costs memory, but the budget is reckoned for this length);
nothing else is needed: streaming and its budget, mmap, the GPU or CPU passes and the prefill
chunk follow the device. `draft=auto` (unsloth's DSpark drafter, 0.4 GB more) is not worth it
when streaming (each drafted token routes its own experts).

**Entitlements**: `com.apple.developer.kernel.increased-memory-limit` (without it an app gets
roughly half the memory and the budget follows: an 8 GB phone still streams, a 16 GB iPad then
streams too) and `com.apple.developer.kernel.extended-virtual-addressing` (to map an 8-12 GB
file); both are in `OxinferTest.entitlements` and need a paid developer account.

**Verified in the Simulator** (Apple silicon, iOS 26.3, iPhone 17 Pro; `ios/capi_greedy.c`
through `xcrun simctl spawn`, the model files read from the Mac's cache), greedy, no system
prompt, against the same program on the Mac:

| File | CPU | Metal (whole passes) | Streamed, phone budget |
|---|---|---|---|
| unsloth UD-IQ2_XXS (8.5 GB) | 200/200 tokens identical | 200/200 identical | identical (8 GB phone: CPU passes, 4.3 GB budget) |
| AtomicChat AD-IQ2_M (8.7 GB) | identical | identical | |
| unsloth UD-IQ3_XXS (11.7 GB) | identical | identical | identical (12 GB phone: 6.7 GB budget) |

(A 200-token story after a 24-token prompt; the streamed runs against the Mac streaming with
`memory_limit_gb`, and against the CPU path without streaming, which they also equal; the
16 GB iPad's budget kept UD-IQ2_XXS in memory with GPU passes, again the same text.)

The Simulator's GPU has no matrix units, so it runs the kernels the Mac runs with
`OXINFER_METAL_MMA=0`, and the outputs are compared with those (with the matrix units the Mac's
prefill rounds differently and its text departs after about 30 tokens, at a near tie). The
Simulator reports no app limit (`os_proc_available_memory` is 0 there) and the Mac's memory, so
the phone budgets were set through the environment.

**Not verified**: a real iPhone or iPad. The speed there (the A18 / A19 CPU, the GPU, flash
reads under streaming), the app limit the entitlement actually grants on each model, whether
Metal's mapping of the file counts against that limit, and jetsam's behaviour near it are
unmeasured; the Simulator's speed means nothing (its Metal is the host GPU behind a
virtualization layer). [ios/README.md](../ios/README.md) has the steps to try it on a device.

## Not done

- 125 tok/s of decoding (the figure reported with EXO, see above): 66 at Q8_0, 82 at 4 bits
  (MLX) and 80 at 3 bits (UD-IQ3_XXS) here. What is left at 4 bits is the experts' bytes (4.7 ms), the dense matmuls (2.1 ms), and
  attention, mixes, gates, convolutions and routing as kernels of one or a few threadgroups per
  token whose latency adds up over 751 dependent stages; fusions that add work to every
  threadgroup of a stage lose (above).
- Speculative decoding is bounded by the experts a verification pass reads (3.1 ms a token at 4
  bits, above), whatever the drafter: no batching of the pass shares experts the tokens do not
  share. The multi-token matvecs of the rest (2.2 ms more for 4 tokens than for 1) could shrink
  with kernels of their own (the decode library's reload each token's input per row pair).
- A CUDA drafter (the DSpark / MTP block runs on the CPU there: no gain over plain decoding).
- A checkpoint larger than the card's free memory on the GPU: it runs on the CPU alone
  ([above](#threadripper-9970x-and-rtx-5080-16-gb): Q8_0 on 16 GB). The looped layers 6-17 are 36
  of a step's 48 applications and half of the experts, so an engine that kept those layers on the
  device and ran the others on the host would have three quarters of a step there; it was not
  built (the stream and the block states would cross between host and device at each boundary,
  and a session's caches would live on both sides).
- On the x86-64 CPU: the message gates' float matmuls of a decode step (9-10 ms of an 80-110 ms
  token: 194 parallel regions of 1.2 MB each on GEMM tiles of 16 tokens holding one; a one-token
  kernel with the same summation order would want those matrices by columns), and a prompt's
  attention (1.0 ms of 3.9 a token at pp2048, next to 1.6 for all the matmuls).
- On CUDA: the 1290 launches of a decoded token were not fused or run as parallel branches of the
  graph (measured only, on the RTX 5080: the routed experts and the dense matvecs are half of
  the kernel time, kernels of 4-11 us the other half).
- No GGUF of Small was tried (the same code reads its layout).
