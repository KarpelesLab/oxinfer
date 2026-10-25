# Core ML (Apple Neural Engine) package format

`tools/coreml_convert.py` writes Hugging Face checkpoints as Core ML packages in the layout of
[ANEMLL](https://github.com/Anemll/Anemll) 0.3.5 (commit `f4ad26d`), so ANEMLL's runtimes
(`tests/chat.py`, the Swift CLI) and oxinfer's Core ML runtime load them. This document is the
runtime contract: the files, every model input, output and state, and the call sequence. Plain
transformer packages (`architecture: qwen3`) need nothing beyond ANEMLL's contract. Two
architectures extend it (marked **extension**): `qwen3_5` (hybrid recurrent: more states and two
calling rules) and `gemma4` (one more embeddings output / decoder input and two KV caches).

`tools/coreml_check.py` is a reference implementation of the runtime side in Python
(coremltools); it is the code to read when something here is unclear.

## Files

```
meta.yaml                                       description + runtime parameters
<prefix>_embeddings[_lutN].mlmodelc             token ids -> hidden states
<prefix>_FFN_PF[_lutN]_chunk_01ofNN.mlmodelc    decoder layers, functions `infer` and `prefill`
...
<prefix>_FFN_PF[_lutN]_chunk_NNofNN.mlmodelc
<prefix>_lm_head[_lutN].mlmodelc                final hidden state -> logits
config.json                                     source config.json (+ tokenizer_class)
tokenizer.json, tokenizer_config.json, ...      tokenizer files copied from the checkpoint
generation_config.json, chat_template.jinja     when the checkpoint has them
```

`_lutN` names the palettization bits (`_int8` / `_int4` for linear quantization). The default
prefix is `qwen` for Qwen3, `qwen35` for Qwen3.5 and `gemma4` for Gemma 4. All models are ML
Programs for iOS 18 / macOS 15 or later (Core ML states), fp16 compute.

## meta.yaml

```yaml
model_info:
  name: anemll-Qwen-Qwen3.5-0.8B-ctx512
  version: 0.3.5                 # ANEMLL format version
  architecture: qwen3_5          # qwen3 | qwen3_5 | gemma4
  parameters:
    context_length: 512          # KV cache length = largest position + 1
    batch_size: 64               # tokens per `prefill` call
    lut_embeddings: none         # palettization bits per part, or none
    lut_ffn: 6
    lut_ffn_per_channel: 8       # output channels sharing one LUT
    lut_lmhead: 6
    lut_lmhead_per_channel: 8
    num_chunks: 1
    model_prefix: qwen35
    embeddings: qwen35_embeddings.mlmodelc
    lm_head: qwen35_lm_head_lut6.mlmodelc
    ffn: qwen35_FFN_PF_lut6_chunk_01of01.mlmodelc   # first chunk (others: _chunk_02ofNN ...)
    split_lm_head: 16            # number of logits outputs
    vocab_size: 248320
    lm_head_chunk_sizes: [15520, 15520, ...]          # width of logits1..logitsN
    prefill_dynamic_slice: true  # prefill writes the KV cache at current_pos (ANEMLL flag)
    prefill_last_token_output: true   # see "prefill output" below
    hidden_size: 1024
    num_layers: 24
    recommended_sampling: {do_sample: true, temperature: 0.6, top_p: 0.95, top_k: 20}
    # --- extensions (qwen3_5) ---
    recurrent_state: true        # states that cannot be rewound: feed every token exactly once
    prefill_full_batches_only: true
    gdn_chunk_size: 64           # informational (chunk of the in-graph delta rule)
    moe: {experts: 16, experts_per_token: 4, evaluation: dense}   # MoE models only, informational
    # --- extensions (qwen3_5 and gemma4) ---
    layer_types: [linear, linear, linear, full, ...]
    states:
      - name: model_model_kv_cache_0
        shape: [12, 2, 512, 256]
        dtype: float16
        layers: [3, 7, 11, 15, 19, 23]
      - name: model_model_gdn_state_0
        shape: [288, 128, 128]
        dtype: float16
        layers: [0, 1, 2, 4, ...]
      - name: model_model_conv_state_0
        shape: [18, 3, 6144]
        dtype: float16
        layers: [0, 1, 2, 4, ...]
```

For `gemma4` the extension keys are (E2B shown):

```yaml
    local_attention_window: 512      # informational: applied inside the graph
    local_cache_length: 640          # only when the local cache is a block-shifting window (see States)
    final_logit_softcap: 30.0        # informational: applied inside the LM head
    per_layer_inputs: 8960           # width of the per_layer_inputs tensor (layers x 256)
    kv_shared_layers: {15: 13, 16: 14, ...}   # informational: layer -> layer whose K/V it reads
    layer_types: [sliding, sliding, sliding, sliding, full, ...]
    states:
      - name: model_model_kv_cache_local    # sliding-window layers, head_dim 256
        shape: [12, 2, 512, 256]
        dtype: float16
        layers: [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13]
      - name: model_model_kv_cache_global   # full attention layers, head_dim 512
        shape: [3, 2, 512, 512]
        dtype: float16
        layers: [4, 9, 14]
```

Keys other runtimes do not know must be ignored. When `quant_ffn` (or `quant_lmhead`,
`quant_embeddings`) is present, the part uses linear weight quantization (`int8` or `int4,B`)
instead of a LUT and `lut_*` is `none`; it changes nothing at run time. Likewise
`lut_ffn_mixed: 4.50` (mixed precision: `lut_ffn` is the narrowest LUT, some weights use the
next width, 4 -> 6 bits, for that average) and the LUT method in the trailing comments are
informational: every weight is a plain `constexpr_lut_to_dense` whatever its width or method.

No format version was bumped for these extensions: packages written before them (full-length
local caches, no `local_cache_length`) keep loading and behave as before.

## Embeddings model

| | name | shape | type |
|---|---|---|---|
| input | `input_ids` | `[1, 1]` or `[1, batch_size]` (enumerated shapes) | int32 |
| output | `hidden_states` | `[1, T, hidden_size]` | fp16 |
| output (**extension**, gemma4) | `per_layer_inputs` | `[1, T, per_layer_inputs]` | fp16 |

A table lookup (Gemma: scaled by `sqrt(hidden_size)`, folded into the table). For Gemma 4 the
model also computes the per-layer embeddings (PLE): `embed_tokens_per_layer[ids] * sqrt(256)`
plus the RMS-normed projection of the token embedding, times `1/sqrt(2)`, packed as
`[layer][256]`; the runtime passes this tensor unchanged to every decoder chunk.

## LM head model

| | name | shape | type |
|---|---|---|---|
| input | `hidden_states` | `[1, 1, hidden_size]` (the final-normed hidden state) | fp16 |
| outputs | `logits1` ... `logitsN` | `[1, 1, lm_head_chunk_sizes[i]]` | fp16 |

N = `split_lm_head`. The logits of the vocabulary are the concatenation of `logits1..logitsN`
in order (the first `vocab_size % N` parts are one wider when the split is uneven). Gemma's
final soft cap (`30 tanh(x / 30)`) is applied inside the model.

## Decoder chunks

The layers are split into `num_chunks` consecutive ranges (layer ranges are in the model
metadata `com.anemll.layers`). Every chunk is a multifunction model with two functions sharing
weights and states:

| function | `hidden_states` | `position_ids` | `causal_mask` | `current_pos` | output `output_hidden_states` |
|---|---|---|---|---|---|
| `infer` | `[1, 1, H]` fp16 | `[1]` int32 | `[1, 1, 1, ctx]` fp16 | `[1]` int32 | `[1, 1, H]` fp16 |
| `prefill` | `[1, B, H]` fp16 | `[B]` int32 | `[1, 1, B, ctx]` fp16 | `[1]` int32 | `[1, B, H]` (non-last chunk) or `[1, 1, H]` (last chunk) |

with `H = hidden_size`, `B = batch_size`, `ctx = context_length`, for a call that processes the
tokens at positions `p .. p+T-1`:

* `position_ids` = `[p, p+1, ..., p+T-1]` (RoPE positions), `current_pos` = `[p]` (where the
  KV cache is written);
* `causal_mask` = rows `p .. p+T-1` of the `ctx x ctx` causal mask: `0` where column <= row,
  `-inf` elsewhere;
* `hidden_states` of the first chunk = the embeddings output, of chunk i+1 = the output of
  chunk i (an internal representation: the residual stream).
* The last chunk's `infer` output is the final-normed hidden state of the token, ready for the
  LM head. The last chunk's `prefill` output is the final-normed hidden state of the **last**
  token of the batch (`prefill_last_token_output: true`); ANEMLL's own packages return an
  unnormalized first-token state there, which runtimes ignore.
* **extension** (gemma4): every chunk has a fifth input `per_layer_inputs`
  `[1, T, per_layer_inputs]` fp16, the embeddings model's second output for the same tokens
  (each chunk reads the slices of its own layers). Sliding-window layers derive their window
  mask from `position_ids` and `causal_mask` inside the graph (keys at positions
  `<= p - local_attention_window` are masked), so the runtime passes the plain causal mask.

### States

All chunks declare the same states with the full shapes (every chunk only touches the slices of
its own layers), so one state object serves the whole model: create it once with
`make_state()` on the first chunk (either function) and pass it to every `predict` call of
every chunk. A fresh state (all zeros) is an empty context.

* `model_model_kv_cache_0` `[2 * n_attn, n_kv_heads, ctx, head_dim]` fp16: keys of attention
  layer `i` (the i-th full attention layer, counting from 0) at index `i`, values at
  `n_attn + i`; position `p` lives at `[:, :, p, :]`. K is stored after its RMSNorm and RoPE.
  For qwen3 every layer is an attention layer (`n_attn = num_layers`, ANEMLL's layout).
* **extension** `model_model_gdn_state_0` `[n_linear * n_v_heads, d_k, d_v]` fp16: the
  recurrent memory of every Gated DeltaNet value head; linear layer `j` (the j-th linear
  attention layer) owns rows `j * n_v_heads .. (j+1) * n_v_heads - 1`. It holds
  `sqrt(d_k) * S` where `S` is the transformers state (`S[k, v]`), a rescaling that keeps the
  fp16 products on the Neural Engine out of the subnormal range.
* **extension** `model_model_conv_state_0` `[n_linear, K - 1, conv_dim]` fp16: the last `K-1`
  inputs of the causal depthwise convolution of layer `j` (the `in_proj_qkv` outputs, before
  the convolution), oldest first. Time-major on purpose: a trailing axis of 3 makes the ANE fail.
* **extension** (gemma4) `model_model_kv_cache_local` `[n_local, 2 * n_kv_heads, ctx, 256]` and
  `model_model_kv_cache_global` `[n_global, 2 * n_kv_heads, ctx, 512]` fp16: one slot per
  non-shared sliding / full attention layer (in layer order, `layers` in meta.yaml), keys in
  `[slot, 0:n_kv]`, values in `[slot, n_kv:2 n_kv]` (one slice update per layer: two separate
  K and V updates with a single KV head fail to compile for the ANE). K is stored after its
  RMSNorm and RoPE, V after its weightless RMSNorm. The KV-shared layers (the last
  `num_kv_shared_layers`) write nothing and attend to the slot of `kv_shared_layers[i]`.
  The global cache spans the whole context. The local cache too when `context_length <=
  local_attention_window + 2 batch_size` (the window is then a mask, see above); for longer
  contexts it is a **block-shifting window** of `local_cache_length = local_attention_window
  + 2 batch_size` positions (meta.yaml key `local_cache_length`, absent for full-length local
  caches). Positions are grouped in blocks of `batch_size`; after a call whose last position is
  `e`, slot `j` holds position `base + j` with `base = (e // batch_size + 1) * batch_size -
  local_cache_length` (negative = empty, beyond `e` = not written yet). A call that enters a
  new block shifts the cache by one block (a prefill batch always enters exactly one, an
  `infer` call when its position is a multiple of `batch_size`), then writes its keys at their
  slots. A position's slot thus depends only on its block, so `prefill` and `infer` read the
  keys in the same order and give bit-identical results, as with full-length caches. One
  block of slack holds the not yet written part of the current block, the other keeps the
  window of a prefill batch that starts inside a block complete. The runtime contract does
  not change (the masks are derived inside the graph from `position_ids` and `current_pos`),
  but the cache cannot be rewound: positions must be fed exactly once and in order, as for
  `recurrent_state` (re-running the last prompt token, the ANEMLL habit, is not needed: these
  packages set `prefill_last_token_output`). Restarting at position 0 is fine without a new
  state (older slots then count as empty); to go back to any other position restore a snapshot
  of the state. E2B at context 8192: 8 MB of local cache instead of 100 MB.
  Packages written by the first version of this layout (`local_cache_length` = window +
  batch, 576 for E2B) dropped `T` positions per call; they work with the same contract, but
  their `prefill` and `infer` sum the attention in different orders (logit differences up to
  ~0.16, like those between two context lengths).

The names are the Core ML state names (`stateSchema` in the compiled model's `metadata.json`,
the same as ANEMLL's for the KV cache). Inference only passes the opaque state; the names matter
for reading or writing state contents (`MLState.withMultiArray(for:)`), e.g. to snapshot a
recurrent state.

## Call sequence

### Plain transformers (ANEMLL contract)

For a prompt of `n` tokens starting at position 0:

1. `p = 0`; while `p + B <= n`: embed tokens `p..p+B-1` (`[1, B]`), run `prefill` through
   all chunks, `p += B`.
2. While `p < n`: embed token `p` (`[1, 1]`), run `infer` through all chunks, `p += 1`.
3. Logits for the next token: LM head on the last chunk's output of the call that processed
   token `n-1`. (ANEMLL's runtimes instead re-run token `n-1` through `infer`, which is
   harmless with a KV cache: the same K/V are written again.)
4. Generation: for each new token at position `p`: embed, `infer` through all chunks, LM head,
   sample, `p += 1`. Stop at `ctx` (ANEMLL runtimes then shift the window and re-prefill).

Partial batches are not passed to `prefill` (they could write K/V beyond the end of the cache).

### Recurrent architectures (`recurrent_state: true`) — extension

The Gated DeltaNet state is updated by every call and cannot be rewound, so:

* **Every position is processed exactly once and in order.** In particular the ANEMLL habit of
  re-running the last prompt token through `infer` corrupts the state: take the logits from
  the call that processed token `n-1` (the `infer` output, or the last chunk's `prefill`
  output when the prompt ends on a batch boundary — `prefill_last_token_output`).
* **`prefill` only takes full batches** of `B` valid tokens (`prefill_full_batches_only`);
  padding tokens would be absorbed into the recurrent state. The remainder goes through
  `infer`.
* No rewinding or window shifting: to drop context (context full, edited history, rejected
  speculative tokens) create a new state and re-feed the tokens to keep, or restore a snapshot
  of all three states taken with `MLState.withMultiArray(for:)` at the position to return to.
* Positions still index the KV cache of the attention layers, so `ctx` remains the limit.

Everything else (inputs, masks, LM head) is as above.

## Architectures

### qwen3

Same graph as ANEMLL's Qwen3 converter: RMSNorm through `layer_norm` on `[x, -x]`, linear layers
as 1x1 convolutions, q/k RMSNorm then RoPE (rotate-half, full head dim), GQA by repeating the
cached K/V heads, softmax attention over the whole cache with the causal mask, SiLU MLP.
Differences: SiLU and sigmoid are written as `x * (0.5 + 0.5 tanh(x / 2))` because the ANE's
native sigmoid is a coarse approximation; the prefill output described above.

### qwen3_5 (Qwen3.5 / 3.6 / 3.8, dense and MoE)

* Zero-centered RMSNorm (`1 + w`, folded into the weights at conversion).
* Full attention layers (every `full_attention_interval`-th): `q_proj` produces query and a
  per-head output gate (split into two projections at conversion), q/k RMSNorm, partial RoPE
  (first `rot = head_dim * partial_rotary_factor` dims, rotate-half within them; for text the
  three M-RoPE sections share one position so it is plain RoPE, theta from `rope_parameters`),
  attention as qwen3, output multiplied by `sigmoid(gate)` before `o_proj`.
* Gated DeltaNet layers: `in_proj_qkv` -> causal depthwise conv (kernel `K`, state above) ->
  SiLU; q, k L2-normalized (repeated to `n_v_heads` when `n_v > n_k`), `beta = sigmoid(b)`,
  `g = -exp(A_log) * softplus(a + dt_bias)`; recurrence per value head
  `S = e^g S; S += k (beta (v - S^T k))^T; o = S^T q / sqrt(d_k)`; output
  `RMSNorm(o) * w * SiLU(z)` per head, `out_proj`.
  * `infer` runs the recurrence for one token.
  * `prefill` runs the chunked (WY) form of the same recurrence on chunks of
    `gdn_chunk_size` tokens (64 = the whole batch by default): decays from term-by-term sums
    of `g` (not differences of a long cumulative sum, which loses fp16 precision), the unit
    lower-triangular inverse by block doubling (`log2(C)` levels of two batched matmuls; the
    power series `(I - N)(I + N^2)...` overflows for correlated keys), then the usual
    `u = T (beta v)`, `w = T (beta k e^G)`, intra-chunk attention and state update. Every op is
    a matmul or elementwise op, all on the Neural Engine.
* MoE variants (`qwen3_5_moe`, `num_experts`): the sparse block is evaluated **densely**. The
  gate, up and down projections of all experts are three wide 1x1 convolutions; the router's
  softmax weights are kept for the top `num_experts_per_tok` experts (threshold found by k-1
  rounds of max-and-mask: reductions and compares, no `topk` op) and renormalized, every
  expert's SiLU-gated hidden state is scaled by its weight (0 for the others), and the shared
  expert is added times `sigmoid(shared_expert_gate)`. Same I/O and states as dense. Compute
  and weight memory grow with the total expert count, so this suits small MoE models only; a
  35B-A3B model would compute all 35B parameters per token. The router stays fp16 (a palettized
  router flips expert choices). Validated on random models (8 layers, hidden 1024, 32 experts,
  top-4): the float32 graph matches transformers (logit cosine 1.000000), every op runs on the
  ANE (prefill 8200 tok/s, decode 110 tok/s on an M5 Max); the fp16 package scores logit cosine
  0.9987 on the GPU and 0.94 on the ANE against float32. The ANE's per-layer error is the same
  as for the dense twin of that model (0.99947 end to end), but random peaky routers turn small
  differences into different experts: with 1% noise per layer the MoE model's final hidden
  state moves 30x more than the dense one's in float32 already.

### gemma4 (Gemma 4 E2B / E4B, text)

* Embeddings scaled by `sqrt(hidden)`; per-layer embeddings as above.
* Layer: `x += post_attn_norm(attn(input_norm(x)))`,
  `x += post_ff_norm(mlp(pre_ff_norm(x)))` (GELU-tanh gated MLP, double width on the KV-shared
  layers when `use_double_wide_mlp`), then
  `x += post_ple_norm(ple_proj(gelu(ple_gate(x)) * per_layer_input[layer]))`, `x *= layer_scalar`.
  RMSNorm with a plain weight.
* Attention: q/k RMSNorm with weight, v RMSNorm without, no `1/sqrt(d)` scaling. Sliding
  layers: head dim 256, RoPE theta 1e4 on all dims, window 512. Full layers: head dim 512,
  "proportional" RoPE (theta 1e6, only the first 25% of the frequency pairs rotate).
* Not supported: `attention_k_eq_v` (12B / 31B), the MoE block (26B-A4B), vision and audio.
* GELU is written with `x (1 + 0.044715 x^2)` inside the tanh so fp16 does not overflow.

## Neural Engine notes

What the converter does differently from a straightforward port, each found by comparing the
ANE output with torch:

* Sigmoid / SiLU through `tanh`: the ANE's sigmoid has errors up to ~0.015.
* Gated DeltaNet state stored as `sqrt(d_k) S`: with the unit `q`, `k` of the reference the
  products in `q^T S` fall into fp16 subnormals, which the ANE flushes (layer output cosine
  0.97 instead of 0.9999).
* No state may have a trailing axis of 3 elements (runtime failure), and every declared state
  must be read by every function (conversion failure), so chunks read one element of the
  states they do not use.
* Only palettized (LUT) weights run correctly: int8 per-channel or per-tensor linear weights
  compile but compute garbage on the ANE (logit cosine 0.75; 0.9992 on the GPU) and int4
  per-block weights do not compile. Measured on an M5 Max, macOS 26.
* Embedding tables stay fp16: Core ML memory-maps them and only pages in the rows a lookup
  touches (Gemma 4 E2B's 5.5 GB embeddings model: instant load, +40 MB RSS), whereas a
  palettized gather table is decompressed when the model loads (LUT4: 87 s, +16.7 GB RSS).
* The CPU_ONLY path of Core ML returns NaN for these stateful fp16 graphs (ANEMLL's own Qwen3
  release too); CPU_AND_GPU and CPU_AND_NE are correct.
* Neural Engine memory (wired memory added by loading a chunk, median of load/unload cycles):
  each function of a multifunction model gets its **own copy** of the weights on the ANE even
  though the package stores them once (Qwen3.8-27B, 4 layers: 0.8 GB for `infer` alone,
  1.3-1.7 GB with `prefill`), and a 3-bit LUT costs as much as a 4-bit one (0.81 vs 0.84 GB
  for the same 4 layers, 2-bit: 0.47 GB). The process footprint does not show it.
* Precomputed LUTs (`--lut-method fisher`) are written into the converted program by weight
  content, so the graph must not let Core ML rewrite a weight: its passes fold a constant
  scale that follows a convolution into the weights (the `y / 2` of `sigmoid(y) = 0.5 + 0.5
  tanh(y / 2)` after Qwen3.5's gate projection), hence that projection holds `W / 2`.
* LUT group size is a speed trade-off: one LUT per output channel (`lutN,1`) is the most
  accurate but halves decode speed on the ANE (Qwen3-0.6B LUT4: 41 vs 92 tok/s), 2 channels
  per LUT costs ~25%, 8 (the default) is as fast as ANEMLL's packages.

## Validation

`tools/coreml_check.py DIR --hf CHECKPOINT` greedy-generates with transformers (float32), feeds
the same tokens through the package (teacher forcing) and compares the logits at every generated
position; `--bench` measures prefill / decode speed, `--plan` lists the compute device of every
op (`MLComputePlan`).
