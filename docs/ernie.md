# ERNIE 5.5 Flash Base

`baidu/ERNIE-5.5-Flash-Base` (`ernie5_5_moe`, `Ernie5_5_MoeForCausalLM`; 280B parameters of which
14B per token, 43 layers of 4096, 512 experts of which 10 per token, a 1M-token context; 288 GB as
FP8, 573 GB as `baidu/ERNIE-5.5-Flash-Base-BF16`) runs on the DeepSeek V4 code (`ernie5.rs`, a
submodule of `deepseek4.rs`): its design is DeepSeek V4's with a second kind of attention layer
and a different feed-forward block. It is a base model, the checkpoint at the end of
pre-training: it continues a text, and has no chat template.

**The real weights were never run.** Neither release fits the 100 GB this work could download,
and the one GGUF conversion found (`orcarouter/ERNIE-5.5-Flash-Base-abliterated-GGUF`, for a
llama.cpp pull request that is not merged) is gated. What follows was measured on tiny random
checkpoints, on the headers of the released files, and on the released tokenizer.

```sh
oxinfer --model baidu/ERNIE-5.5-Flash-Base --prompt "Large language models are" --max-tokens 64
```

The prompt is fed as it is after `<|ERNIE_MASK|>` (the token the model's tokenizer puts in front
of a text, also its end-of-text token), and generation stops at `<|ERNIE_MASK|>` or
`<|ernie:im_end|>` (`generation_config.json`). There is no system role: a system prompt, the
default one included, is left out. A checkpoint larger than 60% of RAM is streamed from storage
as for DeepSeek V4 (a decoded token reads 14 GB of the FP8 release, 26 GB of the BF16 one).

## Architecture

What differs from [DeepSeek V4](deepseek-v4.md#deepseek-v4-flash), as Baidu's model code (below):

- Two kinds of attention layer, chosen per layer by `compress_ratios`. 37 layers have rate 128
  and are V4's "HCA" layers: one shared key / value head of 512, a sliding window of 128 raw
  keys plus one pooled key per 128 tokens, YaRN at `compress_rope_theta`. The 6 layers with a
  rate of 0 or less (one in seven; the MTP layer too) are multi-head latent attention layers
  (MLA, as DeepSeek V3: a query latent of 1024, a key / value latent of 512 plus 64 rotated
  dimensions cached per token, `kv_b_proj` giving each of the 64 heads its keys of 192 and its
  values of 256, plain RoPE at `rope_theta`) over the whole context. Past `index_topk +
  sliding_window` (2176) tokens they are sparse (DSA): a query attends to its window of 128 and
  to the 2048 older tokens a lightning indexer scores highest (64 index queries of 128 from the
  query latent against one index key per token, `LayerNorm(wk(x))`, both rotated on their first
  64 dimensions in half-split pairs; `sum_h w_h relu(q_h . k)`; ties go to the older token).
  Both kinds have a learned sink logit per head.
- On every attention output, the heads are mixed by `I + V U^T` (VHA, rank 16), and a sigmoid
  gate computed from the normed query latent (`gate_proj`) multiplies what the last projection
  reads: the grouped low-rank output of an HCA layer, the concatenated heads of an MLA layer.
- Hyper-connections as V4's, with the RMS factor `1 / (sqrt(mean) + eps)` before a block; the
  final collapse keeps V4's `rsqrt(mean + eps)`.
- Feed-forward: layer 0 has a dense MLP of 14336; the others a latent MoE. Two routers score each
  expert `sigmoid(l0) + sigmoid(l1)`; the 10 best of score plus a selection bias are kept, their
  scores renormalized and multiplied by a learned scale per expert. The routed experts run on
  `fc1_latent_proj(x)` (2048 wide), their weighted sum goes through an RMS norm and
  `fc2_latent_proj`; one shared expert runs on `x`. The activation is SiTU:
  `beta tanh(g / beta) sigmoid(g) * linear_beta tanh(u / linear_beta)` (beta 4, linear beta 25).
- MultiMax on the logits, with four learned ranges and four learned temperatures:
  `l + t0 relu(r0 - l) + t1 relu(l - r1) + t2 relu(r2 - l)^2 + t3 relu(l - r3)^2`.
- An MTP layer after the 43 ("ECHO"; the checkpoint stores it as layer 43, an MLA layer with its
  own weights): see below.

As for V4, the reference's quantization of the cached keys and of the indexer (FP8 or MXFP4, after
a Hadamard rotation that changes no score) is left out.

In memory, an MLA layer keeps `kv_b_proj` as float32 per head (59 MB for each of the 7), with the
key half transposed: the keys are absorbed into the queries (`K_h^T q`), so attention reads the
cached latent rows directly and each head's values are applied to the attended latent.

## Reference

The repository has no model code. Baidu publishes it as a vLLM plugin,
[baidu/vLLM-ERNIE](https://github.com/baidu/vLLM-ERNIE) (`src/vllm_ernie/models/ernie5_5/`), which
subclasses vLLM 0.32's DeepSeek V4 attention and runs on NVIDIA kernels only (FlashMLA, DeepGEMM,
TileLang): it cannot run a tiny checkpoint on a CPU. `tools/ref_ernie55.py` is that plugin's
forward pass in plain PyTorch, in float32:

- the HCA layers are DeepSeek's own `inference/model.py` `Attention` (of
  deepseek-ai/DeepSeek-V4-Flash: the code vLLM's class implements, with its kernels replaced as in
  `tools/ref_deepseek4.py`), with the plugin's two additions, the head mix and the gate;
- the MLA layers with their indexer, the hyper-connections, the SiTU MLP, the latent MoE, the
  MultiMax head and the MTP layer are written out from the plugin's code (`attention.py`,
  `mhc.py` and its tests' reference, `moe.py`, `model.py`, `mtp.py`), with the checkpoint's
  tensor names.

So the reference is Baidu's computation as this work read it, not Baidu's code running: a
misreading of the plugin shared by `ref_ernie55.py` and `ernie5.rs` would not show. The llama.cpp
pull request for the model ([ggml-org/llama.cpp#31925](https://github.com/ggml-org/llama.cpp/pull/31925),
a third party's, not merged) describes the same computation; it was read, not run.

## Verification

`tools/ernie55_tiny.py` builds a tiny random checkpoint in the release's layout with every
feature above: 8 layers of 256 (HCA layers of rate 8 and MLA layers alternating, a window of 16,
an indexer top-k of 24 so that most queries are sparse), one dense layer and latent MoE layers of
8 experts, YaRN, the MultiMax head, the MTP layer, an RMS epsilon of 0.01 (so that the two RMS
forms of the hyper-connections give different numbers), bf16 or (`--fp8`) FP8 with 128 x 128
block scales. Against `tools/ref_ernie55.py` (logits of standard deviation 3.3):

| Checkpoint, prompt | Pass | Max logit diff | Mean KL | Same top token |
|---|---|---:|---:|---:|
| bf16, 300 tokens | prefill in one pass | 8.9e-4 | 8.2e-10 | 300/300 |
| | 16 greedy steps | 4.6e-4 | 7.8e-10 | 16/16 |
| | MTP layer over the prompt | 2.1e-3 | 3.6e-9 | 299/299 |
| | 3 chained MTP drafts | 1.2e-3 | 1.2e-8 | 3/3 |
| bf16, 700 tokens | prefill in one pass | 1.1e-3 | 8.5e-10 | 700/700 |
| | prefill one token at a time | 8.0e-4 | 5.6e-10 | 700/700 |
| | 24 greedy steps | 3.3e-4 | 5.4e-10 | 24/24 |
| | MTP layer over the prompt | 4.1e-3 | 4.2e-9 | 699/699 |
| FP8, 300 tokens | prefill in one pass | 1.0e-2 | 5.0e-9 | 300/300 |
| | 16 greedy steps | 4.6e-4 | 8.4e-10 | 16/16 |
| | MTP layer over the prompt | 7.1e-3 | 8.2e-9 | 299/299 |

```sh
python tools/ernie55_tiny.py tiny REPO            # REPO: tokenizer.json of the release
python tools/ref_ernie55.py tiny MODEL_PY ref.safetensors 300 16   # MODEL_PY: DeepSeek's inference/model.py
cargo run --release --example deepseek4_check -- tiny ref.safetensors [--chunk N] [--mtp N]
```

Two things the comparison showed. The indexer's scores tie at zero for the keys no index head
scores above zero, and a top-k that cuts through such a tie has to break it the same way on both
sides: oxinfer keeps the older token, as a stable top-k does (the plugin calls a "stable top-k"
kernel), and the reference script sorts stably. And on the FP8 tiny checkpoint the reference
itself gives a different result at one position of 300 when it runs the prompt one token at a
time rather than in one pass (a near tie decided by rounding); oxinfer one token at a time
matches the reference one token at a time, and in one pass the reference in one pass.

On the real files, from their headers only: the loader maps every tensor of the FP8 release
(134231 tensors in 128 shards, 288.4 GB) and of the BF16 release (67692 tensors, 572.9 GB), and
`oxinfer --models` lists both. The released `tokenizer.json` gives the same ids as the tokenizers
library on English, Chinese, Japanese, code, special tokens and numbers (it splits digits into
groups of three from the right with `\p{Nd}{3}+\z`, where `+` after an interval repeats it, as in
Oniguruma; oxinfer's regex engine read a possessive interval there, as fancy-regex does, and
grouped from the left until this was changed).

## Checkpoints

Baidu's naming (`model.layers.N.self_attn.linear_q_down_proj.weight`, ...), read under V4-style
names as the plugin's mapper renames them. Both releases are shards without a
`model.safetensors.index.json`: the shards of a folder are found by name, and
`--model baidu/ERNIE-5.5-Flash-Base` takes every shard the repository lists (that download was
not tried). The FP8 release's matrices
(`F8_E4M3` with a `weight_scale_inv` per 128 x 128 block) are used as stored; the grouped output
projection is split along its scale blocks into one matrix per group.

## MTP drafts

For position `i`, the MTP layer takes the four residual copies before the final collapse and the
embedding of token `i + 1` (`h_proj(hnorm(h)) + e_proj(enorm(e))` on every copy), runs one MLA
layer with its own cache at position `i`, its own collapse and norm, then the model's head and
MultiMax: the prediction of token `i + 2`. A further draft feeds the layer its own residual and
its last draft, one position later (the vLLM scheme, as for Qwen's MTP layer).

`deepseek4::ErnieMtpDecoder` decodes greedily with these drafts: each round the layer proposes a
few tokens after the next one, the model runs them in one pass and keeps the ones it would have
chosen, so the output is the model's own greedy output. On the tiny checkpoints the output is
token for token plain greedy decoding (80 tokens after a 700-token prompt, 5 drafts a round),
but a random MTP layer predicts nothing (0 of 395 drafts kept), so the path that keeps drafts was
only exercised on a degenerate checkpoint whose two heads always give the same token (every
draft kept, same output). Whether the released layer's drafts are accepted, and how often, is
unknown. The decoder is in the library only: the CLI and the server decode plainly (`--mtp` is
refused for this model).

## Not done

- A run on the real weights (text quality, speed, memory, the acceptance of MTP drafts).
- Baidu's code itself as the reference (see above), and the llama.cpp pull request as a second
  one.
- GGUF files: the `ernie5_5` architecture of the llama.cpp pull request is not read (its only
  conversion is gated, and the layout is not final).
- A chat template: the base model has none. The instruct model the plugin announces
  (`ERNIE-5.5-Flash`, with a Python chat protocol, reasoning and tool calls) is not released.
- MTP drafts from the CLI and the server, sampling with them, and more than one MTP layer.
- DSpark drafts (the plugin has a path for them; no draft is released), GPU decoding (prefill
  matrices only, as for DeepSeek V4), weight streaming of the MTP layer.
- Fast prefill of long prompts in the MLA layers: a pass of 32 tokens or more scores every
  cached token for each block of queries before the indexer's choice masks them (decoding reads
  only the chosen ones). Everything was run on the CPU only.
- Rewinding a session by more than 64 tokens (as for DeepSeek V4: the compressors of the HCA
  layers keep the rows of the last windows only).
