# DeepSeek V4 Flash

`deepseek-ai/DeepSeek-V4-Flash` (`deepseek4.rs`; 284B parameters, 13B active, 43 layers of 4096)
follows DeepSeek's own inference code (`inference/model.py` of the checkpoint):

- Manifold-constrained hyper-connections: the residual stream is 4 copies of the hidden state.
  Before the attention and before the MoE, a projection of the RMS-scaled copies gives per-token
  weights that collapse the copies into the block's input (`pre`), spread its output back over
  them (`post`) and mix them (`comb`, a softmax made doubly stochastic by 20 Sinkhorn steps); a
  last collapse (`hc_head`) feeds the final norm.
- Attention with one shared 512-wide key-value head (the keys are the values), queries through a
  LoRA (1024) and a weightless RMS norm per head (64 heads), RoPE on the last 64 dimensions undone
  on the output at the query's position, a learned sink logit per head, and a grouped output
  projection (8 groups of 8 heads to 1024 each, then mixed). Every layer sees a window of the last
  128 raw keys; most layers add pooled keys: every 4 tokens (overlapping windows of 8, "CSA",
  with a lightning indexer of 64 heads choosing the 512 best per query) or every 128 tokens
  ("HCA", all of them), pooled by a per-channel softmax over learned gates, RMS-normed and
  rotated (YaRN, theta 160000). A pooled key becomes visible once its window is complete.
- MoE on every layer: 256 SiLU experts with clamped inputs (`swiglu_limit` 10) scored by
  `sqrt(softplus(logit))`, top 6 by score plus a selection bias, renormalized and scaled by 1.5;
  the first 3 layers take their experts from a token-id table instead (hash routing); one
  shared expert.

The reference's quantization-aware-training simulations (FP8 rounding of the cached keys, FP4 of
the indexer, the Hadamard rotation in front of them) are left out, as transformers and llama.cpp
do. Decoding keeps per layer the last 128 + 64 raw keys (the 64 for speculative rollback), the
pooled keys and the rows of the windows still pooling: about 14 KB per token of context (f32)
instead of the 88 KB a full key cache would take.

Checkpoints: llama.cpp GGUF files (`deepseek4`, split or not; unsloth's are tested, including
their two MXFP4 tensors: MXFP4 is a GGUF type here now, with NEON and Metal kernels) and float
safetensors in DeepSeek's naming (`layers.N.attn.wq_a.weight`, as the tiny test checkpoints).
The official release's FP8 / FP4 safetensors are not read (no zero-copy layout for their packed
FP4 experts yet): use a GGUF. The chat format is DeepSeek's `encoding_dsv4.py` (byte-identical
prompts, checked with tools and in both modes): `<｜begin▁of▁sentence｜>` and the system prompt,
`<｜User｜>...<｜Assistant｜>` then `</think>` (chat) or `<think>` (`--think`, the reasoning shown
dimmed), DSML tool calls (`<｜DSML｜tool_calls>` blocks, `--tools`) with results returned as
`<tool_result>` blocks of a user turn.

[Runtime interventions](interventions.md#runtime-interventions-neuron-edits-steering-abliteration-sae-features)
work on this family: neuron edits of the shared and routed experts, steering, ablation, SAE
features, capture and `--make-direction`. The residual stream is the 4 hyper-connection copies,
and "the residual at layer N" is defined as their mean (what the DSpark draft reads too): hooks
read the mean and write to every copy. That section gives the exact hook points and the
agreement with PyTorch hooks, measured on the tiny random checkpoint (the real model was not
run with interventions).

```sh
# unsloth's 97 GB UD-Q2_K_XL on a 128 GB Mac (streamed: more than 60% of RAM)
./target/release/oxinfer --model unsloth/DeepSeek-V4-Flash-GGUF:UD-Q2_K_XL -p "Why is the sky blue?"
./target/release/oxinfer --model ~/.cache/oxinfer/hub/unsloth--DeepSeek-V4-Flash-GGUF/UD-Q2_K_XL --think \
    -p "How many prime numbers are there below 100?"
./target/release/oxinfer --model ~/.cache/oxinfer/hub/unsloth--DeepSeek-V4-Flash-GGUF/UD-Q2_K_XL --tools \
    -p "Use the calculator: what is 1234.5 * 6789 - 2^20?"
```

As Mistral Large 4, a checkpoint larger than 60% of RAM streams by default with a page-cache
budget of 70% of RAM (`--no-stream` / `stream=off` for plain demand paging). For UD-Q2_K_XL the
6.5 GB of dense weights stay resident and the 90 GB of routed experts are read as they are
routed; a decoded token reads 8.6 GB (5.6 GB of it attention weights, Q8_0 in this quant). Prefill
runs the expert GEMMs on the GPU's matrix units over the experts the chunk routes to (wrapped per
call, nothing pinned).

Agreement. DeepSeek's `inference/model.py`, run in float32 with its TileLang kernels replaced by
PyTorch (`tools/ref_deepseek4.py`), on tiny random checkpoints in DeepSeek's layout with every
feature (hash layers, window-only, CSA with a top-8 indexer so most queries are sparse, HCA with
a rate of 16 so a 300-token prompt has many pooled keys, a 16-token window, YaRN, clamped experts;
`tools/deepseek4_tiny.py`, compared by `examples/deepseek4_check.rs`):

| check (bf16 weights) | prefill: max logit diff / mean KL | 16 greedy decode steps |
|---|---|---|
| 60 tokens | 5.7e-5 / 2.9e-11 (60/60 top-1) | 3.8e-5 / 1.6e-11 |
| 300 tokens | 7.7e-5 / 3.0e-11 (300/300) | 3.0e-5 / 1.2e-11 |
| 300 tokens in chunks of 7, of 64 | 6.0e-5, 7.7e-5 | 3.1e-5, 3.0e-5 |

(The reference's own prefill and token-by-token decode agree only when the indexer has no exact
ties: with 4 indexer heads, keys whose every head score is negative tie at 0 and `topk` breaks
the tie differently for different lengths, so the tiny checkpoints use 16.) transformers'
`DeepseekV4ForCausalLM` gives the same prefill logits as the reference (5e-5).

On the real UD-Q2_K_XL against llama.cpp (its master, CPU) on the same file: the same next token
on every prompt tried, top-20 overlap 18/20, 17/20 and 20/20 (a 2550-token prompt, where the
indexer chooses), log-probabilities of the top 5 within 0.31-0.39; greedy continuations identical
for 48/48 tokens (a chat answer with code), 32/32 (a 1978-token prompt) and 31/32 (2550 tokens),
and for 30 tokens of a 7-token raw prompt before a near-tie ("twin brothers who were raised" /
"twin brothers raised"). Layer 0's attention matches llama.cpp to 4 digits per head; the two
quantize the activations to int8 differently, which moves the results a little more than on
Mistral Large 4.

Speed on the M5 Max (128 GB; other jobs kept the load average at 9-20 throughout, so these vary
by +-20%), UD-Q2_K_XL, streaming with the default budget:

| | prefill | decode |
|---|---|---|
| oxinfer, Metal | 64 tok/s (2550-token prompt), 5.5 tok/s (15 tokens) | 6-10 tok/s plain, 8-10 with DSpark; 8-10 after a 16k-token prompt |
| llama.cpp master, CPU (`-ngl 0`, mmap), same file | 47 tok/s (2550 tokens, warm) | 1.6-2.2 tok/s |

Decoding skips work whose result is known (lossless, bit-identical logits): a pooled key is
computed only when a token completes its window (3 tokens in 4 add only their projected rows, as
DeepSeek's code does), and the lightning indexer's query projection and scoring run only once a
query sees more than 512 pooled keys (about 2048 tokens; before that every key is selected). The
indexer's scoring of one query is spread over all threads (it was one task per query, on one
thread), and attention takes the shared-row kernel described under
[Performance](performance.md#performance): the 64 heads share one 512-wide key-value head, so each key is read
once per phase instead of once per head. Before / after on the same file: 8.1-9.3 / 9.9-10.0
tok/s after an 857-token prompt, 6.5-7.9 / 8.3-8.8 after 4191, 5.4-6.5 / 8.2-10.3 after 16076.

## DSpark speculative decoding for DeepSeek V4

`deepseek-ai/DeepSeek-V4-Flash-DSpark` is the same model with three DSpark stages (`mtp.0..2`,
shards 46-48 of the repository, 10.9 GB). llama.cpp's converter turns them into a `dflash` GGUF
(`convert_hf_to_gguf.py DIR --dspark --target-model-dir DIR`, with a folder holding only those
shards; MXFP4 experts, Q8_0 for the FP8 weights, 10.9 GB); named `dspark*.gguf` beside the model's
folder it is used by default, as for Mistral Large 4. The draft (`dspark.rs` with
`deepseek4::Ds4Draft`):

- Context: the target's residual (mean of its 4 copies) after layers 40, 41 and 42, concatenated,
  projected and RMS-normed; each stage turns it into a key of its own 128-position window.
- Block: the last committed token and 4 noise tokens (`dspark_noise_token_id`) at the next
  positions, through three full DeepSeek V4 blocks (hyper-connections, MoE with 256 experts,
  window-only attention to the context window and to the whole block, without a mask), the
  draft's own `hc_head` and norm, the target's head; slot `i` predicts the token `i + 1` after
  the anchor. The block always holds its 5 slots (the attention has no mask), whatever the number
  of drafts used.
- A Markov head adds `markov_w2 markov_w1[previous token]` to each slot's logits, chained on the
  drafted tokens.

Agreement: `tools/ref_dspark_ds4.py` runs the stages of DeepSeek's DSpark `inference/model.py`
in float32 (FP8 / FP4 weights dequantized as used) on the target features oxinfer dumps
(`examples/dspark_check.rs`, which takes DeepSeek V4 too). A tiny random draft
(`deepseek4_tiny.py --dspark`, written as GGUF by `tools/dspark_ds4_gguf.py`) matches to 8e-7
relative on all 5 slots (KL 1e-11); the real converted draft on the real target's features to
6-10e-3 relative (Q8_0 requantization of the FP8 weights, int8 activations), KL under 2e-3, the
same 5 drafts. Greedy output with DSpark is the plain greedy output (128 tokens, checked;
verification passes are bit-identical to one-token forwards, `OXINFER_CHECK_SPEC=1`).

Acceptance and speed (greedy, 128 tokens, `--no-system --threads 12`, three alternated rounds
each, under the load above; tokens per verification pass and drafts accepted in parentheses):

| prompt | plain | 3 drafts, cutoff 0.85 (default) | 3 drafts, no cutoff | 5 drafts, no cutoff |
|---|---|---|---|---|
| "why is the sky blue" | 8.0 tok/s | 8.6 (2.35, 87%) | 8.7 (2.89, 64%) | 7.7 (3.34, 49%) |
| a Python function | 7.1 | 8.3 (3.02, 96%) | 7.7 (3.53, 88%) | 8.3 (4.54, 75%) |

A draft pass takes 15-35 ms. A verification pass of 4 tokens costs about 2.5 times a plain step
(320 ms against 125): the dense attention weights are read once, but every token adds its routed
experts, and with this quant's experts only partly in the page cache (25-30% of the time is spent
waiting on storage) those reads dominate, as on Mistral Large 4. Full blocks of 5 accept more per
pass but lose. `--spec-draft-n-max` (at most 5), `--spec-draft-p-min`, `--no-draft` and the engine /
C options `draft`, `draft_tokens`, `draft_p_min` work as for Mistral Large 4.

Not done: the official FP8 / FP4 safetensors (above), the model's own MTP layer
(`num_nextn_predict_layers`; DSpark replaces it), GPU decoding (no whole-pass Metal graph for this
family; prefill only) and the vision variant (`DeepSeek-V4-Flash-Vision-Exp`). DeepSeek-V4-Pro
(1.6T, 49B active) has the same architecture and config keys and should load as GGUF, but was not
tried (its smallest GGUFs are far beyond this machine's free disk).

## DeepSeek V4.1 Flash

`deepseek-ai/DeepSeek-V4.1-Flash` and `-1017` (`deepseek_v41`; 40 layers of 5120, 384 experts of
which 6 per token, 510 GB as FP8 with FP4 experts, 197 GB of that two engram tables) run on the
DeepSeek V4 code (`deepseek41.rs`, a submodule of `deepseek4.rs`), as DeepSeek's own
`inference/model.py` and `engram.py`. Text only: the vision tower is not implemented. **The real
weights were never run**: every conversion needs more than the 150 GB this work could download
(the smallest, with its engram tables, is 183 GB), so what follows was measured on tiny random
checkpoints and on the headers of the released files. What differs from V4:

- Hyper-connections in one pass: each sublayer computes its `pre` / `post` / `comb` weights from
  the copies entering it but collapses them with the `pre` of the sublayer before it (the first
  attention reads copy 0), and the final norm takes the copies collapsed with the last MoE's
  `pre`. There is no `hc_head`.
- No norm on the query heads; RMS norms with an epsilon of 1e-20.
- Compressed keys are shared between layers. Layers 2, 8, 14 and 20 (`kv_source_layer_ids`)
  compress their own input: at rate 2 (layers 2-19) a softmax over learned gates pools each pair
  of tokens, without V4's position bias or overlap; at rate 1 (layers 20-39) every token gets a
  second key, a projection and a norm. The layers after a key source attend to its keys.
- The lightning indexers are shared too: layers 2, 8, 14, 20, 24, 28, 32 and 36
  (`index_source_layer_ids`; 32 heads of 128, top 512) score the compressed keys, with index
  keys the key source makes from its latents, and the layers down to the next index layer reuse
  the choice. Layer 20 also keeps, per query, the 2048 best blocks of 8 compressed positions (a
  block scores as its best position; the newest block is always kept), and the index layers
  after it choose inside those blocks only.
- Engram: before layers 1 and 14, the 2-grams, 3-grams and 4-grams ending at each token (over
  compressed token ids, a table that folds case, accents and spacing: 99092 ids for the 129280
  tokens) are hashed by 8 heads each into a table of 384M rows of 256; the 24 rows, projected,
  give one key per residual copy and a value, and each copy gets the value times
  `sigmoid(sign(d) sqrt(|d|))`, `d` the normalized product of copy and key. A decoded token
  reads 48 rows of the tables (they are mapped, never loaded).
- MoE: no hash-routed layers; the weights are renormalized with `norm_topk_prob`; the tokens of
  an image span are selected with their own bias (`bias_vl`), take no engram lookup and stop the
  n-grams of the tokens after them.

As for V4, the reference's quantization simulations are left out (here: FP8 of the window keys
and of the attention output, FP4 of the compressed keys with a scale per 16, FP4 of the indexer);
whether llama.cpp leaves them out for this architecture was not checked.

Checkpoints: GGUF files (`deepseek41`), and float safetensors in DeepSeek's naming (the tiny test
checkpoints; the release's FP8 / FP4 tensors are not read, as for V4). Two GGUF layouts exist
and both are read:

- llama.cpp's (unsloth, ggml-org, ISTA-DASLab): the two engram tables as one
  `per_layer_token_embd` (IQ4_NL, 111 GB, a part of its own), the hash inputs in the header
  (`ple.layer_multipliers`, `ple.head_vocab_sizes`, `ple.head_offsets`, `engram.token_map`), the
  product of the gate's two weights as `blk.N.engram_gate`. ISTA-DASLab's builds count the
  table file among their parts without naming it as one
  (`DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf`, at the repository's root): it is found beside the
  parts or one folder up, by its header, but `org/repo:QUANT` does not fetch it.
- AtomicChat's: the reference's hash buffers as tensors (`engram_token_map`, `engram_primes`,
  `engram_offsets`, `engram_multipliers`), `engram_q` / `engram_k` apart, the config's own key
  names, and the table file named by `deepseek41.engram.file` (the same ISTA-DASLab file, not in
  AtomicChat's repository): it is read from beside the model or one folder up.

The chat format is DeepSeek's `encoding/encoding.py` (`chat::DeepSeek41Chat`): V4's with a
`<｜System｜>` token in front of the system part, in thinking mode a first line `Reasoning Effort:
75 (range 1-100, ...)` (the encoder's default, "high"; not adjustable here), and DSML tags whose
names start with a space (`<｜DSML｜ calls>`, ` invoke`, ` parameter`; forced tool calls included).
The rendered prompts are byte-identical to the encoder's for chat and thinking mode, tools with
and without a system prompt, and a round of tool calling (a unit test holds them). Tool
namespaces, mid-conversation system messages, `latest_reminder` and task tokens are not
rendered.

Agreement with DeepSeek's code, run in float32 with its TileLang kernels replaced by PyTorch and
the engram table read as floats (`tools/ref_deepseek41.py`), on a tiny random checkpoint in the
official layout with every feature of the text model (`tools/deepseek41_tiny.py`: 8 layers at
rates 0, 2, 2, 2, 1, 1, 1, 0; key sources at both rates; four index layers, one of them on
another layer's keys; top 8 of up to 300 compressed keys; candidate blocks, 6 of 4, restricting
the last index layer; engram before two layers with 3-grams and the real tokenizer's compressed
ids; a 16-token window; YaRN; clamped experts), compared by `examples/deepseek4_check.rs`:

| check (bf16 weights) | prefill: max logit diff / mean KL | 16 greedy decode steps |
|---|---|---|
| 60 tokens | 8.5e-5 / 5.2e-11 (60/60 top-1) | 8.0e-5 / 3.2e-11 (16/16 top-1) |
| 300 tokens | 8.8e-5 / 5.0e-11 (300/300) | 5.4e-5 / 3.5e-11 (16/16) |
| 300 tokens in passes of 1, 7, 64 | 7.6e-5, 7.6e-5, 8.8e-5 (300/300) | 5.2e-5, 5.2e-5, 5.4e-5 (16/16) |
| 300 tokens, the reference itself prefilling 7 or 64 tokens then stepping | 9.0e-5, 8.3e-5 (300/300) | 4.8e-5, 4.6e-5 (16/16) |
| 300 tokens with an image span of 10 positions (the reference's vision tower embeds a 126x84 picture; oxinfer is given those embeddings) | 1.3e-4 / 5.4e-11 (300/300) | 4.7e-5 / 2.8e-11 (16/16) |
| the same checkpoint as GGUF in each released layout (`tools/deepseek41_gguf.py`: llama.cpp's, AtomicChat's, ISTA's unnamed table part) | 8.8e-5 (300/300), all three | 5.4e-5 (16/16) |

The same top token at every step means the greedy continuations are identical. A unit test
without a reference (`deepseek4::v41::tests::tiny_model`) holds that the logits do not depend on
how the tokens are split into passes, that a rollback leaves no trace, and that the indexer's
choice, the candidate blocks, the engram lookups and an image position each change the logits.

On the released files, without their tensor data (sparse local files with the real headers,
`deepseek4_check FILE --load`): the loader maps all 1043 tensors of
unsloth/DeepSeek-V4.1-Flash-1017-GGUF UD-IQ1_S and of ISTA-DASLab's GSQ-RCO-Coder IQ2_XXS (192
of the 384 experts kept: the expert count comes from the header), and the 1008 + 1 of
AtomicChat's IQ1_M (with the 518 KB of its hash buffers fetched, which equal the other layout's
header values); the tokenizer rebuilt from the GGUF metadata gives the ids of the official
`tokenizer.json` on two test texts (special tokens, accents, CJK, code).

```sh
# every part in one folder (233 GB with the engram tables)
./target/release/oxinfer --model unsloth/DeepSeek-V4.1-Flash-1017-GGUF:UD-IQ1_S -p "Why is the sky blue?"
# ISTA-DASLab / AtomicChat: fetch the parts and DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf by hand, then
./target/release/oxinfer --model DIR/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00001-of-00004.gguf --think -p "..."
```

Not done for V4.1: a run on the real weights (so no speed, no comparison with llama.cpp, and the
assumption that `blk.N.engram_gate` is the product `q_weight * k_weight` is untested: the tiny
GGUFs are written with that reading); image input (the ViT with 2D RoPE, the 3x3 aligner and the
image processor; `mmproj` files are not read; the language model's side is in place and
verified as above); DSpark (the `mtp.*` stages differ from V4's: 128 experts, a Markov head on the
target's embedding and head, a confidence head; a `dspark*.gguf` beside the model is ignored and
`--draft` refused); fetching ISTA-DASLab's table file with `org/repo:QUANT`; the FP8 / FP4
release; GPU decoding. Runtime interventions use V4's hook points and were not measured on this
architecture.
