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
family; prefill only), the vision variant (`DeepSeek-V4-Flash-Vision-Exp`) and DeepSeek V4.1
(`deepseek_v41`, a different architecture). DeepSeek-V4-Pro (1.6T, 49B active) has the same
architecture and config keys and should load as GGUF, but was not tried (its smallest GGUFs are
far beyond this machine's free disk).
