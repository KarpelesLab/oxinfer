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
`inference/model.py` and `engram.py`. Text only: the vision tower is not implemented. It was
checked against DeepSeek's code on tiny random checkpoints, and against llama.cpp on the real
weights (ISTA-DASLab's IQ1_M build, 231 GB: [below](#on-the-real-weights)). What differs from V4:

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
  parts or one folder up, by its header, and `org/repo:QUANT` fetches it with the parts (linked
  under the missing part's name, as llama.cpp opens parts by name).
- AtomicChat's: the reference's hash buffers as tensors (`engram_token_map`, `engram_primes`,
  `engram_offsets`, `engram_multipliers`), `engram_q` / `engram_k` apart, the config's own key
  names, and the table file named by `deepseek41.engram.file` (the same ISTA-DASLab file, not in
  AtomicChat's repository): it is read from beside the model or one folder up (also with one
  table per layer, `blk.N.engram_embd`, as that file first had).

How the two correspond is taken from ISTA-DASLab's `convert_ds41_upstream.py`, which rewrites
one into the other: `engram_gate = engram_q * engram_k`, and llama.cpp's head offsets count rows
from the start of the table of all layers (the reference's from the start of each layer's).

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
# ISTA-DASLab: the parts and the engram file (231 GB), as run below
./target/release/oxinfer --model ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF:IQ1_M --think -p "..."
# AtomicChat: its parts and ISTA-DASLab's DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf beside them, then
./target/release/oxinfer --model DIR/FIRST-PART.gguf -p "..."
```

### On the real weights

ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF IQ1_M (120.7 GB of backbone in five parts and
the 110.6 GB engram file), on an M5 Max with 137 GB of memory, Metal, against llama.cpp master
(`deepseek41`, CPU) on the same file. The checkpoint does not fit in memory, so it is streamed:
5.4 GB stays resident, the 115 GB of routed experts are read on demand through a 90 GB page
cache, and the engram rows are mapped. The process's resident set is about 100 GB (mapped page
cache); its own memory (peak footprint) is 6.7 GB at 8k tokens and 9.7 GB at 128k.

- Greedy decoding: on six prompts (chat, thinking, French, code, a 560-token wikitext excerpt),
  the same 33 tokens as llama.cpp at every step (mean KL 0.0002-0.0045, top-20 overlap 19 of
  20). The chat template renders the same bytes as llama.cpp's Jinja template.
- Perplexity, wikitext-2 test, 16 chunks of 512 (llama-perplexity's chunking and scoring,
  `deepseek4_check --ppl`): 2.9751 against llama.cpp's 2.9774, the running value within 0.9% after every chunk;
  the 287,597 token ids of the whole file are identical. (A first run gave 2.6747: the check
  put a BOS in front of each chunk while this GGUF says `add_bos_token = false`, which
  llama-perplexity follows. With a BOS the model predicts this text better; the tool now follows
  the header.)

Long prompts: a passphrase hidden 2% into wikitext (`Read the following text carefully` ... the
text ... `What is the secret passphrase?`), chat template, greedy. "Load" is the machine's load
average at start and end (other jobs shared the machine).

| prompt | prefill | decode | passphrase | peak footprint | load |
|---|---|---|---|---|---|
| 7,666 tokens | 86.6 tok/s (89 s) | 13.9 tok/s | found | 6.8 GB | 10.0 / 8.5 |
| 32,287 | 73.4 tok/s (7.3 min) | 8.7 tok/s | found | 8.2 GB | 3.2 / 16.3 |
| 126,134 | 65.6 tok/s (32 min) | 8.5 tok/s | found | 9.7 GB | 3.0 / 15.7 |
| 1,004,522 | 44 tok/s average over the first 546,816 (3.4 h), then stopped | not reached | not reached | 20.3 GB | 7.2 / 15.9 |

(Before the cache change below; with it, the 32k prompt reads at 79.4 tok/s. The 126k prompt on
the code before the GPU attention and the indexer changes: 37.9 tok/s, 55 min.)

(Decode is that of the 9 tokens of the answer, with prompt-lookup drafting; a short prompt on a
cold page cache decodes at 2.4-5 tok/s, the experts coming from storage. Those figures predate the
round [below](#decode-and-prefill-against-llamacpp).)

Context memory: per position, the window keys of the last 128 positions (a constant), the
compressed keys of the four key-source layers (512 floats; layers 2, 8 and 14 at rate 2, layer
20 at rate 1: 5 KiB per position in f32) and their index keys (128 floats: 1.25 KiB), and the
engram id (4 bytes): **6.25 KiB per position, 6.4 GB at 1,048,576 tokens**, f32 throughout.
llama.cpp allocates a 40 KiB-per-position window cache (f16, 40 layers) for the whole context
plus its compressed caches: about 45 GB at 1M.

So a 1M-token context fits beside the streamed weights (5.4 GB resident + 6.4 GB of context +
the prefill's scratch). Time is the limit: the cost of a prompt token grows slowly with the
context (the indexers score every compressed key; attention itself reads a fixed 512 chosen
keys and the window).

A 1,004,522-token passphrase prompt (the same test, the passphrase 2% in) was started and
stopped by hand after 3.4 hours, at 546,816 tokens (54%), when the machine had to stop: no
crash, but **no recall result at 1M**. Measured on the way (load average 7 at start, 13-23
during the run from other jobs):

| tokens read | time | rate of the last 2048 | peak footprint |
|---|---|---|---|
| 32,768 | 7.1 min | 73 tok/s | |
| 131,072 | 33 min | 57 tok/s | |
| 262,144 | 78 min | 43 tok/s | 12 GB |
| 393,216 | 2.25 h | 34-37 tok/s | 15 GB |
| 524,288 | 3.25 h | 35-39 tok/s | 16 GB |
| 546,816 | 3.42 h | 35.5 tok/s | 20.3 GB (process peak) |

The resident set stayed at ~100 GB (mapped page cache; 101 GB at most). The process's own
memory grew faster than the 6.25 KiB of context per token (about 20 KB per token: growing
buffers and the allocator's free memory; not chased). A straight-line fit of the time per token
against the position (14.1 ms + 0.031 ms per thousand positions over this run; the rate held at
~35-39 tok/s past 350k) puts the whole 1M prompt at about 8.3 hours, the last chunks near
22 tok/s; the 126k run's fit (12.7 + 0.041) gave 10 hours. Decode at 1M was not reached; at 126k
it is 8.5 tok/s.

So: a 1M context allocates and runs on this machine, and its first 547k tokens were read
without trouble, but finding a fact across the full 1M was not measured; the deepest recall
measured is 126k.

Prefill speed, the changes behind those figures:

- The prefill attention reads, per query, its window and the 512 compressed keys its indexer
  chose. On the CPU the scores and outputs of 32 queries are GEMMs over the union of their keys
  (it used to be every compressed key); on Metal (`metal_ds4.rs`), blocks of 256 queries share a
  table of their keys and each head reads only its query's rows (`OXINFER_DS4_GPU_ATTN=0`: CPU).
  Same greedy tokens as the CPU path, mean KL < 1e-4. 32k prompt: 41.6 -> 68.6 tok/s.
- The indexers score blocks of 64 queries against tiles of 8192 keys (the scores held stay
  bounded at any context), and after V4.1's candidate layer a query scores only the keys of its
  2048 candidate blocks instead of all of them.
- Streamed, a prefill chunk of 256 tokens routes to nearly every expert of every layer and reads
  them again (about 50 GB a chunk): a streamed DeepSeek V4 / V4.1 prefills in chunks of at least
  2048 tokens (`deepseek4::STREAM_PREFILL_CHUNK`; `--prefill-chunk` / `prefill_chunk` above it
  are kept). 8k prompt: as fast as 8192-token chunks, with 6.6 GB of peak memory instead of 17.
- Streamed experts are cached in the page cache up to a budget (90 GB here), least recently used
  first. A prefill chunk uses nearly every expert of every layer, chunk after chunk: a cycle
  over 115 GB, in which least-recently-used evicts each expert just before its next use. A pass
  that uses at least half of a layer's experts now only fills free room in the cache (decode
  steps keep least-recently-used). 32k prompt, back to back: 79.4 tok/s against 74.6, 82% of the
  experts needed found cached instead of 51%, 300 GB read from the SSD instead of 717.

Tried and not kept (32k prompt unless said): reading the next layer's 384 experts ahead whole
during a big pass (8k: 82.9 tok/s against 86.6; 32k: 76.4 against 73.4, but 1840 GB read from the
SSD instead of 717: the read-ahead defeats the page cache); 8192-token chunks (75.7 tok/s against
73.4, 441 GB read instead of 717, but 21.6 GB of peak memory instead of 8.2 and a colder cache
after the prefill).

#### Decode and prefill against llama.cpp

The same file, 18 threads both, greedy, `--no-system`, interleaved runs under one lock (load
average 2-12 from other jobs). Decode: 128 tokens after a 20-token prompt ("Write a short story
about a lighthouse keeper..."), without prompt-lookup drafting; prefill: the 7,666-token
passphrase prompt above and a 1,933-token wikitext prompt. "Cold" is a start after the model's
pages were dropped from the page cache (oxinfer's `--cold-start`, which every streamed run used
to do); "warm" a second run. llama.cpp (b11870+, master `225a534`) runs on the CPU (`-ngl 0`) or
with the dense weights on Metal and the experts on the CPU (`-ngl 99 -ot exps=CPU`, its best
setting here: the 231 GB do not fit in its Metal buffers).

| | oxinfer before (master `19e037a`) | oxinfer now, cold | oxinfer now, warm | llama.cpp CPU | llama.cpp Metal + CPU experts |
|---|---|---|---|---|---|
| decode, tok/s | 8.9 (cold: every run was) | 12.2 | 15.5 | 0.32 cold, 0.38 warm | 5.7 cold, 6.0 warm |
| prefill 7,666 tokens, tok/s | 85.3 | 108.0 | 134.0 | | |
| prefill 1,933 tokens, tok/s | | | 125.4 | | 22.5 |

Greedy tokens: the six prompts above give llama.cpp's 33 tokens at every step (198 of 198; mean
KL 0.0004-0.0045, as before); perplexity (16 chunks of 512) 2.9764 against llama.cpp's 2.9774
(2.9751 before: the prefill router's sums are a GEMM's now).

Where a warm decode step goes (64 ms; `--profile` phases, which nest in its categories):

| phase | ms / token | |
|---|---|---|
| routed + shared experts | 33 | CPU decode kernels; the IQ1_S / IQ1_M / IQ2_XXS experts are compute-bound (one core: IQ1_S 7.5 GB/s, IQ1_M 4.9, IQ2_XXS 8.1; all threads ~60-80 GB/s), the Q5_K shared expert reads at ~200 GB/s |
| attention output projection | 10.2 | 52 MB of Q5_K per layer, ~205 GB/s |
| `wq_b` | 5.2 | 29 MB of Q5_K per layer |
| hyper-connection mixes | 2.6 | |
| attention (window + 512 chosen keys) | 2.5 | |
| expert reads (stalls) | 2.2 | 20-30 ms on a cold cache |
| attention input projections | 2.1 | |
| LM head | 1.9 | |
| router | 1.8 | |
| engram (48 rows of the 110 GB table, projection) | 1.2 | 3.8 cold |

Of the 7.09 GB a token reads, 5.4 GB are the resident dense weights (Q5_K, Q8_0, BF16) and ~1.7
GB routed experts. Decode runs on the CPU: the Metal backend has no decode graph for this family,
and a hybrid did not pay (below).

What changed:

- Streaming kept nothing of an earlier run (it dropped the whole model from the page cache at
  start): the experts found cached now fill the part cache (see [streaming](streaming.md)). The
  part cache treats decode and prefill differently already: a decode step's few experts are
  least-recently-used, a prefill chunk (half of a layer's experts or more) only fills free room.
  On a warm start 100% of a decode step's experts are hits; from cold, 90% over the first 128
  tokens.
- The one-token router (384 x 5120 dot products per layer) and the 24 hyper-connection
  projections ran on one thread: 7.1 -> 1.7 ms and 4.9 -> 2.5 ms per token. The output
  projection's 8 groups run as one batch: 11.2 -> 10.1 ms.
- IQ1_S rows (64% of the expert bytes) take the `+-1/8` delta out of the weights (`8 sum(g x) +
  delta sum(x)`, bit for bit the old sums): 6.0 -> 7.5 GB/s per core.
- Prefill: the engram rows of a chunk (~50k scattered pages) are requested all at once (2.1 ->
  0.9 ms per token at 8k), the router is one f32 GEMM per chunk (0.8 -> 0.4 ms per token). The
  expert GEMMs of a 2048-token chunk do run on the GPU (the routed experts wrapped per pass): 2.8
  ms per token at 8k with the reads they wait for (1.0 ms of it stalls when warm), the GPU busy
  16 s of the 66 s.

Tried and not kept: the feed-forward block on the GPU during decode (the shared expert and the
routed experts of a layer in one Metal plan, the rest of the layer on the CPU): same tokens, but
7.9 tok/s against 11.0 on the CPU, because a GPU that idles between short bursts runs them at a
third of its speed (1.0 ms of GPU time per layer instead of 0.3; see
[streaming](streaming.md#gpu-decoding-with-streamed-weights-mixed-gpu--cpu)), and each of the
40 round trips costs ~0.2 ms besides. The dense matvecs alone on the GPU, one submission per
matmul: 4.8 tok/s.

Not done for V4.1: image input (the ViT with 2D RoPE, the 3x3 aligner and the
image processor; `mmproj` files are not read; the language model's side is in place and
verified as above); DSpark (the `mtp.*` stages differ from V4's: 128 experts, a Markov head on the
target's embedding and head, a confidence head; a `dspark*.gguf` beside the model is ignored and
`--draft` refused); the other GGUF builds on their real weights (unsloth's, the Coder and
AtomicChat builds: headers only); the FP8 / FP4 release; GPU decoding (a decode graph of the whole layer; the feed-forward block alone was tried, above). Runtime interventions use V4's hook points and were not measured on this
architecture.
