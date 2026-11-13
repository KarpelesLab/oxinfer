# GLM-5.3 and GLM-5.3-Flash

Two architectures share the name, and `glm5.rs` runs both (text only):

| | GLM-5.3-Flash (`glm5_next`, llama.cpp `glm5-next`) | GLM-5.3 (`glm_moe_dsa`, llama.cpp `glm-dsa`) |
|---|---|---|
| checkpoints | zai-org/GLM-5.3-Flash, nex-agi/Nex-N2.6 (a fine-tune), their GGUFs | zai-org/GLM-5.3, unsloth/GLM-5.3-GGUF, Scantist/GLM-5.3-SLIM-GGUF (192 of the 256 experts) |
| size | 313B parameters, 17B active; 45 layers of 4096 | 754B parameters; 78 layers of 6144 |
| residual stream | 4 hyper-connection copies | one |
| attention | 34 KDA linear-attention layers, 11 MLA layers without positions | MLA with RoPE on every layer |
| DSA indexer | scores pools of 4 tokens, on every MLA layer | scores single tokens, on 21 layers (the others reuse the last choice) |
| MLP | 3 dense layers, then 288 experts (8 routed + 1 shared), clamped SwiGLU | 3 dense layers, then 256 experts (8 routed + 1 shared) |

GLM-5.3-Flash, as transformers' `Glm5NextTextModel`:

- Manifold-constrained hyper-connections: the residual stream is 4 copies of the hidden state;
  before the attention and before the MLP a projection of the RMS-scaled copies gives the weights
  that collapse them into the block's input, spread its output back and mix them (a softmax made
  doubly stochastic by 20 Sinkhorn steps). This is DeepSeek V4's arithmetic and runs on the same
  code (`deepseek4::hc_pre` / `hc_post`); the final collapse is a plain mean of the copies.
- KDA linear attention on 34 of the 45 layers: `q`, `k` and `v` projections (64 heads of 128)
  through a depthwise causal convolution (kernel 4) and SiLU, `q` and `k` L2-normalized; each
  head keeps a 128 x 128 state updated by the delta rule, `S <- S diag(decay)`, `S += k (beta (v -
  S^T k))^T`, with a decay per key channel `exp(-5 sigmoid(exp(A_log) (f_b f_a x + dt_bias)))` and
  a write strength `beta = sigmoid(b x)`; the read `S^T q` is RMS-normed and gated by
  `sigmoid(g_b g_a x)`. No softmax and no cache that grows with the context: 4 MB of state per
  layer, plus the last 3 rows of the convolution's input. (Qwen3.5's Gated DeltaNet has one decay
  per head and no lower bound; this one cannot share its code.)
- MLA attention without positions on every fourth layer: queries through a LoRA (1536, RMS norm),
  keys and values expanded per head (64 heads of 256) from a cached latent of 512. `kv_b` is
  absorbed into the query and the output, so only the latent is cached and scored: 2 KB per token
  and MLA layer instead of 128 KB.
- The DSA indexer of those layers (32 heads of 128) scores pools of 4 consecutive tokens: a pool's
  key is a per-channel softmax (a gate projection plus a learned position bias) over its tokens'
  LayerNorm'd keys, pooled once its fourth token arrives. `sum_h w_h relu(q_h . pool)` ranks the
  complete pools a query sees; the 512 best are attended token by token, plus the tokens of the
  incomplete pool at the query's end. Up to 2048 pooled tokens every pool is taken, so a short
  context is dense causal attention and the indexer's query projection is not computed.
- MoE after three dense layers: a sigmoid router with a selection bias, the top 8 of 288 experts
  renormalized and scaled by 2.5, one shared expert; every SwiGLU clamps its inputs at 10.

GLM-5.3 (transformers' `GlmMoeDsaModel`) has no hyper-connections and no KDA: every layer is MLA
with a rotary part (the last 64 of each query head's 256 dims against one shared rotary key,
interleaved pairs, theta 8e6), the indexer scores single tokens with its first 64 dims rotated
and picks 2048 of them, and a layer without an indexer ("shared") reuses the last choice. Its
SwiGLUs do not clamp, and the RMS norms of the query LoRA and of the latent use an epsilon of
1e-6 where the others use the config's.

Not done: the vision tower of GLM-5.3-Flash (`model.visual.*`, the `mmproj` GGUF files; images
and video are refused), the MTP layer (`layers.45`, the `mtp-*` GGUF files; transformers does not
run it either, so there is no reference), and speculative decoding of any kind: the KDA state
cannot go back to an earlier position, which a rejected draft needs.

## Checkpoints

- The safetensors releases (zai-org/GLM-5.3-Flash and nex-agi/Nex-N2.6, 328 GB; zai-org/GLM-5.3,
  756 GB): FP8 block-quantized matrices (`weight_scale_inv`, 128 x 128 blocks) used as stored,
  bf16 for the rest, one module per expert. `save_pretrained` writes the same names.
- llama.cpp GGUF files, split or not: `glm5-next` (ggml-org/GLM-5.3-Flash-GGUF,
  unsloth/GLM-5.3-Flash-GGUF, bartowski/GLM-5.3-Flash-BF16-GGUF) and `glm-dsa`
  (unsloth/GLM-5.3-GGUF, Scantist/GLM-5.3-SLIM-GGUF). The tensors are named onto the release's
  (`glm5::gguf_name`); `ssm_a` holds `-exp(A_log)` and `kv_b_proj` comes split per head into
  `attn_k_b` (the key rows transposed) and `attn_v_b`, which is the absorbed form the attention
  wants. An MTP block stored after the model's (unsloth) is skipped. The `glm4` pre-tokenizer is
  the Llama 3 split.

A checkpoint larger than 60% of RAM streams by default with a page-cache budget of 70% of RAM, as
Mistral Large 4 does (`--no-stream` for plain demand paging); see
[Streaming weights from flash](streaming.md#streaming-weights-from-flash-models-larger-than-ram).
The loader reports the bytes a decoded token reads: 9.0 GB for GLM-5.3-Flash UD-Q2_K_XL, 13.9 GB
for Q4_K (most of it the KDA and MLA projections, Q8_0 in those files), 22.2 GB for the FP8
release, 21.2 GB for GLM-5.3-SLIM UD-IQ2_M.

On a 64-core x86 machine with 125 GB of RAM (CPU only, the 109 GB UD-Q2_K_XL streamed with an 87
GB page cache): 2.0 to 2.6 tokens/s decoding once the routed experts are cached, 2 tokens/s for
a 22-token prompt and 6.2 for a 352-token one (the tools turn); llama.cpp on the same file and
machine decodes at 1.9 to 2.7 tokens/s (32 threads).

Prompt lookup (on by default elsewhere) is off for GLM-5.3-Flash: it verifies a draft in one
pass and rolls back what the model refutes, and the KDA state cannot go back.

```sh
./target/release/oxinfer --model unsloth/GLM-5.3-Flash-GGUF:UD-Q2_K_XL -p "Why is the sky blue?"
./target/release/oxinfer --model unsloth/GLM-5.3-Flash-GGUF:UD-Q2_K_XL --think -p "How many primes are there below 100?"
./target/release/oxinfer --model unsloth/GLM-5.3-Flash-GGUF:UD-Q2_K_XL --tools -p "Use the calculator: 1234.5 * 6789 - 2^20"
```

## Chat format

The three repos' templates give the same text. A conversation is `[gMASK]<sop>`, a system turn
with the reasoning effort (`<|system|>Reasoning Effort: Max`, the template's default), the tools
in a system turn of their own, then `<|system|>`, `<|user|>`, `<|assistant|>` and
`<|observation|>` turns with nothing between them. A reply ends on the next role token:
`<|user|>`, or `<|observation|>` after tool calls. The template always opens the reply with
`<think>`: `--think` leaves it open (the reasoning is shown dimmed), without it the block is
closed at once (`<think></think>`), which is how the template renders a past reply that has no
reasoning. Tools are declared as JSON lines inside `<tools>` and called with
`<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>` (strings bare, other
values as JSON; a parameter declared as a string stays the string the model wrote), the results
returned as `<tool_response>` blocks of one `<|observation|>` turn.

The template's `reasoning_effort` (Low / High instead of Max) and `clear_thinking` variables are
not exposed.

[Runtime interventions](interventions.md#runtime-interventions-neuron-edits-steering-abliteration-sae-features)
work as for DeepSeek V4: with hyper-connections the residual at a point is the mean of the
copies, hooks read the mean and write to every copy; neurons are the input of `down_proj` of the
dense MLPs, the shared expert and the routed experts. Checked without a reference only (a
steered point changes the logits the same way however the tokens are split into passes).

## Agreement

transformers 5.23 in float32 (CPU, eager attention), on tiny random checkpoints with every
feature on (`tools/glm5_tiny.py`, compared by `examples/glm5_check.rs`): for GLM-5.3-Flash 11
layers with KDA and MLA layers, a "shared" indexer layer, dense and MoE layers, 4 copies, a
SwiGLU clamp of 0.6 so that it clamps, and an `index_topk` of 8 so that almost every query is
sparse; for GLM-5.3 6 layers, two of them shared, RoPE, `index_topk` 24.

| check (bf16-representable weights) | prefill: max logit diff / mean KL | 16 greedy decode steps |
|---|---|---|
| GLM-5.3-Flash, 150 tokens | 9.8e-5 / 2.0e-11 (150/150 top-1) | 3.3e-5 / 1.3e-11 (16/16) |
| 150 tokens in passes of 1, 7, 64 | 9.2e-5, 9.2e-5, 9.8e-5 | 3.3e-5 |
| 600 tokens | 1.7e-4 / 2.5e-11 (600/600) | 3.0e-5 |
| pools of 2, `index_topk` 16 | 1.9e-4 / 3.6e-11 (150/150) | 1.6e-5 |
| the same checkpoints as GGUF (`tools/glm5_gguf.py`) | 9.0e-5 (150 tokens), 1.8e-4 (600) | 3.2e-5, 3.4e-5 |
| GLM-5.3, 150 tokens | 3.3e-6 / 1.3e-13 (150/150) | 2.3e-6 / 1.1e-13 (16/16) |
| 150 tokens in passes of 1, 7, 64 | 2.7e-6, 2.7e-6, 3.3e-6 | 2.4e-6 |
| 600 tokens, `index_topk` 64 | 5.4e-6 / 1.7e-13 (600/600) | 2.9e-6 |
| the same checkpoints as GGUF | 3.3e-6, 5.4e-6 | 2.3e-6, 2.9e-6 |

The greedy tokens are the reference's at every step. That the indexer matters in these
checkpoints: the same weights against a reference with `index_topk` 4096 (dense attention) differ
by 2.0 (GLM-5.3-Flash, 59/150 top-1) and 3.3 (GLM-5.3, 44/150). GLM-5.3-Flash's differences are
larger than GLM-5.3's and grow in its KDA layers (`glm5_check --layers` compares the residual
stream after each layer): the reference runs a prompt through the chunked form of the recurrence
(64 tokens at a time, a triangular solve per chunk) and oxinfer token by token, the same
arithmetic in another order.

llama.cpp (master, commit 9daa0aa, CPU) reads the tiny GLM-5.3-Flash GGUF files too, which makes
`tools/glm5_gguf.py`'s layout the published one: with dense attention (`index_topk` 4096) its
logits are within 1.5e-4 of transformers' over a 150-token prompt's last position and 16 greedy
steps (17/17 top-1). With a sparse indexer (`index_topk` 8) it runs the indexer but does not
follow the reference: 6.4e-2 at the prompt's last position and 6/17 top-1 over the greedy steps,
where oxinfer is within 1e-4 with the reference's tokens. The real model's `index_topk` of 2048
keeps contexts below about 2050 tokens dense. (llama.cpp did not create a context for the tiny
GLM-5.3 GGUF, so that pair was not compared.)

The command line end to end, on a tiny random GLM-5.3-Flash with the release's tokenizer and
chat template (`glm5_tiny.py --tokenizer DIR --chat CONVERSATION.json`): the prompt text is the
template's (system and user turn, thinking), its 28 tokens give last-token logits within 1.4e-5
of the reference's, and 16 greedy tokens decode to the same text; the same model as a GGUF with a
published file's tokenizer metadata gives the same prompt, 1.3e-5 and the same text. The
template's renderings with tools, tool calls and results, and a second turn are unit tests
(`chat::tests::glm_matches_template`).

### The real weights

unsloth/GLM-5.3-Flash-GGUF UD-Q2_K_XL (109 GB; Q8_0, Q5_K, Q6_K, IQ2_XS, IQ3_XXS tensors) runs:
greedy answers are coherent with and without thinking ("The sky is blue because of a phenomenon
called **Rayleigh scattering**. ..."), and `--tools` makes the calculator call
(`calculate({"expression":"1234.5 * 6789 - 2^20"})`), reads the result from the
`<|observation|>` turn and answers 7,332,444.5.

Against llama.cpp (master, commit 9daa0aa, CPU) on the same file, a 22-token prompt and its 24
greedy tokens fed to both (`tools/llama_logits_dump.cpp`, `glm5_check --dump`):

| 25 positions | same top-1 | KL: mean / median / max |
|---|---|---|
| oxinfer, one token per pass, vs llama.cpp | 24 | 9.7e-3 / 8.0e-4 / 6.6e-2 |
| oxinfer, one 46-token pass, vs llama.cpp | 24 | 8.2e-3 / 1.2e-3 / 6.6e-2 |
| oxinfer's two ways against each other | 25 | 4.9e-3 / 2.1e-3 / 3.1e-2 |

This is looser than the other large models' agreement with llama.cpp (1e-4 to 3e-3), and the
one disagreement is the first generated token, where three candidates are close ("The" 0.41, "A"
0.39 here; "A" 0.55, "The" 0.30 in llama.cpp), so the two greedy replies differ from their first
word ("The user is asking a simple, classic science question ..." / "A simple factual question.
Be brief as instructed."). oxinfer's one-token kernels and its multi-token ones differ from each
other by about as much as either differs from llama.cpp, so the gap is the size of the kernels'
own rounding on this file (2- and 3-bit experts under 45 layers of hyper-connection mixing), but
which side is closer to the float model was not measured: no float reference of the real
weights fits this machine.

Of the other published files only the headers were read (metadata and tensor tables, with a hole
in place of the data): ggml-org's Q4_K (2 files, 181 GB), unsloth's UD-Q4_K_XL (6 files, 200 GB,
with its MTP block), Scantist's GLM-5.3-SLIM UD-IQ2_M (6 files, 180 GB) and the FP8 safetensors
release of GLM-5.3-Flash (62 shards, 328 GB) load, every tensor found with the shape the config
implies, and print the template's prompt. A few small tensors of ggml-org's file were compared by
value with the safetensors release (`A_log`, `dt_bias`, a convolution, the hyper-connection
weights, the indexer's pool bias and norm, the router bias, one head of `kv_b_proj`): identical
in the layouts above. GLM-5.3 itself (`glm-dsa`) was not run with real weights: its smallest
quantization is 180 GB.
