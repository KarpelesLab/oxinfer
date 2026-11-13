# Raon-1 nano

`KRAFTON/Raon-1-nano` (`raon1_nano`, `Raon1NanoForCausalLM`; 24 layers of 1536, 3.1B parameters
of which 1.5B are a per-layer embedding table, 7.1 GB of bf16 safetensors) runs on `raon.rs`.
Text, thinking and tool calls; Korean and English. Verified on the released weights against
KRAFTON's own modeling code.

## The architecture

The repository ships its modeling code (`modeling_raon1_nano.py`), and `raon.rs` follows it.

- No positions anywhere: order comes from the recurrent layers.
- Four residual streams ("Raon_mHC"). The embedding is copied to each. Every sublayer (attention,
  MLP) has a selector that reads the token's four streams side by side, RMS-scaled, through 17
  learned rows: 4 give `h_pre = sigmoid(..)`, the weights of the streams in the sublayer's input
  (which is then RMS-normed); 4 give `h_post = 2 sigmoid(..)`, how much of the sublayer's output
  each stream receives; 9 give a 3 x 3 matrix `Z` that mixes the streams:
  `H = (1 - gamma) I + gamma Phi`, `gamma = sigmoid(eta)`, `Phi = B + E / (1 + rho)` with
  `B = (I + 11^T / 4) / 2`, `E = Q Z Q^T` (`Q`: the 4 x 3 Helmert basis, so rows and columns of
  `E` sum to zero) and `rho = max(0, max(-E / B))`. `Phi` is non-negative with rows and columns
  summing to one, whatever `Z` is. The model's output is the mean of the streams, normed.
- RDA layers (Raon Delta Attention, 18 of the 24): the delta-rule linear attention of Kimi K3
  (16 heads of 128; `q` / `k` / `v` through a causal depthwise convolution over 4 tokens and
  SiLU; `S *= exp(g)` per key channel, `S += beta k (v - S k)`, `o = S q`; the decay
  `g = -5 sigmoid(exp(A_log) (f_b(f_a(x)) + dt_bias))`; a gated RMS norm of the output) with a
  write gate: the value is scaled by `sigmoid(b + w) / sigmoid(b)`, so it enters the state with
  `sigmoid(b + w)` while the old content is erased with `sigmoid(b)`. Queries and keys are
  normed as `x / max(|x|, 1e-6)`. The state is 1 MB per layer whatever the context length.
- MLA layers (layers 4, 8, .. 24): latent attention with a query LoRA (512) and a 512-wide
  latent, without a rotary part (the 64 "rope" lanes of queries and keys are used as they are),
  and a sigmoid output gate on the heads' values. The latent and the 64 shared key lanes are
  cached (2.3 KB per token and layer).
- A dense SwiGLU MLP (6144) in every layer.
- Per-layer embeddings: a second table gives every token 256 values per layer. A layer's input
  is `p = (RMSNorm(W_l e) + 16 T[token]_l) / sqrt(2)` (`e`: the token's embedding), and after
  the layer's MLP every stream receives `RMSNorm(P (gelu(G m) * p))`, `m` being the mean of the
  streams. The table is half of the checkpoint (3.2 GB); a token reads one row of it.
- The LM head is the embedding table (`model-tied-lm-head.safetensors` repeats it and is not
  read).

The two attention kinds and their caches are Kimi's code (`kimi::kda_forward` with the write
gate, `kimi::mla_forward`); the streams and the per-layer embeddings are `raon.rs`.

## Chat format

`chat::RaonChat`: the template is Qwen3.5's ChatML one (trimmed turns, tools declared in the
system turn, calls as `<tool_call><function=NAME><parameter=KEY>`, results as
`<tool_response>` in a user turn) with one difference: without thinking the reply opens with
`<think></think>` and a blank line. The rendered prompts are byte-identical to the template's
(transformers' jinja rendering) with and without thinking, with a system prompt, with tools, with
a past reply and through rounds of tool calling, nested arguments included (a unit test,
`chat::tests::raon_matches_template`, holds them).

One case is not the template's: a conversation rendered again from its messages
(`Engine::chat`) with thinking on writes a past reply after the `<think>` that the generation
prompt opened, closing it empty (`<think>`, a blank line, `</think>`), where the template writes
`<think></think>` for a reply without reasoning. Past reasoning is not kept. The template's
`truncate_history_thinking` option is not offered.

The tokenizer (`tokenizer.json`, byte-level BPE with a Qwen-style split) gives the ids of the
`tokenizers` library on 5 texts (45,305 tokens: English, Korean, Japanese, Chinese, emoji,
code, the special tokens, this README and the model card).

## Agreement

The reference is `modeling_raon1_nano.py`, unchanged, run in float32 on the CPU
(`tools/ref_raon.py`): the bf16 weights are widened, so both sides compute on the same weights.
`examples/lm_check.rs` compares the logits at every prompt position and after each of the
reference's greedy tokens.

| check | prefill: max logit diff / mean KL | greedy decode steps |
|---|---|---|
| released weights, a 190-token chat prompt (English, Korean, code) | 2.7e-4 / 1.1e-10 (190/190 top-1) | 6.9e-5 / 1.2e-11 (48/48) |
| the same in passes of 64 tokens (both sides) | 2.7e-4 / 1.1e-10 (190/190) | 6.9e-5 / 1.2e-11 (48/48) |
| the same in passes of 1 token (oxinfer) | 1.0e-4 / 1.0e-11 (190/190) | 5.6e-5 / 9.4e-12 (48/48) |
| tiny random checkpoint (`tools/raon_tiny.py`), 96 tokens | 3.6e-4 / 1.5e-11 (96/96) | 2.4e-5 / 4.6e-15 (16/16) |
| tiny, passes of 7 (both sides), then of 7 and 1 (oxinfer) | 3.4e-4, 2.5e-4, 2.5e-4 (96/96) | 3.6e-5, 5.5e-5, 5.5e-5 (16/16) |

Logits reach 43 to 51 on the released weights. The same top token at every step means the
greedy continuations are identical: 48 of 48 tokens. The tiny checkpoint (6 layers of 64, RDA
with 3 heads of 16 and a decay rank of 12, MLA layers 4 and 6, 4 streams, per-layer embeddings
of 8) has every feature of the model.

Quantized at load on the released weights (same prompt, against the float32 reference):
`--quant q8` gives a mean KL of 2.3e-3 over the prefill (187/190 top tokens, max logit diff 1.6)
and 1.7e-3 over the decode steps (47/48); `--quant q4` gives 0.39 (151/190, max logit diff 28)
and 0.12 (43/48). The 4-bit loss is spread over the model (keeping the MLPs, or the attention
layers, or the embedding in bf16 each recovers a part: KL 0.22, 0.27, 0.37): for a model this
small, 4-bit weights rounded at load are not a good trade, and bf16 (7.1 GB) is the format to
use.

A unit test without a reference (`raon::tests::tiny_model`) holds that the logits do not depend
on how the tokens are split into passes, that a rollback inside a recorded pass leaves no trace,
that the mixing matrices are doubly stochastic, and that the write-back factor, the selector's
epsilon, the MLA output gate, the per-layer embeddings and the latent norm's epsilon each change
the logits.

```sh
./target/release/oxinfer --model KRAFTON/Raon-1-nano --think -p "Why is the sky blue?"
./target/release/oxinfer --model KRAFTON/Raon-1-nano --tools -p "What is 1234 * 5678?"
# the checks above (CODE_DIR: the two .py files of the repository)
python tools/ref_raon.py MODEL_DIR CODE_DIR ref.safetensors 400 48 0 --text prompt.txt
cargo run --release --example lm_check -- MODEL_DIR ref.safetensors --chunk 64
python tools/raon_tiny.py tiny CODE_DIR && python tools/ref_raon.py tiny CODE_DIR ref.safetensors 96 16
```

## Speed

On 64 x86 cores (AVX-512), bf16 weights in memory: prefill 230 to 270 tokens/s, decoding 17 to 21
tokens/s. A decoded token reads 3.06 GB of weights (the head, tied to the 259,840-row embedding,
is 0.8 GB of it), which is the memory bandwidth of the machine (62 GB/s measured in the matmuls).
With an RTX 5080 the figures are the same: only the matmuls of long passes go to the GPU.
Prompt-lookup rounds run on it (a rejected round replays the delta rule over the kept tokens
from the recorded inputs): on a tool-calling prompt, 27 tokens in 16 forwards.

## Not done

- The auxiliary multi-token prediction layer (`mtp/model.safetensors`, one MLA layer of 55M
  parameters): not read, so no MTP speculative decoding (`--mtp` is refused). The modeling code
  itself only gives a reference forward for it, not a decoding loop.
- GGUF, MLX and LiteRT-LM conversions of the model: not read (the safetensors only).
- GPU decoding: no whole-pass graph for this family; the GPU, when present, runs the matmuls of
  long passes.
- The model card's recommended sampling for chat is temperature 0.1 with a presence penalty of
  1.5. oxinfer has a repetition penalty, not a presence penalty: that setting cannot be
  reproduced.
- The model card recommends thinking mode; as for the other families, thinking is off unless
  `--think` is given.
- Left-padded batches (the modeling code masks padded rows out of the recurrence): there are no
  padded batches here.
- Session snapshots, and reusing the cache of an earlier prompt that shares only a prefix (the
  recurrent states roll back inside the last recorded pass only).
- Runtime interventions act on the four streams as on DeepSeek V4's residual copies
  (`State::point_copies`; captures read the mean of the streams, which is what the modeling code
  returns as hidden states) and on the MLP's neurons; they were not measured on this
  architecture.
