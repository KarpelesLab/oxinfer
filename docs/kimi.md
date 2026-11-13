# Kimi K3.1 Pro, Kimi K3

`moonshotai/Kimi-K3.1-Pro` (`kimi_k31`; 93 layers of 7168, 896 experts of which 16 per token, 2.8T
parameters, 104B active, 1.56 TB of safetensors with 4-bit experts) and `moonshotai/Kimi-K3`
(`kimi_k3`, the same model with dense attention in place of the sparse one) run on `kimi.rs`. Text
only: the vision tower (MoonViT) is not read. **The real weights were never run**: the smallest
conversion is 580 GB, far above the 100 GB this work could download, so what follows was measured
on tiny random checkpoints, against llama.cpp on those, and on the headers of the released files.

## Where the architecture comes from

Moonshot publishes the modeling code with Kimi K3 (`modeling_kimi_linear.py`,
`KimiLinearForCausalLM`), and that is what `kimi.rs` follows. The K3.1 repository ships no
modeling file: its only change, the sparse attention of the MLA layers, is implemented as
TokenSpeed (one of the three engines its model card names) and llama.cpp do. The two were
written apart and agree; TokenSpeed's source says its semantics "follow DeepSeek-V3.2 / GLM-5
prior art and the tensor shapes", and llama.cpp's implementation was run against oxinfer's
(below). vLLM's and SGLang's code was read for the config only (one choice of keys per layer).

- No positions anywhere: order comes from the recurrent layers.
- KDA layers (Kimi Delta Attention, 69 of the 93): `q` / `k` / `v` projections (96 heads of 128)
  through a causal depthwise convolution over 4 tokens and SiLU, then a gated delta rule per head
  over a 128 x 128 state: `S *= exp(g)` per key channel, `S += (beta k)(v - k^T S)`, `o = q^T S`,
  with `q` and `k` L2-normed, `beta = sigmoid(b_proj(x))` per head and the decay
  `g = -5 sigmoid(exp(A_log) (f_b(f_a(x)) + dt_bias))` per head and key channel. The output goes
  through a gated RMS norm (`norm(o) w sigmoid(g_proj(x))`) and `o_proj`. The state is 6 MB per
  layer whatever the context length.
- MLA layers (every fourth layer and the last, 24): DeepSeek's latent attention with a query
  LoRA (1536) and a 512-wide latent, without a rotary part (the 64 "rope" lanes of queries and
  keys are used as they are), and a sigmoid output gate on the heads' values. Only the latent and
  the 64 shared key lanes are cached (2.3 KB per token and layer, 0.5 KB more for K3.1's index
  key); `kv_b` is absorbed into the query and the output, as for Mistral Large 4.
- Sparse attention (K3.1, `dsa_config`): each MLA layer has a lightning indexer (32 heads of 128),
  `I(t, s) = d^-0.5 H^-0.5 sum_h weights_proj(x_t)_h relu(wq_b(c_t)_h . k_s)` with `c_t` the
  normed query latent and `k_s = LayerNorm(wk(x_s))`, and a query attends to its 2048 best keys
  (to all of them while it has no more). Every layer chooses for itself. Equal scores go to the
  earlier key.
- Cross-layer attention residuals (`attn_res_block_size` 12): the stream entering every twelfth
  layer is banked, and the running sum restarts from that layer's attention output. Each
  sublayer reads a softmax mix of the banked streams and the running sum (scored by their
  RMS-normed values times a learned vector), as does the final norm.
- A dense first layer, then MoE: a sigmoid router (top 16 of score plus bias, renormalized), the
  routed experts in a latent space half as wide (a projection down to 3584, the experts, an RMS
  norm, a projection up), and two shared experts on the full width.
- SiTU in place of SwiGLU: `4 tanh(gate / 4) sigmoid(gate) * 25 tanh(up / 25)`.

Three things the files do not say plainly:

- `A_log` is stored as `[128]` (the head dimension) while Moonshot's code declares it per head
  (`[96]`). The released values are per head: entries 96 to 127 are zero (read from the first
  shard by range request), and SGLang, TokenSpeed and llama.cpp take the first 96. A tiny
  checkpoint on Hugging Face (`tiny-random/kimi-k3.1-pro`) patches the modeling code to read it
  as a decay per channel shared by all heads instead: that is not what the weights hold.
- The index keys' LayerNorm has no epsilon in the config. TokenSpeed uses 1e-6 (DeepSeek
  V3.2's), llama.cpp's converter writes the RMS norm's 1e-5 into the GGUF: oxinfer uses 1e-6 for
  safetensors and the file's value for a GGUF (1e-5 when it names none, llama.cpp's default).
- The model card says the model was trained with MXFP8 activations. The checkpoint's
  quantization config declares none (`input_activations: null`) and no activation is rounded
  here (the 4-bit kernels use int8 activations in short passes, see below).

## Checkpoints

- GGUF (`kimi-k31`, llama.cpp's architecture name, and `kimi-k3` with indexer keys, as
  danielhanchen's preliminary conversion has it): streamed from storage when larger than memory,
  like Mistral Large 4.
  llama.cpp's conversions are undone at load: the cross-layer scores fused (`norm * proj`),
  `-exp(A_log)` per head, `kv_b` split per head, the experts stacked, the two `g_proj` gates
  under different names.
- Moonshot's safetensors (tensors under `language_model.`): the routed experts are
  compressed-tensors `mxfp4-pack-quantized` (`weight_packed` / `weight_scale`), which are
  repacked into ggml's MXFP4 blocks in memory (only bits move: the codes and the E8M0 scales are
  the same). That copy cannot be streamed, so the folder must fit in memory: for the released
  1.56 TB the loader says so and points to a GGUF (an `MXFP4_MOE` conversion holds the same
  weights).

The tokenizer is a tiktoken vocabulary (`tiktoken.model`, no `tokenizer.json`). It is turned
into byte-level BPE whose merges are every split of a token into two tokens, ranked by the
merged token's id: tiktoken's rule (merge the adjacent pair that forms the lowest-ranked token,
the leftmost on ties). A GGUF file's own merge list is an approximation of that and is not used.
The ids equal tiktoken's on 410 texts (12332 tokens: multilingual text, code, runs of repeated
characters, random strings of several scripts and of glued tokens, the rendered chats below),
from `tiktoken.model` and from the vocabulary in a GGUF header (a converted tiny checkpoint's
and the released one of danielhanchen's conversion).

The chat format is Moonshot's `encoding_k3.py` (`chat::KimiChat`): no BOS; messages are
`<|open|>message role="user"<|sep|>` ... `<|close|>message<|sep|><|end_of_msg|>`; a reply is a
`think` block (in thinking mode), a `response` block and, when it calls tools, a `tools` block of
`call` and `argument` tags; tools are declared as compact JSON with sorted keys in a first system
message. The rendered prompts are byte-identical to the encoder's with and without thinking, with
a system prompt, with tools, with a past reply and through rounds of tool calling (a unit test
holds them). The blocks are tags of several tokens, so the streamed text is filtered by tag
(`generate::TextStream`): tags are never shown, the `think` block is reported as thinking, and
the text stops at the `tools` tag.

Not rendered: the encoder's optional system messages (`thinking_effort`, `tool_choice`,
`response_format`), a past reply's reasoning, images. Constrained replies (`response_format`,
a forced `tool_choice`) are refused: the grammars know no reply made of tags. One difference
from the encoder: it encodes message text with the special tokens disallowed, so `<|open|>`
typed by a user stays text; here, as for the other models, text is encoded as a whole and such
a string becomes the token.

## Agreement

The reference is Moonshot's `modeling_kimi_linear.py` run in float32 on the CPU
(`tools/ref_kimi_k31.py`): its fla-core kernels are Triton (GPU only) and are replaced by
fla-core's own plain PyTorch functions (`naive_recurrent_kda`, `naive_kda_lowerbound_gate`) and
by one-line equivalents of the short convolution and the gated norm; `A_log` is narrowed to the
heads; the indexer above is added to the MLA layers. The tiny checkpoint
(`tools/kimi_k31_tiny.py`) is in the released layout with every feature of the text model: 13
layers of 64, KDA with 4 heads of 16, MLA layers 4, 8, 12 and 13 with indexers (16 heads, top 12,
so most queries are sparse), streams banked every 4 layers, a dense first layer, 8 experts (top
3) in a latent of 32 with its norm, 2 shared experts, SiTU with both betas, the real vocabulary.
`examples/kimi_check.rs` compares.

| check (bf16 weights) | prefill: max logit diff / mean KL | 16 greedy decode steps |
|---|---|---|
| 96 tokens, three prompts | 2.0e-4 to 7.9e-4 / 7e-11 to 2.7e-10 (96/96 top-1) | 6e-5 to 3.3e-4 / 4e-11 to 4.5e-10 (16/16) |
| 300 tokens | 2.7e-4 / 1.3e-10 (300/300) | 1.4e-4 / 1.0e-10 (16/16) |
| 300 tokens in passes of 64, 7, 1 | 2.7e-4, 3.6e-4, 3.6e-4 (300/300) | 1.4e-4, 1.3e-4, 1.3e-4 (16/16) |
| 300 tokens, the reference itself in passes of 7 | 3.8e-4 / 1.4e-10 (300/300) | 1.7e-4 / 1.1e-10 (16/16) |
| Kimi K3 (no indexers), 300 tokens | 3.5e-4 / 8.1e-11 (300/300) | 6.1e-5 / 3.5e-11 (16/16) |
| MXFP4 experts as released, 300 tokens in one pass | 6.5e-4 / 2.8e-10 (300/300) | see below |
| the same checkpoint as GGUF (llama.cpp's converter; float32 and bf16), 96 and 300 tokens | 2.1e-4 to 2.9e-4 / 8e-11 to 1.3e-10 (all top-1) | 6e-5 to 1.4e-4 / 4e-11 to 1.2e-10 (16/16) |
| that GGUF, 300 tokens in passes of 7 | 2.6e-4 (float32), 3.6e-4 (bf16) (300/300) | 1.2e-4, 1.1e-4 (16/16) |
| Kimi K3 as GGUF (`kimi-k3`), 300 tokens | 3.4e-4 / 8.2e-11 (300/300) | 1.8e-4 / 8.1e-11 (16/16) |
| GGUF with MXFP4 experts, 300 tokens in one pass | 5.8e-4 / 2.2e-10 (300/300) | see below |

Logits reach 10 to 12. The same top token at every step means the greedy continuations are
identical. The reference differs from itself by 2.2e-4 to 2.7e-4 between one pass and passes of
7 or 1 tokens: the model amplifies float32 rounding a few hundred times (with larger random
weights the reference chooses other experts and keys than itself after some 50 tokens, which is
why the tiny checkpoint's weights are small).

That amplification is why the MXFP4 rows stop at the prefill. In passes of 8 tokens or more the
experts run as a grouped matmul over the decoded 4-bit weights, with float activations. In
shorter passes (decoding) they run on the quantized kernels every GGUF model uses, with int8
activations: a product is off by up to 2 or 3% of the largest (unit tests
`kimi::tests::mxfp4_experts*`: the repacked matrix decodes to exactly the values of the codes and
scales, and short and long passes agree within that rounding). On this random model, whose
experts read 32 values (one block), that rounding is not small: in one-token passes 170 of 300
top tokens are the reference's (KL 0.23), as it changes the experts chosen; with no hard choice
(`kimi_k31_tiny.py --smooth`: every expert routed, every key attended) 254 of 300 (KL 1.0e-2;
3.8e-3 over the 16 decode steps, 14 or 15 top tokens). So token-by-token decoding with 4-bit
experts is verified only up to those kernels' rounding. On trained weights the same kernels agree
with llama.cpp for the other models of [the table](models.md); that was not measured for this
one. llama.cpp shows the same on the tiny GGUF with MXFP4 experts, already after the prompt (KL
0.07, another top token): it runs 4-bit weights against int8 activations in every pass.

llama.cpp (master of 2026-11-14, `src/models/kimi-k31.cpp`) on the same float32 GGUF, with a
float32 KV cache, against the same reference (the logits after the prompt and after each of 16
greedy tokens, `tools/llama_logits_dump.cpp`): 16/16 greedy tokens, max logit diff 1.3e-3 and KL
1.6e-8 after 96 tokens, 2.0e-3 and 2.2e-8 after 300, 8.4e-4 and 3.0e-9 for Kimi K3. That is an
independent implementation of the indexer agreeing with the one in the reference and here. (For
this comparison the reference takes llama.cpp's two epsilons: 1e-5 in the queries' and keys' L2
norm, where fla-core has 1e-6, and in the index keys' LayerNorm.) llama.cpp on the bf16 GGUF does
not agree with anything (KL 1.5 after the prompt): its dot products round the activations to bf16
when the weights are bf16, and this model amplifies that.

A unit test without a reference (`kimi::tests::tiny_model`) holds that the logits do not depend
on how the tokens are split into passes, that a rollback inside a recorded pass leaves no trace
(the recurrent states are restored and the kept tokens run again), and that the indexer's choice,
the latent norm, SiTU's second beta and the banking period each change the logits.

On the released files, without their tensor data (sparse local files with the real headers,
`kimi_check FILE --load`): the loader maps all 2693 tensors of
danielhanchen/Kimi-K3.1-Pro-GGUF-Preliminary UD-Q4_K_XL (`kimi-k3` with indexer keys),
DevQuasar/moonshotai.Kimi-K3.1-Pro-GGUF MXFP4_MOE, KarLexAI/Kimi-K3.1-Pro-GGUF KLX-IQ1_S and
midagedev/Kimi-K3.1-Pro-GGUF IQ1_S. The tensor names of the tiny safetensors checkpoint are
those of the release's index (53 patterns, the vision tower aside).

```sh
# every part in one folder (580 GB); streamed from storage, 42 GB read per token
./target/release/oxinfer --model midagedev/Kimi-K3.1-Pro-GGUF:IQ1_S --think -p "Why is the sky blue?"
# the checks above
python tools/kimi_k31_tiny.py tiny CODE_DIR && python tools/ref_kimi_k31.py tiny CODE_DIR ref.safetensors 300 16
cargo run --release --example kimi_check -- tiny ref.safetensors --chunk 7
```

`CODE_DIR` holds `modeling_kimi_linear.py` of moonshotai/Kimi-K3 and the small files of
moonshotai/Kimi-K3.1-Pro.

## Not done

- A run on the real weights: no speed, no text quality, no comparison with llama.cpp on a
  released GGUF. A token reads 42 GB (IQ1_S) to 85 GB (MXFP4_MOE) of weights, the dense layers
  and 16 experts per layer: decoding is slow unless most of that stays in the page cache.
- Image and video input (MoonViT, the patch merger, the processor): `mmproj` files and the
  `vision_tower.*` tensors are not read.
- Streaming the released safetensors (their experts are repacked in memory, see above).
- Token-by-token verification with 4-bit experts (above).
- GPU decoding (no whole-pass graph for this family, and the SiTU experts run on the CPU),
  speculative decoding with a draft (the model has no MTP layer; DSpark drafts for it exist on
  Hugging Face and are not read). Prompt-lookup rounds run on it: a rejected round replays the
  delta rule over the kept tokens from the recorded inputs (the rollback is unit-tested; lookup
  itself only ran on random weights, where it drafts nothing).
- Session snapshots, and reusing the cache of an earlier prompt that shares only a prefix (the
  recurrent states roll back inside the last recorded pass only).
- Kimi-Linear-48B (`kimi_linear` with rotary MLA, a low-rank KDA gate and SiLU): refused at load.
- Runtime interventions use the points around each sublayer on the running sum and the dense
  and shared MLPs' neurons; they were not measured on this architecture, and the routed experts
  (which write into the latent space) are not offered.
