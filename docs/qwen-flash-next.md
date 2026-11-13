# Qwen3.8-Flash-Next

`Qwen/Qwen3.8-Flash-Next` (transformers `qwen4_exp`, "a preview of the Qwen4 architecture";
`qwen4exp.rs` on top of the Qwen3.5 code) is 125B parameters with 6B active, plus a 51B n-gram
embedding table and a 4B MTP layer: 48 layers of 2560, 3 Gated DeltaNet layers (sigmoid output
gate instead of SiLU) then one gated attention layer, MoE with 512 experts (10 routed + 1 shared,
640 wide) everywhere. What is new, following transformers' `modeling_qwen4_exp.py` (and
llama.cpp's `qwen4exp`):

- Hyper-connections ("gated residual") replace every norm: the residual stream is 4 streams.
  Before each block they are RMS-normed per stream, a low-rank bottleneck (10240 -> 320, SiLU of
  a quarter, -> 10240, sigmoid) gives an element-wise read gate and the block input is the mean
  of the gated streams; the output is added to every stream with its own weight
  `2 sigmoid(w_s . normed / 4)`. The embedding starts as 4 copies; a final mixer (no write
  weights) feeds the LM head (there is no output norm).
- An n-gram hash embedding before layer 1: each token hashes its bigram and trigram (the previous
  tokens up to an EOS) into 8 heads each, `(t0 m0 ^ t1 m1 [^ t2 m2]) mod p_h + offset_h` with
  primes around 20 million, gathers 16 rows of 160 from a 320M-row table and concatenates them; a
  value projection gated per stream by `sigmoid(signed_sqrt(norm(key_s) . norm(stream_s) /
  sqrt(2560)))`, plus a causal depthwise conv (kernel 4, dilation 3) of its normed copy, is added
  to the streams. The table (29 GB at IQ4_NL in unsloth's quants, 102 GB in bf16) is never loaded:
  each token reads its 16 rows (1.4 KB) from the mapped file.
- QSA sparse attention: each attention layer's indexer (4 query heads of 128 and one key head,
  partial RoPE) scores blocks of 4 cached tokens by their mean raw key (normed, rotated to the
  block's first position): `sum_h relu(q_h . k) / sqrt(128)`; a token attends to the best 512
  blocks plus the incomplete tail. Up to 2051 visible tokens that is every token.

Checkpoints: llama.cpp GGUF (unsloth's quants, split; the n-gram table is the last part) and the
safetensors layout (tested on tiny random checkpoints: the real bf16 release is 360 GB). The
merged weights llama.cpp's converter can write (`attn_qkv_gate`, `hc_*_down_inject`) are refused
with a clear error; unsloth's files have none. The chat format is Qwen3.8's (`--think` or the server's `enable_thinking` /
`reasoning_effort` turn thinking on, with the template's reasoning-effort system line), tools as
for Qwen3.5.

```sh
# 111 GB: 82 GB of weights read through the page cache, the 29 GB n-gram table a few rows a token
./target/release/oxinfer --model unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL -p "Why is the sky blue?"
# MTP speculative decoding with unsloth's draft head (1.3x on code, slower on prose: opt-in)
./target/release/oxinfer --model unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL \
    --mtp-from ~/.cache/oxinfer/hub/unsloth--Qwen3.8-Flash-Next-GGUF/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
    -p "Write a Python function that checks whether a number is prime."
# little memory: stream with a page-cache budget (GPU or CPU decoding, measured on the first tokens)
./target/release/oxinfer --model unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL --stream --cache-budget-gb 40 -p "Hi"
```

A checkpoint whose weights other than the n-gram table exceed 75% of RAM streams by default (as
Mistral Large 4); on a 128 GB Mac UD-Q4_K_XL does not, and is read in place through the page cache
with whole-token GPU decoding. The GPU decode plan wraps the 78 GB of routed experts once and its
single- and multi-token plans share them; prefill then wraps only the experts each pass routes
to (`Experts::wrap_per_pass`), since wiring them twice exhausts the GPU's memory. Past 2051 tokens
of context (where QSA starts dropping blocks) the decode plan scores the blocks, picks the top 512
(a radix select, ties to the lower block as on the CPU) and attends to the picked positions on
the GPU too: 47.6 tok/s after a 4556-token prompt (CPU decoding: 20.4). The indexer keys live in a
one-head KV cache per layer that the plan mirrors like the attention's; the device pools a block's
key when a token ends it, and the host pools it again (its RMSNorm takes `powf(-0.5)`, which the
GPU's `rsqrt` can miss by an ulp) so the cache always holds the CPU path's keys.
`OXINFER_QSA_CHECK=1` checks every step's selections against the CPU path's scoring of the same
queries (no difference over 160 steps x 12 layers at 4.5k context); multi-token verification
(prompt lookup, MTP) runs the same ops and gives the plain GPU decode's tokens with
`--exact-verify`. On the CPU a token's block scores are computed over all threads and its
attention (24 query heads on 2 KV heads) takes the shared-row CPU kernel (see
[Performance](performance.md#performance)): 13.6 -> 17.7 tok/s after a 4338-token prompt, same tokens.
The MTP layer (`mtp_from` in the engine, C API and server options) is the trunk's 4 residual
streams plus the next token's embedding, per stream `fc([enorm(e) ; hnorm_s(h_s)])`, through one
QSA + MoE layer with hyper-connections and its own final mixer, the LM head shared.

Agreement. transformers' `Qwen4ExpForCausalLM` (main branch) in float32 on tiny random checkpoints
with every feature (`tools/qwen4exp_tiny.py`: QSA with a budget of 16 tokens so most queries are
sparse, EOS resets in the n-gram contexts, MoE top-3 of 8, bf16-representable weights; compared by
`examples/qwen4exp_check.rs`): max logit diff 6.6e-7, KL 1e-14, over 40-token prompts and 12 to
30 cached decode steps, on the CPU and with GPU decoding (GPU against CPU: KL 1e-15), including a
switch from GPU decoding to the CPU's sparse attention mid-generation. One seed differed (3e-3):
an indexer score tie at exactly 0 (every head's dot product negative), which `torch.topk` breaks
in no fixed order.

On the real UD-Q4_K_XL against llama.cpp master on the same files (`tools/llama_logits_dump.cpp`,
greedy, every decode step compared):

| prompt | oxinfer CPU vs llama.cpp CPU | oxinfer Metal vs llama.cpp CPU |
|---|---|---|
| chat, 23 tokens + 33 steps | same greedy 32/33, mean KL 1.4e-3 | 32/33, KL 1.3e-3 |
| code, 24 + 49 | 49/49, KL 3.3e-4 | 49/49, KL 1.9e-4 |
| math (thinking), 65 + 49 | 49/49, KL 1.4e-3 | 49/49, KL 2.5e-3 |
| 3189-token prompt (QSA sparse) + 25 | 25/25, KL 3.3e-3 | 25/25, KL 3.0e-3 |

The one disagreement is a near-tie where llama.cpp's own Metal build picks oxinfer's token too
(llama.cpp CPU against llama.cpp Metal: KL 1.5e-3 at the first step, the same flip). MTP drafts
from unsloth's `mtp-...-shared-Q8_0.gguf` are accepted at 92-97% on code and math, and the output
is the plain greedy output.

Speed on the M5 Max (128 GB; load average 3-9 from other jobs), UD-Q4_K_XL, greedy:

| | prefill | decode |
|---|---|---|
| oxinfer, Metal (default) | 1008-1064 tok/s (4545 tokens), 737-1007 (563; second pass, under load) | 54-59 tok/s (45-49 under load), 47.6 after 4.5k tokens |
| oxinfer, Metal + MTP (3 drafts) | | 72 tok/s on code (3.9 tokens per pass), 46 on prose |
| oxinfer, CPU (`--device cpu`) | | 21.5 tok/s |
| oxinfer, streamed (`--stream --cache-budget-gb 40`) | | 12 tok/s (47% of the time waiting on storage) |
| llama.cpp master, Metal (`llama-bench`) | 1030 tok/s (pp512) | 50 tok/s (tg128) |
| llama.cpp master, CPU (12 threads) | 61 tok/s (pp512) | 21.4 tok/s |

Memory: the GPU run's footprint is 1.5 GB besides the page cache (the weights are file pages,
82 GB of them in use); streamed with a 40 GB budget, 0.7 GB plus 22 GB of page cache.

Image input. The vision tower is Qwen3.5's (a Qwen3-VL ViT without DeepStack: 27 layers of
1152, 16x16 patches over 2 frames, learned 48x48 position grid resampled per image, axial 2D
RoPE, 2x2 merger into 2560-wide tokens placed with interleaved M-RoPE; image positions hash as
`<|image_pad|>` in the n-gram embedding, as transformers does). From safetensors it is the
checkpoint's `model.visual.*`; for a GGUF model it comes from llama.cpp's `mmproj` file
(`clip.projector_type` `qwen3vl_merger`; unsloth publishes `mmproj-F16.gguf` / `-BF16` in the
repo root), found beside the model (its folder or the one above) or named with `--mmproj FILE`
(engine / C / server option `mmproj=FILE`; `mmproj=none` turns images off). The conv kernel
llama.cpp splits per frame (`v.patch_embd.weight`, `.weight.1`) is joined back; F16 weights load
as bf16, which restores the released bf16 values exactly. Images go through `--image`, the
engine's attachments, the C API and the server's `image_url` content parts.

```sh
# the mmproj file in the repo root is picked up automatically
curl -L -o ~/.cache/oxinfer/hub/unsloth--Qwen3.8-Flash-Next-GGUF/mmproj-F16.gguf \
    https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/mmproj-F16.gguf
./target/release/oxinfer --model unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL --image painting.png -p "Describe this image."
```

Checked against transformers' `Qwen4ExpForConditionalGeneration` in float32 on a tiny random
checkpoint with the real tokenizer and processor (`tools/qwen4exp_vl_tiny.py`, compared with
`tools/ref_vl.py`): a 681x975 PNG (630 image tokens) gives image features at 1.9e-4 relative
(pixels differ by 1/255 on a few rounding ties of the bicubic resize), last-position logits within
1.7e-4 (KL 3e-8), and the same 12 greedy tokens on the CPU and Metal; the tower loaded from the
llama.cpp-converted mmproj of that checkpoint matches the checkpoint's own to 4.5e-7. On the real
model the 27-layer tower from `mmproj-F16.gguf` matches transformers' tower with the same weights
to 2.5e-5 relative on the CPU (and with `OXINFER_METAL_MMA=0`), 0.9% with Metal's default
half-precision matmul operands (0.8 s instead of 1.4 s for 630 tokens); the model names Vermeer's
"Girl with a Pearl Earring" from the PNG and describes it (prefill 721 tokens in 3.4 s).

Prefill on Metal runs whole passes on the GPU: the decode graph compiled as a prefill plan
(`MetalBackend::build_plan_pf`, chunks of up to 2048 tokens, `OXINFER_PREFILL_TOKENS`; prefill
chunks of 1024 and up are stretched to it, ~3 GB of device buffers) keeps the
activations on the device and runs the matmuls as matrix-unit GEMMs, the MoE routers as f32 GEMMs,
the routed experts as grouped GEMMs over each expert's tokens (the routing is read back before each
MoE layer: the only host round trips), the Gated DeltaNet rule (per-token norms and gates in
parallel, then the recurrence with each column's four parts in one SIMD-group), the attention
(dense, or over each token's QSA positions, four query heads per threadgroup), the indexer and the
convolutions over the whole chunk, and the expert weights wrapped once for both plans.
`OXINFER_PREFILL_GPU=0` keeps everything but the matmuls on the CPU; `attn,gdn,router` gives the
previous op-by-op offload. On raw text (`examples/prefill_kl.rs`, first pass of a process / a
second pass, under load from other processes): 563 tokens 160-190 -> 215-335 / 737-1007 tok/s,
1323 tokens 178-231 -> 490-632 / 1106-1257, 4545 tokens 214-251 -> 653-871 / 1008-1064 (the first
pass includes building the GPU plans, ~1 s; unloaded the op-by-op offload measured 261 / 324 /
374). llama.cpp master's `llama-bench` under the same load: pp512 817, pp1024 802, pp2048 837,
pp4096 821 tok/s. The results do not depend on the chunk size (every GEMM row is computed alone).
Against the CPU path the last position's KL is 1.6e-2, 3.9e-4 and 9.1e-4 (the
op-by-op offload 2.9e-2, 1.2e-3, 2.4e-4; the older Metal prefill 3.8e-2, 1.8e-4, 1.1e-3): this
model amplifies any rounding (a flipped expert among 512), and the half-precision matrix units
dominate. Image prompts (and past 2051 tokens, QSA) run on the plan too. `OXINFER_PF_PROFILE=1`
runs each op of the plan in a command buffer of its own and prints the GPU time per kind of op
(with `prefill_kl` and `OXINFER_PROFILE`): at 4545 tokens the routed experts take ~40% (grouped
32-row tiles, much of them padding at ~40 tokens per expert, and the dequantization not overlapped
with the matrix units), the attention ~11%, the GEMMs ~25%.

Not done: video input from GGUF is wired the same way but untested, the merged-weight GGUF layout,
and faster routed experts in prefill (less tile padding, or larger chunks).
