# Gemma 4 26B-A4B (MoE)

`google/gemma-4-26B-A4B-it` is the dense Gemma 4 layer (30 layers, 25 sliding with a 1024-token
window and 8 KV heads of 256, 5 global with 2 KV heads of 512 and keys reused as values) with a
mixture of experts next to the MLP. Every layer runs both, on the same residual stream:

- the dense GELU MLP (2112 wide) as usual, its output RMS-normed (`post_feedforward_layernorm_1`);
- a router reading the residual RMS-normed without weight, times a learned scale / sqrt(hidden):
  softmax over 128 experts, the top 8 renormalized, each weight times a learned per-expert scale;
- the 8 chosen GELU experts (704 wide, fused `gate_up_proj`) on the residual through their own
  norm (`pre_feedforward_layernorm_2`), their weighted sum RMS-normed (`post_feedforward_layernorm_2`);
- the two outputs added, normed again (`post_feedforward_layernorm`) and added to the residual.

There are no per-layer embeddings and no KV sharing (unlike E2B / E4B). The vision tower (27
layers of 1152) standardizes its pooled output (`std_bias`, `std_scale`), and with
`use_bidirectional_attention: "vision"` (also 12B / 31B) the soft tokens of one image attend to
each other both ways on the sliding layers (global layers stay causal; a prefill chunk never
splits an image). Its chat template closes an empty thought channel in the generation prompt
when thinking is off (`<|turn>model\n<|channel>thought\n<channel|>`), which oxinfer reproduces.

The routed experts share the Qwen MoE machinery (`moe.rs`): decode batches the 8 experts' input
projections into one matmul, prefill runs all experts as one grouped matmul (on the matrix units
on M5 / A19), `--quant q8|q4` and `--save-quantized` quantize them (a Q4 copy is 13 GB, text only
like the E2B / E4B copies), and `--stream` reads them on demand from storage (with
`--cache-budget-gb 12`: 4.8 GB of dense weights resident, 46 GB of experts read as routed, 5.4
tok/s with the experts mostly in the page cache). The same model loads from llama.cpp GGUF files (`gemma4` architecture, text
only: ggml-org, unsloth and bartowski publish them) and from `mlx-community/gemma-4-26b-a4b-it-4bit`
(4-bit `switch_glu` experts, the router in 8 bits).

```sh
./target/release/oxinfer --model google/gemma-4-26B-A4B-it --mmap -p "Write a haiku about the ocean."
./target/release/oxinfer --model google/gemma-4-26B-A4B-it --mmap --image photo.jpg -p "What is in this picture?"
./target/release/oxinfer --model ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_0 -p "Hello"
./target/release/oxinfer --model mlx-community/gemma-4-26b-a4b-it-4bit -p "Hello"
```

Agreement with transformers (float32 compute on the bf16 weights; five chat prompts of 21-1758
tokens, the longest well past the sliding window, and one image question; last-token logits, then
the 2-32 greedy tokens oxinfer generated fed back to the reference, which must pick each of them):

| oxinfer | max logit diff | KL | same top 1 | other |
|---|---|---|---|---|
| bf16, CPU | 1e-4 - 3e-4 (image: 0.03) | < 4e-9 (image: 6e-7) | 6/6 | same top 10 in the same order; 147/147 greedy tokens |
| bf16, Metal (half-precision matrix units) | 0.004 - 0.44 | < 1.3e-4 | 6/6 | |
| `--quant q8` | 0.38 - 1.1 | 7e-11 - 3.8e-3 | 6/6 | |
| `--quant q4` | 4.8 - 15 | 2e-8 - 0.04 | 6/6 | |
| GGUF Q8_0 (ggml-org) | 0.37 - 1.1 | 1e-11 - 4e-4 | 5/5 | text only |
| GGUF Q4_0 (ggml-org) | 11 - 26 | 2e-6 - 0.07 | 5/5 | text only |
| MLX 4-bit (mlx-community) | 5.5 - 14 | 3e-10 - 0.11 | 6/6 | |

(Every row against the same bf16-weight reference, so the quantized rows measure the quantization;
the 1758-token prompt is the hardest.) One trap when building that reference: transformers loaded
in bf16 rounds the embedding scale buffer, sqrt(2816) = 53.07, to 53.0, which on its own moves the
router logits by 3e-3 and flips expert choices; the float32 reference restores it. With it
restored, every one of the 750-4170 routing decisions per prompt is the same in both.

On the Q8_0 GGUF, oxinfer and llama.cpp (CPU) pick the same next token on all five prompts; their
top-20 log-probabilities differ by 0.2-1.9, but against the float32 forward pass on the bf16
weights oxinfer is the closer of the two on every prompt (0.05-0.66 against 0.25-1.90), so the gap
is llama.cpp's int8 activations rather than a different reading of the file. The GGUF tokenizer,
rebuilt from the vocabulary and merges, gives the same ids as `tokenizer.json` (this README, ten
languages, emoji, special tokens).

Speed on an M5 Max (18 threads, `--mmap`, the page cache warm; prefill of the 1758-token prompt,
decode of 256 tokens of code; median of 2 runs while other jobs kept the load average near 4.5, so
lower bounds):

| weights | device | prefill tok/s | decode tok/s |
|---|---|---|---|
| bf16 (7.6 GB read per token) | CPU | 152 | 30 |
| bf16 | Metal (default) | 544 | 30 |
| `--quant q8` | CPU | 139 | 48 |
| `--quant q8` | Metal | 571 | 48 |
| `--quant q4` | CPU | 136 | 68 |
| `--quant q4` | Metal | 576 | 70 |
| GGUF Q8_0 (2046-token prompt) | CPU / Metal | 133 / 605 | 44 / 45 |
| GGUF Q4_0 (2046-token prompt) | CPU / Metal | 130 / 603 | 61 / 60 |
| MLX 4-bit | CPU / Metal | 133 / 225 | 65 / 64 |

Decode runs on the CPU on either device (one token per pass reads the weights once, and the CPU
reaches the memory bandwidth). Prefill attention runs as GEMMs (on the GPU for a chunk that starts
the context), which took the Metal prefill from 325 to 540 tok/s and the CPU one from 117 to 150.
The MLX experts stay on the CPU in prefill (the GPU's grouped expert kernels take plain Q4, not
MLX's affine form), hence its lower Metal number. An image costs about 0.8 s in the vision tower
on Metal (its 4304-wide MLP is zero-padded to whole GPU tiles), 5 s on the CPU.

Against llama.cpp on the same GGUF files (`tools/bench/bench.py`: same prompt tokens, greedy,
llama.cpp through `llama-server`, oxinfer as a fresh process per run; M5 Max at load 4-5; median of
2 runs for Q4_0, 1 for Q8_0):

| file | engine, device | prefill 512 | prefill 2046 | decode (256 tokens of code) |
|---|---|---|---|---|
| Q4_0 | llama.cpp, Metal | 3230 | 3006 | 122 |
| Q4_0 | oxinfer, Metal | 564 | 603 | 60 |
| Q4_0 | llama.cpp, CPU | 144 | 127 | 57 |
| Q4_0 | oxinfer, CPU | 138 | 130 | 61 |
| Q8_0 | llama.cpp, Metal | 2912 | 2888 | 94 |
| Q8_0 | oxinfer, Metal | 512 | 605 | 45 |
| Q8_0 | llama.cpp, CPU | 148 | 130 | 45 |
| Q8_0 | oxinfer, CPU | 148 | 133 | 44 |

On the CPU the two are level; with the GPU, llama.cpp runs the whole pass there (5x the prefill,
2x the decode) while oxinfer keeps attention, routing and decode on the CPU. The greedy outputs
agree: Q4_0 identical for 64 tokens on both devices, Q8_0 splitting at a near tie after 31-48.

### Why the Metal prefill drifts further from the CPU here than on dense models

After a 2681-token raw prompt (9 kB of this repository's documentation), the last-token
distribution of the Metal prefill (whole-pass plan, matrix units) is a KL of 0.21-0.29 from the
CPU's on the Q4_0 GGUF, where dense models sit at 1e-5 to 1e-4. Neither path is wrong; the model
amplifies rounding through its routing, and this prompt position amplifies it most:

- Routing is crowded. The router logits spread by about 0.8-1.1 over the 128 experts, so the 8th
  and 9th are a median 0.04-0.07 apart; 1.4% of tokens have them within 1e-3, 13% within 1e-2.
  Half-precision operands on the matrix units move the router logits by 9e-5 at the first layer,
  enough for 6 of the 2681 tokens to swap an expert there; each swap changes that token by a few
  percent, attention carries it to later tokens, and by the last layer 800-1000 tokens route
  differently (router logits 2e-2 apart). The swapped experts held 4-9% of their token's routing
  weight on average, the smallest shares, as near-ties do.
- Even two f32 CPU GEMMs (`--prefill-gemm sme` and `f32`, differing only in summation order, 1e-7
  apart after the first layers) disagree by a KL of 0.044 at that position: one near-tie at layer
  12 grows into 10-12 differing routes per layer and a 6% different last hidden state.
- llama.cpp shows the same: its Metal and CPU backends are a KL of 0.64 apart on the same tokens
  (Q4_0), and Q4_0 against Q8_0 is a KL of 4.9 at this position in both engines. llama.cpp Metal is
  0.05 from oxinfer's CPU path and 0.25 from its Metal plan. Other prompts are calmer: the same
  prompt on the Q8_0 file gives 4e-3 (Metal plan against CPU), another 3k-token document 2e-3.
- The paths agree layer by layer up to that amplification. With the pass cut after its first L
  layers, the final hidden states of the plan and of the per-operation GPU path (CPU code, GPU
  matmuls) differ from the CPU's by the same amount at every depth (median per token 1.4e-4 at
  L = 1, 3e-4 at 6, 2e-3 at 12, 2e-2 at 18, 8e-2 at 30), with no step in the first 1024, the next
  1024 or the tokens past the 2048-token plan step; with exact f32 matmuls (`OXINFER_METAL_MMA=0`,
  which skips the plan) the GPU is a KL of 3.5e-6 to 3e-4 from the CPU. The plan's sliding / global
  attention (heads of 256 and 512 in the flash kernel), routing weights, GELU experts,
  per-expert scales and the dense MLP beside them match the CPU on a tiny random checkpoint of
  this layout to rounding (`tests/gemma_moe_gpu.rs`: bf16 / Q8 / Q4, plan steps of 64 tokens past
  a 48-token window, then GPU decode steps; 1e-3 to 3e-3 per token, against 0.37 with SiLU
  experts).
- Greedy decoding agrees where it can: on three chat prompts (64, 62 and 2716 tokens), 200 tokens
  of oxinfer on Metal (plan prefill, GPU decode) are identical to llama.cpp Metal's, with or
  without matrix units. The CPU decode (int8 activations, as llama.cpp's CPU backend) leaves them
  at near-ties after 80, 137 and 34 tokens; llama.cpp's CPU backend leaves its Metal one after 5
  and 122 tokens (and agrees for all 200 on the second prompt). `OXINFER_GPU_DECODE_CHECK=1` puts single GPU and CPU decode steps (same caches) at
  a KL of about 1e-3, up to 0.09 at a routing near-tie.
