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
