# oxinfer

[![CI](https://github.com/KarpelesLab/oxinfer/actions/workflows/ci.yml/badge.svg)](https://github.com/KarpelesLab/oxinfer/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/oxinfer.svg)](https://crates.io/crates/oxinfer)
[![docs.rs](https://img.shields.io/docsrs/oxinfer)](https://docs.rs/oxinfer)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**[Try it in your browser](https://karpeleslab.github.io/oxinfer/)**: Qwen 3.5 0.8B downloads from
Hugging Face and runs in the tab, with WebGPU when the browser has it; MLX 4-bit Gemma 4, Gemma 3
and Qwen3 models load from the mirror too. Early days: the crate is at 0.0.x and the API and file
formats may still change.

Local LLM inference in pure Rust: Gemma 4 (text, images, video, audio), Qwen 3.5 / 3.6 / 3.8
(text, images, video), Gemma 3 and Qwen 3 today, more model families as they come; also
embeddings, Stable Diffusion, Whisper speech to text and Orpheus text to speech. No BLAS, no ML framework: weights are read straight out
of the safetensors files (HuggingFace checkpoints or MLX 4-bit ones) and every operation
(matmuls, attention, norms, RoPE, sampling) is implemented here, with hand-written SIMD kernels
(NEON, AVX2, wasm simd128) and its own thread pool. The CPU path runs everywhere; GPUs (Metal,
CUDA, WebGPU in the browser) are detected at run time and used for prefill when present, with
nothing linked at build time.

Output is validated against the `transformers` float32 reference: on Gemma 4 E2B-it the
last-token logits agree to a max absolute difference of about 1e-4 and greedy decoding produces
identical tokens.

## Goals

1. **Run everything, everywhere.** Any model on any machine: a phone, a browser tab, a laptop with
   no GPU, a server. It may be slow, but it works. A missing accelerator, too little memory or an
   unusual platform should never be the reason a model cannot run. The CPU path is the reference
   and is always there; a 1 TB model on a machine with 1 GB of RAM and no GPU should still produce
   tokens, by reading the weights from storage as it goes (see
   [Streaming weights from flash](#streaming-weights-from-flash-models-larger-than-ram)), very slowly.
2. **Then make it fast.** Anything that improves speed on a given machine is in scope: a GPU
   when there is one (Metal, CUDA, WebGPU, detected at run time), the Neural Engine through Core
   ML, quantization, SIMD kernels per CPU (NEON, SDOT / i8mm, AVX2, wasm simd128), speculative
   decoding (MTP, drafters), prefetching and caching for weights that do not fit in memory. When
   something is missing, oxinfer falls back to a slower path rather than failing.

Correctness comes first in both: every model and every fast path is checked against the reference
implementation (the [verification](#verifying-against-the-reference) section and the per-model
notes give the measured agreement), and a speed-up that changes the output must say so and stay
optional.

## Supported models

| Model | Params (total) | bf16 size | Status |
|---|---|---|---|
| google/gemma-4-E2B-it | 5.1B | 10.2 GB | text, images, video, audio; matches reference (max logit diff 1e-4) |
| google/gemma-4-E4B-it | 8.0B | 16.0 GB | text, images, video, audio; matches reference (max logit diff 8e-5) |
| google/gemma-4-12B-it | 12B | 24 GB | supported (the 26B-A4B code without the experts), untested |
| google/gemma-4-31B-it | 31B | 62 GB | supported, sharded loading, untested |
| google/gemma-4-26B-A4B-it | 25.8B MoE, 3.8B active | 52 GB | text, images; matches transformers (max logit diff 3e-4, same greedy tokens); GGUF and MLX 4-bit copies too; see [Gemma 4 26B-A4B](#gemma-4-26b-a4b-moe) |
| autotrust/GEV-26B-Decide, JEV-Gemma4-26B-A4B | 25.8B MoE + LoRA | 52 GB + 0.15 GB | typed decisions over text and images, on the bundled or Google's base; match transformers + peft to 2e-6 (see [Typed decisions](#typed-decisions-jev-models)) |
| Qwen/Qwen3.8-27B (and Qwen3.5/3.6 dense) | 27.8B | 55.6 GB | text, images, video; matches the bf16 reference; see below |
| Qwen/Qwen3.6-35B-A3B and fine-tunes (PixilabAI/Zawen-v0.1-35B-A3B) | 35B MoE, 3B active | 69 GB | text (the vision code applies to MoE checkpoints too but is untested there: Zawen ships no vision tower); Zawen matches the reference (see below) |
| autotrust/JEV-9B, JEV-27B (Jev decision models) | 9B / 27.8B + LoRA | 18 / 54 GB + 0.16 / 0.44 GB | typed decisions, on the bundled or any matching base; match transformers + peft (see [Typed decisions](#typed-decisions-jev-models)); image decisions with JEV-9B's `vl/` on Qwen/Qwen3.5-9B |
| autotrust/JEV-27B-VL | 27.8B + vision + LoRA | 55.6 GB + 0.45 GB | typed decisions over text and images; matches the reference server's math on transformers + peft (see [Image decisions](#image-decisions)) |
| Qwen/Qwen3.5-0.8B | 0.8B | 1.7 GB | text, images, video; matches transformers float32 (see [Image, video and audio input](#image-video-and-audio-input)); the small one used for the wasm measurements |
| mlx-community/gemma-4-e2b-it-4bit, gemma-4-e4b-it-4bit | 5.1B / 8.0B | 3.6 / 5.2 GB (MLX 4-bit) | text, images and audio; see [MLX checkpoints](#mlx-checkpoints) |
| mlx-community/gemma-4-26b-a4b-it-4bit | 25.8B MoE | 15.4 GB (MLX 4-bit) | text and images (4-bit `switch_glu` experts) |
| Gemma 4 GGUF (ggml-org/gemma-4-26B-A4B-it-GGUF, unsloth, bartowski, ...) | 25.8B MoE tested | 14.6 GB (Q4_0) - 27 GB (Q8_0) | text; same next token as transformers and llama.cpp (see [Gemma 4 26B-A4B](#gemma-4-26b-a4b-moe)) |
| mlx-community/Qwen3-4B-4bit (Qwen3 dense) | 4.0B | 2.3 GB (MLX 4-bit) | text; matches the reference on the same weights (same top 10) |
| mlx-community/gemma-3-4b-it-4bit (Gemma 3) | 4.3B | 3.4 GB (MLX 4-bit) | text only; matches the reference, also past the 1024-token window |
| google/embeddinggemma-300m | 0.3B | 1.2 GB (fp32) | embeddings; matches sentence-transformers (see [Embeddings](#embeddings)) |
| Qwen/Qwen3-Embedding-0.6B | 0.6B | 1.2 GB | embeddings; matches sentence-transformers |
| BAAI/bge-m3 | 0.6B | 2.3 GB (fp32, `pytorch_model.bin`) | dense, sparse and ColBERT embeddings; matches the reference |
| sentence-transformers/all-MiniLM-L6-v2, paraphrase-multilingual-MiniLM-L12-v2 | 23M / 118M | 90 / 470 MB | embeddings (BERT); match sentence-transformers |
| BAAI/bge-small-en-v1.5, intfloat/multilingual-e5-small, mixedbread-ai/mxbai-embed-large-v1 | 33M-335M | 0.13-1.3 GB | embeddings (BERT / XLM-R); match sentence-transformers |
| stabilityai/sd-turbo | 1.3B | 2.6 GB (fp16) | text-to-image in one step, image-to-image in one or two; matches diffusers (see Image generation) |
| stable-diffusion-v1-5/stable-diffusion-v1-5 (and SD 1.x / 2.x fine-tunes) | 1.1B | 2.0 GB (fp16) | text-to-image, image-to-image, inpainting by mask blending; matches diffusers |
| timbrooks/instruct-pix2pix | 1.1B | 2.0 GB (fp16) | instruction edits of an image ("make it snowy"), also in 4 steps with the SD 1.5 LCM-LoRA; matches diffusers (see [Editing images](#editing-images)) |
| stable-diffusion-v1-5/stable-diffusion-inpainting (and other 9-channel SD inpainting UNets) | 1.1B | 2.0 GB (fp16) | inpainting with a mask; matches diffusers |
| vinesmsuic/magicbrush-jul7 (MagicBrush: InstructPix2Pix fine-tuned for local edits) | 1.1B | 4.3 GB (fp32 `.bin`) | instruction edits ("add a red car"); matches diffusers |
| deepseek-ai/Janus-Pro-1B | 2.1B (1.5B LM) | 4.2 GB (`pytorch_model.bin`) | text-to-image (384x384) and image understanding; matches the official `janus` package (all 576 greedy image tokens identical; see [Janus-Pro](#janus-pro-text-to-image-and-image-understanding)) |
| deepseek-ai/Janus-Pro-7B | 7.4B | 14.8 GB (sharded `.bin`) | same; matches the reference (logits, first 64 greedy image tokens) |
| anemll/* (ANEMLL Core ML: Qwen3, Qwen2.5, Gemma 3, Llama 3.2, ...) | 0.27B-1.7B tested | 0.3-1.5 GB (LUT6) | Apple Neural Engine via Core ML; same tokens as ANEMLL's runtime (see [Core ML](#core-ml--apple-neural-engine)) |
| openai/whisper-tiny, -base, -small, -medium (and `.en`) | 39M-769M | 0.15-3.1 GB (fp32) | speech to text; transcripts match transformers token for token (see [Speech to text](#speech-to-text-whisper)) |
| openai/whisper-large-v3, -large-v3-turbo, distil-whisper/distil-large-v3 | 1.55B / 809M / 756M | 3.1 / 1.6 / 1.5 GB (fp16) | speech to text; match transformers token for token |
| canopylabs/orpheus-3b-0.1-ft (ungated copy: unsloth/orpheus-3b-0.1-ft) + hubertsiuzdak/snac_24khz | 3.3B + 20M | 6.6 GB + 80 MB (fp32) | text to speech; greedy tokens and waveform match transformers + snac (see [Text to speech](#text-to-speech-orpheus)); also any Llama 3.x as a text model |
| Orpheus GGUF (isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF, lex-au/Orpheus-3b-FT-Q8_0.gguf) | 3.3B | 2.4 / 4.0 GB | text to speech; whisper-small WER 0-1% like the safetensors model |
| Llama 3.x GGUF (bartowski/Llama-3.2-1B-Instruct-GGUF, ...) | 1.2B tested | 0.8 GB (Q4_K_M) - 2.5 GB (F16) | text; F16 file bit-identical to the safetensors model, quantized files match llama.cpp (see [Llama 3 GGUF](#llama-3-gguf)) |
| canopylabs/orpheus-3b-0.1-pretrained (ungated copy: unsloth/orpheus-3b-0.1-pretrained) | 3.3B | 6.6 GB | voice cloning from a reference clip (SNAC encoder); prompt and greedy tokens match the canopylabs notebook + transformers |
| canopylabs/3b-{fr,de,ko,hi,zh,es_it}-ft-research_release (ungated copies: Our-Org-2/...) | 3.3B | 13 GB (fp32) | multilingual text to speech (French and German tested; French greedy tokens match transformers) |
| hubertsiuzdak/snac_24khz, snac_32khz, snac_44khz | 20M / 55M / 55M | 80 / 220 / 220 MB | the SNAC codec alone, encode and decode (32 / 44 kHz with local attention); codes identical to the `snac` package |

### Gemma 4 26B-A4B (MoE)

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

### Qwen3.5 / Qwen3.8

The Qwen3.5-family dense models are a different architecture (`qwen.rs`): 48 Gated DeltaNet layers
(a causal depthwise conv, L2-normalised keys/queries and a 128x128 delta-rule recurrent state per
value head) interleaved with 16 gated full-attention layers (24 query / 4 KV heads of 256, partial
RoPE on 64 dims, a sigmoid output gate), SiLU MLPs and zero-centred RMSNorm. The recurrent layers
keep a fixed-size state instead of a KV cache, so memory does not grow with context for 48 of the
64 layers. The vision tower takes images and videos (see
[Image, video and audio input](#image-video-and-audio-input)); the MTP head drives speculative
decoding with `--mtp` (see Speculative decoding).

The MoE variants (`qwen3_5_moe`, Qwen3.5/3.6-35B-A3B) replace each MLP with 256 routed SiLU
experts of width 512 (softmax router, top 8, weights renormalised) plus a shared expert scaled by
a sigmoid gate. Experts load from the fused `experts.gate_up_proj` / `down_proj` tensors or from
per-expert `experts.N.*_proj` weights (as transformers saves them), and quantize like any other
matrix (`--save-quantized` included). Decode batches the 8 chosen experts' input projections into
one matmul. Prefill gathers each expert's tokens and runs every expert as one grouped matmul on
the CPU, or on M5 / A19 as one GPU submission per layer on the matrix units. With `--quant q4`
(about 19 GB) on an M5 Max, that is about 220 tok/s of prefill with the experts on the CPU,
2.1-2.3x that with the matrix units (see [Performance](#performance)), and 65 tok/s of decode.
On randomly initialised weights at the real width (2048 hidden, 256 experts, 2
layers, the real vocabulary), oxinfer matches transformers in float32 to a max logit difference of
7e-5, with the same 20 greedy tokens. On Zawen-v0.1-35B-A3B's real weights, cut to its first 4
layers, the float32 match is 3e-5. The full 40 layers against transformers in bf16 give a mean
logit difference of 0.035, with the same top 10 in the same order (Q8: same top 10; Q4: same top
token). MLX-quantized Qwen experts are not read yet (Gemma 4's are).

Thinking is on by default in the Qwen template; `--think` enables it here (the trace is dimmed),
otherwise the generation prompt closes the think block.

Text, images, video and audio are supported (E2B/E4B have both towers).

## Embeddings

sentence-transformers models are recognised by their `modules.json` and run in embedding mode:
the backbone (Gemma 3 with bidirectional attention, Qwen3, or a BERT / XLM-RoBERTa encoder in
`bert.rs`), then the listed modules: pooling (CLS, mean, max, last token), dense projections,
normalization. Named prompts come from `config_sentence_transformers.json`. Each input line
gives one JSON array (or `-p TEXT` one):

```sh
./target/release/oxinfer --model Qwen/Qwen3-Embedding-0.6B --prompt-name query -p "What is the capital of France?"
printf 'Which planet is the Red Planet?\nMars looks red.\nVenus is the hottest planet.\n' |
  ./target/release/oxinfer --model google/embeddinggemma-300m --prompt-name query --similarity
#  1.0000  0.8014  0.5470
#  ...
./target/release/oxinfer --model google/embeddinggemma-300m --dim 256 -p "..."   # Matryoshka
./target/release/oxinfer --model BAAI/bge-m3 --sparse -p "..."     # lexical weights {token: weight}
./target/release/oxinfer --model BAAI/bge-m3 --colbert -p "..."    # one vector per token
```

`google/embeddinggemma-300m` is gated on Hugging Face (set `HF_TOKEN`); `unsloth/embeddinggemma-300m`
is an ungated copy. BGE-M3 ships only a `pytorch_model.bin`: `torch.rs` reads `torch.save`
archives directly (a zip of raw storages plus a pickled state dict; only the handful of pickle
opcodes and constructors that state dicts use are understood, and nothing is executed), so the
same goes for its `sparse_linear.pt` / `colbert_linear.pt` heads.

Checked against sentence-transformers float32 (`tools/ref_embed.py`) on short texts in five
languages and a ~2000-token one: cosine between our vector and the reference at least 0.99999 for
EmbeddingGemma (query and document prompts, and at 256 dimensions) and 1 - 1e-6 for
Qwen3-Embedding; BGE-M3's dense vectors 0.99991 or better. fp32 checkpoints are computed with
bf16-rounded matrices; against a reference using the same rounding the vectors agree to about
1e-5, so that is the whole difference. One caveat: on the long text, BGE-M3's ColBERT vectors
have a mean cosine of 0.9995 to the fp32 reference, but one token drops to 0.40. That text is
unusually sensitive: the reference itself in fp16 already drops a token to 0.89. The sparse
weights select the same tokens, with weights within 0.013.

Any BERT, RoBERTa or XLM-RoBERTa sentence-transformers model should work the same way; besides
the three above, all-MiniLM-L6-v2, paraphrase-multilingual-MiniLM-L12-v2, bge-small-en-v1.5,
multilingual-e5-small and mxbai-embed-large-v1 were checked (cosine >= 0.99999 to the reference).
E5 models expect a `query: ` / `passage: ` prefix in the text itself (they define no prompts).
MPNet (all-mpnet-base-v2: BERT plus a shared T5-style relative position bias) is supported too,
cosine >= 0.999991 to the reference (0.999996 against bf16-rounded reference weights).

NomicBERT (`nomic_bert.rs`: rotary positions, fused Wqkv, SwiGLU, dynamic NTK scaling past 2048
tokens) runs nomic-embed-text-v1 and v1.5: cosine >= 0.99998 to the reference (`--trust-remote-code`)
with and without the `search_query: ` / `search_document: ` prefixes, which go in the text (the
model defines no prompts), and on a 4000-token text; 0.999999 against bf16-rounded weights.
v1.5's Matryoshka recipe layer-norms the pooled vector before truncating, which is not a
sentence-transformers module, so `--dim` alone truncates without it.

ModernBERT (alternating global and sliding-window attention) is supported too:
modernbert-embed-base and gte-modernbert-base give cosines of 0.99993 and 0.99998 against the
fp32 reference, and 1 - 1e-6 against one with the same bf16-rounded matrices. modernbert-embed
expects a `search_query: ` / `search_document: ` prefix in the text itself.
Qwen2-based embedders (gte-Qwen2, stella) are another architecture and are not supported yet.

Two details that are easy to get wrong: transformers halves EmbeddingGemma's sliding window for
bidirectional attention (512 in the config means keys less than 257 positions away on either
side), and Qwen3-Embedding pools the `<|endoftext|>` token its tokenizer appends.

From the library (`examples/embed.rs`):

```rust
let dir = oxinfer::hf::resolve("BAAI/bge-m3", None)?;
let files = oxinfer::safetensors::ModelFiles::open(&dir, false)?;
let emb = oxinfer::embed::Embedder::load(&dir, &files, oxinfer::model::Quant::Bf16)?;
let mut e = emb.embed("what is a panda?", None)?;          // or Some("query") with named prompts
e.truncate(256)?;                                         // Matryoshka (--dim)
let lexical = emb.sparse_tokens(&e);                      // BGE-M3 heads: e.sparse / e.colbert
```

## Tool calling

Both model families can call tools, in the format each was trained on: Gemma 4 declares them in
its system turn (`<|tool>declaration:...<tool|>`) and calls `<|tool_call>call:NAME{...}<tool_call|>`;
Qwen 3.5 / 3.8 lists JSON schemas in `<tools>` and answers with `<tool_call><function=...>`.
`src/tools.rs` formats the declarations, parses the calls and feeds results back the way each
template does, and the rendered prompts are byte-identical to `transformers`'
`apply_chat_template(..., tools=...)` for both families.

On the command line `--tools` enables two built-in tools, the local date and time and an exact
calculator:

```sh
./target/release/oxinfer --model google/gemma-4-E4B-it --quant q4 --tools -p "What is 1234.5 * 6789 - 2^20?"
# [tool calculate({"expression":"1234.5 * 6789 - 2^20"}) -> 7332444.5]
# The result of 1234.5 x 6789 - 2^20 is **7332444.5**.
```

In the browser the model gets `get_datetime` and `run_javascript`, which runs the code it writes in
a Web Worker with the network and storage APIs removed and a 5 s limit (hardening, not a security
boundary). Its system prompt describes where it runs: browser, operating system, language, time
zone, local time, screen, CPU cores and GPU. Both can be changed under Settings.

## Typed decisions (Jev models)

The Jev models (`autotrust/JEV-9B`, `autotrust/JEV-27B`, ...) answer a typed question about a
piece of text (or images: [Image decisions](#image-decisions)) with calibrated probabilities, in one
prefill pass and with no generation ("System 1"): `noul` (yes / no, probabilities for `["false", "true"]`), `score` (a 0-5 scale) or
`choice` (2-16 options labelled A-P, up to 256 with the extra single-token labels Q-Z, AA, AB,
...). The prompt is the `bare-v1` template:

```text
[kind] choice
[state] SKU AX-330 stock at 8% of safety level; supplier late twice this quarter.
[question] Supplier response for this scenario.
[options]
A) issue_warning
B) renegotiate
C) dual_source
D) maintain
[decision]:
```

The last token's final-norm hidden state goes through a small head (`head.safetensors`, 24 slots),
the active slots are divided by a per-kind temperature (`calibration.json`) and soft-maxed. The
backbone of JEV-9B / JEV-27B is the unmodified Qwen3.5-9B / Qwen3.8-27B; the decision skill is a
LoRA adapter (`adapter/`, r=16, about 160 / 435 MB) that oxinfer applies **unmerged, at run time,
on top of whatever base weights are loaded** (bf16, `--quant q8|q4`, pre-quantized, MLX or GGUF)
and only during a decision. One loaded model therefore serves chat (the base model, bit for bit:
the adapter's layers are skipped when it is off) and decisions. The bundles that ship merged
weights and read the decision from the LM head (`adapter_vllm/decision_head.json`, e.g.
`autotrust/JEV-35B`) work too.

```sh
# a Jev bundle (backbone, adapter and head; the repo id downloads all of it)
oxinfer --model autotrust/JEV-9B --decide noul \
  --state "Customer says the parcel arrived damaged and wants their money back." \
  --question "Is the customer asking for a refund?"
#   false  0.0092
# * true   0.9908  ########################################

# any matching base you already have, plus only the decision files of the bundle (~420 MB for
# JEV-27B: adapter, head and configs, not the 54 GB backbone)
oxinfer --model ~/.cache/oxinfer/hub/Qwen3.8-27B-oxinfer-q4 --decision-from autotrust/JEV-27B \
  --decide choice --state "SKU AX-330 stock at 8% of safety level; supplier late twice this quarter." \
  --question "Supplier response for this scenario." \
  --option issue_warning --option renegotiate --option dual_source --option maintain

# requests in the /v1/decide JSON format (an object, an array or one per line): one response per line
oxinfer --model autotrust/JEV-9B --decide-json requests.json
```

`--decision-from` (engine option `decision_from`) takes a directory or a repo id; the base must
match the bundle (hidden size, layer count, vocabulary, verbalizer tokens, and the shape of every
adapted layer), otherwise loading fails with a message saying which. As a library:

```rust
use oxinfer::decide::DecisionRequest;
use oxinfer::engine::{Engine, EngineOptions};
let mut opts = EngineOptions::default();
opts.set("quant", "q4")?;
opts.set("decision_from", "autotrust/JEV-9B")?;
let mut engine = Engine::load("Qwen/Qwen3.5-9B", &opts)?;
let d = engine.decide(&DecisionRequest::choice(
    "I was charged twice for my subscription this month.",
    "Which team should handle this ticket?",
    ["billing", "technical support", "sales", "account security"],
))?;
println!("{} ({:.3})", d.best, d.probabilities[d.best_index]);
```

A decision runs in a scratch context: the conversation of the engine is left as it was, so
`send` and `decide` can be interleaved (`examples/decide.rs` checks that the chat reply is the same
before and after a decision, and the same as without any adapter). `Engine::decide_json` and the C
API's `oxi_decide` take and return the JSON of the reference server's `POST /v1/decide` (`kind`,
`state`, `question`, `options` in; `options`, `probabilities`, `choice_index`, `choice`,
`adaptation` (`native` or `wide-labels`), `usage`, ... out); `oxi_decide_info_json` describes the
loaded assets.

Agreement with the reference (`transformers` + `peft`, unmerged adapter, the card's slot-head
readout; 10 questions: 3 noul, 2 score, 4- and 12-option choices and a 20-option wide-label choice
read through `adapter_vllm`'s LM-head LoRA), JEV-9B:

| oxinfer | vs reference f32 | | vs reference bf16 |
|---|---|---|---|
| | max \|Δp\| | max KL | max \|Δp\| |
| bf16 weights, CPU | 4.8e-4 | 5.3e-7 | 3.9e-3 |
| bf16 weights, Metal | 5.0e-4 | 5.8e-7 | |
| BF16 GGUF base (unsloth) + JEV-9B adapter, Metal | 4.6e-4 | 5.2e-7 | |
| `--quant q8` | 6.5e-3 | 1.2e-4 | |

(The bf16 reference itself differs from the f32 one by 4.0e-3.) Qwen3.5-9B with
`--decision-from autotrust/JEV-9B` gives the same probabilities as the JEV-9B folder, and JEV-27B on
the bf16 Qwen3.8-27B reproduces the model card (refund `true` 0.978, AX-330 `dual_source` 0.62);
on the Q4 conversion the largest difference to bf16 is 0.044 (KL 6.6e-3, from the 4-bit weights).
`autotrust/JEV-35B` (merged Qwen3.5-35B-A3B MoE, LM-head readout with 256 trained choice slots)
against `transformers` in bf16 (an f32 reference does not fit in 128 GB): median max |Δp| 4e-4,
worst 0.042 (KL 6.6e-3) on a near-tie score question; MoE routing flips near ties, so the CPU and
Metal runs of oxinfer differ from each other by up to 0.014 too. It decides in 0.25-0.45 s on
Metal for 45-160 tokens.

Latency (M5 Max, 18 threads; a decision is a prefill, so it scales with the prompt):

| model | device | 105-token prompt | 1461-token prompt | plain prefill of the same 1461 tokens |
|---|---|---|---|---|
| JEV-9B (bf16) | CPU | 1.31 s | 18.3 s | 17.8 s |
| JEV-9B (bf16) | Metal | 0.32 s | 2.63 s | 2.44 s |
| JEV-27B on Qwen3.8-27B Q4 (`decision_from`) | CPU | 5.2 s | 70 s | |
| JEV-27B on Qwen3.8-27B Q4 (`decision_from`) | Metal | 0.72 s | 6.2 s | 5.9 s |

The adapter costs 3-7 % of the prefill (low-rank products in f32 NEON on the CPU, after the base
matmul, also when the base matmul runs on the GPU). Not supported: LoRA on routed MoE experts
(no bundle has it), Core ML models, and the adaptive thinking of the reference server
(`"thinking": "auto"`: System 2 reasoning folded into the probabilities; the base model itself
chats as usual in the same engine).

### Gemma 4 26B-A4B bundles

`autotrust/GEV-26B-Decide` and `autotrust/JEV-Gemma4-26B-A4B` (the same weights published twice)
put System 1 on the unmodified `google/gemma-4-26B-A4B-it`: a LoRA (r=32) on the attention and
dense-MLP projections of all 30 layers (205 modules; the routed experts are not adapted), the
24-slot head and the temperatures. What differs from the Qwen bundles: the prompt starts with
`<bos>`, the head's logits are soft-capped like Gemma's own (`30 * tanh(z / 30)`, from
`judge_config.json`), and a choice with more than 16 options is a **tournament** by default, as in
the bundle's `serve_decide.py`: groups of at most 16 options are read natively, then a final pass
over each group's winner plus the best of the rest (16 in all), and the group shares are combined
as the reference does. `"strategy": "single"` (JSON) or `DecisionRequest::strategy` reads them in
one pass through the wide labels instead (GEV only: it needs `adapter_vllm/`'s LM-head readout;
JEV-Gemma4 has none). Images go where they appear, each as `<|image>` + its soft tokens +
`<image|>` with the processor's default budget, attending both ways within the image.

```sh
# only the decision files of the bundle (150 MB), on the Google base already on disk
oxinfer --model google/gemma-4-26B-A4B-it --mmap --decision-from autotrust/GEV-26B-Decide \
  --decide choice --state "Set an alarm for 6:30 tomorrow morning please." --question "Which intent?" \
  --option weather_query --option alarm_set --option alarm_remove --option play_music
oxinfer --model google/gemma-4-26B-A4B-it --mmap --decision-from autotrust/GEV-26B-Decide --decide-json requests.json
```

Agreement with transformers + peft (the card's readout, unmerged adapter, float32 compute on the
bf16 weights): the 10 text decisions of the JEV-9B check (3 noul, 2 score, 4- and 12-option
choices, a 20-option tournament) and 5 image decisions (fox photo and shapes, 1-2 images of 256-266
soft tokens, noul / score / 5- and 6-option choices):

| oxinfer | text: max \|Δp\| | max KL | images: max \|Δp\| | max KL |
|---|---|---|---|---|
| GEV-26B-Decide, bf16, CPU | 1.8e-6 | 7e-12 | 5.9e-3 (shapes; the others 5e-5) | 1.4e-4 |
| JEV-Gemma4-26B-A4B, bf16, Metal | 3.0e-3 | 2.6e-5 | | |
| 20 options with `"strategy": "single"` vs the reference through `adapter_vllm/` | 2.0e-3 | 2.5e-5 | | |

Prompt token counts match the processor's. The image difference comes from the vision tower's
input resize, as for chat (logits of image prompts differ by 0.03). On Metal (bf16, M5 Max) a text
decision takes 0.18-0.32 s for 46-139 prompt tokens, the 20-option tournament (3 passes, 364
tokens) 0.87 s, and an image decision 1.4 s (0.8 s of it in the vision tower; 2.5 s with two
images); on the CPU 0.4-0.9 s for text and 9-17 s with images.

### Image decisions

`autotrust/JEV-27B-VL` (Qwen3.8-27B with its vision tower, unchanged, plus JEV-27B's System 1 as
an unmerged `adapter_vllm/` LoRA with an `lm_head` readout) and JEV-9B's `vl/` bundle (the same
adapter as `adapter_vllm/` with the layer names of the multimodal class, for the unmodified
`Qwen/Qwen3.5-9B`) decide over images too. As in the reference server (`serve_decide.py`), the
state is a list mixing text and images, and each image goes where it appears:

```text
[kind] noul
[state] Photo: <|vision_start|><|image_pad|> x 256<|vision_end|>
[question] Is this scenario one where: the image shows a wild animal?
[options]
false
true
[decision]:
```

No chat template, no system prompt: text parts as they are, an image as its vision tokens (M-RoPE
positions as in chat), then the readout, adapter, bias and temperatures of the text decisions.
The head never saw an image in training, so these decisions are zero-shot. The reference server
takes images only (no video); so does oxinfer.

```sh
# JEV-9B on the multimodal Qwen3.5-9B: only vl/ is fetched from autotrust/JEV-9B (org/name/subdir)
oxinfer --model Qwen/Qwen3.5-9B --decision-from autotrust/JEV-9B/vl --decide choice \
  --image scene.png --state "Task: pick up the red cube." \
  --question "Where is the red object relative to the blue one?" --option left --option right

# JEV-27B-VL brings its own assets; JSON requests place images anywhere in the state
oxinfer --model autotrust/JEV-27B-VL --decide-json requests.json
```

`--image` (repeatable) puts images before the state text; `--max-soft-tokens N` caps the tokens of
an image (default: the processor's limits; each token covers 32x32 pixels). In JSON (`--decide-json`,
`Engine::decide_json`, `oxi_decide`) an image part is `{"image": REF}` or `{"type": "image_url",
"image_url": {"url": REF}}`, with REF a base64 `data:` URL, an `http(s)://` URL (fetched with the
`hf` feature's client), a `file://` URL or a path; `"attachments": {"images": [...],
"max_soft_tokens": N}` is the JSON form of `DecisionRequest::attachments`. In Rust:

```rust
let d = engine.decide(
    &DecisionRequest::noul(
        serde_json::json!(["Photo: ", {"image": "photo.jpg"}]),
        "Is this scenario one where: the image shows food or cooking?",
    ),
)?;
// or: DecisionRequest::noul("caption text", "...").with_images(["photo.jpg"])
// ImageSource::Bytes(png).to_part() makes a state part from encoded bytes
```

A text-only checkpoint (the `autotrust/JEV-9B` folder itself has no vision tower) refuses images
with a message naming the multimodal base to load instead.

Agreement with the reference server's math on `transformers` + `peft` (HF processor, unmerged
`adapter_vllm`, LM-head logprobs over the verbalizer tokens + bias, per-kind temperature; f32
compute on the bf16 weights), 11 image decisions (2-3 noul, 2 score, 2- and 4-option, two
12-option and a 20-option wide-label choice; one or two images of 140-396 vision tokens, as data
URLs, `image_url` parts and paths) and the 5 text decisions of the text check:

| oxinfer (bf16 weights) | images: max \|Δp\| | max KL | text: max \|Δp\| |
|---|---|---|---|
| Qwen3.5-9B + JEV-9B `vl/`, CPU | 3.7e-6 | 1.6e-8 | |
| Qwen3.5-9B + JEV-9B `vl/`, Metal | 1.1e-3 | 5.7e-6 | 1.3e-4 |
| JEV-27B-VL, CPU | 8.3e-6 | 4.5e-10 | |
| JEV-27B-VL, Metal | 6.0e-4 | 9.4e-7 | 8.7e-5 |

Prompt token counts match the processor's exactly; the Metal differences come from its half-precision
matrix units (`gpu_exact` removes them). Text decisions and chat give the same output as before
images were supported, bit for bit.

Latency of one image decision (512x512 photo = 256 vision tokens, 298 prompt tokens; M5 Max, while
other jobs loaded the machine, so upper bounds): Qwen3.5-9B + `vl/` 1.0-1.5 s on Metal (vision tower
0.4-0.6 s, prefill 0.6-0.9 s) and 7.4 s on the CPU; JEV-27B-VL (bf16) 3.3 s on Metal (tower 0.6 s)
and 22 s on the CPU.

## Runtime interventions: neuron edits, steering, abliteration, SAE features

oxinfer can change what a model computes while it runs, without touching the weights: edit
chosen MLP neurons, add steering vectors to the residual stream, project a direction out of it
("abliteration" at run time), and read or steer sparse-autoencoder (SAE) features. It can also
make steering vectors itself, from two sets of prompts. Everything is reachable from the CLI,
the library (`Engine::set_interventions`, `oxinfer::intervene`) and the C API
(`oxi_engine_set_interventions`). Interventions are set per request: changing a strength
between two messages rebuilds a small set of hooks, not the model, and GPU decoding keeps its
compiled plan. The shared code is in `src/intervene.rs` and `src/sae.rs`, wired into Qwen3.5 /
3.8 (dense and MoE), Qwen3, Llama 3 and Gemma 4 / 3.

**Where the hooks are.** Layers are 0-based, as in the checkpoint's module names
(`model.layers.14`).

| Hook | Meaning |
|---|---|
| neuron `L:n` | element `n` of the input of `layers.L.mlp.down_proj`, that is `act(gate) * up`. MoE layers: per routed expert (`"expert": 3`) or the shared expert (`"expert": "shared"`). |
| `post` of layer L | the layer's output, `hidden_states[L + 1]` in transformers. This is where llama.cpp control vectors and Qwen-Scope SAEs (`resid_post`) act. |
| `mid` of layer L | the residual after the attention (or Gated DeltaNet) block was added |
| `pre` of layer L | the layer's input: the output of layer L - 1, or the embeddings for layer 0 |

**Where they apply.** `positions` is `all` (the default), `prompt` or `generated`. `generated`
covers the last prompt token and every generated token, so every position whose output is
sampled. `prompt` covers the prompt tokens before that.

### The spec

The spec is a JSON object. It goes to `--interventions FILE`, to the engine option
`interventions`, or to `oxi_engine_set_interventions`. The library type is
`oxinfer::intervene::Spec`.

```json
{
  "neurons": [
    {"layer": 14, "neuron": 8949, "dose": 1.0, "unit": "resid_median"},
    {"layer": 23, "neuron": 4790, "op": "scale", "value": 2.0},
    {"layer": 18, "neuron": 6052, "op": "zero"},
    {"layer": 5, "neuron": 7, "expert": "shared", "op": "clamp", "min": -1, "max": 1, "positions": "generated"}
  ],
  "steer": [{"file": "tone.safetensors", "layer": 12, "alpha": 1.5, "point": "post"}],
  "ablate": [{"file": "refusal.safetensors", "direction_layer": 12}],
  "sae": [{"sae": "Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50/layer12.sae.pt",
           "features": [{"id": 10155, "op": "clamp", "value": 4.0}]}],
  "sae_inspect": [{"sae": "Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50/layer12.sae.pt", "top_k": 5}]
}
```

- **Neuron edits.** `op` is one of:
  - `scale` multiplies the activation by `value`;
  - `zero`;
  - `add` adds `value`; `dose` is shorthand for this;
  - `set` replaces it with `value`;
  - `clamp` limits it to `[min, max]`.

  `"unit": "resid_median"` multiplies `value` by `median_t |h_L,t| / |W_down[:, n]|`. Here `h_L`
  is the output of layer L, measured on the unedited model over positions 1 onward (position 0,
  an attention sink with a huge norm, is skipped). With this unit, a dose of 1 moves the
  residual by the typical norm of that layer's output.

  The median comes from one unedited pass over the first prompt the edits see, and is then kept.
  You can instead measure it on a text of your choice (`--calibrate-text T`,
  `Engine::calibrate(text)`) or give it outright with `"resid_median": F`.

  The edit is added at every position (`add`, not `set`). Even-length medians average the two
  middle values (numpy's convention). The reference script applies the same edit to the same
  tensor through a PyTorch pre-hook on `down_proj`.
- **Steering.** Adds `alpha * v`. `v` is either an inline `vector` (with a `layer`) or comes from a
  `file`. With a file you get every layer it holds, or only `layer` / `layers`. `normalize: true`
  scales `v` to unit norm first, so `alpha` becomes the norm of what is added.
- **Ablation.** Applies `x -= (x . r) r` with `r` normalized. It runs on the embeddings and after
  every attention and MLP block, in all layers (or `layers`). This is Arditi et al.'s directional
  ablation; FailSpy's and mlabonne's tools hook the same places. Several directions are
  orthonormalized together.

  At one point, added vectors go in first and then directions are projected out. So an ablated
  direction also removes its part of a steering vector, just as a hook on the next layer's input
  would.
- **SAE features.** `op` is one of:
  - `clamp` sets the activation to `value` (Neuronpedia's clamp);
  - `scale` multiplies it;
  - `zero`;
  - `add` adds `value * W_dec[f]`, whatever the activation (SAELens / Neuronpedia / Qwen-Scope
    steering).

  The SAE's error term is kept: only `(a' - a) * W_dec[f]` is added to the residual. All edits of
  one SAE read the same encoding of the incoming residual. `sae_inspect` records the `top_k` most
  active features of every position: `Engine::sae_features()`, `oxi_engine_sae_features_json`,
  printed by the CLI.

**Vector files.** These are read by `steer` / `ablate` and written by `--make-direction`:
- **safetensors** with 1-D tensors `layers.N` (f32/bf16/f16). `layers.N` is the output of layer N.
  A single `[layers, hidden]` or `[layers + 1, hidden]` matrix is also accepted; the second form
  follows transformers' `hidden_states` and its row 0 is dropped. So is a lone 1-D vector.
- **GGUF control vectors**, as llama.cpp `--control-vector` and repeng's `export_gguf` write them:
  `direction.N` is added to the output of layer N, exactly as llama.cpp applies it. The same
  vectors as GGUF and as safetensors give bit-identical logits.
- **PyTorch `torch.save`** state dicts with the same names.

**SAEs.** Two formats load:
- **Qwen-Scope** (`Qwen/SAE-Res-*`, one `layerN.sae.pt` per layer plus the repo's `config.json`;
  TopK, `topk(relu(x W_enc^T + b_enc), k)`, `resid_post`).
- **SAELens directories** (`cfg.json` + `sae_weights.safetensors`; ReLU, JumpReLU or TopK,
  `apply_b_dec_to_input`; only `normalize_activations: none`).

Qwen-Scope has SAEs for Qwen3.5-2B-Base (32K features, k 50 or 100, 537 MB per layer), 9B-Base,
27B (80K features, the one Neuronpedia lists as `qwen3.5-27b / 31-qwenscope-res-80k`) and
35B-A3B-Base. There are none for 0.8B.

### Command line

```sh
# The user-reported "evil persona" edit on Qwen3.5-4B (layers 0-based, as in the module names)
oxinfer -m Qwen/Qwen3.5-4B -p "Write a short poem about the sea." \
  --neuron-edit 14:8949,dose=1,unit=resid_median --neuron-edit 13:7956,dose=-1,unit=resid_median

# A refusal direction from two prompt sets (one per line, or .json / .jsonl / .csv), then
# runtime abliteration with the layer-12 direction, and the opposite: activation addition
oxinfer -m Qwen/Qwen3.5-2B --make-direction harmful.json,harmless.json -o refusal.safetensors
oxinfer -m Qwen/Qwen3.5-2B --ablate refusal.safetensors:12 -p "..."
oxinfer -m Qwen/Qwen3.5-2B --steer refusal.safetensors:1.0 --steer-layers 12 -p "..."

# A llama.cpp / repeng control vector, generated tokens only
oxinfer -m Qwen/Qwen3.5-2B --steer happy.gguf:2 --positions generated -p "Describe a bus stop."

# SAE: inspect, then steer (the SAE file is fetched from Hugging Face)
oxinfer -m Qwen/Qwen3.5-2B-Base --raw -p "I adopted a cat." \
  --sae Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50/layer12.sae.pt --sae-inspect 5
oxinfer -m Qwen/Qwen3.5-2B-Base --raw -p "My favorite thing to do is" \
  --sae Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50/layer12.sae.pt --sae-feature 10155:add=4

# Pooled layer outputs of one prompt (layers.N, for your own analysis)
oxinfer -m Qwen/Qwen3.5-0.8B -p "..." --capture-activations acts.safetensors --capture-pool mean
```

Other flags:
- `--interventions FILE|JSON` takes the full spec.
- `--steer-point pre|mid|post` and `--ablate-layers A-B` place the hooks.
- `--sae-layer N` overrides the SAE's layer.
- `--capture-pool last|mean` pools the captured layer outputs. `last` is the last prompt token; `mean`
  averages positions 1 onward.

`--make-direction` wraps every prompt in the chat template unless `--raw` is given. It writes
`mean(first set) - mean(second set)` per layer, not normalized. With `alpha` 1, steering then
moves the second set's mean onto the first's.

### Library and C API

```rust
use oxinfer::engine::{Engine, EngineOptions, Flow};
use oxinfer::intervene::{Pool, Spec};
let mut engine = Engine::load("Qwen/Qwen3.5-2B", &EngineOptions::default())?;
// A direction from two prompt sets, saved for later use.
let d = engine.make_direction(&harmful, &harmless, Pool::Last, true)?;
oxinfer::intervene::save_vectors("refusal.safetensors".as_ref(), &d, &[])?;
// Change the interventions between requests: no reload.
for alpha in [0.5, 1.0, 2.0] {
    engine.set_interventions(Some(Spec::from_json(&format!(
        r#"{{"steer":[{{"file":"refusal.safetensors","layer":12,"alpha":{alpha}}}]}}"#
    ))?))?;
    engine.reset();
    let r = engine.send("Write a haiku about tea.", &mut |_| Flow::Continue)?;
}
engine.set_interventions(None)?;
```

`Engine::capture` returns the pooled layer outputs of a prompt list, and `Engine::calibrate`
measures the `resid_median` unit on a text. `examples/steer.rs` runs a prompt file under several
specs and strengths and writes the replies as JSONL; it produced the refusal and tone numbers
below.

The C API mirrors the library:
- `oxi_engine_set_interventions(engine, json)` (NULL clears) and
  `oxi_engine_interventions_json`;
- `oxi_engine_calibrate(engine, text)`;
- `oxi_engine_sae_features_json`;
- `oxi_engine_make_direction(engine, {"positive": [...] | "file", "negative": ..., "pool", "chat", "output"})`;
- the load option `interventions=JSON|@FILE`.

### Agreement with PyTorch hooks

`tools/ref_intervene.py` applies the same JSON spec to a transformers model with forward hooks
and compares logits. The hooks are its own implementation:
- a pre-hook on `down_proj` for neuron edits;
- output hooks on the layer and on the attention block for steering;
- Arditi's three hook sites for ablation;
- the release's own `topk(relu(...))` for Qwen-Scope.

Every row below compares the last prompt position, with oxinfer on the CPU against float32
PyTorch:

| Model | Interventions | max \|logit diff\| | KL | without interventions |
|---|---|---|---|---|
| Qwen3.5-0.8B | 4 neuron edits (scale, zero, dose resid_median, clamp), steering at 2 layers + a `mid` hook on generated positions, ablation in all layers | 2e-5 | 5e-8 | 4e-5 |
| Qwen3.5-4B | the two-neuron dose edit above | 4e-5 | 1e-8 | 6e-5 |
| Qwen3.5-2B-Base | Qwen-Scope layer 12: clamp, add, zero (TopK, error term kept) | 4e-5 | 3e-8 | |
| Qwen3-0.6B | neuron edits, steering (post, mid), ablation | 7e-5 | 5e-8 | |
| Gemma 4 E2B | neuron edits incl. resid_median, steering (post, mid), ablation | 2.2e-4 | ~0 | 1.3e-4 |

The `resid_median` values matched to 4 digits (for example L13 9.1125, L14 10.1972 on the
4B prompt).

Greedy generation on Qwen3.5-4B with the dose edits produced the same 40 of 40 tokens as
transformers' `generate` under the same hooks, with oxinfer decoding on the GPU.

Qwen-Scope inspection agreed exactly: the top-5 features at all 25 positions of a sentence had
the same ids in the same order as the release's encoder, with activations equal to the 2
printed decimals.

GPU decode against the CPU (`OXINFER_GPU_DECODE_CHECK=1`, every step) gave the following KL:

| Model and interventions | KL |
|---|---|
| Qwen3.5-0.8B, all of the above | ≤ 2e-12 |
| Qwen3.5-0.8B, ReLU and JumpReLU SAE features | ≤ 2e-12 |
| a Qwen3.5 MoE test model, routed and shared-expert neuron edits | ≤ 1e-11 |
| Qwen3-0.6B | ≤ 3e-11 |
| Gemma 4 E2B | ≤ 1.3e-10 (1.2e-11 without interventions) |

### Results

**The Qwen3.5-4B persona edit:**
- **Edit:** +1 dose on layer 14 neuron 8949 and -1 on layer 13 neuron 7956.
- **Effect on a poem:** a gentle sea poem turns into a menacing one.
- **Effect on 12 neutral prompts** (introductions, advice, a birthday party plan, recipes; 120
  tokens each):
  - Replies containing hostile vocabulary went from 2 to 6.
  - Hostile words in total went from 2 to 20.
  - One reply proposed a "hostile takeover" as a party plan.
- **Calibration:** the medians come from the first prompt, on the unedited model (for the poem
  prompt: L13 9.11, L14 10.20).

**Runtime abliteration** (Qwen3.5-2B):
- **Direction:** the difference of means between 128 harmful and 128 harmless training
  instructions (Arditi et al.'s public splits), taken at the last token of the chat-templated
  prompt. Making it took 13 s.
- **Choosing a layer** (32 validation prompts, layer-L direction ablated everywhere):

  | Direction from | Refusal rate |
  |---|---|
  | no ablation | 81% |
  | layer 6 | 56% |
  | layer 8 | 38% |
  | layers 10 to 18 | 0% |

- **Held-out test** (100 harmful prompts, 64 tokens each, layer-12 direction): refusals fell from
  89% to 3%.
- **Held-out harmless prompts** (100):
  - ablation left refusals at 2% → 1%, and the replies stayed fluent and on topic;
  - adding the same direction at layer 12 (activation addition, alpha 1) made the model refuse
    92% of them.
- **How refusals were counted:** refusal phrases in the first 300 characters. The completions are
  not reproduced here.

**Tone steering** (Qwen3.5-2B, a happy-minus-sad direction from 24 + 24 prompts, layer 12, 10
neutral "two sentences about X" prompts):
- **alpha +1:** enthusiastic replies with exclamation marks and emoji ("the ultimate two-wheeled
  joyride").
- **alpha -1:** bleak replies ("its metal frame and chain eventually rust and become useless").
- **alpha ±2:** the text degenerates.

**SAE features:**
- **Inspection** finds concept features. For example, feature 10155 at layer 12 of the 2B-Base
  SAE fires on "cat", "kitten" and "Cats".
- **Single-feature steering** on the 2B base model nudged completions mildly (adding 3 to 7 times
  the unit decoder vector at generated positions). Larger strengths (10 and up, against residual
  norms of a few units at that layer) broke the text. Concept steering with one Qwen-Scope
  feature is weak at this size; the 9B / 27B SAEs are the ones Qwen's demo uses.

**The papers:**
- **H-Neurons** (arXiv 2512.01797) scale the `down_proj` input of selected neurons by
  alpha in [0, 3] (`op: scale`). They publish neither neuron lists nor Qwen results, so there is
  nothing to reproduce directly.
- **Perturbation Probing** (arXiv 2604.27401) scales the `W_down` columns of its Qwen3.5-2B
  neurons, for example L23:n4790, L18:n6052 and L14:n200 (2x amplification, zeroing for
  ablation). Scaling a column is the same as `op: scale` on that neuron. Their full 2B lists are
  not published.

### What runs where, and what it costs

| | CPU | Metal prefill | Metal whole-pass decode |
|---|---|---|---|
| neuron edits (dense, shared, routed experts) | yes | yes (hooks run between the GPU matmuls; MoE layers with routed-expert edits keep their experts on the CPU) | yes |
| steering, ablation | yes | yes | yes |
| SAE ReLU / JumpReLU features, SAE `add` | yes | yes | yes (up to 64 directional items per hook point) |
| TopK SAE clamp / scale / zero, SAE inspection, capture | yes | yes | no: those decode steps run on the CPU |

Every combination runs. When the GPU graph cannot take a hook, decoding falls back to the CPU on
its own, and the CLI says so when it loads the spec.

Decode speed on Qwen3.5-2B (bf16, M5 Max, 200 tokens, best of 3):

| Spec | Metal | CPU |
|---|---|---|
| none | 130.4 tok/s | 66.9 tok/s |
| steering at 1 layer | 130.4 | 66.6 |
| ablation at 49 hook points | 125.8 (-3.5%) | 66.1 |
| 3 neuron edits | 129.9 | 66.5 |

A TopK SAE clamp on 2B-Base moves decoding to the CPU: 124.6 → 63.0 tok/s, which is about the
CPU's own speed.

### Limits

- **Core ML (ANEMLL) models** refuse interventions.
- **Speculative decoding.**
  - Qwen MTP drafts come from the unedited MTP head, but the edited main model verifies every
    draft, so the output follows the edited model.
  - The Gemma drafter path of the CLI is not covered.
- **SAEs.**
  - Only residual-stream outputs (resid_post) and `normalize_activations: none` are supported.
  - There is no GemmaScope `.npz` loader.
  - Weights are loaded as f32: 537 MB per Qwen-Scope 2B layer.
- **Making directions.**
  - Capture runs on the CPU paths (prefill matmuls still use the GPU).
  - Pooling is last-token or mean only. Arditi's per-position candidates and his
    bypass/induce/KL layer selection are not built in; the layer sweep above was done with
    `examples/steer.rs`.
- **Calibration.** The `resid_median` unit is measured once, on the first prompt (or the
  calibration text), and kept until `Engine::clear_calibration`.

## Image generation

A diffusers folder (one with `model_index.json`) switches oxinfer to text-to-image with Stable
Diffusion 1.x / 2.x, which includes SD 1.5, SD 2.1 and the one-step SD-Turbo (`src/sd/`):

```sh
./target/release/oxinfer --model models/sd-turbo -p "a red fox in a snowy forest" -o fox.png
./target/release/oxinfer --model models/sd-v1-5 -p "a watercolor lighthouse" --negative "blurry" \
    --steps 25 --guidance 7.5 --sampler euler_a --size 512x512 --seed 7 -o lighthouse.png
```

With `--init-image FILE` the same models edit an image instead (image-to-image, inpainting with
`--mask`, instruction edits with InstructPix2Pix): see [Editing images](#editing-images).

Turbo models (recognised by name) default to 1 step without guidance, others to 25 Euler-ancestral
steps at guidance 7.5. The CLIP tokenizer and text encoder, the UNet (ResNets, spatial transformers
with GEGLU, skip connections), the VAE decoder and the Euler samplers are implemented here.
Activations are channels-last, so every convolution unfolds into the same matmul the language models
use: the SIMD kernels apply on every CPU, and GPU backends run convolutions directly as an implicit
GEMM (`Backend::conv2d`: the kernel gathers each pixel's neighbourhood itself; Metal measured, CUDA
compile-checked, WebGPU in the browser). Weights are converted from fp16 to bf16 at load.

`tools/ref_sd.py` runs diffusers on the same noise and compares every stage (`OXINFER_SD_DUMP=dir`).
With the weights rounded to bf16 on both sides the outputs agree to about 1e-6 (79 dB PSNR on
the SD-Turbo image, 75 dB on SD 1.5 after 4 guided steps), so the only difference from the fp32
reference is the bf16 weight storage (47 dB).

512x512 on the M5 Max:

| | Text | UNet pass | VAE decode | SD-Turbo image | SD 1.5, 25 steps + guidance |
|---|---|---|---|---|---|
| CPU only (18 threads) | 0.08 s | 0.91 s | 2.87 s | 3.9 s | |
| Metal | 0.08 s | 0.37 s | 0.66 s | 1.1 s | 16 s |
| Browser (Chrome, WebGPU, one CPU thread) | 0.2 s | 1.1 s | 1.9 s | 3.2 s | |

With a GPU, convolutions, attention and the group and layer norms run on it: each norm is fused
into the convolution or projection that consumes it (a statistics kernel, then normalization and
SiLU applied while the input is loaded), and attention runs its two products as GPU matmuls with a
softmax kernel between them. What stays on the CPU is element-wise: residual adds, the GEGLU gate,
upsampling and skip concatenation. `OXINFER_GPU_CHECK=1` recomputes every device op on the CPU and
prints the worst relative error per kind (about 4e-6 for attention, 1e-6 for the fused norms, on
Metal and WebGPU). SDXL, SD 3 and Flux use different architectures and are not supported.

**LoRA.** `--lora FILE[:SCALE]` (repeatable, scale 1 by default) merges a LoRA into the UNet and
text-encoder weights at load, `W += scale * alpha / rank * up @ down` in f32 before the bf16
conversion, so generation runs at the base model's speed (`src/sd/lora.rs`). Convolution LoRAs
(LoCon: a `[rank, in, 3, 3]` down and a `[out, rank, 1, 1]` up) are merged the same way. Both
common layouts are read: kohya / A1111 (`lora_unet_down_blocks_0_..._to_q.lora_down.weight`,
`.alpha`, `lora_te_text_model_...`) and diffusers / PEFT (`unet.<module>.lora_A.weight`, `.lora.down`,
`text_encoder.`, with the alpha from the `lora_adapter_metadata` header). Keys that match no weight
of the model, and LyCORIS kinds other than plain LoRA / LoCon (LoHa, LoKr, DoRA), are skipped and
reported; a file that matches nothing, or whose shapes belong to another base model, is an error.
`FILE` may also be a Hugging Face path, `org/name/file.safetensors` (or `org/name` when the repo
holds one safetensors file), fetched into the cache like models; or download it yourself:

```sh
curl -L -o lcm-lora.safetensors \
    https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/resolve/main/pytorch_lora_weights.safetensors
./target/release/oxinfer --model models/sd-v1-5 --lora lcm-lora.safetensors \
    --steps 4 --guidance 1 -p "a red fox in the snow" -o fox.png  # Euler ancestral (default)
./target/release/oxinfer --model models/sd-v1-5 --lora KoRo8888/ufotable_artstyle_locon_v1:0.8 \
    -p "a girl standing in a forest, anime screencap" -o forest.png
```

Against diffusers' `load_lora_weights` + `fuse_lora` (`tools/ref_sd.py --lora`, bf16 weights on
both sides), on SD 1.5: LCM-LoRA (kohya keys, 278 UNet modules including convolutions), 4 Euler
steps, 79.6 dB; the ufotable LoCon (kohya, 278 UNet + 72 text-encoder modules) at 0.8, 20 steps
at guidance 7.5, 78.4 dB, and the same file re-saved by diffusers in the PEFT layout gives the
identical image; both together (LCM 1.0 + ufotable 0.6), 82.3 dB. Scale 0 reproduces the base
model bit for bit. In the library, `sd::Pipeline::load_with_loras(dir, &[LoraSpec { path, scale }],
report)` does the same (`examples/sd_lora.rs`); in the browser build, `web_sd_load` takes the
LoRA files (`sdLoad(dir, [{ path, scale }])` in `oxinfer.js`).

## Editing images

Give a Stable Diffusion pipeline an image and it edits it instead of starting from noise. What
the edit does follows from the model (the UNet's input channels):

| Model | UNet input | `--init-image` (+ prompt) | `--mask` |
|---|---|---|---|
| timbrooks/instruct-pix2pix, vinesmsuic/magicbrush-jul7 | 8 (latents + image latents) | **instruction edit**: the prompt says what to change | not used |
| stable-diffusion-v1-5/stable-diffusion-inpainting (and other SD inpainting models) | 9 (latents + mask + masked-image latents) | needs a mask | repaints the white part |
| any text-to-image model (SD 1.x / 2.x, SD-Turbo, + LoRAs) | 4 | image-to-image at `--strength` | inpainting by blending |

**Editing with a prompt: InstructPix2Pix.** For "change this picture like so", use
InstructPix2Pix. With the SD 1.5 LCM-LoRA (InstructPix2Pix is a fine-tuned SD 1.5, so the LoRA
fits it) an edit takes 4 UNet passes instead of 60, and the edits look as good:

```sh
./target/release/oxinfer --model timbrooks/instruct-pix2pix --lora latent-consistency/lcm-lora-sdv1-5 \
    --init-image cabin.png -p "make it winter, covered in snow" --steps 4 --guidance 1 -o snowy.png
# Without the LoRA: 20 Euler-ancestral steps, three guided UNet passes each
./target/release/oxinfer --model timbrooks/instruct-pix2pix --init-image cat.png \
    -p "turn the cat into a tiger" --image-guidance 1.5 --guidance 7.5 -o tiger.png
```

`--guidance` (default 7.5) is how strongly the result follows the instruction and
`--image-guidance` (default 1.5) how closely it keeps to the input: each step runs the UNet on
(instruction, image), (no text, image) and (no text, no image) and combines them as diffusers'
`StableDiffusionInstructPix2PixPipeline` does, `uncond + g * (text - image) + ig * (image -
uncond)`. Guidance at or below 1 (as with the LCM-LoRA) turns both off: one pass per step. The
image conditioning is the mode of the VAE encoder's latent distribution, unscaled, concatenated
to the latents.

InstructPix2Pix tends to restyle the whole picture ("add a red car" painted the cabin red in our
test); MagicBrush (`vinesmsuic/magicbrush-jul7`, InstructPix2Pix fine-tuned on human-annotated
local edits, same architecture, CreativeML OpenRAIL-M, PyTorch `.bin` weights which load directly)
added the car and left the rest alone. It is a drop-in alternative at the same 20 guided steps; the
LCM-LoRA does not carry over to it (its scheduler config uses "leading" timesteps, and the 4-step
result was unusable).

**Image-to-image.** With a text-to-image model the image is encoded, noised to `--strength`
(default 0.8) and denoised towards the prompt: as diffusers' `StableDiffusionImg2ImgPipeline`,
the last `int(steps * strength)` steps of the schedule run, starting from `latents + sigma *
noise` at the first of them. Turbo models default to just enough steps for one to run (strength
0.5: 2 steps, 1 run):

```sh
./target/release/oxinfer --model models/sd-turbo --init-image cabin.png --strength 0.5 \
    -p "a cabin in a meadow at sunset, golden hour" -o sunset.png
./target/release/oxinfer --model models/sd-v1-5 --init-image cabin.png --strength 0.6 \
    -p "an oil painting of a cabin in a meadow, van gogh style" -o vangogh.png
```

**Inpainting.** `--mask FILE` marks what to repaint in white (Pillow's luma, binarized at 0.5;
any size, scaled like the image). An inpainting UNet sees the mask and the latents of the image
with the masked part zeroed at every step (diffusers' `StableDiffusionInpaintPipeline`); with a
text-to-image UNet the unmasked area is reset after each step to the input's latents noised to the
next level (diffusers' legacy approach: works, but the seams are weaker). `--strength` defaults to
1 (start from noise in the masked area); below 1 it starts from the noised image as in img2img.

```sh
./target/release/oxinfer --model stable-diffusion-v1-5/stable-diffusion-inpainting \
    --init-image cabin.png --mask cabin_mask.png -p "a red barn" -o barn.png
```

**Sizes.** Without `--size`, an edit runs at the model's area (512x512 for SD 1.5, 768x768 for
SD 2.x) with the input's aspect ratio, each side a multiple of 64 (a 768x512 photo runs at
640x448); with `--size WxH`, at that size. The image and the mask are scaled to cover it (Pillow
bicubic) and centre-cropped; the output has the working size. `--image FILE` is accepted in place
of `--init-image` (the flag the language models use); repo ids of diffusers pipelines download
like other models (`model_index.json`, configs, tokenizer and the fp16 safetensors of each
component; the safety checker and the single-file checkpoints are skipped).

**Library.** `sd::Pipeline::generate_from(&params, &EditInput, progress)` covers all three;
`img2img`, `inpaint` and `edit` (InstructPix2Pix) are shorthands, `edit_kind` says what an input
does with a given model, `edit_size` which size it runs at (`params.width` / `height` 0 pick it),
and `cancel_flag` stops a running generation from another thread. Images are
`video::RgbImage`; with the `oxideav` feature, `EditInput::from_picture` /
`with_mask_picture` take oxideav `Picture`s (any pixel format `Picture::to_rgb` converts), and the
result converts back with `Image::to_video_frame`. `examples/sd_edit.rs`:

```rust
use oxinfer::sd::{EditInput, Params, Pipeline, sched::Sampler};
oxinfer::gpu::init(&oxinfer::platform::detect(), oxinfer::platform::DevicePref::Auto);
let pipe = Pipeline::load(&oxinfer::hf::resolve("timbrooks/instruct-pix2pix", None)?)?;
let image = oxinfer::media::Media::new().load_image("cabin.png".as_ref())?;      // a video::RgbImage
let edit = EditInput::new(image).with_image_guidance(1.5);    // or .with_mask(m), .with_strength(s)
let params = Params {
    prompt: "make it snowy".into(), negative: String::new(), steps: 20, guidance: 7.5, seed: 1,
    width: 0, height: 0, sampler: Sampler::EulerAncestral,           // 0 x 0: Pipeline::edit_size
};
let (img, timings) = pipe.generate_from(&params, &edit, &mut |i, n| eprint!("\r{i}/{n}"))?;
```

**C API.** `oxi_sd_load(model, "device=auto;lora=FILE:SCALE", progress, user)`, then
`oxi_sd_generate(s, prompt, options_json, on_image, on_step, user)` with `options_json` such as
`{"init_image": "in.png", "mask": "mask.png", "strength": 0.8, "image_guidance": 1.5, "steps": 4,
"guidance": 1, "seed": 1}` (text to image without `init_image`), or `oxi_sd_edit_rgb(s, prompt,
rgb, width, height, mask_rgb_or_NULL, options_json, ...)` for pixels already in memory;
`oxi_sd_cancel` and returning non-zero from `on_step` stop it (see `include/oxinfer.h`).

**Browser.** Image mode has an InstructPix2Pix preset (with the LCM-LoRA, 4 steps) and, for every
model, an image and a mask upload with the strength or image-guidance setting
(`web_sd_set_input`, `sdSetInput(image, mask, { strength, imageGuidance })` in `oxinfer.js`).

**Agreement.** `tools/ref_sd.py --mode img2img | inpaint | pix2pix` runs diffusers' own
pipelines (Euler scheduler) and records their random draws by role, the initial noise and each
VAE sample, which oxinfer then reads (`OXINFER_SD_DUMP`). Weights rounded to bf16 on both sides,
Metal with `OXINFER_METAL_MMA=0` (the default matrix-unit path rounds activations to half:
about 3e-4 relative, 62-67 dB on the same runs), 512x512, generated test images:

| Edit | Encoder mean | First UNet output | Final latents | Image (rel L2, 8-bit PSNR) |
|---|---|---|---|---|
| InstructPix2Pix, 6 steps, guidance 7.5, image guidance 1.5 | 9.1e-7 | 7.5e-6 | 6.1e-6 | 4.2e-6, 85.8 dB |
| InstructPix2Pix + LCM-LoRA, 4 steps, guidance 1 | 2.1e-6 | 8.3e-7 | 3.1e-6 | 5.8e-6, 86.6 dB |
| SD 1.5 img2img, strength 0.6, 10 steps, guidance 7.5 | 9.1e-7 | 8.0e-6 | 5.7e-6 | 1.5e-5, 79.0 dB |
| SD-Turbo img2img, strength 0.5, 2 steps | 9.1e-7 | 1.7e-6 | 2.8e-6 | 1.7e-5, 77.9 dB |
| SD 1.5 inpainting UNet, strength 1, 10 steps, guidance 7.5 | 1.2e-5 (masked image) | 8.1e-6 | 2.3e-5 | 2.2e-5, 80.0 dB |
| SD 1.5 inpainting UNet, strength 0.75 | 9.1e-7 | 9.9e-6 | 1.3e-5 | 8.8e-6, 81.3 dB |
| SD 1.5 (text-to-image UNet) mask blending, 10 steps | 9.1e-7 | 7.9e-6 | 9.1e-6 | 5.7e-6, 86.1 dB |
| MagicBrush (fp32 `.bin`, "leading" timesteps), 6 steps, guidance 7.5 | 8.1e-7 | 7.5e-6 | 1.0e-5 | 1.1e-5, 84.3 dB |

The encoder's stride-2 convolutions pad one zero row and column after the input only
(`Downsample2D(padding=0)`); `ConvGeom::pad` carries that to the GPU kernels, so the encoder's
convolutions, attention and group norms take the same Metal / CUDA / WebGPU paths as the decoder
(`OXINFER_GPU_CHECK=1`: worst 1.5e-6 relative for the fused norms with `OXINFER_METAL_MMA=0`).

**Timings**, 512x512 on the M5 Max (encode = VAE encoder; the rest as for text to image):

| | Encode | UNet pass | Decode | Whole edit |
|---|---|---|---|---|
| Metal: InstructPix2Pix + LCM-LoRA, 4 steps | 0.43 s | 0.35 s | 0.65 s | 2.5 s |
| Metal: InstructPix2Pix, 20 steps x 3 passes | 0.42 s | 0.33 s | 0.64 s | 21 s |
| Metal: SD-Turbo img2img, strength 0.5 (1 pass) | 0.43 s | 0.26 s | 0.59 s | 1.4 s |
| Metal: SD 1.5 img2img, strength 0.6 (15 steps x 2) | 0.43 s | 0.33 s | 0.60 s | 11 s |
| Metal: inpainting UNet, 25 steps x 2 | 0.47 s | 0.36 s | 0.72 s | 19 s |
| CPU only (18 threads; measured with another 10-core job running, so upper bounds) | 4.2-5.9 s | 3.1-5.5 s | 7.5-9.9 s | LCM edit 38 s, turbo img2img 16 s |
| Browser (Chrome, WebGPU; machine busy): InstructPix2Pix + LCM-LoRA | 3.3 s | 3.0 s | 4.8 s | 20.5 s |

On an idle machine the CPU-only UNet pass is about 0.9 s and the decode 2.9-3.4 s (see Image
generation); the encoder costs about half the decoder. The browser numbers were taken with the
machine loaded too (SD-Turbo text to image took 7.7 s there against 3.2 s quiet), and the
browser output matched the native one.

On a phone the same code runs (iOS and Android cross builds are checked; not measured on a device
for editing): iOS takes the Metal backend like macOS, Android the CPU path. At a few seconds per
CPU UNet pass, the 4-step LCM InstructPix2Pix edit is the realistic choice there; the model needs
about 2 GB of bf16 weights in memory.

**Limits.** SD 1.x / 2.x families only (no SDXL, SD 3 or Flux editing here); one image per call;
the output has the working size (multiples of 64), not the input's; no paste-back of the
unmasked pixels after inpainting (as diffusers by default: the VAE round trip slightly changes
them); the mask is binary (no feathering); InstructPix2Pix with `--strength` is not a thing
(diffusers has none: it always starts from noise); image guidance needs text guidance above 1, so
the 4-step LCM-LoRA path has no image-guidance knob. Bigger instruction editors
(Qwen-Image-Edit, FLUX.1 Kontext, Step1X-Edit, OmniGen2, BAGEL) need their own architectures.

## Janus-Pro: text to image and image understanding

DeepSeek's Janus-Pro is one autoregressive model that both draws and reads images (`src/janus/`).
A Janus checkpoint (`model_type` `multi_modality`) is recognised from its config:

```sh
./target/release/oxinfer --model deepseek-ai/Janus-Pro-1B --generate-image "a red fox in the snow" -o fox.png
./target/release/oxinfer --model deepseek-ai/Janus-Pro-7B --quant q8 -p "a watercolor lighthouse" \
    -o lighthouse.png --num-images 4 --cfg 5 --temp 1 --seed 7          # lighthouse-0.png .. -3.png
./target/release/oxinfer --model deepseek-ai/Janus-Pro-1B --image chart.png -p "Which slice is the largest?"
```

**Text to image.** The prompt goes in the reference's SFT format (`<|User|>: PROMPT\n\n<|Assistant|>:`
plus `<begin_of_image>`), then 576 image tokens (a 24x24 grid) are sampled from `gen_head` with
classifier-free guidance, `uncond + cfg * (cond - uncond)`, the unconditional sequence being the
prompt with everything between BOS and `<begin_of_image>` replaced by padding. Each sampled token
is fed back through `gen_embed` and `gen_aligner`. The conditional and unconditional sequences of
every image run as one batch through the LLaMA decoder (DeepSeek-LLM, own KV cache per sequence),
so each of the 576 steps reads the weights once whatever the number of images; with `--quant
q8|q4` the int8 decode kernels take up to 15 rows (7 images) per step. The LlamaGen VQ-16 decoder
(l2-normalized 16384 x 8 codebook, taming-style ResNet / attention decoder, 16x upsampling) is built
from the Stable Diffusion blocks, so its convolutions run on the GPU when one is active. Options
are the reference's: CFG weight 5, temperature 1 (`--temp 0` is greedy), plus top-k / top-p and
`--num-images` (`parallel_size`). `--cfg` is an alias of SD's `--guidance`; `-p`, `-o` and `--seed`
are shared with Stable Diffusion.

**Image understanding.** Each image is resized (Pillow bicubic, longer side 384), padded to a
square with the processor's grey and encoded by the SigLIP-L/16 tower plus the aligner into 576
embeddings placed at `<image_placeholder>` (one per image, or before the question); the answer is
greedy by default, as in the reference's `inference.py`, with its system prompt.

In the library:

```rust
use oxinfer::janus::{ChatOptions, ImageOptions, Janus, JanusOptions};
let jm = Janus::load("deepseek-ai/Janus-Pro-1B", &JanusOptions::default())?;
let img = jm.generate_image("a red fox in the snow", &ImageOptions { seed: 7, ..Default::default() })?;
std::fs::write("fox.png", oxinfer::png::encode_rgb(img.width, img.height, &img.rgb))?;
// several images, a progress callback per image-token step (Flow::Stop cancels), timings:
let (imgs, stats) = jm.generate_images("a lighthouse", &ImageOptions { parallel_size: 4, ..Default::default() },
    &mut |p| { eprint!("\r{}/{}", p.step, p.steps); oxinfer::generate::Flow::Continue })?;
let chart = oxinfer::media::Media::new().load_image("chart.png".as_ref())?; // or any video::RgbImage
let answer = jm.understand("Which slice is the largest?", &[chart], &ChatOptions::default(), &mut |t| {
    print!("{t}"); oxinfer::generate::Flow::Continue })?;
```

`Image::to_video_frame()` gives an oxideav `VideoFrame` + `CodecParameters` (feature `oxideav`);
`Janus::cancel_handle()` stops a generation from another thread; `generate_tokens`,
`decode_tokens`, `image_logits` (teacher-forced guided logits) and `encode_image` expose the
stages. The C API has `oxi_janus_load`, `oxi_janus_generate` (images through a callback, a step
callback that can cancel), `oxi_janus_understand` / `_rgb` and `oxi_janus_cancel`
(`include/oxinfer.h`). Weights load from `pytorch_model.bin`, sharded `pytorch_model-*.bin` with
their index, or safetensors.

Agreement with the official `janus` package in float32 on CPU (`tools/ref_janus.py` vs
`examples/janus_check.rs`, oxinfer with bf16 weights as stored, `OXINFER_METAL_MMA=0`):

| Janus-Pro-1B | max abs diff | rel L2 | |
|---|---|---|---|
| text logits after a 56-token prompt | 1.2e-4 | 7.4e-7 | same argmax |
| guided image-token logits, all 576 steps teacher-forced | 7.0e-4 | 9.9e-6 | argmax identical at every step |
| greedy generation (`--temp 0`) | | | all 576 image tokens identical |
| VQ decoder, same 576 tokens (Metal, half-precision matrix units) | 1.2e-3 | 6.8e-5 | 94.5 dB PSNR |
| SigLIP features, 1200x799 image (padding path) | 3.0e-3 | 1.9e-5 | |
| aligned image embeddings | 1.6e-3 | 1.6e-5 | |
| logits after the 628-token image prompt | 2.7e-4 | 4.2e-6 | 64 greedy answer tokens identical |

Janus-Pro-7B: text logits max diff 8.4e-5 (rel 5.5e-6), guided logits over 64 teacher-forced
steps 1.1e-3 (rel 1.5e-5), the first 64 greedy image tokens identical (the length of the reference
run). With Metal's default half-precision matrix units the 1B text logits differ by up to 0.03
(rel 5.6e-4), still with the same argmax; generation then runs on the CPU except the prompt.

One 384x384 image on the M5 Max (18 cores; 576 steps of 2 sequences, then the VQ decoder):

| | 1B CPU | 1B Metal | 7B CPU | 7B Metal |
|---|---|---|---|---|
| bf16 | 9.8 s (15.0 ms/step) | 9.4 s | 40.5 s (67.8 ms/step) | 36.1 s |
| `--quant q8` | 6.0 s (8.5 ms/step) | | 19.4 s (31.2 ms/step) | |
| `--quant q4` | 5.0 s (6.5 ms/step) | | 14.1 s (22.0 ms/step) | |
| 4 images in one batch | 25.1 s bf16, 15 s q8 | 23.2 s bf16 | 43.2 s q8 | |

Decoding steps are memory-bound (about 190 GB/s of weights), so the batch of two CFG sequences
costs the same as one, and several images share the weight reads: four images take 2.2-2.6x the
time of one. Metal runs the prompt and the VQ decoder (1.0 s CPU, 0.3 s Metal per image); the
per-step matmuls of a few rows stay on the CPU, where they are faster than one GPU submission
each. Not supported: JanusFlow (rectified-flow generation), image sizes other than 384x384,
multi-turn chat with images (one question per call), and the browser build.

## Image, video and audio input

```sh
./target/release/oxinfer --model models/gemma-4-E4B-it --quant q4 \
    --image photo.jpg --audio clip.wav --prompt "Describe the image and transcribe the audio."
./target/release/oxinfer --model Qwen/Qwen3.5-0.8B --video clip.mp4 --prompt "What happens in this video?"
./target/release/oxinfer --model models/Qwen3.8-27B --quant q8 --image a.png --image b.png -p "Compare them."
```

`--image`, `--video` and `--audio` can be repeated; attachments are placed before the text of the
user turn (or of the first turn in `--chat`): images, then videos, then audio. Gemma 4 takes all
three; Qwen3.5 / 3.6 / 3.8 checkpoints that include the vision tower (`model.visual.*`, as the
official ones do) take images and videos. Files are decoded with the pure-Rust
[oxideav](https://crates.io/crates/oxideav-core) crates: PNG, JPEG, WebP, GIF, BMP; WAV, MP3,
FLAC, AAC/M4A, Ogg Vorbis/Opus, MKV audio (resampled to 16 kHz mono); and video (below).

- **Gemma 4 vision**: aspect-preserving bicubic resize to at most 280 soft tokens
  (`--max-soft-tokens` accepts 70/140/280/560/1120), 16x16 patches, a 16-layer ViT (hidden 768)
  with axial 2D RoPE and clipped linears, 3x3 patch pooling, then the vision embedder. A 640x480
  image becomes 57x42 patches and 266 soft tokens.
- **Gemma 4 video** (as `Gemma4VideoProcessor`): 32 frames spread uniformly over the clip, each
  through the image tower at 70 soft tokens (frames are batched through the tower), written as
  `mm:ss <|image>` + soft tokens + `<image|>` per frame.
- **Gemma 4 audio**: 128-bin log-mel features (20 ms frames, 10 ms hop), two stride-2 conv layers,
  a 12-layer conformer with chunked local attention and relative position bias, then the audio
  embedder: 40 ms of audio per soft token (750 max, i.e. 30 s).
- **Qwen vision** (`qwen_vision.rs`, as `Qwen2VLImageProcessorFast` and `Qwen3_5VisionModel`):
  each side resized to a multiple of 32 pixels within the checkpoint's pixel budget (65536 to
  16.7M pixels; `--max-soft-tokens N` caps it at N tokens of 32x32 pixels), 16x16 patches with
  two temporal frames each, emitted in 2x2 merge blocks; a Conv3d patch embedding, learned
  position embeddings bilinearly resampled to the patch grid, axial 2D RoPE, 12 (0.8B) or 27
  (9B, 27B) pre-LayerNorm blocks with bidirectional attention, and a merger folding each 2x2
  block into one token. A 1024x1024 image is 4096 patches and 1024 tokens. The text model places
  the tokens with interleaved M-RoPE (time / height / width positions per rotary frequency; the
  text after an image continues from its largest side). The tower stays bf16 with `--quant`
  (int8 blocks moved the 27-layer tower's outputs by 4%), and `--save-quantized` keeps it.
- **Qwen video** (as `Qwen3VLVideoProcessor`): 2 frames per second (`--video-fps`), at least 4
  and at most 768 (`--video-frames`), resized together within a 25M-pixel budget for the clip;
  every two frames form one temporal patch, prefixed with its time (`<1.2 seconds>`) in the
  prompt.

Soft tokens replace the placeholder embeddings inside the text model (for Gemma, the
per-layer-embedding lookup uses the pad token there, as in the reference). All attention stays
causal, matching the configurations. The resize reproduces torchvision's uint8 antialiased
bicubic exactly (what the reference processors run by default; `resample.rs` also has Pillow's,
the PIL backend's), so the pixels fed to the towers are bit-identical to transformers'.

**Video files.** oxideav demuxes MP4, MOV, Matroska / WebM and AVI and decodes H.264 (Baseline
to High, CABAC, B-frames), H.265, AV1, VP8, VP9, MPEG-4 Part 2 and Motion JPEG, all in pure Rust;
every one of these was checked on ffmpeg-made clips (first frames within one level of
ffmpeg's, three for MJPEG). A directory of image files (sorted by name) is read as a video at
`--frames-fps` (default 24, the rate the processors assume). The decoder runs on one thread: a
10-s 1080p clip decodes in 8.4 s (H.264 High), 6.0 s (VP9) or 9.9 s (AV1), against 0.1 s for
ffmpeg, and frames up to the last sampled one are all decoded. Not available: other codecs
(ProRes, Theora, ...), the audio track of a video, and hardware decoding. The same attachments
go through the library (`Engine::send_with_media` with `Attachments`) and the C API
(`oxi_engine_send_media` with a JSON list); frames already decoded by oxideav go through the
library too (see [Using oxinfer from oxideav](#using-oxinfer-from-oxideav-and-other-crates)).

Agreement with transformers 5.20 (float32, CPU; `tools/ref_vl.py` runs the reference processor
and model on the same files and compares with oxinfer's `--dump-mm` / `--dump-logits`):

| Model, input | Tower output vs reference | Last-token logits | 40 greedy tokens |
|---|---|---|---|
| Qwen3.5-0.8B, 500x333 image (160 tokens) | max 2.5e-4, rel 2.4e-5 | max diff 1.9e-4 | identical |
| Qwen3.5-0.8B, 6-s 480x320 video (12 frames, 900 tokens) | max 8.9e-4, rel 2.3e-5 | max diff 2.8e-4 | identical |
| Qwen3.5-0.8B, 500x332 video (8 frames resized to 512x320) | max 6.1e-4, rel 2.8e-5 | max diff 2.2e-4 | identical |
| Qwen3.5-0.8B, 2 images + a video (1390 tokens, 9 M-RoPE spans) | | max diff 2.2e-4 | |
| Qwen3.5-9B tower (the 27-layer tower of 3.8-27B), CPU | max 6.0e-4, rel 1.3e-5 | | |
| same, Metal with the matrix units (half-precision inputs) | max 0.17, rel 2.4e-3 | | |
| Gemma 4 E2B, 500x333 image (260 soft tokens) | max 4.2e-5, rel 3.6e-6 | max diff 1.4e-4 | |
| Gemma 4 E2B, 6-s video (32 frames, 1920 soft tokens) | max 6.7e-5, rel 3.2e-6 | max diff 1.2e-4 | identical |

The pixels match bit for bit in every case, and the expanded prompts are identical. On
Qwen3.8-27B (`--quant q8`), the same video gives: "The video begins with a serene winter scene
featuring a red fox perched on a snow-covered tree branch. [...] As the video progresses, there is
a sudden transition to a completely different scene. The new scene is a watercolor painting of a
lighthouse situated on a rocky outcrop by the sea."

Tower time (M5 Max, 18 threads; load average 4-15 from other work on the machine; the Metal
column has the linear layers and attention on the GPU):

| Tower, input | CPU | Metal |
|---|---|---|
| Qwen3.5-0.8B, 1024x1024 image (4096 patches) | 1.9 s | 0.46 s |
| Qwen3.5-9B / Qwen3.8-27B (27 layers, hidden 1152), 1024x1024 image | 6.5 s | 1.6 s |
| Gemma 4 E2B, 1024x1024 image at 1120 soft tokens (9801 patches) | 12.4 s | 2.3 s |
| Gemma 4 E2B, 32 video frames of 480x320 (1920 soft tokens) | 8.0 s | 2.5 s |

## Speech to text (Whisper)

OpenAI's Whisper checkpoints in the Hugging Face layout (`model_type: whisper`) transcribe or
translate audio of any length (`src/whisper/`): openai/whisper-{tiny, base, small, medium,
large-v3, large-v3-turbo}, the English-only `.en` models, and distil-whisper (the same network
with fewer decoder layers, e.g. distil-whisper/distil-large-v3). Sizes come from `config.json`
(80 mel bins, 128 for v3), special tokens and decoding defaults from `generation_config.json`.

```sh
./target/release/oxinfer --model openai/whisper-small --transcribe talk.mp3              # text
./target/release/oxinfer --model openai/whisper-large-v3-turbo --transcribe talk.mp3 --timestamps
./target/release/oxinfer --model openai/whisper-small --transcribe interview.m4a --language fr --translate
./target/release/oxinfer --model openai/whisper-small --transcribe talk.wav --format srt > talk.srt
./target/release/oxinfer --model openai/whisper-small --transcribe talk.wav --word-timestamps --json
```

| Flag | Meaning |
|---|---|
| `--transcribe FILE` | audio to transcribe: WAV, MP3, FLAC, AAC/M4A, Ogg Vorbis/Opus, MKV audio (oxideav), resampled to 16 kHz mono |
| `--language XX` | spoken language, as a code or a name (default: detected from the first 30 s) |
| `--translate` | translate to English text |
| `--timestamps` | timed segments (always on past 30 s) |
| `--word-timestamps` | also word-level times |
| `--format F` | `text` (default), `json`, `srt`, `vtt` (`--json` is `--format json`) |
| `--initial-prompt TEXT`, `--prefix TEXT` | context for the decoder (names, spelling, style); forced start of the transcript |
| `--fallback` | openai-whisper's temperature fallback: temperatures 0, 0.2, ..., 1.0 while a window's tokens compress better than 2.4 or average below -1 log-probability; a window with a no-speech probability over 0.6 and a low log-probability is skipped as silence |
| `--condition-on-prev` | feed each 30 s window the previous text |
| `--precision f32\|f16\|bf16` | weight storage (default bf16); `--quant q8\|q4` quantizes the decoder |

What runs, following transformers' `WhisperFeatureExtractor` and
`WhisperForConditionalGeneration.generate`:

- **Features**: 400-sample periodic Hann window, hop 160, reflect padding, power spectrum,
  Slaney mel filterbank, `log10`, clamp to `max - 8`, `(x + 4) / 4`. Clips up to 30 s are
  zero-padded to 30 s first; longer audio is featurized whole (the clamp sees all of it). The DFT
  (a mixed-radix FFT of 400) and the filterbank run in f64.
- **Encoder**: two convolutions (kernel 3, the second with stride 2) with GELU, sinusoidal
  positions, pre-LayerNorm transformer layers, all through the shared matmul (SIMD kernels, GPU
  backends for the big products) and a fused attention kernel on Metal.
- **Decoder**: learned positions, causal self-attention over a KV cache, cross-attention over the
  window's encoder output (keys and values computed once per window, kept head-major and attended
  with the keys split across threads), the output projection tied to the token embedding.
- **Decoding**: greedy by default (`TranscribeOptions::temperatures` for sampling and fallback);
  language detection from the first-token distribution over the language tokens (`language` then
  holds the code and its probability); `suppress_tokens` and `begin_suppress_tokens`; the rules of
  `WhisperTimeStampLogitsProcessor` (no `<|notimestamps|>`, timestamps in pairs and increasing, a
  timestamp first, at most 1 s in, and a timestamp whenever all timestamps together outweigh every
  text token). Audio over 30 s is decoded in sequential 30 s windows, each moved to the last
  complete segment's end timestamp, as transformers' long-form generation does; `initial_prompt`
  becomes `<|startofprev|>` context; `condition_on_prev_tokens` feeds the last 223 tokens.
- **Word timestamps**: openai-whisper's alignment: the cross-attention of the checkpoint's
  `alignment_heads` over the text tokens, normalized and median-filtered, aligned to the audio
  frames by dynamic time warping, tokens grouped into words, punctuation merged. Against
  transformers' pipeline (`return_timestamps="word"`) on a 29 s LibriSpeech clip, the 60 shared
  words start within 16 ms of it on average.

In Rust, `oxinfer::whisper::Transcriber` does all of it (`examples/transcribe.rs`):

```rust
use oxinfer::whisper::{Transcriber, TranscriberOptions, TranscribeOptions};
let asr = Transcriber::load("openai/whisper-small", &TranscriberOptions::default())?;
let mut opts = TranscribeOptions::default();
opts.language = Some("fr".into());
opts.timestamps = true;
let pcm = Transcriber::load_audio(std::path::Path::new("talk.mp3"))?; // or your own 16 kHz samples
let t = asr.transcribe_with(&pcm, &opts, &mut |seg| {
    println!("[{:.2} -> {:.2}] {}", seg.start, seg.end, seg.text); // as each 30 s window finishes
    oxinfer::engine::Flow::Continue                               // Flow::Stop ends it there
})?;
println!("{} ({}, {:.1}x real time)", t.text, t.language, 1.0 / t.stats.rtf);
```

`asr.cancel_handle()` stops a transcription from another thread (it returns the finished
segments with `stopped` set); `transcribe_file` decodes a file first; `detect_language` returns
the language distribution. `TranscribeOptions` deserializes from JSON, which is how the C API
takes it: `oxi_transcriber_load`, `oxi_transcribe_file` / `oxi_transcribe_pcm` (any sample rate)
with a per-segment callback, `oxi_transcriber_cancel`, results as JSON (`include/oxinfer.h`).

**Agreement with transformers** (float32 `WhisperForConditionalGeneration` + `WhisperProcessor`,
torch 2.14, transformers 5.20; `tools/ref_whisper.py` on the samples oxinfer decoded; oxinfer with
`--precision f32 --device cpu`). Clips: LibriSpeech test samples (5.9 s; 29.4 s with timestamps),
25 s of French from a LibriVox recording (language detection), and the first 100 s of a LibriVox
reading of Woolf's *Kew Gardens* (long-form: four or five windows):

| Model | Clip | Log-mel max abs diff | Encoder max abs diff (rel L2) | First-step logits max abs diff | Greedy transcript |
|---|---|---|---|---|---|
| whisper-tiny | 5.9 s | 1.2e-5 | 5.4e-4 (6.7e-6) | 1.2e-4 | identical, 21 tokens |
| whisper-tiny | 29.4 s, timestamps | 1.6e-5 | 6.3e-3 (1.3e-5) | 9.3e-5 | identical, 99 tokens with timestamps, same 5 segment times |
| whisper-tiny | 25 s French | 3.1e-5 | 1.6e-3 (6.8e-6) | 1.5e-4 | identical, 77 tokens; `fr` detected |
| whisper-tiny | 100 s long-form | 3.0e-5 | 4.7e-3 (1.2e-5) | 1.2e-4 | identical, 348 tokens, same 21 segment times |
| whisper-small | 5.9 s | 1.2e-5 | 3.1e-3 (1.1e-5) | 3.3e-5 | identical, 22 tokens |
| whisper-small | 29.4 s, timestamps | 1.6e-5 | 2.4e-3 (5.8e-6) | 2.6e-5 | identical, 110 tokens, same 7 segment times |
| whisper-small | 25 s French | 3.1e-5 | 1.8e-3 (5.5e-6) | 3.0e-5 | identical; `fr` detected; translated: identical, 60 tokens |
| whisper-small | 100 s long-form | 3.0e-5 | 5.2e-3 (1.3e-5) | 2.7e-5 | identical, 285 tokens, same 15 segment times |
| whisper-large-v3-turbo | 100 s long-form | 2.7e-5 | 2.5e-2 (1.4e-4) | 2.5e-4 | identical, 300 tokens, same 16 segment times |
| whisper-large-v3-turbo | 25 s French | 3.2e-5 | 1.2e-2 (9.3e-5) | 5.8e-5 | identical, 29 tokens |
| distil-large-v3 | 100 s long-form | 2.7e-5 | 3.8e-2 (1.4e-4) | 7.2e-5 | identical, 350 tokens, same 21 segment times |
| whisper-base | 100 s long-form | 3.0e-5 | 2.9e-3 (7.2e-6) | 4.3e-5 | identical, 361 tokens, same 22 segment times |
| whisper-medium | 29.4 s, timestamps | 1.6e-5 | 4.7e-3 (8.7e-6) | 4.1e-5 | identical, 102 tokens, same 6 segment times |
| whisper-large-v3 | 25 s French | 3.2e-5 | 1.1e-3 (7.1e-6) | 2.7e-5 | identical, 25 tokens; `fr` detected |
| whisper-tiny.en | 100 s long-form | 3.0e-5 | 7.7e-5 (3.3e-6) | 7.7e-5 | identical, 354 tokens, same 24 segment times |

The options agree too (whisper-small): an initial prompt (88 tokens), `condition_on_prev_tokens`
over the 100 s clip (213 tokens, 11 segments), both together, and the fallback settings (no window
falls back on that clip). The log-mel difference is torch's own: its f32 STFT is off from an f64
evaluation of the same pipeline by up to 1.2e-5, where oxinfer's features are within 7e-8 (f32
rounding), so bit-identical features would mean reproducing torch's FFT rounding. Encoder
differences grow with depth (32 layers for large-v3-turbo) but stay at a relative 1e-4.

With the default bf16 weights, transcripts stay close but are not always token-identical. On the
100 s clip, whisper-large-v3-turbo in bf16 on Metal gives the f32 transcript token for token;
whisper-small differs in one word of 206, because of the bf16 decoder (bf16 encoder with an f32
decoder, or f16 weights throughout, are token-identical). A q8 decoder: 0 or 1 word of 206 on
whisper-small, 1 of 210 on large-v3-turbo; q4 decoders drift (7 words of 210 on large-v3-turbo,
50 of 206 on whisper-small): not recommended for Whisper.

**Speed** on the M5 Max, 100 s of speech (long-form), wall time including features, language
detection and every window (real-time factor = time / audio duration; lower is faster). Best of
three runs, load average 4-8 from other jobs (one core busy with a compile during the first
rows); `--threads 6` approximates a phone's core count:

| Model | Device | Encoder | Decoder | Total | RTF | Speed |
|---|---|---|---|---|---|---|
| whisper-small | CPU, 18 threads | 2.17 s | 0.66 s (299 tokens) | 2.84 s | 0.028 | 35x real time |
| whisper-small | CPU, 6 threads | 4.58 s | 0.82 s | 5.43 s | 0.054 | 18x |
| whisper-small | Metal | 0.66 s | 0.65 s | 1.32 s | 0.013 | 76x |
| whisper-small, q8 decoder | Metal | 0.54 s | 0.54 s | 1.09 s | 0.011 | 92x |
| whisper-large-v3-turbo | CPU, 18 threads | 9.66 s | 0.57 s (304 tokens) | 10.24 s | 0.102 | 10x |
| whisper-large-v3-turbo | CPU, 6 threads | 19.58 s | 0.70 s | 20.32 s | 0.203 | 5x |
| whisper-large-v3-turbo | Metal | 1.39 s | 0.58 s | 1.99 s | 0.020 | 50x |
| whisper-large-v3-turbo, q8 decoder | Metal | 1.34 s | 0.39 s | 1.76 s | 0.018 | 57x |

On Metal the encoder's attention is one fused kernel (`flash_attn64`: about 3.6 ms per
large-v3-turbo layer, against 10 ms for the GEMM + softmax + GEMM path it replaced, which other
head sizes still use); a large-v3-turbo window then splits into ~160 ms of matmuls, ~115 ms of
attention and ~100 ms of CPU-side layer norms, GELU and residual adds.

The encoder (30 s per window, 1500 positions) is the heavy part: its matmuls and attention go to
the GPU when one is active (Metal, CUDA, WebGPU); decoding is one token at a time on the CPU,
where the bf16 / q8 / q4 kernels of the language models apply. Without a GPU (iOS and Android
builds without one, `--device cpu`), everything runs on the SIMD CPU kernels. f32 and f16 weights
run on the CPU only (the GPU backends take bf16, q8 and q4).

Not supported: beam search (greedy and sampling only), batched transcription of several files at
once, voice activity detection or speaker diarization, openai-whisper's word-timing heuristics
(the punctuation merge is there, not the long-word truncation) and `hallucination_silence_threshold`;
the compression ratio of the fallback is computed with a built-in DEFLATE size estimate (greedy
LZ77, fixed Huffman codes) instead of zlib, as no deflate implementation is in the dependency tree.

## Text to speech (Orpheus)

[Orpheus TTS](https://github.com/canopyai/Orpheus-TTS) is a Llama 3.2 3B fine-tune that writes
audio as extra vocabulary tokens: 7 tokens per frame, codes of the three codebooks of the
[SNAC](https://github.com/hubertsiuzdak/snac) codec, which decodes them to 24 kHz audio (a frame is
2048 samples, 85 ms). `canopylabs/orpheus-3b-0.1-ft` is gated (accept the licence and set
`HF_TOKEN`); `unsloth/orpheus-3b-0.1-ft` is an ungated bf16 copy of it. The SNAC decoder
(`hubertsiuzdak/snac_24khz`, 80 MB, `pytorch_model.bin`) is fetched on first use.

```sh
./target/release/oxinfer --model unsloth/orpheus-3b-0.1-ft --speak "Hey there, I'm Leo. <chuckle> Nice to meet you." --voice leo -o hi.wav
./target/release/oxinfer --model unsloth/orpheus-3b-0.1-ft --speak - --quant q4 --speak-stream < text.txt   # faster than real time
./target/release/oxinfer --model hubertsiuzdak/snac_24khz --decode-tokens tokens.json -o out.wav          # SNAC alone
# voice cloning: the pretrained model, a reference clip and what it says (repeatable)
./target/release/oxinfer --model unsloth/orpheus-3b-0.1-pretrained --ref-audio me.wav --ref-text "What I say in the clip." --speak "Something new." -o clone.wav
# other languages
./target/release/oxinfer --model Our-Org-2/3b-fr-ft-research_release --speak "Bonjour, comment allez-vous ?" --voice pierre -o fr.wav
# the codec alone, any SNAC model: audio -> codes (JSON array per codebook) -> audio
./target/release/oxinfer --model hubertsiuzdak/snac_44khz --snac-encode music.wav -o codes.json
./target/release/oxinfer --model hubertsiuzdak/snac_44khz --decode-codes codes.json -o back.wav
```

Voices of the English model: tara (default), leah, jess, leo, dan, mia, zac, zoe; emotion tags
(`<laugh>`, `<chuckle>`, `<sigh>`, `<cough>`, `<sniffle>`, `<groan>`, `<yawn>`, `<gasp>`) go in the
text. Sampling follows the official package: temperature 0.6, top-p 0.8, repetition penalty 1.1
(over the prompt and the reply, as transformers and vLLM apply it), at most 1200 tokens (about
14 s; `--max-tokens` for more). The prompt is the one `orpheus_tts` builds (start of human,
`<|begin_of_text|>voice: text`, end of text / human, start of AI, start of speech) and generation
stops at end of speech. By default sampling is constrained to the tokens valid at each frame
position (plus the stop tokens), which also computes only those 4096 rows of the 157k-row output
layer; `--unconstrained` samples the whole vocabulary like the reference.

In Rust (`examples/speak.rs`):

```rust
use oxinfer::tts::{Speaker, SpeakerOptions, SpeakOptions};
let mut lo = SpeakerOptions::default();
lo.quant = oxinfer::model::Quant::Q4;
let speaker = Speaker::load("unsloth/orpheus-3b-0.1-ft", &lo)?;
let mut opts = SpeakOptions::default(); // voice, temperature, top_p, top_k, repetition_penalty, max_tokens, seed, ...
opts.voice = Some("leah".into());
let audio = speaker.speak("Hello there!", &opts)?; // Audio { samples: Vec<f32>, sample_rate: 24000 }
audio.write_wav("hello.wav")?;
// Streaming: chunks of audio while tokens are generated; Flow::Stop (or speaker.cancel_handle()) stops.
let speech = speaker.speak_stream("A longer sentence...", &opts, &mut |chunk| {
    play(chunk.samples); // chunk.start: index of its first sample
    oxinfer::engine::Flow::Continue
})?;
println!("RTF {:.2}", speech.stats.rtf());
```

### Voice cloning

The pretrained base (`canopylabs/orpheus-3b-0.1-pretrained`, ungated copy
`unsloth/orpheus-3b-0.1-pretrained`) has no named voices; it continues in the voice of
reference utterances given in the prompt. The prompt is the one of canopylabs' notebook for
that model: per reference a finished turn (start of human, `<|begin_of_text|>transcript`, end
of text / human, start of AI, start of speech, the clip's SNAC codes as audio tokens, end of
speech, end of AI), then the turn for the new text (no voice name). The clip is resampled to
24 kHz and encoded by the SNAC encoder (also in Rust, see below). The model card notes that it
was not trained for cloning, so more reference pairs give a steadier voice; 5-10 s of clean
speech per reference works. Finetuned checkpoints accept references too, but follow them less.

```rust
use oxinfer::tts::{Speaker, SpeakerOptions, SpeakOptions, VoiceRef};
let speaker = Speaker::load("unsloth/orpheus-3b-0.1-pretrained", &SpeakerOptions::default())?;
let mut opts = SpeakOptions::default();
// Encoded once, reusable (serializable): from samples at any rate, a file, or an av::AudioClip.
opts.references.push(speaker.encode_reference(&samples, 16000, "What the clip says.")?);
opts.references.push(speaker.encode_reference_file("other.wav".as_ref(), "Its transcript.")?);
// or encoded at each call: VoiceRef::from_file("What it says.", "ref.wav")
let audio = speaker.speak("Something new.", &opts)?;
```

In JSON (C API, `--speak-json`): `"references": [{"text": "...", "audio": "ref.wav"}]`, or the
objects `oxi_speaker_encode_reference()` / `oxi_speaker_encode_reference_file()` return.

**Agreement** (one 6.5 s reference, a second sentence): the SNAC codes of the clip are identical
to the `snac` package's, and the prompt (586 tokens) identical to the notebook's; greedy
generation from it matches transformers float32 for 120 of 120 tokens.

**Intelligibility and voice**: two references (a male and a female voice, 6.5 s each, made by
the finetuned model's leo and tara), the four sentences of the WER test below, sampled with the
default settings: WER 5.7% for both (3 errors in 53 words each: "riverbank", "rainforest", as for
the finetuned voices). The median pitch follows the reference: 144-167 Hz for the leo clip
(151 Hz), 183-197 Hz for the tara clip (180 Hz), against 93-132 Hz for the pretrained model
without a reference.

### Other languages

The multilingual research release (`canopylabs/3b-{fr,de,ko,hi,zh,es_it}-ft-research_release`,
gated; ungated copies `Our-Org-2/3b-*-ft-research_release`, fp32 safetensors read as bf16) uses the
same prompt format with its own voices: French pierre, amelie, marie; German jana, thomas, max;
Korean 유나, 준서; Hindi ऋतिका; Mandarin 长乐, 白芷; Spanish javi, sergio, maria; Italian pietro,
giulia, carlo (`oxinfer::tts::voices_for`). `Speaker::info()` reports the languages (from the
model card), the known voices and whether the checkpoint is a pretrained base. Greedy French
tokens match transformers float32 (150 of 150).

WER of four sentences per voice (default sampling), transcribed with `--language`:

| Model | Voices | whisper-small | whisper-large-v3-turbo |
|---|---|---|---|
| French (`3b-fr-ft`) | pierre, amelie | 8% (24 utterances, 3 seeds) | 0% (8, seed 42) |
| French | marie | 58% (12, 3 seeds) | 81% (4) |
| German (`3b-de-ft`) | jana, thomas, max | 16% (12) | 1.7% (12) |

The voice "marie" of the French research model mostly produces wrong words (the same with
transformers' greedy tokens, so it is the checkpoint); the others are intelligible, and most of
whisper-small's errors are its own (whisper-large-v3-turbo hears them correctly).

### The SNAC codec

`oxinfer::snac::Snac` loads any SNAC checkpoint: `encode(&samples)` (the reference's `encode`,
input zero-padded to a multiple of the hop times `lcm(vq_strides[0], attn_window_size)`) and
`decode(&codes)`. The 32 and 44 kHz models (four codebooks at 8 / 4 / 2 / 1 times the frame
rate, music and general audio, not used by any Orpheus model) add local windowed attention after
the encoder and before the decoder: LayerNorm, 64-dim heads attending within windows of 32 steps
with rotary positions restarting in every window. Against `snac` 1.2.1 (float32, noise off), on a
4.9 s clip (resampled for 32 / 44 kHz) and 64 frames of random codes:

| Model | Encoder codes | Decode of the clip's codes | Decode of random codes | bf16 weights |
|---|---|---|---|---|
| snac_24khz | identical (57 / 114 / 228) | 1.6e-7 (127 dB) | 2.2e-6 (118 dB) | 58 / 46 dB |
| snac_32khz | identical (52 / 104 / 208 / 416) | 2.2e-7 (126 dB) | 5.6e-6 (118 dB) | 65 / 52 dB |
| snac_44khz | identical (72 / 144 / 288 / 576) | 3.3e-7 (126 dB) | 2.3e-6 (121 dB) | 64 / 54 dB |

(max abs difference and SNR.) Encoding 5 s takes 0.08-0.15 s. The reference requires lengths that
are a multiple of the attention window; here a shorter last window attends within itself.

With the `oxideav` feature, `Audio::to_frame()` / `AudioChunk::to_frame()` give an F32 mono
`oxideav_core::AudioFrame` with its `CodecParameters` (pts in samples), and `Audio::to_clip()` an
`av::AudioClip`. `SpeakOptions` deserializes from JSON; the C API takes it that way:
`oxi_speaker_load`, `oxi_speak` (PCM, optional chunk callback), `oxi_speak_wav`,
`oxi_speaker_cancel`, `oxi_speaker_stats_json` (`include/oxinfer.h`).

Streaming decodes overlapping windows of frames as they arrive, as the official decoder does (it
decodes the last 4 frames and keeps the second: 1 frame of left context, 2 of lookahead, and it
drops the first and last frames). Here every frame is emitted and the context is an option: the
decoder's receptive field is under 3 frames each way, so `context_left` / `context_right` 3 / 3
streams exactly the full decode (max abs diff 3e-7), the default 2 / 2 differs from it by at most
1.7e-3 (59 dB SNR), and the official 1 / 2 by 2.4e-2 (27 dB). The SNAC noise blocks draw their noise per (seed, layer, absolute position), so
overlapping windows see the same noise as a whole-clip decode.

Llama 3 itself runs on the same code as Qwen3 (full attention without query / key norms, llama3
RoPE frequency scaling), so any Llama 3.x checkpoint also works as a text model (`--model` with
the Llama 3 chat template), safetensors or GGUF (see [Llama 3 GGUF](#llama-3-gguf)).

GGUF conversions of Orpheus load the same way, as a path or `org/repo:QUANT`; the audio tokens
are in the GGUF vocabulary (156940 tokens, the 28683 added ones as user-defined tokens), and the
constrained audio head reads its rows of the output layer in place, in the file's own format:

```sh
./target/release/oxinfer --model isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF:Q4_K_M --speak "Hello there!" -o hi.wav
./target/release/oxinfer --model lex-au/Orpheus-3b-FT-Q8_0.gguf:Q8_0 --speak "Hello there!" -o hi.wav
```

On the backbone's first speech-token distribution (tara, a short prompt, CPU) the Q8_0 / Q4_K_M
files match the bf16 safetensors model with the same top 10 and KL 3.4e-4 / 2.6e-3. Eight
sentences (102 words, tara, default sampling, one seed each) transcribed by whisper-small: WER
0% for Q4_K_M, 1.0% for Q8_0 ("rainforests" -> "rainforest"), 0% for the safetensors model at
q4 and q8. Decode speed for a 7 s utterance (~600 tokens, not streamed, 3 interleaved runs, load
7-18):

| Weights | CPU | Default (Metal prefill) |
|---|---|---|
| GGUF Q4_K_M (2.4 GB; Q4_K with Q6_K for some tensors) | 99-102 tok/s | 96-99 tok/s |
| safetensors `--quant q4` | 112-116 tok/s | 113-118 tok/s |
| GGUF Q8_0 | 68-69 tok/s | 69-72 tok/s |
| safetensors `--quant q8` | 68-69 tok/s | 70-72 tok/s |

Q4_K_M decodes faster than real time (82 tokens/s) but ~13% slower than oxinfer's own q4: it keeps
the output layer and half the value and down projections at 6 bits (Q6_K).

**Agreement** with the reference: transformers 5.20 float32 `LlamaForCausalLM` + the `snac` 1.2.1
package, with the token handling of `orpheus_tts` (decoder.py) and the SNAC noise disabled on both
sides (it is random in the reference):

| Check | Result |
|---|---|
| Backbone last-token logits (Orpheus 3B, 14-token text prompt) | max abs diff 3e-5, same top 10 |
| Greedy speech tokens, tara, "The weather today is sunny with a light breeze from the west." (rep. penalty 1.1), CPU | identical, 379 of 379 tokens (54 frames + stop), constrained and unconstrained |
| Same, leo, "Could you send me the report by Friday? ...", Metal prefill | identical, 316 of 316 tokens |
| Waveform of those tokens after SNAC | max abs diff 3.3e-7 / 9.5e-7 (123 dB SNR) |
| SNAC alone, 400 random frames (34 s), f32 weights | max abs diff 6.6e-6 (117 dB); bf16 weights (`snac_f32=0`) 1.4e-2 (47 dB) |
| Streaming windows 1 / 2 vs the official decoder's windows | max abs diff 3e-7 |

q8 and q4 weights change the greedy path within a few tokens (greedy decoding of audio codes is
not a stable regime anyway); their speech is as intelligible as bf16 by the measure below.

**Greedy decoding and quantized weights.** Greedy (`--temp 0`, not the default) usually ends
normally at every precision: 183 of 184 greedy runs (20 texts, 2-8 voices, bf16 / q8 / q4) did.
The failure is a fixed point: on "The weather today is sunny with a light breeze from the
west." in tara's voice the q4 path says "The weather today", then repeats one frame (all 7
codes) forever, a held sound, with end of speech at probability ~1e-8, until `max_tokens`
(171 frames). The repetition penalty cannot break it (it divides any seen token's logit by 1.1,
however often it was seen); it is not a slow drift of the end-of-speech probability, which stays
far from the top as long as speech goes on and wins in one step when the text is done. Normal
speech never repeats a whole frame (0 repeats in all other runs, sampled q4 included), so two
guards are on by default:

- `loop_guard` (3): when a frame repeats the one before it 3 times in a row, the rest of the
  utterance is sampled at temperature >= 0.6, top-p <= 0.8 (a fallback, like Whisper's); after 8
  loops it ends. On the failing case, 4 of 5 fallback seeds then speak the whole sentence and
  stop (single-frame escapes rarely recovered: 2 of 5);
- `max_secs_per_byte` (0.25): at most 3 s plus 0.25 s per UTF-8 byte of text, about 3x the
  slowest speech in any of the supported languages, so a short text cannot run for 14 s.

Sampling (the default) never looped in these tests.

**Intelligibility**: four sentences in four voices each, sampled with the default settings,
transcribed by oxinfer's whisper-small and scored with jiwer (lower case, punctuation removed):
bf16 (tara, leah, leo, zac) WER 4.7%, q4 streamed (tara, jess, dan, mia) WER 4.7% (10 errors in
212 words each). 8 of the 10 come from "river bank" transcribed as "riverbank" (a substitution
and a deletion, in all four voices); the other two are "rainforests" -> "rainforest" and
"frog" -> "fraud" (bf16), "rainforest" twice (q4): about 1% of words actually misheard.

**Speed** on the M5 Max (18 threads, other jobs running, so the best case is a little higher),
an 8.5-10 s utterance, streamed (sliding windows, 2 / 2 context). Real-time output needs
7 x 24000 / 2048 = 82 tokens/s; the real-time factor (RTF) is total time / audio duration:

| Weights | Device | Decode | SNAC | RTF | First audio | Real time? |
|---|---|---|---|---|---|---|
| bf16 | CPU | 43 tok/s | 0.7 s | 2.0 | 0.62 s | no |
| q8 | CPU | 68 tok/s | 0.9 s | 1.33 | 0.50 s | no |
| q4 | CPU | 108-128 tok/s | 0.75 s | 0.76-0.90 | 0.38 s | yes |
| bf16 | Metal | 44 tok/s | 0.9 s | 2.0 | 0.59 s | no |
| q8 | Metal | 68 tok/s | 1.0 s | 1.32 | 0.41 s | no |
| q4 | Metal | 112-128 tok/s | 0.8 s | 0.69-0.85 | 0.28 s | yes |

Decoding is one token at a time on the CPU and memory-bound (2.8B weights per token once the
output layer is cut to the audio rows), so the GPU only speeds up the prompt; q4 is the setting
that keeps up with playback here. SNAC costs about 3 ms per frame decoded once (f32 weights;
streaming decodes each frame in 5 windows, ~14 ms per 85 ms frame).

**Speculative decoding** was measured and not added: no draft gives a speed-up. Acceptance per
generated token over 2976 tokens (8 utterances; "greedy" = the draft matches the greedy token,
"sampled" = the expected accepted mass `sum min(p, q)` at temperature 0.6, top-p 0.8):

| Draft | Greedy | Sampled | Tokens per verify pass (2 drafted) |
|---|---|---|---|
| n-gram prompt lookup (n = 1-4) | 0-0.1% | 0-0.1% | 1.00 |
| the previous frame's token at the same position | 0% | 1.4% | 1.01 |
| edwindn/orpheus-1b-0.1-weightDecay-chkpt78201 (Llama 3.2 1B, same tokens) | 20% | 31% | 1.24 / 1.41 |
| edwindn/orpheus-1b-0.1-finetuned-v3 | 20% | 31% | 1.24 / 1.41 |
| Gapeleon/Orpheus-1B-pt | 26% | 38% | 1.33 / 1.52 |

Audio codes do not repeat, so lookup drafts find nothing, and the community 1B models (canopylabs
never released its smaller sizes) cost about 0.35 of the 3B per token: at best 1.5 tokens per
pass for 1.7-2x the work of one target token, a slowdown. Medusa-style heads do not exist for
Orpheus.

Not supported: batched utterances. Korean, Hindi, Mandarin and the
Spanish / Italian model use the same code but were not tested.

## Getting weights

`--model` accepts a HuggingFace repo id and fetches it on first use (resumable, pure-Rust HTTP via
[rsurl](https://crates.io/crates/rsurl)) into `~/.cache/oxinfer/hub` (or `$OXINFER_CACHE`,
or `--cache-dir`):

```sh
./target/release/oxinfer --model google/gemma-4-E4B-it --quant q4 -p "Hello"
./target/release/oxinfer --model Qwen/Qwen3.8-27B --download-only     # just fetch
```

Set `HF_TOKEN` for gated repos. The Gemma 4 checkpoints are Apache 2.0 and not gated; manual
downloads with plain `curl` work too:

```sh
M=gemma-4-E4B-it
mkdir -p models/$M
for f in config.json generation_config.json tokenizer.json tokenizer_config.json model.safetensors; do
  curl -L -o models/$M/$f https://huggingface.co/google/$M/resolve/main/$f
done
```

For 31B, fetch `model.safetensors.index.json` and the `model-0000N-of-0000M.safetensors` shards instead of `model.safetensors`.
For speculative decoding also fetch `google/gemma-4-E4B-it-assistant` (160 MB) the same way.

### The ATOnline mirror

Repo ids are looked up on the [ATOnline](https://www.atonline.com/) mirror first: its
`AI/Model:get` method (`https://ws.atonline.com/_special/rest/AI/Model:get?key=org/name`) lists a
repo's files with download URLs, small ones inlined. Repos it does not carry come from
huggingface.co. `OXINFER_HUB=mirror` or `OXINFER_HUB=hf` (`oxinfer::hf::set_hub` in the library) pins one source; the HuggingFace token is
only ever sent to huggingface.co. The browser build uses the same lookup (the mirror allows
cross-origin requests).

### GGUF files

`--model` also takes llama.cpp GGUF files, as a path or as `org/repo:QUANT` (the one file of a
GGUF repository whose name ends in `QUANT.gguf`, e.g.
`HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF:Q4_K_P`). Architectures: `qwen35`
(Qwen3.5 / 3.6 / 3.8), `qwen35moe`, `qwen3`, `llama` (Llama 3.x, and Orpheus TTS, see
[below](#llama-3-gguf)) and `gemma4` (Gemma 4 text models, dense and 26B-A4B; see
[Gemma 4 26B-A4B](#gemma-4-26b-a4b-moe)). The tokenizer and chat template come from the
GGUF metadata (the rebuilt tokenizer matches Qwen3.8's `tokenizer.json` on 305 test texts, and
Gemma 4's, a SentencePiece-style BPE with byte fallback, on this README and multilingual text), and
llama.cpp's conversions are undone at load (`gguf.rs`): norms with the 1 folded in, `ssm_a`, the
tiled Gated DeltaNet value-head order, separate 3D expert tensors, Llama's permuted query / key
rows, Gemma 4's proportional RoPE stored as per-frequency divisors (`rope_freqs`). The `mmproj`
vision files are not read, so image input needs the safetensors or MLX checkpoint. Weights stay in their GGUF
format: F32, F16, BF16, Q4_0/Q4_1/Q5_0/Q5_1/Q8_0, Q2_K-Q6_K, IQ1_S/IQ1_M, IQ2_XXS/XS/S,
IQ3_XXS/S and IQ4_NL/XS (`ggml.rs`, dequantization bit-exact with gguf-py), with NEON decode
kernels for every type and Metal prefill kernels. The i-quants (IQ1-IQ3) decode several times
slower than Q4_K on the CPU (table lookups); Q4_K, Q5_K, Q6_K, IQ4_XS and Q8_0 are close to
oxinfer's own Q4/Q8. Embedded MTP layers (`blk.N.nextn.*`) work with `--mtp`.

On HauhauCS's Qwen3.8-27B Q4_K_P the next-token distribution matches llama.cpp on the same file
(same top 10, log-probabilities within 0.03, mean 0.013). F32/BF16 GGUFs from llama.cpp's
converter match transformers to 5e-5. `--save-quantized` does not take GGUF input.

#### Llama 3 GGUF

```sh
./target/release/oxinfer --model bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M -p "Why is the sky blue?"
```

llama.cpp's converter permutes the rows of `q_proj` / `k_proj` within each head for ggml's
adjacent-pair RoPE (`LlamaModel.permute` in `convert_hf_to_gguf.py`: stored row `2i + j` of a head
of `d` rows is Hugging Face row `j * d/2 + i`). Rotating adjacent pairs of the stored rows is the
same arithmetic as rotate-half RoPE on the original rows, and attention scores do not depend on
the order within a head, so the weights stay mapped as they are (no copy) and the Llama GGUF path
uses the interleaved RoPE instead; LoRA adapters on `q_proj` / `k_proj` get their rows reordered
to match. The llama3 RoPE scaling (factor, low / high frequency factors, original context) is not
in the GGUF metadata: the converter stores the resulting divisor of each frequency as the
`rope_freqs.weight` tensor, which divides the inverse frequencies. Base, heads, KV heads, head
size and eps come from the `llama.*` keys. The tokenizer is Llama 3's byte-level BPE with the
`llama-bpe` pre-tokenizer regex, no normalizer and `ignore_merges`, plus the special and
user-defined tokens (Orpheus's 28683 audio tokens): token for token identical to the checkpoints'
own `tokenizer.json` (Llama 3.2 1B and Orpheus 3B) on 2953 texts (this README, source code,
ten languages, emoji, whitespace runs, special tokens; 117k tokens). Chat uses the Llama 3
format when `tokenizer.chat_template` is Llama 3's (header tokens); other Llama-architecture
GGUFs (SmolLM, Mistral, ...) load as text models with `--raw` but their chat formats are refused.

Agreement on Llama-3.2-1B-Instruct (bartowski's GGUFs; seven raw prompts of 5-24 tokens in
English, German, French and Python, each starting with `<|begin_of_text|>`; the last-token
distribution and 64 greedy tokens; CPU):

| Check | Result |
|---|---|
| F16 GGUF vs the bf16 safetensors (unsloth/Llama-3.2-1B-Instruct), oxinfer both | logits bit-identical (max abs diff 0), same greedy text |
| Q8_0 GGUF vs bf16 safetensors | same top 1 in 7/7, top 10 overlap 9-10/10, KL 1e-5..3.5e-3 |
| Q4_K_M GGUF vs bf16 safetensors | same top 1 in 6/7, top 10 overlap 8-10/10, KL 3e-3..0.17 |
| F16 GGUF, oxinfer vs llama.cpp (same file, top 20 log-probabilities) | same top 1 7/7, within 0.027; 64/64 greedy tokens identical in 7/7 |
| Q8_0 GGUF, oxinfer vs llama.cpp | same top 1 7/7, within 0.10-0.20; 64/64 greedy tokens in 6/7, the 7th splits at a near tie after 8 |
| Q4_K_M GGUF, oxinfer vs llama.cpp | same top 1 7/7, within 0.15-0.34; 64/64 greedy tokens in 2/7, the others split at near ties after 1, 8, 19, 32 and 55 |
| Q8_0 / Q4_K_M, transformers float32 on the GGUF's dequantized weights (Q/K rows un-permuted) vs oxinfer | within 0.02-0.12 / 0.02-0.16 (vs llama.cpp: 0.11-0.21 / 0.13-0.36) |

On the F16 file that float32 reference matches oxinfer to 1e-4. The quantized files differ from
llama.cpp by more than the F16 file, but against the exact float32 forward pass on the same
dequantized weights oxinfer is the closer of the two, so the gap comes from how the engines round
intermediate values (both quantize activations to int8 for the integer dot products), not from a
different reading of the weights.

### MLX checkpoints

MLX repos (mlx-community and the like) store each matrix as packed 4-bit values (`.weight`, eight
per uint32) plus a bf16 scale and bias per group of 64: `w = scale * q + bias`. Since
`scale * q + bias = scale * (q - 8) + (bias + 8 * scale)`, the nibbles carry over unchanged into
the Q4 kernels' layout, and the extra term becomes one bf16 offset per group. Decoding adds
`offset * sum(x over the group)` per row after the int8 dot products, and prefill dequantizes it
into the f32 tiles (on the CPU, Metal, CUDA and WebGPU alike). Other bit widths, and quantized
tensors that are read as plain bf16 (Gemma 4's vision and audio projections), are dequantized on
load; embedding tables are dequantized one row at a time straight from the checkpoint. The MLX
names (`language_model.model.*`, `vision_tower.*`) map to the transformers ones, and channels-last
convolution weights are permuted back.

```sh
./target/release/oxinfer --model mlx-community/gemma-4-e2b-it-4bit -p "Hello"
./target/release/oxinfer --model mlx-community/gemma-4-e2b-it-4bit --audio clip.wav -p "Transcribe this."
```

Against transformers float32 on the dequantized MLX weights, last-token logits agree to a mean
absolute difference of 0.016 (Gemma 4 E4B), 0.026 (Qwen3-4B) and 0.055 (Gemma 3 4B), with the
same top 10; the difference to the original bf16 models is MLX's own quantization.

M5 Max, 200-token replies: Gemma 4 E2B 110 tok/s (120 for oxinfer's own Q4 file, which has no
offsets), Gemma 4 E4B 71 tok/s, Gemma 3 4B 77 tok/s, Qwen3-4B 70 tok/s. Prefill of 2153 tokens on
Gemma 3 4B takes 4.0 s with Metal (12.9 s on the CPU).

`tools/mlx_to_hf.py` dequantizes an MLX checkpoint into a float32 transformers folder, which is
the exact reference for those weights; `tools/ref_logits.py` then compares logits. LiteRT-LM
files (`.litertlm`, TFLite graphs) are not supported.

## Build and run

```sh
cargo build --release
./target/release/oxinfer --model models/gemma-4-E4B-it --prompt "Explain RMSNorm in two sentences."
./target/release/oxinfer --model models/gemma-4-E4B-it --chat          # interactive, multi-turn
./target/release/oxinfer --model models/gemma-4-E4B-it --think -p "..."  # enable thinking mode
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--temp F --top-k N --top-p F --seed N` | sampling (default is greedy, `--temp 0`) |
| `--system TEXT` | system prompt (default: a short one saying the model runs locally, with the local date, time and zone) |
| `--no-system` | no system prompt (also implied by `--raw` and `--dump-logits`) |
| `--tools` | let the model call built-in tools: `get_current_datetime` and `calculate` (see Tool calling) |
| `--print-prompt` | print the fully formatted prompt and exit |
| `--raw` | feed the prompt as-is after `<bos>`, no chat template |
| `--max-tokens N` | cap on generated tokens |
| `--threads N` | worker threads (default: all cores) |
| `--quant q8` | quantize streamed weights to int8 at load (blocks of 32, f32 scales); ~1.8x less memory traffic |
| `--quant q4` | GGUF-style Q4_0 (blocks of 32, 4-bit, f16 scale) for all streamed weights; ~3.5x less traffic |
| `--draft DIR` | speculative decoding with Google's `gemma-4-E4B-it-assistant` drafter (greedy only; output identical to plain greedy) |
| `--draft-tokens N` | max draft tokens per round (default 6 with `--draft`, 3 with `--mtp`, 8 with `--mtp --stream`; the length adapts) |
| `--mtp` | Qwen3.5/3.8: speculative decoding with the checkpoint's multi-token-prediction layer (greedy and sampled) |
| `--mtp-fixed` | with `--mtp`, always draft `--draft-tokens` instead of adapting between 1 and N |
| `--mtp-from DIR` | `--mtp` with the MTP weights taken from another checkpoint (e.g. the original of a converted copy) |
| `--mmap` | map weights from disk instead of copying them into RAM |
| `--save-quantized DIR` | write a pre-quantized, text-only checkpoint at `--quant` precision (see below) and exit |
| `--devices` | list detected CPU features and GPUs (Metal, CUDA, Vulkan) and exit |
| `--device D` | `auto` (default: use a GPU backend when one exists), `cpu`, `metal`, `cuda`, `vulkan` |
| `--gpu-decode on\|off` | generate a whole token per GPU submission (default `on` where supported; see [Decoding on the GPU](#decoding-on-the-gpu)); engine / speaker option `gpu_decode` |
| `--prefill-chunk N` | tokens per prefill chunk (default 256 on the CPU, 1024 when a GPU backend is active) |
| `--profile` | per-operation timing and bandwidth breakdown |
| `--dump-logits FILE` | write the logits for the last prompt token (for reference checks) |
| `--show-tokens` | print token ids |

By default the whole safetensors file is read into RAM at startup (a few seconds), which
gives steady decode speed from the first token. `--mmap` starts faster but pays page
faults on the first pass.

### As a library

`oxinfer::engine` is the CLI's text generation as an API: load a directory, GGUF file or repo
id with `EngineOptions` (every flag above that applies to text, also settable as `key=value`
pairs), then hold a conversation with streamed output, cancellation from another thread
(`CancelHandle`), tool calls, attachments (`Engine::send_with_media` with images, videos and
audio clips in an `Attachments`) and per-turn statistics:

```rust
use oxinfer::engine::{Engine, EngineOptions, Flow};
let mut opts = EngineOptions::default();
opts.set("quant", "q8")?;
opts.set("mtp", "1")?;
let mut engine = Engine::load("Qwen/Qwen3.5-0.8B", &opts)?;
let reply = engine.send("Why is the sky blue?", &mut |d| {
    print!("{}", d.text); // d.thinking marks thinking text
    Flow::Continue        // Flow::Stop ends the reply
})?;
println!("\n{:.1} tok/s, {:.0}% drafts accepted", reply.stats.decode_tok_s(), 100.0 * reply.stats.acceptance());
```

`examples/chat.rs` is a complete chat. Wrap `Engine::load` in `oxinfer::hf::with_progress` for
download progress and cancellation. Core ML packages (below) load through the same
`Engine::load` on macOS / iOS, with the options `coreml_units` (`ane`, `all`, `gpu`, `cpu`) and
`coreml_prefill` (`auto`, `batch`, `pad`, `single`); `engine.info().coreml` describes them. The lower-level pieces the CLI is built from are public too:
`oxinfer::generate` (decoding loops with per-token callbacks, MTP and drafter speculative
decoding, streaming detokenization) and the model, session and sampler types. A C API over the
engine (`capi` feature, `include/oxinfer.h`) and a SwiftUI test app for iOS are in `ios/`
(see `ios/README.md`). Neuron edits, steering vectors, directional ablation and SAE features are
`Engine::set_interventions` (see [Runtime interventions](#runtime-interventions-neuron-edits-steering-abliteration-sae-features)).

### Using oxinfer from oxideav (and other crates)

Images, audio and video in the API are [oxideav](https://crates.io/crates/oxideav-core) types:
decoded `VideoFrame`s and `AudioFrame`s with the `CodecParameters` that describe them, as an
oxideav pipeline hands them from stage to stage. Two features split what a dependent pulls in:

| Feature | Brings | Gives |
|---|---|---|
| `oxideav` | `oxideav-core` only | `oxinfer::av` and the in-memory inputs and outputs below; `oxinfer::oxideav_core` re-exports the crate |
| `media` (default, implies `oxideav`) | oxideav-pipeline, oxideav-image and the codec and container crates (PNG, JPEG, WebP, GIF, BMP, MP3, FLAC, AAC, Vorbis, Opus, WAV, MP4, MOV, Matroska / WebM, Ogg, AVI, H.264, H.265, VP8, VP9, AV1, MPEG-4 Part 2) | the path-based inputs (`--image`, `--video`, `--audio`, `transcribe_file`, `Media`) |

`hf` (HuggingFace fetching) and `gpu` (run-time GPU backends) are independent of both. A crate
that already has decoded frames, such as an oxideav filter, needs only the types:

```toml
[dependencies]
oxinfer = { version = "0.0.4", default-features = false, features = ["oxideav", "hf", "gpu"] }
oxideav-core = "0.1.41" # the same 0.1 release oxinfer uses, so the types are the same
```

Chat and multimodal prompts take frames next to file paths (each kind after the files of the
same kind):

```rust
use oxinfer::engine::{Attachments, Engine, EngineOptions, Flow};
let mut engine = Engine::load("Qwen/Qwen3.5-0.8B", &EngineOptions::default())?;
// A picture: one VideoFrame and its parameters (width, height, pixel_format).
let media = Attachments::default().with_image_frame(frame, params);
// A clip: frames in presentation order, their parameters and the stream time base; frames are
// sampled as the model's video processor does, and the prompt's timestamps come from the pts.
let media = media.with_video_frames(frames, video_params, Some(time_base));
// Audio (Gemma 4): frames and their parameters (sample_rate, channels, sample_format).
let media = media.with_audio_frames(audio_frames, audio_params);
let reply = engine.send_with_media("What happens here?", &media, &mut |_| Flow::Continue)?;
```

The other consumers and producers of media:

```rust
// Speech to text from decoded audio (any sample format, rate and channel count).
let t = transcriber.transcribe_frames(&audio_frames, &audio_params, &TranscribeOptions::default())?;
// Typed decisions over a picture (placed before the state text).
let req = DecisionRequest::noul("Backyard camera.", "Is there an animal?").with_image_frames([Picture::new(frame, params)]);
// Stable Diffusion output as an Rgb24 frame for an encoder or a filter graph.
let (img, _) = pipeline.generate(&params, &mut |_, _| {})?;
let (frame, frame_params) = img.into_video_frame();
```

Underneath, `av::video_frame_to_rgb` and `av::audio_frames_to_mono` do the conversions, and
`mm::prepare_frames` (with `mm::MmFrames`) turns frames into soft tokens for the lower-level
`generate` API. Pixel formats: packed RGB / BGR in every byte order (with or without alpha or
padding, 8 and 16 bits, 5-6-5 and 5-5-5), gray (8 to 16 bits, with alpha), 1-bit mono, `Pal8`,
planar YUV 4:2:0, 4:2:2, 4:4:4, 4:4:0 and 4:1:1 at 8 to 16 bits (with alpha, `YuvJ*`), NV12 /
NV21, YUYV / UYVY and planar GBR(A); others (CMYK, float) are an error naming the format. YCbCr
follows the colour signal of the frame or the stream (BT.601 limited range when unspecified;
BT.709, BT.2020, FCC, SMPTE 240M, identity, full range), in oxideav-pixfmt's fixed-point
arithmetic: the RGB is byte-identical to `oxideav_image::Image::to_rgb8` for every format both
convert, so frames and files give the same soft tokens, replies and transcripts
(`tests/oxideav_e2e.rs`). A missing `width`, `height` or `pixel_format` is inferred from the
plane layout, as for decoded files; audio of every `SampleFormat`, interleaved or planar, is
averaged to mono and resampled to the model's rate.

**Dependency cycles.** `media` depends on the codec and container crates directly, not on
`oxideav-meta`, so a transform crate that depends on oxinfer can itself be registered by
`oxideav-meta` without a cycle, whatever features it enables. (Cargo refuses any cycle among the
activated dependencies and accepts one through an optional dependency that is not enabled. With
a `media` that depended on `oxideav-meta`, oxinfer -> oxideav-meta -> transform -> oxinfer would
fail in any build that registers the transform in oxideav-meta and enables `media` anywhere, even
in another crate, since features are unified across the whole graph.) The rule for the transform crate and anything oxideav-meta pulls in:

- depend on `oxinfer` with `default-features = false` and the features you use (`oxideav`, and
  `hf` / `gpu` as wanted); take `media` only if you need oxinfer to open files itself;
- never make oxinfer depend on a crate that depends on the transform (oxideav-meta, an
  "everything" facade): its `media` feature lists the codec crates one by one for that reason.

## Performance

Decode is memory-bandwidth bound: every generated token streams all dense weights once.
On an M5 Max (18 cores, ~280 GB/s measured CPU read bandwidth):

| Model | Precision | Weights per token | Decode | Decode + drafter (prose / code) | Prefill (3k prompt) |
|---|---|---|---|---|---|
| E2B-it | bf16 | 4.6 GB | ~50 tok/s (≈230 GB/s) | n/a (no E2B drafter tested) | ~280 tok/s |
| E2B-it | `--quant q8` | 2.6 GB | ~86 tok/s | | ~255 tok/s |
| E2B-it | `--quant q4` | 1.3 GB | ~130 tok/s | | |
| E4B-it | bf16 | 9.3 GB | ~28 tok/s (≈245 GB/s) | 41 / 47 tok/s | ~135 tok/s |
| E4B-it | `--quant q8` | 5.2 GB | ~45 tok/s | 66 / 82 tok/s | ~135 tok/s |
| E4B-it | `--quant q4` | 2.6 GB | ~76 tok/s | 75 / 99 tok/s | ~135 tok/s |
| Qwen3.8-27B | bf16 | 51 GB | ~5.3 tok/s (≈270 GB/s) | | ~19 tok/s |
| Qwen3.8-27B | `--quant q8` | 29 GB | ~8.8 tok/s | `--mtp`: 1.5x / 2.0x | ~19 tok/s |
| Qwen3.8-27B | `--quant q4` | 14 GB | ~16 tok/s | `--mtp`: 1.1-1.2x / 1.6x | ~20 tok/s |
| Gemma 4 26B-A4B (MoE) | bf16 | 7.6 GB | ~30 tok/s (≈230 GB/s) | | ~150 tok/s (1.8k prompt) |
| Gemma 4 26B-A4B (MoE) | `--quant q8` | 4.3 GB | ~48 tok/s | | ~140 tok/s |
| Gemma 4 26B-A4B (MoE) | `--quant q4` | 2.1 GB | ~69 tok/s | | ~136 tok/s |

For the 27B model `--mmap` is recommended: quantization then reads the checkpoint straight from
the page cache (about 3 s to quantize 27.8B parameters on 18 threads) instead of copying 55 GB
into the process first.

With the Metal backend (default on Apple Silicon) prefill runs on the GPU:

| Model | Precision | Prefill, CPU only | Prefill, Metal |
|---|---|---|---|
| E4B-it | `--quant q4` | ~97 tok/s | ~580 tok/s |
| Qwen3.8-27B | `--quant q4` | ~18 tok/s | ~96 tok/s (256-token chunks) |
| Gemma 4 26B-A4B (MoE) | bf16 / `--quant q4` | ~150 / ~136 tok/s | ~544 / ~576 tok/s (1758-token prompt) |

(616-token prompt.)

### Decoding on the GPU

With the Metal backend, single-token decode steps also run on the GPU, the whole forward pass in
one command buffer per token (see [Platforms](#platforms-and-accelerators) for what runs where).
Measured on an M5 Max, 256 greedy tokens after a short prompt, against the same build with
`--gpu-decode off` (GPU prefill, CPU decode; interleaved runs, load average ~5-11):

| Model | Weights | CPU decode | GPU decode | llama.cpp Metal |
|---|---|---|---|---|
| Llama-3.2-1B-Instruct | GGUF Q4_K_M | 194-205 tok/s | 391 tok/s | 390-407 tok/s |
| Llama-3.2-1B-Instruct | GGUF Q8_0 | 148-156 tok/s | 284 tok/s | 269-285 tok/s |
| Orpheus 3B (text) | GGUF Q4_K_M | 87-94 tok/s | 161-168 tok/s | 176-189 tok/s |
| Orpheus 3B (`--speak`) | GGUF Q4_K_M | 99 tok/s | 191 tok/s (RTF 0.46) | |
| Qwen3.5-0.8B | GGUF Q8_0 | 179-182 tok/s | 356-361 tok/s | 282-284 tok/s |
| Qwen3.5-0.8B | `--quant q8` | 196 tok/s | 351 tok/s | |
| Qwen3.8-27B (HauhauCS) | GGUF Q4_K_P | 11.9-12.0 tok/s | 20.7-20.9 tok/s | 22.9-23.7 tok/s |
| Zawen-v0.1-35B-A3B (MoE) | GGUF Q4_K_M | 60 tok/s | 137-140 tok/s | 105-116 tok/s |
| Zawen-v0.1-35B-A3B (MoE) | GGUF Q8_0 | 47-48 tok/s | 111-114 tok/s | 85-89 tok/s |
| Zawen-v0.1-35B-A3B (MoE) | Q4 (`--save-quantized`) | 70 tok/s | 152 tok/s | |
| Gemma 4 E2B-it | Q4 (`--save-quantized`) | 134 tok/s | 201 tok/s | |
| Gemma 4 E4B-it | Q4 (`--save-quantized`) | 61 tok/s | 115 tok/s | |
| Gemma 4 26B-A4B (MoE) | GGUF Q4_0 | 63 tok/s | 130 tok/s | 115-116 tok/s |

Long contexts cost attention time (the KV cache stays f32): Llama-3.2-1B Q4_K_M decodes at 297
tok/s after a 2.5k-token prompt (CPU decode 139). The first step of a model compiles its plan
(~0.1-0.2 s); a multi-token pass (the prompt) builds it ahead.

Agreement with the CPU path (`OXINFER_GPU_DECODE_CHECK=1` runs every GPU step on the CPU too and
prints the KL divergence of the two next-token distributions): on bf16 weights the two match to a
KL of ~1e-12 (Llama 3.2 1B, Qwen3 0.6B, Qwen3.5-0.8B, Gemma 3 1B, Gemma 4 E2B / E4B / 26B-A4B, a
small Qwen3.5 MoE), with long contexts and split attention included. On quantized weights the CPU
kernels quantize the activations to int8 while the GPU keeps them f32, so the paths differ by a
mean KL of 1e-4 to 6e-4 (dense) and up to 3e-3 (MoE, whose routing amplifies it); with 7 prompts
of 128 tokens on Qwen3.8-27B Q4_K_P, 6 greedy texts were identical to the CPU's and one parted at
token 26. Against llama.cpp Metal the first 64 greedy tokens were identical on every model and
prompt above but Zawen Q4_K_M prose (both parted at token 25, as they did before). Orpheus GGUF
speech made on the GPU transcribes back exactly with whisper-small.

On GPUs whose cores have matrix units ("neural accelerators": M5, A19) the prefill GEMMs and the
MoE expert layers run on them (Metal 4 tensor operations, see
[Platforms](#platforms-and-accelerators)). Measured on an M5 Max with a ~3k-token prompt (default
1024-token chunks; best of two interleaved runs on a shared machine), against the f32 GEMM of the
same build (`OXINFER_METAL_MMA=0`):

| Model | Precision | Metal, f32 GEMM | Metal, matrix units |
|---|---|---|---|
| E2B-it | bf16 | 631 tok/s | 724 tok/s |
| E4B-it | Q4 (`--save-quantized`) | 420 tok/s | 520 tok/s |
| Qwen3.8-27B | Q4 (`--save-quantized`) | 109 tok/s | 165 tok/s |
| Qwen3.8-27B (HauhauCS GGUF) | Q4_K_P | 93 tok/s | 158 tok/s |
| Zawen-v0.1-35B-A3B (MoE) | `--quant q4 --mmap` | 165 tok/s | 353 tok/s |
| Zawen-v0.1-35B-A3B (MoE) | `--quant q8 --mmap` | 161 tok/s | 255 tok/s |
| Zawen-v0.1-35B-A3B (MoE) | Q4 (`--save-quantized`) | 168 tok/s | 379 tok/s |

The GEMMs themselves run 3-4x faster (table under Platforms); what is left of prefill time is
mostly the CPU part of the layers (Gated DeltaNet recurrence, attention, routing). Against the CPU
path the next-token distributions differ by a KL divergence of 1e-7 to 2e-5 nats on the dense
models and 1e-4 to 1e-3 on Zawen (whose top-8 routing amplifies small differences), with the same
top 10 tokens in every case; a 128-token greedy generation of the 27B GGUF with every pass on the
matrix units matched the CPU's token for token.

The matmul kernels reach the machine's bandwidth ceiling with 12 threads
(`cargo run --release --bin membench` prints the raw numbers for your machine), so decode
speed is set by bytes per token: Q8 cuts them by 1.8x and Q4 by 3.5x. Prefill is compute-bound
on an 8x8 outer-product GEMM at roughly 1.3 TFLOP/s (the kernel itself runs at ~90% of FMA peak
when operands are in L1; a 12-token tile was tried and lost to register pressure). Quantized
prefill dequantizes to f32 and uses the same kernel, so prompt processing speed is independent
of precision.

### Comparison with llama.cpp

Same GGUF file in both engines, same prompt tokens (llama.cpp gets the text oxinfer's chat
template produces), greedy, 256 generated tokens (fewer when the model ends its turn) after a
short code prompt and a short prose prompt, plus prompt processing of 512 and 2048 tokens.
llama.cpp is commit 4df29be4 with HauhauCS's FastMTP patch, driven through `llama-server`
(one warm-up request per server start); oxinfer runs as a fresh CLI process per measurement.
M5 Max, 18 cores, 128 GB, with other jobs running: runs alternate between the engines, each
configuration ran 3 times, cells are the median with [min-max], and the load average is
recorded per run (medians 9-15 here; tok/s of the CPU-only rows varied by 2x during a heavier
spell, so treat small differences as noise). "Metal" is llama.cpp's default (every layer on the
GPU) and, in the tables of this section, oxinfer with prefill GEMMs on the GPU and everything else
on the CPU, which was its default before GPU decoding (see
[Decoding on the GPU](#decoding-on-the-gpu) for the current default); "CPU" is
`--device none -ngl 0` / `--device cpu` with each engine's default thread count (llama.cpp 6,
oxinfer 18). The oxinfer default and speculative rows were rerun after the Metal matrix-unit
prefill and the multi-token GGUF decode kernels landed (load ~11-12); the oxinfer CPU rows and
the llama.cpp rows are from the earlier runs.

Qwen3.8-27B (HauhauCS Aggressive, `Q4_K_P`, 17.9 GB; speculative rows use the embedded MTP
head, or the 32k-vocabulary FastMTP sidecar with `--mtp-from`):

| Engine | pp512 tok/s | pp2048 tok/s | Decode, code | Decode, prose | Drafts accepted (code / prose) |
|---|---|---|---|---|---|
| llama.cpp Metal | 650 [620-703] | 577 [538-720] | 24.2 [23.3-26.2] | 24.9 [24.1-25.2] | |
| llama.cpp Metal, MTP depth 2 | | | 29.5 [22.5-30.4] | 22.0 [19.9-22.7] | 84% / 61% |
| llama.cpp Metal, MTP depth 3 | | | 34.2 [26.0-35.3] | 24.0 [19.2-28.5] | 76% / 49% |
| llama.cpp Metal, FastMTP depth 3 | | | 37.7 [32.1-39.0] | 25.3 [22.5-26.9] | 78% / 42% |
| llama.cpp CPU | 20.8 [18.9-21.2] | 19.0 [18.9-20.4] | 6.7 [6.2-6.9] | 6.8 [6.3-6.9] | |
| oxinfer (Metal prefill) | 234 [233-252] | 209 [204-233] | 11.6 [11.1-12.0] | 11.7 [10.5-12.3] | |
| oxinfer `--mtp` | | | 21.0 [19.6-21.4] | 15.2 [14.7-15.2] | 80% / 50% |
| oxinfer `--mtp-from` FastMTP | | | 21.5 [21.2-22.3] | 15.0 [14.4-15.1] | 78% / 46% |
| oxinfer CPU | 24.2 [21.4-27.8] | 23.3 [20.8-27.2] | 10.6 [10.3-11.9] | 11.5 [10.8-11.8] | |

(`llama-bench` on the same file in a quiet moment: pp512 744, pp2048 642, tg256 25.5 tok/s.)

Llama-3.2-1B-Instruct (bartowski) and Orpheus 3B (text generation through the chat template, so
the code / prose replies are short: 29-33 tokens for code on Orpheus Q4_K_M; Orpheus Q8_0 replies
only 2-31 tokens and is left out), llama.cpp built from source (ggml 0.20.0), load average 6-8,
3 runs each, CPU rows with each engine's default thread count as above. llama.cpp gets token
ids (no second BOS):

| Model | Engine | pp512 tok/s | Decode, code | Decode, prose |
|---|---|---|---|---|
| 1B Q4_K_M | llama.cpp Metal | 16323 [16266-16451] | 404 [400-423] | 406 [404-408] |
| | llama.cpp CPU | 582 [529-613] | 162 [72-165] | 161 [71-166] |
| | oxinfer (Metal prefill) | 2048 [1164-3012] | 223 [139-227] | 225 [225-229] |
| | oxinfer CPU | 617 [582-665] | 231 [221-235] | 225 [221-235] |
| 1B Q8_0 | llama.cpp Metal | 16956 [16880-17214] | 305 [301-305] | 304 [303-305] |
| | llama.cpp CPU | 748 [677-749] | 159 [158-160] | 159 [158-160] |
| | oxinfer (Metal prefill) | 3012 [1969-3012] | 163 [163-166] | 161 [159-165] |
| | oxinfer CPU | 610 [533-617] | 164 [150-171] | 166 [152-170] |
| Orpheus 3B Q4_K_M | llama.cpp Metal | 5320 [4066-6098] | 190 [189-191] | 191 [189-192] |
| | llama.cpp CPU | 204 [187-209] | 68 [67-69] | 65 [65-66] |
| | oxinfer (Metal prefill) | 1313 [914-1384] | 94 [91-94] | 88 [87-89] |
| | oxinfer CPU | 226 [204-229] | 94 [94-100] | 89 [89-92] |

On the CPU oxinfer decodes 1.4x faster than llama.cpp on Q4_K_M at the default thread counts
(6 vs 18), but with 12 threads each (1B Q4_K_M, code) the gap is 5% (214-215 vs 221-227 tok/s);
Q8_0 is on par. llama.cpp runs the whole pass on the GPU and decodes 1.8-2.2x faster than
oxinfer's GPU-prefill-only path; with GPU decoding oxinfer is on par (table in
[Decoding on the GPU](#decoding-on-the-gpu)).
Greedy outputs: oxinfer's Metal and CPU runs agree with each other on every workload but
one (Orpheus prose, from token 3), and every CPU divergence between the engines within 64
tokens (at tokens 0, 26, 46 and 62) is at the token where llama.cpp's own Metal and CPU runs
diverge too (near ties); 7 of the 12 workload / backend pairs above are identical over the 64
tokens compared.

Zawen-v0.1-35B-A3B (Qwen3.5-MoE, 256 experts, converted with llama.cpp's `convert_hf_to_gguf.py`
+ `llama-quantize`; the checkpoint has no MTP weights, but the converter still writes
`nextn_predict_layers = 1` from `config.json`, which llama.cpp then refuses to load, so
`gguf_set_metadata.py` set it to 0 and `block_count` to 40; oxinfer loads either file):

| Engine | Q4_K_M pp512 | pp2048 | Decode code / prose | Q8_0 pp512 | pp2048 | Decode code / prose |
|---|---|---|---|---|---|---|
| llama.cpp Metal | 3001 [2998-3007] | 3065 [2381-3091] | 110 / 111 | 2700 [2603-2798] | 2476 [1732-2730] | 80.7 / 81.0 |
| llama.cpp CPU | 124 [122-125] | 117 [116-119] | 47.5 / 46.5 | 156 [150-164] | 153 [141-157] | 43.2 / 43.8 |
| oxinfer (Metal prefill) | 277 [274-280] | 294 [290-296] | 63.9 / 64.0 | 210 [210-230] | 236 [234-237] | 45.2 / 45.0 |
| oxinfer CPU | 157 [154-161] | 146 [143-151] | 54.8 / 55.2 | 122 [112-146] | 111 [106-136] | 46.4 / 43.2 |

Qwen3.5-0.8B (`Q8_0` converted with `convert_hf_to_gguf.py`, keeps the MTP layer):

| Engine | pp512 | pp2048 | Decode code / prose | MTP decode code / prose (accepted) |
|---|---|---|---|---|
| llama.cpp Metal | 14930 [14924-15158] | 16034 [15939-16088] | 280 / 281 | depth 2: 332 / 276 (67% / 47%); depth 3: 335 / 250 (59% / 35%) |
| llama.cpp CPU | 809 [797-811] | 748 [739-768] | 161 / 165 | |
| oxinfer (Metal prefill) | 1969 [1896-1969] | 1829 [1812-1845] | 183 / 184 | `--mtp`: 198 / 152 (61% / 36%) |
| oxinfer CPU | 826 [813-826] | 747 [739-779] | 173 / 172 | |

With 12 threads for both engines on the CPU (a separate interleaved run, load ~15), llama.cpp
pulls ahead on prompts: 27B pp512 27.4 vs 18.4 tok/s, decode 9.4 / 8.4 vs 8.0 / 9.7; Zawen Q4_K_M
pp512 197 vs 129, decode 60.5 / 60.1 vs 58.9 / 56.9.

Where the differences come from:

- **Decode with a GPU**: llama.cpp runs the whole forward pass on the GPU; the rows above have
  oxinfer decode on the CPU (`--profile` on the 27B: matmuls stream 16.9 GB/token at ~200 GB/s, 85
  of 95 ms per token), hence the 2.2x on the 27B and 2x on the MoE. oxinfer now runs the whole
  decode pass on the GPU too ([Decoding on the GPU](#decoding-on-the-gpu)): on par with llama.cpp on
  the 1B models, ahead on Qwen3.5-0.8B (1.27x), Zawen (1.2-1.3x) and Gemma 4 26B-A4B (1.13x),
  behind on Orpheus 3B and the 27B (~0.9x; its Q4_K / Q5_K matvecs read 450-530 GB/s against
  550-590 for Q6_K / Q8_0, the main remaining kernel gap).
- **Prefill with a GPU** (2.8x on the 27B, 10x on the MoE, 8x on 0.8B): only the dense GEMMs
  run on the GPU. They now use the M5 matrix units (half inputs, f32 accumulation; 144 -> 234
  tok/s at pp512 on the 27B, before that f32 simdgroup tiles at ~8.4 TFLOP/s against llama.cpp's
  ~35). Attention and the Gated DeltaNet scan stay on the CPU (2.6 ms/token, now the larger part
  of oxinfer's 27B prefill), and each layer's activations make a round trip through host memory.
  On the MoE, the GPU expert path takes fused `gate_up` weights (safetensors checkpoints: Zawen Q4
  prefill 165 -> 353-379 tok/s) and, since GPU decoding landed, GGUF's separate gate and up
  tensors as well (Zawen Q4_K_M, 695-token prompt: 292 -> 548 tok/s; the GGUF rows above predate
  it).
- **CPU prefill**: llama.cpp repacks K-quants at load (`CPU_REPACK`, interleaved rows for the
  i8mm `SMMLA` instructions) and multiplies in int8; oxinfer dequantizes to f32 and runs an
  FMA GEMM. It is ahead on K-quants only while llama.cpp uses its default 6 threads, and behind
  on Q8_0 even then.
- **MoE decode** on the CPU is not bandwidth-bound (Zawen Q4_K_M: 521 matmul calls per token at
  ~21 us, 170 GB/s over only 1.9 GB): dispatch and per-expert overhead dominate. On the GPU the
  routed experts of a layer are two dispatches (gate/up pairs, weighted down) reading through
  per-expert address tables.
- **Speculative decoding**: acceptance is the same in both engines (78-80% code, 46-50% prose
  on the 27B). The multi-token GGUF kernels (each weight block unpacked once per verification
  pass instead of once per token pair) made oxinfer's verification passes cheap enough that MTP
  now gains 1.8x on code and 1.3x on prose on the 27B (11.6 -> 21.0 / 11.7 -> 15.2 tok/s; it was
  1.35x / 1.04x), more than llama.cpp's 1.4-1.55x / ~1x, though llama.cpp's GPU decode still
  leads in absolute terms (34-38 tok/s on code). On 0.8B every draft step reads the 248k-row LM
  head, so prose loses in both engines.
- **Load**: oxinfer's spin-waiting pool on all 18 cores loses the most when other processes
  compete for cores (27B decode fell to 3-6 tok/s at load averages of 30-90, while llama.cpp's
  GPU decode stayed at ~20); `--threads 12` was faster than 18 in that state.

Output agreement (first 64 greedy tokens, same in all three repetitions): oxinfer matches
llama.cpp Metal token for token on the code prompt for every model and on the prose prompt for
the 27B; on prose the texts part at token 48 (0.8B), 25 (Zawen Q4_K_M) and 21 (Zawen Q8_0) after
an identical prefix. llama.cpp's own CPU and Metal backends part as early or earlier on the same
prompts (27B: tokens 53 / 20, 0.8B prose: 9, Zawen: 25 / 18-30), and oxinfer's CPU and Metal
paths always agree with each other, so these are rounding-order differences at near-ties rather
than a model error. Speculative decoding (embedded MTP or FastMTP) gives exactly the plain greedy
tokens in both engines.

To rerun (`tools/bench/bench.py --help` lists the configurations; `--max-load N` waits for a
quieter machine before each run):

```sh
cargo build --release
tools/bench/bench.py --model M.gguf --sidecar FastMTP.gguf --llama-bin ~/llama.cpp/build/bin \
    --reps 3 --tag mymodel --out results.jsonl
tools/bench/summarize.py results.jsonl          # medians, acceptance, greedy agreement
```

### Quantization quality

Q8 and Q4 are both measured against the bf16 output distribution (itself within 1e-4 of the
transformers float32 reference). KL divergence of the next-token distribution on a test prompt is
about 1e-5 nats for Q8 and 3e-5 nats for Q4 on E4B, with the same top-5 tokens. Google also
publishes `gemma-4-*-it-qat-q4_0-unquantized` checkpoints trained for Q4_0; they load the same
way (`--quant q4` on that directory) but did not measure better than Q4 of the regular weights
here, and they are a different fine-tune (KL 0.03 from the regular model).

### Pre-quantized checkpoints

`--quant` normally quantizes the bf16 weights at every start. `--save-quantized DIR` writes the
result instead, so later runs (and browsers) load the int8 / int4 blocks as they are:

```sh
./target/release/oxinfer --model google/gemma-4-E4B-it --quant q4 --save-quantized models/gemma-4-E4B-it-q4
./target/release/oxinfer --model models/gemma-4-E4B-it-q4 -p "Hello"     # precision comes from the file
```

The output is ordinary safetensors: each quantized matrix `X` becomes `X.q8` (I8) + `X.q8s` (F32
scales) or `X.q4` (U8, two nibbles per byte) + `X.q4s` (F16 scales), in exactly the in-memory
layout of the kernels, so the loader borrows them without copying (`QBuf`). Only the tensors the
model reads are written: Qwen keeps its MTP head and its vision tower (in bf16), Gemma's vision
and audio towers are left out; `__metadata__` records the precision. Shards are at most 1 GiB (one allocation must stay under 2 GiB in wasm32), and large
tensors start on 16 KiB boundaries so Metal wraps them in place. The embedding tables are stored
quantized too and dequantized one row per token; tables that are only ever looked up (Gemma's
per-layer embeddings, Qwen 27B's untied input embedding) go to separate `model-tables-*` shards,
which the browser host keeps outside wasm memory. `OXINFER_TABLE_QUANT=bf16|q8|q4` overrides
their precision (default: the model's).

| Model | bf16 checkpoint | Q4 file | Load (in RAM) | KL vs bf16: Q4 at load / Q4 file |
|---|---|---|---|---|
| E2B-it | 10.2 GB | 2.6 GB | 0.40 s | 6.5e-4 / 5.9e-4 |
| E4B-it | 16.0 GB | 4.2 GB | 0.48 s (was 1.51 s) | 0.167 / 0.181 |
| Qwen3.8-27B | 55.6 GB | 15.1 GB | 2.55 s (4.6 s with `--mmap` before) | 0.130 / 0.133 (vs Q8) |

The KL column uses one fixed prompt whose last-token distribution is unusually sensitive; what it
shows is that the file adds nothing on top of Q4 itself (the per-layer embedding table at Q4 is
within noise of keeping it at higher precision). Q8 files match Q8-at-load the same way (E4B
7.1e-5 vs 6.7e-5). On Qwen3.5-0.8B, Q4 is too lossy in either form (KL ~1.3); use Q8 there.

### Speculative decoding

`--draft models/gemma-4-E4B-it-assistant` enables Google's multi-token-prediction drafter: a
4-layer, 256-wide decoder that cross-attends to the main model's shared KV caches and predicts
from the concatenation of the last token's embedding and the previous hidden state. Each round
drafts 2–6 tokens (adaptive) and verifies them in one multi-token forward pass; greedy
verification makes the output identical to plain greedy decoding. Acceptance is about 40% on prose
and 55–65% on code, giving 2–4 tokens per forward. The drafter's own drafts were checked against
the transformers implementation token for token. It pays off most where the verification pass
is bandwidth-bound (bf16, Q8); at Q4 the multi-token int8 kernels are instruction-bound, so it
only helps on predictable text.

`--mtp` does the same for Qwen3.5 / Qwen3.8 with the checkpoint's own multi-token-prediction
layer (`mtp.*`, `qwen_mtp.rs`): one gated-attention decoder layer that takes the main model's
final-normed hidden state at position i and the embedding of token i+1 (each RMS-normed,
concatenated embedding first, projected by `mtp.fc`), runs at position i with its own KV cache,
and predicts token i+2 through the shared LM head; further drafts feed it its own output (the
vLLM `qwen3_5_mtp` scheme). The default is 3 drafts per round, adapting between 1 and 3. Both
greedy and sampled decoding are supported: greedy output is identical to plain greedy (the
decode kernels give bit-identical rows for up to 7 tokens per pass, so the verification logits
equal one-token logits; checked on long outputs with and without `--think`, bf16 / Q8 / Q4,
dense and MoE), and sampling uses the standard speculative-sampling acceptance rule, so every
token is drawn from exactly the distribution `--temp` / `--top-k` / `--top-p` define (the
random stream differs from plain sampling, so the same seed gives a different text). Rejected
drafts are undone exactly: the verification forward keeps the Gated DeltaNet states from
before the pass plus each token's delta-rule inputs, and a rewind replays the rule up to the
last accepted token (the same arithmetic, so bit-identical; one state copy instead of one per
token, which is 151 MB per drafted token on Qwen3.8-27B); KV caches are truncated. On Qwen3.8-27B (M5 Max, under load from other
jobs):

| Precision | Plain | `--mtp` (code) | `--mtp` (prose) | Acceptance (code / prose) |
|---|---|---|---|---|
| `--quant q8` | 7.5 tok/s | 15.2 tok/s (3.75 tok/forward) | 10.5 tok/s (2.55 tok/forward) | 94% / 59% |
| `--quant q4` | 14.6 tok/s | 24.0 tok/s | 14.7-16.1 tok/s | 94% / 57% |

Q4 gains less because a 4-token Q4 pass costs about twice a one-token pass (SDOT-bound). On
Qwen3.5-0.8B every draft step reads the 248k-row LM head (a third of the model), so `--mtp`
gains ~20-25% on code and loses a little on prose; a trimmed draft head (`DraftHead`: a
`[V_draft, H]` matrix plus a draft-to-target id map, FastMTP style) can replace it through the
library. The MoE models (Qwen3.6-35B-A3B) work the same way, with an MoE MTP layer. A converted
checkpoint (`--save-quantized`) keeps the MTP weights; older converted copies can take them from
the original checkpoint with `--mtp-from DIR`.

From the library: `QwenModel::load_mtp(&files)` (or `QwenMtp::load(cfg, sidecar_files, quant)`),
then `MtpDecoder::new(&model, &mtp, MtpOptions { .. })`: `prefill` the prompt, `push` the first
token, call `step` until a stop token (`examples/qwen_mtp.rs`).

FastMTP sidecars (a GGUF holding only an MTP layer, a trimmed output head and a `d2t` map, such as
HauhauCS's `...-FastMTP-32K.gguf`) load with `--mtp-from FILE` (or `org/repo:FastMTP-32K`): the
sidecar's layer, token embedding and 32k-row head replace the target's, as in HauhauCS's
llama.cpp patch (`QwenMtp::attach_sidecar` in the library). With HauhauCS's Qwen3.8-27B Q4_K_P
GGUF, greedy output with the embedded MTP layer or the FastMTP sidecar is token-for-token the
plain greedy output.

### Streaming weights from flash (models larger than RAM)

`--stream` (Qwen3.5 / Qwen3.8, Gemma 4 / Gemma 3 including 26B-A4B's experts; implies `--mmap`) runs a model whose weights do not fit in memory
by reading them in place from the mapped file while a reader thread fetches the next layers. It
is aimed at phones: HauhauCS's Qwen3.8-27B GGUFs (Q2_K_P 10.7 GB, Q4_K_P 17.9 GB) on a 12 GB
iPhone. Every forward pass reads the units (decoder layers, LM head, MTP layer, draft head)
cyclically, so an LRU page cache smaller than the model never hits: each page is gone before its
next use. With `--cache-budget-gb F` oxinfer manages the cache itself instead: a fixed resident
set of `F` minus the read-ahead window stays cached (draft units first, since a round runs them
up to 8 times, then evenly spaced layers so the reader can work ahead while they compute) and
every other unit is dropped from the cache right after use. The page cache then holds the
resident set plus the window and nothing else, so each pass reads exactly the streamed bytes.

Darwin details that decided the design (measured on an M5 Max): a clean file page only leaves
the cache through `msync(MS_INVALIDATE)` (`madvise(MADV_DONTNEED / MADV_FREE)` leave it cached);
`fcntl(F_RDADVISE)` and `pread` read at 13-14 GB/s while demand faults from compute threads
manage 2-3 GB/s and `MADV_WILLNEED` ~5 GB/s, so the reader issues `F_RDADVISE` per 8 MB and then
touches the pages (mapping a 16 KB page costs ~1 us, so above ~7 GB/s the touching, not the SSD,
limits the reader); a full `mincore` costs ~15 ms per GB, so cached chunks are recognized from
three probe pages; `msync(MS_INVALIDATE)` itself costs ~50 ms per GB, so an evictor thread drops
used units instead of the compute thread (160 ms per pass on Q2_K_P at an 8 GB budget). Metal
wraps the mapped weights without copying for GPU prefill, but a live no-copy buffer keeps its
pages from being evicted, so streamed units get per-call buffers that are released after each
matmul (`Lin::set_transient_device`). A pass's embedding rows are read through the streamer
before the lookup (with priority over the read-ahead) rather than faulting in page by page.

| Option | Meaning |
|---|---|
| `--stream` | stream weights from the mapped file, reading ahead in execution order |
| `--cache-budget-gb F` | page cache the model may use (resident set = F - prefetch window; streamed units evicted after use) |
| `--resident-mb N` | resident set size instead of the budget's |
| `--prefetch-mb N` | read-ahead window (default 768) |
| `--resident-mode M` | `cache` (clean file pages, default), `lock` (`mlock`), `copy` (anonymous memory over the same addresses) |
| `--flash-mbps N` / `OXINFER_FLASH_MBPS` | emulate storage that reads N MB/s (the reader sleeps per uncached byte) |
| `--stream-io M`, `--stream-threads N` | `advise` (default on Apple), `pread` or `touch`; reader threads (default 2) |
| `--stream-trace` | one line per token / speculative round: bytes read vs cache hits, stall time |
| `--expert-cache-mb N` | MoE: page cache kept for recently used routed experts (default: what the budget leaves after the resident set) |
| `--expert-predict` | MoE: guess the next layer's experts and read them ahead (off by default, see below) |
| `--threads N`, `--qos C`, `--efficiency-threads N` | compute threads; their QoS class on Apple platforms (`interactive` default, `initiated`, `utility`, `background` = efficiency cores only); how many of them run at background QoS |

With `--mtp`, streaming defaults to up to 8 drafts with a cost-aware draft length
(`--mtp-cost`): each round picks the length with the most expected tokens per second from the
measured round time at each length and the measured acceptance at each draft position. When a
pass streams 5-12 GB, verifying 9 tokens costs about what verifying 1 does, so long drafts pay
off; in RAM the same policy settles on 2-3. Long drafts stay exact: `--mtp` verifies up to
`draft + 1` tokens on the batch-invariant decode kernels (`MtpOptions::exact`,
`kernels::set_decode_max_tokens`; greedy output checked token for token against plain greedy
over 160 tokens with 8 drafts, embedded layer and FastMTP sidecar). The rewind after a partial acceptance replays
the delta rule instead of keeping per-token states, so an 8-draft round needs one extra copy of
the recurrent state (151 MB on Qwen3.8-27B) rather than eight.

Verification passes of 3-16 tokens on GGUF weights run multi-token kernels (`kernels_ggml.rs`)
that unpack each weight super-block once for all tokens of the pass; before, the decode kernels
ran in token pairs and unpacked every block once per pair, which on HauhauCS's files (Q2_K_P is
mostly IQ3_XXS and IQ3_S grid formats, Q4_K_P mostly Q5_K) made unpacking the main cost. There
are two variants, both repeating the one-token kernels' f32 operations in the same order (the
integer sums are exact either way), so every row stays bit-identical whatever the token count
and the speculative greedy output equals plain greedy; no exactness switch is needed:

- SDOT, one weight row at a time against all tokens: Q4_K, Q5_K, IQ4_XS, and every format
  without i8mm;
- SMMLA (Armv8.6 i8mm, detected at run time: A15 / M2 and later, so the A19 Pro and the M5) on
  2x2 tiles of rows and tokens, the activations interleaved per token pair, from 4 tokens on:
  the formats with integer scales per 16 (Q2_K, Q3_K, Q6_K, IQ2/IQ3/IQ1), where SDOT needs many
  extra instructions to apply the scales. On the M5's cores SMMLA issues at half SDOT's rate
  for twice the multiply-adds (5 vs 10.5 G instructions/s per core, ~9-cycle latency) and does
  not overlap other vector work, so it buys nothing for the per-32 f32-scale formats.

`kernels::set_multi_token_kernels(MultiTokenKernels::{Auto, Dot, Pairs})` (or
`OXINFER_MULTI=auto|dot|pairs`) picks them (`Pairs` is the previous behaviour, for comparisons);
`kernels::cpu_has_i8mm()` reports the feature. One core, a 4096 x 5120 matrix, ns per 256
weights per token (`examples/kernel_bench.rs`):

| Format | 1 token | 4 tokens: pairs / SDOT / auto | 9 tokens: pairs / SDOT / auto | 9 tokens speed-up |
|---|---|---|---|---|
| IQ3_XXS | 20.6 | 11.3 / 6.9 / 6.0 | 12.2 / 5.2 / 4.1 | 2.9x |
| IQ3_S | 23.1 | 12.5 / 7.7 / 7.0 | 13.6 / 5.4 / 4.8 | 2.8x |
| Q2_K | 10.3 | 7.3 / 5.5 / 5.5 | 7.6 / 4.9 / 4.5 | 1.7x |
| Q3_K | 10.7 | 7.6 / 5.4 / 5.0 | 8.0 / 4.3 / 4.0 | 2.0x |
| Q6_K | 10.3 | 6.8 / 5.0 / 4.7 | 7.2 / 4.2 / 3.8 | 1.9x |
| Q4_K | 6.2 | 4.5 / 3.7 / 3.7 | 4.8 / 3.3 / 3.3 | 1.5x |
| Q5_K | 9.7 | 6.0 / 4.6 / 4.5 | 6.6 / 3.7 / 3.7 | 1.8x |
| IQ4_XS | 6.8 | 4.5 / 3.5 / 3.5 | 4.8 / 3.0 / 3.0 | 1.6x |

Whole verification passes (recording forward plus the LM head over every token,
`examples/pass_bench.rs`, the old and new kernels alternating pass by pass, best of 5):

| Model | Threads | 1 token | 2 tokens | 4 tokens: old / new | 9 tokens: old / new |
|---|---|---|---|---|---|
| Q2_K_P | 3 | 0.54 s | 0.65 s | 1.77 / 1.00 s | 3.04 / 1.14 s (2.7x) |
| Q2_K_P | 6 | 0.50 s | 0.54 s | 0.75 / 0.43 s | 1.74 / 0.67 s (2.6x) |
| Q4_K_P | 3 | 0.30 s | 0.46 s | 0.99 / 0.76 s | 1.71 / 1.03 s (1.7x) |
| Q4_K_P | 6 | 0.16 s | 0.21 s | 0.40 / 0.31 s | 0.91 / 0.59 s (1.55x) |

(Best of 5; the machine was shared, so single passes varied by up to 40%. One- and two-token
passes run the same kernels as before.)

The activation quantization for several tokens now runs vectorized and in parallel, with the
same values as before.

MoE models (Zawen-v0.1-35B-A3B and other Qwen3.5/3.6 MoE checkpoints) stream only the experts a
pass routes to: a MoE layer's unit holds its attention / DeltaNet weights, the router and the
shared expert, and each routed expert is an on-demand part of it (`UnitSpec::parts`) that the
layer requests once routing is known (`StreamHook::need_parts`), read with priority over the
read-ahead and then kept in a part cache (least recently used out) sized by what the budget
leaves (`StreamOptions::part_cache_bytes`). Zawen in oxinfer Q4 (19.5 GB), 6 GB budget at
2 GB/s, 3 threads, plain greedy: 0.13 -> 7.3 tok/s (14.2 -> 0.12 GB read per token: the dense
part, 1.1 GB, is resident and 78% of the requested experts come from the 4.1 GB part cache),
same tokens as without streaming. Guessing the next layer's experts from the residual stream
(`--expert-predict`) hides part of the remaining stall (77% -> 44% of the time) but the wrong
guesses read 27% more bytes, which the flash cannot afford: 6.9 tok/s, so it is off by default.

Measured on the M5 Max with the phone's limits emulated (`--cache-budget-gb`, `--flash-mbps`;
greedy, a 40-token answer to a short prompt, CPU decoding; "3 threads" approximates the
iPhone's 2 performance + 4 efficiency cores), before the multi-token kernels. Pass times were
within a few percent between repeated runs even with other jobs on the machine, since the
emulated storage dominates:

| Model | Budget | Flash | Resident / streamed per pass | Plain | `--mtp` (cost-aware, up to 8) |
|---|---|---|---|---|---|
| Q2_K_P | 4 GB | 1 GB/s | 3.1 / 7.1 GB | - | 0.35 tok/s (2.60 tok/pass) |
| Q2_K_P | 6 GB | 2 GB/s | 5.2 / 4.9 GB | 0.39 tok/s | 0.90 tok/s; 3 threads 0.91; FastMTP sidecar 0.73 |
| Q2_K_P | 8 GB | 3 GB/s | 7.1 / 2.9 GB | 0.91 tok/s; 3 threads 0.89 | 2.07 tok/s; 3 threads 1.57 (compute-bound: picks 1-4 drafts) |
| Q4_K_P | 4 GB | 1 GB/s | 3.0 / 13.9 GB | 0.069 tok/s | - |
| Q4_K_P | 6 GB | 2 GB/s | 5.0 / 12.0 GB | 0.15 tok/s | 0.39 tok/s; 3 threads 0.39; sidecar 0.31; fixed 1-3 drafts 0.37 |
| Q4_K_P | 8 GB | 3 GB/s | 7.0 / 10.2 GB | - | 0.66 tok/s |

A pass takes 1.04-1.14x the time the emulated storage needs for its bytes (88-97% of the
bandwidth used), and the page cache held by the model stays at or under the budget (sampled
with `mincore` at exit). Tokens per pass depend on the text, not on the IO: with 8 drafts every
round, 2.4-2.5 on prose, 2.8-3.3 on an explanation and 4.3-4.5 on code (embedded MTP layer a
little ahead of the FastMTP sidecar everywhere), so long drafts multiply the IO-bound speed by
2.3-4x. Sampling (`--temp 0.7`) streams the same way (1.03 tok/s at 6 GB / 2 GB/s on Q2_K_P).
Without a budget (`--stream` alone, the model cached) streaming costs nothing: 10.5 vs 10.9 tok/s
for `--mmap` on Q4_K_P.

With no throttle the Mac's SSD streams a 10 GB pass in ~1.3 s (7-8 GB/s, bounded by mapping the
pages). Prefill streams the weights once per chunk: on the GPU (`--device metal`, mapped weights
wrapped without copying) a 1138-token prompt on Q4_K_P at 6 GB / 2 GB/s took 13.9 s, near the
12 s the two passes' IO needs, against 53.7 s on the CPU.

Projected for an iPhone 17 Pro Max, assuming sustained sequential reads of 1-3 GB/s (iPhone NVMe
reads are usually quoted around 1.5-2.5 GB/s), a page-cache budget of 4-8 GB (the 12 GB device
minus the OS, the app's own memory and other apps), the measured 90% bandwidth efficiency and
2.6 tokens per pass (the table's prompt), with CPU decoding:

| Budget | Q2_K_P plain / `--mtp`, 1 GB/s | 2 GB/s | 3 GB/s | Q4_K_P plain / `--mtp`, 1 GB/s | 2 GB/s | 3 GB/s |
|---|---|---|---|---|---|---|
| 4 GB | 0.13 / 0.33 | 0.26 / 0.66 | 0.40 / 0.99 | 0.07 / 0.17 | 0.13 / 0.33 | 0.20 / 0.50 |
| 6 GB | 0.19 / 0.46 | 0.37 / 0.91 | 0.56 / 1.37 | 0.08 / 0.19 | 0.15 / 0.39 | 0.23 / 0.58 |
| 8 GB | 0.32 / 0.75 | 0.64 / 1.50* | 0.96 / 2.25* | 0.09 / 0.23 | 0.19 / 0.47 | 0.28 / 0.70 |

(tokens/s; * where the phone's CPU rather than the flash was the limit with the token-pair
kernels: on 3 Mac cores a 9-token Q2_K_P verification computed in 2.8 s, and at 8 GB / 3 GB/s the
cost-aware rounds took 1.48 s against 1.03 s of IO. The multi-token kernels bring that pass to
about 1.1-1.5 s, so the CPU is no longer far behind the flash; see below.)
Q2_K_P is the practical choice: about 1 token/s with 6 GB at 2 GB/s, 1.7-1.9 with 8 GB at 3 GB/s.

With the multi-token kernels (same prompt, 40 tokens, base = the commit before them; best of
two interleaved rounds on a machine shared with other jobs, which also read the same model files
and so sometimes lent cache hits to the IO emulation, mostly on Q4_K_P):

| Model | Budget / flash | MTP | base tok/s (s per pass) | new tok/s (s per pass) | GB read per pass |
|---|---|---|---|---|---|
| Q2_K_P | 6 GB / 2 GB/s | embedded | 0.82 (2.98) | 0.91 (2.84) | 5.1 |
| Q2_K_P | 6 GB / 2 GB/s | FastMTP sidecar | 0.74 (3.11) | 0.84 (2.90) | 5.2 |
| Q2_K_P | 8 GB / 3 GB/s | embedded | 1.39 (1.76) | 1.71 (1.43) | 3.1 |
| Q2_K_P | 8 GB / 3 GB/s | FastMTP sidecar | 1.19 (1.92) | 1.93 (1.27) | 3.2 |
| Q2_K_P, 6 threads | 8 GB / 3 GB/s | embedded | 1.75 | 2.00 | 3.1 |
| Q4_K_P | 6 GB / 2 GB/s | embedded | 0.46 | 0.40 | 10.0-11.9 |
| Q4_K_P | 6 GB / 2 GB/s | FastMTP sidecar | 0.34 | 0.31 | 9.6-12.1 |
| Q4_K_P | 8 GB / 3 GB/s | embedded | 0.77 | 0.67 | 7.3-9.3 |
| Q4_K_P | 8 GB / 3 GB/s | FastMTP sidecar | 0.54 | 0.54 | 8.6-9.0 |

(The Q4_K_P differences follow the bytes read per pass, which varied with the cache hits the
other jobs' reads of the same file lent: base 7.3 GB per pass against new 8.5 GB in the 8 GB
embedded runs, for example.)

Where the flash is the bottleneck (6 GB at 2 GB/s, and every Q4_K_P setting) nothing changes:
a pass already took about the time its bytes need. Where the CPU was (Q2_K_P at 8 GB / 3 GB/s)
the faster verification shows up directly, and the cost-aware policy drafts longer.
Greedy output was the same 40 tokens in every run (plain, embedded MTP layer, FastMTP sidecar,
old and new kernels), and `OXINFER_CHECK_SPEC` (every verification compared with one-token
forwards) found no difference over a 77-token answer with 8 drafts every round (embedded layer
and sidecar, Q2_K_P, which runs both kernel variants).

Phones have performance and efficiency cores (A19 Pro: 2 + 4). The pool hands out work block by
block, so a pool spanning both kinds lets the slower cores take fewer blocks; `--threads 6`
(the default `interactive` QoS, which iOS schedules on both kinds) is the setting to start from.
`pool::PoolOptions { threads, qos, efficiency_threads }` (`pool::init_with`, engine keys
`threads`, `qos`, `efficiency_threads`) sets the QoS class and keeps chosen workers on the
efficiency cores; `pool::cpu_cores()` reports the core counts. On the Mac (6 + 12 cores, no
efficiency cores in the phone's sense, loaded by other jobs) background-QoS threads were starved
by the other work (a 9-token Q2_K_P pass: 0.63 s on 6 interactive threads, 1.2 s with 4 of them
at background QoS, 6 s with all six), so the phone's balance has to be measured on the phone.

From the library (everything above is reachable without the CLI):

```rust
use oxinfer::qwen_mtp::{MtpDecoder, MtpOptions};
use oxinfer::stream::{StreamOptions, attach_qwen};

let files = ModelFiles::open(&path, false)?; // false: memory-mapped
let mut model = QwenModel::load(QwenConfig::load(&path)?, &files, Quant::Q4)?;
let mut mtp = model.load_mtp(&files)?;
let opts = StreamOptions { cache_budget: Some(6_000_000_000), ..Default::default() };
let streamer = attach_qwen(&mut model, Some(&mut mtp), &[&files], 8, opts)?;
let mut dec = MtpDecoder::new(&model, &mtp, MtpOptions { draft_tokens: 8, cost_aware: true, exact: true, stop, ..Default::default() });
// ... prefill / push, then per step:
let before = streamer.stats();
let tokens = dec.step(&mut sampler, max_new);
let io = streamer.stats() - before; // bytes_read, bytes_cached, stall, io_busy, ...
```

`Streamer::plan()` reports the resident set and the bytes streamed per pass (and any weight that
is a RAM copy instead of a view of the file, and the on-demand expert bytes and their cache),
`cached_bytes()` the model's current page-cache footprint, `set_flash_bytes_per_sec` changes the
emulated bandwidth while running, `StreamStats` also counts expert requests and part-cache hits
and the eviction time, and `MtpStats::costs` holds what the draft policy measured. The engine
API takes the same settings as keys (`expert_cache_mb`, `expert_predict`, `qos`,
`efficiency_threads`).

## Platforms and accelerators

The CPU path is the reference and works everywhere; everything else is detected at run time and
loaded with [`dlsym`](https://crates.io/crates/dlsym), so one binary runs with or without the
libraries present and nothing GPU-related is needed at build time. `--devices` prints what was found:

```
CPU: aarch64 on macos, 18 logical cores
  features: neon dotprod i8mm bf16 fp16
  memory: 137.4 GB
GPU (metal #0): Apple M5 Max, memory 115.4 GB (unified with CPU)
```

| Platform | Compute | Status |
|---|---|---|
| Apple Silicon (macOS, iOS) | NEON kernels + Metal GEMM for prefill; on M5 / A19 the GPU matrix units (prefill GEMMs and MoE experts); single-token decode entirely on the GPU (Llama 3, Qwen3 / 3.5 / 3.8 incl. MoE, Gemma 3 / 4 incl. 26B-A4B) | measured; the f32 GEMM gives bit-identical logits to the CPU path, the matrix units (default where present) ~3e-4 relative per product (`OXINFER_METAL_MMA=0` for bit-identical); iOS: runs in the Simulator (CPU and Metal; `ios/`) |
| Apple Neural Engine (macOS 15 / iOS 18) | Core ML, for ANEMLL-converted models | measured on macOS; builds for iOS; see [Core ML](#core-ml--apple-neural-engine) |
| x86-64 (Linux, Windows, macOS) | AVX2 + FMA kernels (bf16, Q8 via `maddubs`, Q4), scalar fallback below AVX2 | compiles for all three; the kernel tests pass on the scalar fallback under Rosetta (which exposes no AVX), so the AVX2 kernels themselves are not yet executed anywhere |
| Linux / Windows + NVIDIA | CUDA driver API + NVRTC-compiled GEMM (`cuda.rs`) | compile-checked only, not yet run on NVIDIA hardware |
| Android (aarch64) | NEON kernels | cross-compiles (`aarch64-linux-android`) |
| Vulkan | detection only (device name and memory) | no compute yet |
| wasm32 (WASI) | simd128 kernels (bf16, Q8, Q4), threads via wasi-threads | measured: Qwen3.5-0.8B Q8 at 123 tok/s decode with 16 threads under wasmtime 30, identical output to native; see below |
| Browser (Chrome 137+) | simd128 kernels + WebGPU GEMM for prefill (WGSL, via JSPI) | measured in headless Chrome 155: Qwen3.5-0.8B Q8 prefill 34 -> 231 tok/s, identical output; Firefox/Safari run the CPU path (no JSPI) |

What the GPU does and does not do: every backend only takes prefill-sized matmuls (`t >= 16`
tokens and enough FLOPs to amortize the ~0.15 ms it costs to submit and wait for a Metal command
buffer). Weights are not copied on Apple Silicon: quantized matrices and the safetensors blob
live in page-aligned memory and are wrapped as shared `MTLBuffer`s. The Metal GEMM (64x64 tile per
threadgroup, `simdgroup_float8x8`, weights dequantized to f32 in threadgroup memory) runs at
~10 TFLOP/s for bf16 and ~7.5 for Q4, against ~1.3 TFLOP/s for the NEON GEMM.

On M5 / A19 (Apple10-family GPUs, whose cores have matrix units) the GEMMs run on those instead,
through Metal 4 tensor operations (`matmul2d` from MetalPerformancePrimitives, compiled at run
time with the rest of the shaders; nothing is needed at build time). Input rows are converted to
half with a power-of-two scale per row (no overflow, no subnormals), quantized weights are
dequantized per 64-wide K step into threadgroup memory as half, bf16 weights are read as bfloat
straight from device memory, and accumulation is f32: every format with a GEMM (bf16, Q8, Q4,
the GGUF types) has a matrix-unit version. Kernel throughput at 2048 tokens is ~60 TFLOP/s for
bf16 and ~50 for Q4 (against ~11 for the f32 GEMM); per call, copies and submission included
(`membench gpu`, 17408 x 5120):

| Tokens | bf16 f32 GEMM / matrix units | Q4 | Q4_K | Q8_0 |
|---|---|---|---|---|
| 256 | 9.7 / 10.5 TFLOP/s | 11.0 / 19.8 | 10.7 / 18.1 | 10.9 / 18.0 |
| 512 | 11.7 / 24.4 | 11.6 / 28.9 | 11.5 / 28.2 | 11.5 / 28.3 |
| 1024 | 12.0 / 41.4 | 11.6 / 42.1 | 11.7 / 38.7 | 11.7 / 37.1 |
| 2048 | 12.2 / 43.8 | 11.8 / 42.6 | 11.5 / 39.6 | 12.1 / 40.6 |

The products are accurate to ~3e-4 relative (half operands) instead of bit-identical to the CPU;
`OXINFER_METAL_MMA=0`, the engine option `gpu_exact=1` or `oxinfer::gpu::set_exact(true)` keep
the f32 GEMM (`gpu_moe=0` / `set_moe_offload(false)` keep the experts on the CPU). MoE layers (Qwen3.5/3.6-35B-A3B) run their routed
experts in one submission per layer: the tokens are gathered into one segment per expert, a
grouped GEMM walks per-expert tiles for the gate/up projection, then SiLU, the grouped down
projection and the weighted combination, all on the GPU (the expert weights are kept resident
with an `MTLResidencySet`; making them resident per submission cost more than the GEMMs).

Single-token decode runs entirely on the GPU where the model and its weight formats are supported
(`--gpu-decode on`, the default; `off`, the engine / speaker option `gpu_decode=0` or
`oxinfer::gpu::set_decode(Some(false))` keep it on the CPU): the model describes one token's
forward pass as a graph of device ops (`gpu::decode`), the Metal backend compiles it once into
kernels and buffer bindings, and each token is one command buffer (plus a small first one, so the
GPU starts while the rest is encoded) with a single wait. The host only looks up the input
embedding (and Gemma's per-layer-embedding row) and samples from the logits. What runs on the GPU:

| Model family | On the GPU per token |
|---|---|
| Llama 3.x (safetensors, GGUF), Orpheus TTS | RMSNorm, Q/K/V matvecs, RoPE (rotate-half or GGUF's interleaved pairs), KV append, attention, output projection + residual, fused SiLU gate/up, down + residual, final norm, LM head (skipped when only the hidden state is wanted, as Orpheus's restricted audio head runs on the CPU) |
| Qwen3 (dense) | the same plus per-head Q/K RMSNorm |
| Qwen3.5 / 3.6 / 3.8 (dense) | full attention layers with the output gate and partial RoPE; Gated DeltaNet layers: in/z/b/a projections, causal conv + SiLU over the conv state, the gated delta rule (state rows in registers, read and written once) with the gated RMSNorm; the recurrent states are the host's own page-aligned memory, updated in place |
| Qwen3.5 MoE (Zawen-35B-A3B, ...) | router and the shared expert's gate as one f32 matvec, top-k + softmax, the chosen experts' gate/up and weighted down matvecs read through per-expert address tables (fused or GGUF split gate/up), the shared expert scaled by its sigmoid gate |
| Gemma 4 E2B / E4B / 26B-A4B, Gemma 3 | per-layer embeddings, sliding and full attention (two RoPE tables, head size up to 512), KV sharing across layers, value norms, K = V layers, sandwich norms (each fused with the next norm), GELU-tanh MLPs, 26B-A4B's routed GELU experts beside the dense MLP (router scale, per-expert scales), layer scalars, the softcapped head |

Matvec kernels exist for bf16, oxinfer's Q8 and Q4 (also MLX affine Q4) and every GGUF format
with a GEMM (specialized kernels for F16, Q4_0, Q8_0, Q4_K, Q5_K and Q6_K; the others through
their GEMM dequantizers): one SIMD-group or more per row, lanes taking consecutive chunks of a
row so a SIMD-group reads contiguous memory, nibbles masked rather than shifted (the powers of
two are divided out once per word), K-quant scales decoded without branches; 450-590 GB/s on
large matrices of an M5 Max (`cargo test --release --lib decode_mv_bench -- --ignored`).
Attention scores 32 keys per SIMD-group step (one lane per key), reads each key and value row
once for the query heads sharing it, and splits long contexts over up to 64 threadgroups per head
group. Activations stay f32 end to end (the CPU kernels quantize them to int8 for the quantized
formats), and so do the KV caches: the host caches stay authoritative and the device keeps a
copy per session, appending what CPU passes added (prefill chunks, speculative verification) and
forgetting what a rewind dropped. Not on the GPU: models with LoRA adapters attached, streamed
weights (`--stream`), bidirectional configurations, the vision / audio towers,
multi-token passes (prefill and speculative verification stay on the existing paths), and
CUDA / WebGPU (no decode graph backend yet).

Before that, forwarding only the matmuls of decode passes to the GPU was measured and lost: on
the 27B GGUF (`OXINFER_METAL_MIN_T=1`) a pass cost ~257 submissions with CPU work in between
(~50 ms of submit-and-wait alone), so plain decode dropped from 10.7 to 5.5 tok/s.

### WebAssembly

`.cargo/config.toml` turns on `simd128` for the wasm targets (every current engine supports it;
without it the scalar kernels are used, ~8x slower on Q8 decode) and raises the shared-memory
maximum for the threads target to 4 GiB. The model directory is passed in as a preopened directory:

```sh
# single-threaded, runs under Node's built-in WASI (node --experimental-wasi ... on older versions)
cargo build --release --target wasm32-wasip1 --no-default-features
# threads (wasi-threads): wasmtime up to v30 runs it; v31+ dropped the wasi-threads proposal
cargo build --release --target wasm32-wasip1-threads --no-default-features
wasmtime run -W threads=y -S threads=y --dir ~/.cache/oxinfer/hub/Qwen--Qwen3.5-0.8B::/model \
  target/wasm32-wasip1-threads/release/oxinfer.wasm --model /model --quant q8 -j 16 -p "Hello"
```

Qwen3.5-0.8B Q8, 18-core M5 Max, 17-token prompt:

| Build | Threads | Prefill | Decode |
|---|---|---|---|
| wasm32-wasip1, scalar | 1 | 9 tok/s | 4.6 tok/s |
| wasm32-wasip1, simd128 | 1 | 29 tok/s | 40 tok/s |
| wasm32-wasip1-threads, simd128 | 16 | 327 tok/s | 123 tok/s |
| native (NEON) | 18 | 373 tok/s | 184 tok/s |

Memory is 32-bit, so a model plus its quantized copy must fit in 4 GiB (`--quant q4` on a 2B
model is fine; the 27B models are not). No file mapping exists in WASI, so the checkpoint is
always read into memory.

### Browser (WebGPU)

The site has a Chat mode and an Image mode. Image mode loads SD 1.5 with the LCM-LoRA by default
(2.1 GB of fp16 weights plus 135 MB from Hugging Face; 4 Euler ancestral steps without guidance,
about 8 s per 512x512 image in Chrome on the M5 Max with the machine busy, 63 dB from the native
Metal image); SD-Turbo (2.6 GB, one step) is the other preset, and a custom model can take LoRA
files too (`?lora=org/name/file.safetensors:0.8`, comma separated). With the LCM-LoRA, plain Euler
leaves speckle noise in the image: LCM is trained to be re-noised between steps, as the ancestral
sampler does. A third preset, InstructPix2Pix (with the same LCM-LoRA, 4 steps), edits an
uploaded picture by instruction, and every model takes an input image and a mask (image-to-image,
inpainting; see [Editing images](#editing-images)); headless Chrome produced the same edit as the
native build. Diffusion weights stay in JavaScript memory and the module converts
them tensor by tensor, so only the bf16 copy counts against the 4 GiB of wasm memory.

The `web` feature (`src/web/`) turns the library into a plain C-ABI wasm module, no bindgen;
`cargo build-web` builds it as a `cdylib` for `wasm32-unknown-unknown` (the crate type is given on
the command line, so native builds never link a shared library). `web/` is the site deployed to
[GitHub Pages](https://karpeleslab.github.io/oxinfer/) by `.github/workflows/pages.yml`: a Vue 3 +
Vite app around the JavaScript host `web/src/lib/oxinfer.js`; it is not a crate. Downloads go to
the browser's Cache Storage, so a model is fetched once per browser. MLX checkpoints load too
(Gemma 4 E2B and E4B, Gemma 3 4B and Qwen3-4B from the mirror were checked in headless Chrome):
the host splits each one into shards of at most 1 GiB kept in JavaScript memory as it streams in,
since a 5 GB file fits neither one allocation nor a 32-bit module, and the module converts the
matrices into its own memory, then lets the host drop every shard but the embedding tables. The host registers the model files in the module's in-memory filesystem (streamed straight from
`fetch` or a local folder into module memory) and drives a chat session token by token.

GPU work goes the other way round: the module imports `gpu_upload` / `gpu_matmul`, which the
host implements with WebGPU (WGSL GEMM, 64x64 tile per workgroup, weights dequantized to f32 in
workgroup memory; ~3.9 TFLOP/s at 4096^3 on the M5 Max, ~0.7 ms per round trip). The module's
matmul call is synchronous while the buffer readback is asynchronous, so `gpu_matmul` is wrapped
in `WebAssembly.Suspending` and the generating exports in `WebAssembly.promising` (JavaScript
Promise Integration, Chrome 137+). Without JSPI or WebGPU the same page runs on the CPU.
WebGL is not used: it has no compute shaders.

```sh
cd web
npm install
npm run wasm                                       # cargo build-web, copied to public/oxinfer.wasm
npm run smoke                                      # the JS <-> wasm bridge, headless (Node)
OXINFER_MODELS=~/.cache/oxinfer/hub npm run dev    # local folders load as "./models/<folder>"
# or from Deno directly (Deno has WebGPU, but its readback polls every ~14 ms, so --cpu is as fast there):
deno run --allow-read --allow-env scripts/run.mjs --model ../models/gemma-4-E4B-it-q4 --cpu -p "Hello"
```

Point the page (or `run.mjs`) at a pre-quantized checkpoint (`--save-quantized`, see above): it
downloads only the quantized bytes and uses them in place, and its `model-tables-*` shards stay in
JavaScript memory, read one row per token. That is what makes the larger models fit: E4B at Q4
puts 2.6 GB in wasm memory and 1.6 GB of per-layer embeddings in JavaScript. Plain bf16
checkpoints still work for small models (they are quantized in the module, which then holds both
copies).

Headless Chrome 155, Qwen3.5-0.8B Q8, 123-token prompt, single CPU thread (the browser build has
no threads):

| | Prefill | Decode |
|---|---|---|
| CPU (simd128) | 34 tok/s | 36 tok/s |
| WebGPU prefill, first turn (includes the weight upload) | 165 tok/s | 38 tok/s |
| WebGPU prefill, steady state | 231 tok/s | 38 tok/s |

Gemma 4 from pre-quantized Q4 files, same page, same 115-token prompt (second turn, weights
already on the GPU); the output text is identical with and without WebGPU, and 40 greedy tokens
from the wasm build match the native binary on the same file:

| Model | CPU prefill | WebGPU prefill | Decode |
|---|---|---|---|
| E2B-it Q4 | 10 tok/s | 124 tok/s | 7.7 tok/s |
| E4B-it Q4 | 5 tok/s | 125 tok/s | 4.0 tok/s |

Limits: decode is single-threaded CPU and bandwidth-bound (one core streams ~10 GB/s of Q4
weights), and stays off the GPU for the same reasons as on Metal. Weights in module memory must
fit in 4 GiB, and no single file can exceed 2 GiB. `web/scripts/test_gemm.mjs` checks the WGSL kernels
against a JavaScript reference.

Feature flags: `media` (oxideav image/audio/video file decoding), `hf` (HuggingFace fetching),
`gpu` (dlsym-loaded backends), all on by default; `oxideav` (oxideav-core types in the API, implied
by `media`; see [Using oxinfer from oxideav](#using-oxinfer-from-oxideav-and-other-crates)), `capi`,
and `web` (the browser module, off by default and only meaningful for `wasm32-unknown-unknown`);
`--no-default-features` gives the plain CPU build.
Cross-compiling needs no C toolchain: the tokenizer uses its pure-Rust regex backend.

### Core ML / Apple Neural Engine

Models converted by [ANEMLL](https://github.com/Anemll/Anemll) (the `anemll/*` repos on Hugging
Face: Qwen3, Qwen2.5, Gemma 3, Llama 3.2, VibeThinker, ...) are pre-compiled Core ML packages with
fixed shapes and a stateful KV cache, laid out for the Neural Engine. oxinfer runs them through
Core ML itself (`coreml.rs`: `CoreML.framework` via the Objective-C runtime and `dlsym`, nothing
linked), following ANEMLL's runtime contract (`anemll.rs`: `meta.yaml`, chunked / monolithic /
split-rotate layouts, batch prefill, causal masks, Gemma 3 sliding-window rotate functions, split
LM head or argmax in the model), and the extensions of oxinfer's own converter (below): Qwen3.5
recurrent states, Gemma 4 per-layer inputs and two caches. Needs macOS 15 / iOS 18 on Apple
Silicon (stateful and multifunction models); the module also builds for `aarch64-apple-ios`, and
`oxinfer::engine::Engine` loads these models like any other.

```sh
oxinfer --model anemll/anemll-Qwen-Qwen3-0.6B-ctx512_0.3.4 --prompt "Why is the sky blue?"
oxinfer --model anemll/anemll-google-gemma-3-1b-it-ctx4096_0.3.5 --chat
oxinfer --model DIR --coreml-units cpu      # ane (default: CPU + Neural Engine) | all | gpu | cpu
oxinfer --model DIR --coreml-prefill pad    # auto (default) | batch | pad | single
```

The whole repository is fetched (the `.mlmodelc` directories, tokenizer, `meta.yaml`); the first
load of a model compiles it for the device (30 s for Qwen3-0.6B, 2 min for Gemma-3-1B with its four
functions), later loads take 0.3-1 s. Prompt formatting, sampling, `--chat`, `--think`, `--system`,
`--raw`, `--max-tokens`, `--dump-logits` and `--print-prompt` work as with the native paths, but
models converted with `argmax_in_model` (the Gemma 3 and Llama repos above) return only their top
token: generation is greedy and there are no logits to dump. The context is fixed at conversion
(`ctx512` etc.); a longer prompt is an error and generation stops when the cache is full. The same
API is in the library (`oxinfer::anemll::AnemllModel::{load, prefill, step}`, see
`examples/anemll.rs`).

**Verified** against ANEMLL's own runtime (`tests/chat.py` with coremltools 9): same prompt tokens,
same greedy tokens, and bit-identical logits where the model has them:

| Repo (anemll/...) | Layout | Checked |
|---|---|---|
| anemll-Qwen-Qwen3-0.6B-ctx512_0.3.4 | 1 chunk | 60 + 100 tokens, logits identical, both prefill modes |
| anemll-Qwen-Qwen3-1.7B-ctx2048_0.3.5 | 2 chunks | 300 tokens, logits identical |
| anemll-WeiboAI-VibeThinker-1.5B-ctx2048_0.3.5 | 4 chunks (Qwen2.5) | 300 tokens, logits identical |
| anemll-google-gemma-3-1b-it-ctx4096_0.3.5 | 1 chunk, sliding window 512, argmax | 901-token prompt + 3150 tokens (to 4051 of 4096, across the window) |
| anemll-meta-llama-Llama-3.2-1B-Instruct-ctx1024_0.3.5 | 1 chunk, argmax | 300 tokens |
| anemll-Qwen-Qwen2.5-0.5B-Instruct-ctx2048-monolithic_0.3.5 | monolithic | 300 tokens, logits identical, both prefill modes |
| anemll-google-gemma-3-270m-it-ctx512-monolithic_0.3.5 | monolithic, argmax | 250 tokens, both prefill modes |

Prefill modes: full batches (64 tokens) go through the `prefill` functions and the rest through
`infer`, token by token, as in ANEMLL. `auto` does this only when `meta.yaml` declares
`prefill_dynamic_slice` or `update_mask_prefill` and otherwise feeds every token through `infer`, as
`chat.py` does: the batch prefill of older conversions can be wrong (the Gemma 3 270m monolithic
repo above produces garbage with it, in `chat.py` too, while Qwen3-0.6B 0.3.4 gives bit-identical
logits either way). `pad` also runs the last partial batch through `prefill`, padded: the
prediction is not bit-identical any more (greedy outputs diverged after 70 and 162 tokens on Llama
and Qwen2.5, stayed identical on Qwen3 and Gemma 3) but short prompts no longer pay one Neural
Engine call per token.

Compute plans (`MLComputePlan`) put 99.7-100% of the estimated cost of the `infer` functions on
the Neural Engine (2333 of 2340 ops for Qwen3-0.6B; the embedding lookups run on the CPU, and the
Gemma 3 argmax head is half CPU).

Measured on the M5 Max (macOS 26), 227-250-token prompt, 200 generated tokens, best of 2-3 runs
(the machine was shared):

| Model (Neural Engine) | Prefill (auto) | Prefill (pad) | Decode | Decode, back to back with `chat.py` | RSS | CPU time |
|---|---|---|---|---|---|---|
| Qwen3-0.6B ctx512 | 128 tok/s (715 batched) | 3630 tok/s | 106 tok/s | 114 vs 104 tok/s | 205 MB | 0.5 s |
| Gemma-3-1B ctx4096 | 483 tok/s (3250 at 901 tokens) | 2720 tok/s | 69 tok/s | 70 vs 64 tok/s | 630 MB | 1.5 s |
| Llama-3.2-1B ctx1024 | 284 tok/s | 2270 tok/s | 60 tok/s | 52 vs 51 tok/s | 170 MB | 0.5 s |
| Qwen2.5-0.5B monolithic ctx2048 | 358 tok/s | 2120 tok/s | 90 tok/s | 77 vs 75 tok/s | 170 MB | 0.5 s |
| Gemma-3-270m monolithic ctx512 | 261 tok/s | (wrong output) | 263 tok/s | 233 vs 221 tok/s | 560 MB | 0.8 s |

Decode slows down as the context fills (Gemma-3-1B: 67 tok/s averaged over 4050 tokens). Other
compute units on the same Core ML models, and oxinfer's own kernels on the original checkpoints
(`Qwen/Qwen3-0.6B`, `unsloth/gemma-3-1b-it` = google/gemma-3-1b-it):

| | Qwen3-0.6B prefill | decode | RSS | Gemma-3-1B prefill | decode | RSS |
|---|---|---|---|---|---|---|
| Core ML `ane` (CPU + Neural Engine) | 715 tok/s | 106 tok/s | 0.2 GB | 483 tok/s | 70 tok/s | 0.6 GB |
| Core ML `cpu` | 329 tok/s | 48 tok/s | 2.2 GB | 207 tok/s | 22 tok/s | 3.9 GB |
| Core ML `gpu` (CPU + GPU) | 28 tok/s | 21 tok/s | 8.6 GB | 18 tok/s | 14 tok/s | 14.2 GB |
| Core ML `all` | 26 tok/s | 21 tok/s | 8.6 GB | 15 tok/s | 17 tok/s | 14.2 GB |
| oxinfer CPU bf16 / q8 / q4 | 817 / 783 / 751 tok/s | 135 / 174 / 265 tok/s | 1.7-2.4 GB | 515 / 532 / 477 tok/s | 79 / 146 / 216 tok/s | 2.4-3.5 GB |
| oxinfer Metal prefill, bf16 / q4 | 891 / 901 tok/s | 137 / 259 tok/s | 1.7-2.1 GB | 782 / 760 tok/s | 82 / 228 tok/s | 2.4-3.0 GB |

On this chip the CPU cores are faster than the Neural Engine for these small models (and the
ANEMLL conversions use 6-bit palettized weights, so quality is not equal either). What the Neural
Engine buys is everything else: the generation above takes 0.5 s of CPU time against 13.6 s for
oxinfer's Q4 CPU path (1.3 s wall on 18 threads), the CPU and GPU stay free, and the process stays
at 0.2-0.7 GB resident, less than the weights themselves (0.8 GB for Qwen3-0.6B with its fp16
embedding table): the Neural Engine's copy of the compiled model is not counted in the process.
The `gpu` and `all` units expand the palettized weights (8-14 GB resident) and are slow; use the
native Metal path for the GPU instead. Over a 4050-token Gemma-3-1B generation the RSS stays flat
(680 MB at the start, 570 MB at the end): every prediction runs in its own autorelease pool and the
input / output arrays are allocated once per model.

Speed comes from wiring the parts together once at load: each part's output (an IOSurface-backed
fp16 `MLMultiArray`, what the Neural Engine writes directly) is bound as the next part's input in a
prepared feature provider, so a token only writes its id, position and mask row into existing
arrays and runs the chain; decode runs 2-10% faster than `chat.py` on the same models.

#### Converting models

`tools/coreml_convert.py` converts Hugging Face checkpoints into Core ML packages for the
Neural Engine of iPhones and Macs, in the layout of [ANEMLL](https://github.com/Anemll/Anemll)
(meta.yaml, an embeddings model, decoder chunks with `infer` and `prefill` functions sharing
weights and a Core ML state, a 16-way split LM head, the tokenizer files). Qwen3 packages run
unchanged under ANEMLL's own `chat.py`; Qwen3.5 / 3.6 / 3.8 models (Gated DeltaNet recurrent
state kept as Core ML states, chunked delta rule in the prefill function; MoE variants evaluated
densely, for small models) and Gemma 4 E2B / E4B (per-layer embeddings, KV sharing, local and
global caches) extend the format as described in [docs/coreml-format.md](docs/coreml-format.md);
oxinfer's Core ML runtime above runs them. `tools/coreml_check.py` runs a package with coremltools,
compares it with transformers in float32 over a few hundred teacher-forced tokens, benchmarks it
and lists which ops land on the ANE.

```sh
python3 -m venv venv && venv/bin/pip install coremltools==9.0 torch transformers safetensors \
    scikit-learn pyyaml "numpy<2.3"
venv/bin/python tools/coreml_convert.py --model ~/.cache/oxinfer/hub/Qwen--Qwen3.5-0.8B --output out/qwen35
venv/bin/python tools/coreml_convert.py --model ~/.cache/oxinfer/hub/Qwen--Qwen3.5-0.8B --output out/qwen35-small \
    --ffn-quant lut4 --lut-mix 4.5        # 25% smaller decoder
venv/bin/python tools/coreml_check.py out/qwen35 --hf ~/.cache/oxinfer/hub/Qwen--Qwen3.5-0.8B --bench --plan
```

**Weights.** Only LUT palettization runs correctly on the ANE (Core ML has LUTs of 1, 2, 3, 4, 6
and 8 bits, no 5), 8 output channels per LUT. Core ML's own palettizer fits each LUT with plain
k-means, which costs 15-20 points of top-1 agreement at 4 bits on sub-1B models. The converter
instead computes the LUTs from a calibration set by default (`--lut-method fisher`,
`tools/coreml_quant.py`): the float32 transformers model answers 32 built-in chat prompts
(under a minute on MPS), the squared gradients of the next-token loss give the Fisher information of
every weight, and each LUT is a k-means weighted by it (SqueezeLLM's sensitivity), so the levels
go where the loss is sensitive. `--lut-mix B` adds mixed precision: the weights whose 4-bit
palettization costs the most (estimated 0.5 Σ F Δw²) get 6-bit LUTs until the decoder averages B
bits. Things that did not pay off: GPTQ-style error feedback over the LUT grid (the per-layer
error drops 4x on the calibration data but rises on other text: it overfits a few thousand
tokens), per-channel scales on top of the LUT, 3x more calibration data. One LUT per output
channel is more accurate (Qwen3-0.6B LUT4: 85% top-1) but halves the ANE's decode speed.

Measured on an M5 Max with coremltools (context 512, batch 64; quality = the Core ML logits
against transformers float32 over 300 teacher-forced tokens of a 135-token chat prompt; speed =
best of 5, 256-token prefill and 64-token decode including the Python calls, on a shared
machine; package sizes are embeddings + decoder chunks + LM head):

| Model | Weights (decoder, LUT fit) | Size (MB) | Logit cosine | Top-1 agreement | KL | Prefill ANE / CPU | Decode ANE / CPU |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | fp16 | 311 + 884 + 311 | 0.99999 | 98.7% | | 2680 tok/s | 79 tok/s |
| Qwen3-0.6B | LUT6, k-means | 311 + 338 + 119 | 0.9972 | 93.0% | | 5131 / 1949 tok/s | 114 / 56 tok/s |
| Qwen3-0.6B | LUT6, Fisher (default) | 311 + 338 + 119 | 0.9984 | 93.7% | 0.016 | 5170 tok/s | 116 tok/s |
| Qwen3-0.6B, ANEMLL's release | LUT6 | 311 + 339 + 119 | 0.9971 | 92.3% | | 4409 / 1824 tok/s | 114 / 59 tok/s |
| Qwen3-0.6B | LUT4, k-means | 311 + 224 + 119 | 0.954 | 72.0% | | 4836 tok/s | 103 tok/s |
| Qwen3-0.6B | LUT4, Fisher | 311 + 224 + 119 | 0.9665 | 79.7% | 0.224 | 5219 tok/s | 116 tok/s |
| Qwen3-0.6B | LUT4, Fisher, mix 4.5 bits | 311 + 253 + 119 | 0.9813 | 84.7% | 0.153 | 5218 tok/s | 116 tok/s |
| Qwen3-0.6B | LUT4, Fisher, mix 5 bits | 311 + 281 + 119 | 0.9876 | 86.3% | 0.085 | 5229 tok/s | 116 tok/s |
| Qwen3-0.6B | LUT4, Fisher, 1 channel per LUT | 311 + 234 + 119 | 0.9850 | 85.3% | 0.141 | 2565 tok/s | 47 tok/s |
| Qwen3.5-0.8B | fp16 | 509 + 1000 + 509 | 0.9998 | 97.0% | | 4002 tok/s | 62 tok/s |
| Qwen3.5-0.8B | LUT6, k-means | 509 + 385 + 195 | 0.9974 | 93.0% | | 5176 / 1608 tok/s | 118 / 37 tok/s |
| Qwen3.5-0.8B | LUT6, Fisher (default) | 509 + 384 + 195 | 0.9974 | 94.7% | 0.011 | 5319 tok/s | 136 tok/s |
| Qwen3.5-0.8B | LUT4, k-means | 509 + 254 + 195 | 0.959 | 78.3% | | 5035 tok/s | 113 tok/s |
| Qwen3.5-0.8B | LUT4, Fisher | 509 + 255 + 195 | 0.9654 | 83.0% | 0.154 | 5450 tok/s | 136 tok/s |
| Qwen3.5-0.8B | LUT4, Fisher, mix 4.5 bits | 509 + 286 + 195 | 0.9823 | 85.3% | 0.083 | 5406 tok/s | 136 tok/s |
| Qwen3.5-0.8B | LUT4, Fisher, mix 5 bits | 509 + 319 + 195 | 0.9894 | 89.0% | 0.053 | 5356 tok/s | 137 tok/s |
| Gemma 4 E2B | fp16 | 5530 + 3740 + 805 | 1.00000 | 99.3% | | | |
| Gemma 4 E2B | LUT6, k-means | 5530 + 1419 + 306 | 0.99986 | 97.0% | | 1905 / 623 tok/s | 45 / 14 tok/s |
| Gemma 4 E2B | LUT6, Fisher (default) | 5530 + 1419 + 306 | 0.99987 | 97.7% | 0.010 | 1976 tok/s | 46 tok/s |
| Gemma 4 E2B | LUT4, k-means | 5530 + 942 + 306 | 0.99771 | 83.7% | 0.282 | 2010 tok/s | 47 tok/s |
| Gemma 4 E2B | LUT4, Fisher | 5530 + 942 + 306 | 0.99799 | 84.7% | 0.226 | 2023 tok/s | 47 tok/s |
| Gemma 4 E2B | LUT4, Fisher, mix 4.5 bits | 5530 + 1061 + 306 | 0.99888 | 91.0% | 0.068 | 2010 tok/s | 47 tok/s |
| Gemma 4 E4B | LUT6, k-means | 7034 + 2992 + 508 | 0.99945 | 96.7% | | 1060 / 333 tok/s | 25 / 6 tok/s |

The LM head stays LUT6 in every row (LUT4 there costs another 3-6 points). The Fisher packages of
Qwen3.5 decode faster than the k-means ones because the small depthwise convolution of the Gated
DeltaNet stays fp16 instead of being palettized. Everything
but the embedding lookups and the RoPE-table gathers runs on the ANE (MLComputePlan), including
the Gated DeltaNet recurrence and its chunked prefill. The Gemma embeddings model is large
because the per-layer embedding table stays fp16: Core ML memory-maps it and pages in only the
rows it reads, while a palettized table would be decompressed at load time (16.7 GB for E2B).
"CPU" is Core ML's CPU_ONLY path, whose outputs are NaN for these stateful fp16 graphs (ANEMLL's
packages too), so it only shows what the ANE saves. On a Max-class Mac, oxinfer's own CPU path
decodes faster (Qwen3.5-0.8B Q8: 184 tok/s); what the Neural Engine brings is prefill throughput
and power, with the CPU and GPU left free.

The same packages through oxinfer (`oxinfer --model out/qwen35`, or `Engine::load`): the logits
are bit-identical to coremltools' predictions of the package with `coreml_check.py`'s call
sequence (300 teacher-forced positions of the 135-token prompt, also with the prompt cut to 128
tokens so it ends on a batch boundary, and token-by-token prefill), greedy generation is identical
token for token, and so are three-turn chats through `Engine` replayed by the Python driver (the
Qwen3.5 state carried across turns: logit cosine 0.9975 against transformers over the whole
conversation, as for one prompt). The runtime follows the format's rules: every token fed exactly
once, the prediction from the call that processed the last one (`prefill`'s last-token output on a
batch boundary), no padded batches into the recurrent state, a reset makes a new state. For the
transformer packages `auto` pads the last partial batch, which here gives the same logits as
feeding those tokens one by one. M5 Max, Neural Engine, 227-231-token prompt + 200 tokens,
best of 4 with another job using 13 CPU cores (runs varied by 5-15%; the coremltools driver
measured under the same load is within that noise of oxinfer):

| Package | Load (cached / first) | Prefill | Decode | RSS | CPU time |
|---|---|---|---|---|---|
| Qwen3-0.6B LUT6 | 0.3 s / 46 s | 3070 tok/s | 95 tok/s | 215 MB | 0.6 s |
| Qwen3.5-0.8B LUT6 | 0.4 s / 90 s | 650 tok/s (3430 on a 256-token prompt) | 95 tok/s | 235 MB | 0.8 s |
| Gemma 4 E2B LUT6 | 0.5 s / 100 s | 1030 tok/s | 35 tok/s | 595 MB | 1.6 s |
| Gemma 4 E4B LUT6 | 0.9 s / 180 s | 546 tok/s | 19.6 tok/s | 620 MB | 2.3 s |

Qwen3.5 prefill is slower on prompts that do not fill whole batches: the remainder (39 of 231
tokens here, and all of a short chat turn) goes through `infer` at decode speed, since padding
would enter the recurrent state. The first load compiles the model for the Neural Engine
(cached by Core ML until another process loads the package with other settings).

**Long contexts.** Gemma 4's sliding-window layers keep a block-shifting cache of window + 2 x batch
= 640 positions once the context is longer than that (`--full-local-cache` restores
context-length caches); only the global layers' cache grows with the context. Keys keep their
slot within a block of `batch` positions, so `prefill` and `infer` give bit-identical logits
(oxinfer `--coreml-prefill auto` and `single`: identical last-prompt logits and replies), and
packages that differ only in context length too. Gemma 4 E2B LUT6, 1939-token prompt (a README
excerpt) + 100-200 teacher-forced tokens:

| Context | Local cache | Logit cosine / top-1 | Prefill | Decode (at the end of the prompt) |
|---|---|---|---|---|
| 2048, full-length local cache | 25 MB | 0.99980 / 98.0% | 1215 tok/s (1920 tokens) | 26.6 tok/s |
| 2048, block-shifting local cache | 8 MB | 0.99980 / 98.0% | 1530 tok/s | 33.4 tok/s |
| 8192, block-shifting local cache | 8 MB (vs 100 MB) | 0.99969 / 97.0% | 1080 tok/s (7936 tokens) | 22.6 tok/s |

**Memory on the Neural Engine.** Loading a chunk adds wired memory outside the process
footprint. Each function of the multifunction chunk gets its own copy of the weights even though
the package stores them once, and 3-bit LUTs take as much ANE memory as 4-bit ones. For four
layers of Qwen3.8-27B (1.52 B parameters): 0.38 / 0.57 / 0.76 GB on disk at LUT2 / 3 / 4, and
0.47 / 0.81 / 0.84 GB of ANE memory with only `infer` loaded (1.2 / 1.7-1.9 / 1.3-1.9 GB with
`prefill` too); `infer` takes 14.5 ms per four layers whatever the width. A whole Qwen3.8-27B
(24.3 B decoder parameters) would need about 13 GB of ANE memory at LUT3 or LUT4 with `infer`
alone (twice that with `prefill`), 7.5 GB at LUT2, plus ~0.7 GB for a LUT4 LM head: beyond what an
app can use on a 12 GB iPhone except at 2 bits, which costs far too much quality. On this Mac it
would decode at ~4 tok/s. Dense models up to ~8 B parameters (about 0.55 GB of ANE memory per
billion parameters at LUT4, `infer` only) are the practical range for 12 GB phones.

## How it works

- `safetensors.rs`: parses the header, exposes bf16 tensors as `&[u16]` (single file or sharded).
- `config.rs`: reads `config.json` and derives per-layer facts (sliding vs global, head dims, KV sharing source, k=v).
- `platform.rs`: CPU feature and GPU detection (sysctl / procfs for memory, Metal via `objc_msgSend`,
  CUDA driver API, Vulkan loader), all through `dlsym`; `gpu.rs`: the `Backend` trait the linear layers
  dispatch through and the registry; `metal.rs`, `cuda.rs`: the backends with their shader sources.
- `fsx.rs`, `clock.rs`: file access and monotonic time behind one interface, so the loaders run on a
  real filesystem or on the browser's in-memory table; `src/web/` (feature `web`): the browser module's
  C-ABI exports and the WebGPU backend calling back into the host; `web/` at the root: the
  JavaScript host and page.
- `whisper/`: speech to text: `mel.rs` (Whisper's log-mel features, mixed-radix FFT), `model.rs`
  (encoder, decoder, cross-attention cache), `tokens.rs` (special tokens, languages, generation
  config), `align.rs` (word timestamps: DTW), `compress.rs` (the fallback's compression ratio),
  `mod.rs` (`Transcriber`: decoding rules, long-form windows, fallback); `whisper_cli.rs`: the CLI.
- `sd/`: Stable Diffusion: `clip.rs` (BPE tokenizer, text encoder), `unet.rs`, `vae.rs`, `sched.rs`
  (Euler samplers), `ops.rs` (channels-last convolution, group/layer norm, multi-head attention on
  the f32 GEMM tiles); `png.rs`: a dependency-free PNG writer.
- `kernels.rs`: NEON kernels (`kernels_x86.rs`, `kernels_wasm.rs`: the AVX2 and simd128 equivalents, selected by `build.rs` through the `has_simd` cfg). Decode tiles stream 1–2 weight rows with 8 accumulators (the prefetcher
  prefers few long streams); prefill packs activations into 8-token panels and runs an 8-row x 8-token
  outer-product kernel that broadcasts bf16 weights by lane; `kernels_ggml.rs`: the GGUF formats'
  one-token and multi-token (SDOT / SMMLA) decode kernels and dequantization. Also RMSNorm, GELU (tanh approximation via
  a fast exp), RoPE tables, softmax.
- `quant.rs`: Q8 (32 weights per f32 scale) and Q4_0 (two nibbles per byte, f16 scale) block
  quantization of weights at load time, plus per-matmul activation quantization; decode kernels use
  int8 SDOT over four blocks at a time, with multi-token variants for speculative verification.
- `drafter.rs`: the Gemma 4 assistant model (drafter) and its centroid-masked LM head.
- `qwen_mtp.rs`: the Qwen3.5 MTP layer and the speculative decoder (drafting, verification with
  recorded recurrent states, speculative sampling).
- `vision.rs`, `audio.rs`: Gemma 4's two encoders plus preprocessing (resize/patching, mel
  features) and the embedders; `qwen_vision.rs`: the Qwen3.5-family image / video processors and
  vision tower; `resample.rs`: torchvision- and Pillow-exact bicubic resizing; `video.rs`: frame
  sampling and video decoding; `media.rs`: file decoding through oxideav; `av.rs`: decoded oxideav
  frames to RGB / mono PCM and back; `mm.rs`: attachments to soft tokens and prompt placeholders; `mm_config.rs`: Gemma's tower configs and placeholder
  token ids.
- `pool.rs`: a spin-waiting thread pool with an atomic work counter. Dispatch costs about a microsecond,
  which matters because a decode step issues ~200 small parallel regions.
- `model.rs`: the Gemma 4 decoder: scaled embeddings, per-layer embeddings (PLE), QK-norm, sliding /
  global attention with proportional RoPE on 512-dim global heads, KV-cache sharing for the top layers,
  gated-GELU MLP, post-norms, per-layer scalars, tied softcapped LM head.
- `chat.rs`, `sampler.rs`, `main.rs`: chat template, greedy / top-k / top-p sampling, streaming CLI.
- `coreml.rs`, `anemll.rs`: Core ML through the Objective-C runtime (models, functions, states,
  multi-arrays, IOSurface buffers) and the ANEMLL model runner (`meta.yaml`, prefill / infer
  protocol, sliding-window rotation, split or argmax LM head); `anemll_cli.rs`: its CLI driver.

## Verifying against the reference

```sh
python3 -m venv venv && ./venv/bin/pip install torch transformers safetensors numpy
printf '<bos><|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n' > prompt.txt
./target/release/oxinfer --model models/gemma-4-E2B-it --prompt "What is the capital of France?" --dump-logits ours.bin
./venv/bin/python tools/ref_logits.py models/gemma-4-E2B-it prompt.txt ours.bin --generate 12
```

For images and videos (Qwen3.5-family, and Gemma 4 videos; the reference needs `torchvision`),
dump oxinfer's side without a system prompt and let `tools/ref_vl.py` run the processor and model:

```sh
M=~/.cache/oxinfer/hub/Qwen--Qwen3.5-0.8B
./target/release/oxinfer --model $M --image fox.png -p "Describe this image." --no-system --print-prompt > prompt.txt
./target/release/oxinfer --model $M --image fox.png -p "Describe this image." --device cpu --dump-mm ours --dump-logits ours.bin
./venv/bin/python tools/ref_vl.py $M --image fox.png --prompt "Describe this image." \
    --ours ours --ours-logits ours.bin --ours-prompt prompt.txt --generate 40
```

Videos are compared as directories of frames (`ffmpeg -i clip.mp4 frames/%05d.png`, then
`--video frames --frames-fps 24` on both sides), so both read the same pixels.

Runtime interventions take the same JSON spec on both sides (`tools/ref_intervene.py` applies it
with PyTorch forward hooks; `--generate N --ours-ids IDS.json` also compares greedy text):

```sh
M=~/.cache/oxinfer/hub/Qwen--Qwen3.5-4B
./target/release/oxinfer --model $M --no-system -p "Write a short poem about the sea." --print-prompt > prompt.txt
./target/release/oxinfer --model $M --no-system -p "Write a short poem about the sea." --device cpu \
    --interventions spec.json --dump-logits ours.bin
./venv/bin/python tools/ref_intervene.py $M prompt.txt spec.json ours.bin
```
