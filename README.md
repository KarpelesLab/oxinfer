# oxinfer

[![CI](https://github.com/KarpelesLab/oxinfer/actions/workflows/ci.yml/badge.svg)](https://github.com/KarpelesLab/oxinfer/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/oxinfer.svg)](https://crates.io/crates/oxinfer)
[![docs.rs](https://img.shields.io/docsrs/oxinfer)](https://docs.rs/oxinfer)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**[Try it in your browser](https://karpeleslab.github.io/oxinfer/)**: Qwen 3.5 0.8B downloads from
Hugging Face and runs in the tab, with WebGPU when the browser has it; MLX 4-bit Gemma 4, Gemma 3
and Qwen3 models load from the mirror too. Early days: the crate is at 0.0.x and the API and file
formats may still change.

Local LLM inference in pure Rust: Gemma 4 (text, images, audio), Gemma 3 and Qwen 3 / 3.5 / 3.8
today, more model families as they come. No BLAS, no ML framework: weights are read straight out
of the safetensors files (HuggingFace checkpoints or MLX 4-bit ones) and every operation
(matmuls, attention, norms, RoPE, sampling) is implemented here, with hand-written SIMD kernels
(NEON, AVX2, wasm simd128) and its own thread pool. The CPU path runs everywhere; GPUs (Metal,
CUDA, WebGPU in the browser) are detected at run time and used for prefill when present, with
nothing linked at build time.

Output is validated against the `transformers` float32 reference: on Gemma 4 E2B-it the
last-token logits agree to a max absolute difference of about 1e-4 and greedy decoding produces
identical tokens.

## Supported models

| Model | Params (total) | bf16 size | Status |
|---|---|---|---|
| google/gemma-4-E2B-it | 5.1B | 10.2 GB | tested, matches reference (max logit diff 1e-4) |
| google/gemma-4-E4B-it | 8.0B | 16.0 GB | tested, matches reference (max logit diff 8e-5) |
| google/gemma-4-12B-it | 12B | 24 GB | supported (k=v global attention), untested |
| google/gemma-4-31B-it | 31B | 62 GB | supported, sharded loading, untested |
| google/gemma-4-26B-A4B-it | 26B MoE | 52 GB | not yet (MoE block unimplemented) |
| Qwen/Qwen3.8-27B (and Qwen3.5/3.6 dense) | 27.8B | 55.6 GB | text only; matches the bf16 reference; see below |
| Qwen/Qwen3.6-35B-A3B and fine-tunes (PixilabAI/Zawen-v0.1-35B-A3B) | 35B MoE, 3B active | 69 GB | text only; Zawen matches the reference (see below) |
| Qwen/Qwen3.5-0.8B | 0.8B | 1.7 GB | text only; the small one used for the wasm measurements |
| mlx-community/gemma-4-e2b-it-4bit, gemma-4-e4b-it-4bit | 5.1B / 8.0B | 3.6 / 5.2 GB (MLX 4-bit) | text, images and audio; see [MLX checkpoints](#mlx-checkpoints) |
| mlx-community/Qwen3-4B-4bit (Qwen3 dense) | 4.0B | 2.3 GB (MLX 4-bit) | text; matches the reference on the same weights (same top 10) |
| mlx-community/gemma-3-4b-it-4bit (Gemma 3) | 4.3B | 3.4 GB (MLX 4-bit) | text only; matches the reference, also past the 1024-token window |
| google/embeddinggemma-300m | 0.3B | 1.2 GB (fp32) | embeddings; matches sentence-transformers (see [Embeddings](#embeddings)) |
| Qwen/Qwen3-Embedding-0.6B | 0.6B | 1.2 GB | embeddings; matches sentence-transformers |
| BAAI/bge-m3 | 0.6B | 2.3 GB (fp32, `pytorch_model.bin`) | dense, sparse and ColBERT embeddings; matches the reference |
| sentence-transformers/all-MiniLM-L6-v2, paraphrase-multilingual-MiniLM-L12-v2 | 23M / 118M | 90 / 470 MB | embeddings (BERT); match sentence-transformers |
| BAAI/bge-small-en-v1.5, intfloat/multilingual-e5-small, mixedbread-ai/mxbai-embed-large-v1 | 33M-335M | 0.13-1.3 GB | embeddings (BERT / XLM-R); match sentence-transformers |
| stabilityai/sd-turbo | 1.3B | 2.6 GB (fp16) | text-to-image in one step; matches diffusers (see Image generation) |
| stable-diffusion-v1-5/stable-diffusion-v1-5 (and SD 1.x / 2.x fine-tunes) | 1.1B | 2.0 GB (fp16) | text-to-image; matches diffusers |
| anemll/* (ANEMLL Core ML: Qwen3, Qwen2.5, Gemma 3, Llama 3.2, ...) | 0.27B-1.7B tested | 0.3-1.5 GB (LUT6) | Apple Neural Engine via Core ML; same tokens as ANEMLL's runtime (see [Core ML](#core-ml--apple-neural-engine)) |

### Qwen3.5 / Qwen3.8

The Qwen3.5-family dense models are a different architecture (`qwen.rs`): 48 Gated DeltaNet layers
(a causal depthwise conv, L2-normalised keys/queries and a 128x128 delta-rule recurrent state per
value head) interleaved with 16 gated full-attention layers (24 query / 4 KV heads of 256, partial
RoPE on 64 dims, a sigmoid output gate), SiLU MLPs and zero-centred RMSNorm. The recurrent layers
keep a fixed-size state instead of a KV cache, so memory does not grow with context for 48 of the
64 layers. The vision encoder is not loaded yet; the MTP head drives speculative decoding with
`--mtp` (see Speculative decoding).

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
token). MLX-quantized experts are not read yet.

Thinking is on by default in the Qwen template; `--think` enables it here (the trace is dimmed),
otherwise the generation prompt closes the think block.

Text, images and audio are supported (E2B/E4B have both towers). Video is not.

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

## Image generation

A diffusers folder (one with `model_index.json`) switches oxinfer to text-to-image with Stable
Diffusion 1.x / 2.x, which includes SD 1.5, SD 2.1 and the one-step SD-Turbo (`src/sd/`):

```sh
./target/release/oxinfer --model models/sd-turbo -p "a red fox in a snowy forest" -o fox.png
./target/release/oxinfer --model models/sd-v1-5 -p "a watercolor lighthouse" --negative "blurry" \
    --steps 25 --guidance 7.5 --sampler euler_a --size 512x512 --seed 7 -o lighthouse.png
```

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

## Image and audio input

```sh
./target/release/oxinfer --model models/gemma-4-E4B-it --quant q4 \
    --image photo.jpg --audio clip.wav --prompt "Describe the image and transcribe the audio."
```

`--image` and `--audio` can be repeated; attachments are placed before the text of the user turn
(or of the first turn in `--chat`). Files are decoded with the pure-Rust
[oxideav](https://crates.io/crates/oxideav-core) crates (PNG, JPEG, WebP, GIF, BMP; WAV, MP3, FLAC,
AAC/M4A, Ogg Vorbis/Opus, MKV audio), resampled to 16 kHz mono for audio.

- **Vision**: aspect-preserving bicubic resize to at most 280 soft tokens (`--max-soft-tokens`
  accepts 70/140/280/560/1120), 16x16 patches, a 16-layer ViT (hidden 768) with axial 2D RoPE and
  clipped linears, 3x3 patch pooling, then the vision embedder. A 640x480 image becomes 57x42
  patches and 266 soft tokens in about 2.5 s.
- **Audio**: 128-bin log-mel features (20 ms frames, 10 ms hop), two stride-2 conv layers, a
  12-layer conformer with chunked local attention and relative position bias, then the audio
  embedder: 40 ms of audio per soft token (750 max, i.e. 30 s).

Soft tokens replace the `<|image|>` / `<|audio|>` placeholder embeddings inside the text model
(the per-layer-embedding lookup uses the pad token there, as in the reference). All attention
stays causal, matching the E2B/E4B configuration.

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
(Qwen3.5 / 3.6 / 3.8), `qwen35moe` and `qwen3`. The tokenizer and chat template come from the
GGUF metadata (the rebuilt tokenizer matches Qwen3.8's `tokenizer.json` on 305 test texts), and
llama.cpp's conversions are undone at load (`gguf.rs`): norms with the 1 folded in, `ssm_a`, the
tiled Gated DeltaNet value-head order, separate 3D expert tensors. Weights stay in their GGUF
format: F32, F16, BF16, Q4_0/Q4_1/Q5_0/Q5_1/Q8_0, Q2_K-Q6_K, IQ1_S/IQ1_M, IQ2_XXS/XS/S,
IQ3_XXS/S and IQ4_NL/XS (`ggml.rs`, dequantization bit-exact with gguf-py), with NEON decode
kernels for every type and Metal prefill kernels. The i-quants (IQ1-IQ3) decode several times
slower than Q4_K on the CPU (table lookups); Q4_K, Q5_K, Q6_K, IQ4_XS and Q8_0 are close to
oxinfer's own Q4/Q8. Embedded MTP layers (`blk.N.nextn.*`) work with `--mtp`.

On HauhauCS's Qwen3.8-27B Q4_K_P the next-token distribution matches llama.cpp on the same file
(same top 10, log-probabilities within 0.03, mean 0.013). F32/BF16 GGUFs from llama.cpp's
converter match transformers to 5e-5. `--save-quantized` does not take GGUF input.

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
(`CancelHandle`), tool calls, attachments and per-turn statistics:

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
(see `ios/README.md`).

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

For the 27B model `--mmap` is recommended: quantization then reads the checkpoint straight from
the page cache (about 3 s to quantize 27.8B parameters on 18 threads) instead of copying 55 GB
into the process first.

With the Metal backend (default on Apple Silicon) prefill runs on the GPU:

| Model | Precision | Prefill, CPU only | Prefill, Metal |
|---|---|---|---|
| E4B-it | `--quant q4` | ~97 tok/s | ~580 tok/s |
| Qwen3.8-27B | `--quant q4` | ~18 tok/s | ~96 tok/s (256-token chunks) |

(616-token prompt; decode speed is unchanged, see below for why.)

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
GPU) and oxinfer's default (prefill GEMMs on the GPU, everything else on the CPU); "CPU" is
`--device none -ngl 0` / `--device cpu` with each engine's default thread count (llama.cpp 6,
oxinfer 18).

Qwen3.8-27B (HauhauCS Aggressive, `Q4_K_P`, 17.9 GB; speculative rows use the embedded MTP
head, or the 32k-vocabulary FastMTP sidecar with `--mtp-from`):

| Engine | pp512 tok/s | pp2048 tok/s | Decode, code | Decode, prose | Drafts accepted (code / prose) |
|---|---|---|---|---|---|
| llama.cpp Metal | 650 [620-703] | 577 [538-720] | 24.2 [23.3-26.2] | 24.9 [24.1-25.2] | |
| llama.cpp Metal, MTP depth 2 | | | 29.5 [22.5-30.4] | 22.0 [19.9-22.7] | 84% / 61% |
| llama.cpp Metal, MTP depth 3 | | | 34.2 [26.0-35.3] | 24.0 [19.2-28.5] | 76% / 49% |
| llama.cpp Metal, FastMTP depth 3 | | | 37.7 [32.1-39.0] | 25.3 [22.5-26.9] | 78% / 42% |
| llama.cpp CPU | 20.8 [18.9-21.2] | 19.0 [18.9-20.4] | 6.7 [6.2-6.9] | 6.8 [6.3-6.9] | |
| oxinfer (Metal prefill) | 144 [144-147] | 125 [124-129] | 11.0 [10.8-11.5] | 11.3 [11.1-11.7] | |
| oxinfer `--mtp` | | | 14.8 [12.7-16.0] | 11.8 [9.8-12.1] | 78% / 49% |
| oxinfer `--mtp-from` FastMTP | | | 14.2 [14.1-17.8] | 11.5 [9.9-13.1] | 78% / 46% |
| oxinfer CPU | 24.2 [21.4-27.8] | 23.3 [20.8-27.2] | 10.6 [10.3-11.9] | 11.5 [10.8-11.8] | |

(`llama-bench` on the same file in a quiet moment: pp512 744, pp2048 642, tg256 25.5 tok/s.)

Zawen-v0.1-35B-A3B (Qwen3.5-MoE, 256 experts, converted with llama.cpp's `convert_hf_to_gguf.py`
+ `llama-quantize`; the checkpoint has no MTP weights, but the converter still writes
`nextn_predict_layers = 1` from `config.json`, which llama.cpp then refuses to load, so
`gguf_set_metadata.py` set it to 0 and `block_count` to 40; oxinfer loads either file):

| Engine | Q4_K_M pp512 | pp2048 | Decode code / prose | Q8_0 pp512 | pp2048 | Decode code / prose |
|---|---|---|---|---|---|---|
| llama.cpp Metal | 3001 [2998-3007] | 3065 [2381-3091] | 110 / 111 | 2700 [2603-2798] | 2476 [1732-2730] | 80.7 / 81.0 |
| llama.cpp CPU | 124 [122-125] | 117 [116-119] | 47.5 / 46.5 | 156 [150-164] | 153 [141-157] | 43.2 / 43.8 |
| oxinfer (Metal prefill) | 215 [212-230] | 231 [175-244] | 53.9 / 54.1 | 195 [170-198] | 219 [190-219] | 43.0 / 43.3 |
| oxinfer CPU | 157 [154-161] | 146 [143-151] | 54.8 / 55.2 | 122 [112-146] | 111 [106-136] | 46.4 / 43.2 |

Qwen3.5-0.8B (`Q8_0` converted with `convert_hf_to_gguf.py`, keeps the MTP layer):

| Engine | pp512 | pp2048 | Decode code / prose | MTP decode code / prose (accepted) |
|---|---|---|---|---|
| llama.cpp Metal | 14930 [14924-15158] | 16034 [15939-16088] | 280 / 281 | depth 2: 332 / 276 (67% / 47%); depth 3: 335 / 250 (59% / 35%) |
| llama.cpp CPU | 809 [797-811] | 748 [739-768] | 161 / 165 | |
| oxinfer (Metal prefill) | 1280 [1219-1313] | 1412 [1412-1432] | 174 / 174 | `--mtp`: 185 / 150 (63% / 43%) |
| oxinfer CPU | 826 [813-826] | 747 [739-779] | 173 / 172 | |

With 12 threads for both engines on the CPU (a separate interleaved run, load ~15), llama.cpp
pulls ahead on prompts: 27B pp512 27.4 vs 18.4 tok/s, decode 9.4 / 8.4 vs 8.0 / 9.7; Zawen Q4_K_M
pp512 197 vs 129, decode 60.5 / 60.1 vs 58.9 / 56.9.

Where the differences come from:

- **Decode with a GPU**: llama.cpp runs the whole forward pass on the GPU (~450 GB/s), oxinfer
  decodes on the CPU (`--profile` on the 27B: matmuls stream 16.9 GB/token at ~200 GB/s, 85 of
  95 ms per token; the 1 GB Q6_K LM head another 4.5 ms). That is the 2.2x on the 27B and 2x on
  the MoE. On the CPU alone oxinfer decodes as fast or faster (27B 11 vs 6.7 tok/s with default
  threads, 0.8B 173 vs 161).
- **Prefill with a GPU** (4.5x on the 27B, 14x on the MoE, 11x on 0.8B): only the dense GEMMs
  run on the GPU, as f32 simdgroup tiles over dequantized weights (~8.4 TFLOP/s on the 27B: 13.2
  s of GPU time for 2048 tokens). llama.cpp sustains ~35 TFLOP/s there (650 tok/s), above the
  f32 simdgroup peak of ~20, with half-precision kernels (its Metal backend reports the M5 tensor
  API). Attention and the Gated DeltaNet scan stay on the CPU (2.6 ms/token, 25% of oxinfer's
  27B prefill), and each layer's activations make a round trip through host memory. On the MoE
  the routed experts' grouped matmul (`linear_grouped`) runs on the CPU: only 1.35 s of a 9.2 s
  2048-token prefill is GPU time.
- **CPU prefill**: llama.cpp repacks K-quants at load (`CPU_REPACK`, interleaved rows for the
  i8mm `SMMLA` instructions) and multiplies in int8; oxinfer dequantizes to f32 and runs an
  FMA GEMM. It is ahead on K-quants only while llama.cpp uses its default 6 threads, and behind
  on Q8_0 even then.
- **MoE decode** is not bandwidth-bound in oxinfer (Zawen Q4_K_M: 521 matmul calls per token at
  ~21 us, 170 GB/s over only 1.9 GB): dispatch and per-expert overhead dominate.
- **Speculative decoding**: acceptance is the same in both engines (78% code, 46-49% prose on
  the 27B), so the gap is the verification pass: llama.cpp's multi-token pass on the GPU costs
  little more than one token, while oxinfer's 4-token CPU pass takes 160-180 ms against ~90 ms
  for one token (the K-quant kernels are compute-bound at 4 tokens). MTP thus gains 1.35x on code
  here vs 1.4x (embedded) / 1.55x (FastMTP) in llama.cpp; on prose neither engine gains (0.96-1.04x).
  On 0.8B every draft step reads the 248k-row LM head, so prose loses in both engines.
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
text model reads are written (no vision/audio towers, no Qwen MTP head); `__metadata__` records the
precision. Shards are at most 1 GiB (one allocation must stay under 2 GiB in wasm32), and large
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

`--stream` (Qwen3.5 / Qwen3.8; implies `--mmap`) runs a model whose weights do not fit in memory
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
| Apple Silicon (macOS, iOS) | NEON kernels + Metal GEMM for prefill; on M5 / A19 the GPU matrix units (prefill GEMMs and MoE experts) | measured; the f32 GEMM gives bit-identical logits to the CPU path, the matrix units (default where present) ~3e-4 relative per product (`OXINFER_METAL_MMA=0` for bit-identical); iOS: runs in the Simulator (CPU and Metal; `ios/`) |
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

Decode stays on the CPU on purpose: it is bandwidth-bound, the CPU already streams weights at
~280 GB/s, and the GPU's ~450 GB/s only pays off if the entire forward pass moves over
(attention, norms, the KV cache). Measured on the 27B GGUF with every pass on the GPU
(`OXINFER_METAL_MIN_T=1`, matrix units): a pass costs ~257 submissions with CPU work in between
(~50 ms of submit-and-wait alone) plus 110-150 ms of dequantization-bound GEMMs for 2-16 tokens,
so plain decode drops from 10.7 to 5.5 tok/s, and `--mtp` verification passes (~4 tokens) take
the same ~208 ms as on the CPU (14.1 vs 14.2 tok/s). The GPU only wins clearly for 8-16-token
passes, where the CPU leaves its batch-invariant decode kernels (~0.6-1.4 s vs ~0.2-0.3 s).

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
sampler does. Diffusion weights stay in JavaScript memory and the module converts
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

Feature flags: `media` (oxideav image/audio decoding), `hf` (HuggingFace fetching), `gpu`
(dlsym-loaded backends), all on by default, and `web` (the browser module, off by default and only
meaningful for `wasm32-unknown-unknown`); `--no-default-features` gives the plain CPU build.
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
unchanged under ANEMLL's own `chat.py`; Qwen3.5 / 3.6 / 3.8 dense models (Gated DeltaNet
recurrent state kept as Core ML states, chunked delta rule in the prefill function) and Gemma 4
E2B / E4B (per-layer embeddings, KV sharing, local and global caches) extend the format as
described in [docs/coreml-format.md](docs/coreml-format.md); oxinfer's Core ML runtime above
runs all of them. Weights are 6-bit LUT-palettized by default (ANEMLL's recommended setting);
`tools/coreml_check.py` runs a package with coremltools, compares it with transformers in float32
over a few hundred teacher-forced tokens, benchmarks it and lists which ops land on the ANE.

```sh
python3 -m venv venv && venv/bin/pip install coremltools==9.0 torch transformers safetensors \
    scikit-learn pyyaml "numpy<2.3"
venv/bin/python tools/coreml_convert.py --model ~/.cache/oxinfer/hub/Qwen--Qwen3.5-0.8B --output out/qwen35
venv/bin/python tools/coreml_check.py out/qwen35 --hf ~/.cache/oxinfer/hub/Qwen--Qwen3.5-0.8B --bench --plan
```

Measured on an M5 Max with coremltools (context 512, batch 64; quality = the Core ML logits
against transformers float32 over 300 teacher-forced tokens of a 135-token chat prompt; speed =
best of 5, 256-token prefill and 64-token decode including the Python calls; package sizes are
embeddings + decoder chunks + LM head):

| Model | Weights | Size (MB) | Logit cosine | Top-1 agreement | Prefill ANE / CPU | Decode ANE / CPU |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | fp16 | 311 + 884 + 311 | 0.99999 | 98.7% | 2680 tok/s | 79 tok/s |
| Qwen3-0.6B | LUT6 (default) | 311 + 338 + 119 | 0.9972 | 93.0% | 5131 / 1949 tok/s | 114 / 56 tok/s |
| Qwen3-0.6B, ANEMLL's release | LUT6 | 311 + 339 + 119 | 0.9971 | 92.3% | 4409 / 1824 tok/s | 114 / 59 tok/s |
| Qwen3-0.6B | LUT4 | 311 + 224 + 119 | 0.954 | 72.0% | 4836 tok/s | 103 tok/s |
| Qwen3.5-0.8B | fp16 | 509 + 1000 + 509 | 0.9998 | 97.0% | 4002 tok/s | 62 tok/s |
| Qwen3.5-0.8B | LUT6 (default) | 509 + 385 + 195 | 0.9974 | 93.0% | 5176 / 1608 tok/s | 118 / 37 tok/s |
| Qwen3.5-0.8B | LUT4 | 509 + 254 + 195 | 0.959 | 78.3% | 5035 tok/s | 113 tok/s |
| Gemma 4 E2B | fp16 | 5530 + 3740 + 805 | 1.00000 | 99.3% | | |
| Gemma 4 E2B | LUT6 (default) | 5530 + 1419 + 306 | 0.99986 | 97.0% | 1905 / 623 tok/s | 45 / 14 tok/s |
| Gemma 4 E4B | LUT6 (default) | 7034 + 2992 + 508 | 0.99945 | 96.7% | 1060 / 333 tok/s | 25 / 6 tok/s |

Everything but the embedding lookups and the RoPE-table gathers runs on the ANE (MLComputePlan),
including the Gated DeltaNet recurrence and its chunked prefill. The Gemma embeddings model is
large because the per-layer embedding table stays fp16: Core ML memory-maps it and pages in only
the rows it reads, while a palettized table would be decompressed at load time (16.7 GB for E2B).
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
- `vision.rs`, `audio.rs`: the two encoders plus preprocessing (resize/patching, mel features) and
  the embedders; `media.rs`: file decoding through oxideav; `mm_config.rs`: tower configs and
  placeholder token ids.
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
