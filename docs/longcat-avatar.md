# Avatar video: LongCat-Video-Avatar 1.6

`meituan-longcat/LongCat-Video-Avatar-1.6` animates a character from a speech recording: a
reference image (or only a prompt), the audio and a text prompt become a 25 fps video of the
character speaking, 93 frames (3.7 s) per clip, with further clips chained into long videos and
an optional second speaker. oxinfer runs it from the released files: `src/longcat_avatar/`
follows Meituan's reference (github.com/meituan-longcat/LongCat-Video,
`pipeline_longcat_video_avatar.py` and the two `run_demo_avatar_*` scripts with
`--model_type avatar-v1.6`).

```sh
# the int8 transformer with the two LoRAs, 8 steps: a character speaking the recording
./target/release/oxinfer --preset longcat-avatar-int8 --device cuda \
    --init-image person.png --audio speech.wav \
    -p "A woman is speaking to the camera in a bright studio." -o talk.mp4
# the bf16 transformer; three clips chained (93 + 80 + 80 frames, 10.1 s)
./target/release/oxinfer --preset longcat-avatar --device cuda \
    --init-image person.png --audio speech.wav -p "..." --segments 3 -o talk.mp4
# no reference image: the character comes from the prompt
./target/release/oxinfer --preset longcat-avatar-int8 --audio speech.wav -p "..." --size 832x480 -o talk.mp4
# two speakers, each with a recording and a box (y0,x0,y1,x1 in the image's pixels)
./target/release/oxinfer --preset longcat-avatar-int8 --init-image two.png \
    --audio left.wav --audio right.wav --bbox 100,80,800,640 --bbox 50,720,820,1300 -p "..." -o talk.mp4
```

`--model meituan-longcat/LongCat-Video-Avatar-1.6` (with `--int8` for the int8 transformer) is
the same as the presets; a folder holding the release works too. The files are fetched into the
cache on first use: of the release the chosen transformer (`base_model_int8/` 15.9 GB or
`base_model/` 31.7 GB), `loraA/` and `loraB/` (2.5 GB each), `scheduler/` and
`whisper-large-v3/model.safetensors` (3.1 GB: not the 15 GB of Whisper's other weight formats in
the repository, nor the vocal separator); of the base repository `meituan-longcat/LongCat-Video`
the `text_encoder/` (22.7 GB), `tokenizer/tokenizer.json` and `vae/` (0.5 GB), not its own 54 GB
transformer. 47 GB in all with the int8 transformer, 63 GB with the bf16 one. With a release
folder, the base repository is looked for next to it as `LongCat-Video/` (the reference's
layout), or named with `--avatar-base`.

| flag | |
|---|---|
| `--audio FILE` | the driving speech (anything oxideav decodes: `.wav`, `.mp3`, `.flac`, `.m4a`, ...; resampled to 16 kHz mono). Twice: two speakers |
| `--init-image FILE` | the character. The canvas is the bucket closest to its aspect ratio (the reference's table: 832x480, 768x512, 640x608, 480x832, ... around 400,000 pixels), the image resized to cover it and center-cropped. Without one: `--size WxH` (default 832x480; multiples of 16) and a character from the prompt |
| `-p TEXT` | the scene and what the character does (English or Chinese); umT5, 512 tokens |
| `--int8` | `base_model_int8/` instead of `base_model/` (the `longcat-avatar-int8` preset sets it) |
| `--frames N` | frames per clip, `4 n + 1` (default 93, what the model was trained for) |
| `--segments N` | clips. Each one after the first is conditioned on the last 13 frames of the one before and on the first clip's first frame, and adds `frames - 13`: 93, 173, 253, ... frames |
| `--hd` | the 720p buckets (1248x736, 960x960, ... around 920,000 pixels) |
| `--mix-steps N`, `--mix-alpha F` | the first N of the 8 steps run `F` x the DMD LoRA + `(1 - F)` x the OPSD LoRA, the others the OPSD LoRA alone (default 4 and 1, the model card's setting) |
| `--no-distill` | the base model without the LoRAs: `--steps` (default 50), `--guidance` (text, default 4), `--audio-guidance` (default 4), `--negative`; three forwards per step |
| `--bbox Y0,X0,Y1,X1` | twice, with two `--audio`: each speaker's region in the reference image (default: the left and right halves, a tenth in from their edges) |
| `--audio-mode para\|add` | two speakers: both recordings from the start, cut to the shorter (`para`, default), or one after the other (`add`) |
| `--ref-index N`, `--mask-frames N` | continuation: the temporal position the reference frame is placed at (default 10) and how many frames around it do not see it (default 0) |
| `--seed N` | noise seed (torch's CPU generator; the reference draws on the GPU, so its seeds give other videos) |
| `-o FILE` | `.mp4` (default `avatar.mp4`), `.webm`, `.avi`, `.webp` or a PNG sequence, as for [MiniMax-H3](video.md), at 25 fps with the driving audio as the soundtrack (48 kHz, both speakers mixed, cut to the video's length) |

## Presets

| preset | transformer | LoRAs | settings |
|---|---|---|---|
| `longcat-avatar-int8` | `base_model_int8/` (15.9 GB) | `loraA` (DMD), `loraB` (OPSD) | 8 steps, no guidance, `mix_steps` 4, `mix_alpha` 1 |
| `longcat-avatar` | `base_model/` (31.7 GB, bf16) | the same | the same |

Both are the model card's "Quick Inference" setting (`--use_distill --mix_steps=4
--mix_alpha=1.0`, with `--use_int8` for the first). They are listed by `oxinfer --presets` and
`oxinfer::advisor::presets_json()` / `oxi_presets_json()` with the image presets
([MOSS-Image](moss-image.md#presets)), under the pipeline name `longcat-avatar`, and taken by
`--preset ID` or `--model ID`. The server does not run this model (see
[below](#not-supported)): it answers a request for one of these presets with an error.

## The release

| component | folder | size | |
|---|---|---|---|
| transformer | `base_model/` | 31.7 GB, 15.8B parameters, bf16 | 48 blocks, width 4096, 32 heads of 128, SwiGLU 11008; per block: self-attention, text cross-attention, audio cross-attention, feed-forward, and two AdaLN projections from a 512-wide timestep embedding |
| transformer, int8 | `base_model_int8/` | 15.9 GB | every `nn.Linear` but the output layer as int8 rows with one float32 scale per row (`weight / (max \|row\| / 127)`, rounded); biases bf16, norms and the patch embedding float32 |
| LoRAs | `loraA/`, `loraB/` | 2.5 GB each, float32 | rank 128 on seven matrices per block (`attn.qkv`, `attn.proj`, `cross_attn.q_linear`, `cross_attn.kv_linear`, `ffn.w1`, `ffn.w2`, `ffn.w3`), a separate up-projection per section of the fused ones, `alpha / r = 0.5`. A: DMD, B: OPSD |
| audio encoder | `whisper-large-v3/` | 3.1 GB, float16 | the encoder only (32 layers, width 1280) |
| text encoder | `LongCat-Video/text_encoder/` | 22.7 GB, float32 | umT5-XXL (24 blocks, width 4096) |
| VAE | `LongCat-Video/vae/` | 0.5 GB, float32 | Wan 2.1's: 16 latent channels, 8x in space, 4x in time |
| scheduler | `scheduler/` | | rectified flow, shift 7 |
| vocal separator | `vocal_separator/Kim_Vocal_2.onnx` | 67 MB | not used: see [not supported](#not-supported) |

What a request does (`pipeline.rs`):

- the prompt through `prompt_clean` (HTML entities, whitespace, and ftfy's width, quote and
  ligature fixes: `，` becomes `,`) and umT5; only the prompt's own positions are encoded, since
  the transformer zeroes the padded ones after its caption embedder (they still take part in the
  cross-attention, as the keys and values the biases give them);
- the speech cut or zero-padded to the video's length plus one second, in 30 s pieces through
  Whisper's log-mel front end and encoder; the 33 hidden states of each 20 ms position averaged
  in four groups of eight plus the last one, resampled linearly to 25 rows per second; a clip
  reads, for each of its frames, the rows two before to two after;
- the reference image through the VAE encoder as the first latent frame, kept as it is while the
  other 23 are denoised (it only attends to itself, at timestep 0);
- 8 steps down the distilled schedule (sigmas 1, 0.980, 0.954, 0.921, 0.875, 0.807, 0.699,
  0.498 after the shift), one forward each, the LoRA multipliers set per step;
- a continuation: the last 13 frames of the previous clip (as written, 8 bits) encoded again as
  4 conditioning latent frames, with the first clip's first latent frame in front as a reference
  at temporal position 10; the conditioning frames attend to each other, the reference to
  itself, the 20 new latent frames to everything;
- two speakers: the self-attention of each block also measures how much every token looks at
  each speaker's region of the reference frame, which gives the token a position between the
  two speakers' audio tokens in the audio cross-attention (a 1-D RoPE);
- the VAE decoder, one latent frame at a time with the causal convolutions' two-frame caches,
  as diffusers runs it; `floor(255 x)` to 8 bits as the demo does.

## What was checked

Against Meituan's code, with the same inputs and the same noise on both sides
(`examples/longcat_avatar_check.rs` reads what the two tools below write). The reference's
flash-attn calls go to torch's `scaled_dot_product_attention` in both tools (same softmax scale;
flash-attn has no aarch64 wheel), and neither runs the vocal separator.

**Tiny random checkpoints** (`tools/longcat_avatar_tiny.py`: the reference's own modules with 3
blocks of 128 wide, the reference's int8 quantizer, LoRAs in the released layout, a 32-layer
Whisper, umT5 with the real vocabulary, the Wan VAE layout; float32 on the CPU, torch 2.15,
diffusers 0.43, transformers 5.24 here and torch 2.14.1+cu130, diffusers 0.35.1, transformers
4.41.0 on the DGX Spark), maximum absolute difference (mean \|value\| of the reference 0.1 to 0.8):

| stage | CPU | CUDA, split operands |
|---|---|---|
| scheduler sigmas | 6e-8 | |
| VAE: encode 13 frames, encode one image, decode 4 latent frames | 1.7e-6, 7.9e-7, 9.2e-6 | 1.3e-6, 5.8e-7, 9.6e-6 (GB10) |
| transformer, one forward: text only, image, continuation, continuation with masked reference, two speakers | 8.3e-7, 8.3e-7, 6.6e-7, 9.5e-7, 8.3e-7 | 7.2e-7, 6.6e-7, 7.7e-7, 9.5e-7 (RTX 5080; two speakers run on the CPU) |
| the same on the int8 weights (image) | 7.2e-7 | 7.2e-7 |
| LoRAs: 0.3 A + 0.7 B, B alone (bf16 base, then int8) | 7.2e-7, 8.3e-7, 7.2e-7, 7.2e-7 | 2.2e-5, 2.6e-5, 2.2e-5, 2.7e-5 |
| audio features (2.6 s) | 1.9e-6 | |
| umT5: token ids of two prompts (one with Chinese, fullwidth, halfwidth-katakana and curly characters), hidden states | equal, 5.9e-6 | |
| image + audio + text to 29 frames of 832x480, 8 steps with the LoRA mix: latents, frames | 1.3e-6, at most one 8-bit step (1 value in 10,000) | |
| the same on the int8 weights | 1.3e-6, one 8-bit step | |
| continuation of that clip: latents, frames | 1.0e-3 (it starts from 8-bit frames that differ by a step), one 8-bit step | |
| guidance (3 steps, text 4, audio 4, a negative prompt): latents | 6.1e-6 | |

The CUDA column was measured with split operands (`OXINFER_AVATAR_GPU_EXACT=1`: float32
activations as two halves); the LoRA factors are float16 either way (after a power of two), which
is where the 2e-5 comes from. With the default float16 operands the same cases are within 2e-2
of the mean magnitude (`cargo test --lib longcat_avatar::cuda`, which runs the tiny release's
blocks on the device against the CPU with split operands and on the default path, with its
outputs fused into the GEMMs and not; skipped without an NVIDIA GPU). The VAE's device passes
([below](#the-vae)) are checked the same way: 9 frames encoded and 3 latent frames decoded on
the tiny VAE within 1e-4 of the host's (`cargo test --lib longcat_avatar::vae`); the CUDA
column's VAE figures are those of the per-layer path they replaced.

**The released weights**, against the reference on a DGX Spark (GB10, torch 2.14.1+cu130,
`tools/longcat_avatar_ref.py`: the modules in bfloat16 as the demo loads them, the repository's
`assets/avatar/single` example: `man.png`, the first seconds of `man.mp3`, its prompt):

| stage | against the reference as the demo runs it (bfloat16) | against the reference in float32 |
|---|---|---|
| umT5 hidden states of the example's prompt (51 tokens; mean \|v\| 0.052) | max 2.2e-1, mean 1.3e-3 (PSNR 58.2 dB) | max 7.5e-6, mean 1.7e-7 (137.1 dB) |
| token ids of the prompt | equal | |
| audio features of 4.7 s (118 rows of 5 x 1280; mean \|v\| 0.57; the reference runs Whisper in float32) | | max 1.3e-1, mean 1.1e-4 (90.7 dB) |
| VAE: the reference image (768x512) encoded (mean \|v\| 0.85) | max 4.2e-2, mean 4.4e-3 (56.8 dB) | max 6.3e-6, mean 7.9e-7 (131.8 dB; the reference on the CPU) |
| VAE: 3 latent frames decoded to 9 frames (mean \|v\| 0.58) | max 8.7e-2, mean 1.7e-3 (53.2 dB) | max 2.2e-5, mean 2.0e-7 (129.9 dB) on the CPU, max 2.6e-5, mean 4.0e-7 (125.5 dB) on the GB10 |
| transformer, int8 weights, one forward of 24 latent frames (36,864 tokens) at sigma 0.9, no LoRA (mean \|v\| 0.80) | max 9.6e-2, mean 1.5e-2 (48.9 dB) | max 1.6e-2, mean 2.4e-4 (84.0 dB); split operands: mean 1.9e-4 (85.4 dB) |
| the same with the DMD LoRA (`loraA`) | max 1.1, mean 3.0e-2 (43.2 dB) | max 5.0e-2, mean 7.8e-4 (73.9 dB); split: mean 4.8e-4 (78.3 dB) |
| the same with the OPSD LoRA (`loraB`), sigma 0.5 | max 9.4e-1, mean 1.8e-2 (49.2 dB) | max 9.2e-3, mean 2.8e-4 (85.1 dB); split: mean 2.1e-4 (86.8 dB) |
| transformer, bf16 weights (`base_model/`), the same forward, no LoRA (GB10) | max 8.7e-2, mean 1.5e-2 (49.0 dB) | max 1.1e-2, mean 2.4e-4 (83.9 dB) |
| the same with the DMD LoRA (GB10) | max 3.8e-1, mean 2.8e-2 (44.3 dB) | max 1.1e-1, mean 7.9e-4 (73.2 dB) |
| the same with the OPSD LoRA, sigma 0.5 (GB10) | max 6.9e-1, mean 1.8e-2 (49.0 dB) | max 2.9e-2, mean 2.8e-4 (84.9 dB) |
| the whole clip, int8 weights: image, 3.7 s of audio and the prompt to 93 frames, 8 steps with the LoRA mix, from the reference's noise (GB10): latents (mean \|v\| 0.73) | max 3.9e-1, mean 1.7e-2 (42.2 dB) | max 1.8e-2, mean 4.2e-4 (73.6 dB) |
| its 93 frames of 768x512, 8 bits | max 85 steps of 255, mean 0.98 (42.6 dB) | max 6 steps, mean 0.032 (63.1 dB) |
| the whole clip, bf16 weights (GB10): latents, frames | max 1.2, mean 1.8e-2 (41.3 dB); max 93 steps, mean 1.08 (41.7 dB) | not computed |
| the continuation (93 more frames) from the reference's first clip, its latents and its 8-bit frames (GB10): latents (mean \|v\| 0.74), frames; int8 weights | max 8.9e-1, mean 1.1e-2 (44.7 dB); max 165 steps, mean 0.76 (42.3 dB) | not computed |
| the same, bf16 weights | max 7.6e-1, mean 1.1e-2 (44.5 dB); max 163 steps, mean 0.73 (41.7 dB) | not computed |
| the continuation from oxinfer's own first clip, against the reference's from its own; int8, bf16 weights | latents max 1.7, mean 2.9e-2 (37.3 dB), frames 37.4 dB; latents max 1.5, mean 3.6e-2 (34.7 dB), frames 36.2 dB | not computed |

The transformer rows without "GB10" are oxinfer on the RTX 5080 (float16 operands unless
"split"); the int8 forwards were run again on the GB10 with the same figures to the digits
shown. All are against the reference's outputs for the same latents, text rows, audio rows and
timesteps, the clips from the same noise. The float32 column is the reference's own modules with
float32 activations: the text encoder loaded in float32, the transformer's float tensors and the
LoRAs read again from the files after `.float()`, the VAE on the CPU (no cuDNN) with float32
activations on the weights its bfloat16 load had rounded. oxinfer rounds the VAE's weights to
bfloat16 too (kernels, biases and norm gains) and keeps every other weight as stored.

The reference's bfloat16 arithmetic is what the first column measures: its own bfloat16 forward
is 1.5e-2 (mean) away from its float32 one, sixty times what separates oxinfer from the latter,
and its own bfloat16 clip is max 3.9e-1, mean 1.7e-2 (42.2 dB) away from its float32 clip on the
latents and 42.5 dB on the 8-bit frames, the same distance as oxinfer's clip from it. The
float32 clip took the reference 1780.7 s on the GB10 (431.6 s in bfloat16; 51 s a step, 43 s for
a forward without LoRAs), which is why only the first int8 clip has one; the continuation and
the bf16 clips are compared with the bfloat16 reference alone, where they sit at the same 41 to
45 dB. The continuation from oxinfer's own first clip is further off because the two sides then
start from different frames.

**After the speed work** ([Speed and memory](#speed-and-memory): the default path's kernels and
the VAE's device passes) the figures above were not measured again: the reference's outputs
are not on the machines it was done on. Each stage was compared with itself before instead,
on the same inputs:

| stage | against the same stage before |
|---|---|
| transformer, int8 weights, one forward of 36,864 tokens with the OPSD LoRA on made-up inputs (RTX 5080; mean \|v\| 0.71) | max 1.8e-2, mean 6.8e-5 (93.2 dB) |
| the same against split operands (`OXINFER_AVATAR_GPU_EXACT=1`, whose kernels did not change) | mean 1.13e-4 (88.9 dB); before: 1.10e-4 (90.3 dB) |
| transformer, bf16 weights, the same forward (GB10; mean \|v\| 0.71) | max 1.1e-2, mean 6.8e-5 (93.7 dB); without a LoRA mean 7.8e-5 (91.3 dB) |
| VAE: 5 latent frames decoded to 17 frames of 768x512, made-up latents (both machines) | max 7.6e-6, mean 3.9e-7 (125.9 dB); with float32 products max 4.2e-6, mean 2.3e-7 (130.6 dB) |
| VAE: 13 of those frames encoded | max 1.2e-5, mean 1.4e-6 (120.2 dB); with float32 products 125.0 dB |
| the whole clip, bf16 weights, same seed (GB10): 93 frames of 8 bits | max 5 steps of 255, mean 0.022 (64.6 dB) |
| a clip of 13 frames, int8 weights (GB10) | max 2 steps, mean 0.015 (66.5 dB) |

The clip moved by what a different rounding moves it: 64.6 dB, where 63.1 dB separated it
from the float32 reference. The kernels' changes that are not bit-identical are in the
attention (the scores accumulated by the tensor cores, the device's approximate `2^x` for the
probabilities); the larger GEMM tiles, the fused outputs and the fused norms gave the same bits
as the kernels before them on both machines.

## Speed and memory

Measured on the repository's `single` example at 768x512 (the bucket of `man.png`), clips of 93
frames (24 latent frames, 36,864 tokens), 8 steps, on a Threadripper 9970X with an RTX 5080
(16 GB), from a warm page cache, a `--profile ci` build (the release profile without LTO):

| run | a step | denoising | VAE decode | whole run | peak resident memory | GPU memory in use |
|---|---|---|---|---|---|---|
| `longcat-avatar-int8`, two clips (173 frames, 6.9 s) | 25.9 s (25.3 s in the continuation) | 413.5 s | 23.2 s | 453.9 s | 26.4 GB | 15.3 GB |
| `longcat-avatar` (bf16), one clip | 27.4 s | 225.4 s | 11.4 s | 251.5 s | 41.6 GB | 15.3 GB |

The same example on a DGX Spark (GB10, 20 cores, 121 GB shared by the CPU and the GPU; all 48
blocks stay on the device, 17.4 GB with their LoRA factors for the int8 weights and 32.2 GB for
the bf16 ones), a release build, the files named as folders (`--model DIR --avatar-base DIR`):

| run | a step | denoising | VAE decode | whole run | peak resident memory | GPU memory of the process | memory taken from the machine |
|---|---|---|---|---|---|---|---|
| `longcat-avatar-int8`, two clips (173 frames, 6.9 s) | 42.9 s (42.0 s in the continuation) | 685.4 s | 60.1 s | 757.7 s | 26.2 GB | 29.2 GB | 38.7 GB |
| `longcat-avatar` (bf16), one clip | 46.5 s | 375.8 s | 38.2 s | 425.8 s | 41.5 GB | 42.9 GB | 51.9 GB |
| the reference (PyTorch 2.14.1, bfloat16), `--use_int8`, one clip | 51 s | | | 431.6 s for the clip (models loaded) | | | |
| the reference, `base_model/`, one clip | 52 s a forward with LoRAs | | | 433.9 s for the continuation | | | |

"Memory taken from the machine" is the drop of `MemAvailable` while the run was on it (sampled
every 2 s; the GPU's memory is the machine's there), "GPU memory of the process" what
`nvidia-smi` listed for it. "GPU memory in use" is the RTX 5080's total while the run was on it
(sampled every 2 s), about 2 GB of which belonged to the desktop. "Whole run" has the text
encoder (2 to 4 s for the prompt, the 22.7 GB file mapped), the audio encoder (4 s for 4.7 s of
speech) and the frames written as PNGs in it. The first step of a run also reads the
transformer from disk and fills the blocks that stay on the device. The reference's lines are
its own timings on the other DGX Spark, measured when the model was ported.

Before the changes of this section the same runs took 31.5 s a step and 1041.1 s in all (int8,
two clips) and 38.3 s and 594.7 s (bf16) on the RTX 5080, 101.2 s a step and 2392.0 s (int8, two
clips) and 78.3 s and 1009.7 s (bf16) on the GB10; the bf16 clip was run again on this GB10
with the binary of before: 78.4 s a step, denoising 638.7 s, VAE decode 336.3 s. With split
operands (`OXINFER_AVATAR_GPU_EXACT=1`: the kernels of before in the transformer), measured
before the VAE's device passes existed: 62.8 s a step with the int8 weights and 67.9 s with
the bf16 ones on the RTX 5080.

### Where a step goes

`OXINFER_AVATAR_GPU_PROF=1` prints the stages of each forward (a stream synchronization after
each; the host's part of a forward, patch embedding and final layer, is the 0.6 s the sum
leaves). One forward of 24 latent frames of 64x96 with a reference frame
and the OPSD LoRA, on made-up inputs (`examples/longcat_avatar_check DIR bench-dit --lora=b
--device=cuda --avatar=... --base=...`, `--int8` for the int8 weights), in seconds:

| | RTX 5080, int8 | GB10, bf16 | GB10, int8 |
|---|---|---|---|
| copies: the tokens, the rotary and AdaLN tables (on the RTX 5080 also the blocks that do not stay on the card) | 1.67 | 0.88 | 1.51 |
| first layer norm (with the row split on the GB10) | 0.07 | 0.19 | 0.19 |
| `qkv` | 2.09 | 4.20 | 3.81 |
| query, key and value planes | 0.65 | 1.63 | 1.63 |
| self-attention | 11.55 | 16.40 | 16.46 |
| attention output | 0.84 | 1.82 | 1.66 |
| text: norm, queries | 0.83 | 1.74 | 1.64 |
| text: keys, values, attention | 0.22 | 0.41 | 0.42 |
| text: output | 0.68 | 1.40 | 1.23 |
| audio: norm, queries | 0.70 | 1.38 | 1.24 |
| audio: keys, values, attention | 0.07 | 0.32 | 0.32 |
| audio: output, its norm | 0.75 | 1.60 | 1.44 |
| feed-forward: norm, `w1`, `w3`, SwiGLU | 3.99 | 8.64 | 7.51 |
| feed-forward: `w2` | 2.00 | 5.23 | 4.53 |
| the forward | 26.7 | 46.5 | 44.3 |

The self-attention is the 36,864 x 36,864 scores of 32 heads, 48 times a step: 1.07e15
floating-point operations, at 93 TFLOPS on the RTX 5080 and 65 on the GB10; the GEMMs are
0.95e15 a step.

What the GB10 taught (the engine was written on the RTX 5080, and first ran there at half the
reference's speed, the int8 weights slower than the bf16 ones):

- A register-only loop of `mma.sync.m16n8k16` (16 accumulators per warp) runs at 98 TFLOPS on
  the GB10 and 111 on the RTX 5080, while a float32 FMA kernel (the VAE's convolutions) is 2.2
  times slower there (39.7 s against 18.1 s): the GB10's tensor cores nearly match the
  discrete card's, the rest of it is about half (`nvidia-smi` showed 2.4 GHz and 35 to 97 W
  under these loads).
- So the kernels were not waiting on their products. With the products edited out of the GEMM
  a forward lost 2.5 s of 48 s, with the reads of its operands from the device's memory (once
  per 32-wide K step, the rest of the kernel unchanged) 9.4 s. The first profile had the GEMMs
  at 29 TFLOPS and the attention at 38.
- What the default path does about it: GEMMs in tiles of 128 x 128 instead of 64 x 64 (half the
  reads per product), attention in blocks of 128 queries (each read of a step's keys and values
  serves twice the queries), the approximate `2^x` of the device instead of `expf` for the
  probabilities, no planes for the operands' second halves.
- The weights of a GEMM stay in the GPU's cache in smaller groups of outputs: 12 MB there (8 and
  6 MB within 2%), where the RTX 5080 takes 16 MB and a matrix of up to 48 MB whole. With one
  32 MB group the attention's output projection took 3.1 s a forward, in 12 MB groups 2.0 s;
  with 64 MB groups a forward took 110 s instead of 75 s.
- Every pass over the activations (0.6 GB per 4096-wide tensor at this size) costs what it
  reads and writes, at about 230 GB/s (the first layer norm: 0.6 GB read and 0.6 GB written,
  48 times in 0.25 s). On a GPU that shares the machine's memory what follows a matrix is
  therefore part of its GEMM's output: the LoRA side paths (further K steps of the main
  product instead of a read-modify-write of the output), the gated residual, SwiGLU's product;
  and the norms write the float16 operands of the GEMM after them (that last part: 47.6 s to
  46.1 s a forward). The RTX 5080 runs the separate passes as fast (26.8 s a forward
  separate, 27.1 s fused) and keeps them.
- The int8 weights were slower than the bf16 ones because their bytes become float16 in the
  kernel, per tile and K step: twice the tile is half of that per product too. They are the faster ones there now, as on
  the RTX 5080: 42.9 s a step against 46.5 s.

What did not help: 128 x 128 tiles on 128 threads (warps of 64 x 64: slower on the RTX 5080,
no faster on the GB10), fresh float32 sums per K step in the default GEMM (no closer to split
operands), accumulating the attention's output over all key steps in the tensor cores (twice as
far from split operands: their additions truncate).

### Memory on the device

The device engine (`cuda.rs`) keeps the sequence on the card (4.7 GB of activations for 36,864
tokens on the default path, 5.8 GB with split operands) and as many blocks as fit beside it: 23
of the 48 int8 blocks on the 16 GB card (0.36 GB each with their LoRA factors), fewer of the
bf16 ones (0.67 GB each, converted to float16 as they are copied). The others go through one
of two slots while the block before them runs. Between two clips the engine gives its memory
back only when the VAE would not have room otherwise (a 16 GB card; the GB10 keeps the
blocks, and its process then holds them and the VAE's buffers together: the 29.2 and 42.9 GB
of the table, where it held 24.4 and 38.1 GB). `OXINFER_GPU_PLAN_DEBUG=1` prints the placement, `OXINFER_AVATAR_GPU=0` keeps the
transformer on the CPU.

### The VAE

On CUDA the VAE's encode and decode run with the frames kept on the device (`src/cuda_vae.rs`,
walked by `vae/dev.rs`): one launch per causal convolution (its three input frames read where
they are, the 2x upsampling part of its input addressing, bias and residual part of its
output), one per RMS norm; the mid block's attention (one head over the pixels of a latent
frame) goes through the host. The convolutions run on the tensor cores with float16 weights
(exact for the bf16-rounded kernels: the upload checks) and each activation as two halves
`hi + lo 2^-11` after a power of two per frame; `OXINFER_VAE_GPU_F32=1` runs them as float32
FMAs. Before, each convolution was a round trip through host memory
(`gpu::Backend::conv2d`) with the norms on the CPU; `OXINFER_AVATAR_VAE_GPU=0` still does that,
and it is the fallback when the device cannot hold a pass (8.2 GB of buffers to decode a clip
of 768x512, 7.0 GB to encode 13 frames; the 720p buckets were not run).

| 768x512 | RTX 5080 | GB10 |
|---|---|---|
| decode a clip's 24 latent frames (93 frames) | 251.9 s -> 12.2 s | 351 s -> 34.9 s (37.4 s in a later run) |
| decode 5 latent frames (17 frames) | 48.9 s -> 2.5 s | 64.1 s -> 6.0 s |
| encode 13 frames | 24.6 s -> 1.5 s | 30.5 s -> 3.3 s |
| the same in float32 FMAs: 93 frames, 17 frames, encode | 18.1 s, 3.4 s, 1.8 s | 39.7 s, 7.3 s, 3.7 s |

(`examples/longcat_avatar_check DIR bench-vae --device=cuda --base=... [--frames=24]`, made-up
latents; the clip's "before" figures are the runs above.) The GB10 gains less from the tensor
cores here than the discrete card: the passes over the frames (151 MB each at the last level)
are a larger part of its time.

On the CPU alone (`--device cpu`), the transformer runs in float32 on weights dequantized a
matrix at a time: a 320x192 clip of 13 frames (960 tokens) takes 24.2 s a step and 230 s in all.
A full-size clip on the CPU was not run.

## Not supported

- **The vocal separator** (`vocal_separator/Kim_Vocal_2.onnx`, an MDX-Net the reference runs with
  ONNX Runtime before Whisper): oxinfer has no ONNX path and does not port that network. The
  speech goes to the audio encoder as given, so a recording with music or noise under the voice
  drives the mouth with all of it; separate the vocals beforehand for such audio.
- **The server, the web GUI and the C API**: the model is in the CLI and the library
  (`oxinfer::longcat_avatar::AvatarPipeline`) only. `/v1/videos` is MiniMax-H3's.
- **Two speakers on CUDA**: the transformer runs on the CPU for those requests (the engine
  declines them); a third, silent "background" speaker (`others` boxes of the reference's JSON)
  is not offered at all.
- The undistilled continuation's key / value cache and `enhance_hf` (the mixed-LoRA setting the
  model card recommends turns both off), a statically merged single LoRA (`--mix_steps=0
  --lora_path`: here `--mix-steps 0` runs the OPSD LoRA alone as a side path), block-sparse
  attention, FP8 feed-forwards, `torch.compile`, several GPUs, the 1.0 checkpoints (wav2vec2
  features) and 1.5's loudness normalization.
- `ftfy`'s mojibake repair, Unicode normalization and control-character removal on the prompt.
- Metal and Vulkan: the transformer has no engine there and runs on the CPU kernels.

## Not run

- A float32 reference for the continuation and for the bf16 transformer's clips: those are
  compared with the reference's bfloat16 run only (above), 41 to 45 dB away as the int8 first
  clip is, whose float32 reference is at 73.6 dB.
- The reference's outputs against the engine as it is after the speed work (see the table of
  each stage against itself before), and split operands' timings since then.
- The VAE's device passes at the 720p buckets (they need more than the 8.2 GB of 768x512).
- Two speakers, the 720p buckets, `--no-distill` (guidance, 50 steps) and prompts without a
  reference image on the released weights at full size: the two-speaker forward, guidance and
  the text-only forward are checked on the tiny checkpoints only; a text-only clip of 320x192
  ran on the released weights (a recognizable scene; that size is far from what the model was
  trained on).
- More than two clips in a row.
- Windows, macOS, iOS, Android and wasm: compiled by the CI's checks, not run (no CUDA engine
  outside Linux / Windows with an NVIDIA GPU; the CPU path is plain Rust).
