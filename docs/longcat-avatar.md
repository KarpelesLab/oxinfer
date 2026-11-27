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
blocks on the device against the CPU in both modes; skipped without an NVIDIA GPU).

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

## Speed and memory

Measured on the repository's `single` example at 768x512 (the bucket of `man.png`), clips of 93
frames (24 latent frames, 36,864 tokens), 8 steps, on a Threadripper 9970X with an RTX 5080
(16 GB), from a warm page cache, a `--profile ci` build (the release profile without LTO):

| run | a step | denoising | VAE decode | whole run | peak resident memory | GPU memory in use |
|---|---|---|---|---|---|---|
| `longcat-avatar-int8`, two clips (173 frames, 6.9 s) | 31.5 s (30.5 s in the continuation) | 522.7 s | 484.7 s | 1041.1 s | 30.5 GB | 14.8 GB |
| `longcat-avatar` (bf16), one clip | 38.3 s | 314.0 s | 251.9 s | 594.7 s | 44.9 GB | 15.0 GB |
| `longcat-avatar-int8`, split operands, one clip | 62.8 s | 506.6 s | 241.7 s | 779.3 s | | |
| `longcat-avatar`, split operands, one clip | 67.9 s | 557.6 s | 248.1 s | 835.1 s | 44.3 GB | 15.0 GB |

The same example on a DGX Spark (GB10, 20 cores, 121 GB shared by the CPU and the GPU; all 48
blocks stay on the device, 17.4 GB with their LoRA factors for the int8 weights and 32.2 GB for
the bf16 ones):

| run | a step | denoising | VAE decode | whole run | peak resident memory | GPU memory of the process | memory taken from the machine |
|---|---|---|---|---|---|---|---|
| `longcat-avatar-int8`, two clips (173 frames, 6.9 s) | 101.2 s (101.0 s in the continuation) | 1656.7 s | 706.2 s | 2392.0 s | 31.9 GB | 24.4 GB | 35.4 GB |
| `longcat-avatar` (bf16), one clip | 78.3 s | 633.3 s | 351.2 s | 1009.7 s | 47.1 GB | 38.1 GB | 50.0 GB |
| the reference (PyTorch 2.14.1, bfloat16), `--use_int8`, one clip | 51 s | | | 431.6 s for the clip (models loaded) | | | |
| the reference, `base_model/`, one clip | 52 s a forward with LoRAs | | | 433.9 s for the continuation | | | |

"Memory taken from the machine" is the drop of `MemAvailable` while the run was on it (sampled
every 2 s; the GPU's memory is the machine's there), "GPU memory of the process" what
`nvidia-smi` listed for it. oxinfer's engine is half the reference's speed on this machine, and
the int8 weights are slower than the bf16 ones there, the opposite of the RTX 5080: the engine
was tuned on the latter, and where the GB10's time goes was not profiled.

"GPU memory in use" is the card's total while the run was on it (sampled every 2 s), about 2 GB
of which belonged to the desktop. "Whole run" has the text encoder (2 to 4 s for the prompt, the 22.7 GB file mapped), the audio
encoder (4 s for 4.7 s of speech) and the H.264 / AAC file (14 s) in it. The first step of a run
also reads the transformer from disk and fills the blocks that stay on the card.

Where a step goes on the RTX 5080 (`OXINFER_AVATAR_GPU_PROF=1`), int8 weights, in seconds:

| | self-attention | feed-forward | `qkv` and query / key preparation | text cross-attention | audio cross-attention | attention output | block copies |
|---|---|---|---|---|---|---|---|
| float16 operands (default) | 15.8 | 6.3 | 3.0 | 1.9 | 1.7 | 0.9 | 1.5 |
| split operands (`OXINFER_AVATAR_GPU_EXACT=1`) | 35.7 | 11.9 | 5.0 | 3.5 | 3.0 | 1.6 | 1.5 |

The self-attention is the 36,864 x 36,864 scores of 32 heads, 48 times a step.

The device engine (`cuda.rs`) keeps the sequence on the card (5.8 GB of activations for 36,864
tokens) and as many blocks as fit beside it: 14 of the 48 int8 blocks (0.36 GB each with their
LoRA factors), 7 of the bf16 ones (0.67 GB each, converted to float16 as they are copied). The
others go through one of two slots while the block before them runs.
`OXINFER_GPU_PLAN_DEBUG=1` prints the placement, `OXINFER_AVATAR_GPU=0` keeps the transformer on
the CPU.

The VAE is the slow part next to that: its convolutions go through the shared accelerator path
(`gpu::Backend::conv2d`) one frame and one layer at a time from host memory, so a decode is
bounded by the copies (17 frames of 768x512: 44.2 s on the RTX 5080, 187.5 s on the CPU; encoding
13 frames: 22.5 s and 78.7 s; a clip's 93 frames: 351 s on the GB10). A pass that keeps the frames on the device is not written.

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
- A profile of the engine on the GB10, where a step takes twice the reference's time.
- Two speakers, the 720p buckets, `--no-distill` (guidance, 50 steps) and prompts without a
  reference image on the released weights at full size: the two-speaker forward, guidance and
  the text-only forward are checked on the tiny checkpoints only; a text-only clip of 320x192
  ran on the released weights (a recognizable scene; that size is far from what the model was
  trained on).
- More than two clips in a row.
- Windows, macOS, iOS, Android and wasm: compiled by the CI's checks, not run (no CUDA engine
  outside Linux / Windows with an NVIDIA GPU; the CPU path is plain Rust).
