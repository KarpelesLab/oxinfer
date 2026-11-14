# Video generation: MiniMax-H3

`MiniMaxAI/MiniMax-H3` generates video with a stereo soundtrack from text (`t2va`), from a first
and / or last frame (`fl2va`) or from reference images, videos and audio (`ref2va`). oxinfer runs
the first two, video only: `src/minimax_h3/video.rs` follows diffusers' modular `minimax_h3`
pipeline (`MiniMaxH3ModularPipeline`), with the DiT and the ViT VAE decoder it shares with
[MiniMax-H3-Image](models.md).

```sh
# text to video: 124 frames (5.2 s at 24 fps) at 1344x768, 49 forwards
./target/release/oxinfer --model /path/to/MiniMax-H3 --device cpu \
    -p "integrated_multimodal_description: [Shot 1] A red fox trots across a snowy clearing ..." -o fox.avi
# smaller and shorter
./target/release/oxinfer --model /path/to/MiniMax-H3 --device cpu -p "..." \
    --size 448x256 --frames 124 --steps 50 --seed 1 -o fox.avi
# first frame (and last frame) to video; the canvas takes the first image's aspect ratio
./target/release/oxinfer --model /path/to/MiniMax-H3 --device cpu -p "..." \
    --init-image first.png --last-image last.png -o clip.avi
```

`--model` is a folder in the repository's diffusers layout: `model_index.json`, `transformer/`,
`vae/`, `text_encoder/` (or `OXINFER_H3_TEXT_ENCODER`), `tokenizer/`, and the two
`scheduler*/scheduler_config.json` (140 GB of the repository's 498 GB:
`hf download MiniMaxAI/MiniMax-H3 --include model_index.json "transformer/*" "vae/*" "text_encoder/*" "tokenizer/*" "scheduler/*" "audio_scheduler/*"`).
Passing the repository id itself has not been tried.

| flag | |
|---|---|
| `--frames N` | frames at 24 fps, rounded up to `17 n + 5` (default 124). The release is for 5 to 15 s (124 to 362); fewer frames run with a warning |
| `--size WxH` | multiples of 32; default 16:9 with a 768 short edge (1344x768), or the first keyframe's aspect ratio under the same pixel budget |
| `--steps N` | points of the sigma grid, as diffusers' `num_inference_steps`: `N - 1` forwards (default 50) |
| `--seed N` | noise seed (oxinfer's generator: not torch's, so a seed does not reproduce a diffusers video) |
| `--init-image F`, `--last-image F` | the first / last frame |
| `-o FILE` | `.avi`: Motion JPEG in AVI; `.webp`: animated WebP (lossless only, about 3 s per 320x288 frame to encode); anything else: a PNG sequence `FILE_0000.png`, ... |

There is no pure-Rust H.264, VP9 or AV1 encoder in the oxideav crates oxinfer uses (the H.264
crate decodes only), so no `.mp4` / `.webm` is written: Motion JPEG in AVI plays in ffmpeg-based
players and converts with `ffmpeg -i fox.avi fox.mp4`. Without the `media` feature only the PNG
sequence is available. **No audio track is written**: the audio latents are denoised with the video
(the model is trained on the joint sequence) and then dropped, the audio VAE is not implemented.

`OXINFER_H3_PROMPT_EMBEDS=FILE` keeps the text encoder's output of a text-only prompt in FILE
(written when missing, read when present; the run stops after writing it when the transformer's
weights are absent), so the 63 GB text encoder and the 66 GB transformer need not be on disk
together.

## The release

| component | folder | size | |
|---|---|---|---|
| transformer (`t2va`, `fl2va`) | `transformer/` | 66.3 GB, 33.1B parameters, bf16 (patch, timestep and output layers float32) | 50 blocks, width 5376, 56 heads of 128, SwiGLU 14336, text width 5120; about 13B of the parameters are the AdaLN projections (`6 x 5376 x 3` outputs per block from a 2688-wide timestep embedding) |
| transformer (`ref2va`) | `transformer_ref/` | 66.3 GB | same architecture, other weights; not used |
| video VAE | `vae/` | 10.4 GB, float32 | `f16t4d24`: a causal 3D CNN encoder (128 to 1024 channels, 16x in space, 4x in time, 24 latent channels) and a 36-layer ViT decoder (width 2048, 32 heads, 3-axis RoPE) that emits a 4x16x16 pixel block per latent voxel |
| audio VAE | `audio_vae/` | 0.6 GB | 32 kHz, 40 latents of 32 channels per second and channel (DAC encoder, BigVGAN decoder); not implemented |
| text encoder | `text_encoder/` | 63.4 GB, bf16 | Qwen3-VL-32B (64 layers, width 5120); `hidden_states[50]` conditions the transformer |
| schedulers | `scheduler/`, `audio_scheduler/` | | rectified flow, shift 12 (video) and 3 (audio) |

`FL2VA/` and `Ref2VA/` hold the same checkpoints in MiniMax's original layout (their own `.py`
modules; the transformer's layout there is the one `Dit::load` already reads for H3-Image, with
`audio_patch_proj` / `final_layer.audio_out`; the video VAE's `source/model.safetensors` layout is
not read). The released weights are guidance-distilled (one forward per step, no negative prompt)
but not step-distilled: diffusers' default is 50 grid points.

What a request does (`video.rs` has the details):

- the prompt verbatim, no chat template, through the text encoder; each keyframe prepends
  `"<Picture i>: "` and its Qwen3-VL vision block, whose rows are tagged as video for the AdaLN;
- `17 n + 5` frames become `5 n + 2` latent frames of `H/16 x W/16`, packed as 2x2 patches; the
  soundtrack is `round(frames / 24 * 40)` latents per stereo channel;
- one sequence `[text | keyframe rows | audio rows | video rows]` with `(t, h, w)` rotary
  positions; a 1344x768, 124-frame request has 37 x 1008 = 37,296 video rows and 414 audio rows;
- video and audio rows step down their own schedules in one forward; keyframe rows (the VAE
  encoder's sampled posterior, rounded through float16) are noised to `t = 0.999` and held there;
- the video rows go through the VAE decoder: clips of 7 latent frames in 256-pixel tiles
  overlapping by 64, consecutive clips cross-faded over 5 frames.

Against MiniMax-H3-Image, which oxinfer already ran: the same block stack, AdaLN table, RoPE and
text path; the video model adds `audio_proj_in` / `audio_proj_out` and rows of three modalities at
several timesteps scattered through the sequence, has one output head instead of six per-step
(PDD) heads, and runs a shifted 50-point schedule in the `x_t / x_0` blend form instead of six
fixed sigmas. The VAE decoder is the image decoder with a third rotary axis (`rope_dim_ratio`
0.75), a `4 x 16 x 16` patch head and 36 layers; the encoder is new.

## What was checked

`tools/minimax_h3_video_tiny.py` builds tiny random checkpoints with diffusers and
`examples/h3_video_check.rs` compares, in float32 on the CPU (max absolute difference):

| | |
|---|---|
| VAE encoder, a 22-frame 320x288 clip (2 x 2 tiles) / one frame | 1.3e-6 / 7.7e-7 |
| VAE decoder, 7 / 12 / 9 latent frames (one clip, two cross-faded clips, a padded clip) | 7.2e-7 / 7.0e-7 / 6.0e-7 |
| packed layout positions (text, first and last keyframe, audio, video) and both sigma grids | identical |
| DiT velocities of the first step (video / audio rows; derived from the stepped rows) | 8.2e-6 / 2.5e-6 |
| video / audio rows after the 5-step schedule, keyframe rows held | 7.2e-7 / 4.8e-7 |

On the real weights (CPU, 64 cores, 125 GB RAM):

- the video VAE against diffusers in float32, a synthetic 22-frame 320x256 clip: encoder moments
  within 2.1e-4 (mean magnitude 4.2; one frame: 4.3e-5), the decode of its posterior mean within
  2.9e-5 (128 dB PSNR);
- one text-to-video run, not compared with a reference: a 98-token prompt in the release's
  `integrated_multimodal_description: ...` form (a fox in a snowy clearing), `--size 448x256
  --frames 22 --steps 50 --seed 1`. 858 latent rows; text encoder 29 s, 49 forwards 1836 s (37 s
  each), VAE decode 50 s. The 22 frames show the prompt's scene, sharp and temporally coherent
  (the fox turns its head towards the camera and its breath appears). 22 frames is below the 5 s
  the release is trained for, and nothing larger or longer was run: the cost per forward grows at
  least linearly with the rows, and a default 1344x768, 124-frame request has 44 times as many.
  Running diffusers' pipeline on the 66 GB transformer for a comparison was not attempted.

Not checked or not done:

- no end-to-end comparison with diffusers on the real weights (see above for what was run);
- the prompt path with keyframes (label and vision block through `QwenVl`) ran only on a tiny
  random Qwen3-VL, without a numeric comparison; keyframes are resized with oxinfer's Lanczos, not
  compared with PIL's; the posterior and all noise come from oxinfer's generator;
- `ref2va` (`transformer_ref/`), audio output, LoRAs (the 4 / 8-step turbo LoRAs, Pika H3 Flex,
  character swap), the original `FL2VA/` layout's video VAE, GPU execution (the 66 GB transformer
  runs on the CPU from mapped weights).

## Smaller variants (not supported)

The official release needs 140 GB of downloads. Smaller files exist and none of them loads yet:

| repository | files | why not |
|---|---|---|
| `Comfy-Org/MiniMax-H3` | single-file transformers: bf16 66.3 GB, "pruned" bf16 40.2 GB, pruned fp8-scaled 21.0 GB, int8-convrot 21.0 / 34.0 GB, w6a8 16.0 GB; text encoder as nvfp4 15.7 GB; video VAE fp16 5.2 GB; 4- and 8-step turbo LoRAs (2.0 GB) | ComfyUI's single-file key layout and its quantized formats were not read; the turbo LoRAs would cut 49 forwards to 4 or 8 |
| `VideoRebirthAI/hyperflow-h3-flash` | a 13.7B few-step (4 / 6 / 8) transformer as GGUF, 7.2 GB (Q3_K_M) to 14.9 GB (Q8_0), or 16 GB rotated INT8 | another architecture on the same blocks, per its `hyperflow_h3_flash_transformer.py`: a second (endpoint) timestep embedder mixed with gate 0.25, three dropped blocks, 102 low-rank factored linears, its own sigma grids. The GGUF tensor layout was not read |

Both would also need the text encoder in a smaller format than the 63 GB bf16 release.
