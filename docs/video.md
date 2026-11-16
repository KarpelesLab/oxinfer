# Video generation: MiniMax-H3

`MiniMaxAI/MiniMax-H3` generates video with a stereo soundtrack from text (`t2va`), from a first
and / or last frame (`fl2va`) or from reference images, videos and audio (`ref2va`). oxinfer runs
the first two: `src/minimax_h3/video.rs` follows diffusers' modular `minimax_h3` pipeline
(`MiniMaxH3ModularPipeline`), with the DiT and the ViT VAE decoder it shares with
[MiniMax-H3-Image](models.md), and `src/minimax_h3/audio_vae.rs` decodes the soundtrack.

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
`vae/`, `audio_vae/`, `text_encoder/` (or `OXINFER_H3_TEXT_ENCODER`), `tokenizer/`, and the two
`scheduler*/scheduler_config.json` (141 GB of the repository's 498 GB:
`hf download MiniMaxAI/MiniMax-H3 --include model_index.json "transformer/*" "vae/*" "audio_vae/*" "text_encoder/*" "tokenizer/*" "scheduler/*" "audio_scheduler/*"`).
Without `audio_vae/` (0.6 GB) the soundtrack is skipped and a message says so. Passing the
repository id itself has not been tried.

| flag | |
|---|---|
| `--frames N` | frames at 24 fps, rounded up to `17 n + 5` (default 124). The release is for 5 to 15 s (124 to 362); fewer frames run with a warning |
| `--size WxH` | multiples of 32; default 16:9 with a 768 short edge (1344x768), or the first keyframe's aspect ratio under the same pixel budget |
| `--steps N` | points of the sigma grid, as diffusers' `num_inference_steps`: `N - 1` forwards (default 50) |
| `--seed N` | noise seed (oxinfer's generator: not torch's, so a seed does not reproduce a diffusers video) |
| `--init-image F`, `--last-image F` | the first / last frame |
| `-o FILE` | `.avi`: Motion JPEG and the soundtrack in AVI; `.webp`: animated WebP (lossless only, about 3 s per 320x288 frame to encode) and `FILE.wav`; anything else: a PNG sequence `FILE_0000.png`, ... and `FILE.wav` |

There is no pure-Rust H.264, VP9 or AV1 encoder in the oxideav crates oxinfer uses (the H.264
crate decodes only), so no `.mp4` / `.webm` is written: Motion JPEG in AVI plays in ffmpeg-based
players and converts with `ffmpeg -i fox.avi fox.mp4`. Without the `media` feature only the PNG
sequence is available.

The soundtrack is stereo at 32 kHz, 800 samples per audio latent (a 124-frame request: 207
latents, 5.175 s for 5.167 s of frames; nothing is trimmed). In an `.avi` it is a second stream
of 16-bit PCM, one chunk after each frame (`ffmpeg -i fox.avi fox.mp4` keeps it). Animated WebP
has no audio and a PNG sequence is only frames: both get a 16-bit stereo WAV next to them, named
as the output without its extension (`-o fox.webp` and `-o fox.png` both write `fox.wav`).

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
| audio VAE | `audio_vae/` | 0.6 GB, float32 | mono, 32 kHz, 40 latents of 32 channels per second; a stereo soundtrack is two latent sequences. Decoder (0.26 GB of the file): a 1x1 convolution to 2048 channels and BigVGAN, seven transposed convolutions (x5, x5, x2 five times; 1024 to 8 channels) each followed by three AMP blocks with anti-aliased SnakeBeta activations, clamped to [-1, 1]. The encoder (a DAC waveform encoder and a causal attention projection) is not implemented: only `ref2va`'s reference audio goes through it |
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
  overlapping by 64, consecutive clips cross-faded over 5 frames;
- the audio rows are split into their two channels, de-normalized (`latents * latents_std +
  latents_mean`, 32 values each in the VAE's config) and decoded one channel at a time, whole (no
  windows: 15 s take 3.2 s on the CPU).

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

The audio VAE's decoder has its own pair, `tools/minimax_h3_audio_tiny.py` and
`examples/h3_audio_check.rs`: stereo audio rows as the DiT emits them, through diffusers'
unpacking, de-normalization and `AutoencoderKLMiniMaxH3Audio.decode`, against
`AudioDecoder::decode_rows` (float32, CPU):

| | max abs diff | reference RMS | SNR |
|---|---|---|---|
| tiny random checkpoint (stages x5, x2, x2 with kernels 9, 4, 4), 37 latents per channel | 8.6e-8 | 0.038 | 124 dB |
| real `audio_vae/`, the encoder's latents of a synthetic stereo signal, 207 latents (5.2 s) | 8.8e-6 | 0.27 | 121 dB |
| real `audio_vae/`, seeded normal rows, 207 latents (5.2 s; 4.0% of the samples at the clamp) | 4.7e-5 | 0.52 | 103 dB |
| real `audio_vae/`, seeded normal rows, 603 latents (15.1 s, a 362-frame request) | 5.9e-5 | 0.52 | 104 dB |

The synthetic signal (left: a 100 Hz to 8 kHz sweep; right: a 220 Hz tone with four harmonics,
amplitude-modulated at 3 Hz) is encoded by diffusers (the posterior's mode, normalized) to give
rows in the range the decoder is trained for; diffusers' own decode of them correlates 0.962 /
0.999 with the signal. The decoder takes 1.2 s for the 5.2 s soundtrack and 3.2 s for the 15.1 s
one (64 cores). Normal rows are what the DiT starts from, not what it ends on: they decode to
loud noise, which exercises the clamp.

The AVI's audio stream is checked by a unit test that demuxes it with oxideav-avi (every frame
and every PCM byte comes back); ffprobe reads the same file as `mjpeg` + `pcm_s16le`, 32 kHz,
2 channels.

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
  Running diffusers' pipeline on the 66 GB transformer for a comparison was not attempted. That
  run predates the audio decoder: its audio latents were dropped.

Not checked or not done:

- no end-to-end comparison with diffusers on the real weights (see above for what was run);
- the prompt path with keyframes (label and vision block through `QwenVl`) ran only on a tiny
  random Qwen3-VL, without a numeric comparison; keyframes are resized with oxinfer's Lanczos, not
  compared with PIL's; the posterior and all noise come from oxinfer's generator;
- **no generated soundtrack was decoded or listened to**: the audio decoder was checked on its
  own (above), with latents that did not come from the transformer, and no video run was made
  with it (the transformer and text encoder were not on disk). Whether the real model's audio
  rows decode to the sound the prompt describes is untested; so is the audio VAE on anything but
  the CPU, and the WAV / AVI files were read back by programs, not played;
- `ref2va` (`transformer_ref/`) and with it the audio VAE's encoder, a run with a LoRA or with the
  smaller files (they load, see below), the character-swap LoRA, the original `FL2VA/` layout's video and audio VAEs
  (`FL2VA/audio_vae/model.safetensors` has the same tensor names as `audio_vae/`, but its
  `config.yaml` / `metadata.json` are not read), GPU execution (the 66 GB transformer runs on the
  CPU from mapped weights).

## Smaller files and turbo LoRAs

The official release needs 140 GB of downloads and 49 forwards. The repackaged files of
`Comfy-Org/MiniMax-H3`, the turbo LoRAs of `lightx2v/Minimax-h3-Turbo` and a GGUF of the text
encoder bring that to 48 to 63 GB and 4 or 8 forwards. **None of these files was downloaded or
run here**: their layouts were read from the safetensors headers and from sampled rows (HTTP range
requests), the loaders are checked on tiny random checkpoints in the same forms (below).

```sh
# 4 forwards: the pruned FP8 transformer (21.0 GB), the fp16 VAE (5.2 GB), the 4-step turbo LoRA
# (2.0 GB) and Qwen3-VL-32B-Instruct as a GGUF text encoder (Q8_0: 34.8 GB, Q4_K_M: 19.8 GB)
./target/release/oxinfer --device cpu \
    --model diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors \
    --vae vae/minimax_h3_video_vae_fp16.safetensors \
    --lora loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors \
    --text-encoder Qwen3-VL-32B-Instruct-Q8_0.gguf \
    -p "integrated_multimodal_description: [Shot 1] ..." -o fox.avi
# the release folder, 8 forwards
./target/release/oxinfer --model /path/to/MiniMax-H3 --device cpu \
    --lora minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors -p "..." -o fox.avi
```

| flag | |
|---|---|
| `--model FILE` | a single-file transformer in ComfyUI's layout instead of the release folder; `--vae` is then required, the shifts are 12 and 3 (there is no scheduler folder), and there is no soundtrack: the audio VAE is only read from a release folder's `audio_vae/` |
| `--transformer FILE\|DIR`, `--vae FILE\|DIR` | one part of a release folder from another place |
| `--text-encoder DIR\|FILE.gguf` | a Qwen3-VL-32B-Instruct folder, or its language model as GGUF (text-only prompts: no `--init-image` / `--last-image`) |
| `--quant q8` | the text encoder's safetensors weights quantized at load (memory, not download; `q4` is taken as `q8`) |
| `--lora FILE[:SCALE]` | LoRA files on the transformer, repeatable. A file named `*turbo_Nstep*` sets `--steps N + 1` and the shifts: 6 / 3 for the `768p` files, 12 / 3 for the others |
| `--shift F`, `--audio-shift F` | the sigma shifts of the video and audio schedules |

Each of these takes a local path or `org/name/path/in/repo.safetensors` (`.gguf`), fetched like
`--lora` files elsewhere; fetching was not tried for these repositories.

### What the files are

Read from the files' headers and, where rows are quoted, from the same rows of several files.

| file | size | contents |
|---|---|---|
| `diffusion_models/minimax_h3_fl2va_bf16.safetensors` | 66.3 GB | the 535 tensors of MiniMax's own layout (`FL2VA/transformer`) with the same names, but `qkv_proj` as `[q; k; v]` where MiniMax's is `[q k v]` per head (row 0 of `to_k` of diffusers' block 0 is row 7168 of ComfyUI's `qkv_proj` and row 128 of MiniMax's, bit for bit); `fc1` is `[gate; value]` in both. The configuration is in the header |
| `..._pruned_bf16.safetensors` | 40.2 GB | the same matrices (rows 0 and 7168 of block 0's `qkv_proj` are identical) without the time embedder and the 26 GB of AdaLN weights: `adaln_t_table` `[1025, 8]` and float16 `[.., 8]` projections, see below |
| `..._pruned_fp8_scaled.safetensors` | 21.0 GB | the pruned form with the four matrices of each of the 50 blocks as `float8_e4m3fn` times a scalar `weight_scale` (the refiner's stay bf16) |
| `vae/minimax_h3_video_vae_fp16.safetensors` | 5.2 GB | MiniMax's original tensor names (`FL2VA/video_vae`: `encoder.down.I.block.J`, a fused per-head `to_qkv`, `ff.w1` as `[gate; value]`) in float16, the configuration in the header. Sampled rows equal diffusers' `vae/` within float16 rounding (1.5e-5; the encoder's within 3e-8) |
| `loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors`, `..._turbo_8step_v1.0_comfyui_bf16.safetensors` | 2.0 GB | rank-128 pairs on `qkv_proj` (fused: rank 384 with a block-diagonal `B`), `out_proj`, `fc1`, `fc2` of the 50 blocks and the 2 refiner blocks, `.alpha` 128 / 384 (4-step) and 8 / 24 (8-step: scale 1/16) |
| `text_encoder/` of the release | 63.4 GB | Qwen3-VL-32B-Instruct itself: its 14 shards have the SHA-256 of `Qwen/Qwen3-VL-32B-Instruct`'s. `unsloth/Qwen3-VL-32B-Instruct-GGUF` holds the language model in llama.cpp's `qwen3vl` layout; sampled rows of its BF16 file (queries, keys, values, output, the three MLP matrices, the embedding, three norms) are the release's bit for bit, its vocabulary and merges are the release tokenizer's |

**The pruned form.** `comfy/ldm/minimax/model.py` (`_time_embedding`) reads the row of
`adaln_t_table` at `t * 1024` by linear interpolation and gives it to every AdaLN projection
without the SiLU. Checked against the full file: with the bf16 file's time embedder,
`silu(time_embedder(i / 1024))` minus its mean lies in the span of the table's 8 columns to 4.5e-5
(relative; the curve's 9th singular value is 3.7e-5 of the first), and for sampled rows of blocks
0 and 31 and of the final layer `W silu(temb) + b` of the full file and `W8 table[i] + b8` of the
pruned one differ by at most 2.7e-4 over the 1025 grid points (values up to 1.0), 1.8e-4 at the
midpoints between them. So the pruned files are the same model to about 3e-4 in the AdaLN vectors;
what that does to a video was not measured.

**The turbo LoRAs.** Settings from lightx2v's model table (`github.com/ModelTC/Minimax-H3-Turbo`):
`N` evaluations on the grid `(N - i) / N`, which is `--steps N + 1` here; video / audio shifts
12 / 3 for the 544p files (4-step v0.1, 8-step v1.0) and 6 / 3 for the `768p` ones; no guidance
(as the base model). Their script and their ComfyUI workflow step with Euler, as oxinfer does;
ComfyUI's own template uses `res_multistep` with the 8-step LoRA at shift 12 / 3, which is not
implemented. The LoRA is not merged: each targeted layer keeps its stored weights and adds
`alpha / rank * B (A x)` in float32, as for FLUX. Key namings read: ComfyUI's (the `*_comfyui_*`
files) and diffusers' PEFT keys (`transformer_blocks.N.attn.to_q.lora_A.default.weight`, alpha from
the header's `alpha`); either goes on any of the three layouts of the base weights. The
`8step_v1.2` file also updates the AdaLN projections, which a pruned transformer no longer has in
that form: it loads on the unpruned transformers only (an error otherwise). `pika_h3_flex_*` have
the same keys (alpha folded in) and load as LoRAs; their sampling settings were not looked up.

### What was checked

`tools/minimax_h3_video_comfy.py` writes the tiny checkpoints of `tools/minimax_h3_video_tiny.py`
in these forms, with a LoRA in both namings, and diffusers references computed on the changed
weights; `examples/h3_video_check.rs --transformer / --vae / --lora` compares (float32, CPU, max
absolute difference; the unchanged checkpoint gives 8.2e-6 / 2.5e-6 and 7.2e-7 / 4.8e-7):

| transformer, LoRA (reference) | first step's velocities, video / audio | rows after the 5 steps, video / audio |
|---|---|---|
| ComfyUI bf16 single file (the diffusers model) | 8.2e-6 / 2.5e-6 | 4.8e-7 / 4.8e-7 |
| pruned (diffusers' model computing the curve form) | 1.1e-5 / 1.8e-6 | 6.0e-7 / 7.2e-7 |
| pruned FP8 scaled (the curve form on the decoded weights) | 8.8e-6 / 2.9e-6 | 4.8e-7 / 4.8e-7 |
| LoRA in ComfyUI's naming on the diffusers folder, on the ComfyUI file, on MiniMax's layout (the LoRA merged into the weights) | 9.4e-6 / 2.4e-6 | 4.8e-7 / 4.8e-7 |
| LoRA in PEFT naming on the same three (the same) | 9.4e-6 / 2.4e-6 | 4.8e-7 / 4.8e-7 |
| LoRA with AdaLN modules on the diffusers folder and the ComfyUI file (merged) | 9.7e-6 / 2.7e-6 | 6.0e-7 / 4.8e-7 |
| LoRA on the pruned FP8 file (merged into the decoded weights, curve form) | 1.1e-5 / 2.0e-6 | 6.0e-7 / 4.8e-7 |
| controls: no LoRA, and the LoRA at half strength, against the merged reference | 8.5e-2 / 9.3e-2, 4.3e-2 / 4.3e-2 | 8.7e-2 / 8.5e-2, 4.4e-2 / 4.3e-2 |

The tiny pruned checkpoint itself is 1.5e-4 from the unpruned one (a rank-8 basis and float16
projections of a random 24-wide curve). The fp16 single-file VAE against diffusers with its weights
rounded to float16: encoder 1.1e-6 (clip) / 5.1e-7 (one frame), decoder 7.2e-7 / 7.2e-7 / 6.0e-7.

Text encoder as GGUF: a tiny random Qwen3-VL written with the tensor names and metadata keys of
the published `qwen3vl` files (`tools/qwen3vl_gguf.py`) gives transformers' logits through
`examples/qwen4exp_check.rs` as the folder does (5.1e-7, same top token on 13 of 13 steps). The
header of the real `Qwen3-VL-32B-Instruct-Q4_K_M.gguf` (its first 24 MB in a sparse file: the
weights read as zeros) loads through `QwenVl` as 64 layers of width 5120 with the release's
configuration values, and three prompts tokenize to the same ids as with the release's
`tokenizer.json`.

The command line was run end to end on the tiny files (the pruned FP8 transformer and the fp16
VAE as single files, a LoRA named `*turbo_4step*768p*`, a stored prompt embedding): 4 forwards at
shifts 6 / 3 and 22 frames written; `--steps`, `--shift` and `--audio-shift` override the LoRA's
settings, and a single-file `--model` without `--vae` is refused.

Not checked or not done:

- no run on the real files: not a single-file transformer, the fp16 VAE, a turbo LoRA or a GGUF
  text encoder, so nothing is known here about speed, memory or the quality of 4- and 8-step
  videos;
- what a quantized text encoder (`--quant q8`, a Q8_0 or Q4_K_M GGUF) does to `hidden_states[50]`
  and to the video. Qwen-Image's text encoder at Q4 moves its hidden states by about 25%
  ([qwen-image.md](qwen-image.md)), so the larger GGUF files are the safer choice until measured;
- the video command itself with a GGUF text encoder (the tiny encoders have 3 layers and the
  pipeline conditions on the 50th): the parts above were run separately;
- keyframes with a GGUF text encoder (the vision tower is a separate `mmproj` file, not read
  here), LoRAs on the text encoder, LoRA keys other than the two namings above.

### Still not supported

| files | why not |
|---|---|
| `Comfy-Org/MiniMax-H3`: `*_int8_convrot.safetensors` (transformer 21.0 / 34.0 GB, VAE 2.8 GB, text encoder 27.1 GB; the format ComfyUI's README prefers over FP8) | `int8_tensorwise` with `convrot`: int8 rows with a scale per row, to be multiplied by the input after a grouped Hadamard rotation (groups of 256, ComfyUI's `QUANTIZATION.md`). The rotation itself is in the `comfy_kitchen` package, not in ComfyUI's source: undoing a normalized Sylvester Hadamard per group of 256 on sampled rows of the pruned file does not give the bf16 rows (correlation 0.00 to 0.03) |
| `..._pruned_w6a8.safetensors` (16.0 GB) | `w6a8_int8`: 6-bit codes packed into int8 `[N, 3K/4]` with FP8 group scales (`weight_s_rel`, groups of 32) and a scale per row (`weight_s_channel`), on the same rotation |
| `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` (15.7 GB) | the first 50 layers as NVFP4 with ComfyUI's packing: on sampled rows, reading it as ModelOpt's NVFP4 (which oxinfer has) gives no correlation with the bf16 rows, and with the nibbles swapped 0.76 to 0.93 at the wrong magnitude, so the block scales are in `comfy_kitchen`'s "swizzled" order. It also needs `pre_quant_scale` (an input scale on `o_proj` and `down_proj`), the int8 embedding with a scale per row, and the file's names (`model.layers.N`, no configuration, no final norm) |
| `text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors` (51.5 GB) | the first 50 layers and the vision tower under `model.layers.N` / `visual.*` without a configuration: a key remap that was not written (the GGUF files are smaller) |
| `diffusion_models/minimax_h3_ref2va_*`, `model_patches/*controlnet*`, `embeddings/*`, `vae/minimax_h3_sol_vae_*` | `ref2va`, the Fun ControlNet, prompt embeddings and the "sol" VAE are not implemented |
| `VideoRebirthAI/hyperflow-h3-flash` | a 13.7B few-step (4 / 6 / 8) transformer as GGUF, 7.2 GB (Q3_K_M) to 14.9 GB (Q8_0), or 16 GB rotated INT8: another architecture on the same blocks, per its `hyperflow_h3_flash_transformer.py`: a second (endpoint) timestep embedder mixed with gate 0.25, three dropped blocks, 102 low-rank factored linears, its own sigma grids. The GGUF tensor layout was not read |
