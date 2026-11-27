# Video with sound: LTX-2.5

`Lightricks/LTX-2.5` is a 22B audio-video diffusion transformer: a prompt (and optionally images)
becomes a video with a 48 kHz stereo soundtrack. oxinfer runs the release's distilled pipeline
(`ltx_pipelines.distilled`, `DistilledPipeline` of github.com/Lightricks/LTX-2) from its split
files: `src/ltx/`.

```sh
# text to video: 768x512, 121 frames (5 s at 24 fps), the two-stage distilled pipeline
./target/release/oxinfer --model Lightricks/LTX-2.5 \
    -p "A golden retriever running through a sunny meadow, cinematic lighting" \
    --size 768x512 --frames 121 -o dog.mp4
# image to video: the image is the first frame
./target/release/oxinfer --model Lightricks/LTX-2.5 --init-image first.png \
    -p "The camera slowly dollies out as wind moves through the grass" -o clip.mp4
# no --frames: the duration head picks the length from the prompt
./target/release/oxinfer --model Lightricks/LTX-2.5 -p "..." -o clip.mp4
# one stage at the full size (no spatial upsampler; sides multiples of 32)
./target/release/oxinfer --model Lightricks/LTX-2.5 -p "..." --size 384x256 --frames 25 --one-stage -o small.mp4
# the dev transformer, guided (ti2vid_two_stages: 30 guided steps at half the size, then the
# distilled LoRA's 3 steps at the full size; without --distilled-lora one guided stage)
./target/release/oxinfer --model Lightricks/LTX-2.5 --transformer dev --distilled-lora -p "..." -o dev.mp4
```

The release repository is gated (accepting its terms shares the account's contact details), so
oxinfer fetches its files from the public mirror `comfyicu/LTX-2.5` (the same files, checked
against the mirror's LFS hashes): `--model Lightricks/LTX-2.5` and `--model comfyicu/LTX-2.5`
download only what a run reads, 39.7 GB by default:

| part | file | size |
|---|---|---|
| transformer (distilled, ComfyUI int8 ConvRot) | `diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | 21.5 GB |
| text encoder (Gemma 4 12B and the projections, int8 ConvRot) | `text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | 15.4 GB |
| video VAE (convolutional) | `vae/ltx-2.5-video-vae-conv-bf16.safetensors` | 1.45 GB |
| or the diffusion video decoder (DiffVAE, `--vae diff`; its encoder is the convolutional VAE's) | `vae/ltx-2.5-video-vae-bf16.safetensors` | 1.47 GB |
| audio VAE, vocoder, bandwidth extension | `vae/ltx-2.5-audio-vae-bf16.safetensors` | 0.36 GB |
| spatial x2 latent upsampler | `latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | 1.0 GB |
| duration head | `model_patches/ltx-2.5-duration-head-bf16.safetensors` | 3.8 MB |

`--model DIR` takes a folder in the release's layout, `--model FILE` one of its transformer
files (the other parts from the folder above it).

| flag | |
|---|---|
| `--size WxH` | multiples of 64 (the first stage runs at half the size), 32 with `--one-stage`; default 768x512 |
| `--frames N` | rounded down to `8 k + 1`; without it the duration head's prediction (1 to 20 s), or 121 without the head |
| `--fps F` | the frame rate (default 24): the positions the transformer sees and the soundtrack's length |
| `--seed N` | the noise seed (`--noise torch`: PyTorch's CPU generator; the reference draws on its device, so a seed does not reproduce its videos) |
| `--init-image FILE` | the first frame (`--strength F`, default 1: kept as is) |
| `--last-image FILE` | the last frame (a guide image, below) |
| `--image FILE@FRAME[:STRENGTH]` | an image for pixel frame `FRAME` (repeatable; 0 is the first frame) |
| `--one-stage` | the distilled schedule at the full size, no upsampler |
| `--temporal-upscale`, `--temporal-rounds N` | twice (N = 2: four times) the frames and the rate, the release's DFR temporal rounds (fetches the temporal upsampler, 0.26 GB) |
| `--no-audio` | no soundtrack decoded (the audio stream is denoised anyway: the model is joint) |
| `--transformer int8\|bf16\|nvfp4\|dev\|dev-bf16\|FILE` | the transformer file (default the distilled int8 one) |
| `--distilled-lora` | the release's distilled LoRA (rank 450, 8.9 GB) on the dev transformer |
| `--lora FILE[:SCALE]` | an LTX-2.x LoRA (repeatable) |
| `--exact-attention` | the video self-attention in float32 on Metal (slower; default half operands) |
| `--text-bf16` | fetch the bf16 text encoder (26.3 GB) instead of the int8 one |
| `--vae conv\|diff` | the video decoder: the convolutional VAE (default) or the diffusion decoder ([below](#the-diffusion-decoder-diffvae)) |
| `--text-encoder FILE`, `--vae FILE`, `--audio-vae FILE`, `--upsampler FILE`, `--temporal-upsampler FILE`, `--duration-head FILE` | parts from elsewhere |
| `--sigmas a,b,..,0`, `--stage2-sigmas a,..,0` | other schedules than the distilled ones (or the guided `--steps`) |
| `--eta F` | the ancestral sampler's noise (1, the distilled pipeline's for LTX-2.5; 0: plain Euler, the guided pipeline's) |
| `--steps N` | the guided first stage's steps (default 30, `LTX2Scheduler`) |
| `--cfg F`, `--audio-cfg F` | classifier-free guidance of each stream (default 3 and 7; 1: off) |
| `--negative TEXT` | the CFG pass's prompt (default the release's `DEFAULT_NEGATIVE_PROMPT`) |
| `--stg F`, `--audio-stg F`, `--stg-blocks L\|all`, `--audio-stg-blocks L` | spatio-temporal guidance: the self-attention of blocks `L` skipped (default 1 on block 28; 0: off) |
| `--rescale F`, `--audio-rescale F` | rescale toward the prompt pass's spread (default 0.7) |
| `--modality-scale F`, `--audio-modality-scale F` | guidance away from the pass without audio-video cross-attention (default 3; 1: off) |
| `--skip-step N`, `--audio-skip-step N` | guide one step in `N + 1` (default every step) |
| `--guidance-mode auto\|on\|off`, `--no-guidance` | guidance on or off whatever the transformer (auto: the dev transformer's) |
| `--refine-audio` | the guided pipeline's second stage denoises the soundtrack again (default: the first stage's kept, as the reference) |
| `--enhance-prompt`, `--prompt-enhancer MODEL` | rewrite the prompt first with a Gemma 4 instruct model (default the reference's, `google/gemma-4-E2B-it`; [below](#the-prompt-enhancer)) |
| `--image-crf N` | the H.264 CRF conditioning images are re-compressed at first (default the model's: 18 for LTX-2.5; 0: as they are) |
| `--noise oxinfer\|torch\|files:DIR` | the noise source (`files:DIR`: the generator seeded `s` reads `DIR/noise_{s}.f32`, for comparisons with the reference) |
| `--device cpu` | everything on the CPU |
| `-o FILE` | `.mp4` (H.264 and AAC), `.webm` (VP9 and Opus), `.avi`, `.webp` or a PNG sequence with a WAV, as for [MiniMax-H3](video.md) |

## What runs

1. **Text** ([`text.rs`](../src/ltx/text.rs)): `<bos>` and the stripped prompt's tokens (at most
   1024) through the text encoder (transformers' `gemma4_unified` Gemma 4 12B: 48 layers, sliding
   and global attention, `attention_k_eq_v`; oxinfer's Gemma 4); all 49 hidden states RMS-normed
   per token and interleaved (`FeatureExtractorV2`), the two aggregate projections (to 4096 for
   video, 2048 for audio); then the transformer's two connectors (8 blocks each over 1024 rows:
   the prompt's rows, then 128 learned registers repeated; RoPE over the row index; a final RMS
   norm). The reference left-pads the tokens to 1024 with an attention mask: that leaves the
   prompt's rows as computed here (relative RoPE, a 1024-token window).
2. **Stage 1** at half the size: noise (video, then audio: 25 latents a second), eight forwards on
   `[1, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0]` with the ancestral Euler
   sampler (eta 1: an Euler step to `sigma_down`, renoised to the next sigma; its own noise
   generator, seeded `seed + 10000`).
3. **Spatial upsampler** (the latents un-normalized by the VAE's statistics, doubled in height and
   width, normalized again), then **stage 2** at the full size: video and audio latents noised to
   `sigma = 0.909375` and three forwards on `[0.909375, 0.725, 0.421875, 0]` (noise seed
   `seed + 20000`).
4. **Decoders**: the convolutional video VAE ([`vae.rs`](../src/ltx/vae.rs)) or the diffusion
   decoder ([`diffvae.rs`](../src/ltx/diffvae.rs), [below](#the-diffusion-decoder-diffvae)), the audio VAE
   decoder (a stereo 64-bin mel spectrogram), the BigVGAN v2 vocoder (16 kHz) and its bandwidth
   extension to 48 kHz ([`audio.rs`](../src/ltx/audio.rs)); the soundtrack cut to the video's
   length (it is a little shorter: `4 n - 3` mel frames for `n` latents, as in the reference).

The transformer ([`dit.rs`](../src/ltx/dit.rs)): two streams (video 4096 wide, 32 heads of 128;
audio 2048, 32 heads of 64) through 48 blocks of self-attention (split RoPE over time, y and x,
or time), text cross-attention (the context rows with a shift / scale of their own), audio-to-video
and video-to-audio cross-attention (RoPE over time), and a GELU feed-forward, each with AdaLN
shift / scale / gate from the row's timestep; every attention normalizes its queries and keys
over all heads and scales each head by `2 sigmoid(to_gate_logits(x))`. Rows of a latent frame an
image conditions have timestep `(1 - strength) sigma` and keep their clean latents after each
step. The first latent frame's rows get the learned keyframe embedding.

Images are first re-compressed as the reference does (`load_conditioning_still`: cropped to even
sides, one H.264 intra picture, decoded back; the compression the model was trained against),
then resized to cover the canvas (bilinear) and center-cropped, then encoded by the VAE
encoder. Frame 0 replaces the first latent frame, as the reference's `ImageReplace`; an image for
any other frame (`--last-image`, `--image FILE@FRAME`) is appended as guide rows, as the
reference's `VideoConditionByKeyframeIndex`: its clean latents at that frame's time, with the
image's strength, dropped after each stage.

The re-compression's CRF comes from the VAE's `model_version` as in the reference (`detect_params`:
LTX-2.4 and later 18, earlier 33). oxideav's H.264 encoder has constant-QP rate control only, so
the picture is coded at QP `CRF + 5`: the frame QP x264 (`-preset veryfast`, the reference's
PyAV encode) gives a lone intra picture at that CRF (measured: CRF 10 / 18 / 25 / 33 give QP
15 / 23 / 30 / 38 with adaptive quantization off, on any content); x264's adaptive quantization
then moves single macroblocks' QP, which oxideav's encoder does not do. On three test images the
result is as far from the original as x264's (PSNR 39.1 / 44.2 / 26.6 dB against x264's 38.4 /
39.1 / 26.4) and 38.6 / 39.6 / 36.5 dB from x264's own output.

### The prompt enhancer

`--enhance-prompt` (`enhance_prompt: true` or a model name in the server and C API,
`Params::enhance_prompt`) runs the release's prompt enhancer ([`enhance.rs`](../src/ltx/enhance.rs),
`LTXGemmaTextEncoder.enhance_t2v` / `enhance_i2v`): LTX-2.5's own text encoder is not an instruct
model, so as in the reference a separate Gemma 4 instruct model (default E2B-it, any Gemma 4 chat
model oxinfer runs with `--prompt-enhancer`) rewrites the prompt with the release's system prompts
(`gemma4_t2v_system_prompt.txt`, `gemma4_i2v_system_prompt.txt`, Apache-2.0, copied from the LTX-2
repository), greedily, up to 600 tokens; with a first-frame image, the image goes first in the user
turn; then `clean_response`. "a cat plays piano in a jazz bar" became a 160-word caption (shot
sizes, camera push-in, the music's tone) in 2 s with E2B-it on Metal; the video follows it. The
reference also bans repeated 5-grams (`no_repeat_ngram_size`); oxinfer's sampler has no such ban.

### The dev transformer: guidance

A dev transformer (`--transformer dev`, `dev-bf16`, or a `*-dev-*` file) runs the release's
`ti2vid_*` pipelines: the first stage takes `--steps` (30) plain Euler steps on `LTX2Scheduler`'s
schedule, each step guided (`MultiModalGuider`): besides the prompt's pass, a pass on the negative
prompt (CFG), one with the self-attention of block 28 skipped (STG: the value projection straight
to the output projection, no gate) and one without the audio-video cross-attention (modality),
four transformer passes a step; per stream

```text
pred = cond + (cfg - 1) (cond - uncond) + stg (cond - perturbed) + (modality - 1) (cond - isolated)
pred *= rescale std(cond) / std(pred) + 1 - rescale     (std over the rows being denoised)
```

with the release's parameters for LTX-2.3 and later (which LTX-2.5's `model_version` selects):
video CFG 3, audio CFG 7, STG 1 on block 28, rescale 0.7, modality 3. With the distilled LoRA
(`--distilled-lora`) the first stage runs at half the size with the LoRA off, then the spatial
upsampler and the LoRA's 3 unguided steps at the full size, the first stage's soundtrack kept
(`ti2vid_two_stages`; `--refine-audio` denoises it again); without it, one guided stage at the
full size (`ti2vid_one_stage`). The dev transformer with the distilled LoRA and `--one-stage`
runs the distilled schedule instead (the LoRA makes it a distilled model). The distilled LoRA
is switched per stage: on the GPU its side paths exist only in the second stage's view of the
layers.

### Temporal upscaling: the DFR rounds

`--temporal-upscale` (`--temporal-rounds 2`: twice) runs the release's temporal rounds
(`dfr_pipeline.py`, `dfr_stages.run_one_temporal_round`; [`dfr.rs`](../src/ltx/dfr.rs)):

1. The clip is made on a keyframe canvas: its length padded to whole segments of 24 or 32 frames
   (`resolve_canvas`: the one padding least, ties to 32), with a generated keyframe slot at each
   segment border (`VideoGeneratedKeyframeSlots`: a latent frame of rows per keyframe at one pixel
   frame's time, with the learned keyframe embedding, denoised with the clip; in the second stage
   started from the first stage's slots, upsampled).
2. Each round doubles the latent frames with the temporal upsampler (and the rate; the transformer's
   positions use 60 fps above 30, `conditioning_fps`) and splits the canvas at the keyframe borders
   into `2^round` tiles (`TemporalTilePlan`), each denoised alone on `[0.975, 0.909375, 0.725,
   0.421875, 0]` with the ancestral sampler at eta 0.5 (noise seed `seed + 1000 round + tile`):
   the keyframes at its borders as guides at strength 0.95, a new slot in the middle of each
   segment (started from the nearest latent frame), the first stage's soundtrack of its window
   frozen (resampled to the tile's length). A tile after the first starts on the keyframe plane
   before its border, followed by the previous tile's frames up to the border, all pinned
   (`tile_prefix`, `lead_in_carryover`): the tiles meet on known frames, no blending.
3. The tiles are stitched; every keyframe plane carries over to the next round. The canvas'
   padding is cut at the end: `(frames - 1) 2^rounds + 1` frames.

The release's DFR pipeline runs its second stage with a detailing IC-LoRA that is not among the
release's files (its `detailing_lora` argument is required); here the second stage is the
distilled pipeline's (with the slots), and the shipped soundtrack its refined one. Frame-to-frame
change across a tile border is like anywhere else in the clip (384x256, 49 -> 97 frames: mean
absolute difference 6.1 at the border, 4.3 to 6.3 elsewhere).

## The diffusion decoder (DiffVAE)

`--vae diff` (the server's and the C API's option `vae=diff`, `Paths::video_vae` naming the file
in the library) decodes with the release's "diffusion video decoder"
(`ltx_core.model.video_vae.DiffusionVideoDecoder`, `NADiffusionDecoder` in the file's config), a
neighborhood-attention transformer instead of convolutions:

- stages 1 to 4: the latents un-normalized, a linear `conv_in` to 2048 channels, blocks of RMS
  norm, 3D neighborhood attention (64-wide heads, absolute RoPE over each row's frame, row and
  column with 16 / 24 / 24 of the head's dimensions; windows 3x7x7, 3x7x7, 3x5x5, 3x5x5) and
  SwiGLU (4, 6, 4 and 2 blocks at 2048, 1024, 512 and 512 channels), each stage followed by a
  linear pixel-shuffle upsampler (x2 space, x2 time, x2 all, x2 all; the doubled first frame
  dropped): a 256-channel context at a quarter of the pixel size. The last latent frame is
  repeated twice first and those frames cropped before stage 5 (NATTEN's last-frame border);
  latents smaller than 3x7x7 are edge-padded and the padding cropped from the pixels;
- stage 5: Gaussian noise over the 4x4-patchified pixels at timestep 1, a linear projection,
  then 8 blocks with 11x11x11 windows: the context projected and added, AdaLN-modulated
  attention and SwiGLU (one shared timestep modulation plus a table per block); RMS norm, a
  linear layer to the 48 patch channels. The checkpoint predicts the clean pixels in one step.

Windows shift inward at the volume's borders as NATTEN's do. Stage 5 follows the pipelines'
default decode mode (`DiffVAEMode.CHUNKED_EAGER`): along the width its blocks see the volume
edge-replicated by half a window (the repeated columns at their own, out-of-range RoPE positions),
so the outer columns' windows stay centered; the reference's `COMBINED_COMPILE` mode shifts them
instead. The decode is untiled (the reference tiles only when its device's free memory requires
it). The starting noise is drawn from the run's generator after the denoising stages, as the
reference continues its generator into the decoder.

On Metal every block runs on the device ([`metal_diffvae.rs`](../src/metal_diffvae.rs), one command
buffer per block, the rows in device memory for a stage's blocks): the projections on the
matrix-unit GEMMs, q / k / v normed and rotated into half buffers, the attention on `dv_na3d_k*`
(half operands, float32 arithmetic; a SIMD-group per 32 queries of a row, 4 rows per threadgroup
sharing the staged key rows), the projections and the MLP over chunks of 65536 rows. Without
matrix units, the attention alone runs on the GPU (`ltx_na3d_k*`, float32) and the rest per op.
`OXINFER_DIFFVAE_GPU=0` keeps the blocks on the CPU; `OXINFER_DIFFVAE_PROF=1` prints the stages'
times.

The default stays the convolutional VAE: the two decode in about the same time on an M5 Max
(table below), the DiffVAE needs about three times the memory at 768x512x121 (untiled), and on
the clips checked (a face, a chalkboard sign) its frames were not visibly sharper, only a little
softer. The release's own pipelines default to the DiffVAE.

| decode (M5 Max, Metal) | convolutional VAE | DiffVAE |
|---|---|---|
| 384x256, 25 frames (4 x 8 x 12 latents) | 2.1 s | 1.6 s (CPU: 30 s; the reference, torch CPU float32: 21 s) |
| 768x512, 49 frames, in a two-stage run | 10.9 s | 11.1 s |
| 768x512, 121 frames (16 x 16 x 24 latents) | 25.2 s, 13 GB peak | 32.4 s (stage 5: 30.2 s), 38 GB peak |

## On the GPU

With Metal (M5's matrix units) every block of both streams runs as one submission per forward
([`metal_ltx.rs`](../src/metal_ltx.rs), `crate::gpu::dit::LtxSession`): the projections on the
matrix-unit GEMMs (int8 ConvRot rows as Q8 blocks over the same mapped bytes, their inputs rotated
on the GPU; bf16 as stored), video attention on `flash_attn128_mma` (half operands), audio and the
cross-attention between the streams on `flash_attn64` (float32), the norms, modulations, gates
and residuals on the `ltx_*` kernels of `metal_ltx.metal`. The rest, also as whole passes:

- the text connectors: the same session (`LtxSession::connector`: both connectors' 8 blocks in
  one submission, int8 ConvRot layers as Q8 twins, the norms and residuals on the blocks'
  kernels with a zero modulation and unit gates); `OXINFER_LTX_GPU=0` keeps them on the CPU;
- the video VAE decoder ([`metal_ltx_vae.rs`](../src/metal_ltx_vae.rs)): the activations in
  `metal_unet.metal`'s padded layout with the frames as its images (plus images holding the
  repeated first / last frames of the time padding), each 3x3x3 convolution three of
  `un_gemm`'s shifted GEMMs (one per time tap, bf16 weights, half inputs: `PixelNorm` + SiLU'd by
  `lv_prep`, or scaled into half's range), the depth-to-space steps and the final patches on
  `lv_d2s` / `lv_out`; one command buffer per block (`OXINFER_LTX_VAE_GPU=0`: the CPU path);
- the vocoder's two BigVGAN generators ([`metal_ltx_voc.rs`](../src/metal_ltx_voc.rs)): 1-D
  convolutions as shifted GEMMs over their dilated taps, the transposed convolutions one GEMM
  per output phase, the anti-aliased SnakeBeta `lb_act` writing the next GEMM's half input; the
  STFT log-mel between them and the 3x sinc upsampling stay on the CPU
  (`OXINFER_LTX_VOC_GPU=0`: the CPU path);
- the text encoder (Gemma 4 12B) runs on the GPU prefill plan with every layer's output copied
  out when the model has one (`OXINFER_LTX_TEXT_GPU=0`: the host path); the int8 ConvRot file's
  rotated layers have no plan, so it stays on the host path (CPU kernels, per-op GPU matmuls).

The image encoder (one frame per conditioning image, 0.7 s at 768x512), the latent upsamplers,
the audio VAE decoder and the output heads stay on the CPU kernels with per-op GPU convolutions
and matmuls. `OXINFER_LTX_GPU=0` keeps the transformer on the CPU; the
`device=cpu` option of the server and the C API does the same per model. The NVFP4 transformer
runs in the same session: its layers (repacked into ggml's NVFP4 blocks at load) go through
`mma_NVFP4`, which dequantizes the E2M1 codes, the E4M3 scales of 16 values and the tensor-wide
factor into the GEMMs' half tiles (no re-quantization).

## Runs and speed

Apple M5 Max (128 GB), `--profile ci` build, each run alone on the machine
(`tools/bench/with-lock.sh`), the default files (distilled int8 transformer, int8 text encoder),
wall time from start to the written `.mp4`, files mapped (resident set 17 to 22 GB):

| run | frames | time | |
|---|---|---|---|
| t2v 768x512, two stages (8 forwards at 384x256, 3 at 768x512) | 121 | 87.8 s | a golden retriever running through a meadow: sharp frames, coherent motion |
| i2v 768x512 (`--init-image`, a red sports car on a coastal road) | 121 | 81.0 s | the first frame is the image; the car drives off into the sunset, the camera follows |
| t2v 768x512, no `--frames` (the duration head: 4.32 s) | 97 | 69.8 s | a woman in a kitchen saying the prompt's line; Whisper large-v3-turbo transcribes the soundtrack as "Good morning. Today we are going to bake fresh bread together." |
| first + last frame 768x512 (`--init-image car.png --last-image` a 1.67x crop of it) | 49 | 47.7 s | the camera pushes in from the first frame to the last one |
| dev transformer + `--distilled-lora` (1660 layers as side paths), 384x256 one stage | 25 | 56.8 s (19 s of it loading the LoRA; 40 GB resident) | the dog running, softer than the distilled file's |
| t2v 384x256 two stages, `--temporal-rounds 2` (DFR: 1 then 2 then 4 tiles) | 25 -> 97 (96 fps) | 45.6 s | the dog trotting, smooth, no visible tile borders |
| t2v 384x256 two stages, `--temporal-upscale`, 49 frames | 49 -> 97 (48 fps) | 63 s (machine shared) | no jump at the tile border (frame 48) |
| t2v 384x256 one stage | 25 | 13.7 s | text 3.3 s, 8 forwards 4.4 s (0.55 s each), VAE 2.1 s, audio 1.3 s |
| the same on the CPU (`--device cpu`, release build) | 25 | 79 s | 7.8 s a forward: the frames look the same as the GPU's |
| t2v 384x256 one stage, `--transformer nvfp4` | 25 | 15.1 s | 0.54 s a forward on the Metal session (int8: 0.49 s); `--device cpu`: 7.5 s a forward |

The whole GPU passes of the connectors, the VAE decoder and the vocoder, measured in one locked
session against the build before them (same prompt and seed, `| time |` of the CLI):

| run | before | after | parts before -> after |
|---|---|---|---|
| t2v 768x512x121, two stages | 80.4 s | 53.9 s | text + connectors 3.3 -> 1.4 s, VAE 23.1 -> 3.3 s, audio 4.4 -> 0.6 s (transformer stages unchanged: 35 s) |
| t2v 384x256x25, one stage | 23.2 s | 10.3 s | text 5.8 -> 1.4 s, VAE 3.0 -> 0.4 s, audio 1.2 -> 0.4 s |

Alone: the VAE decode of 768x512x121 takes 23.6 s per op and 3.2 s whole (`ltx_check vaebench`),
both connectors 2.3 s and 0.39 s, the vocoder with its bandwidth extension (5 s of audio) 4.1 s
and 0.29 s (`ltx_check audiobench`).

## Library, C API, server

`oxinfer::ltx::Pipeline::load(Paths)` and `generate(&Params, progress)` (a
`minimax_h3::video::Video`: frames and the soundtrack); `oxinfer::ltx::api::load_with_options`
takes a repository id, folder or file with the server's options. The C API's `oxi_video_load` /
`oxi_video_generate` load LTX-2.5 for any of its names (options `transformer`, `text_encoder`,
`vae` (`conv`, `diff` or a file), `audio_vae`, `upsampler`, `temporal_upsampler`, `duration_head`, `text_bf16`, `temporal`,
`distilled_lora`, `lora`, `device`; parameters `prompt`, `size`, `frames` / `seconds`, `fps`,
`seed`, `one_stage`, `temporal_upscale`, `audio`, `eta`, `sigmas`, `stage2_sigmas`, `noise`,
`first_frame` / `strength`, `last_frame`, `images`). The server's `/v1/videos` takes the same
request fields for an LTX-2.5 model (`oxinfer serve --model Lightricks/LTX-2.5`).

## Verification

Against the release's code (`ltx-core` at github.com/Lightricks/LTX-2 c3a7efb, CPU, float32, on
the int8 weights dequantized as ComfyUI's `comfy_kitchen` does, fixed random inputs; the scripts
are `ref_*.py` beside `examples/ltx_check.rs`'s checks):

| part | rel L2 | |
|---|---|---|
| transformer, 2 blocks, both streams, i2v-style mixed timesteps (CPU) | 2.2e-6 / 1.6e-6 | video / audio velocities |
| the same on the Metal session | 8.8e-5 / 7.5e-5 | half-precision GEMM inputs and attention |
| text connectors (video / audio) | 2.4e-6 / 2.4e-6 | `EmbeddingsProcessor.create_embeddings` with 20 valid tokens |
| text encoder, first 6 layers (int8 ConvRot) | 2.7e-6 to 4.8e-6 | transformers' `Gemma4UnifiedTextModel`, same tokens |
| feature extractor and projections | 1.5e-6 | `FeatureExtractorV2` over oxinfer's 49 hidden states |
| video VAE decoder (3 x 4 x 6 latents) / encoder | 3.5e-6 / 6.0e-6 | `ConvVideoDecoder` / `VideoEncoder` |
| audio decoder (20 latents) / vocoder with BWE | 6.3e-7 / 1.0e-5 | `AudioDecoder`, `VocoderWithBWE` |
| spatial / temporal upsampler | 1.2e-5 / 2.2e-5 | `LatentUpsampler` with the VAE's statistics |
| the guidance passes, 2 blocks (dev int8; STG: video self-attention skipped in block 1, audio in block 0; modality: no AV cross-attention), CPU / Metal | 2.3e-6 / 1.6e-6, 2.2e-6 / 1.5e-6; 9.7e-5 / 1.1e-4, 8.9e-5 / 7.7e-5 | `LTXModel` with a `BatchedPerturbationConfig` (`ref_dit_ptb.py`) |
| guidance arithmetic (video / audio params, masked rescale) | 0 (bit-identical) | `MultiModalGuider.calculate` (`ref_guider.py`, `ltx_check guider`) |
| `LTX2Scheduler(30)` | 6e-8 (one float32 ulp) | the dev schedule |
| DiffVAE decoder, 2 x 4 x 6 latents (edge-padded to 3 x 7 x 7) / 4 x 8 x 12 (CPU) | 8.1e-7 / 8.1e-7 | `DiffusionVideoDecoder`, `CHUNKED` blocks, eager neighborhood attention, the reference's noise injected (`ref_diffvae.py`, `ltx_check diffvae`) |
| the same on the Metal blocks | 1.8e-4 / 2.8e-4 | half GEMM inputs and half q / k / v |
| video VAE decoder, whole Metal pass (3 x 4 x 6 latents) | 2.9e-4 | half GEMM inputs; 2.7e-4 from the CPU path at 768x512x121 |
| text connectors on the Metal session (video / audio) | 1.6e-4 / 7.6e-5 | half GEMM inputs, int8 file |
| vocoder with BWE, whole Metal passes | 1.6e-2 | half GEMM inputs through two 30-convolution generators (35 dB SNR against the CPU path on 5 s of audio; the soundtracks of a run have the same levels) |

Positions (video and audio) are bit-identical to the reference's.

### The whole pipeline

`ref_pipeline.py` runs the release's own pipeline classes (`DistilledPipeline`,
`TI2VidOneStagePipeline`) on the CPU in float32 on the same int8 files (their builders' load
replaced: int8 ConvRot layers keep their int8 rows and multiply the Hadamard-rotated input), with
the noise read from files: the generator seeded `s` reads `noise_{s}.f32`, in both programs
(`--noise files:DIR`; oxinfer's `OXINFER_LTX_DUMP=DIR` writes its intermediate states). Text,
transformer passes, sampler, guidance, upsampler, decoders: all of it compared at once.

| run | text context (video / audio) | stage 1 video / audio | upsampled | stage 2 video / audio | frames | soundtrack |
|---|---|---|---|---|---|---|
| distilled, 384x256, 25 frames, two stages (8 + 3 steps, ancestral), CPU | 1.1e-6 / 1.4e-6 | 3.9e-4 / 2.7e-5 | 5.9e-4 | 2.7e-3 / 1.8e-4 | PSNR 57.8 dB | SNR 61.3 dB |
| the same on Metal | 8.4e-5 / 1.4e-6 | 3.4e-2 / 1.6e-3 | 5.1e-2 | 0.16 / 1.8e-2 | 27.6 dB | 19.7 dB |
| dev, guided, one stage, 256x160, 17 frames, 6 steps (4 passes each), CPU | 1.1e-6 / 4.8e-6 | 6.1e-4 / 1.6e-4 | | | 65.6 dB | 54.7 dB |

(rel L2 of the latents.) On the CPU the whole pipeline follows the reference's to float rounding,
amplified by the ancestral sampler's steps. On Metal the half-precision GEMM inputs and attention
(1e-4 a pass) grow over the steps into a different sample of the same video: the same scene,
motion and sound, details apart (the dog's legs, `side_distilled.png` in the scratch notes).

The other transformer files, on one-block copies (block 0 and connector block 0 of the real files,
fetched by HTTP ranges) against the reference on bf16 weights: the bf16 file 6.7e-7 / 8.7e-7 on
the CPU, 5.1e-5 / 1.0e-4 on Metal; the int8 file is 7e-4 / 1.2e-3 from the bf16 reference (its
quantization) and the NVFP4 file 2.7e-2 / 1.6e-2 (its rows correlate 0.9956 with the bf16 ones).
The NVFP4 file on the Metal session is 5.9e-5 / 1.0e-4 from the same file on the CPU (the
precision of the int8 and bf16 sessions; `ltx_check dit` with `LTX_SAVE` and `LTX_DEVICE=metal`).

## Not done

- x264's adaptive quantization of the re-compressed conditioning images (oxideav's H.264 encoder
  has constant QP only: the frame QP x264 picks is used).
- The guided pipelines' `skip_step` with only one stream skipping a step runs both streams (the
  reference leaves the skipped stream out of that pass); off by default.
- The prompt enhancer's `no_repeat_ngram_size = 5` (greedy decoding without the ban).
- Audio-to-video (`a2vid_two_stage`: needs the audio VAE's encoder and mel front end), retake
  (video-to-video regeneration of a time window: needs whole-clip VAE encoding of a decoded video),
  the IC-LoRA pipelines (`ic_lora`, `hdr_ic_lora`, `dubit`: their LoRAs are not among the release's
  files) and text-to-audio (`t2a_one_stage`).
- `ti2vid_two_stages_hq` (the res2s sampler) and the guidance factories by sigma.
- DFR's detailing IC-LoRA second stage and its spatial epilogue (`spatial_upscalings = 2`): the
  LoRA is not part of the release's files.
- The DiffVAE's tiled decode (the reference's memory-driven tiles and its keyframe-anchored
  decode).
- The NVFP4 file is repacked into ggml's blocks at load (3 s, 18.7 GB of anonymous memory beside
  the mapped file: 34 GB resident at 384x256): the kernels do not read ComfyUI's packing (high
  nibble first, scales in cuBLAS's tiled order) in place.
- The int8 ConvRot text encoder has no whole GPU pass (the prefill plans have no input
  rotation); the image encoder and the upsamplers run per op.
