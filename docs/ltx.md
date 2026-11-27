# Video with sound: LTX-2.5

`Lightricks/LTX-2.5` is a 22B audio-video diffusion transformer: a prompt (and optionally images)
becomes a video with a 48 kHz stereo soundtrack. oxinfer runs the release's pipelines
(github.com/Lightricks/LTX-2's `ltx_pipelines`: `distilled`, `ti2vid_*` with the dev transformer,
`ti2vid_two_stages_hq`, `a2vid_two_stage`, `retake`, `t2a_one_stage`, the DFR temporal rounds)
from its split files: `src/ltx/`.

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
# the HQ pipeline (res2s sampler, 15 guided steps, the distilled LoRA at 0.25 then 0.5)
./target/release/oxinfer --model Lightricks/LTX-2.5 --transformer dev --distilled-lora --hq -p "..." -o hq.mp4
# audio to video: the video made for a soundtrack, which is kept (its length unless --frames)
./target/release/oxinfer --model Lightricks/LTX-2.5 --audio song.wav -p "A singer on a stage..." -o sing.mp4
# retake: seconds 1.5 to 3 of a video regenerated, the rest kept
./target/release/oxinfer --model Lightricks/LTX-2.5 --video clip.mp4 --retake 1.5:3 -p "..." -o retake.mp4
# text to audio (the dev transformer)
./target/release/oxinfer --model Lightricks/LTX-2.5 --transformer dev --audio-only --frames 121 -p "Rain on a tin roof" -o rain.wav
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
| `--audio FILE`, `--audio-start S`, `--audio-max-duration S` | audio to video ([below](#audio-to-video)) |
| `--video FILE --retake START:END`, `--keep-audio`, `--keep-video` | a retake of seconds `START` to `END` ([below](#retake)) |
| `--audio-only` | text to audio ([below](#text-to-audio)) |
| `--hq`, `--sampler euler\|res2s`, `--distilled-lora-strengths A,B` | the HQ pipeline, its sampler, the distilled LoRA's strength per stage ([below](#the-hq-pipeline-res2s)) |
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
sizes, camera push-in, the music's tone) in 2 s with E2B-it on Metal; the video follows it. As
the reference's `generate(no_repeat_ngram_size=5)`, no 5 tokens appear twice in the sequence, the
system prompt and the user's prompt included (`ChatConfig::no_repeat_ngram`,
`sampler::NoRepeatNgram`: transformers' `NoRepeatNGramLogitsProcessor`, for any engine model).

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

### Audio to video

`--audio FILE` (`audio_input` in the server and the C API, `Params::audio_input`) runs the
release's `a2vid_two_stage` ([`audio_enc.rs`](../src/ltx/audio_enc.rs)): the video is made for a
soundtrack that stays as it is.

- The waveform (any file oxideav decodes; `--audio-start S`, `--audio-max-duration S` pick a
  window, as `decode_audio_from_file`) is resampled to 16 kHz as
  `torchaudio.functional.resample` does (a Hann-windowed sinc of 6 zero crossings, rolloff 0.99,
  in float32), turned into the stereo log-mel spectrogram of `AudioProcessor` (torchaudio's
  `MelSpectrogram`: 1024-point centered frames every 160 samples, reflect padding, magnitudes,
  64 Slaney mel bins up to 8 kHz with Slaney's norm, `log(max(m, 1e-5))`) and encoded by the
  audio VAE's encoder (`AudioEncoder`: causal 2-D convolutions as the decoder's, two residual
  blocks per level, stride-2 convolutions between levels, the means of `conv_out` normalized by
  the per-channel statistics). A mono file is used for both channels (the reference needs two).
- Without `--frames`, the clip lasts as long as the audio (`num_frames_from_audio_duration`:
  `int(seconds fps)`, at most 1024, snapped to `8 k + 1`); with it, the audio is cut to the clip.
  The latents are cut (or zero-padded) to the clip's `audio_latents`.
- Both stages keep those latents frozen (timestep and sigma 0, noised at scale 0 so the noise
  stream is the reference's); with the dev transformer the first stage is guided on the video only
  (the audio guider has the default parameters; the CFG pass sees the prompt's own audio context),
  the second runs the distilled LoRA. The written soundtrack is the source waveform itself (at its
  own rate), cut to the clip (`replace_chunks_audio`).

### Retake

`--video FILE --retake START:END` (`video_input`, `retake_start`, `retake_end`; `Params::retake`)
runs the release's `RetakePipeline`: seconds `START` to `END` of a video are regenerated from the
prompt and the rest kept.

- The source's frames (cut to `8 k + 1`, and to multiples of 32 pixels; the reference refuses
  other sizes) are encoded whole as `video_latent_from_file` does:
  `VideoEncoder.tiled_encode` with `TileSizeConfig.default()` (tiles of 80 frames sharing 24,
  causal: each after the first starts a latent frame earlier; 768 pixels sharing 64 in height and
  width; trapezoidal blending in latent units; `vae::VideoEncoder::encode_tiled`). Its soundtrack
  goes through the audio encoder (above); a source without one gets a new soundtrack.
- `TemporalRegionMask`: a latent frame (its pixel span with the causal first frame) or an audio
  latent whose time span meets the window is noised and denoised, the rest keeps its latents.
- One stage at the source's size and rate: the distilled schedule with plain Euler steps (the
  reference's retake has no ancestral sampler), guided on `--steps` with the dev transformer.
- `--keep-audio` / `--keep-video` (`regenerate_audio` / `regenerate_video` false) keep that
  stream whole: frozen, noised at scale 0 (the reference noises a frozen stream at scale 1 before
  freezing it, so its kept stream would be noise).

Decoding: oxideav's YUV to RGB conversion rounds where FFmpeg's swscale (the reference's PyAV)
truncates: the frames differ by +0.34 of a level on average, which moves the encoded latents by
2 % (rel L2); fed the same frames, the encoding matches to 4.5e-6.

### Text to audio

`--audio-only` (`audio_only`) runs `t2a_one_stage`: the audio stream alone (no video rows, so no
audio-video cross-attention), with the dev transformer guided on the audio (CFG 7, STG on block
28, rescale 0.7, no modality guidance) over `LTX2Scheduler`'s schedule; the length from `--frames`
(`frames` / `seconds` in the server and C API) or the duration head on the audio context alone. The whole decoded soundtrack is
written (`-o FILE.wav`; the server returns the WAV, the C API writes `out_wav`). The transformer
runs on the CPU kernels (the Metal session needs both streams): about 1 s a pass for 5 s of audio.

### The HQ pipeline: res2s

`--hq` (`hq`, `Params::hq`) takes `ti2vid_two_stages_hq`'s settings: the res2s sampler (`--sampler
res2s`, `sampler`) in both stages, 15 guided steps on `LTX2Scheduler`'s schedule shifted for the
first stage's latent tokens, video CFG 3 / rescale 0.45 / modality 3, audio CFG 7 / rescale 1 /
modality 3, no STG, and the distilled LoRA at 0.25 in the first stage and 0.5 in the second
(`--distilled-lora-strengths A,B`); the second stage denoises the soundtrack
again alongside the video, but the first stage's is the one decoded (as in the reference).

`res2s_audio_video_denoising_loop` (`legacy_mode`, bong iterations, eta 0.5): per step the
denoised rows at sigma, a midpoint at `sqrt(sigma next)` by the exponential integrator
(`a21 = c2 phi_1(-h c2)`, `h = log(sigma / next)`), SDE noise there, the anchor refined by
fixed-point iterations while `h < 0.5`, the denoised rows at the midpoint, the step
(`b2 = phi_2(-h) / c2`, `b1 = phi_1(-h) - b2`) and SDE noise again; a last evaluation at 0.0011
replaces the final step to 0. Two transformer evaluations a step. The noise comes from two
generators seeded -1 and 9999 (in each stage), every draw normalized to zero mean and unit
variance; the arithmetic is in float64 as the reference's on the CPU. The LoRA's strength is read
at run time on the CPU; on Metal each strength gets its own side paths (2 x 8.9 GB for the HQ
pipeline's two).


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
- the video VAE encoder (conditioning images, and a retake's whole clip, tile by tile) in the same
  session: the residual blocks as the decoder's, the space-to-depth steps a convolution over the
  input with its first frame repeated (`lv_prep` reading frame `b - 1`) and `lv_s2d` (its outputs
  rearranged plus the input's channel-group means); `conv_in`'s 48 inputs zero-padded to 64;
  25 frames at 384x256 in 0.25 s instead of 6 s on the CPU kernels;
- the vocoder's two BigVGAN generators ([`metal_ltx_voc.rs`](../src/metal_ltx_voc.rs)): 1-D
  convolutions as shifted GEMMs over their dilated taps, the transposed convolutions one GEMM
  per output phase, the anti-aliased SnakeBeta `lb_act` writing the next GEMM's input as two half
  rows, the value and its remainder (two GEMMs into the same outputs: about 21 bits of each
  input instead of half's 11, the precision the soundtrack needs); the STFT log-mel between them
  and the 3x sinc upsampling stay on the CPU (`OXINFER_LTX_VOC_GPU=0`: the CPU path);
- the text encoder (Gemma 4 12B) runs on the GPU prefill plan with every layer's output copied
  out (`OXINFER_LTX_TEXT_GPU=0`: the host path); the int8 ConvRot file's layers become Q8 views
  of their mapped bytes (the row scale repeated per block of 32, `Model::q8_views_of_i8`) and the
  plan rotates their inputs (`Op::Fold` with ConvRot's regular Hadamard transform, `dec_fold`'s
  radix-4 butterflies): a 12-token prompt in 0.33 s instead of 3.3 s.

The latent upsamplers, the audio VAE decoder and encoder and the output heads stay on the CPU
kernels with per-op GPU convolutions and matmuls. `OXINFER_LTX_GPU=0` keeps the transformer on the CPU; the
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
| dev transformer, guided one stage (30 steps x 4 passes), 384x256 | 25 | 73.1 s (stage 64.2 s: 0.53 s a pass) | a red fox trotting through snow under pines |
| t2v 384x256 two stages, `--temporal-rounds 2` (DFR: 1 then 2 then 4 tiles) | 25 -> 97 (96 fps) | 45.6 s | the dog trotting, smooth, no visible tile borders |
| t2v 384x256 two stages, `--temporal-upscale`, 49 frames | 49 -> 97 (48 fps) | 63 s (machine shared) | no jump at the tile border (frame 48) |
| t2v 384x256 one stage | 25 | 13.7 s | text 3.3 s, 8 forwards 4.4 s (0.55 s each), VAE 2.1 s, audio 1.3 s |
| the same on the CPU (`--device cpu`, release build) | 25 | 79 s | 7.8 s a forward: the frames look the same as the GPU's |
| t2v 384x256 one stage, `--transformer nvfp4` | 25 | 15.1 s | 0.54 s a forward on the Metal session (int8: 0.49 s); `--device cpu`: 7.5 s a forward |
| t2v 768x512, two stages (the text encoder on the GPU plan, the vocoder's split inputs) | 121 | 51.2 s | text 1.1 s, stages 11.1 + 20.6 s, VAE 3.3 s, audio 0.7 s |
| t2v 384x256 one stage | 25 | 8.6 s | text 0.9 s, stage 3.9 s, VAE 0.4 s, audio 0.3 s |
| the same with `--enhance-prompt` (E2B-it on Metal, the 5-gram ban) | 25 | 11.3 s | text 4.0 s with the rewrite (a 150-word caption) |
| i2v 768x512 (`--init-image`) | 121 | 54.0 s | the image encoder on the GPU |
| audio to video 768x512 (`--audio`: the 4 s line of a generated news anchor) | 89 | 47.1 s | stages 10.9 + 17.6 s; a woman talking to the camera in a kitchen over the kept soundtrack |
| retake 768x512 of seconds 1 to 2.5 of that clip | 97 | 66.9 s | one stage at the full size: 48.0 s with the whole clip's encoding; frames outside the window kept |
| text to audio, dev transformer, 30 guided steps (`--audio-only`, rain on a tin roof) | 5 s | 107.6 s | the audio stream on the CPU kernels: 1.1 s a pass |
| HQ: dev + `--distilled-lora --hq` 768x512 | 121 | 312.9 s | text 11.7 s, stage 1 187.1 s (15 res2s steps: 93 passes at 384x256x121), stage 2 76.0 s; 65 GB resident (the LoRA's side paths at two strengths) |

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
`first_frame` / `strength`, `last_frame`, `images`, `audio_input` / `audio_start` /
`audio_max_duration`, `video_input` / `retake_start` / `retake_end` / `regenerate_video` /
`regenerate_audio`, `audio_only`, `hq`, `sampler`, `distilled_lora_strengths`; `Params` has
`audio_input`, `retake`, `audio_only`, `sampler`, `distilled_lora_strengths`, and `Params::hq`).
The server's `/v1/videos` takes the same request fields for an LTX-2.5 model (`oxinfer serve
--model Lightricks/LTX-2.5`); its `audio_input` and `video_input` are media of the request (a
`data:` or `http(s)://` URL in JSON, a file part in a multipart body), never server paths; an
`audio_only` job has the WAV alone.

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
| the same on the Metal prefill plan, all 49 states / video and audio features | 4e-5 to 2e-3 / 8.1e-4, 5.9e-4 | from the host path (half GEMM inputs; `ltx_check text`) |
| feature extractor and projections | 1.5e-6 | `FeatureExtractorV2` over oxinfer's 49 hidden states |
| video VAE decoder (3 x 4 x 6 latents) / encoder | 3.5e-6 / 6.0e-6 | `ConvVideoDecoder` / `VideoEncoder` |
| audio decoder (20 latents) / vocoder with BWE | 6.3e-7 / 1.0e-5 | `AudioDecoder`, `VocoderWithBWE` |
| spatial / temporal upsampler | 1.2e-5 / 2.2e-5 | `LatentUpsampler` with the VAE's statistics |
| the guidance passes, 2 blocks (dev int8; STG: video self-attention skipped in block 1, audio in block 0; modality: no AV cross-attention), CPU / Metal | 2.3e-6 / 1.6e-6, 2.2e-6 / 1.5e-6; 9.7e-5 / 1.1e-4, 8.9e-5 / 7.7e-5 | `LTXModel` with a `BatchedPerturbationConfig` (`ref_dit_ptb.py`) |
| guidance arithmetic (video / audio params, masked rescale) | 0 (bit-identical) | `MultiModalGuider.calculate` (`ref_guider.py`, `ltx_check guider`) |
| audio encoder front end: resampling 48 / 44.1 kHz to 16 kHz, log-mel | 6.3e-8, 8.3e-8 / 7.0e-6, 7.6e-6 | `AudioProcessor` (`torchaudio` 2.11's `resample`, `MelSpectrogram`; `ref_audio_enc.py`, `ltx_check audioenc`) |
| audio encoder (from the reference's mel / end to end) | 1.4e-6 / 1.1e-5 | `AudioEncoder`, `encode_audio` |
| whole-clip video encoding, 384x256, 25 frames | 4.5e-6 | `VideoEncoder.tiled_encode(TileSizeConfig.default())` on the same frames (`ref_clipenc.py`, `ltx_check clipenc`) |
| `LTX2Scheduler(30)` | 6e-8 (one float32 ulp) | the dev schedule |
| DiffVAE decoder, 2 x 4 x 6 latents (edge-padded to 3 x 7 x 7) / 4 x 8 x 12 (CPU) | 8.1e-7 / 8.1e-7 | `DiffusionVideoDecoder`, `CHUNKED` blocks, eager neighborhood attention, the reference's noise injected (`ref_diffvae.py`, `ltx_check diffvae`) |
| the same on the Metal blocks | 1.8e-4 / 2.8e-4 | half GEMM inputs and half q / k / v |
| video VAE decoder, whole Metal pass (3 x 4 x 6 latents) | 2.9e-4 | half GEMM inputs; 2.7e-4 from the CPU path at 768x512x121 |
| video VAE encoder, whole Metal pass: an image / a 25-frame clip (tiled) | 4.9e-4 / 4.4e-4 | `VideoEncoder` / `tiled_encode` (half GEMM inputs) |
| text connectors on the Metal session (video / audio) | 1.6e-4 / 7.6e-5 | half GEMM inputs, int8 file |
| vocoder with BWE, whole Metal passes | 1.9e-3 | GEMM inputs as two halves (54.5 dB SNR against the CPU path on 5 s of audio; with single half inputs 1.8e-2, 34.8 dB: the rounding of each of the two 30-convolution generators' inputs; a precise `sin` in the SnakeBeta changes nothing). A news anchor's line generated at 768x512 (97 frames) is transcribed by Whisper large-v3-turbo word for word: "Good evening. Tonight we report on the new bridge across the river." |

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
| audio to video (`A2VidPipelineTwoStage`, the distilled transformer, Euler), 384x256, a 1.04 s WAV at 44.1 kHz (25 frames), CPU | 1.3e-6 / 1.0e-6 | 5.1e-5 / 4.7e-5 (the source's latents) | 8.0e-5 | 1.7e-4 / 4.7e-5 | 67.6 dB | the source's samples |
| retake of 0.4 to 0.8 s (`RetakePipeline`, distilled) of a 384x256, 25-frame H.264 + PCM clip, CPU, the same decoded frames | 4.6e-6 / 1.1e-6 | 4.5e-5 / 2.3e-5 | | | 76.0 dB | 83.1 dB |
| text to audio (`T2AOneStagePipeline`, dev, 4 guided steps of 3 passes), 49 frames (2 s), CPU | / 4.8e-6 | / 1.4e-5 | | | | 52.0 dB |
| HQ (`TI2VidTwoStagesHQPipeline` on the distilled transformer, the LoRA at 0: the res2s loop and its guidance), 384x256, 17 frames, 2 steps, CPU | 1.6e-6 / 2.2e-6 | 3.7e-4 / 1.5e-4 | | | | |
| the same with 4 steps (bong iterations) | | 0.25 / 9.9e-2 | 0.36 | 0.58 / | 17.6 dB | 8.1 dB |

(rel L2 of the latents.) On the CPU the whole pipeline follows the reference's to float rounding,
amplified by the ancestral sampler's steps. On Metal the half-precision GEMM inputs and attention
(1e-4 a pass) grow over the steps into a different sample of the same video: the same scene,
motion and sound, details apart (the dog's legs, `side_distilled.png` in the scratch notes).

The HQ pipeline's guided res2s steps amplify float rounding far more than Euler steps do: in the
reference itself a 1e-5 relative change of a midpoint's input moves that evaluation by 1.8e-4.
Every evaluation of the 4-step run matches the reference's to that sensitivity (the first one
3.3e-5; the first midpoint 7e-6 in, 8.4e-4 out), and over its 8 evaluations the two runs become
different samples of the same clip (the same fox in the same snow, placed elsewhere); with 2 steps
(no bong iterations, 4 evaluations) the first stage stays at 3.7e-4. The midpoints' sigmas are
float64 in the reference, and so are their timestep sinusoids here (`StreamIn::t64`).

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
- The IC-LoRA pipelines (`ic_lora`, `hdr_ic_lora`, `dubit`: their LoRAs are not among the
  release's files) and the guidance factories by sigma.
- The chunked long-clip forms of audio to video (`ChunkConfig`: one chunk here) and the release's
  EXR / HDR inputs of retakes.
- oxideav's YUV to RGB conversion rounds where swscale (the reference's PyAV) truncates, which
  moves a retake's source latents by 2 % (above).
- Text to audio on the GPU: the Metal session needs both streams, so the audio stream alone runs
  on the CPU kernels.
- DFR's detailing IC-LoRA second stage and its spatial epilogue (`spatial_upscalings = 2`): the
  LoRA is not part of the release's files.
- The DiffVAE's tiled decode (the reference's memory-driven tiles and its keyframe-anchored
  decode).
- The NVFP4 file is repacked into ggml's blocks at load (3 s, 18.7 GB of anonymous memory beside
  the mapped file: 34 GB resident at 384x256): the kernels do not read ComfyUI's packing (high
  nibble first, scales in cuBLAS's tiled order) in place.
- The latent upsamplers run per op (their group norms reduce over every frame: no whole pass
  yet); the spatial one of a 768x512x121 run's latents takes about 15 s with its loading
  (`ltx_check upsample`), a good part of the second stage's 20.6 s.
- oxinfer's MP4 muxer writes 24 fps clips with 41 ms frames (1000/41 fps): a retake of one of its
  own videos runs at 24.39 fps.
