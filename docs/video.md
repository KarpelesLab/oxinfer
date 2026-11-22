# Video generation: MiniMax-H3

`MiniMaxAI/MiniMax-H3` generates video with a stereo soundtrack from text (`t2va`), from a first
and / or last frame (`fl2va`) or from reference images, videos and audio (`ref2va`). oxinfer runs
all three: `src/minimax_h3/video.rs` (and `ref2va.rs` for references) follows diffusers' modular `minimax_h3` pipeline
(`MiniMaxH3ModularPipeline`), with the DiT and the ViT VAE decoder it shares with
[MiniMax-H3-Image](models.md), and `src/minimax_h3/audio_vae.rs` decodes the soundtrack.

```sh
# text to video: 124 frames (5.2 s at 24 fps) at 1344x768, 49 forwards
./target/release/oxinfer --model /path/to/MiniMax-H3 --device cpu \
    -p "integrated_multimodal_description: [Shot 1] A red fox trots across a snowy clearing ..." -o fox.mp4
# smaller and shorter
./target/release/oxinfer --model /path/to/MiniMax-H3 --device cpu -p "..." \
    --size 448x256 --frames 124 --steps 50 --seed 1 -o fox.mp4
# first frame (and last frame) to video; the canvas takes the first image's aspect ratio
./target/release/oxinfer --model /path/to/MiniMax-H3 --device cpu -p "..." \
    --init-image first.png --last-image last.png -o clip.mp4
# references to video (the release's Ref2VA/): in the order the prompt names them
./target/release/oxinfer --model /path/to/MiniMax-H3/Ref2VA --device cpu -p "subject_definitions: ..." \
    --ref subject.png --ref motion.mp4 --ref voice.wav -o clip.mp4
```

`--model` is a folder in the repository's diffusers layout: `model_index.json`, `transformer/`,
`vae/`, `audio_vae/`, `text_encoder/` (or `OXINFER_H3_TEXT_ENCODER`), `tokenizer/`, and the two
`scheduler*/scheduler_config.json` (141 GB of the repository's 498 GB:
`hf download MiniMaxAI/MiniMax-H3 --include model_index.json "transformer/*" "vae/*" "audio_vae/*" "text_encoder/*" "tokenizer/*" "scheduler/*" "audio_scheduler/*"`).
Without `audio_vae/` (0.6 GB) and without `--audio-vae` the soundtrack is skipped and a message
says so. Passing the repository id itself has not been tried.

| flag | |
|---|---|
| `--frames N` | frames at 24 fps, rounded up to `17 n + 5` (default 124). The release is for 5 to 15 s (124 to 362); fewer frames run with a warning |
| `--size WxH` | multiples of 32; default 16:9 with a 768 short edge (1344x768), or the first keyframe's aspect ratio under the same pixel budget |
| `--steps N` | points of the sigma grid, as diffusers' `num_inference_steps`: `N - 1` forwards (default 50) |
| `--seed N` | noise seed (oxinfer's generator: not torch's, so a seed does not reproduce a diffusers video) |
| `--init-image F`, `--last-image F` | the first / last frame |
| `--ref F` (repeated) | `Ref2VA/` only: a reference, in the order the prompt's labels number them: an image (`.png`, `.jpg`, ...), an audio clip (`.wav`, `.mp3`, `.flac`, `.ogg`, `.opus`, `.m4a`, `.aac`) or a video (anything else, or a folder of frames at 24 fps) with its soundtrack when it has one. At most 9 images, 3 videos and 3 audio clips, 12 in all; audio cannot be the only kind |
| `--ref-image-edge N` | the short edge image references are encoded at (default 2048, the release's; a smaller one makes fewer rows for a quick test) |
| `-o FILE` | `.mp4` (the default name is `video.mp4`): H.264 and the soundtrack as AAC-LC in MP4; `.webm`: VP9 and Opus (48 kHz) in WebM; `.avi`: Motion JPEG and 16-bit PCM in AVI; `.webp`: animated WebP (lossless only, about 3 s per 320x288 frame to encode) and `FILE.wav`; anything else: a PNG sequence `FILE_0000.png`, ... and `FILE.wav` |

The files are written by oxideav (`minimax_h3::encode::encode_video`; the server and the C API
take the same choice): H.264 by oxideav-h264 (Main profile with CABAC, the level that fits the
size and rate, e.g. 2.1 at 448x256, constant QP 19, an IDR picture every second), AAC-LC by
oxideav-aac (64 kb/s per channel) and MP4 by oxideav-mp4 (0.0.11 or later, which strips the AAC
encoder's ADTS headers; the `esds` gets the AAC-LC `AudioSpecificConfig` from oxinfer, since
oxideav-aac 0.1.7 does not report it); VP9 by oxideav-vp9, Opus by oxideav-opus (the 32 kHz
soundtrack is resampled to 48 kHz) and WebM by oxideav-mkv, with the Segment's `Duration`. Each
second of video (an IDR picture or VP9 keyframe and what follows) is encoded by its own encoder
on its own thread and the packets concatenated: oxideav-vp9 has no keyframe interval of its own,
so this is also what makes a WebM seekable. Both play in Chrome with their sound (a `<video>`
element: the MP4, and the WebM, whose duration is known at `loadedmetadata`); Main-profile
H.264 plays in every browser. Encoding one 448x256 frame on one core here (an M5 Max) took about
0.09 s for H.264 (0.18 s with CAVLC / Baseline, which oxideav-h264 0.1.12 decides more slowly;
its unreleased master is 14x faster) and 0.16 s for VP9; with the chunks in parallel a
124-frame 448x256 video takes 4.0 s as MP4 and 8.9 s as WebM (`cargo test --release --lib
encode_speed -- --ignored --nocapture`; 22.8 s and 33.3 s with CAVLC and one encoder). Motion
JPEG is two orders of magnitude faster. Against libx264 at the same PSNR the H.264 files are 25
to 50 % larger, against libvpx the VP9 ones about 70 %. Without the `media` feature only the PNG
sequence is available.

oxideav-aac 0.1.7 decodes ADTS or LOAS only, not the bare access units of an MP4 / Matroska
track, so oxinfer gives those an ADTS header built from the track's `AudioSpecificConfig` when it
reads them (`.m4a` references, a reference video's soundtrack: `Media::decode_audio`).

The soundtrack is stereo at 32 kHz, 800 samples per audio latent (a 124-frame request: 207
latents, 5.175 s for 5.167 s of frames; nothing is trimmed). In an `.mp4` it is an AAC track, in
a `.webm` an Opus track, in an `.avi` a stream of 16-bit PCM, one chunk after each frame. Animated WebP
has no audio and a PNG sequence is only frames: both get a 16-bit stereo WAV next to them, named
as the output without its extension (`-o fox.webp` and `-o fox.png` both write `fox.wav`).

`OXINFER_H3_PROMPT_EMBEDS=FILE` keeps the text encoder's output of a text-only prompt in FILE
(written when missing, read when present; the run stops after writing it when the transformer's
weights are absent), so the 63 GB text encoder and the 66 GB transformer need not be on disk
together. `OXINFER_H3_LATENTS=FILE` keeps the denoised video and audio rows the same way: a second
run with the same size and frame count only decodes them (the two VAEs, no forwards).
`OXINFER_H3_DUMP=DIR` keeps what every forward reads and returns as float32 files in DIR, for a
comparison with another implementation on the same inputs (see
[the real files against torch and ComfyUI](#what-was-checked-1)).


The server runs it too, as asynchronous jobs in the shape of OpenAI's videos API (`/v1/videos`:
see [video generation](server.md#video-generation)), and the web GUI has a *Video* page that
follows a job and plays the result; in C it is `oxi_video_load` / `oxi_video_generate`.

## The release

| component | folder | size | |
|---|---|---|---|
| transformer (`t2va`, `fl2va`) | `transformer/` | 66.3 GB, 33.1B parameters, bf16 (patch, timestep and output layers float32) | 50 blocks, width 5376, 56 heads of 128, SwiGLU 14336, text width 5120; about 13B of the parameters are the AdaLN projections (`6 x 5376 x 3` outputs per block from a 2688-wide timestep embedding) |
| transformer (`ref2va`) | `transformer_ref/` | 66.3 GB | same architecture, other weights; oxinfer reads the copy in `Ref2VA/transformer/` |
| video VAE | `vae/` | 10.4 GB, float32 | `f16t4d24`: a causal 3D CNN encoder (128 to 1024 channels, 16x in space, 4x in time, 24 latent channels) and a 36-layer ViT decoder (width 2048, 32 heads, 3-axis RoPE) that emits a 4x16x16 pixel block per latent voxel |
| audio VAE | `audio_vae/` | 0.6 GB, float32 | mono, 32 kHz, 40 latents of 32 channels per second; a stereo soundtrack is two latent sequences. Decoder (0.26 GB of the file): a 1x1 convolution to 2048 channels and BigVGAN, seven transposed convolutions (x5, x5, x2 five times; 1024 to 8 channels) each followed by three AMP blocks with anti-aliased SnakeBeta activations, clamped to [-1, 1]. Encoder (`AudioEncoder`, only `ref2va`'s reference audio goes through it): a DAC waveform encoder (strides 2, 4, 4, 5, 5; Snake activations) to 2048 channels, a causal attention projection to 32 (8 heads of 256, averaged over the heads and pooled by 8 instead of concatenated) and `mean_proj`; against diffusers on 5.2 s of a synthetic stereo signal the rows are within 6.4e-5 (mean \|v\| 0.50), 1.0 s on the CPU (`tools/minimax_h3_audio_tiny.py OUT audio_vae signal`, `examples/h3_audio_check.rs`) |
| text encoder | `text_encoder/` | 63.4 GB, bf16 | Qwen3-VL-32B (64 layers, width 5120); `hidden_states[50]` conditions the transformer |
| schedulers | `scheduler/`, `audio_scheduler/` | | rectified flow, shift 12 (video) and 3 (audio) |

`FL2VA/` and `Ref2VA/` hold the same checkpoints in MiniMax's original layout (their own `.py`
modules). `--model .../FL2VA` runs too: the transformer's tensors there are the ones `Dit::load`
already reads for H3-Image (with `audio_patch_proj` / `final_layer.audio_out`), the video VAE's
`video_vae/source/model.safetensors` has the original names the VAE loaders read (`x_embedder`,
per-head `to_qkv`, `ff.w1` as `[gate; up]`, `encoder.down.I.block.J`), and its configuration
(`video_vae/config.json` for the latent statistics, clip length and tiles,
`source/config.json` for the architecture) is mapped to diffusers' keys as
`scripts/convert_minimax_h3_to_diffusers.py` does (identical to the root `vae/config.json`); the
audio VAE's `config.json` has the latent statistics and `metadata.json` the decoder rates; the
shifts come from `model_index.json` (`sigma_shift_scales`). `--model .../Ref2VA` is the
`ref2va` checkpoint in the same layout (only its `transformer/` differs from `FL2VA/`'s; the VAEs
and the text encoder are the same files) and takes `--ref`; text-only and keyframe requests on it,
or references on `FL2VA/`, are refused. The released weights are guidance-distilled (one forward per step, no negative prompt)
but not step-distilled: diffusers' default is 50 grid points.

What a request does (`video.rs` has the details):

- the prompt verbatim, no chat template and no token around it, through the text encoder; each
  keyframe prepends `"<Picture i>: "` and its Qwen3-VL vision block, whose rows are tagged as video
  for the AdaLN. The tokenizer is Qwen's with the seven tokens MiniMax's `tokenizer_config.json`
  adds (`<d>`, `</d>`, `<|cutoff|>`, `<|lyrics_start|>`, `<|lyrics_end|>`, `<|caption_start|>`,
  `<|caption_end|>`: ids 151669 to 151675, whatever file the vocabulary comes from);
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
2 channels. The MP4 and WebM files are decoded back by oxideav in unit tests
(`minimax_h3::encode`): every frame, at 43 dB PSNR for an IDR picture and 31 dB for the P
pictures of a synthetic clip with sharp moving edges (libx264 at the same QP and profile gives
the same PSNR at less than half the size), and the tiny random MiniMax-H3 release's noise at
about 19 dB (noise is the hardest content; libx264 does no better); ffmpeg decodes both files
cleanly to the same frames and ffprobe reads `h264` Baseline + `aac` LC, and `vp9` + `opus`.

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
  run predates the audio decoder: its audio latents were dropped. The runs with a soundtrack are
  the ones on the smaller files, [below](#real-runs).

Not checked or not done:

- no end-to-end comparison with diffusers on the real weights (see above for what was run);
- the prompt path with keyframes (label and vision block through `QwenVl`) ran only on a tiny
  random Qwen3-VL, without a numeric comparison; keyframes are resized with oxinfer's Lanczos, not
  compared with PIL's; the posterior and all noise come from oxinfer's generator;
- the soundtracks generated so far ([real runs](#real-runs)) were analysed and transcribed, not
  listened to; the one from the release's own bf16 transformer at 50 steps (below) was only
  measured; the audio VAE ran on the CPU only;
- the character-swap LoRA; GPU execution of the bf16 transformer (the 66 GB of it run on the CPU
  from mapped weights; the FP8 file runs [on CUDA](#on-cuda)).

The release's own bf16 weights from `FL2VA/` (192 cores, 251 GB RAM, CPU only): the fox prompt
above in the `integrated_multimodal_description: [Shot 1] ...` form with an `Audio:` sentence,
`--size 448x256 --frames 22 --steps 8 --seed 1`: 917 rows, 7 forwards in 236 s (34 s each, with a
compile sharing the cores), VAE 54 s, audio VAE 1.8 s. The frames show the scene (a red fox
trotting across the snow in golden backlight, its breath steaming, pines behind), sharp and
coherent across the clip; ffprobe reads `mjpeg` 448x256, 22 frames and `pcm_s16le` 32 kHz stereo,
29,600 samples. Not compared with a reference.

The same prompt at the release's defaults, `--frames 124 --seed 1` with no `--steps` (49 forwards,
shifts 12 / 3 from `model_index.json`): 4144 video + 414 audio + 59 text rows, 49 forwards in
5001 s (102 s each, 81 s when the machine was otherwise idle; attention, with the blocked kernel,
is 467 s of it), VAE 284 s, audio VAE 1.2 s, 88 min in all. Frames 0, 40, 80 and 123 show the same
fox trotting left to right across the snow under the sunrise behind the pines, sharp, with no
drift of the scene or of the animal. ffprobe reads `mjpeg` 448x256, 124 frames (5.17 s) and
`pcm_s16le` 32 kHz stereo, 165,600 samples; the soundtrack is quiet (-41.7 dB mean, -27.1 dB
peak, both channels within 0.7 dB) and was not listened to or compared with a reference.

## References (`ref2va`)

`src/minimax_h3/ref2va.rs` follows diffusers' `MiniMaxH3Ref2VA*` steps (the module docs list
them): references normalized onto 24 fps, their canvas and 32 kHz stereo; the presentation
(`"<Audio j>: "`, `"<Picture i>: "` and a vision block, `"<Video k>: "` and a timestamped vision
block per pair of the frames sampled at 2 fps, through Qwen3-VL's video processor), then the
prompt; images and videos through the video VAE's encoder like keyframes, soundtracks through the
audio VAE's encoder (`AudioEncoder`, the posterior mean); the packed sequence
`[text | reference blocks | target audio | target video]` on the shared audio / video rotary
clock; reference video rows held at `t = 0.999`, reference audio rows clean at `t = 1`. Prompts
follow the release's `docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md` (`subject_definitions`,
`summary`, `retention_analysis`, `detailed_description`, `overall_soundscape`,
`non_diegetic_music`).

What was checked:

- the packed layout (positions, tags, row order, reference row counts) equals diffusers'
  `build_ref2va_packed_sequence` exactly on a request with an image, a video with sound, an audio
  clip and a second image (`tools/minimax_h3_ref2va_layout.py`, `examples/h3_ref2va_check.rs`);
- the audio VAE's encoder: within 7e-7 of diffusers on a tiny random VAE, 6.4e-5 on the release's
  weights (above);
- unit tests: the 2 fps sampling and its `"<0.2 seconds>"` labels (half to even, as Python
  formats them), the frame-rate resampling (30 fps to 24 drops one frame in five);
- one run on the release's bf16 `Ref2VA/` (CPU, 192 cores): the fox frame of the 124-frame clip
  above as `<Picture 1>` (`--ref-image-edge 512`: 896x512, 448 rows) and a synthetic 6 s
  piano-like melody, stereo at 44.1 kHz, as `<Audio 1>` with `fully_copy` as the score;
  `--size 448x256 --frames 124 --steps 9 --seed 1`: 898 text rows, 448 + 414 reference rows, 6318
  rows in all; text encoder 27 s, references 15 s, 8 forwards in 967 s, VAE 73 s. The frames show
  the reference's fox trotting left to right across its snowy clearing, leaving the frame at the
  end, with a blue blotch in the trees in every frame (8 forwards of a 50-step model and a
  reduced reference size; not investigated). The soundtrack copies the melody: the dominant
  pitch of every note checked is the reference's (261, 523, 437, 347 Hz), the 10 ms loudness
  envelopes correlate at 0.71 and the waveforms at 0.36 (left) and 0.29 (right); RMS 0.084 for
  0.109.

Not checked: video references on real weights (each is put on the 768-short-edge canvas, which
makes tens of thousands of rows on the CPU), image references at their full 2048 short edge,
`QwenVl::encode_video` against transformers, the presentation's hidden states against diffusers.

## Smaller files and turbo LoRAs

The official release needs 140 GB of downloads and 49 forwards. The repackaged files of
`Comfy-Org/MiniMax-H3`, the turbo LoRAs of `lightx2v/Minimax-h3-Turbo` and a GGUF of the text
encoder bring that to 39 to 63 GB and 4 or 8 forwards. Their layouts were first read from the
safetensors headers and from sampled rows (HTTP range requests), the quantized formats from the
source of ComfyUI and of its `comfy_kitchen` package, and the loaders checked on tiny random
checkpoints in the same forms; then the pruned FP8 transformer, the fp16 video VAE, the fp32 audio
VAE, the 4-step 768p and the 8-step LoRAs and the Q8_0 and Q4_K_M text encoders were downloaded
and run ([real runs](#real-runs)). The other files in the tables below (the int8 ConvRot and W6A8
files, the single-file text encoders, the bf16 transformers) were not.

```sh
# 4 forwards: the pruned FP8 transformer (21.0 GB), the fp16 VAE (5.2 GB), the 4-step turbo LoRA
# (2.0 GB) and Qwen3-VL-32B-Instruct as a GGUF text encoder (Q8_0: 34.8 GB, Q4_K_M: 19.8 GB)
./target/release/oxinfer --device cpu \
    --model diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors \
    --vae vae/minimax_h3_video_vae_fp16.safetensors \
    --audio-vae vae/minimax_h3_audio_vae_fp32.safetensors \
    --lora loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors \
    --text-encoder Qwen3-VL-32B-Instruct-Q8_0.gguf \
    -p "integrated_multimodal_description: [Shot 1] ..." -o fox.mp4
# the same in ComfyUI's int8 files: transformer 21.0 GB, VAE 2.8 GB, and the text encoder as
# NVFP4 (15.7 GB) with the release's tokenizer/tokenizer.json in a tokenizer/ folder beside it
./target/release/oxinfer --device cpu \
    --model diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
    --vae vae/minimax_h3_video_vae_int8_convrot.safetensors \
    --lora loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors \
    --text-encoder text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
    -p "integrated_multimodal_description: [Shot 1] ..." -o fox.mp4
# the release folder, 8 forwards
./target/release/oxinfer --model /path/to/MiniMax-H3 --device cpu \
    --lora minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors -p "..." -o fox.mp4
```

| flag | |
|---|---|
| `--model FILE` | a single-file transformer in ComfyUI's layout instead of the release folder; `--vae` is then required, the shifts are 12 and 3 (there is no scheduler folder), and there is a soundtrack only with `--audio-vae` |
| `--transformer FILE\|DIR`, `--vae FILE\|DIR` | one part of a release folder from another place |
| `--audio-vae FILE\|DIR` | the audio VAE: ComfyUI's `vae/minimax_h3_audio_vae_fp32.safetensors` or the release's `audio_vae/` folder (0.6 GB either way) |
| `--text-encoder DIR\|FILE.gguf\|FILE.safetensors` | a Qwen3-VL-32B-Instruct folder, or its language model as GGUF, or one of ComfyUI's single files (`text_encoders/qwen3vl_32b_minimax_h3_{bf16,int8_convrot,nvfp4_awq}.safetensors`); a file is for text-only prompts (no `--init-image` / `--last-image`), and a safetensors file needs the release's `tokenizer.json`: in `--model`'s `tokenizer/`, in a `tokenizer/` folder beside the file or beside the file itself |
| `--quant q8` | the text encoder's safetensors weights quantized at load (memory, not download; `q4` is taken as `q8`) |
| `--lora FILE[:SCALE]` | LoRA files on the transformer, repeatable. A file named `*turbo_Nstep*` sets `--steps N + 1` and the shifts: 6 / 3 for the `768p` files, 12 / 3 for the others |
| `--shift F`, `--audio-shift F` | the sigma shifts of the video and audio schedules |

Each of these takes a local path or `org/name/path/in/repo.safetensors` (`.gguf`), fetched like
`--lora` files elsewhere; fetching was not tried for these repositories (the files of the real
runs were downloaded with curl and given as paths).

### What the files are

Read from the files' headers and, where rows are quoted, from the same rows of several files.

| file | size | contents |
|---|---|---|
| `diffusion_models/minimax_h3_fl2va_bf16.safetensors` | 66.3 GB | the 535 tensors of MiniMax's own layout (`FL2VA/transformer`) with the same names, but `qkv_proj` as `[q; k; v]` where MiniMax's is `[q k v]` per head (row 0 of `to_k` of diffusers' block 0 is row 7168 of ComfyUI's `qkv_proj` and row 128 of MiniMax's, bit for bit); `fc1` is `[gate; value]` in both. The configuration is in the header |
| `..._pruned_bf16.safetensors` | 40.2 GB | the same matrices (rows 0 and 7168 of block 0's `qkv_proj` are identical) without the time embedder and the 26 GB of AdaLN weights: `adaln_t_table` `[1025, 8]` and float16 `[.., 8]` projections, see below |
| `..._pruned_fp8_scaled.safetensors` | 21.0 GB | the pruned form with the four matrices of each of the 50 blocks as `float8_e4m3fn` times a scalar `weight_scale` (the refiner's stay bf16). Each also has a `comfy_quant` entry and, but for `fc2` (`"full_precision_matrix_mult": true`), a scalar `input_scale`: ComfyUI can round the layer's input to FP8 with it; oxinfer multiplies the float32 input by the decoded weights and does not read it. No byte of the sampled matrices (blocks 0, 1, 25, 49) is a NaN code |
| `..._int8_convrot.safetensors`, `..._pruned_int8_convrot.safetensors` | 34.0 / 21.0 GB | the bf16 / pruned bf16 file with the four matrices of each of the 50 blocks as `int8_tensorwise` with ConvRot (below); the unpruned file's AdaLN projections too, in groups of 64. The refiner's stay bf16. The format ComfyUI's README prefers over FP8 |
| `..._pruned_w6a8.safetensors` | 16.0 GB | the pruned form with the blocks' four matrices and the video head as `w6a8_int8`, the refiner's four as int8 ConvRot |
| `vae/minimax_h3_video_vae_int8_convrot.safetensors` | 2.8 GB | the fp16 file's tensors with the same names and header; the four matrices of each of the 36 decoder blocks as int8 ConvRot, the decoder's other tensors float32 |
| `text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors`, `..._int8_convrot.safetensors`, `..._nvfp4_awq.safetensors` | 51.5 / 27.1 / 15.7 GB | the first 50 of Qwen3-VL-32B's 64 layers as `model.layers.N.*`, `model.embed_tokens` and the vision tower (`visual.*`), without final norm, head or configuration (`comfy/text_encoders/llama.py`, `Qwen3VL_32BConfig`: the conditioning is the last layer's output). The seven matrices of a layer in bf16, int8 ConvRot, or NVFP4 (below); the NVFP4 file's embedding is int8 with a scale per row. The tower is bf16 in all three |
| `vae/minimax_h3_video_vae_fp16.safetensors` | 5.2 GB | MiniMax's original tensor names (`FL2VA/video_vae`: `encoder.down.I.block.J`, a fused per-head `to_qkv`, `ff.w1` as `[gate; value]`) in float16, the configuration in the header. Sampled rows equal diffusers' `vae/` within float16 rounding (1.5e-5; the encoder's within 3e-8) |
| `vae/minimax_h3_audio_vae_fp32.safetensors` | 0.6 GB | the release's `audio_vae/` in float32 with the weight normalization of the BigVGAN convolutions folded in (`.weight` where the release has `.weight_g` / `.weight_v`; the 743 tensors under the same name are byte for byte the release's), the encoder included. The header has `latents_mean` / `latents_std` (the values of the release's `config.json`), `sample_rate`, and MiniMax's `kwargs` (`decoder_rates`); the AMP kernels and dilations are not in it and are taken as the release's (3, 7, 11 by the tensor shapes; 1, 3, 5) |
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

**The quantized files.** A quantized module has a `comfy_quant` tensor (the bytes of a JSON
object) beside its weight; `comfy/ops.py` loads it into a layout of the `comfy_kitchen` package,
whose eager backend is the definition used here (`src/comfy_quant.rs`; read at comfy_kitchen
0.2.41, `Comfy-Org/comfy-kitchen` at `bada696`, and ComfyUI at `c3fd8b9`):

- `int8_tensorwise` with `"convrot": true` (`tensor/int8.py`, `tensor/int8_utils.py`): int8 rows
  times a float32 scale per row (`weight_scale` `[N, 1]`), of the matrix rotated on its input side,
  `W_rot = W H^T` over consecutive groups of `convrot_groupsize` inputs (256; 64 for the AdaLN
  projections, whose 2688 inputs 256 does not divide). `H` is the normalized *regular* Hadamard
  matrix, the Kronecker power of `[[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]]`
  over `sqrt(n)`: symmetric, its own inverse, not the Sylvester matrix (which is why undoing a
  Sylvester rotation gave correlation 0.00 to 0.03). `W x = W_rot (H x)`: the input is rotated,
  then multiplied by the stored rows.
- `w6a8_int8` (`tensor/w4a8_int8.py`, `backends/eager/w4a8_int8.py`): 6-bit codes in int8
  `[N, 3K/4]`, a `K/2`-byte plane of low nibbles (column `2i` in the low nibble of byte `i`)
  followed by a `K/4`-byte plane of the top two bits (column `c` in byte `c / 4`, bit
  `2 (c % 4)`); `weight_s_rel`, an FP8 E4M3 scale per group of 32 columns (`group_size`), and
  `weight_s_channel`, a float32 scale per row. A weight is
  `clamp(round((code - 32) s_rel), -127, 127) s_channel`, in the ConvRot basis above (groups of
  256): the reference decodes the codes to that int8 matrix before every product.
- `nvfp4` (`tensor/nvfp4.py`, `float_utils.py`): E2M1 pairs with the first value in the *high*
  nibble (ModelOpt and compressed-tensors, which oxinfer already read, put it in the low one), an
  E4M3 scale per 16 values stored in cuBLAS's tiled order (`to_blocked`: tiles of 128 rows by 4
  blocks, each stored as 32 rows of 16 with its four 32-row bands side by side) and a tensor-wide
  `weight_scale_2`. The file is ModelOpt AWQ output: `o_proj` and `down_proj` have a
  `pre_quant_scale` vector that multiplies their input, and the scale of the other five matrices
  is folded into the norm before them (in layers 0, 25 and 49 `input_layernorm` and
  `post_attention_layernorm` differ from the bf16 file's by factors of 0.06 to 18; `q_norm` /
  `k_norm` are identical).

Checked on sampled rows of the real files (range requests; two rows each of matrices in the
first, a middle and the last block or layer), decoded as above against the same rows of the bf16
files:

| file | correlation with the bf16 row | relative L2 error |
|---|---|---|
| pruned int8 ConvRot transformer, the four matrices | 0.99991 to 0.99997 | 0.8% to 1.4% |
| int8 ConvRot transformer, AdaLN projections (groups of 64) | 0.99997 | 0.75% to 0.83% |
| W6A8 transformer, the four matrices (the same rows of the int8 file: 0.8% to 1.4%) | 0.99970 to 0.99974 | 2.3% to 2.5% |
| int8 ConvRot video VAE, decoder matrices (against the fp16 file) | 0.99995 to 0.99997 | 0.7% to 1.0% |
| int8 ConvRot text encoder | 0.99995 to 0.99996 | 0.9% to 1.0% |
| NVFP4 text encoder, `o_proj` / `down_proj` times `pre_quant_scale` (without it: 0.89 to 0.98) | 0.9942 to 0.9954 | 9.5% to 10.8% |
| NVFP4 text encoder, the other five times the ratio of the two files' norm weights | 0.9938 to 0.9957 | 9.3% to 11.1% |
| NVFP4 text encoder, the int8 embedding rows | 0.99993 to 0.99997 | 0.8% to 1.2% |

The errors are what the bit widths give (an int8 step on a rotated, near-Gaussian row; two and a
half times that at 6 bits with a scale per 32; E2M1's eight magnitudes). With a wrong reading the
same rows give: no rotation 0.01 to 0.09, the Sylvester matrix 0.00 to 0.03; NVFP4 with the low nibble first
0.00, with the scales in row-major order 0.84 to 0.93 at a wrong magnitude.

How they run: every product is float32 on the decoded weights, as for FP8. That is what ComfyUI
computes where it has no int8 kernels (its "emulated ops", `supports_int8_compute`: the CPU, MPS,
Intel XPU; and `full_precision_matrix_mult`, which the NVFP4 text encoder's entries ask for on
every device). Elsewhere (CUDA, ROCm) ComfyUI rounds the rotated input of every int8 / W6A8 layer
to int8 with one scale per row and multiplies integers (`int8_linear`: the "A8"). That rounding is
not reproduced: on the tiny checkpoints the two computations differ by 5e-3 (int8) and 2e-2 (W6A8)
after the schedule, where oxinfer is within 2e-6 of the float32 one (below). In memory:

- int8 ConvRot weights are used from the mapped file as stored (one float32 scale per row beside
  them); a row is converted to float32 on each pass, like an FP8 or bf16 row. Timed on a synthetic
  `[21504, 5376]` matrix and 2048 rows (64 cores, while another job loaded the machine): 0.38 to
  0.40 s with the rotation, 0.34 to 0.37 s as FP8, 0.30 to 0.33 s as bf16, so no gain in speed is
  to be expected from the int8 file on the CPU: its advantage over FP8 would be accuracy, which
  was not measured;
- W6A8 weights are decoded at load to the int8 values above: 8 bits per weight in memory for 6 in
  the file (19.3 GB of process memory for the 50 blocks instead of 14.5 GB of mapped file);
- NVFP4 weights are repacked at load into ggml's NVFP4 blocks, as the other NVFP4 packings are
  (4.5 bits per weight: 13.7 GB of process memory for the text encoder's 24.4B weights); its int8
  embedding is read from the mapped file.

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

The quantized forms are written by comfy_kitchen's own quantizers
(`TensorWiseINT8Layout.quantize`, `AsymW4A8Int8Layout.quantize(bits=6)`) for a second tiny
transformer of width 64 (a W6A8 row is a multiple of 32, a ConvRot group a power of 4), with the
`comfy_quant` entries of the real files; the reference is diffusers on the weights comfy_kitchen
dequantizes (same columns; the unquantized wide transformer gives 1.1e-5 / 2.6e-6 and
1.2e-6 / 9.5e-7):

| transformer (reference) | first step's velocities, video / audio | rows after the 5 steps, video / audio |
|---|---|---|
| int8 ConvRot, the blocks' four matrices in groups of 64 and the AdaLN projections in groups of 16 (diffusers on the dequantized weights) | 1.0e-5 / 2.9e-6 | 1.1e-6 / 9.5e-7 |
| pruned int8 ConvRot (the curve form on the dequantized weights) | 1.0e-5 / 2.7e-6 | 7.2e-7 / 9.5e-7 |
| pruned int8 ConvRot with a LoRA (merged into the dequantized weights) | 7.8e-6 / 2.4e-6 | 7.2e-7 / 9.5e-7 |
| pruned W6A8: the blocks and the video head, the refiner as int8 ConvRot (the same) | 8.8e-6 / 2.5e-6 | 2.1e-6 / 7.2e-7 |
| controls: the pruned int8 file without the LoRA; the W6A8 file against the int8 reference; the unquantized file against the int8 reference | 1.2e-1 / 2.2e-1, 9.9e-2 / 3.8e-2, 9.8e-3 / 9.5e-3 | 1.0e-1 / 2.1e-1, 1.0e-1 / 5.8e-2, 1.0e-2 / 1.3e-2 |
| against comfy_kitchen's `int8_linear` / `w4a8_int8_linear` in every quantized layer (int8 activations, what ComfyUI's int8 kernels compute): pruned int8, pruned W6A8 | 6.6e-3 / 9.4e-3, 2.3e-2 / 1.1e-2 | 5.2e-3 / 4.4e-3, 2.0e-2 / 5.8e-3 |

The int8 ConvRot video VAE (the tiny decoder's matrices quantized by comfy_kitchen, diffusers on
the dequantized ones): encoder 1.1e-6 / 5.1e-7, decoder 7.2e-7 / 6.4e-7 / 7.2e-7 (1.8e-3 against
the fp16 reference).

ComfyUI's single-file text encoders: `tools/qwen3vl_comfy.py` writes a tiny random Qwen3-VL (6
layers of width 128) as the three files, truncated to 4 layers, the NVFP4 one with random AWQ
scales folded into the norms or stored as `pre_quant_scale` and an int8 embedding;
`examples/h3_text_check.rs` compares the residual stream after the last stored layer with
transformers' `hidden_states[4]` on the dequantized weights (14 tokens, values up to 6.5):

| text encoder | max abs diff |
|---|---|
| the folder (transformers' own weights) | 5.4e-6 |
| bf16 single file | 5.4e-6 |
| int8 ConvRot single file (groups of 64) | 7.3e-6 |
| NVFP4 single file | 6.0e-6 |
| controls: the NVFP4 file against the bf16 reference, the bf16 file against the NVFP4 one | 3.2 |

The pipeline's own text path (`encode_prompt_quant`: the release's `tokenizer/tokenizer.json`,
the 50th layer) was run on the same tiny model with 52 layers and Qwen's vocabulary, the files
truncated to 50, for a 22-token prompt in the release's form against transformers'
`hidden_states[50]` on the ids of the `tokenizers` package (values up to 22): the folder and the
bf16 file 3.4e-5, the int8 ConvRot file 8.1e-5, the NVFP4 file 4.3e-5 (15 against another file's
reference).

The tiny pruned checkpoint itself is 1.5e-4 from the unpruned one (a rank-8 basis and float16
projections of a random 24-wide curve). The fp16 single-file VAE against diffusers with its weights
rounded to float16: encoder 1.1e-6 (clip) / 5.1e-7 (one frame), decoder 7.2e-7 / 7.2e-7 / 6.0e-7.

Text encoder as GGUF: a tiny random Qwen3-VL written with the tensor names and metadata keys of
the published `qwen3vl` files (`tools/qwen3vl_gguf.py`) gives transformers' logits through
`examples/qwen4exp_check.rs` as the folder does (5.1e-7, same top token on 13 of 13 steps). The
header of the real `Qwen3-VL-32B-Instruct-Q4_K_M.gguf` (its first 24 MB in a sparse file: the
weights read as zeros) loads through `QwenVl` as 64 layers of width 5120 with the release's
configuration values, and three prompts tokenize to the same ids as with the release's
`tokenizer.json`. Neither that file nor the GGUF's vocabulary has MiniMax's seven tokens (the
GGUF's entries from 151669 on are unused `[PAD151669]`, ...): the pipeline adds them to both, see
the prompt's tokens below.

The single-file audio VAE against the release folder (both downloaded): `h3_audio_check` (`--write`
keeps one decoder's waveform as the reference for another) decodes 207 seeded normal latents per
channel with it within 3.7e-5 of the release folder's decode (107 dB SNR; the folded weight
normalization rounds differently).

The command line was run end to end on the tiny files (the pruned FP8 transformer and the fp16
VAE as single files, a LoRA named `*turbo_4step*768p*`, a stored prompt embedding): 4 forwards at
shifts 6 / 3 and 22 frames written; `--steps`, `--shift` and `--audio-shift` override the LoRA's
settings, and a single-file `--model` without `--vae` is refused.

**A real request's sequence on a tiny transformer.** The tiny cases above have 191 rows, heads of
32 and two keyframes. `tools/minimax_h3_video_long.py` writes a tiny random transformer with the
real heads (128 wide, 16 rotary frequencies per axis) and diffusers' outputs on the sequence of a
real request: 448x256, 124 frames (37 latent frames of 8 x 14 rows, 414 audio rows), no
keyframes, 5 sigma points at shifts 6 / 3. `h3_video_check DIR --transformer DIR/transformer`
(float32, CPU, max absolute difference):

| transformer, sequence | first step's velocities, video / audio (mean magnitude) | rows after the 4 steps, video / audio (mean magnitude) |
|---|---|---|
| 3 heads, width 96, 3 blocks; 124 text rows, 4682 in all | 6.7e-6 / 2.5e-6 (0.90 / 0.87) | 1.9e-6 / 1.4e-6 (1.2 / 1.2) |
| 7 heads, width 160, weights 2.5 times larger; 159 text rows, 4717 in all | 4.0e-4 / 3.1e-4 (17 / 14) | 1.5e-4 / 1.6e-4 (20 / 12) |

**The real files against torch and ComfyUI.** A tiny checkpoint has neither the real weights'
ranges (the packed sequence reaches 7e6 in the last blocks) nor their formats at full width, so
one real request was redone outside oxinfer: the dog clip of the [real runs](#real-runs)
(448x256, 124 frames, a 124-token prompt: 4682 rows, seed 1, the pruned FP8 transformer with the
4-step 768p LoRA, the Q8_0 text encoder). `OXINFER_H3_DUMP=DIR` keeps every forward's rows and
velocities, and with `OXINFER_H3_DUMP_BLOCKS=i` forward `i`'s sequence after every block.
`tools/minimax_h3_video_ref.py` decodes the same two files in Python (FP8 times `weight_scale`,
the LoRA merged in float32, the curve form of the timestep embedding) into diffusers'
`MiniMaxH3Transformer3DModel` and runs its `forward`, its layout and timestep helpers and
`MiniMaxH3Scheduler` unchanged, in float32 on the CPU, from oxinfer's noise and its prompt
embedding (relative differences, torch against oxinfer):

| what | video | audio |
|---|---|---|
| each of the 50 blocks on oxinfer's input of that block, first forward: the block's change of the sequence | 2.3e-6 to 2.0e-5 (median 5.8e-6) | (one sequence) |
| the 50 blocks one after the other, first forward: the sequence after the last block | 5.1e-4 | |
| one whole forward on oxinfer's rows before each of its four steps: the velocities | 3.2e-4, 3.1e-4, 2.6e-4, 1.6e-4 | 1.4e-5, 2.1e-5, 3.2e-5, 2.6e-5 |
| the rows after the last step, from oxinfer's rows before it | 1.5e-4 (1.3e-2 at most, on values of standard deviation 1) | 2.7e-5 |
| the whole sampling in torch from the same noise: the velocities of the four steps | 3.2e-4, 3.2e-3, 1.7e-2, 5.3e-2 | 1.4e-5, 9.5e-5, 1.3e-3, 8.1e-3 |
| its final rows | 5.0% (per row: 0.035 rms in the median, 0.28 at most) | 0.84% |

Each forward agrees within 3e-4; two samplings that each follow their own rows drift apart by
about ten times per step, which is how much this model amplifies a difference of its input.
Decoded by oxinfer's VAEs, the torch sampling's rows give the same clip as oxinfer's, flaws
included (see the real runs).

ComfyUI's own code on the same forward (`tools/minimax_h3_video_comfy_ref.py`: a ComfyUI
checkout's `comfy.sd.load_diffusion_model`, `load_lora_for_models` and `MiniMaxH3Model.forward`
on the CPU in float32, float activations in every quantized layer):

- ComfyUI reads the file's unquantized tensors as oxinfer does (`condition_proj`, a norm, an
  AdaLN projection, the video head: no difference), and its forward agrees with diffusers' on
  the weights ComfyUI holds within 6.8e-5 (video) and 2.7e-6 (audio): the layout, the positions,
  the curve form, the audio timestep and the block arithmetic are the same in the three
  implementations;
- its weights are not the same once a LoRA is loaded: on the CPU ComfyUI merges the LoRA into
  each FP8 matrix and rounds the sum back to FP8 (stochastically, unbiased). In the first block
  that rounding moves a matrix by 0.8 to 0.9% (relative, Frobenius) where the LoRA itself moves
  it by 0.13 to 0.21%, and ComfyUI's change of the matrix has a cosine of 0.17 to 0.22 with the
  LoRA's. oxinfer keeps the FP8 matrix and adds the LoRA's product in float32. The first
  forward's velocities differ by 9.2% (video) and 1.8% (audio) for that reason, the last
  forward's by 3.8% and 1.4%.

**The prompt's tokens and its embedding against transformers.** The runs above take oxinfer's
prompt embedding, so the text path was checked on its own against diffusers' text step
(`MiniMaxH3TextEncoderStep` and `get_qwen3vl_prompt_embeds`, diffusers 0.43.0 with transformers
5.23.0), for the two prompts of the [real runs](#real-runs):

- what the reference feeds the conditioner: `tokenizer(prompt, add_special_tokens=False)`, no
  chat template, no system prompt, no padding or truncation, an attention mask of ones, and
  `text_encoder.model(...).hidden_states[50]` handed to the transformer unchanged (no norm, no
  crop, no scaling), every row tagged as text (1). ComfyUI (`comfy/text_encoders/minimax.py`)
  builds the same sequence: the raw prompt, no template, no start or end token, the last of 50
  layers without the final norm;
- the release's `tokenizer/` is Qwen's `tokenizer.json` (26 added tokens, the last `</think>` at
  151668) beside a `tokenizer_config.json` whose `additional_special_tokens` names seven more:
  transformers appends them (`<d>` 151669, `</d>` 151670, `<|cutoff|>`, `<|lyrics_start|>`,
  `<|lyrics_end|>`, `<|caption_start|>`, `<|caption_end|>` up to 151675), and ComfyUI adds the same
  seven to its bundled Qwen tokenizer under the same ids (its `MINIMAX_EXTRA_TOKENS`; both
  tokenizers give the same ids for the two prompts). **oxinfer read `tokenizer.json` alone, and a
  GGUF's vocabulary has unused `[PAD151669]`... entries there, so it split the two tags**: the
  speech prompt's ` <d>[` and `.</d>` were the six tokens ` <`, `d`, `>[` and `.</`, `d`, `>`
  (159 tokens) where the reference has ` `, `<d>`, `[` and `.`, `</d>` (158). The pipeline now adds
  the seven tokens to whichever tokenizer it loaded (`video::EXTRA_TOKENS`, a unit test on a
  small vocabulary);
- with that, `h3_text_check dump` gives the reference's ids for both prompts (124 and 158),
  through the Q8_0 and the Q4_K_M GGUF's vocabulary and through the release's `tokenizer.json`
  (the same embedding byte for byte as with the GGUF's vocabulary). The dog prompt has no such
  tag: its ids were the reference's before the change, and the embedding the dog runs used is
  the one the new code computes, byte for byte;
- the embedding: `tools/minimax_h3_text_ref.py` reads the Q8_0 GGUF with the `gguf` package,
  dequantizes each tensor to float32 and runs transformers' `Qwen3VLTextModel` on the
  reference's ids (the configuration from the GGUF's metadata, which is the release's
  `text_config`; one decoder layer refilled 50 times, each layer's output taken before the
  final norm; float32 on the CPU), against oxinfer on the stored Q8_0 blocks:

| `hidden_states[i]`, oxinfer against transformers | dog (124 tokens): relative, all rows / worst row | speech (158 tokens) | largest difference (values up to) |
|---|---|---|---|
| 1 | 2.6e-6 / 4.0e-6 | 2.6e-6 / 4.4e-6 | 9.0e-5 (rows of norm 23) |
| 2, 3 | 4.0e-6, 4.4e-6 / 1.5e-5, 1.8e-5 | 3.9e-6, 4.3e-6 / 1.5e-5, 1.8e-5 | 6.3e-4 |
| 10, 25 | 4.9e-6, 4.6e-6 / 1.5e-5, 1.4e-5 | 4.9e-6, 4.6e-6 / 1.5e-5, 1.4e-5 | 0.072 in the first row (15,350) |
| 49 | 3.6e-6 / 2.1e-5 | 3.6e-6 / 1.7e-5 | 0.055 in the first row |
| **50, the conditioning** | **3.6e-6 / 1.8e-5** (4.4e-6 without the first row; 1.2e-3 at most on rows of norm 300) | **3.6e-6 / 1.9e-5** | 0.055 in the first row (15,350) |
| controls: oxinfer's 50 against the reference's 49 and 51; its rows shifted by one | 24% and 23% (median row), cosine 0.979 and 0.976; 470% and 73% | | |

Every row's cosine with the reference is 1.0000000. The first row (the first token sees only
itself) has a norm of 15,485, 50 times the others' 301 in the median; the reference has it too,
and it is part of what the transformer is given. So for these prompts oxinfer's conditioning is
transformers' `hidden_states[50]` of the same Q8_0 values to float32 rounding: the layer index,
the missing final norm, the positions and the rows kept are the reference's. The release loads
its text encoder in bfloat16: the same reference with the weights rounded to bfloat16 and
computed in bfloat16 (`--dtype bf16`, the dog prompt) is 1.2% from the float32 one (0.87% in the
median row, 9.1% in the worst; 0.35% already after the first layer), so the reference pipeline's
own embedding is known no better than that.

The other inputs those runs shared with oxinfer, read against the same diffusers source: the
transformer's `forward` takes the video rows, the audio rows, the prompt embedding, the timesteps
with their row indices, the tags, the positions and the three index lists, and nothing else (no
frame rate, duration, size or guidance input, no negative prompt); the torch run built tags,
positions and timesteps with diffusers' own helpers from the request's geometry and found
oxinfer's positions equal. The noise is standard normal with no scaling (the first sigma is 1):
diffusers draws the video noise as a `[1, 24, frames, h, w]` tensor and packs it as oxinfer does
(frame-major rows of `[channel, 2, 2]`), then the audio noise directly as rows, in that order
from one generator; oxinfer draws the same shapes in the same order from its own generator, so
its noise is another sample, not torch's for the same seed.

Not checked or not done:

- the real runs are few and small (see [below](#real-runs)): nothing at 1344x768 longer than 22
  frames, one request compared with torch and with ComfyUI's code on the same files (above) and
  the others not, no sampling in ComfyUI itself (its nodes, its sampler, a GPU), and the
  unpruned and pruned bf16 transformers, the other LoRA versions and `--quant q8` were not run;
- no run on the real int8 ConvRot or W6A8 transformers, the int8 VAE or a single-file text encoder
  (bf16, int8, NVFP4): the sampled rows above show that their weights are read as ComfyUI reads
  them; what int8, 6-bit or FP4 weights do to a video, and whether int8 does better than FP8, was
  not measured;
- ComfyUI's int8 activation path (see "How they run"): a CUDA run of ComfyUI on the int8 or W6A8
  file is a slightly different computation from oxinfer's, by an amount known only on the tiny
  checkpoints;
- the 34.0 GB unpruned int8 file's layout is the tiny file's (int8 AdaLN projections in their own
  group size), but only four of its matrices were sampled; the `ref2va` files were not looked at
  beyond their sizes;
- the command itself with a single-file text encoder: the tiny encoders are 128 wide and the tiny
  transformers read 40, so the text path (above) and the command (run on the three quantized tiny
  transformers with the int8 VAE, a LoRA named `*turbo_4step*768p*` and a stored prompt embedding:
  4 forwards, 22 frames written) were run separately;
- the Q8_0 text encoder is computed as transformers computes it from the same values (above), but
  what Q8_0 weights do to the embedding against the release's bf16 weights was not measured (the
  63 GB encoder was not downloaded); Q4_K_M was compared with Q8_0 on two prompts (below), and
  `--quant q8` was not run;
- keyframes with a GGUF text encoder (the vision tower is a separate `mmproj` file, not read
  here) or with a single-file one (its `visual.*` tensors are there, but the tower's
  configuration is not, and it is not built from the shapes), LoRAs on the text encoder, LoRA
  keys other than the two namings above.

### Real runs

On the CPU (64 cores, 125 GB RAM, shared with other jobs: timings vary by 1.5x between runs of the
same request), a `--profile ci` build (release without LTO), `--seed 1`, every file mapped:
`minimax_h3_fl2va_pruned_fp8_scaled` (21.0 GB), `minimax_h3_video_vae_fp16` (5.2 GB),
`minimax_h3_audio_vae_fp32` (0.6 GB), a turbo LoRA (2.0 GB) and `Qwen3-VL-32B-Instruct-Q8_0.gguf`
(34.8 GB). All of them load without a change to the loaders: transformer, VAEs and LoRA (208
modules) in 6 to 10 s, the text encoder's 124- and 158-token prompts in 5.3 to 5.7 s with Q8_0 and
18.8 s with Q4_K_M (load included). Peak resident memory 28.6 GB (6.9 GB of it allocated, the rest mapped
weights), 31.2 GB when the text encoder runs in the same process. Two prompts in the release's
form: a woman at a kitchen table who says `<d>[English] Good morning. The coffee is ready.</d>`
and lifts a mug ("speech"), and a golden retriever on a lawn that barks three times, then pants
("dog"). The speech prompt was first run with its `<d>` and `</d>` split into three tokens each
(159 text rows, [above](#what-was-checked-1)); the 124-frame Q8_0 run was redone with the
release's tokens (158 rows), the two other speech runs were not, and are marked "split tags".

| run | rows | forwards | VAE | total |
|---|---|---|---|---|
| speech, 448x256, 124 frames, 4-step 768p LoRA (shifts 6 / 3) | 4144 video + 414 audio + 158 text | 4 in 1300 s (325 s each) | 310 s | 27 min |
| the same with split tags | 4144 video + 414 audio + 159 text | 4 in 1286 s (322 s each) | 319 s | 27 min |
| dog, the same settings | 4682 in all | 4 in 1322 to 1894 s | 316 to 345 s | 27 to 38 min |
| speech (split tags), the same with the Q4_K_M text encoder | 4717 in all | 4 in 1617 s | 320 s | 33 min |
| dog, 448x256, 124 frames, 8-step LoRA (shifts 12 / 3) | 4682 in all | 8 in 2743 s (343 s each) | 317 s | 51 min |
| speech (split tags), 1344x768, 22 frames, 4-step 768p LoRA | 7056 video + 74 audio + 159 text | 4 in 3004 s (751 s each) | 663 s | 61 min |
| dog, 448x256, 22 / 56 frames, 4-step 768p LoRA | 982 / 2214 in all | 4 in 147 s / 437 s | 45 s / 135 s | 3.3 / 9.7 min |

These runs predate the blocked attention kernel (`sd::ops::attention_blocked`), which is 8 to 12
times faster at these sizes. The audio VAE takes 0.3 to 1.3 s. Attention was 46 to 60% of the forwards' time (`--profile`) and
grows with the square of the rows: a default request (1344x768, 124 frames, 37,869 rows) was not
run and would take hours per forward here. The same request twice gives the same file byte for
byte.

What the frames show (contact sheets read by the model that ran this, not by a person):

- 448x256, 124 frames, 4-step 768p LoRA: the prompts' scenes, sharp, the same person / dog and
  background through the clip, the described actions in order (she speaks, smiles, lifts the mug
  and drinks; the dog barks with its head raised three times, then pants). **With flaws**: in the
  dog clip coloured blobs (green, yellow, pink, about 60 x 90 pixels) appear on the lawn and the
  fence for about four frames each in most of the clip, and its contrast is harsh. The speech
  clip has them too with the release's tokens (blue, purple and pink, on the plants at the
  window, the mug, the bowl and the cupboard, in about 15 of every third frame's 42); with the
  split tags it had none with the Q8_0 encoder and two with Q4_K_M, so the first speech clip was
  clean by the luck of a slightly different embedding, not because speech is spared. The dog
  prompt and seed at 22 and 56 frames, the speech prompt (split tags) at 1344x768 and the dog
  with the 8-step LoRA have none. The blobs are what these weights give on this request, not
  something oxinfer adds:
  - they are in the denoised rows, not made by the VAE (decoding the first 17 latent frames
    alone gives the same blobs), and no value of those rows is out of range (all within 4.3
    standard deviations);
  - of the four forwards' estimates of the clean rows (`x + sigma v`, decoded), the first two
    (sigma 1 and 0.947) are clean, blurred then sharper; the third (0.857) has the blobs,
    faint; the fourth (0.667), which is the result, has them at full strength;
  - torch on the same files, from the same noise and prompt embedding (diffusers' transformer
    and scheduler, [above](#what-was-checked-1): every forward within 3e-4 of oxinfer's on the
    same rows), gives the same clip with the same blobs at the same places and frames, whether
    it follows its own rows through the four steps or oxinfer's;
  - ComfyUI's own model code with its own reading of the two files (four `MiniMaxH3Model`
    forwards on the CPU from the same noise and embedding, Euler steps, oxinfer's VAEs) gives
    the same scene with blobs too: several at the same places and frames, more saturated
    (orange, teal), and others elsewhere. Its weights differ from oxinfer's by the FP8 rounding
    of the merged LoRA (above), so its clip is another sample of the same request.

  Why these weights do that here was not found. lightx2v's table gives 1344x768 as the 768p
  file's only training resolution, and this request is 448x256; the transformer is the FP8
  rounding of the bf16 one the LoRA's header names as its base; a small change of the
  conditioning (Q4_K_M for Q8_0, or the speech prompt's two tags as one token each instead of
  three) adds or removes blobs. Which of these matters was not separated: the bf16
  transformers were not downloaded, 1344x768 at 124 frames was not run (37,869 rows), and
  nothing was sampled in ComfyUI itself (its nodes and sampler, on a GPU). The prompt embedding,
  which those three runs shared, is not the cause either: the dog prompt's token ids are the
  ones diffusers' text step and ComfyUI's tokenizer give, and its embedding is transformers'
  `hidden_states[50]` on the same Q8_0 values within 3.6e-6 (1.8e-5 in the worst row,
  [above](#what-was-checked-1)). What is left of the conditioning is the Q8_0 rounding of the
  encoder's weights, not measured against the bf16 encoder; for scale, the release's own
  arithmetic (bfloat16 weights and activations instead of float32, in transformers on the same
  values) moves that embedding by 1.2% (0.87% in the median row, 9.1% in the worst), and Q4_K_M
  moves it by 5%. The noise is oxinfer's own generator's, standard normal in the reference's
  layout (it passes normality checks);
- 448x256, 124 frames, 8-step LoRA (the dog): clean, softer and more natural than the 4-step clip;
- 1344x768, 22 frames: clean and detailed (knit of the sweater, the hands on the mug), but the
  whole prompt is played in 0.9 s, below the 5 s the model is made for.

What the soundtracks are (statistics, spectrograms and oxinfer's Whisper, `openai/whisper-small`
with `--transcribe`; nobody listened):

| run | level (RMS / peak, both channels alike) | content | Whisper |
|---|---|---|---|
| speech, 124 frames, Q8_0 | -22 dBFS / 0.71 and 0.76, no clipping, 47% and 54% of the 20 ms windows below -60 dBFS | two voiced segments with harmonics and formants at 0.3 to 1.1 s and 2.0 to 2.9 s, near silence between and after | `Good morning. The coffee is ready.` (English, p = 1.00) |
| the same with split tags | -31 dBFS / 0.33, no clipping, 69% silent windows | two voiced segments at 0 to 0.7 s and 2.0 to 2.9 s | `Good morning. The coffee is ready.` (p = 0.97) |
| speech (split tags), 124 frames, Q4_K_M | -31 dBFS / 0.30 | the same two segments at the same times | `Good morning. The coffee is ready.` |
| speech (split tags), 1344x768, 22 frames | -16.5 dBFS / 0.91 | continuous speech over the 0.9 s | `The coffee is ready.` |
| dog, 124 frames, 4-step | -23 dBFS / 0.71, 63% silent windows | three bursts of 0.15 s at 0.9, 2.3 and 2.8 s, energy up to 15 kHz, most of it below 2 kHz; five weaker bursts from 3.6 s (panting) | `dog barking` |
| dog, 124 frames, 8-step | -16 dBFS / 1.0, 0.17% of the samples at the clamp | three barks with clear harmonics at 0.9, 2.0 and 2.8 s, panting after, short chirps at 4 to 5 kHz (the prompt asks for birdsong) | `Woo! Woo! Woo!` |

The two channels correlate 0.85 to 0.97 (stereo, not two copies). Every soundtrack has 800
samples per latent: 165,600 samples (5.175 s) for 124 frames (5.167 s), 29,600 (0.925 s) for 22
(0.917 s). ffprobe reads each AVI as `mjpeg` 24 fps with the right frame count and `pcm_s16le`
32 kHz stereo with that many samples, and ffmpeg converts them to H.264 + AAC. Whether the sound
is in sync with the picture (lips, barks) was not judged.

**The text encoder at Q4_K_M** against Q8_0: `hidden_states[50]` differs by 5.0% for the speech
prompt and 5.4% for the dog prompt (relative, without the first row, whose norm is 50 times the
others'; 4.9% in the median row of both, 9.9% and 20% at most; cosine 0.9988 in the median row).
The videos were compared for the speech prompt with split tags (the embeddings differed by 4.9%
there too): another sample of the same scene (the same woman, kitchen and actions, a different
bowl on the counter; 22.7 dB PSNR between the two), with the two blobs noted above; the
soundtrack has the same words at the same times (correlation 0.85 with the Q8_0 one). One prompt
and one seed: not enough to say Q4_K_M is worse.

**The speech prompt's tags.** With `<d>` and `</d>` as the release's single tokens instead of
three tokens each, the 87 rows before the first tag are unchanged, and the 57 rows after the
second differ by 6.8% from what they were (3.6% in the median row, 30% at most). The clip is
another sample of the same scene with the same words (`Good morning.` now starts at 0.3 s
instead of 0 s, the second sentence at 2.0 s in both), 9 dB louder, and with the blobs noted
above where the first clip had none: one seed each, so neither the level nor the blobs can be
put down to the tokens.

### On CUDA

With `--device cuda` the transformer's 50 blocks and the video VAE decoder's 36 run on the GPU
(`src/minimax_h3/cuda.rs`, kernels in `cuda.cu`): the sequence stays on the device from the
first block to the last, the products run on the tensor cores, and the host keeps the text
encoder, the layers before and after the blocks, the AdaLN tables and the schedule (the audio
VAE runs as before). It takes the files of the runs above, FP8 "scaled" transformer weights with or without
LoRA side paths and the float16 VAE; any other format (the bf16 and int8 ConvRot transformers,
the release's diffusers folders, W6A8) still runs on the CPU, as do heads that are not a
multiple of 64 wide. Measured on one RTX 5080 (16 GB, 1.5 GB of it in use by the desktop) beside a
Threadripper 9970X shared with other jobs, a `--profile ci` build, the dog request of the table
above (448x256, 124 frames, 4682 rows, the 4-step 768p LoRA, `--seed 1`, the kept Q8_0 prompt
embedding):

| | CPU | CUDA |
|---|---|---|
| one forward | 194 s (457 s in a run of one forward while the machine's other jobs were busier) | 8.6 s (12.8 to 13.4 s the first one: the LoRA factors as halves, 23 to 27 blocks copied to the device) |
| the 4 forwards | 774.7 s | 38.6 s |
| video VAE decode (124 frames) | 160 to 187 s | 6.4 to 8.1 s |
| audio VAE decode | 1.2 s | 1.1 s |
| load (transformer, VAEs, LoRA) | 5.6 s | 6.2 s |
| the whole run, prompt embedding kept | 943 s (15.7 min) | 54 s |

The text encoder is not in these totals (5.3 to 13.6 s on the CPU for the 124-token prompt with
the Q8_0 GGUF, load included: GGUF weights have no device kernels). Peak resident memory of the
CUDA run 30.4 GB, 6.2 GB of it allocated.

Where one forward's time goes (`--profile`; on the device with `OXINFER_H3_GPU_PROF=1`, which
waits for the GPU after every stage):

| stage | CPU | CUDA |
|---|---|---|
| `qkv_proj` with its LoRA | 52.2 s | 1.90 s |
| attention (56 heads over 4682 rows) | 16.0 s | 2.54 s |
| `out_proj` with its LoRA | 17.8 s | 0.42 s |
| `fc1` with its LoRA | 56.0 s | 2.44 s |
| `fc2` with its LoRA | 41.0 s | 0.96 s |
| norms and AdaLN, query / key norms and RoPE, SwiGLU, residuals | 6.0 s | 0.22 s |
| copying the blocks that are not kept on the device | | 1.16 s (while the block before runs: not in the 8.6 s) |
| the whole forward | 193.7 s | 8.6 s |

The CPU column is a quarter of the four forwards of the run above. The CPU's time depends on
what else the machine runs: the forward measured alone took 457.5 s with the same split (the
four projections 426.8 s of it, attention 17.9 s), and the earlier runs above 322 to 543 s per
forward. The four projections, float32 GEMMs over weights decoded eight rows at a time, are 86
to 93% of a CPU forward.

**1344x768.** One forward of the same prompt at 1344x768, 124 frames (37,834 rows; `--steps 2`,
so the frames are noise and only the times mean something): 302.5 s, with 4 of the 50 blocks
kept on the device beside the activations (15.2 of the 16 GB in use), then the VAE's 124 frames
in 82.9 s and the audio in 1.2 s; 396 s in all, peak resident memory 33.0 GB (7.8 GB allocated).
The four forwards of a clip were not run at that size, on either device.

**Memory.** A block is about 0.45 GB on the device (0.385 GB of FP8 bytes as stored, the rest its
LoRA factors), the transformer 22 GB: 23 to 27 blocks stay there and the others go through two
slots, each copied while the block before it runs. The activations of 4682 rows take 1.2 GB
(the feed-forward runs in row chunks and the attention in query chunks of at most 256 MB of
scores, so neither grows with the square of the sequence). The transformer's blocks leave the
device before the VAE decodes (its 36 float16 blocks are 4.8 GB and all fit), and the VAE's
after.

**Arithmetic.** The tensor cores multiply float16 operands and add in float32. FP8 and float16
weights are exact in a half. Float32 operands (the activations, the attention's queries, keys,
probabilities and values, LoRA factors that are not exact in a half) go in as two halves each,
`hi + lo / 2048` after a power of two per row (per head for the values, per matrix for LoRA
factors), and every product as two or three tensor-core products. Against the float32 CPU path
and against torch (diffusers' transformer in float32 on the same inputs,
`tools/minimax_h3_video_ref.py`), relative L2 error of the video / audio velocities of a forward:

| forward | CUDA against the CPU path | CUDA against torch | the CPU path against torch |
|---|---|---|---|
| 1 (the same inputs for all three) | 9.5e-5 / 4.4e-6 (max abs 5.9e-3 / 3.5e-5, mean \|v\| 1.14 / 0.74) | 4.1e-5 / 2.4e-6 | 1.0e-4 / 4.8e-6 |
| 2, 3, 4 (the inputs an earlier CPU run kept) | | 5.0e-5, 1.3e-4, 4.9e-5 / 3.9e-6, 5.1e-6, 3.1e-6 | 3.1e-4, 2.6e-4, 1.6e-4 / 2.1e-5, 3.2e-5, 2.6e-5 (that earlier run) |

After the four steps the video rows are within 2.1% (relative L2) of torch's own four steps
from the same noise and the audio rows within 0.24%; the CPU path's are within 2.6% and 0.29%
of torch's and 2.3% and 0.17% of the CUDA run's (the 4-step schedule amplifies a forward's
rounding, [above](#what-was-checked-1)). The
VAE's decoded frames are within 3.8e-5 of the CPU's (mean 6.2e-7, mean magnitude 0.84). The
engine's own tests compare both kinds of stack (FP8 with AdaLN, weighted query / key norms and
LoRA side paths; float16 with biases and scaled residuals) with their definition in float64 on
random weights: within 2e-5 of the largest value, whole and in chunks
(`cargo test --lib minimax_h3::cuda`, on a machine with an NVIDIA GPU).

Tried and not kept as the default: plain float16 operands (`OXINFER_H3_GPU_EXACT=0`, one
tensor-core product instead of two or three) take 6.5 s of device time per forward instead of
8.5 s, with velocities 9.5e-3 from the CPU's on average and 0.33 at most (against 7.8e-5 and
5.9e-3). `OXINFER_H3_GPU=0` keeps the CPU path; `OXINFER_GPU_PLAN_DEBUG=1` says what the device
took or why it declined.

Not done: the text encoder on the GPU; the transformer's other weight formats there; requests
whose activations do not fit beside two blocks (the CPU path then runs); a comparison of the
decoded frames by eye.

### Still not supported

| files | why not |
|---|---|
| ComfyUI's int8 kernels (`comfy_kitchen`'s `int8_linear`: the rotated input rounded to int8 per row, integer products) | the int8 and W6A8 files run on float32 products of the decoded weights instead (see "How they run"). oxinfer's int8 matmul kernels take activations in blocks of 32 with their own scales and weights with a scale per 32, not a scale per row and a rotation: wiring them in would give a third arithmetic, matching neither ComfyUI path, for a gain in speed that was not measured |
| the vision tower of ComfyUI's single-file text encoders (keyframes with `--text-encoder FILE.safetensors`) | the files carry `visual.*` without the tower's configuration (ComfyUI hardcodes it: depth 27, width 1152, DeepStack after blocks 8, 16, 24); only the language model is read |
| other `comfy_quant` formats: `asym_w4a8_int8` (4-bit codes with an optional codebook), `convrot_w4a4`; a `weight_correction` or `weight_codebook` beside a 6-bit weight; `mxfp8`, `float8_e5m2`; `input_scale` | none of `Comfy-Org/MiniMax-H3`'s files uses them. The first two and a 6-bit weight with a correction are refused by name; `mxfp8` and `float8_e5m2` entries are not interpreted here (nothing was written or checked for them in this model); `input_scale`, a calibrated activation scale for ComfyUI's FP8 / NVFP4 kernels, is ignored |
| `diffusion_models/minimax_h3_ref2va_*`, `model_patches/*controlnet*`, `embeddings/*`, `vae/minimax_h3_sol_vae_*` | `ref2va`, the Fun ControlNet, prompt embeddings and the "sol" VAE are not implemented |
| `VideoRebirthAI/hyperflow-h3-flash` | a 13.7B few-step (4 / 6 / 8) transformer as GGUF, 7.2 GB (Q3_K_M) to 14.9 GB (Q8_0), or 16 GB rotated INT8: another architecture on the same blocks, per its `hyperflow_h3_flash_transformer.py`: a second (endpoint) timestep embedder mixed with gate 0.25, three dropped blocks, 102 low-rank factored linears, its own sigma grids. The GGUF tensor layout was not read |
