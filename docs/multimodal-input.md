# Image, video and audio input

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
library too (see [Using oxinfer from oxideav](cli.md#using-oxinfer-from-oxideav-and-other-crates)).

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

## Qwen2.5-VL

Qwen/Qwen2.5-VL-3B-Instruct, -7B-Instruct (and the other Qwen2.5-VL / Qwen2-VL-style checkpoints,
`model_type` `qwen2_5_vl`) run through the normal engine, with images:

```sh
./target/release/oxinfer --model Qwen/Qwen2.5-VL-3B-Instruct --image fox.png -p "Describe this image."
```

The language model is the Qwen2 transformer (biased query / key / value projections, no query /
key norms, contiguous M-RoPE sections 16 / 24 / 24 over time, height and width) on the Qwen code
path; the vision tower (`src/qwen25_vision.rs`) has 14-pixel patches, axial 2D RoPE, RMSNorm
blocks with a SwiGLU MLP, window attention over 8x8-patch windows in all blocks but
`fullatt_block_indexes` (whole-image attention), and the 2x2 patch merger. The chat format is the
checkpoint's (default system prompt "You are a helpful assistant."). Against transformers 5.20 in
float32 on the 3B model with a 512x512 image (324 image tokens): pixels identical, image features
1.0e-5 relative, last-position logits max diff 1.1e-4 (KL 1.3e-9) on the CPU; on Metal (half-
precision matrix units) 1.9e-3 and KL 1.5e-7; greedy answers identical. Video input and single-
token GPU decoding (biased projections) are not supported for these checkpoints yet (decoding runs
on the CPU: 12 tok/s for the 3B in bf16).

`oxinfer::qwen_vl::QwenVl` is the same model as a conditioning encoder for diffusion pipelines
(Qwen-Image uses it; Step1X-Edit and OmniGen2 use Qwen2.5-VL-7B / 3B the same way):
`QwenVl::load(dir, &QwenVlOptions { quant, mmap, tokenizer_dir, processor_dir, vision })`,
`encode_image(rgb, w, h, max_pixels) -> VlImage` (embeddings, merged grid, `placeholder()` text),
and `hidden_states(text, &images, &layers) -> Hidden` with the final-normed states (transformers'
`hidden_states[-1]`) and the residual stream after any requested layers (`hidden_states[l + 1]`),
M-RoPE positions for the images included. `examples/qwen_vl_hidden.rs` dumps them.
