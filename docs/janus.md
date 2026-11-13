# Janus-Pro: text to image and image understanding

DeepSeek's Janus-Pro is one autoregressive model that both draws and reads images (`src/janus/`);
JanusFlow, its rectified-flow sibling, is described further down.
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
each.

**JanusFlow** (`deepseek-ai/JanusFlow-1.3B`, recognised from its config; `src/janus/flow.rs`)
uses the same entry points:

```sh
./target/release/oxinfer --model deepseek-ai/JanusFlow-1.3B --generate-image "a red fox in the snow" -o fox.png
./target/release/oxinfer --model deepseek-ai/JanusFlow-1.3B -p "a watercolor lighthouse" -o l.png \
    --steps 30 --cfg 2 --seed 7 --num-images 2 --size 768x768
./target/release/oxinfer --model deepseek-ai/JanusFlow-1.3B --image chart.png -p "Which slice is the largest?" --chat
```

It generates by rectified flow in the latent space of the SDXL VAE, with the language model as
the velocity network. Each of the `--steps` Euler steps (default 30, t from 0 to 1) embeds the
48x48x4 latent with the `ShallowUViTEncoder` (sinusoidal time embedding through two linears, a
2x2 patch convolution, two ConvNeXt blocks with global response normalization modulated by the
time embedding) into 576 tokens, runs the LLaMA decoder over the prompt, the time embedding (in
place of the prompt's final `<｜begin▁of▁generation｜>`) and the tokens, and turns its hidden
states at the tokens back into a velocity with the `ShallowUViTDecoder` (RMSNorm and linear to
768 channels, concatenation with the encoder's tokens, the long skip connection, two more ConvNeXt
blocks, 1x1 convolution and pixel shuffle). Guidance is `cfg * v_cond - (cfg - 1) * v_uncond`
(default 2, the reference demo's), the unconditional sequence attending to BOS only as in the
reference's attention mask; both sequences of every image run as one batch, and the prompt's keys
and values are computed once and kept across the steps (`--cfg 1` runs the conditional sequence
alone). The latents are decoded by the Stable Diffusion VAE decoder (`src/sd/vae.rs`) with the
weights of `stabilityai/sdxl-vae`, fetched on first use (`JanusOptions::vae`, the C option `vae=`:
another folder or repo; a `vae/` folder inside the checkpoint is used when present), kept in full
f32 precision: rounded to bf16 they move the image by 0.6%. For understanding, JanusFlow has the
same SigLIP-L/16 tower with a single linear aligner and a learned embedding at `<image_beg>`,
and the older `User:` / `Assistant:` template; all of it is read from the checkpoint's configs.

Agreement with the official `janus.janusflow` code and diffusers' `AutoencoderKL` in float32 on
CPU (`tools/ref_janus.py` vs `examples/janus_check.rs`, oxinfer on the CPU with bf16 weights as
stored; generation from the same starting noise, 30 steps, CFG 2):

| JanusFlow-1.3B | max abs diff | rel L2 | |
|---|---|---|---|
| velocity of the first step (conditional and unconditional) | 9.1e-6 | 1.3e-6 | |
| velocities of all 30 steps | 3.0e-3 | 2.3e-5 | last step 2.2e-3, rel 6.9e-5 |
| final latents | 4.6e-4 | 1.1e-5 | |
| VAE output (384x384x3, in [-1, 1]) | 1.4e-3 | 4.0e-5 | PNG: 78.3 dB, max pixel diff 1, 428 of 442368 bytes differ |
| text logits after a 51-token prompt | 4.9e-5 | 2.6e-6 | same argmax |
| SigLIP features, 1280x960 image | 8.3e-3 | 2.4e-5 | |
| aligned image embeddings | 3.8e-3 | 2.2e-5 | |
| logits after the 630-token image prompt | 7.8e-5 | 3.9e-6 | greedy answer identical |
| the same for a 150x260 image (upscaled, padded left and right) | 5.3e-5 | 2.8e-6 | 64 greedy answer tokens identical |

**Image sizes.** Generation is 384x384 for every model of the family, a limit of the models
rather than of oxinfer: Janus-Pro was trained to write a 24x24 grid of image tokens and JanusFlow
a 48x48 latent (24x24 tokens), in raster order with one-dimensional positions, and the
reference generates nothing else. `--size WxH` (`ImageOptions::width` / `height`, the API's
`size`) therefore resamples the 384x384 image with a Lanczos filter, as the reference's demos do
for display; it adds no detail. For understanding, images of any size and aspect ratio go
through the reference processor's steps: longer side resized to 384 (antialiased bicubic, up or
down), centered on a square of the processor's background colour.

**Conversations.** `--chat` continues after the first answer with one question per line of
standard input, the images belonging to the first question; in the library `Janus::ask` takes a
`Conversation` and each turn may add images, and the C API has `oxi_janus_chat` /
`oxi_janus_chat_reset`. The turns follow the reference's template (`... Assistant: ANSWER<eos>User:
QUESTION\n\nAssistant:`). The language model's state is kept between turns; the previous answer is
read again as text, because the reference formats the whole conversation anew at every turn and
its tokens are not always the ones the model wrote (the first word loses or gains its leading
space). Checked on a second question after an image question and its greedy answer, against the
reference run on the whole two-turn conversation: 664 identical token ids for both models,
logits max diff 9.7e-5 (rel 4.7e-6) for JanusFlow-1.3B and 1.4e-4 (rel 3.0e-6) for Janus-Pro-1B.
Janus-Pro-1B checked again after these changes: guided logits over 48 teacher-forced steps
3.1e-4 (rel 6.4e-6), logits after a 150x260 image 1.2e-4 (rel 1.3e-6), same greedy answer.

Not supported: image generation at a native size other than 384x384 (a limit of the models, see
above) and the browser build. JanusFlow was checked on the CPU only; its ConvNeXt blocks always
run there.
