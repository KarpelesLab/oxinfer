# OmniGen2: image editing, in-context generation and text to image

OmniGen2 edits an image from an instruction, composes several input images ("the cat from the
first image on the porch of the cabin in the second"), and generates from text, with one 4B
diffusion transformer conditioned by Qwen2.5-VL-3B. `src/omnigen2/` follows the official
`OmniGen2Pipeline` (github.com/VectorSpaceLab/OmniGen2; diffusers has no OmniGen2 pipeline); an
`OmniGen2/OmniGen2` folder is recognised from `model_index.json`.

```sh
# edit (50 steps, text guidance 5, image guidance 2, the official negative prompt)
./target/release/oxinfer --model OmniGen2/OmniGen2 --init-image cat.png \
    -p "Make the cat wear a small red wizard hat." -o hat.png
# in-context generation (up to 5 images), quantized transformer: 2.3 GB of weights
./target/release/oxinfer --model OmniGen2/OmniGen2 --quant q4 --init-image cat.png --init-image cabin.png \
    --size 1024x768 -p "The cat from the first image sits on the porch of the cabin in the second image." -o both.png
# text to image; the bf16 Qwen2.5-VL-3B-Instruct (7.5 GB) instead of the folder's float32 copy (15 GB)
./target/release/oxinfer --model OmniGen2/OmniGen2 --text-encoder Qwen/Qwen2.5-VL-3B-Instruct \
    --guidance 4 -p "A red fox sitting in a snowy forest at sunrise" -o fox.png
# understanding: without -o, --image asks the model's Qwen2.5-VL (the mllm/ folder as a chat model)
./target/release/oxinfer --model OmniGen2/OmniGen2 --image hat.png -p "What is the cat wearing?"
```

| Flag | Meaning |
|---|---|
| `--init-image FILE` | input image (repeatable, up to 5; none: text to image) |
| `--steps N`, `--guidance F`, `--image-guidance F`, `--negative TEXT` | 50, text guidance 5, image guidance 2, the official negative prompt |
| `--size WxH` | output size; default: the single input image's size (within 1024x1024 and 1024 per side, multiples of 16), else 1024x1024. With one input image the image is resized to it, so both token grids match |
| `--quant q8\|q4` | transformer precision (the text encoder follows, at Q8 for `q4`; `--text-quant` overrides) |
| `--text-encoder SPEC` | the Qwen2.5-VL-3B weights from another folder or repo id |
| `--transformer FILE`, `--mmap` | the transformer from a GGUF file (calcuis/omnigen2-gguf); map the weights |
| `--scheduler euler\|dpmsolver++` | the sampler: Euler (default) or the official DPM-Solver++ option |
| `--teacache F`, `--taylorseer` | a step cache (off by default; both skip transformer passes and change the image): TeaCache with relative-L1 threshold `F` (the official default is 0.05), or TaylorSeer |
| `--sample-posterior` | sample the VAE posterior of the input images instead of taking its mean (off by default) |

What runs: the instruction, in the pipeline's chat template (a system turn "You are a helpful
assistant that generates high-quality images based on user instructions." and the user turn), goes
through Qwen2.5-VL-3B as text only (the input images are not shown to it), and its final hidden
states condition the transformer. The input images are resized with Lanczos within 1024x1024
pixels and 1024 per side (never up; sides floored to 16), encoded by the FLUX VAE
(`crate::flux::vae`, posterior mean) and cut into 2x2 patches. The transformer
(`OmniGen2Transformer2DModel`, 2520 wide, 21 query / 7 key-value heads of 120) runs refiner blocks
per stream (2 unmodulated over the text, 2 over the noisy target, 2 over each input image on its
own after a learned per-image embedding), then 32 joint blocks over `[text, images..., target]`:
sandwich RMSNorms, tanh-gated `1 + scale` modulation, QK RMSNorm, SwiGLU, and Omni RoPE (text at
`(i, i, i)`, each image on its own grid at axis-0 offset text length + the larger sides of the
images before it). Euler steps over `t` from 0 (noise) to 1 with `t / (m - m t + t)`,
`m = sqrt(latent pixels) / 40`, and dual guidance `uncond + image_scale * (ref - uncond) +
text_scale * (cond - ref)` (three passes per step; `ref` has the images and the negative prompt,
`uncond` neither; two passes without input images). The noise and image refiners do not depend on
the prompt, so they run once per step for the three passes.

The 2520-wide layers fit neither Q8 / Q4 blocks of 32 nor the GPU's 64-row matmul tiles: they are
stored zero-padded (inputs to 2528, outputs to a multiple of 64 on a GPU) and their activations
padded and cropped around each matmul, an exact product. Heads of 120 are padded to 128 (queries
pre-scaled) for the GPU's flash-attention kernels; key-value heads are expanded. The published
checkpoint is float32: it is mapped and converted at load (to bf16, or quantized), so its 31 GB never
sit in memory.

**Agreement** with the official pipeline (torch 2.15, transformers 5.22, diffusers 0.42; float32
weights; same prompt, input image and initial noise; the VAE posterior mean instead of a sample on
both sides; relative L2 errors, PSNR on the 8-bit output):

| oxinfer | against | first velocity (3 passes, CFG) | image PSNR |
|---|---|---|---|
| prompt embeddings (Qwen2.5-VL-3B, bf16 = the float32 values) | float32 | 8.8e-6 | |
| bf16, CPU | reference with the same bf16-rounded weights | 2.4e-6 | |
| bf16, Metal | same | 2.4e-4 | 60.0 dB (edit, 512x512, 50 steps) |
| bf16, Metal, 2 input images | same | 4.2e-4 | 52.0 dB (in-context, 512x512, 50 steps) |
| bf16, Metal, text to image | same | 3.1e-4 | |
| bf16 | float32 weights | 5.4e-2 | 24.0 dB |
| `--quant q8` (CPU / Metal) | reference with our Q8 weights dequantized | 3.4e-4 / 4.0e-4 | |
| `--quant q8` | bf16-weight reference | 5.9e-3 | 40.8 dB (text encoder Q8 too) |
| `--quant q4` (CPU / Metal) | reference with our Q4 weights dequantized | 3.2e-3 / 2.9e-3 | |
| `--quant q4` | bf16-weight reference | 8.5e-2 | 20.6 dB (text encoder Q4 too; visually the same edit) |

Quantization error comes from the weights, not from the int8 activations of the CPU kernels: with
the same dequantized weights in the reference, Q8 agrees to 3e-4 and Q4 to 3e-3. bf16 weight
rounding alone moves the guided velocity by 5% (the official pipeline runs in bf16), and the
images then differ in fine detail only. Qwen2.5-VL-3B at Q4 moves the conditioning by 43% (Q8: 5%),
so `--quant q4` keeps the text encoder at Q8. OmniGen2's `mllm/` is exactly
Qwen/Qwen2.5-VL-3B-Instruct (identical hidden states), stored as float32.

**Speed and memory** (M5 Max; 50 steps, three transformer passes per step with input images, two
without; a 512x512 edit is 1024 target + 1024 image + ~40 text tokens per pass, a 1024x1024 edit
8.2k; rows marked * shared the GPU with another job):

| run | weights | device | per pass | total | peak memory (footprint) |
|---|---|---|---|---|---|
| edit, 512x512 | bf16 (8.0 GB) | Metal | 1.6-2.4 s | 4-6 min | 11.5 GB |
| edit, 512x512 | `--quant q8` | Metal | 3.0 s | 7.5 min | 10.1 GB |
| edit, 512x512 | `--quant q4` (2.3 GB) | Metal | 2.1 s | 5.2 min | 8.4 GB; 5.8 GB with `--text-encoder Qwen/Qwen2.5-VL-3B-Instruct` |
| edit, 512x512 | bf16 | CPU, 18 threads (shared) | 32-39 s | | 10.6 GB |
| edit, 512x512 | calcuis GGUF "Q4_0" (mostly F16) | CPU matmuls | 10.5 s | | |
| edit, 1024x1024 | bf16 | Metal | 6.9 s | 17 min | 20 GB |
| edit, 1024x1024 * | `--quant q4` | Metal | 13.5 s | 34 min | 13 GB |
| text to image, 1024x1024 * | `--quant q4` | Metal | 7.8 s | 13 min | 14 GB |
| 2 images (512² + 1024²) to 1024x768 * | `--quant q8` | Metal | 12.7 s | 32 min | 15 GB |

The text encoder takes 0.5 s; the VAE 0.4 / 1.7 s to encode and 0.7 / 2.8 s to decode at 512 /
1024. Memory beyond the weights is mostly the VAE at full resolution (its mid-block attention now
runs in query chunks, which took 3 GB off a 1024x1024 encode) and, from the float32 `mllm/`, the
text encoder's bf16 copy; for small devices use `--quant q4`, `--text-encoder
Qwen/Qwen2.5-VL-3B-Instruct` (mapped bf16, quantized to Q8) and 512x512 images.

**Sample edits** (bf16, 50 steps): a 512x512 cat photo with "Make the cat wear a small red wizard
hat" gets the hat and keeps the room; a flat 1024x1024 drawing of a house with "Make it night, with
warm light in the windows and stars in the sky" turns the sky dark blue with stars, the sun dark
and the window orange; a 512x512 green meadow cabin with "Change the season to winter" is covered
in snow (the same as the reference, 60 dB); in-context, "The cat from the first image sits on the
doorstep of the house in the second image" (Q8) puts the photographed cat in the drawn doorway;
text to image (Q4), "A photo of a red fox sitting in a snowy forest at sunrise" is a sharp,
detailed fox. The model's Qwen2.5-VL answers "What is the cat wearing?" with "The cat is wearing a
red witch hat." (600 tokens/s prefill, 26 tokens/s decode, bf16 on Metal).

**Library and C API.**

```rust
use oxinfer::omnigen2::{Options, Params, Pipeline};
let mut pipe = Pipeline::load("OmniGen2/OmniGen2", Options { quant: oxinfer::model::Quant::Q4, ..Default::default() })?;
let img = oxinfer::media::Media::new().load_image("cat.png".as_ref())?;
let (out, timings) = pipe.edit(&[img], &Params::new("add a red hat"), &mut |i, n| eprintln!("{i}/{n}"))?;
// pipe.generate(&params, ..) for text to image; edit_pictures(&[oxideav Picture]) with `oxideav`
// Params::new(..).with_teacache(0.05) / .with_taylorseer(); params.scheduler = Scheduler::DpmSolver;
// params.sample_posterior = true
```

The C API has `oxi_omnigen2_load` (`key=value` options: quant, text_quant, text_encoder, mmap,
transformer, keep_text_encoder, device), `oxi_omnigen2_edit` (a JSON request: prompt, images,
negative, steps, text_guidance, image_guidance, cfg_range, seed, width, height, max_pixels,
max_side, scheduler, teacache, taylorseer, sample_posterior; plus RGB8 pixel images), `oxi_omnigen2_cancel` and `oxi_omnigen2_free` (see
`include/oxinfer.h`). For understanding, load `mllm/` as an engine.

**Weights and license.** OmniGen2/OmniGen2: Apache 2.0, ungated; 31 GB, all float32 (15.9 GB
transformer, 15.0 GB Qwen2.5-VL-3B, 0.3 GB VAE). Qwen/Qwen2.5-VL-3B-Instruct (bf16, 7.5 GB) can
stand in for `mllm/` (`--text-encoder`). calcuis/omnigen2-gguf has GGUF transformers, but its
"Q4_0" file is 6 GB: the 2520-wide layers cannot be block-quantized, so most stay F16 (they load,
on the CPU); `--quant q4` (padded blocks) is smaller and runs on the GPU.

**Scheduler, step caches, posterior sampling.** All opt-in: each changes the image. Without them
the output is bit-identical to before they existed (same PNG and final latents with one input
image, two, and none, against a build of the previous commit, on the tiny model below).

- `--scheduler dpmsolver++` is the official script's option: the repository's
  `DPMSolverMultistepScheduler(algorithm_type="dpmsolver++", solver_type="midpoint",
  solver_order=2, prediction_type="flow_prediction")` on the same timesteps, working on
  `sigma = 1 - t` with the data prediction `x + sigma * v`. Its first and last steps are first
  order (and the second in effect, its difference term being divided by an infinite log-SNR
  ratio). `src/omnigen2/sched.rs` is checked step by step against that scheduler without a
  model: `tools/ref_omnigen2_sched.py` prints its sigmas and the sample after each of 6 steps,
  which the unit test holds (sigmas within 1e-7, samples within 2e-6).
- `--teacache F` is the official transformer's `enable_teacache`: per guidance branch
  (conditional, reference, unconditional: each its own state), the relative L1 change of the
  first joint block's modulated input since the previous step goes through the fitted polynomial
  (`-5.4826 x^4 + 11.4877 x^3 - 4.4741 x^2 + 2.4773 x - 0.0332`) and accumulates; while the sum
  is under `F` the 32 joint blocks are skipped and the last computed residual (blocks' output
  minus input) is added to the input instead. The first and last steps always run. The refiners
  run on every step (the decision needs their output).
- `--taylorseer` is the official pipeline's `enable_taylorseer` (order 4, interval 3, first
  enhance 5): per branch, the blocks run on its first 5 passes and then every 3rd; the passes
  between take the Taylor expansion of the hidden states, from finite differences between the
  last runs. The official code keeps the features of all 32 layers over the whole sequence; a
  skipped pass replaces every layer's output by its own prediction, so only the last layer's
  target rows reach the output, and those are what oxinfer keeps (the same arithmetic on the
  same values, megabytes per branch instead of gigabytes). A step whose passes are all
  predictions skips the refiners too. With both flags TaylorSeer wins, as in the official
  script.
- `--sample-posterior` draws each input image's latents as `mean + std * noise` like the official
  `latent_dist.sample()`, instead of the mean. The noise comes from oxinfer's generator
  (xoshiro256** seeded from `--seed`, one stream per input image, apart from the initial
  latents'), in `[c, h, w]` order: the reference's distribution, not its draw. oxinfer does not
  reproduce torch's generator (the initial latents are compared by passing the reference's noise
  in, and so is this), and the official pipeline draws this noise from torch's global generator
  rather than the run's seeded one, so there is no draw to match bit for bit: the mean stays the
  default. The posterior is narrow: for a 256x256 cat photo its standard deviations are 1.4e-4
  of the latents (L2), and sampling moved the 30-step edit below by 66.6 dB PSNR.

Checked against the official `OmniGen2Pipeline` itself (float32, CPU) on a tiny random-weight
model (`tools/omnigen2_tiny.py`: a 4-block 96-wide transformer and a small 16-channel VAE; the
reference's prompt embeddings, initial noise and posterior noise passed to oxinfer; 64x64 output;
relative L2 error of the final latents, and whether every pass made the same run / skip
decision):

| options | input images | steps | passes that ran the blocks (of all) | same decisions | final latents |
|---|---|---|---|---|---|
| none | 1 | 12 | 36 (36) | | 9.8e-7 |
| `--teacache 0.5` | 1 | 30 | 85 (90) | yes | 7.3e-7 |
| `--teacache 1.0` | 1 | 30 | 73 (90) | yes | 7.0e-7 |
| `--teacache 2.0` | 1 | 30 | 43 (90) | yes | 9.6e-7 |
| `--teacache 1.0` | 0 (text to image) | 30 | 28 (60) | yes | 7.5e-7 |
| `--taylorseer` | 1 | 30 | 39 (90) | yes | 9.9e-7 |
| `--taylorseer` | 2 | 30 | 39 (90) | yes | 9.3e-7 |
| `--taylorseer` | 0 (text to image) | 30 | 26 (60) | yes | 9.4e-7 |
| `--scheduler dpmsolver++` | 1 | 20 | 60 (60) | | 9.4e-7 |
| `--taylorseer --scheduler dpmsolver++` | 1 | 50 | 60 (150) | yes | 9.5e-7 |
| `--sample-posterior` | 1 | 12 | 36 (36) | | 9.6e-7 (sampled reference latents: 7.8e-7) |
| `--teacache 2.0 --scheduler dpmsolver++ --sample-posterior` | 2 | 30 | 39 (90) | yes | 8.3e-7 |

The tiny model needs thresholds far above 0.05 to skip anything (its random blocks move the
modulated input a lot between steps). On the real model (`--quant q4`, CUDA, a 256x256 cat photo,
"Make the cat wear a small red wizard hat.", seed 3; PSNR of the output against the same run
without the option, which is the size of the change the option makes, not an error):

| run | steps | passes that ran the blocks (of all) | image against the plain run |
|---|---|---|---|
| `--teacache 0.05` | 50 | 105 (150) | 46.4 dB |
| `--teacache 0.1` | 50 | 71 (150) | 40.5 dB |
| `--teacache 0.05` | 30 | 90 (90): nothing skipped | identical |
| `--teacache 0.2` | 30 | 44 (90) | 37.3 dB |
| `--taylorseer` | 30 | 39 (90) | 32.3 dB |
| `--scheduler dpmsolver++` | 30 | 90 (90) | 38.8 dB |
| `--sample-posterior` | 30 | 90 (90) | 66.6 dB |

All of them give the same edit to the eye (the same hat on the same cat). The step caches were
compared with the official pipeline on the tiny model only; the real model was not run in the
Python reference for them.

Not supported: OmniGen2 LoRAs, batches, `--save-quantized` for the padded transformer layers
(quantization happens at each load, a few seconds), and the reflection / RL variants
(OmniGen2-RL, EditScore).
