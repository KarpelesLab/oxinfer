# BAGEL: image editing, text to image and image understanding in one model

ByteDance's BAGEL-7B-MoT is a single unified model: it edits images from instructions, generates
them from text, answers questions about them, and can write a plan ("thinking") before an image or
an answer. `src/bagel/` follows the official inference code (`github.com/ByteDance-Seed/Bagel`:
`InterleaveInferencer`, `modeling/bagel`); there is no diffusers pipeline. The model is picked
from the folder (`llm_config.json` + `vit_config.json` + `ae.safetensors`).

```sh
# editing: --init-image is the picture, --prompt the instruction (29.5 GB download, bf16)
./target/release/oxinfer --model ByteDance-Seed/BAGEL-7B-MoT --mmap --init-image dog.png \
    -p "Put a red knitted hat on the dog" -o hat.png
# ~4x cheaper: inputs resized to at most 512 pixels per side (the output follows the input's size)
./target/release/oxinfer --model ByteDance-Seed/BAGEL-7B-MoT --mmap --max-size 512 --init-image dog.png \
    -p "Turn this photo into a watercolor painting" -o watercolor.png
# text to image, with a written plan first (printed)
./target/release/oxinfer --model ByteDance-Seed/BAGEL-7B-MoT --mmap --think --size 512x512 \
    -p "a car made of small cars" -o cars.png
# image understanding: without -o the answer is text
./target/release/oxinfer --model ByteDance-Seed/BAGEL-7B-MoT --mmap --init-image cat.png -p "What is in this image?"
```

| Flag | Meaning |
|---|---|
| `--init-image FILE` | with `-o`: the image to edit; without: the image to ask about (repeatable: the instruction follows all images) |
| `--steps N` | timesteps (default 50: 49 model evaluations per guidance branch) |
| `--cfg-text F`, `--cfg-img F` | text guidance (4) and image guidance (2 editing, 1 = off for text to image) |
| `--cfg-interval LO,HI` | timesteps with guidance (0,1 editing; 0.4,1 text to image) |
| `--cfg-renorm T`, `--cfg-renorm-min F` | CFG-renorm: `text_channel` (editing) / `global` / `channel`; lower bound 0 |
| `--timestep-shift F` | 3 |
| `--size WxH` | text-to-image size (default 1024x1024; at most 1024 per side) |
| `--max-size N`, `--min-size N` | input resize bounds (1024 and 512, the reference's `ImageTransform(1024, 512, 16)`) |
| `--think` | plan before the image, or reason before the answer (greedy, at most 1000 tokens) |
| `--taylorseer` | TaylorSeer step cache (off by default: fewer language-model passes, a different image) |
| `--sample-posterior` | sample the VAE posterior of input images instead of taking its mean (off by default) |
| `--quant q8\|q4`, `--mmap` | weight precision (quantized at load); map the checkpoint |

What runs: the language model is Qwen2.5-7B (with q/k RMSNorm added) as a Mixture-of-Transformers:
every layer has an understanding expert (text and SigLIP tokens) and a generation expert (VAE
latent tokens) with their own norms, projections and MLPs, meeting in one self-attention over the
packed sequence. Each text is wrapped as `<|im_start|>...<|im_end|>` and goes through causally. Each
input image is resized (sides multiples of 16, 512..1024), VAE-encoded with the FLUX autoencoder
(`flux::vae`, BFL file layout; posterior mean), cut into 2x2 latent patches through `vae2llm` plus a
timestep embedding of 0 and a 2-D sin-cos position embedding (generation expert), then read again
by SigLIP so400m/14 at its own size up to 980 pixels (NaViT style: one token per patch, learned
70x70 position table, 26 layers, connector MLP; understanding expert); every token of an image has
one RoPE position and attends bidirectionally. The output image's noise latents go after the
context at the next position and are denoised by rectified flow: `steps - 1` Euler steps over
`t = 1 -> 0` shifted by `3t / (1 + 2t)`, the generation expert predicting the velocity through
`llm2vae`. Guidance has two terms, evaluated as extra branches over shared KV caches (no copies):
text (the context before the instruction) and image (the texts without the images, a separate
small context); BAGEL's CFG-renorm caps the guided velocity's norm (per token on the text term for
editing). The three branches of a step run as one batch through every matmul. Thinking and answers
come from the understanding expert's LM head, decoding after `<|im_start|>`; the plan is then
appended to the context as text before the image is generated.

**Agreement** with the official code (the repository's modules with flash-attn replaced by SDPA, same inputs and initial noise; the VAE switched to its posterior mean
as oxinfer encodes):

| check | reference | oxinfer | relative error |
|---|---|---|---|
| SigLIP features, 512x512 cat (1369 patches) | float32, CPU | CPU | 1.6e-5 (connector output 1.3e-5) |
| same | float32 | Metal (matrix units) | 9.3e-4 |
| LM logits after an image question, 2 layers | float32 | CPU | 1.9e-6 (KL 8e-8) |
| first denoising step, editing, 2 layers: conditional / text-free / image-free / guided velocity | float32 | CPU | 8.7e-6 / 7.9e-6 / 7.1e-7 / 2.5e-5 |
| LM logits after an image question, full model | bf16 (CPU autocast) | CPU | 6.6e-2, KL 3.2e-4, same top 5, same 40 greedy tokens |
| first step velocity, full model, editing (cond / guided) | bf16 (MPS) | CPU | 1.3e-2 / 3.0e-2 |
| same | bf16 | Metal | 1.3e-2 / 3.1e-2 (Metal vs CPU: 8.9e-4 / 3.3e-3) |
| same, `--quant q8` / `q4` | bf16 | Metal | 1.3e-2 / 3.9e-2 (vs oxinfer bf16: 2.8e-3 / 3.7e-2) |
| edited image, 50 steps, cat 512x512 "Put sunglasses on the cat" | bf16 | Metal | 41.4 dB PSNR |
| text to image, 512x512 fox | bf16 | Metal | 44.0 dB PSNR |

The truncated float32 runs check the implementation (the f32 reference needs no 58 GB); the
full-model numbers are bf16-activation (reference) against f32-activation (oxinfer) rounding, and
the images still agree to 41-44 dB. Greedy thinking text follows the reference for its first ~30
tokens and then picks another near-tie wording (different plan, different image). Quantized
weights: q8 moves the first velocity by 0.28% against oxinfer's bf16 (the 1024-row matmuls run on
dequantized weights, so activations are never rounded to int8 here), q4 by 3.7%.

**Speed and memory** (M5 Max, machine shared with other jobs; 50 timesteps; an edit runs three
branches per step):

| task | weights | device | per branch pass | total | peak footprint |
|---|---|---|---|---|---|
| edit, 512x512 input (1024 latent + 2404 context tokens) | bf16 (`--mmap`) | Metal | 1.07 s | 2.9 min (147 passes) | 5.5 GB |
| same | `--quant q8` | Metal | 0.99 s | 2.8 min | 20 GB |
| same | `--quant q4` | Metal | 1.01 s | 2.8 min | 13 GB |
| same | bf16 (`--mmap`) | CPU, 18 threads | 18 s | ~45 min | 3.1 GB |
| same | `--quant q4` | CPU, 18 threads | 20 s | ~50 min | 11.6 GB |
| edit, 1024x768 input (3072 latent + 6727 context tokens) | bf16 (`--mmap`) | Metal | 3.4-3.6 s | 9.3 min | 10.4 GB |
| text to image, 512x512 | bf16 (`--mmap`) | Metal | 1.08 s | 1.7 min (90 passes) | 3.9 GB |
| text to image, 1024x1024 | bf16 (`--mmap`) | Metal | 4.1 s | ~6 min (~90 passes) | 15.4 GB |

The official code in PyTorch 2.15 on the same GPU (MPS, bf16) takes 98 s for the 512x512 edit and
109 s for the 512x512 text to image. The context (VAE + SigLIP + text) takes 5-6 s at 512x512 and
17-29 s at 1024x768 on Metal, 46 s at 512x512 on the CPU; the VAE decodes in 0.8 s (512) to 3 s (1024x768).
A 100-token answer about a 512x512 image takes 11.7 s on Metal (bf16), 6.9 s with `--quant q4`,
33 s on the CPU (single-token passes run on the CPU; about 10 tokens/s in bf16). Without `--mmap` the
whole 29 GB checkpoint is read into memory first (quantizing then keeps both: 41 GB at q4), so
`--mmap` is the way to run on smaller machines. At 1024x1024 the activations of the three
batched guidance branches (3 x 4098 tokens) dominate the footprint.

**Sample edits** (bf16, Metal): a golden retriever on a porch, "Put a red knitted hat on the dog",
gets a red beanie and nothing else changes; a green meadow with a cabin, "Make it winter: snow on
the roof and the ground", snows over roof and field; a bakery front, "Change the text on the sign from 'FRESH BREAD' to
'OXINFER CAFE'", rewrites the carved sign in the same style but spells it "OXIVFR CAFE" (BAGEL's
text rendering is its weak spot); "Turn this photo into a watercolor painting" repaints the whole
porch scene. `--quant q8` reproduces the bf16 edit at 51 dB (41.3 dB against the reference), q4 at
36 dB.

**Library and C API.**

```rust
use oxinfer::bagel::{Bagel, ImageOptions, LoadOptions, TextOptions};
let m = Bagel::load("ByteDance-Seed/BAGEL-7B-MoT", &LoadOptions { mmap: true, ..Default::default() })?;
let img = oxinfer::media::Media::new().load_image("dog.png".as_ref())?;
let (out, stats) = m.edit(&img, "put a red hat on the dog", &ImageOptions::default(), &mut |i, n| { eprintln!("{i}/{n}"); true })?;
let (fox, _) = m.generate("a red fox in the snow", &ImageOptions { width: Some(512), height: Some(512), ..Default::default() }, &mut |_, _| true)?;
let answer = m.understand(Some(&img), "What breed is this dog?", &TextOptions::default(), &mut |_| oxinfer::generate::Flow::Continue)?;
// generate_interleaved / understand_interleaved take any sequence of texts and images;
// edit_picture / understand_picture take oxideav pictures (`oxideav` feature)
```

`examples/bagel_edit.rs` edits an image and then describes the result with the same model. The C
API has `oxi_bagel_load` (`key=value` options: quant, mmap, device, cache_dir, generation,
understanding, text), `oxi_bagel_edit_rgb` / `oxi_bagel_edit_file` / `oxi_bagel_generate` (JSON
options as `ImageOptions`), `oxi_bagel_understand_rgb` / `_file` (streamed text),
`oxi_bagel_last_thinking`, `oxi_bagel_cancel` and `oxi_bagel_free` (see `include/oxinfer.h`).
Loading only what a task needs (`generation=0` for questions, `understanding=0` for text to image)
leaves the other expert's weights untouched in the mapped file.

**Weights and license.** ByteDance-Seed/BAGEL-7B-MoT: Apache 2.0, ungated; `ema.safetensors`
29.2 GB (bf16: both experts of the 7B language model, SigLIP, connectors) and `ae.safetensors`
0.34 GB (the FLUX.1 autoencoder, f32). No GGUF conversion of BAGEL exists; the community NF4 /
INT8 / FP8 / DFloat11 checkpoints are not supported (quantize at load with `--quant`).

**TaylorSeer and posterior sampling.** Both opt-in: they change the image. Without them the
output is bit-identical to before they existed (same PNG for an edit and a text-to-image run
against a build of the previous commit; real weights, the language model cut to 2 layers).

`--taylorseer` (`"taylorseer": true`) is the reference's `enable_taylorseer` (order 6, interval
3, first enhance 5): each guidance branch has its own state; it runs the language model on its
first 5 passes and then every 3rd, and the passes between take the Taylor expansion of the
hidden states, from finite differences between the last runs. The reference keeps the features
of every layer; on a skipped pass each layer returns its own prediction, so only the last
layer's latent-token states (before the final norm) reach `llm2vae`, and those are what oxinfer
keeps. Branches that run on the same step still share one batch. Against the official code
(`tools/ref_bagel_taylorseer.py`: the repository's model with the real weights, cut to 2 layers
and run in float32 on the CPU, its initial noise passed to oxinfer run with the same cut;
text to image, 256x256):

| run | steps | passes that ran the model (of all) | same decisions | final latents (relative L2) |
|---|---|---|---|---|
| no cache | 12 | 20 (20) | | 7.8e-6 |
| `--taylorseer` | 30 | 24 (53) | yes | 5.0e-5 |
| `--taylorseer --cfg-img 1.5` (three branches) | 50 | 53 (131) | yes | 9.6e-5 |

The larger error with the cache is float32 rounding through the order-6 differences, on the
same schedule. The full 28-layer model was not run with the cache here, and the editing path
was not compared with the reference (it differs from text to image only in the contexts the
branches attend to).

`--sample-posterior` (`"sample_posterior": true`) draws each input image's VAE latents as
`mean + exp(0.5 logvar) * noise`, as the reference's `DiagonalGaussian`, instead of the mean.
The noise comes from oxinfer's generator (xoshiro256** seeded from `--seed`, one stream per
input image, apart from the initial latents'): the reference's distribution, not its draw, which
comes from torch's global generator and cannot be matched, so the mean stays the default. The
sampling code (`src/vae_posterior.rs`) is the one checked against diffusers on OmniGen2's tiny
model above, here without diffusers' log-variance clamp, which the BFL autoencoder does not
have; BAGEL's own sampler was not compared with the reference. For a 256x256 cat photo the
posterior's standard deviations are 1.4e-4 of the latents (L2).

Not supported: batches, interleaved outputs past one image or one answer, the browser build (the
weights exceed 4 GB). Single-token text decoding runs on the CPU (about 10 tokens/s in bf16).
