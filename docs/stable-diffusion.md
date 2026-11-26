# Stable Diffusion: image generation and editing

## Image generation

A diffusers folder (one with `model_index.json`) switches oxinfer to text-to-image with Stable
Diffusion 1.x / 2.x, which includes SD 1.5, SD 2.1 and the one-step SD-Turbo, and with Stable
Diffusion XL (see [below](#stable-diffusion-xl)) (`src/sd/`):

```sh
./target/release/oxinfer --model models/sd-turbo -p "a red fox in a snowy forest" -o fox.png
./target/release/oxinfer --model models/sd-v1-5 -p "a watercolor lighthouse" --negative "blurry" \
    --steps 25 --guidance 7.5 --sampler euler_a --size 512x512 --seed 7 -o lighthouse.png
```

With `--init-image FILE` the same models edit an image instead (image-to-image, inpainting with
`--mask`, instruction edits with InstructPix2Pix): see [Editing images](#editing-images).

Turbo models (recognised by name) default to 1 step without guidance, others to 25 Euler-ancestral
steps at guidance 7.5. The CLIP tokenizer and text encoder, the UNet (ResNets, spatial transformers
with GEGLU, skip connections), the VAE decoder and the Euler samplers are implemented here.
Activations are channels-last, so every convolution unfolds into the same matmul the language models
use: the SIMD kernels apply on every CPU, and GPU backends run convolutions directly as an implicit
GEMM (`Backend::conv2d`: the kernel gathers each pixel's neighbourhood itself; Metal and CUDA
measured, WebGPU in the browser). Weights are converted from fp16 to bf16 at load.

`tools/ref_sd.py` runs diffusers on the same noise and compares every stage (`OXINFER_SD_DUMP=dir`).
With the weights rounded to bf16 on both sides the outputs agree to about 1e-6 (79 dB PSNR on
the SD-Turbo image, 75 dB on SD 1.5 after 4 guided steps), so the only difference from the fp32
reference is the bf16 weight storage (47 dB).

512x512 on the M5 Max:

| | Text | UNet pass | VAE decode | SD-Turbo image | SD 1.5, 25 steps + guidance |
|---|---|---|---|---|---|
| CPU only (18 threads) | 0.08 s | 0.91 s | 2.87 s | 3.9 s | |
| Metal, op by op (`gpu_pass=0`) | 0.08 s | 0.37 s | 0.66 s | 1.1 s | 16 s |
| Metal, whole passes (default) | 0.07 s | 0.05 s (batch of 2: 0.10 s) | 0.10 s | 0.3 s | 2.8 s |
| Metal, + CLIP as one device pass (default) | 0.01 s | | | | |
| Browser (Chrome, WebGPU, one CPU thread) | 0.2 s | 1.1 s | 1.9 s | 3.2 s | |

**Whole passes on Metal.** On a GPU with matrix units (M5, A19), the CLIP text encoders, every
UNet pass and the VAE decode run as one device plan (`src/gpu_unet.rs`, `src/metal_unet.rs`): the model describes the
pass as a graph of convolutions, concatenations and transformer blocks, the activations stay in
device buffers between ops and between steps, and the guidance branches of a step run as one batch
(both CFG passes in one, InstructPix2Pix's three in one). Activations are f32 in a padded
channels-last layout (each image with a border of zero pixels), so a 3x3 convolution is nine
shifted GEMMs accumulated into the same tile on the matrix units (`matmul2d`, half inputs, bf16
weights, f32 sums), with the group norm and SiLU applied while the half inputs are written;
strided convolutions and narrow inputs (the 4 latent channels) go through an unfolded copy.
The GEMM tiles are 128 rows x 64 outputs (64 x 64 where 128-row tiles would leave the GPU's
cores with fewer than 128 threadgroups). Attention is a fused flash kernel over heads padded to a
multiple of 16 (48 for SD 1.5's 40, 64, 80, 128, 160; a VAE's single head of 512 in value chunks
of 128) in which each SIMD-group owns 16 or 32 queries and keeps the scores, their exponentials,
the probabilities (fed straight back as the left operand of the second product) and the output
in its matrix-unit registers, with the row maxima and sums in row-reduction tensors; an online
softmax in base 2 whose running row maximum moves only when a block's scores pass it by more
than 5 (in log2 units), so most key blocks skip rescaling the output; the probabilities, at most
2^5, go to half scaled by 2^10, so that the thousands of tiny ones of a long row keep their
precision. The feed-forward's first GEMM computes the value and gate outputs of a tile side by
side and stores the GEGLU's half product directly. The text's cross-attention keys and values depend on the prompt
only: the first pass of a generation computes them into buffers the session keeps, the other
steps read them (the session compares the half-precision contexts of the batch with the ones
they were computed from, so a new prompt recomputes them); the op-by-op path below keeps them
per prompt in the UNet the same way (`Unet::clear_text_cache` frees them; the pipeline does at
the end of each generation). Both give the same output bit for bit. Layer norms, GEGLU, residual adds, skip concatenation and 2x
upsampling are fused into row kernels or the GEMMs' stores. Inputs without a norm are scaled by
a power of two when their largest magnitude passes half's range (the SDXL VAE's activations
do), and the products scaled back. `gpu_pass=0` (`Pipeline::set_gpu_pass(false)`, the
C API's and the server's load option, or `OXINFER_SD_GPU_PASS=0`) keeps the op-by-op path below
(`OXINFER_CLIP_DEVICE_PASS=0` only for the text encoders);
`OXINFER_GPU_CHECK=1` and exact mode (`OXINFER_METAL_MMA=0`) use it too. FLUX.1's VAE decoder is
the same code, so it takes the same plan: 1024x1024 FLUX.1-schnell decodes in 0.57 s instead of 2.9 s
(66.9 dB against the op-by-op decode of the same latents; mflux, on MLX, takes 0.5-0.7 s). The
VAE encoders (SD's for image-to-image and inpainting, FLUX's for its edits) run the same way:
SD 1.5 at 512x512 0.44 -> 0.10 s (posterior mean 6.6e-4 from the op-by-op one, the image after
10 img2img steps 58.3 dB), FLUX.1 at 1024x1024 1.75 -> 0.28-0.34 s (latents 1.3e-2 apart, the
image after a 2-step edit 52.8 dB; SD's encoder on the same 1024x1024 picture 3.5e-3).

The text encoders' plan (`src/metal_t5.rs`, shared with FLUX.1's T5 and CLIP) keeps the residual
stream in f32 on the device, runs the projections on the matrix-unit GEMMs (scaled half inputs)
and the causal attention on f32 operands: SD 1.5's text context 0.07 -> 0.01 s (3.9e-4 from the
layer-by-layer encoder, the 20-step image 56.4 dB), SD-Turbo's OpenCLIP-H 0.10 -> 0.02 s (6.7e-4,
58.6 dB), SDXL-Turbo's two encoders 0.19 -> 0.05 s (context 1.2e-4, pooled 2.4e-4, image 51.5 dB).

**Why FLUX.1's encoder is further off.** Against the encoder in float32 on the CPU (1024x1024,
the same picture), stage by stage (both paths stopped after each down block and around the middle
block in a debugging build; relative L2 of the activations):

| After | down 0 | down 1 | down 2 | down 3 | mid ResNet 1 | mid attention | mid ResNet 2 | latent means |
|---|---|---|---|---|---|---|---|---|
| FLUX.1 VAE | 1.2e-3 | 2.5e-3 | 1.6e-3 | 1.4e-3 | 2.9e-3 | 4.6e-3 | 5.1e-3 | 9.3e-3 |
| SD 1.5 VAE | | | | | | | | 2.4e-3 |

Nothing in the plan is wrong: each op's own error is that of rounding its GEMM inputs to half.
The middle block's attention run in float64 on the plan's own input reproduces the plan's output
to 1.6e-5, and its conv_out with half inputs adds 6e-5. What grows is the error coming in. The
down blocks' half inputs leave 2.9e-3 by the middle block. Its single-head attention over the
16384 latent pixels then amplifies that input error to 4.6e-3. That attention is extremely peaked
in FLUX.1's VAE: scores reach 480 (the row maximum sits 26 above the median on average), against
58 in SD 1.5's. The output norm and convolution amplify it again in the 16 mean channels, which
are small next to the log-variance ones (moments overall 2.0e-3, means 9.3e-3; against the
op-by-op Metal path, itself rounded to half, the 1.3e-2 above). Running these GEMMs at f32
precision (hi / lo half pairs) would about double the encoder's GEMM time. The edits only see a 52.8 dB difference, so the
half-precision encoder stays. `OXINFER_METAL_MMA=0` gives the exact one.

Against the op-by-op path (512x512, same seed): the first guided SD 1.5 UNet output differs by
5.2e-4 (relative L2), the image after 20 Euler steps at guidance 7.5 by 53.4 dB (8-bit PSNR; the
latents 4.2e-3), the SD-Turbo image by 52.7 dB, SDXL-Turbo (2 steps) 50.1 dB with a first UNet
output 2.3e-4 apart; the VAE alone (same latents) 66 dB. That is the half-precision rounding of
the GEMM inputs (diffusers' fp16 inference rounds every activation), below the 47 dB the bf16
weight storage already costs against the float32 reference.

Against [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) (master 82c19c8,
Metal, flash attention, cond and uncond in one batch) on the same fp16 weights, M5 Max, 2026-11-26
(the GPU shared with other jobs: ranges over interleaved runs):

| | oxinfer | stable-diffusion.cpp |
|---|---|---|
| SD 1.5 512x512, 20 Euler steps, guidance 7.5: sampling | 2.05-2.90 s (0.10-0.15 s/step) | 4.72-6.02 s (0.24-0.30 s/step) |
| text encoder | 0.06-0.07 s | 0.43 s |
| VAE decode | 0.10 s | 1.21-1.26 s |
| SD 1.5 768x768, same: sampling | 6.77-7.63 s (0.34-0.38 s/step) | 12.3-16.7 s |
| VAE decode | 0.28 s | 2.39-2.50 s |
| SD-Turbo 512x512, 1 step: text | 0.10-0.13 s | 0.24 s |
| UNet | 0.09 s (the first pass, with the session's setup) | 0.15-0.23 s |
| VAE decode | 0.10 s | 1.17-1.19 s |

Before this (op by op, CFG branches one after the other): SD 1.5 0.60 s/step, VAE 0.56 s. A second
round took the guided SD 1.5 step's GPU time from 98.1 to 94.7 ms (q / k / v projections, and the
cross-attention's query, key and value, encoded without barriers between them) and measured
these at under 1 ms each: heads of 40 padded to 48 instead of 64, padded score / probability
rows in threadgroup memory (kept), group-norm sums shifted by the group's first value (kept, for
the precision). The step's GPU time splits roughly as convolutions 33 ms, attention 29 ms, the
transformer projections and feed-forward 22 ms, group norms 4 ms (`OXINFER_UNET_SKIP=conv3,flash,
lin,qkv,gn,...` leaves kinds of dispatches out to measure them).

A third round (November 2026; GPU time per guided SD 1.5 512x512 step, `OXINFER_PF_PROFILE=1
OXINFER_UNET_TIME=1` prints each pass's): **94.8 -> 65.5 ms**, SDXL at 1024x1024 467 -> 361-373 ms.
End to end (release builds interleaved, the GPU shared with a running `oxinfer serve`): SD 1.5
512x512, 20 guided Euler steps, sampling 1.93-2.44 -> 1.33-1.89 s (0.067-0.095 s a step;
stable-diffusion.cpp 4.33-5.35 s on the same runs), SDXL base 1024x1024 9.4-10.0 -> 7.2-7.7 s
(0.36-0.39 s a step; stable-diffusion.cpp 0.83-1.04 s), text 0.06 -> 0.02 s and 0.18 -> 0.04-0.05
s with the text encoders' plan (`src/metal_t5.rs`), the VAE decode unchanged.

- **GEMM tiles.** The 128 x 128 output tiles taken when the outputs divide by 128 ran at 60-75% of
  the speed of 128 x 64 tiles on every UNet shape (`metal::unet::tests::un_gemm_bench`, ignored:
  8192 rows x 320 x 9 taps 0.47 -> 0.29 ms, 2048 x 640 x 9 0.44 -> 0.26 ms, 2048 x 640 -> 5120 0.42
  -> 0.33 ms; 52-60 TFLOP/s now), and on the lowest levels (512 rows x 1280 x 9) 64 x 64 tiles
  beat both (0.66 -> 0.25 ms): 94.8 -> 73.7 ms, the same output bit for bit (each output's sums
  run over the same K in the same order). The prefill GEMMs over bf16 weights had the same
  choice: on the matrix units their 128-token tiles now take 64 outputs too (FLUX.1 [schnell] in
  bf16 at 1024^2 1.95 -> 1.7 s per step, Qwen3.5-4B bf16 prefill of 1800 tokens 3340-3530 ->
  3730-3810 tok/s, the same results).
- **Lazy softmax** (above): 73.7 -> 67.5 ms. The SD 1.5 image moves by 56.6 dB; the SDXL base
  image against diffusers in float32 on the same noise is 44.5-44.7 dB, against 45.1 dB before
  (the same difference as from exp2 alone, i.e. trajectory noise).
- **Probabilities over the scores:** the half probabilities of a key block now overwrite its f32
  scores in threadgroup memory (each pair of lanes reads its row into registers first, and the
  pair runs in one SIMD-group) instead of a buffer of their own: 26 -> 17 KB a threadgroup,
  67.5 -> 65.5 ms, the same output. FLUX.1's attention kernels (`flash_attn128_mma`,
  `dit_flash128_hilo`) do the same (24 -> 16 KB; no change in FLUX.1's step time measurable on
  the shared GPU, the same images).
- **Text keys and values once per generation:** 94.8 -> 94.6 ms. They were 32 small GEMMs per
  step, already encoded without barriers beside the query; the op-by-op path saves the same 1%.

Left after it (min of interleaved runs, GPU shared): convolutions 21 ms (0.93 TFLOP per step at 45
TFLOP/s), attention 17-19 ms (0.3 TFLOP with the heads padded to 48: score bound, about 80 G scores
a second), the transformer projections and feed-forward 11 ms (0.34 TFLOP at 31 TFLOP/s: K of
320-1280), the group norms 4.6 ms, the attention's q / k / v projections 4 ms.

A fourth round (November 2026, same measure): **65.5 -> 47.7 ms** per guided SD 1.5 512x512 step,
SDXL at 1024x1024 353-372 -> 289-304 ms (min / median of interleaved runs on a shared GPU). End to
end, 20 guided steps of SD 1.5 sample in 1.1-1.2 s (1.3-1.6 s before on the same runs), SDXL base
in 5.9 s (7.4-7.6 s).

- **Flash attention in registers** (`un_flash_body`): the scores of a key block came out of the
  matrix units into threadgroup memory, two lanes per row read them back for the softmax and wrote
  the half probabilities over them for the second product, with three threadgroup barriers per
  block. Now each of the 4 SIMD-groups of a threadgroup runs alone over its own queries
  (`execution_simdgroup` matmuls): `reduce_rows` gives the block's row maxima and sums from the
  score tensor, the exponentials are taken in place (`map_iterator` reads each element's row
  maximum), the probabilities are converted to a half destination tensor and handed to the
  second product with `get_left_input_cooperative_tensor`, and only the rescaling factors of a
  block where a maximum moved (and the final 1 / sum) pass through threadgroup memory, MQ floats
  inside the SIMD-group. Measured kernels (`metal::unet::tests::un_flash_bench`, ignored; 2
  images): SD 1.5's 4096 x 4096 x 8 heads of 40 2.56 -> 1.15 ms (105 -> 233 G scores/s), its
  77-key cross-attention 0.14 -> 0.05 ms, 1024 x 1024 x 8 heads of 80 0.29 -> 0.14 ms, SDXL's
  4096 x 4096 x 10 heads of 64 3.4 -> 1.55 ms, the VAE's single head of 512 over 4096 pixels 2.17
  -> 1.8 ms. Queries per SIMD-group and keys per block are chosen per width (32 and 64 for heads
  of 48 over many keys, 32 and 32 for 64, 16 and 64 for the others and for the text's 77 keys;
  32-query tiles at 80 wide, 64-key blocks at 64 wide or 32-key blocks at 80 and 160 were
  slower). Against the softmax in float64 on the same half operands every variant is 2.6-3.1e-4
  apart (the half output's rounding), the old kernel as well. Keeping the queries in a left-input
  cooperative tensor loaded once instead of reading them each block was slower (spills).
  54.4 ms per step.
- **GEGLU in the GEMM** (`un_gemm_geglu`): the feed-forward's first GEMM wrote f32 rows of
  [value | gate] (8192 x 2560 floats at the top level, 84 MB) that a row kernel read back for the
  half GEGLU. One 64 x 64 tile now accumulates its value outputs and the matching gate outputs
  side by side and stores `value * gelu(gate)` as half: 0.42 -> 0.29 ms (8192 x 320 -> 2 x 1280),
  0.32 -> 0.27 ms (2048 x 640 -> 2 x 2560), and the 84 MB scratch buffer is gone.
- **Group-norm statistics** spread over channels: a threadgroup per chunk of 32 pixels (more for
  a VAE's full-size tensors, at most 256 chunks an image), image and block of 256 channels, a
  thread per channel, the partial sums per channel; the second stage sums a group's channels and
  chunks with 64 threads. Before, a chunk of 64 pixels was one threadgroup for all channels, which
  left the 8x8 and 16x16 levels with 2-8 threadgroups: the norms 4.3 -> 1.3 ms a step.

The SD 1.5 image after these against diffusers in float32 (`tools/ref_sd.py`, same noise, 20
Euler steps at guidance 7.5): 37.0 dB before, 36.8 after the attention, 36.7 after all three (the
op-by-op path: 36.9 dB), the first guided UNet output 6.77e-4 -> 6.72e-4 from the reference, i.e.
the differences are the trajectory's; SDXL base against diffusers' float32 image 45.2 -> 44.8 dB
(44.5-45.2 over the earlier rounds). Left: convolutions 21 ms (now 40% of the step; 52-59
TFLOP/s on the shapes above 8x8, 29 TFLOP/s at 8x8, where 1280-channel 3x3 convolutions over 200
padded rows give 80 threadgroups), the projections and feed-forward 9 ms, attention 7.6 ms, the
q / k / v projections 3.5 ms, group norms 1.3 ms.

A fifth round, on the convolutions (November 2026, same measure): **47.7 -> 43.1 ms** per guided
SD 1.5 512x512 step (min 42.9), SDXL at 1024x1024 286-288 -> 276-281 ms, the SD 1.5 VAE decoder
62.0 -> 60.0 ms.

- **Tiles by waves** (`plan` in `src/metal_unet.rs`): the M5 Max GPU runs 160 GEMM threadgroups
  at once (4 per core, either tile size), and a tile takes about as long alone as beside others
  (1280 outputs over 11520 inputs: 20-160 threadgroups of 64 rows all take 0.19-0.25 ms, 180 take
  0.32, 200 take 0.41). A convolution over padded rows leaves a last, partial wave that costs a
  whole one: 8712 rows x 320 outputs at the 64x64 level are 345 tiles of 128 rows (2.16 waves run
  as 3), 2312 x 640 at 32x32 are 190 (1.19 as 2), and the 8x8 level's 80 tiles filled half a
  wave. Now the tiles past the last full wave are split along their inputs (taps x channels) into
  parts that fill it (6 parts of 512 inputs for the 25 left at 64x64), each part stores an f32
  block, and `un_gemm_red` sums them and applies the store (bias, residual, layout); the choice
  between 128-row and 64-row tiles and the split come from a wave count estimate (the GPU's core
  count from IOKit's `gpu-core-count`; `OXINFER_GPU_CORES` overrides it, `OXINFER_UNET_SPLIT=0`
  turns splitting off). Measured (`un_gemm_bench`, kernel `plan`): 8712 x 320 x 9 -> 320 0.307 ->
  0.289 ms, 2312 x 640 x 9 -> 640 0.330 -> 0.312 ms, 200 x 1280 x 9 -> 1280 0.193 -> 0.142 ms.
- **Small levels unfolded:** 3x3 convolutions over inputs whose borders would be more than a fifth
  of the padded rows (16x16 and 8x8 in SD 1.5: 648 rows for 512 pixels, 200 for 128) run on
  unfolded rows instead (`un_im2col`, now 4 halves a thread), so no products are spent on
  borders and the split tiles fill the waves: 128 x 11520 -> 1280 in 0.088 ms (0.19 ms on the
  padded rows), 512 x 11520 -> 1280 in 0.25 ms (160 tiles of 64 rows, one wave; 0.37-0.41 ms on
  the 648 padded rows). `OXINFER_UNET_UNFOLD=N` unfolds the 3x3 convolutions of at most N pixels
  instead (unfolding 32x32 too gave nothing measurable on SDXL).
- **Cooperative stores** for f32 row destinations (projections, residual adds, the padded rows of
  convolutions): the tile's bias and residual (loaded as a cooperative tensor) are added in its
  registers and the tile stored with one `store` instead of element by element. 8192 x 320 ->
  2560 ran at 29 TFLOP/s with the per-element stores, 58 with the cooperative one; in the pass
  47.7 -> 47.2 ms. The padded outputs' border rows now hold the products of zero rows instead of
  zeros: every reader of a padded tensor (the half prep, the norms' statistics, the unfold, the
  output copy) takes only its interior.

The SD 1.5 image against diffusers in float32 (`tools/ref_sd.py`, same noise, 20 Euler steps at
guidance 7.5): 36.8 dB (36.7 before); SDXL base 45.2 dB (44.8). Left per SD 1.5 step: 3x3
convolutions over padded rows 10.5 ms, unfolded ones 6.9 ms (plus their copies), 1x1 convolutions
1.3 ms, the projections and feed-forward 9.1 ms, attention 7.6 ms, q / k / v 3.5 ms.

A sixth round, on the transformers' short GEMMs (November 2026, same measure): **43.1 -> 39.4
ms** per guided SD 1.5 512x512 step (min 39.2), SDXL at 1024x1024 276-292 -> 260-264 ms, the
SD 1.5 VAE decoder 60.1-60.3 -> 59.4-59.9 ms.

- **Lean 1x1 GEMMs** (`un_gemm_lean_body`, `Cx::lean`): the projections, q / k / v and the
  feed-forward's second GEMM ran through the convolutions' kernel, whose general epilogue (split
  parts, per-image biases, the input scale, a residual's registers held whether or not one is
  added) cost these short products (320 to 1280 inputs) a third of their time: 8192 x 320 ->
  320 took 0.067 ms, 0.041 ms without it (`un_gemm_bench`, warm GPU). They now run a kernel with
  only a bias and, as a template choice, a residual, in 64-row tiles up to 1280 inputs.
- **q / k / v in one dispatch, stored whole:** the three projections of a self-attention run as
  one grid (its y picks the weights and the destination), and their half heads are stored with
  one cooperative store per tile instead of element by element. Heads that the flash kernels
  pad (SD 1.5's 40 -> 48) take padded copies of their weights (each head's rows then zero rows,
  made once per session, 0.25 MB a matrix; SDXL's heads of 64 need none), so the padding comes
  out of the GEMM. 8192 x 320 -> 3 x 384 in 0.109 ms against about 0.2 for three. q / k / v
  3.5 -> 2.3 ms a step; the cross-attention's text keys and values (once per generation) go the
  same way.
- **Fewer passes:** the GEGLU's half tile is stored at once (0.29 -> 0.255 ms at 8192 x 320 ->
  2 x 1280); the last block's feed-forward writes proj_out's half input directly (the f32 to half
  pass is gone); proj_out and the output projection of an attention store their padded rows with
  one cooperative store when an image row is a multiple of 64 pixels (a 64-row tile is one image
  row); the convolutions over padded rows skip the first image's top border row and the last
  one's bottom one (fewer tiles: 0.3 ms a step); a convolution's input scale (`un_amax`) goes to
  a slot of its own, zeroed by the host before the pass, instead of a clearing dispatch.

Tried and dropped: unfolding straight from the f32 input with the norm applied on the way (no
`un_prep` pass for the unfolded convolutions: no faster, each input is normalized for each of the
nine taps), flash attention over heads of 40 (the matrix units want products over multiples of
16; a dynamic one ran 6% slower than the padding to 48), unfolding the 32x32 level too (0.1 ms,
within noise). The SD 1.5 image against diffusers in float32: 36.3 dB (36.8 before; the first
guided UNet output 6.71e-4 -> 6.74e-4 from the reference, so the difference is the trajectory's:
the intermediate builds gave 36.8 and 37.4 dB for changes of rounding order only); SDXL base 45.1
dB (44.9-45.2). Left per SD 1.5 step: 3x3 convolutions over padded rows 10.2 ms, unfolded ones
5.4 ms plus 0.8 ms of unfolding copies, the projections and feed-forward 9 ms, attention 7.4 ms,
q / k / v 2.3 ms, the norms' preparation passes 1.7 ms, group-norm statistics 1.3 ms, layer norms
0.8 ms.

Op by op (other GPUs, or `gpu_pass=0`), convolutions, attention and the group and layer norms run on the GPU: each norm is fused
into the convolution or projection that consumes it (a statistics kernel, then normalization and
SiLU applied while the input is loaded), and attention runs its two products as GPU matmuls with a
softmax kernel between them. What stays on the CPU is element-wise: residual adds, the GEGLU gate,
upsampling and skip concatenation. `OXINFER_GPU_CHECK=1` recomputes every device op on the CPU and
prints the worst relative error per kind (about 4e-6 for attention, 1e-6 for the fused norms, on
Metal and WebGPU). SDXL runs through this pipeline (see [Stable Diffusion XL](#stable-diffusion-xl));
SD 3 / 3.5 and FLUX.1 have their own pipelines (see
[Stable Diffusion 3 / 3.5](sd3.md#stable-diffusion-3--35-text-to-image-and-image-to-image) and
[FLUX.1 Kontext](flux.md#flux1-kontext-image-editing-from-instructions-and-flux1-text-to-image)).

**LoRA.** `--lora FILE[:SCALE]` (repeatable, scale 1 by default) merges a LoRA into the UNet and
text-encoder weights at load, `W += scale * alpha / rank * up @ down` in f32 before the bf16
conversion, so generation runs at the base model's speed (`src/sd/lora.rs`). Convolution LoRAs
(LoCon: a `[rank, in, 3, 3]` down and a `[out, rank, 1, 1]` up) are merged the same way. Both
common layouts are read: kohya / A1111 (`lora_unet_down_blocks_0_..._to_q.lora_down.weight`,
`.alpha`, `lora_te_text_model_...`) and diffusers / PEFT (`unet.<module>.lora_A.weight`, `.lora.down`,
`text_encoder.`, with the alpha from the `lora_adapter_metadata` header). Keys that match no weight
of the model, and LyCORIS kinds other than plain LoRA / LoCon (LoHa, LoKr, DoRA), are skipped and
reported; a file that matches nothing, or whose shapes belong to another base model, is an error.
`FILE` may also be a Hugging Face path, `org/name/file.safetensors` (or `org/name` when the repo
holds one safetensors file), fetched into the cache like models; or download it yourself:

```sh
curl -L -o lcm-lora.safetensors \
    https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/resolve/main/pytorch_lora_weights.safetensors
./target/release/oxinfer --model models/sd-v1-5 --lora lcm-lora.safetensors \
    --steps 4 --guidance 1 -p "a red fox in the snow" -o fox.png  # Euler ancestral (default)
./target/release/oxinfer --model models/sd-v1-5 --lora KoRo8888/ufotable_artstyle_locon_v1:0.8 \
    -p "a girl standing in a forest, anime screencap" -o forest.png
```

Against diffusers' `load_lora_weights` + `fuse_lora` (`tools/ref_sd.py --lora`, bf16 weights on
both sides), on SD 1.5: LCM-LoRA (kohya keys, 278 UNet modules including convolutions), 4 Euler
steps, 79.6 dB; the ufotable LoCon (kohya, 278 UNet + 72 text-encoder modules) at 0.8, 20 steps
at guidance 7.5, 78.4 dB, and the same file re-saved by diffusers in the PEFT layout gives the
identical image; both together (LCM 1.0 + ufotable 0.6), 82.3 dB. Scale 0 reproduces the base
model bit for bit. In the library, `sd::Pipeline::load_with_loras(dir, &[LoraSpec { path, scale }],
report)` does the same (`examples/sd_lora.rs`); in the browser build, `web_sd_load` takes the
LoRA files (`sdLoad(dir, [{ path, scale }])` in `oxinfer.js`).

### Stable Diffusion XL

An SDXL diffusers folder (`StableDiffusionXLPipeline` in `model_index.json`, with `text_encoder_2/`
and `tokenizer_2/`) loads through the same pipeline, CLI, library, C API and server entry points;
the model type is read from the folder. SDXL base 1.0 and SDXL-Turbo were run:

```sh
./target/release/oxinfer --model stabilityai/stable-diffusion-xl-base-1.0 \
    -p "a watercolor painting of a lighthouse on a cliff" --negative "blurry" -o lighthouse.png
./target/release/oxinfer --model stabilityai/sdxl-turbo -p "a red fox in a snowy forest" -o fox.png
# image-to-image and inpainting (mask blending), as for SD 1.x
./target/release/oxinfer --model stabilityai/stable-diffusion-xl-base-1.0 --init-image in.png \
    --strength 0.6 -p "an oil painting of a lighthouse at sunset" -o painting.png
# SDXL-Lightning: a few-step LoRA on the base model, over "trailing" timesteps
./target/release/oxinfer --model stabilityai/stable-diffusion-xl-base-1.0 \
    --lora ByteDance/SDXL-Lightning/sdxl_lightning_4step_lora.safetensors \
    --sampler euler_trailing --steps 4 --guidance 0 -p "a red fox in a snowy forest" -o fox.png
```

What SDXL adds is in `src/sd/xl.rs`; the rest is the SD 1.x code, which takes its structure from
the weights and configs:

- **Two text encoders**, CLIP ViT-L and OpenCLIP bigG with its text projection. The penultimate
  hidden states of both, concatenated per token (768 + 1280), are the cross-attention context;
  the projected end-of-text state of the second is the pooled embedding. An empty negative prompt
  conditions on zeros when the pipeline says so (`force_zeros_for_empty_prompt`: the base model,
  not Turbo).
- **Added conditioning.** The pooled embedding and six size ids (original size, crop offset,
  target size; each a 256-wide sinusoidal embedding) go through the UNet's `add_embedding` into
  the time embedding. As in diffusers the ids default to the output size and no crop;
  `Pipeline::size_cond` sets them in the library (`examples/sd_xl.rs`).
- **UNet.** Transformer depth per block (1, 2, 10), no attention at the top level, two
  downsamplings (so sizes are multiples of 32), linear projections: all read from the weights.
- **VAE.** The scaling factor (0.13025) comes from the VAE config. Everything here is computed
  in f32, so the original SDXL VAE's fp16 weights decode without the overflow that the "fp16-fix"
  VAE works around in half-precision runtimes (that VAE has the same architecture; not run).
- **Samplers and defaults.** Without `--sampler` an SDXL model uses the scheduler its config names
  (Euler for the base model, Euler ancestral for Turbo), with guidance 5 and 25 steps at
  1024x1024 for the base model and 1 step without guidance at 512x512 for Turbo.
  `--sampler euler_trailing` / `euler_a_trailing` run over "trailing" timesteps whatever the
  config says, which few-step LoRAs such as SDXL-Lightning need on the base model.
- **LoRA.** `--lora` reads kohya SDXL files, whose UNet keys follow the original (SGM) layout
  (`lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q`, `lora_unet_output_blocks_...`,
  `lora_unet_middle_block_...`) with `lora_te1_` / `lora_te2_` for the two text encoders, the
  diffusers-named kohya form (`lora_unet_down_blocks_...`) and the diffusers / PEFT layout
  (`unet.`, `text_encoder.`, `text_encoder_2.`).

`tools/ref_sdxl.py` runs diffusers' `StableDiffusionXLPipeline`, `...Img2ImgPipeline` and
`...InpaintPipeline` (Euler scheduler) on the same random draws, as `tools/ref_sd.py` does for
SD 1.x. On the CPU (x86-64, AVX-512), 512x512 to keep the float32 reference small, weights rounded
to bf16 on both sides (relative L2 error; image also as 8-bit PSNR):

| Run | Text context | Pooled | First UNet output | Final latents | Image |
|---|---|---|---|---|---|
| SDXL-Turbo, 1 step | 1.9e-6 | 2.0e-6 | 4.5e-7 | 9.4e-6 | 2.8e-5, 79.8 dB |
| SDXL-Turbo, 4 steps | 1.9e-6 | 2.0e-6 | 4.5e-7 | 1.0e-5 | 4.2e-5, 78.8 dB |
| SDXL base, 2 steps, guidance 5 | 1.9e-6 | 1.7e-6 | 1.1e-5 | 1.6e-5 | 3.4e-5, 76.2 dB |
| SDXL base, 6 steps, guidance 5 | 1.9e-6 | 1.7e-6 | 2.8e-6 | 1.5e-4 | 2.6e-4, 68.7 dB |
| SDXL base, 6 steps, guidance 7, negative prompt | 1.9e-6 | 1.7e-6 | 3.5e-6 | 1.3e-4 | 3.2e-4, 67.8 dB |
| SDXL base img2img, strength 0.6, 10 steps, guidance 5 (encoder mean 2.6e-5) | 1.9e-6 | 1.8e-6 | 2.3e-5 | 2.3e-4 | 3.2e-4, 68.5 dB |
| SDXL-Turbo img2img, strength 0.5, 2 steps (encoder mean 2.6e-5) | 1.9e-6 | 1.8e-6 | 2.8e-5 | 4.4e-5 | 7.2e-5, 75.8 dB |
| SDXL base inpainting (mask blending), strength 1, 8 steps, guidance 5 | 1.9e-6 | 2.1e-6 | 2.2e-6 | 4.8e-5 | 1.0e-4, 74.7 dB |
| SDXL base inpainting (mask blending), strength 0.7, 10 steps, guidance 5 | 1.9e-6 | 2.1e-6 | 1.5e-5 | 6.9e-5 | 6.9e-5, 75.8 dB |
| + kohya LoRA at 0.9 (SGM names: 722 UNet, 72 + 192 text-encoder modules), 6 steps, guidance 5 | 9.7e-7 | 1.8e-6 | 2.4e-6 | 1.1e-4 | 3.8e-4, 68.5 dB |
| + kohya LoRA with convolutions (SGM names, 788 UNet modules), 6 steps, guidance 5 | 2.0e-6 | 2.4e-6 | 1.7e-6 | 8.1e-4 | 2.6e-3, 62.0 dB |
| + SDXL-Lightning 4-step LoRA (788 UNet modules), `euler_trailing`, 4 steps | 1.9e-6 | 2.0e-6 | 2.4e-7 | 5.7e-5 | 8.5e-5, 75.9 dB |

Each stage agrees to float rounding; over several guided steps the differences grow along the
trajectory (compare the 2-step and 6-step rows of the same prompt and noise). Against the
float32 reference without the bf16 rounding, the weight storage is what shows: 42.7 dB for the
SDXL-Turbo image, and 30.1 dB after 6 guided steps of the base model (text context 1.2e-3, first
UNet output 2.6e-3, the same picture to the eye). On CUDA (RTX 5080) the 6-step base run gives
69.5 dB and the img2img run 69.4 dB against the same references, and `OXINFER_GPU_CHECK=1` reports
at worst 2.5e-6 for attention, 6.9e-7 for the fused convolution norms and 3.5e-7 for the fused
linear norms; a 1024x1024 image at the defaults (25 steps, guidance 5) was generated there too.

Not supported: the SDXL refiner (its own conditioning: one text encoder and an aesthetic score),
SDXL InstructPix2Pix, and single-file checkpoints (`sd_xl_base_1.0.safetensors`: as for SD 1.x,
only the diffusers folder layout is read). A dedicated 9-channel SDXL inpainting UNet takes the
same code path as the SD 1.x one but was not run. Repo ids download the full-precision
safetensors when a repo has both (about 14 GB for SDXL base); a folder holding only the `.fp16`
files (7 GB) loads the same, and the weights are kept in bf16 either way. To fetch only those,
take `model_index.json`, `scheduler/scheduler_config.json`, the four `tokenizer*/` files of each
tokenizer, each component's `config.json` and `text_encoder/model.fp16.safetensors`,
`text_encoder_2/model.fp16.safetensors`, `unet/diffusion_pytorch_model.fp16.safetensors`,
`vae/diffusion_pytorch_model.fp16.safetensors` from
`https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/` (ungated) into a
folder and pass that folder to `--model`.

**Against stable-diffusion.cpp** (master 82c19c8, Metal, flash attention, cond and uncond in one
batch), the same fp16 SDXL base 1.0 files on both sides (stable-diffusion.cpp reads the diffusers
folder's components with `--diffusion-model unet/... --clip_l text_encoder/... --clip_g
text_encoder_2/... --vae vae/...`; its `-m FOLDER` misses `text_encoder_2`), 1024x1024, 20 Euler
steps, guidance 5, M5 Max, 2026-11-26, the GPU shared with other jobs (ranges over interleaved
runs; a sustained run warms the GPU and oxinfer's steps go from 0.47 to 0.61 s within 20 steps):

| | oxinfer (whole-pass UNet plan) | stable-diffusion.cpp |
|---|---|---|
| sampling, 20 steps | 9.6-12.9 s | 18.0-20.4 s (22-28 s with the GPU busier) |
| per step, steady (cond + uncond) | 0.47-0.62 s | 0.83-1.04 s |
| text encoders (both CLIPs, prompt + negative) | 0.18 s | 0.78-1.0 s |
| VAE decode | 0.45-0.50 s | 4.1-4.3 s |
| peak memory footprint | 11.6 GB (weights in bf16, 6.9 GB) | 7.8 GB |

stable-diffusion.cpp decodes the original SDXL VAE in half precision and overflows: without
`--force-sdxl-vae-conv-scale` (or madebyollin's fp16-fix VAE) its image is plain white. oxinfer
runs the VAE in f32 on the CPU path and scales the inputs that leave half's range on the matrix
units, so the original VAE works as it is.

Quality, against diffusers 0.43 in float32 (the fp16 weights upcast, `tools/ref_sdxl.py`, CPU)
on the same initial noise: `torch.randn` of seed 42 on the CPU, which oxinfer reads from the
reference folder (`OXINFER_SD_DUMP`) and stable-diffusion.cpp draws itself with `--rng cpu`. The
two engines' default Euler schedules differ (diffusers' and oxinfer's follow the model's
"leading" timestep spacing, 951 ... 1; stable-diffusion.cpp's `discrete` scheduler its own), so
the like-for-like rows use "trailing" timesteps (999 ... 49): oxinfer `--sampler euler_trailing`,
stable-diffusion.cpp `--scheduler sgm_uniform`, diffusers `timestep_spacing="trailing"`:

| | image vs diffusers (8-bit PSNR) |
|---|---|
| oxinfer, default (leading) | 45.1 dB (text context 1.2e-3, first UNet output 6.2e-4, final latents 1.5e-2) |
| oxinfer, `euler_trailing` | 47.6 dB (latents 9.0e-3) |
| stable-diffusion.cpp, `sgm_uniform` | 44.9 dB |
| stable-diffusion.cpp, default `discrete` (against the trailing / leading references) | 29.4 / 23.2 dB (same picture, other timesteps) |
| oxinfer against stable-diffusion.cpp, both trailing | 43.5 dB |

Both stay close to the float32 reference; oxinfer's gap is mostly the bf16 storage of the
weights (with the reference's weights rounded to bf16 too, the shorter runs of the table above
agree to 68-80 dB), stable-diffusion.cpp's its fp16 weights and activations.

## Editing images

Give a Stable Diffusion pipeline an image and it edits it instead of starting from noise. What
the edit does follows from the model (the UNet's input channels):

| Model | UNet input | `--init-image` (+ prompt) | `--mask` |
|---|---|---|---|
| timbrooks/instruct-pix2pix, vinesmsuic/magicbrush-jul7 | 8 (latents + image latents) | **instruction edit**: the prompt says what to change | not used |
| stable-diffusion-v1-5/stable-diffusion-inpainting (and other SD inpainting models) | 9 (latents + mask + masked-image latents) | needs a mask | repaints the white part |
| any text-to-image model (SD 1.x / 2.x, SD-Turbo, SDXL, SDXL-Turbo, + LoRAs) | 4 | image-to-image at `--strength` | inpainting by blending |

**Editing with a prompt: InstructPix2Pix.** For "change this picture like so", use
InstructPix2Pix. With the SD 1.5 LCM-LoRA (InstructPix2Pix is a fine-tuned SD 1.5, so the LoRA
fits it) an edit takes 4 UNet passes instead of 60, and the edits look as good:

```sh
./target/release/oxinfer --model timbrooks/instruct-pix2pix --lora latent-consistency/lcm-lora-sdv1-5 \
    --init-image cabin.png -p "make it winter, covered in snow" --steps 4 --guidance 1 -o snowy.png
# Without the LoRA: 20 Euler-ancestral steps, three guided UNet passes each
./target/release/oxinfer --model timbrooks/instruct-pix2pix --init-image cat.png \
    -p "turn the cat into a tiger" --image-guidance 1.5 --guidance 7.5 -o tiger.png
```

`--guidance` (default 7.5) is how strongly the result follows the instruction and
`--image-guidance` (default 1.5) how closely it keeps to the input: each step runs the UNet on
(instruction, image), (no text, image) and (no text, no image) and combines them as diffusers'
`StableDiffusionInstructPix2PixPipeline` does, `uncond + g * (text - image) + ig * (image -
uncond)`. Guidance at or below 1 (as with the LCM-LoRA) turns both off: one pass per step. The
image conditioning is the mode of the VAE encoder's latent distribution, unscaled, concatenated
to the latents.

InstructPix2Pix tends to restyle the whole picture ("add a red car" painted the cabin red in our
test); MagicBrush (`vinesmsuic/magicbrush-jul7`, InstructPix2Pix fine-tuned on human-annotated
local edits, same architecture, CreativeML OpenRAIL-M, PyTorch `.bin` weights which load directly)
added the car and left the rest alone. It is a drop-in alternative at the same 20 guided steps; the
LCM-LoRA does not carry over to it (its scheduler config uses "leading" timesteps, and the 4-step
result was unusable).

**Image-to-image.** With a text-to-image model the image is encoded, noised to `--strength`
(default 0.8) and denoised towards the prompt: as diffusers' `StableDiffusionImg2ImgPipeline`,
the last `int(steps * strength)` steps of the schedule run, starting from `latents + sigma *
noise` at the first of them. Turbo models default to just enough steps for one to run (strength
0.5: 2 steps, 1 run):

```sh
./target/release/oxinfer --model models/sd-turbo --init-image cabin.png --strength 0.5 \
    -p "a cabin in a meadow at sunset, golden hour" -o sunset.png
./target/release/oxinfer --model models/sd-v1-5 --init-image cabin.png --strength 0.6 \
    -p "an oil painting of a cabin in a meadow, van gogh style" -o vangogh.png
```

**Inpainting.** `--mask FILE` marks what to repaint in white (Pillow's luma, binarized at 0.5;
any size, scaled like the image). An inpainting UNet sees the mask and the latents of the image
with the masked part zeroed at every step (diffusers' `StableDiffusionInpaintPipeline`); with a
text-to-image UNet the unmasked area is reset after each step to the input's latents noised to the
next level (diffusers' legacy approach: works, but the seams are weaker). `--strength` defaults to
1 (start from noise in the masked area); below 1 it starts from the noised image as in img2img.

```sh
./target/release/oxinfer --model stable-diffusion-v1-5/stable-diffusion-inpainting \
    --init-image cabin.png --mask cabin_mask.png -p "a red barn" -o barn.png
```

**Sizes.** Without `--size`, an edit runs at the model's area (512x512 for SD 1.5, 768x768 for
SD 2.x) with the input's aspect ratio, each side a multiple of 64 (a 768x512 photo runs at
640x448); with `--size WxH`, at that size. The image and the mask are scaled to cover it (Pillow
bicubic) and centre-cropped; the output has the working size. `--image FILE` is accepted in place
of `--init-image` (the flag the language models use); repo ids of diffusers pipelines download
like other models (`model_index.json`, configs, tokenizer and the fp16 safetensors of each
component; the safety checker and the single-file checkpoints are skipped).

**Library.** `sd::Pipeline::generate_from(&params, &EditInput, progress)` covers all three;
`img2img`, `inpaint` and `edit` (InstructPix2Pix) are shorthands, `edit_kind` says what an input
does with a given model, `edit_size` which size it runs at (`params.width` / `height` 0 pick it),
and `cancel_flag` stops a running generation from another thread. Images are
`video::RgbImage`; with the `oxideav` feature, `EditInput::from_picture` /
`with_mask_picture` take oxideav `Picture`s (any pixel format `Picture::to_rgb` converts), and the
result converts back with `Image::to_video_frame`. `examples/sd_edit.rs`:

```rust
use oxinfer::sd::{EditInput, Params, Pipeline, sched::Sampler};
oxinfer::gpu::init(&oxinfer::platform::detect(), oxinfer::platform::DevicePref::Auto);
let pipe = Pipeline::load(&oxinfer::hf::resolve("timbrooks/instruct-pix2pix", None)?)?;
let image = oxinfer::media::Media::new().load_image("cabin.png".as_ref())?;      // a video::RgbImage
let edit = EditInput::new(image).with_image_guidance(1.5);    // or .with_mask(m), .with_strength(s)
let params = Params {
    prompt: "make it snowy".into(), negative: String::new(), steps: 20, guidance: 7.5, seed: 1,
    width: 0, height: 0, sampler: Sampler::EulerAncestral,           // 0 x 0: Pipeline::edit_size
};
let (img, timings) = pipe.generate_from(&params, &edit, &mut |i, n| eprint!("\r{i}/{n}"))?;
```

**C API.** `oxi_sd_load(model, "device=auto;lora=FILE:SCALE;gpu_pass=1", progress, user)`, then
`oxi_sd_generate(s, prompt, options_json, on_image, on_step, user)` with `options_json` such as
`{"init_image": "in.png", "mask": "mask.png", "strength": 0.8, "image_guidance": 1.5, "steps": 4,
"guidance": 1, "seed": 1}` (text to image without `init_image`), or `oxi_sd_edit_rgb(s, prompt,
rgb, width, height, mask_rgb_or_NULL, options_json, ...)` for pixels already in memory;
`oxi_sd_cancel` and returning non-zero from `on_step` stop it (see `include/oxinfer.h`).

**Browser.** Image mode has an InstructPix2Pix preset (with the LCM-LoRA, 4 steps) and, for every
model, an image and a mask upload with the strength or image-guidance setting
(`web_sd_set_input`, `sdSetInput(image, mask, { strength, imageGuidance })` in `oxinfer.js`).

**Agreement.** `tools/ref_sd.py --mode img2img | inpaint | pix2pix` runs diffusers' own
pipelines (Euler scheduler) and records their random draws by role, the initial noise and each
VAE sample, which oxinfer then reads (`OXINFER_SD_DUMP`). Weights rounded to bf16 on both sides,
Metal with `OXINFER_METAL_MMA=0` (the default matrix-unit path rounds activations to half:
about 3e-4 relative, 62-67 dB on the same runs), 512x512, generated test images:

| Edit | Encoder mean | First UNet output | Final latents | Image (rel L2, 8-bit PSNR) |
|---|---|---|---|---|
| InstructPix2Pix, 6 steps, guidance 7.5, image guidance 1.5 | 9.1e-7 | 7.5e-6 | 6.1e-6 | 4.2e-6, 85.8 dB |
| InstructPix2Pix + LCM-LoRA, 4 steps, guidance 1 | 2.1e-6 | 8.3e-7 | 3.1e-6 | 5.8e-6, 86.6 dB |
| SD 1.5 img2img, strength 0.6, 10 steps, guidance 7.5 | 9.1e-7 | 8.0e-6 | 5.7e-6 | 1.5e-5, 79.0 dB |
| SD-Turbo img2img, strength 0.5, 2 steps | 9.1e-7 | 1.7e-6 | 2.8e-6 | 1.7e-5, 77.9 dB |
| SD 1.5 inpainting UNet, strength 1, 10 steps, guidance 7.5 | 1.2e-5 (masked image) | 8.1e-6 | 2.3e-5 | 2.2e-5, 80.0 dB |
| SD 1.5 inpainting UNet, strength 0.75 | 9.1e-7 | 9.9e-6 | 1.3e-5 | 8.8e-6, 81.3 dB |
| SD 1.5 (text-to-image UNet) mask blending, 10 steps | 9.1e-7 | 7.9e-6 | 9.1e-6 | 5.7e-6, 86.1 dB |
| MagicBrush (fp32 `.bin`, "leading" timesteps), 6 steps, guidance 7.5 | 8.1e-7 | 7.5e-6 | 1.0e-5 | 1.1e-5, 84.3 dB |

The encoder's stride-2 convolutions pad one zero row and column after the input only
(`Downsample2D(padding=0)`); `ConvGeom::pad` carries that to the GPU kernels, so the encoder's
convolutions, attention and group norms take the same Metal / CUDA / WebGPU paths as the decoder
(`OXINFER_GPU_CHECK=1`: worst 1.5e-6 relative for the fused norms with `OXINFER_METAL_MMA=0`).

**Timings**, 512x512 on the M5 Max (encode = VAE encoder; the rest as for text to image):

| | Encode | UNet pass | Decode | Whole edit |
|---|---|---|---|---|
| Metal: InstructPix2Pix + LCM-LoRA, 4 steps | 0.43 s | 0.35 s | 0.65 s | 2.5 s |
| Metal: InstructPix2Pix, 20 steps x 3 passes | 0.42 s | 0.33 s | 0.64 s | 21 s |
| Metal: SD-Turbo img2img, strength 0.5 (1 pass) | 0.43 s | 0.26 s | 0.59 s | 1.4 s |
| Metal: SD 1.5 img2img, strength 0.6 (15 steps x 2) | 0.43 s | 0.33 s | 0.60 s | 11 s |
| Metal: inpainting UNet, 25 steps x 2 | 0.47 s | 0.36 s | 0.72 s | 19 s |
| CPU only (18 threads; measured with another 10-core job running, so upper bounds) | 4.2-5.9 s | 3.1-5.5 s | 7.5-9.9 s | LCM edit 38 s, turbo img2img 16 s |
| Browser (Chrome, WebGPU; machine busy): InstructPix2Pix + LCM-LoRA | 3.3 s | 3.0 s | 4.8 s | 20.5 s |

On an idle machine the CPU-only UNet pass is about 0.9 s and the decode 2.9-3.4 s (see Image
generation); the encoder costs about half the decoder. The browser numbers were taken with the
machine loaded too (SD-Turbo text to image took 7.7 s there against 3.2 s quiet), and the
browser output matched the native one.

On a phone the same code runs (iOS and Android cross builds are checked; not measured on a device
for editing): iOS takes the Metal backend like macOS, Android the CPU path. At a few seconds per
CPU UNet pass, the 4-step LCM InstructPix2Pix edit is the realistic choice there; the model needs
about 2 GB of bf16 weights in memory.

**Limits.** SD 1.x / 2.x and SDXL families only (no SD 3 or Flux editing here; no SDXL
InstructPix2Pix); one image per call;
the output has the working size (multiples of 64), not the input's; no paste-back of the
unmasked pixels after inpainting (as diffusers by default: the VAE round trip slightly changes
them); the mask is binary (no feathering); InstructPix2Pix with `--strength` is not a thing
(diffusers has none: it always starts from noise); image guidance needs text guidance above 1, so
the 4-step LCM-LoRA path has no image-guidance knob. Bigger instruction editors
(Qwen-Image-Edit, FLUX.1 Kontext, Step1X-Edit, OmniGen2, BAGEL) need their own architectures;
FLUX.1 Kontext, Qwen-Image-Edit, BAGEL and Step1X-Edit have one, [below](flux.md#flux1-kontext-image-editing-from-instructions-and-flux1-text-to-image),
[after it](qwen-image.md#qwen-image-edit-image-editing-from-instructions-and-qwen-image-text-to-image),
[further on](bagel.md#bagel-image-editing-text-to-image-and-image-understanding-in-one-model) and
[after BAGEL](step1x-edit.md#step1x-edit-image-editing-from-instructions).
