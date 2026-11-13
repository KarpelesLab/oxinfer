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
| Metal | 0.08 s | 0.37 s | 0.66 s | 1.1 s | 16 s |
| Browser (Chrome, WebGPU, one CPU thread) | 0.2 s | 1.1 s | 1.9 s | 3.2 s | |

With a GPU, convolutions, attention and the group and layer norms run on it: each norm is fused
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
files (7 GB) loads the same, and the weights are kept in bf16 either way.

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

**C API.** `oxi_sd_load(model, "device=auto;lora=FILE:SCALE", progress, user)`, then
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
