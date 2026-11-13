# Qwen-Image-Edit: image editing from instructions (and Qwen-Image text to image)

Qwen-Image-Edit-2511 (and -2509, and the original Qwen-Image-Edit) edits images the way an
instruction says, combining several input images; Qwen-Image / Qwen-Image-2512 generate from text
with the same transformer. `src/qwen_image/` follows diffusers' `QwenImageEditPlusPipeline`,
`QwenImageEditPipeline` and `QwenImagePipeline`; the pipeline kind comes from `model_index.json`.
Qwen-Image-2.1 and Qwen-Image-2.1-Turbo, a different and smaller model for both text to image and
editing, have [their own section](#qwen-image-21-and-qwen-image-21-turbo).

```sh
# full quality: 40 steps, true CFG 4 with a " " negative prompt (Qwen's recommended settings)
./target/release/oxinfer --model Qwen/Qwen-Image-Edit-2511 --mmap --init-image fox.png \
    -p "Add a red knitted scarf around the fox's neck." -o scarf.png
# 4 steps with the Lightning LoRA (its steps, scheduler and guidance are picked from the file name)
./target/release/oxinfer --model Qwen/Qwen-Image-Edit-2511 --mmap --init-image fox.png \
    --lora lightx2v/Qwen-Image-Edit-2511-Lightning/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors \
    -p "Make it winter: snow on the ground, the bench and the branches." -o winter.png
# several images ("Picture 1", "Picture 2" in the instruction), a 12 GB GGUF transformer
./target/release/oxinfer --model Qwen/Qwen-Image-Edit-2511 --mmap \
    --transformer unsloth/Qwen-Image-Edit-2511-GGUF/qwen-image-edit-2511-Q4_0.gguf \
    --init-image fox.png --init-image shop.png \
    -p "The fox from Picture 1 sits on the sidewalk in front of the bookshop from Picture 2." -o both.png
# text to image (a Qwen-Image-2512 folder; fetching the repo id downloads its bf16 transformer too)
./target/release/oxinfer --model Qwen/Qwen-Image-2512 --mmap \
    --transformer unsloth/Qwen-Image-2512-GGUF/qwen-image-2512-Q4_0.gguf \
    --lora lightx2v/Qwen-Image-2512-Lightning/Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors \
    -p "a cozy bookshop on a rainy street at night, a sign that reads OXINFER BOOKS" -o shop.png
# image to image and inpainting (white = repaint) with a text-to-image or an edit model
./target/release/oxinfer --model Qwen/Qwen-Image-2512 --mmap --init-image shop.png --strength 0.6 \
    -p "the same street in heavy snow" -o snow.png
./target/release/oxinfer --model Qwen/Qwen-Image-2512 --mmap --init-image shop.png --mask door.png \
    --strength 1 -p "a red wooden door" -o door.png
# ControlNet (canny / soft edge / depth / pose image; one Union model for all of them)
./target/release/oxinfer --model Qwen/Qwen-Image-2512 --mmap --controlnet InstantX/Qwen-Image-ControlNet-Union \
    --control-image canny.png --control-scale 1.0 -p "a stone cottage in a pine forest" -o cottage.png
# an FP8 "scaled" single-file transformer (ComfyUI / lightx2v), used as stored
./target/release/oxinfer --model Qwen/Qwen-Image-2512 --mmap \
    --transformer lightx2v/Qwen-Image-2512-Lightning/qwen_image_2512_fp8_e4m3fn_scaled_comfyui_4steps_v1.0.safetensors \
    -p "a lighthouse at dusk" -o lighthouse.png
# Qwen-Image-Layered: an image into RGBA layers (layers_0.png ...; without -p the text encoder captions it)
./target/release/oxinfer --model Qwen/Qwen-Image-Layered --mmap --init-image poster.png --layers 4 -o layers.png
```

| Flag | Meaning |
|---|---|
| `--init-image FILE` | image to edit (repeatable: Edit-2509 / 2511 take several; the output size follows the last one) |
| `--strength F` | start from the input image noised to this part of the schedule instead of from noise: image to image with Qwen-Image / -2512 (default 0.6, the output takes the image's size), and with the edit models once given (or with `--mask`, default 1) |
| `--mask FILE` | inpainting: the white region is repainted, the rest returns to the input image after every step |
| `--controlnet DIR`, `--control-image FILE` | a ControlNet folder or repo id and its condition image (repeatable: the images' residuals add up); with `--mask` and the inpainting ControlNet, the image to inpaint |
| `--control-scale F`, `--control-start F`, `--control-end F` | conditioning scale (1; one per image when repeated) and the part of the steps the ControlNet acts on (0 and 1) |
| `--layers N`, `--resolution 640\|1024` | Qwen-Image-Layered: the number of RGBA layers (4) and the working resolution |
| `--steps N`, `--guidance F`, `--negative TEXT` | 40, true CFG 4, `" "` (with a Lightning LoRA: its 4 / 8 steps, CFG 1) |
| `--size WxH` | output size (default: the input's aspect ratio at about 1024x1024, multiples of 16) |
| `--lora FILE[:SCALE]` | LoRA files (lightx2v Lightning, kohya-style or PEFT keys) |
| `--transformer FILE` | the transformer from a GGUF file (unsloth / QuantStack conversions; BF16 outer layers kept) or a single safetensors file (FP8 scaled checkpoints, bf16) |
| `--quant q8\|q4` | transformer and text-encoder precision (the text encoder stays Q8 with `q4`) |
| `--mmap`, `--stream` | map the weights; stream the transformer's 60 blocks from storage (`--cache-budget-gb`) |

What runs: the instruction goes through the pipeline's chat template with Qwen2.5-VL-7B
(`crate::qwen_vl`, below): a system turn, `Picture N: <|vision_start|>...<|vision_end|>` per input
image (resized to ~384x384 with Pillow's Lanczos, then the Qwen2-VL processor), the instruction;
the last hidden states minus the template's first 64 tokens condition the transformer. The input
images are also resized to ~1024x1024 and VAE-encoded (the Wan-style causal 3D VAE, run on one
frame, latents normalized by its per-channel mean / std) into 2x2 patches that follow the noisy
target's as extra tokens. The 60-block MMDiT (20B) keeps text and image streams with their own
AdaLN modulation (reference-image tokens modulated at timestep 0, Edit-2511's `zero_cond_t`), joint
attention with per-head RMSNorm and the 3-axis MSRoPE (frame, row, column centred on each image,
text after the largest image). Flow matching with Euler steps and the resolution-dependent shift
(`crate::flux::sched`), true CFG with the norm-preserving rescale of the combined velocity. The
Lightning LoRA uses its distillation's fixed shift of 3.

Components load when first needed and the 16 GB text encoder is dropped after encoding (the
library keeps it with `keep_text_encoder`). A LoRA is merged into the weights when they are
quantized at load anyway (`--quant`; merged in f32 and quantized from that, since a Lightning
update is below half a bf16 ulp of most weights and rounding the merged weights to bf16 would erase
it), and applied at run time otherwise (bf16 weights borrowed from the mapped checkpoint, GGUF
blocks), as two small extra matmuls, so no 40 GB copy is made. With `--quant`, the layers outside
the blocks stay bf16 (as GGUF conversions keep them), and the modulation layers keep the ~17 input
channels of `silu(timestep embedding)` that carry 99.9% of its energy exact (f32 columns beside the
quantized matrix). On Metal
the text encoder runs with exact (f32) matmuls (its hidden states have outlier channels near 200,
which the half-precision matrix units round by 1%), and the transformer's attention keeps f32
operands (its logits run into the thousands; half-precision queries and keys move the first
velocity ~10%); the transformer's matmuls and its attention both run on the matrix units.

**Agreement** with diffusers 0.42 / transformers 5.20 (`tools/ref_qwen_image.py`: same input image,
prompt and initial noise; bf16 weights on both sides with every reference matmul in f32;
Edit-2511, a 512x512 fox photo, "make it winter", 512x512 output with 4096 reference tokens):

| oxinfer | prompt embeddings | first velocity | final latents | image PSNR |
|---|---|---|---|---|
| bf16, CPU, 2 steps, true CFG 4 | 4.1e-4 | 1.5e-4 | 1.7e-4 | 68.5 dB |
| bf16, Metal (matrix units), 2 steps, CFG 4 | 1.1e-2 (before the exact text encoder) | 2.8e-3 | 3.1e-3 | 55.4 dB |
| bf16 + Lightning LoRA, CPU, 4 steps | 4.1e-4 | 1.8e-3 | 7.6e-3 | 45.7 dB |
| bf16 + Lightning, Metal with `OXINFER_METAL_MMA=0` | 4.4e-4 | 1.7e-3 | 7.9e-3 | 45.7 dB |
| bf16 + Lightning, Metal (default) | 4.4e-4 | 7.3e-3 | 8.3e-2 | 28.9 dB |
| `--quant q8` + Lightning, Metal | 1.7e-2 | 4.7e-2 (was 0.16) | 0.11 (was 0.35) | 26.8 dB (was 17.8) |
| `--quant q4` + Lightning, Metal | 1.7e-2 | 0.14 (was 0.20) | 0.39 (was 0.57) | 15.6 dB (was 12.2) |
| GGUF Q4_0 (unsloth) + Lightning, Metal | 4.4e-4 | 0.14 | 0.22 | 21.3 dB |
| reference itself with oxinfer's Q8 weights (dequantized, f32 matmuls), Lightning | | 3.4e-2 | 0.12 | 25.7 dB |

(Relative L2 errors.) The VAE encoder matches to 1.2e-6. The "was" figures came from two load-time
flaws, both fixed: the Lightning LoRA was merged into bf16-rounded weights before quantizing (its
update is 0.03-0.14% of each weight matrix's norm and the rounding erased 60-90% of it: bf16
merged weights alone moved the first velocity by 16.6%), and the modulation layers' few dominant
input channels shared Q8 blocks with the rest (the first block's modulation alone: 5.6%). What
remains is the model: the reference run with oxinfer's Q8 weights gives 3.4% and 25.7 dB (last
row), and oxinfer's Q8 run with the reference's prompt embeddings 3.5% and 25.3 dB. With the
Lightning LoRA, Qwen-Image amplifies weight noise about 6x into the first velocity (without it,
Q8 moves the first velocity by 0.21% and Q4 by 2.3% against oxinfer's bf16; they were 0.51% and
5.9%), then the four large steps amplify the difference again; the first-velocity error scales
linearly with the weight noise, so 1% would take ~10-bit weights. Quantized Lightning runs give a
visibly equivalent edit with different fine detail rather than the same pixels (as does the
default Metal bf16 run, whose half-precision matmul inputs give 28.9 dB). The text encoder at Q4
moves the conditioning by 24%, so `--quant q4` keeps it at Q8. diffusers 0.42 calls the text encoder without `mm_token_type_ids`, and
transformers 5 then gives the image tokens plain 1D positions; the reference script passes them
(M-RoPE, as the model was trained, and as oxinfer does).

**Image to image, inpainting, ControlNet, layers, FP8 files.** These follow diffusers'
`QwenImageImg2ImgPipeline`, `QwenImageInpaintPipeline`, `QwenImageEditInpaintPipeline`,
`QwenImageControlNetPipeline`, `QwenImageControlNetInpaintPipeline` and `QwenImageLayeredPipeline`:

- *Image to image*: the input is VAE-encoded (a sample of the posterior, as those pipelines take
  it), noised to the first of the last `steps * strength` noise levels, and denoised from there.
- *Inpainting*: the mask is resized to the output, binarized at one half and taken to the latent
  grid by nearest neighbour; after every step the kept region is set back to the input's latents
  noised to the next level (the clean latents after the last step). With Qwen-Image-Edit the
  input is also the reference image, as in `QwenImageEditInpaintPipeline`; Edit-2509 / 2511 have
  no diffusers inpainting pipeline, and oxinfer applies the same blending to their last image
  (no reference to check that combination against).
- *ControlNet* (`src/qwen_image/controlnet.rs`): InstantX's Union model is a copy of the first 5
  blocks with its own input layers; it runs on the noisy latents plus the embedded latents of the
  control image, each block's image stream goes through one linear layer, and the transformer
  adds residual `i / ceil(60 / 5)` after its block `i`. The residuals are computed once per step
  (with the positive prompt) and shared by the guided and unguided passes. The inpainting
  ControlNet (6 blocks) takes the latents of the image with the masked region blanked plus the
  kept-region mask as a 17th latent channel. ControlNets run with the text-to-image models.
- *Qwen-Image-Layered*: the composite and every layer are denoised together as frames 0..N of
  one token sequence, the input image's latents (through the RGBA VAE, opaque) follow at frame
  -1, the timestep embedding gets the "RGBA" embedding added, the sigmas are shifted linearly
  with `mu = sqrt(tokens / 256)`, and the layers (not the composite) are decoded to RGBA PNGs.
  Without a prompt the text encoder writes the caption (the pipeline's captioning prompt, greedy
  with its repetition penalty of 1.05, at most 512 tokens).
- *FP8 scaled checkpoints*: `float8_e4m3fn` weights times a scale, per tensor (ComfyUI's
  `scale_weight` with the `scaled_fp8` marker) or per output row (lightx2v's `weight_scale`),
  borrowed from the mapped file and run by the FP8 GEMMs; a `..._4steps_...` / `..._8steps_...`
  file name selects the Lightning settings like a Lightning LoRA does.

Checked against diffusers 0.42 on tiny random-weight pipelines (`tools/qwen_image_tiny.py make`
builds them: 3 blocks of width 64, a 2-block ControlNet, an RGBA VAE, bf16-exact weights; `ref`
runs the diffusers pipeline in float32 and hands oxinfer its noise), CPU, relative L2 of the final
latents and the 8-bit image:

| pipeline (tiny model) | settings | first residuals / velocity | final latents | image |
|---|---|---|---|---|
| image to image | 128x96, strength 0.6 of 5 steps | 2.1e-7 | 1.1e-7 | max pixel difference 1 |
| image to image, resized input, true CFG 2.5 | strength 0.75 of 6 steps | | 1.4e-7 | identical |
| inpainting | strength 0.8 of 5 steps | 2.8e-7 | 1.3e-7 | identical |
| inpainting, resized input and mask, true CFG 3 | strength 1, 4 steps | | 1.7e-7 | max difference 1 |
| Qwen-Image-Edit inpainting (1024x1024, 4096 + 4096 tokens) | strength 0.7 of 3 steps | 2.8e-7 | 2.0e-7 | max difference 1 |
| ControlNet, scale 0.8, true CFG 3 | 4 steps | 1.8e-7 (residuals) | 2.0e-7 | max difference 1 |
| ControlNet, scale 1.3, steps 0.2..0.8 of 5 | | 2.2e-7 | 9.3e-8 | max difference 1 |
| ControlNet, two control images | 3 steps | 2.5e-7 | 1.1e-7 | max difference 1 |
| inpainting ControlNet, true CFG 3 | 4 steps | 1.9e-7 (residuals) | 2.0e-7 | identical |
| Layered, 3 layers at 736x544 (6256 + 1564 tokens), true CFG 3 | 3 steps | | 4.3e-7 | max difference 1 |
| FP8, ComfyUI layout (reference: the decoded weights) | 3 steps | 2.3e-7 | 1.1e-7 | max difference 1 |
| FP8, per-row scales | 3 steps | 2.2e-7 | 1.1e-7 | max difference 1 |

These runs take the reference's prompt embeddings, so they measure the pipelines; with oxinfer's
own text encoding the tiny inpainting, ControlNet and layered runs give the same figures (prompt
embeddings 1.7e-7). The 512-token caption of the tiny layered model is the same text as
transformers' `generate` gives. On the real FP8 files, `fp8 * scale` of three layers read out of
`qwen_image_2512_fp8_e4m3fn_scaled.safetensors` (per-row scales) and of the ComfyUI-layout
`..._comfyui_4steps_...` file differs from Qwen-Image-2512's bf16 weights by 2.6-2.7% (FP8's
rounding; the latter has the Lightning LoRA merged); no generation was run from a 20 GB FP8 file.

On real weights (Qwen-Image-2512 with unsloth's Q3_K_M GGUF transformer, 512x512, CPU): the
residuals of InstantX's Union ControlNet at the first step, on oxinfer's own latents, control
tokens and prompt embeddings, match diffusers' `QwenImageControlNetModel` (float32) to 3.3e-6.
With 12 steps and true CFG 3 an edge map of a frontal stone cottage turns the three-quarter view
the prompt alone gives ("a wooden log cabin in a snowy pine forest at night") into a frontal
cabin with the door and windows where the edges are; with the 4-step Lightning LoRA the same
control image changes the picture but does not set its layout (the ControlNet was trained on
Qwen-Image, not on -2512 or the distilled model), so use it without the LoRA. Inpainting a
rectangle over the door of a generated cottage (Lightning, strength 1, "a bright blue door")
paints a blue door inside the mask and returns the rest at 32.7 dB of the input (the VAE round
trip of the kept region, as in diffusers); image to image at strength 0.6 ("in deep winter
snow") keeps the cottage and its framing. The inpainting ControlNet, Qwen-Image-Edit inpainting
and Qwen-Image-Layered were checked on the tiny models only (their weights were not downloaded).



**Speed and memory** (M5 Max, 1024x1024 output from a 512x512 photo: 4096 target + 4096 reference
+ ~210 text tokens per pass; Lightning: 4 passes, full quality: 80):

| weights | device | per transformer pass | Lightning edit (total) | 40 steps + CFG | peak memory (footprint) |
|---|---|---|---|---|---|
| bf16 (`--mmap`, 41 GB mapped) | Metal | 18 s | 84 s | 23 min (17 s / pass) | 10 GB |
| `--quant q8` | Metal | 16 s | 89 s | | 34 GB |
| `--quant q4` | Metal | 17 s | 89 s | | 22 GB |
| GGUF Q4_0 (12 GB, `--mmap`) | Metal | 18 s | 84 s | | 11 GB |
| GGUF Q4_0, `--stream --cache-budget-gb 4`, 512x512 | Metal | 13 s | 59 s | | 9.5 GB |
| bf16, 512x512 output (5300 tokens) | CPU, 18 threads | 185 s | 13 min | | |

The text encoder takes 1-2 s, the VAE 3 s to encode and 5 s to decode at 1024x1024; two input
images double the reference tokens (37 s per pass). Qwen-Image-2512 text to image (GGUF Q4_0 +
Lightning, 1024x1024): 58 s, 9.4 GB.

**Sample edits** (Edit-2511 bf16 + Lightning 4 steps, from a 512x512 fox photo and a bookshop that
Qwen-Image-2512 generated): "Make it winter" covers the ground, bench and branches in snow with
falling flakes; "Add a red knitted scarf around the fox's neck" adds the scarf and keeps the rest of
the photo; "Replace the text on the large shop sign with 'QWEN EDIT'" rewrites the gilded sign in
the same lettering and leaves the hanging sign and the window alone; with both images, "The fox
from Picture 1 sits on the sidewalk in front of the bookshop from Picture 2" places the fox there.

**Library and C API.**

```rust
use oxinfer::qwen_image::{Options, Params, Pipeline};
let mut pipe = Pipeline::load("Qwen/Qwen-Image-Edit-2511", Options { mmap: true, ..Default::default() })?;
let img = oxinfer::media::Media::new().load_image("fox.png".as_ref())?;
let (out, timings) = pipe.edit(&[img], &Params::new("make it winter"), &mut |i, n| eprintln!("{i}/{n}"))?;
// out: width, height, rgb; pipe.generate(&params, ..) for text to image; Params::lightning(prompt, 4)
// with a Lightning LoRA in Options::loras; edit_pictures(&[oxideav Picture]) with `oxideav`
// Params { strength: Some(0.6), mask: Some(mask), .. }: image to image / inpainting;
// Options { controlnet: Some(dir), .. } + Params { control: vec![Control { image, scale: 1.0 }], .. };
// pipe.decompose(&img, &Params::layered("", 4), ..) -> Vec<RgbaImage> (Qwen-Image-Layered)
```

The C API has `oxi_qwen_image_load` (`key=value` options: quant, text_quant, mmap, transformer,
controlnet, lora, keep_text_encoder, stream, device), `oxi_qwen_image_edit` (a JSON request:
prompt, images, negative, true_cfg, steps, seed, width, height, lightning, strength, mask,
control_images, control_scale, control_start, control_end), `oxi_qwen_image_edit_rgb` (pixels),
`oxi_qwen_image_cancel` and `oxi_qwen_image_free` (see `include/oxinfer.h`).

**Weights and licenses.** Qwen/Qwen-Image-Edit-2511, -2509, -Edit, Qwen/Qwen-Image, -2512:
Apache 2.0, ungated (57 GB each in bf16: 41 GB transformer, 16 GB text encoder, 0.25 GB VAE; the
text encoder and VAE files are identical across them). unsloth/Qwen-Image-Edit-2511-GGUF and
unsloth/Qwen-Image-2512-GGUF (Q2_K 7.7 GB to Q8_0 22 GB, BF16 41 GB), lightx2v/Qwen-Image-Edit-2511-Lightning
and lightx2v/Qwen-Image-2512-Lightning (0.85 GB LoRAs, 20 GB FP8 scaled transformers): Apache 2.0.
InstantX/Qwen-Image-ControlNet-Union (3.5 GB) and -Inpainting (4.2 GB), Qwen/Qwen-Image-Layered
(57 GB; unsloth/Qwen-Image-Layered-GGUF for its transformer): Apache 2.0.

Not supported: batches (one image per call), ComfyUI's mixed-precision (`comfy_quant`) and int8
single-file checkpoints, ControlNets with the edit models, several different ControlNets at once,
`padding_mask_crop` of the inpainting pipelines, layers through the C API and the server (the
command line and the library return them), the browser build (the weights exceed its 4 GB).
The transformer's activations go between CPU and GPU at every
matmul (about 1 s per pass at 1024x1024), the next lever for speed.

## Qwen-Image-2.1 and Qwen-Image-2.1-Turbo

Qwen-Image-2.1 is a different model behind the same command line and library type
(`model_index.json` says `QwenImage21Pipeline`): a 7B single-stream transformer instead of the
20B MMDiT, Qwen3-VL-8B instead of Qwen2.5-VL-7B as the text encoder, a 16x RGBA VAE with 64
latent channels, and text to image and editing in one checkpoint. Qwen-Image-2.1-Turbo is its
8-step distillation. `src/qwen_image/v21.rs` follows diffusers' `QwenImage21Pipeline`.

```sh
# Qwen-Image-2.1-Turbo: 8 steps, no guidance, a 4 GB GGUF transformer
# (the repo id fetches the whole repo, its 14 GB bf16 transformer included)
./target/release/oxinfer --model Qwen/Qwen-Image-2.1-Turbo --mmap \
    --transformer abenzerps/Qwen-Image-2.1-Turbo-Quantized/qwen-image-2.1-turbo-Q4_K_M.gguf \
    -p 'a cozy bookshop on a rainy street at night, a neon sign that reads "OXINFER BOOKS"' -o shop.png
# an edit: one or more input images (the output takes the last one's aspect ratio)
./target/release/oxinfer --model Qwen/Qwen-Image-2.1-Turbo --mmap --init-image shop.png \
    -p "Make it winter: thick snow on the pavement, snowflakes falling." -o winter.png
# Qwen-Image-2.1: 40 steps, no guidance unless asked (--guidance 4 --negative " ")
./target/release/oxinfer --model Qwen/Qwen-Image-2.1 --mmap --size 1024x1024 -p "..." -o out.png
# a LoRA (diffusers keys) on the Turbo checkpoint
./target/release/oxinfer --model Qwen/Qwen-Image-2.1-Turbo --mmap \
    --lora MicksHF/Qwen-Image-2.1-Anime-Fusal/fusal-qwen-image-2.1-turbo-e6-preview-r128.safetensors:0.7 \
    -p "fusal style. a girl reading in a sunlit library" -o anime.png
```

What changed against Qwen-Image-2512, as the configs and diffusers' code have it:

| | Qwen-Image-2512 | Qwen-Image-2.1 / 2.1-Turbo |
|---|---|---|
| transformer | `QwenImageTransformer2DModel`: 60 dual-stream blocks of width 3072 (24 heads x 128), 20B, biases, GELU MLPs (x4), AdaLN per block with shift / scale / gate | `QwenImage21Transformer2DModel`: 32 single-stream blocks of width 4096 (32 x 128), 7B, no bias anywhere, SwiGLU MLPs (`mlp_ratio` 3), one `modulation` layer shared by all blocks giving scale and `tanh` gate only |
| config keys | `joint_attention_dim` 3584, `patch_size` 2, `in_channels` 64 (16 x 2 x 2), `out_channels` 16, `num_layers` 60, `num_attention_heads` 24, `guidance_embeds` | `context_in_dim` 4096, `patch_size` 1, `in_channels` / `out_channels` 64, `num_layers` 32, `num_attention_heads` 32, `mlp_ratio` 3, `eps`, `causal_condition`; `attention_head_dim` 128 and `axes_dims_rope` [16, 56, 56] as before |
| sequence | text and image streams, joint bidirectional attention; text positions after the image | one sequence `[text (with the condition images where their placeholders are), target]`, block-causal attention (a token sees what precedes it, an image is bidirectional inside), text and condition images modulated as at timestep 0 |
| text encoder | Qwen2.5-VL-7B (3584), last hidden states, 34 / 64 template tokens dropped | Qwen3-VL-8B (4096, 36 layers, DeepStack vision features), the last layer's output before the final norm, the system turn dropped; system prompt "Comprehend and analyze the provided prompt." |
| VAE | `AutoencoderKLQwenImage`: 16 channels, 8x, RGB, 0.25 GB | `AutoencoderKLQwenImage21`: 64 channels, 16x, RGBA, residual groups with shortcuts, decoder width 144 (encoder 96), 0.68 GB in bf16 |
| scheduler | flow-matching Euler, dynamic exponential shift (0.5 to 0.9 over 256 to 8192 tokens), `shift_terminal` 0.02 | the same for Qwen-Image-2.1; Turbo: no shift (`use_dynamic_shifting` false, `shift` 1, no terminal), the sigmas saved in `model_index.json` |
| defaults | 40 steps, true CFG 4 with a `" "` negative prompt, norm rescale | 40 steps, no guidance (true CFG 1; with a negative prompt, plain CFG without the rescale); Turbo: the 8 saved sigmas `1, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568` |

What runs: the prompt goes through the pipeline's template (`<|im_start|>system\nComprehend and
analyze the provided prompt.<|im_end|>`, the user turn, the assistant header) and Qwen3-VL-8B;
the transformer reads the residual stream after the last layer, before the final norm, without
the system turn. The transformer runs the prompt once, keeping every block's keys and values
(nothing before the target depends on the target or on the timestep: text and condition images
are modulated as at timestep 0 and attend backwards only), and each denoising step then computes
the target's tokens alone against them (diffusers' `use_kv_cache`, here without the first full
pass). An edit's input images are resized to about 1024x1024 and go two ways: through the text
encoder's vision tower as `<imageN><|vision_start|>...<|vision_end|>` before the instruction
(with Qwen3-VL's DeepStack: the tower's features after its layers 8, 16 and 24, each through its
own merger, are added to the image tokens after the language model's first three layers), and
through the VAE as opaque RGBA into latent tokens, four of which stand in the transformer's
sequence where the text encoder had one placeholder. The VAE decodes to RGBA: the command line
writes an RGBA PNG when the picture has transparent pixels (alpha under one half somewhere; ask
for them as the model card does, "This is an RGBA image with transparency. ... The image has
alpha channel and the background is transparent.") and RGB otherwise; the library returns the
RGB channels as they are and keeps the alpha in `Pipeline::alpha`.

| Flag | Meaning with Qwen-Image-2.1 |
|---|---|
| `--steps N` | default: the checkpoint's saved sigmas (Turbo: 8) or 40; another count with Turbo uses the uniform grid, which Qwen did not train it for |
| `--guidance F`, `--negative TEXT` | true CFG (default 1: one pass per step) |
| `--size WxH` | multiples of 32; default 1024x1024, or the last input image's aspect ratio at about that area (the model cards use 2048x2048, 1696x2528, ...: four times the tokens) |
| `--init-image FILE` | an edit; repeatable |
| `--transformer FILE` | a GGUF file (ComfyUI-style names without metadata, or `general.architecture = qwen_image21`), an FP8 "scaled" single file in ComfyUI's layout, or bf16 safetensors |
| `--lora FILE[:SCALE]` | LoRAs in diffusers keys (`transformer.<module>.lora_A / lora_B`, with or without `.alpha`); merged at load with `--quant`, applied at run time otherwise |
| `--quant q8\|q4` | quantizes the transformer's blocks and the text encoder at load |

Not supported with 2.1: `--strength`, `--mask`, ControlNets, `--stream`; text encoders as GGUF
files (AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF is the text encoder, llama.cpp's
`qwen3vl` architecture with an `mmproj` file, not a transformer); ComfyUI's fused `gate_up`
single files (the 265-tensor layout), int8 / NVFP4 / MLX / bitsandbytes checkpoints. The
prompt's keys and values stay in memory for the run (float32, 32 blocks): 1 MB per text token,
4.3 GB per 1024x1024 input image.

**Agreement** with diffusers 0.44.0.dev0 (from source; the released 0.43 ships the same pipeline)
and transformers 5.23, `tools/qwen_image_tiny.py ref21`: the same prompt and initial noise, the
reference in float32. On real weights, Qwen-Image-2.1-Turbo with abenzerps' Q4_K_M GGUF
transformer (4.05 GB) and the reference running the weights that file decodes to (gguf-py's
dequantization, so both sides compute with the same numbers), 8 steps, a 768x768 image of a
bookshop with a neon sign, oxinfer's own text encoding (Qwen3-VL-8B in bf16):

| run | prompt embeddings | first velocity | final latents | image (RGBA) |
|---|---|---|---|---|
| text to image, CPU | 7.2e-6 | 2.5e-6 | 6.6e-4 | 64.6 dB, largest pixel difference 13 |
| text to image, CUDA (RTX 5080) | 7.2e-6 | 1.7e-6 | 1.4e-4 | 72.0 dB, largest pixel difference 3 |
| edit of that picture ("Make it winter: ..."), 768x768 from the input at 1024x1024, CUDA | 7.2e-4 (text and image tokens) | 3.5e-5 | 1.1e-4 | 75.5 dB, largest pixel difference 1 |

(Relative L2 errors; the RGB channels alone: 63.3, 70.8 and 74.3 dB. In the edit the VAE
encoder's latents of the input match to 3.4e-6, and the prompt embeddings cover the vision tower
with DeepStack on the real Qwen3-VL-8B. An opaque picture's alpha is not exactly 255: 2.7% of its
pixels are below, down to 230, the same in the reference.) The saved Turbo sigmas go through the
scheduler unchanged, as diffusers prints them.

On tiny random-weight pipelines (`make21`: 3 blocks of width 64, a 5-layer Qwen3-VL with a
4-layer vision tower and three DeepStack levels, the residual VAE with 8 latent channels,
bf16-exact weights), CPU, oxinfer's own text encoding:

| run (tiny model) | settings | prompt embeddings | first velocity | final latents | image |
|---|---|---|---|---|---|
| Turbo, text to image | 96x64, the 8 saved sigmas | 4.7e-7 | 2.3e-7 | 8.3e-8 | max pixel difference 1 |
| Turbo, `--steps 3` (uniform sigmas on both sides) | 64x64 | | 2.1e-7 | 7.2e-8 | identical |
| 2.1 scheduler (dynamic shift, terminal 0.02) | 64x64, 40 steps | | 2.1e-7 | 7.9e-8 | max difference 1 |
| 2.1, true CFG 3 with a negative prompt | 160x224, 5 steps | | | 1.7e-7 | max difference 1 |
| GGUF, `model.diffusion_model.` names, no metadata, Q8_0 blocks | 96x64, 8 steps | | 2.5e-7 | 8.6e-8 | identical |
| GGUF, `qwen_image21`, Q4_0 / Q8_0 blocks, true CFG 2 | 96x64, 8 steps | | | 1.2e-7 | max difference 1 |
| FP8 scaled single file (ComfyUI layout) | 96x64, 8 steps | | 1.9e-7 | 1.1e-7 | max difference 1 |
| LoRA at 0.7 (diffusers keys with alphas), bf16 and GGUF base | 96x64, 8 steps | | 9.0e-5 | 2.9e-5 | max difference 1 |
| LoRA at 0.7 merged at `--quant q8` | 96x64, 8 steps | 1.7e-2 | 6.3e-4 | 1.9e-4 | max difference 1 |
| edit, one image | 128x96, 8 steps | 8.1e-7 | 2.4e-7 | 1.1e-7 | max difference 1 |
| edit, two images, true CFG 2.5, default size (800x1312) | 4 steps | 9.2e-7 | | 2.0e-7 | max difference 1 |
| edit, three images, GGUF transformer | 160x96, 8 steps | 9.7e-7 | 2.4e-7 | 9.5e-8 | max difference 1 |

The GGUF, FP8 and LoRA rows compare with diffusers running the weights those files decode to (a
runtime LoRA is kept as a bf16 pair, hence 9e-5; the `q8` row measures the quantization). In the
edit rows the VAE encoder's latents of the input images match to 9e-7, and the prompt embeddings
cover the vision tower, DeepStack and the image tokens' positions. diffusers 0.44.0.dev0 does
not load a LoRA file that has both diffusers keys and `.alpha` tensors (it prefixes
`transformer.` twice); the reference script folds the alphas into the up matrices first.

Also run on real weights, without a reference: MicksHF's Fusal Turbo LoRA (rank 128, 224
modules, strength 0.7) on the Turbo Q4_K_M GGUF draws "fusal style. a girl with silver hair
reading a book in a sunlit library" as a clean anime illustration; chatrlabs' Q4_K_M GGUF (a
fine-tune of Qwen-Image-2.1 with `general.architecture = qwen_image21`, Q4_K and Q6_K blocks)
with Qwen-Image-2.1's scheduler, 16 steps and true CFG 5 gives a clean 512x512 photograph of its
prompt (a fox on a park bench); the Turbo edit above with oxinfer's own noise keeps the street
and the shop sign's lettering and covers the pavement and ledges in snow, with flakes falling.
Not run on real weights: the bf16 transformers (14 GB; the code path is the tiny models'),
`--quant`, the FP8 single file, Qwen-Image-2.1's own checkpoint and 40-step schedule (only the
fine-tune above, at 16 steps), several input images, transparent pictures, 2048x2048.

**Speed** (64-core x86 with an RTX 5080, a shared machine, Q4_K_M GGUF with `--mmap`): 35 s per
pass at 768x768 and 57 s at 1024x1024 on CUDA, 41 s at 768x768 on the CPU alone (the GGUF
blocks' matmuls run on the CPU either way); the text encoder takes 2-4 s, the VAE 30 s to decode
768x768 and 44 s at 1024x1024 (its decoder is 1152 channels wide at the first three levels). An
edit of one 1024x1024 input adds 13 s of VAE encoding and 15 s of text encoding with the vision
tower, and its steps attend to 4096 more tokens (36 to 43 s per pass at 768x768).

**Weights and licenses.** Qwen/Qwen-Image-2.1 and Qwen/Qwen-Image-2.1-Turbo: the Qwen Research
License Agreement (not Apache 2.0 like the earlier Qwen-Image models), ungated; 33 GB each (17.5 GB text encoder, 14.2 GB transformer, 1.35 /
0.68 GB VAE). abenzerps/Qwen-Image-2.1-Turbo-Quantized (GGUF Q4_0 4.05 GB to Q8_0 7.6 GB, an FP8
file; also the text encoder and VAE as single files, which oxinfer does not read) and
chatrlabs/Qwen-Image-2.1-Uncensored-Omni-Quantized (GGUF Q4_0 to BF16): the same license.
MicksHF/Qwen-Image-2.1-Anime-Fusal: LoRAs for the Turbo and the base checkpoint.
