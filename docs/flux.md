# FLUX.1 Kontext: image editing from instructions (and FLUX.1 text to image)

FLUX.1 Kontext [dev] edits an image the way an instruction says: "make it night", "put a red
knitted hat on the dog", "change the text on the sign to ...". The same transformer family does
text to image as FLUX.1 [schnell] and [dev]. `src/flux/` follows diffusers'
`FluxKontextPipeline` and `FluxPipeline`. A diffusers FLUX folder, or a FLUX GGUF transformer,
switches oxinfer to it:

```sh
# editing: --init-image is the picture, --prompt the instruction (34 GB download, bf16)
./target/release/oxinfer --model unsloth/FLUX.1-Kontext-dev --init-image photo.png \
    -p "Make it night, with the porch lamps lit" --guidance 2.5 -o night.png
# the transformer as a GGUF quant (6.8 GB at Q4_0); T5, CLIP, VAE and tokenizers then come from
# unsloth/FLUX.1-Kontext-dev (--base), without its 24 GB transformer
./target/release/oxinfer --model QuantStack/FLUX.1-Kontext-dev-GGUF:Q4_0 --text-quant q8 \
    --init-image sign.png -p 'Change the text on the sign to "OXINFER CAFE"' -o cafe.png
# quantize the bf16 checkpoint at load instead (q8 / q4), T5 included
./target/release/oxinfer --model unsloth/FLUX.1-Kontext-dev --quant q4 --init-image in.png -p "..." -o out.png
# text to image with FLUX.1 [schnell] (4 steps, no guidance)
./target/release/oxinfer --model city96/FLUX.1-schnell-gguf:Q4_0 \
    -p "a photograph of a golden retriever on a farmhouse porch" --size 1024x768 -o dog.png
```

| Flag | Meaning |
|---|---|
| `--init-image FILE` | the image to edit (PNG, JPEG, WebP, ...); without it, text to image |
| `-p TEXT` | the instruction (editing) or the prompt |
| `--steps N`, `--guidance F`, `--seed N` | 28 steps (4 for schnell); distilled guidance 3.5 as diffusers (BFL suggests 2.5 for edits); seed 42 |
| `--size WxH` | output size, rounded to multiples of 16 (sets `--max-area` to W x H unless given; default: the input's aspect ratio at 1024^2 pixels; diffusers would rescale any size to 1024^2) |
| `--max-area N` | editing: output pixels, default 1048576 (1024^2), rounded to multiples of 16 |
| `--no-auto-resize` | editing: keep the input at its own size (rounded down to 16) instead of the closest of Kontext's 17 preferred ~1 MP resolutions |
| `--quant bf16\|q8\|q4` | transformer precision when stored as float; `--text-quant` sets T5's (default: `--quant`) |
| `--transformer SPEC` | a transformer replacing the folder's: GGUF file or `org/name:QUANT` |
| `--text-encoder SPEC` | a T5 encoder GGUF (`city96/t5-v1_1-xxl-encoder-gguf:Q8_0`) replacing `text_encoder_2/` |
| `--base SPEC` | with a GGUF `--model`: where T5, CLIP, the VAE and the tokenizers come from (default `unsloth/FLUX.1-Kontext-dev`) |

**What runs.** The prompt goes through CLIP-L (the pooled embedding, `sd::clip`) and the T5 v1.1
XXL encoder (512 tokens padded with `<pad>`, attended like any token as diffusers does; 256 for
schnell). For an edit, the input is resized with Pillow-exact Lanczos (diffusers'
`VaeImageProcessor`) to the preferred resolution closest to its aspect ratio, encoded by the VAE
(posterior mean, `(z - 0.1159) * 0.3611`), packed in 2x2 patches and appended to the noise tokens
with RoPE index 1 on the first axis (the noise is at index 0, text tokens at 0, 0, 0). The output
size is the requested one rescaled to `max_area` (Python's round-half-even, then multiples of 16).
Sigmas are `linspace(1, 1/steps)` shifted by `mu`, linear in the number of noise tokens between
(256, 0.5) and (4096, 1.15) (`FlowMatchEulerDiscreteScheduler`, dynamic shifting); FLUX.1 [schnell]
uses no shift. Each step runs the transformer on `[text; noise; reference]` with the guidance scale
as an input (guidance distillation: one pass per step, no unconditional pass; `true_cfg_scale` with
a negative prompt adds one, as in diffusers), drops the reference tokens from its output, and takes
an Euler step `x += (sigma_next - sigma) * v`. The latents are unpacked and decoded by the VAE.

**Weights.** A diffusers folder or repo (`model_index.json`) loads every component; repo ids fetch
only what the pipeline reads (component folders, not the top-level single-file checkpoints), so
`unsloth/FLUX.1-Kontext-dev` is about 34 GB. The transformer loads from `transformer/` (bf16, borrowed
from the mapped shards; `--quant q8|q4` quantizes at load) or from a GGUF file in the original
black-forest-labs layout (city96 / QuantStack quants, any GGUF block format, zero-copy; fused
`qkv` / `linear1` rows are split into separate layers without copying). The modulation vectors are
computed with float activations even for quantized weights (a single row would otherwise take the
int8 decode kernels, a 1e-3 error steering every block).

| Repository | What | Size | License |
|---|---|---|---|
| unsloth/FLUX.1-Kontext-dev (ungated copy of the gated black-forest-labs/FLUX.1-Kontext-dev) | the whole pipeline, diffusers layout | 23.8 GB transformer + 9.5 GB T5 + 0.4 GB CLIP / VAE (bf16) | FLUX.1 [dev] Non-Commercial License |
| QuantStack/FLUX.1-Kontext-dev-GGUF | Kontext transformer as GGUF, Q2_K to Q8_0 | 4.0 GB (Q2_K), 6.8 GB (Q4_0), 12.7 GB (Q8_0) | FLUX.1 [dev] Non-Commercial License |
| city96/FLUX.1-schnell-gguf | FLUX.1 [schnell] transformer (text to image) | 6.8 GB (Q4_0) - 23.8 GB (F16) | Apache 2.0 |
| city96/FLUX.1-dev-gguf | FLUX.1 [dev] transformer (text to image) | same | FLUX.1 [dev] Non-Commercial License |
| city96/t5-v1_1-xxl-encoder-gguf | the T5 encoder as GGUF (`t5encoder`) | 2.9 GB (Q4_K_M), 5.1 GB (Q8_0) | Apache 2.0 |

The FLUX.1 [dev] license (which covers Kontext [dev]) allows non-commercial use of the weights;
it claims no rights over outputs, which may be used commercially except to train a competing
model, and asks for content filtering or review of outputs and AI-generation disclosure where the
law requires it. Read `LICENSE.md` in the repository before use. FLUX.1 [schnell] and the T5
encoder are Apache 2.0.

In the library (`Pipeline::load` takes the same specs as `--model`; `LoadOptions` has `quant`,
`text_quant`, `transformer`, `text_encoder`, `base`, `device`, `cache_dir`, `vae_encoder`, `loras`, `vae`, `redux`;
`Options` are diffusers' call arguments, see `examples/flux_edit.rs`):

```rust
use oxinfer::flux::{LoadOptions, Options, Pipeline};
let pipe = Pipeline::load("QuantStack/FLUX.1-Kontext-dev-GGUF:Q4_0", &LoadOptions::default())?;
let photo = oxinfer::media::Media::new().load_image("photo.png".as_ref())?; // any video::RgbImage
let opts = Options { guidance: Some(2.5), width: Some(photo.width), height: Some(photo.height), ..Default::default() };
let (img, timings) = pipe.edit(&photo, "Make it night", &opts, &mut |step, steps| { eprint!("\r{step}/{steps}"); true })?;
std::fs::write("night.png", oxinfer::png::encode_rgb(img.width, img.height, &img.rgb))?;
let (img, _) = pipe.generate("a red fox in the snow", &Options::default(), &mut |_, _| true)?; // text to image
```

`edit_picture` takes an oxideav `Picture` and `Image::to_picture` returns one (feature
`oxideav`); `cancel_handle()` stops a run from another thread. The C API has `oxi_flux_load`
(`key=value` options), `oxi_flux_edit_rgb`, `oxi_flux_edit_file`, `oxi_flux_generate` (JSON
options, the image through the `oxi_image_fn` callback, steps through `oxi_step_fn`, which can
cancel), `oxi_flux_cancel` and `oxi_flux_free` (`include/oxinfer.h`).

**Reusable parts** (Step1X-Edit, OmniGen2 and BAGEL use the same VAE; Qwen-Image and others the
same scheduler):

| Module | What | API |
|---|---|---|
| `flux::vae` | the 16-channel FLUX `AutoencoderKL`: encoder (new; diffusers' asymmetric-padding downsampler) and `sd::vae::VaeDecoder`, normalization, 2x2 packing | `Vae::load(dir, encoder)`, `Vae::load_bfl(ae.safetensors, encoder)` (BFL layout), `encode(rgb, h, w) -> (latents, lh, lw)`, `decode(latents, lh, lw)`, `pack` / `unpack`, `rgb8_to_unit` / `unit_to_rgb8`, `chw_to_hwc` |
| `flux::sched` | `FlowMatchEulerDiscreteScheduler`: shift, dynamic shifting (exponential / linear), shift_terminal, inverted sigmas | `FlowMatchEuler::load(json)`, `calculate_shift`, `plan(base_sigmas, mu)`, `flux_plan(steps, seq)`, `step`, `scale_noise` |
| `flux::rope` | rotary embeddings over n-axis token ids (FLUX's 16/56/56 split, any split) | `Rope::new(ids, axes, theta)`, `Rope::flux_ids`, `apply(x, heads)` |
| `flux::transformer` | `DoubleBlock` / `SingleBlock` (load from diffusers or BFL names, forward), `FluxTransformer`, `timestep_proj`, `ln_mod` | `FluxTransformer::load(path, quant)`, `forward(img, n_img, txt, n_txt, pooled, t, guidance, rope)` |
| `flux::t5` | T5 v1.1 encoder (relative position buckets, gated GELU), safetensors or GGUF | `T5Encoder::load(dir, tokenizer_dir, quant)`, `encode(text, len)` |
| `flux::weights` | linear layers from safetensors / GGUF at bf16, q8, q4 or any GGUF type, row splitting | `Loader { files, quant }`, `dense`, `split`, `rows`; `Dense::forward`, `forward_vec` |

**Agreement with diffusers** (`tools/ref_flux.py` runs diffusers 0.42 / transformers 5.22 stage by
stage on the same inputs and noise, `OXINFER_FLUX_DUMP=dir` writes oxinfer's stages; float32 on
both sides, oxinfer with `OXINFER_METAL_MMA=0` unless noted):

| Stage | rel L2 | |
|---|---|---|
| T5-XXL, 512 tokens (bf16 weights as stored) | 2.0e-5 | |
| CLIP-L pooled embedding | 1.7e-6 | |
| VAE encode (512x512 reference image) | 4.8e-5 | weights fp16 -> bf16 |
| VAE decode | 5.6e-7 | 97.5 dB |
| transformer, 2 double + 2 single blocks of the bf16 checkpoint, 2560 tokens | 2.9e-6 | `OXINFER_FLUX_LAYERS=2,2` vs the truncated diffusers model |
| transformer, 1 + 1 blocks of a Q4_0 GGUF, weights dequantized by gguf-py | 2.8e-6 | |
| whole transformer (19 + 38 blocks), Q4_0 GGUF, one step | 5.6e-6 | vs diffusers' GGUF loader, 2560 tokens |
| Kontext edit, 512x512, 8 steps, Q4_0 GGUF, CPU | latents 3.1e-3 | 55.1 dB vs diffusers f32 |
| FLUX.1 [schnell] Q4_0, 512x512, 4 steps | latents 9.4e-4 | 61.6 dB |
| same, Metal matrix units (half-precision operands, the default) | latents 5.2e-2 | 37.8 dB (half-precision attention too); the same image to the eye |
| Kontext edit, bf16, 28 steps, Metal default vs diffusers bf16 on MPS | | 43.4 dB |

T5 at `--text-quant q8` is 1.9e-2 from bf16; city96's Q8_0 T5 GGUF is 11% off, almost all of it
from its quantized token-embedding table (rows reach 231 in magnitude; the same weights in
transformers give the same 11%), so the T5 GGUF files are for memory, not fidelity.

**Speed and memory** on the M5 Max (18 cores, 128 GB, shared with other jobs: load average 12-27
during these runs, so take the numbers as upper bounds). Per denoising step, steady state (the
first step also uploads the weights); an edit at 1024x1024 has 8704 tokens per pass (512 text +
4096 noise + 4096 reference), 512x512 with `--no-auto-resize --max-area 262144` 2560:

| Transformer | Metal, 1024^2 | Metal, 512^2 | CPU, 1024^2 | CPU, 512^2 | peak footprint (1024^2, Metal) | max RSS |
|---|---|---|---|---|---|---|
| bf16 (mapped) | 9.4 s | 1.9 s | | 41 s | 12.7 GB | 15.6 GB |
| `--quant q8` | 12.4 s | | | | 31.5 GB | 52.7 GB |
| `--quant q4` | 10.7 s | | | | 22.3 GB | 50.5 GB |
| Q8_0 GGUF (`--text-quant q8`) | 10.9 s | | | | 18.2 GB | 24.4 GB |
| Q4_0 GGUF (`--text-quant q4`) | 13.8 s | 1.6 s | 208 s | 43 s | 15.6 GB | 23.1 GB |

A whole 28-step 1024^2 edit takes 4.5-6.5 minutes on Metal (text encoding 1-2 s, VAE encode
2 s, decode 3-4 s) and about 1.6 hours on the CPU alone; at 512^2 it is about a minute on Metal.
The peak footprint counts the process's own memory (quantized copies, activations, Metal
buffers), max RSS adds the mapped weight pages, which the OS can drop and re-read: bf16 and GGUF
weights are used in place from the mapped files, so a machine with less RAM than the weights
streams them from storage (slowly) instead of failing; `--quant q8|q4` makes in-memory copies
and reads the bf16 files once at load (mapped, so the RSS shows them; the footprint is what
stays). For small machines the GGUF files are the way: a Q4_0 transformer with a Q4_K_M T5
(`--text-encoder city96/t5-v1_1-xxl-encoder-gguf:Q4_K_M`) is 9.7 GB on disk, all of it mapped.
On Metal, every matmul runs on the GPU (M5's matrix units: half-precision operands) and
attention uses a fused kernel on the matrix units (`flash_attn128_mma`: 7.2 -> 3.6 s per 1024^2
step against the f32 kernel); norms, RoPE and the gates stay on the CPU. `OXINFER_METAL_MMA=0`
gives the f32 kernels (the exact rows of the agreement table, 2-2.5x slower).

**Sample edits** (FLUX.1 Kontext [dev] Q8_0, guidance 2.5, 28 steps, 1168x880 from 1024x768 inputs
that FLUX.1 [schnell] generated): "Put a red knitted hat on the dog" keeps the dog, porch and
lighting and adds a knitted beanie with a pom-pom; "Make it night time, with the porch lamps lit"
darkens the scene with both wall lamps glowing; 'Change the text on the sign from "FRESH BREAD" to
"OXINFER CAFE"' rewrites the carved shop sign in the same typeface and wood, leaving the rest of the
storefront unchanged; "Turn this photo into a watercolor painting" restyles the whole picture.

**More FLUX.1 modes.** `Pipeline::render(prompt, &Inputs, &Options, progress)` runs all of them
(`Inputs { references, init, mask, control, image_prompts }`; `generate` and `edit` are shortcuts),
and the CLI picks the mode from the flags and the transformer's input width:

```sh
# several reference images for Kontext ("the woman from the first image wearing the hat from the second")
./target/release/oxinfer --model QuantStack/FLUX.1-Kontext-dev-GGUF:Q4_0 --init-image woman.png \
    --init-image hat.png -p "The woman wears the hat" -o out.png
# a LoRA (here a few-step one: 8 steps instead of 28) on a GGUF transformer
./target/release/oxinfer --model city96/FLUX.1-dev-gguf:Q4_0 --steps 8 \
    --lora alimama-creative/FLUX.1-Turbo-Alpha/diffusion_pytorch_model.safetensors -p "a red fox in the snow" -o fox.png
# image to image (FluxImg2ImgPipeline) and inpainting by mask blending (FluxInpaintPipeline)
./target/release/oxinfer --model city96/FLUX.1-dev-gguf:Q4_0 --init-image photo.png --strength 0.6 -p "an oil painting" -o oil.png
./target/release/oxinfer --model city96/FLUX.1-dev-gguf:Q4_0 --init-image photo.png --mask mask.png --strength 0.85 -p "a red sofa" -o sofa.png
# FLUX.1 Fill (a 384-channel transformer: FluxFillPipeline; guidance 30 is BFL's setting)
./target/release/oxinfer --model flux1-fill-dev-Q4_0.gguf --init-image photo.png --mask mask.png --guidance 30 -p "a red sofa" -o sofa.png
# FLUX.1 Canny / Depth: BFL's control LoRA on FLUX.1 [dev] (or a 128-channel Canny / Depth transformer)
./target/release/oxinfer --model city96/FLUX.1-dev-gguf:Q4_0 --lora flux1-canny-dev-lora.safetensors \
    --control-image edges.png --guidance 30 -p "a glass house in a forest" -o house.png
# FLUX.1 Redux: an image as the prompt (variations; -p adds text)
./target/release/oxinfer --model city96/FLUX.1-dev-gguf:Q4_0 --redux Runware/FLUX.1-Redux-dev --image-prompt photo.png -p "" -o variation.png
# BFL single-file checkpoints: the transformer and the autoencoder
./target/release/oxinfer --model flux1-dev.safetensors --vae ae.safetensors -p "a red fox in the snow" -o fox.png
```

| Flag | Meaning |
|---|---|
| `--init-image FILE` (repeated) | Kontext: several references. Each is VAE-encoded on its own (resized to its own preferred resolution) and appended as its own token sequence |
| `--ref-layout offset\|index\|diffusers` | where the references' tokens sit (RoPE ids `[index, row, column]`; one reference is `[1, y, x]` in all three). `offset` (default; ComfyUI's default for FLUX Kontext): index 1 for all, each reference beside or below the previous ones, whichever keeps the canvas more square. `index` (ComfyUI's `index` method): reference i at index i + 1, all at the origin. `diffusers`: the ids of diffusers' community `pipeline_flux_kontext_multiple_images.py` (every reference resized to the first one's size, index 1, reference i shifted by i x the grid height along the columns). BFL's own code and diffusers' `FluxKontextPipeline` take one image only |
| `--strength F` | `--init-image` becomes the start of an image-to-image run instead of a Kontext reference: the image is noised to step `int(steps * (1 - F))` of the schedule and denoised from there (default 0.6 as diffusers; 1 for Fill) |
| `--mask FILE` | inpaint `--init-image`: white is repainted (resized like the image, luma thresholded at 50%). FLUX.1 Fill reads the masked image's latents and the mask as 320 more input channels per token; any other FLUX.1 blends in latent space as diffusers' `FluxInpaintPipeline` (after each step the kept region is reset to the image at that step's noise level, mask sampled per latent pixel) |
| `--control-image FILE` | the edge / depth map (as an image; computing Canny edges or depth is up to you) for FLUX.1 Canny / Depth: its latents ride along every noise token (128 input channels) |
| `--image-prompt FILE`, `--redux SPEC`, `--image-prompt-scale F` | FLUX.1 Redux: the image through SigLIP so400m/14 at 384 px and the Redux projector into 729 tokens appended to the T5 tokens (repeatable); the scale multiplies them |
| `--lora FILE[:SCALE]` | LoRA files on the transformer (repeatable; a local file or `org/name/file.safetensors`) |
| `--vae SPEC` | the autoencoder from BFL's `ae.safetensors` (or a diffusers `vae` folder) |

**LoRAs** are applied at load as a low-rank side path on each targeted layer
(`y = W x + scale * alpha / rank * up (down x)`, f32, after the base matmul wherever that ran): the
same function as the merged weights `W + scale * alpha / rank * up @ down`, with `W` left as it is
stored (mapped bf16, any GGUF block format, q8 / q4). Rewriting a quantized matrix would mean
dequantizing and requantizing it, and the update of a few-step LoRA is smaller than a Q4 block's
rounding step, so the side path is both exact and free of a 24 GB copy; it costs two small matmuls
per layer and step. Key namings read: diffusers / PEFT (`transformer.<module>.lora_A.weight`,
alpha from the `lora_adapter_metadata` header), kohya / ai-toolkit (`lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight`
with `.alpha`, `lora_transformer_*`), BFL / ComfyUI (`[diffusion_model.]double_blocks.0.img_attn.qkv.lora_A.weight`,
with `lora_B.bias` and the QK-norm scales of BFL's control LoRAs, whose `img_in` update widens the
input to 128 channels) and XLabs (`double_blocks.0.processor.qkv_lora1.down.weight`). Modules on
BFL's fused matrices (`qkv`, `linear1`, the final modulation) are split by rows over the layers
oxinfer keeps. A key that is not a low-rank pair (DoRA, LoKr, LoHa) or a module that matches no
layer is an error; text-encoder modules (`lora_te1_*`, `text_encoder.*`) are not applied and the
load line says how many there were (`[lora FILE x1: 504 modules applied, rank 64]`). Few-step
LoRAs keep the model's scheduler; give `--steps` yourself.

**Single files.** `--model` (or `--transformer`) takes a BFL-layout safetensors file
(`flux1-dev.safetensors`, `flux1-kontext-dev.safetensors`, also with ComfyUI's
`model.diffusion_model.` prefix; bf16 / fp16 / f32, not fp8) the way it takes a GGUF: names are
remapped to the same layers (fused `qkv` / `linear1` split by rows, zero-copy), the configuration
comes from the tensor shapes, the other parts from `--base`. `--vae ae.safetensors` loads the
autoencoder from BFL's layout.

Measured against diffusers 0.42 with `tools/ref_flux_ext.py` (it runs both sides on the same
noise). On a tiny random-weight pipeline from `tools/flux_tiny.py` (2 + 2 blocks, the real
tokenizers; float32 on both sides, CPU), end to end:

| Check (`--mode all`) | final latents, rel L2 | image |
|---|---|---|
| image to image (strength 0.6 of 6 steps; 0.35 of 10 with a resize) vs `FluxImg2ImgPipeline` | 2.3e-6, 2.8e-6 | 86.9, 84.6 dB |
| inpainting (strength 0.8; 1.0) vs `FluxInpaintPipeline` | 2.2e-6, 1.9e-6 | 86.9, 84.3 dB |
| Fill (384 channels; strength 1 and 0.7) vs `FluxFillPipeline` | 2.7e-6, 2.7e-6 | 85.3, 85.6 dB |
| control (128 channels) vs `FluxControlPipeline` | 1.7e-6 | 88.9 dB |
| 3 Kontext references, `offset` / `index` / `diffusers` ids vs a diffusers Kontext loop over ComfyUI's / the community pipeline's ids | 1.6e-6, 1.6e-6, 1.5e-6 | 87.0, 87.0, 86.5 dB |
| LoRA, PEFT keys, vs `load_lora_weights` | 3.3e-6 | 84.3 dB |
| LoRA, kohya keys (fused `qkv` / `linear1`, per-module alpha); XLabs keys; PEFT x0.7 + kohya x-0.5 | 2.3e-6, 2.2e-6, 2.3e-6 | 89.0, 85.2, 85.6 dB |
| BFL control LoRA (`lora_B.bias`, norm scales, 128-channel `img_in`) on a 64-channel model vs `FluxControlPipeline` | 3.6e-6 | 83.9 dB |
| BFL single-file transformer + `ae.safetensors` vs the diffusers folder (and with the kohya LoRA) | 1.9e-6 (2.3e-6) | 88.2 dB (89.0) |
| Redux (tiny SigLIP + projector) vs `FluxPriorReduxPipeline` | 1.9e-6, tokens 6.4e-7 | 84.6 dB |

diffusers' converters for kohya / XLabs / BFL LoRA keys and for single files assume FLUX.1's
3072-wide layers and cannot read a tiny model, so for those rows the reference merges
`W + scale * alpha / rank * up @ down` into the BFL-layout weights in Python and runs the diffusers
pipeline on the result; where diffusers samples the VAE posterior the reference takes its mode,
as oxinfer does. On real weights (FLUX.1 [dev] Q4_0 GGUF, one transformer call at 256x256 against
diffusers' GGUF loader with `load_lora_weights`, on oxinfer's own text embeddings; f32, CPU):

| Check (`--mode step`, `--mode redux-tokens`) | rel L2 |
|---|---|
| alimama-creative/FLUX.1-Turbo-Alpha (PEFT keys, 504 modules, rank 64); the LoRA itself moves the prediction by 0.20 | 2.7e-6 |
| XLabs-AI/flux-RealismLora (XLabs keys, 76 modules, rank 16); moves the prediction by 0.12 | 3.8e-6 |
| BFL's flux1-canny-dev-lora (control LoRA, 314 low-rank modules of rank 64-128 with biases + 152 norm scales), 256x192 with a control image, vs `FluxControlPipeline`'s loader with `--fix-norm-out` | 1.7e-6 |
| the same against diffusers 0.42 as it is | 3.9e-2 |
| Redux tokens (SigLIP so400m + projector of FLUX.1-Redux-dev, bf16 weights) vs transformers | 9.2e-6 |

The last row is diffusers: its converter for BFL control LoRAs swaps the (shift, scale) halves of
`final_layer.adaLN_modulation.1`'s `lora_A` rows as well as `lora_B`'s, but `lora_A`'s rows are
rank components, not output channels, so the product changes; `--fix-norm-out` swaps them back in
the reference (with only that change the two agree, also on 1 + 1 blocks in f32: 1.8e-6). oxinfer
applies the update as BFL's own code does.

Not run on real weights: FLUX.1 Fill and the 128-channel Canny / Depth transformers (plumbing
checked on the tiny models only), BFL's 24 GB single files (the same tensors as the GGUF files,
which load through the same BFL-layout path), whole few-step generations (no picture was
looked at: the LoRA rows above compare one transformer call), and multi-reference Kontext on the
real model (the ids are checked, the picture quality of each layout is not). Redux' SigLIP resize
is Pillow's bicubic (transformers' PIL processor; its torchvision processor antialiases
differently).

Not supported: IP-Adapter (XLabs, InstantX), ControlNets, DoRA / LoKr / LoHa LoRAs and LoRAs on
the text encoders, fp8 single-file checkpoints, masks or `--strength` together with Kontext
references (diffusers' `FluxKontextInpaintPipeline`), the C API and server for the modes above
other than LoRA / VAE loading and `strength` (use the library's `Pipeline::render`), and the
browser build (the weights exceed its 4 GB).
