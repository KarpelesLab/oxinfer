# Stable Diffusion 3 / 3.5: text to image and image to image

Stable Diffusion 3 medium and Stable Diffusion 3.5 (medium, large, large-turbo) are rectified-flow
models like FLUX.1, with their own transformer (MMDiT) and three text encoders. `src/sd3/` follows
diffusers' `StableDiffusion3Pipeline` and `StableDiffusion3Img2ImgPipeline`. A diffusers folder
whose `model_index.json` names a `StableDiffusion3*` class, or an SD 3 GGUF transformer, switches
oxinfer to it:

```sh
# text to image (28 steps, guidance 7, 1024x1024 by default)
./target/release/oxinfer --model DIR -p "a photograph of a red fox sitting in a snowy forest" \
    --negative "blurry, low quality" --size 1024x768 -o fox.png
# without T5 (9.5 GB less to load; prompts are then read by the two CLIP encoders only)
./target/release/oxinfer --model DIR --no-t5 -p "a watercolor lighthouse" -o lighthouse.png
# image to image: --strength of the schedule is redone from the input (default 0.6)
./target/release/oxinfer --model DIR --init-image photo.png --strength 0.6 -p "a watercolor painting" -o wc.png
# the transformer as a GGUF quant; the text encoders, VAE and tokenizers come from --base
./target/release/oxinfer --model city96/stable-diffusion-3.5-medium-gguf:Q4_0 --base DIR -p "..." -o out.png
```

`DIR` is a local folder or a repo id in the diffusers layout. Stability AI's own repositories
(`stabilityai/stable-diffusion-3-medium-diffusers`, `stabilityai/stable-diffusion-3.5-medium`,
`-3.5-large`, `-3.5-large-turbo`) are gated: set `HF_TOKEN` after accepting their license, or
point `--model` at a folder you already have.

| Flag | Meaning |
|---|---|
| `-p TEXT`, `--negative TEXT` | the prompt; the negative prompt of guided runs (default empty) |
| `--steps N`, `--guidance F`, `--seed N` | 28 steps (50 image to image) at classifier-free guidance 7, as diffusers; models with "turbo" in their name default to 4 steps without guidance; seed 42 |
| `--size WxH` | output size, rounded down to multiples of 16 (default 1024x1024; image to image: the input's size, where diffusers would resize it to 1024x1024) |
| `--init-image FILE`, `--strength F` | image to image: the starting image and how much of the schedule is run (0..1, default 0.6) |
| `--no-t5` | do not load T5-XXL: its tokens are zeros, as diffusers does with `text_encoder_3=None` (also when the folder has no `text_encoder_3` weights) |
| `--max-sequence-length N` | T5 tokens, default 256, at most 512 |
| `--quant bf16\|q8\|q4` | transformer precision when stored as float; `--text-quant` sets T5's (default: `--quant`) |
| `--transformer SPEC` | a transformer replacing the folder's: GGUF file or `org/name:QUANT` |
| `--text-encoder SPEC` | a T5 encoder GGUF (`city96/t5-v1_1-xxl-encoder-gguf:Q8_0`) replacing `text_encoder_3/` |
| `--base SPEC` | with a GGUF `--model`: the diffusers SD 3 folder the text encoders, VAE and tokenizers come from (required: there is no ungated default) |

**What runs.** The prompt goes through three encoders. CLIP-L and OpenCLIP-bigG
(`CLIPTextModelWithProjection`, `sd3::clip`) each give their penultimate hidden states (77
tokens, 768 and 1280 wide) and a projected pooled embedding; the states are concatenated to 2048
channels and zero-padded to 4096, then followed by the T5-XXL states (256 tokens, `flux::t5`), so
the transformer reads 333 text tokens of 4096; the two pooled embeddings (2048 together) join the
timestep embedding as the conditioning vector. The latent (16 channels, 1/8 of the image) is cut
into 2x2 patches by a convolution, and each patch gets a row of the checkpoint's 2-D sinusoidal
position table, cropped around its centre (192x192 patches stored in SD 3 medium and 3.5 large,
384x384 in 3.5 medium). Every joint block (`sd3::transformer`) normalizes the two streams with
scale / shift / gate vectors computed from the conditioning vector, runs one attention over
`[text; image]` with separate projections per stream and a feed-forward per stream; SD 3.5 adds RMS
norms on queries and keys, SD 3.5 medium (MMDiT-X) a second attention over the image tokens
alone in its first 13 blocks, and the last block does not update the text stream. Guidance is
classifier-free: with a scale above 1 each step runs a second pass on the negative prompt and
combines `uncond + scale * (cond - uncond)`. The schedule is `FlowMatchEulerDiscreteScheduler`
with the config's constant shift (3 for every SD 3 model) or, when the config sets
`use_dynamic_shifting`, the resolution-dependent `mu` from the number of image tokens; steps are
Euler, `x += (sigma_next - sigma) * v`. Image to image encodes the input with the VAE (a sample of
the posterior, `(z - 0.0609) * 1.5305`), mixes it with noise at the starting sigma
(`sigma * noise + (1 - sigma) * z`) and runs the last `steps * strength` steps. The 16-channel VAE
is the `AutoencoderKL` code FLUX uses (`flux::vae`), with SD 3's scale and shift from its config.

**Weights.** The transformer loads from `transformer/` (float weights run as stored from the
mapped file, `--quant q8|q4` quantizes at load) or from a GGUF file in Stability AI's original
layout (`joint_blocks.N.x_block.attn.qkv`; city96's SD 3 / SD 3.5 quants, any GGUF block format,
zero-copy, fused `qkv` rows split into separate layers), whose configuration (width, depth,
QK norm, dual-attention blocks, position table size) is read from the tensor shapes. The CLIP
encoders run their fp16 weights as stored.

In the library (`Pipeline::load` takes the same specs as `--model`; `LoadOptions` has `quant`,
`text_quant`, `transformer`, `text_encoder`, `base`, `t5`, `device`, `cache_dir`, `vae_encoder`;
see `examples/sd3_generate.rs`):

```rust
use oxinfer::sd3::{LoadOptions, Options, Pipeline};
let pipe = Pipeline::load("models/stable-diffusion-3.5-medium", &LoadOptions::default())?;
let opts = Options { negative_prompt: Some("blurry".into()), width: Some(1024), height: Some(768), ..Default::default() };
let (img, timings) = pipe.generate("a red fox in the snow", &opts, &mut |step, steps| { eprint!("\r{step}/{steps}"); true })?;
std::fs::write("fox.png", oxinfer::png::encode_rgb(img.width, img.height, &img.rgb))?;
let photo = oxinfer::media::Media::new().load_image("photo.png".as_ref())?; // any video::RgbImage
let (img, _) = pipe.img2img(&photo, "a watercolor painting", &Options { strength: 0.6, ..Default::default() }, &mut |_, _| true)?;
```

`encode_prompt` returns the conditioning alone, `sd3::sd3_plan` the schedule, and
`cancel_handle()` stops a run from another thread.

**Agreement with diffusers** (real weights: `v2ray/stable-diffusion-3-medium-diffusers` and
`adamo1139/stable-diffusion-3.5-medium-ungated`, ungated copies of Stability AI's SD 3 medium and
SD 3.5 medium repositories, and `city96/stable-diffusion-3.5-medium-gguf`. `tools/ref_sd3.py` runs
diffusers 0.42 / transformers 5.22 in float32 stage by stage on the same noise,
`OXINFER_SD3_DUMP=dir` writes oxinfer's stages; 512x512, 4 steps at guidance 7 with a negative
prompt, so 8 transformer passes of 1357 tokens, `--device cpu`):

| Stage | rel L2 | |
|---|---|---|
| CLIP-L penultimate states / pooled embedding | 1.6e-6 / 1.6e-6 | |
| OpenCLIP-bigG penultimate states / pooled embedding | 5.1e-6 / 1.9e-6 | |
| T5-XXL, 256 tokens | 1.3e-5 | |
| schedule (sigmas, shift 3) | 4.1e-8 | also unit-tested against the scheduler with dynamic shifting |
| SD 3 medium transformer, one pass (24 blocks) | 3.6e-6 | |
| SD 3.5 medium transformer, one pass (24 blocks, QK norm, 13 dual-attention blocks) | 1.3e-6 | |
| SD 3.5 medium transformer, Q4_0 GGUF, one pass | 5.9e-5 | vs diffusers' GGUF loader on the same file |
| VAE decode, SD 3.5 medium's VAE | 1.7e-6 | 89.4 dB |
| VAE decode, SD 3 medium's VAE | 4.9e-4 | 62.3 dB |
| SD 3 medium, 4 guided steps end to end | latents 4.8e-5 | 62.3 dB (the VAE decode above) |
| same with `--no-t5` | latents 6.0e-5 | 62.0 dB |
| SD 3.5 medium, 4 guided steps end to end | latents 2.2e-4 | 67.3 dB |
| SD 3.5 medium Q4_0 GGUF, 4 guided steps end to end | latents 1.3e-3 | 58.1 dB |
| SD 3 medium image to image, strength 0.6 of 5 steps (3 run) | latents 4.4e-3 | 54.2 dB; the posterior sample from the VAE encoder is 2.6e-3 from diffusers' |
| SD 3.5 medium on CUDA (RTX 5080), 4 guided steps end to end | latents 2.1e-4 | 67.4 dB; `OXINFER_GPU_CHECK=1`: attention 1.9e-6, VAE convolutions with fused norms 4.5e-7 against the CPU ops |
| SD 3.5 medium Q4_0 GGUF as `--model` with `--base`, on CUDA | latents 1.3e-3 | 58.1 dB |

With a GPU, the joint attention and the VAE's convolutions, norms and attention run on it through
the same backend operations as FLUX and Stable Diffusion 1.x / 2.x; the rest is on the CPU.

Not run: SD 3.5 large and large-turbo. They use the same code path as SD 3 medium with QK norms
(38 blocks of 38 heads, no dual attention), both pieces verified above on the medium models, but
their 16 GB transformers were not downloaded, so no agreement is claimed for them. The "turbo"
defaults (4 steps, no guidance) are untested for the same reason. `--quant q8|q4` on the
safetensors transformer and resolution-dependent shifting in a full run (no SD 3 config sets it)
were not measured either.

Not supported: ControlNet, IP-Adapter, LoRAs on SD 3, inpainting, skip-layer guidance, `clip_skip`,
prompts longer than 77 CLIP tokens (truncated, as in diffusers), separate prompts per encoder
(`prompt_2`, `prompt_3`), single-file checkpoints with the text encoders inside
(`sd3.5_medium.safetensors`; the diffusers folder and the GGUF files cover the same weights),
the C API and the browser build.
