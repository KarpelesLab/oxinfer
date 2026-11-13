# Step1X-Edit: image editing from instructions

StepFun's Step1X-Edit edits an image the way an instruction says. `src/step1x/` follows the
`Step1XEditPipelineV1P2` / `Step1XEditPipeline` of StepFun's diffusers fork (branch
`step1xedit_v1p2` of Peyton-Chen/diffusers); v1.2 (`stepfun-ai/Step1X-Edit-v1p2`, "ReasonEdit-S",
and `-v1p2-preview`) and v1.1 (`stepfun-ai/Step1X-Edit-v1p1-diffusers`) are told apart by the
transformer's `enable_text_token_mapping`.

```sh
# 28 steps, true CFG 6 against an empty instruction (the pipeline's defaults; StepFun's examples use 50)
./target/release/oxinfer --model stepfun-ai/Step1X-Edit-v1p2 --mmap --init-image dog.png \
    -p "Put a red knitted hat on the dog" -o hat.png
# quoted text is rendered character by character
./target/release/oxinfer --model stepfun-ai/Step1X-Edit-v1p2 --mmap --init-image bakery.png \
    -p 'Change the text on the sign from "FRESH BREAD" to "OXINFER CAFE"' -o sign.png
# thinking mode: the VL model rewrites the instruction from the image first
./target/release/oxinfer --model stepfun-ai/Step1X-Edit-v1p2 --mmap --init-image cabin.png --think \
    -p "Show what this place would look like after 100 years of abandonment" -o old.png
# a 512x512 output worked at 512x512 (4x fewer tokens), q4 transformer
./target/release/oxinfer --model stepfun-ai/Step1X-Edit-v1p2 --quant q4 --init-image cat.png \
    --size 512x512 -p "Put sunglasses on the cat" -o cat.png
```

| Flag | Meaning |
|---|---|
| `--init-image FILE` | image to edit (without one: text to image, the vision tower seeing a black square) |
| `--steps N`, `--guidance F`, `--negative TEXT` | 28, true CFG 6, `""` |
| `--size WxH` | output size and working area; the input is center-cropped to that aspect ratio. Default: the input's size, worked at about 1024x1024 (multiples of 16) at its aspect ratio |
| `--think` | v1.2 thinking mode (sampled rewrite, temperature 0.7, top-p 0.9, up to 128 tokens) |
| `--sample-posterior` | sample the VAE posterior of the input image instead of taking its mean (off by default) |
| `--quant q8\|q4` | transformer precision (the text encoder: q8 with either) |
| `--mmap`, `--stream` | map the weights; stream the transformer's 57 blocks from storage (`--cache-budget-gb`) |

What runs: the input is resized (Lanczos) to the working size and given to Qwen2.5-VL-7B (v1.2
ships a fine-tuned copy) inside the pipeline's template: a system turn, a fixed editing prefix, the
instruction, then `Picture 1: <|vision_start|>...<|vision_end|>` with 256 to 400 vision tokens.
The template is split at special tokens and quotes and each piece tokenized alone, so quoted text
becomes one token per character; the text encoder gets explicit M-RoPE positions in which, as the
pipeline's `attention_mask = input_ids > 0`, a token with id 0 (a lone `!`) sits at position 1
and the rest close the gap. The last hidden states minus the first 90 tokens condition the DiT. A
connector (HunyuanVideo's token refiner: two self-attention blocks gated by the timestep embedding
plus a projection of the states' mean) maps them to the 4096-wide text stream and pools the
768-wide vector FLUX takes from CLIP; v1.2 adds the raw token embeddings of the quoted characters
through `text_token_mapping`. The DiT is FLUX.1's (19 double + 38 single blocks, the
`crate::flux::transformer` blocks as they are) over `[text; target; reference]`: the reference is
the input resized with torch's antialiased bilinear filter and VAE-encoded (the 16-channel FLUX
VAE), on the target's grid at RoPE index 1 (v1.1: shifted by one row and column). v1.2 gives the
text tokens negative positions on the first RoPE axis and the vision tokens the latent-patch
coordinates of the pixels they cover. Flow matching with FLUX's resolution-dependent shift; true
CFG where the guidance difference is divided by its per-token norm^0.4 when that norm exceeds 1.
The decoded image is resized back to the input's size (Pillow bicubic). v1.1 pads the text stream
to 640 tokens (masked in the connector, not in the DiT) with all text positions at 0.

Components load when first needed; the 16 GB text encoder is dropped after encoding. On Metal the
text encoder and the VAE encoder run with exact (f32) matmuls, and the VAE's f32 weights keep their
precision as a bf16 `hi + lo` pair (Step1X ships the FLUX VAE in f32: rounding its weights to bf16
moves the reference latents by 1.5%, the final image by 20 dB). The transformer uses the matrix
units, attention included (`exact_attention=1` / `OXINFER_EXACT_ATTENTION=1` gives it f32
operands, still on the matrix units).

**Agreement** with the StepFun diffusers fork + transformers 4.55 (bf16 weights on both sides,
every reference matmul in f32, VAE posterior mean, same initial noise; v1.2, a 512x512 cat photo,
`Add a red hat to the cat and write "MEOW" on a sign!` (quotes and a lone `!`), `--size 512x512`,
8 steps, CFG 6; relative L2 errors):

| oxinfer | first velocity (cond / uncond) | final latents | image PSNR |
|---|---|---|---|
| bf16, CPU (1 step) | 9.7e-5 / 3.5e-5 | 4.1e-4 | 62.0 dB |
| bf16, Metal, f32 attention | 3.4e-4 / 2.4e-4 | 1.1e-2 | 48.7 dB |
| bf16, Metal (default) | 3.3e-4 / 2.6e-4 | 1.3e-2 | 47.3 dB |
| `--quant q8`, Metal | 3.8e-3 / 2.5e-3 | 6.6e-2 | 33.9 dB |
| `--quant q4`, Metal | 4.6e-2 / 4.3e-2 | 0.25 | 21.7 dB |
| v1.1 (`-v1p1-diffusers`), Metal, 1024x1024, 1 step | 1.4e-2 / 2.7e-4 | 1.7e-2 | 49.5 dB |

Token ids are identical (both prompts), the conditioning hidden states agree to 2.1e-4, the RoPE
ids of the text to 2e-8, the reference latents to 3.5e-5. The quantized rows were also compared
with a reference running oxinfer's Q8 / Q4_0 weights dequantized: the weights alone move the first
velocity by 2.8e-3 (Q8) and 4.5e-2 (Q4); oxinfer's quantized matmuls add 2.6e-3 / 3.5e-3 on top,
so the Q4 error is the 4-bit weights, not the activations. CFG 6 then amplifies differences over
the steps: quantized edits look the same (same hat, same lettering, same composition) with
different fine detail. Below ~1e-2 the final latents are chaotic: multiplying the queries of the
f32 attention by 1 +- 1e-7 (random signs, about one f32 rounding step) moves them anywhere between
6e-3 and 1.4e-2 against the reference (52 to 47 dB), so the bf16 rows' final numbers are draws from
that band, and their first velocities are what measures precision: half-precision attention
leaves those unchanged. v1.1's text encoder also masks the `!` token as a key; oxinfer reproduces
that for every other token, while the `!` row itself still sees its own key (that one row is 36%
off, hence the 1.4e-2 on the conditioned velocity; prompts without a lone `!` match like v1.2).

**Speed and memory** (M5 Max, 1024x768 input worked at 1168x880: 4015 target + 4015 reference +
~410 text tokens per pass, two passes per step):

| weights | device | per transformer pass | 28-step edit | peak memory (footprint) |
|---|---|---|---|---|
| bf16 (`--mmap`, 24.9 GB mapped) | Metal | 9.3 s (f32 attention: 34% slower) | 9 min | 12.6 GB |
| bf16, read into memory | Metal | 17.5 s | | 40 GB |
| `--quant q8` | Metal | 14.0 s | | 28 GB |
| `--quant q4` | Metal | 14.2 s | | 20 GB |
| bf16, `--stream --cache-budget-gb 8` | Metal | 13.6 s | | 12.7 GB |
| bf16, 512x512 (2400 tokens) | Metal | 3-4 s | 3.5 min | 3.7 GB |
| bf16, 512x512 | CPU, 18 threads | 34 s | | |
| bf16, 1024x1024 (8832 tokens, v1.1) | CPU, 18 threads | 213 s | | |

The text encoder takes 3-6 s, the VAE 2.5 s each way at this size; the thinking mode adds ~9 s.

**Sample edits** (v1.2 bf16, 28 steps, default sizes): "Put a red knitted hat on the dog" (a 1024x768
porch photo) adds the hat and leaves every other pixel's content in place, with no shift or zoom
(`--quant q4` gives the same hat); "Change the text on the sign from "FRESH BREAD" to "OXINFER
CAFE"" rewrites the carved sign in the same lettering; "Put sunglasses on the cat", "Make it
winter, with snow on the roof and the ground" (a snowstorm over the same cabin) and "Turn this
photo into a watercolor painting" do what they say. Thinking mode turned "Show what this place
would look like after 100 years of abandonment" into "Change the wooden shed to a dilapidated
and overgrown log cabin." and weathered the cabin only lightly.

**Library and C API.**

```rust
use oxinfer::step1x::{Options, Params, Pipeline};
let mut pipe = Pipeline::load("stepfun-ai/Step1X-Edit-v1p2", Options { mmap: true, ..Default::default() })?;
let img = oxinfer::media::Media::new().load_image("dog.png".as_ref())?;
let (out, timings) = pipe.edit(&img, &Params::new("put a red hat on the dog"), &mut |i, n| eprintln!("{i}/{n}"))?;
// out: width, height, rgb (the input's size); Params { size_level, width, height, think, true_cfg,
// steps, seed, negative, .. }; pipe.generate(&params, ..) without an image; edit_picture with `oxideav`
```

The C API has `oxi_step1x_load` (`key=value` options: quant, text_quant, mmap, transformer,
keep_text_encoder, stream, exact_attention, device), `oxi_step1x_edit` (a JSON request: prompt,
images, negative, true_cfg, steps, seed, size_level, width, height, think, sample_posterior; or
RGB pixels),
`oxi_step1x_cancel` and `oxi_step1x_free` (see `include/oxinfer.h`). `examples/step1x_edit.rs`
edits one image.

**Weights and licenses.** stepfun-ai/Step1X-Edit-v1p2, -v1p2-preview and -v1p1-diffusers: Apache
2.0, ungated; 42 GB each (24.9 GB transformer in bf16, 16.6 GB text encoder, 0.34 GB VAE in f32).
v1.1's text encoder is stock Qwen2.5-VL-7B-Instruct (the same files as Qwen-Image's, which can be
linked instead of downloaded twice); v1.2's is fine-tuned. No GGUF conversions are published:
`--quant q8 / q4` quantizes at load (2-3 s from the mapped bf16 files).

Not supported: the reflection mode of v1.2 (a multi-round loop where the VL model scores the edit
and retries), the original `stepfun-ai/Step1X-Edit` single-file checkpoints (v1.0 / v1.1 in the
original repository's layout; the diffusers folders cover the same models) and its fp8 copies,
LoRAs, the RegionE acceleration, batches, the browser build (the weights exceed its 4 GB).

The reference image is VAE-encoded at the posterior mean by default (the pipeline samples it).
`--sample-posterior` (`"sample_posterior": true`, `Params::sample_posterior`) draws it as
`mean + std * noise` like the pipeline's `retrieve_latents(..., sample_mode="sample")`, with the
log-variance clamped to [-30, 20] as diffusers does. The noise comes from oxinfer's generator
(xoshiro256** seeded from `--seed`, a stream apart from the initial latents'): the pipeline's
distribution, not its draw, since oxinfer does not reproduce torch's generator (checks pass the
reference's noise in, as for the initial latents). So the mean stays the default, and the
agreement table above is measured with it. The sampling code is the one checked against
diffusers' `latent_dist.sample()` on OmniGen2's tiny model (sampled latents within 7.8e-7 with
the same noise), where the same kind of VAE measured a posterior 1.4e-4 of the latents wide.
Step1X-Edit itself was not run with the flag, nor re-run after this change (its 42 GB of weights
are more than this check could load on the shared machine); its default path calls the same
`Vae::encode` as before.
