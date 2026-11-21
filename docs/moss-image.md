# MOSS-Image and Kromoss: text to image, presets

MOSS-Image (OpenMOSS-Team/MOSS-Image-Base / -SFT) is a 5B single-stream diffusion transformer
conditioned by Qwen3.5-9B and decoded by the FLUX.2 VAE. `lodestones/Kromoss` is a full fine-tune
of MOSS-Image-Base at 1024 px with the same layout, so the same code reads all three (each checked against the
release's pipeline, see [Verification](#verification)). `src/moss_image/`
follows the release's diffusers pipeline (`pipeline_moss_image.py`, `MOSSImagePipeline`); a folder
is recognised from `model_index.json` (`MOSSImagePipeline`) or `transformer/config.json`.

```sh
# the release's settings: 50 steps, CFG 3, the negative branch without block 22, shift 5
./target/release/oxinfer --model lodestones/Kromoss -p "a tabby cat holding a sign that says KROMOSS" -o cat.png
# 8 steps, CFG 1 with the release's DMD2 turbo LoRA (a preset)
./target/release/oxinfer --preset kromoss-dmd2 -p "a street market in Hanoi at night" -o market.png
# Kromoss + FameGrid Moss Preview at 1.4, trigger "famegrid", ER-SDE / beta57, 24 steps, CFG 2.5
./target/release/oxinfer --preset kromoss-famegrid -p "a candid portrait photo of a woman in a cafe" -o famegrid.png
# a GGUF transformer (text encoder and VAE from lodestones/Kromoss; or --transformer FILE.gguf)
./target/release/oxinfer --model molbal/Kromoss-GGUF:Q4_K_M -p "..." -o q4.png
# ComfyUI's ConvRot quants: the int8 single file, or molbal's Q8_CR / Q4_CR_L GGUFs
./target/release/oxinfer --model lodestones/Kromoss --transformer lodestones/Kromoss/diffusion_models/kromoss_int8_convrot.safetensors -p "..." -o i8.png
./target/release/oxinfer --model molbal/Kromoss-GGUF:Q4_CR_L -p "..." -o q4cr.png
# the release's own models, and ComfyUI's single-file FLUX.2 VAE (the full-size decoder)
./target/release/oxinfer --model OpenMOSS-Team/MOSS-Image-SFT -p "..." -o sft.png
./target/release/oxinfer --model lodestones/Kromoss --vae Comfy-Org/MOSS-Image/vae/flux2-vae.safetensors -p "..." -o big-vae.png
# the same initial noise as diffusers (generator=torch.Generator("cpu").manual_seed(7)) and ComfyUI (seed 7)
./target/release/oxinfer --model lodestones/Kromoss --noise torch --seed 7 -p "..." -o same-noise.png
```

| Flag | Meaning |
|---|---|
| `--preset ID` | a curated stack (`oxinfer::advisor::PRESETS`): its base, LoRAs, trigger word and settings; flags override them. `--model ID` works too |
| `--steps N`, `--guidance F`, `--negative TEXT`, `--size WxH`, `--seed N` | 50, 3, empty, 1024x1024 (multiples of 16), 42 |
| `--uncond-skip LAYERS` | blocks left out of the negative branch (default `22`; `none`: plain CFG) |
| `--slg-layers LAYERS`, `--slg-scale F`, `--slg-start F`, `--slg-stop F` | skip-layer guidance: an extra positive pass without those blocks, `+ scale (cond - cond_skip)`; the skipping of both kinds applies on steps `start <= i / steps < stop` |
| `--sampler euler\|er_sde`, `--scheduler moss\|simple\|beta\|beta57`, `--shift F` | the sampler and schedule (default: Euler on the release's snapped schedule, shift 5); ComfyUI's `er_sde` and its schedules over the same model sampling |
| `--lora FILE[:STRENGTH]` | a LoRA (repeatable; local or `org/name/file.safetensors`); `--lora-merge` merges low-rank ones too |
| `--trigger WORD` | put first in the prompt unless it is there (presets set it) |
| `--quant q8\|q4`, `--text-quant` | transformer / text-encoder precision |
| `--transformer SPEC`, `--text-encoder SPEC`, `--vae SPEC` | components from elsewhere: a diffusers folder, ComfyUI's single file (`diffusion_models/kromoss_bf16.safetensors`, `kromoss_int8_convrot.safetensors`; for the VAE `flux2-vae_small.safetensors` or BFL-layout `flux2-vae.safetensors`), a GGUF file (Q8_0 / Q6_K / Q4_K_M, Q8_CR / Q4_CR_L) |
| `--noise oxinfer\|torch` | the noise generator: oxinfer's (default) or PyTorch's CPU stream (see [Noise](#noise)) |
| `--moss KEY=VALUE` | any load option or generation setting by name (`oxinfer::moss_image::Options::KEYS`, `Params::KEYS`), e.g. `gpu_pass=0` (the per-op GPU path), `exact_attention=1`, `batch_guidance=0` (see [On the GPU](#on-the-gpu)) |

## What runs

- **Text.** The prompt, trimmed, as a user turn of Qwen3.5's chat template with an open think block
  (`<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n`), through Qwen3.5-9B's
  language model; its final-normed hidden states (4096 wide, one row per token, no padding) are the
  conditioning. The negative prompt (empty by default) goes the same way. Kromoss' `text_encoder/` is
  `Qwen/Qwen3.5-9B` unchanged (the shards' bytes match), so when the cache holds `Qwen/Qwen3.5-9B` the
  download skips the repo's 17 GB copy and reads that one. The text encoder is mapped, run once per
  call, and dropped (`keep_text_encoder=1` keeps it).
- **Transformer** (`src/moss_image/transformer.rs`). 3072 wide, 24 heads of 128, 42 blocks, SiTU
  feed-forward 8192 wide (`(4 tanh(g/4) sigmoid(g)) * (25 tanh(h/25))`, value half then gate half of
  the fused `ffn_up_gate`). The text goes through a projection and two text blocks of its own (once
  per prompt); image tokens (the latent's 128 channels at 1/16 of the size) then text tokens run
  through the blocks with joint attention. One modulation for all blocks (`shared_block_modulation` of
  the time embedding: shift / scale / gate for attention and feed-forward) and one for the output.
  RMS norms with weights, per-head q / k RMS norms. Rotary embeddings on 4 axes of 16 frequencies
  (`10000^(-i/16)`), rotate-half: image token `(y, x)` at `(0, L, y - (H-1)/2, x - (W-1)/2)` for an
  `L`-token prompt, text token `j` at `(0, j, 0, 0)`; the image positions depend on the prompt's
  length, so the conditional and unconditional passes run separately.
- **Guidance.** `uncond + g (cond - uncond)` with the negative branch run without block 22
  (`uncond_skip_layers`); guidance 1 runs only the conditional pass. Optional skip-layer guidance
  adds a third pass.
- **Sampling.** The release's `MOSSImageScheduler`: `steps` points of the shifted grid snapped to the
  nearest of the 1000 training sigmas, Euler steps, timestep `1000 sigma`. ComfyUI's `simple`,
  `beta` and `beta57` (RES4LYF's alpha 0.5 / beta 0.7) schedules over its MOSS model sampling
  (`time_snr_shift(5, i / 1000)`) and its `er_sde` sampler (VP ER-SDE-Solver-3, fresh noise each step)
  are there for recipes written for ComfyUI.
- **Decoding.** The latents times `sqrt(running_var + eps)` plus `running_mean` (the VAE's batch-norm
  statistics, per patchified channel), unpatchified to 32 channels at 1/8, then the FLUX.2 VAE's
  decoder (`post_quant_conv`, Kromoss' small decoder 96 / 192 / 384 / 384 channels;
  `crate::sd::vae::VaeDecoder`). Its float32 weights are kept exact (a bf16 pair each), as the release
  decodes in float32.

Weights: the diffusers `transformer/` (bf16, borrowed from the mapping; `--quant q8|q4` at load),
ComfyUI's single file (fused `proj.qkv`, split by rows without copying) and GGUF files in that layout
(`molbal/Kromoss-GGUF`: Q8_0, Q6_K, Q4_K_M), and ComfyUI's ConvRot quants (`src/moss_image/convrot.rs`):

- `kromoss_int8_convrot.safetensors` (made by the release's `quantize_convrot.py`) and molbal's `Q8_CR`
  GGUF hold the same values: the 168 block linears as int8 `[out, in]` times a float32 scale per row,
  each row rotated by the normalized regular Hadamard matrix `H` (Kronecker powers of
  `H4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]`, symmetric and its own inverse) over groups of
  256 inputs, so that `W x = (W H) (H x)`. The entry saying so is the JSON of tensor
  `{module}.comfy_quant` (safetensors) or of metadata `comfy.gguf.quant.{module}.weight` (GGUF:
  `{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}`);
- `Q4_CR_L` keeps `qkv` and `o` in that int8 form and stores the feed-forward as `int4_cr`: two
  codes per byte (column `2 i` in the low nibble of byte `i`, two's complement -8..7) times a
  float16 scale per row, rotated the same way (ComfyUI-GGUF's Q4_CR; the `w4a4` / `w4a8` backings
  and `quant_group_size` only name its kernels).

oxinfer keeps the stored values exactly (int8 rows as its int8 blocks with the row scale per block,
borrowed from the mapping; int4 rows as ggml `Q4_0` blocks with the row scale) and rotates the layer's
input instead: on the CPU through the Hadamard-folded weights of `crate::hadamard`, on the GPU in the
whole-pass plan (`dit_rot_half`). LoRAs work over them: a low-rank side path takes the rotated input
with its down matrix rotated too (`A x = (A H) (H x)`), a LoKr update is merged in the original basis,
rotated back and stored as int8 blocks. ComfyUI's kernels also round the rotated activations to int8
(the "A8"); that is not done (activations stay float).

## On the GPU

With Metal on a GPU with matrix units (M5, A19), a transformer pass runs as a whole on the device
(`src/gpu_dit.rs`, `src/metal_dit.rs`, `src/metal_dit.metal`): the model describes its layers once per
call, and every block of every guidance branch of a step is encoded into a few command buffers (6
blocks each, committed as they are encoded) with the activations in device buffers that live as long as
the pipeline. Only the latents, the text tokens, the rotary tables and the modulations go in, only the
velocities come out. The matmuls are the matrix-unit GEMMs of the prefill plans (inputs as half with a
power-of-two scale per row, made by the kernel producing them: the modulated RMS norm, SiTU), attention
the fused `flash_attn128_mma` kernel per sequence (half operands; `exact_attention=1` takes the float32
one), and the per-head q / k norms with the rotary embedding, the gated residuals and biases have
kernels of their own. The branches of a step (conditional, negative, skip-layer) share every matmul:
their rows are stacked (each padded to 64), attention runs per branch, and a block a branch leaves out
is computed but not added for it. `gpu_pass=0` goes back to the per-op path (one GEMM per submission,
the rest on the CPU), `batch_guidance=0` runs the branches one after the other. Without matrix units, or
for a layer the GEMMs do not take, the per-op path runs.

The VAE decoder runs per op on the GPU (its convolutions with the group norm fused in); a layer whose
outputs are not a multiple of 64 (the 96-channel top level of the small decoder, the RGB output) now
runs as a zero-padded copy instead of on the CPU.

## Noise

`noise=torch` (`--noise torch`) draws the initial latents as PyTorch does on the CPU
(`src/torch_rng.rs`: `at::mt19937` seeded as `manual_seed`, then `torch.randn`'s float32 Box-Muller over
blocks of 16): `torch.randn([1, 128, h / 16, w / 16], generator=torch.Generator("cpu").manual_seed(seed))`,
which is both what the release's diffusers pipeline draws with a CPU generator and what ComfyUI's
`prepare_noise` draws for that seed (always on the CPU), bit for bit (checked on 524288 values against
PyTorch 2.15 on macOS; the same libm math as PyTorch's scalar path, so Linux matches Linux; PyTorch's
AVX2 variant on x86 may differ in the last bits). ER-SDE's fresh noise then comes from a second CPU
generator seeded `seed + 1`, one `randn` of the latent's shape per step, as ComfyUI's
`default_noise_sampler` does when ComfyUI samples on the CPU.

This reproduces ComfyUI's noise, not its pixels: when ComfyUI samples on CUDA or MPS, ER-SDE's step
noise comes from that device's generator (Philox on CUDA, Metal's own on MPS), which PyTorch does not
document as a stable algorithm and which is not reproduced here; and with the same noise the image
still differs from ComfyUI's by the arithmetic (its bf16 activations against oxinfer's float32 ones,
the 30 dB of the verification table). With Euler (deterministic once the initial noise is fixed) the
same seed gives ComfyUI's and diffusers' composition.

## LoRAs

`src/moss_image/lora.rs` reads, under a `diffusion_model.` / `model.diffusion_model.` /
`transformer.` / `base_model.model.` prefix or kohya's `lora_unet_` underscores:

- low-rank modules (`lora_down` / `lora_up` / `alpha`, PEFT's `lora_A` / `lora_B`): kept beside the
  stored weights, `strength * alpha / rank * up (down x)` in float32 (on a GPU as two float32 matmuls
  next to the base one, on the CPU by the low-rank kernel of `crate::lora`); the release advises
  against merging its DMD LoRAs into bf16, whose rounding step is about their update's size.
  `--lora-merge` merges them anyway;
- LyCORIS LoKr (`lokr_w1` / `lokr_w2` or their `_a` / `_b` factors) and LoHa, and `diff` tensors:
  full-rank, merged into the weights at load in float32 (`strength * alpha / dim * kron(w1, w2)` with
  ComfyUI's alpha rule), then stored at the transformer's precision;
- ComfyUI's fused `layers.N.proj.qkv` modules, split by rows over q / k / v.

A module that matches no layer is an error; text-encoder modules are counted and not applied.

## Presets

| Id | Stack | Defaults |
|---|---|---|
| `kromoss` | lodestones/Kromoss | 50 steps, CFG 3, negative branch without block 22 (the README) |
| `kromoss-dmd2` | + `8-steps-dmd2-lora.safetensors` at 1 | 8 steps, CFG 1 (the README's recommended turbo LoRA) |
| `kromoss-dmd` | + `8-steps-dmd-lora.safetensors` at 1 | 8 steps, CFG 1 |
| `kromoss-famegrid` | + FameGrid Moss Preview at 1.4, trigger `famegrid` | ER-SDE, beta57, 24 steps, CFG 2.5, plain CFG |

FameGrid Moss Preview is UltraMuse's "Famegrid (Moss Image / Krea 2 / Z-Image) - Realism LoRA", version
"Moss Image Spice Preview" (civitai.com/models/2088956?modelVersionId=3478421,
`FameGrid-MOSS-Preview.safetensors`, 581 MB, SHA-256 `e37fa0fb...97406`). Civitai's download needs an
account; `Lnypuuuu/txt2image` on Hugging Face hosts the same file (same SHA-256) without one, and the
preset reads it from there. The page recommends the trigger word `famegrid` at the start of the prompt,
ER_SDE with the beta57 scheduler, 24 steps, CFG 2.5 and a weight of 1.5 to 2, and says Kromoss bf16
removes most of the preview's anatomy problems; the preset uses 1.4 (the requested 1.3 to 1.5 range).
It is a LyCORIS LoKr on the fused `qkv`, `o` and feed-forward matrices of 41 blocks (alpha 1e10, i.e.
no scaling), merged at load. ComfyUI's recipe has no skip-layer trick, so the preset runs plain CFG.

Presets are listed by `oxinfer --models` (the catalog), as JSON from `oxinfer::advisor::presets_json()`
/ `oxi_presets_json()`, and taken by `--preset ID` or `--model ID`, the server (`--model kromoss-famegrid`
serves the base with `preset=kromoss-famegrid`; requests can override every setting) and the C API
(`oxi_moss_image_load("kromoss-famegrid", ...)`).

## Library, C API, server

```rust
let mut pipe = oxinfer::moss_image::Pipeline::load("kromoss-famegrid", Default::default())?;
let mut p = pipe.params("a candid portrait photo of a woman in a cafe"); // the preset's defaults
p.seed = 7;
let (img, timings) = pipe.generate(&p, &mut |i, n| eprint!("\r{i}/{n}"))?;
```

C: `oxi_moss_image_load(model_or_preset, "lora=...;quant=q8", ...)`, `oxi_moss_image_generate(q,
request_json, on_image, on_step, user)` (every `Params` field as JSON), `oxi_presets_json()`.

Server: `oxinfer serve --model kromoss-famegrid --model lodestones/Kromoss`. `/v1/images/generations`
takes `prompt`, `size`, `n`, `seed`, `steps`, `guidance`, `negative_prompt`, and the MOSS-Image fields
`sampler`, `scheduler`, `shift`, `uncond_skip_layers`, `slg_layers`, `slg_scale`, `slg_start`,
`slg_stop`, `trigger`, `noise`; load options (`lora`, `lora_merge`, `quant`, `preset`, ...) go in the model spec.

## Verification

Against the release's `pipeline_moss_image.py` (diffusers 0.44.0.dev0 from git, transformers 5.24,
torch 2.15) on the real Kromoss weights, prompt "a portrait photo of a woman in a red coat on a city
street at dusk", reference in float32 on the CPU unless said otherwise (`examples/moss_check.rs`
compares the components; reference dumps from a script in the session's scratchpad):

| What | oxinfer CPU | oxinfer Metal |
|---|---|---|
| prompt token ids (26) and the empty negative's (10) | identical | identical |
| Qwen3.5-9B features (bf16 weights) | rel L2 3.3e-6 / 6.3e-6 | 5.5e-4 / 6.3e-6 |
| one transformer pass at 1024x1024, t = 952: conditional / unconditional without block 22 | 2.6e-6 / 1.5e-6 | 3.9e-4 / 3.7e-4 |
| VAE decode of a 64x64x128 latent | 127.3 dB (rel 3.5e-6) | 102.1 dB |
| 10 steps, CFG 3, 512x512, same initial noise and conditioning, float32 reference | 75.2 dB (latent rel 1.5e-4) | |
| 50 steps, CFG 3, 1024x1024 against the official bf16 run on MPS (same noise / conditioning) | | 31.5 dB (same image; the reference's bf16 activations dominate) |
| DMD2 LoRA, 8 steps, CFG 1, against the reference with the README's forward hooks (bf16, MPS) | | 29.6 dB, same image |
| FameGrid LoKr at 1.4 merged into bf16 as ComfyUI does, 24 Euler steps, CFG 2.5, against the same in the reference | | 33.8 dB, same image |
| GGUF Q4_K_M transformer (molbal/Kromoss-GGUF), one pass against the float32 reference | | 6.6e-2 (Q4 weight noise; the 8-step DMD2 image keeps the subject and setting, its details move) |

ComfyUI's schedules and `er_sde` are checked in the unit tests against ComfyUI's own code (beta57,
beta and simple sigmas to 2e-6 with scipy's beta quantiles; the ER-SDE update on a toy model to
1e-4). With `noise=torch` the initial and step noise are ComfyUI's (on the CPU; see [Noise](#noise)).

FameGrid (the full stack, `--preset kromoss-famegrid`, seed 7, "famegrid, a candid portrait photo of a
young man reading a book in a sunlit cafe") against the same settings without the LoRA: the LoRA turns
the scene into a posed phone-camera portrait (the subject faces the camera, holds the book up, warmer
light, shallower depth), the social-media look it is trained for; without it Kromoss gives a
documentary shot looking down at the book.

## Speed

Apple M5 Max (18 CPU cores, 128 GB), 1024x1024, bf16 weights, seconds. A pass is one
transformer forward (4096 image tokens + the prompt); 50 steps with CFG are 100 passes, the 8-step
DMD presets 8.

| | per pass | 50 steps, CFG 3 | 8 steps, DMD2 | VAE decode |
|---|---|---|---|---|
| oxinfer Metal | 3.4-3.8 | 381 | 39 (4.9 per pass with the LoRA) | 6.6-7.4 |
| oxinfer Metal, GGUF Q4_K_M transformer | | | 47 (5.9 per pass) | |
| diffusers on MPS (bf16, the release's pipeline) | 2.43 | 244 | 23 (2.9 per pass with the hooks) | about 3 |
| oxinfer CPU | 32-34 | about 55 min | 58-60 per pass with the LoRA* | 12 |
| diffusers on the CPU, float32 | 121 | | | |

\* measured while other jobs kept the machine's load average at 60-80; the transformer pass alone
measured 32-34 s on a quieter machine.

Where a Metal pass goes (`examples/moss_check.rs` prints it): matmuls 2.7 s (about 14.5 TFLOPS
over the pass's 39 TFLOPs), attention 0.7 s, the SiTU feed-forward activation and rotary embeddings on
the CPU 0.4 s, norms 0.14 s, and 0.8 s of copies in and out of GPU buffers. diffusers on MPS is about
1.5 times faster per pass; closing that gap needs the elementwise work on the GPU and fewer round
trips. ComfyUI and stable-diffusion.cpp (whose `moss_image` support is in a fork) were not run here.
The text encoder takes 2-4 s per call (mapped Qwen3.5-9B, read again after the first call from the
page cache).

